#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
offline_run_calibration_gui_with_parametric_skeleton_parallel_tip.py

OFFLINE CTR shadow calibration runner with:
  1) GUI crop selection using the FIRST raw image
  2) GUI ruler 2-point scale selection (known distance in mm)
  3) analyze_data_batch + postprocess_calibration_data
  4) Parametric skeleton export:
      - A skeleton model defined as a small number of links (cylinders)
      - Link poses vary with B pull via polynomial equations derived from fitted r(b), z(b)
      - Export STL for reference pose and JSON/Python predictors
  5) Tip localization refiners:
      - edge_dt: walk outward along the current tip tangent until background
      - edge_grad: use grayscale gradient along current tip tangent
      - mainray: estimate a robust trunk tangent from the upstream skeleton path, then ray-cast
      - parallel_centerline: fit two parallel side-lines of the distal tube extremity using the
        CURRENT measured tip angle, build the center parallel line, and intersect it with the
        distal black/white boundary to get the tip.

Also annotates the analyzed matplotlib images with the selected tip geometry when available.

This script does NOT require editing shadow_calibration.py; it monkey-patches
cal.analyze_data() at runtime for offline debugging.
"""

import argparse
import json
import os
import shutil
import sys
import types
from heapq import heappush, heappop
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, List

import cv2
import numpy as np

# Add the path to your shadow_calibration script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from shadow_calibration import CTR_Shadow_Calibration  # noqa: E402
from shadow_calibration import _endpoints_8  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_NEI8_W = [
    (-1, -1, 2 ** 0.5), (-1, 0, 1.0), (-1, 1, 2 ** 0.5),
    (0, -1, 1.0),                             (0, 1, 1.0),
    (1, -1, 2 ** 0.5),  (1, 0, 1.0),  (1, 1, 2 ** 0.5),
]


# -----------------------------
# Utilities: IO and discovery
# -----------------------------
def list_images(folder: Path):
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    imgs.sort()
    return imgs


def ensure_project_from_raw(raw_dir: Path, project_dir: Path, link_mode: str = "symlink") -> Path:
    """
    Create a project_dir with raw_image_data_folder containing images from raw_dir.
    Returns raw_image_data_folder path.
    """
    project_dir.mkdir(parents=True, exist_ok=True)
    raw_out = project_dir / "raw_image_data_folder"
    raw_out.mkdir(parents=True, exist_ok=True)

    imgs = list_images(raw_dir)
    if not imgs:
        raise RuntimeError(f"No images found in raw_dir: {raw_dir}")

    for src in imgs:
        dst = raw_out / src.name
        if dst.exists():
            continue
        if link_mode == "symlink":
            try:
                os.symlink(src.resolve(), dst)
            except Exception:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)

    return raw_out


def _json_ready(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def _parse_inner_corners_arg(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])

    txt = str(value).strip().lower().replace("x", ",")
    parts = [p.strip() for p in txt.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected checkerboard inner corners as 'Nx,Ny' or 'NxXNy', got: {value}"
        )
    return int(parts[0]), int(parts[1])


def _undistort_points_to_image(points_xy: np.ndarray, camera_K: np.ndarray, camera_dist: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    undist = cv2.undistortPoints(pts, camera_K, camera_dist, P=camera_K)
    return undist.reshape(-1, 2)


def _scale_mm_from_px_homography(H: np.ndarray, mm_scale: float) -> np.ndarray:
    """
    If x_mm_old = H_old * x_px and new desired mm are x_mm_new = mm_scale * x_mm_old,
    then H_new = S * H_old with S = diag(mm_scale, mm_scale, 1).
    """
    H = np.asarray(H, dtype=np.float64).copy()
    S = np.diag([float(mm_scale), float(mm_scale), 1.0]).astype(np.float64)
    return S @ H


def _scale_px_from_mm_homography(H: np.ndarray, mm_scale: float) -> np.ndarray:
    """
    Inverse companion of _scale_mm_from_px_homography.
    If H_old maps old-mm -> px and new-mm = mm_scale * old-mm,
    then old-mm = new-mm / mm_scale, so H_new = H_old * S_inv.
    """
    H = np.asarray(H, dtype=np.float64).copy()
    if abs(float(mm_scale)) < 1e-12:
        raise ValueError("mm_scale must be non-zero.")
    S_inv = np.diag([1.0 / float(mm_scale), 1.0 / float(mm_scale), 1.0]).astype(np.float64)
    return H @ S_inv


def apply_checkerboard_reference_corrections(
    cal: CTR_Shadow_Calibration,
    mm_scale: float = 0.5,
    flip_planar_x: bool = True,
):
    """
    Correct checkerboard-backed calibration axes so they match the ruler-backed convention.

    Requested fixes:
      1) checkerboard mm values are 2x too large -> scale checkerboard mm conversion by 0.5
      2) checkerboard planar x sign is flipped vs motor -> negate planar x

    This patch is intentionally applied at the checkerboard reference layer so downstream
    post-processing uses the corrected convention without editing shadow_calibration.py.
    """
    if hasattr(cal, "apply_checkerboard_reference_corrections"):
        return cal.apply_checkerboard_reference_corrections(
            mm_scale=mm_scale,
            flip_planar_x=flip_planar_x,
        )

    if getattr(cal, "board_pose", None) is None:
        return cal

    correction_meta = {
        "checkerboard_mm_scale_correction": float(mm_scale),
        "checkerboard_planar_x_flipped": bool(flip_planar_x),
    }

    if getattr(cal, "board_mm_per_px_local", None) is not None:
        cal.board_mm_per_px_local = float(cal.board_mm_per_px_local) * float(mm_scale)
    if getattr(cal, "board_px_per_mm_local", None) is not None:
        if abs(float(mm_scale)) < 1e-12:
            raise ValueError("mm_scale must be non-zero.")
        cal.board_px_per_mm_local = float(cal.board_px_per_mm_local) / float(mm_scale)

    if getattr(cal, "board_homography_mm_from_px", None) is not None:
        cal.board_homography_mm_from_px = _scale_mm_from_px_homography(
            cal.board_homography_mm_from_px,
            mm_scale=float(mm_scale),
        )
    if getattr(cal, "board_homography_px_from_mm", None) is not None:
        cal.board_homography_px_from_mm = _scale_px_from_mm_homography(
            cal.board_homography_px_from_mm,
            mm_scale=float(mm_scale),
        )

    if isinstance(cal.board_pose, dict):
        bp = dict(cal.board_pose)
        if "board_homography_mm_from_px" in bp and bp["board_homography_mm_from_px"] is not None:
            bp["board_homography_mm_from_px"] = _scale_mm_from_px_homography(
                bp["board_homography_mm_from_px"], mm_scale=float(mm_scale)
            )
        if "board_homography_px_from_mm" in bp and bp["board_homography_px_from_mm"] is not None:
            bp["board_homography_px_from_mm"] = _scale_px_from_mm_homography(
                bp["board_homography_px_from_mm"], mm_scale=float(mm_scale)
            )
        if "board_mm_per_px_local" in bp and bp["board_mm_per_px_local"] is not None:
            bp["board_mm_per_px_local"] = float(bp["board_mm_per_px_local"]) * float(mm_scale)
        if "board_px_per_mm_local" in bp and bp["board_px_per_mm_local"] is not None:
            bp["board_px_per_mm_local"] = float(bp["board_px_per_mm_local"]) / float(mm_scale)
        bp.setdefault("corrections_applied", {})
        bp["corrections_applied"].update(correction_meta)
        cal.board_pose = bp

    cal.board_reference_correction_meta = correction_meta

    if hasattr(cal, "pixel_point_to_calibrated_axes"):
        original_pixel_point_to_calibrated_axes = cal.pixel_point_to_calibrated_axes

        def pixel_point_to_calibrated_axes_patched(self, *args, **kwargs):
            u_mm, z_mm = original_pixel_point_to_calibrated_axes(*args, **kwargs)
            u_mm = float(u_mm)
            z_mm = float(z_mm)
            if bool(flip_planar_x):
                u_mm = -u_mm
            return u_mm, z_mm

        cal.pixel_point_to_calibrated_axes = types.MethodType(
            pixel_point_to_calibrated_axes_patched, cal
        )

    if hasattr(cal, "calibrated_axes_to_pixel_point"):
        original_calibrated_axes_to_pixel_point = cal.calibrated_axes_to_pixel_point

        def calibrated_axes_to_pixel_point_patched(self, u_mm, z_mm, *args, **kwargs):
            u_in = -float(u_mm) if bool(flip_planar_x) else float(u_mm)
            return original_calibrated_axes_to_pixel_point(u_in, float(z_mm), *args, **kwargs)

        cal.calibrated_axes_to_pixel_point = types.MethodType(
            calibrated_axes_to_pixel_point_patched, cal
        )

    print(
        "[INFO] Applied checkerboard reference corrections: "
        f"mm_scale={float(mm_scale):.6f}, "
        f"flip_planar_x={bool(flip_planar_x)}"
    )
    return cal


def collect_board_reference_info(cal: CTR_Shadow_Calibration) -> Dict[str, Any]:
    return {
        "camera_calib_path": getattr(cal, "camera_calib_path", None),
        "camera_calib_meta": getattr(cal, "camera_calib_meta", None),
        "board_reference_image_path": getattr(cal, "board_reference_image_path", None),
        "board_pose": getattr(cal, "board_pose", None),
        "true_vertical_img_unit": getattr(cal, "true_vertical_img_unit", None),
        "board_homography_px_from_mm": getattr(cal, "board_homography_px_from_mm", None),
        "board_homography_mm_from_px": getattr(cal, "board_homography_mm_from_px", None),
        "board_px_per_mm_local": getattr(cal, "board_px_per_mm_local", None),
        "board_mm_per_px_local": getattr(cal, "board_mm_per_px_local", None),
        "board_reference_correction_meta": getattr(cal, "board_reference_correction_meta", None),
    }


def draw_checkerboard_analysis_overlay(
    cal: CTR_Shadow_Calibration,
    output_path: Path,
    tracked_rows: np.ndarray,
    image_files: List[Path],
    board_debug_image: np.ndarray,
):
    if board_debug_image is None:
        raise ValueError("board_debug_image is required.")
    if tracked_rows is None or tracked_rows.size == 0:
        raise ValueError("tracked_rows is empty; run analyze_data_batch first.")
    if getattr(cal, "board_pose", None) is None or getattr(cal, "true_vertical_img_unit", None) is None:
        raise RuntimeError("Board reference is unavailable; estimate checkerboard reference first.")

    valid_mask = np.all(np.isfinite(tracked_rows[:, :2]), axis=1)
    if tracked_rows.shape[1] > 3:
        valid_mask &= np.isfinite(tracked_rows[:, 3])
    valid_rows = tracked_rows[valid_mask]
    valid_files = [image_files[i] for i, ok in enumerate(valid_mask) if ok and i < len(image_files)]
    if valid_rows.size == 0:
        raise RuntimeError("No valid tracked points were found for annotation.")

    pts_xy = np.column_stack([valid_rows[:, 1], valid_rows[:, 0]]).astype(np.float64)
    if bool(cal.board_pose.get("use_undistort", False)):
        pts_xy_draw = _undistort_points_to_image(pts_xy, cal.camera_K, cal.camera_dist)
    else:
        pts_xy_draw = pts_xy.copy()

    overlay = board_debug_image.copy()
    h, w = overlay.shape[:2]
    origin = np.asarray(cal.board_pose["origin_px"], dtype=np.float64).reshape(2)
    v_hat = np.asarray(cal.true_vertical_img_unit, dtype=np.float64).reshape(2)
    v_hat /= max(np.linalg.norm(v_hat), 1e-12)
    u_hat = np.array([v_hat[1], -v_hat[0]], dtype=np.float64)

    arrow_len = int(max(60, min(h, w) * 0.14))
    origin_i = tuple(np.round(origin).astype(int))
    horiz_end = tuple(np.round(origin + u_hat * arrow_len).astype(int))
    vert_end = tuple(np.round(origin + v_hat * arrow_len).astype(int))
    cv2.arrowedLine(overlay, origin_i, horiz_end, (40, 80, 255), 3, tipLength=0.14)
    cv2.arrowedLine(overlay, origin_i, vert_end, (0, 220, 0), 3, tipLength=0.14)
    cv2.putText(overlay, "camera horizontal", (horiz_end[0] + 8, horiz_end[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 80, 255), 2)
    cv2.putText(overlay, "camera vertical", (vert_end[0] + 8, vert_end[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)

    for pt in pts_xy_draw:
        px = int(round(pt[0]))
        py = int(round(pt[1]))
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(overlay, (px, py), 4, (0, 255, 255), -1)
            cv2.circle(overlay, (px, py), 7, (0, 0, 0), 1)

    if pts_xy_draw.shape[0] >= 2:
        pts_poly = np.round(pts_xy_draw).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts_poly], False, (255, 200, 0), 1, lineType=cv2.LINE_AA)

    rep_indices = sorted(set(np.linspace(0, len(valid_rows) - 1, num=min(3, len(valid_rows)), dtype=int).tolist()))
    for label_idx, idx in enumerate(rep_indices, start=1):
        pt_draw = pts_xy_draw[idx]
        pt_measure = pt_draw
        u_mm, z_mm = cal.pixel_point_to_calibrated_axes(
            x_px=float(pt_measure[0]),
            y_px=float(pt_measure[1]),
            origin_px=origin,
        )
        delta = pt_draw - origin
        du_px = float(np.dot(delta, u_hat))
        dz_px = float(np.dot(delta, v_hat))
        corner = origin + du_px * u_hat

        p0 = tuple(np.round(origin).astype(int))
        p1 = tuple(np.round(corner).astype(int))
        p2 = tuple(np.round(pt_draw).astype(int))
        cv2.line(overlay, p0, p1, (40, 80, 255), 2, lineType=cv2.LINE_AA)
        cv2.line(overlay, p1, p2, (0, 220, 0), 2, lineType=cv2.LINE_AA)
        cv2.circle(overlay, p2, 7, (255, 255, 255), 2)

        b_pull = float(valid_rows[idx, 3]) if valid_rows.shape[1] > 3 else float("nan")
        file_label = valid_files[idx].stem if idx < len(valid_files) else f"pt_{idx}"
        label = f"P{label_idx}  B={b_pull:.2f}  H={u_mm:.2f}mm  V={z_mm:.2f}mm"
        label_pos = (p2[0] + 10, p2[1] - 10 - 18 * (label_idx - 1))
        cv2.putText(overlay, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 3)
        cv2.putText(overlay, label, label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, (10, 10, 10), 1)
        file_pos = (label_pos[0], label_pos[1] + 18)
        cv2.putText(overlay, file_label, file_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 2)
        cv2.putText(overlay, file_label, file_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.48, (20, 20, 20), 1)

    all_u_mm = []
    all_z_mm = []
    for pt in pts_xy_draw:
        uu, zz = cal.pixel_point_to_calibrated_axes(
            x_px=float(pt[0]),
            y_px=float(pt[1]),
            origin_px=origin,
        )
        all_u_mm.append(uu)
        all_z_mm.append(zz)

    u_span = float(np.max(all_u_mm) - np.min(all_u_mm))
    z_span = float(np.max(all_z_mm) - np.min(all_z_mm))
    summary_lines = [
        f"tracked points: {len(valid_rows)}",
        f"horizontal span: {u_span:.2f} mm",
        f"vertical span: {z_span:.2f} mm",
        f"local scale: {float(cal.board_px_per_mm_local):.4f} px/mm",
    ]
    if getattr(cal, "board_reference_correction_meta", None):
        cmeta = cal.board_reference_correction_meta
        summary_lines.append(f"checkerboard mm scale correction: {float(cmeta['checkerboard_mm_scale_correction']):.3f}")
        summary_lines.append(f"checkerboard x sign flipped: {bool(cmeta['checkerboard_planar_x_flipped'])}")

    y_text = 32
    for line in summary_lines:
        cv2.putText(overlay, line, (24, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 3)
        cv2.putText(overlay, line, (24, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 1)
        y_text += 30

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"Failed to write annotated checkerboard image: {output_path}")
    return output_path


# -------------------------------------------
# GUI: crop + ruler selection from one image
# -------------------------------------------
def interactive_crop_and_ruler_from_image(
    image_bgr: np.ndarray,
    default_crop=None,
    ruler_known_mm: float = 150.0,
    window_crop="Manual Crop Setup (OFFLINE)",
    window_ruler="Ruler Reference Setup (OFFLINE)",
):
    """
    Returns:
      analysis_crop dict (same format CTR_Shadow_Calibration expects)
      ruler dict with fields:
        p1_px, p2_px, mm_per_px, px_per_mm, axis_unit, axis_perp_unit, meta
      If ruler step skipped, ruler dict fields are None except meta.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image passed to interactive GUI.")

    img_h, img_w = image_bgr.shape[:2]

    if default_crop is None:
        default_crop = {
            "crop_width_min": 650,
            "crop_width_max": 2900,
            "crop_height_min": 150,
            "crop_height_max": 1750,
        }

    x_min = int(np.clip(default_crop["crop_width_min"], 0, img_w - 2))
    x_max = int(np.clip(default_crop["crop_width_max"], x_min + 1, img_w - 1))
    y_min = int(np.clip(img_h - default_crop["crop_height_max"], 0, img_h - 2))
    y_max = int(np.clip(img_h - default_crop["crop_height_min"], y_min + 1, img_h - 1))

    corners = {
        "tl": [x_min, y_min],
        "tr": [x_max, y_min],
        "br": [x_max, y_max],
        "bl": [x_min, y_max],
    }
    active_corner = {"name": None}
    drag_threshold_px = 30

    def nearest_corner(mx, my):
        best_name, best_dist = None, 1e18
        for name, (cx, cy) in corners.items():
            d = (mx - cx) ** 2 + (my - cy) ** 2
            if d < best_dist:
                best_dist = d
                best_name = name
        return best_name if best_dist <= drag_threshold_px ** 2 else None

    def clamp_rect():
        xs = [pt[0] for pt in corners.values()]
        ys = [pt[1] for pt in corners.values()]
        x0 = int(np.clip(min(xs), 0, img_w - 2))
        x1 = int(np.clip(max(xs), x0 + 1, img_w - 1))
        y0 = int(np.clip(min(ys), 0, img_h - 2))
        y1 = int(np.clip(max(ys), y0 + 1, img_h - 1))
        corners["tl"] = [x0, y0]
        corners["tr"] = [x1, y0]
        corners["br"] = [x1, y1]
        corners["bl"] = [x0, y1]

    def on_mouse_crop(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            active_corner["name"] = nearest_corner(mx, my)
        elif event == cv2.EVENT_MOUSEMOVE and active_corner["name"] is not None:
            name = active_corner["name"]
            mx = int(np.clip(mx, 0, img_w - 1))
            my = int(np.clip(my, 0, img_h - 1))
            if name == "tl":
                corners["tl"] = [mx, my]
                corners["tr"][1] = my
                corners["bl"][0] = mx
            elif name == "tr":
                corners["tr"] = [mx, my]
                corners["tl"][1] = my
                corners["br"][0] = mx
            elif name == "br":
                corners["br"] = [mx, my]
                corners["bl"][1] = my
                corners["tr"][0] = mx
            elif name == "bl":
                corners["bl"] = [mx, my]
                corners["br"][1] = my
                corners["tl"][0] = mx
            clamp_rect()
        elif event == cv2.EVENT_LBUTTONUP:
            active_corner["name"] = None

    accepted_crop = False
    cv2.namedWindow(window_crop, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_crop, on_mouse_crop)

    print("\n[GUI] Manual crop setup (OFFLINE)")
    print("- Drag rectangle corners with left mouse.")
    print("- Press ENTER or SPACE to confirm.")
    print("- Press R to reset to defaults.")
    print("- Press Q or ESC to cancel (uses defaults).")

    try:
        while True:
            if cv2.getWindowProperty(window_crop, cv2.WND_PROP_VISIBLE) < 1:
                break

            clamp_rect()
            x0, y0 = corners["tl"]
            x1, y1 = corners["br"]

            disp = image_bgr.copy()
            cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 2)
            for pt in corners.values():
                cv2.circle(disp, tuple(pt), 6, (0, 255, 255), -1)

            cv2.putText(
                disp,
                f"x:[{x0},{x1}] y:[{y0},{y1}]  (ENTER confirm | R reset | Q/ESC cancel)",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

            cv2.imshow(window_crop, disp)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 32):
                accepted_crop = True
                break
            if key in (27, ord("q")):
                break
            if key in (ord("r"), ord("R")):
                corners["tl"] = [x_min, y_min]
                corners["tr"] = [x_max, y_min]
                corners["br"] = [x_max, y_max]
                corners["bl"] = [x_min, y_max]
    finally:
        cv2.setMouseCallback(window_crop, lambda *args: None)
        cv2.destroyWindow(window_crop)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    if accepted_crop:
        x0, y0 = corners["tl"]
        x1, y1 = corners["br"]
        analysis_crop = {
            "crop_width_min": int(x0),
            "crop_width_max": int(x1),
            "crop_height_min": int(img_h - y1),
            "crop_height_max": int(img_h - y0),
        }
        print(f"[GUI] Selected analysis_crop: {analysis_crop}")
    else:
        analysis_crop = dict(default_crop)
        print(f"[GUI] Crop cancelled, using default analysis_crop: {analysis_crop}")

    ruler_points = []
    ruler_confirmed = False
    ruler_skipped = False
    min_valid_dist_px = 5.0

    def on_mouse_ruler(event, mx, my, flags, param):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(ruler_points) >= 2:
            return
        mx = int(np.clip(mx, 0, img_w - 1))
        my = int(np.clip(my, 0, img_h - 1))
        ruler_points.append((mx, my))

    print("\n[GUI] Ruler reference setup (OFFLINE)")
    print(f"- Click two points on the physical ruler (known distance = {ruler_known_mm:.1f} mm).")
    print("- Press ENTER or SPACE to confirm once two points are selected.")
    print("- Press R to reset ruler points.")
    print("- Press Q or ESC to skip ruler calibration.")

    cv2.namedWindow(window_ruler, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_ruler, on_mouse_ruler)

    try:
        while True:
            if cv2.getWindowProperty(window_ruler, cv2.WND_PROP_VISIBLE) < 1:
                ruler_skipped = True
                break

            disp = image_bgr.copy()

            x0 = analysis_crop["crop_width_min"]
            x1 = analysis_crop["crop_width_max"]
            y0 = img_h - analysis_crop["crop_height_max"]
            y1 = img_h - analysis_crop["crop_height_min"]
            cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 2)

            for idx, pt in enumerate(ruler_points):
                cv2.circle(disp, pt, 7, (0, 255, 255), -1)
                cv2.putText(
                    disp,
                    f"P{idx + 1}",
                    (pt[0] + 8, pt[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            if len(ruler_points) == 2:
                p1 = np.asarray(ruler_points[0], dtype=float)
                p2 = np.asarray(ruler_points[1], dtype=float)
                dist_px = float(np.linalg.norm(p2 - p1))
                mm_per_px = (ruler_known_mm / dist_px) if dist_px > 1e-9 else float("nan")
                cv2.line(disp, ruler_points[0], ruler_points[1], (50, 200, 255), 2)
                cv2.putText(
                    disp,
                    f"dist_px={dist_px:.2f}  mm/px={mm_per_px:.6f}",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.75,
                    (255, 255, 255),
                    2,
                )

            cv2.putText(
                disp,
                f"Pick 2 ruler points ({ruler_known_mm:.1f} mm): ENTER confirm | R reset | Q/ESC skip",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

            cv2.imshow(window_ruler, disp)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 32):
                if len(ruler_points) < 2:
                    print("[GUI] Select two points before confirming.")
                    continue
                p1 = np.asarray(ruler_points[0], dtype=float)
                p2 = np.asarray(ruler_points[1], dtype=float)
                dist_px = float(np.linalg.norm(p2 - p1))
                if dist_px < min_valid_dist_px:
                    print(f"[GUI] Points too close ({dist_px:.3f}px). Re-pick.")
                    continue
                ruler_confirmed = True
                break
            if key in (27, ord("q")):
                ruler_skipped = True
                break
            if key in (ord("r"), ord("R")):
                ruler_points.clear()
    finally:
        cv2.setMouseCallback(window_ruler, lambda *args: None)
        cv2.destroyWindow(window_ruler)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    ruler = {
        "p1_px": None,
        "p2_px": None,
        "known_distance_mm": float(ruler_known_mm),
        "mm_per_px": None,
        "px_per_mm": None,
        "axis_unit": None,
        "axis_perp_unit": None,
        "meta": None,
    }

    if ruler_confirmed:
        p1 = np.asarray(ruler_points[0], dtype=np.float64)
        p2 = np.asarray(ruler_points[1], dtype=np.float64)
        axis_vec = p2 - p1
        pixel_dist = float(np.linalg.norm(axis_vec))
        axis_unit = axis_vec / pixel_dist
        axis_perp_unit = np.array([axis_unit[1], -axis_unit[0]], dtype=np.float64)

        mm_per_px = float(ruler_known_mm / pixel_dist)
        px_per_mm = float(pixel_dist / ruler_known_mm)

        ruler["p1_px"] = (int(round(p1[0])), int(round(p1[1])))
        ruler["p2_px"] = (int(round(p2[0])), int(round(p2[1])))
        ruler["mm_per_px"] = mm_per_px
        ruler["px_per_mm"] = px_per_mm
        ruler["axis_unit"] = axis_unit.tolist()
        ruler["axis_perp_unit"] = axis_perp_unit.tolist()
        ruler["meta"] = {
            "source": "offline_gui",
            "known_distance_mm": float(ruler_known_mm),
            "pixel_distance": float(pixel_dist),
            "analysis_crop": dict(analysis_crop),
        }

        print("[GUI] Ruler scale set:")
        print(f"  p1_px={ruler['p1_px']}  p2_px={ruler['p2_px']}")
        print(
            f"  pixel_distance={pixel_dist:.6f}px  "
            f"mm_per_px={mm_per_px:.9f}  px_per_mm={px_per_mm:.9f}"
        )
    else:
        print("[GUI] Ruler calibration skipped.")
        ruler["meta"] = {"source": "offline_gui", "skipped": True, "analysis_crop": dict(analysis_crop)}

    return analysis_crop, ruler


# ------------------------------------------
# Tip refinement helpers
# ------------------------------------------
def _get_pos_from_file_name(file_name: str):
    """
    Support both legacy positional filenames and newer prefixed-token filenames.
    """
    base = os.path.splitext(os.path.basename(file_name))[0]
    parts = base.split("_")

    orientation = 0
    if len(parts) > 0:
        try:
            orientation = int(parts[0])
        except Exception:
            orientation = 0

    values = {}
    for p in parts:
        if len(p) >= 2 and p[0] in ("X", "Y", "Z", "B", "C"):
            try:
                values[p[0]] = float(p[1:])
            except Exception:
                pass

    if "X" in values and "B" in values:
        ntnl_pos = float(values["X"])
        ss_pos = float(values["B"])
        return orientation, ntnl_pos, ss_pos

    if len(parts) >= 3:
        try:
            ntnl_pos = float(parts[1])
            ss_pos = float(parts[2])
            return orientation, ntnl_pos, ss_pos
        except Exception:
            pass

    raise ValueError(
        "Could not parse X/B values from filename: "
        f"{file_name}. Expected either '<ori>_<X>_<B>_...' or tokens like 'X...' and 'B...'."
    )


def _tip_angle_to_direction_xy(tip_angle_deg: float) -> np.ndarray:
    ang = np.deg2rad(float(tip_angle_deg))
    vx = float(np.sin(ang))
    vy = float(np.cos(ang))
    d = np.array([vx, vy], dtype=np.float64)
    n = float(np.linalg.norm(d))
    return d / max(n, 1e-12)


def _inside_mask(mask_fg: np.ndarray, xy: np.ndarray) -> bool:
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    h, w = mask_fg.shape[:2]
    return (0 <= x < w) and (0 <= y < h) and (mask_fg[y, x] > 0)


def _backtrack_point_inside_fg(mask_fg: np.ndarray, p0_xy: np.ndarray, dir_xy: np.ndarray,
                               max_back_px: float = 80.0, step_px: float = 0.5):
    p0 = np.asarray(p0_xy, dtype=np.float64)
    d = np.asarray(dir_xy, dtype=np.float64)
    d /= max(np.linalg.norm(d), 1e-12)

    if _inside_mask(mask_fg, p0):
        return p0.copy(), True, 0.0

    n_steps = int(max_back_px / max(step_px, 1e-6))
    for i in range(1, n_steps + 1):
        q = p0 - i * step_px * d
        if _inside_mask(mask_fg, q):
            return q, True, i * step_px
    return p0.copy(), False, None


def ray_last_inside(mask_fg: np.ndarray, p0_xy: np.ndarray, dir_xy: np.ndarray,
                    step_px: float = 0.5, max_len_px: float = 120.0):
    h, w = mask_fg.shape[:2]
    d = np.asarray(dir_xy, dtype=np.float64)
    d /= max(np.linalg.norm(d), 1e-12)

    last_inside = np.asarray(p0_xy, dtype=np.float64).copy()
    n_steps = int(max_len_px / max(step_px, 1e-6))

    for i in range(1, n_steps + 1):
        p = p0_xy + i * step_px * d
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        if not (0 <= x < w and 0 <= y < h):
            break
        if mask_fg[y, x] == 1:
            last_inside = p
        else:
            break
    return last_inside


def _estimate_radius_along_axis(mask_fg: np.ndarray, dist_img: np.ndarray, p_in_xy: np.ndarray,
                                dir_xy: np.ndarray, back_len_px: float = 60.0, step_px: float = 1.0) -> float:
    d = np.asarray(dir_xy, dtype=np.float64)
    d /= max(np.linalg.norm(d), 1e-12)
    vals = []
    n_steps = int(back_len_px / max(step_px, 1e-6))
    for i in range(n_steps + 1):
        p = p_in_xy - i * step_px * d
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        if 0 <= x < mask_fg.shape[1] and 0 <= y < mask_fg.shape[0] and mask_fg[y, x] == 1:
            v = float(dist_img[y, x])
            if np.isfinite(v) and v > 0:
                vals.append(v)
    if not vals:
        x = int(round(float(p_in_xy[0])))
        y = int(round(float(p_in_xy[1])))
        if 0 <= x < dist_img.shape[1] and 0 <= y < dist_img.shape[0]:
            return max(3.0, float(dist_img[y, x]))
        return 6.0
    return max(3.0, float(np.median(vals)))


def _contiguous_runs_from_bool(mask_bool: np.ndarray):
    m = np.asarray(mask_bool, dtype=bool)
    edges = np.diff(np.r_[False, m, False].astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _cross_section_boundaries(mask_fg: np.ndarray, center_xy: np.ndarray, normal_xy: np.ndarray,
                              scan_half_width_px: float = 20.0, step_px: float = 0.5):
    n = np.asarray(normal_xy, dtype=np.float64)
    n /= max(np.linalg.norm(n), 1e-12)
    ts = np.arange(-scan_half_width_px, scan_half_width_px + 0.5 * step_px, step_px, dtype=np.float64)
    inside = np.zeros_like(ts, dtype=bool)

    h, w = mask_fg.shape[:2]
    for i, t in enumerate(ts):
        p = center_xy + t * n
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        inside[i] = (0 <= x < w) and (0 <= y < h) and (mask_fg[y, x] == 1)

    if not np.any(inside):
        return None

    runs = _contiguous_runs_from_bool(inside)
    if not runs:
        return None

    idx0 = int(np.argmin(np.abs(ts)))
    chosen = None
    for s0, s1 in runs:
        if s0 <= idx0 <= s1:
            chosen = (s0, s1)
            break

    if chosen is None:
        centers = [0.5 * (ts[s0] + ts[s1]) for s0, s1 in runs]
        j = int(np.argmin(np.abs(np.asarray(centers))))
        chosen = runs[j]

    s0, s1 = chosen
    t_left = float(ts[s0])
    t_right = float(ts[s1])
    p_left = center_xy + t_left * n
    p_right = center_xy + t_right * n

    return {
        "t_left": t_left,
        "t_right": t_right,
        "p_left_xy": p_left,
        "p_right_xy": p_right,
        "ts": ts,
        "inside": inside,
    }


def refine_tip_edge_distance_transform(
    binary_image: np.ndarray,
    tip_yx: Tuple[int, int],
    tip_angle_deg: float,
    max_step_px: int = 80,
    step_px: float = 1.0,
    exit_mode: str = "first_background",
):
    """
    Move the tip from skeleton distal point to the edge of the dark foreground by stepping
    along the distal tangent direction until leaving the foreground.

    binary_image: 0=foreground (tube), 255=background
    tip_yx: (y, x) tip on skeleton
    tip_angle_deg: signed angle relative to vertical-down
    Returns refined (y,x) float, plus debug dict.
    """
    h, w = binary_image.shape[:2]
    fg = (binary_image == 0).astype(np.uint8)

    d_xy = _tip_angle_to_direction_xy(tip_angle_deg)
    x0, y0 = float(tip_yx[1]), float(tip_yx[0])

    last_inside = np.array([x0, y0], dtype=np.float64)
    exited = False
    n_steps = int(max(1, max_step_px / max(step_px, 1e-6)))

    for i in range(1, n_steps + 1):
        p = np.array([x0, y0], dtype=np.float64) + i * step_px * d_xy
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        if not (0 <= x < w and 0 <= y < h):
            exited = True
            break
        if fg[y, x] == 1:
            last_inside = p
            continue
        exited = True
        if exit_mode == "first_background":
            break

    dbg = {
        "mode": "edge_dt",
        "d_xy": d_xy.tolist(),
        "exited": exited,
        "last_inside_xy": last_inside.tolist(),
        "tip_xy": last_inside.tolist(),
        "center_line_xy": [np.array([x0, y0]).tolist(), last_inside.tolist()],
    }
    return float(last_inside[1]), float(last_inside[0]), dbg


def refine_tip_edge_subpixel_gradient(
    grayscale: np.ndarray,
    binary_image: np.ndarray,
    tip_yx: Tuple[int, int],
    tip_angle_deg: float,
    search_len_px: int = 60,
    step_px: float = 0.25,
):
    """
    Subpixel-ish edge refinement along tangent:
    - Sample grayscale along outward direction from tip
    - Find the maximum absolute gradient location (dark->light edge)
    - Return point at that location (bounded by leaving the foreground)
    """
    h, w = grayscale.shape[:2]
    fg = (binary_image == 0).astype(np.uint8)

    d_xy = _tip_angle_to_direction_xy(tip_angle_deg)
    x0, y0 = float(tip_yx[1]), float(tip_yx[0])
    n = int(max(5, search_len_px / max(step_px, 1e-6)))

    samples = []
    coords = []
    inside_mask = []

    for i in range(n + 1):
        p = np.array([x0, y0], dtype=np.float64) + i * step_px * d_xy
        x = int(np.clip(round(float(p[0])), 0, w - 1))
        y = int(np.clip(round(float(p[1])), 0, h - 1))
        samples.append(float(grayscale[y, x]))
        coords.append(p.copy())
        inside_mask.append(int(fg[y, x] == 1))

    s = np.array(samples, dtype=float)
    g = np.diff(s)
    g_abs = np.abs(g)

    valid = [i for i in range(len(g_abs)) if inside_mask[i] == 1]
    if not valid:
        yy, xx, dbg = refine_tip_edge_distance_transform(binary_image, tip_yx, tip_angle_deg)
        dbg["fallback"] = "edge_dt"
        return yy, xx, dbg

    valid = np.array(valid, dtype=int)
    j = int(valid[np.argmax(g_abs[valid])])
    p_edge = 0.5 * (coords[j] + coords[j + 1])

    dbg = {
        "mode": "edge_grad",
        "d_xy": d_xy.tolist(),
        "j": j,
        "g_max": float(g_abs[j]),
        "step_px": float(step_px),
        "tip_xy": p_edge.tolist(),
        "center_line_xy": [np.array([x0, y0]).tolist(), p_edge.tolist()],
    }
    return float(p_edge[1]), float(p_edge[0]), dbg


def skeletonize_mask(mask_fg: np.ndarray) -> np.ndarray:
    """
    mask_fg: uint8, 1=foreground, 0=background
    returns: uint8, 1=skeleton, 0=background
    """
    try:
        from skimage.morphology import skeletonize
        skel = skeletonize(mask_fg.astype(bool))
        return skel.astype(np.uint8)
    except Exception:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            skel = cv2.ximgproc.thinning((mask_fg * 255).astype(np.uint8))
            return (skel > 0).astype(np.uint8)
        raise RuntimeError(
            "Need either scikit-image (preferred) or cv2.ximgproc.thinning for skeletonization."
        )


def skeleton_degree_image(skel: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), dtype=np.uint8)
    k[1, 1] = 0
    deg = cv2.filter2D(skel.astype(np.uint8), cv2.CV_16U, k)
    return deg


def build_skeleton_graph(skel: np.ndarray):
    pts = np.argwhere(skel > 0)
    pointset = {tuple(p) for p in pts}
    adj = {}
    for y, x in pointset:
        nbrs = []
        for dy, dx, w in _NEI8_W:
            q = (y + dy, x + dx)
            if q in pointset:
                nbrs.append((q, w))
        adj[(y, x)] = nbrs
    return adj


def dijkstra_skeleton(adj, src):
    dist = {src: 0.0}
    prev = {}
    pq = [(0.0, src)]
    visited = set()

    while pq:
        d, u = heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heappush(pq, (nd, v))
    return dist, prev


def reconstruct_path(prev, dst):
    path = [dst]
    cur = dst
    while cur in prev:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path


def path_to_xy(path_yx):
    return np.array([[x, y] for y, x in path_yx], dtype=np.float64)


def cumulative_arclength(xy: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(xy) == 1:
        return np.array([0.0], dtype=np.float64)
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(ds)])


def interp_point_on_path(xy: np.ndarray, s: np.ndarray, target_s: float) -> np.ndarray:
    target_s = float(np.clip(target_s, s[0], s[-1]))
    x = np.interp(target_s, s, xy[:, 0])
    y = np.interp(target_s, s, xy[:, 1])
    return np.array([x, y], dtype=np.float64)


def robust_tangent_from_path_window(xy: np.ndarray, s: np.ndarray, s0: float, s1: float) -> np.ndarray:
    mask = (s >= s0) & (s <= s1)
    pts = xy[mask]

    if pts.shape[0] < 5:
        i1 = max(0, len(xy) - 15)
        i2 = max(i1 + 2, len(xy) - 3)
        pts = xy[i1:i2]

    if pts.shape[0] < 2:
        v = xy[-1] - xy[max(0, len(xy) - 2)]
        n = np.linalg.norm(v)
        return v / max(n, 1e-9)

    mu = pts.mean(axis=0)
    uu, ss, vh = np.linalg.svd(pts - mu, full_matrices=False)
    t = vh[0]
    if np.dot(t, xy[-1] - xy[0]) < 0:
        t = -t
    n = np.linalg.norm(t)
    return t / max(n, 1e-9)


def refine_tip_edge_mainray(
    grayscale: np.ndarray,
    binary_image: np.ndarray,
    tip_yx: tuple,
    fit_back_near_r: float = 1.5,
    fit_back_far_r: float = 6.0,
    anchor_back_r: float = 1.0,
    ray_step_px: float = 0.5,
    ray_max_len_r: float = 8.0,
):
    """
    Robust coarse-tip estimator:
      - skeletonize foreground
      - choose endpoint nearest current coarse tip
      - estimate tangent from upstream path window
      - march a straight ray to the foreground edge

    Returns (y, x, debug).
    """
    mask_fg = (binary_image == 0).astype(np.uint8)
    if mask_fg.sum() == 0:
        return float(tip_yx[0]), float(tip_yx[1]), {"mode": "mainray", "fallback": "empty_fg"}

    skel = skeletonize_mask(mask_fg)
    deg = skeleton_degree_image(skel)
    endpoints = np.argwhere((skel > 0) & (deg == 1))
    if len(endpoints) < 2:
        return float(tip_yx[0]), float(tip_yx[1]), {"mode": "mainray", "fallback": "not_enough_endpoints"}

    tip_arr = np.asarray(tip_yx, dtype=np.float64)
    d2 = np.sum((endpoints.astype(np.float64) - tip_arr[None, :]) ** 2, axis=1)
    distal_ep = tuple(endpoints[int(np.argmin(d2))])

    adj = build_skeleton_graph(skel)
    dist_map, prev = dijkstra_skeleton(adj, distal_ep)
    endpoint_tuples = [tuple(p) for p in endpoints]
    proximal_ep = max(endpoint_tuples, key=lambda p: dist_map.get(p, -1.0))

    path_yx = reconstruct_path(prev, proximal_ep)
    if len(path_yx) < 5:
        return float(tip_yx[0]), float(tip_yx[1]), {"mode": "mainray", "fallback": "short_path"}

    path_xy = path_to_xy(path_yx)
    s = cumulative_arclength(path_xy)
    s_end = float(s[-1])

    dist_img = cv2.distanceTransform(mask_fg, cv2.DIST_L2, 5)
    tail = path_yx[max(0, len(path_yx) - 20):max(1, len(path_yx) - 3)]
    r_vals = [float(dist_img[y, x]) for (y, x) in tail if 0 <= y < dist_img.shape[0] and 0 <= x < dist_img.shape[1]]
    r_px = float(np.median(r_vals)) if len(r_vals) else float(dist_img[distal_ep[0], distal_ep[1]])
    r_px = max(r_px, 3.0)

    fit_near = fit_back_near_r * r_px
    fit_far = fit_back_far_r * r_px
    anchor_back = anchor_back_r * r_px
    ray_max_len = ray_max_len_r * r_px

    s0 = max(0.0, s_end - fit_far)
    s1 = max(0.0, s_end - fit_near)

    tangent_xy = robust_tangent_from_path_window(path_xy, s, s0, s1)
    anchor_xy = interp_point_on_path(path_xy, s, max(0.0, s_end - anchor_back))
    edge_xy = ray_last_inside(mask_fg, anchor_xy, tangent_xy, step_px=ray_step_px, max_len_px=ray_max_len)

    dbg = {
        "mode": "mainray",
        "distal_endpoint_yx": list(distal_ep),
        "proximal_endpoint_yx": list(proximal_ep),
        "radius_px": r_px,
        "fit_window_px": [fit_near, fit_far],
        "anchor_xy": anchor_xy.tolist(),
        "tangent_xy": tangent_xy.tolist(),
        "tip_xy": edge_xy.tolist(),
        "path_len_px": s_end,
        "center_line_xy": [anchor_xy.tolist(), edge_xy.tolist()],
        "path_window_xy": path_xy[(s >= s0) & (s <= s1)].tolist(),
    }
    return float(edge_xy[1]), float(edge_xy[0]), dbg


def refine_tip_parallel_centerline(
    grayscale: np.ndarray,
    binary_image: np.ndarray,
    tip_yx: Tuple[int, int],
    tip_angle_deg: float,
    section_near_r: float = 1.0,
    section_far_r: float = 6.0,
    scan_half_r: float = 3.0,
    num_sections: int = 9,
    cross_step_px: float = 0.5,
    ray_step_px: float = 0.5,
    ray_max_len_r: float = 8.0,
):
    """
    Distal tube extremity via two parallel side-lines constrained to the CURRENT measured tip angle.

    Procedure:
      1) Use the current tip angle as the common line direction.
      2) Backtrack from the coarse tip until inside the foreground.
      3) Sample several cross-sections upstream, orthogonal to the line direction.
      4) Estimate the two side boundaries as robust constant offsets (parallel lines).
      5) Build the center parallel line and ray-cast it distally to the black/white boundary.

    Returns: refined (y, x) and debug dict with annotation geometry.
    """
    mask_fg = (binary_image == 0).astype(np.uint8)
    if mask_fg.sum() == 0:
        return float(tip_yx[0]), float(tip_yx[1]), {"mode": "parallel_centerline", "fallback": "empty_fg"}

    d_xy = _tip_angle_to_direction_xy(tip_angle_deg)
    n_xy = np.array([-d_xy[1], d_xy[0]], dtype=np.float64)
    n_xy /= max(np.linalg.norm(n_xy), 1e-12)

    p_guess_xy = np.array([float(tip_yx[1]), float(tip_yx[0])], dtype=np.float64)
    p_in_xy, found_inside, back_dist = _backtrack_point_inside_fg(mask_fg, p_guess_xy, d_xy, max_back_px=120.0, step_px=0.5)
    if not found_inside:
        return float(tip_yx[0]), float(tip_yx[1]), {
            "mode": "parallel_centerline",
            "fallback": "could_not_backtrack_inside",
        }

    dist_img = cv2.distanceTransform(mask_fg, cv2.DIST_L2, 5)
    r_px = _estimate_radius_along_axis(mask_fg, dist_img, p_in_xy, d_xy, back_len_px=80.0, step_px=1.0)

    section_near_px = max(0.5 * r_px, float(section_near_r) * r_px)
    section_far_px = max(section_near_px + 2.0, float(section_far_r) * r_px)
    scan_half_px = max(2.5 * r_px, float(scan_half_r) * r_px)
    ray_max_len_px = max(20.0, float(ray_max_len_r) * r_px)

    s_samples = np.linspace(section_near_px, section_far_px, int(max(3, num_sections)))
    left_offsets = []
    right_offsets = []
    section_centers = []
    left_points = []
    right_points = []

    for s_back in s_samples:
        c_xy = p_in_xy - s_back * d_xy
        res = _cross_section_boundaries(
            mask_fg,
            center_xy=c_xy,
            normal_xy=n_xy,
            scan_half_width_px=scan_half_px,
            step_px=float(cross_step_px),
        )
        if res is None:
            continue
        t_left = float(res["t_left"])
        t_right = float(res["t_right"])
        if t_left > t_right:
            t_left, t_right = t_right, t_left
        left_offsets.append(t_left)
        right_offsets.append(t_right)
        section_centers.append(c_xy.tolist())
        left_points.append(np.asarray(res["p_left_xy"], dtype=np.float64).tolist())
        right_points.append(np.asarray(res["p_right_xy"], dtype=np.float64).tolist())

    if len(left_offsets) < 2 or len(right_offsets) < 2:
        return float(tip_yx[0]), float(tip_yx[1]), {
            "mode": "parallel_centerline",
            "fallback": "insufficient_cross_sections",
            "radius_px": r_px,
        }

    t_left_med = float(np.median(left_offsets))
    t_right_med = float(np.median(right_offsets))
    if t_left_med > t_right_med:
        t_left_med, t_right_med = t_right_med, t_left_med
    t_center = 0.5 * (t_left_med + t_right_med)
    width_px = float(t_right_med - t_left_med)

    line_back_px = section_far_px + 0.75 * r_px
    base_center_xy = p_in_xy - line_back_px * d_xy

    left_line_start = base_center_xy + t_left_med * n_xy
    right_line_start = base_center_xy + t_right_med * n_xy
    center_line_start = base_center_xy + t_center * n_xy

    if not _inside_mask(mask_fg, center_line_start):
        center_line_start, ok_center, _ = _backtrack_point_inside_fg(mask_fg, center_line_start, d_xy, max_back_px=3.0 * r_px, step_px=0.5)
        if not ok_center:
            center_line_start = p_in_xy + t_center * n_xy

    tip_xy = ray_last_inside(mask_fg, center_line_start, d_xy, step_px=float(ray_step_px), max_len_px=ray_max_len_px + line_back_px)

    line_forward_px = float(np.linalg.norm(tip_xy - base_center_xy)) + 0.5 * r_px
    left_line_end = base_center_xy + line_forward_px * d_xy + t_left_med * n_xy
    right_line_end = base_center_xy + line_forward_px * d_xy + t_right_med * n_xy
    center_line_end = base_center_xy + line_forward_px * d_xy + t_center * n_xy

    dbg = {
        "mode": "parallel_centerline",
        "tip_angle_deg": float(tip_angle_deg),
        "d_xy": d_xy.tolist(),
        "n_xy": n_xy.tolist(),
        "tip_guess_xy": p_guess_xy.tolist(),
        "inside_anchor_xy": p_in_xy.tolist(),
        "backtrack_dist_px": None if back_dist is None else float(back_dist),
        "radius_px": float(r_px),
        "width_px": width_px,
        "left_offset_px": float(t_left_med),
        "right_offset_px": float(t_right_med),
        "center_offset_px": float(t_center),
        "section_centers_xy": section_centers,
        "section_left_points_xy": left_points,
        "section_right_points_xy": right_points,
        "parallel_left_line_xy": [left_line_start.tolist(), left_line_end.tolist()],
        "parallel_right_line_xy": [right_line_start.tolist(), right_line_end.tolist()],
        "center_line_xy": [center_line_start.tolist(), tip_xy.tolist()],
        "center_line_full_xy": [center_line_start.tolist(), center_line_end.tolist()],
        "tip_xy": tip_xy.tolist(),
    }
    return float(tip_xy[1]), float(tip_xy[0]), dbg


def annotate_tip_geometry_on_axes(axs, dbg: Dict[str, Any], title_suffix: str = ""):
    """
    Add overlays to the analysis matplotlib axes returned by the original pipeline.
    Geometry is assumed to be in local crop coordinates.
    """
    if axs is None or not isinstance(dbg, dict):
        return

    try:
        if isinstance(axs, np.ndarray):
            target_axes = [axs.flat[-1]]
            if axs.size >= 3:
                target_axes.append(axs.flat[-2])
        elif isinstance(axs, (list, tuple)):
            target_axes = [axs[-1]]
        else:
            target_axes = [axs]

        used_labels = set()
        mode = str(dbg.get("mode", "tip_refine"))
        tip_xy = np.asarray(dbg.get("tip_xy", []), dtype=float).reshape(-1)

        for ax in target_axes:
            if ax is None:
                continue

            def _plot_line(key, color, label):
                line = dbg.get(key)
                if line is None:
                    return
                arr = np.asarray(line, dtype=float).reshape(-1, 2)
                if arr.shape[0] < 2:
                    return
                line_label = label if label not in used_labels else None
                if line_label is not None:
                    used_labels.add(label)
                ax.plot(arr[:, 0], arr[:, 1], color=color, linewidth=2.0, label=line_label)

            _plot_line("parallel_left_line_xy", "#00ff66", "tube side lines")
            _plot_line("parallel_right_line_xy", "#00ff66", "tube side lines")
            _plot_line("center_line_xy", "#ffd400", "center parallel line")
            _plot_line("center_line_full_xy", "#ffaa00", "center line (full)")

            if "path_window_xy" in dbg:
                pts = np.asarray(dbg["path_window_xy"], dtype=float).reshape(-1, 2)
                if pts.size > 0:
                    lbl = "tangent fit window" if "tangent fit window" not in used_labels else None
                    if lbl is not None:
                        used_labels.add(lbl)
                    ax.plot(pts[:, 0], pts[:, 1], color="#00e5ff", linewidth=2.0, label=lbl)

            if "section_left_points_xy" in dbg and len(dbg["section_left_points_xy"]) > 0:
                pts = np.asarray(dbg["section_left_points_xy"], dtype=float).reshape(-1, 2)
                lbl = "sampled edge points" if "sampled edge points" not in used_labels else None
                if lbl is not None:
                    used_labels.add(lbl)
                ax.scatter(pts[:, 0], pts[:, 1], s=10, c="#44ff44", label=lbl)

            if "section_right_points_xy" in dbg and len(dbg["section_right_points_xy"]) > 0:
                pts = np.asarray(dbg["section_right_points_xy"], dtype=float).reshape(-1, 2)
                ax.scatter(pts[:, 0], pts[:, 1], s=10, c="#44ff44")

            if "inside_anchor_xy" in dbg:
                p = np.asarray(dbg["inside_anchor_xy"], dtype=float).reshape(-1)
                lbl = "inside anchor" if "inside anchor" not in used_labels else None
                if lbl is not None:
                    used_labels.add(lbl)
                ax.scatter([p[0]], [p[1]], s=40, c="#ffffff", edgecolors="#000000", label=lbl, zorder=5)

            if tip_xy.size == 2 and np.all(np.isfinite(tip_xy)):
                lbl = f"refined tip ({mode})" if f"refined tip ({mode})" not in used_labels else None
                if lbl is not None:
                    used_labels.add(lbl)
                ax.scatter([tip_xy[0]], [tip_xy[1]], s=55, c="#ff3b30", edgecolors="#ffffff", label=lbl, zorder=6)
                ax.annotate(
                    f"tip{title_suffix}",
                    (tip_xy[0], tip_xy[1]),
                    xytext=(6, -8),
                    textcoords="offset points",
                    color="white",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55),
                )

            try:
                ax.legend(loc="best", fontsize=8)
            except Exception:
                pass
    except Exception as e:
        print(f"[WARN] Failed to annotate analysis axes: {e}")


def _remap_zoom_axes_to_crop_coordinates(axs, zoom_x_min: int, zoom_x_max: int, zoom_y_min: int, zoom_y_max: int):
    """
    Re-express the lower zoom panels in the same cropped-image pixel coordinates
    used by the refinement geometry so overlays land on the underlying image.
    """
    if axs is None or not isinstance(axs, np.ndarray) or axs.size < 4:
        return

    zoom_axes = [axs[1, 0], axs[1, 1]]
    x_offset = float(zoom_x_min)
    y_offset = float(zoom_y_min)
    extent = [float(zoom_x_min), float(zoom_x_max + 1), float(zoom_y_max + 1), float(zoom_y_min)]

    for ax in zoom_axes:
        if ax is None:
            continue

        for image in ax.images:
            image.set_extent(extent)

        for coll in ax.collections:
            try:
                offsets = coll.get_offsets()
            except Exception:
                continue
            if offsets is None:
                continue
            offsets = np.asarray(offsets, dtype=float)
            if offsets.size == 0:
                continue
            shifted = offsets.copy()
            shifted[:, 0] += x_offset
            shifted[:, 1] += y_offset
            coll.set_offsets(shifted)

        for line in ax.lines:
            try:
                xdata = np.asarray(line.get_xdata(), dtype=float)
                ydata = np.asarray(line.get_ydata(), dtype=float)
            except Exception:
                continue
            if xdata.size == 0 or ydata.size == 0:
                continue
            line.set_xdata(xdata + x_offset)
            line.set_ydata(ydata + y_offset)

        ax.set_xlim(float(zoom_x_min), float(zoom_x_max + 1))
        ax.set_ylim(float(zoom_y_max + 1), float(zoom_y_min))
        ax.set_aspect("equal")


# ---------------------------------------------------------
# Monkey-patch analyze_data to output refined tip coordinates
# ---------------------------------------------------------
def patch_analyze_data_for_tip_refinement(
    cal: CTR_Shadow_Calibration,
    refine_mode: str = "none",
    dt_step_px: float = 1.0,
    dt_max_step_px: int = 80,
    grad_step_px: float = 0.25,
    grad_search_len_px: int = 60,
    mainray_fit_back_near_r: float = 1.5,
    mainray_fit_back_far_r: float = 6.0,
    mainray_anchor_back_r: float = 1.0,
    mainray_ray_step_px: float = 0.5,
    mainray_ray_max_len_r: float = 8.0,
    parallel_section_near_r: float = 1.0,
    parallel_section_far_r: float = 6.0,
    parallel_scan_half_r: float = 3.0,
    parallel_num_sections: int = 9,
    parallel_cross_step_px: float = 0.5,
    parallel_ray_step_px: float = 0.5,
    parallel_ray_max_len_r: float = 8.0,
):
    """
    Patches cal.analyze_data(image_file_name, ...) so that:
      - It still uses the original skeleton + angle pipeline
      - It refines tip_row/tip_col to a distal edge estimate (optional)
      - It writes refined pixel coordinates into tip_locations_array_coarse
      - It annotates the returned matplotlib axes with the fitted geometry when available

    refine_mode:
      - "none"
      - "edge_dt"
      - "edge_grad"
      - "mainray"
      - "parallel_centerline"
    """
    original_analyze_data = cal.analyze_data
    if not hasattr(cal, "tip_refine_debug_records"):
        cal.tip_refine_debug_records = {}

    def analyze_data_patched(
        image_file_name,
        crop_width_min=None,
        crop_width_max=None,
        crop_height_min=None,
        crop_height_max=None,
        threshold=200,
    ):
        fig, axs, coarse_row, fine_row = original_analyze_data(
            image_file_name,
            crop_width_min,
            crop_width_max,
            crop_height_min,
            crop_height_max,
            threshold,
        )

        raw_data_folder = Path(cal.calibration_data_folder) / "raw_image_data_folder"
        img = cv2.imread(str(raw_data_folder / image_file_name), cv2.IMREAD_COLOR)
        if img is None:
            return fig, axs, coarse_row, fine_row

        if crop_width_min is None:
            crop_width_min = cal.analysis_crop["crop_width_min"]
        if crop_width_max is None:
            crop_width_max = cal.analysis_crop["crop_width_max"]
        if crop_height_min is None:
            crop_height_min = cal.analysis_crop["crop_height_min"]
        if crop_height_max is None:
            crop_height_max = cal.analysis_crop["crop_height_max"]

        crop_x_min_img = int(crop_width_min)
        crop_x_max_img = int(crop_width_max)
        crop_y_min_img = int(img.shape[0] - crop_height_max)
        crop_y_max_img = int(img.shape[0] - crop_height_min)

        cropped = img[crop_y_min_img:crop_y_max_img, crop_x_min_img:crop_x_max_img, :]
        if cropped.size == 0:
            return fig, axs, coarse_row, fine_row

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)

        tip_y_full = float(coarse_row[0])
        tip_x_full = float(coarse_row[1])
        tip_angle_deg = float(coarse_row[5])

        tip_y = tip_y_full - crop_y_min_img
        tip_x = tip_x_full - crop_x_min_img
        tip_y = float(np.clip(tip_y, 0, binary.shape[0] - 1))
        tip_x = float(np.clip(tip_x, 0, binary.shape[1] - 1))

        zoom_x_min = int(max(int(round(tip_x)) - 75, 0))
        zoom_x_max = int(min(int(round(tip_x)) + 75, binary.shape[1] - 1))
        zoom_y_min = int(max(int(round(tip_y)) - 75, 0))
        zoom_y_max = int(min(int(round(tip_y)) + 75, binary.shape[0] - 1))

        if refine_mode == "edge_dt":
            yy, xx, _dbg = refine_tip_edge_distance_transform(
                binary_image=binary,
                tip_yx=(int(round(tip_y)), int(round(tip_x))),
                tip_angle_deg=tip_angle_deg,
                max_step_px=int(dt_max_step_px),
                step_px=float(dt_step_px),
            )
        elif refine_mode == "edge_grad":
            yy, xx, _dbg = refine_tip_edge_subpixel_gradient(
                grayscale=gray,
                binary_image=binary,
                tip_yx=(int(round(tip_y)), int(round(tip_x))),
                tip_angle_deg=tip_angle_deg,
                search_len_px=int(grad_search_len_px),
                step_px=float(grad_step_px),
            )
        elif refine_mode == "mainray":
            yy, xx, _dbg = refine_tip_edge_mainray(
                grayscale=gray,
                binary_image=binary,
                tip_yx=(int(round(tip_y)), int(round(tip_x))),
                fit_back_near_r=float(mainray_fit_back_near_r),
                fit_back_far_r=float(mainray_fit_back_far_r),
                anchor_back_r=float(mainray_anchor_back_r),
                ray_step_px=float(mainray_ray_step_px),
                ray_max_len_r=float(mainray_ray_max_len_r),
            )
        elif refine_mode == "parallel_centerline":
            yy, xx, _dbg = refine_tip_parallel_centerline(
                grayscale=gray,
                binary_image=binary,
                tip_yx=(int(round(tip_y)), int(round(tip_x))),
                tip_angle_deg=tip_angle_deg,
                section_near_r=float(parallel_section_near_r),
                section_far_r=float(parallel_section_far_r),
                scan_half_r=float(parallel_scan_half_r),
                num_sections=int(parallel_num_sections),
                cross_step_px=float(parallel_cross_step_px),
                ray_step_px=float(parallel_ray_step_px),
                ray_max_len_r=float(parallel_ray_max_len_r),
            )
        else:
            yy, xx, _dbg = tip_y, tip_x, {"mode": "none", "tip_xy": [tip_x, tip_y]}

        coarse_row_refined = np.array(coarse_row, dtype=float).copy()
        coarse_row_refined[0] = yy + crop_y_min_img
        coarse_row_refined[1] = xx + crop_x_min_img

        dbg_local = dict(_dbg) if isinstance(_dbg, dict) else {}
        dbg_local["image_file_name"] = image_file_name
        dbg_local["tip_angle_deg"] = tip_angle_deg
        dbg_local["coarse_tip_before_local_xy"] = [float(tip_x), float(tip_y)]
        dbg_local["coarse_tip_after_local_xy"] = [float(xx), float(yy)]
        dbg_local["crop_origin_xy"] = [int(crop_x_min_img), int(crop_y_min_img)]
        cal.tip_refine_debug_records[image_file_name] = _json_ready(dbg_local)

        _remap_zoom_axes_to_crop_coordinates(axs, zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max)
        annotate_tip_geometry_on_axes(axs, dbg_local, title_suffix=f" ({refine_mode})")
        return fig, axs, coarse_row_refined, fine_row

    cal.analyze_data = analyze_data_patched
    return cal


# -----------------------------------------
# Parametric skeleton export (equations)
# -----------------------------------------
def _polyval(coeffs, b):
    return np.polyval(np.asarray(coeffs, dtype=float), np.asarray(b, dtype=float))


def _sample_curve_xyz(cal_json: dict, b_vals: np.ndarray):
    coeffs = cal_json.get("cubic_coefficients", {})
    r_coeffs = coeffs.get("r_coeffs")
    z_coeffs = coeffs.get("z_coeffs")
    y_coeffs = coeffs.get("offplane_y_coeffs")

    if r_coeffs is None or z_coeffs is None:
        raise ValueError("Missing r_coeffs/z_coeffs in calibration JSON.")

    x = _polyval(r_coeffs, b_vals)
    z = _polyval(z_coeffs, b_vals)
    if y_coeffs is not None:
        y = _polyval(y_coeffs, b_vals)
    else:
        y = np.zeros_like(x)

    return np.column_stack([x, y, z]).astype(float)


def _downsample_polyline(points: np.ndarray, n_links: int):
    n_links = int(max(1, n_links))
    k = n_links + 1
    idx = np.linspace(0, points.shape[0] - 1, k).round().astype(int)
    return points[idx]


def _compute_link_frames(points_xyz: np.ndarray):
    """
    For link i between P[i] -> P[i+1]:
      - origin at P[i]
      - z-axis along direction
      - x/y axes arbitrary but stable (constructed from world up fallback)
    Returns list of per-link dicts.
    """
    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    frames = []
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)

    for i in range(pts.shape[0] - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            frames.append({
                "origin_mm": p0.tolist(),
                "direction_unit": [0.0, 0.0, 1.0],
                "length_mm": 0.0,
                "R_world_from_link": np.eye(3).tolist(),
            })
            continue

        z = d / L

        up = world_up
        if abs(float(np.dot(up, z))) > 0.95:
            up = np.array([0.0, 1.0, 0.0], dtype=float)

        x = np.cross(up, z)
        xn = float(np.linalg.norm(x))
        if xn < 1e-12:
            x = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            x = x / xn

        y = np.cross(z, x)
        yn = float(np.linalg.norm(y))
        if yn < 1e-12:
            y = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            y = y / yn

        R = np.column_stack([x, y, z])

        frames.append({
            "origin_mm": p0.tolist(),
            "direction_unit": z.tolist(),
            "length_mm": L,
            "R_world_from_link": R.tolist(),
        })

    return frames


def export_parametric_skeleton_model(
    project_dir: Path,
    robot_name: str,
    n_links: int = 6,
    diameter_mm: float = 3.0,
    b_ref: float = 0.0,
    stl_reference_pose: bool = True,
):
    """
    Reads <robot_name>_gcode_calibration.json, creates a parametric skeleton definition.

    Exports:
      - processed_image_data_folder/<robot_name>_skeleton_parametric.json
      - processed_image_data_folder/<robot_name>_skeleton_predict.py
      - optionally an STL for reference pose at b_ref (default 0.0)
    """
    processed = project_dir / "processed_image_data_folder"
    gcode_json_path = processed / f"{robot_name}_gcode_calibration.json"
    if not gcode_json_path.is_file():
        raise FileNotFoundError(f"Missing calibration JSON: {gcode_json_path}")

    with open(gcode_json_path, "r") as f:
        cal_json = json.load(f)

    coeffs = cal_json.get("cubic_coefficients", {})
    r_coeffs = coeffs.get("r_coeffs")
    z_coeffs = coeffs.get("z_coeffs")
    y_coeffs = coeffs.get("offplane_y_coeffs")
    if r_coeffs is None or z_coeffs is None:
        raise ValueError("Calibration JSON missing cubic coeffs for r/z.")

    motor_setup = cal_json.get("motor_setup", {})
    b_rng = motor_setup.get("b_motor_position_range") or coeffs.get("b_motor_range")
    if b_rng is None or len(b_rng) != 2:
        raise ValueError("Calibration JSON missing b_motor_position_range.")

    b_min, b_max = float(b_rng[0]), float(b_rng[1])

    n_links = int(max(1, n_links))
    t = np.linspace(0.0, 1.0, n_links + 1)
    b_knots = b_min + t * (b_max - b_min)

    pts_ref = _sample_curve_xyz(cal_json, b_knots)

    origin = np.array([0.0, 0.0, 0.0], dtype=float)
    if np.linalg.norm(pts_ref[-1] - origin) > 1e-9:
        pts_ref_with_origin = np.vstack([pts_ref, origin])
    else:
        pts_ref_with_origin = pts_ref.copy()

    frames_ref = _compute_link_frames(pts_ref_with_origin)

    skel_param = {
        "robot_name": robot_name,
        "type": "parametric_link_chain",
        "units": "mm",
        "diameter_mm": float(diameter_mm),
        "radius_mm": float(diameter_mm) / 2.0,
        "n_links_curve": int(n_links),
        "unknown_tail_to_origin": True,
        "b_range": [b_min, b_max],
        "knot_definition": {
            "t_i": t.tolist(),
            "b_i": b_knots.tolist(),
            "meaning": "Curve knots are sampled along the full calibrated B range; each knot is a point on the fitted curve.",
        },
        "curve_equations": {
            "x_mm": {"poly_coeffs": r_coeffs, "definition": "x = r(b) signed planar radial deflection"},
            "z_mm": {"poly_coeffs": z_coeffs, "definition": "z = z(b) axial position"},
            "y_mm": {
                "poly_coeffs": y_coeffs,
                "definition": "y = y_offplane(b) if available else 0",
                "available": y_coeffs is not None,
            },
        },
        "reference_pose": {
            "points_xyz_mm": pts_ref_with_origin.round(6).tolist(),
            "link_frames": frames_ref,
            "note": "Reference pose uses knots sampled across B-range and a final link to origin. Use predictor to get poses for arbitrary b.",
        },
        "predictor_convention": {
            "inputs": ["b_pull_mm"],
            "outputs": [
                "points_xyz_mm (N+2 points including origin)",
                "link_frames (per-link origin, direction, length, rotation matrix)",
            ],
            "base_rule": "Append final segment to (0,0,0) if last point isn't already origin.",
        },
    }

    param_path = processed / f"{robot_name}_skeleton_parametric.json"
    with open(param_path, "w") as f:
        json.dump(skel_param, f, indent=2)

    predictor_code = f'''# Auto-generated parametric skeleton predictor for {robot_name}
# Units: mm
import numpy as np

R_COEFFS = {json.dumps(r_coeffs)}
Z_COEFFS = {json.dumps(z_coeffs)}
Y_COEFFS = {json.dumps(y_coeffs)}

B_MIN = {b_min:.10f}
B_MAX = {b_max:.10f}

T_KNOTS = np.array({json.dumps(t.tolist())}, dtype=float)
B_KNOTS = B_MIN + T_KNOTS * (B_MAX - B_MIN)

DIAMETER_MM = {float(diameter_mm):.6f}
RADIUS_MM = {float(diameter_mm) / 2.0:.6f}

def _polyval(c, b):
    if c is None:
        return None
    return np.polyval(np.asarray(c, dtype=float), np.asarray(b, dtype=float))

def curve_point_xyz(b):
    x = float(_polyval(R_COEFFS, b))
    z = float(_polyval(Z_COEFFS, b))
    if Y_COEFFS is None:
        y = 0.0
    else:
        y = float(_polyval(Y_COEFFS, b))
    return np.array([x, y, z], dtype=float)

def skeleton_points(b_pull_mm, include_origin=True):
    pts = np.stack([curve_point_xyz(bi) for bi in B_KNOTS], axis=0)
    if include_origin:
        if np.linalg.norm(pts[-1] - np.array([0.0, 0.0, 0.0])) > 1e-9:
            pts = np.vstack([pts, np.array([0.0, 0.0, 0.0])])
    return pts

def _compute_link_frames(points_xyz):
    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    frames = []
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    for i in range(pts.shape[0] - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            frames.append({{
                "origin_mm": p0.tolist(),
                "direction_unit": [0.0, 0.0, 1.0],
                "length_mm": 0.0,
                "R_world_from_link": np.eye(3).tolist(),
            }})
            continue
        z = d / L
        up = world_up
        if abs(float(np.dot(up, z))) > 0.95:
            up = np.array([0.0, 1.0, 0.0], dtype=float)
        x = np.cross(up, z)
        xn = float(np.linalg.norm(x))
        if xn < 1e-12:
            x = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            x /= xn
        y = np.cross(z, x)
        yn = float(np.linalg.norm(y))
        if yn < 1e-12:
            y = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            y /= yn
        R = np.column_stack([x, y, z])
        frames.append({{
            "origin_mm": p0.tolist(),
            "direction_unit": z.tolist(),
            "length_mm": L,
            "R_world_from_link": R.tolist(),
        }})
    return frames

def skeleton_link_frames(b_pull_mm, include_origin=True):
    pts = skeleton_points(b_pull_mm, include_origin=include_origin)
    return _compute_link_frames(pts)
'''
    py_path = processed / f"{robot_name}_skeleton_predict.py"
    with open(py_path, "w") as f:
        f.write(predictor_code)

    stl_path = None
    if stl_reference_pose:
        verts, faces = polyline_to_cylinder_mesh(pts_ref_with_origin, radius_mm=float(diameter_mm) / 2.0, sides=16)
        stl_path = processed / f"{robot_name}_robot_skeleton_reference.stl"
        write_binary_stl(stl_path, verts, faces, solid_name=f"{robot_name}_skeleton_ref")

    cal_json.setdefault("exported_models", {})
    cal_json["exported_models"]["robot_skeleton_parametric"] = {
        "format": "json+py",
        "parametric_json": param_path.name,
        "predictor_py": py_path.name,
        "reference_stl": stl_path.name if stl_path is not None else None,
        "diameter_mm": float(diameter_mm),
        "n_links": int(n_links),
        "note": "Use predictor to generate link endpoints/frames for any B pull; includes final link to origin for unknown parts.",
    }
    with open(gcode_json_path, "w") as f:
        json.dump(cal_json, f, indent=2)

    return param_path, py_path, stl_path, gcode_json_path


# -----------------------------------------
# STL helpers
# -----------------------------------------
def _unit(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < 1e-12:
        return np.zeros_like(v)
    return v / n


def _build_orthonormal_basis(direction: np.ndarray):
    d = _unit(direction)
    if np.linalg.norm(d) < 1e-12:
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])

    a = np.array([0.0, 0.0, 1.0])
    if abs(float(np.dot(d, a))) > 0.9:
        a = np.array([0.0, 1.0, 0.0])

    u = np.cross(d, a)
    u = _unit(u)
    v = np.cross(d, u)
    v = _unit(v)
    return u, v


def polyline_to_cylinder_mesh(points_xyz: np.ndarray, radius_mm: float = 1.5, sides: int = 16):
    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    if pts.shape[0] < 2:
        raise ValueError("Need at least two points for a polyline mesh.")

    verts = []
    faces = []

    for i in range(pts.shape[0] - 1):
        p0 = pts[i]
        p1 = pts[i + 1]
        d = p1 - p0
        if float(np.linalg.norm(d)) < 1e-9:
            continue

        u, v = _build_orthonormal_basis(d)

        ring0 = []
        ring1 = []
        for k in range(sides):
            theta = 2.0 * np.pi * (k / sides)
            offset = radius_mm * (np.cos(theta) * u + np.sin(theta) * v)
            ring0.append(p0 + offset)
            ring1.append(p1 + offset)

        base_idx = len(verts)
        verts.extend(ring0)
        verts.extend(ring1)

        for k in range(sides):
            k2 = (k + 1) % sides
            a0 = base_idx + k
            b0 = base_idx + k2
            a1 = base_idx + sides + k
            b1 = base_idx + sides + k2

            faces.append([a0, a1, b1])
            faces.append([a0, b1, b0])

    return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)


def write_binary_stl(path: Path, vertices: np.ndarray, faces: np.ndarray, solid_name: str = "robot_skeleton"):
    path = Path(path)
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = np.asarray(faces, dtype=np.int32)

    header = bytearray(80)
    name_bytes = solid_name.encode("ascii", errors="ignore")[:80]
    header[: len(name_bytes)] = name_bytes

    tri_count = faces.shape[0]
    with open(path, "wb") as f:
        f.write(header)
        f.write(np.uint32(tri_count).tobytes())
        for tri in faces:
            p0 = vertices[tri[0]]
            p1 = vertices[tri[1]]
            p2 = vertices[tri[2]]
            n = np.cross(p1 - p0, p2 - p0)
            n = _unit(n).astype(np.float32)
            f.write(n.tobytes())
            f.write(p0.astype(np.float32).tobytes())
            f.write(p1.astype(np.float32).tobytes())
            f.write(p2.astype(np.float32).tobytes())
            f.write(np.uint16(0).tobytes())


# -----------------------------------------
# Tip precision guidance
# -----------------------------------------
TIP_PRECISION_GUIDE = r"""
Tip precision: how to get closer to the distal edge of the black segment
------------------------------------------------------------
Your current pipeline finds a skeleton tip (centerline distal pixel). That is
usually inside the tube and thus upstream from the true physical tip edge.

Available refiners:
  (1) edge_dt
      - Step from the skeleton tip along the distal tangent direction until
        you leave the foreground.
      - Last in-foreground point is the tip estimate.

  (2) edge_grad
      - Sample grayscale along distal tangent direction and locate the strongest
        intensity gradient (dark->light edge).

  (3) mainray
      - Skeletonize the foreground, estimate a stable tangent from the upstream
        trunk path, then ray-cast straight to the boundary.
      - Best when the end has a small weird hook/branch that corrupts the local tip angle.

  (4) parallel_centerline
      - Use the CURRENT measured tip angle to define the tube extremity side-line direction.
      - Find the two side offsets from multiple upstream cross-sections.
      - Build the center parallel line.
      - Intersect that center line with the distal black/white boundary.
      - Annotates the side lines, center line, anchor, and refined tip on the analyzed images.
"""


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", type=str, default=None,
                    help="Existing project folder containing raw_image_data_folder/")
    ap.add_argument("--raw_dir", type=str, default=None,
                    help="raw_image_data_folder or any folder of images (will be wrapped into a project)")
    ap.add_argument("--threshold", type=int, default=200)
    ap.add_argument("--robot_name", type=str, default="calibrated_robot")
    ap.add_argument("--save_plots", action="store_true")

    ap.add_argument("--width_in_pixels", type=float, default=3025)
    ap.add_argument("--width_in_mm", type=float, default=140)

    ap.add_argument("--ruler_mm", type=float, default=150.0,
                    help="Known ruler distance in mm for the 2-point reference.")
    ap.add_argument("--camera_calibration_file", type=str, default=None,
                    help="Path to camera calibration .npz for checkerboard-reference analysis.")
    ap.add_argument("--checkerboard_reference_image", type=str, default=None,
                    help="Path to checkerboard reference image. If omitted, board-reference analysis is skipped.")
    ap.add_argument("--checkerboard_inner_corners", type=_parse_inner_corners_arg, default=None,
                    help="Checkerboard inner-corner grid as 'Nx,Ny' or 'NxXNy'. Defaults to metadata in the camera calibration file.")
    ap.add_argument("--checkerboard_square_size_mm", type=float, default=None,
                    help="Checkerboard square size in mm. Defaults to metadata in the camera calibration file.")
    ap.add_argument("--checkerboard_no_undistort", action="store_true",
                    help="Disable undistortion before checkerboard pose estimation.")

    ap.add_argument("--checkerboard_mm_scale_correction", type=float, default=0.5,
                    help="Multiply checkerboard-derived mm values by this factor. Default 0.5 fixes the observed 2x overscale.")
    ap.add_argument("--checkerboard_no_flip_planar_x", action="store_true",
                    help="Disable the checkerboard planar-x sign flip. By default x is flipped to match the ruler convention.")

    ap.add_argument("--link_mode", type=str, default="symlink", choices=["symlink", "copy"])
    ap.add_argument("--save_analysis_config", action="store_true",
                    help="Write analysis_reference.json into project_dir for reuse without GUI later.")

    ap.add_argument("--tip_refine_mode", type=str, default="none",
                    choices=["none", "edge_dt", "edge_grad", "mainray", "parallel_centerline"],
                    help="Refine tip position toward distal edge of black segment.")
    ap.add_argument("--tip_refine_dt_step_px", type=float, default=1.0)
    ap.add_argument("--tip_refine_max_step_px", type=int, default=80)
    ap.add_argument("--tip_refine_grad_step_px", type=float, default=0.25)
    ap.add_argument("--tip_refine_grad_search_len_px", type=int, default=60)

    ap.add_argument("--tip_mainray_fit_back_near_r", type=float, default=1.5)
    ap.add_argument("--tip_mainray_fit_back_far_r", type=float, default=6.0)
    ap.add_argument("--tip_mainray_anchor_back_r", type=float, default=1.0)
    ap.add_argument("--tip_mainray_ray_step_px", type=float, default=0.5)
    ap.add_argument("--tip_mainray_ray_max_len_r", type=float, default=8.0)

    ap.add_argument("--tip_parallel_section_near_r", type=float, default=1.0)
    ap.add_argument("--tip_parallel_section_far_r", type=float, default=6.0)
    ap.add_argument("--tip_parallel_scan_half_r", type=float, default=3.0)
    ap.add_argument("--tip_parallel_num_sections", type=int, default=9)
    ap.add_argument("--tip_parallel_cross_step_px", type=float, default=0.5)
    ap.add_argument("--tip_parallel_ray_step_px", type=float, default=0.5)
    ap.add_argument("--tip_parallel_ray_max_len_r", type=float, default=8.0)

    ap.add_argument("--export_skeleton", action="store_true",
                    help="Export parametric skeleton (equations) and patch it into *_gcode_calibration.json.")
    ap.add_argument("--skeleton_diameter_mm", type=float, default=3.0)
    ap.add_argument("--skeleton_links", type=int, default=6)
    ap.add_argument("--skeleton_reference_stl", action="store_true",
                    help="Also export a reference-pose STL for quick viewing.")
    ap.add_argument("--print_tip_precision_guide", action="store_true",
                    help="Print tips for improving tip accuracy and exit.")

    args = ap.parse_args()

    if args.print_tip_precision_guide:
        print(TIP_PRECISION_GUIDE)
        return

    if args.project_dir is None and args.raw_dir is None:
        raise SystemExit("Provide --project_dir or --raw_dir")

    project_dir = Path(args.project_dir).expanduser().resolve() if args.project_dir else None
    raw_dir = Path(args.raw_dir).expanduser().resolve() if args.raw_dir else None

    if project_dir and (project_dir / "raw_image_data_folder").is_dir():
        raw_folder = project_dir / "raw_image_data_folder"
    elif raw_dir:
        if raw_dir.name == "raw_image_data_folder":
            raw_folder = raw_dir
            if project_dir is None:
                project_dir = raw_dir.parent
            elif project_dir != raw_dir.parent:
                raw_folder = ensure_project_from_raw(raw_dir, project_dir, link_mode=args.link_mode)
        else:
            if project_dir is None:
                project_dir = raw_dir.parent / (raw_dir.name + "_offline_project")
            raw_folder = ensure_project_from_raw(raw_dir, project_dir, link_mode=args.link_mode)
    else:
        raise SystemExit("Could not resolve folders. Check paths.")

    project_dir = project_dir.resolve()
    raw_folder = raw_folder.resolve()

    cal = CTR_Shadow_Calibration(
        parent_directory=str(project_dir.parent),
        project_name=project_dir.name,
        allow_existing=True,
        add_date=False,
    )
    cal.calibration_data_folder = str(project_dir)

    if args.tip_refine_mode != "none":
        patch_analyze_data_for_tip_refinement(
            cal,
            refine_mode=str(args.tip_refine_mode),
            dt_step_px=float(args.tip_refine_dt_step_px),
            dt_max_step_px=int(args.tip_refine_max_step_px),
            grad_step_px=float(args.tip_refine_grad_step_px),
            grad_search_len_px=int(args.tip_refine_grad_search_len_px),
            mainray_fit_back_near_r=float(args.tip_mainray_fit_back_near_r),
            mainray_fit_back_far_r=float(args.tip_mainray_fit_back_far_r),
            mainray_anchor_back_r=float(args.tip_mainray_anchor_back_r),
            mainray_ray_step_px=float(args.tip_mainray_ray_step_px),
            mainray_ray_max_len_r=float(args.tip_mainray_ray_max_len_r),
            parallel_section_near_r=float(args.tip_parallel_section_near_r),
            parallel_section_far_r=float(args.tip_parallel_section_far_r),
            parallel_scan_half_r=float(args.tip_parallel_scan_half_r),
            parallel_num_sections=int(args.tip_parallel_num_sections),
            parallel_cross_step_px=float(args.tip_parallel_cross_step_px),
            parallel_ray_step_px=float(args.tip_parallel_ray_step_px),
            parallel_ray_max_len_r=float(args.tip_parallel_ray_max_len_r),
        )
        print(f"[INFO] Tip refinement enabled: {args.tip_refine_mode}")

    imgs = list_images(raw_folder)
    if not imgs:
        raise SystemExit(f"No images found in: {raw_folder}")

    first_img_path = imgs[0]
    img_bgr = cv2.imread(str(first_img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise SystemExit(f"Could not read first image: {first_img_path}")

    print(f"[INFO] Using first image for GUI: {first_img_path.name}")

    board_reference_debug_image = None
    if args.camera_calibration_file:
        calib_path = Path(args.camera_calibration_file).expanduser().resolve()
        print(f"[INFO] Loading camera calibration: {calib_path}")
        cal.load_camera_calibration(str(calib_path))

        if args.checkerboard_reference_image:
            board_ref_path = Path(args.checkerboard_reference_image).expanduser().resolve()
            processed_dir = project_dir / "processed_image_data_folder"
            processed_dir.mkdir(parents=True, exist_ok=True)
            checkerboard_debug_path = processed_dir / "checkerboard_reference_debug.png"
            board_result = cal.estimate_board_reference_from_image(
                str(board_ref_path),
                inner_corners=args.checkerboard_inner_corners,
                square_size_mm=args.checkerboard_square_size_mm,
                use_undistort=(not args.checkerboard_no_undistort),
                draw_debug=True,
                save_debug_path=str(checkerboard_debug_path),
            )
            board_reference_debug_image = board_result.get("debug_image")

            apply_checkerboard_reference_corrections(
                cal,
                mm_scale=float(args.checkerboard_mm_scale_correction),
                flip_planar_x=(not bool(args.checkerboard_no_flip_planar_x)),
            )

            print(f"[INFO] Checkerboard reference estimated from: {board_ref_path}")
        else:
            print("[INFO] Camera calibration loaded, but no checkerboard reference image was provided.")
    elif args.checkerboard_reference_image:
        print("[WARN] checkerboard_reference_image was provided without camera_calibration_file; skipping board-reference analysis.")

    analysis_crop, ruler = interactive_crop_and_ruler_from_image(
        img_bgr,
        default_crop=cal.default_analysis_crop,
        ruler_known_mm=float(args.ruler_mm),
    )

    cal.analysis_crop = dict(analysis_crop)

    if ruler["p1_px"] is not None and ruler["p2_px"] is not None:
        cal.ruler_ref_p1_px = tuple(ruler["p1_px"])
        cal.ruler_ref_p2_px = tuple(ruler["p2_px"])
        cal.ruler_ref_distance_mm = float(ruler["known_distance_mm"])
        cal.ruler_mm_per_px = float(ruler["mm_per_px"])
        cal.ruler_px_per_mm = float(ruler["px_per_mm"])
        cal.ruler_axis_unit = np.asarray(ruler["axis_unit"], dtype=float)
        cal.ruler_axis_perp_unit = np.asarray(ruler["axis_perp_unit"], dtype=float)
        cal.ruler_calib_meta = dict(ruler["meta"]) if ruler["meta"] else None
    else:
        cal.ruler_ref_p1_px = None
        cal.ruler_ref_p2_px = None
        cal.ruler_ref_distance_mm = None
        cal.ruler_mm_per_px = None
        cal.ruler_px_per_mm = None
        cal.ruler_axis_unit = None
        cal.ruler_axis_perp_unit = None
        cal.ruler_calib_meta = dict(ruler["meta"]) if ruler["meta"] else None

    if args.save_analysis_config:
        cfg = cal.get_analysis_reference_info()
        cfg["board_reference"] = collect_board_reference_info(cal)
        if hasattr(cal, "tip_refine_debug_records"):
            cfg["tip_refine_mode"] = args.tip_refine_mode
        cfg_path = project_dir / "analysis_reference.json"
        with open(cfg_path, "w") as f:
            json.dump(_json_ready(cfg), f, indent=2)
        print(f"[INFO] Saved analysis config to: {cfg_path}")

    print("\n[INFO] Running analyze_data_batch (offline)...")
    cal.analyze_data_batch(threshold=int(args.threshold))

    if hasattr(cal, "tip_refine_debug_records") and len(cal.tip_refine_debug_records) > 0:
        dbg_path = project_dir / "processed_image_data_folder" / "tip_refine_debug_records.json"
        dbg_path.parent.mkdir(parents=True, exist_ok=True)
        with open(dbg_path, "w") as f:
            json.dump(_json_ready(cal.tip_refine_debug_records), f, indent=2)
        print(f"[INFO] Saved tip refinement debug records: {dbg_path}")

    print("\n[INFO] Running postprocess_calibration_data (offline)...")
    cal.postprocess_calibration_data(
        width_in_pixels=float(args.width_in_pixels),
        width_in_mm=float(args.width_in_mm),
        robot_name=str(args.robot_name),
        save_plots=bool(args.save_plots),
    )

    if board_reference_debug_image is not None:
        try:
            annotated_board_path = project_dir / "processed_image_data_folder" / "checkerboard_reference_annotated_analysis.png"
            draw_checkerboard_analysis_overlay(
                cal=cal,
                output_path=annotated_board_path,
                tracked_rows=np.asarray(cal.tip_locations_array_coarse, dtype=float),
                image_files=imgs,
                board_debug_image=board_reference_debug_image,
            )
            print(f"[INFO] Saved annotated checkerboard analysis image: {annotated_board_path}")
        except Exception as e:
            print(f"[WARN] Failed to create annotated checkerboard analysis image: {e}")

    if args.export_skeleton:
        try:
            param_path, py_path, stl_path, patched_json = export_parametric_skeleton_model(
                project_dir=project_dir,
                robot_name=str(args.robot_name),
                n_links=int(args.skeleton_links),
                diameter_mm=float(args.skeleton_diameter_mm),
                b_ref=0.0,
                stl_reference_pose=bool(args.skeleton_reference_stl),
            )
            print("\n[INFO] Parametric skeleton export complete:")
            print(f"  Parametric JSON: {param_path}")
            print(f"  Predictor PY:    {py_path}")
            if stl_path is not None:
                print(f"  Reference STL:   {stl_path}")
            print(f"  GCode JSON patched: {patched_json}")
        except Exception as e:
            print(f"[WARN] Parametric skeleton export failed: {e}")

    print("\n[DONE] Offline GUI + pipeline complete.")
    print(f"Outputs are in: {project_dir / 'processed_image_data_folder'}")


if __name__ == "__main__":
    main()
