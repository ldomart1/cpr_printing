#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
offline_run_checkerboard_tip_error_analysis.py

Offline CTR shadow calibration runner that:
  1) Opens a project/raw image folder
  2) Lets you choose crop bounds from the FIRST raw image
  3) Uses a checkerboard reference image + camera calibration to define mm axes
  4) Runs analyze_data_batch + existing tracking pipeline
  5) Converts tracked tip pixels to checkerboard-referenced mm coordinates
  6) Computes a reference tracked point = mean of all valid tracked points
  7) Computes per-sample error distance to that mean (in mm)
  8) Saves:
       - CSV of tracked points and errors
       - JSON metrics summary
       - error-vs-sample plot
       - histogram of error distance vs number of samples
       - optional overlay on checkerboard image

Key outputs:
  processed_image_data_folder/
    tracked_tip_positions_mm.csv
    tracked_tip_error_metrics.json
    tracked_tip_error_vs_sample.png
    tracked_tip_error_histogram.png
    checkerboard_reference_annotated_analysis.png   (optional)

Usage example:
  python3 offline_run_checkerboard_tip_error_analysis.py \
    --project_dir "/path/to/project" \
    --camera_calibration_file "/path/to/camera_calibration.npz" \
    --checkerboard_reference_image "/path/to/checkerboard_ref.png" \
    --threshold 200 \
    --save_plots \
    --tip_refine_mode edge_dt

Important:
  - This script patches the filename parser used by shadow_calibration so that files named like:
      00134_tracked_X90.604_Y31.876_Z-173.537_B-3.368_C-52.800_...
    are parsed correctly.
  - It uses X as the "ntnl_pos" placeholder and B as the "ss_pos" placeholder because
    the existing pipeline expects a 3-value tuple from filename parsing.
"""

import argparse
import csv
import json
import os
import shutil
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# Add path to your shadow_calibration script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

import shadow_calibration as shadow_calibration_module  # noqa: E402
from shadow_calibration import CTR_Shadow_Calibration  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


# -----------------------------
# Utilities: IO and discovery
# -----------------------------
def list_images(folder: Path) -> List[Path]:
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
    H = np.asarray(H, dtype=np.float64).copy()
    S = np.diag([float(mm_scale), float(mm_scale), 1.0]).astype(np.float64)
    return S @ H


def _scale_px_from_mm_homography(H: np.ndarray, mm_scale: float) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64).copy()
    if abs(float(mm_scale)) < 1e-12:
        raise ValueError("mm_scale must be non-zero.")
    S_inv = np.diag([1.0 / float(mm_scale), 1.0 / float(mm_scale), 1.0]).astype(np.float64)
    return H @ S_inv


# -----------------------------------------
# Filename parser patch for tracked_X..._B...
# -----------------------------------------
def robust_parse_positions_from_filename(file_name: str):
    """
    Supports filenames like:
      00134_tracked_X90.604_Y31.876_Z-173.537_B-3.368_C-52.800_20260319_113124_658346.png

    Returns:
      orientation, ntnl_pos, ss_pos

    Mapping used to keep compatibility with the existing pipeline:
      - orientation = inferred from side token / C value / legacy numeric token
      - ntnl_pos    = X value
      - ss_pos      = B value
    """
    base = os.path.splitext(os.path.basename(file_name))[0]
    parts = base.split("_")

    orientation = None
    side_token = None

    values = {}
    for p in parts:
        p_lower = p.lower()
        if p_lower in ("right", "left"):
            side_token = p_lower
        if len(p) >= 2 and p[0] in ("X", "Y", "Z", "B", "C"):
            key = p[0]
            try:
                values[key] = float(p[1:])
            except Exception:
                pass

    if side_token == "right":
        orientation = 0
    elif side_token == "left":
        orientation = 1

    if orientation is None and "C" in values:
        c_val = float(values["C"])
        c_norm = ((c_val + 180.0) % 360.0) - 180.0
        if abs(c_norm - 0.0) <= 5.0:
            orientation = 0
        elif abs(abs(c_norm) - 180.0) <= 5.0:
            orientation = 1
        elif abs(c_norm - 90.0) <= 5.0:
            orientation = 2
        elif abs(c_norm + 90.0) <= 5.0:
            orientation = 3

    if orientation is None and len(parts) > 0:
        try:
            candidate = int(parts[0])
            if candidate in (0, 1, 2, 3):
                orientation = candidate
        except Exception:
            pass

    if orientation is None:
        orientation = 0

    if "X" not in values:
        raise ValueError(f"Could not parse X value from filename: {file_name}")
    if "B" not in values:
        raise ValueError(f"Could not parse B value from filename: {file_name}")

    ntnl_pos = float(values["X"])
    ss_pos = float(values["B"])
    return orientation, ntnl_pos, ss_pos


def extract_named_values_from_filename(file_name: str) -> Dict[str, float]:
    """
    Extract numeric values from filename tokens such as X..., Y..., Z..., B..., C...
    """
    base = os.path.splitext(os.path.basename(file_name))[0]
    values: Dict[str, float] = {}
    for p in base.split("_"):
        if len(p) < 2 or p[0] not in ("X", "Y", "Z", "B", "C"):
            continue
        try:
            values[p[0]] = float(p[1:])
        except Exception:
            continue
    return values


def patch_filename_parser():
    """
    Patch shadow_calibration filename parsing so analyze_data_batch can read
    filenames like:
      00134_tracked_X90.604_Y31.876_Z-173.537_B-3.368_C-52.800_...
    """
    shadow_calibration_module._get_pos_from_file_name = robust_parse_positions_from_filename
    print("[INFO] Patched filename parser for tracked_X..._B... style filenames.")


def normalize_raw_filenames_for_shadow_parser(raw_folder: Path) -> List[Tuple[str, str]]:
    """
    Rename files in-place so shadow_calibration's legacy parser can read them.

    Legacy parser requirement:
      {orientation}_{X}_{B}_...
    """
    rename_pairs: List[Tuple[str, str]] = []

    for src in list_images(raw_folder):
        try:
            orientation, ntnl_pos, ss_pos = robust_parse_positions_from_filename(src.name)
        except Exception:
            continue

        base = src.stem
        legacy_prefix = f"{int(orientation)}_{float(ntnl_pos):.6f}_{float(ss_pos):.6f}_"
        if base.startswith(legacy_prefix):
            continue

        dst_name = f"{legacy_prefix}{base}{src.suffix}"
        dst = src.with_name(dst_name)
        counter = 1
        while dst.exists() and dst != src:
            dst_name = f"{legacy_prefix}{base}__{counter}{src.suffix}"
            dst = src.with_name(dst_name)
            counter += 1

        src.rename(dst)
        rename_pairs.append((src.name, dst.name))

    return rename_pairs


# -----------------------------------------
# Checkerboard reference correction helpers
# -----------------------------------------
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


# -------------------------------------------
# GUI: crop selection from one image
# -------------------------------------------
def interactive_crop_from_image(
    image_bgr: np.ndarray,
    default_crop=None,
    window_crop="Manual Crop Setup (OFFLINE)",
):
    """
    Returns analysis_crop dict in the format CTR_Shadow_Calibration expects.
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
        return best_name if best_dist <= drag_threshold_px**2 else None

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

    return analysis_crop


# ------------------------------------------
# Tip refinement: push tip to distal boundary
# ------------------------------------------
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
    tip_yx: (y, x)
    """
    h, w = binary_image.shape[:2]
    fg = (binary_image == 0).astype(np.uint8)

    ang = np.deg2rad(float(tip_angle_deg))
    vx = float(np.sin(ang))
    vy = float(np.cos(ang))

    y0, x0 = float(tip_yx[0]), float(tip_yx[1])

    last_inside = (y0, x0)
    exited = False
    n_steps = int(max(1, max_step_px / max(step_px, 1e-6)))

    for i in range(1, n_steps + 1):
        y = y0 + i * step_px * vy
        x = x0 + i * step_px * vx
        iy = int(round(y))
        ix = int(round(x))
        if not (0 <= iy < h and 0 <= ix < w):
            exited = True
            break
        if fg[iy, ix] == 1:
            last_inside = (y, x)
            continue
        else:
            exited = True
            if exit_mode == "first_background":
                break

    dbg = {
        "vx": vx,
        "vy": vy,
        "exited": exited,
        "last_inside": last_inside,
    }
    return float(last_inside[0]), float(last_inside[1]), dbg


def refine_tip_edge_subpixel_gradient(
    grayscale: np.ndarray,
    binary_image: np.ndarray,
    tip_yx: Tuple[int, int],
    tip_angle_deg: float,
    search_len_px: int = 60,
    step_px: float = 0.25,
):
    """
    Subpixel-ish edge refinement along tangent.
    """
    h, w = grayscale.shape[:2]
    fg = (binary_image == 0).astype(np.uint8)

    ang = np.deg2rad(float(tip_angle_deg))
    vx = float(np.sin(ang))
    vy = float(np.cos(ang))

    y0, x0 = float(tip_yx[0]), float(tip_yx[1])
    n = int(max(5, search_len_px / max(step_px, 1e-6)))

    samples = []
    coords = []
    inside_mask = []

    for i in range(n + 1):
        y = y0 + i * step_px * vy
        x = x0 + i * step_px * vx
        iy = int(np.clip(round(y), 0, h - 1))
        ix = int(np.clip(round(x), 0, w - 1))
        samples.append(float(grayscale[iy, ix]))
        coords.append((y, x))
        inside_mask.append(int(fg[iy, ix] == 1))

    s = np.array(samples, dtype=float)
    g = np.diff(s)
    g_abs = np.abs(g)

    valid = []
    for i in range(len(g_abs)):
        if inside_mask[i] == 1:
            valid.append(i)

    if not valid:
        yy, xx, dbg = refine_tip_edge_distance_transform(binary_image, tip_yx, tip_angle_deg)
        return yy, xx, {"fallback": "edge_dt", "edge_dt": dbg}

    valid = np.array(valid, dtype=int)
    j = int(valid[np.argmax(g_abs[valid])])
    y_edge = 0.5 * (coords[j][0] + coords[j + 1][0])
    x_edge = 0.5 * (coords[j][1] + coords[j + 1][1])

    dbg = {
        "vx": vx,
        "vy": vy,
        "j": j,
        "g_max": float(g_abs[j]),
        "step_px": float(step_px),
    }
    return float(y_edge), float(x_edge), dbg


def patch_analyze_data_for_tip_refinement(
    cal: CTR_Shadow_Calibration,
    refine_mode: str = "none",
    dt_step_px: float = 1.0,
    dt_max_step_px: int = 80,
    grad_step_px: float = 0.25,
    grad_search_len_px: int = 60,
):
    """
    Patches cal.analyze_data(...) to refine coarse tip pixel coordinates.
    """
    original_analyze_data = cal.analyze_data

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
        else:
            yy, xx, _dbg = tip_y, tip_x, {}

        coarse_row_refined = np.array(coarse_row, dtype=float).copy()
        coarse_row_refined[0] = yy + crop_y_min_img
        coarse_row_refined[1] = xx + crop_x_min_img

        return fig, axs, coarse_row_refined, fine_row

    cal.analyze_data = analyze_data_patched
    return cal


# -----------------------------------------
# Visualization helpers
# -----------------------------------------
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
        raise RuntimeError("Board reference is unavailable.")

    valid_mask = np.all(np.isfinite(tracked_rows[:, :2]), axis=1)
    if tracked_rows.shape[1] > 3:
        valid_mask &= np.isfinite(tracked_rows[:, 3])
    valid_rows = tracked_rows[valid_mask]
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

    summary_lines = [
        f"tracked points: {len(valid_rows)}",
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


# -----------------------------------------
# Error analysis in checkerboard mm frame
# -----------------------------------------
def _extract_b_value_from_row(row: np.ndarray) -> float:
    if row is None or len(row) < 4:
        return float("nan")
    try:
        return float(row[3])
    except Exception:
        return float("nan")


def compute_tracked_tip_positions_mm(
    cal: CTR_Shadow_Calibration,
    tracked_rows: np.ndarray,
    image_files: List[Path],
) -> Dict[str, Any]:
    """
    Convert tracked tip pixel coordinates to checkerboard-referenced mm.
    """
    if getattr(cal, "board_pose", None) is None:
        raise RuntimeError("Checkerboard board_pose is not available.")
    if tracked_rows is None or tracked_rows.size == 0:
        raise RuntimeError("tracked_rows is empty.")

    origin = np.asarray(cal.board_pose["origin_px"], dtype=np.float64).reshape(2)

    records = []
    mm_points = []
    valid_indices = []

    for i, row in enumerate(np.asarray(tracked_rows, dtype=float)):
        file_name = image_files[i].name if i < len(image_files) else f"sample_{i:04d}"
        if row.size < 2 or not np.all(np.isfinite(row[:2])):
            records.append({
                "sample_index": i,
                "image_name": file_name,
                "tip_y_px": None,
                "tip_x_px": None,
                "u_mm": None,
                "z_mm": None,
                "b_pull_mm": None,
                "valid": False,
            })
            continue

        y_px = float(row[0])
        x_px = float(row[1])

        try:
            u_mm, z_mm = cal.pixel_point_to_calibrated_axes(
                x_px=x_px,
                y_px=y_px,
                origin_px=origin,
            )
            u_mm = float(u_mm)
            z_mm = float(z_mm)
            valid = np.isfinite(u_mm) and np.isfinite(z_mm)
        except Exception:
            u_mm, z_mm = float("nan"), float("nan")
            valid = False

        rec = {
            "sample_index": i,
            "image_name": file_name,
            "tip_y_px": y_px,
            "tip_x_px": x_px,
            "u_mm": u_mm if valid else None,
            "z_mm": z_mm if valid else None,
            "b_pull_mm": _extract_b_value_from_row(row),
            "valid": bool(valid),
        }
        named_values = extract_named_values_from_filename(file_name)
        rec["desired_x_mm"] = float(named_values["X"]) if "X" in named_values else None
        rec["desired_y_mm"] = float(named_values["Y"]) if "Y" in named_values else None
        rec["desired_z_mm"] = float(named_values["Z"]) if "Z" in named_values else None
        records.append(rec)

        if valid:
            mm_points.append([u_mm, z_mm])
            valid_indices.append(i)

    mm_points = np.asarray(mm_points, dtype=float) if len(mm_points) else np.empty((0, 2), dtype=float)

    return {
        "records": records,
        "valid_indices": valid_indices,
        "mm_points": mm_points,
    }


def compute_error_metrics(mm_points: np.ndarray) -> Dict[str, Any]:
    """
    Reference point = mean of all valid tracked points.
    Error for each sample = Euclidean distance to that mean.
    RMSE = sqrt(mean(error^2))
    """
    mm_points = np.asarray(mm_points, dtype=float).reshape(-1, 2)
    if mm_points.shape[0] == 0:
        raise RuntimeError("No valid mm points available for error analysis.")

    ref_mean = np.mean(mm_points, axis=0)
    deltas = mm_points - ref_mean[None, :]
    dist_mm = np.linalg.norm(deltas, axis=1)

    rmse_mm = float(np.sqrt(np.mean(dist_mm ** 2)))
    mean_err_mm = float(np.mean(dist_mm))
    std_err_mm = float(np.std(dist_mm))
    max_err_mm = float(np.max(dist_mm))
    min_err_mm = float(np.min(dist_mm))
    median_err_mm = float(np.median(dist_mm))

    return {
        "reference_point_mm": {
            "u_mean_mm": float(ref_mean[0]),
            "z_mean_mm": float(ref_mean[1]),
        },
        "errors_mm": dist_mm,
        "deltas_mm": deltas,
        "rmse_mm": rmse_mm,
        "mean_error_mm": mean_err_mm,
        "std_error_mm": std_err_mm,
        "median_error_mm": median_err_mm,
        "min_error_mm": min_err_mm,
        "max_error_mm": max_err_mm,
        "num_samples": int(mm_points.shape[0]),
    }


def attach_errors_to_records(
    records: List[Dict[str, Any]],
    valid_indices: List[int],
    deltas_mm: np.ndarray,
    errors_mm: np.ndarray,
):
    idx_to_local = {global_idx: k for k, global_idx in enumerate(valid_indices)}

    for i, rec in enumerate(records):
        if i not in idx_to_local:
            rec["du_from_mean_mm"] = None
            rec["dz_from_mean_mm"] = None
            rec["error_distance_mm"] = None
            continue

        k = idx_to_local[i]
        rec["du_from_mean_mm"] = float(deltas_mm[k, 0])
        rec["dz_from_mean_mm"] = float(deltas_mm[k, 1])
        rec["error_distance_mm"] = float(errors_mm[k])

    return records


def save_tracked_tip_csv(csv_path: Path, records: List[Dict[str, Any]]):
    fieldnames = [
        "sample_index",
        "image_name",
        "tip_y_px",
        "tip_x_px",
        "u_mm",
        "z_mm",
        "b_pull_mm",
        "desired_x_mm",
        "desired_y_mm",
        "desired_z_mm",
        "du_from_mean_mm",
        "dz_from_mean_mm",
        "error_distance_mm",
        "valid",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def save_error_plot(error_plot_path: Path, records: List[Dict[str, Any]], title_prefix: str = ""):
    valid_recs = [r for r in records if r.get("valid", False) and r.get("error_distance_mm") is not None]
    if not valid_recs:
        raise RuntimeError("No valid records available for error plot.")

    sample_idx = [int(r["sample_index"]) for r in valid_recs]
    errors_mm = [float(r["error_distance_mm"]) for r in valid_recs]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="none")
    ax.set_facecolor("none")
    ax.plot(
        sample_idx,
        errors_mm,
        marker="o",
        linestyle="-",
        color="white",
        markerfacecolor="white",
        markeredgecolor="white",
        linewidth=1.8,
        markersize=4.5,
    )
    ax.set_xlabel("Sample index", color="white")
    ax.set_ylabel("Error distance to mean tracked point (mm)", color="white")
    ax.set_title(f"{title_prefix}Tracked tip error vs sample".strip(), color="white")
    ax.grid(True, alpha=0.22, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    plt.tight_layout()
    plt.savefig(error_plot_path, dpi=200, transparent=True)
    plt.close()


def save_error_histogram(
    hist_path: Path,
    errors_mm: np.ndarray,
    rmse_mm: float,
    title_prefix: str = "",
    bins: int = 20,
):
    errors_mm = np.asarray(errors_mm, dtype=float).reshape(-1)
    if errors_mm.size == 0:
        raise RuntimeError("No error values available for histogram.")

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="none")
    ax.set_facecolor("none")
    ax.hist(
        errors_mm,
        bins=int(max(5, bins)),
        color="white",
        edgecolor="white",
        linewidth=1.0,
        alpha=0.9,
    )
    ax.set_xlabel("Error distance (mm)", color="white")
    ax.set_ylabel("Number of samples", color="white")
    ax.set_title(
        f"{title_prefix}Histogram of tracked tip error | RMSE = {float(rmse_mm):.4f} mm".strip(),
        color="white",
    )
    ax.grid(True, alpha=0.18, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=200, transparent=True)
    plt.close()


def save_desired_vs_actual_plot(
    plot_path: Path,
    records: List[Dict[str, Any]],
    reference_point_mm: Dict[str, float],
    title_prefix: str = "",
):
    valid_recs = [r for r in records if r.get("valid", False) and r.get("u_mm") is not None and r.get("z_mm") is not None]
    desired_recs = [r for r in valid_recs if r.get("desired_x_mm") is not None and r.get("desired_z_mm") is not None]
    if not valid_recs:
        raise RuntimeError("No valid tracked records available for desired-vs-actual plot.")
    if not desired_recs:
        raise RuntimeError("Desired tip X/Z values could not be parsed from filenames.")

    actual_u = np.asarray([float(r["u_mm"]) for r in valid_recs], dtype=float)
    actual_z = np.asarray([float(r["z_mm"]) for r in valid_recs], dtype=float)
    desired_tip_x = np.asarray([float(r["desired_x_mm"]) for r in desired_recs], dtype=float)
    desired_tip_z = np.asarray([float(r["desired_z_mm"]) for r in desired_recs], dtype=float)

    ref_u = float(reference_point_mm["u_mean_mm"])
    ref_z = float(reference_point_mm["z_mean_mm"])

    desired_u = desired_tip_x - np.mean(desired_tip_x) + ref_u
    desired_z = desired_tip_z - np.mean(desired_tip_z) + ref_z

    cmap = LinearSegmentedColormap.from_list(
        "sleek_actual_density",
        ["#081018", "#12344d", "#1b6ca8", "#3ddbd9", "#f4d35e"],
        N=256,
    )

    fig, ax = plt.subplots(figsize=(8.5, 7.0), facecolor="#07111a")
    ax.set_facecolor("#0b1622")

    hb = ax.hexbin(
        actual_u,
        actual_z,
        gridsize=26,
        mincnt=1,
        cmap=cmap,
        linewidths=0.0,
    )
    cbar = fig.colorbar(hb, ax=ax, pad=0.02)
    cbar.set_label("Actual point density", color="#dceaf7")
    cbar.ax.yaxis.set_tick_params(color="#dceaf7")
    plt.setp(cbar.ax.get_yticklabels(), color="#dceaf7")
    cbar.outline.set_edgecolor("#8eb8d8")

    ax.plot(
        desired_u,
        desired_z,
        color="#8cf7ff",
        linewidth=2.2,
        alpha=0.95,
        label="Desired trajectory",
        zorder=3,
    )
    ax.scatter(
        desired_u,
        desired_z,
        s=20,
        color="#f8fafc",
        edgecolors="#8cf7ff",
        linewidths=0.6,
        alpha=0.95,
        label="Desired tracked points",
        zorder=4,
    )
    ax.scatter(
        desired_u[:1],
        desired_z[:1],
        s=64,
        color="#f4d35e",
        edgecolors="none",
        label="Trajectory start",
        zorder=5,
    )

    all_u = np.concatenate([actual_u, desired_u, np.asarray([ref_u])])
    all_z = np.concatenate([actual_z, desired_z, np.asarray([ref_z])])
    span_u = max(float(np.max(np.abs(all_u - ref_u))), 1.0)
    span_z = max(float(np.max(np.abs(all_z - ref_z))), 1.0)
    span = 1.08 * max(span_u, span_z)

    ax.scatter(
        [ref_u],
        [ref_z],
        s=80,
        marker="x",
        color="#ff8fab",
        linewidths=2.0,
        label="Tracked mean reference",
        zorder=6,
    )

    ax.set_xlim(ref_u - span, ref_u + span)
    ax.set_ylim(ref_z - span, ref_z + span)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("u / desired X (mm)", color="#e8f3ff")
    ax.set_ylabel("z / desired tip Z (mm)", color="#e8f3ff")
    ax.set_title(
        f"{title_prefix}Desired tip trajectory vs tracked point density".strip(),
        color="#f8fbff",
    )
    ax.grid(True, color="#8eb8d8", alpha=0.12, linewidth=0.8)
    ax.tick_params(colors="#dceaf7")
    for spine in ax.spines.values():
        spine.set_color("#6b92b3")

    legend = ax.legend(frameon=True, facecolor="#102131", edgecolor="#6b92b3")
    for txt in legend.get_texts():
        txt.set_color("#f8fbff")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close()


def save_metrics_json(
    json_path: Path,
    metrics: Dict[str, Any],
    cal: CTR_Shadow_Calibration,
    args,
):
    payload = {
        "reference_point_mm": metrics["reference_point_mm"],
        "rmse_mm": metrics["rmse_mm"],
        "mean_error_mm": metrics["mean_error_mm"],
        "std_error_mm": metrics["std_error_mm"],
        "median_error_mm": metrics["median_error_mm"],
        "min_error_mm": metrics["min_error_mm"],
        "max_error_mm": metrics["max_error_mm"],
        "num_samples": metrics["num_samples"],
        "analysis_crop": getattr(cal, "analysis_crop", None),
        "board_reference": collect_board_reference_info(cal),
        "settings": {
            "threshold": int(args.threshold),
            "tip_refine_mode": str(args.tip_refine_mode),
            "tip_refine_dt_step_px": float(args.tip_refine_dt_step_px),
            "tip_refine_max_step_px": int(args.tip_refine_max_step_px),
            "tip_refine_grad_step_px": float(args.tip_refine_grad_step_px),
            "tip_refine_grad_search_len_px": int(args.tip_refine_grad_search_len_px),
            "hist_bins": int(args.hist_bins),
        },
    }
    with open(json_path, "w") as f:
        json.dump(_json_ready(payload), f, indent=2)


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
    ap.add_argument("--save_plots", action="store_true")

    ap.add_argument("--camera_calibration_file", type=str, required=True,
                    help="Path to camera calibration .npz for checkerboard-reference analysis.")
    ap.add_argument("--checkerboard_reference_image", type=str, required=True,
                    help="Path to checkerboard reference image.")
    ap.add_argument("--checkerboard_inner_corners", type=_parse_inner_corners_arg, default=None,
                    help="Checkerboard inner-corner grid as 'Nx,Ny' or 'NxXNy'. Defaults to metadata in the calibration file.")
    ap.add_argument("--checkerboard_square_size_mm", type=float, default=None,
                    help="Checkerboard square size in mm. Defaults to metadata in the calibration file.")
    ap.add_argument("--checkerboard_no_undistort", action="store_true",
                    help="Disable undistortion before checkerboard pose estimation.")

    ap.add_argument("--checkerboard_mm_scale_correction", type=float, default=0.5,
                    help="Multiply checkerboard-derived mm values by this factor. Default 0.5 fixes observed 2x overscale.")
    ap.add_argument("--checkerboard_no_flip_planar_x", action="store_true",
                    help="Disable checkerboard planar-x sign flip.")

    ap.add_argument("--link_mode", type=str, default="symlink", choices=["symlink", "copy"])
    ap.add_argument("--save_analysis_config", action="store_true")

    ap.add_argument("--tip_refine_mode", type=str, default="none",
                    choices=["none", "edge_dt", "edge_grad"],
                    help="Refine tip position toward distal edge of black segment.")
    ap.add_argument("--tip_refine_dt_step_px", type=float, default=1.0)
    ap.add_argument("--tip_refine_max_step_px", type=int, default=80)
    ap.add_argument("--tip_refine_grad_step_px", type=float, default=0.25)
    ap.add_argument("--tip_refine_grad_search_len_px", type=int, default=60)

    ap.add_argument("--hist_bins", type=int, default=20,
                    help="Number of histogram bins.")

    args = ap.parse_args()

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

    renamed_files = normalize_raw_filenames_for_shadow_parser(raw_folder)
    if renamed_files:
        print(f"[INFO] Normalized {len(renamed_files)} raw image filenames for shadow_calibration.")
        for old_name, new_name in renamed_files[:10]:
            print(f"  - {old_name} -> {new_name}")
        if len(renamed_files) > 10:
            print(f"  ... and {len(renamed_files) - 10} more")

    cal = CTR_Shadow_Calibration(
        parent_directory=str(project_dir.parent),
        project_name=project_dir.name,
        allow_existing=True,
        add_date=False,
    )
    cal.calibration_data_folder = str(project_dir)

    patch_filename_parser()

    if args.tip_refine_mode != "none":
        patch_analyze_data_for_tip_refinement(
            cal,
            refine_mode=str(args.tip_refine_mode),
            dt_step_px=float(args.tip_refine_dt_step_px),
            dt_max_step_px=int(args.tip_refine_max_step_px),
            grad_step_px=float(args.tip_refine_grad_step_px),
            grad_search_len_px=int(args.tip_refine_grad_search_len_px),
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

    calib_path = Path(args.camera_calibration_file).expanduser().resolve()
    board_ref_path = Path(args.checkerboard_reference_image).expanduser().resolve()

    print(f"[INFO] Loading camera calibration: {calib_path}")
    cal.load_camera_calibration(str(calib_path))

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

    analysis_crop = interactive_crop_from_image(
        img_bgr,
        default_crop=cal.default_analysis_crop,
    )
    cal.analysis_crop = dict(analysis_crop)

    if args.save_analysis_config:
        cfg = {}
        if hasattr(cal, "get_analysis_reference_info"):
            cfg = cal.get_analysis_reference_info()
        cfg["board_reference"] = collect_board_reference_info(cal)
        cfg_path = project_dir / "analysis_reference.json"
        with open(cfg_path, "w") as f:
            json.dump(_json_ready(cfg), f, indent=2)
        print(f"[INFO] Saved analysis config to: {cfg_path}")

    print("\n[INFO] Running analyze_data_batch (offline)...")
    cal.analyze_data_batch(threshold=int(args.threshold))

    tracked_rows = np.asarray(cal.tip_locations_array_coarse, dtype=float)

    if tracked_rows.size == 0:
        raise RuntimeError(
            "Batch analysis completed, but tip_locations_array_coarse is empty. "
            "Check filename parsing, image access, and tracking settings."
        )

    valid_mask = np.all(np.isfinite(tracked_rows[:, :2]), axis=1) if tracked_rows.ndim == 2 and tracked_rows.shape[1] >= 2 else np.zeros((0,), dtype=bool)
    num_valid = int(np.sum(valid_mask))

    print(f"[INFO] Total tracked rows: {tracked_rows.shape[0]}")
    print(f"[INFO] Valid tracked rows: {num_valid}")

    if num_valid == 0:
        raise RuntimeError(
            "Batch analysis completed, but no valid tracked tip points were produced. "
            "Filename parsing is now patched, so the next things to check are threshold/crop/tracking quality."
        )

    print("[INFO] Converting tracked tips to checkerboard-referenced mm...")
    tip_data = compute_tracked_tip_positions_mm(cal, tracked_rows, imgs)

    if tip_data["mm_points"].shape[0] == 0:
        raise RuntimeError(
            "Tracked rows exist, but none could be converted into checkerboard-referenced mm coordinates."
        )

    print("[INFO] Computing mean reference point and error metrics...")
    metrics = compute_error_metrics(tip_data["mm_points"])

    records = attach_errors_to_records(
        records=tip_data["records"],
        valid_indices=tip_data["valid_indices"],
        deltas_mm=metrics["deltas_mm"],
        errors_mm=metrics["errors_mm"],
    )

    csv_path = processed_dir / "tracked_tip_positions_mm.csv"
    metrics_json_path = processed_dir / "tracked_tip_error_metrics.json"
    error_plot_path = processed_dir / "tracked_tip_error_vs_sample.png"
    hist_path = processed_dir / "tracked_tip_error_histogram.png"
    desired_vs_actual_plot_path = processed_dir / "tracked_tip_desired_vs_actual.png"

    save_tracked_tip_csv(csv_path, records)
    save_metrics_json(metrics_json_path, metrics, cal, args)
    save_error_plot(
        error_plot_path,
        records,
        title_prefix="Checkerboard-referenced ",
    )
    save_error_histogram(
        hist_path,
        metrics["errors_mm"],
        rmse_mm=metrics["rmse_mm"],
        title_prefix="Checkerboard-referenced ",
        bins=int(args.hist_bins),
    )
    save_desired_vs_actual_plot(
        desired_vs_actual_plot_path,
        records,
        reference_point_mm=metrics["reference_point_mm"],
        title_prefix="Checkerboard-referenced ",
    )

    print(f"[INFO] Saved CSV: {csv_path}")
    print(f"[INFO] Saved metrics JSON: {metrics_json_path}")
    print(f"[INFO] Saved error plot: {error_plot_path}")
    print(f"[INFO] Saved histogram: {hist_path}")
    print(f"[INFO] Saved desired-vs-actual plot: {desired_vs_actual_plot_path}")

    if board_reference_debug_image is not None:
        try:
            annotated_board_path = processed_dir / "checkerboard_reference_annotated_analysis.png"
            draw_checkerboard_analysis_overlay(
                cal=cal,
                output_path=annotated_board_path,
                tracked_rows=tracked_rows,
                image_files=imgs,
                board_debug_image=board_reference_debug_image,
            )
            print(f"[INFO] Saved annotated checkerboard analysis image: {annotated_board_path}")
        except Exception as e:
            print(f"[WARN] Failed to create annotated checkerboard analysis image: {e}")

    print("\n========== RESULTS ==========")
    print(f"Valid samples: {metrics['num_samples']}")
    print(
        "Reference point (mean tracked point): "
        f"u = {metrics['reference_point_mm']['u_mean_mm']:.6f} mm, "
        f"z = {metrics['reference_point_mm']['z_mean_mm']:.6f} mm"
    )
    print(f"RMSE:         {metrics['rmse_mm']:.6f} mm")
    print(f"Mean error:   {metrics['mean_error_mm']:.6f} mm")
    print(f"Std error:    {metrics['std_error_mm']:.6f} mm")
    print(f"Median error: {metrics['median_error_mm']:.6f} mm")
    print(f"Min error:    {metrics['min_error_mm']:.6f} mm")
    print(f"Max error:    {metrics['max_error_mm']:.6f} mm")
    print("=============================\n")

    print("[DONE] Offline checkerboard tip error analysis complete.")
    print(f"Outputs are in: {processed_dir}")


if __name__ == "__main__":
    main()
