#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
offline_fixed_b_grid_desired_vs_measured_checkerboard_analysis.py

Adapted offline checkerboard-referenced tip analysis for fixed-B / fixed-tip-angle grid runs.

What it does:
  1) Opens a project/raw image folder
  2) Lets you choose crop bounds from the FIRST raw image
  3) Uses a checkerboard reference image + camera calibration to define mm axes
  4) Runs analyze_data_batch + existing tracking pipeline
  5) Converts tracked tip pixels to checkerboard-referenced mm coordinates
  6) Parses desired grid metadata from filenames:
       - tipX
       - tipY
       - tipZ
       - useA (used tip angle)
       - B
  7) Reconstructs desired grid coordinates from unique commanded X/Z values
     using user-specified step sizes (default: 5 mm in X and Z)
  8) For each used tip-angle pass:
       - aligns desired grid to measured points by CENTERING the two point sets
         using the mean of sampled points
       - computes per-point errors
       - computes per-angle RMSE
  9) Computes global RMSE across all valid aligned points
 10) Saves:
       - CSV of desired / aligned desired / measured / errors
       - JSON metrics summary
       - desired-vs-actual plot for each B tip angle
       - optional checkerboard overlay, including centroid alignment overlay

Important note:
  - The requested hard pre-capture time buffer of 5 seconds must be implemented
    in the IMAGE ACQUISITION script that triggers the camera, not in this offline
    analysis script. This file does not take pictures.
"""

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import types
from collections import defaultdict
from heapq import heappop, heappush
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np

# -----------------------------------------------------------------------------
# Acquisition note
# -----------------------------------------------------------------------------
# This constant is intentionally here as a reminder for the acquisition script.
# It is NOT enforceable in this offline analysis script because no images are captured here.
HARD_PRE_CAPTURE_BUFFER_SECONDS = 5.0

# Add path to your shadow_calibration script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

import shadow_calibration as shadow_calibration_module  # noqa: E402
from shadow_calibration import CTR_Shadow_Calibration  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_NEI8_W = [
    (-1, -1, 2 ** 0.5), (-1, 0, 1.0), (-1, 1, 2 ** 0.5),
    (0, -1, 1.0),                             (0, 1, 1.0),
    (1, -1, 2 ** 0.5),  (1, 0, 1.0),  (1, 1, 2 ** 0.5),
]


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
# Tip refinement helpers
# ------------------------------------------
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
    """
    if not hasattr(cal, "tip_refine_debug_records"):
        cal.tip_refine_debug_records = {}

    # Keep the default image processing path identical to shadow_calibration.py.
    # That file already performs coarse tip detection, parallel-centerline refinement,
    # and plot annotation inside CTR_Shadow_Calibration.analyze_data().
    if refine_mode == "parallel_centerline":
        cal.tip_refine_mode = "parallel_centerline"
        cal.tip_parallel_section_near_r = float(parallel_section_near_r)
        cal.tip_parallel_section_far_r = float(parallel_section_far_r)
        cal.tip_parallel_scan_half_r = float(parallel_scan_half_r)
        cal.tip_parallel_num_sections = int(parallel_num_sections)
        cal.tip_parallel_cross_step_px = float(parallel_cross_step_px)
        cal.tip_parallel_ray_step_px = float(parallel_ray_step_px)
        cal.tip_parallel_ray_max_len_r = float(parallel_ray_max_len_r)
        return cal

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
# Visualization helpers
# -----------------------------------------
def _extract_checkerboard_center_px(cal: CTR_Shadow_Calibration, board_debug_image: np.ndarray) -> np.ndarray:
    """
    Robustly infer checkerboard center in image pixels.
    Preference:
      1) mean of any detected/known checkerboard corner array found in board_pose
      2) origin_px
      3) image center
    """
    board_pose = getattr(cal, "board_pose", None)
    if isinstance(board_pose, dict):
        candidate_keys = [
            "corners_px",
            "board_corners_px",
            "image_points",
            "image_points_px",
            "detected_corners_px",
            "checkerboard_corners_px",
        ]
        for key in candidate_keys:
            arr = board_pose.get(key, None)
            if arr is None:
                continue
            arr = np.asarray(arr, dtype=np.float64)
            if arr.size >= 2:
                arr = arr.reshape(-1, 2)
                good = np.all(np.isfinite(arr), axis=1)
                if np.any(good):
                    return np.mean(arr[good], axis=0)

        origin = board_pose.get("origin_px", None)
        if origin is not None:
            origin = np.asarray(origin, dtype=np.float64).reshape(-1)
            if origin.size >= 2 and np.all(np.isfinite(origin[:2])):
                return origin[:2].copy()

    h, w = board_debug_image.shape[:2]
    return np.array([0.5 * (w - 1), 0.5 * (h - 1)], dtype=np.float64)


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

    # Raw measured points
    for pt in pts_xy_draw:
        px = int(round(pt[0]))
        py = int(round(pt[1]))
        if 0 <= px < w and 0 <= py < h:
            cv2.drawMarker(
                overlay,
                (px, py),
                (0, 255, 255),
                markerType=cv2.MARKER_TILTED_CROSS,
                markerSize=8,
                thickness=1,
                line_type=cv2.LINE_AA,
            )

    if pts_xy_draw.shape[0] >= 2:
        pts_poly = np.round(pts_xy_draw).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts_poly], False, (255, 200, 0), 1, lineType=cv2.LINE_AA)

    # Center alignment overlay: align centroid of measured points to checkerboard center
    measured_center = np.mean(pts_xy_draw, axis=0)
    checkerboard_center = _extract_checkerboard_center_px(cal, board_debug_image)
    center_shift = checkerboard_center - measured_center
    pts_xy_centered = pts_xy_draw + center_shift

    for pt in pts_xy_centered:
        px = int(round(pt[0]))
        py = int(round(pt[1]))
        if 0 <= px < w and 0 <= py < h:
            cv2.drawMarker(
                overlay,
                (px, py),
                (255, 0, 255),
                markerType=cv2.MARKER_CROSS,
                markerSize=8,
                thickness=1,
                line_type=cv2.LINE_AA,
            )

    if pts_xy_centered.shape[0] >= 2:
        pts_poly_centered = np.round(pts_xy_centered).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts_poly_centered], False, (255, 0, 255), 1, lineType=cv2.LINE_AA)

    # Draw centers
    mci = tuple(np.round(measured_center).astype(int))
    cci = tuple(np.round(checkerboard_center).astype(int))

    cv2.drawMarker(
        overlay, mci, (0, 255, 255),
        markerType=cv2.MARKER_STAR, markerSize=18, thickness=2, line_type=cv2.LINE_AA
    )
    cv2.putText(
        overlay, "measured points center", (mci[0] + 10, mci[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2
    )

    cv2.drawMarker(
        overlay, cci, (255, 0, 255),
        markerType=cv2.MARKER_STAR, markerSize=18, thickness=2, line_type=cv2.LINE_AA
    )
    cv2.putText(
        overlay, "checkerboard center", (cci[0] + 10, cci[1] - 10),
        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 2
    )

    cv2.arrowedLine(
        overlay,
        mci,
        cci,
        (180, 180, 180),
        2,
        tipLength=0.05
    )

    summary_lines = [
        f"tracked points: {len(valid_rows)}",
        f"local scale: {float(cal.board_px_per_mm_local):.4f} px/mm",
        f"raw measured center: ({measured_center[0]:.2f}, {measured_center[1]:.2f}) px",
        f"checkerboard center: ({checkerboard_center[0]:.2f}, {checkerboard_center[1]:.2f}) px",
        f"center shift applied: dx={center_shift[0]:.2f} px, dy={center_shift[1]:.2f} px",
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
# Filename metadata parsing
# -----------------------------------------
_FLOAT_RE = r"[-+]?\d+(?:\.\d+)?"

_FILENAME_PATTERNS = {
    "requested_tip_angle_deg": re.compile(r"_reqA(" + _FLOAT_RE + r")"),
    "used_tip_angle_deg": re.compile(r"_useA(" + _FLOAT_RE + r")"),
    "tip_x_cmd": re.compile(r"_tipX(" + _FLOAT_RE + r")"),
    "tip_y_cmd": re.compile(r"_tipY(" + _FLOAT_RE + r")"),
    "tip_z_cmd": re.compile(r"_tipZ(" + _FLOAT_RE + r")"),
    "stage_x_cmd": re.compile(r"_stageX(" + _FLOAT_RE + r")"),
    "stage_y_cmd": re.compile(r"_stageY(" + _FLOAT_RE + r")"),
    "stage_z_cmd": re.compile(r"_stageZ(" + _FLOAT_RE + r")"),
    "b_cmd": re.compile(r"_B(" + _FLOAT_RE + r")"),
    "c_cmd": re.compile(r"_C(" + _FLOAT_RE + r")"),
    "pass_index": re.compile(r"_pass(\d+)"),
    "sample_index": re.compile(r"^(\d+)"),
}


def _extract_tagged_float_from_tokens(name: str, tag: str) -> Optional[float]:
    stem = Path(name).stem
    for token in stem.split("_"):
        if token.startswith(tag):
            suffix = token[len(tag):]
            if not suffix:
                continue
            try:
                return float(suffix)
            except Exception:
                continue
    return None


def _extract_tagged_int_from_tokens(name: str, tag: str) -> Optional[int]:
    stem = Path(name).stem
    for token in stem.split("_"):
        if token.startswith(tag):
            suffix = token[len(tag):]
            if not suffix:
                continue
            try:
                return int(suffix)
            except Exception:
                continue
    return None


def _extract_float_from_name(name: str, pattern: re.Pattern) -> Optional[float]:
    m = pattern.search(name)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _extract_int_from_name(name: str, pattern: re.Pattern) -> Optional[int]:
    m = pattern.search(name)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def parse_image_metadata_from_name(image_name: str) -> Dict[str, Any]:
    stem = Path(image_name).stem

    def pick_float(pattern_key: str, token_tag: str) -> Optional[float]:
        value = _extract_float_from_name(stem, _FILENAME_PATTERNS[pattern_key])
        if value is not None:
            return value
        return _extract_tagged_float_from_tokens(stem, token_tag)

    def pick_int(pattern_key: str, token_tag: str) -> Optional[int]:
        value = _extract_int_from_name(stem, _FILENAME_PATTERNS[pattern_key])
        if value is not None:
            return value
        return _extract_tagged_int_from_tokens(stem, token_tag)

    meta = {
        "image_name": image_name,
        "requested_tip_angle_deg": pick_float("requested_tip_angle_deg", "reqA"),
        "used_tip_angle_deg": pick_float("used_tip_angle_deg", "useA"),
        "tip_x_cmd": pick_float("tip_x_cmd", "tipX"),
        "tip_y_cmd": pick_float("tip_y_cmd", "tipY"),
        "tip_z_cmd": pick_float("tip_z_cmd", "tipZ"),
        "stage_x_cmd": pick_float("stage_x_cmd", "stageX"),
        "stage_y_cmd": pick_float("stage_y_cmd", "stageY"),
        "stage_z_cmd": pick_float("stage_z_cmd", "stageZ"),
        "b_cmd": pick_float("b_cmd", "B"),
        "c_cmd": pick_float("c_cmd", "C"),
        "pass_index": pick_int("pass_index", "pass"),
        "sample_index_from_name": _extract_int_from_name(stem, _FILENAME_PATTERNS["sample_index"]),
    }
    return meta


def robust_parse_positions_from_filename(file_name: str):
    """
    Parse the filename format produced by calib_cartes_daq.py and fall back to
    the legacy orientation/X/B format expected by shadow_calibration.

    Returns:
      orientation, ntnl_pos, ss_pos

    Mapping used for shadow_calibration compatibility:
      - orientation = inferred from side token / C value / legacy numeric token
      - ntnl_pos    = tipX when available, otherwise X
      - ss_pos      = B
    """
    base = os.path.splitext(os.path.basename(file_name))[0]
    parts = base.split("_")

    orientation = None
    side_token = None
    values: Dict[str, float] = {}

    for token in parts:
        lower = token.lower()
        if lower in ("right", "left"):
            side_token = lower
            continue

        for key in ("tipX", "tipY", "tipZ", "stageX", "stageY", "stageZ", "reqA", "useA"):
            if token.startswith(key):
                try:
                    values[key] = float(token[len(key):])
                except Exception:
                    pass
                break
        else:
            if len(token) >= 2 and token[0] in ("X", "Y", "Z", "B", "C"):
                try:
                    values[token[0]] = float(token[1:])
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

    if orientation is None and parts:
        try:
            candidate = int(parts[0])
            if candidate in (0, 1, 2, 3):
                orientation = candidate
        except Exception:
            pass

    if orientation is None:
        orientation = 0

    x_value = values.get("tipX", values.get("X"))
    b_value = values.get("B")
    if x_value is None:
        raise ValueError(f"Could not parse X value from filename: {file_name}")
    if b_value is None:
        raise ValueError(f"Could not parse B value from filename: {file_name}")

    return int(orientation), float(x_value), float(b_value)


def patch_filename_parser():
    """
    Patch shadow_calibration filename parsing so analyze_data_batch can read
    Cartesian-tracking filenames like:
      00043_pass01_reqA0.0_useA15.653_tipX100.000_..._B-0.095_C0.000_...
    """
    shadow_calibration_module._get_pos_from_file_name = robust_parse_positions_from_filename
    print("[INFO] Patched filename parser for Cartesian tipX..._B... filenames.")


def angle_group_key(angle_deg: Any, decimals: int = 6) -> Optional[float]:
    if angle_deg is None:
        return None
    try:
        v = float(angle_deg)
        if not np.isfinite(v):
            return None
        return round(v, decimals)
    except Exception:
        return None


# -----------------------------------------
# Convert tracked tips to checkerboard mm
# and attach desired commanded coordinates
# -----------------------------------------
def compute_tracked_tip_positions_mm_with_targets(
    cal: CTR_Shadow_Calibration,
    tracked_rows: np.ndarray,
    image_files: List[Path],
) -> Dict[str, Any]:
    """
    Convert tracked tip pixel coordinates to checkerboard-referenced mm
    and attach filename-derived target metadata.
    """
    if getattr(cal, "board_pose", None) is None:
        raise RuntimeError("Checkerboard board_pose is not available.")
    if tracked_rows is None or tracked_rows.size == 0:
        raise RuntimeError("tracked_rows is empty.")

    origin = np.asarray(cal.board_pose["origin_px"], dtype=np.float64).reshape(2)

    records = []
    valid_indices = []
    measured_mm_points = []

    for i, row in enumerate(np.asarray(tracked_rows, dtype=float)):
        file_name = image_files[i].name if i < len(image_files) else f"sample_{i:04d}"
        meta = parse_image_metadata_from_name(file_name)

        y_px = None
        x_px = None
        u_mm = None
        z_mm = None
        valid = False

        if np.all(np.isfinite(row[:2])):
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
                u_mm, z_mm = None, None
                valid = False

        rec = {
            "sample_index": i,
            "image_name": file_name,
            "tip_y_px": y_px,
            "tip_x_px": x_px,
            "u_mm": u_mm,
            "z_mm": z_mm,
            "valid": bool(valid),

            "requested_tip_angle_deg": meta["requested_tip_angle_deg"],
            "used_tip_angle_deg": meta["used_tip_angle_deg"],
            "tip_x_cmd": meta["tip_x_cmd"],
            "tip_y_cmd": meta["tip_y_cmd"],
            "tip_z_cmd": meta["tip_z_cmd"],
            "stage_x_cmd": meta["stage_x_cmd"],
            "stage_y_cmd": meta["stage_y_cmd"],
            "stage_z_cmd": meta["stage_z_cmd"],
            "b_cmd": meta["b_cmd"],
            "c_cmd": meta["c_cmd"],
            "pass_index": meta["pass_index"],
            "sample_index_from_name": meta["sample_index_from_name"],
        }
        records.append(rec)

        if valid:
            valid_indices.append(i)
            measured_mm_points.append([u_mm, z_mm])

    measured_mm_points = (
        np.asarray(measured_mm_points, dtype=float)
        if len(measured_mm_points) else np.empty((0, 2), dtype=float)
    )

    return {
        "records": records,
        "valid_indices": valid_indices,
        "measured_mm_points": measured_mm_points,
    }


# -----------------------------------------
# Desired grid reconstruction + alignment
# -----------------------------------------
def build_desired_grid_coordinates_per_angle(
    records: List[Dict[str, Any]],
    x_step_mm: float = 5.0,
    z_step_mm: float = 5.0,
):
    """
    For each used tip-angle pass, map commanded tipX/tipZ values to desired grid coordinates:
      desired_grid_u_mm = column_index * x_step_mm
      desired_grid_z_mm = row_index    * z_step_mm

    This removes dependence on absolute commanded origin and uses only spacing/order.
    """
    groups = defaultdict(list)
    for rec in records:
        ang = angle_group_key(rec.get("used_tip_angle_deg"))
        if ang is None:
            continue
        tx = rec.get("tip_x_cmd")
        tz = rec.get("tip_z_cmd")
        if tx is None or tz is None:
            continue
        groups[ang].append(rec)

    def infer_axis_sort_descending(
        grecs: List[Dict[str, Any]],
        cmd_key: str,
        measured_key: str,
    ) -> bool:
        """
        Infer whether commanded values should be mapped in descending order so the
        reconstructed desired grid follows the measured checkerboard axis direction.

        If commanded values increase while measured coordinates decrease, the desired
        index assignment must be reversed.
        """
        pairs = [
            (float(r[cmd_key]), float(r[measured_key]))
            for r in grecs
            if r.get("valid", False)
            and r.get(cmd_key) is not None
            and r.get(measured_key) is not None
        ]
        if len(pairs) < 2:
            return False

        cmd_vals = np.asarray([p[0] for p in pairs], dtype=float)
        measured_vals = np.asarray([p[1] for p in pairs], dtype=float)

        if np.allclose(cmd_vals, cmd_vals[0]) or np.allclose(measured_vals, measured_vals[0]):
            return False

        corr = np.corrcoef(cmd_vals, measured_vals)[0, 1]
        if not np.isfinite(corr):
            return False
        return bool(corr < 0.0)

    group_grid_meta = {}

    for ang, grecs in groups.items():
        x_vals_raw = sorted({float(r["tip_x_cmd"]) for r in grecs if r.get("tip_x_cmd") is not None})
        z_vals_raw = sorted({float(r["tip_z_cmd"]) for r in grecs if r.get("tip_z_cmd") is not None})

        reverse_x = infer_axis_sort_descending(grecs, "tip_x_cmd", "u_mm")
        reverse_z = infer_axis_sort_descending(grecs, "tip_z_cmd", "z_mm")

        x_vals = sorted(x_vals_raw, reverse=reverse_x)
        z_vals = sorted(z_vals_raw, reverse=reverse_z)

        x_map = {x: ix * float(x_step_mm) for ix, x in enumerate(x_vals)}
        z_map = {z: iz * float(z_step_mm) for iz, z in enumerate(z_vals)}

        group_grid_meta[ang] = {
            "raw_tip_x_values": x_vals_raw,
            "raw_tip_z_values": z_vals_raw,
            "ordered_tip_x_values": x_vals,
            "ordered_tip_z_values": z_vals,
            "desired_grid_x_map": x_map,
            "desired_grid_z_map": z_map,
            "desired_grid_x_descending": bool(reverse_x),
            "desired_grid_z_descending": bool(reverse_z),
            "num_x": len(x_vals),
            "num_z": len(z_vals),
            "x_step_mm": float(x_step_mm),
            "z_step_mm": float(z_step_mm),
        }

        for rec in grecs:
            tx = float(rec["tip_x_cmd"])
            tz = float(rec["tip_z_cmd"])
            rec["desired_grid_u_mm"] = float(x_map[tx])
            rec["desired_grid_z_mm"] = float(z_map[tz])

    return records, group_grid_meta


def align_desired_to_measured_per_angle(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[float, Dict[str, Any]]]:
    """
    For each used tip-angle pass:
      aligned_desired_u_mm = desired_grid_u_mm + offset_u
      aligned_desired_z_mm = desired_grid_z_mm + offset_z

    Alignment is center-based using means of sampled points:
      measured_center_u = mean(measured_u)
      measured_center_z = mean(measured_z)
      desired_center_u  = mean(desired_u)
      desired_center_z  = mean(desired_z)

      offset_u = measured_center_u - desired_center_u
      offset_z = measured_center_z - desired_center_z
    """
    grouped = defaultdict(list)
    for rec in records:
        ang = angle_group_key(rec.get("used_tip_angle_deg"))
        if ang is None:
            continue
        grouped[ang].append(rec)

    alignments = {}

    for ang, grecs in grouped.items():
        valid = [
            r for r in grecs
            if r.get("valid", False)
            and r.get("u_mm") is not None
            and r.get("z_mm") is not None
            and r.get("desired_grid_u_mm") is not None
            and r.get("desired_grid_z_mm") is not None
        ]

        if not valid:
            alignments[ang] = {
                "used_tip_angle_deg": float(ang),
                "offset_u_mm": None,
                "offset_z_mm": None,
                "num_valid_points": 0,
                "measured_center_u_mm": None,
                "measured_center_z_mm": None,
                "desired_center_u_mm": None,
                "desired_center_z_mm": None,
            }
            for rec in grecs:
                rec["aligned_desired_u_mm"] = None
                rec["aligned_desired_z_mm"] = None
            continue

        measured_u = np.asarray([float(r["u_mm"]) for r in valid], dtype=float)
        measured_z = np.asarray([float(r["z_mm"]) for r in valid], dtype=float)
        desired_u = np.asarray([float(r["desired_grid_u_mm"]) for r in valid], dtype=float)
        desired_z = np.asarray([float(r["desired_grid_z_mm"]) for r in valid], dtype=float)

        measured_center_u = float(np.mean(measured_u))
        measured_center_z = float(np.mean(measured_z))
        desired_center_u = float(np.mean(desired_u))
        desired_center_z = float(np.mean(desired_z))

        offset_u = float(measured_center_u - desired_center_u)
        offset_z = float(measured_center_z - desired_center_z)

        alignments[ang] = {
            "used_tip_angle_deg": float(ang),
            "offset_u_mm": offset_u,
            "offset_z_mm": offset_z,
            "num_valid_points": int(len(valid)),
            "measured_center_u_mm": measured_center_u,
            "measured_center_z_mm": measured_center_z,
            "desired_center_u_mm": desired_center_u,
            "desired_center_z_mm": desired_center_z,
        }

        for rec in grecs:
            du = rec.get("desired_grid_u_mm")
            dz = rec.get("desired_grid_z_mm")
            rec["aligned_desired_u_mm"] = None if du is None else float(du) + offset_u
            rec["aligned_desired_z_mm"] = None if dz is None else float(dz) + offset_z

    return records, alignments


def compute_errors_against_aligned_desired(
    records: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Adds:
      du_error_mm = measured_u - aligned_desired_u
      dz_error_mm = measured_z - aligned_desired_z
      error_distance_mm = sqrt(du^2 + dz^2)

    Returns updated records and summary metrics:
      - global_rmse_mm
      - per_angle_rmse_mm
    """
    grouped = defaultdict(list)

    for rec in records:
        u = rec.get("u_mm")
        z = rec.get("z_mm")
        du_t = rec.get("aligned_desired_u_mm")
        dz_t = rec.get("aligned_desired_z_mm")

        valid_cmp = (
            rec.get("valid", False)
            and u is not None and z is not None
            and du_t is not None and dz_t is not None
        )

        if valid_cmp:
            du_err = float(u) - float(du_t)
            dz_err = float(z) - float(dz_t)
            dist = float(np.hypot(du_err, dz_err))
        else:
            du_err = None
            dz_err = None
            dist = None

        rec["du_error_mm"] = du_err
        rec["dz_error_mm"] = dz_err
        rec["error_distance_mm"] = dist

        ang = angle_group_key(rec.get("used_tip_angle_deg"))
        if ang is not None:
            grouped[ang].append(rec)

    all_err = np.asarray(
        [float(r["error_distance_mm"]) for r in records if r.get("error_distance_mm") is not None],
        dtype=float
    )

    if all_err.size == 0:
        raise RuntimeError("No valid aligned points available for RMSE calculation.")

    global_rmse_mm = float(np.sqrt(np.mean(all_err ** 2)))
    global_mean_err_mm = float(np.mean(all_err))
    global_std_err_mm = float(np.std(all_err))
    global_median_err_mm = float(np.median(all_err))
    global_min_err_mm = float(np.min(all_err))
    global_max_err_mm = float(np.max(all_err))

    per_angle = {}
    for ang, grecs in grouped.items():
        errs = np.asarray(
            [float(r["error_distance_mm"]) for r in grecs if r.get("error_distance_mm") is not None],
            dtype=float
        )
        if errs.size == 0:
            per_angle[ang] = {
                "used_tip_angle_deg": float(ang),
                "num_samples": 0,
                "rmse_mm": None,
                "mean_error_mm": None,
                "std_error_mm": None,
                "median_error_mm": None,
                "min_error_mm": None,
                "max_error_mm": None,
            }
            continue

        per_angle[ang] = {
            "used_tip_angle_deg": float(ang),
            "num_samples": int(errs.size),
            "rmse_mm": float(np.sqrt(np.mean(errs ** 2))),
            "mean_error_mm": float(np.mean(errs)),
            "std_error_mm": float(np.std(errs)),
            "median_error_mm": float(np.median(errs)),
            "min_error_mm": float(np.min(errs)),
            "max_error_mm": float(np.max(errs)),
        }

    metrics = {
        "global_num_samples": int(all_err.size),
        "global_rmse_mm": global_rmse_mm,
        "global_mean_error_mm": global_mean_err_mm,
        "global_std_error_mm": global_std_err_mm,
        "global_median_error_mm": global_median_err_mm,
        "global_min_error_mm": global_min_err_mm,
        "global_max_error_mm": global_max_err_mm,
        "per_angle_metrics": per_angle,
    }
    return records, metrics


# -----------------------------------------
# Saving outputs
# -----------------------------------------
def save_desired_vs_measured_csv(csv_path: Path, records: List[Dict[str, Any]]):
    fieldnames = [
        "sample_index",
        "sample_index_from_name",
        "image_name",
        "pass_index",
        "requested_tip_angle_deg",
        "used_tip_angle_deg",
        "b_cmd",
        "c_cmd",
        "tip_x_cmd",
        "tip_y_cmd",
        "tip_z_cmd",
        "stage_x_cmd",
        "stage_y_cmd",
        "stage_z_cmd",
        "desired_grid_u_mm",
        "desired_grid_z_mm",
        "aligned_desired_u_mm",
        "aligned_desired_z_mm",
        "tip_y_px",
        "tip_x_px",
        "u_mm",
        "z_mm",
        "du_error_mm",
        "dz_error_mm",
        "error_distance_mm",
        "valid",
    ]
    extra_fields = sorted({
        key
        for record in records
        for key in record.keys()
        if key not in fieldnames
    })
    fieldnames.extend(extra_fields)

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def save_metrics_json(
    json_path: Path,
    metrics: Dict[str, Any],
    alignments: Dict[float, Dict[str, Any]],
    grid_meta: Dict[float, Dict[str, Any]],
    cal: CTR_Shadow_Calibration,
    args,
):
    payload = {
        "global_metrics": {
            "num_samples": metrics["global_num_samples"],
            "rmse_mm": metrics["global_rmse_mm"],
            "mean_error_mm": metrics["global_mean_error_mm"],
            "std_error_mm": metrics["global_std_error_mm"],
            "median_error_mm": metrics["global_median_error_mm"],
            "min_error_mm": metrics["global_min_error_mm"],
            "max_error_mm": metrics["global_max_error_mm"],
        },
        "per_angle_metrics": metrics["per_angle_metrics"],
        "per_angle_alignments": alignments,
        "per_angle_grid_meta": grid_meta,
        "analysis_crop": getattr(cal, "analysis_crop", None),
        "board_reference": collect_board_reference_info(cal),
        "settings": {
            "threshold": int(args.threshold),
            "tip_refine_mode": str(args.tip_refine_mode),
            "tip_refine_dt_step_px": float(args.tip_refine_dt_step_px),
            "tip_refine_max_step_px": int(args.tip_refine_max_step_px),
            "tip_refine_grad_step_px": float(args.tip_refine_grad_step_px),
            "tip_refine_grad_search_len_px": int(args.tip_refine_grad_search_len_px),
            "tip_refine_mainray_fit_back_near_r": float(args.tip_refine_mainray_fit_back_near_r),
            "tip_refine_mainray_fit_back_far_r": float(args.tip_refine_mainray_fit_back_far_r),
            "tip_refine_mainray_anchor_back_r": float(args.tip_refine_mainray_anchor_back_r),
            "tip_refine_mainray_ray_step_px": float(args.tip_refine_mainray_ray_step_px),
            "tip_refine_mainray_ray_max_len_r": float(args.tip_refine_mainray_ray_max_len_r),
            "tip_refine_parallel_section_near_r": float(args.tip_refine_parallel_section_near_r),
            "tip_refine_parallel_section_far_r": float(args.tip_refine_parallel_section_far_r),
            "tip_refine_parallel_scan_half_r": float(args.tip_refine_parallel_scan_half_r),
            "tip_refine_parallel_num_sections": int(args.tip_refine_parallel_num_sections),
            "tip_refine_parallel_cross_step_px": float(args.tip_refine_parallel_cross_step_px),
            "tip_refine_parallel_ray_step_px": float(args.tip_refine_parallel_ray_step_px),
            "tip_refine_parallel_ray_max_len_r": float(args.tip_refine_parallel_ray_max_len_r),
            "desired_x_step_mm": float(args.desired_x_step_mm),
            "desired_z_step_mm": float(args.desired_z_step_mm),
            "hard_pre_capture_buffer_seconds_note_for_acquisition_script": float(HARD_PRE_CAPTURE_BUFFER_SECONDS),
        },
    }
    with open(json_path, "w") as f:
        json.dump(_json_ready(payload), f, indent=2)


def save_desired_vs_actual_plot(
    plot_path: Path,
    records: List[Dict[str, Any]],
    metrics: Dict[str, Any],
    title_prefix: str = "",
):
    def style_ax(ax, title: str):
        ax.set_title(title, color="#f4f8fb")
        ax.set_xlabel("Horizontal / u (mm)", color="#e5f1fb")
        ax.set_ylabel("Vertical / z (mm)", color="#e5f1fb")
        ax.tick_params(colors="#d7e9f8")
        for spine in ax.spines.values():
            spine.set_color((0.78, 0.88, 0.96, 0.55))
        ax.grid(True, alpha=0.14, color="#b7d3eb", linewidth=0.8)
        ax.set_facecolor((0.0, 0.0, 0.0, 0.0))

    grouped = defaultdict(list)
    for rec in records:
        ang = angle_group_key(rec.get("used_tip_angle_deg"))
        if ang is not None:
            grouped[ang].append(rec)

    angle_keys = sorted(grouped.keys())
    if not angle_keys:
        raise RuntimeError("No angle groups available for plotting.")

    # Stack vertically as requested
    n = len(angle_keys)
    ncols = 1
    nrows = n

    fig, axs = plt.subplots(nrows, ncols, figsize=(8.2, 5.2 * nrows), squeeze=False)
    fig.patch.set_alpha(0.0)

    for ax in axs.flat:
        ax.set_visible(False)

    for idx, ang in enumerate(angle_keys):
        ax = axs.flat[idx]
        ax.set_visible(True)

        grecs = grouped[ang]
        valid = [
            r for r in grecs
            if r.get("valid", False)
            and r.get("u_mm") is not None
            and r.get("z_mm") is not None
            and r.get("aligned_desired_u_mm") is not None
            and r.get("aligned_desired_z_mm") is not None
        ]
        if not valid:
            style_ax(ax, f"Used tip angle {ang:.3f}°\n(no valid data)")
            continue

        desired_u = np.asarray([float(r["aligned_desired_u_mm"]) for r in valid], dtype=float)
        desired_z = np.asarray([float(r["aligned_desired_z_mm"]) for r in valid], dtype=float)
        measured_u = np.asarray([float(r["u_mm"]) for r in valid], dtype=float)
        measured_z = np.asarray([float(r["z_mm"]) for r in valid], dtype=float)

        for r in valid:
            ax.plot(
                [float(r["aligned_desired_u_mm"]), float(r["u_mm"])],
                [float(r["aligned_desired_z_mm"]), float(r["z_mm"])],
                linewidth=0.8,
                alpha=0.45,
                color=(0.61, 0.82, 0.96, 0.34),
            )

        # Desired shown as rectangle centers / square markers
        ax.scatter(
            desired_u, desired_z,
            s=46,
            marker="s",
            facecolors="none",
            edgecolors="#8ae1ff",
            linewidths=1.5,
            label="Reference (aligned centers)",
        )

        # Measured points smaller and as points, not circles
        ax.scatter(
            measured_u, measured_z,
            s=14,
            marker=".",
            color="#ffd166",
            label="Measured",
        )

        # Draw centers explicitly
        desired_center = np.array([np.mean(desired_u), np.mean(desired_z)], dtype=float)
        measured_center = np.array([np.mean(measured_u), np.mean(measured_z)], dtype=float)

        ax.scatter(
            [desired_center[0]], [desired_center[1]],
            s=60, marker="s", facecolors="none", edgecolors="#00e5ff", linewidths=2.0,
            label="Reference center",
        )
        ax.scatter(
            [measured_center[0]], [measured_center[1]],
            s=36, marker="+", color="#ff5ea8", linewidths=1.8,
            label="Measured center",
        )
        ax.plot(
            [desired_center[0], measured_center[0]],
            [desired_center[1], measured_center[1]],
            linestyle="--",
            linewidth=1.0,
            alpha=0.7,
            color="#ff5ea8",
        )

        per_angle = metrics["per_angle_metrics"].get(ang, {})
        rmse_txt = "n/a" if per_angle.get("rmse_mm") is None else f"{float(per_angle['rmse_mm']):.4f} mm"

        b_vals = [r.get("b_cmd") for r in valid if r.get("b_cmd") is not None]
        b_txt = f"{float(np.median(np.asarray(b_vals, dtype=float))):.3f}" if b_vals else "n/a"

        style_ax(
            ax,
            f"{title_prefix}Used angle {ang:.3f}°  |  B={b_txt}\n"
            f"RMSE = {rmse_txt}  |  n = {len(valid)}",
        )
        ax.set_aspect("equal", adjustable="box")
        leg = ax.legend(loc="best", frameon=True)
        leg.get_frame().set_facecolor((0.04, 0.09, 0.14, 0.72))
        leg.get_frame().set_edgecolor((0.72, 0.84, 0.94, 0.35))
        for txt in leg.get_texts():
            txt.set_color("#edf6ff")

        all_u = np.concatenate([desired_u, measured_u])
        all_z = np.concatenate([desired_z, measured_z])
        pad_u = max(2.0, 0.06 * max(1.0, np.ptp(all_u)))
        pad_z = max(2.0, 0.06 * max(1.0, np.ptp(all_z)))
        ax.set_xlim(float(np.min(all_u) - pad_u), float(np.max(all_u) + pad_u))
        ax.set_ylim(float(np.min(all_z) - pad_z), float(np.max(all_z) + pad_z))

    global_rmse = float(metrics["global_rmse_mm"])
    fig.suptitle(
        f"{title_prefix}Desired vs measured tracked tip points by used tip angle\n"
        f"Global RMSE = {global_rmse:.4f} mm",
        fontsize=14,
        y=0.995,
        color="#f8fbff",
    )
    fig.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(plot_path, dpi=220, transparent=True)
    plt.close(fig)


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
                    choices=["none", "edge_dt", "edge_grad", "mainray", "parallel_centerline"],
                    help="Refine tip position using the same distal tip analysis modes as offline_run_calibration.py.")
    ap.add_argument("--tip_refine_dt_step_px", type=float, default=1.0)
    ap.add_argument("--tip_refine_max_step_px", type=int, default=80)
    ap.add_argument("--tip_refine_grad_step_px", type=float, default=0.25)
    ap.add_argument("--tip_refine_grad_search_len_px", type=int, default=60)
    ap.add_argument("--tip_refine_mainray_fit_back_near_r", type=float, default=1.5)
    ap.add_argument("--tip_refine_mainray_fit_back_far_r", type=float, default=6.0)
    ap.add_argument("--tip_refine_mainray_anchor_back_r", type=float, default=1.0)
    ap.add_argument("--tip_refine_mainray_ray_step_px", type=float, default=0.5)
    ap.add_argument("--tip_refine_mainray_ray_max_len_r", type=float, default=8.0)
    ap.add_argument("--tip_refine_parallel_section_near_r", type=float, default=1.0)
    ap.add_argument("--tip_refine_parallel_section_far_r", type=float, default=6.0)
    ap.add_argument("--tip_refine_parallel_scan_half_r", type=float, default=3.0)
    ap.add_argument("--tip_refine_parallel_num_sections", type=int, default=9)
    ap.add_argument("--tip_refine_parallel_cross_step_px", type=float, default=0.5)
    ap.add_argument("--tip_refine_parallel_ray_step_px", type=float, default=0.5)
    ap.add_argument("--tip_refine_parallel_ray_max_len_r", type=float, default=8.0)

    ap.add_argument("--desired_x_step_mm", type=float, default=5.0,
                    help="Desired horizontal step between commanded grid columns.")
    ap.add_argument("--desired_z_step_mm", type=float, default=5.0,
                    help="Desired vertical step between commanded grid rows.")

    args = ap.parse_args()
    print(f"[INFO] Using shadow_calibration module: {shadow_calibration_module.__file__}")
    print(
        "[INFO] Requested hard pre-capture buffer = "
        f"{HARD_PRE_CAPTURE_BUFFER_SECONDS:.1f} s (must be enforced in the acquisition script, not here)."
    )

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
            mainray_fit_back_near_r=float(args.tip_refine_mainray_fit_back_near_r),
            mainray_fit_back_far_r=float(args.tip_refine_mainray_fit_back_far_r),
            mainray_anchor_back_r=float(args.tip_refine_mainray_anchor_back_r),
            mainray_ray_step_px=float(args.tip_refine_mainray_ray_step_px),
            mainray_ray_max_len_r=float(args.tip_refine_mainray_ray_max_len_r),
            parallel_section_near_r=float(args.tip_refine_parallel_section_near_r),
            parallel_section_far_r=float(args.tip_refine_parallel_section_far_r),
            parallel_scan_half_r=float(args.tip_refine_parallel_scan_half_r),
            parallel_num_sections=int(args.tip_refine_parallel_num_sections),
            parallel_cross_step_px=float(args.tip_refine_parallel_cross_step_px),
            parallel_ray_step_px=float(args.tip_refine_parallel_ray_step_px),
            parallel_ray_max_len_r=float(args.tip_refine_parallel_ray_max_len_r),
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
    board_type = str(board_result.get("board_type", getattr(cal, "camera_calib_meta", {}).get("board_type", "checkerboard"))).lower()

    print(f"[INFO] Applying checkerboard-style reference correction to board_type={board_type}")
    apply_checkerboard_reference_corrections(
        cal,
        mm_scale=float(args.checkerboard_mm_scale_correction),
        flip_planar_x=(not bool(args.checkerboard_no_flip_planar_x)),
    )
    if isinstance(board_result, dict):
        board_result["board_reference_correction_meta"] = getattr(cal, "board_reference_correction_meta", None)
        board_result["board_reference_correction_source_board_type"] = board_type

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

    patch_filename_parser()

    print("\n[INFO] Running analyze_data_batch (offline)...")
    cal.analyze_data_batch(threshold=int(args.threshold))

    tracked_rows = np.asarray(cal.tip_locations_array_coarse, dtype=float)
    if tracked_rows.size == 0:
        raise RuntimeError("No tracked tip data found in cal.tip_locations_array_coarse after analyze_data_batch.")

    print("[INFO] Converting tracked tips to checkerboard-referenced mm and parsing filename targets...")
    tip_data = compute_tracked_tip_positions_mm_with_targets(cal, tracked_rows, imgs)

    print("[INFO] Reconstructing desired grid coordinates from commanded tipX/tipZ...")
    records, grid_meta = build_desired_grid_coordinates_per_angle(
        tip_data["records"],
        x_step_mm=float(args.desired_x_step_mm),
        z_step_mm=float(args.desired_z_step_mm),
    )

    print("[INFO] Aligning reference and measured point sets by the mean center of sampled points...")
    records, alignments = align_desired_to_measured_per_angle(records)

    print("[INFO] Computing global and per-angle RMSE...")
    records, metrics = compute_errors_against_aligned_desired(records)

    csv_path = processed_dir / "tracked_tip_positions_desired_vs_measured_mm.csv"
    metrics_json_path = processed_dir / "tracked_tip_desired_vs_measured_metrics.json"
    plot_path = processed_dir / "desired_vs_actual_by_angle.png"

    save_desired_vs_measured_csv(csv_path, records)
    save_metrics_json(metrics_json_path, metrics, alignments, grid_meta, cal, args)
    save_desired_vs_actual_plot(
        plot_path,
        records,
        metrics,
        title_prefix="Checkerboard-referenced ",
    )

    print(f"[INFO] Saved CSV: {csv_path}")
    print(f"[INFO] Saved metrics JSON: {metrics_json_path}")
    print(f"[INFO] Saved desired-vs-actual plot: {plot_path}")

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
    print(f"Valid samples: {metrics['global_num_samples']}")
    print(f"Global RMSE:   {metrics['global_rmse_mm']:.6f} mm")
    print(f"Mean error:    {metrics['global_mean_error_mm']:.6f} mm")
    print(f"Std error:     {metrics['global_std_error_mm']:.6f} mm")
    print(f"Median error:  {metrics['global_median_error_mm']:.6f} mm")
    print(f"Min error:     {metrics['global_min_error_mm']:.6f} mm")
    print(f"Max error:     {metrics['global_max_error_mm']:.6f} mm")

    print("\nPer-angle RMSE:")
    for ang in sorted(metrics["per_angle_metrics"].keys()):
        m = metrics["per_angle_metrics"][ang]
        if m["rmse_mm"] is None:
            print(f"  Used angle {ang:.3f} deg: no valid data")
        else:
            print(
                f"  Used angle {ang:.3f} deg: "
                f"RMSE = {m['rmse_mm']:.6f} mm, "
                f"n = {m['num_samples']}"
            )
    print("=============================\n")

    print("[DONE] Offline desired-vs-measured checkerboard analysis complete.")
    print(f"Outputs are in: {processed_dir}")


if __name__ == "__main__":
    main()
