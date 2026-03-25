#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
offline_dual_c_b_hysteresis_checkerboard_analysis.py

Offline hysteresis analysis for the fixed-gantry / B-only curl run.

Goal
----
For each C orientation block (typically C=0 and C=180):
  - track the tip in checkerboard-referenced mm coordinates
  - split the measured trajectory into:
        * curling
        * uncurling
    based on local B-direction
  - overlay those two measured trajectories on the same subplot
  - overlay the predicted trajectory from the calibration polynomial equations
    for tip position vs B
  - for the mirrored view, C=180 is reflected about the vertical axis so it is
    visually comparable to C=0

Main outputs
------------
processed_image_data_folder/
  tracked_tip_positions_mm.csv
  b_hysteresis_metrics.json
  b_hysteresis_vs_sample.png
  b_hysteresis_dual_c_overlay.png
  checkerboard_reference_annotated_analysis.png   (optional)

Expected filename metadata
--------------------------
The acquisition script saves names like:

00001_tracked_block_X100.000_Y20.000_Z-155.000_B-2.345_C180.000_C180_TIP90.000_BPH0.50000_2026....png

This script parses:
  - B
  - C
  - block name (C0 / C180)
  - TIP angle
  - block phase
  - commanded XYZ if present

Important convention
--------------------
- Measured points are converted to checkerboard mm coordinates (u_mm, z_mm).
- For display:
    * C=0 is shown as-is
    * C=180 is mirrored horizontally (u -> -u) for comparison against C=0
- Predicted trajectory is generated from the calibration JSON polynomials using
  the B values actually present in each orientation group.
- Curling vs uncurling is inferred from local B derivative sign.

Usage example
-------------
python3 offline_dual_c_b_hysteresis_checkerboard_analysis.py \
    --project_dir "/path/to/Point_Tracking_Run_2026-03-20_12-34-56" \
    --calibration "/path/to/calibration.json" \
    --camera_calibration_file "/path/to/camera_calibration.npz" \
    --checkerboard_reference_image "/path/to/checkerboard_ref.png" \
    --threshold 200 \
    --tip_refine_mode edge_dt \
    --save_plots

Notes
-----
- Expects shadow_calibration.py to be importable from the same folder or PYTHONPATH.
- Uses the same checkerboard-based mm coordinate workflow as your existing offline script.
- The predicted trajectory comes from the calibration JSON used by the robot script.
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
from dataclasses import dataclass
from heapq import heappop, heappush
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PathCollection
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec

# -----------------------------------------------------------------------------
# Import existing shadow calibration pipeline
# -----------------------------------------------------------------------------
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

OFFPLANE_SIGN = -1.0


# =============================================================================
# Calibration model from the acquisition script
# =============================================================================
@dataclass
class Calibration:
    pr: np.ndarray
    pz: np.ndarray
    py_off: Optional[np.ndarray]
    pa: Optional[np.ndarray]
    r_model: Optional[Dict[str, Any]]
    z_model: Optional[Dict[str, Any]]
    y_off_model: Optional[Dict[str, Any]]
    tip_angle_model: Optional[Dict[str, Any]]

    b_min: float
    b_max: float

    x_axis: str
    y_axis: str
    z_axis: str
    b_axis: str
    c_axis: str

    c_180_deg: float

    offplane_y_equation: Optional[str] = None
    offplane_y_r_squared: Optional[float] = None
    selected_fit_model: Optional[str] = None


def poly_eval(coeffs: Any, u: Any, default_if_none: Optional[float] = None) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    if coeffs is None:
        if default_if_none is None:
            raise ValueError("Missing required polynomial coefficients.")
        return np.full_like(u, float(default_if_none), dtype=float)

    arr = np.asarray(coeffs, dtype=float).reshape(-1)
    if arr.size == 0:
        if default_if_none is None:
            raise ValueError("Polynomial coefficients array is empty.")
        return np.full_like(u, float(default_if_none), dtype=float)

    return np.polyval(arr, u)


def _normalize_model_spec(model_spec: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(model_spec, dict):
        return None
    out = dict(model_spec)
    if out.get("model_type") is not None:
        out["model_type"] = str(out["model_type"]).strip().lower()
    return out


def _pchip_endpoint_slope(h0: float, h1: float, delta0: float, delta1: float) -> float:
    d = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / max(h0 + h1, 1e-12)
    if np.sign(d) != np.sign(delta0):
        return 0.0
    if np.sign(delta0) != np.sign(delta1) and abs(d) > abs(3.0 * delta0):
        return 3.0 * delta0
    return float(d)


def pchip_eval(x_knots: Any, y_knots: Any, x_query: Any) -> np.ndarray:
    x = np.asarray(x_knots, dtype=float).reshape(-1)
    y = np.asarray(y_knots, dtype=float).reshape(-1)
    xq = np.asarray(x_query, dtype=float)

    if x.size != y.size or x.size == 0:
        raise ValueError("PCHIP model requires equal-length non-empty x_knots and y_knots.")
    if x.size == 1:
        return np.full_like(xq, float(y[0]), dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if np.any(np.diff(x) <= 0):
        raise ValueError("PCHIP x_knots must be strictly increasing.")

    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros_like(y)

    if x.size == 2:
        d[:] = delta[0]
    else:
        for k in range(1, x.size - 1):
            if delta[k - 1] == 0.0 or delta[k] == 0.0 or np.sign(delta[k - 1]) != np.sign(delta[k]):
                d[k] = 0.0
            else:
                w1 = 2.0 * h[k] + h[k - 1]
                w2 = h[k] + 2.0 * h[k - 1]
                d[k] = (w1 + w2) / ((w1 / delta[k - 1]) + (w2 / delta[k]))
        d[0] = _pchip_endpoint_slope(h[0], h[1], delta[0], delta[1])
        d[-1] = _pchip_endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])

    flat = xq.reshape(-1)
    idx = np.searchsorted(x, flat, side="right") - 1
    idx = np.clip(idx, 0, x.size - 2)

    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    h_i = x1 - x0
    t = (flat - x0) / h_i

    h00 = (2.0 * t ** 3) - (3.0 * t ** 2) + 1.0
    h10 = t ** 3 - 2.0 * t ** 2 + t
    h01 = (-2.0 * t ** 3) + (3.0 * t ** 2)
    h11 = t ** 3 - t ** 2
    yq = h00 * y0 + h10 * h_i * d[idx] + h01 * y1 + h11 * h_i * d[idx + 1]
    return yq.reshape(xq.shape)


def eval_model_spec(model_spec: Optional[Dict[str, Any]], u: Any, default_if_none: Optional[float] = None) -> np.ndarray:
    if model_spec is None:
        if default_if_none is None:
            raise ValueError("Missing required calibration model.")
        return np.full_like(np.asarray(u, dtype=float), float(default_if_none), dtype=float)

    model_type = str(model_spec.get("model_type") or "").strip().lower()
    if model_type == "pchip":
        return pchip_eval(model_spec.get("x_knots"), model_spec.get("y_knots"), u)
    if model_type == "polynomial":
        coeffs = model_spec.get("coefficients", model_spec.get("coeffs"))
        return poly_eval(coeffs, u, default_if_none=default_if_none)
    raise ValueError(f"Unsupported calibration model_type: {model_type}")


def load_calibration(json_path: str) -> Calibration:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")

    with p.open("r") as f:
        data = json.load(f)

    cubic = data["cubic_coefficients"]
    pr = np.array(cubic["r_coeffs"], dtype=float)
    pz = np.array(cubic["z_coeffs"], dtype=float)
    py_off_raw = cubic.get("offplane_y_coeffs", None)
    py_off = None if py_off_raw is None else np.array(py_off_raw, dtype=float)
    pa_raw = cubic.get("tip_angle_coeffs", None)
    pa = None if pa_raw is None else np.array(pa_raw, dtype=float)
    fit_models = data.get("fit_models", {})
    r_model = _normalize_model_spec(fit_models.get("r"))
    z_model = _normalize_model_spec(fit_models.get("z"))
    y_off_model = _normalize_model_spec(fit_models.get("offplane_y"))
    tip_angle_model = _normalize_model_spec(fit_models.get("tip_angle"))
    selected_fit_model = data.get("selected_fit_model")
    selected_fit_model = None if selected_fit_model is None else str(selected_fit_model).strip().lower()

    motor_setup = data.get("motor_setup", {})
    duet_map = data.get("duet_axis_mapping", {})

    b_range = motor_setup.get("b_motor_position_range", [-5.4, 0.0])
    b_min, b_max = map(float, b_range)
    if b_min > b_max:
        b_min, b_max = b_max, b_min

    x_axis = str(duet_map.get("horizontal_axis") or motor_setup.get("horizontal_axis") or "X")
    y_axis = str(duet_map.get("depth_axis") or motor_setup.get("depth_axis") or "Y")
    z_axis = str(duet_map.get("vertical_axis") or motor_setup.get("vertical_axis") or "Z")
    b_axis = str(duet_map.get("pull_axis") or motor_setup.get("b_motor_axis") or "B")
    c_axis = str(duet_map.get("rotation_axis") or motor_setup.get("rotation_axis") or "C")
    c_180 = float(motor_setup.get("rotation_axis_180_deg", 180.0))

    return Calibration(
        pr=pr,
        pz=pz,
        py_off=py_off,
        pa=pa,
        r_model=r_model,
        z_model=z_model,
        y_off_model=y_off_model,
        tip_angle_model=tip_angle_model,
        b_min=b_min,
        b_max=b_max,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        b_axis=b_axis,
        c_axis=c_axis,
        c_180_deg=c_180,
        offplane_y_equation=cubic.get("offplane_y_equation"),
        offplane_y_r_squared=(
            None if cubic.get("offplane_y_r_squared") is None
            else float(cubic["offplane_y_r_squared"])
        ),
        selected_fit_model=selected_fit_model,
    )


def eval_r(cal: Calibration, b: Any, flip_rz_sign: bool = False) -> np.ndarray:
    s = -1.0 if bool(flip_rz_sign) else 1.0
    if cal.r_model is not None:
        return s * eval_model_spec(cal.r_model, b)
    return s * poly_eval(cal.pr, b)


def eval_z(cal: Calibration, b: Any, flip_rz_sign: bool = False) -> np.ndarray:
    s = -1.0 if bool(flip_rz_sign) else 1.0
    if cal.z_model is not None:
        return s * eval_model_spec(cal.z_model, b)
    return s * poly_eval(cal.pz, b)


def eval_offplane_y(cal: Calibration, b: Any) -> np.ndarray:
    if cal.y_off_model is not None:
        return OFFPLANE_SIGN * eval_model_spec(cal.y_off_model, b, default_if_none=0.0)
    return OFFPLANE_SIGN * poly_eval(cal.py_off, b, default_if_none=0.0)


def eval_tip_angle_deg(cal: Calibration, b: Any) -> np.ndarray:
    if cal.tip_angle_model is not None:
        return eval_model_spec(cal.tip_angle_model, b)
    return poly_eval(cal.pa, b)


def predict_tip_xyz_from_bc(
    cal: Calibration,
    b: Any,
    c_deg: float,
    flip_rz_sign: bool = False,
) -> np.ndarray:
    b = np.asarray(b, dtype=float)
    r = np.asarray(eval_r(cal, b, flip_rz_sign=flip_rz_sign), dtype=float)
    z = np.asarray(eval_z(cal, b, flip_rz_sign=flip_rz_sign), dtype=float)
    y_off = np.asarray(eval_offplane_y(cal, b), dtype=float)

    c = math.radians(float(c_deg))
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.column_stack([x, y, z]).astype(float)


# =============================================================================
# Utilities: IO and discovery
# =============================================================================
def list_images(folder: Path) -> List[Path]:
    imgs = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    imgs.sort()
    return imgs


def ensure_project_from_raw(raw_dir: Path, project_dir: Path, link_mode: str = "symlink") -> Path:
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


# =============================================================================
# GUI: crop selection
# =============================================================================
def interactive_crop_from_image(
    image_bgr: np.ndarray,
    default_crop=None,
    window_crop="Manual Crop Setup (OFFLINE)",
):
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

    return analysis_crop


# =============================================================================
# Tip refinement helpers
# =============================================================================
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
    try:
        from skimage.morphology import skeletonize
        skel = skeletonize(mask_fg.astype(bool))
        return skel.astype(np.uint8)
    except Exception:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            skel = cv2.ximgproc.thinning((mask_fg * 255).astype(np.uint8))
            return (skel > 0).astype(np.uint8)
        raise RuntimeError(
            "Need either scikit-image or cv2.ximgproc.thinning for skeletonization."
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
    _, _, vh = np.linalg.svd(pts - mu, full_matrices=False)
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
            if axs.ndim >= 2 and axs.shape[0] >= 2 and axs.shape[1] >= 1:
                target_axes = [axs[1, 0]]
            else:
                target_axes = [axs.flat[-1]]
        elif isinstance(axs, (list, tuple)):
            target_axes = [axs[-1]]
        else:
            target_axes = [axs]

        used_labels = set()
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
            _plot_line("center_line_xy", "#ffd400", "center line")
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
                lbl = "refined tip" if "refined tip" not in used_labels else None
                if lbl is not None:
                    used_labels.add("refined tip")
                ax.scatter([tip_xy[0]], [tip_xy[1]], s=55, c="#ff3b30", edgecolors="#ffffff", label=lbl, zorder=6)

            try:
                ax.legend(loc="best", fontsize=8)
            except Exception:
                pass
    except Exception as e:
        print(f"[WARN] Failed to annotate analysis axes: {e}")


def _remove_zoom_coarse_tip_markers(axs):
    if axs is None or not isinstance(axs, np.ndarray) or axs.size < 4:
        return

    for ax in (axs[1, 0], axs[1, 1]):
        for coll in list(ax.collections):
            if not isinstance(coll, PathCollection):
                continue
            try:
                offsets = np.asarray(coll.get_offsets(), dtype=float)
            except Exception:
                continue
            if offsets.ndim == 2 and offsets.shape[0] == 1:
                coll.remove()

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    try:
        axs[1, 0].set_title("Refined tip geometry")
    except Exception:
        pass


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
        _remove_zoom_coarse_tip_markers(axs)
        annotate_tip_geometry_on_axes(axs, dbg_local, title_suffix=f" ({refine_mode})")
        return fig, axs, coarse_row_refined, fine_row

    cal.analyze_data = analyze_data_patched
    return cal


# =============================================================================
# Filename metadata parsing
# =============================================================================
_NUM = r"[-+]?\d+(?:\.\d+)?"

RE_STAGE_X = re.compile(rf"_X({_NUM})")
RE_STAGE_Y = re.compile(rf"_Y({_NUM})")
RE_STAGE_Z = re.compile(rf"_Z({_NUM})")
RE_B_VAL = re.compile(rf"_B({_NUM})")
RE_C_VAL = re.compile(rf"_C({_NUM})")
RE_TIP_VAL = re.compile(rf"_TIP({_NUM})")
RE_BPH_VAL = re.compile(rf"_BPH({_NUM})")
RE_SAMPLE_IDX = re.compile(r"^(\d+)_")
RE_PHASE = re.compile(r"^\d+_([^_]+)")
RE_BLOCK_NAME = re.compile(r"_(C0|C180)(?:_|$)")


def _safe_float_from_match(regex: re.Pattern, text: str) -> Optional[float]:
    m = regex.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _safe_int_from_match(regex: re.Pattern, text: str) -> Optional[int]:
    m = regex.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _safe_str_from_match(regex: re.Pattern, text: str) -> Optional[str]:
    m = regex.search(text)
    if not m:
        return None
    return str(m.group(1))


def canonicalize_c_orientation(c_deg: Optional[float], round_decimals: int = 3) -> Optional[float]:
    if c_deg is None or not np.isfinite(c_deg):
        return None
    return float(np.round(float(c_deg), int(round_decimals)))


def infer_block_name_from_c(c_deg: Optional[float]) -> Optional[str]:
    if c_deg is None or not np.isfinite(c_deg):
        return None
    c = canonicalize_c_orientation(c_deg)
    if abs(c - 0.0) < 1e-6:
        return "C0"
    if abs(c - 180.0) < 1e-6:
        return "C180"
    return f"C{c:g}"


def parse_fixed_tip_filename_metadata(image_name: str) -> Dict[str, Any]:
    stem = Path(image_name).stem

    sample_index = _safe_int_from_match(RE_SAMPLE_IDX, stem)
    phase = _safe_str_from_match(RE_PHASE, stem)
    stage_x = _safe_float_from_match(RE_STAGE_X, stem)
    stage_y = _safe_float_from_match(RE_STAGE_Y, stem)
    stage_z = _safe_float_from_match(RE_STAGE_Z, stem)
    b_val = _safe_float_from_match(RE_B_VAL, stem)
    c_val = _safe_float_from_match(RE_C_VAL, stem)
    tip_angle_deg = _safe_float_from_match(RE_TIP_VAL, stem)
    block_phase_01 = _safe_float_from_match(RE_BPH_VAL, stem)
    block_name = _safe_str_from_match(RE_BLOCK_NAME, stem)

    c_orientation_deg = canonicalize_c_orientation(c_val)
    if block_name is None:
        block_name = infer_block_name_from_c(c_orientation_deg)

    return {
        "sample_index_from_name": sample_index,
        "phase_name": phase,
        "stage_x_cmd": stage_x,
        "stage_y_cmd": stage_y,
        "stage_z_cmd": stage_z,
        "b_cmd": b_val,
        "c_cmd_deg": c_val,
        "c_orientation_deg": c_orientation_deg,
        "tip_angle_deg_from_name": tip_angle_deg,
        "block_phase_01": block_phase_01,
        "block_name": block_name,
    }


# =============================================================================
# Coordinate / hysteresis helpers
# =============================================================================
def mirror_u_for_orientation(u_mm: np.ndarray, c_orientation_deg: Optional[float], c180_deg: float = 180.0) -> np.ndarray:
    u_mm = np.asarray(u_mm, dtype=float)
    if c_orientation_deg is None or not np.isfinite(c_orientation_deg):
        return u_mm.copy()
    if abs(float(c_orientation_deg) - float(c180_deg)) < 1e-6:
        return -u_mm
    return u_mm.copy()


def classify_local_motion_direction(b_values: np.ndarray) -> List[str]:
    """
    Classify each sample as curling / uncurling using a local derivative of B.

    Convention:
      db > 0  -> curling
      db < 0  -> uncurling

    If a point lies exactly at an extremum / plateau, inherit nearest valid sign.
    """
    b = np.asarray(b_values, dtype=float).reshape(-1)
    n = len(b)
    if n == 0:
        return []
    if n == 1:
        return ["unknown"]

    dirs = np.zeros(n, dtype=int)

    for i in range(n):
        if i == 0:
            db = b[1] - b[0]
        elif i == n - 1:
            db = b[-1] - b[-2]
        else:
            db = b[i + 1] - b[i - 1]

        if db > 1e-12:
            dirs[i] = 1
        elif db < -1e-12:
            dirs[i] = -1
        else:
            dirs[i] = 0

    # Fill zeros from nearest nonzero
    for i in range(n):
        if dirs[i] != 0:
            continue
        left = None
        right = None
        for j in range(i - 1, -1, -1):
            if dirs[j] != 0:
                left = dirs[j]
                break
        for j in range(i + 1, n):
            if dirs[j] != 0:
                right = dirs[j]
                break
        if left is not None:
            dirs[i] = left
        elif right is not None:
            dirs[i] = right
        else:
            dirs[i] = 0

    out = []
    for d in dirs:
        if d > 0:
            out.append("curling")
        elif d < 0:
            out.append("uncurling")
        else:
            out.append("unknown")
    return out


def _orientation_key(c_deg: Optional[float]) -> str:
    if c_deg is None or not np.isfinite(c_deg):
        return "unknown"
    return f"{float(c_deg):.3f}"


def _orientation_sort_key(k: str):
    try:
        return (0, float(k))
    except Exception:
        return (1, k)


def _orientation_display_label(c_deg: Optional[float]) -> str:
    if c_deg is None or not np.isfinite(c_deg):
        return "C = unknown"
    if abs(float(c_deg) - round(float(c_deg))) < 1e-9:
        return f"C = {int(round(float(c_deg)))}°"
    return f"C = {float(c_deg):.3f}°"


# =============================================================================
# Data conversion / metrics
# =============================================================================
def compute_tracked_tip_positions_mm(
    cal: CTR_Shadow_Calibration,
    tracked_rows: np.ndarray,
    image_files: List[Path],
) -> Dict[str, Any]:
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
        meta = parse_fixed_tip_filename_metadata(file_name)

        if not np.all(np.isfinite(row[:2])):
            records.append({
                "sample_index": i,
                "image_name": file_name,
                "tip_y_px": None,
                "tip_x_px": None,
                "u_mm": None,
                "z_mm": None,
                "valid": False,
                **meta,
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
            "valid": bool(valid),
            **meta,
        }
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


def enrich_records_with_motion_direction(records: List[Dict[str, Any]], c180_deg: float = 180.0) -> List[Dict[str, Any]]:
    grouped: Dict[str, List[int]] = {}
    for idx, rec in enumerate(records):
        if not rec.get("valid", False):
            continue
        grouped.setdefault(_orientation_key(rec.get("c_orientation_deg")), []).append(idx)

    for _, indices in grouped.items():
        indices = sorted(indices, key=lambda i: records[i].get("sample_index", i))
        b_vals = np.asarray([
            np.nan if records[i].get("b_cmd") is None else float(records[i]["b_cmd"])
            for i in indices
        ], dtype=float)

        dirs = classify_local_motion_direction(b_vals)
        for i_rec, motion in zip(indices, dirs):
            records[i_rec]["motion_direction"] = motion
            u_val = records[i_rec].get("u_mm")
            c_deg = records[i_rec].get("c_orientation_deg")
            if u_val is None:
                records[i_rec]["u_plot_mm"] = None
            else:
                records[i_rec]["u_plot_mm"] = float(mirror_u_for_orientation(np.array([float(u_val)]), c_deg, c180_deg=c180_deg)[0])
            records[i_rec]["z_plot_mm"] = None if records[i_rec].get("z_mm") is None else float(records[i_rec]["z_mm"])

    for rec in records:
        rec.setdefault("motion_direction", "unknown")
        if "u_plot_mm" not in rec:
            rec["u_plot_mm"] = None if rec.get("u_mm") is None else float(rec["u_mm"])
        if "z_plot_mm" not in rec:
            rec["z_plot_mm"] = None if rec.get("z_mm") is None else float(rec["z_mm"])

    return records


def compute_orientation_metrics(records: List[Dict[str, Any]]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        if not rec.get("valid", False):
            continue
        grouped.setdefault(_orientation_key(rec.get("c_orientation_deg")), []).append(rec)

    for key, recs in sorted(grouped.items(), key=lambda kv: _orientation_sort_key(kv[0])):
        pts = np.asarray([[float(r["u_plot_mm"]), float(r["z_plot_mm"])] for r in recs], dtype=float)
        b_vals = np.asarray([float(r["b_cmd"]) for r in recs if r.get("b_cmd") is not None], dtype=float)
        tip_vals = np.asarray([float(r["tip_angle_deg_from_name"]) for r in recs if r.get("tip_angle_deg_from_name") is not None], dtype=float)

        motion_groups = {}
        for motion_name in ("curling", "uncurling"):
            subset = [r for r in recs if r.get("motion_direction") == motion_name]
            if subset:
                motion_groups[motion_name] = {
                    "num_samples": int(len(subset)),
                    "u_mean_mm": float(np.mean([float(r["u_plot_mm"]) for r in subset])),
                    "z_mean_mm": float(np.mean([float(r["z_plot_mm"]) for r in subset])),
                }
            else:
                motion_groups[motion_name] = {"num_samples": 0, "u_mean_mm": None, "z_mean_mm": None}

        out[key] = {
            "c_orientation_deg": None if not recs or recs[0].get("c_orientation_deg") is None else float(recs[0]["c_orientation_deg"]),
            "num_samples": int(len(recs)),
            "b_range": None if b_vals.size == 0 else [float(np.min(b_vals)), float(np.max(b_vals))],
            "tip_angle_range_deg": None if tip_vals.size == 0 else [float(np.min(tip_vals)), float(np.max(tip_vals))],
            "u_plot_range_mm": [float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))],
            "z_plot_range_mm": [float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))],
            "trajectory_centroid_plot_mm": {
                "u_plot_mean_mm": float(np.mean(pts[:, 0])),
                "z_plot_mean_mm": float(np.mean(pts[:, 1])),
            },
            "motion_groups": motion_groups,
        }

    return out


def save_tracked_tip_csv(csv_path: Path, records: List[Dict[str, Any]]):
    fieldnames = [
        "sample_index",
        "sample_index_from_name",
        "image_name",
        "phase_name",
        "block_name",
        "block_phase_01",
        "stage_x_cmd",
        "stage_y_cmd",
        "stage_z_cmd",
        "b_cmd",
        "c_cmd_deg",
        "c_orientation_deg",
        "tip_angle_deg_from_name",
        "motion_direction",
        "tip_y_px",
        "tip_x_px",
        "u_mm",
        "z_mm",
        "u_plot_mm",
        "z_plot_mm",
        "valid",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


# =============================================================================
# Predicted trajectory helpers
# =============================================================================
def build_predicted_trajectory_for_orientation(
    model_cal: Calibration,
    c_orientation_deg: float,
    b_values: np.ndarray,
    flip_rz_sign: bool = True,
    mirror_for_plot: bool = True,
    c180_deg: float = 180.0,
) -> Dict[str, np.ndarray]:
    b_values = np.asarray(b_values, dtype=float).reshape(-1)
    b_values = b_values[np.isfinite(b_values)]
    if b_values.size == 0:
        return {
            "b_sorted": np.empty((0,), dtype=float),
            "u_plot_mm": np.empty((0,), dtype=float),
            "z_mm": np.empty((0,), dtype=float),
            "x_mm": np.empty((0,), dtype=float),
            "y_mm": np.empty((0,), dtype=float),
        }

    b_sorted = np.sort(np.unique(b_values))
    pred_xyz = predict_tip_xyz_from_bc(
        model_cal,
        b=b_sorted,
        c_deg=float(c_orientation_deg),
        flip_rz_sign=bool(flip_rz_sign),
    )
    x_mm = pred_xyz[:, 0]
    y_mm = pred_xyz[:, 1]
    z_mm = pred_xyz[:, 2]

    u_plot = x_mm.copy()
    if mirror_for_plot and abs(float(c_orientation_deg) - float(c180_deg)) < 1e-6:
        u_plot = -u_plot

    return {
        "b_sorted": b_sorted,
        "u_plot_mm": u_plot,
        "z_mm": z_mm,
        "x_mm": x_mm,
        "y_mm": y_mm,
    }


def build_predicted_trajectory_for_measured_sequence(
    model_cal: Calibration,
    c_orientation_deg: float,
    b_values: np.ndarray,
    flip_rz_sign: bool = True,
    mirror_for_plot: bool = True,
    c180_deg: float = 180.0,
) -> Dict[str, np.ndarray]:
    b_seq = np.asarray(b_values, dtype=float).reshape(-1)
    valid_mask = np.isfinite(b_seq)
    b_seq = b_seq[valid_mask]
    if b_seq.size == 0:
        return {
            "b_sequence": np.empty((0,), dtype=float),
            "u_plot_mm": np.empty((0,), dtype=float),
            "z_mm": np.empty((0,), dtype=float),
            "x_mm": np.empty((0,), dtype=float),
            "y_mm": np.empty((0,), dtype=float),
        }

    pred_xyz = predict_tip_xyz_from_bc(
        model_cal,
        b=b_seq,
        c_deg=float(c_orientation_deg),
        flip_rz_sign=bool(flip_rz_sign),
    )
    x_mm = pred_xyz[:, 0]
    y_mm = pred_xyz[:, 1]
    z_mm = pred_xyz[:, 2]

    u_plot = x_mm.copy()
    if mirror_for_plot and abs(float(c_orientation_deg) - float(c180_deg)) < 1e-6:
        u_plot = -u_plot

    return {
        "b_sequence": b_seq,
        "u_plot_mm": u_plot,
        "z_mm": z_mm,
        "x_mm": x_mm,
        "y_mm": y_mm,
    }


def align_predicted_path_to_measured_reference(
    pred_u_mm: np.ndarray,
    pred_z_mm: np.ndarray,
    meas_u_mm: np.ndarray,
    meas_z_mm: np.ndarray,
    allow_u_sign_flip: bool = True,
    allow_z_sign_flip: bool = True,
) -> Dict[str, Any]:
    pred_u = np.asarray(pred_u_mm, dtype=float).reshape(-1)
    pred_z = np.asarray(pred_z_mm, dtype=float).reshape(-1)
    meas_u = np.asarray(meas_u_mm, dtype=float).reshape(-1)
    meas_z = np.asarray(meas_z_mm, dtype=float).reshape(-1)

    n = min(pred_u.size, pred_z.size, meas_u.size, meas_z.size)
    if n == 0:
        return {
            "u_plot_mm": pred_u.copy(),
            "z_mm": pred_z.copy(),
            "sign_u": 1.0,
            "sign_z": 1.0,
            "reversed": False,
            "shift_u_mm": 0.0,
            "shift_z_mm": 0.0,
            "rmse_mm": None,
        }

    pred_u = pred_u[:n]
    pred_z = pred_z[:n]
    meas_u = meas_u[:n]
    meas_z = meas_z[:n]

    best = None
    u_sign_options = (1.0, -1.0) if bool(allow_u_sign_flip) else (1.0,)
    z_sign_options = (1.0, -1.0) if bool(allow_z_sign_flip) else (1.0,)

    for reverse in (False, True):
        if reverse:
            cand_u0 = pred_u[::-1]
            cand_z0 = pred_z[::-1]
        else:
            cand_u0 = pred_u
            cand_z0 = pred_z

        for sign_u in u_sign_options:
            for sign_z in z_sign_options:
                cand_u = sign_u * cand_u0
                cand_z = sign_z * cand_z0

                shift_u = float(meas_u[0] - cand_u[0])
                shift_z = float(meas_z[0] - cand_z[0])
                cand_u_shift = cand_u + shift_u
                cand_z_shift = cand_z + shift_z

                rmse = float(np.sqrt(np.mean((cand_u_shift - meas_u) ** 2 + (cand_z_shift - meas_z) ** 2)))
                score = (
                    rmse,
                    0 if not reverse else 1,
                    0 if sign_u > 0 else 1,
                    0 if sign_z > 0 else 1,
                )
                if best is None or score < best["score"]:
                    best = {
                        "score": score,
                        "u_plot_mm": cand_u_shift,
                        "z_mm": cand_z_shift,
                        "sign_u": sign_u,
                        "sign_z": sign_z,
                        "reversed": reverse,
                        "shift_u_mm": shift_u,
                        "shift_z_mm": shift_z,
                        "rmse_mm": rmse,
                    }

    return best


def _build_segmented_series(
    u_vals: np.ndarray,
    z_vals: np.ndarray,
    labels: List[str],
) -> Tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u_vals, dtype=float).reshape(-1)
    z = np.asarray(z_vals, dtype=float).reshape(-1)
    n = min(u.size, z.size, len(labels))
    if n == 0:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)

    pieces_u: List[np.ndarray] = []
    pieces_z: List[np.ndarray] = []
    start = 0
    while start < n:
        end = start + 1
        while end < n and labels[end] == labels[start]:
            end += 1
        pieces_u.append(u[start:end])
        pieces_z.append(z[start:end])
        if end < n:
            pieces_u.append(np.array([np.nan], dtype=float))
            pieces_z.append(np.array([np.nan], dtype=float))
        start = end

    return np.concatenate(pieces_u), np.concatenate(pieces_z)


def _build_segmented_series_for_label(
    u_vals: np.ndarray,
    z_vals: np.ndarray,
    labels: List[str],
    target_label: str,
) -> Tuple[np.ndarray, np.ndarray]:
    u = np.asarray(u_vals, dtype=float).reshape(-1)
    z = np.asarray(z_vals, dtype=float).reshape(-1)
    n = min(u.size, z.size, len(labels))
    if n == 0:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)

    pieces_u: List[np.ndarray] = []
    pieces_z: List[np.ndarray] = []
    start = 0
    while start < n:
        if labels[start] != target_label:
            start += 1
            continue
        end = start + 1
        while end < n and labels[end] == target_label:
            end += 1
        pieces_u.append(u[start:end])
        pieces_z.append(z[start:end])
        if end < n:
            pieces_u.append(np.array([np.nan], dtype=float))
            pieces_z.append(np.array([np.nan], dtype=float))
        start = end

    if not pieces_u:
        return np.empty((0,), dtype=float), np.empty((0,), dtype=float)
    return np.concatenate(pieces_u), np.concatenate(pieces_z)


def overlay_u_for_orientation(
    u_plot_mm: np.ndarray,
    c_orientation_deg: Optional[float],
) -> np.ndarray:
    u_plot = np.asarray(u_plot_mm, dtype=float)
    if c_orientation_deg is None or not np.isfinite(c_orientation_deg):
        return u_plot.copy()
    if abs(float(c_orientation_deg) - 0.0) < 1e-6:
        return -u_plot
    return u_plot.copy()


def predicted_path_descriptor(cal: Calibration) -> str:
    model_type = str(cal.selected_fit_model or "").strip().lower()
    if model_type == "pchip":
        return "PCHIP"

    for model_spec in (cal.r_model, cal.z_model):
        if isinstance(model_spec, dict) and str(model_spec.get("model_type") or "").strip().lower() == "pchip":
            return "PCHIP"
    return "polynomial"


# =============================================================================
# Plot styling
# =============================================================================
def _make_dark_density_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "sleek_density_dark",
        [
            "#070b12",
            "#0f1623",
            "#132235",
            "#17354f",
            "#1c5276",
            "#2087a1",
            "#53d0cf",
            "#d8f16e",
        ],
        N=256,
    )


def _style_dark_transparent_axes(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, color="#f3f7fb", fontsize=12.5, pad=10, weight="semibold")
    ax.set_xlabel(xlabel, color="#d9e5f0")
    ax.set_ylabel(ylabel, color="#d9e5f0")
    ax.tick_params(colors="#cad8e5", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color((0.86, 0.91, 0.96, 0.22))
        spine.set_linewidth(1.05)
    ax.grid(True, color=(0.85, 0.90, 0.95, 0.10), linewidth=0.85)
    ax.set_facecolor((0.0, 0.0, 0.0, 0.0))


def save_hysteresis_vs_sample_plot(plot_path: Path, records: List[Dict[str, Any]]):
    valid_recs = [r for r in records if r.get("valid", False) and r.get("u_plot_mm") is not None and r.get("z_plot_mm") is not None]
    if not valid_recs:
        raise RuntimeError("No valid records available for sample plot.")

    sample_idx = np.asarray([int(r["sample_index"]) for r in valid_recs], dtype=int)
    u_vals = np.asarray([float(r["u_plot_mm"]) for r in valid_recs], dtype=float)
    z_vals = np.asarray([float(r["z_plot_mm"]) for r in valid_recs], dtype=float)
    c_vals = [r.get("c_orientation_deg") for r in valid_recs]

    fig = plt.figure(figsize=(11.8, 6.2))
    fig.patch.set_alpha(0.0)
    gs = GridSpec(2, 1, height_ratios=[1.0, 1.0], hspace=0.22, figure=fig)

    ax_u = fig.add_subplot(gs[0, 0])
    ax_z = fig.add_subplot(gs[1, 0])

    ax_u.plot(sample_idx, u_vals, linewidth=1.4, color="#c6d4e3", alpha=0.70, zorder=1)
    ax_z.plot(sample_idx, z_vals, linewidth=1.4, color="#c6d4e3", alpha=0.70, zorder=1)

    c0_mask = np.array([c is not None and abs(float(c) - 0.0) < 1e-6 for c in c_vals], dtype=bool)
    c180_mask = np.array([c is not None and abs(float(c) - 180.0) < 1e-6 for c in c_vals], dtype=bool)
    other_mask = ~(c0_mask | c180_mask)

    if np.any(c0_mask):
        ax_u.scatter(sample_idx[c0_mask], u_vals[c0_mask], s=18, color="#4cc9f0", edgecolors="none", label="C = 0°", zorder=3)
        ax_z.scatter(sample_idx[c0_mask], z_vals[c0_mask], s=18, color="#4cc9f0", edgecolors="none", label="C = 0°", zorder=3)
    if np.any(c180_mask):
        ax_u.scatter(sample_idx[c180_mask], u_vals[c180_mask], s=18, color="#f72585", edgecolors="none", label="C = 180°", zorder=3)
        ax_z.scatter(sample_idx[c180_mask], z_vals[c180_mask], s=18, color="#f72585", edgecolors="none", label="C = 180°", zorder=3)
    if np.any(other_mask):
        ax_u.scatter(sample_idx[other_mask], u_vals[other_mask], s=16, color="#d8dee9", edgecolors="none", label="other C", zorder=3)
        ax_z.scatter(sample_idx[other_mask], z_vals[other_mask], s=16, color="#d8dee9", edgecolors="none", label="other C", zorder=3)

    _style_dark_transparent_axes(ax_u, "Mirrored horizontal tip position vs sample", "Sample index", "u_plot (mm)")
    _style_dark_transparent_axes(ax_z, "Vertical tip position vs sample", "Sample index", "z (mm)")

    for ax in (ax_u, ax_z):
        leg = ax.legend(loc="upper right", frameon=True, fontsize=9)
        leg.get_frame().set_facecolor((0.06, 0.10, 0.15, 0.72))
        leg.get_frame().set_edgecolor((0.86, 0.91, 0.96, 0.15))
        for txt in leg.get_texts():
            txt.set_color("#eef5fb")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=220, transparent=True)
    plt.close(fig)


def save_dual_c_hysteresis_overlay_plot(
    plot_path: Path,
    records: List[Dict[str, Any]],
    model_cal: Calibration,
    flip_rz_sign: bool = True,
):
    valid_recs = [r for r in records if r.get("valid", False) and r.get("u_plot_mm") is not None and r.get("z_plot_mm") is not None]
    if not valid_recs:
        raise RuntimeError("No valid records available for hysteresis overlay plot.")

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in valid_recs:
        grouped.setdefault(_orientation_key(rec.get("c_orientation_deg")), []).append(rec)

    group_items = sorted(grouped.items(), key=lambda kv: _orientation_sort_key(kv[0]))
    if len(group_items) == 0:
        raise RuntimeError("No orientation groups found.")
    if len(group_items) == 1:
        group_items = [group_items[0], group_items[0]]
    elif len(group_items) > 2:
        group_items = group_items[:2]

    all_z = np.asarray([float(r["z_plot_mm"]) for r in valid_recs], dtype=float)

    z_span = float(np.ptp(all_z))
    z_pad = max(0.45, 0.12 * max(z_span, 1.0))
    ylim = (float(np.min(all_z) - z_pad), float(np.max(all_z) + z_pad))

    cmap = _make_dark_density_cmap()

    fig = plt.figure(figsize=(13.2, 6.8))
    fig.patch.set_alpha(0.0)
    gs = GridSpec(2, 2, height_ratios=[0.16, 1.0], hspace=0.14, wspace=0.16, figure=fig)

    ax_txt = fig.add_subplot(gs[0, :])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])
    pred_desc = predicted_path_descriptor(model_cal)

    ax_txt.axis("off")

    summary_lines = [
        "Measured hysteresis: curling vs uncurling",
        "C = 180° panel is mirrored about the vertical axis (u -> -u) for visual comparison with C = 0°",
        f"Predicted curve is from calibration {pred_desc} fits (x,z) evaluated at the measured B values",
    ]
    y0 = 0.92
    for i, line in enumerate(summary_lines):
        ax_txt.text(
            0.02,
            y0 - i * 0.26,
            line,
            color="#eef5fb" if i == 0 else "#c8d7e5",
            fontsize=14 if i == 0 else 10.5,
            weight="semibold" if i == 0 else "normal",
            ha="left",
            va="top",
            transform=ax_txt.transAxes,
        )

    def _draw_panel(ax, group_key: str, group_recs: List[Dict[str, Any]]):
        c_deg = None
        for r in group_recs:
            if r.get("c_orientation_deg") is not None:
                c_deg = float(r["c_orientation_deg"])
                break

        ordered = sorted(group_recs, key=lambda r: int(r["sample_index"]))
        ordered_u = np.asarray([float(r["u_plot_mm"]) for r in ordered], dtype=float)
        ordered_z = np.asarray([float(r["z_plot_mm"]) for r in ordered], dtype=float)
        ordered_labels = [str(r.get("motion_direction", "unknown")) for r in ordered]

        # Measured points
        pts_u = np.asarray([float(r["u_plot_mm"]) for r in group_recs], dtype=float)
        pts_z = np.asarray([float(r["z_plot_mm"]) for r in group_recs], dtype=float)

        hb = ax.hexbin(
            pts_u,
            pts_z,
            gridsize=26,
            mincnt=1,
            cmap=cmap,
            linewidths=0.0,
            alpha=0.95,
            zorder=1,
        )

        motion_colors = {
            "curling": "#4cc9f0",
            "uncurling": "#f72585",
            "unknown": "#d0d7e2",
        }

        for motion_name in ("curling", "uncurling"):
            subset = [r for r in group_recs if r.get("motion_direction") == motion_name]
            if not subset:
                continue
            u = np.asarray([float(r["u_plot_mm"]) for r in subset], dtype=float)
            z = np.asarray([float(r["z_plot_mm"]) for r in subset], dtype=float)
            seg_u, seg_z = _build_segmented_series_for_label(
                ordered_u,
                ordered_z,
                ordered_labels,
                motion_name,
            )

            ax.plot(
                seg_u,
                seg_z,
                linewidth=2.0,
                color=motion_colors[motion_name],
                alpha=0.95,
                label=f"Measured {motion_name}",
                zorder=4,
            )
            ax.scatter(
                u,
                z,
                s=18,
                color=motion_colors[motion_name],
                edgecolors="none",
                alpha=0.92,
                zorder=5,
            )

        unknown_subset = [r for r in group_recs if r.get("motion_direction") not in ("curling", "uncurling")]
        if unknown_subset:
            u = np.asarray([float(r["u_plot_mm"]) for r in unknown_subset], dtype=float)
            z = np.asarray([float(r["z_plot_mm"]) for r in unknown_subset], dtype=float)
            ax.scatter(u, z, s=14, color=motion_colors["unknown"], edgecolors="none", alpha=0.75, zorder=4)

        # Predicted trajectory, re-referenced into the same plot frame as the measured path.
        b_vals = np.asarray([
            np.nan if r.get("b_cmd") is None else float(r["b_cmd"])
            for r in ordered
        ], dtype=float)
        pred_raw = build_predicted_trajectory_for_measured_sequence(
            model_cal=model_cal,
            c_orientation_deg=float(c_deg) if c_deg is not None else 0.0,
            b_values=b_vals,
            flip_rz_sign=bool(flip_rz_sign),
            mirror_for_plot=True,
            c180_deg=float(model_cal.c_180_deg),
        )
        if pred_raw["u_plot_mm"].size > 0:
            meas_u = np.asarray([float(r["u_plot_mm"]) for r in ordered], dtype=float)
            meas_z = np.asarray([float(r["z_plot_mm"]) for r in ordered], dtype=float)
            measured_labels = [str(r.get("motion_direction", "unknown")) for r in ordered]
            pred = align_predicted_path_to_measured_reference(
                pred_u_mm=pred_raw["u_plot_mm"],
                pred_z_mm=pred_raw["z_mm"],
                meas_u_mm=meas_u,
                meas_z_mm=meas_z,
                allow_u_sign_flip=True,
                allow_z_sign_flip=True,
            )
            pred_u_plot, pred_z_plot = _build_segmented_series(
                pred["u_plot_mm"],
                pred["z_mm"],
                measured_labels,
            )
            ax.plot(
                pred_u_plot,
                pred_z_plot,
                color="#f8f9fa",
                linewidth=1.6,
                alpha=0.95,
                linestyle="-",
                label=f"Predicted {pred_desc} path",
                zorder=6,
            )

        # Start / end markers
        if ordered:
            ax.scatter(
                [float(ordered[0]["u_plot_mm"])],
                [float(ordered[0]["z_plot_mm"])],
                s=72,
                marker="o",
                facecolors="none",
                edgecolors="#f7f7ff",
                linewidths=1.5,
                zorder=7,
                label="Start",
            )
            ax.scatter(
                [float(ordered[-1]["u_plot_mm"])],
                [float(ordered[-1]["z_plot_mm"])],
                s=82,
                marker="x",
                color="#ffe66d",
                linewidths=2.0,
                zorder=7,
                label="End",
            )

        title = _orientation_display_label(c_deg)
        if c_deg is not None and abs(float(c_deg) - float(model_cal.c_180_deg)) < 1e-6:
            title += "  (mirrored)"
        _style_dark_transparent_axes(ax, title, "Mirrored horizontal position (mm)", "Vertical position (mm)")
        ax.set_ylim(*ylim)

        x_sources = [pts_u]
        if pred_raw["u_plot_mm"].size > 0:
            x_sources.append(np.asarray(pred["u_plot_mm"], dtype=float))
        x_all = np.concatenate([arr[np.isfinite(arr)] for arr in x_sources if arr.size > 0])
        if x_all.size > 0:
            x_span = float(np.ptp(x_all))
            x_pad_left = max(1.2, 0.10 * max(x_span, 1.0))
            x_pad_right = max(10.0, 0.32 * max(x_span, 1.0))
            ax.set_xlim(float(np.min(x_all) - x_pad_left), float(np.max(x_all) + x_pad_right))

        leg = ax.legend(loc="lower right", frameon=True, fontsize=7.4)
        leg.get_frame().set_facecolor((0.06, 0.10, 0.15, 0.72))
        leg.get_frame().set_edgecolor((0.86, 0.91, 0.96, 0.15))
        for txt in leg.get_texts():
            txt.set_color("#eef5fb")

        return hb

    hb_left = _draw_panel(ax_left, group_items[0][0], group_items[0][1])
    hb_right = _draw_panel(ax_right, group_items[1][0], group_items[1][1])

    cbar = fig.colorbar(hb_right, ax=[ax_left, ax_right], pad=0.02, fraction=0.025)
    cbar.set_label("Measured point density", color="#dae6f2")
    cbar.ax.yaxis.set_tick_params(color="#dae6f2")
    plt.setp(cbar.ax.get_yticklabels(), color="#dae6f2")
    cbar.outline.set_edgecolor((0.86, 0.91, 0.96, 0.20))
    cbar.ax.set_facecolor((0.0, 0.0, 0.0, 0.0))

    fig.suptitle(
        f"Dual-C B-only hysteresis: measured curling / uncurling vs predicted {pred_desc} path",
        color="#f6fbff",
        fontsize=15,
        weight="semibold",
        y=0.985,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.968])
    fig.savefig(plot_path, dpi=240, transparent=True)
    plt.close(fig)


# =============================================================================
# Checkerboard overlay
# =============================================================================
def draw_checkerboard_analysis_overlay(
    cal: CTR_Shadow_Calibration,
    output_path: Path,
    tracked_rows: np.ndarray,
    board_debug_image: np.ndarray,
):
    if board_debug_image is None:
        raise ValueError("board_debug_image is required.")
    if tracked_rows is None or tracked_rows.size == 0:
        raise ValueError("tracked_rows is empty; run analyze_data_batch first.")
    if getattr(cal, "board_pose", None) is None or getattr(cal, "true_vertical_img_unit", None) is None:
        raise RuntimeError("Board reference is unavailable.")

    valid_mask = np.all(np.isfinite(tracked_rows[:, :2]), axis=1)
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
    y_text = 32
    for line in summary_lines:
        cv2.putText(overlay, line, (24, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 3)
        cv2.putText(overlay, line, (24, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 1)
        y_text += 30

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"Failed to write annotated checkerboard image: {output_path}")
    return output_path


def save_metrics_json(
    json_path: Path,
    orientation_metrics: Dict[str, Any],
    model_cal: Calibration,
    cal: CTR_Shadow_Calibration,
    args,
):
    payload = {
        "per_c_orientation_metrics": orientation_metrics,
        "calibration_model": {
            "b_min": float(model_cal.b_min),
            "b_max": float(model_cal.b_max),
            "c_180_deg": float(model_cal.c_180_deg),
            "selected_fit_model": model_cal.selected_fit_model,
            "predicted_path_descriptor": predicted_path_descriptor(model_cal),
            "offplane_y_equation": model_cal.offplane_y_equation,
            "offplane_y_r_squared": model_cal.offplane_y_r_squared,
            "flip_rz_sign_used": bool(args.flip_rz_sign),
        },
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
        },
    }
    with open(json_path, "w") as f:
        json.dump(_json_ready(payload), f, indent=2)


# =============================================================================
# Main
# =============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", type=str, default=None,
                    help="Existing project folder containing raw_image_data_folder/")
    ap.add_argument("--raw_dir", type=str, default=None,
                    help="raw_image_data_folder or any folder of images (will be wrapped into a project)")
    ap.add_argument("--threshold", type=int, default=200)
    ap.add_argument("--save_plots", action="store_true")

    ap.add_argument("--calibration", type=str, required=True,
                    help="Path to the motion calibration JSON used by the acquisition script.")
    ap.add_argument("--flip_rz_sign", action="store_true", default=True,
                    help="Use flipped r/z sign when evaluating the motion calibration polynomials.")

    ap.add_argument("--camera_calibration_file", type=str, required=True,
                    help="Path to camera calibration .npz for checkerboard-reference analysis.")
    ap.add_argument("--checkerboard_reference_image", type=str, required=True,
                    help="Path to checkerboard reference image.")
    ap.add_argument("--checkerboard_inner_corners", type=_parse_inner_corners_arg, default=None,
                    help="Checkerboard inner-corner grid as 'Nx,Ny' or 'NxXNy'.")
    ap.add_argument("--checkerboard_square_size_mm", type=float, default=None,
                    help="Checkerboard square size in mm.")
    ap.add_argument("--checkerboard_no_undistort", action="store_true",
                    help="Disable undistortion before checkerboard pose estimation.")

    ap.add_argument("--checkerboard_mm_scale_correction", type=float, default=0.5,
                    help="Multiply checkerboard-derived mm values by this factor.")
    ap.add_argument("--checkerboard_no_flip_planar_x", action="store_true",
                    help="Disable checkerboard planar-x sign flip.")

    ap.add_argument("--link_mode", type=str, default="symlink", choices=["symlink", "copy"])
    ap.add_argument("--save_analysis_config", action="store_true")

    ap.add_argument("--tip_refine_mode", type=str, default="parallel_centerline",
                    choices=["none", "edge_dt", "edge_grad", "mainray", "parallel_centerline"])
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

    args = ap.parse_args()
    print(f"[INFO] Using shadow_calibration module: {shadow_calibration_module.__file__}")

    if args.project_dir is None and args.raw_dir is None:
        raise SystemExit("Provide --project_dir or --raw_dir")

    model_cal = load_calibration(args.calibration)

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
    cal.tip_parallel_section_near_r = float(args.tip_refine_parallel_section_near_r)
    cal.tip_parallel_section_far_r = float(args.tip_refine_parallel_section_far_r)
    cal.tip_parallel_scan_half_r = float(args.tip_refine_parallel_scan_half_r)
    cal.tip_parallel_num_sections = int(args.tip_refine_parallel_num_sections)
    cal.tip_parallel_cross_step_px = float(args.tip_refine_parallel_cross_step_px)
    cal.tip_parallel_ray_step_px = float(args.tip_refine_parallel_ray_step_px)
    cal.tip_parallel_ray_max_len_r = float(args.tip_refine_parallel_ray_max_len_r)

    if args.tip_refine_mode not in ("none", "parallel_centerline"):
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
    else:
        print(
            "[INFO] Using native shadow_calibration analyze_data image processing "
            f"and annotation flow ({args.tip_refine_mode})."
        )

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
        raise RuntimeError("No tracked tip data found in cal.tip_locations_array_coarse after analyze_data_batch.")

    print("[INFO] Converting tracked tips to checkerboard-referenced mm...")
    tip_data = compute_tracked_tip_positions_mm(cal, tracked_rows, imgs)

    print("[INFO] Inferring local curling / uncurling direction from B...")
    records = enrich_records_with_motion_direction(tip_data["records"], c180_deg=float(model_cal.c_180_deg))

    print("[INFO] Computing per-orientation metrics...")
    orientation_metrics = compute_orientation_metrics(records)

    csv_path = processed_dir / "tracked_tip_positions_mm.csv"
    metrics_json_path = processed_dir / "b_hysteresis_metrics.json"
    sample_plot_path = processed_dir / "b_hysteresis_vs_sample.png"
    overlay_plot_path = processed_dir / "b_hysteresis_dual_c_overlay.png"

    save_tracked_tip_csv(csv_path, records)
    save_metrics_json(metrics_json_path, orientation_metrics, model_cal, cal, args)

    save_hysteresis_vs_sample_plot(sample_plot_path, records)
    save_dual_c_hysteresis_overlay_plot(
        overlay_plot_path,
        records=records,
        model_cal=model_cal,
        flip_rz_sign=bool(args.flip_rz_sign),
    )

    print(f"[INFO] Saved CSV: {csv_path}")
    print(f"[INFO] Saved metrics JSON: {metrics_json_path}")
    print(f"[INFO] Saved sample plot: {sample_plot_path}")
    print(f"[INFO] Saved hysteresis overlay plot: {overlay_plot_path}")

    if board_reference_debug_image is not None:
        try:
            annotated_board_path = processed_dir / "checkerboard_reference_annotated_analysis.png"
            draw_checkerboard_analysis_overlay(
                cal=cal,
                output_path=annotated_board_path,
                tracked_rows=tracked_rows,
                board_debug_image=board_reference_debug_image,
            )
            print(f"[INFO] Saved annotated checkerboard analysis image: {annotated_board_path}")
        except Exception as e:
            print(f"[WARN] Failed to create annotated checkerboard analysis image: {e}")

    print("\n========== RESULTS ==========")
    if not orientation_metrics:
        print("No valid orientation groups found.")
    else:
        for _, pm in sorted(orientation_metrics.items(), key=lambda kv: _orientation_sort_key(kv[0])):
            c_val = pm["c_orientation_deg"]
            print(f"{_orientation_display_label(c_val)}")
            print(f"  Samples: {pm['num_samples']}")
            print(f"  B range: {pm['b_range']}")
            print(f"  Tip angle range: {pm['tip_angle_range_deg']}")
            print(f"  Plot u range: {pm['u_plot_range_mm']}")
            print(f"  Plot z range: {pm['z_plot_range_mm']}")
            print(f"  Curling samples:   {pm['motion_groups']['curling']['num_samples']}")
            print(f"  Uncurling samples: {pm['motion_groups']['uncurling']['num_samples']}")
    print("=============================\n")

    print("[DONE] Offline dual-C B-only hysteresis analysis complete.")
    print(f"Outputs are in: {processed_dir}")


if __name__ == "__main__":
    main()
