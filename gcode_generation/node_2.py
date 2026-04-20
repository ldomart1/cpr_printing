#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code for the planar node figures using either:

  1) calibrated tip-position planning
       stage_xyz = desired_tip_xyz - offset_tip(B, C)

  2) direct Cartesian stage motion

Compared with the original working node script, this version adds:
  - exact calibration-based tip tracking using the same PCHIP / polynomial pull
    calibration logic used in the vasculature script
  - a second figure set rotated -90 degrees around each figure's local vertical
    trunk and offset in Y (default row-2 offset: -20 mm)
  - an additional third-row XZ-plane duplicate offset farther in -Y (default
    row-3 offset: -40 mm), with branch-like paths over-extended 2 mm into the
    main vertical before leaving the trunk
  - figure write ordering that prints vertical trunks from bottom -> top, then
    prints secondary branches from the trunk-side node outward/upward
  - optional fixed-orientation or tangent-following calibrated motion
  - side-approach travel planning between print paths, adapted from the
    vasculature script

Notes
-----
* The original figure set remains in the XZ plane at constant Y = origin_y.
* The rotated duplicate keeps the same station X positions, shifts the trunk
  baseline by `rotated_plane_y_offset`, and rotates each figure's lateral
  geometry from local +X into local -Y around its own trunk.
* Default calibrated orientation mode is `fixed`, which uses the pull-axis
  calibration to hold a constant tool orientation while still following the tip
  path exactly through the calibration model. Use `--orientation-mode tangent`
  if you want local tangent following like the vasculature writer.
* The added third-row XZ-plane copy pre-extends each branch / curve / arc
  start 2 mm downward into the trunk, then dwells at the original trunk
  junction before continuing outward.
* Use `--rows` to choose which rows are emitted: `all` (default), `1`, `2`,
  `3`, or comma-separated subsets such as `1,3`.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------- Defaults ----------------
DEFAULT_OUT = "gcode_generation/node_calibrated_pattern.gcode"

# Placement / axes
DEFAULT_ORIGIN_X = 65.0
DEFAULT_ORIGIN_Y = 80.0
DEFAULT_ORIGIN_Z = -190.0
DEFAULT_ROTATED_PLANE_Y_OFFSET = -20.0
DEFAULT_INCLUDE_ROTATED_PLANE = True
DEFAULT_THIRD_ROW_XZ_PLANE_Y_OFFSET = -40.0
DEFAULT_INCLUDE_THIRD_ROW_XZ_PLANE = True
DEFAULT_THIRD_ROW_BRANCH_START_EXTENSION_MM = 0.5
DEFAULT_ROWS = "all"
# Backward-compatible aliases from the previous revision.
DEFAULT_SECONDARY_XZ_PLANE_Y_OFFSET = DEFAULT_THIRD_ROW_XZ_PLANE_Y_OFFSET
DEFAULT_INCLUDE_SECONDARY_XZ_PLANE = DEFAULT_INCLUDE_THIRD_ROW_XZ_PLANE
DEFAULT_SECONDARY_XZ_BRANCH_START_EXTENSION_MM = DEFAULT_THIRD_ROW_BRANCH_START_EXTENSION_MM

# Startup / shutdown stage positions
DEFAULT_MACHINE_START_X = 20.0
DEFAULT_MACHINE_START_Y = 60.0
DEFAULT_MACHINE_START_Z = -30.0
DEFAULT_MACHINE_START_B = 0.0
DEFAULT_MACHINE_START_C = 0.0
DEFAULT_MACHINE_END_X = 110.0
DEFAULT_MACHINE_END_Y = 60.0
DEFAULT_MACHINE_END_Z = -30.0
DEFAULT_MACHINE_END_B = 0.0
DEFAULT_MACHINE_END_C = 0.0

# Motion
DEFAULT_TRAVEL_FEED = 1000.0
DEFAULT_APPROACH_FEED = 500.0
DEFAULT_FINE_APPROACH_FEED = 50.0
DEFAULT_PRINT_FEED = 250.0
DEFAULT_C_FEED = 15000.0
DEFAULT_BC_SOLVE_SAMPLES = 5001
DEFAULT_EDGE_SAMPLES = 1
DEFAULT_TRAVEL_LIFT_Z = 10.0
DEFAULT_SIDE_APPROACH_FAR = 7.0
DEFAULT_SIDE_APPROACH_NEAR = 4.0
DEFAULT_SIDE_RETREAT = 10.0
DEFAULT_SIDE_LIFT_Z = 10.0
DEFAULT_C_MAX_STEP_DEG = 15.0

# Sampling for curves
DEFAULT_CURVE_SAMPLES = 60
DEFAULT_ARC_SAMPLES = 30
DEFAULT_TANGENT_SMOOTH_WINDOW = 6
DEFAULT_CENTERLINE_SMOOTH_WINDOW = 0
DEFAULT_MIN_TANGENT_XY = 1e-6
DEFAULT_POINT_MERGE_TOL = 1e-9

# Extrusion / pressure
DEFAULT_EXTRUSION_PER_MM = 0.0015
DEFAULT_PRIME_MM = 0.2
DEFAULT_PRESSURE_OFFSET_MM = 4.0
DEFAULT_PRESSURE_ADVANCE_FEED = 120.0
DEFAULT_PRESSURE_RETRACT_FEED = 240.0
DEFAULT_PREFLOW_DWELL_MS = 500
DEFAULT_NODE_DWELL_MS = 1000
DEFAULT_EXTRUSION_MULTIPLIER_VERTICAL = 1.0
DEFAULT_EXTRUSION_MULTIPLIER_BRANCH = 1.0

# Figure geometry in the XZ plane, relative to origin at the small circle in the drawing.
DEFAULT_FIRST_VERTICAL_X = 5.0
DEFAULT_STATION_SPACING = 18.0
DEFAULT_MAIN_VERTICAL_HEIGHT = 40.0
DEFAULT_NODE_Z = 20.0

# Figure 5
DEFAULT_FIG5_CURVE_START_Z = 15.0
DEFAULT_FIG5_CURVE_END_DX = 10.0
DEFAULT_FIG5_CURVE_END_Z = 30.0
DEFAULT_FIG5_RIGHT_STUB_TOP_Z = 40.0
DEFAULT_FIG5_START_TANGENT_DZ = 10.0
DEFAULT_FIG5_END_TANGENT_DZ = 10.0

# Figure 6
DEFAULT_FIG6_CONNECT_Z = 27.273
DEFAULT_FIG6_ARC_DX = 5.0
DEFAULT_FIG6_ARC_DZ = 5.0
DEFAULT_FIG6_SLANTED_LEN = 8.0
DEFAULT_FIG6_SLANTED_ANGLE_FROM_VERTICAL = 15.0

# Bounding box defaults
DEFAULT_BBOX_X_MIN = -1e9
DEFAULT_BBOX_X_MAX = 1e9
DEFAULT_BBOX_Y_MIN = -1e9
DEFAULT_BBOX_Y_MAX = 1e9
DEFAULT_BBOX_Z_MIN = -1e9
DEFAULT_BBOX_Z_MAX = 1e9


# ---------------- Data classes ----------------
@dataclass(frozen=True)
class Point3:
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class PrimitivePath:
    label: str
    points: Tuple[Point3, ...]
    kind: str  # "vertical" or "branch"
    plane: str
    figure: str
    row_num: int
    lateral_axis: str
    end_is_node: bool = False
    junction_dwell_indices: Tuple[int, ...] = ()


@dataclass
class Calibration:
    pr: np.ndarray
    pz: np.ndarray
    py_off: Optional[np.ndarray]
    pa: Optional[np.ndarray]

    b_min: float
    b_max: float

    x_axis: str
    y_axis: str
    z_axis: str
    b_axis: str
    c_axis: str
    u_axis: str

    c_180_deg: float

    r_model: Optional[Dict[str, Any]] = None
    z_model: Optional[Dict[str, Any]] = None
    y_off_model: Optional[Dict[str, Any]] = None
    y_off_extrap_model: Optional[Dict[str, Any]] = None
    tip_angle_model: Optional[Dict[str, Any]] = None
    selected_fit_model: Optional[str] = None
    selected_offplane_fit_model: Optional[str] = None
    active_phase: str = "pull"
    offplane_y_sign: float = 1.0


# ---------------- Geometry helpers ----------------
def p3(x: float, y: float, z: float) -> Point3:
    return Point3(float(x), float(y), float(z))


def points_to_array(points: Sequence[Point3]) -> np.ndarray:
    return np.asarray([[p.x, p.y, p.z] for p in points], dtype=float)


def figure_station_x(first_x: float, spacing: float, idx: int) -> float:
    return float(first_x) + float(idx) * float(spacing)


def angle_from_vertical_to_dx_dz(length: float, angle_deg: float) -> Tuple[float, float]:
    """
    Convert an angle measured from +Z toward +local_lateral into d_lateral, dz.
    0 deg = vertical up, 90 deg = horizontal +lateral.
    """
    a = math.radians(float(angle_deg))
    d_lateral = float(length) * math.sin(a)
    dz = float(length) * math.cos(a)
    return d_lateral, dz


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def normalize2(vx: float, vz: float) -> Tuple[float, float]:
    n = math.hypot(vx, vz)
    if n <= 1e-12:
        raise ValueError("Zero-length vector.")
    return vx / n, vz / n


def sample_cubic_bezier_3d(
    p0: Point3,
    p1: Point3,
    p2: Point3,
    p3_: Point3,
    samples: int,
) -> Tuple[Point3, ...]:
    pts: List[Point3] = []
    n = max(2, int(samples))
    for i in range(n + 1):
        t = i / n
        u = 1.0 - t
        x = (
            u * u * u * p0.x
            + 3.0 * u * u * t * p1.x
            + 3.0 * u * t * t * p2.x
            + t * t * t * p3_.x
        )
        y = (
            u * u * u * p0.y
            + 3.0 * u * u * t * p1.y
            + 3.0 * u * t * t * p2.y
            + t * t * t * p3_.y
        )
        z = (
            u * u * u * p0.z
            + 3.0 * u * u * t * p1.z
            + 3.0 * u * t * t * p2.z
            + t * t * t * p3_.z
        )
        pts.append(p3(x, y, z))
    return tuple(pts)


def sample_two_point_vertical_tangent_curve(
    start: Point3,
    end: Point3,
    samples: int,
    start_tangent_vec: np.ndarray,
    end_tangent_vec: np.ndarray,
    start_handle_len: float,
    end_handle_len: float,
) -> Tuple[Point3, ...]:
    st = normalize(np.asarray(start_tangent_vec, dtype=float))
    et = normalize(np.asarray(end_tangent_vec, dtype=float))
    c1 = p3(
        start.x + st[0] * float(start_handle_len),
        start.y + st[1] * float(start_handle_len),
        start.z + st[2] * float(start_handle_len),
    )
    c2 = p3(
        end.x - et[0] * float(end_handle_len),
        end.y - et[1] * float(end_handle_len),
        end.z - et[2] * float(end_handle_len),
    )
    return sample_cubic_bezier_3d(start, c1, c2, end, samples)


def sample_tangent_arc_between_points(
    start: Point3,
    end: Point3,
    start_tangent: np.ndarray,
    end_tangent: np.ndarray,
    samples: int,
) -> Tuple[Point3, ...]:
    st = normalize(np.asarray(start_tangent, dtype=float))
    et = normalize(np.asarray(end_tangent, dtype=float))
    chord = float(np.linalg.norm(np.asarray([end.x - start.x, end.y - start.y, end.z - start.z], dtype=float)))
    handle = max(0.5, 0.4 * chord)
    c1 = p3(start.x + st[0] * handle, start.y + st[1] * handle, start.z + st[2] * handle)
    c2 = p3(end.x - et[0] * handle, end.y - et[1] * handle, end.z - et[2] * handle)
    return sample_cubic_bezier_3d(start, c1, c2, end, samples)


def deduplicate_polyline_points(points: np.ndarray, tol: float = DEFAULT_POINT_MERGE_TOL) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return pts.copy()
    out = [pts[0].copy()]
    for p in pts[1:]:
        if float(np.linalg.norm(p - out[-1])) > float(tol):
            out.append(np.asarray(p, dtype=float).copy())
    return np.asarray(out, dtype=float)


def smooth_centerline_points(points: np.ndarray, window: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 2 or int(window) <= 0:
        return pts.copy()
    w = int(window)
    out = pts.copy()
    for i in range(len(pts)):
        i0 = max(0, i - w)
        i1 = min(len(pts), i + w + 1)
        out[i] = pts[i0:i1].mean(axis=0)
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def tangent_for_index(points: np.ndarray, i: int, smooth_window: int) -> np.ndarray:
    n = len(points)
    i0 = max(0, i - int(smooth_window))
    i1 = min(n - 1, i + int(smooth_window))
    if i1 == i0:
        if i == 0 and n > 1:
            return normalize(points[1] - points[0])
        if i == n - 1 and n > 1:
            return normalize(points[-1] - points[-2])
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return normalize(points[i1] - points[i0])


def build_tangents_for_points(points: np.ndarray, smooth_window: int, centerline_smooth_window: int = DEFAULT_CENTERLINE_SMOOTH_WINDOW) -> np.ndarray:
    tangent_points = smooth_centerline_points(points, window=centerline_smooth_window)
    tangents = np.zeros_like(tangent_points)
    for i in range(len(tangent_points)):
        tangents[i] = tangent_for_index(tangent_points, i, smooth_window=max(1, int(smooth_window)))
    return tangents


def unwrap_angle_deg_near(target_deg: float, reference_deg: float) -> float:
    target = float(target_deg)
    ref = float(reference_deg)
    while target - ref > 180.0:
        target -= 360.0
    while target - ref < -180.0:
        target += 360.0
    return float(target)


def limit_angle_step_deg(target_deg: float, reference_deg: float, max_step_deg: float) -> float:
    target = unwrap_angle_deg_near(target_deg, reference_deg)
    if float(max_step_deg) <= 0.0:
        return float(target)
    delta = target - float(reference_deg)
    if delta > float(max_step_deg):
        return float(reference_deg) + float(max_step_deg)
    if delta < -float(max_step_deg):
        return float(reference_deg) - float(max_step_deg)
    return float(target)


def desired_physical_b_angle_from_tangent(tangent: np.ndarray) -> float:
    """
    Calibration tip-angle convention used for B solving:
      tip_angle = 0 deg   -> tip points along +Z (vertical up)
      tip_angle = 90 deg  -> tip is horizontal
      tip_angle = 180 deg -> tip points along -Z (vertical down)

    This is the target angle passed into the calibration pull/tip-angle model.
    """
    t = normalize(np.asarray(tangent, dtype=float))
    tz = float(np.clip(t[2], -1.0, 1.0))
    return float(math.degrees(math.acos(tz)))


def c_angle_from_tangent(tangent: np.ndarray, prev_c: float = 0.0, min_xy: float = DEFAULT_MIN_TANGENT_XY) -> float:
    xy = np.asarray(tangent[:2], dtype=float)
    if float(np.linalg.norm(xy)) < float(min_xy):
        return float(prev_c)
    raw = float(math.degrees(math.atan2(float(xy[1]), float(xy[0]))))
    return unwrap_angle_deg_near(raw, prev_c)


def side_vector_from_tangent(tangent: np.ndarray, fallback: Optional[np.ndarray] = None, min_xy: float = DEFAULT_MIN_TANGENT_XY) -> np.ndarray:
    xy = np.asarray(tangent[:2], dtype=float)
    nxy = float(np.linalg.norm(xy))
    if nxy < float(min_xy):
        if fallback is not None and float(np.linalg.norm(np.asarray(fallback[:2], dtype=float))) >= float(min_xy):
            xy = np.asarray(fallback[:2], dtype=float)
            nxy = float(np.linalg.norm(xy))
        else:
            return np.array([1.0, 0.0, 0.0], dtype=float)
    tx, ty = xy / nxy
    return np.array([-ty, tx, 0.0], dtype=float)


# ---------------- Calibration helpers ----------------
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


def _select_named_model(models: Dict[str, Any], base_name: str, selected_fit_model: Optional[str]) -> Optional[Dict[str, Any]]:
    selected = None if selected_fit_model is None else str(selected_fit_model).strip().lower()
    candidates: List[str] = []
    if selected:
        candidates.append(f"{base_name}_{selected}")
    candidates.append(base_name)
    for key in candidates:
        spec = _normalize_model_spec(models.get(key))
        if spec is not None:
            return spec
    return None


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
    t = np.clip(t, 0.0, 1.0)

    h00 = (2.0 * t**3) - (3.0 * t**2) + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = (-2.0 * t**3) + (3.0 * t**2)
    h11 = t**3 - t**2
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


def eval_pchip_with_linear_extrap(model_spec: Dict[str, Any], extrap_model_spec: Optional[Dict[str, Any]], b: Any) -> np.ndarray:
    x_knots = np.asarray(model_spec.get("x_knots", []), dtype=float).reshape(-1)
    if x_knots.size == 0 or extrap_model_spec is None:
        return eval_model_spec(model_spec, b, default_if_none=0.0)
    b_arr = np.asarray(b, dtype=float)
    pchip_values = eval_model_spec(model_spec, b_arr, default_if_none=0.0)
    out = np.asarray(pchip_values, dtype=float).copy()
    x_min = float(np.min(x_knots))
    x_max = float(np.max(x_knots))
    outside = (b_arr < x_min) | (b_arr > x_max)
    if np.any(outside):
        extrap_values = eval_model_spec(extrap_model_spec, b_arr, default_if_none=0.0)
        out = np.where(outside, extrap_values, out)
    return out


def load_calibration(json_path: str) -> Calibration:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cubic = data.get("cubic_coefficients", {}) or {}
    fit_models = data.get("fit_models", {}) or {}

    def _coeffs_from_model(models: Dict[str, Any], *names: str) -> Optional[np.ndarray]:
        for name in names:
            spec = _normalize_model_spec(models.get(name))
            if spec is None:
                continue
            coeffs = spec.get("coefficients", spec.get("coeffs"))
            if coeffs is not None:
                arr = np.asarray(coeffs, dtype=float).reshape(-1)
                if arr.size > 0:
                    return arr
        return None

    pr_arr = cubic.get("r_coeffs")
    if pr_arr is None:
        pr = _coeffs_from_model(fit_models, "r_cubic", "r_avg_cubic")
        if pr is None:
            pr = np.zeros(1, dtype=float)
    else:
        pr = np.asarray(pr_arr, dtype=float)

    pz_arr = cubic.get("z_coeffs")
    if pz_arr is None:
        pz = _coeffs_from_model(fit_models, "z_cubic", "z_avg_cubic")
        if pz is None:
            pz = np.zeros(1, dtype=float)
    else:
        pz = np.asarray(pz_arr, dtype=float)

    py_off_raw = cubic.get("offplane_y_coeffs", None)
    if py_off_raw is None:
        py_off = _coeffs_from_model(fit_models, "offplane_y_cubic", "offplane_y_avg_cubic", "offplane_y", "offplane_y_linear", "offplane_y_avg_linear")
    else:
        py_off = np.asarray(py_off_raw, dtype=float)

    pa_raw = cubic.get("tip_angle_coeffs", None)
    if pa_raw is None:
        pa = _coeffs_from_model(fit_models, "tip_angle_cubic", "tip_angle_avg_cubic")
    else:
        pa = np.asarray(pa_raw, dtype=float)

    selected_fit_model = data.get("selected_fit_model")
    selected_fit_model = None if selected_fit_model is None else str(selected_fit_model).strip().lower()
    selected_offplane_fit_model = data.get("selected_offplane_fit_model")
    selected_offplane_fit_model = None if selected_offplane_fit_model is None else str(selected_offplane_fit_model).strip().lower()
    active_phase = str(data.get("default_phase_for_legacy_access") or "pull").strip().lower()

    phase_models = data.get("fit_models_by_phase", {}) or {}
    active_phase_models = phase_models.get(active_phase) if isinstance(phase_models, dict) else None
    if not isinstance(active_phase_models, dict):
        active_phase_models = fit_models

    r_model = _select_named_model(active_phase_models, "r", selected_fit_model)
    z_model = _select_named_model(active_phase_models, "z", selected_fit_model)
    y_off_selector = selected_offplane_fit_model or selected_fit_model
    y_off_model = _select_named_model(active_phase_models, "offplane_y", y_off_selector)
    y_off_extrap_model = _normalize_model_spec(active_phase_models.get("offplane_y_linear"))
    if y_off_extrap_model is None:
        y_off_extrap_model = _normalize_model_spec(active_phase_models.get("offplane_y"))
    tip_angle_model = _select_named_model(active_phase_models, "tip_angle", selected_fit_model)

    motor_setup = data.get("motor_setup", {})
    duet_map = data.get("duet_axis_mapping", {})
    b_range = motor_setup.get("b_motor_position_range", [-5.4, 0.0])
    b_min, b_max = map(float, b_range)
    if b_min > b_max:
        b_min, b_max = b_max, b_min

    return Calibration(
        pr=pr,
        pz=pz,
        py_off=py_off,
        pa=pa,
        r_model=r_model,
        z_model=z_model,
        y_off_model=y_off_model,
        y_off_extrap_model=y_off_extrap_model,
        tip_angle_model=tip_angle_model,
        selected_fit_model=selected_fit_model,
        selected_offplane_fit_model=selected_offplane_fit_model,
        active_phase=active_phase,
        b_min=b_min,
        b_max=b_max,
        x_axis=str(duet_map.get("horizontal_axis") or motor_setup.get("horizontal_axis") or "X"),
        y_axis=str(duet_map.get("depth_axis") or motor_setup.get("depth_axis") or "Y"),
        z_axis=str(duet_map.get("vertical_axis") or motor_setup.get("vertical_axis") or "Z"),
        b_axis=str(duet_map.get("pull_axis") or motor_setup.get("b_motor_axis") or "B"),
        c_axis=str(duet_map.get("rotation_axis") or motor_setup.get("rotation_axis") or "C"),
        u_axis=str(duet_map.get("extruder_axis") or "U"),
        c_180_deg=float(motor_setup.get("rotation_axis_180_deg", 180.0)),
    )


def eval_r(cal: Calibration, b: Any) -> np.ndarray:
    if cal.r_model is not None:
        return eval_model_spec(cal.r_model, b)
    return poly_eval(cal.pr, b)


def eval_z(cal: Calibration, b: Any) -> np.ndarray:
    if cal.z_model is not None:
        return eval_model_spec(cal.z_model, b)
    return poly_eval(cal.pz, b)


def eval_offplane_y(cal: Calibration, b: Any) -> np.ndarray:
    if cal.y_off_model is not None:
        if str(cal.y_off_model.get("model_type", "")).lower() == "pchip":
            values = eval_pchip_with_linear_extrap(cal.y_off_model, cal.y_off_extrap_model, b)
        else:
            values = eval_model_spec(cal.y_off_model, b, default_if_none=0.0)
    else:
        values = poly_eval(cal.py_off, b, default_if_none=0.0)
    return float(cal.offplane_y_sign) * np.asarray(values, dtype=float)


def eval_tip_angle_deg(cal: Calibration, b: Any) -> np.ndarray:
    if cal.tip_angle_model is not None:
        return eval_model_spec(cal.tip_angle_model, b)
    if cal.pa is None:
        raise ValueError("Calibration is missing tip_angle_coeffs.")
    return poly_eval(cal.pa, b)


def calibration_model_range_warnings(cal: Calibration) -> List[str]:
    warnings: List[str] = []
    for label, model in (
        ("r", cal.r_model),
        ("z", cal.z_model),
        ("offplane_y", cal.y_off_model),
        ("tip_angle", cal.tip_angle_model),
    ):
        if not isinstance(model, dict) or str(model.get("model_type", "")).lower() != "pchip":
            continue
        x = np.asarray(model.get("x_knots", []), dtype=float).reshape(-1)
        if x.size == 0:
            continue
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if float(cal.b_min) < x_min or float(cal.b_max) > x_max:
            if label == "offplane_y" and cal.y_off_extrap_model is not None:
                warnings.append(
                    f"{label} PCHIP knot range [{x_min:.3f}, {x_max:.3f}] does not cover B range "
                    f"[{float(cal.b_min):.3f}, {float(cal.b_max):.3f}]; values outside the knot range use the linear offplane_y fit."
                )
            else:
                warnings.append(
                    f"{label} PCHIP knot range [{x_min:.3f}, {x_max:.3f}] does not cover B range "
                    f"[{float(cal.b_min):.3f}, {float(cal.b_max):.3f}]; values outside the knot range are clamped."
                )
    return warnings


def describe_model(model: Optional[Dict[str, Any]], fallback_name: str) -> str:
    if model is None:
        return f"{fallback_name}: legacy polynomial fallback"
    model_type = str(model.get("model_type") or "unknown").strip().lower()
    eq = str(model.get("equation") or "").strip()
    x_range = model.get("fit_x_range")
    range_str = ""
    if isinstance(x_range, (list, tuple)) and len(x_range) >= 2:
        range_str = f", fit_x_range=[{float(x_range[0]):.3f}, {float(x_range[1]):.3f}]"
    return f"{fallback_name}: {model_type}{range_str}{'; ' + eq if eq else ''}"


def predict_tip_offset_xyz(cal: Calibration, b: float, c_deg: float) -> np.ndarray:
    r = float(eval_r(cal, b))
    z = float(eval_z(cal, b))
    y_off = float(eval_offplane_y(cal, b))
    c = math.radians(float(c_deg))
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def stage_xyz_for_tip(cal: Calibration, tip_xyz: np.ndarray, b: float, c_deg: float) -> np.ndarray:
    return np.asarray(tip_xyz, dtype=float) - predict_tip_offset_xyz(cal, b, c_deg)


def solve_b_for_target_tip_angle(cal: Calibration, target_angle_deg: float, search_samples: int = DEFAULT_BC_SOLVE_SAMPLES) -> float:
    b_lo, b_hi = float(cal.b_min), float(cal.b_max)
    bb = np.linspace(b_lo, b_hi, int(max(101, search_samples)))
    aa = eval_tip_angle_deg(cal, bb) - float(target_angle_deg)
    i_best_abs = int(np.argmin(np.abs(aa)))
    b_best_abs = float(bb[i_best_abs])

    sign_changes: List[Tuple[float, float, float]] = []
    for i in range(len(bb) - 1):
        a0 = float(aa[i])
        a1 = float(aa[i + 1])
        if a0 == 0.0:
            return float(bb[i])
        if a0 * a1 < 0.0:
            score = min(abs(a0), abs(a1))
            sign_changes.append((score, float(bb[i]), float(bb[i + 1])))

    if sign_changes:
        sign_changes.sort(key=lambda t: t[0])
        _, lo, hi = sign_changes[0]

        def f(x: float) -> float:
            return float(eval_tip_angle_deg(cal, x) - float(target_angle_deg))

        flo = f(lo)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fmid = f(mid)
            if abs(fmid) < 1e-10:
                return float(mid)
            if flo * fmid <= 0.0:
                hi = mid
            else:
                lo = mid
                flo = fmid
        return float(0.5 * (lo + hi))

    return b_best_abs


# ---------------- Pattern builders ----------------
def local_point(station_x: float, plane_y: float, lateral: float, z: float, lateral_axis: str) -> Point3:
    axis = str(lateral_axis).strip().lower()
    if axis == "x":
        return p3(station_x + float(lateral), plane_y, z)
    if axis == "y":
        return p3(station_x, plane_y + float(lateral), z)
    if axis in {"-y", "y_neg", "neg_y"}:
        return p3(station_x, plane_y - float(lateral), z)
    raise ValueError(f"Unsupported lateral axis: {lateral_axis}")


def local_vector(lateral: float, z: float, lateral_axis: str) -> np.ndarray:
    axis = str(lateral_axis).strip().lower()
    if axis == "x":
        return np.array([float(lateral), 0.0, float(z)], dtype=float)
    if axis == "y":
        return np.array([0.0, float(lateral), float(z)], dtype=float)
    if axis in {"-y", "y_neg", "neg_y"}:
        return np.array([0.0, -float(lateral), float(z)], dtype=float)
    raise ValueError(f"Unsupported lateral axis: {lateral_axis}")



def parse_row_selection(rows: Any) -> Optional[Tuple[int, ...]]:
    raw = str(rows).strip().lower()
    if raw in {"", "all"}:
        return None
    pieces = [piece.strip() for piece in raw.replace(";", ",").split(",") if piece.strip()]
    if not pieces:
        return None
    out: List[int] = []
    seen: set[int] = set()
    for piece in pieces:
        if piece == "all":
            return None
        try:
            row = int(piece)
        except ValueError as exc:
            raise ValueError(f"Unsupported row selector {piece!r}; use all or a comma-separated subset of 1,2,3.") from exc
        if row not in {1, 2, 3}:
            raise ValueError(f"Unsupported row selector {piece!r}; allowed rows are 1, 2, 3, or all.")
        if row not in seen:
            out.append(row)
            seen.add(row)
    if not out:
        return None
    return tuple(out)


def build_plane_pattern(
    origin_x: float,
    plane_y: float,
    origin_z: float,
    plane_name: str,
    row_num: int,
    lateral_axis: str,
    first_vertical_x: float,
    station_spacing: float,
    main_vertical_height: float,
    node_z: float,
    curve_samples: int,
    arc_samples: int,
    branch_start_extension_mm: float = 0.0,
) -> List[PrimitivePath]:
    ops: List[PrimitivePath] = []
    y = float(plane_y)
    ox = float(origin_x)
    oz = float(origin_z)
    branch_start_extension_mm = max(0.0, float(branch_start_extension_mm))

    def make_path(
        label: str,
        pts: Sequence[Point3],
        kind: str,
        figure: str,
        end_is_node: bool = False,
        junction_dwell_indices: Sequence[int] = (),
    ) -> PrimitivePath:
        return PrimitivePath(
            label=label,
            points=tuple(pts),
            kind=kind,
            plane=plane_name,
            figure=figure,
            row_num=int(row_num),
            lateral_axis=str(lateral_axis),
            end_is_node=bool(end_is_node),
            junction_dwell_indices=tuple(int(i) for i in junction_dwell_indices),
        )

    def prepend_tangent_extension(points: Sequence[Point3]) -> Tuple[Tuple[Point3, ...], Tuple[int, ...]]:
        pts = tuple(points)
        if len(pts) < 2 or branch_start_extension_mm <= 0.0:
            return pts, ()
        p0 = np.array([pts[0].x, pts[0].y, pts[0].z], dtype=float)
        p1 = np.array([pts[1].x, pts[1].y, pts[1].z], dtype=float)
        tangent = normalize(p1 - p0)
        if float(np.linalg.norm(tangent)) <= 1e-12:
            return pts, ()
        ext_xyz = p0 - float(branch_start_extension_mm) * tangent
        ext = p3(ext_xyz[0], ext_xyz[1], ext_xyz[2])
        return (ext,) + pts, (1,)

    angled_specs = [
        (30.0, 20.0, "fig1"),
        (45.0, 16.0, "fig2"),
        (15.0, 20.706, "fig3"),
        (90.0, 10.0, "fig4"),
    ]
    for idx, (angle_deg, length_mm, tag) in enumerate(angled_specs):
        x_base = ox + figure_station_x(first_vertical_x, station_spacing, idx)
        top = local_point(x_base, y, 0.0, oz + main_vertical_height, lateral_axis)
        base = local_point(x_base, y, 0.0, oz, lateral_axis)
        ops.append(make_path(f"{plane_name}_{tag}_vertical", [base, top], "vertical", tag))

        node = local_point(x_base, y, 0.0, oz + node_z, lateral_axis)
        d_lateral, dz = angle_from_vertical_to_dx_dz(length_mm, angle_deg)
        free = local_point(x_base, y, d_lateral, oz + node_z + dz, lateral_axis)
        branch_pts, dwell_idx = prepend_tangent_extension((node, free))
        ops.append(make_path(f"{plane_name}_{tag}_branch_{angle_deg:.0f}deg", branch_pts, "branch", tag, junction_dwell_indices=dwell_idx))

    x5 = ox + figure_station_x(first_vertical_x, station_spacing, 4)
    fig5_top = local_point(x5, y, 0.0, oz + main_vertical_height, lateral_axis)
    fig5_base = local_point(x5, y, 0.0, oz, lateral_axis)
    ops.append(make_path(f"{plane_name}_fig5_left_vertical", [fig5_base, fig5_top], "vertical", "fig5"))

    x5r_lateral = DEFAULT_FIG5_CURVE_END_DX
    stub_top = local_point(x5, y, x5r_lateral, oz + DEFAULT_FIG5_RIGHT_STUB_TOP_Z, lateral_axis)
    curve_start = local_point(x5, y, 0.0, oz + DEFAULT_FIG5_CURVE_START_Z, lateral_axis)
    curve_end = local_point(x5, y, x5r_lateral, oz + DEFAULT_FIG5_CURVE_END_Z, lateral_axis)
    vertical_up = np.array([0.0, 0.0, 1.0], dtype=float)
    fig5_curve_pts = sample_two_point_vertical_tangent_curve(
        curve_start,
        curve_end,
        curve_samples,
        start_tangent_vec=vertical_up,
        end_tangent_vec=vertical_up,
        start_handle_len=DEFAULT_FIG5_START_TANGENT_DZ,
        end_handle_len=DEFAULT_FIG5_END_TANGENT_DZ,
    )
    fig5_curve_then_vertical_branch = tuple(list(fig5_curve_pts) + [stub_top])
    fig5_curve_then_vertical_branch, fig5_dwell_idx = prepend_tangent_extension(fig5_curve_then_vertical_branch)
    ops.append(make_path(
        f"{plane_name}_fig5_curve_then_vertical_branch",
        fig5_curve_then_vertical_branch,
        "branch",
        "fig5",
        junction_dwell_indices=fig5_dwell_idx,
    ))

    x6 = ox + figure_station_x(first_vertical_x, station_spacing, 5)
    fig6_top = local_point(x6, y, 0.0, oz + main_vertical_height, lateral_axis)
    fig6_base = local_point(x6, y, 0.0, oz, lateral_axis)
    ops.append(make_path(f"{plane_name}_fig6_left_vertical", [fig6_base, fig6_top], "vertical", "fig6"))

    connect_z = oz + DEFAULT_FIG6_CONNECT_Z
    arc_start = local_point(x6, y, 0.0, connect_z, lateral_axis)
    arc_end = local_point(x6, y, DEFAULT_FIG6_ARC_DX, connect_z + DEFAULT_FIG6_ARC_DZ, lateral_axis)
    branch_d_lateral, branch_dz = angle_from_vertical_to_dx_dz(
        DEFAULT_FIG6_SLANTED_LEN,
        DEFAULT_FIG6_SLANTED_ANGLE_FROM_VERTICAL,
    )
    branch_tip = local_point(x6, y, DEFAULT_FIG6_ARC_DX + branch_d_lateral, connect_z + DEFAULT_FIG6_ARC_DZ + branch_dz, lateral_axis)

    arc_start_tangent = local_vector(1.0, 0.0, lateral_axis)
    arc_end_tangent = normalize(local_vector(branch_d_lateral, branch_dz, lateral_axis))
    fig6_arc_pts = sample_tangent_arc_between_points(
        start=arc_start,
        end=arc_end,
        start_tangent=arc_start_tangent,
        end_tangent=arc_end_tangent,
        samples=arc_samples,
    )
    fig6_arc_then_branch = tuple(list(fig6_arc_pts) + [branch_tip])
    fig6_arc_then_branch, fig6_dwell_idx = prepend_tangent_extension(fig6_arc_then_branch)
    ops.append(make_path(
        f"{plane_name}_fig6_arc_then_slanted_branch",
        fig6_arc_then_branch,
        "branch",
        "fig6",
        junction_dwell_indices=fig6_dwell_idx,
    ))

    return ops


def build_all_patterns(
    origin_x: float,
    origin_y: float,
    origin_z: float,
    include_rotated_plane: bool,
    rotated_plane_y_offset: float,
    include_third_row_xz_plane: bool,
    third_row_xz_plane_y_offset: float,
    third_row_branch_start_extension_mm: float,
    rows: Any,
    first_vertical_x: float,
    station_spacing: float,
    main_vertical_height: float,
    node_z: float,
    curve_samples: int,
    arc_samples: int,
) -> List[PrimitivePath]:
    selected_rows = parse_row_selection(rows)
    rows_override_legacy_includes = selected_rows is not None

    def row_enabled(row_num: int, legacy_enabled: bool = True) -> bool:
        if rows_override_legacy_includes:
            return row_num in selected_rows
        return bool(legacy_enabled)

    ops: List[PrimitivePath] = []
    if row_enabled(1, True):
        ops.extend(build_plane_pattern(
            origin_x=origin_x,
            plane_y=origin_y,
            origin_z=origin_z,
            plane_name="plane0_xz",
            row_num=1,
            lateral_axis="x",
            first_vertical_x=first_vertical_x,
            station_spacing=station_spacing,
            main_vertical_height=main_vertical_height,
            node_z=node_z,
            curve_samples=curve_samples,
            arc_samples=arc_samples,
            branch_start_extension_mm=0.0,
        ))
    if row_enabled(2, include_rotated_plane):
        ops.extend(
            build_plane_pattern(
                origin_x=origin_x,
                plane_y=origin_y + float(rotated_plane_y_offset),
                origin_z=origin_z,
                plane_name="plane1_rotm90",
                row_num=2,
                lateral_axis="y_neg",
                first_vertical_x=first_vertical_x,
                station_spacing=station_spacing,
                main_vertical_height=main_vertical_height,
                node_z=node_z,
                curve_samples=curve_samples,
                arc_samples=arc_samples,
                branch_start_extension_mm=0.0,
            )
        )
    if row_enabled(3, include_third_row_xz_plane):
        ops.extend(
            build_plane_pattern(
                origin_x=origin_x,
                plane_y=origin_y + float(third_row_xz_plane_y_offset),
                origin_z=origin_z,
                plane_name="plane2_rotm90_neg_y",
                row_num=3,
                lateral_axis="y_neg",
                first_vertical_x=first_vertical_x,
                station_spacing=station_spacing,
                main_vertical_height=main_vertical_height,
                node_z=node_z,
                curve_samples=curve_samples,
                arc_samples=arc_samples,
                branch_start_extension_mm=third_row_branch_start_extension_mm,
            )
        )
    return ops


# ---------------- G-code helpers ----------------
def _fmt_axes_move(axes_vals: Sequence[Tuple[str, float]]) -> str:
    return " ".join(f"{ax}{val:.3f}" for ax, val in axes_vals)


def row_branch_approach_axis(row_num: int, lateral_axis: str) -> np.ndarray:
    if int(row_num) == 1:
        return np.array([1.0, 0.0, 0.0], dtype=float)
    # Rows 2 and 3 always retreat/approach from negative Y in tip-space.
    return np.array([0.0, -1.0, 0.0], dtype=float)


def branch_approach_offset_direction(start_tangent: np.ndarray, row_num: int, lateral_axis: str) -> np.ndarray:
    t = normalize(np.asarray(start_tangent, dtype=float))
    axis = row_branch_approach_axis(row_num, lateral_axis)
    lateral_mag = abs(float(np.dot(t, axis)))
    vertical_mag = abs(float(t[2]))
    # Place the pre-node approach point laterally offset and above the node so the
    # final move descends into the junction instead of climbing upward to it.
    d = axis * max(1e-9, lateral_mag) + np.array([0.0, 0.0, max(1e-9, vertical_mag)], dtype=float)
    return normalize(d)


class GCodeWriter:
    def __init__(
        self,
        fh,
        cal: Calibration,
        bbox: Dict[str, float],
        travel_feed: float,
        approach_feed: float,
        fine_approach_feed: float,
        print_feed: float,
        c_feed: float,
        edge_samples: int,
        extrusion_per_mm: float,
        pressure_offset_mm: float,
        pressure_advance_feed: float,
        pressure_retract_feed: float,
        preflow_dwell_ms: int,
        node_dwell_ms: int,
        emit_extrusion: bool,
        write_mode: str,
        orientation_mode: str,
        bc_solve_samples: int,
        c_max_step_deg: float,
    ) -> None:
        self.f = fh
        self.cal = cal
        self.bbox = bbox
        self.travel_feed = float(travel_feed)
        self.approach_feed = float(approach_feed)
        self.fine_approach_feed = float(fine_approach_feed)
        self.print_feed = float(print_feed)
        self.c_feed = float(c_feed)
        self.edge_samples = max(1, int(edge_samples))
        self.extrusion_per_mm = float(extrusion_per_mm)
        self.pressure_offset_mm = float(pressure_offset_mm)
        self.pressure_advance_feed = float(pressure_advance_feed)
        self.pressure_retract_feed = float(pressure_retract_feed)
        self.preflow_dwell_ms = int(preflow_dwell_ms)
        self.node_dwell_ms = int(node_dwell_ms)
        self.emit_extrusion = bool(emit_extrusion)
        self.write_mode = str(write_mode).strip().lower()
        self.orientation_mode = str(orientation_mode).strip().lower()
        self.bc_solve_samples = int(bc_solve_samples)
        self.c_max_step_deg = float(c_max_step_deg)

        self.u_material_abs = 0.0
        self.pressure_charged = False
        self.cur_stage_xyz: Optional[np.ndarray] = None
        self.cur_tip_xyz: Optional[np.ndarray] = None
        self.cur_b: float = 0.0
        self.cur_c: float = 0.0
        self.last_tip_tangent: Optional[np.ndarray] = None
        self.warnings: List[str] = []
        self.total_paths_written = 0
        self.stage_min = np.array([np.inf, np.inf, np.inf], dtype=float)
        self.stage_max = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
        self.b_min_used = float("inf")
        self.b_max_used = float("-inf")
        self.c_min_used = float("inf")
        self.c_max_used = float("-inf")

        self.fixed_b = solve_b_for_target_tip_angle(self.cal, 0.0, search_samples=self.bc_solve_samples) if self.write_mode == "calibrated" else 0.0
        self.fixed_c = 90.0

    def orientation_mode_for_path(self, kind: Optional[str] = None, row_num: Optional[int] = None) -> str:
        mode = str(self.orientation_mode).strip().lower()
        if self.write_mode != "calibrated":
            return mode
        row_val = None if row_num is None else int(row_num)
        if row_val == 1:
            return "fixed"
        if row_val in {2, 3}:
            return "tangent"
        return mode

    def clamp_stage(self, p_stage: np.ndarray, context: str) -> np.ndarray:
        x = float(np.clip(p_stage[0], self.bbox["x_min"], self.bbox["x_max"]))
        y = float(np.clip(p_stage[1], self.bbox["y_min"], self.bbox["y_max"]))
        z = float(np.clip(p_stage[2], self.bbox["z_min"], self.bbox["z_max"]))
        if abs(x - float(p_stage[0])) > 1e-12 or abs(y - float(p_stage[1])) > 1e-12 or abs(z - float(p_stage[2])) > 1e-12:
            self.warnings.append(f"WARNING: {context} stage point clamped to bbox.")
        return np.array([x, y, z], dtype=float)

    def u_cmd_actual(self) -> float:
        return self.u_material_abs + (self.pressure_offset_mm if self.pressure_charged else 0.0)

    def bc_for_tangent(
        self,
        tangent: np.ndarray,
        prev_c: Optional[float] = None,
        orientation_mode_override: Optional[str] = None,
    ) -> Tuple[float, float]:
        if self.write_mode != "calibrated":
            return 0.0, 0.0
        mode = str(self.orientation_mode if orientation_mode_override is None else orientation_mode_override).strip().lower()
        if mode == "fixed":
            return float(self.fixed_b), float(self.fixed_c)
        target_b = desired_physical_b_angle_from_tangent(tangent)
        b = solve_b_for_target_tip_angle(self.cal, target_b, search_samples=self.bc_solve_samples)
        return float(b), float(self.fixed_c)

    def tip_to_stage(
        self,
        p_tip: np.ndarray,
        tangent: Optional[np.ndarray] = None,
        prev_c: Optional[float] = None,
        orientation_mode_override: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, float]:
        if self.write_mode == "cartesian":
            p_stage = self.clamp_stage(np.asarray(p_tip, dtype=float), "cartesian_tip_to_stage")
            b, c = (0.0, 0.0) if tangent is None else self.bc_for_tangent(
                tangent,
                prev_c=prev_c,
                orientation_mode_override=orientation_mode_override,
            )
            return p_stage, float(b), float(c)

        if tangent is None:
            mode = str(self.orientation_mode if orientation_mode_override is None else orientation_mode_override).strip().lower()
            if mode == "fixed":
                b, c = float(self.fixed_b), float(self.fixed_c)
            else:
                b, c = self.bc_for_tangent(
                    np.array([0.0, 0.0, 1.0], dtype=float),
                    prev_c=prev_c,
                    orientation_mode_override=orientation_mode_override,
                )
        else:
            b, c = self.bc_for_tangent(
                tangent,
                prev_c=prev_c,
                orientation_mode_override=orientation_mode_override,
            )
        p_stage = stage_xyz_for_tip(self.cal, np.asarray(p_tip, dtype=float), b, c)
        return self.clamp_stage(p_stage, "tip_to_stage"), float(b), float(c)

    def write_move(self, p_stage: np.ndarray, b: float, c: float, feed: float, comment: Optional[str] = None, u_value: Optional[float] = None) -> None:
        if comment:
            self.f.write(f"; {comment}\n")
        axes: List[Tuple[str, float]] = [
            (self.cal.x_axis, float(p_stage[0])),
            (self.cal.y_axis, float(p_stage[1])),
            (self.cal.z_axis, float(p_stage[2])),
        ]
        if self.write_mode == "calibrated":
            axes.extend([(self.cal.b_axis, float(b)), (self.cal.c_axis, float(c))])
        if u_value is not None:
            axes.append((self.cal.u_axis, float(u_value)))
        self.f.write(f"G1 {_fmt_axes_move(axes)} F{float(feed):.0f}\n")
        p_stage_arr = np.asarray(p_stage, dtype=float).copy()
        self.cur_stage_xyz = p_stage_arr
        self.cur_b = float(b)
        self.cur_c = float(c)
        self.stage_min = np.minimum(self.stage_min, p_stage_arr)
        self.stage_max = np.maximum(self.stage_max, p_stage_arr)
        self.b_min_used = min(self.b_min_used, float(b))
        self.b_max_used = max(self.b_max_used, float(b))
        self.c_min_used = min(self.c_min_used, float(c))
        self.c_max_used = max(self.c_max_used, float(c))

    def move_to_tip(
        self,
        p_tip: np.ndarray,
        tangent: Optional[np.ndarray],
        feed: float,
        comment: Optional[str] = None,
        orientation_mode_override: Optional[str] = None,
    ) -> None:
        p_stage, b, c = self.tip_to_stage(
            np.asarray(p_tip, dtype=float),
            tangent=tangent,
            prev_c=self.cur_c,
            orientation_mode_override=orientation_mode_override,
        )
        self.write_move(p_stage, b, c, feed, comment=comment)
        self.cur_tip_xyz = np.asarray(p_tip, dtype=float).copy()
        self.last_tip_tangent = None if tangent is None else np.asarray(tangent, dtype=float).copy()

    def pressure_preload_before_print(self) -> None:
        if self.emit_extrusion and self.pressure_offset_mm > 0.0 and not self.pressure_charged:
            self.pressure_charged = True
            self.f.write("; pressure preload before print pass\n")
            self.f.write(f"G1 {self.cal.u_axis}{self.u_cmd_actual():.3f} F{self.pressure_advance_feed:.0f}\n")
            if self.preflow_dwell_ms > 0:
                self.f.write(f"G4 P{self.preflow_dwell_ms}\n")

    def pressure_release_after_print(self, end_is_node: bool = False) -> None:
        if self.emit_extrusion and self.pressure_offset_mm > 0.0 and self.pressure_charged:
            self.pressure_charged = False
            self.f.write("; pressure release after print pass\n")
            self.f.write(f"G1 {self.cal.u_axis}{self.u_cmd_actual():.3f} F{self.pressure_retract_feed:.0f}\n")

    def approach_start_from_side(
        self,
        start_tip: np.ndarray,
        start_tangent: np.ndarray,
        far_clearance: float,
        near_clearance: float,
        retreat_clearance: float,
        side_lift_z: float,
        label: str,
        kind: str,
        row_num: int,
        lateral_axis: str,
    ) -> None:
        start_tip = np.asarray(start_tip, dtype=float)
        start_tangent = normalize(np.asarray(start_tangent, dtype=float))

        use_row_branch_approach = str(kind).strip().lower() == "branch"
        orientation_mode_override = self.orientation_mode_for_path(kind=kind, row_num=row_num)
        if use_row_branch_approach:
            retreat_axis = row_branch_approach_axis(int(row_num), lateral_axis)
            approach_dir = branch_approach_offset_direction(start_tangent, int(row_num), lateral_axis)
            far_tip = start_tip + approach_dir * float(far_clearance)
            near_tip = start_tip + approach_dir * float(near_clearance)
            angled_rows_2_3_branch = int(row_num) in (2, 3) and abs(float(start_tangent[2])) < 0.999

            if self.cur_tip_xyz is not None:
                retreat_tip = np.asarray(self.cur_tip_xyz, dtype=float) + retreat_axis * float(retreat_clearance) + np.array([0.0, 0.0, -float(side_lift_z)])
                retreat_tangent = self.last_tip_tangent if self.last_tip_tangent is not None else start_tangent
                if int(row_num) in (2, 3):
                    # Rows 2/3 still move in negative Y and negative Z, but keep B at the
                    # same tangent angle that will be used when the branch starts printing.
                    retreat_tangent = start_tangent
                self.move_to_tip(retreat_tip, tangent=retreat_tangent, feed=self.approach_feed, comment=f"{label}: row-aware retreat from previous path (-Y,-Z for rows 2/3, B matches branch start)", orientation_mode_override=orientation_mode_override)

            self.move_to_tip(far_tip, tangent=start_tangent, feed=self.travel_feed, comment=f"{label}: row-aware branch approach far", orientation_mode_override=orientation_mode_override)
            if angled_rows_2_3_branch:
                self.move_to_tip(start_tip, tangent=start_tangent, feed=self.fine_approach_feed, comment=f"{label}: direct fine approach to start for angled row-2/3 branch", orientation_mode_override=orientation_mode_override)
            else:
                self.move_to_tip(near_tip, tangent=start_tangent, feed=self.approach_feed, comment=f"{label}: row-aware branch approach near", orientation_mode_override=orientation_mode_override)
                self.move_to_tip(start_tip, tangent=start_tangent, feed=self.fine_approach_feed, comment=f"{label}: tangential fine approach to start", orientation_mode_override=orientation_mode_override)
            return

        side = side_vector_from_tangent(start_tangent, fallback=self.last_tip_tangent)
        far_tip = start_tip - side * float(far_clearance) + np.array([0.0, 0.0, float(side_lift_z)])
        near_tip = start_tip - side * float(near_clearance)

        if self.cur_tip_xyz is not None:
            retreat_dir = self.last_tip_tangent if self.last_tip_tangent is not None else start_tangent
            cur_side = side_vector_from_tangent(retreat_dir, fallback=side)
            retreat_tip = np.asarray(self.cur_tip_xyz, dtype=float) + cur_side * float(retreat_clearance) + np.array([0.0, 0.0, float(side_lift_z)])
            self.move_to_tip(retreat_tip, tangent=retreat_dir, feed=self.approach_feed, comment=f"{label}: side retreat from previous path", orientation_mode_override=orientation_mode_override)

        self.move_to_tip(far_tip, tangent=start_tangent, feed=self.travel_feed, comment=f"{label}: side approach far", orientation_mode_override=orientation_mode_override)
        if str(kind).strip().lower() == "vertical":
            self.move_to_tip(start_tip, tangent=start_tangent, feed=self.fine_approach_feed, comment=f"{label}: direct fine approach to vertical start", orientation_mode_override=orientation_mode_override)
        else:
            self.move_to_tip(near_tip, tangent=start_tangent, feed=self.approach_feed, comment=f"{label}: side approach near", orientation_mode_override=orientation_mode_override)
            self.move_to_tip(start_tip, tangent=start_tangent, feed=self.fine_approach_feed, comment=f"{label}: fine approach to start", orientation_mode_override=orientation_mode_override)

    def emit_node_write_start(self, label: str, points: np.ndarray, tangents: np.ndarray, extrusion_multiplier: float, kind: str, plane: str, figure: str, end_is_node: bool = False) -> None:
        start_b = desired_physical_b_angle_from_tangent(tangents[0])
        end_b = desired_physical_b_angle_from_tangent(tangents[-1])
        self.f.write(
            "; NODE_WRITE_START "
            f"path_id={label} plane={plane} figure={figure} kind={kind} "
            f"point_count={len(points)} extrusion_multiplier={float(extrusion_multiplier):.6f} "
            f"tip_start_x={float(points[0, 0]):.6f} tip_start_y={float(points[0, 1]):.6f} tip_start_z={float(points[0, 2]):.6f} "
            f"tip_end_x={float(points[-1, 0]):.6f} tip_end_y={float(points[-1, 1]):.6f} tip_end_z={float(points[-1, 2]):.6f} "
            f"physical_b_start_deg={start_b:.6f} physical_b_end_deg={end_b:.6f} "
            f"end_is_node={int(bool(end_is_node))} "
            "tip_angle_convention=0_posZ_90_horizontal_180_negZ\n"
        )

    def emit_node_write_end(self, label: str) -> None:
        self.f.write(f"; NODE_WRITE_END path_id={label}\n")

    def print_polyline(
        self,
        points: np.ndarray,
        tangents: np.ndarray,
        extrusion_multiplier: float,
        label: str,
        kind: str,
        plane: str,
        figure: str,
        row_num: int,
        end_is_node: bool = False,
        junction_dwell_indices: Sequence[int] = (),
    ) -> None:
        if len(points) < 2:
            return

        self.pressure_preload_before_print()
        path_orientation_mode = self.orientation_mode_for_path(kind=kind, row_num=row_num)
        mode_note = "Cartesian centerline" if self.write_mode == "cartesian" else f"calibration-based exact tip tracking ({path_orientation_mode})"
        self.f.write(f"; print {label} ({mode_note})\n")
        self.emit_node_write_start(label, points, tangents, extrusion_multiplier, kind, plane, figure, end_is_node=end_is_node)

        last_tip = np.asarray(points[0], dtype=float).copy()
        self.cur_tip_xyz = last_tip.copy()
        self.last_tip_tangent = np.asarray(tangents[0], dtype=float).copy()
        junction_dwell_set = {int(i) for i in junction_dwell_indices if 0 < int(i) < len(points)}

        for i in range(1, len(points)):
            p0 = np.asarray(points[i - 1], dtype=float)
            p1 = np.asarray(points[i], dtype=float)
            seg_t0 = np.asarray(tangents[i - 1], dtype=float)
            seg_t1 = np.asarray(tangents[i], dtype=float)
            for s in range(1, self.edge_samples + 1):
                t = s / float(self.edge_samples)
                p_tip = p0 + t * (p1 - p0)
                tangent = normalize((1.0 - t) * seg_t0 + t * seg_t1)
                p_stage, b, c = self.tip_to_stage(
                    p_tip,
                    tangent=tangent,
                    prev_c=self.cur_c,
                    orientation_mode_override=path_orientation_mode,
                )

                u_val = None
                if self.emit_extrusion:
                    tip_seg_len = float(np.linalg.norm(p_tip - last_tip))
                    self.u_material_abs += self.extrusion_per_mm * float(extrusion_multiplier) * tip_seg_len
                    u_val = self.u_cmd_actual()

                self.write_move(p_stage, b, c, self.print_feed, comment=None, u_value=u_val)
                self.cur_tip_xyz = p_tip.copy()
                self.last_tip_tangent = tangent.copy()
                last_tip = p_tip.copy()

            if i in junction_dwell_set and self.node_dwell_ms > 0:
                self.f.write("; junction dwell at original trunk attachment point\n")
                self.f.write(f"G4 P{self.node_dwell_ms}\n")

        self.emit_node_write_end(label)
        self.pressure_release_after_print(end_is_node=end_is_node)
        self.total_paths_written += 1


# ---------------- Top-level generation ----------------
def summarize_primitives(paths: Sequence[PrimitivePath]) -> Dict[str, Any]:
    vertical = [p for p in paths if p.kind == "vertical"]
    branch = [p for p in paths if p.kind == "branch"]
    planes = sorted({p.plane for p in paths})
    return {
        "path_count": len(paths),
        "vertical_count": len(vertical),
        "branch_count": len(branch),
        "planes": planes,
    }


def machine_pose_for_tip(
    cal: Calibration,
    p_tip: np.ndarray,
    write_mode: str,
    orientation_mode: str,
    bc_solve_samples: int,
    tangent: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, float]:
    write_mode = str(write_mode).strip().lower()
    orientation_mode = str(orientation_mode).strip().lower()
    if write_mode == "cartesian":
        return (float(p_tip[0]), float(p_tip[1]), float(p_tip[2]), 0.0, 0.0)
    if tangent is None or orientation_mode == "fixed":
        b = solve_b_for_target_tip_angle(cal, 0.0, search_samples=bc_solve_samples)
        c = 90.0
    else:
        c = 90.0
        b = solve_b_for_target_tip_angle(cal, desired_physical_b_angle_from_tangent(tangent), search_samples=bc_solve_samples)
    p_stage = stage_xyz_for_tip(cal, np.asarray(p_tip, dtype=float), float(b), float(c))
    return (float(p_stage[0]), float(p_stage[1]), float(p_stage[2]), float(b), float(c))


def write_node_gcode(
    out: str,
    calibration: Optional[str],
    write_mode: str,
    orientation_mode: str,
    y_offplane_sign: float,
    origin_x: float,
    origin_y: float,
    origin_z: float,
    include_rotated_plane: bool,
    rotated_plane_y_offset: float,
    include_third_row_xz_plane: bool,
    third_row_xz_plane_y_offset: float,
    third_row_branch_start_extension_mm: float,
    rows: Any,
    machine_start_x: float,
    machine_start_y: float,
    machine_start_z: float,
    machine_start_b: float,
    machine_start_c: float,
    machine_end_x: float,
    machine_end_y: float,
    machine_end_z: float,
    machine_end_b: float,
    machine_end_c: float,
    use_explicit_machine_start_end: bool,
    travel_feed: float,
    approach_feed: float,
    fine_approach_feed: float,
    print_feed: float,
    c_feed: float,
    bc_solve_samples: int,
    edge_samples: int,
    travel_lift_z: float,
    side_approach_far: float,
    side_approach_near: float,
    side_retreat: float,
    side_lift_z: float,
    c_max_step_deg: float,
    curve_samples: int,
    arc_samples: int,
    tangent_smooth_window: int,
    centerline_smooth_window: int,
    extrusion_per_mm: float,
    prime_mm: float,
    extrusion_multiplier_vertical: float,
    extrusion_multiplier_branch: float,
    pressure_offset_mm: float,
    pressure_advance_feed: float,
    pressure_retract_feed: float,
    preflow_dwell_ms: int,
    node_dwell_ms: int,
    first_vertical_x: float,
    station_spacing: float,
    main_vertical_height: float,
    node_z: float,
    bbox_x_min: float,
    bbox_x_max: float,
    bbox_y_min: float,
    bbox_y_max: float,
    bbox_z_min: float,
    bbox_z_max: float,
    **_: Any,
) -> None:
    write_mode = str(write_mode).strip().lower()
    orientation_mode = str(orientation_mode).strip().lower()
    if write_mode not in {"calibrated", "cartesian"}:
        raise ValueError("write_mode must be 'calibrated' or 'cartesian'.")
    if orientation_mode not in {"fixed", "tangent"}:
        raise ValueError("orientation_mode must be 'fixed' or 'tangent'.")
    if write_mode == "calibrated" and not calibration:
        raise ValueError("--calibration is required when --write-mode calibrated")

    if write_mode == "calibrated":
        cal = load_calibration(str(calibration))
        cal.offplane_y_sign = float(y_offplane_sign)
    else:
        cal = Calibration(
            pr=np.zeros(1),
            pz=np.zeros(1),
            py_off=None,
            pa=None,
            b_min=0.0,
            b_max=180.0,
            x_axis="X",
            y_axis="Y",
            z_axis="Z",
            b_axis="B",
            c_axis="C",
            u_axis="U",
            c_180_deg=180.0,
        )

    bbox = {
        "x_min": float(bbox_x_min),
        "x_max": float(bbox_x_max),
        "y_min": float(bbox_y_min),
        "y_max": float(bbox_y_max),
        "z_min": float(bbox_z_min),
        "z_max": float(bbox_z_max),
    }

    paths = build_all_patterns(
        origin_x=origin_x,
        origin_y=origin_y,
        origin_z=origin_z,
        include_rotated_plane=bool(include_rotated_plane),
        rotated_plane_y_offset=rotated_plane_y_offset,
        include_third_row_xz_plane=bool(include_third_row_xz_plane),
        third_row_xz_plane_y_offset=third_row_xz_plane_y_offset,
        third_row_branch_start_extension_mm=third_row_branch_start_extension_mm,
        rows=rows,
        first_vertical_x=first_vertical_x,
        station_spacing=station_spacing,
        main_vertical_height=main_vertical_height,
        node_z=node_z,
        curve_samples=curve_samples,
        arc_samples=arc_samples,
    )
    if not paths:
        raise ValueError("No printable paths were generated.")

    summary = summarize_primitives(paths)
    first_points = deduplicate_polyline_points(points_to_array(paths[0].points))
    first_tangents = build_tangents_for_points(first_points, smooth_window=tangent_smooth_window, centerline_smooth_window=centerline_smooth_window)

    machine_start_pose = (
        float(machine_start_x),
        float(machine_start_y),
        float(machine_start_z),
        float(machine_start_b),
        float(machine_start_c),
    )
    machine_end_pose = (
        float(machine_end_x),
        float(machine_end_y),
        float(machine_end_z),
        float(machine_end_b),
        float(machine_end_c),
    )

    out_path = str(out)
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    emit_extrusion = float(extrusion_per_mm) != 0.0

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("; generated by node_calibrated_pattern.py\n")
        f.write("; based on the working node pattern plus vasculature-style exact tip planning\n")
        if write_mode == "calibrated":
            f.write("; calibration-based tip-position planning: stage = tip - offset_tip(B,C)\n")
        else:
            f.write("; Cartesian writing mode: stage XYZ follows the path XYZ directly\n")
        f.write("; verticals are written bottom->top; branches/arcs/curves are written from the trunk-side node outward\n")
        f.write(f"; origin = ({origin_x:.3f}, {origin_y:.3f}, {origin_z:.3f})\n")
        f.write(f"; include_rotated_plane = {bool(include_rotated_plane)}\n")
        f.write(f"; rotated_plane_y_offset = {float(rotated_plane_y_offset):.3f}\n")
        f.write(f"; rows = {rows}\n")
        f.write(f"; include_third_row_xz_plane = {bool(include_third_row_xz_plane)}\n")
        f.write(f"; third_row_xz_plane_y_offset = {float(third_row_xz_plane_y_offset):.3f}\n")
        f.write(f"; third_row_branch_start_extension_mm = {float(third_row_branch_start_extension_mm):.3f}\n")
        f.write(f"; path_count = {summary['path_count']} (vertical={summary['vertical_count']}, branch={summary['branch_count']})\n")
        f.write(f"; planes = {summary['planes']}\n")
        f.write(f"; axes: X->{cal.x_axis}, Y->{cal.y_axis}, Z->{cal.z_axis}, B->{cal.b_axis}, C->{cal.c_axis}, U->{cal.u_axis}\n")
        f.write(f"; write_mode = {write_mode}\n")
        f.write(f"; orientation_mode = {orientation_mode}\n")
        f.write("; calibrated motion holds C at 90 deg during approach and printing\n; calibrated stage-tip kinematics include the off-plane calibration term when write_mode=calibrated\n")
        f.write("; row 1 paths are always fixed-B; row 2 and row 3 paths always solve tangent B from the calibration pull/tip-angle model\n; node junction dwell uses the original trunk-attachment dwell points only; end-of-pass dwell is disabled\n")
        f.write("; startup always uses the configured machine-start coordinates; shutdown always uses the configured machine-end coordinates\n")
        f.write(f"; selected_fit_model = {cal.selected_fit_model or 'legacy-polynomial'}\n")
        f.write(f"; selected_offplane_fit_model = {cal.selected_offplane_fit_model or cal.selected_fit_model or 'legacy-polynomial'}\n")
        f.write(f"; active_phase = {cal.active_phase}\n")
        f.write(f"; y_offplane_sign = {float(cal.offplane_y_sign):.1f}\n")
        f.write(f"; {describe_model(cal.r_model, 'r')}\n")
        f.write(f"; {describe_model(cal.z_model, 'z')}\n")
        f.write(f"; {describe_model(cal.y_off_model, 'offplane_y')}\n")
        f.write(f"; {describe_model(cal.tip_angle_model, 'tip_angle')}\n")
        f.write(f"; travel_lift_z argument retained for compatibility = {float(travel_lift_z):.3f}\n")
        for warning in calibration_model_range_warnings(cal):
            f.write(f"; WARNING: {warning}\n")
        f.write(f"; feeds: travel={travel_feed:.1f}, approach={approach_feed:.1f}, fine_approach={fine_approach_feed:.1f}, print={print_feed:.1f}, C-only={c_feed:.1f}\n")
        f.write(f"; side approach: far={side_approach_far:.3f}, near={side_approach_near:.3f}, retreat={side_retreat:.3f}, lift_z={side_lift_z:.3f}\n")
        f.write("G90\n")
        if emit_extrusion:
            f.write("M82\n")
            f.write(f"G92 {cal.u_axis}0\n")
            if abs(float(prime_mm)) > 0.0:
                f.write(f"G1 {cal.u_axis}{float(prime_mm):.3f} F{max(60.0, float(pressure_advance_feed)):.0f} ; prime material\n")

        g = GCodeWriter(
            fh=f,
            cal=cal,
            bbox=bbox,
            travel_feed=travel_feed,
            approach_feed=approach_feed,
            fine_approach_feed=fine_approach_feed,
            print_feed=print_feed,
            c_feed=c_feed,
            edge_samples=edge_samples,
            extrusion_per_mm=extrusion_per_mm,
            pressure_offset_mm=pressure_offset_mm,
            pressure_advance_feed=pressure_advance_feed,
            pressure_retract_feed=pressure_retract_feed,
            preflow_dwell_ms=preflow_dwell_ms,
            node_dwell_ms=node_dwell_ms,
            emit_extrusion=emit_extrusion,
            write_mode=write_mode,
            orientation_mode=orientation_mode,
            bc_solve_samples=bc_solve_samples,
            c_max_step_deg=c_max_step_deg,
        )

        msx, msy, msz, msb, msc = [float(v) for v in machine_start_pose]
        mex, mey, mez, meb, mec = [float(v) for v in machine_end_pose]
        g.write_move(np.array([msx, msy, msz], dtype=float), msb, msc, travel_feed, comment="startup: move to anchored machine start pose")

        for path in paths:
            points = deduplicate_polyline_points(points_to_array(path.points))
            if len(points) < 2:
                continue
            tangents = build_tangents_for_points(points, smooth_window=tangent_smooth_window, centerline_smooth_window=centerline_smooth_window)
            mult = extrusion_multiplier_vertical if path.kind == "vertical" else extrusion_multiplier_branch

            f.write("; ------------------------------------------------------------\n")
            f.write(f"; {path.label}: plane={path.plane}, figure={path.figure}, kind={path.kind}, points={len(points)}\n")
            g.approach_start_from_side(
                start_tip=points[0],
                start_tangent=tangents[0],
                far_clearance=side_approach_far,
                near_clearance=side_approach_near,
                retreat_clearance=side_retreat,
                side_lift_z=side_lift_z,
                label=path.label,
                kind=path.kind,
                row_num=path.row_num,
                lateral_axis=path.lateral_axis,
            )
            g.print_polyline(
                points,
                tangents,
                extrusion_multiplier=mult,
                label=path.label,
                kind=path.kind,
                plane=path.plane,
                figure=path.figure,
                row_num=path.row_num,
                end_is_node=path.end_is_node,
                junction_dwell_indices=path.junction_dwell_indices,
            )

        g.write_move(np.array([mex, mey, mez], dtype=float), meb, mec, travel_feed, comment="shutdown: move to anchored machine end pose")
        if emit_extrusion:
            f.write(f"G92 {cal.u_axis}0\n")
        f.write(f"; total_paths_written = {g.total_paths_written}\n")

        if g.warnings:
            f.write("; ==================== warnings ====================\n")
            for w in g.warnings:
                f.write(f"; {w}\n")

    return {
        "out": out_path,
        "write_mode": write_mode,
        "active_phase": cal.active_phase,
        "selected_fit_model": cal.selected_fit_model or "legacy-polynomial",
        "selected_offplane_fit_model": cal.selected_offplane_fit_model or cal.selected_fit_model or "legacy-polynomial",
        "y_offplane_sign": float(cal.offplane_y_sign),
        "ranges": {
            "stage_x": (float(g.stage_min[0]), float(g.stage_max[0])),
            "stage_y": (float(g.stage_min[1]), float(g.stage_max[1])),
            "stage_z": (float(g.stage_min[2]), float(g.stage_max[2])),
            "b": (float(g.b_min_used), float(g.b_max_used)),
            "c": (float(g.c_min_used), float(g.c_max_used)),
        },
        "model_descriptions": [
            describe_model(cal.r_model, "r"),
            describe_model(cal.z_model, "z"),
            describe_model(cal.y_off_model, "offplane_y"),
            describe_model(cal.tip_angle_model, "tip_angle"),
        ],
        "warnings": list(g.warnings),
    }


# ---------------- CLI ----------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Generate calibrated or Cartesian G-code for the planar node figures, including a second -90-degree rotated duplicate row and an added third-row negative-Y rotated duplicate, "
            "with exact tip-position planning adapted from the vasculature writer. Calibrated motion holds C at 90 degrees."
        )
    )
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code file.")
    ap.add_argument("--calibration", default=None, help="Calibration JSON. Required for --write-mode calibrated.")
    ap.add_argument("--write-mode", choices=["calibrated", "cartesian"], default="calibrated")
    ap.add_argument("--orientation-mode", choices=["fixed", "tangent"], default="fixed", help="fixed = use pull-axis calibration with constant tip angle; tangent = solve B from the calibration tip-angle model so vertical up is 0 deg and horizontal is 90 deg, while holding C at 90 degrees.")
    ap.add_argument("--y-offplane-sign", type=float, default=1.0, help="Multiplier applied to the calibration off-plane Y term during calibrated kinematics. Use -1 to flip the sign.")

    ap.add_argument("--origin-x", type=float, default=DEFAULT_ORIGIN_X)
    ap.add_argument("--origin-y", type=float, default=DEFAULT_ORIGIN_Y)
    ap.add_argument("--origin-z", type=float, default=DEFAULT_ORIGIN_Z)

    ap.add_argument("--include-rotated-plane", dest="include_rotated_plane", action="store_true", default=DEFAULT_INCLUDE_ROTATED_PLANE)
    ap.add_argument("--no-include-rotated-plane", dest="include_rotated_plane", action="store_false")
    ap.add_argument("--rotated-plane-y-offset", type=float, default=DEFAULT_ROTATED_PLANE_Y_OFFSET, help="Y offset applied to the second figure set before rotating each figure -90 degrees around its own trunk.")
    ap.add_argument("--include-third-row-xz-plane", dest="include_third_row_xz_plane", action="store_true", default=DEFAULT_INCLUDE_THIRD_ROW_XZ_PLANE)
    ap.add_argument("--no-include-third-row-xz-plane", dest="include_third_row_xz_plane", action="store_false")
    ap.add_argument("--third-row-xz-plane-y-offset", type=float, default=DEFAULT_THIRD_ROW_XZ_PLANE_Y_OFFSET, help="Absolute row-3 Y offset relative to origin_y for the added third-row XZ-plane figure set.")
    ap.add_argument("--third-row-branch-start-extension-mm", type=float, default=DEFAULT_THIRD_ROW_BRANCH_START_EXTENSION_MM, help="For the added third-row negative-Y rotated row, extend each branch-like path backward along its local starting tangent by this amount so it crosses the trunk, then dwell at the original junction.")
    ap.add_argument("--rows", default=DEFAULT_ROWS, help="Rows to emit: all (default), 1, 2, 3, or comma-separated subsets such as 1,3. When a subset is given, it overrides the legacy include-row booleans.")
    # Backward-compatible aliases from the previous revision.
    ap.add_argument("--include-secondary-xz-plane", dest="include_third_row_xz_plane", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--no-include-secondary-xz-plane", dest="include_third_row_xz_plane", action="store_false", help=argparse.SUPPRESS)
    ap.add_argument("--secondary-xz-plane-y-offset", dest="third_row_xz_plane_y_offset", type=float, help=argparse.SUPPRESS)
    ap.add_argument("--secondary-xz-branch-start-extension-mm", dest="third_row_branch_start_extension_mm", type=float, help=argparse.SUPPRESS)

    ap.add_argument("--machine-start-x", type=float, default=DEFAULT_MACHINE_START_X)
    ap.add_argument("--machine-start-y", type=float, default=DEFAULT_MACHINE_START_Y)
    ap.add_argument("--machine-start-z", type=float, default=DEFAULT_MACHINE_START_Z)
    ap.add_argument("--machine-start-b", type=float, default=DEFAULT_MACHINE_START_B)
    ap.add_argument("--machine-start-c", type=float, default=DEFAULT_MACHINE_START_C)
    ap.add_argument("--machine-end-x", type=float, default=DEFAULT_MACHINE_END_X)
    ap.add_argument("--machine-end-y", type=float, default=DEFAULT_MACHINE_END_Y)
    ap.add_argument("--machine-end-z", type=float, default=DEFAULT_MACHINE_END_Z)
    ap.add_argument("--machine-end-b", type=float, default=DEFAULT_MACHINE_END_B)
    ap.add_argument("--machine-end-c", type=float, default=DEFAULT_MACHINE_END_C)
    ap.add_argument("--use-explicit-machine-start-end", action="store_true", help="Deprecated compatibility flag. Explicit machine start/end poses are always used.")

    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED)
    ap.add_argument("--approach-feed", type=float, default=DEFAULT_APPROACH_FEED)
    ap.add_argument("--fine-approach-feed", type=float, default=DEFAULT_FINE_APPROACH_FEED)
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED)
    ap.add_argument("--c-feed", type=float, default=DEFAULT_C_FEED)
    ap.add_argument("--bc-solve-samples", type=int, default=DEFAULT_BC_SOLVE_SAMPLES)
    ap.add_argument("--edge-samples", type=int, default=DEFAULT_EDGE_SAMPLES, help="Linear subdivisions per path segment during printing.")
    ap.add_argument("--travel-lift-z", type=float, default=DEFAULT_TRAVEL_LIFT_Z, help="Retained for CLI compatibility; side approach travel is used in calibrated mode.")
    ap.add_argument("--side-approach-far", type=float, default=DEFAULT_SIDE_APPROACH_FAR)
    ap.add_argument("--side-approach-near", type=float, default=DEFAULT_SIDE_APPROACH_NEAR)
    ap.add_argument("--side-retreat", type=float, default=DEFAULT_SIDE_RETREAT)
    ap.add_argument("--side-lift-z", type=float, default=DEFAULT_SIDE_LIFT_Z)
    ap.add_argument("--c-max-step-deg", type=float, default=DEFAULT_C_MAX_STEP_DEG, help="Maximum allowed C change per emitted move in tangent mode. Use 0 to disable limiting.")

    ap.add_argument("--curve-samples", type=int, default=DEFAULT_CURVE_SAMPLES)
    ap.add_argument("--arc-samples", type=int, default=DEFAULT_ARC_SAMPLES)
    ap.add_argument("--tangent-smooth-window", type=int, default=DEFAULT_TANGENT_SMOOTH_WINDOW)
    ap.add_argument("--centerline-smooth-window", type=int, default=DEFAULT_CENTERLINE_SMOOTH_WINDOW)

    ap.add_argument("--extrusion-per-mm", type=float, default=DEFAULT_EXTRUSION_PER_MM)
    ap.add_argument("--prime-mm", type=float, default=DEFAULT_PRIME_MM)
    ap.add_argument("--extrusion-multiplier-vertical", type=float, default=DEFAULT_EXTRUSION_MULTIPLIER_VERTICAL)
    ap.add_argument("--extrusion-multiplier-branch", type=float, default=DEFAULT_EXTRUSION_MULTIPLIER_BRANCH)
    ap.add_argument("--pressure-offset-mm", type=float, default=DEFAULT_PRESSURE_OFFSET_MM)
    ap.add_argument("--pressure-advance-feed", type=float, default=DEFAULT_PRESSURE_ADVANCE_FEED)
    ap.add_argument("--pressure-retract-feed", type=float, default=DEFAULT_PRESSURE_RETRACT_FEED)
    ap.add_argument("--preflow-dwell-ms", type=int, default=DEFAULT_PREFLOW_DWELL_MS)
    ap.add_argument("--node-dwell-ms", type=int, default=DEFAULT_NODE_DWELL_MS)

    ap.add_argument("--first-vertical-x", type=float, default=DEFAULT_FIRST_VERTICAL_X)
    ap.add_argument("--station-spacing", type=float, default=DEFAULT_STATION_SPACING)
    ap.add_argument("--main-vertical-height", type=float, default=DEFAULT_MAIN_VERTICAL_HEIGHT)
    ap.add_argument("--node-z", type=float, default=DEFAULT_NODE_Z)

    ap.add_argument("--bbox-x-min", type=float, default=DEFAULT_BBOX_X_MIN)
    ap.add_argument("--bbox-x-max", type=float, default=DEFAULT_BBOX_X_MAX)
    ap.add_argument("--bbox-y-min", type=float, default=DEFAULT_BBOX_Y_MIN)
    ap.add_argument("--bbox-y-max", type=float, default=DEFAULT_BBOX_Y_MAX)
    ap.add_argument("--bbox-z-min", type=float, default=DEFAULT_BBOX_Z_MIN)
    ap.add_argument("--bbox-z-max", type=float, default=DEFAULT_BBOX_Z_MAX)
    return ap


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    result = write_node_gcode(**vars(args))
    print(f"Wrote {result['out']}")
    print("Gantry ranges used:")
    print(f"  { 'X':>1}: {result['ranges']['stage_x'][0]:.3f} to {result['ranges']['stage_x'][1]:.3f}")
    print(f"  { 'Y':>1}: {result['ranges']['stage_y'][0]:.3f} to {result['ranges']['stage_y'][1]:.3f}")
    print(f"  { 'Z':>1}: {result['ranges']['stage_z'][0]:.3f} to {result['ranges']['stage_z'][1]:.3f}")
    print(f"  { 'B':>1}: {result['ranges']['b'][0]:.3f} to {result['ranges']['b'][1]:.3f}")
    print(f"  { 'C':>1}: {result['ranges']['c'][0]:.3f} to {result['ranges']['c'][1]:.3f}")
    print("Models used:")
    print(f"  active_phase = {result['active_phase']}")
    print(f"  selected_fit_model = {result['selected_fit_model']}")
    print(f"  selected_offplane_fit_model = {result['selected_offplane_fit_model']}")
    print(f"  y_offplane_sign = {result['y_offplane_sign']:.1f}")
    for line in result['model_descriptions']:
        print(f"  {line}")
    if result['warnings']:
        print("Warnings:")
        for warning in result['warnings']:
            print(f"  {warning}")


if __name__ == "__main__":
    main()
