#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code for a single Gaussian-modulated sine wave in
the XZ plane.

Highlights
----------
- Desired tip path lies in the XZ plane at constant Y.
- Default tip-space box matches the request:
    X in [50, 150]
    Z in [-170, -120]
    Y = 52
- Tangent-following orientation:
    * B = 0 deg   -> tip points straight up (+Z)
    * B = 90 deg  -> tip is horizontal
    * B = 180 deg -> tip points straight down (-Z)
- Default C is held at 180 deg so the tool stays in the XZ plane.
- Supports both:
    1) calibrated tip tracking, using
         stage_xyz = desired_tip_xyz - offset_tip(B, C)
    2) direct Cartesian stage motion

The calibration loading / tip-offset planning logic follows the same structure
as the provided reference script: it keeps the same pull-axis calibration model
selection, `predict_tip_offset_xyz(...)`, `stage_xyz_for_tip(...)`,
`solve_b_for_target_tip_angle(...)`, and the same B-angle convention
(0=up, 90=horizontal, 180=down).
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
DEFAULT_OUT = "gcode_generation/gaussian_wave_xz.gcode"

# Path placement
DEFAULT_X_START = 50.0
DEFAULT_X_END = 110.0
DEFAULT_Y = 52.0
DEFAULT_Z_MIN = -155.0
DEFAULT_Z_MAX = -115.0
DEFAULT_Z_BASELINE = 0.5 * (DEFAULT_Z_MIN + DEFAULT_Z_MAX)

# Wave shape
DEFAULT_Z_AMPLITUDE = 18.0
DEFAULT_CYCLES = 2.5
DEFAULT_PHASE_DEG = 0.0
DEFAULT_X_CENTER = 0.5 * (DEFAULT_X_START + DEFAULT_X_END)
DEFAULT_GAUSSIAN_SIGMA = 18.0
DEFAULT_WINDOW_POWER = 1.0
DEFAULT_LEAD_IN = 4.0
DEFAULT_LEAD_OUT = 4.0
DEFAULT_POINTS = 501
DEFAULT_TANGENT_SMOOTH_WINDOW = 4
DEFAULT_CENTERLINE_SMOOTH_WINDOW = 0
DEFAULT_MIN_TANGENT_XY = 1e-9
DEFAULT_POINT_MERGE_TOL = 1e-9
DEFAULT_EXTREMA_Z_EPS = 1e-9

# Orientation
DEFAULT_WRITE_MODE = "calibrated"
DEFAULT_ORIENTATION_MODE = "tangent"
DEFAULT_FIXED_B = 90.0
DEFAULT_FIXED_C = 180.0
DEFAULT_C_DEG = 180.0
DEFAULT_B_ANGLE_BIAS_DEG = 0.0
DEFAULT_BC_SOLVE_SAMPLES = 5001
DEFAULT_Y_OFFPLANE_SIGN = -1.0
DEFAULT_Y_OFFPLANE_FIT_MODEL = "avg_cubic"

# Motion
DEFAULT_TRAVEL_FEED = 1000.0
DEFAULT_APPROACH_FEED = 400.0
DEFAULT_FINE_APPROACH_FEED = 100.0
DEFAULT_PRINT_FEED = 300.0
DEFAULT_TRAVEL_LIFT_Z = 8.0
DEFAULT_APPROACH_SIDE_MM = 4.0
DEFAULT_EDGE_SAMPLES = 1

# Extrusion / pressure
DEFAULT_EMIT_EXTRUSION = True
DEFAULT_EXTRUSION_PER_MM = 0.0015
DEFAULT_PRESSURE_OFFSET_MM = 4.0
DEFAULT_PRESSURE_ADVANCE_FEED = 120.0
DEFAULT_PRESSURE_RETRACT_FEED = 240.0
DEFAULT_PREFLOW_DWELL_MS = 500

# Stage limits
DEFAULT_BBOX_X_MIN = -1e9
DEFAULT_BBOX_X_MAX = 1e9
DEFAULT_BBOX_Y_MIN = -1e9
DEFAULT_BBOX_Y_MAX = 1e9
DEFAULT_BBOX_Z_MIN = -1e9
DEFAULT_BBOX_Z_MAX = 1e9


# ---------------- Data classes ----------------
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
    requested_r_fit_model: Optional[str] = None
    requested_z_fit_model: Optional[str] = None
    requested_offplane_fit_model: Optional[str] = None
    resolved_r_fit_model: Optional[str] = None
    resolved_z_fit_model: Optional[str] = None
    resolved_offplane_fit_model: Optional[str] = None
    active_phase: str = "pull"
    offplane_y_sign: float = 1.0


# ---------------- Math / geometry helpers ----------------
def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if n <= eps:
        return np.zeros_like(arr)
    return arr / n


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
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return normalize(points[i1] - points[i0])


def build_tangents_for_points(
    points: np.ndarray,
    smooth_window: int,
    centerline_smooth_window: int = DEFAULT_CENTERLINE_SMOOTH_WINDOW,
) -> np.ndarray:
    tangent_points = smooth_centerline_points(points, window=centerline_smooth_window)
    tangents = np.zeros_like(tangent_points)
    for i in range(len(tangent_points)):
        tangents[i] = tangent_for_index(tangent_points, i, smooth_window=max(1, int(smooth_window)))
    return tangents


def find_extrema_feature_indices(points: np.ndarray, z_eps: float = DEFAULT_EXTREMA_Z_EPS) -> List[int]:
    pts = np.asarray(points, dtype=float)
    n = len(pts)
    if n == 0:
        return []
    if n == 1:
        return [0]

    z = pts[:, 2]
    eps = float(z_eps)
    features: List[int] = [0]
    for i in range(1, n - 1):
        dz_prev = float(z[i] - z[i - 1])
        dz_next = float(z[i + 1] - z[i])
        is_peak = dz_prev > eps and dz_next < -eps
        is_trough = dz_prev < -eps and dz_next > eps
        if is_peak or is_trough:
            features.append(i)
    if features[-1] != n - 1:
        features.append(n - 1)
    return features


def desired_physical_b_angle_from_tangent(tangent: np.ndarray) -> float:
    """
    B-angle convention:
      0 deg   -> +Z (straight up)
      90 deg  -> horizontal
      180 deg -> -Z (straight down)
    """
    t = normalize(np.asarray(tangent, dtype=float))
    tz = float(np.clip(t[2], -1.0, 1.0))
    return float(math.degrees(math.acos(tz)))


def unwrap_angle_deg_near(target_deg: float, reference_deg: float) -> float:
    target = float(target_deg)
    ref = float(reference_deg)
    while target - ref > 180.0:
        target -= 360.0
    while target - ref < -180.0:
        target += 360.0
    return float(target)


def side_vector_from_tangent(
    tangent: np.ndarray,
    fallback: Optional[np.ndarray] = None,
    min_xy: float = DEFAULT_MIN_TANGENT_XY,
) -> np.ndarray:
    xy = np.asarray(tangent[:2], dtype=float)
    nxy = float(np.linalg.norm(xy))
    if nxy < float(min_xy):
        if fallback is not None and float(np.linalg.norm(np.asarray(fallback[:2], dtype=float))) >= float(min_xy):
            xy = np.asarray(fallback[:2], dtype=float)
            nxy = float(np.linalg.norm(xy))
        else:
            return np.array([0.0, 1.0, 0.0], dtype=float)
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


def load_calibration(
    json_path: str,
    requested_r_fit_model: Optional[str] = None,
    requested_z_fit_model: Optional[str] = None,
    requested_offplane_fit_model: Optional[str] = DEFAULT_Y_OFFPLANE_FIT_MODEL,
) -> Calibration:
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
        py_off = _coeffs_from_model(
            fit_models,
            "offplane_y_cubic",
            "offplane_y_avg_cubic",
            "offplane_y",
            "offplane_y_linear",
            "offplane_y_avg_linear",
        )
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
    requested_r_fit_model = None if requested_r_fit_model is None else str(requested_r_fit_model).strip().lower()
    requested_z_fit_model = None if requested_z_fit_model is None else str(requested_z_fit_model).strip().lower()
    active_phase = str(data.get("default_phase_for_legacy_access") or "pull").strip().lower()

    phase_models = data.get("fit_models_by_phase", {}) or {}
    active_phase_models = phase_models.get(active_phase) if isinstance(phase_models, dict) else None
    if not isinstance(active_phase_models, dict):
        active_phase_models = fit_models

    r_selector = requested_r_fit_model or selected_fit_model
    z_selector = requested_z_fit_model or selected_fit_model
    r_model = _select_named_model(active_phase_models, "r", r_selector)
    z_model = _select_named_model(active_phase_models, "z", z_selector)
    requested_offplane_fit_model = (
        None if requested_offplane_fit_model is None else str(requested_offplane_fit_model).strip().lower()
    )
    resolved_r_fit_model = r_selector if r_model is not None else (selected_fit_model if selected_fit_model else None)
    resolved_z_fit_model = z_selector if z_model is not None else (selected_fit_model if selected_fit_model else None)
    y_off_selector = requested_offplane_fit_model or selected_offplane_fit_model or selected_fit_model
    y_off_model = _select_named_model(active_phase_models, "offplane_y", y_off_selector)
    resolved_offplane_fit_model = None
    if y_off_model is not None:
        resolved_offplane_fit_model = y_off_selector
    elif selected_offplane_fit_model:
        resolved_offplane_fit_model = selected_offplane_fit_model
    elif selected_fit_model:
        resolved_offplane_fit_model = selected_fit_model
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
        requested_r_fit_model=requested_r_fit_model,
        requested_z_fit_model=requested_z_fit_model,
        requested_offplane_fit_model=requested_offplane_fit_model,
        resolved_r_fit_model=resolved_r_fit_model,
        resolved_z_fit_model=resolved_z_fit_model,
        resolved_offplane_fit_model=resolved_offplane_fit_model,
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


# ---------------- Wave construction ----------------
def smoothstep01(s: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    return s * s * (3.0 - 2.0 * s)


def build_gaussian_sine_wave_points(
    x_start: float,
    x_end: float,
    y: float,
    z_baseline: float,
    z_amplitude: float,
    cycles: float,
    phase_deg: float,
    x_center: float,
    gaussian_sigma: float,
    no_gaussian: bool,
    lead_in: float,
    lead_out: float,
    points: int,
    window_power: float,
) -> np.ndarray:
    x0 = float(x_start)
    x1 = float(x_end)
    if not x1 > x0:
        raise ValueError("--x-end must be greater than --x-start")
    if float(lead_in) < 0.0 or float(lead_out) < 0.0:
        raise ValueError("--lead-in and --lead-out must be >= 0")
    if float(lead_in + lead_out) >= (x1 - x0):
        raise ValueError("--lead-in + --lead-out must be smaller than the total X span")
    if (not bool(no_gaussian)) and float(gaussian_sigma) <= 0.0:
        raise ValueError("--gaussian-sigma must be > 0 unless --no-gaussian is used")
    if int(points) < 5:
        raise ValueError("--points must be >= 5")

    wave_x0 = x0 + float(lead_in)
    wave_x1 = x1 - float(lead_out)

    n_total = int(points)
    n_lead_in = max(2, int(round(n_total * (lead_in / (x1 - x0))))) if lead_in > 0.0 else 0
    n_lead_out = max(2, int(round(n_total * (lead_out / (x1 - x0))))) if lead_out > 0.0 else 0
    n_wave = max(5, n_total - n_lead_in - n_lead_out)

    parts: List[np.ndarray] = []

    if n_lead_in > 0:
        xs = np.linspace(x0, wave_x0, n_lead_in, endpoint=False)
        zs = np.full_like(xs, float(z_baseline), dtype=float)
        ys = np.full_like(xs, float(y), dtype=float)
        parts.append(np.column_stack([xs, ys, zs]))

    xs = np.linspace(wave_x0, wave_x1, n_wave, endpoint=(n_lead_out == 0))
    s = np.zeros_like(xs) if wave_x1 == wave_x0 else (xs - wave_x0) / (wave_x1 - wave_x0)
    phase = math.radians(float(phase_deg))
    sinusoid = np.sin(2.0 * math.pi * float(cycles) * s + phase)
    if bool(no_gaussian):
        gaussian = np.ones_like(xs, dtype=float)
    else:
        gaussian = np.exp(-0.5 * ((xs - float(x_center)) / float(gaussian_sigma)) ** 2)

    # Endpoint taper makes the wave smoothly leave and rejoin the horizontal baseline
    # while still reaching full strength near the middle of the write.
    taper = np.sin(np.pi * np.clip(s, 0.0, 1.0)) ** 2
    if float(window_power) != 1.0:
        taper = np.power(np.clip(taper, 0.0, 1.0), float(window_power))

    zs = float(z_baseline) + float(z_amplitude) * gaussian * taper * sinusoid
    ys = np.full_like(xs, float(y), dtype=float)
    parts.append(np.column_stack([xs, ys, zs]))

    if n_lead_out > 0:
        xs = np.linspace(wave_x1, x1, n_lead_out)
        zs = np.full_like(xs, float(z_baseline), dtype=float)
        ys = np.full_like(xs, float(y), dtype=float)
        parts.append(np.column_stack([xs, ys, zs]))

    out = np.vstack(parts)
    return deduplicate_polyline_points(out)


# ---------------- G-code helpers ----------------
def _fmt_axes_move(axes_vals: Sequence[Tuple[str, float]]) -> str:
    return " ".join(f"{ax}{val:.3f}" for ax, val in axes_vals)


class GCodeWriter:
    def __init__(
        self,
        fh,
        cal: Optional[Calibration],
        write_mode: str,
        orientation_mode: str,
        fixed_b: float,
        fixed_c: float,
        c_deg: float,
        b_angle_bias_deg: float,
        bc_solve_samples: int,
        bbox: Dict[str, float],
        travel_feed: float,
        approach_feed: float,
        fine_approach_feed: float,
        print_feed: float,
        edge_samples: int,
        emit_extrusion: bool,
        extrusion_per_mm: float,
        pressure_offset_mm: float,
        pressure_advance_feed: float,
        pressure_retract_feed: float,
        preflow_dwell_ms: int,
    ) -> None:
        self.f = fh
        self.cal = cal
        self.write_mode = str(write_mode).strip().lower()
        self.orientation_mode = str(orientation_mode).strip().lower()
        self.fixed_b = float(fixed_b)
        self.fixed_c = float(fixed_c)
        self.c_deg = float(c_deg)
        self.b_angle_bias_deg = float(b_angle_bias_deg)
        self.bc_solve_samples = int(bc_solve_samples)
        self.bbox = dict(bbox)
        self.travel_feed = float(travel_feed)
        self.approach_feed = float(approach_feed)
        self.fine_approach_feed = float(fine_approach_feed)
        self.print_feed = float(print_feed)
        self.edge_samples = max(1, int(edge_samples))
        self.emit_extrusion = bool(emit_extrusion)
        self.extrusion_per_mm = float(extrusion_per_mm)
        self.pressure_offset_mm = float(pressure_offset_mm)
        self.pressure_advance_feed = float(pressure_advance_feed)
        self.pressure_retract_feed = float(pressure_retract_feed)
        self.preflow_dwell_ms = int(preflow_dwell_ms)

        self.u_material_abs = 0.0
        self.pressure_charged = False
        self.cur_stage_xyz: Optional[np.ndarray] = None
        self.cur_tip_xyz: Optional[np.ndarray] = None
        self.cur_b: float = 0.0
        self.cur_c: float = self.c_deg
        self.last_tip_tangent: Optional[np.ndarray] = None
        self.warnings: List[str] = []

        self.stage_min = np.array([np.inf, np.inf, np.inf], dtype=float)
        self.stage_max = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
        self.b_min_used = float("inf")
        self.b_max_used = float("-inf")
        self.c_min_used = float("inf")
        self.c_max_used = float("-inf")

        if self.write_mode == "calibrated" and self.cal is None:
            raise ValueError("Calibration is required for calibrated mode.")

        if self.write_mode == "calibrated" and self.orientation_mode == "fixed" and self.fixed_b is None:
            self.fixed_b = solve_b_for_target_tip_angle(self.cal, 90.0, search_samples=self.bc_solve_samples)

    @property
    def x_axis(self) -> str:
        return self.cal.x_axis if self.cal is not None else "X"

    @property
    def y_axis(self) -> str:
        return self.cal.y_axis if self.cal is not None else "Y"

    @property
    def z_axis(self) -> str:
        return self.cal.z_axis if self.cal is not None else "Z"

    @property
    def b_axis(self) -> str:
        return self.cal.b_axis if self.cal is not None else "B"

    @property
    def c_axis(self) -> str:
        return self.cal.c_axis if self.cal is not None else "C"

    @property
    def u_axis(self) -> str:
        return self.cal.u_axis if self.cal is not None else "U"

    def clamp_stage(self, p_stage: np.ndarray, context: str) -> np.ndarray:
        x = float(np.clip(p_stage[0], self.bbox["x_min"], self.bbox["x_max"]))
        y = float(np.clip(p_stage[1], self.bbox["y_min"], self.bbox["y_max"]))
        z = float(np.clip(p_stage[2], self.bbox["z_min"], self.bbox["z_max"]))
        if abs(x - float(p_stage[0])) > 1e-12 or abs(y - float(p_stage[1])) > 1e-12 or abs(z - float(p_stage[2])) > 1e-12:
            self.warnings.append(f"WARNING: {context} stage point clamped to bbox.")
        return np.array([x, y, z], dtype=float)

    def u_cmd_actual(self) -> float:
        return self.u_material_abs + (self.pressure_offset_mm if self.pressure_charged else 0.0)

    def bc_for_tangent(self, tangent: np.ndarray) -> Tuple[float, float]:
        if self.orientation_mode == "fixed":
            return float(self.fixed_b), float(self.fixed_c)

        target_b = desired_physical_b_angle_from_tangent(tangent) + float(self.b_angle_bias_deg)
        target_b = float(np.clip(target_b, 0.0, 180.0))

        if self.write_mode == "calibrated":
            assert self.cal is not None
            b = solve_b_for_target_tip_angle(self.cal, target_b, search_samples=self.bc_solve_samples)
        else:
            b = target_b

        return float(b), float(self.c_deg)

    def tip_to_stage_with_bc(self, p_tip: np.ndarray, b: float, c: float) -> np.ndarray:
        if self.write_mode == "calibrated":
            assert self.cal is not None
            p_stage = stage_xyz_for_tip(self.cal, np.asarray(p_tip, dtype=float), float(b), float(c))
        else:
            p_stage = np.asarray(p_tip, dtype=float)
        return self.clamp_stage(np.asarray(p_stage, dtype=float), "tip_to_stage_with_bc")

    def tip_to_stage(self, p_tip: np.ndarray, tangent: Optional[np.ndarray]) -> Tuple[np.ndarray, float, float]:
        tangent_arr = np.array([1.0, 0.0, 0.0], dtype=float) if tangent is None else normalize(np.asarray(tangent, dtype=float))
        b, c = self.bc_for_tangent(tangent_arr)
        return self.tip_to_stage_with_bc(np.asarray(p_tip, dtype=float), b, c), float(b), float(c)

    def write_move(
        self,
        p_stage: np.ndarray,
        b: float,
        c: float,
        feed: float,
        comment: Optional[str] = None,
        u_value: Optional[float] = None,
    ) -> None:
        if comment:
            self.f.write(f"; {comment}\n")
        axes: List[Tuple[str, float]] = [
            (self.x_axis, float(p_stage[0])),
            (self.y_axis, float(p_stage[1])),
            (self.z_axis, float(p_stage[2])),
            (self.b_axis, float(b)),
            (self.c_axis, float(c)),
        ]
        if u_value is not None:
            axes.append((self.u_axis, float(u_value)))
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

    def move_to_tip(self, p_tip: np.ndarray, tangent: Optional[np.ndarray], feed: float, comment: Optional[str] = None) -> None:
        p_stage, b, c = self.tip_to_stage(np.asarray(p_tip, dtype=float), tangent=tangent)
        self.write_move(p_stage, b, c, feed, comment=comment)
        self.cur_tip_xyz = np.asarray(p_tip, dtype=float).copy()
        self.last_tip_tangent = None if tangent is None else np.asarray(tangent, dtype=float).copy()

    def move_to_tip_with_bc(
        self,
        p_tip: np.ndarray,
        b: float,
        c: float,
        feed: float,
        tangent: Optional[np.ndarray],
        comment: Optional[str] = None,
    ) -> None:
        p_stage = self.tip_to_stage_with_bc(np.asarray(p_tip, dtype=float), float(b), float(c))
        self.write_move(p_stage, float(b), float(c), feed, comment=comment)
        self.cur_tip_xyz = np.asarray(p_tip, dtype=float).copy()
        self.last_tip_tangent = None if tangent is None else np.asarray(tangent, dtype=float).copy()

    def pressure_preload_before_print(self) -> None:
        if self.emit_extrusion and not self.pressure_charged:
            self.pressure_charged = True
            self.f.write("; open pressure solenoid before print pass\n")
            self.f.write("M42 P0 S1\n")
            if self.preflow_dwell_ms > 0:
                self.f.write(f"G4 P{self.preflow_dwell_ms}\n")

    def pressure_release_after_print(self) -> None:
        if self.emit_extrusion and self.pressure_charged:
            self.pressure_charged = False
            self.f.write("; close pressure solenoid after print pass\n")
            self.f.write("M42 P0 S0\n")

    def segment_bc(self, tangents: np.ndarray, i0: int, i1: int) -> Tuple[float, float]:
        mid = max(int(i0), min(int(i1), (int(i0) + int(i1)) // 2))
        return self.bc_for_tangent(np.asarray(tangents[mid], dtype=float))

    def print_segment_fixed_bc(self, points: np.ndarray, tangents: np.ndarray, i0: int, i1: int, b: float, c: float) -> None:
        if i1 <= i0:
            return

        for i in range(i0 + 1, i1 + 1):
            p0 = np.asarray(points[i - 1], dtype=float)
            p1 = np.asarray(points[i], dtype=float)
            t0 = np.asarray(tangents[i - 1], dtype=float)
            t1 = np.asarray(tangents[i], dtype=float)

            for s in range(1, self.edge_samples + 1):
                u = s / float(self.edge_samples)
                p_tip = p0 + u * (p1 - p0)
                tangent = normalize((1.0 - u) * t0 + u * t1)
                p_stage = self.tip_to_stage_with_bc(p_tip, b, c)
                self.write_move(p_stage, b, c, self.print_feed, comment=None, u_value=None)
                self.cur_tip_xyz = p_tip.copy()
                self.last_tip_tangent = tangent.copy()

    def tracked_tip_hold_bc_transition(
        self,
        p_tip: np.ndarray,
        b_start: float,
        b_end: float,
        c: float,
        tangent: Optional[np.ndarray],
        feed: float,
        comment: Optional[str] = None,
    ) -> None:
        p_tip_arr = np.asarray(p_tip, dtype=float)
        tangent_arr = None if tangent is None else np.asarray(tangent, dtype=float).copy()
        db = abs(float(b_end) - float(b_start))
        n_steps = max(8, int(math.ceil(db / 2.0)))

        if comment:
            self.f.write(f"; {comment}\n")

        for i in range(1, n_steps + 1):
            u = i / float(n_steps)
            # Smooth endpoint velocity to mimic the sampled tracked style used in calib_plane_daq.py.
            s = u * u * (3.0 - 2.0 * u)
            b = (1.0 - s) * float(b_start) + s * float(b_end)
            p_stage = self.tip_to_stage_with_bc(p_tip_arr, b, c)
            self.write_move(p_stage, b, c, feed, comment=None, u_value=None)
            self.cur_tip_xyz = p_tip_arr.copy()
            self.last_tip_tangent = None if tangent_arr is None else tangent_arr.copy()

    def approach_start(
        self,
        start_tip: np.ndarray,
        start_tangent: np.ndarray,
        travel_lift_z: float,
        approach_side_mm: float,
    ) -> None:
        start_tip = np.asarray(start_tip, dtype=float)
        start_tangent = normalize(np.asarray(start_tangent, dtype=float))
        side = side_vector_from_tangent(start_tangent, fallback=self.last_tip_tangent)
        far_tip = start_tip - side * float(approach_side_mm) + np.array([0.0, 0.0, float(travel_lift_z)], dtype=float)
        near_tip = start_tip - side * (0.5 * float(approach_side_mm))

        if self.cur_tip_xyz is None:
            self.move_to_tip(far_tip, tangent=start_tangent, feed=self.travel_feed, comment="travel above and to the side of wave start")
            self.move_to_tip(near_tip, tangent=start_tangent, feed=self.approach_feed, comment="approach near the wave start")
            self.move_to_tip(start_tip, tangent=start_tangent, feed=self.fine_approach_feed, comment="fine approach to wave start")
            return

        retreat_side = side_vector_from_tangent(self.last_tip_tangent if self.last_tip_tangent is not None else start_tangent, fallback=side)
        retreat_tip = np.asarray(self.cur_tip_xyz, dtype=float) + retreat_side * float(approach_side_mm) + np.array([0.0, 0.0, float(travel_lift_z)], dtype=float)
        retreat_tangent = self.last_tip_tangent if self.last_tip_tangent is not None else start_tangent
        self.move_to_tip(retreat_tip, tangent=retreat_tangent, feed=self.approach_feed, comment="retreat from previous end")
        self.move_to_tip(far_tip, tangent=start_tangent, feed=self.travel_feed, comment="travel above and to the side of wave start")
        self.move_to_tip(near_tip, tangent=start_tangent, feed=self.approach_feed, comment="approach near the wave start")
        self.move_to_tip(start_tip, tangent=start_tangent, feed=self.fine_approach_feed, comment="fine approach to wave start")

    def print_polyline(self, points: np.ndarray, tangents: np.ndarray) -> None:
        if len(points) < 2:
            return

        self.f.write(
            "; WAVE_WRITE_START "
            f"point_count={len(points)} "
            f"tip_start_x={float(points[0, 0]):.6f} tip_start_y={float(points[0, 1]):.6f} tip_start_z={float(points[0, 2]):.6f} "
            f"tip_end_x={float(points[-1, 0]):.6f} tip_end_y={float(points[-1, 1]):.6f} tip_end_z={float(points[-1, 2]):.6f} "
            "tip_angle_convention=0_posZ_90_horizontal_180_negZ\n"
        )

        self.cur_tip_xyz = np.asarray(points[0], dtype=float).copy()
        self.last_tip_tangent = np.asarray(tangents[0], dtype=float).copy()

        if self.orientation_mode == "extrema-step":
            features = find_extrema_feature_indices(points)
            segment_bcs = [self.segment_bc(tangents, i0, i1) for i0, i1 in zip(features[:-1], features[1:])]

            if segment_bcs:
                first_b, first_c = segment_bcs[0]
                self.tracked_tip_hold_bc_transition(
                    points[0],
                    b_start=self.cur_b,
                    b_end=first_b,
                    c=first_c,
                    tangent=tangents[0],
                    feed=self.fine_approach_feed,
                    comment="compensated initial B/C reposition before stepped wave write",
                )
                self.move_to_tip_with_bc(
                    points[0],
                    b=first_b,
                    c=first_c,
                    feed=self.fine_approach_feed,
                    tangent=tangents[0],
                    comment=None,
                )

            for seg_idx, (i0, i1) in enumerate(zip(features[:-1], features[1:])):
                b_seg, c_seg = segment_bcs[seg_idx]
                self.pressure_preload_before_print()
                self.print_segment_fixed_bc(points, tangents, i0, i1, b=b_seg, c=c_seg)

                if seg_idx + 1 < len(segment_bcs):
                    self.pressure_release_after_print()
                    b_next, c_next = segment_bcs[seg_idx + 1]
                    if abs(float(b_next) - float(b_seg)) > 1e-9 or abs(float(c_next) - float(c_seg)) > 1e-9:
                        feature_idx = i1
                        self.tracked_tip_hold_bc_transition(
                            points[feature_idx],
                            b_start=b_seg,
                            b_end=b_next,
                            c=c_next,
                            tangent=tangents[feature_idx],
                            feed=self.approach_feed,
                            comment="compensated B/C reposition at wave extremum with extrusion disabled",
                        )
                        self.move_to_tip_with_bc(
                            points[feature_idx],
                            b=b_next,
                            c=c_next,
                            feed=self.approach_feed,
                            tangent=tangents[feature_idx],
                            comment=None,
                        )

            self.pressure_release_after_print()
            self.f.write("; WAVE_WRITE_END\n")
            return

        self.pressure_preload_before_print()

        for i in range(1, len(points)):
            p0 = np.asarray(points[i - 1], dtype=float)
            p1 = np.asarray(points[i], dtype=float)
            t0 = np.asarray(tangents[i - 1], dtype=float)
            t1 = np.asarray(tangents[i], dtype=float)

            for s in range(1, self.edge_samples + 1):
                u = s / float(self.edge_samples)
                p_tip = p0 + u * (p1 - p0)
                tangent = normalize((1.0 - u) * t0 + u * t1)
                p_stage, b, c = self.tip_to_stage(p_tip, tangent=tangent)

                self.write_move(p_stage, b, c, self.print_feed, comment=None, u_value=None)
                self.cur_tip_xyz = p_tip.copy()
                self.last_tip_tangent = tangent.copy()

        self.pressure_release_after_print()
        self.f.write("; WAVE_WRITE_END\n")

    def finish(self, travel_lift_z: float) -> None:
        if self.cur_tip_xyz is None or self.last_tip_tangent is None:
            return
        end_tip = np.asarray(self.cur_tip_xyz, dtype=float)
        retreat_side = side_vector_from_tangent(self.last_tip_tangent)
        retreat_tip = end_tip + retreat_side * 4.0 + np.array([0.0, 0.0, float(travel_lift_z)], dtype=float)
        self.move_to_tip(retreat_tip, tangent=self.last_tip_tangent, feed=self.approach_feed, comment="retreat after wave")


# ---------------- Top-level generation ----------------
def write_gaussian_wave_gcode(
    out: str,
    calibration: Optional[str],
    write_mode: str,
    orientation_mode: str,
    y_offplane_sign: float,
    r_fit_model: Optional[str],
    z_fit_model: Optional[str],
    x_start: float,
    x_end: float,
    y: float,
    z_min: float,
    z_max: float,
    z_baseline: float,
    z_amplitude: float,
    cycles: float,
    phase_deg: float,
    x_center: float,
    gaussian_sigma: float,
    no_gaussian: bool,
    lead_in: float,
    lead_out: float,
    points: int,
    window_power: float,
    tangent_smooth_window: int,
    centerline_smooth_window: int,
    fixed_b: float,
    fixed_c: float,
    c_deg: float,
    b_angle_bias_deg: float,
    bc_solve_samples: int,
    travel_feed: float,
    approach_feed: float,
    fine_approach_feed: float,
    print_feed: float,
    travel_lift_z: float,
    approach_side_mm: float,
    edge_samples: int,
    emit_extrusion: bool,
    extrusion_per_mm: float,
    pressure_offset_mm: float,
    pressure_advance_feed: float,
    pressure_retract_feed: float,
    preflow_dwell_ms: int,
    bbox_x_min: float,
    bbox_x_max: float,
    bbox_y_min: float,
    bbox_y_max: float,
    bbox_z_min: float,
    bbox_z_max: float,
) -> Dict[str, Any]:
    write_mode = str(write_mode).strip().lower()
    orientation_mode = str(orientation_mode).strip().lower()
    if write_mode not in {"calibrated", "cartesian"}:
        raise ValueError("--write-mode must be calibrated or cartesian")
    if orientation_mode not in {"tangent", "fixed", "extrema-step"}:
        raise ValueError("--orientation-mode must be tangent, fixed, or extrema-step")
    if write_mode == "calibrated" and not calibration:
        raise ValueError("--calibration is required when --write-mode calibrated")

    cal: Optional[Calibration]
    if write_mode == "calibrated":
        cal = load_calibration(
            str(calibration),
            requested_r_fit_model=r_fit_model,
            requested_z_fit_model=z_fit_model,
            requested_offplane_fit_model=DEFAULT_Y_OFFPLANE_FIT_MODEL,
        )
        cal.offplane_y_sign = float(y_offplane_sign)
    else:
        cal = None

    bbox = {
        "x_min": float(bbox_x_min),
        "x_max": float(bbox_x_max),
        "y_min": float(bbox_y_min),
        "y_max": float(bbox_y_max),
        "z_min": float(bbox_z_min),
        "z_max": float(bbox_z_max),
    }

    pts = build_gaussian_sine_wave_points(
        x_start=x_start,
        x_end=x_end,
        y=y,
        z_baseline=z_baseline,
        z_amplitude=z_amplitude,
        cycles=cycles,
        phase_deg=phase_deg,
        x_center=x_center,
        gaussian_sigma=gaussian_sigma,
        no_gaussian=no_gaussian,
        lead_in=lead_in,
        lead_out=lead_out,
        points=points,
        window_power=window_power,
    )
    tangents = build_tangents_for_points(
        pts,
        smooth_window=tangent_smooth_window,
        centerline_smooth_window=centerline_smooth_window,
    )

    # Validate requested tip-space box.
    z_lo = float(np.min(pts[:, 2]))
    z_hi = float(np.max(pts[:, 2]))
    x_lo = float(np.min(pts[:, 0]))
    x_hi = float(np.max(pts[:, 0]))
    if x_lo < float(x_start) - 1e-6 or x_hi > float(x_end) + 1e-6:
        raise ValueError("Generated X coordinates exceed the requested [x-start, x-end] range.")
    if z_lo < float(z_min) - 1e-6 or z_hi > float(z_max) + 1e-6:
        raise ValueError(
            f"Generated Z coordinates [{z_lo:.3f}, {z_hi:.3f}] exceed the requested "
            f"[{float(z_min):.3f}, {float(z_max):.3f}] range. Reduce --z-amplitude or adjust baseline / sigma / cycles."
        )

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = GCodeWriter(
        fh=None,  # type: ignore[arg-type]
        cal=cal,
        write_mode=write_mode,
        orientation_mode=orientation_mode,
        fixed_b=fixed_b,
        fixed_c=fixed_c,
        c_deg=c_deg,
        b_angle_bias_deg=b_angle_bias_deg,
        bc_solve_samples=bc_solve_samples,
        bbox=bbox,
        travel_feed=travel_feed,
        approach_feed=approach_feed,
        fine_approach_feed=fine_approach_feed,
        print_feed=print_feed,
        edge_samples=edge_samples,
        emit_extrusion=emit_extrusion,
        extrusion_per_mm=extrusion_per_mm,
        pressure_offset_mm=pressure_offset_mm,
        pressure_advance_feed=pressure_advance_feed,
        pressure_retract_feed=pressure_retract_feed,
        preflow_dwell_ms=preflow_dwell_ms,
    )

    with out_path.open("w", encoding="utf-8") as fh:
        writer.f = fh

        fh.write("; Gaussian-modulated sine wave in the XZ plane\n")
        fh.write("; Generated by gaussian_wave_xz_generator.py\n")
        fh.write(f"; write_mode={write_mode} orientation_mode={orientation_mode}\n")
        fh.write(
            "; requested_tip_box "
            f"x=[{float(x_start):.3f},{float(x_end):.3f}] "
            f"y={float(y):.3f} "
            f"z=[{float(z_min):.3f},{float(z_max):.3f}]\n"
        )
        fh.write(
            "; wave_parameters "
            f"z_baseline={float(z_baseline):.3f} "
            f"z_amplitude={float(z_amplitude):.3f} "
            f"cycles={float(cycles):.6f} "
            f"phase_deg={float(phase_deg):.3f} "
            f"x_center={float(x_center):.3f} "
            f"gaussian_sigma={float(gaussian_sigma):.3f} "
            f"no_gaussian={int(bool(no_gaussian))} "
            f"lead_in={float(lead_in):.3f} "
            f"lead_out={float(lead_out):.3f} "
            f"window_power={float(window_power):.3f}\n"
        )
        fh.write("G21\n")
        fh.write("G90\n")
        fh.write(f"; B-angle convention: 0=up, 90=horizontal, 180=down\n")
        fh.write(f"; C held constant at {float(c_deg):.3f} deg by default\n")
        if cal is not None:
            fh.write(
                "; calibration_models "
                f"r_requested={cal.requested_r_fit_model or 'calibration_default'} "
                f"r_resolved={cal.resolved_r_fit_model or 'unspecified'} "
                f"z_requested={cal.requested_z_fit_model or 'calibration_default'} "
                f"z_resolved={cal.resolved_z_fit_model or 'unspecified'} "
                f"y_offplane_requested={cal.requested_offplane_fit_model or 'calibration_default'} "
                f"y_offplane_resolved={cal.resolved_offplane_fit_model or 'unspecified'} "
                f"calibration_default={cal.selected_offplane_fit_model or cal.selected_fit_model or 'unspecified'} "
                f"active_phase={cal.active_phase}\n"
            )

        start_tip = pts[0]
        start_tangent = tangents[0]
        writer.approach_start(
            start_tip=start_tip,
            start_tangent=start_tangent,
            travel_lift_z=travel_lift_z,
            approach_side_mm=approach_side_mm,
        )
        writer.print_polyline(pts, tangents)
        writer.finish(travel_lift_z=travel_lift_z)

        fh.write("; End of file\n")

    tip_b = np.array([desired_physical_b_angle_from_tangent(t) for t in tangents], dtype=float)
    if orientation_mode == "fixed":
        b_used = np.full_like(tip_b, float(fixed_b), dtype=float)
    elif orientation_mode == "extrema-step":
        feature_indices = find_extrema_feature_indices(pts)
        b_used = np.zeros_like(tip_b, dtype=float)
        for i0, i1 in zip(feature_indices[:-1], feature_indices[1:]):
            t_mid = tangents[max(i0, min(i1, (i0 + i1) // 2))]
            if write_mode == "calibrated":
                assert cal is not None
                target_b = float(np.clip(desired_physical_b_angle_from_tangent(t_mid) + b_angle_bias_deg, 0.0, 180.0))
                b_seg = solve_b_for_target_tip_angle(cal, target_b, search_samples=bc_solve_samples)
            else:
                b_seg = float(np.clip(desired_physical_b_angle_from_tangent(t_mid) + b_angle_bias_deg, 0.0, 180.0))
            b_used[i0 : i1 + 1] = float(b_seg)
    elif write_mode == "calibrated":
        assert cal is not None
        b_used = np.array(
            [
                solve_b_for_target_tip_angle(cal, float(np.clip(bv + b_angle_bias_deg, 0.0, 180.0)), search_samples=bc_solve_samples)
                for bv in tip_b
            ],
            dtype=float,
        )
    else:
        b_used = np.clip(tip_b + float(b_angle_bias_deg), 0.0, 180.0)

    summary = {
        "out": str(out_path),
        "write_mode": write_mode,
        "orientation_mode": orientation_mode,
        "tip_x_range": (float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))),
        "tip_y_range": (float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))),
        "tip_z_range": (float(np.min(pts[:, 2])), float(np.max(pts[:, 2]))),
        "tip_b_target_range_deg": (float(np.min(tip_b)), float(np.max(tip_b))),
        "b_command_range_deg": (float(np.min(b_used)), float(np.max(b_used))),
        "c_command_deg": float(c_deg if orientation_mode in {"tangent", "extrema-step"} else fixed_c),
        "no_gaussian": bool(no_gaussian),
        "stage_xyz_range": {
            "x": (float(writer.stage_min[0]), float(writer.stage_max[0])),
            "y": (float(writer.stage_min[1]), float(writer.stage_max[1])),
            "z": (float(writer.stage_min[2]), float(writer.stage_max[2])),
        },
        "warnings": list(writer.warnings),
    }
    return summary


# ---------------- CLI ----------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Generate calibrated or Cartesian G-code for a Gaussian-modulated sine wave in the XZ plane. "
            "Default tip-space box is X=[50,130], Y=52, Z=[-190,-150]. "
            "In tangent mode, B follows the local path tangent with 0 deg=up, 90 deg=horizontal, 180 deg=down; "
            "C is held constant at 180 deg by default."
        )
    )
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code file.")
    ap.add_argument("--calibration", default=None, help="Calibration JSON. Required for --write-mode calibrated.")
    ap.add_argument("--write-mode", choices=["calibrated", "cartesian"], default=DEFAULT_WRITE_MODE)
    ap.add_argument("--orientation-mode", choices=["tangent", "fixed", "extrema-step"], default=DEFAULT_ORIENTATION_MODE)
    ap.add_argument("--y-offplane-sign", type=float, default=DEFAULT_Y_OFFPLANE_SIGN, help="Multiplier applied to the calibration off-plane Y term in calibrated mode. Use -1 to flip the sign.")
    ap.add_argument("--r-fit-model", default=None, help="Optional radial fit override, e.g. avg_cubic or cubic.")
    ap.add_argument("--z-fit-model", default=None, help="Optional Z fit override, e.g. avg_cubic or cubic.")

    ap.add_argument("--x-start", type=float, default=DEFAULT_X_START)
    ap.add_argument("--x-end", type=float, default=DEFAULT_X_END)
    ap.add_argument("--y", type=float, default=DEFAULT_Y)
    ap.add_argument("--z-min", type=float, default=DEFAULT_Z_MIN, help="Requested lower tip-space Z bound. The generated path must stay within this range.")
    ap.add_argument("--z-max", type=float, default=DEFAULT_Z_MAX, help="Requested upper tip-space Z bound. The generated path must stay within this range.")
    ap.add_argument("--z-baseline", type=float, default=DEFAULT_Z_BASELINE, help="Baseline Z around which the wave oscillates.")
    ap.add_argument("--z-amplitude", type=float, default=DEFAULT_Z_AMPLITUDE, help="Peak amplitude multiplier before tapering / Gaussian weighting.")
    ap.add_argument("--cycles", type=float, default=DEFAULT_CYCLES, help="Number of sine cycles across the active wave interval.")
    ap.add_argument("--phase-deg", type=float, default=DEFAULT_PHASE_DEG, help="Phase offset in degrees.")
    ap.add_argument("--x-center", type=float, default=DEFAULT_X_CENTER, help="Center of the Gaussian envelope in X.")
    ap.add_argument("--gaussian-sigma", type=float, default=DEFAULT_GAUSSIAN_SIGMA, help="Sigma of the Gaussian envelope in X units.")
    ap.add_argument("--no-gaussian", action="store_true", help="Disable Gaussian modulation and use a pure sine-wave envelope.")
    ap.add_argument("--lead-in", type=float, default=DEFAULT_LEAD_IN, help="Horizontal lead-in segment length at the start of the wave.")
    ap.add_argument("--lead-out", type=float, default=DEFAULT_LEAD_OUT, help="Horizontal lead-out segment length at the end of the wave.")
    ap.add_argument("--window-power", type=float, default=DEFAULT_WINDOW_POWER, help="Endpoint taper exponent. Higher values sharpen the fade-in / fade-out.")
    ap.add_argument("--points", type=int, default=DEFAULT_POINTS, help="Number of tip-space sample points before edge subdivision.")
    ap.add_argument("--tangent-smooth-window", type=int, default=DEFAULT_TANGENT_SMOOTH_WINDOW)
    ap.add_argument("--centerline-smooth-window", type=int, default=DEFAULT_CENTERLINE_SMOOTH_WINDOW)

    ap.add_argument("--fixed-b", type=float, default=DEFAULT_FIXED_B, help="Used only when --orientation-mode fixed.")
    ap.add_argument("--fixed-c", type=float, default=DEFAULT_FIXED_C, help="Used only when --orientation-mode fixed.")
    ap.add_argument("--c-deg", type=float, default=DEFAULT_C_DEG, help="Constant C angle used in tangent mode. Default is 180 deg for an XZ-plane write.")
    ap.add_argument("--b-angle-bias-deg", type=float, default=DEFAULT_B_ANGLE_BIAS_DEG, help="Bias added to the tangent-derived B target before solving / emitting.")
    ap.add_argument("--bc-solve-samples", type=int, default=DEFAULT_BC_SOLVE_SAMPLES)

    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED)
    ap.add_argument("--approach-feed", type=float, default=DEFAULT_APPROACH_FEED)
    ap.add_argument("--fine-approach-feed", type=float, default=DEFAULT_FINE_APPROACH_FEED)
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED)
    ap.add_argument("--travel-lift-z", type=float, default=DEFAULT_TRAVEL_LIFT_Z)
    ap.add_argument("--approach-side-mm", type=float, default=DEFAULT_APPROACH_SIDE_MM)
    ap.add_argument("--edge-samples", type=int, default=DEFAULT_EDGE_SAMPLES, help="Subdivide each polyline segment into this many printed G1 moves.")

    ap.add_argument("--emit-extrusion", dest="emit_extrusion", action="store_true", default=DEFAULT_EMIT_EXTRUSION)
    ap.add_argument("--no-emit-extrusion", dest="emit_extrusion", action="store_false")
    ap.add_argument("--extrusion-per-mm", type=float, default=DEFAULT_EXTRUSION_PER_MM)
    ap.add_argument("--pressure-offset-mm", type=float, default=DEFAULT_PRESSURE_OFFSET_MM)
    ap.add_argument("--pressure-advance-feed", type=float, default=DEFAULT_PRESSURE_ADVANCE_FEED)
    ap.add_argument("--pressure-retract-feed", type=float, default=DEFAULT_PRESSURE_RETRACT_FEED)
    ap.add_argument("--preflow-dwell-ms", type=int, default=DEFAULT_PREFLOW_DWELL_MS)

    ap.add_argument("--bbox-x-min", type=float, default=DEFAULT_BBOX_X_MIN)
    ap.add_argument("--bbox-x-max", type=float, default=DEFAULT_BBOX_X_MAX)
    ap.add_argument("--bbox-y-min", type=float, default=DEFAULT_BBOX_Y_MIN)
    ap.add_argument("--bbox-y-max", type=float, default=DEFAULT_BBOX_Y_MAX)
    ap.add_argument("--bbox-z-min", type=float, default=DEFAULT_BBOX_Z_MIN)
    ap.add_argument("--bbox-z-max", type=float, default=DEFAULT_BBOX_Z_MAX)

    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    summary = write_gaussian_wave_gcode(**vars(args))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
