#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code for a single plain sine wave in
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
DEFAULT_OUT = "gcode_generation/sine_wave_xz.gcode"

# Path placement
DEFAULT_X_START = 50.0
DEFAULT_X_END = 120.0
DEFAULT_Y = 42.0
DEFAULT_Z_MIN = -155.0
DEFAULT_Z_MAX = -100.0
DEFAULT_Z_BASELINE = 0.5 * (DEFAULT_Z_MIN + DEFAULT_Z_MAX)

# Wave shape
DEFAULT_Z_AMPLITUDE = 10.0
DEFAULT_CYCLES = 3.0
DEFAULT_PHASE_DEG = 90.0
DEFAULT_LEAD_IN = 0.0
DEFAULT_LEAD_OUT = 0.0
DEFAULT_POINTS = 501
DEFAULT_TANGENT_SMOOTH_WINDOW = 4
DEFAULT_CENTERLINE_SMOOTH_WINDOW = 0
DEFAULT_MIN_TANGENT_XY = 1e-9
DEFAULT_POINT_MERGE_TOL = 1e-9

# Orientation
DEFAULT_WRITE_MODE = "calibrated"
DEFAULT_ORIENTATION_MODE = "tangent"
DEFAULT_FIXED_B = 90.0
DEFAULT_FIXED_C = 180.0
DEFAULT_C_DEG = 180.0
DEFAULT_B_ANGLE_BIAS_DEG = 0.0
DEFAULT_BC_SOLVE_SAMPLES = 5001
DEFAULT_Y_OFFPLANE_SIGN = -1.0
DEFAULT_PHASE_TRANSITION_VIA_EXTREMA = False
DEFAULT_X_ONLY_B_CYCLE_MODE = False

# Motion
DEFAULT_TRAVEL_FEED = 2000.0
DEFAULT_APPROACH_FEED = 1200.0
DEFAULT_FINE_APPROACH_FEED = 150.0
DEFAULT_PRINT_FEED = 400.0
DEFAULT_TRAVEL_LIFT_Z = 8.0
DEFAULT_APPROACH_SIDE_MM = 4.0
DEFAULT_EDGE_SAMPLES = 1
DEFAULT_SAFE_APPROACH_Z = -50.0

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
    phase_models: Optional[Dict[str, Dict[str, Any]]] = None
    selected_fit_model: Optional[str] = None
    selected_offplane_fit_model: Optional[str] = None
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


def _normalize_motion_phase_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _resolve_phase_name(cal: Calibration, base_name: str) -> str:
    want = _normalize_motion_phase_name(base_name)
    if want is None:
        return str(cal.active_phase)

    phase_models = cal.phase_models or {}
    phase_keys = list(phase_models.keys())
    if want in phase_models:
        return want

    prefix_matches = sorted(k for k in phase_keys if k.startswith(want))
    if prefix_matches:
        return prefix_matches[0]

    contains_matches = sorted(k for k in phase_keys if want in k)
    if contains_matches:
        return contains_matches[0]

    return str(cal.active_phase)


def _select_phase_model(cal: Calibration, model_name: str, motion_phase: Optional[str]) -> Optional[Dict[str, Any]]:
    phase_name = _normalize_motion_phase_name(motion_phase) or str(cal.active_phase)
    phase_models = cal.phase_models or {}
    payload = phase_models.get(phase_name)
    if isinstance(payload, dict):
        spec = _normalize_model_spec(payload.get(model_name))
        if spec is not None:
            return spec

    fallback_attr = {
        "r": cal.r_model,
        "z": cal.z_model,
        "offplane_y": cal.y_off_model,
        "tip_angle": cal.tip_angle_model,
    }.get(model_name)
    return None if fallback_attr is None else dict(fallback_attr)


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


def _model_b_range(model_spec: Optional[Dict[str, Any]], fallback_lo: float, fallback_hi: float) -> Tuple[float, float]:
    if model_spec is None:
        return float(fallback_lo), float(fallback_hi)
    x_knots = np.asarray(model_spec.get("x_knots", []), dtype=float).reshape(-1)
    if x_knots.size >= 2:
        return float(np.min(x_knots)), float(np.max(x_knots))
    fit_x_range = model_spec.get("fit_x_range")
    if fit_x_range is not None and len(fit_x_range) == 2:
        return float(min(fit_x_range)), float(max(fit_x_range))
    return float(fallback_lo), float(fallback_hi)


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
    active_phase = str(data.get("default_phase_for_legacy_access") or "pull").strip().lower()

    raw_phase_models = data.get("fit_models_by_phase", {}) or {}
    phase_models: Dict[str, Dict[str, Any]] = {}
    if isinstance(raw_phase_models, dict):
        for raw_phase_name, raw_models in raw_phase_models.items():
            phase_name = _normalize_motion_phase_name(raw_phase_name)
            if phase_name is None or not isinstance(raw_models, dict):
                continue
            phase_models[phase_name] = dict(raw_models)
    active_phase_models = phase_models.get(active_phase) if isinstance(phase_models, dict) else None
    if not isinstance(active_phase_models, dict):
        active_phase_models = fit_models
        if phase_models:
            active_phase = _resolve_phase_name(
                Calibration(
                    pr=np.zeros(1, dtype=float),
                    pz=np.zeros(1, dtype=float),
                    py_off=None,
                    pa=None,
                    b_min=-5.0,
                    b_max=0.0,
                    x_axis="X",
                    y_axis="Y",
                    z_axis="Z",
                    b_axis="B",
                    c_axis="C",
                    u_axis="U",
                    c_180_deg=180.0,
                    phase_models=phase_models,
                    active_phase=active_phase,
                ),
                active_phase,
            )
            active_phase_models = phase_models.get(active_phase, fit_models)

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
        phase_models=phase_models,
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


def calibration_with_phase_cubic_override(cal: Calibration) -> Calibration:
    cal_out = Calibration(
        pr=np.asarray(cal.pr, dtype=float).copy(),
        pz=np.asarray(cal.pz, dtype=float).copy(),
        py_off=None if cal.py_off is None else np.asarray(cal.py_off, dtype=float).copy(),
        pa=None if cal.pa is None else np.asarray(cal.pa, dtype=float).copy(),
        b_min=float(cal.b_min),
        b_max=float(cal.b_max),
        x_axis=str(cal.x_axis),
        y_axis=str(cal.y_axis),
        z_axis=str(cal.z_axis),
        b_axis=str(cal.b_axis),
        c_axis=str(cal.c_axis),
        u_axis=str(cal.u_axis),
        c_180_deg=float(cal.c_180_deg),
        r_model=None if cal.r_model is None else dict(cal.r_model),
        z_model=None if cal.z_model is None else dict(cal.z_model),
        y_off_model=None if cal.y_off_model is None else dict(cal.y_off_model),
        y_off_extrap_model=None if cal.y_off_extrap_model is None else dict(cal.y_off_extrap_model),
        tip_angle_model=None if cal.tip_angle_model is None else dict(cal.tip_angle_model),
        phase_models=None if cal.phase_models is None else {k: dict(v) for k, v in cal.phase_models.items()},
        selected_fit_model=cal.selected_fit_model,
        selected_offplane_fit_model=cal.selected_offplane_fit_model,
        active_phase=str(cal.active_phase),
        offplane_y_sign=float(cal.offplane_y_sign),
    )
    if not cal_out.phase_models:
        return cal_out
    for phase_name, payload in cal_out.phase_models.items():
        if not isinstance(payload, dict):
            continue
        for base_name, cubic_name in (("r", "r_cubic"), ("z", "z_cubic"), ("tip_angle", "tip_angle_cubic")):
            cubic_model = _normalize_model_spec(payload.get(cubic_name))
            if cubic_model is not None:
                payload[base_name] = cubic_model
    active_payload = cal_out.phase_models.get(str(cal_out.active_phase))
    if isinstance(active_payload, dict):
        cal_out.r_model = _normalize_model_spec(active_payload.get("r"))
        cal_out.z_model = _normalize_model_spec(active_payload.get("z"))
        cal_out.tip_angle_model = _normalize_model_spec(active_payload.get("tip_angle"))
        cal_out.y_off_model = _normalize_model_spec(active_payload.get("offplane_y"))
        cal_out.y_off_extrap_model = _normalize_model_spec(active_payload.get("offplane_y_linear")) or cal_out.y_off_model
    return cal_out


def eval_r(cal: Calibration, b: Any, motion_phase: Optional[str] = None) -> np.ndarray:
    model = _select_phase_model(cal, "r", motion_phase=motion_phase)
    if model is not None:
        return eval_model_spec(model, b)
    return poly_eval(cal.pr, b)


def eval_z(cal: Calibration, b: Any, motion_phase: Optional[str] = None) -> np.ndarray:
    model = _select_phase_model(cal, "z", motion_phase=motion_phase)
    if model is not None:
        return eval_model_spec(model, b)
    return poly_eval(cal.pz, b)


def eval_offplane_y(cal: Calibration, b: Any, motion_phase: Optional[str] = None) -> np.ndarray:
    model = _select_phase_model(cal, "offplane_y", motion_phase=motion_phase)
    extrap_model = cal.y_off_extrap_model
    if motion_phase is not None and cal.phase_models:
        phase_payload = cal.phase_models.get(_normalize_motion_phase_name(motion_phase) or "")
        if isinstance(phase_payload, dict):
            extrap_model = _normalize_model_spec(phase_payload.get("offplane_y_linear")) or model
    if model is not None:
        if str(model.get("model_type", "")).lower() == "pchip":
            values = eval_pchip_with_linear_extrap(model, extrap_model, b)
        else:
            values = eval_model_spec(model, b, default_if_none=0.0)
    else:
        values = poly_eval(cal.py_off, b, default_if_none=0.0)
    return float(cal.offplane_y_sign) * np.asarray(values, dtype=float)


def eval_tip_angle_deg(cal: Calibration, b: Any, motion_phase: Optional[str] = None) -> np.ndarray:
    model = _select_phase_model(cal, "tip_angle", motion_phase=motion_phase)
    if model is not None:
        return eval_model_spec(model, b)
    if cal.pa is None:
        raise ValueError("Calibration is missing tip_angle_coeffs.")
    return poly_eval(cal.pa, b)


def predict_tip_offset_xyz(cal: Calibration, b: float, c_deg: float, motion_phase: Optional[str] = None) -> np.ndarray:
    r = float(eval_r(cal, b, motion_phase=motion_phase))
    z = float(eval_z(cal, b, motion_phase=motion_phase))
    y_off = float(eval_offplane_y(cal, b, motion_phase=motion_phase))
    c = math.radians(float(c_deg))
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def stage_xyz_for_tip(cal: Calibration, tip_xyz: np.ndarray, b: float, c_deg: float, motion_phase: Optional[str] = None) -> np.ndarray:
    return np.asarray(tip_xyz, dtype=float) - predict_tip_offset_xyz(cal, b, c_deg, motion_phase=motion_phase)


def solve_b_for_target_tip_angle(
    cal: Calibration,
    target_angle_deg: float,
    search_samples: int = DEFAULT_BC_SOLVE_SAMPLES,
    motion_phase: Optional[str] = None,
) -> float:
    tip_angle_model = _select_phase_model(cal, "tip_angle", motion_phase=motion_phase)
    b_lo, b_hi = _model_b_range(tip_angle_model, float(cal.b_min), float(cal.b_max))
    bb = np.linspace(b_lo, b_hi, int(max(101, search_samples)))
    aa = eval_tip_angle_deg(cal, bb, motion_phase=motion_phase) - float(target_angle_deg)
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
            return float(eval_tip_angle_deg(cal, x, motion_phase=motion_phase) - float(target_angle_deg))

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
def build_sine_wave_points(
    x_start: float,
    x_end: float,
    y: float,
    z_baseline: float,
    z_amplitude: float,
    cycles: float,
    phase_deg: float,
    lead_in: float,
    lead_out: float,
    points: int,
) -> np.ndarray:
    """
    Build a plain sine-wave tip path in the XZ plane at constant Y.

    Active wave equation, over the active interval after optional lead-in / lead-out:
        Z = z_baseline + z_amplitude * sin(2*pi*cycles*s + phase)
    where s runs from 0 to 1. There is intentionally no Gaussian envelope and
    no endpoint amplitude taper.
    """
    x0 = float(x_start)
    x1 = float(x_end)
    if not x1 > x0:
        raise ValueError("--x-end must be greater than --x-start")
    if float(lead_in) < 0.0 or float(lead_out) < 0.0:
        raise ValueError("--lead-in and --lead-out must be >= 0")
    if float(lead_in + lead_out) >= (x1 - x0):
        raise ValueError("--lead-in + --lead-out must be smaller than the total X span")
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
    zs = float(z_baseline) + float(z_amplitude) * np.sin(2.0 * math.pi * float(cycles) * s + phase)
    ys = np.full_like(xs, float(y), dtype=float)
    parts.append(np.column_stack([xs, ys, zs]))

    if n_lead_out > 0:
        xs = np.linspace(wave_x1, x1, n_lead_out)
        zs = np.full_like(xs, float(z_baseline), dtype=float)
        ys = np.full_like(xs, float(y), dtype=float)
        parts.append(np.column_stack([xs, ys, zs]))

    out = np.vstack(parts)
    return deduplicate_polyline_points(out)


def build_x_only_b_cycle_points(
    x_start: float,
    x_end: float,
    y: float,
    z_baseline: float,
    points: int,
) -> np.ndarray:
    x0 = float(x_start)
    x1 = float(x_end)
    if not x1 > x0:
        raise ValueError("--x-end must be greater than --x-start")
    if int(points) < 3:
        raise ValueError("--points must be >= 3 for x-only B-cycle mode")
    xs = np.linspace(x0, x1, int(points))
    ys = np.full_like(xs, float(y), dtype=float)
    zs = np.full_like(xs, float(z_baseline), dtype=float)
    return np.column_stack([xs, ys, zs])


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
        phase_transition_via_extrema: bool,
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
        self.phase_transition_via_extrema = bool(phase_transition_via_extrema)
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
        self.cur_motion_phase: Optional[str] = None
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
            self.fixed_b = solve_b_for_target_tip_angle(self.cal, 90.0, search_samples=self.bc_solve_samples, motion_phase=_resolve_phase_name(self.cal, "pull"))

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

    def _phase_names(self) -> Tuple[str, str]:
        if self.cal is None:
            return "pull", "release"
        return _resolve_phase_name(self.cal, "pull"), _resolve_phase_name(self.cal, "release")

    def _phase_boundary_b(self, from_phase: str, to_phase: str) -> float:
        if self.cal is None:
            return float(self.cur_b)
        pull_phase, release_phase = self._phase_names()
        pull_model = _select_phase_model(self.cal, "tip_angle", motion_phase=pull_phase)
        release_model = _select_phase_model(self.cal, "tip_angle", motion_phase=release_phase)
        pull_lo, pull_hi = _model_b_range(pull_model, self.cal.b_min, self.cal.b_max)
        release_lo, release_hi = _model_b_range(release_model, self.cal.b_min, self.cal.b_max)
        common_lo = max(float(pull_lo), float(release_lo))
        common_hi = min(float(pull_hi), float(release_hi))
        if to_phase == release_phase:
            return float(common_lo)
        return float(common_hi)

    def _phase_b_for_tangent(self, tangent: np.ndarray, motion_phase: str) -> float:
        target_b = desired_physical_b_angle_from_tangent(tangent) + float(self.b_angle_bias_deg)
        target_b = float(np.clip(target_b, 0.0, 180.0))
        if self.write_mode == "calibrated":
            assert self.cal is not None
            return float(
                solve_b_for_target_tip_angle(
                    self.cal,
                    target_b,
                    search_samples=self.bc_solve_samples,
                    motion_phase=motion_phase,
                )
            )
        return target_b

    def bc_for_tangent(self, tangent: np.ndarray) -> Tuple[float, float, str]:
        if self.orientation_mode == "fixed":
            return float(self.fixed_b), float(self.fixed_c), str(self.cur_motion_phase or "pull")

        if self.write_mode == "calibrated":
            assert self.cal is not None
            pull_phase, release_phase = self._phase_names()
            b_pull = self._phase_b_for_tangent(tangent, pull_phase)
            b_release = self._phase_b_for_tangent(tangent, release_phase)
            if self.cur_motion_phase == pull_phase and self.cur_b <= b_release:
                b = b_pull
                motion_phase = pull_phase
            elif self.cur_motion_phase == release_phase and self.cur_b >= b_pull:
                b = b_release
                motion_phase = release_phase
            elif abs(b_pull - self.cur_b) <= abs(b_release - self.cur_b):
                b = b_pull
                motion_phase = pull_phase
            else:
                b = b_release
                motion_phase = release_phase
            if self.cur_motion_phase is None and self.cur_stage_xyz is None:
                b = b_pull
                motion_phase = pull_phase
        else:
            b = self._phase_b_for_tangent(tangent, "pull")
            motion_phase = "pull"

        return float(b), float(self.c_deg), str(motion_phase)

    def tip_to_stage(
        self,
        p_tip: np.ndarray,
        tangent: Optional[np.ndarray],
        forced_b: Optional[float] = None,
        forced_phase: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, float, str]:
        tangent_arr = np.array([1.0, 0.0, 0.0], dtype=float) if tangent is None else normalize(np.asarray(tangent, dtype=float))
        if forced_b is None or forced_phase is None:
            b, c, motion_phase = self.bc_for_tangent(tangent_arr)
        else:
            b, c, motion_phase = float(forced_b), float(self.c_deg), str(forced_phase)

        if self.write_mode == "calibrated":
            assert self.cal is not None
            if self.cur_stage_xyz is not None and self.cur_tip_xyz is not None and self.cur_motion_phase is not None:
                prev_offset = predict_tip_offset_xyz(self.cal, self.cur_b, self.cur_c, motion_phase=self.cur_motion_phase)
                new_offset = predict_tip_offset_xyz(self.cal, b, c, motion_phase=motion_phase)
                p_stage = (
                    np.asarray(self.cur_stage_xyz, dtype=float)
                    + (np.asarray(p_tip, dtype=float) - np.asarray(self.cur_tip_xyz, dtype=float))
                    - (new_offset - prev_offset)
                )
            else:
                p_stage = stage_xyz_for_tip(self.cal, np.asarray(p_tip, dtype=float), b, c, motion_phase=motion_phase)
        else:
            p_stage = np.asarray(p_tip, dtype=float)

        return self.clamp_stage(p_stage, "tip_to_stage"), float(b), float(c), str(motion_phase)

    def transition_phase_at_current_tip(self, next_phase: str) -> None:
        if (
            not self.phase_transition_via_extrema
            or self.write_mode != "calibrated"
            or self.cal is None
            or self.cur_tip_xyz is None
            or self.cur_motion_phase is None
        ):
            return
        if str(next_phase) == str(self.cur_motion_phase):
            return

        boundary_b = self._phase_boundary_b(str(self.cur_motion_phase), str(next_phase))
        tip_here = np.asarray(self.cur_tip_xyz, dtype=float).copy()

        if abs(boundary_b - float(self.cur_b)) > 1e-9:
            boundary_stage = stage_xyz_for_tip(
                self.cal,
                tip_here,
                boundary_b,
                self.cur_c,
                motion_phase=self.cur_motion_phase,
            )
            self.write_move(
                boundary_stage,
                boundary_b,
                self.cur_c,
                self.print_feed,
                comment=f"phase transition: drive B to boundary for {self.cur_motion_phase}->{next_phase} while holding tip",
            )
            self.cur_tip_xyz = tip_here.copy()

        switched_stage = stage_xyz_for_tip(
            self.cal,
            tip_here,
            boundary_b,
            self.cur_c,
            motion_phase=next_phase,
        )
        self.write_move(
            switched_stage,
            boundary_b,
            self.cur_c,
            self.print_feed,
            comment=f"phase transition: switch to {next_phase} model while holding tip",
        )
        self.cur_tip_xyz = tip_here.copy()
        self.cur_motion_phase = str(next_phase)

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
        p_stage, b, c, motion_phase = self.tip_to_stage(np.asarray(p_tip, dtype=float), tangent=tangent)
        self.write_move(p_stage, b, c, feed, comment=comment)
        self.cur_tip_xyz = np.asarray(p_tip, dtype=float).copy()
        self.cur_motion_phase = motion_phase
        self.last_tip_tangent = None if tangent is None else np.asarray(tangent, dtype=float).copy()

    def move_to_stage(
        self,
        p_stage: np.ndarray,
        b: float,
        c: float,
        feed: float,
        comment: Optional[str] = None,
        tip_hint: Optional[np.ndarray] = None,
        tangent_hint: Optional[np.ndarray] = None,
    ) -> None:
        p_stage_arr = self.clamp_stage(np.asarray(p_stage, dtype=float), "move_to_stage")
        self.write_move(p_stage_arr, b, c, feed, comment=comment)
        if tip_hint is not None:
            self.cur_tip_xyz = np.asarray(tip_hint, dtype=float).copy()
        if self.cur_motion_phase is None:
            self.cur_motion_phase = "pull"
        self.last_tip_tangent = None if tangent_hint is None else np.asarray(tangent_hint, dtype=float).copy()

    def pressure_preload_before_print(self) -> None:
        if self.emit_extrusion and self.pressure_offset_mm > 0.0 and not self.pressure_charged:
            self.pressure_charged = True
            self.f.write("; pressure preload before print pass\n")
            self.f.write(f"G1 {self.u_axis}{self.u_cmd_actual():.3f} F{self.pressure_advance_feed:.0f}\n")
            if self.preflow_dwell_ms > 0:
                self.f.write(f"G4 P{self.preflow_dwell_ms}\n")

    def pressure_release_after_print(self) -> None:
        if self.emit_extrusion and self.pressure_offset_mm > 0.0 and self.pressure_charged:
            self.pressure_charged = False
            self.f.write("; pressure release after print pass\n")
            self.f.write(f"G1 {self.u_axis}{self.u_cmd_actual():.3f} F{self.pressure_retract_feed:.0f}\n")

    def approach_start(
        self,
        start_tip: np.ndarray,
        start_tangent: np.ndarray,
        travel_lift_z: float,
        approach_side_mm: float,
        safe_approach_z: float,
    ) -> None:
        start_tip = np.asarray(start_tip, dtype=float)
        start_tangent = normalize(np.asarray(start_tangent, dtype=float))
        start_stage, start_b, start_c, start_motion_phase = self.tip_to_stage(start_tip, tangent=start_tangent)
        safe_stage = start_stage.copy()
        safe_stage[2] = float(safe_approach_z)

        if self.cur_stage_xyz is not None:
            lifted_stage = np.asarray(self.cur_stage_xyz, dtype=float).copy()
            lifted_stage[2] = float(safe_approach_z)
            self.move_to_stage(
                lifted_stage,
                self.cur_b,
                self.cur_c,
                self.approach_feed,
                comment=f"lift to safe bath-exit height Z{float(safe_approach_z):.3f}",
            )

        self.move_to_stage(
            safe_stage,
            start_b,
            start_c,
            self.travel_feed,
            comment="set B/C and XY at safe approach height above wave start",
        )
        near_start_stage = start_stage.copy()
        near_start_stage[2] = min(float(safe_approach_z), float(start_stage[2]) + max(0.0, float(travel_lift_z)))
        if near_start_stage[2] > float(start_stage[2]) + 1e-9:
            self.move_to_stage(
                near_start_stage,
                start_b,
                start_c,
                self.travel_feed,
                comment="move down in Z toward wave start",
            )
        self.move_to_stage(
            start_stage,
            start_b,
            start_c,
            self.fine_approach_feed,
            comment="move down in Z directly to wave start",
            tip_hint=start_tip,
            tangent_hint=start_tangent,
        )
        self.cur_motion_phase = start_motion_phase

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

        self.pressure_preload_before_print()

        last_tip = np.asarray(points[0], dtype=float).copy()
        self.cur_tip_xyz = last_tip.copy()
        self.last_tip_tangent = np.asarray(tangents[0], dtype=float).copy()

        for i in range(1, len(points)):
            p0 = np.asarray(points[i - 1], dtype=float)
            p1 = np.asarray(points[i], dtype=float)
            t0 = np.asarray(tangents[i - 1], dtype=float)
            t1 = np.asarray(tangents[i], dtype=float)

            for s in range(1, self.edge_samples + 1):
                u = s / float(self.edge_samples)
                p_tip = p0 + u * (p1 - p0)
                tangent = normalize((1.0 - u) * t0 + u * t1)
                forced_b: Optional[float] = None
                forced_phase: Optional[str] = None
                if self.phase_transition_via_extrema and self.write_mode == "calibrated" and self.cal is not None:
                    pull_phase, release_phase = self._phase_names()
                    b_pull = self._phase_b_for_tangent(tangent, pull_phase)
                    b_release = self._phase_b_for_tangent(tangent, release_phase)
                    current_phase = str(self.cur_motion_phase or pull_phase)
                    if current_phase == pull_phase:
                        if b_pull <= float(self.cur_b) + 1e-9:
                            forced_b, forced_phase = float(b_pull), pull_phase
                        else:
                            self.transition_phase_at_current_tip(release_phase)
                            forced_b, forced_phase = float(b_release), release_phase
                    else:
                        if b_release >= float(self.cur_b) - 1e-9:
                            forced_b, forced_phase = float(b_release), release_phase
                        else:
                            self.transition_phase_at_current_tip(pull_phase)
                            forced_b, forced_phase = float(b_pull), pull_phase

                p_stage, b, c, motion_phase = self.tip_to_stage(
                    p_tip,
                    tangent=tangent,
                    forced_b=forced_b,
                    forced_phase=forced_phase,
                )

                u_value = None
                if self.emit_extrusion:
                    tip_seg_len = float(np.linalg.norm(p_tip - last_tip))
                    self.u_material_abs += self.extrusion_per_mm * tip_seg_len
                    u_value = self.u_cmd_actual()

                self.write_move(p_stage, b, c, self.print_feed, comment=None, u_value=u_value)
                self.cur_tip_xyz = p_tip.copy()
                self.cur_motion_phase = motion_phase
                self.last_tip_tangent = tangent.copy()
                last_tip = p_tip.copy()

        self.pressure_release_after_print()
        self.f.write("; WAVE_WRITE_END\n")

    def print_x_only_b_cycle(self, points: np.ndarray, cycles: float) -> None:
        if len(points) < 2:
            return
        if self.cur_stage_xyz is None:
            raise RuntimeError("X-only B-cycle mode requires the start approach to complete first.")

        self.f.write(
            "; X_ONLY_B_CYCLE_START "
            f"point_count={len(points)} "
            f"tip_start_x={float(points[0, 0]):.6f} tip_start_y={float(points[0, 1]):.6f} tip_start_z={float(points[0, 2]):.6f} "
            f"tip_end_x={float(points[-1, 0]):.6f} tip_end_y={float(points[-1, 1]):.6f} tip_end_z={float(points[-1, 2]):.6f} "
            f"cycles={float(cycles):.6f} "
            "stage_yz_constant=1 b_tip_angle_profile=90_to_0_to_90\n"
        )

        self.pressure_preload_before_print()

        stage_start = np.asarray(self.cur_stage_xyz, dtype=float).copy()
        stage_y = float(stage_start[1])
        stage_z = float(stage_start[2])
        stage_x0 = float(stage_start[0])
        span_x = float(points[-1, 0] - points[0, 0])

        last_stage_xyz = stage_start.copy()
        last_tip = np.asarray(points[0], dtype=float).copy()

        pull_phase, release_phase = self._phase_names()
        for i in range(1, len(points)):
            u = i / float(len(points) - 1)
            cycle_pos = (float(cycles) * u) % 1.0
            if cycle_pos <= 0.5:
                frac = cycle_pos / 0.5
                tip_angle_target = 90.0 * (1.0 - frac)
                motion_phase = release_phase
            else:
                frac = (cycle_pos - 0.5) / 0.5
                tip_angle_target = 90.0 * frac
                motion_phase = pull_phase

            b = solve_b_for_target_tip_angle(
                self.cal,
                float(np.clip(tip_angle_target + self.b_angle_bias_deg, 0.0, 180.0)),
                search_samples=self.bc_solve_samples,
                motion_phase=motion_phase,
            ) if self.write_mode == "calibrated" and self.cal is not None else float(np.clip(tip_angle_target, 0.0, 180.0))

            stage_x = stage_x0 + span_x * u
            p_stage = np.array([stage_x, stage_y, stage_z], dtype=float)

            u_value = None
            if self.emit_extrusion:
                seg_len = float(np.linalg.norm(p_stage - last_stage_xyz))
                self.u_material_abs += self.extrusion_per_mm * seg_len
                u_value = self.u_cmd_actual()

            self.write_move(
                self.clamp_stage(p_stage, "x_only_b_cycle"),
                float(b),
                float(self.c_deg),
                self.print_feed,
                comment=None,
                u_value=u_value,
            )
            self.cur_tip_xyz = np.asarray(points[i], dtype=float).copy()
            self.cur_motion_phase = str(motion_phase)
            self.last_tip_tangent = np.array([1.0, 0.0, 0.0], dtype=float)
            last_stage_xyz = p_stage.copy()
            last_tip = np.asarray(points[i], dtype=float).copy()

        self.pressure_release_after_print()
        self.f.write("; X_ONLY_B_CYCLE_END\n")

    def finish(self, travel_lift_z: float, safe_approach_z: float) -> None:
        if self.cur_tip_xyz is None or self.last_tip_tangent is None:
            return
        if self.cur_stage_xyz is None:
            return
        safe_stage = np.asarray(self.cur_stage_xyz, dtype=float).copy()
        safe_stage[2] = float(safe_approach_z)
        self.move_to_stage(
            safe_stage,
            self.cur_b,
            self.cur_c,
            self.approach_feed,
            comment=f"move out of bath to safe Z{float(safe_approach_z):.3f}",
        )


# ---------------- Top-level generation ----------------
def write_sine_wave_gcode(
    out: str,
    calibration: Optional[str],
    write_mode: str,
    orientation_mode: str,
    use_cubic_phase_models: bool,
    phase_transition_via_extrema: bool,
    x_only_b_cycle_mode: bool,
    y_offplane_sign: float,
    x_start: float,
    x_end: float,
    y: float,
    z_min: float,
    z_max: float,
    z_baseline: float,
    z_amplitude: float,
    cycles: float,
    phase_deg: float,
    lead_in: float,
    lead_out: float,
    points: int,
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
    safe_approach_z: float,
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
    if orientation_mode not in {"tangent", "fixed"}:
        raise ValueError("--orientation-mode must be tangent or fixed")
    if write_mode == "calibrated" and not calibration:
        raise ValueError("--calibration is required when --write-mode calibrated")

    cal: Optional[Calibration]
    if write_mode == "calibrated":
        cal = load_calibration(str(calibration))
        cal.offplane_y_sign = float(y_offplane_sign)
        if bool(use_cubic_phase_models):
            cal = calibration_with_phase_cubic_override(cal)
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

    if bool(x_only_b_cycle_mode):
        pts = build_x_only_b_cycle_points(
            x_start=x_start,
            x_end=x_end,
            y=y,
            z_baseline=z_baseline,
            points=points,
        )
        tangents = np.tile(np.array([[1.0, 0.0, 0.0]], dtype=float), (len(pts), 1))
    else:
        pts = build_sine_wave_points(
            x_start=x_start,
            x_end=x_end,
            y=y,
            z_baseline=z_baseline,
            z_amplitude=z_amplitude,
            cycles=cycles,
            phase_deg=phase_deg,
            lead_in=lead_in,
            lead_out=lead_out,
            points=points,
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
        phase_transition_via_extrema=phase_transition_via_extrema,
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

        fh.write("; Plain sine wave in the XZ plane\n")
        fh.write("; Generated by sine_wave_xz_generator.py\n")
        fh.write(f"; write_mode={write_mode} orientation_mode={orientation_mode}\n")
        fh.write(f"; x_only_b_cycle_mode={int(bool(x_only_b_cycle_mode))}\n")
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
            f"lead_in={float(lead_in):.3f} "
            f"lead_out={float(lead_out):.3f} "
            "gaussian_modulation=0 endpoint_taper=0\n"
        )
        fh.write("G21\n")
        fh.write("G90\n")
        fh.write(f"; B-angle convention: 0=up, 90=horizontal, 180=down\n")
        fh.write(f"; C held constant at {float(c_deg):.3f} deg by default\n")

        start_tip = pts[0]
        start_tangent = tangents[0]
        writer.approach_start(
            start_tip=start_tip,
            start_tangent=start_tangent,
            travel_lift_z=travel_lift_z,
            approach_side_mm=approach_side_mm,
            safe_approach_z=safe_approach_z,
        )
        if bool(x_only_b_cycle_mode):
            writer.print_x_only_b_cycle(pts, cycles=cycles)
        else:
            writer.print_polyline(pts, tangents)
        writer.finish(travel_lift_z=travel_lift_z, safe_approach_z=safe_approach_z)

        fh.write("; End of file\n")

    tip_b = np.array([desired_physical_b_angle_from_tangent(t) for t in tangents], dtype=float)
    if bool(x_only_b_cycle_mode):
        phase_u = np.linspace(0.0, float(cycles), len(pts))
        cycle_pos = np.mod(phase_u, 1.0)
        tip_b = np.where(cycle_pos <= 0.5, 90.0 * (1.0 - cycle_pos / 0.5), 90.0 * ((cycle_pos - 0.5) / 0.5))
        if write_mode == "calibrated":
            assert cal is not None
            pull_phase = _resolve_phase_name(cal, "pull")
            release_phase = _resolve_phase_name(cal, "release")
            b_used = np.array(
                [
                    solve_b_for_target_tip_angle(
                        cal,
                        float(np.clip(tb + b_angle_bias_deg, 0.0, 180.0)),
                        search_samples=bc_solve_samples,
                        motion_phase=(release_phase if (np.mod(float(cycles) * (i / float(max(1, len(pts) - 1))), 1.0) <= 0.5) else pull_phase),
                    )
                    for i, tb in enumerate(tip_b)
                ],
                dtype=float,
            )
        else:
            b_used = np.clip(tip_b + float(b_angle_bias_deg), 0.0, 180.0)
    elif orientation_mode == "fixed":
        b_used = np.full_like(tip_b, float(fixed_b), dtype=float)
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
        "use_cubic_phase_models": bool(use_cubic_phase_models),
        "phase_transition_via_extrema": bool(phase_transition_via_extrema),
        "x_only_b_cycle_mode": bool(x_only_b_cycle_mode),
        "tip_x_range": (float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))),
        "tip_y_range": (float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))),
        "tip_z_range": (float(np.min(pts[:, 2])), float(np.max(pts[:, 2]))),
        "tip_b_target_range_deg": (float(np.min(tip_b)), float(np.max(tip_b))),
        "b_command_range_deg": (float(np.min(b_used)), float(np.max(b_used))),
        "c_command_deg": float(c_deg if orientation_mode == "tangent" else fixed_c),
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
            "Generate calibrated or Cartesian G-code for a plain sine wave in the XZ plane. "
            "Default tip-space box is X=[50,110], Y=52, Z=[-160,-120]. "
            "In tangent mode, B follows the local path tangent with 0 deg=up, 90 deg=horizontal, 180 deg=down; "
            "C is held constant at 180 deg by default."
        )
    )
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code file.")
    ap.add_argument("--calibration", default=None, help="Calibration JSON. Required for --write-mode calibrated.")
    ap.add_argument("--write-mode", choices=["calibrated", "cartesian"], default=DEFAULT_WRITE_MODE)
    ap.add_argument("--orientation-mode", choices=["tangent", "fixed"], default=DEFAULT_ORIENTATION_MODE)
    ap.add_argument("--use-cubic-phase-models", action="store_true", help="Use phase-specific cubic polynomial models for X-offset/r, Z, and tip angle instead of the phase PCHIP fits.")
    ap.add_argument("--phase-transition-via-extrema", action="store_true", default=DEFAULT_PHASE_TRANSITION_VIA_EXTREMA, help="When changing between pull and release models, first run B to the shared phase boundary while holding the current tip point, then continue under the new phase model.")
    ap.add_argument("--x-only-b-cycle-mode", action="store_true", default=DEFAULT_X_ONLY_B_CYCLE_MODE, help="Move only in stage X during the print pass, keep stage Z fixed, and cycle B through 90->0->90 for each requested cycle.")
    ap.add_argument("--y-offplane-sign", type=float, default=DEFAULT_Y_OFFPLANE_SIGN, help="Multiplier applied to the calibration off-plane Y term in calibrated mode. Use -1 to flip the sign.")

    ap.add_argument("--x-start", type=float, default=DEFAULT_X_START)
    ap.add_argument("--x-end", type=float, default=DEFAULT_X_END)
    ap.add_argument("--y", type=float, default=DEFAULT_Y)
    ap.add_argument("--z-min", type=float, default=DEFAULT_Z_MIN, help="Requested lower tip-space Z bound. The generated path must stay within this range.")
    ap.add_argument("--z-max", type=float, default=DEFAULT_Z_MAX, help="Requested upper tip-space Z bound. The generated path must stay within this range.")
    ap.add_argument("--z-baseline", type=float, default=DEFAULT_Z_BASELINE, help="Baseline Z around which the wave oscillates.")
    ap.add_argument("--z-amplitude", type=float, default=DEFAULT_Z_AMPLITUDE, help="Peak sine amplitude in Z units.")
    ap.add_argument("--cycles", type=float, default=DEFAULT_CYCLES, help="Number of sine cycles across the active wave interval.")
    ap.add_argument("--phase-deg", type=float, default=DEFAULT_PHASE_DEG, help="Phase offset in degrees.")
    ap.add_argument("--lead-in", type=float, default=DEFAULT_LEAD_IN, help="Horizontal lead-in segment length at the start of the wave.")
    ap.add_argument("--lead-out", type=float, default=DEFAULT_LEAD_OUT, help="Horizontal lead-out segment length at the end of the wave.")
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
    ap.add_argument("--safe-approach-z", type=float, default=DEFAULT_SAFE_APPROACH_Z, help="Absolute safe Z used to enter and exit the bath.")
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
    summary = write_sine_wave_gcode(**vars(args))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
