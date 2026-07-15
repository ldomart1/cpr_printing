#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code for an oscillating triangle-wave / zig-zag
path in the XZ plane.

The requested tip path is made from straight line segments only.  At each
triangle-wave vertex, the script:

  1) closes the pressure valve,
  2) keeps the desired tip point fixed,
  3) changes the B/C attack angle while compensating XYZ by tip calibration,
  4) reopens pressure before printing the next straight segment.

Highlights
----------
- Desired tip path lies in the XZ plane at constant Y.
- Default tip-space box:
    X in [60, 115]
    Z in [-155, -125]
    Y = 52
- Segment-following orientation:
    * B = 0 deg   -> tip points straight up (+Z)
    * B = 90 deg  -> tip is horizontal
    * B = 180 deg -> tip points straight down (-Z)
- Default C is held at 180 deg so the tool stays in the XZ plane.
- Supports both:
    1) calibrated tip tracking, using
         stage_xyz = desired_tip_xyz - offset_tip(B, C)
    2) direct Cartesian stage motion

This keeps the same calibration model loading, `predict_tip_offset_xyz(...)`,
`stage_xyz_for_tip(...)`, `solve_b_for_target_tip_angle(...)`, and B-angle
convention as the supplied Gaussian-wave reference script, but replaces the
curved sine/Gaussian path with a straight-segment triangle wave.
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
DEFAULT_OUT = "gcode_generation/triangle_wave_xz.gcode"

# Tip-space path placement
DEFAULT_X_START = 60.0
DEFAULT_X_END = 115.0
DEFAULT_Y = 52.0
DEFAULT_Z_MIN = -155.0
DEFAULT_Z_MAX = -125.0

# Triangle-wave shape
DEFAULT_HALF_CYCLES = 6
DEFAULT_CYCLES = DEFAULT_HALF_CYCLES / 2.0
DEFAULT_START_Z = "low"  # low or high
DEFAULT_LEAD_IN = 0.0
DEFAULT_LEAD_OUT = 0.0
DEFAULT_POINT_MERGE_TOL = 1e-9
DEFAULT_MIN_TANGENT_XY = 1e-9

# Orientation
DEFAULT_WRITE_MODE = "calibrated"
DEFAULT_ORIENTATION_MODE = "segment-step"
DEFAULT_FIXED_B = 90.0
DEFAULT_FIXED_C = 180.0
DEFAULT_C_DEG = 180.0
DEFAULT_B_ANGLE_BIAS_DEG = 0.0
DEFAULT_BC_SOLVE_SAMPLES = 5001
DEFAULT_Y_OFFPLANE_SIGN = -1.0
DEFAULT_Y_OFFPLANE_FIT_MODEL = "avg_pchip"

# Motion
DEFAULT_TRAVEL_FEED = 1000.0
DEFAULT_APPROACH_FEED = 400.0
DEFAULT_FINE_APPROACH_FEED = 200.0
DEFAULT_PRINT_FEED = 300.0
DEFAULT_TRACK_FEED = 250.0
DEFAULT_TRAVEL_LIFT_Z = 8.0
DEFAULT_APPROACH_SIDE_MM = 4.0
DEFAULT_EDGE_SAMPLES = 1  # 1 emits one printed G1 per triangle-wave side.
DEFAULT_TRACK_B_STEP_DEG = 2.0
DEFAULT_TRACK_MIN_STEPS = 8

# Pressure actuation
DEFAULT_EMIT_EXTRUSION = True
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
    c_180_deg: float

    r_model: Optional[Dict[str, Any]] = None
    z_model: Optional[Dict[str, Any]] = None
    y_off_model: Optional[Dict[str, Any]] = None
    y_off_extrap_model: Optional[Dict[str, Any]] = None
    tip_angle_model: Optional[Dict[str, Any]] = None
    phase_models: Optional[Dict[str, Dict[str, Any]]] = None
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


def segment_tangent(points: np.ndarray, i0: int, i1: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    return normalize(pts[int(i1)] - pts[int(i0)])


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


def infer_motion_phase_for_b(
    default_phase: str,
    prev_b: Optional[float],
    curr_b: float,
    next_b: Optional[float],
) -> str:
    deltas: List[float] = []
    if prev_b is not None:
        deltas.append(float(curr_b) - float(prev_b))
    if next_b is not None:
        deltas.append(float(next_b) - float(curr_b))

    for delta in deltas:
        if abs(delta) <= 1e-9:
            continue
        return "pull" if delta < 0.0 else "release"

    return str(default_phase)


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

    phase_models_raw = data.get("fit_models_by_phase", {}) or {}
    phase_models: Dict[str, Dict[str, Any]] = {}
    if isinstance(phase_models_raw, dict):
        for raw_phase_name, models in phase_models_raw.items():
            phase_name = _normalize_motion_phase_name(raw_phase_name)
            if phase_name is None or not isinstance(models, dict):
                continue
            phase_models[phase_name] = dict(models)
    active_phase_models = phase_models.get(active_phase) if isinstance(phase_models, dict) else None
    if not isinstance(active_phase_models, dict):
        active_phase_models = fit_models

    r_selector = requested_r_fit_model or selected_fit_model
    z_selector = requested_z_fit_model or selected_fit_model
    r_model = _select_named_model(active_phase_models, "r", r_selector)
    z_model = _select_named_model(active_phase_models, "z", z_selector)

    requested_offplane_fit_model = None if requested_offplane_fit_model is None else str(requested_offplane_fit_model).strip().lower()
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
        phase_models=phase_models,
        selected_fit_model=selected_fit_model,
        selected_offplane_fit_model=selected_offplane_fit_model,
        requested_r_fit_model=requested_r_fit_model,
        requested_z_fit_model=requested_z_fit_model,
        requested_offplane_fit_model=requested_offplane_fit_model,
        resolved_r_fit_model=r_selector if r_model is not None else (selected_fit_model if selected_fit_model else None),
        resolved_z_fit_model=z_selector if z_model is not None else (selected_fit_model if selected_fit_model else None),
        resolved_offplane_fit_model=resolved_offplane_fit_model,
        active_phase=active_phase,
        b_min=b_min,
        b_max=b_max,
        x_axis=str(duet_map.get("horizontal_axis") or motor_setup.get("horizontal_axis") or "X"),
        y_axis=str(duet_map.get("depth_axis") or motor_setup.get("depth_axis") or "Y"),
        z_axis=str(duet_map.get("vertical_axis") or motor_setup.get("vertical_axis") or "Z"),
        b_axis=str(duet_map.get("pull_axis") or motor_setup.get("b_motor_axis") or "B"),
        c_axis=str(duet_map.get("rotation_axis") or motor_setup.get("rotation_axis") or "C"),
        c_180_deg=float(motor_setup.get("rotation_axis_180_deg", 180.0)),
    )


def _select_phase_fit_model(cal: Calibration, model_name: str, motion_phase: Optional[str] = None) -> Optional[Dict[str, Any]]:
    phase_name = _normalize_motion_phase_name(motion_phase)
    if phase_name and cal.phase_models and phase_name in cal.phase_models:
        spec = _normalize_model_spec(cal.phase_models[phase_name].get(model_name))
        if spec is not None:
            return spec
    fallback = getattr(cal, f"{model_name}_model", None)
    return None if fallback is None else _normalize_model_spec(fallback)


def eval_r(cal: Calibration, b: Any, motion_phase: Optional[str] = None) -> np.ndarray:
    model = _select_phase_fit_model(cal, "r", motion_phase=motion_phase)
    if model is not None:
        return eval_model_spec(model, b)
    return poly_eval(cal.pr, b)


def eval_z(cal: Calibration, b: Any, motion_phase: Optional[str] = None) -> np.ndarray:
    model = _select_phase_fit_model(cal, "z", motion_phase=motion_phase)
    if model is not None:
        return eval_model_spec(model, b)
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


def eval_tip_angle_deg(cal: Calibration, b: Any, motion_phase: Optional[str] = None) -> np.ndarray:
    model = _select_phase_fit_model(cal, "tip_angle", motion_phase=motion_phase)
    if model is not None:
        return eval_model_spec(model, b)
    if cal.pa is None:
        raise ValueError("Calibration is missing tip_angle_coeffs.")
    return poly_eval(cal.pa, b)


def predict_tip_offset_xyz(cal: Calibration, b: float, c_deg: float, motion_phase: Optional[str] = None) -> np.ndarray:
    r = float(eval_r(cal, b, motion_phase=motion_phase))
    z = float(eval_z(cal, b, motion_phase=motion_phase))
    y_off = float(eval_offplane_y(cal, b))
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
    b_lo, b_hi = float(cal.b_min), float(cal.b_max)
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


# ---------------- Triangle-wave construction ----------------
def resolve_half_cycles(half_cycles: Optional[int], cycles: Optional[float]) -> int:
    if half_cycles is not None:
        out = int(half_cycles)
    else:
        c = DEFAULT_CYCLES if cycles is None else float(cycles)
        out = int(round(2.0 * c))
    if out < 1:
        raise ValueError("Triangle wave requires at least one half-cycle / straight diagonal segment.")
    return out


def build_triangle_wave_points(
    x_start: float,
    x_end: float,
    y: float,
    z_min: float,
    z_max: float,
    half_cycles: Optional[int],
    cycles: Optional[float],
    start_z: str,
    lead_in: float,
    lead_out: float,
) -> np.ndarray:
    x0 = float(x_start)
    x1 = float(x_end)
    if not x1 > x0:
        raise ValueError("--x-end must be greater than --x-start")
    if not float(z_max) > float(z_min):
        raise ValueError("--z-max must be greater than --z-min")
    if float(lead_in) < 0.0 or float(lead_out) < 0.0:
        raise ValueError("--lead-in and --lead-out must be >= 0")
    if float(lead_in + lead_out) >= (x1 - x0):
        raise ValueError("--lead-in + --lead-out must be smaller than the total X span")

    n_half = resolve_half_cycles(half_cycles=half_cycles, cycles=cycles)
    active_x0 = x0 + float(lead_in)
    active_x1 = x1 - float(lead_out)
    if not active_x1 > active_x0:
        raise ValueError("Active triangle-wave span must be positive after lead-in / lead-out.")

    start_mode = str(start_z).strip().lower()
    if start_mode not in {"low", "high"}:
        raise ValueError("--start-z must be 'low' or 'high'")

    z_low = float(z_min)
    z_high = float(z_max)
    zs: List[float] = []
    for i in range(n_half + 1):
        is_even = (i % 2) == 0
        if start_mode == "low":
            zs.append(z_low if is_even else z_high)
        else:
            zs.append(z_high if is_even else z_low)

    xs = np.linspace(active_x0, active_x1, n_half + 1)
    ys = np.full_like(xs, float(y), dtype=float)
    active_pts = np.column_stack([xs, ys, np.asarray(zs, dtype=float)])

    parts: List[np.ndarray] = []
    if float(lead_in) > 0.0:
        parts.append(np.array([[x0, float(y), float(active_pts[0, 2])]], dtype=float))
    parts.append(active_pts)
    if float(lead_out) > 0.0:
        parts.append(np.array([[x1, float(y), float(active_pts[-1, 2])]], dtype=float))

    return deduplicate_polyline_points(np.vstack(parts))


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
        track_feed: float,
        edge_samples: int,
        track_b_step_deg: float,
        track_min_steps: int,
        emit_extrusion: bool,
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
        self.track_feed = float(track_feed)
        self.edge_samples = max(1, int(edge_samples))
        self.track_b_step_deg = max(0.1, float(track_b_step_deg))
        self.track_min_steps = max(1, int(track_min_steps))
        self.emit_extrusion = bool(emit_extrusion)
        self.preflow_dwell_ms = int(preflow_dwell_ms)

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

    def clamp_stage(self, p_stage: np.ndarray, context: str) -> np.ndarray:
        x = float(np.clip(p_stage[0], self.bbox["x_min"], self.bbox["x_max"]))
        y = float(np.clip(p_stage[1], self.bbox["y_min"], self.bbox["y_max"]))
        z = float(np.clip(p_stage[2], self.bbox["z_min"], self.bbox["z_max"]))
        if abs(x - float(p_stage[0])) > 1e-12 or abs(y - float(p_stage[1])) > 1e-12 or abs(z - float(p_stage[2])) > 1e-12:
            self.warnings.append(f"WARNING: {context} stage point clamped to bbox.")
        return np.array([x, y, z], dtype=float)

    def bc_for_tangent(self, tangent: np.ndarray, motion_phase: Optional[str] = None) -> Tuple[float, float]:
        if self.orientation_mode == "fixed":
            return float(self.fixed_b), float(self.fixed_c)

        target_b = desired_physical_b_angle_from_tangent(tangent) + float(self.b_angle_bias_deg)
        target_b = float(np.clip(target_b, 0.0, 180.0))

        if self.write_mode == "calibrated":
            assert self.cal is not None
            b = solve_b_for_target_tip_angle(
                self.cal,
                target_b,
                search_samples=self.bc_solve_samples,
                motion_phase=motion_phase,
            )
        else:
            b = target_b

        return float(b), float(self.c_deg)

    def tip_to_stage_with_bc(self, p_tip: np.ndarray, b: float, c: float, motion_phase: Optional[str] = None) -> np.ndarray:
        if self.write_mode == "calibrated":
            assert self.cal is not None
            p_stage = stage_xyz_for_tip(
                self.cal,
                np.asarray(p_tip, dtype=float),
                float(b),
                float(c),
                motion_phase=motion_phase,
            )
        else:
            p_stage = np.asarray(p_tip, dtype=float)
        return self.clamp_stage(np.asarray(p_stage, dtype=float), "tip_to_stage_with_bc")

    def tip_to_stage(self, p_tip: np.ndarray, tangent: Optional[np.ndarray], motion_phase: Optional[str] = None) -> Tuple[np.ndarray, float, float]:
        tangent_arr = np.array([1.0, 0.0, 0.0], dtype=float) if tangent is None else normalize(np.asarray(tangent, dtype=float))
        b, c = self.bc_for_tangent(tangent_arr, motion_phase=motion_phase)
        return self.tip_to_stage_with_bc(np.asarray(p_tip, dtype=float), b, c, motion_phase=motion_phase), float(b), float(c)

    def write_move(self, p_stage: np.ndarray, b: float, c: float, feed: float, comment: Optional[str] = None) -> None:
        if comment:
            self.f.write(f"; {comment}\n")
        axes: List[Tuple[str, float]] = [
            (self.x_axis, float(p_stage[0])),
            (self.y_axis, float(p_stage[1])),
            (self.z_axis, float(p_stage[2])),
            (self.b_axis, float(b)),
            (self.c_axis, float(c)),
        ]
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

    def move_to_tip(self, p_tip: np.ndarray, tangent: Optional[np.ndarray], feed: float, comment: Optional[str] = None, motion_phase: Optional[str] = None) -> None:
        p_stage, b, c = self.tip_to_stage(np.asarray(p_tip, dtype=float), tangent=tangent, motion_phase=motion_phase)
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
        motion_phase: Optional[str] = None,
    ) -> None:
        p_stage = self.tip_to_stage_with_bc(np.asarray(p_tip, dtype=float), float(b), float(c), motion_phase=motion_phase)
        self.write_move(p_stage, float(b), float(c), feed, comment=comment)
        self.cur_tip_xyz = np.asarray(p_tip, dtype=float).copy()
        self.last_tip_tangent = None if tangent is None else np.asarray(tangent, dtype=float).copy()

    def pressure_preload_before_print(self) -> None:
        if self.emit_extrusion and not self.pressure_charged:
            self.pressure_charged = True
            self.f.write("; open pressure solenoid before printed triangle segment\n")
            self.f.write("M42 P0 S1\n")
            if self.preflow_dwell_ms > 0:
                self.f.write(f"G4 P{self.preflow_dwell_ms}\n")

    def pressure_release_after_print(self) -> None:
        if self.emit_extrusion and self.pressure_charged:
            self.pressure_charged = False
            self.f.write("; close pressure solenoid; subsequent point-tracking move has no extrusion\n")
            self.f.write("M42 P0 S0\n")

    def tracked_tip_hold_bc_transition(
        self,
        p_tip: np.ndarray,
        b_start: float,
        b_end: float,
        c_start: float,
        c_end: float,
        tangent: Optional[np.ndarray],
        feed: float,
        comment: Optional[str] = None,
        motion_phase: Optional[str] = None,
    ) -> None:
        """Change attack angle while holding the desired tip point fixed.

        Pressure is explicitly closed before this move.  In calibrated mode each
        intermediate B/C sample recomputes XYZ so the tip remains at p_tip.
        """
        self.pressure_release_after_print()

        p_tip_arr = np.asarray(p_tip, dtype=float)
        tangent_arr = None if tangent is None else np.asarray(tangent, dtype=float).copy()
        db = abs(float(b_end) - float(b_start))
        dc = abs(float(c_end) - float(c_start))
        n_steps = max(self.track_min_steps, int(math.ceil(max(db / self.track_b_step_deg, dc / 2.0))))

        if comment:
            self.f.write(f"; {comment}\n")
        self.f.write("; pressure is OFF during this compensated point-tracking attack-angle change\n")

        for i in range(1, n_steps + 1):
            u = i / float(n_steps)
            s = u * u * (3.0 - 2.0 * u)
            b = (1.0 - s) * float(b_start) + s * float(b_end)
            c = (1.0 - s) * float(c_start) + s * float(c_end)
            p_stage = self.tip_to_stage_with_bc(p_tip_arr, b, c, motion_phase=motion_phase)
            self.write_move(p_stage, b, c, feed, comment=None)
            self.cur_tip_xyz = p_tip_arr.copy()
            self.last_tip_tangent = None if tangent_arr is None else tangent_arr.copy()

    def approach_start(self, start_tip: np.ndarray, start_tangent: np.ndarray, travel_lift_z: float, approach_side_mm: float) -> None:
        start_tip = np.asarray(start_tip, dtype=float)
        start_tangent = normalize(np.asarray(start_tangent, dtype=float))
        side = side_vector_from_tangent(start_tangent, fallback=self.last_tip_tangent)
        far_tip = start_tip - side * float(approach_side_mm) + np.array([0.0, 0.0, float(travel_lift_z)], dtype=float)
        near_tip = start_tip - side * (0.5 * float(approach_side_mm))

        if self.cur_tip_xyz is None:
            self.move_to_tip(far_tip, tangent=start_tangent, feed=self.travel_feed, comment="travel above and to the side of triangle-wave start")
            self.move_to_tip(near_tip, tangent=start_tangent, feed=self.approach_feed, comment="approach near triangle-wave start")
            self.move_to_tip(start_tip, tangent=start_tangent, feed=self.fine_approach_feed, comment="fine approach to triangle-wave start")
            return

        retreat_side = side_vector_from_tangent(self.last_tip_tangent if self.last_tip_tangent is not None else start_tangent, fallback=side)
        retreat_tip = np.asarray(self.cur_tip_xyz, dtype=float) + retreat_side * float(approach_side_mm) + np.array([0.0, 0.0, float(travel_lift_z)], dtype=float)
        retreat_tangent = self.last_tip_tangent if self.last_tip_tangent is not None else start_tangent
        self.move_to_tip(retreat_tip, tangent=retreat_tangent, feed=self.approach_feed, comment="retreat from previous end")
        self.move_to_tip(far_tip, tangent=start_tangent, feed=self.travel_feed, comment="travel above and to the side of triangle-wave start")
        self.move_to_tip(near_tip, tangent=start_tangent, feed=self.approach_feed, comment="approach near triangle-wave start")
        self.move_to_tip(start_tip, tangent=start_tangent, feed=self.fine_approach_feed, comment="fine approach to triangle-wave start")

    def compute_segment_bcs_and_phases(self, points: np.ndarray) -> Tuple[List[np.ndarray], List[float], List[float], List[str], List[float]]:
        pts = np.asarray(points, dtype=float)
        seg_tangents = [segment_tangent(pts, i, i + 1) for i in range(len(pts) - 1)]
        target_physical_bs = [
            float(np.clip(desired_physical_b_angle_from_tangent(t) + self.b_angle_bias_deg, 0.0, 180.0))
            for t in seg_tangents
        ]
        default_phase = self.cal.active_phase if self.cal is not None else "pull"
        phases = [
            infer_motion_phase_for_b(
                default_phase=default_phase,
                prev_b=(None if i == 0 else target_physical_bs[i - 1]),
                curr_b=target_physical_bs[i],
                next_b=(None if i + 1 >= len(target_physical_bs) else target_physical_bs[i + 1]),
            )
            for i in range(len(target_physical_bs))
        ]
        bcs = [self.bc_for_tangent(seg_tangents[i], motion_phase=phases[i]) for i in range(len(seg_tangents))]
        b_cmds = [float(b) for b, _ in bcs]
        c_cmds = [float(c) for _, c in bcs]
        return seg_tangents, b_cmds, c_cmds, phases, target_physical_bs

    def print_one_straight_segment(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        tangent: np.ndarray,
        b: float,
        c: float,
        motion_phase: Optional[str],
        seg_idx: int,
    ) -> None:
        """Print one triangle side using fixed B/C and straight G1 moves only."""
        for s in range(1, self.edge_samples + 1):
            u = s / float(self.edge_samples)
            p_tip = np.asarray(p0, dtype=float) + u * (np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float))
            comment = f"print straight triangle segment {seg_idx + 1}" if s == 1 else None
            self.move_to_tip_with_bc(
                p_tip,
                b=b,
                c=c,
                feed=self.print_feed,
                tangent=tangent,
                comment=comment,
                motion_phase=motion_phase,
            )

    def print_polyline(self, points: np.ndarray) -> None:
        pts = np.asarray(points, dtype=float)
        if len(pts) < 2:
            return

        seg_tangents, b_cmds, c_cmds, phases, target_physical_bs = self.compute_segment_bcs_and_phases(pts)

        self.f.write(
            "; TRIANGLE_WAVE_WRITE_START "
            f"point_count={len(pts)} "
            f"segment_count={len(pts) - 1} "
            f"tip_start_x={float(pts[0, 0]):.6f} tip_start_y={float(pts[0, 1]):.6f} tip_start_z={float(pts[0, 2]):.6f} "
            f"tip_end_x={float(pts[-1, 0]):.6f} tip_end_y={float(pts[-1, 1]):.6f} tip_end_z={float(pts[-1, 2]):.6f} "
            "tip_angle_convention=0_posZ_90_horizontal_180_negZ\n"
        )
        self.f.write("; Each printed side is a straight G1 line with fixed B/C.\n")
        self.f.write("; At vertices, pressure is closed and B/C changes while the tip point is held fixed.\n")

        self.cur_tip_xyz = np.asarray(pts[0], dtype=float).copy()
        self.last_tip_tangent = np.asarray(seg_tangents[0], dtype=float).copy()

        # Make sure the first printed side starts at the correct attack angle with no extrusion.
        if abs(float(self.cur_b) - b_cmds[0]) > 1e-9 or abs(float(self.cur_c) - c_cmds[0]) > 1e-9:
            self.tracked_tip_hold_bc_transition(
                pts[0],
                b_start=self.cur_b,
                b_end=b_cmds[0],
                c_start=self.cur_c,
                c_end=c_cmds[0],
                tangent=seg_tangents[0],
                feed=self.track_feed,
                comment="initial compensated attack-angle set at first triangle point",
                motion_phase=phases[0],
            )

        for i in range(len(pts) - 1):
            self.pressure_preload_before_print()
            self.print_one_straight_segment(
                p0=pts[i],
                p1=pts[i + 1],
                tangent=seg_tangents[i],
                b=b_cmds[i],
                c=c_cmds[i],
                motion_phase=phases[i],
                seg_idx=i,
            )
            self.pressure_release_after_print()

            if i + 1 < len(pts) - 1:
                self.tracked_tip_hold_bc_transition(
                    pts[i + 1],
                    b_start=b_cmds[i],
                    b_end=b_cmds[i + 1],
                    c_start=c_cmds[i],
                    c_end=c_cmds[i + 1],
                    tangent=seg_tangents[i + 1],
                    feed=self.track_feed,
                    comment="compensated attack-angle change at triangle vertex",
                    motion_phase=phases[i + 1],
                )

        self.pressure_release_after_print()
        self.f.write("; TRIANGLE_WAVE_WRITE_END\n")

    def finish(self, travel_lift_z: float) -> None:
        if self.cur_tip_xyz is None or self.last_tip_tangent is None:
            return
        end_tip = np.asarray(self.cur_tip_xyz, dtype=float)
        retreat_side = side_vector_from_tangent(self.last_tip_tangent)
        retreat_tip = end_tip + retreat_side * 4.0 + np.array([0.0, 0.0, float(travel_lift_z)], dtype=float)
        self.move_to_tip(retreat_tip, tangent=self.last_tip_tangent, feed=self.approach_feed, comment="retreat after triangle wave")


# ---------------- Top-level generation ----------------
def write_triangle_wave_gcode(
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
    half_cycles: Optional[int],
    cycles: Optional[float],
    start_z: str,
    lead_in: float,
    lead_out: float,
    fixed_b: float,
    fixed_c: float,
    c_deg: float,
    b_angle_bias_deg: float,
    bc_solve_samples: int,
    travel_feed: float,
    approach_feed: float,
    fine_approach_feed: float,
    print_feed: float,
    track_feed: float,
    travel_lift_z: float,
    approach_side_mm: float,
    edge_samples: int,
    track_b_step_deg: float,
    track_min_steps: int,
    emit_extrusion: bool,
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
    if orientation_mode not in {"segment-step", "fixed"}:
        raise ValueError("--orientation-mode must be segment-step or fixed")
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

    pts = build_triangle_wave_points(
        x_start=x_start,
        x_end=x_end,
        y=y,
        z_min=z_min,
        z_max=z_max,
        half_cycles=half_cycles,
        cycles=cycles,
        start_z=start_z,
        lead_in=lead_in,
        lead_out=lead_out,
    )

    # Validate requested tip-space box.
    z_lo = float(np.min(pts[:, 2]))
    z_hi = float(np.max(pts[:, 2]))
    x_lo = float(np.min(pts[:, 0]))
    x_hi = float(np.max(pts[:, 0]))
    if x_lo < float(x_start) - 1e-6 or x_hi > float(x_end) + 1e-6:
        raise ValueError("Generated X coordinates exceed the requested [x-start, x-end] range.")
    if z_lo < float(z_min) - 1e-6 or z_hi > float(z_max) + 1e-6:
        raise ValueError("Generated Z coordinates exceed the requested [z-min, z-max] range.")

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
        track_feed=track_feed,
        edge_samples=edge_samples,
        track_b_step_deg=track_b_step_deg,
        track_min_steps=track_min_steps,
        emit_extrusion=emit_extrusion,
        preflow_dwell_ms=preflow_dwell_ms,
    )

    with out_path.open("w", encoding="utf-8") as fh:
        writer.f = fh

        n_half = len(pts) - 1
        fh.write("; Oscillating triangle wave in the XZ plane\n")
        fh.write("; Generated by triangle_wave_xz_generator.py\n")
        fh.write(f"; write_mode={write_mode} orientation_mode={orientation_mode}\n")
        fh.write(
            "; requested_tip_box "
            f"x=[{float(x_start):.3f},{float(x_end):.3f}] "
            f"y={float(y):.3f} "
            f"z=[{float(z_min):.3f},{float(z_max):.3f}]\n"
        )
        fh.write(
            "; triangle_parameters "
            f"half_cycles={int(resolve_half_cycles(half_cycles, cycles))} "
            f"effective_segment_count={int(n_half)} "
            f"start_z={str(start_z).strip().lower()} "
            f"lead_in={float(lead_in):.3f} "
            f"lead_out={float(lead_out):.3f}\n"
        )
        fh.write("G21\n")
        fh.write("G90\n")
        fh.write("; pressure actuation: open with M42 P0 S1, close with M42 P0 S0\n")
        fh.write("; IMPORTANT: pressure is closed during vertex point-tracking B/C changes\n")
        fh.write("; B-angle convention: 0=up, 90=horizontal, 180=down\n")
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

        first_tangent = segment_tangent(pts, 0, 1)
        writer.approach_start(
            start_tip=pts[0],
            start_tangent=first_tangent,
            travel_lift_z=travel_lift_z,
            approach_side_mm=approach_side_mm,
        )
        writer.print_polyline(pts)
        writer.finish(travel_lift_z=travel_lift_z)

        fh.write("; End of file\n")

    seg_tangents, b_cmds, c_cmds, phases, target_physical_bs = writer.compute_segment_bcs_and_phases(pts)
    summary = {
        "out": str(out_path),
        "write_mode": write_mode,
        "orientation_mode": orientation_mode,
        "tip_x_range": (float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))),
        "tip_y_range": (float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))),
        "tip_z_range": (float(np.min(pts[:, 2])), float(np.max(pts[:, 2]))),
        "point_count": int(len(pts)),
        "printed_segment_count": int(len(pts) - 1),
        "edge_samples_per_segment": int(max(1, edge_samples)),
        "tip_b_target_range_deg": (float(np.min(target_physical_bs)), float(np.max(target_physical_bs))),
        "b_command_range_deg": (float(np.min(b_cmds)), float(np.max(b_cmds))),
        "c_command_range_deg": (float(np.min(c_cmds)), float(np.max(c_cmds))),
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
            "Generate calibrated or Cartesian G-code for a straight-segment oscillating triangle wave in the XZ plane. "
            "Default tip-space box is X=[60,115], Y=52, Z=[-155,-125]. "
            "In segment-step mode, each triangle side prints with fixed B/C; at each vertex, pressure closes and "
            "B/C changes while calibrated XYZ compensation holds the tip at the same point. "
            "B convention: 0 deg=up, 90 deg=horizontal, 180 deg=down; C is held at 180 deg by default."
        )
    )
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code file.")
    ap.add_argument("--calibration", default=None, help="Calibration JSON. Required for --write-mode calibrated.")
    ap.add_argument("--write-mode", choices=["calibrated", "cartesian"], default=DEFAULT_WRITE_MODE)
    ap.add_argument("--orientation-mode", choices=["segment-step", "fixed"], default=DEFAULT_ORIENTATION_MODE)
    ap.add_argument("--y-offplane-sign", type=float, default=DEFAULT_Y_OFFPLANE_SIGN, help="Multiplier applied to the calibration off-plane Y term in calibrated mode. Use -1 to flip the sign.")
    ap.add_argument("--r-fit-model", default=None, help="Optional radial fit override, e.g. avg_cubic or cubic.")
    ap.add_argument("--z-fit-model", default=None, help="Optional Z fit override, e.g. avg_cubic or cubic.")

    ap.add_argument("--x-start", type=float, default=DEFAULT_X_START)
    ap.add_argument("--x-end", type=float, default=DEFAULT_X_END)
    ap.add_argument("--y", type=float, default=DEFAULT_Y)
    ap.add_argument("--z-min", type=float, default=DEFAULT_Z_MIN, help="Low tip-space Z of the triangle wave.")
    ap.add_argument("--z-max", type=float, default=DEFAULT_Z_MAX, help="High tip-space Z of the triangle wave.")
    ap.add_argument("--half-cycles", type=int, default=None, help="Number of straight diagonal triangle-wave sides. Overrides --cycles when provided.")
    ap.add_argument("--cycles", type=float, default=DEFAULT_CYCLES, help="Number of full up/down cycles. Used only when --half-cycles is not provided; half_cycles=round(2*cycles).")
    ap.add_argument("--start-z", choices=["low", "high"], default=DEFAULT_START_Z, help="Start the triangle wave at z-min or z-max.")
    ap.add_argument("--lead-in", type=float, default=DEFAULT_LEAD_IN, help="Optional straight lead-in segment length at the start.")
    ap.add_argument("--lead-out", type=float, default=DEFAULT_LEAD_OUT, help="Optional straight lead-out segment length at the end.")

    ap.add_argument("--fixed-b", type=float, default=DEFAULT_FIXED_B, help="Used only when --orientation-mode fixed.")
    ap.add_argument("--fixed-c", type=float, default=DEFAULT_FIXED_C, help="Used only when --orientation-mode fixed.")
    ap.add_argument("--c-deg", type=float, default=DEFAULT_C_DEG, help="Constant C angle used in segment-step mode. Default is 180 deg for an XZ-plane write.")
    ap.add_argument("--b-angle-bias-deg", type=float, default=DEFAULT_B_ANGLE_BIAS_DEG, help="Bias added to the segment-tangent-derived B target before solving / emitting.")
    ap.add_argument("--bc-solve-samples", type=int, default=DEFAULT_BC_SOLVE_SAMPLES)

    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED)
    ap.add_argument("--approach-feed", type=float, default=DEFAULT_APPROACH_FEED)
    ap.add_argument("--fine-approach-feed", type=float, default=DEFAULT_FINE_APPROACH_FEED)
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED)
    ap.add_argument("--track-feed", type=float, default=DEFAULT_TRACK_FEED, help="Feed used while changing attack angle at a fixed tip point with pressure off.")
    ap.add_argument("--travel-lift-z", type=float, default=DEFAULT_TRAVEL_LIFT_Z)
    ap.add_argument("--approach-side-mm", type=float, default=DEFAULT_APPROACH_SIDE_MM)
    ap.add_argument("--edge-samples", type=int, default=DEFAULT_EDGE_SAMPLES, help="Subdivide each printed straight side into this many collinear G1 moves. Use 1 for one line per side.")
    ap.add_argument("--track-b-step-deg", type=float, default=DEFAULT_TRACK_B_STEP_DEG, help="Approximate maximum B increment per compensated point-tracking sample.")
    ap.add_argument("--track-min-steps", type=int, default=DEFAULT_TRACK_MIN_STEPS, help="Minimum samples for each compensated point-tracking B/C transition.")

    ap.add_argument("--emit-extrusion", dest="emit_extrusion", action="store_true", default=DEFAULT_EMIT_EXTRUSION)
    ap.add_argument("--no-emit-extrusion", dest="emit_extrusion", action="store_false")
    ap.add_argument("--preflow-dwell-ms", type=int, default=DEFAULT_PREFLOW_DWELL_MS, help="Dwell after opening the pressure valve and before printing each straight side.")

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
    summary = write_triangle_wave_gcode(**vars(args))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
