#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code for a calibrated robot writing a torus with a
hybrid strategy. From theta=-180 deg to a switch angle (default 150 deg), the
path is a continuous toroidal helix controlled by --layer-height. The remaining
cross-section layers are generated over theta=150..180 but printed in reverse
order, starting at theta=180 and finishing back at the switch angle. Each
remainder layer is split into two max-Y to min-Y semicircles: first the
largest global-X side, then an X-offset reposition to the untraced side, then
the smallest global-X side.

The final-strategy B angle flares from 110 deg at theta=150 to 90 deg at
theta=180 by default. The first semicircle ramps C from 110 to 90 deg; the
second ramps C from 70 to 90 deg. Calibrated mode uses:

    stage_xyz = desired_tip_xyz - offset_tip(B_machine, C_deg)
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------- Defaults ----------------
DEFAULT_OUT = "gcode_generation/torus_hybrid_calibrated.gcode"

DEFAULT_CENTER_X = 100.0
DEFAULT_CENTER_Y = 52.0
DEFAULT_CENTER_Z = -145.0
DEFAULT_MAJOR_RADIUS = 20.0
DEFAULT_TUBE_DIAMETER = 10.0
DEFAULT_LAYER_HEIGHT = 1.0

# 36 half-steps => 5 degree spacing around the major circle.
DEFAULT_MAJOR_HALF_LAYERS = 36
DEFAULT_MINOR_SEGMENTS = 72
DEFAULT_INCLUDE_CLOSING_LAYER = True
DEFAULT_MINOR_SEGMENTS_PER_TURN = 72
DEFAULT_FINAL_ARC_SEGMENTS = 36
DEFAULT_PHI0_DEG = 0.0

# User-requested C schedule.
DEFAULT_C_LEFT_HALF = 180.0
DEFAULT_C_RIGHT_HALF = 0.0
DEFAULT_SPIN_STEPS = 18
DEFAULT_STRATEGY_SWITCH_THETA_DEG = 150.0
DEFAULT_ADJUST_REMAINDER_LAYER_SPACING = True
DEFAULT_REMAINDER_B_START_DEG = 110.0
DEFAULT_REMAINDER_B_END_DEG = 90.0
DEFAULT_REMAINDER_C_NOMINAL_DEG = 90.0
DEFAULT_REMAINDER_C_FIRST_START_DEG = 110.0
DEFAULT_REMAINDER_C_FIRST_END_DEG = 90.0
DEFAULT_REMAINDER_C_SECOND_START_DEG = 70.0
DEFAULT_REMAINDER_C_SECOND_END_DEG = 90.0
DEFAULT_REMAINDER_X_OFFSET_MM = 0.5
DEFAULT_HANDOFF_ORIENTATION_STEPS = 12
DEFAULT_END_DWELL_MS = 0
DEFAULT_EXIT_Y_MM = 5.0
DEFAULT_EXIT_Z_MM = 20.0
DEFAULT_MOVE_MACHINE_END_AFTER_EXIT = False

# Startup / shutdown stage positions. These are machine-stage positions, not tip coordinates.
DEFAULT_MACHINE_START_X = 65.0
DEFAULT_MACHINE_START_Y = 80.0
DEFAULT_MACHINE_START_Z = -30.0
DEFAULT_MACHINE_START_B = 0.0
DEFAULT_MACHINE_START_C = 0.0
DEFAULT_MACHINE_END_X = 110.0
DEFAULT_MACHINE_END_Y = 80.0
DEFAULT_MACHINE_END_Z = -30.0
DEFAULT_MACHINE_END_B = 0.0
DEFAULT_MACHINE_END_C = 0.0

# Feeds.
DEFAULT_TRAVEL_FEED = 1000.0
DEFAULT_APPROACH_FEED = 500.0
DEFAULT_FINE_APPROACH_FEED = 80.0
DEFAULT_PRINT_FEED = 250.0
DEFAULT_C_SPIN_FEED = 15000.0
DEFAULT_REMAINDER_REPOSITION_FEED = 500.0
DEFAULT_PRESSURE_ADVANCE_FEED = 120.0
DEFAULT_PRESSURE_RETRACT_FEED = 240.0

# Extrusion / pressure. These are copied in spirit from the provided node script.
DEFAULT_EXTRUSION_PER_MM = 0.0015
DEFAULT_EXTRUSION_MULTIPLIER = 1.0
DEFAULT_PRIME_MM = 0.2
DEFAULT_PRESSURE_OFFSET_MM = 4.0
DEFAULT_PREFLOW_DWELL_MS = 500
DEFAULT_LAYER_END_DWELL_MS = 0

# Solver / safety.
DEFAULT_BC_SOLVE_SAMPLES = 5001
DEFAULT_TRANSITION_MAX_STEP_DEG = 5.0
DEFAULT_BBOX_MIN = -1e9
DEFAULT_BBOX_MAX = 1e9


# ---------------- Calibration model ----------------
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


def poly_eval(coeffs: Any, u: Any, default_if_none: Optional[float] = None) -> np.ndarray:
    u_arr = np.asarray(u, dtype=float)
    if coeffs is None:
        if default_if_none is None:
            raise ValueError("Missing required polynomial coefficients.")
        return np.full_like(u_arr, float(default_if_none), dtype=float)
    arr = np.asarray(coeffs, dtype=float).reshape(-1)
    if arr.size == 0:
        if default_if_none is None:
            raise ValueError("Polynomial coefficients array is empty.")
        return np.full_like(u_arr, float(default_if_none), dtype=float)
    return np.polyval(arr, u_arr)


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
    """Small dependency-free PCHIP evaluator matching the style of the supplied script."""
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


def load_calibration(json_path: str, offplane_y_sign: float = 1.0) -> Calibration:
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
        offplane_y_sign=float(offplane_y_sign),
    )


def make_cartesian_calibration() -> Calibration:
    """Fallback for dry-run/cartesian output. No tip offset is applied."""
    return Calibration(
        pr=np.zeros(1),
        pz=np.zeros(1),
        py_off=np.zeros(1),
        pa=np.array([1.0, 0.0]),
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
        raise ValueError("Calibration is missing tip_angle_coeffs / tip_angle model.")
    return poly_eval(cal.pa, b)


def predict_tip_offset_xyz(cal: Calibration, b_machine: float, c_deg: float) -> np.ndarray:
    """Tip offset in machine XYZ for the given calibrated B machine value and C angle."""
    r = float(eval_r(cal, b_machine))
    z = float(eval_z(cal, b_machine))
    y_off = float(eval_offplane_y(cal, b_machine))
    c = math.radians(float(c_deg))
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def stage_xyz_for_tip(cal: Calibration, tip_xyz: np.ndarray, b_machine: float, c_deg: float) -> np.ndarray:
    return np.asarray(tip_xyz, dtype=float) - predict_tip_offset_xyz(cal, b_machine, c_deg)


def solve_b_for_target_tip_angle(cal: Calibration, target_angle_deg: float, search_samples: int = DEFAULT_BC_SOLVE_SAMPLES) -> float:
    """Return B machine coordinate whose calibrated physical tip angle is closest to target_angle_deg."""
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
                    f"[{float(cal.b_min):.3f}, {float(cal.b_max):.3f}]; outside values use linear offplane_y fit."
                )
            else:
                warnings.append(
                    f"{label} PCHIP knot range [{x_min:.3f}, {x_max:.3f}] does not cover B range "
                    f"[{float(cal.b_min):.3f}, {float(cal.b_max):.3f}]; outside values are clamped by PCHIP evaluator."
                )
    return warnings
# ---------------- Torus geometry ----------------
@dataclass(frozen=True)
class HelixSegment:
    points: np.ndarray
    theta_deg: np.ndarray
    phi_deg: np.ndarray
    sample_count: int
    spin_index: int
    arc_length_mm: float
    minor_turns: float


@dataclass(frozen=True)
class RemainderLayer:
    index: int
    theta_deg: float
    b_target_deg: float
    first_points: np.ndarray
    first_c_deg: np.ndarray
    second_points: np.ndarray
    second_c_deg: np.ndarray
    x_offset_direction: float


@dataclass(frozen=True)
class HybridTorusPlan:
    helix: HelixSegment
    remainder_layers: Tuple[RemainderLayer, ...]
    requested_layer_height: float
    actual_remainder_layer_spacing: float
    remainder_spacing_adjusted: bool
    major_circumference: float
    switch_theta_deg: float
    final_theta_deg: float


def torus_center(theta_deg: float, center_xyz: np.ndarray, major_radius: float) -> np.ndarray:
    th = math.radians(float(theta_deg))
    return center_xyz + np.array([major_radius * math.cos(th), 0.0, major_radius * math.sin(th)], dtype=float)


def torus_radial(theta_deg: float) -> np.ndarray:
    th = math.radians(float(theta_deg))
    return np.array([math.cos(th), 0.0, math.sin(th)], dtype=float)


def torus_point(theta_deg: float, phi_rad: float, center_xyz: np.ndarray, major_radius: float, tube_radius: float) -> np.ndarray:
    center = torus_center(theta_deg, center_xyz, major_radius)
    radial = torus_radial(theta_deg)
    y_axis = np.array([0.0, 1.0, 0.0], dtype=float)
    return center + float(tube_radius) * (math.cos(float(phi_rad)) * radial + math.sin(float(phi_rad)) * y_axis)


def helix_b_target_angle_deg(theta_deg: float) -> float:
    return abs(float(theta_deg))


def helix_c_angle_for_theta(theta_deg: float, c_left_half: float, c_right_half: float) -> float:
    return float(c_left_half if float(theta_deg) <= 0.0 else c_right_half)


def lerp(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * float(t)


def dedup_sorted(values: Sequence[float], tol: float = 1e-12) -> List[float]:
    vals = sorted(float(v) for v in values)
    out: List[float] = []
    for v in vals:
        if not out or abs(v - out[-1]) > tol:
            out.append(v)
    return out


def build_helix_to_switch(
    center_xyz: np.ndarray,
    major_radius: float,
    tube_radius: float,
    layer_height: float,
    minor_segments_per_turn: int,
    phi0_deg: float,
    switch_theta_deg: float,
) -> HelixSegment:
    if not (-180.0 < float(switch_theta_deg) <= 180.0):
        raise ValueError("--strategy-switch-theta-deg must be in (-180, 180]")
    if float(switch_theta_deg) <= 0.0:
        raise ValueError("This hybrid strategy expects --strategy-switch-theta-deg > 0 so the C spin at theta=0 remains in the helix part.")
    if float(major_radius) <= 0.0:
        raise ValueError("--major-radius must be positive")
    if float(tube_radius) <= 0.0:
        raise ValueError("--tube-diameter must be positive")
    if float(layer_height) <= 0.0:
        raise ValueError("--layer-height must be positive")

    theta_span = float(switch_theta_deg) - (-180.0)
    arc_length = float(major_radius) * math.radians(theta_span)
    minor_turns = arc_length / float(layer_height)
    sample_count = max(8, int(math.ceil(minor_turns * max(8, int(minor_segments_per_turn)))))

    # Include theta=0 exactly so the C spin tracks a fixed tip at the rightmost point.
    u_zero = 180.0 / theta_span
    u_values = dedup_sorted(list(np.linspace(0.0, 1.0, sample_count + 1)) + [u_zero])

    phi0 = math.radians(float(phi0_deg))
    pts: List[np.ndarray] = []
    theta_values: List[float] = []
    phi_values: List[float] = []
    spin_index = 0
    for idx, u in enumerate(u_values):
        theta = -180.0 + theta_span * u
        s = float(major_radius) * math.radians(theta + 180.0)
        phi = phi0 + 2.0 * math.pi * s / float(layer_height)
        pts.append(torus_point(theta, phi, center_xyz, major_radius, tube_radius))
        theta_values.append(theta)
        phi_values.append(math.degrees(phi))
        if abs(theta) < 1e-10:
            spin_index = idx

    return HelixSegment(
        points=np.asarray(pts, dtype=float),
        theta_deg=np.asarray(theta_values, dtype=float),
        phi_deg=np.asarray(phi_values, dtype=float),
        sample_count=len(pts) - 1,
        spin_index=int(spin_index),
        arc_length_mm=float(arc_length),
        minor_turns=float(minor_turns),
    )


def make_remainder_theta_stations(
    major_radius: float,
    layer_height: float,
    switch_theta_deg: float,
    final_theta_deg: float,
    adjust_spacing: bool,
) -> Tuple[np.ndarray, float, bool]:
    theta0 = float(switch_theta_deg)
    theta1 = float(final_theta_deg)
    if theta1 <= theta0:
        raise ValueError("final theta must be greater than switch theta for the remainder strategy")
    arc_len = float(major_radius) * math.radians(theta1 - theta0)
    if bool(adjust_spacing):
        steps = max(1, int(round(arc_len / float(layer_height))))
        actual = arc_len / steps
        return np.linspace(theta0, theta1, steps + 1), float(actual), True

    step_deg = math.degrees(float(layer_height) / float(major_radius))
    vals = [theta0]
    cur = theta0
    while cur + step_deg < theta1 - 1e-9:
        cur += step_deg
        vals.append(cur)
    if abs(vals[-1] - theta1) > 1e-9:
        vals.append(theta1)
    return np.asarray(vals, dtype=float), float(layer_height), False


def semicircle_phi_arrays_for_global_x(theta_deg: float, arc_segments: int) -> Tuple[np.ndarray, np.ndarray]:
    radial_x = float(torus_radial(theta_deg)[0])
    n = max(4, int(arc_segments))
    # Max Y is phi=90 deg. Min Y is phi=-90/270 deg.
    if radial_x >= 0.0:
        first = np.linspace(math.radians(90.0), math.radians(-90.0), n + 1)
        second = np.linspace(math.radians(90.0), math.radians(270.0), n + 1)
    else:
        first = np.linspace(math.radians(90.0), math.radians(270.0), n + 1)
        second = np.linspace(math.radians(90.0), math.radians(-90.0), n + 1)
    return first, second


def build_remainder_layers(
    center_xyz: np.ndarray,
    major_radius: float,
    tube_radius: float,
    theta_stations: np.ndarray,
    b_start_deg: float,
    b_end_deg: float,
    c_first_start_deg: float,
    c_first_end_deg: float,
    c_second_start_deg: float,
    c_second_end_deg: float,
    arc_segments: int,
) -> Tuple[RemainderLayer, ...]:
    layers: List[RemainderLayer] = []
    theta0 = float(theta_stations[0])
    theta1 = float(theta_stations[-1])
    denom = max(1e-12, theta1 - theta0)
    for i, theta in enumerate(theta_stations):
        t = (float(theta) - theta0) / denom
        b_target = lerp(b_start_deg, b_end_deg, t)
        phi_first, phi_second = semicircle_phi_arrays_for_global_x(float(theta), arc_segments)
        first_pts = np.asarray([torus_point(float(theta), p, center_xyz, major_radius, tube_radius) for p in phi_first], dtype=float)
        second_pts = np.asarray([torus_point(float(theta), p, center_xyz, major_radius, tube_radius) for p in phi_second], dtype=float)
        first_c = np.linspace(float(c_first_start_deg), float(c_first_end_deg), len(first_pts))
        second_c = np.linspace(float(c_second_start_deg), float(c_second_end_deg), len(second_pts))
        first_mid_x = float(first_pts[len(first_pts) // 2, 0])
        second_mid_x = float(second_pts[len(second_pts) // 2, 0])
        x_dir = 1.0 if (second_mid_x - first_mid_x) >= 0.0 else -1.0
        layers.append(RemainderLayer(i, float(theta), float(b_target), first_pts, first_c, second_pts, second_c, float(x_dir)))
    return tuple(layers)


def reverse_remainder_layers(layers: Sequence[RemainderLayer]) -> Tuple[RemainderLayer, ...]:
    reversed_layers: List[RemainderLayer] = []
    for i, layer in enumerate(reversed(layers)):
        reversed_layers.append(
            RemainderLayer(
                i,
                layer.theta_deg,
                layer.b_target_deg,
                layer.first_points,
                layer.first_c_deg,
                layer.second_points,
                layer.second_c_deg,
                layer.x_offset_direction,
            )
        )
    return tuple(reversed_layers)


def build_hybrid_torus_plan(
    center_xyz: np.ndarray,
    major_radius: float,
    tube_radius: float,
    layer_height: float,
    minor_segments_per_turn: int,
    final_arc_segments: int,
    phi0_deg: float,
    switch_theta_deg: float,
    adjust_remainder_layer_spacing: bool,
    b_start_deg: float,
    b_end_deg: float,
    c_first_start_deg: float,
    c_first_end_deg: float,
    c_second_start_deg: float,
    c_second_end_deg: float,
) -> HybridTorusPlan:
    final_theta = 180.0
    helix = build_helix_to_switch(center_xyz, major_radius, tube_radius, layer_height, minor_segments_per_turn, phi0_deg, switch_theta_deg)
    theta_stations, actual_spacing, adjusted = make_remainder_theta_stations(major_radius, layer_height, switch_theta_deg, final_theta, adjust_remainder_layer_spacing)
    remainder_layers = reverse_remainder_layers(
        build_remainder_layers(
            center_xyz,
            major_radius,
            tube_radius,
            theta_stations,
            b_start_deg,
            b_end_deg,
            c_first_start_deg,
            c_first_end_deg,
            c_second_start_deg,
            c_second_end_deg,
            final_arc_segments,
        )
    )
    return HybridTorusPlan(
        helix=helix,
        remainder_layers=remainder_layers,
        requested_layer_height=float(layer_height),
        actual_remainder_layer_spacing=float(actual_spacing),
        remainder_spacing_adjusted=bool(adjusted),
        major_circumference=float(2.0 * math.pi * major_radius),
        switch_theta_deg=float(switch_theta_deg),
        final_theta_deg=final_theta,
    )


# ---------------- G-code writer ----------------
def fmt_axes(axes_vals: Sequence[Tuple[str, float]]) -> str:
    return " ".join(f"{ax}{val:.3f}" for ax, val in axes_vals)


class TorusGCodeWriter:
    def __init__(
        self,
        fh,
        cal: Calibration,
        calibrated: bool,
        emit_bc: bool,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        print_feed: float,
        travel_feed: float,
        approach_feed: float,
        fine_approach_feed: float,
        c_spin_feed: float,
        reposition_feed: float,
        extrusion_per_mm: float,
        extrusion_multiplier: float,
        pressure_offset_mm: float,
        pressure_advance_feed: float,
        pressure_retract_feed: float,
        preflow_dwell_ms: int,
        end_dwell_ms: int,
        bc_solve_samples: int,
    ) -> None:
        self.fh = fh
        self.cal = cal
        self.calibrated = bool(calibrated)
        self.emit_bc = bool(emit_bc)
        self.bbox_min = np.asarray(bbox_min, dtype=float)
        self.bbox_max = np.asarray(bbox_max, dtype=float)
        self.print_feed = float(print_feed)
        self.travel_feed = float(travel_feed)
        self.approach_feed = float(approach_feed)
        self.fine_approach_feed = float(fine_approach_feed)
        self.c_spin_feed = float(c_spin_feed)
        self.reposition_feed = float(reposition_feed)
        self.extrusion_per_mm = float(extrusion_per_mm)
        self.extrusion_multiplier = float(extrusion_multiplier)
        self.pressure_offset_mm = float(pressure_offset_mm)
        self.pressure_advance_feed = float(pressure_advance_feed)
        self.pressure_retract_feed = float(pressure_retract_feed)
        self.preflow_dwell_ms = int(preflow_dwell_ms)
        self.end_dwell_ms = int(end_dwell_ms)
        self.bc_solve_samples = int(bc_solve_samples)

        self.u_material_abs = 0.0
        self.pressure_charged = False
        self.cur_tip: Optional[np.ndarray] = None
        self.cur_stage: Optional[np.ndarray] = None
        self.cur_b_machine = 0.0
        self.cur_b_target_deg = 0.0
        self.cur_c = 0.0
        self.stage_min = np.array([np.inf, np.inf, np.inf], dtype=float)
        self.stage_max = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
        self.b_machine_min = np.inf
        self.b_machine_max = -np.inf
        self.c_min = np.inf
        self.c_max = -np.inf
        self.total_print_mm = 0.0
        self.total_travel_mm = 0.0
        self.warnings: List[str] = []
        self._b_cache: Dict[float, float] = {}

    def b_machine_for_target(self, b_target_deg: float) -> float:
        key = round(float(b_target_deg), 8)
        if key not in self._b_cache:
            if self.calibrated:
                self._b_cache[key] = solve_b_for_target_tip_angle(self.cal, key, self.bc_solve_samples)
            else:
                self._b_cache[key] = key
        return self._b_cache[key]

    def clamp_stage(self, p: np.ndarray, context: str) -> np.ndarray:
        p = np.asarray(p, dtype=float)
        q = np.minimum(np.maximum(p, self.bbox_min), self.bbox_max)
        if np.linalg.norm(q - p) > 1e-9:
            self.warnings.append(f"WARNING: clamped stage point during {context}: requested={p.tolist()} clamped={q.tolist()}")
        return q

    def u_cmd_actual(self) -> float:
        return self.u_material_abs + (self.pressure_offset_mm if self.pressure_charged else 0.0)

    def tip_to_stage(self, tip_xyz: np.ndarray, b_target_deg: float, c_deg: float) -> Tuple[np.ndarray, float]:
        b_machine = self.b_machine_for_target(b_target_deg)
        if self.calibrated:
            stage = stage_xyz_for_tip(self.cal, tip_xyz, b_machine, c_deg)
        else:
            stage = np.asarray(tip_xyz, dtype=float)
        return self.clamp_stage(stage, "tip_to_stage"), b_machine

    def write_move(self, stage_xyz: np.ndarray, b_machine: float, b_target_deg: float, c_deg: float, feed: float, u_value: Optional[float] = None, comment: Optional[str] = None) -> None:
        if comment:
            self.fh.write(f"; {comment}\n")
        axes: List[Tuple[str, float]] = [(self.cal.x_axis, float(stage_xyz[0])), (self.cal.y_axis, float(stage_xyz[1])), (self.cal.z_axis, float(stage_xyz[2]))]
        if self.calibrated or self.emit_bc:
            axes.extend([(self.cal.b_axis, float(b_machine)), (self.cal.c_axis, float(c_deg))])
        if u_value is not None:
            axes.append((self.cal.u_axis, float(u_value)))
        self.fh.write(f"G1 {fmt_axes(axes)} F{float(feed):.0f}\n")
        self.cur_stage = np.asarray(stage_xyz, dtype=float).copy()
        self.cur_b_machine = float(b_machine)
        self.cur_b_target_deg = float(b_target_deg)
        self.cur_c = float(c_deg)
        self.stage_min = np.minimum(self.stage_min, self.cur_stage)
        self.stage_max = np.maximum(self.stage_max, self.cur_stage)
        self.b_machine_min = min(self.b_machine_min, float(b_machine))
        self.b_machine_max = max(self.b_machine_max, float(b_machine))
        self.c_min = min(self.c_min, float(c_deg))
        self.c_max = max(self.c_max, float(c_deg))

    def move_to_tip(self, tip_xyz: np.ndarray, b_target_deg: float, c_deg: float, feed: float, comment: Optional[str] = None, extrude_from_current_tip: bool = False) -> None:
        tip = np.asarray(tip_xyz, dtype=float)
        prev_tip = None if self.cur_tip is None else self.cur_tip.copy()
        stage, b_machine = self.tip_to_stage(tip, b_target_deg, c_deg)
        u_val = None
        if extrude_from_current_tip:
            seg_len = 0.0 if prev_tip is None else float(np.linalg.norm(tip - prev_tip))
            self.total_print_mm += seg_len
            self.u_material_abs += self.extrusion_per_mm * self.extrusion_multiplier * seg_len
            u_val = self.u_cmd_actual()
        else:
            if prev_tip is not None:
                self.total_travel_mm += float(np.linalg.norm(tip - prev_tip))
        self.write_move(stage, b_machine, b_target_deg, c_deg, feed, u_value=u_val, comment=comment)
        self.cur_tip = tip.copy()

    def pressure_preload_before_print(self) -> None:
        if self.extrusion_per_mm == 0.0:
            return
        if self.pressure_offset_mm > 0.0 and not self.pressure_charged:
            self.pressure_charged = True
            self.fh.write("; pressure preload before hybrid torus pass\n")
            self.fh.write(f"G1 {self.cal.u_axis}{self.u_cmd_actual():.3f} F{self.pressure_advance_feed:.0f}\n")
            if self.preflow_dwell_ms > 0:
                self.fh.write(f"G4 P{self.preflow_dwell_ms}\n")

    def pressure_release_after_print(self) -> None:
        if self.extrusion_per_mm == 0.0:
            return
        if self.pressure_offset_mm > 0.0 and self.pressure_charged:
            self.pressure_charged = False
            self.fh.write("; pressure release after hybrid torus pass\n")
            self.fh.write(f"G1 {self.cal.u_axis}{self.u_cmd_actual():.3f} F{self.pressure_retract_feed:.0f}\n")

    def spin_c_tracking_tip(self, tip_xyz: np.ndarray, b_target_deg: float, c_from: float, c_to: float, steps: int) -> None:
        n = max(1, int(steps))
        self.fh.write(f"; C spin while tracking fixed tip: C {float(c_from):.3f} -> {float(c_to):.3f}, physical_B_target={float(b_target_deg):.3f}; no extrusion during spin\n")
        for i in range(1, n + 1):
            c = float(c_from + (c_to - c_from) * i / n)
            self.move_to_tip(tip_xyz, b_target_deg, c, self.c_spin_feed, extrude_from_current_tip=False)

    def reorient_tracking_tip(self, tip_xyz: np.ndarray, b_from_deg: float, c_from_deg: float, b_to_deg: float, c_to_deg: float, steps: int, feed: float, comment: str) -> None:
        n = max(1, int(steps))
        self.fh.write(f"; {comment}: track fixed tip while B {float(b_from_deg):.3f}->{float(b_to_deg):.3f}, C {float(c_from_deg):.3f}->{float(c_to_deg):.3f}; no extrusion\n")
        for i in range(1, n + 1):
            t = i / n
            self.move_to_tip(tip_xyz, lerp(b_from_deg, b_to_deg, t), lerp(c_from_deg, c_to_deg, t), feed, extrude_from_current_tip=False)

    def print_helix_to_switch(self, helix: HelixSegment, c_left_half: float, c_right_half: float, spin_steps: int) -> None:
        points = helix.points
        theta = helix.theta_deg
        if len(points) < 2:
            raise ValueError("Helix path has fewer than 2 points")
        self.fh.write(f"; HELIX_SEGMENT_START theta_start={theta[0]:.6f} theta_end={theta[-1]:.6f} samples={helix.sample_count} minor_turns={helix.minor_turns:.6f}\n")
        self.move_to_tip(points[0], helix_b_target_angle_deg(theta[0]), helix_c_angle_for_theta(theta[0], c_left_half, c_right_half), self.travel_feed, comment="travel to hybrid torus start at theta=-180", extrude_from_current_tip=False)
        self.move_to_tip(points[0], helix_b_target_angle_deg(theta[0]), helix_c_angle_for_theta(theta[0], c_left_half, c_right_half), self.fine_approach_feed, comment="fine approach to hybrid torus start", extrude_from_current_tip=False)
        self.pressure_preload_before_print()
        spun_at_zero = False
        for i in range(1, len(points)):
            th = float(theta[i])
            c = helix_c_angle_for_theta(th, c_left_half, c_right_half)
            self.move_to_tip(points[i], helix_b_target_angle_deg(th), c, self.print_feed, extrude_from_current_tip=True)
            if (not spun_at_zero) and abs(th) < 1e-9:
                self.fh.write(f"; reached right-most point theta={th:.6f}; switching C for second helix half\n")
                self.spin_c_tracking_tip(points[i], helix_b_target_angle_deg(th), c_left_half, c_right_half, spin_steps)
                spun_at_zero = True
        self.fh.write("; HELIX_SEGMENT_END\n")

    def print_arc_points(self, points: np.ndarray, c_values: np.ndarray, b_target_deg: float, label: str) -> None:
        self.fh.write(f"; {label}: extruding {len(points)} points, B_target={float(b_target_deg):.3f}\n")
        if self.cur_tip is None or float(np.linalg.norm(np.asarray(points[0]) - self.cur_tip)) > 1e-8:
            self.move_to_tip(points[0], b_target_deg, float(c_values[0]), self.fine_approach_feed, comment=f"{label}: move to arc start", extrude_from_current_tip=False)
        else:
            self.move_to_tip(points[0], b_target_deg, float(c_values[0]), self.fine_approach_feed, comment=f"{label}: set arc-start B/C", extrude_from_current_tip=False)
        for p, c in zip(points[1:], c_values[1:]):
            self.move_to_tip(p, b_target_deg, float(c), self.print_feed, extrude_from_current_tip=True)

    def print_remainder_layers(self, layers: Sequence[RemainderLayer], handoff_orientation_steps: int, x_offset_mm: float, nominal_c_deg: float) -> None:
        if not layers:
            return
        first_layer = layers[0]
        if self.cur_tip is not None:
            self.reorient_tracking_tip(self.cur_tip, self.cur_b_target_deg, self.cur_c, first_layer.b_target_deg, float(first_layer.first_c_deg[0]), handoff_orientation_steps, self.c_spin_feed, "handoff from helix to two-semicircle remainder strategy")
        self.fh.write(f"; REMAINDER_LAYER_STRATEGY_START layers={len(layers)} theta_start={layers[0].theta_deg:.6f} theta_end={layers[-1].theta_deg:.6f}\n")
        for layer in layers:
            self.fh.write(f"; REMAINDER_LAYER_START index={layer.index} theta={layer.theta_deg:.6f} B_target={layer.b_target_deg:.6f} x_offset_direction={layer.x_offset_direction:+.0f}\n")
            self.print_arc_points(layer.first_points, layer.first_c_deg, layer.b_target_deg, f"layer {layer.index} first/largest-X semicircle")
            offset_vec = np.array([layer.x_offset_direction * abs(float(x_offset_mm)), 0.0, 0.0], dtype=float)
            first_end = layer.first_points[-1]
            second_start = layer.second_points[0]
            self.move_to_tip(first_end + offset_vec, layer.b_target_deg, float(nominal_c_deg), self.reposition_feed, comment=f"layer {layer.index}: X offset toward untraced semicircle after first arc", extrude_from_current_tip=False)
            self.move_to_tip(second_start + offset_vec, layer.b_target_deg, float(layer.second_c_deg[0]), self.reposition_feed, comment=f"layer {layer.index}: move offset to max-Y start of second arc", extrude_from_current_tip=False)
            self.move_to_tip(second_start, layer.b_target_deg, float(layer.second_c_deg[0]), self.fine_approach_feed, comment=f"layer {layer.index}: fine move onto second arc start", extrude_from_current_tip=False)
            self.print_arc_points(layer.second_points, layer.second_c_deg, layer.b_target_deg, f"layer {layer.index} second/smallest-X semicircle")
            self.fh.write(f"; REMAINDER_LAYER_END index={layer.index}\n")
            if self.end_dwell_ms > 0:
                self.fh.write(f"G4 P{self.end_dwell_ms}\n")
        self.fh.write("; REMAINDER_LAYER_STRATEGY_END\n")

    def exit_print_space(self, minus_y_mm: float, plus_z_mm: float, b_target_deg: float, c_deg: float) -> None:
        if self.cur_tip is None:
            return
        self.pressure_release_after_print()
        self.fh.write(f"; exit print space: move tip -Y by {float(minus_y_mm):.3f} mm, then +Z by {float(plus_z_mm):.3f} mm\n")
        p1 = self.cur_tip + np.array([0.0, -abs(float(minus_y_mm)), 0.0], dtype=float)
        self.move_to_tip(p1, b_target_deg, c_deg, self.approach_feed, comment="exit: move in -Y", extrude_from_current_tip=False)
        p2 = p1 + np.array([0.0, 0.0, abs(float(plus_z_mm))], dtype=float)
        self.move_to_tip(p2, b_target_deg, c_deg, self.travel_feed, comment="exit: lift in +Z", extrude_from_current_tip=False)

    def print_hybrid_torus(self, plan: HybridTorusPlan, c_left_half: float, c_right_half: float, spin_steps: int, handoff_orientation_steps: int, x_offset_mm: float, nominal_c_deg: float, exit_y_mm: float, exit_z_mm: float, final_b_target_deg: float, final_c_deg: float) -> None:
        self.fh.write(f"; TORUS_HYBRID_START requested_layer_height={plan.requested_layer_height:.6f} switch_theta={plan.switch_theta_deg:.6f} final_theta={plan.final_theta_deg:.6f}\n")
        self.print_helix_to_switch(plan.helix, c_left_half, c_right_half, spin_steps)
        self.print_remainder_layers(plan.remainder_layers, handoff_orientation_steps, x_offset_mm, nominal_c_deg)
        if self.end_dwell_ms > 0:
            self.fh.write(f"G4 P{self.end_dwell_ms}\n")
        self.exit_print_space(exit_y_mm, exit_z_mm, final_b_target_deg, final_c_deg)
        self.fh.write("; TORUS_HYBRID_END\n")


# ---------------- Generation ----------------
def write_torus_gcode(args: argparse.Namespace) -> None:
    calibrated = str(args.write_mode).strip().lower() == "calibrated"
    if calibrated and not args.calibration:
        raise ValueError("--calibration is required when --write-mode calibrated")
    cal = load_calibration(args.calibration) if calibrated else make_cartesian_calibration()
    cal.offplane_y_sign = float(args.y_offplane_sign)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    center_xyz = np.array([args.center_x, args.center_y, args.center_z], dtype=float)
    tube_radius = 0.5 * float(args.tube_diameter)
    plan = build_hybrid_torus_plan(
        center_xyz=center_xyz,
        major_radius=args.major_radius,
        tube_radius=tube_radius,
        layer_height=args.layer_height,
        minor_segments_per_turn=args.minor_segments_per_turn,
        final_arc_segments=args.final_arc_segments,
        phi0_deg=args.phi0_deg,
        switch_theta_deg=args.strategy_switch_theta_deg,
        adjust_remainder_layer_spacing=args.adjust_remainder_layer_spacing,
        b_start_deg=args.remainder_b_start_deg,
        b_end_deg=args.remainder_b_end_deg,
        c_first_start_deg=args.remainder_c_first_start_deg,
        c_first_end_deg=args.remainder_c_first_end_deg,
        c_second_start_deg=args.remainder_c_second_start_deg,
        c_second_end_deg=args.remainder_c_second_end_deg,
    )

    bbox_min = np.array([args.bbox_x_min, args.bbox_y_min, args.bbox_z_min], dtype=float)
    bbox_max = np.array([args.bbox_x_max, args.bbox_y_max, args.bbox_z_max], dtype=float)

    with out_path.open("w", encoding="utf-8") as f:
        f.write("; generated by generate_torus_hybrid_calibrated.py\n")
        f.write("; hybrid torus: continuous helix to switch angle, then two-semicircle remainder layers printed from theta=180 back to the switch angle\n")
        if calibrated:
            f.write("; calibration-based tip-position planning: stage_xyz = desired_tip_xyz - offset_tip(B_machine, C)\n")
        else:
            f.write("; cartesian mode: XYZ follows desired tip XYZ directly; B/C emitted only if --emit-bc-in-cartesian is set\n")
        f.write(f"; torus_center = ({args.center_x:.3f}, {args.center_y:.3f}, {args.center_z:.3f})\n")
        f.write(f"; major_centerline_radius = {args.major_radius:.3f}\n")
        f.write(f"; major_centerline_circumference = {plan.major_circumference:.6f}\n")
        f.write(f"; tube_diameter = {args.tube_diameter:.3f}\n")
        f.write(f"; requested_layer_height = {plan.requested_layer_height:.6f}\n")
        f.write(f"; strategy_switch_theta_deg = {plan.switch_theta_deg:.6f}\n")
        f.write(f"; helix_arc_length_mm = {plan.helix.arc_length_mm:.6f}\n")
        f.write(f"; helix_minor_turns = {plan.helix.minor_turns:.6f}\n")
        f.write(f"; helix_samples = {plan.helix.sample_count}; minor_segments_per_turn = {args.minor_segments_per_turn}\n")
        f.write(f"; remainder_layers = {len(plan.remainder_layers)}\n")
        f.write(f"; actual_remainder_layer_spacing = {plan.actual_remainder_layer_spacing:.6f}; adjusted={int(plan.remainder_spacing_adjusted)}\n")
        f.write(f"; final_arc_segments_per_semicircle = {args.final_arc_segments}\n")
        f.write(f"; phi0_deg = {args.phi0_deg:.6f}\n")
        f.write(f"; helix C schedule: theta<=0 C={args.c_left_half:.3f}; theta>0 C={args.c_right_half:.3f}; spin steps={args.spin_steps}\n")
        f.write(f"; remainder B flare: print order theta=180 -> B={args.remainder_b_end_deg:.3f}, theta={args.strategy_switch_theta_deg:.3f} -> B={args.remainder_b_start_deg:.3f}\n")
        f.write(f"; remainder C flare: first/largest-X semicircle C={args.remainder_c_first_start_deg:.3f}->{args.remainder_c_first_end_deg:.3f}; second/smallest-X semicircle C={args.remainder_c_second_start_deg:.3f}->{args.remainder_c_second_end_deg:.3f}\n")
        f.write(f"; remainder_x_offset_mm = {args.remainder_x_offset_mm:.6f}\n")
        f.write(f"; exit: tip -Y {args.exit_y_mm:.3f} mm, then +Z {args.exit_z_mm:.3f} mm\n")
        f.write(f"; axes: X->{cal.x_axis}, Y->{cal.y_axis}, Z->{cal.z_axis}, B->{cal.b_axis}, C->{cal.c_axis}, U->{cal.u_axis}\n")
        f.write("; physical B convention: 0=+Z tangent, 90=horizontal tangent, 180=-Z tangent\n")
        f.write(f"; selected_fit_model = {cal.selected_fit_model or 'legacy-polynomial/cartesian'}\n")
        f.write(f"; selected_offplane_fit_model = {cal.selected_offplane_fit_model or cal.selected_fit_model or 'legacy-polynomial/cartesian'}\n")
        f.write(f"; active_phase = {cal.active_phase}\n")
        f.write(f"; y_offplane_sign = {float(cal.offplane_y_sign):.1f}\n")
        f.write(f"; {describe_model(cal.r_model, 'r')}\n")
        f.write(f"; {describe_model(cal.z_model, 'z')}\n")
        f.write(f"; {describe_model(cal.y_off_model, 'offplane_y')}\n")
        f.write(f"; {describe_model(cal.tip_angle_model, 'tip_angle')}\n")
        for warning in calibration_model_range_warnings(cal):
            f.write(f"; WARNING: {warning}\n")
        f.write(f"; feeds: travel={args.travel_feed:.1f}, approach={args.approach_feed:.1f}, fine_approach={args.fine_approach_feed:.1f}, print={args.print_feed:.1f}, C_spin={args.c_spin_feed:.1f}, reposition={args.remainder_reposition_feed:.1f}\n")
        f.write(f"; extrusion: extrusion_per_mm={args.extrusion_per_mm:.6f}, multiplier={args.extrusion_multiplier:.6f}, prime={args.prime_mm:.3f}, pressure_offset={args.pressure_offset_mm:.3f}\n")

        f.write("G90\n")
        if args.extrusion_per_mm != 0.0:
            f.write("M82\n")
            f.write(f"G92 {cal.u_axis}0\n")
            if abs(float(args.prime_mm)) > 0.0:
                f.write(f"G1 {cal.u_axis}{float(args.prime_mm):.3f} F{max(60.0, float(args.pressure_advance_feed)):.0f} ; prime material\n")

        writer = TorusGCodeWriter(
            fh=f,
            cal=cal,
            calibrated=calibrated,
            emit_bc=bool(args.emit_bc_in_cartesian),
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            print_feed=args.print_feed,
            travel_feed=args.travel_feed,
            approach_feed=args.approach_feed,
            fine_approach_feed=args.fine_approach_feed,
            c_spin_feed=args.c_spin_feed,
            reposition_feed=args.remainder_reposition_feed,
            extrusion_per_mm=args.extrusion_per_mm,
            extrusion_multiplier=args.extrusion_multiplier,
            pressure_offset_mm=args.pressure_offset_mm,
            pressure_advance_feed=args.pressure_advance_feed,
            pressure_retract_feed=args.pressure_retract_feed,
            preflow_dwell_ms=args.preflow_dwell_ms,
            end_dwell_ms=args.end_dwell_ms,
            bc_solve_samples=args.bc_solve_samples,
        )

        f.write("; move to configured machine start pose\n")
        start_axes: List[Tuple[str, float]] = [(cal.x_axis, args.machine_start_x), (cal.y_axis, args.machine_start_y), (cal.z_axis, args.machine_start_z)]
        if calibrated or args.emit_bc_in_cartesian:
            start_axes.extend([(cal.b_axis, args.machine_start_b), (cal.c_axis, args.machine_start_c)])
        f.write(f"G1 {fmt_axes(start_axes)} F{args.travel_feed:.0f}\n")

        writer.print_hybrid_torus(
            plan=plan,
            c_left_half=args.c_left_half,
            c_right_half=args.c_right_half,
            spin_steps=args.spin_steps,
            handoff_orientation_steps=args.handoff_orientation_steps,
            x_offset_mm=args.remainder_x_offset_mm,
            nominal_c_deg=args.remainder_c_nominal_deg,
            exit_y_mm=args.exit_y_mm,
            exit_z_mm=args.exit_z_mm,
            final_b_target_deg=args.remainder_b_end_deg,
            final_c_deg=args.remainder_c_nominal_deg,
        )
        writer.pressure_release_after_print()

        if args.move_machine_end_after_exit:
            f.write("; optional move to configured machine end pose after tip-space exit\n")
            end_axes: List[Tuple[str, float]] = [(cal.x_axis, args.machine_end_x), (cal.y_axis, args.machine_end_y), (cal.z_axis, args.machine_end_z)]
            if calibrated or args.emit_bc_in_cartesian:
                end_axes.extend([(cal.b_axis, args.machine_end_b), (cal.c_axis, args.machine_end_c)])
            f.write(f"G1 {fmt_axes(end_axes)} F{args.travel_feed:.0f}\n")

        f.write("; SUMMARY\n")
        f.write("; mode = hybrid_helix_then_two_semicircle_remainder\n")
        f.write(f"; requested_layer_height = {plan.requested_layer_height:.6f}\n")
        f.write(f"; actual_remainder_layer_spacing = {plan.actual_remainder_layer_spacing:.6f}\n")
        f.write(f"; helix_minor_turns = {plan.helix.minor_turns:.6f}\n")
        f.write(f"; helix_samples = {plan.helix.sample_count}\n")
        f.write(f"; remainder_layers = {len(plan.remainder_layers)}\n")
        f.write(f"; total_printed_path_mm = {writer.total_print_mm:.6f}\n")
        f.write(f"; total_nonextruding_tip_travel_mm = {writer.total_travel_mm:.6f}\n")
        f.write(f"; final_U_material_abs = {writer.u_material_abs:.6f}\n")
        f.write(f"; stage_min = ({writer.stage_min[0]:.6f}, {writer.stage_min[1]:.6f}, {writer.stage_min[2]:.6f})\n")
        f.write(f"; stage_max = ({writer.stage_max[0]:.6f}, {writer.stage_max[1]:.6f}, {writer.stage_max[2]:.6f})\n")
        f.write(f"; B_machine_range = [{writer.b_machine_min:.6f}, {writer.b_machine_max:.6f}]\n")
        f.write(f"; C_range = [{writer.c_min:.6f}, {writer.c_max:.6f}]\n")
        for warning in writer.warnings:
            f.write(f"; {warning}\n")

    print(f"Wrote {out_path}")


# ---------------- CLI ----------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate calibrated Duet/RRF G-code for a hybrid continuous torus with a two-semicircle final strategy.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code path.")
    ap.add_argument("--calibration", default=None, help="Calibration JSON path. Required for --write-mode calibrated.")
    ap.add_argument("--write-mode", choices=("calibrated", "cartesian"), default="calibrated")
    ap.add_argument("--emit-bc-in-cartesian", action="store_true", help="Also emit B/C axes in cartesian dry-run mode.")
    ap.add_argument("--y-offplane-sign", type=float, default=1.0, help="Multiply offplane_y calibration term by this sign.")

    ap.add_argument("--center-x", type=float, default=DEFAULT_CENTER_X)
    ap.add_argument("--center-y", type=float, default=DEFAULT_CENTER_Y)
    ap.add_argument("--center-z", type=float, default=DEFAULT_CENTER_Z)
    ap.add_argument("--major-radius", type=float, default=DEFAULT_MAJOR_RADIUS, help="Torus centerline radius in mm.")
    ap.add_argument("--tube-diameter", type=float, default=DEFAULT_TUBE_DIAMETER, help="Tube/cone diameter in mm.")
    ap.add_argument("--layer-height", type=float, default=DEFAULT_LAYER_HEIGHT, help="Helix pitch and approximate remainder station spacing along the major centerline.")
    ap.add_argument("--minor-segments-per-turn", type=int, default=DEFAULT_MINOR_SEGMENTS_PER_TURN, help="Interpolation segments per tube revolution in the helical part.")
    ap.add_argument("--final-arc-segments", type=int, default=DEFAULT_FINAL_ARC_SEGMENTS, help="Interpolation segments per final-strategy semicircle.")
    ap.add_argument("--phi0-deg", type=float, default=DEFAULT_PHI0_DEG, help="Starting minor/tube angle for the helical part.")

    ap.add_argument("--strategy-switch-theta-deg", type=float, default=DEFAULT_STRATEGY_SWITCH_THETA_DEG, help="Major trig-circle angle where the final two-semicircle strategy begins.")
    ap.add_argument("--adjust-remainder-layer-spacing", dest="adjust_remainder_layer_spacing", action="store_true", default=DEFAULT_ADJUST_REMAINDER_LAYER_SPACING, help="Adjust final station spacing slightly so theta=180 is reached exactly.")
    ap.add_argument("--no-adjust-remainder-layer-spacing", dest="adjust_remainder_layer_spacing", action="store_false", help="Use requested layer height for final station spacing, with a shorter last spacing if needed.")
    ap.add_argument("--remainder-b-start-deg", type=float, default=DEFAULT_REMAINDER_B_START_DEG, help="Physical B target at the strategy switch, default over-bent 110 deg.")
    ap.add_argument("--remainder-b-end-deg", type=float, default=DEFAULT_REMAINDER_B_END_DEG, help="Physical B target at theta=180, default 90 deg.")
    ap.add_argument("--remainder-c-nominal-deg", type=float, default=DEFAULT_REMAINDER_C_NOMINAL_DEG, help="Nominal C used for final repositions and exit.")
    ap.add_argument("--remainder-c-first-start-deg", type=float, default=DEFAULT_REMAINDER_C_FIRST_START_DEG)
    ap.add_argument("--remainder-c-first-end-deg", type=float, default=DEFAULT_REMAINDER_C_FIRST_END_DEG)
    ap.add_argument("--remainder-c-second-start-deg", type=float, default=DEFAULT_REMAINDER_C_SECOND_START_DEG)
    ap.add_argument("--remainder-c-second-end-deg", type=float, default=DEFAULT_REMAINDER_C_SECOND_END_DEG)
    ap.add_argument("--remainder-x-offset-mm", type=float, default=DEFAULT_REMAINDER_X_OFFSET_MM, help="Small X offset used between the two semicircles in a final layer.")
    ap.add_argument("--handoff-orientation-steps", type=int, default=DEFAULT_HANDOFF_ORIENTATION_STEPS, help="Non-extruding fixed-tip B/C interpolation steps at the strategy switch.")

    ap.add_argument("--c-left-half", type=float, default=DEFAULT_C_LEFT_HALF, help="Helix C angle for theta -180..0.")
    ap.add_argument("--c-right-half", type=float, default=DEFAULT_C_RIGHT_HALF, help="Helix C angle for theta 0..switch after the right-side spin.")
    ap.add_argument("--spin-steps", type=int, default=DEFAULT_SPIN_STEPS, help="Number of C-only tracking moves at theta=0.")

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
    ap.add_argument("--move-machine-end-after-exit", action="store_true", default=DEFAULT_MOVE_MACHINE_END_AFTER_EXIT)

    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED)
    ap.add_argument("--approach-feed", type=float, default=DEFAULT_APPROACH_FEED)
    ap.add_argument("--fine-approach-feed", type=float, default=DEFAULT_FINE_APPROACH_FEED)
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED)
    ap.add_argument("--c-spin-feed", type=float, default=DEFAULT_C_SPIN_FEED)
    ap.add_argument("--remainder-reposition-feed", type=float, default=DEFAULT_REMAINDER_REPOSITION_FEED)

    ap.add_argument("--extrusion-per-mm", type=float, default=DEFAULT_EXTRUSION_PER_MM)
    ap.add_argument("--extrusion-multiplier", type=float, default=DEFAULT_EXTRUSION_MULTIPLIER)
    ap.add_argument("--prime-mm", type=float, default=DEFAULT_PRIME_MM)
    ap.add_argument("--pressure-offset-mm", type=float, default=DEFAULT_PRESSURE_OFFSET_MM)
    ap.add_argument("--pressure-advance-feed", type=float, default=DEFAULT_PRESSURE_ADVANCE_FEED)
    ap.add_argument("--pressure-retract-feed", type=float, default=DEFAULT_PRESSURE_RETRACT_FEED)
    ap.add_argument("--preflow-dwell-ms", type=int, default=DEFAULT_PREFLOW_DWELL_MS)
    ap.add_argument("--end-dwell-ms", type=int, default=DEFAULT_END_DWELL_MS)

    ap.add_argument("--exit-y-mm", type=float, default=DEFAULT_EXIT_Y_MM, help="After print, move tip in -Y by this amount.")
    ap.add_argument("--exit-z-mm", type=float, default=DEFAULT_EXIT_Z_MM, help="After -Y exit move, lift tip in +Z by this amount.")

    ap.add_argument("--bc-solve-samples", type=int, default=DEFAULT_BC_SOLVE_SAMPLES)
    ap.add_argument("--bbox-x-min", type=float, default=DEFAULT_BBOX_MIN)
    ap.add_argument("--bbox-x-max", type=float, default=DEFAULT_BBOX_MAX)
    ap.add_argument("--bbox-y-min", type=float, default=DEFAULT_BBOX_MIN)
    ap.add_argument("--bbox-y-max", type=float, default=DEFAULT_BBOX_MAX)
    ap.add_argument("--bbox-z-min", type=float, default=DEFAULT_BBOX_MIN)
    ap.add_argument("--bbox-z-max", type=float, default=DEFAULT_BBOX_MAX)
    return ap


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    write_torus_gcode(args)


if __name__ == "__main__":
    main()
