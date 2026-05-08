#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code for a sine-wave TUBE in the XZ plane.

The tip-space *centerline* is a plain sine wave in the XZ plane at constant Y
(identical in spirit to sine_wave_xz_generator.py). Instead of tracing the
1-D centerline, this generator traces a 3-D tube that winds helically around
that centerline (the torus script's strategy applied to a sine-wave spine).

Diameter profile (s in [0, 1] is the normalized arc length along the
centerline):

    s in [0, transition_in]
        -> linear interp from --diameter-start to --diameter-main
    s in [transition_in, transition_out]
        -> --diameter-main
    s in [transition_out, 1]
        -> linear interp from --diameter-main to --diameter-end

Orientation
-----------
- B angle follows the LOCAL CENTERLINE TANGENT, with the same convention as
  the other scripts:
      0 deg   -> tip points +Z (straight up)
      90 deg  -> tip is horizontal (i.e. centerline tangent in XZ plane)
      180 deg -> tip points -Z (straight down)
  This is what "extrude tangent to the centerline" means here: the tip is
  pointed along the centerline direction even though it traces a helix
  that winds around it.

- C is held constant (default 180 deg) so the tool stays oriented in the
  XZ plane, matching sine_wave_xz_generator.py.

Stage planning
--------------
In calibrated mode:
    stage_xyz = desired_tip_xyz - offset_tip(B_machine, C_deg)
where B_machine is the calibration motor coordinate that produces a physical
tip angle equal to the desired centerline-tangent angle.

In cartesian mode XYZ follows the desired tip XYZ directly.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ============================================================================
#                                  Defaults
# ============================================================================
DEFAULT_OUT = "gcode_generation/sine_wave_xz_tube.gcode"

# Centerline placement / shape
DEFAULT_X_START = 60.0
DEFAULT_X_END = 120.0
DEFAULT_Y = 52.0
DEFAULT_Z_MIN = -170.0
DEFAULT_Z_MAX = -120.0
DEFAULT_Z_BASELINE = 0.5 * (DEFAULT_Z_MIN + DEFAULT_Z_MAX)
DEFAULT_Z_AMPLITUDE = 6.5
DEFAULT_CYCLES = 2.55
DEFAULT_PHASE_DEG = 0.0
DEFAULT_LEAD_IN = 0.0
DEFAULT_LEAD_OUT = 0.0
DEFAULT_CENTERLINE_SAMPLES = 2001  # dense for arc-length integration

# Tube diameter profile
DEFAULT_DIAMETER_START = 1.0
DEFAULT_DIAMETER_MAIN = 6.0
DEFAULT_DIAMETER_END = 2.0
DEFAULT_TRANSITION_IN_FRAC = 0.20   # diameter reaches main here
DEFAULT_TRANSITION_OUT_FRAC = 0.95  # diameter starts tapering to end here

# Helical sampling
DEFAULT_LAYER_HEIGHT = 0.6          # axial pitch along centerline per minor turn
DEFAULT_MINOR_SEGMENTS_PER_TURN = 36
DEFAULT_PHI0_DEG = 0.0
DEFAULT_FRAME_FLIP_OUTPLANE_SIGN = 1.0  # multiply the +Y out-of-plane axis by this

# Orientation
DEFAULT_WRITE_MODE = "calibrated"   # "calibrated" | "cartesian"
DEFAULT_C_DEG = 180.0
DEFAULT_B_ANGLE_BIAS_DEG = 0.0
DEFAULT_BC_SOLVE_SAMPLES = 5001
DEFAULT_Y_OFFPLANE_SIGN = -1.0

# Motion
DEFAULT_TRAVEL_FEED = 2000.0
DEFAULT_APPROACH_FEED = 1200.0
DEFAULT_FINE_APPROACH_FEED = 150.0
DEFAULT_PRINT_FEED = 400.0
DEFAULT_TRAVEL_LIFT_Z = 8.0
DEFAULT_SAFE_APPROACH_Z = -50.0

# Extrusion / pressure
DEFAULT_EMIT_EXTRUSION = True
DEFAULT_EXTRUSION_PER_MM = 0.0015
DEFAULT_EXTRUSION_MULTIPLIER = 1.0
DEFAULT_PRIME_MM = 0.2
DEFAULT_PRESSURE_OFFSET_MM = 4.0
DEFAULT_PRESSURE_ADVANCE_FEED = 120.0
DEFAULT_PRESSURE_RETRACT_FEED = 240.0
DEFAULT_PREFLOW_DWELL_MS = 500
DEFAULT_END_DWELL_MS = 0

# Stage limits
DEFAULT_BBOX = 1e9


# ============================================================================
#                                  Dataclasses
# ============================================================================
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


@dataclass
class TubePlan:
    """Sampled helical tube path in tip-space."""
    tube_points: np.ndarray             # (N, 3) helix points in tip-space
    centerline_tangents: np.ndarray     # (N, 3) unit centerline tangent at each helix sample
    centerline_points: np.ndarray       # (N, 3) centerline point colocated with each helix sample
    arc_lengths: np.ndarray             # (N,) cumulative arc length at each helix sample
    diameters: np.ndarray               # (N,) diameter at each helix sample
    total_arc_length: float
    minor_turns: float


# ============================================================================
#                          Math / geometry helpers
# ============================================================================
def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if n <= eps:
        return np.zeros_like(arr)
    return arr / n


def desired_physical_b_angle_from_tangent(tangent: np.ndarray) -> float:
    """B-angle from tangent: 0 -> +Z, 90 -> horizontal, 180 -> -Z."""
    t = normalize(np.asarray(tangent, dtype=float))
    tz = float(np.clip(t[2], -1.0, 1.0))
    return float(math.degrees(math.acos(tz)))


def in_plane_normal_xz(tangent: np.ndarray) -> np.ndarray:
    """Unit normal in the XZ plane, rotated +90 deg from `tangent` about +Y.

    With tangent (tx, 0, tz), this returns (-tz, 0, tx). When tangent = +X,
    the normal = +Z; this is the consistent "outward toward +Z when tangent
    is +X" convention used to wind the helix.
    """
    t = np.asarray(tangent, dtype=float)
    return normalize(np.array([-t[2], 0.0, t[0]], dtype=float))


# ============================================================================
#                           Calibration loading
# ============================================================================
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
    """Trivial calibration for cartesian dry-runs (no tip offset)."""
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


# ============================================================================
#                        Sine-wave centerline
# ============================================================================
def build_sine_wave_centerline(
    x_start: float,
    x_end: float,
    y: float,
    z_baseline: float,
    z_amplitude: float,
    cycles: float,
    phase_deg: float,
    lead_in: float,
    lead_out: float,
    samples: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build a dense centerline polyline + analytic-gradient unit tangents +
    cumulative arc length.

    Active wave equation, after optional lead-in / lead-out:
        Z = z_baseline + z_amplitude * sin(2*pi*cycles*s + phase)
    where s in [0, 1] across the active interval. Lead-in / lead-out are
    flat (constant Z = z_baseline) horizontal segments at the very start
    and end. No envelope or endpoint taper is applied.
    """
    x0 = float(x_start)
    x1 = float(x_end)
    if not x1 > x0:
        raise ValueError("--x-end must be greater than --x-start")
    if float(lead_in) < 0.0 or float(lead_out) < 0.0:
        raise ValueError("--lead-in and --lead-out must be >= 0")
    if float(lead_in + lead_out) >= (x1 - x0):
        raise ValueError("--lead-in + --lead-out must be smaller than the total X span")
    if int(samples) < 11:
        raise ValueError("--centerline-samples must be >= 11")

    wave_x0 = x0 + float(lead_in)
    wave_x1 = x1 - float(lead_out)

    n_total = int(samples)
    span = x1 - x0
    n_lead_in = max(2, int(round(n_total * (lead_in / span)))) if lead_in > 0.0 else 0
    n_lead_out = max(2, int(round(n_total * (lead_out / span)))) if lead_out > 0.0 else 0
    n_wave = max(11, n_total - n_lead_in - n_lead_out)

    parts_pts: List[np.ndarray] = []
    parts_tan: List[np.ndarray] = []

    omega = 2.0 * math.pi * float(cycles)
    phase = math.radians(float(phase_deg))

    if n_lead_in > 0:
        xs = np.linspace(x0, wave_x0, n_lead_in, endpoint=False)
        ys = np.full_like(xs, float(y))
        zs = np.full_like(xs, float(z_baseline))
        ts = np.tile(np.array([1.0, 0.0, 0.0]), (len(xs), 1))
        parts_pts.append(np.column_stack([xs, ys, zs]))
        parts_tan.append(ts)

    xs = np.linspace(wave_x0, wave_x1, n_wave, endpoint=(n_lead_out == 0))
    if wave_x1 > wave_x0:
        s = (xs - wave_x0) / (wave_x1 - wave_x0)
    else:
        s = np.zeros_like(xs)
    zs = float(z_baseline) + float(z_amplitude) * np.sin(omega * s + phase)
    ys = np.full_like(xs, float(y))
    # dZ/dx = dZ/ds * ds/dx
    ds_dx = 1.0 / max(wave_x1 - wave_x0, 1e-12)
    dz_dx = float(z_amplitude) * omega * np.cos(omega * s + phase) * ds_dx
    raw_tan = np.column_stack([np.ones_like(xs), np.zeros_like(xs), dz_dx])
    norms = np.linalg.norm(raw_tan, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    ts = raw_tan / norms
    parts_pts.append(np.column_stack([xs, ys, zs]))
    parts_tan.append(ts)

    if n_lead_out > 0:
        xs = np.linspace(wave_x1, x1, n_lead_out)
        ys = np.full_like(xs, float(y))
        zs = np.full_like(xs, float(z_baseline))
        ts = np.tile(np.array([1.0, 0.0, 0.0]), (len(xs), 1))
        parts_pts.append(np.column_stack([xs, ys, zs]))
        parts_tan.append(ts)

    pts = np.vstack(parts_pts)
    tans = np.vstack(parts_tan)

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    return pts, tans, arc


# ============================================================================
#                          Diameter profile
# ============================================================================
def diameter_at_s(
    s: np.ndarray | float,
    d_start: float,
    d_main: float,
    d_end: float,
    transition_in: float,
    transition_out: float,
) -> np.ndarray:
    """Piecewise-linear diameter profile in s in [0, 1]."""
    s_arr = np.asarray(s, dtype=float)
    t_in = float(np.clip(transition_in, 0.0, 1.0))
    t_out = float(np.clip(transition_out, 0.0, 1.0))
    if t_in > t_out:
        raise ValueError("--transition-in-frac must be <= --transition-out-frac")

    out = np.full_like(s_arr, float(d_main), dtype=float)

    if t_in > 0.0:
        mask = s_arr <= t_in
        if np.any(mask):
            t = s_arr[mask] / max(t_in, 1e-12)
            out[mask] = float(d_start) + (float(d_main) - float(d_start)) * t

    if t_out < 1.0:
        mask = s_arr >= t_out
        if np.any(mask):
            t = (s_arr[mask] - t_out) / max(1.0 - t_out, 1e-12)
            out[mask] = float(d_main) + (float(d_end) - float(d_main)) * t

    return out


# ============================================================================
#                          Helical tube sampling
# ============================================================================
def _interpolate_along_polyline(
    polyline_pts: np.ndarray,
    polyline_tangents: np.ndarray,
    cumulative_arc: np.ndarray,
    arc_target: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """Linear interpolation of polyline position + tangent at a target arc length."""
    s_total = float(cumulative_arc[-1])
    s = float(np.clip(arc_target, 0.0, s_total))
    idx = int(np.searchsorted(cumulative_arc, s, side="left"))
    if idx <= 0:
        return polyline_pts[0].copy(), normalize(polyline_tangents[0])
    if idx >= len(cumulative_arc):
        return polyline_pts[-1].copy(), normalize(polyline_tangents[-1])
    seg_len = float(cumulative_arc[idx] - cumulative_arc[idx - 1])
    alpha = 0.0 if seg_len <= 1e-12 else (s - float(cumulative_arc[idx - 1])) / seg_len
    pt = (1.0 - alpha) * polyline_pts[idx - 1] + alpha * polyline_pts[idx]
    t = normalize((1.0 - alpha) * polyline_tangents[idx - 1] + alpha * polyline_tangents[idx])
    return pt, t


def build_tube_helix(
    centerline_pts: np.ndarray,
    centerline_tangents: np.ndarray,
    cumulative_arc: np.ndarray,
    diameter_start: float,
    diameter_main: float,
    diameter_end: float,
    transition_in_frac: float,
    transition_out_frac: float,
    layer_height: float,
    minor_segments_per_turn: int,
    phi0_deg: float,
    frame_flip_outplane_sign: float,
) -> TubePlan:
    """Sample the helix-on-tube path uniformly in centerline arc length."""
    if float(layer_height) <= 0.0:
        raise ValueError("--layer-height must be > 0")
    if int(minor_segments_per_turn) < 4:
        raise ValueError("--minor-segments-per-turn must be >= 4")
    for d_label, d_val in (("diameter-start", diameter_start), ("diameter-main", diameter_main), ("diameter-end", diameter_end)):
        if float(d_val) <= 0.0:
            raise ValueError(f"--{d_label} must be > 0")

    total_arc = float(cumulative_arc[-1])
    if total_arc <= 0.0:
        raise ValueError("Centerline has zero length.")

    minor_turns = total_arc / float(layer_height)
    n_samples = max(8, int(math.ceil(minor_turns * float(minor_segments_per_turn))))
    arc_targets = np.linspace(0.0, total_arc, n_samples + 1)

    s_norm = arc_targets / total_arc
    diameters = diameter_at_s(s_norm, diameter_start, diameter_main, diameter_end, transition_in_frac, transition_out_frac)

    phi0 = math.radians(float(phi0_deg))
    out_plane_axis = np.array([0.0, float(np.sign(frame_flip_outplane_sign) or 1.0), 0.0], dtype=float)

    tube_pts = np.zeros((len(arc_targets), 3), dtype=float)
    cl_pts_at = np.zeros_like(tube_pts)
    cl_tan_at = np.zeros_like(tube_pts)

    for i, arc_s in enumerate(arc_targets):
        cl_pt, cl_t = _interpolate_along_polyline(centerline_pts, centerline_tangents, cumulative_arc, float(arc_s))
        cl_pts_at[i] = cl_pt
        cl_tan_at[i] = cl_t

        n_in = in_plane_normal_xz(cl_t)
        b_out = out_plane_axis

        phi = phi0 + 2.0 * math.pi * float(arc_s) / float(layer_height)
        radius = 0.5 * float(diameters[i])
        offset = radius * (math.cos(phi) * n_in + math.sin(phi) * b_out)
        tube_pts[i] = cl_pt + offset

    return TubePlan(
        tube_points=tube_pts,
        centerline_tangents=cl_tan_at,
        centerline_points=cl_pts_at,
        arc_lengths=arc_targets,
        diameters=np.asarray(diameters, dtype=float),
        total_arc_length=float(total_arc),
        minor_turns=float(minor_turns),
    )


# ============================================================================
#                              G-code writer
# ============================================================================
def _fmt_axes(axes_vals: Sequence[Tuple[str, float]]) -> str:
    return " ".join(f"{ax}{val:.3f}" for ax, val in axes_vals)


class TubeGCodeWriter:
    """Streaming G-code writer for the sine-wave tube."""

    def __init__(
        self,
        fh,
        cal: Calibration,
        write_mode: str,
        c_deg: float,
        b_angle_bias_deg: float,
        bc_solve_samples: int,
        bbox_min: np.ndarray,
        bbox_max: np.ndarray,
        travel_feed: float,
        approach_feed: float,
        fine_approach_feed: float,
        print_feed: float,
        emit_extrusion: bool,
        extrusion_per_mm: float,
        extrusion_multiplier: float,
        pressure_offset_mm: float,
        pressure_advance_feed: float,
        pressure_retract_feed: float,
        preflow_dwell_ms: int,
        end_dwell_ms: int,
    ) -> None:
        self.fh = fh
        self.cal = cal
        self.write_mode = str(write_mode).strip().lower()
        if self.write_mode not in {"calibrated", "cartesian"}:
            raise ValueError("--write-mode must be calibrated or cartesian")
        self.calibrated = self.write_mode == "calibrated"
        self.c_deg = float(c_deg)
        self.b_angle_bias_deg = float(b_angle_bias_deg)
        self.bc_solve_samples = int(bc_solve_samples)
        self.bbox_min = np.asarray(bbox_min, dtype=float)
        self.bbox_max = np.asarray(bbox_max, dtype=float)
        self.travel_feed = float(travel_feed)
        self.approach_feed = float(approach_feed)
        self.fine_approach_feed = float(fine_approach_feed)
        self.print_feed = float(print_feed)
        self.emit_extrusion = bool(emit_extrusion)
        self.extrusion_per_mm = float(extrusion_per_mm)
        self.extrusion_multiplier = float(extrusion_multiplier)
        self.pressure_offset_mm = float(pressure_offset_mm)
        self.pressure_advance_feed = float(pressure_advance_feed)
        self.pressure_retract_feed = float(pressure_retract_feed)
        self.preflow_dwell_ms = int(preflow_dwell_ms)
        self.end_dwell_ms = int(end_dwell_ms)

        self.u_material_abs = 0.0
        self.pressure_charged = False
        self.cur_tip: Optional[np.ndarray] = None
        self.cur_stage: Optional[np.ndarray] = None
        self.cur_b_machine: float = 0.0
        self.cur_b_target_deg: float = 0.0
        self.cur_c: float = self.c_deg
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

    # -------- axis name shortcuts --------
    @property
    def x_axis(self) -> str: return self.cal.x_axis
    @property
    def y_axis(self) -> str: return self.cal.y_axis
    @property
    def z_axis(self) -> str: return self.cal.z_axis
    @property
    def b_axis(self) -> str: return self.cal.b_axis
    @property
    def c_axis(self) -> str: return self.cal.c_axis
    @property
    def u_axis(self) -> str: return self.cal.u_axis

    # -------- core helpers --------
    def b_machine_for_target(self, b_target_deg: float) -> float:
        b_target = float(np.clip(float(b_target_deg) + self.b_angle_bias_deg, 0.0, 180.0))
        key = round(b_target, 8)
        if key not in self._b_cache:
            if self.calibrated:
                self._b_cache[key] = solve_b_for_target_tip_angle(self.cal, key, self.bc_solve_samples)
            else:
                self._b_cache[key] = key
        return self._b_cache[key]

    def clamp_stage(self, p: np.ndarray, context: str) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        q = np.minimum(np.maximum(p_arr, self.bbox_min), self.bbox_max)
        if float(np.linalg.norm(q - p_arr)) > 1e-9:
            self.warnings.append(f"WARNING: clamped stage point during {context}: requested={p_arr.tolist()} clamped={q.tolist()}")
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

    def write_move(
        self,
        stage_xyz: np.ndarray,
        b_machine: float,
        b_target_deg: float,
        c_deg: float,
        feed: float,
        u_value: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> None:
        if comment:
            self.fh.write(f"; {comment}\n")
        axes: List[Tuple[str, float]] = [
            (self.x_axis, float(stage_xyz[0])),
            (self.y_axis, float(stage_xyz[1])),
            (self.z_axis, float(stage_xyz[2])),
            (self.b_axis, float(b_machine)),
            (self.c_axis, float(c_deg)),
        ]
        if u_value is not None:
            axes.append((self.u_axis, float(u_value)))
        self.fh.write(f"G1 {_fmt_axes(axes)} F{float(feed):.0f}\n")

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

    def move_to_tip(
        self,
        tip_xyz: np.ndarray,
        b_target_deg: float,
        c_deg: float,
        feed: float,
        comment: Optional[str] = None,
        extrude_from_current_tip: bool = False,
    ) -> None:
        tip = np.asarray(tip_xyz, dtype=float)
        prev_tip = None if self.cur_tip is None else self.cur_tip.copy()
        stage, b_machine = self.tip_to_stage(tip, b_target_deg, c_deg)
        u_val = None
        if extrude_from_current_tip and self.emit_extrusion:
            seg_len = 0.0 if prev_tip is None else float(np.linalg.norm(tip - prev_tip))
            self.total_print_mm += seg_len
            self.u_material_abs += self.extrusion_per_mm * self.extrusion_multiplier * seg_len
            u_val = self.u_cmd_actual()
        elif extrude_from_current_tip and not self.emit_extrusion:
            if prev_tip is not None:
                self.total_print_mm += float(np.linalg.norm(tip - prev_tip))
        else:
            if prev_tip is not None:
                self.total_travel_mm += float(np.linalg.norm(tip - prev_tip))
        self.write_move(stage, b_machine, b_target_deg, c_deg, feed, u_value=u_val, comment=comment)
        self.cur_tip = tip.copy()

    def pressure_preload_before_print(self) -> None:
        if not self.emit_extrusion or self.pressure_offset_mm <= 0.0 or self.pressure_charged:
            return
        self.pressure_charged = True
        self.fh.write("; pressure preload before tube print pass\n")
        self.fh.write(f"G1 {self.u_axis}{self.u_cmd_actual():.3f} F{self.pressure_advance_feed:.0f}\n")
        if self.preflow_dwell_ms > 0:
            self.fh.write(f"G4 P{self.preflow_dwell_ms}\n")

    def pressure_release_after_print(self) -> None:
        if not self.emit_extrusion or self.pressure_offset_mm <= 0.0 or not self.pressure_charged:
            return
        self.pressure_charged = False
        self.fh.write("; pressure release after tube print pass\n")
        self.fh.write(f"G1 {self.u_axis}{self.u_cmd_actual():.3f} F{self.pressure_retract_feed:.0f}\n")

    # -------- approach / exit --------
    def approach_start(
        self,
        start_tip: np.ndarray,
        start_b_target_deg: float,
        c_deg: float,
        safe_approach_z: float,
        travel_lift_z: float,
    ) -> None:
        start_tip = np.asarray(start_tip, dtype=float)
        start_stage, start_b_machine = self.tip_to_stage(start_tip, start_b_target_deg, c_deg)
        safe_stage = start_stage.copy()
        safe_stage[2] = float(safe_approach_z)

        if self.cur_stage is not None:
            lifted = np.asarray(self.cur_stage, dtype=float).copy()
            lifted[2] = float(safe_approach_z)
            self.write_move(
                self.clamp_stage(lifted, "approach lift"),
                self.cur_b_machine,
                self.cur_b_target_deg,
                self.cur_c,
                self.approach_feed,
                comment=f"lift to safe bath-exit height Z{float(safe_approach_z):.3f}",
            )

        self.write_move(
            self.clamp_stage(safe_stage, "approach safe"),
            start_b_machine,
            start_b_target_deg,
            c_deg,
            self.travel_feed,
            comment="set B/C and XY at safe approach height above tube start",
        )

        near_start_stage = start_stage.copy()
        near_start_stage[2] = min(float(safe_approach_z), float(start_stage[2]) + max(0.0, float(travel_lift_z)))
        if near_start_stage[2] > float(start_stage[2]) + 1e-9:
            self.write_move(
                self.clamp_stage(near_start_stage, "approach near"),
                start_b_machine,
                start_b_target_deg,
                c_deg,
                self.travel_feed,
                comment="move down in Z toward tube start",
            )

        self.write_move(
            self.clamp_stage(start_stage, "approach final"),
            start_b_machine,
            start_b_target_deg,
            c_deg,
            self.fine_approach_feed,
            comment="move down in Z directly to tube start",
        )
        self.cur_tip = start_tip.copy()

    def exit_to_safe_z(self, safe_approach_z: float) -> None:
        if self.cur_stage is None:
            return
        safe_stage = np.asarray(self.cur_stage, dtype=float).copy()
        safe_stage[2] = float(safe_approach_z)
        self.write_move(
            self.clamp_stage(safe_stage, "exit"),
            self.cur_b_machine,
            self.cur_b_target_deg,
            self.cur_c,
            self.approach_feed,
            comment=f"move out of bath to safe Z{float(safe_approach_z):.3f}",
        )

    # -------- main tube print --------
    def print_tube(self, plan: TubePlan) -> None:
        pts = plan.tube_points
        cl_tans = plan.centerline_tangents
        if len(pts) < 2:
            raise ValueError("Tube plan has fewer than 2 sample points.")

        b_targets = np.array([desired_physical_b_angle_from_tangent(t) for t in cl_tans], dtype=float)

        self.fh.write(
            "; TUBE_WRITE_START "
            f"sample_count={len(pts)} "
            f"total_centerline_arc={plan.total_arc_length:.6f} "
            f"minor_turns={plan.minor_turns:.6f} "
            f"diameter_start={float(plan.diameters[0]):.4f} "
            f"diameter_end={float(plan.diameters[-1]):.4f} "
            "tip_angle_convention=0_posZ_90_horizontal_180_negZ\n"
        )

        # First point: come down to it via approach (caller has already done that).
        self.pressure_preload_before_print()

        for i in range(1, len(pts)):
            self.move_to_tip(
                pts[i],
                float(b_targets[i]),
                float(self.c_deg),
                self.print_feed,
                extrude_from_current_tip=True,
            )

        self.pressure_release_after_print()
        if self.end_dwell_ms > 0:
            self.fh.write(f"G4 P{self.end_dwell_ms}\n")
        self.fh.write("; TUBE_WRITE_END\n")


# ============================================================================
#                          Top-level generation
# ============================================================================
def write_sine_wave_tube_gcode(args: argparse.Namespace) -> Dict[str, Any]:
    write_mode = str(args.write_mode).strip().lower()
    if write_mode not in {"calibrated", "cartesian"}:
        raise ValueError("--write-mode must be calibrated or cartesian")
    if write_mode == "calibrated" and not args.calibration:
        raise ValueError("--calibration is required when --write-mode calibrated")

    cal = load_calibration(args.calibration, offplane_y_sign=args.y_offplane_sign) if write_mode == "calibrated" else make_cartesian_calibration()
    cal.offplane_y_sign = float(args.y_offplane_sign)

    # 1) centerline
    centerline_pts, centerline_tans, arc_lengths = build_sine_wave_centerline(
        x_start=args.x_start,
        x_end=args.x_end,
        y=args.y,
        z_baseline=args.z_baseline,
        z_amplitude=args.z_amplitude,
        cycles=args.cycles,
        phase_deg=args.phase_deg,
        lead_in=args.lead_in,
        lead_out=args.lead_out,
        samples=args.centerline_samples,
    )

    # validate centerline against requested tip box
    z_lo = float(np.min(centerline_pts[:, 2]))
    z_hi = float(np.max(centerline_pts[:, 2]))
    if z_lo < float(args.z_min) - 1e-6 or z_hi > float(args.z_max) + 1e-6:
        raise ValueError(
            f"Centerline Z range [{z_lo:.3f}, {z_hi:.3f}] exceeds requested "
            f"[{float(args.z_min):.3f}, {float(args.z_max):.3f}]. Adjust amplitude / baseline / cycles."
        )

    # 2) helical tube
    plan = build_tube_helix(
        centerline_pts=centerline_pts,
        centerline_tangents=centerline_tans,
        cumulative_arc=arc_lengths,
        diameter_start=args.diameter_start,
        diameter_main=args.diameter_main,
        diameter_end=args.diameter_end,
        transition_in_frac=args.transition_in_frac,
        transition_out_frac=args.transition_out_frac,
        layer_height=args.layer_height,
        minor_segments_per_turn=args.minor_segments_per_turn,
        phi0_deg=args.phi0_deg,
        frame_flip_outplane_sign=args.frame_flip_outplane_sign,
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bbox_min = np.array([args.bbox_x_min, args.bbox_y_min, args.bbox_z_min], dtype=float)
    bbox_max = np.array([args.bbox_x_max, args.bbox_y_max, args.bbox_z_max], dtype=float)

    with out_path.open("w", encoding="utf-8") as fh:
        # ---- header ----
        fh.write("; Sine-wave tube in the XZ plane\n")
        fh.write("; Generated by sine_wave_xz_tube_generator.py\n")
        fh.write(f"; write_mode={write_mode}\n")
        fh.write(
            "; requested_tip_box "
            f"x=[{float(args.x_start):.3f},{float(args.x_end):.3f}] "
            f"y={float(args.y):.3f} "
            f"z=[{float(args.z_min):.3f},{float(args.z_max):.3f}]\n"
        )
        fh.write(
            "; centerline_wave "
            f"z_baseline={float(args.z_baseline):.3f} "
            f"z_amplitude={float(args.z_amplitude):.3f} "
            f"cycles={float(args.cycles):.6f} "
            f"phase_deg={float(args.phase_deg):.3f} "
            f"lead_in={float(args.lead_in):.3f} "
            f"lead_out={float(args.lead_out):.3f}\n"
        )
        fh.write(
            "; tube_diameter "
            f"start={float(args.diameter_start):.4f} "
            f"main={float(args.diameter_main):.4f} "
            f"end={float(args.diameter_end):.4f} "
            f"transition_in_frac={float(args.transition_in_frac):.4f} "
            f"transition_out_frac={float(args.transition_out_frac):.4f}\n"
        )
        fh.write(
            "; helical_winding "
            f"layer_height={float(args.layer_height):.4f} "
            f"minor_segments_per_turn={int(args.minor_segments_per_turn)} "
            f"phi0_deg={float(args.phi0_deg):.3f} "
            f"frame_flip_outplane_sign={float(args.frame_flip_outplane_sign):+.1f}\n"
        )
        fh.write(
            "; sampled "
            f"samples={len(plan.tube_points)} "
            f"total_centerline_arc={plan.total_arc_length:.6f} "
            f"minor_turns={plan.minor_turns:.6f}\n"
        )
        fh.write(f"; B-angle convention: 0=+Z (up), 90=horizontal, 180=-Z (down)\n")
        fh.write(f"; B follows centerline tangent (extrude tangent to centerline)\n")
        fh.write(f"; C held constant at {float(args.c_deg):.3f} deg\n")
        fh.write(f"; selected_fit_model = {cal.selected_fit_model or 'legacy-polynomial/cartesian'}\n")
        fh.write(f"; selected_offplane_fit_model = {cal.selected_offplane_fit_model or cal.selected_fit_model or 'legacy-polynomial/cartesian'}\n")
        fh.write(f"; active_phase = {cal.active_phase}\n")
        fh.write(f"; y_offplane_sign = {float(cal.offplane_y_sign):.1f}\n")
        fh.write(f"; {describe_model(cal.r_model, 'r')}\n")
        fh.write(f"; {describe_model(cal.z_model, 'z')}\n")
        fh.write(f"; {describe_model(cal.y_off_model, 'offplane_y')}\n")
        fh.write(f"; {describe_model(cal.tip_angle_model, 'tip_angle')}\n")
        fh.write(f"; axes: X->{cal.x_axis}, Y->{cal.y_axis}, Z->{cal.z_axis}, B->{cal.b_axis}, C->{cal.c_axis}, U->{cal.u_axis}\n")
        fh.write(f"; feeds: travel={args.travel_feed:.1f}, approach={args.approach_feed:.1f}, "
                 f"fine_approach={args.fine_approach_feed:.1f}, print={args.print_feed:.1f}\n")
        fh.write(f"; extrusion: emit={int(bool(args.emit_extrusion))} per_mm={args.extrusion_per_mm:.6f} "
                 f"multiplier={args.extrusion_multiplier:.6f} prime={args.prime_mm:.3f} "
                 f"pressure_offset={args.pressure_offset_mm:.3f}\n")

        fh.write("G21\n")
        fh.write("G90\n")
        if args.emit_extrusion:
            fh.write("M82\n")
            fh.write(f"G92 {cal.u_axis}0\n")
            if abs(float(args.prime_mm)) > 0.0:
                fh.write(f"G1 {cal.u_axis}{float(args.prime_mm):.3f} "
                         f"F{max(60.0, float(args.pressure_advance_feed)):.0f} ; prime material\n")

        # ---- writer ----
        writer = TubeGCodeWriter(
            fh=fh,
            cal=cal,
            write_mode=write_mode,
            c_deg=args.c_deg,
            b_angle_bias_deg=args.b_angle_bias_deg,
            bc_solve_samples=args.bc_solve_samples,
            bbox_min=bbox_min,
            bbox_max=bbox_max,
            travel_feed=args.travel_feed,
            approach_feed=args.approach_feed,
            fine_approach_feed=args.fine_approach_feed,
            print_feed=args.print_feed,
            emit_extrusion=args.emit_extrusion,
            extrusion_per_mm=args.extrusion_per_mm,
            extrusion_multiplier=args.extrusion_multiplier,
            pressure_offset_mm=args.pressure_offset_mm,
            pressure_advance_feed=args.pressure_advance_feed,
            pressure_retract_feed=args.pressure_retract_feed,
            preflow_dwell_ms=args.preflow_dwell_ms,
            end_dwell_ms=args.end_dwell_ms,
        )

        # ---- approach + print + exit ----
        start_tip = plan.tube_points[0]
        start_b_target = desired_physical_b_angle_from_tangent(plan.centerline_tangents[0])
        writer.approach_start(
            start_tip=start_tip,
            start_b_target_deg=start_b_target,
            c_deg=args.c_deg,
            safe_approach_z=args.safe_approach_z,
            travel_lift_z=args.travel_lift_z,
        )
        writer.print_tube(plan)
        writer.exit_to_safe_z(args.safe_approach_z)

        # ---- summary ----
        fh.write("; SUMMARY\n")
        fh.write(f"; total_printed_path_mm = {writer.total_print_mm:.6f}\n")
        fh.write(f"; total_nonextruding_tip_travel_mm = {writer.total_travel_mm:.6f}\n")
        fh.write(f"; final_U_material_abs = {writer.u_material_abs:.6f}\n")
        if np.all(np.isfinite(writer.stage_min)) and np.all(np.isfinite(writer.stage_max)):
            fh.write(f"; stage_min = ({writer.stage_min[0]:.6f}, {writer.stage_min[1]:.6f}, {writer.stage_min[2]:.6f})\n")
            fh.write(f"; stage_max = ({writer.stage_max[0]:.6f}, {writer.stage_max[1]:.6f}, {writer.stage_max[2]:.6f})\n")
        if np.isfinite(writer.b_machine_min) and np.isfinite(writer.b_machine_max):
            fh.write(f"; B_machine_range = [{writer.b_machine_min:.6f}, {writer.b_machine_max:.6f}]\n")
        if np.isfinite(writer.c_min) and np.isfinite(writer.c_max):
            fh.write(f"; C_range = [{writer.c_min:.6f}, {writer.c_max:.6f}]\n")
        for w in writer.warnings:
            fh.write(f"; {w}\n")
        fh.write("; End of file\n")

    # ---- in-memory summary returned to caller ----
    summary = {
        "out": str(out_path),
        "write_mode": write_mode,
        "centerline_samples_dense": int(args.centerline_samples),
        "tube_samples": int(len(plan.tube_points)),
        "minor_turns": float(plan.minor_turns),
        "total_centerline_arc_mm": float(plan.total_arc_length),
        "tip_x_range": (float(np.min(plan.tube_points[:, 0])), float(np.max(plan.tube_points[:, 0]))),
        "tip_y_range": (float(np.min(plan.tube_points[:, 1])), float(np.max(plan.tube_points[:, 1]))),
        "tip_z_range": (float(np.min(plan.tube_points[:, 2])), float(np.max(plan.tube_points[:, 2]))),
        "diameter_at_start": float(plan.diameters[0]),
        "diameter_at_end": float(plan.diameters[-1]),
        "diameter_main_actual_at_midpoint": float(plan.diameters[len(plan.diameters) // 2]),
        "stage_xyz_range": {
            "x": (float(writer.stage_min[0]), float(writer.stage_max[0])),
            "y": (float(writer.stage_min[1]), float(writer.stage_max[1])),
            "z": (float(writer.stage_min[2]), float(writer.stage_max[2])),
        },
        "b_machine_range": (float(writer.b_machine_min), float(writer.b_machine_max)),
        "c_range": (float(writer.c_min), float(writer.c_max)),
        "total_print_mm": float(writer.total_print_mm),
        "total_travel_mm": float(writer.total_travel_mm),
        "final_U_material_abs": float(writer.u_material_abs),
        "warnings": list(writer.warnings),
    }
    return summary


# ============================================================================
#                                    CLI
# ============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Generate calibrated or cartesian Duet/RRF G-code for a sine-wave TUBE in the XZ plane. "
            "The tip traces a helical winding around a sine-wave centerline at constant Y. "
            "Diameter varies linearly from --diameter-start to --diameter-main between s=0 and "
            "--transition-in-frac, holds at --diameter-main, then tapers linearly to --diameter-end "
            "between --transition-out-frac and s=1. B follows the centerline tangent (0=up, "
            "90=horizontal, 180=down); C is held constant (default 180)."
        )
    )

    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code file.")
    ap.add_argument("--calibration", default=None, help="Calibration JSON. Required for --write-mode calibrated.")
    ap.add_argument("--write-mode", choices=["calibrated", "cartesian"], default=DEFAULT_WRITE_MODE)
    ap.add_argument("--y-offplane-sign", type=float, default=DEFAULT_Y_OFFPLANE_SIGN, help="Multiplier on the calibration off-plane Y term.")

    # centerline
    ap.add_argument("--x-start", type=float, default=DEFAULT_X_START)
    ap.add_argument("--x-end", type=float, default=DEFAULT_X_END)
    ap.add_argument("--y", type=float, default=DEFAULT_Y)
    ap.add_argument("--z-min", type=float, default=DEFAULT_Z_MIN, help="Requested lower tip-space Z bound for the centerline.")
    ap.add_argument("--z-max", type=float, default=DEFAULT_Z_MAX, help="Requested upper tip-space Z bound for the centerline.")
    ap.add_argument("--z-baseline", type=float, default=DEFAULT_Z_BASELINE)
    ap.add_argument("--z-amplitude", type=float, default=DEFAULT_Z_AMPLITUDE)
    ap.add_argument("--cycles", type=float, default=DEFAULT_CYCLES)
    ap.add_argument("--phase-deg", type=float, default=DEFAULT_PHASE_DEG)
    ap.add_argument("--lead-in", type=float, default=DEFAULT_LEAD_IN)
    ap.add_argument("--lead-out", type=float, default=DEFAULT_LEAD_OUT)
    ap.add_argument("--centerline-samples", type=int, default=DEFAULT_CENTERLINE_SAMPLES,
                    help="Dense sample count for arc-length integration along the sine wave.")

    # tube diameter profile
    ap.add_argument("--diameter-start", type=float, default=DEFAULT_DIAMETER_START,
                    help="Tube diameter at the very start of the centerline (s=0).")
    ap.add_argument("--diameter-main", type=float, default=DEFAULT_DIAMETER_MAIN,
                    help="Tube diameter through the main body of the centerline.")
    ap.add_argument("--diameter-end", type=float, default=DEFAULT_DIAMETER_END,
                    help="Tube diameter at the very end of the centerline (s=1).")
    ap.add_argument("--transition-in-frac", type=float, default=DEFAULT_TRANSITION_IN_FRAC,
                    help="Normalized arc-length fraction (0..1) where the diameter has fully reached --diameter-main.")
    ap.add_argument("--transition-out-frac", type=float, default=DEFAULT_TRANSITION_OUT_FRAC,
                    help="Normalized arc-length fraction (0..1) where the diameter starts tapering toward --diameter-end.")

    # helical winding
    ap.add_argument("--layer-height", type=float, default=DEFAULT_LAYER_HEIGHT,
                    help="Centerline-axial advance per minor revolution of the helix.")
    ap.add_argument("--minor-segments-per-turn", type=int, default=DEFAULT_MINOR_SEGMENTS_PER_TURN,
                    help="Number of helix samples per minor turn around the tube cross-section.")
    ap.add_argument("--phi0-deg", type=float, default=DEFAULT_PHI0_DEG, help="Starting minor angle.")
    ap.add_argument("--frame-flip-outplane-sign", type=float, default=DEFAULT_FRAME_FLIP_OUTPLANE_SIGN,
                    help="Multiply the +Y out-of-plane axis by this sign to reverse helix handedness.")

    # orientation
    ap.add_argument("--c-deg", type=float, default=DEFAULT_C_DEG, help="Constant C angle. Default 180 keeps tool oriented in the XZ plane.")
    ap.add_argument("--b-angle-bias-deg", type=float, default=DEFAULT_B_ANGLE_BIAS_DEG,
                    help="Bias added to the centerline-tangent-derived B target before solving / emitting.")
    ap.add_argument("--bc-solve-samples", type=int, default=DEFAULT_BC_SOLVE_SAMPLES)

    # motion / safety
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED)
    ap.add_argument("--approach-feed", type=float, default=DEFAULT_APPROACH_FEED)
    ap.add_argument("--fine-approach-feed", type=float, default=DEFAULT_FINE_APPROACH_FEED)
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED)
    ap.add_argument("--travel-lift-z", type=float, default=DEFAULT_TRAVEL_LIFT_Z)
    ap.add_argument("--safe-approach-z", type=float, default=DEFAULT_SAFE_APPROACH_Z,
                    help="Absolute safe Z used to enter and exit the bath.")

    # extrusion / pressure
    ap.add_argument("--emit-extrusion", dest="emit_extrusion", action="store_true", default=DEFAULT_EMIT_EXTRUSION)
    ap.add_argument("--no-emit-extrusion", dest="emit_extrusion", action="store_false")
    ap.add_argument("--extrusion-per-mm", type=float, default=DEFAULT_EXTRUSION_PER_MM)
    ap.add_argument("--extrusion-multiplier", type=float, default=DEFAULT_EXTRUSION_MULTIPLIER)
    ap.add_argument("--prime-mm", type=float, default=DEFAULT_PRIME_MM)
    ap.add_argument("--pressure-offset-mm", type=float, default=DEFAULT_PRESSURE_OFFSET_MM)
    ap.add_argument("--pressure-advance-feed", type=float, default=DEFAULT_PRESSURE_ADVANCE_FEED)
    ap.add_argument("--pressure-retract-feed", type=float, default=DEFAULT_PRESSURE_RETRACT_FEED)
    ap.add_argument("--preflow-dwell-ms", type=int, default=DEFAULT_PREFLOW_DWELL_MS)
    ap.add_argument("--end-dwell-ms", type=int, default=DEFAULT_END_DWELL_MS)

    # bbox
    ap.add_argument("--bbox-x-min", type=float, default=-DEFAULT_BBOX)
    ap.add_argument("--bbox-x-max", type=float, default=+DEFAULT_BBOX)
    ap.add_argument("--bbox-y-min", type=float, default=-DEFAULT_BBOX)
    ap.add_argument("--bbox-y-max", type=float, default=+DEFAULT_BBOX)
    ap.add_argument("--bbox-z-min", type=float, default=-DEFAULT_BBOX)
    ap.add_argument("--bbox-z-max", type=float, default=+DEFAULT_BBOX)

    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    summary = write_sine_wave_tube_gcode(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()