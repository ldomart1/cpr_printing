#!/usr/bin/env python3
"""
King Cobra G-code generator.

Generates Duet/RRF G-code for a king-cobra shape built on the sine-wave
tube infrastructure from sine_wave_xz_tube_generator.py.

Cobra features
──────────────
1. Body XY oscillation
   Gentle lateral (Y) undulation of the centerline that fades to zero as
   the neck region begins, giving the body a natural serpentine look.

2. Neck boost  (cycles ~1.65–2.05 amplified)
   The sine-wave upstroke that would normally travel only z_amplitude higher
   is multiplied by --neck-boost-mult (default 3.8 ×).  After the peak the
   wave curls back naturally through cycles 2.05–2.55, just as in the
   unmodified script.

3. Hood / neck cross-section ellipse
   From --neck-ellipse-start-frac (~72 % arc length) to
   --neck-ellipse-peak-frac (~90 % arc length) the circular cross-section
   transitions smoothly to an ellipse whose Y (depth / lateral) semi-axis
   is --neck-ellipse-y-ratio × the in-plane-radial semi-axis.  This
   produces the flattened cobra-hood look when viewed along the spine.

4. Tongue
   Two thin flat prongs protrude from the head tip.  They share a short
   flat-ellipse body section, diverge by ±--tongue-fork-angle-deg in the
   lateral plane, and each tapers to a near-zero-diameter tip.
   The natural gap between the two prongs forms the forked-tongue slit.
   Printed as two separate passes after the main body.

All coordinate conventions, calibration handling, axis naming and G-code
formatting are identical to sine_wave_xz_tube_generator.py.

B-angle convention (same as original):
  0 deg  → tip points +Z (straight up)
  90 deg → tip is horizontal
  180 deg → tip points -Z (straight down)
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
# Defaults  (unchanged from original unless noted with *COBRA*)
# ============================================================================
DEFAULT_OUT = "gcode_generation/king_cobra.gcode"

DEFAULT_X_START  = 60.0
DEFAULT_X_END    = 120.0
DEFAULT_Y        = 52.0
DEFAULT_Z_MIN    = -175.0          # *COBRA* extended downward
DEFAULT_Z_MAX    = -80.0           # *COBRA* extended upward for raised neck
DEFAULT_Z_BASELINE = -135.0
DEFAULT_Z_AMPLITUDE = 6.5
DEFAULT_CYCLES   = 2.35
DEFAULT_PHASE_DEG = 0.0
DEFAULT_LEAD_IN  = 0.0
DEFAULT_LEAD_OUT = 0.0
DEFAULT_CENTERLINE_SAMPLES = 2001

DEFAULT_DIAMETER_START       = 1.0
DEFAULT_DIAMETER_MAIN        = 6.0
DEFAULT_DIAMETER_END         = 3.5   # *COBRA* head slightly wider than tail
DEFAULT_TRANSITION_IN_FRAC   = 0.10
DEFAULT_TRANSITION_OUT_FRAC  = 0.92

DEFAULT_LAYER_HEIGHT             = 0.6
DEFAULT_MINOR_SEGMENTS_PER_TURN  = 36
DEFAULT_PHI0_DEG                 = 0.0
DEFAULT_FRAME_FLIP_OUTPLANE_SIGN = 1.0

DEFAULT_WRITE_MODE         = "calibrated"
DEFAULT_C_DEG              = 180.0
DEFAULT_B_ANGLE_BIAS_DEG   = 0.0
DEFAULT_BC_SOLVE_SAMPLES   = 5001
DEFAULT_Y_OFFPLANE_SIGN    = -1.0

DEFAULT_TRAVEL_FEED        = 2000.0
DEFAULT_APPROACH_FEED      = 1200.0
DEFAULT_FINE_APPROACH_FEED = 150.0
DEFAULT_PRINT_FEED         = 400.0
DEFAULT_TRAVEL_LIFT_Z      = 8.0
DEFAULT_SAFE_APPROACH_Z    = -50.0

DEFAULT_EMIT_EXTRUSION       = True
DEFAULT_EXTRUSION_PER_MM     = 0.0015
DEFAULT_EXTRUSION_MULTIPLIER = 1.0
DEFAULT_PRIME_MM             = 0.2
DEFAULT_PRESSURE_OFFSET_MM   = 4.0
DEFAULT_PRESSURE_ADVANCE_FEED  = 120.0
DEFAULT_PRESSURE_RETRACT_FEED  = 240.0
DEFAULT_PREFLOW_DWELL_MS     = 500
DEFAULT_END_DWELL_MS         = 0
DEFAULT_BBOX                 = 1e9

# ---- Cobra-specific defaults ------------------------------------------------

# Body XY oscillation
DEFAULT_XY_OSC_AMPLITUDE = 0.95    # mm lateral (Y) oscillation amplitude
DEFAULT_XY_OSC_CYCLES    = 3.5    # number of full XY oscillation cycles
DEFAULT_XY_OSC_PHASE_DEG = 90.0   # phase so oscillation starts gently

# Neck boost  (amplified upstroke → cobra raises neck)
DEFAULT_NECK_BOOST_MULT        = 3.0   # amplitude multiplier at boost peak
DEFAULT_NECK_BOOST_START_CYCLE = 1.8  # wave-cycle number where boost begins
DEFAULT_NECK_BOOST_PEAK_CYCLE  = 1.9  # wave-cycle number at max boost; holds beyond

# Hood / neck cross-section ellipse
DEFAULT_NECK_ELLIPSE_Y_RATIO    = 3.0   # max Y:in-plane-radial ratio (Y longer)
DEFAULT_NECK_ELLIPSE_START_FRAC = 0.72  # normalised arc frac where ellipse starts
DEFAULT_NECK_ELLIPSE_PEAK_FRAC  = 0.90  # normalised arc frac where ellipse is max

# Tongue
DEFAULT_TONGUE_ENABLE           = True
DEFAULT_TONGUE_LENGTH           = 8.0    # total tongue length (mm)
DEFAULT_TONGUE_BASE_DIAM_Y      = 2.5    # lateral (Y) diameter at tongue base (mm)
DEFAULT_TONGUE_BASE_DIAM_Z      = 0.6    # height (thin dimension) at tongue base (mm)
DEFAULT_TONGUE_FORK_FRAC        = 0.50   # fraction of tongue length at which prongs split
DEFAULT_TONGUE_FORK_ANGLE_DEG   = 22.0   # half-angle of fork spread (degrees)
DEFAULT_TONGUE_TAPER_START_FRAC = 0.65   # fraction along prong where taper to tip begins
DEFAULT_TONGUE_TIP_DIAM         = 0.25   # prong tip diameter (mm)
DEFAULT_TONGUE_LAYER_HEIGHT     = 0.25   # helical layer height for tongue winding
DEFAULT_TONGUE_SEGS_PER_TURN    = 16     # helix segments per minor turn (tongue)
DEFAULT_TONGUE_EXTRUSION_MULT   = 0.45   # extrusion-multiplier scale for thin tongue


# ============================================================================
# Dataclasses
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
    """Sampled helical tube path in tip-space.  y_ratios optionally stores the
    Y:X cross-section ratio at each sample (informational; geometry is already
    encoded in tube_points)."""
    tube_points: np.ndarray             # (N,3) helix points in tip-space
    centerline_tangents: np.ndarray     # (N,3) unit centerline tangent
    centerline_points: np.ndarray       # (N,3) centerline point per helix sample
    arc_lengths: np.ndarray             # (N,) cumulative arc length
    diameters: np.ndarray               # (N,) diameter at each sample
    total_arc_length: float
    minor_turns: float
    y_ratios: Optional[np.ndarray] = None  # (N,) Y:X ratios; None → all 1.0


# ============================================================================
# Math / geometry helpers
# ============================================================================
def smootherstep(t: np.ndarray) -> np.ndarray:
    """6th-order smooth step (C2 continuous): 0 at t=0, 1 at t=1."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return t * t * t * (t * (t * 6.0 - 15.0) + 10.0)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    return arr / n if n > eps else np.zeros_like(arr)


def desired_physical_b_angle_from_tangent(tangent: np.ndarray) -> float:
    """B-angle from tangent: 0 → +Z, 90 → horizontal, 180 → -Z."""
    t = normalize(np.asarray(tangent, dtype=float))
    tz = float(np.clip(t[2], -1.0, 1.0))
    return float(math.degrees(math.acos(tz)))


def in_plane_normal_xz(tangent: np.ndarray) -> np.ndarray:
    """Unit normal in XZ plane rotated +90 deg from tangent about +Y.
    With tangent (tx, 0, tz) → (-tz, 0, tx)."""
    t = np.asarray(tangent, dtype=float)
    return normalize(np.array([-t[2], 0.0, t[0]], dtype=float))


def _compute_tangents_fd(pts: np.ndarray) -> np.ndarray:
    """Unit tangents via central (interior) and one-sided (endpoints) differences."""
    n = len(pts)
    raw = np.zeros_like(pts)
    if n >= 3:
        raw[1:-1] = pts[2:] - pts[:-2]
    if n >= 2:
        raw[0]  = pts[1]  - pts[0]
        raw[-1] = pts[-1] - pts[-2]
    else:
        raw[:] = np.array([1.0, 0.0, 0.0])
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return raw / norms


# ============================================================================
# Calibration loading  (identical to original)
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


def _select_named_model(
    models: Dict[str, Any], base_name: str, selected_fit_model: Optional[str]
) -> Optional[Dict[str, Any]]:
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
    x, y = x[order], y[order]
    if np.any(np.diff(x) <= 0):
        raise ValueError("PCHIP x_knots must be strictly increasing.")
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros_like(y)
    if x.size == 2:
        d[:] = delta[0]
    else:
        for k in range(1, x.size - 1):
            if delta[k-1] == 0.0 or delta[k] == 0.0 or np.sign(delta[k-1]) != np.sign(delta[k]):
                d[k] = 0.0
            else:
                w1 = 2.0*h[k] + h[k-1]; w2 = h[k] + 2.0*h[k-1]
                d[k] = (w1+w2) / (w1/delta[k-1] + w2/delta[k])
        d[0]  = _pchip_endpoint_slope(h[0],  h[1],  delta[0],  delta[1])
        d[-1] = _pchip_endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])
    flat = xq.reshape(-1)
    idx = np.clip(np.searchsorted(x, flat, side="right") - 1, 0, x.size - 2)
    x0, x1_ = x[idx], x[idx+1]
    y0, y1_ = y[idx], y[idx+1]
    h_i = x1_ - x0
    t = np.clip((flat - x0) / h_i, 0.0, 1.0)
    h00 = 2*t**3 - 3*t**2 + 1; h10 = t**3 - 2*t**2 + t
    h01 = -2*t**3 + 3*t**2;     h11 = t**3 - t**2
    yq = h00*y0 + h10*h_i*d[idx] + h01*y1_ + h11*h_i*d[idx+1]
    return yq.reshape(xq.shape)


def eval_model_spec(
    model_spec: Optional[Dict[str, Any]], u: Any, default_if_none: Optional[float] = None
) -> np.ndarray:
    if model_spec is None:
        if default_if_none is None:
            raise ValueError("Missing required calibration model.")
        return np.full_like(np.asarray(u, dtype=float), float(default_if_none), dtype=float)
    model_type = str(model_spec.get("model_type") or "").strip().lower()
    if model_type == "pchip":
        return pchip_eval(model_spec.get("x_knots"), model_spec.get("y_knots"), u)
    if model_type == "polynomial":
        return poly_eval(model_spec.get("coefficients", model_spec.get("coeffs")), u, default_if_none)
    raise ValueError(f"Unsupported calibration model_type: {model_type}")


def eval_pchip_with_linear_extrap(
    model_spec: Dict[str, Any], extrap_model_spec: Optional[Dict[str, Any]], b: Any
) -> np.ndarray:
    x_knots = np.asarray(model_spec.get("x_knots", []), dtype=float).reshape(-1)
    if x_knots.size == 0 or extrap_model_spec is None:
        return eval_model_spec(model_spec, b, default_if_none=0.0)
    b_arr = np.asarray(b, dtype=float)
    pchip_values = eval_model_spec(model_spec, b_arr, default_if_none=0.0)
    out = np.asarray(pchip_values, dtype=float).copy()
    x_min, x_max = float(np.min(x_knots)), float(np.max(x_knots))
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
    cubic    = data.get("cubic_coefficients", {}) or {}
    fit_models = data.get("fit_models", {}) or {}

    def _coeffs(models, *names):
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

    pr = np.asarray(cubic["r_coeffs"], dtype=float) if cubic.get("r_coeffs") is not None \
         else (_coeffs(fit_models, "r_cubic", "r_avg_cubic") or np.zeros(1))
    pz = np.asarray(cubic["z_coeffs"], dtype=float) if cubic.get("z_coeffs") is not None \
         else (_coeffs(fit_models, "z_cubic", "z_avg_cubic") or np.zeros(1))
    py_off = (np.asarray(cubic["offplane_y_coeffs"], dtype=float)
              if cubic.get("offplane_y_coeffs") is not None
              else _coeffs(fit_models, "offplane_y_avg_cubic", "offplane_y_cubic",
                           "offplane_y", "offplane_y_linear", "offplane_y_avg_linear"))
    pa = (np.asarray(cubic["tip_angle_coeffs"], dtype=float)
          if cubic.get("tip_angle_coeffs") is not None
          else _coeffs(fit_models, "tip_angle_cubic", "tip_angle_avg_cubic"))

    sel    = data.get("selected_fit_model")
    sel    = None if sel    is None else str(sel).strip().lower()
    sel_op = data.get("selected_offplane_fit_model")
    sel_op = None if sel_op is None else str(sel_op).strip().lower()
    active_phase = str(data.get("default_phase_for_legacy_access") or "pull").strip().lower()

    phase_models = data.get("fit_models_by_phase", {}) or {}
    apm = phase_models.get(active_phase) if isinstance(phase_models, dict) else None
    if not isinstance(apm, dict):
        apm = fit_models

    y_off_sel = sel_op or sel
    r_model   = _select_named_model(apm, "r",          sel)
    z_model   = _select_named_model(apm, "z",          sel)
    yo_model  = (_normalize_model_spec(apm.get("offplane_y_avg_cubic")) or
                 _select_named_model(apm, "offplane_y", y_off_sel))
    yoe_model = (_normalize_model_spec(apm.get("offplane_y_avg_linear")) or
                 _normalize_model_spec(apm.get("offplane_y_linear")) or
                 _normalize_model_spec(apm.get("offplane_y")))
    ta_model  = _select_named_model(apm, "tip_angle",  sel)

    motor   = data.get("motor_setup", {})
    duet    = data.get("duet_axis_mapping", {})
    b_range = motor.get("b_motor_position_range", [-5.4, 0.0])
    b_min, b_max = sorted(map(float, b_range))

    return Calibration(
        pr=pr, pz=pz, py_off=py_off, pa=pa,
        r_model=r_model, z_model=z_model,
        y_off_model=yo_model, y_off_extrap_model=yoe_model, tip_angle_model=ta_model,
        selected_fit_model=sel, selected_offplane_fit_model=sel_op,
        active_phase=active_phase,
        b_min=b_min, b_max=b_max,
        x_axis=str(duet.get("horizontal_axis")  or motor.get("horizontal_axis")  or "X"),
        y_axis=str(duet.get("depth_axis")        or motor.get("depth_axis")        or "Y"),
        z_axis=str(duet.get("vertical_axis")     or motor.get("vertical_axis")     or "Z"),
        b_axis=str(duet.get("pull_axis")         or motor.get("b_motor_axis")      or "B"),
        c_axis=str(duet.get("rotation_axis")     or motor.get("rotation_axis")     or "C"),
        u_axis=str(duet.get("extruder_axis")     or "U"),
        c_180_deg=float(motor.get("rotation_axis_180_deg", 180.0)),
        offplane_y_sign=float(offplane_y_sign),
    )


def make_cartesian_calibration() -> Calibration:
    return Calibration(
        pr=np.zeros(1), pz=np.zeros(1), py_off=np.zeros(1),
        pa=np.array([1.0, 0.0]),
        b_min=0.0, b_max=180.0,
        x_axis="X", y_axis="Y", z_axis="Z",
        b_axis="B", c_axis="C", u_axis="U",
        c_180_deg=180.0,
    )


def eval_r(cal: Calibration, b: Any) -> np.ndarray:
    return eval_model_spec(cal.r_model, b) if cal.r_model is not None else poly_eval(cal.pr, b)


def eval_z(cal: Calibration, b: Any) -> np.ndarray:
    return eval_model_spec(cal.z_model, b) if cal.z_model is not None else poly_eval(cal.pz, b)


def eval_offplane_y(cal: Calibration, b: Any) -> np.ndarray:
    if cal.y_off_model is not None:
        if str(cal.y_off_model.get("model_type","")).lower() == "pchip":
            vals = eval_pchip_with_linear_extrap(cal.y_off_model, cal.y_off_extrap_model, b)
        else:
            vals = eval_model_spec(cal.y_off_model, b, default_if_none=0.0)
    else:
        vals = poly_eval(cal.py_off, b, default_if_none=0.0)
    return float(cal.offplane_y_sign) * np.asarray(vals, dtype=float)


def eval_tip_angle_deg(cal: Calibration, b: Any) -> np.ndarray:
    if cal.tip_angle_model is not None:
        return eval_model_spec(cal.tip_angle_model, b)
    if cal.pa is None:
        raise ValueError("Calibration missing tip_angle_coeffs / tip_angle model.")
    return poly_eval(cal.pa, b)


def predict_tip_offset_xyz(cal: Calibration, b_machine: float, c_deg: float) -> np.ndarray:
    r = float(eval_r(cal, b_machine))
    z = float(eval_z(cal, b_machine))
    y_off = float(eval_offplane_y(cal, b_machine))
    c = math.radians(float(c_deg))
    return np.array([r*math.cos(c) - y_off*math.sin(c),
                     r*math.sin(c) + y_off*math.cos(c),
                     z], dtype=float)


def stage_xyz_for_tip(
    cal: Calibration, tip_xyz: np.ndarray, b_machine: float, c_deg: float
) -> np.ndarray:
    return np.asarray(tip_xyz, dtype=float) - predict_tip_offset_xyz(cal, b_machine, c_deg)


def solve_b_for_target_tip_angle(
    cal: Calibration, target_angle_deg: float,
    search_samples: int = DEFAULT_BC_SOLVE_SAMPLES,
) -> float:
    bb = np.linspace(float(cal.b_min), float(cal.b_max), int(max(101, search_samples)))
    aa = eval_tip_angle_deg(cal, bb) - float(target_angle_deg)
    i_best = int(np.argmin(np.abs(aa)))
    b_best = float(bb[i_best])

    sign_changes: List[Tuple[float, float, float]] = []
    for i in range(len(bb)-1):
        a0, a1 = float(aa[i]), float(aa[i+1])
        if a0 == 0.0:
            return float(bb[i])
        if a0 * a1 < 0.0:
            sign_changes.append((min(abs(a0), abs(a1)), float(bb[i]), float(bb[i+1])))

    if sign_changes:
        sign_changes.sort(key=lambda t: t[0])
        _, lo, hi = sign_changes[0]
        flo = float(eval_tip_angle_deg(cal, lo) - float(target_angle_deg))
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fmid = float(eval_tip_angle_deg(cal, mid) - float(target_angle_deg))
            if abs(fmid) < 1e-10:
                return float(mid)
            if flo * fmid <= 0.0:
                hi = mid
            else:
                lo, flo = mid, fmid
        return float(0.5 * (lo + hi))
    return b_best


def describe_model(model: Optional[Dict[str, Any]], fallback_name: str) -> str:
    if model is None:
        return f"{fallback_name}: legacy polynomial fallback"
    mt  = str(model.get("model_type") or "unknown").strip().lower()
    eq  = str(model.get("equation") or "").strip()
    xr  = model.get("fit_x_range")
    rng = (f", fit_x_range=[{float(xr[0]):.3f}, {float(xr[1]):.3f}]"
           if isinstance(xr, (list, tuple)) and len(xr) >= 2 else "")
    return f"{fallback_name}: {mt}{rng}{'; ' + eq if eq else ''}"


# ============================================================================
# Cobra centerline builder
# ============================================================================

def _neck_boost_array(
    cycle_nums: np.ndarray,
    start_cycle: float,
    peak_cycle: float,
    max_mult: float,
) -> np.ndarray:
    """
    Per-sample neck-boost amplitude multiplier.

    Below start_cycle  → 1.0 (no boost)
    start→peak cycle   → smooth ramp 1.0 → max_mult
    Above peak_cycle   → max_mult (holds; the sine curls back naturally)
    """
    t = np.clip(
        (cycle_nums - float(start_cycle)) / max(float(peak_cycle) - float(start_cycle), 1e-12),
        0.0, 1.0,
    )
    ramp = smootherstep(t)
    # Any sample below start_cycle has t=0, ramp=0, giving mult=1
    return 1.0 + (float(max_mult) - 1.0) * ramp


def build_cobra_centerline(
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
    # XY body oscillation
    xy_osc_amplitude: float = DEFAULT_XY_OSC_AMPLITUDE,
    xy_osc_cycles: float    = DEFAULT_XY_OSC_CYCLES,
    xy_osc_phase_deg: float = DEFAULT_XY_OSC_PHASE_DEG,
    # Neck boost
    neck_boost_mult: float        = DEFAULT_NECK_BOOST_MULT,
    neck_boost_start_cycle: float = DEFAULT_NECK_BOOST_START_CYCLE,
    neck_boost_peak_cycle: float  = DEFAULT_NECK_BOOST_PEAK_CYCLE,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build the cobra centerline polyline, unit tangents, and cumulative arc length.

    Returns
    -------
    pts   : (N, 3) float array
    tans  : (N, 3) unit tangent array
    arc   : (N,)  cumulative arc-length array
    """
    x0, x1 = float(x_start), float(x_end)
    if not x1 > x0:
        raise ValueError("--x-end must be > --x-start")
    li, lo = float(lead_in), float(lead_out)
    if li < 0.0 or lo < 0.0:
        raise ValueError("--lead-in and --lead-out must be >= 0")
    if li + lo >= x1 - x0:
        raise ValueError("--lead-in + --lead-out must be < total X span")
    if int(samples) < 11:
        raise ValueError("--centerline-samples must be >= 11")

    wave_x0 = x0 + li
    wave_x1 = x1 - lo
    total_span = x1 - x0
    wave_span  = wave_x1 - wave_x0

    n      = int(samples)
    xs     = np.linspace(x0, x1, n)
    s_glob = (xs - x0) / total_span          # 0 → 1 over full span

    # --- Z: boosted sine wave ---
    in_wave = (xs >= wave_x0) & (xs <= wave_x1)
    s_wave  = np.where(in_wave,
                       np.clip((xs - wave_x0) / max(wave_span, 1e-12), 0.0, 1.0),
                       0.0)
    cycle_nums = float(cycles) * s_wave
    boost = _neck_boost_array(cycle_nums, neck_boost_start_cycle, neck_boost_peak_cycle, neck_boost_mult)

    omega = 2.0 * math.pi * float(cycles)
    phase = math.radians(float(phase_deg))

    z_wave = float(z_baseline) + float(z_amplitude) * boost * np.sin(omega * s_wave + phase)
    zs     = np.where(in_wave, z_wave, float(z_baseline))

    # --- Y: gentle lateral oscillation that fades to zero as neck begins ---
    # The fade-out starts at the global-s equivalent of neck_boost_start_cycle
    neck_start_global = li / total_span + (float(neck_boost_start_cycle) / float(cycles)) * (wave_span / total_span)
    fade_width = max(1.0 - neck_start_global, 1e-3)
    fade = np.clip(1.0 - (s_glob - neck_start_global) / fade_width, 0.0, 1.0)

    osc_phase = math.radians(float(xy_osc_phase_deg))
    y_osc = float(xy_osc_amplitude) * np.sin(2.0 * math.pi * float(xy_osc_cycles) * s_glob + osc_phase)
    ys    = float(y) + y_osc * fade

    pts  = np.column_stack([xs, ys, zs])
    tans = _compute_tangents_fd(pts)
    segs = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    arc  = np.concatenate([[0.0], np.cumsum(segs)])
    return pts, tans, arc


# ============================================================================
# Diameter profile  (unchanged from original)
# ============================================================================
def diameter_at_s(
    s: Any,
    d_start: float,
    d_main: float,
    d_end: float,
    transition_in: float,
    transition_out: float,
) -> np.ndarray:
    s_arr = np.asarray(s, dtype=float)
    t_in  = float(np.clip(transition_in,  0.0, 1.0))
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
# Neck Y-ratio profile  (new for cobra)
# ============================================================================
def compute_neck_y_ratio_profile(
    s_norm: np.ndarray,
    neck_ellipse_start_frac: float,
    neck_ellipse_peak_frac: float,
    max_y_ratio: float,
) -> np.ndarray:
    """
    Smooth Y:X cross-section ratio profile along the normalised arc length.

    1.0 at s < neck_ellipse_start_frac  (circular)
    Smooth ramp to max_y_ratio up to neck_ellipse_peak_frac
    Holds at max_y_ratio beyond neck_ellipse_peak_frac
    """
    s_start = float(np.clip(neck_ellipse_start_frac, 0.0, 1.0))
    s_peak  = float(np.clip(neck_ellipse_peak_frac,  0.0, 1.0))
    if s_start >= s_peak:
        return np.ones_like(s_norm, dtype=float)
    t = np.clip((s_norm - s_start) / (s_peak - s_start), 0.0, 1.0)
    return 1.0 + (float(max_y_ratio) - 1.0) * smootherstep(t)


# ============================================================================
# Cobra tube helix builder  (extended from original to support y_ratio_profile)
# ============================================================================
def _interpolate_along_polyline(
    pts: np.ndarray, tangents: np.ndarray, arc: np.ndarray, arc_target: float
) -> Tuple[np.ndarray, np.ndarray]:
    s_total = float(arc[-1])
    s = float(np.clip(arc_target, 0.0, s_total))
    idx = int(np.searchsorted(arc, s, side="left"))
    if idx <= 0:
        return pts[0].copy(), normalize(tangents[0])
    if idx >= len(arc):
        return pts[-1].copy(), normalize(tangents[-1])
    seg_len = float(arc[idx] - arc[idx-1])
    alpha   = 0.0 if seg_len <= 1e-12 else (s - float(arc[idx-1])) / seg_len
    pt  = (1.0-alpha)*pts[idx-1]      + alpha*pts[idx]
    tan = normalize((1.0-alpha)*tangents[idx-1] + alpha*tangents[idx])
    return pt, tan


def build_cobra_tube_helix(
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
    y_ratio_profile: Optional[np.ndarray] = None,
) -> TubePlan:
    """
    Sample the helical cobra-tube path uniformly in centerline arc length.

    y_ratio_profile : (M,) array of Y:X ratios sampled at M equally spaced
                      normalised-arc positions, or None for a circular helix.
                      Values > 1 produce an ellipse taller/wider in the Y
                      (out-of-plane) direction.
    """
    if float(layer_height) <= 0:
        raise ValueError("--layer-height must be > 0")
    if int(minor_segments_per_turn) < 4:
        raise ValueError("--minor-segments-per-turn must be >= 4")

    total_arc   = float(cumulative_arc[-1])
    if total_arc <= 0:
        raise ValueError("Centerline has zero length.")
    minor_turns = total_arc / float(layer_height)
    n_samples   = max(8, int(math.ceil(minor_turns * float(minor_segments_per_turn))))
    arc_targets = np.linspace(0.0, total_arc, n_samples + 1)
    s_norm      = arc_targets / total_arc

    diameters = diameter_at_s(
        s_norm, diameter_start, diameter_main, diameter_end,
        transition_in_frac, transition_out_frac,
    )

    # Pre-compute per-sample y_ratio by interpolating the profile
    if y_ratio_profile is not None:
        profile_s = np.linspace(0.0, 1.0, len(y_ratio_profile))
        y_ratios_at = np.interp(s_norm, profile_s, y_ratio_profile.astype(float))
    else:
        y_ratios_at = np.ones(len(arc_targets), dtype=float)

    phi0      = math.radians(float(phi0_deg))
    b_out_sign = float(np.sign(frame_flip_outplane_sign) or 1.0)
    out_plane = np.array([0.0, b_out_sign, 0.0], dtype=float)

    tube_pts   = np.zeros((len(arc_targets), 3), dtype=float)
    cl_pts_at  = np.zeros_like(tube_pts)
    cl_tan_at  = np.zeros_like(tube_pts)

    for i, arc_s in enumerate(arc_targets):
        cl_pt, cl_t = _interpolate_along_polyline(
            centerline_pts, centerline_tangents, cumulative_arc, float(arc_s)
        )
        cl_pts_at[i] = cl_pt
        cl_tan_at[i] = cl_t

        n_in   = in_plane_normal_xz(cl_t)
        phi    = phi0 + 2.0 * math.pi * float(arc_s) / float(layer_height)
        radius = 0.5 * float(diameters[i])
        r_in   = radius
        r_out  = radius * float(y_ratios_at[i])
        offset = r_in * math.cos(phi) * n_in + r_out * math.sin(phi) * out_plane
        tube_pts[i] = cl_pt + offset

    return TubePlan(
        tube_points=tube_pts,
        centerline_tangents=cl_tan_at,
        centerline_points=cl_pts_at,
        arc_lengths=arc_targets,
        diameters=np.asarray(diameters, dtype=float),
        total_arc_length=float(total_arc),
        minor_turns=float(minor_turns),
        y_ratios=y_ratios_at,
    )


# ============================================================================
# Tongue builder  (new for cobra)
# ============================================================================

def _build_prong_helix(
    start_pt: np.ndarray,
    direction: np.ndarray,
    length: float,
    base_diam_z: float,         # thin (in-XZ-plane-normal) diameter at base
    base_diam_y: float,         # lateral (Y) diameter at base
    taper_start_frac: float,    # normalised position along prong where taper begins
    tip_diam: float,            # diameter at prong tip
    layer_height: float,
    segs_per_turn: int,
    phi0_deg: float,
    frame_flip_outplane_sign: float,
) -> TubePlan:
    """
    Build the helical surface for one tongue prong.

    The prong travels from start_pt in `direction` for `length` mm.
    Cross-section is elliptical: base_diam_z in the XZ in-plane direction,
    base_diam_y in the lateral (Y) direction.  Both dimensions taper together
    toward tip_diam at the prong tip.
    """
    direction = normalize(np.asarray(direction, dtype=float))
    length    = float(length)
    total_arc = length

    minor_turns = total_arc / float(layer_height)
    n_samples   = max(8, int(math.ceil(minor_turns * float(segs_per_turn))))
    arc_targets = np.linspace(0.0, total_arc, n_samples + 1)
    s_norm      = arc_targets / total_arc

    # Diameter of the thin (in-plane) cross-section axis, tapers to tip_diam
    t_taper = float(taper_start_frac)
    dz_at = np.where(
        s_norm < t_taper,
        float(base_diam_z),
        float(base_diam_z) + (float(tip_diam) - float(base_diam_z)) *
            np.clip((s_norm - t_taper) / max(1.0 - t_taper, 1e-12), 0.0, 1.0),
    )
    dz_at = np.clip(dz_at, float(tip_diam), float(base_diam_z))

    y_ratio = float(base_diam_y) / max(float(base_diam_z), 1e-12)

    phi0      = math.radians(float(phi0_deg))
    b_out_sign = float(np.sign(frame_flip_outplane_sign) or 1.0)
    out_plane  = np.array([0.0, b_out_sign, 0.0], dtype=float)
    n_in       = in_plane_normal_xz(direction)

    tube_pts  = np.zeros((len(arc_targets), 3), dtype=float)
    cl_pts_at = np.zeros_like(tube_pts)
    cl_tan_at = np.zeros_like(tube_pts)

    for i, arc_s in enumerate(arc_targets):
        cl_pt = np.asarray(start_pt, dtype=float) + arc_s * direction
        cl_pts_at[i] = cl_pt
        cl_tan_at[i] = direction

        phi    = phi0 + 2.0 * math.pi * float(arc_s) / float(layer_height)
        r_in   = 0.5 * float(dz_at[i])
        r_out  = r_in * y_ratio
        offset = r_in * math.cos(phi) * n_in + r_out * math.sin(phi) * out_plane
        tube_pts[i] = cl_pt + offset

    return TubePlan(
        tube_points=tube_pts,
        centerline_tangents=cl_tan_at,
        centerline_points=cl_pts_at,
        arc_lengths=arc_targets,
        diameters=dz_at,
        total_arc_length=float(total_arc),
        minor_turns=float(minor_turns),
        y_ratios=np.full(len(arc_targets), y_ratio, dtype=float),
    )


def build_tongue(
    cobra_head_pt: np.ndarray,
    cobra_head_tangent: np.ndarray,
    tongue_length: float,
    base_diam_y: float,
    base_diam_z: float,
    fork_frac: float,
    fork_angle_deg: float,
    taper_start_frac: float,
    tip_diam: float,
    layer_height: float,
    segs_per_turn: int,
    phi0_deg: float,
    frame_flip_outplane_sign: float,
) -> Tuple[TubePlan, TubePlan, TubePlan]:
    """
    Build three TubePlans for the tongue:
      body  – from the cobra head tip to the fork point
      prong_right – right prong from fork point to right tip
      prong_left  – left prong from fork point to left tip

    The fork spreads in the lateral (Y) direction.  The gap between the two
    prongs constitutes the tongue slit.

    Returns (body_plan, prong_right_plan, prong_left_plan).
    """
    head    = np.asarray(cobra_head_pt,      dtype=float)
    tang    = normalize(np.asarray(cobra_head_tangent, dtype=float))
    lat_dir = np.array([0.0, 1.0, 0.0], dtype=float)  # lateral = Y axis

    body_len  = float(tongue_length) * float(fork_frac)
    prong_len = float(tongue_length) * (1.0 - float(fork_frac))
    fork_pt   = head + body_len * tang

    fa = math.radians(float(fork_angle_deg))
    # Fork directions diverge in ±Y from the cobra tangent
    dir_right = normalize(tang + math.tan(fa) * lat_dir)
    dir_left  = normalize(tang - math.tan(fa) * lat_dir)

    body_plan = _build_prong_helix(
        start_pt=head,
        direction=tang,
        length=body_len,
        base_diam_z=base_diam_z,
        base_diam_y=base_diam_y,
        taper_start_frac=1.0,        # body does not taper
        tip_diam=base_diam_z,        # constant diameter along body
        layer_height=layer_height,
        segs_per_turn=segs_per_turn,
        phi0_deg=phi0_deg,
        frame_flip_outplane_sign=frame_flip_outplane_sign,
    )

    prong_right = _build_prong_helix(
        start_pt=fork_pt,
        direction=dir_right,
        length=prong_len,
        base_diam_z=base_diam_z,
        base_diam_y=base_diam_y,
        taper_start_frac=taper_start_frac,
        tip_diam=tip_diam,
        layer_height=layer_height,
        segs_per_turn=segs_per_turn,
        phi0_deg=phi0_deg,
        frame_flip_outplane_sign=frame_flip_outplane_sign,
    )

    prong_left = _build_prong_helix(
        start_pt=fork_pt,
        direction=dir_left,
        length=prong_len,
        base_diam_z=base_diam_z,
        base_diam_y=base_diam_y,
        taper_start_frac=taper_start_frac,
        tip_diam=tip_diam,
        layer_height=layer_height,
        segs_per_turn=segs_per_turn,
        phi0_deg=phi0_deg,
        frame_flip_outplane_sign=frame_flip_outplane_sign,
    )

    return body_plan, prong_right, prong_left


# ============================================================================
# G-code writer  (same as original, with extrusion_scale support added)
# ============================================================================
def _fmt_axes(axes_vals: Sequence[Tuple[str, float]]) -> str:
    return " ".join(f"{ax}{val:.3f}" for ax, val in axes_vals)


class CobraGCodeWriter:
    """Streaming G-code writer for the king-cobra shape."""

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
        self.c_deg                 = float(c_deg)
        self.b_angle_bias_deg      = float(b_angle_bias_deg)
        self.bc_solve_samples      = int(bc_solve_samples)
        self.bbox_min              = np.asarray(bbox_min, dtype=float)
        self.bbox_max              = np.asarray(bbox_max, dtype=float)
        self.travel_feed           = float(travel_feed)
        self.approach_feed         = float(approach_feed)
        self.fine_approach_feed    = float(fine_approach_feed)
        self.print_feed            = float(print_feed)
        self.emit_extrusion        = bool(emit_extrusion)
        self.extrusion_per_mm      = float(extrusion_per_mm)
        self.extrusion_multiplier  = float(extrusion_multiplier)
        self.pressure_offset_mm    = float(pressure_offset_mm)
        self.pressure_advance_feed = float(pressure_advance_feed)
        self.pressure_retract_feed = float(pressure_retract_feed)
        self.preflow_dwell_ms      = int(preflow_dwell_ms)
        self.end_dwell_ms          = int(end_dwell_ms)

        self.u_material_abs    = 0.0
        self.pressure_charged  = False
        self.cur_tip: Optional[np.ndarray] = None
        self.cur_stage: Optional[np.ndarray] = None
        self.cur_b_machine     = 0.0
        self.cur_b_target_deg  = 0.0
        self.cur_c             = self.c_deg
        self.stage_min         = np.array([np.inf, np.inf, np.inf])
        self.stage_max         = np.array([-np.inf, -np.inf, -np.inf])
        self.b_machine_min     = np.inf
        self.b_machine_max     = -np.inf
        self.c_min             = np.inf
        self.c_max             = -np.inf
        self.total_print_mm    = 0.0
        self.total_travel_mm   = 0.0
        self.warnings: List[str] = []
        self._b_cache: Dict[float, float] = {}

    # axis shortcuts
    @property
    def x_axis(self): return self.cal.x_axis
    @property
    def y_axis(self): return self.cal.y_axis
    @property
    def z_axis(self): return self.cal.z_axis
    @property
    def b_axis(self): return self.cal.b_axis
    @property
    def c_axis(self): return self.cal.c_axis
    @property
    def u_axis(self): return self.cal.u_axis

    def b_machine_for_target(self, b_target_deg: float) -> float:
        b_target = float(np.clip(b_target_deg + self.b_angle_bias_deg, 0.0, 180.0))
        key = round(b_target, 8)
        if key not in self._b_cache:
            self._b_cache[key] = (
                solve_b_for_target_tip_angle(self.cal, key, self.bc_solve_samples)
                if self.calibrated else key
            )
        return self._b_cache[key]

    def clamp_stage(self, p: np.ndarray, context: str) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        q = np.minimum(np.maximum(p_arr, self.bbox_min), self.bbox_max)
        if float(np.linalg.norm(q - p_arr)) > 1e-9:
            self.warnings.append(
                f"WARNING: clamped during {context}: "
                f"requested={p_arr.tolist()} clamped={q.tolist()}"
            )
        return q

    def tip_to_stage(
        self, tip_xyz: np.ndarray, b_target_deg: float, c_deg: float
    ) -> Tuple[np.ndarray, float]:
        b_machine = self.b_machine_for_target(b_target_deg)
        stage = (stage_xyz_for_tip(self.cal, tip_xyz, b_machine, c_deg)
                 if self.calibrated else np.asarray(tip_xyz, dtype=float))
        return self.clamp_stage(stage, "tip_to_stage"), b_machine

    def write_move(
        self,
        stage_xyz: np.ndarray,
        b_machine: float,
        b_target_deg: float,
        c_deg: float,
        feed: float,
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
        self.fh.write(f"G1 {_fmt_axes(axes)} F{float(feed):.0f}\n")
        self.cur_stage       = np.asarray(stage_xyz, dtype=float).copy()
        self.cur_b_machine   = float(b_machine)
        self.cur_b_target_deg = float(b_target_deg)
        self.cur_c           = float(c_deg)
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
        extrude: bool = False,
        extrusion_scale: float = 1.0,
    ) -> None:
        tip  = np.asarray(tip_xyz, dtype=float)
        prev = None if self.cur_tip is None else self.cur_tip.copy()
        stage, b_machine = self.tip_to_stage(tip, b_target_deg, c_deg)
        if extrude and self.emit_extrusion:
            seg = 0.0 if prev is None else float(np.linalg.norm(tip - prev))
            self.total_print_mm += seg
            self.u_material_abs += (self.extrusion_per_mm
                                    * self.extrusion_multiplier
                                    * extrusion_scale
                                    * seg)
        elif extrude:
            if prev is not None:
                self.total_print_mm += float(np.linalg.norm(tip - prev))
        else:
            if prev is not None:
                self.total_travel_mm += float(np.linalg.norm(tip - prev))
        self.write_move(stage, b_machine, b_target_deg, c_deg, feed, comment=comment)
        self.cur_tip = tip.copy()

    def pressure_preload(self) -> None:
        if not self.emit_extrusion or self.pressure_charged:
            return
        self.pressure_charged = True
        self.fh.write("; pressure preload\n")
        self.fh.write("M42 P0 S1\n")
        if self.preflow_dwell_ms > 0:
            self.fh.write(f"G4 P{self.preflow_dwell_ms}\n")

    def pressure_release(self) -> None:
        if not self.emit_extrusion or not self.pressure_charged:
            return
        self.pressure_charged = False
        self.fh.write("; pressure release\n")
        self.fh.write("M42 P0 S0\n")

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
        safe_stage = start_stage.copy(); safe_stage[2] = float(safe_approach_z)

        if self.cur_stage is not None:
            lifted = self.cur_stage.copy(); lifted[2] = float(safe_approach_z)
            self.write_move(
                self.clamp_stage(lifted, "approach lift"),
                self.cur_b_machine, self.cur_b_target_deg, self.cur_c,
                self.approach_feed, comment=f"lift to safe Z{safe_approach_z:.3f}",
            )
        self.write_move(
            self.clamp_stage(safe_stage, "approach safe"),
            start_b_machine, start_b_target_deg, c_deg,
            self.travel_feed, comment="set B/C, move to XY above start",
        )
        near = start_stage.copy()
        near[2] = min(float(safe_approach_z), float(start_stage[2]) + max(0.0, float(travel_lift_z)))
        if near[2] > float(start_stage[2]) + 1e-9:
            self.write_move(
                self.clamp_stage(near, "approach near"),
                start_b_machine, start_b_target_deg, c_deg,
                self.travel_feed, comment="descend toward start",
            )
        self.write_move(
            self.clamp_stage(start_stage, "approach final"),
            start_b_machine, start_b_target_deg, c_deg,
            self.fine_approach_feed, comment="fine approach to start",
        )
        self.cur_tip = start_tip.copy()

    def exit_to_safe_z(self, safe_approach_z: float) -> None:
        if self.cur_stage is None:
            return
        safe = self.cur_stage.copy(); safe[2] = float(safe_approach_z)
        self.write_move(
            self.clamp_stage(safe, "exit"),
            self.cur_b_machine, self.cur_b_target_deg, self.cur_c,
            self.approach_feed, comment=f"exit to safe Z{safe_approach_z:.3f}",
        )

    def print_tube(self, plan: TubePlan, extrusion_scale: float = 1.0, label: str = "tube") -> None:
        """Follow `plan.tube_points`, extruding at each step."""
        pts   = plan.tube_points
        tans  = plan.centerline_tangents
        if len(pts) < 2:
            raise ValueError(f"TubePlan '{label}' has fewer than 2 sample points.")

        b_targets = np.array(
            [desired_physical_b_angle_from_tangent(t) for t in tans], dtype=float
        )
        self.fh.write(
            f"; {label.upper()}_START "
            f"samples={len(pts)} arc={plan.total_arc_length:.4f} "
            f"turns={plan.minor_turns:.4f}\n"
        )
        self.pressure_preload()
        for i in range(1, len(pts)):
            self.move_to_tip(
                pts[i], float(b_targets[i]), float(self.c_deg),
                self.print_feed, extrude=True, extrusion_scale=extrusion_scale,
            )
        self.pressure_release()
        if self.end_dwell_ms > 0:
            self.fh.write(f"G4 P{self.end_dwell_ms}\n")
        self.fh.write(f"; {label.upper()}_END\n")


# ============================================================================
# Top-level generation
# ============================================================================
def write_cobra_gcode(args: argparse.Namespace) -> Dict[str, Any]:
    write_mode = str(args.write_mode).strip().lower()
    if write_mode not in {"calibrated", "cartesian"}:
        raise ValueError("--write-mode must be calibrated or cartesian")
    if write_mode == "calibrated" and not args.calibration:
        raise ValueError("--calibration required when --write-mode calibrated")

    cal = (load_calibration(args.calibration, offplane_y_sign=args.y_offplane_sign)
           if write_mode == "calibrated" else make_cartesian_calibration())
    cal.offplane_y_sign = float(args.y_offplane_sign)

    # ── 1. Cobra centerline ──────────────────────────────────────────────────
    centerline_pts, centerline_tans, arc_lengths = build_cobra_centerline(
        x_start=args.x_start, x_end=args.x_end, y=args.y,
        z_baseline=args.z_baseline, z_amplitude=args.z_amplitude,
        cycles=args.cycles, phase_deg=args.phase_deg,
        lead_in=args.lead_in, lead_out=args.lead_out,
        samples=args.centerline_samples,
        xy_osc_amplitude=args.xy_osc_amplitude,
        xy_osc_cycles=args.xy_osc_cycles,
        xy_osc_phase_deg=args.xy_osc_phase_deg,
        neck_boost_mult=args.neck_boost_mult,
        neck_boost_start_cycle=args.neck_boost_start_cycle,
        neck_boost_peak_cycle=args.neck_boost_peak_cycle,
    )

    # Warn (not error) if cobra neck exceeds original z_min/max
    z_lo = float(np.min(centerline_pts[:, 2]))
    z_hi = float(np.max(centerline_pts[:, 2]))
    z_warnings: List[str] = []
    if z_lo < float(args.z_min) - 1e-6:
        z_warnings.append(
            f"Centerline Z min {z_lo:.3f} < requested z_min {args.z_min:.3f}"
        )
    if z_hi > float(args.z_max) + 1e-6:
        z_warnings.append(
            f"Centerline Z max {z_hi:.3f} > requested z_max {args.z_max:.3f}"
        )
    for w in z_warnings:
        print(f"WARNING: {w}")

    # ── 2. Neck y-ratio profile ──────────────────────────────────────────────
    s_norm_cl = arc_lengths / max(float(arc_lengths[-1]), 1e-12)
    y_ratio_profile = compute_neck_y_ratio_profile(
        s_norm_cl,
        neck_ellipse_start_frac=args.neck_ellipse_start_frac,
        neck_ellipse_peak_frac=args.neck_ellipse_peak_frac,
        max_y_ratio=args.neck_ellipse_y_ratio,
    )

    # ── 3. Cobra body helix ──────────────────────────────────────────────────
    body_plan = build_cobra_tube_helix(
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
        y_ratio_profile=y_ratio_profile,
    )

    # ── 4. Tongue ─────────────────────────────────────────────────────────────
    tongue_plans: Optional[Tuple[TubePlan, TubePlan, TubePlan]] = None
    if args.tongue_enable:
        head_pt  = centerline_pts[-1]
        head_tan = centerline_tans[-1]
        tongue_plans = build_tongue(
            cobra_head_pt=head_pt,
            cobra_head_tangent=head_tan,
            tongue_length=args.tongue_length,
            base_diam_y=args.tongue_base_diam_y,
            base_diam_z=args.tongue_base_diam_z,
            fork_frac=args.tongue_fork_frac,
            fork_angle_deg=args.tongue_fork_angle_deg,
            taper_start_frac=args.tongue_taper_start_frac,
            tip_diam=args.tongue_tip_diam,
            layer_height=args.tongue_layer_height,
            segs_per_turn=args.tongue_segs_per_turn,
            phi0_deg=args.phi0_deg,
            frame_flip_outplane_sign=args.frame_flip_outplane_sign,
        )

    # ── 5. Write G-code ───────────────────────────────────────────────────────
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bbox_min = np.array([args.bbox_x_min, args.bbox_y_min, args.bbox_z_min], dtype=float)
    bbox_max = np.array([args.bbox_x_max, args.bbox_y_max, args.bbox_z_max], dtype=float)

    with out_path.open("w", encoding="utf-8") as fh:
        # ── Header ────────────────────────────────────────────────────────────
        fh.write("; King Cobra G-code\n")
        fh.write("; Generated by king_cobra_gcode_generator.py\n")
        fh.write(f"; write_mode={write_mode}\n")
        fh.write(
            "; centerline_wave "
            f"z_baseline={args.z_baseline:.3f} z_amplitude={args.z_amplitude:.3f} "
            f"cycles={args.cycles:.4f} phase_deg={args.phase_deg:.3f}\n"
        )
        fh.write(
            "; neck_boost "
            f"mult={args.neck_boost_mult:.3f} "
            f"start_cycle={args.neck_boost_start_cycle:.3f} "
            f"peak_cycle={args.neck_boost_peak_cycle:.3f}\n"
        )
        fh.write(
            "; neck_ellipse "
            f"y_ratio={args.neck_ellipse_y_ratio:.3f} "
            f"start_frac={args.neck_ellipse_start_frac:.3f} "
            f"peak_frac={args.neck_ellipse_peak_frac:.3f}\n"
        )
        fh.write(
            "; xy_oscillation "
            f"amplitude={args.xy_osc_amplitude:.3f} "
            f"cycles={args.xy_osc_cycles:.3f} "
            f"phase_deg={args.xy_osc_phase_deg:.3f}\n"
        )
        fh.write(
            "; tube_diameter "
            f"start={args.diameter_start:.4f} main={args.diameter_main:.4f} "
            f"end={args.diameter_end:.4f}\n"
        )
        fh.write(
            "; helical_winding "
            f"layer_height={args.layer_height:.4f} "
            f"segs_per_turn={args.minor_segments_per_turn}\n"
        )
        fh.write(
            "; sampled body "
            f"samples={len(body_plan.tube_points)} "
            f"arc={body_plan.total_arc_length:.4f} "
            f"turns={body_plan.minor_turns:.4f}\n"
        )
        if tongue_plans:
            tb, tp_r, tp_l = tongue_plans
            fh.write(
                f"; tongue length={args.tongue_length:.3f} "
                f"fork_frac={args.tongue_fork_frac:.3f} "
                f"fork_angle_deg={args.tongue_fork_angle_deg:.3f} "
                f"diam_y={args.tongue_base_diam_y:.3f} "
                f"diam_z={args.tongue_base_diam_z:.3f}\n"
            )
        fh.write(f"; selected_fit_model = {cal.selected_fit_model or 'legacy/cartesian'}\n")
        fh.write(f"; active_phase = {cal.active_phase}\n")
        fh.write(f"; {describe_model(cal.r_model, 'r')}\n")
        fh.write(f"; {describe_model(cal.z_model, 'z')}\n")
        fh.write(f"; {describe_model(cal.y_off_model, 'offplane_y')}\n")
        fh.write(f"; {describe_model(cal.tip_angle_model, 'tip_angle')}\n")
        fh.write(
            f"; axes X→{cal.x_axis} Y→{cal.y_axis} Z→{cal.z_axis} "
            f"B→{cal.b_axis} C→{cal.c_axis} U→{cal.u_axis}\n"
        )
        for w in z_warnings:
            fh.write(f"; WARNING: {w}\n")

        fh.write("G21\nG90\n")
        if args.emit_extrusion:
            fh.write("M42 P0 S0\n")

        writer = CobraGCodeWriter(
            fh=fh, cal=cal, write_mode=write_mode,
            c_deg=args.c_deg, b_angle_bias_deg=args.b_angle_bias_deg,
            bc_solve_samples=args.bc_solve_samples,
            bbox_min=bbox_min, bbox_max=bbox_max,
            travel_feed=args.travel_feed, approach_feed=args.approach_feed,
            fine_approach_feed=args.fine_approach_feed, print_feed=args.print_feed,
            emit_extrusion=args.emit_extrusion,
            extrusion_per_mm=args.extrusion_per_mm,
            extrusion_multiplier=args.extrusion_multiplier,
            pressure_offset_mm=args.pressure_offset_mm,
            pressure_advance_feed=args.pressure_advance_feed,
            pressure_retract_feed=args.pressure_retract_feed,
            preflow_dwell_ms=args.preflow_dwell_ms,
            end_dwell_ms=args.end_dwell_ms,
        )

        # ── Body pass ─────────────────────────────────────────────────────────
        fh.write("\n; ===== COBRA BODY =====\n")
        writer.approach_start(
            start_tip=body_plan.tube_points[0],
            start_b_target_deg=desired_physical_b_angle_from_tangent(
                body_plan.centerline_tangents[0]
            ),
            c_deg=args.c_deg,
            safe_approach_z=args.safe_approach_z,
            travel_lift_z=args.travel_lift_z,
        )
        writer.print_tube(body_plan, label="cobra_body")
        writer.exit_to_safe_z(args.safe_approach_z)

        # ── Tongue passes ─────────────────────────────────────────────────────
        if tongue_plans:
            tongue_body_plan, prong_r_plan, prong_l_plan = tongue_plans
            esc = args.tongue_extrusion_mult  # extrusion scale for thin tongue

            # -- tongue body --
            fh.write("\n; ===== TONGUE BODY =====\n")
            writer.approach_start(
                start_tip=tongue_body_plan.tube_points[0],
                start_b_target_deg=desired_physical_b_angle_from_tangent(
                    tongue_body_plan.centerline_tangents[0]
                ),
                c_deg=args.c_deg,
                safe_approach_z=args.safe_approach_z,
                travel_lift_z=args.travel_lift_z,
            )
            writer.print_tube(tongue_body_plan, extrusion_scale=esc, label="tongue_body")
            writer.exit_to_safe_z(args.safe_approach_z)

            # -- right prong --
            fh.write("\n; ===== TONGUE PRONG RIGHT =====\n")
            writer.approach_start(
                start_tip=prong_r_plan.tube_points[0],
                start_b_target_deg=desired_physical_b_angle_from_tangent(
                    prong_r_plan.centerline_tangents[0]
                ),
                c_deg=args.c_deg,
                safe_approach_z=args.safe_approach_z,
                travel_lift_z=args.travel_lift_z,
            )
            writer.print_tube(prong_r_plan, extrusion_scale=esc, label="tongue_prong_right")
            writer.exit_to_safe_z(args.safe_approach_z)

            # -- left prong --
            fh.write("\n; ===== TONGUE PRONG LEFT =====\n")
            writer.approach_start(
                start_tip=prong_l_plan.tube_points[0],
                start_b_target_deg=desired_physical_b_angle_from_tangent(
                    prong_l_plan.centerline_tangents[0]
                ),
                c_deg=args.c_deg,
                safe_approach_z=args.safe_approach_z,
                travel_lift_z=args.travel_lift_z,
            )
            writer.print_tube(prong_l_plan, extrusion_scale=esc, label="tongue_prong_left")
            writer.exit_to_safe_z(args.safe_approach_z)

        # ── Summary ───────────────────────────────────────────────────────────
        fh.write("\n; SUMMARY\n")
        fh.write(f"; total_printed_path_mm = {writer.total_print_mm:.6f}\n")
        fh.write(f"; total_travel_mm = {writer.total_travel_mm:.6f}\n")
        fh.write(f"; final_U_material_abs = {writer.u_material_abs:.6f}\n")
        if np.all(np.isfinite(writer.stage_min)):
            fh.write(
                f"; stage_min = ({writer.stage_min[0]:.4f}, "
                f"{writer.stage_min[1]:.4f}, {writer.stage_min[2]:.4f})\n"
            )
            fh.write(
                f"; stage_max = ({writer.stage_max[0]:.4f}, "
                f"{writer.stage_max[1]:.4f}, {writer.stage_max[2]:.4f})\n"
            )
        if np.isfinite(writer.b_machine_min):
            fh.write(
                f"; B_machine_range = [{writer.b_machine_min:.4f}, "
                f"{writer.b_machine_max:.4f}]\n"
            )
        for w in writer.warnings:
            fh.write(f"; {w}\n")
        fh.write("; End of file\n")

    # ── Return summary dict ───────────────────────────────────────────────────
    summary: Dict[str, Any] = {
        "out": str(out_path),
        "write_mode": write_mode,
        "cobra_body": {
            "tube_samples": int(len(body_plan.tube_points)),
            "minor_turns": float(body_plan.minor_turns),
            "total_centerline_arc_mm": float(body_plan.total_arc_length),
            "tip_x_range": (float(np.min(body_plan.tube_points[:, 0])),
                            float(np.max(body_plan.tube_points[:, 0]))),
            "tip_y_range": (float(np.min(body_plan.tube_points[:, 1])),
                            float(np.max(body_plan.tube_points[:, 1]))),
            "tip_z_range": (float(np.min(body_plan.tube_points[:, 2])),
                            float(np.max(body_plan.tube_points[:, 2]))),
            "diameter_at_start": float(body_plan.diameters[0]),
            "diameter_at_end":   float(body_plan.diameters[-1]),
            "y_ratio_at_start":  float(body_plan.y_ratios[0]) if body_plan.y_ratios is not None else 1.0,
            "y_ratio_at_end":    float(body_plan.y_ratios[-1]) if body_plan.y_ratios is not None else 1.0,
        },
        "centerline_z_range": (float(z_lo), float(z_hi)),
        "stage_xyz_range": {
            "x": (float(writer.stage_min[0]), float(writer.stage_max[0])),
            "y": (float(writer.stage_min[1]), float(writer.stage_max[1])),
            "z": (float(writer.stage_min[2]), float(writer.stage_max[2])),
        },
        "b_machine_range": (float(writer.b_machine_min), float(writer.b_machine_max)),
        "total_print_mm": float(writer.total_print_mm),
        "total_travel_mm": float(writer.total_travel_mm),
        "final_U_material_abs": float(writer.u_material_abs),
        "warnings": list(writer.warnings) + z_warnings,
    }
    if tongue_plans:
        tb_plan, pr_plan, pl_plan = tongue_plans
        summary["tongue"] = {
            "body_arc_mm":        float(tb_plan.total_arc_length),
            "prong_right_arc_mm": float(pr_plan.total_arc_length),
            "prong_left_arc_mm":  float(pl_plan.total_arc_length),
            "fork_point_xyz":     (centerline_pts[-1] +
                                   float(args.tongue_length) *
                                   float(args.tongue_fork_frac) *
                                   normalize(centerline_tans[-1])).tolist(),
        }
    return summary


# ============================================================================
# CLI
# ============================================================================
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="King Cobra G-code generator (sine-wave tube + raised neck + tongue)."
    )
    ap.add_argument("--out",         default=DEFAULT_OUT)
    ap.add_argument("--calibration", default=None,
                    help="Calibration JSON. Required for --write-mode calibrated.")
    ap.add_argument("--write-mode",  choices=["calibrated", "cartesian"],
                    default=DEFAULT_WRITE_MODE)
    ap.add_argument("--y-offplane-sign", type=float, default=DEFAULT_Y_OFFPLANE_SIGN)

    # Centerline geometry
    ap.add_argument("--x-start",   type=float, default=DEFAULT_X_START)
    ap.add_argument("--x-end",     type=float, default=DEFAULT_X_END)
    ap.add_argument("--y",         type=float, default=DEFAULT_Y)
    ap.add_argument("--z-min",     type=float, default=DEFAULT_Z_MIN,
                    help="Soft lower Z bound (warning only for cobra neck).")
    ap.add_argument("--z-max",     type=float, default=DEFAULT_Z_MAX,
                    help="Soft upper Z bound (warning only for cobra neck).")
    ap.add_argument("--z-baseline",   type=float, default=DEFAULT_Z_BASELINE)
    ap.add_argument("--z-amplitude",  type=float, default=DEFAULT_Z_AMPLITUDE)
    ap.add_argument("--cycles",       type=float, default=DEFAULT_CYCLES)
    ap.add_argument("--phase-deg",    type=float, default=DEFAULT_PHASE_DEG)
    ap.add_argument("--lead-in",      type=float, default=DEFAULT_LEAD_IN)
    ap.add_argument("--lead-out",     type=float, default=DEFAULT_LEAD_OUT)
    ap.add_argument("--centerline-samples", type=int,
                    default=DEFAULT_CENTERLINE_SAMPLES)

    # Diameter profile
    ap.add_argument("--diameter-start",      type=float, default=DEFAULT_DIAMETER_START)
    ap.add_argument("--diameter-main",       type=float, default=DEFAULT_DIAMETER_MAIN)
    ap.add_argument("--diameter-end",        type=float, default=DEFAULT_DIAMETER_END)
    ap.add_argument("--transition-in-frac",  type=float, default=DEFAULT_TRANSITION_IN_FRAC)
    ap.add_argument("--transition-out-frac", type=float, default=DEFAULT_TRANSITION_OUT_FRAC)

    # Helical winding
    ap.add_argument("--layer-height",              type=float, default=DEFAULT_LAYER_HEIGHT)
    ap.add_argument("--minor-segments-per-turn",   type=int,   default=DEFAULT_MINOR_SEGMENTS_PER_TURN)
    ap.add_argument("--phi0-deg",                  type=float, default=DEFAULT_PHI0_DEG)
    ap.add_argument("--frame-flip-outplane-sign",  type=float, default=DEFAULT_FRAME_FLIP_OUTPLANE_SIGN)

    # Orientation
    ap.add_argument("--c-deg",             type=float, default=DEFAULT_C_DEG)
    ap.add_argument("--b-angle-bias-deg",  type=float, default=DEFAULT_B_ANGLE_BIAS_DEG)
    ap.add_argument("--bc-solve-samples",  type=int,   default=DEFAULT_BC_SOLVE_SAMPLES)

    # Motion
    ap.add_argument("--travel-feed",        type=float, default=DEFAULT_TRAVEL_FEED)
    ap.add_argument("--approach-feed",      type=float, default=DEFAULT_APPROACH_FEED)
    ap.add_argument("--fine-approach-feed", type=float, default=DEFAULT_FINE_APPROACH_FEED)
    ap.add_argument("--print-feed",         type=float, default=DEFAULT_PRINT_FEED)
    ap.add_argument("--travel-lift-z",      type=float, default=DEFAULT_TRAVEL_LIFT_Z)
    ap.add_argument("--safe-approach-z",    type=float, default=DEFAULT_SAFE_APPROACH_Z)

    # Extrusion
    ap.add_argument("--emit-extrusion",    dest="emit_extrusion",
                    action="store_true",  default=DEFAULT_EMIT_EXTRUSION)
    ap.add_argument("--no-emit-extrusion", dest="emit_extrusion", action="store_false")
    ap.add_argument("--extrusion-per-mm",       type=float, default=DEFAULT_EXTRUSION_PER_MM)
    ap.add_argument("--extrusion-multiplier",   type=float, default=DEFAULT_EXTRUSION_MULTIPLIER)
    ap.add_argument("--prime-mm",               type=float, default=DEFAULT_PRIME_MM)
    ap.add_argument("--pressure-offset-mm",     type=float, default=DEFAULT_PRESSURE_OFFSET_MM)
    ap.add_argument("--pressure-advance-feed",  type=float, default=DEFAULT_PRESSURE_ADVANCE_FEED)
    ap.add_argument("--pressure-retract-feed",  type=float, default=DEFAULT_PRESSURE_RETRACT_FEED)
    ap.add_argument("--preflow-dwell-ms",       type=int,   default=DEFAULT_PREFLOW_DWELL_MS)
    ap.add_argument("--end-dwell-ms",           type=int,   default=DEFAULT_END_DWELL_MS)

    # Bbox
    ap.add_argument("--bbox-x-min", type=float, default=-DEFAULT_BBOX)
    ap.add_argument("--bbox-x-max", type=float, default=+DEFAULT_BBOX)
    ap.add_argument("--bbox-y-min", type=float, default=-DEFAULT_BBOX)
    ap.add_argument("--bbox-y-max", type=float, default=+DEFAULT_BBOX)
    ap.add_argument("--bbox-z-min", type=float, default=-DEFAULT_BBOX)
    ap.add_argument("--bbox-z-max", type=float, default=+DEFAULT_BBOX)

    # ---- Cobra-specific ----

    # XY oscillation
    ap.add_argument("--xy-osc-amplitude", type=float, default=DEFAULT_XY_OSC_AMPLITUDE,
                    help="Y (lateral) oscillation amplitude for body undulation (mm).")
    ap.add_argument("--xy-osc-cycles",    type=float, default=DEFAULT_XY_OSC_CYCLES,
                    help="Number of full lateral oscillation cycles along the body.")
    ap.add_argument("--xy-osc-phase-deg", type=float, default=DEFAULT_XY_OSC_PHASE_DEG,
                    help="Phase offset of the XY oscillation (degrees).")

    # Neck boost
    ap.add_argument("--neck-boost-mult",        type=float, default=DEFAULT_NECK_BOOST_MULT,
                    help="Amplitude multiplier at the peak of the raised neck (e.g. 3.8).")
    ap.add_argument("--neck-boost-start-cycle", type=float, default=DEFAULT_NECK_BOOST_START_CYCLE,
                    help="Wave-cycle number where the neck-boost ramp begins.")
    ap.add_argument("--neck-boost-peak-cycle",  type=float, default=DEFAULT_NECK_BOOST_PEAK_CYCLE,
                    help="Wave-cycle number where the neck-boost reaches its maximum.")

    # Neck / hood cross-section ellipse
    ap.add_argument("--neck-ellipse-y-ratio",    type=float, default=DEFAULT_NECK_ELLIPSE_Y_RATIO,
                    help="Max Y:X cross-section ratio for the cobra hood (Y longer, e.g. 2.5).")
    ap.add_argument("--neck-ellipse-start-frac", type=float, default=DEFAULT_NECK_ELLIPSE_START_FRAC,
                    help="Normalised arc fraction where the elliptical hood begins.")
    ap.add_argument("--neck-ellipse-peak-frac",  type=float, default=DEFAULT_NECK_ELLIPSE_PEAK_FRAC,
                    help="Normalised arc fraction where the ellipse reaches its maximum ratio.")

    # Tongue
    ap.add_argument("--tongue-enable",  dest="tongue_enable",
                    action="store_true",  default=DEFAULT_TONGUE_ENABLE)
    ap.add_argument("--no-tongue",      dest="tongue_enable", action="store_false")
    ap.add_argument("--tongue-length",           type=float, default=DEFAULT_TONGUE_LENGTH,
                    help="Total tongue length (mm), from head tip to prong tips.")
    ap.add_argument("--tongue-base-diam-y",      type=float, default=DEFAULT_TONGUE_BASE_DIAM_Y,
                    help="Lateral (Y) diameter of the tongue base (mm).")
    ap.add_argument("--tongue-base-diam-z",      type=float, default=DEFAULT_TONGUE_BASE_DIAM_Z,
                    help="Thin-axis (in-XZ-plane) diameter of the tongue base (mm).")
    ap.add_argument("--tongue-fork-frac",        type=float, default=DEFAULT_TONGUE_FORK_FRAC,
                    help="Fraction of tongue length before the fork (0–1).")
    ap.add_argument("--tongue-fork-angle-deg",   type=float, default=DEFAULT_TONGUE_FORK_ANGLE_DEG,
                    help="Half-angle of prong divergence from the tongue axis (degrees).")
    ap.add_argument("--tongue-taper-start-frac", type=float, default=DEFAULT_TONGUE_TAPER_START_FRAC,
                    help="Normalised position along each prong where tapering begins.")
    ap.add_argument("--tongue-tip-diam",         type=float, default=DEFAULT_TONGUE_TIP_DIAM,
                    help="Diameter at each prong tip (mm, near-zero for sharp point).")
    ap.add_argument("--tongue-layer-height",     type=float, default=DEFAULT_TONGUE_LAYER_HEIGHT,
                    help="Helical layer height for tongue winding (mm).")
    ap.add_argument("--tongue-segs-per-turn",    type=int,   default=DEFAULT_TONGUE_SEGS_PER_TURN,
                    help="Helix segments per minor turn for the tongue.")
    ap.add_argument("--tongue-extrusion-mult",   type=float, default=DEFAULT_TONGUE_EXTRUSION_MULT,
                    help="Extrusion-multiplier scale factor for the thin tongue (< 1.0).")

    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    summary = write_cobra_gcode(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
