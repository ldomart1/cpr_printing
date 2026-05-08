#!/usr/bin/env python3
"""
Klein Bottle G-code generator, cobra-style.

This script generates Duet/RRF-style G-code for a printable Klein-bottle-like
surface using the same broad conventions as the king-cobra generator:

  B angle convention:
    0 deg   -> tip points +Z, straight up
    90 deg  -> tip is horizontal
    180 deg -> tip points -Z, straight down

Print order / intent:
  1. Inner tube interior, top-down, B ~ 180 deg, upside-down printing.
  2. Base transition, expanding away from the inner tube while uncurling
     B smoothly from ~180 deg to ~0 deg.
  3. Large bottle base, written as top-down contour lines with B ~ 0 deg.
     The already-printed vertical inner tube is treated as a cylindrical
     obstacle; each contour is split around it rather than crossing through it.
  4. Neck/curl pass, following a centroid curve of the bottle neck while
     B ramps from base-up orientation into tangent-following curl.

The geometry is intentionally parametric rather than a fixed academic Klein
bottle parametrization.  For printing, this is usually easier to tune: body
ellipse, body height, inner tube location/radius, obstacle clearance, neck
height, neck radius, curl, layer spacing, and path density are all adjustable.

Quick preview:
  python klein_bottle_gcode_generator.py --preview --no-gcode --write-mode cartesian

Generate Cartesian/debug G-code:
  python klein_bottle_gcode_generator.py --write-mode cartesian --out klein_debug.gcode

Generate calibrated G-code:
  python klein_bottle_gcode_generator.py --write-mode calibrated \
      --calibration path/to/calibration.json --out gcode_generation/klein_bottle.gcode

Dependencies:
  Required: numpy
  Preview only: matplotlib
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# =============================================================================
# Defaults
# =============================================================================
DEFAULT_OUT = "gcode_generation/klein_bottle.gcode"

# Geometry / build position
DEFAULT_CENTER_X = 90.0
DEFAULT_CENTER_Y = 55.0
DEFAULT_BODY_TOP_Z = -95.0
DEFAULT_BODY_HEIGHT = 72.0
DEFAULT_BODY_RADIUS_X = 28.0
DEFAULT_BODY_RADIUS_Y = 21.0
DEFAULT_MOUTH_RADIUS = 6.0
DEFAULT_FOOT_SCALE = 0.68
DEFAULT_SHOULDER_POWER = 1.15
DEFAULT_BODY_LEAN_X = 0.0
DEFAULT_BODY_LEAN_Y = 0.0

# Inner vertical tube / obstacle
DEFAULT_INNER_X_OFFSET = -10.0
DEFAULT_INNER_Y_OFFSET = 0.0
DEFAULT_INNER_TUBE_RADIUS = 4.2
DEFAULT_INNER_TUBE_TOP_Z = -74.0
DEFAULT_INNER_TUBE_BOTTOM_Z = -118.0
DEFAULT_OBSTACLE_CLEARANCE = 2.2

# Transition from inner tube to base
DEFAULT_TRANSITION_TURNS = 0.62
DEFAULT_TRANSITION_SAMPLES = 240
DEFAULT_TRANSITION_RADIUS_END = 12.0
DEFAULT_TRANSITION_Z_DROP = 8.0

# Base contours
DEFAULT_BASE_LAYERS = 52
DEFAULT_RING_SEGMENTS = 240
DEFAULT_MIN_ARC_POINTS = 8
DEFAULT_LINE_SPACING_Z = 1.25

# Neck/curl path
DEFAULT_NECK_ENABLE = True
DEFAULT_NECK_RADIUS = 3.8
DEFAULT_NECK_CENTERLINE_SAMPLES = 650
DEFAULT_NECK_SEGS_PER_TURN = 28
DEFAULT_NECK_ATTACH_ANGLE_DEG = 12.0
DEFAULT_NECK_ATTACH_DROP = 20.0
DEFAULT_NECK_OUTWARD = 18.0
DEFAULT_NECK_HEIGHT = 46.0
DEFAULT_NECK_CURL_B_BLEND_FRAC = 0.35
DEFAULT_NECK_TWIST_TURNS = 1.2

# Helix/pathing
DEFAULT_LAYER_HEIGHT = 0.6
DEFAULT_MINOR_SEGMENTS_PER_TURN = 36
DEFAULT_PHI0_DEG = 0.0
DEFAULT_FRAME_FLIP_OUTPLANE_SIGN = 1.0

# Machine / G-code
DEFAULT_WRITE_MODE = "calibrated"
DEFAULT_C_DEG = 180.0
DEFAULT_B_ANGLE_BIAS_DEG = 0.0
DEFAULT_BC_SOLVE_SAMPLES = 5001
DEFAULT_Y_OFFPLANE_SIGN = -1.0
DEFAULT_TRAVEL_FEED = 2000.0
DEFAULT_APPROACH_FEED = 1200.0
DEFAULT_FINE_APPROACH_FEED = 150.0
DEFAULT_PRINT_FEED = 400.0
DEFAULT_TRAVEL_LIFT_Z = 8.0
DEFAULT_SAFE_APPROACH_Z = -50.0
DEFAULT_EMIT_EXTRUSION = True
DEFAULT_EXTRUSION_PER_MM = 0.0015
DEFAULT_EXTRUSION_MULTIPLIER = 1.0
DEFAULT_PREFLOW_DWELL_MS = 500
DEFAULT_END_DWELL_MS = 0
DEFAULT_PRESSURE_OFFSET_MM = 4.0
DEFAULT_PRESSURE_ADVANCE_FEED = 120.0
DEFAULT_PRESSURE_RETRACT_FEED = 240.0
DEFAULT_BBOX = 1e9


# =============================================================================
# Dataclasses
# =============================================================================
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
class SegmentPlan:
    label: str
    points: np.ndarray              # (N,3) tip-space path points
    tangents: np.ndarray            # (N,3) path tangents or centerline tangents
    b_targets: np.ndarray           # (N,) desired physical B angles in degrees
    extrusion_scale: float = 1.0
    closed: bool = False


@dataclass
class KleinPlan:
    segments: List[SegmentPlan]
    inner_points: np.ndarray
    transition_points: np.ndarray
    base_segments: List[SegmentPlan]
    neck_centerline: Optional[np.ndarray]
    neck_surface: Optional[np.ndarray]
    obstacle_center_xy: Tuple[float, float]
    obstacle_radius: float


# =============================================================================
# Math helpers
# =============================================================================
def smootherstep(t: Any) -> np.ndarray:
    t_arr = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return t_arr * t_arr * t_arr * (t_arr * (t_arr * 6.0 - 15.0) + 10.0)


def normalize(v: Any, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    return arr / n if n > eps else np.zeros_like(arr)


def compute_tangents_fd(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    n = len(pts)
    raw = np.zeros_like(pts)
    if n >= 3:
        raw[1:-1] = pts[2:] - pts[:-2]
    if n >= 2:
        raw[0] = pts[1] - pts[0]
        raw[-1] = pts[-1] - pts[-2]
    elif n == 1:
        raw[0] = np.array([1.0, 0.0, 0.0], dtype=float)
    norms = np.linalg.norm(raw, axis=1, keepdims=True)
    norms = np.where(norms < 1e-12, 1.0, norms)
    return raw / norms


def cumulative_arc_length(pts: np.ndarray) -> np.ndarray:
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 2:
        return np.zeros(len(pts), dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def desired_physical_b_angle_from_tangent(tangent: np.ndarray) -> float:
    """B-angle from tangent: 0 -> +Z, 90 -> horizontal, 180 -> -Z."""
    t = normalize(np.asarray(tangent, dtype=float))
    tz = float(np.clip(t[2], -1.0, 1.0))
    return float(math.degrees(math.acos(tz)))


def rotate_vector(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rodrigues rotation."""
    v = np.asarray(v, dtype=float)
    k = normalize(axis)
    if np.linalg.norm(k) < 1e-12 or abs(angle_rad) < 1e-14:
        return v.copy()
    return (v * math.cos(angle_rad) +
            np.cross(k, v) * math.sin(angle_rad) +
            k * np.dot(k, v) * (1.0 - math.cos(angle_rad)))


def fmt_axes(axes_vals: Sequence[Tuple[str, float]]) -> str:
    return " ".join(f"{ax}{val:.3f}" for ax, val in axes_vals)


# =============================================================================
# Calibration loading and evaluation
# =============================================================================
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
            if delta[k - 1] == 0.0 or delta[k] == 0.0 or np.sign(delta[k - 1]) != np.sign(delta[k]):
                d[k] = 0.0
            else:
                w1 = 2.0 * h[k] + h[k - 1]
                w2 = h[k] + 2.0 * h[k - 1]
                d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
        d[0] = _pchip_endpoint_slope(h[0], h[1], delta[0], delta[1])
        d[-1] = _pchip_endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])
    flat = xq.reshape(-1)
    idx = np.clip(np.searchsorted(x, flat, side="right") - 1, 0, x.size - 2)
    x0, x1 = x[idx], x[idx + 1]
    y0, y1 = y[idx], y[idx + 1]
    h_i = x1 - x0
    t = np.clip((flat - x0) / h_i, 0.0, 1.0)
    h00 = 2 * t ** 3 - 3 * t ** 2 + 1
    h10 = t ** 3 - 2 * t ** 2 + t
    h01 = -2 * t ** 3 + 3 * t ** 2
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
        return poly_eval(model_spec.get("coefficients", model_spec.get("coeffs")), u, default_if_none)
    raise ValueError(f"Unsupported calibration model_type: {model_type}")


def eval_pchip_with_linear_extrap(model_spec: Dict[str, Any], extrap_model_spec: Optional[Dict[str, Any]], b: Any) -> np.ndarray:
    x_knots = np.asarray(model_spec.get("x_knots", []), dtype=float).reshape(-1)
    if x_knots.size == 0 or extrap_model_spec is None:
        return eval_model_spec(model_spec, b, default_if_none=0.0)
    b_arr = np.asarray(b, dtype=float)
    out = np.asarray(eval_model_spec(model_spec, b_arr, default_if_none=0.0), dtype=float).copy()
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

    cubic = data.get("cubic_coefficients", {}) or {}
    fit_models = data.get("fit_models", {}) or {}

    def _coeffs(models: Dict[str, Any], *names: str) -> Optional[np.ndarray]:
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

    pr = (np.asarray(cubic["r_coeffs"], dtype=float) if cubic.get("r_coeffs") is not None
          else (_coeffs(fit_models, "r_cubic", "r_avg_cubic") or np.zeros(1)))
    pz = (np.asarray(cubic["z_coeffs"], dtype=float) if cubic.get("z_coeffs") is not None
          else (_coeffs(fit_models, "z_cubic", "z_avg_cubic") or np.zeros(1)))
    py_off = (np.asarray(cubic["offplane_y_coeffs"], dtype=float)
              if cubic.get("offplane_y_coeffs") is not None
              else _coeffs(fit_models, "offplane_y_avg_cubic", "offplane_y_cubic",
                           "offplane_y", "offplane_y_linear", "offplane_y_avg_linear"))
    pa = (np.asarray(cubic["tip_angle_coeffs"], dtype=float)
          if cubic.get("tip_angle_coeffs") is not None
          else _coeffs(fit_models, "tip_angle_cubic", "tip_angle_avg_cubic"))

    sel = data.get("selected_fit_model")
    sel = None if sel is None else str(sel).strip().lower()
    sel_op = data.get("selected_offplane_fit_model")
    sel_op = None if sel_op is None else str(sel_op).strip().lower()
    active_phase = str(data.get("default_phase_for_legacy_access") or "pull").strip().lower()

    phase_models = data.get("fit_models_by_phase", {}) or {}
    apm = phase_models.get(active_phase) if isinstance(phase_models, dict) else None
    if not isinstance(apm, dict):
        apm = fit_models

    y_off_sel = sel_op or sel
    r_model = _select_named_model(apm, "r", sel)
    z_model = _select_named_model(apm, "z", sel)
    yo_model = (_normalize_model_spec(apm.get("offplane_y_avg_cubic")) or
                _select_named_model(apm, "offplane_y", y_off_sel))
    yoe_model = (_normalize_model_spec(apm.get("offplane_y_avg_linear")) or
                 _normalize_model_spec(apm.get("offplane_y_linear")) or
                 _normalize_model_spec(apm.get("offplane_y")))
    ta_model = _select_named_model(apm, "tip_angle", sel)

    motor = data.get("motor_setup", {}) or {}
    duet = data.get("duet_axis_mapping", {}) or {}
    b_range = motor.get("b_motor_position_range", [-5.4, 0.0])
    b_min, b_max = sorted(map(float, b_range))

    return Calibration(
        pr=pr, pz=pz, py_off=py_off, pa=pa,
        r_model=r_model, z_model=z_model,
        y_off_model=yo_model, y_off_extrap_model=yoe_model, tip_angle_model=ta_model,
        selected_fit_model=sel, selected_offplane_fit_model=sel_op,
        active_phase=active_phase,
        b_min=b_min, b_max=b_max,
        x_axis=str(duet.get("horizontal_axis") or motor.get("horizontal_axis") or "X"),
        y_axis=str(duet.get("depth_axis") or motor.get("depth_axis") or "Y"),
        z_axis=str(duet.get("vertical_axis") or motor.get("vertical_axis") or "Z"),
        b_axis=str(duet.get("pull_axis") or motor.get("b_motor_axis") or "B"),
        c_axis=str(duet.get("rotation_axis") or motor.get("rotation_axis") or "C"),
        u_axis=str(duet.get("extruder_axis") or "U"),
        c_180_deg=float(motor.get("rotation_axis_180_deg", 180.0)),
        offplane_y_sign=float(offplane_y_sign),
    )


def make_cartesian_calibration() -> Calibration:
    return Calibration(
        pr=np.zeros(1), pz=np.zeros(1), py_off=np.zeros(1), pa=np.array([1.0, 0.0]),
        b_min=0.0, b_max=180.0,
        x_axis="X", y_axis="Y", z_axis="Z", b_axis="B", c_axis="C", u_axis="U",
        c_180_deg=180.0,
    )


def eval_r(cal: Calibration, b: Any) -> np.ndarray:
    return eval_model_spec(cal.r_model, b) if cal.r_model is not None else poly_eval(cal.pr, b)


def eval_z(cal: Calibration, b: Any) -> np.ndarray:
    return eval_model_spec(cal.z_model, b) if cal.z_model is not None else poly_eval(cal.pz, b)


def eval_offplane_y(cal: Calibration, b: Any) -> np.ndarray:
    if cal.y_off_model is not None:
        if str(cal.y_off_model.get("model_type", "")).lower() == "pchip":
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
    return np.array([
        r * math.cos(c) - y_off * math.sin(c),
        r * math.sin(c) + y_off * math.cos(c),
        z,
    ], dtype=float)


def stage_xyz_for_tip(cal: Calibration, tip_xyz: np.ndarray, b_machine: float, c_deg: float) -> np.ndarray:
    return np.asarray(tip_xyz, dtype=float) - predict_tip_offset_xyz(cal, b_machine, c_deg)


def solve_b_for_target_tip_angle(cal: Calibration, target_angle_deg: float, search_samples: int = DEFAULT_BC_SOLVE_SAMPLES) -> float:
    bb = np.linspace(float(cal.b_min), float(cal.b_max), int(max(101, search_samples)))
    aa = eval_tip_angle_deg(cal, bb) - float(target_angle_deg)
    i_best = int(np.argmin(np.abs(aa)))
    b_best = float(bb[i_best])

    sign_changes: List[Tuple[float, float, float]] = []
    for i in range(len(bb) - 1):
        a0, a1 = float(aa[i]), float(aa[i + 1])
        if a0 == 0.0:
            return float(bb[i])
        if a0 * a1 < 0.0:
            sign_changes.append((min(abs(a0), abs(a1)), float(bb[i]), float(bb[i + 1])))

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
        return f"{fallback_name}: legacy polynomial/cartesian fallback"
    mt = str(model.get("model_type") or "unknown").strip().lower()
    eq = str(model.get("equation") or "").strip()
    xr = model.get("fit_x_range")
    rng = (f", fit_x_range=[{float(xr[0]):.3f}, {float(xr[1]):.3f}]"
           if isinstance(xr, (list, tuple)) and len(xr) >= 2 else "")
    return f"{fallback_name}: {mt}{rng}{'; ' + eq if eq else ''}"


# =============================================================================
# Geometry builders
# =============================================================================
def body_bottom_z(args: argparse.Namespace) -> float:
    return float(args.body_top_z) - float(args.body_height)


def inner_center_xy(args: argparse.Namespace) -> Tuple[float, float]:
    return float(args.center_x + args.inner_x_offset), float(args.center_y + args.inner_y_offset)


def body_center_at_u(args: argparse.Namespace, u: float) -> Tuple[float, float]:
    """Center shift for body contours. u=0 top, u=1 bottom."""
    ease = float(smootherstep(u))
    return (float(args.center_x) + float(args.body_lean_x) * (ease - 0.5),
            float(args.center_y) + float(args.body_lean_y) * (ease - 0.5))


def body_radii_at_u(args: argparse.Namespace, u: float) -> Tuple[float, float]:
    """Bottle body profile from small mouth/shoulder to wide body to tucked foot."""
    u = float(np.clip(u, 0.0, 1.0))
    shoulder = float(smootherstep(np.clip(u / 0.48, 0.0, 1.0))) ** float(args.shoulder_power)
    foot_tuck = 1.0 - (1.0 - float(args.foot_scale)) * float(smootherstep(np.clip((u - 0.72) / 0.28, 0.0, 1.0)))
    rx = float(args.mouth_radius) * (1.0 - shoulder) + float(args.body_radius_x) * shoulder * foot_tuck
    ry = float(args.mouth_radius) * (1.0 - shoulder) + float(args.body_radius_y) * shoulder * foot_tuck
    return max(0.1, rx), max(0.1, ry)


def ellipse_points(cx: float, cy: float, z: float, rx: float, ry: float, theta: np.ndarray) -> np.ndarray:
    return np.column_stack([cx + rx * np.cos(theta), cy + ry * np.sin(theta), np.full_like(theta, float(z))])


def split_circular_valid_runs(points: np.ndarray, mask: np.ndarray, min_points: int) -> List[np.ndarray]:
    """Split a circular contour into valid arcs. If all points valid, return closed contour."""
    points = np.asarray(points, dtype=float)
    mask = np.asarray(mask, dtype=bool)
    n = len(points)
    if n == 0 or not np.any(mask):
        return []
    if np.all(mask):
        return [np.vstack([points, points[0]])]

    invalid_idx = np.where(~mask)[0]
    start = int((invalid_idx[0] + 1) % n)
    idx_order = (np.arange(n) + start) % n
    m = mask[idx_order]
    pts = points[idx_order]

    runs: List[np.ndarray] = []
    i = 0
    while i < n:
        while i < n and not m[i]:
            i += 1
        if i >= n:
            break
        j = i
        while j < n and m[j]:
            j += 1
        run = pts[i:j]
        if len(run) >= int(min_points):
            runs.append(run.copy())
        i = j
    return runs


def build_vertical_inner_tube(args: argparse.Namespace) -> SegmentPlan:
    ix, iy = inner_center_xy(args)
    top_z = float(args.inner_tube_top_z)
    bottom_z = float(args.inner_tube_bottom_z)
    if bottom_z >= top_z:
        raise ValueError("--inner-tube-bottom-z must be below --inner-tube-top-z")
    height = top_z - bottom_z
    turns = max(0.25, height / float(args.layer_height))
    n = max(12, int(math.ceil(turns * float(args.minor_segments_per_turn))))
    s = np.linspace(0.0, 1.0, n + 1)
    theta = math.radians(float(args.phi0_deg)) + 2.0 * math.pi * turns * s
    z = top_z - height * s
    r = float(args.inner_tube_radius)
    pts = np.column_stack([ix + r * np.cos(theta), iy + r * np.sin(theta), z])
    tangents = np.tile(np.array([0.0, 0.0, -1.0], dtype=float), (len(pts), 1))
    b = np.full(len(pts), 180.0, dtype=float)
    return SegmentPlan("inner_tube_top_down_B180", pts, tangents, b, extrusion_scale=float(args.inner_extrusion_scale))


def build_base_transition(args: argparse.Namespace, start_point: np.ndarray) -> SegmentPlan:
    ix, iy = inner_center_xy(args)
    n = int(max(20, args.transition_samples))
    s = np.linspace(0.0, 1.0, n)
    ss = smootherstep(s)
    start_r = float(args.inner_tube_radius)
    end_r = float(args.transition_radius_end)
    turns = float(args.transition_turns)

    start_theta = math.atan2(float(start_point[1]) - iy, float(start_point[0]) - ix)
    theta = start_theta + 2.0 * math.pi * turns * s
    radius = start_r + (end_r - start_r) * ss
    z0 = float(start_point[2])
    z1 = float(args.body_top_z) - float(args.transition_z_drop)
    z = z0 + (z1 - z0) * ss
    pts = np.column_stack([ix + radius * np.cos(theta), iy + radius * np.sin(theta), z])
    pts[0] = np.asarray(start_point, dtype=float)
    tangents = compute_tangents_fd(pts)
    b = 180.0 + (0.0 - 180.0) * ss
    return SegmentPlan("uncurl_transition_B180_to_B0", pts, tangents, b, extrusion_scale=float(args.transition_extrusion_scale))


def build_base_contours(args: argparse.Namespace) -> List[SegmentPlan]:
    ix, iy = inner_center_xy(args)
    obstacle_radius = float(args.inner_tube_radius) + float(args.obstacle_clearance)
    layers = int(max(2, args.base_layers))
    theta = np.linspace(0.0, 2.0 * math.pi, int(max(16, args.ring_segments)), endpoint=False)
    z_top = float(args.body_top_z)
    z_bottom = body_bottom_z(args)

    segments: List[SegmentPlan] = []
    # Top-down: u=0 top to u=1 bottom.
    for layer_i, u in enumerate(np.linspace(0.0, 1.0, layers)):
        z = z_top + (z_bottom - z_top) * u
        cx, cy = body_center_at_u(args, float(u))
        rx, ry = body_radii_at_u(args, float(u))
        pts = ellipse_points(cx, cy, z, rx, ry, theta)

        # Obstacle is the already printed vertical inner tube. We skip points inside
        # a clearance cylinder and split the ring into printable arcs around it.
        dxy = np.sqrt((pts[:, 0] - ix) ** 2 + (pts[:, 1] - iy) ** 2)
        mask = dxy >= obstacle_radius
        runs = split_circular_valid_runs(pts, mask, min_points=int(args.min_arc_points))
        for run_i, run in enumerate(runs):
            tangents = compute_tangents_fd(run)
            b = np.zeros(len(run), dtype=float)
            segments.append(SegmentPlan(
                label=f"base_layer_{layer_i:03d}_arc_{run_i:02d}_B0_obstacle_aware",
                points=run,
                tangents=tangents,
                b_targets=b,
                extrusion_scale=float(args.base_extrusion_scale),
                closed=np.all(mask),
            ))
    return segments


def bezier_cubic(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    u = 1.0 - t
    return ((u ** 3)[:, None] * p0 +
            (3 * u ** 2 * t)[:, None] * p1 +
            (3 * u * t ** 2)[:, None] * p2 +
            (t ** 3)[:, None] * p3)


def build_neck_centerline(args: argparse.Namespace) -> np.ndarray:
    ix, iy = inner_center_xy(args)
    attach_angle = math.radians(float(args.neck_attach_angle_deg))
    attach_u = np.clip(float(args.neck_attach_drop) / max(float(args.body_height), 1e-9), 0.0, 1.0)
    cx_att, cy_att = body_center_at_u(args, attach_u)
    rx_att, ry_att = body_radii_at_u(args, attach_u)
    z_att = float(args.body_top_z) - float(args.neck_attach_drop)

    p0 = np.array([
        cx_att + rx_att * math.cos(attach_angle),
        cy_att + ry_att * math.sin(attach_angle),
        z_att,
    ], dtype=float)
    radial = normalize(np.array([math.cos(attach_angle), math.sin(attach_angle), 0.0], dtype=float))
    if np.linalg.norm(radial) < 1e-12:
        radial = np.array([1.0, 0.0, 0.0], dtype=float)

    p1 = p0 + radial * float(args.neck_outward) + np.array([0.0, 0.0, float(args.neck_height) * 0.35])
    p2 = np.array([
        ix + float(args.neck_outward) * 0.35,
        iy,
        float(args.inner_tube_top_z) + float(args.neck_height),
    ], dtype=float)
    p3 = np.array([ix, iy, float(args.inner_tube_top_z)], dtype=float)

    n = int(max(30, args.neck_centerline_samples))
    t = np.linspace(0.0, 1.0, n)
    curve = bezier_cubic(p0, p1, p2, p3, t)

    # Add a gentle lateral curl/twist around the centroid path to make the neck
    # visually and mechanically Klein-like without overcomplicating the control UI.
    twist = float(args.neck_twist_turns)
    if abs(twist) > 1e-12:
        # Offset perpendicular to the p0->p3 chord, fading at endpoints.
        chord = normalize(p3 - p0)
        side = normalize(np.cross(chord, np.array([0.0, 0.0, 1.0])))
        if np.linalg.norm(side) < 1e-12:
            side = np.array([0.0, 1.0, 0.0], dtype=float)
        envelope = np.sin(math.pi * t)
        curve += (side[None, :] * (0.15 * float(args.neck_outward) * envelope * np.sin(2.0 * math.pi * twist * t))[:, None])
    return curve


def parallel_transport_frames(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return tangent, normal, binormal arrays along a polyline."""
    pts = np.asarray(points, dtype=float)
    tangents = compute_tangents_fd(pts)
    n_pts = len(pts)
    normals = np.zeros_like(pts)
    binormals = np.zeros_like(pts)

    # Initial normal: prefer projected global Z, then global Y.
    t0 = tangents[0]
    guess = np.array([0.0, 0.0, 1.0], dtype=float)
    n0 = guess - np.dot(guess, t0) * t0
    if np.linalg.norm(n0) < 1e-8:
        guess = np.array([0.0, 1.0, 0.0], dtype=float)
        n0 = guess - np.dot(guess, t0) * t0
    normals[0] = normalize(n0)
    binormals[0] = normalize(np.cross(t0, normals[0]))

    for i in range(1, n_pts):
        prev_t = tangents[i - 1]
        cur_t = tangents[i]
        axis = np.cross(prev_t, cur_t)
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-10:
            n_i = normals[i - 1]
        else:
            dot = float(np.clip(np.dot(prev_t, cur_t), -1.0, 1.0))
            angle = math.atan2(axis_norm, dot)
            n_i = rotate_vector(normals[i - 1], axis / axis_norm, angle)
        # Re-orthogonalize to avoid drift.
        n_i = n_i - np.dot(n_i, cur_t) * cur_t
        normals[i] = normalize(n_i)
        binormals[i] = normalize(np.cross(cur_t, normals[i]))
    return tangents, normals, binormals


def interpolate_curve_by_arc(points: np.ndarray, arc_targets: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=float)
    tangents, normals, binormals = parallel_transport_frames(pts)
    arc = cumulative_arc_length(pts)
    out_pts = np.zeros((len(arc_targets), 3), dtype=float)
    out_tan = np.zeros_like(out_pts)
    out_n = np.zeros_like(out_pts)
    out_b = np.zeros_like(out_pts)
    for i, s in enumerate(arc_targets):
        s = float(np.clip(s, 0.0, arc[-1]))
        idx = int(np.searchsorted(arc, s, side="left"))
        if idx <= 0:
            out_pts[i], out_tan[i], out_n[i], out_b[i] = pts[0], tangents[0], normals[0], binormals[0]
        elif idx >= len(arc):
            out_pts[i], out_tan[i], out_n[i], out_b[i] = pts[-1], tangents[-1], normals[-1], binormals[-1]
        else:
            seg_len = float(arc[idx] - arc[idx - 1])
            a = 0.0 if seg_len <= 1e-12 else (s - float(arc[idx - 1])) / seg_len
            out_pts[i] = (1.0 - a) * pts[idx - 1] + a * pts[idx]
            out_tan[i] = normalize((1.0 - a) * tangents[idx - 1] + a * tangents[idx])
            out_n[i] = normalize((1.0 - a) * normals[idx - 1] + a * normals[idx])
            out_b[i] = normalize((1.0 - a) * binormals[idx - 1] + a * binormals[idx])
    return out_pts, out_tan, out_n, out_b


def build_neck_surface_helix(args: argparse.Namespace, centerline: np.ndarray) -> SegmentPlan:
    arc = cumulative_arc_length(centerline)
    total = float(arc[-1])
    if total <= 1e-9:
        raise ValueError("Neck centerline has zero length")
    turns = max(0.25, total / float(args.layer_height))
    n = max(20, int(math.ceil(turns * float(args.neck_segs_per_turn))))
    arc_targets = np.linspace(0.0, total, n + 1)
    cl, tan, normal, binormal = interpolate_curve_by_arc(centerline, arc_targets)
    s_norm = arc_targets / total
    phi = math.radians(float(args.phi0_deg)) + 2.0 * math.pi * arc_targets / float(args.layer_height)
    r = float(args.neck_radius)
    surface = cl + r * np.cos(phi)[:, None] * normal + r * np.sin(phi)[:, None] * binormal * float(np.sign(args.frame_flip_outplane_sign) or 1.0)

    tangent_b = np.array([desired_physical_b_angle_from_tangent(t) for t in tan], dtype=float)
    blend = smootherstep(s_norm / max(float(args.neck_curl_b_blend_frac), 1e-9))
    b_targets = (1.0 - blend) * 0.0 + blend * tangent_b
    return SegmentPlan("neck_curl_follow_centroid", surface, tan, b_targets, extrusion_scale=float(args.neck_extrusion_scale))


def build_klein_plan(args: argparse.Namespace) -> KleinPlan:
    inner = build_vertical_inner_tube(args)
    transition = build_base_transition(args, inner.points[-1])
    base_segments = build_base_contours(args)

    segments = [inner, transition]
    segments.extend(base_segments)

    neck_centerline = None
    neck_surface = None
    if bool(args.neck_enable):
        neck_centerline = build_neck_centerline(args)
        neck = build_neck_surface_helix(args, neck_centerline)
        neck_surface = neck.points
        segments.append(neck)

    ix, iy = inner_center_xy(args)
    obstacle_radius = float(args.inner_tube_radius) + float(args.obstacle_clearance)
    return KleinPlan(
        segments=segments,
        inner_points=inner.points,
        transition_points=transition.points,
        base_segments=base_segments,
        neck_centerline=neck_centerline,
        neck_surface=neck_surface,
        obstacle_center_xy=(ix, iy),
        obstacle_radius=obstacle_radius,
    )


# =============================================================================
# G-code writer
# =============================================================================
class KleinGCodeWriter:
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
        self.cur_b_machine = 0.0
        self.cur_b_target_deg = 0.0
        self.cur_c = self.c_deg
        self.stage_min = np.array([np.inf, np.inf, np.inf])
        self.stage_max = np.array([-np.inf, -np.inf, -np.inf])
        self.tip_min = np.array([np.inf, np.inf, np.inf])
        self.tip_max = np.array([-np.inf, -np.inf, -np.inf])
        self.b_machine_min = np.inf
        self.b_machine_max = -np.inf
        self.total_print_mm = 0.0
        self.total_travel_mm = 0.0
        self.warnings: List[str] = []
        self._b_cache: Dict[float, float] = {}

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

    def b_machine_for_target(self, b_target_deg: float) -> float:
        b_target = float(np.clip(b_target_deg + self.b_angle_bias_deg, 0.0, 180.0))
        key = round(b_target, 8)
        if key not in self._b_cache:
            self._b_cache[key] = solve_b_for_target_tip_angle(self.cal, key, self.bc_solve_samples) if self.calibrated else key
        return self._b_cache[key]

    def clamp_stage(self, p: np.ndarray, context: str) -> np.ndarray:
        p_arr = np.asarray(p, dtype=float)
        q = np.minimum(np.maximum(p_arr, self.bbox_min), self.bbox_max)
        if float(np.linalg.norm(q - p_arr)) > 1e-9:
            self.warnings.append(f"WARNING: clamped during {context}: requested={p_arr.tolist()} clamped={q.tolist()}")
        return q

    def tip_to_stage(self, tip_xyz: np.ndarray, b_target_deg: float, c_deg: float) -> Tuple[np.ndarray, float]:
        b_machine = self.b_machine_for_target(b_target_deg)
        stage = stage_xyz_for_tip(self.cal, tip_xyz, b_machine, c_deg) if self.calibrated else np.asarray(tip_xyz, dtype=float)
        return self.clamp_stage(stage, "tip_to_stage"), b_machine

    def write_move(self, stage_xyz: np.ndarray, b_machine: float, b_target_deg: float, c_deg: float, feed: float, comment: Optional[str] = None) -> None:
        if comment:
            self.fh.write(f"; {comment}\n")
        axes: List[Tuple[str, float]] = [
            (self.x_axis, float(stage_xyz[0])),
            (self.y_axis, float(stage_xyz[1])),
            (self.z_axis, float(stage_xyz[2])),
            (self.b_axis, float(b_machine)),
            (self.c_axis, float(c_deg)),
        ]
        self.fh.write(f"G1 {fmt_axes(axes)} F{float(feed):.0f}\n")
        self.cur_stage = np.asarray(stage_xyz, dtype=float).copy()
        self.cur_b_machine = float(b_machine)
        self.cur_b_target_deg = float(b_target_deg)
        self.cur_c = float(c_deg)
        self.stage_min = np.minimum(self.stage_min, self.cur_stage)
        self.stage_max = np.maximum(self.stage_max, self.cur_stage)
        self.b_machine_min = min(self.b_machine_min, float(b_machine))
        self.b_machine_max = max(self.b_machine_max, float(b_machine))

    def move_to_tip(self, tip_xyz: np.ndarray, b_target_deg: float, c_deg: float, feed: float, *, comment: Optional[str] = None, extrude: bool = False, extrusion_scale: float = 1.0) -> None:
        tip = np.asarray(tip_xyz, dtype=float)
        prev = None if self.cur_tip is None else self.cur_tip.copy()
        stage, b_machine = self.tip_to_stage(tip, b_target_deg, c_deg)
        if prev is not None:
            dist = float(np.linalg.norm(tip - prev))
            if extrude:
                self.total_print_mm += dist
                if self.emit_extrusion:
                    self.u_material_abs += self.extrusion_per_mm * self.extrusion_multiplier * float(extrusion_scale) * dist
            else:
                self.total_travel_mm += dist
        self.write_move(stage, b_machine, b_target_deg, c_deg, feed, comment=comment)
        self.cur_tip = tip.copy()
        self.tip_min = np.minimum(self.tip_min, tip)
        self.tip_max = np.maximum(self.tip_max, tip)

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

    def approach_start(self, start_tip: np.ndarray, start_b_target_deg: float, c_deg: float, safe_approach_z: float, travel_lift_z: float) -> None:
        start_tip = np.asarray(start_tip, dtype=float)
        start_stage, start_b_machine = self.tip_to_stage(start_tip, start_b_target_deg, c_deg)
        safe_stage = start_stage.copy()
        safe_stage[2] = float(safe_approach_z)

        if self.cur_stage is not None:
            lifted = self.cur_stage.copy()
            lifted[2] = float(safe_approach_z)
            self.write_move(self.clamp_stage(lifted, "approach lift"), self.cur_b_machine, self.cur_b_target_deg, self.cur_c, self.approach_feed, comment=f"lift to safe Z{safe_approach_z:.3f}")
        self.write_move(self.clamp_stage(safe_stage, "approach safe"), start_b_machine, start_b_target_deg, c_deg, self.travel_feed, comment="set B/C, move above segment start")
        near = start_stage.copy()
        near[2] = min(float(safe_approach_z), float(start_stage[2]) + max(0.0, float(travel_lift_z)))
        if near[2] > float(start_stage[2]) + 1e-9:
            self.write_move(self.clamp_stage(near, "approach near"), start_b_machine, start_b_target_deg, c_deg, self.travel_feed, comment="descend toward segment start")
        self.write_move(self.clamp_stage(start_stage, "approach final"), start_b_machine, start_b_target_deg, c_deg, self.fine_approach_feed, comment="fine approach to segment start")
        self.cur_tip = start_tip.copy()

    def exit_to_safe_z(self, safe_approach_z: float) -> None:
        if self.cur_stage is None:
            return
        safe = self.cur_stage.copy()
        safe[2] = float(safe_approach_z)
        self.write_move(self.clamp_stage(safe, "exit"), self.cur_b_machine, self.cur_b_target_deg, self.cur_c, self.approach_feed, comment=f"exit to safe Z{safe_approach_z:.3f}")

    def print_segment(self, seg: SegmentPlan, c_deg: float, safe_approach_z: float, travel_lift_z: float) -> None:
        pts = np.asarray(seg.points, dtype=float)
        if len(pts) < 2:
            return
        b_targets = np.asarray(seg.b_targets, dtype=float)
        if len(b_targets) != len(pts):
            raise ValueError(f"Segment {seg.label} has points/b_targets length mismatch")
        self.fh.write(f"\n; ===== {seg.label} =====\n")
        self.fh.write(f"; points={len(pts)} B_start={b_targets[0]:.3f} B_end={b_targets[-1]:.3f} extrusion_scale={seg.extrusion_scale:.3f}\n")
        self.approach_start(pts[0], float(b_targets[0]), c_deg, safe_approach_z, travel_lift_z)
        self.pressure_preload()
        for i in range(1, len(pts)):
            self.move_to_tip(pts[i], float(b_targets[i]), c_deg, self.print_feed, extrude=True, extrusion_scale=float(seg.extrusion_scale))
        self.pressure_release()
        if self.end_dwell_ms > 0:
            self.fh.write(f"G4 P{self.end_dwell_ms}\n")
        self.exit_to_safe_z(safe_approach_z)


# =============================================================================
# G-code generation
# =============================================================================
def validate_args(args: argparse.Namespace) -> None:
    if args.write_mode == "calibrated" and not args.calibration:
        raise ValueError("--calibration is required when --write-mode calibrated")
    if args.body_height <= 0:
        raise ValueError("--body-height must be > 0")
    if args.body_radius_x <= 0 or args.body_radius_y <= 0:
        raise ValueError("--body-radius-x and --body-radius-y must be > 0")
    if args.inner_tube_radius <= 0:
        raise ValueError("--inner-tube-radius must be > 0")
    if args.layer_height <= 0:
        raise ValueError("--layer-height must be > 0")


def write_klein_gcode(args: argparse.Namespace) -> Dict[str, Any]:
    validate_args(args)
    cal = load_calibration(args.calibration, offplane_y_sign=args.y_offplane_sign) if args.write_mode == "calibrated" else make_cartesian_calibration()
    cal.offplane_y_sign = float(args.y_offplane_sign)
    plan = build_klein_plan(args)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    bbox_min = np.array([args.bbox_x_min, args.bbox_y_min, args.bbox_z_min], dtype=float)
    bbox_max = np.array([args.bbox_x_max, args.bbox_y_max, args.bbox_z_max], dtype=float)

    all_points = np.vstack([s.points for s in plan.segments if len(s.points) > 0])
    tip_min = np.min(all_points, axis=0)
    tip_max = np.max(all_points, axis=0)

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("; Klein Bottle G-code\n")
        fh.write("; Generated by klein_bottle_gcode_generator.py\n")
        fh.write(f"; write_mode={args.write_mode}\n")
        fh.write("; print_order=inner tube top-down B180 -> uncurl transition B180-to-B0 -> base top-down B0 obstacle-aware -> neck centroid curl\n")
        fh.write("; B-angle convention: 0 deg = +Z/up, 90 deg = horizontal, 180 deg = -Z/down\n")
        fh.write(f"; body center=({args.center_x:.3f},{args.center_y:.3f}) top_z={args.body_top_z:.3f} height={args.body_height:.3f}\n")
        fh.write(f"; body radii=({args.body_radius_x:.3f},{args.body_radius_y:.3f}) mouth_radius={args.mouth_radius:.3f} foot_scale={args.foot_scale:.3f}\n")
        ix, iy = inner_center_xy(args)
        fh.write(f"; inner_tube center=({ix:.3f},{iy:.3f}) radius={args.inner_tube_radius:.3f} z_top={args.inner_tube_top_z:.3f} z_bottom={args.inner_tube_bottom_z:.3f}\n")
        fh.write(f"; obstacle_radius_with_clearance={plan.obstacle_radius:.3f}\n")
        fh.write(f"; base_layers={args.base_layers} ring_segments={args.ring_segments}\n")
        fh.write(f"; neck_enable={bool(args.neck_enable)} neck_radius={args.neck_radius:.3f} neck_height={args.neck_height:.3f}\n")
        fh.write(f"; tip_bbox_min=({tip_min[0]:.3f},{tip_min[1]:.3f},{tip_min[2]:.3f})\n")
        fh.write(f"; tip_bbox_max=({tip_max[0]:.3f},{tip_max[1]:.3f},{tip_max[2]:.3f})\n")
        fh.write(f"; selected_fit_model = {cal.selected_fit_model or 'legacy/cartesian'}\n")
        fh.write(f"; active_phase = {cal.active_phase}\n")
        fh.write(f"; {describe_model(cal.r_model, 'r')}\n")
        fh.write(f"; {describe_model(cal.z_model, 'z')}\n")
        fh.write(f"; {describe_model(cal.y_off_model, 'offplane_y')}\n")
        fh.write(f"; {describe_model(cal.tip_angle_model, 'tip_angle')}\n")
        fh.write(f"; axes X->{cal.x_axis} Y->{cal.y_axis} Z->{cal.z_axis} B->{cal.b_axis} C->{cal.c_axis} U->{cal.u_axis}\n")
        fh.write("G21\nG90\n")
        if args.emit_extrusion:
            fh.write("M42 P0 S0\n")

        writer = KleinGCodeWriter(
            fh=fh,
            cal=cal,
            write_mode=args.write_mode,
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

        for seg in plan.segments:
            writer.print_segment(seg, c_deg=float(args.c_deg), safe_approach_z=float(args.safe_approach_z), travel_lift_z=float(args.travel_lift_z))

        fh.write("\n; SUMMARY\n")
        fh.write(f"; segment_count = {len(plan.segments)}\n")
        fh.write(f"; base_arc_count = {len(plan.base_segments)}\n")
        fh.write(f"; total_printed_path_mm = {writer.total_print_mm:.6f}\n")
        fh.write(f"; total_travel_mm = {writer.total_travel_mm:.6f}\n")
        fh.write(f"; final_U_material_abs = {writer.u_material_abs:.6f}\n")
        if np.all(np.isfinite(writer.stage_min)):
            fh.write(f"; stage_min = ({writer.stage_min[0]:.4f}, {writer.stage_min[1]:.4f}, {writer.stage_min[2]:.4f})\n")
            fh.write(f"; stage_max = ({writer.stage_max[0]:.4f}, {writer.stage_max[1]:.4f}, {writer.stage_max[2]:.4f})\n")
        if np.isfinite(writer.b_machine_min):
            fh.write(f"; B_machine_range = [{writer.b_machine_min:.4f}, {writer.b_machine_max:.4f}]\n")
        for w in writer.warnings:
            fh.write(f"; {w}\n")
        fh.write("; End of file\n")

    return {
        "out": str(out_path),
        "write_mode": args.write_mode,
        "segment_count": len(plan.segments),
        "base_arc_count": len(plan.base_segments),
        "tip_bbox_min": tuple(map(float, tip_min)),
        "tip_bbox_max": tuple(map(float, tip_max)),
        "obstacle_center_xy": plan.obstacle_center_xy,
        "obstacle_radius": float(plan.obstacle_radius),
    }


# =============================================================================
# Preview UI
# =============================================================================
def set_axes_equal(ax: Any) -> None:
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])
    max_range = max(x_range, y_range, z_range)
    x_mid = np.mean(x_limits)
    y_mid = np.mean(y_limits)
    z_mid = np.mean(z_limits)
    ax.set_xlim3d([x_mid - max_range / 2, x_mid + max_range / 2])
    ax.set_ylim3d([y_mid - max_range / 2, y_mid + max_range / 2])
    ax.set_zlim3d([z_mid - max_range / 2, z_mid + max_range / 2])


def draw_preview(ax: Any, args: argparse.Namespace, *, reduced: bool = True) -> None:
    a = copy.copy(args)
    if reduced:
        a.base_layers = min(int(args.base_layers), 28)
        a.ring_segments = min(int(args.ring_segments), 120)
        a.neck_centerline_samples = min(int(args.neck_centerline_samples), 220)
        a.neck_segs_per_turn = min(int(args.neck_segs_per_turn), 14)
        a.minor_segments_per_turn = min(int(args.minor_segments_per_turn), 18)
    plan = build_klein_plan(a)
    ax.clear()
    ax.set_title("Klein bottle quick geometry preview")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")

    # Plot actual tool paths.
    for seg in plan.segments:
        pts = seg.points
        lw = 1.8 if seg.label.startswith("inner") or seg.label.startswith("neck") else 0.8
        ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], linewidth=lw)

    # Plot neck centerline for orientation.
    if plan.neck_centerline is not None:
        cl = plan.neck_centerline
        ax.plot(cl[:, 0], cl[:, 1], cl[:, 2], linestyle="--", linewidth=1.0)

    # Plot obstacle cylinder wireframe.
    ix, iy = plan.obstacle_center_xy
    zc = np.linspace(float(a.inner_tube_bottom_z), float(a.inner_tube_top_z), 24)
    th = np.linspace(0.0, 2.0 * math.pi, 36)
    for z in zc[::6]:
        ax.plot(ix + plan.obstacle_radius * np.cos(th), iy + plan.obstacle_radius * np.sin(th), np.full_like(th, z), linewidth=0.5)
    for th0 in th[::6]:
        ax.plot(np.full_like(zc, ix + plan.obstacle_radius * math.cos(th0)), np.full_like(zc, iy + plan.obstacle_radius * math.sin(th0)), zc, linewidth=0.5)

    # Build-volume/cube visualization if requested.
    if bool(a.preview_cube):
        x0 = float(a.center_x) - float(a.preview_cube_x) / 2.0
        x1 = float(a.center_x) + float(a.preview_cube_x) / 2.0
        y0 = float(a.center_y) - float(a.preview_cube_y) / 2.0
        y1 = float(a.center_y) + float(a.preview_cube_y) / 2.0
        z1 = float(a.body_top_z) + float(a.preview_cube_z) * 0.15
        z0 = z1 - float(a.preview_cube_z)
        corners = np.array([[x0,y0,z0],[x1,y0,z0],[x1,y1,z0],[x0,y1,z0],[x0,y0,z1],[x1,y0,z1],[x1,y1,z1],[x0,y1,z1]])
        edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
        for i,j in edges:
            ax.plot([corners[i,0], corners[j,0]], [corners[i,1], corners[j,1]], [corners[i,2], corners[j,2]], linewidth=0.5)

    all_pts = np.vstack([seg.points for seg in plan.segments if len(seg.points) > 0])
    margin = 8.0
    ax.set_xlim(np.min(all_pts[:, 0]) - margin, np.max(all_pts[:, 0]) + margin)
    ax.set_ylim(np.min(all_pts[:, 1]) - margin, np.max(all_pts[:, 1]) + margin)
    ax.set_zlim(np.min(all_pts[:, 2]) - margin, np.max(all_pts[:, 2]) + margin)
    set_axes_equal(ax)


def preview_ui(args: argparse.Namespace) -> None:
    try:
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider, CheckButtons
    except ImportError as exc:
        raise RuntimeError("Preview requires matplotlib: pip install matplotlib") from exc

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")
    plt.subplots_adjust(left=0.08, right=0.72, bottom=0.08, top=0.95)

    slider_defs = [
        ("body_radius_x", "body Rx", 8.0, 60.0, float(args.body_radius_x)),
        ("body_radius_y", "body Ry", 8.0, 50.0, float(args.body_radius_y)),
        ("body_height", "height", 20.0, 120.0, float(args.body_height)),
        ("inner_x_offset", "inner X off", -40.0, 40.0, float(args.inner_x_offset)),
        ("inner_tube_radius", "inner r", 1.0, 12.0, float(args.inner_tube_radius)),
        ("obstacle_clearance", "clearance", 0.0, 8.0, float(args.obstacle_clearance)),
        ("neck_radius", "neck r", 1.0, 10.0, float(args.neck_radius)),
        ("neck_height", "neck h", 5.0, 80.0, float(args.neck_height)),
        ("neck_outward", "neck out", 0.0, 45.0, float(args.neck_outward)),
        ("transition_radius_end", "trans r", 4.0, 30.0, float(args.transition_radius_end)),
    ]

    sliders: Dict[str, Slider] = {}
    y0 = 0.92
    dy = 0.055
    for i, (attr, label, vmin, vmax, val) in enumerate(slider_defs):
        sax = fig.add_axes([0.76, y0 - i * dy, 0.20, 0.025])
        sliders[attr] = Slider(sax, label, vmin, vmax, valinit=val)

    cax = fig.add_axes([0.76, 0.08, 0.18, 0.07])
    checks = CheckButtons(cax, ["neck", "cube"], [bool(args.neck_enable), bool(args.preview_cube)])

    def apply_values() -> None:
        for attr, slider in sliders.items():
            setattr(args, attr, float(slider.val))
        status = checks.get_status()
        args.neck_enable = bool(status[0])
        args.preview_cube = bool(status[1])

    def update(_val: Any = None) -> None:
        apply_values()
        draw_preview(ax, args, reduced=True)
        fig.canvas.draw_idle()

    for slider in sliders.values():
        slider.on_changed(update)
    checks.on_clicked(update)

    draw_preview(ax, args, reduced=True)
    plt.show()


# =============================================================================
# CLI
# =============================================================================
def add_bool_arg(parser: argparse.ArgumentParser, name: str, default: bool, help_text: str) -> None:
    group = parser.add_mutually_exclusive_group()
    group.add_argument(f"--{name}", dest=name.replace("-", "_"), action="store_true", help=help_text)
    group.add_argument(f"--no-{name}", dest=name.replace("-", "_"), action="store_false", help=f"Disable: {help_text}")
    parser.set_defaults(**{name.replace("-", "_"): default})


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate cobra-style Duet/RRF G-code for a parametric Klein bottle.")

    # Main behavior
    p.add_argument("--out", default=DEFAULT_OUT, help="Output G-code path")
    p.add_argument("--preview", action="store_true", help="Open a small 3D Matplotlib preview with geometry sliders")
    p.add_argument("--no-gcode", action="store_true", help="Do not write G-code; useful with --preview")

    # Geometry controls
    g = p.add_argument_group("Klein bottle geometry")
    g.add_argument("--center-x", type=float, default=DEFAULT_CENTER_X)
    g.add_argument("--center-y", type=float, default=DEFAULT_CENTER_Y)
    g.add_argument("--body-top-z", type=float, default=DEFAULT_BODY_TOP_Z)
    g.add_argument("--body-height", type=float, default=DEFAULT_BODY_HEIGHT)
    g.add_argument("--body-radius-x", type=float, default=DEFAULT_BODY_RADIUS_X)
    g.add_argument("--body-radius-y", type=float, default=DEFAULT_BODY_RADIUS_Y)
    g.add_argument("--mouth-radius", type=float, default=DEFAULT_MOUTH_RADIUS)
    g.add_argument("--foot-scale", type=float, default=DEFAULT_FOOT_SCALE, help="Bottom radius scale relative to body bulge")
    g.add_argument("--shoulder-power", type=float, default=DEFAULT_SHOULDER_POWER)
    g.add_argument("--body-lean-x", type=float, default=DEFAULT_BODY_LEAN_X)
    g.add_argument("--body-lean-y", type=float, default=DEFAULT_BODY_LEAN_Y)

    inner = p.add_argument_group("Inner tube / obstacle")
    inner.add_argument("--inner-x-offset", type=float, default=DEFAULT_INNER_X_OFFSET)
    inner.add_argument("--inner-y-offset", type=float, default=DEFAULT_INNER_Y_OFFSET)
    inner.add_argument("--inner-tube-radius", type=float, default=DEFAULT_INNER_TUBE_RADIUS)
    inner.add_argument("--inner-tube-top-z", type=float, default=DEFAULT_INNER_TUBE_TOP_Z)
    inner.add_argument("--inner-tube-bottom-z", type=float, default=DEFAULT_INNER_TUBE_BOTTOM_Z)
    inner.add_argument("--obstacle-clearance", type=float, default=DEFAULT_OBSTACLE_CLEARANCE)

    trans = p.add_argument_group("Uncurling transition")
    trans.add_argument("--transition-turns", type=float, default=DEFAULT_TRANSITION_TURNS)
    trans.add_argument("--transition-samples", type=int, default=DEFAULT_TRANSITION_SAMPLES)
    trans.add_argument("--transition-radius-end", type=float, default=DEFAULT_TRANSITION_RADIUS_END)
    trans.add_argument("--transition-z-drop", type=float, default=DEFAULT_TRANSITION_Z_DROP)

    base = p.add_argument_group("Base contour writing")
    base.add_argument("--base-layers", type=int, default=DEFAULT_BASE_LAYERS)
    base.add_argument("--ring-segments", type=int, default=DEFAULT_RING_SEGMENTS)
    base.add_argument("--min-arc-points", type=int, default=DEFAULT_MIN_ARC_POINTS)
    base.add_argument("--line-spacing-z", type=float, default=DEFAULT_LINE_SPACING_Z, help="Reserved/tuning note; base-layers currently controls layer count")

    neck = p.add_argument_group("Neck/curl")
    add_bool_arg(neck, "neck-enable", DEFAULT_NECK_ENABLE, "Generate the neck/curl pass")
    neck.add_argument("--neck-radius", type=float, default=DEFAULT_NECK_RADIUS)
    neck.add_argument("--neck-centerline-samples", type=int, default=DEFAULT_NECK_CENTERLINE_SAMPLES)
    neck.add_argument("--neck-segs-per-turn", type=int, default=DEFAULT_NECK_SEGS_PER_TURN)
    neck.add_argument("--neck-attach-angle-deg", type=float, default=DEFAULT_NECK_ATTACH_ANGLE_DEG)
    neck.add_argument("--neck-attach-drop", type=float, default=DEFAULT_NECK_ATTACH_DROP)
    neck.add_argument("--neck-outward", type=float, default=DEFAULT_NECK_OUTWARD)
    neck.add_argument("--neck-height", type=float, default=DEFAULT_NECK_HEIGHT)
    neck.add_argument("--neck-curl-b-blend-frac", type=float, default=DEFAULT_NECK_CURL_B_BLEND_FRAC)
    neck.add_argument("--neck-twist-turns", type=float, default=DEFAULT_NECK_TWIST_TURNS)

    # Helix and extrusion scales
    h = p.add_argument_group("Path density / extrusion scales")
    h.add_argument("--layer-height", type=float, default=DEFAULT_LAYER_HEIGHT)
    h.add_argument("--minor-segments-per-turn", type=int, default=DEFAULT_MINOR_SEGMENTS_PER_TURN)
    h.add_argument("--phi0-deg", type=float, default=DEFAULT_PHI0_DEG)
    h.add_argument("--frame-flip-outplane-sign", type=float, default=DEFAULT_FRAME_FLIP_OUTPLANE_SIGN)
    h.add_argument("--inner-extrusion-scale", type=float, default=1.0)
    h.add_argument("--transition-extrusion-scale", type=float, default=1.0)
    h.add_argument("--base-extrusion-scale", type=float, default=1.0)
    h.add_argument("--neck-extrusion-scale", type=float, default=1.0)

    # Machine / calibration
    m = p.add_argument_group("Machine / calibration")
    m.add_argument("--write-mode", choices=["calibrated", "cartesian"], default=DEFAULT_WRITE_MODE)
    m.add_argument("--calibration", default=None, help="Calibration JSON for calibrated mode")
    m.add_argument("--c-deg", type=float, default=DEFAULT_C_DEG)
    m.add_argument("--b-angle-bias-deg", type=float, default=DEFAULT_B_ANGLE_BIAS_DEG)
    m.add_argument("--bc-solve-samples", type=int, default=DEFAULT_BC_SOLVE_SAMPLES)
    m.add_argument("--y-offplane-sign", type=float, default=DEFAULT_Y_OFFPLANE_SIGN)
    m.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED)
    m.add_argument("--approach-feed", type=float, default=DEFAULT_APPROACH_FEED)
    m.add_argument("--fine-approach-feed", type=float, default=DEFAULT_FINE_APPROACH_FEED)
    m.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED)
    m.add_argument("--travel-lift-z", type=float, default=DEFAULT_TRAVEL_LIFT_Z)
    m.add_argument("--safe-approach-z", type=float, default=DEFAULT_SAFE_APPROACH_Z)
    add_bool_arg(m, "emit-extrusion", DEFAULT_EMIT_EXTRUSION, "Emit pressure on/off M42 extrusion control comments and pulses")
    m.add_argument("--extrusion-per-mm", type=float, default=DEFAULT_EXTRUSION_PER_MM)
    m.add_argument("--extrusion-multiplier", type=float, default=DEFAULT_EXTRUSION_MULTIPLIER)
    m.add_argument("--pressure-offset-mm", type=float, default=DEFAULT_PRESSURE_OFFSET_MM)
    m.add_argument("--pressure-advance-feed", type=float, default=DEFAULT_PRESSURE_ADVANCE_FEED)
    m.add_argument("--pressure-retract-feed", type=float, default=DEFAULT_PRESSURE_RETRACT_FEED)
    m.add_argument("--preflow-dwell-ms", type=int, default=DEFAULT_PREFLOW_DWELL_MS)
    m.add_argument("--end-dwell-ms", type=int, default=DEFAULT_END_DWELL_MS)

    # Bounding box clamps
    b = p.add_argument_group("Stage bounding box clamps")
    b.add_argument("--bbox-x-min", type=float, default=-DEFAULT_BBOX)
    b.add_argument("--bbox-x-max", type=float, default=DEFAULT_BBOX)
    b.add_argument("--bbox-y-min", type=float, default=-DEFAULT_BBOX)
    b.add_argument("--bbox-y-max", type=float, default=DEFAULT_BBOX)
    b.add_argument("--bbox-z-min", type=float, default=-DEFAULT_BBOX)
    b.add_argument("--bbox-z-max", type=float, default=DEFAULT_BBOX)

    # Preview cube/build volume
    v = p.add_argument_group("Preview build cube")
    add_bool_arg(v, "preview-cube", False, "Show a wireframe build cube in preview")
    v.add_argument("--preview-cube-x", type=float, default=90.0)
    v.add_argument("--preview-cube-y", type=float, default=80.0)
    v.add_argument("--preview-cube-z", type=float, default=110.0)

    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.preview:
        preview_ui(args)

    if args.no_gcode:
        return 0

    summary = write_klein_gcode(args)
    print("Wrote Klein bottle G-code:")
    for key, val in summary.items():
        print(f"  {key}: {val}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
