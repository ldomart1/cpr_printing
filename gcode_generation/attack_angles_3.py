#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code for:
  1) A prepended X-line stack in the XZ plane
  2) The requested XZ-plane line/rectangle-angle test pattern

using calibration-based tip-position planning
(exact tip tracking via stage = tip - offset_tip(B,C)).

Prepended line-stack behavior
-----------------------------
- Prints 10 straight lines before the existing pattern.
- Each line is 60 mm long in +X.
- All lines lie in the XZ plane with constant Y=20 mm.
- First line starts at X=50 mm, Z=-175 mm.
- Each successive line is offset by +1 mm in Z.
- The lines alternate direction:
    line 1: X=50 -> 110
    line 2: X=110 -> 50
    line 3: X=50 -> 110
    ...
  so after moving up by +1 mm in Z, the next line is extruded back toward the original X.

Existing pattern behavior
-------------------------
- Uses calibration JSON (cubic_coefficients + axis mapping), just like your other scripts.
- Computes B command by solving the active calibration tip-angle model for 0.00 deg.
- Prefers `fit_models_by_phase.pull` when present, and falls back to the legacy
  cubic coefficients only if the phase-aware model is missing.
- Keeps B fixed at that value during printing and travel for pattern generation.
- Keeps C fixed at 0 deg during printing and travel (but supports a separate C-only feedrate).
- Uses XYZ stage moves computed from desired tip-space geometry using the calibration offset.
- Preserves U-axis extrusion / pressure preload / dwell / retract math style.
- Step 2 second-half fix:
    lines start at the end of 95°, then end of 100°, ... through end of 135°,
    and pair with 85°, 80°, ..., 45° lines to the vertical-top point.

Coordinate convention
---------------------
- Prepended lines are parallel to X at varying Z values, with constant Y.
- Requested geometry is described in the XZ plane for the existing pattern.
- Calibration converts desired tip XYZ to stage XYZ.

Important note on prompt ambiguity
----------------------------------
The prompt mentioned extending the 90° line to y_rel=80mm while describing XZ-plane geometry.
This script interprets that as extending the 90° line vertically in Z to z_rel=80mm in the XZ plane,
which is geometrically consistent and matches the "straight line only" intent.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------- Defaults ----------------
DEFAULT_OUT = "gcode_generation/xz_line_pattern_test_20260306.gcode"

# Pattern placement in TIP SPACE (world coordinates)
DEFAULT_START_X = 40.0
DEFAULT_START_Y = 60.0
DEFAULT_START_Z = -195.0

# Prepended X-line stack defaults (TIP SPACE)
DEFAULT_PREP_LINE_START_X = 40.0
DEFAULT_PREP_LINE_START_Y = 60.0
DEFAULT_PREP_LINE_START_Z = -195.0
DEFAULT_PREP_LINE_LENGTH = 60.0
DEFAULT_PREP_LINE_COUNT = 10
DEFAULT_PREP_LINE_Z_STEP = 1.0

# Startup/end MACHINE STAGE poses (raw stage axes)
DEFAULT_MACHINE_START_X = 40.0
DEFAULT_MACHINE_START_Y = 60.0
DEFAULT_MACHINE_START_Z = -20.0
DEFAULT_MACHINE_START_B = 0.0
DEFAULT_MACHINE_START_C = 0.0

DEFAULT_MACHINE_END_X = 100.0
DEFAULT_MACHINE_END_Y = 60.0
DEFAULT_MACHINE_END_Z = -20.0
DEFAULT_MACHINE_END_B = 0.0
DEFAULT_MACHINE_END_C = 0.0
DEFAULT_SAFE_APPROACH_Z = 0.0

# Motion
DEFAULT_TRAVEL_FEED = 1000.0       # mm/min
DEFAULT_PRINT_FEED = 200.0         # mm/min
DEFAULT_C_FEED = 5000.0            # deg/min (requested separate feed)

# Geometry specifics
DEFAULT_LINE_LENGTH = 60.0
DEFAULT_MID_X = 30.0
DEFAULT_ANGLE_STEP = 5.0
DEFAULT_ANGLE_MAX_1 = 45.0
DEFAULT_RETURN_LIFT_Z = 50.0
DEFAULT_VERTICAL_LINE_LEN = 15.0

DEFAULT_PLANE_Y_STEP = 5.0
DEFAULT_STEP2_ORIGIN_X_REL = 30.0
DEFAULT_STEP2_Z_TARGET_REL = 30.0
DEFAULT_STEP2_STRAIGHT_EXTEND_Z_REL = 60.0
DEFAULT_TOP_HORIZONTAL_LEN = 5.0

# Print interpolation
DEFAULT_EDGE_SAMPLES = 24

# Extrusion (coordinated with print feed)
DEFAULT_EXTRUSION_PER_MM = 0.0015
DEFAULT_PRIME_MM = 1.0

# Pressure offset / dwell sequencing (U-axis)
DEFAULT_PRESSURE_OFFSET_MM = 4.0
DEFAULT_PRESSURE_ADVANCE_FEED = 120.0
DEFAULT_PRESSURE_RETRACT_FEED = 240.0
DEFAULT_PREFLOW_DWELL_MS = 500
DEFAULT_NODE_DWELL_MS = 750

DEFAULT_EXTRUSION_MULTIPLIERS = (1.0, 1.5, 2.0)

# Virtual stage-space bounding box
DEFAULT_BBOX_X_MIN = 40.0
DEFAULT_BBOX_X_MAX = 180.0
DEFAULT_BBOX_Y_MIN = 0.0
DEFAULT_BBOX_Y_MAX = 200.0
DEFAULT_BBOX_Z_MIN = -200.0
DEFAULT_BBOX_Z_MAX = 00.0


# ---------------- Data classes ----------------

@dataclass
class Calibration:
    pr: np.ndarray
    pz: np.ndarray
    py_off: Optional[np.ndarray]
    pa: Optional[np.ndarray]  # tip_angle_coeffs

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
    tip_angle_model: Optional[Dict[str, Any]] = None
    selected_fit_model: Optional[str] = None
    active_phase: str = "pull"


@dataclass
class Segment:
    """
    Tip-space print segment.
    """
    p0_tip: np.ndarray
    p1_tip: np.ndarray
    label: str


# ---------------- Calibration / kinematics helpers ----------------

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
    selected_fit_model = data.get("selected_fit_model")
    selected_fit_model = None if selected_fit_model is None else str(selected_fit_model).strip().lower()
    active_phase = str(data.get("default_phase_for_legacy_access") or "pull").strip().lower()

    fit_models = data.get("fit_models", {}) or {}
    phase_models = data.get("fit_models_by_phase", {}) or {}
    active_phase_models = phase_models.get(active_phase) if isinstance(phase_models, dict) else None
    if not isinstance(active_phase_models, dict):
        active_phase_models = fit_models

    r_model = _normalize_model_spec(active_phase_models.get("r"))
    z_model = _normalize_model_spec(active_phase_models.get("z"))
    y_off_model = _normalize_model_spec(active_phase_models.get("offplane_y"))
    tip_angle_model = _normalize_model_spec(active_phase_models.get("tip_angle"))

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
    u_axis = str(duet_map.get("extruder_axis") or "U")
    c_180 = float(motor_setup.get("rotation_axis_180_deg", 180.0))

    return Calibration(
        pr=pr, pz=pz, py_off=py_off, pa=pa,
        r_model=r_model, z_model=z_model, y_off_model=y_off_model, tip_angle_model=tip_angle_model,
        selected_fit_model=selected_fit_model, active_phase=active_phase,
        b_min=b_min, b_max=b_max,
        x_axis=x_axis, y_axis=y_axis, z_axis=z_axis,
        b_axis=b_axis, c_axis=c_axis, u_axis=u_axis,
        c_180_deg=c_180
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
        return eval_model_spec(cal.y_off_model, b, default_if_none=0.0)
    return poly_eval(cal.py_off, b, default_if_none=0.0)


def eval_tip_angle_deg(cal: Calibration, b: Any) -> np.ndarray:
    if cal.tip_angle_model is not None:
        return eval_model_spec(cal.tip_angle_model, b)
    if cal.pa is None:
        raise ValueError("Calibration is missing tip_angle_coeffs.")
    return poly_eval(cal.pa, b)


def predict_tip_xyz_from_bc(cal: Calibration, b: float, c_deg: float) -> np.ndarray:
    """
    Physical tip offset from stage origin:
      local transverse [r(B), y_off(B)] rotated by C into XY, z=z(B)
    """
    r = float(eval_r(cal, b))
    z = float(eval_z(cal, b))
    y_off = float(eval_offplane_y(cal, b))

    c = math.radians(float(c_deg))
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def tip_offset_xyz_physical(cal: Calibration, b: float, c_deg: float) -> np.ndarray:
    return predict_tip_xyz_from_bc(cal, b, c_deg)


def stage_xyz_for_tip(cal: Calibration, tip_xyz: np.ndarray, b: float, c_deg: float) -> np.ndarray:
    """
    Exact tip-position planning:
      p_stage = p_tip_desired - offset_tip(B, C)
    """
    return np.asarray(tip_xyz, dtype=float) - tip_offset_xyz_physical(cal, b, c_deg)


def solve_b_for_tip_angle_zero(cal: Calibration, search_samples: int = 5001) -> float:
    """
    Solve for B such that the active tip-angle calibration model evaluates to 0 deg,
    restricted to the calibration B range.
    """
    if cal.pa is None:
        if cal.tip_angle_model is None:
            raise ValueError("Calibration JSON has no tip-angle model; cannot solve B for 0 deg angle.")

    b_lo, b_hi = float(cal.b_min), float(cal.b_max)
    bb = np.linspace(b_lo, b_hi, int(max(101, search_samples)))
    aa = eval_tip_angle_deg(cal, bb)

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
            return float(eval_tip_angle_deg(cal, x))

        flo = f(lo)
        fhi = f(hi)
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fmid = f(mid)
            if abs(fmid) < 1e-10:
                return float(mid)
            if flo * fmid <= 0.0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid
        return float(0.5 * (lo + hi))

    return b_best_abs


# ---------------- Utility / extrusion math ----------------

def _fmt_axes_move(axes_vals: List[Tuple[str, float]]) -> str:
    return " ".join(f"{ax}{val:.3f}" for ax, val in axes_vals)


def _clamp_stage_xyz_to_bbox(
    x: float, y: float, z: float, bbox: dict, context: str, warn_log: List[str]
) -> Tuple[float, float, float]:
    def clamp_one(axis: str, value: float, lo: float, hi: float) -> float:
        if value < lo:
            warn_log.append(f"WARNING: {context} {axis}={value:.3f} below bbox min {lo:.3f}; clamped to {lo:.3f}")
            return lo
        if value > hi:
            warn_log.append(f"WARNING: {context} {axis}={value:.3f} above bbox max {hi:.3f}; clamped to {hi:.3f}")
            return hi
        return value

    xc = clamp_one("X", float(x), float(bbox["x_min"]), float(bbox["x_max"]))
    yc = clamp_one("Y", float(y), float(bbox["y_min"]), float(bbox["y_max"]))
    zc = clamp_one("Z", float(z), float(bbox["z_min"]), float(bbox["z_max"]))
    return xc, yc, zc


def tube_area_mm2_from_id_inch(id_inch: float) -> float:
    d_mm = float(id_inch) * 25.4
    r_mm = 0.5 * d_mm
    return math.pi * r_mm * r_mm


def extrusion_math_summary(
    syringe_mm_per_ml: float,
    tube_id_inch: float,
    print_feed_mm_min: float,
    extrusion_per_mm_u: float,
) -> dict:
    if syringe_mm_per_ml <= 0:
        raise ValueError("syringe_mm_per_ml must be > 0.")
    if print_feed_mm_min <= 0:
        raise ValueError("print_feed_mm_min must be > 0.")

    path_speed_mm_s = float(print_feed_mm_min) / 60.0
    u_speed_mm_s = float(extrusion_per_mm_u) * path_speed_mm_s

    q_mm3_s = (1000.0 / float(syringe_mm_per_ml)) * u_speed_mm_s
    q_ml_min = q_mm3_s * 60.0 / 1000.0
    q_ul_s = q_mm3_s

    area_tube_mm2 = tube_area_mm2_from_id_inch(float(tube_id_inch))
    tube_velocity_mm_s = q_mm3_s / area_tube_mm2 if area_tube_mm2 > 0 else float("nan")
    tube_velocity_m_s = tube_velocity_mm_s / 1000.0

    return {
        "path_speed_mm_s": path_speed_mm_s,
        "u_speed_mm_s": u_speed_mm_s,
        "q_mm3_s": q_mm3_s,
        "q_ml_min": q_ml_min,
        "q_ul_s": q_ul_s,
        "tube_id_mm": float(tube_id_inch) * 25.4,
        "tube_area_mm2": area_tube_mm2,
        "tube_velocity_mm_s": tube_velocity_mm_s,
        "tube_velocity_m_s": tube_velocity_m_s,
    }


# ---------------- Geometry builders (TIP SPACE) ----------------

def p_xyz(x: float, y: float, z: float) -> np.ndarray:
    return np.array([float(x), float(y), float(z)], dtype=float)


def make_prep_line_segments(
    start_x: float,
    start_y: float,
    start_z: float,
    line_length: float,
    line_count: int,
    z_step: float,
    label_prefix: str,
) -> List[Segment]:
    """
    Build a stack of straight X-parallel lines in the XZ plane at constant Y.
    Alternate direction each line so that after stepping +Z, the next line
    extrudes back toward the original X.
    """
    segs: List[Segment] = []
    x0 = float(start_x)
    x1 = float(start_x) + float(line_length)
    y = float(start_y)

    for i in range(int(line_count)):
        z_i = float(start_z) + float(i) * float(z_step)
        if i % 2 == 0:
            p0 = p_xyz(x0, y, z_i)
            p1 = p_xyz(x1, y, z_i)
        else:
            p0 = p_xyz(x1, y, z_i)
            p1 = p_xyz(x0, y, z_i)
        segs.append(Segment(p0, p1, f"{label_prefix}_{i+1:02d}"))
    return segs


def make_step1_segments(start_tip: np.ndarray, y_plane: float,
                        line_len: float, mid_x: float,
                        angle_step: float, angle_max: float,
                        vertical_len: float) -> List[Segment]:
    sx, _, sz = [float(v) for v in start_tip]
    segs: List[Segment] = []

    p0 = p_xyz(sx, y_plane, sz)
    p1 = p_xyz(sx + line_len, y_plane, sz)
    segs.append(Segment(p0, p1, "step1_horizontal_0deg"))

    angles = np.arange(float(angle_step), float(angle_max) + 1e-9, float(angle_step))
    for a in angles:
        t = math.tan(math.radians(float(a)))
        z_mid = sz + t * mid_x
        p_start = p_xyz(sx, y_plane, sz)
        p_mid = p_xyz(sx + mid_x, y_plane, z_mid)
        p_end = p_xyz(sx + line_len, y_plane, sz)
        segs.append(Segment(p_start, p_mid, f"step1_roof_a{a:.0f}_seg1"))
        segs.append(Segment(p_mid, p_end,   f"step1_roof_a{a:.0f}_seg2"))

    pv0 = p_xyz(sx, y_plane, sz)
    pv1 = p_xyz(sx, y_plane, sz + vertical_len)
    segs.append(Segment(pv0, pv1, "step1_vertical_20mm"))
    return segs


def _ray_to_z_rel(origin: np.ndarray, angle_deg: float, z_rel_target: float) -> np.ndarray:
    theta = math.radians(float(angle_deg))
    s = math.sin(theta)
    c = math.cos(theta)
    if abs(s) < 1e-12:
        raise ValueError(f"Angle {angle_deg} deg cannot reach z_rel target (sin≈0).")
    L = float(z_rel_target) / s
    return origin + np.array([L * c, 0.0, L * s], dtype=float)


def make_step2_segments(
    start_step1_tip: np.ndarray,
    y_plane: float,
    origin_x_rel: float,
    z_target_rel: float,
    straight_extend_z_rel: float,
    top_horizontal_len: float,
) -> Tuple[List[Segment], np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """
    Returns:
      segments, step2_origin, vertical_top, endpoints_famA_by_angle
    """
    sx, _, sz = [float(v) for v in start_step1_tip]
    origin = p_xyz(sx + origin_x_rel, y_plane, sz)

    segs: List[Segment] = []
    endpoints_famA: Dict[int, np.ndarray] = {}

    for ang in range(135, 90, -5):  # 135..95
        p_end = _ray_to_z_rel(origin, ang, z_target_rel)
        endpoints_famA[int(ang)] = p_end.copy()
        segs.append(Segment(origin.copy(), p_end, f"step2_famA_{ang}deg"))

    p90_end = _ray_to_z_rel(origin, 90.0, straight_extend_z_rel)
    segs.append(Segment(origin.copy(), p90_end, "step2_famA_90deg_extended"))
    vertical_top = p90_end.copy()

    start_angles = list(range(95, 140, 5))     # 95,100,...,135
    draw_angles = list(range(85, 40, -5))      # 85,80,...,45

    if len(start_angles) != len(draw_angles):
        raise RuntimeError("Step2 family-B pairing mismatch.")

    for start_ang, draw_ang in zip(start_angles, draw_angles):
        p0 = endpoints_famA[int(start_ang)].copy()
        p1 = vertical_top.copy()
        segs.append(Segment(p0, p1, f"step2_famB_{draw_ang}deg_from_end{start_ang}"))

    top_h0 = vertical_top.copy()
    top_h1 = vertical_top + np.array([float(top_horizontal_len), 0.0, 0.0], dtype=float)
    segs.append(Segment(top_h0, top_h1, "step2_top_horizontal_10mm"))

    return segs, origin, vertical_top, endpoints_famA


# ---------------- G-code writer (calibrated tip planning) ----------------

class CalibratedPatternWriter:
    def __init__(
        self,
        fh,
        cal: Calibration,
        b_fixed: float,
        c_fixed: float,
        bbox: dict,
        travel_feed: float,
        print_feed: float,
        c_feed: float,
        extrusion_per_mm: float,
        pressure_offset_mm: float,
        pressure_advance_feed: float,
        pressure_retract_feed: float,
        preflow_dwell_ms: int,
        node_dwell_ms: int,
        edge_samples: int,
        emit_extrusion: bool,
        warn_log: List[str],
    ):
        self.f = fh
        self.cal = cal
        self.b_fixed = float(b_fixed)
        self.c_fixed = float(c_fixed)
        self.tip_offset_fixed = tip_offset_xyz_physical(cal, self.b_fixed, self.c_fixed)

        self.bbox = bbox
        self.travel_feed = float(travel_feed)
        self.print_feed = float(print_feed)
        self.c_feed = float(c_feed)

        self.extrusion_per_mm = float(extrusion_per_mm)
        self.pressure_offset_mm = float(pressure_offset_mm)
        self.pressure_advance_feed = float(pressure_advance_feed)
        self.pressure_retract_feed = float(pressure_retract_feed)
        self.preflow_dwell_ms = int(preflow_dwell_ms)
        self.node_dwell_ms = int(node_dwell_ms)
        self.edge_samples = max(2, int(edge_samples))
        self.emit_extrusion = bool(emit_extrusion)
        self.warn_log = warn_log

        self.u_material_abs = 0.0
        self.pressure_charged = False

        self.cur_stage_xyz: Optional[np.ndarray] = None
        self.cur_b = 0.0
        self.cur_c = 0.0

    def _clamp_stage(self, p_stage: np.ndarray, context: str) -> np.ndarray:
        x, y, z = _clamp_stage_xyz_to_bbox(
            p_stage[0], p_stage[1], p_stage[2], self.bbox, context, self.warn_log
        )
        return np.array([x, y, z], dtype=float)

    def u_cmd_actual(self) -> float:
        return self.u_material_abs + (self.pressure_offset_mm if self.pressure_charged else 0.0)

    def tip_to_stage(self, p_tip: np.ndarray) -> np.ndarray:
        return stage_xyz_for_tip(self.cal, p_tip, self.b_fixed, self.c_fixed)

    def move_c_only(self, c_target: float, comment: Optional[str] = None):
        if comment:
            self.f.write(f"; {comment}\n")
        self.f.write(f"G1 {self.cal.c_axis}{float(c_target):.3f} F{self.c_feed:.0f}\n")
        self.cur_c = float(c_target)

    def move_stage_xyzbc(self, p_stage: np.ndarray, b: Optional[float] = None, c: Optional[float] = None,
                         feed: Optional[float] = None, comment: Optional[str] = None):
        if comment:
            self.f.write(f"; {comment}\n")
        pc = self._clamp_stage(np.asarray(p_stage, dtype=float), comment or "move_stage_xyzbc")
        b_use = self.b_fixed if b is None else float(b)
        c_use = self.c_fixed if c is None else float(c)
        fval = self.travel_feed if feed is None else float(feed)

        axes = [
            (self.cal.x_axis, pc[0]),
            (self.cal.y_axis, pc[1]),
            (self.cal.z_axis, pc[2]),
            (self.cal.b_axis, b_use),
            (self.cal.c_axis, c_use),
        ]
        self.f.write(f"G1 {_fmt_axes_move(axes)} F{fval:.0f}\n")
        self.cur_stage_xyz = pc.copy()
        self.cur_b = b_use
        self.cur_c = c_use

    def move_to_tip(self, p_tip: np.ndarray, feed: Optional[float] = None, comment: Optional[str] = None):
        p_stage = self.tip_to_stage(np.asarray(p_tip, dtype=float))
        self.move_stage_xyzbc(p_stage, b=self.b_fixed, c=self.c_fixed, feed=feed, comment=comment)

    def safe_travel_z_up_x_then_z_down_tip(self, target_tip: np.ndarray, lift_dz_stage: float, comment_prefix: str):
        if self.cur_stage_xyz is None:
            raise RuntimeError("Current stage XYZ unknown before safe travel.")

        cur = self.cur_stage_xyz.copy()

        z_up = cur.copy()
        z_up[2] = cur[2] + float(lift_dz_stage)
        self.move_stage_xyzbc(z_up, feed=self.travel_feed, comment=f"{comment_prefix}: raise Z")

        target_stage = self.tip_to_stage(target_tip)
        xmove = z_up.copy()
        xmove[0] = target_stage[0]
        xmove[1] = target_stage[1]
        self.move_stage_xyzbc(xmove, feed=self.travel_feed, comment=f"{comment_prefix}: move X/Y")

        dive = xmove.copy()
        dive[2] = target_stage[2]
        self.move_stage_xyzbc(dive, feed=self.travel_feed, comment=f"{comment_prefix}: dive Z")

    def safe_travel_x_first_then_down_z_tip(self, target_tip: np.ndarray, comment_prefix: str):
        if self.cur_stage_xyz is None:
            raise RuntimeError("Current stage XYZ unknown before safe travel.")

        cur = self.cur_stage_xyz.copy()
        target_stage = self.tip_to_stage(target_tip)

        xmove = cur.copy()
        xmove[0] = target_stage[0]
        xmove[1] = target_stage[1]
        self.move_stage_xyzbc(xmove, feed=self.travel_feed, comment=f"{comment_prefix}: move X/Y first")

        zmove = xmove.copy()
        zmove[2] = target_stage[2]
        self.move_stage_xyzbc(zmove, feed=self.travel_feed, comment=f"{comment_prefix}: down Z")

    def pressure_preload_before_print(self):
        if self.emit_extrusion and self.pressure_offset_mm > 0.0 and not self.pressure_charged:
            self.pressure_charged = True
            self.f.write("; pressure preload before print pass\n")
            self.f.write(f"G1 {self.cal.u_axis}{self.u_cmd_actual():.3f} F{self.pressure_advance_feed:.0f}\n")
            if self.preflow_dwell_ms > 0:
                self.f.write(f"G4 P{self.preflow_dwell_ms}\n")

    def pressure_release_after_print(self):
        if self.emit_extrusion and self.pressure_offset_mm > 0.0 and self.pressure_charged:
            if self.node_dwell_ms > 0:
                self.f.write("; end-of-pass dwell for node formation / liquid flow\n")
                self.f.write(f"G4 P{self.node_dwell_ms}\n")
            self.pressure_charged = False
            self.f.write("; pressure release after print pass\n")
            self.f.write(f"G1 {self.cal.u_axis}{self.u_cmd_actual():.3f} F{self.pressure_retract_feed:.0f}\n")

    def print_tip_segment(self, seg: Segment, extrusion_multiplier: float):
        p0_tip = np.asarray(seg.p0_tip, dtype=float)
        p1_tip = np.asarray(seg.p1_tip, dtype=float)

        p0_stage = self.tip_to_stage(p0_tip)
        if self.cur_stage_xyz is None or np.linalg.norm(self.cur_stage_xyz - self._clamp_stage(p0_stage, "compare p0")) > 1e-6:
            self.move_stage_xyzbc(p0_stage, feed=self.travel_feed, comment=f"{seg.label}: move to print start")

        self.pressure_preload_before_print()

        self.f.write(f"; print {seg.label} (tip-space exact tracking, fixed B/C)\n")
        ts = np.linspace(0.0, 1.0, self.edge_samples + 1)
        last_stage = self.cur_stage_xyz.copy()

        for j in range(1, len(ts)):
            t = float(ts[j])
            p_tip = p0_tip + t * (p1_tip - p0_tip)
            p_stage = self.tip_to_stage(p_tip)

            if np.linalg.norm((p_stage + self.tip_offset_fixed) - p_tip) > 1e-8:
                raise RuntimeError(f"Tip-tracking consistency failed in {seg.label} segment {j}.")

            pc = self._clamp_stage(p_stage, f"{seg.label} seg {j}")
            axes = [
                (self.cal.x_axis, pc[0]),
                (self.cal.y_axis, pc[1]),
                (self.cal.z_axis, pc[2]),
                (self.cal.b_axis, self.b_fixed),
                (self.cal.c_axis, self.c_fixed),
            ]

            if self.emit_extrusion:
                seg_len = float(np.linalg.norm(pc - last_stage))
                self.u_material_abs += (self.extrusion_per_mm * float(extrusion_multiplier)) * seg_len
                axes.append((self.cal.u_axis, self.u_cmd_actual()))

            self.f.write(f"G1 {_fmt_axes_move(axes)} F{self.print_feed:.0f}\n")
            last_stage = pc.copy()

        self.cur_stage_xyz = last_stage
        self.cur_b = self.b_fixed
        self.cur_c = self.c_fixed

        self.pressure_release_after_print()


# ---------------- Pattern generation ----------------

def write_pattern_gcode(
    out_path: str,
    cal: Calibration,
    prep_line_start_tip: Tuple[float, float, float],
    prep_line_length: float,
    prep_line_count: int,
    prep_line_z_step: float,
    pattern_start_tip: Tuple[float, float, float],
    machine_start_pose: Tuple[float, float, float, float, float],
    machine_end_pose: Tuple[float, float, float, float, float],
    safe_approach_z: float,
    travel_feed: float,
    print_feed: float,
    c_feed: float,
    edge_samples: int,
    extrusion_per_mm: float,
    prime_mm: float,
    pressure_offset_mm: float,
    pressure_advance_feed: float,
    pressure_retract_feed: float,
    preflow_dwell_ms: int,
    node_dwell_ms: int,
    bbox: dict,
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    b_zero_angle = solve_b_for_tip_angle_zero(cal)
    c_fixed = 0.0
    tip_offset = tip_offset_xyz_physical(cal, b_zero_angle, c_fixed)
    tip_angle_at_b = float(eval_tip_angle_deg(cal, b_zero_angle))

    emit_extrusion = float(extrusion_per_mm) != 0.0
    bbox_warnings: List[str] = []

    sx, sy, sz = [float(v) for v in pattern_start_tip]
    plx, ply, plz = [float(v) for v in prep_line_start_tip]

    with open(out_path, "w") as f:
        f.write("; generated by xz_line_pattern_test_calibrated.py\n")
        f.write("; calibration-based tip-position planning (exact stage = tip - offset_tip(B,C))\n")
        f.write("; prepended pattern: X-line stack in XZ plane\n")
        f.write("; main pattern: XZ-plane line tests in Y-offset planes\n")
        f.write("; B chosen by solving active tip-angle calibration model = 0 deg (within calibration B range)\n")
        f.write(f"; axes: X->{cal.x_axis}, Y->{cal.y_axis}, Z->{cal.z_axis}, B->{cal.b_axis}, C->{cal.c_axis}, U->{cal.u_axis}\n")
        f.write(f"; selected_fit_model = {cal.selected_fit_model or 'legacy-polynomial'}\n")
        f.write(f"; active_phase = {cal.active_phase}\n")
        f.write(f"; calibration B range: [{cal.b_min:.6f}, {cal.b_max:.6f}]\n")
        f.write(f"; solved fixed B for tip_angle=0 deg: {b_zero_angle:.6f}\n")
        f.write(f"; tip_angle_model(B_fixed) = {tip_angle_at_b:.9f} deg\n")
        f.write(f"; fixed C = {c_fixed:.3f} deg\n")
        f.write(f"; fixed tip offset(B_fixed,C_fixed) = [{tip_offset[0]:.6f}, {tip_offset[1]:.6f}, {tip_offset[2]:.6f}] mm\n")
        f.write(f"; prepended line start (first line) = [{plx:.3f}, {ply:.3f}, {plz:.3f}] mm\n")
        f.write(f"; prepended line length (+X magnitude) = {float(prep_line_length):.3f} mm\n")
        f.write(f"; prepended line count = {int(prep_line_count)}\n")
        f.write(f"; prepended line z step = {float(prep_line_z_step):.3f} mm\n")
        f.write("; prepended line directions alternate in X on successive Z levels\n")
        f.write(f"; pattern start (tip-space) = [{sx:.3f}, {sy:.3f}, {sz:.3f}]\n")
        f.write(f"; feeds: travel={travel_feed:.1f} mm/min, print={print_feed:.1f} mm/min, C-only={c_feed:.1f} deg/min\n")
        f.write(f"; extrusion_per_mm(base) = {extrusion_per_mm:.6f} U/mm-path\n")
        f.write(f"; extrusion multipliers = {DEFAULT_EXTRUSION_MULTIPLIERS}\n")
        f.write("G90\n")

        if emit_extrusion:
            f.write("M82\n")
            f.write(f"G92 {cal.u_axis}0\n")
            if abs(float(prime_mm)) > 0.0:
                f.write(f"G1 {cal.u_axis}{float(prime_mm):.3f} F{max(60.0, float(pressure_advance_feed)):.0f} ; prime material\n")

        g = CalibratedPatternWriter(
            fh=f,
            cal=cal,
            b_fixed=b_zero_angle,
            c_fixed=c_fixed,
            bbox=bbox,
            travel_feed=travel_feed,
            print_feed=print_feed,
            c_feed=c_feed,
            extrusion_per_mm=extrusion_per_mm,
            pressure_offset_mm=pressure_offset_mm,
            pressure_advance_feed=pressure_advance_feed,
            pressure_retract_feed=pressure_retract_feed,
            preflow_dwell_ms=preflow_dwell_ms,
            node_dwell_ms=node_dwell_ms,
            edge_samples=edge_samples,
            emit_extrusion=emit_extrusion,
            warn_log=bbox_warnings,
        )
        if emit_extrusion:
            g.u_material_abs = float(prime_mm)

        # Safe startup
        msx, msy, msz, msb, msc = [float(v) for v in machine_start_pose]
        f.write("; safe startup approach (machine stage coordinates)\n")
        g.move_stage_xyzbc(
            np.array([msx, msy, safe_approach_z], dtype=float),
            b=msb, c=msc, feed=travel_feed,
            comment="startup: move to safe Z at machine-start XY"
        )
        g.move_stage_xyzbc(
            np.array([msx, msy, msz], dtype=float),
            b=msb, c=msc, feed=travel_feed,
            comment="startup: dive to machine-start Z"
        )

        if abs(g.cur_c - c_fixed) > 1e-9:
            g.move_c_only(c_fixed, comment="preposition C to fixed C using C-only feed")

        # ---------------- Prepended line stack ----------------
        total_print_passes = 0
        f.write("; ==================== prepended X-line stack in XZ plane ====================\n")

        prep_line_segments = make_prep_line_segments(
            start_x=plx,
            start_y=ply,
            start_z=plz,
            line_length=float(prep_line_length),
            line_count=int(prep_line_count),
            z_step=float(prep_line_z_step),
            label_prefix="prep_line",
        )

        for i, seg in enumerate(prep_line_segments, start=1):
            if g.cur_stage_xyz is None:
                g.move_to_tip(seg.p0_tip, feed=travel_feed, comment=f"move to prep line {i} start")
            else:
                g.safe_travel_z_up_x_then_z_down_tip(
                    seg.p0_tip,
                    lift_dz_stage=DEFAULT_RETURN_LIFT_Z,
                    comment_prefix=f"prep line {i} move to start"
                )

            g.print_tip_segment(seg, extrusion_multiplier=1.0)
            total_print_passes += 1

        # ---------------- Existing pattern ----------------
        pattern_start_tip_vec = p_xyz(sx, sy, sz)
        g.safe_travel_z_up_x_then_z_down_tip(
            pattern_start_tip_vec,
            lift_dz_stage=DEFAULT_RETURN_LIFT_Z,
            comment_prefix="move from prep line stack to pattern start"
        )

        y_cursor = sy

        for rep_idx, ex_mult in enumerate(DEFAULT_EXTRUSION_MULTIPLIERS, start=1):
            f.write(f"; ==================== repetition {rep_idx}/{len(DEFAULT_EXTRUSION_MULTIPLIERS)} ====================\n")
            f.write(f"; extrusion multiplier = {float(ex_mult):.3f}\n")

            # ---- Step 1 ----
            y_step1 = y_cursor
            step1_start_tip = p_xyz(sx, y_step1, sz)

            g.safe_travel_z_up_x_then_z_down_tip(
                step1_start_tip, lift_dz_stage=DEFAULT_RETURN_LIFT_Z,
                comment_prefix=f"rep{rep_idx} step1 move to origin"
            )

            step1_segments = make_step1_segments(
                start_tip=step1_start_tip,
                y_plane=y_step1,
                line_len=DEFAULT_LINE_LENGTH,
                mid_x=DEFAULT_MID_X,
                angle_step=DEFAULT_ANGLE_STEP,
                angle_max=DEFAULT_ANGLE_MAX_1,
                vertical_len=DEFAULT_VERTICAL_LINE_LEN,
            )

            horizontal = step1_segments[0]
            roof_parts = step1_segments[1:-1]
            vertical = step1_segments[-1]

            g.print_tip_segment(horizontal, extrusion_multiplier=ex_mult)
            total_print_passes += 1
            g.safe_travel_z_up_x_then_z_down_tip(
                step1_start_tip, lift_dz_stage=DEFAULT_RETURN_LIFT_Z,
                comment_prefix=f"rep{rep_idx} step1 return after horizontal"
            )

            if len(roof_parts) % 2 != 0:
                raise RuntimeError("Unexpected roof segment count (must be paired).")
            for k in range(0, len(roof_parts), 2):
                seg1 = roof_parts[k]
                seg2 = roof_parts[k + 1]
                g.print_tip_segment(seg1, extrusion_multiplier=ex_mult)
                total_print_passes += 1
                g.print_tip_segment(seg2, extrusion_multiplier=ex_mult)
                total_print_passes += 1
                g.safe_travel_z_up_x_then_z_down_tip(
                    step1_start_tip, lift_dz_stage=DEFAULT_RETURN_LIFT_Z,
                    comment_prefix=f"rep{rep_idx} step1 return after roof pair {k//2 + 1}"
                )

            g.print_tip_segment(vertical, extrusion_multiplier=ex_mult)
            total_print_passes += 1

            # ---- Step 2 ----
            y_step2 = y_step1 + DEFAULT_PLANE_Y_STEP
            step2_segments, step2_origin_tip, step2_vertical_top_tip, endpoints_famA = make_step2_segments(
                start_step1_tip=step1_start_tip,
                y_plane=y_step2,
                origin_x_rel=DEFAULT_STEP2_ORIGIN_X_REL,
                z_target_rel=DEFAULT_STEP2_Z_TARGET_REL,
                straight_extend_z_rel=DEFAULT_STEP2_STRAIGHT_EXTEND_Z_REL,
                top_horizontal_len=DEFAULT_TOP_HORIZONTAL_LEN,
            )

            g.safe_travel_x_first_then_down_z_tip(
                step2_origin_tip,
                comment_prefix=f"rep{rep_idx} step2 move to origin (x first then z)"
            )

            famA = [s for s in step2_segments if s.label.startswith("step2_famA_")]
            famB = [s for s in step2_segments if s.label.startswith("step2_famB_")]
            topH = [s for s in step2_segments if s.label == "step2_top_horizontal_10mm"]

            for i, seg in enumerate(famA):
                if g.cur_stage_xyz is None:
                    raise RuntimeError("Unexpected missing stage state.")
                if np.linalg.norm(g.tip_to_stage(seg.p0_tip) - g.cur_stage_xyz) > 1e-6:
                    g.safe_travel_x_first_then_down_z_tip(
                        seg.p0_tip,
                        comment_prefix=f"rep{rep_idx} step2 famA line {i+1} travel to start"
                    )
                g.print_tip_segment(seg, extrusion_multiplier=ex_mult)
                total_print_passes += 1

                if i < len(famA) - 1:
                    g.safe_travel_x_first_then_down_z_tip(
                        step2_origin_tip,
                        comment_prefix=f"rep{rep_idx} step2 famA return to origin"
                    )

            for i, seg in enumerate(famB):
                g.safe_travel_x_first_then_down_z_tip(
                    seg.p0_tip,
                    comment_prefix=f"rep{rep_idx} step2 famB line {i+1} move to start (end of first-half line)"
                )
                g.print_tip_segment(seg, extrusion_multiplier=ex_mult)
                total_print_passes += 1

            if topH:
                g.safe_travel_x_first_then_down_z_tip(
                    topH[0].p0_tip,
                    comment_prefix=f"rep{rep_idx} step2 move to top point"
                )
                g.print_tip_segment(topH[0], extrusion_multiplier=ex_mult)
                total_print_passes += 1

            y_cursor = y_step2 + DEFAULT_PLANE_Y_STEP

        if emit_extrusion and g.pressure_charged:
            f.write("; final pressure release at end of print\n")
            g.pressure_charged = False
            f.write(f"G1 {cal.u_axis}{g.u_cmd_actual():.3f} F{float(pressure_retract_feed):.0f}\n")

        mex, mey, mez, meb, mec = [float(v) for v in machine_end_pose]
        f.write("; safe end move (machine stage coordinates)\n")
        g.move_stage_xyzbc(
            np.array([g.cur_stage_xyz[0], g.cur_stage_xyz[1], safe_approach_z], dtype=float),
            b=meb, c=mec, feed=travel_feed,
            comment="end: raise to safe Z"
        )
        g.move_stage_xyzbc(
            np.array([mex, mey, safe_approach_z], dtype=float),
            b=meb, c=mec, feed=travel_feed,
            comment="end: move XY at safe Z"
        )
        g.move_stage_xyzbc(
            np.array([mex, mey, mez], dtype=float),
            b=meb, c=mec, feed=travel_feed,
            comment="end: dive to machine end Z"
        )

        if bbox_warnings:
            f.write("; virtual bbox clamp warnings:\n")
            for msg in bbox_warnings:
                f.write(f"; {msg}\n")
        f.write(f"; total print passes = {total_print_passes}\n")
        f.write(f"; bbox warning count = {len(bbox_warnings)}\n")
        f.write("; --- end ---\n")

    print(f"Wrote {out_path}")
    print("Mode: calibrated tip-position planning, fixed B/C during pattern")
    print(f"Axes mapping: X={cal.x_axis}, Y={cal.y_axis}, Z={cal.z_axis}, B={cal.b_axis}, C={cal.c_axis}, U={cal.u_axis}")
    print(f"Calibration B range: [{cal.b_min:.6f}, {cal.b_max:.6f}]")
    print(f"Solved B for tip_angle=0 deg: {b_zero_angle:.6f}")
    print(f"tip_angle_model(B_fixed) = {tip_angle_at_b:.9f} deg")
    print(f"Fixed tip offset(B,C=0): [{tip_offset[0]:.6f}, {tip_offset[1]:.6f}, {tip_offset[2]:.6f}] mm")
    print(f"Prepended line start: [{plx:.3f}, {ply:.3f}, {plz:.3f}]")
    print(f"Prepended line length: {float(prep_line_length):.3f} mm, count={int(prep_line_count)}, z_step={float(prep_line_z_step):.3f} mm")
    print("Prepended lines alternate X direction on successive Z levels")
    print(f"Feeds: travel={travel_feed:.1f} mm/min, print={print_feed:.1f} mm/min, C-only={c_feed:.1f} deg/min")
    print("B/C behavior: B fixed at solved zero-angle value, C fixed at 0 deg")
    if bbox_warnings:
        print(f"Virtual bounding-box warnings: {len(bbox_warnings)} (values clamped)")
        for msg in bbox_warnings:
            print(msg)


# ---------------- Main ----------------

def main(args):
    cal = load_calibration(args.calibration)

    bbox = {
        "x_min": float(args.bbox_x_min), "x_max": float(args.bbox_x_max),
        "y_min": float(args.bbox_y_min), "y_max": float(args.bbox_y_max),
        "z_min": float(args.bbox_z_min), "z_max": float(args.bbox_z_max),
    }
    if bbox["x_min"] > bbox["x_max"]:
        bbox["x_min"], bbox["x_max"] = bbox["x_max"], bbox["x_min"]
    if bbox["y_min"] > bbox["y_max"]:
        bbox["y_min"], bbox["y_max"] = bbox["y_max"], bbox["y_min"]
    if bbox["z_min"] > bbox["z_max"]:
        bbox["z_min"], bbox["z_max"] = bbox["z_max"], bbox["z_min"]

    prep_line_start_tip = (
        float(args.prep_line_start_x),
        float(args.prep_line_start_y),
        float(args.prep_line_start_z),
    )

    pattern_start_tip = (float(args.start_x), float(args.start_y), float(args.start_z))
    machine_start_pose = (
        float(args.machine_start_x), float(args.machine_start_y), float(args.machine_start_z),
        float(args.machine_start_b), float(args.machine_start_c),
    )
    machine_end_pose = (
        float(args.machine_end_x), float(args.machine_end_y), float(args.machine_end_z),
        float(args.machine_end_b), float(args.machine_end_c),
    )

    write_pattern_gcode(
        out_path=str(args.out),
        cal=cal,
        prep_line_start_tip=prep_line_start_tip,
        prep_line_length=float(args.prep_line_length),
        prep_line_count=int(args.prep_line_count),
        prep_line_z_step=float(args.prep_line_z_step),
        pattern_start_tip=pattern_start_tip,
        machine_start_pose=machine_start_pose,
        machine_end_pose=machine_end_pose,
        safe_approach_z=float(args.safe_approach_z),
        travel_feed=float(args.travel_feed),
        print_feed=float(args.print_feed),
        c_feed=float(args.c_feed),
        edge_samples=int(args.edge_samples),
        extrusion_per_mm=float(args.extrusion_per_mm),
        prime_mm=float(args.prime_mm),
        pressure_offset_mm=float(args.pressure_offset_mm),
        pressure_advance_feed=float(args.pressure_advance_feed),
        pressure_retract_feed=float(args.pressure_retract_feed),
        preflow_dwell_ms=int(args.preflow_dwell_ms),
        node_dwell_ms=int(args.node_dwell_ms),
        bbox=bbox,
    )

    ex_math = extrusion_math_summary(
        syringe_mm_per_ml=float(args.syringe_mm_per_ml),
        tube_id_inch=float(args.tube_id_inch),
        print_feed_mm_min=float(args.print_feed),
        extrusion_per_mm_u=float(args.extrusion_per_mm),
    )
    print("\nExtrusion / fluid math (base print feed + base extrusion_per_mm):")
    print(f"  Syringe calibration: {float(args.syringe_mm_per_ml):.3f} mm U / mL")
    print(f"  Tube ID: {ex_math['tube_id_mm']:.3f} mm ({float(args.tube_id_inch):.5f} in)")
    print(f"  Tube area: {ex_math['tube_area_mm2']:.6f} mm^2")
    print(f"  Path speed: {ex_math['path_speed_mm_s']:.6f} mm/s")
    print(f"  U speed: {ex_math['u_speed_mm_s']:.6f} mm/s")
    print(f"  Volumetric flow: {ex_math['q_mm3_s']:.6f} mm^3/s = {ex_math['q_ul_s']:.6f} uL/s = {ex_math['q_ml_min']:.6f} mL/min")
    print(f"  Mean fluid velocity in tube: {ex_math['tube_velocity_mm_s']:.6f} mm/s = {ex_math['tube_velocity_m_s']:.6f} m/s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Generate calibrated G-code for a prepended X-line stack in the XZ plane plus the requested XZ-plane line test pattern "
            "using exact tip-position planning from calibration, with fixed B chosen from the active tip-angle calibration model = 0 deg "
            "and fixed C=0."
        )
    )
    ap.add_argument("--calibration", required=True, help="Path to calibration JSON.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code file.")

    # Prepended X-lines in TIP SPACE
    ap.add_argument("--prep-line-start-x", type=float, default=DEFAULT_PREP_LINE_START_X,
                    help="Prepended line stack first-line start X (tip-space).")
    ap.add_argument("--prep-line-start-y", type=float, default=DEFAULT_PREP_LINE_START_Y,
                    help="Prepended line stack constant Y (tip-space).")
    ap.add_argument("--prep-line-start-z", type=float, default=DEFAULT_PREP_LINE_START_Z,
                    help="Prepended line stack first-line Z (tip-space).")
    ap.add_argument("--prep-line-length", type=float, default=DEFAULT_PREP_LINE_LENGTH,
                    help="Prepended line length in X (mm).")
    ap.add_argument("--prep-line-count", type=int, default=DEFAULT_PREP_LINE_COUNT,
                    help="Number of prepended lines to print before the main pattern.")
    ap.add_argument("--prep-line-z-step", type=float, default=DEFAULT_PREP_LINE_Z_STEP,
                    help="Z offset between successive prepended lines in mm.")

    # Main pattern placement in TIP SPACE
    ap.add_argument("--start-x", type=float, default=DEFAULT_START_X, help="Pattern start X (tip-space).")
    ap.add_argument("--start-y", type=float, default=DEFAULT_START_Y, help="Pattern start Y (tip-space).")
    ap.add_argument("--start-z", type=float, default=DEFAULT_START_Z, help="Pattern start Z (tip-space).")

    # Machine startup / end poses
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

    ap.add_argument("--safe-approach-z", type=float, default=DEFAULT_SAFE_APPROACH_Z,
                    help="Safe Z used before XY startup/end positioning.")

    # Motion
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED,
                    help="Feedrate for non-print travel moves (mm/min).")
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED,
                    help="Coordinated feedrate for print moves (mm/min).")
    ap.add_argument("--c-feed", type=float, default=DEFAULT_C_FEED,
                    help="Feedrate for C-only moves (deg/min).")
    ap.add_argument("--edge-samples", type=int, default=DEFAULT_EDGE_SAMPLES,
                    help="Interpolation segments per printed line.")

    # Extrusion + pressure math
    ap.add_argument("--extrusion-per-mm", type=float, default=DEFAULT_EXTRUSION_PER_MM,
                    help="U-axis displacement (mm) per mm of printed path. 0 disables extrusion.")
    ap.add_argument("--prime-mm", type=float, default=DEFAULT_PRIME_MM,
                    help="Optional U-axis material prime at start (absolute extrusion mode).")

    ap.add_argument("--pressure-offset-mm", type=float, default=DEFAULT_PRESSURE_OFFSET_MM,
                    help="U preload offset (mm) before each print pass, retracted after each pass.")
    ap.add_argument("--pressure-advance-feed", type=float, default=DEFAULT_PRESSURE_ADVANCE_FEED,
                    help="Feedrate for U-only pressure advance moves (mm/min).")
    ap.add_argument("--pressure-retract-feed", type=float, default=DEFAULT_PRESSURE_RETRACT_FEED,
                    help="Feedrate for U-only pressure retract moves (mm/min).")
    ap.add_argument("--preflow-dwell-ms", type=int, default=DEFAULT_PREFLOW_DWELL_MS,
                    help="Dwell after pressure advance before printing each pass (ms).")
    ap.add_argument("--node-dwell-ms", type=int, default=DEFAULT_NODE_DWELL_MS,
                    help="Dwell at end of each pass before pressure retract (ms).")

    # Virtual stage-space bounding box
    ap.add_argument("--bbox-x-min", type=float, default=DEFAULT_BBOX_X_MIN)
    ap.add_argument("--bbox-x-max", type=float, default=DEFAULT_BBOX_X_MAX)
    ap.add_argument("--bbox-y-min", type=float, default=DEFAULT_BBOX_Y_MIN)
    ap.add_argument("--bbox-y-max", type=float, default=DEFAULT_BBOX_Y_MAX)
    ap.add_argument("--bbox-z-min", type=float, default=DEFAULT_BBOX_Z_MIN)
    ap.add_argument("--bbox-z-max", type=float, default=DEFAULT_BBOX_Z_MAX)

    # Diagnostics-only extrusion helpers
    ap.add_argument("--syringe-mm-per-ml", type=float, default=6.0,
                    help="Syringe U-axis calibration (mm of U travel per mL displaced).")
    ap.add_argument("--tube-id-inch", type=float, default=0.02,
                    help="Tube inner diameter in inches (for flow velocity diagnostics).")

    args = ap.parse_args()
    main(args)
