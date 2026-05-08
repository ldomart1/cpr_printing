#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code for a jumping leg that traces a circular XY
path while hopping on a ground plane.

The script follows the same calibration-based tip planning idea as the provided
reference script:

    stage_xyz = desired_tip_xyz - offset_tip(B, C)

Key motion behavior
-------------------
- The tip follows a circle in XY.
- The tip remains on the ground plane until lift-off.
- The tip does not advance in XY while on the ground; XY only advances during the airborne part of each hop.
- Each hop uses a ballistic Z arc above the ground plane.
- B is used like a spring-leg posture command:
    compressed -> launch -> extended in air -> precompressed -> landing over-compression -> recovery
- C stays tangent to the circle.
- C is kept within [-360, 360].
- At the top point of the circle, when the next loop would exceed +360, the
  script inserts a fast 720-degree C reset during the hop so the commanded C
  value cycles back to -360 while preserving physical orientation.

Important note about the user's example
---------------------------------------
For a circle centered at (100, 70) with radius 40, the cardinal right point is
(140, 70), not (140, 100). This script uses the mathematically consistent circle
based on the stated center and radius:

- top    : (100, 30) -> C =   0 deg
- right  : (140, 70) -> C =  90 deg
- bottom : (100,110) -> C = 180 deg
- left   : ( 60, 70) -> C = 270 / -90 deg

Clockwise parameterization is used so C increases naturally as:
0 -> 90 -> 180 -> 270 -> 360.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------- defaults ----------------
DEFAULT_OUT = "gcode_generation/jumping_leg_circle.gcode"

DEFAULT_CENTER_X = 100.0
DEFAULT_CENTER_Y = 70.0
DEFAULT_RADIUS = 40.0
DEFAULT_GROUND_Z = -177.0

DEFAULT_CIRCLE_TIME_S = 5.0
DEFAULT_NUM_CIRCLES = 3
DEFAULT_HOPS_PER_LOOP = 8
DEFAULT_HOP_HEIGHT = 18.0
DEFAULT_SAMPLES_PER_HOP = 60

DEFAULT_MACHINE_START_X = 90.0
DEFAULT_MACHINE_START_Y = 60.0
DEFAULT_MACHINE_START_Z = -50.0
DEFAULT_MACHINE_START_B = 0.0
DEFAULT_MACHINE_START_C = 0.0
DEFAULT_MACHINE_END_X = 100.0
DEFAULT_MACHINE_END_Y = 60.0
DEFAULT_MACHINE_END_Z = -50.0
DEFAULT_MACHINE_END_B = 0.0
DEFAULT_MACHINE_END_C = 0.0

DEFAULT_TRAVEL_FEED = 2000.0   # mm/min
DEFAULT_PRINT_FEED_MIN = 120.0
DEFAULT_PRINT_FEED_MAX = 2400.0
DEFAULT_C_SPIN_FEED = 20000.0  # deg/min (pure C move)
DEFAULT_BC_SOLVE_SAMPLES = 5001

DEFAULT_STANCE_HOLD_FRACTION = 0.03
DEFAULT_RELEASE_FRACTION = 0.07
DEFAULT_LANDING_FRACTION = 0.09
DEFAULT_AIR_EXTEND_SPLIT = 0.35
DEFAULT_SPIN_PHASE = 0.34  # happens during the first hop of an odd loop, just after lift-off

DEFAULT_COMPRESSED_TIP_ANGLE_DEG = 70.0
DEFAULT_LAUNCH_TIP_ANGLE_DEG = 42.0
DEFAULT_EXTENDED_TIP_ANGLE_DEG = 22.0
DEFAULT_PRECOMPRESS_TIP_ANGLE_DEG = 56.0
DEFAULT_LANDING_TIP_ANGLE_DEG = 82.0

DEFAULT_BBOX_X_MIN = -1.0e9
DEFAULT_BBOX_X_MAX = +1.0e9
DEFAULT_BBOX_Y_MIN = -1.0e9
DEFAULT_BBOX_Y_MAX = +1.0e9
DEFAULT_BBOX_Z_MIN = -1.0e9
DEFAULT_BBOX_Z_MAX = +1.0e9


# ---------------- calibration data model ----------------

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
    active_phase: str = "pull"


@dataclass
class TrajectorySample:
    time_s: float
    tip_xyz: np.ndarray
    target_tip_angle_deg: float
    b_cmd: float
    c_cmd: float
    phase: str
    feed_mm_min: float
    emit_spin_reset_before: bool = False
    spin_reset_from_c: Optional[float] = None
    spin_reset_to_c: Optional[float] = None


# ---------------- calibration helpers ----------------

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

    r_model = _select_named_model(active_phase_models, "r", selected_fit_model)
    z_model = _select_named_model(active_phase_models, "z", selected_fit_model)
    y_off_model = _select_named_model(active_phase_models, "offplane_y", selected_fit_model)
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

    x_axis = str(duet_map.get("horizontal_axis") or motor_setup.get("horizontal_axis") or "X")
    y_axis = str(duet_map.get("depth_axis") or motor_setup.get("depth_axis") or "Y")
    z_axis = str(duet_map.get("vertical_axis") or motor_setup.get("vertical_axis") or "Z")
    b_axis = str(duet_map.get("pull_axis") or motor_setup.get("b_motor_axis") or "B")
    c_axis = str(duet_map.get("rotation_axis") or motor_setup.get("rotation_axis") or "C")
    u_axis = str(duet_map.get("extruder_axis") or "U")
    c_180 = float(motor_setup.get("rotation_axis_180_deg", 180.0))

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
        active_phase=active_phase,
        b_min=b_min,
        b_max=b_max,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        b_axis=b_axis,
        c_axis=c_axis,
        u_axis=u_axis,
        c_180_deg=c_180,
    )


def eval_r(cal: Calibration, b: Any) -> np.ndarray:
    if cal.r_model is not None:
        return eval_model_spec(cal.r_model, b)
    return poly_eval(cal.pr, b)


def eval_z(cal: Calibration, b: Any) -> np.ndarray:
    if cal.z_model is not None:
        return eval_model_spec(cal.z_model, b)
    return poly_eval(cal.pz, b)


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


def eval_offplane_y(cal: Calibration, b: Any) -> np.ndarray:
    if cal.y_off_model is not None:
        if str(cal.y_off_model.get("model_type", "")).lower() == "pchip":
            return eval_pchip_with_linear_extrap(cal.y_off_model, cal.y_off_extrap_model, b)
        return eval_model_spec(cal.y_off_model, b, default_if_none=0.0)
    return poly_eval(cal.py_off, b, default_if_none=0.0)


def eval_tip_angle_deg(cal: Calibration, b: Any) -> np.ndarray:
    if cal.tip_angle_model is not None:
        return eval_model_spec(cal.tip_angle_model, b)
    if cal.pa is None:
        raise ValueError("Calibration is missing tip_angle_coeffs.")
    return poly_eval(cal.pa, b)


def predict_tip_offset_xyz(cal: Calibration, b: float, c_deg: float) -> np.ndarray:
    """
    Physical tip offset from stage origin:
      local transverse [r(B), y_off(B)] rotated by C into XY, z=z(B)

    This matches the convention used in the reference script.
    """
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
    """
    Solve the calibrated B command such that tip_angle(B) ~= target_angle_deg.

    This follows the same solve pattern as the reference script.
    """
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


# ---------------- motion helpers ----------------

def smoothstep01(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def lerp(a: float, b: float, t: float) -> float:
    return float(a + (b - a) * t)


def clamp(v: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, v)))


def unwrap_angle_deg_near(target_deg: float, reference_deg: float) -> float:
    target = float(target_deg)
    ref = float(reference_deg)
    while target - ref > 180.0:
        target -= 360.0
    while target - ref < -180.0:
        target += 360.0
    return float(target)


def circle_xy(center_x: float, center_y: float, radius: float, phi_deg: float) -> Tuple[float, float]:
    """
    Clockwise parameterization chosen to match:
        top    -> C=0
        right  -> C=90
        bottom -> C=180

    x = cx + r*sin(phi)
    y = cy - r*cos(phi)
    """
    phi = math.radians(float(phi_deg))
    x = float(center_x + radius * math.sin(phi))
    y = float(center_y - radius * math.cos(phi))
    return x, y


def circle_tangent_c_deg(phi_deg: float) -> float:
    """For the parameterization above, tangent azimuth equals phi_deg."""
    return float(phi_deg)


def loop_c_offset(loop_index: int) -> float:
    """
    Keep C within [-360, 360].

    Loop 0: base range   [   0,  360]
    Loop 1: wrapped to   [-360,    0]
    Loop 2: wrapped to   [   0,  360]
    Loop 3: wrapped to   [-360,    0]
    ...
    """
    return float(-720.0 * ((int(loop_index) + 1) // 2))


def hop_state(
    u: float,
    hop_height: float,
    stance_hold_fraction: float,
    release_fraction: float,
    landing_fraction: float,
    air_extend_split: float,
    compressed_tip_angle_deg: float,
    launch_tip_angle_deg: float,
    extended_tip_angle_deg: float,
    precompress_tip_angle_deg: float,
    landing_tip_angle_deg: float,
    is_first_hop: bool,
) -> Tuple[float, float, str]:
    """
    Returns:
        z_offset_above_ground_mm,
        target_tip_angle_deg,
        phase_name

    The sequence is:
      1) on-ground recovery / loaded stance without XY motion
      2) release on ground until lift-off
      3) airborne extension while rising while XY advances along the circle
      4) airborne precompression while descending while XY keeps advancing
      5) landing over-compression on ground at fixed XY
    """
    u = clamp(u, 0.0, 1.0)
    hold_end = stance_hold_fraction
    liftoff = stance_hold_fraction + release_fraction
    landing_start = 1.0 - landing_fraction

    if liftoff >= landing_start:
        raise ValueError("Invalid hop fractions: release leaves no airborne interval.")

    if u <= hold_end:
        if is_first_hop:
            return 0.0, float(compressed_tip_angle_deg), "stance_hold"
        s = smoothstep01(u / max(hold_end, 1e-9))
        angle = lerp(landing_tip_angle_deg, compressed_tip_angle_deg, s)
        return 0.0, angle, "stance_recover"

    if u <= liftoff:
        s = smoothstep01((u - hold_end) / max(liftoff - hold_end, 1e-9))
        angle = lerp(compressed_tip_angle_deg, launch_tip_angle_deg, s)
        return 0.0, angle, "ground_release"

    if u <= landing_start:
        tau = (u - liftoff) / max(landing_start - liftoff, 1e-9)
        z = 4.0 * float(hop_height) * tau * (1.0 - tau)
        split = clamp(air_extend_split, 0.05, 0.95)
        if tau <= split:
            s = smoothstep01(tau / split)
            angle = lerp(launch_tip_angle_deg, extended_tip_angle_deg, s)
            return z, angle, "air_extend"
        s = smoothstep01((tau - split) / max(1.0 - split, 1e-9))
        angle = lerp(extended_tip_angle_deg, precompress_tip_angle_deg, s)
        return z, angle, "air_precompress"

    s = smoothstep01((u - landing_start) / max(1.0 - landing_start, 1e-9))
    angle = lerp(precompress_tip_angle_deg, landing_tip_angle_deg, s)
    return 0.0, angle, "landing_compress"


def resolve_b_command(
    target_tip_angle_deg: float,
    cal: Optional[Calibration],
    bc_solve_samples: int,
    assume_b_equals_tip_angle_without_calibration: bool,
) -> float:
    if cal is not None:
        return float(solve_b_for_target_tip_angle(cal, target_tip_angle_deg, search_samples=bc_solve_samples))
    if assume_b_equals_tip_angle_without_calibration:
        return float(target_tip_angle_deg)
    raise ValueError(
        "Calibration is required to convert tip-angle targets into B commands. "
        "If you deliberately want raw B=angle behavior, pass --assume-b-equals-tip-angle-without-calibration."
    )


def compute_stage_xyz(
    tip_xyz: np.ndarray,
    b_cmd: float,
    c_cmd: float,
    write_mode: str,
    cal: Optional[Calibration],
) -> np.ndarray:
    write_mode = str(write_mode).strip().lower()
    if write_mode == "calibrated":
        if cal is None:
            raise ValueError("--calibration is required for --write-mode calibrated")
        return stage_xyz_for_tip(cal, tip_xyz, b_cmd, c_cmd)
    if write_mode == "cartesian":
        return np.asarray(tip_xyz, dtype=float).copy()
    raise ValueError(f"Unsupported write mode: {write_mode}")


def generate_hop_trajectory(
    *,
    center_x: float,
    center_y: float,
    radius: float,
    ground_z: float,
    circle_time_s: float,
    num_circles: int,
    hops_per_loop: int,
    hop_height: float,
    samples_per_hop: int,
    stance_hold_fraction: float,
    release_fraction: float,
    landing_fraction: float,
    air_extend_split: float,
    compressed_tip_angle_deg: float,
    launch_tip_angle_deg: float,
    extended_tip_angle_deg: float,
    precompress_tip_angle_deg: float,
    landing_tip_angle_deg: float,
    cal: Optional[Calibration],
    write_mode: str,
    bc_solve_samples: int,
    assume_b_equals_tip_angle_without_calibration: bool,
    print_feed_min: float,
    print_feed_max: float,
    spin_phase: float,
) -> List[TrajectorySample]:
    if num_circles < 1:
        raise ValueError("--num-circles must be >= 1")
    if hops_per_loop < 1:
        raise ValueError("--hops-per-loop must be >= 1")
    if samples_per_hop < 4:
        raise ValueError("--samples-per-hop must be >= 4")
    if circle_time_s <= 0.0:
        raise ValueError("--circle-time-s must be > 0")
    if radius <= 0.0:
        raise ValueError("--radius must be > 0")

    total_hops = int(num_circles * hops_per_loop)
    hop_time_s = float(circle_time_s) / float(hops_per_loop)
    dt = hop_time_s / float(samples_per_hop)
    hop_span_deg = 360.0 / float(hops_per_loop)

    samples: List[TrajectorySample] = []
    prev_tip_xyz: Optional[np.ndarray] = None

    for hop_index in range(total_hops):
        loop_index = hop_index // hops_per_loop
        hop_in_loop = hop_index % hops_per_loop
        loop_offset_now = loop_c_offset(loop_index)
        loop_offset_prev = loop_c_offset(loop_index - 1) if loop_index > 0 else loop_offset_now

        needs_spin_reset = (loop_index > 0 and hop_in_loop == 0 and abs(loop_offset_now - loop_offset_prev) > 1e-9)
        spin_inserted = False

        phi_start_base = 360.0 * float(loop_index) + hop_span_deg * float(hop_in_loop)

        for k in range(samples_per_hop + 1):
            if hop_index > 0 and k == 0:
                continue  # avoid duplicate point at hop boundary

            u = float(k) / float(samples_per_hop)
            time_s = hop_index * hop_time_s + u * hop_time_s
            phi_contact_start = phi_start_base
            phi_contact_end = phi_start_base + hop_span_deg

            z_offset, target_tip_angle_deg, phase_name = hop_state(
                u=u,
                hop_height=hop_height,
                stance_hold_fraction=stance_hold_fraction,
                release_fraction=release_fraction,
                landing_fraction=landing_fraction,
                air_extend_split=air_extend_split,
                compressed_tip_angle_deg=compressed_tip_angle_deg,
                launch_tip_angle_deg=launch_tip_angle_deg,
                extended_tip_angle_deg=extended_tip_angle_deg,
                precompress_tip_angle_deg=precompress_tip_angle_deg,
                landing_tip_angle_deg=landing_tip_angle_deg,
                is_first_hop=(hop_index == 0),
            )

            liftoff = stance_hold_fraction + release_fraction
            landing_start = 1.0 - landing_fraction
            if u <= liftoff:
                phi_base = phi_contact_start
            elif u >= landing_start:
                phi_base = phi_contact_end
            else:
                air_tau = (u - liftoff) / max(landing_start - liftoff, 1e-9)
                air_tau = smoothstep01(air_tau)
                phi_base = lerp(phi_contact_start, phi_contact_end, air_tau)

            x, y = circle_xy(center_x, center_y, radius, phi_base)
            z = float(ground_z + z_offset)
            tip_xyz = np.array([x, y, z], dtype=float)

            b_cmd = resolve_b_command(
                target_tip_angle_deg=target_tip_angle_deg,
                cal=cal,
                bc_solve_samples=bc_solve_samples,
                assume_b_equals_tip_angle_without_calibration=assume_b_equals_tip_angle_without_calibration,
            )

            base_c = circle_tangent_c_deg(phi_base)
            wrapped_c = base_c + loop_offset_now

            emit_spin_reset_before = False
            spin_from_c = None
            spin_to_c = None

            if needs_spin_reset and (not spin_inserted) and u >= spin_phase and z_offset > 1e-6:
                emit_spin_reset_before = True
                spin_from_c = 360.0
                spin_to_c = -360.0
                spin_inserted = True

            if needs_spin_reset and not spin_inserted:
                # Hold at the equivalent top orientation while still in the early part
                # of the hop so C never exceeds +360 before the reset happens.
                c_cmd = 360.0
            else:
                c_cmd = wrapped_c

            if c_cmd < -360.000001 or c_cmd > 360.000001:
                raise ValueError(
                    f"Internal C wrapping failed at hop={hop_index}, u={u:.4f}, C={c_cmd:.6f}."
                )

            if prev_tip_xyz is None:
                feed_mm_min = float(print_feed_min)
            else:
                ds = float(np.linalg.norm(tip_xyz - prev_tip_xyz))
                feed_mm_min = clamp(60.0 * ds / max(dt, 1e-9), print_feed_min, print_feed_max)

            samples.append(
                TrajectorySample(
                    time_s=time_s,
                    tip_xyz=tip_xyz,
                    target_tip_angle_deg=float(target_tip_angle_deg),
                    b_cmd=float(b_cmd),
                    c_cmd=float(c_cmd),
                    phase=phase_name,
                    feed_mm_min=float(feed_mm_min),
                    emit_spin_reset_before=emit_spin_reset_before,
                    spin_reset_from_c=spin_from_c,
                    spin_reset_to_c=spin_to_c,
                )
            )
            prev_tip_xyz = tip_xyz

        if needs_spin_reset and not spin_inserted:
            raise ValueError(
                "Spin reset was requested for a wrapped loop but was never inserted. "
                "Try increasing --samples-per-hop or adjusting --spin-phase."
            )

    return samples


# ---------------- g-code writer ----------------

def _fmt_axes_move(axes_vals: Sequence[Tuple[str, float]]) -> str:
    return " ".join(f"{ax}{val:.3f}" for ax, val in axes_vals)


class GCodeWriter:
    def __init__(
        self,
        fh,
        cal: Optional[Calibration],
        bbox: Dict[str, float],
        write_mode: str,
        travel_feed: float,
        c_spin_feed: float,
    ) -> None:
        self.f = fh
        self.cal = cal
        self.bbox = dict(bbox)
        self.write_mode = str(write_mode).strip().lower()
        self.travel_feed = float(travel_feed)
        self.c_spin_feed = float(c_spin_feed)

        if cal is None:
            self.x_axis = "X"
            self.y_axis = "Y"
            self.z_axis = "Z"
            self.b_axis = "B"
            self.c_axis = "C"
        else:
            self.x_axis = cal.x_axis
            self.y_axis = cal.y_axis
            self.z_axis = cal.z_axis
            self.b_axis = cal.b_axis
            self.c_axis = cal.c_axis

        self.cur_stage_xyz: Optional[np.ndarray] = None
        self.cur_b: Optional[float] = None
        self.cur_c: Optional[float] = None
        self.warnings: List[str] = []

        self.x_cmds: List[float] = []
        self.y_cmds: List[float] = []
        self.z_cmds: List[float] = []
        self.b_cmds: List[float] = []
        self.c_cmds: List[float] = []

    def write_header(self, cli_args: argparse.Namespace) -> None:
        self.f.write("; generated by jumping_leg_circle_gcode.py\n")
        self.f.write("; calibration-based tip planning follows the same stage_xyz_for_tip convention as the reference script\n")
        self.f.write("; reference conventions: load_calibration / predict_tip_offset_xyz / stage_xyz_for_tip / solve_b_for_target_tip_angle\n")
        self.f.write("; reference source: uploaded script excerpt\n")
        self.f.write("G90\n")
        self.f.write("M400\n")
        self.f.write("; ---------------- requested motion ----------------\n")
        self.f.write(f"; center_x = {cli_args.center_x:.6f}\n")
        self.f.write(f"; center_y = {cli_args.center_y:.6f}\n")
        self.f.write(f"; radius = {cli_args.radius:.6f}\n")
        self.f.write(f"; ground_z = {cli_args.ground_z:.6f}\n")
        self.f.write(f"; circle_time_s = {cli_args.circle_time_s:.6f}\n")
        self.f.write(f"; num_circles = {cli_args.num_circles}\n")
        self.f.write(f"; hops_per_loop = {cli_args.hops_per_loop}\n")
        self.f.write(f"; hop_height = {cli_args.hop_height:.6f}\n")
        self.f.write(f"; samples_per_hop = {cli_args.samples_per_hop}\n")
        self.f.write(f"; landing_tip_angle_deg = {cli_args.landing_tip_angle_deg:.6f}\n")
        self.f.write(f"; write_mode = {self.write_mode}\n")
        if self.cal is not None:
            self.f.write(f"; calibration active_phase = {self.cal.active_phase}\n")
            self.f.write(f"; B command range = [{self.cal.b_min:.6f}, {self.cal.b_max:.6f}]\n")
            self.f.write(f"; axis mapping: {self.x_axis} {self.y_axis} {self.z_axis} {self.b_axis} {self.c_axis}\n")
        self.f.write("; -----------------------------------------------\n")

    def clamp_stage(self, p_stage: np.ndarray, context: str) -> np.ndarray:
        x = float(np.clip(p_stage[0], self.bbox["x_min"], self.bbox["x_max"]))
        y = float(np.clip(p_stage[1], self.bbox["y_min"], self.bbox["y_max"]))
        z = float(np.clip(p_stage[2], self.bbox["z_min"], self.bbox["z_max"]))
        if abs(x - float(p_stage[0])) > 1e-12 or abs(y - float(p_stage[1])) > 1e-12 or abs(z - float(p_stage[2])) > 1e-12:
            self.warnings.append(f"WARNING: {context} stage point clamped to bbox.")
        return np.array([x, y, z], dtype=float)

    def write_move(self, stage_xyz: np.ndarray, b_cmd: float, c_cmd: float, feed: float, comment: Optional[str] = None) -> None:
        p = self.clamp_stage(np.asarray(stage_xyz, dtype=float), "write_move")
        axes_vals = [
            (self.x_axis, float(p[0])),
            (self.y_axis, float(p[1])),
            (self.z_axis, float(p[2])),
            (self.b_axis, float(b_cmd)),
            (self.c_axis, float(c_cmd)),
        ]
        line = f"G1 {_fmt_axes_move(axes_vals)} F{float(feed):.3f}"
        if comment:
            line += f" ; {comment}"
        self.f.write(line + "\n")

        self.cur_stage_xyz = p.copy()
        self.cur_b = float(b_cmd)
        self.cur_c = float(c_cmd)

        self.x_cmds.append(float(p[0]))
        self.y_cmds.append(float(p[1]))
        self.z_cmds.append(float(p[2]))
        self.b_cmds.append(float(b_cmd))
        self.c_cmds.append(float(c_cmd))

    def write_spin_reset(self, c_from: float, c_to: float, comment: Optional[str] = None) -> None:
        if self.cur_stage_xyz is None or self.cur_b is None:
            raise RuntimeError("Cannot emit a spin reset before an initial pose exists.")

        if abs(float(self.cur_c) - float(c_from)) > 1e-6:
            self.write_move(
                self.cur_stage_xyz,
                self.cur_b,
                c_from,
                self.travel_feed,
                comment="pre-spin orient to exact equivalent C reset pose",
            )

        self.write_move(
            self.cur_stage_xyz,
            self.cur_b,
            c_to,
            self.c_spin_feed,
            comment=comment or "fast 720-degree C reset during hop",
        )

    def command_ranges(self) -> Dict[str, Tuple[float, float]]:
        def mm(values: List[float]) -> Tuple[float, float]:
            if not values:
                return (0.0, 0.0)
            return (float(min(values)), float(max(values)))

        return {
            "x": mm(self.x_cmds),
            "y": mm(self.y_cmds),
            "z": mm(self.z_cmds),
            "b": mm(self.b_cmds),
            "c": mm(self.c_cmds),
        }


def write_jump_circle_gcode(
    out_path: str,
    samples: List[TrajectorySample],
    cal: Optional[Calibration],
    write_mode: str,
    travel_feed: float,
    c_spin_feed: float,
    machine_start_pose: Tuple[float, float, float, float, float],
    machine_end_pose: Tuple[float, float, float, float, float],
    bbox: Dict[str, float],
    cli_args: argparse.Namespace,
) -> Dict[str, Tuple[float, float]]:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        g = GCodeWriter(
            fh=f,
            cal=cal,
            bbox=bbox,
            write_mode=write_mode,
            travel_feed=travel_feed,
            c_spin_feed=c_spin_feed,
        )
        g.write_header(cli_args)

        msx, msy, msz, msb, msc = [float(v) for v in machine_start_pose]
        mex, mey, mez, meb, mec = [float(v) for v in machine_end_pose]

        g.write_move(
            np.array([msx, msy, msz], dtype=float),
            msb,
            msc,
            travel_feed,
            comment="startup move to machine start pose",
        )

        prev_stage_xyz: Optional[np.ndarray] = None
        for i, s in enumerate(samples):
            stage_xyz = compute_stage_xyz(
                tip_xyz=s.tip_xyz,
                b_cmd=s.b_cmd,
                c_cmd=s.c_cmd,
                write_mode=write_mode,
                cal=cal,
            )

            if s.emit_spin_reset_before:
                if prev_stage_xyz is None:
                    raise RuntimeError("Spin reset requested before any motion sample was emitted.")
                # First move to the current tip sample while holding the equivalent +360 pose.
                g.write_move(
                    stage_xyz,
                    s.b_cmd,
                    float(s.spin_reset_from_c),
                    s.feed_mm_min,
                    comment=f"t={s.time_s:.3f}s {s.phase}: move to spin-reset pose while airborne",
                )
                # Then do the pure C reset at constant XYZ/B.
                g.write_spin_reset(
                    c_from=float(s.spin_reset_from_c),
                    c_to=float(s.spin_reset_to_c),
                    comment="fast 720-degree C reset in jump to keep C within [-360, 360]",
                )
                # No second XYZ move needed here; the spin move already leaves us at the same XYZ/B,
                # and the next sample continues with the wrapped C frame.
                prev_stage_xyz = stage_xyz.copy()
                continue

            tip_comment = (
                f"t={s.time_s:.3f}s {s.phase}; "
                f"tip=({s.tip_xyz[0]:.3f},{s.tip_xyz[1]:.3f},{s.tip_xyz[2]:.3f}) ; "
                f"tip_angle_target={s.target_tip_angle_deg:.3f} ; C_tangent={s.c_cmd:.3f}"
            )
            g.write_move(stage_xyz, s.b_cmd, s.c_cmd, s.feed_mm_min, comment=tip_comment)
            prev_stage_xyz = stage_xyz.copy()

        g.write_move(
            np.array([mex, mey, mez], dtype=float),
            meb,
            mec,
            travel_feed,
            comment="shutdown move to machine end pose",
        )

        if g.warnings:
            f.write("; ==================== warnings ====================\n")
            for w in g.warnings:
                f.write(f"; {w}\n")

        ranges = g.command_ranges()
        f.write("; ==================== command ranges ====================\n")
        f.write(
            "; "
            f"{g.x_axis}=[{ranges['x'][0]:.6f}, {ranges['x'][1]:.6f}], "
            f"{g.y_axis}=[{ranges['y'][0]:.6f}, {ranges['y'][1]:.6f}], "
            f"{g.z_axis}=[{ranges['z'][0]:.6f}, {ranges['z'][1]:.6f}], "
            f"{g.b_axis}=[{ranges['b'][0]:.6f}, {ranges['b'][1]:.6f}], "
            f"{g.c_axis}=[{ranges['c'][0]:.6f}, {ranges['c'][1]:.6f}]\n"
        )
        return ranges


# ---------------- CLI ----------------

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Generate calibrated or Cartesian G-code for a jumping leg that traces a circular XY path "
            "with ballistic hops, tangent-following C, and spring-like B posture changes."
        )
    )
    ap.add_argument("--calibration", default=None, help="Path to the calibration JSON.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code file.")

    ap.add_argument("--write-mode", choices=["calibrated", "cartesian"], default="calibrated")
    ap.add_argument(
        "--assume-b-equals-tip-angle-without-calibration",
        action="store_true",
        help="If no calibration is supplied, use the target tip angle directly as the B command.",
    )

    ap.add_argument("--center-x", type=float, default=DEFAULT_CENTER_X)
    ap.add_argument("--center-y", type=float, default=DEFAULT_CENTER_Y)
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    ap.add_argument("--ground-z", type=float, default=DEFAULT_GROUND_Z)

    ap.add_argument("--circle-time-s", type=float, default=DEFAULT_CIRCLE_TIME_S, help="Time to complete one full circle.")
    ap.add_argument("--num-circles", type=int, default=DEFAULT_NUM_CIRCLES)
    ap.add_argument("--hops-per-loop", type=int, default=DEFAULT_HOPS_PER_LOOP)
    ap.add_argument("--hop-height", type=float, default=DEFAULT_HOP_HEIGHT)
    ap.add_argument("--samples-per-hop", type=int, default=DEFAULT_SAMPLES_PER_HOP)

    ap.add_argument("--compressed-tip-angle-deg", type=float, default=DEFAULT_COMPRESSED_TIP_ANGLE_DEG)
    ap.add_argument("--launch-tip-angle-deg", type=float, default=DEFAULT_LAUNCH_TIP_ANGLE_DEG)
    ap.add_argument("--extended-tip-angle-deg", type=float, default=DEFAULT_EXTENDED_TIP_ANGLE_DEG)
    ap.add_argument("--precompress-tip-angle-deg", type=float, default=DEFAULT_PRECOMPRESS_TIP_ANGLE_DEG)
    ap.add_argument("--landing-tip-angle-deg", type=float, default=DEFAULT_LANDING_TIP_ANGLE_DEG, help="Deeper on-ground compression reached right after touchdown before recovering for the next launch.")

    ap.add_argument("--stance-hold-fraction", type=float, default=DEFAULT_STANCE_HOLD_FRACTION)
    ap.add_argument("--release-fraction", type=float, default=DEFAULT_RELEASE_FRACTION)
    ap.add_argument("--landing-fraction", type=float, default=DEFAULT_LANDING_FRACTION)
    ap.add_argument("--air-extend-split", type=float, default=DEFAULT_AIR_EXTEND_SPLIT)
    ap.add_argument("--spin-phase", type=float, default=DEFAULT_SPIN_PHASE, help="Normalized hop phase for the 720-degree C reset on wrapped loops.")

    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED)
    ap.add_argument("--print-feed-min", type=float, default=DEFAULT_PRINT_FEED_MIN)
    ap.add_argument("--print-feed-max", type=float, default=DEFAULT_PRINT_FEED_MAX)
    ap.add_argument("--c-spin-feed", type=float, default=DEFAULT_C_SPIN_FEED)
    ap.add_argument("--bc-solve-samples", type=int, default=DEFAULT_BC_SOLVE_SAMPLES)

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
    ap.add_argument(
        "--use-explicit-machine-start-end",
        action="store_true",
        help="Use the explicit machine start/end poses instead of anchoring start/end to the first generated motion sample.",
    )

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

    cal: Optional[Calibration] = None
    if args.calibration is not None:
        cal = load_calibration(args.calibration)

    if args.write_mode == "calibrated" and cal is None:
        raise ValueError("--calibration is required for --write-mode calibrated")

    samples = generate_hop_trajectory(
        center_x=args.center_x,
        center_y=args.center_y,
        radius=args.radius,
        ground_z=args.ground_z,
        circle_time_s=args.circle_time_s,
        num_circles=args.num_circles,
        hops_per_loop=args.hops_per_loop,
        hop_height=args.hop_height,
        samples_per_hop=args.samples_per_hop,
        stance_hold_fraction=args.stance_hold_fraction,
        release_fraction=args.release_fraction,
        landing_fraction=args.landing_fraction,
        air_extend_split=args.air_extend_split,
        compressed_tip_angle_deg=args.compressed_tip_angle_deg,
        launch_tip_angle_deg=args.launch_tip_angle_deg,
        extended_tip_angle_deg=args.extended_tip_angle_deg,
        precompress_tip_angle_deg=args.precompress_tip_angle_deg,
        landing_tip_angle_deg=args.landing_tip_angle_deg,
        cal=cal,
        write_mode=args.write_mode,
        bc_solve_samples=args.bc_solve_samples,
        assume_b_equals_tip_angle_without_calibration=args.assume_b_equals_tip_angle_without_calibration,
        print_feed_min=args.print_feed_min,
        print_feed_max=args.print_feed_max,
        spin_phase=args.spin_phase,
    )

    first = samples[0]
    first_stage_xyz = compute_stage_xyz(first.tip_xyz, first.b_cmd, first.c_cmd, args.write_mode, cal)

    if args.use_explicit_machine_start_end:
        machine_start_pose = (
            float(args.machine_start_x),
            float(args.machine_start_y),
            float(args.machine_start_z),
            float(args.machine_start_b),
            float(args.machine_start_c),
        )
        machine_end_pose = (
            float(args.machine_end_x),
            float(args.machine_end_y),
            float(args.machine_end_z),
            float(args.machine_end_b),
            float(args.machine_end_c),
        )
    else:
        machine_start_pose = (
            float(first_stage_xyz[0]),
            float(first_stage_xyz[1]),
            float(first_stage_xyz[2]),
            float(first.b_cmd),
            float(first.c_cmd),
        )
        machine_end_pose = machine_start_pose

    bbox = {
        "x_min": float(args.bbox_x_min),
        "x_max": float(args.bbox_x_max),
        "y_min": float(args.bbox_y_min),
        "y_max": float(args.bbox_y_max),
        "z_min": float(args.bbox_z_min),
        "z_max": float(args.bbox_z_max),
    }

    ranges = write_jump_circle_gcode(
        out_path=args.out,
        samples=samples,
        cal=cal,
        write_mode=args.write_mode,
        travel_feed=args.travel_feed,
        c_spin_feed=args.c_spin_feed,
        machine_start_pose=machine_start_pose,
        machine_end_pose=machine_end_pose,
        bbox=bbox,
        cli_args=args,
    )

    print(f"Wrote {args.out}")
    print(
        "Command ranges: "
        f"X[{ranges['x'][0]:.3f}, {ranges['x'][1]:.3f}] "
        f"Y[{ranges['y'][0]:.3f}, {ranges['y'][1]:.3f}] "
        f"Z[{ranges['z'][0]:.3f}, {ranges['z'][1]:.3f}] "
        f"B[{ranges['b'][0]:.3f}, {ranges['b'][1]:.3f}] "
        f"C[{ranges['c'][0]:.3f}, {ranges['c'][1]:.3f}]"
    )


if __name__ == "__main__":
    main()
