#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone fixed-tip acquisition script for dual fixed-C orientations.

What it does:
- Loads the calibration JSON
- Tracks ONE fixed tip point
- Runs two acquisition blocks:
    1) C = 0 deg
    2) C = 180 deg
- For EACH C orientation, performs 6 smooth B oscillations while keeping the
  tip fixed in space by compensating with XYZ
- Connects to the Duet robot and the camera
- Executes the motion directly on the robot
- Captures images at selected samples and saves them to:
    <project folder>/raw_image_data_folder/

Key behavior changes vs the old cyclic C-sweep script:
- C is NOT swept continuously anymore
- The point is tracked only at C=0, then at C=180
- Each orientation gets its own smooth B-only oscillation block
- After the first large move onto the first tracked point, the script waits 4 s
  before starting the actual sweep
- Accuracy tricks included:
    * smooth sinusoidal B motion with zero velocity at block boundaries
    * exact fixed-tip XYZ compensation at every sample
    * per-segment feed scheduling
    * explicit M400 waits
    * fine re-approach to the first point of each block at a slower feed
    * conservative large-move handling between blocks

If requested tip angles fall outside calibration, the script uses the closest
tip angles that do exist in calibration.

Optional sweep-range override:
- Use --b-0-to-90-only to force the B/tip oscillation to stay within
  0 to 90 degrees.

Optional sign correction:
- Use --flip-rz-sign if your calibration file has r and z polynomial signs flipped.
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple, Set

import cv2
import numpy as np
from scipy.interpolate import PchipInterpolator

try:
    from duetwebapi import DuetWebAPI
except Exception:
    raise ImportError(
        "Missing duetwebapi. Install with:\n"
        "    pip install duetwebapi==1.1.0"
    )


# =========================
# Defaults
# =========================

DEFAULT_DUET_WEB_ADDRESS = "http://192.168.2.21"
DEFAULT_CAMERA_PORT = 0
DEFAULT_PROJECT_NAME = "Point_Tracking_Run"
DEFAULT_ALLOW_EXISTING = True
DEFAULT_ADD_DATE = True

DEFAULT_POINT_X = 100.0
DEFAULT_POINT_Y = 20.0
DEFAULT_POINT_Z = -155.0

DEFAULT_TRAVEL_FEED = 2000.0
DEFAULT_FINE_APPROACH_FEED = 500.0
DEFAULT_PROBE_FEED = 100.0
DEFAULT_B_MAX_FEED = 100.0
DEFAULT_B_ACCEL_TIME_S = 0.5
DEFAULT_B_DECEL_TIME_S = 0.5

DEFAULT_CUSTOM_INV_SAMPLES = 20000

DEFAULT_ORIENTATION_SEQUENCE = (0.0, 180.0)
DEFAULT_OSCILLATIONS_PER_ORIENTATION = 2.0
DEFAULT_ORIENTATION_MOVE_STEPS = 1200
DEFAULT_ORIENTATION_CAPTURE_STEPS = 100

DEFAULT_SWEEP_TIP_MIN_DEG = 0.0
DEFAULT_SWEEP_TIP_MAX_DEG = 180.0
DEFAULT_B_PHASE_OFFSET_DEG = -90.0   # starts at tip_min and ends at tip_min

DEFAULT_START_X = 100.0
DEFAULT_START_Y = 20.0
DEFAULT_START_Z = -155.0
DEFAULT_START_B = 0.0
DEFAULT_START_C = 0.0

DEFAULT_END_X = 100.0
DEFAULT_END_Y = 20.0
DEFAULT_END_Z = -155.0
DEFAULT_END_B = 0.0
DEFAULT_END_C = 0.0

DEFAULT_SAFE_APPROACH_Z = -155.0

DEFAULT_DWELL_BEFORE_MS = 0.5
DEFAULT_DWELL_AFTER_MS = 0
DEFAULT_INITIAL_SWEEP_WAIT_S = 4.0
DEFAULT_C_FLIP_FEED = 10000.0

DEFAULT_BBOX_X_MIN = 0.0
DEFAULT_BBOX_X_MAX = 200.0
DEFAULT_BBOX_Y_MIN = -20.0
DEFAULT_BBOX_Y_MAX = 200.0
DEFAULT_BBOX_Z_MIN = -200.0
DEFAULT_BBOX_Z_MAX = 0.0

DEFAULT_MANUAL_FOCUS = True
DEFAULT_MANUAL_FOCUS_VAL = 60
DEFAULT_CAMERA_WIDTH = 3840
DEFAULT_CAMERA_HEIGHT = 2160
DEFAULT_CAMERA_FLUSH_FRAMES = 1

DEFAULT_TRACKED_MOVE_SETTLE_S = 0.02
DEFAULT_TRAVEL_MOVE_SETTLE_S = 0.05
DEFAULT_CAPTURE_AT_START = False
DEFAULT_CAPTURE_EVERY_MOVE_POINT = False

DEFAULT_FLIP_RZ_SIGN = True

OFFPLANE_SIGN = -1.0


# =========================
# Data structures
# =========================

@dataclass
class Calibration:
    r_model: dict
    z_model: dict
    y_off_model: Optional[dict]
    tip_angle_model: Optional[dict]

    b_min: float
    b_max: float

    x_axis: str
    y_axis: str
    z_axis: str
    b_axis: str
    c_axis: str

    c_180_deg: float

    offplane_y_equation: Optional[str] = None
    offplane_y_r_squared: Optional[float] = None


@dataclass
class TrajectoryPoint:
    b: float
    c: float
    stage_xyz: np.ndarray
    segment_kind: str
    capture_image: bool = False
    tip_angle_deg: Optional[float] = None
    block_name: Optional[str] = None
    block_phase_01: Optional[float] = None
    oscillation_phase_rad: Optional[float] = None
    block_index: Optional[int] = None


# =========================
# Calibration / kinematics
# =========================

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


def evaluate_fit_model(model: Any, u: Any, default_if_none: Optional[float] = None) -> np.ndarray:
    u_arr = np.asarray(u, dtype=float)
    if model is None:
        if default_if_none is None:
            raise ValueError("Missing required fit model.")
        return np.full_like(u_arr, float(default_if_none), dtype=float)

    model_type = str(model.get("model_type", "polynomial")).lower()
    if model_type == "polynomial":
        return poly_eval(model.get("coefficients"), u_arr, default_if_none=default_if_none)

    if model_type == "pchip":
        x_knots = model.get("x_knots")
        y_knots = model.get("y_knots")
        if x_knots is None or y_knots is None:
            raise ValueError("PCHIP fit model is missing knots.")
        interp = PchipInterpolator(
            np.asarray(x_knots, dtype=float),
            np.asarray(y_knots, dtype=float),
            extrapolate=True,
        )
        return np.asarray(interp(u_arr), dtype=float)

    raise ValueError(f"Unsupported fit model type: {model_type}")


def legacy_poly_model(coeffs: Any, equation: Optional[str], value_name: str) -> Optional[dict]:
    if coeffs is None:
        return None
    coeff_list = np.asarray(coeffs, dtype=float).reshape(-1).tolist()
    return {
        "model_type": "polynomial",
        "basis": "monomial",
        "degree": len(coeff_list) - 1,
        "input_axis": "b_motor",
        "value_name": value_name,
        "coefficients": coeff_list,
        "equation": equation,
    }


def load_calibration(json_path: str) -> Calibration:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")

    with p.open("r") as f:
        data = json.load(f)

    fit_models = data.get("fit_models", {})
    cubic = data.get("cubic_coefficients", {})

    r_model = fit_models.get("r") or legacy_poly_model(
        cubic.get("r_coeffs"),
        cubic.get("r_equation"),
        "r",
    )
    z_model = fit_models.get("z") or legacy_poly_model(
        cubic.get("z_coeffs"),
        cubic.get("z_equation"),
        "z",
    )
    y_off_model = fit_models.get("offplane_y") or legacy_poly_model(
        cubic.get("offplane_y_coeffs"),
        cubic.get("offplane_y_equation"),
        "y_offplane_mm",
    )
    tip_angle_model = fit_models.get("tip_angle") or legacy_poly_model(
        cubic.get("tip_angle_coeffs"),
        cubic.get("tip_angle_equation"),
        "tip_angle_deg",
    )

    if r_model is None or z_model is None:
        raise ValueError("Calibration JSON is missing usable r/z fit models.")

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
    c_180 = float(motor_setup.get("rotation_axis_180_deg", 180.0))

    return Calibration(
        r_model=r_model,
        z_model=z_model,
        y_off_model=y_off_model,
        tip_angle_model=tip_angle_model,
        b_min=b_min,
        b_max=b_max,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        b_axis=b_axis,
        c_axis=c_axis,
        c_180_deg=c_180,
        offplane_y_equation=cubic.get("offplane_y_equation"),
        offplane_y_r_squared=(
            None if cubic.get("offplane_y_r_squared") is None
            else float(cubic["offplane_y_r_squared"])
        ),
    )


def eval_r(cal: Calibration, b: Any, flip_rz_sign: bool = False) -> np.ndarray:
    s = -1.0 if bool(flip_rz_sign) else 1.0
    return s * evaluate_fit_model(cal.r_model, b)


def eval_z(cal: Calibration, b: Any, flip_rz_sign: bool = False) -> np.ndarray:
    s = -1.0 if bool(flip_rz_sign) else 1.0
    return s * evaluate_fit_model(cal.z_model, b)


def eval_offplane_y(cal: Calibration, b: Any) -> np.ndarray:
    return OFFPLANE_SIGN * evaluate_fit_model(cal.y_off_model, b, default_if_none=0.0)


def eval_tip_angle_deg(cal: Calibration, b: Any) -> np.ndarray:
    return evaluate_fit_model(cal.tip_angle_model, b)


def predict_r_z_offplane(
    cal: Calibration,
    b: float,
    flip_rz_sign: bool = False,
) -> Tuple[float, float, float]:
    r = float(eval_r(cal, b, flip_rz_sign=flip_rz_sign))
    z = float(eval_z(cal, b, flip_rz_sign=flip_rz_sign))
    y_off = float(eval_offplane_y(cal, b))
    return r, z, y_off


def predict_tip_xyz_from_bc(
    cal: Calibration,
    b: float,
    c_deg: float,
    flip_rz_sign: bool = False,
) -> np.ndarray:
    r, z, y_off = predict_r_z_offplane(cal, b, flip_rz_sign=flip_rz_sign)
    c = math.radians(float(c_deg))
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def tip_offset_xyz_physical(
    cal: Calibration,
    b: float,
    c_deg: float,
    flip_rz_sign: bool = False,
) -> np.ndarray:
    return predict_tip_xyz_from_bc(cal, b, c_deg, flip_rz_sign=flip_rz_sign)


def stage_xyz_for_fixed_tip(
    cal: Calibration,
    p_tip_xyz: np.ndarray,
    b: float,
    c_deg: float,
    flip_rz_sign: bool = False,
) -> np.ndarray:
    return p_tip_xyz - tip_offset_xyz_physical(cal, b, c_deg, flip_rz_sign=flip_rz_sign)


# =========================
# Utilities
# =========================

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def clamp_c_bounded(c_deg: float) -> float:
    return clamp(float(c_deg), -360.0, 360.0)


def build_tip_angle_inverse_table(
    cal: Calibration,
    num_samples: int = DEFAULT_CUSTOM_INV_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray]:
    if cal.tip_angle_model is None:
        raise ValueError(
            "This motion mode requires a tip-angle fit model in the calibration JSON."
        )

    ns = max(1000, int(num_samples))
    b_samples = np.linspace(float(cal.b_min), float(cal.b_max), ns, dtype=float)
    angle_samples = eval_tip_angle_deg(cal, b_samples)

    order = np.argsort(angle_samples)
    angle_sorted = np.asarray(angle_samples[order], dtype=float)
    b_sorted = np.asarray(b_samples[order], dtype=float)

    angle_unique, unique_idx = np.unique(angle_sorted, return_index=True)
    b_unique = b_sorted[unique_idx]

    if angle_unique.size < 2:
        raise ValueError("Could not build a usable tip-angle inverse table from calibration.")

    return angle_unique, b_unique


def tip_angle_deg_to_b_clipped(
    requested_tip_angle_deg: float,
    angle_table_deg: np.ndarray,
    b_table: np.ndarray,
) -> Tuple[float, float]:
    amin = float(angle_table_deg[0])
    amax = float(angle_table_deg[-1])
    used_angle = clamp(float(requested_tip_angle_deg), amin, amax)
    b_val = float(np.interp(used_angle, angle_table_deg, b_table))
    return b_val, used_angle


def _tip_angle_for_block_phase_deg(
    block_phase_01: float,
    tip_min_deg: float,
    tip_max_deg: float,
    oscillations: float,
    phase_offset_deg: float,
) -> Tuple[float, float]:
    center = 0.5 * (float(tip_min_deg) + float(tip_max_deg))
    amp = 0.5 * (float(tip_max_deg) - float(tip_min_deg))
    ph = math.radians(float(phase_offset_deg))
    osc_phase = 2.0 * math.pi * float(oscillations) * float(block_phase_01) + ph
    return float(center + amp * math.sin(osc_phase)), float(osc_phase)


# =========================
# Fixed-C dual-orientation trajectory
# =========================

def _append_fixed_c_block(
    traj: List[TrajectoryPoint],
    cal: Calibration,
    p_tip_fixed: np.ndarray,
    angle_table_deg: np.ndarray,
    b_table: np.ndarray,
    c_deg: float,
    block_name: str,
    block_index: int,
    move_steps: int,
    capture_steps: int,
    tip_min_deg: float,
    tip_max_deg: float,
    oscillations: float,
    phase_offset_deg: float,
    flip_rz_sign: bool = False,
    include_first_point: bool = True,
    capture_every_move_point: bool = False,
):
    nmove = max(1, int(move_steps))
    ncap = max(1, int(capture_steps))

    capture_move_indices: Set[int] = set()
    if capture_every_move_point:
        capture_move_indices = set(range(0 if include_first_point else 1, nmove + 1))
    else:
        for j in range(0 if include_first_point else 1, ncap + 1):
            phase = j / float(ncap)
            idx = int(round(phase * nmove))
            idx = max(0 if include_first_point else 1, min(nmove, idx))
            capture_move_indices.add(idx)

    i_start = 0 if include_first_point else 1
    for i in range(i_start, nmove + 1):
        block_phase = i / float(nmove)
        req_tip, osc_phase = _tip_angle_for_block_phase_deg(
            block_phase_01=block_phase,
            tip_min_deg=tip_min_deg,
            tip_max_deg=tip_max_deg,
            oscillations=oscillations,
            phase_offset_deg=phase_offset_deg,
        )
        b_cmd, used_tip = tip_angle_deg_to_b_clipped(
            requested_tip_angle_deg=req_tip,
            angle_table_deg=angle_table_deg,
            b_table=b_table,
        )
        p_stage = stage_xyz_for_fixed_tip(
            cal=cal,
            p_tip_xyz=p_tip_fixed,
            b=b_cmd,
            c_deg=c_deg,
            flip_rz_sign=flip_rz_sign,
        )

        traj.append(
            TrajectoryPoint(
                b=float(b_cmd),
                c=float(c_deg),
                stage_xyz=p_stage,
                segment_kind="tracked_block",
                capture_image=(i in capture_move_indices),
                tip_angle_deg=float(used_tip),
                block_name=str(block_name),
                block_phase_01=float(block_phase),
                oscillation_phase_rad=float(osc_phase),
                block_index=int(block_index),
            )
        )


def generate_dual_orientation_fixed_tip_trajectory(
    cal: Calibration,
    p_tip_fixed: np.ndarray,
    move_steps_per_orientation: int = DEFAULT_ORIENTATION_MOVE_STEPS,
    capture_steps_per_orientation: int = DEFAULT_ORIENTATION_CAPTURE_STEPS,
    tip_min_deg: float = DEFAULT_SWEEP_TIP_MIN_DEG,
    tip_max_deg: float = DEFAULT_SWEEP_TIP_MAX_DEG,
    oscillations_per_orientation: float = DEFAULT_OSCILLATIONS_PER_ORIENTATION,
    b_phase_offset_deg: float = DEFAULT_B_PHASE_OFFSET_DEG,
    inverse_samples: int = DEFAULT_CUSTOM_INV_SAMPLES,
    capture_every_move_point: bool = False,
    flip_rz_sign: bool = False,
) -> Tuple[List[TrajectoryPoint], dict]:
    angle_table_deg, b_table = build_tip_angle_inverse_table(
        cal=cal,
        num_samples=int(inverse_samples),
    )

    available_tip_min = float(angle_table_deg[0])
    available_tip_max = float(angle_table_deg[-1])

    used_tip_min = clamp(float(tip_min_deg), available_tip_min, available_tip_max)
    used_tip_max = clamp(float(tip_max_deg), available_tip_min, available_tip_max)
    if used_tip_min > used_tip_max:
        used_tip_min, used_tip_max = used_tip_max, used_tip_min

    traj: List[TrajectoryPoint] = []

    c0 = 0.0
    c180 = float(cal.c_180_deg)

    _append_fixed_c_block(
        traj=traj,
        cal=cal,
        p_tip_fixed=p_tip_fixed,
        angle_table_deg=angle_table_deg,
        b_table=b_table,
        c_deg=c0,
        block_name="C0",
        block_index=0,
        move_steps=int(move_steps_per_orientation),
        capture_steps=int(capture_steps_per_orientation),
        tip_min_deg=float(used_tip_min),
        tip_max_deg=float(used_tip_max),
        oscillations=float(oscillations_per_orientation),
        phase_offset_deg=float(b_phase_offset_deg),
        flip_rz_sign=flip_rz_sign,
        include_first_point=True,
        capture_every_move_point=bool(capture_every_move_point),
    )

    _append_fixed_c_block(
        traj=traj,
        cal=cal,
        p_tip_fixed=p_tip_fixed,
        angle_table_deg=angle_table_deg,
        b_table=b_table,
        c_deg=c180,
        block_name="C180",
        block_index=1,
        move_steps=int(move_steps_per_orientation),
        capture_steps=int(capture_steps_per_orientation),
        tip_min_deg=float(used_tip_min),
        tip_max_deg=float(used_tip_max),
        oscillations=float(oscillations_per_orientation),
        phase_offset_deg=float(b_phase_offset_deg),
        flip_rz_sign=flip_rz_sign,
        include_first_point=True,
        capture_every_move_point=bool(capture_every_move_point),
    )

    n_captures = int(sum(1 for pt in traj if pt.capture_image))
    meta = {
        "requested_tip_min_deg": float(tip_min_deg),
        "requested_tip_max_deg": float(tip_max_deg),
        "used_tip_min_deg": float(used_tip_min),
        "used_tip_max_deg": float(used_tip_max),
        "available_tip_angle_range_deg": [available_tip_min, available_tip_max],
        "move_steps_per_orientation": int(move_steps_per_orientation),
        "capture_steps_per_orientation": int(capture_steps_per_orientation),
        "oscillations_per_orientation": float(oscillations_per_orientation),
        "b_phase_offset_deg": float(b_phase_offset_deg),
        "orientation_sequence_deg": [float(c0), float(c180)],
        "planned_capture_points": n_captures,
        "capture_every_move_point": bool(capture_every_move_point),
        "flip_rz_sign": bool(flip_rz_sign),
    }
    return traj, meta


# =========================
# Feed scheduling
# =========================

def _smoothstep01(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _speed_envelope_factor(
    t01: float,
    accel_s: float,
    decel_s: float,
    total_s: float,
    floor_frac: float = 0.08,
) -> float:
    if total_s <= 1e-9:
        return 1.0

    ta = max(0.0, float(accel_s))
    td = max(0.0, float(decel_s))
    t = float(t01) * float(total_s)

    up = 1.0
    dn = 1.0
    if ta > 1e-9:
        up = _smoothstep01(t / ta)
    if td > 1e-9:
        dn = _smoothstep01((total_s - t) / td)

    f = min(up, dn)
    return float(floor_frac + (1.0 - floor_frac) * f)


def plan_segment_feeds_with_b_envelope(
    traj: List[TrajectoryPoint],
    probe_feed_mm_min: float,
    b_max_feed_units_min: float,
    b_accel_time_s: float,
    b_decel_time_s: float,
    min_seg_time_s: float = 0.008,
) -> Tuple[List[float], dict]:
    nseg = max(0, len(traj) - 1)
    if nseg == 0:
        return [], {"est_total_time_s": 0.0, "max_est_b_speed_units_min": 0.0}

    probe_feed = max(1e-6, float(probe_feed_mm_min))
    bmax = max(1e-6, float(b_max_feed_units_min))

    xyzlens = []
    dbs = []
    for i in range(1, len(traj)):
        p0 = traj[i - 1].stage_xyz
        p1 = traj[i].stage_xyz
        b0 = traj[i - 1].b
        b1 = traj[i].b
        xyzlens.append(float(np.linalg.norm(p1 - p0)))
        dbs.append(abs(float(b1) - float(b0)))

    xyzlens = np.asarray(xyzlens, dtype=float)
    dbs = np.asarray(dbs, dtype=float)

    dt_xyz0 = xyzlens / (probe_feed / 60.0)
    dt_b0 = dbs / (bmax / 60.0)
    dt0 = np.maximum(dt_xyz0, dt_b0)
    dt0 = np.maximum(dt0, min_seg_time_s)
    total_est = float(np.sum(dt0))

    feeds = []
    dts = []
    t_cum = 0.0
    max_est_b_speed = 0.0

    for i in range(nseg):
        t_mid = t_cum + 0.5 * float(dt0[i])
        t01 = 0.0 if total_est <= 1e-9 else (t_mid / total_est)
        env = _speed_envelope_factor(
            t01=t01,
            accel_s=float(b_accel_time_s),
            decel_s=float(b_decel_time_s),
            total_s=total_est,
            floor_frac=0.08,
        )
        b_cap_i = bmax * env

        dt_xyz = xyzlens[i] / (probe_feed / 60.0)
        dt_b = dbs[i] / (b_cap_i / 60.0)
        dt = max(float(dt_xyz), float(dt_b), float(min_seg_time_s))

        if xyzlens[i] > 1e-9:
            f_i = 60.0 * xyzlens[i] / dt
            f_i = min(f_i, probe_feed)
            f_i = max(f_i, 1.0)
        else:
            f_i = probe_feed

        b_speed_est = (dbs[i] / dt) * 60.0 if dt > 1e-12 else 0.0
        max_est_b_speed = max(max_est_b_speed, b_speed_est)

        feeds.append(float(f_i))
        dts.append(float(dt))
        t_cum += dt

    return feeds, {
        "est_total_time_s": float(sum(dts)),
        "max_est_b_speed_units_min": float(max_est_b_speed),
        "mean_seg_time_ms": float(1000.0 * np.mean(dts)) if dts else 0.0,
    }


# =========================
# Diagnostics / utilities
# =========================

def compute_traj_meta(traj: List[TrajectoryPoint]) -> dict:
    if not traj:
        return {
            "n_samples": 0,
            "n_segments": 0,
            "x_stage_min": 0.0,
            "x_stage_max": 0.0,
            "y_stage_min": 0.0,
            "y_stage_max": 0.0,
            "z_stage_min": 0.0,
            "z_stage_max": 0.0,
            "b_min_used": 0.0,
            "b_max_used": 0.0,
            "c_min_used": 0.0,
            "c_max_used": 0.0,
            "xyz_path_len_mm": 0.0,
            "max_db_step": 0.0,
            "b_abs_path": 0.0,
            "n_capture_points": 0,
        }

    xyz = np.vstack([pt.stage_xyz for pt in traj])
    bb = np.array([pt.b for pt in traj], dtype=float)
    cc = np.array([pt.c for pt in traj], dtype=float)

    diffs_xyz = xyz[1:] - xyz[:-1] if len(xyz) > 1 else np.zeros((0, 3))
    seglens = np.linalg.norm(diffs_xyz, axis=1) if len(diffs_xyz) else np.array([], dtype=float)
    db = np.diff(bb) if len(bb) > 1 else np.array([], dtype=float)

    return {
        "n_samples": int(len(traj)),
        "n_segments": int(max(0, len(traj) - 1)),
        "x_stage_min": float(np.min(xyz[:, 0])),
        "x_stage_max": float(np.max(xyz[:, 0])),
        "y_stage_min": float(np.min(xyz[:, 1])),
        "y_stage_max": float(np.max(xyz[:, 1])),
        "z_stage_min": float(np.min(xyz[:, 2])),
        "z_stage_max": float(np.max(xyz[:, 2])),
        "b_min_used": float(np.min(bb)),
        "b_max_used": float(np.max(bb)),
        "c_min_used": float(np.min(cc)),
        "c_max_used": float(np.max(cc)),
        "xyz_path_len_mm": float(np.sum(seglens)) if len(seglens) else 0.0,
        "max_db_step": float(np.max(np.abs(db))) if len(db) else 0.0,
        "b_abs_path": float(np.sum(np.abs(db))) if len(db) else 0.0,
        "n_capture_points": int(sum(1 for pt in traj if pt.capture_image)),
    }


def _clamp_stage_xyz_to_bbox(
    x: float,
    y: float,
    z: float,
    bbox: dict,
    context: str,
    warn_log: List[str],
) -> Tuple[float, float, float]:
    def clamp_one(axis: str, value: float, lo: float, hi: float) -> float:
        if value < lo:
            warn_log.append(
                f"WARNING: {context} {axis}={value:.3f} below bbox min {lo:.3f}; clamped to {lo:.3f}"
            )
            return lo
        if value > hi:
            warn_log.append(
                f"WARNING: {context} {axis}={value:.3f} above bbox max {hi:.3f}; clamped to {hi:.3f}"
            )
            return hi
        return value

    xc = clamp_one("X", float(x), float(bbox["x_min"]), float(bbox["x_max"]))
    yc = clamp_one("Y", float(y), float(bbox["y_min"]), float(bbox["y_max"]))
    zc = clamp_one("Z", float(z), float(bbox["z_min"]), float(bbox["z_max"]))
    return xc, yc, zc


def split_trajectory_into_blocks(traj: List[TrajectoryPoint]) -> List[List[TrajectoryPoint]]:
    if not traj:
        return []

    blocks: List[List[TrajectoryPoint]] = []
    current: List[TrajectoryPoint] = [traj[0]]

    for pt in traj[1:]:
        prev = current[-1]
        if pt.block_index != prev.block_index:
            blocks.append(current)
            current = [pt]
        else:
            current.append(pt)

    blocks.append(current)
    return blocks


# =========================
# Acquisition runner
# =========================

class FixedTipPointTracker:
    """
    Robot + camera execution runner.
    """

    def __init__(
        self,
        parent_directory: str,
        project_name: str,
        allow_existing: bool = True,
        add_date: bool = True,
    ):
        parent_directory = os.path.abspath(parent_directory)
        os.makedirs(parent_directory, exist_ok=True)

        if add_date:
            folder_name = f"{project_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        else:
            folder_name = project_name

        self.run_folder = os.path.join(parent_directory, folder_name)
        if os.path.isdir(self.run_folder):
            if not allow_existing:
                raise FileExistsError(
                    f"Run folder already exists: {self.run_folder}. "
                    f"Use --allow-existing to reuse it."
                )
        else:
            os.makedirs(self.run_folder, exist_ok=True)

        self.point_tracking_folder = os.path.join(self.run_folder, "raw_image_data_folder")
        os.makedirs(self.point_tracking_folder, exist_ok=True)

        self.cam = None
        self.rrf = None
        self.cam_port = None

        print(f"Using run folder: {self.run_folder}")
        print(f"Using point-tracking folder: {self.point_tracking_folder}")

    # ---------- Camera ----------

    def connect_to_camera(
        self,
        cam_port: int = 0,
        show_preview: bool = False,
        enable_manual_focus: bool = True,
        manual_focus_val: float = 60,
        width: int = 3840,
        height: int = 2160,
    ):
        self.cam_port = cam_port
        self.cam = cv2.VideoCapture(cam_port)

        if not self.cam.isOpened():
            raise RuntimeError(f"Could not open camera at port {cam_port}")

        self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, int(width))
        self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, int(height))

        if enable_manual_focus:
            try:
                self.cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                self.cam.set(cv2.CAP_PROP_FOCUS, float(manual_focus_val))
                print(f"Manual focus enabled (FOCUS={manual_focus_val})")
            except Exception as e:
                print(f"Warning: could not set manual focus: {e}")
        else:
            try:
                self.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            except Exception:
                pass

        if show_preview:
            print("Showing camera preview. Press 'q' to close preview.")
            while True:
                ret, frame = self.cam.read()
                if not ret:
                    print("Camera preview read failed.")
                    break
                cv2.imshow("preview", frame)
                if (cv2.waitKey(1) & 0xFF) == ord("q"):
                    break
            cv2.destroyAllWindows()

        print("Camera connected.")

    def disconnect_camera(self):
        if self.cam is not None:
            self.cam.release()
            self.cam = None
            cv2.destroyAllWindows()
            print("Camera disconnected.")

    def capture_and_save(
        self,
        sample_idx: int,
        phase: str,
        x: float,
        y: float,
        z: float,
        b: float,
        c: float,
        flush_frames: int = 1,
        tip_angle_deg: Optional[float] = None,
        block_name: Optional[str] = None,
        block_phase_01: Optional[float] = None,
    ) -> Optional[str]:
        if self.cam is None:
            raise RuntimeError("Camera is not connected.")

        for _ in range(max(0, int(flush_frames))):
            _ = self.cam.read()

        ret, image = self.cam.read()
        if not ret:
            ret, image = self.cam.read()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

        extra = ""
        if block_name is not None:
            extra += f"_{block_name}"
        if tip_angle_deg is not None:
            extra += f"_TIP{float(tip_angle_deg):.3f}"
        if block_phase_01 is not None:
            extra += f"_BPH{float(block_phase_01):.5f}"

        filename = (
            f"{sample_idx:05d}"
            f"_{phase}"
            f"_X{x:.3f}_Y{y:.3f}_Z{z:.3f}"
            f"_B{b:.3f}_C{c:.3f}"
            f"{extra}"
            f"_{timestamp}.png"
        ).replace(" ", "_")

        path = os.path.join(self.point_tracking_folder, filename)
        if ret and image is not None:
            cv2.imwrite(path, image)
            print(f" ✓ Saved image: {filename}")
            return path

        print(f" ✗ Failed to capture image: {filename}")
        return None

    # ---------- Robot ----------

    def connect_to_robot(self, duet_web_address: str):
        self.rrf = DuetWebAPI(duet_web_address)
        print("Connection attempted. Requesting diagnostics.")
        resp = self.rrf.send_code("M122")
        print("Returned diagnostics data:")
        print(resp)
        print("Robot connected.")

    def disconnect_robot(self):
        self.rrf = None

    def wait_for_duet_motion_complete(self, extra_settle: float = 0.0):
        if self.rrf is None:
            raise RuntimeError("Robot is not connected.")

        try:
            self.rrf.send_code("M400")
        except Exception as e:
            print(f"Warning: M400 wait failed ({e}); applying settle only.")

        if extra_settle > 0:
            time.sleep(extra_settle)

    def send_absolute_move(self, feedrate: float, **axes_targets):
        if self.rrf is None:
            raise RuntimeError("Robot is not connected.")

        parts = ["G90", "G1"]
        for ax, val in axes_targets.items():
            if val is None:
                continue
            parts.append(f"{ax}{float(val):.3f}")
        parts.append(f"F{float(feedrate):.3f}")
        gcode = " ".join(parts)
        print(f" Command: {gcode}")
        self.rrf.send_code(gcode)

    def _move_to_pose_safe(
        self,
        cal: Calibration,
        pose: Tuple[float, float, float, float, float],
        safe_approach_z: float,
        travel_feed: float,
        settle_s: float,
    ):
        x, y, z, b, c = [float(v) for v in pose]

        self.send_absolute_move(
            travel_feed,
            **{
                cal.z_axis: float(safe_approach_z),
                cal.b_axis: b,
                cal.c_axis: clamp_c_bounded(c),
            }
        )
        self.wait_for_duet_motion_complete(extra_settle=settle_s)

        self.send_absolute_move(
            travel_feed,
            **{
                cal.x_axis: x,
                cal.y_axis: y,
                cal.b_axis: b,
                cal.c_axis: clamp_c_bounded(c),
            }
        )
        self.wait_for_duet_motion_complete(extra_settle=settle_s)

        self.send_absolute_move(
            travel_feed,
            **{
                cal.z_axis: z,
                cal.b_axis: b,
                cal.c_axis: clamp_c_bounded(c),
            }
        )
        self.wait_for_duet_motion_complete(extra_settle=settle_s)

    def _fine_land_on_point(
        self,
        cal: Calibration,
        x: float,
        y: float,
        z: float,
        b: float,
        c: float,
        fine_feed: float,
        settle_s: float,
    ):
        print(" Fine landing move for accuracy...")
        self.send_absolute_move(
            fine_feed,
            **{
                cal.x_axis: x,
                cal.y_axis: y,
                cal.z_axis: z,
                cal.b_axis: b,
                cal.c_axis: clamp_c_bounded(c),
            }
        )
        self.wait_for_duet_motion_complete(extra_settle=settle_s)

    def execute_motion_and_capture(
        self,
        cal: Calibration,
        traj: List[TrajectoryPoint],
        start_pose: Tuple[float, float, float, float, float],
        end_pose: Tuple[float, float, float, float, float],
        safe_approach_z: float,
        travel_feed: float,
        fine_approach_feed: float,
        probe_feed: float,
        b_max_feed: float,
        b_accel_time_s: float,
        b_decel_time_s: float,
        virtual_bbox: dict,
        dwell_before_ms: int = 0,
        dwell_after_ms: int = 0,
        use_segment_feed_scheduler: bool = True,
        tracked_move_settle_s: float = 0.0,
        travel_move_settle_s: float = 0.0,
        camera_flush_frames: int = 1,
        capture_at_start: bool = True,
        initial_sweep_wait_s: float = DEFAULT_INITIAL_SWEEP_WAIT_S,
        c_flip_feed: float = DEFAULT_C_FLIP_FEED,
    ):
        if self.cam is None:
            raise RuntimeError("Camera is not connected.")
        if self.rrf is None:
            raise RuntimeError("Robot is not connected.")

        bbox_warnings: List[str] = []
        sample_counter = 0

        blocks = split_trajectory_into_blocks(traj)

        print("\n" + "=" * 72)
        print("STARTING TRACKED POINT-ACQUISITION RUN")
        print("=" * 72)
        print(f"Tracked samples total: {len(traj)}")
        print(f"Orientation blocks: {len(blocks)}")

        print("\nSafe startup approach...")
        self._move_to_pose_safe(
            cal=cal,
            pose=start_pose,
            safe_approach_z=float(safe_approach_z),
            travel_feed=float(travel_feed),
            settle_s=float(travel_move_settle_s),
        )

        if int(dwell_before_ms) > 0:
            print(f"Dwell before tracked acquisition: {int(dwell_before_ms)} ms")
            time.sleep(float(dwell_before_ms) / 1000.0)

        for block_idx, block in enumerate(blocks):
            if not block:
                continue

            first_pt = block[0]
            block_name = first_pt.block_name or f"block_{block_idx}"
            print("\n" + "-" * 72)
            print(f"Starting block {block_idx + 1}/{len(blocks)}: {block_name} (C={first_pt.c:.3f})")
            print("-" * 72)

            block_sched_feeds: List[float] = []
            block_sched_meta = {
                "est_total_time_s": 0.0,
                "max_est_b_speed_units_min": 0.0,
                "mean_seg_time_ms": 0.0,
            }
            if use_segment_feed_scheduler and len(block) > 1:
                block_sched_feeds, block_sched_meta = plan_segment_feeds_with_b_envelope(
                    traj=block,
                    probe_feed_mm_min=float(probe_feed),
                    b_max_feed_units_min=float(b_max_feed),
                    b_accel_time_s=float(b_accel_time_s),
                    b_decel_time_s=float(b_decel_time_s),
                )

            print(f"Estimated block time: {block_sched_meta['est_total_time_s']:.3f} s")
            print(f"Estimated max B speed: {block_sched_meta['max_est_b_speed_units_min']:.3f} units/min")
            print(f"Mean segment time: {block_sched_meta['mean_seg_time_ms']:.2f} ms")

            x0, y0, z0 = _clamp_stage_xyz_to_bbox(
                first_pt.stage_xyz[0], first_pt.stage_xyz[1], first_pt.stage_xyz[2],
                virtual_bbox,
                f"{block_name} first point",
                bbox_warnings,
            )

            if block_idx > 0 and math.isclose(float(first_pt.c), float(cal.c_180_deg), abs_tol=1e-6):
                print(f"Flipping C to {float(first_pt.c):.3f} at feed {float(c_flip_feed):.3f}...")
                self.send_absolute_move(
                    float(c_flip_feed),
                    **{cal.c_axis: clamp_c_bounded(float(first_pt.c))}
                )
                self.wait_for_duet_motion_complete(extra_settle=float(travel_move_settle_s))

            print("Large move onto first tracked point of block...")
            self.send_absolute_move(
                float(travel_feed),
                **{
                    cal.x_axis: x0,
                    cal.y_axis: y0,
                    cal.z_axis: z0,
                    cal.b_axis: float(first_pt.b),
                    cal.c_axis: clamp_c_bounded(float(first_pt.c)),
                }
            )
            self.wait_for_duet_motion_complete(extra_settle=float(travel_move_settle_s))

            self._fine_land_on_point(
                cal=cal,
                x=x0,
                y=y0,
                z=z0,
                b=float(first_pt.b),
                c=float(first_pt.c),
                fine_feed=float(fine_approach_feed),
                settle_s=max(float(tracked_move_settle_s), 0.05),
            )

            if (
                float(initial_sweep_wait_s) > 0
                and (
                    block_idx == 0
                    or (block_idx > 0 and math.isclose(float(first_pt.c), float(cal.c_180_deg), abs_tol=1e-6))
                )
            ):
                sweep_label = "first" if block_idx == 0 else "second"
                print(f"Waiting {float(initial_sweep_wait_s):.3f} s before starting the {sweep_label} sweep...")
                time.sleep(float(initial_sweep_wait_s))

            if capture_at_start and first_pt.capture_image:
                sample_counter += 1
                self.capture_and_save(
                    sample_idx=sample_counter,
                    phase="tracked_start",
                    x=x0,
                    y=y0,
                    z=z0,
                    b=float(first_pt.b),
                    c=clamp_c_bounded(float(first_pt.c)),
                    flush_frames=camera_flush_frames,
                    tip_angle_deg=first_pt.tip_angle_deg,
                    block_name=first_pt.block_name,
                    block_phase_01=first_pt.block_phase_01,
                )
            else:
                print("Start point not captured.")

            print("Executing tracked oscillation block...")
            for i, point in enumerate(block[1:], start=1):
                x, y, z = _clamp_stage_xyz_to_bbox(
                    point.stage_xyz[0], point.stage_xyz[1], point.stage_xyz[2],
                    virtual_bbox,
                    f"{block_name} sample {i}",
                    bbox_warnings,
                )

                fseg = block_sched_feeds[i - 1] if block_sched_feeds else float(probe_feed)

                self.send_absolute_move(
                    fseg,
                    **{
                        cal.x_axis: x,
                        cal.y_axis: y,
                        cal.z_axis: z,
                        cal.b_axis: float(point.b),
                        cal.c_axis: clamp_c_bounded(float(point.c)),
                    }
                )
                self.wait_for_duet_motion_complete(extra_settle=float(tracked_move_settle_s))

                if point.capture_image:
                    sample_counter += 1
                    self.capture_and_save(
                        sample_idx=sample_counter,
                        phase=point.segment_kind,
                        x=x,
                        y=y,
                        z=z,
                        b=float(point.b),
                        c=clamp_c_bounded(float(point.c)),
                        flush_frames=camera_flush_frames,
                        tip_angle_deg=point.tip_angle_deg,
                        block_name=point.block_name,
                        block_phase_01=point.block_phase_01,
                    )

        if int(dwell_after_ms) > 0:
            print(f"Dwell after motion: {int(dwell_after_ms)} ms")
            time.sleep(float(dwell_after_ms) / 1000.0)

        print("\nSafe end move...")
        self._move_to_pose_safe(
            cal=cal,
            pose=end_pose,
            safe_approach_z=float(safe_approach_z),
            travel_feed=float(travel_feed),
            settle_s=float(travel_move_settle_s),
        )

        print("\n" + "=" * 72)
        print("RUN COMPLETE")
        print("=" * 72)
        print(f"Images saved: {sample_counter}")
        print(f"Point-tracking folder: {self.point_tracking_folder}")
        print(f"BBox warnings: {len(bbox_warnings)}")
        for msg in bbox_warnings:
            print(msg)

        return {
            "images_saved": sample_counter,
            "bbox_warnings": bbox_warnings,
        }


# =========================
# Main
# =========================

def main(args):
    cal = load_calibration(args.calibration)

    p_tip_fixed = np.array(
        [float(args.point_x), float(args.point_y), float(args.point_z)],
        dtype=float
    )

    sweep_tip_min_deg = float(args.sweep_tip_min_deg)
    sweep_tip_max_deg = float(args.sweep_tip_max_deg)
    if bool(args.b_0_to_90_only):
        sweep_tip_min_deg = 0.0
        sweep_tip_max_deg = 90.0

    traj, custom_meta = generate_dual_orientation_fixed_tip_trajectory(
        cal=cal,
        p_tip_fixed=p_tip_fixed,
        move_steps_per_orientation=int(args.orientation_move_steps),
        capture_steps_per_orientation=int(args.orientation_capture_steps),
        tip_min_deg=sweep_tip_min_deg,
        tip_max_deg=sweep_tip_max_deg,
        oscillations_per_orientation=float(args.oscillations_per_orientation),
        b_phase_offset_deg=float(args.b_phase_offset_deg),
        inverse_samples=int(args.custom_inverse_samples),
        capture_every_move_point=bool(args.capture_every_move_point),
        flip_rz_sign=bool(args.flip_rz_sign),
    )

    meta = compute_traj_meta(traj)
    print("Trajectory summary:")
    print(f"  Samples: {meta['n_samples']} (segments={meta['n_segments']})")
    print(f"  Capture points: {meta['n_capture_points']}")
    print(f"  B range used: [{meta['b_min_used']:.3f}, {meta['b_max_used']:.3f}]")
    if cal.tip_angle_model is not None and meta["n_samples"] > 0:
        bb = np.array([meta["b_min_used"], meta["b_max_used"]], dtype=float)
        tip_angle_used = eval_tip_angle_deg(cal, bb)
        print(
            "  Tip-angle range at used B endpoints: "
            f"[{float(np.min(tip_angle_used)):.3f}, {float(np.max(tip_angle_used)):.3f}] deg"
        )
    print(f"  C range used: [{meta['c_min_used']:.3f}, {meta['c_max_used']:.3f}]")
    print(f"  XYZ path length: {meta['xyz_path_len_mm']:.3f} mm")
    print(f"  Max B step: {meta['max_db_step']:.6f}")

    print("Block summary:")
    print(f"  Requested tip min/max: [{custom_meta['requested_tip_min_deg']}, {custom_meta['requested_tip_max_deg']}]")
    print(f"  Used tip min/max:      [{custom_meta['used_tip_min_deg']}, {custom_meta['used_tip_max_deg']}]")
    print(f"  B 0..90 only mode: {bool(args.b_0_to_90_only)}")
    print(f"  Available calibrated tip-angle range: {custom_meta['available_tip_angle_range_deg']}")
    print(f"  Move steps per orientation: {custom_meta['move_steps_per_orientation']}")
    print(f"  Capture steps per orientation: {custom_meta['capture_steps_per_orientation']}")
    print(f"  Oscillations per orientation: {custom_meta['oscillations_per_orientation']}")
    print(f"  B phase offset deg: {custom_meta['b_phase_offset_deg']}")
    print(f"  Orientation sequence (deg): {custom_meta['orientation_sequence_deg']}")
    print(f"  Planned capture points: {custom_meta['planned_capture_points']}")
    print(f"  Capture every move point: {custom_meta['capture_every_move_point']}")
    print(f"  flip_rz_sign: {custom_meta['flip_rz_sign']}")

    start_pose = (
        float(args.start_x),
        float(args.start_y),
        float(args.start_z),
        float(args.start_b),
        float(args.start_c),
    )
    end_pose = (
        float(args.end_x),
        float(args.end_y),
        float(args.end_z),
        float(args.end_b),
        float(args.end_c),
    )

    virtual_bbox = {
        "x_min": float(args.bbox_x_min),
        "x_max": float(args.bbox_x_max),
        "y_min": float(args.bbox_y_min),
        "y_max": float(args.bbox_y_max),
        "z_min": float(args.bbox_z_min),
        "z_max": float(args.bbox_z_max),
    }
    if virtual_bbox["x_min"] > virtual_bbox["x_max"]:
        virtual_bbox["x_min"], virtual_bbox["x_max"] = virtual_bbox["x_max"], virtual_bbox["x_min"]
    if virtual_bbox["y_min"] > virtual_bbox["y_max"]:
        virtual_bbox["y_min"], virtual_bbox["y_max"] = virtual_bbox["y_max"], virtual_bbox["y_min"]
    if virtual_bbox["z_min"] > virtual_bbox["z_max"]:
        virtual_bbox["z_min"], virtual_bbox["z_max"] = virtual_bbox["z_max"], virtual_bbox["z_min"]

    runner = FixedTipPointTracker(
        parent_directory=args.parent_directory,
        project_name=args.project_name,
        allow_existing=bool(args.allow_existing),
        add_date=bool(args.add_date),
    )

    try:
        runner.connect_to_camera(
            cam_port=int(args.cam_port),
            show_preview=bool(args.show_preview),
            enable_manual_focus=bool(args.enable_manual_focus),
            manual_focus_val=float(args.manual_focus_val),
            width=int(args.camera_width),
            height=int(args.camera_height),
        )

        runner.connect_to_robot(args.duet_web_address)

        results = runner.execute_motion_and_capture(
            cal=cal,
            traj=traj,
            start_pose=start_pose,
            end_pose=end_pose,
            safe_approach_z=float(args.safe_approach_z),
            travel_feed=float(args.travel_feed),
            fine_approach_feed=float(args.fine_approach_feed),
            probe_feed=float(args.probe_feed),
            b_max_feed=float(args.b_max_feed),
            b_accel_time_s=float(args.b_accel_time),
            b_decel_time_s=float(args.b_decel_time),
            virtual_bbox=virtual_bbox,
            dwell_before_ms=int(args.dwell_before_ms),
            dwell_after_ms=int(args.dwell_after_ms),
            use_segment_feed_scheduler=(not bool(args.disable_segment_feed_scheduler)),
            tracked_move_settle_s=float(args.tracked_move_settle_s),
            travel_move_settle_s=float(args.travel_move_settle_s),
            camera_flush_frames=int(args.camera_flush_frames),
            capture_at_start=bool(args.capture_at_start),
            initial_sweep_wait_s=float(args.initial_sweep_wait_s),
        )

        print("\nFinal results:")
        print(results)

    finally:
        try:
            runner.disconnect_camera()
        except Exception:
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Run a fixed-tip tracked XYZ/B/C motion at C=0 and C=180 only, "
            "with 6 B oscillations at each orientation."
        )
    )

    # Run / folders
    ap.add_argument("--parent-directory", default=os.getcwd(), help="Parent folder for the run output.")
    ap.add_argument("--project-name", default=DEFAULT_PROJECT_NAME, help="Run folder name.")
    ap.add_argument("--allow-existing", action="store_true", default=DEFAULT_ALLOW_EXISTING,
                    help="Allow reuse of an existing run folder.")
    ap.add_argument("--add-date", action="store_true", default=DEFAULT_ADD_DATE,
                    help="Append timestamp to the run folder name.")

    # Connectivity
    ap.add_argument("--duet-web-address", default=DEFAULT_DUET_WEB_ADDRESS, help="Duet web address.")
    ap.add_argument("--cam-port", type=int, default=DEFAULT_CAMERA_PORT, help="Camera port index.")

    # Camera
    ap.add_argument("--show-preview", action="store_true", help="Show camera preview before the run.")
    ap.add_argument("--enable-manual-focus", action="store_true", default=DEFAULT_MANUAL_FOCUS,
                    help="Enable manual focus on camera.")
    ap.add_argument("--manual-focus-val", type=float, default=DEFAULT_MANUAL_FOCUS_VAL,
                    help="Manual focus value.")
    ap.add_argument("--camera-width", type=int, default=DEFAULT_CAMERA_WIDTH,
                    help="Camera capture width.")
    ap.add_argument("--camera-height", type=int, default=DEFAULT_CAMERA_HEIGHT,
                    help="Camera capture height.")
    ap.add_argument("--camera-flush-frames", type=int, default=DEFAULT_CAMERA_FLUSH_FRAMES,
                    help="Frames to flush before each capture.")

    # Calibration input
    ap.add_argument("--calibration", required=True, help="Path to calibration JSON.")

    # Fixed tip point
    ap.add_argument("--point-x", type=float, default=DEFAULT_POINT_X, help="Fixed tip X (Cartesian/world).")
    ap.add_argument("--point-y", type=float, default=DEFAULT_POINT_Y, help="Fixed tip Y (Cartesian/world).")
    ap.add_argument("--point-z", type=float, default=DEFAULT_POINT_Z, help="Fixed tip Z (Cartesian/world).")

    # Sign correction
    ap.add_argument(
        "--flip-rz-sign",
        action="store_true",
        default=DEFAULT_FLIP_RZ_SIGN,
        help="Multiply the polynomial-derived r and z offsets by -1. Use this if your calibration file has flipped r/z signs.",
    )

    # Fixed-C dual-orientation sweep controls
    ap.add_argument("--orientation-move-steps", type=int, default=DEFAULT_ORIENTATION_MOVE_STEPS,
                    help="Tracked motion segments used for each fixed-C orientation block.")
    ap.add_argument("--orientation-capture-steps", type=int, default=DEFAULT_ORIENTATION_CAPTURE_STEPS,
                    help="Capture opportunity samples evaluated over each fixed-C orientation block.")
    ap.add_argument("--oscillations-per-orientation", type=float, default=DEFAULT_OSCILLATIONS_PER_ORIENTATION,
                    help="How many B oscillations occur at each fixed C orientation. Default is 6.")
    ap.add_argument("--sweep-tip-min-deg", type=float, default=DEFAULT_SWEEP_TIP_MIN_DEG,
                    help="Minimum tip angle during the B oscillation.")
    ap.add_argument("--sweep-tip-max-deg", type=float, default=DEFAULT_SWEEP_TIP_MAX_DEG,
                    help="Maximum tip angle during the B oscillation.")
    ap.add_argument("--b-0-to-90-only", action="store_true", default=False,
                    help="Force the cyclic B/tip oscillation range to 0..90 deg.")
    ap.add_argument("--b-phase-offset-deg", type=float, default=DEFAULT_B_PHASE_OFFSET_DEG,
                    help="Phase offset for the B oscillation. -90 starts at tip_min and ends at tip_min.")
    ap.add_argument("--custom-inverse-samples", type=int, default=DEFAULT_CUSTOM_INV_SAMPLES,
                    help="Dense sampling count used for numeric tip-angle -> B inversion.")
    ap.add_argument("--capture-every-move-point", action="store_true", default=DEFAULT_CAPTURE_EVERY_MOVE_POINT,
                    help="Capture at every tracked motion point instead of only at capture-step samples.")

    # Feedrates
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED,
                    help="Feedrate for large moves.")
    ap.add_argument("--fine-approach-feed", type=float, default=DEFAULT_FINE_APPROACH_FEED,
                    help="Slow final landing feed for the first point of each orientation block.")
    ap.add_argument("--probe-feed", type=float, default=DEFAULT_PROBE_FEED,
                    help="Nominal tracked-motion XYZ feed.")
    ap.add_argument("--b-max-feed", type=float, default=DEFAULT_B_MAX_FEED,
                    help="Max allowed B-axis speed used by the segment feed scheduler.")

    # Tracked B cap + feed scheduler
    ap.add_argument("--b-accel-time", type=float, default=DEFAULT_B_ACCEL_TIME_S)
    ap.add_argument("--b-decel-time", type=float, default=DEFAULT_B_DECEL_TIME_S)
    ap.add_argument("--disable-segment-feed-scheduler", action="store_true",
                    help="Disable per-segment feed scheduling.")

    # Optional waits / capture behavior
    ap.add_argument("--dwell-before-ms", type=int, default=DEFAULT_DWELL_BEFORE_MS)
    ap.add_argument("--dwell-after-ms", type=int, default=DEFAULT_DWELL_AFTER_MS)
    ap.add_argument("--initial-sweep-wait-s", type=float, default=DEFAULT_INITIAL_SWEEP_WAIT_S,
                    help="Wait after the first large move to the first tracked point, before starting the first sweep.")
    ap.add_argument("--tracked-move-settle-s", type=float, default=DEFAULT_TRACKED_MOVE_SETTLE_S,
                    help="Extra settle time after each tracked move, before capture.")
    ap.add_argument("--travel-move-settle-s", type=float, default=DEFAULT_TRAVEL_MOVE_SETTLE_S,
                    help="Extra settle time after travel moves.")
    ap.add_argument("--capture-at-start", action="store_true", default=DEFAULT_CAPTURE_AT_START,
                    help="Capture at the first point of each block if that point is marked for acquisition.")

    # Startup / end poses
    ap.add_argument("--safe-approach-z", type=float, default=DEFAULT_SAFE_APPROACH_Z)

    ap.add_argument("--start-x", type=float, default=DEFAULT_START_X)
    ap.add_argument("--start-y", type=float, default=DEFAULT_START_Y)
    ap.add_argument("--start-z", type=float, default=DEFAULT_START_Z)
    ap.add_argument("--start-b", type=float, default=DEFAULT_START_B)
    ap.add_argument("--start-c", type=float, default=DEFAULT_START_C)

    ap.add_argument("--end-x", type=float, default=DEFAULT_END_X)
    ap.add_argument("--end-y", type=float, default=DEFAULT_END_Y)
    ap.add_argument("--end-z", type=float, default=DEFAULT_END_Z)
    ap.add_argument("--end-b", type=float, default=DEFAULT_END_B)
    ap.add_argument("--end-c", type=float, default=DEFAULT_END_C)

    # Virtual stage-space bounding box
    ap.add_argument("--bbox-x-min", type=float, default=DEFAULT_BBOX_X_MIN)
    ap.add_argument("--bbox-x-max", type=float, default=DEFAULT_BBOX_X_MAX)
    ap.add_argument("--bbox-y-min", type=float, default=DEFAULT_BBOX_Y_MIN)
    ap.add_argument("--bbox-y-max", type=float, default=DEFAULT_BBOX_Y_MAX)
    ap.add_argument("--bbox-z-min", type=float, default=DEFAULT_BBOX_Z_MIN)
    ap.add_argument("--bbox-z-max", type=float, default=DEFAULT_BBOX_Z_MAX)

    main(ap.parse_args())
