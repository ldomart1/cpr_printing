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
    * queued constant-feed tracked motion
    * explicit M400 waits by default, or estimated-time settled captures
      with --settled-capture-mode
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
import csv
import json
import math
import os
import subprocess
import sys
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
    DuetWebAPI = None


# =========================
# Defaults
# =========================

DEFAULT_DUET_WEB_ADDRESS = "http://192.168.2.21"
DEFAULT_CAMERA_PORT = 0
DEFAULT_PROJECT_NAME = "Hysteresis_Run"
DEFAULT_ALLOW_EXISTING = True
DEFAULT_ADD_DATE = True

DEFAULT_POINT_X = 90.0
DEFAULT_POINT_Y = 55.0
DEFAULT_POINT_Z = -150

DEFAULT_TRAVEL_FEED = 8000.0
DEFAULT_FINE_APPROACH_FEED = 180.0
DEFAULT_PROBE_FEED = 180.0
DEFAULT_B_MAX_FEED = 500.0
DEFAULT_B_ACCEL_TIME_S = 0.05
DEFAULT_B_DECEL_TIME_S = 0.05

DEFAULT_CUSTOM_INV_SAMPLES = 20000

DEFAULT_ORIENTATION_SEQUENCE = (0.0, 180.0)
DEFAULT_OSCILLATIONS_PER_ORIENTATION = 1.0
DEFAULT_ORIENTATION_MOVE_STEPS = 3000
DEFAULT_ORIENTATION_CAPTURE_STEPS = 100

DEFAULT_SWEEP_TIP_MIN_DEG = 0.0
DEFAULT_SWEEP_TIP_MAX_DEG = 180.0
DEFAULT_B_PHASE_OFFSET_DEG = -90.0   # starts at tip_min and ends at tip_min

DEFAULT_START_X = 100.0
DEFAULT_START_Y = 55.0
DEFAULT_START_Z = -150
DEFAULT_START_B = 0.0
DEFAULT_START_C = 0.0

DEFAULT_END_X = 100.0
DEFAULT_END_Y = 55.0
DEFAULT_END_Z = -150
DEFAULT_END_B = 0.0
DEFAULT_END_C = 0.0

DEFAULT_SAFE_APPROACH_Z = -150

DEFAULT_DWELL_BEFORE_MS = 0.5
DEFAULT_DWELL_AFTER_MS = 0
DEFAULT_INITIAL_SWEEP_WAIT_S = 6.0
DEFAULT_C_FLIP_FEED = 20000.0
DEFAULT_C_FLIP_DELAY_S = 4.0
DEFAULT_PULL_SEQUENCE_BUFFER_S = 3.0

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
DEFAULT_B_EXTRA_SETTLE_S = 0.0
DEFAULT_INTER_COMMAND_DELAY_S = 0.005
DEFAULT_CAPTURE_AT_START = False
DEFAULT_CAPTURE_EVERY_MOVE_POINT = False
DEFAULT_SETTLED_CAPTURE_MODE = True
DEFAULT_SETTLED_CAPTURE_BUFFER_S = 0.5
DEFAULT_MIN_ESTIMATED_MOVE_TIME_S = 0.020

DEFAULT_FLIP_RZ_SIGN = True
DEFAULT_POST_TIP_DETECTION_MODE = "classical"
DEFAULT_POST_TIP_REFINER_MODEL = (
    "CNN_Calib/"
    "processed_image_data_folder/tip_refinement_model/best_tip_refiner.pt"
)
DEFAULT_Y_OFFSET_FIT = "avg_pchip"

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
    phase_models: dict
    default_motion_phase: str

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
    motion_phase: Optional[str] = None


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


def _normalize_motion_phase_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _extract_phase_models(data: dict) -> Tuple[dict, str]:
    phase_payload = data.get("fit_models_by_phase") or {}
    phase_models: dict = {}
    for raw_phase_name, models in phase_payload.items():
        phase_name = _normalize_motion_phase_name(raw_phase_name)
        if phase_name is None or not isinstance(models, dict):
            continue
        phase_models[phase_name] = dict(models)

    default_phase = _normalize_motion_phase_name(
        data.get("default_phase_for_legacy_access")
    )
    if default_phase is None or default_phase not in phase_models:
        if "pull" in phase_models:
            default_phase = "pull"
        elif phase_models:
            default_phase = next(iter(phase_models))
        else:
            default_phase = "pull"

    return phase_models, default_phase


def _select_y_offset_model(
    fit_models: dict,
    default_phase_models: dict,
    cubic: dict,
    y_offset_fit: str,
) -> Optional[dict]:
    mode = str(y_offset_fit).strip().lower()

    if mode == "avg_pchip":
        return (
            fit_models.get("offplane_y_avg_pchip")
            or fit_models.get("offplane_y")
            or default_phase_models.get("offplane_y_avg_pchip")
            or default_phase_models.get("offplane_y")
            or legacy_poly_model(
                cubic.get("offplane_y_coeffs"),
                cubic.get("offplane_y_equation"),
                "y_offplane_mm",
            )
        )

    if mode == "avg_cubic":
        return (
            fit_models.get("offplane_y_avg_cubic")
            or fit_models.get("offplane_y_cubic")
            or default_phase_models.get("offplane_y_cubic")
            or legacy_poly_model(
                cubic.get("offplane_y_cubic_coeffs"),
                cubic.get("offplane_y_cubic_equation"),
                "y_offplane_mm",
            )
            or legacy_poly_model(
                cubic.get("offplane_y_coeffs"),
                cubic.get("offplane_y_equation"),
                "y_offplane_mm",
            )
        )

    if mode == "pchip":
        return (
            fit_models.get("offplane_y_pchip")
            or fit_models.get("offplane_y")
            or default_phase_models.get("offplane_y_pchip")
            or default_phase_models.get("offplane_y")
            or fit_models.get("offplane_y_avg_pchip")
            or legacy_poly_model(
                cubic.get("offplane_y_coeffs"),
                cubic.get("offplane_y_equation"),
                "y_offplane_mm",
            )
        )

    if mode == "cubic":
        return (
            fit_models.get("offplane_y_cubic")
            or default_phase_models.get("offplane_y_cubic")
            or fit_models.get("offplane_y_avg_cubic")
            or legacy_poly_model(
                cubic.get("offplane_y_cubic_coeffs"),
                cubic.get("offplane_y_cubic_equation"),
                "y_offplane_mm",
            )
            or legacy_poly_model(
                cubic.get("offplane_y_coeffs"),
                cubic.get("offplane_y_equation"),
                "y_offplane_mm",
            )
        )

    if mode == "legacy":
        return legacy_poly_model(
            cubic.get("offplane_y_coeffs"),
            cubic.get("offplane_y_equation"),
            "y_offplane_mm",
        )

    raise ValueError(f"Unsupported y-offset fit selection: {y_offset_fit}")


def load_calibration(json_path: str, y_offset_fit: str = DEFAULT_Y_OFFSET_FIT) -> Calibration:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")

    with p.open("r") as f:
        data = json.load(f)

    phase_models, default_phase = _extract_phase_models(data)
    fit_models = data.get("fit_models", {})
    cubic = data.get("cubic_coefficients", {})

    default_phase_models = phase_models.get(default_phase, {})

    r_model = fit_models.get("r") or default_phase_models.get("r") or legacy_poly_model(
        cubic.get("r_coeffs"),
        cubic.get("r_equation"),
        "r",
    )
    z_model = fit_models.get("z") or default_phase_models.get("z") or legacy_poly_model(
        cubic.get("z_coeffs"),
        cubic.get("z_equation"),
        "z",
    )
    y_off_model = _select_y_offset_model(
        fit_models=fit_models,
        default_phase_models=default_phase_models,
        cubic=cubic,
        y_offset_fit=y_offset_fit,
    )
    tip_angle_model = fit_models.get("tip_angle") or default_phase_models.get("tip_angle") or legacy_poly_model(
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
        phase_models=phase_models,
        default_motion_phase=default_phase,
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


def _select_fit_model(cal: Calibration, model_name: str, motion_phase: Optional[str] = None) -> Any:
    phase_name = _normalize_motion_phase_name(motion_phase) or cal.default_motion_phase
    phase_model = cal.phase_models.get(phase_name, {}).get(model_name)
    if phase_model is not None:
        return phase_model

    fallback_attr = {
        "r": cal.r_model,
        "z": cal.z_model,
        "offplane_y": cal.y_off_model,
        "tip_angle": cal.tip_angle_model,
    }.get(model_name)
    return fallback_attr


def eval_r(
    cal: Calibration,
    b: Any,
    flip_rz_sign: bool = False,
    motion_phase: Optional[str] = None,
) -> np.ndarray:
    s = -1.0 * (-1.0 if bool(flip_rz_sign) else 1.0)
    return s * evaluate_fit_model(_select_fit_model(cal, "r", motion_phase=motion_phase), b)


def eval_z(
    cal: Calibration,
    b: Any,
    flip_rz_sign: bool = False,
    motion_phase: Optional[str] = None,
) -> np.ndarray:
    return evaluate_fit_model(_select_fit_model(cal, "z", motion_phase=motion_phase), b)


def eval_offplane_y(cal: Calibration, b: Any, motion_phase: Optional[str] = None) -> np.ndarray:
    return OFFPLANE_SIGN * evaluate_fit_model(
        _select_fit_model(cal, "offplane_y", motion_phase=motion_phase),
        b,
        default_if_none=0.0,
    )


def eval_tip_angle_deg(cal: Calibration, b: Any, motion_phase: Optional[str] = None) -> np.ndarray:
    return evaluate_fit_model(_select_fit_model(cal, "tip_angle", motion_phase=motion_phase), b)


def predict_r_z_offplane(
    cal: Calibration,
    b: float,
    flip_rz_sign: bool = False,
    motion_phase: Optional[str] = None,
) -> Tuple[float, float, float]:
    r = float(eval_r(cal, b, flip_rz_sign=flip_rz_sign, motion_phase=motion_phase))
    z = float(eval_z(cal, b, flip_rz_sign=flip_rz_sign, motion_phase=motion_phase))
    y_off = float(eval_offplane_y(cal, b, motion_phase=motion_phase))
    return r, z, y_off


def predict_tip_xyz_from_bc(
    cal: Calibration,
    b: float,
    c_deg: float,
    flip_rz_sign: bool = False,
    motion_phase: Optional[str] = None,
) -> np.ndarray:
    r, z, y_off = predict_r_z_offplane(
        cal,
        b,
        flip_rz_sign=flip_rz_sign,
        motion_phase=motion_phase,
    )
    c = math.radians(float(c_deg))
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def tip_offset_xyz_physical(
    cal: Calibration,
    b: float,
    c_deg: float,
    flip_rz_sign: bool = False,
    motion_phase: Optional[str] = None,
) -> np.ndarray:
    return predict_tip_xyz_from_bc(
        cal,
        b,
        c_deg,
        flip_rz_sign=flip_rz_sign,
        motion_phase=motion_phase,
    )


def stage_xyz_for_fixed_tip(
    cal: Calibration,
    p_tip_xyz: np.ndarray,
    b: float,
    c_deg: float,
    flip_rz_sign: bool = False,
    motion_phase: Optional[str] = None,
) -> np.ndarray:
    return p_tip_xyz - tip_offset_xyz_physical(
        cal,
        b,
        c_deg,
        flip_rz_sign=flip_rz_sign,
        motion_phase=motion_phase,
    )


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


def infer_motion_phase_for_b(
    cal: Calibration,
    prev_b: Optional[float],
    curr_b: float,
    next_b: Optional[float],
) -> str:
    deltas = []
    if prev_b is not None:
        deltas.append(float(curr_b) - float(prev_b))
    if next_b is not None:
        deltas.append(float(next_b) - float(curr_b))

    for delta in deltas:
        if abs(delta) <= 1e-9:
            continue
        return "pull" if delta < 0.0 else "release"

    return cal.default_motion_phase


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

    raw_points = []
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
        raw_points.append(
            {
                "b": float(b_cmd),
                "tip_angle_deg": float(used_tip),
                "block_phase_01": float(block_phase),
                "oscillation_phase_rad": float(osc_phase),
                "capture_image": (i in capture_move_indices),
            }
        )

    last_pull_stage_xyz: Optional[np.ndarray] = None
    for idx, point in enumerate(raw_points):
        prev_b = None if idx == 0 else raw_points[idx - 1]["b"]
        next_b = None if idx + 1 >= len(raw_points) else raw_points[idx + 1]["b"]
        motion_phase = infer_motion_phase_for_b(
            cal=cal,
            prev_b=prev_b,
            curr_b=point["b"],
            next_b=next_b,
        )
        tracked_pull_stage = stage_xyz_for_fixed_tip(
            cal=cal,
            p_tip_xyz=p_tip_fixed,
            b=point["b"],
            c_deg=c_deg,
            flip_rz_sign=flip_rz_sign,
            motion_phase="pull",
        )
        if motion_phase == "pull" or last_pull_stage_xyz is None:
            p_stage = tracked_pull_stage
            last_pull_stage_xyz = np.array(tracked_pull_stage, dtype=float)
        else:
            p_stage = np.array(last_pull_stage_xyz, dtype=float)

        traj.append(
            TrajectoryPoint(
                b=point["b"],
                c=float(c_deg),
                stage_xyz=p_stage,
                segment_kind="tracked_block",
                capture_image=bool(point["capture_image"]) and motion_phase == "pull",
                tip_angle_deg=point["tip_angle_deg"],
                block_name=str(block_name),
                block_phase_01=point["block_phase_01"],
                oscillation_phase_rad=point["oscillation_phase_rad"],
                block_index=int(block_index),
                motion_phase=motion_phase,
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
        self.command_log: List[dict] = []
        self.commanded_axes: dict = {}
        self.estimated_motion_done_at: float = time.monotonic()

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
        motion_phase: Optional[str] = None,
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
        if motion_phase is not None:
            extra += f"_DIR{str(motion_phase)}"

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
        if DuetWebAPI is None:
            raise ImportError("Missing duetwebapi. Install with: pip install duetwebapi==1.1.0")
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

    def _seed_commanded_pose(self, cal: Calibration, pose: Tuple[float, float, float, float, float]):
        x, y, z, b, c = [float(v) for v in pose]
        self.commanded_axes = {
            cal.x_axis: x,
            cal.y_axis: y,
            cal.z_axis: z,
            cal.b_axis: b,
            cal.c_axis: clamp_c_bounded(c),
        }
        self.estimated_motion_done_at = time.monotonic()

    def _estimate_move_time_s(
        self,
        cal: Calibration,
        previous_axes: dict,
        axes_targets: dict,
        feedrate: float,
        b_max_feed: float,
        min_move_time_s: float = DEFAULT_MIN_ESTIMATED_MOVE_TIME_S,
    ) -> float:
        feed_units_s = max(1e-9, float(feedrate) / 60.0)
        b_units_s = max(1e-9, min(float(feedrate), float(b_max_feed)) / 60.0)

        xyz_sq = 0.0
        for axis in (cal.x_axis, cal.y_axis, cal.z_axis):
            if axis not in axes_targets or axes_targets[axis] is None:
                continue
            if axis not in previous_axes:
                continue
            delta = float(axes_targets[axis]) - float(previous_axes[axis])
            xyz_sq += delta * delta
        xyz_time_s = math.sqrt(xyz_sq) / feed_units_s if xyz_sq > 0 else 0.0

        b_time_s = 0.0
        if cal.b_axis in axes_targets and axes_targets[cal.b_axis] is not None and cal.b_axis in previous_axes:
            b_delta = abs(float(axes_targets[cal.b_axis]) - float(previous_axes[cal.b_axis]))
            b_time_s = b_delta / b_units_s

        c_time_s = 0.0
        if cal.c_axis in axes_targets and axes_targets[cal.c_axis] is not None and cal.c_axis in previous_axes:
            c_delta = abs(float(axes_targets[cal.c_axis]) - float(previous_axes[cal.c_axis]))
            c_time_s = c_delta / feed_units_s

        est_s = max(float(xyz_time_s), float(b_time_s), float(c_time_s))
        if est_s <= 0.0:
            return 0.0
        return max(float(est_s), float(min_move_time_s))

    def _record_estimated_motion(
        self,
        cal: Calibration,
        command_record: dict,
        b_max_feed: float,
    ) -> float:
        est_s = self._estimate_move_time_s(
            cal=cal,
            previous_axes=command_record.get("previous_axes", {}),
            axes_targets=command_record.get("axes_targets", {}),
            feedrate=float(command_record.get("feedrate", 0.0)),
            b_max_feed=float(b_max_feed),
        )
        now = time.monotonic()
        self.estimated_motion_done_at = max(now, self.estimated_motion_done_at) + est_s
        return est_s

    def _wait_for_estimated_motion_complete(self, extra_settle: float = 0.0, reason: str = "motion"):
        wait_s = max(0.0, self.estimated_motion_done_at - time.monotonic()) + max(0.0, float(extra_settle))
        if wait_s > 0:
            print(f" Estimated wait for {reason}: {wait_s:.3f} s")
            time.sleep(wait_s)
        self.estimated_motion_done_at = max(self.estimated_motion_done_at, time.monotonic())

    def send_absolute_move(self, feedrate: float, **axes_targets):
        if self.rrf is None:
            raise RuntimeError("Robot is not connected.")

        previous_axes = dict(self.commanded_axes)
        parts = ["G90", "G1"]
        for ax, val in axes_targets.items():
            if val is None:
                continue
            parts.append(f"{ax}{float(val):.3f}")
        parts.append(f"F{float(feedrate):.3f}")
        gcode = " ".join(parts)
        print(f" Command: {gcode}")
        for ax, val in axes_targets.items():
            if val is None:
                continue
            self.commanded_axes[str(ax)] = float(val)
        command_record = {
            "command_index": int(len(self.command_log) + 1),
            "feedrate": float(feedrate),
            "gcode": gcode,
            "previous_axes": previous_axes,
            "axes_targets": {
                str(ax): float(val)
                for ax, val in axes_targets.items()
                if val is not None
            },
            "resolved_axes": dict(self.commanded_axes),
        }
        self.command_log.append(command_record)
        self.rrf.send_code(gcode)
        return command_record

    def write_command_log_csv(self, cal: Calibration) -> str:
        csv_path = os.path.join(self.run_folder, "commanded_motor_positions.csv")
        fieldnames = [
            "command_index",
            "feedrate",
            "gcode",
            "x_axis_name",
            "y_axis_name",
            "z_axis_name",
            "b_axis_name",
            "c_axis_name",
            "x_cmd",
            "y_cmd",
            "z_cmd",
            "b_cmd",
            "c_cmd",
        ]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.command_log:
                resolved = row.get("resolved_axes", {})
                writer.writerow(
                    {
                        "command_index": row.get("command_index"),
                        "feedrate": row.get("feedrate"),
                        "gcode": row.get("gcode"),
                        "x_axis_name": cal.x_axis,
                        "y_axis_name": cal.y_axis,
                        "z_axis_name": cal.z_axis,
                        "b_axis_name": cal.b_axis,
                        "c_axis_name": cal.c_axis,
                        "x_cmd": resolved.get(cal.x_axis),
                        "y_cmd": resolved.get(cal.y_axis),
                        "z_cmd": resolved.get(cal.z_axis),
                        "b_cmd": resolved.get(cal.b_axis),
                        "c_cmd": resolved.get(cal.c_axis),
                    }
                )
        return csv_path

    def _move_to_pose_safe(
        self,
        cal: Calibration,
        pose: Tuple[float, float, float, float, float],
        safe_approach_z: float,
        travel_feed: float,
        settle_s: float,
        use_estimated_waits: bool = False,
        b_max_feed: float = DEFAULT_B_MAX_FEED,
    ):
        x, y, z, b, c = [float(v) for v in pose]

        cmd = self.send_absolute_move(
            travel_feed,
            **{
                cal.z_axis: float(safe_approach_z),
                cal.b_axis: b,
                cal.c_axis: clamp_c_bounded(c),
            }
        )
        if use_estimated_waits:
            self._record_estimated_motion(cal=cal, command_record=cmd, b_max_feed=b_max_feed)
            self._wait_for_estimated_motion_complete(extra_settle=settle_s, reason="safe Z approach")
        else:
            self.wait_for_duet_motion_complete(extra_settle=settle_s)

        cmd = self.send_absolute_move(
            travel_feed,
            **{
                cal.x_axis: x,
                cal.y_axis: y,
                cal.b_axis: b,
                cal.c_axis: clamp_c_bounded(c),
            }
        )
        if use_estimated_waits:
            self._record_estimated_motion(cal=cal, command_record=cmd, b_max_feed=b_max_feed)
            self._wait_for_estimated_motion_complete(extra_settle=settle_s, reason="safe XY approach")
        else:
            self.wait_for_duet_motion_complete(extra_settle=settle_s)

        cmd = self.send_absolute_move(
            travel_feed,
            **{
                cal.z_axis: z,
                cal.b_axis: b,
                cal.c_axis: clamp_c_bounded(c),
            }
        )
        if use_estimated_waits:
            self._record_estimated_motion(cal=cal, command_record=cmd, b_max_feed=b_max_feed)
            self._wait_for_estimated_motion_complete(extra_settle=settle_s, reason="safe Z landing")
        else:
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
        use_estimated_waits: bool = False,
        b_max_feed: float = DEFAULT_B_MAX_FEED,
    ):
        print(" Fine landing move for accuracy...")
        cmd = self.send_absolute_move(
            fine_feed,
            **{
                cal.x_axis: x,
                cal.y_axis: y,
                cal.z_axis: z,
                cal.b_axis: b,
                cal.c_axis: clamp_c_bounded(c),
            }
        )
        if use_estimated_waits:
            self._record_estimated_motion(cal=cal, command_record=cmd, b_max_feed=b_max_feed)
            self._wait_for_estimated_motion_complete(extra_settle=settle_s, reason="fine landing")
        else:
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
        b_extra_settle_s: float = 0.0,
        inter_command_delay_s: float = 0.0,
        camera_flush_frames: int = 1,
        capture_at_start: bool = True,
        initial_sweep_wait_s: float = DEFAULT_INITIAL_SWEEP_WAIT_S,
        c_flip_feed: float = DEFAULT_C_FLIP_FEED,
        settled_capture_mode: bool = DEFAULT_SETTLED_CAPTURE_MODE,
        settled_capture_buffer_s: float = DEFAULT_SETTLED_CAPTURE_BUFFER_S,
    ):
        if self.cam is None:
            raise RuntimeError("Camera is not connected.")
        if self.rrf is None:
            raise RuntimeError("Robot is not connected.")

        bbox_warnings: List[str] = []
        sample_counter = 0
        self.command_log = []
        self.commanded_axes = {}
        self._seed_commanded_pose(cal=cal, pose=start_pose)

        blocks = split_trajectory_into_blocks(traj)

        print("\n" + "=" * 72)
        print("STARTING TRACKED POINT-ACQUISITION RUN")
        print("=" * 72)
        print(f"Tracked samples total: {len(traj)}")
        print(f"Orientation blocks: {len(blocks)}")
        if settled_capture_mode:
            print(
                "Settled capture mode: using estimated move time plus "
                f"{float(settled_capture_buffer_s):.3f} s buffer; M400 waits are skipped."
            )

        print("\nSafe startup approach...")
        self._move_to_pose_safe(
            cal=cal,
            pose=start_pose,
            safe_approach_z=float(safe_approach_z),
            travel_feed=float(travel_feed),
            settle_s=float(travel_move_settle_s),
            use_estimated_waits=bool(settled_capture_mode),
            b_max_feed=float(b_max_feed),
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

            if use_segment_feed_scheduler:
                print(
                    "Segment feed scheduling disabled for execution; "
                    "using controller-side acceleration with constant queued feed."
                )
            print(f"Tracked block feed: {float(probe_feed):.3f}")

            x0, y0, z0 = _clamp_stage_xyz_to_bbox(
                first_pt.stage_xyz[0], first_pt.stage_xyz[1], first_pt.stage_xyz[2],
                virtual_bbox,
                f"{block_name} first point",
                bbox_warnings,
            )

            transition_feed = float(travel_feed)
            c_flip_delay_s = 0.0
            if block_idx > 0 and math.isclose(float(first_pt.c), float(cal.c_180_deg), abs_tol=1e-6):
                transition_feed = float(c_flip_feed)
                c_flip_delay_s = float(DEFAULT_C_FLIP_DELAY_S)
                print(
                    f"Tracked transition into {block_name}: "
                    f"moving XYZ/B/C together with C turn at feed {transition_feed:.3f}..."
                )
            else:
                print("Large move onto first tracked point of block...")

            cmd = self.send_absolute_move(
                transition_feed,
                **{
                    cal.x_axis: x0,
                    cal.y_axis: y0,
                    cal.z_axis: z0,
                    cal.b_axis: float(first_pt.b),
                    cal.c_axis: clamp_c_bounded(float(first_pt.c)),
                }
            )
            if settled_capture_mode:
                self._record_estimated_motion(cal=cal, command_record=cmd, b_max_feed=float(b_max_feed))
                self._wait_for_estimated_motion_complete(
                    extra_settle=float(travel_move_settle_s),
                    reason=f"{block_name} transition",
                )
            else:
                self.wait_for_duet_motion_complete(extra_settle=float(travel_move_settle_s))
            if c_flip_delay_s > 0:
                print(f"Holding {c_flip_delay_s:.3f} s after C rotation...")
                time.sleep(c_flip_delay_s)

            self._fine_land_on_point(
                cal=cal,
                x=x0,
                y=y0,
                z=z0,
                b=float(first_pt.b),
                c=float(first_pt.c),
                fine_feed=float(fine_approach_feed),
                settle_s=max(float(tracked_move_settle_s), 0.05),
                use_estimated_waits=bool(settled_capture_mode),
                b_max_feed=float(b_max_feed),
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
                if settled_capture_mode:
                    self._wait_for_estimated_motion_complete(
                        extra_settle=float(settled_capture_buffer_s),
                        reason=f"{block_name} start capture",
                    )
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
                    motion_phase=first_pt.motion_phase,
                )
            else:
                print("Start point not captured.")

            print("Executing pull-tracked / release-untracked oscillation block...")
            for i, point in enumerate(block[1:], start=1):
                prev_point = block[i - 1]
                if point.motion_phase == "pull" and prev_point.motion_phase == "release":
                    if settled_capture_mode:
                        self._wait_for_estimated_motion_complete(
                            extra_settle=float(tracked_move_settle_s),
                            reason="release-to-pull boundary",
                        )
                    else:
                        self.wait_for_duet_motion_complete(extra_settle=float(tracked_move_settle_s))
                    if float(b_extra_settle_s) > 0:
                        time.sleep(float(b_extra_settle_s))
                    print(f"Holding {float(DEFAULT_PULL_SEQUENCE_BUFFER_S):.3f} s before next pull sequence...")
                    time.sleep(float(DEFAULT_PULL_SEQUENCE_BUFFER_S))

                x, y, z = _clamp_stage_xyz_to_bbox(
                    point.stage_xyz[0], point.stage_xyz[1], point.stage_xyz[2],
                    virtual_bbox,
                    f"{block_name} sample {i}",
                    bbox_warnings,
                )

                cmd = self.send_absolute_move(
                    float(probe_feed),
                    **{
                        cal.x_axis: x,
                        cal.y_axis: y,
                        cal.z_axis: z,
                        cal.b_axis: float(point.b),
                        cal.c_axis: clamp_c_bounded(float(point.c)),
                    }
                )
                if settled_capture_mode:
                    est_s = self._record_estimated_motion(
                        cal=cal,
                        command_record=cmd,
                        b_max_feed=float(b_max_feed),
                    )
                    if point.capture_image:
                        print(f" Estimated segment time before capture: {est_s:.3f} s")
                if float(inter_command_delay_s) > 0:
                    time.sleep(float(inter_command_delay_s))
                if point.capture_image:
                    if settled_capture_mode:
                        self._wait_for_estimated_motion_complete(
                            extra_settle=float(settled_capture_buffer_s),
                            reason=f"{block_name} sample {i} capture",
                        )
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
                        motion_phase=point.motion_phase,
                    )

            if len(block) > 1:
                if settled_capture_mode:
                    self._wait_for_estimated_motion_complete(
                        extra_settle=float(tracked_move_settle_s),
                        reason=f"{block_name} block completion",
                    )
                else:
                    self.wait_for_duet_motion_complete(extra_settle=float(tracked_move_settle_s))
                if float(b_extra_settle_s) > 0:
                    time.sleep(float(b_extra_settle_s))

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
            use_estimated_waits=bool(settled_capture_mode),
            b_max_feed=float(b_max_feed),
        )

        print("\n" + "=" * 72)
        print("RUN COMPLETE")
        print("=" * 72)
        print(f"Images saved: {sample_counter}")
        print(f"Point-tracking folder: {self.point_tracking_folder}")
        print(f"BBox warnings: {len(bbox_warnings)}")
        for msg in bbox_warnings:
            print(msg)

        command_csv_path = self.write_command_log_csv(cal)
        print(f"Commanded motor positions CSV: {command_csv_path}")

        return {
            "images_saved": sample_counter,
            "bbox_warnings": bbox_warnings,
            "command_csv_path": command_csv_path,
        }


# =========================
# Main
# =========================

def run_acquisition_legacy(args):
    script_dir = Path(__file__).resolve().parent
    cal = load_calibration(args.calibration, y_offset_fit=args.y_offset_fit)

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
            b_extra_settle_s=float(args.b_extra_settle_s),
            inter_command_delay_s=float(args.inter_command_delay_s),
            camera_flush_frames=int(args.camera_flush_frames),
            capture_at_start=bool(args.capture_at_start),
            initial_sweep_wait_s=float(args.initial_sweep_wait_s),
            c_flip_feed=float(args.c_flip_feed),
            settled_capture_mode=bool(args.settled_capture_mode),
            settled_capture_buffer_s=float(args.settled_capture_buffer_s),
        )

        print("\nFinal results:")
        print(results)

        if bool(args.enable_post):
            post_tip_refiner_model = Path(args.post_tip_refiner_model).expanduser()
            if not post_tip_refiner_model.is_absolute():
                post_tip_refiner_model = script_dir / post_tip_refiner_model
            post_cmd = [
                sys.executable,
                str(script_dir / "calib_plane_process.py"),
                "--project_dir",
                runner.run_folder,
                "--camera_calibration_file",
                str(script_dir / "captures/calibration_webcam_20260406_104136.npz"),
                "--checkerboard_reference_image",
                str(script_dir / "captures/photo_20260428_162904.png"),
                "--threshold",
                "200",
                "--tip_detection_mode",
                str(args.post_tip_detection_mode),
                "--tip_refine_mode",
                "none",
                "--tip_refiner_model",
                str(post_tip_refiner_model.resolve()),
                "--save_plots",
            ]
            print("\nRunning post-processing:")
            print(" ".join(post_cmd))
            subprocess.run(post_cmd, check=True, cwd=str(script_dir))

    finally:
        try:
            runner.disconnect_camera()
        except Exception:
            pass




# =============================================================================
# Embedded checkerboard processing script
# =============================================================================

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
offline_fixed_tip_dual_c_checkerboard_error_analysis.py

Offline checkerboard-referenced tip error analysis tailored to the fixed-tip
dual-C acquisition script.

What it does
------------
1) Opens a project/raw image folder from the fixed-tip dual-C acquisition run
2) Lets you choose crop bounds from the first raw image
3) Uses a checkerboard reference image + camera calibration to define mm axes
4) Runs analyze_data_batch + existing tracking pipeline
5) Converts tracked tip pixels to checkerboard-referenced mm coordinates
6) Parses C orientation / tip angle / block metadata from image filenames
7) Builds per-sample reference points from the predicted Cartesian targets
   encoded in the filenames and aligns that target set into checkerboard mm
   coordinates
8) Computes:
      - global RMSE to the corresponding per-sample reference points
      - RMSE for each C orientation angle to those same per-sample references
9) Saves:
      - CSV with tracked points, filename metadata, errors
      - JSON metrics summary including per-C RMSE
      - error-vs-sample plot
      - dark-theme histogram + side-by-side per-C density heatmaps
      - optional checkerboard overlay

Expected filename format
------------------------
The fixed-tip acquisition script saves images like:

00001_tracked_block_X..._Y..._Z..._B..._C180.000_C180_TIP90.000_BPH0.50000_2026....png

This script parses:
  - stage X/Y/Z
  - B
  - C
  - block name like C0 / C180
  - TIP angle
  - BPH block phase

Usage example
-------------
python3 offline_fixed_tip_dual_c_checkerboard_error_analysis.py \
    --project_dir "/path/to/Point_Tracking_Run_2026-03-20_12-34-56" \
    --camera_calibration_file "/path/to/camera_calibration.npz" \
    --checkerboard_reference_image "/path/to/checkerboard_ref.png" \
    --threshold 200 \
    --save_plots \
    --tip_refine_mode edge_dt

Notes
-----
- This expects shadow_calibration.py to be importable from the same folder or PYTHONPATH.
- When per-sample predicted Cartesian targets are available from filename
  metadata, each measured point is compared against its corresponding target
  after centroid alignment into checkerboard mm coordinates.
- If those predicted targets are unavailable, the script falls back to the
  global measured centroid.
"""

import argparse
import csv
import json
import math
import os
import re
import shutil
import sys
import types
from heapq import heappop, heappush
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.collections import PathCollection
from matplotlib.gridspec import GridSpec

# -----------------------------------------------------------------------------
# Import existing shadow calibration pipeline
# -----------------------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.insert(0, SCRIPT_DIR)

try:
    import shadow_calibration as shadow_calibration_module  # noqa: E402
    from shadow_calibration import CTR_Shadow_Calibration  # noqa: E402
except Exception:
    shadow_calibration_module = None
    CTR_Shadow_Calibration = None

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_NEI8_W = [
    (-1, -1, 2 ** 0.5), (-1, 0, 1.0), (-1, 1, 2 ** 0.5),
    (0, -1, 1.0),                             (0, 1, 1.0),
    (1, -1, 2 ** 0.5),  (1, 0, 1.0),  (1, 1, 2 ** 0.5),
]
DEFAULT_CHARUCO_BOARD = {
    "squares_x": 10,
    "squares_y": 14,
    "square_size_mm": 15.0,
    "marker_size_mm": 11.0,
    "aruco_dictionary": "DICT_4X4",
}


# =============================================================================
# Utilities: IO and discovery
# =============================================================================
def list_images(folder: Path) -> List[Path]:
    imgs = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    imgs.sort()
    return imgs


def ensure_project_from_raw(raw_dir: Path, project_dir: Path, link_mode: str = "symlink") -> Path:
    """
    Create a project_dir with raw_image_data_folder containing images from raw_dir.
    Returns raw_image_data_folder path.
    """
    raw_dir = Path(raw_dir).expanduser().resolve()
    project_dir = Path(project_dir).expanduser().resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    raw_out = project_dir / "raw_image_data_folder"
    raw_out.mkdir(parents=True, exist_ok=True)

    imgs = list_images(raw_dir)
    if not imgs:
        raise RuntimeError(f"No images found in raw_dir: {raw_dir}")

    for src in imgs:
        dst = raw_out / src.relative_to(raw_dir)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if link_mode == "symlink":
            try:
                os.symlink(src.resolve(), dst)
            except Exception:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)

    return raw_out


def _json_ready(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def load_optional_tip_refiner(cal: CTR_Shadow_Calibration, args) -> None:
    if not getattr(args, "tip_refiner_model", None):
        return
    model_path = Path(args.tip_refiner_model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Tip refiner model not found: {model_path}")
    cal.load_tip_refiner_model(
        str(model_path),
        anchor_name=args.tip_refiner_anchor,
        use_as_selected=(not bool(args.tip_refiner_compare_only)),
    )


def select_tracked_rows_for_analysis(cal: CTR_Shadow_Calibration, args) -> Tuple[np.ndarray, str]:
    source = str(getattr(args, "tracked_tip_source", "auto")).strip().lower()
    if source == "auto":
        tip_detection_mode = str(getattr(args, "tip_detection_mode", "classical")).strip().lower()
        if tip_detection_mode in ("red", "red_dot", "red_marker", "red_centroid", "marker", "auto_red", "auto_red_dot", "red_dot_auto"):
            source = "selected"
        elif getattr(args, "tip_refiner_model", None) and not bool(getattr(args, "tip_refiner_compare_only", False)):
            source = "selected"
        else:
            source = "coarse"

    if source == "cnn":
        arr = getattr(cal, "tip_locations_array_cnn", None)
        if arr is None:
            raise RuntimeError("tracked_tip_source=cnn requested, but tip_locations_array_cnn is unavailable.")
        return np.asarray(arr, dtype=float), "cnn"

    if source == "selected":
        arr = getattr(cal, "tip_locations_array_selected", None)
        if arr is None:
            raise RuntimeError("tracked_tip_source=selected requested, but tip_locations_array_selected is unavailable.")
        return np.asarray(arr, dtype=float), "selected"

    if source == "coarse":
        arr = getattr(cal, "tip_locations_array_coarse", None)
        if arr is None:
            raise RuntimeError("tracked_tip_source=coarse requested, but tip_locations_array_coarse is unavailable.")
        return np.asarray(arr, dtype=float), "coarse"

    raise ValueError(f"Unsupported tracked_tip_source: {source}")


def _board_reference_kwargs(cal: CTR_Shadow_Calibration, args) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {
        "inner_corners": args.checkerboard_inner_corners,
        "square_size_mm": args.checkerboard_square_size_mm,
        "use_undistort": (not args.checkerboard_no_undistort),
        "draw_debug": True,
    }
    meta = getattr(cal, "camera_calib_meta", None) or {}
    board_type = str(meta.get("board_type", "checkerboard")).strip().lower()
    if board_type == "charuco":
        kwargs.update({
            "squares_x": int(meta.get("squares_x", DEFAULT_CHARUCO_BOARD["squares_x"])),
            "squares_y": int(meta.get("squares_y", DEFAULT_CHARUCO_BOARD["squares_y"])),
            "marker_size_mm": float(meta.get("marker_size_mm", DEFAULT_CHARUCO_BOARD["marker_size_mm"])),
            "aruco_dictionary": str(meta.get("aruco_dictionary", DEFAULT_CHARUCO_BOARD["aruco_dictionary"])),
        })
        if kwargs["square_size_mm"] is None:
            kwargs["square_size_mm"] = float(meta.get("square_size_mm", DEFAULT_CHARUCO_BOARD["square_size_mm"]))
    return kwargs


def _parse_inner_corners_arg(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])

    txt = str(value).strip().lower().replace("x", ",")
    parts = [p.strip() for p in txt.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected checkerboard inner corners as 'Nx,Ny' or 'NxXNy', got: {value}"
        )
    return int(parts[0]), int(parts[1])


def _undistort_points_to_image(points_xy: np.ndarray, camera_K: np.ndarray, camera_dist: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    undist = cv2.undistortPoints(pts, camera_K, camera_dist, P=camera_K)
    return undist.reshape(-1, 2)


def _scale_mm_from_px_homography(H: np.ndarray, mm_scale: float) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64).copy()
    S = np.diag([float(mm_scale), float(mm_scale), 1.0]).astype(np.float64)
    return S @ H


def _scale_px_from_mm_homography(H: np.ndarray, mm_scale: float) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64).copy()
    if abs(float(mm_scale)) < 1e-12:
        raise ValueError("mm_scale must be non-zero.")
    S_inv = np.diag([1.0 / float(mm_scale), 1.0 / float(mm_scale), 1.0]).astype(np.float64)
    return H @ S_inv


def apply_checkerboard_reference_corrections(
    cal: CTR_Shadow_Calibration,
    mm_scale: float = 0.5,
    flip_planar_x: bool = True,
):
    """
    Correct checkerboard-backed calibration axes so they match the ruler-backed convention.

    Requested fixes:
      1) checkerboard mm values are 2x too large -> scale checkerboard mm conversion by 0.5
      2) checkerboard planar x sign is flipped vs motor -> negate planar x
    """
    if hasattr(cal, "apply_checkerboard_reference_corrections"):
        return cal.apply_checkerboard_reference_corrections(
            mm_scale=mm_scale,
            flip_planar_x=flip_planar_x,
        )

    if getattr(cal, "board_pose", None) is None:
        return cal

    correction_meta = {
        "checkerboard_mm_scale_correction": float(mm_scale),
        "checkerboard_planar_x_flipped": bool(flip_planar_x),
    }

    if getattr(cal, "board_mm_per_px_local", None) is not None:
        cal.board_mm_per_px_local = float(cal.board_mm_per_px_local) * float(mm_scale)
    if getattr(cal, "board_px_per_mm_local", None) is not None:
        if abs(float(mm_scale)) < 1e-12:
            raise ValueError("mm_scale must be non-zero.")
        cal.board_px_per_mm_local = float(cal.board_px_per_mm_local) / float(mm_scale)

    if getattr(cal, "board_homography_mm_from_px", None) is not None:
        cal.board_homography_mm_from_px = _scale_mm_from_px_homography(
            cal.board_homography_mm_from_px,
            mm_scale=float(mm_scale),
        )
    if getattr(cal, "board_homography_px_from_mm", None) is not None:
        cal.board_homography_px_from_mm = _scale_px_from_mm_homography(
            cal.board_homography_px_from_mm,
            mm_scale=float(mm_scale),
        )

    if isinstance(cal.board_pose, dict):
        bp = dict(cal.board_pose)
        if "board_homography_mm_from_px" in bp and bp["board_homography_mm_from_px"] is not None:
            bp["board_homography_mm_from_px"] = _scale_mm_from_px_homography(
                bp["board_homography_mm_from_px"], mm_scale=float(mm_scale)
            )
        if "board_homography_px_from_mm" in bp and bp["board_homography_px_from_mm"] is not None:
            bp["board_homography_px_from_mm"] = _scale_px_from_mm_homography(
                bp["board_homography_px_from_mm"], mm_scale=float(mm_scale)
            )
        if "board_mm_per_px_local" in bp and bp["board_mm_per_px_local"] is not None:
            bp["board_mm_per_px_local"] = float(bp["board_mm_per_px_local"]) * float(mm_scale)
        if "board_px_per_mm_local" in bp and bp["board_px_per_mm_local"] is not None:
            bp["board_px_per_mm_local"] = float(bp["board_px_per_mm_local"]) / float(mm_scale)
        bp.setdefault("corrections_applied", {})
        bp["corrections_applied"].update(correction_meta)
        cal.board_pose = bp

    cal.board_reference_correction_meta = correction_meta

    if hasattr(cal, "pixel_point_to_calibrated_axes"):
        original_pixel_point_to_calibrated_axes = cal.pixel_point_to_calibrated_axes

        def pixel_point_to_calibrated_axes_patched(self, *args, **kwargs):
            u_mm, z_mm = original_pixel_point_to_calibrated_axes(*args, **kwargs)
            u_mm = float(u_mm)
            z_mm = float(z_mm)
            if bool(flip_planar_x):
                u_mm = -u_mm
            return u_mm, z_mm

        cal.pixel_point_to_calibrated_axes = types.MethodType(
            pixel_point_to_calibrated_axes_patched, cal
        )

    if hasattr(cal, "calibrated_axes_to_pixel_point"):
        original_calibrated_axes_to_pixel_point = cal.calibrated_axes_to_pixel_point

        def calibrated_axes_to_pixel_point_patched(self, u_mm, z_mm, *args, **kwargs):
            u_in = -float(u_mm) if bool(flip_planar_x) else float(u_mm)
            return original_calibrated_axes_to_pixel_point(u_in, float(z_mm), *args, **kwargs)

        cal.calibrated_axes_to_pixel_point = types.MethodType(
            calibrated_axes_to_pixel_point_patched, cal
        )

    print(
        "[INFO] Applied checkerboard reference corrections: "
        f"mm_scale={float(mm_scale):.6f}, "
        f"flip_planar_x={bool(flip_planar_x)}"
    )
    return cal


def collect_board_reference_info(cal: CTR_Shadow_Calibration) -> Dict[str, Any]:
    return {
        "camera_calib_path": getattr(cal, "camera_calib_path", None),
        "camera_calib_meta": getattr(cal, "camera_calib_meta", None),
        "board_reference_image_path": getattr(cal, "board_reference_image_path", None),
        "board_pose": getattr(cal, "board_pose", None),
        "true_vertical_img_unit": getattr(cal, "true_vertical_img_unit", None),
        "board_homography_px_from_mm": getattr(cal, "board_homography_px_from_mm", None),
        "board_homography_mm_from_px": getattr(cal, "board_homography_mm_from_px", None),
        "board_px_per_mm_local": getattr(cal, "board_px_per_mm_local", None),
        "board_mm_per_px_local": getattr(cal, "board_mm_per_px_local", None),
        "board_reference_correction_meta": getattr(cal, "board_reference_correction_meta", None),
    }


# =============================================================================
# GUI: crop selection from one image
# =============================================================================
def interactive_crop_from_image(
    image_bgr: np.ndarray,
    default_crop=None,
    window_crop="Manual Crop Setup (OFFLINE)",
):
    """
    Returns analysis_crop dict in the format CTR_Shadow_Calibration expects.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image passed to interactive GUI.")

    img_h, img_w = image_bgr.shape[:2]

    if default_crop is None:
        default_crop = {
            "crop_width_min": 650,
            "crop_width_max": 2900,
            "crop_height_min": 150,
            "crop_height_max": 1750,
        }

    x_min = int(np.clip(default_crop["crop_width_min"], 0, img_w - 2))
    x_max = int(np.clip(default_crop["crop_width_max"], x_min + 1, img_w - 1))
    y_min = int(np.clip(img_h - default_crop["crop_height_max"], 0, img_h - 2))
    y_max = int(np.clip(img_h - default_crop["crop_height_min"], y_min + 1, img_h - 1))

    corners = {
        "tl": [x_min, y_min],
        "tr": [x_max, y_min],
        "br": [x_max, y_max],
        "bl": [x_min, y_max],
    }
    active_corner = {"name": None}
    drag_threshold_px = 30

    def nearest_corner(mx, my):
        best_name, best_dist = None, 1e18
        for name, (cx, cy) in corners.items():
            d = (mx - cx) ** 2 + (my - cy) ** 2
            if d < best_dist:
                best_dist = d
                best_name = name
        return best_name if best_dist <= drag_threshold_px**2 else None

    def clamp_rect():
        xs = [pt[0] for pt in corners.values()]
        ys = [pt[1] for pt in corners.values()]
        x0 = int(np.clip(min(xs), 0, img_w - 2))
        x1 = int(np.clip(max(xs), x0 + 1, img_w - 1))
        y0 = int(np.clip(min(ys), 0, img_h - 2))
        y1 = int(np.clip(max(ys), y0 + 1, img_h - 1))
        corners["tl"] = [x0, y0]
        corners["tr"] = [x1, y0]
        corners["br"] = [x1, y1]
        corners["bl"] = [x0, y1]

    def on_mouse_crop(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            active_corner["name"] = nearest_corner(mx, my)
        elif event == cv2.EVENT_MOUSEMOVE and active_corner["name"] is not None:
            name = active_corner["name"]
            mx = int(np.clip(mx, 0, img_w - 1))
            my = int(np.clip(my, 0, img_h - 1))
            if name == "tl":
                corners["tl"] = [mx, my]
                corners["tr"][1] = my
                corners["bl"][0] = mx
            elif name == "tr":
                corners["tr"] = [mx, my]
                corners["tl"][1] = my
                corners["br"][0] = mx
            elif name == "br":
                corners["br"] = [mx, my]
                corners["bl"][1] = my
                corners["tr"][0] = mx
            elif name == "bl":
                corners["bl"] = [mx, my]
                corners["br"][1] = my
                corners["tl"][0] = mx
            clamp_rect()
        elif event == cv2.EVENT_LBUTTONUP:
            active_corner["name"] = None

    accepted_crop = False
    cv2.namedWindow(window_crop, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_crop, on_mouse_crop)

    print("\n[GUI] Manual crop setup (OFFLINE)")
    print("- Drag rectangle corners with left mouse.")
    print("- Press ENTER or SPACE to confirm.")
    print("- Press R to reset to defaults.")
    print("- Press Q or ESC to cancel (uses defaults).")

    try:
        while True:
            if cv2.getWindowProperty(window_crop, cv2.WND_PROP_VISIBLE) < 1:
                break

            clamp_rect()
            x0, y0 = corners["tl"]
            x1, y1 = corners["br"]

            disp = image_bgr.copy()
            cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 2)
            for pt in corners.values():
                cv2.circle(disp, tuple(pt), 6, (0, 255, 255), -1)

            cv2.putText(
                disp,
                f"x:[{x0},{x1}] y:[{y0},{y1}]  (ENTER confirm | R reset | Q/ESC cancel)",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

            cv2.imshow(window_crop, disp)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 32):
                accepted_crop = True
                break
            if key in (27, ord("q")):
                break
            if key in (ord("r"), ord("R")):
                corners["tl"] = [x_min, y_min]
                corners["tr"] = [x_max, y_min]
                corners["br"] = [x_max, y_max]
                corners["bl"] = [x_min, y_max]
    finally:
        cv2.setMouseCallback(window_crop, lambda *args: None)
        cv2.destroyWindow(window_crop)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    if accepted_crop:
        x0, y0 = corners["tl"]
        x1, y1 = corners["br"]
        analysis_crop = {
            "crop_width_min": int(x0),
            "crop_width_max": int(x1),
            "crop_height_min": int(img_h - y1),
            "crop_height_max": int(img_h - y0),
        }
        print(f"[GUI] Selected analysis_crop: {analysis_crop}")
    else:
        analysis_crop = dict(default_crop)
        print(f"[GUI] Crop cancelled, using default analysis_crop: {analysis_crop}")

    return analysis_crop


# =============================================================================
# Tip refinement helpers
# =============================================================================
def _tip_angle_to_direction_xy(tip_angle_deg: float) -> np.ndarray:
    ang = np.deg2rad(float(tip_angle_deg))
    vx = float(np.sin(ang))
    vy = float(np.cos(ang))
    d = np.array([vx, vy], dtype=np.float64)
    n = float(np.linalg.norm(d))
    return d / max(n, 1e-12)


def _inside_mask(mask_fg: np.ndarray, xy: np.ndarray) -> bool:
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    h, w = mask_fg.shape[:2]
    return (0 <= x < w) and (0 <= y < h) and (mask_fg[y, x] > 0)


def _backtrack_point_inside_fg(mask_fg: np.ndarray, p0_xy: np.ndarray, dir_xy: np.ndarray,
                               max_back_px: float = 80.0, step_px: float = 0.5):
    p0 = np.asarray(p0_xy, dtype=np.float64)
    d = np.asarray(dir_xy, dtype=np.float64)
    d /= max(np.linalg.norm(d), 1e-12)

    if _inside_mask(mask_fg, p0):
        return p0.copy(), True, 0.0

    n_steps = int(max_back_px / max(step_px, 1e-6))
    for i in range(1, n_steps + 1):
        q = p0 - i * step_px * d
        if _inside_mask(mask_fg, q):
            return q, True, i * step_px
    return p0.copy(), False, None


def ray_last_inside(mask_fg: np.ndarray, p0_xy: np.ndarray, dir_xy: np.ndarray,
                    step_px: float = 0.5, max_len_px: float = 120.0):
    h, w = mask_fg.shape[:2]
    d = np.asarray(dir_xy, dtype=np.float64)
    d /= max(np.linalg.norm(d), 1e-12)

    last_inside = np.asarray(p0_xy, dtype=np.float64).copy()
    n_steps = int(max_len_px / max(step_px, 1e-6))

    for i in range(1, n_steps + 1):
        p = p0_xy + i * step_px * d
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        if not (0 <= x < w and 0 <= y < h):
            break
        if mask_fg[y, x] == 1:
            last_inside = p
        else:
            break
    return last_inside


def _estimate_radius_along_axis(mask_fg: np.ndarray, dist_img: np.ndarray, p_in_xy: np.ndarray,
                                dir_xy: np.ndarray, back_len_px: float = 60.0, step_px: float = 1.0) -> float:
    d = np.asarray(dir_xy, dtype=np.float64)
    d /= max(np.linalg.norm(d), 1e-12)
    vals = []
    n_steps = int(back_len_px / max(step_px, 1e-6))
    for i in range(n_steps + 1):
        p = p_in_xy - i * step_px * d
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        if 0 <= x < mask_fg.shape[1] and 0 <= y < mask_fg.shape[0] and mask_fg[y, x] == 1:
            v = float(dist_img[y, x])
            if np.isfinite(v) and v > 0:
                vals.append(v)
    if not vals:
        x = int(round(float(p_in_xy[0])))
        y = int(round(float(p_in_xy[1])))
        if 0 <= x < dist_img.shape[1] and 0 <= y < dist_img.shape[0]:
            return max(3.0, float(dist_img[y, x]))
        return 6.0
    return max(3.0, float(np.median(vals)))


def _contiguous_runs_from_bool(mask_bool: np.ndarray):
    m = np.asarray(mask_bool, dtype=bool)
    edges = np.diff(np.r_[False, m, False].astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _cross_section_boundaries(mask_fg: np.ndarray, center_xy: np.ndarray, normal_xy: np.ndarray,
                              scan_half_width_px: float = 20.0, step_px: float = 0.5):
    n = np.asarray(normal_xy, dtype=np.float64)
    n /= max(np.linalg.norm(n), 1e-12)
    ts = np.arange(-scan_half_width_px, scan_half_width_px + 0.5 * step_px, step_px, dtype=np.float64)
    inside = np.zeros_like(ts, dtype=bool)

    h, w = mask_fg.shape[:2]
    for i, t in enumerate(ts):
        p = center_xy + t * n
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        inside[i] = (0 <= x < w and 0 <= y < h and mask_fg[y, x] == 1)

    if not np.any(inside):
        return None

    runs = _contiguous_runs_from_bool(inside)
    if not runs:
        return None

    idx0 = int(np.argmin(np.abs(ts)))
    chosen = None
    for s0, s1 in runs:
        if s0 <= idx0 <= s1:
            chosen = (s0, s1)
            break

    if chosen is None:
        centers = [0.5 * (ts[s0] + ts[s1]) for s0, s1 in runs]
        j = int(np.argmin(np.abs(np.asarray(centers))))
        chosen = runs[j]

    s0, s1 = chosen
    t_left = float(ts[s0])
    t_right = float(ts[s1])
    p_left = center_xy + t_left * n
    p_right = center_xy + t_right * n

    return {
        "t_left": t_left,
        "t_right": t_right,
        "p_left_xy": p_left,
        "p_right_xy": p_right,
        "ts": ts,
        "inside": inside,
    }


def refine_tip_edge_distance_transform(
    binary_image: np.ndarray,
    tip_yx: Tuple[int, int],
    tip_angle_deg: float,
    max_step_px: int = 80,
    step_px: float = 1.0,
    exit_mode: str = "first_background",
):
    h, w = binary_image.shape[:2]
    fg = (binary_image == 0).astype(np.uint8)

    d_xy = _tip_angle_to_direction_xy(tip_angle_deg)
    x0, y0 = float(tip_yx[1]), float(tip_yx[0])

    last_inside = np.array([x0, y0], dtype=np.float64)
    exited = False
    n_steps = int(max(1, max_step_px / max(step_px, 1e-6)))

    for i in range(1, n_steps + 1):
        p = np.array([x0, y0], dtype=np.float64) + i * step_px * d_xy
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        if not (0 <= x < w and 0 <= y < h):
            exited = True
            break
        if fg[y, x] == 1:
            last_inside = p
            continue
        exited = True
        if exit_mode == "first_background":
            break

    dbg = {
        "mode": "edge_dt",
        "d_xy": d_xy.tolist(),
        "exited": exited,
        "last_inside_xy": last_inside.tolist(),
        "tip_xy": last_inside.tolist(),
        "center_line_xy": [np.array([x0, y0]).tolist(), last_inside.tolist()],
    }
    return float(last_inside[1]), float(last_inside[0]), dbg


def refine_tip_edge_subpixel_gradient(
    grayscale: np.ndarray,
    binary_image: np.ndarray,
    tip_yx: Tuple[int, int],
    tip_angle_deg: float,
    search_len_px: int = 60,
    step_px: float = 0.25,
):
    h, w = grayscale.shape[:2]
    fg = (binary_image == 0).astype(np.uint8)

    d_xy = _tip_angle_to_direction_xy(tip_angle_deg)
    x0, y0 = float(tip_yx[1]), float(tip_yx[0])
    n = int(max(5, search_len_px / max(step_px, 1e-6)))

    samples = []
    coords = []
    inside_mask = []

    for i in range(n + 1):
        p = np.array([x0, y0], dtype=np.float64) + i * step_px * d_xy
        x = int(np.clip(round(float(p[0])), 0, w - 1))
        y = int(np.clip(round(float(p[1])), 0, h - 1))
        samples.append(float(grayscale[y, x]))
        coords.append(p.copy())
        inside_mask.append(int(fg[y, x] == 1))

    s = np.array(samples, dtype=float)
    g = np.diff(s)
    g_abs = np.abs(g)

    valid = [i for i in range(len(g_abs)) if inside_mask[i] == 1]
    if not valid:
        yy, xx, dbg = refine_tip_edge_distance_transform(binary_image, tip_yx, tip_angle_deg)
        dbg["fallback"] = "edge_dt"
        return yy, xx, dbg

    valid = np.array(valid, dtype=int)
    j = int(valid[np.argmax(g_abs[valid])])
    p_edge = 0.5 * (coords[j] + coords[j + 1])

    dbg = {
        "mode": "edge_grad",
        "d_xy": d_xy.tolist(),
        "j": j,
        "g_max": float(g_abs[j]),
        "step_px": float(step_px),
        "tip_xy": p_edge.tolist(),
        "center_line_xy": [np.array([x0, y0]).tolist(), p_edge.tolist()],
    }
    return float(p_edge[1]), float(p_edge[0]), dbg


def skeletonize_mask(mask_fg: np.ndarray) -> np.ndarray:
    try:
        from skimage.morphology import skeletonize
        skel = skeletonize(mask_fg.astype(bool))
        return skel.astype(np.uint8)
    except Exception:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            skel = cv2.ximgproc.thinning((mask_fg * 255).astype(np.uint8))
            return (skel > 0).astype(np.uint8)
        raise RuntimeError(
            "Need either scikit-image (preferred) or cv2.ximgproc.thinning for skeletonization."
        )


def skeleton_degree_image(skel: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), dtype=np.uint8)
    k[1, 1] = 0
    deg = cv2.filter2D(skel.astype(np.uint8), cv2.CV_16U, k)
    return deg


def build_skeleton_graph(skel: np.ndarray):
    pts = np.argwhere(skel > 0)
    pointset = {tuple(p) for p in pts}
    adj = {}
    for y, x in pointset:
        nbrs = []
        for dy, dx, w in _NEI8_W:
            q = (y + dy, x + dx)
            if q in pointset:
                nbrs.append((q, w))
        adj[(y, x)] = nbrs
    return adj


def dijkstra_skeleton(adj, src):
    dist = {src: 0.0}
    prev = {}
    pq = [(0.0, src)]
    visited = set()

    while pq:
        d, u = heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heappush(pq, (nd, v))
    return dist, prev


def reconstruct_path(prev, dst):
    path = [dst]
    cur = dst
    while cur in prev:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path


def path_to_xy(path_yx):
    return np.array([[x, y] for y, x in path_yx], dtype=np.float64)


def cumulative_arclength(xy: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(xy) == 1:
        return np.array([0.0], dtype=np.float64)
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(ds)])


def interp_point_on_path(xy: np.ndarray, s: np.ndarray, target_s: float) -> np.ndarray:
    target_s = float(np.clip(target_s, s[0], s[-1]))
    x = np.interp(target_s, s, xy[:, 0])
    y = np.interp(target_s, s, xy[:, 1])
    return np.array([x, y], dtype=np.float64)


def robust_tangent_from_path_window(xy: np.ndarray, s: np.ndarray, s0: float, s1: float) -> np.ndarray:
    mask = (s >= s0) & (s <= s1)
    pts = xy[mask]

    if pts.shape[0] < 5:
        i1 = max(0, len(xy) - 15)
        i2 = max(i1 + 2, len(xy) - 3)
        pts = xy[i1:i2]

    if pts.shape[0] < 2:
        v = xy[-1] - xy[max(0, len(xy) - 2)]
        n = np.linalg.norm(v)
        return v / max(n, 1e-9)

    mu = pts.mean(axis=0)
    _, _, vh = np.linalg.svd(pts - mu, full_matrices=False)
    t = vh[0]
    if np.dot(t, xy[-1] - xy[0]) < 0:
        t = -t
    n = np.linalg.norm(t)
    return t / max(n, 1e-9)


def refine_tip_edge_mainray(
    grayscale: np.ndarray,
    binary_image: np.ndarray,
    tip_yx: tuple,
    fit_back_near_r: float = 1.5,
    fit_back_far_r: float = 6.0,
    anchor_back_r: float = 1.0,
    ray_step_px: float = 0.5,
    ray_max_len_r: float = 8.0,
):
    mask_fg = (binary_image == 0).astype(np.uint8)
    if mask_fg.sum() == 0:
        return float(tip_yx[0]), float(tip_yx[1]), {"mode": "mainray", "fallback": "empty_fg"}

    skel = skeletonize_mask(mask_fg)
    deg = skeleton_degree_image(skel)
    endpoints = np.argwhere((skel > 0) & (deg == 1))
    if len(endpoints) < 2:
        return float(tip_yx[0]), float(tip_yx[1]), {"mode": "mainray", "fallback": "not_enough_endpoints"}

    tip_arr = np.asarray(tip_yx, dtype=np.float64)
    d2 = np.sum((endpoints.astype(np.float64) - tip_arr[None, :]) ** 2, axis=1)
    distal_ep = tuple(endpoints[int(np.argmin(d2))])

    adj = build_skeleton_graph(skel)
    dist_map, prev = dijkstra_skeleton(adj, distal_ep)
    endpoint_tuples = [tuple(p) for p in endpoints]
    proximal_ep = max(endpoint_tuples, key=lambda p: dist_map.get(p, -1.0))

    path_yx = reconstruct_path(prev, proximal_ep)
    if len(path_yx) < 5:
        return float(tip_yx[0]), float(tip_yx[1]), {"mode": "mainray", "fallback": "short_path"}

    path_xy = path_to_xy(path_yx)
    s = cumulative_arclength(path_xy)
    s_end = float(s[-1])

    dist_img = cv2.distanceTransform(mask_fg, cv2.DIST_L2, 5)
    tail = path_yx[max(0, len(path_yx) - 20):max(1, len(path_yx) - 3)]
    r_vals = [float(dist_img[y, x]) for (y, x) in tail if 0 <= y < dist_img.shape[0] and 0 <= x < dist_img.shape[1]]
    r_px = float(np.median(r_vals)) if len(r_vals) else float(dist_img[distal_ep[0], distal_ep[1]])
    r_px = max(r_px, 3.0)

    fit_near = fit_back_near_r * r_px
    fit_far = fit_back_far_r * r_px
    anchor_back = anchor_back_r * r_px
    ray_max_len = ray_max_len_r * r_px

    s0 = max(0.0, s_end - fit_far)
    s1 = max(0.0, s_end - fit_near)

    tangent_xy = robust_tangent_from_path_window(path_xy, s, s0, s1)
    anchor_xy = interp_point_on_path(path_xy, s, max(0.0, s_end - anchor_back))
    edge_xy = ray_last_inside(mask_fg, anchor_xy, tangent_xy, step_px=ray_step_px, max_len_px=ray_max_len)

    dbg = {
        "mode": "mainray",
        "distal_endpoint_yx": list(distal_ep),
        "proximal_endpoint_yx": list(proximal_ep),
        "radius_px": r_px,
        "fit_window_px": [fit_near, fit_far],
        "anchor_xy": anchor_xy.tolist(),
        "tangent_xy": tangent_xy.tolist(),
        "tip_xy": edge_xy.tolist(),
        "path_len_px": s_end,
        "center_line_xy": [anchor_xy.tolist(), edge_xy.tolist()],
        "path_window_xy": path_xy[(s >= s0) & (s <= s1)].tolist(),
    }
    return float(edge_xy[1]), float(edge_xy[0]), dbg


def refine_tip_parallel_centerline(
    grayscale: np.ndarray,
    binary_image: np.ndarray,
    tip_yx: Tuple[int, int],
    tip_angle_deg: float,
    section_near_r: float = 1.0,
    section_far_r: float = 6.0,
    scan_half_r: float = 3.0,
    num_sections: int = 9,
    cross_step_px: float = 0.5,
    ray_step_px: float = 0.5,
    ray_max_len_r: float = 8.0,
):
    mask_fg = (binary_image == 0).astype(np.uint8)
    if mask_fg.sum() == 0:
        return float(tip_yx[0]), float(tip_yx[1]), {"mode": "parallel_centerline", "fallback": "empty_fg"}

    d_xy = _tip_angle_to_direction_xy(tip_angle_deg)
    n_xy = np.array([-d_xy[1], d_xy[0]], dtype=np.float64)
    n_xy /= max(np.linalg.norm(n_xy), 1e-12)

    p_guess_xy = np.array([float(tip_yx[1]), float(tip_yx[0])], dtype=np.float64)
    p_in_xy, found_inside, back_dist = _backtrack_point_inside_fg(mask_fg, p_guess_xy, d_xy, max_back_px=120.0, step_px=0.5)
    if not found_inside:
        return float(tip_yx[0]), float(tip_yx[1]), {
            "mode": "parallel_centerline",
            "fallback": "could_not_backtrack_inside",
        }

    dist_img = cv2.distanceTransform(mask_fg, cv2.DIST_L2, 5)
    r_px = _estimate_radius_along_axis(mask_fg, dist_img, p_in_xy, d_xy, back_len_px=80.0, step_px=1.0)

    section_near_px = max(0.5 * r_px, float(section_near_r) * r_px)
    section_far_px = max(section_near_px + 2.0, float(section_far_r) * r_px)
    scan_half_px = max(2.5 * r_px, float(scan_half_r) * r_px)
    ray_max_len_px = max(20.0, float(ray_max_len_r) * r_px)

    s_samples = np.linspace(section_near_px, section_far_px, int(max(3, num_sections)))
    left_offsets = []
    right_offsets = []
    section_centers = []
    left_points = []
    right_points = []

    for s_back in s_samples:
        c_xy = p_in_xy - s_back * d_xy
        res = _cross_section_boundaries(
            mask_fg,
            center_xy=c_xy,
            normal_xy=n_xy,
            scan_half_width_px=scan_half_px,
            step_px=float(cross_step_px),
        )
        if res is None:
            continue
        t_left = float(res["t_left"])
        t_right = float(res["t_right"])
        if t_left > t_right:
            t_left, t_right = t_right, t_left
        left_offsets.append(t_left)
        right_offsets.append(t_right)
        section_centers.append(c_xy.tolist())
        left_points.append(np.asarray(res["p_left_xy"], dtype=np.float64).tolist())
        right_points.append(np.asarray(res["p_right_xy"], dtype=np.float64).tolist())

    if len(left_offsets) < 2 or len(right_offsets) < 2:
        return float(tip_yx[0]), float(tip_yx[1]), {
            "mode": "parallel_centerline",
            "fallback": "insufficient_cross_sections",
            "radius_px": r_px,
        }

    t_left_med = float(np.median(left_offsets))
    t_right_med = float(np.median(right_offsets))
    if t_left_med > t_right_med:
        t_left_med, t_right_med = t_right_med, t_left_med
    t_center = 0.5 * (t_left_med + t_right_med)
    width_px = float(t_right_med - t_left_med)

    line_back_px = section_far_px + 0.75 * r_px
    base_center_xy = p_in_xy - line_back_px * d_xy

    left_line_start = base_center_xy + t_left_med * n_xy
    right_line_start = base_center_xy + t_right_med * n_xy
    center_line_start = base_center_xy + t_center * n_xy

    if not _inside_mask(mask_fg, center_line_start):
        center_line_start, ok_center, _ = _backtrack_point_inside_fg(mask_fg, center_line_start, d_xy, max_back_px=3.0 * r_px, step_px=0.5)
        if not ok_center:
            center_line_start = p_in_xy + t_center * n_xy

    tip_xy = ray_last_inside(mask_fg, center_line_start, d_xy, step_px=float(ray_step_px), max_len_px=ray_max_len_px + line_back_px)

    line_forward_px = float(np.linalg.norm(tip_xy - base_center_xy)) + 0.5 * r_px
    left_line_end = base_center_xy + line_forward_px * d_xy + t_left_med * n_xy
    right_line_end = base_center_xy + line_forward_px * d_xy + t_right_med * n_xy
    center_line_end = base_center_xy + line_forward_px * d_xy + t_center * n_xy

    dbg = {
        "mode": "parallel_centerline",
        "tip_angle_deg": float(tip_angle_deg),
        "d_xy": d_xy.tolist(),
        "n_xy": n_xy.tolist(),
        "tip_guess_xy": p_guess_xy.tolist(),
        "inside_anchor_xy": p_in_xy.tolist(),
        "backtrack_dist_px": None if back_dist is None else float(back_dist),
        "radius_px": float(r_px),
        "width_px": width_px,
        "left_offset_px": float(t_left_med),
        "right_offset_px": float(t_right_med),
        "center_offset_px": float(t_center),
        "section_centers_xy": section_centers,
        "section_left_points_xy": left_points,
        "section_right_points_xy": right_points,
        "parallel_left_line_xy": [left_line_start.tolist(), left_line_end.tolist()],
        "parallel_right_line_xy": [right_line_start.tolist(), right_line_end.tolist()],
        "center_line_xy": [center_line_start.tolist(), tip_xy.tolist()],
        "center_line_full_xy": [center_line_start.tolist(), center_line_end.tolist()],
        "tip_xy": tip_xy.tolist(),
    }
    return float(tip_xy[1]), float(tip_xy[0]), dbg


def _parallel_distance_tip_xy(dbg: Dict[str, Any]) -> Optional[np.ndarray]:
    if not isinstance(dbg, dict):
        return None

    tip_xy = np.asarray(dbg.get("tip_xy", []), dtype=float).reshape(-1)
    d_xy = np.asarray(dbg.get("d_xy", []), dtype=float).reshape(-1)
    if tip_xy.size != 2 or d_xy.size != 2 or not np.all(np.isfinite(tip_xy)) or not np.all(np.isfinite(d_xy)):
        return None

    width_px = dbg.get("width_px")
    radius_px = dbg.get("radius_px")
    if width_px is not None and np.isfinite(float(width_px)):
        backoff_px = 0.5 * float(width_px)
    elif radius_px is not None and np.isfinite(float(radius_px)):
        backoff_px = float(radius_px)
    else:
        return None

    if backoff_px <= 0:
        return None
    return tip_xy - backoff_px * d_xy


def annotate_tip_geometry_on_axes(axs, dbg: Dict[str, Any], title_suffix: str = ""):
    if axs is None or not isinstance(dbg, dict):
        return

    try:
        if isinstance(axs, np.ndarray):
            target_axes = [axs.flat[-1]]
            if axs.size >= 3:
                target_axes.append(axs.flat[-2])
        elif isinstance(axs, (list, tuple)):
            target_axes = [axs[-1]]
        else:
            target_axes = [axs]

        used_labels = set()
        tip_xy = np.asarray(dbg.get("tip_xy", []), dtype=float).reshape(-1)

        for ax in target_axes:
            if ax is None:
                continue

            def _plot_line(key, color, label):
                line = dbg.get(key)
                if line is None:
                    return
                arr = np.asarray(line, dtype=float).reshape(-1, 2)
                if arr.shape[0] < 2:
                    return
                line_label = label if label not in used_labels else None
                if line_label is not None:
                    used_labels.add(label)
                ax.plot(arr[:, 0], arr[:, 1], color=color, linewidth=2.0, label=line_label)

            _plot_line("parallel_left_line_xy", "#00ff66", "tube side lines")
            _plot_line("parallel_right_line_xy", "#00ff66", "tube side lines")
            _plot_line("center_line_xy", "#ffd400", "center line")
            _plot_line("center_line_full_xy", "#ffaa00", "center line (full)")

            if "path_window_xy" in dbg:
                pts = np.asarray(dbg["path_window_xy"], dtype=float).reshape(-1, 2)
                if pts.size > 0:
                    lbl = "tangent fit window" if "tangent fit window" not in used_labels else None
                    if lbl is not None:
                        used_labels.add(lbl)
                    ax.plot(pts[:, 0], pts[:, 1], color="#00e5ff", linewidth=2.0, label=lbl)

            if "section_left_points_xy" in dbg and len(dbg["section_left_points_xy"]) > 0:
                pts = np.asarray(dbg["section_left_points_xy"], dtype=float).reshape(-1, 2)
                lbl = "sampled edge points" if "sampled edge points" not in used_labels else None
                if lbl is not None:
                    used_labels.add(lbl)
                ax.scatter(pts[:, 0], pts[:, 1], s=10, c="#44ff44", label=lbl)

            if "section_right_points_xy" in dbg and len(dbg["section_right_points_xy"]) > 0:
                pts = np.asarray(dbg["section_right_points_xy"], dtype=float).reshape(-1, 2)
                ax.scatter(pts[:, 0], pts[:, 1], s=10, c="#44ff44")

            if "inside_anchor_xy" in dbg:
                p = np.asarray(dbg["inside_anchor_xy"], dtype=float).reshape(-1)
                lbl = "inside anchor" if "inside anchor" not in used_labels else None
                if lbl is not None:
                    used_labels.add(lbl)
                ax.scatter([p[0]], [p[1]], s=40, c="#ffffff", edgecolors="#000000", label=lbl, zorder=5)

            if tip_xy.size == 2 and np.all(np.isfinite(tip_xy)):
                lbl = "refined tip" if "refined tip" not in used_labels else None
                if lbl is not None:
                    used_labels.add("refined tip")
                ax.scatter([tip_xy[0]], [tip_xy[1]], s=55, c="#ff3b30", edgecolors="#ffffff", label=lbl, zorder=6)
                ax.annotate(
                    "tip",
                    (tip_xy[0], tip_xy[1]),
                    xytext=(6, -8),
                    textcoords="offset points",
                    color="white",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55),
                )

    except Exception as e:
        print(f"[WARN] Failed to annotate analysis axes: {e}")


def _remove_zoom_coarse_tip_markers(axs):
    if axs is None or not isinstance(axs, np.ndarray) or axs.size < 4:
        return

    for ax in (axs[1, 0], axs[1, 1]):
        for coll in list(ax.collections):
            if not isinstance(coll, PathCollection):
                continue
            try:
                offsets = np.asarray(coll.get_offsets(), dtype=float)
            except Exception:
                continue
            if offsets.ndim == 2 and offsets.shape[0] == 1:
                coll.remove()

        legend = ax.get_legend()
        if legend is not None:
            legend.remove()

    try:
        axs[1, 0].set_title("Refined tip geometry")
    except Exception:
        pass


def _remove_analysis_legends(axs):
    if axs is None:
        return

    if isinstance(axs, np.ndarray):
        target_axes = list(axs.flat)
    elif isinstance(axs, (list, tuple)):
        target_axes = list(axs)
    else:
        target_axes = [axs]

    for ax in target_axes:
        if ax is None:
            continue
        legend = ax.get_legend()
        if legend is not None:
            legend.remove()


def _remap_zoom_axes_to_crop_coordinates(axs, zoom_x_min: int, zoom_x_max: int, zoom_y_min: int, zoom_y_max: int):
    if axs is None or not isinstance(axs, np.ndarray) or axs.size < 4:
        return

    zoom_axes = [axs[1, 0], axs[1, 1]]
    x_offset = float(zoom_x_min)
    y_offset = float(zoom_y_min)
    extent = [float(zoom_x_min), float(zoom_x_max + 1), float(zoom_y_max + 1), float(zoom_y_min)]

    for ax in zoom_axes:
        if ax is None:
            continue

        for image in ax.images:
            image.set_extent(extent)

        for coll in ax.collections:
            try:
                offsets = coll.get_offsets()
            except Exception:
                continue
            if offsets is None:
                continue
            offsets = np.asarray(offsets, dtype=float)
            if offsets.size == 0:
                continue
            shifted = offsets.copy()
            shifted[:, 0] += x_offset
            shifted[:, 1] += y_offset
            coll.set_offsets(shifted)

        for line in ax.lines:
            try:
                xdata = np.asarray(line.get_xdata(), dtype=float)
                ydata = np.asarray(line.get_ydata(), dtype=float)
            except Exception:
                continue
            if xdata.size == 0 or ydata.size == 0:
                continue
            line.set_xdata(xdata + x_offset)
            line.set_ydata(ydata + y_offset)

        ax.set_xlim(float(zoom_x_min), float(zoom_x_max + 1))
        ax.set_ylim(float(zoom_y_max + 1), float(zoom_y_min))
        ax.set_aspect("equal")


def patch_analyze_data_for_tip_refinement(
    cal: CTR_Shadow_Calibration,
    refine_mode: str = "none",
    dt_step_px: float = 1.0,
    dt_max_step_px: int = 80,
    grad_step_px: float = 0.25,
    grad_search_len_px: int = 60,
    mainray_fit_back_near_r: float = 1.5,
    mainray_fit_back_far_r: float = 6.0,
    mainray_anchor_back_r: float = 1.0,
    mainray_ray_step_px: float = 0.5,
    mainray_ray_max_len_r: float = 8.0,
    parallel_section_near_r: float = 1.0,
    parallel_section_far_r: float = 6.0,
    parallel_scan_half_r: float = 3.0,
    parallel_num_sections: int = 9,
    parallel_cross_step_px: float = 0.5,
    parallel_ray_step_px: float = 0.5,
    parallel_ray_max_len_r: float = 8.0,
):
    original_analyze_data = cal.analyze_data
    if not hasattr(cal, "tip_refine_debug_records"):
        cal.tip_refine_debug_records = {}

    def analyze_data_patched(
        image_file_name,
        crop_width_min=None,
        crop_width_max=None,
        crop_height_min=None,
        crop_height_max=None,
        threshold=200,
    ):
        fig, axs, coarse_row, fine_row = original_analyze_data(
            image_file_name,
            crop_width_min,
            crop_width_max,
            crop_height_min,
            crop_height_max,
            threshold,
        )

        raw_data_folder = Path(cal.calibration_data_folder) / "raw_image_data_folder"
        img = cv2.imread(str(raw_data_folder / image_file_name), cv2.IMREAD_COLOR)
        if img is None:
            return fig, axs, coarse_row, fine_row

        if crop_width_min is None:
            crop_width_min = cal.analysis_crop["crop_width_min"]
        if crop_width_max is None:
            crop_width_max = cal.analysis_crop["crop_width_max"]
        if crop_height_min is None:
            crop_height_min = cal.analysis_crop["crop_height_min"]
        if crop_height_max is None:
            crop_height_max = cal.analysis_crop["crop_height_max"]

        crop_x_min_img = int(crop_width_min)
        crop_x_max_img = int(crop_width_max)
        crop_y_min_img = int(img.shape[0] - crop_height_max)
        crop_y_max_img = int(img.shape[0] - crop_height_min)

        cropped = img[crop_y_min_img:crop_y_max_img, crop_x_min_img:crop_x_max_img, :]
        if cropped.size == 0:
            return fig, axs, coarse_row, fine_row

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)

        tip_y_full = float(coarse_row[0])
        tip_x_full = float(coarse_row[1])
        tip_angle_deg = float(coarse_row[5])

        tip_y = tip_y_full - crop_y_min_img
        tip_x = tip_x_full - crop_x_min_img
        tip_y = float(np.clip(tip_y, 0, binary.shape[0] - 1))
        tip_x = float(np.clip(tip_x, 0, binary.shape[1] - 1))

        zoom_x_min = int(max(int(round(tip_x)) - 75, 0))
        zoom_x_max = int(min(int(round(tip_x)) + 75, binary.shape[1] - 1))
        zoom_y_min = int(max(int(round(tip_y)) - 75, 0))
        zoom_y_max = int(min(int(round(tip_y)) + 75, binary.shape[0] - 1))

        if refine_mode == "edge_dt":
            yy, xx, _dbg = refine_tip_edge_distance_transform(
                binary_image=binary,
                tip_yx=(int(round(tip_y)), int(round(tip_x))),
                tip_angle_deg=tip_angle_deg,
                max_step_px=int(dt_max_step_px),
                step_px=float(dt_step_px),
            )
        elif refine_mode == "edge_grad":
            yy, xx, _dbg = refine_tip_edge_subpixel_gradient(
                grayscale=gray,
                binary_image=binary,
                tip_yx=(int(round(tip_y)), int(round(tip_x))),
                tip_angle_deg=tip_angle_deg,
                search_len_px=int(grad_search_len_px),
                step_px=float(grad_step_px),
            )
        elif refine_mode == "mainray":
            yy, xx, _dbg = refine_tip_edge_mainray(
                grayscale=gray,
                binary_image=binary,
                tip_yx=(int(round(tip_y)), int(round(tip_x))),
                fit_back_near_r=float(mainray_fit_back_near_r),
                fit_back_far_r=float(mainray_fit_back_far_r),
                anchor_back_r=float(mainray_anchor_back_r),
                ray_step_px=float(mainray_ray_step_px),
                ray_max_len_r=float(mainray_ray_max_len_r),
            )
        elif refine_mode == "parallel_centerline":
            yy, xx, _dbg = refine_tip_parallel_centerline(
                grayscale=gray,
                binary_image=binary,
                tip_yx=(int(round(tip_y)), int(round(tip_x))),
                tip_angle_deg=tip_angle_deg,
                section_near_r=float(parallel_section_near_r),
                section_far_r=float(parallel_section_far_r),
                scan_half_r=float(parallel_scan_half_r),
                num_sections=int(parallel_num_sections),
                cross_step_px=float(parallel_cross_step_px),
                ray_step_px=float(parallel_ray_step_px),
                ray_max_len_r=float(parallel_ray_max_len_r),
            )
        else:
            yy, xx, _dbg = tip_y, tip_x, {"mode": "none", "tip_xy": [tip_x, tip_y]}

        coarse_row_refined = np.array(coarse_row, dtype=float).copy()
        coarse_row_refined[0] = yy + crop_y_min_img
        coarse_row_refined[1] = xx + crop_x_min_img

        dbg_local = dict(_dbg) if isinstance(_dbg, dict) else {}
        dbg_local["image_file_name"] = image_file_name
        dbg_local["tip_angle_deg"] = tip_angle_deg
        dbg_local["coarse_tip_before_local_xy"] = [float(tip_x), float(tip_y)]
        dbg_local["crop_origin_xy"] = [int(crop_x_min_img), int(crop_y_min_img)]

        if refine_mode == "parallel_centerline":
            analysis_tip_xy = _parallel_distance_tip_xy(dbg_local)
            if analysis_tip_xy is not None:
                dbg_local["branch_end_tip_xy"] = _json_ready(dbg_local.get("tip_xy"))
                dbg_local["parallel_distance_tip_xy"] = analysis_tip_xy.tolist()
                dbg_local["tip_xy"] = analysis_tip_xy.tolist()
                center_line_xy = np.asarray(dbg_local.get("center_line_xy", []), dtype=float).reshape(-1, 2)
                if center_line_xy.shape[0] >= 1:
                    dbg_local["center_line_xy"] = [center_line_xy[0].tolist(), analysis_tip_xy.tolist()]
                xx = float(analysis_tip_xy[0])
                yy = float(analysis_tip_xy[1])

        dbg_local["coarse_tip_after_local_xy"] = [float(xx), float(yy)]
        cal.tip_refine_debug_records[image_file_name] = _json_ready(dbg_local)

        _remap_zoom_axes_to_crop_coordinates(axs, zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max)
        _remove_zoom_coarse_tip_markers(axs)
        _remove_analysis_legends(axs)
        if refine_mode != "parallel_centerline":
            annotate_tip_geometry_on_axes(axs, dbg_local, title_suffix=f" ({refine_mode})")
            _remove_analysis_legends(axs)

        coarse_row_refined[0] = yy + crop_y_min_img
        coarse_row_refined[1] = xx + crop_x_min_img
        return fig, axs, coarse_row_refined, fine_row

    cal.analyze_data = analyze_data_patched
    return cal


# =============================================================================
# Filename metadata parsing for the first script
# =============================================================================
_NUM = r"[-+]?\d+(?:\.\d+)?"

RE_STAGE_X = re.compile(rf"_X({_NUM})")
RE_STAGE_Y = re.compile(rf"_Y({_NUM})")
RE_STAGE_Z = re.compile(rf"_Z({_NUM})")
RE_B_VAL = re.compile(rf"_B({_NUM})")
RE_C_VAL = re.compile(rf"_C({_NUM})")
RE_TIP_VAL = re.compile(rf"_TIP({_NUM})")
RE_BPH_VAL = re.compile(rf"_BPH({_NUM})")
RE_SAMPLE_IDX = re.compile(r"^(\d+)_")
RE_PHASE = re.compile(r"^\d+_([^_]+)")
RE_BLOCK_NAME = re.compile(
    r"_((?:C0|C180|"
    r"BATTACK_C[-+]?\d+(?:\.\d+)?|"
    r"ALLMETRICS(?:_[A-Z0-9]+(?:_[A-Z0-9]+)*)?_C[-+]?\d+(?:\.\d+)?|"
    r"HLINE_C[-+]?\d+(?:\.\d+)?|VLINE_C[-+]?\d+(?:\.\d+)?|"
    r"HOFF_C[-+]?\d+(?:\.\d+)?|ROFF_C[-+]?\d+(?:\.\d+)?|"
    r"ZOFF_C[-+]?\d+(?:\.\d+)?|VOFF_C[-+]?\d+(?:\.\d+)?"
    r"))(?=_(?:TIP|BPH|DIR)|$)"
)
RE_DIR_VAL = re.compile(r"_DIR([A-Za-z0-9.+-]+)")


def _safe_float_from_match(regex: re.Pattern, text: str) -> Optional[float]:
    m = regex.search(text)
    if not m:
        return None
    try:
        return float(m.group(1))
    except Exception:
        return None


def _safe_int_from_match(regex: re.Pattern, text: str) -> Optional[int]:
    m = regex.search(text)
    if not m:
        return None
    try:
        return int(m.group(1))
    except Exception:
        return None


def _safe_str_from_match(regex: re.Pattern, text: str) -> Optional[str]:
    m = regex.search(text)
    if not m:
        return None
    return str(m.group(1))


def canonicalize_c_orientation(c_deg: Optional[float], round_decimals: int = 3) -> Optional[float]:
    if c_deg is None or not np.isfinite(c_deg):
        return None
    return float(np.round(float(c_deg), int(round_decimals)))


def infer_block_name_from_c(c_deg: Optional[float]) -> Optional[str]:
    if c_deg is None or not np.isfinite(c_deg):
        return None
    c = canonicalize_c_orientation(c_deg)
    if abs(c - 0.0) < 1e-6:
        return "C0"
    if abs(c - 180.0) < 1e-6:
        return "C180"
    return f"C{c:g}"


def parse_fixed_tip_filename_metadata(image_name: str) -> Dict[str, Any]:
    """
    Parse metadata embedded in filenames produced by the first script.
    """
    stem = Path(image_name).stem

    sample_index = _safe_int_from_match(RE_SAMPLE_IDX, stem)
    phase = _safe_str_from_match(RE_PHASE, stem)
    stage_x = _safe_float_from_match(RE_STAGE_X, stem)
    stage_y = _safe_float_from_match(RE_STAGE_Y, stem)
    stage_z = _safe_float_from_match(RE_STAGE_Z, stem)
    b_val = _safe_float_from_match(RE_B_VAL, stem)
    c_val = _safe_float_from_match(RE_C_VAL, stem)
    tip_angle_deg = _safe_float_from_match(RE_TIP_VAL, stem)
    block_phase_01 = _safe_float_from_match(RE_BPH_VAL, stem)
    block_name = _safe_str_from_match(RE_BLOCK_NAME, stem)
    motion_phase = _safe_str_from_match(RE_DIR_VAL, stem)

    c_orientation_deg = canonicalize_c_orientation(c_val)
    if block_name is None:
        block_name = infer_block_name_from_c(c_orientation_deg)

    return {
        "sample_index_from_name": sample_index,
        "phase_name": phase,
        "stage_x_cmd": stage_x,
        "stage_y_cmd": stage_y,
        "stage_z_cmd": stage_z,
        "b_cmd": b_val,
        "c_cmd_deg": c_val,
        "c_orientation_deg": c_orientation_deg,
        "tip_angle_deg_from_name": tip_angle_deg,
        "block_phase_01": block_phase_01,
        "block_name": block_name,
        "motion_phase": motion_phase,
    }


# =============================================================================
# Overlay helper
# =============================================================================
def draw_checkerboard_analysis_overlay(
    cal: CTR_Shadow_Calibration,
    output_path: Path,
    tracked_rows: np.ndarray,
    image_files: List[Path],
    board_debug_image: np.ndarray,
):
    if board_debug_image is None:
        raise ValueError("board_debug_image is required.")
    if tracked_rows is None or tracked_rows.size == 0:
        raise ValueError("tracked_rows is empty; run analyze_data_batch first.")
    if getattr(cal, "board_pose", None) is None or getattr(cal, "true_vertical_img_unit", None) is None:
        raise RuntimeError("Board reference is unavailable.")

    valid_mask = np.all(np.isfinite(tracked_rows[:, :2]), axis=1)
    if tracked_rows.shape[1] > 3:
        valid_mask &= np.isfinite(tracked_rows[:, 3])
    valid_rows = tracked_rows[valid_mask]
    if valid_rows.size == 0:
        raise RuntimeError("No valid tracked points were found for annotation.")

    pts_xy = np.column_stack([valid_rows[:, 1], valid_rows[:, 0]]).astype(np.float64)
    if bool(cal.board_pose.get("use_undistort", False)):
        pts_xy_draw = _undistort_points_to_image(pts_xy, cal.camera_K, cal.camera_dist)
    else:
        pts_xy_draw = pts_xy.copy()

    overlay = board_debug_image.copy()
    h, w = overlay.shape[:2]
    origin = np.asarray(cal.board_pose["origin_px"], dtype=np.float64).reshape(2)
    v_hat = np.asarray(cal.true_vertical_img_unit, dtype=np.float64).reshape(2)
    v_hat /= max(np.linalg.norm(v_hat), 1e-12)
    u_hat = np.array([v_hat[1], -v_hat[0]], dtype=np.float64)

    arrow_len = int(max(60, min(h, w) * 0.14))
    origin_i = tuple(np.round(origin).astype(int))
    horiz_end = tuple(np.round(origin + u_hat * arrow_len).astype(int))
    vert_end = tuple(np.round(origin + v_hat * arrow_len).astype(int))
    cv2.arrowedLine(overlay, origin_i, horiz_end, (40, 80, 255), 3, tipLength=0.14)
    cv2.arrowedLine(overlay, origin_i, vert_end, (0, 220, 0), 3, tipLength=0.14)
    cv2.putText(overlay, "camera horizontal", (horiz_end[0] + 8, horiz_end[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 80, 255), 2)
    cv2.putText(overlay, "camera vertical", (vert_end[0] + 8, vert_end[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)

    for pt in pts_xy_draw:
        px = int(round(pt[0]))
        py = int(round(pt[1]))
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(overlay, (px, py), 4, (0, 255, 255), -1)
            cv2.circle(overlay, (px, py), 7, (0, 0, 0), 1)

    if pts_xy_draw.shape[0] >= 2:
        pts_poly = np.round(pts_xy_draw).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts_poly], False, (255, 200, 0), 1, lineType=cv2.LINE_AA)

    summary_lines = [
        f"tracked points: {len(valid_rows)}",
        f"local scale: {float(cal.board_px_per_mm_local):.4f} px/mm",
    ]
    if getattr(cal, "board_reference_correction_meta", None):
        cmeta = cal.board_reference_correction_meta
        summary_lines.append(f"checkerboard mm scale correction: {float(cmeta['checkerboard_mm_scale_correction']):.3f}")
        summary_lines.append(f"checkerboard x sign flipped: {bool(cmeta['checkerboard_planar_x_flipped'])}")

    y_text = 32
    for line in summary_lines:
        cv2.putText(overlay, line, (24, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 3)
        cv2.putText(overlay, line, (24, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 1)
        y_text += 30

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"Failed to write annotated checkerboard image: {output_path}")
    return output_path


# =============================================================================
# Core mm conversion + error analysis
# =============================================================================
def compute_tracked_tip_positions_mm(
    cal: CTR_Shadow_Calibration,
    tracked_rows: np.ndarray,
    image_files: List[Path],
) -> Dict[str, Any]:
    """
    Convert tracked tip pixel coordinates to checkerboard-referenced mm and
    attach metadata parsed from fixed-tip dual-C filenames.
    """
    if getattr(cal, "board_pose", None) is None:
        raise RuntimeError("Checkerboard board_pose is not available.")
    if tracked_rows is None or tracked_rows.size == 0:
        raise RuntimeError("tracked_rows is empty.")

    origin = np.asarray(cal.board_pose["origin_px"], dtype=np.float64).reshape(2)

    records = []
    mm_points = []
    valid_indices = []

    for i, row in enumerate(np.asarray(tracked_rows, dtype=float)):
        file_name = image_files[i].name if i < len(image_files) else f"sample_{i:04d}"
        meta = parse_fixed_tip_filename_metadata(file_name)

        if not np.all(np.isfinite(row[:2])):
            records.append({
                "sample_index": i,
                "image_name": file_name,
                "tip_y_px": None,
                "tip_x_px": None,
                "u_mm": None,
                "z_mm": None,
                "valid": False,
                **meta,
            })
            continue

        y_px = float(row[0])
        x_px = float(row[1])

        try:
            u_mm, z_mm = cal.pixel_point_to_calibrated_axes(
                x_px=x_px,
                y_px=y_px,
                origin_px=origin,
            )
            u_mm = float(u_mm)
            z_mm = float(z_mm)
            valid = np.isfinite(u_mm) and np.isfinite(z_mm)
        except Exception:
            u_mm, z_mm = float("nan"), float("nan")
            valid = False

        rec = {
            "sample_index": i,
            "image_name": file_name,
            "tip_y_px": y_px,
            "tip_x_px": x_px,
            "u_mm": u_mm if valid else None,
            "z_mm": z_mm if valid else None,
            "valid": bool(valid),
            **meta,
        }
        records.append(rec)

        if valid:
            mm_points.append([u_mm, z_mm])
            valid_indices.append(i)

    mm_points = np.asarray(mm_points, dtype=float) if len(mm_points) else np.empty((0, 2), dtype=float)

    return {
        "records": records,
        "valid_indices": valid_indices,
        "mm_points": mm_points,
    }


def build_reference_points_mm(
    records: List[Dict[str, Any]],
    valid_indices: List[int],
    mm_points: np.ndarray,
) -> Dict[str, Any]:
    """
    Use the centroid of each sample's C-orientation group as that sample's
    reference point.
    """
    mm_points = np.asarray(mm_points, dtype=float).reshape(-1, 2)
    if mm_points.shape[0] == 0:
        raise RuntimeError("No valid mm points available for error analysis.")

    if len(valid_indices) != mm_points.shape[0]:
        raise RuntimeError("valid_indices must match mm_points rows.")

    grouped_points: Dict[str, List[np.ndarray]] = {}
    local_keys: List[str] = []
    for global_idx, pt_mm in zip(valid_indices, mm_points):
        rec = records[global_idx] if 0 <= global_idx < len(records) else {}
        c_orientation_deg = rec.get("c_orientation_deg", None)
        key = "unknown" if c_orientation_deg is None else f"{float(c_orientation_deg):.3f}"
        grouped_points.setdefault(key, []).append(np.asarray(pt_mm, dtype=float))
        local_keys.append(key)

    group_centroids = {
        key: np.mean(np.asarray(pts, dtype=float).reshape(-1, 2), axis=0)
        for key, pts in grouped_points.items()
    }
    reference_points = np.asarray([group_centroids[key] for key in local_keys], dtype=float).reshape(-1, 2)
    ref_mean = np.mean(reference_points, axis=0)
    return {
        "reference_points_mm": reference_points,
        "reference_point_mm": {
            "u_mean_mm": float(ref_mean[0]),
            "z_mean_mm": float(ref_mean[1]),
        },
        "reference_points_raw_mm": None,
        "reference_mode": "per_c_centroid",
        "reference_description": (
            "Each measured point is compared to the centroid of its own valid "
            "C-orientation group in checkerboard-referenced coordinates."
        ),
        "alignment_shift_mm": {
            "u_mm": 0.0,
            "z_mm": 0.0,
        },
    }


def compute_global_error_metrics(
    mm_points: np.ndarray,
    reference_points_mm: np.ndarray,
    reference_meta: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Error = Euclidean distance between each measured point and its corresponding
    reference point.
    RMSE = sqrt(mean(error^2))
    """
    mm_points = np.asarray(mm_points, dtype=float).reshape(-1, 2)
    reference_points_mm = np.asarray(reference_points_mm, dtype=float).reshape(-1, 2)
    if mm_points.shape[0] == 0:
        raise RuntimeError("No valid mm points available for error analysis.")
    if reference_points_mm.shape != mm_points.shape:
        raise RuntimeError("Reference points must match measured mm_points shape.")

    deltas = mm_points - reference_points_mm
    dist_mm = np.linalg.norm(deltas, axis=1)

    rmse_mm = float(np.sqrt(np.mean(dist_mm ** 2)))
    mean_err_mm = float(np.mean(dist_mm))
    std_err_mm = float(np.std(dist_mm))
    max_err_mm = float(np.max(dist_mm))
    min_err_mm = float(np.min(dist_mm))
    median_err_mm = float(np.median(dist_mm))

    return {
        "reference_point_mm": reference_meta["reference_point_mm"],
        "reference_points_mm": reference_points_mm,
        "reference_points_raw_mm": reference_meta.get("reference_points_raw_mm"),
        "reference_mode": str(reference_meta.get("reference_mode", "unknown")),
        "reference_description": str(reference_meta.get("reference_description", "")),
        "alignment_shift_mm": dict(reference_meta.get("alignment_shift_mm", {})),
        "errors_mm": dist_mm,
        "deltas_mm": deltas,
        "rmse_mm": rmse_mm,
        "mean_error_mm": mean_err_mm,
        "std_error_mm": std_err_mm,
        "median_error_mm": median_err_mm,
        "min_error_mm": min_err_mm,
        "max_error_mm": max_err_mm,
        "num_samples": int(mm_points.shape[0]),
    }


def attach_errors_to_records(
    records: List[Dict[str, Any]],
    valid_indices: List[int],
    reference_points_mm: np.ndarray,
    reference_points_raw_mm: Optional[np.ndarray],
    deltas_mm: np.ndarray,
    errors_mm: np.ndarray,
):
    idx_to_local = {global_idx: k for k, global_idx in enumerate(valid_indices)}

    for i, rec in enumerate(records):
        if i not in idx_to_local:
            rec["du_to_reference_mm"] = None
            rec["dz_to_reference_mm"] = None
            rec["reference_u_mm"] = None
            rec["reference_z_mm"] = None
            rec["reference_u_raw_mm"] = None
            rec["reference_z_raw_mm"] = None
            rec["du_from_global_mean_mm"] = None
            rec["dz_from_global_mean_mm"] = None
            rec["du_from_c_centroid_mm"] = None
            rec["dz_from_c_centroid_mm"] = None
            rec["error_distance_mm"] = None
            continue

        k = idx_to_local[i]
        rec["reference_u_mm"] = float(reference_points_mm[k, 0])
        rec["reference_z_mm"] = float(reference_points_mm[k, 1])
        if reference_points_raw_mm is None:
            rec["reference_u_raw_mm"] = None
            rec["reference_z_raw_mm"] = None
        else:
            rec["reference_u_raw_mm"] = float(reference_points_raw_mm[k, 0])
            rec["reference_z_raw_mm"] = float(reference_points_raw_mm[k, 1])
        rec["du_to_reference_mm"] = float(deltas_mm[k, 0])
        rec["dz_to_reference_mm"] = float(deltas_mm[k, 1])
        rec["du_from_global_mean_mm"] = float(deltas_mm[k, 0])
        rec["dz_from_global_mean_mm"] = float(deltas_mm[k, 1])
        rec["du_from_c_centroid_mm"] = float(deltas_mm[k, 0])
        rec["dz_from_c_centroid_mm"] = float(deltas_mm[k, 1])
        rec["error_distance_mm"] = float(errors_mm[k])

    return records


def compute_per_orientation_metrics(
    records: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """
    Compute metrics grouped by C orientation using the same per-sample reference
    definition already attached to each record.
    """
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        if not rec.get("valid", False) or rec.get("error_distance_mm") is None:
            continue
        c_orientation_deg = rec.get("c_orientation_deg", None)
        if c_orientation_deg is None or not np.isfinite(c_orientation_deg):
            key = "unknown"
        else:
            key = f"{float(c_orientation_deg):.3f}"
        grouped.setdefault(key, []).append(rec)

    out = {}
    for key, recs in sorted(grouped.items(), key=lambda kv: kv[0]):
        pts = np.asarray([[float(r["u_mm"]), float(r["z_mm"])] for r in recs], dtype=float)
        if pts.size == 0:
            continue

        refs = np.asarray(
            [[float(r["reference_u_mm"]), float(r["reference_z_mm"])] for r in recs],
            dtype=float,
        )
        deltas = pts - refs
        errs = np.asarray([float(r["error_distance_mm"]) for r in recs], dtype=float)
        c_vals = [r.get("c_orientation_deg") for r in recs if r.get("c_orientation_deg") is not None]
        tip_vals = [r.get("tip_angle_deg_from_name") for r in recs if r.get("tip_angle_deg_from_name") is not None]

        out[key] = {
            "c_orientation_deg": None if not c_vals else float(np.median(np.asarray(c_vals, dtype=float))),
            "num_samples": int(len(recs)),
            "rmse_mm": float(np.sqrt(np.mean(errs ** 2))),
            "mean_error_mm": float(np.mean(errs)),
            "std_error_mm": float(np.std(errs)),
            "median_error_mm": float(np.median(errs)),
            "min_error_mm": float(np.min(errs)),
            "max_error_mm": float(np.max(errs)),
            "mean_u_mm": float(np.mean(pts[:, 0])),
            "mean_z_mm": float(np.mean(pts[:, 1])),
            "mean_reference_u_mm": float(np.mean(refs[:, 0])),
            "mean_reference_z_mm": float(np.mean(refs[:, 1])),
            "centroid_offset_from_reference_centroid_mm": float(
                np.linalg.norm(np.mean(pts, axis=0) - np.mean(refs, axis=0))
            ),
            "tip_angle_range_deg": None if not tip_vals else [float(np.min(tip_vals)), float(np.max(tip_vals))],
        }

    return out


def save_tracked_tip_csv(csv_path: Path, records: List[Dict[str, Any]]):
    fieldnames = [
        "sample_index",
        "sample_index_from_name",
        "image_name",
        "phase_name",
        "block_name",
        "block_phase_01",
        "motion_phase",
        "stage_x_cmd",
        "stage_y_cmd",
        "stage_z_cmd",
        "b_cmd",
        "c_cmd_deg",
        "c_orientation_deg",
        "tip_angle_deg_from_name",
        "tip_y_px",
        "tip_x_px",
        "u_mm",
        "z_mm",
        "reference_u_mm",
        "reference_z_mm",
        "reference_u_raw_mm",
        "reference_z_raw_mm",
        "du_to_reference_mm",
        "dz_to_reference_mm",
        "du_from_global_mean_mm",
        "dz_from_global_mean_mm",
        "du_from_c_centroid_mm",
        "dz_from_c_centroid_mm",
        "error_distance_mm",
        "commanded_tip_x_mm",
        "commanded_tip_y_mm",
        "commanded_tip_z_mm",
        "line_reference_mode",
        "line_group",
        "line_reference_axis",
        "line_reference_const_mm",
        "line_along_mm",
        "line_along_relative_mm",
        "line_perpendicular_error_mm",
        "line_abs_error_mm",
        "line_input_attack_angle_deg",
        "line_input_command_axis",
        "line_input_commanded_mm",
        "line_input_commanded_relative_mm",
        "line_input_commanded_source",
        "offset_command_mode",
        "offset_command_axis",
        "offset_commanded_mm",
        "offset_commanded_relative_mm",
        "offset_measured_axis",
        "offset_measured_mm",
        "offset_measured_relative_mm",
        "offset_error_mm",
        "offset_abs_error_mm",
        "offset_command_source",
        "one_daq_metrics_mode",
        "one_daq_group",
        "one_daq_angle_fit",
        "one_daq_offset_fit",
        "one_daq_fit_branch",
        "one_daq_angle_command_source",
        "one_daq_radial_command_source",
        "one_daq_z_command_source",
        "one_daq_angle_command_deg",
        "one_daq_angle_measured_deg",
        "one_daq_angle_error_deg",
        "one_daq_radial_commanded_relative_mm",
        "one_daq_radial_measured_relative_mm",
        "one_daq_radial_error_mm",
        "one_daq_z_commanded_relative_mm",
        "one_daq_z_measured_relative_mm",
        "one_daq_z_error_mm",
        "one_daq_metrics_warning",
        "line_command_reconstruction_warning",
        "valid",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for r in records:
            writer.writerow(r)


# =============================================================================
# Plot styling
# =============================================================================
def _apply_dark_axes_style(ax, title: str, xlabel: str, ylabel: str):
    ax.set_title(title, color="#f4f7fb", fontsize=12.5, pad=10, weight="semibold")
    ax.set_xlabel(xlabel, color="#d7e2ee")
    ax.set_ylabel(ylabel, color="#d7e2ee")
    ax.tick_params(colors="#c8d5e3", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color((0.75, 0.84, 0.93, 0.25))
        spine.set_linewidth(1.1)
    ax.grid(True, color=(0.75, 0.84, 0.93, 0.10), linewidth=0.8)
    ax.set_facecolor("#0f1723")


def _make_dark_density_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "sleek_density_dark",
        [
            "#0a0f18",
            "#122033",
            "#17324d",
            "#1c4f73",
            "#1f6fa8",
            "#26a0b8",
            "#79d9cf",
            "#f3d67a",
        ],
        N=256,
    )


def save_error_plot(error_plot_path: Path, records: List[Dict[str, Any]], title_prefix: str = ""):
    valid_recs = [r for r in records if r.get("valid", False) and r.get("error_distance_mm") is not None]
    if not valid_recs:
        raise RuntimeError("No valid records available for error plot.")

    sample_idx = [int(r["sample_index"]) for r in valid_recs]
    errors_mm = [float(r["error_distance_mm"]) for r in valid_recs]
    c_vals = [r.get("c_orientation_deg") for r in valid_recs]

    fig, ax = plt.subplots(figsize=(10.5, 5.4))
    fig.patch.set_facecolor("#0b1118")
    fig.patch.set_alpha(1.0)

    # Subtle segment coloring by C group
    c0_x, c0_y, c180_x, c180_y, other_x, other_y = [], [], [], [], [], []
    for x, y, c in zip(sample_idx, errors_mm, c_vals):
        if c is None:
            other_x.append(x)
            other_y.append(y)
        elif abs(float(c) - 0.0) < 1e-6:
            c0_x.append(x)
            c0_y.append(y)
        elif abs(float(c) - 180.0) < 1e-6:
            c180_x.append(x)
            c180_y.append(y)
        else:
            other_x.append(x)
            other_y.append(y)

    ax.plot(
        sample_idx,
        errors_mm,
        color="#bcd2e8",
        linewidth=1.5,
        alpha=0.7,
        zorder=1,
    )
    if c0_x:
        ax.scatter(c0_x, c0_y, s=22, color="#5ec8ff", edgecolors="none", label="C = 0°", zorder=3)
    if c180_x:
        ax.scatter(c180_x, c180_y, s=22, color="#f7a8ff", edgecolors="none", label="C = 180°", zorder=3)
    if other_x:
        ax.scatter(other_x, other_y, s=20, color="#d6dee8", edgecolors="none", label="other C", zorder=3)

    _apply_dark_axes_style(
        ax,
        title=f"{title_prefix}Tracked tip error vs sample".strip(),
        xlabel="Sample index",
        ylabel="Error distance to corresponding reference point (mm)",
    )
    leg = ax.legend(loc="upper right", frameon=True, fontsize=9)
    leg.get_frame().set_facecolor("#121c28")
    leg.get_frame().set_edgecolor((0.75, 0.84, 0.93, 0.20))
    for txt in leg.get_texts():
        txt.set_color("#e5edf6")

    fig.tight_layout()
    fig.savefig(error_plot_path, dpi=220, facecolor=fig.get_facecolor())
    plt.close(fig)


def _orientation_display_label(c_deg: Optional[float]) -> str:
    if c_deg is None or not np.isfinite(c_deg):
        return "C = unknown"
    if abs(float(c_deg) - round(float(c_deg))) < 1e-9:
        return f"C = {int(round(float(c_deg)))}°"
    return f"C = {float(c_deg):.3f}°"


def _record_b_value(rec: Dict[str, Any]) -> Optional[float]:
    b_val = rec.get("b_cmd")
    if b_val is None:
        return None
    try:
        b_val_f = float(b_val)
    except Exception:
        return None
    return b_val_f if np.isfinite(b_val_f) else None


def save_bpull_error_components_plot(plot_path: Path, records: List[Dict[str, Any]], title_prefix: str = ""):
    valid_recs = [
        r for r in records
        if r.get("valid", False)
        and r.get("du_to_reference_mm") is not None
        and r.get("dz_to_reference_mm") is not None
        and _record_b_value(r) is not None
    ]
    if not valid_recs:
        raise RuntimeError("No valid records available for B-value error component plot.")

    groups = [
        (0.0, "#5ec8ff", "C = 0°"),
        (180.0, "#f7a8ff", "C = 180°"),
    ]

    fig, axs = plt.subplots(2, 1, figsize=(11.2, 8.0), sharex=True)
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    component_specs = [
        ("du_to_reference_mm", "u error to mean of all points (mm)"),
        ("dz_to_reference_mm", "z error to mean of all points (mm)"),
    ]

    any_group = False
    for ax, (field, ylabel) in zip(axs, component_specs):
        for c_deg, color, label in groups:
            group_recs = []
            for rec in valid_recs:
                c_val = rec.get("c_orientation_deg")
                if c_val is None:
                    continue
                try:
                    c_val_f = float(c_val)
                except Exception:
                    continue
                if abs(c_val_f - c_deg) >= 1e-6:
                    continue
                x_val = _record_b_value(rec)
                if x_val is None:
                    continue
                y_val = float(rec[field])
                if not np.isfinite(y_val):
                    continue
                group_recs.append((x_val, y_val))

            if not group_recs:
                continue

            any_group = True
            group_recs.sort(key=lambda item: item[0])
            xs = np.asarray([p[0] for p in group_recs], dtype=float)
            ys = np.asarray([p[1] for p in group_recs], dtype=float)

            ax.plot(xs, ys, color=color, linewidth=1.8, alpha=0.92, label=label, zorder=2)
            ax.scatter(xs, ys, s=26, color=color, edgecolors="none", zorder=3)

        ax.axhline(0.0, color=(0.93, 0.97, 1.0, 0.28), linestyle="--", linewidth=1.1, zorder=1)
        _apply_dark_axes_style(
            ax,
            title=f"{title_prefix}{ylabel} vs B value".strip(),
            xlabel="B value",
            ylabel=ylabel,
        )
        ax.set_facecolor("none")

    if not any_group:
        plt.close(fig)
        raise RuntimeError("No C = 0° or C = 180° records available for B-pull error component plot.")

    axs[-1].set_xlabel("B value", color="#d7e2ee")
    leg = axs[0].legend(loc="upper right", frameon=True, fontsize=9)
    leg.get_frame().set_facecolor("#121c28")
    leg.get_frame().set_edgecolor((0.75, 0.84, 0.93, 0.20))
    for txt in leg.get_texts():
        txt.set_color("#e5edf6")

    fig.tight_layout()
    fig.savefig(plot_path, dpi=220, transparent=True)
    plt.close(fig)


def save_error_histogram_and_dual_orientation_heatmaps(
    plot_path: Path,
    records: List[Dict[str, Any]],
    global_metrics: Dict[str, Any],
    per_orientation_metrics: Dict[str, Any],
    title_prefix: str = "",
    bins: int = 24,
):
    valid_recs = [r for r in records if r.get("valid", False) and r.get("u_mm") is not None and r.get("z_mm") is not None]
    if not valid_recs:
        raise RuntimeError("No valid records available for summary plot.")

    errors_mm = np.asarray([float(r["error_distance_mm"]) for r in valid_recs if r.get("error_distance_mm") is not None], dtype=float)
    if errors_mm.size == 0:
        raise RuntimeError("No valid error values available.")

    global_rmse = float(global_metrics["rmse_mm"])

    # Group valid records by C orientation
    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for rec in valid_recs:
        c_deg = rec.get("c_orientation_deg", None)
        key = "unknown" if c_deg is None else f"{float(c_deg):.3f}"
        grouped.setdefault(key, []).append(rec)

    # Sort orientations numerically when possible
    def _sort_key(k: str):
        try:
            return (0, float(k))
        except Exception:
            return (1, k)

    group_items = sorted(grouped.items(), key=lambda kv: _sort_key(kv[0]))

    # Prefer side-by-side for first two groups, matching the dual-C run
    if len(group_items) == 0:
        raise RuntimeError("No grouped records found.")
    if len(group_items) == 1:
        group_items = [group_items[0], group_items[0]]
    elif len(group_items) > 2:
        # Keep the first two numeric groups for the summary figure; all groups remain in JSON/CSV.
        group_items = group_items[:2]

    all_u = np.asarray([float(r["u_mm"]) for r in valid_recs], dtype=float)
    all_z = np.asarray([float(r["z_mm"]) for r in valid_recs], dtype=float)
    u_span = float(np.ptp(all_u))
    z_span = float(np.ptp(all_z))
    u_pad = max(0.35, 0.10 * max(u_span, 1.0))
    z_pad = max(0.35, 0.10 * max(z_span, 1.0))
    xlim = (float(np.min(all_u) - u_pad), float(np.max(all_u) + u_pad))
    ylim = (float(np.min(all_z) - z_pad), float(np.max(all_z) + z_pad))

    fig = plt.figure(figsize=(13.6, 8.7), facecolor="none")
    fig.patch.set_alpha(0.0)
    gs = GridSpec(2, 2, height_ratios=[0.92, 1.35], hspace=0.28, wspace=0.16, figure=fig)

    ax_hist = fig.add_subplot(gs[0, :])
    ax_left = fig.add_subplot(gs[1, 0])
    ax_right = fig.add_subplot(gs[1, 1])

    # Histogram
    ax_hist.hist(
        errors_mm,
        bins=int(max(6, bins)),
        color=(0.44, 0.81, 1.0, 0.85),
        edgecolor=(0.93, 0.97, 1.0, 0.95),
        linewidth=0.95,
    )
    ax_hist.axvline(global_rmse, color="#ffd166", linestyle="--", linewidth=1.8, alpha=0.95, label=f"Global RMSE = {global_rmse:.4f} mm")
    _apply_dark_axes_style(
        ax_hist,
        title=f"{title_prefix}Tracked tip error distribution".strip(),
        xlabel="Error distance to corresponding reference point (mm)",
        ylabel="Number of samples",
    )
    ax_hist.set_facecolor("none")
    leg_hist = ax_hist.legend(loc="upper right", frameon=True, fontsize=10)
    leg_hist.get_frame().set_facecolor("#121c28")
    leg_hist.get_frame().set_edgecolor((0.75, 0.84, 0.93, 0.20))
    for txt in leg_hist.get_texts():
        txt.set_color("#e5edf6")

    def _draw_group_points(ax, group_key: str, group_recs: List[Dict[str, Any]], point_color: str):
        pts = np.asarray([[float(r["u_mm"]), float(r["z_mm"])] for r in group_recs], dtype=float)
        c_deg = None
        for r in group_recs:
            if r.get("c_orientation_deg") is not None:
                c_deg = float(r["c_orientation_deg"])
                break

        rmse_txt = ""
        if group_key in per_orientation_metrics:
            rmse_txt = f"  |  RMSE = {float(per_orientation_metrics[group_key]['rmse_mm']):.4f} mm"

        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            s=28,
            color=point_color,
            alpha=0.82,
            edgecolors="#f3f8ff",
            linewidths=0.35,
            zorder=3,
            label="Measured points",
        )
        refs = np.asarray(
            [[float(r["reference_u_mm"]), float(r["reference_z_mm"])] for r in group_recs],
            dtype=float,
        )
        ax.scatter(
            refs[:, 0],
            refs[:, 1],
            s=54,
            marker="s",
            facecolors="none",
            edgecolors="#8fd3ff",
            linewidths=1.2,
            zorder=4,
            label="Reference points",
        )

        for p_ref, p_meas in zip(refs, pts):
            ax.plot(
                [p_ref[0], p_meas[0]],
                [p_ref[1], p_meas[1]],
                color=(0.74, 0.88, 1.0, 0.42),
                linewidth=0.9,
                zorder=2,
            )

        ref_center = np.mean(refs, axis=0)
        meas_center = np.mean(pts, axis=0)
        ax.scatter(
            [ref_center[0]],
            [ref_center[1]],
            s=115,
            marker="s",
            facecolors="none",
            edgecolors="#00e5ff",
            linewidths=1.8,
            zorder=5,
            label="Reference center",
        )
        ax.scatter(
            [meas_center[0]],
            [meas_center[1]],
            s=85,
            marker="+",
            color="#ff66c4",
            linewidths=2.0,
            zorder=5,
            label="Measured center",
        )

        _apply_dark_axes_style(
            ax,
            title=f"{_orientation_display_label(c_deg)}{rmse_txt}",
            xlabel="u (mm)",
            ylabel="z (mm)",
        )
        ax.set_facecolor("none")
        ax.set_xlim(*xlim)
        ax.set_ylim(*ylim)

        if u_span > 1e-12 and z_span > 1e-12:
            ax.set_box_aspect((ylim[1] - ylim[0]) / max(xlim[1] - xlim[0], 1e-12))

        legend_loc = "lower left" if c_deg is not None and math.isclose(c_deg, 180.0, abs_tol=1e-6) else "upper right"
        leg = ax.legend(loc=legend_loc, frameon=True, fontsize=9)
        leg.get_frame().set_facecolor("#121c28")
        leg.get_frame().set_edgecolor((0.75, 0.84, 0.93, 0.18))
        for txt in leg.get_texts():
            txt.set_color("#e8f0f8")

    _draw_group_points(ax_left, group_items[0][0], group_items[0][1], "#7fd6ff")
    _draw_group_points(ax_right, group_items[1][0], group_items[1][1], "#ffb0d9")

    fig.suptitle(
        f"{title_prefix}Desired vs measured tracked tip points by C orientation".strip(),
        color="#f7fbff",
        fontsize=14.2,
        weight="semibold",
        y=0.985,
    )

    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.972])
    fig.savefig(plot_path, dpi=230, transparent=True)
    plt.close(fig)


def annotate_analysis_output_images(analysis_output_dir: Path, records: List[Dict[str, Any]]):
    analysis_output_dir = Path(analysis_output_dir)
    if not analysis_output_dir.is_dir():
        return

    for rec in records:
        image_name = rec.get("image_name")
        if not image_name:
            continue

        analysis_image_path = analysis_output_dir / f"{Path(image_name).stem}_analysis.png"
        if not analysis_image_path.is_file():
            continue

        image = cv2.imread(str(analysis_image_path), cv2.IMREAD_COLOR)
        if image is None:
            continue

        err = rec.get("error_distance_mm")
        c_txt = rec.get("c_orientation_deg")
        tip_txt = rec.get("tip_angle_deg_from_name")

        if err is None:
            err_str = "Error: n/a"
        else:
            err_str = f"Error: {float(err):.3f} mm"

        c_str = "C: n/a" if c_txt is None else f"C: {float(c_txt):.3f} deg"
        tip_str = "TIP: n/a" if tip_txt is None else f"TIP: {float(tip_txt):.3f} deg"

        h, w = image.shape[:2]
        banner_h = max(62, int(round(0.09 * h)))
        banner = np.full((banner_h, w, 3), 255, dtype=np.uint8)

        cv2.putText(
            banner, err_str, (18, int(round(banner_h * 0.43))),
            cv2.FONT_HERSHEY_SIMPLEX, 0.84, (20, 20, 20), 2, lineType=cv2.LINE_AA
        )
        cv2.putText(
            banner, f"{c_str}   |   {tip_str}", (18, int(round(banner_h * 0.80))),
            cv2.FONT_HERSHEY_SIMPLEX, 0.72, (40, 40, 40), 2, lineType=cv2.LINE_AA
        )

        annotated = np.vstack([banner, image])
        cv2.imwrite(str(analysis_image_path), annotated)


def save_metrics_json(
    json_path: Path,
    global_metrics: Dict[str, Any],
    per_orientation_metrics: Dict[str, Any],
    cal: CTR_Shadow_Calibration,
    args,
):
    payload = {
        "reference_point_mm": global_metrics["reference_point_mm"],
        "reference_mode": global_metrics["reference_mode"],
        "reference_description": global_metrics["reference_description"],
        "alignment_shift_mm": global_metrics["alignment_shift_mm"],
        "reference_points_mm": global_metrics["reference_points_mm"],
        "reference_points_raw_mm": global_metrics["reference_points_raw_mm"],
        "global_metrics": {
            "rmse_mm": global_metrics["rmse_mm"],
            "mean_error_mm": global_metrics["mean_error_mm"],
            "std_error_mm": global_metrics["std_error_mm"],
            "median_error_mm": global_metrics["median_error_mm"],
            "min_error_mm": global_metrics["min_error_mm"],
            "max_error_mm": global_metrics["max_error_mm"],
            "num_samples": global_metrics["num_samples"],
        },
        "per_c_orientation_metrics": per_orientation_metrics,
        "analysis_crop": getattr(cal, "analysis_crop", None),
        "board_reference": collect_board_reference_info(cal),
        "settings": {
            "threshold": int(args.threshold),
            "tracked_tip_source": str(getattr(args, "tracked_tip_source", "auto")),
            "tip_refiner_model": None if getattr(args, "tip_refiner_model", None) is None else str(Path(args.tip_refiner_model).expanduser().resolve()),
            "tip_refiner_anchor": getattr(args, "tip_refiner_anchor", None),
            "tip_refiner_compare_only": bool(getattr(args, "tip_refiner_compare_only", False)),
            "tip_detection_mode": str(args.tip_detection_mode),
            "tip_refine_mode": str(args.tip_refine_mode),
            "tip_refine_dt_step_px": float(args.tip_refine_dt_step_px),
            "tip_refine_max_step_px": int(args.tip_refine_max_step_px),
            "tip_refine_grad_step_px": float(args.tip_refine_grad_step_px),
            "tip_refine_grad_search_len_px": int(args.tip_refine_grad_search_len_px),
            "tip_refine_mainray_fit_back_near_r": float(args.tip_refine_mainray_fit_back_near_r),
            "tip_refine_mainray_fit_back_far_r": float(args.tip_refine_mainray_fit_back_far_r),
            "tip_refine_mainray_anchor_back_r": float(args.tip_refine_mainray_anchor_back_r),
            "tip_refine_mainray_ray_step_px": float(args.tip_refine_mainray_ray_step_px),
            "tip_refine_mainray_ray_max_len_r": float(args.tip_refine_mainray_ray_max_len_r),
            "tip_refine_parallel_section_near_r": float(args.tip_refine_parallel_section_near_r),
            "tip_refine_parallel_section_far_r": float(args.tip_refine_parallel_section_far_r),
            "tip_refine_parallel_scan_half_r": float(args.tip_refine_parallel_scan_half_r),
            "tip_refine_parallel_num_sections": int(args.tip_refine_parallel_num_sections),
            "tip_refine_parallel_cross_step_px": float(args.tip_refine_parallel_cross_step_px),
            "tip_refine_parallel_ray_step_px": float(args.tip_refine_parallel_ray_step_px),
            "tip_refine_parallel_ray_max_len_r": float(args.tip_refine_parallel_ray_max_len_r),
            "hist_bins": int(args.hist_bins),
        },
    }
    with open(json_path, "w") as f:
        json.dump(_json_ready(payload), f, indent=2)


# =============================================================================
# Main
# =============================================================================
def run_processing(args=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", type=str, default=None,
                    help="Existing project folder containing raw_image_data_folder/")
    ap.add_argument("--raw_dir", type=str, default=None,
                    help="raw_image_data_folder or any folder of images (will be wrapped into a project)")
    ap.add_argument("--threshold", type=int, default=200)
    ap.add_argument("--save_plots", action="store_true")

    ap.add_argument("--camera_calibration_file", type=str, required=True,
                    help="Path to camera calibration .npz for checkerboard-reference analysis.")
    ap.add_argument("--checkerboard_reference_image", type=str, required=True,
                    help="Path to checkerboard reference image.")
    ap.add_argument("--checkerboard_inner_corners", type=_parse_inner_corners_arg, default=None,
                    help="Checkerboard inner-corner grid as 'Nx,Ny' or 'NxXNy'. Defaults to metadata in the calibration file.")
    ap.add_argument("--checkerboard_square_size_mm", type=float, default=None,
                    help="Checkerboard square size in mm. Defaults to metadata in the calibration file.")
    ap.add_argument("--checkerboard_no_undistort", action="store_true",
                    help="Disable undistortion before checkerboard pose estimation.")

    ap.add_argument("--checkerboard_mm_scale_correction", type=float, default=0.5,
                    help="Multiply checkerboard-derived mm values by this factor. Default 0.5 fixes observed 2x overscale.")
    ap.add_argument("--checkerboard_no_flip_planar_x", action="store_true",
                    help="Disable checkerboard planar-x sign flip.")

    ap.add_argument("--link_mode", type=str, default="symlink", choices=["symlink", "copy"])
    ap.add_argument("--save_analysis_config", action="store_true")

    ap.add_argument("--tip_refine_mode", type=str, default="parallel_centerline",
                    choices=["none", "edge_dt", "edge_grad", "mainray", "parallel_centerline"],
                    help="Refine tip position using the same distal tip analysis modes as offline_run_calibration.py.")
    ap.add_argument("--tip_refine_dt_step_px", type=float, default=1.0)
    ap.add_argument("--tip_refine_max_step_px", type=int, default=80)
    ap.add_argument("--tip_refine_grad_step_px", type=float, default=0.25)
    ap.add_argument("--tip_refine_grad_search_len_px", type=int, default=60)
    ap.add_argument("--tip_refine_mainray_fit_back_near_r", type=float, default=1.5)
    ap.add_argument("--tip_refine_mainray_fit_back_far_r", type=float, default=6.0)
    ap.add_argument("--tip_refine_mainray_anchor_back_r", type=float, default=1.0)
    ap.add_argument("--tip_refine_mainray_ray_step_px", type=float, default=0.5)
    ap.add_argument("--tip_refine_mainray_ray_max_len_r", type=float, default=8.0)
    ap.add_argument("--tip_refine_parallel_section_near_r", type=float, default=1.0)
    ap.add_argument("--tip_refine_parallel_section_far_r", type=float, default=6.0)
    ap.add_argument("--tip_refine_parallel_scan_half_r", type=float, default=3.0)
    ap.add_argument("--tip_refine_parallel_num_sections", type=int, default=9)
    ap.add_argument("--tip_refine_parallel_cross_step_px", type=float, default=0.5)
    ap.add_argument("--tip_refine_parallel_ray_step_px", type=float, default=0.5)
    ap.add_argument("--tip_refine_parallel_ray_max_len_r", type=float, default=8.0)
    ap.add_argument("--tip_refiner_model", type=str, default=None,
                    help="Path to cnn/train_tip_refiner.py best_tip_refiner.pt for CNN tip detection.")
    ap.add_argument("--tip_refiner_anchor", type=str, default=None, choices=["coarse", "selected", "refined"],
                    help="Patch anchor for CNN inference. Defaults to the model checkpoint anchor.")
    ap.add_argument("--tip_refiner_compare_only", action="store_true",
                    help="Save CNN tips but keep non-CNN tips as the default tracked source.")
    ap.add_argument("--tip_detection_mode", type=str, default="classical",
                    choices=["classical", "red_dot", "auto_red_dot"],
                    help="Tip detection mode from CTR shadow calibration. red_dot uses the red marker centroid as the selected tip.")
    ap.add_argument("--tracked_tip_source", type=str, default="auto", choices=["auto", "coarse", "selected", "cnn"],
                    help="Which tip rows to convert to mm. auto uses selected rows when red-dot or CNN-selected tips are active, otherwise coarse.")

    ap.add_argument("--hist_bins", type=int, default=24,
                    help="Number of histogram bins.")
    ap.add_argument("--process_all_fits", "--process-all-fits", dest="process_all_fits", action="store_true",
                    help="During processing, reuse the same DAQ and generate one-DAQ metrics/plots for all requested angle/offset fit modes.")
    ap.add_argument("--plot_all_calibration_models", "--plot-all-calibration-models", "--process-all-calibration-models", dest="plot_all_calibration_models", action="store_true",
                    help="During processing, discover every usable tip_angle/r/z model family in the calibration JSON and plot all of them as panel columns.")
    ap.add_argument("--processing_angle_fits", "--processing-angle-fits", dest="processing_angle_fits", nargs="*", default=None,
                    help="Angle fit modes to process. Accepts space- or comma-separated values. Default with --process-all-fits: all modes.")
    ap.add_argument("--processing_offset_fits", "--processing-offset-fits", dest="processing_offset_fits", nargs="*", default=None,
                    help="Offset fit modes to process. Accepts space- or comma-separated values. Default with --process-all-fits: same modes as angle fits.")
    ap.add_argument("--processing_fit_branches", "--processing-fit-branches", dest="processing_fit_branches", nargs="*", default=None,
                    help="Calibration branch labels to process for fit-sweep panels, e.g. 0_90_0 0_180_0 90_180_90 or all. Each curl sequence gets its own output panel.")
    ap.add_argument("--fit_combination_mode", "--fit-combination-mode", dest="fit_combination_mode", type=str, default="paired", choices=["paired", "cross"],
                    help="How to combine angle/offset fit lists for processing sweeps. paired keeps same-named fits together; cross makes every combination.")
    ap.add_argument("--fit_output_root", "--fit-output-root", dest="fit_output_root", type=str, default="fit_sweeps",
                    help="Subfolder under processed_image_data_folder where per-fit processing outputs are saved.")

    if args is None:
        args = ap.parse_args()
    if shadow_calibration_module is None or CTR_Shadow_Calibration is None:
        raise ImportError("Processing requires shadow_calibration.py to be importable from this folder or PYTHONPATH.")
    print(f"[INFO] Using shadow_calibration module: {shadow_calibration_module.__file__}")

    if args.project_dir is None and args.raw_dir is None:
        raise SystemExit("Provide --project_dir or --raw_dir")

    project_dir = Path(args.project_dir).expanduser().resolve() if args.project_dir else None
    raw_dir = Path(args.raw_dir).expanduser().resolve() if args.raw_dir else None

    if project_dir and (project_dir / "raw_image_data_folder").is_dir():
        raw_folder = project_dir / "raw_image_data_folder"
    elif raw_dir:
        if raw_dir.name == "raw_image_data_folder":
            raw_folder = raw_dir
            if project_dir is None:
                project_dir = raw_dir.parent
            elif project_dir != raw_dir.parent:
                raw_folder = ensure_project_from_raw(raw_dir, project_dir, link_mode=args.link_mode)
        else:
            if project_dir is None:
                project_dir = raw_dir.parent / (raw_dir.name + "_offline_project")
            raw_folder = ensure_project_from_raw(raw_dir, project_dir, link_mode=args.link_mode)
    else:
        raise SystemExit("Could not resolve folders. Check paths.")

    project_dir = project_dir.resolve()
    raw_folder = raw_folder.resolve()

    cal = CTR_Shadow_Calibration(
        parent_directory=str(project_dir.parent),
        project_name=project_dir.name,
        allow_existing=True,
        add_date=False,
    )
    cal.calibration_data_folder = str(project_dir)
    cal.tip_detection_mode = str(args.tip_detection_mode)
    load_optional_tip_refiner(cal, args)
    cal.tip_parallel_section_near_r = float(args.tip_refine_parallel_section_near_r)
    cal.tip_parallel_section_far_r = float(args.tip_refine_parallel_section_far_r)
    cal.tip_parallel_scan_half_r = float(args.tip_refine_parallel_scan_half_r)
    cal.tip_parallel_num_sections = int(args.tip_refine_parallel_num_sections)
    cal.tip_parallel_cross_step_px = float(args.tip_refine_parallel_cross_step_px)
    cal.tip_parallel_ray_step_px = float(args.tip_refine_parallel_ray_step_px)
    cal.tip_parallel_ray_max_len_r = float(args.tip_refine_parallel_ray_max_len_r)

    if args.tip_refine_mode not in ("none", "parallel_centerline"):
        patch_analyze_data_for_tip_refinement(
            cal,
            refine_mode=str(args.tip_refine_mode),
            dt_step_px=float(args.tip_refine_dt_step_px),
            dt_max_step_px=int(args.tip_refine_max_step_px),
            grad_step_px=float(args.tip_refine_grad_step_px),
            grad_search_len_px=int(args.tip_refine_grad_search_len_px),
            mainray_fit_back_near_r=float(args.tip_refine_mainray_fit_back_near_r),
            mainray_fit_back_far_r=float(args.tip_refine_mainray_fit_back_far_r),
            mainray_anchor_back_r=float(args.tip_refine_mainray_anchor_back_r),
            mainray_ray_step_px=float(args.tip_refine_mainray_ray_step_px),
            mainray_ray_max_len_r=float(args.tip_refine_mainray_ray_max_len_r),
            parallel_section_near_r=float(args.tip_refine_parallel_section_near_r),
            parallel_section_far_r=float(args.tip_refine_parallel_section_far_r),
            parallel_scan_half_r=float(args.tip_refine_parallel_scan_half_r),
            parallel_num_sections=int(args.tip_refine_parallel_num_sections),
            parallel_cross_step_px=float(args.tip_refine_parallel_cross_step_px),
            parallel_ray_step_px=float(args.tip_refine_parallel_ray_step_px),
            parallel_ray_max_len_r=float(args.tip_refine_parallel_ray_max_len_r),
        )
        print(f"[INFO] Tip refinement enabled: {args.tip_refine_mode}")
    else:
        print(
            "[INFO] Using native shadow_calibration analyze_data image processing "
            f"and annotation flow ({args.tip_refine_mode})."
        )

    imgs = list_images(raw_folder)
    if not imgs:
        raise SystemExit(f"No images found in: {raw_folder}")

    first_img_path = imgs[0]
    img_bgr = cv2.imread(str(first_img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise SystemExit(f"Could not read first image: {first_img_path}")

    print(f"[INFO] Using first image for GUI: {first_img_path.name}")

    calib_path = Path(args.camera_calibration_file).expanduser().resolve()
    board_ref_path = Path(args.checkerboard_reference_image).expanduser().resolve()

    print(f"[INFO] Loading camera calibration: {calib_path}")
    cal.load_camera_calibration(str(calib_path))

    processed_dir = project_dir / "processed_image_data_folder"
    processed_dir.mkdir(parents=True, exist_ok=True)
    checkerboard_debug_path = processed_dir / "checkerboard_reference_debug.png"

    board_ref_kwargs = _board_reference_kwargs(cal, args)
    board_result = cal.estimate_board_reference_from_image(
        str(board_ref_path),
        save_debug_path=str(checkerboard_debug_path),
        **board_ref_kwargs,
    )
    board_reference_debug_image = board_result.get("debug_image")

    apply_checkerboard_reference_corrections(
        cal,
        mm_scale=float(args.checkerboard_mm_scale_correction),
        flip_planar_x=(not bool(args.checkerboard_no_flip_planar_x)),
    )

    print(f"[INFO] Checkerboard reference estimated from: {board_ref_path}")

    analysis_crop = interactive_crop_from_image(
        img_bgr,
        default_crop=cal.default_analysis_crop,
    )
    cal.analysis_crop = dict(analysis_crop)

    if args.save_analysis_config:
        cfg = {}
        if hasattr(cal, "get_analysis_reference_info"):
            cfg = cal.get_analysis_reference_info()
        cfg["board_reference"] = collect_board_reference_info(cal)
        cfg_path = project_dir / "analysis_reference.json"
        with open(cfg_path, "w") as f:
            json.dump(_json_ready(cfg), f, indent=2)
        print(f"[INFO] Saved analysis config to: {cfg_path}")

    print("\n[INFO] Running analyze_data_batch (offline)...")
    cal.analyze_data_batch(threshold=int(args.threshold))

    tracked_rows, tracked_source = select_tracked_rows_for_analysis(cal, args)
    if tracked_rows.size == 0:
        raise RuntimeError(f"No tracked tip data found for source={tracked_source} after analyze_data_batch.")
    print(f"[INFO] Using tracked tip source for mm analysis: {tracked_source}")

    print("[INFO] Converting tracked tips to checkerboard-referenced mm...")
    tip_data = compute_tracked_tip_positions_mm(cal, tracked_rows, imgs)

    print("[INFO] Building per-sample reference points and error metrics...")
    reference_meta = build_reference_points_mm(
        records=tip_data["records"],
        valid_indices=tip_data["valid_indices"],
        mm_points=tip_data["mm_points"],
    )
    global_metrics = compute_global_error_metrics(
        mm_points=tip_data["mm_points"],
        reference_points_mm=reference_meta["reference_points_mm"],
        reference_meta=reference_meta,
    )

    records = attach_errors_to_records(
        records=tip_data["records"],
        valid_indices=tip_data["valid_indices"],
        reference_points_mm=global_metrics["reference_points_mm"],
        reference_points_raw_mm=global_metrics["reference_points_raw_mm"],
        deltas_mm=global_metrics["deltas_mm"],
        errors_mm=global_metrics["errors_mm"],
    )

    print("[INFO] Computing per-C orientation metrics...")
    per_orientation_metrics = compute_per_orientation_metrics(records=records)

    print("[INFO] Computing line-reference metrics when horizontal/vertical line runs are detected...")
    robot_cal_for_line = None
    if getattr(args, "calibration", None):
        try:
            robot_cal_for_line = load_calibration(args.calibration, y_offset_fit=args.y_offset_fit)
            processing_fit_branch = resolve_fit_branch_for_window(
                getattr(args, "fit_branch", "auto"),
                attack_min_deg=getattr(args, "attack_min_deg", None),
                attack_max_deg=getattr(args, "attack_max_deg", None),
            )
            robot_cal_for_line.active_fit_branch = processing_fit_branch
            print("[INFO] Loaded robot calibration for commanded line-coordinate reconstruction.")
            print(f"[INFO] Active calibration fit branch for base processing: {_fit_branch_title(processing_fit_branch)}")
            if getattr(robot_cal_for_line, "available_fit_branches", None):
                print(f"[INFO] Branch labels detected in calibration JSON: {getattr(robot_cal_for_line, 'available_fit_branches', [])}")
        except Exception as exc:
            print(f"[WARN] Could not load robot calibration for commanded line-coordinate reconstruction: {exc}")
            print("[WARN] Line hysteresis plots will fall back to non-tip command coordinates if needed.")
    else:
        print("[WARN] No --calibration was provided for processing; commanded tip X/Z reconstruction is unavailable.")
    line_metrics = attach_line_reference_errors(
        records,
        robot_cal=robot_cal_for_line,
        flip_rz_sign=bool(getattr(args, "flip_rz_sign", False)),
    )

    print("[INFO] Computing fixed-stage offset-command hysteresis metrics when offset runs are detected...")
    offset_metrics = attach_offset_command_hysteresis(
        records,
        robot_cal=robot_cal_for_line,
        offset_fit=str(getattr(args, "offset_fit", None) or getattr(args, "angle_fit", "avg_pchip")),
        flip_rz_sign=bool(getattr(args, "flip_rz_sign", False)),
    )

    print("[INFO] Computing one-DAQ combined angle/radial/Z metrics when angle-command records are detected...")
    one_daq_metrics = attach_one_daq_all_metrics(
        records,
        robot_cal=robot_cal_for_line,
        angle_fit=str(getattr(args, "angle_fit", "avg_pchip")),
        offset_fit=str(getattr(args, "offset_fit", None) or getattr(args, "angle_fit", "avg_pchip")),
        flip_rz_sign=bool(getattr(args, "flip_rz_sign", False)),
    )

    csv_path = processed_dir / "tracked_tip_positions_mm.csv"
    metrics_json_path = processed_dir / "tracked_tip_error_metrics.json"
    line_metrics_json_path = processed_dir / "line_reference_error_metrics.json"
    offset_metrics_json_path = processed_dir / "offset_command_hysteresis_metrics.json"
    one_daq_metrics_json_path = processed_dir / "one_daq_all_metrics.json"
    error_plot_path = processed_dir / "tracked_tip_error_vs_sample.png"
    bpull_error_plot_path = processed_dir / "tracked_tip_error_components_vs_bpull.png"
    hist_hexbin_path = processed_dir / "tracked_tip_error_histogram_dual_c_heatmaps.png"

    save_tracked_tip_csv(csv_path, records)
    save_metrics_json(metrics_json_path, global_metrics, per_orientation_metrics, cal, args)
    save_line_reference_metrics_json(line_metrics_json_path, line_metrics)
    save_offset_command_metrics_json(offset_metrics_json_path, offset_metrics)
    save_one_daq_metrics_json(one_daq_metrics_json_path, one_daq_metrics)

    save_error_plot(
        error_plot_path,
        records,
        title_prefix="Checkerboard-referenced ",
    )
    save_bpull_error_components_plot(
        bpull_error_plot_path,
        records,
        title_prefix="Checkerboard-referenced ",
    )
    save_error_histogram_and_dual_orientation_heatmaps(
        hist_hexbin_path,
        records=records,
        global_metrics=global_metrics,
        per_orientation_metrics=per_orientation_metrics,
        title_prefix="Checkerboard-referenced ",
        bins=int(args.hist_bins),
    )
    hysteresis_plot_path = processed_dir / "hysteresis_output_angle_vs_input_attack_angle.png"
    save_hysteresis_output_angle_plot(
        hysteresis_plot_path,
        records=records,
        title_prefix="Checkerboard-referenced ",
    )
    line_hysteresis_paths = save_line_reference_hysteresis_plots(
        processed_dir=processed_dir,
        records=records,
        line_metrics=line_metrics,
        title_prefix="Checkerboard-referenced ",
    )
    offset_hysteresis_paths = save_offset_command_hysteresis_plots(
        processed_dir=processed_dir,
        records=records,
        offset_metrics=offset_metrics,
        title_prefix="Checkerboard-referenced ",
    )
    one_daq_hysteresis_paths = save_one_daq_all_metrics_plots(
        processed_dir=processed_dir,
        records=records,
        title_prefix="Checkerboard-referenced ",
    )

    fit_sweep_summary = None
    if (bool(getattr(args, "process_all_fits", False))
            or bool(getattr(args, "plot_all_calibration_models", False))
            or getattr(args, "processing_angle_fits", None) is not None
            or getattr(args, "processing_offset_fits", None) is not None
            or getattr(args, "processing_fit_branches", None) is not None):
        plot_all_cal_models = bool(getattr(args, "plot_all_calibration_models", False))
        base_branch = resolve_fit_branch_for_window(
            getattr(args, "fit_branch", "auto"),
            attack_min_deg=getattr(args, "attack_min_deg", None),
            attack_max_deg=getattr(args, "attack_max_deg", None),
        )
        default_modes = [
            "avg_pchip", "avg_cubic", "linear", "pull", "release", "phase_specific",
            "per_phase_pchip", "per_phase_cubic",
        ]
        if plot_all_cal_models and robot_cal_for_line is not None:
            discovered_default = discover_calibration_model_fit_modes(robot_cal_for_line, branch=base_branch)
            if discovered_default.get("fit_modes"):
                default_modes = list(discovered_default["fit_modes"])
                print(f"[INFO] Discovered calibration model fit modes for base branch {_fit_branch_label(base_branch)}: {default_modes}")
        angle_modes = _normalize_requested_fit_modes(
            getattr(args, "processing_angle_fits", None),
            fallback=(default_modes if (bool(getattr(args, "process_all_fits", False)) or plot_all_cal_models) else [str(getattr(args, "angle_fit", "avg_pchip"))]),
        )
        offset_modes = _normalize_requested_fit_modes(
            getattr(args, "processing_offset_fits", None),
            fallback=(angle_modes if (bool(getattr(args, "process_all_fits", False)) or plot_all_cal_models) else [str(getattr(args, "offset_fit", None) or getattr(args, "angle_fit", "avg_pchip"))]),
        )
        detected_branches = list(getattr(robot_cal_for_line, "available_fit_branches", []) or []) if robot_cal_for_line is not None else []
        fallback_branches = (detected_branches if (plot_all_cal_models and detected_branches) else [base_branch])
        branches = _normalize_processing_fit_branches(
            getattr(args, "processing_fit_branches", None),
            fallback=fallback_branches,
        )
        if not branches:
            branches = [None]
        print(
            "[INFO] Running one-DAQ fit sweep: "
            f"angle_fits={angle_modes}, offset_fits={offset_modes}, "
            f"branches={[ _fit_branch_label(b) for b in branches ]}, "
            f"combination_mode={getattr(args, 'fit_combination_mode', 'paired')}"
        )
        fit_sweep_summary = {
            "branches": [],
            "angle_fit_modes": angle_modes,
            "offset_fit_modes": offset_modes,
            "fit_combination_mode": str(getattr(args, "fit_combination_mode", "paired")),
        }
        original_branch = getattr(robot_cal_for_line, "active_fit_branch", None) if robot_cal_for_line is not None else None
        for branch in branches:
            if robot_cal_for_line is not None:
                robot_cal_for_line.active_fit_branch = normalize_fit_branch(branch)
            branch_label = _fit_branch_label(branch)
            branch_title = _fit_branch_title(branch)
            branch_output_root = str(Path(str(getattr(args, "fit_output_root", "fit_sweeps"))) / f"branch_{_sanitize_fit_label_for_path(branch_label)}")
            print(f"[INFO] Processing fit branch {branch_title}; outputs under {branch_output_root}")
            branch_angle_modes = list(angle_modes)
            branch_offset_modes = list(offset_modes)
            if plot_all_cal_models and robot_cal_for_line is not None and getattr(args, "processing_angle_fits", None) is None:
                discovered_branch = discover_calibration_model_fit_modes(robot_cal_for_line, branch=branch)
                if discovered_branch.get("fit_modes"):
                    branch_angle_modes = list(discovered_branch["fit_modes"])
                    if getattr(args, "processing_offset_fits", None) is None:
                        branch_offset_modes = list(discovered_branch["fit_modes"])
                    print(f"[INFO] Branch {branch_title}: discovered model columns: {branch_angle_modes}")
            branch_summary = save_one_daq_fit_sweep_outputs(
                processed_dir=processed_dir,
                records=records,
                robot_cal=robot_cal_for_line,
                angle_fit_modes=branch_angle_modes,
                offset_fit_modes=branch_offset_modes,
                flip_rz_sign=bool(getattr(args, "flip_rz_sign", False)),
                combination_mode=str(getattr(args, "fit_combination_mode", "paired")),
                output_root_name=branch_output_root,
                title_prefix="Checkerboard-referenced ",
            )
            fit_sweep_summary["branches"].append(branch_summary)
        if robot_cal_for_line is not None:
            robot_cal_for_line.active_fit_branch = original_branch

    annotate_analysis_output_images(processed_dir / "analysis_outputs", records)

    print(f"[INFO] Saved CSV: {csv_path}")
    print(f"[INFO] Saved metrics JSON: {metrics_json_path}")
    print(f"[INFO] Saved line metrics JSON: {line_metrics_json_path}")
    print(f"[INFO] Saved offset-command metrics JSON: {offset_metrics_json_path}")
    print(f"[INFO] Saved one-DAQ all-metrics JSON: {one_daq_metrics_json_path}")
    print(f"[INFO] Saved error plot: {error_plot_path}")
    print(f"[INFO] Saved B-pull error component plot: {bpull_error_plot_path}")
    print(f"[INFO] Saved histogram + per-C heatmap plot: {hist_hexbin_path}")
    print(f"[INFO] Saved hysteresis plot: {hysteresis_plot_path}")
    for _line_path in line_hysteresis_paths:
        print(f"[INFO] Saved line-reference hysteresis plot: {_line_path}")
    for _offset_path in offset_hysteresis_paths:
        print(f"[INFO] Saved offset-command hysteresis plot: {_offset_path}")
    for _one_daq_path in one_daq_hysteresis_paths:
        print(f"[INFO] Saved one-DAQ all-metrics plot: {_one_daq_path}")
    if fit_sweep_summary is not None:
        if "branches" in fit_sweep_summary:
            for _branch_summary in fit_sweep_summary.get("branches", []):
                print(f"[INFO] Saved one-DAQ fit sweep summary: {_branch_summary.get('summary_json')}")
                if _branch_summary.get("panel_plot"):
                    print(f"[INFO] Saved one-DAQ fit-sweep panel: {_branch_summary.get('panel_plot')}")
                for _run in _branch_summary.get("runs", []):
                    print(
                        "[INFO] Saved fit-sweep outputs: "
                        f"branch={_run.get('fit_branch')} angle={_run.get('angle_fit')} offset={_run.get('offset_fit')} -> {_run.get('output_dir')}"
                    )
        else:
            print(f"[INFO] Saved one-DAQ fit sweep summary: {fit_sweep_summary.get('summary_json')}")
            if fit_sweep_summary.get("panel_plot"):
                print(f"[INFO] Saved one-DAQ fit-sweep panel: {fit_sweep_summary.get('panel_plot')}")
            for _run in fit_sweep_summary.get("runs", []):
                print(
                    "[INFO] Saved fit-sweep outputs: "
                    f"angle={_run.get('angle_fit')} offset={_run.get('offset_fit')} -> {_run.get('output_dir')}"
                )
    print(f"[INFO] Updated analysis outputs with per-sample error titles: {processed_dir / 'analysis_outputs'}")

    if board_reference_debug_image is not None:
        try:
            annotated_board_path = processed_dir / "checkerboard_reference_annotated_analysis.png"
            draw_checkerboard_analysis_overlay(
                cal=cal,
                output_path=annotated_board_path,
                tracked_rows=tracked_rows,
                image_files=imgs,
                board_debug_image=board_reference_debug_image,
            )
            print(f"[INFO] Saved annotated checkerboard analysis image: {annotated_board_path}")
        except Exception as e:
            print(f"[WARN] Failed to create annotated checkerboard analysis image: {e}")

    print("\n========== RESULTS ==========")
    print(f"Valid samples: {global_metrics['num_samples']}")
    print(
        "Mean assigned per-C reference point: "
        f"u = {global_metrics['reference_point_mm']['u_mean_mm']:.6f} mm, "
        f"z = {global_metrics['reference_point_mm']['z_mean_mm']:.6f} mm"
    )
    print(f"Reference mode: {global_metrics['reference_mode']}")
    print(f"Global RMSE:   {global_metrics['rmse_mm']:.6f} mm")
    print(f"Mean error:    {global_metrics['mean_error_mm']:.6f} mm")
    print(f"Std error:     {global_metrics['std_error_mm']:.6f} mm")
    print(f"Median error:  {global_metrics['median_error_mm']:.6f} mm")
    print(f"Min error:     {global_metrics['min_error_mm']:.6f} mm")
    print(f"Max error:     {global_metrics['max_error_mm']:.6f} mm")

    print("\nPer-C RMSE (to the corresponding per-sample reference points):")
    if not per_orientation_metrics:
        print("  No valid per-orientation groups found.")
    else:
        for _, pm in sorted(
            per_orientation_metrics.items(),
            key=lambda kv: (999999.0 if kv[1]["c_orientation_deg"] is None else kv[1]["c_orientation_deg"])
        ):
            c_val = pm["c_orientation_deg"]
            c_label = "unknown" if c_val is None else f"{float(c_val):.3f} deg"
            print(
                f"  C = {c_label:<12} "
                f"RMSE = {pm['rmse_mm']:.6f} mm   "
                f"N = {pm['num_samples']}"
            )
    print("=============================\n")

    print("[DONE] Offline fixed-tip dual-C checkerboard tip error analysis complete.")
    print(f"Outputs are in: {processed_dir}")





# =============================================================================
# Unified angle-command acquisition + optional checkerboard processing additions
# =============================================================================

ANGLE_FIT_CHOICES = (
    "avg_pchip", "pchip_avg",
    "avg_cubic", "cubic_avg",
    "linear",
    "pull", "release",
    "phase_specific",
    "per_phase_pchip",
    "per_phase_cubic",
)

_BASE_LOAD_CALIBRATION = load_calibration

def load_calibration(json_path: str, y_offset_fit: str = DEFAULT_Y_OFFSET_FIT) -> Calibration:
    """Load calibration and retain the raw JSON fit models for unified angle-fit selection."""
    cal = _BASE_LOAD_CALIBRATION(json_path, y_offset_fit=y_offset_fit)
    try:
        with Path(json_path).open("r") as f:
            raw = json.load(f)
        cal.raw_calibration_json = raw
        cal.raw_fit_models = raw.get("fit_models", {}) or {}
        cal.raw_fit_models_by_phase = raw.get("fit_models_by_phase", {}) or {}
        cal.raw_shared_aux_fit_models = raw.get("shared_aux_fit_models", {}) or {}
        cal.raw_cubic_coefficients = raw.get("cubic_coefficients", {}) or {}
        cal.available_fit_branches = detect_fit_branches_in_calibration_raw(raw)
        cal.active_fit_branch = None
    except Exception:
        cal.raw_calibration_json = {}
        cal.raw_fit_models = {}
        cal.raw_fit_models_by_phase = {}
        cal.raw_shared_aux_fit_models = {}
        cal.raw_cubic_coefficients = {}
        cal.available_fit_branches = []
        cal.active_fit_branch = None
    return cal

def _model_summary(model: Any) -> str:
    if model is None:
        return "None"
    if not isinstance(model, dict):
        return str(type(model))
    bits = [str(model.get("model_type", "unknown"))]
    if model.get("equation"):
        bits.append(str(model.get("equation")))
    elif model.get("coefficients") is not None:
        bits.append(f"coefficients={np.asarray(model.get('coefficients')).reshape(-1).tolist()}")
    elif model.get("x_knots") is not None:
        bits.append(f"knots={len(model.get('x_knots'))}")
    return " | ".join(bits)

def _first_existing_model(*models):
    for m in models:
        if m is not None:
            return m
    return None


def _alnum_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").strip().lower())


def normalize_fit_branch(value: Any) -> Optional[str]:
    """Normalize calibration branch labels such as 0-90, 0_90, 0to90."""
    if value is None:
        return None
    txt = str(value).strip().lower()
    if txt in ("", "auto", "default", "none", "global", "legacy", "all"):
        return None
    key = _alnum_key(txt)
    # Prefer explicit 0-180 over a generic 180 token.
    if key in ("0180", "0to180", "zeroto180", "range0180", "branch0180", "deg0180") or ("180" in key):
        return "0_180"
    if key in ("090", "0to90", "zeroto90", "range090", "branch090", "deg090") or ("90" in key):
        return "0_90"
    return txt.replace("-", "_").replace(" ", "_")


def _fit_branch_label(branch: Any) -> str:
    b = normalize_fit_branch(branch)
    return b if b is not None else "global"


def _fit_branch_title(branch: Any) -> str:
    b = normalize_fit_branch(branch)
    if b == "0_90":
        return "0–90°"
    if b == "0_180":
        return "0–180°"
    return "global/default"


def _branch_token_set(branch: Any) -> Set[str]:
    b = normalize_fit_branch(branch)
    if b is None:
        return set()
    if b == "0_90":
        return {"090", "0to90", "zeroto90", "range090", "branch090", "angle090", "attack090", "sweep090", "fit090", "90"}
    if b == "0_180":
        return {"0180", "0to180", "zeroto180", "range0180", "branch0180", "angle0180", "attack0180", "sweep0180", "fit0180", "180"}
    return {_alnum_key(b)}


def _branch_matches_key(key: Any, branch: Any) -> bool:
    b = normalize_fit_branch(branch)
    if b is None:
        return True
    k = _alnum_key(key)
    toks = _branch_token_set(b)
    return any(t and t in k for t in toks)


def _phase_matches_key(key: Any, phase: Optional[str]) -> bool:
    p = _normalize_motion_phase_name(phase)
    if p is None:
        return True
    k = _alnum_key(key)
    if p.startswith("pull"):
        return "pull" in k and "release" not in k
    if p.startswith("release"):
        return "release" in k
    return _alnum_key(p) in k


def _dict_model_by_candidate_keys(models: Any, candidate_keys: List[str], branch: Any = None) -> Optional[Any]:
    """Return a model from a model dictionary using exact, normalized, then fuzzy branch-aware keys."""
    if not isinstance(models, dict):
        return None
    # Exact key pass.
    for key in candidate_keys:
        if key in models and isinstance(models.get(key), dict):
            return models.get(key)
    # Normalized exact pass.
    norm_map = {_alnum_key(k): k for k in models.keys()}
    for key in candidate_keys:
        nk = _alnum_key(key)
        if nk in norm_map and isinstance(models.get(norm_map[nk]), dict):
            return models.get(norm_map[nk])
    # Fuzzy token pass with branch preference.
    norm_candidates = [_alnum_key(k) for k in candidate_keys]
    for mk, mv in models.items():
        if not isinstance(mv, dict):
            continue
        nmk = _alnum_key(mk)
        if normalize_fit_branch(branch) is not None and not _branch_matches_key(mk, branch):
            # Also allow branch metadata/equation/value_name matches, not only key names.
            hay = " ".join(str(mv.get(x, "")) for x in ("value_name", "fit_branch", "angle_range", "range_label", "sweep_range"))
            if not _branch_matches_key(hay, branch):
                continue
        for nc in norm_candidates:
            if nc and (nmk == nc or nc in nmk or nmk.endswith(nc)):
                return mv
    return None


def _model_keys_for(base: str, kind: str, branch: Any = None) -> List[str]:
    """Generate likely keys for base model name and fit kind."""
    base = str(base)
    kind = str(kind or "").strip().lower()
    branch_labels = []
    b = normalize_fit_branch(branch)
    if b == "0_90":
        branch_labels = ["0_90", "0-90", "0to90", "90", "range_0_90", "branch_0_90"]
    elif b == "0_180":
        branch_labels = ["0_180", "0-180", "0to180", "180", "range_0_180", "branch_0_180"]
    keys: List[str] = []
    suffixes = []
    if kind == "pchip":
        suffixes = ["pchip", "", "avg_pchip", "pchip_avg"]
    elif kind == "avg_pchip":
        suffixes = ["avg_pchip", "pchip_avg", "pchip", ""]
    elif kind == "cubic":
        suffixes = ["cubic", "avg_cubic", "cubic_avg"]
    elif kind == "avg_cubic":
        suffixes = ["avg_cubic", "cubic_avg", "cubic"]
    elif kind == "linear":
        suffixes = ["avg_linear", "linear"]
    elif kind in ("any", "phase_any"):
        suffixes = ["", "pchip", "avg_pchip", "cubic", "avg_cubic", "linear"]
    else:
        suffixes = [kind, ""]
    for suff in suffixes:
        keys.append(base if suff == "" else f"{base}_{suff}")
    for bl in branch_labels:
        for suff in suffixes:
            keys.extend([
                f"{base}_{bl}" if suff == "" else f"{base}_{bl}_{suff}",
                f"{base}_{suff}_{bl}" if suff else f"{base}_{bl}",
                f"{bl}_{base}" if suff == "" else f"{bl}_{base}_{suff}",
            ])
    # Preserve order and uniqueness.
    out: List[str] = []
    for k in keys:
        if k not in out:
            out.append(k)
    return out


def _raw_branch_containers(cal: Calibration, branch: Any) -> List[Tuple[str, dict]]:
    raw = getattr(cal, "raw_calibration_json", {}) or {}
    out: List[Tuple[str, dict]] = []
    possible_top_keys = [
        "fit_models_by_branch", "fit_models_by_angle_branch", "fit_models_by_angle_range",
        "fit_models_by_range", "fit_models_by_sweep_range", "fit_models_by_attack_range",
        "fit_models_by_tip_angle_range", "fit_models_by_window", "branch_fit_models",
        "range_fit_models", "angle_range_fit_models", "exported_models_by_branch",
    ]
    for top_key in possible_top_keys:
        top = raw.get(top_key)
        if isinstance(top, dict):
            for bk, bv in top.items():
                if _branch_matches_key(bk, branch) and isinstance(bv, dict):
                    out.append((f"{top_key}/{bk}", bv))
    # exported_models sometimes stores named branches.
    exported = raw.get("exported_models")
    if isinstance(exported, dict):
        for bk, bv in exported.items():
            if _branch_matches_key(bk, branch) and isinstance(bv, dict):
                out.append((f"exported_models/{bk}", bv))
    return out


def _candidate_model_groups(cal: Calibration, phase: Optional[str] = None, branch: Any = None) -> List[Tuple[str, dict]]:
    """Collect possible model dictionaries, ordered branch-specific first, then phase, then global."""
    groups: List[Tuple[str, dict]] = []
    branch_norm = normalize_fit_branch(branch if branch is not None else getattr(cal, "active_fit_branch", None))
    phase_norm = _normalize_motion_phase_name(phase)

    # 1) Explicit branch containers, with nested phase maps preferred.
    if branch_norm is not None:
        for label, container in _raw_branch_containers(cal, branch_norm):
            fbyp = container.get("fit_models_by_phase") if isinstance(container, dict) else None
            if isinstance(fbyp, dict):
                for pk, pv in fbyp.items():
                    if isinstance(pv, dict) and _phase_matches_key(pk, phase_norm):
                        groups.append((f"{label}/fit_models_by_phase/{pk}", pv))
            for phase_key_name in ("phases", "phase_models", "models_by_phase"):
                phmap = container.get(phase_key_name) if isinstance(container, dict) else None
                if isinstance(phmap, dict):
                    for pk, pv in phmap.items():
                        if isinstance(pv, dict) and _phase_matches_key(pk, phase_norm):
                            groups.append((f"{label}/{phase_key_name}/{pk}", pv))
            # Direct models container.
            fm = container.get("fit_models") if isinstance(container, dict) else None
            if isinstance(fm, dict):
                groups.append((f"{label}/fit_models", fm))
            # Some branch containers are already model dictionaries.
            if any(isinstance(container.get(k), dict) for k in ("r", "z", "tip_angle", "r_pchip", "z_pchip", "tip_angle_pchip")):
                groups.append((label, container))

    # 2) Phase dictionaries whose names carry the branch token, e.g. pull_0_90.
    raw_phase_models = getattr(cal, "raw_fit_models_by_phase", {}) or {}
    phase_items = []
    if isinstance(raw_phase_models, dict):
        phase_items.extend(raw_phase_models.items())
    phase_items.extend((getattr(cal, "phase_models", {}) or {}).items())
    seen_labels: Set[str] = set()
    for pk, pv in phase_items:
        if not isinstance(pv, dict):
            continue
        if not _phase_matches_key(pk, phase_norm):
            continue
        if branch_norm is not None and not _branch_matches_key(pk, branch_norm):
            # Branch can also be encoded in model value names/equations.
            hay = " ".join(" ".join(str(v.get(x, "")) for x in ("value_name", "fit_branch", "angle_range", "range_label", "sweep_range")) for v in pv.values() if isinstance(v, dict))
            if not _branch_matches_key(hay, branch_norm):
                continue
        lab = f"phase/{pk}"
        if lab not in seen_labels:
            groups.append((lab, pv)); seen_labels.add(lab)

    # 3) Generic phase dictionaries as fallback.
    for pk, pv in (getattr(cal, "phase_models", {}) or {}).items():
        if isinstance(pv, dict) and _phase_matches_key(pk, phase_norm):
            lab = f"phase_fallback/{pk}"
            if lab not in seen_labels:
                groups.append((lab, pv)); seen_labels.add(lab)

    # 4) Global/shared models last.
    for label, models in [
        ("global_fit_models", getattr(cal, "raw_fit_models", {}) or {}),
        ("shared_aux_fit_models", getattr(cal, "raw_shared_aux_fit_models", {}) or {}),
    ]:
        if isinstance(models, dict):
            groups.append((label, models))
    return groups


def _select_model_from_groups(cal: Calibration, base: str, kind: str, phase: Optional[str] = None, branch: Any = None) -> Tuple[Optional[Any], str]:
    branch_norm = normalize_fit_branch(branch if branch is not None else getattr(cal, "active_fit_branch", None))
    groups = _candidate_model_groups(cal, phase=phase, branch=branch_norm)
    for label, models in groups:
        model = _dict_model_by_candidate_keys(models, _model_keys_for(base, kind, branch_norm), branch=branch_norm)
        if model is not None:
            branch_label = _fit_branch_label(branch_norm)
            phase_label = _normalize_motion_phase_name(phase) or "global"
            return model, f"{label} | branch={branch_label} | phase={phase_label} | {base}_{kind}"
    return None, f"no-model-found branch={_fit_branch_label(branch_norm)} phase={phase} {base}_{kind}"


def detect_fit_branches_in_calibration_raw(raw: dict) -> List[str]:
    """
    Detect only calibration-fit branch labels, not C-axis orientation labels.
    We inspect fit-model containers/keys and model value/equation metadata, but
    avoid datasets/orientation sections where C90/C180 can appear.
    """
    found: List[str] = []
    def add(x):
        b = normalize_fit_branch(x)
        if b is not None and b not in found:
            found.append(b)
    branch_map_keys = [
        "fit_models_by_branch", "fit_models_by_angle_branch", "fit_models_by_angle_range",
        "fit_models_by_range", "fit_models_by_sweep_range", "fit_models_by_attack_range",
        "fit_models_by_tip_angle_range", "fit_models_by_window", "branch_fit_models",
        "range_fit_models", "angle_range_fit_models", "exported_models_by_branch",
    ]
    for top_key in branch_map_keys:
        top = raw.get(top_key) if isinstance(raw, dict) else None
        if isinstance(top, dict):
            for bk in top.keys():
                if _branch_matches_key(bk, "0_90"):
                    add("0_90")
                if _branch_matches_key(bk, "0_180"):
                    add("0_180")

    def scan_model_container(container: Any):
        if not isinstance(container, dict):
            return
        for k, v in container.items():
            # Skip pure C-orientation model/dataset labels.
            kl = str(k).lower()
            if kl in ("c90", "c180", "orientation_90", "orientation_180"):
                continue
            hay = str(k)
            if isinstance(v, dict):
                hay += " " + " ".join(str(v.get(x, "")) for x in ("value_name", "fit_branch", "range_label", "angle_range", "sweep_range"))
            if _branch_matches_key(hay, "0_90"):
                add("0_90")
            if _branch_matches_key(hay, "0_180"):
                add("0_180")

    scan_model_container(raw.get("fit_models", {}) if isinstance(raw, dict) else {})
    scan_model_container(raw.get("shared_aux_fit_models", {}) if isinstance(raw, dict) else {})
    fbyp = raw.get("fit_models_by_phase", {}) if isinstance(raw, dict) else {}
    if isinstance(fbyp, dict):
        for pk, pv in fbyp.items():
            # Phase keys like pull_0_90/release_0_180 are valid branch labels.
            if _branch_matches_key(pk, "0_90"):
                add("0_90")
            if _branch_matches_key(pk, "0_180"):
                add("0_180")
            scan_model_container(pv)
    return found


def resolve_fit_branch_for_window(branch_arg: Any, attack_min_deg: Optional[float] = None, attack_max_deg: Optional[float] = None) -> Optional[str]:
    txt = str(branch_arg or "auto").strip().lower()
    if txt in ("", "auto", "by_window", "from_window"):
        if attack_max_deg is not None and float(attack_max_deg) <= 90.000001:
            return "0_90"
        if attack_max_deg is not None and float(attack_max_deg) > 90.000001:
            return "0_180"
        return None
    return normalize_fit_branch(txt)


def _normalize_processing_fit_branches(value: Any, fallback: Optional[List[Optional[str]]] = None) -> List[Optional[str]]:
    if value is None:
        return list(fallback or [])
    if isinstance(value, str):
        raw_parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    else:
        raw_parts = []
        for item in value:
            if item is None:
                continue
            raw_parts.extend([p.strip() for p in str(item).replace(";", ",").split(",") if p.strip()])
    out: List[Optional[str]] = []
    for part in raw_parts:
        if str(part).strip().lower() == "all":
            for b in ("0_90", "0_180"):
                if b not in out:
                    out.append(b)
            continue
        b = normalize_fit_branch(part)
        if b not in out:
            out.append(b)
    return out or list(fallback or [])

def select_tip_angle_model_for_mode(cal: Calibration, angle_fit: str, phase: Optional[str] = None):
    """
    Select B->output-angle fit model.

    Existing modes are preserved. Additional modes:
      per_phase_pchip : choose the PCHIP model from the current pull/release phase,
                        honoring cal.active_fit_branch when the calibration provides
                        0-90 or 0-180 branch-specific models.
      per_phase_cubic : same, but choose cubic/polynomial phase models.

    Branch selection is controlled by cal.active_fit_branch. Use --fit-branch for
    acquisition/ordinary processing and --processing-fit-branches for sweep panels.
    """
    mode = str(angle_fit or "avg_pchip").strip().lower()
    if mode == "pchip_avg":
        mode = "avg_pchip"
    if mode == "cubic_avg":
        mode = "avg_cubic"

    branch = normalize_fit_branch(getattr(cal, "active_fit_branch", None))
    global_models = getattr(cal, "raw_fit_models", {}) or {}
    shared_models = getattr(cal, "raw_shared_aux_fit_models", {}) or {}
    cubic = getattr(cal, "raw_cubic_coefficients", {}) or {}

    if mode == "phase_specific":
        mode = _normalize_motion_phase_name(phase) or "pull"

    if mode == "per_phase_pchip":
        phase_name = _normalize_motion_phase_name(phase) or "pull"
        model, label = _select_model_from_groups(cal, "tip_angle", "pchip", phase=phase_name, branch=branch)
        if model is not None:
            return model, f"per-phase-pchip | {label}"
        model, label = _select_model_from_groups(cal, "tip_angle", "any", phase=phase_name, branch=branch)
        if model is not None:
            return model, f"per-phase-pchip fallback-any | {label}"

    if mode == "per_phase_cubic":
        phase_name = _normalize_motion_phase_name(phase) or "pull"
        model, label = _select_model_from_groups(cal, "tip_angle", "cubic", phase=phase_name, branch=branch)
        if model is not None:
            return model, f"per-phase-cubic | {label}"
        model, label = _select_model_from_groups(cal, "tip_angle", "any", phase=phase_name, branch=branch)
        if model is not None:
            return model, f"per-phase-cubic fallback-any | {label}"

    phase_name = _normalize_motion_phase_name(mode)
    if phase_name in ("pull", "release"):
        model, label = _select_model_from_groups(cal, "tip_angle", "any", phase=phase_name, branch=branch)
        if model is not None:
            return model, f"{phase_name}-specific | {label}"

    if mode == "avg_pchip":
        # Prefer branch-specific average PCHIP if it exists, then global averages.
        model, label = _select_model_from_groups(cal, "tip_angle", "avg_pchip", phase=None, branch=branch)
        if model is not None:
            return model, f"average-pchip | {label}"
        model = _first_existing_model(
            global_models.get("tip_angle_avg_pchip"),
            global_models.get("tip_angle_pchip_avg"),
            global_models.get("tip_angle_pchip"),
            global_models.get("tip_angle"),
            shared_models.get("tip_angle_avg_pchip") if isinstance(shared_models, dict) else None,
            cal.tip_angle_model,
        )
        return model, f"average-pchip global/fallback branch={_fit_branch_label(branch)}"

    if mode == "avg_cubic":
        model, label = _select_model_from_groups(cal, "tip_angle", "avg_cubic", phase=None, branch=branch)
        if model is not None:
            return model, f"average-cubic | {label}"
        model = _first_existing_model(
            global_models.get("tip_angle_avg_cubic"),
            global_models.get("tip_angle_cubic_avg"),
            global_models.get("tip_angle_cubic"),
            shared_models.get("tip_angle_avg_cubic") if isinstance(shared_models, dict) else None,
            legacy_poly_model(
                cubic.get("tip_angle_cubic_coeffs") or cubic.get("tip_angle_coeffs"),
                cubic.get("tip_angle_cubic_equation") or cubic.get("tip_angle_equation"),
                "tip_angle_deg",
            ),
            cal.tip_angle_model,
        )
        return model, f"average-cubic global/fallback branch={_fit_branch_label(branch)}"

    if mode == "linear":
        model, label = _select_model_from_groups(cal, "tip_angle", "linear", phase=None, branch=branch)
        if model is not None:
            return model, f"linear-explicit | {label}"
        explicit = _first_existing_model(
            global_models.get("tip_angle_linear"),
            global_models.get("tip_angle_avg_linear"),
        )
        if explicit is not None:
            return explicit, f"linear-explicit global branch={_fit_branch_label(branch)}"
        # If no explicit linear model survived loading, build a linear approximation
        # over the calibrated B range from the default tip-angle evaluator.
        b = np.linspace(float(cal.b_min), float(cal.b_max), 400)
        a = eval_tip_angle_deg(cal, b)
        coeff = np.polyfit(b, a, 1).tolist()
        return {
            "model_type": "polynomial",
            "basis": "monomial",
            "degree": 1,
            "input_axis": "b_motor",
            "value_name": "tip_angle_deg",
            "coefficients": coeff,
            "equation": f"linear least-squares fallback over B=[{cal.b_min:.6g},{cal.b_max:.6g}]",
        }, f"linear-fallback branch={_fit_branch_label(branch)}"

    # Final fallback.
    return cal.tip_angle_model, f"{mode}/fallback-tip_angle branch={_fit_branch_label(branch)}"

def eval_tip_angle_deg_with_model(model: Any, b: Any) -> np.ndarray:
    return evaluate_fit_model(model, b)

def build_tip_angle_inverse_table_for_fit(
    cal: Calibration,
    angle_fit: str = "avg_pchip",
    motion_phase: Optional[str] = None,
    num_samples: int = DEFAULT_CUSTOM_INV_SAMPLES,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    model, model_label = select_tip_angle_model_for_mode(cal, angle_fit, phase=motion_phase)
    if model is None:
        raise ValueError("This angle-command mode requires a usable tip-angle model in the calibration JSON.")
    ns = max(1000, int(num_samples))
    b_samples = np.linspace(float(cal.b_min), float(cal.b_max), ns, dtype=float)
    angle_samples = np.asarray(eval_tip_angle_deg_with_model(model, b_samples), dtype=float)

    finite = np.isfinite(angle_samples) & np.isfinite(b_samples)
    if not np.any(finite):
        raise ValueError(f"Tip-angle model produced no finite values for angle_fit={angle_fit}.")
    b_samples = b_samples[finite]
    angle_samples = angle_samples[finite]

    order = np.argsort(angle_samples)
    angle_sorted = angle_samples[order]
    b_sorted = b_samples[order]
    angle_unique, unique_idx = np.unique(angle_sorted, return_index=True)
    b_unique = b_sorted[unique_idx]
    if angle_unique.size < 2:
        raise ValueError(f"Could not invert angle fit {angle_fit}; it produced fewer than two unique angles.")
    meta = {
        "angle_fit_requested": str(angle_fit),
        "fit_branch": _fit_branch_label(getattr(cal, "active_fit_branch", None)),
        "motion_phase": motion_phase,
        "selected_model_label": model_label,
        "selected_model_summary": _model_summary(model),
        "available_tip_angle_range_deg": [float(angle_unique[0]), float(angle_unique[-1])],
        "b_motor_range": [float(np.min(b_samples)), float(np.max(b_samples))],
    }
    return angle_unique, b_unique, meta

def angle_to_b_for_fit(
    cal: Calibration,
    requested_angle_deg: float,
    angle_fit: str,
    motion_phase: Optional[str],
    inverse_samples: int,
) -> Tuple[float, float, dict]:
    angle_table, b_table, meta = build_tip_angle_inverse_table_for_fit(
        cal=cal,
        angle_fit=angle_fit,
        motion_phase=motion_phase,
        num_samples=inverse_samples,
    )
    amin = float(angle_table[0])
    amax = float(angle_table[-1])
    used_angle = clamp(float(requested_angle_deg), amin, amax)
    b_cmd = float(np.interp(used_angle, angle_table, b_table))
    return b_cmd, used_angle, meta



def select_output_offset_model_for_mode(
    cal: Calibration,
    output_axis: str,
    fit_mode: str,
    phase: Optional[str] = None,
):
    """
    Select a B->offset model for fixed-stage offset/one-DAQ metrics.

    Additional modes:
      per_phase_pchip : phase-specific PCHIP r/z model, honoring cal.active_fit_branch.
      per_phase_cubic : phase-specific cubic r/z model, honoring cal.active_fit_branch.
    """
    axis = str(output_axis or "radial").strip().lower()
    if axis in ("horizontal", "x", "r"):
        axis = "radial"
    if axis in ("vertical",):
        axis = "z"
    if axis not in ("radial", "z"):
        raise ValueError(f"Unsupported output offset axis: {output_axis}")

    model_name = "r" if axis == "radial" else "z"
    fallback_model = cal.r_model if axis == "radial" else cal.z_model

    mode = str(fit_mode or "avg_pchip").strip().lower()
    if mode == "pchip_avg":
        mode = "avg_pchip"
    if mode == "cubic_avg":
        mode = "avg_cubic"
    if mode == "phase_specific":
        mode = _normalize_motion_phase_name(phase) or "pull"

    branch = normalize_fit_branch(getattr(cal, "active_fit_branch", None))
    global_models = getattr(cal, "raw_fit_models", {}) or {}
    shared_models = getattr(cal, "raw_shared_aux_fit_models", {}) or {}
    cubic = getattr(cal, "raw_cubic_coefficients", {}) or {}

    if mode == "per_phase_pchip":
        phase_name = _normalize_motion_phase_name(phase) or "pull"
        model, label = _select_model_from_groups(cal, model_name, "pchip", phase=phase_name, branch=branch)
        if model is not None:
            return model, f"per-phase-pchip-{model_name} | {label}"
        model, label = _select_model_from_groups(cal, model_name, "any", phase=phase_name, branch=branch)
        if model is not None:
            return model, f"per-phase-pchip-{model_name} fallback-any | {label}"

    if mode == "per_phase_cubic":
        phase_name = _normalize_motion_phase_name(phase) or "pull"
        model, label = _select_model_from_groups(cal, model_name, "cubic", phase=phase_name, branch=branch)
        if model is not None:
            return model, f"per-phase-cubic-{model_name} | {label}"
        model, label = _select_model_from_groups(cal, model_name, "any", phase=phase_name, branch=branch)
        if model is not None:
            return model, f"per-phase-cubic-{model_name} fallback-any | {label}"

    phase_name = _normalize_motion_phase_name(mode)
    if phase_name in ("pull", "release"):
        model, label = _select_model_from_groups(cal, model_name, "any", phase=phase_name, branch=branch)
        if model is not None:
            return model, f"{phase_name}-specific-{model_name} | {label}"

    if mode == "avg_pchip":
        model, label = _select_model_from_groups(cal, model_name, "avg_pchip", phase=None, branch=branch)
        if model is not None:
            return model, f"average-pchip-{model_name} | {label}"
        model = _first_existing_model(
            global_models.get(f"{model_name}_avg_pchip"),
            global_models.get(f"{model_name}_pchip_avg"),
            global_models.get(f"{model_name}_pchip"),
            global_models.get(model_name),
            shared_models.get(f"{model_name}_avg_pchip") if isinstance(shared_models, dict) else None,
            fallback_model,
        )
        return model, f"average-pchip-{model_name} global/fallback branch={_fit_branch_label(branch)}"

    if mode == "avg_cubic":
        model, label = _select_model_from_groups(cal, model_name, "avg_cubic", phase=None, branch=branch)
        if model is not None:
            return model, f"average-cubic-{model_name} | {label}"
        model = _first_existing_model(
            global_models.get(f"{model_name}_avg_cubic"),
            global_models.get(f"{model_name}_cubic_avg"),
            global_models.get(f"{model_name}_cubic"),
            shared_models.get(f"{model_name}_avg_cubic") if isinstance(shared_models, dict) else None,
            legacy_poly_model(
                cubic.get(f"{model_name}_cubic_coeffs") or cubic.get(f"{model_name}_coeffs"),
                cubic.get(f"{model_name}_cubic_equation") or cubic.get(f"{model_name}_equation"),
                f"{model_name}_mm",
            ),
            fallback_model,
        )
        return model, f"average-cubic-{model_name} global/fallback branch={_fit_branch_label(branch)}"

    if mode == "linear":
        model, label = _select_model_from_groups(cal, model_name, "linear", phase=None, branch=branch)
        if model is not None:
            return model, f"linear-explicit-{model_name} | {label}"
        explicit = _first_existing_model(
            global_models.get(f"{model_name}_linear"),
            global_models.get(f"{model_name}_avg_linear"),
        )
        if explicit is not None:
            return explicit, f"linear-explicit-{model_name} global branch={_fit_branch_label(branch)}"
        b = np.linspace(float(cal.b_min), float(cal.b_max), 400)
        vals = evaluate_fit_model(fallback_model, b)
        coeff = np.polyfit(b, vals, 1).tolist()
        return {
            "model_type": "polynomial",
            "basis": "monomial",
            "degree": 1,
            "input_axis": "b_motor",
            "value_name": f"{model_name}_mm",
            "coefficients": coeff,
            "equation": f"linear least-squares fallback for {model_name} over B=[{cal.b_min:.6g},{cal.b_max:.6g}]",
        }, f"linear-fallback-{model_name} branch={_fit_branch_label(branch)}"

    return fallback_model, f"{mode}/fallback-{model_name} branch={_fit_branch_label(branch)}"

def eval_output_offset_with_model(
    cal: Calibration,
    model: Any,
    output_axis: str,
    b: Any,
    flip_rz_sign: bool = False,
) -> np.ndarray:
    """Evaluate the selected offset model with the same r sign convention used by eval_r."""
    axis = str(output_axis or "radial").strip().lower()
    if axis in ("horizontal", "x", "r"):
        axis = "radial"
    vals = evaluate_fit_model(model, b)
    if axis == "radial":
        # Match eval_r's convention for the radial model sign.
        sgn = -1.0 * (-1.0 if bool(flip_rz_sign) else 1.0)
        return sgn * np.asarray(vals, dtype=float)
    if axis in ("z", "vertical"):
        return np.asarray(vals, dtype=float)
    raise ValueError(f"Unsupported output offset axis: {output_axis}")


def build_output_offset_inverse_table_for_fit(
    cal: Calibration,
    output_axis: str,
    fit_mode: str = "avg_pchip",
    motion_phase: Optional[str] = None,
    num_samples: int = DEFAULT_CUSTOM_INV_SAMPLES,
    flip_rz_sign: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    model, model_label = select_output_offset_model_for_mode(
        cal=cal,
        output_axis=output_axis,
        fit_mode=fit_mode,
        phase=motion_phase,
    )
    if model is None:
        raise ValueError(f"The {output_axis} offset-command mode requires a usable offset model in the calibration JSON.")

    ns = max(1000, int(num_samples))
    b_samples = np.linspace(float(cal.b_min), float(cal.b_max), ns, dtype=float)
    val_samples = np.asarray(
        eval_output_offset_with_model(
            cal=cal,
            model=model,
            output_axis=output_axis,
            b=b_samples,
            flip_rz_sign=bool(flip_rz_sign),
        ),
        dtype=float,
    )

    finite = np.isfinite(val_samples) & np.isfinite(b_samples)
    if not np.any(finite):
        raise ValueError(f"Offset model produced no finite values for fit_mode={fit_mode}, output_axis={output_axis}.")
    b_samples = b_samples[finite]
    val_samples = val_samples[finite]

    order = np.argsort(val_samples)
    val_sorted = val_samples[order]
    b_sorted = b_samples[order]
    val_unique, unique_idx = np.unique(val_sorted, return_index=True)
    b_unique = b_sorted[unique_idx]
    if val_unique.size < 2:
        raise ValueError(f"Could not invert {output_axis} offset fit {fit_mode}; fewer than two unique offset values.")

    axis_label = "radial" if str(output_axis).lower() in ("radial", "horizontal", "x", "r") else "z"
    meta = {
        "offset_fit_requested": str(fit_mode),
        "fit_branch": _fit_branch_label(getattr(cal, "active_fit_branch", None)),
        "output_offset_axis": axis_label,
        "motion_phase": motion_phase,
        "selected_model_label": model_label,
        "selected_model_summary": _model_summary(model),
        "available_offset_range_mm": [float(val_unique[0]), float(val_unique[-1])],
        "b_motor_range": [float(np.min(b_samples)), float(np.max(b_samples))],
    }
    return val_unique, b_unique, meta


def output_offset_to_b_for_fit(
    cal: Calibration,
    requested_offset_mm: float,
    output_axis: str,
    fit_mode: str,
    motion_phase: Optional[str],
    inverse_samples: int,
    flip_rz_sign: bool = False,
) -> Tuple[float, float, dict]:
    val_table, b_table, meta = build_output_offset_inverse_table_for_fit(
        cal=cal,
        output_axis=output_axis,
        fit_mode=fit_mode,
        motion_phase=motion_phase,
        num_samples=inverse_samples,
        flip_rz_sign=bool(flip_rz_sign),
    )
    vmin = float(val_table[0])
    vmax = float(val_table[-1])
    used_val = clamp(float(requested_offset_mm), vmin, vmax)
    b_cmd = float(np.interp(used_val, val_table, b_table))
    return b_cmd, used_val, meta


def _resolve_requested_offset_range(
    cal: Calibration,
    output_axis: str,
    fit_mode: str,
    offset_min_mm: Optional[float],
    offset_max_mm: Optional[float],
    inverse_samples: int,
    flip_rz_sign: bool = False,
) -> Tuple[float, float, Tuple[float, float], List[dict]]:
    """Resolve desired offset command range. None means use the calibrated common pull/release range."""
    fit = str(fit_mode or "avg_pchip").strip().lower()
    pull_fit = "pull" if fit == "phase_specific" else fit
    rel_fit = "release" if fit == "phase_specific" else fit

    pull_table, _, pull_meta = build_output_offset_inverse_table_for_fit(
        cal=cal,
        output_axis=output_axis,
        fit_mode=pull_fit,
        motion_phase="pull",
        num_samples=inverse_samples,
        flip_rz_sign=bool(flip_rz_sign),
    )
    rel_table, _, rel_meta = build_output_offset_inverse_table_for_fit(
        cal=cal,
        output_axis=output_axis,
        fit_mode=rel_fit,
        motion_phase="release",
        num_samples=inverse_samples,
        flip_rz_sign=bool(flip_rz_sign),
    )
    avail_min = max(float(pull_table[0]), float(rel_table[0]))
    avail_max = min(float(pull_table[-1]), float(rel_table[-1]))
    if avail_min > avail_max:
        # Fall back to the union if pull/release fits do not overlap.
        avail_min = min(float(pull_table[0]), float(rel_table[0]))
        avail_max = max(float(pull_table[-1]), float(rel_table[-1]))

    req_min = avail_min if offset_min_mm is None else float(offset_min_mm)
    req_max = avail_max if offset_max_mm is None else float(offset_max_mm)
    used_min = clamp(req_min, avail_min, avail_max)
    used_max = clamp(req_max, avail_min, avail_max)
    if used_min > used_max:
        used_min, used_max = used_max, used_min
    return float(used_min), float(used_max), (float(avail_min), float(avail_max)), [pull_meta, rel_meta]


def _offset_value_profile(min_mm: float, max_mm: float, samples_per_leg: int) -> List[Tuple[float, str, float]]:
    n = max(2, int(samples_per_leg))
    pull = np.linspace(float(min_mm), float(max_mm), n, dtype=float)
    release = np.linspace(float(max_mm), float(min_mm), n, dtype=float)
    out: List[Tuple[float, str, float]] = []
    for i, v in enumerate(pull):
        out.append((float(v), "pull", i / float(n - 1)))
    for i, v in enumerate(release[1:], start=1):
        out.append((float(v), "release", i / float(n - 1)))
    return out


def generate_fixed_stage_offset_command_trajectory(
    cal: Calibration,
    base_stage_xyz: np.ndarray,
    c_deg: float,
    offset_mode: str,
    offset_min_mm: Optional[float] = None,
    offset_max_mm: Optional[float] = None,
    offset_fit: str = "avg_pchip",
    inverse_samples: int = DEFAULT_CUSTOM_INV_SAMPLES,
    samples_per_leg: int = 101,
    capture_every_move_point: bool = True,
    capture_steps: Optional[int] = None,
    compensate_y: bool = True,
    flip_rz_sign: bool = False,
) -> Tuple[List[TrajectoryPoint], dict]:
    """
    New fixed-stage output-offset modes.

    Unlike horizontal_line/vertical_line, these modes do not move X or Z to trace a
    line. The stage XYZ stays fixed, except optional Y compensation to keep the tip
    in the imaging plane while B curls/uncurls.

    horizontal_offset/radial_offset:
        User commands a desired calibrated radial/horizontal offset r(B).
        B is chosen by inverting the selected r model.

    z_offset/vertical_offset:
        User commands a desired calibrated vertical offset z(B).
        B is chosen by inverting the selected z model.
    """
    base_stage_xyz = np.asarray(base_stage_xyz, dtype=float).reshape(3)
    mode = str(offset_mode).strip().lower()
    if mode in ("horizontal_offset", "radial_offset", "r_offset"):
        output_axis = "radial"
        block_name = f"HOFF_C{float(c_deg):.3f}"
        segment_kind = "horizontal_offset"
        human_mode = "horizontal_offset"
    elif mode in ("z_offset", "vertical_offset"):
        output_axis = "z"
        block_name = f"ZOFF_C{float(c_deg):.3f}"
        segment_kind = "z_offset"
        human_mode = "z_offset"
    else:
        raise ValueError(f"Unsupported fixed-stage offset mode: {offset_mode}")

    used_min, used_max, avail_range, model_metas = _resolve_requested_offset_range(
        cal=cal,
        output_axis=output_axis,
        fit_mode=str(offset_fit),
        offset_min_mm=offset_min_mm,
        offset_max_mm=offset_max_mm,
        inverse_samples=int(inverse_samples),
        flip_rz_sign=bool(flip_rz_sign),
    )
    profile = _offset_value_profile(used_min, used_max, samples_per_leg=samples_per_leg)

    capture_indices: Set[int] = set(range(len(profile))) if bool(capture_every_move_point) else set()
    if not capture_every_move_point:
        ncap = max(1, int(capture_steps if capture_steps is not None else DEFAULT_ORIENTATION_CAPTURE_STEPS))
        for j in range(ncap + 1):
            capture_indices.add(int(round(j * (len(profile) - 1) / float(ncap))))

    first_offset, first_phase, _ = profile[0]
    first_fit = first_phase if str(offset_fit).strip().lower() == "phase_specific" else str(offset_fit)
    first_b, _, first_inv_meta = output_offset_to_b_for_fit(
        cal=cal,
        requested_offset_mm=float(first_offset),
        output_axis=output_axis,
        fit_mode=first_fit,
        motion_phase=first_phase,
        inverse_samples=inverse_samples,
        flip_rz_sign=bool(flip_rz_sign),
    )
    if not model_metas or first_inv_meta.get("motion_phase") != model_metas[0].get("motion_phase"):
        model_metas.insert(0, first_inv_meta)

    offset0_xyz = tip_offset_xyz_physical(
        cal=cal,
        b=float(first_b),
        c_deg=float(c_deg),
        flip_rz_sign=bool(flip_rz_sign),
        motion_phase=first_phase,
    )

    traj: List[TrajectoryPoint] = []
    for idx, (requested_offset, phase, phase01) in enumerate(profile):
        fit_for_this = phase if str(offset_fit).strip().lower() == "phase_specific" else str(offset_fit)
        b_cmd, used_offset, inv_meta = output_offset_to_b_for_fit(
            cal=cal,
            requested_offset_mm=float(requested_offset),
            output_axis=output_axis,
            fit_mode=fit_for_this,
            motion_phase=phase,
            inverse_samples=inverse_samples,
            flip_rz_sign=bool(flip_rz_sign),
        )
        if idx == 0 or phase != profile[idx - 1][1]:
            if not model_metas or inv_meta.get("motion_phase") != model_metas[-1].get("motion_phase"):
                model_metas.append(inv_meta)

        stage_xyz = np.array(base_stage_xyz, dtype=float)
        if bool(compensate_y):
            off_xyz = tip_offset_xyz_physical(
                cal=cal,
                b=float(b_cmd),
                c_deg=float(c_deg),
                flip_rz_sign=bool(flip_rz_sign),
                motion_phase=phase,
            )
            stage_xyz[1] = base_stage_xyz[1] - (float(off_xyz[1]) - float(offset0_xyz[1]))

        traj.append(
            TrajectoryPoint(
                b=float(b_cmd),
                c=float(c_deg),
                stage_xyz=stage_xyz,
                segment_kind=segment_kind,
                capture_image=(idx in capture_indices),
                tip_angle_deg=None,
                block_name=block_name,
                block_phase_01=float(phase01),
                oscillation_phase_rad=None,
                block_index=0,
                motion_phase=phase,
            )
        )

    meta = compute_traj_meta(traj)
    meta.update({
        "trajectory_mode": human_mode,
        "offset_mode": human_mode,
        "offset_output_axis": output_axis,
        "offset_fit": str(offset_fit),
        "requested_offset_range_mm": [None if offset_min_mm is None else float(offset_min_mm), None if offset_max_mm is None else float(offset_max_mm)],
        "used_offset_range_mm": [float(used_min), float(used_max)],
        "calibrated_common_offset_range_mm": [float(avail_range[0]), float(avail_range[1])],
        "samples_per_leg": int(samples_per_leg),
        "y_plane_compensation": bool(compensate_y),
        "stage_xyz_behavior": "X and Z stay fixed; Y compensates calibrated off-plane offset" if bool(compensate_y) else "XYZ stays fixed",
        "return_to_start": {
            "profile_returns_to_start_offset": True,
            "start_b": float(traj[0].b) if traj else None,
            "end_b": float(traj[-1].b) if traj else None,
            "start_x": float(traj[0].stage_xyz[0]) if traj else None,
            "end_x": float(traj[-1].stage_xyz[0]) if traj else None,
            "start_y": float(traj[0].stage_xyz[1]) if traj else None,
            "end_y": float(traj[-1].stage_xyz[1]) if traj else None,
            "start_z": float(traj[0].stage_xyz[2]) if traj else None,
            "end_z": float(traj[-1].stage_xyz[2]) if traj else None,
        },
        "model_metas": model_metas,
    })
    return traj, meta

def _attack_angle_profile(min_deg: float, max_deg: float, samples_per_leg: int) -> List[Tuple[float, str, float]]:
    n = max(2, int(samples_per_leg))
    pull = np.linspace(float(min_deg), float(max_deg), n, dtype=float)
    release = np.linspace(float(max_deg), float(min_deg), n, dtype=float)
    out: List[Tuple[float, str, float]] = []
    for i, a in enumerate(pull):
        out.append((float(a), "pull", i / float(n - 1)))
    for i, a in enumerate(release[1:], start=1):
        out.append((float(a), "release", i / float(n - 1)))
    return out

def generate_angle_command_curl_uncurl_trajectory(
    cal: Calibration,
    base_stage_xyz: np.ndarray,
    c_deg: float,
    attack_min_deg: float,
    attack_max_deg: float,
    angle_fit: str = "avg_pchip",
    inverse_samples: int = DEFAULT_CUSTOM_INV_SAMPLES,
    samples_per_leg: int = 101,
    capture_every_move_point: bool = True,
    capture_steps: Optional[int] = None,
    track_xz_subgoal: bool = False,
    track_xyz_subgoal: bool = False,
    p_tip_subgoal_xyz: Optional[np.ndarray] = None,
    flip_rz_sign: bool = False,
) -> Tuple[List[TrajectoryPoint], dict]:
    """
    Build one curl/uncurl at a fixed C attack orientation.
    Default behavior does NOT track a fixed point: XYZ stays at base_stage_xyz while B is
    commanded from the requested input attack angle.
    Optional --track-xz-subgoal keeps the tip's X/Z projection fixed using the current
    calibration. Optional --track-xyz-subgoal keeps full XYZ fixed.
    """
    base_stage_xyz = np.asarray(base_stage_xyz, dtype=float).reshape(3)
    if p_tip_subgoal_xyz is None:
        p_tip_subgoal_xyz = np.asarray(base_stage_xyz, dtype=float).reshape(3)
    else:
        p_tip_subgoal_xyz = np.asarray(p_tip_subgoal_xyz, dtype=float).reshape(3)

    # Optionally replace requested range with calibrated range before the profile is built.
    # For phase-specific, use the union/overlap visible through pull and release tables.
    pull_table, _, pull_meta = build_tip_angle_inverse_table_for_fit(
        cal, angle_fit=("pull" if str(angle_fit).lower() == "phase_specific" else angle_fit),
        motion_phase="pull", num_samples=inverse_samples
    )
    rel_table, _, rel_meta = build_tip_angle_inverse_table_for_fit(
        cal, angle_fit=("release" if str(angle_fit).lower() == "phase_specific" else angle_fit),
        motion_phase="release", num_samples=inverse_samples
    )
    avail_min = max(float(pull_table[0]), float(rel_table[0]))
    avail_max = min(float(pull_table[-1]), float(rel_table[-1]))
    if avail_min > avail_max:
        avail_min = min(float(pull_table[0]), float(rel_table[0]))
        avail_max = max(float(pull_table[-1]), float(rel_table[-1]))

    used_min = clamp(float(attack_min_deg), avail_min, avail_max)
    used_max = clamp(float(attack_max_deg), avail_min, avail_max)
    if used_min > used_max:
        used_min, used_max = used_max, used_min

    profile = _attack_angle_profile(used_min, used_max, samples_per_leg=samples_per_leg)

    capture_indices: Set[int] = set(range(len(profile))) if bool(capture_every_move_point) else set()
    if not capture_every_move_point:
        ncap = max(1, int(capture_steps if capture_steps is not None else DEFAULT_ORIENTATION_CAPTURE_STEPS))
        for j in range(ncap + 1):
            capture_indices.add(int(round(j * (len(profile) - 1) / float(ncap))))

    traj: List[TrajectoryPoint] = []
    model_metas = []
    for idx, (input_angle, phase, phase01) in enumerate(profile):
        fit_for_this = str(angle_fit).lower()
        if fit_for_this == "phase_specific":
            fit_for_this = phase
        b_cmd, used_angle, inv_meta = angle_to_b_for_fit(
            cal=cal,
            requested_angle_deg=input_angle,
            angle_fit=fit_for_this,
            motion_phase=phase,
            inverse_samples=inverse_samples,
        )
        if idx == 0 or phase != profile[idx - 1][1]:
            model_metas.append(inv_meta)

        if track_xyz_subgoal or track_xz_subgoal:
            compensated = stage_xyz_for_fixed_tip(
                cal=cal,
                p_tip_xyz=p_tip_subgoal_xyz,
                b=b_cmd,
                c_deg=float(c_deg),
                flip_rz_sign=flip_rz_sign,
                motion_phase=phase,
            )
            if track_xyz_subgoal:
                stage_xyz = compensated
            else:
                stage_xyz = np.array([compensated[0], base_stage_xyz[1], compensated[2]], dtype=float)
        else:
            stage_xyz = np.array(base_stage_xyz, dtype=float)

        traj.append(
            TrajectoryPoint(
                b=float(b_cmd),
                c=float(c_deg),
                stage_xyz=stage_xyz,
                segment_kind="angle_command",
                capture_image=(idx in capture_indices),
                tip_angle_deg=float(used_angle),
                block_name=f"BATTACK_C{float(c_deg):.3f}",
                block_phase_01=float(phase01),
                oscillation_phase_rad=None,
                block_index=0,
                motion_phase=phase,
            )
        )

    meta = compute_traj_meta(traj)
    meta.update({
        "trajectory_mode": "angle_command_curl_uncurl",
        "angle_fit": str(angle_fit),
        "requested_attack_angle_range_deg": [float(attack_min_deg), float(attack_max_deg)],
        "used_attack_angle_range_deg": [float(used_min), float(used_max)],
        "calibrated_common_attack_angle_range_deg": [float(avail_min), float(avail_max)],
        "samples_per_leg": int(samples_per_leg),
        "track_xz_subgoal": bool(track_xz_subgoal),
        "track_xyz_subgoal": bool(track_xyz_subgoal),
        "model_metas": model_metas,
    })
    return traj, meta



def _common_available_attack_range_for_fit(
    cal: Calibration,
    angle_fit: str,
    inverse_samples: int,
) -> Tuple[float, float, List[dict]]:
    """Return the common pull/release attack-angle range for the selected fit."""
    mode = str(angle_fit or "avg_pchip").strip().lower()
    pull_fit = "pull" if mode == "phase_specific" else mode
    release_fit = "release" if mode == "phase_specific" else mode

    pull_table, _, pull_meta = build_tip_angle_inverse_table_for_fit(
        cal, angle_fit=pull_fit, motion_phase="pull", num_samples=inverse_samples
    )
    release_table, _, release_meta = build_tip_angle_inverse_table_for_fit(
        cal, angle_fit=release_fit, motion_phase="release", num_samples=inverse_samples
    )

    avail_min = max(float(pull_table[0]), float(release_table[0]))
    avail_max = min(float(pull_table[-1]), float(release_table[-1]))
    if avail_min > avail_max:
        # If the two phase-specific fits do not overlap numerically, fall back to the union
        # and let angle_to_b_for_fit clip each leg independently.
        avail_min = min(float(pull_table[0]), float(release_table[0]))
        avail_max = max(float(pull_table[-1]), float(release_table[-1]))
    return float(avail_min), float(avail_max), [pull_meta, release_meta]


def _resolve_requested_attack_range(
    cal: Calibration,
    angle_fit: str,
    attack_min_deg: float,
    attack_max_deg: float,
    inverse_samples: int,
    use_calibrated_attack_range: bool = False,
) -> Tuple[float, float, Tuple[float, float], List[dict]]:
    avail_min, avail_max, model_metas = _common_available_attack_range_for_fit(
        cal=cal,
        angle_fit=angle_fit,
        inverse_samples=inverse_samples,
    )
    if bool(use_calibrated_attack_range):
        used_min, used_max = avail_min, avail_max
    else:
        used_min = clamp(float(attack_min_deg), avail_min, avail_max)
        used_max = clamp(float(attack_max_deg), avail_min, avail_max)
        if used_min > used_max:
            used_min, used_max = used_max, used_min
    return float(used_min), float(used_max), (float(avail_min), float(avail_max)), model_metas


def _apply_daq_attack_window(
    attack_min: float,
    attack_max: float,
    window_min: float = 0.0,
    window_max: float = 180.0,
    enforce: bool = True,
) -> Tuple[float, float, Optional[str]]:
    """Intersect a requested/calibrated DAQ attack range with the prescribed safe recording window."""
    a0, a1 = sorted([float(attack_min), float(attack_max)])
    if not bool(enforce):
        return float(a0), float(a1), None
    w0, w1 = sorted([float(window_min), float(window_max)])
    lo = max(a0, w0)
    hi = min(a1, w1)
    if lo > hi:
        raise ValueError(
            f"DAQ attack-angle range [{a0:.6g}, {a1:.6g}] deg has no overlap with "
            f"the enforced recording window [{w0:.6g}, {w1:.6g}] deg."
        )
    msg = None
    if abs(lo - a0) > 1e-12 or abs(hi - a1) > 1e-12:
        msg = (
            f"DAQ attack range clipped from [{a0:.6g}, {a1:.6g}] deg to "
            f"[{lo:.6g}, {hi:.6g}] deg by enforced recording window [{w0:.6g}, {w1:.6g}] deg."
        )
    return float(lo), float(hi), msg


def generate_line_tracking_trajectory(
    cal: Calibration,
    base_stage_xyz: np.ndarray,
    c_deg: float,
    attack_min_deg: float,
    attack_max_deg: float,
    line_mode: str,
    angle_fit: str = "avg_pchip",
    inverse_samples: int = DEFAULT_CUSTOM_INV_SAMPLES,
    samples_per_leg: int = 101,
    capture_every_move_point: bool = True,
    capture_steps: Optional[int] = None,
    compensate_y: bool = True,
    flip_rz_sign: bool = False,
) -> Tuple[List[TrajectoryPoint], dict]:
    """
    Build a curl/uncurl line-tracking trajectory.

    horizontal_line:
        B curl creates the radial/horizontal motion. The stage Z axis compensates the
        calibrated B-dependent z offset so the tip is commanded to remain on a
        horizontal XZ-plane line. The profile curls and uncurls back to the starting
        B and Z.

    vertical_line:
        B curl creates vertical motion. The stage X axis compensates the calibrated
        radial/x offset so the tip is commanded to remain on a vertical XZ-plane line.
        The profile curls and uncurls back to the starting B and X.

    In both modes, --no-y-plane-compensation disables stage-Y compensation. With the
    default compensation enabled, stage Y cancels the calibrated off-plane y offset so
    the tip stays in the same physical plane while B changes.
    """
    mode = str(line_mode).strip().lower()
    if mode not in ("horizontal_line", "vertical_line"):
        raise ValueError(f"Unsupported line_mode: {line_mode}")

    base_stage_xyz = np.asarray(base_stage_xyz, dtype=float).reshape(3)
    used_min, used_max, avail_range, common_metas = _resolve_requested_attack_range(
        cal=cal,
        angle_fit=angle_fit,
        attack_min_deg=attack_min_deg,
        attack_max_deg=attack_max_deg,
        inverse_samples=inverse_samples,
        use_calibrated_attack_range=False,
    )
    profile = _attack_angle_profile(used_min, used_max, samples_per_leg=samples_per_leg)

    capture_indices: Set[int] = set(range(len(profile))) if bool(capture_every_move_point) else set()
    if not capture_every_move_point:
        ncap = max(1, int(capture_steps if capture_steps is not None else DEFAULT_ORIENTATION_CAPTURE_STEPS))
        for j in range(ncap + 1):
            capture_indices.add(int(round(j * (len(profile) - 1) / float(ncap))))

    # Use the first profile sample as the line origin. The end of the curl/uncurl profile
    # returns to this same input angle, B value, and compensated stage axis by construction.
    first_input_angle, first_phase, _ = profile[0]
    first_fit = first_phase if str(angle_fit).strip().lower() == "phase_specific" else str(angle_fit)
    first_b, first_used_angle, first_inv_meta = angle_to_b_for_fit(
        cal=cal,
        requested_angle_deg=float(first_input_angle),
        angle_fit=first_fit,
        motion_phase=first_phase,
        inverse_samples=inverse_samples,
    )
    offset0 = tip_offset_xyz_physical(
        cal=cal,
        b=float(first_b),
        c_deg=float(c_deg),
        flip_rz_sign=flip_rz_sign,
        motion_phase=first_phase,
    )

    traj: List[TrajectoryPoint] = []
    model_metas: List[dict] = [first_inv_meta]
    block_name = f"HLINE_C{float(c_deg):.3f}" if mode == "horizontal_line" else f"VLINE_C{float(c_deg):.3f}"
    segment_kind = "horizontal_line" if mode == "horizontal_line" else "vertical_line"

    for idx, (input_angle, phase, phase01) in enumerate(profile):
        fit_for_this = phase if str(angle_fit).strip().lower() == "phase_specific" else str(angle_fit)
        b_cmd, used_angle, inv_meta = angle_to_b_for_fit(
            cal=cal,
            requested_angle_deg=float(input_angle),
            angle_fit=fit_for_this,
            motion_phase=phase,
            inverse_samples=inverse_samples,
        )
        if idx == 0 or phase != profile[idx - 1][1]:
            # Keep a concise record of which fit model drove each leg.
            if not model_metas or inv_meta.get("motion_phase") != model_metas[-1].get("motion_phase"):
                model_metas.append(inv_meta)

        offset = tip_offset_xyz_physical(
            cal=cal,
            b=float(b_cmd),
            c_deg=float(c_deg),
            flip_rz_sign=flip_rz_sign,
            motion_phase=phase,
        )
        stage_xyz = np.array(base_stage_xyz, dtype=float)

        if mode == "horizontal_line":
            # Hold tip Z constant; B supplies the radial/horizontal change.
            stage_xyz[2] = base_stage_xyz[2] - (float(offset[2]) - float(offset0[2]))
        else:
            # Hold tip X constant; B supplies the vertical Z change.
            stage_xyz[0] = base_stage_xyz[0] - (float(offset[0]) - float(offset0[0]))

        if bool(compensate_y):
            stage_xyz[1] = base_stage_xyz[1] - (float(offset[1]) - float(offset0[1]))

        traj.append(
            TrajectoryPoint(
                b=float(b_cmd),
                c=float(c_deg),
                stage_xyz=stage_xyz,
                segment_kind=segment_kind,
                capture_image=(idx in capture_indices),
                tip_angle_deg=float(used_angle),
                block_name=block_name,
                block_phase_01=float(phase01),
                oscillation_phase_rad=None,
                block_index=0,
                motion_phase=phase,
            )
        )

    meta = compute_traj_meta(traj)
    meta.update({
        "trajectory_mode": mode,
        "line_mode": "horizontal" if mode == "horizontal_line" else "vertical",
        "angle_fit": str(angle_fit),
        "requested_attack_angle_range_deg": [float(attack_min_deg), float(attack_max_deg)],
        "used_attack_angle_range_deg": [float(used_min), float(used_max)],
        "calibrated_common_attack_angle_range_deg": [float(avail_range[0]), float(avail_range[1])],
        "samples_per_leg": int(samples_per_leg),
        "y_plane_compensation": bool(compensate_y),
        "line_axis_compensation": "Z axis compensates vertical offset; B curl supplies horizontal/radial motion" if mode == "horizontal_line" else "X axis compensates radial offset; B curl supplies vertical motion",
        "return_to_start": {
            "profile_returns_to_start_attack_angle": True,
            "start_b": float(traj[0].b) if traj else None,
            "end_b": float(traj[-1].b) if traj else None,
            "start_x": float(traj[0].stage_xyz[0]) if traj else None,
            "end_x": float(traj[-1].stage_xyz[0]) if traj else None,
            "start_z": float(traj[0].stage_xyz[2]) if traj else None,
            "end_z": float(traj[-1].stage_xyz[2]) if traj else None,
        },
        "model_metas": model_metas,
    })
    return traj, meta

def run_acquisition(args):
    cal = load_calibration(args.calibration, y_offset_fit=args.y_offset_fit)

    base_stage_xyz = np.array([float(args.point_x), float(args.point_y), float(args.point_z)], dtype=float)

    attack_min = float(args.attack_min_deg)
    attack_max = float(args.attack_max_deg)

    trajectory_mode = str(args.trajectory_mode).strip().lower()
    active_fit_branch = resolve_fit_branch_for_window(
        getattr(args, "fit_branch", "auto"),
        attack_min_deg=attack_min,
        attack_max_deg=attack_max,
    )
    cal.active_fit_branch = active_fit_branch
    print(f"[INFO] Active calibration fit branch for acquisition: {_fit_branch_title(active_fit_branch)}")
    if getattr(cal, "available_fit_branches", None):
        print(f"[INFO] Branch labels detected in calibration JSON: {getattr(cal, 'available_fit_branches', [])}")
    if bool(args.use_calibrated_attack_range) and trajectory_mode in ("angle_command", "one_daq_all_metrics", "fixed_xyz_all_metrics", "all_metrics", "horizontal_line", "vertical_line"):
        attack_min, attack_max, _, _ = _resolve_requested_attack_range(
            cal=cal,
            angle_fit=str(args.angle_fit),
            attack_min_deg=attack_min,
            attack_max_deg=attack_max,
            inverse_samples=int(args.custom_inverse_samples),
            use_calibrated_attack_range=True,
        )
    if trajectory_mode in ("angle_command", "one_daq_all_metrics", "fixed_xyz_all_metrics", "all_metrics", "horizontal_line", "vertical_line"):
        attack_min, attack_max, attack_clip_msg = _apply_daq_attack_window(
            attack_min=attack_min,
            attack_max=attack_max,
            window_min=float(getattr(args, "daq_attack_window_min_deg", 0.0)),
            window_max=float(getattr(args, "daq_attack_window_max_deg", 180.0)),
            enforce=(not bool(getattr(args, "no_enforce_daq_attack_window", False))),
        )
        if attack_clip_msg:
            print(f"[INFO] {attack_clip_msg}")

    if trajectory_mode in ("angle_command", "one_daq_all_metrics", "fixed_xyz_all_metrics", "all_metrics"):
        traj, meta = generate_angle_command_curl_uncurl_trajectory(
            cal=cal,
            base_stage_xyz=base_stage_xyz,
            c_deg=float(args.attack_c_deg),
            attack_min_deg=attack_min,
            attack_max_deg=attack_max,
            angle_fit=str(args.angle_fit),
            inverse_samples=int(args.custom_inverse_samples),
            samples_per_leg=int(args.samples_per_leg),
            capture_every_move_point=bool(args.capture_every_move_point),
            capture_steps=int(args.orientation_capture_steps),
            track_xz_subgoal=bool(args.track_xz_subgoal),
            track_xyz_subgoal=bool(args.track_xyz_subgoal),
            p_tip_subgoal_xyz=base_stage_xyz,
            flip_rz_sign=bool(args.flip_rz_sign),
        )
        if trajectory_mode in ("one_daq_all_metrics", "fixed_xyz_all_metrics", "all_metrics"):
            # Same physical DAQ as angle_command: X/Z stay fixed unless the user explicitly
            # enables tracking. The tag just tells processing to compute angle + r + z metrics
            # from this one curl/uncurl acquisition.
            for _pt in traj:
                _pt.segment_kind = "one_daq_all_metrics"
                _pt.block_name = f"ALLMETRICS_{_fit_branch_label(getattr(cal, 'active_fit_branch', None)).upper()}_C{float(args.attack_c_deg):.3f}"
            meta["trajectory_mode"] = "one_daq_all_metrics"
            meta["single_daq_metrics"] = True
            meta["stage_xyz_behavior"] = (
                "fixed XYZ by default; only Y or XYZ changes if a compensation/tracking flag is enabled"
            )
    elif trajectory_mode in ("horizontal_line", "vertical_line"):
        traj, meta = generate_line_tracking_trajectory(
            cal=cal,
            base_stage_xyz=base_stage_xyz,
            c_deg=float(args.attack_c_deg),
            attack_min_deg=attack_min,
            attack_max_deg=attack_max,
            line_mode=trajectory_mode,
            angle_fit=str(args.angle_fit),
            inverse_samples=int(args.custom_inverse_samples),
            samples_per_leg=int(args.samples_per_leg),
            capture_every_move_point=bool(args.capture_every_move_point),
            capture_steps=int(args.orientation_capture_steps),
            compensate_y=(not bool(args.no_y_plane_compensation)),
            flip_rz_sign=bool(args.flip_rz_sign),
        )
    elif trajectory_mode in ("horizontal_offset", "radial_offset", "r_offset", "z_offset", "vertical_offset"):
        offset_min = getattr(args, "offset_min_mm", None)
        offset_max = getattr(args, "offset_max_mm", None)
        if trajectory_mode in ("horizontal_offset", "radial_offset", "r_offset"):
            if getattr(args, "horizontal_offset_min_mm", None) is not None:
                offset_min = args.horizontal_offset_min_mm
            if getattr(args, "horizontal_offset_max_mm", None) is not None:
                offset_max = args.horizontal_offset_max_mm
        else:
            if getattr(args, "z_offset_min_mm", None) is not None:
                offset_min = args.z_offset_min_mm
            if getattr(args, "z_offset_max_mm", None) is not None:
                offset_max = args.z_offset_max_mm
        traj, meta = generate_fixed_stage_offset_command_trajectory(
            cal=cal,
            base_stage_xyz=base_stage_xyz,
            c_deg=float(args.attack_c_deg),
            offset_mode=trajectory_mode,
            offset_min_mm=offset_min,
            offset_max_mm=offset_max,
            offset_fit=str(args.offset_fit or args.angle_fit),
            inverse_samples=int(args.custom_inverse_samples),
            samples_per_leg=int(args.samples_per_leg),
            capture_every_move_point=bool(args.capture_every_move_point),
            capture_steps=int(args.orientation_capture_steps),
            compensate_y=(not bool(args.no_y_plane_compensation)),
            flip_rz_sign=bool(args.flip_rz_sign),
        )
    else:
        raise ValueError(f"Unsupported --trajectory-mode: {args.trajectory_mode}")

    meta["fit_branch"] = _fit_branch_label(getattr(cal, "active_fit_branch", None))
    meta["fit_branch_title"] = _fit_branch_title(getattr(cal, "active_fit_branch", None))

    if trajectory_mode in ("angle_command", "one_daq_all_metrics", "fixed_xyz_all_metrics", "all_metrics", "horizontal_line", "vertical_line"):
        meta["enforced_daq_attack_window"] = not bool(getattr(args, "no_enforce_daq_attack_window", False))
        meta["daq_attack_window_deg"] = [
            float(getattr(args, "daq_attack_window_min_deg", 0.0)),
            float(getattr(args, "daq_attack_window_max_deg", 180.0)),
        ]

    print("Unified trajectory summary:")
    for k in [
        "trajectory_mode", "fit_branch", "fit_branch_title", "line_mode", "offset_mode", "offset_output_axis", "n_samples", "n_capture_points", "angle_fit", "offset_fit",
        "requested_attack_angle_range_deg", "used_attack_angle_range_deg",
        "calibrated_common_attack_angle_range_deg", "enforced_daq_attack_window", "daq_attack_window_deg",
        "requested_offset_range_mm", "used_offset_range_mm",
        "calibrated_common_offset_range_mm", "b_min_used", "b_max_used",
        "track_xz_subgoal", "track_xyz_subgoal", "y_plane_compensation",
        "line_axis_compensation", "stage_xyz_behavior", "return_to_start",
    ]:
        if k in meta:
            print(f"  {k}: {meta[k]}")
    if meta.get("model_metas"):
        print("  selected angle models:")
        seen = set()
        for mm in meta["model_metas"]:
            key = (mm.get("motion_phase"), mm.get("selected_model_label"), mm.get("selected_model_summary"))
            if key in seen:
                continue
            seen.add(key)
            print(f"    {mm.get('motion_phase')}: {mm.get('selected_model_label')} -> {mm.get('selected_model_summary')}")

    start_pose = (float(args.start_x), float(args.start_y), float(args.start_z), float(args.start_b), float(args.start_c))
    end_pose = (float(args.end_x), float(args.end_y), float(args.end_z), float(args.end_b), float(args.end_c))

    virtual_bbox = {
        "x_min": float(args.bbox_x_min), "x_max": float(args.bbox_x_max),
        "y_min": float(args.bbox_y_min), "y_max": float(args.bbox_y_max),
        "z_min": float(args.bbox_z_min), "z_max": float(args.bbox_z_max),
    }
    for lo, hi in [("x_min", "x_max"), ("y_min", "y_max"), ("z_min", "z_max")]:
        if virtual_bbox[lo] > virtual_bbox[hi]:
            virtual_bbox[lo], virtual_bbox[hi] = virtual_bbox[hi], virtual_bbox[lo]

    runner = FixedTipPointTracker(
        parent_directory=args.parent_directory,
        project_name=args.project_name,
        allow_existing=bool(args.allow_existing),
        add_date=bool(args.add_date),
    )
    try:
        # Save the exact planned trajectory metadata before any motion starts.
        try:
            meta_path = Path(runner.run_folder) / "trajectory_metadata.json"
            with open(meta_path, "w") as f:
                json.dump(_json_ready(meta), f, indent=2)
            print(f"Trajectory metadata JSON: {meta_path}")
        except Exception as e:
            print(f"[WARN] Could not save trajectory metadata JSON: {e}")

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
            b_extra_settle_s=float(args.b_extra_settle_s),
            inter_command_delay_s=float(args.inter_command_delay_s),
            camera_flush_frames=int(args.camera_flush_frames),
            capture_at_start=bool(args.capture_at_start),
            initial_sweep_wait_s=float(args.initial_sweep_wait_s),
            c_flip_feed=float(args.c_flip_feed),
            settled_capture_mode=bool(args.settled_capture_mode),
            settled_capture_buffer_s=float(args.settled_capture_buffer_s),
        )
        print("\nFinal acquisition results:")
        print(results)
        args.project_dir = runner.run_folder
        return runner.run_folder
    finally:
        try:
            runner.disconnect_camera()
        except Exception:
            pass
        try:
            runner.disconnect_robot()
        except Exception:
            pass


def _line_mode_from_record(rec: Dict[str, Any]) -> Optional[str]:
    text = " ".join(str(rec.get(k) or "") for k in ("block_name", "phase_name", "image_name")).upper()
    if "HLINE" in text or "HORIZONTAL_LINE" in text:
        return "horizontal"
    if "VLINE" in text or "VERTICAL_LINE" in text:
        return "vertical"
    return None


def _finite_float_or_none(value: Any) -> Optional[float]:
    try:
        v = float(value)
    except Exception:
        return None
    return v if np.isfinite(v) else None


def _commanded_tip_xyz_from_record(
    rec: Dict[str, Any],
    robot_cal: Optional[Calibration],
    flip_rz_sign: bool = False,
) -> Tuple[Optional[float], Optional[float], Optional[float], str]:
    """
    Reconstruct the commanded physical tip XYZ for a captured image from the
    stage XYZ + B/C metadata embedded in the filename.

    For line-mode processing this is the correct input coordinate. The stage
    axes in the filename are where the robot was commanded; the calibrated
    B/C-dependent tip offset tells us where the tip was commanded relative to
    that stage pose.
    """
    sx = _finite_float_or_none(rec.get("stage_x_cmd"))
    sy = _finite_float_or_none(rec.get("stage_y_cmd"))
    sz = _finite_float_or_none(rec.get("stage_z_cmd"))
    b = _finite_float_or_none(rec.get("b_cmd"))
    c = _finite_float_or_none(rec.get("c_cmd_deg"))
    phase = _normalize_motion_phase_name(rec.get("motion_phase")) or None

    if robot_cal is not None and sx is not None and sy is not None and sz is not None and b is not None and c is not None:
        try:
            off = tip_offset_xyz_physical(
                cal=robot_cal,
                b=float(b),
                c_deg=float(c),
                flip_rz_sign=bool(flip_rz_sign),
                motion_phase=phase,
            )
            return (
                float(sx + float(off[0])),
                float(sy + float(off[1])),
                float(sz + float(off[2])),
                "stage_xyz_plus_calibrated_tip_offset",
            )
        except Exception as exc:
            rec["line_command_reconstruction_warning"] = str(exc)

    # Fallback keeps processing usable for old folders processed without a robot
    # calibration JSON. It is not the true commanded tip coordinate for line
    # modes, so the source is recorded explicitly.
    return sx, sy, sz, "stage_xyz_fallback_no_robot_calibration"


def attach_line_reference_errors(
    records: List[Dict[str, Any]],
    robot_cal: Optional[Calibration] = None,
    flip_rz_sign: bool = False,
) -> Dict[str, Any]:
    """
    Attach line-reference residuals for horizontal_line and vertical_line runs.

    Horizontal line:
        Reference is z = median(z_mm) for each line block. The perpendicular
        error is measured z_mm - z_line_ref_mm, and the hysteresis input axis is
        commanded physical tip X reconstructed from stage X + calibrated tip
        offset X(B, C).

    Vertical line:
        Reference is u = median(u_mm) for each line block. The perpendicular
        error is measured u_mm - u_line_ref_mm, and the hysteresis input axis is
        commanded physical tip Z reconstructed from stage Z + calibrated tip
        offset Z(B).
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for rec in records:
        mode = _line_mode_from_record(rec)
        if mode is None or not rec.get("valid", False):
            continue
        if rec.get("u_mm") is None or rec.get("z_mm") is None:
            continue
        group = str(rec.get("block_name") or rec.get("phase_name") or mode)
        groups.setdefault((mode, group), []).append(rec)

    metrics: Dict[str, Any] = {
        "num_line_groups": int(len(groups)),
        "groups": {},
        "by_mode": {},
        "commanded_input_coordinate": {
            "horizontal": "commanded physical tip X = stage_x_cmd + calibrated tip_offset_x(B,C)",
            "vertical": "commanded physical tip Z = stage_z_cmd + calibrated tip_offset_z(B)",
            "robot_calibration_available": bool(robot_cal is not None),
            "fallback_when_missing_calibration": "stage coordinate only; not true commanded tip coordinate",
        },
    }
    if not groups:
        return metrics

    by_mode_errors: Dict[str, List[float]] = {}
    by_mode_cmd: Dict[str, List[float]] = {}

    for (mode, group), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rows.sort(key=lambda r: int(r.get("sample_index", 0)))
        us = np.asarray([float(r["u_mm"]) for r in rows], dtype=float)
        zs = np.asarray([float(r["z_mm"]) for r in rows], dtype=float)

        cmd_xyz = [
            _commanded_tip_xyz_from_record(
                rec=r,
                robot_cal=robot_cal,
                flip_rz_sign=bool(flip_rz_sign),
            )
            for r in rows
        ]
        cmd_x = np.asarray([
            np.nan if xyz[0] is None else float(xyz[0])
            for xyz in cmd_xyz
        ], dtype=float)
        cmd_y = np.asarray([
            np.nan if xyz[1] is None else float(xyz[1])
            for xyz in cmd_xyz
        ], dtype=float)
        cmd_z = np.asarray([
            np.nan if xyz[2] is None else float(xyz[2])
            for xyz in cmd_xyz
        ], dtype=float)
        cmd_sources = [xyz[3] for xyz in cmd_xyz]

        if mode == "horizontal":
            const = float(np.median(zs))
            perp = zs - const
            along_measured = us
            axis = "z"
            command_axis = "x"
            commanded = cmd_x.copy()
            if not np.any(np.isfinite(commanded)):
                commanded = along_measured.copy()
                command_source = "measured_u_fallback_no_commanded_tip_x"
            else:
                command_source = cmd_sources[0] if len(set(cmd_sources)) == 1 else "mixed"
        else:
            const = float(np.median(us))
            perp = us - const
            along_measured = zs
            axis = "u"
            command_axis = "z"
            commanded = cmd_z.copy()
            if not np.any(np.isfinite(commanded)):
                commanded = along_measured.copy()
                command_source = "measured_z_fallback_no_commanded_tip_z"
            else:
                command_source = cmd_sources[0] if len(set(cmd_sources)) == 1 else "mixed"

        # Fill any isolated missing command values by the measured along-line
        # coordinate so plotting still produces a diagnostic trace, but keep the
        # source flag explicit in the CSV/JSON.
        finite_cmd = np.isfinite(commanded)
        if not np.all(finite_cmd):
            commanded = commanded.copy()
            commanded[~finite_cmd] = along_measured[~finite_cmd]
            command_source = command_source + "+measured_along_fill"

        cmd0 = float(commanded[0]) if commanded.size and np.isfinite(commanded[0]) else 0.0
        along0 = float(along_measured[0]) if along_measured.size else 0.0
        abs_err = np.abs(perp)
        by_mode_errors.setdefault(mode, []).extend(abs_err.astype(float).tolist())
        by_mode_cmd.setdefault(mode, []).extend(commanded.astype(float).tolist())

        for rec, a_meas, p, e, cx, cy, cz, cval in zip(rows, along_measured, perp, abs_err, cmd_x, cmd_y, cmd_z, commanded):
            rec["commanded_tip_x_mm"] = None if not np.isfinite(cx) else float(cx)
            rec["commanded_tip_y_mm"] = None if not np.isfinite(cy) else float(cy)
            rec["commanded_tip_z_mm"] = None if not np.isfinite(cz) else float(cz)
            rec["line_reference_mode"] = mode
            rec["line_group"] = group
            rec["line_reference_axis"] = axis
            rec["line_reference_const_mm"] = const
            rec["line_along_mm"] = float(a_meas)
            rec["line_along_relative_mm"] = float(a_meas - along0)
            rec["line_perpendicular_error_mm"] = float(p)
            rec["line_abs_error_mm"] = float(e)
            rec["line_input_attack_angle_deg"] = rec.get("tip_angle_deg_from_name")
            rec["line_input_command_axis"] = command_axis
            rec["line_input_commanded_mm"] = float(cval)
            rec["line_input_commanded_relative_mm"] = float(cval - cmd0)
            rec["line_input_commanded_source"] = command_source

        key = f"{mode}:{group}"
        finite_for_range = commanded[np.isfinite(commanded)]
        metrics["groups"][key] = {
            "line_mode": mode,
            "group": group,
            "num_samples": int(len(rows)),
            "reference_axis": axis,
            "reference_const_mm": const,
            "command_axis": command_axis,
            "command_source": command_source,
            "rmse_perpendicular_error_mm": float(np.sqrt(np.mean(perp ** 2))),
            "mean_abs_perpendicular_error_mm": float(np.mean(abs_err)),
            "median_abs_perpendicular_error_mm": float(np.median(abs_err)),
            "max_abs_perpendicular_error_mm": float(np.max(abs_err)),
            "measured_along_range_mm": [float(np.min(along_measured)), float(np.max(along_measured))],
            "commanded_input_range_mm": (
                None if finite_for_range.size == 0
                else [float(np.min(finite_for_range)), float(np.max(finite_for_range))]
            ),
            "commanded_input_relative_range_mm": (
                None if finite_for_range.size == 0
                else [float(np.min(finite_for_range - cmd0)), float(np.max(finite_for_range - cmd0))]
            ),
            "input_attack_angle_range_deg": [
                float(np.nanmin([r.get("tip_angle_deg_from_name", np.nan) for r in rows])),
                float(np.nanmax([r.get("tip_angle_deg_from_name", np.nan) for r in rows])),
            ],
        }

    for mode, errs in by_mode_errors.items():
        arr = np.asarray(errs, dtype=float)
        cmd_arr = np.asarray(by_mode_cmd.get(mode, []), dtype=float)
        cmd_arr = cmd_arr[np.isfinite(cmd_arr)]
        metrics["by_mode"][mode] = {
            "num_samples": int(arr.size),
            "rmse_perpendicular_error_mm": float(np.sqrt(np.mean(arr ** 2))) if arr.size else None,
            "mean_abs_perpendicular_error_mm": float(np.mean(arr)) if arr.size else None,
            "median_abs_perpendicular_error_mm": float(np.median(arr)) if arr.size else None,
            "max_abs_perpendicular_error_mm": float(np.max(arr)) if arr.size else None,
            "commanded_input_range_mm": (
                None if cmd_arr.size == 0 else [float(np.min(cmd_arr)), float(np.max(cmd_arr))]
            ),
        }
    return metrics


def save_line_reference_metrics_json(json_path: Path, line_metrics: Dict[str, Any]):
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(_json_ready(line_metrics), f, indent=2)


def save_line_reference_hysteresis_plots(
    processed_dir: Path,
    records: List[Dict[str, Any]],
    line_metrics: Dict[str, Any],
    title_prefix: str = "",
) -> List[Path]:
    """Save transparent dark-style line-reference hysteresis plots.

    For line modes, the x-axis is the commanded line coordinate, not the
    input attack angle:

      horizontal_line:
          y = measured z_mm - z_line_ref_mm
          x = commanded physical tip X

      vertical_line:
          y = measured u_mm - u_line_ref_mm
          x = commanded physical tip Z

    Pull and release are drawn as separate traces so hysteresis is visible in
    the physical line-following coordinate.
    """
    processed_dir = Path(processed_dir)
    paths: List[Path] = []

    palette = ["#57c7ff", "#79d9cf", "#f3d67a", "#f472b6", "#a78bfa", "#34d399"]

    for mode in ("horizontal", "vertical"):
        mode_rows = [
            r for r in records
            if r.get("valid", False)
            and r.get("line_reference_mode") == mode
            and r.get("line_input_commanded_mm") is not None
            and r.get("line_perpendicular_error_mm") is not None
        ]
        if len(mode_rows) < 2:
            continue

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in mode_rows:
            grouped.setdefault(str(r.get("line_group") or mode), []).append(r)

        fig, ax = plt.subplots(figsize=(10.2, 5.8), dpi=150)
        fig.patch.set_alpha(0.0)

        if mode == "horizontal":
            title = f"{title_prefix}Horizontal-line hysteresis: z error vs commanded tip X".strip()
            x_label = "Commanded tip X (mm)"
            y_label = "Perpendicular line error: measured z - horizontal-line z_ref (mm)"
            out_name = "hysteresis_horizontal_line_z_error_vs_commanded_tip_x.png"
        else:
            title = f"{title_prefix}Vertical-line hysteresis: u error vs commanded tip Z".strip()
            x_label = "Commanded tip Z (mm)"
            y_label = "Perpendicular line error: measured u - vertical-line u_ref (mm)"
            out_name = "hysteresis_vertical_line_u_error_vs_commanded_tip_z.png"

        _apply_dark_axes_style(ax, title, x_label, y_label)
        ax.set_facecolor((0.0, 0.0, 0.0, 0.0))
        ax.axhline(0.0, color="#d7e2ee", linewidth=1.1, linestyle="--", alpha=0.65)

        any_plotted = False
        trace_i = 0
        for group, rows in sorted(grouped.items(), key=lambda kv: kv[0]):
            rows = sorted(rows, key=lambda rr: int(rr.get("sample_index", 0)))
            phase_groups: Dict[str, List[Dict[str, Any]]] = {}
            for rr in rows:
                phase_groups.setdefault(str(rr.get("motion_phase") or rr.get("phase_name") or "unknown"), []).append(rr)
            for phase, rr_list in sorted(phase_groups.items(), key=lambda kv: kv[0]):
                x = np.asarray([float(rr["line_input_commanded_mm"]) for rr in rr_list], dtype=float)
                y_err = np.asarray([float(rr["line_perpendicular_error_mm"]) for rr in rr_list], dtype=float)
                finite = np.isfinite(x) & np.isfinite(y_err)
                if int(np.count_nonzero(finite)) < 2:
                    continue
                x = x[finite]
                y_err = y_err[finite]
                label = f"{group} {phase}"
                color = palette[trace_i % len(palette)]
                ax.plot(
                    x,
                    y_err,
                    marker="o",
                    markersize=3.2,
                    linewidth=1.5,
                    color=color,
                    markerfacecolor=color,
                    markeredgewidth=0.0,
                    label=label,
                )
                any_plotted = True
                trace_i += 1

        mode_summary = (line_metrics.get("by_mode") or {}).get(mode) or {}
        if mode_summary.get("rmse_perpendicular_error_mm") is not None:
            ax.text(
                0.015,
                0.965,
                f"RMSE perp. error = {float(mode_summary['rmse_perpendicular_error_mm']):.4f} mm",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="#f4f7fb",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": (0.0, 0.0, 0.0, 0.0),
                    "edgecolor": (1, 1, 1, 0.16),
                },
            )

        if any_plotted:
            leg = ax.legend(
                facecolor=(0.0, 0.0, 0.0, 0.0),
                edgecolor=(1, 1, 1, 0.18),
                fontsize=8.2,
                labelcolor="#d7e2ee",
            )
            if leg is not None:
                leg.get_frame().set_facecolor((0.0, 0.0, 0.0, 0.0))
            fig.tight_layout()
            out = processed_dir / out_name
            fig.savefig(out, facecolor=(0.0, 0.0, 0.0, 0.0), transparent=True, bbox_inches="tight")
            paths.append(out)
        plt.close(fig)

    return paths




# =============================================================================
# One-DAQ combined metrics: angle + radial offset + Z offset from the same curl
# =============================================================================

def _one_daq_metrics_mode_from_record(rec: Dict[str, Any]) -> Optional[str]:
    """
    Detect records from a single fixed-stage curl/uncurl DAQ that should yield all
    metrics at once. This accepts the explicit one_daq_all_metrics tag and also
    ordinary angle_command / BATTACK records, because angle_command already keeps
    XYZ fixed by default.
    """
    text = " ".join(str(rec.get(k) or "") for k in ("block_name", "phase_name", "image_name")).upper()
    if any(tok in text for tok in ("ALLMETRICS", "ONE_DAQ", "ONE-DAQ", "FIXED_XYZ_ALL_METRICS")):
        return "one_daq_all_metrics"
    if "ANGLE_COMMAND" in text or "BATTACK" in text:
        return "angle_command_single_daq"
    return None


def _fit_circle_for_one_daq(u_vals: np.ndarray, z_vals: np.ndarray) -> Optional[Tuple[float, float, float]]:
    uu = np.asarray(u_vals, dtype=float).reshape(-1)
    zz = np.asarray(z_vals, dtype=float).reshape(-1)
    finite = np.isfinite(uu) & np.isfinite(zz)
    uu = uu[finite]
    zz = zz[finite]
    if uu.size < 3:
        return None
    A = np.column_stack([uu, zz, np.ones_like(uu)])
    b = -(uu ** 2 + zz ** 2)
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
    except Exception:
        return None
    a_coef, b_coef, c_coef = [float(v) for v in sol]
    uc = -0.5 * a_coef
    zc = -0.5 * b_coef
    rad_sq = uc * uc + zc * zc - c_coef
    if not np.isfinite(rad_sq) or rad_sq <= 0.0:
        return None
    return uc, zc, float(np.sqrt(rad_sq))


def _output_angle_from_circle_scaled_to_command(
    u_vals: np.ndarray,
    z_vals: np.ndarray,
    command_angle_deg: np.ndarray,
) -> np.ndarray:
    """
    Compute output angle from measured XZ arc geometry and linearly orient/scale it
    onto the command-angle span. This intentionally mirrors the existing
    hysteresis_output_angle_vs_input_attack_angle plot so the metric and plot agree.
    """
    u_vals = np.asarray(u_vals, dtype=float)
    z_vals = np.asarray(z_vals, dtype=float)
    command_angle_deg = np.asarray(command_angle_deg, dtype=float)
    fit = _fit_circle_for_one_daq(u_vals, z_vals)
    if fit is None:
        return np.full_like(command_angle_deg, np.nan, dtype=float)
    uc, zc, _ = fit
    raw_deg = np.degrees(np.unwrap(np.arctan2(u_vals - uc, z_vals - zc)))
    finite = np.isfinite(raw_deg) & np.isfinite(command_angle_deg)
    if int(np.count_nonzero(finite)) < 2:
        return np.full_like(command_angle_deg, np.nan, dtype=float)

    raw_fit = raw_deg[finite]
    cmd_fit = command_angle_deg[finite]
    raw_min = float(np.min(raw_fit))
    raw_max = float(np.max(raw_fit))
    cmd_min = float(np.min(cmd_fit))
    cmd_max = float(np.max(cmd_fit))
    raw_span = raw_max - raw_min
    cmd_span = cmd_max - cmd_min
    if raw_span < 1e-9 or cmd_span < 1e-9:
        return np.full_like(command_angle_deg, np.nan, dtype=float)

    # Choose the orientation that best agrees with command order.
    try:
        coeffs, *_ = np.linalg.lstsq(np.column_stack([raw_fit, np.ones_like(raw_fit)]), cmd_fit, rcond=None)
        slope = float(coeffs[0])
    except Exception:
        slope = 1.0
    if slope >= 0.0:
        out = cmd_min + (raw_deg - raw_min) * (cmd_span / raw_span)
    else:
        out = cmd_max - (raw_deg - raw_min) * (cmd_span / raw_span)
    return np.asarray(out, dtype=float)


def _commanded_model_angle_at_b(
    rec: Dict[str, Any],
    robot_cal: Optional[Calibration],
    angle_fit: str,
) -> Tuple[Optional[float], str]:
    """
    Reconstruct commanded/output attack angle from the selected B->angle model at
    the captured B value. This lets one physical DAQ be reprocessed against many
    angle-fit definitions. If no robot calibration is available, it falls back to
    the angle embedded in the filename.
    """
    b = _finite_float_or_none(rec.get("b_cmd"))
    phase = _normalize_motion_phase_name(rec.get("motion_phase")) or None
    fallback_angle = _finite_float_or_none(rec.get("tip_angle_deg_from_name"))
    if robot_cal is None or b is None:
        return fallback_angle, "filename_tip_angle_fallback"
    fit_for_this = phase if str(angle_fit).strip().lower() == "phase_specific" else str(angle_fit)
    try:
        model, model_label = select_tip_angle_model_for_mode(
            cal=robot_cal,
            angle_fit=fit_for_this,
            phase=phase,
        )
        val = eval_tip_angle_deg_with_model(model, float(b))
        return float(np.asarray(val, dtype=float)), model_label
    except Exception as exc:
        rec["one_daq_metrics_warning"] = str(exc)
        return fallback_angle, "angle_reconstruction_failed_filename_fallback"


def _commanded_model_offset_at_b(
    rec: Dict[str, Any],
    robot_cal: Optional[Calibration],
    output_axis: str,
    offset_fit: str,
    flip_rz_sign: bool,
) -> Tuple[Optional[float], str]:
    b = _finite_float_or_none(rec.get("b_cmd"))
    phase = _normalize_motion_phase_name(rec.get("motion_phase")) or None
    if robot_cal is None or b is None:
        return None, "missing_robot_calibration_or_b"
    fit_for_this = phase if str(offset_fit).strip().lower() == "phase_specific" else str(offset_fit)
    try:
        model, model_label = select_output_offset_model_for_mode(
            cal=robot_cal,
            output_axis=output_axis,
            fit_mode=fit_for_this,
            phase=phase,
        )
        val = eval_output_offset_with_model(
            cal=robot_cal,
            model=model,
            output_axis=output_axis,
            b=float(b),
            flip_rz_sign=bool(flip_rz_sign),
        )
        return float(np.asarray(val, dtype=float)), model_label
    except Exception as exc:
        rec["one_daq_metrics_warning"] = str(exc)
        return None, "offset_reconstruction_failed"


def _finite_or_nan(value: Any) -> float:
    try:
        x = float(value)
        return x if np.isfinite(x) else float("nan")
    except Exception:
        return float("nan")


def _basic_metric_summary(error: np.ndarray) -> Dict[str, Any]:
    arr = np.asarray(error, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {
            "num_samples": 0,
            "rmse": None,
            "mean_error": None,
            "mean_abs_error": None,
            "median_abs_error": None,
            "max_abs_error": None,
        }
    abs_arr = np.abs(arr)
    return {
        "num_samples": int(arr.size),
        "rmse": float(np.sqrt(np.mean(arr ** 2))),
        "mean_error": float(np.mean(arr)),
        "mean_abs_error": float(np.mean(abs_arr)),
        "median_abs_error": float(np.median(abs_arr)),
        "max_abs_error": float(np.max(abs_arr)),
    }


def attach_one_daq_all_metrics(
    records: List[Dict[str, Any]],
    robot_cal: Optional[Calibration] = None,
    angle_fit: str = "avg_pchip",
    offset_fit: str = "avg_pchip",
    flip_rz_sign: bool = False,
) -> Dict[str, Any]:
    """
    From one fixed-stage curl/uncurl acquisition, compute three commanded-vs-measured
    hysteresis metrics:
      1) commanded attack angle vs measured output angle
      2) commanded radial/r(B) offset vs measured checkerboard u offset
      3) commanded z(B) offset vs measured checkerboard z offset

    This is intended for one_daq_all_metrics / fixed_xyz_all_metrics, but ordinary
    angle_command runs also qualify because their default motion keeps XYZ fixed.
    """
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for rec in records:
        mode = _one_daq_metrics_mode_from_record(rec)
        if mode is None or not rec.get("valid", False):
            continue
        if rec.get("u_mm") is None or rec.get("z_mm") is None:
            continue
        if rec.get("b_cmd") is None or rec.get("tip_angle_deg_from_name") is None:
            continue
        group = str(rec.get("block_name") or rec.get("phase_name") or mode)
        groups.setdefault(group, []).append(rec)

    metrics: Dict[str, Any] = {
        "num_groups": int(len(groups)),
        "angle_fit": str(angle_fit),
        "offset_fit": str(offset_fit),
        "fit_branch": _fit_branch_label(getattr(robot_cal, "active_fit_branch", None)) if robot_cal is not None else "global",
        "fit_branch_title": _fit_branch_title(getattr(robot_cal, "active_fit_branch", None)) if robot_cal is not None else "global/default",
        "robot_calibration_available": bool(robot_cal is not None),
        "definition": {
            "angle": "x=commanded/output attack angle from selected B->angle model at each captured B; y=output angle from measured XZ circle fit",
            "radial": "x=commanded r(B) from selected model relative to first sample; y=measured checkerboard u relative to first sample",
            "z": "x=commanded z(B) from selected model relative to first sample; y=measured checkerboard z relative to first sample",
        },
        "groups": {},
        "overall": {},
    }
    if not groups:
        return metrics

    all_angle_err: List[float] = []
    all_radial_err: List[float] = []
    all_z_err: List[float] = []

    for group, rows in sorted(groups.items(), key=lambda kv: kv[0]):
        rows.sort(key=lambda r: int(r.get("sample_index", 0)))
        u = np.asarray([float(r["u_mm"]) for r in rows], dtype=float)
        z = np.asarray([float(r["z_mm"]) for r in rows], dtype=float)
        cmd_angle_vals: List[float] = []
        angle_sources: List[str] = []
        cmd_rad_abs_vals: List[float] = []
        cmd_z_abs_vals: List[float] = []
        radial_sources: List[str] = []
        z_sources: List[str] = []
        for r in rows:
            av, a_src = _commanded_model_angle_at_b(r, robot_cal, angle_fit)
            rv, rs = _commanded_model_offset_at_b(r, robot_cal, "radial", offset_fit, flip_rz_sign)
            zv, zs = _commanded_model_offset_at_b(r, robot_cal, "z", offset_fit, flip_rz_sign)
            cmd_angle_vals.append(np.nan if av is None else float(av))
            angle_sources.append(a_src)
            cmd_rad_abs_vals.append(np.nan if rv is None else float(rv))
            cmd_z_abs_vals.append(np.nan if zv is None else float(zv))
            radial_sources.append(rs)
            z_sources.append(zs)

        cmd_angle = np.asarray(cmd_angle_vals, dtype=float)
        measured_angle = _output_angle_from_circle_scaled_to_command(u, z, cmd_angle)
        angle_error = measured_angle - cmd_angle

        cmd_rad_abs = np.asarray(cmd_rad_abs_vals, dtype=float)
        cmd_z_abs = np.asarray(cmd_z_abs_vals, dtype=float)

        # If command reconstruction fails, keep metrics explicit rather than silently using output as command.
        if np.any(np.isfinite(cmd_rad_abs)):
            cmd_rad_rel = cmd_rad_abs - float(cmd_rad_abs[np.where(np.isfinite(cmd_rad_abs))[0][0]])
        else:
            cmd_rad_rel = np.full_like(u, np.nan, dtype=float)
        if np.any(np.isfinite(cmd_z_abs)):
            cmd_z_rel = cmd_z_abs - float(cmd_z_abs[np.where(np.isfinite(cmd_z_abs))[0][0]])
        else:
            cmd_z_rel = np.full_like(z, np.nan, dtype=float)

        meas_rad_rel = u - float(u[0])
        meas_z_rel = z - float(z[0])
        radial_error = meas_rad_rel - cmd_rad_rel
        z_error = meas_z_rel - cmd_z_rel

        for rec, ca, ma, ae, cra, mrr, re, cza, mzr, ze, asrc, rsrc, zsrc in zip(
            rows,
            cmd_angle,
            measured_angle,
            angle_error,
            cmd_rad_rel,
            meas_rad_rel,
            radial_error,
            cmd_z_rel,
            meas_z_rel,
            z_error,
            angle_sources,
            radial_sources,
            z_sources,
        ):
            rec["one_daq_metrics_mode"] = _one_daq_metrics_mode_from_record(rec)
            rec["one_daq_group"] = group
            rec["one_daq_angle_fit"] = str(angle_fit)
            rec["one_daq_offset_fit"] = str(offset_fit)
            rec["one_daq_fit_branch"] = _fit_branch_label(getattr(robot_cal, "active_fit_branch", None)) if robot_cal is not None else "global"
            rec["one_daq_angle_command_source"] = str(asrc)
            rec["one_daq_radial_command_source"] = str(rsrc)
            rec["one_daq_z_command_source"] = str(zsrc)
            rec["one_daq_angle_command_deg"] = None if not np.isfinite(ca) else float(ca)
            rec["one_daq_angle_measured_deg"] = None if not np.isfinite(ma) else float(ma)
            rec["one_daq_angle_error_deg"] = None if not np.isfinite(ae) else float(ae)
            rec["one_daq_radial_commanded_relative_mm"] = None if not np.isfinite(cra) else float(cra)
            rec["one_daq_radial_measured_relative_mm"] = float(mrr)
            rec["one_daq_radial_error_mm"] = None if not np.isfinite(re) else float(re)
            rec["one_daq_z_commanded_relative_mm"] = None if not np.isfinite(cza) else float(cza)
            rec["one_daq_z_measured_relative_mm"] = float(mzr)
            rec["one_daq_z_error_mm"] = None if not np.isfinite(ze) else float(ze)

        all_angle_err.extend([float(v) for v in angle_error if np.isfinite(v)])
        all_radial_err.extend([float(v) for v in radial_error if np.isfinite(v)])
        all_z_err.extend([float(v) for v in z_error if np.isfinite(v)])

        metrics["groups"][group] = {
            "num_samples": int(len(rows)),
            "motion_phases": sorted(set(str(r.get("motion_phase") or "unknown") for r in rows)),
            "angle": _basic_metric_summary(angle_error),
            "radial_offset_mm": _basic_metric_summary(radial_error),
            "z_offset_mm": _basic_metric_summary(z_error),
            "commanded_angle_range_deg": [float(np.nanmin(cmd_angle)), float(np.nanmax(cmd_angle))],
            "commanded_radial_relative_range_mm": None if not np.any(np.isfinite(cmd_rad_rel)) else [float(np.nanmin(cmd_rad_rel)), float(np.nanmax(cmd_rad_rel))],
            "measured_radial_relative_range_mm": [float(np.nanmin(meas_rad_rel)), float(np.nanmax(meas_rad_rel))],
            "commanded_z_relative_range_mm": None if not np.any(np.isfinite(cmd_z_rel)) else [float(np.nanmin(cmd_z_rel)), float(np.nanmax(cmd_z_rel))],
            "measured_z_relative_range_mm": [float(np.nanmin(meas_z_rel)), float(np.nanmax(meas_z_rel))],
            "angle_command_source": "mixed" if len(set(angle_sources)) > 1 else (angle_sources[0] if angle_sources else None),
            "radial_command_source": "mixed" if len(set(radial_sources)) > 1 else (radial_sources[0] if radial_sources else None),
            "z_command_source": "mixed" if len(set(z_sources)) > 1 else (z_sources[0] if z_sources else None),
        }

    metrics["overall"] = {
        "angle_deg": _basic_metric_summary(np.asarray(all_angle_err, dtype=float)),
        "radial_offset_mm": _basic_metric_summary(np.asarray(all_radial_err, dtype=float)),
        "z_offset_mm": _basic_metric_summary(np.asarray(all_z_err, dtype=float)),
    }
    return metrics


def save_one_daq_metrics_json(json_path: Path, metrics: Dict[str, Any]):
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(_json_ready(metrics), f, indent=2)


def save_one_daq_all_metrics_plots(
    processed_dir: Path,
    records: List[Dict[str, Any]],
    title_prefix: str = "",
) -> List[Path]:
    processed_dir = Path(processed_dir)
    paths: List[Path] = []
    specs = [
        (
            "one_daq_angle_command_deg",
            "one_daq_angle_measured_deg",
            "one_daq_angle_error_deg",
            "one_daq_angle_measured_vs_commanded.png",
            "One-DAQ angle hysteresis: measured output angle vs command",
            "Commanded input attack angle (deg)",
            "Measured output angle (deg)",
            "deg",
        ),
        (
            "one_daq_radial_commanded_relative_mm",
            "one_daq_radial_measured_relative_mm",
            "one_daq_radial_error_mm",
            "one_daq_radial_offset_measured_vs_commanded.png",
            "One-DAQ radial hysteresis: measured u offset vs commanded r(B)",
            "Commanded radial offset from start (mm)",
            "Measured checkerboard u offset from start (mm)",
            "mm",
        ),
        (
            "one_daq_z_commanded_relative_mm",
            "one_daq_z_measured_relative_mm",
            "one_daq_z_error_mm",
            "one_daq_z_offset_measured_vs_commanded.png",
            "One-DAQ Z hysteresis: measured z offset vs commanded z(B)",
            "Commanded Z offset from start (mm)",
            "Measured checkerboard z offset from start (mm)",
            "mm",
        ),
    ]

    for x_key, y_key, err_key, out_name, title, xlabel, ylabel, units in specs:
        rows = [
            r for r in records
            if r.get("valid", False)
            and r.get("one_daq_group") is not None
            and r.get(x_key) is not None
            and r.get(y_key) is not None
        ]
        if len(rows) < 2:
            continue
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in rows:
            grouped.setdefault(str(r.get("one_daq_group") or "one_daq"), []).append(r)

        fig, ax = plt.subplots(figsize=(10.2, 6.0), dpi=150)
        fig.patch.set_alpha(0.0)
        _apply_dark_axes_style(ax, f"{title_prefix}{title}".strip(), xlabel, ylabel)
        ax.set_facecolor((0.0, 0.0, 0.0, 0.0))
        ax.axline((0.0, 0.0), slope=1.0, color="#d7e2ee", linewidth=1.0, linestyle="--", alpha=0.45, label="ideal y=x")

        colors = ["#57c7ff", "#f472b6", "#79d9cf", "#f3d67a", "#a78bfa", "#34d399"]
        any_plotted = False
        ci = 0
        for group, group_rows in sorted(grouped.items(), key=lambda kv: kv[0]):
            phase_groups: Dict[str, List[Dict[str, Any]]] = {}
            for rr in sorted(group_rows, key=lambda r: int(r.get("sample_index", 0))):
                phase_groups.setdefault(str(rr.get("motion_phase") or "unknown"), []).append(rr)
            for phase, phase_rows in sorted(phase_groups.items(), key=lambda kv: kv[0]):
                x = np.asarray([float(rr[x_key]) for rr in phase_rows], dtype=float)
                y = np.asarray([float(rr[y_key]) for rr in phase_rows], dtype=float)
                finite = np.isfinite(x) & np.isfinite(y)
                if int(np.count_nonzero(finite)) < 1:
                    continue
                color = colors[ci % len(colors)]
                ci += 1
                ax.plot(
                    x[finite], y[finite],
                    marker="o", markersize=3.2, linewidth=1.5,
                    color=color, markerfacecolor=color, markeredgewidth=0.0,
                    label=f"{group} {phase}",
                )
                any_plotted = True

        if any_plotted:
            leg = ax.legend(facecolor=(0.0, 0.0, 0.0, 0.0), edgecolor=(1, 1, 1, 0.18), labelcolor="#d7e2ee", fontsize=8)
            if leg is not None:
                leg.get_frame().set_facecolor((0.0, 0.0, 0.0, 0.0))
            fig.tight_layout()
            out = processed_dir / out_name
            fig.savefig(out, facecolor=(0.0, 0.0, 0.0, 0.0), transparent=True, bbox_inches="tight")
            paths.append(out)
        plt.close(fig)

        # Also save error-vs-command for each metric.
        err_rows = [r for r in rows if r.get(err_key) is not None]
        if len(err_rows) >= 2:
            fig, ax = plt.subplots(figsize=(10.2, 6.0), dpi=150)
            fig.patch.set_alpha(0.0)
            _apply_dark_axes_style(
                ax,
                f"{title_prefix}{title.replace('measured', 'error')}".strip(),
                xlabel,
                f"Measured - commanded error ({units})",
            )
            ax.set_facecolor((0.0, 0.0, 0.0, 0.0))
            ax.axhline(0.0, color="#d7e2ee", linewidth=1.0, linestyle="--", alpha=0.45)
            ci = 0
            any_err = False
            for group, group_rows in sorted(grouped.items(), key=lambda kv: kv[0]):
                phase_groups = {}
                for rr in sorted(group_rows, key=lambda r: int(r.get("sample_index", 0))):
                    if rr.get(err_key) is None:
                        continue
                    phase_groups.setdefault(str(rr.get("motion_phase") or "unknown"), []).append(rr)
                for phase, phase_rows in sorted(phase_groups.items(), key=lambda kv: kv[0]):
                    x = np.asarray([float(rr[x_key]) for rr in phase_rows], dtype=float)
                    y = np.asarray([float(rr[err_key]) for rr in phase_rows], dtype=float)
                    finite = np.isfinite(x) & np.isfinite(y)
                    if int(np.count_nonzero(finite)) < 1:
                        continue
                    color = colors[ci % len(colors)]
                    ci += 1
                    ax.plot(x[finite], y[finite], marker="o", markersize=3.2, linewidth=1.5,
                            color=color, markerfacecolor=color, markeredgewidth=0.0,
                            label=f"{group} {phase}")
                    any_err = True
            if any_err:
                leg = ax.legend(facecolor=(0.0, 0.0, 0.0, 0.0), edgecolor=(1, 1, 1, 0.18), labelcolor="#d7e2ee", fontsize=8)
                if leg is not None:
                    leg.get_frame().set_facecolor((0.0, 0.0, 0.0, 0.0))
                fig.tight_layout()
                err_out = processed_dir / out_name.replace("measured_vs_commanded", "error_vs_commanded")
                fig.savefig(err_out, facecolor=(0.0, 0.0, 0.0, 0.0), transparent=True, bbox_inches="tight")
                paths.append(err_out)
            plt.close(fig)

    return paths


def _sanitize_fit_label_for_path(value: Any) -> str:
    text = str(value or "default").strip().lower()
    text = text.replace("/", "_").replace("\\", "_").replace(" ", "_")
    return re.sub(r"[^a-z0-9_.-]+", "_", text).strip("_") or "default"


def _normalize_requested_fit_modes(value: Any, fallback: Optional[List[str]] = None) -> List[str]:
    """Normalize CLI-provided fit mode lists. Supports comma-separated strings or nargs lists."""
    if value is None:
        return list(fallback or [])
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    else:
        parts = []
        for item in value:
            if item is None:
                continue
            parts.extend([p.strip() for p in str(item).replace(";", ",").split(",") if p.strip()])
    out: List[str] = []
    valid = {str(x).lower() for x in ANGLE_FIT_CHOICES}
    aliases = {"pchip_avg": "avg_pchip", "cubic_avg": "avg_cubic"}
    for part in parts:
        mode = aliases.get(part.strip().lower(), part.strip().lower())
        if mode not in valid and mode not in ("avg_pchip", "avg_cubic"):
            raise ValueError(f"Unsupported fit mode requested for processing sweep: {part}")
        if mode not in out:
            out.append(mode)
    return out or list(fallback or [])


def _plot_one_daq_metric_on_axes(
    ax,
    records: List[Dict[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
    show_legend: bool = True,
):
    rows = [
        r for r in records
        if r.get("valid", False)
        and r.get("one_daq_group") is not None
        and r.get(x_key) is not None
        and r.get(y_key) is not None
    ]
    _apply_dark_axes_style(ax, title, xlabel, ylabel)
    ax.set_facecolor((0.0, 0.0, 0.0, 0.0))
    ax.axline((0.0, 0.0), slope=1.0, color="#d7e2ee", linewidth=1.0, linestyle="--", alpha=0.45, label="ideal y=x")
    if len(rows) < 2:
        ax.text(0.5, 0.5, "Not enough valid data", ha="center", va="center", transform=ax.transAxes, color="#d7e2ee", fontsize=10)
        return False

    grouped: Dict[str, List[Dict[str, Any]]] = {}
    for r in rows:
        grouped.setdefault(str(r.get("one_daq_group") or "one_daq"), []).append(r)

    colors = ["#57c7ff", "#f472b6", "#79d9cf", "#f3d67a", "#a78bfa", "#34d399"]
    any_plotted = False
    ci = 0
    for group, group_rows in sorted(grouped.items(), key=lambda kv: kv[0]):
        phase_groups: Dict[str, List[Dict[str, Any]]] = {}
        for rr in sorted(group_rows, key=lambda r: int(r.get("sample_index", 0))):
            phase_groups.setdefault(str(rr.get("motion_phase") or "unknown"), []).append(rr)
        for phase, phase_rows in sorted(phase_groups.items(), key=lambda kv: kv[0]):
            x = np.asarray([float(rr[x_key]) for rr in phase_rows], dtype=float)
            y = np.asarray([float(rr[y_key]) for rr in phase_rows], dtype=float)
            finite = np.isfinite(x) & np.isfinite(y)
            if int(np.count_nonzero(finite)) < 1:
                continue
            color = colors[ci % len(colors)]
            ci += 1
            ax.plot(
                x[finite], y[finite],
                marker="o", markersize=2.8, linewidth=1.35,
                color=color, markerfacecolor=color, markeredgewidth=0.0,
                label=f"{group} {phase}",
            )
            any_plotted = True

    if show_legend and any_plotted:
        leg = ax.legend(facecolor=(0.0, 0.0, 0.0, 0.0), edgecolor=(1, 1, 1, 0.18), labelcolor="#d7e2ee", fontsize=7, loc="best")
        if leg is not None:
            leg.get_frame().set_facecolor((0.0, 0.0, 0.0, 0.0))
    return any_plotted


def save_one_daq_fit_sweep_panel(
    output_path: Path,
    fit_run_payloads: List[Dict[str, Any]],
    title_prefix: str = "Checkerboard-referenced ",
) -> Optional[Path]:
    """
    Build one large panel with one column per requested fit mode/pair and three rows:
    angle hysteresis, radial hysteresis, and Z hysteresis.

    This is primarily intended for paired processing sweeps, where each column
    corresponds to one fit mode. For cross sweeps, the columns correspond to the
    individual angle/offset combinations in order.
    """
    if not fit_run_payloads:
        return None

    specs = [
        (
            "one_daq_angle_command_deg",
            "one_daq_angle_measured_deg",
            "Measured output angle (deg)",
            "Commanded input attack angle (deg)",
            "Angle",
        ),
        (
            "one_daq_radial_commanded_relative_mm",
            "one_daq_radial_measured_relative_mm",
            "Measured checkerboard u offset from start (mm)",
            "Commanded radial offset from start (mm)",
            "Radial offset",
        ),
        (
            "one_daq_z_commanded_relative_mm",
            "one_daq_z_measured_relative_mm",
            "Measured checkerboard z offset from start (mm)",
            "Commanded Z offset from start (mm)",
            "Z offset",
        ),
    ]

    ncols = max(1, len(fit_run_payloads))
    fig, axes = plt.subplots(3, ncols, figsize=(4.4 * ncols, 10.8), dpi=150, squeeze=False)
    fig.patch.set_alpha(0.0)

    for col, payload in enumerate(fit_run_payloads):
        angle_fit = str(payload.get("angle_fit") or "?")
        offset_fit = str(payload.get("offset_fit") or "?")
        records = payload.get("records") or []
        same_fit = (angle_fit == offset_fit)
        col_title = angle_fit if same_fit else f"a={angle_fit}\no={offset_fit}"
        for row_idx, (x_key, y_key, ylabel, xlabel, row_name) in enumerate(specs):
            ax = axes[row_idx][col]
            title = col_title if row_idx == 0 else ""
            plotted = _plot_one_daq_metric_on_axes(
                ax=ax,
                records=records,
                x_key=x_key,
                y_key=y_key,
                title=title,
                xlabel=xlabel if row_idx == 2 else (xlabel if ncols == 1 else ""),
                ylabel=ylabel if col == 0 else "",
                show_legend=(col == 0 and row_idx == 0),
            )
            if col != 0:
                ax.set_ylabel("")
            if row_idx != 2 and ncols > 1:
                ax.set_xlabel("")
            if col == 0:
                ax.text(-0.24, 0.5, row_name, transform=ax.transAxes, rotation=90, va="center", ha="center", color="#d7e2ee", fontsize=10, fontweight="bold")

    fig.suptitle(f"{title_prefix}One-DAQ hysteresis comparison across fit modes".strip(), color="#d7e2ee", fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0.02, 0.02, 0.995, 0.98))
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=(0.0, 0.0, 0.0, 0.0), transparent=True, bbox_inches="tight")
    plt.close(fig)
    return output_path


# -----------------------------------------------------------------------------
# Calibration-model auto-discovery for "plot every model possibility"
# -----------------------------------------------------------------------------

_CAL_DISCOVERY_SUFFIX_ORDER = [
    "default", "linear", "pchip", "cubic",
    "avg_linear", "avg_pchip", "avg_cubic",
]


def _model_keys_for(base: str, kind: str, branch: Any = None) -> List[str]:
    """Generate likely keys for base model name and fit kind.

    This override keeps the earlier keys but distinguishes linear vs avg_linear
    and supports default/direct model keys. It is used by all selectors below.
    """
    base = str(base)
    kind = str(kind or "").strip().lower()
    branch_labels: List[str] = []
    b = normalize_fit_branch(branch)
    if b == "0_90":
        branch_labels = ["0_90", "0-90", "0to90", "90", "range_0_90", "branch_0_90"]
    elif b == "0_180":
        branch_labels = ["0_180", "0-180", "0to180", "180", "range_0_180", "branch_0_180"]

    if kind in ("", "default", "direct"):
        suffixes = ["", "pchip"]
    elif kind == "pchip":
        suffixes = ["pchip", "", "avg_pchip", "pchip_avg"]
    elif kind == "avg_pchip":
        suffixes = ["avg_pchip", "pchip_avg", "pchip", ""]
    elif kind == "cubic":
        suffixes = ["cubic", "avg_cubic", "cubic_avg"]
    elif kind == "avg_cubic":
        suffixes = ["avg_cubic", "cubic_avg", "cubic"]
    elif kind == "linear":
        suffixes = ["linear", "avg_linear"]
    elif kind == "avg_linear":
        suffixes = ["avg_linear", "linear"]
    elif kind in ("any", "phase_any"):
        suffixes = ["", "pchip", "linear", "cubic", "avg_pchip", "avg_linear", "avg_cubic"]
    else:
        suffixes = [kind, ""]

    keys: List[str] = []
    for suff in suffixes:
        keys.append(base if suff == "" else f"{base}_{suff}")
    for bl in branch_labels:
        for suff in suffixes:
            keys.extend([
                f"{base}_{bl}" if suff == "" else f"{base}_{bl}_{suff}",
                f"{base}_{suff}_{bl}" if suff else f"{base}_{bl}",
                f"{bl}_{base}" if suff == "" else f"{bl}_{base}_{suff}",
            ])
    out: List[str] = []
    for k in keys:
        if k not in out:
            out.append(k)
    return out


def _suffix_from_model_key(base: str, key: Any) -> Optional[str]:
    key_s = str(key or "")
    if key_s == base:
        return "default"
    prefix = f"{base}_"
    if not key_s.startswith(prefix):
        return None
    suffix = key_s[len(prefix):].strip().lower()
    if not suffix:
        return "default"
    aliases = {
        "pchip_avg": "avg_pchip",
        "cubic_avg": "avg_cubic",
        "linear_avg": "avg_linear",
    }
    return aliases.get(suffix, suffix)


def _has_model_key_for_suffix(models: Any, base: str, suffix: str) -> bool:
    if not isinstance(models, dict):
        return False
    candidates = _model_keys_for(base, suffix if suffix != "default" else "default")
    norm = {_alnum_key(k): k for k in models.keys() if isinstance(models.get(k), dict)}
    for c in candidates:
        if c in models and isinstance(models.get(c), dict):
            return True
        if _alnum_key(c) in norm:
            return True
    return False


def _discover_suffixes_in_model_container(models: Any) -> List[str]:
    """Return suffixes where tip_angle, r, and z all exist in a model container."""
    if not isinstance(models, dict):
        return []
    by_base = {"tip_angle": set(), "r": set(), "z": set()}
    for key, val in models.items():
        if not isinstance(val, dict):
            continue
        for base in by_base.keys():
            suffix = _suffix_from_model_key(base, key)
            if suffix is not None:
                by_base[base].add(suffix)
    common = set.intersection(*by_base.values()) if all(by_base.values()) else set()
    # Keep only suffixes that can be selected by our key generator for all bases.
    common = {s for s in common if all(_has_model_key_for_suffix(models, base, s) for base in by_base.keys())}
    return sorted(common, key=lambda s: (_CAL_DISCOVERY_SUFFIX_ORDER.index(s) if s in _CAL_DISCOVERY_SUFFIX_ORDER else 999, s))


def _global_model_containers_for_branch(cal: Calibration, branch: Any = None) -> List[Tuple[str, dict]]:
    branch_norm = normalize_fit_branch(branch if branch is not None else getattr(cal, "active_fit_branch", None))
    groups: List[Tuple[str, dict]] = []
    if branch_norm is not None:
        for label, container in _raw_branch_containers(cal, branch_norm):
            if isinstance(container, dict):
                fm = container.get("fit_models")
                if isinstance(fm, dict):
                    groups.append((f"{label}/fit_models", fm))
                if any(isinstance(container.get(k), dict) for k in ("r", "z", "tip_angle", "r_pchip", "z_pchip", "tip_angle_pchip")):
                    groups.append((label, container))
    global_models = getattr(cal, "raw_fit_models", {}) or {}
    if isinstance(global_models, dict):
        groups.append(("global_fit_models", global_models))
    shared = getattr(cal, "raw_shared_aux_fit_models", {}) or {}
    if isinstance(shared, dict):
        groups.append(("shared_aux_fit_models", shared))
    return groups


def _phase_model_containers_for_branch(cal: Calibration, branch: Any = None) -> List[Tuple[str, str, dict]]:
    branch_norm = normalize_fit_branch(branch if branch is not None else getattr(cal, "active_fit_branch", None))
    out: List[Tuple[str, str, dict]] = []
    # Branch containers may have nested phase maps.
    if branch_norm is not None:
        for label, container in _raw_branch_containers(cal, branch_norm):
            if not isinstance(container, dict):
                continue
            for phase_key_name in ("fit_models_by_phase", "phases", "phase_models", "models_by_phase"):
                phmap = container.get(phase_key_name)
                if isinstance(phmap, dict):
                    for pk, pv in phmap.items():
                        if isinstance(pv, dict):
                            out.append((f"{label}/{phase_key_name}/{pk}", str(pk), pv))
    raw_phase_models = getattr(cal, "raw_fit_models_by_phase", {}) or {}
    if isinstance(raw_phase_models, dict):
        for pk, pv in raw_phase_models.items():
            if not isinstance(pv, dict):
                continue
            if branch_norm is not None and not _branch_matches_key(pk, branch_norm):
                hay = " ".join(" ".join(str(v.get(x, "")) for x in ("value_name", "fit_branch", "angle_range", "range_label", "sweep_range")) for v in pv.values() if isinstance(v, dict))
                # Generic pull/release entries are still valid fallback for separate 0-90/0-180 calibration files.
                if _branch_matches_key(hay, branch_norm):
                    out.append((f"phase/{pk}", str(pk), pv))
                elif not getattr(cal, "available_fit_branches", []):
                    out.append((f"phase/{pk}", str(pk), pv))
                continue
            out.append((f"phase/{pk}", str(pk), pv))
    return out


def discover_calibration_model_fit_modes(cal: Optional[Calibration], branch: Any = None) -> Dict[str, Any]:
    """Discover every usable angle/radial/z model family present in the calibration JSON.

    Returned fit mode labels are directly accepted by this script, e.g.
    global_avg_pchip or per_phase_cubic. A label is included only when a matching
    tip_angle, r, and z model can be found in the corresponding scope.
    """
    if cal is None:
        return {"fit_modes": [], "global_modes": [], "per_phase_modes": [], "branch": _fit_branch_label(branch)}
    branch_norm = normalize_fit_branch(branch if branch is not None else getattr(cal, "active_fit_branch", None))

    global_suffixes: Set[str] = set()
    global_sources: Dict[str, List[str]] = {}
    for label, models in _global_model_containers_for_branch(cal, branch_norm):
        for suffix in _discover_suffixes_in_model_container(models):
            global_suffixes.add(suffix)
            global_sources.setdefault(suffix, []).append(label)

    phase_suffixes: Set[str] = set()
    phase_sources: Dict[str, List[str]] = {}
    phase_groups = _phase_model_containers_for_branch(cal, branch_norm)
    # Include suffixes present in at least one pull-like and one release-like group when possible.
    suffix_by_phase_kind: Dict[str, Set[str]] = {"pull": set(), "release": set(), "other": set()}
    suffix_sources_tmp: Dict[str, List[str]] = {}
    for label, phase_key, models in phase_groups:
        suffixes = _discover_suffixes_in_model_container(models)
        if not suffixes:
            continue
        pk = _normalize_motion_phase_name(phase_key) or "other"
        kind = "pull" if pk.startswith("pull") else ("release" if pk.startswith("release") else "other")
        suffix_by_phase_kind[kind].update(suffixes)
        for suffix in suffixes:
            suffix_sources_tmp.setdefault(suffix, []).append(label)
    if suffix_by_phase_kind["pull"] and suffix_by_phase_kind["release"]:
        phase_suffixes = suffix_by_phase_kind["pull"].intersection(suffix_by_phase_kind["release"])
    else:
        phase_suffixes = set().union(*suffix_by_phase_kind.values()) if suffix_by_phase_kind else set()
    for suffix in phase_suffixes:
        phase_sources[suffix] = suffix_sources_tmp.get(suffix, [])

    def sort_suffixes(values: Set[str]) -> List[str]:
        return sorted(values, key=lambda s: (_CAL_DISCOVERY_SUFFIX_ORDER.index(s) if s in _CAL_DISCOVERY_SUFFIX_ORDER else 999, s))

    global_modes = [f"global_{s}" for s in sort_suffixes(global_suffixes)]
    phase_modes = [f"per_phase_{s}" for s in sort_suffixes(phase_suffixes)]
    fit_modes = global_modes + phase_modes
    # De-duplicate while preserving order.
    dedup: List[str] = []
    for mode in fit_modes:
        if mode not in dedup:
            dedup.append(mode)

    return {
        "branch": _fit_branch_label(branch_norm),
        "global_modes": global_modes,
        "per_phase_modes": phase_modes,
        "fit_modes": dedup,
        "global_sources": global_sources,
        "phase_sources": phase_sources,
    }


def _parse_calibration_fit_mode_label(mode: Any) -> Optional[Tuple[str, str]]:
    text = str(mode or "").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "global": "global_default",
        "calibration": "global_default",
        "phase": "per_phase_default",
        "phase_default": "per_phase_default",
        "perphase_default": "per_phase_default",
    }
    text = aliases.get(text, text)
    if text.startswith("model:"):
        text = text.split(":", 1)[1]
    if text.startswith("global_"):
        return "global", text[len("global_"):]
    if text.startswith("cal_global_"):
        return "global", text[len("cal_global_"):]
    if text.startswith("per_phase_"):
        return "per_phase", text[len("per_phase_"):]
    if text.startswith("phase_"):
        return "per_phase", text[len("phase_"):]
    return None


def _select_direct_model_from_containers(cal: Calibration, base: str, suffix: str, branch: Any = None) -> Tuple[Optional[Any], str]:
    branch_norm = normalize_fit_branch(branch if branch is not None else getattr(cal, "active_fit_branch", None))
    for label, models in _global_model_containers_for_branch(cal, branch_norm):
        model = _dict_model_by_candidate_keys(models, _model_keys_for(base, suffix, branch_norm), branch=branch_norm)
        if model is not None:
            return model, f"{label} | branch={_fit_branch_label(branch_norm)} | {base}_{suffix}"
    return None, f"no-global-direct-model branch={_fit_branch_label(branch_norm)} {base}_{suffix}"


def _select_per_phase_model_from_containers(cal: Calibration, base: str, suffix: str, phase: Optional[str], branch: Any = None) -> Tuple[Optional[Any], str]:
    branch_norm = normalize_fit_branch(branch if branch is not None else getattr(cal, "active_fit_branch", None))
    phase_norm = _normalize_motion_phase_name(phase) or "pull"
    # First try the general branch-aware selector, which prefers branch phase containers.
    model, label = _select_model_from_groups(cal, base, suffix, phase=phase_norm, branch=branch_norm)
    if model is not None:
        return model, f"per-phase-direct | {label}"
    # Then direct scan phase containers.
    for label, phase_key, models in _phase_model_containers_for_branch(cal, branch_norm):
        if not _phase_matches_key(phase_key, phase_norm):
            continue
        model = _dict_model_by_candidate_keys(models, _model_keys_for(base, suffix, branch_norm), branch=branch_norm)
        if model is not None:
            return model, f"{label} | branch={_fit_branch_label(branch_norm)} | phase={phase_norm} | {base}_{suffix}"
    return None, f"no-per-phase-direct-model branch={_fit_branch_label(branch_norm)} phase={phase_norm} {base}_{suffix}"


_BASE_SELECT_TIP_ANGLE_MODEL_FOR_MODE = select_tip_angle_model_for_mode
_BASE_SELECT_OUTPUT_OFFSET_MODEL_FOR_MODE = select_output_offset_model_for_mode


def select_tip_angle_model_for_mode(cal: Calibration, angle_fit: str, phase: Optional[str] = None):
    parsed = _parse_calibration_fit_mode_label(angle_fit)
    if parsed is not None:
        scope, suffix = parsed
        branch = normalize_fit_branch(getattr(cal, "active_fit_branch", None))
        if scope == "global":
            model, label = _select_direct_model_from_containers(cal, "tip_angle", suffix, branch=branch)
            if model is not None:
                return model, f"{angle_fit} | {label}"
        else:
            model, label = _select_per_phase_model_from_containers(cal, "tip_angle", suffix, phase=phase, branch=branch)
            if model is not None:
                return model, f"{angle_fit} | {label}"
        # Fall back to the original abstract modes if a direct discovered model disappears.
        return _BASE_SELECT_TIP_ANGLE_MODEL_FOR_MODE(cal, suffix, phase=phase)
    return _BASE_SELECT_TIP_ANGLE_MODEL_FOR_MODE(cal, angle_fit, phase=phase)


def select_output_offset_model_for_mode(
    cal: Calibration,
    output_axis: str,
    fit_mode: str,
    phase: Optional[str] = None,
):
    parsed = _parse_calibration_fit_mode_label(fit_mode)
    if parsed is not None:
        scope, suffix = parsed
        axis = str(output_axis or "radial").strip().lower()
        if axis in ("horizontal", "x", "r"):
            axis = "radial"
        model_name = "r" if axis == "radial" else "z"
        branch = normalize_fit_branch(getattr(cal, "active_fit_branch", None))
        if scope == "global":
            model, label = _select_direct_model_from_containers(cal, model_name, suffix, branch=branch)
            if model is not None:
                return model, f"{fit_mode}-{model_name} | {label}"
        else:
            model, label = _select_per_phase_model_from_containers(cal, model_name, suffix, phase=phase, branch=branch)
            if model is not None:
                return model, f"{fit_mode}-{model_name} | {label}"
        return _BASE_SELECT_OUTPUT_OFFSET_MODEL_FOR_MODE(cal, output_axis, suffix, phase=phase)
    return _BASE_SELECT_OUTPUT_OFFSET_MODEL_FOR_MODE(cal, output_axis, fit_mode, phase=phase)


def _normalize_requested_fit_modes(value: Any, fallback: Optional[List[str]] = None) -> List[str]:
    """Normalize fit mode lists. Allows abstract modes and discovered calibration-model labels."""
    if value is None:
        return list(fallback or [])
    if isinstance(value, str):
        parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    else:
        parts = []
        for item in value:
            if item is None:
                continue
            parts.extend([p.strip() for p in str(item).replace(";", ",").split(",") if p.strip()])
    out: List[str] = []
    aliases = {"pchip_avg": "avg_pchip", "cubic_avg": "avg_cubic"}
    for part in parts:
        mode = aliases.get(part.strip().lower(), part.strip().lower().replace("-", "_"))
        if mode and mode not in out:
            out.append(mode)
    return out or list(fallback or [])



def save_one_daq_fit_sweep_outputs(

    processed_dir: Path,
    records: List[Dict[str, Any]],
    robot_cal: Optional[Calibration],
    angle_fit_modes: List[str],
    offset_fit_modes: List[str],
    flip_rz_sign: bool = False,
    combination_mode: str = "paired",
    output_root_name: str = "fit_sweeps",
    title_prefix: str = "Checkerboard-referenced ",
) -> Dict[str, Any]:
    """
    Reuse one tracked-image dataset and recompute one-DAQ angle/radial/Z metrics for
    multiple requested angle/offset model choices. Each result is isolated in its own
    subfolder so figures are not overwritten.
    """
    processed_dir = Path(processed_dir)
    root = processed_dir / str(output_root_name or "fit_sweeps")
    root.mkdir(parents=True, exist_ok=True)

    angle_modes = _normalize_requested_fit_modes(angle_fit_modes, fallback=["avg_pchip"])
    offset_modes = _normalize_requested_fit_modes(offset_fit_modes, fallback=angle_modes)
    combo_mode = str(combination_mode or "paired").strip().lower()
    if combo_mode not in ("paired", "cross"):
        combo_mode = "paired"

    if combo_mode == "cross":
        combos = [(a, o) for a in angle_modes for o in offset_modes]
    else:
        # Paired mode is the safe default: avg_pchip uses avg_pchip, pull uses pull, etc.
        # If the two lists have different lengths, cycle the shorter list so every requested
        # entry is still represented without producing a large cross-product by accident.
        n = max(len(angle_modes), len(offset_modes))
        combos = [(angle_modes[i % len(angle_modes)], offset_modes[i % len(offset_modes)]) for i in range(n)]

    active_branch_label = _fit_branch_label(getattr(robot_cal, "active_fit_branch", None)) if robot_cal is not None else "global"
    active_branch_title = _fit_branch_title(getattr(robot_cal, "active_fit_branch", None)) if robot_cal is not None else "global/default"

    summary: Dict[str, Any] = {
        "output_root": str(root),
        "fit_branch": active_branch_label,
        "fit_branch_title": active_branch_title,
        "combination_mode": combo_mode,
        "angle_fit_modes": list(angle_modes),
        "offset_fit_modes": list(offset_modes),
        "num_combinations": int(len(combos)),
        "robot_calibration_available": bool(robot_cal is not None),
        "runs": [],
    }

    panel_payloads: List[Dict[str, Any]] = []

    for angle_fit, offset_fit in combos:
        subdir = root / f"angle_{_sanitize_fit_label_for_path(angle_fit)}__offset_{_sanitize_fit_label_for_path(offset_fit)}"
        subdir.mkdir(parents=True, exist_ok=True)
        fit_records = [dict(r) for r in records]
        metrics = attach_one_daq_all_metrics(
            fit_records,
            robot_cal=robot_cal,
            angle_fit=str(angle_fit),
            offset_fit=str(offset_fit),
            flip_rz_sign=bool(flip_rz_sign),
        )
        json_path = subdir / "one_daq_all_metrics.json"
        csv_path = subdir / "tracked_tip_positions_with_fit_metrics.csv"
        save_one_daq_metrics_json(json_path, metrics)
        save_tracked_tip_csv(csv_path, fit_records)
        plot_paths = save_one_daq_all_metrics_plots(
            processed_dir=subdir,
            records=fit_records,
            title_prefix=f"{title_prefix}angle={angle_fit}, offset={offset_fit} | ",
        )
        panel_payloads.append({
            "angle_fit": str(angle_fit),
            "offset_fit": str(offset_fit),
            "fit_branch": active_branch_label,
            "records": fit_records,
        })
        summary["runs"].append({
            "angle_fit": str(angle_fit),
            "offset_fit": str(offset_fit),
            "fit_branch": active_branch_label,
            "output_dir": str(subdir),
            "metrics_json": str(json_path),
            "csv": str(csv_path),
            "plots": [str(p) for p in plot_paths],
            "overall": metrics.get("overall", {}),
        })

    panel_path = root / f"one_daq_all_fits_panel_{_sanitize_fit_label_for_path(active_branch_label)}.png"
    panel_saved = save_one_daq_fit_sweep_panel(
        panel_path,
        panel_payloads,
        title_prefix=f"{title_prefix}{active_branch_title} branch | ",
    )
    if panel_saved is not None:
        summary["panel_plot"] = str(panel_saved)

    summary_path = root / "fit_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(_json_ready(summary), f, indent=2)
    summary["summary_json"] = str(summary_path)
    return summary


def _offset_command_mode_from_record(rec: Dict[str, Any]) -> Optional[str]:
    text = " ".join(str(rec.get(k) or "") for k in ("block_name", "phase_name", "image_name")).upper()
    if "HOFF" in text or "ROFF" in text or "HORIZONTAL_OFFSET" in text or "RADIAL_OFFSET" in text:
        return "horizontal_offset"
    if "ZOFF" in text or "VOFF" in text or "Z_OFFSET" in text or "VERTICAL_OFFSET" in text:
        return "z_offset"
    return None


def _commanded_offset_from_record(
    rec: Dict[str, Any],
    robot_cal: Optional[Calibration],
    mode: str,
    offset_fit: str = "avg_pchip",
    flip_rz_sign: bool = False,
) -> Tuple[Optional[float], str]:
    b = _finite_float_or_none(rec.get("b_cmd"))
    phase = _normalize_motion_phase_name(rec.get("motion_phase")) or None
    if robot_cal is None or b is None:
        return None, "missing_robot_calibration_or_b"
    axis = "radial" if mode == "horizontal_offset" else "z"
    fit_for_this = phase if str(offset_fit).strip().lower() == "phase_specific" else str(offset_fit)
    try:
        model, model_label = select_output_offset_model_for_mode(
            cal=robot_cal,
            output_axis=axis,
            fit_mode=fit_for_this,
            phase=phase,
        )
        val = eval_output_offset_with_model(
            cal=robot_cal,
            model=model,
            output_axis=axis,
            b=float(b),
            flip_rz_sign=bool(flip_rz_sign),
        )
        return float(np.asarray(val, dtype=float)), model_label
    except Exception as exc:
        rec["offset_command_reconstruction_warning"] = str(exc)
        return None, "offset_reconstruction_failed"


def attach_offset_command_hysteresis(
    records: List[Dict[str, Any]],
    robot_cal: Optional[Calibration] = None,
    offset_fit: str = "avg_pchip",
    flip_rz_sign: bool = False,
) -> Dict[str, Any]:
    """
    Attach measured-vs-commanded hysteresis data for the new fixed-stage modes.

    horizontal_offset/radial_offset:
        x-axis command: calibrated radial/horizontal offset r(B), relative to the
        first sample in the block.
        measured output: checkerboard u displacement relative to the first sample.

    z_offset/vertical_offset:
        x-axis command: calibrated z offset z(B), relative to the first sample.
        measured output: checkerboard z displacement relative to the first sample.
    """
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for rec in records:
        mode = _offset_command_mode_from_record(rec)
        if mode is None or not rec.get("valid", False):
            continue
        if rec.get("u_mm") is None or rec.get("z_mm") is None:
            continue
        group = str(rec.get("block_name") or rec.get("phase_name") or mode)
        groups.setdefault((mode, group), []).append(rec)

    metrics: Dict[str, Any] = {
        "num_offset_groups": int(len(groups)),
        "offset_fit": str(offset_fit),
        "robot_calibration_available": bool(robot_cal is not None),
        "groups": {},
        "by_mode": {},
        "definition": {
            "horizontal_offset": {
                "command": "calibrated radial/horizontal offset r(B) from selected model, relative to first sample",
                "measurement": "checkerboard u displacement relative to first sample",
            },
            "z_offset": {
                "command": "calibrated z offset z(B) from selected model, relative to first sample",
                "measurement": "checkerboard z displacement relative to first sample",
            },
        },
    }
    if not groups:
        return metrics

    by_mode_errors: Dict[str, List[float]] = {}
    by_mode_cmd: Dict[str, List[float]] = {}

    for (mode, group), rows in sorted(groups.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        rows.sort(key=lambda r: int(r.get("sample_index", 0)))
        if mode == "horizontal_offset":
            measured_abs = np.asarray([float(r["u_mm"]) for r in rows], dtype=float)
            measured_axis = "u"
            command_axis = "radial"
        else:
            measured_abs = np.asarray([float(r["z_mm"]) for r in rows], dtype=float)
            measured_axis = "z"
            command_axis = "z"

        cmd_pairs = [
            _commanded_offset_from_record(
                rec=r,
                robot_cal=robot_cal,
                mode=mode,
                offset_fit=offset_fit,
                flip_rz_sign=bool(flip_rz_sign),
            )
            for r in rows
        ]
        commanded_abs = np.asarray([
            np.nan if pair[0] is None else float(pair[0])
            for pair in cmd_pairs
        ], dtype=float)
        sources = [pair[1] for pair in cmd_pairs]
        command_source = sources[0] if len(set(sources)) == 1 else "mixed"

        finite_cmd = np.isfinite(commanded_abs)
        if not np.any(finite_cmd):
            # Last-resort fallback so a diagnostic plot is still created.
            commanded_abs = measured_abs.copy()
            command_source = "measured_output_fallback_no_commanded_offset"
        elif not np.all(finite_cmd):
            # Fill gaps by interpolation in sample order if possible; otherwise use measured output.
            idx = np.arange(commanded_abs.size, dtype=float)
            fill = commanded_abs.copy()
            good = np.isfinite(fill)
            if int(np.count_nonzero(good)) >= 2:
                fill[~good] = np.interp(idx[~good], idx[good], fill[good])
            else:
                fill[~good] = measured_abs[~good]
            commanded_abs = fill
            command_source = command_source + "+filled_missing"

        cmd0 = float(commanded_abs[0]) if commanded_abs.size and np.isfinite(commanded_abs[0]) else 0.0
        meas0 = float(measured_abs[0]) if measured_abs.size and np.isfinite(measured_abs[0]) else 0.0
        commanded_rel = commanded_abs - cmd0
        measured_rel = measured_abs - meas0
        error = measured_rel - commanded_rel
        abs_err = np.abs(error)

        by_mode_errors.setdefault(mode, []).extend(abs_err.astype(float).tolist())
        by_mode_cmd.setdefault(mode, []).extend(commanded_rel.astype(float).tolist())

        for rec, c_abs, c_rel, m_abs, m_rel, err, ae in zip(
            rows, commanded_abs, commanded_rel, measured_abs, measured_rel, error, abs_err
        ):
            rec["offset_command_mode"] = mode
            rec["offset_command_axis"] = command_axis
            rec["offset_commanded_mm"] = float(c_abs)
            rec["offset_commanded_relative_mm"] = float(c_rel)
            rec["offset_measured_axis"] = measured_axis
            rec["offset_measured_mm"] = float(m_abs)
            rec["offset_measured_relative_mm"] = float(m_rel)
            rec["offset_error_mm"] = float(err)
            rec["offset_abs_error_mm"] = float(ae)
            rec["offset_command_source"] = command_source

        finite_cmd_rel = commanded_rel[np.isfinite(commanded_rel)]
        finite_meas_rel = measured_rel[np.isfinite(measured_rel)]
        key = f"{mode}:{group}"
        metrics["groups"][key] = {
            "offset_command_mode": mode,
            "group": group,
            "num_samples": int(len(rows)),
            "command_axis": command_axis,
            "measured_axis": measured_axis,
            "command_source": command_source,
            "rmse_measured_minus_commanded_mm": float(np.sqrt(np.mean(error ** 2))),
            "mean_abs_error_mm": float(np.mean(abs_err)),
            "median_abs_error_mm": float(np.median(abs_err)),
            "max_abs_error_mm": float(np.max(abs_err)),
            "commanded_relative_range_mm": None if finite_cmd_rel.size == 0 else [float(np.min(finite_cmd_rel)), float(np.max(finite_cmd_rel))],
            "measured_relative_range_mm": None if finite_meas_rel.size == 0 else [float(np.min(finite_meas_rel)), float(np.max(finite_meas_rel))],
        }

    for mode, errs in by_mode_errors.items():
        arr = np.asarray(errs, dtype=float)
        cmd_arr = np.asarray(by_mode_cmd.get(mode, []), dtype=float)
        cmd_arr = cmd_arr[np.isfinite(cmd_arr)]
        metrics["by_mode"][mode] = {
            "num_samples": int(arr.size),
            "rmse_measured_minus_commanded_mm": float(np.sqrt(np.mean(arr ** 2))) if arr.size else None,
            "mean_abs_error_mm": float(np.mean(arr)) if arr.size else None,
            "median_abs_error_mm": float(np.median(arr)) if arr.size else None,
            "max_abs_error_mm": float(np.max(arr)) if arr.size else None,
            "commanded_relative_range_mm": None if cmd_arr.size == 0 else [float(np.min(cmd_arr)), float(np.max(cmd_arr))],
        }
    return metrics


def save_offset_command_metrics_json(json_path: Path, offset_metrics: Dict[str, Any]):
    json_path = Path(json_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    with open(json_path, "w") as f:
        json.dump(_json_ready(offset_metrics), f, indent=2)


def save_offset_command_hysteresis_plots(
    processed_dir: Path,
    records: List[Dict[str, Any]],
    offset_metrics: Dict[str, Any],
    title_prefix: str = "",
) -> List[Path]:
    """Save measured-vs-commanded hysteresis plots for fixed-stage offset modes."""
    processed_dir = Path(processed_dir)
    paths: List[Path] = []
    palette = ["#57c7ff", "#79d9cf", "#f3d67a", "#f472b6", "#a78bfa", "#34d399"]

    specs = {
        "horizontal_offset": {
            "title": "Horizontal/radial offset hysteresis: measured u offset vs commanded radial offset",
            "x_label": "Commanded radial/horizontal offset from start (mm)",
            "y_label": "Measured checkerboard u offset from start (mm)",
            "out_name": "hysteresis_horizontal_offset_measured_u_vs_commanded_radial.png",
        },
        "z_offset": {
            "title": "Z offset hysteresis: measured z offset vs commanded z offset",
            "x_label": "Commanded Z offset from start (mm)",
            "y_label": "Measured checkerboard z offset from start (mm)",
            "out_name": "hysteresis_z_offset_measured_z_vs_commanded_z.png",
        },
    }

    for mode, spec in specs.items():
        mode_rows = [
            r for r in records
            if r.get("valid", False)
            and r.get("offset_command_mode") == mode
            and r.get("offset_commanded_relative_mm") is not None
            and r.get("offset_measured_relative_mm") is not None
        ]
        if len(mode_rows) < 2:
            continue

        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for r in mode_rows:
            grouped.setdefault(str(r.get("block_name") or mode), []).append(r)

        fig, ax = plt.subplots(figsize=(10.2, 6.0), dpi=150)
        fig.patch.set_alpha(0.0)
        _apply_dark_axes_style(
            ax,
            f"{title_prefix}{spec['title']}".strip(),
            spec["x_label"],
            spec["y_label"],
        )
        ax.set_facecolor((0.0, 0.0, 0.0, 0.0))
        ax.axhline(0.0, color="#d7e2ee", linewidth=1.0, linestyle="--", alpha=0.45)
        ax.axvline(0.0, color="#d7e2ee", linewidth=1.0, linestyle="--", alpha=0.45)

        all_x: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        any_plotted = False
        trace_i = 0
        for group, rows in sorted(grouped.items(), key=lambda kv: kv[0]):
            rows = sorted(rows, key=lambda rr: int(rr.get("sample_index", 0)))
            phase_groups: Dict[str, List[Dict[str, Any]]] = {}
            for rr in rows:
                phase_groups.setdefault(str(rr.get("motion_phase") or rr.get("phase_name") or "unknown"), []).append(rr)
            for phase, rr_list in sorted(phase_groups.items(), key=lambda kv: kv[0]):
                x = np.asarray([float(rr["offset_commanded_relative_mm"]) for rr in rr_list], dtype=float)
                y = np.asarray([float(rr["offset_measured_relative_mm"]) for rr in rr_list], dtype=float)
                finite = np.isfinite(x) & np.isfinite(y)
                if int(np.count_nonzero(finite)) < 2:
                    continue
                x = x[finite]
                y = y[finite]
                all_x.append(x)
                all_y.append(y)
                color = palette[trace_i % len(palette)]
                ax.plot(
                    x,
                    y,
                    marker="o",
                    markersize=3.2,
                    linewidth=1.5,
                    color=color,
                    markerfacecolor=color,
                    markeredgewidth=0.0,
                    label=f"{group} {phase}",
                )
                any_plotted = True
                trace_i += 1

        # Add a 1:1 ideal line over the plotted range.
        if all_x and all_y:
            xy = np.concatenate(all_x + all_y)
            xy = xy[np.isfinite(xy)]
            if xy.size:
                lo = float(np.min(xy))
                hi = float(np.max(xy))
                if abs(hi - lo) < 1e-9:
                    lo -= 1.0
                    hi += 1.0
                pad = 0.04 * max(abs(hi - lo), 1.0)
                ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], color="#d7e2ee", linewidth=1.1, linestyle=":", alpha=0.75, label="ideal 1:1")

        mode_summary = (offset_metrics.get("by_mode") or {}).get(mode) or {}
        if mode_summary.get("rmse_measured_minus_commanded_mm") is not None:
            ax.text(
                0.015,
                0.965,
                f"RMSE measured-commanded = {float(mode_summary['rmse_measured_minus_commanded_mm']):.4f} mm",
                transform=ax.transAxes,
                ha="left",
                va="top",
                color="#f4f7fb",
                bbox={
                    "boxstyle": "round,pad=0.35",
                    "facecolor": (0.0, 0.0, 0.0, 0.0),
                    "edgecolor": (1, 1, 1, 0.16),
                },
            )

        if any_plotted:
            leg = ax.legend(
                facecolor=(0.0, 0.0, 0.0, 0.0),
                edgecolor=(1, 1, 1, 0.18),
                fontsize=8.2,
                labelcolor="#d7e2ee",
            )
            if leg is not None:
                leg.get_frame().set_facecolor((0.0, 0.0, 0.0, 0.0))
            fig.tight_layout()
            out = processed_dir / spec["out_name"]
            fig.savefig(out, facecolor=(0.0, 0.0, 0.0, 0.0), transparent=True, bbox_inches="tight")
            paths.append(out)
        plt.close(fig)

    return paths

def save_hysteresis_output_angle_plot(output_path: Path, records: List[Dict[str, Any]], title_prefix: str = ""):
    valid = [
        r for r in records
        if r.get("valid", False)
        and r.get("u_mm") is not None
        and r.get("z_mm") is not None
        and r.get("tip_angle_deg_from_name") is not None
    ]
    if len(valid) < 2:
        print("[WARN] Not enough valid records to build hysteresis output-angle plot.")
        return

    def _fit_circle_xz(u_vals: np.ndarray, z_vals: np.ndarray) -> Optional[Tuple[float, float, float]]:
        uu = np.asarray(u_vals, dtype=float).reshape(-1)
        zz = np.asarray(z_vals, dtype=float).reshape(-1)
        if uu.size < 3 or zz.size != uu.size:
            return None
        A = np.column_stack([uu, zz, np.ones_like(uu)])
        b = -(uu ** 2 + zz ** 2)
        try:
            sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        except Exception:
            return None
        a_coef, b_coef, c_coef = [float(v) for v in sol]
        uc = -0.5 * a_coef
        zc = -0.5 * b_coef
        rad_sq = uc * uc + zc * zc - c_coef
        if not np.isfinite(rad_sq) or rad_sq <= 0.0:
            return None
        return uc, zc, float(np.sqrt(rad_sq))

    def _compute_output_arc_angle_deg(u_vals: np.ndarray, z_vals: np.ndarray, x_in_deg: np.ndarray) -> np.ndarray:
        fit = _fit_circle_xz(u_vals, z_vals)
        if fit is None:
            return np.full_like(x_in_deg, np.nan, dtype=float)
        uc, zc, _ = fit
        raw_rad = np.unwrap(np.arctan2(u_vals - uc, z_vals - zc))
        raw_deg = np.degrees(raw_rad)
        finite = np.isfinite(raw_deg) & np.isfinite(x_in_deg)
        if int(np.count_nonzero(finite)) < 2:
            return np.full_like(x_in_deg, np.nan, dtype=float)
        raw_fit = raw_deg[finite]
        x_fit = x_in_deg[finite]
        raw_min = float(np.min(raw_fit))
        raw_max = float(np.max(raw_fit))
        x_min = float(np.min(x_fit))
        x_max = float(np.max(x_fit))
        raw_span = raw_max - raw_min
        x_span = x_max - x_min
        if (not np.isfinite(raw_span)) or raw_span < 1e-9 or (not np.isfinite(x_span)) or x_span < 1e-9:
            return np.full_like(x_in_deg, np.nan, dtype=float)
        M = np.column_stack([raw_fit, np.ones_like(raw_fit)])
        try:
            coeffs, *_ = np.linalg.lstsq(M, x_fit, rcond=None)
        except Exception:
            return np.full_like(x_in_deg, np.nan, dtype=float)
        slope = float(coeffs[0])
        if slope >= 0.0:
            out_deg = x_min + (raw_deg - raw_min) * (x_span / raw_span)
        else:
            out_deg = x_max - (raw_deg - raw_min) * (x_span / raw_span)
        return np.asarray(out_deg, dtype=float)

    # Measure output angle from the XZ arc geometry itself. Using the displacement from the
    # first sample gives a chord angle, not the tip's actual absolute orientation angle.
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for r in valid:
        phase = str(r.get("phase_name") or r.get("motion_phase") or "unknown")
        # Filenames produced by this unified script store motion_phase separately in the
        # phase field only if capture_and_save includes it. If unavailable, infer from
        # the block phase trend later by leaving it as one group.
        groups.setdefault(str(r.get("block_name") or "block"), []).append(r)

    fig, ax = plt.subplots(figsize=(9.5, 6.0), dpi=150)
    fig.patch.set_alpha(0.0)
    _apply_dark_axes_style(
        ax,
        f"{title_prefix}Hysteresis: output angle vs input attack angle",
        "Input attack angle command (deg)",
        "Output angle (deg)",
    )
    ax.set_facecolor((0.0, 0.0, 0.0, 0.0))

    any_plotted = False
    for block_name, rows in groups.items():
        rows = sorted(rows, key=lambda rr: int(rr.get("sample_index", 0)))
        x_in = np.array([float(rr["tip_angle_deg_from_name"]) for rr in rows], dtype=float)
        u_vals = np.array([float(rr["u_mm"]) for rr in rows], dtype=float)
        z_vals = np.array([float(rr["z_mm"]) for rr in rows], dtype=float)
        out_ang = _compute_output_arc_angle_deg(u_vals, z_vals, x_in)
        ax.plot(
            x_in,
            out_ang,
            marker="o",
            markersize=3.2,
            linewidth=1.5,
            color="#57c7ff",
            markerfacecolor="#57c7ff",
            markeredgewidth=0.0,
            label=str(block_name),
        )
        any_plotted = True

    if any_plotted:
        leg = ax.legend(facecolor=(0.0, 0.0, 0.0, 0.0), edgecolor=(1, 1, 1, 0.18), labelcolor="#d7e2ee")
        if leg is not None:
            leg.get_frame().set_facecolor((0.0, 0.0, 0.0, 0.0))
        fig.tight_layout()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, facecolor=(0.0, 0.0, 0.0, 0.0), transparent=True, bbox_inches="tight")
    plt.close(fig)

def add_unified_cli_args(ap: argparse.ArgumentParser):
    # Run mode
    ap.add_argument("--no-acquire", action="store_true",
                    help="Skip robot/camera acquisition. Use with --run-processing and --project-dir/--raw-dir for processing-only.")
    ap.add_argument("--run-processing", action="store_true",
                    help="Run checkerboard processing after acquisition, or process an existing project when --no-acquire is set.")

    # Acquisition / folders
    ap.add_argument("--parent-directory", default=os.getcwd(), help="Parent folder for acquisition output.")
    ap.add_argument("--project-name", default=DEFAULT_PROJECT_NAME, help="Run folder name.")
    ap.add_argument("--allow-existing", action="store_true", default=DEFAULT_ALLOW_EXISTING)
    ap.add_argument("--add-date", action="store_true", default=DEFAULT_ADD_DATE)

    # Connectivity
    ap.add_argument("--duet-web-address", default=DEFAULT_DUET_WEB_ADDRESS)
    ap.add_argument("--cam-port", type=int, default=DEFAULT_CAMERA_PORT)

    # Camera
    ap.add_argument("--show-preview", action="store_true")
    ap.add_argument("--enable-manual-focus", action="store_true", default=DEFAULT_MANUAL_FOCUS)
    ap.add_argument("--manual-focus-val", type=float, default=DEFAULT_MANUAL_FOCUS_VAL)
    ap.add_argument("--camera-width", type=int, default=DEFAULT_CAMERA_WIDTH)
    ap.add_argument("--camera-height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    ap.add_argument("--camera-flush-frames", type=int, default=DEFAULT_CAMERA_FLUSH_FRAMES)

    # Calibration / angle fit
    ap.add_argument("--trajectory-mode", type=str, default="angle_command",
                    choices=[
                        "angle_command", "one_daq_all_metrics", "fixed_xyz_all_metrics", "all_metrics",
                        "horizontal_line", "vertical_line",
                        "horizontal_offset", "radial_offset", "r_offset",
                        "z_offset", "vertical_offset",
                    ],
                    help=(
                        "Acquisition trajectory. angle_command keeps XYZ fixed by default. "
                        "one_daq_all_metrics/fixed_xyz_all_metrics are aliases of angle_command with filenames tagged for combined metrics. "
                        "horizontal_line/vertical_line trace compensated lines. "
                        "horizontal_offset/radial_offset commands calibrated radial offset r(B); "
                        "z_offset/vertical_offset commands calibrated z(B)."
                    ))
    ap.add_argument("--calibration", required=False, help="Path to robot calibration JSON. Required unless --no-acquire.")
    ap.add_argument("--y-offset-fit", type=str, default=DEFAULT_Y_OFFSET_FIT,
                    choices=["avg_pchip", "avg_cubic", "pchip", "cubic", "legacy"])
    ap.add_argument("--angle-fit", type=str, default="avg_pchip",
                    help="B-pull/input-angle fit. Supports abstract modes plus discovered labels like global_avg_pchip or per_phase_cubic.")
    ap.add_argument("--offset-fit", type=str, default=None,
                    help="Optional model selection for horizontal_offset/z_offset modes. Defaults to --angle-fit. Supports discovered labels like global_avg_pchip or per_phase_cubic.")
    ap.add_argument("--fit-branch", type=str, default="auto",
                    help="Calibration branch for acquisition/base processing. Use auto, 0_90, 0_180, global/none. auto chooses 0_90 when attack-max <= 90, otherwise 0_180.")
    ap.add_argument("--attack-min-deg", type=float, default=0.0,
                    help="Requested minimum input B attack angle. Default 0 deg.")
    ap.add_argument("--attack-max-deg", type=float, default=180.0,
                    help="Requested maximum input B attack angle. Default 180 deg.")
    ap.add_argument("--use-calibrated-attack-range", action="store_true",
                    help="Start from the calibrated angle range for the selected angle fit, then apply the DAQ recording window unless disabled.")
    ap.add_argument("--daq-attack-window-min-deg", type=float, default=0.0,
                    help="Minimum B attack angle allowed during acquisition/capture. Default 0 deg.")
    ap.add_argument("--daq-attack-window-max-deg", type=float, default=180.0,
                    help="Maximum B attack angle allowed during acquisition/capture. Default 180 deg.")
    ap.add_argument("--no-enforce-daq-attack-window", action="store_true",
                    help="Allow acquisition outside the prescribed DAQ attack-angle window.")
    ap.add_argument("--offset-min-mm", type=float, default=None,
                    help="Requested minimum calibrated offset for horizontal_offset/z_offset modes. Default: calibrated common range minimum.")
    ap.add_argument("--offset-max-mm", type=float, default=None,
                    help="Requested maximum calibrated offset for horizontal_offset/z_offset modes. Default: calibrated common range maximum.")
    ap.add_argument("--horizontal-offset-min-mm", type=float, default=None,
                    help="Optional minimum radial/horizontal offset override for horizontal_offset/radial_offset modes.")
    ap.add_argument("--horizontal-offset-max-mm", type=float, default=None,
                    help="Optional maximum radial/horizontal offset override for horizontal_offset/radial_offset modes.")
    ap.add_argument("--z-offset-min-mm", type=float, default=None,
                    help="Optional minimum Z offset override for z_offset/vertical_offset modes.")
    ap.add_argument("--z-offset-max-mm", type=float, default=None,
                    help="Optional maximum Z offset override for z_offset/vertical_offset modes.")
    ap.add_argument("--attack-c-deg", type=float, default=0.0,
                    help="Fixed C orientation used while curling/uncurling once.")
    ap.add_argument("--samples-per-leg", type=int, default=101,
                    help="Move samples for curl and for uncurl. Total points are 2*N-1.")

    # Optional XZ/XYZ tip subgoal
    ap.add_argument("--point-x", type=float, default=DEFAULT_POINT_X,
                    help="Base stage X, and optional XZ/XYZ tip-subgoal X.")
    ap.add_argument("--point-y", type=float, default=DEFAULT_POINT_Y,
                    help="Base stage Y, and optional XYZ tip-subgoal Y.")
    ap.add_argument("--point-z", type=float, default=DEFAULT_POINT_Z,
                    help="Base stage Z, and optional XZ/XYZ tip-subgoal Z.")
    ap.add_argument("--track-xz-subgoal", action="store_true",
                    help="Secondary subgoal: compensate XYZ enough to keep the tip's XZ projection fixed.")
    ap.add_argument("--track-xyz-subgoal", action="store_true",
                    help="Compensate XYZ to keep the full calibrated tip point fixed.")
    ap.add_argument("--no-y-plane-compensation", action="store_true",
                    help="For horizontal_line/vertical_line modes, do not move Y to cancel the calibrated off-plane Y offset.")
    ap.add_argument("--flip-rz-sign", action="store_true", default=DEFAULT_FLIP_RZ_SIGN)

    # Capture and motion discretization
    ap.add_argument("--orientation-capture-steps", type=int, default=DEFAULT_ORIENTATION_CAPTURE_STEPS)
    ap.add_argument("--custom-inverse-samples", type=int, default=DEFAULT_CUSTOM_INV_SAMPLES)
    ap.add_argument("--capture-every-move-point", action="store_true", default=True,
                    help="Capture every angle-command point. Use --no-capture-every-move-point to use capture steps only.")
    ap.add_argument("--no-capture-every-move-point", dest="capture_every_move_point", action="store_false")

    # Legacy compatibility args retained; not used by the new angle-command generator.
    ap.add_argument("--orientation-move-steps", type=int, default=DEFAULT_ORIENTATION_MOVE_STEPS)
    ap.add_argument("--oscillations-per-orientation", type=float, default=DEFAULT_OSCILLATIONS_PER_ORIENTATION)
    ap.add_argument("--sweep-tip-min-deg", type=float, default=DEFAULT_SWEEP_TIP_MIN_DEG)
    ap.add_argument("--sweep-tip-max-deg", type=float, default=DEFAULT_SWEEP_TIP_MAX_DEG)
    ap.add_argument("--b-0-to-90-only", action="store_true", default=False)
    ap.add_argument("--b-phase-offset-deg", type=float, default=DEFAULT_B_PHASE_OFFSET_DEG)

    # Feedrates and waits
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED)
    ap.add_argument("--c-flip-feed", type=float, default=DEFAULT_C_FLIP_FEED)
    ap.add_argument("--fine-approach-feed", type=float, default=DEFAULT_FINE_APPROACH_FEED)
    ap.add_argument("--probe-feed", type=float, default=DEFAULT_PROBE_FEED)
    ap.add_argument("--b-max-feed", type=float, default=DEFAULT_B_MAX_FEED)
    ap.add_argument("--b-accel-time", type=float, default=DEFAULT_B_ACCEL_TIME_S)
    ap.add_argument("--b-decel-time", type=float, default=DEFAULT_B_DECEL_TIME_S)
    ap.add_argument("--disable-segment-feed-scheduler", action="store_true")
    ap.add_argument("--dwell-before-ms", type=int, default=DEFAULT_DWELL_BEFORE_MS)
    ap.add_argument("--dwell-after-ms", type=int, default=DEFAULT_DWELL_AFTER_MS)
    ap.add_argument("--initial-sweep-wait-s", type=float, default=DEFAULT_INITIAL_SWEEP_WAIT_S)
    ap.add_argument("--tracked-move-settle-s", type=float, default=DEFAULT_TRACKED_MOVE_SETTLE_S)
    ap.add_argument("--travel-move-settle-s", type=float, default=DEFAULT_TRAVEL_MOVE_SETTLE_S)
    ap.add_argument("--b-extra-settle-s", type=float, default=DEFAULT_B_EXTRA_SETTLE_S)
    ap.add_argument("--inter-command-delay-s", type=float, default=DEFAULT_INTER_COMMAND_DELAY_S)
    ap.add_argument("--capture-at-start", action="store_true", default=DEFAULT_CAPTURE_AT_START)
    ap.add_argument("--settled-capture-mode", action="store_true", default=DEFAULT_SETTLED_CAPTURE_MODE)
    ap.add_argument("--settled-capture-buffer-s", type=float, default=DEFAULT_SETTLED_CAPTURE_BUFFER_S)

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

    # Processing args. Only used with --run-processing.
    ap.add_argument("--project_dir", "--project-dir", dest="project_dir", type=str, default=None,
                    help="Existing project folder containing raw_image_data_folder/. Defaults to new acquisition run folder.")
    ap.add_argument("--raw_dir", "--raw-dir", dest="raw_dir", type=str, default=None)
    ap.add_argument("--threshold", type=int, default=200)
    ap.add_argument("--save_plots", "--save-plots", dest="save_plots", action="store_true", default=True)
    ap.add_argument("--camera_calibration_file", "--camera-calibration-file", dest="camera_calibration_file", type=str, default=None)
    ap.add_argument("--checkerboard_reference_image", "--checkerboard-reference-image", dest="checkerboard_reference_image", type=str, default=None)
    ap.add_argument("--checkerboard_inner_corners", "--checkerboard-inner-corners", dest="checkerboard_inner_corners",
                    type=_parse_inner_corners_arg, default=None)
    ap.add_argument("--checkerboard_square_size_mm", "--checkerboard-square-size-mm", dest="checkerboard_square_size_mm", type=float, default=None)
    ap.add_argument("--checkerboard_no_undistort", "--checkerboard-no-undistort", dest="checkerboard_no_undistort", action="store_true")
    ap.add_argument("--checkerboard_mm_scale_correction", "--checkerboard-mm-scale-correction", dest="checkerboard_mm_scale_correction", type=float, default=0.5)
    ap.add_argument("--checkerboard_no_flip_planar_x", "--checkerboard-no-flip-planar-x", dest="checkerboard_no_flip_planar_x", action="store_true")
    ap.add_argument("--link_mode", "--link-mode", dest="link_mode", type=str, default="symlink", choices=["symlink", "copy"])
    ap.add_argument("--save_analysis_config", "--save-analysis-config", dest="save_analysis_config", action="store_true")
    ap.add_argument("--tip_detection_mode", "--tip-detection-mode", dest="tip_detection_mode", type=str, default="classical",
                    choices=["classical", "red_dot", "auto_red_dot"])
    ap.add_argument("--tip_refine_mode", "--tip-refine-mode", dest="tip_refine_mode", type=str, default="parallel_centerline",
                    choices=["none", "edge_dt", "edge_grad", "mainray", "parallel_centerline"])
    ap.add_argument("--tip_refine_dt_step_px", dest="tip_refine_dt_step_px", type=float, default=1.0)
    ap.add_argument("--tip_refine_max_step_px", dest="tip_refine_max_step_px", type=int, default=80)
    ap.add_argument("--tip_refine_grad_step_px", dest="tip_refine_grad_step_px", type=float, default=0.25)
    ap.add_argument("--tip_refine_grad_search_len_px", dest="tip_refine_grad_search_len_px", type=int, default=60)
    ap.add_argument("--tip_refine_mainray_fit_back_near_r", dest="tip_refine_mainray_fit_back_near_r", type=float, default=1.5)
    ap.add_argument("--tip_refine_mainray_fit_back_far_r", dest="tip_refine_mainray_fit_back_far_r", type=float, default=6.0)
    ap.add_argument("--tip_refine_mainray_anchor_back_r", dest="tip_refine_mainray_anchor_back_r", type=float, default=1.0)
    ap.add_argument("--tip_refine_mainray_ray_step_px", dest="tip_refine_mainray_ray_step_px", type=float, default=0.5)
    ap.add_argument("--tip_refine_mainray_ray_max_len_r", dest="tip_refine_mainray_ray_max_len_r", type=float, default=8.0)
    ap.add_argument("--tip_refine_parallel_section_near_r", dest="tip_refine_parallel_section_near_r", type=float, default=1.0)
    ap.add_argument("--tip_refine_parallel_section_far_r", dest="tip_refine_parallel_section_far_r", type=float, default=6.0)
    ap.add_argument("--tip_refine_parallel_scan_half_r", dest="tip_refine_parallel_scan_half_r", type=float, default=3.0)
    ap.add_argument("--tip_refine_parallel_num_sections", dest="tip_refine_parallel_num_sections", type=int, default=9)
    ap.add_argument("--tip_refine_parallel_cross_step_px", dest="tip_refine_parallel_cross_step_px", type=float, default=0.5)
    ap.add_argument("--tip_refine_parallel_ray_step_px", dest="tip_refine_parallel_ray_step_px", type=float, default=0.5)
    ap.add_argument("--tip_refine_parallel_ray_max_len_r", dest="tip_refine_parallel_ray_max_len_r", type=float, default=8.0)
    ap.add_argument("--tip_refiner_model", "--tip-refiner-model", dest="tip_refiner_model", type=str, default=None)
    ap.add_argument("--tip_refiner_anchor", "--tip-refiner-anchor", dest="tip_refiner_anchor", type=str, default=None,
                    choices=["coarse", "selected", "refined"])
    ap.add_argument("--tip_refiner_compare_only", "--tip-refiner-compare-only", dest="tip_refiner_compare_only", action="store_true")
    ap.add_argument("--tracked_tip_source", "--tracked-tip-source", dest="tracked_tip_source", type=str, default="auto",
                    choices=["auto", "coarse", "selected", "cnn"])
    ap.add_argument("--hist_bins", "--hist-bins", dest="hist_bins", type=int, default=24)
    ap.add_argument("--process_all_fits", "--process-all-fits", dest="process_all_fits", action="store_true",
                    help="During processing, reuse the same DAQ and generate one-DAQ metrics/plots for all requested angle/offset fit modes.")
    ap.add_argument("--plot_all_calibration_models", "--plot-all-calibration-models", "--process-all-calibration-models", dest="plot_all_calibration_models", action="store_true",
                    help="During processing, discover every usable tip_angle/r/z model family in the calibration JSON and plot all of them as panel columns.")
    ap.add_argument("--processing_angle_fits", "--processing-angle-fits", dest="processing_angle_fits", nargs="*", default=None,
                    help="Angle fit modes to process. Example: --processing-angle-fits avg_pchip avg_cubic linear pull release phase_specific per_phase_pchip per_phase_cubic")
    ap.add_argument("--processing_offset_fits", "--processing-offset-fits", dest="processing_offset_fits", nargs="*", default=None,
                    help="Offset fit modes to process. Example: --processing-offset-fits avg_pchip avg_cubic linear pull release phase_specific per_phase_pchip per_phase_cubic")
    ap.add_argument("--processing_fit_branches", "--processing-fit-branches", dest="processing_fit_branches", nargs="*", default=None,
                    help="Calibration branch labels to process for fit-sweep panels, e.g. 0_90_0 0_180_0 90_180_90 or all. Each curl sequence gets its own output panel.")
    ap.add_argument("--fit_combination_mode", "--fit-combination-mode", dest="fit_combination_mode", type=str, default="paired", choices=["paired", "cross"],
                    help="paired: angle/offset modes are paired by index/name. cross: every angle fit with every offset fit.")
    ap.add_argument("--fit_output_root", "--fit-output-root", dest="fit_output_root", type=str, default="fit_sweeps",
                    help="Subfolder under processed_image_data_folder where per-fit outputs are saved.")


# -----------------------------------------------------------------------------
# Curl-angle-specific calibration branch support
# -----------------------------------------------------------------------------
# The main gcode calibration JSON can contain curl_angle_specific_fit_models with
# sequences such as 0-90-0, 0-180-0, and 90-180-90. These overrides make
# --processing-fit-branches all discover those sequences and make global_*/
# per_phase_* model labels select models from the requested sequence before
# falling back to generic fit_models / fit_models_by_phase.

def _angle_sequence_branch_label(seq: Any) -> Optional[str]:
    if not isinstance(seq, (list, tuple)) or len(seq) < 2:
        return None
    try:
        vals = [int(round(float(v))) for v in seq]
    except Exception:
        return None
    if vals[:3] == [0, 90, 0]:
        return "0_90_0"
    if vals[:3] == [0, 180, 0]:
        return "0_180_0"
    if vals[:3] == [90, 180, 90]:
        return "90_180_90"
    return "_".join(str(v) for v in vals)


def normalize_fit_branch(value: Any) -> Optional[str]:
    """Normalize calibration branch labels such as 0-90-0, 0_180, 90-180-90."""
    if value is None:
        return None
    txt = str(value).strip().lower()
    if txt in ("", "auto", "default", "none", "global", "legacy", "all"):
        return None
    key = _alnum_key(txt)
    # Full curl sequences first; otherwise 90-180-90 would be swallowed by 0-180.
    if key in ("0900", "0to90to0", "zerotoninetytozero", "curl0900", "sequence0900", "range0900"):
        return "0_90_0"
    if key in ("01800", "0to180to0", "zerotooneeightytozero", "curl01800", "sequence01800", "range01800"):
        return "0_180_0"
    if key in ("9018090", "90to180to90", "ninetytooneeightytotoninety", "curl9018090", "sequence9018090", "range9018090"):
        return "90_180_90"
    # Backward-compatible aliases from earlier versions of this script.
    if key in ("090", "0to90", "zeroto90", "zerotoninety", "range090", "branch090", "deg090"):
        return "0_90_0"
    if key in ("0180", "0to180", "zeroto180", "zerotooneeighty", "range0180", "branch0180", "deg0180"):
        return "0_180_0"
    # Conservative substring handling.
    if "9018090" in key:
        return "90_180_90"
    if "01800" in key or "0to180to0" in key:
        return "0_180_0"
    if "0900" in key or "0to90to0" in key:
        return "0_90_0"
    return txt.replace("-", "_").replace(" ", "_")


def _fit_branch_title(branch: Any) -> str:
    b = normalize_fit_branch(branch)
    if b == "0_90_0":
        return "0–90–0° curl sequence"
    if b == "0_180_0":
        return "0–180–0° curl sequence"
    if b == "90_180_90":
        return "90–180–90° curl sequence"
    if b == "0_90":
        return "0–90°"
    if b == "0_180":
        return "0–180°"
    return "global/default"


def _branch_token_set(branch: Any) -> Set[str]:
    b = normalize_fit_branch(branch)
    if b is None:
        return set()
    if b == "0_90_0":
        return {"0900", "0to90to0", "curl0900", "sequence0900", "angle0900", "range0900"}
    if b == "0_180_0":
        return {"01800", "0to180to0", "curl01800", "sequence01800", "angle01800", "range01800"}
    if b == "90_180_90":
        return {"9018090", "90to180to90", "curl9018090", "sequence9018090", "angle9018090", "range9018090"}
    return {_alnum_key(b)}


def _branch_matches_key(key: Any, branch: Any) -> bool:
    b = normalize_fit_branch(branch)
    if b is None:
        return True
    k = _alnum_key(key)
    if not k:
        return False
    return any(t and t in k for t in _branch_token_set(b))


def _is_model_like_dict(d: Any) -> bool:
    return isinstance(d, dict) and any(
        isinstance(d.get(k), dict)
        for k in (
            "r", "z", "tip_angle", "offplane_y", "y_offset_pchip",
            "r_linear", "r_pchip", "r_cubic", "r_avg_linear", "r_avg_pchip", "r_avg_cubic",
            "z_linear", "z_pchip", "z_cubic", "z_avg_linear", "z_avg_pchip", "z_avg_cubic",
            "tip_angle_linear", "tip_angle_pchip", "tip_angle_cubic",
            "tip_angle_avg_linear", "tip_angle_avg_pchip", "tip_angle_avg_cubic",
        )
    )


def _as_models_dict(candidate: Any) -> Optional[dict]:
    if not isinstance(candidate, dict):
        return None
    for nested_key in ("fit_models", "models", "fit_descriptors", "model_descriptors"):
        nested = candidate.get(nested_key)
        if _is_model_like_dict(nested):
            return nested
    if _is_model_like_dict(candidate):
        return candidate
    return None


def _branch_label_from_container_key_and_value(key: Any, value: Any) -> Optional[str]:
    # Prefer explicit angle_sequence_deg metadata when present.
    if isinstance(value, dict):
        for seq_key in ("angle_sequence_deg", "curl_angle_sequence_deg", "sequence_deg", "angles_deg"):
            label = _angle_sequence_branch_label(value.get(seq_key))
            if label:
                return label
        # Some exports store metadata one level lower.
        for nested_key in ("metadata", "sequence_metadata", "curl_sequence_metadata"):
            nested = value.get(nested_key)
            if isinstance(nested, dict):
                for seq_key in ("angle_sequence_deg", "curl_angle_sequence_deg", "sequence_deg", "angles_deg"):
                    label = _angle_sequence_branch_label(nested.get(seq_key))
                    if label:
                        return label
    return normalize_fit_branch(key)


def _curl_sequence_container_variants(label: str, container: dict) -> List[Tuple[str, dict]]:
    """Return container variants so direct pull/release maps and averaged maps are visible."""
    out: List[Tuple[str, dict]] = [(label, container)]
    if not isinstance(container, dict):
        return out

    # Direct phase entries: {"pull": {models...}, "release": {models...}}
    phase_map: Dict[str, dict] = {}
    for pk, pv in container.items():
        phase_name = _normalize_motion_phase_name(pk)
        if phase_name is None:
            continue
        if phase_name.startswith("pull") or phase_name.startswith("release"):
            models = _as_models_dict(pv)
            if models is not None:
                phase_map[str(pk)] = models
    if phase_map:
        wrapper: Dict[str, Any] = {"fit_models_by_phase": phase_map}
        for avg_key in (
            "fit_models", "averaged_fits", "averaged_fit_models", "average_fit_models",
            "avg_fit_models", "shared_aux_fit_models", "global_fit_models",
        ):
            avg = _as_models_dict(container.get(avg_key))
            if avg is not None:
                wrapper["fit_models"] = avg
                break
        out.append((f"{label}/direct_phase_entries", wrapper))

    # Some exports store sequences under a metadata wrapper with a nested model container.
    for nested_key in (
        "fit_models", "averaged_fits", "averaged_fit_models", "average_fit_models",
        "avg_fit_models", "shared_aux_fit_models", "global_fit_models",
    ):
        models = _as_models_dict(container.get(nested_key))
        if models is not None:
            out.append((f"{label}/{nested_key}", {"fit_models": models}))

    return out


def _raw_branch_containers(cal: Calibration, branch: Any) -> List[Tuple[str, dict]]:
    raw = getattr(cal, "raw_calibration_json", {}) or {}
    out: List[Tuple[str, dict]] = []
    branch_norm = normalize_fit_branch(branch)

    # Main calibration JSON: curl-angle-specific sequences.
    curl_top = raw.get("curl_angle_specific_fit_models") if isinstance(raw, dict) else None
    if isinstance(curl_top, dict):
        for seq_key, seq_container in curl_top.items():
            if not isinstance(seq_container, dict):
                continue
            seq_label = _branch_label_from_container_key_and_value(seq_key, seq_container)
            if branch_norm is None or normalize_fit_branch(seq_label) == branch_norm or _branch_matches_key(seq_key, branch_norm):
                for lab, cont in _curl_sequence_container_variants(f"curl_angle_specific_fit_models/{seq_key}", seq_container):
                    out.append((lab, cont))

    # Other possible branch containers retained from earlier script versions.
    possible_top_keys = [
        "fit_models_by_branch", "fit_models_by_angle_branch", "fit_models_by_angle_range",
        "fit_models_by_range", "fit_models_by_sweep_range", "fit_models_by_attack_range",
        "fit_models_by_tip_angle_range", "fit_models_by_window", "branch_fit_models",
        "range_fit_models", "angle_range_fit_models", "exported_models_by_branch",
    ]
    for top_key in possible_top_keys:
        top = raw.get(top_key) if isinstance(raw, dict) else None
        if isinstance(top, dict):
            for bk, bv in top.items():
                if isinstance(bv, dict) and (branch_norm is None or _branch_matches_key(bk, branch_norm)):
                    out.append((f"{top_key}/{bk}", bv))

    exported = raw.get("exported_models") if isinstance(raw, dict) else None
    if isinstance(exported, dict):
        for bk, bv in exported.items():
            if isinstance(bv, dict) and (branch_norm is None or _branch_matches_key(bk, branch_norm)):
                out.append((f"exported_models/{bk}", bv))

    # Deduplicate labels while preserving order.
    dedup: List[Tuple[str, dict]] = []
    seen: Set[str] = set()
    for label, cont in out:
        if label not in seen:
            dedup.append((label, cont)); seen.add(label)
    return dedup


def detect_fit_branches_in_calibration_raw(raw: dict) -> List[str]:
    found: List[str] = []
    def add(x):
        b = normalize_fit_branch(x)
        if b is not None and b not in found:
            found.append(b)

    # Main gcode JSON curl-specific section.
    curl_top = raw.get("curl_angle_specific_fit_models") if isinstance(raw, dict) else None
    if isinstance(curl_top, dict):
        for seq_key, seq_container in curl_top.items():
            add(_branch_label_from_container_key_and_value(seq_key, seq_container))

    # Generic branch containers.
    branch_map_keys = [
        "fit_models_by_branch", "fit_models_by_angle_branch", "fit_models_by_angle_range",
        "fit_models_by_range", "fit_models_by_sweep_range", "fit_models_by_attack_range",
        "fit_models_by_tip_angle_range", "fit_models_by_window", "branch_fit_models",
        "range_fit_models", "angle_range_fit_models", "exported_models_by_branch",
    ]
    for top_key in branch_map_keys:
        top = raw.get(top_key) if isinstance(raw, dict) else None
        if isinstance(top, dict):
            for bk, bv in top.items():
                add(_branch_label_from_container_key_and_value(bk, bv))

    # Phase keys can also encode branch labels.
    fbyp = raw.get("fit_models_by_phase", {}) if isinstance(raw, dict) else {}
    if isinstance(fbyp, dict):
        for pk in fbyp.keys():
            b = normalize_fit_branch(pk)
            if b in ("0_90_0", "0_180_0", "90_180_90"):
                add(b)
    return found


def _normalize_processing_fit_branches(value: Any, fallback: Optional[List[Optional[str]]] = None) -> List[Optional[str]]:
    fallback_clean: List[Optional[str]] = []
    for fb in (fallback or []):
        nb = normalize_fit_branch(fb)
        if nb not in fallback_clean:
            fallback_clean.append(nb)
    if value is None:
        return fallback_clean
    if isinstance(value, str):
        raw_parts = [p.strip() for p in value.replace(";", ",").split(",") if p.strip()]
    else:
        raw_parts = []
        for item in value:
            if item is None:
                continue
            raw_parts.extend([p.strip() for p in str(item).replace(";", ",").split(",") if p.strip()])
    out: List[Optional[str]] = []
    for part in raw_parts:
        if str(part).strip().lower() == "all":
            source = fallback_clean or ["0_90_0", "0_180_0", "90_180_90"]
            for b in source:
                nb = normalize_fit_branch(b)
                if nb not in out:
                    out.append(nb)
            continue
        b = normalize_fit_branch(part)
        if b not in out:
            out.append(b)
    return out or fallback_clean


def _model_keys_for(base: str, kind: str, branch: Any = None) -> List[str]:
    base = str(base)
    kind = str(kind or "").strip().lower()
    branch_labels: List[str] = []
    b = normalize_fit_branch(branch)
    if b == "0_90_0":
        branch_labels = ["0_90_0", "0-90-0", "0to90to0", "0900"]
    elif b == "0_180_0":
        branch_labels = ["0_180_0", "0-180-0", "0to180to0", "01800"]
    elif b == "90_180_90":
        branch_labels = ["90_180_90", "90-180-90", "90to180to90", "9018090"]
    elif b == "0_90":
        branch_labels = ["0_90", "0-90", "0to90", "090"]
    elif b == "0_180":
        branch_labels = ["0_180", "0-180", "0to180", "0180"]

    suffixes: List[str]
    if kind in ("default", ""):
        suffixes = [""]
    elif kind == "pchip":
        suffixes = ["pchip", "", "avg_pchip", "pchip_avg"]
    elif kind == "avg_pchip":
        suffixes = ["avg_pchip", "pchip_avg", "pchip", ""]
    elif kind == "cubic":
        suffixes = ["cubic", "avg_cubic", "cubic_avg"]
    elif kind == "avg_cubic":
        suffixes = ["avg_cubic", "cubic_avg", "cubic"]
    elif kind == "linear":
        suffixes = ["linear", "avg_linear"]
    elif kind == "avg_linear":
        suffixes = ["avg_linear", "linear"]
    elif kind in ("any", "phase_any"):
        suffixes = ["", "linear", "pchip", "cubic", "avg_linear", "avg_pchip", "avg_cubic"]
    else:
        suffixes = [kind, ""]

    keys: List[str] = []
    for suff in suffixes:
        keys.append(base if suff == "" else f"{base}_{suff}")
    for bl in branch_labels:
        for suff in suffixes:
            keys.extend([
                f"{base}_{bl}" if suff == "" else f"{base}_{bl}_{suff}",
                f"{base}_{suff}_{bl}" if suff else f"{base}_{bl}",
                f"{bl}_{base}" if suff == "" else f"{bl}_{base}_{suff}",
            ])
    out: List[str] = []
    for k in keys:
        if k not in out:
            out.append(k)
    return out


def _global_model_containers_for_branch(cal: Calibration, branch: Any = None) -> List[Tuple[str, dict]]:
    branch_norm = normalize_fit_branch(branch if branch is not None else getattr(cal, "active_fit_branch", None))
    groups: List[Tuple[str, dict]] = []
    if branch_norm is not None:
        for label, container in _raw_branch_containers(cal, branch_norm):
            if not isinstance(container, dict):
                continue
            for key in (
                "fit_models", "averaged_fits", "averaged_fit_models", "average_fit_models",
                "avg_fit_models", "shared_aux_fit_models", "global_fit_models",
            ):
                models = _as_models_dict(container.get(key))
                if models is not None:
                    groups.append((f"{label}/{key}", models))
            if _is_model_like_dict(container):
                groups.append((label, container))
    global_models = getattr(cal, "raw_fit_models", {}) or {}
    if isinstance(global_models, dict):
        groups.append(("global_fit_models", global_models))
    shared = getattr(cal, "raw_shared_aux_fit_models", {}) or {}
    if isinstance(shared, dict):
        groups.append(("shared_aux_fit_models", shared))
    return groups


def _phase_model_containers_for_branch(cal: Calibration, branch: Any = None) -> List[Tuple[str, str, dict]]:
    branch_norm = normalize_fit_branch(branch if branch is not None else getattr(cal, "active_fit_branch", None))
    out: List[Tuple[str, str, dict]] = []
    if branch_norm is not None:
        for label, container in _raw_branch_containers(cal, branch_norm):
            if not isinstance(container, dict):
                continue
            # Standard nested phase maps.
            for phase_key_name in ("fit_models_by_phase", "phases", "phase_models", "models_by_phase"):
                phmap = container.get(phase_key_name)
                if isinstance(phmap, dict):
                    for pk, pv in phmap.items():
                        models = _as_models_dict(pv)
                        if models is not None:
                            out.append((f"{label}/{phase_key_name}/{pk}", str(pk), models))
            # Direct phase entries on the sequence container.
            for pk, pv in container.items():
                phase_name = _normalize_motion_phase_name(pk)
                if phase_name and (phase_name.startswith("pull") or phase_name.startswith("release")):
                    models = _as_models_dict(pv)
                    if models is not None:
                        out.append((f"{label}/{pk}", str(pk), models))

    raw_phase_models = getattr(cal, "raw_fit_models_by_phase", {}) or {}
    if isinstance(raw_phase_models, dict):
        for pk, pv in raw_phase_models.items():
            models = _as_models_dict(pv)
            if models is None:
                continue
            if branch_norm is not None and not _branch_matches_key(pk, branch_norm):
                hay = " ".join(
                    " ".join(str(v.get(x, "")) for x in ("value_name", "fit_branch", "angle_range", "range_label", "sweep_range"))
                    for v in models.values() if isinstance(v, dict)
                )
                # If explicit branch containers exist, don't use generic phase entries as a branch match.
                if _branch_matches_key(hay, branch_norm):
                    out.append((f"phase/{pk}", str(pk), models))
                elif not getattr(cal, "available_fit_branches", []):
                    out.append((f"phase/{pk}", str(pk), models))
                continue
            out.append((f"phase/{pk}", str(pk), models))
    return out


def _candidate_model_groups(cal: Calibration, phase: Optional[str] = None, branch: Any = None) -> List[Tuple[str, dict]]:
    groups: List[Tuple[str, dict]] = []
    branch_norm = normalize_fit_branch(branch if branch is not None else getattr(cal, "active_fit_branch", None))
    phase_norm = _normalize_motion_phase_name(phase)

    if branch_norm is not None:
        for label, phase_key, models in _phase_model_containers_for_branch(cal, branch_norm):
            if _phase_matches_key(phase_key, phase_norm):
                groups.append((label, models))
        for label, models in _global_model_containers_for_branch(cal, branch_norm):
            groups.append((label, models))

    # Generic fallback only after branch-specific containers.
    raw_phase_models = getattr(cal, "raw_fit_models_by_phase", {}) or {}
    if isinstance(raw_phase_models, dict):
        for pk, pv in raw_phase_models.items():
            models = _as_models_dict(pv)
            if models is not None and _phase_matches_key(pk, phase_norm):
                groups.append((f"phase_fallback/{pk}", models))
    for label, models in [
        ("global_fit_models", getattr(cal, "raw_fit_models", {}) or {}),
        ("shared_aux_fit_models", getattr(cal, "raw_shared_aux_fit_models", {}) or {}),
    ]:
        if isinstance(models, dict):
            groups.append((label, models))

    dedup: List[Tuple[str, dict]] = []
    seen: Set[int] = set()
    for label, models in groups:
        ident = id(models)
        if ident not in seen:
            dedup.append((label, models)); seen.add(ident)
    return dedup



# -----------------------------------------------------------------------------
# Strict curl-sequence panel filtering and discovery override
# -----------------------------------------------------------------------------
# The curl_angle_specific_fit_models section is sequence-specific. When plotting a
# 0-90-0, 0-180-0, or 90-180-90 branch panel, do not mix in full-run samples or
# generic global fallback columns. Filter records to the branch angle window and
# expose only the equations that are present inside that curl sequence.


def _curl_sequence_container_for_branch(cal: Optional[Calibration], branch: Any) -> Optional[dict]:
    if cal is None:
        return None
    branch_norm = normalize_fit_branch(branch)
    raw = getattr(cal, "raw_calibration_json", {}) or {}
    curl_top = raw.get("curl_angle_specific_fit_models") if isinstance(raw, dict) else None
    if not isinstance(curl_top, dict):
        return None
    for seq_key, seq_container in curl_top.items():
        if not isinstance(seq_container, dict):
            continue
        seq_label = _branch_label_from_container_key_and_value(seq_key, seq_container)
        if normalize_fit_branch(seq_label) == branch_norm:
            return seq_container
    return None


def _curl_sequence_angle_window(cal: Optional[Calibration], branch: Any) -> Optional[Tuple[float, float]]:
    cont = _curl_sequence_container_for_branch(cal, branch)
    if not isinstance(cont, dict):
        return None
    seq = None
    for key in ("angle_sequence_deg", "curl_angle_sequence_deg", "sequence_deg", "angles_deg"):
        if key in cont:
            seq = cont.get(key)
            break
    if not isinstance(seq, (list, tuple)) or len(seq) < 2:
        return None
    try:
        vals = [float(v) for v in seq if v is not None]
    except Exception:
        return None
    if len(vals) < 2:
        return None
    return float(min(vals)), float(max(vals))


def _record_command_angle_for_window_filter(rec: Dict[str, Any]) -> Optional[float]:
    for key in (
        "tip_angle_deg_from_name",
        "one_daq_angle_command_deg",
        "tip_angle_deg",
        "attack_angle_deg",
        "commanded_attack_angle_deg",
    ):
        value = _finite_float_or_none(rec.get(key))
        if value is not None:
            return value
    return None


def _filter_records_to_curl_sequence_window(
    records: List[Dict[str, Any]],
    robot_cal: Optional[Calibration],
    branch: Any,
    tolerance_deg: float = 1.0,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Return only samples whose commanded input angle belongs to the branch window."""
    branch_norm = normalize_fit_branch(branch)
    window = _curl_sequence_angle_window(robot_cal, branch_norm)
    if window is None:
        return [dict(r) for r in records], {
            "branch": _fit_branch_label(branch_norm),
            "filtered": False,
            "reason": "no curl_angle_specific_fit_models angle_sequence_deg window found",
            "input_count": int(len(records)),
            "output_count": int(len(records)),
        }
    lo, hi = window
    out: List[Dict[str, Any]] = []
    missing_angle = 0
    for rec in records:
        a = _record_command_angle_for_window_filter(rec)
        if a is None:
            missing_angle += 1
            continue
        if (lo - float(tolerance_deg)) <= float(a) <= (hi + float(tolerance_deg)):
            out.append(dict(rec))
    return out, {
        "branch": _fit_branch_label(branch_norm),
        "filtered": True,
        "angle_window_deg": [float(lo), float(hi)],
        "tolerance_deg": float(tolerance_deg),
        "input_count": int(len(records)),
        "output_count": int(len(out)),
        "dropped_count": int(len(records) - len(out)),
        "missing_angle_count": int(missing_angle),
    }


def _suffixes_available_in_model_dict(models: Any, base: str) -> Set[str]:
    if not isinstance(models, dict):
        return set()
    out: Set[str] = set()
    base_norm = _alnum_key(base)
    for key, value in models.items():
        if not isinstance(value, dict):
            continue
        k = str(key).strip().lower()
        kn = _alnum_key(k)
        if kn == base_norm:
            out.add("default")
        prefix = f"{base}_"
        if k.startswith(prefix):
            out.add(k[len(prefix):])
    return out


def _ordered_fit_suffixes(suffixes: Set[str]) -> List[str]:
    preferred = ["default", "linear", "pchip", "cubic", "avg_linear", "avg_pchip", "avg_cubic"]
    ordered = [s for s in preferred if s in suffixes]
    ordered.extend(sorted(s for s in suffixes if s not in set(preferred)))
    return ordered


def _discover_curl_specific_model_fit_modes(cal: Optional[Calibration], branch: Any) -> Optional[Dict[str, Any]]:
    branch_norm = normalize_fit_branch(branch)
    cont = _curl_sequence_container_for_branch(cal, branch_norm)
    if not isinstance(cont, dict):
        return None
    fbyp = cont.get("fit_models_by_phase")
    if not isinstance(fbyp, dict):
        return None

    phase_suffix_union: Set[str] = set()
    phases_seen: List[str] = []
    for phase_key, models in fbyp.items():
        if not isinstance(models, dict):
            continue
        phases_seen.append(str(phase_key))
        # A usable column must be able to produce tip_angle, radial r, and z.
        common = (
            _suffixes_available_in_model_dict(models, "tip_angle")
            & _suffixes_available_in_model_dict(models, "r")
            & _suffixes_available_in_model_dict(models, "z")
        )
        phase_suffix_union |= common

    suffixes = _ordered_fit_suffixes(phase_suffix_union)
    modes = [f"per_phase_{s}" for s in suffixes]
    return {
        "fit_modes": modes,
        "global_modes": [],
        "per_phase_modes": modes,
        "branch": _fit_branch_label(branch_norm),
        "branch_title": _fit_branch_title(branch_norm),
        "source": "curl_angle_specific_fit_models",
        "phases_seen": phases_seen,
        "angle_window_deg": _curl_sequence_angle_window(cal, branch_norm),
    }


_BASE_DISCOVER_CALIBRATION_MODEL_FIT_MODES_STRICT = discover_calibration_model_fit_modes


def discover_calibration_model_fit_modes(cal: Optional[Calibration], branch: Any = None) -> Dict[str, Any]:
    """
    For explicit curl branches, discover only model columns from the matching
    curl_angle_specific_fit_models/<sequence>/fit_models_by_phase section.
    This prevents 0-90/90-180 panels from silently reusing full-range global models.
    """
    curl_specific = _discover_curl_specific_model_fit_modes(cal, branch)
    if curl_specific is not None:
        return curl_specific
    return _BASE_DISCOVER_CALIBRATION_MODEL_FIT_MODES_STRICT(cal, branch=branch)


_BASE_SAVE_ONE_DAQ_FIT_SWEEP_OUTPUTS_UNFILTERED = save_one_daq_fit_sweep_outputs


def save_one_daq_fit_sweep_outputs(
    processed_dir: Path,
    records: List[Dict[str, Any]],
    robot_cal: Optional[Calibration],
    angle_fit_modes: List[str],
    offset_fit_modes: List[str],
    flip_rz_sign: bool = False,
    combination_mode: str = "paired",
    output_root_name: str = "fit_sweeps",
    title_prefix: str = "Checkerboard-referenced ",
) -> Dict[str, Any]:
    active_branch = normalize_fit_branch(getattr(robot_cal, "active_fit_branch", None)) if robot_cal is not None else None
    filtered_records, filter_info = _filter_records_to_curl_sequence_window(records, robot_cal, active_branch)
    if filter_info.get("filtered"):
        print(
            "[INFO] Curl-sequence sample filter "
            f"branch={filter_info.get('branch')} "
            f"window={filter_info.get('angle_window_deg')} deg: "
            f"kept {filter_info.get('output_count')}/{filter_info.get('input_count')} records"
        )
        if int(filter_info.get("output_count", 0)) == 0:
            print(
                "[WARN] No records remain after curl-sequence angle filtering. "
                "This usually means the selected project folder is not a DAQ for that sequence."
            )

    summary = _BASE_SAVE_ONE_DAQ_FIT_SWEEP_OUTPUTS_UNFILTERED(
        processed_dir=processed_dir,
        records=filtered_records,
        robot_cal=robot_cal,
        angle_fit_modes=angle_fit_modes,
        offset_fit_modes=offset_fit_modes,
        flip_rz_sign=flip_rz_sign,
        combination_mode=combination_mode,
        output_root_name=output_root_name,
        title_prefix=title_prefix,
    )
    summary["curl_sequence_filter"] = filter_info
    try:
        summary_path = Path(summary.get("summary_json", ""))
        if summary_path:
            with open(summary_path, "w") as f:
                json.dump(_json_ready(summary), f, indent=2)
    except Exception:
        pass
    return summary


def main():
    ap = argparse.ArgumentParser(
        description=(
            "One-file angle-command, line, or fixed-stage offset-command curl/uncurl "
            "acquisition plus optional checkerboard processing and hysteresis plotting."
        )
    )
    add_unified_cli_args(ap)
    args = ap.parse_args()

    if not bool(args.no_acquire):
        if not args.calibration:
            ap.error("--calibration is required unless --no-acquire is set.")
        run_acquisition(args)

    if bool(args.run_processing):
        if args.project_dir is None and args.raw_dir is None:
            ap.error("--run-processing needs --project-dir/--raw-dir when --no-acquire is used.")
        if not args.camera_calibration_file:
            ap.error("--run-processing requires --camera-calibration-file.")
        if not args.checkerboard_reference_image:
            ap.error("--run-processing requires --checkerboard-reference-image.")
        run_processing(args)
    else:
        print("[INFO] Processing was not run. Add --run-processing to analyze images and output hysteresis plots.")

if __name__ == "__main__":
    main()
