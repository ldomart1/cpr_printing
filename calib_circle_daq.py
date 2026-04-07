#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone circle-tracking acquisition script.

What it does
------------
- Loads the calibration JSON.
- Builds a circle in tip-space using only B-driven lateral motion plus commanded Z motion
  during the recorded segments.
- Starts at B=0, C=0 on the bottom of the circle.
- Quarter 1: curl using the pull PCHIP.
- Quarter 2: uncurl using the release PCHIP, translated at the pull->release transition so
  the tip starts the uncurl from the current tip location.
- Midpoint: stop recording, move to the exact top point while rotating C by 180 deg
  (default C feed 20000).
- Quarter 3: curl again with C=180 while descending.
- Quarter 4: uncurl again until the path returns near the starting point.
- X motion is suppressed during recorded quarter segments and is only allowed during
  transition phases (pull/release shifts, midpoint C flip, optional final recenter).
- Y tracking is still solved so the tip remains at the requested constant tip-space Y.

This preserves the robot/camera execution style of the uploaded star-tracking script while
replacing the star path planner with a circle planner.
"""

import argparse
import copy
import csv
import json
import math
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

try:
    from duetwebapi import DuetWebAPI
except Exception:
    DuetWebAPI = None  # type: ignore

try:
    from scipy.interpolate import PchipInterpolator  # type: ignore
except Exception:
    PchipInterpolator = None


# =========================
# Defaults
# =========================

DEFAULT_DUET_WEB_ADDRESS = "http://192.168.2.21"
DEFAULT_CAMERA_PORT = 0
DEFAULT_PROJECT_NAME = "Circle_Tracking_Run"
DEFAULT_ALLOW_EXISTING = True
DEFAULT_ADD_DATE = True

DEFAULT_MANUAL_FOCUS = True
DEFAULT_MANUAL_FOCUS_VAL = 60
DEFAULT_CAMERA_WIDTH = 3840
DEFAULT_CAMERA_HEIGHT = 2160
DEFAULT_CAMERA_FLUSH_FRAMES = 1

DEFAULT_TRAVEL_FEED = 2000.0
DEFAULT_PRINT_FEED = 2000.0
DEFAULT_FINE_APPROACH_FEED = 150.0
DEFAULT_PRINT_FEED_B = 200.0
DEFAULT_PRINT_FEED_C = 20000.0
DEFAULT_TRANSITION_FEED = 1200.0
DEFAULT_C_FLIP_DELAY_S = 4.0

DEFAULT_DWELL_BEFORE_MS = 0
DEFAULT_DWELL_AFTER_MS = 0
DEFAULT_INITIAL_SWEEP_WAIT_S = 6.0
DEFAULT_TRACKED_MOVE_SETTLE_S = 0.0
DEFAULT_TRAVEL_MOVE_SETTLE_S = 0.0
DEFAULT_B_EXTRA_SETTLE_S = 0.0
DEFAULT_CAPTURE_AT_START = False
DEFAULT_INTER_COMMAND_DELAY_S = 0.005
DEFAULT_FINAL_RECENTER = True
DEFAULT_ENABLE_POST = False
DEFAULT_USE_AVERAGE_CUBIC_FIT = False

DEFAULT_POST_CAMERA_CALIBRATION_FILE = "captures/calibration_webcam_20260406_104136.npz"
DEFAULT_POST_CHECKERBOARD_REFERENCE_IMAGE = "captures/photo_20260406_104134.png"

DEFAULT_SAFE_APPROACH_Z = -155.0

DEFAULT_START_X = 100.0
DEFAULT_START_Y = 52.0
DEFAULT_START_Z = -155.0
DEFAULT_START_B = 0.0
DEFAULT_START_C = 0.0

DEFAULT_END_X = 100.0
DEFAULT_END_Y = 52.0
DEFAULT_END_Z = -155.0
DEFAULT_END_B = 0.0
DEFAULT_END_C = 0.0

DEFAULT_BBOX_X_MIN = 0.0
DEFAULT_BBOX_X_MAX = 200.0
DEFAULT_BBOX_Y_MIN = -20.0
DEFAULT_BBOX_Y_MAX = 200.0
DEFAULT_BBOX_Z_MIN = -200.0
DEFAULT_BBOX_Z_MAX = 0.0

CIRCLE_CENTER_X = 100.0
CIRCLE_CENTER_Y = 52.0
CIRCLE_CENTER_Z = -155.0

DEFAULT_SAMPLES_PER_QUARTER = 200
DEFAULT_CAPTURE_EVERY_N_CIRCLE_MOVES = 7
DEFAULT_C0_DEG = 0.0
DEFAULT_FLIP_RZ_SIGN = True
DEFAULT_QUARTER_GAP_MM = 15.0

OFFPLANE_SIGN = -1.0
C_HARD_MIN_DEG = -360.0
C_HARD_MAX_DEG = 360.0

RECORDED_PHASES = {
    "q1_pull",
    "q2_release",
    "q3_pull",
    "q4_release",
}
TRANSITION_PHASES = {
    "pull_to_release_1",
    "midpoint_c_flip",
    "pull_to_release_2",
    "final_recenter",
}


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
class CommandPoint:
    phase: str
    x: float
    y: float
    z: float
    b: float
    c: float
    feed: float
    tip_x: float
    tip_y: float
    tip_z: float
    motion_phase: str


# =========================
# Safety helpers
# =========================

def assert_c_in_safe_range(name: str, c_deg: float) -> float:
    c = float(c_deg)
    if c < C_HARD_MIN_DEG or c > C_HARD_MAX_DEG:
        raise ValueError(
            f"{name}={c:.3f} is outside the safe physical bound "
            f"[{C_HARD_MIN_DEG:.1f}, {C_HARD_MAX_DEG:.1f}] deg."
        )
    return c


def assert_all_command_c_safe(command_sequence: List[CommandPoint]) -> None:
    bad = [cp.c for cp in command_sequence if cp.c < C_HARD_MIN_DEG or cp.c > C_HARD_MAX_DEG]
    if bad:
        raise ValueError(
            "Generated command sequence contains out-of-range C values: "
            + ", ".join(f"{v:.3f}" for v in bad[:10])
        )


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
        if PchipInterpolator is None:
            raise ImportError("PCHIP calibration model requires scipy.interpolate.PchipInterpolator.")
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

    default_phase = _normalize_motion_phase_name(data.get("default_phase_for_legacy_access"))
    if default_phase is None or default_phase not in phase_models:
        if "pull" in phase_models:
            default_phase = "pull"
        elif phase_models:
            default_phase = next(iter(phase_models))
        else:
            default_phase = "pull"

    return phase_models, default_phase


def load_calibration(json_path: str) -> Calibration:
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
        cubic.get("r_coeffs"), cubic.get("r_equation"), "r"
    )
    z_model = fit_models.get("z") or default_phase_models.get("z") or legacy_poly_model(
        cubic.get("z_coeffs"), cubic.get("z_equation"), "z"
    )
    y_off_model = fit_models.get("offplane_y") or default_phase_models.get("offplane_y") or legacy_poly_model(
        cubic.get("offplane_y_coeffs"), cubic.get("offplane_y_equation"), "y_offplane_mm"
    )
    tip_angle_model = fit_models.get("tip_angle") or default_phase_models.get("tip_angle") or legacy_poly_model(
        cubic.get("tip_angle_coeffs"), cubic.get("tip_angle_equation"), "tip_angle_deg"
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
            None if cubic.get("offplane_y_r_squared") is None else float(cubic["offplane_y_r_squared"])
        ),
    )


def resolve_phase_name(cal: Calibration, base_name: str) -> str:
    want = _normalize_motion_phase_name(base_name)
    if want is None:
        return cal.default_motion_phase

    phase_keys = list(cal.phase_models.keys())
    if want in cal.phase_models:
        return want

    exact_prefix_matches = [k for k in phase_keys if k.startswith(want)]
    if exact_prefix_matches:
        exact_prefix_matches.sort()
        return exact_prefix_matches[0]

    contains_matches = [k for k in phase_keys if want in k]
    if contains_matches:
        contains_matches.sort()
        return contains_matches[0]

    return cal.default_motion_phase


def _select_fit_model(cal: Calibration, model_name: str, motion_phase: Optional[str] = None) -> Any:
    phase_name = _normalize_motion_phase_name(motion_phase) or cal.default_motion_phase
    if phase_name in cal.phase_models and model_name in cal.phase_models[phase_name]:
        return cal.phase_models[phase_name][model_name]

    fallback_attr = {
        "r": cal.r_model,
        "z": cal.z_model,
        "offplane_y": cal.y_off_model,
        "tip_angle": cal.tip_angle_model,
    }.get(model_name)
    return fallback_attr


def _phase_model_variant(cal: Calibration, phase_name: str, variant_key: str) -> Optional[dict]:
    phase = _normalize_motion_phase_name(phase_name)
    if phase is None:
        return None
    payload = cal.phase_models.get(phase)
    if not isinstance(payload, dict):
        return None
    model = payload.get(variant_key)
    return dict(model) if isinstance(model, dict) else None


def _average_polynomial_models(models: List[dict], value_name: str) -> dict:
    usable = [m for m in models if isinstance(m, dict)]
    if not usable:
        raise ValueError(f"No usable polynomial models were provided for {value_name}.")

    coeff_arrays = [np.asarray(m.get("coefficients"), dtype=float).reshape(-1) for m in usable]
    if any(arr.size == 0 for arr in coeff_arrays):
        raise ValueError(f"Polynomial model for {value_name} is missing coefficients.")
    degrees = {arr.size for arr in coeff_arrays}
    if len(degrees) != 1:
        raise ValueError(f"Polynomial models for {value_name} do not share the same degree.")

    avg_coeffs = np.mean(np.stack(coeff_arrays, axis=0), axis=0)

    fit_lo = -np.inf
    fit_hi = np.inf
    sample_count = 0
    for model in usable:
        fit_x_range = model.get("fit_x_range")
        if fit_x_range and len(fit_x_range) == 2:
            lo = float(min(fit_x_range))
            hi = float(max(fit_x_range))
            fit_lo = max(fit_lo, lo)
            fit_hi = min(fit_hi, hi)
        sample_count = max(sample_count, int(model.get("sample_count", 0) or 0))

    avg_model = {
        "model_type": "polynomial",
        "basis": "monomial",
        "degree": int(avg_coeffs.size - 1),
        "input_axis": "b_motor",
        "value_name": str(value_name),
        "coefficients": avg_coeffs.tolist(),
        "equation": f"{value_name}(b) = averaged polynomial coefficients across phases",
        "sample_count": int(sample_count),
    }
    if np.isfinite(fit_lo) and np.isfinite(fit_hi) and fit_lo <= fit_hi:
        avg_model["fit_x_range"] = [float(fit_lo), float(fit_hi)]
    return avg_model


def calibration_with_average_cubic_override(cal: Calibration) -> Calibration:
    cal_out = copy.deepcopy(cal)

    pull_phase = resolve_phase_name(cal_out, "pull")
    release_phase = resolve_phase_name(cal_out, "release")

    avg_r_model = _average_polynomial_models(
        [
            _phase_model_variant(cal_out, pull_phase, "r_cubic"),
            _phase_model_variant(cal_out, release_phase, "r_cubic"),
        ],
        value_name="r_avg_cubic_override",
    )
    avg_z_model = _average_polynomial_models(
        [
            _phase_model_variant(cal_out, pull_phase, "z_cubic"),
            _phase_model_variant(cal_out, release_phase, "z_cubic"),
        ],
        value_name="z_avg_cubic_override",
    )

    for phase_name in {pull_phase, release_phase}:
        phase_payload = cal_out.phase_models.get(phase_name)
        if not isinstance(phase_payload, dict):
            phase_payload = {}
            cal_out.phase_models[phase_name] = phase_payload
        phase_payload["r"] = dict(avg_r_model)
        phase_payload["z"] = dict(avg_z_model)

    cal_out.r_model = dict(avg_r_model)
    cal_out.z_model = dict(avg_z_model)
    return cal_out


def eval_r(cal: Calibration, b: Any, flip_rz_sign: bool = False, motion_phase: Optional[str] = None) -> np.ndarray:
    s = -1.0 * (-1.0 if bool(flip_rz_sign) else 1.0)
    return s * evaluate_fit_model(_select_fit_model(cal, "r", motion_phase=motion_phase), b)


def eval_z(cal: Calibration, b: Any, flip_rz_sign: bool = False, motion_phase: Optional[str] = None) -> np.ndarray:
    return evaluate_fit_model(_select_fit_model(cal, "z", motion_phase=motion_phase), b)


def eval_offplane_y(cal: Calibration, b: Any, motion_phase: Optional[str] = None) -> np.ndarray:
    return OFFPLANE_SIGN * evaluate_fit_model(
        _select_fit_model(cal, "offplane_y", motion_phase=motion_phase),
        b,
        default_if_none=0.0,
    )


def tip_offset_xyz_physical(
    cal: Calibration,
    b: float,
    c_deg: float,
    flip_rz_sign: bool = False,
    motion_phase: Optional[str] = None,
) -> np.ndarray:
    c_deg = assert_c_in_safe_range("tip_offset C", c_deg)

    r = float(eval_r(cal, b, flip_rz_sign=flip_rz_sign, motion_phase=motion_phase))
    z = float(eval_z(cal, b, flip_rz_sign=flip_rz_sign, motion_phase=motion_phase))
    y_off = float(eval_offplane_y(cal, b, motion_phase=motion_phase))

    c = math.radians(float(c_deg))
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def stage_xyz_for_tip_target(
    cal: Calibration,
    p_tip_xyz: np.ndarray,
    b: float,
    c_deg: float,
    flip_rz_sign: bool = False,
    motion_phase: Optional[str] = None,
) -> np.ndarray:
    return p_tip_xyz - tip_offset_xyz_physical(
        cal, b, c_deg, flip_rz_sign=flip_rz_sign, motion_phase=motion_phase
    )


def _model_b_range(model: Optional[dict], fallback_lo: float, fallback_hi: float) -> Tuple[float, float]:
    if model is None:
        return float(fallback_lo), float(fallback_hi)

    if model.get("model_type", "").lower() == "pchip":
        x_knots = model.get("x_knots") or []
        if len(x_knots) >= 2:
            return float(min(x_knots)), float(max(x_knots))

    fit_x_range = model.get("fit_x_range")
    if fit_x_range and len(fit_x_range) == 2:
        return float(min(fit_x_range)), float(max(fit_x_range))

    return float(fallback_lo), float(fallback_hi)


def common_b_window_for_pull_release(
    cal: Calibration,
    b_lo_user: float,
    b_hi_user: float,
    pull_phase: str,
    release_phase: str,
) -> Tuple[float, float]:
    pull_r_lo, pull_r_hi = _model_b_range(_select_fit_model(cal, "r", motion_phase=pull_phase), cal.b_min, cal.b_max)
    rel_r_lo, rel_r_hi = _model_b_range(_select_fit_model(cal, "r", motion_phase=release_phase), cal.b_min, cal.b_max)

    lo = max(float(b_lo_user), pull_r_lo, rel_r_lo)
    hi = min(float(b_hi_user), pull_r_hi, rel_r_hi)
    if lo > hi:
        raise RuntimeError(
            f"No common B range available for pull/release within requested limits: "
            f"pull=[{pull_r_lo:.3f},{pull_r_hi:.3f}], release=[{rel_r_lo:.3f},{rel_r_hi:.3f}], "
            f"user=[{b_lo_user:.3f},{b_hi_user:.3f}]"
        )
    return float(lo), float(hi)


# =========================
# Circle planner
# =========================

def find_circle_switch_b(
    cal: Calibration,
    flip_rz_sign: bool,
    c0_deg: float,
    pull_phase: str,
    release_phase: str,
    b_lo: float,
    b_hi: float,
    quarter_b_override: Optional[float] = None,
    n_dense: int = 6001,
) -> Tuple[float, float]:
    common_lo, common_hi = common_b_window_for_pull_release(
        cal=cal,
        b_lo_user=b_lo,
        b_hi_user=b_hi,
        pull_phase=pull_phase,
        release_phase=release_phase,
    )

    b_start = 0.0
    if not (common_lo <= b_start <= common_hi):
        raise RuntimeError(
            f"B=0 is not inside the common pull/release range [{common_lo:.3f}, {common_hi:.3f}]."
        )

    if quarter_b_override is not None:
        b_switch = float(quarter_b_override)
        if b_switch < common_lo or b_switch > common_hi:
            raise ValueError(
                f"--quarter-b={b_switch:.3f} is outside the common pull/release range "
                f"[{common_lo:.3f}, {common_hi:.3f}]"
            )
        x0 = float(tip_offset_xyz_physical(
            cal, b_start, c0_deg, flip_rz_sign=flip_rz_sign, motion_phase=pull_phase
        )[0])
        xq = float(tip_offset_xyz_physical(
            cal, b_switch, c0_deg, flip_rz_sign=flip_rz_sign, motion_phase=pull_phase
        )[0])
        radius = abs(xq - x0)
        if radius <= 0.0:
            raise RuntimeError("Computed circle radius is not positive.")
        return float(b_switch), float(radius)

    bb = np.linspace(common_lo, b_start, int(n_dense), dtype=float)
    x_pull = np.asarray([
        tip_offset_xyz_physical(
            cal, float(b), c0_deg, flip_rz_sign=flip_rz_sign, motion_phase=pull_phase
        )[0]
        for b in bb
    ], dtype=float)
    x_release = np.asarray([
        tip_offset_xyz_physical(
            cal, float(b), c0_deg, flip_rz_sign=flip_rz_sign, motion_phase=release_phase
        )[0]
        for b in bb
    ], dtype=float)

    x0_pull = float(tip_offset_xyz_physical(
        cal, b_start, c0_deg, flip_rz_sign=flip_rz_sign, motion_phase=pull_phase
    )[0])

    best_radius = -1.0
    best_b = None

    for i, b_switch in enumerate(bb):
        signed_dx_q = float(x_pull[i] - x0_pull)
        radius = abs(signed_dx_q)
        if radius <= 0.0:
            continue

        # Pull quarter must stay inside the chosen radius before the switch.
        dx_pull_path = x_pull[i:] - x0_pull
        if np.max(np.abs(dx_pull_path)) > radius + 1e-6:
            continue

        # Release quarter is translated so it starts exactly at the current tip location.
        # It must also stay inside the same circle radius while returning toward B=0.
        dx_release_shifted = signed_dx_q + (x_release[i:] - x_release[i])
        if np.max(np.abs(dx_release_shifted)) > radius + 1e-6:
            continue

        if radius > best_radius:
            best_radius = float(radius)
            best_b = float(b_switch)

    if best_b is None or best_radius <= 0.0:
        raise RuntimeError(
            "Could not find a feasible curl->uncurl switch point that stays inside a single-radius circle."
        )

    return float(best_b), float(best_radius)


def _safe_circle_dz(radius: float, dx: float) -> float:
    residual = float(radius * radius - dx * dx)
    if residual < -1e-6:
        raise RuntimeError(
            f"Requested circle point is not reachable with the chosen segment radius: "
            f"radius={radius:.6f}, dx={dx:.6f}"
        )
    return math.sqrt(max(0.0, residual))


def build_circle_quarter_segment(
    cal: Calibration,
    flip_rz_sign: bool,
    center_x: float,
    center_y: float,
    center_z: float,
    radius: float,
    b_values: np.ndarray,
    c_state: float,
    motion_phase: str,
    upper_half: bool,
    start_label: str,
    rest_label: str,
    move_feed_start: float,
    move_feed_rest: float,
    stage_x_const: Optional[float] = None,
) -> Tuple[List[CommandPoint], Dict[str, float], float]:
    c_state = assert_c_in_safe_range("c_state", c_state)
    b_values = np.asarray(b_values, dtype=float)
    if b_values.size < 2:
        raise ValueError("Each circle quarter must contain at least 2 B samples.")

    if stage_x_const is None:
        x_offset_start = float(tip_offset_xyz_physical(
            cal, float(b_values[0]), c_state, flip_rz_sign=flip_rz_sign, motion_phase=motion_phase
        )[0])
        stage_x_const = float(center_x - x_offset_start)

    pts: List[CommandPoint] = []
    stage_x_trace = []

    for i, b_i in enumerate(b_values):
        offset = tip_offset_xyz_physical(
            cal, float(b_i), c_state, flip_rz_sign=flip_rz_sign, motion_phase=motion_phase
        )
        x_tip = float(stage_x_const + offset[0])
        dx = float(x_tip - float(center_x))
        dz_mag = _safe_circle_dz(float(radius), dx)
        z_tip = float(center_z + (dz_mag if upper_half else -dz_mag))
        tip_target = np.array([x_tip, float(center_y), z_tip], dtype=float)

        stage_xyz = stage_xyz_for_tip_target(
            cal,
            tip_target,
            float(b_i),
            float(c_state),
            flip_rz_sign=flip_rz_sign,
            motion_phase=motion_phase,
        )

        tip_check = stage_xyz + tip_offset_xyz_physical(
            cal,
            float(b_i),
            float(c_state),
            flip_rz_sign=flip_rz_sign,
            motion_phase=motion_phase,
        )
        if not np.allclose(tip_check, tip_target, atol=1e-9, rtol=0.0):
            raise RuntimeError("Internal tip-tracking consistency check failed in circle planner.")

        stage_x_trace.append(float(stage_xyz[0]))
        pts.append(
            CommandPoint(
                phase=start_label if i == 0 else rest_label,
                x=float(stage_xyz[0]),
                y=float(stage_xyz[1]),
                z=float(stage_xyz[2]),
                b=float(b_i),
                c=float(c_state),
                feed=float(move_feed_start if i == 0 else move_feed_rest),
                tip_x=float(tip_target[0]),
                tip_y=float(tip_target[1]),
                tip_z=float(tip_target[2]),
                motion_phase=str(motion_phase),
            )
        )

    x_var = float(np.max(stage_x_trace) - np.min(stage_x_trace)) if stage_x_trace else 0.0
    meta = {
        "stage_x_const": float(stage_x_const),
        "stage_x_variation": float(x_var),
        "tip_x_min": float(min(p.tip_x for p in pts)),
        "tip_x_max": float(max(p.tip_x for p in pts)),
        "tip_z_min": float(min(p.tip_z for p in pts)),
        "tip_z_max": float(max(p.tip_z for p in pts)),
        "stage_y_min": float(min(p.y for p in pts)),
        "stage_y_max": float(max(p.y for p in pts)),
        "stage_z_min": float(min(p.z for p in pts)),
        "stage_z_max": float(max(p.z for p in pts)),
    }
    return pts, meta, float(stage_x_const)


def transition_point_for_same_tip(
    cal: Calibration,
    flip_rz_sign: bool,
    tip_target: np.ndarray,
    b_value: float,
    c_state: float,
    motion_phase: str,
    phase_label: str,
    feed: float,
) -> CommandPoint:
    stage_xyz = stage_xyz_for_tip_target(
        cal,
        tip_target,
        float(b_value),
        float(c_state),
        flip_rz_sign=flip_rz_sign,
        motion_phase=motion_phase,
    )
    return CommandPoint(
        phase=str(phase_label),
        x=float(stage_xyz[0]),
        y=float(stage_xyz[1]),
        z=float(stage_xyz[2]),
        b=float(b_value),
        c=float(c_state),
        feed=float(feed),
        tip_x=float(tip_target[0]),
        tip_y=float(tip_target[1]),
        tip_z=float(tip_target[2]),
        motion_phase=str(motion_phase),
    )


def build_circle_command_sequence(
    cal: Calibration,
    flip_rz_sign: bool,
    center_x: float,
    center_y: float,
    center_z: float,
    samples_per_quarter: int,
    b_lo: float,
    b_hi: float,
    c0_deg: float,
    c180_deg: float,
    jog_feed: float,
    print_feed_b: float,
    print_feed_c: float,
    transition_feed: float,
    quarter_b_override: Optional[float] = None,
    final_recenter: bool = True,
    quarter_gap_mm: float = DEFAULT_QUARTER_GAP_MM,
) -> Tuple[List[CommandPoint], dict]:
    c0_deg = assert_c_in_safe_range("c0_deg", c0_deg)
    c180_deg = assert_c_in_safe_range("c180_deg", c180_deg)

    pull_phase = resolve_phase_name(cal, "pull")
    release_phase = resolve_phase_name(cal, "release")

    b_switch, radius = find_circle_switch_b(
        cal=cal,
        flip_rz_sign=flip_rz_sign,
        c0_deg=c0_deg,
        pull_phase=pull_phase,
        release_phase=release_phase,
        b_lo=b_lo,
        b_hi=b_hi,
        quarter_b_override=quarter_b_override,
    )

    n = max(2, int(samples_per_quarter))
    gap = max(0.0, float(quarter_gap_mm))
    b_q_pull = np.linspace(0.0, float(b_switch), n, dtype=float)
    b_q_release = np.linspace(float(b_switch), 0.0, n, dtype=float)

    q1_center_x = float(center_x)
    q1_center_z = float(center_z)
    q2_center_x = float(center_x)
    q2_center_z = float(center_z + gap)
    q3_center_x = float(center_x - gap)
    q3_center_z = float(center_z + gap)
    q4_center_x = float(center_x - gap)
    q4_center_z = float(center_z)

    q1_pts, meta_q1, x_stage_q1 = build_circle_quarter_segment(
        cal=cal,
        flip_rz_sign=flip_rz_sign,
        center_x=float(q1_center_x),
        center_y=float(center_y),
        center_z=float(q1_center_z),
        radius=float(radius),
        b_values=b_q_pull,
        c_state=float(c0_deg),
        motion_phase=pull_phase,
        upper_half=False,
        start_label="q1_pull_start",
        rest_label="q1_pull",
        move_feed_start=float(jog_feed),
        move_feed_rest=float(print_feed_b),
        stage_x_const=None,
    )

    q1_end_tip = np.array([q1_pts[-1].tip_x, q1_pts[-1].tip_y, q1_pts[-1].tip_z], dtype=float)
    q2_start_tip = np.array(
        [float(q2_center_x + radius), float(center_y), float(q2_center_z)],
        dtype=float,
    )
    pull_to_release_1 = transition_point_for_same_tip(
        cal=cal,
        flip_rz_sign=flip_rz_sign,
        tip_target=q2_start_tip,
        b_value=float(b_switch),
        c_state=float(c0_deg),
        motion_phase=release_phase,
        phase_label="pull_to_release_1",
        feed=float(transition_feed),
    )

    q2_pts, meta_q2, x_stage_q2 = build_circle_quarter_segment(
        cal=cal,
        flip_rz_sign=flip_rz_sign,
        center_x=float(q2_center_x),
        center_y=float(center_y),
        center_z=float(q2_center_z),
        radius=float(radius),
        b_values=b_q_release,
        c_state=float(c0_deg),
        motion_phase=release_phase,
        upper_half=True,
        start_label="q2_release_start",
        rest_label="q2_release",
        move_feed_start=float(transition_feed),
        move_feed_rest=float(print_feed_b),
        stage_x_const=float(pull_to_release_1.x),
    )

    q3_preflip_tip = np.array(
        [float(q3_center_x), float(center_y), float(q3_center_z + radius)],
        dtype=float,
    )
    quarter_gap_2 = transition_point_for_same_tip(
        cal=cal,
        flip_rz_sign=flip_rz_sign,
        tip_target=q3_preflip_tip,
        b_value=0.0,
        c_state=float(c0_deg),
        motion_phase=release_phase,
        phase_label="quarter_gap_2",
        feed=float(transition_feed),
    )
    midpoint_c_flip = transition_point_for_same_tip(
        cal=cal,
        flip_rz_sign=flip_rz_sign,
        tip_target=q3_preflip_tip,
        b_value=0.0,
        c_state=float(c180_deg),
        motion_phase=pull_phase,
        phase_label="midpoint_c_flip",
        feed=float(print_feed_c),
    )

    q3_pts, meta_q3, x_stage_q3 = build_circle_quarter_segment(
        cal=cal,
        flip_rz_sign=flip_rz_sign,
        center_x=float(q3_center_x),
        center_y=float(center_y),
        center_z=float(q3_center_z),
        radius=float(radius),
        b_values=b_q_pull,
        c_state=float(c180_deg),
        motion_phase=pull_phase,
        upper_half=True,
        start_label="q3_pull_start",
        rest_label="q3_pull",
        move_feed_start=float(transition_feed),
        move_feed_rest=float(print_feed_b),
        stage_x_const=float(midpoint_c_flip.x),
    )

    q3_end_tip = np.array([q3_pts[-1].tip_x, q3_pts[-1].tip_y, q3_pts[-1].tip_z], dtype=float)
    q4_start_tip = np.array(
        [float(q4_center_x - radius), float(center_y), float(q4_center_z)],
        dtype=float,
    )
    pull_to_release_2 = transition_point_for_same_tip(
        cal=cal,
        flip_rz_sign=flip_rz_sign,
        tip_target=q4_start_tip,
        b_value=float(b_switch),
        c_state=float(c180_deg),
        motion_phase=release_phase,
        phase_label="pull_to_release_2",
        feed=float(transition_feed),
    )

    q4_pts, meta_q4, x_stage_q4 = build_circle_quarter_segment(
        cal=cal,
        flip_rz_sign=flip_rz_sign,
        center_x=float(q4_center_x),
        center_y=float(center_y),
        center_z=float(q4_center_z),
        radius=float(radius),
        b_values=b_q_release,
        c_state=float(c180_deg),
        motion_phase=release_phase,
        upper_half=False,
        start_label="q4_release_start",
        rest_label="q4_release",
        move_feed_start=float(transition_feed),
        move_feed_rest=float(print_feed_b),
        stage_x_const=float(pull_to_release_2.x),
    )

    sequence: List[CommandPoint] = []
    sequence.extend(q1_pts)
    sequence.append(pull_to_release_1)
    sequence.extend(q2_pts[1:])
    sequence.append(quarter_gap_2)
    sequence.append(midpoint_c_flip)
    sequence.extend(q3_pts[1:])
    sequence.append(pull_to_release_2)
    sequence.extend(q4_pts[1:])

    final_recenter_cp = None
    exact_bottom_tip = np.array([float(q4_center_x), float(center_y), float(q4_center_z - radius)], dtype=float)
    if bool(final_recenter):
        final_recenter_cp = transition_point_for_same_tip(
            cal=cal,
            flip_rz_sign=flip_rz_sign,
            tip_target=np.array([float(center_x), float(center_y), float(center_z - radius)], dtype=float),
            b_value=0.0,
            c_state=float(c180_deg),
            motion_phase=release_phase,
            phase_label="final_recenter",
            feed=float(transition_feed),
        )
        sequence.append(final_recenter_cp)

    assert_all_command_c_safe(sequence)

    stage_x_ranges = {
        "q1": float(meta_q1["stage_x_variation"]),
        "q2": float(meta_q2["stage_x_variation"]),
        "q3": float(meta_q3["stage_x_variation"]),
        "q4": float(meta_q4["stage_x_variation"]),
    }

    q2_end = q2_pts[-1]
    q4_end = q4_pts[-1]
    midpoint_preflip_error = {
        "dx": float(q2_end.tip_x - q3_preflip_tip[0]),
        "dy": float(q2_end.tip_y - q3_preflip_tip[1]),
        "dz": float(q2_end.tip_z - q3_preflip_tip[2]),
        "dist": float(np.linalg.norm(
            np.array([q2_end.tip_x, q2_end.tip_y, q2_end.tip_z], dtype=float) - q3_preflip_tip
        )),
    }
    closure_error_before_final = {
        "dx": float(q4_end.tip_x - exact_bottom_tip[0]),
        "dy": float(q4_end.tip_y - exact_bottom_tip[1]),
        "dz": float(q4_end.tip_z - exact_bottom_tip[2]),
        "dist": float(np.linalg.norm(
            np.array([q4_end.tip_x, q4_end.tip_y, q4_end.tip_z], dtype=float) - exact_bottom_tip
        )),
    }

    meta = {
        "pull_phase": str(pull_phase),
        "release_phase": str(release_phase),
        "quarter_b": float(b_switch),
        "circle_radius": float(radius),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "center_z": float(center_z),
        "quarter_gap_mm": float(gap),
        "quarter_centers": {
            "q1": {"x": float(q1_center_x), "z": float(q1_center_z)},
            "q2": {"x": float(q2_center_x), "z": float(q2_center_z)},
            "q3": {"x": float(q3_center_x), "z": float(q3_center_z)},
            "q4": {"x": float(q4_center_x), "z": float(q4_center_z)},
        },
        "c0_deg": float(c0_deg),
        "c180_deg": float(c180_deg),
        "flip_rz_sign": bool(flip_rz_sign),
        "samples_per_quarter": int(n),
        "stage_x_variation_by_recorded_quarter": stage_x_ranges,
        "midpoint_preflip_error": midpoint_preflip_error,
        "closure_error_before_final": closure_error_before_final,
        "transition_stage_x": {
            "q1": float(x_stage_q1),
            "q2": float(x_stage_q2),
            "q3": float(x_stage_q3),
            "q4": float(x_stage_q4),
        },
        "meta_q1": meta_q1,
        "meta_q2": meta_q2,
        "meta_q3": meta_q3,
        "meta_q4": meta_q4,
        "x_stage_min": float(min(p.x for p in sequence)),
        "x_stage_max": float(max(p.x for p in sequence)),
        "y_stage_min": float(min(p.y for p in sequence)),
        "y_stage_max": float(max(p.y for p in sequence)),
        "z_stage_min": float(min(p.z for p in sequence)),
        "z_stage_max": float(max(p.z for p in sequence)),
        "tip_x_min": float(min(p.tip_x for p in sequence)),
        "tip_x_max": float(max(p.tip_x for p in sequence)),
        "tip_z_min": float(min(p.tip_z for p in sequence)),
        "tip_z_max": float(max(p.tip_z for p in sequence)),
        "final_recenter_enabled": bool(final_recenter),
        "final_recenter_point": None if final_recenter_cp is None else asdict(final_recenter_cp),
    }
    return sequence, meta


# =========================
# Utilities
# =========================

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


def save_desired_circle_motion_plot(plot_path: str, command_sequence: List[CommandPoint]) -> str:
    if not command_sequence:
        raise RuntimeError("Cannot save desired circle motion plot: command_sequence is empty.")

    import matplotlib.pyplot as plt

    tip_x = np.asarray([cp.tip_x for cp in command_sequence if cp.phase in RECORDED_PHASES or cp.phase.endswith("_start")], dtype=float)
    tip_z = np.asarray([cp.tip_z for cp in command_sequence if cp.phase in RECORDED_PHASES or cp.phase.endswith("_start")], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor=(0.0, 0.0, 0.0, 0.0))
    ax.set_facecolor((0.04, 0.09, 0.14, 0.88))

    ax.plot(
        tip_x,
        tip_z,
        color="#8cf7ff",
        linewidth=2.4,
        alpha=0.98,
        label="Desired circle motion",
        zorder=2,
    )
    ax.scatter(
        tip_x,
        tip_z,
        s=12,
        color="#f8fafc",
        edgecolors="#8cf7ff",
        linewidths=0.4,
        alpha=0.95,
        label="Sampled tip targets",
        zorder=3,
    )
    ax.scatter([tip_x[0]], [tip_z[0]], s=72, color="#f4d35e", edgecolors="none", label="Start", zorder=4)

    cx = float(np.mean([np.min(tip_x), np.max(tip_x)]))
    cz = float(np.mean([np.min(tip_z), np.max(tip_z)]))
    span_x = max(float(np.max(np.abs(tip_x - cx))), 1.0)
    span_z = max(float(np.max(np.abs(tip_z - cz))), 1.0)
    span = 1.08 * max(span_x, span_z)

    ax.set_xlim(cx - span, cx + span)
    ax.set_ylim(cz - span, cz + span)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Desired tip X (mm)", color="#e8f3ff")
    ax.set_ylabel("Desired tip Z (mm)", color="#e8f3ff")
    ax.set_title("Desired Generated Circle Motion", color="#f8fbff")
    ax.grid(True, color="#8eb8d8", alpha=0.14, linewidth=0.8)
    ax.tick_params(colors="#dceaf7")
    for spine in ax.spines.values():
        spine.set_color("#6b92b3")

    legend = ax.legend(frameon=True, facecolor="#102131", edgecolor="#6b92b3")
    for txt in legend.get_texts():
        txt.set_color("#f8fbff")

    plt.tight_layout()
    plt.savefig(plot_path, dpi=220, transparent=True)
    plt.close(fig)
    return plot_path


def save_command_sequence_csv(csv_path: str, command_sequence: List[CommandPoint]) -> str:
    fieldnames = [
        "idx", "phase", "motion_phase", "x", "y", "z", "b", "c", "feed", "tip_x", "tip_y", "tip_z"
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for idx, cp in enumerate(command_sequence):
            row = asdict(cp)
            row["idx"] = idx
            writer.writerow({k: row[k] for k in fieldnames})
    return csv_path


def save_json(path: str, payload: Any) -> str:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return path


# =========================
# Acquisition runner
# =========================

class CircleTrackerRunner:
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

        self.raw_image_data_folder = os.path.join(self.run_folder, "raw_image_data_folder")
        os.makedirs(self.raw_image_data_folder, exist_ok=True)

        self.cam = None
        self.rrf = None
        self.cam_port = None
        self.commanded_axes: dict = {}

        print(f"Using run folder: {self.run_folder}")
        print(f"Using raw image folder: {self.raw_image_data_folder}")

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
    ) -> Optional[str]:
        if self.cam is None:
            raise RuntimeError("Camera is not connected.")

        for _ in range(max(0, int(flush_frames))):
            _ = self.cam.read()

        ret, image = self.cam.read()
        if not ret:
            ret, image = self.cam.read()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = (
            f"{sample_idx:05d}"
            f"_{phase}"
            f"_X{x:.3f}_Y{y:.3f}_Z{z:.3f}"
            f"_B{b:.3f}_C{c:.3f}"
            f"_{timestamp}.png"
        ).replace(" ", "_")

        path = os.path.join(self.raw_image_data_folder, filename)
        if ret and image is not None:
            cv2.imwrite(path, image)
            print(f" ✓ Saved image: {filename}")
            return path

        print(f" ✗ Failed to capture image: {filename}")
        return None

    def connect_to_robot(self, duet_web_address: str):
        if DuetWebAPI is None:
            raise ImportError(
                "Missing duetwebapi. Install with:\n"
                "    pip install duetwebapi==1.1.0"
            )
        self.rrf = DuetWebAPI(duet_web_address)
        print("Connection attempted. Requesting diagnostics.")
        resp = self.rrf.send_code("M122")
        print("Returned diagnostics data:")
        print(resp)
        print("Robot connected.")

    def disconnect_robot(self):
        self.rrf = None

    def _estimate_move_time_s(self, feedrate: float, axes_targets: dict) -> float:
        feed = max(1e-6, float(feedrate))
        deltas = []
        for ax, val in axes_targets.items():
            if val is None:
                continue
            prev = self.commanded_axes.get(str(ax))
            if prev is None:
                continue
            deltas.append(abs(float(val) - float(prev)))

        if not deltas:
            return 0.0

        return 60.0 * max(deltas) / feed

    def _compute_inter_command_wait_s(
        self,
        est_move_time_s: float,
        configured_floor_s: float = 0.0,
    ) -> float:
        est = max(0.0, float(est_move_time_s))
        floor_s = max(0.0, float(configured_floor_s))
        if est <= 0.0:
            return floor_s
        paced_wait = min(2.0, max(0.05, 0.95 * est))
        return max(floor_s, paced_wait)

    def wait_for_duet_motion_complete(self, extra_settle: float = 0.0):
        if self.rrf is None:
            raise RuntimeError("Robot is not connected.")

        try:
            self.rrf.send_code("M400")
        except Exception as e:
            print(f"Warning: M400 wait failed ({e}); applying settle only.")

        if extra_settle > 0:
            time.sleep(extra_settle)

    def send_absolute_move(self, feedrate: float, **axes_targets) -> float:
        if self.rrf is None:
            raise RuntimeError("Robot is not connected.")

        est_move_time_s = self._estimate_move_time_s(feedrate, axes_targets)
        parts = ["G90", "G1"]
        for ax, val in axes_targets.items():
            if val is None:
                continue
            parts.append(f"{ax}{float(val):.3f}")
        parts.append(f"F{float(feedrate):.3f}")
        gcode = " ".join(parts)
        print(f" Command: {gcode}")
        self.rrf.send_code(gcode)
        for ax, val in axes_targets.items():
            if val is None:
                continue
            self.commanded_axes[str(ax)] = float(val)
        return est_move_time_s

    def _move_to_pose_safe(
        self,
        cal: Calibration,
        pose: Tuple[float, float, float, float, float],
        safe_approach_z: float,
        travel_feed: float,
        settle_s: float,
    ):
        x, y, z, b, c = [float(v) for v in pose]
        c = assert_c_in_safe_range("safe pose C", c)

        self.send_absolute_move(
            travel_feed,
            **{
                cal.z_axis: float(safe_approach_z),
                cal.b_axis: b,
                cal.c_axis: c,
            }
        )
        self.wait_for_duet_motion_complete(extra_settle=settle_s)

        self.send_absolute_move(
            travel_feed,
            **{
                cal.x_axis: x,
                cal.y_axis: y,
                cal.b_axis: b,
                cal.c_axis: c,
            }
        )
        self.wait_for_duet_motion_complete(extra_settle=settle_s)

        self.send_absolute_move(
            travel_feed,
            **{
                cal.z_axis: z,
                cal.b_axis: b,
                cal.c_axis: c,
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
                cal.c_axis: assert_c_in_safe_range("fine approach C", c),
            }
        )
        self.wait_for_duet_motion_complete(extra_settle=settle_s)

    def execute_circle_motion_and_capture(
        self,
        cal: Calibration,
        command_sequence: List[CommandPoint],
        start_pose: Tuple[float, float, float, float, float],
        end_pose: Tuple[float, float, float, float, float],
        safe_approach_z: float,
        travel_feed: float,
        fine_approach_feed: float,
        virtual_bbox: dict,
        dwell_before_ms: int = 0,
        dwell_after_ms: int = 0,
        initial_sweep_wait_s: float = DEFAULT_INITIAL_SWEEP_WAIT_S,
        tracked_move_settle_s: float = 0.0,
        travel_move_settle_s: float = 0.0,
        b_extra_settle_s: float = 0.0,
        inter_command_delay_s: float = 0.0,
        camera_flush_frames: int = 1,
        capture_at_start: bool = True,
        capture_every_n_circle_moves: int = 1,
    ):
        if self.cam is None:
            raise RuntimeError("Camera is not connected.")
        if self.rrf is None:
            raise RuntimeError("Robot is not connected.")

        bbox_warnings: List[str] = []
        sample_counter = 0
        circle_move_counter = 0
        capture_every_n_circle_moves = max(1, int(capture_every_n_circle_moves))
        self.commanded_axes = {}

        print("\n" + "=" * 72)
        print("STARTING CIRCLE-TRACKING ACQUISITION RUN")
        print("=" * 72)
        print(f"Tracked samples: {len(command_sequence)}")

        assert_all_command_c_safe(command_sequence)

        print("\nSafe startup approach...")
        self._move_to_pose_safe(
            cal=cal,
            pose=start_pose,
            safe_approach_z=float(safe_approach_z),
            travel_feed=float(travel_feed),
            settle_s=float(travel_move_settle_s),
        )

        if not command_sequence:
            print("No command sequence generated.")
        else:
            first = command_sequence[0]
            x0, y0, z0 = _clamp_stage_xyz_to_bbox(
                first.x, first.y, first.z, virtual_bbox, "move to tracked start", bbox_warnings
            )

            print("\nMoving to first tracked sample...")
            self.send_absolute_move(
                first.feed,
                **{
                    cal.x_axis: x0,
                    cal.y_axis: y0,
                    cal.z_axis: z0,
                    cal.b_axis: first.b,
                    cal.c_axis: first.c,
                }
            )
            self.wait_for_duet_motion_complete(extra_settle=travel_move_settle_s)

            self._fine_land_on_point(
                cal=cal,
                x=x0,
                y=y0,
                z=z0,
                b=float(first.b),
                c=float(first.c),
                fine_feed=float(fine_approach_feed),
                settle_s=max(float(tracked_move_settle_s), 0.05),
            )

            if float(initial_sweep_wait_s) > 0:
                print(f"Waiting {float(initial_sweep_wait_s):.3f} s before starting the sweep...")
                time.sleep(float(initial_sweep_wait_s))

            if capture_at_start:
                sample_counter += 1
                self.capture_and_save(
                    sample_idx=sample_counter,
                    phase=first.phase,
                    x=x0,
                    y=y0,
                    z=z0,
                    b=first.b,
                    c=first.c,
                    flush_frames=camera_flush_frames,
                )

            if int(dwell_before_ms) > 0:
                print(f"Dwell before motion: {int(dwell_before_ms)} ms")
                time.sleep(float(dwell_before_ms) / 1000.0)

            print("\nExecuting circle tracking motion...")
            for i, cp in enumerate(command_sequence[1:], start=1):
                x, y, z = _clamp_stage_xyz_to_bbox(
                    cp.x, cp.y, cp.z, virtual_bbox, f"tracked sample {i}", bbox_warnings
                )

                est_move_time_s = self.send_absolute_move(
                    cp.feed,
                    **{
                        cal.x_axis: x,
                        cal.y_axis: y,
                        cal.z_axis: z,
                        cal.b_axis: cp.b,
                        cal.c_axis: cp.c,
                    }
                )
                wait_s = self._compute_inter_command_wait_s(
                    est_move_time_s=est_move_time_s,
                    configured_floor_s=float(inter_command_delay_s),
                )
                if wait_s > 0.0:
                    time.sleep(wait_s)

                if cp.phase == "midpoint_c_flip":
                    print(f"Holding {float(DEFAULT_C_FLIP_DELAY_S):.3f} s after C rotation...")
                    time.sleep(float(DEFAULT_C_FLIP_DELAY_S))

                if cp.phase in RECORDED_PHASES:
                    circle_move_counter += 1
                    if (circle_move_counter % capture_every_n_circle_moves) == 0:
                        sample_counter += 1
                        self.capture_and_save(
                            sample_idx=sample_counter,
                            phase=cp.phase,
                            x=x,
                            y=y,
                            z=z,
                            b=cp.b,
                            c=cp.c,
                            flush_frames=camera_flush_frames,
                        )

            if len(command_sequence) > 1:
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
        )

        print("\n" + "=" * 72)
        print("RUN COMPLETE")
        print("=" * 72)
        print(f"Images saved: {sample_counter}")
        print(f"Raw image folder: {self.raw_image_data_folder}")
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
    if bool(args.use_average_cubic_fit):
        cal = calibration_with_average_cubic_override(cal)

    flip_rz_sign = bool(args.flip_rz_sign)

    b_lo = cal.b_min if args.min_b is None else float(args.min_b)
    b_hi = cal.b_max if args.max_b is None else float(args.max_b)
    if b_lo > b_hi:
        b_lo, b_hi = b_hi, b_lo

    c0_deg = assert_c_in_safe_range("c0_deg", float(args.c0_deg))
    c180_deg = assert_c_in_safe_range(
        "c180_deg",
        float(cal.c_180_deg if args.c180_deg is None else args.c180_deg),
    )

    start_c = assert_c_in_safe_range("start_c", float(args.start_c))
    end_c = assert_c_in_safe_range("end_c", float(args.end_c))

    command_sequence, meta = build_circle_command_sequence(
        cal=cal,
        flip_rz_sign=flip_rz_sign,
        center_x=float(args.center_x),
        center_y=float(args.center_y),
        center_z=float(args.center_z),
        samples_per_quarter=int(args.samples_per_quarter),
        b_lo=b_lo,
        b_hi=b_hi,
        c0_deg=c0_deg,
        c180_deg=c180_deg,
        jog_feed=float(args.jog_feed),
        print_feed_b=float(args.print_feed if args.print_feed_b is None else args.print_feed_b),
        print_feed_c=float(args.print_feed if args.print_feed_c is None else args.print_feed_c),
        transition_feed=float(args.transition_feed),
        quarter_b_override=(None if args.quarter_b is None else float(args.quarter_b)),
        final_recenter=bool(args.final_recenter),
        quarter_gap_mm=float(args.quarter_gap_mm),
    )
    meta["fit_mode"] = "shared_average_cubic" if bool(args.use_average_cubic_fit) else "phase_specific_default"

    print("Trajectory summary:")
    print(f"  fit_mode: {'shared_average_cubic' if bool(args.use_average_cubic_fit) else 'phase_specific_default'}")
    print(f"  quarter_gap_mm: {meta['quarter_gap_mm']:.3f}")
    print(f"  pull_phase: {meta['pull_phase']}")
    print(f"  release_phase: {meta['release_phase']}")
    print(f"  Circle radius used: {meta['circle_radius']:.3f} mm")
    print(f"  Quarter-switch B:   {meta['quarter_b']:.3f}")
    print(f"  flip_rz_sign: {meta['flip_rz_sign']}")
    print(f"  X stage range: [{meta['x_stage_min']:.3f}, {meta['x_stage_max']:.3f}]")
    print(f"  Y stage range: [{meta['y_stage_min']:.3f}, {meta['y_stage_max']:.3f}]")
    print(f"  Z stage range: [{meta['z_stage_min']:.3f}, {meta['z_stage_max']:.3f}]")
    print(f"  C values used: [{meta['c0_deg']:.3f}, {meta['c180_deg']:.3f}]")
    print(
        "  Recorded-quarter stage X variation (should be ~0): "
        + ", ".join(f"{k}={v:.6f}" for k, v in meta["stage_x_variation_by_recorded_quarter"].items())
    )
    print(
        f"  Midpoint pre-flip error before exact C-flip recenter: "
        f"{meta['midpoint_preflip_error']['dist']:.6f} mm"
    )
    print(
        f"  Final closure error before optional recenter: "
        f"{meta['closure_error_before_final']['dist']:.6f} mm"
    )

    start_pose = (
        float(args.start_x),
        float(args.start_y),
        float(args.start_z),
        float(args.start_b),
        float(start_c),
    )
    end_pose = (
        float(args.end_x),
        float(args.end_y),
        float(args.end_z),
        float(args.end_b),
        float(end_c),
    )

    virtual_bbox = {
        "x_min": float(args.bbox_x_min),
        "x_max": float(args.bbox_x_max),
        "y_min": float(args.bbox_y_min),
        "y_max": float(args.bbox_y_max),
        "z_min": float(args.bbox_z_min),
        "z_max": float(args.bbox_z_max),
    }

    runner = CircleTrackerRunner(
        parent_directory=args.parent_directory,
        project_name=args.project_name,
        allow_existing=bool(args.allow_existing),
        add_date=bool(args.add_date),
    )

    try:
        meta_path = save_json(os.path.join(runner.run_folder, "trajectory_meta.json"), meta)
        csv_path = save_command_sequence_csv(
            os.path.join(runner.run_folder, "planned_command_sequence.csv"),
            command_sequence,
        )
        plot_path = save_desired_circle_motion_plot(
            os.path.join(runner.run_folder, "desired_circle_motion.png"),
            command_sequence,
        )
        print(f"Saved plan metadata: {meta_path}")
        print(f"Saved command CSV:   {csv_path}")
        print(f"Saved path plot:     {plot_path}")

        runner.connect_to_camera(
            cam_port=int(args.cam_port),
            show_preview=bool(args.show_preview),
            enable_manual_focus=bool(args.enable_manual_focus),
            manual_focus_val=float(args.manual_focus_val),
            width=int(args.camera_width),
            height=int(args.camera_height),
        )

        runner.connect_to_robot(args.duet_web_address)

        results = runner.execute_circle_motion_and_capture(
            cal=cal,
            command_sequence=command_sequence,
            start_pose=start_pose,
            end_pose=end_pose,
            safe_approach_z=float(args.safe_approach_z),
            travel_feed=float(args.travel_feed),
            fine_approach_feed=float(args.fine_approach_feed),
            virtual_bbox=virtual_bbox,
            dwell_before_ms=int(args.dwell_before_ms),
            dwell_after_ms=int(args.dwell_after_ms),
            initial_sweep_wait_s=float(args.initial_sweep_wait_s),
            tracked_move_settle_s=float(args.tracked_move_settle_s),
            travel_move_settle_s=float(args.travel_move_settle_s),
            b_extra_settle_s=float(args.b_extra_settle_s),
            inter_command_delay_s=float(args.inter_command_delay_s),
            camera_flush_frames=int(args.camera_flush_frames),
            capture_at_start=bool(args.capture_at_start),
            capture_every_n_circle_moves=int(args.capture_every_n_circle_moves),
        )

        print("\nFinal results:")
        print(results)

        if args.enable_post:
            post_camera_calibration = Path(args.post_camera_calibration_file).expanduser().resolve()
            post_reference_image = Path(args.post_checkerboard_reference_image).expanduser().resolve()

            if not post_camera_calibration.is_file():
                raise FileNotFoundError(
                    f"Post-processing camera calibration file not found: {post_camera_calibration}"
                )
            if not post_reference_image.is_file():
                raise FileNotFoundError(
                    f"Post-processing checkerboard reference image not found: {post_reference_image}"
                )

            post_cmd = [
                sys.executable,
                str((Path(__file__).resolve().parent / "calib_circle_process.py").resolve()),
                "--project_dir",
                str(Path(runner.run_folder).resolve()),
                "--camera_calibration_file",
                str(post_camera_calibration),
                "--checkerboard_reference_image",
                str(post_reference_image),
                "--capture_every_n_circle_moves",
                str(int(args.capture_every_n_circle_moves)),
            ]
            if bool(args.capture_at_start):
                post_cmd.append("--capture_at_start")
            if bool(args.post_save_plots):
                post_cmd.append("--save_plots")

            print("\nStarting post-processing:")
            print("  " + " ".join(post_cmd))
            subprocess.run(post_cmd, check=True)

    finally:
        try:
            runner.disconnect_camera()
        except Exception:
            pass
        try:
            runner.disconnect_robot()
        except Exception:
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Run a circle-tracking acquisition directly on the robot using pull/release "
            "PCHIP B calibration plus commanded Z motion, with only transition phases "
            "allowed to change X."
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

    # Kinematic sign override
    ap.add_argument("--flip-rz-sign", action="store_true", default=DEFAULT_FLIP_RZ_SIGN,
                    help="Match the uploaded tracking script: flip only the planar r/X sign from calibration.")
    ap.add_argument("--use-average-cubic-fit", action="store_true", default=DEFAULT_USE_AVERAGE_CUBIC_FIT,
                    help="Override pull/release r/z PCHIP models with one shared average cubic fit built from the phase cubic coefficients.")

    # Circle placement (tip-space)
    ap.add_argument("--center-x", type=float, default=CIRCLE_CENTER_X,
                    help="Nominal circle center X in tip space.")
    ap.add_argument("--center-y", type=float, default=CIRCLE_CENTER_Y,
                    help="Constant circle center Y in tip space. Stage Y is solved to hold this exactly.")
    ap.add_argument("--center-z", type=float, default=CIRCLE_CENTER_Z,
                    help="Circle center Z in tip space.")

    # Circle shape from B curve
    ap.add_argument("--samples-per-quarter", type=int, default=DEFAULT_SAMPLES_PER_QUARTER,
                    help="Interpolation points per quarter-circle segment.")
    ap.add_argument("--quarter-b", type=float, default=None,
                    help=(
                        "Optional B value where curl switches to uncurl. "
                        "Default: auto-pick the pull-curve quarter point inside the common pull/release B window."
                    ))
    ap.add_argument("--quarter-gap-mm", type=float, default=DEFAULT_QUARTER_GAP_MM,
                    help="Untracked straight-line spacing inserted between quarter arcs in tip space.")
    ap.add_argument("--final-recenter", dest="final_recenter", action="store_true", default=DEFAULT_FINAL_RECENTER,
                    help="After Q4, add a final unrecorded recenter to the exact start point at C=180.")
    ap.add_argument("--no-final-recenter", dest="final_recenter", action="store_false",
                    help="Do not add the final exact-bottom recenter after Q4.")

    # Motion / feeds
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED,
                    help="Feedrate for safe travel moves.")
    ap.add_argument("--fine-approach-feed", type=float, default=DEFAULT_FINE_APPROACH_FEED,
                    help="Slow landing move used before the queued circle sweep.")
    ap.add_argument("--jog-feed", type=float, default=DEFAULT_PRINT_FEED,
                    help="Feedrate for startup move to first tracked point.")
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED,
                    help="Base feedrate for drawing moves.")
    ap.add_argument("--print-feed-b", type=float, default=DEFAULT_PRINT_FEED_B,
                    help="Feedrate for recorded B/Z circle moves.")
    ap.add_argument("--print-feed-c", type=float, default=DEFAULT_PRINT_FEED_C,
                    help="Feedrate for the midpoint 180-degree C rotation while tracking.")
    ap.add_argument("--transition-feed", type=float, default=DEFAULT_TRANSITION_FEED,
                    help="Feedrate for pull/release transition moves and the optional final recenter.")

    # B/C overrides
    ap.add_argument("--min-b", type=float, default=None, help="Lower bound for commanded B (default: calibration).")
    ap.add_argument("--max-b", type=float, default=None, help="Upper bound for commanded B (default: calibration).")
    ap.add_argument("--c0-deg", type=float, default=DEFAULT_C0_DEG,
                    help="Fixed C value for the first half.")
    ap.add_argument("--c180-deg", type=float, default=None,
                    help="Fixed C value for the second half (default from calibration).")

    # Optional waits / capture behavior
    ap.add_argument("--dwell-before-ms", type=int, default=DEFAULT_DWELL_BEFORE_MS)
    ap.add_argument("--dwell-after-ms", type=int, default=DEFAULT_DWELL_AFTER_MS)
    ap.add_argument("--initial-sweep-wait-s", type=float, default=DEFAULT_INITIAL_SWEEP_WAIT_S,
                    help="Hold time after landing on the first tracked point before queuing the circle sweep.")
    ap.add_argument("--tracked-move-settle-s", type=float, default=DEFAULT_TRACKED_MOVE_SETTLE_S,
                    help="Extra settle time after the queued tracked move block, before finishing.")
    ap.add_argument("--travel-move-settle-s", type=float, default=DEFAULT_TRAVEL_MOVE_SETTLE_S,
                    help="Extra settle time after travel moves.")
    ap.add_argument("--b-extra-settle-s", type=float, default=DEFAULT_B_EXTRA_SETTLE_S,
                    help="Additional hold after the queued circle motion to let the mechanism settle.")
    ap.add_argument("--inter-command-delay-s", type=float, default=DEFAULT_INTER_COMMAND_DELAY_S,
                    help="Small delay between queued tracked commands.")
    ap.add_argument("--capture-every-n-circle-moves", type=int, default=DEFAULT_CAPTURE_EVERY_N_CIRCLE_MOVES,
                    help="Capture one image every N recorded circle-path moves. Transition moves are never captured.")
    ap.add_argument("--capture-at-start", action="store_true", default=DEFAULT_CAPTURE_AT_START,
                    help="Also capture once at the first tracked sample after positioning there.")
    ap.add_argument("--enable-post", action="store_true", default=DEFAULT_ENABLE_POST,
                    help="Run calib_circle_process.py automatically after acquisition completes.")
    ap.add_argument("--post-camera-calibration-file", default=DEFAULT_POST_CAMERA_CALIBRATION_FILE,
                    help="Camera calibration .npz to pass to post-processing.")
    ap.add_argument("--post-checkerboard-reference-image", default=DEFAULT_POST_CHECKERBOARD_REFERENCE_IMAGE,
                    help="Checkerboard reference image to pass to post-processing.")
    ap.add_argument("--post-save-plots", action="store_true",
                    help="Pass --save_plots to the post-processing script.")

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
