#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone fixed-tip acquisition script with cyclic continuous sweep.

What it does:
- Loads the calibration JSON
- Generates a fixed-tip tracked XYZ/B/C motion
- Connects to the Duet robot and the camera
- Executes the motion directly on the robot
- Captures images only at selected visible phases and saves them to:
    <project folder>/raw_image_data_folder/

Motion model:
- One cycle consists of:
    1) forward leg: C = -360 -> 360
    2) return  leg: C =  360 -> -360
- C moves near constant speed on each leg, with small cosine easing at the
  boundaries / reversals.
- B tip angle oscillates smoothly over the whole cycle with continuous phase.
- You set how many B oscillations happen during ONE one-way sweep using:
      --b-oscillations-per-sweep
  Example:
      --b-oscillations-per-sweep 2
  means:
      - 2 oscillations during -360 -> 360
      - 2 oscillations during 360 -> -360
- Motion sampling and capture sampling are independent:
    * leg_move_steps     = robot motion discretization per leg
    * leg_capture_steps  = image opportunity discretization per leg
- Only capture during visible phases:
    * if -20 <= tip_angle <= 90: capture at all C
    * if tip_angle > 90: capture only when C is visible:
          (-75 < C_wrapped < 75) or (105 < C_wrapped < 255)
      where C_wrapped is modulo 360 in [0, 360)

If requested tip angles fall outside calibration, the script uses the closest
tip angles that do exist in calibration.

Optional sweep-range override:
- Use --b-0-to-90-only to force the cyclic B/tip oscillation to stay within
  0 to 90 degrees.

New sign correction option:
- Use --flip-rz-sign if your calibration file has a flipped planar X sign.
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

SCRIPT_DIR = Path(__file__).resolve().parent

try:
    from duetwebapi import DuetWebAPI
except Exception:
    # Export-only usage does not need a live Duet connection.  Defer the
    # dependency error until connect_to_robot() is actually called.
    DuetWebAPI = None


# =========================
# Defaults
# =========================

DEFAULT_DUET_WEB_ADDRESS = "http://192.168.2.21"
DEFAULT_CAMERA_PORT = 0
DEFAULT_PROJECT_NAME = "Test_Gcode_point_track"
DEFAULT_ALLOW_EXISTING = True
DEFAULT_ADD_DATE = True

DEFAULT_POINT_X = 100.0
DEFAULT_POINT_Y = 50.0
DEFAULT_POINT_Z = -100.0

DEFAULT_TRAVEL_FEED = 2000.0
DEFAULT_FINE_APPROACH_FEED = 1000.0
DEFAULT_PROBE_FEED = 4000.0
DEFAULT_C_FEED = 25000.0
DEFAULT_C_MAX_FEED = 25000.0
DEFAULT_C_ACCEL_TIME_S = 0.1
DEFAULT_C_DECEL_TIME_S = 0.1
DEFAULT_MOTION_MODE = "custom"
DEFAULT_ROUTINE_REPEATS = 1
DEFAULT_MOTION_ACCEL_MM_S2 = 0.0

DEFAULT_CUSTOM_INV_SAMPLES = 20000

DEFAULT_CYCLE_REPEATS = 2
DEFAULT_LEG_MOVE_STEPS = 400
DEFAULT_LEG_CAPTURE_STEPS = 50
DEFAULT_PHASE_TRANSITION_STEPS = 20
DEFAULT_C_TWO_WAY_SWEEPS_PER_B_OSCILLATION = 2

DEFAULT_SWEEP_TIP_MIN_DEG = 0.0
DEFAULT_SWEEP_TIP_MAX_DEG = 180.0
DEFAULT_B_OSCILLATIONS_PER_SWEEP = 1.0
DEFAULT_B_PHASE_OFFSET_DEG = -90.0  # starts at tip_min
DEFAULT_B_0_TO_90_ONLY = False

DEFAULT_CAPTURE_TIP_FULL_VISIBLE_MIN_DEG = -20.0
DEFAULT_CAPTURE_TIP_FULL_VISIBLE_MAX_DEG = 90.0

DEFAULT_C_VISIBLE_WIN1_MIN = -75.0
DEFAULT_C_VISIBLE_WIN1_MAX = 75.0
DEFAULT_C_VISIBLE_WIN2_MIN = 105.0
DEFAULT_C_VISIBLE_WIN2_MAX = 255.0

DEFAULT_C_BOUNDARY_EASE_FRAC = 0.04

DEFAULT_START_X = 100.0
DEFAULT_START_Y = 80.0
DEFAULT_START_Z = 0.0
DEFAULT_START_B = 0.0
DEFAULT_START_C = 0.0

DEFAULT_END_X = 100.0
DEFAULT_END_Y = 80.0
DEFAULT_END_Z = 0.0
DEFAULT_END_B = 0.0
DEFAULT_END_C = 0.0

DEFAULT_SAFE_APPROACH_Z = 0.0

DEFAULT_DWELL_BEFORE_MS = 0.1
DEFAULT_DWELL_AFTER_MS = 0
DEFAULT_INITIAL_SWEEP_WAIT_S = 6.0

DEFAULT_BBOX_X_MIN = 0.0
DEFAULT_BBOX_X_MAX = 180.0
DEFAULT_BBOX_Y_MIN = 0.0
DEFAULT_BBOX_Y_MAX = 170.0
DEFAULT_BBOX_Z_MIN = -200.0
DEFAULT_BBOX_Z_MAX = 0.0

DEFAULT_MANUAL_FOCUS = True
DEFAULT_MANUAL_FOCUS_VAL = 60
DEFAULT_CAMERA_WIDTH = 3840
DEFAULT_CAMERA_HEIGHT = 2160
DEFAULT_CAMERA_FLUSH_FRAMES = 1

DEFAULT_ROTATION_SETTLE_S = 0.0
DEFAULT_TRACKED_MOVE_SETTLE_S = 0.0
DEFAULT_TRAVEL_MOVE_SETTLE_S = 0.0
DEFAULT_B_EXTRA_SETTLE_S = 0.0
DEFAULT_CAPTURE_AT_START = False
DEFAULT_INTER_COMMAND_DELAY_S = 0.002
DEFAULT_SETTLED_CAPTURE_MODE = True
DEFAULT_SETTLED_CAPTURE_BUFFER_S = 0.0
DEFAULT_MIN_ESTIMATED_MOVE_TIME_S = 0.008
DEFAULT_STREAMING_LOOKAHEAD_S = 0.12

DEFAULT_FLIP_RZ_SIGN = True
DEFAULT_Y_OFFSET_FIT = "avg_pchip"
DEFAULT_CURVE_SET = "auto"
DEFAULT_ALLOW_GLOBAL_CURVE_SET_FALLBACK = True
DEFAULT_ENABLE_BRANCH_CONDITIONING = True
DEFAULT_BRANCH_CONDITIONING_TIP_MAX_DEG = 180.0

DEFAULT_POST_CAMERA_CALIBRATION_FILE = str(SCRIPT_DIR / "captures" / "calibration_webcam_20260708_120830.npz")
DEFAULT_POST_CHECKERBOARD_REFERENCE_IMAGE = str(SCRIPT_DIR / "captures" / "photo_20260708_120944.png")
DEFAULT_POST_THRESHOLD = 220
DEFAULT_POST_TIP_REFINE_MODE = "none"
DEFAULT_POST_TIP_DETECTION_MODE = "red_dot"
DEFAULT_POST_TRACKED_TIP_SOURCE = "auto"
DEFAULT_POST_RED_TIP_SAT_MIN = 80
DEFAULT_POST_RED_TIP_VAL_MIN = 80
DEFAULT_POST_RED_TIP_MIN_AREA_PX = 20
DEFAULT_POST_RED_TIP_MORPH_KERNEL = 1
DEFAULT_POST_RED_TIP_HUE1_MIN = 0
DEFAULT_POST_RED_TIP_HUE1_MAX = 10
DEFAULT_POST_RED_TIP_HUE2_MIN = 150
DEFAULT_POST_RED_TIP_HUE2_MAX = 179
DEFAULT_POST_RED_TIP_SEARCH_RADIUS_PX = 320.0
DEFAULT_POST_RED_TIP_LOCAL_MIN_AREA_PX = 10
DEFAULT_POST_RED_TIP_DISTANCE_WEIGHT = 3.0
DEFAULT_POST_RED_TIP_MIN_CIRCULARITY = 0.0
DEFAULT_POST_RED_TIP_COMPONENT_SELECTION = "nearest_largest"
DEFAULT_POST_RED_TIP_USE_RGB_EXCESS = True
DEFAULT_POST_RED_TIP_RGB_EXCESS_MIN = 35
DEFAULT_POST_RED_TIP_DEBUG_SAVE_MASK = True

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
    selected_curve_set: str
    requested_curve_set: str
    available_curve_sets: List[str]
    curve_selection_reason: str
    y_offset_fit: str

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
    cycle_phase_01: Optional[float] = None
    leg_phase_01: Optional[float] = None
    leg_name: Optional[str] = None
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


def _normalize_curve_set_name(value: Any) -> str:
    return str(value).strip().lower().replace("_", "-")


def _curve_set_lookup(curl_payload: Any) -> dict:
    if not isinstance(curl_payload, dict):
        return {}
    out = {}
    for key, payload in curl_payload.items():
        if isinstance(payload, dict):
            out[_normalize_curve_set_name(key)] = (str(key), payload)
    return out


def _extract_selected_phase_payload(
    data: dict,
    curve_set: str,
    allow_global_fallback: bool = DEFAULT_ALLOW_GLOBAL_CURVE_SET_FALLBACK,
    prefer_0_to_90: bool = False,
) -> Tuple[dict, str, List[str], str]:
    """Select the branch-model payload.

    Auto policy used by point tracking:
      - In --b-0-to-90-only mode, prefer curl_angle_specific_fit_models['0-90-0']
        when it exists.
      - Otherwise, or when 0-90-0 is unavailable, use 0-180-0 when it exists.
      - If neither curl-specific set exists, fall back to the top-level/global
        fit_models_by_phase when fallback is allowed.

    This makes the point-tracking script use the shortest valid hysteresis loop
    available in the calibration file, while retaining compatibility with older
    calibration exports that only contain global pull/release branches.
    """
    requested_norm = _normalize_curve_set_name(curve_set)
    curl_payload = data.get("curl_angle_specific_fit_models") or {}
    curve_lookup = _curve_set_lookup(curl_payload)
    available = sorted(str(k) for k in curl_payload.keys()) if isinstance(curl_payload, dict) else []

    def global_payload(reason: str) -> Tuple[dict, str, List[str], str]:
        return {"fit_models_by_phase": data.get("fit_models_by_phase", {})}, "global_fallback", available, reason

    if requested_norm in {"", "auto"}:
        if bool(prefer_0_to_90) and "0-90-0" in curve_lookup:
            selected_key, payload = curve_lookup["0-90-0"]
            return payload, selected_key, available, "auto: selected 0-90-0 because --b-0-to-90-only is active"
        if "0-180-0" in curve_lookup:
            selected_key, payload = curve_lookup["0-180-0"]
            reason = "auto: selected 0-180-0"
            if bool(prefer_0_to_90):
                reason = "auto: 0-90-0 unavailable; selected 0-180-0"
            return payload, selected_key, available, reason
        if bool(allow_global_fallback):
            reason = "auto: no curl-specific 0-90-0/0-180-0 set found; using global fit_models_by_phase"
            return global_payload(reason)
        raise KeyError(
            "Calibration does not contain curl-specific 0-90-0 or 0-180-0 branch models. "
            f"Available curl-specific sets: {available}. "
            "Use --curve-set global or enable global fallback."
        )

    if requested_norm in {"global", "fitmodelsbyphase", "globalfallback"}:
        return {"fit_models_by_phase": data.get("fit_models_by_phase", {})}, "global", available, "explicit: using global fit_models_by_phase"

    # For legacy commands that still say --curve-set 0-180-0 in 0..90-only
    # mode, prefer the dedicated 0-90-0 loop if it exists. This is intentional:
    # a 0-90-0 release branch has the correct hysteretic history for a 90-deg
    # reversal, so no hidden 90->180->90 conditioning is needed.
    if bool(prefer_0_to_90) and requested_norm == "0-180-0" and "0-90-0" in curve_lookup:
        selected_key, payload = curve_lookup["0-90-0"]
        return payload, selected_key, available, "0..90 mode: overriding 0-180-0 request with available 0-90-0 branch set"

    if requested_norm in curve_lookup:
        selected_key, payload = curve_lookup[requested_norm]
        return payload, selected_key, available, f"explicit: selected {selected_key}"

    if bool(allow_global_fallback):
        return global_payload(
            f"requested {curve_set!r} was not found in curl_angle_specific_fit_models; using global fit_models_by_phase"
        )

    raise KeyError(
        f"Calibration does not contain curl_angle_specific_fit_models['{curve_set}']. "
        f"Available curl-specific sets: {available}. "
        "Use --curve-set auto/global, or pass --allow-global-curve-set-fallback."
    )


def _get_phase_model_descriptor(bundle: dict, quantity: str, y_offset_fit: str = DEFAULT_Y_OFFSET_FIT) -> Optional[dict]:
    quantity = str(quantity).strip().lower()
    mode = str(y_offset_fit).strip().lower()

    if quantity == "tip_angle":
        candidates = [
            "tip_angle_pchip",
            "tip_angle",
            "tip_angle_avg_pchip",
            "angle_pchip",
            "angle",
        ]
    elif quantity == "offplane_y":
        if mode == "avg_pchip":
            candidates = ["offplane_y_avg_pchip", "offplane_y_pchip", "offplane_y"]
        elif mode == "avg_cubic":
            candidates = ["offplane_y_avg_cubic", "offplane_y_cubic", "offplane_y"]
        elif mode == "pchip":
            candidates = ["offplane_y_pchip", "offplane_y", "offplane_y_avg_pchip"]
        elif mode == "cubic":
            candidates = ["offplane_y_cubic", "offplane_y", "offplane_y_avg_cubic"]
        elif mode == "legacy":
            candidates = ["offplane_y"]
        else:
            raise ValueError(f"Unsupported y-offset fit selection: {y_offset_fit}")
    else:
        candidates = [
            f"{quantity}_pchip",
            quantity,
            f"{quantity}_avg_pchip",
            f"{quantity}_cubic",
            f"{quantity}_avg_cubic",
        ]

    for key in candidates:
        descriptor = bundle.get(key)
        if isinstance(descriptor, dict):
            return descriptor
    return None


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


def load_calibration(
    json_path: str,
    y_offset_fit: str = DEFAULT_Y_OFFSET_FIT,
    curve_set: str = DEFAULT_CURVE_SET,
    allow_global_curve_set_fallback: bool = DEFAULT_ALLOW_GLOBAL_CURVE_SET_FALLBACK,
    prefer_0_to_90_curve: bool = False,
) -> Calibration:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")

    with p.open("r") as f:
        data = json.load(f)

    selected_phase_payload, selected_curve_set, available_curve_sets, curve_selection_reason = _extract_selected_phase_payload(
        data,
        curve_set=curve_set,
        allow_global_fallback=allow_global_curve_set_fallback,
        prefer_0_to_90=bool(prefer_0_to_90_curve),
    )
    phase_models, default_phase = _extract_phase_models(selected_phase_payload)
    fit_models = data.get("fit_models", {})
    cubic = data.get("cubic_coefficients", {})
    default_phase_models = phase_models.get(default_phase, {})

    r_model = _get_phase_model_descriptor(default_phase_models, "r", y_offset_fit=y_offset_fit) or fit_models.get("r") or legacy_poly_model(
        cubic.get("r_coeffs"),
        cubic.get("r_equation"),
        "r",
    )
    z_model = _get_phase_model_descriptor(default_phase_models, "z", y_offset_fit=y_offset_fit) or fit_models.get("z") or legacy_poly_model(
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
    tip_angle_model = _get_phase_model_descriptor(default_phase_models, "tip_angle", y_offset_fit=y_offset_fit) or fit_models.get("tip_angle") or legacy_poly_model(
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
        selected_curve_set=selected_curve_set,
        requested_curve_set=str(curve_set),
        available_curve_sets=list(available_curve_sets),
        curve_selection_reason=str(curve_selection_reason),
        y_offset_fit=str(y_offset_fit),
        offplane_y_equation=cubic.get("offplane_y_equation"),
        offplane_y_r_squared=(
            None if cubic.get("offplane_y_r_squared") is None
            else float(cubic["offplane_y_r_squared"])
        ),
    )


def _resolve_phase_bundle(cal: Calibration, motion_phase: Optional[str]) -> dict:
    phase_name = _normalize_motion_phase_name(motion_phase) or cal.default_motion_phase
    if phase_name in cal.phase_models:
        return cal.phase_models[phase_name]

    # Calibration exports can name phases either pull/release or pull_1/release_1.
    # Match the requested logical branch first by prefix before falling back.
    phase_prefix = str(phase_name).split("_")[0].split("-")[0]
    for key, bundle in cal.phase_models.items():
        key_prefix = str(key).split("_")[0].split("-")[0]
        if key_prefix == phase_prefix:
            return bundle

    if cal.default_motion_phase in cal.phase_models:
        return cal.phase_models[cal.default_motion_phase]
    return {}


def _select_fit_model(cal: Calibration, model_name: str, motion_phase: Optional[str] = None) -> Any:
    phase_model = _get_phase_model_descriptor(
        _resolve_phase_bundle(cal, motion_phase),
        model_name,
        y_offset_fit=cal.y_offset_fit,
    )
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


def wrap_deg_360(angle_deg: float) -> float:
    return float(angle_deg) % 360.0


def build_tip_angle_inverse_table(
    cal: Calibration,
    num_samples: int = DEFAULT_CUSTOM_INV_SAMPLES,
    motion_phase: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if cal.tip_angle_model is None:
        raise ValueError(
            "This motion mode requires a tip-angle fit model in the calibration JSON."
        )

    ns = max(1000, int(num_samples))
    b_samples = np.linspace(float(cal.b_min), float(cal.b_max), ns, dtype=float)
    angle_samples = eval_tip_angle_deg(cal, b_samples, motion_phase=motion_phase)

    order = np.argsort(angle_samples)
    angle_sorted = np.asarray(angle_samples[order], dtype=float)
    b_sorted = np.asarray(b_samples[order], dtype=float)

    angle_unique, unique_idx = np.unique(angle_sorted, return_index=True)
    b_unique = b_sorted[unique_idx]

    if angle_unique.size < 2:
        raise ValueError("Could not build a usable tip-angle inverse table from calibration.")

    return angle_unique, b_unique


def build_tip_angle_phase_samples(
    cal: Calibration,
    num_samples: int = DEFAULT_CUSTOM_INV_SAMPLES,
    motion_phase: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if cal.tip_angle_model is None:
        raise ValueError(
            "This motion mode requires a tip-angle fit model in the calibration JSON."
        )

    ns = max(1000, int(num_samples))
    b_samples = np.linspace(float(cal.b_min), float(cal.b_max), ns, dtype=float)
    angle_samples = np.asarray(eval_tip_angle_deg(cal, b_samples, motion_phase=motion_phase), dtype=float)
    if b_samples.size < 2 or angle_samples.size < 2:
        raise ValueError("Could not build usable phase tip-angle samples from calibration.")
    return b_samples, angle_samples


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


def tip_angle_deg_to_b_continuous(
    requested_tip_angle_deg: float,
    b_samples: np.ndarray,
    angle_samples_deg: np.ndarray,
    prev_b: Optional[float] = None,
) -> Tuple[float, float]:
    angle_arr = np.asarray(angle_samples_deg, dtype=float).reshape(-1)
    b_arr = np.asarray(b_samples, dtype=float).reshape(-1)
    if angle_arr.size != b_arr.size or angle_arr.size < 2:
        raise ValueError("Continuous tip-angle inversion requires matched dense B/angle samples.")

    finite_mask = np.isfinite(angle_arr) & np.isfinite(b_arr)
    angle_arr = angle_arr[finite_mask]
    b_arr = b_arr[finite_mask]
    if angle_arr.size < 2:
        raise ValueError("Continuous tip-angle inversion has too few finite samples.")

    amin = float(np.nanmin(angle_arr))
    amax = float(np.nanmax(angle_arr))
    used_angle = clamp(float(requested_tip_angle_deg), amin, amax)

    # Fast path for the usual calibration case: tip angle is monotonic with B.
    # This avoids scanning every dense interval for every trajectory point.
    diffs = np.diff(angle_arr)
    tol = 1e-9
    if np.all(diffs >= -tol):
        b_val = float(np.interp(used_angle, angle_arr, b_arr))
        return b_val, used_angle
    if np.all(diffs <= tol):
        b_val = float(np.interp(used_angle, angle_arr[::-1], b_arr[::-1]))
        return b_val, used_angle

    candidates: List[float] = []
    for i in range(len(angle_arr) - 1):
        a0 = float(angle_arr[i])
        a1 = float(angle_arr[i + 1])
        b0 = float(b_arr[i])
        b1 = float(b_arr[i + 1])
        if not (np.isfinite(a0) and np.isfinite(a1) and np.isfinite(b0) and np.isfinite(b1)):
            continue
        lo = min(a0, a1) - tol
        hi = max(a0, a1) + tol
        if used_angle < lo or used_angle > hi:
            continue
        if abs(a1 - a0) <= tol:
            candidates.append(0.5 * (b0 + b1))
            continue
        t = (used_angle - a0) / (a1 - a0)
        if -tol <= t <= 1.0 + tol:
            t = clamp(t, 0.0, 1.0)
            candidates.append((1.0 - t) * b0 + t * b1)

    if not candidates:
        nearest_idx = int(np.nanargmin(np.abs(angle_arr - used_angle)))
        return float(b_arr[nearest_idx]), used_angle

    cand_arr = np.asarray(candidates, dtype=float)
    cand_arr = np.unique(np.round(cand_arr, decimals=9))
    if prev_b is not None and np.isfinite(prev_b):
        pick_idx = int(np.argmin(np.abs(cand_arr - float(prev_b))))
        return float(cand_arr[pick_idx]), used_angle

    nearest_idx = int(np.nanargmin(np.abs(angle_arr - used_angle)))
    seed_b = float(b_arr[nearest_idx])
    pick_idx = int(np.argmin(np.abs(cand_arr - seed_b)))
    return float(cand_arr[pick_idx]), used_angle


def _smooth_cosine_edge_map_01(u: float, edge_frac: float) -> float:
    """
    Monotone map [0,1] -> [0,1] that is nearly linear in the middle and uses
    cosine easing in small edge regions.
    """
    u = clamp(float(u), 0.0, 1.0)
    e = clamp(float(edge_frac), 0.0, 0.49)

    if e <= 1e-12:
        return u

    if u < e:
        t = u / e
        return e * (1.0 - math.cos(0.5 * math.pi * t))

    if u > 1.0 - e:
        t = (u - (1.0 - e)) / e
        return 1.0 - e * math.cos(0.5 * math.pi * t)

    return u


def _c_deg_for_leg_phase(
    leg_phase_01: float,
    c_start_deg: float,
    c_end_deg: float,
    boundary_ease_frac: float,
) -> float:
    s = _smoothstep01(_smooth_cosine_edge_map_01(leg_phase_01, boundary_ease_frac))
    c = (1.0 - s) * float(c_start_deg) + s * float(c_end_deg)
    return clamp_c_bounded(c)


def _tip_angle_leg_deg(
    leg_phase_01: float,
    tip_start_deg: float,
    tip_end_deg: float,
    boundary_ease_frac: float,
) -> float:
    s = _smoothstep01(_smooth_cosine_edge_map_01(leg_phase_01, boundary_ease_frac))
    return float((1.0 - s) * float(tip_start_deg) + s * float(tip_end_deg))


def _tip_angle_monotone_cycle_deg(
    cycle_phase_01: float,
    tip_min_deg: float,
    tip_max_deg: float,
    boundary_ease_frac: float,
) -> float:
    s = _smoothstep01(_smooth_cosine_edge_map_01(cycle_phase_01, boundary_ease_frac))
    return float((1.0 - s) * float(tip_min_deg) + s * float(tip_max_deg))


def _tip_angle_cycle_deg(
    cycle_phase_01: float,
    tip_min_deg: float,
    tip_max_deg: float,
    oscillations_per_cycle: float,
    phase_offset_deg: float,
) -> float:
    center = 0.5 * (float(tip_min_deg) + float(tip_max_deg))
    amp = 0.5 * (float(tip_max_deg) - float(tip_min_deg))
    ph = math.radians(float(phase_offset_deg))
    return float(center + amp * math.sin(2.0 * math.pi * float(oscillations_per_cycle) * float(cycle_phase_01) + ph))


def accel_limited_move_time_seconds(distance_mm: float, feedrate_mm_min: float, accel_mm_s2: float) -> float:
    distance = abs(float(distance_mm))
    v = abs(float(feedrate_mm_min)) / 60.0
    accel = max(0.0, float(accel_mm_s2))
    if distance <= 1e-12 or v <= 1e-12:
        return 0.0
    if accel <= 1e-12:
        return distance / v
    accel_distance = (v ** 2) / (2.0 * accel)
    if distance <= accel_distance:
        return math.sqrt((2.0 * distance) / accel)
    t_accel = v / accel
    return t_accel + ((distance - accel_distance) / v)


def _is_c_visible_high_tip(
    c_deg: float,
    vis1_min_deg: float,
    vis1_max_deg: float,
    vis2_min_deg: float,
    vis2_max_deg: float,
) -> bool:
    cw = wrap_deg_360(c_deg)

    w1_lo = wrap_deg_360(vis1_min_deg)
    w1_hi = wrap_deg_360(vis1_max_deg)
    if w1_lo <= w1_hi:
        in_w1 = (w1_lo < cw < w1_hi)
    else:
        in_w1 = (cw > w1_lo) or (cw < w1_hi)

    w2_lo = wrap_deg_360(vis2_min_deg)
    w2_hi = wrap_deg_360(vis2_max_deg)
    if w2_lo <= w2_hi:
        in_w2 = (w2_lo < cw < w2_hi)
    else:
        in_w2 = (cw > w2_lo) or (cw < w2_hi)

    return bool(in_w1 or in_w2)


def _capture_allowed(
    tip_angle_deg: float,
    c_deg: float,
    tip_full_visible_min_deg: float,
    tip_full_visible_max_deg: float,
    vis1_min_deg: float,
    vis1_max_deg: float,
    vis2_min_deg: float,
    vis2_max_deg: float,
) -> bool:
    tip = float(tip_angle_deg)
    if float(tip_full_visible_min_deg) <= tip <= float(tip_full_visible_max_deg):
        return True
    if tip > float(tip_full_visible_max_deg):
        return _is_c_visible_high_tip(
            c_deg=c_deg,
            vis1_min_deg=vis1_min_deg,
            vis1_max_deg=vis1_max_deg,
            vis2_min_deg=vis2_min_deg,
            vis2_max_deg=vis2_max_deg,
        )
    return False


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


def _apply_motion_phases_and_stage_xyz(
    traj: List[TrajectoryPoint],
    cal: Calibration,
    p_tip_fixed: np.ndarray,
    flip_rz_sign: bool = False,
) -> None:
    for idx, point in enumerate(traj):
        prev_b = None if idx == 0 else traj[idx - 1].b
        next_b = None if idx + 1 >= len(traj) else traj[idx + 1].b
        motion_phase = infer_motion_phase_for_b(
            cal=cal,
            prev_b=prev_b,
            curr_b=point.b,
            next_b=next_b,
        )
        point.motion_phase = motion_phase
        point.stage_xyz = stage_xyz_for_fixed_tip(
            cal=cal,
            p_tip_xyz=p_tip_fixed,
            b=point.b,
            c_deg=point.c,
            flip_rz_sign=flip_rz_sign,
            motion_phase=motion_phase,
        )


# =========================
# Cyclic continuous trajectory
# =========================

def _append_recorded_phase_leg(
    traj: List[TrajectoryPoint],
    cal: Calibration,
    p_tip_fixed: np.ndarray,
    angle_table_deg: np.ndarray,
    b_table: np.ndarray,
    leg_name: str,
    c_start_deg: float,
    c_end_deg: float,
    cycle_phase_start: float,
    cycle_phase_end: float,
    move_steps: int,
    capture_steps: int,
    tip_min_deg: float,
    tip_max_deg: float,
    b_oscillations_per_cycle: float,
    b_phase_offset_deg: float,
    tip_full_visible_min_deg: float,
    tip_full_visible_max_deg: float,
    vis1_min_deg: float,
    vis1_max_deg: float,
    vis2_min_deg: float,
    vis2_max_deg: float,
    boundary_ease_frac: float,
    tip_boundary_ease_frac: float,
    forced_motion_phase: str,
    flip_rz_sign: bool = False,
    include_first_point: bool = False,
    b_samples: Optional[np.ndarray] = None,
    angle_samples_deg: Optional[np.ndarray] = None,
    initial_prev_b: Optional[float] = None,
):
    """
    Append a recorded C/B leg.

    Important fix relative to the old point-tracking script:
    use branch-continuous inversion when dense branch samples are supplied.
    The old sorted angle->B table could choose a different root at the
    pull/release handoff. The circle script avoids that by solving B with the
    previous B as a reference; this function now does the same.
    """
    nmove = max(1, int(move_steps))
    ncap = max(1, int(capture_steps))
    monotone_tip_leg = float(b_oscillations_per_cycle) <= 1.0 + 1e-9

    capture_move_indices: Set[int] = set()

    def tip_for_phases(leg_phase: float, cycle_phase: float) -> float:
        if monotone_tip_leg:
            return _tip_angle_monotone_cycle_deg(
                cycle_phase_01=cycle_phase,
                tip_min_deg=tip_min_deg,
                tip_max_deg=tip_max_deg,
                boundary_ease_frac=tip_boundary_ease_frac,
            )
        tip_phase = _smoothstep01(_smooth_cosine_edge_map_01(leg_phase, tip_boundary_ease_frac))
        cycle_phase_for_tip = (
            (1.0 - tip_phase) * float(cycle_phase_start) + tip_phase * float(cycle_phase_end)
        )
        return _tip_angle_cycle_deg(
            cycle_phase_01=cycle_phase_for_tip,
            tip_min_deg=tip_min_deg,
            tip_max_deg=tip_max_deg,
            oscillations_per_cycle=b_oscillations_per_cycle,
            phase_offset_deg=b_phase_offset_deg,
        )

    for j in range(1, ncap + 1):
        leg_phase = j / float(ncap)
        cycle_phase = (1.0 - leg_phase) * float(cycle_phase_start) + leg_phase * float(cycle_phase_end)
        c_cmd = _c_deg_for_leg_phase(
            leg_phase_01=leg_phase,
            c_start_deg=c_start_deg,
            c_end_deg=c_end_deg,
            boundary_ease_frac=boundary_ease_frac,
        )
        req_tip = tip_for_phases(leg_phase, cycle_phase)
        if _capture_allowed(
            tip_angle_deg=req_tip,
            c_deg=c_cmd,
            tip_full_visible_min_deg=tip_full_visible_min_deg,
            tip_full_visible_max_deg=tip_full_visible_max_deg,
            vis1_min_deg=vis1_min_deg,
            vis1_max_deg=vis1_max_deg,
            vis2_min_deg=vis2_min_deg,
            vis2_max_deg=vis2_max_deg,
        ):
            idx = int(round(leg_phase * nmove))
            idx = max(1, min(nmove, idx))
            capture_move_indices.add(idx)

    prev_b = initial_prev_b
    if prev_b is None and traj:
        prev_b = float(traj[-1].b)

    i_start = 0 if include_first_point else 1
    for i in range(i_start, nmove + 1):
        leg_phase = i / float(nmove)
        cycle_phase = (1.0 - leg_phase) * float(cycle_phase_start) + leg_phase * float(cycle_phase_end)
        req_tip = tip_for_phases(leg_phase, cycle_phase)

        if b_samples is not None and angle_samples_deg is not None:
            b_cmd, used_tip = tip_angle_deg_to_b_continuous(
                requested_tip_angle_deg=req_tip,
                b_samples=b_samples,
                angle_samples_deg=angle_samples_deg,
                prev_b=prev_b,
            )
        else:
            b_cmd, used_tip = tip_angle_deg_to_b_clipped(
                requested_tip_angle_deg=req_tip,
                angle_table_deg=angle_table_deg,
                b_table=b_table,
            )
        prev_b = float(b_cmd)

        c_cmd = _c_deg_for_leg_phase(
            leg_phase_01=leg_phase,
            c_start_deg=c_start_deg,
            c_end_deg=c_end_deg,
            boundary_ease_frac=boundary_ease_frac,
        )
        stage_xyz = stage_xyz_for_fixed_tip(
            cal=cal,
            p_tip_xyz=p_tip_fixed,
            b=float(b_cmd),
            c_deg=float(c_cmd),
            flip_rz_sign=flip_rz_sign,
            motion_phase=forced_motion_phase,
        )
        traj.append(
            TrajectoryPoint(
                b=float(b_cmd),
                c=float(c_cmd),
                stage_xyz=stage_xyz,
                segment_kind="cycle",
                capture_image=bool(i in capture_move_indices),
                tip_angle_deg=float(used_tip),
                cycle_phase_01=float(cycle_phase),
                leg_phase_01=float(leg_phase),
                leg_name=str(leg_name),
                motion_phase=str(forced_motion_phase),
            )
        )

def _append_phase_reposition_point(
    traj: List[TrajectoryPoint],
    cal: Calibration,
    p_tip_fixed: np.ndarray,
    b: float,
    c: float,
    leg_name: str,
    motion_phase: str,
    flip_rz_sign: bool = False,
):
    stage_xyz = stage_xyz_for_fixed_tip(
        cal=cal,
        p_tip_xyz=p_tip_fixed,
        b=float(b),
        c_deg=float(c),
        flip_rz_sign=flip_rz_sign,
        motion_phase=motion_phase,
    )
    tip_angle_deg = float(eval_tip_angle_deg(cal, float(b), motion_phase=motion_phase))
    traj.append(
        TrajectoryPoint(
            b=float(b),
            c=float(c),
            stage_xyz=stage_xyz,
            segment_kind="phase_reposition",
            capture_image=False,
            tip_angle_deg=tip_angle_deg,
            cycle_phase_01=None,
            leg_phase_01=None,
            leg_name=str(leg_name),
            motion_phase=str(motion_phase),
        )
    )


def _append_phase_transition_ramp(
    traj: List[TrajectoryPoint],
    cal: Calibration,
    p_tip_fixed: np.ndarray,
    b_start: float,
    b_end: float,
    c: float,
    leg_name: str,
    motion_phase: str,
    steps: int,
    stage_xyz_start: Optional[np.ndarray] = None,
    stage_xyz_end: Optional[np.ndarray] = None,
    tip_angle_start_deg: Optional[float] = None,
    tip_angle_end_deg: Optional[float] = None,
    flip_rz_sign: bool = False,
):
    nsteps = max(1, int(steps))
    stage_start = None if stage_xyz_start is None else np.asarray(stage_xyz_start, dtype=float)
    stage_end = None if stage_xyz_end is None else np.asarray(stage_xyz_end, dtype=float)
    for i in range(1, nsteps + 1):
        s = _smoothstep01(i / float(nsteps))
        b_i = (1.0 - s) * float(b_start) + s * float(b_end)
        if stage_start is not None and stage_end is not None:
            stage_xyz = (1.0 - s) * stage_start + s * stage_end
        else:
            stage_xyz = stage_xyz_for_fixed_tip(
                cal=cal,
                p_tip_xyz=p_tip_fixed,
                b=float(b_i),
                c_deg=float(c),
                flip_rz_sign=flip_rz_sign,
                motion_phase=motion_phase,
            )
        if tip_angle_start_deg is not None and tip_angle_end_deg is not None:
            tip_angle_deg = (1.0 - s) * float(tip_angle_start_deg) + s * float(tip_angle_end_deg)
        else:
            tip_angle_deg = float(eval_tip_angle_deg(cal, float(b_i), motion_phase=motion_phase))
        traj.append(
            TrajectoryPoint(
                b=float(b_i),
                c=float(c),
                stage_xyz=stage_xyz,
                segment_kind="phase_transition",
                capture_image=False,
                tip_angle_deg=tip_angle_deg,
                cycle_phase_01=None,
                leg_phase_01=None,
                leg_name=str(leg_name),
                motion_phase=str(motion_phase),
            )
        )


def _append_unrecorded_tip_angle_ramp(
    traj: List[TrajectoryPoint],
    cal: Calibration,
    p_tip_fixed: np.ndarray,
    b_samples: np.ndarray,
    angle_samples_deg: np.ndarray,
    tip_start_deg: float,
    tip_end_deg: float,
    c_deg: float,
    leg_name: str,
    motion_phase: str,
    steps: int,
    boundary_ease_frac: float = DEFAULT_C_BOUNDARY_EASE_FRAC,
    flip_rz_sign: bool = False,
    initial_prev_b: Optional[float] = None,
    include_first_point: bool = False,
):
    """
    Hidden fixed-tip angle ramp used to respect hysteretic branch history.

    Example for --curve-set 0-180-0 with --b-0-to-90-only:
      visible pull:    0 -> 90
      hidden pull:    90 -> 180
      branch switch at the calibrated endpoint
      hidden release: 180 -> 90
      visible release:90 -> 0

    This matches the circle script's branch history: branch switches occur at
    the calibrated endpoints, not at a truncated visible angle.
    """
    nsteps = max(1, int(steps))
    prev_b = initial_prev_b
    if prev_b is None and traj:
        prev_b = float(traj[-1].b)

    i_start = 0 if include_first_point else 1
    for i in range(i_start, nsteps + 1):
        u = i / float(nsteps)
        s = _smoothstep01(_smooth_cosine_edge_map_01(u, boundary_ease_frac))
        req_tip = (1.0 - s) * float(tip_start_deg) + s * float(tip_end_deg)
        b_cmd, used_tip = tip_angle_deg_to_b_continuous(
            requested_tip_angle_deg=req_tip,
            b_samples=b_samples,
            angle_samples_deg=angle_samples_deg,
            prev_b=prev_b,
        )
        prev_b = float(b_cmd)
        stage_xyz = stage_xyz_for_fixed_tip(
            cal=cal,
            p_tip_xyz=p_tip_fixed,
            b=float(b_cmd),
            c_deg=float(c_deg),
            flip_rz_sign=flip_rz_sign,
            motion_phase=motion_phase,
        )
        traj.append(
            TrajectoryPoint(
                b=float(b_cmd),
                c=float(c_deg),
                stage_xyz=stage_xyz,
                segment_kind="branch_conditioning",
                capture_image=False,
                tip_angle_deg=float(used_tip),
                cycle_phase_01=None,
                leg_phase_01=float(u),
                leg_name=str(leg_name),
                motion_phase=str(motion_phase),
            )
        )


def _append_multi_sweep_monotone_leg(
    traj: List[TrajectoryPoint],
    cal: Calibration,
    p_tip_fixed: np.ndarray,
    angle_table_deg: np.ndarray,
    b_table: np.ndarray,
    leg_name: str,
    c_first_deg: float,
    c_second_deg: float,
    cycle_phase_start: float,
    cycle_phase_end: float,
    move_steps_per_sweep: int,
    capture_steps_per_sweep: int,
    tip_min_deg: float,
    tip_max_deg: float,
    tip_full_visible_min_deg: float,
    tip_full_visible_max_deg: float,
    vis1_min_deg: float,
    vis1_max_deg: float,
    vis2_min_deg: float,
    vis2_max_deg: float,
    boundary_ease_frac: float,
    tip_boundary_ease_frac: float,
    forced_motion_phase: str,
    c_two_way_sweeps: int,
    flip_rz_sign: bool = False,
    include_first_point: bool = False,
    b_samples: Optional[np.ndarray] = None,
    angle_samples_deg: Optional[np.ndarray] = None,
    initial_prev_b: Optional[float] = None,
):
    n_two_way = max(1, int(c_two_way_sweeps))
    n_sublegs = max(1, 2 * n_two_way - 1)
    dc = float(cycle_phase_end) - float(cycle_phase_start)

    for subleg_idx in range(n_sublegs):
        leg_c_start = float(c_first_deg if (subleg_idx % 2 == 0) else c_second_deg)
        leg_c_end = float(c_second_deg if (subleg_idx % 2 == 0) else c_first_deg)
        sub_phase_start = float(cycle_phase_start) + dc * (float(subleg_idx) / float(n_sublegs))
        sub_phase_end = float(cycle_phase_start) + dc * (float(subleg_idx + 1) / float(n_sublegs))
        _append_recorded_phase_leg(
            traj=traj,
            cal=cal,
            p_tip_fixed=p_tip_fixed,
            angle_table_deg=angle_table_deg,
            b_table=b_table,
            leg_name=leg_name,
            c_start_deg=leg_c_start,
            c_end_deg=leg_c_end,
            cycle_phase_start=sub_phase_start,
            cycle_phase_end=sub_phase_end,
            move_steps=int(move_steps_per_sweep),
            capture_steps=int(capture_steps_per_sweep),
            tip_min_deg=float(tip_min_deg),
            tip_max_deg=float(tip_max_deg),
            b_oscillations_per_cycle=1.0,
            b_phase_offset_deg=-90.0,
            tip_full_visible_min_deg=float(tip_full_visible_min_deg),
            tip_full_visible_max_deg=float(tip_full_visible_max_deg),
            vis1_min_deg=float(vis1_min_deg),
            vis1_max_deg=float(vis1_max_deg),
            vis2_min_deg=float(vis2_min_deg),
            vis2_max_deg=float(vis2_max_deg),
            boundary_ease_frac=float(boundary_ease_frac),
            tip_boundary_ease_frac=float(tip_boundary_ease_frac),
            forced_motion_phase=str(forced_motion_phase),
            flip_rz_sign=flip_rz_sign,
            include_first_point=bool(include_first_point and subleg_idx == 0),
            b_samples=b_samples,
            angle_samples_deg=angle_samples_deg,
            initial_prev_b=(initial_prev_b if subleg_idx == 0 else None),
        )


def generate_cyclic_visibility_gated_trajectory(
    cal: Calibration,
    p_tip_fixed: np.ndarray,
    repeats: int = DEFAULT_CYCLE_REPEATS,
    leg_move_steps: int = DEFAULT_LEG_MOVE_STEPS,
    leg_capture_steps: int = DEFAULT_LEG_CAPTURE_STEPS,
    tip_min_deg: float = DEFAULT_SWEEP_TIP_MIN_DEG,
    tip_max_deg: float = DEFAULT_SWEEP_TIP_MAX_DEG,
    b_oscillations_per_sweep: float = DEFAULT_B_OSCILLATIONS_PER_SWEEP,
    b_phase_offset_deg: float = DEFAULT_B_PHASE_OFFSET_DEG,
    tip_full_visible_min_deg: float = DEFAULT_CAPTURE_TIP_FULL_VISIBLE_MIN_DEG,
    tip_full_visible_max_deg: float = DEFAULT_CAPTURE_TIP_FULL_VISIBLE_MAX_DEG,
    vis1_min_deg: float = DEFAULT_C_VISIBLE_WIN1_MIN,
    vis1_max_deg: float = DEFAULT_C_VISIBLE_WIN1_MAX,
    vis2_min_deg: float = DEFAULT_C_VISIBLE_WIN2_MIN,
    vis2_max_deg: float = DEFAULT_C_VISIBLE_WIN2_MAX,
    boundary_ease_frac: float = DEFAULT_C_BOUNDARY_EASE_FRAC,
    inverse_samples: int = DEFAULT_CUSTOM_INV_SAMPLES,
    phase_transition_steps: int = DEFAULT_PHASE_TRANSITION_STEPS,
    c_two_way_sweeps_per_b_oscillation: int = DEFAULT_C_TWO_WAY_SWEEPS_PER_B_OSCILLATION,
    flip_rz_sign: bool = False,
    enable_branch_conditioning: bool = DEFAULT_ENABLE_BRANCH_CONDITIONING,
    branch_conditioning_tip_max_deg: float = DEFAULT_BRANCH_CONDITIONING_TIP_MAX_DEG,
) -> Tuple[List[TrajectoryPoint], dict]:
    """
    Generate the fixed-tip cyclic point-tracking trajectory.

    Fixes compared with the old version:
    1) B is solved with previous-B continuity, matching the circle script's
       solve-with-reference behavior.
    2) If a truncated visible range such as 0..90 is requested while using a
       0-180-0 branch calibration, the robot still completes the hidden
       90->180 pull and 180->90 release conditioning at fixed tip/C before
       recording the visible release. This avoids reversing hysteresis at 90
       while applying a release model calibrated after 180.
    3) --phase-transition-steps is honored for monotone branch switches too.
    """
    pull_b_dense, pull_angle_dense = build_tip_angle_phase_samples(
        cal=cal,
        num_samples=int(inverse_samples),
        motion_phase="pull",
    )
    release_b_dense, release_angle_dense = build_tip_angle_phase_samples(
        cal=cal,
        num_samples=int(inverse_samples),
        motion_phase="release",
    )

    pull_angle_table_deg, pull_b_table = build_tip_angle_inverse_table(
        cal=cal,
        num_samples=int(inverse_samples),
        motion_phase="pull",
    )
    release_angle_table_deg, release_b_table = build_tip_angle_inverse_table(
        cal=cal,
        num_samples=int(inverse_samples),
        motion_phase="release",
    )

    available_tip_min = max(float(pull_angle_table_deg[0]), float(release_angle_table_deg[0]))
    available_tip_max = min(float(pull_angle_table_deg[-1]), float(release_angle_table_deg[-1]))
    if available_tip_min >= available_tip_max:
        raise ValueError("Pull/release tip-angle ranges do not overlap enough to build a cyclic trajectory.")

    used_tip_min = clamp(float(tip_min_deg), available_tip_min, available_tip_max)
    used_tip_max = clamp(float(tip_max_deg), available_tip_min, available_tip_max)
    if used_tip_min > used_tip_max:
        used_tip_min, used_tip_max = used_tip_max, used_tip_min

    b_oscillations_per_cycle = float(b_oscillations_per_sweep)
    monotone_multi_sweep = float(b_oscillations_per_cycle) <= 1.0 + 1e-9

    curve_set_norm = _normalize_curve_set_name(cal.selected_curve_set)
    requested_condition_tip_max = clamp(
        float(branch_conditioning_tip_max_deg),
        available_tip_min,
        available_tip_max,
    )
    uses_dedicated_0_90_loop = ("0-90-0" in curve_set_norm)
    branch_conditioning_active = (
        bool(enable_branch_conditioning)
        and monotone_multi_sweep
        and (not uses_dedicated_0_90_loop)
        and (requested_condition_tip_max > used_tip_max + 1e-6)
    )
    conditioning_tip_max = requested_condition_tip_max if branch_conditioning_active else used_tip_max
    conditioning_steps = max(2, int(phase_transition_steps))

    traj: List[TrajectoryPoint] = []

    # Exact first point: start of curl cycle at C = -360 using the pull branch.
    b0, used_tip0 = tip_angle_deg_to_b_continuous(
        requested_tip_angle_deg=used_tip_min,
        b_samples=pull_b_dense,
        angle_samples_deg=pull_angle_dense,
        prev_b=0.0,
    )
    c0 = -360.0
    stage_xyz0 = stage_xyz_for_fixed_tip(
        cal=cal,
        p_tip_xyz=p_tip_fixed,
        b=float(b0),
        c_deg=float(c0),
        flip_rz_sign=flip_rz_sign,
        motion_phase="pull",
    )
    traj.append(
        TrajectoryPoint(
            b=float(b0),
            c=float(c0),
            stage_xyz=stage_xyz0,
            segment_kind="start",
            capture_image=False,
            tip_angle_deg=float(used_tip0),
            cycle_phase_01=0.0,
            leg_phase_01=0.0,
            leg_name="curl_record",
            motion_phase="pull",
        )
    )

    def append_branch_switch(
        target_phase: str,
        target_b: float,
        target_tip: float,
        c_deg: float,
        leg_name: str,
    ) -> None:
        if not traj:
            return
        if int(phase_transition_steps) <= 0:
            return
        start_point = traj[-1]
        target_stage = stage_xyz_for_fixed_tip(
            cal=cal,
            p_tip_xyz=p_tip_fixed,
            b=float(target_b),
            c_deg=float(c_deg),
            flip_rz_sign=flip_rz_sign,
            motion_phase=target_phase,
        )
        _append_phase_transition_ramp(
            traj=traj,
            cal=cal,
            p_tip_fixed=p_tip_fixed,
            b_start=float(start_point.b),
            b_end=float(target_b),
            c=float(c_deg),
            leg_name=str(leg_name),
            motion_phase=str(target_phase),
            steps=int(phase_transition_steps),
            stage_xyz_start=np.asarray(start_point.stage_xyz, dtype=float),
            stage_xyz_end=target_stage,
            tip_angle_start_deg=float(start_point.tip_angle_deg if start_point.tip_angle_deg is not None else target_tip),
            tip_angle_end_deg=float(target_tip),
            flip_rz_sign=flip_rz_sign,
        )

    for rep in range(max(1, int(repeats))):
        # Visible/recorded pull portion.
        if monotone_multi_sweep:
            _append_multi_sweep_monotone_leg(
                traj=traj,
                cal=cal,
                p_tip_fixed=p_tip_fixed,
                angle_table_deg=pull_angle_table_deg,
                b_table=pull_b_table,
                leg_name="curl_record",
                c_first_deg=-360.0,
                c_second_deg=360.0,
                cycle_phase_start=0.0,
                cycle_phase_end=1.0,
                move_steps_per_sweep=int(leg_move_steps),
                capture_steps_per_sweep=int(leg_capture_steps),
                tip_min_deg=float(used_tip_min),
                tip_max_deg=float(used_tip_max),
                tip_full_visible_min_deg=float(tip_full_visible_min_deg),
                tip_full_visible_max_deg=float(tip_full_visible_max_deg),
                vis1_min_deg=float(vis1_min_deg),
                vis1_max_deg=float(vis1_max_deg),
                vis2_min_deg=float(vis2_min_deg),
                vis2_max_deg=float(vis2_max_deg),
                boundary_ease_frac=float(boundary_ease_frac),
                tip_boundary_ease_frac=float(boundary_ease_frac),
                forced_motion_phase="pull",
                c_two_way_sweeps=int(c_two_way_sweeps_per_b_oscillation),
                flip_rz_sign=flip_rz_sign,
                include_first_point=False,
                b_samples=pull_b_dense,
                angle_samples_deg=pull_angle_dense,
            )
        else:
            _append_recorded_phase_leg(
                traj=traj,
                cal=cal,
                p_tip_fixed=p_tip_fixed,
                angle_table_deg=pull_angle_table_deg,
                b_table=pull_b_table,
                leg_name="curl_record",
                c_start_deg=-360.0,
                c_end_deg=360.0,
                cycle_phase_start=0.0,
                cycle_phase_end=1.0,
                move_steps=int(leg_move_steps),
                capture_steps=int(leg_capture_steps),
                tip_min_deg=float(used_tip_min),
                tip_max_deg=float(used_tip_max),
                b_oscillations_per_cycle=b_oscillations_per_cycle,
                b_phase_offset_deg=-90.0,
                tip_full_visible_min_deg=float(tip_full_visible_min_deg),
                tip_full_visible_max_deg=float(tip_full_visible_max_deg),
                vis1_min_deg=float(vis1_min_deg),
                vis1_max_deg=float(vis1_max_deg),
                vis2_min_deg=float(vis2_min_deg),
                vis2_max_deg=float(vis2_max_deg),
                boundary_ease_frac=float(boundary_ease_frac),
                tip_boundary_ease_frac=float(boundary_ease_frac),
                forced_motion_phase="pull",
                flip_rz_sign=flip_rz_sign,
                include_first_point=False,
                b_samples=pull_b_dense,
                angle_samples_deg=pull_angle_dense,
            )

        # If visible max is truncated (e.g. 90) but the selected calibration is
        # 0-180-0, first complete the hidden pull to 180 before switching to release.
        if branch_conditioning_active:
            _append_unrecorded_tip_angle_ramp(
                traj=traj,
                cal=cal,
                p_tip_fixed=p_tip_fixed,
                b_samples=pull_b_dense,
                angle_samples_deg=pull_angle_dense,
                tip_start_deg=float(used_tip_max),
                tip_end_deg=float(conditioning_tip_max),
                c_deg=360.0,
                leg_name="hidden_pull_to_branch_endpoint",
                motion_phase="pull",
                steps=conditioning_steps,
                boundary_ease_frac=float(boundary_ease_frac),
                flip_rz_sign=flip_rz_sign,
            )

        pull_end = traj[-1]
        release_b_start, release_tip_start = tip_angle_deg_to_b_continuous(
            requested_tip_angle_deg=float(conditioning_tip_max),
            b_samples=release_b_dense,
            angle_samples_deg=release_angle_dense,
            prev_b=float(pull_end.b),
        )
        append_branch_switch(
            target_phase="release",
            target_b=float(release_b_start),
            target_tip=float(release_tip_start),
            c_deg=360.0,
            leg_name="to_uncurl_transition",
        )

        if branch_conditioning_active:
            _append_unrecorded_tip_angle_ramp(
                traj=traj,
                cal=cal,
                p_tip_fixed=p_tip_fixed,
                b_samples=release_b_dense,
                angle_samples_deg=release_angle_dense,
                tip_start_deg=float(conditioning_tip_max),
                tip_end_deg=float(used_tip_max),
                c_deg=360.0,
                leg_name="hidden_release_from_branch_endpoint",
                motion_phase="release",
                steps=conditioning_steps,
                boundary_ease_frac=float(boundary_ease_frac),
                flip_rz_sign=flip_rz_sign,
                include_first_point=(int(phase_transition_steps) <= 0),
                initial_prev_b=float(release_b_start),
            )

        # Visible/recorded release portion.
        if monotone_multi_sweep:
            _append_multi_sweep_monotone_leg(
                traj=traj,
                cal=cal,
                p_tip_fixed=p_tip_fixed,
                angle_table_deg=release_angle_table_deg,
                b_table=release_b_table,
                leg_name="uncurl_record",
                c_first_deg=360.0,
                c_second_deg=-360.0,
                cycle_phase_start=1.0,
                cycle_phase_end=0.0,
                move_steps_per_sweep=int(leg_move_steps),
                capture_steps_per_sweep=int(leg_capture_steps),
                tip_min_deg=float(used_tip_min),
                tip_max_deg=float(used_tip_max),
                tip_full_visible_min_deg=float(tip_full_visible_min_deg),
                tip_full_visible_max_deg=float(tip_full_visible_max_deg),
                vis1_min_deg=float(vis1_min_deg),
                vis1_max_deg=float(vis1_max_deg),
                vis2_min_deg=float(vis2_min_deg),
                vis2_max_deg=float(vis2_max_deg),
                boundary_ease_frac=float(boundary_ease_frac),
                tip_boundary_ease_frac=float(boundary_ease_frac),
                forced_motion_phase="release",
                c_two_way_sweeps=int(c_two_way_sweeps_per_b_oscillation),
                flip_rz_sign=flip_rz_sign,
                include_first_point=(not branch_conditioning_active and int(phase_transition_steps) <= 0),
                b_samples=release_b_dense,
                angle_samples_deg=release_angle_dense,
                initial_prev_b=float(release_b_start),
            )
        else:
            _append_recorded_phase_leg(
                traj=traj,
                cal=cal,
                p_tip_fixed=p_tip_fixed,
                angle_table_deg=release_angle_table_deg,
                b_table=release_b_table,
                leg_name="uncurl_record",
                c_start_deg=360.0,
                c_end_deg=-360.0,
                cycle_phase_start=1.0,
                cycle_phase_end=0.0,
                move_steps=int(leg_move_steps),
                capture_steps=int(leg_capture_steps),
                tip_min_deg=float(used_tip_min),
                tip_max_deg=float(used_tip_max),
                b_oscillations_per_cycle=b_oscillations_per_cycle,
                b_phase_offset_deg=-90.0,
                tip_full_visible_min_deg=float(tip_full_visible_min_deg),
                tip_full_visible_max_deg=float(tip_full_visible_max_deg),
                vis1_min_deg=float(vis1_min_deg),
                vis1_max_deg=float(vis1_max_deg),
                vis2_min_deg=float(vis2_min_deg),
                vis2_max_deg=float(vis2_max_deg),
                boundary_ease_frac=float(boundary_ease_frac),
                tip_boundary_ease_frac=float(boundary_ease_frac),
                forced_motion_phase="release",
                flip_rz_sign=flip_rz_sign,
                include_first_point=(int(phase_transition_steps) <= 0 and not branch_conditioning_active),
                b_samples=release_b_dense,
                angle_samples_deg=release_angle_dense,
                initial_prev_b=float(release_b_start),
            )

        # Prepare the next repeat by switching release->pull at the lower endpoint.
        if rep + 1 < max(1, int(repeats)):
            release_end = traj[-1]
            pull_b_start, pull_tip_start = tip_angle_deg_to_b_continuous(
                requested_tip_angle_deg=float(used_tip_min),
                b_samples=pull_b_dense,
                angle_samples_deg=pull_angle_dense,
                prev_b=float(release_end.b),
            )
            append_branch_switch(
                target_phase="pull",
                target_b=float(pull_b_start),
                target_tip=float(pull_tip_start),
                c_deg=-360.0,
                leg_name="to_curl_transition",
            )

    n_captures = int(sum(1 for pt in traj if pt.capture_image))
    meta = {
        "requested_tip_min_deg": float(tip_min_deg),
        "requested_tip_max_deg": float(tip_max_deg),
        "used_tip_min_deg": float(used_tip_min),
        "used_tip_max_deg": float(used_tip_max),
        "branch_conditioning_active": bool(branch_conditioning_active),
        "uses_dedicated_0_90_loop": bool(uses_dedicated_0_90_loop),
        "branch_conditioning_tip_max_deg": float(conditioning_tip_max),
        "available_tip_angle_range_deg": [available_tip_min, available_tip_max],
        "leg_move_steps": int(leg_move_steps),
        "leg_capture_steps": int(leg_capture_steps),
        "boundary_ease_frac": float(boundary_ease_frac),
        "b_oscillations_per_sweep": float(b_oscillations_per_sweep),
        "b_oscillations_per_cycle": float(b_oscillations_per_cycle),
        "c_two_way_sweeps_per_b_oscillation": int(c_two_way_sweeps_per_b_oscillation),
        "b_phase_offset_deg": float(b_phase_offset_deg),
        "capture_tip_full_visible_min_deg": float(tip_full_visible_min_deg),
        "capture_tip_full_visible_max_deg": float(tip_full_visible_max_deg),
        "visible_c_windows_deg": [
            [float(vis1_min_deg), float(vis1_max_deg)],
            [float(vis2_min_deg), float(vis2_max_deg)],
        ],
        "planned_capture_points": n_captures,
        "flip_rz_sign": bool(flip_rz_sign),
        "phase_sequence": ["pull_record", "hidden_pull_to_branch_endpoint", "release_record" if not branch_conditioning_active else "hidden_release_from_branch_endpoint", "release_record"],
        "phase_transition_steps": int(phase_transition_steps),
        "phase_transition_mode": "continuous_reference_b_with_endpoint_conditioning" if branch_conditioning_active else "continuous_reference_b_blended_ramp",
    }
    return traj, meta

def generate_fast_routine_trajectory(
    cal: Calibration,
    p_tip_fixed: np.ndarray,
    repeats: int = DEFAULT_ROUTINE_REPEATS,
    move_steps_per_routine: int = DEFAULT_LEG_MOVE_STEPS * 2,
    capture_steps_per_routine: int = DEFAULT_LEG_CAPTURE_STEPS * 2,
    tip_min_deg: float = DEFAULT_SWEEP_TIP_MIN_DEG,
    tip_max_deg: float = DEFAULT_SWEEP_TIP_MAX_DEG,
    tip_full_visible_min_deg: float = DEFAULT_CAPTURE_TIP_FULL_VISIBLE_MIN_DEG,
    tip_full_visible_max_deg: float = DEFAULT_CAPTURE_TIP_FULL_VISIBLE_MAX_DEG,
    vis1_min_deg: float = DEFAULT_C_VISIBLE_WIN1_MIN,
    vis1_max_deg: float = DEFAULT_C_VISIBLE_WIN1_MAX,
    vis2_min_deg: float = DEFAULT_C_VISIBLE_WIN2_MIN,
    vis2_max_deg: float = DEFAULT_C_VISIBLE_WIN2_MAX,
    inverse_samples: int = DEFAULT_CUSTOM_INV_SAMPLES,
    flip_rz_sign: bool = False,
) -> Tuple[List[TrajectoryPoint], dict]:
    c_cycles_per_routine = 5.0
    b_cycles_per_routine = 3.0
    c_center_deg = 0.0
    c_amp_deg = 360.0
    b_phase_offset_deg = -90.0

    pull_b_dense, pull_angle_dense = build_tip_angle_phase_samples(
        cal=cal,
        num_samples=int(inverse_samples),
        motion_phase="pull",
    )
    release_b_dense, release_angle_dense = build_tip_angle_phase_samples(
        cal=cal,
        num_samples=int(inverse_samples),
        motion_phase="release",
    )

    pull_angle_table_deg, pull_b_table = build_tip_angle_inverse_table(
        cal=cal,
        num_samples=int(inverse_samples),
        motion_phase="pull",
    )
    release_angle_table_deg, release_b_table = build_tip_angle_inverse_table(
        cal=cal,
        num_samples=int(inverse_samples),
        motion_phase="release",
    )

    available_tip_min = max(float(pull_angle_table_deg[0]), float(release_angle_table_deg[0]))
    available_tip_max = min(float(pull_angle_table_deg[-1]), float(release_angle_table_deg[-1]))
    if available_tip_min >= available_tip_max:
        raise ValueError("Pull/release tip-angle ranges do not overlap enough to build a fast routine trajectory.")

    used_tip_min = clamp(float(tip_min_deg), available_tip_min, available_tip_max)
    used_tip_max = clamp(float(tip_max_deg), available_tip_min, available_tip_max)
    if used_tip_min > used_tip_max:
        used_tip_min, used_tip_max = used_tip_max, used_tip_min

    nmove = max(8, int(move_steps_per_routine))
    ncap = max(4, int(capture_steps_per_routine))
    capture_indices: Set[Tuple[int, int]] = set()
    repeats = max(1, int(repeats))

    def c_for_phase(t01: float) -> float:
        return float(c_center_deg + c_amp_deg * math.sin(2.0 * math.pi * c_cycles_per_routine * float(t01)))

    def tip_for_phase(t01: float) -> float:
        return _tip_angle_cycle_deg(
            cycle_phase_01=float(t01),
            tip_min_deg=used_tip_min,
            tip_max_deg=used_tip_max,
            oscillations_per_cycle=b_cycles_per_routine,
            phase_offset_deg=b_phase_offset_deg,
        )

    for rep in range(repeats):
        for j in range(1, ncap + 1):
            t01 = j / float(ncap)
            c_cmd = c_for_phase(t01)
            tip_cmd = tip_for_phase(t01)
            if _capture_allowed(
                tip_angle_deg=tip_cmd,
                c_deg=c_cmd,
                tip_full_visible_min_deg=float(tip_full_visible_min_deg),
                tip_full_visible_max_deg=float(tip_full_visible_max_deg),
                vis1_min_deg=float(vis1_min_deg),
                vis1_max_deg=float(vis1_max_deg),
                vis2_min_deg=float(vis2_min_deg),
                vis2_max_deg=float(vis2_max_deg),
            ):
                idx = int(round(t01 * nmove))
                idx = max(1, min(nmove, idx))
                capture_indices.add((rep, idx))

    raw_points: List[dict] = []
    for rep in range(repeats):
        i_start = 0 if rep == 0 else 1
        for i in range(i_start, nmove + 1):
            t01 = i / float(nmove)
            raw_points.append(
                {
                    "rep": rep,
                    "i": i,
                    "routine_phase_01": float(t01),
                    "c": c_for_phase(t01),
                    "tip": tip_for_phase(t01),
                    "capture_image": ((rep, i) in capture_indices),
                }
            )

    traj: List[TrajectoryPoint] = []
    segment_stage_offset = np.zeros(3, dtype=float)
    prev_phase: Optional[str] = None
    prev_stage_xyz: Optional[np.ndarray] = None
    prev_b_cmd: Optional[float] = None
    phase_switch_count = 0

    for idx, point in enumerate(raw_points):
        prev_tip = None if idx == 0 else float(raw_points[idx - 1]["tip"])
        next_tip = None if idx + 1 >= len(raw_points) else float(raw_points[idx + 1]["tip"])
        deltas = []
        if prev_tip is not None:
            deltas.append(float(point["tip"]) - prev_tip)
        if next_tip is not None:
            deltas.append(next_tip - float(point["tip"]))
        motion_phase = cal.default_motion_phase
        for delta in deltas:
            if abs(delta) <= 1e-9:
                continue
            motion_phase = "pull" if delta > 0.0 else "release"
            break

        if motion_phase == "pull":
            b_cmd, used_tip = tip_angle_deg_to_b_continuous(
                requested_tip_angle_deg=float(point["tip"]),
                b_samples=pull_b_dense,
                angle_samples_deg=pull_angle_dense,
                prev_b=prev_b_cmd,
            )
        else:
            b_cmd, used_tip = tip_angle_deg_to_b_continuous(
                requested_tip_angle_deg=float(point["tip"]),
                b_samples=release_b_dense,
                angle_samples_deg=release_angle_dense,
                prev_b=prev_b_cmd,
            )

        raw_stage_xyz = stage_xyz_for_fixed_tip(
            cal=cal,
            p_tip_xyz=p_tip_fixed,
            b=float(b_cmd),
            c_deg=float(point["c"]),
            flip_rz_sign=flip_rz_sign,
            motion_phase=motion_phase,
        )
        if prev_phase is None:
            segment_stage_offset = np.zeros(3, dtype=float)
        elif motion_phase != prev_phase:
            phase_switch_count += 1
            if prev_stage_xyz is not None:
                # Branch switches happen with no motor move, so anchor the new phase to the
                # current commanded stage pose and follow the new branch from there.
                segment_stage_offset = prev_stage_xyz - raw_stage_xyz
        stage_xyz = np.asarray(raw_stage_xyz, dtype=float) + np.asarray(segment_stage_offset, dtype=float)
        traj.append(
            TrajectoryPoint(
                b=float(b_cmd),
                c=float(point["c"]),
                stage_xyz=stage_xyz,
                segment_kind=("start" if idx == 0 else "cycle"),
                capture_image=bool(point["capture_image"]),
                tip_angle_deg=float(used_tip),
                cycle_phase_01=float(point["routine_phase_01"]),
                leg_phase_01=float(point["routine_phase_01"]),
                leg_name="fast_routine",
                motion_phase=str(motion_phase),
            )
        )
        prev_phase = motion_phase
        prev_b_cmd = float(b_cmd)
        prev_stage_xyz = np.asarray(stage_xyz, dtype=float)

    n_captures = int(sum(1 for pt in traj if pt.capture_image))
    meta = {
        "requested_tip_min_deg": float(tip_min_deg),
        "requested_tip_max_deg": float(tip_max_deg),
        "used_tip_min_deg": float(used_tip_min),
        "used_tip_max_deg": float(used_tip_max),
        "available_tip_angle_range_deg": [available_tip_min, available_tip_max],
        "leg_move_steps": int(nmove),
        "leg_capture_steps": int(ncap),
        "boundary_ease_frac": 0.0,
        "phase_transition_steps": 0,
        "b_oscillations_per_sweep": float(b_cycles_per_routine),
        "b_oscillations_per_cycle": float(b_cycles_per_routine),
        "b_phase_offset_deg": float(b_phase_offset_deg),
        "capture_tip_full_visible_min_deg": float(tip_full_visible_min_deg),
        "capture_tip_full_visible_max_deg": float(tip_full_visible_max_deg),
        "visible_c_windows_deg": [
            [float(vis1_min_deg), float(vis1_max_deg)],
            [float(vis2_min_deg), float(vis2_max_deg)],
        ],
        "planned_capture_points": n_captures,
        "flip_rz_sign": bool(flip_rz_sign),
        "phase_sequence": ["fast_routine"],
        "fast_routine": True,
        "routine_repeats": int(repeats),
        "c_cycles_per_routine": float(c_cycles_per_routine),
        "b_cycles_per_routine": float(b_cycles_per_routine),
        "phase_switch_count": int(phase_switch_count),
    }
    return traj, meta


# =========================
# Feed scheduling
# =========================

def _smoothstep01(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _c_speed_envelope_factor(
    t01: float,
    accel_s: float,
    decel_s: float,
    total_s: float,
    floor_frac: float = 0.05,
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


def plan_segment_feeds_with_c_envelope(
    traj: List[TrajectoryPoint],
    probe_feed_mm_min: float,
    c_max_feed_deg_min: float,
    c_accel_time_s: float,
    c_decel_time_s: float,
    min_seg_time_s: float = 0.005,
) -> Tuple[List[float], dict]:
    nseg = max(0, len(traj) - 1)
    if nseg == 0:
        return [], {"est_total_time_s": 0.0, "max_est_c_speed_deg_min": 0.0}

    probe_feed = max(1e-6, float(probe_feed_mm_min))
    cmax = max(1e-6, float(c_max_feed_deg_min))

    xyzlens = []
    dcs = []
    for i in range(1, len(traj)):
        p0 = traj[i - 1].stage_xyz
        p1 = traj[i].stage_xyz
        c0 = traj[i - 1].c
        c1 = traj[i].c
        xyzlens.append(float(np.linalg.norm(p1 - p0)))
        dcs.append(abs(float(c1) - float(c0)))

    xyzlens = np.asarray(xyzlens, dtype=float)
    dcs = np.asarray(dcs, dtype=float)

    dt_xyz0 = xyzlens / (probe_feed / 60.0)
    dt_c0 = dcs / (cmax / 60.0)
    dt0 = np.maximum(dt_xyz0, dt_c0)
    dt0 = np.maximum(dt0, min_seg_time_s)
    total_est = float(np.sum(dt0))

    feeds = []
    dts = []
    t_cum = 0.0
    max_est_c_speed = 0.0

    for i in range(nseg):
        t_mid = t_cum + 0.5 * float(dt0[i])
        t01 = 0.0 if total_est <= 1e-9 else (t_mid / total_est)
        env = _c_speed_envelope_factor(
            t01=t01,
            accel_s=float(c_accel_time_s),
            decel_s=float(c_decel_time_s),
            total_s=total_est,
            floor_frac=0.05,
        )
        c_cap_i = cmax * env

        dt_xyz = xyzlens[i] / (probe_feed / 60.0)
        dt_c = dcs[i] / (c_cap_i / 60.0)
        dt = max(float(dt_xyz), float(dt_c), float(min_seg_time_s))

        if xyzlens[i] > 1e-9:
            f_i = 60.0 * xyzlens[i] / dt
            f_i = min(f_i, probe_feed)
            f_i = max(f_i, 1.0)
        else:
            f_i = probe_feed

        c_speed_est = (dcs[i] / dt) * 60.0 if dt > 1e-12 else 0.0
        max_est_c_speed = max(max_est_c_speed, c_speed_est)

        feeds.append(float(f_i))
        dts.append(float(dt))
        t_cum += dt

    return feeds, {
        "est_total_time_s": float(sum(dts)),
        "max_est_c_speed_deg_min": float(max_est_c_speed),
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
            "max_dc_step_deg": 0.0,
            "c_abs_path_deg": 0.0,
            "n_capture_points": 0,
        }

    xyz = np.vstack([pt.stage_xyz for pt in traj])
    bb = np.array([pt.b for pt in traj], dtype=float)
    cc = np.array([pt.c for pt in traj], dtype=float)

    diffs_xyz = xyz[1:] - xyz[:-1] if len(xyz) > 1 else np.zeros((0, 3))
    seglens = np.linalg.norm(diffs_xyz, axis=1) if len(diffs_xyz) else np.array([], dtype=float)
    dc = np.diff(cc) if len(cc) > 1 else np.array([], dtype=float)

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
        "max_dc_step_deg": float(np.max(np.abs(dc))) if len(dc) else 0.0,
        "c_abs_path_deg": float(np.sum(np.abs(dc))) if len(dc) else 0.0,
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


def build_absolute_move_gcode(feedrate: float, **axes_targets) -> str:
    parts = ["G90", "G1"]
    for ax, val in axes_targets.items():
        if val is None:
            continue
        parts.append(f"{ax}{float(val):.3f}")
    parts.append(f"F{float(feedrate):.3f}")
    return " ".join(parts)


def _pose_axes(cal: Calibration, pose: Tuple[float, float, float, float, float]) -> dict:
    x, y, z, b, c = [float(v) for v in pose]
    return {
        cal.x_axis: x,
        cal.y_axis: y,
        cal.z_axis: z,
        cal.b_axis: b,
        cal.c_axis: clamp_c_bounded(c),
    }


def _append_move_record(
    commands: List[str],
    command_records: List[dict],
    current_axes: dict,
    cal: Calibration,
    command_type: str,
    feedrate: float,
    source_trajectory_index: Optional[int] = None,
    **axes_targets,
) -> str:
    gcode = build_absolute_move_gcode(float(feedrate), **axes_targets)
    commands.append(gcode)
    for ax, val in axes_targets.items():
        if val is not None:
            current_axes[str(ax)] = float(val)

    command_records.append(
        {
            "command_index": int(len(command_records) + 1),
            "command_type": str(command_type),
            "source_trajectory_index": source_trajectory_index,
            "feedrate": float(feedrate),
            "gcode": gcode,
            "x_axis_name": cal.x_axis,
            "y_axis_name": cal.y_axis,
            "z_axis_name": cal.z_axis,
            "b_axis_name": cal.b_axis,
            "c_axis_name": cal.c_axis,
            "x_cmd": current_axes.get(cal.x_axis),
            "y_cmd": current_axes.get(cal.y_axis),
            "z_cmd": current_axes.get(cal.z_axis),
            "b_cmd": current_axes.get(cal.b_axis),
            "c_cmd": current_axes.get(cal.c_axis),
        }
    )
    return gcode


def _append_wait_command(
    commands: List[str],
    command_records: List[dict],
    current_axes: dict,
    cal: Calibration,
    command_type: str,
    gcode: str,
):
    commands.append(str(gcode))
    command_records.append(
        {
            "command_index": int(len(command_records) + 1),
            "command_type": str(command_type),
            "source_trajectory_index": None,
            "feedrate": None,
            "gcode": str(gcode),
            "x_axis_name": cal.x_axis,
            "y_axis_name": cal.y_axis,
            "z_axis_name": cal.z_axis,
            "b_axis_name": cal.b_axis,
            "c_axis_name": cal.c_axis,
            "x_cmd": current_axes.get(cal.x_axis),
            "y_cmd": current_axes.get(cal.y_axis),
            "z_cmd": current_axes.get(cal.z_axis),
            "b_cmd": current_axes.get(cal.b_axis),
            "c_cmd": current_axes.get(cal.c_axis),
        }
    )


def _append_safe_pose_moves(
    commands: List[str],
    command_records: List[dict],
    current_axes: dict,
    cal: Calibration,
    pose: Tuple[float, float, float, float, float],
    safe_approach_z: float,
    travel_feed: float,
    label: str,
    include_m400: bool = True,
    include_z_landing: bool = True,
):
    """Append the same three-step safe approach/retreat used by robot execution."""
    x, y, z, b, c = [float(v) for v in pose]

    _append_move_record(
        commands,
        command_records,
        current_axes,
        cal,
        f"{label}_safe_z",
        travel_feed,
        **{
            cal.z_axis: float(safe_approach_z),
        },
    )
    if include_m400:
        _append_wait_command(commands, command_records, current_axes, cal, f"{label}_safe_z_wait", "M400")

    _append_move_record(
        commands,
        command_records,
        current_axes,
        cal,
        f"{label}_safe_xy",
        travel_feed,
        **{
            cal.x_axis: x,
            cal.y_axis: y,
            cal.b_axis: b,
            cal.c_axis: clamp_c_bounded(c),
        },
    )
    if include_m400:
        _append_wait_command(commands, command_records, current_axes, cal, f"{label}_safe_xy_wait", "M400")

    if include_z_landing:
        _append_move_record(
            commands,
            command_records,
            current_axes,
            cal,
            f"{label}_safe_z_landing",
            travel_feed,
            **{
                cal.z_axis: z,
                cal.b_axis: b,
                cal.c_axis: clamp_c_bounded(c),
            },
        )
        if include_m400:
            _append_wait_command(commands, command_records, current_axes, cal, f"{label}_safe_z_landing_wait", "M400")


def _trajectory_rows_for_export(traj: List[TrajectoryPoint], seg_feeds: List[float]) -> List[dict]:
    rows: List[dict] = []
    for i, point in enumerate(traj):
        feed_to_point = None
        if i > 0:
            feed_to_point = seg_feeds[i - 1] if (i - 1) < len(seg_feeds) else None
        rows.append(
            {
                "trajectory_index": int(i),
                "segment_kind": point.segment_kind,
                "capture_image": bool(point.capture_image),
                "x_stage": float(point.stage_xyz[0]),
                "y_stage": float(point.stage_xyz[1]),
                "z_stage": float(point.stage_xyz[2]),
                "b_cmd": float(point.b),
                "c_cmd": float(clamp_c_bounded(point.c)),
                "c_raw_cmd": float(point.c),
                "tip_angle_deg": point.tip_angle_deg,
                "cycle_phase_01": point.cycle_phase_01,
                "leg_phase_01": point.leg_phase_01,
                "leg_name": point.leg_name,
                "motion_phase": point.motion_phase,
                "feed_to_point": feed_to_point,
            }
        )
    return rows


def _write_dict_csv(path: Path, rows: List[dict], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys: List[str] = []
        for row in rows:
            for key in row.keys():
                if key not in keys:
                    keys.append(key)
        fieldnames = keys
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return [_json_safe(v) for v in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return value


def build_smooth_tracking_gcode(
    cal: Calibration,
    traj: List[TrajectoryPoint],
    start_pose: Tuple[float, float, float, float, float],
    end_pose: Tuple[float, float, float, float, float],
    safe_approach_z: float,
    travel_feed: float,
    fine_approach_feed: float,
    probe_feed: float,
    c_max_feed: float,
    c_accel_time_s: float,
    c_decel_time_s: float,
    virtual_bbox: dict,
    dwell_before_ms: int = 0,
    dwell_after_ms: int = 0,
    initial_sweep_wait_s: float = DEFAULT_INITIAL_SWEEP_WAIT_S,
    use_segment_feed_scheduler: bool = True,
    include_startup_waits: bool = False,
    include_boundary_m400: bool = True,
    include_final_tracked_m400: bool = True,
) -> dict:
    """
    Build motion-only G-code for the dense tracking trajectory.

    Important: no M400/G4/camera waits are inserted at capture points. Capture
    flags are preserved in the trajectory CSV only so exported motion follows
    the same planned path but does not stop for image recording.
    """
    bbox_warnings: List[str] = []
    current_axes = _pose_axes(cal, start_pose)
    command_records: List[dict] = []
    commands: List[str] = [
        "; Exported by cyclic fixed-tip tracking script",
        "; Smooth motion-only program: capture markers are ignored, so the tracked sweep does not stop for images.",
        "G90",
    ]

    seg_feeds: List[float] = []
    sched_meta = {"est_total_time_s": 0.0, "max_est_c_speed_deg_min": 0.0, "mean_seg_time_ms": 0.0}
    if use_segment_feed_scheduler and len(traj) > 1:
        seg_feeds, sched_meta = plan_segment_feeds_with_c_envelope(
            traj=traj,
            probe_feed_mm_min=float(probe_feed),
            c_max_feed_deg_min=float(c_max_feed),
            c_accel_time_s=float(c_accel_time_s),
            c_decel_time_s=float(c_decel_time_s),
        )

    _append_safe_pose_moves(
        commands=commands,
        command_records=command_records,
        current_axes=current_axes,
        cal=cal,
        pose=start_pose,
        safe_approach_z=float(safe_approach_z),
        travel_feed=float(travel_feed),
        label="startup",
        include_m400=bool(include_boundary_m400),
        include_z_landing=True,
    )

    if traj:
        p0 = traj[0]
        x0, y0, z0 = _clamp_stage_xyz_to_bbox(
            p0.stage_xyz[0], p0.stage_xyz[1], p0.stage_xyz[2],
            virtual_bbox,
            "move to tracked start",
            bbox_warnings,
        )
        _append_move_record(
            commands,
            command_records,
            current_axes,
            cal,
            "tracked_start_transition",
            float(travel_feed),
            source_trajectory_index=0,
            **{
                cal.x_axis: x0,
                cal.y_axis: y0,
                cal.z_axis: z0,
                cal.b_axis: float(p0.b),
                cal.c_axis: clamp_c_bounded(float(p0.c)),
            },
        )
        if include_boundary_m400:
            _append_wait_command(commands, command_records, current_axes, cal, "tracked_start_transition_wait", "M400")

        _append_move_record(
            commands,
            command_records,
            current_axes,
            cal,
            "tracked_start_fine_land",
            float(fine_approach_feed),
            source_trajectory_index=0,
            **{
                cal.x_axis: x0,
                cal.y_axis: y0,
                cal.z_axis: z0,
                cal.b_axis: float(p0.b),
                cal.c_axis: clamp_c_bounded(float(p0.c)),
            },
        )
        if include_boundary_m400:
            _append_wait_command(commands, command_records, current_axes, cal, "tracked_start_fine_land_wait", "M400")

        if bool(include_startup_waits):
            if float(initial_sweep_wait_s) > 0:
                _append_wait_command(
                    commands,
                    command_records,
                    current_axes,
                    cal,
                    "initial_sweep_wait",
                    f"G4 S{float(initial_sweep_wait_s):.3f}",
                )
            if int(dwell_before_ms) > 0:
                _append_wait_command(
                    commands,
                    command_records,
                    current_axes,
                    cal,
                    "dwell_before_motion",
                    f"G4 P{int(dwell_before_ms)}",
                )

        commands.append("; Begin dense tracked motion. No capture waits inside this block.")
        for i, point in enumerate(traj[1:], start=1):
            x, y, z = _clamp_stage_xyz_to_bbox(
                point.stage_xyz[0], point.stage_xyz[1], point.stage_xyz[2],
                virtual_bbox,
                f"tracked sample {i}",
                bbox_warnings,
            )
            if point.segment_kind == "phase_transition":
                fseg = max(float(probe_feed), float(travel_feed))
            elif seg_feeds:
                fseg = float(seg_feeds[i - 1])
            else:
                fseg = float(probe_feed)
            _append_move_record(
                commands,
                command_records,
                current_axes,
                cal,
                "tracked_motion",
                fseg,
                source_trajectory_index=i,
                **{
                    cal.x_axis: x,
                    cal.y_axis: y,
                    cal.z_axis: z,
                    cal.b_axis: float(point.b),
                    cal.c_axis: clamp_c_bounded(float(point.c)),
                },
            )
        commands.append("; End dense tracked motion.")

        if len(traj) > 1 and bool(include_final_tracked_m400):
            _append_wait_command(commands, command_records, current_axes, cal, "tracked_motion_final_wait", "M400")

        if bool(include_startup_waits) and int(dwell_after_ms) > 0:
            _append_wait_command(
                commands,
                command_records,
                current_axes,
                cal,
                "dwell_after_motion",
                f"G4 P{int(dwell_after_ms)}",
            )

    _append_safe_pose_moves(
        commands=commands,
        command_records=command_records,
        current_axes=current_axes,
        cal=cal,
        pose=end_pose,
        safe_approach_z=float(safe_approach_z),
        travel_feed=float(travel_feed),
        label="ending",
        include_m400=bool(include_boundary_m400),
        include_z_landing=True,
    )

    capture_markers = int(sum(1 for pt in traj if pt.capture_image))
    return {
        "commands": commands,
        "command_records": command_records,
        "trajectory_rows": _trajectory_rows_for_export(traj, seg_feeds),
        "bbox_warnings": bbox_warnings,
        "scheduler_meta": sched_meta,
        "capture_markers_ignored": capture_markers,
        "tracked_command_count": int(max(0, len(traj) - 1)),
        "contains_capture_waits": False,
    }


def export_gcode_from_trajectory(
    output_path: Path,
    cal: Calibration,
    traj: List[TrajectoryPoint],
    start_pose: Tuple[float, float, float, float, float],
    end_pose: Tuple[float, float, float, float, float],
    safe_approach_z: float,
    travel_feed: float,
    fine_approach_feed: float,
    probe_feed: float,
    c_max_feed: float,
    c_accel_time_s: float,
    c_decel_time_s: float,
    virtual_bbox: dict,
    motion_meta: Optional[dict] = None,
    dwell_before_ms: int = 0,
    dwell_after_ms: int = 0,
    initial_sweep_wait_s: float = DEFAULT_INITIAL_SWEEP_WAIT_S,
    use_segment_feed_scheduler: bool = True,
    include_startup_waits: bool = False,
    include_boundary_m400: bool = True,
    include_final_tracked_m400: bool = True,
) -> dict:
    output_path = Path(output_path).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    base = output_path.with_suffix("")

    built = build_smooth_tracking_gcode(
        cal=cal,
        traj=traj,
        start_pose=start_pose,
        end_pose=end_pose,
        safe_approach_z=float(safe_approach_z),
        travel_feed=float(travel_feed),
        fine_approach_feed=float(fine_approach_feed),
        probe_feed=float(probe_feed),
        c_max_feed=float(c_max_feed),
        c_accel_time_s=float(c_accel_time_s),
        c_decel_time_s=float(c_decel_time_s),
        virtual_bbox=virtual_bbox,
        dwell_before_ms=int(dwell_before_ms),
        dwell_after_ms=int(dwell_after_ms),
        initial_sweep_wait_s=float(initial_sweep_wait_s),
        use_segment_feed_scheduler=bool(use_segment_feed_scheduler),
        include_startup_waits=bool(include_startup_waits),
        include_boundary_m400=bool(include_boundary_m400),
        include_final_tracked_m400=bool(include_final_tracked_m400),
    )

    output_path.write_text("\n".join(built["commands"]) + "\n", encoding="utf-8")

    command_csv_path = base.parent / f"{base.name}_commanded_motor_positions.csv"
    trajectory_csv_path = base.parent / f"{base.name}_trajectory_points.csv"
    metadata_path = base.parent / f"{base.name}_export_metadata.json"

    _write_dict_csv(command_csv_path, built["command_records"])
    _write_dict_csv(trajectory_csv_path, built["trajectory_rows"])

    metadata = {
        "gcode_path": str(output_path),
        "commanded_motor_positions_csv": str(command_csv_path),
        "trajectory_points_csv": str(trajectory_csv_path),
        "command_count": int(len(built["commands"])),
        "motion_command_count": int(len(built["command_records"])),
        "tracked_command_count": built["tracked_command_count"],
        "capture_markers_ignored": built["capture_markers_ignored"],
        "contains_capture_waits": built["contains_capture_waits"],
        "scheduler_meta": built["scheduler_meta"],
        "trajectory_meta": compute_traj_meta(traj),
        "motion_meta": motion_meta or {},
        "bbox_warnings": built["bbox_warnings"],
        "export_options": {
            "use_segment_feed_scheduler": bool(use_segment_feed_scheduler),
            "include_startup_waits": bool(include_startup_waits),
            "include_boundary_m400": bool(include_boundary_m400),
            "include_final_tracked_m400": bool(include_final_tracked_m400),
        },
    }
    metadata_path.write_text(json.dumps(_json_safe(metadata), indent=2) + "\n", encoding="utf-8")

    return {
        "path": str(output_path),
        "command_count": len(built["commands"]),
        "motion_command_count": len(built["command_records"]),
        "commanded_motor_positions_csv": str(command_csv_path),
        "trajectory_points_csv": str(trajectory_csv_path),
        "metadata_json": str(metadata_path),
        "bbox_warnings": built["bbox_warnings"],
        "capture_markers_ignored": built["capture_markers_ignored"],
        "scheduler_meta": built["scheduler_meta"],
    }


def _default_export_config() -> dict:
    return {
        "y_offset_fit": DEFAULT_Y_OFFSET_FIT,
        "curve_set": DEFAULT_CURVE_SET,
        "allow_global_curve_set_fallback": DEFAULT_ALLOW_GLOBAL_CURVE_SET_FALLBACK,
        "motion_mode": DEFAULT_MOTION_MODE,
        "point_x": DEFAULT_POINT_X,
        "point_y": DEFAULT_POINT_Y,
        "point_z": DEFAULT_POINT_Z,
        "cycle_repeats": DEFAULT_CYCLE_REPEATS,
        "routine_repeats": DEFAULT_ROUTINE_REPEATS,
        "leg_move_steps": DEFAULT_LEG_MOVE_STEPS,
        "leg_capture_steps": DEFAULT_LEG_CAPTURE_STEPS,
        "sweep_tip_min_deg": DEFAULT_SWEEP_TIP_MIN_DEG,
        "sweep_tip_max_deg": DEFAULT_SWEEP_TIP_MAX_DEG,
        "b_0_to_90_only": DEFAULT_B_0_TO_90_ONLY,
        "b_oscillations_per_sweep": DEFAULT_B_OSCILLATIONS_PER_SWEEP,
        "b_phase_offset_deg": DEFAULT_B_PHASE_OFFSET_DEG,
        "c_boundary_ease_frac": DEFAULT_C_BOUNDARY_EASE_FRAC,
        "custom_inverse_samples": DEFAULT_CUSTOM_INV_SAMPLES,
        "phase_transition_steps": DEFAULT_PHASE_TRANSITION_STEPS,
        "c_two_way_sweeps_per_b_oscillation": DEFAULT_C_TWO_WAY_SWEEPS_PER_B_OSCILLATION,
        "enable_branch_conditioning": DEFAULT_ENABLE_BRANCH_CONDITIONING,
        "branch_conditioning_tip_max_deg": DEFAULT_BRANCH_CONDITIONING_TIP_MAX_DEG,
        "capture_tip_full_visible_min_deg": DEFAULT_CAPTURE_TIP_FULL_VISIBLE_MIN_DEG,
        "capture_tip_full_visible_max_deg": DEFAULT_CAPTURE_TIP_FULL_VISIBLE_MAX_DEG,
        "c_visible_win1_min_deg": DEFAULT_C_VISIBLE_WIN1_MIN,
        "c_visible_win1_max_deg": DEFAULT_C_VISIBLE_WIN1_MAX,
        "c_visible_win2_min_deg": DEFAULT_C_VISIBLE_WIN2_MIN,
        "c_visible_win2_max_deg": DEFAULT_C_VISIBLE_WIN2_MAX,
        "flip_rz_sign": DEFAULT_FLIP_RZ_SIGN,
        "start_x": DEFAULT_START_X,
        "start_y": DEFAULT_START_Y,
        "start_z": DEFAULT_START_Z,
        "start_b": DEFAULT_START_B,
        "start_c": DEFAULT_START_C,
        "end_x": DEFAULT_END_X,
        "end_y": DEFAULT_END_Y,
        "end_z": DEFAULT_END_Z,
        "end_b": DEFAULT_END_B,
        "end_c": DEFAULT_END_C,
        "safe_approach_z": DEFAULT_SAFE_APPROACH_Z,
        "bbox_x_min": DEFAULT_BBOX_X_MIN,
        "bbox_x_max": DEFAULT_BBOX_X_MAX,
        "bbox_y_min": DEFAULT_BBOX_Y_MIN,
        "bbox_y_max": DEFAULT_BBOX_Y_MAX,
        "bbox_z_min": DEFAULT_BBOX_Z_MIN,
        "bbox_z_max": DEFAULT_BBOX_Z_MAX,
        "travel_feed": DEFAULT_TRAVEL_FEED,
        "fine_approach_feed": DEFAULT_FINE_APPROACH_FEED,
        "probe_feed": DEFAULT_PROBE_FEED,
        "c_max_feed": DEFAULT_C_MAX_FEED,
        "c_accel_time_s": DEFAULT_C_ACCEL_TIME_S,
        "c_decel_time_s": DEFAULT_C_DECEL_TIME_S,
        "dwell_before_ms": DEFAULT_DWELL_BEFORE_MS,
        "dwell_after_ms": DEFAULT_DWELL_AFTER_MS,
        "initial_sweep_wait_s": DEFAULT_INITIAL_SWEEP_WAIT_S,
        "use_segment_feed_scheduler": True,
        "include_startup_waits": False,
        "include_boundary_m400": True,
        "include_final_tracked_m400": True,
    }


def _generate_trajectory_from_config(cal: Calibration, cfg: dict) -> Tuple[List[TrajectoryPoint], dict]:
    p_tip_fixed = np.array(
        [float(cfg["point_x"]), float(cfg["point_y"]), float(cfg["point_z"])],
        dtype=float,
    )
    sweep_tip_min_deg = float(cfg["sweep_tip_min_deg"])
    sweep_tip_max_deg = float(cfg["sweep_tip_max_deg"])
    if bool(cfg.get("b_0_to_90_only", False)):
        sweep_tip_min_deg = 0.0
        sweep_tip_max_deg = 90.0

    motion_mode = str(cfg.get("motion_mode", DEFAULT_MOTION_MODE)).lower()
    if motion_mode == "fast_routine":
        return generate_fast_routine_trajectory(
            cal=cal,
            p_tip_fixed=p_tip_fixed,
            repeats=int(cfg["routine_repeats"]),
            move_steps_per_routine=int(cfg["leg_move_steps"]) * 2,
            capture_steps_per_routine=int(cfg["leg_capture_steps"]) * 2,
            tip_min_deg=sweep_tip_min_deg,
            tip_max_deg=sweep_tip_max_deg,
            tip_full_visible_min_deg=float(cfg["capture_tip_full_visible_min_deg"]),
            tip_full_visible_max_deg=float(cfg["capture_tip_full_visible_max_deg"]),
            vis1_min_deg=float(cfg["c_visible_win1_min_deg"]),
            vis1_max_deg=float(cfg["c_visible_win1_max_deg"]),
            vis2_min_deg=float(cfg["c_visible_win2_min_deg"]),
            vis2_max_deg=float(cfg["c_visible_win2_max_deg"]),
            inverse_samples=int(cfg["custom_inverse_samples"]),
            flip_rz_sign=bool(cfg["flip_rz_sign"]),
        )

    return generate_cyclic_visibility_gated_trajectory(
        cal=cal,
        p_tip_fixed=p_tip_fixed,
        repeats=int(cfg["cycle_repeats"]),
        leg_move_steps=int(cfg["leg_move_steps"]),
        leg_capture_steps=int(cfg["leg_capture_steps"]),
        tip_min_deg=sweep_tip_min_deg,
        tip_max_deg=sweep_tip_max_deg,
        b_oscillations_per_sweep=float(cfg["b_oscillations_per_sweep"]),
        b_phase_offset_deg=float(cfg["b_phase_offset_deg"]),
        tip_full_visible_min_deg=float(cfg["capture_tip_full_visible_min_deg"]),
        tip_full_visible_max_deg=float(cfg["capture_tip_full_visible_max_deg"]),
        vis1_min_deg=float(cfg["c_visible_win1_min_deg"]),
        vis1_max_deg=float(cfg["c_visible_win1_max_deg"]),
        vis2_min_deg=float(cfg["c_visible_win2_min_deg"]),
        vis2_max_deg=float(cfg["c_visible_win2_max_deg"]),
        boundary_ease_frac=float(cfg["c_boundary_ease_frac"]),
        inverse_samples=int(cfg["custom_inverse_samples"]),
        phase_transition_steps=int(cfg["phase_transition_steps"]),
        c_two_way_sweeps_per_b_oscillation=int(cfg.get("c_two_way_sweeps_per_b_oscillation", DEFAULT_C_TWO_WAY_SWEEPS_PER_B_OSCILLATION)),
        flip_rz_sign=bool(cfg["flip_rz_sign"]),
        enable_branch_conditioning=bool(cfg.get("enable_branch_conditioning", DEFAULT_ENABLE_BRANCH_CONDITIONING)),
        branch_conditioning_tip_max_deg=float(cfg.get("branch_conditioning_tip_max_deg", DEFAULT_BRANCH_CONDITIONING_TIP_MAX_DEG)),
    )


def _bbox_from_config(cfg: dict) -> dict:
    virtual_bbox = {
        "x_min": float(cfg["bbox_x_min"]),
        "x_max": float(cfg["bbox_x_max"]),
        "y_min": float(cfg["bbox_y_min"]),
        "y_max": float(cfg["bbox_y_max"]),
        "z_min": float(cfg["bbox_z_min"]),
        "z_max": float(cfg["bbox_z_max"]),
    }
    if virtual_bbox["x_min"] > virtual_bbox["x_max"]:
        virtual_bbox["x_min"], virtual_bbox["x_max"] = virtual_bbox["x_max"], virtual_bbox["x_min"]
    if virtual_bbox["y_min"] > virtual_bbox["y_max"]:
        virtual_bbox["y_min"], virtual_bbox["y_max"] = virtual_bbox["y_max"], virtual_bbox["y_min"]
    if virtual_bbox["z_min"] > virtual_bbox["z_max"]:
        virtual_bbox["z_min"], virtual_bbox["z_max"] = virtual_bbox["z_max"], virtual_bbox["z_min"]
    return virtual_bbox


def export_gcode(calibration: str, output_path: str, **overrides) -> dict:
    """
    Importable convenience function for smooth motion-only export.

    Example:
        export_gcode(
            "calibration.json",
            "smooth_tracking_motion.gcode",
            cycle_repeats=2,
            leg_move_steps=800,
            include_startup_waits=False,
        )

    Keyword overrides use the same names as the CLI options with dashes changed
    to underscores. The exported tracked sweep has no image-capture stops.
    """
    cfg = _default_export_config()
    unknown = sorted(set(overrides.keys()) - set(cfg.keys()))
    if unknown:
        raise TypeError(f"Unknown export_gcode override(s): {unknown}")
    cfg.update(overrides)

    cal = load_calibration(
        str(calibration),
        y_offset_fit=str(cfg["y_offset_fit"]),
        curve_set=str(cfg["curve_set"]),
        allow_global_curve_set_fallback=bool(cfg["allow_global_curve_set_fallback"]),
    )
    traj, custom_meta = _generate_trajectory_from_config(cal, cfg)
    start_pose = (
        float(cfg["start_x"]), float(cfg["start_y"]), float(cfg["start_z"]),
        float(cfg["start_b"]), float(cfg["start_c"]),
    )
    end_pose = (
        float(cfg["end_x"]), float(cfg["end_y"]), float(cfg["end_z"]),
        float(cfg["end_b"]), float(cfg["end_c"]),
    )
    return export_gcode_from_trajectory(
        output_path=Path(output_path),
        cal=cal,
        traj=traj,
        start_pose=start_pose,
        end_pose=end_pose,
        safe_approach_z=float(cfg["safe_approach_z"]),
        travel_feed=float(cfg["travel_feed"]),
        fine_approach_feed=float(cfg["fine_approach_feed"]),
        probe_feed=float(cfg["probe_feed"]),
        c_max_feed=float(cfg["c_max_feed"]),
        c_accel_time_s=float(cfg["c_accel_time_s"]),
        c_decel_time_s=float(cfg["c_decel_time_s"]),
        virtual_bbox=_bbox_from_config(cfg),
        motion_meta=custom_meta,
        dwell_before_ms=int(cfg["dwell_before_ms"]),
        dwell_after_ms=int(cfg["dwell_after_ms"]),
        initial_sweep_wait_s=float(cfg["initial_sweep_wait_s"]),
        use_segment_feed_scheduler=bool(cfg["use_segment_feed_scheduler"]),
        include_startup_waits=bool(cfg["include_startup_waits"]),
        include_boundary_m400=bool(cfg["include_boundary_m400"]),
        include_final_tracked_m400=bool(cfg["include_final_tracked_m400"]),
    )


# Backwards-compatible wrapper for the old export name.
def export_commanded_gcode(
    output_path: Path,
    cal: Calibration,
    traj: List[TrajectoryPoint],
    start_pose: Tuple[float, float, float, float, float],
    end_pose: Tuple[float, float, float, float, float],
    safe_approach_z: float,
    travel_feed: float,
    fine_approach_feed: float,
    probe_feed: float,
    virtual_bbox: dict,
    dwell_before_ms: int = 0,
    dwell_after_ms: int = 0,
    initial_sweep_wait_s: float = DEFAULT_INITIAL_SWEEP_WAIT_S,
    c_max_feed: float = DEFAULT_C_MAX_FEED,
    c_accel_time_s: float = DEFAULT_C_ACCEL_TIME_S,
    c_decel_time_s: float = DEFAULT_C_DECEL_TIME_S,
    use_segment_feed_scheduler: bool = True,
    include_startup_waits: bool = False,
) -> dict:
    return export_gcode_from_trajectory(
        output_path=Path(output_path),
        cal=cal,
        traj=traj,
        start_pose=start_pose,
        end_pose=end_pose,
        safe_approach_z=float(safe_approach_z),
        travel_feed=float(travel_feed),
        fine_approach_feed=float(fine_approach_feed),
        probe_feed=float(probe_feed),
        c_max_feed=float(c_max_feed),
        c_accel_time_s=float(c_accel_time_s),
        c_decel_time_s=float(c_decel_time_s),
        virtual_bbox=virtual_bbox,
        motion_meta={},
        dwell_before_ms=int(dwell_before_ms),
        dwell_after_ms=int(dwell_after_ms),
        initial_sweep_wait_s=float(initial_sweep_wait_s),
        use_segment_feed_scheduler=bool(use_segment_feed_scheduler),
        include_startup_waits=bool(include_startup_waits),
    )


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
        cycle_phase_01: Optional[float] = None,
        leg_name: Optional[str] = None,
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
        if leg_name is not None:
            extra += f"_{leg_name}"
        if tip_angle_deg is not None:
            extra += f"_TIP{float(tip_angle_deg):.3f}"
        if cycle_phase_01 is not None:
            extra += f"_CPH{float(cycle_phase_01):.5f}"
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

    def _estimate_move_time_s(
        self,
        cal: Calibration,
        previous_axes: dict,
        axes_targets: dict,
        feedrate: float,
        c_max_feed: float,
        motion_accel_mm_s2: float = DEFAULT_MOTION_ACCEL_MM_S2,
        min_move_time_s: float = DEFAULT_MIN_ESTIMATED_MOVE_TIME_S,
    ) -> float:
        feed_s = max(1e-9, float(feedrate) / 60.0)
        c_feed_s = max(1e-9, float(c_max_feed) / 60.0)

        xyz_sq = 0.0
        for axis in (cal.x_axis, cal.y_axis, cal.z_axis):
            prev = previous_axes.get(str(axis))
            tgt = axes_targets.get(str(axis))
            if prev is None or tgt is None:
                continue
            delta = float(tgt) - float(prev)
            xyz_sq += delta * delta
        xyz_dist = math.sqrt(xyz_sq) if xyz_sq > 0 else 0.0
        xyz_t = accel_limited_move_time_seconds(
            distance_mm=xyz_dist,
            feedrate_mm_min=float(feedrate),
            accel_mm_s2=float(motion_accel_mm_s2),
        ) if xyz_dist > 0 else 0.0

        c_t = 0.0
        prev_c = previous_axes.get(str(cal.c_axis))
        tgt_c = axes_targets.get(str(cal.c_axis))
        if prev_c is not None and tgt_c is not None:
            c_t = abs(float(tgt_c) - float(prev_c)) / c_feed_s

        b_t = 0.0
        prev_b = previous_axes.get(str(cal.b_axis))
        tgt_b = axes_targets.get(str(cal.b_axis))
        if prev_b is not None and tgt_b is not None:
            b_t = abs(float(tgt_b) - float(prev_b)) / feed_s

        est = max(xyz_t, c_t, b_t)
        return max(float(est), float(min_move_time_s)) if est > 0 else 0.0

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

    def _record_estimated_motion(
        self,
        cal: Calibration,
        command_record: dict,
        c_max_feed: float,
        motion_accel_mm_s2: float = DEFAULT_MOTION_ACCEL_MM_S2,
    ) -> float:
        est_s = self._estimate_move_time_s(
            cal=cal,
            previous_axes=command_record["previous_axes"],
            axes_targets=command_record["axes_targets"],
            feedrate=command_record["feedrate"],
            c_max_feed=float(c_max_feed),
            motion_accel_mm_s2=float(motion_accel_mm_s2),
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

    def _wait_for_estimated_motion_progress(
        self,
        lookahead_s: float = DEFAULT_STREAMING_LOOKAHEAD_S,
        extra_settle: float = 0.0,
        reason: str = "motion",
    ):
        wait_s = max(
            0.0,
            self.estimated_motion_done_at - time.monotonic() - max(0.0, float(lookahead_s)),
        ) + max(0.0, float(extra_settle))
        if wait_s > 0:
            print(f" Estimated paced wait for {reason}: {wait_s:.3f} s")
            time.sleep(wait_s)

    def send_absolute_move(self, feedrate: float, **axes_targets) -> dict:
        if self.rrf is None:
            raise RuntimeError("Robot is not connected.")

        previous_axes = dict(self.commanded_axes)
        gcode = build_absolute_move_gcode(feedrate, **axes_targets)
        print(f" Command: {gcode}")
        self.rrf.send_code(gcode)
        for ax, val in axes_targets.items():
            if val is None:
                continue
            self.commanded_axes[str(ax)] = float(val)
        return {
            "feedrate": float(feedrate),
            "gcode": gcode,
            "previous_axes": previous_axes,
            "axes_targets": {
                str(ax): float(val)
                for ax, val in axes_targets.items()
                if val is not None
            },
        }

    def _move_to_pose_safe(
        self,
        cal: Calibration,
        pose: Tuple[float, float, float, float, float],
        safe_approach_z: float,
        travel_feed: float,
        settle_s: float,
        use_estimated_waits: bool = False,
        c_max_feed: float = DEFAULT_C_MAX_FEED,
        motion_accel_mm_s2: float = DEFAULT_MOTION_ACCEL_MM_S2,
        include_z_landing: bool = True,
    ):
        x, y, z, b, c = [float(v) for v in pose]

        cmd = self.send_absolute_move(
            travel_feed,
            **{
                cal.z_axis: float(safe_approach_z),
            }
        )
        if use_estimated_waits:
            self._record_estimated_motion(
                cal=cal,
                command_record=cmd,
                c_max_feed=c_max_feed,
                motion_accel_mm_s2=motion_accel_mm_s2,
            )
            self._wait_for_estimated_motion_complete(extra_settle=0.0, reason="safe Z raise")
        else:
            self.wait_for_duet_motion_complete(extra_settle=0.0)

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
            self._record_estimated_motion(
                cal=cal,
                command_record=cmd,
                c_max_feed=c_max_feed,
                motion_accel_mm_s2=motion_accel_mm_s2,
            )
            self._wait_for_estimated_motion_complete(extra_settle=0.0, reason="safe XYBC reposition")
        else:
            self.wait_for_duet_motion_complete(extra_settle=0.0)

        if include_z_landing:
            cmd = self.send_absolute_move(
                travel_feed,
                **{
                    cal.z_axis: z,
                }
            )
            if use_estimated_waits:
                self._record_estimated_motion(
                    cal=cal,
                    command_record=cmd,
                    c_max_feed=c_max_feed,
                    motion_accel_mm_s2=motion_accel_mm_s2,
                )
                self._wait_for_estimated_motion_complete(extra_settle=settle_s, reason="safe Z landing")
            else:
                self.wait_for_duet_motion_complete(extra_settle=settle_s)
        elif settle_s > 0:
            time.sleep(float(settle_s))

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
        c_max_feed: float = DEFAULT_C_MAX_FEED,
        motion_accel_mm_s2: float = DEFAULT_MOTION_ACCEL_MM_S2,
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
            self._record_estimated_motion(
                cal=cal,
                command_record=cmd,
                c_max_feed=c_max_feed,
                motion_accel_mm_s2=motion_accel_mm_s2,
            )
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
        c_feed: float,
        c_max_feed: float,
        c_accel_time_s: float,
        c_decel_time_s: float,
        virtual_bbox: dict,
        dwell_before_ms: int = 0,
        dwell_after_ms: int = 0,
        preposition_c_only: bool = False,
        use_segment_feed_scheduler: bool = True,
        tracked_move_settle_s: float = 0.0,
        travel_move_settle_s: float = 0.0,
        b_extra_settle_s: float = 0.0,
        inter_command_delay_s: float = 0.0,
        camera_flush_frames: int = 1,
        capture_at_start: bool = True,
        initial_sweep_wait_s: float = DEFAULT_INITIAL_SWEEP_WAIT_S,
        settled_capture_mode: bool = DEFAULT_SETTLED_CAPTURE_MODE,
        settled_capture_buffer_s: float = DEFAULT_SETTLED_CAPTURE_BUFFER_S,
        motion_accel_mm_s2: float = DEFAULT_MOTION_ACCEL_MM_S2,
        streaming_lookahead_s: float = DEFAULT_STREAMING_LOOKAHEAD_S,
    ):
        if self.cam is None:
            raise RuntimeError("Camera is not connected.")
        if self.rrf is None:
            raise RuntimeError("Robot is not connected.")

        bbox_warnings: List[str] = []
        sample_counter = 0
        self.commanded_axes = {}
        self._seed_commanded_pose(cal=cal, pose=start_pose)
        is_fast_routine = bool(traj) and str(traj[0].leg_name) == "fast_routine"
        effective_streaming_lookahead_s = (
            max(float(streaming_lookahead_s), 0.25) if is_fast_routine else float(streaming_lookahead_s)
        )
        settled_capture_buffer_s = 0.0

        seg_feeds: List[float] = []
        sched_meta = {"est_total_time_s": 0.0, "max_est_c_speed_deg_min": 0.0, "mean_seg_time_ms": 0.0}
        if use_segment_feed_scheduler and len(traj) > 1:
            seg_feeds, sched_meta = plan_segment_feeds_with_c_envelope(
                traj=traj,
                probe_feed_mm_min=float(probe_feed),
                c_max_feed_deg_min=float(c_max_feed),
                c_accel_time_s=float(c_accel_time_s),
                c_decel_time_s=float(c_decel_time_s),
            )

        print("\n" + "=" * 72)
        print("STARTING TRACKED POINT-ACQUISITION RUN")
        print("=" * 72)
        print(f"Tracked samples: {len(traj)}")
        print(f"Estimated tracked time: {sched_meta['est_total_time_s']:.3f} s")
        print(f"Estimated max C speed: {sched_meta['max_est_c_speed_deg_min']:.1f} deg/min")
        print(f"Mean segment time: {sched_meta['mean_seg_time_ms']:.2f} ms")
        if use_segment_feed_scheduler:
            print("Segment feed scheduling enabled for tracked execution.")
        print(f"Tracked block feed: {float(probe_feed):.3f}")
        if settled_capture_mode:
            print(
                f"Settled capture mode: streaming commands with estimated pacing "
                f"(lookahead={float(effective_streaming_lookahead_s):.3f} s, "
                f"accel={float(motion_accel_mm_s2):.3f} mm/s^2)."
            )
        if is_fast_routine:
            print("Fast routine execution: capture samples stay in-stream; only the routine boundaries fully stop.")

        print("\nSafe startup approach...")
        self._move_to_pose_safe(
            cal=cal,
            pose=start_pose,
            safe_approach_z=float(safe_approach_z),
            travel_feed=float(travel_feed),
            settle_s=float(travel_move_settle_s),
            use_estimated_waits=bool(settled_capture_mode),
            c_max_feed=float(c_max_feed),
            motion_accel_mm_s2=float(motion_accel_mm_s2),
            include_z_landing=True,
        )

        if not traj:
            print("No trajectory points generated.")
        else:
            p0 = traj[0]
            b0, c0, p0_stage = p0.b, p0.c, p0.stage_xyz

            x0, y0, z0 = _clamp_stage_xyz_to_bbox(
                p0_stage[0], p0_stage[1], p0_stage[2],
                virtual_bbox,
                "move to tracked start",
                bbox_warnings,
            )

            print("\nMoving to first tracked sample...")
            cmd = self.send_absolute_move(
                travel_feed,
                **{
                    cal.x_axis: x0,
                    cal.y_axis: y0,
                    cal.z_axis: z0,
                    cal.b_axis: b0,
                    cal.c_axis: clamp_c_bounded(c0),
                }
            )
            if settled_capture_mode:
                self._record_estimated_motion(
                    cal=cal,
                    command_record=cmd,
                    c_max_feed=float(c_max_feed),
                    motion_accel_mm_s2=float(motion_accel_mm_s2),
                )
                self._wait_for_estimated_motion_complete(
                    extra_settle=float(travel_move_settle_s),
                    reason="tracked start transition",
                )
            else:
                self.wait_for_duet_motion_complete(extra_settle=travel_move_settle_s)

            self._fine_land_on_point(
                cal=cal,
                x=x0,
                y=y0,
                z=z0,
                b=float(b0),
                c=float(c0),
                fine_feed=float(fine_approach_feed),
                settle_s=max(float(tracked_move_settle_s), 0.05),
                use_estimated_waits=bool(settled_capture_mode),
                c_max_feed=float(c_max_feed),
                motion_accel_mm_s2=float(motion_accel_mm_s2),
            )

            if float(initial_sweep_wait_s) > 0:
                print(f"Waiting {float(initial_sweep_wait_s):.3f} s before starting the sweep...")
                time.sleep(float(initial_sweep_wait_s))

            if capture_at_start and p0.capture_image:
                sample_counter += 1
                self.capture_and_save(
                    sample_idx=sample_counter,
                    phase="tracked_start",
                    x=x0,
                    y=y0,
                    z=z0,
                    b=b0,
                    c=clamp_c_bounded(c0),
                    flush_frames=camera_flush_frames,
                    tip_angle_deg=p0.tip_angle_deg,
                    cycle_phase_01=p0.cycle_phase_01,
                    leg_name=p0.leg_name,
                    motion_phase=p0.motion_phase,
                )
            else:
                print("Start point not captured.")

            if int(dwell_before_ms) > 0:
                print(f"Dwell before motion: {int(dwell_before_ms)} ms")
                time.sleep(float(dwell_before_ms) / 1000.0)

            print("\nExecuting coordinated tracked motion...")
            for i, point in enumerate(traj[1:], start=1):
                b, c, p_stage = point.b, point.c, point.stage_xyz
                x, y, z = _clamp_stage_xyz_to_bbox(
                    p_stage[0], p_stage[1], p_stage[2],
                    virtual_bbox,
                    f"tracked sample {i}",
                    bbox_warnings,
                )

                if point.segment_kind == "phase_transition":
                    fseg = max(float(probe_feed), float(travel_feed))
                elif seg_feeds:
                    fseg = seg_feeds[i - 1]
                else:
                    fseg = float(probe_feed)

                cmd = self.send_absolute_move(
                    fseg,
                    **{
                        cal.x_axis: x,
                        cal.y_axis: y,
                        cal.z_axis: z,
                        cal.b_axis: b,
                        cal.c_axis: clamp_c_bounded(c),
                    }
                )
                if settled_capture_mode:
                    self._record_estimated_motion(
                        cal=cal,
                        command_record=cmd,
                        c_max_feed=float(c_max_feed),
                        motion_accel_mm_s2=float(motion_accel_mm_s2),
                    )
                    if point.capture_image and not is_fast_routine:
                        self._wait_for_estimated_motion_complete(
                            extra_settle=0.0,
                            reason=f"tracked sample {i} capture",
                        )
                    else:
                        self._wait_for_estimated_motion_progress(
                            lookahead_s=float(effective_streaming_lookahead_s),
                            reason=f"tracked sample {i}",
                        )
                if float(inter_command_delay_s) > 0:
                    time.sleep(float(inter_command_delay_s))

                if point.capture_image:
                    if not settled_capture_mode:
                        self.wait_for_duet_motion_complete(extra_settle=0.0)
                    sample_counter += 1
                    self.capture_and_save(
                        sample_idx=sample_counter,
                        phase=point.segment_kind,
                        x=x,
                        y=y,
                        z=z,
                        b=b,
                        c=clamp_c_bounded(c),
                        flush_frames=camera_flush_frames,
                        tip_angle_deg=point.tip_angle_deg,
                        cycle_phase_01=point.cycle_phase_01,
                        leg_name=point.leg_name,
                        motion_phase=point.motion_phase,
                    )

            if len(traj) > 1:
                if settled_capture_mode:
                    self._wait_for_estimated_motion_complete(
                        extra_settle=float(tracked_move_settle_s),
                        reason="tracked motion end",
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
            c_max_feed=float(c_max_feed),
            motion_accel_mm_s2=float(motion_accel_mm_s2),
            include_z_landing=True,
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
            "scheduler_meta": sched_meta,
        }


# =========================
# Main
# =========================

def main(args):
    script_dir = Path(__file__).resolve().parent
    prefer_0_to_90_curve = bool(args.b_0_to_90_only)
    cal = load_calibration(
        args.calibration,
        y_offset_fit=args.y_offset_fit,
        curve_set=args.curve_set,
        allow_global_curve_set_fallback=bool(args.allow_global_curve_set_fallback),
        prefer_0_to_90_curve=prefer_0_to_90_curve,
    )

    p_tip_fixed = np.array(
        [float(args.point_x), float(args.point_y), float(args.point_z)],
        dtype=float
    )

    sweep_tip_min_deg = float(args.sweep_tip_min_deg)
    sweep_tip_max_deg = float(args.sweep_tip_max_deg)
    if bool(args.b_0_to_90_only):
        sweep_tip_min_deg = 0.0
        sweep_tip_max_deg = 90.0

    motion_mode = str(args.motion_mode).lower()
    cycle_repeats = int(args.cycle_repeats)
    if motion_mode == "fast_routine":
        cycle_repeats = int(args.routine_repeats)
        traj, custom_meta = generate_fast_routine_trajectory(
            cal=cal,
            p_tip_fixed=p_tip_fixed,
            repeats=cycle_repeats,
            move_steps_per_routine=int(args.leg_move_steps) * 2,
            capture_steps_per_routine=int(args.leg_capture_steps) * 2,
            tip_min_deg=sweep_tip_min_deg,
            tip_max_deg=sweep_tip_max_deg,
            tip_full_visible_min_deg=float(args.capture_tip_full_visible_min_deg),
            tip_full_visible_max_deg=float(args.capture_tip_full_visible_max_deg),
            vis1_min_deg=float(args.c_visible_win1_min_deg),
            vis1_max_deg=float(args.c_visible_win1_max_deg),
            vis2_min_deg=float(args.c_visible_win2_min_deg),
            vis2_max_deg=float(args.c_visible_win2_max_deg),
            inverse_samples=int(args.custom_inverse_samples),
            flip_rz_sign=bool(args.flip_rz_sign),
        )
    else:
        traj, custom_meta = generate_cyclic_visibility_gated_trajectory(
            cal=cal,
            p_tip_fixed=p_tip_fixed,
            repeats=cycle_repeats,
            leg_move_steps=int(args.leg_move_steps),
            leg_capture_steps=int(args.leg_capture_steps),
            tip_min_deg=sweep_tip_min_deg,
            tip_max_deg=sweep_tip_max_deg,
            b_oscillations_per_sweep=float(args.b_oscillations_per_sweep),
            b_phase_offset_deg=float(args.b_phase_offset_deg),
            tip_full_visible_min_deg=float(args.capture_tip_full_visible_min_deg),
            tip_full_visible_max_deg=float(args.capture_tip_full_visible_max_deg),
            vis1_min_deg=float(args.c_visible_win1_min_deg),
            vis1_max_deg=float(args.c_visible_win1_max_deg),
            vis2_min_deg=float(args.c_visible_win2_min_deg),
            vis2_max_deg=float(args.c_visible_win2_max_deg),
            boundary_ease_frac=float(args.c_boundary_ease_frac),
            inverse_samples=int(args.custom_inverse_samples),
            phase_transition_steps=int(args.phase_transition_steps),
            c_two_way_sweeps_per_b_oscillation=int(args.c_two_way_sweeps_per_b_oscillation),
            flip_rz_sign=bool(args.flip_rz_sign),
            enable_branch_conditioning=not bool(args.disable_branch_conditioning),
            branch_conditioning_tip_max_deg=float(args.branch_conditioning_tip_max_deg),
        )

    meta = compute_traj_meta(traj)
    print("Trajectory summary:")
    print(f"  Motion mode: {motion_mode}")
    print(f"  Samples: {meta['n_samples']} (segments={meta['n_segments']})")
    print(f"  Capture points: {meta['n_capture_points']}")
    print(f"  B range used: [{meta['b_min_used']:.3f}, {meta['b_max_used']:.3f}]")
    print(f"  Requested curve set: {cal.requested_curve_set}")
    print(f"  Selected curve set:  {cal.selected_curve_set}")
    print(f"  Curve selection:     {cal.curve_selection_reason}")
    if cal.available_curve_sets:
        print(f"  Available curl-specific sets: {cal.available_curve_sets}")
    if cal.tip_angle_model is not None and meta["n_samples"] > 0:
        bb = np.array([meta["b_min_used"], meta["b_max_used"]], dtype=float)
        tip_angle_used = eval_tip_angle_deg(cal, bb)
        print(
            "  Tip-angle range at used B endpoints: "
            f"[{float(np.min(tip_angle_used)):.3f}, {float(np.max(tip_angle_used)):.3f}] deg"
        )
    print(f"  C range used: [{meta['c_min_used']:.3f}, {meta['c_max_used']:.3f}]")
    print(f"  XYZ path length: {meta['xyz_path_len_mm']:.3f} mm")

    print("Cycle / capture summary:")
    print(f"  Requested tip min/max: [{custom_meta['requested_tip_min_deg']}, {custom_meta['requested_tip_max_deg']}]")
    print(f"  Used tip min/max:      [{custom_meta['used_tip_min_deg']}, {custom_meta['used_tip_max_deg']}]")
    print(f"  B 0..90 only mode: {bool(args.b_0_to_90_only)}")
    print(f"  Available calibrated tip-angle range: {custom_meta['available_tip_angle_range_deg']}")
    print(f"  Leg move steps: {custom_meta['leg_move_steps']}")
    print(f"  Leg capture steps: {custom_meta['leg_capture_steps']}")
    print(f"  Boundary ease fraction: {custom_meta['boundary_ease_frac']}")
    print(f"  Phase transition steps: {custom_meta['phase_transition_steps']}")
    if "phase_transition_mode" in custom_meta:
        print(f"  Phase transition mode: {custom_meta['phase_transition_mode']}")
    print(f"  B oscillations per sweep: {custom_meta['b_oscillations_per_sweep']}")
    print(f"  B oscillations per cycle: {custom_meta['b_oscillations_per_cycle']}")
    if "c_two_way_sweeps_per_b_oscillation" in custom_meta:
        print(f"  C two-way sweeps per B oscillation: {custom_meta['c_two_way_sweeps_per_b_oscillation']}")
    print(f"  B phase offset deg: {custom_meta['b_phase_offset_deg']}")
    print(f"  Full-visible tip range: [{custom_meta['capture_tip_full_visible_min_deg']}, "
          f"{custom_meta['capture_tip_full_visible_max_deg']}]")
    print(f"  Visible C windows (deg): {custom_meta['visible_c_windows_deg']}")
    print(f"  Planned capture points: {custom_meta['planned_capture_points']}")
    print(f"  flip_rz_sign: {custom_meta['flip_rz_sign']}")
    if "phase_switch_count" in custom_meta:
        print(f"  Phase switches: {custom_meta['phase_switch_count']}")

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

    export_requested = bool(args.export_gcode_only) or (args.export_gcode is not None)
    if export_requested:
        export_name = args.export_gcode or "commanded_motion.gcode"
        export_path = Path(export_name).expanduser()
        if not export_path.is_absolute():
            export_path = Path(runner.run_folder) / export_path

        export_result = export_gcode_from_trajectory(
            output_path=export_path,
            cal=cal,
            traj=traj,
            start_pose=start_pose,
            end_pose=end_pose,
            safe_approach_z=float(args.safe_approach_z),
            travel_feed=float(args.travel_feed),
            fine_approach_feed=float(args.fine_approach_feed),
            probe_feed=float(args.probe_feed),
            c_max_feed=float(args.c_max_feed),
            c_accel_time_s=float(args.c_accel_time),
            c_decel_time_s=float(args.c_decel_time),
            virtual_bbox=virtual_bbox,
            motion_meta=custom_meta,
            dwell_before_ms=int(args.dwell_before_ms),
            dwell_after_ms=int(args.dwell_after_ms),
            initial_sweep_wait_s=float(args.initial_sweep_wait_s),
            use_segment_feed_scheduler=(not bool(args.disable_segment_feed_scheduler)),
            include_startup_waits=bool(args.export_include_startup_waits),
        )
        print("\nExported smooth motion-only G-code; capture markers were not converted into waits/stops.")
        print(f"Exported G-code: {export_result['path']}")
        print(f"Commanded-motor CSV: {export_result['commanded_motor_positions_csv']}")
        print(f"Trajectory CSV: {export_result['trajectory_points_csv']}")
        print(f"Export metadata: {export_result['metadata_json']}")
        print(f"Exported command count: {export_result['command_count']}")
        print(f"Capture markers ignored during export: {export_result['capture_markers_ignored']}")
        print(f"BBox warnings: {len(export_result['bbox_warnings'])}")
        for msg in export_result["bbox_warnings"]:
            print(msg)
        if bool(args.export_gcode_only):
            print("Export-only mode enabled; skipped camera, robot, acquisition, and post-processing.")
            return

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
            c_feed=float(args.c_feed),
            c_max_feed=float(args.c_max_feed),
            c_accel_time_s=float(args.c_accel_time),
            c_decel_time_s=float(args.c_decel_time),
            virtual_bbox=virtual_bbox,
            dwell_before_ms=int(args.dwell_before_ms),
            dwell_after_ms=int(args.dwell_after_ms),
            preposition_c_only=False,
            use_segment_feed_scheduler=(not bool(args.disable_segment_feed_scheduler)),
            tracked_move_settle_s=float(args.tracked_move_settle_s),
            travel_move_settle_s=float(args.travel_move_settle_s),
            b_extra_settle_s=float(args.b_extra_settle_s),
            inter_command_delay_s=float(args.inter_command_delay_s),
            camera_flush_frames=int(args.camera_flush_frames),
            capture_at_start=bool(args.capture_at_start),
            initial_sweep_wait_s=float(args.initial_sweep_wait_s),
            settled_capture_mode=bool(args.settled_capture_mode),
            settled_capture_buffer_s=float(args.settled_capture_buffer_s),
            motion_accel_mm_s2=float(args.motion_accel_mm_s2),
            streaming_lookahead_s=float(args.streaming_lookahead_s),
        )

        print("\nFinal results:")
        print(results)

        if bool(args.enable_post):
            post_cmd = [
                sys.executable,
                str(script_dir / "calib_point_process.py"),
                "--project_dir",
                runner.run_folder,
                "--camera_calibration_file",
                str(Path(args.post_camera_calibration_file).expanduser()),
                "--checkerboard_reference_image",
                str(Path(args.post_checkerboard_reference_image).expanduser()),
                "--threshold",
                str(int(args.post_threshold)),
                "--tip_refine_mode",
                str(args.post_tip_refine_mode),
                "--tip_detection_mode",
                str(args.post_tip_detection_mode),
                "--tracked_tip_source",
                str(args.post_tracked_tip_source),
                "--red_tip_sat_min",
                str(int(args.post_red_tip_sat_min)),
                "--red_tip_val_min",
                str(int(args.post_red_tip_val_min)),
                "--red_tip_min_area_px",
                str(int(args.post_red_tip_min_area_px)),
                "--red_tip_morph_kernel",
                str(int(args.post_red_tip_morph_kernel)),
                "--red_tip_hue1_min",
                str(int(args.post_red_tip_hue1_min)),
                "--red_tip_hue1_max",
                str(int(args.post_red_tip_hue1_max)),
                "--red_tip_hue2_min",
                str(int(args.post_red_tip_hue2_min)),
                "--red_tip_hue2_max",
                str(int(args.post_red_tip_hue2_max)),
                "--red_tip_search_radius_px",
                str(float(args.post_red_tip_search_radius_px)),
                "--red_tip_local_min_area_px",
                str(int(args.post_red_tip_local_min_area_px)),
                "--red_tip_distance_weight",
                str(float(args.post_red_tip_distance_weight)),
                "--red_tip_min_circularity",
                str(float(args.post_red_tip_min_circularity)),
                "--red_tip_component_selection",
                str(args.post_red_tip_component_selection),
                "--red_tip_rgb_excess_min",
                str(int(args.post_red_tip_rgb_excess_min)),
            ]
            if bool(args.post_red_tip_use_rgb_excess):
                post_cmd.append("--red_tip_use_rgb_excess")
            else:
                post_cmd.append("--no_red_tip_use_rgb_excess")
            if bool(args.post_red_tip_debug_save_mask):
                post_cmd.append("--red_tip_debug_save_mask")
            else:
                post_cmd.append("--no_red_tip_debug_save_mask")
            if bool(args.post_save_plots):
                post_cmd.append("--save_plots")

            print("\nRunning post-processing:")
            print(" ".join(post_cmd))
            subprocess.run(post_cmd, check=True, cwd=str(script_dir))

    finally:
        try:
            runner.disconnect_camera()
        except Exception:
            pass


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Run a cyclic bounded-C fixed-tip tracked XYZ/B/C motion from the calibration JSON "
            "and capture images only at visible phases."
        )
    )

    # Run / folders
    ap.add_argument("--parent-directory", default=os.getcwd(), help="Parent folder for the run output.")
    ap.add_argument("--project-name", default=DEFAULT_PROJECT_NAME, help="Run folder name.")
    ap.add_argument("--allow-existing", action="store_true", default=DEFAULT_ALLOW_EXISTING,
                    help="Allow reuse of an existing run folder.")
    ap.add_argument("--add-date", action="store_true", default=DEFAULT_ADD_DATE,
                    help="Append timestamp to the run folder name.")
    ap.add_argument("--enable-post", action="store_true",
                    help="Run calib_point_process.py on the generated project directory after acquisition.")
    ap.add_argument("--export-gcode", nargs="?", const="commanded_motion.gcode", default=None,
                    help="Export a smooth motion-only G-code file. If a relative path is given, it is written inside the run folder.")
    ap.add_argument("--export-gcode-only", action="store_true",
                    help="Only export the smooth motion-only G-code file and skip robot/camera execution.")
    ap.add_argument("--export-include-startup-waits", action="store_true", default=False,
                    help="Include initial-sweep and dwell waits in exported G-code. Capture waits are never inserted.")

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
    ap.add_argument("--y-offset-fit", type=str, default=DEFAULT_Y_OFFSET_FIT,
                    choices=["avg_pchip", "avg_cubic", "pchip", "cubic", "legacy"],
                    help="Which calibration y-offset fit to use for DAQ motion.")
    ap.add_argument("--curve-set", type=str, default=DEFAULT_CURVE_SET,
                    help="Branch model source from curl_angle_specific_fit_models. Use 'auto' to prefer 0-90-0 in --b-0-to-90-only mode, otherwise 0-180-0; use 'global' for top-level fit_models_by_phase. Default: auto.")
    ap.add_argument("--allow-global-curve-set-fallback", action="store_true",
                    default=DEFAULT_ALLOW_GLOBAL_CURVE_SET_FALLBACK,
                    help="If the requested --curve-set is missing, fall back to global fit_models_by_phase. Enabled by default in this version for older calibration JSONs.")
    ap.add_argument("--motion-mode", type=str, default=DEFAULT_MOTION_MODE,
                    choices=["custom", "fast_routine"],
                    help="Use the legacy cyclic trajectory ('custom') or the requested fast routine preset.")
    ap.add_argument("--routine-repeats", type=int, default=DEFAULT_ROUTINE_REPEATS,
                    help="Repeat the fast routine this many times. Alias for cycle repeats in fast_routine mode.")

    # Fixed tip point
    ap.add_argument("--point-x", type=float, default=DEFAULT_POINT_X, help="Fixed tip X (Cartesian/world).")
    ap.add_argument("--point-y", type=float, default=DEFAULT_POINT_Y, help="Fixed tip Y (Cartesian/world).")
    ap.add_argument("--point-z", type=float, default=DEFAULT_POINT_Z, help="Fixed tip Z (Cartesian/world).")

    # Sign correction
    ap.add_argument(
        "--flip-rz-sign",
        action="store_true",
        default=DEFAULT_FLIP_RZ_SIGN,
        help="Multiply only the planar r/X offset by -1. Use this if your calibration file has a flipped X sign.",
    )

    # Cyclic sweep controls
    ap.add_argument("--cycle-repeats", type=int, default=DEFAULT_CYCLE_REPEATS,
                    help="How many full forward+return cycles to run.")
    ap.add_argument("--leg-move-steps", type=int, default=DEFAULT_LEG_MOVE_STEPS,
                    help="Tracked motion segments used for each one-way C leg.")
    ap.add_argument("--leg-capture-steps", type=int, default=DEFAULT_LEG_CAPTURE_STEPS,
                    help="Capture opportunity samples evaluated over each one-way C leg.")
    ap.add_argument("--sweep-tip-min-deg", type=float, default=DEFAULT_SWEEP_TIP_MIN_DEG,
                    help="Minimum tip angle during the cyclic B oscillation.")
    ap.add_argument("--sweep-tip-max-deg", type=float, default=DEFAULT_SWEEP_TIP_MAX_DEG,
                    help="Maximum tip angle during the cyclic B oscillation.")
    ap.add_argument("--b-0-to-90-only", action="store_true", default=DEFAULT_B_0_TO_90_ONLY,
                    help="Force the cyclic B/tip oscillation range to 0..90 deg.")
    ap.add_argument("--b-oscillations-per-sweep", type=float, default=DEFAULT_B_OSCILLATIONS_PER_SWEEP,
                    help="How many B oscillations occur during one one-way sweep of C. Use 1 for one full curl or uncurl per sweep.")
    ap.add_argument("--b-phase-offset-deg", type=float, default=DEFAULT_B_PHASE_OFFSET_DEG,
                    help="Phase offset for the B oscillation. -90 starts at tip_min.")
    ap.add_argument("--c-boundary-ease-frac", type=float, default=DEFAULT_C_BOUNDARY_EASE_FRAC,
                    help="Small edge fraction for C boundary easing, in [0, 0.49].")
    ap.add_argument("--custom-inverse-samples", type=int, default=DEFAULT_CUSTOM_INV_SAMPLES,
                    help="Dense sampling count used for numeric tip-angle -> B inversion.")
    ap.add_argument("--phase-transition-steps", type=int, default=DEFAULT_PHASE_TRANSITION_STEPS,
                    help="Non-recorded smoothing steps inserted when switching between curl/pull and uncurl/release.")
    ap.add_argument("--c-two-way-sweeps-per-b-oscillation", type=int,
                    default=DEFAULT_C_TWO_WAY_SWEEPS_PER_B_OSCILLATION,
                    help="For monotonic B curl/uncurl mode, how many back-and-forth C sweep groups are spread across one B oscillation. 2 gives -360->360->-360->360 during one curl.")
    ap.add_argument("--disable-branch-conditioning", action="store_true",
                    help="Disable hidden endpoint conditioning for truncated visible ranges such as --b-0-to-90-only with --curve-set 0-180-0.")
    ap.add_argument("--branch-conditioning-tip-max-deg", type=float,
                    default=DEFAULT_BRANCH_CONDITIONING_TIP_MAX_DEG,
                    help="Endpoint angle used for hidden hysteresis conditioning when --curve-set is 0-180-0 and the visible sweep is truncated. Default: 180.")

    # Capture visibility controls
    ap.add_argument("--capture-tip-full-visible-min-deg", type=float,
                    default=DEFAULT_CAPTURE_TIP_FULL_VISIBLE_MIN_DEG,
                    help="Below/within this tip range, capture is allowed at all C.")
    ap.add_argument("--capture-tip-full-visible-max-deg", type=float,
                    default=DEFAULT_CAPTURE_TIP_FULL_VISIBLE_MAX_DEG,
                    help="Below/within this tip range, capture is allowed at all C.")
    ap.add_argument("--c-visible-win1-min-deg", type=float, default=DEFAULT_C_VISIBLE_WIN1_MIN)
    ap.add_argument("--c-visible-win1-max-deg", type=float, default=DEFAULT_C_VISIBLE_WIN1_MAX)
    ap.add_argument("--c-visible-win2-min-deg", type=float, default=DEFAULT_C_VISIBLE_WIN2_MIN)
    ap.add_argument("--c-visible-win2-max-deg", type=float, default=DEFAULT_C_VISIBLE_WIN2_MAX)

    # Feedrates
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED)
    ap.add_argument("--fine-approach-feed", type=float, default=DEFAULT_FINE_APPROACH_FEED)
    ap.add_argument("--probe-feed", type=float, default=DEFAULT_PROBE_FEED)
    ap.add_argument("--c-feed", type=float, default=DEFAULT_C_FEED)

    # Tracked C cap + feed scheduler
    ap.add_argument("--c-max-feed", type=float, default=DEFAULT_C_MAX_FEED)
    ap.add_argument("--c-accel-time", type=float, default=DEFAULT_C_ACCEL_TIME_S)
    ap.add_argument("--c-decel-time", type=float, default=DEFAULT_C_DECEL_TIME_S)
    ap.add_argument("--motion-accel-mm-s2", type=float, default=DEFAULT_MOTION_ACCEL_MM_S2,
                    help="Acceleration used for settled-capture move-time estimates.")
    ap.add_argument("--disable-segment-feed-scheduler", action="store_true",
                    help="Disable per-segment feed scheduling.")

    # Optional waits / capture behavior
    ap.add_argument("--dwell-before-ms", type=int, default=DEFAULT_DWELL_BEFORE_MS)
    ap.add_argument("--dwell-after-ms", type=int, default=DEFAULT_DWELL_AFTER_MS)
    ap.add_argument("--initial-sweep-wait-s", type=float, default=DEFAULT_INITIAL_SWEEP_WAIT_S,
                    help="Hold time after landing on the first tracked point before queuing the sweep.")
    ap.add_argument("--tracked-move-settle-s", type=float, default=DEFAULT_TRACKED_MOVE_SETTLE_S,
                    help="Extra settle time after each tracked move, before capture.")
    ap.add_argument("--travel-move-settle-s", type=float, default=DEFAULT_TRAVEL_MOVE_SETTLE_S,
                    help="Extra settle time after travel moves.")
    ap.add_argument("--b-extra-settle-s", type=float, default=DEFAULT_B_EXTRA_SETTLE_S,
                    help="Additional hold after each tracked move to let the B-axis mechanically settle.")
    ap.add_argument("--inter-command-delay-s", type=float, default=DEFAULT_INTER_COMMAND_DELAY_S,
                    help="Small delay between queued tracked commands.")
    ap.add_argument("--capture-at-start", action="store_true", default=DEFAULT_CAPTURE_AT_START,
                    help="Legacy option; only used if the first trajectory point is marked for acquisition.")
    ap.add_argument("--settled-capture-mode", action="store_true", default=DEFAULT_SETTLED_CAPTURE_MODE,
                    help="Stream commands without M400 waits; wait only at capture points using estimated timing.")
    ap.add_argument("--settled-capture-buffer-s", type=float, default=DEFAULT_SETTLED_CAPTURE_BUFFER_S,
                    help="Extra idle time after estimated motion completion before each capture.")
    ap.add_argument("--streaming-lookahead-s", type=float, default=DEFAULT_STREAMING_LOOKAHEAD_S,
                    help="How early to send the next non-capture tracked command before the previous one is estimated to finish.")

    # Post-processing
    ap.add_argument("--post-camera-calibration-file", type=str, default=DEFAULT_POST_CAMERA_CALIBRATION_FILE,
                    help="Camera calibration file passed to calib_point_process.py.")
    ap.add_argument("--post-checkerboard-reference-image", type=str,
                    default=DEFAULT_POST_CHECKERBOARD_REFERENCE_IMAGE,
                    help="Checkerboard reference image passed to calib_point_process.py.")
    ap.add_argument("--post-threshold", type=int, default=DEFAULT_POST_THRESHOLD,
                    help="Threshold passed to calib_point_process.py.")
    ap.add_argument("--post-tip-refine-mode", type=str, default=DEFAULT_POST_TIP_REFINE_MODE,
                    help="Tip refinement mode passed to calib_point_process.py.")
    ap.add_argument("--post-tip-detection-mode", type=str, default=DEFAULT_POST_TIP_DETECTION_MODE,
                    choices=["classical", "red_dot", "auto_red_dot"],
                    help="Tip detection mode passed to calib_point_process.py.")
    ap.add_argument("--post-tracked-tip-source", type=str, default=DEFAULT_POST_TRACKED_TIP_SOURCE,
                    choices=["auto", "coarse", "selected", "cnn"],
                    help="Tracked tip source passed to calib_point_process.py. Use auto or selected for red-dot post-processing.")
    ap.add_argument("--post-red-tip-sat-min", type=int, default=DEFAULT_POST_RED_TIP_SAT_MIN)
    ap.add_argument("--post-red-tip-val-min", type=int, default=DEFAULT_POST_RED_TIP_VAL_MIN)
    ap.add_argument("--post-red-tip-min-area-px", type=int, default=DEFAULT_POST_RED_TIP_MIN_AREA_PX)
    ap.add_argument("--post-red-tip-morph-kernel", type=int, default=DEFAULT_POST_RED_TIP_MORPH_KERNEL)
    ap.add_argument("--post-red-tip-hue1-min", type=int, default=DEFAULT_POST_RED_TIP_HUE1_MIN)
    ap.add_argument("--post-red-tip-hue1-max", type=int, default=DEFAULT_POST_RED_TIP_HUE1_MAX)
    ap.add_argument("--post-red-tip-hue2-min", type=int, default=DEFAULT_POST_RED_TIP_HUE2_MIN)
    ap.add_argument("--post-red-tip-hue2-max", type=int, default=DEFAULT_POST_RED_TIP_HUE2_MAX)
    ap.add_argument("--post-red-tip-search-radius-px", type=float, default=DEFAULT_POST_RED_TIP_SEARCH_RADIUS_PX)
    ap.add_argument("--post-red-tip-local-min-area-px", type=int, default=DEFAULT_POST_RED_TIP_LOCAL_MIN_AREA_PX)
    ap.add_argument("--post-red-tip-distance-weight", type=float, default=DEFAULT_POST_RED_TIP_DISTANCE_WEIGHT)
    ap.add_argument("--post-red-tip-min-circularity", type=float, default=DEFAULT_POST_RED_TIP_MIN_CIRCULARITY)
    ap.add_argument("--post-red-tip-component-selection", type=str, default=DEFAULT_POST_RED_TIP_COMPONENT_SELECTION,
                    choices=["largest", "nearest", "nearest_largest"])
    ap.add_argument("--post-red-tip-use-rgb-excess", dest="post_red_tip_use_rgb_excess",
                    action="store_true", default=DEFAULT_POST_RED_TIP_USE_RGB_EXCESS)
    ap.add_argument("--no-post-red-tip-use-rgb-excess", dest="post_red_tip_use_rgb_excess",
                    action="store_false")
    ap.add_argument("--post-red-tip-rgb-excess-min", type=int, default=DEFAULT_POST_RED_TIP_RGB_EXCESS_MIN)
    ap.add_argument("--post-red-tip-debug-save-mask", dest="post_red_tip_debug_save_mask",
                    action="store_true", default=DEFAULT_POST_RED_TIP_DEBUG_SAVE_MASK)
    ap.add_argument("--no-post-red-tip-debug-save-mask", dest="post_red_tip_debug_save_mask",
                    action="store_false")
    ap.add_argument("--post-save-plots", dest="post_save_plots", action="store_true", default=True,
                    help="Pass --save_plots to calib_point_process.py.")
    ap.add_argument("--no-post-save-plots", dest="post_save_plots", action="store_false",
                    help="Do not pass --save_plots to calib_point_process.py.")

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
