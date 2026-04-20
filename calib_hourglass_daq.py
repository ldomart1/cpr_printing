"""
Standalone hourglass-tracking acquisition script.

What it does
------------
- Loads the calibration JSON.
- Builds a closed hourglass trajectory in tip-space using two circular half-arcs and four
  straight diagonal connectors.
- Geometry is controlled by the vertical gap between the two arcs, the circle diameter,
  and the horizontal waist gap between the left/right diagonals at their closest point.
- Slight arc overtravel is applied so the diagonal connectors attach beyond the strict
  semicircle endpoints, which produces a cleaner hourglass silhouette.
- Converts the desired tip trajectory into tracked stage XYZ + B motion commands using the
  same calibration / capture / robot plumbing as the uploaded circle script.
- Adds a --plot-only CLI flag that saves the planned trajectory plot / CSV / metadata and
  exits before any robot motion, camera capture, or post-processing.
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
DEFAULT_PROJECT_NAME = "Hourglass_Tracking_Run"
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
DEFAULT_POST_SCRIPT = "calib_hourglass_process.py"
DEFAULT_POST_TRACKED_TIP_SOURCE = "cnn"

DEFAULT_POST_TIP_REFINER_MODEL = (
    "Test_Calibration_2026-04-07_02_daq/"
    "processed_image_data_folder/tip_refinement_model/best_tip_refiner.pt"
)

DEFAULT_SAFE_APPROACH_Z = -135.0

DEFAULT_START_X = 100.0
DEFAULT_START_Y = 52.0
DEFAULT_START_Z = -135.0
DEFAULT_START_B = 0.0
DEFAULT_START_C = 0.0

DEFAULT_END_X = 100.0
DEFAULT_END_Y = 52.0
DEFAULT_END_Z = -135.0
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
CIRCLE_CENTER_Z = -130.0

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
# Hourglass planner
# =========================

HOURGLASS_CENTER_X = CIRCLE_CENTER_X
HOURGLASS_CENTER_Y = CIRCLE_CENTER_Y
HOURGLASS_CENTER_Z = CIRCLE_CENTER_Z
DEFAULT_ARC_VERTICAL_GAP_MM = 18.0
DEFAULT_CIRCLE_DIAMETER_MM = 18.0
DEFAULT_MIDDLE_GAP_MM = 6.0
DEFAULT_ARC_OVERTRAVEL_DEG = 20.0
DEFAULT_SAMPLES_PER_ARC = 180
DEFAULT_SAMPLES_PER_DIAGONAL = 120

RECORDED_PHASES = {
    "top_arc_pull",
    "right_diag_upper_release",
    "right_diag_lower_pull",
    "bottom_arc_release",
    "left_diag_lower_pull",
    "left_diag_upper_release",
}
TRANSITION_PHASES = {
    "final_recenter",
}


def is_recorded_phase(phase: str) -> bool:
    phase_name = str(phase)
    if phase_name in RECORDED_PHASES:
        return True
    if phase_name.endswith("_start"):
        return phase_name[:-6] in RECORDED_PHASES
    return False


def choose_segment_b_endpoint(
    cal: Calibration,
    b_lo: float,
    b_hi: float,
    pull_phase: str,
    release_phase: str,
    segment_b_override: Optional[float] = None,
) -> float:
    common_lo, common_hi = common_b_window_for_pull_release(
        cal=cal,
        b_lo_user=b_lo,
        b_hi_user=b_hi,
        pull_phase=pull_phase,
        release_phase=release_phase,
    )

    if not (common_lo <= 0.0 <= common_hi):
        raise RuntimeError(
            f"B=0 is not inside the common pull/release range [{common_lo:.3f}, {common_hi:.3f}]."
        )

    if segment_b_override is not None:
        b_ext = float(segment_b_override)
        if not (common_lo <= b_ext <= common_hi):
            raise ValueError(
                f"--segment-b={b_ext:.3f} is outside the common pull/release range "
                f"[{common_lo:.3f}, {common_hi:.3f}]"
            )
        if abs(b_ext) < 1e-9:
            raise ValueError("--segment-b must differ from 0 so the tracked motion is non-degenerate.")
        return float(b_ext)

    candidates = [float(v) for v in (common_lo, common_hi) if abs(float(v)) > 1e-9]
    if not candidates:
        raise RuntimeError(
            "Could not choose a non-zero B endpoint for the hourglass path inside the common pull/release window."
        )
    return float(max(candidates, key=lambda v: abs(v)))


def make_arc_points(
    center_x: float,
    center_z: float,
    radius: float,
    theta_start_deg: float,
    theta_end_deg: float,
    n: int,
) -> np.ndarray:
    theta = np.deg2rad(np.linspace(float(theta_start_deg), float(theta_end_deg), max(2, int(n)), dtype=float))
    x = float(center_x) + float(radius) * np.cos(theta)
    z = float(center_z) + float(radius) * np.sin(theta)
    return np.column_stack([x, z]).astype(float)


def make_line_points(p0: np.ndarray, p1: np.ndarray, n: int) -> np.ndarray:
    p0 = np.asarray(p0, dtype=float).reshape(2)
    p1 = np.asarray(p1, dtype=float).reshape(2)
    t = np.linspace(0.0, 1.0, max(2, int(n)), dtype=float)
    pts = (1.0 - t)[:, None] * p0[None, :] + t[:, None] * p1[None, :]
    return np.asarray(pts, dtype=float)


def build_recorded_segment_from_tip_targets(
    cal: Calibration,
    flip_rz_sign: bool,
    center_y: float,
    tip_xz_points: np.ndarray,
    b_start: float,
    b_end: float,
    c_state: float,
    motion_phase: str,
    phase_name: str,
    move_feed_start: float,
    move_feed_rest: float,
) -> Tuple[List[CommandPoint], Dict[str, float]]:
    c_state = assert_c_in_safe_range("c_state", c_state)
    tip_xz_points = np.asarray(tip_xz_points, dtype=float)
    if tip_xz_points.ndim != 2 or tip_xz_points.shape[1] != 2 or tip_xz_points.shape[0] < 2:
        raise ValueError("tip_xz_points must be an Nx2 array with at least 2 rows.")

    b_values = np.linspace(float(b_start), float(b_end), int(tip_xz_points.shape[0]), dtype=float)
    pts: List[CommandPoint] = []
    stage_x_trace: List[float] = []
    stage_y_trace: List[float] = []
    stage_z_trace: List[float] = []

    for i, ((tip_x, tip_z), b_i) in enumerate(zip(tip_xz_points, b_values)):
        tip_target = np.array([float(tip_x), float(center_y), float(tip_z)], dtype=float)
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
            raise RuntimeError("Internal tip-tracking consistency check failed in hourglass planner.")

        stage_x_trace.append(float(stage_xyz[0]))
        stage_y_trace.append(float(stage_xyz[1]))
        stage_z_trace.append(float(stage_xyz[2]))
        pts.append(
            CommandPoint(
                phase=f"{phase_name}_start" if i == 0 else str(phase_name),
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

    meta = {
        "phase_name": str(phase_name),
        "motion_phase": str(motion_phase),
        "b_start": float(b_start),
        "b_end": float(b_end),
        "stage_x_min": float(min(stage_x_trace)),
        "stage_x_max": float(max(stage_x_trace)),
        "stage_x_span": float(max(stage_x_trace) - min(stage_x_trace)),
        "stage_y_min": float(min(stage_y_trace)),
        "stage_y_max": float(max(stage_y_trace)),
        "stage_z_min": float(min(stage_z_trace)),
        "stage_z_max": float(max(stage_z_trace)),
        "tip_x_min": float(min(p.tip_x for p in pts)),
        "tip_x_max": float(max(p.tip_x for p in pts)),
        "tip_z_min": float(min(p.tip_z for p in pts)),
        "tip_z_max": float(max(p.tip_z for p in pts)),
    }
    return pts, meta


def _tip_offset_x_for_phase(
    cal: Calibration,
    b: float,
    c_deg: float,
    motion_phase: str,
    flip_rz_sign: bool,
) -> float:
    return float(
        tip_offset_xyz_physical(
            cal,
            float(b),
            float(c_deg),
            flip_rz_sign=flip_rz_sign,
            motion_phase=motion_phase,
        )[0]
    )



def _dense_b_and_x_for_phase(
    cal: Calibration,
    b_start: float,
    b_end: float,
    c_deg: float,
    motion_phase: str,
    flip_rz_sign: bool,
    n_dense: int = 4001,
) -> Tuple[np.ndarray, np.ndarray]:
    b_dense = np.linspace(float(b_start), float(b_end), max(64, int(n_dense)), dtype=float)
    x_dense = np.asarray([
        _tip_offset_x_for_phase(
            cal,
            float(bv),
            float(c_deg),
            motion_phase=str(motion_phase),
            flip_rz_sign=flip_rz_sign,
        )
        for bv in b_dense
    ], dtype=float)
    return b_dense, x_dense



def _invert_b_from_x_targets(
    x_targets: np.ndarray,
    b_dense: np.ndarray,
    x_dense: np.ndarray,
) -> np.ndarray:
    x_targets = np.asarray(x_targets, dtype=float).reshape(-1)
    b_dense = np.asarray(b_dense, dtype=float).reshape(-1)
    x_dense = np.asarray(x_dense, dtype=float).reshape(-1)
    if b_dense.size != x_dense.size or b_dense.size < 2:
        raise ValueError('Dense B/X samples are invalid for inversion.')

    order = np.argsort(x_dense)
    x_sorted = x_dense[order]
    b_sorted = b_dense[order]

    keep = np.ones_like(x_sorted, dtype=bool)
    keep[1:] = np.abs(np.diff(x_sorted)) > 1e-9
    x_unique = x_sorted[keep]
    b_unique = b_sorted[keep]
    if x_unique.size < 2:
        raise RuntimeError('Could not invert B from X because the sampled radial X curve is degenerate.')

    x_min = float(np.min(x_unique))
    x_max = float(np.max(x_unique))
    if np.min(x_targets) < x_min - 1e-6 or np.max(x_targets) > x_max + 1e-6:
        raise RuntimeError(
            f'Target X values [{np.min(x_targets):.6f}, {np.max(x_targets):.6f}] are outside the reachable range '
            f'[{x_min:.6f}, {x_max:.6f}] for this motion phase.'
        )

    return np.interp(x_targets, x_unique, b_unique).astype(float)



def build_fixed_x_segment_from_desired_tip_targets(
    cal: Calibration,
    flip_rz_sign: bool,
    center_y: float,
    desired_tip_xz_points: np.ndarray,
    stage_x_const: float,
    b_start: float,
    b_end: float,
    c_state: float,
    motion_phase: str,
    phase_name: str,
    move_feed_start: float,
    move_feed_rest: float,
) -> Tuple[List[CommandPoint], Dict[str, float]]:
    c_state = assert_c_in_safe_range('c_state', c_state)
    desired_tip_xz_points = np.asarray(desired_tip_xz_points, dtype=float)
    if desired_tip_xz_points.ndim != 2 or desired_tip_xz_points.shape[1] != 2 or desired_tip_xz_points.shape[0] < 2:
        raise ValueError('desired_tip_xz_points must be an Nx2 array with at least 2 rows.')

    b_dense, x_dense = _dense_b_and_x_for_phase(
        cal=cal,
        b_start=float(b_start),
        b_end=float(b_end),
        c_deg=float(c_state),
        motion_phase=str(motion_phase),
        flip_rz_sign=flip_rz_sign,
    )
    target_offsets_x = desired_tip_xz_points[:, 0] - float(stage_x_const)
    b_values = _invert_b_from_x_targets(target_offsets_x, b_dense=b_dense, x_dense=x_dense)

    pts: List[CommandPoint] = []
    stage_x_trace: List[float] = []
    stage_y_trace: List[float] = []
    stage_z_trace: List[float] = []
    tip_x_trace: List[float] = []
    tip_y_trace: List[float] = []
    tip_z_trace: List[float] = []
    x_error_trace: List[float] = []

    for i, ((desired_tip_x, desired_tip_z), b_i) in enumerate(zip(desired_tip_xz_points, b_values)):
        offset_xyz = tip_offset_xyz_physical(
            cal,
            float(b_i),
            float(c_state),
            flip_rz_sign=flip_rz_sign,
            motion_phase=str(motion_phase),
        )
        stage_x = float(stage_x_const)
        stage_y = float(center_y) - float(offset_xyz[1])
        stage_z = float(desired_tip_z) - float(offset_xyz[2])

        tip_check = np.array([stage_x, stage_y, stage_z], dtype=float) + offset_xyz
        actual_tip_x = float(tip_check[0])
        actual_tip_y = float(tip_check[1])
        actual_tip_z = float(tip_check[2])

        stage_x_trace.append(stage_x)
        stage_y_trace.append(stage_y)
        stage_z_trace.append(stage_z)
        tip_x_trace.append(actual_tip_x)
        tip_y_trace.append(actual_tip_y)
        tip_z_trace.append(actual_tip_z)
        x_error_trace.append(actual_tip_x - float(desired_tip_x))

        pts.append(
            CommandPoint(
                phase=f'{phase_name}_start' if i == 0 else str(phase_name),
                x=stage_x,
                y=stage_y,
                z=stage_z,
                b=float(b_i),
                c=float(c_state),
                feed=float(move_feed_start if i == 0 else move_feed_rest),
                tip_x=actual_tip_x,
                tip_y=actual_tip_y,
                tip_z=actual_tip_z,
                motion_phase=str(motion_phase),
            )
        )

    meta = {
        'phase_name': str(phase_name),
        'motion_phase': str(motion_phase),
        'b_start': float(b_values[0]),
        'b_end': float(b_values[-1]),
        'stage_x_min': float(min(stage_x_trace)),
        'stage_x_max': float(max(stage_x_trace)),
        'stage_x_span': float(max(stage_x_trace) - min(stage_x_trace)),
        'stage_y_min': float(min(stage_y_trace)),
        'stage_y_max': float(max(stage_y_trace)),
        'stage_z_min': float(min(stage_z_trace)),
        'stage_z_max': float(max(stage_z_trace)),
        'tip_x_min': float(min(tip_x_trace)),
        'tip_x_max': float(max(tip_x_trace)),
        'tip_z_min': float(min(tip_z_trace)),
        'tip_z_max': float(max(tip_z_trace)),
        'max_abs_tip_x_error_mm': float(np.max(np.abs(np.asarray(x_error_trace, dtype=float)))),
        'mean_abs_tip_x_error_mm': float(np.mean(np.abs(np.asarray(x_error_trace, dtype=float)))),
    }
    return pts, meta



def build_hourglass_command_sequence(
    cal: Calibration,
    flip_rz_sign: bool,
    center_x: float,
    center_y: float,
    center_z: float,
    arc_vertical_gap_mm: float,
    circle_diameter_mm: float,
    middle_gap_mm: float,
    arc_overtravel_deg: float,
    samples_per_arc: int,
    samples_per_diagonal: int,
    b_lo: float,
    b_hi: float,
    c0_deg: float,
    jog_feed: float,
    print_feed_b: float,
    transition_feed: float,
    segment_b_override: Optional[float] = None,
    final_recenter: bool = True,
) -> Tuple[List[CommandPoint], dict]:
    c0_deg = assert_c_in_safe_range('c0_deg', c0_deg)

    radius_req = 0.5 * float(circle_diameter_mm)
    if radius_req <= 0.0:
        raise ValueError('--circle-diameter-mm must be positive.')
    if float(arc_vertical_gap_mm) < 0.0:
        raise ValueError('--arc-vertical-gap-mm must be non-negative.')
    if float(middle_gap_mm) < 0.0:
        raise ValueError('--middle-gap-mm must be non-negative.')
    if float(middle_gap_mm) >= 2.0 * float(radius_req):
        raise ValueError('--middle-gap-mm must be smaller than the circle diameter so the diagonals converge inward.')

    phi_deg = float(max(0.0, min(80.0, float(arc_overtravel_deg))))
    n_arc = max(2, int(samples_per_arc))
    n_diag = max(2, int(samples_per_diagonal))

    pull_phase = resolve_phase_name(cal, 'pull')
    release_phase = resolve_phase_name(cal, 'release')
    b_ext = choose_segment_b_endpoint(
        cal=cal,
        b_lo=b_lo,
        b_hi=b_hi,
        pull_phase=pull_phase,
        release_phase=release_phase,
        segment_b_override=segment_b_override,
    )

    b_pull_dense, x_pull_dense = _dense_b_and_x_for_phase(
        cal=cal,
        b_start=0.0,
        b_end=float(b_ext),
        c_deg=float(c0_deg),
        motion_phase=pull_phase,
        flip_rz_sign=flip_rz_sign,
    )
    b_release_dense, x_release_dense = _dense_b_and_x_for_phase(
        cal=cal,
        b_start=float(b_ext),
        b_end=0.0,
        c_deg=float(c0_deg),
        motion_phase=release_phase,
        flip_rz_sign=flip_rz_sign,
    )

    common_x_min_off = max(float(np.min(x_pull_dense)), float(np.min(x_release_dense)))
    common_x_max_off = min(float(np.max(x_pull_dense)), float(np.max(x_release_dense)))
    if common_x_min_off >= common_x_max_off:
        raise RuntimeError(
            'No common fixed-X radial range is available between pull and release phases at C=0. '
            f'pull=[{float(np.min(x_pull_dense)):.6f}, {float(np.max(x_pull_dense)):.6f}], '
            f'release=[{float(np.min(x_release_dense)):.6f}, {float(np.max(x_release_dense)):.6f}]'
        )
    common_half_span = 0.5 * (common_x_max_off - common_x_min_off)
    common_mid_off = 0.5 * (common_x_max_off + common_x_min_off)
    stage_x_const = float(center_x) - common_mid_off

    radius_nom = float(radius_req)
    top_center_z_nom = float(center_z) + 0.5 * float(arc_vertical_gap_mm) + float(radius_nom)
    bottom_center_z_nom = float(center_z) - 0.5 * float(arc_vertical_gap_mm) - float(radius_nom)

    top_arc_nom = make_arc_points(
        center_x=float(center_x),
        center_z=float(top_center_z_nom),
        radius=float(radius_nom),
        theta_start_deg=180.0 + phi_deg,
        theta_end_deg=-phi_deg,
        n=n_arc,
    )
    bottom_arc_nom = make_arc_points(
        center_x=float(center_x),
        center_z=float(bottom_center_z_nom),
        radius=float(radius_nom),
        theta_start_deg=phi_deg,
        theta_end_deg=-(180.0 + phi_deg),
        n=n_arc,
    )
    top_left_nom = np.asarray(top_arc_nom[0], dtype=float)
    top_right_nom = np.asarray(top_arc_nom[-1], dtype=float)
    bottom_right_nom = np.asarray(bottom_arc_nom[0], dtype=float)
    bottom_left_nom = np.asarray(bottom_arc_nom[-1], dtype=float)
    waist_right_nom = np.array([float(center_x) + 0.5 * float(middle_gap_mm), float(center_z)], dtype=float)
    waist_left_nom = np.array([float(center_x) - 0.5 * float(middle_gap_mm), float(center_z)], dtype=float)

    nominal_points = np.vstack([
        top_arc_nom,
        bottom_arc_nom,
        waist_right_nom[None, :],
        waist_left_nom[None, :],
    ])
    nominal_half_width = float(np.max(np.abs(nominal_points[:, 0] - float(center_x))))
    if nominal_half_width <= 1e-9:
        raise RuntimeError('Nominal hourglass width is degenerate.')
    scale_to_fit = min(1.0, common_half_span / nominal_half_width)

    scaled_radius = float(radius_req) * scale_to_fit
    scaled_arc_vertical_gap = float(arc_vertical_gap_mm) * scale_to_fit
    scaled_middle_gap = float(middle_gap_mm) * scale_to_fit
    top_center_z = float(center_z) + 0.5 * float(scaled_arc_vertical_gap) + float(scaled_radius)
    bottom_center_z = float(center_z) - 0.5 * float(scaled_arc_vertical_gap) - float(scaled_radius)

    top_arc = make_arc_points(
        center_x=float(center_x),
        center_z=float(top_center_z),
        radius=float(scaled_radius),
        theta_start_deg=180.0 + phi_deg,
        theta_end_deg=-phi_deg,
        n=n_arc,
    )
    bottom_arc = make_arc_points(
        center_x=float(center_x),
        center_z=float(bottom_center_z),
        radius=float(scaled_radius),
        theta_start_deg=phi_deg,
        theta_end_deg=-(180.0 + phi_deg),
        n=n_arc,
    )
    top_left = np.asarray(top_arc[0], dtype=float)
    top_right = np.asarray(top_arc[-1], dtype=float)
    bottom_right = np.asarray(bottom_arc[0], dtype=float)
    bottom_left = np.asarray(bottom_arc[-1], dtype=float)
    waist_right = np.array([float(center_x) + 0.5 * float(scaled_middle_gap), float(center_z)], dtype=float)
    waist_left = np.array([float(center_x) - 0.5 * float(scaled_middle_gap), float(center_z)], dtype=float)

    right_diag_upper = make_line_points(top_right, waist_right, n_diag)
    right_diag_lower = make_line_points(waist_right, bottom_right, n_diag)
    left_diag_lower = make_line_points(bottom_left, waist_left, n_diag)
    left_diag_upper = make_line_points(waist_left, top_left, n_diag)

    segments = [
        ('top_arc_pull', top_arc, pull_phase, 0.0, b_ext, float(jog_feed), float(print_feed_b)),
        ('right_diag_upper_release', right_diag_upper, release_phase, b_ext, 0.0, float(transition_feed), float(print_feed_b)),
        ('right_diag_lower_pull', right_diag_lower, pull_phase, 0.0, b_ext, float(transition_feed), float(print_feed_b)),
        ('bottom_arc_release', bottom_arc, release_phase, b_ext, 0.0, float(transition_feed), float(print_feed_b)),
        ('left_diag_lower_pull', left_diag_lower, pull_phase, 0.0, b_ext, float(transition_feed), float(print_feed_b)),
        ('left_diag_upper_release', left_diag_upper, release_phase, b_ext, 0.0, float(transition_feed), float(print_feed_b)),
    ]

    sequence: List[CommandPoint] = []
    segment_meta: Dict[str, Dict[str, float]] = {}
    for seg_idx, (phase_name, desired_tip_xz, motion_phase, b_start, b_end, move_feed_start, move_feed_rest) in enumerate(segments):
        pts, meta_seg = build_fixed_x_segment_from_desired_tip_targets(
            cal=cal,
            flip_rz_sign=flip_rz_sign,
            center_y=float(center_y),
            desired_tip_xz_points=np.asarray(desired_tip_xz, dtype=float),
            stage_x_const=float(stage_x_const),
            b_start=float(b_start),
            b_end=float(b_end),
            c_state=float(c0_deg),
            motion_phase=str(motion_phase),
            phase_name=str(phase_name),
            move_feed_start=float(move_feed_start),
            move_feed_rest=float(move_feed_rest),
        )
        segment_meta[str(phase_name)] = meta_seg
        if seg_idx > 0:
            pts = pts[1:]
        sequence.extend(pts)

    if not sequence:
        raise RuntimeError('Hourglass command sequence is empty.')

    start_tip = np.array([sequence[0].tip_x, sequence[0].tip_y, sequence[0].tip_z], dtype=float)
    end_tip = np.array([sequence[-1].tip_x, sequence[-1].tip_y, sequence[-1].tip_z], dtype=float)
    closure_error_before_final = {
        'dx': float(end_tip[0] - start_tip[0]),
        'dy': float(end_tip[1] - start_tip[1]),
        'dz': float(end_tip[2] - start_tip[2]),
        'dist': float(np.linalg.norm(end_tip - start_tip)),
    }

    final_recenter_cp = None
    if bool(final_recenter) and closure_error_before_final['dist'] > 1e-9:
        offset0 = tip_offset_xyz_physical(
            cal,
            0.0,
            float(c0_deg),
            flip_rz_sign=flip_rz_sign,
            motion_phase=release_phase,
        )
        final_recenter_cp = CommandPoint(
            phase='final_recenter',
            x=float(stage_x_const),
            y=float(start_tip[1] - offset0[1]),
            z=float(start_tip[2] - offset0[2]),
            b=0.0,
            c=float(c0_deg),
            feed=float(transition_feed),
            tip_x=float(stage_x_const + offset0[0]),
            tip_y=float(start_tip[1]),
            tip_z=float(start_tip[2]),
            motion_phase=str(release_phase),
        )
        sequence.append(final_recenter_cp)

    assert_all_command_c_safe(sequence)

    x_stage_spans = {k: float(v['stage_x_span']) for k, v in segment_meta.items()}
    max_stage_x_span = float(max(x_stage_spans.values())) if x_stage_spans else 0.0

    meta = {
        'pull_phase': str(pull_phase),
        'release_phase': str(release_phase),
        'segment_b': float(b_ext),
        'center_x': float(center_x),
        'center_y': float(center_y),
        'center_z': float(center_z),
        'arc_vertical_gap_mm_requested': float(arc_vertical_gap_mm),
        'circle_diameter_mm_requested': float(circle_diameter_mm),
        'middle_gap_mm_requested': float(middle_gap_mm),
        'arc_vertical_gap_mm_used': float(scaled_arc_vertical_gap),
        'circle_diameter_mm_used': float(2.0 * scaled_radius),
        'circle_radius_mm_used': float(scaled_radius),
        'middle_gap_mm_used': float(scaled_middle_gap),
        'uniform_scale_to_fit_radial_range': float(scale_to_fit),
        'scaled_to_fit_radial_range': bool(scale_to_fit < 0.999999),
        'arc_overtravel_deg': float(phi_deg),
        'top_arc_center_z': float(top_center_z),
        'bottom_arc_center_z': float(bottom_center_z),
        'c0_deg': float(c0_deg),
        'flip_rz_sign': bool(flip_rz_sign),
        'samples_per_arc': int(n_arc),
        'samples_per_diagonal': int(n_diag),
        'x_stage_const_after_start': float(stage_x_const),
        'common_radial_x_offset_range_pull_release': [float(common_x_min_off), float(common_x_max_off)],
        'common_radial_half_span_mm': float(common_half_span),
        'nominal_half_width_mm': float(nominal_half_width),
        'segment_meta': segment_meta,
        'connection_points': {
            'top_left': top_left.tolist(),
            'top_right': top_right.tolist(),
            'waist_right': waist_right.tolist(),
            'bottom_right': bottom_right.tolist(),
            'bottom_left': bottom_left.tolist(),
            'waist_left': waist_left.tolist(),
        },
        'x_stage_min': float(min(p.x for p in sequence)),
        'x_stage_max': float(max(p.x for p in sequence)),
        'x_stage_span': float(max(p.x for p in sequence) - min(p.x for p in sequence)),
        'max_segment_stage_x_span': float(max_stage_x_span),
        'y_stage_min': float(min(p.y for p in sequence)),
        'y_stage_max': float(max(p.y for p in sequence)),
        'z_stage_min': float(min(p.z for p in sequence)),
        'z_stage_max': float(max(p.z for p in sequence)),
        'tip_x_min': float(min(p.tip_x for p in sequence)),
        'tip_x_max': float(max(p.tip_x for p in sequence)),
        'tip_z_min': float(min(p.tip_z for p in sequence)),
        'tip_z_max': float(max(p.tip_z for p in sequence)),
        'closure_error_before_final': closure_error_before_final,
        'final_recenter_enabled': bool(final_recenter),
        'final_recenter_point': None if final_recenter_cp is None else asdict(final_recenter_cp),
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


def save_desired_hourglass_motion_plot(plot_path: str, command_sequence: List[CommandPoint]) -> str:
    if not command_sequence:
        raise RuntimeError("Cannot save desired hourglass motion plot: command_sequence is empty.")

    import matplotlib.pyplot as plt

    recorded = [cp for cp in command_sequence if is_recorded_phase(cp.phase)]
    if not recorded:
        raise RuntimeError("No recorded hourglass points are available for plotting.")

    tip_x = np.asarray([cp.tip_x for cp in recorded], dtype=float)
    tip_z = np.asarray([cp.tip_z for cp in recorded], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor=(0.0, 0.0, 0.0, 0.0))
    ax.set_facecolor((0.04, 0.09, 0.14, 0.88))

    ax.plot(
        tip_x,
        tip_z,
        color="#8cf7ff",
        linewidth=2.4,
        alpha=0.98,
        label="Desired hourglass motion",
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
    ax.set_title("Desired Generated Hourglass Motion", color="#f8fbff")
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
        print("STARTING HOURGLASS-TRACKING ACQUISITION RUN")
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

            print("\nExecuting hourglass tracking motion...")
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
    start_c = assert_c_in_safe_range("start_c", float(args.start_c))
    end_c = assert_c_in_safe_range("end_c", float(args.end_c))

    command_sequence, meta = build_hourglass_command_sequence(
        cal=cal,
        flip_rz_sign=flip_rz_sign,
        center_x=float(args.center_x),
        center_y=float(args.center_y),
        center_z=float(args.center_z),
        arc_vertical_gap_mm=float(args.arc_vertical_gap_mm),
        circle_diameter_mm=float(args.circle_diameter_mm),
        middle_gap_mm=float(args.middle_gap_mm),
        arc_overtravel_deg=float(args.arc_overtravel_deg),
        samples_per_arc=int(args.samples_per_arc),
        samples_per_diagonal=int(args.samples_per_diagonal),
        b_lo=b_lo,
        b_hi=b_hi,
        c0_deg=c0_deg,
        jog_feed=float(args.jog_feed),
        print_feed_b=float(args.print_feed if args.print_feed_b is None else args.print_feed_b),
        transition_feed=float(args.transition_feed),
        segment_b_override=(None if args.segment_b is None else float(args.segment_b)),
        final_recenter=bool(args.final_recenter),
    )
    meta["fit_mode"] = "shared_average_cubic" if bool(args.use_average_cubic_fit) else "phase_specific_default"

    print("Trajectory summary:")
    print(f"  fit_mode: {'shared_average_cubic' if bool(args.use_average_cubic_fit) else 'phase_specific_default'}")
    print(f"  pull_phase: {meta['pull_phase']}")
    print(f"  release_phase: {meta['release_phase']}")
    circle_diameter_requested = float(
        meta.get("circle_diameter_mm_requested", meta.get("circle_diameter_mm"))
    )
    circle_diameter_used = float(
        meta.get("circle_diameter_mm_used", meta.get("circle_diameter_mm"))
    )
    arc_vertical_gap_requested = float(
        meta.get("arc_vertical_gap_mm_requested", meta.get("arc_vertical_gap_mm"))
    )
    arc_vertical_gap_used = float(
        meta.get("arc_vertical_gap_mm_used", meta.get("arc_vertical_gap_mm"))
    )
    middle_gap_requested = float(
        meta.get("middle_gap_mm_requested", meta.get("middle_gap_mm"))
    )
    middle_gap_used = float(
        meta.get("middle_gap_mm_used", meta.get("middle_gap_mm"))
    )

    print(f"  circle_diameter_mm: requested={circle_diameter_requested:.3f}, used={circle_diameter_used:.3f}")
    print(f"  arc_vertical_gap_mm: requested={arc_vertical_gap_requested:.3f}, used={arc_vertical_gap_used:.3f}")
    print(f"  middle_gap_mm: requested={middle_gap_requested:.3f}, used={middle_gap_used:.3f}")
    print(f"  arc_overtravel_deg: {meta['arc_overtravel_deg']:.3f}")
    print(f"  segment_b: {meta['segment_b']:.3f}")
    print(f"  flip_rz_sign: {meta['flip_rz_sign']}")
    print(f"  X stage range: [{meta['x_stage_min']:.3f}, {meta['x_stage_max']:.3f}]")
    print(f"  Y stage range: [{meta['y_stage_min']:.3f}, {meta['y_stage_max']:.3f}]")
    print(f"  Z stage range: [{meta['z_stage_min']:.3f}, {meta['z_stage_max']:.3f}]")
    print(f"  C value used: {meta['c0_deg']:.3f}")
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
            os.path.join(runner.run_folder, "planned_hourglass_command_sequence.csv"),
            command_sequence,
        )
        plot_path = save_desired_hourglass_motion_plot(
            os.path.join(runner.run_folder, "desired_hourglass_motion.png"),
            command_sequence,
        )
        print(f"Saved plan metadata: {meta_path}")
        print(f"Saved command CSV:   {csv_path}")
        print(f"Saved path plot:     {plot_path}")

        if bool(args.plot_only):
            print("\nPlot-only mode requested; skipping robot motion, camera capture, and post-processing.")
            return

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
            capture_every_n_circle_moves=int(args.capture_every_n_shape_moves),
        )

        print("\nFinal results:")
        print(results)

        if args.enable_post:
            script_dir = Path(__file__).resolve().parent

            requested_post_script_path = Path(args.post_script).expanduser()
            candidate_post_paths = []
            if requested_post_script_path.is_absolute():
                candidate_post_paths.append(requested_post_script_path)
            else:
                candidate_post_paths.append(script_dir / requested_post_script_path)
                candidate_post_paths.append(script_dir / 'calib_hourglass_process_compat.py')
                candidate_post_paths.append(script_dir / 'calib_hourglass_process.py')

            post_script_path = None
            seen_candidates = set()
            for candidate in candidate_post_paths:
                candidate = candidate.resolve()
                if str(candidate) in seen_candidates:
                    continue
                seen_candidates.add(str(candidate))
                if candidate.is_file():
                    post_script_path = candidate
                    break

            if post_script_path is None:
                print('[WARN] Hourglass post-processing script not found. Checked:')
                for candidate in candidate_post_paths:
                    print(f'  - {candidate.resolve()}')
                print('[WARN] Skipping automatic post-processing.')
                post_script_path = None

            if post_script_path is None:
                pass
            else:
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

            planned_csv = Path(runner.run_folder).resolve() / "planned_hourglass_command_sequence.csv"
            post_cmd = [
                sys.executable,
                str(post_script_path),
                "--project_dir",
                str(Path(runner.run_folder).resolve()),
                "--camera_calibration_file",
                str(post_camera_calibration),
                "--calibration",
                str(Path(args.calibration).expanduser().resolve()),
                "--checkerboard_reference_image",
                str(post_reference_image),
                "--planned_command_csv",
                str(planned_csv),
                "--capture_every_n_shape_moves",
                str(int(args.capture_every_n_shape_moves)),
            ]

            post_tip_refiner_model = Path(args.post_tip_refiner_model).expanduser()
            if not post_tip_refiner_model.is_absolute():
                post_tip_refiner_model = script_dir / post_tip_refiner_model
            post_tip_refiner_model = post_tip_refiner_model.resolve()
            if post_tip_refiner_model.is_file():
                post_cmd.extend(["--tip_refiner_model", str(post_tip_refiner_model)])
                post_cmd.extend(["--tracked_tip_source", str(args.post_tracked_tip_source)])
            else:
                fallback_source = "auto" if str(args.post_tracked_tip_source).strip().lower() == "cnn" else str(args.post_tracked_tip_source)
                print(
                    f"[WARN] Post tip refiner model not found, cannot use "
                    f"tracked tip source '{args.post_tracked_tip_source}'. Falling back to '{fallback_source}': "
                    f"{post_tip_refiner_model}"
                )
                post_cmd.extend(["--tracked_tip_source", fallback_source])

            if bool(args.capture_at_start):
                post_cmd.append("--capture_at_start")
            if bool(args.post_save_plots):
                post_cmd.append("--save_plots")

            print("\nStarting hourglass post-processing:")
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
            "Run an hourglass-tracking acquisition directly on the robot using the existing "
            "calibration and capture pipeline, or save only the desired trajectory plot."
        )
    )

    # Run / folders
    ap.add_argument("--parent-directory", default=os.getcwd(), help="Parent folder for the run output.")
    ap.add_argument("--project-name", default=DEFAULT_PROJECT_NAME, help="Run folder name.")
    ap.add_argument("--allow-existing", action="store_true", default=DEFAULT_ALLOW_EXISTING,
                    help="Allow reuse of an existing run folder.")
    ap.add_argument("--add-date", action="store_true", default=DEFAULT_ADD_DATE,
                    help="Append timestamp to the run folder name.")
    ap.add_argument("--plot-only", action="store_true",
                    help="Only generate the desired trajectory CSV / metadata / plot. Skip robot motion, capture, and processing.")

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

    # Hourglass placement (tip-space)
    ap.add_argument("--center-x", type=float, default=HOURGLASS_CENTER_X,
                    help="Hourglass center X in tip space.")
    ap.add_argument("--center-y", type=float, default=HOURGLASS_CENTER_Y,
                    help="Constant hourglass center Y in tip space. Stage Y is solved to hold this exactly.")
    ap.add_argument("--center-z", type=float, default=HOURGLASS_CENTER_Z,
                    help="Hourglass center Z in tip space.")

    # Hourglass geometry
    ap.add_argument("--arc-vertical-gap-mm", type=float, default=DEFAULT_ARC_VERTICAL_GAP_MM,
                    help="Vertical distance between the top and bottom arc families at their closest centerline approach.")
    ap.add_argument("--circle-diameter-mm", type=float, default=DEFAULT_CIRCLE_DIAMETER_MM,
                    help="Diameter of each half-circle lobe.")
    ap.add_argument("--middle-gap-mm", type=float, default=DEFAULT_MIDDLE_GAP_MM,
                    help="Horizontal distance between the left and right diagonal lines at the narrow waist.")
    ap.add_argument("--arc-overtravel-deg", type=float, default=DEFAULT_ARC_OVERTRAVEL_DEG,
                    help="Extra angular extension used when attaching the diagonal connectors to each half-circle.")
    ap.add_argument("--samples-per-arc", type=int, default=DEFAULT_SAMPLES_PER_ARC,
                    help="Interpolation points used for each circular half-arc.")
    ap.add_argument("--samples-per-diagonal", type=int, default=DEFAULT_SAMPLES_PER_DIAGONAL,
                    help="Interpolation points used for each straight diagonal segment.")

    # Motion / feeds
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED,
                    help="Feedrate for safe travel moves.")
    ap.add_argument("--fine-approach-feed", type=float, default=DEFAULT_FINE_APPROACH_FEED,
                    help="Slow landing move used before the queued trajectory sweep.")
    ap.add_argument("--jog-feed", type=float, default=DEFAULT_PRINT_FEED,
                    help="Feedrate for startup move to first tracked point.")
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED,
                    help="Base feedrate for drawing moves.")
    ap.add_argument("--print-feed-b", type=float, default=DEFAULT_PRINT_FEED_B,
                    help="Feedrate for recorded tracked moves.")
    ap.add_argument("--transition-feed", type=float, default=DEFAULT_TRANSITION_FEED,
                    help="Feedrate for optional recenter / non-shape transition moves.")

    # B/C overrides
    ap.add_argument("--min-b", type=float, default=None, help="Lower bound for commanded B (default: calibration).")
    ap.add_argument("--max-b", type=float, default=None, help="Upper bound for commanded B (default: calibration).")
    ap.add_argument("--segment-b", type=float, default=None,
                    help="Optional non-zero B endpoint used for each tracked segment. Default: farthest valid common pull/release value from 0.")
    ap.add_argument("--c0-deg", type=float, default=DEFAULT_C0_DEG,
                    help="Fixed C value used for the full hourglass path.")

    # Optional waits / capture behavior
    ap.add_argument("--dwell-before-ms", type=int, default=DEFAULT_DWELL_BEFORE_MS)
    ap.add_argument("--dwell-after-ms", type=int, default=DEFAULT_DWELL_AFTER_MS)
    ap.add_argument("--initial-sweep-wait-s", type=float, default=DEFAULT_INITIAL_SWEEP_WAIT_S,
                    help="Hold time after landing on the first tracked point before queuing the trajectory sweep.")
    ap.add_argument("--tracked-move-settle-s", type=float, default=DEFAULT_TRACKED_MOVE_SETTLE_S,
                    help="Extra settle time after the queued tracked move block, before finishing.")
    ap.add_argument("--travel-move-settle-s", type=float, default=DEFAULT_TRAVEL_MOVE_SETTLE_S,
                    help="Extra settle time after travel moves.")
    ap.add_argument("--b-extra-settle-s", type=float, default=DEFAULT_B_EXTRA_SETTLE_S,
                    help="Additional hold after the queued trajectory motion to let the mechanism settle.")
    ap.add_argument("--inter-command-delay-s", type=float, default=DEFAULT_INTER_COMMAND_DELAY_S,
                    help="Small delay between queued tracked commands.")
    ap.add_argument("--capture-every-n-shape-moves", type=int, default=DEFAULT_CAPTURE_EVERY_N_CIRCLE_MOVES,
                    help="Capture one image every N recorded trajectory moves. Transition moves are never captured.")
    ap.add_argument("--capture-at-start", action="store_true", default=DEFAULT_CAPTURE_AT_START,
                    help="Also capture once at the first tracked sample after positioning there.")
    ap.add_argument("--final-recenter", dest="final_recenter", action="store_true", default=DEFAULT_FINAL_RECENTER,
                    help="After the last diagonal, optionally add an unrecorded exact recenter to the start point.")
    ap.add_argument("--no-final-recenter", dest="final_recenter", action="store_false",
                    help="Do not add the optional exact recenter after the hourglass closes.")

    ap.add_argument("--enable-post", action="store_true", default=DEFAULT_ENABLE_POST,
                    help="Run the hourglass checkerboard post-processing script automatically after acquisition completes.")
    ap.add_argument("--post-script", default=DEFAULT_POST_SCRIPT,
                    help="Path to the hourglass post-processing script. Relative paths are resolved next to this file.")
    ap.add_argument("--post-camera-calibration-file", default=DEFAULT_POST_CAMERA_CALIBRATION_FILE,
                    help="Camera calibration .npz to pass to post-processing.")
    ap.add_argument("--post-checkerboard-reference-image", default=DEFAULT_POST_CHECKERBOARD_REFERENCE_IMAGE,
                    help="Checkerboard reference image to pass to post-processing.")
    ap.add_argument("--post-tip-refiner-model", default=DEFAULT_POST_TIP_REFINER_MODEL,
                    help="Optional CNN tip refiner model passed to the hourglass post-processing script when present.")
    ap.add_argument("--post-tracked-tip-source", default=DEFAULT_POST_TRACKED_TIP_SOURCE,
                    choices=["auto", "coarse", "selected", "cnn"],
                    help="Tracked tip source to use during automatic hourglass post-processing. Defaults to cnn when the tip refiner model is available.")
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
