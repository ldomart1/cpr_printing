#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone star-tracking acquisition script.

What it does
------------
- Loads the calibration JSON.
- Builds the same star-point ordering expected by calib_star_process.py:
  right_start at the top mirror-point, then the right-half samples, then mirror_flip,
  then the mirrored left-half samples.
- Uses the phase-specific pull/release fit model that matches the actual branch motion.
- Inserts explicit unrecorded transition points whenever:
    * the planner needs to jump between disconnected star branches,
    * the motion changes from pull to release (or vice versa),
    * the C-axis flips from C0 to C180.
- During recorded star-branch motion, stage X is held fixed for each branch and only
  the calibrated B-driven lateral motion plus commanded Z is used to trace the line.
- Y tracking is solved so the tip remains at the requested constant tip-space Y.
- Optionally runs calib_star_process.py afterwards.

This is intentionally structured like the uploaded circle-tracking acquisition script,
but with a star planner that preserves the legacy star post-processing sample order.
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

try:
    from scipy.optimize import brentq  # type: ignore
except Exception:
    def brentq(f, a, b, maxiter=100, xtol=1e-10, rtol=4*np.finfo(float).eps):
        fa = f(a)
        fb = f(b)
        if np.sign(fa) == np.sign(fb):
            raise ValueError("Root not bracketed in [a,b].")
        x0, x1, f0, f1 = a, b, fa, fb
        x2 = 0.5 * (x0 + x1)
        for _ in range(maxiter):
            if f1 != f0:
                x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            else:
                x2 = 0.5 * (x0 + x1)
            if not (min(x0, x1) <= x2 <= max(x0, x1)):
                x2 = 0.5 * (x0 + x1)
            f2 = f(x2)
            if abs(f2) < max(xtol, rtol * abs(x2)):
                return x2
            if np.sign(f2) == np.sign(f0):
                x0, f0 = x2, f2
            else:
                x1, f1 = x2, f2
        return x2


# =========================
# Defaults
# =========================

DEFAULT_DUET_WEB_ADDRESS = "http://192.168.2.21"
DEFAULT_CAMERA_PORT = 0
DEFAULT_PROJECT_NAME = "Star_Tracking_Run"
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
DEFAULT_ENABLE_POST = False
DEFAULT_USE_AVERAGE_CUBIC_FIT = False

DEFAULT_POST_CAMERA_CALIBRATION_FILE = "captures/calibration_webcam_20260406_104136.npz"
DEFAULT_POST_CHECKERBOARD_REFERENCE_IMAGE = "captures/photo_20260406_104134.png"
DEFAULT_POST_THRESHOLD = 200
DEFAULT_POST_TIP_REFINE_MODE = "none"

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

STAR_CENTER_X = 100.0
STAR_CENTER_Y = 52.0
STAR_CENTER_Z = -125.0
STAR_OUTER_RADIUS = 18.0
STAR_INNER_RATIO = 0.38196601125
STAR_ROTATION_DEG = 270.0
STAR_ROTATION_CORRECTION_DEG = 180.0
SAMPLES_PER_EDGE = 30
DEFAULT_CAPTURE_EVERY_N_STAR_MOVES = 3
DEFAULT_SAFETY_MARGIN = 0.25
DEFAULT_C0_DEG = 0.0
DEFAULT_FLIP_RZ_SIGN = True

OFFPLANE_SIGN = -1.0
C_HARD_MIN_DEG = -360.0
C_HARD_MAX_DEG = 360.0

RECORDED_PHASES = {"right", "left"}
TRANSITION_PHASES = {
    "right_intro_jump",
    "left_intro_jump",
    "mirror_flip",
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


def tip_x_offset_physical(
    cal: Calibration,
    b: Any,
    c_deg: float,
    flip_rz_sign: bool = False,
    motion_phase: Optional[str] = None,
) -> np.ndarray:
    b_arr = np.asarray(b, dtype=float)
    r = np.asarray(eval_r(cal, b_arr, flip_rz_sign=flip_rz_sign, motion_phase=motion_phase), dtype=float)
    y_off = np.asarray(eval_offplane_y(cal, b_arr, motion_phase=motion_phase), dtype=float)
    c = math.radians(float(assert_c_in_safe_range("tip_x_offset C", c_deg)))
    return r * math.cos(c) - y_off * math.sin(c)


# =========================
# Star geometry / extraction
# =========================

def build_star_vertices(outer_radius: float, inner_ratio: float, rotation_deg: float) -> np.ndarray:
    inner_radius = outer_radius * inner_ratio
    rot = math.radians(rotation_deg + STAR_ROTATION_CORRECTION_DEG)
    verts: List[Tuple[float, float]] = []

    for i in range(10):
        ang = rot + i * math.pi / 5.0
        r = outer_radius if (i % 2 == 0) else inner_radius
        x = r * math.cos(ang)
        z = r * math.sin(ang)
        verts.append((x, z))

    verts.append(verts[0])
    return np.array(verts, dtype=float)


def densify_polyline(vertices: np.ndarray, samples_per_edge: int) -> np.ndarray:
    if samples_per_edge < 2:
        samples_per_edge = 2

    pts = []
    for i in range(len(vertices) - 1):
        p0 = vertices[i]
        p1 = vertices[i + 1]
        t = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)
        seg = (1.0 - t[:, None]) * p0[None, :] + t[:, None] * p1[None, :]
        pts.append(seg)
    pts.append(vertices[-1][None, :])
    return np.vstack(pts)


def densify_polyline_with_edge_ids(vertices: np.ndarray, samples_per_edge: int) -> Tuple[np.ndarray, np.ndarray]:
    if samples_per_edge < 2:
        samples_per_edge = 2

    pts = []
    edge_ids: List[int] = []
    for edge_i in range(len(vertices) - 1):
        p0 = vertices[edge_i]
        p1 = vertices[edge_i + 1]
        t = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)
        seg = (1.0 - t[:, None]) * p0[None, :] + t[:, None] * p1[None, :]
        pts.append(seg)
        edge_ids.extend([edge_i] * seg.shape[0])
    pts.append(vertices[-1][None, :])
    edge_ids.append(len(vertices) - 2)
    return np.vstack(pts), np.asarray(edge_ids, dtype=int)


def _clip_segment_to_right_half(p0: np.ndarray, p1: np.ndarray, eps: float = 1e-12) -> List[np.ndarray]:
    x0, x1 = float(p0[0]), float(p1[0])
    in0 = x0 >= -eps
    in1 = x1 >= -eps

    if in0 and in1:
        return [p0, p1]
    if (not in0) and (not in1):
        return []

    dx = x1 - x0
    if abs(dx) < eps:
        return [p0, p1] if (in0 or in1) else []

    t = -x0 / dx
    t = float(np.clip(t, 0.0, 1.0))
    p_cross = (1.0 - t) * p0 + t * p1
    p_cross[0] = 0.0

    if in0 and not in1:
        return [p0, p_cross]
    if (not in0) and in1:
        return [p_cross, p1]
    return []


def extract_right_half_polyline(full_pts: np.ndarray, dedup_tol: float = 1e-9) -> np.ndarray:
    out: List[np.ndarray] = []

    for i in range(len(full_pts) - 1):
        p0 = full_pts[i].copy()
        p1 = full_pts[i + 1].copy()
        clipped = _clip_segment_to_right_half(p0, p1)
        if len(clipped) != 2:
            continue
        q0, q1 = clipped

        if not out:
            out.append(q0)
        else:
            if np.linalg.norm(out[-1] - q0) > dedup_tol:
                out.append(q0)
        if np.linalg.norm(out[-1] - q1) > dedup_tol:
            out.append(q1)

    if not out:
        raise RuntimeError("Right-half extraction produced no points.")
    return np.vstack(out)


def extract_right_half_polyline_with_edge_ids(
    full_pts: np.ndarray,
    edge_ids: np.ndarray,
    dedup_tol: float = 1e-9,
) -> Tuple[np.ndarray, np.ndarray]:
    out: List[np.ndarray] = []
    out_edge_ids: List[int] = []

    for i in range(len(full_pts) - 1):
        p0 = full_pts[i].copy()
        p1 = full_pts[i + 1].copy()
        clipped = _clip_segment_to_right_half(p0, p1)
        if len(clipped) != 2:
            continue
        q0, q1 = clipped
        edge_i = int(edge_ids[i])

        if not out:
            out.append(q0)
            out_edge_ids.append(edge_i)
        else:
            if np.linalg.norm(out[-1] - q0) > dedup_tol:
                out.append(q0)
                out_edge_ids.append(edge_i)

        if np.linalg.norm(out[-1] - q1) > dedup_tol:
            out.append(q1)
            out_edge_ids.append(edge_i)

    if not out:
        raise RuntimeError("Right-half extraction produced no points.")
    return np.vstack(out), np.asarray(out_edge_ids, dtype=int)


def build_star_recorded_local_sequences(
    outer_radius: float,
    inner_ratio: float,
    rotation_deg: float,
    samples_per_edge: int,
) -> Tuple[np.ndarray, np.ndarray, List[Tuple[int, int, int]]]:
    """
    Returns:
      right_pts_local  : exact right-half point sequence expected by calib_star_process.py
      left_pts_local   : mirrored exact left-half point sequence expected by calib_star_process.py
      branch_runs      : list[(edge_id, start_idx, end_idx)] into right_pts_local for the actual
                         recorded star-edge branches. Each run includes the shared start point.

    The first point in right_pts_local / left_pts_local is the top mirror-point (intro point).
    The actual recorded right/left motion begins at index 1 after an unrecorded intro jump.
    """
    vertices = build_star_vertices(
        outer_radius=float(outer_radius),
        inner_ratio=float(inner_ratio),
        rotation_deg=float(rotation_deg),
    )
    full_pts, full_edge_ids = densify_polyline_with_edge_ids(vertices, int(samples_per_edge))
    right_pts, right_edge_ids = extract_right_half_polyline_with_edge_ids(full_pts, full_edge_ids)

    # Keep compatibility with the legacy star post-processing order.
    # The mirrored left side is NOT reversed because calib_star_process.py mirrors the same right-half ordering.
    left_pts = right_pts.copy()
    left_pts[:, 0] *= -1.0

    runs_raw: List[Tuple[int, int, int]] = []
    s = 0
    for i in range(1, len(right_edge_ids) + 1):
        if i == len(right_edge_ids) or right_edge_ids[i] != right_edge_ids[s]:
            runs_raw.append((int(right_edge_ids[s]), s, i - 1))
            s = i

    branch_runs: List[Tuple[int, int, int]] = []
    for edge_id, s_idx, e_idx in runs_raw:
        # Single-point clipped runs are intro/jump anchors, not actual drawn branches.
        if e_idx - s_idx < 1:
            continue
        branch_runs.append((edge_id, max(0, s_idx - 1), e_idx))

    if not branch_runs:
        raise RuntimeError("No drawable star branches were extracted for the right half.")

    return right_pts, left_pts, branch_runs


# =========================
# Branch solvers / planner
# =========================

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


def solve_b_for_tip_x_target(
    cal: Calibration,
    flip_rz_sign: bool,
    c_state: float,
    motion_phase: str,
    stage_x_const: float,
    tip_x_target: float,
    b_lo: float,
    b_hi: float,
    b_hint: Optional[float] = None,
    n_dense: int = 6001,
    tol_mm: float = 1e-5,
) -> float:
    lo = float(min(b_lo, b_hi))
    hi = float(max(b_lo, b_hi))
    if hi - lo <= 1e-12:
        pred = float(stage_x_const + tip_x_offset_physical(
            cal, np.asarray([lo], dtype=float), c_state, flip_rz_sign=flip_rz_sign, motion_phase=motion_phase
        )[0])
        if abs(pred - tip_x_target) > tol_mm:
            raise RuntimeError(
                f"No B span available while solving target tip X={tip_x_target:.6f} mm "
                f"for phase={motion_phase}; predicted {pred:.6f} mm at fixed B={lo:.6f}."
            )
        return lo

    bb = np.linspace(lo, hi, int(n_dense), dtype=float)
    xx = stage_x_const + tip_x_offset_physical(
        cal,
        bb,
        c_state,
        flip_rz_sign=flip_rz_sign,
        motion_phase=motion_phase,
    )
    ff = np.asarray(xx - float(tip_x_target), dtype=float)

    best_idx = int(np.argmin(np.abs(ff)))
    roots: List[float] = []
    if np.isfinite(ff[best_idx]) and abs(float(ff[best_idx])) <= tol_mm:
        roots.append(float(bb[best_idx]))

    for i in range(len(bb) - 1):
        f0 = float(ff[i])
        f1 = float(ff[i + 1])
        if not (np.isfinite(f0) and np.isfinite(f1)):
            continue
        if abs(f0) <= tol_mm:
            roots.append(float(bb[i]))
            continue
        if f0 == 0.0 or f1 == 0.0 or np.sign(f0) != np.sign(f1):
            a = float(bb[i])
            b = float(bb[i + 1])
            try:
                root = brentq(
                    lambda u: float(
                        stage_x_const
                        + tip_x_offset_physical(
                            cal,
                            np.asarray([u], dtype=float),
                            c_state,
                            flip_rz_sign=flip_rz_sign,
                            motion_phase=motion_phase,
                        )[0]
                        - tip_x_target
                    ),
                    a,
                    b,
                )
                roots.append(float(root))
            except Exception:
                pass

    if roots:
        roots_arr = np.asarray(sorted(set(round(r, 12) for r in roots)), dtype=float)
        if roots_arr.size == 1:
            return float(roots_arr[0])
        if b_hint is None:
            return float(roots_arr[0])
        return float(roots_arr[int(np.argmin(np.abs(roots_arr - float(b_hint))))])

    best_b = float(bb[best_idx])
    best_err = float(abs(ff[best_idx]))
    if best_err > max(tol_mm, 1e-4):
        raise RuntimeError(
            f"Could not solve B for target tip X={tip_x_target:.6f} mm in phase={motion_phase}; "
            f"best error was {best_err:.6f} mm over B in [{lo:.6f}, {hi:.6f}]."
        )
    return best_b


def build_star_edge_segment(
    cal: Calibration,
    flip_rz_sign: bool,
    center_x: float,
    center_y: float,
    center_z: float,
    local_pts_xz: np.ndarray,
    start_b: float,
    c_state: float,
    motion_phase: str,
    pull_phase: str,
    release_phase: str,
    common_b_lo: float,
    common_b_hi: float,
    start_label: str,
    rest_label: str,
    move_feed_start: float,
    move_feed_rest: float,
) -> Tuple[List[CommandPoint], Dict[str, float], float]:
    c_state = assert_c_in_safe_range("c_state", c_state)
    local_pts_xz = np.asarray(local_pts_xz, dtype=float)
    if local_pts_xz.ndim != 2 or local_pts_xz.shape[1] != 2 or local_pts_xz.shape[0] < 2:
        raise ValueError("Each star branch must contain at least two local X/Z points.")

    x_local = local_pts_xz[:, 0].astype(float)
    z_local = local_pts_xz[:, 1].astype(float)

    x0_tip = float(center_x + x_local[0])
    x1_tip = float(center_x + x_local[-1])
    if abs(x1_tip - x0_tip) <= 1e-12:
        raise RuntimeError(
            "Encountered a star branch with zero X span while recorded X is constrained to stay fixed."
        )

    xoff_start = float(tip_x_offset_physical(
        cal,
        np.asarray([float(start_b)], dtype=float),
        c_state,
        flip_rz_sign=flip_rz_sign,
        motion_phase=motion_phase,
    )[0])
    stage_x_const = float(x0_tip - xoff_start)

    b_vals = np.zeros(local_pts_xz.shape[0], dtype=float)
    b_vals[0] = float(start_b)

    for i in range(1, local_pts_xz.shape[0]):
        target_tip_x = float(center_x + x_local[i])
        prev_b = float(b_vals[i - 1])
        if motion_phase == pull_phase:
            lo = float(common_b_lo)
            hi = float(prev_b)
            hint = float(lo)
        elif motion_phase == release_phase:
            lo = float(prev_b)
            hi = float(common_b_hi)
            hint = float(hi)
        else:
            raise ValueError(f"Unsupported motion phase for star edge: {motion_phase}")

        b_vals[i] = solve_b_for_tip_x_target(
            cal=cal,
            flip_rz_sign=flip_rz_sign,
            c_state=float(c_state),
            motion_phase=motion_phase,
            stage_x_const=float(stage_x_const),
            tip_x_target=float(target_tip_x),
            b_lo=float(lo),
            b_hi=float(hi),
            b_hint=float(hint),
        )

    pts: List[CommandPoint] = []
    stage_x_trace: List[float] = []
    for i, (b_i, x_loc, z_loc) in enumerate(zip(b_vals, x_local, z_local)):
        tip_target = np.array([
            float(center_x + x_loc),
            float(center_y),
            float(center_z + z_loc),
        ], dtype=float)

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
            raise RuntimeError("Internal tip-tracking consistency check failed in star planner.")

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
        "motion_phase": str(motion_phase),
        "b_start": float(b_vals[0]),
        "b_end": float(b_vals[-1]),
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
    return pts, meta, float(b_vals[-1])


def infer_branch_phase_from_abs_x(
    branch_local_pts: np.ndarray,
    pull_phase: str,
    release_phase: str,
    prev_phase: Optional[str] = None,
    tol: float = 1e-9,
) -> str:
    branch_local_pts = np.asarray(branch_local_pts, dtype=float)
    start_abs = float(abs(branch_local_pts[0, 0]))
    end_abs = float(abs(branch_local_pts[-1, 0]))
    if end_abs > start_abs + tol:
        return str(pull_phase)
    if end_abs < start_abs - tol:
        return str(release_phase)
    return str(prev_phase or pull_phase)


def _build_star_sequence_for_radius(
    cal: Calibration,
    flip_rz_sign: bool,
    center_x: float,
    center_y: float,
    center_z: float,
    outer_radius: float,
    inner_ratio: float,
    rotation_deg: float,
    samples_per_edge: int,
    b_lo: float,
    b_hi: float,
    c0_deg: float,
    c180_deg: float,
    jog_feed: float,
    print_feed_b: float,
    print_feed_c: float,
    transition_feed: float,
) -> Tuple[List[CommandPoint], dict]:
    c0_deg = assert_c_in_safe_range("c0_deg", c0_deg)
    c180_deg = assert_c_in_safe_range("c180_deg", c180_deg)

    pull_phase = resolve_phase_name(cal, "pull")
    release_phase = resolve_phase_name(cal, "release")
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

    right_local, left_local, branch_runs = build_star_recorded_local_sequences(
        outer_radius=float(outer_radius),
        inner_ratio=float(inner_ratio),
        rotation_deg=float(rotation_deg),
        samples_per_edge=int(samples_per_edge),
    )

    right_intro_tip = np.array([float(center_x + right_local[0, 0]), float(center_y), float(center_z + right_local[0, 1])], dtype=float)
    left_intro_tip = np.array([float(center_x + left_local[0, 0]), float(center_y), float(center_z + left_local[0, 1])], dtype=float)

    right_first_phase = infer_branch_phase_from_abs_x(right_local[branch_runs[0][1]:branch_runs[0][2] + 1], pull_phase, release_phase)
    left_first_phase = infer_branch_phase_from_abs_x(left_local[branch_runs[0][1]:branch_runs[0][2] + 1], pull_phase, release_phase)

    sequence: List[CommandPoint] = []
    branch_meta: List[dict] = []
    current_b = 0.0

    # Right intro point (capturable as right_start when requested).
    right_start_cp = transition_point_for_same_tip(
        cal=cal,
        flip_rz_sign=flip_rz_sign,
        tip_target=right_intro_tip,
        b_value=float(current_b),
        c_state=float(c0_deg),
        motion_phase=str(right_first_phase),
        phase_label="right_start",
        feed=float(jog_feed),
    )
    sequence.append(right_start_cp)

    # Unrecorded jump from top intro point to the first actual right-half point (bottom mirror-point).
    first_right_branch_local = right_local[branch_runs[0][1]:branch_runs[0][2] + 1]
    first_right_branch_start_tip = np.array([
        float(center_x + first_right_branch_local[0, 0]),
        float(center_y),
        float(center_z + first_right_branch_local[0, 1]),
    ], dtype=float)
    sequence.append(
        transition_point_for_same_tip(
            cal=cal,
            flip_rz_sign=flip_rz_sign,
            tip_target=first_right_branch_start_tip,
            b_value=float(current_b),
            c_state=float(c0_deg),
            motion_phase=str(right_first_phase),
            phase_label="right_intro_jump",
            feed=float(transition_feed),
        )
    )

    prev_phase: Optional[str] = None
    phase_transition_count = 0
    for branch_idx, (edge_id, start_idx, end_idx) in enumerate(branch_runs):
        branch_local = right_local[start_idx:end_idx + 1]
        branch_phase = infer_branch_phase_from_abs_x(branch_local, pull_phase, release_phase, prev_phase=prev_phase)

        branch_start_tip = np.array([
            float(center_x + branch_local[0, 0]),
            float(center_y),
            float(center_z + branch_local[0, 1]),
        ], dtype=float)
        if prev_phase is not None and branch_phase != prev_phase:
            phase_transition_count += 1
            sequence.append(
                transition_point_for_same_tip(
                    cal=cal,
                    flip_rz_sign=flip_rz_sign,
                    tip_target=branch_start_tip,
                    b_value=float(current_b),
                    c_state=float(c0_deg),
                    motion_phase=str(branch_phase),
                    phase_label=f"right_phase_transition_{phase_transition_count:02d}",
                    feed=float(transition_feed),
                )
            )

        branch_pts, meta_edge, current_b = build_star_edge_segment(
            cal=cal,
            flip_rz_sign=flip_rz_sign,
            center_x=float(center_x),
            center_y=float(center_y),
            center_z=float(center_z),
            local_pts_xz=branch_local,
            start_b=float(current_b),
            c_state=float(c0_deg),
            motion_phase=str(branch_phase),
            pull_phase=str(pull_phase),
            release_phase=str(release_phase),
            common_b_lo=float(common_lo),
            common_b_hi=float(common_hi),
            start_label="right",
            rest_label="right",
            move_feed_start=float(transition_feed),
            move_feed_rest=float(print_feed_b),
        )
        if branch_idx == 0:
            sequence.extend(branch_pts)
        else:
            sequence.extend(branch_pts[1:])

        meta_edge.update({
            "side": "right",
            "edge_id": int(edge_id),
            "start_index": int(start_idx),
            "end_index": int(end_idx),
        })
        branch_meta.append(meta_edge)
        prev_phase = str(branch_phase)

    # Midpoint C flip at the top intro point using the phase needed by the first left branch.
    sequence.append(
        transition_point_for_same_tip(
            cal=cal,
            flip_rz_sign=flip_rz_sign,
            tip_target=left_intro_tip,
            b_value=float(current_b),
            c_state=float(c180_deg),
            motion_phase=str(left_first_phase),
            phase_label="mirror_flip",
            feed=float(print_feed_c),
        )
    )

    # Unrecorded jump on the left side from top intro point to the first actual left-half point.
    first_left_branch_local = left_local[branch_runs[0][1]:branch_runs[0][2] + 1]
    first_left_branch_start_tip = np.array([
        float(center_x + first_left_branch_local[0, 0]),
        float(center_y),
        float(center_z + first_left_branch_local[0, 1]),
    ], dtype=float)
    sequence.append(
        transition_point_for_same_tip(
            cal=cal,
            flip_rz_sign=flip_rz_sign,
            tip_target=first_left_branch_start_tip,
            b_value=float(current_b),
            c_state=float(c180_deg),
            motion_phase=str(left_first_phase),
            phase_label="left_intro_jump",
            feed=float(transition_feed),
        )
    )

    prev_phase = None
    phase_transition_count = 0
    for branch_idx, (edge_id, start_idx, end_idx) in enumerate(branch_runs):
        branch_local = left_local[start_idx:end_idx + 1]
        branch_phase = infer_branch_phase_from_abs_x(branch_local, pull_phase, release_phase, prev_phase=prev_phase)

        branch_start_tip = np.array([
            float(center_x + branch_local[0, 0]),
            float(center_y),
            float(center_z + branch_local[0, 1]),
        ], dtype=float)
        if prev_phase is not None and branch_phase != prev_phase:
            phase_transition_count += 1
            sequence.append(
                transition_point_for_same_tip(
                    cal=cal,
                    flip_rz_sign=flip_rz_sign,
                    tip_target=branch_start_tip,
                    b_value=float(current_b),
                    c_state=float(c180_deg),
                    motion_phase=str(branch_phase),
                    phase_label=f"left_phase_transition_{phase_transition_count:02d}",
                    feed=float(transition_feed),
                )
            )

        branch_pts, meta_edge, current_b = build_star_edge_segment(
            cal=cal,
            flip_rz_sign=flip_rz_sign,
            center_x=float(center_x),
            center_y=float(center_y),
            center_z=float(center_z),
            local_pts_xz=branch_local,
            start_b=float(current_b),
            c_state=float(c180_deg),
            motion_phase=str(branch_phase),
            pull_phase=str(pull_phase),
            release_phase=str(release_phase),
            common_b_lo=float(common_lo),
            common_b_hi=float(common_hi),
            start_label="left",
            rest_label="left",
            move_feed_start=float(transition_feed),
            move_feed_rest=float(print_feed_b),
        )
        if branch_idx == 0:
            sequence.extend(branch_pts)
        else:
            sequence.extend(branch_pts[1:])

        meta_edge.update({
            "side": "left",
            "edge_id": int(edge_id),
            "start_index": int(start_idx),
            "end_index": int(end_idx),
        })
        branch_meta.append(meta_edge)
        prev_phase = str(branch_phase)

    assert_all_command_c_safe(sequence)

    meta = {
        "pull_phase": str(pull_phase),
        "release_phase": str(release_phase),
        "outer_radius_used": float(outer_radius),
        "center_x": float(center_x),
        "center_y": float(center_y),
        "center_z": float(center_z),
        "inner_ratio": float(inner_ratio),
        "rotation_deg": float(rotation_deg),
        "samples_per_edge": int(samples_per_edge),
        "c0_deg": float(c0_deg),
        "c180_deg": float(c180_deg),
        "flip_rz_sign": bool(flip_rz_sign),
        "common_b_window": [float(common_lo), float(common_hi)],
        "right_points": int(right_local.shape[0]),
        "left_points": int(left_local.shape[0]),
        "branch_meta": branch_meta,
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
        "final_b": float(current_b),
        "recorded_stage_x_variation_max": float(max(m["stage_x_variation"] for m in branch_meta) if branch_meta else 0.0),
    }
    return sequence, meta


def build_star_command_sequence(
    cal: Calibration,
    flip_rz_sign: bool,
    center_x: float,
    center_y: float,
    center_z: float,
    outer_radius: float,
    inner_ratio: float,
    rotation_deg: float,
    samples_per_edge: int,
    safety_margin: float,
    b_lo: float,
    b_hi: float,
    c0_deg: float,
    c180_deg: float,
    jog_feed: float,
    print_feed_b: float,
    print_feed_c: float,
    transition_feed: float,
) -> Tuple[List[CommandPoint], dict]:
    desired_outer_radius = float(outer_radius)
    if desired_outer_radius <= 0.0:
        raise ValueError("Star outer radius must be positive.")

    # Try the requested radius first. If it fails, scale down via binary search.
    try:
        seq, meta = _build_star_sequence_for_radius(
            cal=cal,
            flip_rz_sign=flip_rz_sign,
            center_x=float(center_x),
            center_y=float(center_y),
            center_z=float(center_z),
            outer_radius=float(desired_outer_radius),
            inner_ratio=float(inner_ratio),
            rotation_deg=float(rotation_deg),
            samples_per_edge=int(samples_per_edge),
            b_lo=float(b_lo),
            b_hi=float(b_hi),
            c0_deg=float(c0_deg),
            c180_deg=float(c180_deg),
            jog_feed=float(jog_feed),
            print_feed_b=float(print_feed_b),
            print_feed_c=float(print_feed_c),
            transition_feed=float(transition_feed),
        )
        meta["outer_radius_requested"] = float(desired_outer_radius)
        meta["outer_radius_scale_factor"] = 1.0
        meta["safety_margin"] = float(safety_margin)
        return seq, meta
    except Exception as requested_exc:
        last_exc = requested_exc

    lo = 0.0
    hi = 1.0
    best_seq: Optional[List[CommandPoint]] = None
    best_meta: Optional[dict] = None
    best_scale = 0.0

    for _ in range(28):
        mid = 0.5 * (lo + hi)
        try_radius = max(1e-6, desired_outer_radius * mid)
        try:
            seq, meta = _build_star_sequence_for_radius(
                cal=cal,
                flip_rz_sign=flip_rz_sign,
                center_x=float(center_x),
                center_y=float(center_y),
                center_z=float(center_z),
                outer_radius=float(try_radius),
                inner_ratio=float(inner_ratio),
                rotation_deg=float(rotation_deg),
                samples_per_edge=int(samples_per_edge),
                b_lo=float(b_lo),
                b_hi=float(b_hi),
                c0_deg=float(c0_deg),
                c180_deg=float(c180_deg),
                jog_feed=float(jog_feed),
                print_feed_b=float(print_feed_b),
                print_feed_c=float(print_feed_c),
                transition_feed=float(transition_feed),
            )
            best_seq = seq
            best_meta = meta
            best_scale = mid
            lo = mid
        except Exception as exc:
            last_exc = exc
            hi = mid

    if best_seq is None or best_meta is None:
        raise RuntimeError(f"Could not build a reachable star path: {last_exc}")

    used_radius = desired_outer_radius * best_scale
    if used_radius <= max(0.0, desired_outer_radius - float(safety_margin)) and float(safety_margin) > 0.0:
        used_radius = max(used_radius, 1e-6)

    best_meta["outer_radius_requested"] = float(desired_outer_radius)
    best_meta["outer_radius_used"] = float(used_radius)
    best_meta["outer_radius_scale_factor"] = float(best_scale)
    best_meta["safety_margin"] = float(safety_margin)
    return best_seq, best_meta


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


def save_desired_star_motion_plot(plot_path: str, command_sequence: List[CommandPoint]) -> str:
    if not command_sequence:
        raise RuntimeError("Cannot save desired star motion plot: command_sequence is empty.")

    import matplotlib.pyplot as plt

    tip_x = np.asarray([cp.tip_x for cp in command_sequence if cp.phase in RECORDED_PHASES or cp.phase == "right_start"], dtype=float)
    tip_z = np.asarray([cp.tip_z for cp in command_sequence if cp.phase in RECORDED_PHASES or cp.phase == "right_start"], dtype=float)

    fig, ax = plt.subplots(figsize=(7.5, 7.5), facecolor=(0.0, 0.0, 0.0, 0.0))
    ax.set_facecolor((0.04, 0.09, 0.14, 0.88))

    ax.plot(
        tip_x,
        tip_z,
        color="#8cf7ff",
        linewidth=2.4,
        alpha=0.98,
        label="Desired star motion",
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
    ax.set_title("Desired Generated Star Motion", color="#f8fbff")
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

class StarTrackerRunner:
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

    def execute_star_motion_and_capture(
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
        capture_every_n_star_moves: int = 1,
    ):
        if self.cam is None:
            raise RuntimeError("Camera is not connected.")
        if self.rrf is None:
            raise RuntimeError("Robot is not connected.")

        bbox_warnings: List[str] = []
        sample_counter = 0
        star_move_counter = 0
        capture_every_n_star_moves = max(1, int(capture_every_n_star_moves))
        self.commanded_axes = {}

        print("\n" + "=" * 72)
        print("STARTING STAR-TRACKING ACQUISITION RUN")
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

            print("\nExecuting star tracking motion...")
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

                per_move_floor = float(inter_command_delay_s)
                if cp.phase in RECORDED_PHASES:
                    self.wait_for_duet_motion_complete(extra_settle=float(tracked_move_settle_s))
                else:
                    wait_s = self._compute_inter_command_wait_s(est_move_time_s, configured_floor_s=per_move_floor)
                    if wait_s > 0:
                        time.sleep(wait_s)
                    self.wait_for_duet_motion_complete(extra_settle=0.0)

                if cp.phase in RECORDED_PHASES:
                    star_move_counter += 1
                    if (star_move_counter % capture_every_n_star_moves) == 0:
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

                if float(b_extra_settle_s) > 0 and cp.phase in RECORDED_PHASES:
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

    command_sequence, meta = build_star_command_sequence(
        cal=cal,
        flip_rz_sign=flip_rz_sign,
        center_x=float(args.center_x),
        center_y=float(args.center_y),
        center_z=float(args.center_z),
        outer_radius=float(args.outer_radius),
        inner_ratio=float(args.inner_ratio),
        rotation_deg=float(args.rotation_deg),
        samples_per_edge=int(args.samples_per_edge),
        safety_margin=float(args.safety_margin),
        b_lo=b_lo,
        b_hi=b_hi,
        c0_deg=c0_deg,
        c180_deg=c180_deg,
        jog_feed=float(args.jog_feed),
        print_feed_b=float(args.print_feed if args.print_feed_b is None else args.print_feed_b),
        print_feed_c=float(args.print_feed if args.print_feed_c is None else args.print_feed_c),
        transition_feed=float(args.transition_feed),
    )
    meta["fit_mode"] = "shared_average_cubic" if bool(args.use_average_cubic_fit) else "phase_specific_default"

    print("Trajectory summary:")
    print(f"  fit_mode: {'shared_average_cubic' if bool(args.use_average_cubic_fit) else 'phase_specific_default'}")
    print(f"  pull_phase: {meta['pull_phase']}")
    print(f"  release_phase: {meta['release_phase']}")
    print(f"  outer_radius_requested: {meta['outer_radius_requested']:.3f} mm")
    print(f"  outer_radius_used:      {meta['outer_radius_used']:.3f} mm")
    print(f"  outer_radius_scale:     {meta['outer_radius_scale_factor']:.6f}")
    print(f"  flip_rz_sign: {meta['flip_rz_sign']}")
    print(f"  X stage range: [{meta['x_stage_min']:.3f}, {meta['x_stage_max']:.3f}]")
    print(f"  Y stage range: [{meta['y_stage_min']:.3f}, {meta['y_stage_max']:.3f}]")
    print(f"  Z stage range: [{meta['z_stage_min']:.3f}, {meta['z_stage_max']:.3f}]")
    print(f"  C values used: [{meta['c0_deg']:.3f}, {meta['c180_deg']:.3f}]")
    print(f"  Max recorded-branch stage X variation: {meta['recorded_stage_x_variation_max']:.6f} mm")
    print(f"  Final B after full star: {meta['final_b']:.6f}")

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
    if virtual_bbox["x_min"] > virtual_bbox["x_max"]:
        virtual_bbox["x_min"], virtual_bbox["x_max"] = virtual_bbox["x_max"], virtual_bbox["x_min"]
    if virtual_bbox["y_min"] > virtual_bbox["y_max"]:
        virtual_bbox["y_min"], virtual_bbox["y_max"] = virtual_bbox["y_max"], virtual_bbox["y_min"]
    if virtual_bbox["z_min"] > virtual_bbox["z_max"]:
        virtual_bbox["z_min"], virtual_bbox["z_max"] = virtual_bbox["z_max"], virtual_bbox["z_min"]

    runner = StarTrackerRunner(
        parent_directory=args.parent_directory,
        project_name=args.project_name,
        allow_existing=bool(args.allow_existing),
        add_date=bool(args.add_date),
    )

    script_dir = Path(__file__).resolve().parent

    try:
        meta_path = save_json(os.path.join(runner.run_folder, "trajectory_meta.json"), meta)
        csv_path = save_command_sequence_csv(
            os.path.join(runner.run_folder, "planned_command_sequence.csv"),
            command_sequence,
        )
        plot_path = save_desired_star_motion_plot(
            os.path.join(runner.run_folder, "desired_star_motion.png"),
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

        results = runner.execute_star_motion_and_capture(
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
            capture_every_n_star_moves=int(args.capture_every_n_star_moves),
        )

        print("\nFinal results:")
        print(results)

        if bool(args.enable_post):
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
                str((script_dir / "calib_star_process.py").resolve()),
                "--project_dir",
                str(Path(runner.run_folder).resolve()),
                "--camera_calibration_file",
                str(post_camera_calibration),
                "--checkerboard_reference_image",
                str(post_reference_image),
                "--threshold",
                str(int(args.post_threshold)),
                "--tip_refine_mode",
                str(args.post_tip_refine_mode),
                "--star_center_x_mm",
                str(float(args.center_x)),
                "--star_center_z_mm",
                str(float(args.center_z)),
                "--star_outer_radius_mm",
                str(float(meta["outer_radius_used"])),
                "--star_inner_ratio",
                str(float(args.inner_ratio)),
                "--star_rotation_deg",
                str(float(args.rotation_deg)),
                "--star_samples_per_edge",
                str(int(args.samples_per_edge)),
                "--capture_every_n_star_moves",
                str(int(args.capture_every_n_star_moves)),
            ]
            if bool(args.capture_at_start):
                post_cmd.append("--capture_at_start")
            if bool(args.post_save_analysis_config):
                post_cmd.append("--save_analysis_config")
            if bool(args.post_save_plots):
                post_cmd.append("--save_plots")

            print("\nRunning post-processing:")
            print("  " + " ".join(post_cmd))
            subprocess.run(post_cmd, check=True, cwd=str(script_dir))

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
            "Run a star-tracking acquisition directly on the robot using the phase-specific "
            "pull/release PCHIP B calibration plus commanded Z motion, with unrecorded branch "
            "transitions inserted wherever the star path is discontinuous or changes hysteresis branch."
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

    # Star placement (tip-space)
    ap.add_argument("--center-x", type=float, default=STAR_CENTER_X,
                    help="Nominal star center X in tip space.")
    ap.add_argument("--center-y", type=float, default=STAR_CENTER_Y,
                    help="Constant star center Y in tip space. Stage Y is solved to hold this exactly.")
    ap.add_argument("--center-z", type=float, default=STAR_CENTER_Z,
                    help="Star center Z in tip space.")

    # Star geometry
    ap.add_argument("--outer-radius", type=float, default=STAR_OUTER_RADIUS,
                    help="Requested outer star radius in mm (auto-scaled down only if needed for reachability).")
    ap.add_argument("--inner-ratio", type=float, default=STAR_INNER_RATIO,
                    help="Inner radius / outer radius for the 5-point star.")
    ap.add_argument("--rotation-deg", type=float, default=STAR_ROTATION_DEG,
                    help="Star rotation in XZ plane (deg).")
    ap.add_argument("--samples-per-edge", type=int, default=SAMPLES_PER_EDGE,
                    help="Interpolation points per star edge, matching calib_star_process.py geometry lookup.")
    ap.add_argument("--safety-margin", type=float, default=DEFAULT_SAFETY_MARGIN,
                    help="Reserved margin used when the requested star must be scaled down for reachability.")

    # Motion / feeds
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED,
                    help="Feedrate for safe travel moves.")
    ap.add_argument("--fine-approach-feed", type=float, default=DEFAULT_FINE_APPROACH_FEED,
                    help="Slow landing move used before the queued star sweep.")
    ap.add_argument("--jog-feed", type=float, default=DEFAULT_PRINT_FEED,
                    help="Feedrate for startup move to the first tracked point.")
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED,
                    help="Base feedrate for drawing moves.")
    ap.add_argument("--print-feed-b", type=float, default=DEFAULT_PRINT_FEED_B,
                    help="Feedrate for recorded B/Z star moves.")
    ap.add_argument("--print-feed-c", type=float, default=DEFAULT_PRINT_FEED_C,
                    help="Feedrate for the midpoint 180-degree C rotation while tracking.")
    ap.add_argument("--transition-feed", type=float, default=DEFAULT_TRANSITION_FEED,
                    help="Feedrate for branch jumps and pull/release transition moves.")

    # B/C overrides
    ap.add_argument("--min-b", type=float, default=None, help="Lower bound for commanded B (default: calibration).")
    ap.add_argument("--max-b", type=float, default=None, help="Upper bound for commanded B (default: calibration).")
    ap.add_argument("--c0-deg", type=float, default=DEFAULT_C0_DEG,
                    help="Fixed C value for the right-half side.")
    ap.add_argument("--c180-deg", type=float, default=None,
                    help="Fixed C value for the mirrored left-half side (default from calibration).")

    # Optional waits / capture behavior
    ap.add_argument("--dwell-before-ms", type=int, default=DEFAULT_DWELL_BEFORE_MS)
    ap.add_argument("--dwell-after-ms", type=int, default=DEFAULT_DWELL_AFTER_MS)
    ap.add_argument("--initial-sweep-wait-s", type=float, default=DEFAULT_INITIAL_SWEEP_WAIT_S,
                    help="Hold time after landing on the first tracked point before queuing the star sweep.")
    ap.add_argument("--tracked-move-settle-s", type=float, default=DEFAULT_TRACKED_MOVE_SETTLE_S,
                    help="Extra settle time after each recorded tracked move, before capture.")
    ap.add_argument("--travel-move-settle-s", type=float, default=DEFAULT_TRAVEL_MOVE_SETTLE_S,
                    help="Extra settle time after travel moves.")
    ap.add_argument("--b-extra-settle-s", type=float, default=DEFAULT_B_EXTRA_SETTLE_S,
                    help="Additional hold after each recorded star move to let the mechanism settle.")
    ap.add_argument("--inter-command-delay-s", type=float, default=DEFAULT_INTER_COMMAND_DELAY_S,
                    help="Small delay between queued tracked commands.")
    ap.add_argument("--capture-every-n-star-moves", type=int, default=DEFAULT_CAPTURE_EVERY_N_STAR_MOVES,
                    help="Capture one image every N recorded star-path moves. Transition moves are never captured.")
    ap.add_argument("--capture-at-start", action="store_true", default=DEFAULT_CAPTURE_AT_START,
                    help="Also capture once at the first tracked sample after positioning there.")

    # Post-processing
    ap.add_argument("--enable-post", action="store_true", default=DEFAULT_ENABLE_POST,
                    help="Run calib_star_process.py automatically after acquisition completes.")
    ap.add_argument("--post-camera-calibration-file", default=DEFAULT_POST_CAMERA_CALIBRATION_FILE,
                    help="Camera calibration .npz to pass to post-processing.")
    ap.add_argument("--post-checkerboard-reference-image", default=DEFAULT_POST_CHECKERBOARD_REFERENCE_IMAGE,
                    help="Checkerboard reference image to pass to post-processing.")
    ap.add_argument("--post-threshold", type=int, default=DEFAULT_POST_THRESHOLD,
                    help="Threshold passed to calib_star_process.py.")
    ap.add_argument("--post-tip-refine-mode", default=DEFAULT_POST_TIP_REFINE_MODE,
                    help="Tip refinement mode passed to calib_star_process.py.")
    ap.add_argument("--post-save-analysis-config", dest="post_save_analysis_config",
                    action="store_true", default=True,
                    help="Pass --save_analysis_config to the post-processing script.")
    ap.add_argument("--no-post-save-analysis-config", dest="post_save_analysis_config",
                    action="store_false",
                    help="Do not pass --save_analysis_config to the post-processing script.")
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
