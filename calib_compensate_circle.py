#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Circle branch capture + print G-code exporter.

This script combines the circle-trace strategy with the camera/capture workflow:

1. Load a CTR_Shadow_Calibration global calibration export.
2. Use pull/release branch models from the calibration export.
   By default this uses the global:
       fit_models_by_phase
   and it can optionally target a curl-specific set:
       curl_angle_specific_fit_models['<curve_set>']['fit_models_by_phase']
   with pull/release bundles containing r, z, tip_angle, and optional offplane_y
   PCHIP descriptors with x_knots/y_knots.
3. Build a full XZ circle in tip space at constant Y.
4. Pre-position at the circle start with attack angle 0 deg.
5. Curl at the fixed start point from 0 -> 180 deg using the selected pull branch.
6. Trace the first half of the circle as release:
       attack angle 180 -> 0 deg, using the selected release branch.
7. At the rightmost point, perform a fixed-tip C spin from C=180 to C=0.
8. Trace the second half as pull/curl:
       attack angle 0 -> 180 deg, using the selected pull branch.
8. Export continuous print G-code for the compensated/calibrated circle path.
9. Optionally execute camera acquisitions of:
       - compensated/calibrated circle pass
       - non-compensated circle pass
   The non-compensated pass follows the same B/C phase schedule but does not subtract
   the calibrated tip offset from the gantry position.

Place this file next to shadow_calibration.py.

Example: export only, no robot/camera connection
-----------------------------------------------
python circle_0_180_capture_and_gcode.py \
  --calibration_file Test_Calibration_2026-06-24_00/processed_image_data_folder/calibrated_robot_gcode_calibration.json \
  --project_name Circle_0_180_Test \
  --center_x 100 --center_y 52 --center_z -130 --radius 20 \
  --export_only

Example: acquire compensated and non-compensated capture passes, and export print G-code
--------------------------------------------------------------------------------------
python circle_0_180_capture_and_gcode.py \
  --calibration_file Test_Calibration_2026-06-24_00/processed_image_data_folder/calibrated_robot_gcode_calibration.json \
  --project_name Circle_0_180_Capture \
  --center_x 100 --center_y 52 --center_z -130 --radius 20 \
  --capture_compensated \
  --capture_noncompensated \
  --export_analysis_outputs

For actual printing, add --emit_extrusion when exporting/running the generated print G-code.
By default extrusion uses M42 P10 S1 to start and M42 P10 S0 to stop.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from shadow_calibration import CTR_Shadow_Calibration  # noqa: E402


# -----------------------------------------------------------------------------
# Defaults
# -----------------------------------------------------------------------------

DEFAULT_DUET_WEB_ADDRESS = "http://192.168.2.21"
DEFAULT_CAMERA_PORT = 0
DEFAULT_PROJECT_NAME = "Circle_0_180_Capture"

DEFAULT_CENTER_X = 100.0
DEFAULT_CENTER_Y = 55.0
DEFAULT_CENTER_Z = -130.0
DEFAULT_RADIUS = 20.0
DEFAULT_POINTS_PER_MM = 8.0
DEFAULT_CAPTURE_POINTS_PER_HALF = 37
DEFAULT_DEBUG_MOTION_POINTS_PER_HALF = 0
DEFAULT_PRECURL_STEPS = 37

DEFAULT_CAMERA_CALIBRATION_FILE = SCRIPT_DIR / "captures" / "calibration_webcam_20260406_104136.npz"
DEFAULT_BOARD_REFERENCE_IMAGE = SCRIPT_DIR / "captures" / "photo_20260627_161714.png"
DEFAULT_THRESHOLD = 220
DEFAULT_MANUAL_CROP_ADJUSTMENT = True
DEFAULT_CAMERA_WIDTH = 3840
DEFAULT_CAMERA_HEIGHT = 2160
DEFAULT_CAMERA_FLUSH_FRAMES = 1
DEFAULT_MANUAL_FOCUS = True
DEFAULT_MANUAL_FOCUS_VAL = 60

DEFAULT_TRAVEL_FEED = 1000.0
DEFAULT_APPROACH_FEED = 400.0
DEFAULT_FINE_APPROACH_FEED = 80.0
DEFAULT_INITIAL_XYZC_FEED = 5000.0
DEFAULT_INITIAL_B_FEED = 400.0
DEFAULT_INITIAL_DAQ_WAIT_S = 6.0
DEFAULT_TRACE_FEED = 200.0
DEFAULT_C_SPIN_FEED = 10000.0
DEFAULT_SPIN_STEPS = 24
DEFAULT_PREFLOW_DWELL_MS = 500
DEFAULT_EMIT_EXTRUSION = False
DEFAULT_PRESSURE_OUTPUT_PIN = 10

DEFAULT_FLIP_RZ_SIGN = True
DEFAULT_OFFPLANE_Y_SIGN = -1.0


# -----------------------------------------------------------------------------
# Data structures
# -----------------------------------------------------------------------------

@dataclass
class AxisInfo:
    x_axis: str = "X"
    y_axis: str = "Y"
    z_axis: str = "Z"
    b_axis: str = "B"
    c_axis: str = "C"
    c_180_axis_units: float = 180.0


@dataclass
class BranchModels:
    tip_angle: dict[str, Any]
    r: dict[str, Any]
    z: dict[str, Any]
    offplane_y: dict[str, Any] | None = None


@dataclass
class CirclePlan:
    dense_df: pd.DataFrame
    capture_df: pd.DataFrame
    print_gcode_path: Path
    print_points_csv_path: Path
    metadata_path: Path


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Trace/capture/export an XZ circle using global or curl-specific pull/release equations."
    )

    ap.add_argument("--project_name", type=str, default=DEFAULT_PROJECT_NAME)
    ap.add_argument("--calibration_file", type=str, required=True)
    ap.add_argument("--duet_web_address", type=str, default=DEFAULT_DUET_WEB_ADDRESS)
    ap.add_argument("--cam_port", type=int, default=DEFAULT_CAMERA_PORT)
    ap.add_argument("--show_preview", action="store_true", default=False)
    ap.add_argument("--export_only", action="store_true", default=False, help="Only build CSV/G-code/plots; do not connect to camera or robot.")
    ap.add_argument("--process_only", action="store_true", default=False, help="Only process existing raw images for an existing project folder; do not build plans or connect to hardware.")
    ap.add_argument("--process_plan_csv", type=str, default="", help="Optional capture-plan CSV to use for process-only mode. Defaults to the project's circle_capture_plan.csv.")

    # Circle geometry in tip/world space.
    ap.add_argument("--center_x", type=float, default=DEFAULT_CENTER_X)
    ap.add_argument("--center_y", type=float, default=DEFAULT_CENTER_Y)
    ap.add_argument("--center_z", type=float, default=DEFAULT_CENTER_Z)
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    ap.add_argument("--points_per_mm", type=float, default=DEFAULT_POINTS_PER_MM)
    ap.add_argument("--capture_points_per_half", type=int, default=DEFAULT_CAPTURE_POINTS_PER_HALF)
    ap.add_argument("--max_capture_points_per_half", type=int, default=0, help="Alias/limit for quick tests. 0 disables limit.")
    ap.add_argument(
        "--debug_motion_points_per_half",
        "--motion_points_per_half",
        dest="debug_motion_points_per_half",
        type=int,
        default=DEFAULT_DEBUG_MOTION_POINTS_PER_HALF,
        help="Debug override for dense motion samples per half-circle. 0 uses points_per_mm.",
    )
    ap.add_argument("--precurl_steps", type=int, default=DEFAULT_PRECURL_STEPS, help="Samples for the fixed-tip 0->180 start curl. Use a small value for debugging.")
    ap.add_argument("--start_with_fixed_tip_precurl", action="store_true", default=True, help="Start at attack 0, then curl 0->180 while tracking the circle start point before release.")
    ap.add_argument("--no_start_with_fixed_tip_precurl", dest="start_with_fixed_tip_precurl", action="store_false")

    # Acquisition pass selection.
    ap.add_argument("--capture_compensated", action="store_true", default=True)
    ap.add_argument("--no_capture_compensated", dest="capture_compensated", action="store_false")
    ap.add_argument("--capture_noncompensated", action="store_true", default=True)
    ap.add_argument("--no_capture_noncompensated", dest="capture_noncompensated", action="store_false")
    ap.add_argument("--repeats", type=int, default=1)

    # Feedrates and extrusion.
    ap.add_argument("--travel_feed", type=float, default=DEFAULT_TRAVEL_FEED)
    ap.add_argument("--approach_feed", type=float, default=DEFAULT_APPROACH_FEED)
    ap.add_argument("--fine_approach_feed", type=float, default=DEFAULT_FINE_APPROACH_FEED)
    ap.add_argument("--initial_xyzc_feed", type=float, default=DEFAULT_INITIAL_XYZC_FEED, help="Very first prepositioning feedrate for X/Y/Z/C only. Default F5000.")
    ap.add_argument("--initial_b_feed", type=float, default=DEFAULT_INITIAL_B_FEED, help="Very first B-only prepositioning feedrate. Default F400.")
    ap.add_argument("--initial_daq_wait_s", type=float, default=DEFAULT_INITIAL_DAQ_WAIT_S, help="Wait after the initial XYZC + B prepositioning before starting acquisition/print motion. Default 6 s.")
    ap.add_argument("--trace_feed", type=float, default=DEFAULT_TRACE_FEED)
    ap.add_argument("--c_spin_feed", type=float, default=DEFAULT_C_SPIN_FEED)
    ap.add_argument("--spin_steps", type=int, default=DEFAULT_SPIN_STEPS)
    ap.add_argument("--emit_extrusion", action="store_true", default=DEFAULT_EMIT_EXTRUSION)
    ap.add_argument("--preflow_dwell_ms", type=int, default=DEFAULT_PREFLOW_DWELL_MS)
    ap.add_argument("--pressure_output_pin", type=int, default=DEFAULT_PRESSURE_OUTPUT_PIN, help="Duet M42 output pin for extrusion pressure. Default P10, so extrusion on/off is M42 P10 S1 / M42 P10 S0.")
    ap.add_argument("--curve_overtrace_mm", type=float, default=8.0, help="Extra arc length to trace after the nominal circle closes, while staying fully curled. Default 8 mm.")
    ap.add_argument("--combined_noncomp_y_offset", type=float, default=5.0, help="Y offset applied to the non-compensated section in the combined print G-code. Default 5 mm.")
    ap.add_argument("--combined_transition_z", type=float, default=-10.0, help="Absolute Z move inserted between the compensated and non-compensated sections in the combined print G-code. Default Z-10.")
    ap.add_argument("--end_delta_x", type=float, default=-5.0, help="Final relative X move appended after the print path. Default -5 mm.")
    ap.add_argument("--end_delta_z", type=float, default=-10.0, help="Final relative Z move appended after the print path. Default -10 mm.")
    ap.add_argument("--end_z_feed", type=float, default=2000.0, help="Feedrate for the final Z move. Default F2000.")
    ap.add_argument("--end_b_target", type=float, default=0.0, help="Final absolute B target appended after the print path. Default B0.")
    ap.add_argument("--end_b_feed", type=float, default=200.0, help="Feedrate for the final B move. Default F200.")

    # Sign/model selection.
    ap.add_argument("--curve_set", type=str, default="0-180-0", help="Branch model source to use: 'global' for top-level fit_models_by_phase, or a curl-specific key such as '0-180-0'. Default: 0-180-0.")
    ap.add_argument("--flip_rz_sign", action="store_true", default=DEFAULT_FLIP_RZ_SIGN)
    ap.add_argument("--no_flip_rz_sign", dest="flip_rz_sign", action="store_false")
    ap.add_argument("--offplane_y_sign", type=float, default=DEFAULT_OFFPLANE_Y_SIGN)
    ap.add_argument("--allow_global_fallback", action="store_true", default=False, help="If a curl-specific --curve_set is missing, fall back to global fit_models_by_phase.")

    # Camera / red-dot analysis defaults from the hysteresis runner.
    ap.add_argument("--manual_crop_adjustment", action="store_true", default=DEFAULT_MANUAL_CROP_ADJUSTMENT)
    ap.add_argument("--no_manual_crop_adjustment", dest="manual_crop_adjustment", action="store_false")
    ap.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    ap.add_argument("--export_analysis_outputs", action="store_true", default=False)
    ap.add_argument("--camera_calibration_file", type=str, default=str(DEFAULT_CAMERA_CALIBRATION_FILE))
    ap.add_argument("--board_reference_image", type=str, default=str(DEFAULT_BOARD_REFERENCE_IMAGE))
    ap.add_argument("--board_xz_axis_sign", type=int, choices=[-1, 1], default=1)
    ap.add_argument("--width_in_pixels", type=float, default=3025.0)
    ap.add_argument("--width_in_mm", type=float, default=140.0)
    ap.add_argument("--camera_width", type=int, default=DEFAULT_CAMERA_WIDTH)
    ap.add_argument("--camera_height", type=int, default=DEFAULT_CAMERA_HEIGHT)
    ap.add_argument("--camera_flush_frames", type=int, default=DEFAULT_CAMERA_FLUSH_FRAMES)
    ap.add_argument("--enable_manual_focus", action="store_true", default=DEFAULT_MANUAL_FOCUS)
    ap.add_argument("--no_enable_manual_focus", dest="enable_manual_focus", action="store_false")
    ap.add_argument("--manual_focus_val", type=float, default=DEFAULT_MANUAL_FOCUS_VAL)

    ap.add_argument("--tip_detection_mode", type=str, default="red_dot", choices=["classical", "red_dot", "auto_red_dot"])
    ap.add_argument("--tip_refine_mode", type=str, default="coarse", choices=["coarse", "parallel_centerline", "auto"])
    ap.add_argument("--red_tip_sat_min", type=int, default=80)
    ap.add_argument("--red_tip_val_min", type=int, default=80)
    ap.add_argument("--red_tip_min_area_px", type=int, default=20)
    ap.add_argument("--red_tip_morph_kernel", type=int, default=1)
    ap.add_argument("--red_tip_hue1_min", type=int, default=0)
    ap.add_argument("--red_tip_hue1_max", type=int, default=10)
    ap.add_argument("--red_tip_hue2_min", type=int, default=150)
    ap.add_argument("--red_tip_hue2_max", type=int, default=179)
    ap.add_argument("--red_tip_search_radius_px", type=float, default=180.0)
    ap.add_argument("--red_tip_local_min_area_px", type=int, default=10)
    ap.add_argument("--red_tip_distance_weight", type=float, default=3.0)
    ap.add_argument("--red_tip_min_circularity", type=float, default=0.0)
    ap.add_argument("--red_tip_component_selection", type=str, default="nearest_largest", choices=["largest", "nearest", "nearest_largest"])
    ap.add_argument("--red_tip_use_rgb_excess", dest="red_tip_use_rgb_excess", action="store_true", default=True)
    ap.add_argument("--no_red_tip_use_rgb_excess", dest="red_tip_use_rgb_excess", action="store_false")
    ap.add_argument("--red_tip_rgb_excess_min", type=int, default=35)
    ap.add_argument("--red_tip_debug_save_mask", action="store_true", default=True)
    ap.add_argument("--no_red_tip_debug_save_mask", dest="red_tip_debug_save_mask", action="store_false")

    ap.add_argument("--capture_dwell_s", type=float, default=0.5)
    ap.add_argument("--settle_after_start_move_s", type=float, default=1.0)
    ap.add_argument("--move_buffer_s", type=float, default=0.2)
    ap.add_argument("--post_capture_buffer_s", type=float, default=0.15)
    ap.add_argument("--append_raw_data", action="store_true", default=False)
    ap.add_argument("--return_to_b0_on_exit", action="store_true", default=True)
    ap.add_argument("--no_return_to_b0_on_exit", dest="return_to_b0_on_exit", action="store_false")

    return ap


# -----------------------------------------------------------------------------
# Generic helpers
# -----------------------------------------------------------------------------


def safe_token(value: Any) -> str:
    text = str(value).strip().replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9+\-]+", "_", text).strip("_")
    return text or "x"


def format_feedrate(feedrate: float) -> str:
    return str(int(round(float(feedrate))))


def json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def circle_point(center_x: float, center_y: float, center_z: float, radius: float, theta_deg: float) -> np.ndarray:
    th = math.radians(float(theta_deg))
    return np.array(
        [
            float(center_x) + float(radius) * math.cos(th),
            float(center_y),
            float(center_z) + float(radius) * math.sin(th),
        ],
        dtype=float,
    )


def c_axis_to_physical_deg(c_cmd: float, c_180_axis_units: float) -> float:
    if abs(float(c_180_axis_units)) < 1e-12:
        return float(c_cmd)
    return 180.0 * float(c_cmd) / float(c_180_axis_units)


def estimate_motion_time_s(distance_mm: float, feedrate_mm_per_min: float) -> float:
    if not np.isfinite(distance_mm) or distance_mm <= 0.0:
        return 0.0
    if not np.isfinite(feedrate_mm_per_min) or feedrate_mm_per_min <= 0.0:
        return 0.0
    return 60.0 * float(distance_mm) / float(feedrate_mm_per_min)


# -----------------------------------------------------------------------------
# Calibration loading/model helpers
# -----------------------------------------------------------------------------


def load_calibration_export(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")
    if path.suffix.lower() == ".json":
        with path.open("r") as f:
            return json.load(f)
    if path.suffix.lower() in {".pkl", ".pickle"}:
        with path.open("rb") as f:
            return pickle.load(f)
    raise ValueError(f"Unsupported calibration file extension: {path.suffix}")


def _norm_name(value: Any) -> str:
    return str(value).strip().lower().replace("_", "-")


def get_axis_info(cal_data: dict[str, Any]) -> AxisInfo:
    motor_setup = cal_data.get("motor_setup", {}) or {}
    duet_map = cal_data.get("duet_axis_mapping", {}) or {}
    return AxisInfo(
        x_axis=str(duet_map.get("horizontal_axis") or motor_setup.get("horizontal_axis") or "X"),
        y_axis=str(duet_map.get("depth_axis") or motor_setup.get("depth_axis") or "Y"),
        z_axis=str(duet_map.get("vertical_axis") or motor_setup.get("vertical_axis") or "Z"),
        b_axis=str(duet_map.get("pull_axis") or motor_setup.get("b_motor_axis") or "B"),
        c_axis=str(duet_map.get("rotation_axis") or motor_setup.get("rotation_axis") or "C"),
        c_180_axis_units=float(motor_setup.get("rotation_axis_180_deg", 180.0)),
    )


def get_model_descriptor(bundle: dict[str, Any], quantity: str, fit_family: str = "pchip") -> dict[str, Any] | None:
    quantity = str(quantity).strip().lower()
    fit_family = str(fit_family).strip().lower()
    if quantity == "tip_angle":
        candidates = [
            f"tip_angle_{fit_family}",
            "tip_angle_pchip",
            "tip_angle",
            "tip_angle_avg_pchip",
            "angle_pchip",
            "angle",
        ]
    elif quantity in {"r", "z", "offplane_y"}:
        candidates = [
            f"{quantity}_{fit_family}",
            f"{quantity}_pchip",
            quantity,
            f"{quantity}_avg_pchip",
        ]
    else:
        candidates = [f"{quantity}_{fit_family}", f"{quantity}_pchip", quantity]
    for key in candidates:
        descriptor = bundle.get(key)
        if isinstance(descriptor, dict):
            return descriptor
    return None


def _phase_bundle_from_payload(payload: dict[str, Any], phase: str) -> dict[str, Any]:
    phase_norm = _norm_name(phase)
    by_phase = payload.get("fit_models_by_phase") if "fit_models_by_phase" in payload else payload
    if not isinstance(by_phase, dict):
        raise KeyError("Expected fit_models_by_phase dictionary.")
    for key, bundle in by_phase.items():
        key_norm = _norm_name(key)
        if key_norm == phase_norm or key_norm.startswith(phase_norm):
            if isinstance(bundle, dict):
                return bundle
    available = sorted(str(k) for k in by_phase.keys())
    raise KeyError(f"No {phase} bundle found. Available phase keys: {available}")


def get_curl_specific_models(cal_data: dict[str, Any], curve_set: str, allow_global_fallback: bool = False) -> tuple[BranchModels, BranchModels, str]:
    requested_norm = _norm_name(curve_set)
    curl_payload = cal_data.get("curl_angle_specific_fit_models") or {}
    selected_key = None
    selected_payload = None

    if requested_norm in {"", "global", "globalfallback", "fitmodelsbyphase"}:
        selected_key = "global"
        selected_payload = {"fit_models_by_phase": cal_data.get("fit_models_by_phase", {})}

    if isinstance(curl_payload, dict):
        for key, payload in curl_payload.items():
            if _norm_name(key) == requested_norm and isinstance(payload, dict):
                selected_key = str(key)
                selected_payload = payload
                break

    if selected_payload is None:
        if not allow_global_fallback:
            available = sorted(str(k) for k in curl_payload.keys()) if isinstance(curl_payload, dict) else []
            raise KeyError(
                f"Calibration does not contain curl_angle_specific_fit_models['{curve_set}']. "
                f"Available curl-specific sets: {available}. "
                "Use --curve_set global, or pass --allow_global_fallback to fall back automatically."
            )
        selected_key = "global_fallback"
        selected_payload = {"fit_models_by_phase": cal_data.get("fit_models_by_phase", {})}

    pull_bundle = _phase_bundle_from_payload(selected_payload, "pull")
    release_bundle = _phase_bundle_from_payload(selected_payload, "release")

    def make_branch(bundle: dict[str, Any], branch_name: str) -> BranchModels:
        tip = get_model_descriptor(bundle, "tip_angle", "pchip")
        r = get_model_descriptor(bundle, "r", "pchip")
        z = get_model_descriptor(bundle, "z", "pchip")
        off = get_model_descriptor(bundle, "offplane_y", "pchip")
        missing = []
        if tip is None:
            missing.append(f"{branch_name} tip_angle")
        if r is None:
            missing.append(f"{branch_name} r")
        if z is None:
            missing.append(f"{branch_name} z")
        if missing:
            raise RuntimeError("Missing required branch PCHIP models: " + ", ".join(missing))
        return BranchModels(tip_angle=tip, r=r, z=z, offplane_y=off)

    return make_branch(pull_bundle, "pull"), make_branch(release_bundle, "release"), str(selected_key)


def attach_calibration_state_for_class(cal: CTR_Shadow_Calibration, cal_data: dict[str, Any]) -> None:
    if isinstance(cal_data.get("fit_models_by_phase"), dict):
        cal._postprocessed_fit_models_by_phase = cal_data["fit_models_by_phase"]
    if isinstance(cal_data.get("datasets_by_phase"), dict):
        cal._postprocessed_datasets = cal_data["datasets_by_phase"]
    if isinstance(cal_data.get("redundancy_diagnostics"), dict):
        cal._postprocessed_redundancy_diagnostics = cal_data["redundancy_diagnostics"]


def eval_model_array(cal: CTR_Shadow_Calibration, model: dict[str, Any] | None, b_values: Any, default: float | None = None) -> np.ndarray:
    b_arr = np.asarray(b_values, dtype=float)
    if model is None:
        if default is None:
            return np.full_like(b_arr, np.nan, dtype=float)
        return np.full_like(b_arr, float(default), dtype=float)
    pred = cal._evaluate_curve_model(model, b_arr.reshape(-1))
    return np.asarray(pred, dtype=float).reshape(b_arr.shape)


def eval_model_scalar(cal: CTR_Shadow_Calibration, model: dict[str, Any] | None, b_value: float, default: float | None = None) -> float:
    arr = eval_model_array(cal, model, np.asarray([float(b_value)]), default=default)
    return float(np.asarray(arr).reshape(-1)[0])


def solve_b_for_angle(cal: CTR_Shadow_Calibration, tip_angle_model: dict[str, Any], target_angle_deg: float, reference_b: float) -> float:
    return float(
        cal._solve_curve_model_input_for_target(
            tip_angle_model,
            float(target_angle_deg),
            reference_b=float(reference_b),
        )
    )


def predict_tip_offset_xyz(
    cal: CTR_Shadow_Calibration,
    branch: BranchModels,
    b_cmd: float,
    c_cmd: float,
    axes: AxisInfo,
    flip_rz_sign: bool,
    offplane_y_sign: float,
) -> np.ndarray:
    # Match the fixed-tip tracking convention: with --flip_rz_sign enabled,
    # use the calibration r sign as-is; disabling it applies the legacy opposite sign.
    r_sign = 1.0 if bool(flip_rz_sign) else -1.0
    r = r_sign * eval_model_scalar(cal, branch.r, b_cmd)
    z = eval_model_scalar(cal, branch.z, b_cmd)
    y_off_raw = eval_model_scalar(cal, branch.offplane_y, b_cmd, default=0.0)
    y_off = float(offplane_y_sign) * y_off_raw
    c_phys_deg = c_axis_to_physical_deg(float(c_cmd), float(axes.c_180_axis_units))
    c = math.radians(c_phys_deg)
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def stage_xyz_for_tip(
    cal: CTR_Shadow_Calibration,
    branch: BranchModels,
    tip_xyz: np.ndarray,
    b_cmd: float,
    c_cmd: float,
    axes: AxisInfo,
    apply_tip_offset_correction: bool,
    flip_rz_sign: bool,
    offplane_y_sign: float,
) -> np.ndarray:
    tip = np.asarray(tip_xyz, dtype=float)
    if not bool(apply_tip_offset_correction):
        return tip.copy()
    return tip - predict_tip_offset_xyz(
        cal=cal,
        branch=branch,
        b_cmd=b_cmd,
        c_cmd=c_cmd,
        axes=axes,
        flip_rz_sign=flip_rz_sign,
        offplane_y_sign=offplane_y_sign,
    )


# -----------------------------------------------------------------------------
# Trajectory generation
# -----------------------------------------------------------------------------


def _capture_indices(n: int, count: int) -> set[int]:
    n = int(n)
    count = max(2, int(count))
    if count >= n:
        return set(range(n))
    idx = np.linspace(0, n - 1, count).round().astype(int)
    return set(int(v) for v in np.unique(idx))


def build_circle_dense_rows(
    cal: CTR_Shadow_Calibration,
    args: argparse.Namespace,
    axes: AxisInfo,
    pull: BranchModels,
    release: BranchModels,
    compensated: bool,
    pass_name: str,
    repeat_index: int,
) -> list[dict[str, Any]]:
    radius = float(args.radius)
    if radius <= 0.0:
        raise ValueError("--radius must be positive.")
    points_per_mm = float(args.points_per_mm)
    if points_per_mm <= 0.0:
        raise ValueError("--points_per_mm must be positive.")

    if int(getattr(args, "debug_motion_points_per_half", 0)) > 0:
        n_half = max(2, int(args.debug_motion_points_per_half))
    else:
        n_half = max(2, int(math.ceil(math.pi * radius * points_per_mm)) + 1)
    cap_count = int(args.capture_points_per_half)
    if int(args.max_capture_points_per_half) > 0:
        cap_count = min(cap_count, int(args.max_capture_points_per_half))
    cap_rel = _capture_indices(n_half, cap_count)
    cap_pull = _capture_indices(n_half, cap_count)

    c_release = float(axes.c_180_axis_units)
    c_pull = 0.0

    start_tip = circle_point(args.center_x, args.center_y, args.center_z, radius, 180.0)
    right_tip = circle_point(args.center_x, args.center_y, args.center_z, radius, 360.0)

    b_pull_180 = solve_b_for_angle(cal, pull.tip_angle, 180.0, reference_b=0.0)
    b_pull_0 = solve_b_for_angle(cal, pull.tip_angle, 0.0, reference_b=b_pull_180)

    rows: list[dict[str, Any]] = []
    point_counter = 0
    use_single_branch_pull = not bool(compensated)
    apply_tip_offset_correction = True

    def add_row(
        segment_name: str,
        motion_phase: str,
        theta_deg: float,
        attack_angle_deg: float,
        branch: BranchModels,
        b_cmd: float,
        c_cmd: float,
        tip_xyz: np.ndarray,
        stage_xyz: np.ndarray,
        feedrate: float,
        capture_image: bool,
        local_index: int,
    ) -> None:
        nonlocal point_counter
        image_file = ""
        if capture_image:
            image_file = (
                f"circle_R{repeat_index + 1:02d}_{safe_token(pass_name)}_"
                f"{safe_token(segment_name)}_P{point_counter:04d}_"
                f"A{attack_angle_deg:.2f}_T{theta_deg:.2f}.png"
            )
        rows.append(
            {
                "repeat_index": int(repeat_index),
                "repeat_number": int(repeat_index + 1),
                "pass_name": str(pass_name),
                "compensated": bool(compensated),
                "point_order": int(point_counter),
                "local_index": int(local_index),
                "segment_name": str(segment_name),
                "motion_phase": str(motion_phase),
                "theta_deg": float(theta_deg),
                "attack_angle_deg": float(attack_angle_deg),
                "b_cmd": float(b_cmd),
                "c_cmd": float(c_cmd),
                "tip_x_desired": float(tip_xyz[0]),
                "tip_y_desired": float(tip_xyz[1]),
                "tip_z_desired": float(tip_xyz[2]),
                "x_cmd": float(stage_xyz[0]),
                "y_cmd": float(stage_xyz[1]),
                "z_cmd": float(stage_xyz[2]),
                "feedrate": float(feedrate),
                "capture_image": bool(capture_image),
                "image_file": image_file,
            }
        )
        point_counter += 1

    # Start strategy: go to the circle start at attack 0, then curl in place
    # from 0 -> 180 using the pull branch while tracking the same start point.
    # The subsequent release/pull path is therefore a 0-180-0-180 branch sequence.
    b_start_0 = solve_b_for_angle(cal, pull.tip_angle, 0.0, reference_b=0.0)
    if bool(getattr(args, "start_with_fixed_tip_precurl", True)):
        pre_stage_0 = stage_xyz_for_tip(
            cal=cal,
            branch=pull,
            tip_xyz=start_tip,
            b_cmd=b_start_0,
            c_cmd=c_release,
            axes=axes,
            apply_tip_offset_correction=apply_tip_offset_correction,
            flip_rz_sign=bool(args.flip_rz_sign),
            offplane_y_sign=float(args.offplane_y_sign),
        )
        add_row(
            segment_name="preposition_start_attack_0",
            motion_phase="pull_preposition_b0",
            theta_deg=180.0,
            attack_angle_deg=0.0,
            branch=pull,
            b_cmd=b_start_0,
            c_cmd=c_release,
            tip_xyz=start_tip,
            stage_xyz=pre_stage_0,
            feedrate=float(args.approach_feed),
            capture_image=False,
            local_index=0,
        )

        precurl_steps = max(2, int(getattr(args, "precurl_steps", DEFAULT_PRECURL_STEPS)))
        prev_b_precurl = b_start_0
        for j, attack in enumerate(np.linspace(0.0, 180.0, precurl_steps)):
            b_precurl = solve_b_for_angle(cal, pull.tip_angle, float(attack), reference_b=prev_b_precurl)
            prev_b_precurl = b_precurl
            pre_stage = stage_xyz_for_tip(
                cal=cal,
                branch=pull,
                tip_xyz=start_tip,
                b_cmd=b_precurl,
                c_cmd=c_release,
                axes=axes,
                apply_tip_offset_correction=apply_tip_offset_correction,
                flip_rz_sign=bool(args.flip_rz_sign),
                offplane_y_sign=float(args.offplane_y_sign),
            )
            add_row(
                segment_name="fixed_tip_precurl_0_to_180",
                motion_phase="pull_precurl",
                theta_deg=180.0,
                attack_angle_deg=float(attack),
                branch=pull,
                b_cmd=b_precurl,
                c_cmd=c_release,
                tip_xyz=start_tip,
                stage_xyz=pre_stage,
                feedrate=float(args.trace_feed),
                capture_image=False,
                local_index=j,
            )
        b_at_release_start = prev_b_precurl
    else:
        # Legacy behavior: preposition directly at attack 180.
        pre_stage = stage_xyz_for_tip(
            cal=cal,
            branch=pull,
            tip_xyz=start_tip,
            b_cmd=b_pull_180,
            c_cmd=c_release,
            axes=axes,
            apply_tip_offset_correction=apply_tip_offset_correction,
            flip_rz_sign=bool(args.flip_rz_sign),
            offplane_y_sign=float(args.offplane_y_sign),
        )
        add_row(
            segment_name="prepull_to_180",
            motion_phase="pull_preposition",
            theta_deg=180.0,
            attack_angle_deg=180.0,
            branch=pull,
            b_cmd=b_pull_180,
            c_cmd=c_release,
            tip_xyz=start_tip,
            stage_xyz=pre_stage,
            feedrate=float(args.approach_feed),
            capture_image=False,
            local_index=0,
        )
        b_at_release_start = b_pull_180

    # First half: release 180 -> 0 while tip traces left->bottom->right.
    prev_b = b_at_release_start
    theta_vals = np.linspace(180.0, 360.0, n_half)
    attack_vals = np.linspace(180.0, 0.0, n_half)
    for i, (theta, attack) in enumerate(zip(theta_vals, attack_vals)):
        active_branch = pull if use_single_branch_pull else release
        b_cmd = solve_b_for_angle(cal, active_branch.tip_angle, float(attack), reference_b=prev_b)
        prev_b = b_cmd
        tip_xyz = circle_point(args.center_x, args.center_y, args.center_z, radius, float(theta))
        stage = stage_xyz_for_tip(
            cal=cal,
            branch=active_branch,
            tip_xyz=tip_xyz,
            b_cmd=b_cmd,
            c_cmd=c_release,
            axes=axes,
            apply_tip_offset_correction=apply_tip_offset_correction,
            flip_rz_sign=bool(args.flip_rz_sign),
            offplane_y_sign=float(args.offplane_y_sign),
        )
        add_row(
            segment_name="release_180_to_0",
            motion_phase="release",
            theta_deg=float(theta),
            attack_angle_deg=float(attack),
            branch=release,
            b_cmd=b_cmd,
            c_cmd=c_release,
            tip_xyz=tip_xyz,
            stage_xyz=stage,
            feedrate=float(args.trace_feed),
            capture_image=(i in cap_rel),
            local_index=i,
        )

    # Fixed-tip C spin at attack 0, switching to pull branch at the rightmost point.
    b_pull_0 = solve_b_for_angle(cal, pull.tip_angle, 0.0, reference_b=prev_b)
    spin_n = max(1, int(args.spin_steps))
    for s in range(1, spin_n + 1):
        u = s / float(spin_n)
        c_here = (1.0 - u) * c_release + u * c_pull
        stage = stage_xyz_for_tip(
            cal=cal,
            branch=pull,
            tip_xyz=right_tip,
            b_cmd=b_pull_0,
            c_cmd=c_here,
            axes=axes,
            apply_tip_offset_correction=apply_tip_offset_correction,
            flip_rz_sign=bool(args.flip_rz_sign),
            offplane_y_sign=float(args.offplane_y_sign),
        )
        add_row(
            segment_name="fixed_tip_c_spin",
            motion_phase="spin_to_pull",
            theta_deg=360.0,
            attack_angle_deg=0.0,
            branch=pull,
            b_cmd=b_pull_0,
            c_cmd=c_here,
            tip_xyz=right_tip,
            stage_xyz=stage,
            feedrate=float(args.c_spin_feed),
            capture_image=False,
            local_index=s,
        )

    # Second half: pull/curl 0 -> 180 while tip traces right->top->left.
    prev_b = b_pull_0
    theta_vals = np.linspace(0.0, 180.0, n_half)
    attack_vals = np.linspace(0.0, 180.0, n_half)
    for i, (theta, attack) in enumerate(zip(theta_vals, attack_vals)):
        b_cmd = solve_b_for_angle(cal, pull.tip_angle, float(attack), reference_b=prev_b)
        prev_b = b_cmd
        tip_xyz = circle_point(args.center_x, args.center_y, args.center_z, radius, float(theta))
        stage = stage_xyz_for_tip(
            cal=cal,
            branch=pull,
            tip_xyz=tip_xyz,
            b_cmd=b_cmd,
            c_cmd=c_pull,
            axes=axes,
            apply_tip_offset_correction=apply_tip_offset_correction,
            flip_rz_sign=bool(args.flip_rz_sign),
            offplane_y_sign=float(args.offplane_y_sign),
        )
        add_row(
            segment_name="pull_0_to_180",
            motion_phase="pull",
            theta_deg=float(theta),
            attack_angle_deg=float(attack),
            branch=pull,
            b_cmd=b_cmd,
            c_cmd=c_pull,
            tip_xyz=tip_xyz,
            stage_xyz=stage,
            feedrate=float(args.trace_feed),
            capture_image=(i in cap_pull),
            local_index=i,
        )

    overtrace_mm = max(0.0, float(getattr(args, "curve_overtrace_mm", 0.0)))
    if overtrace_mm > 0.0:
        extra_angle_deg = math.degrees(overtrace_mm / radius)
        extra_steps = max(1, int(math.ceil(overtrace_mm * points_per_mm)))
        theta_over = np.linspace(180.0, 180.0 + extra_angle_deg, extra_steps + 1)[1:]
        local_offset = n_half
        for j, theta in enumerate(theta_over, start=1):
            tip_xyz = circle_point(args.center_x, args.center_y, args.center_z, radius, float(theta))
            stage = stage_xyz_for_tip(
                cal=cal,
                branch=pull,
                tip_xyz=tip_xyz,
                b_cmd=prev_b,
                c_cmd=c_pull,
                axes=axes,
                apply_tip_offset_correction=apply_tip_offset_correction,
                flip_rz_sign=bool(args.flip_rz_sign),
                offplane_y_sign=float(args.offplane_y_sign),
            )
            add_row(
                segment_name="pull_overtrace_close_gap",
                motion_phase="pull_overtrace",
                theta_deg=float(theta),
                attack_angle_deg=180.0,
                branch=pull,
                b_cmd=prev_b,
                c_cmd=c_pull,
                tip_xyz=tip_xyz,
                stage_xyz=stage,
                feedrate=float(args.trace_feed),
                capture_image=False,
                local_index=local_offset + j,
            )

    return rows


def build_full_plan(
    cal: CTR_Shadow_Calibration,
    args: argparse.Namespace,
    axes: AxisInfo,
    pull: BranchModels,
    release: BranchModels,
) -> pd.DataFrame:
    all_rows: list[dict[str, Any]] = []
    repeats = max(1, int(args.repeats))
    for r in range(repeats):
        if bool(args.capture_compensated):
            all_rows.extend(
                build_circle_dense_rows(cal, args, axes, pull, release, compensated=True, pass_name="compensated", repeat_index=r)
            )
        if bool(args.capture_noncompensated):
            all_rows.extend(
                build_circle_dense_rows(cal, args, axes, pull, release, compensated=False, pass_name="noncompensated", repeat_index=r)
            )
    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("No acquisition rows were generated. Enable at least one capture pass.")
    df.insert(0, "global_order", np.arange(len(df), dtype=int))
    return df


# -----------------------------------------------------------------------------
# Print G-code export
# -----------------------------------------------------------------------------


def write_print_gcode(
    df: pd.DataFrame,
    args: argparse.Namespace,
    axes: AxisInfo,
    output_dir: Path,
) -> tuple[Path, Path, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    def _emit_path_section(
        f,
        rows: pd.DataFrame,
        section_label: str,
        y_offset_mm: float,
        include_final_clearance: bool,
    ) -> None:
        extrusion_on = False
        last_segment = None
        first_row = rows.iloc[0]
        f.write(f"; --- {section_label}: initial split preposition before print/capture path ---\n")
        f.write(
            "G1 "
            f"{axes.x_axis}{float(first_row['x_cmd']):.5f} "
            f"{axes.y_axis}{float(first_row['y_cmd']) + float(y_offset_mm):.5f} "
            f"{axes.z_axis}{float(first_row['z_cmd']):.5f} "
            f"{axes.c_axis}{float(first_row['c_cmd']):.5f} "
            f"F{format_feedrate(float(args.initial_xyzc_feed))}\n"
        )
        f.write(
            "G1 "
            f"{axes.b_axis}{float(first_row['b_cmd']):.5f} "
            f"F{format_feedrate(float(args.initial_b_feed))}\n"
        )
        f.write("M400 ; wait for initial prepositioning moves to finish\n")
        if float(args.initial_daq_wait_s) > 0.0:
            f.write(f"G4 S{float(args.initial_daq_wait_s):.3f} ; initial settle before path starts\n")

        for _, row in rows.iloc[1:].iterrows():
            segment = str(row["segment_name"])
            if segment != last_segment:
                f.write(f"; --- {section_label}: {segment} ---\n")
                last_segment = segment

            if bool(args.emit_extrusion):
                should_extrude = segment in {"release_180_to_0", "pull_0_to_180", "pull_overtrace_close_gap"}
                if should_extrude and not extrusion_on:
                    f.write("; pressure on\n")
                    f.write(f"M42 P{int(args.pressure_output_pin)} S1\n")
                    if int(args.preflow_dwell_ms) > 0:
                        f.write(f"G4 P{int(args.preflow_dwell_ms)}\n")
                    extrusion_on = True
                elif (not should_extrude) and extrusion_on:
                    f.write("; pressure off\n")
                    f.write(f"M42 P{int(args.pressure_output_pin)} S0\n")
                    extrusion_on = False

            f.write(
                "G1 "
                f"{axes.x_axis}{float(row['x_cmd']):.5f} "
                f"{axes.y_axis}{float(row['y_cmd']) + float(y_offset_mm):.5f} "
                f"{axes.z_axis}{float(row['z_cmd']):.5f} "
                f"{axes.b_axis}{float(row['b_cmd']):.5f} "
                f"{axes.c_axis}{float(row['c_cmd']):.5f} "
                f"F{format_feedrate(float(row['feedrate']))}\n"
            )

        if extrusion_on:
            f.write("; pressure off\n")
            f.write(f"M42 P{int(args.pressure_output_pin)} S0\n")
        if include_final_clearance:
            _emit_clearance_trailer(f)

    def _emit_clearance_trailer(f) -> None:
        f.write("; --- end clearance moves ---\n")
        f.write("G91 ; relative X clearance move\n")
        f.write(
            "G1 "
            f"{axes.x_axis}{float(args.end_delta_x):.5f} "
            f"F{format_feedrate(float(args.travel_feed))}\n"
        )
        f.write("G90 ; back to absolute positioning for Z/B\n")
        f.write(
            "G1 "
            f"{axes.z_axis}{float(args.end_delta_z):.5f} "
            f"F{format_feedrate(float(args.end_z_feed))}\n"
        )
        f.write(
            "G1 "
            f"{axes.b_axis}{float(args.end_b_target):.5f} "
            f"F{format_feedrate(float(args.end_b_feed))}\n"
        )
        f.write("M400 ; wait for all moves\n")
        f.write("; END\n")

    def _write_single_print_path(path_df: pd.DataFrame, pass_name: str, strategy_note: str) -> tuple[Path, Path, Path]:
        rows = path_df.sort_values("point_order").copy()
        if rows.empty:
            raise RuntimeError(f"No {pass_name} path is available for print G-code export.")

        gcode_path_local = output_dir / f"circle_0_180_0_{pass_name}_print.gcode"
        points_csv_local = output_dir / f"circle_0_180_0_{pass_name}_print_points.csv"
        metadata_path_local = output_dir / f"circle_0_180_0_{pass_name}_print_metadata.json"

        with gcode_path_local.open("w") as f:
            f.write(f"; Circle 0-180-0 {pass_name} print G-code\n")
            f.write("; Generated by circle_0_180_capture_and_gcode.py\n")
            f.write(f"; center=({args.center_x:.6f}, {args.center_y:.6f}, {args.center_z:.6f}) radius={args.radius:.6f}\n")
            f.write(f"; Strategy: {strategy_note}\n")
            f.write("G90 ; absolute positioning\n")
            f.write("G94 ; units/min\n")
            _emit_path_section(
                f,
                rows,
                section_label=str(pass_name),
                y_offset_mm=0.0,
                include_final_clearance=True,
            )

        rows.to_csv(points_csv_local, index=False)
        metadata = {
            "gcode_path": str(gcode_path_local),
            "points_csv": str(points_csv_local),
            "pass_name": str(pass_name),
            "center": [float(args.center_x), float(args.center_y), float(args.center_z)],
            "radius": float(args.radius),
            "points_per_mm": float(args.points_per_mm),
            "emit_extrusion": bool(args.emit_extrusion),
            "axis_info": json_ready(axes.__dict__),
            "row_count": int(len(rows)),
            "initial_xyzc_feed": float(args.initial_xyzc_feed),
            "initial_b_feed": float(args.initial_b_feed),
            "initial_daq_wait_s": float(args.initial_daq_wait_s),
            "capture_points_per_half": int(args.capture_points_per_half),
            "debug_motion_points_per_half": int(getattr(args, "debug_motion_points_per_half", 0)),
            "precurl_steps": int(getattr(args, "precurl_steps", DEFAULT_PRECURL_STEPS)),
            "start_with_fixed_tip_precurl": bool(getattr(args, "start_with_fixed_tip_precurl", True)),
            "curve_overtrace_mm": float(getattr(args, "curve_overtrace_mm", 0.0)),
            "end_delta_x": float(args.end_delta_x),
            "end_delta_z": float(args.end_delta_z),
            "end_z_feed": float(args.end_z_feed),
            "end_b_target": float(args.end_b_target),
            "end_b_feed": float(args.end_b_feed),
        }
        metadata_path_local.write_text(json.dumps(json_ready(metadata), indent=2))
        return gcode_path_local, points_csv_local, metadata_path_local

    comp = df[(df["pass_name"] == "compensated") & (df["repeat_index"].astype(int) == 0)].copy()
    if comp.empty:
        raise RuntimeError("No compensated path is available for print G-code export.")
    gcode_path, points_csv, metadata_path = _write_single_print_path(
        comp,
        "compensated",
        "start at attack 0, fixed-tip pull 0->180, release 180->0 using curl-specific 0-180-0 release branch, C spin, pull 0->180 using curl-specific 0-180-0 pull branch.",
    )

    noncomp = df[(df["pass_name"] == "noncompensated") & (df["repeat_index"].astype(int) == 0)].copy()
    if not noncomp.empty:
        _write_single_print_path(
            noncomp.assign(y_cmd=noncomp["y_cmd"].astype(float) + float(args.combined_noncomp_y_offset)),
            "noncompensated",
            f"start at attack 0, fixed-tip pull 0->180, then use the pull/curl branch for the full circle while still applying predicted tip-offset correction, with Y offset {float(args.combined_noncomp_y_offset):.6f} mm.",
        )
        combined_gcode_path = output_dir / "circle_0_180_0_combined_comp_then_noncomp_print.gcode"
        combined_points_csv = output_dir / "circle_0_180_0_combined_comp_then_noncomp_print_points.csv"
        combined_metadata_path = output_dir / "circle_0_180_0_combined_comp_then_noncomp_print_metadata.json"
        comp_combined = comp.sort_values("point_order").copy()
        noncomp_combined = noncomp.sort_values("point_order").copy()
        noncomp_combined["y_cmd"] = noncomp_combined["y_cmd"].astype(float) + float(args.combined_noncomp_y_offset)
        combined_points_df = pd.concat([comp_combined, noncomp_combined], ignore_index=True)
        with combined_gcode_path.open("w") as f:
            f.write("; Circle 0-180-0 combined compensated then non-compensated print G-code\n")
            f.write("; Generated by circle_0_180_capture_and_gcode.py\n")
            f.write(f"; center=({args.center_x:.6f}, {args.center_y:.6f}, {args.center_z:.6f}) radius={args.radius:.6f}\n")
            f.write(
                f"; Strategy: run compensated path first, then run non-compensated curl-only path with Y offset {float(args.combined_noncomp_y_offset):.6f} mm.\n"
            )
            f.write("G90 ; absolute positioning\n")
            f.write("G94 ; units/min\n")
            _emit_path_section(
                f,
                comp.sort_values("point_order"),
                section_label="compensated",
                y_offset_mm=0.0,
                include_final_clearance=False,
            )
            f.write("; --- inter-circle clearance after compensated section ---\n")
            _emit_clearance_trailer(f)
            f.write(
                f"; --- transition to noncompensated section with Y offset {float(args.combined_noncomp_y_offset):.5f} mm ---\n"
            )
            f.write(
                "G1 "
                f"{axes.z_axis}{float(args.combined_transition_z):.5f} "
                f"F{format_feedrate(float(args.end_z_feed))}\n"
            )
            _emit_path_section(
                f,
                noncomp.sort_values("point_order"),
                section_label="noncompensated",
                y_offset_mm=float(args.combined_noncomp_y_offset),
                include_final_clearance=True,
            )
        combined_points_df.to_csv(combined_points_csv, index=False)
        combined_metadata = {
            "gcode_path": str(combined_gcode_path),
            "points_csv": str(combined_points_csv),
            "pass_sequence": ["compensated", "noncompensated"],
            "combined_noncomp_y_offset": float(args.combined_noncomp_y_offset),
            "combined_transition_z": float(args.combined_transition_z),
            "curve_overtrace_mm": float(getattr(args, "curve_overtrace_mm", 0.0)),
            "emit_extrusion": bool(args.emit_extrusion),
        }
        combined_metadata_path.write_text(json.dumps(json_ready(combined_metadata), indent=2))

    return gcode_path, points_csv, metadata_path


# -----------------------------------------------------------------------------
# CTR/camera setup
# -----------------------------------------------------------------------------


def configure_tip_detection(cal: CTR_Shadow_Calibration, args: argparse.Namespace) -> None:
    cal.tip_refine_mode = str(args.tip_refine_mode)
    cal.tip_detection_mode = str(args.tip_detection_mode)
    cal.red_tip_sat_min = int(args.red_tip_sat_min)
    cal.red_tip_val_min = int(args.red_tip_val_min)
    cal.red_tip_min_area_px = int(args.red_tip_min_area_px)
    cal.red_tip_morph_kernel = int(args.red_tip_morph_kernel)
    cal.red_tip_hue1_min = int(args.red_tip_hue1_min)
    cal.red_tip_hue1_max = int(args.red_tip_hue1_max)
    cal.red_tip_hue2_min = int(args.red_tip_hue2_min)
    cal.red_tip_hue2_max = int(args.red_tip_hue2_max)
    cal.red_tip_search_radius_px = float(args.red_tip_search_radius_px)
    cal.red_tip_local_min_area_px = int(args.red_tip_local_min_area_px)
    cal.red_tip_distance_weight = float(args.red_tip_distance_weight)
    cal.red_tip_min_circularity = float(args.red_tip_min_circularity)
    cal.red_tip_component_selection = str(args.red_tip_component_selection)
    cal.red_tip_use_rgb_excess = bool(args.red_tip_use_rgb_excess)
    cal.red_tip_rgb_excess_min = int(args.red_tip_rgb_excess_min)
    cal.red_tip_debug_save_mask = bool(args.red_tip_debug_save_mask)
    cal.export_analysis_outputs = bool(args.export_analysis_outputs)


def load_camera_and_board_calibration(cal: CTR_Shadow_Calibration, args: argparse.Namespace) -> None:
    camera_file = Path(str(args.camera_calibration_file)).expanduser()
    board_image = Path(str(args.board_reference_image)).expanduser()
    if camera_file.is_file():
        cal.load_camera_calibration(str(camera_file))
        if board_image.is_file():
            debug_path = Path(cal.calibration_data_folder) / "checkerboard_reference_debug.png"
            cal.estimate_board_reference_from_image(
                str(board_image),
                board_xz_axis_sign=float(args.board_xz_axis_sign),
                draw_debug=True,
                save_debug_path=str(debug_path),
            )
        else:
            print(f"[WARN] Board reference image not found: {board_image}")
    else:
        print(f"[WARN] Camera calibration file not found: {camera_file}; using pixel/ruler fallback for plots.")


def configure_camera_capture_settings(cal: CTR_Shadow_Calibration, args: argparse.Namespace) -> None:
    cam = getattr(cal, "cam", None)
    if cam is None:
        return
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, int(args.camera_width))
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, int(args.camera_height))
    if bool(args.enable_manual_focus):
        cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        cam.set(cv2.CAP_PROP_FOCUS, float(args.manual_focus_val))
    print(
        "[CAMERA] "
        f"width={int(args.camera_width)}, height={int(args.camera_height)}, "
        f"manual_focus={bool(args.enable_manual_focus)}, focus={float(args.manual_focus_val)}, "
        f"flush_frames={int(args.camera_flush_frames)}"
    )


# -----------------------------------------------------------------------------
# Motion acquisition
# -----------------------------------------------------------------------------


def get_commanded_axis_positions(cal: CTR_Shadow_Calibration) -> dict[str, float]:
    positions = getattr(cal, "_commanded_axis_positions", None)
    if not isinstance(positions, dict):
        positions = {}
        setattr(cal, "_commanded_axis_positions", positions)
    return positions


def get_estimated_motion_done_at(cal: CTR_Shadow_Calibration) -> float:
    done_at = getattr(cal, "_estimated_motion_done_at", None)
    if not isinstance(done_at, (int, float)):
        done_at = time.monotonic()
        setattr(cal, "_estimated_motion_done_at", float(done_at))
    return float(done_at)


def set_estimated_motion_done_at(cal: CTR_Shadow_Calibration, done_at: float) -> None:
    setattr(cal, "_estimated_motion_done_at", float(done_at))


def send_absolute_move(cal: CTR_Shadow_Calibration, feedrate: float, axes_targets: dict[str, float]) -> dict[str, Any]:
    if getattr(cal, "rrf", None) is None:
        raise RuntimeError("Robot is not connected.")
    positions = get_commanded_axis_positions(cal)
    previous = dict(positions)
    parts = ["G1"]
    for axis, value in axes_targets.items():
        parts.append(f"{axis}{float(value):.5f}")
        positions[str(axis)] = float(value)
    parts.append(f"F{format_feedrate(feedrate)}")
    gcode = " ".join(parts)
    print(f"Command: {gcode}")
    cal.rrf.send_code(gcode)
    return {"gcode": gcode, "previous_axes": previous, "axes_targets": dict(axes_targets), "feedrate": float(feedrate)}


def record_estimated_motion(cal: CTR_Shadow_Calibration, command_record: dict[str, Any]) -> float:
    previous = command_record.get("previous_axes", {})
    targets = command_record.get("axes_targets", {})
    deltas = []
    for axis, target in targets.items():
        if axis in previous:
            deltas.append(float(target) - float(previous[axis]))
    dist = float(np.linalg.norm(deltas)) if deltas else 0.0
    est_s = estimate_motion_time_s(dist, float(command_record.get("feedrate", 0.0)))
    done_at = max(time.monotonic(), get_estimated_motion_done_at(cal)) + est_s
    set_estimated_motion_done_at(cal, done_at)
    return est_s


def wait_for_estimated_motion_complete(cal: CTR_Shadow_Calibration, extra_settle_s: float, extra_buffer_s: float, reason: str) -> None:
    wait_s = max(0.0, get_estimated_motion_done_at(cal) - time.monotonic()) + max(0.0, extra_settle_s) + max(0.0, extra_buffer_s)
    if wait_s > 0.0:
        print(f"[WAIT] {reason}: {wait_s:.3f} s")
        time.sleep(wait_s)
    set_estimated_motion_done_at(cal, max(get_estimated_motion_done_at(cal), time.monotonic()))


def capture_frame(cal: CTR_Shadow_Calibration, output_path: Path, args: argparse.Namespace) -> None:
    cam = cal.cam
    if cam is None:
        raise RuntimeError("Camera is not connected.")
    for _ in range(max(0, int(args.camera_flush_frames))):
        cam.read()
    ret, frame = cam.read()
    if not ret:
        ret, frame = cam.read()
    if not ret or frame is None:
        raise RuntimeError(f"Could not capture frame for {output_path.name}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(output_path), frame)
    if not ok:
        raise RuntimeError(f"cv2.imwrite failed for {output_path}")
    print(f"Saved {output_path.name}")



def send_initial_split_preposition(
    cal: CTR_Shadow_Calibration,
    args: argparse.Namespace,
    axes: AxisInfo,
    first_row: pd.Series,
) -> dict[str, Any]:
    """
    Very first acquisition move:
      1) fast XYZC prepositioning at --initial_xyzc_feed, without changing B
      2) slow B-only prepositioning at --initial_b_feed
      3) M400, then --initial_daq_wait_s seconds of extra settle before DAQ starts
    """
    xyzc_targets = {
        axes.x_axis: float(first_row["x_cmd"]),
        axes.y_axis: float(first_row["y_cmd"]),
        axes.z_axis: float(first_row["z_cmd"]),
        axes.c_axis: float(first_row["c_cmd"]),
    }
    b_target = {axes.b_axis: float(first_row["b_cmd"])}

    print("\n" + "=" * 80)
    print(
        "Initial split preposition: "
        f"XYZC @ F{format_feedrate(float(args.initial_xyzc_feed))}, "
        f"B @ F{format_feedrate(float(args.initial_b_feed))}, "
        f"then wait {float(args.initial_daq_wait_s):.3f} s before DAQ"
    )
    print("=" * 80)

    cmd_xyzc = send_absolute_move(cal, float(args.initial_xyzc_feed), xyzc_targets)
    cmd_b = send_absolute_move(cal, float(args.initial_b_feed), b_target)

    # Force firmware-side completion before the explicit DAQ settle wait.
    try:
        print("Command: M400 ; wait for initial prepositioning moves to finish")
        cal.rrf.send_code("M400")
    except Exception as exc:
        print(f"[WARN] Could not send M400 after initial preposition: {exc}")

    if float(args.initial_daq_wait_s) > 0.0:
        print(f"[WAIT] initial DAQ settle: {float(args.initial_daq_wait_s):.3f} s")
        time.sleep(float(args.initial_daq_wait_s))

    set_estimated_motion_done_at(cal, time.monotonic())
    return {
        "initial_xyzc_gcode": cmd_xyzc["gcode"],
        "initial_b_gcode": cmd_b["gcode"],
        "initial_wait_s": float(args.initial_daq_wait_s),
    }


def run_acquisition(cal: CTR_Shadow_Calibration, args: argparse.Namespace, axes: AxisInfo, plan_df: pd.DataFrame) -> pd.DataFrame:
    raw_dir = Path(cal.calibration_data_folder) / "raw_image_data_folder"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not bool(args.append_raw_data):
        cal.clear_raw_image_data_folder()

    acquired_rows: list[dict[str, Any]] = []
    sorted_plan = plan_df.sort_values("global_order").reset_index(drop=True)

    initial_info: dict[str, Any] | None = None
    if not sorted_plan.empty:
        first_row = sorted_plan.iloc[0]
        initial_info = send_initial_split_preposition(cal, args, axes, first_row)
        first_out = first_row.to_dict()
        first_out.update(initial_info)
        first_out["command_gcode"] = (
            str(initial_info["initial_xyzc_gcode"]) + " ; " + str(initial_info["initial_b_gcode"])
        )
        first_out["estimated_motion_s"] = float("nan")
        first_out["initial_split_preposition"] = True
        acquired_rows.append(first_out)

    for _, row in sorted_plan.iloc[1:].iterrows():
        targets = {
            axes.x_axis: float(row["x_cmd"]),
            axes.y_axis: float(row["y_cmd"]),
            axes.z_axis: float(row["z_cmd"]),
            axes.b_axis: float(row["b_cmd"]),
            axes.c_axis: float(row["c_cmd"]),
        }
        print(
            "\n" + "=" * 80 + "\n"
            f"{row['pass_name']} | repeat {int(row['repeat_number'])} | {row['segment_name']} | "
            f"theta={float(row['theta_deg']):.2f} attack={float(row['attack_angle_deg']):.2f} "
            f"capture={bool(row['capture_image'])}\n"
            + "=" * 80
        )
        cmd = send_absolute_move(cal, float(row["feedrate"]), targets)
        est = record_estimated_motion(cal, cmd)
        settle = float(args.settle_after_start_move_s) if int(row["point_order"]) == 0 else float(args.capture_dwell_s)
        wait_for_estimated_motion_complete(cal, settle, float(args.move_buffer_s), str(row["segment_name"]))

        out = row.to_dict()
        out["command_gcode"] = cmd["gcode"]
        out["estimated_motion_s"] = float(est)
        if bool(row["capture_image"]):
            capture_frame(cal, raw_dir / str(row["image_file"]), args)
            if float(args.post_capture_buffer_s) > 0.0:
                time.sleep(float(args.post_capture_buffer_s))
        acquired_rows.append(out)

    out_df = pd.DataFrame(acquired_rows)
    out_df.to_csv(Path(cal.calibration_data_folder) / "circle_capture_plan_executed.csv", index=False)
    return out_df


# -----------------------------------------------------------------------------
# Analysis and plotting
# -----------------------------------------------------------------------------


def px_to_rz_mm(cal: CTR_Shadow_Calibration, x_px: float, y_px: float, width_in_pixels: float, width_in_mm: float) -> tuple[float, float, str]:
    has_calibrated_axes = (
        getattr(cal, "board_homography_mm_from_px", None) is not None
        or (
            getattr(cal, "true_vertical_img_unit", None) is not None
            and getattr(cal, "board_mm_per_px_local", None) is not None
        )
    )
    if has_calibrated_axes and hasattr(cal, "pixel_point_to_calibrated_axes"):
        origin_px = None
        if getattr(cal, "board_pose", None) is not None and "origin_px" in cal.board_pose:
            origin_px = np.asarray(cal.board_pose["origin_px"], dtype=float).reshape(2)
        r_mm, z_mm = cal.pixel_point_to_calibrated_axes(x_px=float(x_px), y_px=float(y_px), origin_px=origin_px)
        return float(r_mm), float(z_mm), "board_reference_calibrated"
    if getattr(cal, "ruler_mm_per_px", None) is not None:
        scale = float(cal.ruler_mm_per_px)
        return float(x_px) * scale, -float(y_px) * scale, "ruler_reference_scale"
    scale = float(width_in_mm) / float(width_in_pixels)
    return float(x_px) * scale, -float(y_px) * scale, "legacy_linear_scale"


def build_results_tables(cal: CTR_Shadow_Calibration, args: argparse.Namespace, plan_df: pd.DataFrame) -> pd.DataFrame:
    processed_dir = Path(cal.calibration_data_folder) / "processed_image_data_folder"
    selected_csv = processed_dir / "tip_locations_selected.csv"
    if not selected_csv.exists():
        raise FileNotFoundError(f"Missing analyzed tip CSV: {selected_csv}")
    tip_df = pd.read_csv(selected_csv)
    cap_plan = plan_df[plan_df["capture_image"].astype(bool)].copy()
    df = tip_df.merge(cap_plan, on="image_file", how="inner", suffixes=("_analysis", ""))
    if df.empty:
        raise RuntimeError("No analyzed images matched circle capture plan.")

    r_vals, z_vals, modes = [], [], []
    for _, row in df.iterrows():
        r, z, mode = px_to_rz_mm(cal, float(row["tip_column"]), float(row["tip_row"]), float(args.width_in_pixels), float(args.width_in_mm))
        r_vals.append(r)
        z_vals.append(z)
        modes.append(mode)
    df["r_measured_mm"] = r_vals
    df["z_measured_mm"] = z_vals
    df["coordinate_mode"] = modes

    # Relative coordinates per pass/repeat.
    df["r_measured_rel_mm"] = np.nan
    df["z_measured_rel_mm"] = np.nan
    df["tip_x_desired_rel"] = np.nan
    df["tip_z_desired_rel"] = np.nan
    for (repeat_idx, pass_name), g in df.groupby(["repeat_index", "pass_name"]):
        start = g.sort_values("point_order").iloc[0]
        idx = g.index
        df.loc[idx, "r_measured_rel_mm"] = df.loc[idx, "r_measured_mm"].astype(float) - float(start["r_measured_mm"])
        df.loc[idx, "z_measured_rel_mm"] = df.loc[idx, "z_measured_mm"].astype(float) - float(start["z_measured_mm"])
        df.loc[idx, "tip_x_desired_rel"] = df.loc[idx, "tip_x_desired"].astype(float) - float(start["tip_x_desired"])
        df.loc[idx, "tip_z_desired_rel"] = df.loc[idx, "tip_z_desired"].astype(float) - float(start["tip_z_desired"])

    out_csv = processed_dir / "circle_capture_points_raw.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved analyzed circle points: {out_csv}")
    return df


def _style_axes_for_dark_theme(ax: plt.Axes) -> None:
    ax.set_facecolor("none")
    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.title.set_color("white")
    for spine in ax.spines.values():
        spine.set_color((1.0, 1.0, 1.0, 0.45))


def _style_legend_for_dark_theme(ax: plt.Axes) -> None:
    legend = ax.get_legend()
    if legend is None:
        return
    legend.get_frame().set_facecolor((0, 0, 0, 0))
    legend.get_frame().set_edgecolor((1, 1, 1, 0.25))
    for text in legend.get_texts():
        text.set_color("white")


def _save_transparent_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.patch.set_alpha(0.0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)


def make_plots(results_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 8))
    # Desired circle, plotted in tip-space relative X/Z.
    desired = results_df.sort_values("point_order")
    if not desired.empty:
        ref = desired[desired["pass_name"] == desired["pass_name"].iloc[0]].copy()
        ax.plot(ref["tip_x_desired_rel"], ref["tip_z_desired_rel"], linewidth=2.0, label="desired tip circle")

    marker_map = {"compensated": "o", "noncompensated": "s"}
    for (pass_name, repeat_number), g in results_df.groupby(["pass_name", "repeat_number"]):
        g = g.sort_values("point_order")
        ax.plot(
            g["r_measured_rel_mm"],
            g["z_measured_rel_mm"],
            marker=marker_map.get(str(pass_name), "o"),
            linewidth=1.4,
            markersize=3,
            label=f"{pass_name} R{int(repeat_number)}",
        )
    ax.set_title("Circle capture trajectory")
    ax.set_xlabel("Relative measured R / X-plane coordinate (mm)")
    ax.set_ylabel("Relative measured Z coordinate (mm)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25, color="white")
    ax.legend(fontsize=8)
    _style_axes_for_dark_theme(ax)
    _style_legend_for_dark_theme(ax)
    fig.tight_layout()
    path = output_dir / "circle_capture_trajectory_comparison.png"
    _save_transparent_figure(fig, path)
    plt.close(fig)
    print(f"Saved plot: {path}")

    fig, ax = plt.subplots(figsize=(9, 6))
    for pass_name, g in results_df.groupby("pass_name"):
        g = g.sort_values("theta_deg")
        # Rough same-frame errors relative to desired circle. Sign conventions may differ,
        # so treat this as diagnostic, not calibrated truth unless the board axes match tip-space X/Z.
        err = np.sqrt(
            (g["r_measured_rel_mm"].astype(float) - g["tip_x_desired_rel"].astype(float)) ** 2
            + (g["z_measured_rel_mm"].astype(float) - g["tip_z_desired_rel"].astype(float)) ** 2
        )
        ax.plot(g["theta_deg"], err, marker="o", linewidth=1.3, markersize=3, label=str(pass_name))
    ax.set_title("Diagnostic distance to desired circle samples")
    ax.set_xlabel("Circle theta (deg)")
    ax.set_ylabel("Distance error in plotted coordinates (mm)")
    ax.grid(True, alpha=0.25, color="white")
    ax.legend(fontsize=8)
    _style_axes_for_dark_theme(ax)
    _style_legend_for_dark_theme(ax)
    fig.tight_layout()
    path = output_dir / "circle_capture_distance_error_vs_theta.png"
    _save_transparent_figure(fig, path)
    plt.close(fig)
    print(f"Saved plot: {path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    args = build_arg_parser().parse_args()

    cal_data = load_calibration_export(Path(args.calibration_file))
    axes = get_axis_info(cal_data)

    cal = CTR_Shadow_Calibration(
        parent_directory=str(SCRIPT_DIR),
        project_name=str(args.project_name),
        allow_existing=True,
        add_date=False,
    )
    attach_calibration_state_for_class(cal, cal_data)
    configure_tip_detection(cal, args)
    load_camera_and_board_calibration(cal, args)
    project_dir = Path(cal.calibration_data_folder)
    processed_dir = project_dir / "processed_image_data_folder"
    processed_dir.mkdir(parents=True, exist_ok=True)

    if bool(args.process_only):
        if str(args.process_plan_csv).strip():
            capture_csv = Path(args.process_plan_csv).expanduser()
        else:
            capture_csv = processed_dir / "circle_capture_plan.csv"
            if not capture_csv.exists():
                capture_csv = project_dir / "circle_capture_plan.csv"
        if not capture_csv.exists():
            raise FileNotFoundError(
                "Could not find a capture-plan CSV for process-only mode. "
                f"Looked for: {capture_csv}"
            )
        plan_df = pd.read_csv(capture_csv)
        if "image_file" not in plan_df.columns:
            raise ValueError(f"Capture-plan CSV is missing required 'image_file' column: {capture_csv}")

        cal.setup_analysis_crop(enable_manual_adjustment=bool(args.manual_crop_adjustment))
        cal.analyze_data_batch(
            threshold=int(args.threshold),
            export_analysis_outputs=bool(args.export_analysis_outputs),
            capture_metadata_source=plan_df,
        )
        results_df = build_results_tables(cal, args, plan_df)
        make_plots(results_df, processed_dir)

        print("\nDone.")
        print(f"Project folder: {project_dir}")
        print(f"Processed outputs: {processed_dir}")
        return

    pull, release, selected_curve_key = get_curl_specific_models(
        cal_data,
        curve_set=str(args.curve_set),
        allow_global_fallback=bool(args.allow_global_fallback),
    )
    print(f"Using branch model source: {selected_curve_key}")
    print(f"Axis mapping: {axes}")

    plan_df = build_full_plan(cal, args, axes, pull, release)

    dense_csv = project_dir / "circle_dense_motion_plan.csv"
    capture_csv = project_dir / "circle_capture_plan.csv"
    plan_df.to_csv(dense_csv, index=False)
    plan_df[plan_df["capture_image"].astype(bool)].to_csv(capture_csv, index=False)
    print(f"Saved dense motion plan: {dense_csv}")
    print(f"Saved capture plan: {capture_csv}")

    gcode_path, points_csv, metadata_path = write_print_gcode(plan_df, args, axes, project_dir / "gcode_generation")
    print(f"Saved print G-code: {gcode_path}")
    print(f"Saved print points CSV: {points_csv}")
    print(f"Saved print metadata: {metadata_path}")

    if bool(args.export_only):
        print("Export-only mode: not connecting to camera or robot.")
        return

    cal.connect_to_camera(cam_port=int(args.cam_port), show_preview=bool(args.show_preview))
    configure_camera_capture_settings(cal, args)
    cal.setup_analysis_crop(enable_manual_adjustment=bool(args.manual_crop_adjustment))
    cal.connect_to_robot(duet_web_address=str(args.duet_web_address))

    try:
        # Initialize estimated-position bookkeeping from first row.
        first = plan_df.sort_values("global_order").iloc[0]
        get_commanded_axis_positions(cal).update(
            {
                axes.x_axis: float(first["x_cmd"]),
                axes.y_axis: float(first["y_cmd"]),
                axes.z_axis: float(first["z_cmd"]),
                axes.b_axis: float(first["b_cmd"]),
                axes.c_axis: float(first["c_cmd"]),
            }
        )

        executed_df = run_acquisition(cal, args, axes, plan_df)
        executed_csv = project_dir / "circle_capture_plan_executed.csv"
        executed_df.to_csv(executed_csv, index=False)
        executed_capture_df = executed_df[executed_df["capture_image"].astype(bool)].copy()
        executed_capture_csv = processed_dir / "circle_capture_plan.csv"
        executed_capture_df.to_csv(executed_capture_csv, index=False)

        cal.analyze_data_batch(
            threshold=int(args.threshold),
            export_analysis_outputs=bool(args.export_analysis_outputs),
            capture_metadata_source=executed_capture_df,
        )
        results_df = build_results_tables(cal, args, executed_df)
        make_plots(results_df, processed_dir)

        print("\nDone.")
        print(f"Project folder: {project_dir}")
        print(f"Print G-code: {gcode_path}")
        print(f"Processed outputs: {processed_dir}")

    finally:
        if bool(args.return_to_b0_on_exit) and getattr(cal, "rrf", None) is not None:
            try:
                print("[INFO] Returning B to 0 before exit.")
                cmd = send_absolute_move(cal, float(args.approach_feed), {axes.b_axis: 0.0})
                est = record_estimated_motion(cal, cmd)
                wait_for_estimated_motion_complete(cal, 0.5, float(args.move_buffer_s), f"return B to 0 ({est:.3f} s est)")
            except Exception as exc:
                print(f"[WARN] Failed to return B to 0: {exc}")
        try:
            if getattr(cal, "cam", None) is not None:
                cal.disconnect_camera()
        except Exception:
            pass


if __name__ == "__main__":
    main()
