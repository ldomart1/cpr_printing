#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CTR red-dot hysteresis test runner.

Runs three independent curl/uncurl tests at a feedrate sweep:
    0-90-0
    90-180-0
    0-180-0

For each test and feedrate, the script:
    1. Loads a previously exported calibration file.
    2. Uses the PCHIP *pull* tip-angle model to invert target angles -> B axis positions.
    3. Captures the red-dot tip location at start, apogee, and end.
    4. Uses shadow_calibration.CTR_Shadow_Calibration to analyze images.
    5. Exports CSV summaries and plots:
        - apogee/end position relative to start versus feedrate
        - apogee/end angle relative to start versus feedrate
        - RZ scatter plots
        - one combined panel with all plots

Place this file next to shadow_calibration.py and run, for example:

python run_hysteresis_tests.py \
  --project_name Hysteresis_2026_05_29 \
  --calibration_file Test_Calibration_2026-05-29_00/processed_image_data_folder/calibrated_robot_gcode_calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import pickle
import re
import sys
import time
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


DEFAULT_FEEDRATES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
#DEFAULT_FEEDRATES = [10, 20, 25, 50, 100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
DEFAULT_TESTS = [
    ("0-90-0", [0.0, 90.0, 0.0]),
    ("90-180-90", [90.0, 180.0, 90.0]),
    ("0-180-0", [0.0, 180.0, 0.0]),
    ("180-90-0", [180.0, 90.0, 0.0]),
    ("90-0-90", [90.0, 0.0, 90.0]),
]

DEFAULT_CAMERA_CALIBRATION_FILE = SCRIPT_DIR / "captures" / "calibration_webcam_20260406_104136.npz"
DEFAULT_BOARD_REFERENCE_IMAGE = SCRIPT_DIR / "captures" / "photo_20260526_200532.png"


# ----------------------------- CLI -----------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CTR red-dot hysteresis tests and export speed-dependent plots."
    )

    parser.add_argument("--project_name", type=str, default="Hysteresis_Tests")
    parser.add_argument(
        "--calibration_file",
        type=str,
        required=True,
        help=(
            "Path to a calibration export made by postprocess_calibration_data(), "
            "usually calibrated_robot_gcode_calibration.json or calibrated_robot_cubic_calibration.pkl."
        ),
    )

    parser.add_argument("--cam_port", type=int, default=0)
    parser.add_argument("--duet_web_address", type=str, default=r"http://192.168.2.21")
    parser.add_argument("--manual_crop_adjustment", action="store_true", default=True)
    parser.add_argument("--no_manual_crop_adjustment", dest="manual_crop_adjustment", action="store_false")
    parser.add_argument("--threshold", type=int, default=220)
    parser.add_argument("--export_analysis_outputs", action="store_true", default=False)

    parser.add_argument("--camera_calibration_file", type=str, default=str(DEFAULT_CAMERA_CALIBRATION_FILE))
    parser.add_argument("--board_reference_image", type=str, default=str(DEFAULT_BOARD_REFERENCE_IMAGE))
    parser.add_argument("--board_xz_axis_sign", type=int, choices=[-1, 1], default=1)
    parser.add_argument("--width_in_pixels", type=float, default=3025.0)
    parser.add_argument("--width_in_mm", type=float, default=140.0)

    parser.add_argument("--probe_x", type=float, default=95.0)
    parser.add_argument("--probe_y", type=float, default=55.0)
    parser.add_argument("--probe_z", type=float, default=-170.0)
    parser.add_argument("--initial_positioning_feedrate", type=float, default=300.0)
    parser.add_argument("--preposition_b_feedrate", type=float, default=200.0)
    parser.add_argument("--capture_dwell_s", type=float, default=0.5)
    parser.add_argument("--settle_after_start_move_s", type=float, default=1.0)
    parser.add_argument("--move_buffer_s", type=float, default=0.2)
    parser.add_argument("--post_capture_buffer_s", type=float, default=0.15)

    parser.add_argument(
        "--feedrates",
        type=str,
        default=",".join(str(v) for v in DEFAULT_FEEDRATES),
        help="Comma-separated B-axis feedrates to test, in machine feedrate units/min.",
    )
    parser.add_argument(
        "--orientation_ids",
        type=str,
        default="0",
        help=(
            "Comma-separated camera/orientation IDs to capture. Default is 0. "
            "Plots use --analysis_orientation only, so keep this at 0 unless you need extra raw images."
        ),
    )
    parser.add_argument("--analysis_orientation", type=int, default=0)
    parser.add_argument("--robot_front_axis_name", type=str, default="X")
    parser.add_argument("--robot_stage_y_axis_name", type=str, default="Y")
    parser.add_argument("--robot_stage_z_axis_name", type=str, default="Z")
    parser.add_argument("--robot_rear_axis_name", type=str, default="B")
    parser.add_argument("--robot_rotation_axis_name", type=str, default="C")
    parser.add_argument("--robot_rotation_axis_180_deg", type=float, default=180.0)
    parser.add_argument("--rotation_feedrate", type=float, default=8000.0)
    parser.add_argument("--rotation_settle_s", type=float, default=2.0)

    parser.add_argument(
        "--enable_stage_y_compensation",
        action="store_true",
        default=False,
        help="For C+90/C-90 captures, apply stage-Y compensation from the pull radial PCHIP model.",
    )

    parser.add_argument("--tip_detection_mode", type=str, default="red_dot", choices=["classical", "red_dot", "auto_red_dot"])
    parser.add_argument("--tip_refine_mode", type=str, default="coarse", choices=["coarse", "parallel_centerline", "auto"])
    parser.add_argument("--red_tip_sat_min", type=int, default=80)
    parser.add_argument("--red_tip_val_min", type=int, default=80)
    parser.add_argument("--red_tip_min_area_px", type=int, default=20)
    parser.add_argument("--red_tip_morph_kernel", type=int, default=1)
    parser.add_argument("--red_tip_hue1_min", type=int, default=0)
    parser.add_argument("--red_tip_hue1_max", type=int, default=10)
    parser.add_argument("--red_tip_hue2_min", type=int, default=150)
    parser.add_argument("--red_tip_hue2_max", type=int, default=179)
    parser.add_argument("--red_tip_search_radius_px", type=float, default=180.0)
    parser.add_argument("--red_tip_local_min_area_px", type=int, default=10)
    parser.add_argument("--red_tip_distance_weight", type=float, default=3.0)
    parser.add_argument("--red_tip_min_circularity", type=float, default=0.0)
    parser.add_argument(
        "--red_tip_component_selection",
        type=str,
        default="nearest_largest",
        choices=["largest", "nearest", "nearest_largest"],
    )
    parser.add_argument("--red_tip_use_rgb_excess", dest="red_tip_use_rgb_excess", action="store_true", default=True)
    parser.add_argument("--no_red_tip_use_rgb_excess", dest="red_tip_use_rgb_excess", action="store_false")
    parser.add_argument("--red_tip_rgb_excess_min", type=int, default=35)
    parser.add_argument("--red_tip_debug_save_mask", action="store_true", default=True)
    parser.add_argument("--no_red_tip_debug_save_mask", dest="red_tip_debug_save_mask", action="store_false")

    parser.add_argument("--append_raw_data", action="store_true", default=False)
    parser.add_argument("--return_to_b0_on_exit", action="store_true", default=True)
    parser.add_argument("--no_return_to_b0_on_exit", dest="return_to_b0_on_exit", action="store_false")

    return parser


# ----------------------------- Utilities -----------------------------


def parse_csv_floats(text: str) -> list[float]:
    vals = []
    for token in str(text).split(","):
        token = token.strip()
        if token:
            vals.append(float(token))
    if not vals:
        raise ValueError("Expected at least one numeric value.")
    return vals


def parse_csv_ints(text: str) -> list[int]:
    return [int(round(v)) for v in parse_csv_floats(text)]


def safe_token(value: Any) -> str:
    text = str(value).strip()
    text = text.replace(".", "p")
    text = re.sub(r"[^A-Za-z0-9+\-]+", "_", text)
    text = text.strip("_")
    return text or "x"


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


def wrap_angle_delta_deg(delta_deg: Any) -> Any:
    arr = np.asarray(delta_deg, dtype=float)
    wrapped = (arr + 180.0) % 360.0 - 180.0
    if np.ndim(wrapped) == 0:
        return float(wrapped)
    return wrapped


def circular_mean_deg(values: Any) -> float:
    arr = np.asarray(values, dtype=float).reshape(-1)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    rad = np.deg2rad(arr)
    return float(np.rad2deg(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))))


def format_feedrate(feedrate: float) -> str:
    return str(int(round(float(feedrate))))


def estimate_motion_time_s(distance_mm: float, feedrate_mm_per_min: float) -> float:
    if not np.isfinite(distance_mm) or distance_mm <= 0.0:
        return 0.0
    if not np.isfinite(feedrate_mm_per_min) or feedrate_mm_per_min <= 0.0:
        return 0.0
    return 60.0 * float(distance_mm) / float(feedrate_mm_per_min)


def get_commanded_axis_positions(cal: CTR_Shadow_Calibration) -> dict[str, float]:
    positions = getattr(cal, "_commanded_axis_positions", None)
    if not isinstance(positions, dict):
        positions = {}
        setattr(cal, "_commanded_axis_positions", positions)
    return positions


def get_command_log(cal: CTR_Shadow_Calibration) -> list[dict[str, Any]]:
    command_log = getattr(cal, "_command_log", None)
    if not isinstance(command_log, list):
        command_log = []
        setattr(cal, "_command_log", command_log)
    return command_log


def get_estimated_motion_done_at(cal: CTR_Shadow_Calibration) -> float:
    done_at = getattr(cal, "_estimated_motion_done_at", None)
    if not isinstance(done_at, (int, float)):
        done_at = time.monotonic()
        setattr(cal, "_estimated_motion_done_at", float(done_at))
    return float(done_at)


def set_estimated_motion_done_at(cal: CTR_Shadow_Calibration, done_at: float) -> None:
    setattr(cal, "_estimated_motion_done_at", float(done_at))


# ----------------------------- Calibration model loading -----------------------------


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


def get_phase_fit_bundle(cal_data: dict[str, Any], phase: str) -> dict[str, Any]:
    phase_key = str(phase).strip().lower()
    by_phase = cal_data.get("fit_models_by_phase")
    if not isinstance(by_phase, dict):
        raise KeyError("Calibration export does not contain fit_models_by_phase.")

    for candidate in (phase_key, f"{phase_key}_1", f"{phase_key}_combined"):
        bundle = by_phase.get(candidate)
        if isinstance(bundle, dict):
            return bundle

    available = sorted(str(k) for k in by_phase.keys())
    raise KeyError(f"No fit model bundle found for phase '{phase_key}'. Available: {available}")


def get_model_descriptor(bundle: dict[str, Any], quantity: str, fit_family: str = "pchip") -> dict[str, Any] | None:
    quantity = str(quantity).strip().lower()
    fit_family = str(fit_family).strip().lower()
    candidates = []

    if quantity == "tip_angle":
        candidates = [f"tip_angle_{fit_family}", "tip_angle_pchip", "tip_angle", "tip_angle_avg_pchip"]
    elif quantity == "r":
        candidates = [f"r_{fit_family}", "r_pchip", "r", "r_avg_pchip"]
    elif quantity == "z":
        candidates = [f"z_{fit_family}", "z_pchip", "z", "z_avg_pchip"]
    elif quantity == "offplane_y":
        candidates = [f"offplane_y_{fit_family}", "offplane_y_pchip", "offplane_y", "offplane_y_avg_pchip"]
    else:
        candidates = [f"{quantity}_{fit_family}", quantity]

    for key in candidates:
        descriptor = bundle.get(key)
        if isinstance(descriptor, dict):
            return descriptor
    return None


def attach_calibration_state_for_class(cal: CTR_Shadow_Calibration, cal_data: dict[str, Any]) -> None:
    """
    This is not required for plotting, but it makes class helper methods and debugging
    work as if postprocess_calibration_data() had just run.
    """
    if isinstance(cal_data.get("fit_models_by_phase"), dict):
        cal._postprocessed_fit_models_by_phase = cal_data["fit_models_by_phase"]
    if isinstance(cal_data.get("datasets_by_phase"), dict):
        cal._postprocessed_datasets = cal_data["datasets_by_phase"]
    if isinstance(cal_data.get("redundancy_diagnostics"), dict):
        cal._postprocessed_redundancy_diagnostics = cal_data["redundancy_diagnostics"]


def solve_b_for_angle_from_pull_pchip(
    cal: CTR_Shadow_Calibration,
    pull_tip_angle_model: dict[str, Any],
    target_angle_deg: float,
    reference_b: float,
) -> float:
    """Invert the pull-phase PCHIP tip-angle model to a B-axis position."""
    return float(
        cal._solve_curve_model_input_for_target(
            pull_tip_angle_model,
            float(target_angle_deg),
            reference_b=float(reference_b),
        )
    )


def eval_model_scalar(cal: CTR_Shadow_Calibration, model: dict[str, Any] | None, b_value: float) -> float:
    if model is None:
        return float("nan")
    pred = cal._evaluate_curve_model(model, np.asarray([float(b_value)], dtype=float))
    arr = np.asarray(pred, dtype=float).reshape(-1)
    if arr.size == 0:
        return float("nan")
    return float(arr[0])


# ----------------------------- CTR setup -----------------------------


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


def send_absolute_move(
    cal: CTR_Shadow_Calibration,
    feedrate: float,
    **axes: float | None,
) -> dict[str, Any]:
    if getattr(cal, "rrf", None) is None:
        raise RuntimeError("Robot is not connected.")
    axis_positions = get_commanded_axis_positions(cal)
    command_log = get_command_log(cal)
    previous_axes = dict(axis_positions)
    pieces = ["G1"]
    for axis_name, value in axes.items():
        if value is None:
            continue
        value_f = float(value)
        pieces.append(f"{axis_name}{value_f:.5f}")
    pieces.append(f"F{format_feedrate(feedrate)}")
    gcode = " ".join(pieces)
    print(f"Command: {gcode}")
    for axis_name, value in axes.items():
        if value is not None:
            axis_positions[axis_name] = float(value)
    command_record = {
        "command_index": int(len(command_log) + 1),
        "feedrate": float(feedrate),
        "gcode": gcode,
        "previous_axes": previous_axes,
        "axes_targets": {str(axis_name): float(value) for axis_name, value in axes.items() if value is not None},
        "resolved_axes": dict(axis_positions),
    }
    command_log.append(command_record)
    cal.rrf.send_code(gcode)
    return command_record


def record_estimated_motion(cal: CTR_Shadow_Calibration, command_record: dict[str, Any]) -> float:
    previous_axes = command_record.get("previous_axes", {})
    axes_targets = command_record.get("axes_targets", {})
    feedrate = float(command_record.get("feedrate", 0.0))
    deltas = []
    for axis_name, target_value in axes_targets.items():
        if axis_name not in previous_axes:
            continue
        deltas.append(float(target_value) - float(previous_axes[axis_name]))
    travel_distance_mm = float(np.linalg.norm(deltas)) if deltas else 0.0
    estimated_motion_s = estimate_motion_time_s(travel_distance_mm, feedrate)
    now = time.monotonic()
    done_at = max(now, get_estimated_motion_done_at(cal)) + estimated_motion_s
    set_estimated_motion_done_at(cal, done_at)
    return estimated_motion_s


def wait_for_estimated_motion_complete(
    cal: CTR_Shadow_Calibration,
    extra_settle_s: float = 0.0,
    extra_buffer_s: float = 0.0,
    reason: str = "motion",
) -> None:
    wait_s = (
        max(0.0, get_estimated_motion_done_at(cal) - time.monotonic())
        + max(0.0, float(extra_settle_s))
        + max(0.0, float(extra_buffer_s))
    )
    if wait_s > 0.0:
        print(f"[WAIT] {reason}: estimated {wait_s:.3f} s including settle buffer")
        time.sleep(wait_s)
    set_estimated_motion_done_at(cal, max(get_estimated_motion_done_at(cal), time.monotonic()))


def rotate_to_orientation(
    cal: CTR_Shadow_Calibration,
    current_orientation: int,
    target_orientation: int,
    args: argparse.Namespace,
) -> int:
    orientation_to_c_deg = {0: 0.0, 1: 180.0, 2: 90.0, 3: -90.0}
    if target_orientation not in orientation_to_c_deg:
        raise ValueError("orientation_ids must be in {0,1,2,3}.")
    cur_deg = orientation_to_c_deg[int(current_orientation)]
    tgt_deg = orientation_to_c_deg[int(target_orientation)]
    delta_deg = tgt_deg - cur_deg
    if abs(delta_deg) < 1e-12:
        return int(target_orientation)
    c_units = float(delta_deg) / 180.0 * float(args.robot_rotation_axis_180_deg)
    gcode = f"G91 G1 {args.robot_rotation_axis_name}{c_units:.5f} F{format_feedrate(args.rotation_feedrate)}"
    print(f"Command: {gcode}")
    cal.rrf.send_code(gcode)
    set_estimated_motion_done_at(
        cal,
        max(time.monotonic(), get_estimated_motion_done_at(cal))
        + estimate_motion_time_s(abs(c_units), float(args.rotation_feedrate)),
    )
    wait_for_estimated_motion_complete(
        cal,
        extra_settle_s=float(args.rotation_settle_s),
        extra_buffer_s=float(args.move_buffer_s),
        reason=f"rotation ({abs(c_units):.3f} axis units @ F{format_feedrate(args.rotation_feedrate)})",
    )
    cal.rrf.send_code("G90")
    return int(target_orientation)


def capture_frame(cal: CTR_Shadow_Calibration, output_path: Path) -> None:
    cam = cal.cam
    if cam is None:
        raise RuntimeError("Camera is not connected.")
    # Throw away one frame to reduce stale-frame risk.
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


def stage_y_compensation_sign(orientation_id: int) -> float:
    if int(orientation_id) == 2:
        return -1.0
    if int(orientation_id) == 3:
        return +1.0
    return 0.0


# ----------------------------- Acquisition -----------------------------


def build_b_plan(
    cal: CTR_Shadow_Calibration,
    pull_tip_angle_model: dict[str, Any],
    tests: list[tuple[str, list[float]]],
) -> dict[str, dict[str, Any]]:
    b_plan: dict[str, dict[str, Any]] = {}
    for test_name, angles in tests:
        if len(angles) != 3:
            raise ValueError(f"Test {test_name} must have exactly three angles.")
        start_angle, apogee_angle, end_angle = [float(v) for v in angles]

        # All targets are inverted using the pull PCHIP model, as requested.
        start_b = solve_b_for_angle_from_pull_pchip(cal, pull_tip_angle_model, start_angle, reference_b=0.0)
        apogee_b = solve_b_for_angle_from_pull_pchip(cal, pull_tip_angle_model, apogee_angle, reference_b=start_b)
        end_b = solve_b_for_angle_from_pull_pchip(cal, pull_tip_angle_model, end_angle, reference_b=0.0)

        b_plan[test_name] = {
            "angle_sequence_deg": [start_angle, apogee_angle, end_angle],
            "b_sequence": {
                "start": float(start_b),
                "apogee": float(apogee_b),
                "end": float(end_b),
            },
        }
        print(
            f"[B PLAN] {test_name}: "
            f"{start_angle:.1f}° -> B {start_b:.4f}, "
            f"{apogee_angle:.1f}° -> B {apogee_b:.4f}, "
            f"{end_angle:.1f}° -> B {end_b:.4f}"
        )
    return b_plan


def run_hysteresis_acquisition(
    cal: CTR_Shadow_Calibration,
    args: argparse.Namespace,
    b_plan: dict[str, dict[str, Any]],
    pull_r_model: dict[str, Any] | None,
) -> pd.DataFrame:
    feedrates = [float(v) for v in parse_csv_floats(args.feedrates)]
    orientation_ids = parse_csv_ints(args.orientation_ids)
    probe_x = float(args.probe_x)
    probe_y = float(args.probe_y)
    probe_z = float(args.probe_z)

    raw_dir = Path(cal.calibration_data_folder) / "raw_image_data_folder"
    raw_dir.mkdir(parents=True, exist_ok=True)
    if not bool(args.append_raw_data):
        cal.clear_raw_image_data_folder()

    plan_records: list[dict[str, Any]] = []
    current_orientation = 0

    for test_name, plan in b_plan.items():
        angles = plan["angle_sequence_deg"]
        b_start = float(plan["b_sequence"]["start"])
        b_apogee = float(plan["b_sequence"]["apogee"])
        b_end = float(plan["b_sequence"]["end"])
        test_token = safe_token(test_name)

        for feedrate in feedrates:
            feed_token = safe_token(int(feedrate) if float(feedrate).is_integer() else feedrate)
            capture_group = f"hyst_{test_token}_F{feed_token}"

            print("\n" + "=" * 78)
            print(f"TEST {test_name} | feedrate F{feedrate:g}")
            print("=" * 78)

            for orientation_id in orientation_ids:
                current_orientation = rotate_to_orientation(cal, current_orientation, int(orientation_id), args)

                point_plan = [
                    {
                        "point_name": "start",
                        "point_order": 0,
                        "motion_phase": "pull",
                        "step_idx": 0,
                        "target_angle_deg": angles[0],
                        "b_value": b_start,
                        "move_feedrate": float(feedrate),
                        "settle_s": float(args.settle_after_start_move_s),
                    },
                    {
                        "point_name": "apogee",
                        "point_order": 1,
                        "motion_phase": "pull",
                        "step_idx": 1,
                        "target_angle_deg": angles[1],
                        "b_value": b_apogee,
                        "move_feedrate": float(feedrate),
                        "settle_s": float(args.capture_dwell_s),
                    },
                    {
                        "point_name": "end",
                        "point_order": 2,
                        "motion_phase": "release",
                        "step_idx": 1,
                        "target_angle_deg": angles[2],
                        "b_value": b_end,
                        "move_feedrate": float(feedrate),
                        "settle_s": float(args.capture_dwell_s),
                    },
                ]

                for point in point_plan:
                    b_value = float(point["b_value"])
                    y_offset = 0.0
                    if bool(args.enable_stage_y_compensation):
                        sign = stage_y_compensation_sign(int(orientation_id))
                        if abs(sign) > 0.0 and pull_r_model is not None:
                            y_offset = sign * eval_model_scalar(cal, pull_r_model, b_value)
                            if not np.isfinite(y_offset):
                                y_offset = 0.0
                    y_cmd = probe_y + float(y_offset)

                    print(
                        f"{test_name} F{feedrate:g} C{orientation_id} "
                        f"{point['point_name']}: B={b_value:.4f}, target angle={point['target_angle_deg']:.1f}°"
                    )
                    command_record = send_absolute_move(
                        cal,
                        feedrate=float(point["move_feedrate"]),
                        **{
                            args.robot_rear_axis_name: b_value,
                            args.robot_stage_y_axis_name: y_cmd,
                        },
                    )
                    estimated_motion_s = record_estimated_motion(cal, command_record)
                    wait_for_estimated_motion_complete(
                        cal,
                        extra_settle_s=float(point["settle_s"]),
                        extra_buffer_s=float(args.move_buffer_s),
                        reason=(
                            f"abs move ({estimated_motion_s:.3f} s est @ "
                            f"F{format_feedrate(point['move_feedrate'])})"
                        ),
                    )

                    file_name = (
                        f"{int(orientation_id)}_{probe_x:.2f}_{b_value:.2f}"
                        f"_Y{y_cmd:.2f}_Z{probe_z:.2f}_P01_S{int(point['step_idx']):02d}"
                        f"_DIR{point['motion_phase']}_PASS1_GRP{capture_group}"
                        f"_CURLPASS{test_token}_F{feed_token}.png"
                    )
                    capture_frame(cal, raw_dir / file_name)
                    if float(args.post_capture_buffer_s) > 0.0:
                        time.sleep(float(args.post_capture_buffer_s))

                    plan_records.append(
                        {
                            "image_file": file_name,
                            "test_name": test_name,
                            "feedrate": float(feedrate),
                            "orientation": int(orientation_id),
                            "point_name": point["point_name"],
                            "point_order": int(point["point_order"]),
                            "target_angle_deg": float(point["target_angle_deg"]),
                            "b_value": b_value,
                            "motion_phase": point["motion_phase"],
                            "capture_group": capture_group,
                            "probe_x": probe_x,
                            "probe_y": probe_y,
                            "probe_z": probe_z,
                            "stage_y_cmd": y_cmd,
                            "stage_y_offset": float(y_offset),
                        }
                    )

    current_orientation = rotate_to_orientation(cal, current_orientation, 0, args)

    plan_df = pd.DataFrame(plan_records)
    plan_path = Path(cal.calibration_data_folder) / "hysteresis_capture_plan.csv"
    plan_df.to_csv(plan_path, index=False)
    print(f"\nSaved capture plan: {plan_path}")
    return plan_df


# ----------------------------- Analysis and plotting -----------------------------


def px_to_rz_mm(
    cal: CTR_Shadow_Calibration,
    x_px: float,
    y_px: float,
    width_in_pixels: float,
    width_in_mm: float,
) -> tuple[float, float, str]:
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
        r_mm, z_mm = cal.pixel_point_to_calibrated_axes(
            x_px=float(x_px),
            y_px=float(y_px),
            origin_px=origin_px,
        )
        return float(r_mm), float(z_mm), "board_reference_calibrated"

    if getattr(cal, "ruler_mm_per_px", None) is not None:
        scale = float(cal.ruler_mm_per_px)
        return float(x_px) * scale, -float(y_px) * scale, "ruler_reference_scale"

    scale = float(width_in_mm) / float(width_in_pixels)
    return float(x_px) * scale, -float(y_px) * scale, "legacy_linear_scale"


def build_results_tables(cal: CTR_Shadow_Calibration, args: argparse.Namespace, plan_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    processed_dir = Path(cal.calibration_data_folder) / "processed_image_data_folder"
    selected_csv = processed_dir / "tip_locations_selected.csv"
    if not selected_csv.exists():
        raise FileNotFoundError(f"Missing analyzed tip CSV: {selected_csv}")

    tip_df = pd.read_csv(selected_csv)
    df = tip_df.merge(plan_df, on="image_file", how="inner", suffixes=("_analysis", ""))
    if df.empty:
        raise RuntimeError("No analyzed images matched hysteresis_capture_plan.csv.")

    r_vals = []
    z_vals = []
    modes = []
    for _, row in df.iterrows():
        r_mm, z_mm, mode = px_to_rz_mm(
            cal,
            x_px=float(row["tip_column"]),
            y_px=float(row["tip_row"]),
            width_in_pixels=float(args.width_in_pixels),
            width_in_mm=float(args.width_in_mm),
        )
        r_vals.append(r_mm)
        z_vals.append(z_mm)
        modes.append(mode)
    df["r_mm"] = r_vals
    df["z_mm"] = z_vals
    df["coordinate_mode"] = modes

    # Use one orientation for summary plots to avoid mixing mirrored camera views.
    plot_df = df[df["orientation"].astype(int) == int(args.analysis_orientation)].copy()
    if plot_df.empty:
        raise RuntimeError(
            f"No rows for analysis_orientation={args.analysis_orientation}. "
            f"Available orientations: {sorted(df['orientation'].dropna().astype(int).unique().tolist())}"
        )

    summary_rows = []
    group_cols = ["test_name", "feedrate", "point_name", "point_order"]
    for keys, g in plot_df.groupby(group_cols, dropna=False):
        test_name, feedrate, point_name, point_order = keys
        summary_rows.append(
            {
                "test_name": test_name,
                "feedrate": float(feedrate),
                "point_name": point_name,
                "point_order": int(point_order),
                "n": int(len(g)),
                "target_angle_deg": float(np.nanmean(g["target_angle_deg"].astype(float))),
                "b_value": float(np.nanmean(g["b_value"].astype(float))),
                "r_mm": float(np.nanmean(g["r_mm"].astype(float))),
                "z_mm": float(np.nanmean(g["z_mm"].astype(float))),
                "tip_angle_deg": circular_mean_deg(g["tip_angle_deg"].astype(float)),
                "r_std_mm": float(np.nanstd(g["r_mm"].astype(float), ddof=0)),
                "z_std_mm": float(np.nanstd(g["z_mm"].astype(float), ddof=0)),
            }
        )
    summary_df = pd.DataFrame(summary_rows).sort_values(["test_name", "feedrate", "point_order"])

    rel_rows = []
    for (test_name, feedrate), g in summary_df.groupby(["test_name", "feedrate"]):
        start = g[g["point_name"] == "start"]
        if start.empty:
            continue
        start_row = start.iloc[0]
        for _, row in g.iterrows():
            dr = float(row["r_mm"] - start_row["r_mm"])
            dz = float(row["z_mm"] - start_row["z_mm"])
            rel_rows.append(
                {
                    "test_name": test_name,
                    "feedrate": float(feedrate),
                    "point_name": row["point_name"],
                    "point_order": int(row["point_order"]),
                    "target_angle_deg": float(row["target_angle_deg"]),
                    "b_value": float(row["b_value"]),
                    "r_mm": float(row["r_mm"]),
                    "z_mm": float(row["z_mm"]),
                    "tip_angle_deg": float(row["tip_angle_deg"]),
                    "delta_r_mm": dr,
                    "delta_z_mm": dz,
                    "delta_position_mm": float(math.hypot(dr, dz)),
                    "delta_tip_angle_deg": wrap_angle_delta_deg(float(row["tip_angle_deg"] - start_row["tip_angle_deg"])),
                }
            )
    rel_df = pd.DataFrame(rel_rows).sort_values(["test_name", "feedrate", "point_order"])

    raw_path = processed_dir / "hysteresis_points_raw.csv"
    summary_path = processed_dir / "hysteresis_points_summary.csv"
    rel_path = processed_dir / "hysteresis_points_relative_to_start.csv"
    df.to_csv(raw_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    rel_df.to_csv(rel_path, index=False)
    print(f"Saved raw points: {raw_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Saved relative-to-start summary: {rel_path}")

    return df, summary_df, rel_df


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
    frame = legend.get_frame()
    frame.set_facecolor((0.0, 0.0, 0.0, 0.0))
    frame.set_edgecolor((1.0, 1.0, 1.0, 0.25))
    for text in legend.get_texts():
        text.set_color("white")


def _save_transparent_figure(fig: plt.Figure, output_path: Path) -> None:
    fig.patch.set_alpha(0.0)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", transparent=True)


def _series_with_common_start(g: pd.DataFrame, y_col: str) -> pd.DataFrame:
    g = g.sort_values("feedrate").copy()
    if g.empty:
        return g
    first_value = float(g[y_col].iloc[0])
    g[y_col] = g[y_col].astype(float) - first_value
    return g


def _line_plot(
    ax,
    rel_df: pd.DataFrame,
    test_name: str,
    y_col: str,
    ylabel: str,
    align_series_start: bool = False,
) -> None:
    for point_name, marker in [("apogee", "o"), ("end", "s")]:
        g = rel_df[(rel_df["test_name"] == test_name) & (rel_df["point_name"] == point_name)].copy()
        if g.empty:
            continue
        if align_series_start:
            g = _series_with_common_start(g, y_col)
        else:
            g = g.sort_values("feedrate")
        ax.plot(g["feedrate"], g[y_col], marker=marker, linewidth=1.5, label=point_name)
    ax.axhline(0.0, linewidth=0.8, alpha=0.5)
    ax.set_title(f"{test_name}: {ylabel}")
    ax.set_xlabel("Feedrate")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, color="white")
    ax.legend(fontsize=8)
    _style_axes_for_dark_theme(ax)
    _style_legend_for_dark_theme(ax)


def _rz_scatter(ax, summary_df: pd.DataFrame, test_name: str) -> None:
    g = summary_df[summary_df["test_name"] == test_name].copy()
    if g.empty:
        ax.set_title(f"{test_name}: RZ scatter")
        _style_axes_for_dark_theme(ax)
        return

    marker_map = {"start": "o", "apogee": "^", "end": "s"}
    for point_name in ["start", "apogee", "end"]:
        gp = g[g["point_name"] == point_name].copy()
        if gp.empty:
            continue
        ax.scatter(
            gp["r_mm"],
            gp["z_mm"],
            s=28,
            marker=marker_map.get(point_name, "o"),
            label=point_name,
            alpha=0.85,
        )
        for _, row in gp.iterrows():
            ax.annotate(
                f"{int(row['feedrate'])}",
                (float(row["r_mm"]), float(row["z_mm"])),
                fontsize=6,
                xytext=(2, 2),
                textcoords="offset points",
                alpha=0.8,
            )
    ax.set_title(f"{test_name}: RZ red-dot points")
    ax.set_xlabel("R / lateral axis (mm)")
    ax.set_ylabel("Z axis (mm)")
    ax.axis("equal")
    ax.grid(True, alpha=0.25, color="white")
    ax.legend(fontsize=8)
    _style_axes_for_dark_theme(ax)
    _style_legend_for_dark_theme(ax)


def make_plots(summary_df: pd.DataFrame, rel_df: pd.DataFrame, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    test_order = [name for name, _ in DEFAULT_TESTS]
    present_tests = [t for t in test_order if t in set(summary_df["test_name"].astype(str))]
    if not present_tests:
        present_tests = sorted(summary_df["test_name"].astype(str).unique().tolist())

    # One panel with all requested plots.
    fig, axes = plt.subplots(
        nrows=len(present_tests),
        ncols=4,
        figsize=(22, 4.8 * len(present_tests)),
        squeeze=False,
    )
    for row_idx, test_name in enumerate(present_tests):
        _line_plot(
            axes[row_idx, 0],
            rel_df,
            test_name,
            "delta_r_mm",
            "ΔR from start, aligned at first feedrate (mm)",
            align_series_start=True,
        )
        _line_plot(
            axes[row_idx, 1],
            rel_df,
            test_name,
            "delta_z_mm",
            "ΔZ from start, aligned at first feedrate (mm)",
            align_series_start=True,
        )
        _line_plot(
            axes[row_idx, 2],
            rel_df,
            test_name,
            "delta_tip_angle_deg",
            "Δtip angle from start, aligned at first feedrate (deg)",
            align_series_start=True,
        )
        _rz_scatter(axes[row_idx, 3], summary_df, test_name)
    fig.patch.set_alpha(0.0)
    fig.suptitle("CTR hysteresis speed sweep: red-dot tracked tip", fontsize=16, color="white")
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    panel_path = output_dir / "hysteresis_all_plots_panel.png"
    _save_transparent_figure(fig, panel_path)
    plt.close(fig)
    print(f"Saved plot panel: {panel_path}")

    # Individual figures for easier review.
    for y_col, ylabel, filename in [
        ("delta_r_mm", "ΔR from start, aligned at first feedrate (mm)", "hysteresis_delta_r_vs_feedrate.png"),
        ("delta_z_mm", "ΔZ from start, aligned at first feedrate (mm)", "hysteresis_delta_z_vs_feedrate.png"),
        (
            "delta_position_mm",
            "Δposition magnitude from start, aligned at first feedrate (mm)",
            "hysteresis_delta_position_vs_feedrate.png",
        ),
        (
            "delta_tip_angle_deg",
            "Δtip angle from start, aligned at first feedrate (deg)",
            "hysteresis_delta_angle_vs_feedrate.png",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(9, 6))
        fig.patch.set_alpha(0.0)
        for test_name in present_tests:
            for point_name, marker in [("apogee", "o"), ("end", "s")]:
                g = rel_df[(rel_df["test_name"] == test_name) & (rel_df["point_name"] == point_name)].copy()
                if g.empty:
                    continue
                g = _series_with_common_start(g, y_col)
                ax.plot(
                    g["feedrate"],
                    g[y_col],
                    marker=marker,
                    linewidth=1.5,
                    label=f"{test_name} {point_name}",
                )
        ax.axhline(0.0, linewidth=0.8, alpha=0.5)
        ax.set_xlabel("Feedrate")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel + " vs feedrate")
        ax.grid(True, alpha=0.25, color="white")
        ax.legend(fontsize=8, ncol=2)
        _style_axes_for_dark_theme(ax)
        _style_legend_for_dark_theme(ax)
        fig.tight_layout()
        _save_transparent_figure(fig, output_dir / filename)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 8))
    fig.patch.set_alpha(0.0)
    for test_name in present_tests:
        g = summary_df[summary_df["test_name"] == test_name].copy()
        for point_name, marker in [("start", "o"), ("apogee", "^"), ("end", "s")]:
            gp = g[g["point_name"] == point_name]
            if gp.empty:
                continue
            ax.scatter(gp["r_mm"], gp["z_mm"], marker=marker, s=30, label=f"{test_name} {point_name}", alpha=0.8)
    ax.set_xlabel("R / lateral axis (mm)")
    ax.set_ylabel("Z axis (mm)")
    ax.set_title("RZ scatter of start/apogee/end red-dot tip points")
    ax.axis("equal")
    ax.grid(True, alpha=0.25, color="white")
    ax.legend(fontsize=7, ncol=2)
    _style_axes_for_dark_theme(ax)
    _style_legend_for_dark_theme(ax)
    fig.tight_layout()
    rz_path = output_dir / "hysteresis_rz_scatter.png"
    _save_transparent_figure(fig, rz_path)
    plt.close(fig)
    print(f"Saved RZ scatter: {rz_path}")


# ----------------------------- Main -----------------------------


def main() -> None:
    args = build_arg_parser().parse_args()

    cal_data = load_calibration_export(Path(args.calibration_file))
    pull_bundle = get_phase_fit_bundle(cal_data, "pull")
    pull_tip_angle_model = get_model_descriptor(pull_bundle, "tip_angle", fit_family="pchip")
    pull_r_model = get_model_descriptor(pull_bundle, "r", fit_family="pchip")
    if pull_tip_angle_model is None:
        raise RuntimeError("Calibration file does not contain a pull-phase tip_angle PCHIP model.")

    cal = CTR_Shadow_Calibration(
        parent_directory=str(SCRIPT_DIR),
        project_name=str(args.project_name),
        allow_existing=True,
        add_date=False,
    )
    attach_calibration_state_for_class(cal, cal_data)
    configure_tip_detection(cal, args)
    load_camera_and_board_calibration(cal, args)

    b_plan = build_b_plan(cal, pull_tip_angle_model, DEFAULT_TESTS)
    plan_json = Path(cal.calibration_data_folder) / "hysteresis_b_plan.json"
    with plan_json.open("w") as f:
        json.dump(json_ready(b_plan), f, indent=2)
    print(f"Saved B plan: {plan_json}")

    cal.connect_to_camera(cam_port=int(args.cam_port), show_preview=False)
    cal.setup_analysis_crop(enable_manual_adjustment=bool(args.manual_crop_adjustment))
    cal.connect_to_robot(duet_web_address=str(args.duet_web_address))

    try:
        plan_df = run_hysteresis_acquisition(cal, args, b_plan, pull_r_model)

        cal.analyze_data_batch(
            threshold=int(args.threshold),
            export_analysis_outputs=bool(args.export_analysis_outputs),
        )

        processed_dir = Path(cal.calibration_data_folder) / "processed_image_data_folder"
        plan_df.to_csv(processed_dir / "hysteresis_capture_plan.csv", index=False)
        raw_df, summary_df, rel_df = build_results_tables(cal, args, plan_df)
        make_plots(summary_df, rel_df, processed_dir)

        print("\nDone.")
        print(f"Project folder: {cal.calibration_data_folder}")
        print(f"Processed outputs: {processed_dir}")
        print("Key files:")
        print(f" - {processed_dir / 'hysteresis_points_raw.csv'}")
        print(f" - {processed_dir / 'hysteresis_points_summary.csv'}")
        print(f" - {processed_dir / 'hysteresis_points_relative_to_start.csv'}")
        print(f" - {processed_dir / 'hysteresis_all_plots_panel.png'}")

    finally:
        if bool(args.return_to_b0_on_exit) and getattr(cal, "rrf", None) is not None:
            try:
                print("[INFO] Returning B to 0 before exit.")
                command_record = send_absolute_move(
                    cal,
                    feedrate=float(args.preposition_b_feedrate),
                    **{args.robot_rear_axis_name: 0.0},
                )
                estimated_motion_s = record_estimated_motion(cal, command_record)
                wait_for_estimated_motion_complete(
                    cal,
                    extra_settle_s=0.5,
                    extra_buffer_s=float(args.move_buffer_s),
                    reason=(
                        f"return B to 0 ({estimated_motion_s:.3f} s est @ "
                        f"F{format_feedrate(args.preposition_b_feedrate)})"
                    ),
                )
            except Exception as exc:
                print(f"[WARN] Failed to return B to 0: {exc}")
        try:
            if getattr(cal, "cam", None) is not None:
                cal.disconnect_camera()
        except Exception:
            pass


if __name__ == "__main__":
    main()
