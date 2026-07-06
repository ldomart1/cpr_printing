#!/usr/bin/env python3
import argparse
import csv
import inspect
import json
import math
import os
import shutil
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from shadow_calibration import CTR_Shadow_Calibration


# =========================
# DEFAULT USER CONFIGURATION
# =========================
PROJECT_NAME = "Y_Axis_Speed_2026_06_16_overnight"
ALLOW_EXISTING_PROJECT = True
ADD_DATE_TO_PROJECT_FOLDER = True

CAMERA_PORT = 0
SHOW_CAMERA_PREVIEW = False
CAMERA_WARMUP_FRAMES = 5
RAW_IMAGE_EXTENSION = ".jpg"
RAW_IMAGE_JPEG_QUALITY = 80
RESET_EXISTING_OUTPUTS = False

MANUAL_CROP_ADJUSTMENT = True
USE_CLASS_ANALYSIS_CROP_SETUP = False

ROBOT_FRONT_AXIS_NAME = "V"
ROBOT_STAGE_Y_AXIS_NAME = "Y"
ROBOT_STAGE_Z_AXIS_NAME = "Z"
ROBOT_REAR_AXIS_NAME = "B"
ROBOT_ROTATION_AXIS_NAME = "C"

# Fixed robot pose during acquisition
FIXED_X = 90.0
FIXED_Y = 0.0
FIXED_Z = -170.0
FIXED_B = 0.0

# Motion settings
Y_MIDDLE_MM = -10.0
Y_FINAL_MM = -50.0
Y_ACCEL_MM_S2 = 200.0
POSITION_ACCEL_MM_S2 = 200.0
RETURN_FEEDRATE = 1000.0
Z_LIFT_MM = -80.0
Z_LIFT_FEEDRATE = 5000.0
Z_FINAL_APPROACH_DISTANCE_MM = 90.0
Z_FINAL_APPROACH_FEEDRATE = 800.0
ROTATION_FEEDRATE = 15000.0
MIDDLE_PAUSE_S = 2.0
INITIAL_IDLE_BEFORE_MOTION_S = 2.0
PREMOVE_SETTLE_S = 0.15
POSTMOVE_SETTLE_S = 0.10
POSITION_WAIT_BUFFER_S = 0.05

MOTION_WAIT_BUFFER_S = 0.05
ORIENTATION_CHANGE_SETTLE_S = 0.15
SMOOTH_MOVE_SAMPLES = 200
INTER_COMMAND_DELAY_S = 0.005

# Orientation series
ORIENTATION_SEQUENCE_DEG = [0.0, 180.0, 90.0]

# DAQ mode selection
DAQ_MODE = "speed_series"

# B-attack sweep mode defaults. The target attack angle is the physical curl/attack
# angle in degrees; the commanded B-axis position is evaluated from the average
# PCHIP calibration fit loaded from --b-curl-calibration-file.
B_ATTACK_START_DEG = 0.0
B_ATTACK_STOP_DEG = 180.0
B_ATTACK_STEP_DEG = 5.0
B_ATTACK_ORIENTATION_SEQUENCE_DEG = [0.0]
B_ATTACK_RUN_TYPE = "two_stage"

# Capture settings
CAPTURE_MAX_FPS = 6.0
CAPTURE_RETRY_LIMIT = 10
CAPTURE_RETRY_WAIT_S = 0.2
MAX_SWEEP_SPEED_MM_MIN = 1000.0
MAX_B_FEEDRATE_MM_MIN = 1500.0


@dataclass(frozen=True)
class OrientationSpec:
    angle_deg: float
    dataset_label: str


@dataclass(frozen=True)
class RunSpec:
    run_type: str
    folder_suffix: str
    segment_names: Tuple[str, ...]


ORIENTATION_SPECS_DEFAULT = [
    OrientationSpec(angle_deg=0.0, dataset_label="forward"),
    OrientationSpec(angle_deg=180.0, dataset_label="backward"),
    OrientationSpec(angle_deg=90.0, dataset_label="transverse"),
]

RUN_SPECS = [
    RunSpec(run_type="direct", folder_suffix="direct", segment_names=("full_sweep",)),
    RunSpec(run_type="two_stage", folder_suffix="two_stage", segment_names=("start_to_mid", "mid_to_end")),
]


# =========================
# HELPERS
# =========================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def reset_dir(path: str) -> str:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


def append_csv_row(csv_path: str, row: Dict[str, object]) -> None:
    file_exists = os.path.isfile(csv_path)
    fieldnames = list(row.keys())
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def sanitize_speed_folder_name(speed_mm_min: float) -> str:
    speed_str = f"{float(speed_mm_min):.3f}".rstrip("0").rstrip(".")
    speed_str = speed_str.replace(".", "p")
    return f"speed_{speed_str}_mm_min"


def normalize_orientation_deg(angle_deg: float) -> float:
    angle = float(angle_deg) % 360.0
    if math.isclose(angle, 360.0, abs_tol=1e-9):
        angle = 0.0
    return angle


def orientation_name(angle_deg: float) -> str:
    angle = int(round(normalize_orientation_deg(angle_deg)))
    return f"c{angle:03d}"


def dataset_label_for_orientation(angle_deg: float) -> str:
    angle = int(round(normalize_orientation_deg(angle_deg)))
    if angle == 0:
        return "forward"
    if angle == 180:
        return "backward"
    if angle == 90:
        return "transverse"
    return f"orientation_{orientation_name(angle_deg)}"


def build_orientation_specs(orientation_angles_deg: Sequence[float]) -> List[OrientationSpec]:
    specs: List[OrientationSpec] = []
    for angle_deg in orientation_angles_deg:
        specs.append(
            OrientationSpec(
                angle_deg=normalize_orientation_deg(float(angle_deg)),
                dataset_label=dataset_label_for_orientation(float(angle_deg)),
            )
        )
    return specs


def build_run_folder_name(
    orientation_spec: OrientationSpec,
    run_spec: RunSpec,
    speed_mm_min: float,
    repeat_idx: int = 1,
    repeat_count: int = 1,
) -> str:
    return (
        f"{orientation_name(orientation_spec.angle_deg)}_"
        f"{orientation_spec.dataset_label}_"
        f"{run_spec.folder_suffix}_"
        f"{sanitize_speed_folder_name(speed_mm_min)}_"
        f"pass_{int(repeat_idx):02d}_of_{int(repeat_count):02d}"
    )



def sanitize_angle_folder_token(angle_deg: float) -> str:
    angle_str = f"{float(angle_deg):.3f}".rstrip("0").rstrip(".")
    angle_str = angle_str.replace("-", "m").replace(".", "p")
    return angle_str


def build_b_attack_run_folder_name(
    orientation_spec: OrientationSpec,
    run_spec: RunSpec,
    speed_mm_min: float,
    attack_angle_deg: float,
    b_command: float,
    repeat_idx: int = 1,
    repeat_count: int = 1,
) -> str:
    return (
        f"{orientation_name(orientation_spec.angle_deg)}_"
        f"{orientation_spec.dataset_label}_"
        f"b_attack_"
        f"attack_{sanitize_angle_folder_token(attack_angle_deg)}deg_"
        f"bcmd_{sanitize_angle_folder_token(b_command)}_"
        f"{run_spec.folder_suffix}_"
        f"{sanitize_speed_folder_name(speed_mm_min)}_"
        f"pass_{int(repeat_idx):02d}_of_{int(repeat_count):02d}"
    )


def build_attack_angle_sequence(start_deg: float, stop_deg: float, step_deg: float) -> List[float]:
    start = float(start_deg)
    stop = float(stop_deg)
    step = float(step_deg)
    if step <= 0.0:
        raise ValueError("B attack step must be > 0")
    if stop < start:
        raise ValueError("B attack stop must be >= start")
    values: List[float] = []
    cur = start
    guard = 0
    while cur <= stop + 1e-9:
        values.append(round(cur, 10))
        cur += step
        guard += 1
        if guard > 10000:
            raise RuntimeError("B attack angle sequence generation exceeded safety guard")
    if not math.isclose(values[-1], stop, abs_tol=1e-9):
        values.append(stop)
    return [float(v) for v in values]


def selected_run_specs(run_type: str) -> List[RunSpec]:
    selected = str(run_type).strip().lower()
    if selected == "both":
        return list(RUN_SPECS)
    return [run_spec for run_spec in RUN_SPECS if run_spec.run_type == selected]


def fixed_b_folder_name(fixed_b: float) -> str:
    return f"fixed_b_{sanitize_angle_folder_token(float(fixed_b))}"


def _as_float_list(value: object) -> Optional[List[float]]:
    if not isinstance(value, list) or len(value) < 2:
        return None
    out: List[float] = []
    for item in value:
        if isinstance(item, (int, float)) and math.isfinite(float(item)):
            out.append(float(item))
        else:
            return None
    return out


def _walk_json_numeric_arrays(obj: object, path: Tuple[str, ...] = ()) -> List[Tuple[Tuple[str, ...], List[float]]]:
    arrays: List[Tuple[Tuple[str, ...], List[float]]] = []
    arr = _as_float_list(obj)
    if arr is not None:
        arrays.append((path, arr))
        return arrays
    if isinstance(obj, dict):
        for key, value in obj.items():
            arrays.extend(_walk_json_numeric_arrays(value, path + (str(key),)))
    elif isinstance(obj, list):
        for idx, value in enumerate(obj):
            arrays.extend(_walk_json_numeric_arrays(value, path + (str(idx),)))
    return arrays


def _path_text(path: Tuple[str, ...]) -> str:
    return "/".join(path).lower()


def _array_range_score(values: Sequence[float], lo: float, hi: float) -> float:
    if not values:
        return -1e9
    mn = min(values)
    mx = max(values)
    span = mx - mn
    score = 0.0
    if mn <= lo + 2.0 and mx >= hi - 2.0:
        score += 20.0
    if span >= (hi - lo) * 0.75:
        score += 10.0
    score -= abs(mn - lo) * 0.02
    score -= abs(mx - hi) * 0.02
    return score


def _make_interp(x: Sequence[float], y: Sequence[float]):
    pairs = sorted((float(a), float(b)) for a, b in zip(x, y) if math.isfinite(float(a)) and math.isfinite(float(b)))
    dedup_x: List[float] = []
    dedup_y: List[float] = []
    for x_val, y_val in pairs:
        if dedup_x and math.isclose(x_val, dedup_x[-1], abs_tol=1e-12):
            dedup_y[-1] = y_val
        else:
            dedup_x.append(x_val)
            dedup_y.append(y_val)
    if len(dedup_x) < 2:
        raise ValueError("Need at least two unique calibration points for interpolation")
    try:
        from scipy.interpolate import PchipInterpolator  # type: ignore
        interpolator = PchipInterpolator(np.asarray(dedup_x, dtype=float), np.asarray(dedup_y, dtype=float), extrapolate=True)
        return lambda q: float(interpolator(float(q)))
    except Exception:
        return lambda q: float(np.interp(float(q), np.asarray(dedup_x, dtype=float), np.asarray(dedup_y, dtype=float)))


def _extract_model_by_path(payload: Dict[str, object], path: Sequence[str]) -> Optional[Dict[str, object]]:
    cur: object = payload
    for token in path:
        if not isinstance(cur, dict) or token not in cur:
            return None
        cur = cur[token]
    return cur if isinstance(cur, dict) else None


def _build_clamped_inverse_fit_from_model(
    model: Dict[str, object],
    model_label: str,
):
    x_vals = _as_float_list(model.get("x_knots"))
    y_vals = _as_float_list(model.get("y_knots"))
    if x_vals is None or y_vals is None or len(x_vals) != len(y_vals) or len(x_vals) < 2:
        return None

    fit = _make_interp(y_vals, x_vals)
    min_b = min(float(v) for v in x_vals)
    max_b = max(float(v) for v in x_vals)
    min_angle = min(float(v) for v in y_vals)
    max_angle = max(float(v) for v in y_vals)

    print(
        "[INFO] B attack calibration fit selected | "
        f"mode=inverse-known-model | model={model_label} | "
        f"angle_range=[{min_angle:.3f}, {max_angle:.3f}] deg | "
        f"b_range=[{min_b:.5f}, {max_b:.5f}]"
    )

    def attack_to_b_command(query_angle_deg: float) -> float:
        query = float(query_angle_deg)
        clamped_angle = min(max(query, min_angle), max_angle)
        if not math.isclose(query, clamped_angle, abs_tol=1e-9):
            print(
                f"[WARN] Requested attack angle {query:.3f} deg is outside calibrated range "
                f"[{min_angle:.3f}, {max_angle:.3f}] deg. Clamping to {clamped_angle:.3f} deg."
            )
        b_command = float(fit(clamped_angle))
        return min(max(b_command, min_b), max_b)

    return attack_to_b_command


def load_b_attack_to_b_command_fit(calibration_file: str):
    """
    Return a callable attack_angle_deg -> B command from a calibration JSON.

    The loader is intentionally permissive because the calibration JSON has evolved
    between experiments. It first looks for numeric arrays whose path names suggest
    a physical/tip/curl angle in degrees and a B-axis command/position, prioritizing
    paths containing "avg", "average", and "pchip". If the detected relationship is
    B -> angle, it inverts it by interpolating angle -> B.
    """
    if not calibration_file:
        raise ValueError("A B curl calibration JSON is required for b_attack_sweep mode")
    with open(os.path.abspath(calibration_file), "r") as f:
        payload = json.load(f)

    known_model_paths = [
        ("shared_aux_fit_models", "tip_angle_avg_pchip"),
        ("fit_models", "tip_angle_avg_pchip"),
        ("fit_models_by_phase", "pull", "tip_angle_avg_pchip"),
        ("fit_models_by_phase", "pull_1", "tip_angle_avg_pchip"),
        ("fit_models", "tip_angle_pchip"),
        ("fit_models_by_phase", "pull", "tip_angle_pchip"),
        ("fit_models_by_phase", "pull_1", "tip_angle_pchip"),
    ]
    for model_path in known_model_paths:
        model = _extract_model_by_path(payload, model_path)
        if model is None:
            continue
        fit = _build_clamped_inverse_fit_from_model(
            model=model,
            model_label="/".join(model_path),
        )
        if fit is not None:
            return fit

    arrays = _walk_json_numeric_arrays(payload)
    if not arrays:
        raise ValueError(f"No numeric arrays found in calibration file: {calibration_file}")

    candidates: List[Tuple[float, str, Tuple[str, ...], Tuple[str, ...], List[float], List[float]]] = []
    for x_path, x_vals in arrays:
        for y_path, y_vals in arrays:
            if x_path == y_path or len(x_vals) != len(y_vals) or len(x_vals) < 3:
                continue
            x_text = _path_text(x_path)
            y_text = _path_text(y_path)
            joined = x_text + " " + y_text
            score = 0.0
            # Direct map: attack/tip/curl angle degrees -> B command.
            if any(tok in x_text for tok in ("attack", "tip_angle", "curl", "angle_deg", "angle")):
                score += 14.0
            if "deg" in x_text or "degree" in x_text:
                score += 5.0
            if any(tok in y_text for tok in ("b_command", "b_cmd", "b_axis", "rear", "b_mm", "b_position")):
                score += 14.0
            if any(tok in joined for tok in ("avg", "average", "mean")):
                score += 8.0
            if "pchip" in joined:
                score += 8.0
            score += _array_range_score(x_vals, 0.0, 180.0)
            if score > 25.0:
                candidates.append((score, "direct", x_path, y_path, x_vals, y_vals))

            # Inverse map: B command -> measured attack/tip/curl angle degrees.
            inv_score = 0.0
            if any(tok in x_text for tok in ("b_command", "b_cmd", "b_axis", "rear", "b_mm", "b_position")):
                inv_score += 12.0
            if any(tok in y_text for tok in ("attack", "tip_angle", "curl", "angle_deg", "angle")):
                inv_score += 14.0
            if "deg" in y_text or "degree" in y_text:
                inv_score += 5.0
            if any(tok in joined for tok in ("avg", "average", "mean")):
                inv_score += 8.0
            if "pchip" in joined:
                inv_score += 8.0
            inv_score += _array_range_score(y_vals, 0.0, 180.0)
            if inv_score > 25.0:
                candidates.append((inv_score, "inverse", x_path, y_path, x_vals, y_vals))

    if not candidates:
        raise ValueError(
            "Could not auto-detect avg PCHIP B-curl calibration arrays. Expected arrays resembling "
            "attack/curl/tip angle degrees and B command/position."
        )

    candidates.sort(key=lambda item: item[0], reverse=True)
    score, mode, x_path, y_path, x_vals, y_vals = candidates[0]
    print(
        "[INFO] B attack calibration fit selected | "
        f"mode={mode} | score={score:.2f} | x={'/'.join(x_path)} | y={'/'.join(y_path)}"
    )
    if mode == "direct":
        return _make_interp(x_vals, y_vals)
    return _make_interp(y_vals, x_vals)


def capture_frame(camera: cv2.VideoCapture, warmup_frames: int = 2) -> np.ndarray:
    frame = None
    for _ in range(max(1, int(warmup_frames))):
        ret, tmp = camera.read()
        if ret and tmp is not None:
            frame = tmp

    ret, tmp = camera.read()
    if ret and tmp is not None:
        frame = tmp

    if frame is None:
        raise RuntimeError("Could not capture frame from camera.")
    return frame


def capture_frame_with_retries(camera: cv2.VideoCapture) -> np.ndarray:
    for attempt_idx in range(1, CAPTURE_RETRY_LIMIT + 1):
        try:
            ret, frame = camera.read()
            if ret and frame is not None:
                return frame
        except Exception:
            pass

        if attempt_idx < CAPTURE_RETRY_LIMIT:
            time.sleep(float(max(0.0, CAPTURE_RETRY_WAIT_S)))

    raise RuntimeError("Could not capture frame from camera after retries.")


def save_compressed_image(
    image_bgr: np.ndarray,
    output_path: str,
    jpeg_quality: Optional[int] = None,
) -> None:
    ext = os.path.splitext(output_path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        quality = RAW_IMAGE_JPEG_QUALITY if jpeg_quality is None else int(jpeg_quality)
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    else:
        params = []

    ok = cv2.imwrite(output_path, image_bgr, params)
    if not ok:
        raise RuntimeError(f"Failed to save image: {output_path}")


def prepare_calibration_object(project_name: str) -> CTR_Shadow_Calibration:
    print(f"[DEBUG] SCRIPT_DIR={SCRIPT_DIR}")
    import shadow_calibration
    print(f"[DEBUG] shadow_calibration loaded from: {inspect.getsourcefile(shadow_calibration)}")

    cal = CTR_Shadow_Calibration(
        parent_directory=SCRIPT_DIR,
        project_name=project_name,
        allow_existing=ALLOW_EXISTING_PROJECT,
        add_date=ADD_DATE_TO_PROJECT_FOLDER,
    )
    print("[INFO] Calibration object created.")

    cal.connect_to_camera(cam_port=CAMERA_PORT, show_preview=SHOW_CAMERA_PREVIEW)

    if USE_CLASS_ANALYSIS_CROP_SETUP:
        cal.setup_analysis_crop(enable_manual_adjustment=MANUAL_CROP_ADJUSTMENT)

    cal.connect_to_robot()
    return cal


# =========================
# MOTION PROFILE HELPERS
# =========================
def compute_motion_profile(distance_mm: float, feedrate_mm_min: float, accel_mm_s2: float) -> Dict[str, float]:
    if distance_mm == 0:
        raise ValueError("distance_mm must be != 0")
    if feedrate_mm_min <= 0:
        raise ValueError("feedrate_mm_min must be > 0")
    if accel_mm_s2 <= 0:
        raise ValueError("accel_mm_s2 must be > 0")

    distance_mm = abs(float(distance_mm))
    accel_mm_s2 = float(accel_mm_s2)
    speed_mm_s = float(feedrate_mm_min) / 60.0
    d_accel_mm = (speed_mm_s ** 2) / (2.0 * accel_mm_s2)

    if 2.0 * d_accel_mm < distance_mm:
        t_accel_s = speed_mm_s / accel_mm_s2
        d_ss_mm = distance_mm - 2.0 * d_accel_mm
        t_ss_s = d_ss_mm / speed_mm_s
        t_total_s = 2.0 * t_accel_s + t_ss_s
        return {
            "speed_mm_s": speed_mm_s,
            "accel_mm_s2": accel_mm_s2,
            "distance_mm": distance_mm,
            "t_accel_s": t_accel_s,
            "t_ss_s": t_ss_s,
            "t_total_s": t_total_s,
            "d_accel_mm": d_accel_mm,
            "d_ss_mm": d_ss_mm,
            "v_peak_mm_s": speed_mm_s,
            "has_steady_state": True,
            "profile_type": "trapezoidal",
            "t_ss_start_s": t_accel_s,
            "t_ss_end_s": t_accel_s + t_ss_s,
            "t_decel_start_s": t_accel_s + t_ss_s,
        }

    t_accel_s = math.sqrt(distance_mm / accel_mm_s2)
    v_peak_mm_s = accel_mm_s2 * t_accel_s
    t_total_s = 2.0 * t_accel_s
    return {
        "speed_mm_s": speed_mm_s,
        "accel_mm_s2": accel_mm_s2,
        "distance_mm": distance_mm,
        "t_accel_s": t_accel_s,
        "t_ss_s": 0.0,
        "t_total_s": t_total_s,
        "d_accel_mm": distance_mm / 2.0,
        "d_ss_mm": 0.0,
        "v_peak_mm_s": v_peak_mm_s,
        "has_steady_state": False,
        "profile_type": "triangular",
        "t_ss_start_s": t_accel_s,
        "t_ss_end_s": t_accel_s,
        "t_decel_start_s": t_accel_s,
    }


def motion_profile_distance_mm(elapsed_s: float, profile: Dict[str, float]) -> float:
    t = min(max(0.0, float(elapsed_s)), float(profile["t_total_s"]))
    accel = float(profile["accel_mm_s2"])
    t_accel = float(profile["t_accel_s"])
    v_peak = float(profile["v_peak_mm_s"])

    if t <= t_accel:
        return 0.5 * accel * t * t

    if bool(profile["has_steady_state"]):
        if t <= float(profile["t_ss_end_s"]):
            accel_distance = (v_peak ** 2) / (2.0 * accel)
            cruise_time = t - t_accel
            return accel_distance + v_peak * cruise_time

        t_decel = t - float(profile["t_decel_start_s"])
        return (
            float(profile["d_accel_mm"])
            + float(profile["d_ss_mm"])
            + v_peak * t_decel
            - 0.5 * accel * t_decel * t_decel
        )

    t_decel = t - t_accel
    return float(profile["d_accel_mm"]) + v_peak * t_decel - 0.5 * accel * t_decel * t_decel


def motion_profile_distance_fraction(elapsed_s: float, profile: Dict[str, float]) -> float:
    distance = float(profile["distance_mm"])
    if distance <= 0.0:
        return 1.0
    return min(1.0, max(0.0, motion_profile_distance_mm(elapsed_s, profile) / distance))


def classify_motion_phase(elapsed_s: float, profile: Dict[str, float]) -> str:
    t_accel_s = float(profile["t_accel_s"])
    t_ss_end_s = float(profile["t_ss_end_s"])
    t_total_s = float(profile["t_total_s"])

    if elapsed_s < 0.0:
        return "pre"
    if elapsed_s < t_accel_s:
        return "accel"
    if bool(profile["has_steady_state"]) and elapsed_s < t_ss_end_s:
        return "ss"
    if elapsed_s <= t_total_s:
        return "decel"
    return "post"


def estimate_move_time_s(distance_mm: float, feedrate_mm_min: float, accel_mm_s2: float) -> float:
    if abs(float(distance_mm)) <= 1e-9:
        return 0.0
    profile = compute_motion_profile(
        distance_mm=abs(float(distance_mm)),
        feedrate_mm_min=float(feedrate_mm_min),
        accel_mm_s2=float(accel_mm_s2),
    )
    return float(profile["t_total_s"])


def estimate_absolute_move_time_s(
    current_pos: Dict[str, float],
    feedrate: float,
    accel_mm_s2: float,
    axes_targets: Dict[str, float],
) -> float:
    deltas = []
    for axis_name, axis_val in axes_targets.items():
        if axis_val is None:
            continue
        prev = current_pos.get(str(axis_name))
        if prev is None:
            continue
        deltas.append(abs(float(axis_val) - float(prev)))
    if not deltas:
        return 0.0
    return estimate_move_time_s(
        distance_mm=max(deltas),
        feedrate_mm_min=float(feedrate),
        accel_mm_s2=float(accel_mm_s2),
    )


def format_duration_hms(total_seconds: float) -> str:
    total_seconds = max(0, int(round(float(total_seconds))))
    hours, rem = divmod(total_seconds, 3600)
    minutes, seconds = divmod(rem, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def render_progress_bar(current: int, total: int, width: int = 30) -> str:
    total = max(1, int(total))
    current = min(max(0, int(current)), total)
    filled = int(round(width * current / total))
    filled = min(max(0, filled), width)
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def print_pass_progress(current_pass: int, total_passes: int) -> None:
    total_passes = max(1, int(total_passes))
    current_pass = min(max(0, int(current_pass)), total_passes)
    percent = 100.0 * float(current_pass) / float(total_passes)
    sys.stdout.write(
        "\r"
        f"[PROGRESS] {render_progress_bar(current_pass, total_passes)} "
        f"Pass {current_pass}/{total_passes} ({percent:.1f}%)"
    )
    sys.stdout.flush()
    if current_pass >= total_passes:
        sys.stdout.write("\n")
        sys.stdout.flush()


def estimate_descend_z_with_final_approach_time_s(
    current_z: float,
    target_z: float,
    fast_feedrate: float,
    slow_feedrate: float,
    slow_distance_mm: float,
    accel_mm_s2: float,
) -> float:
    current_z = float(current_z)
    target_z = float(target_z)
    slow_distance_mm = max(0.0, float(slow_distance_mm))

    if math.isclose(current_z, target_z, abs_tol=1e-9):
        return 0.0

    if target_z >= current_z:
        return estimate_move_time_s(
            distance_mm=abs(target_z - current_z),
            feedrate_mm_min=float(fast_feedrate),
            accel_mm_s2=float(accel_mm_s2),
        ) + float(POSITION_WAIT_BUFFER_S)

    intermediate_z = max(target_z, current_z - slow_distance_mm)
    total_s = 0.0
    if not math.isclose(current_z, intermediate_z, abs_tol=1e-9):
        total_s += estimate_move_time_s(
            distance_mm=abs(intermediate_z - current_z),
            feedrate_mm_min=float(fast_feedrate),
            accel_mm_s2=float(accel_mm_s2),
        ) + float(POSITION_WAIT_BUFFER_S)
    if not math.isclose(intermediate_z, target_z, abs_tol=1e-9):
        total_s += estimate_move_time_s(
            distance_mm=abs(target_z - intermediate_z),
            feedrate_mm_min=float(slow_feedrate),
            accel_mm_s2=float(accel_mm_s2),
        ) + float(POSITION_WAIT_BUFFER_S)
    return total_s


def estimate_move_to_run_start_pose_time_s(
    current_pos: Dict[str, float],
    fixed_y: float,
    fixed_z: float,
    fixed_b: float,
    fixed_c_deg: float,
    return_feedrate: float,
    z_lift_feedrate: float,
    rotation_feedrate: float,
    position_accel_mm_s2: float,
    z_lift_mm: float,
) -> float:
    current_y = float(current_pos.get(ROBOT_STAGE_Y_AXIS_NAME, fixed_y))
    current_z = float(current_pos.get(ROBOT_STAGE_Z_AXIS_NAME, z_lift_mm))
    current_b = float(current_pos.get(ROBOT_REAR_AXIS_NAME, fixed_b))
    current_c = float(current_pos.get(ROBOT_ROTATION_AXIS_NAME, fixed_c_deg))

    total_s = 0.0
    need_lifted_reposition = (
        (not math.isclose(current_y, float(fixed_y), abs_tol=1e-9))
        or (not math.isclose(current_b, float(fixed_b), abs_tol=1e-9))
        or (not math.isclose(current_c, float(fixed_c_deg), abs_tol=1e-9))
    )

    estimated_pos = dict(current_pos)
    if need_lifted_reposition and not math.isclose(current_z, float(z_lift_mm), abs_tol=1e-9):
        total_s += estimate_absolute_move_time_s(
            current_pos=estimated_pos,
            feedrate=float(z_lift_feedrate),
            accel_mm_s2=float(position_accel_mm_s2),
            axes_targets={ROBOT_STAGE_Z_AXIS_NAME: float(z_lift_mm)},
        ) + float(POSITION_WAIT_BUFFER_S)
        estimated_pos[ROBOT_STAGE_Z_AXIS_NAME] = float(z_lift_mm)

    if need_lifted_reposition:
        if not math.isclose(float(estimated_pos.get(ROBOT_STAGE_Y_AXIS_NAME, fixed_y)), float(fixed_y), abs_tol=1e-9):
            total_s += estimate_absolute_move_time_s(
                current_pos=estimated_pos,
                feedrate=float(return_feedrate),
                accel_mm_s2=float(position_accel_mm_s2),
                axes_targets={ROBOT_STAGE_Y_AXIS_NAME: float(fixed_y)},
            ) + float(POSITION_WAIT_BUFFER_S)
            estimated_pos[ROBOT_STAGE_Y_AXIS_NAME] = float(fixed_y)

        if not math.isclose(float(estimated_pos.get(ROBOT_ROTATION_AXIS_NAME, fixed_c_deg)), float(fixed_c_deg), abs_tol=1e-9):
            total_s += estimate_absolute_move_time_s(
                current_pos=estimated_pos,
                feedrate=float(rotation_feedrate),
                accel_mm_s2=float(position_accel_mm_s2),
                axes_targets={ROBOT_ROTATION_AXIS_NAME: float(fixed_c_deg)},
            ) + float(POSITION_WAIT_BUFFER_S)
            estimated_pos[ROBOT_ROTATION_AXIS_NAME] = float(fixed_c_deg)

        if not math.isclose(float(estimated_pos.get(ROBOT_REAR_AXIS_NAME, fixed_b)), float(fixed_b), abs_tol=1e-9):
            total_s += estimate_absolute_move_time_s(
                current_pos=estimated_pos,
                feedrate=min(float(return_feedrate), float(MAX_B_FEEDRATE_MM_MIN)),
                accel_mm_s2=float(position_accel_mm_s2),
                axes_targets={ROBOT_REAR_AXIS_NAME: float(fixed_b)},
            ) + float(POSITION_WAIT_BUFFER_S)
            estimated_pos[ROBOT_REAR_AXIS_NAME] = float(fixed_b)

    total_s += estimate_descend_z_with_final_approach_time_s(
        current_z=float(estimated_pos.get(ROBOT_STAGE_Z_AXIS_NAME, z_lift_mm)),
        target_z=float(fixed_z),
        fast_feedrate=float(z_lift_feedrate),
        slow_feedrate=float(Z_FINAL_APPROACH_FEEDRATE),
        slow_distance_mm=float(Z_FINAL_APPROACH_DISTANCE_MM),
        accel_mm_s2=float(position_accel_mm_s2),
    )
    return total_s


def estimate_lift_return_rotate_and_lower_time_s(
    current_pos: Dict[str, float],
    fixed_y: float,
    acquisition_z: float,
    z_lift: float,
    z_lift_feedrate: float,
    return_feedrate: float,
    rotation_feedrate: float,
    position_accel_mm_s2: float,
    target_c_deg: Optional[float],
) -> float:
    total_s = estimate_absolute_move_time_s(
        current_pos=current_pos,
        feedrate=float(z_lift_feedrate),
        accel_mm_s2=float(position_accel_mm_s2),
        axes_targets={ROBOT_STAGE_Z_AXIS_NAME: float(z_lift)},
    )
    total_s += float(POSITION_WAIT_BUFFER_S)

    estimated_pos = dict(current_pos)
    estimated_pos[ROBOT_STAGE_Z_AXIS_NAME] = float(z_lift)

    if not math.isclose(float(estimated_pos.get(ROBOT_STAGE_Y_AXIS_NAME, fixed_y)), float(fixed_y), abs_tol=1e-9):
        total_s += estimate_absolute_move_time_s(
            current_pos=estimated_pos,
            feedrate=float(return_feedrate),
            accel_mm_s2=float(position_accel_mm_s2),
            axes_targets={ROBOT_STAGE_Y_AXIS_NAME: float(fixed_y)},
        ) + float(POSITION_WAIT_BUFFER_S)
        estimated_pos[ROBOT_STAGE_Y_AXIS_NAME] = float(fixed_y)

    if target_c_deg is not None:
        current_c_deg = float(estimated_pos.get(ROBOT_ROTATION_AXIS_NAME, target_c_deg))
        if not math.isclose(current_c_deg, float(target_c_deg), abs_tol=1e-9):
            total_s += estimate_absolute_move_time_s(
                current_pos=estimated_pos,
                feedrate=float(rotation_feedrate),
                accel_mm_s2=float(position_accel_mm_s2),
                axes_targets={ROBOT_ROTATION_AXIS_NAME: float(target_c_deg)},
            ) + float(POSITION_WAIT_BUFFER_S) + float(ORIENTATION_CHANGE_SETTLE_S)
            estimated_pos[ROBOT_ROTATION_AXIS_NAME] = float(target_c_deg)

    total_s += estimate_descend_z_with_final_approach_time_s(
        current_z=float(estimated_pos.get(ROBOT_STAGE_Z_AXIS_NAME, z_lift)),
        target_z=float(acquisition_z),
        fast_feedrate=float(z_lift_feedrate),
        slow_feedrate=float(Z_FINAL_APPROACH_FEEDRATE),
        slow_distance_mm=float(Z_FINAL_APPROACH_DISTANCE_MM),
        accel_mm_s2=float(position_accel_mm_s2),
    )
    return total_s


def estimate_lift_return_rotate_no_lower_time_s(
    current_pos: Dict[str, float],
    fixed_y: float,
    z_lift: float,
    z_lift_feedrate: float,
    return_feedrate: float,
    rotation_feedrate: float,
    position_accel_mm_s2: float,
    target_c_deg: Optional[float],
) -> float:
    total_s = estimate_absolute_move_time_s(
        current_pos=current_pos,
        feedrate=float(z_lift_feedrate),
        accel_mm_s2=float(position_accel_mm_s2),
        axes_targets={ROBOT_STAGE_Z_AXIS_NAME: float(z_lift)},
    )
    total_s += float(POSITION_WAIT_BUFFER_S)

    estimated_pos = dict(current_pos)
    estimated_pos[ROBOT_STAGE_Z_AXIS_NAME] = float(z_lift)

    if not math.isclose(float(estimated_pos.get(ROBOT_STAGE_Y_AXIS_NAME, fixed_y)), float(fixed_y), abs_tol=1e-9):
        total_s += estimate_absolute_move_time_s(
            current_pos=estimated_pos,
            feedrate=float(return_feedrate),
            accel_mm_s2=float(position_accel_mm_s2),
            axes_targets={ROBOT_STAGE_Y_AXIS_NAME: float(fixed_y)},
        ) + float(POSITION_WAIT_BUFFER_S)
        estimated_pos[ROBOT_STAGE_Y_AXIS_NAME] = float(fixed_y)

    if target_c_deg is not None:
        current_c_deg = float(estimated_pos.get(ROBOT_ROTATION_AXIS_NAME, target_c_deg))
        if not math.isclose(current_c_deg, float(target_c_deg), abs_tol=1e-9):
            total_s += estimate_absolute_move_time_s(
                current_pos=estimated_pos,
                feedrate=float(rotation_feedrate),
                accel_mm_s2=float(position_accel_mm_s2),
                axes_targets={ROBOT_ROTATION_AXIS_NAME: float(target_c_deg)},
            ) + float(POSITION_WAIT_BUFFER_S) + float(ORIENTATION_CHANGE_SETTLE_S)
            estimated_pos[ROBOT_ROTATION_AXIS_NAME] = float(target_c_deg)

    return total_s


def estimate_direct_run_time_s(
    speed_mm_min: float,
    y_start_mm: float,
    y_end_mm: float,
    y_accel_mm_s2: float,
) -> float:
    profile = compute_motion_profile(
        distance_mm=abs(float(y_end_mm) - float(y_start_mm)),
        feedrate_mm_min=float(speed_mm_min),
        accel_mm_s2=float(y_accel_mm_s2),
    )
    return (
        float(PREMOVE_SETTLE_S)
        + float(INITIAL_IDLE_BEFORE_MOTION_S)
        + float(profile["t_total_s"])
        + float(MOTION_WAIT_BUFFER_S)
        + float(POSTMOVE_SETTLE_S)
    )


def estimate_two_stage_run_time_s(
    speed_mm_min: float,
    y_start_mm: float,
    y_middle_mm: float,
    y_end_mm: float,
    y_accel_mm_s2: float,
    middle_pause_s: float,
) -> float:
    first_profile = compute_motion_profile(
        distance_mm=abs(float(y_middle_mm) - float(y_start_mm)),
        feedrate_mm_min=float(speed_mm_min),
        accel_mm_s2=float(y_accel_mm_s2),
    )
    second_profile = compute_motion_profile(
        distance_mm=abs(float(y_end_mm) - float(y_middle_mm)),
        feedrate_mm_min=float(speed_mm_min),
        accel_mm_s2=float(y_accel_mm_s2),
    )
    return (
        float(PREMOVE_SETTLE_S)
        + float(INITIAL_IDLE_BEFORE_MOTION_S)
        + float(first_profile["t_total_s"])
        + float(MOTION_WAIT_BUFFER_S)
        + float(POSTMOVE_SETTLE_S)
        + float(middle_pause_s)
        + float(second_profile["t_total_s"])
        + float(MOTION_WAIT_BUFFER_S)
        + float(POSTMOVE_SETTLE_S)
    )


def print_runtime_estimate(label: str, estimated_total_s: float) -> None:
    now = datetime.now()
    eta = now.timestamp() + max(0.0, float(estimated_total_s))
    completion_dt = datetime.fromtimestamp(eta)
    print("\n" + "=" * 80)
    print(f"[ESTIMATE] {label}")
    print(f"[ESTIMATE] Approx. runtime: {format_duration_hms(estimated_total_s)}")
    print(f"[ESTIMATE] Approx. completion time: {completion_dt.strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)


def move_abs(
    cal: CTR_Shadow_Calibration,
    current_pos: Dict[str, float],
    feedrate: float,
    accel_mm_s2: float,
    wait_buffer_s: float = POSITION_WAIT_BUFFER_S,
    **axes_targets: float,
) -> Dict[str, float]:
    cmd = ["G90", "G1"]
    target_pos = dict(current_pos)
    moved = False

    for axis_name, axis_val in axes_targets.items():
        if axis_val is None:
            continue
        cmd.append(f"{axis_name}{float(axis_val):.5f}")
        target_pos[axis_name] = float(axis_val)
        moved = True

    cmd.append(f"F{float(feedrate):.3f}")
    gcode = " ".join(cmd)

    if moved:
        move_time_s = estimate_absolute_move_time_s(
            current_pos=current_pos,
            feedrate=feedrate,
            accel_mm_s2=accel_mm_s2,
            axes_targets=axes_targets,
        )
        print(f"[MOVE] {gcode}")
        cal.rrf.send_code(gcode)
        wait_s = max(0.0, float(move_time_s) + float(wait_buffer_s))
        if wait_s > 0.0:
            print(f"[WAIT] Timed positioning wait {wait_s:.3f} s")
            time.sleep(wait_s)

    return target_pos


def send_abs_move_no_wait(
    cal: CTR_Shadow_Calibration,
    feedrate: float,
    **axes_targets: float,
) -> None:
    cmd = ["G90", "G1"]
    moved = False

    for axis_name, axis_val in axes_targets.items():
        if axis_val is None:
            continue
        cmd.append(f"{axis_name}{float(axis_val):.5f}")
        moved = True

    cmd.append(f"F{float(feedrate):.3f}")
    gcode = " ".join(cmd)

    if moved:
        print(f"[MOVE] {gcode}")
        cal.rrf.send_code(gcode)


def save_capture_log_frame(
    camera: cv2.VideoCapture,
    output_dir: str,
    csv_path: str,
    frame_idx: int,
    run_meta: Dict[str, object],
    segment_name: str,
    segment_index: int,
    phase: str,
    sample_type: str,
    elapsed_s_from_run_start: float,
    elapsed_s_from_segment_start: float,
    speed_mm_min: float,
    profile: Dict[str, float],
    y_start: float,
    y_end: float,
    y_est_mm: float,
) -> str:
    frame_bgr = capture_frame_with_retries(camera)
    elapsed_for_name = max(0.0, float(elapsed_s_from_run_start))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    image_name = (
        f"frame_{frame_idx:06d}_"
        f"dataset-{run_meta['dataset_label']}_"
        f"orientation-{run_meta['orientation_name']}_"
        f"run-{run_meta['run_type']}_"
        f"repeat-{int(run_meta['repeat_idx']):02d}of{int(run_meta['repeat_count']):02d}_"
        f"segment-{segment_name}_"
        f"sample-{sample_type}_"
        f"phase-{phase}_"
        f"t{elapsed_for_name:.6f}s_"
        f"speed{speed_mm_min:.3f}_"
        f"V{float(run_meta['fixed_v_mm']):.3f}_"
        f"Yest{float(y_est_mm):.3f}_"
        f"Z{float(run_meta['fixed_z_mm']):.3f}_"
        f"B{float(run_meta['fixed_b_mm']):.3f}_"
        f"C{float(run_meta['orientation_deg']):.1f}_"
        f"{timestamp}{RAW_IMAGE_EXTENSION}"
    )
    image_path = os.path.join(output_dir, image_name)
    save_compressed_image(frame_bgr, image_path)

    row = {
        "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
        "frame_idx": int(frame_idx),
        "dataset_label": str(run_meta["dataset_label"]),
        "orientation_name": str(run_meta["orientation_name"]),
        "orientation_deg": float(run_meta["orientation_deg"]),
        "run_type": str(run_meta["run_type"]),
        "repeat_idx": int(run_meta["repeat_idx"]),
        "repeat_count": int(run_meta["repeat_count"]),
        "direction": str(run_meta["direction"]),
        "segment_name": str(segment_name),
        "segment_index": int(segment_index),
        "sample_type": sample_type,
        "phase": phase,
        "profile_type": profile["profile_type"],
        "elapsed_s_from_motion_start": float(elapsed_s_from_run_start),
        "elapsed_s_from_run_start": float(elapsed_s_from_run_start),
        "elapsed_s_from_segment_start": float(elapsed_s_from_segment_start),
        "speed_mm_min": float(speed_mm_min),
        "speed_command_mm_s": float(profile["speed_mm_s"]),
        "peak_speed_mm_s": float(profile["v_peak_mm_s"]),
        "accel_mm_s2": float(profile["accel_mm_s2"]),
        "y_start_mm": float(y_start),
        "y_distance_commanded_mm": float(y_end - y_start),
        "y_end_mm": float(y_end),
        "y_estimated_mm": float(y_est_mm),
        "t_accel_s": float(profile["t_accel_s"]),
        "t_ss_s": float(profile["t_ss_s"]),
        "t_total_s": float(profile["t_total_s"]),
        "has_steady_state": bool(profile["has_steady_state"]),
        "v_fixed_mm": float(run_meta["fixed_v_mm"]),
        "z_fixed_mm": float(run_meta["fixed_z_mm"]),
        "b_fixed_mm": float(run_meta["fixed_b_mm"]),
        "c_fixed_deg": float(run_meta["orientation_deg"]),
        "daq_mode": str(run_meta.get("daq_mode", "speed_series")),
        "attack_angle_deg": float(run_meta.get("attack_angle_deg", float("nan"))),
        "b_command_mm": float(run_meta.get("b_command_mm", run_meta.get("fixed_b_mm", float("nan")))),
        "image_file": image_name,
    }
    append_csv_row(csv_path, row)
    return image_name


def capture_idle_frame(
    cal: CTR_Shadow_Calibration,
    output_dir: str,
    csv_path: str,
    frame_idx: int,
    run_meta: Dict[str, object],
    segment_name: str,
    segment_index: int,
    phase: str,
    sample_type: str,
    elapsed_s_from_run_start: float,
    speed_mm_min: float,
    profile: Dict[str, float],
    y_target_mm: float,
) -> int:
    save_capture_log_frame(
        camera=cal.cam,
        output_dir=output_dir,
        csv_path=csv_path,
        frame_idx=frame_idx,
        run_meta=run_meta,
        segment_name=segment_name,
        segment_index=segment_index,
        phase=phase,
        sample_type=sample_type,
        elapsed_s_from_run_start=elapsed_s_from_run_start,
        elapsed_s_from_segment_start=0.0,
        speed_mm_min=speed_mm_min,
        profile=profile,
        y_start=y_target_mm,
        y_end=y_target_mm,
        y_est_mm=y_target_mm,
    )
    return frame_idx + 1


def wait_before_motion(label: str, wait_s: float) -> None:
    if wait_s > 0.0:
        print(f"[WAIT] {label} {wait_s:.3f} s")
        time.sleep(float(wait_s))


def descend_z_with_final_approach(
    cal: CTR_Shadow_Calibration,
    current_pos: Dict[str, float],
    target_z: float,
    fast_feedrate: float,
    slow_feedrate: float,
    slow_distance_mm: float,
    accel_mm_s2: float,
    label: str,
) -> Dict[str, float]:
    current_z = float(current_pos.get(ROBOT_STAGE_Z_AXIS_NAME, target_z))
    target_z = float(target_z)
    slow_distance_mm = max(0.0, float(slow_distance_mm))

    if math.isclose(current_z, target_z, abs_tol=1e-9):
        return current_pos

    if target_z >= current_z:
        print(f"[INFO] {label} {target_z:.3f}")
        return move_abs(
            cal=cal,
            current_pos=current_pos,
            feedrate=fast_feedrate,
            accel_mm_s2=accel_mm_s2,
            **{ROBOT_STAGE_Z_AXIS_NAME: target_z},
        )

    intermediate_z = max(target_z, current_z - slow_distance_mm)
    if not math.isclose(current_z, intermediate_z, abs_tol=1e-9):
        print(f"[INFO] {label} {target_z:.3f} | fast to Z={intermediate_z:.3f}")
        current_pos = move_abs(
            cal=cal,
            current_pos=current_pos,
            feedrate=fast_feedrate,
            accel_mm_s2=accel_mm_s2,
            **{ROBOT_STAGE_Z_AXIS_NAME: intermediate_z},
        )

    if not math.isclose(float(current_pos.get(ROBOT_STAGE_Z_AXIS_NAME, intermediate_z)), target_z, abs_tol=1e-9):
        print(
            f"[INFO] {label} {target_z:.3f} | final {slow_distance_mm:.3f} mm at F{float(slow_feedrate):.3f}"
        )
        current_pos = move_abs(
            cal=cal,
            current_pos=current_pos,
            feedrate=slow_feedrate,
            accel_mm_s2=accel_mm_s2,
            **{ROBOT_STAGE_Z_AXIS_NAME: target_z},
        )

    return current_pos


def capture_motion_segment(
    cal: CTR_Shadow_Calibration,
    current_pos: Dict[str, float],
    output_dir: str,
    csv_path: str,
    frame_idx: int,
    run_meta: Dict[str, object],
    run_start_monotonic: float,
    segment_name: str,
    segment_index: int,
    speed_mm_min: float,
    y_start: float,
    y_end: float,
    y_accel_mm_s2: float,
    capture_max_fps: float,
) -> Tuple[int, Dict[str, float], Dict[str, float]]:
    if cal.cam is None:
        raise RuntimeError("Camera is not connected.")
    if cal.rrf is None:
        raise RuntimeError("Robot is not connected.")

    profile = compute_motion_profile(
        distance_mm=abs(float(y_end) - float(y_start)),
        feedrate_mm_min=float(speed_mm_min),
        accel_mm_s2=float(y_accel_mm_s2),
    )

    print(
        f"[INFO] Segment {segment_name} | speed={speed_mm_min:.3f} mm/min | "
        f"type={profile['profile_type']} | v_cmd={profile['speed_mm_s']:.6f} mm/s | "
        f"t_accel={profile['t_accel_s']:.6f} s | t_ss={profile['t_ss_s']:.6f} s | "
        f"t_total={profile['t_total_s']:.6f} s"
    )
    if not bool(profile["has_steady_state"]):
        print("[WARN] No steady-state region exists for this segment.")

    min_frame_period_s = 0.0 if capture_max_fps <= 0 else 1.0 / float(capture_max_fps)
    last_capture_elapsed_s = -1e9

    send_abs_move_no_wait(
        cal,
        feedrate=speed_mm_min,
        **{ROBOT_STAGE_Y_AXIS_NAME: y_end},
    )
    segment_start_monotonic = time.monotonic()

    while True:
        now = time.monotonic()
        elapsed_segment_s = now - segment_start_monotonic
        if elapsed_segment_s > float(profile["t_total_s"]):
            break

        if min_frame_period_s > 0 and (elapsed_segment_s - last_capture_elapsed_s) < min_frame_period_s:
            remaining_period = min_frame_period_s - (elapsed_segment_s - last_capture_elapsed_s)
            time.sleep(max(0.001, min(remaining_period, 0.02)))
            continue

        elapsed_for_phase = min(elapsed_segment_s, float(profile["t_total_s"]))
        phase = classify_motion_phase(elapsed_for_phase, profile)
        distance_fraction = motion_profile_distance_fraction(elapsed_for_phase, profile)
        y_est_mm = float(y_start) + (float(y_end) - float(y_start)) * distance_fraction
        elapsed_run_s = time.monotonic() - run_start_monotonic

        try:
            save_capture_log_frame(
                camera=cal.cam,
                output_dir=output_dir,
                csv_path=csv_path,
                frame_idx=frame_idx,
                run_meta=run_meta,
                segment_name=segment_name,
                segment_index=segment_index,
                phase=phase,
                sample_type="motion",
                elapsed_s_from_run_start=elapsed_run_s,
                elapsed_s_from_segment_start=elapsed_for_phase,
                speed_mm_min=speed_mm_min,
                profile=profile,
                y_start=y_start,
                y_end=y_end,
                y_est_mm=y_est_mm,
            )
        except Exception as exc:
            print(
                f"[WARN] Frame capture failed during {segment_name} "
                f"at t={elapsed_segment_s:.6f} s: {exc}"
            )
            continue

        frame_idx += 1
        last_capture_elapsed_s = elapsed_segment_s

    elapsed_after_capture_s = time.monotonic() - segment_start_monotonic
    remaining_motion_s = max(
        0.0,
        float(profile["t_total_s"]) - elapsed_after_capture_s + float(MOTION_WAIT_BUFFER_S),
    )
    print(
        f"[INFO] Segment {segment_name} capture-loop elapsed {elapsed_after_capture_s:.3f} s "
        f"over predicted {float(profile['t_total_s']):.3f} s"
    )
    if remaining_motion_s > 0.0:
        print(f"[WAIT] Timed segment completion wait {remaining_motion_s:.3f} s")
        time.sleep(remaining_motion_s)
    if POSTMOVE_SETTLE_S > 0.0:
        print(f"[WAIT] Post-segment settle wait {POSTMOVE_SETTLE_S:.3f} s")
        time.sleep(float(POSTMOVE_SETTLE_S))

    next_pos = dict(current_pos)
    next_pos[ROBOT_STAGE_Y_AXIS_NAME] = float(y_end)
    return frame_idx, profile, next_pos


def move_to_run_start_pose(
    cal: CTR_Shadow_Calibration,
    current_pos: Dict[str, float],
    fixed_x: float,
    fixed_y: float,
    fixed_z: float,
    fixed_b: float,
    fixed_c_deg: float,
    return_feedrate: float,
    z_lift_feedrate: float,
    rotation_feedrate: float,
    position_accel_mm_s2: float,
    z_lift_mm: float,
    label: str,
) -> Dict[str, float]:
    print(f"[INFO] {label}: moving to run-start pose with Y positioned before Z lowers")

    current_y = float(current_pos.get(ROBOT_STAGE_Y_AXIS_NAME, fixed_y))
    current_z = float(current_pos.get(ROBOT_STAGE_Z_AXIS_NAME, z_lift_mm))
    current_b = float(current_pos.get(ROBOT_REAR_AXIS_NAME, fixed_b))
    current_c = float(current_pos.get(ROBOT_ROTATION_AXIS_NAME, fixed_c_deg))

    need_lifted_reposition = (
        (not math.isclose(current_y, float(fixed_y), abs_tol=1e-9))
        or (not math.isclose(current_b, float(fixed_b), abs_tol=1e-9))
        or (not math.isclose(current_c, float(fixed_c_deg), abs_tol=1e-9))
    )

    if need_lifted_reposition and not math.isclose(current_z, float(z_lift_mm), abs_tol=1e-9):
        current_pos = move_abs(
            cal=cal,
            current_pos=current_pos,
            feedrate=z_lift_feedrate,
            accel_mm_s2=position_accel_mm_s2,
            **{ROBOT_STAGE_Z_AXIS_NAME: z_lift_mm},
        )

    if need_lifted_reposition:
        if not math.isclose(float(current_pos.get(ROBOT_STAGE_Y_AXIS_NAME, fixed_y)), float(fixed_y), abs_tol=1e-9):
            print(f"[INFO] {label}: returning Y to {float(fixed_y):.3f} while lifted")
            current_pos = move_abs(
                cal=cal,
                current_pos=current_pos,
                feedrate=return_feedrate,
                accel_mm_s2=position_accel_mm_s2,
                **{ROBOT_STAGE_Y_AXIS_NAME: fixed_y},
            )

        if not math.isclose(float(current_pos.get(ROBOT_ROTATION_AXIS_NAME, fixed_c_deg)), float(fixed_c_deg), abs_tol=1e-9):
            print(f"[INFO] {label}: rotating C to {float(fixed_c_deg):.1f} deg while lifted")
            current_pos = move_abs(
                cal=cal,
                current_pos=current_pos,
                feedrate=rotation_feedrate,
                accel_mm_s2=position_accel_mm_s2,
                **{ROBOT_ROTATION_AXIS_NAME: fixed_c_deg},
            )

        if not math.isclose(float(current_pos.get(ROBOT_REAR_AXIS_NAME, fixed_b)), float(fixed_b), abs_tol=1e-9):
            print(f"[INFO] {label}: setting B to {float(fixed_b):.5f} while lifted")
            current_pos = move_abs(
                cal=cal,
                current_pos=current_pos,
                feedrate=min(float(return_feedrate), float(MAX_B_FEEDRATE_MM_MIN)),
                accel_mm_s2=position_accel_mm_s2,
                **{ROBOT_REAR_AXIS_NAME: fixed_b},
            )

    current_z = float(current_pos.get(ROBOT_STAGE_Z_AXIS_NAME, z_lift_mm))
    if not math.isclose(current_z, float(fixed_z), abs_tol=1e-9):
        current_pos = descend_z_with_final_approach(
            cal=cal,
            current_pos=current_pos,
            target_z=float(fixed_z),
            fast_feedrate=float(z_lift_feedrate),
            slow_feedrate=float(Z_FINAL_APPROACH_FEEDRATE),
            slow_distance_mm=float(Z_FINAL_APPROACH_DISTANCE_MM),
            accel_mm_s2=position_accel_mm_s2,
            label=f"{label}: lowering Z to acquisition position",
        )

    return current_pos


def lift_return_rotate_and_lower(
    cal: CTR_Shadow_Calibration,
    current_pos: Dict[str, float],
    fixed_y: float,
    acquisition_z: float,
    z_lift: float,
    z_lift_feedrate: float,
    return_feedrate: float,
    rotation_feedrate: float,
    position_accel_mm_s2: float,
    target_c_deg: Optional[float],
    label: str,
) -> Dict[str, float]:
    print(f"[INFO] {label}: lifting Z to {z_lift:.3f}")
    current_pos = move_abs(
        cal=cal,
        current_pos=current_pos,
        feedrate=z_lift_feedrate,
        accel_mm_s2=position_accel_mm_s2,
        **{ROBOT_STAGE_Z_AXIS_NAME: z_lift},
    )

    if not math.isclose(float(current_pos.get(ROBOT_STAGE_Y_AXIS_NAME, fixed_y)), float(fixed_y), abs_tol=1e-9):
        print(f"[INFO] {label}: returning Y to {fixed_y:.3f} while lifted")
        current_pos = move_abs(
            cal=cal,
            current_pos=current_pos,
            feedrate=return_feedrate,
            accel_mm_s2=position_accel_mm_s2,
            **{ROBOT_STAGE_Y_AXIS_NAME: fixed_y},
        )

    if target_c_deg is not None:
        current_c_deg = float(current_pos.get(ROBOT_ROTATION_AXIS_NAME, target_c_deg))
        if not math.isclose(current_c_deg, float(target_c_deg), abs_tol=1e-9):
            print(f"[INFO] {label}: rotating C to {float(target_c_deg):.1f} deg")
            current_pos = move_abs(
                cal=cal,
                current_pos=current_pos,
                feedrate=rotation_feedrate,
                accel_mm_s2=position_accel_mm_s2,
                **{ROBOT_ROTATION_AXIS_NAME: target_c_deg},
            )
            if ORIENTATION_CHANGE_SETTLE_S > 0.0:
                print(f"[WAIT] Orientation settle wait {ORIENTATION_CHANGE_SETTLE_S:.3f} s")
                time.sleep(float(ORIENTATION_CHANGE_SETTLE_S))

    current_pos = descend_z_with_final_approach(
        cal=cal,
        current_pos=current_pos,
        target_z=float(acquisition_z),
        fast_feedrate=float(z_lift_feedrate),
        slow_feedrate=float(Z_FINAL_APPROACH_FEEDRATE),
        slow_distance_mm=float(Z_FINAL_APPROACH_DISTANCE_MM),
        accel_mm_s2=position_accel_mm_s2,
        label=f"{label}: lowering Z to acquisition position",
    )
    return current_pos


def lift_return_rotate_no_lower(
    cal: CTR_Shadow_Calibration,
    current_pos: Dict[str, float],
    fixed_y: float,
    z_lift: float,
    z_lift_feedrate: float,
    return_feedrate: float,
    rotation_feedrate: float,
    position_accel_mm_s2: float,
    target_c_deg: Optional[float],
    label: str,
) -> Dict[str, float]:
    print(f"[INFO] {label}: lifting Z to {z_lift:.3f}")
    current_pos = move_abs(
        cal=cal,
        current_pos=current_pos,
        feedrate=z_lift_feedrate,
        accel_mm_s2=position_accel_mm_s2,
        **{ROBOT_STAGE_Z_AXIS_NAME: z_lift},
    )

    if not math.isclose(float(current_pos.get(ROBOT_STAGE_Y_AXIS_NAME, fixed_y)), float(fixed_y), abs_tol=1e-9):
        print(f"[INFO] {label}: returning Y to {fixed_y:.3f} while lifted")
        current_pos = move_abs(
            cal=cal,
            current_pos=current_pos,
            feedrate=return_feedrate,
            accel_mm_s2=position_accel_mm_s2,
            **{ROBOT_STAGE_Y_AXIS_NAME: fixed_y},
        )

    if target_c_deg is not None:
        current_c_deg = float(current_pos.get(ROBOT_ROTATION_AXIS_NAME, target_c_deg))
        if not math.isclose(current_c_deg, float(target_c_deg), abs_tol=1e-9):
            print(f"[INFO] {label}: rotating C to {float(target_c_deg):.1f} deg")
            current_pos = move_abs(
                cal=cal,
                current_pos=current_pos,
                feedrate=rotation_feedrate,
                accel_mm_s2=position_accel_mm_s2,
                **{ROBOT_ROTATION_AXIS_NAME: target_c_deg},
            )
            if ORIENTATION_CHANGE_SETTLE_S > 0.0:
                print(f"[WAIT] Orientation settle wait {ORIENTATION_CHANGE_SETTLE_S:.3f} s")
                time.sleep(float(ORIENTATION_CHANGE_SETTLE_S))

    return current_pos


def build_run_meta(
    orientation_spec: OrientationSpec,
    run_type: str,
    fixed_x: float,
    fixed_z: float,
    fixed_b: float,
    repeat_idx: int = 1,
    repeat_count: int = 1,
    extra: Optional[Dict[str, object]] = None,
) -> Dict[str, object]:
    meta: Dict[str, object] = {
        "dataset_label": orientation_spec.dataset_label,
        "orientation_name": orientation_name(orientation_spec.angle_deg),
        "orientation_deg": float(orientation_spec.angle_deg),
        "run_type": str(run_type),
        "repeat_idx": int(repeat_idx),
        "repeat_count": int(repeat_count),
        "direction": "forward",
        "fixed_v_mm": float(fixed_x),
        "fixed_z_mm": float(fixed_z),
        "fixed_b_mm": float(fixed_b),
        "daq_mode": "speed_series",
        "attack_angle_deg": float("nan"),
        "b_command_mm": float(fixed_b),
    }
    if extra:
        meta.update(extra)
    return meta


def execute_direct_run(
    cal: CTR_Shadow_Calibration,
    current_pos: Dict[str, float],
    output_dir: str,
    csv_path: str,
    speed_mm_min: float,
    y_start_mm: float,
    y_end_mm: float,
    y_accel_mm_s2: float,
    capture_max_fps: float,
    run_meta: Dict[str, object],
) -> Tuple[Dict[str, float], Dict[str, float], int]:
    if cal.cam is None:
        raise RuntimeError("Camera is not connected.")

    _ = capture_frame(cal.cam, warmup_frames=CAMERA_WARMUP_FRAMES)
    if PREMOVE_SETTLE_S > 0.0:
        time.sleep(float(PREMOVE_SETTLE_S))

    frame_idx = 1
    profile = compute_motion_profile(
        distance_mm=abs(float(y_end_mm) - float(y_start_mm)),
        feedrate_mm_min=float(speed_mm_min),
        accel_mm_s2=float(y_accel_mm_s2),
    )
    frame_idx = capture_idle_frame(
        cal=cal,
        output_dir=output_dir,
        csv_path=csv_path,
        frame_idx=frame_idx,
        run_meta=run_meta,
        segment_name="full_sweep",
        segment_index=1,
        phase="pre",
        sample_type="idle_before_motion",
        elapsed_s_from_run_start=0.0,
        speed_mm_min=speed_mm_min,
        profile=profile,
        y_target_mm=y_start_mm,
    )
    wait_before_motion("Initial idle before motion wait", INITIAL_IDLE_BEFORE_MOTION_S)
    run_start_monotonic = time.monotonic()
    frame_idx, profile, current_pos = capture_motion_segment(
        cal=cal,
        current_pos=current_pos,
        output_dir=output_dir,
        csv_path=csv_path,
        frame_idx=frame_idx,
        run_meta=run_meta,
        run_start_monotonic=run_start_monotonic,
        segment_name="full_sweep",
        segment_index=1,
        speed_mm_min=speed_mm_min,
        y_start=y_start_mm,
        y_end=y_end_mm,
        y_accel_mm_s2=y_accel_mm_s2,
        capture_max_fps=capture_max_fps,
    )
    frame_idx = capture_idle_frame(
        cal=cal,
        output_dir=output_dir,
        csv_path=csv_path,
        frame_idx=frame_idx,
        run_meta=run_meta,
        segment_name="full_sweep",
        segment_index=1,
        phase="post",
        sample_type="idle_after_motion",
        elapsed_s_from_run_start=time.monotonic() - run_start_monotonic,
        speed_mm_min=speed_mm_min,
        profile=profile,
        y_target_mm=y_end_mm,
    )
    return current_pos, profile, frame_idx - 1


def execute_two_stage_run(
    cal: CTR_Shadow_Calibration,
    current_pos: Dict[str, float],
    output_dir: str,
    csv_path: str,
    speed_mm_min: float,
    y_start_mm: float,
    y_middle_mm: float,
    y_end_mm: float,
    y_accel_mm_s2: float,
    capture_max_fps: float,
    middle_pause_s: float,
    run_meta: Dict[str, object],
) -> Tuple[Dict[str, float], Dict[str, float], int]:
    if cal.cam is None:
        raise RuntimeError("Camera is not connected.")

    _ = capture_frame(cal.cam, warmup_frames=CAMERA_WARMUP_FRAMES)
    if PREMOVE_SETTLE_S > 0.0:
        time.sleep(float(PREMOVE_SETTLE_S))

    frame_idx = 1
    first_profile = compute_motion_profile(
        distance_mm=abs(float(y_middle_mm) - float(y_start_mm)),
        feedrate_mm_min=float(speed_mm_min),
        accel_mm_s2=float(y_accel_mm_s2),
    )
    frame_idx = capture_idle_frame(
        cal=cal,
        output_dir=output_dir,
        csv_path=csv_path,
        frame_idx=frame_idx,
        run_meta=run_meta,
        segment_name="start_to_mid",
        segment_index=1,
        phase="pre",
        sample_type="idle_before_motion",
        elapsed_s_from_run_start=0.0,
        speed_mm_min=speed_mm_min,
        profile=first_profile,
        y_target_mm=y_start_mm,
    )
    wait_before_motion("Initial idle before motion wait", INITIAL_IDLE_BEFORE_MOTION_S)

    run_start_monotonic = time.monotonic()
    frame_idx, first_profile, current_pos = capture_motion_segment(
        cal=cal,
        current_pos=current_pos,
        output_dir=output_dir,
        csv_path=csv_path,
        frame_idx=frame_idx,
        run_meta=run_meta,
        run_start_monotonic=run_start_monotonic,
        segment_name="start_to_mid",
        segment_index=1,
        speed_mm_min=speed_mm_min,
        y_start=y_start_mm,
        y_end=y_middle_mm,
        y_accel_mm_s2=y_accel_mm_s2,
        capture_max_fps=capture_max_fps,
    )

    if middle_pause_s > 0.0:
        print(f"[WAIT] Middle pause at Y={y_middle_mm:.3f} for {middle_pause_s:.3f} s")
        time.sleep(float(middle_pause_s))
    frame_idx = capture_idle_frame(
        cal=cal,
        output_dir=output_dir,
        csv_path=csv_path,
        frame_idx=frame_idx,
        run_meta=run_meta,
        segment_name="start_to_mid",
        segment_index=1,
        phase="pause",
        sample_type="idle_mid_pause",
        elapsed_s_from_run_start=time.monotonic() - run_start_monotonic,
        speed_mm_min=speed_mm_min,
        profile=first_profile,
        y_target_mm=y_middle_mm,
    )

    second_profile = compute_motion_profile(
        distance_mm=abs(float(y_end_mm) - float(y_middle_mm)),
        feedrate_mm_min=float(speed_mm_min),
        accel_mm_s2=float(y_accel_mm_s2),
    )
    frame_idx, second_profile, current_pos = capture_motion_segment(
        cal=cal,
        current_pos=current_pos,
        output_dir=output_dir,
        csv_path=csv_path,
        frame_idx=frame_idx,
        run_meta=run_meta,
        run_start_monotonic=run_start_monotonic,
        segment_name="mid_to_end",
        segment_index=2,
        speed_mm_min=speed_mm_min,
        y_start=y_middle_mm,
        y_end=y_end_mm,
        y_accel_mm_s2=y_accel_mm_s2,
        capture_max_fps=capture_max_fps,
    )
    frame_idx = capture_idle_frame(
        cal=cal,
        output_dir=output_dir,
        csv_path=csv_path,
        frame_idx=frame_idx,
        run_meta=run_meta,
        segment_name="mid_to_end",
        segment_index=2,
        phase="post",
        sample_type="idle_after_motion",
        elapsed_s_from_run_start=time.monotonic() - run_start_monotonic,
        speed_mm_min=speed_mm_min,
        profile=second_profile,
        y_target_mm=y_end_mm,
    )
    return current_pos, second_profile, frame_idx - 1


def run_speed_series(
    cal: CTR_Shadow_Calibration,
    speeds_mm_min: List[float],
    fixed_x: float,
    fixed_y: float,
    fixed_z: float,
    fixed_b: float,
    fixed_b_values: Sequence[float],
    y_middle_mm: float,
    y_final_mm: float,
    y_accel_mm_s2: float,
    position_accel_mm_s2: float,
    return_feedrate: float,
    z_lift_mm: float,
    z_lift_feedrate: float,
    rotation_feedrate: float,
    middle_pause_s: float,
    capture_max_fps: float,
    orientation_angles_deg: Sequence[float],
    pass_repeats: int,
    run_specs: Sequence[RunSpec],
) -> Dict[str, str]:
    if cal.cam is None:
        raise RuntimeError("Camera is not connected.")
    if cal.rrf is None:
        raise RuntimeError("Robot is not connected.")

    project_dir = cal.calibration_data_folder
    raw_root = os.path.join(project_dir, "raw_image_data_folder")
    processed_root = os.path.join(project_dir, "processed_image_data_folder")

    if RESET_EXISTING_OUTPUTS:
        raw_root = reset_dir(raw_root)
        processed_root = reset_dir(processed_root)
    else:
        raw_root = ensure_dir(raw_root)
        processed_root = ensure_dir(processed_root)

    summary_csv_path = os.path.join(processed_root, "speed_run_summary.csv")
    orientation_specs = build_orientation_specs(orientation_angles_deg)
    fixed_b_values = [float(v) for v in fixed_b_values]
    if not fixed_b_values:
        fixed_b_values = [float(fixed_b)]

    current_pos = {
        ROBOT_FRONT_AXIS_NAME: float(fixed_x),
        ROBOT_STAGE_Y_AXIS_NAME: float(fixed_y),
        ROBOT_STAGE_Z_AXIS_NAME: float(z_lift_mm),
        ROBOT_REAR_AXIS_NAME: float(fixed_b_values[0]),
        ROBOT_ROTATION_AXIS_NAME: float(orientation_specs[0].angle_deg),
    }
    current_pos = move_to_run_start_pose(
        cal=cal,
        current_pos=current_pos,
        fixed_x=fixed_x,
        fixed_y=fixed_y,
        fixed_z=fixed_z,
        fixed_b=fixed_b_values[0],
        fixed_c_deg=orientation_specs[0].angle_deg,
        return_feedrate=return_feedrate,
        z_lift_feedrate=z_lift_feedrate,
        rotation_feedrate=rotation_feedrate,
        position_accel_mm_s2=position_accel_mm_s2,
        z_lift_mm=z_lift_mm,
        label="startup",
    )

    total_passes = len(fixed_b_values) * len(orientation_specs) * len(speeds_mm_min) * len(run_specs) * int(pass_repeats)
    estimated_total_s = 0.0
    estimate_pos = dict(current_pos)
    for fixed_b_value in fixed_b_values:
        for orientation_spec in orientation_specs:
            estimated_total_s += estimate_lift_return_rotate_and_lower_time_s(
                current_pos=estimate_pos,
                fixed_y=fixed_y,
                acquisition_z=fixed_z,
                z_lift=z_lift_mm,
                z_lift_feedrate=z_lift_feedrate,
                return_feedrate=return_feedrate,
                rotation_feedrate=rotation_feedrate,
                position_accel_mm_s2=position_accel_mm_s2,
                target_c_deg=orientation_spec.angle_deg,
            )
            estimate_pos = {
                ROBOT_FRONT_AXIS_NAME: float(fixed_x),
                ROBOT_STAGE_Y_AXIS_NAME: float(fixed_y),
                ROBOT_STAGE_Z_AXIS_NAME: float(fixed_z),
                ROBOT_REAR_AXIS_NAME: float(fixed_b_value),
                ROBOT_ROTATION_AXIS_NAME: float(orientation_spec.angle_deg),
            }
            for speed_mm_min in speeds_mm_min:
                for run_spec in run_specs:
                    for _ in range(1, int(pass_repeats) + 1):
                        estimated_total_s += estimate_move_to_run_start_pose_time_s(
                            current_pos=estimate_pos,
                            fixed_y=fixed_y,
                            fixed_z=fixed_z,
                            fixed_b=fixed_b_value,
                            fixed_c_deg=orientation_spec.angle_deg,
                            return_feedrate=return_feedrate,
                            z_lift_feedrate=z_lift_feedrate,
                            rotation_feedrate=rotation_feedrate,
                            position_accel_mm_s2=position_accel_mm_s2,
                            z_lift_mm=z_lift_mm,
                        )
                        if run_spec.run_type == "direct":
                            estimated_total_s += estimate_direct_run_time_s(
                                speed_mm_min=speed_mm_min,
                                y_start_mm=fixed_y,
                                y_end_mm=y_final_mm,
                                y_accel_mm_s2=y_accel_mm_s2,
                            )
                        else:
                            estimated_total_s += estimate_two_stage_run_time_s(
                                speed_mm_min=speed_mm_min,
                                y_start_mm=fixed_y,
                                y_middle_mm=y_middle_mm,
                                y_end_mm=y_final_mm,
                                y_accel_mm_s2=y_accel_mm_s2,
                                middle_pause_s=middle_pause_s,
                            )
                        estimated_total_s += estimate_lift_return_rotate_and_lower_time_s(
                            current_pos={
                                **estimate_pos,
                                ROBOT_STAGE_Y_AXIS_NAME: float(y_final_mm),
                            },
                            fixed_y=fixed_y,
                            acquisition_z=fixed_z,
                            z_lift=z_lift_mm,
                            z_lift_feedrate=z_lift_feedrate,
                            return_feedrate=return_feedrate,
                            rotation_feedrate=rotation_feedrate,
                            position_accel_mm_s2=position_accel_mm_s2,
                            target_c_deg=orientation_spec.angle_deg,
                        )
                        estimate_pos = {
                            ROBOT_FRONT_AXIS_NAME: float(fixed_x),
                            ROBOT_STAGE_Y_AXIS_NAME: float(fixed_y),
                            ROBOT_STAGE_Z_AXIS_NAME: float(fixed_z),
                            ROBOT_REAR_AXIS_NAME: float(fixed_b_value),
                            ROBOT_ROTATION_AXIS_NAME: float(orientation_spec.angle_deg),
                        }

    print_runtime_estimate(
        label=f"speed_series | {total_passes} total passes",
        estimated_total_s=estimated_total_s,
    )

    run_counter = 0
    for b_idx, fixed_b_value in enumerate(fixed_b_values, start=1):
        b_raw_root = ensure_dir(os.path.join(raw_root, fixed_b_folder_name(fixed_b_value)))
        for orientation_idx, orientation_spec in enumerate(orientation_specs, start=1):
            current_pos = lift_return_rotate_and_lower(
                cal=cal,
                current_pos=current_pos,
                fixed_y=fixed_y,
                acquisition_z=fixed_z,
                z_lift=z_lift_mm,
                z_lift_feedrate=z_lift_feedrate,
                return_feedrate=return_feedrate,
                rotation_feedrate=rotation_feedrate,
                position_accel_mm_s2=position_accel_mm_s2,
                target_c_deg=orientation_spec.angle_deg,
                label=(
                    f"B={fixed_b_value:.5f} orientation "
                    f"{orientation_spec.dataset_label} ({orientation_name(orientation_spec.angle_deg)})"
                ),
            )

            print("\n" + "=" * 80)
            print(
                f"[SERIES] B {b_idx}/{len(fixed_b_values)} | fixed B={fixed_b_value:.5f} | "
                f"Orientation {orientation_idx}/{len(orientation_specs)} | "
                f"{orientation_spec.dataset_label} | C={orientation_spec.angle_deg:.1f} deg"
            )
            print("=" * 80)

            for run_idx, speed_mm_min in enumerate(speeds_mm_min, start=1):
                for run_spec in run_specs:
                    for repeat_idx in range(1, int(pass_repeats) + 1):
                        run_counter += 1
                        print_pass_progress(run_counter, total_passes)
                        folder_name = build_run_folder_name(
                            orientation_spec=orientation_spec,
                            run_spec=run_spec,
                            speed_mm_min=speed_mm_min,
                            repeat_idx=repeat_idx,
                            repeat_count=pass_repeats,
                        )
                        run_output_dir = ensure_dir(os.path.join(b_raw_root, folder_name))
                        capture_log_path = os.path.join(run_output_dir, "capture_log.csv")
                        run_meta = build_run_meta(
                            orientation_spec=orientation_spec,
                            run_type=run_spec.run_type,
                            fixed_x=fixed_x,
                            fixed_z=fixed_z,
                            fixed_b=fixed_b_value,
                            repeat_idx=repeat_idx,
                            repeat_count=pass_repeats,
                            extra={
                                "b_sweep_idx": int(b_idx),
                                "b_sweep_count": int(len(fixed_b_values)),
                            },
                        )

                        current_pos = move_to_run_start_pose(
                            cal=cal,
                            current_pos=current_pos,
                            fixed_x=fixed_x,
                            fixed_y=fixed_y,
                            fixed_z=fixed_z,
                            fixed_b=fixed_b_value,
                            fixed_c_deg=orientation_spec.angle_deg,
                            return_feedrate=return_feedrate,
                            z_lift_feedrate=z_lift_feedrate,
                            rotation_feedrate=rotation_feedrate,
                            position_accel_mm_s2=position_accel_mm_s2,
                            z_lift_mm=z_lift_mm,
                            label=(
                                f"pre-run B={fixed_b_value:.5f} "
                                f"{orientation_spec.dataset_label} {run_spec.run_type} "
                                f"speed {speed_mm_min:.3f} repeat {repeat_idx}/{pass_repeats}"
                            ),
                        )

                        print("\n" + "=" * 80)
                        print(
                            f"[RUN] {run_counter} | B={fixed_b_value:.5f} | "
                            f"{orientation_spec.dataset_label} | {run_spec.run_type} | "
                            f"speed={speed_mm_min:.3f} mm/min | "
                            f"repeat {repeat_idx}/{pass_repeats}"
                        )
                        print("=" * 80)
                        run_start_time = time.time()

                        if run_spec.run_type == "direct":
                            current_pos, profile, frames_saved = execute_direct_run(
                                cal=cal,
                                current_pos=current_pos,
                                output_dir=run_output_dir,
                                csv_path=capture_log_path,
                                speed_mm_min=speed_mm_min,
                                y_start_mm=fixed_y,
                                y_end_mm=y_final_mm,
                                y_accel_mm_s2=y_accel_mm_s2,
                                capture_max_fps=capture_max_fps,
                                run_meta=run_meta,
                            )
                        else:
                            current_pos, profile, frames_saved = execute_two_stage_run(
                                cal=cal,
                                current_pos=current_pos,
                                output_dir=run_output_dir,
                                csv_path=capture_log_path,
                                speed_mm_min=speed_mm_min,
                                y_start_mm=fixed_y,
                                y_middle_mm=y_middle_mm,
                                y_end_mm=y_final_mm,
                                y_accel_mm_s2=y_accel_mm_s2,
                                capture_max_fps=capture_max_fps,
                                middle_pause_s=middle_pause_s,
                                run_meta=run_meta,
                            )

                        current_pos = lift_return_rotate_and_lower(
                            cal=cal,
                            current_pos=current_pos,
                            fixed_y=fixed_y,
                            acquisition_z=fixed_z,
                            z_lift=z_lift_mm,
                            z_lift_feedrate=z_lift_feedrate,
                            return_feedrate=return_feedrate,
                            rotation_feedrate=rotation_feedrate,
                            position_accel_mm_s2=position_accel_mm_s2,
                            target_c_deg=orientation_spec.angle_deg,
                            label=(
                                f"after B={fixed_b_value:.5f} "
                                f"{orientation_spec.dataset_label} {run_spec.run_type} "
                                f"speed {speed_mm_min:.3f} repeat {repeat_idx}/{pass_repeats}"
                            ),
                        )

                        run_elapsed_s = time.time() - run_start_time
                        append_csv_row(
                            summary_csv_path,
                            {
                                "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
                                "run_counter": int(run_counter),
                                "b_sweep_idx": int(b_idx),
                                "b_sweep_count": int(len(fixed_b_values)),
                                "fixed_b_mm": float(fixed_b_value),
                                "run_idx_within_orientation": int(run_idx),
                                "repeat_idx": int(repeat_idx),
                                "repeat_count": int(pass_repeats),
                                "dataset_label": orientation_spec.dataset_label,
                                "orientation_name": orientation_name(orientation_spec.angle_deg),
                                "orientation_deg": float(orientation_spec.angle_deg),
                                "run_type": run_spec.run_type,
                                "direction": "forward",
                                "speed_mm_min": float(speed_mm_min),
                                "speed_command_mm_s": float(profile["speed_mm_s"]),
                                "peak_speed_mm_s": float(profile["v_peak_mm_s"]),
                                "accel_mm_s2": float(profile["accel_mm_s2"]),
                                "profile_type": profile["profile_type"],
                                "has_steady_state": bool(profile["has_steady_state"]),
                                "t_accel_s": float(profile["t_accel_s"]),
                                "t_ss_s": float(profile["t_ss_s"]),
                                "t_total_s": float(profile["t_total_s"]),
                                "y_start_mm": float(fixed_y),
                                "y_middle_mm": float(y_middle_mm),
                                "y_end_mm": float(y_final_mm),
                                "z_acquisition_mm": float(fixed_z),
                                "z_lift_mm": float(z_lift_mm),
                                "middle_pause_s": float(middle_pause_s),
                                "frames_saved": int(frames_saved),
                                "output_folder": run_output_dir,
                                "elapsed_s": float(run_elapsed_s),
                            },
                        )

    summary = {
        "project_dir": project_dir,
        "raw_root": raw_root,
        "processed_root": processed_root,
        "summary_csv_path": summary_csv_path,
    }

    print("\n[INFO] Output summary")
    for key, value in summary.items():
        print(f"  - {key}: {value}")

    return summary



def run_b_attack_sweep(
    cal: CTR_Shadow_Calibration,
    speed_mm_min: float,
    fixed_x: float,
    fixed_y: float,
    fixed_z: float,
    fixed_b: float,
    y_middle_mm: float,
    y_final_mm: float,
    y_accel_mm_s2: float,
    position_accel_mm_s2: float,
    return_feedrate: float,
    z_lift_mm: float,
    z_lift_feedrate: float,
    rotation_feedrate: float,
    middle_pause_s: float,
    capture_max_fps: float,
    orientation_angles_deg: Sequence[float],
    pass_repeats: int,
    attack_start_deg: float,
    attack_stop_deg: float,
    attack_step_deg: float,
    b_curl_calibration_file: str,
    run_specs: Sequence[RunSpec],
) -> Dict[str, str]:
    if cal.cam is None:
        raise RuntimeError("Camera is not connected.")
    if cal.rrf is None:
        raise RuntimeError("Robot is not connected.")

    b_from_attack = load_b_attack_to_b_command_fit(b_curl_calibration_file)
    attack_angles = build_attack_angle_sequence(attack_start_deg, attack_stop_deg, attack_step_deg)
    orientation_specs = build_orientation_specs(orientation_angles_deg)

    project_dir = cal.calibration_data_folder
    raw_root = os.path.join(project_dir, "raw_image_data_folder")
    processed_root = os.path.join(project_dir, "processed_image_data_folder")

    if RESET_EXISTING_OUTPUTS:
        raw_root = reset_dir(raw_root)
        processed_root = reset_dir(processed_root)
    else:
        raw_root = ensure_dir(raw_root)
        processed_root = ensure_dir(processed_root)

    summary_csv_path = os.path.join(processed_root, "b_attack_sweep_run_summary.csv")

    first_b_command = float(b_from_attack(attack_angles[0]))
    current_pos = {
        ROBOT_FRONT_AXIS_NAME: float(fixed_x),
        ROBOT_STAGE_Y_AXIS_NAME: float(fixed_y),
        ROBOT_STAGE_Z_AXIS_NAME: float(z_lift_mm),
        ROBOT_REAR_AXIS_NAME: float(fixed_b),
        ROBOT_ROTATION_AXIS_NAME: float(orientation_specs[0].angle_deg),
    }

    current_pos = move_to_run_start_pose(
        cal=cal,
        current_pos=current_pos,
        fixed_x=fixed_x,
        fixed_y=fixed_y,
        fixed_z=fixed_z,
        fixed_b=first_b_command,
        fixed_c_deg=orientation_specs[0].angle_deg,
        return_feedrate=return_feedrate,
        z_lift_feedrate=z_lift_feedrate,
        rotation_feedrate=rotation_feedrate,
        position_accel_mm_s2=position_accel_mm_s2,
        z_lift_mm=z_lift_mm,
        label="b-attack startup",
    )

    total_passes = len(orientation_specs) * len(attack_angles) * len(run_specs) * int(pass_repeats)
    estimated_total_s = 0.0
    estimate_pos = dict(current_pos)
    for orientation_spec in orientation_specs:
        for attack_angle_deg in attack_angles:
            b_command = float(b_from_attack(float(attack_angle_deg)))
            for run_spec in run_specs:
                for _ in range(1, int(pass_repeats) + 1):
                    estimated_total_s += estimate_move_to_run_start_pose_time_s(
                        current_pos=estimate_pos,
                        fixed_y=fixed_y,
                        fixed_z=fixed_z,
                        fixed_b=b_command,
                        fixed_c_deg=orientation_spec.angle_deg,
                        return_feedrate=return_feedrate,
                        z_lift_feedrate=z_lift_feedrate,
                        rotation_feedrate=rotation_feedrate,
                        position_accel_mm_s2=position_accel_mm_s2,
                        z_lift_mm=z_lift_mm,
                    )
                    if run_spec.run_type == "direct":
                        estimated_total_s += estimate_direct_run_time_s(
                            speed_mm_min=speed_mm_min,
                            y_start_mm=fixed_y,
                            y_end_mm=y_final_mm,
                            y_accel_mm_s2=y_accel_mm_s2,
                        )
                    else:
                        estimated_total_s += estimate_two_stage_run_time_s(
                            speed_mm_min=speed_mm_min,
                            y_start_mm=fixed_y,
                            y_middle_mm=y_middle_mm,
                            y_end_mm=y_final_mm,
                            y_accel_mm_s2=y_accel_mm_s2,
                            middle_pause_s=middle_pause_s,
                        )
                    estimated_total_s += estimate_lift_return_rotate_no_lower_time_s(
                        current_pos={
                            ROBOT_FRONT_AXIS_NAME: float(fixed_x),
                            ROBOT_STAGE_Y_AXIS_NAME: float(y_final_mm),
                            ROBOT_STAGE_Z_AXIS_NAME: float(fixed_z),
                            ROBOT_REAR_AXIS_NAME: float(b_command),
                            ROBOT_ROTATION_AXIS_NAME: float(orientation_spec.angle_deg),
                        },
                        fixed_y=fixed_y,
                        z_lift=z_lift_mm,
                        z_lift_feedrate=z_lift_feedrate,
                        return_feedrate=return_feedrate,
                        rotation_feedrate=rotation_feedrate,
                        position_accel_mm_s2=position_accel_mm_s2,
                        target_c_deg=orientation_spec.angle_deg,
                    )
                    estimate_pos = {
                        ROBOT_FRONT_AXIS_NAME: float(fixed_x),
                        ROBOT_STAGE_Y_AXIS_NAME: float(fixed_y),
                        ROBOT_STAGE_Z_AXIS_NAME: float(fixed_z),
                        ROBOT_REAR_AXIS_NAME: float(b_command),
                        ROBOT_ROTATION_AXIS_NAME: float(orientation_spec.angle_deg),
                    }

    print_runtime_estimate(
        label=f"b_attack_sweep | {total_passes} total passes",
        estimated_total_s=estimated_total_s,
    )

    run_counter = 0
    for orientation_idx, orientation_spec in enumerate(orientation_specs, start=1):
        print("\n" + "=" * 80)
        print(
            f"[B-ATTACK SERIES] Orientation {orientation_idx}/{len(orientation_specs)} | "
            f"{orientation_spec.dataset_label} | C={orientation_spec.angle_deg:.1f} deg | "
            f"Y sweep speed={speed_mm_min:.3f} mm/min"
        )
        print("=" * 80)

        for attack_idx, attack_angle_deg in enumerate(attack_angles, start=1):
            b_command = float(b_from_attack(float(attack_angle_deg)))
            for run_spec in run_specs:
                for repeat_idx in range(1, int(pass_repeats) + 1):
                    run_counter += 1
                    print_pass_progress(run_counter, total_passes)
                    folder_name = build_b_attack_run_folder_name(
                        orientation_spec=orientation_spec,
                        run_spec=run_spec,
                        speed_mm_min=speed_mm_min,
                        attack_angle_deg=attack_angle_deg,
                        b_command=b_command,
                        repeat_idx=repeat_idx,
                        repeat_count=pass_repeats,
                    )
                    run_output_dir = ensure_dir(os.path.join(raw_root, folder_name))
                    capture_log_path = os.path.join(run_output_dir, "capture_log.csv")
                    run_meta = build_run_meta(
                        orientation_spec=orientation_spec,
                        run_type=run_spec.run_type,
                        fixed_x=fixed_x,
                        fixed_z=fixed_z,
                        fixed_b=b_command,
                        repeat_idx=repeat_idx,
                        repeat_count=pass_repeats,
                        extra={
                            "daq_mode": "b_attack_sweep",
                            "attack_angle_deg": float(attack_angle_deg),
                            "b_command_mm": float(b_command),
                            "attack_idx": int(attack_idx),
                            "attack_count": int(len(attack_angles)),
                        },
                    )

                    current_pos = move_to_run_start_pose(
                        cal=cal,
                        current_pos=current_pos,
                        fixed_x=fixed_x,
                        fixed_y=fixed_y,
                        fixed_z=fixed_z,
                        fixed_b=b_command,
                        fixed_c_deg=orientation_spec.angle_deg,
                        return_feedrate=return_feedrate,
                        z_lift_feedrate=z_lift_feedrate,
                        rotation_feedrate=rotation_feedrate,
                        position_accel_mm_s2=position_accel_mm_s2,
                        z_lift_mm=z_lift_mm,
                        label=(
                            f"pre-run b_attack attack={attack_angle_deg:.1f} deg "
                            f"{run_spec.run_type} B={b_command:.5f} repeat {repeat_idx}/{pass_repeats}"
                        ),
                    )

                    print("\n" + "=" * 80)
                    print(
                        f"[B-ATTACK RUN] {run_counter} | C={orientation_spec.angle_deg:.1f} deg | "
                        f"attack={attack_angle_deg:.1f} deg | {run_spec.run_type} | "
                        f"Bcmd={b_command:.5f} | Y sweep speed={speed_mm_min:.3f} mm/min | "
                        f"repeat {repeat_idx}/{pass_repeats}"
                    )
                    print("=" * 80)
                    run_start_time = time.time()

                    if run_spec.run_type == "direct":
                        current_pos, profile, frames_saved = execute_direct_run(
                            cal=cal,
                            current_pos=current_pos,
                            output_dir=run_output_dir,
                            csv_path=capture_log_path,
                            speed_mm_min=speed_mm_min,
                            y_start_mm=fixed_y,
                            y_end_mm=y_final_mm,
                            y_accel_mm_s2=y_accel_mm_s2,
                            capture_max_fps=capture_max_fps,
                            run_meta=run_meta,
                        )
                    else:
                        current_pos, profile, frames_saved = execute_two_stage_run(
                            cal=cal,
                            current_pos=current_pos,
                            output_dir=run_output_dir,
                            csv_path=capture_log_path,
                            speed_mm_min=speed_mm_min,
                            y_start_mm=fixed_y,
                            y_middle_mm=y_middle_mm,
                            y_end_mm=y_final_mm,
                            y_accel_mm_s2=y_accel_mm_s2,
                            capture_max_fps=capture_max_fps,
                            middle_pause_s=middle_pause_s,
                            run_meta=run_meta,
                        )

                    current_pos = lift_return_rotate_no_lower(
                        cal=cal,
                        current_pos=current_pos,
                        fixed_y=fixed_y,
                        z_lift=z_lift_mm,
                        z_lift_feedrate=z_lift_feedrate,
                        return_feedrate=return_feedrate,
                        rotation_feedrate=rotation_feedrate,
                        position_accel_mm_s2=position_accel_mm_s2,
                        target_c_deg=orientation_spec.angle_deg,
                        label=(
                            f"after b_attack attack={attack_angle_deg:.1f} deg "
                            f"{run_spec.run_type} repeat {repeat_idx}/{pass_repeats}"
                        ),
                    )

                    run_elapsed_s = time.time() - run_start_time
                    append_csv_row(
                        summary_csv_path,
                        {
                            "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
                            "daq_mode": "b_attack_sweep",
                            "run_counter": int(run_counter),
                            "attack_idx": int(attack_idx),
                            "attack_count": int(len(attack_angles)),
                            "repeat_idx": int(repeat_idx),
                            "repeat_count": int(pass_repeats),
                            "dataset_label": orientation_spec.dataset_label,
                            "orientation_name": orientation_name(orientation_spec.angle_deg),
                            "orientation_deg": float(orientation_spec.angle_deg),
                            "run_type": run_spec.run_type,
                            "direction": "forward",
                            "speed_mm_min": float(speed_mm_min),
                            "attack_angle_deg": float(attack_angle_deg),
                            "b_command_mm": float(b_command),
                            "speed_command_mm_s": float(profile["speed_mm_s"]),
                            "peak_speed_mm_s": float(profile["v_peak_mm_s"]),
                            "accel_mm_s2": float(profile["accel_mm_s2"]),
                            "profile_type": profile["profile_type"],
                            "has_steady_state": bool(profile["has_steady_state"]),
                            "t_accel_s": float(profile["t_accel_s"]),
                            "t_ss_s": float(profile["t_ss_s"]),
                            "t_total_s": float(profile["t_total_s"]),
                            "y_start_mm": float(fixed_y),
                            "y_middle_mm": float(y_middle_mm),
                            "y_end_mm": float(y_final_mm),
                            "z_acquisition_mm": float(fixed_z),
                            "z_lift_mm": float(z_lift_mm),
                            "middle_pause_s": float(middle_pause_s),
                            "frames_saved": int(frames_saved),
                            "output_folder": run_output_dir,
                            "elapsed_s": float(run_elapsed_s),
                        },
                    )

    summary = {
        "project_dir": project_dir,
        "raw_root": raw_root,
        "processed_root": processed_root,
        "summary_csv_path": summary_csv_path,
    }
    print("\n[INFO] B-attack output summary")
    for key, value in summary.items():
        print(f"  - {key}: {value}")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Acquire deflection image series for forward Y sweeps at multiple bath speeds, "
            "including configurable direct and/or staged 0->Ymid pause ->Yend sweeps. "
            "The script repeats the same runs at the requested C orientations."
        )
    )

    parser.add_argument(
        "--daq-mode",
        type=str,
        choices=["speed_series", "b_attack_sweep"],
        default=DAQ_MODE,
        help="DAQ mode: existing speed_series or new b_attack_sweep (default: speed_series)",
    )
    parser.add_argument(
        "--speeds",
        type=float,
        nargs="+",
        required=False,
        default=None,
        help="List of Y-axis sweep feedrates in mm/min. For b_attack_sweep, the first speed is used as the fixed Y sweep speed unless --attack-speed-mm-min is provided.",
    )
    parser.add_argument(
        "--attack-speed-mm-min",
        type=float,
        default=None,
        help="Fixed Y-axis sweep speed for b_attack_sweep only. This does not change return, Z, C, or B-axis positioning feedrates. Overrides the first value in --speeds.",
    )
    parser.add_argument(
        "--b-curl-calibration-file",
        type=str,
        default=None,
        help="Calibration JSON containing the average PCHIP fit mapping B curl/attack angle to B command. Required for b_attack_sweep.",
    )
    parser.add_argument(
        "--b-attack-start-deg",
        type=float,
        default=B_ATTACK_START_DEG,
        help=f"Start attack angle in degrees for b_attack_sweep (default: {B_ATTACK_START_DEG})",
    )
    parser.add_argument(
        "--b-attack-stop-deg",
        type=float,
        default=B_ATTACK_STOP_DEG,
        help=f"Stop attack angle in degrees for b_attack_sweep, included (default: {B_ATTACK_STOP_DEG})",
    )
    parser.add_argument(
        "--b-attack-step-deg",
        type=float,
        default=B_ATTACK_STEP_DEG,
        help=f"Attack angle step in degrees for b_attack_sweep (default: {B_ATTACK_STEP_DEG})",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=PROJECT_NAME,
        help=f"Project name for the output folder (default: {PROJECT_NAME})",
    )
    parser.add_argument(
        "--fixed-x",
        type=float,
        default=FIXED_X,
        help=f"Fixed V position during capture (default: {FIXED_X})",
    )
    parser.add_argument(
        "--fixed-y",
        type=float,
        default=FIXED_Y,
        help=f"Starting Y position during capture (default: {FIXED_Y})",
    )
    parser.add_argument(
        "--fixed-z",
        type=float,
        default=FIXED_Z,
        help=f"Fixed Z position during capture (default: {FIXED_Z})",
    )
    parser.add_argument(
        "--fixed-b",
        type=float,
        default=FIXED_B,
        help=f"Fixed B position during capture (default: {FIXED_B})",
    )
    parser.add_argument(
        "--fixed-b-values",
        type=float,
        nargs="+",
        default=None,
        help="Optional fixed-B sweep for speed_series. The script runs the full DAQ at each listed B value in order.",
    )
    parser.add_argument(
        "--y-middle-mm",
        type=float,
        default=Y_MIDDLE_MM,
        help=f"Intermediate pause Y position for staged runs (default: {Y_MIDDLE_MM})",
    )
    parser.add_argument(
        "--y-final-mm",
        type=float,
        default=Y_FINAL_MM,
        help=f"Final Y position reached during all runs (default: {Y_FINAL_MM})",
    )
    parser.add_argument(
        "--y-accel-mm-s2",
        type=float,
        default=Y_ACCEL_MM_S2,
        help=f"Y-axis acceleration in mm/s^2 (default: {Y_ACCEL_MM_S2})",
    )
    parser.add_argument(
        "--position-accel-mm-s2",
        type=float,
        default=POSITION_ACCEL_MM_S2,
        help=f"Acceleration used for timed non-capture waits (default: {POSITION_ACCEL_MM_S2})",
    )
    parser.add_argument(
        "--return-feedrate",
        type=float,
        default=RETURN_FEEDRATE,
        help=f"Feedrate used for non-recorded positioning moves (default: {RETURN_FEEDRATE})",
    )
    parser.add_argument(
        "--z-lift-mm",
        type=float,
        default=Z_LIFT_MM,
        help=f"Lifted Z position used for return and rotate moves (default: {Z_LIFT_MM})",
    )
    parser.add_argument(
        "--z-lift-feedrate",
        type=float,
        default=Z_LIFT_FEEDRATE,
        help=f"Feedrate for Z lift/lower moves in mm/min (default: {Z_LIFT_FEEDRATE})",
    )
    parser.add_argument(
        "--rotation-feedrate",
        type=float,
        default=ROTATION_FEEDRATE,
        help=f"Feedrate for C-axis orientation changes in axis-units/min (default: {ROTATION_FEEDRATE})",
    )
    parser.add_argument(
        "--middle-pause-s",
        type=float,
        default=MIDDLE_PAUSE_S,
        help=f"Pause duration at the middle pose during staged runs (default: {MIDDLE_PAUSE_S})",
    )
    parser.add_argument(
        "--capture-max-fps",
        type=float,
        default=CAPTURE_MAX_FPS,
        help=f"Max image save rate during motion (default: {CAPTURE_MAX_FPS})",
    )
    parser.add_argument(
        "--orientation-seq-deg",
        type=float,
        nargs="+",
        default=ORIENTATION_SEQUENCE_DEG,
        help="Orientation sequence in C-axis degrees, e.g. --orientation-seq-deg 0 180 90",
    )
    parser.add_argument(
        "--pass-repeats",
        type=int,
        default=1,
        help="Number of times to repeat each pass before moving to the next condition (default: 1)",
    )
    parser.add_argument(
        "--run-type",
        type=str,
        choices=["direct", "two_stage", "both"],
        default="both",
        help="Select direct-only, two_stage-only, or both run types (default: both)",
    )
    parser.add_argument(
        "--smooth-move-samples",
        type=int,
        default=SMOOTH_MOVE_SAMPLES,
        help="Deprecated compatibility option. Capture uses continuous moves, so this is ignored.",
    )
    parser.add_argument(
        "--inter-command-delay-s",
        type=float,
        default=INTER_COMMAND_DELAY_S,
        help="Deprecated compatibility option. Capture uses continuous moves, so this is ignored.",
    )

    args = parser.parse_args()
    run_specs = selected_run_specs(args.run_type)
    if not run_specs:
        raise ValueError(f"Unsupported --run-type selection: {args.run_type}")
    args.run_specs = run_specs

    if args.daq_mode == "speed_series":
        if not args.speeds:
            raise ValueError("You must provide at least one speed via --speeds for speed_series mode")
        if any(s <= 0 for s in args.speeds):
            raise ValueError("All speeds must be > 0")
        if any(s > MAX_SWEEP_SPEED_MM_MIN for s in args.speeds):
            raise ValueError(f"All speeds must be <= {MAX_SWEEP_SPEED_MM_MIN:g} mm/min")
        if args.fixed_b_values is not None:
            if len(args.fixed_b_values) < 1:
                raise ValueError("--fixed-b-values must contain at least one value")
            args.fixed_b_values = [float(v) for v in args.fixed_b_values]
        else:
            args.fixed_b_values = [float(args.fixed_b)]
    else:
        attack_speed = args.attack_speed_mm_min if args.attack_speed_mm_min is not None else (args.speeds[0] if args.speeds else None)
        if attack_speed is None or attack_speed <= 0:
            raise ValueError("b_attack_sweep requires --attack-speed-mm-min or at least one positive --speeds value for the fixed Y-axis sweep speed")
        args.attack_speed_mm_min = float(attack_speed)
        if args.attack_speed_mm_min > MAX_SWEEP_SPEED_MM_MIN:
            raise ValueError(
                f"b_attack_sweep speed must be <= {MAX_SWEEP_SPEED_MM_MIN:g} mm/min"
            )
        if not args.b_curl_calibration_file:
            raise ValueError("b_attack_sweep requires --b-curl-calibration-file")
        if not os.path.isfile(os.path.abspath(args.b_curl_calibration_file)):
            raise FileNotFoundError(f"B curl calibration file not found: {args.b_curl_calibration_file}")
        _ = build_attack_angle_sequence(args.b_attack_start_deg, args.b_attack_stop_deg, args.b_attack_step_deg)
        if args.fixed_b_values is not None:
            raise ValueError("--fixed-b-values is only supported for --daq-mode speed_series")
    if args.y_final_mm == args.fixed_y:
        raise ValueError("--y-final-mm must be different from --fixed-y")
    if any(run_spec.run_type == "two_stage" for run_spec in run_specs):
        if args.y_middle_mm == args.fixed_y or args.y_middle_mm == args.y_final_mm:
            raise ValueError("--y-middle-mm must differ from both --fixed-y and --y-final-mm when --run-type includes two_stage")
        if not (min(args.fixed_y, args.y_final_mm) <= args.y_middle_mm <= max(args.fixed_y, args.y_final_mm)):
            raise ValueError("--y-middle-mm must lie between --fixed-y and --y-final-mm when --run-type includes two_stage")
    if args.y_accel_mm_s2 <= 0:
        raise ValueError("--y-accel-mm-s2 must be > 0")
    if args.position_accel_mm_s2 <= 0:
        raise ValueError("--position-accel-mm-s2 must be > 0")
    if args.return_feedrate <= 0:
        raise ValueError("--return-feedrate must be > 0")
    if args.z_lift_feedrate <= 0:
        raise ValueError("--z-lift-feedrate must be > 0")
    if args.rotation_feedrate <= 0:
        raise ValueError("--rotation-feedrate must be > 0")
    if args.middle_pause_s < 0:
        raise ValueError("--middle-pause-s must be >= 0")
    if args.capture_max_fps < 0:
        raise ValueError("--capture-max-fps must be >= 0")
    if args.smooth_move_samples < 2:
        raise ValueError("--smooth-move-samples must be >= 2")
    if args.inter_command_delay_s < 0:
        raise ValueError("--inter-command-delay-s must be >= 0")
    if not args.orientation_seq_deg:
        raise ValueError("--orientation-seq-deg must contain at least one angle")
    if args.pass_repeats < 1:
        raise ValueError("--pass-repeats must be >= 1")

    return args


def main() -> None:
    args = parse_args()
    cal = prepare_calibration_object(project_name=args.project_name)

    try:
        if args.daq_mode == "speed_series":
            run_speed_series(
                cal=cal,
                speeds_mm_min=args.speeds,
                fixed_x=args.fixed_x,
                fixed_y=args.fixed_y,
                fixed_z=args.fixed_z,
                fixed_b=args.fixed_b,
                fixed_b_values=args.fixed_b_values,
                y_middle_mm=args.y_middle_mm,
                y_final_mm=args.y_final_mm,
                y_accel_mm_s2=args.y_accel_mm_s2,
                position_accel_mm_s2=args.position_accel_mm_s2,
                return_feedrate=args.return_feedrate,
                z_lift_mm=args.z_lift_mm,
                z_lift_feedrate=args.z_lift_feedrate,
                rotation_feedrate=args.rotation_feedrate,
                middle_pause_s=args.middle_pause_s,
                capture_max_fps=args.capture_max_fps,
                orientation_angles_deg=args.orientation_seq_deg,
                pass_repeats=args.pass_repeats,
                run_specs=args.run_specs,
            )
        else:
            run_b_attack_sweep(
                cal=cal,
                speed_mm_min=float(args.attack_speed_mm_min),
                fixed_x=args.fixed_x,
                fixed_y=args.fixed_y,
                fixed_z=args.fixed_z,
                fixed_b=args.fixed_b,
                y_middle_mm=args.y_middle_mm,
                y_final_mm=args.y_final_mm,
                y_accel_mm_s2=args.y_accel_mm_s2,
                position_accel_mm_s2=args.position_accel_mm_s2,
                return_feedrate=args.return_feedrate,
                z_lift_mm=args.z_lift_mm,
                z_lift_feedrate=args.z_lift_feedrate,
                rotation_feedrate=args.rotation_feedrate,
                middle_pause_s=args.middle_pause_s,
                capture_max_fps=args.capture_max_fps,
                orientation_angles_deg=args.orientation_seq_deg,
                pass_repeats=args.pass_repeats,
                attack_start_deg=args.b_attack_start_deg,
                attack_stop_deg=args.b_attack_stop_deg,
                attack_step_deg=args.b_attack_step_deg,
                b_curl_calibration_file=args.b_curl_calibration_file,
                run_specs=args.run_specs,
            )
    finally:
        try:
            if cal.cam is not None:
                cal.disconnect_camera()
        except Exception as exc:
            print(f"[WARN] Camera disconnect failed: {exc}")


if __name__ == "__main__":
    main()
