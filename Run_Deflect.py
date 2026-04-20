#!/usr/bin/env python3
import argparse
import csv
import inspect
import math
import os
import shutil
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from shadow_calibration import CTR_Shadow_Calibration


# =========================
# DEFAULT USER CONFIGURATION
# =========================
PROJECT_NAME = "Y_Axis_Speed_2026_04_14_short_tube"
ALLOW_EXISTING_PROJECT = True
ADD_DATE_TO_PROJECT_FOLDER = True

CAMERA_PORT = 0
SHOW_CAMERA_PREVIEW = False
CAMERA_WARMUP_FRAMES = 5
RAW_IMAGE_EXTENSION = ".jpg"
RAW_IMAGE_JPEG_QUALITY = 95
RESET_EXISTING_OUTPUTS = False

MANUAL_CROP_ADJUSTMENT = True
USE_CLASS_ANALYSIS_CROP_SETUP = False  # Not needed for pure image capture

ROBOT_FRONT_AXIS_NAME = "V"
ROBOT_STAGE_Y_AXIS_NAME = "Y"
ROBOT_STAGE_Z_AXIS_NAME = "Z"
ROBOT_REAR_AXIS_NAME = "B"

# Fixed robot pose during acquisition
FIXED_X = 105.0
FIXED_Y = 0.0
FIXED_Z = -130.0
FIXED_B = 0.25

# Motion settings
Y_TRAVEL_MM = -65.0
Y_ACCEL_MM_S2 = 500.0
RETURN_FEEDRATE = 1200.0  # mm/min, return motion only, no capture
Z_LIFT_MM = -30.0
Z_LIFT_FEEDRATE = 2000.0
PREMOVE_SETTLE_S = 0.25
POSTMOVE_SETTLE_S = 0.25
POSTMOTION_IDLE_BEFORE_RETRACT_S = 10.0
SMOOTH_MOVE_SAMPLES = 200

INTER_COMMAND_DELAY_S = 0.005
FORWARD_TO_BACKWARD_PAUSE_S = 20.0
BACKWARD_TO_NEXT_RUN_PAUSE_S = 20.0

# Capture settings
CAPTURE_MAX_FPS = 6.0
CAPTURE_RETRY_LIMIT = 10
CAPTURE_RETRY_WAIT_S = 0.2


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


def sanitize_speed_folder_name(speed_mm_min: float) -> str:
    speed_str = f"{float(speed_mm_min):.3f}".rstrip("0").rstrip(".")
    speed_str = speed_str.replace(".", "p")
    return f"speed_{speed_str}_mm_min"


def move_abs(
    cal: CTR_Shadow_Calibration,
    current_pos: Dict[str, float],
    feedrate: float,
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
        move_time_s = estimate_absolute_move_time_s(current_pos, feedrate, axes_targets)
        print(f"[MOVE] {gcode}")
        cal.rrf.send_code(gcode)
        if move_time_s > 0.0:
            print(f"[WAIT] Timed positioning wait {move_time_s:.3f} s")
            time.sleep(move_time_s)

    return target_pos


def estimate_absolute_move_time_s(
    current_pos: Dict[str, float],
    feedrate: float,
    axes_targets: Dict[str, float],
) -> float:
    feed = max(1e-6, float(feedrate))
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

    return 60.0 * max(deltas) / feed


def compute_inter_command_wait_s(est_move_time_s: float, configured_floor_s: float = 0.0) -> float:
    est = max(0.0, float(est_move_time_s))
    floor_s = max(0.0, float(configured_floor_s))
    if est <= 0.0:
        return floor_s
    paced_wait = min(2.0, max(0.05, 0.95 * est))
    return max(floor_s, paced_wait)


def send_abs_move_no_wait(
    cal: CTR_Shadow_Calibration,
    feedrate: float,
    **axes_targets: float,
) -> str:
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

    return gcode


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


def save_capture_log_frame(
    camera: cv2.VideoCapture,
    output_dir: str,
    csv_path: str,
    frame_idx: int,
    direction_label: str,
    phase: str,
    sample_type: str,
    elapsed_s_from_motion_start: float,
    speed_mm_min: float,
    profile: Dict[str, float],
    x_fixed: float,
    y_start: float,
    y_distance_mm: float,
    y_end: float,
    y_est_mm: float,
    z_fixed: float,
    b_fixed: float,
) -> str:
    frame_bgr = capture_frame_with_retries(camera)
    elapsed_for_name = max(0.0, float(elapsed_s_from_motion_start))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")

    image_name = (
        f"frame_{frame_idx:06d}_"
        f"sample-{sample_type}_"
        f"direction-{direction_label}_"
        f"phase-{phase}_"
        f"t{elapsed_for_name:.6f}s_"
        f"speed{speed_mm_min:.3f}_"
        f"V{x_fixed:.3f}_Yest{y_est_mm:.3f}_Z{z_fixed:.3f}_B{b_fixed:.3f}_"
        f"{timestamp}{RAW_IMAGE_EXTENSION}"
    )
    image_path = os.path.join(output_dir, image_name)
    save_compressed_image(frame_bgr, image_path)

    row = {
        "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
        "frame_idx": int(frame_idx),
        "sample_type": sample_type,
        "direction": direction_label,
        "phase": phase,
        "profile_type": profile["profile_type"],
        "elapsed_s_from_motion_start": float(elapsed_s_from_motion_start),
        "speed_mm_min": float(speed_mm_min),
        "speed_command_mm_s": float(profile["speed_mm_s"]),
        "peak_speed_mm_s": float(profile["v_peak_mm_s"]),
        "accel_mm_s2": float(profile["accel_mm_s2"]),
        "y_start_mm": float(y_start),
        "y_distance_commanded_mm": float(y_distance_mm),
        "y_end_mm": float(y_end),
        "y_estimated_mm": float(y_est_mm),
        "t_accel_s": float(profile["t_accel_s"]),
        "t_ss_s": float(profile["t_ss_s"]),
        "t_total_s": float(profile["t_total_s"]),
        "has_steady_state": bool(profile["has_steady_state"]),
        "v_fixed_mm": float(x_fixed),
        "z_fixed_mm": float(z_fixed),
        "b_fixed_mm": float(b_fixed),
        "image_file": image_name,
    }
    append_csv_row(csv_path, row)
    return image_name


def append_csv_row(csv_path: str, row: Dict[str, object]) -> None:
    file_exists = os.path.isfile(csv_path)
    fieldnames = list(row.keys())
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


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
    """
    Returns a trapezoidal or triangular profile for a move starting and ending at rest.

    Output keys:
      speed_mm_s
      accel_mm_s2
      distance_mm
      t_accel_s
      t_ss_s
      t_total_s
      d_accel_mm
      d_ss_mm
      v_peak_mm_s
      has_steady_state
      profile_type  # 'trapezoidal' or 'triangular'
      t_ss_start_s
      t_ss_end_s
      t_decel_start_s
    """
    if distance_mm == 0:
        raise ValueError("distance_mm must be != 0")
    if feedrate_mm_min <= 0:
        raise ValueError("feedrate_mm_min must be > 0")
    if accel_mm_s2 <= 0:
        raise ValueError("accel_mm_s2 must be > 0")

    distance_mm = abs(float(distance_mm))
    accel_mm_s2 = float(accel_mm_s2)
    speed_mm_s = float(feedrate_mm_min) / 60.0

    # Distance needed to accelerate from 0 to commanded speed
    d_accel_mm = (speed_mm_s ** 2) / (2.0 * accel_mm_s2)

    if 2.0 * d_accel_mm < distance_mm:
        # Trapezoidal profile: accel -> steady-state -> decel
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

    # Triangular profile: accel -> decel, never reaches commanded speed
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


def classify_motion_phase(elapsed_s: float, profile: Dict[str, float]) -> str:
    """
    Returns 'accel', 'ss', or 'decel' based on time from motion start.

    For triangular moves there is no steady state, so only 'accel' and 'decel' are returned.
    """
    t_accel_s = float(profile["t_accel_s"])
    t_ss_end_s = float(profile["t_ss_end_s"])
    t_total_s = float(profile["t_total_s"])
    has_ss = bool(profile["has_steady_state"])

    if elapsed_s < 0.0:
        return "pre"

    if has_ss:
        if elapsed_s < t_accel_s:
            return "accel"
        if elapsed_s < t_ss_end_s:
            return "ss"
        if elapsed_s <= t_total_s:
            return "decel"
        return "post"

    # triangular
    if elapsed_s < t_accel_s:
        return "accel"
    if elapsed_s <= t_total_s:
        return "decel"
    return "post"


def motion_profile_distance_fraction(elapsed_s: float, profile: Dict[str, float]) -> float:
    t = min(max(0.0, float(elapsed_s)), float(profile["t_total_s"]))
    distance = float(profile["distance_mm"])
    accel = float(profile["accel_mm_s2"])
    t_accel = float(profile["t_accel_s"])
    t_decel_start = float(profile["t_decel_start_s"])
    v_peak = float(profile["v_peak_mm_s"])

    if distance <= 0.0:
        return 1.0

    if t <= t_accel:
        traveled = 0.5 * accel * t * t
    elif bool(profile["has_steady_state"]) and t <= float(profile["t_ss_end_s"]):
        traveled = float(profile["d_accel_mm"]) + v_peak * (t - t_accel)
    else:
        t_decel = t - t_decel_start
        traveled = float(profile["d_accel_mm"]) + float(profile["d_ss_mm"])
        traveled += v_peak * t_decel - 0.5 * accel * t_decel * t_decel

    return min(1.0, max(0.0, traveled / distance))


# =========================
# CAPTURE
# =========================
def capture_during_y_motion(
    cal: CTR_Shadow_Calibration,
    speed_mm_min: float,
    y_distance_mm: float,
    y_accel_mm_s2: float,
    output_dir: str,
    csv_path: str,
    capture_max_fps: float,
    x_fixed: float,
    y_start: float,
    y_end: float,
    z_fixed: float,
    b_fixed: float,
    smooth_move_samples: int,
    inter_command_delay_s: float,
) -> Tuple[int, Dict[str, float], Dict[str, float]]:
    """
    Starts one continuous Y move and captures frames while it is in progress.
    Each filename includes the phase marker: accel / ss / decel.
    """
    if cal.cam is None:
        raise RuntimeError("Camera is not connected.")
    if cal.rrf is None:
        raise RuntimeError("Robot is not connected.")
    _ = smooth_move_samples, inter_command_delay_s

    profile = compute_motion_profile(
        distance_mm=y_distance_mm,
        feedrate_mm_min=speed_mm_min,
        accel_mm_s2=y_accel_mm_s2,
    )
    direction_label = "forward" if float(y_end) < float(y_start) else "backward"

    min_frame_period_s = 0.0 if capture_max_fps <= 0 else 1.0 / float(capture_max_fps)

    print(
        f"[INFO] Motion profile for speed={speed_mm_min:.3f} mm/min | "
        f"type={profile['profile_type']} | "
        f"v_cmd={profile['speed_mm_s']:.6f} mm/s | "
        f"v_peak={profile['v_peak_mm_s']:.6f} mm/s | "
        f"t_accel={profile['t_accel_s']:.6f} s | "
        f"t_ss={profile['t_ss_s']:.6f} s | "
        f"t_total={profile['t_total_s']:.6f} s"
    )

    if not profile["has_steady_state"]:
        print(
            "[WARN] No steady-state region exists for this speed/distance/acceleration. "
            "Frames will only be labeled 'accel' or 'decel'."
        )

    # Warm camera immediately before the idle baseline and motion.
    _ = capture_frame(cal.cam, warmup_frames=CAMERA_WARMUP_FRAMES)
    time.sleep(float(max(0.0, PREMOVE_SETTLE_S)))

    frame_idx = 1
    capture_errors = 0
    last_capture_elapsed_s = -1e9

    try:
        save_capture_log_frame(
            camera=cal.cam,
            output_dir=output_dir,
            csv_path=csv_path,
            frame_idx=frame_idx,
            direction_label=direction_label,
            phase="pre",
            sample_type="idle_before_motion",
            elapsed_s_from_motion_start=0.0,
            speed_mm_min=speed_mm_min,
            profile=profile,
            x_fixed=x_fixed,
            y_start=y_start,
            y_distance_mm=y_distance_mm,
            y_end=y_end,
            y_est_mm=y_start,
            z_fixed=z_fixed,
            b_fixed=b_fixed,
        )
        print(f"[INFO] Saved idle baseline before {direction_label} pass")
        frame_idx += 1
    except Exception as exc:
        capture_errors += 1
        print(f"[WARN] Idle baseline capture failed before motion: {exc}")

    send_abs_move_no_wait(
        cal,
        feedrate=speed_mm_min,
        **{
            ROBOT_STAGE_Y_AXIS_NAME: y_end,
        },
    )
    motion_start_monotonic = time.monotonic()

    while True:
        now = time.monotonic()
        elapsed_s = now - motion_start_monotonic

        if elapsed_s > float(profile["t_total_s"]):
            break

        if min_frame_period_s > 0 and (elapsed_s - last_capture_elapsed_s) < min_frame_period_s:
            remaining_period = min_frame_period_s - (elapsed_s - last_capture_elapsed_s)
            time.sleep(max(0.001, min(remaining_period, 0.02)))
            continue

        elapsed_for_phase = min(elapsed_s, float(profile["t_total_s"]))
        phase = classify_motion_phase(elapsed_for_phase, profile)

        distance_fraction = motion_profile_distance_fraction(elapsed_for_phase, profile)
        y_est_mm = float(y_start) + (float(y_end) - float(y_start)) * distance_fraction

        try:
            save_capture_log_frame(
                camera=cal.cam,
                output_dir=output_dir,
                csv_path=csv_path,
                frame_idx=frame_idx,
                direction_label=direction_label,
                phase=phase,
                sample_type="motion",
                elapsed_s_from_motion_start=elapsed_for_phase,
                speed_mm_min=speed_mm_min,
                profile=profile,
                x_fixed=x_fixed,
                y_start=y_start,
                y_distance_mm=y_distance_mm,
                y_end=y_end,
                y_est_mm=y_est_mm,
                z_fixed=z_fixed,
                b_fixed=b_fixed,
            )
        except Exception as exc:
            capture_errors += 1
            print(f"[WARN] Frame capture failed during motion at t={elapsed_s:.6f}s")
            continue

        frame_idx += 1
        last_capture_elapsed_s = elapsed_s

    elapsed_after_capture_s = time.monotonic() - motion_start_monotonic
    remaining_motion_s = max(0.0, float(profile["t_total_s"]) - elapsed_after_capture_s)
    if remaining_motion_s > 0.0:
        print(f"[WAIT] Timed pass completion wait {remaining_motion_s:.3f} s")
        time.sleep(remaining_motion_s)
    if POSTMOVE_SETTLE_S > 0:
        print(f"[WAIT] Post-motion settle wait {POSTMOVE_SETTLE_S:.3f} s")
        time.sleep(float(POSTMOVE_SETTLE_S))

    current_pos = {
        ROBOT_STAGE_Y_AXIS_NAME: float(y_end),
        ROBOT_STAGE_Z_AXIS_NAME: float(z_fixed),
    }
    if POSTMOTION_IDLE_BEFORE_RETRACT_S > 0:
        print(
            f"[INFO] Waiting {POSTMOTION_IDLE_BEFORE_RETRACT_S:.3f} s at post-motion idle "
            "position before post capture and Z retract"
        )
        time.sleep(float(POSTMOTION_IDLE_BEFORE_RETRACT_S))

    try:
        save_capture_log_frame(
            camera=cal.cam,
            output_dir=output_dir,
            csv_path=csv_path,
            frame_idx=frame_idx,
            direction_label=direction_label,
            phase="post",
            sample_type="idle_after_motion",
            elapsed_s_from_motion_start=float(profile["t_total_s"]),
            speed_mm_min=speed_mm_min,
            profile=profile,
            x_fixed=x_fixed,
            y_start=y_start,
            y_distance_mm=y_distance_mm,
            y_end=y_end,
            y_est_mm=y_end,
            z_fixed=z_fixed,
            b_fixed=b_fixed,
        )
        print(f"[INFO] Saved post-motion idle frame after {direction_label} pass")
        frame_idx += 1
    except Exception as exc:
        capture_errors += 1
        print(f"[WARN] Post-motion idle capture failed after motion: {exc}")

    frame_count = max(0, frame_idx - 1)

    print(
        f"[INFO] Finished capture run for speed {speed_mm_min:.3f} mm/min | "
        f"saved {frame_count} frames | capture_errors={capture_errors}"
    )

    return frame_count, profile, current_pos


def lift_return_and_lower(
    cal: CTR_Shadow_Calibration,
    current_pos: Dict[str, float],
    y_target: float,
    acquisition_z: float,
    z_lift: float,
    z_lift_feedrate: float,
    y_return_feedrate: float,
    settle_s: float,
    label: str,
) -> Dict[str, float]:
    print(f"[INFO] {label}: lifting Z to {z_lift:.3f} before Y return")
    current_pos = move_abs(
        cal,
        current_pos,
        feedrate=z_lift_feedrate,
        **{
            ROBOT_STAGE_Z_AXIS_NAME: z_lift,
        },
    )

    print(f"[INFO] {label}: returning Y to {y_target:.3f} while lifted")
    current_pos = move_abs(
        cal,
        current_pos,
        feedrate=y_return_feedrate,
        **{
            ROBOT_STAGE_Y_AXIS_NAME: y_target,
        },
    )

    print(f"[INFO] {label}: lowering Z to acquisition position {acquisition_z:.3f}")
    current_pos = move_abs(
        cal,
        current_pos,
        feedrate=z_lift_feedrate,
        **{
            ROBOT_STAGE_Z_AXIS_NAME: acquisition_z,
        },
    )

    if settle_s > 0:
        print(f"[INFO] {label}: pausing {settle_s:.3f} s before next pass")
        time.sleep(float(settle_s))

    return current_pos


def run_speed_series(
    cal: CTR_Shadow_Calibration,
    speeds_mm_min: List[float],
    fixed_x: float,
    fixed_y: float,
    fixed_z: float,
    fixed_b: float,
    y_travel_mm: float,
    y_accel_mm_s2: float,
    return_feedrate: float,
    capture_max_fps: float,
    smooth_move_samples: int,
    inter_command_delay_s: float,
    forward_to_backward_pause_s: float,
    backward_to_next_run_pause_s: float,
    z_lift_mm: float,
    z_lift_feedrate: float,
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

    current_pos = {
        ROBOT_STAGE_Y_AXIS_NAME: 0.0,
        ROBOT_STAGE_Z_AXIS_NAME: z_lift_mm,
    }

    print("\n" + "=" * 80)
    print("[INFO] Moving robot to fixed acquisition pose")
    print("=" * 80)
    current_pos = move_abs(
        cal,
        current_pos,
        feedrate=max(return_feedrate, 600.0),
        **{
            ROBOT_STAGE_Y_AXIS_NAME: fixed_y,
            ROBOT_STAGE_Z_AXIS_NAME: fixed_z,
        },
    )

    y_forward_start = float(fixed_y)
    y_forward_end = float(fixed_y + y_travel_mm)

    for run_idx, speed_mm_min in enumerate(speeds_mm_min, start=1):
        print("\n" + "=" * 80)
        print(f"[RUN] Forward {run_idx}/{len(speeds_mm_min)} | speed={speed_mm_min:.3f} mm/min")
        print("=" * 80)

        speed_folder_name = sanitize_speed_folder_name(speed_mm_min)
        forward_raw_dir = ensure_dir(os.path.join(raw_root, f"{speed_folder_name}_forward"))
        forward_csv_path = os.path.join(forward_raw_dir, "capture_log.csv")

        run_start_time = time.time()

        current_pos = move_abs(
            cal,
            current_pos,
            feedrate=max(return_feedrate, z_lift_feedrate),
            **{
                ROBOT_STAGE_Y_AXIS_NAME: y_forward_start,
                ROBOT_STAGE_Z_AXIS_NAME: fixed_z,
            },
        )

        frames_saved_forward, profile_forward, current_pos = capture_during_y_motion(
            cal=cal,
            speed_mm_min=speed_mm_min,
            y_distance_mm=y_travel_mm,
            y_accel_mm_s2=y_accel_mm_s2,
            output_dir=forward_raw_dir,
            csv_path=forward_csv_path,
            capture_max_fps=capture_max_fps,
            x_fixed=fixed_x,
            y_start=y_forward_start,
            y_end=y_forward_end,
            z_fixed=fixed_z,
            b_fixed=fixed_b,
            smooth_move_samples=smooth_move_samples,
            inter_command_delay_s=inter_command_delay_s,
        )

        current_pos = lift_return_and_lower(
            cal=cal,
            current_pos=current_pos,
            y_target=y_forward_start,
            acquisition_z=fixed_z,
            z_lift=z_lift_mm,
            z_lift_feedrate=z_lift_feedrate,
            y_return_feedrate=return_feedrate,
            settle_s=backward_to_next_run_pause_s,
            label=f"after forward speed {speed_mm_min:.3f}",
        )

        run_elapsed_s = time.time() - run_start_time
        summary_row_forward = {
            "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
            "run_idx": run_idx,
            "direction": "forward",
            "speed_mm_min": float(speed_mm_min),
            "speed_command_mm_s": float(profile_forward["speed_mm_s"]),
            "peak_speed_mm_s": float(profile_forward["v_peak_mm_s"]),
            "accel_mm_s2": float(profile_forward["accel_mm_s2"]),
            "profile_type": profile_forward["profile_type"],
            "has_steady_state": bool(profile_forward["has_steady_state"]),
            "t_accel_s": float(profile_forward["t_accel_s"]),
            "t_ss_s": float(profile_forward["t_ss_s"]),
            "t_total_s": float(profile_forward["t_total_s"]),
            "y_start_mm": float(y_forward_start),
            "y_travel_mm": float(y_travel_mm),
            "y_end_mm": float(y_forward_end),
            "z_acquisition_mm": float(fixed_z),
            "z_lift_mm": float(z_lift_mm),
            "frames_saved": int(frames_saved_forward),
            "output_folder": forward_raw_dir,
            "elapsed_s": float(run_elapsed_s),
        }
        append_csv_row(summary_csv_path, summary_row_forward)

    print("\n" + "=" * 80)
    print("[INFO] Forward passes complete. Preparing backward pass series.")
    print("=" * 80)

    current_pos = lift_return_and_lower(
        cal=cal,
        current_pos=current_pos,
        y_target=y_forward_end,
        acquisition_z=fixed_z,
        z_lift=z_lift_mm,
        z_lift_feedrate=z_lift_feedrate,
        y_return_feedrate=return_feedrate,
        settle_s=forward_to_backward_pause_s,
        label="between forward and backward series",
    )

    for run_idx, speed_mm_min in enumerate(speeds_mm_min, start=1):
        print("\n" + "=" * 80)
        print(f"[RUN] Backward {run_idx}/{len(speeds_mm_min)} | speed={speed_mm_min:.3f} mm/min")
        print("=" * 80)

        speed_folder_name = sanitize_speed_folder_name(speed_mm_min)
        backward_raw_dir = ensure_dir(os.path.join(raw_root, f"{speed_folder_name}_backward"))
        backward_csv_path = os.path.join(backward_raw_dir, "capture_log.csv")

        run_start_time = time.time()

        current_pos = move_abs(
            cal,
            current_pos,
            feedrate=max(return_feedrate, z_lift_feedrate),
            **{
                ROBOT_STAGE_Y_AXIS_NAME: y_forward_end,
                ROBOT_STAGE_Z_AXIS_NAME: fixed_z,
            },
        )

        frames_saved_backward, profile_backward, current_pos = capture_during_y_motion(
            cal=cal,
            speed_mm_min=speed_mm_min,
            y_distance_mm=abs(y_travel_mm),
            y_accel_mm_s2=y_accel_mm_s2,
            output_dir=backward_raw_dir,
            csv_path=backward_csv_path,
            capture_max_fps=capture_max_fps,
            x_fixed=fixed_x,
            y_start=y_forward_end,
            y_end=y_forward_start,
            z_fixed=fixed_z,
            b_fixed=fixed_b,
            smooth_move_samples=smooth_move_samples,
            inter_command_delay_s=inter_command_delay_s,
        )

        current_pos = lift_return_and_lower(
            cal=cal,
            current_pos=current_pos,
            y_target=y_forward_end,
            acquisition_z=fixed_z,
            z_lift=z_lift_mm,
            z_lift_feedrate=z_lift_feedrate,
            y_return_feedrate=return_feedrate,
            settle_s=backward_to_next_run_pause_s,
            label=f"after backward speed {speed_mm_min:.3f}",
        )

        run_elapsed_s = time.time() - run_start_time
        summary_row_backward = {
            "timestamp_iso": datetime.now().isoformat(timespec="milliseconds"),
            "run_idx": run_idx,
            "direction": "backward",
            "speed_mm_min": float(speed_mm_min),
            "speed_command_mm_s": float(profile_backward["speed_mm_s"]),
            "peak_speed_mm_s": float(profile_backward["v_peak_mm_s"]),
            "accel_mm_s2": float(profile_backward["accel_mm_s2"]),
            "profile_type": profile_backward["profile_type"],
            "has_steady_state": bool(profile_backward["has_steady_state"]),
            "t_accel_s": float(profile_backward["t_accel_s"]),
            "t_ss_s": float(profile_backward["t_ss_s"]),
            "t_total_s": float(profile_backward["t_total_s"]),
            "y_start_mm": float(y_forward_end),
            "y_travel_mm": float(abs(y_travel_mm)),
            "y_end_mm": float(y_forward_start),
            "z_acquisition_mm": float(fixed_z),
            "z_lift_mm": float(z_lift_mm),
            "frames_saved": int(frames_saved_backward),
            "output_folder": backward_raw_dir,
            "elapsed_s": float(run_elapsed_s),
        }
        append_csv_row(summary_csv_path, summary_row_backward)

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move the robot with V fixed and sweep Y with one continuous absolute move, "
            "capturing images during motion and labeling them by accel, ss, or decel "
            "timing phase."
        )
    )

    parser.add_argument(
        "--speeds",
        type=float,
        nargs="+",
        required=True,
        help="List of Y-axis feedrates in mm/min, e.g. --speeds 300 600 900",
    )
    parser.add_argument(
        "--project-name",
        type=str,
        default=PROJECT_NAME,
        help=f"Project name for the output folder (default: {PROJECT_NAME})",
    )
    parser.add_argument(
        "--y-travel-mm",
        type=float,
        default=Y_TRAVEL_MM,
        help=f"Distance to move along Y during each run (default: {Y_TRAVEL_MM})",
    )
    parser.add_argument(
        "--y-accel-mm-s2",
        type=float,
        default=Y_ACCEL_MM_S2,
        help=f"Y-axis acceleration in mm/s^2 (default: {Y_ACCEL_MM_S2})",
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
        "--return-feedrate",
        type=float,
        default=RETURN_FEEDRATE,
        help=f"Feedrate used for non-recorded positioning moves between runs (default: {RETURN_FEEDRATE})",
    )
    parser.add_argument(
        "--z-lift-mm",
        type=float,
        default=Z_LIFT_MM,
        help=f"Lifted Z position used for non-recorded Y returns (default: {Z_LIFT_MM})",
    )
    parser.add_argument(
        "--z-lift-feedrate",
        type=float,
        default=Z_LIFT_FEEDRATE,
        help=f"Feedrate for Z lift/lower moves in mm/min (default: {Z_LIFT_FEEDRATE})",
    )
    parser.add_argument(
        "--capture-max-fps",
        type=float,
        default=CAPTURE_MAX_FPS,
        help=f"Max image save rate during motion (default: {CAPTURE_MAX_FPS})",
    )
    parser.add_argument(
        "--smooth-move-samples",
        type=int,
        default=SMOOTH_MOVE_SAMPLES,
        help="Deprecated compatibility option. Capture now uses one continuous move, so this is ignored.",
    )
    parser.add_argument(
        "--inter-command-delay-s",
        type=float,
        default=INTER_COMMAND_DELAY_S,
        help="Deprecated compatibility option. Capture now uses one continuous move, so this is ignored.",
    )
    parser.add_argument(
        "--forward-to-backward-pause-s",
        type=float,
        default=FORWARD_TO_BACKWARD_PAUSE_S,
        help=f"Pause after the forward pass before starting the backward pass (default: {FORWARD_TO_BACKWARD_PAUSE_S})",
    )
    parser.add_argument(
        "--backward-to-next-run-pause-s",
        type=float,
        default=BACKWARD_TO_NEXT_RUN_PAUSE_S,
        help=f"Pause after the backward pass before the next run (default: {BACKWARD_TO_NEXT_RUN_PAUSE_S})",
    )

    args = parser.parse_args()

    if not args.speeds:
        raise ValueError("You must provide at least one speed via --speeds")
    if any(s <= 0 for s in args.speeds):
        raise ValueError("All speeds must be > 0")
    if args.y_travel_mm == 0:
        raise ValueError("--y-travel-mm must be != 0")
    if args.y_accel_mm_s2 <= 0:
        raise ValueError("--y-accel-mm-s2 must be > 0")
    if args.capture_max_fps < 0:
        raise ValueError("--capture-max-fps must be >= 0")
    if args.z_lift_feedrate <= 0:
        raise ValueError("--z-lift-feedrate must be > 0")
    if args.smooth_move_samples < 2:
        raise ValueError("--smooth-move-samples must be >= 2")
    if args.inter_command_delay_s < 0:
        raise ValueError("--inter-command-delay-s must be >= 0")
    if args.forward_to_backward_pause_s < 0:
        raise ValueError("--forward-to-backward-pause-s must be >= 0")
    if args.backward_to_next_run_pause_s < 0:
        raise ValueError("--backward-to-next-run-pause-s must be >= 0")

    return args


def main() -> None:
    args = parse_args()
    cal = prepare_calibration_object(project_name=args.project_name)

    try:
        run_speed_series(
            cal=cal,
            speeds_mm_min=args.speeds,
            fixed_x=args.fixed_x,
            fixed_y=args.fixed_y,
            fixed_z=args.fixed_z,
            fixed_b=args.fixed_b,
            y_travel_mm=args.y_travel_mm,
            y_accel_mm_s2=args.y_accel_mm_s2,
            return_feedrate=args.return_feedrate,
            capture_max_fps=args.capture_max_fps,
            smooth_move_samples=args.smooth_move_samples,
            inter_command_delay_s=args.inter_command_delay_s,
            forward_to_backward_pause_s=args.forward_to_backward_pause_s,
            backward_to_next_run_pause_s=args.backward_to_next_run_pause_s,
            z_lift_mm=args.z_lift_mm,
            z_lift_feedrate=args.z_lift_feedrate,
        )
    finally:
        try:
            if cal.cam is not None:
                cal.disconnect_camera()
        except Exception as exc:
            print(f"[WARN] Camera disconnect failed: {exc}")


if __name__ == "__main__":
    main()
