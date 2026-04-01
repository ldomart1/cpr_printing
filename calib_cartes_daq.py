#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fixed-orientation grid acquisition script.

What it does:
- Loads the calibration JSON
- Converts requested fixed tip angles (0, 90, 180 deg) into B motor values
  using the calibration tip-angle polynomial
- Clamps requested tip angles to the nearest available calibrated value if needed
- Holds C fixed at 0 deg for the whole run
- For each requested fixed tip-angle pass:
    * moves B once to the chosen fixed value
    * scans a grid of tip positions in gantry/world coordinates
    * captures one image at every grid point
- Saves images to:
    <project folder>/raw_image_data_folder/

Grid:
- X tip positions: 60, 61, 62, ..., 140
- Z tip positions: 150, 151, 152, ..., 195
- Y tip position: constant, user-configurable via --point-y

Important:
- This script assumes the calibration JSON contains tip_angle_coeffs.
- C is fixed at 0 for all captures.
- B is fixed within each pass.
- If a requested tip angle (0, 90, 180) is outside calibration, the nearest
  available calibrated angle is used.
- If your calibration uses opposite sign for r/z, keep --flip-rz-sign enabled.
"""

import argparse
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, List, Optional, Tuple

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
DEFAULT_PROJECT_NAME = "Fixed_Orientation_Grid_Run"
DEFAULT_ALLOW_EXISTING = True
DEFAULT_ADD_DATE = True

DEFAULT_POINT_Y = 52.0

DEFAULT_X_START = 85.0
DEFAULT_X_END = 115.0
DEFAULT_X_STEP = 5.0

DEFAULT_Z_START = -140.0
DEFAULT_Z_END = -110.0
DEFAULT_Z_STEP = 5.0

DEFAULT_FIXED_C = 0.0
DEFAULT_REQUESTED_TIP_ANGLES_DEG = [0.0, 90.0, 180.0]

DEFAULT_TRAVEL_FEED = 2000.0
DEFAULT_SCAN_FEED = 2000.0

DEFAULT_START_X = 85.0
DEFAULT_START_Y = 52.0
DEFAULT_START_Z = -140.0
DEFAULT_START_B = 0.0
DEFAULT_START_C = 0.0

DEFAULT_END_X = 75.0
DEFAULT_END_Y = 52.0
DEFAULT_END_Z = -160.0
DEFAULT_END_B = 0.0
DEFAULT_END_C = 0.0

DEFAULT_SAFE_APPROACH_Z = -140.0

DEFAULT_DWELL_BEFORE_MS = 0.2
DEFAULT_DWELL_AFTER_MS = 0.2
DEFAULT_CAPTURE_SETTLE_S = 0.6
DEFAULT_TRAVEL_MOVE_SETTLE_S = 0.2
DEFAULT_INTER_GRID_WAIT_S = 5.0
DEFAULT_PRE_CAPTURE_BUFFER_S = 0.2

DEFAULT_BBOX_X_MIN = 0.0
DEFAULT_BBOX_X_MAX = 200.0
DEFAULT_BBOX_Y_MIN = -20.0
DEFAULT_BBOX_Y_MAX = 200.0
DEFAULT_BBOX_Z_MIN = -200.0
DEFAULT_BBOX_Z_MAX = 200.0

DEFAULT_MANUAL_FOCUS = True
DEFAULT_MANUAL_FOCUS_VAL = 60
DEFAULT_CAMERA_WIDTH = 3840
DEFAULT_CAMERA_HEIGHT = 2160
DEFAULT_CAMERA_FLUSH_FRAMES = 1

DEFAULT_CUSTOM_INV_SAMPLES = 20000
DEFAULT_FLIP_RZ_SIGN = True

OFFPLANE_SIGN = -1.0


# =========================
# Data structures
# =========================

@dataclass
class Calibration:
    r_model: Any
    z_model: Any
    y_off_model: Optional[Any]
    tip_angle_model: Optional[Any]
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
class GridPoint:
    tip_x: float
    tip_y: float
    tip_z: float
    stage_x: float
    stage_y: float
    stage_z: float
    row_index: int
    col_index: int
    capture_index: int


@dataclass
class FixedAnglePass:
    requested_tip_angle_deg: float
    used_tip_angle_deg: float
    b_cmd: float
    c_cmd: float
    grid_points: List[GridPoint]


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
        or (data.get("cubic_coefficients") or {}).get("default_phase")
        or "pull"
    ) or "pull"

    return phase_models, default_phase


def _select_fit_model(
    cal: Calibration,
    model_name: str,
    motion_phase: Optional[str] = None,
) -> Any:
    phase_name = _normalize_motion_phase_name(motion_phase) or cal.default_motion_phase
    if phase_name in cal.phase_models:
        model = cal.phase_models[phase_name].get(model_name)
        if model is not None:
            return model

    fallback_attr = {
        "r": cal.r_model,
        "z": cal.z_model,
        "offplane_y": cal.y_off_model,
        "tip_angle": cal.tip_angle_model,
    }.get(model_name)
    return fallback_attr


def load_calibration(json_path: str) -> Calibration:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")

    with p.open("r") as f:
        data = json.load(f)

    cubic = data.get("cubic_coefficients", {})
    fit_models = data.get("fit_models", {})
    phase_models, default_motion_phase = _extract_phase_models(data)

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
        "offplane_y",
    )
    tip_angle_model = fit_models.get("tip_angle") or legacy_poly_model(
        cubic.get("tip_angle_coeffs"),
        cubic.get("tip_angle_equation"),
        "tip_angle",
    )

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
        default_motion_phase=default_motion_phase,
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


def eval_r(
    cal: Calibration,
    b: Any,
    flip_rz_sign: bool = False,
    motion_phase: Optional[str] = None,
) -> np.ndarray:
    s = -1.0 if bool(flip_rz_sign) else 1.0
    return s * evaluate_fit_model(_select_fit_model(cal, "r", motion_phase=motion_phase), b)


def eval_z(
    cal: Calibration,
    b: Any,
    flip_rz_sign: bool = False,
    motion_phase: Optional[str] = None,
) -> np.ndarray:
    s = -1.0 if bool(flip_rz_sign) else 1.0
    return s * evaluate_fit_model(_select_fit_model(cal, "z", motion_phase=motion_phase), b)


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
    x = -r * math.cos(c) - y_off * math.sin(c)
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


def inclusive_float_range(start: float, end: float, step: float) -> List[float]:
    if step == 0:
        raise ValueError("Step cannot be zero.")
    step = float(step)
    start = float(start)
    end = float(end)

    vals: List[float] = []
    if step > 0:
        cur = start
        while cur <= end + 1e-9:
            vals.append(round(cur, 10))
            cur += step
    else:
        cur = start
        while cur >= end - 1e-9:
            vals.append(round(cur, 10))
            cur += step
    return vals


def build_tip_angle_inverse_table(
    cal: Calibration,
    num_samples: int = DEFAULT_CUSTOM_INV_SAMPLES,
    motion_phase: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    if _select_fit_model(cal, "tip_angle", motion_phase=motion_phase) is None:
        raise ValueError(
            "This script requires a tip-angle fit model in the calibration JSON."
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


def estimate_move_duration_s(
    start_pose: Tuple[float, float, float, float, float],
    end_pose: Tuple[float, float, float, float, float],
    feedrate_mm_per_min: float,
) -> float:
    if float(feedrate_mm_per_min) <= 0:
        return 0.0

    start = np.asarray(start_pose, dtype=float)
    end = np.asarray(end_pose, dtype=float)
    distance = float(np.linalg.norm(end - start))
    return distance / (float(feedrate_mm_per_min) / 60.0)


# =========================
# Grid planning
# =========================

def generate_fixed_angle_grid_pass(
    cal: Calibration,
    requested_tip_angle_deg: float,
    fixed_c_deg: float,
    tip_y: float,
    x_values: List[float],
    z_values: List[float],
    inverse_samples: int = DEFAULT_CUSTOM_INV_SAMPLES,
    flip_rz_sign: bool = False,
    snake_scan: bool = True,
    motion_phase: Optional[str] = None,
) -> FixedAnglePass:
    angle_table_deg, b_table = build_tip_angle_inverse_table(
        cal=cal,
        num_samples=int(inverse_samples),
        motion_phase=motion_phase,
    )

    b_cmd, used_tip_angle_deg = tip_angle_deg_to_b_clipped(
        requested_tip_angle_deg=requested_tip_angle_deg,
        angle_table_deg=angle_table_deg,
        b_table=b_table,
    )

    c_cmd = clamp_c_bounded(float(fixed_c_deg))

    grid_points: List[GridPoint] = []
    capture_index = 0

    for row_index, z_tip in enumerate(z_values):
        x_row = list(x_values)
        if snake_scan and (row_index % 2 == 1):
            x_row = list(reversed(x_row))

        for col_index, x_tip in enumerate(x_row):
            tip_xyz = np.array([float(x_tip), float(tip_y), float(z_tip)], dtype=float)
            stage_xyz = stage_xyz_for_fixed_tip(
                cal=cal,
                p_tip_xyz=tip_xyz,
                b=b_cmd,
                c_deg=c_cmd,
                flip_rz_sign=flip_rz_sign,
                motion_phase=motion_phase,
            )

            capture_index += 1
            grid_points.append(
                GridPoint(
                    tip_x=float(x_tip),
                    tip_y=float(tip_y),
                    tip_z=float(z_tip),
                    stage_x=float(stage_xyz[0]),
                    stage_y=float(stage_xyz[1]),
                    stage_z=float(stage_xyz[2]),
                    row_index=int(row_index),
                    col_index=int(col_index),
                    capture_index=int(capture_index),
                )
            )

    return FixedAnglePass(
        requested_tip_angle_deg=float(requested_tip_angle_deg),
        used_tip_angle_deg=float(used_tip_angle_deg),
        b_cmd=float(b_cmd),
        c_cmd=float(c_cmd),
        grid_points=grid_points,
    )


def summarize_grid_pass(grid_pass: FixedAnglePass) -> dict:
    if not grid_pass.grid_points:
        return {
            "n_points": 0,
            "stage_x_min": 0.0,
            "stage_x_max": 0.0,
            "stage_y_min": 0.0,
            "stage_y_max": 0.0,
            "stage_z_min": 0.0,
            "stage_z_max": 0.0,
        }

    xs = np.array([p.stage_x for p in grid_pass.grid_points], dtype=float)
    ys = np.array([p.stage_y for p in grid_pass.grid_points], dtype=float)
    zs = np.array([p.stage_z for p in grid_pass.grid_points], dtype=float)

    return {
        "n_points": int(len(grid_pass.grid_points)),
        "stage_x_min": float(np.min(xs)),
        "stage_x_max": float(np.max(xs)),
        "stage_y_min": float(np.min(ys)),
        "stage_y_max": float(np.max(ys)),
        "stage_z_min": float(np.min(zs)),
        "stage_z_max": float(np.max(zs)),
    }


# =========================
# Acquisition runner
# =========================

class FixedOrientationGridRunner:
    """
    Robot + camera execution runner for fixed-B/fixed-C grid scans.
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

        self.raw_folder = os.path.join(self.run_folder, "raw_image_data_folder")
        os.makedirs(self.raw_folder, exist_ok=True)

        self.cam = None
        self.rrf = None
        self.cam_port = None

        print(f"Using run folder: {self.run_folder}")
        print(f"Using raw image folder: {self.raw_folder}")

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
        pass_idx: int,
        requested_tip_angle_deg: float,
        used_tip_angle_deg: float,
        tip_x: float,
        tip_y: float,
        tip_z: float,
        stage_x: float,
        stage_y: float,
        stage_z: float,
        b: float,
        c: float,
        flush_frames: int = 1,
    ) -> Optional[str]:
        if self.cam is None:
            raise RuntimeError("Camera is not connected.")

        print(
            f"Waiting {DEFAULT_PRE_CAPTURE_BUFFER_S:.1f} s before capture "
            "to let large moves settle..."
        )
        time.sleep(DEFAULT_PRE_CAPTURE_BUFFER_S)

        for _ in range(max(0, int(flush_frames))):
            _ = self.cam.read()

        ret, image = self.cam.read()
        if not ret:
            ret, image = self.cam.read()

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = (
            f"{sample_idx:05d}"
            f"_pass{pass_idx:02d}"
            f"_reqA{requested_tip_angle_deg:.1f}"
            f"_useA{used_tip_angle_deg:.3f}"
            f"_tipX{tip_x:.3f}_tipY{tip_y:.3f}_tipZ{tip_z:.3f}"
            f"_stageX{stage_x:.3f}_stageY{stage_y:.3f}_stageZ{stage_z:.3f}"
            f"_B{b:.3f}_C{c:.3f}"
            f"_{timestamp}.png"
        ).replace(" ", "_")

        path = os.path.join(self.raw_folder, filename)
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

    def execute_passes_and_capture(
        self,
        cal: Calibration,
        passes: List[FixedAnglePass],
        start_pose: Tuple[float, float, float, float, float],
        end_pose: Tuple[float, float, float, float, float],
        safe_approach_z: float,
        travel_feed: float,
        scan_feed: float,
        virtual_bbox: dict,
        dwell_before_ms: int = 0,
        dwell_after_ms: int = 0,
        capture_settle_s: float = 0.0,
        travel_move_settle_s: float = 0.0,
        inter_grid_wait_s: float = 0.0,
        camera_flush_frames: int = 1,
    ):
        if self.cam is None:
            raise RuntimeError("Camera is not connected.")
        if self.rrf is None:
            raise RuntimeError("Robot is not connected.")

        bbox_warnings: List[str] = []
        total_images_saved = 0

        print("\n" + "=" * 72)
        print("STARTING FIXED-ORIENTATION GRID ACQUISITION RUN")
        print("=" * 72)

        sx, sy, sz, sb, sc = [float(v) for v in start_pose]
        current_pose = (sx, sy, sz, sb, clamp_c_bounded(sc))

        print("\nSafe startup approach...")
        self.send_absolute_move(
            travel_feed,
            **{
                cal.z_axis: float(safe_approach_z),
                cal.b_axis: sb,
                cal.c_axis: clamp_c_bounded(sc),
            }
        )
        self.wait_for_duet_motion_complete(extra_settle=travel_move_settle_s)

        self.send_absolute_move(
            travel_feed,
            **{
                cal.x_axis: sx,
                cal.y_axis: sy,
                cal.b_axis: sb,
                cal.c_axis: clamp_c_bounded(sc),
            }
        )
        self.wait_for_duet_motion_complete(extra_settle=travel_move_settle_s)

        self.send_absolute_move(
            travel_feed,
            **{
                cal.z_axis: sz,
                cal.b_axis: sb,
                cal.c_axis: clamp_c_bounded(sc),
            }
        )
        self.wait_for_duet_motion_complete(extra_settle=travel_move_settle_s)
        current_pose = (sx, sy, sz, sb, clamp_c_bounded(sc))

        if int(dwell_before_ms) > 0:
            print(f"Dwell before run: {int(dwell_before_ms)} ms")
            time.sleep(float(dwell_before_ms) / 1000.0)

        for pass_idx, pass_plan in enumerate(passes, start=1):
            print("\n" + "-" * 72)
            print(
                f"Pass {pass_idx}/{len(passes)}: "
                f"requested tip angle={pass_plan.requested_tip_angle_deg:.3f} deg, "
                f"used tip angle={pass_plan.used_tip_angle_deg:.3f} deg, "
                f"B={pass_plan.b_cmd:.3f}, C={pass_plan.c_cmd:.3f}"
            )
            print(f"Grid points in this pass: {len(pass_plan.grid_points)}")
            print("-" * 72)

            if not pass_plan.grid_points:
                print("Skipping empty pass.")
                continue

            print("Returning to start pose before pass...")
            self.send_absolute_move(
                travel_feed,
                **{
                    cal.z_axis: float(safe_approach_z),
                    cal.b_axis: current_pose[3],
                    cal.c_axis: current_pose[4],
                }
            )
            self.wait_for_duet_motion_complete(extra_settle=travel_move_settle_s)

            self.send_absolute_move(
                travel_feed,
                **{
                    cal.x_axis: sx,
                    cal.y_axis: sy,
                    cal.b_axis: sb,
                    cal.c_axis: clamp_c_bounded(sc),
                }
            )
            self.wait_for_duet_motion_complete(extra_settle=travel_move_settle_s)

            self.send_absolute_move(
                travel_feed,
                **{
                    cal.z_axis: sz,
                    cal.b_axis: sb,
                    cal.c_axis: clamp_c_bounded(sc),
                }
            )
            self.wait_for_duet_motion_complete(extra_settle=travel_move_settle_s)
            current_pose = (sx, sy, sz, sb, clamp_c_bounded(sc))

            if inter_grid_wait_s > 0:
                print(f"Waiting at start pose before pass: {inter_grid_wait_s:.3f} s")
                time.sleep(inter_grid_wait_s)

            first_pt = pass_plan.grid_points[0]
            x0, y0, z0 = _clamp_stage_xyz_to_bbox(
                first_pt.stage_x, first_pt.stage_y, first_pt.stage_z,
                virtual_bbox,
                f"pass {pass_idx} first grid point",
                bbox_warnings,
            )
            pass_entry_segments = [
                ((current_pose[0], current_pose[1], current_pose[2], current_pose[3], current_pose[4]),
                 (current_pose[0], current_pose[1], float(safe_approach_z), pass_plan.b_cmd, clamp_c_bounded(pass_plan.c_cmd))),
                ((current_pose[0], current_pose[1], float(safe_approach_z), pass_plan.b_cmd, clamp_c_bounded(pass_plan.c_cmd)),
                 (x0, y0, float(safe_approach_z), pass_plan.b_cmd, clamp_c_bounded(pass_plan.c_cmd))),
                ((x0, y0, float(safe_approach_z), pass_plan.b_cmd, clamp_c_bounded(pass_plan.c_cmd)),
                 (x0, y0, z0, pass_plan.b_cmd, clamp_c_bounded(pass_plan.c_cmd))),
            ]
            pass_entry_travel_dwell_s = sum(
                estimate_move_duration_s(seg_start, seg_end, travel_feed)
                for seg_start, seg_end in pass_entry_segments
            )

            print("Moving to pass safe approach with fixed B/C...")
            self.send_absolute_move(
                travel_feed,
                **{
                    cal.z_axis: float(safe_approach_z),
                    cal.b_axis: pass_plan.b_cmd,
                    cal.c_axis: clamp_c_bounded(pass_plan.c_cmd),
                }
            )
            self.wait_for_duet_motion_complete(extra_settle=travel_move_settle_s)

            self.send_absolute_move(
                travel_feed,
                **{
                    cal.x_axis: x0,
                    cal.y_axis: y0,
                    cal.b_axis: pass_plan.b_cmd,
                    cal.c_axis: clamp_c_bounded(pass_plan.c_cmd),
                }
            )
            self.wait_for_duet_motion_complete(extra_settle=travel_move_settle_s)

            self.send_absolute_move(
                travel_feed,
                **{
                    cal.z_axis: z0,
                    cal.b_axis: pass_plan.b_cmd,
                    cal.c_axis: clamp_c_bounded(pass_plan.c_cmd),
                }
            )
            self.wait_for_duet_motion_complete(extra_settle=travel_move_settle_s)

            if pass_entry_travel_dwell_s > 0:
                print(
                    "Waiting for pass entry travel to settle before first capture: "
                    f"{pass_entry_travel_dwell_s:.3f} s"
                )
                time.sleep(pass_entry_travel_dwell_s)

            # Capture the first point only after the initial approach move has fully settled.
            self.wait_for_duet_motion_complete(extra_settle=capture_settle_s)

            total_images_saved += 1
            self.capture_and_save(
                sample_idx=total_images_saved,
                pass_idx=pass_idx,
                requested_tip_angle_deg=pass_plan.requested_tip_angle_deg,
                used_tip_angle_deg=pass_plan.used_tip_angle_deg,
                tip_x=first_pt.tip_x,
                tip_y=first_pt.tip_y,
                tip_z=first_pt.tip_z,
                stage_x=x0,
                stage_y=y0,
                stage_z=z0,
                b=pass_plan.b_cmd,
                c=clamp_c_bounded(pass_plan.c_cmd),
                flush_frames=camera_flush_frames,
            )
            current_pose = (x0, y0, z0, pass_plan.b_cmd, clamp_c_bounded(pass_plan.c_cmd))

            for point_idx, pt in enumerate(pass_plan.grid_points[1:], start=2):
                xg, yg, zg = _clamp_stage_xyz_to_bbox(
                    pt.stage_x, pt.stage_y, pt.stage_z,
                    virtual_bbox,
                    f"pass {pass_idx} point {point_idx}",
                    bbox_warnings,
                )

                self.send_absolute_move(
                    scan_feed,
                    **{
                        cal.x_axis: xg,
                        cal.y_axis: yg,
                        cal.z_axis: zg,
                        cal.b_axis: pass_plan.b_cmd,
                        cal.c_axis: clamp_c_bounded(pass_plan.c_cmd),
                    }
                )
                self.wait_for_duet_motion_complete(extra_settle=capture_settle_s)

                total_images_saved += 1
                self.capture_and_save(
                    sample_idx=total_images_saved,
                    pass_idx=pass_idx,
                    requested_tip_angle_deg=pass_plan.requested_tip_angle_deg,
                    used_tip_angle_deg=pass_plan.used_tip_angle_deg,
                    tip_x=pt.tip_x,
                    tip_y=pt.tip_y,
                    tip_z=pt.tip_z,
                    stage_x=xg,
                    stage_y=yg,
                    stage_z=zg,
                    b=pass_plan.b_cmd,
                    c=clamp_c_bounded(pass_plan.c_cmd),
                    flush_frames=camera_flush_frames,
                )
                current_pose = (xg, yg, zg, pass_plan.b_cmd, clamp_c_bounded(pass_plan.c_cmd))

        if int(dwell_after_ms) > 0:
            print(f"Dwell after run: {int(dwell_after_ms)} ms")
            time.sleep(float(dwell_after_ms) / 1000.0)

        ex, ey, ez, eb, ec = [float(v) for v in end_pose]

        print("\nSafe end move...")
        self.send_absolute_move(
            travel_feed,
            **{
                cal.z_axis: float(safe_approach_z),
                cal.b_axis: eb,
                cal.c_axis: clamp_c_bounded(ec),
            }
        )
        self.wait_for_duet_motion_complete(extra_settle=travel_move_settle_s)

        self.send_absolute_move(
            travel_feed,
            **{
                cal.x_axis: ex,
                cal.y_axis: ey,
                cal.b_axis: eb,
                cal.c_axis: clamp_c_bounded(ec),
            }
        )
        self.wait_for_duet_motion_complete(extra_settle=travel_move_settle_s)

        self.send_absolute_move(
            travel_feed,
            **{
                cal.z_axis: ez,
                cal.b_axis: eb,
                cal.c_axis: clamp_c_bounded(ec),
            }
        )
        self.wait_for_duet_motion_complete(extra_settle=travel_move_settle_s)

        print("\n" + "=" * 72)
        print("RUN COMPLETE")
        print("=" * 72)
        print(f"Images saved: {total_images_saved}")
        print(f"Raw image folder: {self.raw_folder}")
        print(f"BBox warnings: {len(bbox_warnings)}")
        for msg in bbox_warnings:
            print(msg)

        return {
            "images_saved": total_images_saved,
            "bbox_warnings": bbox_warnings,
        }


# =========================
# Main
# =========================

def main(args):
    cal = load_calibration(args.calibration)

    x_values = inclusive_float_range(args.x_start, args.x_end, args.x_step)
    z_values = inclusive_float_range(args.z_start, args.z_end, args.z_step)

    requested_tip_angles_deg = [0.0, 90.0, 180.0]
    motion_phase = _normalize_motion_phase_name(args.motion_phase) or cal.default_motion_phase

    passes: List[FixedAnglePass] = []
    for req_angle in requested_tip_angles_deg:
        pass_plan = generate_fixed_angle_grid_pass(
            cal=cal,
            requested_tip_angle_deg=req_angle,
            fixed_c_deg=float(args.fixed_c),
            tip_y=float(args.point_y),
            x_values=x_values,
            z_values=z_values,
            inverse_samples=int(args.custom_inverse_samples),
            flip_rz_sign=bool(args.flip_rz_sign),
            snake_scan=(not bool(args.disable_snake_scan)),
            motion_phase=motion_phase,
        )
        passes.append(pass_plan)

    print("\nPlanned passes:")
    for i, p in enumerate(passes, start=1):
        s = summarize_grid_pass(p)
        print(
            f"  Pass {i}: requested angle={p.requested_tip_angle_deg:.3f} deg, "
            f"used angle={p.used_tip_angle_deg:.3f} deg, "
            f"B={p.b_cmd:.3f}, C={p.c_cmd:.3f}, "
            f"points={s['n_points']}"
        )
        print(
            f"          stage X [{s['stage_x_min']:.3f}, {s['stage_x_max']:.3f}]  "
            f"Y [{s['stage_y_min']:.3f}, {s['stage_y_max']:.3f}]  "
            f"Z [{s['stage_z_min']:.3f}, {s['stage_z_max']:.3f}]"
        )

    print("\nGrid summary:")
    print(f"  Tip X values: {len(x_values)} points from {x_values[0]:.3f} to {x_values[-1]:.3f}")
    print(f"  Tip Z values: {len(z_values)} points from {z_values[0]:.3f} to {z_values[-1]:.3f}")
    print(f"  Tip Y fixed: {float(args.point_y):.3f}")
    print(f"  Total captures planned: {sum(len(p.grid_points) for p in passes)}")
    print(f"  C fixed for all passes: {float(args.fixed_c):.3f}")
    print(f"  flip_rz_sign: {bool(args.flip_rz_sign)}")
    print(f"  motion_phase: {motion_phase}")

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

    runner = FixedOrientationGridRunner(
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

        results = runner.execute_passes_and_capture(
            cal=cal,
            passes=passes,
            start_pose=start_pose,
            end_pose=end_pose,
            safe_approach_z=float(args.safe_approach_z),
            travel_feed=float(args.travel_feed),
            scan_feed=float(args.scan_feed),
            virtual_bbox=virtual_bbox,
            dwell_before_ms=int(args.dwell_before_ms),
            dwell_after_ms=int(args.dwell_after_ms),
            capture_settle_s=float(args.capture_settle_s),
            travel_move_settle_s=float(args.travel_move_settle_s),
            inter_grid_wait_s=float(args.inter_grid_wait_s),
            camera_flush_frames=int(args.camera_flush_frames),
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
            "Run a fixed-orientation grid acquisition using the calibration JSON. "
            "B is held fixed for each pass, C is held fixed at 0, and images are "
            "captured at every requested tip-space grid point."
        )
    )

    # Run / folders
    ap.add_argument("--parent-directory", default=os.getcwd(), help="Parent folder for the run output.")
    ap.add_argument("--project-name", default=DEFAULT_PROJECT_NAME, help="Run folder name.")
    ap.add_argument("--allow-existing", action="store_true", default=DEFAULT_ALLOW_EXISTING,
                    help="Allow reuse of an existing run folder.")
    ap.add_argument("--add-date", action="store_true", default=DEFAULT_ADD_DATE,
                    help="Append timestamp to the run folder.")

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

    # Fixed tip-space Y
    ap.add_argument("--point-y", type=float, default=DEFAULT_POINT_Y,
                    help="Fixed tip-space Y value for the grid.")

    # Tip-space grid
    ap.add_argument("--x-start", type=float, default=DEFAULT_X_START, help="Grid tip X start.")
    ap.add_argument("--x-end", type=float, default=DEFAULT_X_END, help="Grid tip X end.")
    ap.add_argument("--x-step", type=float, default=DEFAULT_X_STEP, help="Grid tip X step.")
    ap.add_argument("--z-start", type=float, default=DEFAULT_Z_START, help="Grid tip Z start.")
    ap.add_argument("--z-end", type=float, default=DEFAULT_Z_END, help="Grid tip Z end.")
    ap.add_argument("--z-step", type=float, default=DEFAULT_Z_STEP, help="Grid tip Z step.")

    # Fixed orientation
    ap.add_argument("--fixed-c", type=float, default=DEFAULT_FIXED_C,
                    help="Fixed C angle for all passes. Default is 0 deg.")

    # Sign correction
    ap.add_argument(
        "--flip-rz-sign",
        action="store_true",
        default=DEFAULT_FLIP_RZ_SIGN,
        help="Multiply the polynomial-derived r and z offsets by -1.",
    )

    # Inversion resolution
    ap.add_argument("--custom-inverse-samples", type=int, default=DEFAULT_CUSTOM_INV_SAMPLES,
                    help="Dense sampling count used for numeric tip-angle -> B inversion.")
    ap.add_argument(
        "--motion-phase",
        choices=["pull", "release"],
        default=None,
        help="Which phase-specific fit models to use for tip angle and kinematics. "
             "Defaults to the calibration JSON's default phase.",
    )

    # Scan order
    ap.add_argument("--disable-snake-scan", action="store_true",
                    help="Disable serpentine scan order. Default uses snake scan.")

    # Feedrates
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED,
                    help="Feedrate for larger approach/reposition moves.")
    ap.add_argument("--scan-feed", type=float, default=DEFAULT_SCAN_FEED,
                    help="Feedrate for grid scanning moves.")

    # Optional waits
    ap.add_argument("--dwell-before-ms", type=int, default=DEFAULT_DWELL_BEFORE_MS)
    ap.add_argument("--dwell-after-ms", type=int, default=DEFAULT_DWELL_AFTER_MS)
    ap.add_argument("--capture-settle-s", type=float, default=DEFAULT_CAPTURE_SETTLE_S,
                    help="Extra settle time after each scan move, before capture (default: 0.5 s).")
    ap.add_argument("--travel-move-settle-s", type=float, default=DEFAULT_TRAVEL_MOVE_SETTLE_S,
                    help="Extra settle time after travel moves.")
    ap.add_argument("--inter-grid-wait-s", type=float, default=DEFAULT_INTER_GRID_WAIT_S,
                    help="Wait time at the configured start pose before each grid/pass.")

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
