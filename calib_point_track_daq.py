#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone point-tracking acquisition script.

What it does:
- Loads the calibration JSON
- Generates a fixed-tip tracked XYZ/B/C motion
- Connects to the Duet robot and the camera
- Executes the motion directly on the robot
- Captures an image after every executed tracked move and saves it to:
    <project folder>/raw_image_data_folder/

Custom bounded-C cycle:
- Start at C = 0
- Track to C = -360 at the starting tip angle
- Sweep C: -360 -> 360 while tip angle goes 0 -> 90
- Track to C = -60 while holding/repositioning tip angle appropriately
- Sweep C: -60 -> 60 while tip angle goes 90 -> 180
- Track to C = 120 while returning tip angle 180 -> 90
- Sweep C: 120 -> 240 while tip angle goes 90 -> 180
- Track back toward next cycle start while preserving fixed-tip tracking
- Repeat 3 times

If requested tip angles fall outside calibration, the script uses the closest
tip angles that do exist in calibration.

New sign correction option:
- Use --flip-rz-sign if your calibration file has r and z polynomial signs flipped.
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
DEFAULT_PROJECT_NAME = "Point_Tracking_Run"
DEFAULT_ALLOW_EXISTING = True
DEFAULT_ADD_DATE = True

DEFAULT_POINT_X = 100.0
DEFAULT_POINT_Y = 20.0
DEFAULT_POINT_Z = -155.0

DEFAULT_TRAVEL_FEED = 1500.0
DEFAULT_PROBE_FEED = 600.0
DEFAULT_C_FEED = 10000.0
DEFAULT_C_MAX_FEED = 15000.0
DEFAULT_C_ACCEL_TIME_S = 0.2
DEFAULT_C_DECEL_TIME_S = 0.2

DEFAULT_TIP_ANGLE_SAMPLES = 5000
DEFAULT_CUSTOM_INV_SAMPLES = 20000

DEFAULT_CUSTOM_REPEATS = 1
DEFAULT_CUSTOM_SWEEP_STEPS = 50
DEFAULT_CUSTOM_TRAVEL_STEPS = 40

DEFAULT_START_X = 100.0
DEFAULT_START_Y = 20.0
DEFAULT_START_Z = -155.0
DEFAULT_START_B = 0.0
DEFAULT_START_C = 0.0

DEFAULT_END_X = 100.0
DEFAULT_END_Y = 20.0
DEFAULT_END_Z = -155.0
DEFAULT_END_B = 0.0
DEFAULT_END_C = 0.0

DEFAULT_SAFE_APPROACH_Z = -155.0

DEFAULT_DWELL_BEFORE_MS = 0
DEFAULT_DWELL_AFTER_MS = 0

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

DEFAULT_ROTATION_SETTLE_S = 0.0
DEFAULT_TRACKED_MOVE_SETTLE_S = 0.0
DEFAULT_TRAVEL_MOVE_SETTLE_S = 0.0
DEFAULT_CAPTURE_AT_START = True

DEFAULT_FLIP_RZ_SIGN = True

OFFPLANE_SIGN = -1.0


# =========================
# Data structures
# =========================

@dataclass
class Calibration:
    pr: np.ndarray
    pz: np.ndarray
    py_off: Optional[np.ndarray]
    pa: Optional[np.ndarray]

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


def load_calibration(json_path: str) -> Calibration:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")

    with p.open("r") as f:
        data = json.load(f)

    cubic = data["cubic_coefficients"]
    pr = np.array(cubic["r_coeffs"], dtype=float)
    pz = np.array(cubic["z_coeffs"], dtype=float)
    py_off_raw = cubic.get("offplane_y_coeffs", None)
    py_off = None if py_off_raw is None else np.array(py_off_raw, dtype=float)
    pa_raw = cubic.get("tip_angle_coeffs", None)
    pa = None if pa_raw is None else np.array(pa_raw, dtype=float)

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
        pr=pr,
        pz=pz,
        py_off=py_off,
        pa=pa,
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


def eval_r(cal: Calibration, b: Any, flip_rz_sign: bool = False) -> np.ndarray:
    s = -1.0 if bool(flip_rz_sign) else 1.0
    return s * poly_eval(cal.pr, b)


def eval_z(cal: Calibration, b: Any, flip_rz_sign: bool = False) -> np.ndarray:
    s = -1.0 if bool(flip_rz_sign) else 1.0
    return s * poly_eval(cal.pz, b)


def eval_offplane_y(cal: Calibration, b: Any) -> np.ndarray:
    return OFFPLANE_SIGN * poly_eval(cal.py_off, b, default_if_none=0.0)


def eval_tip_angle_deg(cal: Calibration, b: Any) -> np.ndarray:
    return poly_eval(cal.pa, b)


def predict_r_z_offplane(
    cal: Calibration,
    b: float,
    flip_rz_sign: bool = False,
) -> Tuple[float, float, float]:
    r = float(eval_r(cal, b, flip_rz_sign=flip_rz_sign))
    z = float(eval_z(cal, b, flip_rz_sign=flip_rz_sign))
    y_off = float(eval_offplane_y(cal, b))
    return r, z, y_off


def predict_tip_xyz_from_bc(
    cal: Calibration,
    b: float,
    c_deg: float,
    flip_rz_sign: bool = False,
) -> np.ndarray:
    r, z, y_off = predict_r_z_offplane(cal, b, flip_rz_sign=flip_rz_sign)
    c = math.radians(float(c_deg))
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def tip_offset_xyz_physical(
    cal: Calibration,
    b: float,
    c_deg: float,
    flip_rz_sign: bool = False,
) -> np.ndarray:
    return predict_tip_xyz_from_bc(cal, b, c_deg, flip_rz_sign=flip_rz_sign)


def stage_xyz_for_fixed_tip(
    cal: Calibration,
    p_tip_xyz: np.ndarray,
    b: float,
    c_deg: float,
    flip_rz_sign: bool = False,
) -> np.ndarray:
    return p_tip_xyz - tip_offset_xyz_physical(cal, b, c_deg, flip_rz_sign=flip_rz_sign)


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
    if cal.pa is None:
        raise ValueError(
            "This motion mode requires 'tip_angle_coeffs' in the calibration JSON."
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


# =========================
# Custom tracked trajectory
# =========================

def _append_tracked_linear_segment(
    traj: List[Tuple[float, float, np.ndarray]],
    cal: Calibration,
    p_tip_fixed: np.ndarray,
    angle_table_deg: np.ndarray,
    b_table: np.ndarray,
    tip_angle_start_deg: float,
    tip_angle_end_deg: float,
    c_start_deg: float,
    c_end_deg: float,
    n_segments: int,
    flip_rz_sign: bool = False,
    include_first_point: bool = False,
):
    nseg = max(1, int(n_segments))

    i_start = 0 if include_first_point else 1
    for i in range(i_start, nseg + 1):
        t = i / float(nseg)

        req_tip = (1.0 - t) * float(tip_angle_start_deg) + t * float(tip_angle_end_deg)
        b_cmd, _used_tip = tip_angle_deg_to_b_clipped(
            requested_tip_angle_deg=req_tip,
            angle_table_deg=angle_table_deg,
            b_table=b_table,
        )

        c_cmd = (1.0 - t) * float(c_start_deg) + t * float(c_end_deg)
        c_cmd = clamp_c_bounded(c_cmd)

        p_stage = stage_xyz_for_fixed_tip(
            cal,
            p_tip_fixed,
            b_cmd,
            c_cmd,
            flip_rz_sign=flip_rz_sign,
        )
        traj.append((float(b_cmd), float(c_cmd), p_stage))


def generate_bounded_custom_cycle_trajectory(
    cal: Calibration,
    p_tip_fixed: np.ndarray,
    repeats: int = DEFAULT_CUSTOM_REPEATS,
    sweep_steps: int = DEFAULT_CUSTOM_SWEEP_STEPS,
    travel_steps: int = DEFAULT_CUSTOM_TRAVEL_STEPS,
    inverse_samples: int = DEFAULT_CUSTOM_INV_SAMPLES,
    flip_rz_sign: bool = False,
) -> Tuple[List[Tuple[float, float, np.ndarray]], dict]:
    """
    Bounded custom cycle that always keeps C in [-360, 360].

    Per cycle:
      A) tracked move:  C 0    -> -360, tip 0   -> 0
      B) tracked sweep: C -360 ->  360, tip 0   -> 90
      C) tracked move:  C 360  ->  -60, tip 90  -> 90
      D) tracked sweep: C -60  ->   60, tip 90  -> 180
      E) tracked move:  C 60   ->  120, tip 180 -> 90
      F) tracked sweep: C 120  ->  240, tip 90  -> 180

    Between cycles:
      G) tracked move:  C 240  ->    0, tip 180 -> 0
    """
    angle_table_deg, b_table = build_tip_angle_inverse_table(
        cal=cal,
        num_samples=int(inverse_samples),
    )

    available_tip_min = float(angle_table_deg[0])
    available_tip_max = float(angle_table_deg[-1])

    req_angles = {"a0": 0.0, "a90": 90.0, "a180": 180.0}
    used_angles = {
        k: clamp(v, available_tip_min, available_tip_max)
        for k, v in req_angles.items()
    }

    a0 = used_angles["a0"]
    a90 = used_angles["a90"]
    a180 = used_angles["a180"]

    traj: List[Tuple[float, float, np.ndarray]] = []

    # Initial point exactly at cycle start state: C=0, tip=a0
    b0, _ = tip_angle_deg_to_b_clipped(a0, angle_table_deg, b_table)
    p0 = stage_xyz_for_fixed_tip(cal, p_tip_fixed, b0, 0.0, flip_rz_sign=flip_rz_sign)
    traj.append((float(b0), 0.0, p0))

    for rep in range(max(1, int(repeats))):
        _append_tracked_linear_segment(
            traj, cal, p_tip_fixed, angle_table_deg, b_table,
            tip_angle_start_deg=a0, tip_angle_end_deg=a0,
            c_start_deg=0.0, c_end_deg=-360.0,
            n_segments=travel_steps,
            flip_rz_sign=flip_rz_sign,
            include_first_point=False,
        )

        _append_tracked_linear_segment(
            traj, cal, p_tip_fixed, angle_table_deg, b_table,
            tip_angle_start_deg=a0, tip_angle_end_deg=a90,
            c_start_deg=-360.0, c_end_deg=360.0,
            n_segments=sweep_steps,
            flip_rz_sign=flip_rz_sign,
            include_first_point=False,
        )

        _append_tracked_linear_segment(
            traj, cal, p_tip_fixed, angle_table_deg, b_table,
            tip_angle_start_deg=a90, tip_angle_end_deg=a90,
            c_start_deg=360.0, c_end_deg=-60.0,
            n_segments=travel_steps,
            flip_rz_sign=flip_rz_sign,
            include_first_point=False,
        )

        _append_tracked_linear_segment(
            traj, cal, p_tip_fixed, angle_table_deg, b_table,
            tip_angle_start_deg=a90, tip_angle_end_deg=a180,
            c_start_deg=-60.0, c_end_deg=60.0,
            n_segments=sweep_steps,
            flip_rz_sign=flip_rz_sign,
            include_first_point=False,
        )

        _append_tracked_linear_segment(
            traj, cal, p_tip_fixed, angle_table_deg, b_table,
            tip_angle_start_deg=a180, tip_angle_end_deg=a90,
            c_start_deg=60.0, c_end_deg=120.0,
            n_segments=travel_steps,
            flip_rz_sign=flip_rz_sign,
            include_first_point=False,
        )

        _append_tracked_linear_segment(
            traj, cal, p_tip_fixed, angle_table_deg, b_table,
            tip_angle_start_deg=a90, tip_angle_end_deg=a180,
            c_start_deg=120.0, c_end_deg=240.0,
            n_segments=sweep_steps,
            flip_rz_sign=flip_rz_sign,
            include_first_point=False,
        )

        if rep < int(repeats) - 1:
            _append_tracked_linear_segment(
                traj, cal, p_tip_fixed, angle_table_deg, b_table,
                tip_angle_start_deg=a180, tip_angle_end_deg=a0,
                c_start_deg=240.0, c_end_deg=0.0,
                n_segments=travel_steps,
                flip_rz_sign=flip_rz_sign,
                include_first_point=False,
            )

    meta = {
        "requested_tip_angles_deg": req_angles,
        "used_tip_angles_deg": used_angles,
        "available_tip_angle_range_deg": [available_tip_min, available_tip_max],
        "flip_rz_sign": bool(flip_rz_sign),
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
    traj: List[Tuple[float, float, np.ndarray]],
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
        p0 = traj[i - 1][2]
        p1 = traj[i][2]
        c0 = traj[i - 1][1]
        c1 = traj[i][1]
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

def compute_traj_meta(traj: List[Tuple[float, float, np.ndarray]]) -> dict:
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
        }

    xyz = np.vstack([p for _, _, p in traj])
    bb = np.array([b for b, _, _ in traj], dtype=float)
    cc = np.array([c for _, c, _ in traj], dtype=float)

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

        path = os.path.join(self.point_tracking_folder, filename)
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

    def execute_motion_and_capture(
        self,
        cal: Calibration,
        traj: List[Tuple[float, float, np.ndarray]],
        start_pose: Tuple[float, float, float, float, float],
        end_pose: Tuple[float, float, float, float, float],
        safe_approach_z: float,
        travel_feed: float,
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
        rotation_settle_s: float = 0.0,
        camera_flush_frames: int = 1,
        capture_at_start: bool = True,
    ):
        if self.cam is None:
            raise RuntimeError("Camera is not connected.")
        if self.rrf is None:
            raise RuntimeError("Robot is not connected.")

        bbox_warnings: List[str] = []
        sample_counter = 0

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

        sx, sy, sz, sb, sc = [float(v) for v in start_pose]

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

        if not traj:
            print("No trajectory points generated.")
        else:
            b0, c0, p0_stage = traj[0]

            x0, y0, z0 = _clamp_stage_xyz_to_bbox(
                p0_stage[0], p0_stage[1], p0_stage[2],
                virtual_bbox,
                "move to tracked start",
                bbox_warnings,
            )

            print("\nMoving to first tracked sample...")
            self.send_absolute_move(
                travel_feed,
                **{
                    cal.x_axis: x0,
                    cal.y_axis: y0,
                    cal.z_axis: z0,
                    cal.b_axis: b0,
                    cal.c_axis: clamp_c_bounded(c0),
                }
            )
            self.wait_for_duet_motion_complete(extra_settle=travel_move_settle_s)

            if capture_at_start:
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
                )

            if int(dwell_before_ms) > 0:
                print(f"Dwell before motion: {int(dwell_before_ms)} ms")
                time.sleep(float(dwell_before_ms) / 1000.0)

            print("\nExecuting coordinated tracked motion...")
            for i, (b, c, p_stage) in enumerate(traj[1:], start=1):
                x, y, z = _clamp_stage_xyz_to_bbox(
                    p_stage[0], p_stage[1], p_stage[2],
                    virtual_bbox,
                    f"tracked sample {i}",
                    bbox_warnings,
                )

                if seg_feeds:
                    fseg = seg_feeds[i - 1]
                else:
                    fseg = float(probe_feed)

                self.send_absolute_move(
                    fseg,
                    **{
                        cal.x_axis: x,
                        cal.y_axis: y,
                        cal.z_axis: z,
                        cal.b_axis: b,
                        cal.c_axis: clamp_c_bounded(c),
                    }
                )
                self.wait_for_duet_motion_complete(extra_settle=tracked_move_settle_s)

                sample_counter += 1
                self.capture_and_save(
                    sample_idx=sample_counter,
                    phase="tracked",
                    x=x,
                    y=y,
                    z=z,
                    b=b,
                    c=clamp_c_bounded(c),
                    flush_frames=camera_flush_frames,
                )

            if int(dwell_after_ms) > 0:
                print(f"Dwell after motion: {int(dwell_after_ms)} ms")
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
    cal = load_calibration(args.calibration)

    p_tip_fixed = np.array(
        [float(args.point_x), float(args.point_y), float(args.point_z)],
        dtype=float
    )

    traj, custom_meta = generate_bounded_custom_cycle_trajectory(
        cal=cal,
        p_tip_fixed=p_tip_fixed,
        repeats=int(args.custom_repeats),
        sweep_steps=int(args.custom_sweep_steps),
        travel_steps=int(args.custom_travel_steps),
        inverse_samples=int(args.custom_inverse_samples),
        flip_rz_sign=bool(args.flip_rz_sign),
    )

    meta = compute_traj_meta(traj)
    print("Trajectory summary:")
    print(f"  Samples: {meta['n_samples']} (segments={meta['n_segments']})")
    print(f"  B range used: [{meta['b_min_used']:.3f}, {meta['b_max_used']:.3f}]")
    if cal.pa is not None and meta["n_samples"] > 0:
        bb = np.array([meta["b_min_used"], meta["b_max_used"]], dtype=float)
        tip_angle_used = eval_tip_angle_deg(cal, bb)
        print(
            "  Tip-angle range at used B endpoints: "
            f"[{float(np.min(tip_angle_used)):.3f}, {float(np.max(tip_angle_used)):.3f}] deg"
        )
    print(f"  C range used: [{meta['c_min_used']:.3f}, {meta['c_max_used']:.3f}]")
    print(f"  XYZ path length: {meta['xyz_path_len_mm']:.3f} mm")

    print("Tip-angle request/usage summary:")
    print(f"  Requested: {custom_meta['requested_tip_angles_deg']}")
    print(f"  Used:      {custom_meta['used_tip_angles_deg']}")
    print(f"  Available calibrated tip-angle range: {custom_meta['available_tip_angle_range_deg']}")
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
            rotation_settle_s=float(args.rotation_settle_s),
            camera_flush_frames=int(args.camera_flush_frames),
            capture_at_start=bool(args.capture_at_start),
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
            "Run a bounded-C fixed-tip tracked XYZ/B/C motion from the calibration JSON "
            "and capture images into a point_tracking folder."
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

    # Fixed tip point
    ap.add_argument("--point-x", type=float, default=DEFAULT_POINT_X, help="Fixed tip X (Cartesian/world).")
    ap.add_argument("--point-y", type=float, default=DEFAULT_POINT_Y, help="Fixed tip Y (Cartesian/world).")
    ap.add_argument("--point-z", type=float, default=DEFAULT_POINT_Z, help="Fixed tip Z (Cartesian/world).")

    # Sign correction
    ap.add_argument(
        "--flip-rz-sign",
        action="store_true",
        default=DEFAULT_FLIP_RZ_SIGN,
        help="Multiply the polynomial-derived r and z offsets by -1. Use this if your calibration file has flipped r/z signs.",
    )

    # Custom bounded cycle controls
    ap.add_argument("--custom-repeats", type=int, default=DEFAULT_CUSTOM_REPEATS,
                    help="How many times to repeat the full cycle.")
    ap.add_argument("--custom-sweep-steps", type=int, default=DEFAULT_CUSTOM_SWEEP_STEPS,
                    help="Tracked segments used for each sweep.")
    ap.add_argument("--custom-travel-steps", type=int, default=DEFAULT_CUSTOM_TRAVEL_STEPS,
                    help="Tracked segments used for each inter-sweep travel.")
    ap.add_argument("--custom-inverse-samples", type=int, default=DEFAULT_CUSTOM_INV_SAMPLES,
                    help="Dense sampling count used for numeric tip-angle -> B inversion.")

    # Feedrates
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED)
    ap.add_argument("--probe-feed", type=float, default=DEFAULT_PROBE_FEED)
    ap.add_argument("--c-feed", type=float, default=DEFAULT_C_FEED)

    # Tracked C cap + feed scheduler
    ap.add_argument("--c-max-feed", type=float, default=DEFAULT_C_MAX_FEED)
    ap.add_argument("--c-accel-time", type=float, default=DEFAULT_C_ACCEL_TIME_S)
    ap.add_argument("--c-decel-time", type=float, default=DEFAULT_C_DECEL_TIME_S)
    ap.add_argument("--disable-segment-feed-scheduler", action="store_true",
                    help="Disable per-segment feed scheduling.")

    # Optional waits / capture behavior
    ap.add_argument("--dwell-before-ms", type=int, default=DEFAULT_DWELL_BEFORE_MS)
    ap.add_argument("--dwell-after-ms", type=int, default=DEFAULT_DWELL_AFTER_MS)
    ap.add_argument("--tracked-move-settle-s", type=float, default=DEFAULT_TRACKED_MOVE_SETTLE_S,
                    help="Extra settle time after each tracked move, before capture.")
    ap.add_argument("--travel-move-settle-s", type=float, default=DEFAULT_TRAVEL_MOVE_SETTLE_S,
                    help="Extra settle time after travel moves.")
    ap.add_argument("--rotation-settle-s", type=float, default=DEFAULT_ROTATION_SETTLE_S,
                    help="Extra settle time after any rotation-related move.")
    ap.add_argument("--capture-at-start", action="store_true", default=DEFAULT_CAPTURE_AT_START,
                    help="Capture one image at the first tracked sample before the tracked sequence starts.")

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