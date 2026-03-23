#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Standalone star-tracking acquisition script.

What it does:
- Loads the calibration JSON
- Builds the mirrored-half star motion plan from script 2:
    right half at fixed C=c0
    single C flip
    mirrored left half at fixed C=c180
- Preserves the robot/camera execution style of script 1
- Captures images only during star-path moves (not startup/end travel) and saves them to:
    <project folder>/point_tracking/
- Motion interpolation resolution and image-capture cadence are configured independently.
- Adds offplane-tracking:
    Y stage motion is solved so the tip remains at the requested constant tip-space Y
    while compensating the calibrated offplane offset of the robot.
- Adds optional sign flips for the calibrated r(B) and z(B) polynomial outputs.

IMPORTANT C-AXIS BEHAVIOR:
- C is held constant on the right half
- C is held constant on the left half
- C changes only once during the mirror-plane flip
- All commanded C values are validated to remain within [-360, 360]

IMPORTANT X-TRACKING BEHAVIOR:
- X is allowed to move whenever needed to preserve exact tip tracking.
- This includes cases where the calibration cannot realize very small |r| values
  or cannot hit the nominal mirror plane directly.
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

DEFAULT_TRAVEL_FEED = 1200.0
DEFAULT_PRINT_FEED = 200.0
DEFAULT_PRINT_FEED_B = 500.0
DEFAULT_PRINT_FEED_C = 1000.0

DEFAULT_DWELL_BEFORE_MS = 0
DEFAULT_DWELL_AFTER_MS = 0
DEFAULT_TRACKED_MOVE_SETTLE_S = 0.0
DEFAULT_TRAVEL_MOVE_SETTLE_S = 0.0
DEFAULT_CAPTURE_AT_START = False

DEFAULT_SAFE_APPROACH_Z = -145.0

DEFAULT_START_X = 100.0
DEFAULT_START_Y = 20.0
DEFAULT_START_Z = -145.0
DEFAULT_START_B = 0.0
DEFAULT_START_C = 0.0

DEFAULT_END_X = 100.0
DEFAULT_END_Y = 20.0
DEFAULT_END_Z = -145.0
DEFAULT_END_B = 0.0
DEFAULT_END_C = 0.0

DEFAULT_BBOX_X_MIN = 0.0
DEFAULT_BBOX_X_MAX = 200.0
DEFAULT_BBOX_Y_MIN = -20.0
DEFAULT_BBOX_Y_MAX = 200.0
DEFAULT_BBOX_Z_MIN = -200.0
DEFAULT_BBOX_Z_MAX = 0.0

STAR_CENTER_X = 100.0
STAR_CENTER_Y = 20.0
STAR_CENTER_Z = -145.0

DESIRED_STAR_OUTER_RADIUS = 18.0
INNER_RADIUS_RATIO = 0.38196601125
STAR_ROTATION_DEG = 270.0
SAMPLES_PER_EDGE = 30
DEFAULT_CAPTURE_EVERY_N_STAR_MOVES = 1

DEFAULT_SAFETY_MARGIN = 0.5
DEFAULT_C0_DEG = 0.0

OFFPLANE_SIGN = -1.0
C_HARD_MIN_DEG = -360.0
C_HARD_MAX_DEG = 360.0


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


@dataclass
class KinematicSignConfig:
    r_sign: float = 1.0
    z_sign: float = 1.0


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


def assert_c_is_piecewise_constant_with_single_flip(
    command_sequence: List[CommandPoint],
    c0_deg: float,
    c180_deg: float,
    atol: float = 1e-9,
) -> None:
    seen_flip = False
    prev_c = None

    for cp in command_sequence:
        if cp.phase in ("right_start", "right"):
            if abs(cp.c - c0_deg) > atol:
                raise RuntimeError(
                    f"C changed during right half: expected {c0_deg:.3f}, got {cp.c:.3f}"
                )
        elif cp.phase in ("mirror_flip", "left"):
            if abs(cp.c - c180_deg) > atol:
                raise RuntimeError(
                    f"C changed during left half / flip: expected {c180_deg:.3f}, got {cp.c:.3f}"
                )

        if prev_c is not None and abs(cp.c - prev_c) > atol:
            if seen_flip:
                raise RuntimeError("More than one C transition detected in command sequence.")
            seen_flip = True
        prev_c = cp.c


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


def eval_r(cal: Calibration, b: Any, signs: KinematicSignConfig) -> np.ndarray:
    return float(signs.r_sign) * poly_eval(cal.pr, b)


def eval_z(cal: Calibration, b: Any, signs: KinematicSignConfig) -> np.ndarray:
    return float(signs.z_sign) * poly_eval(cal.pz, b)


def eval_offplane_y(cal: Calibration, b: Any) -> np.ndarray:
    return OFFPLANE_SIGN * poly_eval(cal.py_off, b, default_if_none=0.0)


def tip_offset_xyz_physical(cal: Calibration, b: float, c_deg: float, signs: KinematicSignConfig) -> np.ndarray:
    """
    Full 3D tip offset in world/stage axes due to B and C.

    r(B) and z(B) may be sign-flipped by CLI flags.
    offplane_y(B) uses the calibration convention from script 1.
    """
    c_deg = assert_c_in_safe_range("tip_offset C", c_deg)

    r = float(eval_r(cal, b, signs))
    z = float(eval_z(cal, b, signs))
    y_off = float(eval_offplane_y(cal, b))

    c = math.radians(float(c_deg))
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def stage_xyz_for_tip_target(
    cal: Calibration,
    p_tip_xyz: np.ndarray,
    b: float,
    c_deg: float,
    signs: KinematicSignConfig,
) -> np.ndarray:
    return p_tip_xyz - tip_offset_xyz_physical(cal, b, c_deg, signs)


def sampled_r_abs_range(
    cal: Calibration,
    signs: KinematicSignConfig,
    b_lo: Optional[float] = None,
    b_hi: Optional[float] = None,
    n: int = 2001,
) -> Tuple[float, float]:
    lo = cal.b_min if b_lo is None else float(b_lo)
    hi = cal.b_max if b_hi is None else float(b_hi)
    if lo > hi:
        lo, hi = hi, lo
    bb = np.linspace(lo, hi, n)
    rr = eval_r(cal, bb, signs)
    return float(np.min(np.abs(rr))), float(np.max(np.abs(rr)))


def _find_roots_for_target(coeffs: np.ndarray, target: float, b_min: float, b_max: float) -> List[float]:
    f = lambda b: float(np.polyval(coeffs, b) - target)
    bs = np.linspace(b_min, b_max, 1200)
    fs = np.array([f(b) for b in bs], dtype=float)
    fs = np.where(np.isfinite(fs), fs, 1e9)

    roots: List[float] = []
    for a, b, fa, fb in zip(bs[:-1], bs[1:], fs[:-1], fs[1:]):
        if fa == 0.0:
            roots.append(float(a))
            continue
        if np.sign(fa) != np.sign(fb):
            try:
                roots.append(float(brentq(f, float(a), float(b), maxiter=100)))
            except Exception:
                roots.append(float(0.5 * (float(a) + float(b))))

    roots = sorted(roots)
    dedup: List[float] = []
    for r in roots:
        if not dedup or abs(r - dedup[-1]) > 1e-6:
            dedup.append(r)
    return dedup


def invert_r_to_b(
    r_target_abs: float,
    cal: Calibration,
    b_lo: float,
    b_hi: float,
    b_prev: Optional[float] = None,
) -> float:
    """
    Solve raw calibration polynomial r_raw(B)=r_target_abs.

    Note:
    - root finding is done against the stored calibration polynomial itself
    - sign flips are applied later when computing the physical tip offset
    - this is correct because the planning target is based on |r|
    """
    roots = _find_roots_for_target(cal.pr, r_target_abs, b_lo, b_hi)
    f = lambda b: float(np.polyval(cal.pr, b) - r_target_abs)

    if not roots:
        return float(b_lo if abs(f(b_lo)) <= abs(f(b_hi)) else b_hi)

    if b_prev is None or len(roots) == 1:
        return float(roots[0])

    arr = np.array(roots, dtype=float)
    return float(arr[np.argmin(np.abs(arr - float(b_prev)))])


# =========================
# Star geometry / path extraction
# =========================

def build_star_vertices(outer_radius: float, inner_ratio: float, rotation_deg: float) -> np.ndarray:
    inner_radius = outer_radius * inner_ratio
    rot = math.radians(rotation_deg)
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


def scale_star_if_needed(
    desired_outer_radius: float,
    cal: Calibration,
    signs: KinematicSignConfig,
    b_lo: float,
    b_hi: float,
    safety_margin: float,
) -> Tuple[float, float, float]:
    r_abs_min, r_abs_max = sampled_r_abs_range(cal, signs=signs, b_lo=b_lo, b_hi=b_hi)
    max_reachable = max(0.0, r_abs_max - float(safety_margin))
    outer_used = min(float(desired_outer_radius), max_reachable)
    return outer_used, r_abs_min, r_abs_max


# =========================
# Command generation
# =========================

def local_star_points_to_command_points_tip_priority(
    local_pts_xz: np.ndarray,
    cal: Calibration,
    signs: KinematicSignConfig,
    center_x: float,
    center_y: float,
    center_z: float,
    b_lo: float,
    b_hi: float,
    c_state: float,
    move_phase_start: str,
    move_phase_rest: str,
    move_feed_start: float,
    move_feed_rest: float,
    b_seed_from: Optional[np.ndarray] = None,
) -> Tuple[List[CommandPoint], dict, np.ndarray]:
    c_state = assert_c_in_safe_range("c_state", c_state)

    N = local_pts_xz.shape[0]
    pts: List[CommandPoint] = []
    b_cmd = np.zeros(N, dtype=float)

    r_abs_min, r_abs_max = sampled_r_abs_range(cal, signs=signs, b_lo=b_lo, b_hi=b_hi)

    n_inner_clamped = 0
    n_outer_clamped = 0
    max_abs_x_offset_from_nominal = 0.0
    max_abs_y_offset_from_nominal = 0.0

    b_prev: Optional[float] = None

    for i in range(N):
        x_local = float(local_pts_xz[i, 0])
        z_local = float(local_pts_xz[i, 1])

        x_tip_target = float(center_x + x_local)
        y_tip_target = float(center_y)
        z_tip_target = float(center_z + z_local)

        # same script-2 policy: choose B from desired radial magnitude |x_local|
        # and allow X_stage to shift if exact centerline reach is impossible
        r_target_abs = abs(x_local)
        if r_target_abs < r_abs_min:
            r_target_abs = r_abs_min
            n_inner_clamped += 1
        elif r_target_abs > r_abs_max:
            r_target_abs = r_abs_max
            n_outer_clamped += 1

        b_hint = None
        if b_seed_from is not None and i < len(b_seed_from):
            b_hint = float(b_seed_from[i])
        elif b_prev is not None:
            b_hint = float(b_prev)

        b_i = invert_r_to_b(r_target_abs, cal, b_lo=b_lo, b_hi=b_hi, b_prev=b_hint)
        b_cmd[i] = b_i
        b_prev = b_i

        tip_target = np.array([x_tip_target, y_tip_target, z_tip_target], dtype=float)
        stage_xyz = stage_xyz_for_tip_target(cal, tip_target, float(b_i), float(c_state), signs)

        tip_check = stage_xyz + tip_offset_xyz_physical(cal, float(b_i), float(c_state), signs)
        if not np.allclose(tip_check, tip_target, atol=1e-9, rtol=0.0):
            raise RuntimeError("Internal tip-tracking consistency check failed.")

        x_offset = stage_xyz[0] - float(center_x)
        y_offset = stage_xyz[1] - float(center_y)
        max_abs_x_offset_from_nominal = max(max_abs_x_offset_from_nominal, abs(x_offset))
        max_abs_y_offset_from_nominal = max(max_abs_y_offset_from_nominal, abs(y_offset))

        pts.append(
            CommandPoint(
                phase=move_phase_start if i == 0 else move_phase_rest,
                x=float(stage_xyz[0]),
                y=float(stage_xyz[1]),
                z=float(stage_xyz[2]),
                b=float(b_i),
                c=float(c_state),
                feed=float(move_feed_start if i == 0 else move_feed_rest),
                tip_x=float(x_tip_target),
                tip_y=float(y_tip_target),
                tip_z=float(z_tip_target),
            )
        )

    meta = {
        "r_abs_min": float(r_abs_min),
        "r_abs_max": float(r_abs_max),
        "n_inner_clamped": int(n_inner_clamped),
        "n_outer_clamped": int(n_outer_clamped),
        "max_abs_x_offset_from_nominal": float(max_abs_x_offset_from_nominal),
        "max_abs_y_offset_from_nominal": float(max_abs_y_offset_from_nominal),
    }
    return pts, meta, b_cmd


def build_star_command_sequence(
    cal: Calibration,
    signs: KinematicSignConfig,
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
) -> Tuple[List[CommandPoint], dict]:
    c0_deg = assert_c_in_safe_range("c0_deg", c0_deg)
    c180_deg = assert_c_in_safe_range("c180_deg", c180_deg)

    outer_used, r_abs_min, r_abs_max = scale_star_if_needed(
        desired_outer_radius=float(outer_radius),
        cal=cal,
        signs=signs,
        b_lo=b_lo,
        b_hi=b_hi,
        safety_margin=float(safety_margin),
    )
    if outer_used <= 0.0:
        raise RuntimeError("Star radius not reachable after safety margin.")

    star_vertices = build_star_vertices(
        outer_radius=outer_used,
        inner_ratio=float(inner_ratio),
        rotation_deg=float(rotation_deg),
    )
    full_pts = densify_polyline(star_vertices, samples_per_edge=int(samples_per_edge))
    right_half_pts = extract_right_half_polyline(full_pts)

    right_cmds, meta_right, b_right = local_star_points_to_command_points_tip_priority(
        local_pts_xz=right_half_pts,
        cal=cal,
        signs=signs,
        center_x=float(center_x),
        center_y=float(center_y),
        center_z=float(center_z),
        b_lo=b_lo,
        b_hi=b_hi,
        c_state=float(c0_deg),
        move_phase_start="right_start",
        move_phase_rest="right",
        move_feed_start=float(jog_feed),
        move_feed_rest=float(print_feed_b),
        b_seed_from=None,
    )

    left_half_pts = right_half_pts.copy()
    left_half_pts[:, 0] *= -1.0

    left_cmds, meta_left, b_left = local_star_points_to_command_points_tip_priority(
        local_pts_xz=left_half_pts,
        cal=cal,
        signs=signs,
        center_x=float(center_x),
        center_y=float(center_y),
        center_z=float(center_z),
        b_lo=b_lo,
        b_hi=b_hi,
        c_state=float(c180_deg),
        move_phase_start="mirror_flip",
        move_phase_rest="left",
        move_feed_start=float(print_feed_c),
        move_feed_rest=float(print_feed_b),
        b_seed_from=b_right,
    )

    sequence = []
    sequence.extend(right_cmds)
    sequence.extend(left_cmds)

    assert_all_command_c_safe(sequence)
    assert_c_is_piecewise_constant_with_single_flip(sequence, c0_deg=float(c0_deg), c180_deg=float(c180_deg))

    rr_right = eval_r(cal, b_right, signs)
    rr_left = eval_r(cal, b_left, signs)
    zz_right = eval_z(cal, b_right, signs)
    zz_left = eval_z(cal, b_left, signs)
    yy_right = eval_offplane_y(cal, b_right)
    yy_left = eval_offplane_y(cal, b_left)

    meta = {
        "outer_radius_requested": float(outer_radius),
        "outer_radius_used": float(outer_used),
        "r_abs_min_reachable": float(r_abs_min),
        "r_abs_max_reachable": float(r_abs_max),
        "right_half_points": int(len(right_half_pts)),
        "meta_right": meta_right,
        "meta_left": meta_left,
        "b_right_min": float(np.min(b_right)),
        "b_right_max": float(np.max(b_right)),
        "b_left_min": float(np.min(b_left)),
        "b_left_max": float(np.max(b_left)),
        "r_right_min": float(np.min(rr_right)),
        "r_right_max": float(np.max(rr_right)),
        "r_left_min": float(np.min(rr_left)),
        "r_left_max": float(np.max(rr_left)),
        "z_right_min": float(np.min(zz_right)),
        "z_right_max": float(np.max(zz_right)),
        "z_left_min": float(np.min(zz_left)),
        "z_left_max": float(np.max(zz_left)),
        "yoff_right_min": float(np.min(yy_right)),
        "yoff_right_max": float(np.max(yy_right)),
        "yoff_left_min": float(np.min(yy_left)),
        "yoff_left_max": float(np.max(yy_left)),
        "x_stage_min": float(min(p.x for p in sequence)),
        "x_stage_max": float(max(p.x for p in sequence)),
        "y_stage_min": float(min(p.y for p in sequence)),
        "y_stage_max": float(max(p.y for p in sequence)),
        "z_stage_min": float(min(p.z for p in sequence)),
        "z_stage_max": float(max(p.z for p in sequence)),
        "c0_deg": float(c0_deg),
        "c180_deg": float(c180_deg),
        "r_sign": float(signs.r_sign),
        "z_sign": float(signs.z_sign),
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


def save_desired_star_motion_plot(
    plot_path: str,
    command_sequence: List[CommandPoint],
) -> str:
    if not command_sequence:
        raise RuntimeError("Cannot save desired star motion plot: command_sequence is empty.")

    import matplotlib.pyplot as plt

    tip_x = np.asarray([cp.tip_x for cp in command_sequence], dtype=float)
    tip_z = np.asarray([cp.tip_z for cp in command_sequence], dtype=float)

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
        s=14,
        color="#f8fafc",
        edgecolors="#8cf7ff",
        linewidths=0.45,
        alpha=0.95,
        label="Sampled tip targets",
        zorder=3,
    )
    ax.scatter(
        [tip_x[0]],
        [tip_z[0]],
        s=72,
        color="#f4d35e",
        edgecolors="none",
        label="Start",
        zorder=4,
    )

    phases = [cp.phase for cp in command_sequence]
    mirror_idx = next((i for i, phase in enumerate(phases) if phase == "mirror_flip"), None)
    if mirror_idx is not None:
        ax.scatter(
            [tip_x[mirror_idx]],
            [tip_z[mirror_idx]],
            s=60,
            color="#ff8fab",
            edgecolors="none",
            label="Mirror flip",
            zorder=4,
        )

    cx = float(np.mean(tip_x))
    cz = float(np.mean(tip_z))
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

        self.point_tracking_folder = os.path.join(self.run_folder, "point_tracking")
        os.makedirs(self.point_tracking_folder, exist_ok=True)

        self.cam = None
        self.rrf = None
        self.cam_port = None

        print(f"Using run folder: {self.run_folder}")
        print(f"Using point-tracking folder: {self.point_tracking_folder}")

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

    def execute_star_motion_and_capture(
        self,
        cal: Calibration,
        command_sequence: List[CommandPoint],
        start_pose: Tuple[float, float, float, float, float],
        end_pose: Tuple[float, float, float, float, float],
        safe_approach_z: float,
        travel_feed: float,
        virtual_bbox: dict,
        dwell_before_ms: int = 0,
        dwell_after_ms: int = 0,
        tracked_move_settle_s: float = 0.0,
        travel_move_settle_s: float = 0.0,
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

        print("\n" + "=" * 72)
        print("STARTING STAR-TRACKING ACQUISITION RUN")
        print("=" * 72)
        print(f"Tracked samples: {len(command_sequence)}")

        sx, sy, sz, sb, sc = [float(v) for v in start_pose]
        ex, ey, ez, eb, ec = [float(v) for v in end_pose]

        sc = assert_c_in_safe_range("start_c", sc)
        ec = assert_c_in_safe_range("end_c", ec)
        assert_all_command_c_safe(command_sequence)

        print("\nSafe startup approach...")
        self.send_absolute_move(
            travel_feed,
            **{
                cal.z_axis: float(safe_approach_z),
                cal.b_axis: sb,
                cal.c_axis: sc,
            }
        )
        self.wait_for_duet_motion_complete(extra_settle=travel_move_settle_s)

        self.send_absolute_move(
            travel_feed,
            **{
                cal.x_axis: sx,
                cal.y_axis: sy,
                cal.b_axis: sb,
                cal.c_axis: sc,
            }
        )
        self.wait_for_duet_motion_complete(extra_settle=travel_move_settle_s)

        self.send_absolute_move(
            travel_feed,
            **{
                cal.z_axis: sz,
                cal.b_axis: sb,
                cal.c_axis: sc,
            }
        )
        self.wait_for_duet_motion_complete(extra_settle=travel_move_settle_s)

        if not command_sequence:
            print("No command sequence generated.")
        else:
            first = command_sequence[0]
            x0, y0, z0 = _clamp_stage_xyz_to_bbox(
                first.x, first.y, first.z,
                virtual_bbox,
                "move to tracked start",
                bbox_warnings,
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
                    cp.x, cp.y, cp.z,
                    virtual_bbox,
                    f"tracked sample {i}",
                    bbox_warnings,
                )

                self.send_absolute_move(
                    cp.feed,
                    **{
                        cal.x_axis: x,
                        cal.y_axis: y,
                        cal.z_axis: z,
                        cal.b_axis: cp.b,
                        cal.c_axis: cp.c,
                    }
                )
                self.wait_for_duet_motion_complete(extra_settle=tracked_move_settle_s)

                if cp.phase in {"right", "left"}:
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

            if int(dwell_after_ms) > 0:
                print(f"Dwell after motion: {int(dwell_after_ms)} ms")
                time.sleep(float(dwell_after_ms) / 1000.0)

        print("\nSafe end move...")
        self.send_absolute_move(
            travel_feed,
            **{
                cal.z_axis: float(safe_approach_z),
                cal.b_axis: eb,
                cal.c_axis: ec,
            }
        )
        self.wait_for_duet_motion_complete(extra_settle=travel_move_settle_s)

        self.send_absolute_move(
            travel_feed,
            **{
                cal.x_axis: ex,
                cal.y_axis: ey,
                cal.b_axis: eb,
                cal.c_axis: ec,
            }
        )
        self.wait_for_duet_motion_complete(extra_settle=travel_move_settle_s)

        self.send_absolute_move(
            travel_feed,
            **{
                cal.z_axis: ez,
                cal.b_axis: eb,
                cal.c_axis: ec,
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
        }


# =========================
# Main
# =========================

def main(args):
    cal = load_calibration(args.calibration)

    signs = KinematicSignConfig(
        r_sign=(-1.0 if bool(args.flip_r_sign) else 1.0),
        z_sign=(-1.0 if bool(args.flip_z_sign) else 1.0),
    )

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
        signs=signs,
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
    )

    print("Trajectory summary:")
    print(f"  Right-half points: {meta['right_half_points']}")
    print(f"  Star outer radius requested: {meta['outer_radius_requested']:.3f} mm")
    print(f"  Star outer radius used:      {meta['outer_radius_used']:.3f} mm")
    print(f"  r_sign: {meta['r_sign']:+.1f}")
    print(f"  z_sign: {meta['z_sign']:+.1f}")
    print(
        f"  Reachable |r(B)| over commanded range: "
        f"[{meta['r_abs_min_reachable']:.3f}, {meta['r_abs_max_reachable']:.3f}] mm"
    )
    print(f"  B right range: [{meta['b_right_min']:.3f}, {meta['b_right_max']:.3f}]")
    print(f"  B left  range: [{meta['b_left_min']:.3f}, {meta['b_left_max']:.3f}]")
    print(f"  X stage range: [{meta['x_stage_min']:.3f}, {meta['x_stage_max']:.3f}]")
    print(f"  Y stage range: [{meta['y_stage_min']:.3f}, {meta['y_stage_max']:.3f}]")
    print(f"  Z stage range: [{meta['z_stage_min']:.3f}, {meta['z_stage_max']:.3f}]")
    print(f"  C values used: [{meta['c0_deg']:.3f}, {meta['c180_deg']:.3f}] (single flip only)")
    print(
        f"  Right max |X offset from nominal center|: "
        f"{meta['meta_right']['max_abs_x_offset_from_nominal']:.3f} mm"
    )
    print(
        f"  Left  max |X offset from nominal center|: "
        f"{meta['meta_left']['max_abs_x_offset_from_nominal']:.3f} mm"
    )
    print(
        f"  Right max |Y offset from nominal center_y|: "
        f"{meta['meta_right']['max_abs_y_offset_from_nominal']:.3f} mm"
    )
    print(
        f"  Left  max |Y offset from nominal center_y|: "
        f"{meta['meta_left']['max_abs_y_offset_from_nominal']:.3f} mm"
    )

    if meta["meta_right"]["n_inner_clamped"] or meta["meta_left"]["n_inner_clamped"]:
        print("[info] Inner radial clamp occurred; X/Y correction preserved exact tip position near centerline.")
    if meta["meta_right"]["n_outer_clamped"] or meta["meta_left"]["n_outer_clamped"]:
        print("[info] Outer radial clamp occurred; consider reducing star radius for lower correction.")

    start_pose = (
        float(args.start_x),
        float(args.start_y),
        float(args.start_z),
        float(args.start_b),
        start_c,
    )
    end_pose = (
        float(args.end_x),
        float(args.end_y),
        float(args.end_z),
        float(args.end_b),
        end_c,
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

    desired_star_plot_path = os.path.join(runner.run_folder, "desired_star_motion.png")
    save_desired_star_motion_plot(
        plot_path=desired_star_plot_path,
        command_sequence=command_sequence,
    )
    print(f"Saved desired star motion plot: {desired_star_plot_path}")

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

        results = runner.execute_star_motion_and_capture(
            cal=cal,
            command_sequence=command_sequence,
            start_pose=start_pose,
            end_pose=end_pose,
            safe_approach_z=float(args.safe_approach_z),
            travel_feed=float(args.travel_feed),
            virtual_bbox=virtual_bbox,
            dwell_before_ms=int(args.dwell_before_ms),
            dwell_after_ms=int(args.dwell_after_ms),
            tracked_move_settle_s=float(args.tracked_move_settle_s),
            travel_move_settle_s=float(args.travel_move_settle_s),
            camera_flush_frames=int(args.camera_flush_frames),
            capture_at_start=bool(args.capture_at_start),
            capture_every_n_star_moves=int(args.capture_every_n_star_moves),
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
            "Run mirrored-half star tracking directly on the robot, preserving the "
            "motion planning of script 2 while keeping the camera/image-acquisition "
            "workflow of script 1, with offplane Y compensation enabled."
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

    # Kinematic sign overrides
    ap.add_argument("--flip-r-sign", action="store_true",
                    help="Flip the sign of the calibrated r(B) polynomial output.")
    ap.add_argument("--flip-z-sign", action="store_true",
                    help="Flip the sign of the calibrated z(B) polynomial output.")

    # Star placement (tip-space)
    ap.add_argument("--center-x", type=float, default=STAR_CENTER_X,
                    help="Nominal star center X in tip space.")
    ap.add_argument("--center-y", type=float, default=STAR_CENTER_Y,
                    help="Constant star center Y in tip space. Stage Y is solved to hold this exactly.")
    ap.add_argument("--center-z", type=float, default=STAR_CENTER_Z,
                    help="Star center Z in tip space.")

    # Star geometry
    ap.add_argument("--outer-radius", type=float, default=DESIRED_STAR_OUTER_RADIUS,
                    help="Desired outer star radius in mm (auto-scaled down if needed).")
    ap.add_argument("--inner-ratio", type=float, default=INNER_RADIUS_RATIO,
                    help="Inner radius / outer radius for 5-point star.")
    ap.add_argument("--rotation-deg", type=float, default=STAR_ROTATION_DEG,
                    help="Star rotation in XZ plane (deg).")
    ap.add_argument("--samples-per-edge", type=int, default=SAMPLES_PER_EDGE,
                    help="Interpolation points per star edge for motion generation before half-plane extraction.")
    ap.add_argument("--safety-margin", type=float, default=DEFAULT_SAFETY_MARGIN,
                    help="Margin subtracted from max reachable |r(B)| when auto-scaling.")

    # Motion / feeds
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED,
                    help="Feedrate for safe travel moves.")
    ap.add_argument("--jog-feed", type=float, default=DEFAULT_PRINT_FEED,
                    help="Feedrate for startup move to first tracked point.")
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED,
                    help="Base feedrate for drawing moves.")
    ap.add_argument("--print-feed-b", type=float, default=DEFAULT_PRINT_FEED_B,
                    help="Feedrate for drawing moves without C rotation.")
    ap.add_argument("--print-feed-c", type=float, default=DEFAULT_PRINT_FEED_C,
                    help="Feedrate for the single mirror-flip move that also changes C.")

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
    ap.add_argument("--tracked-move-settle-s", type=float, default=DEFAULT_TRACKED_MOVE_SETTLE_S,
                    help="Extra settle time after each tracked move, before capture.")
    ap.add_argument("--travel-move-settle-s", type=float, default=DEFAULT_TRAVEL_MOVE_SETTLE_S,
                    help="Extra settle time after travel moves.")
    ap.add_argument("--capture-every-n-star-moves", type=int, default=DEFAULT_CAPTURE_EVERY_N_STAR_MOVES,
                    help="Capture one image every N star-path moves. Travel moves and the mirror flip are not captured.")
    ap.add_argument("--capture-at-start", action="store_true", default=DEFAULT_CAPTURE_AT_START,
                    help="Also capture once at the first tracked sample after positioning there.")

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
