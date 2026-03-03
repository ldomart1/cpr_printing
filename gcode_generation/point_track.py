#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code to probe a fixed Cartesian tip point while moving C and B,
with exact tip-position tracking by moving X/Y/Z/B/C together.

Adds:
- TRUE sine / triangle / constant oscillation support for C
- Optional per-segment feed scheduling using a C-axis speed envelope:
    * --c-max-feed (deg/min) cap for C during coordinated tracked motion
    * --c-accel-time / --c-decel-time (seconds) ramp on C speed envelope
- Keeps --probe-feed as the baseline coordinated feed (e.g. 500)

Important note:
A single G1 move has one feedrate F. This script uses per-segment F scheduling to *approximate*
independent limits: it ensures the commanded segment time is long enough to satisfy both:
  (a) baseline path-feed from --probe-feed
  (b) C-axis speed cap + accel/decel envelope
This is the practical way to do it while keeping XYZ/B/C synchronized in standard G1 moves.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

# ---------------- Defaults (CLI-overridable) ----------------
DEFAULT_OUT = "gcode_generation/probe_point_c_motion_fixed_tip.gcode"

DEFAULT_POINT_X = 65.0
DEFAULT_POINT_Y = 40.0
DEFAULT_POINT_Z = -91.0

DEFAULT_TRAVEL_FEED = 1200.0
DEFAULT_PROBE_FEED = 500.0            # user requested baseline coordinated feed
DEFAULT_C_FEED = 10000.0              # C-only preposition feed
DEFAULT_C_MAX_FEED = 10000.0          # max C speed cap during tracked motion (deg/min)
DEFAULT_C_ACCEL_TIME_S = 1.0          # seconds to ramp C speed from low->max
DEFAULT_C_DECEL_TIME_S = 1.0          # seconds to ramp C speed max->low

DEFAULT_STEPS = 360
DEFAULT_C_START_DEG = 0.0
DEFAULT_C_SWEEP_DEG = 360.0

DEFAULT_C_PROFILE = "sweep"           # sweep or oscillate
DEFAULT_C_CENTER_DEG = 0.0
DEFAULT_C_AMP_DEG = 360.0
DEFAULT_C_CYCLES = 5
DEFAULT_C_WAVEFORM = "sine"           # restored true sine by default
DEFAULT_OSC_SAMPLES_PER_CYCLE = 120

DEFAULT_B_PROFILE = "triangle"

DEFAULT_START_X = 60.0
DEFAULT_START_Y = 40.0
DEFAULT_START_Z = -30.0
DEFAULT_START_B = 0.0
DEFAULT_START_C = 0.0
DEFAULT_END_X = 60.0
DEFAULT_END_Y = 40.0
DEFAULT_END_Z = -30.0
DEFAULT_END_B = 0.0
DEFAULT_END_C = 0.0
DEFAULT_SAFE_APPROACH_Z = 0.0

DEFAULT_DWELL_BEFORE_MS = 0
DEFAULT_DWELL_AFTER_MS = 0

DEFAULT_BBOX_X_MIN = 10.0
DEFAULT_BBOX_X_MAX = 110.0
DEFAULT_BBOX_Y_MIN = 10.0
DEFAULT_BBOX_Y_MAX = 110.0
DEFAULT_BBOX_Z_MIN = -120.0
DEFAULT_BBOX_Z_MAX = -20.0

OFFPLANE_SIGN = -1.0


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


# ---------------- Calibration / kinematics helpers ----------------

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


def unwrap_deg_near(angle_deg: float, ref_deg: Optional[float]) -> float:
    a = float(angle_deg)
    if ref_deg is None:
        return float(a % 360.0)
    k = round((float(ref_deg) - a) / 360.0)
    return float(a + 360.0 * k)


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
        pr=pr, pz=pz, py_off=py_off, pa=pa,
        b_min=b_min, b_max=b_max,
        x_axis=x_axis, y_axis=y_axis, z_axis=z_axis,
        b_axis=b_axis, c_axis=c_axis,
        c_180_deg=c_180,
        offplane_y_equation=cubic.get("offplane_y_equation"),
        offplane_y_r_squared=(None if cubic.get("offplane_y_r_squared") is None else float(cubic["offplane_y_r_squared"])),
    )


def eval_r(cal: Calibration, b: Any) -> np.ndarray:
    return poly_eval(cal.pr, b)


def eval_z(cal: Calibration, b: Any) -> np.ndarray:
    return poly_eval(cal.pz, b)


def eval_offplane_y(cal: Calibration, b: Any) -> np.ndarray:
    return OFFPLANE_SIGN * poly_eval(cal.py_off, b, default_if_none=0.0)


def predict_r_z_offplane(cal: Calibration, b: float) -> Tuple[float, float, float]:
    r = float(eval_r(cal, b))
    z = float(eval_z(cal, b))
    y_off = float(eval_offplane_y(cal, b))
    return r, z, y_off


def predict_tip_xyz_from_bc(cal: Calibration, b: float, c_deg: float) -> np.ndarray:
    r, z, y_off = predict_r_z_offplane(cal, b)
    c = math.radians(float(c_deg))
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def tip_offset_xyz_physical(cal: Calibration, b: float, c_deg: float) -> np.ndarray:
    return predict_tip_xyz_from_bc(cal, b, c_deg)


def stage_xyz_for_fixed_tip(cal: Calibration, p_tip_xyz: np.ndarray, b: float, c_deg: float) -> np.ndarray:
    return p_tip_xyz - tip_offset_xyz_physical(cal, b, c_deg)


# ---------------- Profiles ----------------

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def choose_b_peak_if_unspecified(cal: Calibration, b_start: float) -> float:
    d_to_min = abs(float(b_start) - float(cal.b_min))
    d_to_max = abs(float(cal.b_max) - float(b_start))
    return float(cal.b_min if d_to_min >= d_to_max else cal.b_max)


def b_profile_triangle(t: float, b_start: float, b_peak: float) -> float:
    if t <= 0.5:
        u = t / 0.5
        return (1.0 - u) * b_start + u * b_peak
    u = (t - 0.5) / 0.5
    return (1.0 - u) * b_peak + u * b_start


def b_profile_sine(t: float, b_start: float, b_peak: float) -> float:
    s = math.sin(math.pi * t)
    return float(b_start + (b_peak - b_start) * s)


def c_wave_sweep(t: float, c_start_deg: float, c_sweep_deg: float) -> float:
    return float(c_start_deg + c_sweep_deg * t)


def triangle_oscillation_unit(cycles_phase: float) -> float:
    frac = cycles_phase % 1.0
    if frac < 0.25:
        return frac / 0.25
    if frac < 0.5:
        return 1.0 - (frac - 0.25) / 0.25
    if frac < 0.75:
        return -(frac - 0.5) / 0.25
    return -1.0 + (frac - 0.75) / 0.25


def sine_oscillation_unit(cycles_phase: float) -> float:
    # Starts at 0 and rises positive
    return math.sin(2.0 * math.pi * cycles_phase)


def c_wave_oscillate(
    t: float,
    c_center_deg: float,
    c_amp_deg: float,
    c_cycles: float,
    waveform: str = "sine",
) -> float:
    phase_cycles = float(c_cycles) * float(t)
    wf = waveform.lower()
    if wf == "constant":
        w = triangle_oscillation_unit(phase_cycles)
    elif wf == "triangle":
        w = triangle_oscillation_unit(phase_cycles)
    elif wf == "sine":
        w = sine_oscillation_unit(phase_cycles)
    else:
        raise ValueError("--c-waveform must be one of: constant, sine, triangle")
    return float(c_center_deg + c_amp_deg * w)


def generate_motion_trajectory(
    cal: Calibration,
    p_tip_fixed: np.ndarray,
    c_profile: str,
    c_start_deg: float,
    c_sweep_deg: float,
    steps: int,
    c_center_deg: float,
    c_amp_deg: float,
    c_cycles: int,
    c_waveform: str,
    osc_samples_per_cycle: int,
    b_start: float,
    b_peak: float,
    b_profile_name: str,
    unwrap_c_continuous: bool = True,
) -> List[Tuple[float, float, np.ndarray]]:
    c_profile = c_profile.lower()
    if c_profile not in ("sweep", "oscillate"):
        raise ValueError("--c-profile must be 'sweep' or 'oscillate'")
    if b_profile_name.lower() not in ("triangle", "sine"):
        raise ValueError("--b-profile must be 'triangle' or 'sine'")

    if c_profile == "sweep":
        n_segments = max(2, int(steps))
    else:
        n_segments = max(4, int(max(1, c_cycles) * max(8, osc_samples_per_cycle)))

    pts: List[Tuple[float, float, np.ndarray]] = []
    prev_c: Optional[float] = None

    for i in range(n_segments + 1):
        t = i / float(n_segments)

        if c_profile == "sweep":
            c_nom = c_wave_sweep(t, c_start_deg=float(c_start_deg), c_sweep_deg=float(c_sweep_deg))
        else:
            c_nom = c_wave_oscillate(
                t,
                c_center_deg=float(c_center_deg),
                c_amp_deg=float(c_amp_deg),
                c_cycles=float(c_cycles),
                waveform=str(c_waveform),
            )

        c_cmd = unwrap_deg_near(c_nom, prev_c) if unwrap_c_continuous else float(c_nom % 360.0)

        if b_profile_name.lower() == "triangle":
            b_cmd = b_profile_triangle(t, b_start=float(b_start), b_peak=float(b_peak))
        else:
            b_cmd = b_profile_sine(t, b_start=float(b_start), b_peak=float(b_peak))

        p_stage = stage_xyz_for_fixed_tip(cal, p_tip_fixed, float(b_cmd), float(c_cmd))
        pts.append((float(b_cmd), float(c_cmd), p_stage))
        prev_c = float(c_cmd)

    return pts


# ---------------- Diagnostics ----------------

def sampled_ranges(cal: Calibration, b_lo: float, b_hi: float, n: int = 2001) -> dict:
    bb = np.linspace(b_lo, b_hi, n)
    rr = eval_r(cal, bb)
    yy = eval_offplane_y(cal, bb)
    zz = eval_z(cal, bb)
    rho = np.hypot(rr, yy)
    alpha = np.rad2deg(np.arctan2(yy, rr))
    return {
        "r_min": float(np.min(rr)),
        "r_max": float(np.max(rr)),
        "abs_r_max": float(np.max(np.abs(rr))),
        "yoff_min": float(np.min(yy)),
        "yoff_max": float(np.max(yy)),
        "rho_min": float(np.min(rho)),
        "rho_max": float(np.max(rho)),
        "alpha_min_deg": float(np.min(alpha)),
        "alpha_max_deg": float(np.max(alpha)),
        "z_min": float(np.min(zz)),
        "z_max": float(np.max(zz)),
    }


def compute_traj_meta(traj: List[Tuple[float, float, np.ndarray]]) -> dict:
    if not traj:
        return {
            "n_samples": 0, "n_segments": 0,
            "x_stage_min": 0.0, "x_stage_max": 0.0,
            "y_stage_min": 0.0, "y_stage_max": 0.0,
            "z_stage_min": 0.0, "z_stage_max": 0.0,
            "b_min_used": 0.0, "b_max_used": 0.0,
            "c_min_used": 0.0, "c_max_used": 0.0,
            "xyz_path_len_mm": 0.0, "max_dc_step_deg": 0.0,
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
        "x_stage_min": float(np.min(xyz[:, 0])), "x_stage_max": float(np.max(xyz[:, 0])),
        "y_stage_min": float(np.min(xyz[:, 1])), "y_stage_max": float(np.max(xyz[:, 1])),
        "z_stage_min": float(np.min(xyz[:, 2])), "z_stage_max": float(np.max(xyz[:, 2])),
        "b_min_used": float(np.min(bb)), "b_max_used": float(np.max(bb)),
        "c_min_used": float(np.min(cc)), "c_max_used": float(np.max(cc)),
        "xyz_path_len_mm": float(np.sum(seglens)) if len(seglens) else 0.0,
        "max_dc_step_deg": float(np.max(np.abs(dc))) if len(dc) else 0.0,
        "c_abs_path_deg": float(np.sum(np.abs(dc))) if len(dc) else 0.0,
    }


# ---------------- Feed scheduling ----------------

def _smoothstep01(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def _c_speed_envelope_factor(t01: float, accel_s: float, decel_s: float, total_s: float, floor_frac: float = 0.05) -> float:
    """
    Returns factor in [floor_frac, 1] for C max speed cap.
    Smooth ramp up over accel_s, hold, then smooth ramp down over decel_s.
    """
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
    """
    Compute a per-segment G-code feedrate (F) using a practical segment-time method.

    For each segment, choose dt >= max(
      XYZ path time from probe_feed,
      C time from C-speed cap * envelope_factor
    )
    Then set F = XYZ_path_len / dt (mm/min), clamped so we don't underflow on zero-XYZ segments.

    This keeps XYZ path feed near/below probe_feed and lets C run faster when possible,
    while honoring a start/end acceleration envelope on C.
    """
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

    # First-pass estimate of total time using plain caps (no envelope)
    dt_xyz0 = xyzlens / (probe_feed / 60.0)
    dt_c0 = dcs / (cmax / 60.0)
    dt0 = np.maximum(dt_xyz0, dt_c0)
    dt0 = np.maximum(dt0, min_seg_time_s)
    total_est = float(np.sum(dt0))

    # Second pass with C accel/decel envelope based on segment midpoint time
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

        # Convert chosen dt back to an F value for this segment.
        # If XYZ length is tiny/zero but C moves, F can be near zero and unhelpful; in that case
        # set F to probe_feed (controller still executes by geometry + its own limits).
        if xyzlens[i] > 1e-9:
            f_i = 60.0 * xyzlens[i] / dt
            # don't exceed probe_feed baseline on XYZ path
            f_i = min(f_i, probe_feed)
            # avoid tiny F values on tiny segments
            f_i = max(f_i, 1.0)
        else:
            f_i = probe_feed

        # Diagnostic estimated C speed
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


# ---------------- G-code emission ----------------

def _clamp_stage_xyz_to_bbox(
    x: float, y: float, z: float,
    bbox: dict,
    context: str,
    warn_log: List[str],
) -> Tuple[float, float, float]:
    def clamp_one(axis: str, value: float, lo: float, hi: float) -> float:
        if value < lo:
            warn_log.append(f"WARNING: {context} {axis}={value:.3f} below bbox min {lo:.3f}; clamped to {lo:.3f}")
            return lo
        if value > hi:
            warn_log.append(f"WARNING: {context} {axis}={value:.3f} above bbox max {hi:.3f}; clamped to {hi:.3f}")
            return hi
        return value

    xc = clamp_one("X", float(x), float(bbox["x_min"]), float(bbox["x_max"]))
    yc = clamp_one("Y", float(y), float(bbox["y_min"]), float(bbox["y_max"]))
    zc = clamp_one("Z", float(z), float(bbox["z_min"]), float(bbox["z_max"]))
    return xc, yc, zc


def write_gcode_fixed_tip_motion(
    out_path: str,
    cal: Calibration,
    p_tip_fixed: np.ndarray,
    traj: List[Tuple[float, float, np.ndarray]],
    travel_feed: float,
    probe_feed: float,
    c_feed: float,
    c_max_feed: float,
    c_accel_time_s: float,
    c_decel_time_s: float,
    header_meta: dict,
    start_pose: Tuple[float, float, float, float, float],
    end_pose: Tuple[float, float, float, float, float],
    safe_approach_z: float,
    virtual_bbox: dict,
    dwell_before_ms: int = 0,
    dwell_after_ms: int = 0,
    preposition_c_only: bool = True,
    use_segment_feed_scheduler: bool = True,
):
    bbox_warnings: List[str] = []

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

    with open(out_path, "w") as f:
        f.write("; generated by point_track.py (modified)\n")
        f.write("; fixed-tip exact tracking with coordinated XYZBC\n")
        f.write("; C waveform restored (sine/triangle/constant supported)\n")
        f.write("; tracked-motion feed scheduling enabled for C speed cap+envelope\n")
        f.write(f"; axes: X->{cal.x_axis}, Y->{cal.y_axis}, Z->{cal.z_axis}, B->{cal.b_axis}, C->{cal.c_axis}\n")
        f.write(f"; fixed tip point = [{float(p_tip_fixed[0]):.3f}, {float(p_tip_fixed[1]):.3f}, {float(p_tip_fixed[2]):.3f}]\n")
        f.write(f"; trajectory samples = {header_meta.get('n_samples', 0)} (segments={header_meta.get('n_segments', 0)})\n")
        f.write(f"; mode detail: {header_meta.get('mode_desc', 'unknown')}\n")
        f.write(f"; feeds: travel F{float(travel_feed):.1f}, probe baseline F{float(probe_feed):.1f}, C-only F{float(c_feed):.1f}\n")
        f.write(f"; C tracked cap: {float(c_max_feed):.1f} deg/min, accel={float(c_accel_time_s):.3f}s, decel={float(c_decel_time_s):.3f}s\n")
        f.write(f"; feed scheduler est total tracked time: {sched_meta['est_total_time_s']:.3f}s\n")
        f.write(f"; feed scheduler est max C speed: {sched_meta['max_est_c_speed_deg_min']:.1f} deg/min\n")
        f.write(f"; feed scheduler mean segment time: {sched_meta['mean_seg_time_ms']:.2f} ms\n")
        f.write("G90\n")

        # Safe startup
        sx, sy, sz, sb, sc = [float(v) for v in start_pose]
        f.write("; safe startup approach\n")
        f.write(f"G1 {cal.z_axis}{float(safe_approach_z):.3f} {cal.b_axis}{sb:.3f} {cal.c_axis}{sc:.3f} F{float(travel_feed):.0f}\n")
        f.write(f"G1 {cal.x_axis}{sx:.3f} {cal.y_axis}{sy:.3f} {cal.b_axis}{sb:.3f} {cal.c_axis}{sc:.3f} F{float(travel_feed):.0f}\n")
        f.write(f"G1 {cal.z_axis}{sz:.3f} {cal.b_axis}{sb:.3f} {cal.c_axis}{sc:.3f} F{float(travel_feed):.0f}\n")

        if not traj:
            f.write("; no trajectory points generated\n")
        else:
            b0, c0, p0_stage = traj[0]

            if preposition_c_only and abs(float(c0) - float(sc)) > 1e-9:
                f.write("; optional C-only pre-position move (tunable C feedrate)\n")
                f.write(f"G1 {cal.c_axis}{float(c0):.3f} F{float(c_feed):.0f}\n")

            x0, y0, z0 = _clamp_stage_xyz_to_bbox(
                p0_stage[0], p0_stage[1], p0_stage[2], virtual_bbox, "move to tracked start", bbox_warnings
            )
            f.write("; move to first tracked sample (tip fixed at probe point)\n")
            f.write(f"G1 {cal.x_axis}{x0:.3f} {cal.y_axis}{y0:.3f} {cal.z_axis}{z0:.3f} "
                    f"{cal.b_axis}{b0:.3f} {cal.c_axis}{c0:.3f} F{float(travel_feed):.0f}\n")

            if int(dwell_before_ms) > 0:
                f.write(f"G4 P{int(dwell_before_ms)} ; dwell before motion\n")

            f.write("; coordinated tracked motion (XYZ+B+C), per-segment feed scheduling\n")
            for i, (b, c, p_stage) in enumerate(traj[1:], start=1):
                x, y, z = _clamp_stage_xyz_to_bbox(
                    p_stage[0], p_stage[1], p_stage[2], virtual_bbox, f"tracked sample {i}", bbox_warnings
                )
                if seg_feeds:
                    fseg = seg_feeds[i - 1]
                else:
                    fseg = float(probe_feed)
                f.write(
                    f"G1 {cal.x_axis}{x:.3f} {cal.y_axis}{y:.3f} {cal.z_axis}{z:.3f} "
                    f"{cal.b_axis}{b:.3f} {cal.c_axis}{c:.3f} F{float(fseg):.1f}\n"
                )

            if int(dwell_after_ms) > 0:
                f.write(f"G4 P{int(dwell_after_ms)} ; dwell after motion\n")

        # Safe end
        ex, ey, ez, eb, ec = [float(v) for v in end_pose]
        f.write("; safe end move\n")
        f.write(f"G1 {cal.z_axis}{float(safe_approach_z):.3f} {cal.b_axis}{eb:.3f} {cal.c_axis}{ec:.3f} F{float(travel_feed):.0f}\n")
        f.write(f"G1 {cal.x_axis}{ex:.3f} {cal.y_axis}{ey:.3f} {cal.b_axis}{eb:.3f} {cal.c_axis}{ec:.3f} F{float(travel_feed):.0f}\n")
        f.write(f"G1 {cal.z_axis}{ez:.3f} {cal.b_axis}{eb:.3f} {cal.c_axis}{ec:.3f} F{float(travel_feed):.0f}\n")

        if bbox_warnings:
            f.write("; virtual bbox clamp warnings:\n")
            for msg in bbox_warnings:
                f.write(f"; {msg}\n")
        f.write(f"; bbox warning count = {len(bbox_warnings)}\n")
        f.write("; --- end ---\n")

    if bbox_warnings:
        print(f"Virtual bounding-box warnings: {len(bbox_warnings)} (values clamped)")
        for msg in bbox_warnings:
            print(msg)
    return sched_meta


# ---------------- Main ----------------

def main(args):
    cal = load_calibration(args.calibration)

    b_lo = cal.b_min if args.min_b is None else float(args.min_b)
    b_hi = cal.b_max if args.max_b is None else float(args.max_b)
    if b_lo > b_hi:
        b_lo, b_hi = b_hi, b_lo

    p_tip_fixed = np.array([float(args.point_x), float(args.point_y), float(args.point_z)], dtype=float)

    b_start = float(args.b_start) if args.b_start is not None else float(args.start_b)
    b_start = clamp(b_start, b_lo, b_hi)

    if args.b_peak is not None:
        b_peak = clamp(float(args.b_peak), b_lo, b_hi)
    else:
        b_peak = clamp(choose_b_peak_if_unspecified(cal, b_start), b_lo, b_hi)

    traj = generate_motion_trajectory(
        cal=cal,
        p_tip_fixed=p_tip_fixed,
        c_profile=str(args.c_profile),
        c_start_deg=float(args.c_start_deg),
        c_sweep_deg=float(args.c_sweep_deg),
        steps=int(args.steps),
        c_center_deg=float(args.c_center_deg),
        c_amp_deg=float(args.c_amp_deg),
        c_cycles=int(args.c_cycles),
        c_waveform=str(args.c_waveform),   # restored actual waveform selection
        osc_samples_per_cycle=int(args.osc_samples_per_cycle),
        b_start=float(b_start),
        b_peak=float(b_peak),
        b_profile_name=str(args.b_profile),
        unwrap_c_continuous=(not bool(args.wrap_c)),
    )

    meta = compute_traj_meta(traj)
    c_profile = str(args.c_profile).lower()
    if c_profile == "sweep":
        mode_desc = f"sweep: C from {float(args.c_start_deg):.3f} by {float(args.c_sweep_deg):.3f} deg"
    else:
        mode_desc = (
            f"oscillate: C = {float(args.c_center_deg):.3f} ± {float(args.c_amp_deg):.3f} deg, "
            f"{int(args.c_cycles)} cycles, waveform={str(args.c_waveform)}"
        )
    meta.update({"mode_desc": mode_desc, "b_start": float(b_start), "b_peak": float(b_peak)})

    start_pose = (
        float(args.start_x), float(args.start_y), float(args.start_z),
        float(args.start_b), float(args.start_c),
    )
    end_pose = (
        float(args.end_x), float(args.end_y), float(args.end_z),
        float(args.end_b), float(args.end_c),
    )

    virtual_bbox = {
        "x_min": float(args.bbox_x_min), "x_max": float(args.bbox_x_max),
        "y_min": float(args.bbox_y_min), "y_max": float(args.bbox_y_max),
        "z_min": float(args.bbox_z_min), "z_max": float(args.bbox_z_max),
    }
    if virtual_bbox["x_min"] > virtual_bbox["x_max"]:
        virtual_bbox["x_min"], virtual_bbox["x_max"] = virtual_bbox["x_max"], virtual_bbox["x_min"]
    if virtual_bbox["y_min"] > virtual_bbox["y_max"]:
        virtual_bbox["y_min"], virtual_bbox["y_max"] = virtual_bbox["y_max"], virtual_bbox["y_min"]
    if virtual_bbox["z_min"] > virtual_bbox["z_max"]:
        virtual_bbox["z_min"], virtual_bbox["z_max"] = virtual_bbox["z_max"], virtual_bbox["z_min"]

    sched_meta = write_gcode_fixed_tip_motion(
        out_path=args.out,
        cal=cal,
        p_tip_fixed=p_tip_fixed,
        traj=traj,
        travel_feed=float(args.travel_feed),
        probe_feed=float(args.probe_feed),
        c_feed=float(args.c_feed),
        c_max_feed=float(args.c_max_feed),
        c_accel_time_s=float(args.c_accel_time),
        c_decel_time_s=float(args.c_decel_time),
        header_meta=meta,
        start_pose=start_pose,
        end_pose=end_pose,
        safe_approach_z=float(args.safe_approach_z),
        virtual_bbox=virtual_bbox,
        dwell_before_ms=int(args.dwell_before_ms),
        dwell_after_ms=int(args.dwell_after_ms),
        preposition_c_only=(not bool(args.no_c_preposition)),
        use_segment_feed_scheduler=(not bool(args.disable_segment_feed_scheduler)),
    )

    ranges = sampled_ranges(cal, b_lo, b_hi)

    print(f"Wrote {args.out}")
    print("Mode: fixed-point probe with exact tip tracking while moving C and B")
    print(f"Fixed tip point (Cartesian): [{p_tip_fixed[0]:.3f}, {p_tip_fixed[1]:.3f}, {p_tip_fixed[2]:.3f}]")
    print(f"Calibration B range: [{cal.b_min:.3f}, {cal.b_max:.3f}]")
    print(f"Commanded B bounds: [{b_lo:.3f}, {b_hi:.3f}]")
    print(f"Planned B start/peak/start: {b_start:.3f} -> {b_peak:.3f} -> {b_start:.3f}   profile={args.b_profile}")
    print(f"C mode: {meta['mode_desc']}")
    print(f"Samples: {meta['n_samples']} (segments={meta['n_segments']})")
    print(f"Feeds: travel={float(args.travel_feed):.1f} mm/min, probe baseline={float(args.probe_feed):.1f}, "
          f"C-only={float(args.c_feed):.1f}, C-tracked-cap={float(args.c_max_feed):.1f} deg/min")
    print(f"C accel/decel envelope: accel={float(args.c_accel_time):.3f}s decel={float(args.c_decel_time):.3f}s")
    print(f"Estimated tracked time: {sched_meta.get('est_total_time_s', 0.0):.3f}s")
    print(f"Estimated max C speed (scheduled): {sched_meta.get('max_est_c_speed_deg_min', 0.0):.1f} deg/min")
    print(f"Mean segment time: {sched_meta.get('mean_seg_time_ms', 0.0):.2f} ms")

    print(f"Sampled r(B): [{ranges['r_min']:.3f}, {ranges['r_max']:.3f}] mm   |r| max={ranges['abs_r_max']:.3f}")
    print(f"Sampled y_off(B): [{ranges['yoff_min']:.3f}, {ranges['yoff_max']:.3f}] mm")
    print(f"Sampled transverse rho(B): [{ranges['rho_min']:.3f}, {ranges['rho_max']:.3f}] mm")
    print(f"Sampled transverse alpha(B): [{ranges['alpha_min_deg']:.3f}, {ranges['alpha_max_deg']:.3f}] deg")
    print(f"Sampled z(B): [{ranges['z_min']:.3f}, {ranges['z_max']:.3f}] mm")

    print(f"B used range: [{meta['b_min_used']:.3f}, {meta['b_max_used']:.3f}]")
    print(f"C used range: [{meta['c_min_used']:.3f}, {meta['c_max_used']:.3f}]")
    print(f"Total |ΔC| path: {meta['c_abs_path_deg']:.3f} deg")
    print(f"X gantry range used: [{meta['x_stage_min']:.3f}, {meta['x_stage_max']:.3f}]")
    print(f"Y gantry range used: [{meta['y_stage_min']:.3f}, {meta['y_stage_max']:.3f}]")
    print(f"Z gantry range used: [{meta['z_stage_min']:.3f}, {meta['z_stage_max']:.3f}]")
    print(f"Approx XYZ tracked path length: {meta['xyz_path_len_mm']:.3f} mm")
    print(f"Max per-step ΔC: {meta['max_dc_step_deg']:.3f} deg")

    print("\nNotes:")
    print("  - --probe-feed is the baseline coordinated path feed.")
    print("  - --c-feed applies only to optional C-only preposition.")
    print("  - --c-max-feed + accel/decel are enforced via per-segment feed scheduling (best practical approximation with G1).")
    print("  - If you need exact time-law execution, the next step is inverse-time feed (G93) or firmware-side motion control.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Generate G-code to hold a fixed Cartesian tip point while moving C (sweep/oscillate) and "
            "curling/uncurling B, with exact X/Y/Z/B/C coordinated tracking."
        )
    )
    ap.add_argument("--calibration", required=True, help="Path to calibration JSON.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code file.")

    # Fixed tip point
    ap.add_argument("--point-x", type=float, default=DEFAULT_POINT_X, help="Fixed tip X (Cartesian/world).")
    ap.add_argument("--point-y", type=float, default=DEFAULT_POINT_Y, help="Fixed tip Y (Cartesian/world).")
    ap.add_argument("--point-z", type=float, default=DEFAULT_POINT_Z, help="Fixed tip Z (Cartesian/world).")

    # C motion mode
    ap.add_argument("--c-profile", choices=["sweep", "oscillate"], default=DEFAULT_C_PROFILE,
                    help="C motion mode: single sweep or oscillation.")
    ap.add_argument("--c-start-deg", type=float, default=DEFAULT_C_START_DEG, help="Sweep mode start C angle (deg).")
    ap.add_argument("--c-sweep-deg", type=float, default=DEFAULT_C_SWEEP_DEG, help="Sweep mode total C sweep (deg).")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="Sweep mode segments.")

    ap.add_argument("--c-center-deg", type=float, default=DEFAULT_C_CENTER_DEG, help="Oscillate mode center angle (deg).")
    ap.add_argument("--c-amp-deg", type=float, default=DEFAULT_C_AMP_DEG, help="Oscillate mode amplitude (deg), center ± amp.")
    ap.add_argument("--c-cycles", type=int, default=DEFAULT_C_CYCLES, help="Oscillate mode number of cycles.")
    ap.add_argument("--c-waveform", choices=["constant", "sine", "triangle"], default=DEFAULT_C_WAVEFORM,
                    help="Oscillate waveform. 'sine' is real sine again. 'constant' uses triangle/constant-speed approximation.")
    ap.add_argument("--osc-samples-per-cycle", type=int, default=DEFAULT_OSC_SAMPLES_PER_CYCLE,
                    help="Oscillate mode segments per cycle.")
    ap.add_argument("--wrap-c", action="store_true", help="Wrap C to [0,360) instead of continuous/unwrapped.")

    # B profile
    ap.add_argument("--b-profile", choices=["triangle", "sine"], default=DEFAULT_B_PROFILE,
                    help="B motion profile over full motion.")
    ap.add_argument("--b-start", type=float, default=None, help="B at motion start/end. Default: --start-b (clamped).")
    ap.add_argument("--b-peak", type=float, default=None, help="B at motion midpoint. Default: farther B boundary.")
    ap.add_argument("--min-b", type=float, default=None, help="Lower commanded B bound.")
    ap.add_argument("--max-b", type=float, default=None, help="Upper commanded B bound.")

    # Feedrates
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED, help="Feedrate for non-tracked travel moves.")
    ap.add_argument("--probe-feed", type=float, default=DEFAULT_PROBE_FEED,
                    help="Baseline coordinated feed for tracked XYZ+B+C motion (mm/min path feed).")
    ap.add_argument("--c-feed", type=float, default=DEFAULT_C_FEED,
                    help="Feedrate for optional C-only pre-position move (deg/min).")

    # New tracked C cap + accel/decel envelope
    ap.add_argument("--c-max-feed", type=float, default=DEFAULT_C_MAX_FEED,
                    help="Max C speed during tracked motion (deg/min), enforced via per-segment feed scheduling.")
    ap.add_argument("--c-accel-time", type=float, default=DEFAULT_C_ACCEL_TIME_S,
                    help="Seconds for C speed envelope acceleration at start of tracked motion.")
    ap.add_argument("--c-decel-time", type=float, default=DEFAULT_C_DECEL_TIME_S,
                    help="Seconds for C speed envelope deceleration at end of tracked motion.")
    ap.add_argument("--disable-segment-feed-scheduler", action="store_true",
                    help="Disable per-segment feed scheduling (tracked motion uses constant --probe-feed).")

    ap.add_argument("--no-c-preposition", action="store_true",
                    help="Disable optional C-only pre-position move to first trajectory C.")

    ap.add_argument("--safe-approach-z", type=float, default=DEFAULT_SAFE_APPROACH_Z,
                    help="Safe Z used before XY startup/end positioning.")

    # Optional dwells
    ap.add_argument("--dwell-before-ms", type=int, default=DEFAULT_DWELL_BEFORE_MS, help="Dwell before motion (ms).")
    ap.add_argument("--dwell-after-ms", type=int, default=DEFAULT_DWELL_AFTER_MS, help="Dwell after motion (ms).")

    # Startup / end poses
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

    args = ap.parse_args()
    main(args)