#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code to probe a fixed Cartesian tip point while moving C and B,
with exact tip-position tracking by moving X/Y/Z/B/C together.

Supports:
- single C sweep (legacy behavior)
- true C oscillation (back/forth) around a center angle for N cycles
- B curl/uncurl over the total motion
- exact tip tracking:
      p_stage = p_tip_fixed - offset_tip(B, C)
  where
      offset_tip(B,C) = [r(B) cos(C), r(B) sin(C), z(B)]

Notes
-----
- Physical tip kinematics only (no tangency/nozzle-direction model)
- No extrusion/U-axis commands
- probe-feed controls coordinated XYZBC moves
- c-feed controls optional C-only positioning moves (e.g., pre-positioning C)
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

# Fixed tip point in world/cartesian space
DEFAULT_POINT_X = 65.0
DEFAULT_POINT_Y = 40.0
DEFAULT_POINT_Z = -85.0

# Motion
DEFAULT_TRAVEL_FEED = 1200.0
DEFAULT_PROBE_FEED = 40.0            # coordinated XYZBC move feed (user requested)
DEFAULT_C_FEED = 1000.0              # C-only move feed (user requested tunable example)

# Sampling / trajectory
DEFAULT_STEPS = 360                  # used for sweep mode (segments)
DEFAULT_C_START_DEG = 0.0
DEFAULT_C_SWEEP_DEG = 360.0

# C motion profile
DEFAULT_C_PROFILE = "sweep"          # sweep or oscillate
DEFAULT_C_CENTER_DEG = 0.0           # oscillation center
DEFAULT_C_AMP_DEG = 360.0            # oscillation amplitude (±amp around center)
DEFAULT_C_CYCLES = 10                # number of oscillation cycles
DEFAULT_C_WAVEFORM = "triangle"          # sine or triangle
DEFAULT_OSC_SAMPLES_PER_CYCLE = 120  # total segments per cycle (increase for smoother)

# B motion during the full path
DEFAULT_B_PROFILE = "triangle"       # triangle or sine

# Startup/end poses (stage axes)
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

# Optional dwell at probe point before/after motion
DEFAULT_DWELL_BEFORE_MS = 0
DEFAULT_DWELL_AFTER_MS = 0

# Virtual XYZ bounding box (enforced during motion)
DEFAULT_BBOX_X_MIN = 10.0
DEFAULT_BBOX_X_MAX = 110.0
DEFAULT_BBOX_Y_MIN = 10.0
DEFAULT_BBOX_Y_MAX = 110.0
DEFAULT_BBOX_Z_MIN = -120.0
DEFAULT_BBOX_Z_MAX = -20.0
# ------------------------------------------------------------


@dataclass
class Calibration:
    pr: np.ndarray            # r(B) coeffs
    pz: np.ndarray            # z(B) coeffs

    b_min: float
    b_max: float

    x_axis: str
    y_axis: str
    z_axis: str
    b_axis: str
    c_axis: str

    c_180_deg: float


# ---------------- Calibration / kinematics helpers ----------------

def _polyval4(coeffs: Any, u: Any) -> np.ndarray:
    a, b, c, d = coeffs
    u = np.asarray(u, dtype=float)
    return ((a * u + b) * u + c) * u + d


def unwrap_deg_near(angle_deg: float, ref_deg: Optional[float]) -> float:
    """Shift angle by k*360 to be nearest ref_deg."""
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

    pr = np.array(data["cubic_coefficients"]["r_coeffs"], dtype=float)
    pz = np.array(data["cubic_coefficients"]["z_coeffs"], dtype=float)

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
        pr=pr, pz=pz,
        b_min=b_min, b_max=b_max,
        x_axis=x_axis, y_axis=y_axis, z_axis=z_axis,
        b_axis=b_axis, c_axis=c_axis,
        c_180_deg=c_180,
    )


def eval_r(cal: Calibration, b: Any) -> np.ndarray:
    return _polyval4(cal.pr, b)


def eval_z(cal: Calibration, b: Any) -> np.ndarray:
    return _polyval4(cal.pz, b)


def tip_offset_xyz_physical(cal: Calibration, b: float, c_deg: float) -> np.ndarray:
    """Physical tip offset from stage origin for a given B/C: [r(B)cosC, r(B)sinC, z(B)]"""
    r = float(eval_r(cal, b))
    z = float(eval_z(cal, b))
    c = math.radians(float(c_deg))
    return np.array([r * math.cos(c), r * math.sin(c), z], dtype=float)


def stage_xyz_for_fixed_tip(cal: Calibration, p_tip_xyz: np.ndarray, b: float, c_deg: float) -> np.ndarray:
    """Exact stage position required so the tip stays at p_tip_xyz."""
    return p_tip_xyz - tip_offset_xyz_physical(cal, b, c_deg)


# ---------------- Profiles ----------------

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def choose_b_peak_if_unspecified(cal: Calibration, b_start: float) -> float:
    """Choose the farther calibration boundary from b_start."""
    d_to_min = abs(float(b_start) - float(cal.b_min))
    d_to_max = abs(float(cal.b_max) - float(b_start))
    return float(cal.b_min if d_to_min >= d_to_max else cal.b_max)


def b_profile_triangle(t: float, b_start: float, b_peak: float) -> float:
    """0->start, 0.5->peak, 1->start"""
    if t <= 0.5:
        u = t / 0.5
        return (1.0 - u) * b_start + u * b_peak
    u = (t - 0.5) / 0.5
    return (1.0 - u) * b_peak + u * b_start


def b_profile_sine(t: float, b_start: float, b_peak: float) -> float:
    """Smooth 0->start, 0.5->peak, 1->start"""
    s = math.sin(math.pi * t)
    return float(b_start + (b_peak - b_start) * s)


def c_wave_sweep(t: float, c_start_deg: float, c_sweep_deg: float) -> float:
    """Single sweep from start to start+sweep over t in [0,1]."""
    return float(c_start_deg + c_sweep_deg * t)


def triangle_wave_unit(cycles_phase: float) -> float:
    """
    Triangle wave in [-1,1], where cycles_phase is in cycles (e.g., 0..10 for 10 cycles).
    """
    frac = cycles_phase % 1.0
    # piecewise: 0->0, 0.25->1, 0.5->0, 0.75->-1, 1->0
    if frac < 0.25:
        return frac / 0.25
    if frac < 0.5:
        return 1.0 - (frac - 0.25) / 0.25
    if frac < 0.75:
        return - (frac - 0.5) / 0.25
    return -1.0 + (frac - 0.75) / 0.25


def c_wave_oscillate(
    t: float,
    c_center_deg: float,
    c_amp_deg: float,
    c_cycles: float,
    waveform: str = "sine",
) -> float:
    """
    Oscillate around center:
      C(t) = center + amp * wave(2*pi*cycles*t)
    where wave is sine or triangle in [-1,1].
    """
    phase_cycles = float(c_cycles) * float(t)
    if waveform == "sine":
        w = math.sin(2.0 * math.pi * phase_cycles)
    elif waveform == "triangle":
        w = triangle_wave_unit(phase_cycles)
    else:
        raise ValueError("--c-waveform must be 'sine' or 'triangle'")
    return float(c_center_deg + c_amp_deg * w)


def generate_motion_trajectory(
    cal: Calibration,
    p_tip_fixed: np.ndarray,
    c_profile: str,
    # sweep params
    c_start_deg: float,
    c_sweep_deg: float,
    steps: int,
    # oscillation params
    c_center_deg: float,
    c_amp_deg: float,
    c_cycles: int,
    c_waveform: str,
    osc_samples_per_cycle: int,
    # B profile over the total motion
    b_start: float,
    b_peak: float,
    b_profile_name: str,
    # angle continuity
    unwrap_c_continuous: bool = True,
) -> List[Tuple[float, float, np.ndarray]]:
    """
    Returns samples (including endpoints):
        (b_cmd, c_cmd, stage_xyz)
    """
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

        # C command
        if c_profile == "sweep":
            c_nom = c_wave_sweep(t, c_start_deg=float(c_start_deg), c_sweep_deg=float(c_sweep_deg))
        else:
            c_nom = c_wave_oscillate(
                t,
                c_center_deg=float(c_center_deg),
                c_amp_deg=float(c_amp_deg),
                c_cycles=float(c_cycles),
                waveform=str(c_waveform).lower(),
            )

        c_cmd = unwrap_deg_near(c_nom, prev_c) if unwrap_c_continuous else float(c_nom % 360.0)

        # B command (spans entire motion, curl then uncurl once)
        if b_profile_name.lower() == "triangle":
            b_cmd = b_profile_triangle(t, b_start=float(b_start), b_peak=float(b_peak))
        else:
            b_cmd = b_profile_sine(t, b_start=float(b_start), b_peak=float(b_peak))

        # Exact stage tracking for fixed tip
        p_stage = stage_xyz_for_fixed_tip(cal, p_tip_fixed, float(b_cmd), float(c_cmd))
        pts.append((float(b_cmd), float(c_cmd), p_stage))
        prev_c = float(c_cmd)

    return pts


# ---------------- Diagnostics ----------------

def sampled_ranges(cal: Calibration, b_lo: float, b_hi: float, n: int = 2001) -> dict:
    bb = np.linspace(b_lo, b_hi, n)
    rr = eval_r(cal, bb)
    zz = eval_z(cal, bb)
    return {
        "r_min": float(np.min(rr)),
        "r_max": float(np.max(rr)),
        "abs_r_max": float(np.max(np.abs(rr))),
        "z_min": float(np.min(zz)),
        "z_max": float(np.max(zz)),
    }


def compute_traj_meta(traj: List[Tuple[float, float, np.ndarray]]) -> dict:
    if not traj:
        return {
            "n_samples": 0,
            "n_segments": 0,
            "x_stage_min": 0.0, "x_stage_max": 0.0,
            "y_stage_min": 0.0, "y_stage_max": 0.0,
            "z_stage_min": 0.0, "z_stage_max": 0.0,
            "b_min_used": 0.0, "b_max_used": 0.0,
            "c_min_used": 0.0, "c_max_used": 0.0,
            "xyz_path_len_mm": 0.0,
            "max_dc_step_deg": 0.0,
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
    }


# ---------------- G-code emission ----------------

def _fmt_axes_move(axes_vals: List[Tuple[str, float]]) -> str:
    return " ".join(f"{ax}{val:.3f}" for ax, val in axes_vals)


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
    header_meta: dict,
    start_pose: Tuple[float, float, float, float, float],
    end_pose: Tuple[float, float, float, float, float],
    safe_approach_z: float,
    virtual_bbox: dict,
    dwell_before_ms: int = 0,
    dwell_after_ms: int = 0,
    preposition_c_only: bool = True,
):
    """
    Emit G-code for fixed-tip coordinated XYZBC motion.
    - probe_feed: feed for coordinated XYZBC trajectory
    - c_feed: feed for optional C-only positioning moves (e.g. pre-positioning C)
    """
    bbox_warnings: List[str] = []

    with open(out_path, "w") as f:
        f.write("; generated by probe_point_c_motion_fixed_tip.py\n")
        f.write("; mode: fixed-point probe with exact tip tracking while moving C and B\n")
        f.write("; exact tip tracking: stage = tip_fixed - [r(B)cos(C), r(B)sin(C), z(B)]\n")
        f.write(f"; axes: X->{cal.x_axis}, Y->{cal.y_axis}, Z->{cal.z_axis}, B->{cal.b_axis}, C->{cal.c_axis}\n")
        f.write(f"; fixed tip point = [{float(p_tip_fixed[0]):.3f}, {float(p_tip_fixed[1]):.3f}, {float(p_tip_fixed[2]):.3f}]\n")
        f.write(f"; trajectory samples = {header_meta.get('n_samples', 0)} (segments={header_meta.get('n_segments', 0)})\n")
        f.write(f"; mode detail: {header_meta.get('mode_desc', 'unknown')}\n")
        f.write(f"; B used range = [{header_meta.get('b_min_used', 0.0):.3f}, {header_meta.get('b_max_used', 0.0):.3f}]\n")
        f.write(f"; C used range = [{header_meta.get('c_min_used', 0.0):.3f}, {header_meta.get('c_max_used', 0.0):.3f}]\n")
        f.write(f"; X gantry range used = [{header_meta.get('x_stage_min', 0.0):.3f}, {header_meta.get('x_stage_max', 0.0):.3f}]\n")
        f.write(f"; Y gantry range used = [{header_meta.get('y_stage_min', 0.0):.3f}, {header_meta.get('y_stage_max', 0.0):.3f}]\n")
        f.write(f"; Z gantry range used = [{header_meta.get('z_stage_min', 0.0):.3f}, {header_meta.get('z_stage_max', 0.0):.3f}]\n")
        f.write(f"; approx XYZ path length = {header_meta.get('xyz_path_len_mm', 0.0):.3f} mm\n")
        f.write(f"; max per-step ΔC = {header_meta.get('max_dc_step_deg', 0.0):.3f} deg\n")
        f.write(f"; virtual bbox (enforced): "
                f"X[{virtual_bbox['x_min']:.3f},{virtual_bbox['x_max']:.3f}] "
                f"Y[{virtual_bbox['y_min']:.3f},{virtual_bbox['y_max']:.3f}] "
                f"Z[{virtual_bbox['z_min']:.3f},{virtual_bbox['z_max']:.3f}]\n")
        f.write(f"; feeds: travel F{float(travel_feed):.1f}, probe/coordinated F{float(probe_feed):.1f}, C-only F{float(c_feed):.1f}\n")
        f.write("G90\n")

        # Safe startup move (Z first, then XY, then dive)
        sx, sy, sz, sb, sc = [float(v) for v in start_pose]
        f.write("; safe startup approach\n")
        f.write(f"G1 {cal.z_axis}{float(safe_approach_z):.3f} {cal.b_axis}{sb:.3f} {cal.c_axis}{sc:.3f} F{float(travel_feed):.0f}\n")
        f.write(f"G1 {cal.x_axis}{sx:.3f} {cal.y_axis}{sy:.3f} {cal.b_axis}{sb:.3f} {cal.c_axis}{sc:.3f} F{float(travel_feed):.0f}\n")
        f.write(f"G1 {cal.z_axis}{sz:.3f} {cal.b_axis}{sb:.3f} {cal.c_axis}{sc:.3f} F{float(travel_feed):.0f}\n")

        if not traj:
            f.write("; no trajectory points generated\n")
        else:
            b0, c0, p0_stage = traj[0]

            # Optional C-only preposition from current start C to first trajectory C.
            # This does NOT hold the tip fixed during the move (it's just a staging move),
            # but gives the user a tunable C-axis feedrate.
            if preposition_c_only and abs(float(c0) - float(sc)) > 1e-9:
                f.write("; optional C-only pre-position move (uses tunable C feedrate)\n")
                f.write(f"G1 {cal.c_axis}{float(c0):.3f} F{float(c_feed):.0f}\n")

            # Move to first tracked sample (exact tip fixed)
            x0, y0, z0 = _clamp_stage_xyz_to_bbox(
                p0_stage[0], p0_stage[1], p0_stage[2],
                virtual_bbox, "move to tracked start", bbox_warnings
            )
            f.write("; move to first tracked sample (tip fixed at probe point)\n")
            f.write(f"G1 {cal.x_axis}{x0:.3f} {cal.y_axis}{y0:.3f} {cal.z_axis}{z0:.3f} "
                    f"{cal.b_axis}{b0:.3f} {cal.c_axis}{c0:.3f} F{float(travel_feed):.0f}\n")

            if int(dwell_before_ms) > 0:
                f.write(f"G4 P{int(dwell_before_ms)} ; dwell before motion\n")

            f.write("; coordinated motion with exact fixed-tip tracking (XYZ+B+C)\n")
            for i, (b, c, p_stage) in enumerate(traj[1:], start=1):
                x, y, z = _clamp_stage_xyz_to_bbox(
                    p_stage[0], p_stage[1], p_stage[2],
                    virtual_bbox, f"tracked sample {i}", bbox_warnings
                )
                # Important: this is coordinated XYZBC, so probe_feed applies to the whole move
                f.write(f"G1 {cal.x_axis}{x:.3f} {cal.y_axis}{y:.3f} {cal.z_axis}{z:.3f} "
                        f"{cal.b_axis}{b:.3f} {cal.c_axis}{c:.3f} F{float(probe_feed):.0f}\n")

            if int(dwell_after_ms) > 0:
                f.write(f"G4 P{int(dwell_after_ms)} ; dwell after motion\n")

        # Safe end move
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


# ---------------- Main ----------------

def main(args):
    cal = load_calibration(args.calibration)

    # B command bounds
    b_lo = cal.b_min if args.min_b is None else float(args.min_b)
    b_hi = cal.b_max if args.max_b is None else float(args.max_b)
    if b_lo > b_hi:
        b_lo, b_hi = b_hi, b_lo

    p_tip_fixed = np.array([float(args.point_x), float(args.point_y), float(args.point_z)], dtype=float)

    # B start / peak
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
        c_waveform=str(args.c_waveform),
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
        mode_desc = (f"oscillate: C = {float(args.c_center_deg):.3f} ± {float(args.c_amp_deg):.3f} deg, "
                     f"{int(args.c_cycles)} cycles, waveform={str(args.c_waveform)}")
    meta.update({
        "mode_desc": mode_desc,
        "b_start": float(b_start),
        "b_peak": float(b_peak),
    })

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

    write_gcode_fixed_tip_motion(
        out_path=args.out,
        cal=cal,
        p_tip_fixed=p_tip_fixed,
        traj=traj,
        travel_feed=float(args.travel_feed),
        probe_feed=float(args.probe_feed),
        c_feed=float(args.c_feed),
        header_meta=meta,
        start_pose=start_pose,
        end_pose=end_pose,
        safe_approach_z=float(args.safe_approach_z),
        virtual_bbox=virtual_bbox,
        dwell_before_ms=int(args.dwell_before_ms),
        dwell_after_ms=int(args.dwell_after_ms),
        preposition_c_only=(not bool(args.no_c_preposition)),
    )

    ranges = sampled_ranges(cal, b_lo, b_hi)

    # Diagnostics
    print(f"Wrote {args.out}")
    print("Mode: fixed-point probe with exact tip tracking while moving C and B")
    print(f"Fixed tip point (Cartesian): [{p_tip_fixed[0]:.3f}, {p_tip_fixed[1]:.3f}, {p_tip_fixed[2]:.3f}]")
    print(f"Calibration B range: [{cal.b_min:.3f}, {cal.b_max:.3f}]")
    print(f"Commanded B bounds: [{b_lo:.3f}, {b_hi:.3f}]")
    print(f"Planned B start/peak/start: {b_start:.3f} -> {b_peak:.3f} -> {b_start:.3f}   profile={args.b_profile}")
    print(f"C mode: {meta['mode_desc']}")
    print(f"Samples: {meta['n_samples']} (segments={meta['n_segments']})")
    print(f"Feeds: travel={float(args.travel_feed):.3f} mm/min, probe/coordinated={float(args.probe_feed):.3f} mm/min, C-only={float(args.c_feed):.3f} mm/min")

    print(f"Sampled r(B): [{ranges['r_min']:.3f}, {ranges['r_max']:.3f}] mm   |r| max={ranges['abs_r_max']:.3f}")
    print(f"Sampled z(B): [{ranges['z_min']:.3f}, {ranges['z_max']:.3f}] mm")

    print(f"B used range: [{meta['b_min_used']:.3f}, {meta['b_max_used']:.3f}]")
    print(f"C used range: [{meta['c_min_used']:.3f}, {meta['c_max_used']:.3f}]")
    print(f"X gantry range used: [{meta['x_stage_min']:.3f}, {meta['x_stage_max']:.3f}]")
    print(f"Y gantry range used: [{meta['y_stage_min']:.3f}, {meta['y_stage_max']:.3f}]")
    print(f"Z gantry range used: [{meta['z_stage_min']:.3f}, {meta['z_stage_max']:.3f}]")
    print(f"Approx XYZ tracked path length: {meta['xyz_path_len_mm']:.3f} mm")
    print(f"Max per-step ΔC: {meta['max_dc_step_deg']:.3f} deg")

    print(f"User start pose (stage): [{start_pose[0]:.3f}, {start_pose[1]:.3f}, {start_pose[2]:.3f}, {start_pose[3]:.3f}, {start_pose[4]:.3f}]")
    print(f"User end pose (stage):   [{end_pose[0]:.3f}, {end_pose[1]:.3f}, {end_pose[2]:.3f}, {end_pose[3]:.3f}, {end_pose[4]:.3f}]")
    print(f"Safe approach Z: {float(args.safe_approach_z):.3f}")
    print(f"Virtual bbox (enforced): X[{virtual_bbox['x_min']:.3f}, {virtual_bbox['x_max']:.3f}] "
          f"Y[{virtual_bbox['y_min']:.3f}, {virtual_bbox['y_max']:.3f}] "
          f"Z[{virtual_bbox['z_min']:.3f}, {virtual_bbox['z_max']:.3f}]")

    print("\nImportant feedrate note:")
    print("  - probe-feed applies to coordinated XYZ+B+C moves (the fixed-tip tracked motion).")
    print("  - c-feed applies only to optional C-only pre-position moves.")
    print("  - If you need a true angular-speed target during coordinated motion, we can add time-based resampling.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Generate G-code to hold a fixed Cartesian tip point while moving C (sweep or oscillate) and "
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
    # Sweep mode params
    ap.add_argument("--c-start-deg", type=float, default=DEFAULT_C_START_DEG,
                    help="Sweep mode: starting C angle (deg).")
    ap.add_argument("--c-sweep-deg", type=float, default=DEFAULT_C_SWEEP_DEG,
                    help="Sweep mode: total C sweep (deg).")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS,
                    help="Sweep mode: number of segments over the sweep.")
    # Oscillate mode params
    ap.add_argument("--c-center-deg", type=float, default=DEFAULT_C_CENTER_DEG,
                    help="Oscillate mode: center angle (deg).")
    ap.add_argument("--c-amp-deg", type=float, default=DEFAULT_C_AMP_DEG,
                    help="Oscillate mode: amplitude (deg), giving center ± amp.")
    ap.add_argument("--c-cycles", type=int, default=DEFAULT_C_CYCLES,
                    help="Oscillate mode: number of cycles.")
    ap.add_argument("--c-waveform", choices=["sine", "triangle"], default=DEFAULT_C_WAVEFORM,
                    help="Oscillate mode waveform.")
    ap.add_argument("--osc-samples-per-cycle", type=int, default=DEFAULT_OSC_SAMPLES_PER_CYCLE,
                    help="Oscillate mode: segments per cycle (higher = smoother).")
    ap.add_argument("--wrap-c", action="store_true",
                    help="Wrap C to [0,360) instead of keeping continuous/unwrapped C values.")

    # B curl/uncurl profile over total motion
    ap.add_argument("--b-profile", choices=["triangle", "sine"], default=DEFAULT_B_PROFILE,
                    help="B motion profile over the full motion (curl then uncurl once).")
    ap.add_argument("--b-start", type=float, default=None,
                    help="B at motion start/end. Default: --start-b (clamped).")
    ap.add_argument("--b-peak", type=float, default=None,
                    help="B at motion midpoint. Default: farther B boundary from b-start.")
    ap.add_argument("--min-b", type=float, default=None,
                    help="Lower commanded B bound (default: calibration).")
    ap.add_argument("--max-b", type=float, default=None,
                    help="Upper commanded B bound (default: calibration).")

    # Feedrates
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED,
                    help="Feedrate for non-tracked travel moves (mm/min).")
    ap.add_argument("--probe-feed", type=float, default=DEFAULT_PROBE_FEED,
                    help="Feedrate for coordinated XYZ+B+C tracked motion (mm/min).")
    ap.add_argument("--c-feed", type=float, default=DEFAULT_C_FEED,
                    help="Feedrate for optional C-only pre-position moves (mm/min).")
    ap.add_argument("--no-c-preposition", action="store_true",
                    help="Disable optional C-only pre-position move to first trajectory C.")

    ap.add_argument("--safe-approach-z", type=float, default=DEFAULT_SAFE_APPROACH_Z,
                    help="Safe Z used before XY startup/end positioning, then dive to requested Z.")

    # Optional dwells
    ap.add_argument("--dwell-before-ms", type=int, default=DEFAULT_DWELL_BEFORE_MS,
                    help="Dwell at probe point before motion (ms).")
    ap.add_argument("--dwell-after-ms", type=int, default=DEFAULT_DWELL_AFTER_MS,
                    help="Dwell at probe point after motion (ms).")

    # Startup / end poses (stage axes)
    ap.add_argument("--start-x", type=float, default=DEFAULT_START_X, help="Startup stage X target.")
    ap.add_argument("--start-y", type=float, default=DEFAULT_START_Y, help="Startup stage Y target.")
    ap.add_argument("--start-z", type=float, default=DEFAULT_START_Z, help="Startup stage Z target after safe approach.")
    ap.add_argument("--start-b", type=float, default=DEFAULT_START_B, help="Startup B target.")
    ap.add_argument("--start-c", type=float, default=DEFAULT_START_C, help="Startup C target.")

    ap.add_argument("--end-x", type=float, default=DEFAULT_END_X, help="End stage X target.")
    ap.add_argument("--end-y", type=float, default=DEFAULT_END_Y, help="End stage Y target.")
    ap.add_argument("--end-z", type=float, default=DEFAULT_END_Z, help="End stage Z target after safe approach.")
    ap.add_argument("--end-b", type=float, default=DEFAULT_END_B, help="End B target.")
    ap.add_argument("--end-c", type=float, default=DEFAULT_END_C, help="End C target.")

    # Virtual stage-space bounding box
    ap.add_argument("--bbox-x-min", type=float, default=DEFAULT_BBOX_X_MIN, help="Virtual bbox min X.")
    ap.add_argument("--bbox-x-max", type=float, default=DEFAULT_BBOX_X_MAX, help="Virtual bbox max X.")
    ap.add_argument("--bbox-y-min", type=float, default=DEFAULT_BBOX_Y_MIN, help="Virtual bbox min Y.")
    ap.add_argument("--bbox-y-max", type=float, default=DEFAULT_BBOX_Y_MAX, help="Virtual bbox max Y.")
    ap.add_argument("--bbox-z-min", type=float, default=DEFAULT_BBOX_Z_MIN, help="Virtual bbox min Z.")
    ap.add_argument("--bbox-z-max", type=float, default=DEFAULT_BBOX_Z_MAX, help="Virtual bbox max Z.")

    args = ap.parse_args()
    main(args)