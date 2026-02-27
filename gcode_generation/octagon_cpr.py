#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code to trace a regular octagon in the XZ plane
using calibration coefficients exported by shadow_calibration.

Kinematic model (same as circle_cpr.py):
  X_tip = X_stage + sgn * r(B)
  Z_tip = Z_stage + z(B)

where:
  - sgn is +1 when C==0 deg, and -1 when C==180 deg
  - r(B), z(B), tip_angle_deg(B) are cubic polynomials from calibration JSON

This script differs from circle_cpr.py by deriving the target tip angle from the
local path tangent on each octagon edge so you can validate the tip-angle fit.

Important convention:
  - calibration tip_angle_deg is 0..180 and encodes angle vs vertical only
  - left/right sign is encoded by C (0/180)
  - by default, the *tip->base* direction is matched to the path tangent
    (equivalently, the motion direction is opposite the tangent vector used for
    the angle decomposition). This mirrors the circle generator behavior.
"""

import argparse
import math
from typing import Optional, Tuple

import numpy as np

from circle_cpr import (
    Calibration,
    _polyval4,
    compute_r_range,
    invert_tip_angle_to_b,
    load_calibration,
)


# ---------------- Defaults (CLI-overridable) ----------------
OCTAGON_CENTER_X = 65.0
OCTAGON_CENTER_Z = -130.0
DESIRED_CIRCUMRADIUS = 25.0   # mm, radius to octagon vertices in TIP space
N_OCTAGONS = 1
SAMPLES_PER_EDGE = 40
DEFAULT_OUT = "octagon_xzb_cppr.gcode"
DEFAULT_JOG_FEED = 200.0
DEFAULT_PRINT_FEED = 200.0
DEFAULT_MIN_B = -5.0
DEFAULT_MAX_B = -0.0
# ------------------------------------------------------------


def _normalize(vx: float, vz: float) -> Tuple[float, float]:
    n = math.hypot(vx, vz)
    if n <= 0.0:
        return 0.0, 1.0
    return vx / n, vz / n


def make_octagon_tip_trajectory(
    x_center: float,
    z_center: float,
    circumradius: float,
    n_octagons: int,
    samples_per_edge: int,
    start_theta: float = math.pi,
    ccw: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      tip_xz: (N,2) desired TIP points in XZ
      tan_xz: (N,2) unit tangent vectors (path direction) in XZ
    """
    if samples_per_edge < 1:
        raise ValueError("samples_per_edge must be >= 1")

    direction = 1.0 if ccw else -1.0
    dtheta = direction * (2.0 * math.pi / 8.0)

    pts_all = []
    tans_all = []

    for k in range(n_octagons):
        theta0 = start_theta + direction * (2.0 * math.pi * k)
        verts = []
        for i in range(8):
            th = theta0 + i * dtheta
            verts.append((
                x_center + circumradius * math.cos(th),
                z_center + circumradius * math.sin(th),
            ))
        verts = np.asarray(verts, dtype=float)

        for i in range(8):
            p0 = verts[i]
            p1 = verts[(i + 1) % 8]
            edge = p1 - p0
            tx, tz = _normalize(float(edge[0]), float(edge[1]))
            tvals = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)
            seg_pts = p0[None, :] + tvals[:, None] * edge[None, :]
            seg_tan = np.tile(np.array([[tx, tz]], dtype=float), (samples_per_edge, 1))
            pts_all.append(seg_pts)
            tans_all.append(seg_tan)

    return np.vstack(pts_all), np.vstack(tans_all)


def tangent_to_tip_angle_and_c(
    tan_x: float,
    tan_z: float,
    c0_deg: float,
    c180_deg: float,
    tip_follows_motion: bool = False,
) -> Tuple[float, float]:
    """
    Convert a path tangent in machine XZ to (C command, tip_angle_deg target).

    Default behavior matches the circle script convention more closely by using
    the tip->base direction opposite the motion tangent.
    """
    vx, vz = _normalize(tan_x, tan_z)
    if not tip_follows_motion:
        vx, vz = -vx, -vz

    # tip_angle_deg from calibration is 0..180 vs vertical (+Z up in machine frame):
    # 0 = +Z, 90 = horizontal, 180 = -Z
    tip_angle_deg = math.degrees(math.acos(max(-1.0, min(1.0, vz))))
    c_cmd = float(c0_deg if vx >= 0.0 else c180_deg)
    return tip_angle_deg, c_cmd


def generate_gcode_octagon_xzb(
    tip_xz: np.ndarray,
    tan_xz: np.ndarray,
    cal: Calibration,
    out_path: str,
    jog_feed: float,
    print_feed: float,
    c0_deg: float = 0.0,
    c180_deg: Optional[float] = None,
    min_b: Optional[float] = None,
    max_b: Optional[float] = None,
    tip_follows_motion: bool = False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    c180 = float(cal.c_180_deg if c180_deg is None else c180_deg)

    b_lo = cal.b_min if min_b is None else float(min_b)
    b_hi = cal.b_max if max_b is None else float(max_b)
    if b_lo > b_hi:
        b_lo, b_hi = b_hi, b_lo

    N = tip_xz.shape[0]
    if tan_xz.shape[0] != N:
        raise ValueError("tip_xz and tan_xz must have the same length")

    x_stage = np.zeros(N, dtype=float)
    z_stage = np.zeros(N, dtype=float)
    b_cmd = np.zeros(N, dtype=float)
    c_cmd = np.zeros(N, dtype=float)
    angle_cmd = np.zeros(N, dtype=float)

    b_prev: Optional[float] = None
    n_angle_clamped = 0

    for i in range(N):
        x_tip = float(tip_xz[i, 0])
        z_tip = float(tip_xz[i, 1])
        tx = float(tan_xz[i, 0])
        tz = float(tan_xz[i, 1])

        tip_angle_target, c_i = tangent_to_tip_angle_and_c(
            tx, tz, c0_deg=c0_deg, c180_deg=c180, tip_follows_motion=tip_follows_motion
        )
        c_cmd[i] = c_i

        tip_angle_limited = float(np.clip(tip_angle_target, cal.tip_angle_min, cal.tip_angle_max))
        if tip_angle_limited != tip_angle_target:
            n_angle_clamped += 1
        angle_cmd[i] = tip_angle_limited

        b_i = invert_tip_angle_to_b(tip_angle_limited, cal, b_prev=b_prev)
        b_i = float(np.clip(b_i, b_lo, b_hi))
        b_cmd[i] = b_i
        b_prev = b_i

        z_curl = float(_polyval4(cal.pz, b_i))
        x_r = float(_polyval4(cal.pr, b_i))
        sgn = +1.0 if c_i == float(c0_deg) else -1.0

        x_stage[i] = x_tip - sgn * x_r
        z_stage[i] = z_tip - z_curl

    with open(out_path, "w") as f:
        f.write("; generated by octagon_cpr.py\n")
        f.write(f"; axes: X→{cal.x_axis}, Z→{cal.z_axis}, pull→{cal.pull_axis}, rot→{cal.rot_axis}\n")
        f.write("; model:\n")
        f.write(";   X_tip = X_stage + sgn*r(B)   (sgn=+1 at C=0, sgn=-1 at C=180)\n")
        f.write(";   Z_tip = Z_stage + z(B)\n")
        f.write(";   tip_angle_deg = tip_angle(B)  (0..180 vs vertical)\n")
        f.write("; path: regular octagon in TIP XZ, tangent-based tip-angle targeting per edge.\n")
        f.write(f"; tangent convention: {'motion' if tip_follows_motion else 'tip->base opposite motion'}\n")
        rmin, rmax = compute_r_range(cal)
        f.write(f"; calibrated r(B) range (sampled): [{rmin:.3f}, {rmax:.3f}] mm\n")
        f.write(f"; calibrated tip-angle range: [{cal.tip_angle_min:.3f}, {cal.tip_angle_max:.3f}] deg\n")
        if n_angle_clamped > 0:
            f.write(f"; info: {n_angle_clamped}/{N} tangent-angle targets were clamped to calibration range.\n")
        f.write(f"; B clamp: [{b_lo:.3f}, {b_hi:.3f}]\n")
        f.write("G90\n")

        # Safe initial positioning.
        f.write(f"G1 {cal.rot_axis}{c_cmd[0]:.3f} F{(10.0*jog_feed):.0f}\n")
        f.write(
            f"G1 {cal.x_axis}{x_stage[0]:.3f} {cal.z_axis}{z_stage[0]:.3f} "
            f"{cal.pull_axis}{b_cmd[0]:.3f} F{jog_feed:.0f}\n"
        )

        # Use synchronized moves so C switches happen at the commanded point index.
        for i in range(1, N):
            f.write(
                f"G1 {cal.x_axis}{x_stage[i]:.3f} {cal.z_axis}{z_stage[i]:.3f} "
                f"{cal.pull_axis}{b_cmd[i]:.3f} {cal.rot_axis}{c_cmd[i]:.3f} F{print_feed:.0f}\n"
            )

    return x_stage, z_stage, b_cmd, c_cmd, angle_cmd, n_angle_clamped


def main(args: argparse.Namespace) -> None:
    cal = load_calibration(args.calibration)

    tip_xz, tan_xz = make_octagon_tip_trajectory(
        x_center=args.center_x,
        z_center=args.center_z,
        circumradius=args.radius,
        n_octagons=args.octagons,
        samples_per_edge=args.samples_per_edge,
        start_theta=args.start_theta_deg * math.pi / 180.0,
        ccw=not args.cw,
    )

    x_stage, z_stage, b_cmd, c_cmd, angle_cmd, n_angle_clamped = generate_gcode_octagon_xzb(
        tip_xz=tip_xz,
        tan_xz=tan_xz,
        cal=cal,
        out_path=args.out,
        jog_feed=args.jog_feed,
        print_feed=args.print_feed,
        c0_deg=args.c0_deg,
        c180_deg=args.c180_deg,
        min_b=args.min_b,
        max_b=args.max_b,
        tip_follows_motion=args.tip_follows_motion,
    )

    print(f"Wrote {args.out} with {len(b_cmd)} points.")
    print(f"Axes used: X={cal.x_axis}, Z={cal.z_axis}, B={cal.pull_axis}, C={cal.rot_axis}")
    print(f"C values used: {sorted(set(np.round(c_cmd, 6).tolist()))}")
    print(f"B range used: [{b_cmd.min():.3f}, {b_cmd.max():.3f}] (cal: [{cal.b_min:.3f}, {cal.b_max:.3f}])")
    print(f"tip-angle targets used: [{angle_cmd.min():.3f}, {angle_cmd.max():.3f}] deg")
    if n_angle_clamped > 0:
        print(
            f"[info] tangent-angle target saturated at {n_angle_clamped}/{len(b_cmd)} points "
            f"to [{cal.tip_angle_min:.3f}, {cal.tip_angle_max:.3f}] deg."
        )
    rr = _polyval4(cal.pr, b_cmd)
    aa = _polyval4(cal.pa, b_cmd)
    print(f"r(B) range used: [{rr.min():.3f}, {rr.max():.3f}] mm")
    print(f"tip-angle(B) range used: [{aa.min():.3f}, {aa.max():.3f}] deg")
    print(f"X stage range: [{x_stage.min():.3f}, {x_stage.max():.3f}]")
    print(f"Z stage range: [{z_stage.min():.3f}, {z_stage.max():.3f}]")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate X+Z+B(+C) G-code to trace a regular octagon in the XZ plane using calibration JSON."
    )
    ap.add_argument("--calibration", required=True, help="Path to calibration JSON (shadow_calibration schema).")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code filepath.")
    ap.add_argument("--center-x", type=float, default=OCTAGON_CENTER_X, help="Octagon center X in TIP space.")
    ap.add_argument("--center-z", type=float, default=OCTAGON_CENTER_Z, help="Octagon center Z in TIP space.")
    ap.add_argument("--radius", type=float, default=DESIRED_CIRCUMRADIUS, help="Octagon circumradius (center to vertex) in TIP space.")
    ap.add_argument("--octagons", type=int, default=N_OCTAGONS, help="Number of octagon loops.")
    ap.add_argument("--samples-per-edge", type=int, default=SAMPLES_PER_EDGE, help="Points emitted per octagon edge.")
    ap.add_argument("--start-theta-deg", type=float, default=180.0, help="Angle of first vertex in degrees (0=+X, 90=+Z).")
    ap.add_argument("--cw", action="store_true", default=False, help="Trace clockwise instead of CCW.")
    ap.add_argument("--jog-feed", type=float, default=DEFAULT_JOG_FEED, help="Jog feedrate for first positioning move (units/min).")
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED, help="Feedrate for octagon trace moves (units/min).")

    ap.add_argument("--tip-follows-motion", action="store_true", default=False,
                    help="Use the path motion tangent directly for tip-angle targeting (default uses opposite/tip->base direction).")

    # Optional overrides
    ap.add_argument("--min-b", type=float, default=None, help="Lower bound for commanded B (default: from calibration).")
    ap.add_argument("--max-b", type=float, default=None, help="Upper bound for commanded B (default: from calibration).")
    ap.add_argument("--c0-deg", type=float, default=0.0, help="C value used for +X tip/base direction (default 0).")
    ap.add_argument("--c180-deg", type=float, default=None, help="C value used for -X tip/base direction (default from calibration, else 180).")

    main(ap.parse_args())
