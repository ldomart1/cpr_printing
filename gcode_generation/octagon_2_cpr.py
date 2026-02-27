#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code to trace a *planar orthogonal “octagon”* in the XZ plane
(8 edges, all perfectly horizontal/vertical), while keeping the tip angle tangent to the
current edge direction.

Tip-angle convention (per user):
  tip_angle = 0° means the tip points toward -Z.
  Positive angles rotate toward +X (i.e., +X=90°, +Z=180°, -X=270°).

Kinematic model:
  X_tip = X_stage + sgn * r(B)
  Z_tip = Z_stage + z(B)

where:
  - sgn is +1 when C==c0_deg, and -1 when C==c180_deg
  - r(B), z(B), tip_angle_deg(B) are cubic polynomials from calibration JSON

"Tangent to edge direction":
  For each segment direction (dx,dz), we compute the heading angle alpha such that:
      alpha = 0   for -Z
      alpha = 90  for +X
      alpha = 180 for +Z
      alpha = 270 for -X
  Then we map alpha into:
      if alpha < 180:  C=c0_deg,   tip_angle=alpha
      else:            C=c180_deg, tip_angle=alpha-180
  so requested tip_angle is always in [0,180].

Shape:
  An 8-edge closed polyline with axis-aligned edges (a “staircase” loop):
      +X, +Z, +X, +Z, -X, -Z, -X, -Z
  This is a great stress-test for calibration because direction changes are abrupt and
  C may flip at the 180 boundary, but the geometry stays purely horizontal/vertical.

You control overall size with:
  --dx (horizontal run per step)
  --dz (vertical run per step)
  The resulting bounding box is width=2*dx and height=2*dz centered about (center_x, center_z).

Corners:
  Pure polyline (sharp corners). Increase sampling with --samples-per-edge.
"""

import json
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional
from pathlib import Path

import numpy as np
from numpy.typing import ArrayLike

try:
    from scipy.optimize import brentq  # type: ignore
except Exception:
    def brentq(f, a, b, maxiter=100, xtol=1e-10, rtol=4*np.finfo(float).eps):
        fa = f(a); fb = f(b)
        if np.sign(fa) == np.sign(fb):
            raise ValueError("Root not bracketed in [a,b].")
        x0, x1, f0, f1 = a, b, fa, fb
        for _ in range(maxiter):
            x2 = x1 - f1*(x1-x0)/(f1-f0) if f1 != f0 else 0.5*(x0+x1)
            if not (min(x0, x1) <= x2 <= max(x0, x1)):
                x2 = 0.5*(x0+x1)
            f2 = f(x2)
            if abs(f2) < max(xtol, rtol*abs(x2)):
                return x2
            if np.sign(f2) == np.sign(f0):
                x0, f0 = x2, f2
            else:
                x1, f1 = x2, f2
        return x2

# ---------------- Defaults (CLI-overridable) ----------------
CENTER_X = 65.0
CENTER_Z = -130.0
DX = 12.0                 # mm: horizontal run for each step
DZ = 12.0                 # mm: vertical run for each step
N_LOOPS = 1
SAMPLES_PER_EDGE = 80
DEFAULT_OUT = "orth_octagon_xzb_cppr.gcode"
DEFAULT_JOG_FEED = 300.0
DEFAULT_PRINT_FEED = 300.0
DEFAULT_MIN_B = -5.0
DEFAULT_MAX_B = -0.0
# ------------------------------------------------------------

@dataclass
class Calibration:
    pr: np.ndarray
    pz: np.ndarray
    pa: np.ndarray
    b_min: float
    b_max: float
    tip_angle_min: float
    tip_angle_max: float
    pull_axis: str
    rot_axis: str
    x_axis: str
    z_axis: str
    c_180_deg: float


def _polyval4(coeffs: ArrayLike, u: ArrayLike) -> np.ndarray:
    a, b, c, d = coeffs
    u = np.asarray(u)
    return ((a*u + b)*u + c)*u + d


def load_calibration(json_path: str) -> Calibration:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")

    with p.open("r") as f:
        data = json.load(f)

    pr = np.array(data["cubic_coefficients"]["r_coeffs"], dtype=float)
    pz = np.array(data["cubic_coefficients"]["z_coeffs"], dtype=float)
    pa = np.array(data["cubic_coefficients"]["tip_angle_coeffs"], dtype=float)

    motor_setup = data.get("motor_setup", {})
    duet_map = data.get("duet_axis_mapping", {})

    b_range = motor_setup.get("b_motor_position_range", [DEFAULT_MIN_B, DEFAULT_MAX_B])
    b_min, b_max = map(float, b_range)
    if b_min > b_max:
        b_min, b_max = b_max, b_min

    pull_axis = str(duet_map.get("pull_axis") or motor_setup.get("b_motor_axis") or "B")
    rot_axis  = str(duet_map.get("rotation_axis") or motor_setup.get("rotation_axis") or "C")
    x_axis    = str(duet_map.get("horizontal_axis") or motor_setup.get("horizontal_axis") or "X")
    z_axis    = str(duet_map.get("vertical_axis") or motor_setup.get("vertical_axis") or "Z")

    c_180 = float(motor_setup.get("rotation_axis_180_deg", 180.0))

    if pr.shape[0] != 4 or pz.shape[0] != 4 or pa.shape[0] != 4:
        raise ValueError("Expected 4 coeffs for r_coeffs, z_coeffs, and tip_angle_coeffs")

    tip_env = data.get("working_envelope", {}).get("tip_angle_range_deg")
    if isinstance(tip_env, list) and len(tip_env) == 2:
        tip_angle_min = float(min(tip_env))
        tip_angle_max = float(max(tip_env))
    else:
        bb = np.linspace(b_min, b_max, 801)
        aa = _polyval4(pa, bb)
        tip_angle_min = float(np.min(aa))
        tip_angle_max = float(np.max(aa))

    return Calibration(
        pr=pr, pz=pz, pa=pa,
        b_min=b_min, b_max=b_max,
        tip_angle_min=tip_angle_min, tip_angle_max=tip_angle_max,
        pull_axis=pull_axis, rot_axis=rot_axis,
        x_axis=x_axis, z_axis=z_axis,
        c_180_deg=c_180,
    )


def _find_roots_for_target(coeffs: np.ndarray, target: float, b_min: float, b_max: float) -> List[float]:
    f = lambda b: float(_polyval4(coeffs, b) - target)
    bs = np.linspace(b_min, b_max, 600)
    fs = np.array([f(b) for b in bs], dtype=float)
    fs = np.where(np.isfinite(fs), fs, 1e9)

    roots: List[float] = []
    for a, b, fa, fb in zip(bs[:-1], bs[1:], fs[:-1], fs[1:]):
        if np.sign(fa) == 0:
            return [float(a)]
        if np.sign(fa) != np.sign(fb):
            try:
                roots.append(float(brentq(f, float(a), float(b), maxiter=100)))
            except Exception:
                roots.append(float(0.5*(float(a) + float(b))))
    return [r for r in roots if np.isfinite(r)]


def invert_tip_angle_to_b(angle_target_deg: float, cal: Calibration, b_prev: Optional[float] = None) -> float:
    f = lambda b: float(_polyval4(cal.pa, b) - angle_target_deg)
    roots = _find_roots_for_target(cal.pa, angle_target_deg, cal.b_min, cal.b_max)
    if not roots:
        b0, b1 = cal.b_min, cal.b_max
        return float(b0 if abs(f(b0)) < abs(f(b1)) else b1)
    if len(roots) == 1 or b_prev is None:
        return float(roots[0])
    arr = np.array(roots, dtype=float)
    return float(arr[np.argmin(np.abs(arr - float(b_prev)))])


def alpha_from_segment_negZ_zero(dx: float, dz: float) -> float:
    """
    Return heading alpha in degrees where:
      0   = -Z
      90  = +X
      180 = +Z
      270 = -X
    For a segment vector (dx,dz) in XZ.
    """
    # Using atan2 with arguments (dx, -dz) gives 0 when dx=0,dz<0 (-Z).
    a = (math.degrees(math.atan2(dx, -dz)) + 360.0) % 360.0
    return a


def heading_to_C_and_tip_angle(alpha_deg_0_360: float, c0_deg: float, c180_deg: float) -> Tuple[float, float, float]:
    """
    Map alpha (0..360) into:
      - C command (either c0 or c180)
      - sgn (+1 for c0, -1 for c180)
      - tip_angle_target in [0,180]
    """
    a = float(alpha_deg_0_360) % 360.0
    if a < 180.0:
        return float(c0_deg), +1.0, a
    else:
        return float(c180_deg), -1.0, (a - 180.0)


def make_orthogonal_octagon_tip_trajectory(
    x_center: float,
    z_center: float,
    dx: float,
    dz: float,
    n_loops: int,
    samples_per_edge: int,
    ccw: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    8 axis-aligned edges in a closed loop. Starts at bottom-left corner of the bounding box.

    If ccw=True, the edge sequence is:
      +X, +Z, +X, +Z, -X, -Z, -X, -Z

    If ccw=False, the reverse:
      +Z, +X, +Z, +X, -Z, -X, -Z, -X

    Returns:
      tip_xz: (N,2) points
      alpha_deg: (N,) tangent heading per point
    """
    dx = float(abs(dx))
    dz = float(abs(dz))

    # Build vertices for a “staircase” rectangle centered at (x_center,z_center)
    # Bounding box is width=2*dx and height=2*dz.
    x0 = x_center - dx
    x1 = x_center
    x2 = x_center + dx
    z0 = z_center - dz
    z1 = z_center
    z2 = z_center + dz

    if ccw:
        verts = np.array([
            [x0, z0],  # start
            [x1, z0],
            [x1, z1],
            [x2, z1],
            [x2, z2],
            [x1, z2],
            [x1, z1],
            [x0, z1],
            [x0, z0],  # close
        ], dtype=float)
    else:
        # CW traversal of the same loop
        verts = np.array([
            [x0, z0],
            [x0, z1],
            [x1, z1],
            [x1, z2],
            [x2, z2],
            [x2, z1],
            [x1, z1],
            [x1, z0],
            [x0, z0],
        ], dtype=float)

    pts_all = []
    alpha_all = []

    for _ in range(n_loops):
        for i in range(len(verts) - 1):
            p0 = verts[i]
            p1 = verts[i + 1]
            d = p1 - p0
            seg_len = float(np.hypot(d[0], d[1]))
            if seg_len < 1e-12:
                continue

            alpha = alpha_from_segment_negZ_zero(float(d[0]), float(d[1]))

            t = np.linspace(0.0, 1.0, samples_per_edge, endpoint=False)
            seg_pts = p0[None, :] + t[:, None] * d[None, :]
            pts_all.append(seg_pts)
            alpha_all.append(np.full(seg_pts.shape[0], alpha, dtype=float))

    return np.vstack(pts_all), np.concatenate(alpha_all)


def generate_gcode_orth_octagon_xzb(
    tip_xz: np.ndarray,
    alpha_deg: np.ndarray,
    cal: Calibration,
    out_path: str,
    jog_feed: float,
    print_feed: float,
    c0_deg: float = 0.0,
    c180_deg: Optional[float] = None,
    min_b: Optional[float] = None,
    max_b: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    c180 = float(cal.c_180_deg if c180_deg is None else c180_deg)

    b_lo = cal.b_min if min_b is None else float(min_b)
    b_hi = cal.b_max if max_b is None else float(max_b)
    if b_lo > b_hi:
        b_lo, b_hi = b_hi, b_lo

    N = tip_xz.shape[0]
    x_stage = np.zeros(N, dtype=float)
    z_stage = np.zeros(N, dtype=float)
    b_cmd   = np.zeros(N, dtype=float)
    c_cmd   = np.zeros(N, dtype=float)

    b_prev: Optional[float] = None
    n_angle_clamped = 0

    for i in range(N):
        x_tip = float(tip_xz[i, 0])
        z_tip = float(tip_xz[i, 1])

        C_i, sgn, tip_angle_target = heading_to_C_and_tip_angle(
            float(alpha_deg[i]), c0_deg=float(c0_deg), c180_deg=float(c180)
        )
        c_cmd[i] = C_i

        tip_angle_limited = float(np.clip(tip_angle_target, cal.tip_angle_min, cal.tip_angle_max))
        if tip_angle_limited != tip_angle_target:
            n_angle_clamped += 1

        b_i = invert_tip_angle_to_b(tip_angle_limited, cal, b_prev=b_prev)
        b_i = float(np.clip(b_i, b_lo, b_hi))
        b_cmd[i] = b_i
        b_prev = b_i

        z_curl = float(_polyval4(cal.pz, b_i))
        x_r    = float(_polyval4(cal.pr, b_i))

        x_stage[i] = x_tip - sgn * x_r
        z_stage[i] = z_tip - z_curl

    with open(out_path, "w") as f:
        f.write("; generated by orth_octagon_xzb_from_calibration.py\n")
        f.write(f"; axes: X→{cal.x_axis}, Z→{cal.z_axis}, pull→{cal.pull_axis}, rot→{cal.rot_axis}\n")
        f.write("; model:\n")
        f.write(";   X_tip = X_stage + sgn*r(B)   (sgn=+1 at C=c0, sgn=-1 at C=c180)\n")
        f.write(";   Z_tip = Z_stage + z(B)\n")
        f.write("; tip-angle convention:\n")
        f.write(";   0° = -Z, 90° = +X, 180° = +Z, 270° = -X\n")
        f.write("; tangent mapping:\n")
        f.write(";   if alpha<180:  C=c0,   tip_angle=alpha\n")
        f.write(";   else:          C=c180, tip_angle=alpha-180\n")
        f.write(f"; calibrated tip-angle range: [{cal.tip_angle_min:.3f}, {cal.tip_angle_max:.3f}] deg\n")
        if n_angle_clamped > 0:
            f.write(f"; info: {n_angle_clamped}/{N} points requested outside calibrated tip-angle range and were clamped.\n")
        f.write(f"; B clamp: [{b_lo:.3f}, {b_hi:.3f}]\n")
        f.write("G90\n")

        f.write(f"G1 {cal.rot_axis}{c_cmd[0]:.3f} F{(10.0*jog_feed):.0f}\n")
        f.write(
            f"G1 {cal.x_axis}{x_stage[0]:.3f} {cal.z_axis}{z_stage[0]:.3f} "
            f"{cal.pull_axis}{b_cmd[0]:.3f} F{jog_feed:.0f}\n"
        )

        c_prev = c_cmd[0]
        for i in range(1, N):
            if c_cmd[i] != c_prev:
                f.write(f"G1 {cal.rot_axis}{c_cmd[i]:.3f} F{(10.0*jog_feed):.0f}\n")
                c_prev = c_cmd[i]
            f.write(
                f"G1 {cal.x_axis}{x_stage[i]:.3f} {cal.z_axis}{z_stage[i]:.3f} "
                f"{cal.pull_axis}{b_cmd[i]:.3f} F{print_feed:.0f}\n"
            )

    return x_stage, z_stage, b_cmd, c_cmd, n_angle_clamped


def main(args):
    cal = load_calibration(args.calibration)

    tip_xz, alpha_deg = make_orthogonal_octagon_tip_trajectory(
        x_center=args.center_x,
        z_center=args.center_z,
        dx=args.dx,
        dz=args.dz,
        n_loops=args.loops,
        samples_per_edge=args.samples_per_edge,
        ccw=not args.cw,
    )

    x_stage, z_stage, b_cmd, c_cmd, n_angle_clamped = generate_gcode_orth_octagon_xzb(
        tip_xz=tip_xz,
        alpha_deg=alpha_deg,
        cal=cal,
        out_path=args.out,
        jog_feed=args.jog_feed,
        print_feed=args.print_feed,
        c0_deg=args.c0_deg,
        c180_deg=args.c180_deg,
        min_b=args.min_b,
        max_b=args.max_b,
    )

    print(f"Wrote {args.out} with {len(b_cmd)} points.")
    print(f"Axes used: X={cal.x_axis}, Z={cal.z_axis}, B={cal.pull_axis}, C={cal.rot_axis}")
    print(f"C values used: {sorted(set(np.round(c_cmd, 6).tolist()))}")
    print(f"B range used: [{b_cmd.min():.3f}, {b_cmd.max():.3f}] (cal: [{cal.b_min:.3f}, {cal.b_max:.3f}])")
    if n_angle_clamped > 0:
        print(f"[info] tip-angle target saturated at {n_angle_clamped}/{len(b_cmd)} points "
              f"to [{cal.tip_angle_min:.3f}, {cal.tip_angle_max:.3f}] deg.")
    rr = _polyval4(cal.pr, b_cmd)
    aa = _polyval4(cal.pa, b_cmd)
    print(f"r(B) range used: [{rr.min():.3f}, {rr.max():.3f}] mm")
    print(f"tip-angle(B) range used: [{aa.min():.3f}, {aa.max():.3f}] deg")
    print(f"X stage range: [{x_stage.min():.3f}, {x_stage.max():.3f}]")
    print(f"Z stage range: [{z_stage.min():.3f}, {z_stage.max():.3f}]")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(
        description="Generate X+Z+B(+C) G-code to trace an orthogonal 8-edge loop in the XZ plane using calibration JSON."
    )
    ap.add_argument("--calibration", required=True, help="Path to calibration JSON (shadow_calibration schema).")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code filepath.")
    ap.add_argument("--center-x", type=float, default=CENTER_X, help="Loop center X in TIP space.")
    ap.add_argument("--center-z", type=float, default=CENTER_Z, help="Loop center Z in TIP space.")
    ap.add_argument("--dx", type=float, default=DX, help="Horizontal step run (mm). Bounding width = 2*dx.")
    ap.add_argument("--dz", type=float, default=DZ, help="Vertical step run (mm). Bounding height = 2*dz.")
    ap.add_argument("--loops", type=int, default=N_LOOPS, help="Number of loop repetitions.")
    ap.add_argument("--samples-per-edge", type=int, default=SAMPLES_PER_EDGE, help="Points per edge (endpoint excluded).")
    ap.add_argument("--cw", action="store_true", default=False, help="Trace clockwise instead of CCW.")
    ap.add_argument("--jog-feed", type=float, default=DEFAULT_JOG_FEED, help="Jog feedrate for first positioning move.")
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED, help="Feedrate for the tracing moves.")

    ap.add_argument("--min-b", type=float, default=None, help="Lower bound for commanded B (default: from calibration).")
    ap.add_argument("--max-b", type=float, default=None, help="Upper bound for commanded B (default: from calibration).")
    ap.add_argument("--c0-deg", type=float, default=0.0, help="C value used for alpha in [0,180) (default 0).")
    ap.add_argument("--c180-deg", type=float, default=None, help="C value used for alpha in [180,360) (default from calibration).")

    args = ap.parse_args()
    main(args)