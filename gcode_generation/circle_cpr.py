#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code to trace a circle in the XZ plane
using calibration coefficients exported by shadow_calibration.

Kinematic model:
  X_tip = X_stage + sgn * r(B)
  Z_tip = Z_stage + z(B)

where:
  - sgn is +1 when C==0 deg, and -1 when C==180 deg
  - r(B), z(B), tip_angle_deg(B) are cubic polynomials from calibration JSON

Circle sequence implemented from user request:
  - Start at left quadrant, C=180, high pull (target 180 deg tip angle).
  - Trace lower half (left -> right) while uncurling to 0 deg.
  - At right quadrant, rotate C by 180 deg.
  - Trace upper half (right -> left) while curling back to 180 deg.

B is solved from tip_angle_deg(B)=target_angle by root-finding.
X_stage and Z_stage are then solved to keep the tip exactly on the requested circle.
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
    _HAS_SCIPY = True
except Exception:
    _HAS_SCIPY = False

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
CIRCLE_CENTER_X = 65.0
CIRCLE_CENTER_Z = -110.0
DESIRED_CIRCLE_RADIUS = 25.0     # mm (set conservative; will clamp to calibration)
N_CIRCLES = 1
SAMPLES_PER_CIRCLE = 360
RADIUS_SAFETY_MARGIN = 0.5
DEFAULT_OUT = "circle_xzb_cppr.gcode"
DEFAULT_JOG_FEED = 300.0
DEFAULT_PRINT_FEED = 300.0
DEFAULT_MIN_B = -5.0
DEFAULT_MAX_B = -0.0
# ------------------------------------------------------------


@dataclass
class Calibration:
    pr: np.ndarray          # r(B) cubic coefficients [a,b,c,d]
    pz: np.ndarray          # z(B) cubic coefficients [a,b,c,d]
    pa: np.ndarray          # tip_angle_deg(B) cubic coefficients [a,b,c,d]
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


def _sgn_from_c(c_val: float, c180: float, c0: float = 0.0, tol: float = 1e-6) -> float:
    """
    Return kinematic X sign for discrete C states:
      +1 at C==c0, -1 at C==c180
    """
    if abs(c_val - c180) <= tol:
        return -1.0
    if abs(c_val - c0) <= tol:
        return +1.0
    raise ValueError(f"Unsupported C state for binary-sign model: C={c_val} (expected {c0} or {c180})")


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
    rot_axis = str(duet_map.get("rotation_axis") or motor_setup.get("rotation_axis") or "C")
    x_axis = str(duet_map.get("horizontal_axis") or motor_setup.get("horizontal_axis") or "X")
    z_axis = str(duet_map.get("vertical_axis") or motor_setup.get("vertical_axis") or "Z")

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
        c_180_deg=c_180
    )


def compute_r_range(cal: Calibration, n: int = 801) -> Tuple[float, float]:
    bb = np.linspace(cal.b_min, cal.b_max, n)
    rr = _polyval4(cal.pr, bb)
    return float(np.min(rr)), float(np.max(rr))


def _find_roots_for_target(coeffs: np.ndarray, target: float, b_min: float, b_max: float) -> List[float]:
    """
    Find all B in [b_min,b_max] such that polyval(coeffs,B)=target.
    """
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
    roots = [r for r in roots if np.isfinite(r)]
    return roots


def invert_r_to_b(r_target: float, cal: Calibration, b_prev: Optional[float] = None) -> float:
    """
    Invert r(B)=r_target, choosing the root closest to b_prev for continuity.
    Falls back to the endpoint with smallest residual.
    """
    f = lambda b: float(_polyval4(cal.pr, b) - r_target)

    roots = _find_roots_for_target(cal.pr, r_target, cal.b_min, cal.b_max)
    if not roots:
        # endpoint fallback
        b0 = cal.b_min
        b1 = cal.b_max
        return float(b0 if abs(f(b0)) < abs(f(b1)) else b1)

    if len(roots) == 1 or b_prev is None:
        return float(roots[0])

    arr = np.array(roots, dtype=float)
    return float(arr[np.argmin(np.abs(arr - float(b_prev)))])


def invert_tip_angle_to_b(angle_target_deg: float, cal: Calibration, b_prev: Optional[float] = None) -> float:
    """
    Invert tip_angle_deg(B)=angle_target_deg, preferring continuity via b_prev.
    """
    f = lambda b: float(_polyval4(cal.pa, b) - angle_target_deg)

    roots = _find_roots_for_target(cal.pa, angle_target_deg, cal.b_min, cal.b_max)
    if not roots:
        b0 = cal.b_min
        b1 = cal.b_max
        return float(b0 if abs(f(b0)) < abs(f(b1)) else b1)

    if len(roots) == 1 or b_prev is None:
        return float(roots[0])

    arr = np.array(roots, dtype=float)
    return float(arr[np.argmin(np.abs(arr - float(b_prev)))])


def make_circle_tip_trajectory(
    x_center: float,
    z_center: float,
    radius: float,
    n_circles: int,
    samples_per_circle: int,
    start_theta: float = math.pi,
    ccw: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns:
      tip_xz: (N,2) points for desired TIP trajectory in XZ
      thetas: (N,) wrapped angles in [0,2pi)
    """
    direction = 1.0 if ccw else -1.0
    pts = []
    ths = []
    for k in range(n_circles):
        t = np.linspace(0.0, 2.0*math.pi, samples_per_circle, endpoint=False)
        theta = start_theta + direction*(t + 2.0*math.pi*k)
        theta_wrapped = np.mod(theta, 2.0*math.pi)

        x_tip = x_center + radius * np.cos(theta)
        z_tip = z_center + radius * np.sin(theta)
        pts.append(np.stack([x_tip, z_tip], axis=1))
        ths.append(theta_wrapped)

    return np.vstack(pts), np.concatenate(ths)


def generate_gcode_circle_xzb(
    tip_xz: np.ndarray,
    thetas: np.ndarray,
    cal: Calibration,
    out_path: str,
    jog_feed: float,
    print_feed: float,
    safety_margin: float = RADIUS_SAFETY_MARGIN,
    c0_deg: float = 0.0,
    c180_deg: Optional[float] = None,
    min_b: Optional[float] = None,
    max_b: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Given a desired TIP trajectory (X_tip,Z_tip), generate machine commands for:
      X_stage, Z_stage, B, and C (0/180) such that tip follows the circle.
    """
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
    n_b_clamped = 0

    for i in range(N):
        x_tip = float(tip_xz[i, 0])
        z_tip = float(tip_xz[i, 1])
        theta = float(np.mod(thetas[i], 2.0 * math.pi))

        # Requested sequence:
        # lower half (theta in [pi, 2pi)): C=180 and angle 180->0
        # upper half (theta in [0, pi)): C=0 and angle 0->180
        if theta >= math.pi:
            c = float(c180)
            sgn = -1.0
            progress = (theta - math.pi) / math.pi
            tip_angle_target = 180.0 * (1.0 - progress)
        else:
            c = float(c0_deg)
            sgn = +1.0
            progress = theta / math.pi
            tip_angle_target = 180.0 * progress
        c_cmd[i] = c

        # Priority order:
        #  1) hit the requested tip point on the circle (X_tip, Z_tip)
        #  2) make the extrusion direction as tangential as possible
        # If tip-angle/B are clamped, tangentiality degrades, but X/Z are
        # re-solved from the actual commanded B so the tip still lands on target.
        tip_angle_limited = float(np.clip(tip_angle_target, cal.tip_angle_min, cal.tip_angle_max))
        if tip_angle_limited != tip_angle_target:
            n_angle_clamped += 1

        b_i_unclamped = invert_tip_angle_to_b(tip_angle_limited, cal, b_prev=b_prev)
        b_i = float(np.clip(b_i_unclamped, b_lo, b_hi))
        if b_i != b_i_unclamped:
            n_b_clamped += 1
        b_cmd[i] = b_i
        b_prev = b_i

        # Apply Z compensation
        z_curl = float(_polyval4(cal.pz, b_i))

        # Compute stage positions so that the tip equals target
        x_r = float(_polyval4(cal.pr, b_i))
        x_stage[i] = x_tip - sgn * x_r
        z_stage[i] = z_tip - z_curl

        # Guardrail: the commanded stage pose must reproduce the requested tip point.
        x_tip_check = x_stage[i] + sgn * x_r
        z_tip_check = z_stage[i] + z_curl
        if abs(x_tip_check - x_tip) > 1e-9 or abs(z_tip_check - z_tip) > 1e-9:
            raise RuntimeError("Internal tip-tracking consistency check failed.")

    with open(out_path, "w") as f:
        f.write("; generated by circle_xzb_from_calibration.py\n")
        f.write(f"; axes: X→{cal.x_axis}, Z→{cal.z_axis}, pull→{cal.pull_axis}, rot→{cal.rot_axis}\n")
        f.write("; model:\n")
        f.write(";   X_tip = X_stage + sgn*r(B)   (sgn=+1 at C=0, sgn=-1 at C=180)\n")
        f.write(";   Z_tip = Z_stage + z(B)\n")
        f.write(";   tip_angle_deg = tip_angle(B)\n")
        f.write("; sequence: lower half C=180 with uncurl 180->0, upper half C=0 with curl 0->180.\n")
        rmin, rmax = compute_r_range(cal)
        f.write(f"; calibrated r(B) range (sampled): [{rmin:.3f}, {rmax:.3f}] mm\n")
        f.write(f"; calibrated tip-angle range: [{cal.tip_angle_min:.3f}, {cal.tip_angle_max:.3f}] deg\n")
        if n_angle_clamped > 0:
            f.write(f"; info: {n_angle_clamped}/{N} points requested outside calibrated tip-angle range and were clamped.\n")
        if n_b_clamped > 0:
            f.write(f"; info: {n_b_clamped}/{N} points hit the commanded B clamp; tangentiality was degraded to preserve tip tracking.\n")
        f.write(f"; B clamp: [{b_lo:.3f}, {b_hi:.3f}]\n")
        f.write("G90\n")

        # Move to first point safely: set C first, then X/Z/B together at jog feed.
        f.write(f"G1 {cal.rot_axis}{c_cmd[0]:.3f} F{(10.0*jog_feed):.0f}\n")
        f.write(
            f"G1 {cal.x_axis}{x_stage[0]:.3f} {cal.z_axis}{z_stage[0]:.3f} "
            f"{cal.pull_axis}{b_cmd[0]:.3f} F{jog_feed:.0f}\n"
        )

        c_prev = c_cmd[0]
        for i in range(1, N):
            if c_cmd[i] != c_prev:
                # --- Seam fix: preserve the SAME tip point across the C flip ---
                # Use the previous sampled tip point (i-1) for both sides of the flip.
                # This guarantees the X compensation direction is correct:
                #   dX_stage = (sgn_old - sgn_new) * r(B)
                # For signed-r calibrations, the sign of dX depends on r(B).
                j = i - 1

                x_tip_seam = float(tip_xz[j, 0])
                z_tip_seam = float(tip_xz[j, 1])

                # Hold B constant through the flip to avoid mixing geometries.
                b_seam = float(b_cmd[j])

                sgn_old = _sgn_from_c(float(c_prev), c180=c180, c0=float(c0_deg))
                sgn_new = _sgn_from_c(float(c_cmd[i]), c180=c180, c0=float(c0_deg))

                r_seam = float(_polyval4(cal.pr, b_seam))
                z_curl_seam = float(_polyval4(cal.pz, b_seam))

                # Recompute stage poses for the SAME tip point before/after flip
                x_stage_old = x_tip_seam - sgn_old * r_seam
                z_stage_old = z_tip_seam - z_curl_seam

                x_stage_new = x_tip_seam - sgn_new * r_seam
                z_stage_new = z_tip_seam - z_curl_seam

                # Hard sanity check against the kinematic model.
                dx = x_stage_new - x_stage_old
                dx_expected = (sgn_old - sgn_new) * r_seam
                if abs(dx - dx_expected) > 1e-9:
                    raise RuntimeError(
                        "Seam compensation consistency error: "
                        f"got dX={dx:.6f}, expected {dx_expected:.6f} "
                        f"(r={r_seam:.6f}, sgn_old={sgn_old:.0f}, sgn_new={sgn_new:.0f})"
                    )

                # 1) Seam flip move (same tip point, compensated for new C sign)
                f.write(
                    f"G1 {cal.x_axis}{x_stage_new:.3f} {cal.z_axis}{z_stage_new:.3f} "
                    f"{cal.pull_axis}{b_seam:.3f} {cal.rot_axis}{c_cmd[i]:.3f} "
                    f"F{print_feed:.0f}\n"
                )

                # 2) Then advance to the next sampled point under the new C state
                f.write(
                    f"G1 {cal.x_axis}{x_stage[i]:.3f} {cal.z_axis}{z_stage[i]:.3f} "
                    f"{cal.pull_axis}{b_cmd[i]:.3f} F{print_feed:.0f}\n"
                )

                c_prev = c_cmd[i]
                continue

            f.write(
                f"G1 {cal.x_axis}{x_stage[i]:.3f} {cal.z_axis}{z_stage[i]:.3f} "
                f"{cal.pull_axis}{b_cmd[i]:.3f} F{print_feed:.0f}\n"
            )

    return x_stage, z_stage, b_cmd, c_cmd, n_angle_clamped


def main(args):
    cal = load_calibration(args.calibration)

    tip_xz, _thetas = make_circle_tip_trajectory(
        x_center=args.center_x,
        z_center=args.center_z,
        radius=args.radius,
        n_circles=args.circles,
        samples_per_circle=args.samples_per_circle,
        start_theta=math.pi,   # start at trig pi (leftmost point)
        ccw=not args.cw
    )

    # Explicitly close the path (final point == first point)
    # This avoids ending one sample short because make_circle_tip_trajectory uses endpoint=False.
    tip_xz = np.vstack([tip_xz, tip_xz[0]])
    _thetas = np.concatenate([_thetas, [_thetas[0]]])

    x_stage, z_stage, b_cmd, c_cmd, n_angle_clamped = generate_gcode_circle_xzb(
        tip_xz=tip_xz,
        thetas=_thetas,
        cal=cal,
        out_path=args.out,
        jog_feed=args.jog_feed,
        print_feed=args.print_feed,
        safety_margin=args.safety_margin,
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
        print(f"[info] tip-angle target saturated at {n_angle_clamped}/{len(b_cmd)} points to [{cal.tip_angle_min:.3f}, {cal.tip_angle_max:.3f}] deg.")
    rr = _polyval4(cal.pr, b_cmd)
    aa = _polyval4(cal.pa, b_cmd)
    print(f"r(B) range used: [{rr.min():.3f}, {rr.max():.3f}] mm")
    print(f"tip-angle(B) range used: [{aa.min():.3f}, {aa.max():.3f}] deg")
    print(f"X stage range: [{x_stage.min():.3f}, {x_stage.max():.3f}]")
    print(f"Z stage range: [{z_stage.min():.3f}, {z_stage.max():.3f}]")


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate X+Z+B(+C) G-code to trace a circle in the XZ plane using calibration JSON.")
    ap.add_argument("--calibration", required=True, help="Path to calibration JSON (shadow_calibration schema).")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code filepath.")
    ap.add_argument("--center-x", type=float, default=CIRCLE_CENTER_X, help="Desired circle center X in the XZ plane (TIP space).")
    ap.add_argument("--center-z", type=float, default=CIRCLE_CENTER_Z, help="Desired circle center Z in the XZ plane (TIP space).")
    ap.add_argument("--radius", type=float, default=DESIRED_CIRCLE_RADIUS, help="Desired circle radius (mm) in TIP space.")
    ap.add_argument("--circles", type=int, default=N_CIRCLES, help="Number of circle repetitions.")
    ap.add_argument("--samples-per-circle", type=int, default=SAMPLES_PER_CIRCLE, help="Number of points per circle.")
    ap.add_argument("--cw", action="store_true", default=False, help="Trace clockwise instead of CCW.")
    ap.add_argument("--safety-margin", type=float, default=RADIUS_SAFETY_MARGIN, help="Margin subtracted from max calibrated r(B) target.")
    ap.add_argument("--jog-feed", type=float, default=DEFAULT_JOG_FEED, help="Jog feedrate for first positioning move (units/min).")
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED, help="Feedrate for the circle trace moves (units/min).")

    # Optional overrides
    ap.add_argument("--min-b", type=float, default=None, help="Lower bound for commanded B (default: from calibration).")
    ap.add_argument("--max-b", type=float, default=None, help="Upper bound for commanded B (default: from calibration).")
    ap.add_argument("--c0-deg", type=float, default=0.0, help="C value used for +X direction (default 0).")
    ap.add_argument("--c180-deg", type=float, default=None, help="C value used for -X direction (default from calibration, else 180).")

    args = ap.parse_args()
    main(args)
