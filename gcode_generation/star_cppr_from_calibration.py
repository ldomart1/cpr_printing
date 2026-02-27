#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code for a 5-point star using calibration
coefficients exported by shadow_calibration.

- Loads: cubic_coefficients.r_coeffs / z_coeffs, motor_setup.b_motor_position_range
- Uses inverse mapping per target point:
  - |x_target| -> B via r(B)
  - sign(x_target) -> C in {0,180}
  - z_target -> Z_stage via Z_stage = z_target - z(B)
- USER-TUNABLE defaults + CLI flags
- Clamps requested star radius to calibrated r(u) range (minus safety margin)
- Writes real G-code (axis letters + numeric feedrates).
- Commands only B + Z + C (0/180).
"""

import json, math
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
            if not (min(x0,x1) <= x2 <= max(x0,x1)):
                x2 = 0.5*(x0+x1)
            f2 = f(x2)
            if abs(f2) < max(xtol, rtol*abs(x2)):
                return x2
            if np.sign(f2) == np.sign(f0):
                x0, f0 = x2, f2
            else:
                x1, f1 = x2, f2
        return x2

# ----- Defaults (overridable by CLI) -----
STAR_CENTER_X = 0.0
STAR_CENTER_Z = -130.0
DESIRED_STAR_RADIUS = 10.0    # mm
N_STARS = 1 #20
SAMPLES_PER_LEG = 20
RADIUS_SAFETY_MARGIN = 0.5    # mm
EXPECT_INCREASING_RADIUS = True
DEFAULT_OUT = "star_10x_CPPR.gcode"
DEFAULT_JOG_FEED = 300.0     # units/min (used to be 1200)
DEFAULT_PRINT_FEED = 300.0    # units/min (used to be 800)
DEFAULT_PULL_AXIS  = "B"
DEFAULT_ROT_AXIS   = "C"
DEFAULT_Z_AXIS = "Z"
DEFAULT_MIN_B      = -5.0
DEFAULT_MAX_B      = 0.0
# -----------------------------------------

@dataclass
class Calibration:
    pr: np.ndarray   # [a,b,c,d] for r(u)
    pz: np.ndarray   # [a,b,c,d] for z(u)
    u_min: float
    u_max: float
    pull_axis: str
    rot_axis: str
    z_axis: str

def _polyval4(coeffs: ArrayLike, u: ArrayLike) -> np.ndarray:
    a,b,c,d = coeffs
    u = np.asarray(u)
    return ((a*u + b)*u + c)*u + d

def _compose_neg_u(coeffs: np.ndarray) -> np.ndarray:
    a,b,c,d = coeffs
    return np.array([-a, b, -c, d], dtype=float)

def load_calibration(json_path: str) -> Calibration:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Calibration JSON not found: {json_path}\n"
            "Tip: quote paths with spaces.\n"
        )
    with p.open("r") as f:
        data = json.load(f)

    pr = np.array(data["cubic_coefficients"]["r_coeffs"], dtype=float)
    pz = np.array(data["cubic_coefficients"]["z_coeffs"], dtype=float)
    motor_setup = data.get("motor_setup", {})
    duet_map = data.get("duet_axis_mapping", {})

    # New schema uses b_motor_position_range. Keep delta_motor_range fallback for compatibility.
    if "b_motor_position_range" in motor_setup:
        u_min, u_max = map(float, motor_setup["b_motor_position_range"])
    elif "delta_motor_range" in motor_setup:
        u_min, u_max = map(float, motor_setup["delta_motor_range"])
    else:
        raise KeyError("motor_setup must contain 'b_motor_position_range' (or legacy 'delta_motor_range').")

    pull_axis = str(
        duet_map.get("pull_axis")
        or duet_map.get("b_motor_axis")
        or motor_setup.get("b_motor_axis")
        or DEFAULT_PULL_AXIS
    )
    rot_axis = str(
        duet_map.get("rotation_axis")
        or motor_setup.get("rotation_axis")
        or DEFAULT_ROT_AXIS
    )
    z_axis = str(
        duet_map.get("vertical_axis")
        or motor_setup.get("vertical_axis")
        or DEFAULT_Z_AXIS
    )

    if pr.shape[0] != 4 or pz.shape[0] != 4:
        raise ValueError("Expected 4 coeffs for r and z")
    if u_min > u_max:
        u_min, u_max = u_max, u_min
    return Calibration(
        pr=pr, pz=pz, u_min=u_min, u_max=u_max,
        pull_axis=pull_axis, rot_axis=rot_axis, z_axis=z_axis
    )

def ensure_monotone_direction(cal: Calibration, expect_increasing_radius=True) -> Calibration:
    # Keep for compatibility with older workflows; no variable remapping in B+Z+C mode.
    _ = expect_increasing_radius
    return cal

def invert_r_to_u(r_target: float, cal: Calibration, u_prev: Optional[float]=None) -> float:
    f = lambda u: _polyval4(cal.pr, u) - r_target
    usamp = np.linspace(cal.u_min, cal.u_max, 200)
    fsamp = f(usamp)
    fsamp = np.where(np.isfinite(fsamp), fsamp, 1e9)
    brackets: List[Tuple[float,float]] = []
    for a,b,fa,fb in zip(usamp[:-1], usamp[1:], fsamp[:-1], fsamp[1:]):
        if np.sign(fa) == 0: return a
        if np.sign(fa) != np.sign(fb): brackets.append((a,b))
    if not brackets:
        return float(cal.u_min if abs(f(cal.u_min)) < abs(f(cal.u_max)) else cal.u_max)
    def solve_bracket(a,b):
        try:    return float(brentq(f, a, b, maxiter=100))
        except: return float(0.5*(a+b))
    if len(brackets) == 1 or u_prev is None:
        a,b = brackets[0]; return solve_bracket(a,b)
    roots = [solve_bracket(a,b) for (a,b) in brackets]
    roots = [r for r in roots if np.isfinite(r)]
    if not roots:
        return float(cal.u_min if abs(f(cal.u_min)) < abs(f(cal.u_max)) else cal.u_max)
    roots = np.array(roots)
    return float(roots[np.argmin(np.abs(roots - u_prev))])

def compute_max_r_from_calibration(cal: Calibration) -> Tuple[float,float,float]:
    uu = np.linspace(cal.u_min, cal.u_max, 501)
    rr = _polyval4(cal.pr, uu)
    r_min = float(np.min(rr)); r_max = float(np.max(rr))
    return r_min, r_max, r_max - r_min

def make_star_polyline_xz(n_stars: int, circ_radius: float, samples_per_leg: int) -> np.ndarray:
    R_raw = math.sqrt((5 - math.sqrt(5))/10)      # unit-star outer radius
    scale = circ_radius / R_raw
    R = scale * R_raw
    r = scale * math.sqrt((25 - 11*math.sqrt(5))/10)

    def rotz(deg: float) -> np.ndarray:
        th = math.radians(deg)
        c, s = math.cos(th), math.sin(th)
        return np.array([[ c,-s,0],[ s, c,0],[ 0, 0,1]], dtype=float)

    PT   = np.array([0.0, R, 1.0])
    pt_r = np.array([0.0,-r, 1.0])
    pt_r = rotz(2*360/5) @ pt_r
    pt_l = pt_r.copy(); pt_l[0] = -pt_l[0]

    pts = []
    for _ in range(n_stars):
        for _i in range(5):
            x_add = np.linspace(pt_r[0], PT[0], samples_per_leg)
            y_add = np.linspace(pt_r[1], PT[1], samples_per_leg)
            pts.append(np.stack([x_add, y_add], axis=1))
            x_add = np.linspace(PT[0], pt_l[0], samples_per_leg)
            y_add = np.linspace(PT[1], pt_l[1], samples_per_leg)
            pts.append(np.stack([x_add, y_add], axis=1))
            PT   = rotz(360/5) @ PT
            pt_r = rotz(360/5) @ pt_r
            pt_l = rotz(360/5) @ pt_l

    return np.vstack(pts)  # columns: x, z in local plane

def generate_cppr_gcode_from_path(
    pts_xz: np.ndarray,
    cal: Calibration,
    center_x: float = STAR_CENTER_X,
    center_z: float = STAR_CENTER_Z,
    filename: str = DEFAULT_OUT,
    jog_feed: float = DEFAULT_JOG_FEED,
    print_feed: float = DEFAULT_PRINT_FEED,
    pull_axis: str  = DEFAULT_PULL_AXIS,
    rot_axis: str   = DEFAULT_ROT_AXIS,
    z_axis: str = DEFAULT_Z_AXIS,
    min_b: float = DEFAULT_MIN_B,
    max_b: float = DEFAULT_MAX_B,
    c_pos_deg: float = 0.0,
    c_neg_deg: float = 180.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    x_target = pts_xz[:, 0] + float(center_x)
    z_target = pts_xz[:, 1] + float(center_z)
    r_targets = np.abs(x_target)
    r_min_cal, r_max_cal, _ = compute_max_r_from_calibration(cal)
    r_targets = np.clip(r_targets, r_min_cal, r_max_cal)

    N = pts_xz.shape[0]
    b_cmd = np.zeros(N)
    z_stage = np.zeros(N)
    c_cmd = np.zeros(N)
    lo, hi = (min_b, max_b) if min_b <= max_b else (max_b, min_b)

    b_prev: Optional[float] = None
    for i in range(N):
        b_i = invert_r_to_u(float(r_targets[i]), cal, u_prev=b_prev)
        b_i = float(np.clip(b_i, lo, hi))
        z_curl_i = float(_polyval4(cal.pz, b_i))
        z_stage[i] = float(z_target[i] - z_curl_i)
        b_cmd[i] = b_i
        c_cmd[i] = c_pos_deg if x_target[i] >= 0.0 else c_neg_deg
        b_prev = b_i

    with open(filename, "w") as f:
        f.write("; generated by star_cppr_from_calibration.py\n")
        f.write(f"; axes: z→{z_axis}, pull→{pull_axis}, rot→{rot_axis}\n")
        f.write("; inverse mode: |x|->B, sign(x)->C(0/180), z->Z_stage using z(B)\n")
        f.write(f"; star center target: X={center_x:.3f}, Z={center_z:.3f}\n")
        f.write(f"; B clamp range: [{lo:.3f}, {hi:.3f}]\n")
        rmin, rmax, _ = compute_max_r_from_calibration(cal)
        f.write(f"; calibrated r range: [{rmin:.3f}, {rmax:.3f}] mm\n")
        f.write("G90\n")

        # Initialize C then move to first Z/B target.
        f.write(f"G1 {rot_axis}{c_cmd[0]:.3f} F{(10.0*jog_feed):.0f}\n")
        f.write(f"G1 {z_axis}{z_stage[0]:.3f} {pull_axis}{b_cmd[0]:.3f} F{jog_feed:.0f}\n")
        c_prev = c_cmd[0]
        for i in range(1, N):
            if c_cmd[i] != c_prev:
                f.write(f"G1 {rot_axis}{c_cmd[i]:.3f} F{(10.0*jog_feed):.0f}\n")
                c_prev = c_cmd[i]
            f.write(f"G1 {z_axis}{z_stage[i]:.3f} {pull_axis}{b_cmd[i]:.3f} F{print_feed:.0f}\n")

    return b_cmd, z_stage, c_cmd

def clamp_radius_to_calibration(desired_radius: float, cal: Calibration, safety_margin: float) -> float:
    _, r_max, _ = compute_max_r_from_calibration(cal)
    return min(desired_radius, max(0.0, r_max - safety_margin))

def main(args):
    cal0 = load_calibration(args.calibration)
    cal = ensure_monotone_direction(cal0, expect_increasing_radius=args.expect_increasing_radius)

    pull_axis = args.pull_axis if args.pull_axis is not None else cal.pull_axis
    rot_axis = args.rot_axis if args.rot_axis is not None else cal.rot_axis
    z_axis = args.z_axis if args.z_axis is not None else cal.z_axis
    min_b = args.min_b if args.min_b is not None else cal.u_min
    max_b = args.max_b if args.max_b is not None else cal.u_max

    clamped_radius = clamp_radius_to_calibration(args.radius, cal, args.safety_margin)
    if clamped_radius < args.radius:
        print(f"[info] Desired star radius {args.radius:.3f} mm exceeds calibrated max; clamped to {clamped_radius:.3f} mm (safety {args.safety_margin:.3f} mm).")

    pts = make_star_polyline_xz(
        n_stars=args.stars,
        circ_radius=clamped_radius,
        samples_per_leg=args.samples_per_leg,
    )
    r_min_cal, r_max_cal, _ = compute_max_r_from_calibration(cal)
    x_target_preview = pts[:, 0] + float(args.center_x)
    x_abs_max = float(np.max(np.abs(x_target_preview)))
    if x_abs_max > r_max_cal:
        print(f"[warn] |x| target max {x_abs_max:.3f} exceeds calibrated r_max {r_max_cal:.3f}; values will be clipped.")

    b_cmd, z_stage, c_cmd = generate_cppr_gcode_from_path(
        pts_xz=pts, cal=cal, center_x=args.center_x, center_z=args.center_z,
        filename=args.out, jog_feed=args.jog_feed, print_feed=args.print_feed,
        pull_axis=pull_axis, rot_axis=rot_axis,
        z_axis=z_axis, min_b=min_b, max_b=max_b,
        c_pos_deg=args.c_pos_deg, c_neg_deg=args.c_neg_deg,
    )
    print(f"Wrote {args.out} with {len(b_cmd)} points.")
    print(f"B range used: [{b_cmd.min():.3f}, {b_cmd.max():.3f}] (cal: [{cal.u_min:.3f}, {cal.u_max:.3f}])")
    rr = _polyval4(cal.pr, b_cmd)
    print(f"|x| range used from r(B): [{rr.min():.3f}, {rr.max():.3f}] centered at ({args.center_x:.3f}, {args.center_z:.3f})")
    print(f"Z-stage range used: [{z_stage.min():.3f}, {z_stage.max():.3f}]")
    print(f"C values used: {sorted(set(np.round(c_cmd, 6).tolist()))}")
    print(f"Axes used: z={z_axis}, pull={pull_axis}, rot={rot_axis}")
    print(f"Pull clamp used: [{min(min_b, max_b):.3f}, {max(min_b, max_b):.3f}]")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser(description="Generate CPPR G-code for a star path using calibration JSON.")
    ap.add_argument("--calibration", required=True, help="Path to calibration JSON (shadow_calibration schema).")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code filepath.")
    ap.add_argument("--center-x", type=float, default=STAR_CENTER_X, help="Target star center in X for inverse mapping.")
    ap.add_argument("--center-z", type=float, default=STAR_CENTER_Z, help="Target star center in Z for inverse mapping.")
    ap.add_argument("--radius", type=float, default=DESIRED_STAR_RADIUS, help="Desired star circumcircle radius (mm).")
    ap.add_argument("--stars", type=int, default=N_STARS, help="Number of star repetitions around the circle.")
    ap.add_argument("--samples-per-leg", type=int, default=SAMPLES_PER_LEG, help="Polyline samples per star leg.")
    ap.add_argument("--expect-increasing-radius", action="store_true", default=EXPECT_INCREASING_RADIUS,
                    help="Compatibility flag; ignored in B+Z+C inverse mode.")
    ap.add_argument("--safety-margin", type=float, default=RADIUS_SAFETY_MARGIN, help="Margin subtracted from r_max when clamping radius.")
    ap.add_argument("--jog-feed", type=float, default=DEFAULT_JOG_FEED, help="Jog feedrate for first move (units/min).")
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED, help="Nominal print feedrate (units/min).")
    ap.add_argument("--pull-axis", type=str, default=None, help="Axis letter for pull linear axis (default: from calibration JSON).")
    ap.add_argument("--rot-axis",   type=str, default=None,   help="Axis letter for ROTARY axis (default: from calibration JSON).")
    ap.add_argument("--z-axis", type=str, default=None, help="Axis letter for stage Z motion (default: from calibration JSON).")
    ap.add_argument("--min-b",      type=float, default=None,    help="Lower bound for commanded B (default: calibration range min).")
    ap.add_argument("--max-b",      type=float, default=None,    help="Upper bound for commanded B (default: calibration range max).")
    ap.add_argument("--c-pos-deg",  type=float, default=0.0,   help="C value used for x>=0.")
    ap.add_argument("--c-neg-deg",  type=float, default=180.0, help="C value used for x<0.")
    args = ap.parse_args()
    main(args)
