#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code to trace a 5-point star in the XZ plane
as TWO EXPLICIT MIRRORED HALF PATHS using:
  - Z axis
  - B axis
  - C axis (0 / 180 mirror)
and allowing X correction moves WHEN NEEDED to preserve exact tip tracking.

This version changes the previous "X only at startup" behavior:
  - X is still kept fixed whenever possible.
  - BUT if the requested tip point cannot be realized on the nominal mirror geometry
    because of B/r limits (and practical nonzero minimum tip-angle / nonzero min |r|),
    the script will move X left/right as needed to preserve the requested tip point.

In other words:
  PRIORITY 1: hit the requested tip position (trace the full star)
  PRIORITY 2: keep X at the nominal star center mirror plane if possible
  PRIORITY 3: preserve mirrored-half replay structure (C=0 half, one C flip, C=180 half)

Path strategy:
  1) Build full star polyline in local tip-space (x,z), centered at (0,0).
  2) Extract right-half explicit path (x>=0) by clipping against x=0.
  3) Convert right-half points to B/Z/X (C=0), solving tip position first:
       X_stage = X_tip - r(B)
     with B chosen from reachable |r| (clamped if necessary).
  4) Flip C once to 180.
  5) Replay mirrored left-half path (same geometry mirrored in x), again solving tip position first:
       X_stage = X_tip + r(B)
     so the tip lands exactly even when the nominal mirror plane is not reachable.

This means X may move during drawing near centerline/limit points, which is explicitly allowed
in this version to enforce exact tip tracking.

Kinematic model (from calibration):
  X_tip = X_stage + sgn * r(B)
  Z_tip = Z_stage + z(B)
where:
  sgn = +1 at C=0, -1 at C=180
"""

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
from numpy.typing import ArrayLike

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


# ---------------- Defaults (CLI-overridable) ----------------
STAR_CENTER_X = 65.0
STAR_CENTER_Z = -110.0

DESIRED_STAR_OUTER_RADIUS = 18.0
INNER_RADIUS_RATIO = 0.38196601125
STAR_ROTATION_DEG = 90.0
SAMPLES_PER_EDGE = 30

DEFAULT_OUT = "gcode_generation/star_two_mirrored_halves_tip_priority_xoffset.gcode"
DEFAULT_JOG_FEED = 200.0
DEFAULT_PRINT_FEED = 200.0
DEFAULT_PRINT_FEED_B = 500.0
DEFAULT_PRINT_FEED_C = 1000.0
RADIUS_SAFETY_MARGIN = 0.5
# ------------------------------------------------------------


@dataclass
class Calibration:
    pr: np.ndarray
    pz: np.ndarray
    pa: np.ndarray
    b_min: float
    b_max: float
    x_axis: str
    z_axis: str
    b_axis: str
    c_axis: str
    c_180_deg: float


def _polyval4(coeffs: ArrayLike, u: ArrayLike) -> np.ndarray:
    a, b, c, d = coeffs
    u = np.asarray(u, dtype=float)
    return ((a * u + b) * u + c) * u + d


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

    b_range = motor_setup.get("b_motor_position_range", [-5.4, 0.0])
    b_min, b_max = map(float, b_range)
    if b_min > b_max:
        b_min, b_max = b_max, b_min

    x_axis = str(duet_map.get("horizontal_axis") or motor_setup.get("horizontal_axis") or "X")
    z_axis = str(duet_map.get("vertical_axis") or motor_setup.get("vertical_axis") or "Z")
    b_axis = str(duet_map.get("pull_axis") or motor_setup.get("b_motor_axis") or "B")
    c_axis = str(duet_map.get("rotation_axis") or motor_setup.get("rotation_axis") or "C")
    c_180 = float(motor_setup.get("rotation_axis_180_deg", 180.0))

    return Calibration(
        pr=pr, pz=pz, pa=pa,
        b_min=b_min, b_max=b_max,
        x_axis=x_axis, z_axis=z_axis, b_axis=b_axis, c_axis=c_axis,
        c_180_deg=c_180
    )


def sampled_r_abs_range(cal: Calibration, b_lo: Optional[float] = None, b_hi: Optional[float] = None, n: int = 2001) -> Tuple[float, float]:
    lo = cal.b_min if b_lo is None else float(b_lo)
    hi = cal.b_max if b_hi is None else float(b_hi)
    if lo > hi:
        lo, hi = hi, lo
    bb = np.linspace(lo, hi, n)
    rr = _polyval4(cal.pr, bb)
    return float(np.min(np.abs(rr))), float(np.max(np.abs(rr)))


def _find_roots_for_target(coeffs: np.ndarray, target: float, b_min: float, b_max: float) -> List[float]:
    f = lambda b: float(_polyval4(coeffs, b) - target)
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


def invert_r_to_b(r_target: float, cal: Calibration, b_lo: float, b_hi: float, b_prev: Optional[float] = None) -> float:
    """
    Solve r(B)=r_target, choosing a root near b_prev if possible.
    """
    roots = _find_roots_for_target(cal.pr, r_target, b_lo, b_hi)
    f = lambda b: float(_polyval4(cal.pr, b) - r_target)

    if not roots:
        return float(b_lo if abs(f(b_lo)) <= abs(f(b_hi)) else b_hi)

    if b_prev is None or len(roots) == 1:
        return float(roots[0])

    arr = np.array(roots, dtype=float)
    return float(arr[np.argmin(np.abs(arr - float(b_prev)))])


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
    """
    Clip one segment to x>=0 half-plane.
    """
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
    """
    Extract x>=0 portions from full densified star path as an explicit half-path.
    """
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


def scale_star_if_needed(desired_outer_radius: float, cal: Calibration, b_lo: float, b_hi: float, safety_margin: float) -> Tuple[float, float, float]:
    r_abs_min, r_abs_max = sampled_r_abs_range(cal, b_lo=b_lo, b_hi=b_hi)
    max_reachable = max(0.0, r_abs_max - float(safety_margin))
    outer_used = min(float(desired_outer_radius), max_reachable)
    return outer_used, r_abs_min, r_abs_max


def right_half_points_to_commands_tip_priority(
    right_half_pts_local_xz: np.ndarray,
    cal: Calibration,
    center_x_nominal: float,
    center_z_tip: float,
    b_lo: float,
    b_hi: float,
    c_state: float,
    c0_deg: float,
    c180_deg: float,
    b_seed_from: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Convert RIGHT-HALF local points (x>=0) into exact tip-tracking commands for a chosen C state.

    Tip-space target:
      x_tip_target = center_x_nominal + x_local   (for right-half geometry)
      z_tip_target = center_z_tip + z_local

    For C=0 (sgn=+1):
      X_stage = x_tip_target - r(B)
    For C=180 (sgn=-1):
      X_stage = x_tip_target + r(B)

    B selection policy:
      - target radial magnitude is x_local (>=0)
      - clamp to reachable |r(B)| if needed
      - choose root near previous B for continuity (or optional seed path)
    X offset emerges automatically from exact tip-solving if x_local is not achievable.
    """
    if abs(c_state - c180_deg) < 1e-9:
        sgn = -1.0
    elif abs(c_state - c0_deg) < 1e-9:
        sgn = +1.0
    else:
        raise ValueError(f"Unsupported C state {c_state}; expected {c0_deg} or {c180_deg}")

    N = right_half_pts_local_xz.shape[0]
    x_stage = np.zeros(N, dtype=float)
    z_stage = np.zeros(N, dtype=float)
    b_cmd = np.zeros(N, dtype=float)
    c_cmd = np.full(N, float(c_state), dtype=float)

    r_abs_min, r_abs_max = sampled_r_abs_range(cal, b_lo=b_lo, b_hi=b_hi)

    n_inner_clamped = 0
    n_outer_clamped = 0
    max_abs_x_offset_from_nominal = 0.0

    b_prev: Optional[float] = None

    for i in range(N):
        x_local = abs(float(right_half_pts_local_xz[i, 0]))  # x>=0 for explicit right half
        z_local = float(right_half_pts_local_xz[i, 1])

        x_tip_target = float(center_x_nominal + x_local)
        z_tip_target = float(center_z_tip + z_local)

        r_target = x_local
        if r_target < r_abs_min:
            r_target = r_abs_min
            n_inner_clamped += 1
        elif r_target > r_abs_max:
            r_target = r_abs_max
            n_outer_clamped += 1

        # Optional seed continuity from another path (e.g., mirror replay), else local continuity.
        b_hint = None
        if b_seed_from is not None and i < len(b_seed_from):
            b_hint = float(b_seed_from[i])
        elif b_prev is not None:
            b_hint = float(b_prev)

        b_i = invert_r_to_b(r_target, cal, b_lo=b_lo, b_hi=b_hi, b_prev=b_hint)
        b_cmd[i] = b_i
        b_prev = b_i

        r_i = float(_polyval4(cal.pr, b_i))
        z_curl = float(_polyval4(cal.pz, b_i))

        # Exact tip-tracking solve (this is the key change)
        x_stage[i] = x_tip_target - sgn * r_i
        z_stage[i] = z_tip_target - z_curl

        # Track how far X drifts from the nominal center plane command.
        x_offset = x_stage[i] - center_x_nominal
        max_abs_x_offset_from_nominal = max(max_abs_x_offset_from_nominal, abs(x_offset))

        # Hard consistency check
        x_tip_check = x_stage[i] + sgn * r_i
        z_tip_check = z_stage[i] + z_curl
        if abs(x_tip_check - x_tip_target) > 1e-9 or abs(z_tip_check - z_tip_target) > 1e-9:
            raise RuntimeError("Internal tip-tracking consistency check failed.")

    meta = {
        "r_abs_min": float(r_abs_min),
        "r_abs_max": float(r_abs_max),
        "n_inner_clamped": int(n_inner_clamped),
        "n_outer_clamped": int(n_outer_clamped),
        "max_abs_x_offset_from_nominal": float(max_abs_x_offset_from_nominal),
    }
    return x_stage, z_stage, b_cmd, c_cmd, meta


def write_gcode_two_mirrored_halves_tip_priority(
    out_path: str,
    cal: Calibration,
    center_x_nominal: float,
    center_z_tip: float,
    x_right: np.ndarray,
    z_right: np.ndarray,
    b_right: np.ndarray,
    x_left: np.ndarray,
    z_left: np.ndarray,
    b_left: np.ndarray,
    outer_radius_requested: float,
    outer_radius_used: float,
    right_half_npts: int,
    meta_right: dict,
    meta_left: dict,
    jog_feed: float,
    print_feed_b: float,
    print_feed_c: float,
    c0_deg: float,
    c180_deg: float,
):
    """
    Emit G-code:
      startup -> right half (C=0) -> single C flip -> left half (C=180)
    X is allowed during drawing to preserve exact tip tracking when needed.
    """
    if not (len(x_right) == len(z_right) == len(b_right) == len(x_left) == len(z_left) == len(b_left)):
        raise ValueError("Array length mismatch between half-path command arrays.")

    N = len(b_right)

    with open(out_path, "w") as f:
        f.write("; generated by star_two_mirrored_halves_tip_priority_xoffset.py\n")
        f.write(f"; axes: X->{cal.x_axis}, Z->{cal.z_axis}, B->{cal.b_axis}, C->{cal.c_axis}\n")
        f.write("; strategy: explicit right-half path (C=0), single C flip, explicit mirrored left-half path (C=180)\n")
        f.write("; priority: exact tip tracking first; X offset from nominal mirror plane allowed when needed\n")
        f.write("; model: X_tip = X_stage + sgn*r(B), Z_tip = Z_stage + z(B)\n")
        f.write(f"; nominal star center: X={center_x_nominal:.3f}, Z={center_z_tip:.3f}\n")
        f.write(f"; outer radius requested={outer_radius_requested:.3f} mm, used={outer_radius_used:.3f} mm\n")
        f.write(f"; right-half points={right_half_npts}, replayed as explicit mirrored left-half path\n")
        f.write(f"; C states: C0={c0_deg:.3f}, C180={c180_deg:.3f}\n")
        f.write(f"; reachable |r(B)| over commanded range: [{meta_right['r_abs_min']:.3f}, {meta_right['r_abs_max']:.3f}] mm\n")
        f.write(f"; right-half clamps: inner={meta_right['n_inner_clamped']}, outer={meta_right['n_outer_clamped']}\n")
        f.write(f"; left-half clamps:  inner={meta_left['n_inner_clamped']}, outer={meta_left['n_outer_clamped']}\n")
        f.write(f"; max |X_stage - nominal_center_x| right={meta_right['max_abs_x_offset_from_nominal']:.3f} mm\n")
        f.write(f"; max |X_stage - nominal_center_x| left ={meta_left['max_abs_x_offset_from_nominal']:.3f} mm\n")
        f.write("G90\n")

        f.write("; --- startup ---\n")
        f.write(f"G1 {cal.c_axis}{c0_deg:.3f} F{(10.0*jog_feed):.0f}\n")
        f.write(
            f"G1 {cal.x_axis}{x_right[0]:.3f} {cal.z_axis}{z_right[0]:.3f} "
            f"{cal.b_axis}{b_right[0]:.3f} F{jog_feed:.0f}\n"
        )

        f.write("; --- right half (C=0), tip-priority exact tracking ---\n")
        for i in range(1, N):
            f.write(
                f"G1 {cal.x_axis}{x_right[i]:.3f} {cal.z_axis}{z_right[i]:.3f} "
                f"{cal.b_axis}{b_right[i]:.3f} F{print_feed_b:.0f}\n"
            )

        f.write("; --- single mirror flip and continue on left half ---\n")
        # Move to first left-half point while flipping C (same line; exact tip target under new sign)
        f.write(
            f"G1 {cal.x_axis}{x_left[0]:.3f} {cal.z_axis}{z_left[0]:.3f} "
            f"{cal.b_axis}{b_left[0]:.3f} {cal.c_axis}{c180_deg:.3f} F{print_feed_c:.0f}\n"
        )

        f.write("; --- left half (C=180), tip-priority exact tracking ---\n")
        for i in range(1, N):
            f.write(
                f"G1 {cal.x_axis}{x_left[i]:.3f} {cal.z_axis}{z_left[i]:.3f} "
                f"{cal.b_axis}{b_left[i]:.3f} F{print_feed_b:.0f}\n"
            )


def main(args):
    cal = load_calibration(args.calibration)

    b_lo = cal.b_min if args.min_b is None else float(args.min_b)
    b_hi = cal.b_max if args.max_b is None else float(args.max_b)
    if b_lo > b_hi:
        b_lo, b_hi = b_hi, b_lo

    c0_deg = float(args.c0_deg)
    c180_deg = float(cal.c_180_deg if args.c180_deg is None else args.c180_deg)

    # Scale based on max reachable |r(B)| (same as before).
    outer_used, r_abs_min, r_abs_max = scale_star_if_needed(
        desired_outer_radius=float(args.outer_radius),
        cal=cal,
        b_lo=b_lo,
        b_hi=b_hi,
        safety_margin=float(args.safety_margin),
    )
    if outer_used <= 0.0:
        raise RuntimeError("Star radius not reachable after safety margin.")

    # Full star and explicit right-half path
    star_vertices = build_star_vertices(
        outer_radius=outer_used,
        inner_ratio=float(args.inner_ratio),
        rotation_deg=float(args.rotation_deg),
    )
    full_pts = densify_polyline(star_vertices, samples_per_edge=int(args.samples_per_edge))
    right_half_pts = extract_right_half_polyline(full_pts)

    # Right half (C=0): exact tip tracking with X offset if needed
    x_right, z_right, b_right, c_right, meta_right = right_half_points_to_commands_tip_priority(
        right_half_pts_local_xz=right_half_pts,
        cal=cal,
        center_x_nominal=float(args.center_x),
        center_z_tip=float(args.center_z),
        b_lo=b_lo,
        b_hi=b_hi,
        c_state=c0_deg,
        c0_deg=c0_deg,
        c180_deg=c180_deg,
        b_seed_from=None,
    )

    # Left half target is explicit mirrored geometry: x_local -> -x_local.
    # We can reuse the same extracted right-half path by mirroring x coordinate.
    left_half_pts = right_half_pts.copy()
    left_half_pts[:, 0] *= -1.0

    # Convert left-half explicit mirrored path under C=180, preserving exact tip positions.
    # Seed B continuity with b_right to keep similar branch selection where possible.
    # The function expects "right-half-like" x>=0 local input, so pass abs(x) while
    # forming x_tip_target from mirrored geometry outside it by temporarily remapping:
    #
    # To keep the helper generic, we directly call it with the mirrored path converted back
    # to abs(x)-based right geometry, then fix the target x by post-transform via a small wrapper.
    #
    # Simpler and clearer: build a dedicated command solve inline for left half.
    N = left_half_pts.shape[0]
    x_left = np.zeros(N, dtype=float)
    z_left = np.zeros(N, dtype=float)
    b_left = np.zeros(N, dtype=float)
    c_left = np.full(N, c180_deg, dtype=float)

    rmin_cmd, rmax_cmd = sampled_r_abs_range(cal, b_lo=b_lo, b_hi=b_hi)
    n_inner_left = 0
    n_outer_left = 0
    max_abs_xoff_left = 0.0
    b_prev: Optional[float] = None

    for i in range(N):
        x_local_left = float(left_half_pts[i, 0])  # <= 0 generally
        z_local_left = float(left_half_pts[i, 1])

        x_tip_target = float(args.center_x) + x_local_left
        z_tip_target = float(args.center_z) + z_local_left

        r_target = abs(x_local_left)
        if r_target < rmin_cmd:
            r_target = rmin_cmd
            n_inner_left += 1
        elif r_target > rmax_cmd:
            r_target = rmax_cmd
            n_outer_left += 1

        b_hint = float(b_right[i]) if i < len(b_right) else (b_prev if b_prev is not None else None)
        b_i = invert_r_to_b(r_target, cal, b_lo=b_lo, b_hi=b_hi, b_prev=b_hint)
        b_left[i] = b_i
        b_prev = b_i

        r_i = float(_polyval4(cal.pr, b_i))
        z_curl = float(_polyval4(cal.pz, b_i))

        # C=180 => sgn=-1 => X_tip = X_stage - r  => X_stage = X_tip + r
        x_left[i] = x_tip_target + r_i
        z_left[i] = z_tip_target - z_curl

        xoff = x_left[i] - float(args.center_x)
        max_abs_xoff_left = max(max_abs_xoff_left, abs(xoff))

        x_tip_check = x_left[i] - r_i
        z_tip_check = z_left[i] + z_curl
        if abs(x_tip_check - x_tip_target) > 1e-9 or abs(z_tip_check - z_tip_target) > 1e-9:
            raise RuntimeError("Left-half tip-tracking consistency check failed.")

    meta_left = {
        "r_abs_min": float(rmin_cmd),
        "r_abs_max": float(rmax_cmd),
        "n_inner_clamped": int(n_inner_left),
        "n_outer_clamped": int(n_outer_left),
        "max_abs_x_offset_from_nominal": float(max_abs_xoff_left),
    }

    write_gcode_two_mirrored_halves_tip_priority(
        out_path=args.out,
        cal=cal,
        center_x_nominal=float(args.center_x),
        center_z_tip=float(args.center_z),
        x_right=x_right,
        z_right=z_right,
        b_right=b_right,
        x_left=x_left,
        z_left=z_left,
        b_left=b_left,
        outer_radius_requested=float(args.outer_radius),
        outer_radius_used=outer_used,
        right_half_npts=len(right_half_pts),
        meta_right=meta_right,
        meta_left=meta_left,
        jog_feed=float(args.jog_feed),
        print_feed_b=float(args.print_feed if args.print_feed_b is None else args.print_feed_b),
        print_feed_c=float(args.print_feed if args.print_feed_c is None else args.print_feed_c),
        c0_deg=c0_deg,
        c180_deg=c180_deg,
    )

    rr_right = _polyval4(cal.pr, b_right)
    rr_left = _polyval4(cal.pr, b_left)
    zz_right = _polyval4(cal.pz, b_right)
    zz_left = _polyval4(cal.pz, b_left)

    print(f"Wrote {args.out}")
    print("Mode: explicit mirrored halves with tip-priority exact tracking and allowed X correction")
    print(f"Star outer radius requested: {args.outer_radius:.3f} mm")
    print(f"Star outer radius used:      {outer_used:.3f} mm")
    print(f"Right-half points: {len(right_half_pts)} (explicit left half generated, single C flip)")
    print(f"Commanded B clamp: [{b_lo:.3f}, {b_hi:.3f}]  calibration B: [{cal.b_min:.3f}, {cal.b_max:.3f}]")
    print(f"Reachable |r(B)| over commanded range: approx [{r_abs_min:.3f}, {r_abs_max:.3f}] mm")
    print(f"Right half B range used: [{np.min(b_right):.3f}, {np.max(b_right):.3f}]")
    print(f"Left  half B range used: [{np.min(b_left):.3f}, {np.max(b_left):.3f}]")
    print(f"Right half r(B) range: [{np.min(rr_right):.3f}, {np.max(rr_right):.3f}]")
    print(f"Left  half r(B) range: [{np.min(rr_left):.3f}, {np.max(rr_left):.3f}]")
    print(f"Right half z(B) range: [{np.min(zz_right):.3f}, {np.max(zz_right):.3f}]")
    print(f"Left  half z(B) range: [{np.min(zz_left):.3f}, {np.max(zz_left):.3f}]")
    print(f"Right half X stage range: [{np.min(x_right):.3f}, {np.max(x_right):.3f}]")
    print(f"Left  half X stage range: [{np.min(x_left):.3f}, {np.max(x_left):.3f}]")
    print(f"Right half Z stage range: [{np.min(z_right):.3f}, {np.max(z_right):.3f}]")
    print(f"Left  half Z stage range: [{np.min(z_left):.3f}, {np.max(z_left):.3f}]")
    print(f"C values used: [{c0_deg:.3f}, {c180_deg:.3f}] (single flip)")
    if meta_right['n_inner_clamped'] or meta_left['n_inner_clamped']:
        print("[info] Inner radial clamp occurred; X correction preserved exact tip position near mirror plane.")
    if meta_right['n_outer_clamped'] or meta_left['n_outer_clamped']:
        print("[info] Outer radial clamp occurred; star radius may need reduction for lower X correction.")
    print(f"Max |X offset from nominal center| right: {meta_right['max_abs_x_offset_from_nominal']:.3f} mm")
    print(f"Max |X offset from nominal center| left : {meta_left['max_abs_x_offset_from_nominal']:.3f} mm")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(
        description="Generate star G-code as two explicit mirrored half-paths, prioritizing exact tip tracking and allowing X offset when B/r limits prevent nominal mirror-plane reach."
    )
    ap.add_argument("--calibration", required=True, help="Path to calibration JSON.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code file.")

    # Placement
    ap.add_argument("--center-x", type=float, default=STAR_CENTER_X,
                    help="Nominal star center X (mirror plane reference). X may offset from this to preserve tip tracking.")
    ap.add_argument("--center-z", type=float, default=STAR_CENTER_Z,
                    help="Star center Z in tip space.")

    # Geometry
    ap.add_argument("--outer-radius", type=float, default=DESIRED_STAR_OUTER_RADIUS,
                    help="Desired outer star radius in mm (auto-scaled down if needed).")
    ap.add_argument("--inner-ratio", type=float, default=INNER_RADIUS_RATIO,
                    help="Inner radius / outer radius for 5-point star.")
    ap.add_argument("--rotation-deg", type=float, default=STAR_ROTATION_DEG,
                    help="Star rotation in XZ plane (deg).")
    ap.add_argument("--samples-per-edge", type=int, default=SAMPLES_PER_EDGE,
                    help="Interpolation points per star edge before half-plane extraction.")

    # Motion
    ap.add_argument("--jog-feed", type=float, default=DEFAULT_JOG_FEED,
                    help="Feedrate for startup positioning (startup C move uses 10x this value).")
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED,
                    help="Feedrate for drawing moves.")
    ap.add_argument("--print-feed-b", type=float, default=DEFAULT_PRINT_FEED_B,
                    help="Optional feedrate override for drawing moves without C-axis rotation (defaults to --print-feed).")
    ap.add_argument("--print-feed-c", type=float, default=DEFAULT_PRINT_FEED_C,
                    help="Optional feedrate override for the drawing transition move that includes the C-axis flip (defaults to --print-feed).")
    ap.add_argument("--safety-margin", type=float, default=RADIUS_SAFETY_MARGIN,
                    help="Margin subtracted from max reachable |r(B)| when auto-scaling.")

    # B/C overrides
    ap.add_argument("--min-b", type=float, default=None, help="Lower bound for commanded B (default: calibration).")
    ap.add_argument("--max-b", type=float, default=None, help="Upper bound for commanded B (default: calibration).")
    ap.add_argument("--c0-deg", type=float, default=0.0, help="C value for +X side (default 0).")
    ap.add_argument("--c180-deg", type=float, default=None, help="C value for -X side (default from calibration or 180).")

    args = ap.parse_args()
    main(args)
