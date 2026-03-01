#!/usr/bin/env python3
"""
Generate G-code for the requested XZ-plane calibration geometries, including:

1) Chevron family (0..45 deg) on the first XZ plane (direct XYZ only, B=0, C=0)
2) Same chevron family on Y+20 plane with tangential extrusion using B pull (calibration cubics), C=180
3) Fan geometry on first plane with lower-half (135..90) + upper-half (85..45) line set (direct XYZ only, B=0, C=0)
4) Tangential version split across planes:
   - lower half on Y+20, C=0
   - upper half on Y+40, C=180, starting from the top common vertex (90 deg first)

Tangential moves use the calibration polynomial fit equations:
    r(B), z(B)
and exact tip tracking:
    stage = tip - [r(B) cos(C), r(B) sin(C), z(B)]

Notes
-----
- This script generates path G-code only (no extrusion/U/E axis commands).
- "Print" vs "travel" is encoded by feedrate and comments.
- For tangential straight segments, B and C are held constant during each segment.
- B is solved per line angle using the tangent direction of the fitted tip curve wrt B.
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------- Defaults ----------------
DEFAULT_OUT = "gcode_generation/xz_geometry_calibration_tangential.gcode"

# Geometry origin (first XZ plane, first figure start point)
DEFAULT_ORIGIN_X = 20.0
DEFAULT_ORIGIN_Y = 20.0
DEFAULT_ORIGIN_Z = -80.0

# Plane offsets in Y
DEFAULT_Y_OFFSET_TANGENTIAL = 20.0   # second XZ plane
DEFAULT_Y_OFFSET_TOP_HALF = 40.0     # third XZ plane for step 4 top half

# Figure 1 chevrons
DEFAULT_FIG1_LENGTH = 80.0
DEFAULT_FIG1_MID_X = 40.0
DEFAULT_FIG1_ANGLE_STEP = 5.0
DEFAULT_FIG1_ANGLE_MAX = 45.0

# Figure 3 fan geometry
DEFAULT_FIG3_X_OFFSET = 100.0
DEFAULT_FIG3_MID_Z = 40.0
DEFAULT_FIG3_TOP_Z = 80.0

# Motion
DEFAULT_TRAVEL_FEED = 1200.0
DEFAULT_PRINT_FEED = 500.0
DEFAULT_ORIENT_FEED = 200.0    # tracked B reorientation feed (fixed tip compensation)
DEFAULT_C_ONLY_FEED = 5000.0   # tracked/direct C reorientation feed
DEFAULT_SAFE_APPROACH_Z = 0.0

# Startup/end poses (stage axes)
DEFAULT_START_X = 20.0
DEFAULT_START_Y = 20.0
DEFAULT_START_Z = -30.0
DEFAULT_START_B = 0.0
DEFAULT_START_C = 0.0
DEFAULT_END_X = 20.0
DEFAULT_END_Y = 20.0
DEFAULT_END_Z = -30.0
DEFAULT_END_B = 0.0
DEFAULT_END_C = 0.0

# Bounding box (optional clamp)
DEFAULT_BBOX_X_MIN = -1e9
DEFAULT_BBOX_X_MAX = 1e9
DEFAULT_BBOX_Y_MIN = -1e9
DEFAULT_BBOX_Y_MAX = 1e9
DEFAULT_BBOX_Z_MIN = -1e9
DEFAULT_BBOX_Z_MAX = 1e9

# Tangential B solver
DEFAULT_B_SOLVER_SAMPLES = 4001
DEFAULT_B_SOLVER_WARN_DEG = 3.0


# ---------------- Data classes ----------------

@dataclass
class Calibration:
    pr: np.ndarray  # cubic coeffs for r(B): [a,b,c,d]
    pz: np.ndarray  # cubic coeffs for z(B): [a,b,c,d]
    b_min: float
    b_max: float
    x_axis: str
    y_axis: str
    z_axis: str
    b_axis: str
    c_axis: str
    c_180_deg: float


@dataclass
class MachineState:
    # stage coordinates (last commanded absolute values)
    x: Optional[float] = None
    y: Optional[float] = None
    z: Optional[float] = None
    b: Optional[float] = None
    c: Optional[float] = None
    # logical tip point for tracked operations (if known)
    tip_xyz: Optional[np.ndarray] = None


@dataclass
class LineSpec:
    start: np.ndarray  # tip coordinates [x,y,z]
    end: np.ndarray    # tip coordinates [x,y,z]
    angle_deg: float   # XZ angle from +X axis (directed), used for tangential B solve
    label: str


# ---------------- Calibration helpers ----------------

def _polyval_cubic(coeffs: Sequence[float], u: Any) -> np.ndarray:
    a, b, c, d = coeffs
    u = np.asarray(u, dtype=float)
    return ((a * u + b) * u + c) * u + d


def _polyder_cubic(coeffs: Sequence[float], u: Any) -> np.ndarray:
    a, b, c, _ = coeffs
    u = np.asarray(u, dtype=float)
    return (3.0 * a * u + 2.0 * b) * u + c


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
        pr=pr,
        pz=pz,
        b_min=b_min,
        b_max=b_max,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        b_axis=b_axis,
        c_axis=c_axis,
        c_180_deg=c_180,
    )


def eval_r(cal: Calibration, b: Any) -> np.ndarray:
    return _polyval_cubic(cal.pr, b)


def eval_z(cal: Calibration, b: Any) -> np.ndarray:
    return _polyval_cubic(cal.pz, b)


def eval_dr(cal: Calibration, b: Any) -> np.ndarray:
    return _polyder_cubic(cal.pr, b)


def eval_dz(cal: Calibration, b: Any) -> np.ndarray:
    return _polyder_cubic(cal.pz, b)


def tip_offset_xyz_physical(cal: Calibration, b: float, c_deg: float) -> np.ndarray:
    r = float(eval_r(cal, b))
    z = float(eval_z(cal, b))
    c = math.radians(float(c_deg))
    return np.array([r * math.cos(c), r * math.sin(c), z], dtype=float)


def stage_xyz_for_tip(cal: Calibration, tip_xyz: np.ndarray, b: float, c_deg: float) -> np.ndarray:
    return np.asarray(tip_xyz, dtype=float) - tip_offset_xyz_physical(cal, b, c_deg)


# ---------------- Tangential solver ----------------

def angle_mod180(angle_deg: float) -> float:
    a = float(angle_deg) % 180.0
    if a < 0:
        a += 180.0
    return a


def angular_diff_mod180(a_deg: float, b_deg: float) -> float:
    """Smallest absolute difference between two orientations in [0,180)."""
    d = (float(a_deg) - float(b_deg)) % 180.0
    if d > 90.0:
        d = 180.0 - d
    return abs(d)


def tangent_angle_xz_for_b(cal: Calibration, b: float, c_deg: float) -> float:
    """Orientation angle (mod 180) of the tip tangent wrt B in XZ projection."""
    dr = float(eval_dr(cal, b))
    dz = float(eval_dz(cal, b))
    c = math.radians(float(c_deg))
    dx = dr * math.cos(c)
    # Y component ignored: we only care about XZ plane orientation
    ang = math.degrees(math.atan2(dz, dx))
    return angle_mod180(ang)


def line_angle_xz(start_xyz: np.ndarray, end_xyz: np.ndarray) -> float:
    d = np.asarray(end_xyz, dtype=float) - np.asarray(start_xyz, dtype=float)
    return angle_mod180(math.degrees(math.atan2(float(d[2]), float(d[0]))))


def figure2_polynomial_target_angle(line: LineSpec) -> float:
    """Map figure-2 line angle to the physical polynomial-angle convention (user corrected).

    Examples requested by user (with C=180):
      0 deg line   ->  90 deg polynomial target
     +15 deg line  ->  75 deg polynomial target
     -15 deg line  -> 165 deg polynomial target
    """
    a = float(line.angle_deg)
    if a >= 0.0:
        return angle_mod180(90.0 - a)
    return angle_mod180(180.0 + a)


def solve_b_for_line_angle(
    cal: Calibration,
    target_angle_deg: float,
    c_deg: float,
    b_lo: float,
    b_hi: float,
    samples: int = DEFAULT_B_SOLVER_SAMPLES,
    prefer_b: Optional[float] = None,
) -> Tuple[float, float]:
    """
    Solve for B whose XZ tangent orientation best matches target line angle (mod 180).

    Returns (b_best, angle_error_deg).
    """
    if b_lo > b_hi:
        b_lo, b_hi = b_hi, b_lo

    bb = np.linspace(float(b_lo), float(b_hi), int(max(101, samples)))
    dr = eval_dr(cal, bb)
    dz = eval_dz(cal, bb)
    dx = dr * math.cos(math.radians(float(c_deg)))
    ang = np.degrees(np.arctan2(dz, dx))
    ang = np.mod(ang, 180.0)

    target = angle_mod180(target_angle_deg)
    # orientation error in [0, 90]
    diff = np.abs((ang - target + 90.0) % 180.0 - 90.0)

    # Tie-break by proximity to prefer_b if provided
    if prefer_b is None:
        idx = int(np.argmin(diff))
    else:
        pref = float(prefer_b)
        # Weighted score: tiny angle priority, then B continuity
        score = diff + 1e-6 * np.abs(bb - pref)
        idx = int(np.argmin(score))

    return float(bb[idx]), float(diff[idx])


# ---------------- G-code writer ----------------

class GCodeWriter:
    def __init__(
        self,
        cal: Calibration,
        travel_feed: float,
        print_feed: float,
        orient_feed: float,
        c_only_feed: float,
        bbox: Dict[str, float],
    ):
        self.cal = cal
        self.travel_feed = float(travel_feed)
        self.print_feed = float(print_feed)
        self.orient_feed = float(orient_feed)
        self.c_only_feed = float(c_only_feed)
        self.bbox = dict(bbox)
        self.lines: List[str] = []
        self.state = MachineState()
        self.warnings: List[str] = []
        self._last_feed: Optional[float] = None

    def comment(self, text: str) -> None:
        self.lines.append(f"; {text}")

    def raw(self, text: str) -> None:
        self.lines.append(text)

    def _fmt_axis(self, axis: str, value: float) -> str:
        return f"{axis}{float(value):.3f}"

    def _clamp_xyz(self, x: float, y: float, z: float, context: str) -> Tuple[float, float, float]:
        def _clamp(v: float, lo: float, hi: float, axis: str) -> float:
            if v < lo:
                self.warnings.append(f"{context}: {axis}={v:.3f} < {lo:.3f}, clamped")
                return lo
            if v > hi:
                self.warnings.append(f"{context}: {axis}={v:.3f} > {hi:.3f}, clamped")
                return hi
            return v

        x = _clamp(float(x), self.bbox["x_min"], self.bbox["x_max"], "X")
        y = _clamp(float(y), self.bbox["y_min"], self.bbox["y_max"], "Y")
        z = _clamp(float(z), self.bbox["z_min"], self.bbox["z_max"], "Z")
        return x, y, z

    def _emit_g1(
        self,
        axes: List[Tuple[str, float]],
        feed: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> None:
        parts = ["G1"]
        parts.extend(self._fmt_axis(ax, val) for ax, val in axes)
        if feed is not None and (self._last_feed is None or abs(self._last_feed - float(feed)) > 1e-9):
            parts.append(f"F{float(feed):.0f}")
            self._last_feed = float(feed)
        if comment:
            self.lines.append(" ".join(parts) + f" ; {comment}")
        else:
            self.lines.append(" ".join(parts))

    # --- direct stage-space moves (XYZ or XYZBC as requested) ---
    def move_stage(
        self,
        x: Optional[float] = None,
        y: Optional[float] = None,
        z: Optional[float] = None,
        b: Optional[float] = None,
        c: Optional[float] = None,
        feed: Optional[float] = None,
        comment: Optional[str] = None,
        clamp_xyz: bool = True,
        xyz_only: bool = False,
    ) -> None:
        axes: List[Tuple[str, float]] = []

        tx = self.state.x if x is None else float(x)
        ty = self.state.y if y is None else float(y)
        tz = self.state.z if z is None else float(z)

        if (tx is not None) and (ty is not None) and (tz is not None) and clamp_xyz:
            tx, ty, tz = self._clamp_xyz(tx, ty, tz, context=comment or "stage move")

        if x is not None:
            axes.append((self.cal.x_axis, tx))
            self.state.x = tx
        if y is not None:
            axes.append((self.cal.y_axis, ty))
            self.state.y = ty
        if z is not None:
            axes.append((self.cal.z_axis, tz))
            self.state.z = tz

        if not xyz_only:
            if b is not None:
                axes.append((self.cal.b_axis, float(b)))
                self.state.b = float(b)
            if c is not None:
                axes.append((self.cal.c_axis, float(c)))
                self.state.c = float(c)
        else:
            # state B/C unchanged unless explicitly moved in another call
            pass

        if not axes:
            return
        self._emit_g1(axes, feed=feed, comment=comment)
        if xyz_only:
            self.state.tip_xyz = None  # direct stage moves do not preserve tracked tip semantics

    # --- tracked tip-space moves (exact stage compensation using B/C) ---
    def move_tip_tracked(
        self,
        tip_xyz: Sequence[float],
        b: float,
        c: float,
        feed: float,
        comment: Optional[str] = None,
    ) -> None:
        tip = np.asarray(tip_xyz, dtype=float)
        stage = stage_xyz_for_tip(self.cal, tip, float(b), float(c))
        sx, sy, sz = self._clamp_xyz(stage[0], stage[1], stage[2], context=comment or "tracked move")
        self._emit_g1(
            [
                (self.cal.x_axis, sx),
                (self.cal.y_axis, sy),
                (self.cal.z_axis, sz),
                (self.cal.b_axis, float(b)),
                (self.cal.c_axis, float(c)),
            ],
            feed=float(feed),
            comment=comment,
        )
        self.state.x, self.state.y, self.state.z = sx, sy, sz
        self.state.b, self.state.c = float(b), float(c)
        self.state.tip_xyz = tip.copy()

    def reorient_at_fixed_tip(self, tip_xyz: Sequence[float], b: float, c: float, comment: str = "reorient at fixed tip") -> None:
        """Reorient while keeping the same tip point, splitting C and B changes so each uses its requested feed."""
        tip = np.asarray(tip_xyz, dtype=float)
        bt = float(b)
        ct = float(c)

        # If tracked tip is unknown (or differs), fall back to a single tracked move.
        if self.state.tip_xyz is None or np.linalg.norm(np.asarray(self.state.tip_xyz, dtype=float) - tip) > 1e-9:
            self.move_tip_tracked(tip_xyz=tip, b=bt, c=ct, feed=self.orient_feed, comment=comment)
            return

        b_cur = bt if self.state.b is None else float(self.state.b)
        c_cur = ct if self.state.c is None else float(self.state.c)
        moved = False

        # Change C first (fast C feed), with fixed tip and current B.
        if abs(c_cur - ct) > 1e-9:
            self.move_tip_tracked(tip_xyz=tip, b=b_cur, c=ct, feed=self.c_only_feed, comment=f"{comment} (C)")
            c_cur = ct
            moved = True

        # Then change B (B feed), with fixed tip and final C.
        if abs(b_cur - bt) > 1e-9:
            self.move_tip_tracked(tip_xyz=tip, b=bt, c=ct, feed=self.orient_feed, comment=f"{comment} (B)")
            b_cur = bt
            moved = True

        if not moved:
            # Keep a no-op-ish tracked command semantics only when needed by caller comments/state.
            self.move_tip_tracked(tip_xyz=tip, b=bt, c=ct, feed=self.orient_feed, comment=comment)

    def travel_tip_ordered(
        self,
        start_tip_xyz: Sequence[float],
        end_tip_xyz: Sequence[float],
        b: float,
        c: float,
        order: str,
        comment_prefix: str = "travel",
    ) -> np.ndarray:
        """Ordered 2-axis tip travel (x then z, or z then x) with exact B/C compensation."""
        s = np.asarray(start_tip_xyz, dtype=float)
        e = np.asarray(end_tip_xyz, dtype=float)
        cur = s.copy()

        # Ensure orientation first (without moving tip)
        self.reorient_at_fixed_tip(cur, b=float(b), c=float(c), comment=f"{comment_prefix}: set B/C")

        if order.lower() == "xz":
            if abs(e[0] - cur[0]) > 1e-9:
                cur[0] = e[0]
                self.move_tip_tracked(cur, b=float(b), c=float(c), feed=self.travel_feed,
                                      comment=f"{comment_prefix}: X-first")
            if abs(e[2] - cur[2]) > 1e-9:
                cur[2] = e[2]
                self.move_tip_tracked(cur, b=float(b), c=float(c), feed=self.travel_feed,
                                      comment=f"{comment_prefix}: Z-second")
            if abs(e[1] - cur[1]) > 1e-9:
                cur[1] = e[1]
                self.move_tip_tracked(cur, b=float(b), c=float(c), feed=self.travel_feed,
                                      comment=f"{comment_prefix}: Y-final")
        elif order.lower() == "zx":
            if abs(e[2] - cur[2]) > 1e-9:
                cur[2] = e[2]
                self.move_tip_tracked(cur, b=float(b), c=float(c), feed=self.travel_feed,
                                      comment=f"{comment_prefix}: Z-first")
            if abs(e[0] - cur[0]) > 1e-9:
                cur[0] = e[0]
                self.move_tip_tracked(cur, b=float(b), c=float(c), feed=self.travel_feed,
                                      comment=f"{comment_prefix}: X-second")
            if abs(e[1] - cur[1]) > 1e-9:
                cur[1] = e[1]
                self.move_tip_tracked(cur, b=float(b), c=float(c), feed=self.travel_feed,
                                      comment=f"{comment_prefix}: Y-final")
        elif order.lower() == "yx":
            if abs(e[1] - cur[1]) > 1e-9:
                cur[1] = e[1]
                self.move_tip_tracked(cur, b=float(b), c=float(c), feed=self.travel_feed,
                                      comment=f"{comment_prefix}: Y-first")
            if abs(e[0] - cur[0]) > 1e-9:
                cur[0] = e[0]
                self.move_tip_tracked(cur, b=float(b), c=float(c), feed=self.travel_feed,
                                      comment=f"{comment_prefix}: X-second")
            if abs(e[2] - cur[2]) > 1e-9:
                cur[2] = e[2]
                self.move_tip_tracked(cur, b=float(b), c=float(c), feed=self.travel_feed,
                                      comment=f"{comment_prefix}: Z-third")
        else:
            raise ValueError(f"Unsupported travel order '{order}' (use xz or zx or yx)")
        return cur

    def finalize(self, out_path: str) -> None:
        if self.warnings:
            self.comment("bbox clamp warnings:")
            for w in self.warnings:
                self.comment(f"WARNING: {w}")
        self.comment("end of file")
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        Path(out_path).write_text("\n".join(self.lines) + "\n")


# ---------------- Geometry builders ----------------

def p3(x: float, y: float, z: float) -> np.ndarray:
    return np.array([float(x), float(y), float(z)], dtype=float)


def build_figure1_chevrons(
    origin_xyz: np.ndarray,
    y_plane: float,
    length: float,
    mid_x: float,
    angle_step_deg: float,
    angle_max_deg: float,
) -> List[List[LineSpec]]:
    """
    Returns a list of polylines; each polyline is a list of 1 or 2 LineSpec segments.
    First polyline is the horizontal line, then chevrons for +5/-5 ... +45/-45.
    """
    x0, _, z0 = map(float, origin_xyz)
    y = float(y_plane)
    linesets: List[List[LineSpec]] = []

    # Horizontal line (0 deg), length = 80 mm (default)
    s = p3(x0, y, z0)
    e = p3(x0 + length, y, z0)
    linesets.append([LineSpec(start=s, end=e, angle_deg=0.0, label="fig1 horizontal 0deg")])

    n = int(round(angle_max_deg / angle_step_deg))
    for k in range(1, n + 1):
        a = float(k) * float(angle_step_deg)
        h = float(mid_x) * math.tan(math.radians(a))
        p0 = p3(x0, y, z0)
        p1 = p3(x0 + mid_x, y, z0 + h)
        p2 = p3(x0 + length, y, z0)
        linesets.append([
            LineSpec(start=p0, end=p1, angle_deg=+a, label=f"fig1 {a:.0f}deg up"),
            LineSpec(start=p1, end=p2, angle_deg=-a, label=f"fig1 {a:.0f}deg down"),
        ])
    return linesets


def endpoint_from_origin_and_target_z(x0: float, y: float, z0: float, angle_deg: float, z_rel: float) -> np.ndarray:
    """Line from origin at angle_deg until reaching z_rel (relative to origin)."""
    th = math.radians(float(angle_deg))
    dz = float(z_rel)
    # x_rel = dz / tan(theta); for 90° -> x_rel = 0
    if abs(math.cos(th)) < 1e-12:
        x_rel = 0.0
    else:
        t = dz / math.sin(th)
        x_rel = t * math.cos(th)
    return p3(x0 + x_rel, y, z0 + dz)


def build_figure3_points(
    base_origin_xyz: np.ndarray,
    y_plane: float,
    x_offset: float,
    mid_z_rel: float,
    top_z_rel: float,
) -> Dict[str, Any]:
    """
    Constructs the fan geometry from user description.

    Returns dict with:
      O, T, lower_endpoints (by angle 135..95), top_startpoints_by_angle (85..45 mapped to lower endpoints 95..135)
    """
    x0, _, z0 = map(float, base_origin_xyz)
    y = float(y_plane)
    ox = x0 + float(x_offset)
    O = p3(ox, y, z0)
    T = p3(ox, y, z0 + float(top_z_rel))

    lower_endpoints: Dict[int, np.ndarray] = {}
    for a in range(135, 90, -5):  # 135,130,...95
        lower_endpoints[a] = endpoint_from_origin_and_target_z(ox, y, z0, a, mid_z_rel)

    # Map top-half line angle (85..45) to the corresponding lower endpoint of (180-angle)
    top_startpoints: Dict[int, np.ndarray] = {}
    for a in range(85, 40, -5):  # 85,80,...45
        top_startpoints[a] = lower_endpoints[180 - a]

    return {
        "O": O,
        "T": T,
        "lower_endpoints": lower_endpoints,
        "top_startpoints": top_startpoints,
        "mid_z_rel": float(mid_z_rel),
        "top_z_rel": float(top_z_rel),
    }


# ---------------- Program generation helpers ----------------

def write_header(
    g: GCodeWriter,
    args: argparse.Namespace,
    cal: Calibration,
) -> None:
    g.comment("generated by generate_xz_geometry_tangential_gcode.py")
    g.comment("Requested XZ-plane geometry sequence with direct and tangential variants")
    g.comment(f"Axes: X->{cal.x_axis}, Y->{cal.y_axis}, Z->{cal.z_axis}, B->{cal.b_axis}, C->{cal.c_axis}")
    g.comment(f"Calibration B range: [{cal.b_min:.3f}, {cal.b_max:.3f}]")
    g.comment(f"Feeds: travel={g.travel_feed:.1f}, print={g.print_feed:.1f}, B-feed={g.orient_feed:.1f}, C-feed={g.c_only_feed:.1f}")
    g.comment(f"Origin (first plane): X={args.origin_x:.3f} Y={args.origin_y:.3f} Z={args.origin_z:.3f}")
    g.comment(f"Y plane offsets: tangential={args.y_offset_tangential:.3f}, top_half={args.y_offset_top_half:.3f}")
    g.raw("G90")
    g.raw("G21")


def safe_startup(g: GCodeWriter, args: argparse.Namespace) -> None:
    g.comment("safe startup")
    g.move_stage(z=float(args.safe_approach_z), b=float(args.start_b), c=float(args.start_c), feed=g.travel_feed,
                 comment="safe approach Z + set B/C")
    g.move_stage(x=float(args.start_x), y=float(args.start_y), feed=g.travel_feed, comment="startup XY")
    g.move_stage(z=float(args.start_z), feed=g.travel_feed, comment="startup Z")


def safe_shutdown(g: GCodeWriter, args: argparse.Namespace) -> None:
    g.comment("safe shutdown")
    g.move_stage(z=float(args.safe_approach_z), b=float(args.end_b), c=float(args.end_c), feed=g.travel_feed,
                 comment="safe approach Z + set end B/C")
    g.move_stage(x=float(args.end_x), y=float(args.end_y), feed=g.travel_feed, comment="end XY")
    g.move_stage(z=float(args.end_z), feed=g.travel_feed, comment="end Z")


def direct_travel_xz_ordered(g: GCodeWriter, target: np.ndarray, order: str, comment_prefix: str) -> None:
    """Direct XYZ-only ordered moves in the active plane (for non-tangential sections)."""
    tx, ty, tz = map(float, target)
    if g.state.x is None or g.state.y is None or g.state.z is None:
        # unknown current stage position -> direct move
        g.move_stage(x=tx, y=ty, z=tz, feed=g.travel_feed, comment=f"{comment_prefix}: init", xyz_only=True)
        return

    if order.lower() == "xz":
        if abs(float(g.state.x) - tx) > 1e-9:
            g.move_stage(x=tx, feed=g.travel_feed, comment=f"{comment_prefix}: X-first", xyz_only=True)
        if abs(float(g.state.z) - tz) > 1e-9:
            g.move_stage(z=tz, feed=g.travel_feed, comment=f"{comment_prefix}: Z-second", xyz_only=True)
        if abs(float(g.state.y) - ty) > 1e-9:
            g.move_stage(y=ty, feed=g.travel_feed, comment=f"{comment_prefix}: Y-final", xyz_only=True)
    elif order.lower() == "zx":
        if abs(float(g.state.z) - tz) > 1e-9:
            g.move_stage(z=tz, feed=g.travel_feed, comment=f"{comment_prefix}: Z-first", xyz_only=True)
        if abs(float(g.state.x) - tx) > 1e-9:
            g.move_stage(x=tx, feed=g.travel_feed, comment=f"{comment_prefix}: X-second", xyz_only=True)
        if abs(float(g.state.y) - ty) > 1e-9:
            g.move_stage(y=ty, feed=g.travel_feed, comment=f"{comment_prefix}: Y-final", xyz_only=True)
    else:
        raise ValueError(f"Unsupported direct travel order '{order}'")


def direct_print_line(g: GCodeWriter, line: LineSpec) -> None:
    # Travel to start (direct XYZ only)
    sx, sy, sz = map(float, line.start)
    ex, ey, ez = map(float, line.end)
    g.move_stage(x=sx, y=sy, z=sz, b=0.0, c=0.0, feed=g.travel_feed, comment=f"travel to {line.label} start", xyz_only=True)
    g.move_stage(x=ex, y=ey, z=ez, feed=g.print_feed, comment=f"PRINT {line.label}", xyz_only=True)


def direct_return_fig1_lift_retract(g: GCodeWriter, origin: np.ndarray, z_lift: float = 50.0, x_retract: float = 80.0) -> None:
    """User-requested figure-1 return motion: +Z lift, -X retract, then -Z back to origin height."""
    if g.state.x is None or g.state.y is None or g.state.z is None:
        direct_travel_xz_ordered(g, origin, order="xz", comment_prefix="fig1 return (fallback)")
        return
    ox, oy, oz = map(float, origin)
    # 1) go up fully in Z by +z_lift from current point
    g.move_stage(z=float(g.state.z) + float(z_lift), feed=g.travel_feed,
                 comment=f"fig1 return: lift Z +{float(z_lift):.1f}mm", xyz_only=True)
    # 2) retract X by x_retract (user requested 80 mm); in figure-1 this returns to common origin X
    g.move_stage(x=float(g.state.x) - float(x_retract), feed=g.travel_feed,
                 comment=f"fig1 return: retract X -{float(x_retract):.1f}mm", xyz_only=True)
    # 3) go back down to the origin Z (user requested down 50 mm)
    g.move_stage(z=oz, y=oy, feed=g.travel_feed,
                 comment="fig1 return: lower Z to origin", xyz_only=True)
    # Snap X to exact origin in case of accumulated mismatch / nonstandard length
    if abs(float(g.state.x) - ox) > 1e-9:
        g.move_stage(x=ox, feed=g.travel_feed, comment="fig1 return: align exact origin X", xyz_only=True)


def solve_b_for_lines(
    cal: Calibration,
    lines: Sequence[LineSpec],
    c_deg: float,
    b_lo: float,
    b_hi: float,
    samples: int,
    warn_thresh_deg: float,
    target_angle_transform: Optional[Callable[[LineSpec], float]] = None,
) -> Tuple[List[Tuple[float, float, float]], List[str]]:
    """Returns list of (target_angle, b, err_deg) and warnings."""
    out: List[Tuple[float, float, float]] = []
    warns: List[str] = []
    prev_b: Optional[float] = None
    for ln in lines:
        target = float(target_angle_transform(ln)) if target_angle_transform is not None else line_angle_xz(ln.start, ln.end)
        b, err = solve_b_for_line_angle(
            cal=cal,
            target_angle_deg=target,
            c_deg=float(c_deg),
            b_lo=float(b_lo),
            b_hi=float(b_hi),
            samples=int(samples),
            prefer_b=prev_b,
        )
        out.append((target, b, err))
        if err > float(warn_thresh_deg):
            warns.append(
                f"{ln.label}: target angle {target:.2f} deg not well matched at C={c_deg:.1f}; "
                f"best B={b:.4f}, error={err:.2f} deg"
            )
        prev_b = b
    return out, warns


def tangential_print_line(g: GCodeWriter, cal: Calibration, line: LineSpec, b: float, c_deg: float) -> None:
    """Exact tip-tracked line print with constant B/C for the segment."""
    # 1) Move to line start with fixed-tip tracking at target B/C (travel)
    g.reorient_at_fixed_tip(line.start, b=float(b), c=float(c_deg), comment=f"{line.label}: set B/C at start")
    g.move_tip_tracked(line.start, b=float(b), c=float(c_deg), feed=g.travel_feed, comment=f"travel to {line.label} start")
    # 2) Print the segment
    g.move_tip_tracked(line.end, b=float(b), c=float(c_deg), feed=g.print_feed, comment=f"PRINT {line.label}")


def generate_program(args: argparse.Namespace) -> Tuple[str, List[str]]:
    cal = load_calibration(args.calibration)

    # Commanded B bounds
    b_lo = cal.b_min if args.min_b is None else float(args.min_b)
    b_hi = cal.b_max if args.max_b is None else float(args.max_b)
    if b_lo > b_hi:
        b_lo, b_hi = b_hi, b_lo

    c180 = float(cal.c_180_deg if args.c180_deg is None else args.c180_deg)

    bbox = {
        "x_min": float(args.bbox_x_min), "x_max": float(args.bbox_x_max),
        "y_min": float(args.bbox_y_min), "y_max": float(args.bbox_y_max),
        "z_min": float(args.bbox_z_min), "z_max": float(args.bbox_z_max),
    }
    if bbox["x_min"] > bbox["x_max"]:
        bbox["x_min"], bbox["x_max"] = bbox["x_max"], bbox["x_min"]
    if bbox["y_min"] > bbox["y_max"]:
        bbox["y_min"], bbox["y_max"] = bbox["y_max"], bbox["y_min"]
    if bbox["z_min"] > bbox["z_max"]:
        bbox["z_min"], bbox["z_max"] = bbox["z_max"], bbox["z_min"]

    g = GCodeWriter(
        cal=cal,
        travel_feed=float(args.travel_feed),
        print_feed=float(args.print_feed),
        orient_feed=float(args.orient_feed),
        c_only_feed=float(args.c_only_feed),
        bbox=bbox,
    )
    warnings: List[str] = []

    write_header(g, args, cal)
    safe_startup(g, args)

    origin = p3(float(args.origin_x), float(args.origin_y), float(args.origin_z))
    y0 = float(args.origin_y)
    y1 = float(args.origin_y + args.y_offset_tangential)
    y2 = float(args.origin_y + args.y_offset_top_half)

    # ---------- 1) Figure 1 on first plane, direct XYZ only (B=0, C=0) ----------
    g.comment("STEP 1: Figure 1 chevrons on first XZ plane (direct XYZ only, B=0, C=0)")
    g.move_stage(b=0.0, c=0.0, feed=g.travel_feed, comment="set B=0, C=0 for direct printing")
    fig1_plane0 = build_figure1_chevrons(
        origin_xyz=origin,
        y_plane=y0,
        length=float(args.fig1_length),
        mid_x=float(args.fig1_mid_x),
        angle_step_deg=float(args.fig1_angle_step),
        angle_max_deg=float(args.fig1_angle_max),
    )
    for poly in fig1_plane0:
        # travel to polyline start (direct XYZ)
        direct_travel_xz_ordered(g, poly[0].start, order="xz", comment_prefix=f"{poly[0].label} start travel")
        # print segments
        # ensure B/C are not emitted during direct-only motion; keep XYZ only
        for seg in poly:
            g.move_stage(x=float(seg.end[0]), y=float(seg.end[1]), z=float(seg.end[2]), feed=g.print_feed,
                         comment=f"PRINT {seg.label}", xyz_only=True)
        # User-requested figure-1 return: lift +50 Z, retract -80 X, then lower Z
        direct_return_fig1_lift_retract(g, poly[0].start, z_lift=50.0, x_retract=float(args.fig1_length))

    # ---------- 2) Figure 1 on second plane, tangential (C=180) ----------
    g.comment("STEP 2: Figure 1 chevrons on second XZ plane (Y+20), tangential, C=180")
    g.comment("Figure-2 tangential B targets use user-corrected physical polynomial-angle mapping (e.g., 0->90, +15->75, -15->165)")
    # Move to new plane first in stage/direct mode, then switch to tracked.
    g.move_stage(y=y1, feed=g.travel_feed, comment="move to second XZ plane (Y+20)")
    # Initialize logical tip position to origin on new plane for tracked operations.
    current_tip = p3(float(args.origin_x), y1, float(args.origin_z))
    # Use C-only preposition if desired (like sample script pattern)
    if not args.no_c_preposition:
        g.move_stage(c=c180, feed=g.c_only_feed, comment="optional C-only pre-position to 180 deg")
    # Put tracked state at current tip with B=0, C=180 (exact fixed tip)
    g.reorient_at_fixed_tip(current_tip, b=0.0, c=c180, comment="initialize tracked state on second plane")

    fig1_plane1 = build_figure1_chevrons(
        origin_xyz=origin,
        y_plane=y1,
        length=float(args.fig1_length),
        mid_x=float(args.fig1_mid_x),
        angle_step_deg=float(args.fig1_angle_step),
        angle_max_deg=float(args.fig1_angle_max),
    )
    for poly in fig1_plane1:
        # Solve B independently per segment using fixed C=180
        seg_solutions, warns = solve_b_for_lines(
            cal=cal,
            lines=poly,
            c_deg=c180,
            b_lo=b_lo,
            b_hi=b_hi,
            samples=int(args.b_solver_samples),
            warn_thresh_deg=float(args.b_solver_warn_deg),
            target_angle_transform=figure2_polynomial_target_angle,
        )
        warnings.extend(warns)

        # Travel tip to poly start (keep first segment B/C)
        first_b = seg_solutions[0][1]
        g.travel_tip_ordered(current_tip, poly[0].start, b=first_b, c=c180, order="xz",
                             comment_prefix=f"{poly[0].label} poly start travel")
        current_tip = poly[0].start.copy()

        # Print each segment with constant B/C; reorient at junctions to keep segment tangency constant
        for seg, (_ang, b_seg, _err) in zip(poly, seg_solutions):
            g.reorient_at_fixed_tip(current_tip, b=b_seg, c=c180, comment=f"{seg.label}: set B/C")
            if np.linalg.norm(seg.start - current_tip) > 1e-9:
                g.move_tip_tracked(seg.start, b=b_seg, c=c180, feed=g.travel_feed, comment=f"travel to {seg.label} start")
                current_tip = seg.start.copy()
            g.move_tip_tracked(seg.end, b=b_seg, c=c180, feed=g.print_feed, comment=f"PRINT {seg.label} (tangential)")
            current_tip = seg.end.copy()

        # Return to common origin before next chevron (ordered X then Z; user didn't specify but consistent)
        g.travel_tip_ordered(current_tip, poly[0].start, b=first_b, c=c180, order="xz",
                             comment_prefix=f"return after {poly[0].label}")
        current_tip = poly[0].start.copy()

    # ---------- 3) Figure 3 on first plane, direct XYZ only ----------
    g.comment("STEP 3: Figure 3 full fan on first XZ plane (direct XYZ only, B=0, C=0)")
    g.move_stage(b=0.0, c=0.0, feed=g.travel_feed, comment="set B=0, C=0")
    # Move back to first plane in Y (direct)
    g.move_stage(y=y0, feed=g.travel_feed, comment="return to first XZ plane")

    fig3_0 = build_figure3_points(
        base_origin_xyz=origin,
        y_plane=y0,
        x_offset=float(args.fig3_x_offset),
        mid_z_rel=float(args.fig3_mid_z),
        top_z_rel=float(args.fig3_top_z),
    )
    O0 = fig3_0["O"]
    T0 = fig3_0["T"]
    lower0: Dict[int, np.ndarray] = fig3_0["lower_endpoints"]
    top_starts0: Dict[int, np.ndarray] = fig3_0["top_startpoints"]

    # Go to figure 3 origin
    direct_travel_xz_ordered(g, O0, order="xz", comment_prefix="fig3 origin travel")

    # Lower half: 135..95 to z_rel=40, return to origin each time; then 90 deg vertical to z_rel=80
    for a in range(135, 90, -5):
        endp = lower0[a]
        g.move_stage(x=float(endp[0]), y=float(endp[1]), z=float(endp[2]), feed=g.print_feed,
                     comment=f"PRINT fig3 lower {a}deg", xyz_only=True)
        # Return to origin (general safe order: move x back before z down)
        direct_travel_xz_ordered(g, O0, order="xz", comment_prefix=f"return from fig3 lower {a}deg")

    # 90 degree line (extended to z_rel=80)
    g.move_stage(x=float(T0[0]), y=float(T0[1]), z=float(T0[2]), feed=g.print_feed,
                 comment="PRINT fig3 lower 90deg (extended to z_rel=80)", xyz_only=True)

    # Upper half: start at end of 95deg line, print 85..45 to top common vertex.
    # Position from top vertex to first startpoint with X first then Z down (explicit user request)
    start_85 = top_starts0[85]
    direct_travel_xz_ordered(g, start_85, order="xz", comment_prefix="go to fig3 upper 85deg start (end of 95deg line)")
    g.move_stage(x=float(T0[0]), y=float(T0[1]), z=float(T0[2]), feed=g.print_feed,
                 comment="PRINT fig3 upper 85deg to top common vertex", xyz_only=True)

    for a in range(80, 40, -5):
        startp = top_starts0[a]
        # Going back down: move x first, then z down (user request)
        direct_travel_xz_ordered(g, startp, order="xz", comment_prefix=f"go to fig3 upper {a}deg start")
        g.move_stage(x=float(T0[0]), y=float(T0[1]), z=float(T0[2]), feed=g.print_feed,
                     comment=f"PRINT fig3 upper {a}deg to top common vertex", xyz_only=True)

    # ---------- 4a) Figure 3 bottom half on second plane, tangential (C=0) ----------
    g.comment("STEP 4a: Figure 3 bottom half on second XZ plane (Y+20), tangential, C=0")
    fig3_1 = build_figure3_points(
        base_origin_xyz=origin,
        y_plane=y1,
        x_offset=float(args.fig3_x_offset),
        mid_z_rel=float(args.fig3_mid_z),
        top_z_rel=float(args.fig3_top_z),
    )
    O1 = fig3_1["O"]
    T1 = fig3_1["T"]
    lower1: Dict[int, np.ndarray] = fig3_1["lower_endpoints"]

    # Set tracked state at O1 with C=0, B=0 (exact fixed tip)
    g.reorient_at_fixed_tip(O1, b=0.0, c=0.0, comment="initialize tracked state for fig3 bottom half tangential")
    current_tip = O1.copy()

    lower_lines_tangential: List[LineSpec] = []
    for a in range(135, 90, -5):
        lower_lines_tangential.append(LineSpec(start=O1.copy(), end=lower1[a].copy(), angle_deg=float(a), label=f"fig3 lower {a}deg"))
    lower_lines_tangential.append(LineSpec(start=O1.copy(), end=T1.copy(), angle_deg=90.0, label="fig3 lower 90deg vertical"))

    lower_solutions, warns = solve_b_for_lines(
        cal=cal,
        lines=lower_lines_tangential,
        c_deg=0.0,
        b_lo=b_lo,
        b_hi=b_hi,
        samples=int(args.b_solver_samples),
        warn_thresh_deg=float(args.b_solver_warn_deg),
    )
    warnings.extend(warns)

    for ln, (_ang, b_seg, _err) in zip(lower_lines_tangential, lower_solutions):
        # Print from origin to endpoint
        g.travel_tip_ordered(current_tip, ln.start, b=b_seg, c=0.0, order="xz", comment_prefix=f"{ln.label} start travel")
        current_tip = ln.start.copy()
        g.reorient_at_fixed_tip(current_tip, b=b_seg, c=0.0, comment=f"{ln.label}: set B/C")
        g.move_tip_tracked(ln.end, b=b_seg, c=0.0, feed=g.print_feed, comment=f"PRINT {ln.label} (tangential)")
        current_tip = ln.end.copy()

        # Return to starting point O1. User request: move X positively first then decrease Z.
        # Endpoints are left/up of origin, so X-first then Z-second satisfies this.
        g.travel_tip_ordered(current_tip, O1, b=b_seg, c=0.0, order="xz",
                             comment_prefix=f"return to fig3 bottom origin after {ln.label}")
        current_tip = O1.copy()

    # ---------- 4b) Figure 3 top half on third plane (Y+40), tangential (C=180), start from top vertex ----------
    g.comment("STEP 4b: Figure 3 top half on third XZ plane (Y+40), tangential, C=180")
    fig3_2 = build_figure3_points(
        base_origin_xyz=origin,
        y_plane=y2,
        x_offset=float(args.fig3_x_offset),
        mid_z_rel=float(args.fig3_mid_z),
        top_z_rel=float(args.fig3_top_z),
    )
    O2 = fig3_2["O"]
    T2 = fig3_2["T"]
    top_starts2: Dict[int, np.ndarray] = fig3_2["top_startpoints"]

    # Top-half lines now explicitly start from top common vertex, draw 90 first
    top_lines_tangential: List[LineSpec] = [
        LineSpec(start=T2.copy(), end=O2.copy(), angle_deg=90.0, label="fig3 top half 90deg vertical from top")
    ]
    for a in range(85, 40, -5):
        top_lines_tangential.append(LineSpec(start=T2.copy(), end=top_starts2[a].copy(), angle_deg=float(a), label=f"fig3 top half {a}deg from top"))

    top_solutions, warns = solve_b_for_lines(
        cal=cal,
        lines=top_lines_tangential,
        c_deg=c180,
        b_lo=b_lo,
        b_hi=b_hi,
        samples=int(args.b_solver_samples),
        warn_thresh_deg=float(args.b_solver_warn_deg),
    )
    warnings.extend(warns)

    # Initialize tracked state at top vertex with C=180
    if not args.no_c_preposition:
        g.move_stage(c=c180, feed=g.c_only_feed, comment="optional C-only pre-position to 180 deg for top half")
    g.reorient_at_fixed_tip(T2, b=0.0, c=c180, comment="initialize tracked state for fig3 top half tangential")
    current_tip = T2.copy()

    for ln, (_ang, b_seg, _err) in zip(top_lines_tangential, top_solutions):
        # Start each line from top common vertex (user request)
        g.travel_tip_ordered(current_tip, T2, b=b_seg, c=c180, order="zx", comment_prefix=f"{ln.label} start at top vertex")
        current_tip = T2.copy()
        g.reorient_at_fixed_tip(current_tip, b=b_seg, c=c180, comment=f"{ln.label}: set B/C")
        g.move_tip_tracked(ln.end, b=b_seg, c=c180, feed=g.print_feed, comment=f"PRINT {ln.label} (tangential)")
        current_tip = ln.end.copy()

        # Return to top vertex; user request: move up, then move X back.
        g.travel_tip_ordered(current_tip, T2, b=b_seg, c=c180, order="zx",
                             comment_prefix=f"return to top vertex after {ln.label}")
        current_tip = T2.copy()

    safe_shutdown(g, args)
    g.finalize(args.out)
    return args.out, warnings + [f"bbox clamping warnings: {len(g.warnings)}"]


# ---------------- CLI ----------------

def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Generate G-code for the requested XZ-plane geometry sequence (direct + tangential calibration-tracked variants)."
    )
    ap.add_argument("--calibration", required=True, help="Path to calibration JSON containing cubic r(B), z(B) fits.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code path.")

    # Geometry origin (user parameter requested)
    ap.add_argument("--origin-x", type=float, default=DEFAULT_ORIGIN_X, help="Start point X for figure 1 on the first XZ plane.")
    ap.add_argument("--origin-y", type=float, default=DEFAULT_ORIGIN_Y, help="Base Y plane for the first XZ plane.")
    ap.add_argument("--origin-z", type=float, default=DEFAULT_ORIGIN_Z, help="Start point Z for figure 1 on the first XZ plane.")

    # Plane offsets
    ap.add_argument("--y-offset-tangential", type=float, default=DEFAULT_Y_OFFSET_TANGENTIAL,
                    help="Y offset for the second XZ plane (step 2 and step 4 bottom half).")
    ap.add_argument("--y-offset-top-half", type=float, default=DEFAULT_Y_OFFSET_TOP_HALF,
                    help="Y offset for the third XZ plane used for step 4 top half.")

    # Figure 1 params
    ap.add_argument("--fig1-length", type=float, default=DEFAULT_FIG1_LENGTH, help="Figure 1 total x length.")
    ap.add_argument("--fig1-mid-x", type=float, default=DEFAULT_FIG1_MID_X, help="Figure 1 midpoint x_rel for the apex.")
    ap.add_argument("--fig1-angle-step", type=float, default=DEFAULT_FIG1_ANGLE_STEP, help="Figure 1 angle increment (deg).")
    ap.add_argument("--fig1-angle-max", type=float, default=DEFAULT_FIG1_ANGLE_MAX, help="Figure 1 maximum chevron angle (deg).")

    # Figure 3 params
    ap.add_argument("--fig3-x-offset", type=float, default=DEFAULT_FIG3_X_OFFSET,
                    help="X offset from the main origin to figure 3 origin.")
    ap.add_argument("--fig3-mid-z", type=float, default=DEFAULT_FIG3_MID_Z,
                    help="Relative Z for the intermediate fan endpoints (user text used y_rel=40 in XZ plane).")
    ap.add_argument("--fig3-top-z", type=float, default=DEFAULT_FIG3_TOP_Z,
                    help="Relative Z for the top common vertex / extended vertical line.")

    # Feeds
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED, help="Travel feedrate (mm/min).")
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED, help="Print/path feedrate (mm/min).")
    ap.add_argument("--b-feed", "--orient-feed", dest="orient_feed", type=float, default=DEFAULT_ORIENT_FEED,
                    help="B reorientation feedrate (fixed-tip tracked compensation move).")
    ap.add_argument("--c-feed", "--c-only-feed", dest="c_only_feed", type=float, default=DEFAULT_C_ONLY_FEED,
                    help="C-axis feedrate for C-only pre-positioning and tracked C reorientation moves.")
    ap.add_argument("--no-c-preposition", action="store_true",
                    help="Disable optional C-only pre-positioning moves before C=180 tangential sections.")

    # C / B limits and solver
    ap.add_argument("--c180-deg", type=float, default=None,
                    help="Override calibration rotation_axis_180_deg for C=180 sections.")
    ap.add_argument("--min-b", type=float, default=None, help="Lower commanded B bound (default: calibration bound).")
    ap.add_argument("--max-b", type=float, default=None, help="Upper commanded B bound (default: calibration bound).")
    ap.add_argument("--b-solver-samples", type=int, default=DEFAULT_B_SOLVER_SAMPLES,
                    help="Dense sample count for B-angle solver.")
    ap.add_argument("--b-solver-warn-deg", type=float, default=DEFAULT_B_SOLVER_WARN_DEG,
                    help="Warn if best tangency match error exceeds this value (deg).")

    # Startup/end poses
    ap.add_argument("--safe-approach-z", type=float, default=DEFAULT_SAFE_APPROACH_Z,
                    help="Safe Z used before XY start/end positioning.")
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

    # Virtual bbox clamp
    ap.add_argument("--bbox-x-min", type=float, default=DEFAULT_BBOX_X_MIN)
    ap.add_argument("--bbox-x-max", type=float, default=DEFAULT_BBOX_X_MAX)
    ap.add_argument("--bbox-y-min", type=float, default=DEFAULT_BBOX_Y_MIN)
    ap.add_argument("--bbox-y-max", type=float, default=DEFAULT_BBOX_Y_MAX)
    ap.add_argument("--bbox-z-min", type=float, default=DEFAULT_BBOX_Z_MIN)
    ap.add_argument("--bbox-z-max", type=float, default=DEFAULT_BBOX_Z_MAX)

    return ap


def main() -> None:
    ap = build_argparser()
    args = ap.parse_args()
    out_path, warns = generate_program(args)

    print(f"Wrote {out_path}")
    print("Done. Generated requested geometry sequence:")
    print("  1) Figure 1 chevrons (direct XYZ, B=0, C=0)")
    print("  2) Figure 1 chevrons on Y+offset (tangential via B + calibration, C=180)")
    print("  3) Figure 3 full fan (direct XYZ, B=0, C=0)")
    print("  4) Figure 3 bottom/top halves on Y+offset planes (tangential, C=0 / C=180)")
    if warns:
        print("\nWarnings / diagnostics:")
        for w in warns:
            print(f" - {w}")


if __name__ == "__main__":
    main()
