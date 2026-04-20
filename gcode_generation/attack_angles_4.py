#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code for the XZ-plane line/rectangle-angle test pattern,
using calibration-based exact tip-position planning:

  p_stage = p_tip_desired - offset_tip(B_motor, C_deg)

NEW (per your request)
----------------------
1) Axial extrusion: B now changes per printed line so the tip "attack" aligns to the line direction.
   We do this using your B-pull angle relations:

   - If C = 180°:
       physical_angle_deg (trig circle) = 90 - B_pull_deg
       => B_pull_deg = 90 - physical_angle_deg

   - If C = 0°:
       physical_angle_deg (trig circle) = PI - (90 - B_pull_deg)
                                      = 180 - (90 - B_pull_deg)
                                      = 90 + B_pull_deg
       => B_pull_deg = physical_angle_deg - 90

   We then solve the calibration polynomial tip_angle_poly(B_motor) = B_pull_deg for B_motor.

2) Physical setup change + sequencing:
   - Figure 1 starts with C=180°.
   - Figure 2 split in two parts:
       (a) Bottom half at y = y_base + 5: C=180°, start at origin, print 135°,130°,...,95° lines (stop before 90°).
       (b) Top half at y = y_base + 10: C=0°, start at top point:
             * print the vertical line top->origin first (this yields B_pull=180 because physical angle is 270°)
             * then print top->endpoint(95), top->endpoint(100), ..., top->endpoint(135)
               (these yield B_pull 175,170,...,135 exactly as you described)
             * between each line, return to top point using: -X move, Z up, then X/Y to point, then Z down.

3) Same 3 extrusion settings (multipliers) preserved.

4) Replace the “extra little line” at the end of each figure with a small zigzag of the same
   overall length (10 mm in X by default).

5) Summaries:
   - G-code includes a segment summary (tip endpoints, stage endpoints, B_pull, solved B_motor, C).
   - Console prints a compact summary and the unique B_pull->B_motor solves used.

Notes / assumptions
-------------------
- "Physical angle" here is computed from the printed motion direction in the XZ plane
  using the standard trig circle convention:
     0° = +X, 90° = +Z, 180° = -X, 270° = -Z.
- This matches your Figure 2 top-half description:
     vertical top->origin is ~270° => B_pull = 270-90 = 180 (with C=0),
     then 265° => 175, ... down to 225° => 135 (“-135° line”).
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------- Defaults ----------------
DEFAULT_OUT = "gcode_generation/xz_line_pattern_test_calibrated_axial.gcode"

# Pattern placement in TIP SPACE (world coordinates)
DEFAULT_START_X = 58.0
DEFAULT_START_Y = 0.0
DEFAULT_START_Z = -160.0

# Startup/end MACHINE STAGE poses (raw stage axes)
DEFAULT_MACHINE_START_X = 58.0
DEFAULT_MACHINE_START_Y = 0.0
DEFAULT_MACHINE_START_Z = 0.0
DEFAULT_MACHINE_START_B = 0.0
DEFAULT_MACHINE_START_C = 0.0

DEFAULT_MACHINE_END_X = 58.0
DEFAULT_MACHINE_END_Y = 40.0
DEFAULT_MACHINE_END_Z = 0.0
DEFAULT_MACHINE_END_B = 0.0
DEFAULT_MACHINE_END_C = 0.0
DEFAULT_SAFE_APPROACH_Z = 0.0

# Motion
DEFAULT_TRAVEL_FEED = 1200.0       # mm/min
DEFAULT_PRINT_FEED = 300.0         # mm/min
DEFAULT_C_FEED = 5000.0            # deg/min (requested separate feed)

# Geometry specifics (same as your current script)
DEFAULT_LINE_LENGTH = 60.0
DEFAULT_MID_X = 30.0
DEFAULT_ANGLE_STEP = 5.0
DEFAULT_ANGLE_MAX_1 = 45.0
DEFAULT_RETURN_LIFT_Z = 50.0
DEFAULT_VERTICAL_LINE_LEN = 20.0

DEFAULT_PLANE_Y_STEP = 20.0
DEFAULT_STEP2_ORIGIN_X_REL = 40.0
DEFAULT_STEP2_Z_TARGET_REL = 40.0
DEFAULT_STEP2_STRAIGHT_EXTEND_Z_REL = 80.0

# Zigzag (replaces the old little extra line)
DEFAULT_ZIGZAG_LEN = 10.0
DEFAULT_ZIGZAG_AMP_Z = 2.0
DEFAULT_ZIGZAG_POINTS = 11  # must be odd >= 3 (start/end on baseline)

# Print interpolation
DEFAULT_EDGE_SAMPLES = 24

# Extrusion (coordinated with print feed)
DEFAULT_EXTRUSION_PER_MM = 0.01
DEFAULT_PRIME_MM = 1.0

# Pressure offset / dwell sequencing (U-axis)
DEFAULT_PRESSURE_OFFSET_MM = 5.0
DEFAULT_PRESSURE_ADVANCE_FEED = 120.0
DEFAULT_PRESSURE_RETRACT_FEED = 240.0
DEFAULT_PREFLOW_DWELL_MS = 150
DEFAULT_NODE_DWELL_MS = 250

DEFAULT_EXTRUSION_MULTIPLIERS = (1.0, 1.5, 2.0)

# Virtual stage-space bounding box
DEFAULT_BBOX_X_MIN = 0.0
DEFAULT_BBOX_X_MAX = 200.0
DEFAULT_BBOX_Y_MIN = 0.0
DEFAULT_BBOX_Y_MAX = 200.0
DEFAULT_BBOX_Z_MIN = -168.0
DEFAULT_BBOX_Z_MAX = 0.0


# ---------------- Data classes ----------------

@dataclass
class Calibration:
    pr: np.ndarray
    pz: np.ndarray
    py_off: Optional[np.ndarray]
    pa: Optional[np.ndarray]  # tip_angle_coeffs

    b_min: float
    b_max: float

    x_axis: str
    y_axis: str
    z_axis: str
    b_axis: str
    c_axis: str
    u_axis: str

    c_180_deg: float


@dataclass
class Segment:
    """Tip-space segment (we will decide print direction and B/C at emission time)."""
    p0_tip: np.ndarray
    p1_tip: np.ndarray
    label: str


@dataclass
class EmittedSegment:
    """Segment with chosen print direction + chosen C + solved B motor."""
    p0_tip: np.ndarray
    p1_tip: np.ndarray
    label: str
    c_deg: float
    physical_angle_deg: float
    b_pull_deg: float
    b_motor: float


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
    u_axis = str(duet_map.get("extruder_axis") or "U")
    c_180 = float(motor_setup.get("rotation_axis_180_deg", 180.0))

    return Calibration(
        pr=pr, pz=pz, py_off=py_off, pa=pa,
        b_min=b_min, b_max=b_max,
        x_axis=x_axis, y_axis=y_axis, z_axis=z_axis,
        b_axis=b_axis, c_axis=c_axis, u_axis=u_axis,
        c_180_deg=c_180
    )


def eval_r(cal: Calibration, b: Any) -> np.ndarray:
    return poly_eval(cal.pr, b)


def eval_z(cal: Calibration, b: Any) -> np.ndarray:
    return poly_eval(cal.pz, b)


def eval_offplane_y(cal: Calibration, b: Any) -> np.ndarray:
    return poly_eval(cal.py_off, b, default_if_none=0.0)


def eval_tip_angle_deg(cal: Calibration, b: Any) -> np.ndarray:
    if cal.pa is None:
        raise ValueError("Calibration is missing tip_angle_coeffs.")
    return poly_eval(cal.pa, b)


def predict_tip_offset_from_bc(cal: Calibration, b_motor: float, c_deg: float) -> np.ndarray:
    """
    Physical tip offset from stage origin:
      local transverse [r(B), y_off(B)] rotated by C into XY, z=z(B)
    """
    r = float(eval_r(cal, b_motor))
    z = float(eval_z(cal, b_motor))
    y_off = float(eval_offplane_y(cal, b_motor))

    c = math.radians(float(c_deg))
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def stage_xyz_for_tip(cal: Calibration, tip_xyz: np.ndarray, b_motor: float, c_deg: float) -> np.ndarray:
    return np.asarray(tip_xyz, dtype=float) - predict_tip_offset_from_bc(cal, b_motor, c_deg)


# ---------------- B solving (invert tip_angle_poly) ----------------

def solve_b_for_tip_angle(cal: Calibration, target_angle_deg: float, search_samples: int = 5001) -> float:
    """
    Solve for B_motor such that tip_angle_poly(B_motor) ~= target_angle_deg, within [b_min, b_max].

    Strategy:
      1) Dense sampling to find sign changes on f(b)=angle(b)-target
      2) Bisection on the best interval if available
      3) Fallback to sampled minimizer of |f|
    """
    if cal.pa is None:
        raise ValueError("Calibration JSON has no tip_angle_coeffs; cannot solve B for target angle.")

    b_lo, b_hi = float(cal.b_min), float(cal.b_max)
    bb = np.linspace(b_lo, b_hi, int(max(101, search_samples)))
    aa = eval_tip_angle_deg(cal, bb) - float(target_angle_deg)

    i_best = int(np.argmin(np.abs(aa)))
    b_best_abs = float(bb[i_best])

    sign_changes: List[Tuple[float, float, float]] = []
    for i in range(len(bb) - 1):
        a0 = float(aa[i])
        a1 = float(aa[i + 1])
        if a0 == 0.0:
            return float(bb[i])
        if a0 * a1 < 0.0:
            score = min(abs(a0), abs(a1))
            sign_changes.append((score, float(bb[i]), float(bb[i + 1])))

    if sign_changes:
        sign_changes.sort(key=lambda t: t[0])
        _, lo, hi = sign_changes[0]

        def f(x: float) -> float:
            return float(eval_tip_angle_deg(cal, x) - float(target_angle_deg))

        flo = f(lo)
        fhi = f(hi)

        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fmid = f(mid)
            if abs(fmid) < 1e-10:
                return float(mid)
            if flo * fmid <= 0.0:
                hi = mid
                fhi = fmid
            else:
                lo = mid
                flo = fmid
        return float(0.5 * (lo + hi))

    return b_best_abs


class BAngleSolverCache:
    def __init__(self, cal: Calibration, search_samples: int = 5001):
        self.cal = cal
        self.search_samples = int(search_samples)
        self.cache: Dict[float, float] = {}

    def motor_for_b_pull(self, b_pull_deg: float) -> float:
        # cache key at 1e-6 deg
        key = float(round(float(b_pull_deg), 6))
        if key not in self.cache:
            self.cache[key] = solve_b_for_tip_angle(self.cal, key, search_samples=self.search_samples)
        return float(self.cache[key])


# ---------------- Utility / extrusion math ----------------

def _fmt_axes_move(axes_vals: List[Tuple[str, float]]) -> str:
    return " ".join(f"{ax}{val:.3f}" for ax, val in axes_vals)


def _clamp_stage_xyz_to_bbox(
    x: float, y: float, z: float, bbox: dict, context: str, warn_log: List[str]
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


def tube_area_mm2_from_id_inch(id_inch: float) -> float:
    d_mm = float(id_inch) * 25.4
    r_mm = 0.5 * d_mm
    return math.pi * r_mm * r_mm


def extrusion_math_summary(
    syringe_mm_per_ml: float,
    tube_id_inch: float,
    print_feed_mm_min: float,
    extrusion_per_mm_u: float,
) -> dict:
    if syringe_mm_per_ml <= 0:
        raise ValueError("syringe_mm_per_ml must be > 0.")
    if print_feed_mm_min <= 0:
        raise ValueError("print_feed_mm_min must be > 0.")

    path_speed_mm_s = float(print_feed_mm_min) / 60.0
    u_speed_mm_s = float(extrusion_per_mm_u) * path_speed_mm_s

    q_mm3_s = (1000.0 / float(syringe_mm_per_ml)) * u_speed_mm_s
    q_ml_min = q_mm3_s * 60.0 / 1000.0
    q_ul_s = q_mm3_s

    area_tube_mm2 = tube_area_mm2_from_id_inch(float(tube_id_inch))
    tube_velocity_mm_s = q_mm3_s / area_tube_mm2 if area_tube_mm2 > 0 else float("nan")
    tube_velocity_m_s = tube_velocity_mm_s / 1000.0

    return {
        "path_speed_mm_s": path_speed_mm_s,
        "u_speed_mm_s": u_speed_mm_s,
        "q_mm3_s": q_mm3_s,
        "q_ml_min": q_ml_min,
        "q_ul_s": q_ul_s,
        "tube_id_mm": float(tube_id_inch) * 25.4,
        "tube_area_mm2": area_tube_mm2,
        "tube_velocity_mm_s": tube_velocity_mm_s,
        "tube_velocity_m_s": tube_velocity_m_s,
    }


# ---------------- Geometry builders (TIP SPACE) ----------------

def p_xyz(x: float, y: float, z: float) -> np.ndarray:
    return np.array([float(x), float(y), float(z)], dtype=float)


def make_step1_segments(start_tip: np.ndarray, y_plane: float,
                        line_len: float, mid_x: float,
                        angle_step: float, angle_max: float,
                        vertical_len: float) -> List[Segment]:
    sx, _, sz = [float(v) for v in start_tip]
    segs: List[Segment] = []

    # Horizontal baseline
    p0 = p_xyz(sx, y_plane, sz)
    p1 = p_xyz(sx + line_len, y_plane, sz)
    segs.append(Segment(p0, p1, "fig1_horizontal_0deg"))

    # Roof lines: +a to x=40, then -a to x=80
    angles = np.arange(float(angle_step), float(angle_max) + 1e-9, float(angle_step))
    for a in angles:
        t = math.tan(math.radians(float(a)))
        z_mid = sz + t * mid_x
        p_start = p_xyz(sx, y_plane, sz)
        p_mid = p_xyz(sx + mid_x, y_plane, z_mid)
        p_end = p_xyz(sx + line_len, y_plane, sz)
        segs.append(Segment(p_start, p_mid, f"fig1_roof_a{a:.0f}_seg1"))
        segs.append(Segment(p_mid, p_end,   f"fig1_roof_a{a:.0f}_seg2"))

    # Vertical 20 mm line from start
    pv0 = p_xyz(sx, y_plane, sz)
    pv1 = p_xyz(sx, y_plane, sz + vertical_len)
    segs.append(Segment(pv0, pv1, "fig1_vertical_20mm"))
    return segs


def _ray_to_z_rel(origin: np.ndarray, angle_deg: float, z_rel_target: float) -> np.ndarray:
    """
    Cast a ray in XZ from origin with direction angle_deg (0=+X, 90=+Z) until dz == z_rel_target.
    """
    theta = math.radians(float(angle_deg))
    s = math.sin(theta)
    c = math.cos(theta)
    if abs(s) < 1e-12:
        raise ValueError(f"Angle {angle_deg} deg cannot reach z_rel target (sin≈0).")
    L = float(z_rel_target) / s
    return origin + np.array([L * c, 0.0, L * s], dtype=float)


def make_step2_points(
    start_step1_tip: np.ndarray,
    y_plane: float,
    origin_x_rel: float,
    z_target_rel: float,
    straight_extend_z_rel: float,
) -> Tuple[np.ndarray, np.ndarray, Dict[int, np.ndarray]]:
    """
    Returns:
      origin_tip, vertical_top_tip, endpoints_famA_by_angle (for angles 95..135 step 5)
    """
    sx, _, sz = [float(v) for v in start_step1_tip]
    origin = p_xyz(sx + origin_x_rel, y_plane, sz)

    endpoints_famA: Dict[int, np.ndarray] = {}
    for ang in range(135, 90, -5):  # 135..95
        p_end = _ray_to_z_rel(origin, ang, z_target_rel)
        endpoints_famA[int(ang)] = p_end.copy()

    vertical_top = _ray_to_z_rel(origin, 90.0, straight_extend_z_rel)  # z_rel=80
    return origin, vertical_top, endpoints_famA


def make_zigzag_segments(
    start_tip: np.ndarray,
    y_plane: float,
    x_len: float = DEFAULT_ZIGZAG_LEN,
    z_amp: float = DEFAULT_ZIGZAG_AMP_Z,
    n_points: int = DEFAULT_ZIGZAG_POINTS,
    label_prefix: str = "zigzag",
) -> List[Segment]:
    """
    Zigzag in XZ plane starting at start_tip, total x extent = x_len, returns to baseline Z at end.
    n_points must be odd >= 3: e.g. 5 gives: 0, +amp, 0, +amp, 0
    """
    if n_points < 3 or (n_points % 2) == 0:
        raise ValueError("n_points for zigzag must be odd and >= 3.")
    x0, _, z0 = [float(v) for v in start_tip]
    pts: List[np.ndarray] = []
    for i in range(n_points):
        t = i / (n_points - 1)
        x = x0 + t * float(x_len)
        z = z0 + (float(z_amp) if (i % 2 == 1) else 0.0)
        pts.append(p_xyz(x, y_plane, z))

    segs: List[Segment] = []
    for i in range(len(pts) - 1):
        segs.append(Segment(pts[i], pts[i + 1], f"{label_prefix}_{i+1}"))
    return segs


# ---------------- Angle helpers (your B-pull mapping) ----------------

def physical_angle_deg_from_tip_segment(p0_tip: np.ndarray, p1_tip: np.ndarray) -> float:
    """
    Standard trig circle in XZ plane:
      0° = +X, 90° = +Z, 180° = -X, 270° = -Z
    """
    dx = float(p1_tip[0] - p0_tip[0])
    dz = float(p1_tip[2] - p0_tip[2])
    ang = math.degrees(math.atan2(dz, dx))  # (-180, 180]
    if ang < 0.0:
        ang += 360.0
    return float(ang)


def b_pull_from_physical_angle(c_deg: float, physical_angle_deg: float) -> float:
    """
    Your mapping:

    If C=180: physical = 90 - B_pull  => B_pull = 90 - physical
    If C=0:   physical = 90 + B_pull  => B_pull = physical - 90
    """
    c = float(c_deg)
    phi = float(physical_angle_deg)
    if abs(c - 180.0) < 1e-6:
        return 90.0 - phi
    if abs(c - 0.0) < 1e-6:
        return phi - 90.0
    # If you ever use other C values, you can extend this, but for now we restrict:
    raise ValueError(f"Unsupported C={c_deg} for B-pull mapping (expected 0 or 180).")


# ---------------- G-code writer (variable B/C, exact tip tracking) ----------------

class AxialPatternWriter:
    def __init__(
        self,
        fh,
        cal: Calibration,
        b_solver: BAngleSolverCache,
        bbox: dict,
        travel_feed: float,
        print_feed: float,
        c_feed: float,
        extrusion_per_mm: float,
        pressure_offset_mm: float,
        pressure_advance_feed: float,
        pressure_retract_feed: float,
        preflow_dwell_ms: int,
        node_dwell_ms: int,
        edge_samples: int,
        emit_extrusion: bool,
        warn_log: List[str],
        segment_log: List[dict],
    ):
        self.f = fh
        self.cal = cal
        self.b_solver = b_solver

        self.bbox = bbox
        self.travel_feed = float(travel_feed)
        self.print_feed = float(print_feed)
        self.c_feed = float(c_feed)

        self.extrusion_per_mm = float(extrusion_per_mm)
        self.pressure_offset_mm = float(pressure_offset_mm)
        self.pressure_advance_feed = float(pressure_advance_feed)
        self.pressure_retract_feed = float(pressure_retract_feed)
        self.preflow_dwell_ms = int(preflow_dwell_ms)
        self.node_dwell_ms = int(node_dwell_ms)
        self.edge_samples = max(2, int(edge_samples))
        self.emit_extrusion = bool(emit_extrusion)
        self.warn_log = warn_log
        self.segment_log = segment_log

        self.u_material_abs = 0.0
        self.pressure_charged = False

        # Track current stage state
        self.cur_stage_xyz: Optional[np.ndarray] = None
        self.cur_b = 0.0
        self.cur_c = 0.0

    def _clamp_stage(self, p_stage: np.ndarray, context: str) -> np.ndarray:
        x, y, z = _clamp_stage_xyz_to_bbox(
            p_stage[0], p_stage[1], p_stage[2], self.bbox, context, self.warn_log
        )
        return np.array([x, y, z], dtype=float)

    def u_cmd_actual(self) -> float:
        return self.u_material_abs + (self.pressure_offset_mm if self.pressure_charged else 0.0)

    def tip_to_stage(self, p_tip: np.ndarray, b_motor: float, c_deg: float) -> np.ndarray:
        return stage_xyz_for_tip(self.cal, p_tip, b_motor, c_deg)

    def move_stage_xyzbc(self, p_stage: np.ndarray, b: float, c: float,
                         feed: Optional[float] = None, comment: Optional[str] = None):
        if comment:
            self.f.write(f"; {comment}\n")
        pc = self._clamp_stage(np.asarray(p_stage, dtype=float), comment or "move_stage_xyzbc")
        fval = self.travel_feed if feed is None else float(feed)

        axes = [
            (self.cal.x_axis, pc[0]),
            (self.cal.y_axis, pc[1]),
            (self.cal.z_axis, pc[2]),
            (self.cal.b_axis, float(b)),
            (self.cal.c_axis, float(c)),
        ]
        self.f.write(f"G1 {_fmt_axes_move(axes)} F{fval:.0f}\n")
        self.cur_stage_xyz = pc.copy()
        self.cur_b = float(b)
        self.cur_c = float(c)

    def move_c_only(self, c_target: float, comment: Optional[str] = None):
        if comment:
            self.f.write(f"; {comment}\n")
        self.f.write(f"G1 {self.cal.c_axis}{float(c_target):.3f} F{self.c_feed:.0f}\n")
        self.cur_c = float(c_target)

    def lift_stage_z_only(self, dz: float, comment: str):
        if self.cur_stage_xyz is None:
            raise RuntimeError("Current stage XYZ unknown before lift.")
        p = self.cur_stage_xyz.copy()
        p[2] += float(dz)
        self.move_stage_xyzbc(p, b=self.cur_b, c=self.cur_c, feed=self.travel_feed, comment=comment)

    def set_bc_in_place(self, b: float, c: float, comment: str):
        """
        Change B/C without changing XYZ (uses travel feed; C-only feed not used here to avoid mixed-axis surprises).
        """
        if self.cur_stage_xyz is None:
            raise RuntimeError("Current stage XYZ unknown before set_bc_in_place.")
        self.move_stage_xyzbc(self.cur_stage_xyz.copy(), b=b, c=c, feed=self.travel_feed, comment=comment)

    def travel_to_tip_safe_z_up_xy_down(
        self,
        target_tip: np.ndarray,
        b_target: float,
        c_target: float,
        lift_dz_stage: float,
        comment_prefix: str,
    ):
        """
        General safe travel:
          1) lift Z only (no B/C change)
          2) set B/C at safe Z
          3) move X/Y at safe Z
          4) move Z down to target
        """
        if self.cur_stage_xyz is None:
            raise RuntimeError("Current stage XYZ unknown before travel.")

        self.lift_stage_z_only(lift_dz_stage, comment=f"{comment_prefix}: lift Z")
        self.set_bc_in_place(b_target, c_target, comment=f"{comment_prefix}: set B/C at safe Z")

        target_stage = self.tip_to_stage(target_tip, b_target, c_target)
        target_stage = self._clamp_stage(target_stage, f"{comment_prefix}: target clamp")

        p_xy = self.cur_stage_xyz.copy()
        p_xy[0] = target_stage[0]
        p_xy[1] = target_stage[1]
        self.move_stage_xyzbc(p_xy, b=b_target, c=c_target, feed=self.travel_feed, comment=f"{comment_prefix}: move X/Y")

        p_z = p_xy.copy()
        p_z[2] = target_stage[2]
        self.move_stage_xyzbc(p_z, b=b_target, c=c_target, feed=self.travel_feed, comment=f"{comment_prefix}: move Z down")

    def return_to_tip_negx_then_up_then_to_tip(
        self,
        target_tip: np.ndarray,
        b_target: float,
        c_target: float,
        negx_mm: float,
        lift_dz_stage: float,
        comment_prefix: str,
    ):
        """
        Your requested return style for Figure 2 top half:
          -x move, z up, then x to point (and y), then z down
        """
        if self.cur_stage_xyz is None:
            raise RuntimeError("Current stage XYZ unknown before return.")

        # 1) -X move at current Z (no B/C change)
        p1 = self.cur_stage_xyz.copy()
        p1[0] -= float(negx_mm)
        self.move_stage_xyzbc(p1, b=self.cur_b, c=self.cur_c, feed=self.travel_feed, comment=f"{comment_prefix}: -X clearance")

        # 2) Z up (no B/C change)
        self.lift_stage_z_only(lift_dz_stage, comment=f"{comment_prefix}: Z up")

        # 3) set B/C at safe Z
        self.set_bc_in_place(b_target, c_target, comment=f"{comment_prefix}: set B/C at safe Z")

        # 4) move X/Y to target at safe Z
        target_stage = self.tip_to_stage(target_tip, b_target, c_target)
        target_stage = self._clamp_stage(target_stage, f"{comment_prefix}: target clamp")

        p_xy = self.cur_stage_xyz.copy()
        p_xy[0] = target_stage[0]
        p_xy[1] = target_stage[1]
        self.move_stage_xyzbc(p_xy, b=b_target, c=c_target, feed=self.travel_feed, comment=f"{comment_prefix}: X/Y to target")

        # 5) Z down
        p_z = p_xy.copy()
        p_z[2] = target_stage[2]
        self.move_stage_xyzbc(p_z, b=b_target, c=c_target, feed=self.travel_feed, comment=f"{comment_prefix}: Z down")

    def pressure_preload_before_print(self):
        if self.emit_extrusion and self.pressure_offset_mm > 0.0 and not self.pressure_charged:
            self.pressure_charged = True
            self.f.write("; pressure preload before print pass\n")
            self.f.write(f"G1 {self.cal.u_axis}{self.u_cmd_actual():.3f} F{self.pressure_advance_feed:.0f}\n")
            if self.preflow_dwell_ms > 0:
                self.f.write(f"G4 P{self.preflow_dwell_ms}\n")

    def pressure_release_after_print(self):
        if self.emit_extrusion and self.pressure_offset_mm > 0.0 and self.pressure_charged:
            if self.node_dwell_ms > 0:
                self.f.write("; end-of-pass dwell for node formation / liquid flow\n")
                self.f.write(f"G4 P{self.node_dwell_ms}\n")
            self.pressure_charged = False
            self.f.write("; pressure release after print pass\n")
            self.f.write(f"G1 {self.cal.u_axis}{self.u_cmd_actual():.3f} F{self.pressure_retract_feed:.0f}\n")

    def print_emitted_segment(self, seg: EmittedSegment, extrusion_multiplier: float):
        """
        Print a line segment in tip-space with its chosen B_motor and C, with exact stage solve.
        """
        p0_tip = np.asarray(seg.p0_tip, dtype=float)
        p1_tip = np.asarray(seg.p1_tip, dtype=float)
        b = float(seg.b_motor)
        c = float(seg.c_deg)

        # Ensure we're at start (travel, safe)
        if self.cur_stage_xyz is None:
            raise RuntimeError("Stage state unknown before printing; call a startup move first.")

        p0_stage_des = self.tip_to_stage(p0_tip, b, c)
        p0_stage = self._clamp_stage(p0_stage_des, f"{seg.label}: p0 clamp")
        if np.linalg.norm(self.cur_stage_xyz - p0_stage) > 1e-6 or abs(self.cur_b - b) > 1e-9 or abs(self.cur_c - c) > 1e-9:
            self.travel_to_tip_safe_z_up_xy_down(
                target_tip=p0_tip, b_target=b, c_target=c,
                lift_dz_stage=DEFAULT_RETURN_LIFT_Z,
                comment_prefix=f"{seg.label}: travel to start"
            )

        # Log endpoints (desired + clamped)
        p1_stage_des = self.tip_to_stage(p1_tip, b, c)
        p1_stage = self._clamp_stage(p1_stage_des, f"{seg.label}: p1 clamp")
        self.segment_log.append({
            "label": seg.label,
            "c_deg": c,
            "physical_angle_deg": seg.physical_angle_deg,
            "b_pull_deg": seg.b_pull_deg,
            "b_motor": b,
            "p0_tip": p0_tip.copy(),
            "p1_tip": p1_tip.copy(),
            "p0_stage_des": p0_stage_des.copy(),
            "p1_stage_des": p1_stage_des.copy(),
            "p0_stage_clamped": p0_stage.copy(),
            "p1_stage_clamped": p1_stage.copy(),
        })

        self.pressure_preload_before_print()
        self.f.write(f"; print {seg.label} | C={c:.1f} | physical={seg.physical_angle_deg:.1f} | B_pull={seg.b_pull_deg:.1f} | B_motor={b:.6f}\n")

        # Precompute offset once for this segment
        offset = predict_tip_offset_from_bc(self.cal, b, c)

        ts = np.linspace(0.0, 1.0, self.edge_samples + 1)
        last_stage = self.cur_stage_xyz.copy()

        for j in range(1, len(ts)):
            t = float(ts[j])
            p_tip = p0_tip + t * (p1_tip - p0_tip)
            p_stage_des = np.asarray(p_tip, dtype=float) - offset

            # Clamp if needed (note: clamping breaks exact tip tracking, so we don't hard-fail)
            pc = self._clamp_stage(p_stage_des, f"{seg.label} seg {j}")

            axes = [
                (self.cal.x_axis, pc[0]),
                (self.cal.y_axis, pc[1]),
                (self.cal.z_axis, pc[2]),
                (self.cal.b_axis, b),
                (self.cal.c_axis, c),
            ]

            if self.emit_extrusion:
                seg_len = float(np.linalg.norm(pc - last_stage))
                self.u_material_abs += (self.extrusion_per_mm * float(extrusion_multiplier)) * seg_len
                axes.append((self.cal.u_axis, self.u_cmd_actual()))

            self.f.write(f"G1 {_fmt_axes_move(axes)} F{self.print_feed:.0f}\n")
            last_stage = pc.copy()

        self.cur_stage_xyz = last_stage
        self.cur_b = b
        self.cur_c = c

        self.pressure_release_after_print()


# ---------------- Segment emission helpers ----------------

def emit_segment_with_fixed_c(
    cal: Calibration,
    b_solver: BAngleSolverCache,
    seg: Segment,
    c_deg: float,
    reverse: bool,
) -> EmittedSegment:
    """
    Choose print direction (maybe reversed), compute physical angle from motion direction,
    compute B_pull using your equation for this C, solve B motor.
    """
    if not reverse:
        p0, p1 = seg.p0_tip.copy(), seg.p1_tip.copy()
    else:
        p0, p1 = seg.p1_tip.copy(), seg.p0_tip.copy()

    phi = physical_angle_deg_from_tip_segment(p0, p1)
    b_pull = b_pull_from_physical_angle(c_deg=c_deg, physical_angle_deg=phi)
    b_motor = b_solver.motor_for_b_pull(b_pull)

    return EmittedSegment(
        p0_tip=p0, p1_tip=p1, label=seg.label,
        c_deg=float(c_deg),
        physical_angle_deg=float(phi),
        b_pull_deg=float(b_pull),
        b_motor=float(b_motor),
    )


# ---------------- Pattern generation ----------------

def write_pattern_gcode(
    out_path: str,
    cal: Calibration,
    pattern_start_tip: Tuple[float, float, float],
    machine_start_pose: Tuple[float, float, float, float, float],
    machine_end_pose: Tuple[float, float, float, float, float],
    safe_approach_z: float,
    travel_feed: float,
    print_feed: float,
    c_feed: float,
    edge_samples: int,
    extrusion_per_mm: float,
    prime_mm: float,
    pressure_offset_mm: float,
    pressure_advance_feed: float,
    pressure_retract_feed: float,
    preflow_dwell_ms: int,
    node_dwell_ms: int,
    bbox: dict,
    zigzag_len: float,
    zigzag_amp_z: float,
    zigzag_points: int,
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    emit_extrusion = float(extrusion_per_mm) != 0.0
    bbox_warnings: List[str] = []
    segment_log: List[dict] = []

    b_solver = BAngleSolverCache(cal, search_samples=5001)

    sx, sy, sz = [float(v) for v in pattern_start_tip]

    # For header diagnostics, solve a few canonical angles if possible
    demo_angles = [0.0, 45.0, 85.0, 90.0, 135.0, 180.0]
    demo_solves = []
    for a in demo_angles:
        try:
            bm = b_solver.motor_for_b_pull(a)
            achieved = float(eval_tip_angle_deg(cal, bm))
            demo_solves.append((a, bm, achieved, achieved - a))
        except Exception as e:
            demo_solves.append((a, float("nan"), float("nan"), float("nan")))

    with open(out_path, "w") as f:
        f.write("; generated by xz_line_pattern_test_calibrated_axial.py\n")
        f.write("; calibration-based exact tip-position planning: stage = tip - offset(B_motor, C)\n")
        f.write("; axial extrusion: B motor solved per-segment from B_pull angle derived from motion direction\n")
        f.write("; B-pull mapping used (per your note):\n")
        f.write(";   C=180: physical = 90 - B_pull  => B_pull = 90 - physical\n")
        f.write(";   C=0:   physical = 90 + B_pull  => B_pull = physical - 90\n")
        f.write(f"; axes: X->{cal.x_axis}, Y->{cal.y_axis}, Z->{cal.z_axis}, B->{cal.b_axis}, C->{cal.c_axis}, U->{cal.u_axis}\n")
        f.write(f"; calibration B range: [{cal.b_min:.6f}, {cal.b_max:.6f}]\n")
        f.write(f"; pattern start (tip-space) = [{sx:.3f}, {sy:.3f}, {sz:.3f}]\n")
        f.write(f"; feeds: travel={travel_feed:.1f} mm/min, print={print_feed:.1f} mm/min, C-only={c_feed:.1f} deg/min\n")
        f.write(f"; extrusion_per_mm(base) = {extrusion_per_mm:.6f} U/mm-path\n")
        f.write(f"; extrusion multipliers = {DEFAULT_EXTRUSION_MULTIPLIERS}\n")
        f.write(f"; zigzag: len={zigzag_len:.2f}mm, ampZ={zigzag_amp_z:.2f}mm, points={zigzag_points}\n")
        f.write("; demo B_pull->B_motor solves (target, motor, achieved, error):\n")
        for (t, bm, ach, err) in demo_solves:
            f.write(f";   {t:7.2f} deg -> {bm: .6f} (ach {ach: .3f}, err {err: .3f})\n")
        f.write("G90\n")

        if emit_extrusion:
            f.write("M82\n")
            f.write(f"G92 {cal.u_axis}0\n")
            if abs(float(prime_mm)) > 0.0:
                f.write(f"G1 {cal.u_axis}{float(prime_mm):.3f} F{max(60.0, float(pressure_advance_feed)):.0f} ; prime material\n")

        g = AxialPatternWriter(
            fh=f,
            cal=cal,
            b_solver=b_solver,
            bbox=bbox,
            travel_feed=travel_feed,
            print_feed=print_feed,
            c_feed=c_feed,
            extrusion_per_mm=extrusion_per_mm,
            pressure_offset_mm=pressure_offset_mm,
            pressure_advance_feed=pressure_advance_feed,
            pressure_retract_feed=pressure_retract_feed,
            preflow_dwell_ms=preflow_dwell_ms,
            node_dwell_ms=node_dwell_ms,
            edge_samples=edge_samples,
            emit_extrusion=emit_extrusion,
            warn_log=bbox_warnings,
            segment_log=segment_log,
        )
        if emit_extrusion:
            g.u_material_abs = float(prime_mm)

        # ---- Safe startup (machine-stage coordinates) ----
        msx, msy, msz, msb, msc = [float(v) for v in machine_start_pose]
        f.write("; safe startup approach (machine stage coordinates)\n")
        g.move_stage_xyzbc(
            np.array([msx, msy, safe_approach_z], dtype=float),
            b=msb, c=msc, feed=travel_feed,
            comment="startup: move to safe Z at machine-start XY"
        )
        g.move_stage_xyzbc(
            np.array([msx, msy, msz], dtype=float),
            b=msb, c=msc, feed=travel_feed,
            comment="startup: dive to machine-start Z"
        )

        # ---- Pattern execution ----
        y_cursor = sy
        total_print_passes = 0

        # Figure 1 starts with C=180° (your note)
        # We'll do a C-only rotate here so the first figure begins with C=180.
        g.move_c_only(180.0, comment="preposition C for Figure 1 start (C=180)")

        for rep_idx, ex_mult in enumerate(DEFAULT_EXTRUSION_MULTIPLIERS, start=1):
            f.write(f"; ==================== repetition {rep_idx}/{len(DEFAULT_EXTRUSION_MULTIPLIERS)} ====================\n")
            f.write(f"; extrusion multiplier = {float(ex_mult):.3f}\n")

            # -------- Figure 1 (y = y_cursor) --------
            y_fig1 = y_cursor
            fig1_start_tip = p_xyz(sx, y_fig1, sz)

            # Travel to fig1 start with C=180 and B_pull corresponding to horizontal (physical=0 => B_pull=90)
            # For C=180: B_pull=90-physical. physical=0 => 90.
            b_motor_start = b_solver.motor_for_b_pull(90.0)
            g.travel_to_tip_safe_z_up_xy_down(
                target_tip=fig1_start_tip,
                b_target=b_motor_start,
                c_target=180.0,
                lift_dz_stage=DEFAULT_RETURN_LIFT_Z,
                comment_prefix=f"rep{rep_idx} fig1 move to origin"
            )

            fig1_segments = make_step1_segments(
                start_tip=fig1_start_tip,
                y_plane=y_fig1,
                line_len=DEFAULT_LINE_LENGTH,
                mid_x=DEFAULT_MID_X,
                angle_step=DEFAULT_ANGLE_STEP,
                angle_max=DEFAULT_ANGLE_MAX_1,
                vertical_len=DEFAULT_VERTICAL_LINE_LEN,
            )

            # Print baseline (C=180, forward)
            seg0 = emit_segment_with_fixed_c(cal, b_solver, fig1_segments[0], c_deg=180.0, reverse=False)
            g.print_emitted_segment(seg0, extrusion_multiplier=ex_mult); total_print_passes += 1
            g.travel_to_tip_safe_z_up_xy_down(fig1_start_tip, b_motor_start, 180.0, DEFAULT_RETURN_LIFT_Z,
                                              comment_prefix=f"rep{rep_idx} fig1 return after baseline")

            # Roof pairs: seg1 is +a (C=180 forward), seg2 is -a but we print reversed with C=0 (so it becomes 180-a direction)
            roof_parts = fig1_segments[1:-1]
            if len(roof_parts) % 2 != 0:
                raise RuntimeError("Unexpected roof segment count (must be paired).")

            for k in range(0, len(roof_parts), 2):
                s1 = roof_parts[k]
                s2 = roof_parts[k + 1]

                e1 = emit_segment_with_fixed_c(cal, b_solver, s1, c_deg=180.0, reverse=False)
                g.print_emitted_segment(e1, extrusion_multiplier=ex_mult); total_print_passes += 1

                # For s2: print reversed with C=0 (this gives the desired 180-a motion direction,
                # and B_pull becomes (physical-90) which matches your mapping)
                e2 = emit_segment_with_fixed_c(cal, b_solver, s2, c_deg=0.0, reverse=True)
                g.print_emitted_segment(e2, extrusion_multiplier=ex_mult); total_print_passes += 1

                g.travel_to_tip_safe_z_up_xy_down(fig1_start_tip, b_motor_start, 180.0, DEFAULT_RETURN_LIFT_Z,
                                                  comment_prefix=f"rep{rep_idx} fig1 return after roof pair {k//2 + 1}")

            # Vertical 20 mm line (C=180 forward)
            vseg = emit_segment_with_fixed_c(cal, b_solver, fig1_segments[-1], c_deg=180.0, reverse=False)
            g.print_emitted_segment(vseg, extrusion_multiplier=ex_mult); total_print_passes += 1

            # End of Figure 1: zigzag at current tip (end of vertical)
            zig_start_tip = vseg.p1_tip.copy()
            zig1 = make_zigzag_segments(
                start_tip=zig_start_tip, y_plane=y_fig1,
                x_len=zigzag_len, z_amp=zigzag_amp_z, n_points=zigzag_points,
                label_prefix=f"rep{rep_idx}_fig1_zigzag"
            )
            for zs in zig1:
                # Keep C=180 for the zigzag
                ez = emit_segment_with_fixed_c(cal, b_solver, zs, c_deg=180.0, reverse=False)
                g.print_emitted_segment(ez, extrusion_multiplier=ex_mult); total_print_passes += 1

            # -------- Figure 2 split (bottom + top) --------
            # Bottom half: y = y_cursor + 5, C=180, print origin->endpoints for angles 135..95
            y_fig2_bot = y_cursor + DEFAULT_PLANE_Y_STEP
            origin_bot, top_bot, endpoints_bot = make_step2_points(
                start_step1_tip=fig1_start_tip,
                y_plane=y_fig2_bot,
                origin_x_rel=DEFAULT_STEP2_ORIGIN_X_REL,
                z_target_rel=DEFAULT_STEP2_Z_TARGET_REL,
                straight_extend_z_rel=DEFAULT_STEP2_STRAIGHT_EXTEND_Z_REL,
            )

            # Travel to origin_bot with C=180; for the first line (135) physical=135 => B_pull=90-135=-45
            b_first_bot = b_solver.motor_for_b_pull(90.0 - 135.0)
            g.travel_to_tip_safe_z_up_xy_down(
                target_tip=origin_bot,
                b_target=b_first_bot,
                c_target=180.0,
                lift_dz_stage=DEFAULT_RETURN_LIFT_Z,
                comment_prefix=f"rep{rep_idx} fig2 bottom move to origin (C=180)"
            )

            for ang in range(135, 90, -5):  # 135..95
                p_end = endpoints_bot[int(ang)].copy()
                s = Segment(origin_bot.copy(), p_end, f"rep{rep_idx}_fig2_bot_{ang}deg")
                e = emit_segment_with_fixed_c(cal, b_solver, s, c_deg=180.0, reverse=False)
                g.print_emitted_segment(e, extrusion_multiplier=ex_mult); total_print_passes += 1

                # Return to origin between lines (safe)
                g.travel_to_tip_safe_z_up_xy_down(
                    target_tip=origin_bot,
                    b_target=b_first_bot,
                    c_target=180.0,
                    lift_dz_stage=DEFAULT_RETURN_LIFT_Z,
                    comment_prefix=f"rep{rep_idx} fig2 bottom return to origin"
                )

            # Top half: y = y_cursor + 10, C=0, start from top point, print vertical top->origin then top->endpoints(95..135)
            y_fig2_top = y_cursor + 2.0 * DEFAULT_PLANE_Y_STEP
            origin_top = origin_bot.copy(); origin_top[1] = y_fig2_top
            top_top = top_bot.copy(); top_top[1] = y_fig2_top

            # Travel to top point with C=0 and B_pull for vertical down:
            # physical angle for top->origin is 270 => B_pull = 270 - 90 = 180 (matches your note)
            b_vertical_down = b_solver.motor_for_b_pull(180.0)
            g.travel_to_tip_safe_z_up_xy_down(
                target_tip=top_top,
                b_target=b_vertical_down,
                c_target=0.0,
                lift_dz_stage=DEFAULT_RETURN_LIFT_Z,
                comment_prefix=f"rep{rep_idx} fig2 top move to top point (C=0)"
            )

            # Vertical line first: top -> origin (C=0)
            s_vert = Segment(top_top.copy(), origin_top.copy(), f"rep{rep_idx}_fig2_top_vertical_top_to_origin")
            e_vert = emit_segment_with_fixed_c(cal, b_solver, s_vert, c_deg=0.0, reverse=False)
            g.print_emitted_segment(e_vert, extrusion_multiplier=ex_mult); total_print_passes += 1

            # Return to top point with your requested return style
            g.return_to_tip_negx_then_up_then_to_tip(
                target_tip=top_top,
                b_target=b_vertical_down,
                c_target=0.0,
                negx_mm=5.0,
                lift_dz_stage=DEFAULT_RETURN_LIFT_Z,
                comment_prefix=f"rep{rep_idx} fig2 top return to top after vertical"
            )

            # Then lines at B_pull 175,170,...,135 by printing top->endpoint(95..135)
            for ang in range(95, 140, 5):  # 95,100,...,135
                p_end_bot = endpoints_bot[int(ang)].copy()
                p_end_top = p_end_bot.copy()
                p_end_top[1] = y_fig2_top
                s = Segment(top_top.copy(), p_end_top, f"rep{rep_idx}_fig2_top_to_end{ang}")
                e = emit_segment_with_fixed_c(cal, b_solver, s, c_deg=0.0, reverse=False)
                g.print_emitted_segment(e, extrusion_multiplier=ex_mult); total_print_passes += 1

                # Return to top with your -x, z up, then x/y to point
                g.return_to_tip_negx_then_up_then_to_tip(
                    target_tip=top_top,
                    b_target=b_vertical_down,
                    c_target=0.0,
                    negx_mm=5.0,
                    lift_dz_stage=DEFAULT_RETURN_LIFT_Z,
                    comment_prefix=f"rep{rep_idx} fig2 top return to top after line to end{ang}"
                )

            # End of Figure 2: zigzag at top point (replaces old little line)
            zig2 = make_zigzag_segments(
                start_tip=top_top, y_plane=y_fig2_top,
                x_len=zigzag_len, z_amp=zigzag_amp_z, n_points=zigzag_points,
                label_prefix=f"rep{rep_idx}_fig2_zigzag"
            )
            for zs in zig2:
                ez = emit_segment_with_fixed_c(cal, b_solver, zs, c_deg=0.0, reverse=False)
                g.print_emitted_segment(ez, extrusion_multiplier=ex_mult); total_print_passes += 1

            # Advance y for next repetition: fig1 at y, fig2_bot at y+5, fig2_top at y+10
            y_cursor = y_cursor + 3.0 * DEFAULT_PLANE_Y_STEP

        # Final pressure release (safety)
        if emit_extrusion and g.pressure_charged:
            f.write("; final pressure release at end of print\n")
            g.pressure_charged = False
            f.write(f"G1 {cal.u_axis}{g.u_cmd_actual():.3f} F{float(pressure_retract_feed):.0f}\n")

        # ---- Safe end move (machine-stage coordinates) ----
        mex, mey, mez, meb, mec = [float(v) for v in machine_end_pose]
        f.write("; safe end move (machine stage coordinates)\n")
        if g.cur_stage_xyz is None:
            raise RuntimeError("Missing final stage XYZ.")
        g.move_stage_xyzbc(
            np.array([g.cur_stage_xyz[0], g.cur_stage_xyz[1], safe_approach_z], dtype=float),
            b=meb, c=mec, feed=travel_feed,
            comment="end: raise to safe Z"
        )
        g.move_stage_xyzbc(
            np.array([mex, mey, safe_approach_z], dtype=float),
            b=meb, c=mec, feed=travel_feed,
            comment="end: move XY at safe Z"
        )
        g.move_stage_xyzbc(
            np.array([mex, mey, mez], dtype=float),
            b=meb, c=mec, feed=travel_feed,
            comment="end: dive to machine end Z"
        )

        # ---- Summaries in G-code ----
        if bbox_warnings:
            f.write("; virtual bbox clamp warnings:\n")
            for msg in bbox_warnings:
                f.write(f"; {msg}\n")

        # Unique B_pull solutions used
        f.write("; --- Unique B_pull (deg) -> B_motor solves used ---\n")
        used = sorted({float(round(r["b_pull_deg"], 6)) for r in segment_log})
        for bp in used:
            bm = b_solver.motor_for_b_pull(bp)
            ach = float(eval_tip_angle_deg(cal, bm))
            f.write(f";   B_pull={bp: .6f} -> B_motor={bm: .6f} | achieved={ach: .6f} | err={ach-bp: .6f}\n")

        # Per-segment summary (what you asked for)
        f.write("; --- Segment summary (tip endpoints + stage endpoints + motor values) ---\n")
        f.write("; label | tip0(x,y,z) -> tip1(x,y,z) | stage0(x,y,z) -> stage1(x,y,z) | B_pull | B_motor | C\n")
        for r in segment_log:
            p0 = r["p0_tip"]; p1 = r["p1_tip"]
            s0 = r["p0_stage_clamped"]; s1 = r["p1_stage_clamped"]
            f.write(
                "; "
                f"{r['label']} | "
                f"tip0=({p0[0]:.3f},{p0[1]:.3f},{p0[2]:.3f}) -> tip1=({p1[0]:.3f},{p1[1]:.3f},{p1[2]:.3f}) | "
                f"stage0=({s0[0]:.3f},{s0[1]:.3f},{s0[2]:.3f}) -> stage1=({s1[0]:.3f},{s1[1]:.3f},{s1[2]:.3f}) | "
                f"B_pull={r['b_pull_deg']:.1f} | B_motor={r['b_motor']:.6f} | C={r['c_deg']:.1f}\n"
            )

        f.write(f"; total print passes = {total_print_passes}\n")
        f.write(f"; bbox warning count = {len(bbox_warnings)}\n")
        f.write("; --- end ---\n")

    # Console diagnostics
    print(f"Wrote {out_path}")
    print("Mode: calibrated tip-position planning, variable B per segment from B-pull equation, C=180/0 as specified")
    print(f"Axes mapping: X={cal.x_axis}, Y={cal.y_axis}, Z={cal.z_axis}, B={cal.b_axis}, C={cal.c_axis}, U={cal.u_axis}")
    print(f"Calibration B range: [{cal.b_min:.6f}, {cal.b_max:.6f}]")
    print(f"Unique B_pull targets used: {len(set(float(round(r['b_pull_deg'],6)) for r in segment_log))}")
    if bbox_warnings:
        print(f"Virtual bounding-box warnings: {len(bbox_warnings)} (values clamped)")


# ---------------- Main ----------------

def main(args):
    cal = load_calibration(args.calibration)

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

    pattern_start_tip = (float(args.start_x), float(args.start_y), float(args.start_z))
    machine_start_pose = (
        float(args.machine_start_x), float(args.machine_start_y), float(args.machine_start_z),
        float(args.machine_start_b), float(args.machine_start_c),
    )
    machine_end_pose = (
        float(args.machine_end_x), float(args.machine_end_y), float(args.machine_end_z),
        float(args.machine_end_b), float(args.machine_end_c),
    )

    write_pattern_gcode(
        out_path=str(args.out),
        cal=cal,
        pattern_start_tip=pattern_start_tip,
        machine_start_pose=machine_start_pose,
        machine_end_pose=machine_end_pose,
        safe_approach_z=float(args.safe_approach_z),
        travel_feed=float(args.travel_feed),
        print_feed=float(args.print_feed),
        c_feed=float(args.c_feed),
        edge_samples=int(args.edge_samples),
        extrusion_per_mm=float(args.extrusion_per_mm),
        prime_mm=float(args.prime_mm),
        pressure_offset_mm=float(args.pressure_offset_mm),
        pressure_advance_feed=float(args.pressure_advance_feed),
        pressure_retract_feed=float(args.pressure_retract_feed),
        preflow_dwell_ms=int(args.preflow_dwell_ms),
        node_dwell_ms=int(args.node_dwell_ms),
        bbox=bbox,
        zigzag_len=float(args.zigzag_len),
        zigzag_amp_z=float(args.zigzag_amp_z),
        zigzag_points=int(args.zigzag_points),
    )

    ex_math = extrusion_math_summary(
        syringe_mm_per_ml=float(args.syringe_mm_per_ml),
        tube_id_inch=float(args.tube_id_inch),
        print_feed_mm_min=float(args.print_feed),
        extrusion_per_mm_u=float(args.extrusion_per_mm),
    )
    print("\nExtrusion / fluid math (base print feed + base extrusion_per_mm):")
    print(f"  Syringe calibration: {float(args.syringe_mm_per_ml):.3f} mm U / mL")
    print(f"  Tube ID: {ex_math['tube_id_mm']:.3f} mm ({float(args.tube_id_inch):.5f} in)")
    print(f"  Tube area: {ex_math['tube_area_mm2']:.6f} mm^2")
    print(f"  Path speed: {ex_math['path_speed_mm_s']:.6f} mm/s")
    print(f"  U speed: {ex_math['u_speed_mm_s']:.6f} mm/s")
    print(f"  Volumetric flow: {ex_math['q_mm3_s']:.6f} mm^3/s = {ex_math['q_ul_s']:.6f} uL/s = {ex_math['q_ml_min']:.6f} mL/min")
    print(f"  Mean fluid velocity in tube: {ex_math['tube_velocity_mm_s']:.6f} mm/s = {ex_math['tube_velocity_m_s']:.6f} m/s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Generate calibrated G-code for the XZ-plane line test pattern using exact tip-position planning "
            "from calibration, with per-line axial extrusion: B solved from B-pull angle derived from line direction "
            "and C=180/0 rules. Figure2 split into bottom(C=180) and top(C=0, y+5)."
        )
    )
    ap.add_argument("--calibration", required=True, help="Path to calibration JSON.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code file.")

    # Pattern placement in TIP SPACE
    ap.add_argument("--start-x", type=float, default=DEFAULT_START_X, help="Pattern start X (tip-space).")
    ap.add_argument("--start-y", type=float, default=DEFAULT_START_Y, help="Pattern start Y (tip-space).")
    ap.add_argument("--start-z", type=float, default=DEFAULT_START_Z, help="Pattern start Z (tip-space).")

    # Machine startup / end poses (raw stage coordinates)
    ap.add_argument("--machine-start-x", type=float, default=DEFAULT_MACHINE_START_X)
    ap.add_argument("--machine-start-y", type=float, default=DEFAULT_MACHINE_START_Y)
    ap.add_argument("--machine-start-z", type=float, default=DEFAULT_MACHINE_START_Z)
    ap.add_argument("--machine-start-b", type=float, default=DEFAULT_MACHINE_START_B)
    ap.add_argument("--machine-start-c", type=float, default=DEFAULT_MACHINE_START_C)

    ap.add_argument("--machine-end-x", type=float, default=DEFAULT_MACHINE_END_X)
    ap.add_argument("--machine-end-y", type=float, default=DEFAULT_MACHINE_END_Y)
    ap.add_argument("--machine-end-z", type=float, default=DEFAULT_MACHINE_END_Z)
    ap.add_argument("--machine-end-b", type=float, default=DEFAULT_MACHINE_END_B)
    ap.add_argument("--machine-end-c", type=float, default=DEFAULT_MACHINE_END_C)

    ap.add_argument("--safe-approach-z", type=float, default=DEFAULT_SAFE_APPROACH_Z,
                    help="Safe Z used before XY startup/end positioning.")

    # Motion
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED,
                    help="Feedrate for non-print travel moves (mm/min).")
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED,
                    help="Coordinated feedrate for print moves (mm/min).")
    ap.add_argument("--c-feed", type=float, default=DEFAULT_C_FEED,
                    help="Feedrate for C-only moves (deg/min).")
    ap.add_argument("--edge-samples", type=int, default=DEFAULT_EDGE_SAMPLES,
                    help="Interpolation segments per printed line.")

    # Extrusion + pressure math
    ap.add_argument("--extrusion-per-mm", type=float, default=DEFAULT_EXTRUSION_PER_MM,
                    help="U-axis displacement (mm) per mm of printed path. 0 disables extrusion.")
    ap.add_argument("--prime-mm", type=float, default=DEFAULT_PRIME_MM,
                    help="Optional U-axis material prime at start (absolute extrusion mode).")

    ap.add_argument("--pressure-offset-mm", type=float, default=DEFAULT_PRESSURE_OFFSET_MM,
                    help="U preload offset (mm) before each print pass, retracted after each pass.")
    ap.add_argument("--pressure-advance-feed", type=float, default=DEFAULT_PRESSURE_ADVANCE_FEED,
                    help="Feedrate for U-only pressure advance moves (mm/min).")
    ap.add_argument("--pressure-retract-feed", type=float, default=DEFAULT_PRESSURE_RETRACT_FEED,
                    help="Feedrate for U-only pressure retract moves (mm/min).")
    ap.add_argument("--preflow-dwell-ms", type=int, default=DEFAULT_PREFLOW_DWELL_MS,
                    help="Dwell after pressure advance before printing each pass (ms).")
    ap.add_argument("--node-dwell-ms", type=int, default=DEFAULT_NODE_DWELL_MS,
                    help="Dwell at end of each pass before pressure retract (ms).")

    # Zigzag settings
    ap.add_argument("--zigzag-len", type=float, default=DEFAULT_ZIGZAG_LEN,
                    help="Total zigzag length in X (mm).")
    ap.add_argument("--zigzag-amp-z", type=float, default=DEFAULT_ZIGZAG_AMP_Z,
                    help="Z amplitude of zigzag (mm).")
    ap.add_argument("--zigzag-points", type=int, default=DEFAULT_ZIGZAG_POINTS,
                    help="Odd number of points for zigzag polyline (>=3, odd).")

    # Virtual stage-space bounding box
    ap.add_argument("--bbox-x-min", type=float, default=DEFAULT_BBOX_X_MIN)
    ap.add_argument("--bbox-x-max", type=float, default=DEFAULT_BBOX_X_MAX)
    ap.add_argument("--bbox-y-min", type=float, default=DEFAULT_BBOX_Y_MIN)
    ap.add_argument("--bbox-y-max", type=float, default=DEFAULT_BBOX_Y_MAX)
    ap.add_argument("--bbox-z-min", type=float, default=DEFAULT_BBOX_Z_MIN)
    ap.add_argument("--bbox-z-max", type=float, default=DEFAULT_BBOX_Z_MAX)

    # Diagnostics-only extrusion helpers
    ap.add_argument("--syringe-mm-per-ml", type=float, default=6.0,
                    help="Syringe U-axis calibration (mm of U travel per mL displaced).")
    ap.add_argument("--tube-id-inch", type=float, default=0.02,
                    help="Tube inner diameter in inches (for flow velocity diagnostics).")

    args = ap.parse_args()
    main(args)