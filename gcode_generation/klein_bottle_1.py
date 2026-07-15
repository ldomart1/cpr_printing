#!/usr/bin/env python3
"""
klein_bottle_slicer_calibrated_variable_radius.py

Generates Duet/RRF-friendly G-code for a “Klein-bottle-like” swept tube, using your
calibration JSON for exact tip-position planning:

    stage_xyz = desired_tip_xyz - tip_offset_xyz(B_cmd, C_deg)

Key behavior (matches your description)
---------------------------------------
- Start at the base with a circular ring (cross-section loop) while tool axis is aligned
  with the spine tangent. With the provided spine, tangent is vertical at the endpoints,
  so tilt ≈ 0° at start, ≈ 90° near the top, ≈ 180° at the end.
- Move along the surface while always keeping tool axis perpendicular to the cross-section
  plane (tool axis = ±spine tangent).
- Uses calibration JSON: r_coeffs, z_coeffs, offplane_y_coeffs (optional), tip_angle_coeffs.
- Converts desired *physical tilt angle* (deg) -> commanded B position via inversion of
  tip_angle_coeffs (dense sample + interpolation).
- Computes stage XYZ from desired tip XYZ using the calibration offset model.

New feature added
-----------------
VARIABLE TUBE RADIUS along the path:
- Large base radius and smaller neck radius (smooth pinch).
- CLI args: --tube-radius-base, --tube-radius-neck, --neck-center, --neck-width, --neck-power

Usage
-----
python3 klein_bottle_slicer_calibrated_variable_radius.py \
  --calibration path/to/calibration.json \
  --out klein_variable_radius.gcode

If your physical “tilt direction” is flipped, use:
  --tool-axis-sign -1
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------- Defaults ----------------
DEFAULT_OUT = "gcode_generation/klein_bottle_swept_tube_variable_radius_calibrated.gcode"

# Tip-space placement (mm)
DEFAULT_START_X = 58.0
DEFAULT_START_Y = 50.0
DEFAULT_START_Z = -160.0

# Machine stage startup/end poses (raw stage axes)
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
DEFAULT_TRAVEL_FEED = 1200.0  # mm/min
DEFAULT_PRINT_FEED = 300.0    # mm/min

# Extrusion (U axis)
DEFAULT_EXTRUSION_PER_MM = 0.01
DEFAULT_PRIME_MM = 1.0

# Pressure offset / dwell sequencing (U axis)
DEFAULT_PRESSURE_OFFSET_MM = 5.0
DEFAULT_PRESSURE_ADVANCE_FEED = 120.0
DEFAULT_PRESSURE_RETRACT_FEED = 240.0
DEFAULT_PREFLOW_DWELL_MS = 150
DEFAULT_END_DWELL_MS = 250

# Debugging
DEFAULT_DEBUG_EVERY = 250  # emit periodic comments

# Path sampling
DEFAULT_BASE_RING_SEGMENTS = 240
DEFAULT_PATH_STEPS = 10000

# “Klein-ish” spine parameters (mm)
DEFAULT_SPINE_HEIGHT = 80.0
DEFAULT_SPINE_R0 = 45.0      # lateral sweep size
DEFAULT_SPINE_R1 = 15.0      # modulation
DEFAULT_SPINE_S = 18.0       # crossing-ish term strength
DEFAULT_SPINE_Z_WOBBLE = 0.0 # optional (usually 0)

# Tube radius (VARIABLE)
DEFAULT_TUBE_RADIUS_BASE = 30.0
DEFAULT_TUBE_RADIUS_NECK = 5.0
DEFAULT_NECK_CENTER = 0.5    # 0..1 along the loop; 0.5 tends to be near the “top”
DEFAULT_NECK_WIDTH = 0.30    # full width (0..1) affected by neck pinch
DEFAULT_NECK_POWER = 3.0     # higher = sharper pinch

# Tube twist and spiral wraps
DEFAULT_TWIST_DEG = 180.0    # rotate tube frame around tangent by 180° over the path (Klein/Möbius-ish)
DEFAULT_TURNS = 100.0

# Tool axis convention
DEFAULT_TOOL_AXIS_SIGN = +1  # +1 uses tangent, -1 uses -tangent

# Virtual stage-space bounding box
DEFAULT_BBOX_X_MIN = 0.0
DEFAULT_BBOX_X_MAX = 160.0
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
    pa: Optional[np.ndarray]  # tip_angle_coeffs (deg as function of B command)

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
class Waypoint:
    tip_xyz: np.ndarray
    b_cmd: float
    c_deg: float
    comment: str = ""


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


def tip_offset_xyz_physical(cal: Calibration, b_cmd: float, c_deg: float) -> np.ndarray:
    """
    Physical tip offset from stage origin:
      local transverse [r(B), y_off(B)] rotated by C into XY, z=z(B)
    """
    r = float(eval_r(cal, b_cmd))
    z = float(eval_z(cal, b_cmd))
    y_off = float(eval_offplane_y(cal, b_cmd))

    c = math.radians(float(c_deg))
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def stage_xyz_for_tip(cal: Calibration, tip_xyz: np.ndarray, b_cmd: float, c_deg: float) -> np.ndarray:
    return np.asarray(tip_xyz, dtype=float) - tip_offset_xyz_physical(cal, b_cmd, c_deg)


# ---------------- Tip-angle inversion (angle -> B command) ----------------

class TipAngleInverter:
    """
    Approximate inverse for tip_angle_deg(B_cmd) using dense sampling + interpolation.
    Works even if not strictly monotonic by sorting samples by angle.
    """
    def __init__(self, cal: Calibration, samples: int = 20001):
        if cal.pa is None:
            raise ValueError("Calibration JSON has no tip_angle_coeffs.")
        b = np.linspace(float(cal.b_min), float(cal.b_max), int(max(1001, samples)))
        a = eval_tip_angle_deg(cal, b)

        order = np.argsort(a)
        a_sorted = np.asarray(a[order], dtype=float)
        b_sorted = np.asarray(b[order], dtype=float)

        a_unique, idx = np.unique(a_sorted, return_index=True)
        b_unique = b_sorted[idx]

        self.a = a_unique
        self.b = b_unique
        self.a_min = float(self.a[0])
        self.a_max = float(self.a[-1])

    def angle_to_b(self, angle_deg: float) -> Tuple[float, bool]:
        """
        Returns (b_cmd, was_clamped_out_of_range).
        """
        ang = float(angle_deg)
        clamped = False
        if ang <= self.a_min:
            ang = self.a_min
            clamped = True
        if ang >= self.a_max:
            ang = self.a_max
            clamped = True
        b_cmd = float(np.interp(ang, self.a, self.b))
        return b_cmd, clamped


# ---------------- Geometry helpers: smooth spine + rotation-minimizing frame ----------------

def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n < eps:
        return np.zeros_like(v)
    return v / n


def _rodrigues_rotate(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize(axis)
    if np.linalg.norm(axis) < 1e-12:
        return v.copy()
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return v * c + np.cross(axis, v) * s + axis * (np.dot(axis, v)) * (1 - c)


def kleinish_spine(
    t: float,
    base_xyz: np.ndarray,
    height: float,
    r0: float,
    r1: float,
    s_cross: float,
    z_wobble: float,
) -> np.ndarray:
    """
    Smooth “up then down” spine:
      - xy excursion multiplied by sin(pi t)^2 so xy velocity is 0 at endpoints,
        giving vertical tangent at t=0 and t=1.
      - z = base_z + height*sin(pi t) gives up at start, horizontal near mid, down at end.
    """
    t = float(t)
    u = 2.0 * math.pi * t
    s = math.sin(math.pi * t)
    s2 = s * s

    loop_x = (r0 + r1 * math.cos(u)) * math.cos(u) + s_cross * math.sin(2.0 * u)
    loop_y = (r0 + r1 * math.cos(u)) * math.sin(u)

    x = float(base_xyz[0]) + s2 * loop_x
    y = float(base_xyz[1]) + s2 * loop_y
    z = float(base_xyz[2]) + height * math.sin(math.pi * t) + z_wobble * s2 * math.sin(2.0 * u)
    return np.array([x, y, z], dtype=float)


def build_rmf_frames(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Rotation-minimizing frame along a polyline.
    Returns T, N, B arrays (each Nx3) with N x B = T.
    """
    n = points.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 points to build frames.")

    T = np.zeros_like(points)
    for i in range(n):
        if i == 0:
            d = points[i + 1] - points[i]
        elif i == n - 1:
            d = points[i] - points[i - 1]
        else:
            d = points[i + 1] - points[i - 1]
        T[i] = _normalize(d)

    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(ref, T[0]))) > 0.95:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)

    N = np.zeros_like(points)
    N0 = ref - float(np.dot(ref, T[0])) * T[0]
    N[0] = _normalize(N0)
    B = np.zeros_like(points)
    B[0] = _normalize(np.cross(T[0], N[0]))

    for i in range(n - 1):
        Ti = T[i]
        Tj = T[i + 1]
        v = np.cross(Ti, Tj)
        nv = float(np.linalg.norm(v))
        if nv < 1e-10:
            N[i + 1] = N[i].copy()
            B[i + 1] = _normalize(np.cross(Tj, N[i + 1]))
            continue
        axis = v / nv
        angle = math.atan2(nv, float(np.dot(Ti, Tj)))
        N[i + 1] = _normalize(_rodrigues_rotate(N[i], axis, angle))
        B[i + 1] = _normalize(np.cross(Tj, N[i + 1]))

    return T, N, B


def apply_twist_about_tangent(T: np.ndarray, N: np.ndarray, B: np.ndarray, twist_deg_total: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Rotates (N,B) about T by twist angle that increases linearly from 0..twist_deg_total.
    """
    n = T.shape[0]
    N2 = np.zeros_like(N)
    B2 = np.zeros_like(B)
    for i in range(n):
        frac = 0.0 if n == 1 else (i / (n - 1))
        ang = math.radians(float(twist_deg_total) * frac)
        N2[i] = _normalize(_rodrigues_rotate(N[i], T[i], ang))
        B2[i] = _normalize(np.cross(T[i], N2[i]))
    return N2, B2


# ---------------- Variable tube radius profile ----------------

def smoothstep(x: float) -> float:
    x = max(0.0, min(1.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def tube_radius_profile(
    t: float,
    r_base: float,
    r_neck: float,
    neck_center: float,
    neck_width: float,
    neck_power: float,
) -> float:
    """
    Smooth pinch around neck_center:
      - outside neck region -> radius ~ r_base
      - at center -> radius ~ r_neck

    neck_width is full width in [0..1], neck_power controls sharpness.
    """
    t = float(t)
    half = max(1e-6, 0.5 * float(neck_width))
    d = abs(t - float(neck_center)) / half  # 0 at center, 1 at region edge
    w = 1.0 - smoothstep(min(1.0, d))       # 1 at center, 0 at edge/outside
    w = w ** float(neck_power)
    return float(r_base) * (1.0 - w) + float(r_neck) * w


# ---------------- Tool orientation mapping (direction -> tilt, azimuth) ----------------

def direction_to_tilt_azimuth_deg(d: np.ndarray, prev_c_deg: float) -> Tuple[float, float]:
    """
    Given a unit direction vector d:
      tilt = angle from +Z in [0..180]
      azimuth = atan2(y,x) in degrees
    If near-vertical, azimuth is ill-defined; keep prev_c_deg.
    """
    d = _normalize(d)
    dz = float(np.clip(d[2], -1.0, 1.0))
    tilt = math.degrees(math.acos(dz))

    xy = math.hypot(float(d[0]), float(d[1]))
    if xy < 1e-8:
        return tilt, float(prev_c_deg)

    az = math.degrees(math.atan2(float(d[1]), float(d[0])))
    return tilt, az


def unwrap_degrees(prev: float, cur: float) -> float:
    a = float(cur)
    p = float(prev)
    while a - p > 180.0:
        a -= 360.0
    while a - p < -180.0:
        a += 360.0
    return a


# ---------------- G-code writer ----------------

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


class CalibratedGCodeWriter:
    def __init__(
        self,
        fh,
        cal: Calibration,
        bbox: dict,
        travel_feed: float,
        print_feed: float,
        extrusion_per_mm: float,
        pressure_offset_mm: float,
        pressure_advance_feed: float,
        pressure_retract_feed: float,
        preflow_dwell_ms: int,
        end_dwell_ms: int,
        emit_extrusion: bool,
        warn_log: List[str],
        debug_every: int,
    ):
        self.f = fh
        self.cal = cal
        self.bbox = bbox
        self.travel_feed = float(travel_feed)
        self.print_feed = float(print_feed)
        self.extrusion_per_mm = float(extrusion_per_mm)
        self.pressure_offset_mm = float(pressure_offset_mm)
        self.pressure_advance_feed = float(pressure_advance_feed)
        self.pressure_retract_feed = float(pressure_retract_feed)
        self.preflow_dwell_ms = int(preflow_dwell_ms)
        self.end_dwell_ms = int(end_dwell_ms)
        self.emit_extrusion = bool(emit_extrusion)
        self.warn_log = warn_log
        self.debug_every = int(max(0, debug_every))

        self.u_material_abs = 0.0
        self.pressure_charged = False

        self.cur_stage_xyz: Optional[np.ndarray] = None
        self.cur_tip_xyz: Optional[np.ndarray] = None
        self.cur_b = 0.0
        self.cur_c = 0.0
        self.step_counter = 0

    def u_cmd_actual(self) -> float:
        return self.u_material_abs + (self.pressure_offset_mm if self.pressure_charged else 0.0)

    def _clamp_stage(self, p_stage: np.ndarray, context: str) -> np.ndarray:
        x, y, z = _clamp_stage_xyz_to_bbox(
            p_stage[0], p_stage[1], p_stage[2], self.bbox, context, self.warn_log
        )
        return np.array([x, y, z], dtype=float)

    def move_stage_xyzbc(self, p_stage: np.ndarray, b_cmd: float, c_deg: float, feed: float, comment: str = ""):
        if comment:
            self.f.write(f"; {comment}\n")
        pc = self._clamp_stage(np.asarray(p_stage, dtype=float), comment or "move_stage_xyzbc")
        axes = [
            (self.cal.x_axis, pc[0]),
            (self.cal.y_axis, pc[1]),
            (self.cal.z_axis, pc[2]),
            (self.cal.b_axis, float(b_cmd)),
            (self.cal.c_axis, float(c_deg)),
        ]
        self.f.write(f"G1 {_fmt_axes_move(axes)} F{float(feed):.0f}\n")
        self.cur_stage_xyz = pc.copy()
        self.cur_b = float(b_cmd)
        self.cur_c = float(c_deg)

    def move_to_tip(self, tip_xyz: np.ndarray, b_cmd: float, c_deg: float, feed: float, comment: str = ""):
        stage_xyz = stage_xyz_for_tip(self.cal, tip_xyz, b_cmd, c_deg)
        self.move_stage_xyzbc(stage_xyz, b_cmd=b_cmd, c_deg=c_deg, feed=feed, comment=comment)
        self.cur_tip_xyz = np.asarray(tip_xyz, dtype=float).copy()

    def pressure_preload(self):
        if self.emit_extrusion and self.pressure_offset_mm > 0.0 and not self.pressure_charged:
            self.pressure_charged = True
            self.f.write("; pressure preload (continuous)\n")
            self.f.write(f"G1 {self.cal.u_axis}{self.u_cmd_actual():.3f} F{self.pressure_advance_feed:.0f}\n")
            if self.preflow_dwell_ms > 0:
                self.f.write(f"G4 P{self.preflow_dwell_ms}\n")

    def pressure_release(self):
        if self.emit_extrusion and self.pressure_offset_mm > 0.0 and self.pressure_charged:
            if self.end_dwell_ms > 0:
                self.f.write("; end dwell\n")
                self.f.write(f"G4 P{self.end_dwell_ms}\n")
            self.pressure_charged = False
            self.f.write("; pressure release\n")
            self.f.write(f"G1 {self.cal.u_axis}{self.u_cmd_actual():.3f} F{self.pressure_retract_feed:.0f}\n")

    def print_waypoint(self, wp: Waypoint, feed: float, do_extrude: bool, prev_tip_xyz: Optional[np.ndarray]):
        tip_xyz = np.asarray(wp.tip_xyz, dtype=float)
        b_cmd = float(wp.b_cmd)
        c_deg = float(wp.c_deg)

        stage_xyz = stage_xyz_for_tip(self.cal, tip_xyz, b_cmd, c_deg)

        off = tip_offset_xyz_physical(self.cal, b_cmd, c_deg)
        if np.linalg.norm((stage_xyz + off) - tip_xyz) > 1e-7:
            raise RuntimeError("Tip tracking consistency failed (stage + offset != tip).")

        pc = self._clamp_stage(stage_xyz, wp.comment or "print_waypoint")

        axes = [
            (self.cal.x_axis, pc[0]),
            (self.cal.y_axis, pc[1]),
            (self.cal.z_axis, pc[2]),
            (self.cal.b_axis, b_cmd),
            (self.cal.c_axis, c_deg),
        ]

        if self.emit_extrusion and do_extrude:
            seg_len = 0.0 if prev_tip_xyz is None else float(np.linalg.norm(tip_xyz - prev_tip_xyz))
            self.u_material_abs += self.extrusion_per_mm * seg_len
            axes.append((self.cal.u_axis, self.u_cmd_actual()))

        if wp.comment and (self.debug_every > 0) and (self.step_counter % self.debug_every == 0):
            tip_ang = float(eval_tip_angle_deg(self.cal, b_cmd))
            self.f.write(f"; step={self.step_counter} {wp.comment} | Bcmd={b_cmd:.4f} C={c_deg:.2f} tipAngle={tip_ang:.2f}\n")

        self.f.write(f"G1 {_fmt_axes_move(axes)} F{float(feed):.0f}\n")
        self.cur_stage_xyz = pc.copy()
        self.cur_tip_xyz = tip_xyz.copy()
        self.cur_b = b_cmd
        self.cur_c = c_deg
        self.step_counter += 1


# ---------------- Path construction ----------------

def make_base_ring_waypoints(
    center: np.ndarray,
    T_dir: np.ndarray,
    N_dir: np.ndarray,
    B_dir: np.ndarray,
    tube_radius: float,
    inverter: TipAngleInverter,
    tool_axis_sign: int,
    c_start_deg: float,
    segments: int,
    comment_prefix: str,
) -> List[Waypoint]:
    """
    A single closed circular ring in the plane perpendicular to T_dir.
    Tool axis is perpendicular to cross-section => aligned to tool_dir (±T_dir).
    """
    tool_dir = _normalize(tool_axis_sign * _normalize(T_dir))
    prev_c = float(c_start_deg)

    tilt, az = direction_to_tilt_azimuth_deg(tool_dir, prev_c)
    az = unwrap_degrees(prev_c, az)
    b_cmd, clamped = inverter.angle_to_b(tilt)
    c_deg = az

    wps: List[Waypoint] = []
    m = int(max(8, segments))
    for k in range(m + 1):  # close ring
        phi = 2.0 * math.pi * (k / m)
        p = center + float(tube_radius) * (math.cos(phi) * N_dir + math.sin(phi) * B_dir)
        wps.append(Waypoint(
            tip_xyz=p,
            b_cmd=b_cmd,
            c_deg=c_deg,
            comment=f"{comment_prefix} ring k={k}/{m}" + (" (B clamped)" if clamped else "")
        ))
    return wps


def make_spiral_surface_waypoints_variable_radius(
    spine_pts: np.ndarray,
    T: np.ndarray,
    N: np.ndarray,
    B: np.ndarray,
    r_base: float,
    r_neck: float,
    neck_center: float,
    neck_width: float,
    neck_power: float,
    twist_deg_total: float,
    turns: float,
    inverter: TipAngleInverter,
    tool_axis_sign: int,
    c_start_deg: float,
    comment_prefix: str,
) -> List[Waypoint]:
    """
    Continuous spiral on tube surface with VARIABLE radius:
      phi(i) = 2*pi*turns*frac
      r(i) = tube_radius_profile(frac, ...)
      tip(i) = spine(i) + r(i)*(cos(phi)*N + sin(phi)*B)

    Tool axis is always perpendicular to cross-section plane => tool axis = ±T(i).
    """
    n = spine_pts.shape[0]
    if n < 3:
        raise ValueError("Need at least 3 spine samples.")

    if abs(float(twist_deg_total)) > 1e-9:
        N2, B2 = apply_twist_about_tangent(T, N, B, twist_deg_total=float(twist_deg_total))
    else:
        N2, B2 = N, B

    wps: List[Waypoint] = []
    prev_c = float(c_start_deg)

    for i in range(n):
        frac = 0.0 if n == 1 else (i / (n - 1))
        phi = 2.0 * math.pi * float(turns) * frac

        r_i = tube_radius_profile(
            frac,
            r_base=float(r_base),
            r_neck=float(r_neck),
            neck_center=float(neck_center),
            neck_width=float(neck_width),
            neck_power=float(neck_power),
        )

        tip = spine_pts[i] + r_i * (math.cos(phi) * N2[i] + math.sin(phi) * B2[i])

        tool_dir = _normalize(tool_axis_sign * _normalize(T[i]))
        tilt, az = direction_to_tilt_azimuth_deg(tool_dir, prev_c)
        az = unwrap_degrees(prev_c, az)
        prev_c = az

        b_cmd, clamped = inverter.angle_to_b(tilt)

        wps.append(Waypoint(
            tip_xyz=tip,
            b_cmd=b_cmd,
            c_deg=az,
            comment=f"{comment_prefix} i={i}/{n-1} frac={frac:.3f} r={r_i:.3f} tilt={tilt:.2f}" + (" (B clamped)" if clamped else "")
        ))
    return wps


# ---------------- Main G-code generation ----------------

def write_klein_gcode(
    out_path: str,
    cal: Calibration,
    pattern_start_tip: Tuple[float, float, float],
    machine_start_pose: Tuple[float, float, float, float, float],
    machine_end_pose: Tuple[float, float, float, float, float],
    safe_approach_z: float,
    travel_feed: float,
    print_feed: float,
    extrusion_per_mm: float,
    prime_mm: float,
    pressure_offset_mm: float,
    pressure_advance_feed: float,
    pressure_retract_feed: float,
    preflow_dwell_ms: int,
    end_dwell_ms: int,
    bbox: dict,
    debug_every: int,
    # geometry
    spine_height: float,
    spine_r0: float,
    spine_r1: float,
    spine_s: float,
    spine_z_wobble: float,
    tube_radius_base: float,
    tube_radius_neck: float,
    neck_center: float,
    neck_width: float,
    neck_power: float,
    twist_deg_total: float,
    turns: float,
    base_ring_segments: int,
    path_steps: int,
    tool_axis_sign: int,
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    emit_extrusion = float(extrusion_per_mm) != 0.0
    bbox_warnings: List[str] = []

    base_xyz = np.array([float(pattern_start_tip[0]), float(pattern_start_tip[1]), float(pattern_start_tip[2])], dtype=float)

    # Build spine samples (t in [0,1])
    n_spine = int(max(100, path_steps))
    ts = np.linspace(0.0, 1.0, n_spine)
    spine_pts = np.stack([
        kleinish_spine(
            t=float(t),
            base_xyz=base_xyz,
            height=float(spine_height),
            r0=float(spine_r0),
            r1=float(spine_r1),
            s_cross=float(spine_s),
            z_wobble=float(spine_z_wobble),
        )
        for t in ts
    ], axis=0)

    # Frames
    T, N, B = build_rmf_frames(spine_pts)

    inverter = TipAngleInverter(cal)

    # Base ring (use BASE radius)
    base_center = spine_pts[0]
    base_ring = make_base_ring_waypoints(
        center=base_center,
        T_dir=T[0],
        N_dir=N[0],
        B_dir=B[0],
        tube_radius=float(tube_radius_base),
        inverter=inverter,
        tool_axis_sign=int(tool_axis_sign),
        c_start_deg=0.0,
        segments=int(base_ring_segments),
        comment_prefix="base",
    )

    # Spiral with VARIABLE radius
    spiral = make_spiral_surface_waypoints_variable_radius(
        spine_pts=spine_pts,
        T=T,
        N=N,
        B=B,
        r_base=float(tube_radius_base),
        r_neck=float(tube_radius_neck),
        neck_center=float(neck_center),
        neck_width=float(neck_width),
        neck_power=float(neck_power),
        twist_deg_total=float(twist_deg_total),
        turns=float(turns),
        inverter=inverter,
        tool_axis_sign=int(tool_axis_sign),
        c_start_deg=float(base_ring[-1].c_deg if base_ring else 0.0),
        comment_prefix="spiral",
    )

    waypoints: List[Waypoint] = []
    waypoints.extend(base_ring)
    waypoints.append(Waypoint(
        tip_xyz=spiral[0].tip_xyz,
        b_cmd=spiral[0].b_cmd,
        c_deg=spiral[0].c_deg,
        comment="transition to spiral start"
    ))
    waypoints.extend(spiral)

    # Header summaries
    b_cmd_start = float(spiral[0].b_cmd)
    c_deg_start = float(spiral[0].c_deg)
    tilt_start = float(eval_tip_angle_deg(cal, b_cmd_start))
    b_cmd_mid = float(spiral[len(spiral)//2].b_cmd)
    tilt_mid = float(eval_tip_angle_deg(cal, b_cmd_mid))
    b_cmd_end = float(spiral[-1].b_cmd)
    tilt_end = float(eval_tip_angle_deg(cal, b_cmd_end))

    # radius samples
    r0s = tube_radius_profile(0.0, tube_radius_base, tube_radius_neck, neck_center, neck_width, neck_power)
    rmid = tube_radius_profile(0.5, tube_radius_base, tube_radius_neck, neck_center, neck_width, neck_power)
    r1s = tube_radius_profile(1.0, tube_radius_base, tube_radius_neck, neck_center, neck_width, neck_power)

    with open(out_path, "w") as f:
        f.write("; generated by klein_bottle_slicer_calibrated_variable_radius.py\n")
        f.write("; calibrated tip-position planning: stage = tip - offset_tip(B_cmd,C)\n")
        f.write("; tool axis is kept perpendicular to cross-section plane by aligning tool axis to spine tangent\n")
        f.write(f"; axes: X->{cal.x_axis}, Y->{cal.y_axis}, Z->{cal.z_axis}, B->{cal.b_axis}, C->{cal.c_axis}, U->{cal.u_axis}\n")
        f.write(f"; calibration B range (command units): [{cal.b_min:.6f}, {cal.b_max:.6f}]\n")
        f.write(f"; start tip-space base = [{base_xyz[0]:.3f}, {base_xyz[1]:.3f}, {base_xyz[2]:.3f}]\n")
        f.write(f"; spine: height={spine_height:.3f}, r0={spine_r0:.3f}, r1={spine_r1:.3f}, s={spine_s:.3f}, z_wobble={spine_z_wobble:.3f}\n")
        f.write("; tube radius profile:\n")
        f.write(f";   r_base={tube_radius_base:.3f}, r_neck={tube_radius_neck:.3f}, neck_center={neck_center:.3f}, neck_width={neck_width:.3f}, neck_power={neck_power:.3f}\n")
        f.write(f";   radius samples: r(t=0)={r0s:.3f}, r(t=0.5)={rmid:.3f}, r(t=1)={r1s:.3f}\n")
        f.write(f"; tube: twist_total={twist_deg_total:.3f} deg, spiral_turns={turns:.3f}\n")
        f.write(f"; sampling: base_ring_segments={base_ring_segments}, path_steps={path_steps}\n")
        f.write(f"; feeds: travel={travel_feed:.1f} mm/min, print={print_feed:.1f} mm/min\n")
        f.write(f"; extrusion_per_mm={extrusion_per_mm:.6f}, pressure_offset={pressure_offset_mm:.3f}\n")
        f.write(f"; tool_axis_sign={tool_axis_sign} (use -1 if you need to flip the tool direction)\n")
        f.write(f"; expected tip-angle samples (from calibration poly): start≈{tilt_start:.2f}°, mid≈{tilt_mid:.2f}°, end≈{tilt_end:.2f}°\n")
        f.write("G90\n")

        if emit_extrusion:
            f.write("M82\n")
            f.write(f"G92 {cal.u_axis}0\n")
            if abs(float(prime_mm)) > 0.0:
                f.write(f"G1 {cal.u_axis}{float(prime_mm):.3f} F{max(60.0, float(pressure_advance_feed)):.0f} ; prime\n")

        g = CalibratedGCodeWriter(
            fh=f,
            cal=cal,
            bbox=bbox,
            travel_feed=travel_feed,
            print_feed=print_feed,
            extrusion_per_mm=extrusion_per_mm,
            pressure_offset_mm=pressure_offset_mm,
            pressure_advance_feed=pressure_advance_feed,
            pressure_retract_feed=pressure_retract_feed,
            preflow_dwell_ms=preflow_dwell_ms,
            end_dwell_ms=end_dwell_ms,
            emit_extrusion=emit_extrusion,
            warn_log=bbox_warnings,
            debug_every=debug_every,
        )
        if emit_extrusion:
            g.u_material_abs = float(prime_mm)

        # Safe startup (machine-stage coordinates)
        msx, msy, msz, msb, msc = [float(v) for v in machine_start_pose]
        f.write("; safe startup approach (machine stage coordinates)\n")
        g.move_stage_xyzbc(
            np.array([msx, msy, float(safe_approach_z)], dtype=float),
            b_cmd=msb, c_deg=msc, feed=travel_feed,
            comment="startup: move to safe Z at machine-start XY"
        )
        g.move_stage_xyzbc(
            np.array([msx, msy, msz], dtype=float),
            b_cmd=msb, c_deg=msc, feed=travel_feed,
            comment="startup: dive to machine-start Z"
        )

        # Move to first waypoint (travel)
        first = waypoints[0]
        g.move_to_tip(first.tip_xyz, first.b_cmd, first.c_deg, feed=travel_feed, comment="move to first waypoint")

        # Preload once, print continuously
        g.pressure_preload()

        prev_tip: Optional[np.ndarray] = first.tip_xyz.copy()
        for wp in waypoints[1:]:
            g.print_waypoint(wp, feed=print_feed, do_extrude=True, prev_tip_xyz=prev_tip)
            prev_tip = np.asarray(wp.tip_xyz, dtype=float)

        # Release pressure at end
        g.pressure_release()

        # Safe end move (machine-stage coordinates)
        mex, mey, mez, meb, mec = [float(v) for v in machine_end_pose]
        f.write("; safe end move (machine stage coordinates)\n")
        if g.cur_stage_xyz is not None:
            g.move_stage_xyzbc(
                np.array([g.cur_stage_xyz[0], g.cur_stage_xyz[1], float(safe_approach_z)], dtype=float),
                b_cmd=meb, c_deg=mec, feed=travel_feed,
                comment="end: raise to safe Z"
            )
        g.move_stage_xyzbc(
            np.array([mex, mey, float(safe_approach_z)], dtype=float),
            b_cmd=meb, c_deg=mec, feed=travel_feed,
            comment="end: move XY at safe Z"
        )
        g.move_stage_xyzbc(
            np.array([mex, mey, mez], dtype=float),
            b_cmd=meb, c_deg=mec, feed=travel_feed,
            comment="end: dive to machine end Z"
        )

        if bbox_warnings:
            f.write("; virtual bbox clamp warnings:\n")
            for msg in bbox_warnings:
                f.write(f"; {msg}\n")

        f.write(f"; total waypoints printed (incl. base ring + spiral) = {len(waypoints)}\n")
        f.write(f"; bbox warning count = {len(bbox_warnings)}\n")
        f.write("; --- end ---\n")

    print(f"Wrote {out_path}")
    print(f"Waypoints: {len(waypoints)}")
    print(f"Tip-angle (poly) samples: start≈{tilt_start:.2f}°, mid≈{tilt_mid:.2f}°, end≈{tilt_end:.2f}°")
    print(f"Radius samples: r(0)={r0s:.3f}, r(0.5)={rmid:.3f}, r(1)={r1s:.3f}")
    if bbox_warnings:
        print(f"WARNING: bbox clamps occurred: {len(bbox_warnings)}")


# ---------------- Main ----------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate calibrated G-code for a swept-tube Klein-bottle-like print with variable tube radius (base->neck)."
    )
    ap.add_argument("--calibration", required=True, help="Path to calibration JSON.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code file.")

    # Tip-space base placement
    ap.add_argument("--start-x", type=float, default=DEFAULT_START_X)
    ap.add_argument("--start-y", type=float, default=DEFAULT_START_Y)
    ap.add_argument("--start-z", type=float, default=DEFAULT_START_Z)

    # Machine startup/end (raw stage)
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

    ap.add_argument("--safe-approach-z", type=float, default=DEFAULT_SAFE_APPROACH_Z)

    # Motion
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED)
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED)

    # Extrusion + pressure
    ap.add_argument("--extrusion-per-mm", type=float, default=DEFAULT_EXTRUSION_PER_MM)
    ap.add_argument("--prime-mm", type=float, default=DEFAULT_PRIME_MM)
    ap.add_argument("--pressure-offset-mm", type=float, default=DEFAULT_PRESSURE_OFFSET_MM)
    ap.add_argument("--pressure-advance-feed", type=float, default=DEFAULT_PRESSURE_ADVANCE_FEED)
    ap.add_argument("--pressure-retract-feed", type=float, default=DEFAULT_PRESSURE_RETRACT_FEED)
    ap.add_argument("--preflow-dwell-ms", type=int, default=DEFAULT_PREFLOW_DWELL_MS)
    ap.add_argument("--end-dwell-ms", type=int, default=DEFAULT_END_DWELL_MS)

    # Debug
    ap.add_argument("--debug-every", type=int, default=DEFAULT_DEBUG_EVERY)

    # Path sampling
    ap.add_argument("--base-ring-segments", type=int, default=DEFAULT_BASE_RING_SEGMENTS)
    ap.add_argument("--path-steps", type=int, default=DEFAULT_PATH_STEPS)

    # Spine geometry
    ap.add_argument("--spine-height", type=float, default=DEFAULT_SPINE_HEIGHT)
    ap.add_argument("--spine-r0", type=float, default=DEFAULT_SPINE_R0)
    ap.add_argument("--spine-r1", type=float, default=DEFAULT_SPINE_R1)
    ap.add_argument("--spine-s", type=float, default=DEFAULT_SPINE_S)
    ap.add_argument("--spine-z-wobble", type=float, default=DEFAULT_SPINE_Z_WOBBLE)

    # Tube VARIABLE radius
    ap.add_argument("--tube-radius-base", type=float, default=DEFAULT_TUBE_RADIUS_BASE)
    ap.add_argument("--tube-radius-neck", type=float, default=DEFAULT_TUBE_RADIUS_NECK)
    ap.add_argument("--neck-center", type=float, default=DEFAULT_NECK_CENTER)
    ap.add_argument("--neck-width", type=float, default=DEFAULT_NECK_WIDTH)
    ap.add_argument("--neck-power", type=float, default=DEFAULT_NECK_POWER)

    # Twist + surface wrap
    ap.add_argument("--twist-deg", type=float, default=DEFAULT_TWIST_DEG)
    ap.add_argument("--turns", type=float, default=DEFAULT_TURNS, help="How many wraps around the tube over the full path (spiral).")

    # Tool direction convention
    ap.add_argument("--tool-axis-sign", type=int, choices=[-1, 1], default=DEFAULT_TOOL_AXIS_SIGN,
                    help="Use -1 to flip tool axis (uses -tangent).")

    # Virtual stage bbox
    ap.add_argument("--bbox-x-min", type=float, default=DEFAULT_BBOX_X_MIN)
    ap.add_argument("--bbox-x-max", type=float, default=DEFAULT_BBOX_X_MAX)
    ap.add_argument("--bbox-y-min", type=float, default=DEFAULT_BBOX_Y_MIN)
    ap.add_argument("--bbox-y-max", type=float, default=DEFAULT_BBOX_Y_MAX)
    ap.add_argument("--bbox-z-min", type=float, default=DEFAULT_BBOX_Z_MIN)
    ap.add_argument("--bbox-z-max", type=float, default=DEFAULT_BBOX_Z_MAX)

    args = ap.parse_args()

    cal = load_calibration(args.calibration)

    bbox = {
        "x_min": float(min(args.bbox_x_min, args.bbox_x_max)),
        "x_max": float(max(args.bbox_x_min, args.bbox_x_max)),
        "y_min": float(min(args.bbox_y_min, args.bbox_y_max)),
        "y_max": float(max(args.bbox_y_min, args.bbox_y_max)),
        "z_min": float(min(args.bbox_z_min, args.bbox_z_max)),
        "z_max": float(max(args.bbox_z_min, args.bbox_z_max)),
    }

    pattern_start_tip = (float(args.start_x), float(args.start_y), float(args.start_z))
    machine_start_pose = (
        float(args.machine_start_x), float(args.machine_start_y), float(args.machine_start_z),
        float(args.machine_start_b), float(args.machine_start_c),
    )
    machine_end_pose = (
        float(args.machine_end_x), float(args.machine_end_y), float(args.machine_end_z),
        float(args.machine_end_b), float(args.machine_end_c),
    )

    write_klein_gcode(
        out_path=str(args.out),
        cal=cal,
        pattern_start_tip=pattern_start_tip,
        machine_start_pose=machine_start_pose,
        machine_end_pose=machine_end_pose,
        safe_approach_z=float(args.safe_approach_z),
        travel_feed=float(args.travel_feed),
        print_feed=float(args.print_feed),
        extrusion_per_mm=float(args.extrusion_per_mm),
        prime_mm=float(args.prime_mm),
        pressure_offset_mm=float(args.pressure_offset_mm),
        pressure_advance_feed=float(args.pressure_advance_feed),
        pressure_retract_feed=float(args.pressure_retract_feed),
        preflow_dwell_ms=int(args.preflow_dwell_ms),
        end_dwell_ms=int(args.end_dwell_ms),
        bbox=bbox,
        debug_every=int(args.debug_every),
        spine_height=float(args.spine_height),
        spine_r0=float(args.spine_r0),
        spine_r1=float(args.spine_r1),
        spine_s=float(args.spine_s),
        spine_z_wobble=float(args.spine_z_wobble),
        tube_radius_base=float(args.tube_radius_base),
        tube_radius_neck=float(args.tube_radius_neck),
        neck_center=float(args.neck_center),
        neck_width=float(args.neck_width),
        neck_power=float(args.neck_power),
        twist_deg_total=float(args.twist_deg),
        turns=float(args.turns),
        base_ring_segments=int(args.base_ring_segments),
        path_steps=int(args.path_steps),
        tool_axis_sign=int(args.tool_axis_sign),
    )


if __name__ == "__main__":
    main()