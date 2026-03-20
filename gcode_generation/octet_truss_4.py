#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code for an octet truss lattice housed inside a cube,
using calibrated B/C kinematics with exact tip-position tracking and soft tangent alignment.

This script preserves the original motion-planning / kinematic features from the tetrahedral
version and swaps only the geometry layer to an octet-based design inside a cubic envelope.

IMPORTANT CORRECTION
--------------------
Physical tip-position kinematics support an optional off-plane transverse term from calibration.

We only apply an azimuth flip to the *attack/nozzle direction* model used for tangency:
    theta_attack = C + attack_azimuth_flip_deg   (default +180 deg)

This means:
- Tip compensation / stage solve uses physical kinematics:
    local transverse v(B) = [r(B), y_off(B)]
    [x,y] = Rot(C) @ v(B), z = z(B)
    offset_tip(B,C) = [x(B,C), y(B,C), z(B)]
- Tangency / nozzle attack direction uses corrected azimuth:
    nozzle_dir(B,C) = [sin(a(B)) cos(theta_attack), sin(a(B)) sin(theta_attack), cos(a(B))]

PRIORITIES
----------
1) Exact tip position (hard priority):
     p_stage = p_tip_desired - offset_tip(B, C)

2) Tangential nozzle/extrusion direction along each strut (soft priority)
   using the user convention:
     tip angle is measured FROM VERTICAL
     - vertical line -> ~0 deg
     - horizontal line -> ~90 deg

3) Head-in-front direction preference (edge direction selection):
   Prefer edge direction where the print head (stage origin) is in front of the deposited tip
   along the writing direction:
       head_lead = dot(-offset_tip(B,C), tangent)
   Prefer head_lead > 0.

Geometry
--------
Octet truss based on an FCC nearest-neighbor graph embedded in a cube.
- --order N : N octet unit cells along each cube edge.
- --pitch   : strut length of the octet lattice (nearest-neighbor FCC edge length).
- optional cube frame: adds subdivided outer cube edges so the lattice is visibly housed in a cube.

Conventions:
- Cube origin (--origin-x/y/z) is the lower-min corner of the cube in tip space.
- Octet cell edge length = pitch * sqrt(2)
- Total cube edge length = order * pitch * sqrt(2)

Build ordering / collision reduction heuristic
----------------------------------------------
Edges are ordered to start from the bottom-center and build upward:
  - lower edges first (z_min, then z_max)
  - then center-out on the bottom plane
  - then head-front preference
  - then shorter travel
  - then tangent error

Extrusion pressure offset behavior
----------------------------------
If extrusion is enabled and pressure_offset_mm > 0:
  Before each print pass (edge):
    - advance U by pressure_offset_mm
    - dwell preflow_dwell_ms
  After each print pass:
    - dwell node_dwell_ms (allow node to form / liquid to flow)
    - retract U by pressure_offset_mm
  At end of print:
    - retract U by pressure_offset_mm if still charged

Notes
-----
- Every generated edge is extruded exactly once.
- Each strut is printed as a straight line with constant B/C (constant tangent orientation).
- XYZ stage moves are exact for the tip path.
- Optional U-axis extrusion is supported (absolute mode).
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import ArrayLike


# ---------------- Defaults (CLI-overridable) ----------------
DEFAULT_OUT = "gcode_generation/octet_truss_cube_tip_priority_tangent_soft.gcode"

# Placement in tip space (world coordinates): lower/min cube corner
DEFAULT_ORIGIN_X = 65.0
DEFAULT_ORIGIN_Y = 50.0
DEFAULT_ORIGIN_Z = -175.0

# Geometry
DEFAULT_ORDER = 1                 # number of octet unit cells along cube edge
DEFAULT_PITCH_MM = 20.0           # octet strut length (FCC nearest-neighbor distance)
DEFAULT_EDGE_SAMPLES = 12         # interpolation segments per strut (>=2)
DEFAULT_INCLUDE_CUBE_FRAME = True

# Motion
DEFAULT_TRAVEL_FEED = 1000.0      # mm/min
DEFAULT_PRINT_FEED = 200.0       # mm/min (coordinated path feed)

# User-defined pre/post print poses (stage axes)
DEFAULT_START_X = 65.0
DEFAULT_START_Y = 20.0
DEFAULT_START_Z = 0.0
DEFAULT_START_B = 0.0
DEFAULT_START_C = 0.0
DEFAULT_END_X = 65.0
DEFAULT_END_Y = 20.0
DEFAULT_END_Z = 0.0
DEFAULT_END_B = 0.0
DEFAULT_END_C = 0.0
DEFAULT_SAFE_APPROACH_Z = 0.0
DEFAULT_TRAVEL_SAMPLES = 24        # segmented samples for non-print tip-tracked travel / B-C pivots
DEFAULT_TRAVEL_Z_CLEARANCE_MM = 1.0
DEFAULT_TRAVEL_LINE_CLEARANCE_MM = 0.25
DEFAULT_NEUTRAL_PITCH_TARGET_DEG = 0.0
DEFAULT_C_WINDOW_MIN = -360.0
DEFAULT_C_WINDOW_MAX = 360.0
DEFAULT_C_RECENTER_MARGIN_DEG = 45.0

# Virtual XYZ bounding box (enforced during print/travel)
DEFAULT_BBOX_X_MIN = 0.0
DEFAULT_BBOX_X_MAX = 160.0
DEFAULT_BBOX_Y_MIN = 0.0
DEFAULT_BBOX_Y_MAX = 160.0
DEFAULT_BBOX_Z_MIN = -190.0
DEFAULT_BBOX_Z_MAX = -10.0

# Extrusion
DEFAULT_EXTRUSION_PER_MM = 0.004    # U mm per mm of printed path; set >0 to enable U extrusion
DEFAULT_PRIME_MM = 1.0            # U mm material prime (not pressure preload)

# Pressure offset / dwell sequencing (U-axis)
DEFAULT_PRESSURE_OFFSET_MM = 4.0
DEFAULT_PRESSURE_ADVANCE_FEED = 120.0
DEFAULT_PRESSURE_RETRACT_FEED = 240.0
DEFAULT_PREFLOW_DWELL_MS = 750
DEFAULT_NODE_DWELL_MS = 750

# B/C solve
DEFAULT_B_SEARCH_SAMPLES = 721
DEFAULT_B_CONTINUITY_WEIGHT = 0.03
DEFAULT_C_CONTINUITY_WEIGHT = 0.003
DEFAULT_MIN_HEAD_LEAD_MM = 0.0

# Tip-angle convention tweak (for calibration conventions)
DEFAULT_TIP_ANGLE_SIGN = 1.0
DEFAULT_TIP_ANGLE_OFFSET_DEG = 0.0

# Attack-direction correction ONLY (does NOT affect physical tip kinematics)
DEFAULT_ATTACK_AZIMUTH_FLIP_DEG = 180.0

# Syringe/tube math helpers
DEFAULT_SYRINGE_MM_PER_ML = 6.0
DEFAULT_TUBE_ID_INCH = 0.02
OFFPLANE_SIGN = -1.0
# ------------------------------------------------------------


@dataclass
class Calibration:
    pr: np.ndarray
    pz: np.ndarray
    py_off: Optional[np.ndarray]
    pa: np.ndarray

    b_min: float
    b_max: float

    x_axis: str
    y_axis: str
    z_axis: str
    b_axis: str
    c_axis: str
    u_axis: str

    c_180_deg: float

    offplane_y_equation: Optional[str] = None
    offplane_y_r_squared: Optional[float] = None


@dataclass
class EdgePlan:
    p0_tip: np.ndarray
    p1_tip: np.ndarray
    b_cmd: float
    c_cmd: float
    angle_error_deg: float
    nozzle_dir: np.ndarray
    offset_vec: np.ndarray
    edge_len: float
    head_lead_mm: float
    head_front_ok: bool


# ---------------- Calibration / kinematics helpers ----------------

def poly_eval(coeffs: ArrayLike, u: ArrayLike, default_if_none: Optional[float] = None) -> np.ndarray:
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


def unwrap_deg_near(angle_deg: ArrayLike, ref_deg: Optional[float]) -> np.ndarray:
    """
    Return representation(s) of angle_deg shifted by k*360 to be nearest ref_deg.
    If ref_deg is None, wrap to [0, 360).
    """
    a = np.asarray(angle_deg, dtype=float)
    if ref_deg is None:
        return np.mod(a, 360.0)
    k = np.round((float(ref_deg) - a) / 360.0)
    return a + 360.0 * k


def choose_equivalent_c_in_window(
    target_c_deg: float,
    ref_c_deg: Optional[float],
    c_min_deg: float = DEFAULT_C_WINDOW_MIN,
    c_max_deg: float = DEFAULT_C_WINDOW_MAX,
) -> float:
    target = float(target_c_deg)
    ref = 0.0 if ref_c_deg is None else float(ref_c_deg)
    candidates = [target + 360.0 * k for k in range(-8, 9) if c_min_deg <= target + 360.0 * k <= c_max_deg]
    if candidates:
        return min(candidates, key=lambda v: abs(v - ref))
    nearest = target + 360.0 * round((ref - target) / 360.0)
    return max(c_min_deg, min(c_max_deg, nearest))


def c_needs_recentering(
    c_deg: float,
    c_min_deg: float = DEFAULT_C_WINDOW_MIN,
    c_max_deg: float = DEFAULT_C_WINDOW_MAX,
    margin_deg: float = DEFAULT_C_RECENTER_MARGIN_DEG,
) -> bool:
    c_val = float(c_deg)
    return c_val < (float(c_min_deg) + float(margin_deg)) or c_val > (float(c_max_deg) - float(margin_deg))


def max_written_z_value(
    written_segments: List[Tuple[np.ndarray, np.ndarray]],
    default_z: float,
) -> float:
    if not written_segments:
        return float(default_z)
    return max(max(float(a[2]), float(b[2])) for a, b in written_segments)


def c_equivalent_near_ref(target_c_deg: float, ref_c_deg: float) -> float:
    return float(target_c_deg) + 360.0 * round((float(ref_c_deg) - float(target_c_deg)) / 360.0)


def same_tip_reorient_needed(
    b_from: float,
    c_from: float,
    b_to: float,
    c_to: float,
    b_tol: float = 1e-6,
    c_tol_deg: float = 1e-3,
) -> bool:
    c_to_near = c_equivalent_near_ref(float(c_to), float(c_from))
    return abs(float(b_to) - float(b_from)) > float(b_tol) or abs(c_to_near - float(c_from)) > float(c_tol_deg)


def load_calibration(json_path: str) -> Calibration:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")

    with p.open("r") as f:
        data = json.load(f)

    cubic = data["cubic_coefficients"]
    pr = np.array(cubic["r_coeffs"], dtype=float)
    pz = np.array(cubic["z_coeffs"], dtype=float)
    pa = np.array(cubic["tip_angle_coeffs"], dtype=float)
    py_off_raw = cubic.get("offplane_y_coeffs", None)
    py_off = None if py_off_raw is None else np.array(py_off_raw, dtype=float)

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
        c_180_deg=c_180,
        offplane_y_equation=cubic.get("offplane_y_equation"),
        offplane_y_r_squared=(None if cubic.get("offplane_y_r_squared") is None else float(cubic["offplane_y_r_squared"])),
    )


def eval_r(cal: Calibration, b: ArrayLike) -> np.ndarray:
    return poly_eval(cal.pr, b)


def eval_z(cal: Calibration, b: ArrayLike) -> np.ndarray:
    return poly_eval(cal.pz, b)


def eval_offplane_y(cal: Calibration, b: ArrayLike) -> np.ndarray:
    return OFFPLANE_SIGN * poly_eval(cal.py_off, b, default_if_none=0.0)


def predict_r_z_offplane(cal: Calibration, b: float) -> Tuple[float, float, float]:
    r = float(eval_r(cal, b))
    z = float(eval_z(cal, b))
    y_off = float(eval_offplane_y(cal, b))
    return r, z, y_off


def transverse_radius_from_b(cal: Calibration, b: float) -> float:
    r, _, y_off = predict_r_z_offplane(cal, b)
    return float(math.hypot(r, y_off))


def transverse_phase_from_b(cal: Calibration, b: float) -> float:
    return float(math.atan2(predict_r_z_offplane(cal, b)[2], predict_r_z_offplane(cal, b)[0]))


def predict_tip_xyz_from_bc(cal: Calibration, b: float, c_deg: float) -> np.ndarray:
    r, z, y_off = predict_r_z_offplane(cal, b)
    c = math.radians(c_deg)
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def solve_c_from_b_and_xy(cal: Calibration, b: float, x: float, y: float) -> float:
    phi = math.atan2(float(y), float(x))
    alpha = transverse_phase_from_b(cal, b)
    return float(math.degrees(phi - alpha))


def eval_tip_angle_pitch_from_vertical_deg(
    cal: Calibration,
    b: ArrayLike,
    angle_sign: float = 1.0,
    angle_offset_deg: float = 0.0,
) -> np.ndarray:
    raw = poly_eval(cal.pa, b)
    return angle_sign * raw + angle_offset_deg


def tip_offset_xyz_physical(cal: Calibration, b: float, c_deg: float) -> np.ndarray:
    return predict_tip_xyz_from_bc(cal, b, c_deg)


def attack_theta_deg(c_deg: ArrayLike, attack_azimuth_flip_deg: float) -> np.ndarray:
    return np.asarray(c_deg, dtype=float) + float(attack_azimuth_flip_deg)


def nozzle_axis_unit_xyz_from_vertical_pitch(
    cal: Calibration,
    b: float,
    c_deg: float,
    angle_sign: float = 1.0,
    angle_offset_deg: float = 0.0,
    attack_azimuth_flip_deg: float = DEFAULT_ATTACK_AZIMUTH_FLIP_DEG,
) -> np.ndarray:
    a_deg = float(eval_tip_angle_pitch_from_vertical_deg(cal, b, angle_sign=angle_sign, angle_offset_deg=angle_offset_deg))
    a = math.radians(a_deg)
    theta = math.radians(float(c_deg) + float(attack_azimuth_flip_deg))
    v = np.array([math.sin(a) * math.cos(theta), math.sin(a) * math.sin(theta), math.cos(a)], dtype=float)
    n = np.linalg.norm(v)
    if n <= 0.0:
        raise RuntimeError("Invalid zero-length nozzle axis vector.")
    return v / n


def sampled_ranges(
    cal: Calibration,
    b_lo: float,
    b_hi: float,
    angle_sign: float,
    angle_offset_deg: float,
    n: int = 2001,
) -> dict:
    bb = np.linspace(b_lo, b_hi, n)
    rr = eval_r(cal, bb)
    yy = eval_offplane_y(cal, bb)
    zz = eval_z(cal, bb)
    aa_raw = poly_eval(cal.pa, bb)
    aa_eff = eval_tip_angle_pitch_from_vertical_deg(cal, bb, angle_sign=angle_sign, angle_offset_deg=angle_offset_deg)
    return {
        "r_min": float(np.min(rr)),
        "r_max": float(np.max(rr)),
        "abs_r_min": float(np.min(np.abs(rr))),
        "abs_r_max": float(np.max(np.abs(rr))),
        "yoff_min": float(np.min(yy)),
        "yoff_max": float(np.max(yy)),
        "rho_min": float(np.min(np.hypot(rr, yy))),
        "rho_max": float(np.max(np.hypot(rr, yy))),
        "alpha_min_deg": float(np.min(np.rad2deg(np.arctan2(yy, rr)))),
        "alpha_max_deg": float(np.max(np.rad2deg(np.arctan2(yy, rr)))),
        "z_min": float(np.min(zz)),
        "z_max": float(np.max(zz)),
        "a_raw_min": float(np.min(aa_raw)),
        "a_raw_max": float(np.max(aa_raw)),
        "a_eff_min": float(np.min(aa_eff)),
        "a_eff_max": float(np.max(aa_eff)),
    }


# ---------------- Syringe / tube extrusion math helpers ----------------

def tube_area_mm2_from_id_inch(id_inch: float) -> float:
    d_mm = float(id_inch) * 25.4
    r_mm = 0.5 * d_mm
    return math.pi * r_mm * r_mm


def extrusion_math_summary(
    syringe_mm_per_ml: float,
    tube_id_inch: float,
    print_feed_mm_min: float,
    extrusion_per_mm_u: float,
    bead_area_mm2: Optional[float] = None,
    bead_diameter_mm: Optional[float] = None,
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

    bead_area_used = None
    if bead_area_mm2 is not None:
        bead_area_used = float(bead_area_mm2)
    elif bead_diameter_mm is not None:
        d = float(bead_diameter_mm)
        bead_area_used = math.pi * (0.5 * d) ** 2

    recommended_extrusion_per_mm = None
    if bead_area_used is not None:
        recommended_extrusion_per_mm = float(syringe_mm_per_ml) * bead_area_used / 1000.0

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
        "bead_area_used_mm2": bead_area_used,
        "recommended_extrusion_per_mm": recommended_extrusion_per_mm,
    }


# ---------------- Geometry: octet lattice in cube ----------------

def _sorted_edge_key(a, b):
    return (a, b) if a <= b else (b, a)


def _point_key_from_xyz(p: np.ndarray, tol_decimals: int = 9) -> Tuple[float, float, float]:
    return (
        round(float(p[0]), tol_decimals),
        round(float(p[1]), tol_decimals),
        round(float(p[2]), tol_decimals),
    )


def build_octet_lattice_cube_edges(
    order: int,
    pitch_mm: float,
    origin_xyz: np.ndarray,
    include_cube_frame: bool = True,
    tol_decimals: int = 9,
) -> Tuple[np.ndarray, List[Tuple[int, int]], dict]:
    """
    Build an octet truss lattice inside a cube using the nearest-neighbor graph of an FCC lattice.

    FCC representation:
      integer index nodes (i,j,k) with even parity, 0 <= i,j,k <= 2*order
      real position = origin + (a/2) * [i,j,k]
    where:
      a = cube cell edge length = pitch * sqrt(2)
      nearest-neighbor strut length = a / sqrt(2) = pitch

    Added optional outer cube frame:
      boundary cube edges subdivided into `order` segments, i.e. points differing by 2 in one
      index coordinate along the cube boundary. These frame segments have length a.
    """
    if order < 1:
        raise ValueError("order must be >= 1.")
    if pitch_mm <= 0.0:
        raise ValueError("pitch_mm must be > 0.")

    n = int(order)
    pitch = float(pitch_mm)
    cell_edge = pitch * math.sqrt(2.0)
    half_cell = 0.5 * cell_edge
    max_idx = 2 * n

    idx_nodes: List[Tuple[int, int, int]] = []
    idx_to_node_idx: Dict[Tuple[int, int, int], int] = {}
    nodes_xyz_list: List[np.ndarray] = []

    for i in range(max_idx + 1):
        for j in range(max_idx + 1):
            for k in range(max_idx + 1):
                if (i + j + k) % 2 != 0:
                    continue
                idx = (i, j, k)
                pos = origin_xyz + half_cell * np.array([i, j, k], dtype=float)
                idx_to_node_idx[idx] = len(idx_nodes)
                idx_nodes.append(idx)
                nodes_xyz_list.append(pos)

    nodes_xyz = np.vstack(nodes_xyz_list)

    # FCC nearest-neighbor directions in integer-index coordinates: permutations of (±1, ±1, 0)
    nn_steps = []
    for axis0 in range(3):
        for axis1 in range(axis0 + 1, 3):
            axis2 = 3 - axis0 - axis1
            for s0 in (-1, 1):
                for s1 in (-1, 1):
                    step = [0, 0, 0]
                    step[axis0] = s0
                    step[axis1] = s1
                    step[axis2] = 0
                    nn_steps.append(tuple(step))

    edge_keys: Set[Tuple[int, int]] = set()
    for idx in idx_nodes:
        ia = idx_to_node_idx[idx]
        for di, dj, dk in nn_steps:
            nbr = (idx[0] + di, idx[1] + dj, idx[2] + dk)
            ib = idx_to_node_idx.get(nbr)
            if ib is not None and ia != ib:
                edge_keys.add(_sorted_edge_key(ia, ib))

    if include_cube_frame:
        xyz_to_idx: Dict[Tuple[float, float, float], int] = {
            _point_key_from_xyz(p, tol_decimals=tol_decimals): i for i, p in enumerate(nodes_xyz)
        }

        def add_frame_edge(pa: np.ndarray, pb: np.ndarray):
            ka = _point_key_from_xyz(pa, tol_decimals=tol_decimals)
            kb = _point_key_from_xyz(pb, tol_decimals=tol_decimals)
            ia = xyz_to_idx.get(ka)
            ib = xyz_to_idx.get(kb)
            if ia is None or ib is None or ia == ib:
                return
            edge_keys.add(_sorted_edge_key(ia, ib))

        # 12 cube edges, each subdivided into n segments.
        coord_vals = [0, max_idx]
        for y in coord_vals:
            for z in coord_vals:
                for i0 in range(0, max_idx, 2):
                    pa = origin_xyz + half_cell * np.array([i0, y, z], dtype=float)
                    pb = origin_xyz + half_cell * np.array([i0 + 2, y, z], dtype=float)
                    add_frame_edge(pa, pb)
        for x in coord_vals:
            for z in coord_vals:
                for j0 in range(0, max_idx, 2):
                    pa = origin_xyz + half_cell * np.array([x, j0, z], dtype=float)
                    pb = origin_xyz + half_cell * np.array([x, j0 + 2, z], dtype=float)
                    add_frame_edge(pa, pb)
        for x in coord_vals:
            for y in coord_vals:
                for k0 in range(0, max_idx, 2):
                    pa = origin_xyz + half_cell * np.array([x, y, k0], dtype=float)
                    pb = origin_xyz + half_cell * np.array([x, y, k0 + 2], dtype=float)
                    add_frame_edge(pa, pb)

    meta = {
        "cell_edge_mm": float(cell_edge),
        "cube_edge_mm": float(n * cell_edge),
        "fcc_half_step_mm": float(half_cell),
        "n_idx_max": int(max_idx),
        "include_cube_frame": bool(include_cube_frame),
    }
    return nodes_xyz, sorted(edge_keys), meta


# ---------------- Tangent solve / edge planning ----------------

def solve_directed_edge_tangent_soft(
    p0_tip: np.ndarray,
    p1_tip: np.ndarray,
    cal: Calibration,
    b_lo: float,
    b_hi: float,
    b_search_samples: int,
    prev_b: Optional[float],
    prev_c: Optional[float],
    angle_sign: float,
    angle_offset_deg: float,
    b_cont_weight: float,
    c_cont_weight: float,
    min_head_lead_mm: float,
    attack_azimuth_flip_deg: float,
    c_window_min: float,
    c_window_max: float,
) -> EdgePlan:
    v = p1_tip - p0_tip
    L = float(np.linalg.norm(v))
    if L <= 1e-12:
        raise ValueError("Zero-length edge encountered.")
    t = v / L

    horiz = float(math.hypot(float(t[0]), float(t[1])))
    has_horiz = horiz > 1e-12
    phi_deg = math.degrees(math.atan2(float(t[1]), float(t[0]))) if has_horiz else 0.0

    n_samples = max(11, int(b_search_samples))
    bb = np.linspace(float(b_lo), float(b_hi), n_samples)

    a_deg = eval_tip_angle_pitch_from_vertical_deg(
        cal, bb, angle_sign=angle_sign, angle_offset_deg=angle_offset_deg
    )
    a_rad = np.deg2rad(a_deg)
    sin_a = np.sin(a_rad)
    cos_a = np.cos(a_rad)

    if has_horiz:
        theta_attack_base = np.where(sin_a >= 0.0, phi_deg, phi_deg + 180.0)
        c_base = theta_attack_base - float(attack_azimuth_flip_deg)
    else:
        c0 = 0.0 if prev_c is None else float(prev_c)
        c_base = np.full_like(bb, c0, dtype=float)

    c_cmd = np.array([
        choose_equivalent_c_in_window(
            target_c_deg=float(cb),
            ref_c_deg=prev_c,
            c_min_deg=float(c_window_min),
            c_max_deg=float(c_window_max),
        )
        for cb in np.asarray(c_base, dtype=float)
    ], dtype=float)

    theta_attack_deg = attack_theta_deg(c_cmd, attack_azimuth_flip_deg=float(attack_azimuth_flip_deg))
    theta_attack_rad = np.deg2rad(theta_attack_deg)

    nx = sin_a * np.cos(theta_attack_rad)
    ny = sin_a * np.sin(theta_attack_rad)
    nz = cos_a

    dots = nx * t[0] + ny * t[1] + nz * t[2]
    dots = np.clip(dots, -1.0, 1.0)
    angle_err_deg = np.rad2deg(np.arccos(dots))

    cost = angle_err_deg.astype(float).copy()
    if prev_b is not None:
        cost += float(b_cont_weight) * np.abs(bb - float(prev_b))
    if prev_c is not None:
        cost += float(c_cont_weight) * np.abs(c_cmd - float(prev_c))

    i = int(np.argmin(cost))

    b_star = float(bb[i])
    c_star = float(c_cmd[i])
    err_star = float(angle_err_deg[i])

    nozzle_dir = np.array([float(nx[i]), float(ny[i]), float(nz[i])], dtype=float)
    nozzle_dir /= np.linalg.norm(nozzle_dir)

    offset_vec = tip_offset_xyz_physical(cal, b_star, c_star)
    head_lead = float(np.dot(-offset_vec, t))
    head_front_ok = bool(head_lead >= float(min_head_lead_mm))

    p0_stage = p0_tip - offset_vec
    p1_stage = p1_tip - offset_vec
    if np.linalg.norm((p0_stage + offset_vec) - p0_tip) > 1e-9:
        raise RuntimeError("Tip-tracking consistency failed at edge start.")
    if np.linalg.norm((p1_stage + offset_vec) - p1_tip) > 1e-9:
        raise RuntimeError("Tip-tracking consistency failed at edge end.")

    return EdgePlan(
        p0_tip=p0_tip.copy(),
        p1_tip=p1_tip.copy(),
        b_cmd=b_star,
        c_cmd=c_star,
        angle_error_deg=err_star,
        nozzle_dir=nozzle_dir,
        offset_vec=offset_vec,
        edge_len=L,
        head_lead_mm=head_lead,
        head_front_ok=head_front_ok,
    )


def choose_oriented_edge_plan(
    pa_tip: np.ndarray,
    pb_tip: np.ndarray,
    cal: Calibration,
    b_lo: float,
    b_hi: float,
    b_search_samples: int,
    prev_b: Optional[float],
    prev_c: Optional[float],
    current_tip: Optional[np.ndarray],
    angle_sign: float,
    angle_offset_deg: float,
    b_cont_weight: float,
    c_cont_weight: float,
    min_head_lead_mm: float,
    attack_azimuth_flip_deg: float,
    c_window_min: float,
    c_window_max: float,
    horizontal_z_tol_mm: float = 1e-6,
) -> Tuple[EdgePlan, float]:
    plan_ab = solve_directed_edge_tangent_soft(
        p0_tip=pa_tip, p1_tip=pb_tip,
        cal=cal, b_lo=b_lo, b_hi=b_hi,
        b_search_samples=b_search_samples,
        prev_b=prev_b, prev_c=prev_c,
        angle_sign=angle_sign, angle_offset_deg=angle_offset_deg,
        b_cont_weight=b_cont_weight, c_cont_weight=c_cont_weight,
        min_head_lead_mm=min_head_lead_mm,
        attack_azimuth_flip_deg=attack_azimuth_flip_deg,
        c_window_min=c_window_min, c_window_max=c_window_max,
    )
    plan_ba = solve_directed_edge_tangent_soft(
        p0_tip=pb_tip, p1_tip=pa_tip,
        cal=cal, b_lo=b_lo, b_hi=b_hi,
        b_search_samples=b_search_samples,
        prev_b=prev_b, prev_c=prev_c,
        angle_sign=angle_sign, angle_offset_deg=angle_offset_deg,
        b_cont_weight=b_cont_weight, c_cont_weight=c_cont_weight,
        min_head_lead_mm=min_head_lead_mm,
        attack_azimuth_flip_deg=attack_azimuth_flip_deg,
        c_window_min=c_window_min, c_window_max=c_window_max,
    )

    travel_ab = 0.0 if current_tip is None else float(np.linalg.norm(plan_ab.p0_tip - current_tip))
    travel_ba = 0.0 if current_tip is None else float(np.linalg.norm(plan_ba.p0_tip - current_tip))

    dz_ab = float(plan_ab.p1_tip[2] - plan_ab.p0_tip[2])
    dz_ba = float(plan_ba.p1_tip[2] - plan_ba.p0_tip[2])
    is_horizontal = abs(dz_ab) <= float(horizontal_z_tol_mm)

    if not is_horizontal:
        ab_bottom_up = dz_ab > 0.0
        ba_bottom_up = dz_ba > 0.0
        if ab_bottom_up and not ba_bottom_up:
            return plan_ab, travel_ab
        if ba_bottom_up and not ab_bottom_up:
            return plan_ba, travel_ba

    def score(plan: EdgePlan, travel: float) -> Tuple[int, float, float, float]:
        front_rank = 0 if plan.head_front_ok else 1
        neg_head_lead = -float(plan.head_lead_mm)
        return (front_rank, neg_head_lead, float(travel), float(plan.angle_error_deg))

    if score(plan_ab, travel_ab) <= score(plan_ba, travel_ba):
        return plan_ab, travel_ab
    return plan_ba, travel_ba


def compute_bottom_center(nodes_xyz: np.ndarray) -> np.ndarray:
    zmin = float(np.min(nodes_xyz[:, 2]))
    bottom_mask = np.isclose(nodes_xyz[:, 2], zmin, atol=1e-9)
    bottom_nodes = nodes_xyz[bottom_mask]
    if len(bottom_nodes) == 0:
        raise RuntimeError("Failed to identify bottom nodes.")
    xy_center = np.mean(bottom_nodes[:, :2], axis=0)
    return np.array([float(xy_center[0]), float(xy_center[1]), zmin], dtype=float)


def _same_point(a: np.ndarray, b: np.ndarray, tol_mm: float = 1e-6) -> bool:
    return float(np.linalg.norm(a - b)) <= float(tol_mm)


def _edge_is_horizontal_points(p0: np.ndarray, p1: np.ndarray, z_tol_mm: float = 1e-6) -> bool:
    return abs(float(p1[2] - p0[2])) <= float(z_tol_mm)


def _plan_is_horizontal(plan: EdgePlan, z_tol_mm: float = 1e-6) -> bool:
    return _edge_is_horizontal_points(plan.p0_tip, plan.p1_tip, z_tol_mm=z_tol_mm)


def _continuous_horizontal_tip_pivot_transition(
    prev_plan: EdgePlan,
    next_plan: EdgePlan,
    tip_tol_mm: float = 1e-6,
    z_tol_mm: float = 1e-6,
) -> bool:
    return (
        _plan_is_horizontal(prev_plan, z_tol_mm=z_tol_mm)
        and _plan_is_horizontal(next_plan, z_tol_mm=z_tol_mm)
        and _same_point(prev_plan.p1_tip, next_plan.p0_tip, tol_mm=tip_tol_mm)
    )


def plan_all_edges_bottom_center_first(
    nodes_xyz: np.ndarray,
    edges_idx: List[Tuple[int, int]],
    cal: Calibration,
    b_lo: float,
    b_hi: float,
    b_search_samples: int,
    angle_sign: float,
    angle_offset_deg: float,
    b_cont_weight: float,
    c_cont_weight: float,
    min_head_lead_mm: float,
    attack_azimuth_flip_deg: float,
    c_window_min: float,
    c_window_max: float,
    horizontal_z_tol_mm: float = 1e-6,
    vertex_match_tol_mm: float = 1e-6,
) -> Tuple[List[EdgePlan], dict]:
    if len(edges_idx) == 0:
        return [], {
            "n_edges": 0,
            "total_tip_travel_mm": 0.0,
            "total_print_length_mm": 0.0,
            "angle_error_min_deg": 0.0,
            "angle_error_mean_deg": 0.0,
            "angle_error_max_deg": 0.0,
            "b_min_used": 0.0,
            "b_max_used": 0.0,
            "c_min_used": 0.0,
            "c_max_used": 0.0,
            "head_front_ok_count": 0,
            "head_front_violation_count": 0,
            "head_lead_min_mm": 0.0,
            "head_lead_mean_mm": 0.0,
            "head_lead_max_mm": 0.0,
        }

    bottom_center = compute_bottom_center(nodes_xyz)
    current_tip = bottom_center.copy()
    prev_b: Optional[float] = None
    prev_c: Optional[float] = None

    remaining = set(range(len(edges_idx)))
    plans: List[EdgePlan] = []

    total_tip_travel = 0.0
    total_print_len = 0.0
    z_global_min = float(np.min(nodes_xyz[:, 2]))

    while remaining:
        current_layer_z = float(current_tip[2])
        has_lower_diagonal_remaining = False
        for ei in remaining:
            ia, ib = edges_idx[ei]
            pa = nodes_xyz[ia]
            pb = nodes_xyz[ib]
            if _edge_is_horizontal_points(pa, pb, z_tol_mm=horizontal_z_tol_mm):
                continue
            if float(min(pa[2], pb[2])) < (current_layer_z - float(horizontal_z_tol_mm)):
                has_lower_diagonal_remaining = True
                break

        if not has_lower_diagonal_remaining:
            horiz_best_key = None
            horiz_best_plan: Optional[EdgePlan] = None
            horiz_best_ei = None

            for ei in remaining:
                ia, ib = edges_idx[ei]
                pa = nodes_xyz[ia]
                pb = nodes_xyz[ib]

                if not _edge_is_horizontal_points(pa, pb, z_tol_mm=horizontal_z_tol_mm):
                    continue

                if _same_point(pa, current_tip, tol_mm=vertex_match_tol_mm):
                    cand = solve_directed_edge_tangent_soft(
                        p0_tip=pa, p1_tip=pb,
                        cal=cal, b_lo=b_lo, b_hi=b_hi,
                        b_search_samples=b_search_samples,
                        prev_b=prev_b, prev_c=prev_c,
                        angle_sign=angle_sign, angle_offset_deg=angle_offset_deg,
                        b_cont_weight=b_cont_weight, c_cont_weight=c_cont_weight,
                        min_head_lead_mm=min_head_lead_mm,
                        attack_azimuth_flip_deg=attack_azimuth_flip_deg,
                        c_window_min=c_window_min, c_window_max=c_window_max,
                    )
                elif _same_point(pb, current_tip, tol_mm=vertex_match_tol_mm):
                    cand = solve_directed_edge_tangent_soft(
                        p0_tip=pb, p1_tip=pa,
                        cal=cal, b_lo=b_lo, b_hi=b_hi,
                        b_search_samples=b_search_samples,
                        prev_b=prev_b, prev_c=prev_c,
                        angle_sign=angle_sign, angle_offset_deg=angle_offset_deg,
                        b_cont_weight=b_cont_weight, c_cont_weight=c_cont_weight,
                        min_head_lead_mm=min_head_lead_mm,
                        attack_azimuth_flip_deg=attack_azimuth_flip_deg,
                        c_window_min=c_window_min, c_window_max=c_window_max,
                    )
                else:
                    continue

                key = (
                    0 if cand.head_front_ok else 1,
                    -float(cand.head_lead_mm),
                    float(cand.angle_error_deg),
                )

                if horiz_best_key is None or key < horiz_best_key:
                    horiz_best_key = key
                    horiz_best_plan = cand
                    horiz_best_ei = ei

            if horiz_best_plan is not None and horiz_best_ei is not None:
                plans.append(horiz_best_plan)
                total_tip_travel += 0.0
                total_print_len += float(horiz_best_plan.edge_len)
                current_tip = horiz_best_plan.p1_tip.copy()
                prev_b = horiz_best_plan.b_cmd
                prev_c = horiz_best_plan.c_cmd
                remaining.remove(horiz_best_ei)
                continue

        best_key = None
        best_plan: Optional[EdgePlan] = None
        best_travel = 0.0
        best_ei = None

        for ei in remaining:
            ia, ib = edges_idx[ei]
            pa = nodes_xyz[ia]
            pb = nodes_xyz[ib]

            plan, travel_to_start = choose_oriented_edge_plan(
                pa_tip=pa, pb_tip=pb,
                cal=cal,
                b_lo=b_lo, b_hi=b_hi,
                b_search_samples=b_search_samples,
                prev_b=prev_b, prev_c=prev_c,
                current_tip=current_tip,
                angle_sign=angle_sign, angle_offset_deg=angle_offset_deg,
                b_cont_weight=b_cont_weight, c_cont_weight=c_cont_weight,
                min_head_lead_mm=min_head_lead_mm,
                attack_azimuth_flip_deg=attack_azimuth_flip_deg,
                c_window_min=c_window_min, c_window_max=c_window_max,
            )

            z_min_edge = float(min(plan.p0_tip[2], plan.p1_tip[2]))
            z_max_edge = float(max(plan.p0_tip[2], plan.p1_tip[2]))
            mid = 0.5 * (plan.p0_tip + plan.p1_tip)
            radial_from_bottom_center = float(np.linalg.norm(mid[:2] - bottom_center[:2]))
            z_start = float(plan.p0_tip[2])
            z_dir_bias = max(0.0, z_start - z_min_edge)

            key = (
                z_min_edge - z_global_min,
                z_max_edge - z_global_min,
                radial_from_bottom_center,
                0 if plan.head_front_ok else 1,
                -plan.head_lead_mm,
                float(travel_to_start),
                float(plan.angle_error_deg),
                z_dir_bias,
            )

            if best_key is None or key < best_key:
                best_key = key
                best_plan = plan
                best_travel = travel_to_start
                best_ei = ei

        assert best_plan is not None and best_ei is not None

        plans.append(best_plan)
        total_tip_travel += float(best_travel)
        total_print_len += float(best_plan.edge_len)

        current_tip = best_plan.p1_tip.copy()
        prev_b = best_plan.b_cmd
        prev_c = best_plan.c_cmd

        remaining.remove(best_ei)

    angle_errs = np.array([p.angle_error_deg for p in plans], dtype=float)
    bs = np.array([p.b_cmd for p in plans], dtype=float)
    cs = np.array([p.c_cmd for p in plans], dtype=float)
    leads = np.array([p.head_lead_mm for p in plans], dtype=float)
    head_front_ok_count = int(sum(1 for p in plans if p.head_front_ok))

    meta = {
        "n_edges": len(plans),
        "total_tip_travel_mm": float(total_tip_travel),
        "total_print_length_mm": float(total_print_len),
        "angle_error_min_deg": float(np.min(angle_errs)),
        "angle_error_mean_deg": float(np.mean(angle_errs)),
        "angle_error_max_deg": float(np.max(angle_errs)),
        "b_min_used": float(np.min(bs)),
        "b_max_used": float(np.max(bs)),
        "c_min_used": float(np.min(cs)),
        "c_max_used": float(np.max(cs)),
        "head_front_ok_count": int(head_front_ok_count),
        "head_front_violation_count": int(len(plans) - head_front_ok_count),
        "head_lead_min_mm": float(np.min(leads)),
        "head_lead_mean_mm": float(np.mean(leads)),
        "head_lead_max_mm": float(np.max(leads)),
        "bottom_center_x": float(bottom_center[0]),
        "bottom_center_y": float(bottom_center[1]),
        "bottom_center_z": float(bottom_center[2]),
    }
    return plans, meta


# ---------------- Stage (gantry) range diagnostics ----------------

def compute_stage_gantry_ranges(plans: List[EdgePlan]) -> dict:
    if not plans:
        return {
            "x_stage_min": 0.0, "x_stage_max": 0.0,
            "y_stage_min": 0.0, "y_stage_max": 0.0,
            "z_stage_min": 0.0, "z_stage_max": 0.0,
        }

    pts = []
    for p in plans:
        d = p.offset_vec
        pts.append(p.p0_tip - d)
        pts.append(p.p1_tip - d)

    arr = np.vstack(pts)
    return {
        "x_stage_min": float(np.min(arr[:, 0])),
        "x_stage_max": float(np.max(arr[:, 0])),
        "y_stage_min": float(np.min(arr[:, 1])),
        "y_stage_max": float(np.max(arr[:, 1])),
        "z_stage_min": float(np.min(arr[:, 2])),
        "z_stage_max": float(np.max(arr[:, 2])),
    }


# ---------------- G-code emission ----------------

def _fmt_axes_move(axes_vals: List[Tuple[str, float]]) -> str:
    return " ".join(f"{ax}{val:.3f}" for ax, val in axes_vals)


def _clamp_stage_xyz_to_bbox(
    x: float,
    y: float,
    z: float,
    bbox: dict,
    context: str,
    warn_log: List[str],
) -> Tuple[float, float, float]:
    def clamp_one(axis: str, value: float, lo: float, hi: float) -> float:
        if value < lo:
            warn_log.append(
                f"WARNING: {context} {axis}={value:.3f} below bbox min {lo:.3f}; clamped to {lo:.3f}"
            )
            return lo
        if value > hi:
            warn_log.append(
                f"WARNING: {context} {axis}={value:.3f} above bbox max {hi:.3f}; clamped to {hi:.3f}"
            )
            return hi
        return value

    xc = clamp_one("X", float(x), float(bbox["x_min"]), float(bbox["x_max"]))
    yc = clamp_one("Y", float(y), float(bbox["y_min"]), float(bbox["y_max"]))
    zc = clamp_one("Z", float(z), float(bbox["z_min"]), float(bbox["z_max"]))
    return xc, yc, zc




def find_b_for_pitch_target(
    cal: Calibration,
    b_lo: float,
    b_hi: float,
    target_pitch_deg: float,
    angle_sign: float,
    angle_offset_deg: float,
    samples: int = 4001,
) -> Tuple[float, float]:
    bb = np.linspace(float(b_lo), float(b_hi), max(51, int(samples)))
    aa = eval_tip_angle_pitch_from_vertical_deg(
        cal, bb, angle_sign=float(angle_sign), angle_offset_deg=float(angle_offset_deg)
    )
    i = int(np.argmin(np.abs(aa - float(target_pitch_deg))))
    return float(bb[i]), float(aa[i])


def _segment_segment_distance_3d(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> float:
    """Minimum distance between two 3D line segments."""
    u = q1 - p1
    v = q2 - p2
    w = p1 - p2
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    D = a * c - b * b
    SMALL = 1e-12

    sN = 0.0
    sD = D
    tN = 0.0
    tD = D

    if D < SMALL:
        sN = 0.0
        sD = 1.0
        tN = e
        tD = c
    else:
        sN = (b * e - c * d)
        tN = (a * e - b * d)
        if sN < 0.0:
            sN = 0.0
            tN = e
            tD = c
        elif sN > sD:
            sN = sD
            tN = e + b
            tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0:
            sN = 0.0
        elif -d > a:
            sN = sD
        else:
            sN = -d
            sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0:
            sN = 0.0
        elif (-d + b) > a:
            sN = sD
        else:
            sN = (-d + b)
            sD = a

    sc = 0.0 if abs(sN) < SMALL else sN / sD
    tc = 0.0 if abs(tN) < SMALL else tN / tD
    dP = w + sc * u - tc * v
    return float(np.linalg.norm(dP))


def _point_matches_any(p: np.ndarray, refs: List[np.ndarray], tol_mm: float = 1e-6) -> bool:
    return any(float(np.linalg.norm(p - r)) <= float(tol_mm) for r in refs)


def _segment_clear_of_written(
    a: np.ndarray,
    b: np.ndarray,
    written_segments: List[Tuple[np.ndarray, np.ndarray]],
    clearance_mm: float,
    ignore_points: Optional[List[np.ndarray]] = None,
    endpoint_tol_mm: float = 1e-6,
) -> bool:
    if ignore_points is None:
        ignore_points = []
    if float(np.linalg.norm(b - a)) <= 1e-12:
        return True

    for ws, we in written_segments:
        if _point_matches_any(ws, ignore_points, tol_mm=endpoint_tol_mm) or _point_matches_any(we, ignore_points, tol_mm=endpoint_tol_mm):
            continue
        if _segment_segment_distance_3d(a, b, ws, we) <= float(clearance_mm) + float(endpoint_tol_mm):
            return False
    return True


def _polyline_clear_of_written(
    waypoints: List[np.ndarray],
    written_segments: List[Tuple[np.ndarray, np.ndarray]],
    clearance_mm: float,
    endpoint_tol_mm: float = 1e-6,
) -> bool:
    if len(waypoints) < 2:
        return True

    route_start = waypoints[0]
    route_end = waypoints[-1]
    for i in range(len(waypoints) - 1):
        a = waypoints[i]
        b = waypoints[i + 1]
        ignore_points: List[np.ndarray] = []
        if float(np.linalg.norm(a - route_start)) <= float(endpoint_tol_mm):
            ignore_points.append(route_start)
        if float(np.linalg.norm(b - route_end)) <= float(endpoint_tol_mm):
            ignore_points.append(route_end)
        if not _segment_clear_of_written(
            a, b,
            written_segments=written_segments,
            clearance_mm=clearance_mm,
            ignore_points=ignore_points,
            endpoint_tol_mm=endpoint_tol_mm,
        ):
            return False
    return True


def plan_tip_safe_travel_waypoints(
    current_tip: np.ndarray,
    target_tip: np.ndarray,
    written_segments: List[Tuple[np.ndarray, np.ndarray]],
    travel_z_clearance_mm: float,
    travel_line_clearance_mm: float,
    tol_mm: float = 1e-6,
) -> Tuple[List[np.ndarray], float, bool]:
    """
    Plan a conservative non-print tip path.

    Preference:
      1) move with final descent last (XY at z=max(current_z, target_z), then Z-down if needed)
      2) if that XY leg is blocked by previously written segments, lift above max written Z
         by travel_z_clearance_mm, then XY, then descend.
    """
    current_tip = np.asarray(current_tip, dtype=float)
    target_tip = np.asarray(target_tip, dtype=float)
    if float(np.linalg.norm(target_tip - current_tip)) <= float(tol_mm):
        return [current_tip.copy()], float(current_tip[2]), False

    max_written_z = max(
        [max(float(a[2]), float(b[2])) for a, b in written_segments],
        default=max(float(current_tip[2]), float(target_tip[2])),
    )
    base_plane_z = max(float(current_tip[2]), float(target_tip[2]))

    def build_waypoints(z_plane: float) -> List[np.ndarray]:
        pts = [current_tip.copy()]
        if z_plane > float(current_tip[2]) + float(tol_mm):
            pts.append(np.array([current_tip[0], current_tip[1], z_plane], dtype=float))
        xy_pt = np.array([target_tip[0], target_tip[1], z_plane], dtype=float)
        if float(np.linalg.norm(xy_pt - pts[-1])) > float(tol_mm):
            pts.append(xy_pt)
        if float(np.linalg.norm(target_tip - pts[-1])) > float(tol_mm):
            pts.append(target_tip.copy())
        return pts

    candidate_planes = [base_plane_z, max(base_plane_z, max_written_z + float(travel_z_clearance_mm))]
    for k, z_plane in enumerate(candidate_planes):
        way = build_waypoints(z_plane)
        if _polyline_clear_of_written(
            waypoints=way,
            written_segments=written_segments,
            clearance_mm=float(travel_line_clearance_mm),
            endpoint_tol_mm=tol_mm,
        ):
            return way, float(z_plane), bool(k > 0)

    z_plane = max(base_plane_z, max_written_z + float(travel_z_clearance_mm))
    return build_waypoints(z_plane), float(z_plane), True

def write_gcode_octet_truss(
    out_path: str,
    cal: Calibration,
    plans: List[EdgePlan],
    edge_samples: int,
    travel_samples: int,
    travel_feed: float,
    print_feed: float,
    extrusion_per_mm: float,
    prime_mm: float,
    emit_extrusion: bool,
    pressure_offset_mm: float,
    pressure_advance_feed: float,
    pressure_retract_feed: float,
    preflow_dwell_ms: int,
    node_dwell_ms: int,
    header_meta: dict,
    angle_sign: float,
    angle_offset_deg: float,
    attack_azimuth_flip_deg: float,
    neutral_b_cmd: float,
    neutral_pitch_deg: float,
    start_pose: Tuple[float, float, float, float, float],
    end_pose: Tuple[float, float, float, float, float],
    safe_approach_z: float,
    virtual_bbox: dict,
    include_cube_frame: bool,
    travel_z_clearance_mm: float,
    travel_line_clearance_mm: float,
    c_window_min: float,
    c_window_max: float,
    c_recenter_margin_deg: float,
):
    if edge_samples < 2:
        edge_samples = 2
    travel_samples = max(2, int(travel_samples))

    u_material_abs = 0.0
    pressure_charged = False
    bbox_warnings: List[str] = []
    written_segments: List[Tuple[np.ndarray, np.ndarray]] = []

    def u_cmd_actual() -> float:
        return u_material_abs + (float(pressure_offset_mm) if pressure_charged else 0.0)

    def emit_tip_tracked_segmented_move(
        fobj,
        p_tip_start: np.ndarray,
        p_tip_end: np.ndarray,
        b_start: float,
        b_end: float,
        c_start: float,
        c_end: float,
        samples: int,
        feed: float,
        comment: Optional[str] = None,
        context_prefix: str = 'travel',
    ) -> Tuple[np.ndarray, float, float]:
        if comment:
            fobj.write(comment.rstrip() + "\n")
        n = max(1, int(samples))
        tvals = np.linspace(0.0, 1.0, n + 1)
        last_stage = None
        for j in range(1, len(tvals)):
            tt = float(tvals[j])
            p_tip = (1.0 - tt) * p_tip_start + tt * p_tip_end
            b_cmd = (1.0 - tt) * float(b_start) + tt * float(b_end)
            c_cmd = (1.0 - tt) * float(c_start) + tt * float(c_end)
            stage_xyz = p_tip - tip_offset_xyz_physical(cal, b_cmd, c_cmd)
            mx, my, mz = _clamp_stage_xyz_to_bbox(
                x=float(stage_xyz[0]),
                y=float(stage_xyz[1]),
                z=float(stage_xyz[2]),
                bbox=virtual_bbox,
                context=f"{context_prefix} seg {j}",
                warn_log=bbox_warnings,
            )
            axes = [
                (cal.x_axis, mx),
                (cal.y_axis, my),
                (cal.z_axis, mz),
                (cal.b_axis, float(b_cmd)),
                (cal.c_axis, float(c_cmd)),
            ]
            fobj.write(f"G1 {_fmt_axes_move(axes)} F{float(feed):.0f}\n")
            last_stage = np.array([mx, my, mz], dtype=float)
        if last_stage is None:
            last_stage = np.array([0.0, 0.0, 0.0], dtype=float)
        return last_stage, float(b_end), float(c_end)

    def emit_tip_fixed_reorient_via_neutral(
        fobj,
        tip_point: np.ndarray,
        b_from: float,
        c_from: float,
        b_to: float,
        c_to: float,
        comment_prefix: str,
    ) -> Tuple[float, float]:
        cur_b = float(b_from)
        cur_c = float(c_from)
        target_c = float(c_to)
        if abs(cur_b - float(neutral_b_cmd)) > 1e-9:
            _, cur_b, cur_c = emit_tip_tracked_segmented_move(
                fobj, tip_point, tip_point, cur_b, float(neutral_b_cmd), cur_c, cur_c,
                samples=travel_samples,
                feed=float(travel_feed),
                comment=f"; {comment_prefix}: uncurl B toward neutral pitch",
                context_prefix=f"{comment_prefix} uncurl",
            )
        if abs(target_c - cur_c) > 1e-9:
            _, cur_b, cur_c = emit_tip_tracked_segmented_move(
                fobj, tip_point, tip_point, cur_b, cur_b, cur_c, target_c,
                samples=travel_samples,
                feed=float(travel_feed),
                comment=f"; {comment_prefix}: rotate C at neutral B while holding tip fixed",
                context_prefix=f"{comment_prefix} rotate",
            )
        if abs(float(b_to) - cur_b) > 1e-9:
            _, cur_b, cur_c = emit_tip_tracked_segmented_move(
                fobj, tip_point, tip_point, cur_b, float(b_to), cur_c, cur_c,
                samples=travel_samples,
                feed=float(travel_feed),
                comment=f"; {comment_prefix}: curl B from neutral to print angle while holding tip fixed",
                context_prefix=f"{comment_prefix} curl",
            )
        return cur_b, cur_c

    def emit_raise_tip_to_safe_recenter_z(
        fobj,
        tip_point: np.ndarray,
        cur_b: float,
        cur_c: float,
        comment_prefix: str,
    ) -> Tuple[np.ndarray, float, float]:
        safe_z = max(
            float(tip_point[2]),
            max_written_z_value(written_segments, default_z=float(tip_point[2])) + float(travel_z_clearance_mm),
        )
        if safe_z <= float(tip_point[2]) + 1e-9:
            return tip_point.copy(), float(cur_b), float(cur_c)
        safe_tip = np.array([float(tip_point[0]), float(tip_point[1]), float(safe_z)], dtype=float)
        _, cur_b, cur_c = emit_tip_tracked_segmented_move(
            fobj, tip_point, safe_tip, float(cur_b), float(cur_b), float(cur_c), float(cur_c),
            samples=travel_samples,
            feed=float(travel_feed),
            comment=f"; {comment_prefix}: raise tip above all written geometry before neutral-B C recenter",
            context_prefix=f"{comment_prefix} raise",
        )
        return safe_tip, float(cur_b), float(cur_c)

    def emit_tip_fixed_c_recenter_if_needed(
        fobj,
        tip_point: np.ndarray,
        cur_b: float,
        cur_c: float,
        comment_prefix: str,
    ) -> Tuple[np.ndarray, float, float]:
        if not c_needs_recentering(
            float(cur_c),
            c_min_deg=float(c_window_min),
            c_max_deg=float(c_window_max),
            margin_deg=float(c_recenter_margin_deg),
        ):
            return tip_point.copy(), float(cur_b), float(cur_c)

        safe_tip, cur_b, cur_c = emit_raise_tip_to_safe_recenter_z(
            fobj, tip_point, float(cur_b), float(cur_c), comment_prefix=comment_prefix,
        )
        b_restore = float(cur_b)
        desired_c = choose_equivalent_c_in_window(
            target_c_deg=float(cur_c),
            ref_c_deg=0.0,
            c_min_deg=float(c_window_min),
            c_max_deg=float(c_window_max),
        )
        if abs(float(cur_b) - float(neutral_b_cmd)) > 1e-9:
            _, cur_b, cur_c = emit_tip_tracked_segmented_move(
                fobj, safe_tip, safe_tip, float(cur_b), float(neutral_b_cmd), float(cur_c), float(cur_c),
                samples=travel_samples,
                feed=float(travel_feed),
                comment=f"; {comment_prefix}: uncurl B to neutral before bounded-C recenter",
                context_prefix=f"{comment_prefix} uncurl",
            )
        if abs(float(desired_c) - float(cur_c)) > 1e-9:
            _, cur_b, cur_c = emit_tip_tracked_segmented_move(
                fobj, safe_tip, safe_tip, float(cur_b), float(cur_b), float(cur_c), float(desired_c),
                samples=travel_samples,
                feed=float(travel_feed),
                comment=f"; {comment_prefix}: rotate C to equivalent bounded value at neutral B",
                context_prefix=f"{comment_prefix} rotate",
            )
        if abs(float(b_restore) - float(cur_b)) > 1e-9:
            _, cur_b, cur_c = emit_tip_tracked_segmented_move(
                fobj, safe_tip, safe_tip, float(cur_b), float(b_restore), float(cur_c), float(cur_c),
                samples=travel_samples,
                feed=float(travel_feed),
                comment=f"; {comment_prefix}: restore B after bounded-C recenter",
                context_prefix=f"{comment_prefix} restore",
            )
        return safe_tip, float(cur_b), float(cur_c)

    with open(out_path, 'w') as f:
        f.write('; generated by octet_truss_cube_tip_priority_tangent_soft.py\n')
        f.write('; strategy: exact tip position (hard) + tangent/nozzle angle alignment (soft)\n')
        f.write('; user tip-angle convention: angle measured FROM VERTICAL (horizontal line ~90 deg)\n')
        f.write('; head-front preference: choose edge direction so head/stage leads deposition point when possible\n')
        f.write('; build order heuristic: bottom-center-first, bottom-up, center-out\n')
        f.write('; geometry: octet truss lattice in a cube\n')
        f.write(f'; cube frame included: {int(bool(include_cube_frame))}\n')
        f.write('; PHYSICAL tip kinematics uses local [r(B), y_off(B)] rotated by C; attack azimuth remap is tangency-only\n')
        f.write('; non-print travel policy: neutral-B travel, rotate C during XY travel, final Z descent last\n')
        f.write('; if XY travel would cross a previously written strut, lift above max written Z before XY travel\n')
        f.write('; same-tip turns are executed as: B->neutral, C rotate, B->target, while holding tip fixed when reorientation is actually needed\n')
        f.write('; bounded C policy: choose equivalent C inside a configurable window and recenter only at safe high-Z neutral-B points\n')
        f.write('; model assumptions:\n')
        f.write(';   physical tip_offset(B,C): [x,y]=Rot(C)@[r(B), y_off(B)], z=z(B)\n')
        f.write(';   nozzle attack dir(B,C) = [sin(a(B)) cos(theta_attack), sin(a(B)) sin(theta_attack), cos(a(B))]\n')
        f.write(f';   theta_attack = C + {float(attack_azimuth_flip_deg):+.3f} deg\n')
        f.write(f';   a(B) = ({angle_sign:+.3f}) * tip_angle_poly(B) + {angle_offset_deg:+.3f} deg\n')
        f.write(f'; neutral travel pitch target = {float(neutral_pitch_deg):+.3f} deg  -> B={float(neutral_b_cmd):.6f}\n')
        f.write(f'; travel samples = {int(travel_samples)}  line clearance = {float(travel_line_clearance_mm):.3f} mm  z clearance = {float(travel_z_clearance_mm):.3f} mm\n')
        f.write(f'; bounded C window = [{float(c_window_min):+.3f}, {float(c_window_max):+.3f}]  recenter margin = {float(c_recenter_margin_deg):.3f} deg\n')
        if cal.offplane_y_equation:
            f.write(f'; offplane_y_equation (calibration): {cal.offplane_y_equation}\n')
        if cal.offplane_y_r_squared is not None:
            f.write(f'; offplane_y_r_squared: {cal.offplane_y_r_squared:.6f}\n')
        f.write(f'; offplane sign convention applied: {OFFPLANE_SIGN:+.1f}\n')
        f.write(f'; axes: X->{cal.x_axis}, Y->{cal.y_axis}, Z->{cal.z_axis}, B->{cal.b_axis}, C->{cal.c_axis}, U->{cal.u_axis}\n')
        f.write(f'; edges planned = {header_meta.get("n_edges", 0)} (all edges extruded exactly once)\n')
        f.write(f'; total print length ~ {header_meta.get("total_print_length_mm", 0.0):.3f} mm\n')
        f.write(f'; total tip travel  ~ {header_meta.get("total_tip_travel_mm", 0.0):.3f} mm\n')
        f.write(f'; tangent angle error [min/mean/max] = {header_meta.get("angle_error_min_deg", 0.0):.3f} / {header_meta.get("angle_error_mean_deg", 0.0):.3f} / {header_meta.get("angle_error_max_deg", 0.0):.3f} deg\n')
        f.write(f'; head-lead [min/mean/max] mm = {header_meta.get("head_lead_min_mm", 0.0):.3f} / {header_meta.get("head_lead_mean_mm", 0.0):.3f} / {header_meta.get("head_lead_max_mm", 0.0):.3f}\n')
        f.write(f'; head-front ok / violations = {header_meta.get("head_front_ok_count", 0)} / {header_meta.get("head_front_violation_count", 0)}\n')
        f.write(f'; B used range = [{header_meta.get("b_min_used", 0.0):.3f}, {header_meta.get("b_max_used", 0.0):.3f}]\n')
        f.write(f'; C used range = [{header_meta.get("c_min_used", 0.0):.3f}, {header_meta.get("c_max_used", 0.0):.3f}]\n')
        f.write(f'; X gantry range used = [{header_meta.get("x_stage_min", 0.0):.3f}, {header_meta.get("x_stage_max", 0.0):.3f}]\n')
        f.write(f'; Y gantry range used = [{header_meta.get("y_stage_min", 0.0):.3f}, {header_meta.get("y_stage_max", 0.0):.3f}]\n')
        f.write(f'; Z gantry range used = [{header_meta.get("z_stage_min", 0.0):.3f}, {header_meta.get("z_stage_max", 0.0):.3f}]\n')
        f.write(f'; octet cell edge length = {header_meta.get("cell_edge_mm", 0.0):.3f} mm\n')
        f.write(f'; cube edge length = {header_meta.get("cube_edge_mm", 0.0):.3f} mm\n')
        f.write(f'; virtual bbox (enforced during print/travel): X[{float(virtual_bbox["x_min"]):.3f}, {float(virtual_bbox["x_max"]):.3f}] Y[{float(virtual_bbox["y_min"]):.3f}, {float(virtual_bbox["y_max"]):.3f}] Z[{float(virtual_bbox["z_min"]):.3f}, {float(virtual_bbox["z_max"]):.3f}]\n')
        f.write(f'; startup safe approach Z = {float(safe_approach_z):.3f}\n')
        f.write(f'; start pose (stage) = [{start_pose[0]:.3f}, {start_pose[1]:.3f}, {start_pose[2]:.3f}, {start_pose[3]:.3f}, {start_pose[4]:.3f}]\n')
        f.write(f'; end pose (stage)   = [{end_pose[0]:.3f}, {end_pose[1]:.3f}, {end_pose[2]:.3f}, {end_pose[3]:.3f}, {end_pose[4]:.3f}]\n')
        f.write(f'; bottom-center start (tip-space heuristic origin) = [{header_meta.get("bottom_center_x", 0.0):.3f}, {header_meta.get("bottom_center_y", 0.0):.3f}, {header_meta.get("bottom_center_z", 0.0):.3f}]\n')
        if emit_extrusion:
            f.write(f'; extrusion_per_mm = {float(extrusion_per_mm):.6f} U/mm-path (coordinated with print feed)\n')
            f.write(f'; pressure_offset_mm = {float(pressure_offset_mm):.3f} U mm\n')
            f.write(f'; pressure advance/retract feeds = {float(pressure_advance_feed):.1f}/{float(pressure_retract_feed):.1f} mm/min on U\n')
            f.write(f'; preflow_dwell_ms = {int(preflow_dwell_ms)}   node_dwell_ms = {int(node_dwell_ms)}\n')
        f.write('G90\n')

        if emit_extrusion:
            f.write('M82\n')
            f.write(f'G92 {cal.u_axis}0\n')
            if abs(prime_mm) > 0.0:
                u_material_abs += float(prime_mm)
                f.write(f'G1 {cal.u_axis}{u_cmd_actual():.3f} F{max(60.0, float(pressure_advance_feed)):.0f} ; prime material\n')

        sx, sy, sz, sb, sc = [float(v) for v in start_pose]
        f.write('; safe startup approach: Z first, then XY, then dive to start Z\n')
        f.write(f'G1 {cal.z_axis}{float(safe_approach_z):.3f} {cal.b_axis}{sb:.3f} {cal.c_axis}{sc:.3f} F{float(travel_feed):.0f}\n')
        f.write(f'G1 {cal.x_axis}{sx:.3f} {cal.y_axis}{sy:.3f} {cal.b_axis}{sb:.3f} {cal.c_axis}{sc:.3f} F{float(travel_feed):.0f}\n')
        f.write(f'G1 {cal.z_axis}{sz:.3f} {cal.b_axis}{sb:.3f} {cal.c_axis}{sc:.3f} F{float(travel_feed):.0f}\n')

        first_edge = True
        emitted_edges = 0
        current_tip: Optional[np.ndarray] = None
        current_b: Optional[float] = None
        current_c: Optional[float] = None

        for ei, plan in enumerate(plans):
            d = plan.offset_vec
            b = float(plan.b_cmd)
            c = float(plan.c_cmd)
            p0_tip = plan.p0_tip.copy()
            p1_tip = plan.p1_tip.copy()
            p0_stage = p0_tip - d

            f.write(f'; --- edge {ei+1}/{len(plans)} ---\n')
            f.write(f'; len={plan.edge_len:.3f} mm, angle_error={plan.angle_error_deg:.3f} deg, B={b:.3f}, C={c:.3f}, head_lead={plan.head_lead_mm:.3f} mm, head_front_ok={int(plan.head_front_ok)}\n')

            if first_edge:
                tx, ty, tz = _clamp_stage_xyz_to_bbox(
                    x=float(p0_stage[0]),
                    y=float(p0_stage[1]),
                    z=float(p0_stage[2]),
                    bbox=virtual_bbox,
                    context=f'edge {ei+1} first-edge travel',
                    warn_log=bbox_warnings,
                )
                travel_axes = [
                    (cal.x_axis, tx),
                    (cal.y_axis, ty),
                    (cal.z_axis, tz),
                    (cal.b_axis, float(b)),
                    (cal.c_axis, float(c)),
                ]
                f.write('; startup move to first strut start (no printed geometry yet)\n')
                f.write(f'G1 {_fmt_axes_move(travel_axes)} F{float(travel_feed):.0f}\n')
                first_edge = False
                current_tip = p0_tip.copy()
                current_b = float(b)
                current_c = float(c)
            else:
                assert current_tip is not None and current_b is not None and current_c is not None
                same_tip_start = float(np.linalg.norm(current_tip - p0_tip)) <= 1e-6

                if same_tip_start:
                    current_tip, current_b, current_c = emit_tip_fixed_c_recenter_if_needed(
                        f,
                        tip_point=current_tip,
                        cur_b=float(current_b),
                        cur_c=float(current_c),
                        comment_prefix='same-tip C recenter',
                    )
                    if float(np.linalg.norm(current_tip - p0_tip)) > 1e-9:
                        _, current_b, current_c = emit_tip_tracked_segmented_move(
                            f, current_tip, p0_tip, float(current_b), float(current_b), float(current_c), float(current_c),
                            samples=travel_samples,
                            feed=float(travel_feed),
                            comment='; same-tip C recenter: return down to print node after high-Z recenter',
                            context_prefix=f'edge {ei+1} same-tip recenter descend',
                        )
                        current_tip = p0_tip.copy()

                    target_c_same_tip = choose_equivalent_c_in_window(
                        target_c_deg=float(c),
                        ref_c_deg=float(current_c),
                        c_min_deg=float(c_window_min),
                        c_max_deg=float(c_window_max),
                    )
                    if same_tip_reorient_needed(float(current_b), float(current_c), float(b), float(target_c_same_tip)):
                        current_b, current_c = emit_tip_fixed_reorient_via_neutral(
                            f,
                            tip_point=p0_tip,
                            b_from=float(current_b),
                            c_from=float(current_c),
                            b_to=float(b),
                            c_to=float(target_c_same_tip),
                            comment_prefix='same-tip turn',
                        )
                    else:
                        f.write('; same-tip transition: orientation already aligned, skipping neutral-B pivot\n')
                    current_tip = p0_tip.copy()
                else:
                    current_tip, current_b, current_c = emit_tip_fixed_c_recenter_if_needed(
                        f,
                        tip_point=current_tip,
                        cur_b=float(current_b),
                        cur_c=float(current_c),
                        comment_prefix='non-print travel C recenter',
                    )

                    if abs(float(current_b) - float(neutral_b_cmd)) > 1e-9:
                        _, current_b, current_c = emit_tip_tracked_segmented_move(
                            f, current_tip, current_tip, float(current_b), float(neutral_b_cmd), float(current_c), float(current_c),
                            samples=travel_samples,
                            feed=float(travel_feed),
                            comment='; non-print travel: uncurl B to neutral at current tip',
                            context_prefix=f'edge {ei+1} travel uncurl',
                        )

                    waypoints, z_plane, used_lift = plan_tip_safe_travel_waypoints(
                        current_tip=current_tip,
                        target_tip=p0_tip,
                        written_segments=written_segments,
                        travel_z_clearance_mm=float(travel_z_clearance_mm),
                        travel_line_clearance_mm=float(travel_line_clearance_mm),
                    )
                    if used_lift:
                        f.write(f'; non-print travel: XY route blocked, lifting to z={z_plane:.3f} above written geometry\n')
                    else:
                        f.write(f'; non-print travel: final descent last, using travel plane z={z_plane:.3f}\n')

                    seg_idx = 0
                    if len(waypoints) >= 2 and abs(float(waypoints[1][2]) - float(waypoints[0][2])) > 1e-9 and float(np.linalg.norm(waypoints[1][:2] - waypoints[0][:2])) <= 1e-9:
                        _, current_b, current_c = emit_tip_tracked_segmented_move(
                            f, waypoints[0], waypoints[1], float(current_b), float(current_b), float(current_c), float(current_c),
                            samples=travel_samples,
                            feed=float(travel_feed),
                            comment='; non-print travel: raise tip in Z before XY move',
                            context_prefix=f'edge {ei+1} raise',
                        )
                        current_tip = waypoints[1].copy()
                        seg_idx = 1

                    has_xy_leg = False
                    if seg_idx + 1 < len(waypoints):
                        a = waypoints[seg_idx]
                        bxy = waypoints[seg_idx + 1]
                        if float(np.linalg.norm(bxy[:2] - a[:2])) > 1e-9:
                            has_xy_leg = True
                            _, current_b, current_c = emit_tip_tracked_segmented_move(
                                f, a, bxy, float(current_b), float(current_b), float(current_c), float(c),
                                samples=travel_samples,
                                feed=float(travel_feed),
                                comment='; non-print travel: move in XY while rotating C at neutral B',
                                context_prefix=f'edge {ei+1} xy',
                            )
                            current_tip = bxy.copy()
                            seg_idx += 1

                    if not has_xy_leg and abs(float(current_c) - float(c)) > 1e-9:
                        _, current_b, current_c = emit_tip_tracked_segmented_move(
                            f, current_tip, current_tip, float(current_b), float(current_b), float(current_c), float(c),
                            samples=travel_samples,
                            feed=float(travel_feed),
                            comment='; non-print travel: rotate C at neutral B while holding tip fixed',
                            context_prefix=f'edge {ei+1} rotate only',
                        )

                    if seg_idx + 1 < len(waypoints):
                        _, current_b, current_c = emit_tip_tracked_segmented_move(
                            f, waypoints[seg_idx], waypoints[seg_idx + 1], float(current_b), float(current_b), float(current_c), float(current_c),
                            samples=travel_samples,
                            feed=float(travel_feed),
                            comment='; non-print travel: final descent in Z to print start',
                            context_prefix=f'edge {ei+1} descend',
                        )
                        current_tip = waypoints[seg_idx + 1].copy()

                    if abs(float(current_b) - float(b)) > 1e-9:
                        _, current_b, current_c = emit_tip_tracked_segmented_move(
                            f, p0_tip, p0_tip, float(current_b), float(b), float(current_c), float(current_c),
                            samples=travel_samples,
                            feed=float(travel_feed),
                            comment='; non-print travel: curl B from neutral to print angle at start point',
                            context_prefix=f'edge {ei+1} curl',
                        )
                    current_tip = p0_tip.copy()
                    current_b = float(b)
                    current_c = float(c)

            if emit_extrusion and float(pressure_offset_mm) > 0.0 and not pressure_charged:
                pressure_charged = True
                f.write('; pressure preload before print pass\n')
                f.write(f'G1 {cal.u_axis}{u_cmd_actual():.3f} F{float(pressure_advance_feed):.0f}\n')
                if int(preflow_dwell_ms) > 0:
                    f.write(f'G4 P{int(preflow_dwell_ms)}\n')

            ts = np.linspace(0.0, 1.0, edge_samples + 1)
            edge_vec_tip = p1_tip - p0_tip
            start_stage_xyz = p0_tip - tip_offset_xyz_physical(cal, float(b), float(c))
            tx, ty, tz = _clamp_stage_xyz_to_bbox(
                x=float(start_stage_xyz[0]),
                y=float(start_stage_xyz[1]),
                z=float(start_stage_xyz[2]),
                bbox=virtual_bbox,
                context=f'edge {ei+1} print start',
                warn_log=bbox_warnings,
            )
            last_stage = np.array([tx, ty, tz], dtype=float)

            for j in range(1, len(ts)):
                tlin = float(ts[j])
                p_tip = p0_tip + tlin * edge_vec_tip
                p_stage = p_tip - d

                if np.linalg.norm((p_stage + d) - p_tip) > 1e-8:
                    raise RuntimeError('Tip-tracking consistency failed during edge interpolation.')

                mx, my, mz = _clamp_stage_xyz_to_bbox(
                    x=float(p_stage[0]),
                    y=float(p_stage[1]),
                    z=float(p_stage[2]),
                    bbox=virtual_bbox,
                    context=f'edge {ei+1} print seg {j}',
                    warn_log=bbox_warnings,
                )
                move_axes = [
                    (cal.x_axis, mx),
                    (cal.y_axis, my),
                    (cal.z_axis, mz),
                    (cal.b_axis, float(b)),
                    (cal.c_axis, float(c)),
                ]

                if emit_extrusion:
                    p_stage_clamped = np.array([mx, my, mz], dtype=float)
                    seg_len = float(np.linalg.norm(p_stage_clamped - last_stage))
                    u_material_abs += float(extrusion_per_mm) * seg_len
                    move_axes.append((cal.u_axis, float(u_cmd_actual())))

                f.write(f'G1 {_fmt_axes_move(move_axes)} F{float(print_feed):.0f}\n')
                last_stage = np.array([mx, my, mz], dtype=float)

            next_is_same_tip = ((ei + 1) < len(plans) and float(np.linalg.norm(p1_tip - plans[ei + 1].p0_tip)) <= 1e-6)
            if emit_extrusion and float(pressure_offset_mm) > 0.0 and pressure_charged:
                if int(node_dwell_ms) > 0:
                    f.write('; end-of-pass dwell for node formation / liquid flow\n')
                    f.write(f'G4 P{int(node_dwell_ms)}\n')
                if next_is_same_tip:
                    f.write('; keep pressure charged for same-tip turn into next edge\n')
                else:
                    pressure_charged = False
                    f.write('; pressure release before non-print travel\n')
                    f.write(f'G1 {cal.u_axis}{u_cmd_actual():.3f} F{float(pressure_retract_feed):.0f}\n')

            written_segments.append((p0_tip.copy(), p1_tip.copy()))
            current_tip = p1_tip.copy()
            current_b = float(b)
            current_c = float(c)
            emitted_edges += 1

        if emit_extrusion and float(pressure_offset_mm) > 0.0 and pressure_charged:
            f.write('; final pressure release at end of print\n')
            pressure_charged = False
            f.write(f'G1 {cal.u_axis}{u_cmd_actual():.3f} F{float(pressure_retract_feed):.0f}\n')

        ex, ey, ez, eb, ec = [float(v) for v in end_pose]
        f.write('; safe end move: raise to safe Z, move XY, then dive to end Z\n')
        f.write(f'G1 {cal.z_axis}{float(safe_approach_z):.3f} {cal.b_axis}{eb:.3f} {cal.c_axis}{ec:.3f} F{float(travel_feed):.0f}\n')
        f.write(f'G1 {cal.x_axis}{ex:.3f} {cal.y_axis}{ey:.3f} {cal.b_axis}{eb:.3f} {cal.c_axis}{ec:.3f} F{float(travel_feed):.0f}\n')
        f.write(f'G1 {cal.z_axis}{ez:.3f} {cal.b_axis}{eb:.3f} {cal.c_axis}{ec:.3f} F{float(travel_feed):.0f}\n')

        if bbox_warnings:
            f.write('; virtual bbox clamp warnings:\n')
            for msg in bbox_warnings:
                f.write(f'; {msg}\n')

        f.write(f'; emitted edges = {emitted_edges}\n')
        f.write(f'; written segment count = {len(written_segments)}\n')
        f.write(f'; bbox warning count = {len(bbox_warnings)}\n')
        f.write('; --- end ---\n')

    if bbox_warnings:
        print(f'Virtual bounding-box warnings: {len(bbox_warnings)} (values clamped)')
        for msg in bbox_warnings:
            print(msg)


# ---------------- Main ----------------

def main(args):
    cal = load_calibration(args.calibration)

    b_lo = cal.b_min if args.min_b is None else float(args.min_b)
    b_hi = cal.b_max if args.max_b is None else float(args.max_b)
    if b_lo > b_hi:
        b_lo, b_hi = b_hi, b_lo

    origin_xyz = np.array([float(args.origin_x), float(args.origin_y), float(args.origin_z)], dtype=float)

    nodes_xyz, edges_idx, geom_meta = build_octet_lattice_cube_edges(
        order=int(args.order),
        pitch_mm=float(args.pitch),
        origin_xyz=origin_xyz,
        include_cube_frame=bool(args.include_cube_frame),
    )

    plans, plan_meta = plan_all_edges_bottom_center_first(
        nodes_xyz=nodes_xyz,
        edges_idx=edges_idx,
        cal=cal,
        b_lo=b_lo,
        b_hi=b_hi,
        b_search_samples=int(args.b_search_samples),
        angle_sign=float(args.tip_angle_sign),
        angle_offset_deg=float(args.tip_angle_offset_deg),
        b_cont_weight=float(args.b_continuity_weight),
        c_cont_weight=float(args.c_continuity_weight),
        min_head_lead_mm=float(args.min_head_lead),
        attack_azimuth_flip_deg=float(args.attack_azimuth_flip_deg),
        c_window_min=float(args.c_window_min),
        c_window_max=float(args.c_window_max),
    )

    if len(plans) != len(edges_idx):
        raise RuntimeError(f"Planned edge count mismatch: planned {len(plans)} vs generated {len(edges_idx)}")

    stage_ranges = compute_stage_gantry_ranges(plans)
    plan_meta.update(stage_ranges)
    plan_meta.update(geom_meta)

    emit_extrusion = float(args.extrusion_per_mm) != 0.0
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

    neutral_b_cmd, neutral_pitch_eff = find_b_for_pitch_target(
        cal=cal,
        b_lo=b_lo,
        b_hi=b_hi,
        target_pitch_deg=float(args.neutral_pitch_target_deg),
        angle_sign=float(args.tip_angle_sign),
        angle_offset_deg=float(args.tip_angle_offset_deg),
    )

    write_gcode_octet_truss(
        out_path=args.out,
        cal=cal,
        plans=plans,
        edge_samples=int(args.edge_samples),
        travel_samples=int(args.travel_samples),
        travel_feed=float(args.travel_feed),
        print_feed=float(args.print_feed),
        extrusion_per_mm=float(args.extrusion_per_mm),
        prime_mm=float(args.prime_mm),
        emit_extrusion=emit_extrusion,
        pressure_offset_mm=float(args.pressure_offset_mm),
        pressure_advance_feed=float(args.pressure_advance_feed),
        pressure_retract_feed=float(args.pressure_retract_feed),
        preflow_dwell_ms=int(args.preflow_dwell_ms),
        node_dwell_ms=int(args.node_dwell_ms),
        header_meta=plan_meta,
        angle_sign=float(args.tip_angle_sign),
        angle_offset_deg=float(args.tip_angle_offset_deg),
        attack_azimuth_flip_deg=float(args.attack_azimuth_flip_deg),
        neutral_b_cmd=float(neutral_b_cmd),
        neutral_pitch_deg=float(neutral_pitch_eff),
        start_pose=start_pose,
        end_pose=end_pose,
        safe_approach_z=float(args.safe_approach_z),
        virtual_bbox=virtual_bbox,
        include_cube_frame=bool(args.include_cube_frame),
        travel_z_clearance_mm=float(args.travel_z_clearance_mm),
        travel_line_clearance_mm=float(args.travel_line_clearance_mm),
        c_window_min=float(args.c_window_min),
        c_window_max=float(args.c_window_max),
        c_recenter_margin_deg=float(args.c_recenter_margin_deg),
    )

    ranges = sampled_ranges(
        cal, b_lo, b_hi,
        angle_sign=float(args.tip_angle_sign),
        angle_offset_deg=float(args.tip_angle_offset_deg),
    )

    bead_area_arg = args.bead_area_mm2 if args.bead_area_mm2 is not None else None
    bead_dia_arg = args.bead_diameter_mm if args.bead_diameter_mm is not None else None
    ex_math = extrusion_math_summary(
        syringe_mm_per_ml=float(args.syringe_mm_per_ml),
        tube_id_inch=float(args.tube_id_inch),
        print_feed_mm_min=float(args.print_feed),
        extrusion_per_mm_u=float(args.extrusion_per_mm),
        bead_area_mm2=(float(bead_area_arg) if bead_area_arg is not None else None),
        bead_diameter_mm=(float(bead_dia_arg) if bead_dia_arg is not None else None),
    )

    pressure_offset_ml = float(args.pressure_offset_mm) / float(args.syringe_mm_per_ml)
    pressure_offset_ul = pressure_offset_ml * 1000.0

    print(f"Wrote {args.out}")
    print("Mode: octet truss in cube, exact tip tracking (hard), tangent alignment (soft)")
    print("Build order: bottom-center-first, bottom-up, center-out (collision reduction heuristic)")
    print(f"Octet order (cells per cube edge): {args.order}")
    print(f"Octet strut length (pitch): {args.pitch:.3f} mm")
    print(f"Cube frame included: {bool(args.include_cube_frame)}")
    print(f"Octet cell edge length: {geom_meta['cell_edge_mm']:.3f} mm")
    print(f"Total cube edge length: {geom_meta['cube_edge_mm']:.3f} mm")
    print(f"Nodes: {len(nodes_xyz)}")
    print(f"Edges generated: {len(edges_idx)}")
    print(f"Edges planned/extruded: {len(plans)}  (all edges extruded exactly once)")
    print(f"Edge samples: {args.edge_samples}")

    xmin = float(np.min(nodes_xyz[:, 0]))
    xmax = float(np.max(nodes_xyz[:, 0]))
    ymin = float(np.min(nodes_xyz[:, 1]))
    ymax = float(np.max(nodes_xyz[:, 1]))
    zmin = float(np.min(nodes_xyz[:, 2]))
    zmax = float(np.max(nodes_xyz[:, 2]))
    print(f"Tip-space structure X range: [{xmin:.3f}, {xmax:.3f}]")
    print(f"Tip-space structure Y range: [{ymin:.3f}, {ymax:.3f}]")
    print(f"Tip-space structure Z range: [{zmin:.3f}, {zmax:.3f}]")

    print(f"Calibration B range: [{cal.b_min:.3f}, {cal.b_max:.3f}]")
    print(f"Commanded B range:  [{b_lo:.3f}, {b_hi:.3f}]")
    print(f"Sampled r(B):       [{ranges['r_min']:.3f}, {ranges['r_max']:.3f}] mm   |r| max={ranges['abs_r_max']:.3f}")
    print(f"Sampled y_off(B):   [{ranges['yoff_min']:.3f}, {ranges['yoff_max']:.3f}] mm")
    print(f"Sampled rho(B):     [{ranges['rho_min']:.3f}, {ranges['rho_max']:.3f}] mm")
    print(f"Sampled alpha(B):   [{ranges['alpha_min_deg']:.3f}, {ranges['alpha_max_deg']:.3f}] deg")
    print(f"Sampled z(B):       [{ranges['z_min']:.3f}, {ranges['z_max']:.3f}] mm")
    print(f"Raw tip_angle poly(B) range:   [{ranges['a_raw_min']:.3f}, {ranges['a_raw_max']:.3f}] deg")
    print(f"Effective pitch (from vertical) range used for solve: "
          f"[{ranges['a_eff_min']:.3f}, {ranges['a_eff_max']:.3f}] deg")

    print(f"Neutral travel pitch target: {float(args.neutral_pitch_target_deg):+.3f} deg -> B={float(neutral_b_cmd):.6f} (effective pitch {float(neutral_pitch_eff):+.6f} deg)")
    print(f"Travel samples per non-print move: {int(args.travel_samples)}")
    print(f"Travel line clearance: {float(args.travel_line_clearance_mm):.3f} mm")
    print(f"Travel Z clearance above written max: {float(args.travel_z_clearance_mm):.3f} mm")
    print(f"Bounded C window: [{float(args.c_window_min):+.3f}, {float(args.c_window_max):+.3f}] deg  (recenter margin {float(args.c_recenter_margin_deg):.3f} deg)")

    print(f"Attack azimuth flip applied (tangency only): {float(args.attack_azimuth_flip_deg):+.3f} deg")
    print("Physical tip kinematics: [x,y]=Rot(C)@[r(B), y_off(B)], z=z(B)")
    if cal.offplane_y_equation:
        print(f"Off-plane equation (from calibration): y_off(B) = {cal.offplane_y_equation}")
    if cal.offplane_y_r_squared is not None:
        print(f"Off-plane fit R^2: {cal.offplane_y_r_squared:.6f}")
    print(f"Off-plane sign convention applied in script: {OFFPLANE_SIGN:+.1f}")

    print(f"B used range:       [{plan_meta['b_min_used']:.3f}, {plan_meta['b_max_used']:.3f}]")
    print(f"C used range:       [{plan_meta['c_min_used']:.3f}, {plan_meta['c_max_used']:.3f}]")

    print(f"Total print length: {plan_meta['total_print_length_mm']:.3f} mm")
    print(f"Total tip travel:   {plan_meta['total_tip_travel_mm']:.3f} mm")

    print(f"Tangent error min/mean/max: "
          f"{plan_meta['angle_error_min_deg']:.3f} / "
          f"{plan_meta['angle_error_mean_deg']:.3f} / "
          f"{plan_meta['angle_error_max_deg']:.3f} deg")

    print(f"Head lead min/mean/max (stage ahead of tip along motion): "
          f"{plan_meta['head_lead_min_mm']:.3f} / "
          f"{plan_meta['head_lead_mean_mm']:.3f} / "
          f"{plan_meta['head_lead_max_mm']:.3f} mm")
    print(f"Head-front feasible edges: {plan_meta['head_front_ok_count']} / {len(plans)} "
          f"(violations: {plan_meta['head_front_violation_count']})")

    print(f"Bottom-center start (tip-space): "
          f"[{plan_meta['bottom_center_x']:.3f}, {plan_meta['bottom_center_y']:.3f}, {plan_meta['bottom_center_z']:.3f}]")
    print(f"User start pose (stage): [{start_pose[0]:.3f}, {start_pose[1]:.3f}, {start_pose[2]:.3f}, "
          f"{start_pose[3]:.3f}, {start_pose[4]:.3f}]")
    print(f"User end pose (stage):   [{end_pose[0]:.3f}, {end_pose[1]:.3f}, {end_pose[2]:.3f}, "
          f"{end_pose[3]:.3f}, {end_pose[4]:.3f}]")
    print(f"Safe approach Z (startup/end): {float(args.safe_approach_z):.3f}")
    print(f"Virtual bbox (enforced during print/travel): "
          f"X[{virtual_bbox['x_min']:.3f}, {virtual_bbox['x_max']:.3f}] "
          f"Y[{virtual_bbox['y_min']:.3f}, {virtual_bbox['y_max']:.3f}] "
          f"Z[{virtual_bbox['z_min']:.3f}, {virtual_bbox['z_max']:.3f}]")

    print(f"X gantry range used: [{plan_meta['x_stage_min']:.3f}, {plan_meta['x_stage_max']:.3f}]")
    print(f"Y gantry range used: [{plan_meta['y_stage_min']:.3f}, {plan_meta['y_stage_max']:.3f}]")
    print(f"Z gantry range used: [{plan_meta['z_stage_min']:.3f}, {plan_meta['z_stage_max']:.3f}]")

    print("\nExtrusion / fluid math:")
    print(f"  Syringe calibration: {float(args.syringe_mm_per_ml):.3f} mm U / mL")
    print(f"  Tube ID: {ex_math['tube_id_mm']:.3f} mm ({float(args.tube_id_inch):.5f} in)")
    print(f"  Tube area: {ex_math['tube_area_mm2']:.6f} mm^2")
    print(f"  Print feed (path): {float(args.print_feed):.3f} mm/min = {ex_math['path_speed_mm_s']:.6f} mm/s")
    print(f"  extrusion_per_mm: {float(args.extrusion_per_mm):.6f} U mm / mm path")
    print(f"  => U speed during coordinated print moves: {ex_math['u_speed_mm_s']:.6f} mm/s")
    print(f"  => Volumetric flow: {ex_math['q_mm3_s']:.6f} mm^3/s = {ex_math['q_ul_s']:.6f} uL/s = {ex_math['q_ml_min']:.6f} mL/min")
    print(f"  => Mean fluid velocity in tube: {ex_math['tube_velocity_mm_s']:.6f} mm/s = {ex_math['tube_velocity_m_s']:.6f} m/s")

    if ex_math["recommended_extrusion_per_mm"] is not None:
        bead_desc = (f"bead area {ex_math['bead_area_used_mm2']:.6f} mm^2"
                     if args.bead_area_mm2 is not None
                     else f"bead dia {float(args.bead_diameter_mm):.6f} mm (circular area assumption)")
        print(f"  From {bead_desc}: recommended extrusion_per_mm ≈ {ex_math['recommended_extrusion_per_mm']:.6f} U mm/mm path")

    print("\nPressure offset sequencing:")
    print(f"  pressure_offset_mm: {float(args.pressure_offset_mm):.3f} U mm  (~{pressure_offset_ml:.6f} mL = {pressure_offset_ul:.3f} uL)")
    print(f"  per pass: advance offset -> dwell {int(args.preflow_dwell_ms)} ms -> print -> dwell {int(args.node_dwell_ms)} ms -> retract offset")
    print(f"  pressure advance/retract feed: {float(args.pressure_advance_feed):.1f}/{float(args.pressure_retract_feed):.1f} mm/min on U")
    if emit_extrusion:
        print(f"  Extrusion enabled: U-axis absolute, prime={float(args.prime_mm):.3f} mm")
    else:
        print("  Extrusion disabled (set --extrusion-per-mm > 0 to emit U-axis extrusion commands)")

    print(f"\nPitch-angle convention used by solver: a(B) = {args.tip_angle_sign:+.3f}*tip_angle_poly(B) + {args.tip_angle_offset_deg:+.3f} deg")
    print(f"Min head-lead target: {args.min_head_lead:.3f} mm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description=(
            "Generate G-code for an octet truss lattice housed inside a cube using calibrated B/C kinematics "
            "with exact tip tracking and soft tangent alignment. Non-print travel is collision-aware in tip space: "
            "neutral-B travel, rotate C during XY travel, and optionally lift above the written structure. "
            "Equivalent C commands are kept inside a bounded window and only recentred at safe high-Z neutral-B points. "
            "Physical tip kinematics are preserved; attack-direction azimuth can be remapped (default +180 deg). "
            "Includes per-pass U pressure preload/release and dwell timing."
        )
    )
    ap.add_argument("--calibration", required=True, help="Path to calibration JSON.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code file.")

    # Placement
    ap.add_argument("--origin-x", type=float, default=DEFAULT_ORIGIN_X,
                    help="Tip-space X of lower/min cube corner.")
    ap.add_argument("--origin-y", type=float, default=DEFAULT_ORIGIN_Y,
                    help="Tip-space Y of lower/min cube corner.")
    ap.add_argument("--origin-z", type=float, default=DEFAULT_ORIGIN_Z,
                    help="Tip-space Z of lower/min cube corner.")

    # Geometry
    ap.add_argument("--order", type=int, default=DEFAULT_ORDER,
                    help="Number of octet unit cells along each cube edge.")
    ap.add_argument("--pitch", type=float, default=DEFAULT_PITCH_MM,
                    help="Octet strut length (mm): FCC nearest-neighbor edge length.")
    ap.add_argument("--edge-samples", type=int, default=DEFAULT_EDGE_SAMPLES,
                    help="Interpolation segments per strut (>=2).")
    ap.add_argument("--include-cube-frame", dest="include_cube_frame", action="store_true",
                    help="Include the outer subdivided cube frame.")
    ap.add_argument("--no-cube-frame", dest="include_cube_frame", action="store_false",
                    help="Do not include the outer subdivided cube frame.")
    ap.set_defaults(include_cube_frame=DEFAULT_INCLUDE_CUBE_FRAME)

    # Motion
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED,
                    help="Feedrate for non-print travel moves (mm/min).")
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED,
                    help="Coordinated feedrate for printing moves (mm/min).")
    ap.add_argument("--safe-approach-z", type=float, default=DEFAULT_SAFE_APPROACH_Z,
                    help="Safe Z used before XY startup/end positioning, then dive to requested Z.")
    ap.add_argument("--travel-samples", type=int, default=DEFAULT_TRAVEL_SAMPLES,
                    help="Segment count used for non-print tip-tracked travel and B/C pivot moves.")
    ap.add_argument("--travel-z-clearance-mm", type=float, default=DEFAULT_TRAVEL_Z_CLEARANCE_MM,
                    help="Additional Z lift above the current max written Z when XY travel would cross printed geometry.")
    ap.add_argument("--travel-line-clearance-mm", type=float, default=DEFAULT_TRAVEL_LINE_CLEARANCE_MM,
                    help="Minimum allowed distance between any non-print tip travel segment and previously extruded struts.")
    ap.add_argument("--neutral-pitch-target-deg", type=float, default=DEFAULT_NEUTRAL_PITCH_TARGET_DEG,
                    help="Pitch-from-vertical target used as the neutral B state for safe travel and C rotation.")
    ap.add_argument("--c-window-min", type=float, default=DEFAULT_C_WINDOW_MIN,
                    help="Lower bound for commanded C angle window; equivalent angles are selected inside this window when possible.")
    ap.add_argument("--c-window-max", type=float, default=DEFAULT_C_WINDOW_MAX,
                    help="Upper bound for commanded C angle window; equivalent angles are selected inside this window when possible.")
    ap.add_argument("--c-recenter-margin-deg", type=float, default=DEFAULT_C_RECENTER_MARGIN_DEG,
                    help="When current C enters this margin near a C-window edge, recenter at a safe high-Z neutral-B point before travel or same-tip handling.")

    # User-defined startup / end poses (stage axes)
    ap.add_argument("--start-x", type=float, default=DEFAULT_START_X,
                    help="Startup stage X target.")
    ap.add_argument("--start-y", type=float, default=DEFAULT_START_Y,
                    help="Startup stage Y target.")
    ap.add_argument("--start-z", type=float, default=DEFAULT_START_Z,
                    help="Startup stage Z target after safe approach.")
    ap.add_argument("--start-b", type=float, default=DEFAULT_START_B,
                    help="Startup B target.")
    ap.add_argument("--start-c", type=float, default=DEFAULT_START_C,
                    help="Startup C target.")
    ap.add_argument("--end-x", type=float, default=DEFAULT_END_X,
                    help="End stage X target.")
    ap.add_argument("--end-y", type=float, default=DEFAULT_END_Y,
                    help="End stage Y target.")
    ap.add_argument("--end-z", type=float, default=DEFAULT_END_Z,
                    help="End stage Z target after safe approach.")
    ap.add_argument("--end-b", type=float, default=DEFAULT_END_B,
                    help="End B target.")
    ap.add_argument("--end-c", type=float, default=DEFAULT_END_C,
                    help="End C target.")

    # Virtual stage-space bounding box
    ap.add_argument("--bbox-x-min", type=float, default=DEFAULT_BBOX_X_MIN,
                    help="Virtual bbox lower bound for X (enforced during print/travel).")
    ap.add_argument("--bbox-x-max", type=float, default=DEFAULT_BBOX_X_MAX,
                    help="Virtual bbox upper bound for X (enforced during print/travel).")
    ap.add_argument("--bbox-y-min", type=float, default=DEFAULT_BBOX_Y_MIN,
                    help="Virtual bbox lower bound for Y (enforced during print/travel).")
    ap.add_argument("--bbox-y-max", type=float, default=DEFAULT_BBOX_Y_MAX,
                    help="Virtual bbox upper bound for Y (enforced during print/travel).")
    ap.add_argument("--bbox-z-min", type=float, default=DEFAULT_BBOX_Z_MIN,
                    help="Virtual bbox lower bound for Z (enforced during print/travel).")
    ap.add_argument("--bbox-z-max", type=float, default=DEFAULT_BBOX_Z_MAX,
                    help="Virtual bbox upper bound for Z (enforced during print/travel).")

    # Extrusion (coordinated with print feed)
    ap.add_argument("--extrusion-per-mm", type=float, default=DEFAULT_EXTRUSION_PER_MM,
                    help="U-axis displacement (mm) per mm of printed path. Set 0 to disable U extrusion on print moves.")
    ap.add_argument("--prime-mm", type=float, default=DEFAULT_PRIME_MM,
                    help="Optional material prime on U axis at start (absolute extrusion mode).")

    # Pressure offset / dwell sequencing on U
    ap.add_argument("--pressure-offset-mm", type=float, default=DEFAULT_PRESSURE_OFFSET_MM,
                    help="U preload offset (mm) before each print pass, retracted after each pass.")
    ap.add_argument("--pressure-advance-feed", type=float, default=DEFAULT_PRESSURE_ADVANCE_FEED,
                    help="Feedrate for U-only pressure advance moves (mm/min).")
    ap.add_argument("--pressure-retract-feed", type=float, default=DEFAULT_PRESSURE_RETRACT_FEED,
                    help="Feedrate for U-only pressure retract moves (mm/min).")
    ap.add_argument("--preflow-dwell-ms", type=int, default=DEFAULT_PREFLOW_DWELL_MS,
                    help="Dwell after pressure advance before printing each edge (milliseconds).")
    ap.add_argument("--node-dwell-ms", type=int, default=DEFAULT_NODE_DWELL_MS,
                    help="Dwell at end of each edge before pressure retract (milliseconds).")

    # B/C solve
    ap.add_argument("--min-b", type=float, default=None, help="Lower commanded B bound (default: calibration).")
    ap.add_argument("--max-b", type=float, default=None, help="Upper commanded B bound (default: calibration).")
    ap.add_argument("--b-search-samples", type=int, default=DEFAULT_B_SEARCH_SAMPLES,
                    help="Number of B samples used when solving tangent alignment per edge.")
    ap.add_argument("--b-continuity-weight", type=float, default=DEFAULT_B_CONTINUITY_WEIGHT,
                    help="Soft continuity weight on ΔB in the tangent solver.")
    ap.add_argument("--c-continuity-weight", type=float, default=DEFAULT_C_CONTINUITY_WEIGHT,
                    help="Soft continuity weight on ΔC in the tangent solver.")
    ap.add_argument("--min-head-lead", type=float, default=DEFAULT_MIN_HEAD_LEAD_MM,
                    help="Preferred minimum head lead (mm): dot(-physical_offset, tangent) >= this when feasible.")

    # Tip-angle convention remap
    ap.add_argument("--tip-angle-sign", type=float, default=DEFAULT_TIP_ANGLE_SIGN,
                    help="Multiplier applied to calibrated tip_angle(B) before tangent solve/model.")
    ap.add_argument("--tip-angle-offset-deg", type=float, default=DEFAULT_TIP_ANGLE_OFFSET_DEG,
                    help="Offset (deg) added to calibrated tip_angle(B) before tangent solve/model.")

    # Attack-direction correction ONLY (does not affect tip kinematics)
    ap.add_argument("--attack-azimuth-flip-deg", type=float, default=DEFAULT_ATTACK_AZIMUTH_FLIP_DEG,
                    help="Attack/nozzle horizontal azimuth uses theta_attack = C + flip (deg). Default 180.")

    # Syringe / tube math helpers (diagnostic only)
    ap.add_argument("--syringe-mm-per-ml", type=float, default=DEFAULT_SYRINGE_MM_PER_ML,
                    help="Syringe U-axis calibration (mm of U travel per mL displaced).")
    ap.add_argument("--tube-id-inch", type=float, default=DEFAULT_TUBE_ID_INCH,
                    help="Tube inner diameter in inches (for flow velocity diagnostics).")
    ap.add_argument("--bead-area-mm2", type=float, default=None,
                    help="Optional target bead cross-sectional area (mm^2). Prints recommended extrusion_per_mm.")
    ap.add_argument("--bead-diameter-mm", type=float, default=None,
                    help="Optional target bead diameter (mm), treated as circular area for recommendation.")

    args = ap.parse_args()
    main(args)
