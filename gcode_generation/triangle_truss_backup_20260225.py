#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code for a tetrahedral truss (tetrahedron-based lattice),
optionally as a DOUBLE tetrahedron mirrored across the base plane, using calibrated B/C
kinematics with exact tip-position tracking and soft tangent alignment.

IMPORTANT CORRECTION
--------------------
The physical tip-position kinematics are LEFT UNCHANGED.

We only apply an azimuth flip to the *attack/nozzle direction* model used for tangency:
    theta_attack = C + attack_azimuth_flip_deg   (default +180 deg)

This means:
- Tip compensation / stage solve still uses the robot's physical kinematics:
    offset_tip(B,C) = [r(B) cos(C), r(B) sin(C), z(B)]
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
Tetrahedral truss based on a subdivided regular tetrahedron simplex lattice.
- --order 1 : single tetrahedron (identity element)
- --order N : subdivided tetrahedral lattice edge graph (all small tetra edges included)
- --double-tetra : mirror the tetrahedral lattice along the base plane (shared base plane)

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
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import ArrayLike


# ---------------- Defaults (CLI-overridable) ----------------
DEFAULT_OUT = "gcode_generation/tetrahedral_truss_tip_priority_tangent_soft.gcode"

# Placement in tip space (world coordinates): this is the V0 corner of the tetrahedron base
DEFAULT_ORIGIN_X = 65.0
DEFAULT_ORIGIN_Y = 0.0
DEFAULT_ORIGIN_Z = -110.0

# Geometry
DEFAULT_ORDER = 2                 # 1 = single tetrahedron (identity element)
DEFAULT_PITCH_MM = 18.0           # edge length of the identity tetrahedral element
DEFAULT_EDGE_SAMPLES = 12         # interpolation segments per strut (>=2)

# Motion
DEFAULT_TRAVEL_FEED = 1200.0      # mm/min
DEFAULT_PRINT_FEED = 300.0        # mm/min (coordinated path feed)

# Extrusion
DEFAULT_EXTRUSION_PER_MM = 0.0    # U mm per mm of printed path; set >0 to enable U extrusion
DEFAULT_PRIME_MM = 0.0            # U mm material prime (not pressure preload)

# Pressure offset / dwell sequencing (U-axis)
DEFAULT_PRESSURE_OFFSET_MM = 5.0          # user-requested example
DEFAULT_PRESSURE_ADVANCE_FEED = 120.0      # U-only feed for preloading (mm/min on U)
DEFAULT_PRESSURE_RETRACT_FEED = 240.0      # U-only feed for pressure release (mm/min on U)
DEFAULT_PREFLOW_DWELL_MS = 150             # wait after pressure advance before printing edge
DEFAULT_NODE_DWELL_MS = 250                # wait at edge end before pressure retract

# B/C solve
DEFAULT_B_SEARCH_SAMPLES = 721
DEFAULT_B_CONTINUITY_WEIGHT = 0.03    # soft continuity on B
DEFAULT_C_CONTINUITY_WEIGHT = 0.003   # soft continuity on C (deg)
DEFAULT_MIN_HEAD_LEAD_MM = 0.0        # prefer head_lead >= this if possible

# Tip-angle convention tweak (for calibration conventions)
DEFAULT_TIP_ANGLE_SIGN = 1.0
DEFAULT_TIP_ANGLE_OFFSET_DEG = 0.0

# Attack-direction correction ONLY (does NOT affect physical tip kinematics)
DEFAULT_ATTACK_AZIMUTH_FLIP_DEG = 180.0

# Syringe/tube math helpers
DEFAULT_SYRINGE_MM_PER_ML = 6.0
DEFAULT_TUBE_ID_INCH = 0.02
# ------------------------------------------------------------


@dataclass
class Calibration:
    pr: np.ndarray            # r(B) coeffs
    pz: np.ndarray            # z(B) coeffs
    pa: np.ndarray            # tip_angle(B) coeffs (deg; calibration convention)

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
class EdgePlan:
    p0_tip: np.ndarray        # desired tip start point (3,)
    p1_tip: np.ndarray        # desired tip end point   (3,)
    b_cmd: float
    c_cmd: float              # absolute (possibly unwrapped) C value in deg
    angle_error_deg: float
    nozzle_dir: np.ndarray    # unit vector (3,) -- attack/tangent model
    offset_vec: np.ndarray    # tip offset vector from stage to tip (3,) -- PHYSICAL kinematics
    edge_len: float
    head_lead_mm: float       # dot(-offset, tangent); >0 means head is ahead
    head_front_ok: bool       # head_lead_mm >= threshold


# ---------------- Calibration / kinematics helpers ----------------

def _polyval4(coeffs: ArrayLike, u: ArrayLike) -> np.ndarray:
    a, b, c, d = coeffs
    u = np.asarray(u, dtype=float)
    return ((a * u + b) * u + c) * u + d


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
    y_axis = str(duet_map.get("depth_axis") or motor_setup.get("depth_axis") or "Y")
    z_axis = str(duet_map.get("vertical_axis") or motor_setup.get("vertical_axis") or "Z")
    b_axis = str(duet_map.get("pull_axis") or motor_setup.get("b_motor_axis") or "B")
    c_axis = str(duet_map.get("rotation_axis") or motor_setup.get("rotation_axis") or "C")
    u_axis = str(duet_map.get("extruder_axis") or "U")
    c_180 = float(motor_setup.get("rotation_axis_180_deg", 180.0))

    return Calibration(
        pr=pr, pz=pz, pa=pa,
        b_min=b_min, b_max=b_max,
        x_axis=x_axis, y_axis=y_axis, z_axis=z_axis,
        b_axis=b_axis, c_axis=c_axis, u_axis=u_axis,
        c_180_deg=c_180
    )


def eval_r(cal: Calibration, b: ArrayLike) -> np.ndarray:
    return _polyval4(cal.pr, b)


def eval_z(cal: Calibration, b: ArrayLike) -> np.ndarray:
    return _polyval4(cal.pz, b)


def eval_tip_angle_pitch_from_vertical_deg(
    cal: Calibration,
    b: ArrayLike,
    angle_sign: float = 1.0,
    angle_offset_deg: float = 0.0,
) -> np.ndarray:
    """
    Effective pitch angle used by the truss solver/model in the USER convention:
      0 deg = vertical
      90 deg = horizontal

    Starts from calibration polynomial and applies sign/offset remap.
    """
    raw = _polyval4(cal.pa, b)
    return angle_sign * raw + angle_offset_deg


# ---- PHYSICAL tip kinematics (unchanged) ----
def tip_offset_xyz_physical(cal: Calibration, b: float, c_deg: float) -> np.ndarray:
    """
    Physical tip offset from stage origin for a given B/C:
        [r(B) cos(C), r(B) sin(C), z(B)]

    This remains unchanged to preserve correct tip positioning on the robot.
    """
    r = float(eval_r(cal, b))
    z = float(eval_z(cal, b))
    c = math.radians(c_deg)
    return np.array([r * math.cos(c), r * math.sin(c), z], dtype=float)


# ---- ATTACK / tangency model (can be remapped) ----
def attack_theta_deg(c_deg: ArrayLike, attack_azimuth_flip_deg: float) -> np.ndarray:
    """
    Attack/nozzle horizontal azimuth used only for tangent alignment:
        theta_attack = C + attack_azimuth_flip_deg
    """
    return np.asarray(c_deg, dtype=float) + float(attack_azimuth_flip_deg)


def nozzle_axis_unit_xyz_from_vertical_pitch(
    cal: Calibration,
    b: float,
    c_deg: float,
    angle_sign: float = 1.0,
    angle_offset_deg: float = 0.0,
    attack_azimuth_flip_deg: float = DEFAULT_ATTACK_AZIMUTH_FLIP_DEG,
) -> np.ndarray:
    """
    User convention:
      pitch a(B) measured FROM VERTICAL.
      a = 0° => +Z
      a = 90° => horizontal

    Attack-direction model (used for tangential alignment):
      nozzle_dir = [sin(a) cos(theta_attack), sin(a) sin(theta_attack), cos(a)]
      theta_attack = C + attack_azimuth_flip_deg
    """
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
    zz = eval_z(cal, bb)
    aa_raw = _polyval4(cal.pa, bb)
    aa_eff = eval_tip_angle_pitch_from_vertical_deg(cal, bb, angle_sign=angle_sign, angle_offset_deg=angle_offset_deg)
    return {
        "r_min": float(np.min(rr)),
        "r_max": float(np.max(rr)),
        "abs_r_min": float(np.min(np.abs(rr))),
        "abs_r_max": float(np.max(np.abs(rr))),
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
    """
    Returns useful extrusion/feed math numbers.

    syringe_mm_per_ml: U mm per mL (e.g. 6 mm/mL)
    extrusion_per_mm_u: U mm commanded per 1 mm of path during coordinated print move
    print_feed_mm_min: coordinated path feed (mm/min)
    """
    if syringe_mm_per_ml <= 0:
        raise ValueError("syringe_mm_per_ml must be > 0.")
    if print_feed_mm_min <= 0:
        raise ValueError("print_feed_mm_min must be > 0.")

    path_speed_mm_s = float(print_feed_mm_min) / 60.0
    u_speed_mm_s = float(extrusion_per_mm_u) * path_speed_mm_s

    # 1 mL = 1000 mm^3
    q_mm3_s = (1000.0 / float(syringe_mm_per_ml)) * u_speed_mm_s
    q_ml_min = q_mm3_s * 60.0 / 1000.0
    q_ul_s = q_mm3_s  # numerically same because 1 mm^3 = 1 uL

    area_tube_mm2 = tube_area_mm2_from_id_inch(float(tube_id_inch))
    tube_velocity_mm_s = q_mm3_s / area_tube_mm2 if area_tube_mm2 > 0 else float("nan")
    tube_velocity_m_s = tube_velocity_mm_s / 1000.0

    # Optional bead area estimate
    bead_area_used = None
    if bead_area_mm2 is not None:
        bead_area_used = float(bead_area_mm2)
    elif bead_diameter_mm is not None:
        d = float(bead_diameter_mm)
        bead_area_used = math.pi * (0.5 * d) ** 2

    recommended_extrusion_per_mm = None
    if bead_area_used is not None:
        # U_per_path_mm = (syringe_mm_per_ml mm/mL) * (bead_area mm^3 per mm / 1000 mm^3/mL)
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


# ---------------- Geometry: tetrahedral truss (simplex lattice graph) ----------------

def _sorted_edge_key(a, b):
    return (a, b) if a <= b else (b, a)


def build_single_tetrahedral_truss_edges(
    order: int,
    pitch_mm: float,
    origin_xyz: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """
    Build the edge graph of a subdivided regular tetrahedron (simplex lattice).

    - order = 1 => single tetrahedron (identity element)
    - order = N => edges of tetrahedral lattice with unit element edge length = pitch_mm
      and outer tetrahedron edge length = N * pitch_mm

    Nodes are integer barycentric tuples (a,b,c,d) with a+b+c+d = order.
    Two nodes are connected if one unit of barycentric weight is moved from one coordinate
    to another (i.e., they differ by +1 in one coord and -1 in another): this gives the
    1-skeleton of the tetrahedral simplex grid, ensuring all small tetra edges are included.

    Base plane is the face V0-V1-V2 at z = origin_xyz[2].
    Apex V3 is above the base plane.
    """
    if order < 1:
        raise ValueError("order must be >= 1.")
    if pitch_mm <= 0.0:
        raise ValueError("pitch_mm must be > 0.")

    n = int(order)
    L = n * float(pitch_mm)  # outer tetra edge length

    # Regular tetrahedron vertices (V0 base corner at origin_xyz)
    V0 = np.array([0.0, 0.0, 0.0], dtype=float)
    V1 = np.array([L, 0.0, 0.0], dtype=float)
    V2 = np.array([0.5 * L, (math.sqrt(3.0) / 2.0) * L, 0.0], dtype=float)
    V3 = np.array([0.5 * L, (math.sqrt(3.0) / 6.0) * L, math.sqrt(2.0 / 3.0) * L], dtype=float)

    bary_nodes: List[Tuple[int, int, int, int]] = []
    node_index: Dict[Tuple[int, int, int, int], int] = {}

    # Enumerate all barycentric integer nodes
    for a in range(n + 1):
        for b in range(n + 1 - a):
            for c in range(n + 1 - a - b):
                d = n - a - b - c
                q = (a, b, c, d)
                node_index[q] = len(bary_nodes)
                bary_nodes.append(q)

    # Build edges by one-unit transfers in barycentric coordinates
    edge_keys = set()
    for q in bary_nodes:
        q_arr = list(q)
        for i in range(4):
            if q_arr[i] <= 0:
                continue
            for j in range(4):
                if i == j:
                    continue
                q2 = q_arr.copy()
                q2[i] -= 1
                q2[j] += 1
                q2t = tuple(q2)
                if q2t in node_index:
                    edge_keys.add(_sorted_edge_key(q, q2t))

    # Map barycentric nodes to Cartesian positions
    nodes_xyz = np.zeros((len(bary_nodes), 3), dtype=float)
    for idx, (a, b, c, d) in enumerate(bary_nodes):
        p = (a * V0 + b * V1 + c * V2 + d * V3) / float(n)
        nodes_xyz[idx, :] = origin_xyz + p

    edges_idx: List[Tuple[int, int]] = []
    for qa, qb in sorted(edge_keys):
        ia = node_index[qa]
        ib = node_index[qb]
        if ia != ib:
            edges_idx.append((ia, ib))

    bary_arr = np.array(bary_nodes, dtype=int)
    return nodes_xyz, edges_idx, bary_arr


def mirror_tetrahedral_truss_along_base_plane(
    nodes_xyz: np.ndarray,
    edges_idx: List[Tuple[int, int]],
    z_base: float,
    tol_decimals: int = 9,
) -> Tuple[np.ndarray, List[Tuple[int, int]]]:
    """
    Mirror the tetrahedral truss across the base plane z = z_base.

    Base-plane nodes are shared (deduplicated).
    All edges from the mirrored copy are included exactly once in the combined graph.
    """
    if len(nodes_xyz) == 0:
        return nodes_xyz.copy(), list(edges_idx)

    def key_of_point(p: np.ndarray) -> Tuple[float, float, float]:
        return (round(float(p[0]), tol_decimals),
                round(float(p[1]), tol_decimals),
                round(float(p[2]), tol_decimals))

    combined_nodes: List[np.ndarray] = []
    point_to_idx: Dict[Tuple[float, float, float], int] = {}

    # Seed with original nodes
    for p in nodes_xyz:
        k = key_of_point(p)
        if k not in point_to_idx:
            point_to_idx[k] = len(combined_nodes)
            combined_nodes.append(np.array(p, dtype=float))

    # Original edges
    combined_edge_keys = set()
    for ia, ib in edges_idx:
        a, b = (ia, ib) if ia <= ib else (ib, ia)
        if a != b:
            combined_edge_keys.add((a, b))

    # Mirror nodes and add mirrored edges
    mirror_idx_map: Dict[int, int] = {}
    for i, p in enumerate(nodes_xyz):
        pm = np.array([p[0], p[1], 2.0 * z_base - p[2]], dtype=float)
        k = key_of_point(pm)
        if k not in point_to_idx:
            point_to_idx[k] = len(combined_nodes)
            combined_nodes.append(pm)
        mirror_idx_map[i] = point_to_idx[k]

    for ia, ib in edges_idx:
        ma = mirror_idx_map[ia]
        mb = mirror_idx_map[ib]
        if ma == mb:
            continue
        a, b = (ma, mb) if ma <= mb else (mb, ma)
        combined_edge_keys.add((a, b))

    combined_nodes_arr = np.vstack(combined_nodes)
    combined_edges = sorted(combined_edge_keys)
    return combined_nodes_arr, combined_edges


def build_tetrahedral_truss_edges(
    order: int,
    pitch_mm: float,
    origin_xyz: np.ndarray,
    double_tetra: bool = False,
) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """
    Build tetrahedral truss edges, optionally mirrored along the base plane.

    Returns:
      nodes_xyz, edges_idx, bary_nodes_upper
    """
    nodes_xyz, edges_idx, bary_nodes = build_single_tetrahedral_truss_edges(
        order=order,
        pitch_mm=pitch_mm,
        origin_xyz=origin_xyz,
    )

    if not bool(double_tetra):
        return nodes_xyz, edges_idx, bary_nodes

    z_base = float(origin_xyz[2])  # base face plane of the upper tetra lattice
    nodes2, edges2 = mirror_tetrahedral_truss_along_base_plane(
        nodes_xyz=nodes_xyz,
        edges_idx=edges_idx,
        z_base=z_base,
    )
    return nodes2, edges2, bary_nodes


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
) -> EdgePlan:
    """
    Solve B/C for a directed strut (p0 -> p1) with:
      - exact tip tracking (hard, via stage = tip - physical_offset)
      - tangent alignment (soft, choose B/C minimizing angle error + continuity)
      - compute head_lead metric for later direction preference

    IMPORTANT:
    - attack azimuth flip affects ONLY the nozzle/tangency model
    - physical tip offset remains unchanged
    """
    v = p1_tip - p0_tip
    L = float(np.linalg.norm(v))
    if L <= 1e-12:
        raise ValueError("Zero-length edge encountered.")
    t = v / L

    # Desired horizontal azimuth of tangent in world coords
    horiz = float(math.hypot(float(t[0]), float(t[1])))
    has_horiz = horiz > 1e-12
    phi_deg = math.degrees(math.atan2(float(t[1]), float(t[0]))) if has_horiz else 0.0

    n_samples = max(11, int(b_search_samples))
    bb = np.linspace(float(b_lo), float(b_hi), n_samples)

    # Effective pitch angle in user convention (from vertical)
    a_deg = eval_tip_angle_pitch_from_vertical_deg(
        cal, bb, angle_sign=angle_sign, angle_offset_deg=angle_offset_deg
    )
    a_rad = np.deg2rad(a_deg)
    sin_a = np.sin(a_rad)
    cos_a = np.cos(a_rad)

    # Tangency target uses theta_attack = C + attack_flip.
    # To get desired attack azimuth theta_attack_base, command:
    #   C = theta_attack_base - attack_flip
    if has_horiz:
        theta_attack_base = np.where(sin_a >= 0.0, phi_deg, phi_deg + 180.0)
        c_base = theta_attack_base - float(attack_azimuth_flip_deg)
    else:
        c0 = 0.0 if prev_c is None else float(prev_c)
        c_base = np.full_like(bb, c0, dtype=float)

    # Unwrap near previous C to avoid unnecessary spin
    c_cmd = unwrap_deg_near(c_base, prev_c)

    # Attack/nozzle direction uses theta_attack = C + attack_flip
    theta_attack_deg = attack_theta_deg(c_cmd, attack_azimuth_flip_deg=float(attack_azimuth_flip_deg))
    theta_attack_rad = np.deg2rad(theta_attack_deg)

    nx = sin_a * np.cos(theta_attack_rad)
    ny = sin_a * np.sin(theta_attack_rad)
    nz = cos_a

    dots = nx * t[0] + ny * t[1] + nz * t[2]
    dots = np.clip(dots, -1.0, 1.0)
    angle_err_deg = np.rad2deg(np.arccos(dots))

    # Soft continuity cost only (head-front handled at edge-direction choice stage)
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

    # PHYSICAL tip offset and head lead (unchanged kinematics)
    offset_vec = tip_offset_xyz_physical(cal, b_star, c_star)
    head_lead = float(np.dot(-offset_vec, t))  # >0 means stage/head is ahead along writing direction
    head_front_ok = bool(head_lead >= float(min_head_lead_mm))

    # Exact tip reconstruction sanity
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
) -> Tuple[EdgePlan, float]:
    """
    Evaluate both edge directions and choose the better one.

    Preference order (strong to weak):
      1) head_front_ok=True over False
      2) larger head_lead
      3) shorter travel to start
      4) lower tangent error
    """
    plan_ab = solve_directed_edge_tangent_soft(
        p0_tip=pa_tip, p1_tip=pb_tip,
        cal=cal, b_lo=b_lo, b_hi=b_hi,
        b_search_samples=b_search_samples,
        prev_b=prev_b, prev_c=prev_c,
        angle_sign=angle_sign, angle_offset_deg=angle_offset_deg,
        b_cont_weight=b_cont_weight, c_cont_weight=c_cont_weight,
        min_head_lead_mm=min_head_lead_mm,
        attack_azimuth_flip_deg=attack_azimuth_flip_deg,
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
    )

    travel_ab = 0.0 if current_tip is None else float(np.linalg.norm(plan_ab.p0_tip - current_tip))
    travel_ba = 0.0 if current_tip is None else float(np.linalg.norm(plan_ba.p0_tip - current_tip))

    def score(plan: EdgePlan, travel: float) -> Tuple[int, float, float, float]:
        front_rank = 0 if plan.head_front_ok else 1
        neg_head_lead = -float(plan.head_lead_mm)  # larger head lead preferred
        return (front_rank, neg_head_lead, float(travel), float(plan.angle_error_deg))

    if score(plan_ab, travel_ab) <= score(plan_ba, travel_ba):
        return plan_ab, travel_ab
    return plan_ba, travel_ba


def compute_bottom_center(nodes_xyz: np.ndarray) -> np.ndarray:
    """
    Bottom-center point in tip space:
      XY center = mean of nodes on minimum Z plane
      Z = minimum Z
    """
    zmin = float(np.min(nodes_xyz[:, 2]))
    bottom_mask = np.isclose(nodes_xyz[:, 2], zmin, atol=1e-9)
    bottom_nodes = nodes_xyz[bottom_mask]
    if len(bottom_nodes) == 0:
        raise RuntimeError("Failed to identify bottom nodes.")
    xy_center = np.mean(bottom_nodes[:, :2], axis=0)
    return np.array([float(xy_center[0]), float(xy_center[1]), zmin], dtype=float)


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
) -> Tuple[List[EdgePlan], dict]:
    """
    Bottom-center-first, bottom-up heuristic ordering to reduce kinematic collisions.

    Global priority (strongest to weakest):
      1) lower z_min first
      2) lower z_max first
      3) center-out from bottom-center
      4) head-front feasible preferred, then higher head_lead
      5) shorter travel to start
      6) lower tangent error
    """
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
    current_tip = bottom_center.copy()   # requested start heuristic
    prev_b: Optional[float] = None
    prev_c: Optional[float] = None

    remaining = set(range(len(edges_idx)))
    plans: List[EdgePlan] = []

    total_tip_travel = 0.0
    total_print_len = 0.0
    z_global_min = float(np.min(nodes_xyz[:, 2]))

    while remaining:
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
            )

            z_min_edge = float(min(plan.p0_tip[2], plan.p1_tip[2]))
            z_max_edge = float(max(plan.p0_tip[2], plan.p1_tip[2]))
            mid = 0.5 * (plan.p0_tip + plan.p1_tip)
            radial_from_bottom_center = float(np.linalg.norm(mid[:2] - bottom_center[:2]))

            z_start = float(plan.p0_tip[2])
            z_dir_bias = max(0.0, z_start - z_min_edge)  # prefer starting from lower endpoint

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
    """
    Compute gantry (stage) XYZ ranges used across all planned struts.

    For each strut, B/C are constant so the tip->stage offset is constant, and the stage
    path is linear. Therefore axis extrema occur at the strut endpoints.
    """
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


def write_gcode_tetrahedral_truss(
    out_path: str,
    cal: Calibration,
    plans: List[EdgePlan],
    edge_samples: int,
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
    double_tetra: bool,
    attack_azimuth_flip_deg: float,
):
    """
    Emit G-code:
      - absolute XYZ (G90)
      - optional absolute U extrusion (M82 + G92 U0)
      - every edge is printed once, constant B/C per edge
      - exact tip tracking via stage = tip - physical_offset(B,C)
      - optional per-edge pressure preload/release on U axis
    """
    if edge_samples < 2:
        edge_samples = 2

    # We track material displacement separately from pressure preload.
    u_material_abs = 0.0
    pressure_charged = False

    def u_cmd_actual() -> float:
        return u_material_abs + (float(pressure_offset_mm) if pressure_charged else 0.0)

    with open(out_path, "w") as f:
        f.write("; generated by tetrahedral_truss_tip_priority_tangent_soft.py\n")
        f.write("; strategy: exact tip position (hard) + tangent/nozzle angle alignment (soft)\n")
        f.write("; user tip-angle convention: angle measured FROM VERTICAL (horizontal line ~90 deg)\n")
        f.write("; head-front preference: choose edge direction so head/stage leads deposition point when possible\n")
        f.write("; build order heuristic: bottom-center-first, bottom-up, center-out\n")
        f.write(f"; geometry: {'double tetrahedron (mirrored along base plane)' if double_tetra else 'single tetrahedron / tetrahedral lattice'}\n")
        f.write("; PHYSICAL tip kinematics unchanged; attack azimuth remap used only for tangency\n")
        f.write("; model assumptions:\n")
        f.write(";   physical tip_offset(B,C) = [r(B) cos(C), r(B) sin(C), z(B)]\n")
        f.write(";   nozzle attack dir(B,C) = [sin(a(B)) cos(theta_attack), sin(a(B)) sin(theta_attack), cos(a(B))]\n")
        f.write(f";   theta_attack = C + {float(attack_azimuth_flip_deg):+.3f} deg\n")
        f.write(f";   a(B) = ({angle_sign:+.3f}) * tip_angle_poly(B) + {angle_offset_deg:+.3f} deg\n")
        f.write(f"; axes: X->{cal.x_axis}, Y->{cal.y_axis}, Z->{cal.z_axis}, B->{cal.b_axis}, C->{cal.c_axis}, U->{cal.u_axis}\n")
        f.write(f"; edges planned = {header_meta.get('n_edges', 0)} (all edges extruded exactly once)\n")
        f.write(f"; total print length ~ {header_meta.get('total_print_length_mm', 0.0):.3f} mm\n")
        f.write(f"; total tip travel  ~ {header_meta.get('total_tip_travel_mm', 0.0):.3f} mm\n")
        f.write(f"; tangent angle error [min/mean/max] = "
                f"{header_meta.get('angle_error_min_deg', 0.0):.3f} / "
                f"{header_meta.get('angle_error_mean_deg', 0.0):.3f} / "
                f"{header_meta.get('angle_error_max_deg', 0.0):.3f} deg\n")
        f.write(f"; head-lead [min/mean/max] mm = "
                f"{header_meta.get('head_lead_min_mm', 0.0):.3f} / "
                f"{header_meta.get('head_lead_mean_mm', 0.0):.3f} / "
                f"{header_meta.get('head_lead_max_mm', 0.0):.3f}\n")
        f.write(f"; head-front ok / violations = "
                f"{header_meta.get('head_front_ok_count', 0)} / "
                f"{header_meta.get('head_front_violation_count', 0)}\n")
        f.write(f"; B used range = [{header_meta.get('b_min_used', 0.0):.3f}, {header_meta.get('b_max_used', 0.0):.3f}]\n")
        f.write(f"; C used range = [{header_meta.get('c_min_used', 0.0):.3f}, {header_meta.get('c_max_used', 0.0):.3f}]\n")
        f.write(f"; X gantry range used = [{header_meta.get('x_stage_min', 0.0):.3f}, {header_meta.get('x_stage_max', 0.0):.3f}]\n")
        f.write(f"; Y gantry range used = [{header_meta.get('y_stage_min', 0.0):.3f}, {header_meta.get('y_stage_max', 0.0):.3f}]\n")
        f.write(f"; Z gantry range used = [{header_meta.get('z_stage_min', 0.0):.3f}, {header_meta.get('z_stage_max', 0.0):.3f}]\n")
        f.write(f"; bottom-center start (tip-space heuristic origin) = "
                f"[{header_meta.get('bottom_center_x', 0.0):.3f}, "
                f"{header_meta.get('bottom_center_y', 0.0):.3f}, "
                f"{header_meta.get('bottom_center_z', 0.0):.3f}]\n")
        if emit_extrusion:
            f.write(f"; extrusion_per_mm = {float(extrusion_per_mm):.6f} U/mm-path (coordinated with print feed)\n")
            f.write(f"; pressure_offset_mm = {float(pressure_offset_mm):.3f} U mm\n")
            f.write(f"; pressure advance/retract feeds = {float(pressure_advance_feed):.1f}/{float(pressure_retract_feed):.1f} mm/min on U\n")
            f.write(f"; preflow_dwell_ms = {int(preflow_dwell_ms)}   node_dwell_ms = {int(node_dwell_ms)}\n")
        f.write("G90\n")

        if emit_extrusion:
            # Absolute extrusion mode for U, if supported by controller. (Many RRF setups do.)
            f.write("M82\n")
            f.write(f"G92 {cal.u_axis}0\n")
            if abs(prime_mm) > 0.0:
                u_material_abs += float(prime_mm)
                f.write(f"G1 {cal.u_axis}{u_cmd_actual():.3f} F{max(60.0, float(pressure_advance_feed)):.0f} ; prime material\n")

        first_edge = True
        emitted_edges = 0

        for ei, plan in enumerate(plans):
            d = plan.offset_vec
            b = plan.b_cmd
            c = plan.c_cmd

            p0_tip = plan.p0_tip
            p1_tip = plan.p1_tip
            p0_stage = p0_tip - d

            f.write(f"; --- edge {ei+1}/{len(plans)} ---\n")
            f.write(f"; len={plan.edge_len:.3f} mm, angle_error={plan.angle_error_deg:.3f} deg, "
                    f"B={b:.3f}, C={c:.3f}, head_lead={plan.head_lead_mm:.3f} mm, "
                    f"head_front_ok={int(plan.head_front_ok)}\n")

            # Travel to edge start (tip exact)
            travel_axes = [
                (cal.x_axis, float(p0_stage[0])),
                (cal.y_axis, float(p0_stage[1])),
                (cal.z_axis, float(p0_stage[2])),
                (cal.b_axis, float(b)),
                (cal.c_axis, float(c)),
            ]
            if first_edge:
                f.write("; startup move to first strut start (bottom-center-first ordering)\n")
                f.write(f"G1 {_fmt_axes_move(travel_axes)} F{float(travel_feed):.0f}\n")
                first_edge = False
            else:
                f.write("; non-print travel to next strut start (tip exact)\n")
                f.write(f"G1 {_fmt_axes_move(travel_axes)} F{float(travel_feed):.0f}\n")

            # Pressure advance before print pass
            if emit_extrusion and float(pressure_offset_mm) > 0.0 and not pressure_charged:
                pressure_charged = True
                f.write("; pressure preload before print pass\n")
                f.write(f"G1 {cal.u_axis}{u_cmd_actual():.3f} F{float(pressure_advance_feed):.0f}\n")
                if int(preflow_dwell_ms) > 0:
                    f.write(f"G4 P{int(preflow_dwell_ms)}\n")

            # Print edge with constant B/C and exact tip tracking
            ts = np.linspace(0.0, 1.0, edge_samples + 1)
            edge_vec_tip = p1_tip - p0_tip
            last_stage = p0_stage.copy()

            for j in range(1, len(ts)):
                tlin = float(ts[j])
                p_tip = p0_tip + tlin * edge_vec_tip
                p_stage = p_tip - d

                # Exact tip consistency
                if np.linalg.norm((p_stage + d) - p_tip) > 1e-8:
                    raise RuntimeError("Tip-tracking consistency failed during edge interpolation.")

                move_axes = [
                    (cal.x_axis, float(p_stage[0])),
                    (cal.y_axis, float(p_stage[1])),
                    (cal.z_axis, float(p_stage[2])),
                    (cal.b_axis, float(b)),
                    (cal.c_axis, float(c)),
                ]

                if emit_extrusion:
                    seg_len = float(np.linalg.norm(p_stage - last_stage))
                    # Material displacement only; pressure preload stays as additive offset
                    u_material_abs += float(extrusion_per_mm) * seg_len
                    move_axes.append((cal.u_axis, float(u_cmd_actual())))

                f.write(f"G1 {_fmt_axes_move(move_axes)} F{float(print_feed):.0f}\n")
                last_stage = p_stage

            # End-of-pass dwell + pressure retract before next travel
            if emit_extrusion and float(pressure_offset_mm) > 0.0 and pressure_charged:
                if int(node_dwell_ms) > 0:
                    f.write("; end-of-pass dwell for node formation / liquid flow\n")
                    f.write(f"G4 P{int(node_dwell_ms)}\n")
                pressure_charged = False
                f.write("; pressure release before travel\n")
                f.write(f"G1 {cal.u_axis}{u_cmd_actual():.3f} F{float(pressure_retract_feed):.0f}\n")

            emitted_edges += 1

        # Safety: ensure pressure is released at print end
        if emit_extrusion and float(pressure_offset_mm) > 0.0 and pressure_charged:
            f.write("; final pressure release at end of print\n")
            pressure_charged = False
            f.write(f"G1 {cal.u_axis}{u_cmd_actual():.3f} F{float(pressure_retract_feed):.0f}\n")

        f.write(f"; emitted edges = {emitted_edges}\n")
        f.write("; --- end ---\n")


# ---------------- Main ----------------

def main(args):
    cal = load_calibration(args.calibration)

    b_lo = cal.b_min if args.min_b is None else float(args.min_b)
    b_hi = cal.b_max if args.max_b is None else float(args.max_b)
    if b_lo > b_hi:
        b_lo, b_hi = b_hi, b_lo

    origin_xyz = np.array([float(args.origin_x), float(args.origin_y), float(args.origin_z)], dtype=float)

    nodes_xyz, edges_idx, _bary_nodes_upper = build_tetrahedral_truss_edges(
        order=int(args.order),
        pitch_mm=float(args.pitch),
        origin_xyz=origin_xyz,
        double_tetra=bool(args.double_tetra),
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
    )

    # Ensure every generated edge is planned exactly once
    if len(plans) != len(edges_idx):
        raise RuntimeError(f"Planned edge count mismatch: planned {len(plans)} vs generated {len(edges_idx)}")

    stage_ranges = compute_stage_gantry_ranges(plans)
    plan_meta.update(stage_ranges)

    # Extrusion config
    emit_extrusion = float(args.extrusion_per_mm) != 0.0

    write_gcode_tetrahedral_truss(
        out_path=args.out,
        cal=cal,
        plans=plans,
        edge_samples=int(args.edge_samples),
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
        double_tetra=bool(args.double_tetra),
        attack_azimuth_flip_deg=float(args.attack_azimuth_flip_deg),
    )

    ranges = sampled_ranges(
        cal, b_lo, b_hi,
        angle_sign=float(args.tip_angle_sign),
        angle_offset_deg=float(args.tip_angle_offset_deg),
    )

    # Extrusion math summary / diagnostics
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

    # Diagnostics
    print(f"Wrote {args.out}")
    print("Mode: tetrahedral truss (identity element = tetrahedron), exact tip tracking (hard), tangent alignment (soft)")
    print("Build order: bottom-center-first, bottom-up, center-out (collision reduction heuristic)")
    print(f"Geometry: {'DOUBLE tetrahedron (mirrored across base plane)' if args.double_tetra else 'single tetrahedron / tetrahedral lattice'}")
    print(f"Tetrahedral order: {args.order}  (order=1 is a single tetrahedron identity element)")
    print(f"Identity element edge length (pitch): {args.pitch:.3f} mm")
    print(f"Nodes: {len(nodes_xyz)}")
    print(f"Edges generated: {len(edges_idx)}")
    print(f"Edges planned/extruded: {len(plans)}  (all edges extruded exactly once)")
    print(f"Edge samples: {args.edge_samples}")

    zmin = float(np.min(nodes_xyz[:, 2]))
    zmax = float(np.max(nodes_xyz[:, 2]))
    print(f"Tip-space structure Z range: [{zmin:.3f}, {zmax:.3f}]")

    print(f"Calibration B range: [{cal.b_min:.3f}, {cal.b_max:.3f}]")
    print(f"Commanded B range:  [{b_lo:.3f}, {b_hi:.3f}]")
    print(f"Sampled r(B):       [{ranges['r_min']:.3f}, {ranges['r_max']:.3f}] mm   |r| max={ranges['abs_r_max']:.3f}")
    print(f"Sampled z(B):       [{ranges['z_min']:.3f}, {ranges['z_max']:.3f}] mm")
    print(f"Raw tip_angle poly(B) range:   [{ranges['a_raw_min']:.3f}, {ranges['a_raw_max']:.3f}] deg")
    print(f"Effective pitch (from vertical) range used for solve: "
          f"[{ranges['a_eff_min']:.3f}, {ranges['a_eff_max']:.3f}] deg")

    print(f"Attack azimuth flip applied (tangency only): {float(args.attack_azimuth_flip_deg):+.3f} deg")
    print("Physical tip kinematics: unchanged (offset uses C directly)")

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

    # Gantry ranges
    print(f"X gantry range used: [{plan_meta['x_stage_min']:.3f}, {plan_meta['x_stage_max']:.3f}]")
    print(f"Y gantry range used: [{plan_meta['y_stage_min']:.3f}, {plan_meta['y_stage_max']:.3f}]")
    print(f"Z gantry range used: [{plan_meta['z_stage_min']:.3f}, {plan_meta['z_stage_max']:.3f}]")

    # Extrusion + fluid math
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
            "Generate G-code for a tetrahedral truss (tetrahedron-based lattice), optionally as a double tetrahedron "
            "mirrored across the base plane, using calibrated B/C kinematics with exact tip tracking and soft tangent alignment. "
            "Physical tip kinematics are preserved; attack-direction azimuth can be remapped (default +180 deg). "
            "Includes per-pass U pressure preload/release and dwell timing."
        )
    )
    ap.add_argument("--calibration", required=True, help="Path to calibration JSON.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code file.")

    # Placement
    ap.add_argument("--origin-x", type=float, default=DEFAULT_ORIGIN_X,
                    help="Tip-space X of tetrahedral lattice V0 base corner.")
    ap.add_argument("--origin-y", type=float, default=DEFAULT_ORIGIN_Y,
                    help="Tip-space Y of tetrahedral lattice V0 base corner.")
    ap.add_argument("--origin-z", type=float, default=DEFAULT_ORIGIN_Z,
                    help="Tip-space Z of tetrahedral lattice base plane (V0/V1/V2 face plane).")

    # Geometry
    ap.add_argument("--order", type=int, default=DEFAULT_ORDER,
                    help="Tetrahedral subdivision order. 1 = single tetrahedron (identity element).")
    ap.add_argument("--pitch", type=float, default=DEFAULT_PITCH_MM,
                    help="Edge length (mm) of the identity tetrahedral element.")
    ap.add_argument("--double-tetra", action="store_true",
                    help="Mirror the tetrahedral lattice along the base plane to make a double tetrahedron.")
    ap.add_argument("--edge-samples", type=int, default=DEFAULT_EDGE_SAMPLES,
                    help="Interpolation segments per strut (>=2).")

    # Motion
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED,
                    help="Feedrate for non-print travel moves (mm/min).")
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED,
                    help="Coordinated feedrate for printing moves (mm/min).")

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