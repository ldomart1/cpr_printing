#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code for a 3D octet-based truss lattice using
calibrated B/C kinematics with exact tip-position tracking and soft tangent alignment.

PRIORITIES
----------
1) Exact tip position (hard priority):
     p_stage = p_tip_desired - offset(B, C)
   where
     offset(B, C) = [r(B) cos C, r(B) sin C, z(B)]

2) Tangential nozzle/extrusion angle along each strut (soft priority):
   For each straight edge (strut), choose B and C that best align the nozzle axis
   with the edge tangent, using the calibrated tip angle polynomial and C-axis azimuth.

Assumed nozzle axis model (3D extension of planar calibration):
    nozzle_dir(B, C) = [cos(a(B)) cos C, cos(a(B)) sin C, sin(a(B))]
where a(B) is the calibrated tip_angle polynomial (optionally adjusted via CLI).

Geometry
--------
Octet truss "cell" is generated as:
  - 8 cube corners
  - 6 face centers
  - edges connecting each face center to the 4 corners of that face
This is tiled over Nx x Ny x Nz cells and deduplicated globally.

Notes
-----
- Edge order is a simple greedy nearest-endpoint traversal (reasonable for many cases).
- Each strut is printed as a straight line with constant B/C (constant tangent orientation).
- XYZ stage moves are exact for the tip path.
- Optional U-axis extrusion is supported (absolute extrusion mode).

Authoring style intentionally mirrors the provided star script:
- Calibration dataclass
- cubic polynomial eval helpers
- calibration-driven axis mapping
- clear CLI and printed diagnostics
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
DEFAULT_OUT = "gcode_generation/octet_truss_tip_priority_tangent_soft.gcode"

# Placement in tip space (world coordinates)
DEFAULT_ORIGIN_X = 65.0
DEFAULT_ORIGIN_Y = 0.0
DEFAULT_ORIGIN_Z = -110.0

# Geometry
DEFAULT_CELLS_X = 1
DEFAULT_CELLS_Y = 1
DEFAULT_CELLS_Z = 1
DEFAULT_PITCH_MM = 18.0              # cube edge length for one octet cell
DEFAULT_EDGE_SAMPLES = 20            # interpolation points per strut (>=2)
DEFAULT_GREEDY_START_X = 65.0
DEFAULT_GREEDY_START_Y = 0.0
DEFAULT_GREEDY_START_Z = -110.0

# Motion
DEFAULT_TRAVEL_FEED = 1200.0
DEFAULT_PRINT_FEED = 300.0

# Extrusion
DEFAULT_EXTRUSION_PER_MM = 0.0       # set >0 to emit U extrusion
DEFAULT_PRIME_MM = 0.0

# B/C solve
DEFAULT_B_SEARCH_SAMPLES = 721
DEFAULT_B_CONTINUITY_WEIGHT = 0.03   # deg-equivalent per B-unit (soft continuity)
DEFAULT_C_CONTINUITY_WEIGHT = 0.003  # deg-equivalent per C-degree (soft continuity)

# Tip-angle convention tweak (for calibration conventions)
DEFAULT_TIP_ANGLE_SIGN = 1.0
DEFAULT_TIP_ANGLE_OFFSET_DEG = 0.0
# ------------------------------------------------------------


@dataclass
class Calibration:
    pr: np.ndarray            # r(B) coeffs
    pz: np.ndarray            # z(B) coeffs
    pa: np.ndarray            # tip_angle(B) coeffs (deg)

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
    nozzle_dir: np.ndarray    # unit vector (3,)
    offset_vec: np.ndarray    # tip offset vector from stage to tip (3,)
    edge_len: float


def _polyval4(coeffs: ArrayLike, u: ArrayLike) -> np.ndarray:
    a, b, c, d = coeffs
    u = np.asarray(u, dtype=float)
    return ((a * u + b) * u + c) * u + d


def wrap_deg_360(angle_deg: float) -> float:
    return float(angle_deg % 360.0)


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


def angular_distance_deg(a_deg: float, b_deg: float) -> float:
    d = (a_deg - b_deg + 180.0) % 360.0 - 180.0
    return abs(d)


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


def eval_tip_angle_deg(
    cal: Calibration,
    b: ArrayLike,
    angle_sign: float = 1.0,
    angle_offset_deg: float = 0.0,
) -> np.ndarray:
    raw = _polyval4(cal.pa, b)
    return angle_sign * raw + angle_offset_deg


def tip_offset_xyz(cal: Calibration, b: float, c_deg: float) -> np.ndarray:
    """
    Tip position offset from stage origin for a given B/C:
        [r(B) cos C, r(B) sin C, z(B)]
    """
    r = float(eval_r(cal, b))
    z = float(eval_z(cal, b))
    c = math.radians(c_deg)
    return np.array([r * math.cos(c), r * math.sin(c), z], dtype=float)


def nozzle_axis_unit_xyz(
    cal: Calibration,
    b: float,
    c_deg: float,
    angle_sign: float = 1.0,
    angle_offset_deg: float = 0.0,
) -> np.ndarray:
    """
    Assumed nozzle/extrusion axis direction in world frame for a given B/C:
        [cos(a) cos C, cos(a) sin C, sin(a)]
    with a = tip_angle(B) in degrees (possibly sign/offset adjusted).
    """
    a_deg = float(eval_tip_angle_deg(cal, b, angle_sign=angle_sign, angle_offset_deg=angle_offset_deg))
    a = math.radians(a_deg)
    c = math.radians(c_deg)
    v = np.array([math.cos(a) * math.cos(c), math.cos(a) * math.sin(c), math.sin(a)], dtype=float)
    n = np.linalg.norm(v)
    if n <= 0.0:
        raise RuntimeError("Invalid zero-length nozzle axis vector.")
    return v / n


def sampled_ranges(cal: Calibration, b_lo: float, b_hi: float, n: int = 2001) -> dict:
    bb = np.linspace(b_lo, b_hi, n)
    rr = eval_r(cal, bb)
    zz = eval_z(cal, bb)
    aa = eval_tip_angle_deg(cal, bb)
    return {
        "r_min": float(np.min(rr)),
        "r_max": float(np.max(rr)),
        "abs_r_min": float(np.min(np.abs(rr))),
        "abs_r_max": float(np.max(np.abs(rr))),
        "z_min": float(np.min(zz)),
        "z_max": float(np.max(zz)),
        "a_min": float(np.min(aa)),
        "a_max": float(np.max(aa)),
    }


# ---------------- Geometry: Octet Truss ----------------

def _sorted_edge_key(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    return (a, b) if a <= b else (b, a)


def build_octet_truss_edges(
    cells_x: int,
    cells_y: int,
    cells_z: int,
    pitch_mm: float,
    origin_xyz: np.ndarray,
) -> Tuple[np.ndarray, List[Tuple[int, int]], np.ndarray]:
    """
    Build an octet truss lattice using tiled unit cells.

    Integer lattice keys are in half-pitch units (so coordinates are integers).
    Physical position = origin + (key * pitch_mm / 2).

    Unit cell at integer corner base (2i,2j,2k):
      corners: (0/2, 0/2, 0/2)
      face centers:
        (1,1,0), (1,1,2), (1,0,1), (1,2,1), (0,1,1), (2,1,1)
      struts: each face center connected to 4 corners of that face
    """
    if cells_x < 1 or cells_y < 1 or cells_z < 1:
        raise ValueError("cells_x, cells_y, cells_z must be >= 1.")
    if pitch_mm <= 0:
        raise ValueError("pitch_mm must be > 0.")

    node_index: Dict[Tuple[int, int, int], int] = {}
    node_keys: List[Tuple[int, int, int]] = []
    edge_keys = set()

    def add_node(k: Tuple[int, int, int]) -> int:
        if k not in node_index:
            node_index[k] = len(node_keys)
            node_keys.append(k)
        return node_index[k]

    for i in range(cells_x):
        for j in range(cells_y):
            for k in range(cells_z):
                bx, by, bz = 2 * i, 2 * j, 2 * k

                # Local corners
                c000 = (bx + 0, by + 0, bz + 0)
                c200 = (bx + 2, by + 0, bz + 0)
                c020 = (bx + 0, by + 2, bz + 0)
                c220 = (bx + 2, by + 2, bz + 0)

                c002 = (bx + 0, by + 0, bz + 2)
                c202 = (bx + 2, by + 0, bz + 2)
                c022 = (bx + 0, by + 2, bz + 2)
                c222 = (bx + 2, by + 2, bz + 2)

                # Face centers
                fz0 = (bx + 1, by + 1, bz + 0)
                fz2 = (bx + 1, by + 1, bz + 2)

                fy0 = (bx + 1, by + 0, bz + 1)
                fy2 = (bx + 1, by + 2, bz + 1)

                fx0 = (bx + 0, by + 1, bz + 1)
                fx2 = (bx + 2, by + 1, bz + 1)

                # Face-center -> 4 corners on each face
                face_connections = [
                    (fz0, [c000, c200, c020, c220]),  # z = 0 face
                    (fz2, [c002, c202, c022, c222]),  # z = 2 face
                    (fy0, [c000, c200, c002, c202]),  # y = 0 face
                    (fy2, [c020, c220, c022, c222]),  # y = 2 face
                    (fx0, [c000, c020, c002, c022]),  # x = 0 face
                    (fx2, [c200, c220, c202, c222]),  # x = 2 face
                ]

                for center, corners in face_connections:
                    add_node(center)
                    for corner in corners:
                        add_node(corner)
                        edge_keys.add(_sorted_edge_key(center, corner))

    # Convert node keys to physical positions
    scale = pitch_mm / 2.0
    node_positions = np.zeros((len(node_keys), 3), dtype=float)
    for idx, k in enumerate(node_keys):
        node_positions[idx, :] = origin_xyz + scale * np.array(k, dtype=float)

    # Convert edges to index pairs
    edges_idx: List[Tuple[int, int]] = []
    for ka, kb in sorted(edge_keys):
        ia = node_index[ka]
        ib = node_index[kb]
        if ia != ib:
            edges_idx.append((ia, ib))

    # Also return node_keys as array for optional debugging
    node_keys_arr = np.array(node_keys, dtype=int)
    return node_positions, edges_idx, node_keys_arr


# ---------------- Path planning / tangent solve ----------------

def choose_nearest_edge_greedy(
    remaining_edge_indices: List[int],
    edges_idx: List[Tuple[int, int]],
    nodes_xyz: np.ndarray,
    current_tip: Optional[np.ndarray],
) -> int:
    """
    Choose the undirected edge whose closer endpoint is nearest to current_tip.
    If current_tip is None, return first edge.
    """
    if not remaining_edge_indices:
        raise ValueError("No remaining edges.")
    if current_tip is None:
        return remaining_edge_indices[0]

    best_ei = remaining_edge_indices[0]
    best_d = float("inf")
    for ei in remaining_edge_indices:
        ia, ib = edges_idx[ei]
        pa = nodes_xyz[ia]
        pb = nodes_xyz[ib]
        d = min(float(np.linalg.norm(pa - current_tip)), float(np.linalg.norm(pb - current_tip)))
        if d < best_d:
            best_d = d
            best_ei = ei
    return best_ei


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
) -> EdgePlan:
    """
    Solve B/C for a *directed* strut (p0 -> p1), exact tip tracking hard, tangent alignment soft.

    Strategy:
      - tangent target t = (p1-p0)/||...||
      - sample B over [b_lo, b_hi]
      - for each B, choose C that best matches target horizontal azimuth (with sign handled by cos(a))
      - cost = angle_error_deg + continuity penalties
      - choose minimum-cost B/C
      - compute constant offset vector for exact tip tracking

    Exact tip tracking is achieved later via stage = tip - offset(B,C).
    """
    v = p1_tip - p0_tip
    L = float(np.linalg.norm(v))
    if L <= 1e-12:
        raise ValueError("Zero-length edge encountered.")
    t = v / L

    # Desired horizontal azimuth
    horiz = float(math.hypot(float(t[0]), float(t[1])))
    has_horiz = horiz > 1e-12
    phi_deg = math.degrees(math.atan2(float(t[1]), float(t[0]))) if has_horiz else 0.0

    n_samples = max(11, int(b_search_samples))
    bb = np.linspace(float(b_lo), float(b_hi), n_samples)

    a_deg = eval_tip_angle_deg(cal, bb, angle_sign=angle_sign, angle_offset_deg=angle_offset_deg)
    a_rad = np.deg2rad(a_deg)
    cos_a = np.cos(a_rad)
    sin_a = np.sin(a_rad)

    # Choose C to align horizontal direction to edge tangent.
    # If cos(a) < 0, the horizontal component points opposite the C azimuth,
    # so add 180° to C to keep the nozzle axis horizontal projection aligned.
    if has_horiz:
        c_base = np.where(cos_a >= 0.0, phi_deg, phi_deg + 180.0)
    else:
        c0 = 0.0 if prev_c is None else float(prev_c)
        c_base = np.full_like(bb, c0, dtype=float)

    c_cmd = unwrap_deg_near(c_base, prev_c)  # minimize spin relative to previous C
    c_rad = np.deg2rad(c_cmd)

    # Nozzle direction candidates
    nx = cos_a * np.cos(c_rad)
    ny = cos_a * np.sin(c_rad)
    nz = sin_a

    dots = nx * t[0] + ny * t[1] + nz * t[2]
    dots = np.clip(dots, -1.0, 1.0)
    angle_err_deg = np.rad2deg(np.arccos(dots))

    cost = angle_err_deg.astype(float).copy()

    # Soft continuity penalties (lower priority than angle matching)
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

    offset_vec = tip_offset_xyz(cal, b_star, c_star)

    # Sanity check of exact tip-reconstruction algebra at endpoints
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
) -> Tuple[EdgePlan, float]:
    """
    Evaluate both orientations (a->b, b->a), then pick using:
      primary: lower total (travel + soft angle/continuity inside plan)
      where the plan already includes angle/continuity penalties in its B/C solve,
      and here we add travel-to-start distance as a practical routing penalty.

    Returns (best_plan, travel_distance_to_start).
    """
    plan_ab = solve_directed_edge_tangent_soft(
        p0_tip=pa_tip, p1_tip=pb_tip,
        cal=cal, b_lo=b_lo, b_hi=b_hi,
        b_search_samples=b_search_samples,
        prev_b=prev_b, prev_c=prev_c,
        angle_sign=angle_sign, angle_offset_deg=angle_offset_deg,
        b_cont_weight=b_cont_weight, c_cont_weight=c_cont_weight,
    )
    plan_ba = solve_directed_edge_tangent_soft(
        p0_tip=pb_tip, p1_tip=pa_tip,
        cal=cal, b_lo=b_lo, b_hi=b_hi,
        b_search_samples=b_search_samples,
        prev_b=prev_b, prev_c=prev_c,
        angle_sign=angle_sign, angle_offset_deg=angle_offset_deg,
        b_cont_weight=b_cont_weight, c_cont_weight=c_cont_weight,
    )

    travel_ab = 0.0 if current_tip is None else float(np.linalg.norm(plan_ab.p0_tip - current_tip))
    travel_ba = 0.0 if current_tip is None else float(np.linalg.norm(plan_ba.p0_tip - current_tip))

    # Routing penalty: prioritize shorter travel, but keep angle/continuity from edge solve itself.
    # Here we combine as a simple additive cost in mm + (deg-like angle metric scaled lightly).
    # Since angle is "soft" and position is exact, we let travel dominate practical ordering.
    cost_ab = travel_ab + 0.10 * plan_ab.angle_error_deg
    cost_ba = travel_ba + 0.10 * plan_ba.angle_error_deg

    if cost_ab <= cost_ba:
        return plan_ab, travel_ab
    return plan_ba, travel_ba


def plan_all_edges_greedy(
    nodes_xyz: np.ndarray,
    edges_idx: List[Tuple[int, int]],
    cal: Calibration,
    b_lo: float,
    b_hi: float,
    b_search_samples: int,
    start_tip_xyz: Optional[np.ndarray],
    angle_sign: float,
    angle_offset_deg: float,
    b_cont_weight: float,
    c_cont_weight: float,
) -> Tuple[List[EdgePlan], dict]:
    """
    Greedy traversal:
      1) pick nearest undirected edge (by nearest endpoint to current tip)
      2) choose orientation + B/C (soft tangent optimization)
      3) continue from chosen edge end

    Returns plans and summary metadata.
    """
    remaining = list(range(len(edges_idx)))
    plans: List[EdgePlan] = []

    current_tip = None if start_tip_xyz is None else np.array(start_tip_xyz, dtype=float)
    prev_b: Optional[float] = None
    prev_c: Optional[float] = None

    total_tip_travel = 0.0
    total_print_len = 0.0

    while remaining:
        ei = choose_nearest_edge_greedy(remaining, edges_idx, nodes_xyz, current_tip)
        ia, ib = edges_idx[ei]
        pa = nodes_xyz[ia]
        pb = nodes_xyz[ib]

        plan, travel_to_start = choose_oriented_edge_plan(
            pa_tip=pa, pb_tip=pb,
            cal=cal, b_lo=b_lo, b_hi=b_hi,
            b_search_samples=b_search_samples,
            prev_b=prev_b, prev_c=prev_c,
            current_tip=current_tip,
            angle_sign=angle_sign, angle_offset_deg=angle_offset_deg,
            b_cont_weight=b_cont_weight, c_cont_weight=c_cont_weight,
        )

        plans.append(plan)
        total_tip_travel += float(travel_to_start)
        total_print_len += float(plan.edge_len)

        current_tip = plan.p1_tip.copy()
        prev_b = plan.b_cmd
        prev_c = plan.c_cmd

        remaining.remove(ei)

    angle_errs = np.array([p.angle_error_deg for p in plans], dtype=float)
    bs = np.array([p.b_cmd for p in plans], dtype=float)
    cs = np.array([p.c_cmd for p in plans], dtype=float)

    meta = {
        "n_edges": len(plans),
        "total_tip_travel_mm": float(total_tip_travel),
        "total_print_length_mm": float(total_print_len),
        "angle_error_min_deg": float(np.min(angle_errs)) if len(angle_errs) else 0.0,
        "angle_error_mean_deg": float(np.mean(angle_errs)) if len(angle_errs) else 0.0,
        "angle_error_max_deg": float(np.max(angle_errs)) if len(angle_errs) else 0.0,
        "b_min_used": float(np.min(bs)) if len(bs) else 0.0,
        "b_max_used": float(np.max(bs)) if len(bs) else 0.0,
        "c_min_used": float(np.min(cs)) if len(cs) else 0.0,
        "c_max_used": float(np.max(cs)) if len(cs) else 0.0,
    }
    return plans, meta


# ---------------- G-code emission ----------------

def _fmt_axes_move(axes_vals: List[Tuple[str, float]]) -> str:
    return " ".join(f"{ax}{val:.3f}" for ax, val in axes_vals)


def write_gcode_octet_truss(
    out_path: str,
    cal: Calibration,
    plans: List[EdgePlan],
    edge_samples: int,
    travel_feed: float,
    print_feed: float,
    extrusion_per_mm: float,
    prime_mm: float,
    emit_extrusion: bool,
    header_meta: dict,
    angle_sign: float,
    angle_offset_deg: float,
):
    """
    Emit G-code with:
      - absolute positioning (G90)
      - optional absolute extrusion (M82 / G92 U0)
      - for each edge:
          travel to edge start (tip-exact with chosen B/C)
          print along edge with constant B/C, exact tip path via stage XYZ
    """
    if edge_samples < 2:
        edge_samples = 2

    u_abs = 0.0

    with open(out_path, "w") as f:
        f.write("; generated by octet_truss_tip_priority_tangent_soft.py\n")
        f.write("; strategy: exact tip position (hard) + tangent/nozzle angle alignment (soft)\n")
        f.write("; 3D model assumptions:\n")
        f.write(";   tip_offset(B,C) = [r(B) cos C, r(B) sin C, z(B)]\n")
        f.write(";   nozzle_dir(B,C) = [cos(a(B)) cos C, cos(a(B)) sin C, sin(a(B))]\n")
        f.write(f";   a(B) = ({angle_sign:+.3f}) * tip_angle_poly(B) + {angle_offset_deg:+.3f} deg\n")
        f.write(f"; axes: X->{cal.x_axis}, Y->{cal.y_axis}, Z->{cal.z_axis}, B->{cal.b_axis}, C->{cal.c_axis}, U->{cal.u_axis}\n")
        f.write(f"; edges={header_meta.get('n_edges', 0)}\n")
        f.write(f"; total print length ~ {header_meta.get('total_print_length_mm', 0.0):.3f} mm\n")
        f.write(f"; total tip travel  ~ {header_meta.get('total_tip_travel_mm', 0.0):.3f} mm\n")
        f.write(f"; tangent angle error [min/mean/max] = "
                f"{header_meta.get('angle_error_min_deg', 0.0):.3f} / "
                f"{header_meta.get('angle_error_mean_deg', 0.0):.3f} / "
                f"{header_meta.get('angle_error_max_deg', 0.0):.3f} deg\n")
        f.write(f"; B used range = [{header_meta.get('b_min_used', 0.0):.3f}, {header_meta.get('b_max_used', 0.0):.3f}]\n")
        f.write(f"; C used range = [{header_meta.get('c_min_used', 0.0):.3f}, {header_meta.get('c_max_used', 0.0):.3f}]\n")
        f.write("G90\n")
        if emit_extrusion:
            f.write("M82\n")
            f.write(f"G92 {cal.u_axis}0\n")
            if abs(prime_mm) > 0.0:
                u_abs += float(prime_mm)
                f.write(f"G1 {cal.u_axis}{u_abs:.3f} F{max(60.0, travel_feed):.0f} ; prime\n")

        first_edge = True

        for ei, plan in enumerate(plans):
            d = plan.offset_vec
            b = plan.b_cmd
            c = plan.c_cmd

            # Tip endpoints
            p0_tip = plan.p0_tip
            p1_tip = plan.p1_tip

            # Stage endpoints (exact tip tracking)
            p0_stage = p0_tip - d
            p1_stage = p1_tip - d

            # Travel to edge start with chosen B/C
            f.write(f"; --- edge {ei+1}/{len(plans)} ---\n")
            f.write(f"; len={plan.edge_len:.3f} mm, angle_error={plan.angle_error_deg:.3f} deg, B={b:.3f}, C={c:.3f}\n")

            travel_axes = [
                (cal.x_axis, float(p0_stage[0])),
                (cal.y_axis, float(p0_stage[1])),
                (cal.z_axis, float(p0_stage[2])),
                (cal.b_axis, float(b)),
                (cal.c_axis, float(c)),
            ]

            if first_edge:
                f.write("; startup move to first strut start\n")
                f.write(f"G1 {_fmt_axes_move(travel_axes)} F{travel_feed:.0f}\n")
                first_edge = False
            else:
                f.write("; non-print travel to next strut start (tip exact)\n")
                f.write(f"G1 {_fmt_axes_move(travel_axes)} F{travel_feed:.0f}\n")

            # Print the edge (constant B/C, linearly sampled in tip space)
            # Since d is constant for this edge, stage path is also a straight line.
            ts = np.linspace(0.0, 1.0, edge_samples + 1)
            edge_vec_tip = p1_tip - p0_tip
            last_stage = p0_stage.copy()

            for j in range(1, len(ts)):
                t = float(ts[j])
                p_tip = p0_tip + t * edge_vec_tip
                p_stage = p_tip - d

                # Exact tip consistency check
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
                    seg_len = float(np.linalg.norm(p_stage - last_stage))  # equals tip segment length here
                    u_abs += float(extrusion_per_mm) * seg_len
                    move_axes.append((cal.u_axis, float(u_abs)))

                f.write(f"G1 {_fmt_axes_move(move_axes)} F{print_feed:.0f}\n")
                last_stage = p_stage

        f.write("; --- end ---\n")


# ---------------- Main ----------------

def main(args):
    cal = load_calibration(args.calibration)

    b_lo = cal.b_min if args.min_b is None else float(args.min_b)
    b_hi = cal.b_max if args.max_b is None else float(args.max_b)
    if b_lo > b_hi:
        b_lo, b_hi = b_hi, b_lo

    origin_xyz = np.array([float(args.origin_x), float(args.origin_y), float(args.origin_z)], dtype=float)
    greedy_start = np.array([float(args.start_x), float(args.start_y), float(args.start_z)], dtype=float)

    nodes_xyz, edges_idx, _node_keys = build_octet_truss_edges(
        cells_x=int(args.cells_x),
        cells_y=int(args.cells_y),
        cells_z=int(args.cells_z),
        pitch_mm=float(args.pitch),
        origin_xyz=origin_xyz,
    )

    plans, plan_meta = plan_all_edges_greedy(
        nodes_xyz=nodes_xyz,
        edges_idx=edges_idx,
        cal=cal,
        b_lo=b_lo,
        b_hi=b_hi,
        b_search_samples=int(args.b_search_samples),
        start_tip_xyz=greedy_start,
        angle_sign=float(args.tip_angle_sign),
        angle_offset_deg=float(args.tip_angle_offset_deg),
        b_cont_weight=float(args.b_continuity_weight),
        c_cont_weight=float(args.c_continuity_weight),
    )

    emit_extrusion = float(args.extrusion_per_mm) != 0.0

    write_gcode_octet_truss(
        out_path=args.out,
        cal=cal,
        plans=plans,
        edge_samples=int(args.edge_samples),
        travel_feed=float(args.travel_feed),
        print_feed=float(args.print_feed),
        extrusion_per_mm=float(args.extrusion_per_mm),
        prime_mm=float(args.prime_mm),
        emit_extrusion=emit_extrusion,
        header_meta=plan_meta,
        angle_sign=float(args.tip_angle_sign),
        angle_offset_deg=float(args.tip_angle_offset_deg),
    )

    ranges = sampled_ranges(cal, b_lo, b_hi)

    print(f"Wrote {args.out}")
    print("Mode: 3D octet truss, exact tip tracking (hard), tangent alignment by B/C (soft)")
    print(f"Cells: {args.cells_x} x {args.cells_y} x {args.cells_z}")
    print(f"Pitch: {args.pitch:.3f} mm")
    print(f"Nodes: {len(nodes_xyz)}")
    print(f"Edges (struts): {len(edges_idx)}")
    print(f"Edge samples: {args.edge_samples}")
    print(f"Calibration B range: [{cal.b_min:.3f}, {cal.b_max:.3f}]")
    print(f"Commanded B range:  [{b_lo:.3f}, {b_hi:.3f}]")
    print(f"Sampled r(B):       [{ranges['r_min']:.3f}, {ranges['r_max']:.3f}] mm   |r| max={ranges['abs_r_max']:.3f}")
    print(f"Sampled z(B):       [{ranges['z_min']:.3f}, {ranges['z_max']:.3f}] mm")
    print(f"Sampled tip angle:  [{ranges['a_min']:.3f}, {ranges['a_max']:.3f}] deg  (before sign/offset CLI adjustment report)")
    print(f"B used range:       [{plan_meta['b_min_used']:.3f}, {plan_meta['b_max_used']:.3f}]")
    print(f"C used range:       [{plan_meta['c_min_used']:.3f}, {plan_meta['c_max_used']:.3f}]")
    print(f"Total print length: {plan_meta['total_print_length_mm']:.3f} mm")
    print(f"Total tip travel:   {plan_meta['total_tip_travel_mm']:.3f} mm")
    print(f"Tangent error min/mean/max: "
          f"{plan_meta['angle_error_min_deg']:.3f} / "
          f"{plan_meta['angle_error_mean_deg']:.3f} / "
          f"{plan_meta['angle_error_max_deg']:.3f} deg")
    if emit_extrusion:
        print(f"Extrusion: U-axis absolute, {args.extrusion_per_mm:.6f} U/mm, prime={args.prime_mm:.3f}")
    else:
        print("Extrusion: disabled (set --extrusion-per-mm > 0 to emit U-axis extrusion commands)")
    print(f"Tip angle convention: a(B) = {args.tip_angle_sign:+.3f}*tip_angle_poly(B) + {args.tip_angle_offset_deg:+.3f} deg")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate G-code for a 3D octet-based truss lattice using calibrated B/C kinematics with exact tip tracking and soft tangent alignment."
    )
    ap.add_argument("--calibration", required=True, help="Path to calibration JSON.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code file.")

    # Placement
    ap.add_argument("--origin-x", type=float, default=DEFAULT_ORIGIN_X,
                    help="Tip-space X of lattice minimum corner (cell grid origin).")
    ap.add_argument("--origin-y", type=float, default=DEFAULT_ORIGIN_Y,
                    help="Tip-space Y of lattice minimum corner (cell grid origin).")
    ap.add_argument("--origin-z", type=float, default=DEFAULT_ORIGIN_Z,
                    help="Tip-space Z of lattice minimum corner (cell grid origin).")

    # Geometry
    ap.add_argument("--cells-x", type=int, default=DEFAULT_CELLS_X, help="Number of octet cells in X.")
    ap.add_argument("--cells-y", type=int, default=DEFAULT_CELLS_Y, help="Number of octet cells in Y.")
    ap.add_argument("--cells-z", type=int, default=DEFAULT_CELLS_Z, help="Number of octet cells in Z.")
    ap.add_argument("--pitch", type=float, default=DEFAULT_PITCH_MM,
                    help="Cube edge length (mm) of one octet unit cell.")
    ap.add_argument("--edge-samples", type=int, default=DEFAULT_EDGE_SAMPLES,
                    help="Interpolation segments per strut (>=2).")

    # Greedy pathing start hint (tip space)
    ap.add_argument("--start-x", type=float, default=DEFAULT_GREEDY_START_X,
                    help="Greedy planner start hint tip X.")
    ap.add_argument("--start-y", type=float, default=DEFAULT_GREEDY_START_Y,
                    help="Greedy planner start hint tip Y.")
    ap.add_argument("--start-z", type=float, default=DEFAULT_GREEDY_START_Z,
                    help="Greedy planner start hint tip Z.")

    # Motion
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED,
                    help="Feedrate for non-print travel moves.")
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED,
                    help="Feedrate for printing moves.")

    # Extrusion
    ap.add_argument("--extrusion-per-mm", type=float, default=DEFAULT_EXTRUSION_PER_MM,
                    help="Absolute U-axis extrusion increment per mm of printed strut. Set 0 to disable U.")
    ap.add_argument("--prime-mm", type=float, default=DEFAULT_PRIME_MM,
                    help="Optional prime amount on U axis at start (absolute extrusion mode).")

    # B/C solve
    ap.add_argument("--min-b", type=float, default=None, help="Lower commanded B bound (default: calibration).")
    ap.add_argument("--max-b", type=float, default=None, help="Upper commanded B bound (default: calibration).")
    ap.add_argument("--b-search-samples", type=int, default=DEFAULT_B_SEARCH_SAMPLES,
                    help="Number of B samples used when solving tangent alignment per edge.")
    ap.add_argument("--b-continuity-weight", type=float, default=DEFAULT_B_CONTINUITY_WEIGHT,
                    help="Soft continuity weight on ΔB in the tangent solver.")
    ap.add_argument("--c-continuity-weight", type=float, default=DEFAULT_C_CONTINUITY_WEIGHT,
                    help="Soft continuity weight on ΔC in the tangent solver.")

    # Tip-angle convention tweaks
    ap.add_argument("--tip-angle-sign", type=float, default=DEFAULT_TIP_ANGLE_SIGN,
                    help="Multiplier applied to calibrated tip_angle(B) before tangent solve/model.")
    ap.add_argument("--tip-angle-offset-deg", type=float, default=DEFAULT_TIP_ANGLE_OFFSET_DEG,
                    help="Offset (deg) added to calibrated tip_angle(B) before tangent solve/model.")

    args = ap.parse_args()
    main(args)