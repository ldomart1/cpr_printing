#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code for an octet truss lattice housed inside a cube,
printed in simple Cartesian bottom-up order with fixed B/C.

This version adds:
- Fine speed ramp near nodes (default: slow down within 5 mm of either node)
- Strut end extension into node direction (default: +1.5 mm beyond end node)
- Node revisit routine:
    after over-extending the strut, lift up, move back to the true node position,
    lower down, and dwell while extruding in place

Behavior:
- Fixed B and C throughout the print (defaults: B=0, C=0)
- No per-edge tangent / nozzle-direction solve
- No B/C recentering logic
- No special tip-tracked B/C pivot travel logic
- Edges are ordered from bottom up in Cartesian space
- Stage XYZ is still computed from calibrated tip offset:
      p_stage = p_tip - offset_tip(B_fixed, C_fixed)
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PchipInterpolator


# ---------------- Defaults (CLI-overridable) ----------------
DEFAULT_OUT = "gcode_generation/octet_truss_cartesian_topdown.gcode"

# Placement in tip space (world coordinates): lower/min cube corner
DEFAULT_ORIGIN_X = 40.0
DEFAULT_ORIGIN_Y = 52.0
DEFAULT_ORIGIN_Z = -200.0

# Geometry
DEFAULT_ORDER = 3
DEFAULT_PITCH_MM = 10.0
DEFAULT_EDGE_SAMPLES = 12
DEFAULT_INCLUDE_CUBE_FRAME = True

# Motion
DEFAULT_TRAVEL_FEED = 2000.0      # mm/min
DEFAULT_PRINT_FEED = 150.0        # mm/min
DEFAULT_FINE_PRINT_FEED = 50.0    # mm/min when very close to node
DEFAULT_FINE_RAMP_DISTANCE_MM = 5.0

# Strut overprint into node
DEFAULT_NODE_EXTENSION_MM = 1.5   # mm beyond the end node along the strut direction

# Node revisit / reinforcement
DEFAULT_NODE_REVISIT_LIFT_MM = 2.0
DEFAULT_NODE_DWELL_EXTRUSION_MM = 0.15   # additional U extrusion while dwelling on node

# User-defined pre/post print poses (stage axes)
DEFAULT_START_X = 40.0
DEFAULT_START_Y = 52.0
DEFAULT_START_Z = -20.0
DEFAULT_END_X = 40.0
DEFAULT_END_Y = 52.0
DEFAULT_END_Z = -20.0
DEFAULT_SAFE_APPROACH_Z = -100.0

# Virtual XYZ bounding box (enforced during print/travel)
DEFAULT_BBOX_X_MIN = 10.0
DEFAULT_BBOX_X_MAX = 160.0
DEFAULT_BBOX_Y_MIN = 0.0
DEFAULT_BBOX_Y_MAX = 180.0
DEFAULT_BBOX_Z_MIN = -200.0
DEFAULT_BBOX_Z_MAX = -10.0

# Extrusion
DEFAULT_EXTRUSION_PER_MM = 0.005
DEFAULT_PRIME_MM = 1.0

# Pressure offset / dwell sequencing (U-axis)
DEFAULT_PRESSURE_OFFSET_MM = 4.0
DEFAULT_PRESSURE_ADVANCE_FEED = 1000.0
DEFAULT_PRESSURE_RETRACT_FEED = 1000.0
DEFAULT_PREFLOW_DWELL_MS = 1000
DEFAULT_NODE_DWELL_MS = 1500

# Fixed B/C for this simplified version
DEFAULT_FIXED_B = 0.0
DEFAULT_FIXED_C = 0.0

OFFPLANE_SIGN = -1.0
# ------------------------------------------------------------


@dataclass
class CurveModel:
    model_type: str
    coefficients: Optional[np.ndarray] = None
    x_knots: Optional[np.ndarray] = None
    y_knots: Optional[np.ndarray] = None
    equation: Optional[str] = None


@dataclass
class Calibration:
    pull_r_model: CurveModel
    pull_z_model: CurveModel
    pull_y_model: Optional[CurveModel]
    pull_angle_model: CurveModel

    release_r_model: CurveModel
    release_z_model: CurveModel
    release_y_model: Optional[CurveModel]
    release_angle_model: CurveModel

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


def curve_model_eval(model: Optional[CurveModel], u: ArrayLike, default_if_none: Optional[float] = None) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    if model is None:
        if default_if_none is None:
            raise ValueError("Missing curve model.")
        return np.full_like(u, float(default_if_none), dtype=float)

    mt = str(model.model_type).lower()
    if mt == 'polynomial':
        return poly_eval(model.coefficients, u, default_if_none=default_if_none)
    if mt == 'pchip':
        if model.x_knots is None or model.y_knots is None:
            raise ValueError('PCHIP model missing knots.')
        return np.asarray(PchipInterpolator(model.x_knots, model.y_knots, extrapolate=True)(u), dtype=float)

    raise ValueError(f"Unsupported curve model type: {model.model_type}")


def _curve_model_from_spec(spec: Optional[dict]) -> Optional[CurveModel]:
    if spec is None:
        return None
    mt = str(spec.get('model_type', 'polynomial')).lower()
    if mt == 'pchip':
        return CurveModel(
            model_type='pchip',
            x_knots=np.asarray(spec.get('x_knots', []), dtype=float),
            y_knots=np.asarray(spec.get('y_knots', []), dtype=float),
            equation=spec.get('equation'),
        )
    if mt == 'polynomial':
        return CurveModel(
            model_type='polynomial',
            coefficients=np.asarray(spec.get('coefficients', []), dtype=float),
            equation=spec.get('equation'),
        )
    raise ValueError(f"Unsupported curve model type in calibration: {mt}")


def _pick_phase_model(phase_models: dict, base_name: str) -> Optional[CurveModel]:
    for key in (f'{base_name}_pchip', base_name, f'{base_name}_linear', f'{base_name}_cubic'):
        model = _curve_model_from_spec(phase_models.get(key))
        if model is not None:
            return model
    return None


def _pick_phase_entry(data: dict, preferred_names: List[str]) -> Optional[dict]:
    phase_map = data.get('fit_models_by_phase', {}) or {}
    for name in preferred_names:
        entry = phase_map.get(name)
        if isinstance(entry, dict):
            return entry
    return None


def load_calibration(json_path: str) -> Calibration:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")

    with p.open("r") as f:
        data = json.load(f)

    pull_phase = _pick_phase_entry(data, ['pull_1', 'pull'])
    release_phase = _pick_phase_entry(data, ['release_1', 'release'])

    if pull_phase is not None:
        pull_r_model = _pick_phase_model(pull_phase, 'r')
        pull_z_model = _pick_phase_model(pull_phase, 'z')
        pull_y_model = _pick_phase_model(pull_phase, 'offplane_y')
        pull_angle_model = _pick_phase_model(pull_phase, 'tip_angle')

        release_phase = pull_phase if release_phase is None else release_phase
        release_r_model = _pick_phase_model(release_phase, 'r')
        release_z_model = _pick_phase_model(release_phase, 'z')
        release_y_model = _pick_phase_model(release_phase, 'offplane_y')
        release_angle_model = _pick_phase_model(release_phase, 'tip_angle')

        if pull_r_model is None or pull_z_model is None or pull_angle_model is None:
            raise ValueError('Calibration JSON is missing required pull-phase models.')
        if release_r_model is None or release_z_model is None or release_angle_model is None:
            raise ValueError('Calibration JSON is missing required release-phase models.')

        offplane_eq_model = pull_y_model or release_y_model
    else:
        cubic = data['cubic_coefficients']
        pull_r_model = CurveModel(model_type='polynomial', coefficients=np.asarray(cubic['r_coeffs'], dtype=float), equation=cubic.get('r_equation'))
        pull_z_model = CurveModel(model_type='polynomial', coefficients=np.asarray(cubic['z_coeffs'], dtype=float), equation=cubic.get('z_equation'))
        pull_angle_model = CurveModel(model_type='polynomial', coefficients=np.asarray(cubic['tip_angle_coeffs'], dtype=float), equation=cubic.get('tip_angle_equation'))
        py_off_raw = cubic.get('offplane_y_coeffs', None)
        pull_y_model = None if py_off_raw is None else CurveModel(model_type='polynomial', coefficients=np.asarray(py_off_raw, dtype=float), equation=cubic.get('offplane_y_equation'))
        release_r_model = pull_r_model
        release_z_model = pull_z_model
        release_angle_model = pull_angle_model
        release_y_model = pull_y_model
        offplane_eq_model = pull_y_model

    motor_setup = data.get('motor_setup', {})
    duet_map = data.get('duet_axis_mapping', {})

    b_range = motor_setup.get('b_motor_position_range', [-5.4, 0.0])
    b_min, b_max = map(float, b_range)
    if b_min > b_max:
        b_min, b_max = b_max, b_min

    x_axis = str(duet_map.get('horizontal_axis') or motor_setup.get('horizontal_axis') or 'X')
    y_axis = str(duet_map.get('depth_axis') or motor_setup.get('depth_axis') or 'Y')
    z_axis = str(duet_map.get('vertical_axis') or motor_setup.get('vertical_axis') or 'Z')
    b_axis = str(duet_map.get('pull_axis') or motor_setup.get('b_motor_axis') or 'B')
    c_axis = str(duet_map.get('rotation_axis') or motor_setup.get('rotation_axis') or 'C')
    u_axis = str(duet_map.get('extruder_axis') or 'U')
    c_180 = float(motor_setup.get('rotation_axis_180_deg', 180.0))

    return Calibration(
        pull_r_model=pull_r_model,
        pull_z_model=pull_z_model,
        pull_y_model=pull_y_model,
        pull_angle_model=pull_angle_model,
        release_r_model=release_r_model,
        release_z_model=release_z_model,
        release_y_model=release_y_model,
        release_angle_model=release_angle_model,
        b_min=b_min,
        b_max=b_max,
        x_axis=x_axis, y_axis=y_axis, z_axis=z_axis,
        b_axis=b_axis, c_axis=c_axis, u_axis=u_axis,
        c_180_deg=c_180,
        offplane_y_equation=(None if offplane_eq_model is None else offplane_eq_model.equation),
        offplane_y_r_squared=None,
    )


def _phase_curve_model(cal: Calibration, quantity: str, phase: str) -> Optional[CurveModel]:
    phase_key = str(phase).lower()
    if phase_key not in {'pull', 'release'}:
        raise ValueError(f'Unsupported motion phase: {phase}')
    return {
        ('pull', 'r'): cal.pull_r_model,
        ('pull', 'z'): cal.pull_z_model,
        ('pull', 'offplane_y'): cal.pull_y_model,
        ('pull', 'tip_angle'): cal.pull_angle_model,
        ('release', 'r'): cal.release_r_model,
        ('release', 'z'): cal.release_z_model,
        ('release', 'offplane_y'): cal.release_y_model,
        ('release', 'tip_angle'): cal.release_angle_model,
    }[(phase_key, quantity)]


def eval_r(cal: Calibration, b: ArrayLike, phase: str = 'pull') -> np.ndarray:
    return curve_model_eval(_phase_curve_model(cal, 'r', phase), b)


def eval_z(cal: Calibration, b: ArrayLike, phase: str = 'pull') -> np.ndarray:
    return curve_model_eval(_phase_curve_model(cal, 'z', phase), b)


def eval_offplane_y(cal: Calibration, b: ArrayLike, phase: str = 'pull') -> np.ndarray:
    return OFFPLANE_SIGN * curve_model_eval(_phase_curve_model(cal, 'offplane_y', phase), b, default_if_none=0.0)


def predict_r_z_offplane(cal: Calibration, b: float, phase: str = 'pull') -> Tuple[float, float, float]:
    r = float(eval_r(cal, b, phase=phase))
    z = float(eval_z(cal, b, phase=phase))
    y_off = float(eval_offplane_y(cal, b, phase=phase))
    return r, z, y_off


def predict_tip_xyz_from_bc(cal: Calibration, b: float, c_deg: float, phase: str = 'pull') -> np.ndarray:
    r, z, y_off = predict_r_z_offplane(cal, b, phase=phase)
    c = math.radians(c_deg)
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def tip_offset_xyz_physical(cal: Calibration, b: float, c_deg: float, phase: str = 'pull') -> np.ndarray:
    return predict_tip_xyz_from_bc(cal, b, c_deg, phase=phase)


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


# ---------------- Simple fixed-B/C planning ----------------

def plan_all_edges_bottom_up_cartesian(
    nodes_xyz: np.ndarray,
    edges_idx: List[Tuple[int, int]],
    fixed_b: float = 0.0,
    fixed_c: float = 0.0,
) -> Tuple[List[EdgePlan], dict]:
    planned: List[EdgePlan] = []
    sortable = []

    for ia, ib in edges_idx:
        pa = nodes_xyz[ia].copy()
        pb = nodes_xyz[ib].copy()

        if float(pb[2]) < float(pa[2]):
            p0, p1 = pb, pa
        else:
            p0, p1 = pa, pb

        z_max = max(float(p0[2]), float(p1[2]))
        z_min = min(float(p0[2]), float(p1[2]))
        mid = 0.5 * (p0 + p1)

        sortable.append((
            z_min,
            z_max,
            float(mid[1]),
            float(mid[0]),
            p0,
            p1,
        ))

    sortable.sort(key=lambda x: (x[0], x[1], x[2], x[3]))

    total_print_len = 0.0
    total_tip_travel = 0.0
    prev_end = None

    for _, _, _, _, p0, p1 in sortable:
        edge_len = float(np.linalg.norm(p1 - p0))
        if edge_len <= 1e-12:
            continue

        if prev_end is not None:
            total_tip_travel += float(np.linalg.norm(p0 - prev_end))

        planned.append(
            EdgePlan(
                p0_tip=p0,
                p1_tip=p1,
                b_cmd=float(fixed_b),
                c_cmd=float(fixed_c),
                angle_error_deg=0.0,
                nozzle_dir=np.array([0.0, 0.0, -1.0], dtype=float),
                offset_vec=np.zeros(3, dtype=float),
                edge_len=edge_len,
                head_lead_mm=0.0,
                head_front_ok=True,
            )
        )
        total_print_len += edge_len
        prev_end = p1

    meta = {
        "n_edges": len(planned),
        "total_tip_travel_mm": float(total_tip_travel),
        "total_print_length_mm": float(total_print_len),
        "angle_error_min_deg": 0.0,
        "angle_error_mean_deg": 0.0,
        "angle_error_max_deg": 0.0,
        "b_min_used": float(fixed_b),
        "b_max_used": float(fixed_b),
        "c_min_used": float(fixed_c),
        "c_max_used": float(fixed_c),
        "head_front_ok_count": len(planned),
        "head_front_violation_count": 0,
        "head_lead_min_mm": 0.0,
        "head_lead_mean_mm": 0.0,
        "head_lead_max_mm": 0.0,
    }
    return planned, meta


# ---------------- Diagnostics ----------------

def compute_fixed_stage_gantry_ranges(plans: List[EdgePlan], tip_offset: np.ndarray) -> dict:
    if not plans:
        return {
            "x_stage_min": 0.0, "x_stage_max": 0.0,
            "y_stage_min": 0.0, "y_stage_max": 0.0,
            "z_stage_min": 0.0, "z_stage_max": 0.0,
        }

    pts = []
    for p in plans:
        pts.append(p.p0_tip - tip_offset)
        pts.append(p.p1_tip - tip_offset)

    arr = np.vstack(pts)
    return {
        "x_stage_min": float(np.min(arr[:, 0])),
        "x_stage_max": float(np.max(arr[:, 0])),
        "y_stage_min": float(np.min(arr[:, 1])),
        "y_stage_max": float(np.max(arr[:, 1])),
        "z_stage_min": float(np.min(arr[:, 2])),
        "z_stage_max": float(np.max(arr[:, 2])),
    }


# ---------------- Feed ramp helpers ----------------

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def smoothstep01(x: float) -> float:
    x = clamp(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def compute_segment_feedrate(
    distance_from_start_mm: float,
    total_length_mm: float,
    base_feed: float,
    fine_feed: float,
    ramp_distance_mm: float,
) -> float:
    if total_length_mm <= 0.0:
        return float(fine_feed)

    if ramp_distance_mm <= 1e-9 or fine_feed >= base_feed:
        return float(base_feed)

    d_start = float(distance_from_start_mm)
    d_end = float(total_length_mm - distance_from_start_mm)
    d_near = min(d_start, d_end)

    if d_near >= ramp_distance_mm:
        return float(base_feed)

    alpha = smoothstep01(d_near / ramp_distance_mm)
    return float(fine_feed + alpha * (base_feed - fine_feed))


# ---------------- G-code helpers ----------------

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


def write_gcode_octet_truss_cartesian_bottom_up(
    out_path: str,
    cal: Calibration,
    plans: List[EdgePlan],
    edge_samples: int,
    travel_feed: float,
    print_feed: float,
    fine_print_feed: float,
    fine_ramp_distance_mm: float,
    node_extension_mm: float,
    node_revisit_lift_mm: float,
    node_dwell_extrusion_mm: float,
    extrusion_per_mm: float,
    prime_mm: float,
    emit_extrusion: bool,
    pressure_offset_mm: float,
    pressure_advance_feed: float,
    pressure_retract_feed: float,
    preflow_dwell_ms: int,
    node_dwell_ms: int,
    header_meta: dict,
    start_pose: Tuple[float, float, float, float, float],
    end_pose: Tuple[float, float, float, float, float],
    safe_approach_z: float,
    virtual_bbox: dict,
    fixed_b: float = 0.0,
    fixed_c: float = 0.0,
):
    if edge_samples < 2:
        edge_samples = 2

    bbox_warnings: List[str] = []
    u_material_abs = 0.0
    pressure_charged = False

    def u_cmd_actual() -> float:
        return u_material_abs + (float(pressure_offset_mm) if pressure_charged else 0.0)

    tip_offset = tip_offset_xyz_physical(cal, float(fixed_b), float(fixed_c), phase='pull')

    def clamp_xyz(xyz: np.ndarray, context: str) -> np.ndarray:
        x, y, z = _clamp_stage_xyz_to_bbox(
            xyz[0], xyz[1], xyz[2],
            bbox=virtual_bbox,
            context=context,
            warn_log=bbox_warnings,
        )
        return np.array([x, y, z], dtype=float)

    out_parent = Path(out_path).parent
    if str(out_parent) not in ('', '.'):
        out_parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w") as f:
        f.write("; generated by octet_truss_cartesian_topdown.py\n")
        f.write("; mode: bottom-up Cartesian octet print\n")
        f.write("; every sloped print line is oriented low-Z -> high-Z\n")
        f.write(f"; non-print travel uses raise-Z by +{DEFAULT_PITCH_MM:.3f} mm, move-XY, lower-Z\n")
        f.write("; fixed B and C for whole print\n")
        f.write(f"; fixed B = {float(fixed_b):.3f}\n")
        f.write(f"; fixed C = {float(fixed_c):.3f}\n")
        f.write(f"; fixed tip offset = [{tip_offset[0]:.6f}, {tip_offset[1]:.6f}, {tip_offset[2]:.6f}] mm\n")
        f.write(f"; fine_print_feed = {float(fine_print_feed):.3f} mm/min\n")
        f.write(f"; fine_ramp_distance_mm = {float(fine_ramp_distance_mm):.3f} mm\n")
        f.write(f"; node_extension_mm = {float(node_extension_mm):.3f} mm\n")
        f.write(f"; node_revisit_lift_mm = {float(node_revisit_lift_mm):.3f} mm\n")
        f.write(f"; node_dwell_extrusion_mm = {float(node_dwell_extrusion_mm):.3f} mm\n")
        for k, v in header_meta.items():
            f.write(f"; {k}: {v}\n")
        f.write("G90\n")

        if emit_extrusion:
            f.write("M82\n")
            f.write(f"G92 {cal.u_axis}0\n")
            if abs(float(prime_mm)) > 0.0:
                u_material_abs += float(prime_mm)
                f.write(f"G1 {cal.u_axis}{u_cmd_actual():.3f} F{max(60.0, float(pressure_advance_feed)):.0f}\n")

        sx, sy, sz, _, _ = [float(v) for v in start_pose]
        ex, ey, ez, _, _ = [float(v) for v in end_pose]

        f.write("; startup\n")
        f.write(f"G1 {cal.z_axis}{safe_approach_z:.3f} {cal.b_axis}{fixed_b:.3f} {cal.c_axis}{fixed_c:.3f} F{travel_feed:.1f}\n")
        f.write(f"G1 {cal.x_axis}{sx:.3f} {cal.y_axis}{sy:.3f} F{travel_feed:.1f}\n")
        f.write(f"G1 {cal.z_axis}{sz:.3f} F{travel_feed:.1f}\n")

        current_tip = None
        current_stage = np.array([sx, sy, sz], dtype=float)
        emitted_edges = 0

        def emit_travel_move(target_stage: np.ndarray, comment: str):
            nonlocal current_stage
            tx, ty, tz = [float(v) for v in target_stage]
            cx, cy, cz = [float(v) for v in current_stage]
            travel_lift = float(DEFAULT_PITCH_MM)
            travel_z = max(cz, tz) + travel_lift
            travel_stage = clamp_xyz(np.array([tx, ty, travel_z], dtype=float), f"travel {comment}")
            travel_z_clamped = float(travel_stage[2])
            if abs(travel_z_clamped - cz) > 1e-9:
                f.write(f"; travel raise for {comment}\n")
                f.write(f"G1 {cal.z_axis}{travel_z_clamped:.3f} {cal.b_axis}{fixed_b:.3f} {cal.c_axis}{fixed_c:.3f} F{travel_feed:.1f}\n")
                cz = travel_z_clamped
            if abs(tx - cx) > 1e-9 or abs(ty - cy) > 1e-9:
                f.write(f"; travel XY for {comment}\n")
                f.write(f"G1 {cal.x_axis}{tx:.3f} {cal.y_axis}{ty:.3f} {cal.b_axis}{fixed_b:.3f} {cal.c_axis}{fixed_c:.3f} F{travel_feed:.1f}\n")
                cx, cy = tx, ty
            if abs(tz - cz) > 1e-9:
                f.write(f"; travel lower for {comment}\n")
                f.write(f"G1 {cal.z_axis}{tz:.3f} {cal.b_axis}{fixed_b:.3f} {cal.c_axis}{fixed_c:.3f} F{travel_feed:.1f}\n")
            current_stage = np.array([tx, ty, tz], dtype=float)

        def retract_pressure_if_needed():
            nonlocal pressure_charged
            if emit_extrusion and float(pressure_offset_mm) > 0.0 and pressure_charged:
                pressure_charged = False
                f.write(f"G1 {cal.u_axis}{u_cmd_actual():.3f} F{float(pressure_retract_feed):.1f}\n")

        def advance_pressure_if_needed():
            nonlocal pressure_charged
            if emit_extrusion and float(pressure_offset_mm) > 0.0 and not pressure_charged:
                pressure_charged = True
                f.write(f"G1 {cal.u_axis}{u_cmd_actual():.3f} F{float(pressure_advance_feed):.1f}\n")

        def emit_node_revisit(true_node_tip: np.ndarray, comment: str):
            nonlocal current_stage, current_tip, u_material_abs

            true_node_stage = clamp_xyz(true_node_tip - tip_offset, f"{comment} true node")
            cx, cy, cz = [float(v) for v in current_stage]
            tx, ty, tz = [float(v) for v in true_node_stage]

            revisit_lift_z = cz + float(node_revisit_lift_mm)
            lifted_stage = clamp_xyz(np.array([cx, cy, revisit_lift_z], dtype=float), f"{comment} lift")

            if abs(lifted_stage[2] - cz) > 1e-9:
                f.write(f"; node revisit lift for {comment}\n")
                f.write(f"G1 {cal.z_axis}{lifted_stage[2]:.3f} {cal.b_axis}{fixed_b:.3f} {cal.c_axis}{fixed_c:.3f} F{travel_feed:.1f}\n")
                cz = float(lifted_stage[2])

            if abs(tx - cx) > 1e-9 or abs(ty - cy) > 1e-9:
                f.write(f"; node revisit XY for {comment}\n")
                f.write(f"G1 {cal.x_axis}{tx:.3f} {cal.y_axis}{ty:.3f} {cal.b_axis}{fixed_b:.3f} {cal.c_axis}{fixed_c:.3f} F{travel_feed:.1f}\n")
                cx, cy = tx, ty

            if abs(tz - cz) > 1e-9:
                f.write(f"; node revisit lower for {comment}\n")
                f.write(f"G1 {cal.z_axis}{tz:.3f} {cal.b_axis}{fixed_b:.3f} {cal.c_axis}{fixed_c:.3f} F{travel_feed:.1f}\n")

            current_stage = np.array([tx, ty, tz], dtype=float)
            current_tip = true_node_tip.copy()

            if emit_extrusion:
                advance_pressure_if_needed()

                if float(node_dwell_extrusion_mm) > 0.0:
                    u_material_abs += float(node_dwell_extrusion_mm)
                    f.write(f"; node dwell extrusion for {comment}\n")
                    f.write(
                        f"G1 {cal.x_axis}{tx:.3f} {cal.y_axis}{ty:.3f} {cal.z_axis}{tz:.3f} "
                        f"{cal.b_axis}{fixed_b:.3f} {cal.c_axis}{fixed_c:.3f} "
                        f"{cal.u_axis}{u_cmd_actual():.3f} F{max(60.0, float(pressure_advance_feed)):.1f}\n"
                    )

                if int(node_dwell_ms) > 0:
                    f.write(f"G4 P{int(node_dwell_ms)}\n")

                retract_pressure_if_needed()

        for ei, plan in enumerate(plans, start=1):
            p0_tip = plan.p0_tip.copy()
            p1_tip_nominal = plan.p1_tip.copy()

            if float(p1_tip_nominal[2]) < float(p0_tip[2]) - 1e-9:
                raise RuntimeError(f"Edge {ei} descends in Z after planning, which is not allowed.")

            edge_vec = p1_tip_nominal - p0_tip
            edge_len_nominal = float(np.linalg.norm(edge_vec))
            if edge_len_nominal <= 1e-12:
                continue

            edge_dir = edge_vec / edge_len_nominal
            p1_tip_extended = p1_tip_nominal + float(node_extension_mm) * edge_dir
            edge_len_total = float(np.linalg.norm(p1_tip_extended - p0_tip))

            p0_stage = clamp_xyz(p0_tip - tip_offset, f"edge {ei} start")
            p1_stage_extended = clamp_xyz(p1_tip_extended - tip_offset, f"edge {ei} end extended")

            f.write(f"; ---- edge {ei} ----\n")
            f.write(f"; tip start          = [{p0_tip[0]:.3f}, {p0_tip[1]:.3f}, {p0_tip[2]:.3f}]\n")
            f.write(f"; tip end nominal    = [{p1_tip_nominal[0]:.3f}, {p1_tip_nominal[1]:.3f}, {p1_tip_nominal[2]:.3f}]\n")
            f.write(f"; tip end extended   = [{p1_tip_extended[0]:.3f}, {p1_tip_extended[1]:.3f}, {p1_tip_extended[2]:.3f}]\n")
            f.write(f"; edge length nominal= {edge_len_nominal:.3f} mm\n")
            f.write(f"; edge length total  = {edge_len_total:.3f} mm\n")

            if current_tip is None or np.linalg.norm(current_tip - p0_tip) > 1e-9:
                emit_travel_move(p0_stage, f"edge {ei} start")
            else:
                current_stage = p0_stage.copy()

            if emit_extrusion and float(pressure_offset_mm) > 0.0 and not pressure_charged:
                pressure_charged = True
                f.write(f"G1 {cal.u_axis}{u_cmd_actual():.3f} F{float(pressure_advance_feed):.1f}\n")
                if int(preflow_dwell_ms) > 0:
                    f.write(f"G4 P{int(preflow_dwell_ms)}\n")

            prev_tip = p0_tip.copy()

            for k in range(1, edge_samples + 1):
                t = k / float(edge_samples)
                p_tip = (1.0 - t) * p0_tip + t * p1_tip_extended
                p_stage = clamp_xyz(p_tip - tip_offset, f"edge {ei} seg {k}")

                dist_from_start = t * edge_len_total

                seg_feed = compute_segment_feedrate(
                    distance_from_start_mm=dist_from_start,
                    total_length_mm=edge_len_total,
                    base_feed=float(print_feed),
                    fine_feed=float(fine_print_feed),
                    ramp_distance_mm=float(fine_ramp_distance_mm),
                )

                line = (
                    f"G1 {cal.x_axis}{p_stage[0]:.3f} "
                    f"{cal.y_axis}{p_stage[1]:.3f} "
                    f"{cal.z_axis}{p_stage[2]:.3f} "
                    f"{cal.b_axis}{fixed_b:.3f} "
                    f"{cal.c_axis}{fixed_c:.3f}"
                )

                if emit_extrusion:
                    seg_len = float(np.linalg.norm(p_tip - prev_tip))
                    u_material_abs += float(extrusion_per_mm) * seg_len
                    line += f" {cal.u_axis}{u_cmd_actual():.3f}"

                line += f" F{seg_feed:.1f}\n"
                f.write(line)

                prev_tip = p_tip.copy()
                current_stage = p_stage.copy()

            current_tip = p1_tip_extended.copy()
            current_stage = p1_stage_extended.copy()

            # End-of-edge sequence:
            # 1) retract pressure after over-extension
            # 2) lift + move back to actual node
            # 3) lower onto node
            # 4) repressurize and dwell/extrude at true node
            retract_pressure_if_needed()
            emit_node_revisit(p1_tip_nominal, f"edge {ei} node")

            emitted_edges += 1

        f.write("; end\n")
        end_stage = clamp_xyz(np.array([ex, ey, ez], dtype=float), "end pose")
        emit_travel_move(end_stage, "end pose")
        f.write(f"; emitted edges = {emitted_edges}\n")

        if bbox_warnings:
            f.write("; bbox warnings:\n")
            for msg in bbox_warnings:
                f.write(f"; {msg}\n")


# ---------------- CLI ----------------

def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate bottom-up Cartesian octet-truss G-code with fixed B/C.")

    p.add_argument("--calibration", required=True, help="Path to calibration JSON.")
    p.add_argument("--out", default=DEFAULT_OUT, help="Output G-code path.")

    p.add_argument("--origin-x", type=float, default=DEFAULT_ORIGIN_X)
    p.add_argument("--origin-y", type=float, default=DEFAULT_ORIGIN_Y)
    p.add_argument("--origin-z", type=float, default=DEFAULT_ORIGIN_Z)

    p.add_argument("--order", type=int, default=DEFAULT_ORDER)
    p.add_argument("--pitch", type=float, default=DEFAULT_PITCH_MM)
    p.add_argument("--edge-samples", type=int, default=DEFAULT_EDGE_SAMPLES)
    p.add_argument("--include-cube-frame", action="store_true", default=DEFAULT_INCLUDE_CUBE_FRAME)
    p.add_argument("--no-cube-frame", dest="include_cube_frame", action="store_false")

    p.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED)
    p.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED)
    p.add_argument("--fine-print-feed", type=float, default=DEFAULT_FINE_PRINT_FEED)
    p.add_argument("--fine-ramp-distance-mm", type=float, default=DEFAULT_FINE_RAMP_DISTANCE_MM)
    p.add_argument("--node-extension-mm", type=float, default=DEFAULT_NODE_EXTENSION_MM)
    p.add_argument("--node-revisit-lift-mm", type=float, default=DEFAULT_NODE_REVISIT_LIFT_MM)
    p.add_argument("--node-dwell-extrusion-mm", type=float, default=DEFAULT_NODE_DWELL_EXTRUSION_MM)

    p.add_argument("--start-x", type=float, default=DEFAULT_START_X)
    p.add_argument("--start-y", type=float, default=DEFAULT_START_Y)
    p.add_argument("--start-z", type=float, default=DEFAULT_START_Z)
    p.add_argument("--end-x", type=float, default=DEFAULT_END_X)
    p.add_argument("--end-y", type=float, default=DEFAULT_END_Y)
    p.add_argument("--end-z", type=float, default=DEFAULT_END_Z)
    p.add_argument("--safe-approach-z", type=float, default=DEFAULT_SAFE_APPROACH_Z)

    p.add_argument("--bbox-x-min", type=float, default=DEFAULT_BBOX_X_MIN)
    p.add_argument("--bbox-x-max", type=float, default=DEFAULT_BBOX_X_MAX)
    p.add_argument("--bbox-y-min", type=float, default=DEFAULT_BBOX_Y_MIN)
    p.add_argument("--bbox-y-max", type=float, default=DEFAULT_BBOX_Y_MAX)
    p.add_argument("--bbox-z-min", type=float, default=DEFAULT_BBOX_Z_MIN)
    p.add_argument("--bbox-z-max", type=float, default=DEFAULT_BBOX_Z_MAX)

    p.add_argument("--extrusion-per-mm", type=float, default=DEFAULT_EXTRUSION_PER_MM)
    p.add_argument("--prime-mm", type=float, default=DEFAULT_PRIME_MM)
    p.add_argument("--pressure-offset-mm", type=float, default=DEFAULT_PRESSURE_OFFSET_MM)
    p.add_argument("--pressure-advance-feed", type=float, default=DEFAULT_PRESSURE_ADVANCE_FEED)
    p.add_argument("--pressure-retract-feed", type=float, default=DEFAULT_PRESSURE_RETRACT_FEED)
    p.add_argument("--preflow-dwell-ms", type=int, default=DEFAULT_PREFLOW_DWELL_MS)
    p.add_argument("--node-dwell-ms", type=int, default=DEFAULT_NODE_DWELL_MS)

    p.add_argument("--fixed-b", type=float, default=DEFAULT_FIXED_B, help="Fixed B for whole print.")
    p.add_argument("--fixed-c", type=float, default=DEFAULT_FIXED_C, help="Fixed C for whole print.")

    return p


def main(args):
    cal = load_calibration(args.calibration)

    origin_xyz = np.array([
        float(args.origin_x),
        float(args.origin_y),
        float(args.origin_z),
    ], dtype=float)

    nodes_xyz, edges_idx, geom_meta = build_octet_lattice_cube_edges(
        order=int(args.order),
        pitch_mm=float(args.pitch),
        origin_xyz=origin_xyz,
        include_cube_frame=bool(args.include_cube_frame),
    )

    fixed_b = float(args.fixed_b)
    fixed_c = float(args.fixed_c)

    plans, plan_meta = plan_all_edges_bottom_up_cartesian(
        nodes_xyz=nodes_xyz,
        edges_idx=edges_idx,
        fixed_b=fixed_b,
        fixed_c=fixed_c,
    )

    fixed_offset = tip_offset_xyz_physical(cal, fixed_b, fixed_c, phase='pull')
    gantry_meta = compute_fixed_stage_gantry_ranges(plans, fixed_offset)

    header_meta = {}
    header_meta.update(geom_meta)
    header_meta.update(plan_meta)
    header_meta.update(gantry_meta)

    start_pose = (
        float(args.start_x), float(args.start_y), float(args.start_z),
        fixed_b, fixed_c,
    )
    end_pose = (
        float(args.end_x), float(args.end_y), float(args.end_z),
        fixed_b, fixed_c,
    )
    bbox = {
        "x_min": float(args.bbox_x_min),
        "x_max": float(args.bbox_x_max),
        "y_min": float(args.bbox_y_min),
        "y_max": float(args.bbox_y_max),
        "z_min": float(args.bbox_z_min),
        "z_max": float(args.bbox_z_max),
    }

    write_gcode_octet_truss_cartesian_bottom_up(
        out_path=str(args.out),
        cal=cal,
        plans=plans,
        edge_samples=int(args.edge_samples),
        travel_feed=float(args.travel_feed),
        print_feed=float(args.print_feed),
        fine_print_feed=float(args.fine_print_feed),
        fine_ramp_distance_mm=float(args.fine_ramp_distance_mm),
        node_extension_mm=float(args.node_extension_mm),
        node_revisit_lift_mm=float(args.node_revisit_lift_mm),
        node_dwell_extrusion_mm=float(args.node_dwell_extrusion_mm),
        extrusion_per_mm=float(args.extrusion_per_mm),
        prime_mm=float(args.prime_mm),
        emit_extrusion=bool(float(args.extrusion_per_mm) > 0.0),
        pressure_offset_mm=float(args.pressure_offset_mm),
        pressure_advance_feed=float(args.pressure_advance_feed),
        pressure_retract_feed=float(args.pressure_retract_feed),
        preflow_dwell_ms=int(args.preflow_dwell_ms),
        node_dwell_ms=int(args.node_dwell_ms),
        header_meta=header_meta,
        start_pose=start_pose,
        end_pose=end_pose,
        safe_approach_z=float(args.safe_approach_z),
        virtual_bbox=bbox,
        fixed_b=fixed_b,
        fixed_c=fixed_c,
    )

    print(f"Wrote G-code to: {args.out}")
    print(f"Edges: {header_meta['n_edges']}")
    print(f"Fixed tip offset @ B={fixed_b:.3f}, C={fixed_c:.3f}: {fixed_offset}")
    print(f"Fine node ramp distance: {float(args.fine_ramp_distance_mm):.3f} mm")
    print(f"Fine print feed: {float(args.fine_print_feed):.3f} mm/min")
    print(f"Node extension: {float(args.node_extension_mm):.3f} mm")
    print(f"Node revisit lift: {float(args.node_revisit_lift_mm):.3f} mm")
    print(f"Node dwell extrusion: {float(args.node_dwell_extrusion_mm):.3f} mm")


if __name__ == "__main__":
    parser = build_arg_parser()
    main(parser.parse_args())