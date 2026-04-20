#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code from a vessel centerline text file using the
same calibration-based tip-position planning idea used in the octet script:

    stage_xyz = desired_tip_xyz - offset_tip(B, C)

This version is tailored for branched vessel centerlines:
  - parses blocks like: "Vessel: i, Number of Points: N"
  - writes major/root vessels first
  - writes smaller branch vessels afterward, starting from their attachment node
  - orients each vessel bottom-to-top when possible
  - approaches vessel starts from the side to reduce collisions
  - uses a slower/fine approach feedrate near the node
  - optionally varies B/C from the local vessel tangent so the tip follows the
    local tangent direction
  - emits viewer-friendly VASCULATURE_WRITE_START / VASCULATURE_WRITE_END
    comment flags only around actual deposition moves

Assumptions / conventions
-------------------------
1) The vessel file rows are:
       x, y, z, radius_like_value, ignored
   The script uses the first 3 columns as XYZ and the 4th column as the vessel
   size for ordering / parent detection. The 5th column is ignored.

2) The calibration JSON follows the same schema as the octet script:
   - cubic_coefficients.r_coeffs
   - cubic_coefficients.z_coeffs
   - cubic_coefficients.offplane_y_coeffs (optional)
   - cubic_coefficients.tip_angle_coeffs (optional)
   - optional fit_models / fit_models_by_phase PCHIP or polynomial models
   - motor_setup.b_motor_position_range
   - duet_axis_mapping / motor_setup axis names

3) B-angle convention used here for tangential writing:
      B = 0 deg   -> tip points along -Z
      B = 90 deg  -> tip is horizontal
      B = 180 deg -> tip points along +Z
   Tangential writing aligns the tool axis to the full 3D vessel tangent, not
   only the XY projection.

4) Tangential writing uses the local 3D centerline tangent:
      C = azimuth of the tangent's XY projection
      B = acos(t_z) in degrees so bottom-to-top vertical writing keeps the tip
      physically pointing down.
   When the tangent is nearly vertical, C is held near the previous value to
   avoid spurious spin. If you want octet-style fixed orientation instead,
   pass:
      --orientation-mode fixed

Example
-------
python vessel_tangent_gcode.py \
  --vessels vessels.txt \
  --calibration calibration.json \
  --out vessels.gcode
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ---------------- Defaults ----------------
DEFAULT_OUT = "gcode_generation/vessels_tangent.gcode"

DEFAULT_MACHINE_START_X = 90.0
DEFAULT_MACHINE_START_Y = 60.0
DEFAULT_MACHINE_START_Z = -50.0
DEFAULT_MACHINE_START_B = 0.0
DEFAULT_MACHINE_START_C = 0.0

DEFAULT_MACHINE_END_X = 100.0
DEFAULT_MACHINE_END_Y = 60.0
DEFAULT_MACHINE_END_Z = -20.0
DEFAULT_MACHINE_END_B = 0.0
DEFAULT_MACHINE_END_C = 0.0

DEFAULT_TRAVEL_FEED = 2000.0            # mm/min
DEFAULT_APPROACH_FEED = 1000.0           # mm/min
DEFAULT_FINE_APPROACH_FEED = 50.0      # mm/min
DEFAULT_PRINT_FEED = 250.0              # mm/min
DEFAULT_C_FEED = 5000.0                 # deg/min
DEFAULT_BC_SOLVE_SAMPLES = 5001
DEFAULT_EXTRUSION_PER_MM = 0.0015       # U per mm of tip path
DEFAULT_PRIME_MM = 0.2

DEFAULT_PRESSURE_OFFSET_MM = 5.0
DEFAULT_PRESSURE_ADVANCE_FEED = 1000.0
DEFAULT_PRESSURE_RETRACT_FEED = 2000.0
DEFAULT_PREFLOW_DWELL_MS = 1000
DEFAULT_NODE_DWELL_MS = 1000

DEFAULT_EDGE_SAMPLES = 1
DEFAULT_NODE_ATTACH_THRESHOLD = 0.30    # mm
DEFAULT_ROOT_RADIUS_FACTOR = 1.0
DEFAULT_SIDE_APPROACH_FAR = 1.0         # mm in tip-space
DEFAULT_SIDE_APPROACH_NEAR = 0.25       # mm in tip-space
DEFAULT_SIDE_RETREAT = 5            # mm in tip-space
DEFAULT_SIDE_LIFT_Z = 5               # mm in tip-space
DEFAULT_TRAVEL_CLEARANCE_ABOVE_PRINTED_Z = 10.0
DEFAULT_TRAVEL_BBOX_MARGIN = 0.0
DEFAULT_TRAVEL_EDGE_CLEARANCE = 5.0
DEFAULT_FINE_APPROACH_DISTANCE = 5.0
DEFAULT_TANGENT_SMOOTH_WINDOW = 6
DEFAULT_CENTERLINE_SMOOTH_WINDOW = 0
DEFAULT_PATH_GEOMETRY_SMOOTH_WINDOW = 0
DEFAULT_PATH_RESAMPLE_SPACING = 0.0
DEFAULT_SKELETON_COLLISION_CLEARANCE = 0.25
DEFAULT_SKELETON_COLLISION_SAMPLE_STEP_MM = 1.0
DEFAULT_B_MAX_STEP_DEG = 10.0
DEFAULT_B_SMOOTHING_ALPHA = 1.0
DEFAULT_C_SMOOTHING_ALPHA = 1.0
DEFAULT_POINT_MERGE_TOL = 1e-6
DEFAULT_MIN_TANGENT_XY = 0.05
DEFAULT_MIN_PATH_POINTS = 2
DEFAULT_C_MAX_STEP_DEG = 15.0
DEFAULT_MIN_GROUP_LENGTH_MM = 0.0
DEFAULT_BRANCH_OVERLAP_TANGENT_WINDOW_MM = 5.0

DEFAULT_BBOX_X_MIN = -1e9
DEFAULT_BBOX_X_MAX =  1e9
DEFAULT_BBOX_Y_MIN = -1e9
DEFAULT_BBOX_Y_MAX =  1e9
DEFAULT_BBOX_Z_MIN = -1e9
DEFAULT_BBOX_Z_MAX =  1e9


# ---------------- Data classes ----------------

@dataclass
class Calibration:
    pr: np.ndarray
    pz: np.ndarray
    py_off: Optional[np.ndarray]
    pa: Optional[np.ndarray]

    b_min: float
    b_max: float

    x_axis: str
    y_axis: str
    z_axis: str
    b_axis: str
    c_axis: str
    u_axis: str

    c_180_deg: float

    r_model: Optional[Dict[str, Any]] = None
    z_model: Optional[Dict[str, Any]] = None
    y_off_model: Optional[Dict[str, Any]] = None
    y_off_extrap_model: Optional[Dict[str, Any]] = None
    tip_angle_model: Optional[Dict[str, Any]] = None
    selected_fit_model: Optional[str] = None
    active_phase: str = "pull"


@dataclass
class RobotSkeletonReference:
    points_xyz_mm: np.ndarray
    diameter_mm: float
    source_path: Path
    phase_point_models: Optional[Dict[str, dict]] = None
    default_phase: Optional[str] = None


@dataclass
class Vessel:
    vessel_id: int
    declared_points: int
    points: np.ndarray               # (N, 3)
    raw_rows: np.ndarray             # (N, M)
    radius_like: float
    start: np.ndarray = field(init=False)
    end: np.ndarray = field(init=False)
    min_z: float = field(init=False)
    max_z: float = field(init=False)
    path_len: float = field(init=False)

    parent_id: Optional[int] = None
    attach_end: Optional[str] = None     # 'start' or 'end'
    attach_dist: Optional[float] = None
    attach_parent_index: Optional[int] = None
    depth: int = 0

    ordered_points: Optional[np.ndarray] = None
    start_is_attachment: bool = False
    start_node_z: Optional[float] = None

    def __post_init__(self) -> None:
        self.start = self.points[0].copy()
        self.end = self.points[-1].copy()
        self.min_z = float(self.points[:, 2].min())
        self.max_z = float(self.points[:, 2].max())
        if len(self.points) > 1:
            self.path_len = float(np.linalg.norm(np.diff(self.points, axis=0), axis=1).sum())
        else:
            self.path_len = 0.0



@dataclass
class PrintPath:
    path_id: str
    source_vessel_ids: List[int]
    points: np.ndarray
    root_vessel_id: int
    parent_vessel_id: Optional[int]
    depth: int
    is_main: bool
    radius_like: float
    connection_xyz: Optional[np.ndarray] = None
    connection_z: Optional[float] = None


@dataclass
class PrintedVasculaturePath:
    points_xyz_mm: np.ndarray
    radius_mm: float


@dataclass
class PlannedTipMove:
    tip_xyz: np.ndarray
    tangent: Optional[np.ndarray]
    feed: float
    comment: str
    allow_tip_segment_contact: bool = False


@dataclass
class GroupDisplacement:
    group_number: int
    delta_stage_xyz: np.ndarray
    delta_b_deg: float = 0.0
    delta_c_deg: float = 0.0
    feed: Optional[float] = None
    enabled: bool = True
    note: str = ""


# ---------------- Calibration helpers ----------------

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


def _normalize_model_spec(model_spec: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(model_spec, dict):
        return None
    out = dict(model_spec)
    if out.get("model_type") is not None:
        out["model_type"] = str(out["model_type"]).strip().lower()
    return out


def _select_named_model(models: Dict[str, Any], base_name: str, selected_fit_model: Optional[str]) -> Optional[Dict[str, Any]]:
    selected = None if selected_fit_model is None else str(selected_fit_model).strip().lower()
    candidates: List[str] = []
    if selected:
        candidates.append(f"{base_name}_{selected}")
    candidates.append(base_name)
    for key in candidates:
        spec = _normalize_model_spec(models.get(key))
        if spec is not None:
            return spec
    return None


def _pchip_endpoint_slope(h0: float, h1: float, delta0: float, delta1: float) -> float:
    d = ((2.0 * h0 + h1) * delta0 - h0 * delta1) / max(h0 + h1, 1e-12)
    if np.sign(d) != np.sign(delta0):
        return 0.0
    if np.sign(delta0) != np.sign(delta1) and abs(d) > abs(3.0 * delta0):
        return 3.0 * delta0
    return float(d)


def pchip_eval(x_knots: Any, y_knots: Any, x_query: Any) -> np.ndarray:
    x = np.asarray(x_knots, dtype=float).reshape(-1)
    y = np.asarray(y_knots, dtype=float).reshape(-1)
    xq = np.asarray(x_query, dtype=float)

    if x.size != y.size or x.size == 0:
        raise ValueError("PCHIP model requires equal-length non-empty x_knots and y_knots.")
    if x.size == 1:
        return np.full_like(xq, float(y[0]), dtype=float)

    order = np.argsort(x)
    x = x[order]
    y = y[order]

    if np.any(np.diff(x) <= 0):
        raise ValueError("PCHIP x_knots must be strictly increasing.")

    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros_like(y)

    if x.size == 2:
        d[:] = delta[0]
    else:
        for k in range(1, x.size - 1):
            if delta[k - 1] == 0.0 or delta[k] == 0.0 or np.sign(delta[k - 1]) != np.sign(delta[k]):
                d[k] = 0.0
            else:
                w1 = 2.0 * h[k] + h[k - 1]
                w2 = h[k] + 2.0 * h[k - 1]
                d[k] = (w1 + w2) / ((w1 / delta[k - 1]) + (w2 / delta[k]))
        d[0] = _pchip_endpoint_slope(h[0], h[1], delta[0], delta[1])
        d[-1] = _pchip_endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])

    flat = xq.reshape(-1)
    idx = np.searchsorted(x, flat, side="right") - 1
    idx = np.clip(idx, 0, x.size - 2)

    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    h_i = x1 - x0
    t = (flat - x0) / h_i
    t = np.clip(t, 0.0, 1.0)

    h00 = (2.0 * t**3) - (3.0 * t**2) + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = (-2.0 * t**3) + (3.0 * t**2)
    h11 = t**3 - t**2
    yq = h00 * y0 + h10 * h_i * d[idx] + h01 * y1 + h11 * h_i * d[idx + 1]
    return yq.reshape(xq.shape)


def eval_model_spec(model_spec: Optional[Dict[str, Any]], u: Any, default_if_none: Optional[float] = None) -> np.ndarray:
    if model_spec is None:
        if default_if_none is None:
            raise ValueError("Missing required calibration model.")
        return np.full_like(np.asarray(u, dtype=float), float(default_if_none), dtype=float)

    model_type = str(model_spec.get("model_type") or "").strip().lower()
    if model_type == "pchip":
        return pchip_eval(model_spec.get("x_knots"), model_spec.get("y_knots"), u)
    if model_type == "polynomial":
        coeffs = model_spec.get("coefficients", model_spec.get("coeffs"))
        return poly_eval(coeffs, u, default_if_none=default_if_none)
    raise ValueError(f"Unsupported calibration model_type: {model_type}")


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

    selected_fit_model = data.get("selected_fit_model")
    selected_fit_model = None if selected_fit_model is None else str(selected_fit_model).strip().lower()
    active_phase = str(data.get("default_phase_for_legacy_access") or "pull").strip().lower()

    fit_models = data.get("fit_models", {}) or {}
    phase_models = data.get("fit_models_by_phase", {}) or {}
    active_phase_models = phase_models.get(active_phase) if isinstance(phase_models, dict) else None
    if not isinstance(active_phase_models, dict):
        active_phase_models = fit_models

    r_model = _select_named_model(active_phase_models, "r", selected_fit_model)
    z_model = _select_named_model(active_phase_models, "z", selected_fit_model)
    y_off_model = _select_named_model(active_phase_models, "offplane_y", selected_fit_model)
    y_off_extrap_model = _normalize_model_spec(active_phase_models.get("offplane_y_linear"))
    if y_off_extrap_model is None:
        y_off_extrap_model = _normalize_model_spec(active_phase_models.get("offplane_y"))
    tip_angle_model = _select_named_model(active_phase_models, "tip_angle", selected_fit_model)

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
        pr=pr,
        pz=pz,
        py_off=py_off,
        pa=pa,
        r_model=r_model,
        z_model=z_model,
        y_off_model=y_off_model,
        y_off_extrap_model=y_off_extrap_model,
        tip_angle_model=tip_angle_model,
        selected_fit_model=selected_fit_model,
        active_phase=active_phase,
        b_min=b_min,
        b_max=b_max,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        b_axis=b_axis,
        c_axis=c_axis,
        u_axis=u_axis,
        c_180_deg=c_180,
    )


def _normalize_phase_key(name: Optional[str]) -> str:
    raw = str(name or "pull").strip().lower()
    if raw.startswith("release"):
        return "release"
    return "pull"


def _evaluate_skeleton_descriptor(model_descriptor: Optional[dict], b_val: float) -> float:
    if model_descriptor is None:
        return 0.0
    out = eval_model_spec(model_descriptor, np.asarray([float(b_val)], dtype=float))
    return float(np.asarray(out, dtype=float).ravel()[0])


def load_robot_skeleton_reference(robot_json_path: str) -> RobotSkeletonReference:
    p = Path(robot_json_path)
    if not p.exists():
        raise FileNotFoundError(f"Robot skeleton JSON not found: {robot_json_path}")

    with p.open("r") as f:
        data = json.load(f)

    pts_raw = ((data.get("reference_pose") or {}).get("points_xyz_mm"))
    if pts_raw is None:
        raise ValueError(f"Robot skeleton JSON is missing reference_pose.points_xyz_mm: {robot_json_path}")

    pts = np.asarray(pts_raw, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
        raise ValueError(
            f"Robot skeleton JSON must provide an Nx3 reference_pose.points_xyz_mm array with at least 2 points: {robot_json_path}"
        )

    phase_models_raw = data.get("phase_point_models")
    phase_models = None
    default_phase = None
    if isinstance(phase_models_raw, dict) and phase_models_raw:
        phase_models = {_normalize_phase_key(k): v for k, v in phase_models_raw.items() if isinstance(v, dict)}
        default_phase = _normalize_phase_key(data.get("default_phase_for_legacy_access"))

    return RobotSkeletonReference(
        points_xyz_mm=pts,
        diameter_mm=float(data.get("diameter_mm", 0.0)),
        source_path=p.resolve(),
        phase_point_models=phase_models,
        default_phase=default_phase,
    )


def resolve_robot_skeleton_from_calibration(calibration_path: str) -> Optional[Path]:
    p = Path(calibration_path)
    if not p.exists():
        return None
    with p.open("r") as f:
        data = json.load(f)
    exported = data.get("exported_models", {})
    sk = exported.get("robot_skeleton_parametric")
    if not isinstance(sk, dict):
        return None
    rel = sk.get("parametric_json")
    if not rel:
        return None
    candidate = (p.resolve().parent / str(rel)).resolve()
    return candidate if candidate.exists() else None


def eval_r(cal: Calibration, b: Any) -> np.ndarray:
    if cal.r_model is not None:
        return eval_model_spec(cal.r_model, b)
    return poly_eval(cal.pr, b)


def eval_z(cal: Calibration, b: Any) -> np.ndarray:
    if cal.z_model is not None:
        return eval_model_spec(cal.z_model, b)
    return poly_eval(cal.pz, b)


def eval_pchip_with_linear_extrap(model_spec: Dict[str, Any], extrap_model_spec: Optional[Dict[str, Any]], b: Any) -> np.ndarray:
    x_knots = np.asarray(model_spec.get("x_knots", []), dtype=float).reshape(-1)
    if x_knots.size == 0 or extrap_model_spec is None:
        return eval_model_spec(model_spec, b, default_if_none=0.0)

    b_arr = np.asarray(b, dtype=float)
    pchip_values = eval_model_spec(model_spec, b_arr, default_if_none=0.0)
    out = np.asarray(pchip_values, dtype=float).copy()

    x_min = float(np.min(x_knots))
    x_max = float(np.max(x_knots))
    outside = (b_arr < x_min) | (b_arr > x_max)
    if np.any(outside):
        extrap_values = eval_model_spec(extrap_model_spec, b_arr, default_if_none=0.0)
        out = np.where(outside, extrap_values, out)
    return out


def eval_offplane_y(cal: Calibration, b: Any) -> np.ndarray:
    if cal.y_off_model is not None:
        if str(cal.y_off_model.get("model_type", "")).lower() == "pchip":
            return eval_pchip_with_linear_extrap(cal.y_off_model, cal.y_off_extrap_model, b)
        return eval_model_spec(cal.y_off_model, b, default_if_none=0.0)
    return poly_eval(cal.py_off, b, default_if_none=0.0)


def eval_tip_angle_deg(cal: Calibration, b: Any) -> np.ndarray:
    if cal.tip_angle_model is not None:
        return eval_model_spec(cal.tip_angle_model, b)
    if cal.pa is None:
        raise ValueError("Calibration is missing tip_angle_coeffs.")
    return poly_eval(cal.pa, b)


def calibration_model_range_warnings(cal: Calibration) -> List[str]:
    warnings: List[str] = []
    for label, model in (
        ("r", cal.r_model),
        ("z", cal.z_model),
        ("offplane_y", cal.y_off_model),
        ("tip_angle", cal.tip_angle_model),
    ):
        if not isinstance(model, dict) or str(model.get("model_type", "")).lower() != "pchip":
            continue
        x = np.asarray(model.get("x_knots", []), dtype=float).reshape(-1)
        if x.size == 0:
            continue
        x_min = float(np.min(x))
        x_max = float(np.max(x))
        if float(cal.b_min) < x_min or float(cal.b_max) > x_max:
            if label == "offplane_y" and cal.y_off_extrap_model is not None:
                warnings.append(
                    f"{label} PCHIP knot range [{x_min:.3f}, {x_max:.3f}] does not cover B range "
                    f"[{float(cal.b_min):.3f}, {float(cal.b_max):.3f}]; values outside the knot range use the linear offplane_y fit."
                )
            else:
                warnings.append(
                    f"{label} PCHIP knot range [{x_min:.3f}, {x_max:.3f}] does not cover B range "
                    f"[{float(cal.b_min):.3f}, {float(cal.b_max):.3f}]; values outside the knot range are clamped."
                )
    return warnings


def parse_group_displacements_file(path: str) -> Dict[int, GroupDisplacement]:
    out: Dict[int, GroupDisplacement] = {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Group displacement file not found: {path}")

    with p.open("r") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue

            note = ""
            if "#" in line:
                line, note = line.split("#", 1)
                line = line.strip()
                note = note.strip()
            if not line:
                continue

            parts = [tok for tok in re.split(r"[\s,]+", line) if tok]
            if len(parts) < 4:
                raise ValueError(
                    f"Invalid group displacement row at {path}:{line_no}. "
                    "Expected: group_number dx dy dz [db_deg] [dc_deg] [feed_mm_min] [enabled]"
                )

            group_number = int(parts[0])
            dx = float(parts[1])
            dy = float(parts[2])
            dz = float(parts[3])
            db = float(parts[4]) if len(parts) >= 5 else 0.0
            dc = float(parts[5]) if len(parts) >= 6 else 0.0
            feed = float(parts[6]) if len(parts) >= 7 else None
            enabled = True
            if len(parts) >= 8:
                enabled = str(parts[7]).strip().lower() not in {"0", "false", "no", "off", "disable", "disabled"}

            out[group_number] = GroupDisplacement(
                group_number=group_number,
                delta_stage_xyz=np.array([dx, dy, dz], dtype=float),
                delta_b_deg=db,
                delta_c_deg=dc,
                feed=feed,
                enabled=enabled,
                note=note,
            )
    return out


def predict_tip_offset_xyz(cal: Calibration, b: float, c_deg: float) -> np.ndarray:
    """
    Physical tip offset from stage origin:
      local transverse [r(B), y_off(B)] rotated by C into XY, z=z(B)
    """
    r = float(eval_r(cal, b))
    z = float(eval_z(cal, b))
    y_off = float(eval_offplane_y(cal, b))

    c = math.radians(float(c_deg))
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def stage_xyz_for_tip(cal: Calibration, tip_xyz: np.ndarray, b: float, c_deg: float) -> np.ndarray:
    return np.asarray(tip_xyz, dtype=float) - predict_tip_offset_xyz(cal, b, c_deg)


def tip_xyz_for_stage(cal: Calibration, stage_xyz: np.ndarray, b: float, c_deg: float) -> np.ndarray:
    return np.asarray(stage_xyz, dtype=float) + predict_tip_offset_xyz(cal, b, c_deg)


def compute_tracked_skeleton_world(
    skeleton_ref: RobotSkeletonReference,
    b_cmd: float,
    c_cmd_deg: float,
    tip_world: np.ndarray,
    equation_phase: str,
) -> np.ndarray:
    pts_local = np.asarray(skeleton_ref.points_xyz_mm, dtype=float).copy()
    phase_models = skeleton_ref.phase_point_models or {}
    phase_key = _normalize_phase_key(equation_phase)
    payload = phase_models.get(phase_key)
    if payload is None and skeleton_ref.default_phase is not None:
        payload = phase_models.get(_normalize_phase_key(skeleton_ref.default_phase))
    if payload is None and phase_models:
        payload = next(iter(phase_models.values()))

    if payload is not None:
        ref_positions = np.asarray(payload.get("reference_positions_xyz_mm", pts_local), dtype=float)
        if ref_positions.shape == pts_local.shape:
            pts_local = ref_positions.copy()
        for point_idx, point_model in enumerate(payload.get("point_models", [])[: pts_local.shape[0]]):
            disp_models = point_model.get("displacement_models", {})
            pts_local[point_idx, 0] += _evaluate_skeleton_descriptor(disp_models.get("x_mm"), b_cmd)
            pts_local[point_idx, 1] += _evaluate_skeleton_descriptor(disp_models.get("y_mm"), b_cmd)
            pts_local[point_idx, 2] += _evaluate_skeleton_descriptor(disp_models.get("z_mm"), b_cmd)

    tip_local = pts_local[-1, :].copy()
    offsets = pts_local - tip_local[None, :]
    c_rad = math.radians(float(c_cmd_deg))
    cosc = math.cos(c_rad)
    sinc = math.sin(c_rad)
    x_rot = offsets[:, 0] * cosc - offsets[:, 1] * sinc
    y_rot = offsets[:, 0] * sinc + offsets[:, 1] * cosc
    pts_world = np.column_stack([x_rot, y_rot, offsets[:, 2]]).astype(float)
    pts_world += np.asarray(tip_world, dtype=float).reshape(1, 3)
    pts_world[-1, :] = np.asarray(tip_world, dtype=float).reshape(3)
    return pts_world


def solve_b_for_target_tip_angle(cal: Calibration, target_angle_deg: float, search_samples: int = DEFAULT_BC_SOLVE_SAMPLES) -> float:
    """
    Solve B such that tip_angle(B) ~= target_angle_deg, constrained to [b_min, b_max].
    Returns the closest feasible B if there is no sign change.
    """
    b_lo, b_hi = float(cal.b_min), float(cal.b_max)
    bb = np.linspace(b_lo, b_hi, int(max(101, search_samples)))
    aa = eval_tip_angle_deg(cal, bb) - float(target_angle_deg)

    i_best_abs = int(np.argmin(np.abs(aa)))
    b_best_abs = float(bb[i_best_abs])

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
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fmid = f(mid)
            if abs(fmid) < 1e-10:
                return float(mid)
            if flo * fmid <= 0.0:
                hi = mid
            else:
                lo = mid
                flo = fmid
        return float(0.5 * (lo + hi))

    return b_best_abs


def solve_b_for_tip_angle_zero(cal: Calibration, search_samples: int = DEFAULT_BC_SOLVE_SAMPLES) -> float:
    return solve_b_for_target_tip_angle(cal, 0.0, search_samples=search_samples)


# ---------------- Vessel parsing / ordering ----------------

VESSEL_BLOCK_RE = re.compile(
    r"Vessel:\s*(\d+),\s*Number of Points:\s*(\d+)\s*\n\n(.*?)(?=\nVessel:\s*\d+,\s*Number of Points:|\Z)",
    re.S,
)


def parse_vessel_file(path: str) -> List[Vessel]:
    text = Path(path).read_text()
    matches = VESSEL_BLOCK_RE.findall(text)
    if not matches:
        raise ValueError(f"No vessel blocks found in {path}")

    vessels: List[Vessel] = []
    for vessel_id_s, n_points_s, body in matches:
        rows: List[List[float]] = []
        for line in body.strip().splitlines():
            line = line.strip()
            if not line:
                continue
            vals = [float(x.strip()) for x in line.split(",")]
            if len(vals) < 4:
                raise ValueError(f"Expected >= 4 columns in vessel point row, got {len(vals)}: {line}")
            rows.append(vals)

        raw = np.array(rows, dtype=float)
        if raw.shape[0] < DEFAULT_MIN_PATH_POINTS:
            continue

        vessel = Vessel(
            vessel_id=int(vessel_id_s),
            declared_points=int(n_points_s),
            points=raw[:, :3].copy(),
            raw_rows=raw,
            radius_like=float(np.median(raw[:, 3])),
        )
        vessels.append(vessel)

    vessels.sort(key=lambda v: v.vessel_id)
    return vessels


def parse_vessel_id_selection(raw_selection: Sequence[str]) -> Optional[List[int]]:
    tokens: List[str] = []
    for raw in raw_selection:
        tokens.extend(part.strip() for part in str(raw).split(","))
    tokens = [token for token in tokens if token]

    if not tokens or any(token.lower() == "all" for token in tokens):
        return None

    vessel_ids: List[int] = []
    seen: set[int] = set()
    for token in tokens:
        try:
            vessel_id = int(token)
        except ValueError as exc:
            raise ValueError(f"Invalid vessel id {token!r}; use integer IDs or All.") from exc
        if vessel_id not in seen:
            vessel_ids.append(vessel_id)
            seen.add(vessel_id)
    return vessel_ids


def filter_vessels_by_id(vessels: List[Vessel], selected_ids: Optional[List[int]]) -> List[Vessel]:
    if selected_ids is None:
        return vessels

    by_id = {v.vessel_id: v for v in vessels}
    missing = [vessel_id for vessel_id in selected_ids if vessel_id not in by_id]
    if missing:
        available = ", ".join(str(v.vessel_id) for v in vessels)
        raise ValueError(f"Requested vessel id(s) not found: {missing}. Available vessel ids: {available}")

    return [by_id[vessel_id] for vessel_id in selected_ids]


def _refresh_vessel_geometry(vessel: Vessel) -> None:
    vessel.start = vessel.points[0].copy()
    vessel.end = vessel.points[-1].copy()
    vessel.min_z = float(vessel.points[:, 2].min())
    vessel.max_z = float(vessel.points[:, 2].max())
    if len(vessel.points) > 1:
        vessel.path_len = float(np.linalg.norm(np.diff(vessel.points, axis=0), axis=1).sum())
    else:
        vessel.path_len = 0.0
    vessel.parent_id = None
    vessel.attach_end = None
    vessel.attach_dist = None
    vessel.attach_parent_index = None
    vessel.depth = 0
    vessel.ordered_points = None
    vessel.start_is_attachment = False
    vessel.start_node_z = None


def rotate_points_y(points: np.ndarray, angle_deg: float, origin: np.ndarray) -> np.ndarray:
    theta = math.radians(float(angle_deg))
    c = math.cos(theta)
    s = math.sin(theta)
    shifted = np.asarray(points, dtype=float) - np.asarray(origin, dtype=float)
    out = shifted.copy()
    out[:, 0] = c * shifted[:, 0] + s * shifted[:, 2]
    out[:, 2] = -s * shifted[:, 0] + c * shifted[:, 2]
    return out + np.asarray(origin, dtype=float)


def transform_vessel_geometry(
    vessels: List[Vessel],
    rotate_y_deg: float = 0.0,
    rotate_origin: Sequence[float] = (0.0, 0.0, 0.0),
) -> None:
    if abs(float(rotate_y_deg)) <= 1e-12:
        return
    origin = np.asarray(rotate_origin, dtype=float).reshape(3)
    for vessel in vessels:
        vessel.points = rotate_points_y(vessel.points, float(rotate_y_deg), origin)
        vessel.raw_rows[:, :3] = rotate_points_y(vessel.raw_rows[:, :3], float(rotate_y_deg), origin)
        _refresh_vessel_geometry(vessel)


def vessel_lowest_centroid_anchor(vessels: List[Vessel]) -> np.ndarray:
    if not vessels:
        raise ValueError("Cannot compute vasculature anchor for an empty vessel list.")
    pts = np.vstack([v.points for v in vessels if len(v.points) > 0])
    if len(pts) == 0:
        raise ValueError("Cannot compute vasculature anchor because no vessel points were found.")
    return np.array([float(np.mean(pts[:, 0])), float(np.mean(pts[:, 1])), float(np.min(pts[:, 2]))], dtype=float)


def translate_vessel_geometry(vessels: List[Vessel], delta_xyz: Sequence[float]) -> None:
    delta = np.asarray(delta_xyz, dtype=float).reshape(3)
    if float(np.linalg.norm(delta)) <= 1e-12:
        return
    for vessel in vessels:
        vessel.points = vessel.points + delta[None, :]
        vessel.raw_rows[:, :3] = vessel.raw_rows[:, :3] + delta[None, :]
        _refresh_vessel_geometry(vessel)


def scale_vessel_geometry(
    vessels: List[Vessel],
    scale: float,
    origin: Optional[Sequence[float]] = None,
) -> Tuple[np.ndarray, float]:
    factor = float(scale)
    if factor <= 0.0:
        raise ValueError("--geometry-scale must be greater than 0.")
    scale_origin = vessel_lowest_centroid_anchor(vessels) if origin is None else np.asarray(origin, dtype=float).reshape(3)
    if abs(factor - 1.0) <= 1e-12:
        return scale_origin, factor
    for vessel in vessels:
        vessel.points = scale_origin[None, :] + (vessel.points - scale_origin[None, :]) * factor
        vessel.raw_rows[:, :3] = scale_origin[None, :] + (vessel.raw_rows[:, :3] - scale_origin[None, :]) * factor
        _refresh_vessel_geometry(vessel)
    return scale_origin, factor


def align_vessel_lowest_centroid(vessels: List[Vessel], target_xyz: Sequence[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    before = vessel_lowest_centroid_anchor(vessels)
    target = np.asarray(target_xyz, dtype=float).reshape(3)
    delta = target - before
    translate_vessel_geometry(vessels, delta)
    after = vessel_lowest_centroid_anchor(vessels)
    return before, delta, after


def min_distance_point_to_polyline(point: np.ndarray, polyline: np.ndarray) -> Tuple[float, int]:
    dists = np.linalg.norm(polyline - point[None, :], axis=1)
    idx = int(np.argmin(dists))
    return float(dists[idx]), idx


def infer_parent_relationships(vessels: List[Vessel], attach_threshold: float) -> None:
    """
    Parent heuristic for branch ordering:
      choose the closest endpoint-to-existing-point match on a strictly larger
      vessel radius, if the distance is <= attach_threshold.
    This metadata is used for ordering and for deciding which simplified paths
    are main trunks vs secondary branches.
    """
    by_id = {v.vessel_id: v for v in vessels}

    for vessel in vessels:
        best: Optional[Tuple[float, int, str, int]] = None
        for parent in vessels:
            if parent.vessel_id == vessel.vessel_id:
                continue
            if parent.radius_like <= vessel.radius_like:
                continue

            d_start, idx_start = min_distance_point_to_polyline(vessel.start, parent.points)
            d_end, idx_end = min_distance_point_to_polyline(vessel.end, parent.points)
            if d_start <= d_end:
                candidate = (d_start, parent.vessel_id, "start", idx_start)
            else:
                candidate = (d_end, parent.vessel_id, "end", idx_end)

            if best is None or candidate[0] < best[0]:
                best = candidate

        if best is not None and best[0] <= float(attach_threshold):
            vessel.parent_id = best[1]
            vessel.attach_end = best[2]
            vessel.attach_dist = float(best[0])
            vessel.attach_parent_index = int(best[3])
        else:
            vessel.parent_id = None
            vessel.attach_end = None
            vessel.attach_dist = None
            vessel.attach_parent_index = None

    def depth_of(v: Vessel, cache: Dict[int, int]) -> int:
        if v.vessel_id in cache:
            return cache[v.vessel_id]
        if v.parent_id is None or v.parent_id not in by_id:
            cache[v.vessel_id] = 0
            return 0
        d = 1 + depth_of(by_id[v.parent_id], cache)
        cache[v.vessel_id] = d
        return d

    cache: Dict[int, int] = {}
    for vessel in vessels:
        vessel.depth = depth_of(vessel, cache)


def orient_vessels(vessels: List[Vessel]) -> None:
    """
    Root/main vessels: bottom-to-top by Z.
    Child vessels: start at the attachment node and go outward from the parent.
    """
    for vessel in vessels:
        pts = vessel.points
        if vessel.parent_id is None:
            ordered = pts.copy() if pts[0, 2] <= pts[-1, 2] else pts[::-1].copy()
            vessel.start_is_attachment = False
        else:
            ordered = pts.copy() if vessel.attach_end == "start" else pts[::-1].copy()
            vessel.start_is_attachment = True

        vessel.ordered_points = ordered
        vessel.start_node_z = float(ordered[0, 2])


def vessel_print_order(vessels: List[Vessel], order_mode: str = "ascending_id") -> List[Vessel]:
    mode = str(order_mode).strip().lower()
    if mode == "ascending_id":
        return sorted(vessels, key=lambda v: v.vessel_id)
    if mode == "ascending_start_z":
        return sorted(vessels, key=lambda v: (v.start_node_z if v.start_node_z is not None else v.min_z, v.vessel_id))
    if mode == "bottom_up_hierarchy":
        roots = [v for v in vessels if v.parent_id is None]
        children = [v for v in vessels if v.parent_id is not None]
        roots.sort(key=lambda v: (v.min_z, -v.radius_like, v.vessel_id))
        children.sort(key=lambda v: (v.depth, v.start_node_z if v.start_node_z is not None else v.min_z, -v.radius_like, v.vessel_id))
        return roots + children
    if mode == "lowest_longest_roots":
        roots = [v for v in vessels if v.parent_id is None]
        children = [v for v in vessels if v.parent_id is not None]
        roots.sort(key=lambda v: (v.min_z, -v.path_len, -v.radius_like, v.vessel_id))
        children.sort(key=lambda v: (v.depth, v.start_node_z if v.start_node_z is not None else v.min_z, -v.path_len, -v.radius_like, v.vessel_id))
        return roots + children
    raise ValueError(f"Unsupported vessel order mode: {order_mode}")


def _component_sort_key(vessels: List[Vessel], component_vessel_ids: Iterable[int], order_mode: str) -> Tuple[float, int]:
    comp = [v for v in vessels if v.vessel_id in set(component_vessel_ids)]
    if not comp:
        return (0.0, 0)
    mode = str(order_mode).strip().lower()
    if mode == "ascending_id":
        return (float(min(v.vessel_id for v in comp)), int(min(v.vessel_id for v in comp)))
    if mode == "lowest_longest_roots":
        return (
            float(min(v.min_z for v in comp)),
            int(-round(max(v.path_len for v in comp) * 1000000.0)),
        )
    z0 = min(v.min_z for v in comp)
    return (float(z0), int(min(v.vessel_id for v in comp)))


def _cluster_endpoint_nodes(vessels: List[Vessel], merge_threshold: float) -> Tuple[Dict[Tuple[int, int], int], Dict[int, np.ndarray]]:
    node_points: Dict[int, List[np.ndarray]] = {}
    endpoint_to_node: Dict[Tuple[int, int], int] = {}
    next_node_id = 0

    def find_existing_node(pt: np.ndarray) -> Optional[int]:
        for node_id, pts in node_points.items():
            center = np.mean(np.asarray(pts, dtype=float), axis=0)
            if float(np.linalg.norm(np.asarray(pt, dtype=float) - center)) <= float(merge_threshold):
                return node_id
        return None

    for vessel in vessels:
        for end_idx, pt in ((0, vessel.points[0]), (1, vessel.points[-1])):
            node_id = find_existing_node(pt)
            if node_id is None:
                node_id = next_node_id
                next_node_id += 1
                node_points[node_id] = []
            node_points[node_id].append(np.asarray(pt, dtype=float).copy())
            endpoint_to_node[(vessel.vessel_id, end_idx)] = node_id

    node_xyz = {nid: np.mean(np.asarray(pts, dtype=float), axis=0) for nid, pts in node_points.items()}
    return endpoint_to_node, node_xyz


def _concat_polyline_segments(segments: List[np.ndarray], merge_threshold: float) -> np.ndarray:
    if not segments:
        return np.zeros((0, 3), dtype=float)
    out = np.asarray(segments[0], dtype=float).copy()
    for seg in segments[1:]:
        seg = np.asarray(seg, dtype=float)
        if len(seg) == 0:
            continue
        if float(np.linalg.norm(out[-1] - seg[0])) <= float(merge_threshold):
            out = np.vstack([out, seg[1:]])
        else:
            out = np.vstack([out, seg])
    return out


def simplify_vessel_paths(
    vessels: List[Vessel],
    merge_threshold: float,
    order_mode: str = "ascending_id",
    preferred_main_vessel_ids: Optional[List[int]] = None,
) -> List[PrintPath]:
    """
    Simplify endpoint-connected vessel segments into longer continuous polylines.

    Strategy:
    - Build an endpoint graph where each vessel is an edge between two snapped
      endpoint nodes.
    - In each connected component, start from a leaf/root node.
    - At every node, continue along the locally longest downstream branch.
    - Emit that merged path as the local main branch, then recurse into the
      remaining side branches.

    This combines segmented centerlines that share endpoints while preserving
    secondary branches as separate later print passes.
    """
    if not vessels:
        return []

    by_id = {v.vessel_id: v for v in vessels}
    endpoint_to_node, node_xyz = _cluster_endpoint_nodes(vessels, merge_threshold=float(merge_threshold))

    edge_nodes: Dict[int, Tuple[int, int]] = {}
    node_to_edges: Dict[int, List[int]] = {}
    for vessel in vessels:
        n0 = endpoint_to_node[(vessel.vessel_id, 0)]
        n1 = endpoint_to_node[(vessel.vessel_id, 1)]
        edge_nodes[vessel.vessel_id] = (n0, n1)
        node_to_edges.setdefault(n0, []).append(vessel.vessel_id)
        node_to_edges.setdefault(n1, []).append(vessel.vessel_id)

    # Build edge connected components using endpoint adjacency.
    unseen = {v.vessel_id for v in vessels}
    components: List[List[int]] = []
    while unseen:
        seed = next(iter(unseen))
        stack = [seed]
        comp: List[int] = []
        unseen.remove(seed)
        while stack:
            eid = stack.pop()
            comp.append(eid)
            n0, n1 = edge_nodes[eid]
            for nid in (n0, n1):
                for other in node_to_edges.get(nid, []):
                    if other in unseen:
                        unseen.remove(other)
                        stack.append(other)
        components.append(sorted(comp))

    components.sort(key=lambda comp: _component_sort_key(vessels, comp, order_mode))

    def edge_points_from_node(edge_id: int, from_node: int) -> np.ndarray:
        vessel = by_id[edge_id]
        n0, n1 = edge_nodes[edge_id]
        if from_node == n0 and from_node != n1:
            return vessel.points.copy()
        if from_node == n1 and from_node != n0:
            return vessel.points[::-1].copy()
        # degenerate case: same snapped node on both ends, keep original order
        return vessel.points.copy()

    def other_node(edge_id: int, node_id: int) -> int:
        n0, n1 = edge_nodes[edge_id]
        return n1 if node_id == n0 else n0

    def edge_tangent_from_node(edge_id: int, from_node: int, at_start: bool = True) -> np.ndarray:
        pts = edge_points_from_node(edge_id, from_node)
        if len(pts) < 2:
            return np.zeros(3, dtype=float)
        if at_start:
            return normalize(pts[1] - pts[0])
        return normalize(pts[-1] - pts[-2])

    def continuation_alignment(prev_edge: Optional[int], cur_node: int, candidate_edge: int) -> float:
        if prev_edge is None:
            return 0.0
        prev_from_node = other_node(prev_edge, cur_node)
        incoming = edge_tangent_from_node(prev_edge, prev_from_node, at_start=False)
        outgoing = edge_tangent_from_node(candidate_edge, cur_node, at_start=True)
        return float(np.dot(incoming, outgoing))

    def best_score_from(node_id: int, incoming_edge: Optional[int], allowed_edges: set[int], memo: Dict[Tuple[int, Optional[int]], float]) -> float:
        key = (node_id, incoming_edge)
        if key in memo:
            return memo[key]
        best = 0.0
        for eid in node_to_edges.get(node_id, []):
            if eid == incoming_edge or eid not in allowed_edges:
                continue
            score = by_id[eid].path_len + best_score_from(other_node(eid, node_id), eid, allowed_edges, memo)
            if score > best:
                best = score
        memo[key] = float(best)
        return float(best)

    all_paths: List[PrintPath] = []

    for comp_index, comp_edge_ids in enumerate(components):
        comp_set = set(comp_edge_ids)
        comp_nodes = sorted({n for eid in comp_edge_ids for n in edge_nodes[eid]})
        degree = {nid: sum(1 for eid in node_to_edges.get(nid, []) if eid in comp_set) for nid in comp_nodes}
        leaf_nodes = [nid for nid in comp_nodes if degree[nid] <= 1]
        if leaf_nodes:
            root_node = min(leaf_nodes, key=lambda nid: (float(node_xyz[nid][2]), float(node_xyz[nid][1]), float(node_xyz[nid][0])))
        else:
            root_node = min(comp_nodes, key=lambda nid: (float(node_xyz[nid][2]), float(node_xyz[nid][1]), float(node_xyz[nid][0])))

        local_counter = 0

        def emit_from(node_id: int, incoming_edge: Optional[int], depth: int, parent_vessel_id: Optional[int], is_main: bool, allowed_edges: set[int]) -> None:
            nonlocal local_counter
            memo: Dict[Tuple[int, Optional[int]], float] = {}

            chain_edges: List[Tuple[int, int]] = []  # (edge_id, from_node)
            used_in_chain: set[int] = set()
            cur_node = node_id
            prev_edge = incoming_edge

            while True:
                candidates = [
                    eid
                    for eid in node_to_edges.get(cur_node, [])
                    if eid != prev_edge and eid in allowed_edges and eid not in used_in_chain
                ]
                if str(order_mode).strip().lower() == "lowest_longest_roots":
                    upward_candidates = [
                        eid
                        for eid in candidates
                        if float(node_xyz[other_node(eid, cur_node)][2]) >= float(node_xyz[cur_node][2]) - 1e-9
                    ]
                    if upward_candidates:
                        candidates = upward_candidates
                if not candidates:
                    break
                best_edge = max(
                    candidates,
                    key=lambda eid: (
                        (
                            1
                            if str(order_mode).strip().lower() == "lowest_longest_roots" and prev_edge is not None and by_id[eid].parent_id == prev_edge
                            else 0
                        ),
                        (
                            continuation_alignment(prev_edge, cur_node, eid)
                            if str(order_mode).strip().lower() == "lowest_longest_roots"
                            else by_id[eid].path_len + best_score_from(other_node(eid, cur_node), eid, allowed_edges, memo)
                        ),
                        float(node_xyz[other_node(eid, cur_node)][2]),
                        by_id[eid].path_len + best_score_from(other_node(eid, cur_node), eid, allowed_edges, memo),
                        -by_id[eid].depth,
                        by_id[eid].vessel_id,
                    ),
                )
                chain_edges.append((best_edge, cur_node))
                used_in_chain.add(best_edge)
                prev_edge = best_edge
                cur_node = other_node(best_edge, cur_node)

            if not chain_edges:
                return

            used_edge_ids = [eid for eid, _ in chain_edges]
            local_counter += 1
            segs = [edge_points_from_node(eid, from_node) for eid, from_node in chain_edges]
            merged_points = _concat_polyline_segments(segs, merge_threshold=float(merge_threshold))
            source_ids = [eid for eid, _ in chain_edges]
            source_vessels = [by_id[eid] for eid in source_ids]
            external_parent_ids = [v.parent_id for v in source_vessels if v.parent_id is not None and v.parent_id not in source_ids]
            path_parent = external_parent_ids[0] if external_parent_ids else parent_vessel_id
            path_depth = min(v.depth for v in source_vessels) if source_vessels else depth
            path_radius = float(np.median([v.radius_like for v in source_vessels])) if source_vessels else 0.0
            all_paths.append(PrintPath(
                path_id=f"component{comp_index:03d}_path{local_counter:03d}",
                source_vessel_ids=source_ids,
                points=merged_points,
                root_vessel_id=source_ids[0],
                parent_vessel_id=path_parent,
                depth=max(depth, path_depth),
                is_main=is_main,
                radius_like=path_radius,
                connection_xyz=None if is_main else np.asarray(node_xyz[node_id], dtype=float).copy(),
                connection_z=None if is_main else float(node_xyz[node_id][2]),
            ))

            # Recurse into the secondary branches that were not selected as the main continuation.
            branch_parent = source_ids[-1]
            for used_edge_id in used_edge_ids:
                allowed_edges.discard(used_edge_id)

            for idx, (eid, from_node) in enumerate(chain_edges):
                next_node = other_node(eid, from_node)

                non_main_here = [x for x in node_to_edges.get(from_node, []) if x in allowed_edges and x != incoming_edge]
                for child_eid in sorted(non_main_here, key=lambda x: (-by_id[x].path_len, by_id[x].vessel_id)):
                    emit_from(from_node, None, depth + 1, by_id[eid].vessel_id, False, allowed_edges)

                incoming_edge = eid

            tail_candidates = [x for x in node_to_edges.get(cur_node, []) if x in allowed_edges]
            for child_eid in sorted(tail_candidates, key=lambda x: (-by_id[x].path_len, by_id[x].vessel_id)):
                emit_from(cur_node, None, depth + 1, branch_parent, False, allowed_edges)

        def emit_preferred_main_path(allowed_edges: set[int]) -> bool:
            nonlocal local_counter
            preferred = [int(eid) for eid in (preferred_main_vessel_ids or []) if int(eid) in allowed_edges]
            if not preferred or len(preferred) != len(preferred_main_vessel_ids or []):
                return False

            first = preferred[0]
            n0, n1 = edge_nodes[first]
            cur_node = n0 if float(node_xyz[n0][2]) <= float(node_xyz[n1][2]) else n1
            chain_edges: List[Tuple[int, int]] = []
            for eid in preferred:
                a, b = edge_nodes[eid]
                if cur_node == a:
                    from_node = a
                    cur_node = b
                elif cur_node == b:
                    from_node = b
                    cur_node = a
                else:
                    return False
                chain_edges.append((eid, from_node))

            local_counter += 1
            segs = [edge_points_from_node(eid, from_node) for eid, from_node in chain_edges]
            merged_points = _concat_polyline_segments(segs, merge_threshold=float(merge_threshold))
            source_ids = [eid for eid, _ in chain_edges]
            source_vessels = [by_id[eid] for eid in source_ids]
            path_radius = float(np.median([v.radius_like for v in source_vessels])) if source_vessels else 0.0
            all_paths.append(PrintPath(
                path_id=f"component{comp_index:03d}_path{local_counter:03d}",
                source_vessel_ids=source_ids,
                points=merged_points,
                root_vessel_id=source_ids[0],
                parent_vessel_id=None,
                depth=0,
                is_main=True,
                radius_like=path_radius,
                connection_xyz=None,
                connection_z=None,
            ))

            for eid in source_ids:
                allowed_edges.discard(eid)

            incoming_edge: Optional[int] = None
            for eid, from_node in chain_edges:
                branch_candidates = [x for x in node_to_edges.get(from_node, []) if x in allowed_edges and x != incoming_edge]
                for child_eid in sorted(branch_candidates, key=lambda x: (float(node_xyz[from_node][2]), -by_id[x].path_len, by_id[x].vessel_id)):
                    emit_from(from_node, None, depth=1, parent_vessel_id=eid, is_main=False, allowed_edges=allowed_edges)
                incoming_edge = eid

            tail_candidates = [x for x in node_to_edges.get(cur_node, []) if x in allowed_edges]
            for child_eid in sorted(tail_candidates, key=lambda x: (float(node_xyz[cur_node][2]), -by_id[x].path_len, by_id[x].vessel_id)):
                emit_from(cur_node, None, depth=1, parent_vessel_id=source_ids[-1], is_main=False, allowed_edges=allowed_edges)
            return True

        if not emit_preferred_main_path(comp_set):
            emit_from(root_node, None, depth=0, parent_vessel_id=None, is_main=True, allowed_edges=comp_set)

        # Any leftover edges usually correspond to disconnected leftovers from a non-tree component.
        while comp_set:
            seed = min(comp_set)
            n0, n1 = edge_nodes[seed]
            start_node = n0 if float(node_xyz[n0][2]) <= float(node_xyz[n1][2]) else n1
            emit_from(start_node, None, depth=0, parent_vessel_id=None, is_main=True, allowed_edges=comp_set)

    # Stable final order: main paths before branches, then hierarchy depth, then requested component key.
    def final_key(p: PrintPath) -> Tuple[int, int, float, int]:
        if str(order_mode).strip().lower() == "lowest_longest_roots":
            pts = np.asarray(p.points, dtype=float)
            connection_z = p.connection_z if p.connection_z is not None else (float(pts[0, 2]) if len(pts) else 0.0)
            return (0 if p.is_main else 1, int(round(float(connection_z) * 1000000.0)), float(connection_z), int(min(p.source_vessel_ids)))
        comp_key = _component_sort_key(vessels, p.source_vessel_ids, order_mode)
        return (0 if p.is_main else 1, int(p.depth), float(comp_key[0]), int(min(p.source_vessel_ids)))

    all_paths.sort(key=final_key)
    return all_paths


# ---------------- Geometry / tangent helpers ----------------

def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return np.zeros_like(v)
    return v / n


def smooth_centerline_points(points: np.ndarray, window: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 2 or int(window) <= 0:
        return pts.copy()

    w = int(window)
    out = pts.copy()
    for i in range(len(pts)):
        i0 = max(0, i - w)
        i1 = min(len(pts), i + w + 1)
        out[i] = pts[i0:i1].mean(axis=0)

    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def deduplicate_polyline_points(points: np.ndarray, tol: float = DEFAULT_POINT_MERGE_TOL) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return pts.copy()

    out = [pts[0].copy()]
    for p in pts[1:]:
        if float(np.linalg.norm(p - out[-1])) > float(tol):
            out.append(np.asarray(p, dtype=float).copy())
    return np.asarray(out, dtype=float)


def resample_polyline_by_spacing(points: np.ndarray, spacing: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1 or float(spacing) <= 0.0:
        return pts.copy()

    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    total = float(seg.sum())
    if total <= 0.0:
        return pts[[0, -1]].copy() if len(pts) > 1 else pts.copy()

    spacing = float(spacing)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    sample_s = list(np.arange(0.0, total, spacing))
    if not sample_s or abs(sample_s[-1] - total) > 1e-12:
        sample_s.append(total)

    out = []
    j = 0
    for s in sample_s:
        while j < len(seg) - 1 and cum[j + 1] < s:
            j += 1
        denom = seg[j] if seg[j] > 0 else 1.0
        t = float((s - cum[j]) / denom)
        t = min(1.0, max(0.0, t))
        p = pts[j] + t * (pts[j + 1] - pts[j])
        out.append(np.asarray(p, dtype=float))
    return deduplicate_polyline_points(np.asarray(out, dtype=float), tol=1e-12)


def prepare_print_path_points(
    points: np.ndarray,
    point_merge_tol: float = DEFAULT_POINT_MERGE_TOL,
    resample_spacing: float = DEFAULT_PATH_RESAMPLE_SPACING,
    geometry_smooth_window: int = DEFAULT_PATH_GEOMETRY_SMOOTH_WINDOW,
) -> np.ndarray:
    pts = deduplicate_polyline_points(points, tol=point_merge_tol)
    if len(pts) <= 1:
        return pts
    if float(resample_spacing) > 0.0:
        pts = resample_polyline_by_spacing(pts, spacing=float(resample_spacing))
        pts = deduplicate_polyline_points(pts, tol=point_merge_tol)
    if int(geometry_smooth_window) > 0 and len(pts) > 2:
        pts = smooth_centerline_points(pts, window=int(geometry_smooth_window))
        pts = deduplicate_polyline_points(pts, tol=point_merge_tol)
    return pts


def unwrap_angle_deg_near(target_deg: float, reference_deg: float) -> float:
    target = float(target_deg)
    ref = float(reference_deg)
    while target - ref > 180.0:
        target -= 360.0
    while target - ref < -180.0:
        target += 360.0
    return float(target)


def limit_angle_step_deg(target_deg: float, reference_deg: float, max_step_deg: float) -> float:
    max_step = float(max_step_deg)
    target = unwrap_angle_deg_near(target_deg, reference_deg)
    if max_step <= 0.0:
        return float(target)
    delta = float(target - float(reference_deg))
    if delta > max_step:
        return float(reference_deg + max_step)
    if delta < -max_step:
        return float(reference_deg - max_step)
    return float(target)


def smooth_scalar_toward(target: float, reference: float, alpha: float) -> float:
    a = float(np.clip(float(alpha), 0.0, 1.0))
    return float(float(reference) + a * (float(target) - float(reference)))


def smooth_angle_deg_toward(target_deg: float, reference_deg: float, alpha: float) -> float:
    target = unwrap_angle_deg_near(target_deg, reference_deg)
    return smooth_scalar_toward(target, reference_deg, alpha)


def tangent_for_index(points: np.ndarray, i: int, smooth_window: int = DEFAULT_TANGENT_SMOOTH_WINDOW) -> np.ndarray:
    n = len(points)
    i0 = max(0, i - int(smooth_window))
    i1 = min(n - 1, i + int(smooth_window))
    if i1 == i0:
        if i == 0 and n > 1:
            return normalize(points[1] - points[0])
        if i == n - 1 and n > 1:
            return normalize(points[-1] - points[-2])
        return np.array([0.0, 0.0, 1.0], dtype=float)
    return normalize(points[i1] - points[i0])


def desired_physical_b_angle_from_tangent(tangent: np.ndarray) -> float:
    """
    Convert a 3D tangent vector into the physical B-angle convention used by
    this robot and its calibration:

      B = 0 deg   -> tip points along -Z
      B = 90 deg  -> tip is horizontal
      B = 180 deg -> tip points along +Z

    For bottom-to-top writing, the tip should still point down into the printed
    material. Therefore upward vertical tangent t=(0,0,1) maps to B=0.

      B = acos(t_z)
    """
    t = normalize(tangent)
    tz = float(np.clip(t[2], -1.0, 1.0))
    return float(math.degrees(math.acos(tz)))


def b_angle_comment_from_tangent(tangent: np.ndarray) -> float:
    """
    Viewer/comment helper that reports the same physical B convention:
      0 -> -Z, 90 -> horizontal, 180 -> +Z
    """
    return desired_physical_b_angle_from_tangent(tangent)


def c_angle_from_tangent(
    tangent: np.ndarray,
    prev_c: float = 0.0,
    min_xy: float = DEFAULT_MIN_TANGENT_XY,
    azimuth_offset_deg: float = 0.0,
) -> float:
    xy = np.asarray(tangent[:2], dtype=float)
    if float(np.linalg.norm(xy)) < float(min_xy):
        return float(prev_c)
    raw = float(math.degrees(math.atan2(float(xy[1]), float(xy[0])))) + float(azimuth_offset_deg)
    return unwrap_angle_deg_near(raw, prev_c)


def side_vector_from_tangent(tangent: np.ndarray, fallback: Optional[np.ndarray] = None, min_xy: float = DEFAULT_MIN_TANGENT_XY) -> np.ndarray:
    xy = np.asarray(tangent[:2], dtype=float)
    nxy = float(np.linalg.norm(xy))
    if nxy < float(min_xy):
        if fallback is not None and float(np.linalg.norm(fallback[:2])) >= float(min_xy):
            xy = np.asarray(fallback[:2], dtype=float)
            nxy = float(np.linalg.norm(xy))
        else:
            return np.array([1.0, 0.0, 0.0], dtype=float)
    tx, ty = xy / nxy
    # +90 degree rotation in XY
    return np.array([-ty, tx, 0.0], dtype=float)


def path_length(points: np.ndarray) -> float:
    if len(points) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(points, axis=0), axis=1).sum())


def point_inside_xy_bbox(point_xy: np.ndarray, bbox_xy: Tuple[float, float, float, float]) -> bool:
    x_min, x_max, y_min, y_max = bbox_xy
    x = float(point_xy[0])
    y = float(point_xy[1])
    return x_min <= x <= x_max and y_min <= y <= y_max


def segment_intersects_xy_bbox(p0_xy: np.ndarray, p1_xy: np.ndarray, bbox_xy: Tuple[float, float, float, float]) -> bool:
    x_min, x_max, y_min, y_max = bbox_xy
    p0 = np.asarray(p0_xy, dtype=float)
    p1 = np.asarray(p1_xy, dtype=float)
    if point_inside_xy_bbox(p0, bbox_xy) or point_inside_xy_bbox(p1, bbox_xy):
        return True

    d = p1 - p0
    t0 = 0.0
    t1 = 1.0
    for p, q in (
        (-float(d[0]), float(p0[0] - x_min)),
        ( float(d[0]), float(x_max - p0[0])),
        (-float(d[1]), float(p0[1] - y_min)),
        ( float(d[1]), float(y_max - p0[1])),
    ):
        if abs(p) <= 1e-12:
            if q < 0.0:
                return False
            continue
        r = q / p
        if p < 0.0:
            if r > t1:
                return False
            t0 = max(t0, r)
        else:
            if r < t0:
                return False
            t1 = min(t1, r)
    return t0 <= t1


def rotate_xy_vector(vec_xy: np.ndarray, angle_deg: float) -> np.ndarray:
    v = np.asarray(vec_xy, dtype=float).reshape(2)
    theta = math.radians(float(angle_deg))
    c = math.cos(theta)
    s = math.sin(theta)
    return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=float)


def point_segment_distance_3d(point: np.ndarray, seg_a: np.ndarray, seg_b: np.ndarray) -> float:
    p = np.asarray(point, dtype=float).reshape(3)
    a = np.asarray(seg_a, dtype=float).reshape(3)
    b = np.asarray(seg_b, dtype=float).reshape(3)
    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(p - a))
    t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
    q = a + t * ab
    return float(np.linalg.norm(p - q))


def segment_segment_distance_3d(seg0_a: np.ndarray, seg0_b: np.ndarray, seg1_a: np.ndarray, seg1_b: np.ndarray) -> float:
    p1 = np.asarray(seg0_a, dtype=float).reshape(3)
    q1 = np.asarray(seg0_b, dtype=float).reshape(3)
    p2 = np.asarray(seg1_a, dtype=float).reshape(3)
    q2 = np.asarray(seg1_b, dtype=float).reshape(3)

    u = q1 - p1
    v = q2 - p2
    w = p1 - p2
    a = float(np.dot(u, u))
    b = float(np.dot(u, v))
    c = float(np.dot(v, v))
    d = float(np.dot(u, w))
    e = float(np.dot(v, w))
    denom = a * c - b * b
    small = 1e-12

    if a <= small and c <= small:
        return float(np.linalg.norm(p1 - p2))
    if a <= small:
        return point_segment_distance_3d(p1, p2, q2)
    if c <= small:
        return point_segment_distance_3d(p2, p1, q1)

    if denom <= small:
        s = 0.0
        t = float(np.clip(e / c, 0.0, 1.0))
    else:
        s = float(np.clip((b * e - c * d) / denom, 0.0, 1.0))
        t = float((a * e - b * d) / denom)
        if t < 0.0:
            t = 0.0
            s = float(np.clip(-d / a, 0.0, 1.0))
        elif t > 1.0:
            t = 1.0
            s = float(np.clip((b - d) / a, 0.0, 1.0))

    c1 = p1 + s * u
    c2 = p2 + t * v
    return float(np.linalg.norm(c1 - c2))


def polyline_polyline_min_distance(poly0: np.ndarray, poly1: np.ndarray) -> float:
    pts0 = np.asarray(poly0, dtype=float)
    pts1 = np.asarray(poly1, dtype=float)
    if len(pts0) == 0 or len(pts1) == 0:
        return float("inf")
    if len(pts0) == 1 and len(pts1) == 1:
        return float(np.linalg.norm(pts0[0] - pts1[0]))
    if len(pts0) == 1:
        return min(point_segment_distance_3d(pts0[0], pts1[i], pts1[i + 1]) for i in range(len(pts1) - 1))
    if len(pts1) == 1:
        return min(point_segment_distance_3d(pts1[0], pts0[i], pts0[i + 1]) for i in range(len(pts0) - 1))

    best = float("inf")
    for i in range(len(pts0) - 1):
        for j in range(len(pts1) - 1):
            best = min(best, segment_segment_distance_3d(pts0[i], pts0[i + 1], pts1[j], pts1[j + 1]))
            if best <= 0.0:
                return 0.0
    return best


# ---------------- G-code helpers ----------------

def _fmt_axes_move(axes_vals: Sequence[Tuple[str, float]]) -> str:
    return " ".join(f"{ax}{val:.3f}" for ax, val in axes_vals)


class GCodeWriter:
    def __init__(
        self,
        fh,
        cal: Calibration,
        bbox: Dict[str, float],
        travel_feed: float,
        approach_feed: float,
        fine_approach_feed: float,
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
        write_mode: str,
        orientation_mode: str,
        bc_solve_samples: int,
        b_max_step_deg: float,
        c_max_step_deg: float,
        b_smoothing_alpha: float,
        c_smoothing_alpha: float,
        min_tangent_xy_for_c: float,
        travel_clearance_above_printed_z: float,
        travel_bbox_margin: float,
        travel_edge_clearance: float,
        enable_travel_bbox_clearance: bool,
        fine_approach_distance: float,
        robot_skeleton_ref: Optional[RobotSkeletonReference],
        skeleton_collision_clearance: float,
        skeleton_collision_sample_step_mm: float,
    ) -> None:
        self.f = fh
        self.cal = cal
        self.bbox = bbox
        self.travel_feed = float(travel_feed)
        self.approach_feed = float(approach_feed)
        self.fine_approach_feed = float(fine_approach_feed)
        self.print_feed = float(print_feed)
        self.c_feed = float(c_feed)
        self.extrusion_per_mm = float(extrusion_per_mm)
        self.pressure_offset_mm = float(pressure_offset_mm)
        self.pressure_advance_feed = float(pressure_advance_feed)
        self.pressure_retract_feed = float(pressure_retract_feed)
        self.preflow_dwell_ms = int(preflow_dwell_ms)
        self.node_dwell_ms = int(node_dwell_ms)
        self.edge_samples = max(1, int(edge_samples))
        self.emit_extrusion = bool(emit_extrusion)
        self.write_mode = str(write_mode).strip().lower()
        self.orientation_mode = str(orientation_mode).strip().lower()
        self.bc_solve_samples = int(bc_solve_samples)
        self.b_max_step_deg = float(b_max_step_deg)
        self.c_max_step_deg = float(c_max_step_deg)
        self.b_smoothing_alpha = float(np.clip(float(b_smoothing_alpha), 0.0, 1.0))
        self.c_smoothing_alpha = float(np.clip(float(c_smoothing_alpha), 0.0, 1.0))
        self.min_tangent_xy_for_c = float(min_tangent_xy_for_c)
        self.travel_clearance_above_printed_z = float(travel_clearance_above_printed_z)
        self.travel_bbox_margin = float(travel_bbox_margin)
        self.travel_edge_clearance = float(travel_edge_clearance)
        self.enable_travel_bbox_clearance = bool(enable_travel_bbox_clearance)
        self.fine_approach_distance = float(fine_approach_distance)
        self.robot_skeleton_ref = robot_skeleton_ref
        self.skeleton_collision_clearance = max(0.0, float(skeleton_collision_clearance))
        self.skeleton_collision_sample_step_mm = max(0.25, float(skeleton_collision_sample_step_mm))

        self.u_material_abs = 0.0
        self.pressure_charged = False
        self.cur_stage_xyz: Optional[np.ndarray] = None
        self.cur_tip_xyz: Optional[np.ndarray] = None
        self.cur_b: float = 0.0
        self.cur_c: float = 0.0
        self.last_tip_tangent: Optional[np.ndarray] = None
        self.printed_tip_points: List[np.ndarray] = []
        self.printed_paths: List[PrintedVasculaturePath] = []
        self.warnings: List[str] = []
        self.command_min = np.array([math.inf, math.inf, math.inf], dtype=float)
        self.command_max = np.array([-math.inf, -math.inf, -math.inf], dtype=float)
        self.b_min_used = math.inf
        self.b_max_used = -math.inf
        self.c_min_used = math.inf
        self.c_max_used = -math.inf

        self.fixed_b = solve_b_for_target_tip_angle(self.cal, 0.0, search_samples=self.bc_solve_samples) if self.write_mode == "calibrated" else 0.0
        self.fixed_c = 0.0

    def clamp_stage(self, p_stage: np.ndarray, context: str) -> np.ndarray:
        x = float(np.clip(p_stage[0], self.bbox["x_min"], self.bbox["x_max"]))
        y = float(np.clip(p_stage[1], self.bbox["y_min"], self.bbox["y_max"]))
        z = float(np.clip(p_stage[2], self.bbox["z_min"], self.bbox["z_max"]))
        if abs(x - float(p_stage[0])) > 1e-12 or abs(y - float(p_stage[1])) > 1e-12 or abs(z - float(p_stage[2])) > 1e-12:
            self.warnings.append(f"WARNING: {context} stage point clamped to bbox.")
        return np.array([x, y, z], dtype=float)

    def u_cmd_actual(self) -> float:
        return self.u_material_abs + (self.pressure_offset_mm if self.pressure_charged else 0.0)

    def bc_for_tangent(self, tangent: np.ndarray, prev_c: Optional[float] = None, prev_b: Optional[float] = None) -> Tuple[float, float]:
        if self.orientation_mode == "fixed" or self.write_mode != "calibrated":
            return float(self.fixed_b), float(self.fixed_c)

        c_ref = self.cur_c if prev_c is None else float(prev_c)
        b_ref = self.cur_b if prev_b is None else float(prev_b)
        # For pull-aligned writing the rotary azimuth needs to point opposite
        # the local XY tangent, not along it. The calibration provides the
        # command-space delta corresponding to a physical 180 degree flip.
        c_raw = c_angle_from_tangent(
            tangent,
            prev_c=c_ref,
            min_xy=self.min_tangent_xy_for_c,
            azimuth_offset_deg=self.cal.c_180_deg,
        )
        c_limited = limit_angle_step_deg(c_raw, c_ref, self.c_max_step_deg)
        c_deg = smooth_angle_deg_toward(c_limited, c_ref, self.c_smoothing_alpha)
        target_b = desired_physical_b_angle_from_tangent(tangent)
        b_raw = solve_b_for_target_tip_angle(self.cal, target_b, search_samples=self.bc_solve_samples)
        b_limited = limit_angle_step_deg(b_raw, b_ref, self.b_max_step_deg)
        b = smooth_scalar_toward(b_limited, b_ref, self.b_smoothing_alpha)
        return float(b), float(c_deg)

    def tip_to_stage(self, p_tip: np.ndarray, tangent: Optional[np.ndarray] = None, prev_c: Optional[float] = None, prev_b: Optional[float] = None) -> Tuple[np.ndarray, float, float]:
        if self.write_mode == "cartesian":
            p_stage = self.clamp_stage(np.asarray(p_tip, dtype=float), "cartesian_tip_to_stage")
            b, c = (0.0, 0.0) if tangent is None else self.bc_for_tangent(tangent, prev_c=prev_c, prev_b=prev_b)
            return p_stage, float(b), float(c)

        if tangent is None:
            b = float(self.fixed_b)
            c = float(self.fixed_c)
        else:
            b, c = self.bc_for_tangent(tangent, prev_c=prev_c, prev_b=prev_b)
        p_stage = stage_xyz_for_tip(self.cal, np.asarray(p_tip, dtype=float), b, c)
        return self.clamp_stage(p_stage, "tip_to_stage"), float(b), float(c)

    def write_move(
        self,
        p_stage: np.ndarray,
        b: float,
        c: float,
        feed: float,
        comment: Optional[str] = None,
        u_value: Optional[float] = None,
    ) -> None:
        if comment:
            self.f.write(f"; {comment}\n")
        p_stage = np.asarray(p_stage, dtype=float)

        # Let the rotary axis run at its own feed so short XYZ steps are not
        # throttled by a larger required C reorientation.
        if self.write_mode == "calibrated" and abs(float(c) - self.cur_c) > 1e-9:
            self.f.write(f"G1 {self.cal.c_axis}{float(c):.3f} F{self.c_feed:.0f}\n")
            self.cur_c = float(c)
            self.c_min_used = min(self.c_min_used, self.cur_c)
            self.c_max_used = max(self.c_max_used, self.cur_c)

        axes: List[Tuple[str, float]] = [
            (self.cal.x_axis, float(p_stage[0])),
            (self.cal.y_axis, float(p_stage[1])),
            (self.cal.z_axis, float(p_stage[2])),
        ]
        if self.write_mode == "calibrated":
            axes.append((self.cal.b_axis, float(b)))
        if u_value is not None:
            axes.append((self.cal.u_axis, float(u_value)))
        self.f.write(f"G1 {_fmt_axes_move(axes)} F{float(feed):.0f}\n")
        self.cur_stage_xyz = p_stage.copy()
        self.cur_b = float(b)
        self.command_min = np.minimum(self.command_min, self.cur_stage_xyz)
        self.command_max = np.maximum(self.command_max, self.cur_stage_xyz)
        self.b_min_used = min(self.b_min_used, self.cur_b)
        self.b_max_used = max(self.b_max_used, self.cur_b)

    def command_ranges(self) -> Dict[str, Tuple[float, float]]:
        return {
            "x": (float(self.command_min[0]), float(self.command_max[0])),
            "y": (float(self.command_min[1]), float(self.command_max[1])),
            "z": (float(self.command_min[2]), float(self.command_max[2])),
            "b": (float(self.b_min_used), float(self.b_max_used)),
            "c": (float(self.c_min_used), float(self.c_max_used)),
        }

    def move_to_tip(self, p_tip: np.ndarray, tangent: Optional[np.ndarray], feed: float, comment: Optional[str] = None) -> None:
        p_stage, b, c = self.tip_to_stage(np.asarray(p_tip, dtype=float), tangent=tangent, prev_c=self.cur_c, prev_b=self.cur_b)
        self.write_move(p_stage, b, c, feed, comment=comment)
        self.cur_tip_xyz = np.asarray(p_tip, dtype=float).copy()
        self.last_tip_tangent = None if tangent is None else np.asarray(tangent, dtype=float).copy()

    def sync_tip_state_from_stage(self, tangent: Optional[np.ndarray] = None) -> None:
        if self.cur_stage_xyz is None:
            self.cur_tip_xyz = None
            self.last_tip_tangent = None if tangent is None else np.asarray(tangent, dtype=float).copy()
            return
        if self.write_mode == "cartesian":
            self.cur_tip_xyz = np.asarray(self.cur_stage_xyz, dtype=float).copy()
        else:
            self.cur_tip_xyz = tip_xyz_for_stage(self.cal, self.cur_stage_xyz, self.cur_b, self.cur_c)
        self.last_tip_tangent = None if tangent is None else np.asarray(tangent, dtype=float).copy()

    def apply_group_displacement(self, displacement: GroupDisplacement, default_feed: float, label: str) -> None:
        if not displacement.enabled:
            return
        if self.cur_stage_xyz is None:
            raise RuntimeError(f"Cannot apply group displacement for {label} before any stage pose has been established.")

        delta_stage = np.asarray(displacement.delta_stage_xyz, dtype=float)
        if (
            float(np.linalg.norm(delta_stage)) <= 1e-12
            and abs(float(displacement.delta_b_deg)) <= 1e-12
            and abs(float(displacement.delta_c_deg)) <= 1e-12
        ):
            return
        target_stage = np.asarray(self.cur_stage_xyz, dtype=float) + delta_stage
        target_b = float(self.cur_b) + float(displacement.delta_b_deg)
        target_c = float(self.cur_c) + float(displacement.delta_c_deg)
        feed = float(default_feed if displacement.feed is None else displacement.feed)
        note_suffix = f" ({displacement.note})" if displacement.note else ""
        self.write_move(
            self.clamp_stage(target_stage, f"{label}_post_group_displacement"),
            target_b,
            target_c,
            feed,
            comment=(
                f"{label}: manual post-group displacement "
                f"dXYZ=[{delta_stage[0]:.3f}, {delta_stage[1]:.3f}, {delta_stage[2]:.3f}] "
                f"dB={float(displacement.delta_b_deg):.3f} dC={float(displacement.delta_c_deg):.3f}"
                f"{note_suffix}"
            ),
        )
        self.sync_tip_state_from_stage(tangent=None)

    def current_equation_phase(self) -> str:
        return _normalize_phase_key(self.cal.active_phase)

    def skeleton_points_for_pose(self, p_tip: np.ndarray, tangent: Optional[np.ndarray], prev_c: float, prev_b: float) -> Tuple[np.ndarray, float, float]:
        if self.robot_skeleton_ref is None:
            return np.zeros((0, 3), dtype=float), float(prev_b), float(prev_c)
        _, b_cmd, c_cmd = self.tip_to_stage(np.asarray(p_tip, dtype=float), tangent=tangent, prev_c=prev_c, prev_b=prev_b)
        pts_world = compute_tracked_skeleton_world(
            self.robot_skeleton_ref,
            b_cmd=b_cmd,
            c_cmd_deg=c_cmd,
            tip_world=np.asarray(p_tip, dtype=float),
            equation_phase=self.current_equation_phase(),
        )
        return pts_world, float(b_cmd), float(c_cmd)

    def skeleton_collides_with_printed_paths(self, skeleton_points: np.ndarray, allow_tip_segment_contact: bool = False) -> bool:
        if self.robot_skeleton_ref is None or len(self.printed_paths) == 0:
            return False

        pts = np.asarray(skeleton_points, dtype=float)
        if len(pts) < 2:
            return False
        if allow_tip_segment_contact and len(pts) > 2:
            pts = pts[:-1]
            if len(pts) < 2:
                return False

        skeleton_radius = max(0.0, 0.5 * float(self.robot_skeleton_ref.diameter_mm))
        for printed in self.printed_paths:
            clearance = skeleton_radius + float(printed.radius_mm) + float(self.skeleton_collision_clearance)
            if polyline_polyline_min_distance(pts, printed.points_xyz_mm) < clearance:
                return True
        return False

    def route_collides_with_printed_paths(self, moves: Sequence[PlannedTipMove]) -> bool:
        if self.robot_skeleton_ref is None or len(self.printed_paths) == 0 or self.cur_tip_xyz is None:
            return False

        prev_tip = np.asarray(self.cur_tip_xyz, dtype=float).copy()
        prev_tangent = None if self.last_tip_tangent is None else np.asarray(self.last_tip_tangent, dtype=float).copy()
        prev_c = float(self.cur_c)
        prev_b = float(self.cur_b)

        for move in moves:
            target_tip = np.asarray(move.tip_xyz, dtype=float)
            seg_len = float(np.linalg.norm(target_tip - prev_tip))
            steps = max(1, int(math.ceil(seg_len / float(self.skeleton_collision_sample_step_mm))))
            for step_idx in range(1, steps + 1):
                t = step_idx / float(steps)
                p_tip = prev_tip + t * (target_tip - prev_tip)
                tangent = move.tangent
                if tangent is not None and prev_tangent is not None:
                    tangent = normalize((1.0 - t) * prev_tangent + t * np.asarray(move.tangent, dtype=float))
                elif tangent is None:
                    tangent = prev_tangent
                skeleton_points, prev_b, prev_c = self.skeleton_points_for_pose(p_tip, tangent=tangent, prev_c=prev_c, prev_b=prev_b)
                allow_contact = bool(move.allow_tip_segment_contact and step_idx == steps)
                if self.skeleton_collides_with_printed_paths(skeleton_points, allow_tip_segment_contact=allow_contact):
                    return True
            prev_tip = target_tip
            prev_tangent = None if move.tangent is None else np.asarray(move.tangent, dtype=float).copy()
        return False

    def printed_tip_bbox(self) -> Optional[Tuple[float, float, float, float, float]]:
        if not self.printed_tip_points:
            return None
        pts = np.vstack(self.printed_tip_points)
        margin = float(self.travel_bbox_margin)
        return (
            float(np.min(pts[:, 0]) - margin),
            float(np.max(pts[:, 0]) + margin),
            float(np.min(pts[:, 1]) - margin),
            float(np.max(pts[:, 1]) + margin),
            float(np.max(pts[:, 2])),
        )

    def travel_crosses_printed_bbox(self, target_tip: np.ndarray) -> bool:
        if self.cur_tip_xyz is None:
            return False
        bbox = self.printed_tip_bbox()
        if bbox is None:
            return False
        x_min, x_max, y_min, y_max, _ = bbox
        return segment_intersects_xy_bbox(self.cur_tip_xyz[:2], np.asarray(target_tip, dtype=float)[:2], (x_min, x_max, y_min, y_max))

    def travel_crosses_printed_bbox_from(self, start_tip: np.ndarray, target_tip: np.ndarray) -> bool:
        bbox = self.printed_tip_bbox()
        if bbox is None:
            return False
        x_min, x_max, y_min, y_max, _ = bbox
        return segment_intersects_xy_bbox(np.asarray(start_tip, dtype=float)[:2], np.asarray(target_tip, dtype=float)[:2], (x_min, x_max, y_min, y_max))

    def expanded_travel_bbox_xy(self, bbox: Tuple[float, float, float, float, float]) -> Tuple[float, float, float, float]:
        x_min, x_max, y_min, y_max, _ = bbox
        clearance = max(0.0, float(self.travel_edge_clearance))
        return (x_min - clearance, x_max + clearance, y_min - clearance, y_max + clearance)

    def nearest_bbox_edge_point(self, point_xy: np.ndarray, bbox_xy: Tuple[float, float, float, float]) -> np.ndarray:
        x_min, x_max, y_min, y_max = bbox_xy
        x = float(point_xy[0])
        y = float(point_xy[1])
        candidates = [
            np.array([x_min, float(np.clip(y, y_min, y_max))], dtype=float),
            np.array([x_max, float(np.clip(y, y_min, y_max))], dtype=float),
            np.array([float(np.clip(x, x_min, x_max)), y_min], dtype=float),
            np.array([float(np.clip(x, x_min, x_max)), y_max], dtype=float),
        ]
        return min(candidates, key=lambda p: float(np.linalg.norm(p - np.asarray(point_xy, dtype=float))))

    def perimeter_s(self, point_xy: np.ndarray, bbox_xy: Tuple[float, float, float, float]) -> float:
        x_min, x_max, y_min, y_max = bbox_xy
        x = float(point_xy[0])
        y = float(point_xy[1])
        w = max(0.0, x_max - x_min)
        h = max(0.0, y_max - y_min)
        eps = 1e-7
        if abs(y - y_min) <= eps:
            return float(np.clip(x - x_min, 0.0, w))
        if abs(x - x_max) <= eps:
            return float(w + np.clip(y - y_min, 0.0, h))
        if abs(y - y_max) <= eps:
            return float(w + h + np.clip(x_max - x, 0.0, w))
        return float(2.0 * w + h + np.clip(y_max - y, 0.0, h))

    def perimeter_corners_between(self, s0: float, s1: float, bbox_xy: Tuple[float, float, float, float], clockwise: bool) -> List[np.ndarray]:
        x_min, x_max, y_min, y_max = bbox_xy
        w = max(0.0, x_max - x_min)
        h = max(0.0, y_max - y_min)
        perim = max(1e-12, 2.0 * (w + h))
        corners = [
            (0.0, np.array([x_min, y_min], dtype=float)),
            (w, np.array([x_max, y_min], dtype=float)),
            (w + h, np.array([x_max, y_max], dtype=float)),
            (2.0 * w + h, np.array([x_min, y_max], dtype=float)),
            (perim, np.array([x_min, y_min], dtype=float)),
        ]
        if clockwise:
            end = s1 if s1 >= s0 else s1 + perim
            out = []
            for cs, pt in corners + [(cs + perim, pt) for cs, pt in corners[1:]]:
                if s0 + 1e-9 < cs < end - 1e-9:
                    out.append(pt.copy())
            return out

        # Counter-clockwise is clockwise on the unwrapped interval from s1 to s0.
        end = s0 if s0 >= s1 else s0 + perim
        rev = []
        for cs, pt in corners + [(cs + perim, pt) for cs, pt in corners[1:]]:
            if s1 + 1e-9 < cs < end - 1e-9:
                rev.append(pt.copy())
        return list(reversed(rev))

    def bbox_edge_route_xy(self, start_xy: np.ndarray, target_xy: np.ndarray, bbox_xy: Tuple[float, float, float, float]) -> List[np.ndarray]:
        exit_xy = self.nearest_bbox_edge_point(start_xy, bbox_xy)
        entry_xy = self.nearest_bbox_edge_point(target_xy, bbox_xy)
        s0 = self.perimeter_s(exit_xy, bbox_xy)
        s1 = self.perimeter_s(entry_xy, bbox_xy)
        x_min, x_max, y_min, y_max = bbox_xy
        perim = max(1e-12, 2.0 * ((x_max - x_min) + (y_max - y_min)))
        cw_len = s1 - s0 if s1 >= s0 else s1 + perim - s0
        ccw_len = perim - cw_len
        clockwise = cw_len <= ccw_len
        route = [exit_xy]
        route.extend(self.perimeter_corners_between(s0, s1, bbox_xy, clockwise=clockwise))
        route.append(entry_xy)
        deduped: List[np.ndarray] = []
        for pt in route:
            if not deduped or float(np.linalg.norm(pt - deduped[-1])) > 1e-9:
                deduped.append(pt)
        return deduped

    def plan_tip_route_with_printed_bbox_clearance(
        self,
        p_tip: np.ndarray,
        tangent: Optional[np.ndarray],
        feed: float,
        label: str,
        start_tip: Optional[np.ndarray] = None,
        start_tangent: Optional[np.ndarray] = None,
    ) -> List[PlannedTipMove]:
        target = np.asarray(p_tip, dtype=float)
        start_tip_arr = self.cur_tip_xyz if start_tip is None else np.asarray(start_tip, dtype=float)
        start_tangent_arr = self.last_tip_tangent if start_tangent is None else np.asarray(start_tangent, dtype=float)
        bbox = self.printed_tip_bbox()
        if (
            not self.enable_travel_bbox_clearance
            or start_tip_arr is None
            or bbox is None
            or not self.travel_crosses_printed_bbox_from(start_tip_arr, target)
        ):
            return [
                PlannedTipMove(
                    tip_xyz=target,
                    tangent=None if tangent is None else np.asarray(tangent, dtype=float).copy(),
                    feed=feed,
                    comment=f"{label}: travel without printed-bbox crossing",
                )
            ]

        _, _, _, _, max_printed_z = bbox
        safe_z = max(float(start_tip_arr[2]), float(target[2]), max_printed_z + float(self.travel_clearance_above_printed_z))
        route_xy = self.bbox_edge_route_xy(start_tip_arr[:2], target[:2], self.expanded_travel_bbox_xy(bbox))
        moves: List[PlannedTipMove] = []
        exit_tip = np.array([float(route_xy[0][0]), float(route_xy[0][1]), float(start_tip_arr[2])], dtype=float)
        moves.append(PlannedTipMove(
            tip_xyz=exit_tip,
            tangent=None if start_tangent_arr is None else np.asarray(start_tangent_arr, dtype=float).copy(),
            feed=feed,
            comment=f"{label}: horizontal exit to printed bbox edge before lift",
        ))
        lift_tip = exit_tip.copy()
        lift_tip[2] = safe_z
        moves.append(PlannedTipMove(
            tip_xyz=lift_tip,
            tangent=None if start_tangent_arr is None else np.asarray(start_tangent_arr, dtype=float).copy(),
            feed=feed,
            comment=f"{label}: clearance lift above printed bbox",
        ))
        for i, xy in enumerate(route_xy[1:], start=2):
            edge_tip = np.array([float(xy[0]), float(xy[1]), safe_z], dtype=float)
            moves.append(PlannedTipMove(
                tip_xyz=edge_tip,
                tangent=None if tangent is None else np.asarray(tangent, dtype=float).copy(),
                feed=feed,
                comment=f"{label}: clearance edge route waypoint {i}",
            ))
        edge_descend = np.array([float(route_xy[-1][0]), float(route_xy[-1][1]), float(target[2])], dtype=float)
        moves.append(PlannedTipMove(
            tip_xyz=edge_descend,
            tangent=None if tangent is None else np.asarray(tangent, dtype=float).copy(),
            feed=feed,
            comment=f"{label}: clearance descend at printed bbox edge",
        ))
        moves.append(PlannedTipMove(
            tip_xyz=target,
            tangent=None if tangent is None else np.asarray(tangent, dtype=float).copy(),
            feed=self.approach_feed,
            comment=f"{label}: final approach from bbox edge",
        ))
        return moves

    def execute_planned_tip_moves(self, moves: Sequence[PlannedTipMove]) -> None:
        for move in moves:
            self.move_to_tip(move.tip_xyz, tangent=move.tangent, feed=move.feed, comment=move.comment)

    def record_printed_path(self, points: np.ndarray, radius_mm: float = 0.0) -> None:
        pts = np.asarray(points, dtype=float)
        if len(pts) > 0:
            self.printed_tip_points.append(pts.copy())
            self.printed_paths.append(PrintedVasculaturePath(points_xyz_mm=pts.copy(), radius_mm=max(0.0, float(radius_mm))))

    def pressure_preload_before_print(self) -> None:
        if self.emit_extrusion and self.pressure_offset_mm > 0.0 and not self.pressure_charged:
            self.pressure_charged = True
            self.f.write("; pressure preload before print pass\n")
            self.f.write(f"G1 {self.cal.u_axis}{self.u_cmd_actual():.3f} F{self.pressure_advance_feed:.0f}\n")
            if self.preflow_dwell_ms > 0:
                self.f.write(f"G4 P{self.preflow_dwell_ms}\n")

    def pressure_release_after_print(self) -> None:
        if self.emit_extrusion and self.pressure_offset_mm > 0.0 and self.pressure_charged:
            if self.node_dwell_ms > 0:
                self.f.write("; end-of-pass dwell for node formation / liquid flow\n")
                self.f.write(f"G4 P{self.node_dwell_ms}\n")
            self.pressure_charged = False
            self.f.write("; pressure release after print pass\n")
            self.f.write(f"G1 {self.cal.u_axis}{self.u_cmd_actual():.3f} F{self.pressure_retract_feed:.0f}\n")

    def approach_start_from_side(
        self,
        start_tip: np.ndarray,
        start_tangent: np.ndarray,
        far_clearance: float,
        near_clearance: float,
        retreat_clearance: float,
        side_lift_z: float,
        label: str,
    ) -> None:
        start_tip = np.asarray(start_tip, dtype=float)
        base_side = side_vector_from_tangent(start_tangent, fallback=self.last_tip_tangent)
        approach_distance = max(float(far_clearance), float(self.fine_approach_distance))

        direction_candidates: List[np.ndarray] = []
        for angle_deg in (0.0, 180.0, 45.0, -45.0, 90.0, -90.0, 135.0, -135.0):
            rotated_xy = rotate_xy_vector(base_side[:2], angle_deg)
            cand = normalize(np.array([rotated_xy[0], rotated_xy[1], 0.0], dtype=float))
            if float(np.linalg.norm(cand[:2])) <= 1e-12:
                continue
            if any(float(np.linalg.norm(cand - existing)) < 1e-6 for existing in direction_candidates):
                continue
            direction_candidates.append(cand)

        lift_candidates = [float(side_lift_z)]
        extra_lift = float(side_lift_z) + float(self.travel_clearance_above_printed_z)
        if abs(extra_lift - lift_candidates[0]) > 1e-9:
            lift_candidates.append(extra_lift)

        fallback_moves: Optional[List[PlannedTipMove]] = None
        best_moves: Optional[List[PlannedTipMove]] = None
        best_score = float("inf")

        for lift_z in lift_candidates:
            for side in direction_candidates:
                candidate_moves: List[PlannedTipMove] = []
                far_tip = start_tip - side * approach_distance + np.array([0.0, 0.0, lift_z], dtype=float)
                fine_start_tip = start_tip - side * float(self.fine_approach_distance)
                near_tip = start_tip - side * float(near_clearance)

                if self.cur_tip_xyz is not None:
                    cur_side = side_vector_from_tangent(
                        self.last_tip_tangent if self.last_tip_tangent is not None else start_tangent,
                        fallback=side,
                    )
                    retreat_tip = np.asarray(self.cur_tip_xyz, dtype=float) + cur_side * float(retreat_clearance) + np.array([0.0, 0.0, lift_z], dtype=float)
                    candidate_moves.append(PlannedTipMove(
                        tip_xyz=retreat_tip,
                        tangent=self.last_tip_tangent if self.last_tip_tangent is None else np.asarray(self.last_tip_tangent, dtype=float).copy(),
                        feed=self.approach_feed,
                        comment=f"{label}: side retreat from previous vessel",
                    ))

                candidate_moves.extend(
                    self.plan_tip_route_with_printed_bbox_clearance(
                        far_tip,
                        tangent=start_tangent,
                        feed=self.travel_feed,
                        label=f"{label}: side approach far",
                        start_tip=self.cur_tip_xyz if len(candidate_moves) == 0 else candidate_moves[-1].tip_xyz,
                        start_tangent=self.last_tip_tangent if len(candidate_moves) == 0 else candidate_moves[-1].tangent,
                    )
                )
                if float(np.linalg.norm(fine_start_tip - far_tip)) > 1e-9:
                    candidate_moves.append(PlannedTipMove(
                        tip_xyz=fine_start_tip,
                        tangent=np.asarray(start_tangent, dtype=float).copy(),
                        feed=self.approach_feed,
                        comment=f"{label}: approach to 5mm fine-ramp start",
                    ))
                if float(near_clearance) > 0.0 and float(near_clearance) < float(self.fine_approach_distance):
                    candidate_moves.append(PlannedTipMove(
                        tip_xyz=near_tip,
                        tangent=np.asarray(start_tangent, dtype=float).copy(),
                        feed=self.fine_approach_feed,
                        comment=f"{label}: fine approach ramp near",
                    ))
                candidate_moves.append(PlannedTipMove(
                    tip_xyz=start_tip,
                    tangent=np.asarray(start_tangent, dtype=float).copy(),
                    feed=self.fine_approach_feed,
                    comment=f"{label}: fine approach ramp to node",
                    allow_tip_segment_contact=True,
                ))

                if fallback_moves is None:
                    fallback_moves = candidate_moves

                if self.route_collides_with_printed_paths(candidate_moves):
                    continue

                prev_route_tip = np.asarray(self.cur_tip_xyz, dtype=float).copy() if self.cur_tip_xyz is not None else start_tip.copy()
                route_score = 0.0
                for move in candidate_moves:
                    move_tip = np.asarray(move.tip_xyz, dtype=float)
                    route_score += float(np.linalg.norm(move_tip - prev_route_tip))
                    prev_route_tip = move_tip
                if route_score < best_score:
                    best_score = route_score
                    best_moves = candidate_moves

        chosen_moves = best_moves
        if chosen_moves is None:
            chosen_moves = fallback_moves or []
            if chosen_moves:
                self.warnings.append(
                    f"WARNING: {label} side approach could not find a skeleton-clear route; using the lowest-cost fallback route."
                )

        self.execute_planned_tip_moves(chosen_moves)

    def emit_vasculature_write_start(
        self,
        label: str,
        points: np.ndarray,
        tangents: np.ndarray,
        extrusion_multiplier: float,
    ) -> None:
        start_b = b_angle_comment_from_tangent(tangents[0])
        end_b = b_angle_comment_from_tangent(tangents[-1])
        self.f.write(
            "; VASCULATURE_WRITE_START "
            f"path_id={label} "
            f"point_count={len(points)} "
            f"extrusion_multiplier={float(extrusion_multiplier):.6f} "
            f"tip_start_x={float(points[0, 0]):.6f} "
            f"tip_start_y={float(points[0, 1]):.6f} "
            f"tip_start_z={float(points[0, 2]):.6f} "
            f"tip_end_x={float(points[-1, 0]):.6f} "
            f"tip_end_y={float(points[-1, 1]):.6f} "
            f"tip_end_z={float(points[-1, 2]):.6f} "
            f"physical_b_start_deg={start_b:.6f} "
            f"physical_b_end_deg={end_b:.6f} "
            "physical_b_convention=0_negZ_90_horizontal_180_posZ\n"
        )

    def emit_vasculature_write_end(self, label: str) -> None:
        self.f.write(f"; VASCULATURE_WRITE_END path_id={label}\n")

    def print_polyline(self, points: np.ndarray, tangents: np.ndarray, extrusion_multiplier: float, label: str, path_radius_mm: float = 0.0) -> None:
        if len(points) < 2:
            return

        self.pressure_preload_before_print()
        mode_note = "Cartesian centerline" if self.write_mode == "cartesian" else "calibration-based exact tip tracking"
        self.f.write(f"; print {label} ({mode_note})\n")
        self.emit_vasculature_write_start(label, points, tangents, extrusion_multiplier)

        last_tip = np.asarray(points[0], dtype=float).copy()
        self.cur_tip_xyz = last_tip.copy()
        self.last_tip_tangent = np.asarray(tangents[0], dtype=float).copy()

        for i in range(1, len(points)):
            p0 = np.asarray(points[i - 1], dtype=float)
            p1 = np.asarray(points[i], dtype=float)
            seg_t0 = np.asarray(tangents[i - 1], dtype=float)
            seg_t1 = np.asarray(tangents[i], dtype=float)
            for s in range(1, self.edge_samples + 1):
                t = s / float(self.edge_samples)
                p_tip = p0 + t * (p1 - p0)
                tangent = normalize((1.0 - t) * seg_t0 + t * seg_t1)
                p_stage, b, c = self.tip_to_stage(p_tip, tangent=tangent, prev_c=self.cur_c)

                u_val = None
                if self.emit_extrusion:
                    tip_seg_len = float(np.linalg.norm(p_tip - last_tip))
                    self.u_material_abs += self.extrusion_per_mm * float(extrusion_multiplier) * tip_seg_len
                    u_val = self.u_cmd_actual()

                self.write_move(p_stage, b, c, self.print_feed, comment=None, u_value=u_val)
                self.cur_tip_xyz = p_tip.copy()
                self.last_tip_tangent = tangent.copy()
                last_tip = p_tip.copy()

        self.emit_vasculature_write_end(label)
        self.pressure_release_after_print()
        self.record_printed_path(points, radius_mm=path_radius_mm)


# ---------------- Top-level generation ----------------

def build_tangents_for_points(points: np.ndarray, smooth_window: int, centerline_smooth_window: int = DEFAULT_CENTERLINE_SMOOTH_WINDOW) -> np.ndarray:
    tangent_points = smooth_centerline_points(points, window=centerline_smooth_window)
    tangents = np.zeros_like(tangent_points)
    for i in range(len(tangent_points)):
        tangents[i] = tangent_for_index(tangent_points, i, smooth_window=smooth_window)
    return tangents


def build_print_paths(
    vessels: List[Vessel],
    vessel_order_mode: str,
    simplify_paths: bool,
    chain_merge_threshold: float,
    path_resample_spacing: float,
    path_geometry_smooth_window: int,
    point_merge_tol: float,
    min_group_length_mm: float = DEFAULT_MIN_GROUP_LENGTH_MM,
    force_bottom_to_top: bool = False,
    branch_start_overlap_mm: float = 0.0,
    branch_overlap_tangent_window_mm: float = DEFAULT_BRANCH_OVERLAP_TANGENT_WINDOW_MM,
    preferred_main_vessel_ids: Optional[List[int]] = None,
) -> List[PrintPath]:
    ordered_vessels = vessel_print_order(vessels, order_mode=vessel_order_mode)
    if simplify_paths:
        raw_paths = simplify_vessel_paths(
            ordered_vessels,
            merge_threshold=chain_merge_threshold,
            order_mode=vessel_order_mode,
            preferred_main_vessel_ids=preferred_main_vessel_ids,
        )
    else:
        raw_paths = [
            PrintPath(
                path_id=f"vessel_{v.vessel_id:03d}",
                source_vessel_ids=[v.vessel_id],
                points=(v.ordered_points if v.ordered_points is not None else v.points).copy(),
                root_vessel_id=v.vessel_id,
                parent_vessel_id=v.parent_id,
                depth=v.depth,
                is_main=(v.parent_id is None),
                radius_like=float(v.radius_like),
                connection_xyz=None,
                connection_z=None,
            )
            for v in ordered_vessels
        ]

    prepared_paths: List[PrintPath] = []
    for path in raw_paths:
        pts = prepare_print_path_points(
            np.asarray(path.points, dtype=float),
            point_merge_tol=point_merge_tol,
            resample_spacing=path_resample_spacing,
            geometry_smooth_window=path_geometry_smooth_window,
        )
        if len(pts) < 2:
            continue
        source_vessel_ids = list(path.source_vessel_ids)
        if bool(force_bottom_to_top) and float(pts[0, 2]) > float(pts[-1, 2]):
            pts = pts[::-1].copy()
            source_vessel_ids = list(reversed(source_vessel_ids))
        prepared_paths.append(PrintPath(
            path_id=path.path_id,
            source_vessel_ids=source_vessel_ids,
            points=pts,
            root_vessel_id=path.root_vessel_id,
            parent_vessel_id=path.parent_vessel_id,
            depth=path.depth,
            is_main=path.is_main,
            radius_like=path.radius_like,
            connection_xyz=None if path.connection_xyz is None else np.asarray(path.connection_xyz, dtype=float).copy(),
            connection_z=path.connection_z,
        ))
    prepared_paths = order_paths_from_printed_tree(
        prepared_paths,
        connection_threshold=max(float(chain_merge_threshold) * 2.0, float(point_merge_tol) * 10.0),
    )
    final_paths = extend_branch_starts_into_printed_tree(
        prepared_paths,
        overlap_mm=float(branch_start_overlap_mm),
        point_merge_tol=float(point_merge_tol),
        tangent_window_mm=float(branch_overlap_tangent_window_mm),
    )
    min_len = max(0.0, float(min_group_length_mm))
    if min_len <= 0.0:
        return final_paths
    return [path for path in final_paths if polyline_length(path.points) >= min_len]


def summarize_vessels(vessels: List[Vessel]) -> Dict[str, Any]:
    roots = [v for v in vessels if v.parent_id is None]
    children = [v for v in vessels if v.parent_id is not None]
    return {
        "vessel_count": len(vessels),
        "root_count": len(roots),
        "child_count": len(children),
        "min_radius": float(min(v.radius_like for v in vessels)) if vessels else 0.0,
        "max_radius": float(max(v.radius_like for v in vessels)) if vessels else 0.0,
    }


def summarize_paths(paths: List[PrintPath]) -> Dict[str, Any]:
    main_paths = [p for p in paths if p.is_main]
    secondary_paths = [p for p in paths if not p.is_main]
    return {
        "path_count": len(paths),
        "main_path_count": len(main_paths),
        "secondary_path_count": len(secondary_paths),
    }


def polyline_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())


def polyline_arc_lengths(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.zeros(0, dtype=float)
    if len(pts) == 1:
        return np.zeros(1, dtype=float)
    seg = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def point_on_polyline_at_arclength(polyline: np.ndarray, s_query: float) -> np.ndarray:
    pts = np.asarray(polyline, dtype=float)
    if len(pts) == 0:
        return np.zeros(3, dtype=float)
    if len(pts) == 1:
        return pts[0].copy()

    arc = polyline_arc_lengths(pts)
    total = float(arc[-1])
    s = float(np.clip(float(s_query), 0.0, total))
    idx = int(np.searchsorted(arc, s, side="right") - 1)
    idx = min(max(idx, 0), len(pts) - 2)
    seg_len = float(arc[idx + 1] - arc[idx])
    if seg_len <= 1e-12:
        return pts[idx].copy()
    t = (s - float(arc[idx])) / seg_len
    return pts[idx] + t * (pts[idx + 1] - pts[idx])


def projected_point_and_arclength_on_polyline(point: np.ndarray, polyline: np.ndarray) -> Tuple[np.ndarray, float, np.ndarray, float]:
    p = np.asarray(point, dtype=float).reshape(3)
    pts = np.asarray(polyline, dtype=float)
    if len(pts) == 0:
        return p.copy(), 0.0, np.array([0.0, 0.0, 1.0], dtype=float), float("inf")
    if len(pts) == 1:
        return pts[0].copy(), 0.0, np.array([0.0, 0.0, 1.0], dtype=float), float(np.linalg.norm(p - pts[0]))

    arc = polyline_arc_lengths(pts)
    best_point = pts[0].copy()
    best_s = 0.0
    best_tangent = normalize(pts[1] - pts[0])
    best_dist = float("inf")
    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            continue
        t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
        q = a + t * ab
        d = float(np.linalg.norm(p - q))
        if d < best_dist:
            best_dist = d
            best_point = q
            best_s = float(arc[i] + t * (arc[i + 1] - arc[i]))
            best_tangent = normalize(ab)
    return best_point, best_s, best_tangent, best_dist


def nearest_point_and_tangent_on_polyline(point: np.ndarray, polyline: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    best_point, _, best_tangent, best_dist = projected_point_and_arclength_on_polyline(point, polyline)
    return best_point, best_tangent, best_dist


def extend_branch_starts_into_printed_tree(
    paths: List[PrintPath],
    overlap_mm: float,
    point_merge_tol: float,
    tangent_window_mm: float,
) -> List[PrintPath]:
    overlap = float(overlap_mm)
    if overlap <= 0.0 or not paths:
        return paths

    out: List[PrintPath] = []
    printed_paths: List[np.ndarray] = []
    for path in paths:
        pts = np.asarray(path.points, dtype=float)
        if path.is_main or len(printed_paths) == 0 or len(pts) < 2:
            out.append(path)
            printed_paths.append(pts)
            continue

        start = path.connection_xyz if path.connection_xyz is not None else pts[0]
        best_point = start.copy()
        best_projected_s = 0.0
        best_polyline: Optional[np.ndarray] = None
        best_dist = float("inf")
        for printed in printed_paths:
            q, projected_s, _, dist = projected_point_and_arclength_on_polyline(start, printed)
            if dist < best_dist:
                best_point = q
                best_projected_s = projected_s
                best_polyline = printed
                best_dist = dist

        overlap_point = best_point.copy()
        if best_polyline is not None and len(best_polyline) >= 2:
            tangent_window = max(float(tangent_window_mm), float(overlap))
            upstream_s = max(0.0, float(best_projected_s) - tangent_window)
            upstream_point = point_on_polyline_at_arclength(best_polyline, upstream_s)
            averaged_tangent = normalize(best_point - upstream_point)
            if float(np.linalg.norm(averaged_tangent)) > 1e-12:
                overlap_point = best_point - averaged_tangent * overlap
        branch_pts = pts.copy()
        if float(np.linalg.norm(branch_pts[0] - best_point)) > float(point_merge_tol):
            branch_pts = np.vstack([best_point, branch_pts])
        if float(np.linalg.norm(branch_pts[0] - overlap_point)) > float(point_merge_tol):
            branch_pts = np.vstack([overlap_point, branch_pts])

        updated = PrintPath(
            path_id=path.path_id,
            source_vessel_ids=list(path.source_vessel_ids),
            points=branch_pts,
            root_vessel_id=path.root_vessel_id,
            parent_vessel_id=path.parent_vessel_id,
            depth=path.depth,
            is_main=path.is_main,
            radius_like=path.radius_like,
            connection_xyz=path.connection_xyz,
            connection_z=path.connection_z,
        )
        out.append(updated)
        printed_paths.append(branch_pts)
    return out


def order_paths_from_printed_tree(paths: List[PrintPath], connection_threshold: float) -> List[PrintPath]:
    if len(paths) <= 2:
        return paths

    main_paths = [p for p in paths if p.is_main]
    remaining = [p for p in paths if not p.is_main]
    if not main_paths:
        return paths

    ordered: List[PrintPath] = list(main_paths)
    printed_paths = [np.asarray(p.points, dtype=float) for p in ordered]
    threshold = max(float(connection_threshold), 1e-9)

    def path_connection_point(path: PrintPath) -> np.ndarray:
        if path.connection_xyz is not None:
            return np.asarray(path.connection_xyz, dtype=float)
        return np.asarray(path.points[0], dtype=float)

    while remaining:
        connectable: List[Tuple[float, float, int, PrintPath]] = []
        fallback: List[Tuple[float, int, PrintPath]] = []
        for idx, path in enumerate(remaining):
            conn = path_connection_point(path)
            best_dist = min(nearest_point_and_tangent_on_polyline(conn, printed)[2] for printed in printed_paths)
            conn_z = path.connection_z if path.connection_z is not None else float(conn[2])
            fallback.append((float(conn_z), idx, path))
            if best_dist <= threshold:
                connectable.append((float(conn_z), best_dist, idx, path))

        if connectable:
            _, _, idx, path = min(connectable, key=lambda item: (item[0], item[1], int(min(item[3].source_vessel_ids))))
        else:
            _, idx, path = min(fallback, key=lambda item: (item[0], int(min(item[2].source_vessel_ids))))

        ordered.append(path)
        printed_paths.append(np.asarray(path.points, dtype=float))
        remaining.pop(idx)

    return ordered


def export_print_groups(out_path: str, paths: List[PrintPath]) -> None:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        f.write("# Generated print groups from vasc_1.py\n")
        f.write("# group_number\tpath_id\tvessel_ids_in_order\tpath_length_mm\tstart_z\tend_z\tmin_z\tmax_z\tconnection_z\tis_main\tdepth\tparent_vessel_id\n")
        for i, path in enumerate(paths, start=1):
            pts = np.asarray(path.points, dtype=float)
            vessel_ids = ",".join(str(vessel_id) for vessel_id in path.source_vessel_ids)
            parent = "" if path.parent_vessel_id is None else str(path.parent_vessel_id)
            connection_z = "" if path.connection_z is None else f"{float(path.connection_z):.6f}"
            f.write(
                f"{i}\t"
                f"{path.path_id}\t"
                f"{vessel_ids}\t"
                f"{polyline_length(pts):.6f}\t"
                f"{float(pts[0, 2]):.6f}\t"
                f"{float(pts[-1, 2]):.6f}\t"
                f"{float(np.min(pts[:, 2])):.6f}\t"
                f"{float(np.max(pts[:, 2])):.6f}\t"
                f"{connection_z}\t"
                f"{int(bool(path.is_main))}\t"
                f"{int(path.depth)}\t"
                f"{parent}\n"
            )


def machine_pose_for_tip(
    cal: Calibration,
    p_tip: np.ndarray,
    write_mode: str,
    orientation_mode: str,
    bc_solve_samples: int,
    tangent: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float, float]:
    write_mode = str(write_mode).strip().lower()
    if write_mode == "cartesian":
        return (float(p_tip[0]), float(p_tip[1]), float(p_tip[2]), 0.0, 0.0)

    if tangent is None or str(orientation_mode).strip().lower() == "fixed":
        b = solve_b_for_target_tip_angle(cal, 0.0, search_samples=bc_solve_samples)
        c = 0.0
    else:
        c = c_angle_from_tangent(tangent, prev_c=0.0, azimuth_offset_deg=cal.c_180_deg)
        b = solve_b_for_target_tip_angle(cal, desired_physical_b_angle_from_tangent(tangent), search_samples=bc_solve_samples)
    p_stage = stage_xyz_for_tip(cal, np.asarray(p_tip, dtype=float), float(b), float(c))
    return (float(p_stage[0]), float(p_stage[1]), float(p_stage[2]), float(b), float(c))


def write_vessel_gcode(
    out_path: str,
    vessels: List[Vessel],
    cal: Calibration,
    machine_start_pose: Optional[Tuple[float, float, float, float, float]],
    machine_end_pose: Optional[Tuple[float, float, float, float, float]],
    travel_feed: float,
    approach_feed: float,
    fine_approach_feed: float,
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
    bbox: Dict[str, float],
    write_mode: str,
    orientation_mode: str,
    bc_solve_samples: int,
    extrusion_multiplier_main: float,
    extrusion_multiplier_branch: float,
    side_approach_far: float,
    side_approach_near: float,
    side_retreat: float,
    side_lift_z: float,
    tangent_smooth_window: int,
    centerline_smooth_window: int,
    b_max_step_deg: float,
    c_max_step_deg: float,
    b_smoothing_alpha: float,
    c_smoothing_alpha: float,
    min_tangent_xy_for_c: float,
    rotate_y_deg: float,
    rotate_origin: Sequence[float],
    geometry_scale: float,
    vessel_order_mode: str,
    simplify_paths: bool,
    chain_merge_threshold: float,
    path_resample_spacing: float,
    path_geometry_smooth_window: int,
    point_merge_tol: float,
    min_group_length_mm: float = DEFAULT_MIN_GROUP_LENGTH_MM,
    force_bottom_to_top: bool = False,
    branch_start_overlap_mm: float = 0.0,
    branch_overlap_tangent_window_mm: float = DEFAULT_BRANCH_OVERLAP_TANGENT_WINDOW_MM,
    preferred_main_vessel_ids: Optional[List[int]] = None,
    travel_clearance_above_printed_z: float = DEFAULT_TRAVEL_CLEARANCE_ABOVE_PRINTED_Z,
    travel_bbox_margin: float = DEFAULT_TRAVEL_BBOX_MARGIN,
    travel_edge_clearance: float = DEFAULT_TRAVEL_EDGE_CLEARANCE,
    enable_travel_bbox_clearance: bool = True,
    fine_approach_distance: float = DEFAULT_FINE_APPROACH_DISTANCE,
    robot_skeleton_ref: Optional[RobotSkeletonReference] = None,
    skeleton_collision_clearance: float = DEFAULT_SKELETON_COLLISION_CLEARANCE,
    skeleton_collision_sample_step_mm: float = DEFAULT_SKELETON_COLLISION_SAMPLE_STEP_MM,
    group_displacements: Optional[Dict[int, GroupDisplacement]] = None,
) -> Dict[str, Tuple[float, float]]:
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    emit_extrusion = float(extrusion_per_mm) != 0.0

    paths = build_print_paths(
        vessels=vessels,
        vessel_order_mode=vessel_order_mode,
        simplify_paths=simplify_paths,
        chain_merge_threshold=chain_merge_threshold,
        path_resample_spacing=path_resample_spacing,
        path_geometry_smooth_window=path_geometry_smooth_window,
        point_merge_tol=point_merge_tol,
        min_group_length_mm=min_group_length_mm,
        force_bottom_to_top=force_bottom_to_top,
        branch_start_overlap_mm=branch_start_overlap_mm,
        branch_overlap_tangent_window_mm=branch_overlap_tangent_window_mm,
        preferred_main_vessel_ids=preferred_main_vessel_ids,
    )
    if not paths:
        raise ValueError("No printable vessel paths were generated.")

    vessel_summary = summarize_vessels(vessels)
    path_summary = summarize_paths(paths)
    first_points = np.asarray(paths[0].points, dtype=float)
    first_tangents = build_tangents_for_points(first_points, smooth_window=tangent_smooth_window, centerline_smooth_window=centerline_smooth_window)

    if machine_start_pose is None:
        machine_start_pose = machine_pose_for_tip(cal, first_points[0], write_mode=write_mode, orientation_mode=orientation_mode, bc_solve_samples=bc_solve_samples, tangent=first_tangents[0])
    if machine_end_pose is None:
        machine_end_pose = machine_pose_for_tip(cal, first_points[0], write_mode=write_mode, orientation_mode=orientation_mode, bc_solve_samples=bc_solve_samples, tangent=first_tangents[0])

    with open(out_path, "w") as f:
        f.write("; generated by vessel_tangent_gcode.py\n")
        if write_mode == "calibrated":
            f.write("; calibration-based tip-position planning: stage = tip - offset_tip(B,C)\n")
        else:
            f.write("; Cartesian writing mode: stage XYZ follows vessel XYZ directly\n")
        f.write("; centerline-only continuous polylines\n")
        if simplify_paths:
            f.write("; endpoint-connected vessel segments simplified into longer local-main branches\n")
        else:
            f.write("; one continuous centerline polyline is written for each vessel block\n")
        f.write(f"; vessel_count = {vessel_summary['vessel_count']}\n")
        f.write(f"; root_count = {vessel_summary['root_count']}\n")
        f.write(f"; child_count = {vessel_summary['child_count']}\n")
        f.write(f"; path_count = {path_summary['path_count']} (main={path_summary['main_path_count']}, secondary={path_summary['secondary_path_count']})\n")
        f.write(f"; radius_like range = [{vessel_summary['min_radius']:.6f}, {vessel_summary['max_radius']:.6f}]\n")
        f.write(f"; axes: X->{cal.x_axis}, Y->{cal.y_axis}, Z->{cal.z_axis}, B->{cal.b_axis}, C->{cal.c_axis}, U->{cal.u_axis}\n")
        f.write(f"; write_mode = {write_mode}\n")
        f.write(f"; orientation_mode = {orientation_mode}\n")
        f.write("; physical B convention = 0=>-Z, 90=>horizontal, 180=>+Z\n")
        f.write(f"; vessel_order_mode = {vessel_order_mode}\n")
        f.write(f"; force_bottom_to_top = {int(bool(force_bottom_to_top))}\n")
        if preferred_main_vessel_ids:
            f.write(f"; preferred_main_vessel_ids = {preferred_main_vessel_ids}\n")
        f.write(f"; tangent_smooth_window = {tangent_smooth_window}\n")
        f.write(f"; centerline_smooth_window = {centerline_smooth_window}\n")
        f.write(f"; path_geometry_smooth_window = {int(path_geometry_smooth_window)}\n")
        f.write(f"; b_max_step_deg = {float(b_max_step_deg):.6f}\n")
        f.write(f"; c_max_step_deg = {float(c_max_step_deg):.6f}\n")
        f.write(f"; b_smoothing_alpha = {float(b_smoothing_alpha):.6f}\n")
        f.write(f"; c_smoothing_alpha = {float(c_smoothing_alpha):.6f}\n")
        f.write(f"; min_tangent_xy_for_c = {float(min_tangent_xy_for_c):.6f}\n")
        f.write(f"; geometry_rotate_y_deg = {float(rotate_y_deg):.6f}\n")
        f.write(f"; geometry_rotate_origin = [{float(rotate_origin[0]):.6f}, {float(rotate_origin[1]):.6f}, {float(rotate_origin[2]):.6f}]\n")
        f.write(f"; geometry_scale = {float(geometry_scale):.6f}\n")
        f.write(f"; chain_merge_threshold = {chain_merge_threshold:.6f}\n")
        f.write(f"; min_group_length_mm = {float(min_group_length_mm):.6f}\n")
        f.write(f"; branch_start_overlap_mm = {float(branch_start_overlap_mm):.6f}\n")
        f.write(f"; branch_overlap_tangent_window_mm = {float(branch_overlap_tangent_window_mm):.6f}\n")
        f.write(f"; travel_clearance_above_printed_z = {float(travel_clearance_above_printed_z):.6f}\n")
        f.write(f"; travel_bbox_margin = {float(travel_bbox_margin):.6f}\n")
        f.write(f"; travel_edge_clearance = {float(travel_edge_clearance):.6f}\n")
        f.write(f"; enable_travel_bbox_clearance = {int(bool(enable_travel_bbox_clearance))}\n")
        f.write(f"; fine_approach_distance = {float(fine_approach_distance):.6f}\n")
        if robot_skeleton_ref is not None:
            f.write(f"; robot_skeleton_file = {robot_skeleton_ref.source_path}\n")
            f.write(f"; robot_skeleton_diameter_mm = {float(robot_skeleton_ref.diameter_mm):.6f}\n")
            f.write(f"; skeleton_collision_clearance = {float(skeleton_collision_clearance):.6f}\n")
            f.write(f"; skeleton_collision_sample_step_mm = {float(skeleton_collision_sample_step_mm):.6f}\n")
        if group_displacements:
            f.write(f"; manual_group_displacements = {len(group_displacements)} configured groups\n")
        f.write(f"; selected_fit_model = {cal.selected_fit_model or 'legacy-polynomial'}\n")
        f.write(f"; active_phase = {cal.active_phase}\n")
        for warning in calibration_model_range_warnings(cal):
            f.write(f"; WARNING: {warning}\n")
        f.write(f"; feeds: travel={travel_feed:.1f}, approach={approach_feed:.1f}, fine_approach={fine_approach_feed:.1f}, print={print_feed:.1f}, C-only={c_feed:.1f}\n")
        f.write(f"; side approach: far={side_approach_far:.3f}, near={side_approach_near:.3f}, retreat={side_retreat:.3f}, lift_z={side_lift_z:.3f}\n")
        f.write("G90\n")
        if emit_extrusion:
            f.write("M82\n")
            f.write(f"G92 {cal.u_axis}0\n")
            if abs(float(prime_mm)) > 0.0:
                f.write(f"G1 {cal.u_axis}{float(prime_mm):.3f} F{max(60.0, float(pressure_advance_feed)):.0f} ; prime material\n")

        g = GCodeWriter(
            fh=f,
            cal=cal,
            bbox=bbox,
            travel_feed=travel_feed,
            approach_feed=approach_feed,
            fine_approach_feed=fine_approach_feed,
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
            write_mode=write_mode,
            orientation_mode=orientation_mode,
            bc_solve_samples=bc_solve_samples,
            b_max_step_deg=b_max_step_deg,
            c_max_step_deg=c_max_step_deg,
            b_smoothing_alpha=b_smoothing_alpha,
            c_smoothing_alpha=c_smoothing_alpha,
            min_tangent_xy_for_c=min_tangent_xy_for_c,
            travel_clearance_above_printed_z=travel_clearance_above_printed_z,
            travel_bbox_margin=travel_bbox_margin,
            travel_edge_clearance=travel_edge_clearance,
            enable_travel_bbox_clearance=enable_travel_bbox_clearance,
            fine_approach_distance=fine_approach_distance,
            robot_skeleton_ref=robot_skeleton_ref,
            skeleton_collision_clearance=skeleton_collision_clearance,
            skeleton_collision_sample_step_mm=skeleton_collision_sample_step_mm,
        )

        msx, msy, msz, msb, msc = [float(v) for v in machine_start_pose]
        mex, mey, mez, meb, mec = [float(v) for v in machine_end_pose]

        g.write_move(np.array([msx, msy, msz], dtype=float), msb, msc, travel_feed, comment="startup: move to anchored machine start pose")

        for group_idx, path in enumerate(paths, start=1):
            points = np.asarray(path.points, dtype=float)
            if len(points) < 2:
                continue
            tangents = build_tangents_for_points(points, smooth_window=tangent_smooth_window, centerline_smooth_window=centerline_smooth_window)
            mult = extrusion_multiplier_main if path.is_main else extrusion_multiplier_branch

            f.write("; ------------------------------------------------------------\n")
            f.write(f"; group_number = {group_idx}\n")
            f.write(f"; {path.path_id}: source_vessels={path.source_vessel_ids}, depth={path.depth}, parent={path.parent_vessel_id}, is_main={path.is_main}\n")
            f.write(f"; centerline polyline points = {len(points)}\n")
            f.write("; this vessel/path is emitted as a single continuous print pass\n")

            g.approach_start_from_side(
                start_tip=points[0],
                start_tangent=tangents[0],
                far_clearance=side_approach_far,
                near_clearance=side_approach_near,
                retreat_clearance=side_retreat,
                side_lift_z=side_lift_z,
                label=path.path_id,
            )
            g.print_polyline(
                points,
                tangents,
                extrusion_multiplier=mult,
                label=path.path_id,
                path_radius_mm=max(0.0, float(path.radius_like) * float(geometry_scale)),
            )
            displacement = None if group_displacements is None else group_displacements.get(group_idx)
            if displacement is not None:
                g.apply_group_displacement(
                    displacement=displacement,
                    default_feed=travel_feed,
                    label=f"group_{group_idx:03d}_{path.path_id}",
                )

        g.write_move(np.array([mex, mey, mez], dtype=float), meb, mec, travel_feed, comment="shutdown: move to anchored machine end pose")
        if emit_extrusion:
            f.write(f"G92 {cal.u_axis}0\n")

        if g.warnings:
            f.write("; ==================== warnings ====================\n")
            for w in g.warnings:
                f.write(f"; {w}\n")

        ranges = g.command_ranges()
        f.write("; ==================== command ranges ====================\n")
        f.write(
            "; "
            f"{cal.x_axis}=[{ranges['x'][0]:.6f}, {ranges['x'][1]:.6f}], "
            f"{cal.y_axis}=[{ranges['y'][0]:.6f}, {ranges['y'][1]:.6f}], "
            f"{cal.z_axis}=[{ranges['z'][0]:.6f}, {ranges['z'][1]:.6f}], "
            f"{cal.b_axis}=[{ranges['b'][0]:.6f}, {ranges['b'][1]:.6f}], "
            f"{cal.c_axis}=[{ranges['c'][0]:.6f}, {ranges['c'][1]:.6f}]\n"
        )
        return ranges


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Generate calibrated or Cartesian G-code from a vessel centerline text file using exact tip-position planning, "
            "one continuous centerline polyline per vessel by default, true 3D axial tangent following, and optional endpoint-chain simplification for main-branch merging."
        )
    )
    ap.add_argument("--vessels", required=True, help="Path to the vessel text file.")
    ap.add_argument("--vessel-ids", nargs="+", default=["All"], help="Vessel IDs to write. Use All to write every vessel. Accepts space-separated IDs or comma-separated IDs.")
    ap.add_argument("--main-vessel-ids", nargs="+", default=None, help="Optional ordered vessel IDs to force as the first/main printed tree path. Accepts space-separated IDs or comma-separated IDs.")
    ap.add_argument("--calibration", required=False, default=None, help="Path to the calibration JSON. Required for --write-mode calibrated; optional for --write-mode cartesian.")
    ap.add_argument("--robot-skeleton-file", default=None, help="Optional tracked robot skeleton JSON exported by shadow_calibration.py. When provided, side-approach travel is replanned to keep the robot body clear of already printed vasculature.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code file.")
    ap.add_argument("--groups-out", default=None, help="Optional text file listing generated print groups and vessel IDs in print order.")
    ap.add_argument(
        "--group-displacements-file",
        default=None,
        help=(
            "Optional text file defining one manual post-group displacement per generated print group. "
            "Each non-comment row must be: group_number dx dy dz [db_deg] [dc_deg] [feed_mm_min] [enabled]. "
            "The displacement is applied in machine/stage coordinates immediately after that group is printed."
        ),
    )
    ap.add_argument("--rotate-y-deg", type=float, default=0.0, help="Rotate input vessel tip geometry around the Y axis before ordering and writing.")
    ap.add_argument("--rotate-origin-x", type=float, default=0.0)
    ap.add_argument("--rotate-origin-y", type=float, default=0.0)
    ap.add_argument("--rotate-origin-z", type=float, default=0.0)
    ap.add_argument("--geometry-scale", "--scale", type=float, default=1.0, help="Isotropic scale factor for vessel XYZ geometry, applied about the vasculature XY-centroid/lowest-Z anchor before optional alignment.")
    ap.add_argument(
        "--align-lowest-centroid",
        nargs=3,
        type=float,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Translate vessel tip geometry so its XY centroid and lowest Z point are at this target XYZ.",
    )

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
    ap.add_argument("--use-explicit-machine-start-end", action="store_true", help="Use the explicit machine start/end poses instead of anchoring both to the first written point.")

    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED)
    ap.add_argument("--approach-feed", type=float, default=DEFAULT_APPROACH_FEED)
    ap.add_argument("--fine-approach-feed", type=float, default=DEFAULT_FINE_APPROACH_FEED)
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED)
    ap.add_argument("--c-feed", type=float, default=DEFAULT_C_FEED)
    ap.add_argument("--edge-samples", type=int, default=DEFAULT_EDGE_SAMPLES)
    ap.add_argument("--bc-solve-samples", type=int, default=DEFAULT_BC_SOLVE_SAMPLES)

    ap.add_argument("--write-mode", choices=["calibrated", "cartesian"], default="calibrated")
    ap.add_argument("--orientation-mode", choices=["tangent", "fixed"], default="tangent", help="tangent = align B/C to the local 3D vessel tangent; fixed = hold a constant orientation.")
    ap.add_argument("--tangent-smooth-window", type=int, default=DEFAULT_TANGENT_SMOOTH_WINDOW, help="Neighborhood size used when differentiating the smoothed 3D centerline into local tangents.")
    ap.add_argument("--centerline-smooth-window", type=int, default=DEFAULT_CENTERLINE_SMOOTH_WINDOW, help="Optional moving-average half-window applied in XYZ before tangent estimation. 0 keeps the vessel geometry exact.")
    ap.add_argument("--path-geometry-smooth-window", type=int, default=DEFAULT_PATH_GEOMETRY_SMOOTH_WINDOW, help="Optional moving-average half-window applied to the actual emitted XYZ path after resampling. Increase this to reduce local line oscillations.")
    ap.add_argument("--b-max-step-deg", type=float, default=DEFAULT_B_MAX_STEP_DEG, help="Maximum allowed B change per emitted move in tangent mode. Use 0 to disable B step limiting.")
    ap.add_argument("--c-max-step-deg", type=float, default=DEFAULT_C_MAX_STEP_DEG, help="Maximum allowed C change per emitted move in tangent mode. Use 0 to disable C step limiting.")
    ap.add_argument("--b-smoothing-alpha", type=float, default=DEFAULT_B_SMOOTHING_ALPHA, help="Low-pass smoothing alpha for B commands in tangent mode. 1 disables smoothing; smaller values damp local B oscillation.")
    ap.add_argument("--c-smoothing-alpha", type=float, default=DEFAULT_C_SMOOTHING_ALPHA, help="Low-pass smoothing alpha for C commands in tangent mode. 1 disables smoothing; smaller values damp local C oscillation.")
    ap.add_argument("--min-tangent-xy-for-c", type=float, default=DEFAULT_MIN_TANGENT_XY, help="Minimum XY tangent magnitude needed to update C. Below this, C is held to avoid azimuth spin on near-vertical segments.")
    ap.add_argument("--path-resample-spacing", type=float, default=DEFAULT_PATH_RESAMPLE_SPACING, help="Optional arc-length spacing used to densify each vessel centerline before writing. 0 disables resampling.")
    ap.add_argument("--point-merge-tol", type=float, default=DEFAULT_POINT_MERGE_TOL, help="Tolerance used to remove duplicate consecutive points from a vessel polyline.")
    ap.add_argument("--min-group-length-mm", type=float, default=DEFAULT_MIN_GROUP_LENGTH_MM, help="Omit printable vessel groups/paths shorter than this final polyline length in mm.")
    ap.add_argument("--vessel-order-mode", choices=["ascending_id", "ascending_start_z", "bottom_up_hierarchy", "lowest_longest_roots"], default="ascending_id")

    ap.add_argument("--simplify-endpoint-chains", dest="simplify_endpoint_chains", action="store_true", default=False, help="Merge endpoint-connected vessel segments into longer continuous main-branch polylines.")
    ap.add_argument("--no-simplify-endpoint-chains", dest="simplify_endpoint_chains", action="store_false", help="Disable endpoint-chain simplification and write each vessel block separately.")
    ap.add_argument("--chain-merge-threshold", type=float, default=DEFAULT_NODE_ATTACH_THRESHOLD, help="Endpoint distance threshold for merging adjacent vessel segments into a continuous main branch.")
    ap.add_argument("--force-bottom-to-top", action="store_true", help="Reverse each generated print path when needed so it starts at the lower-Z endpoint and ends at the higher-Z endpoint.")
    ap.add_argument("--branch-start-overlap-mm", type=float, default=0.0, help="For branch paths, prepend this much axial overlap into the already printed tree at the connection node.")
    ap.add_argument("--branch-overlap-tangent-window-mm", type=float, default=DEFAULT_BRANCH_OVERLAP_TANGENT_WINDOW_MM, help="Distance upstream from a branch attachment used to estimate the general branch direction for --branch-start-overlap-mm.")

    ap.add_argument("--extrusion-per-mm", type=float, default=DEFAULT_EXTRUSION_PER_MM)
    ap.add_argument("--prime-mm", type=float, default=DEFAULT_PRIME_MM)
    ap.add_argument("--extrusion-multiplier-main", type=float, default=1.0)
    ap.add_argument("--extrusion-multiplier-branch", type=float, default=1.0)

    ap.add_argument("--pressure-offset-mm", type=float, default=DEFAULT_PRESSURE_OFFSET_MM)
    ap.add_argument("--pressure-advance-feed", type=float, default=DEFAULT_PRESSURE_ADVANCE_FEED)
    ap.add_argument("--pressure-retract-feed", type=float, default=DEFAULT_PRESSURE_RETRACT_FEED)
    ap.add_argument("--preflow-dwell-ms", type=int, default=DEFAULT_PREFLOW_DWELL_MS)
    ap.add_argument("--node-dwell-ms", type=int, default=DEFAULT_NODE_DWELL_MS)

    ap.add_argument("--node-attach-threshold", type=float, default=DEFAULT_NODE_ATTACH_THRESHOLD, help="Endpoint-to-parent distance threshold used to classify a vessel as a branch.")
    ap.add_argument("--side-approach-far", type=float, default=DEFAULT_SIDE_APPROACH_FAR)
    ap.add_argument("--side-approach-near", type=float, default=DEFAULT_SIDE_APPROACH_NEAR)
    ap.add_argument("--side-retreat", type=float, default=DEFAULT_SIDE_RETREAT)
    ap.add_argument("--side-lift-z", type=float, default=DEFAULT_SIDE_LIFT_Z)
    ap.add_argument("--travel-clearance-above-printed-z", type=float, default=DEFAULT_TRAVEL_CLEARANCE_ABOVE_PRINTED_Z, help="When a travel move crosses the printed XY bounding box, lift to this many mm above the highest printed tip Z before crossing.")
    ap.add_argument("--travel-bbox-margin", type=float, default=DEFAULT_TRAVEL_BBOX_MARGIN, help="XY margin added around the already printed vasculature bounding box for travel-crossing checks.")
    ap.add_argument("--travel-edge-clearance", type=float, default=DEFAULT_TRAVEL_EDGE_CLEARANCE, help="Extra XY clearance outside the already printed bounding box used for edge-routing travel moves.")
    ap.add_argument("--enable-travel-bbox-clearance", dest="enable_travel_bbox_clearance", action="store_true", help="Enable printed-bounding-box-aware travel rerouting.")
    ap.add_argument("--disable-travel-bbox-clearance", dest="enable_travel_bbox_clearance", action="store_false", help="Disable printed-bounding-box-aware travel rerouting and use direct travel moves.")
    ap.set_defaults(enable_travel_bbox_clearance=True)
    ap.add_argument("--fine-approach-distance", type=float, default=DEFAULT_FINE_APPROACH_DISTANCE, help="Distance from the vessel start/node over which approach moves use --fine-approach-feed.")
    ap.add_argument("--skeleton-collision-clearance", type=float, default=DEFAULT_SKELETON_COLLISION_CLEARANCE, help="Additional radial clearance added between the robot skeleton body and already printed vessel centerlines during side-approach replanning.")
    ap.add_argument("--skeleton-collision-sample-step-mm", type=float, default=DEFAULT_SKELETON_COLLISION_SAMPLE_STEP_MM, help="Sampling step used to check robot-skeleton collisions along candidate travel and side-approach routes.")

    ap.add_argument("--bbox-x-min", type=float, default=DEFAULT_BBOX_X_MIN)
    ap.add_argument("--bbox-x-max", type=float, default=DEFAULT_BBOX_X_MAX)
    ap.add_argument("--bbox-y-min", type=float, default=DEFAULT_BBOX_Y_MIN)
    ap.add_argument("--bbox-y-max", type=float, default=DEFAULT_BBOX_Y_MAX)
    ap.add_argument("--bbox-z-min", type=float, default=DEFAULT_BBOX_Z_MIN)
    ap.add_argument("--bbox-z-max", type=float, default=DEFAULT_BBOX_Z_MAX)
    return ap


def main(args: argparse.Namespace) -> None:
    write_mode = str(args.write_mode).strip().lower()
    robot_skeleton_ref: Optional[RobotSkeletonReference] = None
    group_displacements = None if args.group_displacements_file is None else parse_group_displacements_file(str(args.group_displacements_file))
    vessels = parse_vessel_file(args.vessels)
    selected_vessel_ids = parse_vessel_id_selection(args.vessel_ids)
    vessels = filter_vessels_by_id(vessels, selected_vessel_ids)
    preferred_main_vessel_ids = None if args.main_vessel_ids is None else parse_vessel_id_selection(args.main_vessel_ids)
    rotate_origin = (
        float(args.rotate_origin_x),
        float(args.rotate_origin_y),
        float(args.rotate_origin_z),
    )
    transform_vessel_geometry(
        vessels,
        rotate_y_deg=float(args.rotate_y_deg),
        rotate_origin=rotate_origin,
    )
    scale_origin, geometry_scale = scale_vessel_geometry(vessels, float(args.geometry_scale))
    alignment_info: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None
    if args.align_lowest_centroid is not None:
        alignment_info = align_vessel_lowest_centroid(vessels, args.align_lowest_centroid)
    infer_parent_relationships(vessels, attach_threshold=float(args.node_attach_threshold))
    orient_vessels(vessels)

    if write_mode == "calibrated":
        if not args.calibration:
            raise ValueError("--calibration is required when --write-mode calibrated")
        cal = load_calibration(args.calibration)
    else:
        cal = Calibration(
            pr=np.zeros(1),
            pz=np.zeros(1),
            py_off=None,
            pa=None,
            b_min=0.0,
            b_max=180.0,
            x_axis="X", y_axis="Y", z_axis="Z", b_axis="B", c_axis="C", u_axis="U",
            c_180_deg=180.0,
        )

    robot_skeleton_path = None if args.robot_skeleton_file is None else Path(str(args.robot_skeleton_file)).expanduser()
    if robot_skeleton_path is None and args.calibration:
        robot_skeleton_path = resolve_robot_skeleton_from_calibration(str(args.calibration))
    if robot_skeleton_path is not None:
        robot_skeleton_ref = load_robot_skeleton_reference(str(robot_skeleton_path))

    bbox = {
        "x_min": float(args.bbox_x_min),
        "x_max": float(args.bbox_x_max),
        "y_min": float(args.bbox_y_min),
        "y_max": float(args.bbox_y_max),
        "z_min": float(args.bbox_z_min),
        "z_max": float(args.bbox_z_max),
    }

    preview_paths = build_print_paths(
        vessels=vessels,
        vessel_order_mode=str(args.vessel_order_mode),
        simplify_paths=bool(args.simplify_endpoint_chains),
        chain_merge_threshold=float(args.chain_merge_threshold),
        path_resample_spacing=float(args.path_resample_spacing),
        path_geometry_smooth_window=int(args.path_geometry_smooth_window),
        point_merge_tol=float(args.point_merge_tol),
        min_group_length_mm=float(args.min_group_length_mm),
        force_bottom_to_top=bool(args.force_bottom_to_top),
        branch_start_overlap_mm=float(args.branch_start_overlap_mm),
        branch_overlap_tangent_window_mm=float(args.branch_overlap_tangent_window_mm),
        preferred_main_vessel_ids=preferred_main_vessel_ids,
    )
    if not preview_paths:
        raise ValueError("No printable vessel paths were generated.")

    first_points = np.asarray(preview_paths[0].points, dtype=float)
    first_tangents = build_tangents_for_points(first_points, smooth_window=int(args.tangent_smooth_window), centerline_smooth_window=int(args.centerline_smooth_window))

    if args.use_explicit_machine_start_end:
        machine_start_pose = (
            float(args.machine_start_x),
            float(args.machine_start_y),
            float(args.machine_start_z),
            float(args.machine_start_b),
            float(args.machine_start_c),
        )
        machine_end_pose = (
            float(args.machine_end_x),
            float(args.machine_end_y),
            float(args.machine_end_z),
            float(args.machine_end_b),
            float(args.machine_end_c),
        )
    else:
        anchored_pose = machine_pose_for_tip(
            cal,
            first_points[0],
            write_mode=write_mode,
            orientation_mode=str(args.orientation_mode),
            bc_solve_samples=int(args.bc_solve_samples),
            tangent=first_tangents[0],
        )
        machine_start_pose = anchored_pose
        machine_end_pose = anchored_pose

    command_ranges = write_vessel_gcode(
        out_path=str(args.out),
        vessels=vessels,
        cal=cal,
        machine_start_pose=machine_start_pose,
        machine_end_pose=machine_end_pose,
        travel_feed=float(args.travel_feed),
        approach_feed=float(args.approach_feed),
        fine_approach_feed=float(args.fine_approach_feed),
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
        write_mode=write_mode,
        orientation_mode=str(args.orientation_mode),
        bc_solve_samples=int(args.bc_solve_samples),
        extrusion_multiplier_main=float(args.extrusion_multiplier_main),
        extrusion_multiplier_branch=float(args.extrusion_multiplier_branch),
        side_approach_far=float(args.side_approach_far),
        side_approach_near=float(args.side_approach_near),
        side_retreat=float(args.side_retreat),
        side_lift_z=float(args.side_lift_z),
        tangent_smooth_window=int(args.tangent_smooth_window),
        centerline_smooth_window=int(args.centerline_smooth_window),
        b_max_step_deg=float(args.b_max_step_deg),
        c_max_step_deg=float(args.c_max_step_deg),
        b_smoothing_alpha=float(args.b_smoothing_alpha),
        c_smoothing_alpha=float(args.c_smoothing_alpha),
        min_tangent_xy_for_c=float(args.min_tangent_xy_for_c),
        rotate_y_deg=float(args.rotate_y_deg),
        rotate_origin=rotate_origin,
        geometry_scale=float(args.geometry_scale),
        vessel_order_mode=str(args.vessel_order_mode),
        simplify_paths=bool(args.simplify_endpoint_chains),
        chain_merge_threshold=float(args.chain_merge_threshold),
        path_resample_spacing=float(args.path_resample_spacing),
        path_geometry_smooth_window=int(args.path_geometry_smooth_window),
        point_merge_tol=float(args.point_merge_tol),
        min_group_length_mm=float(args.min_group_length_mm),
        force_bottom_to_top=bool(args.force_bottom_to_top),
        branch_start_overlap_mm=float(args.branch_start_overlap_mm),
        branch_overlap_tangent_window_mm=float(args.branch_overlap_tangent_window_mm),
        preferred_main_vessel_ids=preferred_main_vessel_ids,
        travel_clearance_above_printed_z=float(args.travel_clearance_above_printed_z),
        travel_bbox_margin=float(args.travel_bbox_margin),
        travel_edge_clearance=float(args.travel_edge_clearance),
        enable_travel_bbox_clearance=bool(args.enable_travel_bbox_clearance),
        fine_approach_distance=float(args.fine_approach_distance),
        robot_skeleton_ref=robot_skeleton_ref,
        skeleton_collision_clearance=float(args.skeleton_collision_clearance),
        skeleton_collision_sample_step_mm=float(args.skeleton_collision_sample_step_mm),
        group_displacements=group_displacements,
    )

    if args.groups_out:
        export_print_groups(str(args.groups_out), preview_paths)

    summary = summarize_vessels(vessels)
    path_summary = summarize_paths(preview_paths)
    print(f"Wrote G-code to {args.out}")
    if robot_skeleton_ref is not None:
        print(f"Robot skeleton collision avoidance enabled from {robot_skeleton_ref.source_path}")
    if args.groups_out:
        print(f"Wrote print groups to {args.groups_out}")
    if group_displacements:
        print(f"Loaded manual post-group displacements for {len(group_displacements)} groups from {args.group_displacements_file}")
    if abs(float(geometry_scale) - 1.0) > 1e-12:
        print(
            "Scaled vasculature XYZ geometry "
            f"by {float(geometry_scale):.6f} about "
            f"({scale_origin[0]:.6f}, {scale_origin[1]:.6f}, {scale_origin[2]:.6f})"
        )
    if alignment_info is not None:
        anchor_before, anchor_delta, anchor_after = alignment_info
        print(
            "Aligned vasculature anchor "
            f"(XY centroid, lowest Z) from "
            f"({anchor_before[0]:.6f}, {anchor_before[1]:.6f}, {anchor_before[2]:.6f}) "
            f"by delta ({anchor_delta[0]:.6f}, {anchor_delta[1]:.6f}, {anchor_delta[2]:.6f}) "
            f"to ({anchor_after[0]:.6f}, {anchor_after[1]:.6f}, {anchor_after[2]:.6f})"
        )
    print(
        "Gantry command ranges: "
        f"X=[{command_ranges['x'][0]:.6f}, {command_ranges['x'][1]:.6f}], "
        f"Y=[{command_ranges['y'][0]:.6f}, {command_ranges['y'][1]:.6f}], "
        f"Z=[{command_ranges['z'][0]:.6f}, {command_ranges['z'][1]:.6f}], "
        f"B=[{command_ranges['b'][0]:.6f}, {command_ranges['b'][1]:.6f}], "
        f"C=[{command_ranges['c'][0]:.6f}, {command_ranges['c'][1]:.6f}]"
    )
    path_note = "merged into" if args.simplify_endpoint_chains else "mapped to"
    print(
        f"Parsed {summary['vessel_count']} vessels "
        f"({summary['root_count']} roots, {summary['child_count']} branches); "
        f"{path_note} {path_summary['path_count']} printable paths; "
        f"radius_like range [{summary['min_radius']:.6f}, {summary['max_radius']:.6f}]"
    )
    if float(args.min_group_length_mm) > 0.0:
        print(f"Omitted printable groups shorter than {float(args.min_group_length_mm):.6f} mm")


if __name__ == "__main__":
    parser = build_arg_parser()
    main(parser.parse_args())
