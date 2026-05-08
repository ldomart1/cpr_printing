#!/usr/bin/env python3
"""
Interactive 3D tip-position visualizer for G-code using calibration JSON
(+ optional robot-link overlay from a robot configuration JSON).

Notes:
  - Matplotlib 3D is not true “blitting-capable” everywhere; so this script focuses on
    *reducing work per interaction* and *debouncing* slider callbacks.
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection

PULL_PHASE = "pull"
RELEASE_PHASE = "release"
PHASE_SWITCH_RE = re.compile(r";\s*EQUATION_SWITCH\s+([A-Z0-9_]+)")
WRITE_START_RE = re.compile(r";\s*VASCULATURE_WRITE_START\b", re.IGNORECASE)
WRITE_END_RE = re.compile(r";\s*VASCULATURE_WRITE_END\b", re.IGNORECASE)
WAVE_WRITE_START_RE = re.compile(r";\s*WAVE_WRITE_START\b", re.IGNORECASE)
WAVE_WRITE_END_RE = re.compile(r";\s*WAVE_WRITE_END\b", re.IGNORECASE)
PRESSURE_SOLENOID_RE = re.compile(r"(?<!\d)M42(?!\d).*?\bP\s*0\b.*?\bS\s*([01])(?:\.0+)?\b", re.IGNORECASE)


# ---------------- Defaults ----------------
DEFAULT_MIN_B = -5.0
DEFAULT_MAX_B = -0.0
DEFAULT_C0_DEG = 0.0
DEFAULT_FIGSIZE = (11.8, 8.4)
DEFAULT_OFFPLANE_SIGN = -1.0
DEFAULT_Y_OFFPLANE_FIT_MODEL = "avg_cubic"

DEFAULT_ROBOT_DIAMETER_MM = 3.0
DEFAULT_ROBOT_LINKS = 6
DEFAULT_PRESSURE_TOGGLE_U_MM = 2.0
DEFAULT_COLLISION_TIP_EXCLUSION_MM = 2.0

# UI tuning
UI_DEBOUNCE_MS_INDEX = 18
UI_DEBOUNCE_MS_ZOOM = 18
MAX_BACKGROUND_POINTS = 25000  # background path decimation for faster initial draw on huge files
ROBOT_TUBE_RADIAL_SEGMENTS = 16
ROBOT_TUBE_SAMPLES_PER_SEGMENT = 6
ROBOT_TUBE_COLOR = "#d9d9d9"
# -----------------------------------------


def remove_mpl_keymap_entries(keys):
    """Prevent Matplotlib toolbar shortcuts from also handling app keys."""
    for keymap_name in (
        "keymap.back",
        "keymap.forward",
        "keymap.home",
        "keymap.xscale",
        "keymap.yscale",
    ):
        try:
            existing = list(plt.rcParams.get(keymap_name, []))
            plt.rcParams[keymap_name] = [k for k in existing if k not in keys]
        except Exception:
            pass


# =========================
# Calibration model
# =========================
@dataclass
class Calibration:
    pr: np.ndarray
    pz: np.ndarray
    pa: np.ndarray
    py_off: Optional[np.ndarray]
    phase_models: Dict[str, Dict[str, dict]]
    default_phase: str
    b_min: float
    b_max: float
    tip_angle_min: float
    tip_angle_max: float
    pull_axis: str
    rot_axis: str
    x_axis: str
    z_axis: str
    c_180_deg: float
    b_home: float
    requested_offplane_fit_model: Optional[str]


def _polyval4(coeffs: np.ndarray, u) -> np.ndarray:
    a, b, c, d = coeffs
    u = np.asarray(u, dtype=float)
    return ((a * u + b) * u + c) * u + d


def _as_float_array(values) -> np.ndarray:
    return np.asarray(values, dtype=float).ravel()


def _pchip_slopes(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = _as_float_array(x)
    y = _as_float_array(y)
    n = x.size
    if n < 2:
        raise ValueError("PCHIP requires at least two knots")
    h = np.diff(x)
    delta = np.diff(y) / h
    d = np.zeros(n, dtype=float)
    if n == 2:
        d[:] = delta[0]
        return d

    for k in range(1, n - 1):
        if delta[k - 1] == 0.0 or delta[k] == 0.0 or np.sign(delta[k - 1]) != np.sign(delta[k]):
            d[k] = 0.0
        else:
            w1 = 2.0 * h[k] + h[k - 1]
            w2 = h[k] + 2.0 * h[k - 1]
            d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])

    d0 = ((2.0 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1])
    if np.sign(d0) != np.sign(delta[0]):
        d0 = 0.0
    elif np.sign(delta[0]) != np.sign(delta[1]) and abs(d0) > abs(3.0 * delta[0]):
        d0 = 3.0 * delta[0]
    d[0] = d0

    dn = ((2.0 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2])
    if np.sign(dn) != np.sign(delta[-1]):
        dn = 0.0
    elif np.sign(delta[-1]) != np.sign(delta[-2]) and abs(dn) > abs(3.0 * delta[-1]):
        dn = 3.0 * delta[-1]
    d[-1] = dn
    return d


def _eval_pchip(x_knots, y_knots, xq) -> np.ndarray:
    x = _as_float_array(x_knots)
    y = _as_float_array(y_knots)
    xq_arr = np.asarray(xq, dtype=float)
    flat = xq_arr.reshape(-1)
    slopes = _pchip_slopes(x, y)

    idx = np.searchsorted(x, flat, side="right") - 1
    idx = np.clip(idx, 0, x.size - 2)

    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    h = x1 - x0
    t = (flat - x0) / h
    t = np.clip(t, 0.0, 1.0)

    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2

    out = h00 * y0 + h10 * h * slopes[idx] + h01 * y1 + h11 * h * slopes[idx + 1]
    return out.reshape(xq_arr.shape)


def _eval_model(model: dict, xq) -> np.ndarray:
    model_type = str(model.get("model_type", "")).strip().lower()
    if model_type == "pchip":
        return _eval_pchip(model["x_knots"], model["y_knots"], xq)
    if model_type == "polynomial":
        coeffs = _as_float_array(model["coefficients"])
        return np.polyval(coeffs, np.asarray(xq, dtype=float))
    raise ValueError(f"Unsupported model type '{model_type}'")


def _eval_pchip_with_linear_extrap(model: dict, extrap_model: Optional[dict], xq) -> np.ndarray:
    if extrap_model is None:
        return _eval_model(model, xq)
    x = _as_float_array(model.get("x_knots", []))
    if x.size == 0:
        return _eval_model(model, xq)

    xq_arr = np.asarray(xq, dtype=float)
    out = np.asarray(_eval_model(model, xq_arr), dtype=float).copy()
    outside = (xq_arr < float(np.min(x))) | (xq_arr > float(np.max(x)))
    if np.any(outside):
        out = np.where(outside, _eval_model(extrap_model, xq_arr), out)
    return out


def _normalize_phase_name(name: Optional[str]) -> str:
    raw = str(name or PULL_PHASE).strip().lower()
    if raw.startswith(RELEASE_PHASE):
        return raw
    if raw.startswith(PULL_PHASE):
        return raw
    return PULL_PHASE


def _phase_alias(name: str) -> str:
    phase = _normalize_phase_name(name)
    return RELEASE_PHASE if phase.startswith(RELEASE_PHASE) else PULL_PHASE


def _phase_from_switch_token(token: str) -> Optional[str]:
    tok = str(token or "").strip().upper()
    if tok.endswith("_TO_PCHIP_RELEASE"):
        return RELEASE_PHASE
    if tok.endswith("_TO_PCHIP_PULL"):
        return PULL_PHASE
    return None


def _normalize_model_spec(model_spec):
    if not isinstance(model_spec, dict):
        return None
    out = dict(model_spec)
    if out.get("model_type") is not None:
        out["model_type"] = str(out["model_type"]).strip().lower()
    return out


def _select_named_model(models: Dict[str, dict], base_name: str, selected_fit_model: Optional[str]) -> Optional[dict]:
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


def load_calibration_json(json_path: str) -> dict:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")
    with p.open("r") as f:
        return json.load(f)


def load_calibration(json_path: str, requested_offplane_fit_model: Optional[str] = DEFAULT_Y_OFFPLANE_FIT_MODEL) -> Calibration:
    data = load_calibration_json(json_path)
    cubic = data["cubic_coefficients"]

    pr = np.array(cubic["r_coeffs"], dtype=float)
    pz = np.array(cubic["z_coeffs"], dtype=float)

    pa_raw = cubic.get("tip_angle_coeffs", None)
    if pa_raw is None:
        pa = np.array([0.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        pa = np.array(pa_raw, dtype=float)

    py_off_raw = cubic.get("offplane_y_coeffs", None)
    py_off = None if py_off_raw is None else np.array(py_off_raw, dtype=float)

    if pr.shape[0] != 4 or pz.shape[0] != 4 or pa.shape[0] != 4:
        raise ValueError("Expected 4 coeffs for r_coeffs, z_coeffs, and tip_angle_coeffs")

    phase_models_raw = data.get("fit_models_by_phase", {})
    phase_models: Dict[str, Dict[str, dict]] = {}
    for phase_name, models in phase_models_raw.items():
        norm_phase = _normalize_phase_name(phase_name)
        if isinstance(models, dict):
            phase_models[norm_phase] = models
            phase_models.setdefault(_phase_alias(norm_phase), models)

    default_phase = _normalize_phase_name(
        data.get("default_phase_for_legacy_access")
        or cubic.get("default_phase")
        or PULL_PHASE
    )
    selected_fit_model = data.get("selected_fit_model")
    selected_fit_model = None if selected_fit_model is None else str(selected_fit_model).strip().lower()
    selected_offplane_fit_model = data.get("selected_offplane_fit_model")
    selected_offplane_fit_model = None if selected_offplane_fit_model is None else str(selected_offplane_fit_model).strip().lower()
    requested_offplane_fit_model = (
        None if requested_offplane_fit_model is None else str(requested_offplane_fit_model).strip().lower()
    )

    motor_setup = data.get("motor_setup", {})
    duet_map = data.get("duet_axis_mapping", {})

    b_range = motor_setup.get("b_motor_position_range", [DEFAULT_MIN_B, DEFAULT_MAX_B])
    b_min, b_max = map(float, b_range)
    if b_min > b_max:
        b_min, b_max = b_max, b_min

    pull_axis = str(duet_map.get("pull_axis") or motor_setup.get("b_motor_axis") or "B")
    rot_axis = str(duet_map.get("rotation_axis") or motor_setup.get("rotation_axis") or "C")
    x_axis = str(duet_map.get("horizontal_axis") or motor_setup.get("horizontal_axis") or "X")
    z_axis = str(duet_map.get("vertical_axis") or motor_setup.get("vertical_axis") or "Z")

    c_180 = float(motor_setup.get("rotation_axis_180_deg", 180.0))
    b_home = float(motor_setup.get("b_motor_home_position", 0.0))

    tip_env = data.get("working_envelope", {}).get("tip_angle_range_deg")
    if isinstance(tip_env, list) and len(tip_env) == 2 and pa_raw is not None:
        tip_angle_min = float(min(tip_env))
        tip_angle_max = float(max(tip_env))
    else:
        bb = np.linspace(b_min, b_max, 801)
        aa = _polyval4(pa, bb)
        tip_angle_min = float(np.min(aa))
        tip_angle_max = float(np.max(aa))

    return Calibration(
        pr=pr, pz=pz, pa=pa, py_off=py_off,
        phase_models=phase_models, default_phase=default_phase,
        b_min=b_min, b_max=b_max,
        tip_angle_min=tip_angle_min, tip_angle_max=tip_angle_max,
        pull_axis=pull_axis, rot_axis=rot_axis,
        x_axis=x_axis, z_axis=z_axis,
        c_180_deg=c_180,
        b_home=b_home,
        requested_offplane_fit_model=requested_offplane_fit_model or selected_offplane_fit_model or selected_fit_model,
    )


# =========================
# Robot link configuration
# =========================
@dataclass
class RobotLinkConfig:
    diameter_mm: float
    n_links: int
    t_knots: np.ndarray
    show_default: bool
    anchor_mode: str
    shape_mode: str
    y_mode: str
    base_tangent_scale: float
    tip_tangent_scale: float


@dataclass
class RobotSkeletonReference:
    points_xyz_mm: np.ndarray
    diameter_mm: float
    source_path: Path
    phase_point_models: Optional[Dict[str, dict]] = None
    default_phase: Optional[str] = None


def _normalize_phase_key(name: Optional[str]) -> str:
    return str(name or PULL_PHASE).strip().lower()


def _safe_float(x, default):
    try:
        return float(x)
    except Exception:
        return float(default)


def _safe_int(x, default):
    try:
        return int(x)
    except Exception:
        return int(default)


def load_robot_config(robot_json_path: str) -> RobotLinkConfig:
    p = Path(robot_json_path)
    if not p.exists():
        raise FileNotFoundError(f"Robot config JSON not found: {robot_json_path}")

    with p.open("r") as f:
        data = json.load(f)

    diameter = _safe_float(data.get("diameter_mm", DEFAULT_ROBOT_DIAMETER_MM), DEFAULT_ROBOT_DIAMETER_MM)

    n_links = data.get("n_links_curve", None)
    if n_links is None:
        n_links = data.get("n_links", None)
    if n_links is None:
        n_links = DEFAULT_ROBOT_LINKS
    n_links = int(max(1, _safe_int(n_links, DEFAULT_ROBOT_LINKS)))

    t_knots = None
    knot_def = data.get("knot_definition", {})
    if isinstance(knot_def, dict) and "t_i" in knot_def:
        try:
            t_knots = np.array(knot_def["t_i"], dtype=float).ravel()
        except Exception:
            t_knots = None
    if t_knots is None or t_knots.size < 2:
        t_knots = np.linspace(0.0, 1.0, n_links + 1)
    t_knots[0] = 0.0
    t_knots[-1] = 1.0
    t_knots = np.clip(t_knots, 0.0, 1.0)

    show_default = bool(data.get("show_default", False))

    anchor_mode = str(data.get("anchor_mode", "tip")).strip().lower()
    if anchor_mode not in ("tip", "base"):
        anchor_mode = "tip"

    shape_mode = str(data.get("shape_mode", "hermite_angle")).strip().lower()
    if shape_mode not in ("hermite_angle", "poly_b"):
        shape_mode = "hermite_angle"

    y_mode = str(data.get("y_mode", "poly")).strip().lower()
    if y_mode not in ("poly", "linear"):
        y_mode = "poly"

    hermite = data.get("hermite", {}) if isinstance(data.get("hermite", {}), dict) else {}
    base_tangent_scale = _safe_float(hermite.get("base_tangent_scale", 0.6), 0.6)
    tip_tangent_scale = _safe_float(hermite.get("tip_tangent_scale", 0.9), 0.9)

    return RobotLinkConfig(
        diameter_mm=float(diameter),
        n_links=int(n_links),
        t_knots=t_knots,
        show_default=show_default,
        anchor_mode=anchor_mode,
        shape_mode=shape_mode,
        y_mode=y_mode,
        base_tangent_scale=float(base_tangent_scale),
        tip_tangent_scale=float(tip_tangent_scale),
    )


def load_robot_skeleton_reference(robot_json_path: str) -> RobotSkeletonReference:
    p = Path(robot_json_path)
    if not p.exists():
        raise FileNotFoundError(f"Robot skeleton JSON not found: {robot_json_path}")

    with p.open("r") as f:
        data = json.load(f)

    ref_pose = data.get("reference_pose", {})
    pts_raw = ref_pose.get("points_xyz_mm")
    if pts_raw is None:
        raise ValueError(f"Robot skeleton JSON is missing reference_pose.points_xyz_mm: {robot_json_path}")

    pts = np.asarray(pts_raw, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3 or pts.shape[0] < 2:
        raise ValueError(
            f"Robot skeleton JSON must provide an Nx3 reference_pose.points_xyz_mm array with at least 2 points: {robot_json_path}"
        )

    diameter = _safe_float(data.get("diameter_mm", DEFAULT_ROBOT_DIAMETER_MM), DEFAULT_ROBOT_DIAMETER_MM)
    phase_models_raw = data.get("phase_point_models")
    phase_models = None
    default_phase = None
    if isinstance(phase_models_raw, dict) and len(phase_models_raw) > 0:
        phase_models = {_normalize_phase_key(k): v for k, v in phase_models_raw.items() if isinstance(v, dict)}
        default_phase = _normalize_phase_key(data.get("default_phase_for_legacy_access"))
    return RobotSkeletonReference(
        points_xyz_mm=pts,
        diameter_mm=float(diameter),
        source_path=p.resolve(),
        phase_point_models=phase_models,
        default_phase=default_phase,
    )


def _load_json_file(path: str) -> dict:
    p = Path(path)
    with p.open("r") as f:
        return json.load(f)


def _evaluate_descriptor(model_descriptor: Optional[dict], b_val: float) -> float:
    if model_descriptor is None:
        return 0.0
    out = _eval_model(model_descriptor, np.asarray([float(b_val)], dtype=float))
    return float(np.asarray(out, dtype=float).ravel()[0])


def compute_tracked_skeleton_world(
    skeleton_ref: RobotSkeletonReference,
    b_cmd: float,
    c_cmd_deg: float,
    tip_world: Tuple[float, float, float],
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
        point_models = payload.get("point_models", [])
        for point_idx, point_model in enumerate(point_models[: pts_local.shape[0]]):
            disp_models = point_model.get("displacement_models", {})
            pts_local[point_idx, 0] += _evaluate_descriptor(disp_models.get("x_mm"), b_cmd)
            pts_local[point_idx, 1] += _evaluate_descriptor(disp_models.get("y_mm"), b_cmd)
            pts_local[point_idx, 2] += _evaluate_descriptor(disp_models.get("z_mm"), b_cmd)

    if pts_local.shape[0] == 0:
        return pts_local

    tip_local = pts_local[-1, :].copy()
    offsets = pts_local - tip_local[None, :]
    c_rad = np.deg2rad(float(c_cmd_deg))
    cosc = float(np.cos(c_rad))
    sinc = float(np.sin(c_rad))
    x_rot = offsets[:, 0] * cosc - offsets[:, 1] * sinc
    y_rot = offsets[:, 0] * sinc + offsets[:, 1] * cosc
    pts_world = np.column_stack([x_rot, y_rot, offsets[:, 2]]).astype(float)
    pts_world += np.asarray(tip_world, dtype=float).reshape(1, 3)
    pts_world[-1, :] = np.asarray(tip_world, dtype=float).reshape(3)
    return pts_world


def resolve_robot_config_from_calibration(calibration_path: str, cal_data: dict) -> Optional[Path]:
    exported = cal_data.get("exported_models", {})
    sk = exported.get("robot_skeleton_parametric", None)
    if not isinstance(sk, dict):
        return None
    rel = sk.get("parametric_json", None)
    if not rel:
        return None
    base = Path(calibration_path).resolve().parent
    cand = (base / rel).resolve()
    return cand if cand.exists() else None


# =========================
# G-code parsing
# =========================
@dataclass
class MotionState:
    idx: int
    gcode_line_no: int
    gcode_raw: str
    motion_code: str
    x_stage: float
    y_stage: float
    z_stage: float
    b_cmd: float
    c_cmd: float
    u_cmd: float
    pressure_active: bool
    equation_phase: str
    feed: Optional[float]


_AXVAL_RE = re.compile(r"([A-Za-z])\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))")


def strip_comment(line: str) -> str:
    no_semicolon = line.split(";", 1)[0]
    no_paren = re.sub(r"\([^)]*\)", "", no_semicolon)
    return no_paren.strip()


def parse_gcode_metadata(gcode_path: str) -> Dict[str, str]:
    metadata: Dict[str, str] = {}
    p = Path(gcode_path)
    if not p.exists():
        return metadata
    with p.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            if not raw.lstrip().startswith(";"):
                continue
            match = re.match(r"\s*;\s*([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$", raw)
            if match:
                metadata[match.group(1).strip().lower()] = match.group(2).strip()
    return metadata


def parse_gcode_motion_states(
    gcode_path: str,
    x_axis: str,
    y_axis: str,
    z_axis: str,
    b_axis: str,
    c_axis: str,
    u_axis: str = "U",
    default_c0: float = DEFAULT_C0_DEG,
) -> List[MotionState]:
    p = Path(gcode_path)
    if not p.exists():
        raise FileNotFoundError(f"G-code file not found: {gcode_path}")

    abs_mode = True
    current_motion = "G1"

    pos: Dict[str, float] = {
        x_axis.upper(): 0.0,
        y_axis.upper(): 0.0,
        z_axis.upper(): 0.0,
        b_axis.upper(): 0.0,
        c_axis.upper(): default_c0,
        u_axis.upper(): 0.0,
    }
    current_feed: Optional[float] = None
    current_phase = PULL_PHASE
    pressure_active = False
    write_marker_active = False
    states: List[MotionState] = []

    with p.open("r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    has_write_markers = any(
        WRITE_START_RE.search(raw)
        or WRITE_END_RE.search(raw)
        or WAVE_WRITE_START_RE.search(raw)
        or WAVE_WRITE_END_RE.search(raw)
        for raw in lines
    )
    has_pressure_solenoid = any(PRESSURE_SOLENOID_RE.search(raw) for raw in lines)
    metadata = parse_gcode_metadata(str(p))
    current_phase = _normalize_phase_name(metadata.get("active_phase", current_phase))

    for line_no, raw in enumerate(lines, start=1):
        switch_match = PHASE_SWITCH_RE.search(raw)
        if switch_match:
            phase_from_marker = _phase_from_switch_token(switch_match.group(1))
            if phase_from_marker is not None:
                current_phase = phase_from_marker

        solenoid_match = PRESSURE_SOLENOID_RE.search(raw)
        if solenoid_match:
            pressure_active = solenoid_match.group(1) == "1"

        if has_write_markers:
            if WRITE_START_RE.search(raw) or WAVE_WRITE_START_RE.search(raw):
                write_marker_active = True
            if WRITE_END_RE.search(raw) or WAVE_WRITE_END_RE.search(raw):
                write_marker_active = False

        line = strip_comment(raw)
        if not line:
            continue

        uline = line.upper()
        if "G90" in uline:
            abs_mode = True
        if "G91" in uline:
            abs_mode = False

        if re.search(r"(?<!\d)G0(?!\d)", uline):
            current_motion = "G0"
        if re.search(r"(?<!\d)G1(?!\d)", uline):
            current_motion = "G1"

        words = {m.group(1).upper(): float(m.group(2)) for m in _AXVAL_RE.finditer(line)}

        if "F" in words:
            current_feed = float(words["F"])

        pose_axes = {x_axis.upper(), y_axis.upper(), z_axis.upper(), b_axis.upper(), c_axis.upper()}
        tracked_axes = pose_axes | {u_axis.upper()}
        if re.search(r"(?<!\d)G92(?!\d)", uline):
            for ax in tracked_axes:
                if ax in words:
                    pos[ax] = float(words[ax])
            continue

        is_motion_line = current_motion in ("G0", "G1")
        if not is_motion_line:
            continue

        u_axis_u = u_axis.upper()
        if u_axis_u in words:
            prev_u = pos[u_axis_u]
            if abs_mode:
                pos[u_axis_u] = float(words[u_axis_u])
            else:
                pos[u_axis_u] += float(words[u_axis_u])
            u_delta = pos[u_axis_u] - prev_u
            if u_delta >= DEFAULT_PRESSURE_TOGGLE_U_MM:
                pressure_active = True
            elif u_delta <= -DEFAULT_PRESSURE_TOGGLE_U_MM:
                pressure_active = False

        has_pose_move = any(ax in words for ax in pose_axes)
        if not has_pose_move:
            continue

        for ax in pose_axes:
            if ax in words:
                if abs_mode:
                    pos[ax] = float(words[ax])
                else:
                    pos[ax] += float(words[ax])

        states.append(MotionState(
            idx=len(states),
            gcode_line_no=line_no,
            gcode_raw=raw.rstrip("\n"),
            motion_code=current_motion,
            x_stage=pos[x_axis.upper()],
            y_stage=pos[y_axis.upper()],
            z_stage=pos[z_axis.upper()],
            b_cmd=pos[b_axis.upper()],
            c_cmd=pos[c_axis.upper()],
            u_cmd=pos[u_axis.upper()],
            pressure_active=(
                pressure_active
                if has_pressure_solenoid
                else (write_marker_active if has_write_markers else pressure_active)
            ),
            equation_phase=current_phase,
            feed=current_feed,
        ))

    if not states:
        raise RuntimeError("No G0/G1 motion states with tracked axes were found in the G-code file.")

    return states


# =========================
# Tip reconstruction
# =========================
@dataclass
class TipTrajectory:
    x_tip: np.ndarray
    y_tip: np.ndarray
    z_tip: np.ndarray
    x_stage: np.ndarray
    y_stage: np.ndarray
    z_stage: np.ndarray
    b_cmd: np.ndarray
    c_cmd: np.ndarray
    r_of_b: np.ndarray
    z_of_b: np.ndarray
    y_off_of_b: np.ndarray
    tip_angle_deg: np.ndarray
    sgn: np.ndarray
    equation_phase: np.ndarray


def evaluate_calibration_phase_models(
    cal: Calibration,
    b_cmd: np.ndarray,
    equation_phase: np.ndarray,
    offplane_sign: float = DEFAULT_OFFPLANE_SIGN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r_of_b = np.empty_like(b_cmd, dtype=float)
    z_of_b = np.empty_like(b_cmd, dtype=float)
    tip_ang = np.empty_like(b_cmd, dtype=float)
    y_off_of_b = np.empty_like(b_cmd, dtype=float)

    for phase in sorted(set(str(p) for p in equation_phase)):
        mask = (equation_phase == phase)
        if not np.any(mask):
            continue
        models = cal.phase_models.get(phase) or cal.phase_models.get(_phase_alias(phase))
        if not models:
            raise ValueError(f"Calibration is missing phase models for '{phase}'")

        r_model = _select_named_model(models, "r", "pchip")
        z_model = _select_named_model(models, "z", "pchip")
        tip_model = _select_named_model(models, "tip_angle", "pchip")
        y_selector = cal.requested_offplane_fit_model or "pchip"
        y_model = _select_named_model(models, "offplane_y", y_selector)
        y_extrap_model = _normalize_model_spec(models.get("offplane_y_linear"))
        if y_extrap_model is None:
            y_extrap_model = _normalize_model_spec(models.get("offplane_y"))

        if r_model is None or z_model is None or tip_model is None:
            raise ValueError(f"Calibration phase '{phase}' is missing required PCHIP models")

        r_of_b[mask] = _eval_model(r_model, b_cmd[mask])
        z_of_b[mask] = _eval_model(z_model, b_cmd[mask])
        tip_ang[mask] = _eval_model(tip_model, b_cmd[mask])

        if y_model is None:
            y_off_of_b[mask] = 0.0
        elif str(y_model.get("model_type", "")).strip().lower() == "pchip":
            y_off_of_b[mask] = offplane_sign * _eval_pchip_with_linear_extrap(y_model, y_extrap_model, b_cmd[mask])
        else:
            y_off_of_b[mask] = offplane_sign * _eval_model(y_model, b_cmd[mask])

    return r_of_b, z_of_b, tip_ang, y_off_of_b


def _angle_diff_deg(a: np.ndarray, b: float) -> np.ndarray:
    return ((a - b + 180.0) % 360.0) - 180.0


def infer_sgn_from_c(c_vals: np.ndarray, c0_deg: float, c180_deg: float) -> np.ndarray:
    d0 = np.abs(_angle_diff_deg(c_vals, c0_deg))
    d180 = np.abs(_angle_diff_deg(c_vals, c180_deg))
    return np.where(d0 <= d180, 1.0, -1.0)


def reconstruct_tip_trajectory(
    states: List[MotionState],
    cal: Calibration,
    c0_deg: float = DEFAULT_C0_DEG,
    offplane_sign: float = DEFAULT_OFFPLANE_SIGN,
    write_mode: str = "calibrated",
) -> TipTrajectory:
    x_stage = np.array([s.x_stage for s in states], dtype=float)
    y_stage = np.array([s.y_stage for s in states], dtype=float)
    z_stage = np.array([s.z_stage for s in states], dtype=float)
    b_cmd = np.array([s.b_cmd for s in states], dtype=float)
    c_cmd = np.array([s.c_cmd for s in states], dtype=float)
    equation_phase = np.array([_normalize_phase_name(s.equation_phase) for s in states], dtype=object)

    if str(write_mode).strip().lower() == "cartesian":
        zeros = np.zeros_like(b_cmd, dtype=float)
        sgn = infer_sgn_from_c(c_cmd, c0_deg=c0_deg, c180_deg=cal.c_180_deg)
        return TipTrajectory(
            x_tip=x_stage.copy(), y_tip=y_stage.copy(), z_tip=z_stage.copy(),
            x_stage=x_stage, y_stage=y_stage, z_stage=z_stage,
            b_cmd=b_cmd, c_cmd=c_cmd,
            r_of_b=zeros.copy(), z_of_b=zeros.copy(), y_off_of_b=zeros.copy(),
            tip_angle_deg=zeros.copy(), sgn=sgn, equation_phase=equation_phase,
        )

    if cal.phase_models:
        r_of_b, z_of_b, tip_ang, y_off_of_b = evaluate_calibration_phase_models(
            cal, b_cmd, equation_phase, offplane_sign=offplane_sign
        )
    else:
        r_of_b = _polyval4(cal.pr, b_cmd)
        z_of_b = _polyval4(cal.pz, b_cmd)
        tip_ang = _polyval4(cal.pa, b_cmd)

        if cal.py_off is None:
            y_off_of_b = np.zeros_like(b_cmd, dtype=float)
        else:
            y_off_of_b = offplane_sign * np.polyval(cal.py_off, b_cmd)

    c_rad = np.deg2rad(c_cmd)
    x_tip = x_stage + r_of_b * np.cos(c_rad) - y_off_of_b * np.sin(c_rad)
    y_tip = y_stage + r_of_b * np.sin(c_rad) + y_off_of_b * np.cos(c_rad)
    z_tip = z_stage + z_of_b

    sgn = infer_sgn_from_c(c_cmd, c0_deg=c0_deg, c180_deg=cal.c_180_deg)

    return TipTrajectory(
        x_tip=x_tip, y_tip=y_tip, z_tip=z_tip,
        x_stage=x_stage, y_stage=y_stage, z_stage=z_stage,
        b_cmd=b_cmd, c_cmd=c_cmd,
        r_of_b=r_of_b, z_of_b=z_of_b, y_off_of_b=y_off_of_b,
        tip_angle_deg=tip_ang, equation_phase=equation_phase,
        sgn=sgn,
    )


# =========================
# Robot link reconstruction
# =========================
def _hermite_curve_2d(p0: np.ndarray, p1: np.ndarray, m0: np.ndarray, m1: np.ndarray, s: np.ndarray) -> np.ndarray:
    s = np.asarray(s, dtype=float)
    h00 = (2*s**3 - 3*s**2 + 1)
    h10 = (s**3 - 2*s**2 + s)
    h01 = (-2*s**3 + 3*s**2)
    h11 = (s**3 - s**2)
    return (h00[:, None]*p0[None, :] +
            h10[:, None]*m0[None, :] +
            h01[:, None]*p1[None, :] +
            h11[:, None]*m1[None, :])


def compute_robot_polyline_world(
    cal: Calibration,
    robot_cfg: RobotLinkConfig,
    x_stage: float,
    y_stage: float,
    z_stage: float,
    b_cmd: float,
    c_cmd_deg: float,
    tip_world: Tuple[float, float, float],
    tip_angle_deg: float,
    r_tip: float,
    z_tip: float,
    y_off_tip: float,
    equation_phase: str = PULL_PHASE,
    offplane_sign: float = DEFAULT_OFFPLANE_SIGN,
    start_world: Optional[Tuple[float, float, float]] = None,
) -> np.ndarray:
    t = robot_cfg.t_knots
    b0 = float(cal.b_home)
    b = float(b_cmd)
    phase_name = _normalize_phase_name(equation_phase)

    if cal.phase_models:
        r0_arr, z0_arr, _, y0_arr = evaluate_calibration_phase_models(
            cal,
            np.array([b0], dtype=float),
            np.array([phase_name], dtype=object),
            offplane_sign=offplane_sign,
        )
        r0 = float(r0_arr[0])
        z0 = float(z0_arr[0])
        y0 = float(y0_arr[0])
    else:
        r0 = float(_polyval4(cal.pr, b0))
        z0 = float(_polyval4(cal.pz, b0))
        if cal.py_off is None:
            y0 = 0.0
        else:
            y0 = float(offplane_sign * np.polyval(cal.py_off, b0))

    r_end = float(r_tip - r0)
    z_end = float(z_tip - z0)
    y_end = float(y_off_tip - y0)

    if robot_cfg.shape_mode == "poly_b":
        b_k = b0 + t * (b - b0)
        if cal.phase_models:
            phase_arr = np.full(b_k.shape, phase_name, dtype=object)
            r_abs, z_abs, _, y_abs = evaluate_calibration_phase_models(
                cal, b_k, phase_arr, offplane_sign=offplane_sign
            )
            r_k = r_abs - r0
            z_k = z_abs - z0
            y_k = y_abs - y0
        else:
            r_k = _polyval4(cal.pr, b_k) - r0
            z_k = _polyval4(cal.pz, b_k) - z0
            if cal.py_off is None:
                y_k = np.zeros_like(r_k)
            else:
                y_k = offplane_sign * np.polyval(cal.py_off, b_k) - y0
    else:
        p0 = np.array([0.0, 0.0], dtype=float)
        p1 = np.array([r_end, z_end], dtype=float)

        L = float(np.linalg.norm(p1 - p0))
        L = max(L, 1e-6)

        m0 = robot_cfg.base_tangent_scale * L * np.array([0.0, 1.0], dtype=float)

        th = np.deg2rad(float(tip_angle_deg))
        tip_dir = np.array([np.sin(th), np.cos(th)], dtype=float)
        if not np.all(np.isfinite(tip_dir)) or float(np.linalg.norm(tip_dir)) < 1e-9:
            tip_dir = (p1 - p0) / L
        tip_dir = tip_dir / max(float(np.linalg.norm(tip_dir)), 1e-9)
        m1 = robot_cfg.tip_tangent_scale * L * tip_dir

        rz = _hermite_curve_2d(p0, p1, m0, m1, t)
        r_k = rz[:, 0]
        z_k = rz[:, 1]

        if cal.py_off is None:
            y_k = np.zeros_like(r_k)
        else:
            if robot_cfg.y_mode == "linear":
                y_k = t * y_end
            else:
                b_k = b0 + t * (b - b0)
                if cal.phase_models:
                    phase_arr = np.full(b_k.shape, phase_name, dtype=object)
                    _, _, _, y_abs = evaluate_calibration_phase_models(
                        cal, b_k, phase_arr, offplane_sign=offplane_sign
                    )
                    y_k = y_abs - y0
                else:
                    y_k = offplane_sign * np.polyval(cal.py_off, b_k) - y0

    c_rad = np.deg2rad(float(c_cmd_deg))
    cosc = float(np.cos(c_rad))
    sinc = float(np.sin(c_rad))

    x_rot = r_k * cosc - y_k * sinc
    y_rot = r_k * sinc + y_k * cosc

    pts_world = np.column_stack([
        float(x_stage) + x_rot,
        float(y_stage) + y_rot,
        float(z_stage) + z_k
    ]).astype(float)

    pts_world[0, :] = np.array([x_stage, y_stage, z_stage], dtype=float)
    tip_world_vec = np.array(tip_world, dtype=float).reshape(3)

    if robot_cfg.anchor_mode == "tip":
        delta = tip_world_vec - pts_world[-1, :]
        pts_world = pts_world + delta[None, :]
        # Keep the final point exactly on the reconstructed tip to avoid any
        # visible mismatch from floating-point drift during redraws.
        pts_world[-1, :] = tip_world_vec

    if start_world is not None and pts_world.shape[0] >= 2:
        start_world_vec = np.asarray(start_world, dtype=float).reshape(3)
        delta0 = start_world_vec - pts_world[0, :]
        # Pin the robot base to the first parsed motion position while
        # preserving the already-computed end pose.
        blend = np.linspace(1.0, 0.0, pts_world.shape[0], dtype=float)[:, None]
        pts_world = pts_world + blend * delta0[None, :]
        pts_world[0, :] = start_world_vec
        if robot_cfg.anchor_mode == "tip":
            pts_world[-1, :] = tip_world_vec

    return pts_world


# =========================
# Plot helpers
# =========================
def make_line_segments(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    pts = np.stack([xs, ys, zs], axis=1)
    if len(pts) < 2:
        return np.empty((0, 2, 3), dtype=float)
    return np.stack([pts[:-1], pts[1:]], axis=1)


def smooth_polyline_points(points_xyz: np.ndarray, samples_per_segment: int = ROBOT_TUBE_SAMPLES_PER_SEGMENT) -> np.ndarray:
    pts = np.asarray(points_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 3:
        return pts

    step = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    keep = np.concatenate([[True], step > 1e-9])
    pts = pts[keep]
    if pts.shape[0] < 2:
        return np.asarray(points_xyz, dtype=float)

    arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))])
    total_length = float(arc[-1])
    if total_length <= 1e-9:
        return pts

    sample_count = max(
        pts.shape[0],
        int((pts.shape[0] - 1) * max(1, int(samples_per_segment))) + 1,
    )
    arc_q = np.linspace(0.0, total_length, sample_count)
    smooth = np.column_stack([
        _eval_pchip(arc, pts[:, dim], arc_q)
        for dim in range(3)
    ]).astype(float)
    smooth[0, :] = pts[0, :]
    smooth[-1, :] = pts[-1, :]
    return smooth


def _safe_unit_vector(vec: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12 or not np.isfinite(norm):
        return np.asarray(fallback, dtype=float)
    return arr / norm


def _make_tube_faces(points_xyz: np.ndarray, radius_mm: float, color: str, alpha: float):
    pts = np.asarray(points_xyz, dtype=float)
    if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] != 3 or radius_mm <= 0.0:
        return [], []

    tangents = np.gradient(pts, axis=0)
    tangents = np.asarray([
        _safe_unit_vector(t, np.array([0.0, 0.0, 1.0], dtype=float))
        for t in tangents
    ], dtype=float)

    normals = np.zeros_like(tangents)
    binormals = np.zeros_like(tangents)

    def _fallback_normal(tangent: np.ndarray) -> np.ndarray:
        ref = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(tangent, ref))) > 0.92:
            ref = np.array([0.0, 1.0, 0.0], dtype=float)
        normal = np.cross(tangent, ref)
        if float(np.linalg.norm(normal)) <= 1e-12:
            normal = np.cross(tangent, np.array([1.0, 0.0, 0.0], dtype=float))
        return _safe_unit_vector(normal, np.array([1.0, 0.0, 0.0], dtype=float))

    normals[0, :] = _fallback_normal(tangents[0, :])
    binormals[0, :] = _safe_unit_vector(
        np.cross(tangents[0, :], normals[0, :]),
        np.array([0.0, 1.0, 0.0], dtype=float),
    )

    for i in range(1, pts.shape[0]):
        tangent = tangents[i, :]
        normal = normals[i - 1, :] - np.dot(normals[i - 1, :], tangent) * tangent
        if float(np.linalg.norm(normal)) <= 1e-12:
            normal = _fallback_normal(tangent)
        normal = _safe_unit_vector(normal, normals[i - 1, :])
        binormal = _safe_unit_vector(np.cross(tangent, normal), binormals[i - 1, :])
        normal = _safe_unit_vector(np.cross(binormal, tangent), normal)
        normals[i, :] = normal
        binormals[i, :] = binormal

    theta = np.linspace(0.0, 2.0 * np.pi, ROBOT_TUBE_RADIAL_SEGMENTS, endpoint=False)
    cos_t = np.cos(theta)[None, :, None]
    sin_t = np.sin(theta)[None, :, None]
    rings = pts[:, None, :] + radius_mm * (cos_t * normals[:, None, :] + sin_t * binormals[:, None, :])

    base_rgb = np.asarray(to_rgb(color), dtype=float)
    light_dir = _safe_unit_vector(np.array([0.35, -0.25, 0.90], dtype=float), np.array([0.0, 0.0, 1.0], dtype=float))
    faces = []
    facecolors = []

    for i in range(pts.shape[0] - 1):
        ring0 = rings[i]
        ring1 = rings[i + 1]
        for j in range(ROBOT_TUBE_RADIAL_SEGMENTS):
            jn = (j + 1) % ROBOT_TUBE_RADIAL_SEGMENTS
            tri_a = np.array([ring0[j], ring0[jn], ring1[jn]], dtype=float)
            tri_b = np.array([ring0[j], ring1[jn], ring1[j]], dtype=float)
            for tri in (tri_a, tri_b):
                tri_normal = np.cross(tri[1] - tri[0], tri[2] - tri[0])
                tri_normal = _safe_unit_vector(tri_normal, np.array([0.0, 0.0, 1.0], dtype=float))
                brightness = 0.48 + 0.52 * max(0.0, float(np.dot(tri_normal, light_dir)))
                face_rgb = np.clip(base_rgb * brightness + 0.12, 0.0, 1.0)
                faces.append(tri)
                facecolors.append((face_rgb[0], face_rgb[1], face_rgb[2], alpha))

    start_center = pts[0, :]
    end_center = pts[-1, :]
    start_rgb = tuple(np.clip(base_rgb * 0.75 + 0.08, 0.0, 1.0)) + (alpha,)
    end_rgb = tuple(np.clip(base_rgb * 0.92 + 0.10, 0.0, 1.0)) + (alpha,)
    for j in range(ROBOT_TUBE_RADIAL_SEGMENTS):
        jn = (j + 1) % ROBOT_TUBE_RADIAL_SEGMENTS
        faces.append(np.array([start_center, rings[0, jn], rings[0, j]], dtype=float))
        facecolors.append(start_rgb)
        faces.append(np.array([end_center, rings[-1, j], rings[-1, jn]], dtype=float))
        facecolors.append(end_rgb)

    return faces, facecolors


def create_tube_artist(
    ax,
    points_xyz: np.ndarray,
    diameter_mm: float,
    color: str,
    alpha: float = 0.96,
):
    smooth_points = smooth_polyline_points(points_xyz)
    faces, facecolors = _make_tube_faces(smooth_points, radius_mm=0.5 * float(diameter_mm), color=color, alpha=alpha)
    artist = Poly3DCollection(faces, facecolors=facecolors, edgecolors="none")
    try:
        artist.set_zsort("average")
    except Exception:
        pass
    artist.set_antialiased(True)
    ax.add_collection3d(artist)
    return artist, smooth_points


def update_tube_artist(
    artist: Optional[Poly3DCollection],
    points_xyz: np.ndarray,
    diameter_mm: float,
    color: str,
    alpha: float = 0.96,
) -> np.ndarray:
    if artist is None:
        return np.asarray(points_xyz, dtype=float)
    smooth_points = smooth_polyline_points(points_xyz)
    faces, facecolors = _make_tube_faces(smooth_points, radius_mm=0.5 * float(diameter_mm), color=color, alpha=alpha)
    artist.set_verts(faces)
    artist.set_facecolor(facecolors)
    return smooth_points


def _compute_collision_highlight_segments(
    body_points_xyz: np.ndarray,
    candidate_segments_xyz: np.ndarray,
    collision_radius_mm: float,
    tip_exclusion_mm: float = DEFAULT_COLLISION_TIP_EXCLUSION_MM,
) -> Tuple[np.ndarray, np.ndarray, bool, float]:
    body_pts = np.asarray(body_points_xyz, dtype=float)
    candidate_segs = np.asarray(candidate_segments_xyz, dtype=float)
    if body_pts.ndim != 2 or body_pts.shape[0] < 2 or body_pts.shape[1] != 3:
        return np.empty((0, 2, 3), dtype=float), np.zeros((0,), dtype=bool), False, float("inf")
    if candidate_segs.ndim != 3 or candidate_segs.shape[0] == 0 or candidate_segs.shape[1:] != (2, 3):
        return np.empty((0, 2, 3), dtype=float), np.zeros((0,), dtype=bool), False, float("inf")

    body_arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(body_pts, axis=0), axis=1))])
    body_dist_to_tip = body_arc[-1] - body_arc
    body_mask = body_dist_to_tip >= max(0.0, float(tip_exclusion_mm))
    if not np.any(body_mask):
        body_mask[:-1] = True
    candidate_body_pts = body_pts[body_mask]
    if candidate_body_pts.shape[0] == 0:
        return np.empty((0, 2, 3), dtype=float), np.zeros(candidate_segs.shape[0], dtype=bool), False, float("inf")

    min_dist = float("inf")
    seg_start = candidate_segs[:, 0, :]
    seg_end = candidate_segs[:, 1, :]
    seg_vec = seg_end - seg_start
    seg_len_sq = np.sum(seg_vec * seg_vec, axis=1)
    valid_seg_mask = seg_len_sq > 1e-12
    if not np.any(valid_seg_mask):
        return np.empty((0, 2, 3), dtype=float), np.zeros(candidate_segs.shape[0], dtype=bool), False, float("inf")

    seg_start_valid = seg_start[valid_seg_mask]
    seg_vec_valid = seg_vec[valid_seg_mask]
    seg_len_sq_valid = seg_len_sq[valid_seg_mask]
    candidate_segs_valid = candidate_segs[valid_seg_mask]
    hit_mask_any = np.zeros(candidate_segs_valid.shape[0], dtype=bool)

    for body_pt in candidate_body_pts:
        rel = body_pt[None, :] - seg_start_valid
        t = np.sum(rel * seg_vec_valid, axis=1) / seg_len_sq_valid
        t = np.clip(t, 0.0, 1.0)
        proj = seg_start_valid + t[:, None] * seg_vec_valid
        dists = np.linalg.norm(proj - body_pt[None, :], axis=1)
        if dists.size:
            min_dist = min(min_dist, float(np.min(dists)))
        hit_mask_any |= (dists <= collision_radius_mm)

    collision_mask = np.zeros(candidate_segs.shape[0], dtype=bool)
    collision_mask[valid_seg_mask] = hit_mask_any
    if not np.any(collision_mask):
        return np.empty((0, 2, 3), dtype=float), collision_mask, False, min_dist
    return np.asarray(candidate_segs[collision_mask], dtype=float), collision_mask, True, min_dist


def compute_equal_box_center_radius(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, pad_frac: float = 0.04):
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    z_min, z_max = float(np.min(zs)), float(np.max(zs))

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    cz = 0.5 * (z_min + z_max)

    dx = max(x_max - x_min, 1e-9)
    dy = max(y_max - y_min, 1e-9)
    dz = max(z_max - z_min, 1e-9)
    r = 0.5 * max(dx, dy, dz) * (1.0 + pad_frac)
    return (cx, cy, cz), r


def apply_equal_axes_with_zoom(ax, center: Tuple[float, float, float], base_radius: float, zoom: float):
    zoom = max(float(zoom), 1e-6)
    cx, cy, cz = center
    r = base_radius / zoom

    # Preserve camera to avoid planar view "popping" on updates
    elev, azim = getattr(ax, "elev", None), getattr(ax, "azim", None)

    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass

    if elev is not None and azim is not None:
        try:
            ax.view_init(elev=elev, azim=azim)
        except Exception:
            pass


def capture_view_state(ax) -> dict:
    state = {
        "xlim": ax.get_xlim3d(),
        "ylim": ax.get_ylim3d(),
        "zlim": ax.get_zlim3d(),
        "elev": getattr(ax, "elev", None),
        "azim": getattr(ax, "azim", None),
    }
    if hasattr(ax, "roll"):
        state["roll"] = getattr(ax, "roll", None)
    return state


def restore_view_state(ax, state: dict):
    ax.set_xlim(state["xlim"])
    ax.set_ylim(state["ylim"])
    ax.set_zlim(state["zlim"])
    elev = state.get("elev")
    azim = state.get("azim")
    if elev is None or azim is None:
        return
    try:
        roll = state.get("roll")
        if roll is None:
            ax.view_init(elev=elev, azim=azim)
        else:
            ax.view_init(elev=elev, azim=azim, roll=roll)
    except Exception:
        ax.view_init(elev=elev, azim=azim)


def set_major_view(ax, view_name: str):
    name = view_name.upper()
    if name == "XY":
        ax.view_init(elev=90, azim=-90)
    elif name == "XZ":
        ax.view_init(elev=0, azim=-90)
    elif name == "YZ":
        ax.view_init(elev=0, azim=0)
    elif name == "ISO":
        ax.view_init(elev=25, azim=-60)
    else:
        raise ValueError(f"Unknown view '{view_name}'")


def style_dark_3d_axes(fig, ax):
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    try:
        ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
        ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
        ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    except Exception:
        pass

    try:
        ax.grid(True, color=(0.35, 0.35, 0.35, 0.6))
    except Exception:
        ax.grid(True)

    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")
    ax.title.set_color("white")

    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.line.set_color("white")
        except Exception:
            pass
        try:
            axis._axinfo["grid"]["color"] = (0.35, 0.35, 0.35, 0.6)
            axis._axinfo["tick"]["color"] = (1, 1, 1, 1)
            axis._axinfo["axisline"]["color"] = (1, 1, 1, 1)
        except Exception:
            pass


def _decimate_xyz(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, max_points: int):
    n = len(xs)
    if n <= max_points:
        return xs, ys, zs
    idx = np.linspace(0, n - 1, max_points).astype(int)
    idx[0] = 0
    idx[-1] = n - 1
    return xs[idx], ys[idx], zs[idx]


def _decimate_idx(n: int, max_points: int) -> np.ndarray:
    if n <= max_points:
        return np.arange(n, dtype=int)
    idx = np.linspace(0, n - 1, max_points).astype(int)
    idx[0] = 0
    idx[-1] = n - 1
    return idx


def format_duration(seconds: float) -> str:
    if not np.isfinite(seconds) or seconds < 0:
        return "--:--"
    total_seconds = int(round(float(seconds)))
    hours, rem = divmod(total_seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours > 0:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def estimate_print_time_seconds(states: List[MotionState], traj: TipTrajectory) -> Tuple[np.ndarray, int]:
    """Estimate elapsed time from modal G-code F values and reconstructed tip travel.

    Feed rates are treated as mm/min. Segments without a positive feed are skipped
    and counted so the UI can flag that the estimate is partial.
    """
    n = len(states)
    elapsed = np.zeros(n, dtype=float)
    skipped_segments = 0

    if n <= 1:
        return elapsed, skipped_segments

    tip_xyz = np.column_stack([traj.x_tip, traj.y_tip, traj.z_tip])
    dists = np.linalg.norm(np.diff(tip_xyz, axis=0), axis=1)

    for i, dist in enumerate(dists, start=1):
        feed = states[i].feed
        dt = 0.0
        if feed is None or not np.isfinite(feed) or feed <= 0.0:
            if dist > 0.0:
                skipped_segments += 1
        else:
            dt = float(dist) / float(feed) * 60.0
        elapsed[i] = elapsed[i - 1] + dt

    return elapsed, skipped_segments


def print_gantry_ranges_used(traj: TipTrajectory, axis_names: Dict[str, str]) -> None:
    print("Gantry ranges used:")
    print(f"  {axis_names['x']}: {np.min(traj.x_stage):.4f} to {np.max(traj.x_stage):.4f}")
    print(f"  {axis_names['y']}: {np.min(traj.y_stage):.4f} to {np.max(traj.y_stage):.4f}")
    print(f"  {axis_names['z']}: {np.min(traj.z_stage):.4f} to {np.max(traj.z_stage):.4f}")
    print(f"  {axis_names['b']}: {np.min(traj.b_cmd):.4f} to {np.max(traj.b_cmd):.4f}")
    print(f"  {axis_names['c']}: {np.min(traj.c_cmd):.4f} to {np.max(traj.c_cmd):.4f}")


# =========================
# Interactive UI
# =========================
def launch_interactive_plot(
    states: List[MotionState],
    traj: TipTrajectory,
    cal: Calibration,
    robot_cfg: Optional[RobotLinkConfig] = None,
    robot_skeleton_ref: Optional[RobotSkeletonReference] = None,
    c0_deg: float = DEFAULT_C0_DEG,
    offplane_sign: float = DEFAULT_OFFPLANE_SIGN,
    show_stage: bool = False,
):
    # Matplotlib performance knobs (safe defaults)
    try:
        plt.rcParams["path.simplify"] = True
        plt.rcParams["path.simplify_threshold"] = 0.8
        plt.rcParams["agg.path.chunksize"] = 20000
    except Exception:
        pass
    remove_mpl_keymap_entries(("left", "right", "home", "h", "j", "k", "l"))

    n = len(states)
    idx0 = 0
    robot_start_world = (
        float(traj.x_stage[0]),
        float(traj.y_stage[0]),
        float(traj.z_stage[0]),
    )
    elapsed_seconds, skipped_time_segments = estimate_print_time_seconds(states, traj)
    total_estimated_seconds = float(elapsed_seconds[-1]) if elapsed_seconds.size else 0.0
    gantry_ranges = {
        "x": (float(np.min(traj.x_stage)), float(np.max(traj.x_stage))),
        "y": (float(np.min(traj.y_stage)), float(np.max(traj.y_stage))),
        "z": (float(np.min(traj.z_stage)), float(np.max(traj.z_stage))),
    }

    fig = plt.figure(figsize=DEFAULT_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    # Reserve less space for controls so the 3D axes stay larger.
    plt.subplots_adjust(left=0.04, right=0.99, bottom=0.245, top=0.975)
    style_dark_3d_axes(fig, ax)

    xs, ys, zs = traj.x_tip, traj.y_tip, traj.z_tip
    segs = make_line_segments(xs, ys, zs)
    phase_colors = {
        PULL_PHASE: "#66d9ff",
        RELEASE_PHASE: "tomato",
    }
    unprinted_line_width = 0.8
    unprinted_line_alpha = 0.24
    printed_line_color = "#66d9ff"
    printed_line_width = 2.2
    printed_line_alpha = 0.95
    point_phase = np.asarray(traj.equation_phase, dtype=object)

    pressure_active = np.array([s.pressure_active for s in states], dtype=bool)
    motion_codes = np.array([s.motion_code for s in states], dtype=object)
    extrude_seg_mask = pressure_active[1:] & (motion_codes[1:] == "G1") if n > 1 else np.empty((0,), dtype=bool)

    # Precompute extruding segments for fast redraw (O(log N) selection)
    extrude_segs = segs[extrude_seg_mask] if len(segs) else np.empty((0, 2, 3), dtype=float)
    extrude_end_idx = (np.nonzero(extrude_seg_mask)[0] + 1).astype(int)  # segment ends at index i

    # Full reference tip path, tinted by active equation set.
    bg_idx = _decimate_idx(len(xs), MAX_BACKGROUND_POINTS)
    bg_segs = make_line_segments(xs[bg_idx], ys[bg_idx], zs[bg_idx])
    bg_seg_phase = point_phase[bg_idx[1:]] if bg_idx.size > 1 else np.empty((0,), dtype=object)
    if len(bg_segs) > 0:
        bg_colors = [phase_colors.get(str(phase), "deepskyblue") for phase in bg_seg_phase]
        lc = Line3DCollection(
            bg_segs,
            colors=bg_colors,
            linewidths=unprinted_line_width,
            alpha=unprinted_line_alpha,
        )
        ax.add_collection3d(lc)

    # Start/end markers
    ax.scatter([xs[0]], [ys[0]], [zs[0]], marker="o", s=46, color="lime", label="Start")
    ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], marker="x", s=46, color="red", label="End")

    # Current markers
    current_tip_marker, = ax.plot([xs[idx0]], [ys[idx0]], [zs[idx0]],
                                  marker="o", markersize=8, linestyle="None", color="white")
    current_stage_marker, = ax.plot([traj.x_stage[idx0]], [traj.y_stage[idx0]], [traj.z_stage[idx0]],
                                    marker="^", markersize=6, linestyle="None", alpha=0.95, color="magenta")

    # Current extruding portion
    p0 = np.array([[xs[idx0], ys[idx0], zs[idx0]], [xs[idx0], ys[idx0], zs[idx0]]], dtype=float)
    current_print_lc = Line3DCollection(
        [p0],
        colors=[printed_line_color],
        linewidths=printed_line_width,
        alpha=printed_line_alpha,
    )
    ax.add_collection3d(current_print_lc)
    current_collision_seed = np.array(
        [[[xs[idx0], ys[idx0], zs[idx0]], [xs[idx0], ys[idx0], zs[idx0]]]],
        dtype=float,
    )
    current_collision_lc = Line3DCollection(
        current_collision_seed,
        colors=["red"],
        linewidths=printed_line_width + 1.0,
        alpha=1.0,
    )
    ax.add_collection3d(current_collision_lc)

    # Robot overlays (optional)
    robot_lc = None
    robot_visible = {"value": False}
    active_body_radius_mm = 0.0
    if robot_cfg is not None:
        rp = compute_robot_polyline_world(
            cal, robot_cfg,
            x_stage=traj.x_stage[idx0],
            y_stage=traj.y_stage[idx0],
            z_stage=traj.z_stage[idx0],
            b_cmd=traj.b_cmd[idx0],
            c_cmd_deg=traj.c_cmd[idx0],
            tip_world=(traj.x_tip[idx0], traj.y_tip[idx0], traj.z_tip[idx0]),
            tip_angle_deg=traj.tip_angle_deg[idx0],
            r_tip=traj.r_of_b[idx0],
            z_tip=traj.z_of_b[idx0],
            y_off_tip=traj.y_off_of_b[idx0],
            equation_phase=str(traj.equation_phase[idx0]),
            offplane_sign=offplane_sign,
            start_world=robot_start_world,
        )
        robot_lc, _ = create_tube_artist(
            ax,
            rp,
            diameter_mm=robot_cfg.diameter_mm,
            color=ROBOT_TUBE_COLOR,
            alpha=0.94,
        )
        robot_visible["value"] = bool(robot_cfg.show_default)
        robot_lc.set_visible(robot_visible["value"])
        active_body_radius_mm = 0.5 * float(robot_cfg.diameter_mm)

    skeleton_ref_lc = None
    if robot_skeleton_ref is not None:
        pts_ref = compute_tracked_skeleton_world(
            robot_skeleton_ref,
            b_cmd=float(traj.b_cmd[idx0]),
            c_cmd_deg=float(traj.c_cmd[idx0]),
            tip_world=(float(traj.x_tip[idx0]), float(traj.y_tip[idx0]), float(traj.z_tip[idx0])),
            equation_phase=str(traj.equation_phase[idx0]),
        )
        skeleton_ref_lc, _ = create_tube_artist(
            ax,
            pts_ref,
            diameter_mm=robot_skeleton_ref.diameter_mm,
            color=ROBOT_TUBE_COLOR,
            alpha=0.97,
        )
        if robot_cfg is None:
            robot_visible["value"] = True
        skeleton_ref_lc.set_visible(robot_visible["value"])
        active_body_radius_mm = 0.5 * float(robot_skeleton_ref.diameter_mm)

    # Title further up
    ax.set_title("Tip Trajectory from G-code and Calibration", pad=18)
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.set_zlabel("Z (mm)")

    all_x = xs if not show_stage else np.concatenate([xs, traj.x_stage])
    all_y = ys if not show_stage else np.concatenate([ys, traj.y_stage])
    all_z = zs if not show_stage else np.concatenate([zs, traj.z_stage])
    if robot_skeleton_ref is not None and robot_skeleton_ref.points_xyz_mm.size:
        pts_ref = compute_tracked_skeleton_world(
            robot_skeleton_ref,
            b_cmd=float(traj.b_cmd[idx0]),
            c_cmd_deg=float(traj.c_cmd[idx0]),
            tip_world=(float(traj.x_tip[idx0]), float(traj.y_tip[idx0]), float(traj.z_tip[idx0])),
            equation_phase=str(traj.equation_phase[idx0]),
        )
        all_x = np.concatenate([all_x, pts_ref[:, 0]])
        all_y = np.concatenate([all_y, pts_ref[:, 1]])
        all_z = np.concatenate([all_z, pts_ref[:, 2]])

    center, base_radius = compute_equal_box_center_radius(all_x, all_y, all_z, pad_frac=0.05)
    overlay_radius = 0.0
    if robot_cfg is not None:
        overlay_radius = max(overlay_radius, 0.5 * float(robot_cfg.diameter_mm))
    if robot_skeleton_ref is not None:
        overlay_radius = max(overlay_radius, 0.5 * float(robot_skeleton_ref.diameter_mm))
    base_radius += overlay_radius
    current_zoom = {"value": 1.0}

    def apply_view_limits():
        apply_equal_axes_with_zoom(ax, center=center, base_radius=base_radius, zoom=current_zoom["value"])

    apply_view_limits()
    set_major_view(ax, "ISO")

    # ----- Info panel -----
    text_ax = fig.add_axes([0.04, 0.025, 0.60, 0.165])
    text_ax.set_facecolor("black")
    text_ax.axis("off")
    info_text = text_ax.text(
        0.0, 1.0, "",
        va="top", ha="left", family="monospace", fontsize=8.3, color="white", clip_on=True
    )

    # ----- Index slider -----
    slider_ax = fig.add_axes([0.70, 0.080, 0.26, 0.026], facecolor="#111111")
    idx_slider = Slider(
        ax=slider_ax,
        label="Index",
        valmin=0,
        valmax=n - 1,
        valinit=idx0,
        valstep=1,
    )
    try:
        idx_slider.label.set_color("white")
        idx_slider.valtext.set_color("white")
    except Exception:
        pass

    # ----- Zoom slider -----
    zoom_ax = fig.add_axes([0.70, 0.125, 0.26, 0.026], facecolor="#111111")
    zoom_slider = Slider(
        ax=zoom_ax,
        label="Zoom",
        valmin=0.25,
        valmax=8.0,
        valinit=1.0,
        valstep=0.01,
    )
    try:
        zoom_slider.label.set_color("white")
        zoom_slider.valtext.set_color("white")
    except Exception:
        pass

    def _apply_robot_visibility():
        if robot_lc is not None:
            robot_lc.set_visible(robot_visible["value"])
        if skeleton_ref_lc is not None:
            skeleton_ref_lc.set_visible(robot_visible["value"])

    help_ax = fig.add_axes([0.70, 0.025, 0.28, 0.035], facecolor="black")
    help_ax.axis("off")
    help_ax.text(
        0.0, 1.0,
        "Keys: left/right, j/k, r=robot\n1=XY 2=XZ 3=YZ 0=ISO, +/- zoom",
        va="top", ha="left", fontsize=8, color="white"
    )
    collision_state = {"active": False, "min_dist_mm": float("inf")}

    # ---------------------------
    # Fast redraw (precomputed extrude segments + debounced sliders)
    # ---------------------------
    def fmt_state(i: int) -> str:
        s = states[i]
        f_text = f"{s.feed:.3f}" if s.feed is not None and np.isfinite(s.feed) else "---"
        time_note = "" if skipped_time_segments == 0 else f"  skipped_no_F={skipped_time_segments}"
        rob_line = ""
        if robot_cfg is not None:
            rob_line = (
                f"robot_links: n={robot_cfg.n_links}  dia={robot_cfg.diameter_mm:.2f}mm  "
                f"shape={robot_cfg.shape_mode}  anchor={robot_cfg.anchor_mode}  visible={robot_visible['value']}\n"
            )
        skeleton_line = ""
        if robot_skeleton_ref is not None:
            skeleton_line = (
                f"robot_skeleton: points={robot_skeleton_ref.points_xyz_mm.shape[0]}  "
                f"dia={robot_skeleton_ref.diameter_mm:.2f}mm  tracked={bool(robot_skeleton_ref.phase_point_models)}  "
                f"visible={robot_visible['value']}\n"
            )
        collision_line = ""
        if robot_skeleton_ref is not None:
            min_dist = collision_state["min_dist_mm"]
            min_dist_text = f"{min_dist:.3f}mm" if np.isfinite(min_dist) else "---"
            collision_line = (
                f"body_path_collision={collision_state['active']}  "
                f"threshold={active_body_radius_mm:.3f}mm  min_dist={min_dist_text}  "
                f"highlight=full_print_move\n"
            )
        return (
            f"idx={s.idx:6d}   gcode_line={s.gcode_line_no:6d}   motion={s.motion_code}   F={f_text}\n"
            f"est_time={format_duration(elapsed_seconds[i])} / {format_duration(total_estimated_seconds)}"
            f"  ({elapsed_seconds[i]:.1f}s / {total_estimated_seconds:.1f}s){time_note}\n"
            f"print_active={s.pressure_active}   equation_phase={traj.equation_phase[i]}   "
            f"default_phase={cal.default_phase}\n"
            f"gantry_range: X=[{gantry_ranges['x'][0]: .3f}, {gantry_ranges['x'][1]: .3f}]  "
            f"Y=[{gantry_ranges['y'][0]: .3f}, {gantry_ranges['y'][1]: .3f}]  "
            f"Z=[{gantry_ranges['z'][0]: .3f}, {gantry_ranges['z'][1]: .3f}]\n"
            f"stage: X={traj.x_stage[i]: .4f}  Y={traj.y_stage[i]: .4f}  Z={traj.z_stage[i]: .4f}  "
            f"{cal.pull_axis}={traj.b_cmd[i]: .4f}  {cal.rot_axis}={traj.c_cmd[i]: .4f}\n"
            f"tip:   X={traj.x_tip[i]: .4f}  Y={traj.y_tip[i]: .4f}  Z={traj.z_tip[i]: .4f}    "
            f"r(B)={traj.r_of_b[i]: .4f}  y_off(B)={traj.y_off_of_b[i]: .4f}  z(B)={traj.z_of_b[i]: .4f}  "
            f"tip_angle={traj.tip_angle_deg[i]: .3f}°  sgn={int(traj.sgn[i]):+d}\n"
            f"{rob_line}"
            f"{skeleton_line}"
            f"{collision_line}"
            f"zoom={current_zoom['value']:.2f}x   raw: {s.gcode_raw}"
        )

    def _update_print_segments(i: int):
        if i <= 0 or extrude_segs.shape[0] == 0:
            current_print_lc.set_segments([])
            return np.empty((0, 2, 3), dtype=float)
        # how many extrude segments end at/before i?
        k = int(np.searchsorted(extrude_end_idx, i, side="right"))
        visible_printed_segs = extrude_segs[:k]
        current_print_lc.set_segments(visible_printed_segs)
        current_print_lc.set_color(printed_line_color)
        return visible_printed_segs

    def _update_collision_overlay(visible_printed_segs: np.ndarray, body_points_xyz: Optional[np.ndarray]):
        collision_state["active"] = False
        collision_state["min_dist_mm"] = float("inf")
        if robot_skeleton_ref is None or body_points_xyz is None or active_body_radius_mm <= 0.0:
            current_collision_lc.set_segments([])
            return
        hit_segments, collision_mask, is_collision, min_dist = _compute_collision_highlight_segments(
            body_points_xyz=body_points_xyz,
            candidate_segments_xyz=visible_printed_segs,
            collision_radius_mm=active_body_radius_mm,
        )
        collision_state["active"] = bool(is_collision)
        collision_state["min_dist_mm"] = float(min_dist)
        if visible_printed_segs.shape[0] > 0:
            current_print_lc.set_segments(visible_printed_segs[~collision_mask])
        current_collision_lc.set_segments(hit_segments)

    def _update_robot(i: int):
        if robot_cfg is None or robot_lc is None:
            return
        rp = compute_robot_polyline_world(
            cal, robot_cfg,
            x_stage=traj.x_stage[i],
            y_stage=traj.y_stage[i],
            z_stage=traj.z_stage[i],
            b_cmd=traj.b_cmd[i],
            c_cmd_deg=traj.c_cmd[i],
            tip_world=(traj.x_tip[i], traj.y_tip[i], traj.z_tip[i]),
            tip_angle_deg=traj.tip_angle_deg[i],
            r_tip=traj.r_of_b[i],
            z_tip=traj.z_of_b[i],
            y_off_tip=traj.y_off_of_b[i],
            equation_phase=str(traj.equation_phase[i]),
            offplane_sign=offplane_sign,
            start_world=robot_start_world,
        )
        update_tube_artist(
            robot_lc,
            rp,
            diameter_mm=robot_cfg.diameter_mm,
            color=ROBOT_TUBE_COLOR,
            alpha=0.94,
        )

    def _update_skeleton(i: int) -> Optional[np.ndarray]:
        if robot_skeleton_ref is None or skeleton_ref_lc is None:
            return None
        rp = compute_tracked_skeleton_world(
            robot_skeleton_ref,
            b_cmd=float(traj.b_cmd[i]),
            c_cmd_deg=float(traj.c_cmd[i]),
            tip_world=(float(traj.x_tip[i]), float(traj.y_tip[i]), float(traj.z_tip[i])),
            equation_phase=str(traj.equation_phase[i]),
        )
        return update_tube_artist(
            skeleton_ref_lc,
            rp,
            diameter_mm=robot_skeleton_ref.diameter_mm,
            color=ROBOT_TUBE_COLOR,
            alpha=0.97,
        )

    def redraw(i: int):
        i = int(np.clip(i, 0, n - 1))
        view_state = capture_view_state(ax)
        current_tip_marker.set_data_3d([traj.x_tip[i]], [traj.y_tip[i]], [traj.z_tip[i]])
        current_stage_marker.set_data_3d([traj.x_stage[i]], [traj.y_stage[i]], [traj.z_stage[i]])
        visible_printed_segs = _update_print_segments(i)
        _update_robot(i)
        body_points_xyz = _update_skeleton(i)
        _update_collision_overlay(visible_printed_segs, body_points_xyz)
        info_text.set_text(fmt_state(i))
        restore_view_state(ax, view_state)
        fig.canvas.draw_idle()

    # Debounce helpers (so sliders feel fluid)
    pending = {"idx": idx0, "zoom": 1.0, "idx_dirty": False, "zoom_dirty": False}

    idx_timer = fig.canvas.new_timer(interval=UI_DEBOUNCE_MS_INDEX)
    zoom_timer = fig.canvas.new_timer(interval=UI_DEBOUNCE_MS_ZOOM)

    def _idx_timer_cb():
        if pending["idx_dirty"]:
            pending["idx_dirty"] = False
            redraw(int(pending["idx"]))
        # returning None is fine for mpl timers

    def _zoom_timer_cb():
        if pending["zoom_dirty"]:
            pending["zoom_dirty"] = False
            current_zoom["value"] = float(pending["zoom"])
            apply_view_limits()
            redraw(int(idx_slider.val))

    idx_timer.add_callback(_idx_timer_cb)
    zoom_timer.add_callback(_zoom_timer_cb)

    def on_index_slider(val):
        pending["idx"] = int(val)
        pending["idx_dirty"] = True
        try:
            idx_timer.stop()
        except Exception:
            pass
        idx_timer.start()

    def on_zoom_slider(val):
        pending["zoom"] = float(val)
        pending["zoom_dirty"] = True
        try:
            zoom_timer.stop()
        except Exception:
            pass
        zoom_timer.start()

    idx_slider.on_changed(on_index_slider)
    zoom_slider.on_changed(on_zoom_slider)

    def on_key(event):
        key = event.key.lower() if isinstance(event.key, str) else event.key
        i = int(idx_slider.val)

        if key in ("right", "k", "l"):
            idx_slider.set_val(min(n - 1, i + 1))
        elif key in ("left", "j", "h"):
            idx_slider.set_val(max(0, i - 1))
        elif key == "pagedown":
            idx_slider.set_val(min(n - 1, i + 10))
        elif key == "pageup":
            idx_slider.set_val(max(0, i - 10))
        elif key == "home":
            idx_slider.set_val(0)
        elif key == "end":
            idx_slider.set_val(n - 1)
        elif key == "1":
            set_major_view(ax, "XY"); apply_view_limits(); fig.canvas.draw_idle()
        elif key == "2":
            set_major_view(ax, "XZ"); apply_view_limits(); fig.canvas.draw_idle()
        elif key == "3":
            set_major_view(ax, "YZ"); apply_view_limits(); fig.canvas.draw_idle()
        elif key == "0":
            set_major_view(ax, "ISO"); apply_view_limits(); fig.canvas.draw_idle()
        elif key in ("+", "="):
            zoom_slider.set_val(min(8.0, float(zoom_slider.val) * 1.15))
        elif key in ("-", "_"):
            zoom_slider.set_val(max(0.25, float(zoom_slider.val) / 1.15))
        elif key == "r":
            robot_visible["value"] = not robot_visible["value"]
            _apply_robot_visibility()
            fig.canvas.draw_idle()

    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw(idx0)
    plt.show()


# =========================
# CLI / main
# =========================
def main():
    ap = argparse.ArgumentParser(description="Interactive 3D tip-position plot from G-code + calibration JSON.")
    ap.add_argument("--gcode", required=True, help="Path to input G-code file.")
    ap.add_argument("--calibration", required=True, help="Path to calibration JSON (shadow_calibration schema).")
    ap.add_argument("--robot-config", default=None,
                    help="Optional robot skeleton config JSON (e.g. *_skeleton_parametric.json). "
                         "If omitted, tries to auto-load from calibration JSON exported_models.")
    ap.add_argument("--robot-skeleton-file", default=None,
                    help="Optional exported skeleton JSON to overlay directly from reference_pose.points_xyz_mm in millimeters.")
    ap.add_argument("--y-axis", default="Y", help="Stage Y axis letter in G-code (default: Y).")
    ap.add_argument("--c0-deg", type=float, default=DEFAULT_C0_DEG,
                    help="C value corresponding to +X sign in calibration model (default: 0).")
    ap.add_argument("--show-stage", action="store_true", help="Overlay stage path.")
    ap.add_argument("--print-summary", action="store_true", help="Print summary before plotting.")
    ap.add_argument("--offplane-sign", type=float, default=DEFAULT_OFFPLANE_SIGN,
                    help="Sign multiplier applied to offplane_y(B) when reconstructing tip/links (default -1).")
    args = ap.parse_args()

    cal_data = load_calibration_json(args.calibration)
    cal = load_calibration(args.calibration)
    gcode_metadata = parse_gcode_metadata(args.gcode)
    write_mode = str(gcode_metadata.get("write_mode", "calibrated")).strip().lower()

    robot_cfg = None
    robot_skeleton_ref = None

    robot_cfg_path = Path(args.robot_config).expanduser().resolve() if args.robot_config else None
    auto_robot_path = resolve_robot_config_from_calibration(args.calibration, cal_data) if not args.robot_config else None
    skeleton_path = Path(args.robot_skeleton_file).expanduser().resolve() if args.robot_skeleton_file else None

    if robot_cfg_path is not None:
        if robot_cfg_path.exists():
            try:
                robot_cfg = load_robot_config(str(robot_cfg_path))
                print(f"[INFO] Loaded robot config: {robot_cfg_path}")
            except Exception as e:
                print(f"[WARN] Failed to load robot config ({robot_cfg_path}): {e}")
        else:
            print(f"[WARN] Robot config path not found: {args.robot_config}")

    if skeleton_path is None and auto_robot_path is not None and auto_robot_path.exists():
        try:
            auto_robot_data = _load_json_file(str(auto_robot_path))
        except Exception:
            auto_robot_data = {}
        auto_type = str(auto_robot_data.get("type", "")).strip().lower()
        if auto_type == "tracked_visible_skeleton_points":
            skeleton_path = auto_robot_path
        elif robot_cfg is None:
            try:
                robot_cfg = load_robot_config(str(auto_robot_path))
                print(f"[INFO] Loaded robot config: {auto_robot_path}")
            except Exception as e:
                print(f"[WARN] Failed to load robot config ({auto_robot_path}): {e}")

    if skeleton_path is not None:
        if skeleton_path.exists():
            try:
                robot_skeleton_ref = load_robot_skeleton_reference(str(skeleton_path))
                print(f"[INFO] Loaded robot skeleton: {skeleton_path}")
            except Exception as e:
                print(f"[WARN] Failed to load robot skeleton ({skeleton_path}): {e}")
                robot_skeleton_ref = None
        else:
            print(f"[WARN] Robot skeleton path not found: {skeleton_path}")

    y_axis = args.y_axis.upper()

    states = parse_gcode_motion_states(
        gcode_path=args.gcode,
        x_axis=cal.x_axis,
        y_axis=y_axis,
        z_axis=cal.z_axis,
        b_axis=cal.pull_axis,
        c_axis=cal.rot_axis,
        default_c0=args.c0_deg,
    )
    traj = reconstruct_tip_trajectory(
        states,
        cal=cal,
        c0_deg=args.c0_deg,
        offplane_sign=float(args.offplane_sign),
        write_mode=write_mode,
    )
    print_gantry_ranges_used(
        traj,
        axis_names={
            "x": cal.x_axis,
            "y": y_axis,
            "z": cal.z_axis,
            "b": cal.pull_axis,
            "c": cal.rot_axis,
        },
    )

    if args.print_summary:
        c_unique = np.unique(np.round(traj.c_cmd, 6))
        print(f"Parsed {len(states)} motion states from {args.gcode}")
        print(f"Axis mapping: X={cal.x_axis}, Y={y_axis}, Z={cal.z_axis}, B={cal.pull_axis}, C={cal.rot_axis}")
        print(f"C0={args.c0_deg:.3f}, C180(cal)={cal.c_180_deg:.3f}")
        print(f"write_mode={write_mode}")
        print(f"offplane_sign={float(args.offplane_sign):.3f}")
        print(f"B home={cal.b_home:.4f}")
        print(f"Unique C values (rounded): {c_unique.tolist()[:20]}{' ...' if len(c_unique) > 20 else ''}")
        print(f"B range used: [{np.min(traj.b_cmd):.4f}, {np.max(traj.b_cmd):.4f}] (cal: [{cal.b_min:.4f}, {cal.b_max:.4f}])")
        print(f"TIP X range: [{np.min(traj.x_tip):.4f}, {np.max(traj.x_tip):.4f}]")
        print(f"TIP Y range: [{np.min(traj.y_tip):.4f}, {np.max(traj.y_tip):.4f}]")
        print(f"TIP Z range: [{np.min(traj.z_tip):.4f}, {np.max(traj.z_tip):.4f}]")
        if robot_cfg is not None:
            print(
                f"Robot links enabled: n_links={robot_cfg.n_links}, dia={robot_cfg.diameter_mm:.2f}mm, "
                f"shape={robot_cfg.shape_mode}, anchor={robot_cfg.anchor_mode}, knots={len(robot_cfg.t_knots)}"
            )
        if robot_skeleton_ref is not None:
            mins = np.min(robot_skeleton_ref.points_xyz_mm, axis=0)
            maxs = np.max(robot_skeleton_ref.points_xyz_mm, axis=0)
            print(
                f"Robot skeleton enabled: points={robot_skeleton_ref.points_xyz_mm.shape[0]}, "
                f"dia={robot_skeleton_ref.diameter_mm:.2f}mm, "
                f"tracked={bool(robot_skeleton_ref.phase_point_models)}, "
                f"bounds_mm=X[{mins[0]:.4f}, {maxs[0]:.4f}] "
                f"Y[{mins[1]:.4f}, {maxs[1]:.4f}] "
                f"Z[{mins[2]:.4f}, {maxs[2]:.4f}]"
            )

    launch_interactive_plot(
        states=states,
        traj=traj,
        cal=cal,
        robot_cfg=robot_cfg,
        robot_skeleton_ref=robot_skeleton_ref,
        c0_deg=args.c0_deg,
        offplane_sign=float(args.offplane_sign),
        show_stage=args.show_stage,
    )


if __name__ == "__main__":
    main()
