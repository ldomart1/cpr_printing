#!/usr/bin/env python3
"""
klein_bottle_gui_slicer.py

Interactive Klein-bottle-like G-code generator with live preview.

What this script does
---------------------
1) Lets you tweak the swept-surface geometry live with a GUI.
2) Exports Duet/RRF-friendly G-code using a calibration JSON.
3) Supports shell -> lattice -> shell switching along the curve.
4) Uses calibration-aware tip tracking with one continuous centroid-following
   orientation rule across the print path.
5) Supports a large base diameter, a thinner tube diameter, a smooth fillet between them,
   and an additional outer-edge roundover control on the large base.

Notes
-----
- The geometry is a controllable "Klein-bottle-like" swept surface rather than an exact
  analytic Klein bottle. That gives you robust local control and predictable toolpaths.
- The lattice region is a low-poly surface-following mesh made from rings, meridians,
  and alternating diagonals. It behaves like an octahedral / diamond-style skin lattice.
- Orientation follows the local spine/centroid direction continuously across the exported
  print path rather than switching between separate base and surface tracking modes.

Dependencies
------------
Python 3.10+
NumPy, Matplotlib
Tkinter (usually ships with Python)

Usage
-----
python3 klein_bottle_gui_slicer.py
python3 klein_bottle_gui_slicer.py --calibration path/to/calibration.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import traceback
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
try:
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:
    matplotlib.use("Agg")
    FigureCanvasTkAgg = None
from matplotlib.figure import Figure


# ============================================================================
# Defaults
# ============================================================================

DEFAULT_OUT = "klein_bottle_shell_lattice_shell.gcode"

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
DEFAULT_TRAVEL_FEED = 1200.0
DEFAULT_PRINT_FEED = 300.0
DEFAULT_EXTRUSION_PER_MM = 0.01
DEFAULT_PRIME_MM = 1.0
DEFAULT_PRESSURE_OFFSET_MM = 5.0
DEFAULT_PRESSURE_ADVANCE_FEED = 120.0
DEFAULT_PRESSURE_RETRACT_FEED = 240.0
DEFAULT_PREFLOW_DWELL_MS = 150
DEFAULT_END_DWELL_MS = 250
DEFAULT_DEBUG_EVERY = 250

# Sampling
DEFAULT_PREVIEW_SPINE_SAMPLES = 240
DEFAULT_PREVIEW_WIREFRAME_U = 48
DEFAULT_PREVIEW_WIREFRAME_V = 24
DEFAULT_BASE_RING_SEGMENTS = 160
DEFAULT_PATH_STEPS = 2600
DEFAULT_LATTICE_EDGE_SAMPLES = 5

# Spine geometry
DEFAULT_SPINE_HEIGHT = 80.0
DEFAULT_SPINE_R0 = 45.0
DEFAULT_SPINE_R1 = 15.0
DEFAULT_SPINE_S = 18.0
DEFAULT_SPINE_Z_WOBBLE = 0.0
DEFAULT_TWIST_DEG = 180.0
DEFAULT_CONTINUOUS_LAYER_HEIGHT = 0.9
DEFAULT_SHELL_PHASE_DEG = 0.0

# Radius controls
DEFAULT_LARGE_BASE_RADIUS = 30.0
DEFAULT_LARGE_BASE_HEIGHT = 0.0
DEFAULT_BASE_TUBE_RADIUS = 14.0
DEFAULT_BASE_TUBE_HEIGHT = 14.0
DEFAULT_NECK_RADIUS = 5.0
DEFAULT_NECK_CENTER = 0.52
DEFAULT_NECK_WIDTH = 0.24
DEFAULT_NECK_POWER = 3.0
DEFAULT_BASE_FILLET_SPAN = 0.12
DEFAULT_BASE_OUTER_EDGE_FILLET = 3.0

# Dark UI colors
UI_BG = "#17191f"
UI_PANEL_BG = "#20242c"
UI_PANEL_ALT_BG = "#262b34"
UI_FG = "#e6edf3"
UI_MUTED_FG = "#a8b3c2"
UI_ACCENT = "#5aa9ff"
UI_BORDER = "#3b4350"
UI_ENTRY_BG = "#11151b"
UI_PLOT_BG = "#11151b"
UI_GRID = "#4b5563"

# Lattice window + density
DEFAULT_LATTICE_START = 0.42
DEFAULT_LATTICE_END = 0.70
DEFAULT_LATTICE_U_COUNT = 11
DEFAULT_LATTICE_V_COUNT = 12
DEFAULT_LATTICE_INCLUDE_RINGS = 1
DEFAULT_LATTICE_INCLUDE_MERIDIANS = 1
DEFAULT_LATTICE_INCLUDE_DIAGONALS = 1

# Orientation convention
DEFAULT_TOOL_NORMAL_SIGN = 1

# BBox
DEFAULT_BBOX_X_MIN = 0.0
DEFAULT_BBOX_X_MAX = 160.0
DEFAULT_BBOX_Y_MIN = 0.0
DEFAULT_BBOX_Y_MAX = 200.0
DEFAULT_BBOX_Z_MIN = -168.0
DEFAULT_BBOX_Z_MAX = 0.0


# ============================================================================
# Data classes
# ============================================================================

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
    selected_fit_model: Optional[str] = None
    selected_offplane_fit_model: Optional[str] = None
    active_phase: str = "pull"


@dataclass
class Params:
    calibration_path: str = ""
    output_path: str = DEFAULT_OUT

    start_x: float = DEFAULT_START_X
    start_y: float = DEFAULT_START_Y
    start_z: float = DEFAULT_START_Z

    machine_start_x: float = DEFAULT_MACHINE_START_X
    machine_start_y: float = DEFAULT_MACHINE_START_Y
    machine_start_z: float = DEFAULT_MACHINE_START_Z
    machine_start_b: float = DEFAULT_MACHINE_START_B
    machine_start_c: float = DEFAULT_MACHINE_START_C

    machine_end_x: float = DEFAULT_MACHINE_END_X
    machine_end_y: float = DEFAULT_MACHINE_END_Y
    machine_end_z: float = DEFAULT_MACHINE_END_Z
    machine_end_b: float = DEFAULT_MACHINE_END_B
    machine_end_c: float = DEFAULT_MACHINE_END_C

    safe_approach_z: float = DEFAULT_SAFE_APPROACH_Z

    travel_feed: float = DEFAULT_TRAVEL_FEED
    print_feed: float = DEFAULT_PRINT_FEED
    extrusion_per_mm: float = DEFAULT_EXTRUSION_PER_MM
    prime_mm: float = DEFAULT_PRIME_MM
    pressure_offset_mm: float = DEFAULT_PRESSURE_OFFSET_MM
    pressure_advance_feed: float = DEFAULT_PRESSURE_ADVANCE_FEED
    pressure_retract_feed: float = DEFAULT_PRESSURE_RETRACT_FEED
    preflow_dwell_ms: int = DEFAULT_PREFLOW_DWELL_MS
    end_dwell_ms: int = DEFAULT_END_DWELL_MS
    debug_every: int = DEFAULT_DEBUG_EVERY

    preview_spine_samples: int = DEFAULT_PREVIEW_SPINE_SAMPLES
    preview_wireframe_u: int = DEFAULT_PREVIEW_WIREFRAME_U
    preview_wireframe_v: int = DEFAULT_PREVIEW_WIREFRAME_V
    base_ring_segments: int = DEFAULT_BASE_RING_SEGMENTS
    path_steps: int = DEFAULT_PATH_STEPS
    lattice_edge_samples: int = DEFAULT_LATTICE_EDGE_SAMPLES

    spine_height: float = DEFAULT_SPINE_HEIGHT
    spine_r0: float = DEFAULT_SPINE_R0
    spine_r1: float = DEFAULT_SPINE_R1
    spine_s: float = DEFAULT_SPINE_S
    spine_z_wobble: float = DEFAULT_SPINE_Z_WOBBLE
    twist_deg: float = DEFAULT_TWIST_DEG
    continuous_layer_height: float = DEFAULT_CONTINUOUS_LAYER_HEIGHT
    shell_phase_deg: float = DEFAULT_SHELL_PHASE_DEG

    large_base_radius: float = DEFAULT_LARGE_BASE_RADIUS
    large_base_height: float = DEFAULT_LARGE_BASE_HEIGHT
    base_tube_radius: float = DEFAULT_BASE_TUBE_RADIUS
    base_tube_height: float = DEFAULT_BASE_TUBE_HEIGHT
    neck_radius: float = DEFAULT_NECK_RADIUS
    neck_center: float = DEFAULT_NECK_CENTER
    neck_width: float = DEFAULT_NECK_WIDTH
    neck_power: float = DEFAULT_NECK_POWER
    base_fillet_span: float = DEFAULT_BASE_FILLET_SPAN
    base_outer_edge_fillet: float = DEFAULT_BASE_OUTER_EDGE_FILLET

    lattice_start: float = DEFAULT_LATTICE_START
    lattice_end: float = DEFAULT_LATTICE_END
    lattice_u_count: int = DEFAULT_LATTICE_U_COUNT
    lattice_v_count: int = DEFAULT_LATTICE_V_COUNT
    lattice_include_rings: int = DEFAULT_LATTICE_INCLUDE_RINGS
    lattice_include_meridians: int = DEFAULT_LATTICE_INCLUDE_MERIDIANS
    lattice_include_diagonals: int = DEFAULT_LATTICE_INCLUDE_DIAGONALS

    tool_normal_sign: int = DEFAULT_TOOL_NORMAL_SIGN

    bbox_x_min: float = DEFAULT_BBOX_X_MIN
    bbox_x_max: float = DEFAULT_BBOX_X_MAX
    bbox_y_min: float = DEFAULT_BBOX_Y_MIN
    bbox_y_max: float = DEFAULT_BBOX_Y_MAX
    bbox_z_min: float = DEFAULT_BBOX_Z_MIN
    bbox_z_max: float = DEFAULT_BBOX_Z_MAX

    show_wireframe: int = 1
    show_normals: int = 0
    normal_stride: int = 24


@dataclass
class SurfaceCache:
    ts: np.ndarray
    spine: np.ndarray
    spine_s_mm: np.ndarray
    T: np.ndarray
    N: np.ndarray
    B: np.ndarray
    Nt: np.ndarray
    Bt: np.ndarray


@dataclass
class ParamPoint:
    t: float
    phi: float
    mode: str
    comment: str = ""
    track_sign: float = 1.0
    track_mode: str = "centerline_tangent"
    track_u: float = 0.0


@dataclass
class ParamPolyline:
    name: str
    points: List[ParamPoint]


@dataclass
class Waypoint:
    tip_xyz: np.ndarray
    b_cmd: float
    c_deg: float
    comment: str = ""


# ============================================================================
# Parameter specs for GUI
# ============================================================================

PARAM_GROUPS: List[Tuple[str, List[Dict[str, Any]]]] = [
    (
        "Placement",
        [
            {"key": "start_x", "label": "Start X", "from_": -50.0, "to": 150.0, "resolution": 0.5, "step": 0.5},
            {"key": "start_y", "label": "Start Y", "from_": -50.0, "to": 150.0, "resolution": 0.5, "step": 0.5},
            {"key": "start_z", "label": "Start Z", "from_": -250.0, "to": 50.0, "resolution": 0.5, "step": 0.5},
        ],
    ),
    (
        "Spine",
        [
            {"key": "spine_height", "label": "Spine Height", "from_": 10.0, "to": 180.0, "resolution": 0.5, "step": 0.5},
            {"key": "spine_r0", "label": "Spine R0", "from_": 0.0, "to": 100.0, "resolution": 0.5, "step": 0.5},
            {"key": "spine_r1", "label": "Spine R1", "from_": 0.0, "to": 80.0, "resolution": 0.5, "step": 0.5},
            {"key": "spine_s", "label": "Crossing Term", "from_": -80.0, "to": 80.0, "resolution": 0.5, "step": 0.5},
            {"key": "spine_z_wobble", "label": "Z Wobble", "from_": -40.0, "to": 40.0, "resolution": 0.25, "step": 0.25},
            {"key": "twist_deg", "label": "Twist Deg", "from_": -720.0, "to": 720.0, "resolution": 1.0, "step": 1.0},
            {"key": "continuous_layer_height", "label": "Continuous Layer H", "from_": 0.1, "to": 10.0, "resolution": 0.05, "step": 0.05},
            {"key": "shell_phase_deg", "label": "Shell Phase", "from_": -180.0, "to": 180.0, "resolution": 1.0, "step": 1.0},
        ],
    ),
    (
        "Radius / Base / Neck",
        [
            {"key": "large_base_radius", "label": "Large Base Radius", "from_": 1.0, "to": 60.0, "resolution": 0.25, "step": 0.25},
            {"key": "large_base_height", "label": "Large Base Height", "from_": 0.0, "to": 80.0, "resolution": 0.25, "step": 0.25},
            {"key": "base_tube_radius", "label": "Base Tube Radius", "from_": 1.0, "to": 40.0, "resolution": 0.25, "step": 0.25},
            {"key": "base_tube_height", "label": "Base Tube Height", "from_": 0.0, "to": 80.0, "resolution": 0.25, "step": 0.25},
            {"key": "neck_radius", "label": "Neck Radius", "from_": 1.0, "to": 25.0, "resolution": 0.25, "step": 0.25},
            {"key": "neck_center", "label": "Neck Center", "from_": 0.0, "to": 1.0, "resolution": 0.005, "step": 0.005},
            {"key": "neck_width", "label": "Neck Width", "from_": 0.01, "to": 1.0, "resolution": 0.005, "step": 0.005},
            {"key": "neck_power", "label": "Neck Power", "from_": 0.5, "to": 8.0, "resolution": 0.05, "step": 0.05},
            {"key": "base_fillet_span", "label": "Base Fillet Span", "from_": 0.0, "to": 0.5, "resolution": 0.002, "step": 0.002},
            {"key": "base_outer_edge_fillet", "label": "Base Outer Edge Fillet", "from_": 0.0, "to": 20.0, "resolution": 0.1, "step": 0.1},
        ],
    ),
    (
        "Lattice Window",
        [
            {"key": "lattice_start", "label": "Lattice Start", "from_": 0.0, "to": 1.0, "resolution": 0.005, "step": 0.005},
            {"key": "lattice_end", "label": "Lattice End", "from_": 0.0, "to": 1.0, "resolution": 0.005, "step": 0.005},
            {"key": "lattice_u_count", "label": "Lattice U Count", "from_": 2, "to": 30, "resolution": 1, "step": 1},
            {"key": "lattice_v_count", "label": "Lattice V Count", "from_": 3, "to": 36, "resolution": 1, "step": 1},
            {"key": "lattice_edge_samples", "label": "Lattice Edge Samples", "from_": 2, "to": 20, "resolution": 1, "step": 1},
        ],
    ),
    (
        "Sampling / Orientation",
        [
            {"key": "base_ring_segments", "label": "Base Ring Segments", "from_": 16, "to": 400, "resolution": 1, "step": 1},
            {"key": "path_steps", "label": "Shell Path Steps", "from_": 100, "to": 12000, "resolution": 10, "step": 10},
            {"key": "preview_spine_samples", "label": "Preview Samples", "from_": 60, "to": 1000, "resolution": 10, "step": 10},
            {"key": "tool_normal_sign", "label": "Tool Normal Sign", "from_": -1, "to": 1, "resolution": 2, "step": 2},
            {"key": "normal_stride", "label": "Normal Stride", "from_": 4, "to": 100, "resolution": 1, "step": 1},
        ],
    ),
    (
        "Process",
        [
            {"key": "travel_feed", "label": "Travel Feed", "from_": 50.0, "to": 5000.0, "resolution": 10.0, "step": 10.0},
            {"key": "print_feed", "label": "Print Feed", "from_": 10.0, "to": 2000.0, "resolution": 5.0, "step": 5.0},
            {"key": "extrusion_per_mm", "label": "Extrusion / mm", "from_": 0.0, "to": 0.2, "resolution": 0.0005, "step": 0.0005},
            {"key": "safe_approach_z", "label": "Safe Approach Z", "from_": -50.0, "to": 50.0, "resolution": 0.5, "step": 0.5},
        ],
    ),
]


# ============================================================================
# Numeric helpers
# ============================================================================

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def smoothstep(x: float) -> float:
    x = clamp01(x)
    return x * x * (3.0 - 2.0 * x)


def smootherstep(x: float) -> float:
    x = clamp01(x)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def lerp(a: np.ndarray, b: np.ndarray, t: float) -> np.ndarray:
    return (1.0 - t) * a + t * b


def _normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    vv = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(vv))
    if n < eps:
        return np.zeros_like(vv)
    return vv / n


def _rodrigues_rotate(v: np.ndarray, axis: np.ndarray, angle_rad: float) -> np.ndarray:
    axis = _normalize(axis)
    if np.linalg.norm(axis) < 1e-12:
        return np.asarray(v, dtype=float).copy()
    v = np.asarray(v, dtype=float)
    c = math.cos(angle_rad)
    s = math.sin(angle_rad)
    return v * c + np.cross(axis, v) * s + axis * (np.dot(axis, v)) * (1.0 - c)


def unwrap_degrees(prev: float, cur: float) -> float:
    a = float(cur)
    p = float(prev)
    while a - p > 180.0:
        a -= 360.0
    while a - p < -180.0:
        a += 360.0
    return a


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


def _coeffs_from_models(models: Dict[str, Any], *names: str) -> Optional[np.ndarray]:
    for name in names:
        spec = models.get(name)
        if not isinstance(spec, dict):
            continue
        coeffs = spec.get("coefficients", spec.get("coeffs"))
        if coeffs is None:
            continue
        arr = np.asarray(coeffs, dtype=float).reshape(-1)
        if arr.size > 0:
            return arr
    return None


# ============================================================================
# Calibration / kinematics
# ============================================================================

def load_calibration(json_path: str) -> Calibration:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cubic = data.get("cubic_coefficients", {}) or {}
    fit_models = data.get("fit_models", {}) or {}
    selected_fit_model = data.get("selected_fit_model")
    selected_fit_model = None if selected_fit_model is None else str(selected_fit_model).strip().lower()
    selected_offplane_fit_model = data.get("selected_offplane_fit_model")
    selected_offplane_fit_model = None if selected_offplane_fit_model is None else str(selected_offplane_fit_model).strip().lower()
    active_phase = str(data.get("default_phase_for_legacy_access") or "pull").strip().lower()

    phase_models = data.get("fit_models_by_phase", {}) or {}
    active_models = phase_models.get(active_phase) if isinstance(phase_models, dict) else None
    if not isinstance(active_models, dict):
        active_models = fit_models

    pr = (np.asarray(cubic["r_coeffs"], dtype=float) if cubic.get("r_coeffs") is not None
          else (_coeffs_from_models(active_models, "r_cubic", "r_avg_cubic") or np.zeros(1)))
    pz = (np.asarray(cubic["z_coeffs"], dtype=float) if cubic.get("z_coeffs") is not None
          else (_coeffs_from_models(active_models, "z_cubic", "z_avg_cubic") or np.zeros(1)))

    py_off = _coeffs_from_models(
        active_models,
        "offplane_y_avg_cubic",
        "offplane_y_cubic",
        "offplane_y",
        "offplane_y_linear",
        "offplane_y_avg_linear",
    )
    if py_off is None and cubic.get("offplane_y_coeffs") is not None:
        py_off = np.asarray(cubic["offplane_y_coeffs"], dtype=float)

    pa = (np.asarray(cubic["tip_angle_coeffs"], dtype=float) if cubic.get("tip_angle_coeffs") is not None
          else _coeffs_from_models(active_models, "tip_angle_cubic", "tip_angle_avg_cubic"))

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
        b_min=b_min,
        b_max=b_max,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        b_axis=b_axis,
        c_axis=c_axis,
        u_axis=u_axis,
        c_180_deg=c_180,
        selected_fit_model=selected_fit_model,
        selected_offplane_fit_model=selected_offplane_fit_model,
        active_phase=active_phase,
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
    r = float(eval_r(cal, b_cmd))
    z = float(eval_z(cal, b_cmd))
    y_off = float(eval_offplane_y(cal, b_cmd))

    c = math.radians(float(c_deg))
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def stage_xyz_for_tip(cal: Calibration, tip_xyz: np.ndarray, b_cmd: float, c_deg: float) -> np.ndarray:
    return np.asarray(tip_xyz, dtype=float) - tip_offset_xyz_physical(cal, b_cmd, c_deg)


class TipAngleInverter:
    """Approximate inverse for tip_angle_deg(B_cmd)."""

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


# ============================================================================
# Geometry core
# ============================================================================

def kleinish_spine(
    t: float,
    base_xyz: np.ndarray,
    base_tube_height: float,
    height: float,
    r0: float,
    r1: float,
    s_cross: float,
    z_wobble: float,
) -> np.ndarray:
    """
    Smooth up-and-down centerline with endpoint vertical tangent and an optional
    vertical base tube before the main loop departs laterally.
    """
    t = float(t)
    base_tube_height = max(0.0, float(base_tube_height))
    height = max(0.0, float(height))
    total_vertical = base_tube_height + height
    if total_vertical <= 1e-9:
        return np.asarray(base_xyz, dtype=float).copy()

    base_frac = base_tube_height / total_vertical
    if base_frac >= 1.0 - 1e-9:
        return np.array(
            [
                float(base_xyz[0]),
                float(base_xyz[1]),
                float(base_xyz[2]) + base_tube_height * clamp01(t),
            ],
            dtype=float,
        )

    if t <= base_frac:
        s_lin = 0.0 if base_frac <= 1e-9 else (t / base_frac)
        return np.array(
            [
                float(base_xyz[0]),
                float(base_xyz[1]),
                float(base_xyz[2]) + base_tube_height * s_lin,
            ],
            dtype=float,
        )

    t = (t - base_frac) / max(1e-9, 1.0 - base_frac)
    u = 2.0 * math.pi * t
    s = math.sin(math.pi * t)
    s2 = s * s

    loop_x = (r0 + r1 * math.cos(u)) * math.cos(u) + s_cross * math.sin(2.0 * u)
    loop_y = (r0 + r1 * math.cos(u)) * math.sin(u)

    x = float(base_xyz[0]) + s2 * loop_x
    y = float(base_xyz[1]) + s2 * loop_y
    z = float(base_xyz[2]) + base_tube_height + height * math.sin(math.pi * t) + z_wobble * s2 * math.sin(2.0 * u)
    return np.array([x, y, z], dtype=float)


def build_rmf_frames(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    n = T.shape[0]
    N2 = np.zeros_like(N)
    B2 = np.zeros_like(B)
    for i in range(n):
        frac = 0.0 if n == 1 else (i / (n - 1))
        ang = math.radians(float(twist_deg_total) * frac)
        N2[i] = _normalize(_rodrigues_rotate(N[i], T[i], ang))
        B2[i] = _normalize(np.cross(T[i], N2[i]))
    return N2, B2


def body_radius_profile(t: float, p: Params) -> float:
    half = max(1e-6, 0.5 * float(p.neck_width))
    d = abs(float(t) - float(p.neck_center)) / half
    w = 1.0 - smoothstep(min(1.0, d))
    w = w ** float(p.neck_power)
    return float(p.base_tube_radius) * (1.0 - w) + float(p.neck_radius) * w


def base_tube_fraction(p: Params) -> float:
    base_h = max(0.0, float(p.base_tube_height))
    loop_h = max(0.0, float(p.spine_height))
    total = base_h + loop_h
    if total <= 1e-9:
        return 0.0
    return clamp01(base_h / total)


def vertical_height_at_t(t: float, p: Params) -> float:
    t = clamp01(t)
    base_h = max(0.0, float(p.base_tube_height))
    loop_h = max(0.0, float(p.spine_height))
    frac = base_tube_fraction(p)
    if frac > 1.0 - 1e-9:
        return base_h * t
    if frac > 1e-9 and t <= frac:
        return base_h * (t / frac)
    if loop_h <= 1e-9:
        return base_h
    u = 0.0 if frac >= 1.0 else (t - frac) / max(1e-9, 1.0 - frac)
    return base_h + loop_h * math.sin(math.pi * clamp01(u))


def large_base_top_height(p: Params) -> float:
    return max(0.0, float(p.large_base_height))


def radius_profile(t: float, p: Params) -> float:
    """
    Swept-tube radius with:
    - base_tube_radius as the nominal body tube radius
    - neck pinch around neck_center
    - a larger base collar at t=0 that fillets into the thinner tube
    - an optional extra outer-edge roundover on the large base

    The outer-edge roundover is implemented by slightly reducing the very first radius and
    smoothly letting it grow to the full large_base_radius over a vertical span equal to
    base_tube_radius, so the span and base tube radius stay coincident.
    """
    t = clamp01(t)
    r_body = body_radius_profile(t, p)

    h = vertical_height_at_t(t, p)
    collar_top_h = large_base_top_height(p)
    fillet_h = max(0.0, float(p.base_fillet_span)) * max(1e-9, float(p.base_tube_height) + float(p.spine_height))
    if h <= collar_top_h + 1e-9:
        collar_w = 1.0
    elif fillet_h > 1e-9:
        s = clamp01((h - collar_top_h) / fillet_h)
        collar_w = 1.0 - smootherstep(s)
    else:
        collar_w = 0.0

    r = r_body * (1.0 - collar_w) + float(p.large_base_radius) * collar_w

    edge_span_h = max(0.0, float(p.base_tube_radius))
    if float(p.base_outer_edge_fillet) > 1e-9 and edge_span_h > 1e-9:
        s2 = clamp01(vertical_height_at_t(t, p) / edge_span_h)
        edge_w = 1.0 - smootherstep(s2)
        r -= float(p.base_outer_edge_fillet) * edge_w

    return max(0.1, r)


def build_surface_cache(p: Params, samples: int) -> SurfaceCache:
    n = int(max(32, samples))
    ts = np.linspace(0.0, 1.0, n)
    base_xyz = np.array([float(p.start_x), float(p.start_y), float(p.start_z)], dtype=float)
    spine = np.stack(
        [
            kleinish_spine(
                t=float(t),
                base_xyz=base_xyz,
                base_tube_height=float(p.base_tube_height),
                height=float(p.spine_height),
                r0=float(p.spine_r0),
                r1=float(p.spine_r1),
                s_cross=float(p.spine_s),
                z_wobble=float(p.spine_z_wobble),
            )
            for t in ts
        ],
        axis=0,
    )
    spine_s_mm = np.zeros(n, dtype=float)
    if n > 1:
        spine_s_mm[1:] = np.cumsum(np.linalg.norm(np.diff(spine, axis=0), axis=1))
    T, N, B = build_rmf_frames(spine)
    Nt, Bt = apply_twist_about_tangent(T, N, B, float(p.twist_deg))
    return SurfaceCache(ts=ts, spine=spine, spine_s_mm=spine_s_mm, T=T, N=N, B=B, Nt=Nt, Bt=Bt)


def _interp_frame(cache: SurfaceCache, arr: np.ndarray, t: float) -> np.ndarray:
    t = clamp01(t)
    idx = np.searchsorted(cache.ts, t)
    if idx <= 0:
        return arr[0].copy()
    if idx >= len(cache.ts):
        return arr[-1].copy()
    t0 = float(cache.ts[idx - 1])
    t1 = float(cache.ts[idx])
    if t1 <= t0 + 1e-15:
        return arr[idx].copy()
    a = (t - t0) / (t1 - t0)
    v = lerp(arr[idx - 1], arr[idx], a)
    return _normalize(v)


def spine_point(cache: SurfaceCache, t: float) -> np.ndarray:
    t = clamp01(t)
    idx = np.searchsorted(cache.ts, t)
    if idx <= 0:
        return cache.spine[0].copy()
    if idx >= len(cache.ts):
        return cache.spine[-1].copy()
    t0 = float(cache.ts[idx - 1])
    t1 = float(cache.ts[idx])
    if t1 <= t0 + 1e-15:
        return cache.spine[idx].copy()
    a = (t - t0) / (t1 - t0)
    return lerp(cache.spine[idx - 1], cache.spine[idx], a)


def tangent_dir(cache: SurfaceCache, t: float) -> np.ndarray:
    return _interp_frame(cache, cache.T, t)


def normal_frame(cache: SurfaceCache, t: float) -> Tuple[np.ndarray, np.ndarray]:
    n = _interp_frame(cache, cache.Nt, t)
    b = _interp_frame(cache, cache.Bt, t)
    # re-orthogonalize gently
    tt = tangent_dir(cache, t)
    n = _normalize(n - np.dot(n, tt) * tt)
    b = _normalize(np.cross(tt, n))
    return n, b


def spine_arclength_mm(cache: SurfaceCache, t: float) -> float:
    t = clamp01(t)
    idx = np.searchsorted(cache.ts, t)
    if idx <= 0:
        return float(cache.spine_s_mm[0])
    if idx >= len(cache.ts):
        return float(cache.spine_s_mm[-1])
    t0 = float(cache.ts[idx - 1])
    t1 = float(cache.ts[idx])
    if t1 <= t0 + 1e-15:
        return float(cache.spine_s_mm[idx])
    a = (t - t0) / (t1 - t0)
    return float((1.0 - a) * cache.spine_s_mm[idx - 1] + a * cache.spine_s_mm[idx])


def shell_phi_for_t(cache: SurfaceCache, t: float, p: Params) -> float:
    pitch = max(0.05, float(p.continuous_layer_height))
    turns = spine_arclength_mm(cache, t) / pitch
    return math.radians(float(p.shell_phase_deg)) + 2.0 * math.pi * turns


def point_on_surface(cache: SurfaceCache, p: Params, t: float, phi: float) -> np.ndarray:
    c = spine_point(cache, t)
    n, b = normal_frame(cache, t)
    r = radius_profile(t, p)
    return c + r * (math.cos(phi) * n + math.sin(phi) * b)


def radial_direction(cache: SurfaceCache, t: float, phi: float) -> np.ndarray:
    n, b = normal_frame(cache, t)
    return _normalize(math.cos(phi) * n + math.sin(phi) * b)


def surface_normal(cache: SurfaceCache, p: Params, t: float, phi: float, dt: float = 0.001, dphi: float = 0.01) -> np.ndarray:
    t0 = clamp01(t - dt)
    t1 = clamp01(t + dt)
    if abs(t1 - t0) < 1e-9:
        t0 = clamp01(t - 2.0 * dt)
        t1 = clamp01(t + 2.0 * dt)

    s_t0 = point_on_surface(cache, p, t0, phi)
    s_t1 = point_on_surface(cache, p, t1, phi)
    s_p0 = point_on_surface(cache, p, t, phi - dphi)
    s_p1 = point_on_surface(cache, p, t, phi + dphi)

    d_t = s_t1 - s_t0
    d_p = s_p1 - s_p0
    nrm = _normalize(np.cross(d_t, d_p))

    radial = radial_direction(cache, t, phi)
    if float(np.dot(nrm, radial)) < 0.0:
        nrm = -nrm
    return nrm


def direction_to_tilt_azimuth_deg(d: np.ndarray, prev_c_deg: float) -> Tuple[float, float]:
    d = _normalize(d)
    dz = float(np.clip(d[2], -1.0, 1.0))
    tilt = math.degrees(math.acos(dz))
    xy = math.hypot(float(d[0]), float(d[1]))
    if xy < 1e-8:
        return tilt, float(prev_c_deg)
    az = math.degrees(math.atan2(float(d[1]), float(d[0])))
    return tilt, az


def polyline_length_xyz(xyzs: Sequence[np.ndarray]) -> float:
    if len(xyzs) < 2:
        return 0.0
    total = 0.0
    for a, b in zip(xyzs[:-1], xyzs[1:]):
        total += float(np.linalg.norm(np.asarray(b) - np.asarray(a)))
    return total


# ============================================================================
# Parametric toolpath construction
# ============================================================================

def make_base_ring_polyline(p: Params) -> ParamPolyline:
    m = int(max(8, p.base_ring_segments))
    pts = [
        ParamPoint(t=0.0, phi=2.0 * math.pi * (k / m), mode="base_ring", comment=f"base ring {k}/{m}")
        for k in range(m + 1)
    ]
    return ParamPolyline(name="base_ring", points=pts)


def make_shell_polyline(
    cache: SurfaceCache,
    p: Params,
    t0: float,
    t1: float,
    steps: int,
    name: str,
    *,
    phase_offset: float = 0.0,
    track_sign: float = 1.0,
) -> ParamPolyline:
    t0 = clamp01(t0)
    t1 = clamp01(t1)
    n = int(max(2, steps))
    ts = np.linspace(t0, t1, n)
    pts = [
        ParamPoint(
            t=float(t),
            phi=shell_phi_for_t(cache, float(t), p) + float(phase_offset),
            mode="surface",
            comment=f"{name} t={t:.4f}",
            track_sign=float(track_sign),
        )
        for t in ts
    ]
    return ParamPolyline(name=name, points=pts)


def make_param_edge(t0: float, phi0: float, t1: float, phi1: float, samples: int, name: str) -> ParamPolyline:
    n = int(max(2, samples))
    pts: List[ParamPoint] = []
    for i in range(n):
        a = 0.0 if n == 1 else (i / (n - 1))
        t = (1.0 - a) * float(t0) + a * float(t1)
        phi = (1.0 - a) * float(phi0) + a * float(phi1)
        pts.append(ParamPoint(t=t, phi=phi, mode="surface", comment=f"{name} a={a:.3f}"))
    return ParamPolyline(name=name, points=pts)


def make_base_transition_polyline(phi0: float, phi1: float, samples: int, name: str) -> ParamPolyline:
    n = int(max(3, samples))
    pts: List[ParamPoint] = []
    for i in range(n):
        a = 0.0 if n == 1 else (i / (n - 1))
        phi = (1.0 - a) * float(phi0) + a * float(phi1)
        pts.append(
            ParamPoint(
                t=0.0,
                phi=phi,
                mode="surface",
                comment=f"{name} a={a:.3f}",
                track_mode="bottom_turn",
                track_u=float(a),
            )
        )
    return ParamPolyline(name=name, points=pts)


def build_lattice_polylines(p: Params) -> List[ParamPolyline]:
    t0 = clamp01(min(float(p.lattice_start), float(p.lattice_end)))
    t1 = clamp01(max(float(p.lattice_start), float(p.lattice_end)))
    if t1 <= t0 + 1e-6:
        return []

    nu = int(max(2, p.lattice_u_count))
    nv = int(max(3, p.lattice_v_count))
    edge_samples = int(max(2, p.lattice_edge_samples))

    ts = np.linspace(t0, t1, nu)
    phis = np.linspace(0.0, 2.0 * math.pi, nv, endpoint=False)

    polylines: List[ParamPolyline] = []

    if int(p.lattice_include_rings):
        for i, t in enumerate(ts):
            pts = [ParamPoint(t=float(t), phi=float(phi), mode="surface", comment=f"lattice ring {i}") for phi in phis]
            pts.append(ParamPoint(t=float(t), phi=float(phis[0] + 2.0 * math.pi), mode="surface", comment=f"lattice ring {i} close"))
            polylines.append(ParamPolyline(name=f"lattice_ring_{i}", points=pts))

    if int(p.lattice_include_meridians):
        for j, phi in enumerate(phis):
            pts = [ParamPoint(t=float(t), phi=float(phi), mode="surface", comment=f"lattice meridian {j}") for t in ts]
            polylines.append(ParamPolyline(name=f"lattice_meridian_{j}", points=pts))

    if int(p.lattice_include_diagonals):
        # Alternating diagonals to give a low-poly diamond / octahedral skin feel.
        for i in range(nu - 1):
            for j in range(nv):
                jn = (j + 1) % nv
                if (i + j) % 2 == 0:
                    polylines.append(
                        make_param_edge(
                            float(ts[i]), float(phis[j]),
                            float(ts[i + 1]), float(phis[jn]),
                            samples=edge_samples,
                            name=f"diag_a_{i}_{j}",
                        )
                    )
                else:
                    polylines.append(
                        make_param_edge(
                            float(ts[i]), float(phis[jn]),
                            float(ts[i + 1]), float(phis[j]),
                            samples=edge_samples,
                            name=f"diag_b_{i}_{j}",
                        )
                    )
    return polylines


def build_param_polylines(cache: SurfaceCache, p: Params) -> List[ParamPolyline]:
    lat0 = clamp01(min(float(p.lattice_start), float(p.lattice_end)))
    lat1 = clamp01(max(float(p.lattice_start), float(p.lattice_end)))
    base_top_t = base_tube_fraction(p)
    base_phase_shift = math.pi

    steps_total = int(max(100, p.path_steps))
    shell_down_steps = max(2, int(round(steps_total * max(0.0, base_top_t))))
    shell_1_steps = max(2, int(round(steps_total * max(0.0, lat0))))
    shell_2_steps = max(2, int(round(steps_total * max(0.0, 1.0 - lat1))))
    base_turn_steps = max(12, int(round(0.5 * max(16, p.base_ring_segments))))

    polylines: List[ParamPolyline] = []

    if base_top_t > 1e-6:
        polylines.append(
            make_shell_polyline(
                cache,
                p,
                base_top_t,
                0.0,
                shell_down_steps,
                "shell_base_descent",
                track_sign=-1.0,
            )
        )

    phi0 = shell_phi_for_t(cache, 0.0, p)
    polylines.append(make_base_transition_polyline(phi0, phi0 + base_phase_shift, base_turn_steps, "shell_base_turn"))

    if lat0 > 1e-6:
        polylines.append(
            make_shell_polyline(
                cache,
                p,
                0.0,
                lat0,
                shell_1_steps,
                "shell_pre_lattice",
                phase_offset=base_phase_shift,
                track_sign=1.0,
            )
        )
    else:
        polylines.append(
            make_shell_polyline(
                cache,
                p,
                0.0,
                min(0.05, 1.0),
                max(2, shell_1_steps),
                "shell_pre_lattice",
                phase_offset=base_phase_shift,
                track_sign=1.0,
            )
        )

    polylines.extend(build_lattice_polylines(p))

    if lat1 < 1.0 - 1e-6:
        polylines.append(
            make_shell_polyline(
                cache,
                p,
                lat1,
                1.0,
                shell_2_steps,
                "shell_post_lattice",
                phase_offset=base_phase_shift,
                track_sign=1.0,
            )
        )

    return polylines


def param_point_to_xyz(cache: SurfaceCache, p: Params, pt: ParamPoint) -> np.ndarray:
    if pt.mode == "base_ring":
        t = 0.0
        c = spine_point(cache, t)
        n, b = normal_frame(cache, t)
        r = radius_profile(0.0, p)
        return c + r * (math.cos(pt.phi) * n + math.sin(pt.phi) * b)
    if pt.mode == "surface":
        return point_on_surface(cache, p, pt.t, pt.phi)
    raise ValueError(f"Unknown point mode: {pt.mode}")


def param_point_to_direction(cache: SurfaceCache, p: Params, pt: ParamPoint) -> np.ndarray:
    tool_sign = 1.0 if int(p.tool_normal_sign) >= 0 else -1.0
    if pt.mode in {"base_ring", "surface"}:
        centerline_t = tangent_dir(cache, pt.t)
        if pt.track_mode == "bottom_turn":
            n, b = normal_frame(cache, pt.t)
            ring_tangent = _normalize(-math.sin(pt.phi) * n + math.cos(pt.phi) * b)
            a = clamp01(pt.track_u)
            d = -math.cos(math.pi * a) * centerline_t + math.sin(math.pi * a) * ring_tangent
            return _normalize(tool_sign * d)
        return _normalize(tool_sign * float(pt.track_sign) * centerline_t)
    raise ValueError(f"Unknown point mode: {pt.mode}")


def polyline_to_xyz(cache: SurfaceCache, p: Params, poly: ParamPolyline) -> List[np.ndarray]:
    return [param_point_to_xyz(cache, p, pt) for pt in poly.points]


def polyline_to_waypoints(cache: SurfaceCache, p: Params, poly: ParamPolyline, inverter: TipAngleInverter, c_start_deg: float) -> List[Waypoint]:
    waypoints: List[Waypoint] = []
    prev_c = float(c_start_deg)
    for pt in poly.points:
        xyz = param_point_to_xyz(cache, p, pt)
        direction = param_point_to_direction(cache, p, pt)
        tilt, az = direction_to_tilt_azimuth_deg(direction, prev_c)
        az = unwrap_degrees(prev_c, az)
        prev_c = az
        b_cmd, clamped = inverter.angle_to_b(tilt)
        comment = pt.comment + (" (B clamped)" if clamped else "")
        waypoints.append(Waypoint(tip_xyz=xyz, b_cmd=b_cmd, c_deg=az, comment=comment))
    return waypoints


def total_print_length(cache: SurfaceCache, p: Params, polys: Sequence[ParamPolyline]) -> float:
    total = 0.0
    for poly in polys:
        total += polyline_length_xyz(polyline_to_xyz(cache, p, poly))
    return total


# ============================================================================
# G-code writer
# ============================================================================

def _fmt_axes_move(axes_vals: List[Tuple[str, float]]) -> str:
    return " ".join(f"{ax}{val:.3f}" for ax, val in axes_vals)


def _clamp_stage_xyz_to_bbox(x: float, y: float, z: float, bbox: Dict[str, float], context: str, warn_log: List[str]) -> Tuple[float, float, float]:
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
        bbox: Dict[str, float],
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
        x, y, z = _clamp_stage_xyz_to_bbox(p_stage[0], p_stage[1], p_stage[2], self.bbox, context, self.warn_log)
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
        if self.emit_extrusion and not self.pressure_charged:
            self.pressure_charged = True
            self.f.write("; pressure preload\n")
            self.f.write("M42 P0 S1\n")
            if self.preflow_dwell_ms > 0:
                self.f.write(f"G4 P{self.preflow_dwell_ms}\n")

    def pressure_release(self):
        if self.emit_extrusion and self.pressure_charged:
            if self.end_dwell_ms > 0:
                self.f.write("; end dwell\n")
                self.f.write(f"G4 P{self.end_dwell_ms}\n")
            self.pressure_charged = False
            self.f.write("; pressure release\n")
            self.f.write("M42 P0 S0\n")

    def print_waypoint(self, wp: Waypoint, feed: float, do_extrude: bool, prev_tip_xyz: Optional[np.ndarray]):
        tip_xyz = np.asarray(wp.tip_xyz, dtype=float)
        stage_xyz = stage_xyz_for_tip(self.cal, tip_xyz, wp.b_cmd, wp.c_deg)

        off = tip_offset_xyz_physical(self.cal, wp.b_cmd, wp.c_deg)
        if np.linalg.norm((stage_xyz + off) - tip_xyz) > 1e-7:
            raise RuntimeError("Tip tracking consistency failed.")

        pc = self._clamp_stage(stage_xyz, wp.comment or "print_waypoint")
        axes = [
            (self.cal.x_axis, pc[0]),
            (self.cal.y_axis, pc[1]),
            (self.cal.z_axis, pc[2]),
            (self.cal.b_axis, wp.b_cmd),
            (self.cal.c_axis, wp.c_deg),
        ]

        if self.emit_extrusion and do_extrude:
            seg_len = 0.0 if prev_tip_xyz is None else float(np.linalg.norm(tip_xyz - prev_tip_xyz))
            self.u_material_abs += self.extrusion_per_mm * seg_len

        if wp.comment and self.debug_every > 0 and (self.step_counter % self.debug_every == 0):
            tip_ang = float(eval_tip_angle_deg(self.cal, wp.b_cmd))
            self.f.write(f"; step={self.step_counter} {wp.comment} | Bcmd={wp.b_cmd:.4f} C={wp.c_deg:.2f} tipAngle={tip_ang:.2f}\n")

        self.f.write(f"G1 {_fmt_axes_move(axes)} F{float(feed):.0f}\n")
        self.cur_stage_xyz = pc.copy()
        self.cur_tip_xyz = tip_xyz.copy()
        self.cur_b = float(wp.b_cmd)
        self.cur_c = float(wp.c_deg)
        self.step_counter += 1

    def print_polyline(self, poly_name: str, wps: Sequence[Waypoint]):
        if not wps:
            return
        self.f.write(f"; --- polyline start: {poly_name} ---\n")
        first = wps[0]
        self.move_to_tip(first.tip_xyz, first.b_cmd, first.c_deg, feed=self.travel_feed, comment=f"travel to {poly_name} start")
        self.pressure_preload()
        prev_tip = np.asarray(first.tip_xyz, dtype=float)
        for wp in wps[1:]:
            self.print_waypoint(wp, feed=self.print_feed, do_extrude=True, prev_tip_xyz=prev_tip)
            prev_tip = np.asarray(wp.tip_xyz, dtype=float)
        self.pressure_release()
        self.f.write(f"; --- polyline end: {poly_name} ---\n")


# ============================================================================
# Export
# ============================================================================

def sanitize_params(p: Params) -> Params:
    q = Params(**asdict(p))
    q.base_ring_segments = int(max(8, q.base_ring_segments))
    q.path_steps = int(max(100, q.path_steps))
    q.preview_spine_samples = int(max(32, q.preview_spine_samples))
    q.preview_wireframe_u = int(max(8, q.preview_wireframe_u))
    q.preview_wireframe_v = int(max(6, q.preview_wireframe_v))
    q.lattice_edge_samples = int(max(2, q.lattice_edge_samples))
    q.lattice_u_count = int(max(2, q.lattice_u_count))
    q.lattice_v_count = int(max(3, q.lattice_v_count))
    q.tool_normal_sign = 1 if int(q.tool_normal_sign) >= 0 else -1
    q.lattice_start = clamp01(q.lattice_start)
    q.lattice_end = clamp01(q.lattice_end)
    if q.lattice_end < q.lattice_start:
        q.lattice_start, q.lattice_end = q.lattice_end, q.lattice_start
    q.neck_center = clamp01(q.neck_center)
    q.neck_width = max(1e-6, float(q.neck_width))
    q.base_fillet_span = max(0.0, float(q.base_fillet_span))
    q.base_outer_edge_fillet = max(0.0, float(q.base_outer_edge_fillet))
    q.large_base_height = max(0.0, float(q.large_base_height))
    q.base_tube_radius = max(0.1, float(q.base_tube_radius))
    q.base_tube_height = max(0.0, float(q.base_tube_height))
    q.large_base_radius = max(0.1, float(q.large_base_radius))
    q.neck_radius = max(0.1, float(q.neck_radius))
    q.continuous_layer_height = max(0.05, float(q.continuous_layer_height))
    return q


def params_from_mapping(data: Dict[str, Any], base: Optional[Params] = None) -> Params:
    merged = asdict(base or Params())
    allowed = {f.name for f in fields(Params)}
    for key, value in data.items():
        if key in allowed:
            merged[key] = value

    if "continuous_layer_height" not in data and "shell_turns" in data:
        shell_turns = max(1e-6, float(data["shell_turns"]))
        merged["continuous_layer_height"] = max(0.05, float(merged["spine_height"]) / shell_turns)

    return sanitize_params(Params(**merged))


def params_bbox(p: Params) -> Dict[str, float]:
    return {
        "x_min": float(min(p.bbox_x_min, p.bbox_x_max)),
        "x_max": float(max(p.bbox_x_min, p.bbox_x_max)),
        "y_min": float(min(p.bbox_y_min, p.bbox_y_max)),
        "y_max": float(max(p.bbox_y_min, p.bbox_y_max)),
        "z_min": float(min(p.bbox_z_min, p.bbox_z_max)),
        "z_max": float(max(p.bbox_z_min, p.bbox_z_max)),
    }


def export_gcode(params: Params, out_path: str) -> Dict[str, Any]:
    p = sanitize_params(params)
    if not p.calibration_path:
        raise ValueError("Calibration JSON path is required for G-code export.")
    cal = load_calibration(p.calibration_path)
    inverter = TipAngleInverter(cal)
    cache = build_surface_cache(p, samples=int(max(300, p.path_steps // 4)))
    polylines = build_param_polylines(cache, p)
    bbox = params_bbox(p)
    emit_extrusion = float(p.extrusion_per_mm) != 0.0
    warn_log: List[str] = []

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        total_len = total_print_length(cache, p, polylines)
        f.write("; generated by klein_bottle_gui_slicer.py\n")
        f.write("; calibrated tip-position planning: stage = tip - offset_tip(B_cmd,C)\n")
        f.write("; continuous centroid-following tracking across the print path\n")
        f.write("; extrusion actuation: pressure solenoid valve via M42 P0 S1 / M42 P0 S0\n")
        f.write(f"; axes: X->{cal.x_axis}, Y->{cal.y_axis}, Z->{cal.z_axis}, B->{cal.b_axis}, C->{cal.c_axis}, U->{cal.u_axis}\n")
        f.write(f"; calibration B range (command units): [{cal.b_min:.6f}, {cal.b_max:.6f}]\n")
        f.write(f"; selected_offplane_fit_model={cal.selected_offplane_fit_model or cal.selected_fit_model or 'avg_cubic'} active_phase={cal.active_phase}\n")
        f.write(f"; start tip-space base = [{p.start_x:.3f}, {p.start_y:.3f}, {p.start_z:.3f}]\n")
        f.write(f"; spine: height={p.spine_height:.3f}, base_tube_height={p.base_tube_height:.3f}, r0={p.spine_r0:.3f}, r1={p.spine_r1:.3f}, s={p.spine_s:.3f}, z_wobble={p.spine_z_wobble:.3f}\n")
        f.write(f"; twist_deg={p.twist_deg:.3f}, continuous_layer_height={p.continuous_layer_height:.3f}, shell_phase_deg={p.shell_phase_deg:.3f}\n")
        f.write(f"; radii: large_base={p.large_base_radius:.3f}, large_base_height={p.large_base_height:.3f}, base_tube={p.base_tube_radius:.3f}, neck={p.neck_radius:.3f}\n")
        f.write(f"; neck: center={p.neck_center:.3f}, width={p.neck_width:.3f}, power={p.neck_power:.3f}\n")
        f.write(f"; base fillets: base_fillet_span={p.base_fillet_span:.3f}, outer_edge_fillet={p.base_outer_edge_fillet:.3f}, outer_edge_span=base_tube_radius({p.base_tube_radius:.3f})\n")
        f.write(f"; lattice window: start={p.lattice_start:.3f}, end={p.lattice_end:.3f}, u={p.lattice_u_count}, v={p.lattice_v_count}\n")
        f.write(f"; lengths: approx_total_print_length={total_len:.3f} mm, approx_total_extrusion={total_len * p.extrusion_per_mm:.3f}\n")
        f.write(f"; feeds: travel={p.travel_feed:.1f}, print={p.print_feed:.1f}\n")
        f.write(f"; tool_normal_sign={p.tool_normal_sign}\n")
        f.write("G90\n")
        if emit_extrusion:
            f.write("M42 P0 S0\n")

        g = CalibratedGCodeWriter(
            fh=f,
            cal=cal,
            bbox=bbox,
            travel_feed=float(p.travel_feed),
            print_feed=float(p.print_feed),
            extrusion_per_mm=float(p.extrusion_per_mm),
            pressure_offset_mm=float(p.pressure_offset_mm),
            pressure_advance_feed=float(p.pressure_advance_feed),
            pressure_retract_feed=float(p.pressure_retract_feed),
            preflow_dwell_ms=int(p.preflow_dwell_ms),
            end_dwell_ms=int(p.end_dwell_ms),
            emit_extrusion=emit_extrusion,
            warn_log=warn_log,
            debug_every=int(p.debug_every),
        )

        g.move_stage_xyzbc(
            np.array([float(p.machine_start_x), float(p.machine_start_y), float(p.safe_approach_z)], dtype=float),
            b_cmd=float(p.machine_start_b),
            c_deg=float(p.machine_start_c),
            feed=float(p.travel_feed),
            comment="startup: move to safe Z at machine-start XY",
        )
        g.move_stage_xyzbc(
            np.array([float(p.machine_start_x), float(p.machine_start_y), float(p.machine_start_z)], dtype=float),
            b_cmd=float(p.machine_start_b),
            c_deg=float(p.machine_start_c),
            feed=float(p.travel_feed),
            comment="startup: dive to machine-start Z",
        )

        c_seed = float(p.machine_start_c)
        total_waypoints = 0
        for poly in polylines:
            waypoints = polyline_to_waypoints(cache, p, poly, inverter, c_seed)
            if not waypoints:
                continue
            c_seed = float(waypoints[-1].c_deg)
            total_waypoints += len(waypoints)
            g.print_polyline(poly.name, waypoints)

        if g.cur_stage_xyz is not None:
            g.move_stage_xyzbc(
                np.array([float(g.cur_stage_xyz[0]), float(g.cur_stage_xyz[1]), float(p.safe_approach_z)], dtype=float),
                b_cmd=float(p.machine_end_b),
                c_deg=float(p.machine_end_c),
                feed=float(p.travel_feed),
                comment="end: raise to safe Z",
            )
        g.move_stage_xyzbc(
            np.array([float(p.machine_end_x), float(p.machine_end_y), float(p.safe_approach_z)], dtype=float),
            b_cmd=float(p.machine_end_b),
            c_deg=float(p.machine_end_c),
            feed=float(p.travel_feed),
            comment="end: move XY at safe Z",
        )
        g.move_stage_xyzbc(
            np.array([float(p.machine_end_x), float(p.machine_end_y), float(p.machine_end_z)], dtype=float),
            b_cmd=float(p.machine_end_b),
            c_deg=float(p.machine_end_c),
            feed=float(p.travel_feed),
            comment="end: dive to machine end Z",
        )

        if warn_log:
            f.write("; virtual bbox clamp warnings:\n")
            for msg in warn_log:
                f.write(f"; {msg}\n")

        f.write(f"; total polylines = {len(polylines)}\n")
        f.write(f"; total waypoints = {total_waypoints}\n")
        f.write(f"; bbox warning count = {len(warn_log)}\n")
        f.write("; --- end ---\n")

    return {
        "out_path": out_path,
        "warning_count": len(warn_log),
        "warnings": warn_log,
        "polyline_count": len(polylines),
        "approx_print_length_mm": total_print_length(cache, p, polylines),
    }


# ============================================================================
# GUI widgets
# ============================================================================

class ParameterControl(ttk.Frame):
    def __init__(self, master, spec: Dict[str, Any], initial_value: Any, on_change, set_active):
        super().__init__(master)
        self.spec = spec
        self.key = spec["key"]
        self.label_text = spec["label"]
        self.step = spec.get("step", spec.get("resolution", 1.0))
        self.on_change = on_change
        self.set_active = set_active
        self.is_int = int(spec.get("resolution", 1)) == 1 and isinstance(initial_value, int)

        self.columnconfigure(2, weight=1)

        self.lbl = ttk.Label(self, text=self.label_text, width=20)
        self.lbl.grid(row=0, column=0, sticky="w", padx=(0, 4))

        self.btn_minus = ttk.Button(self, text="◀", width=3, command=lambda: self.bump(-1))
        self.btn_minus.grid(row=0, column=1, padx=2)

        if self.is_int:
            self.var = tk.IntVar(value=int(initial_value))
        else:
            self.var = tk.DoubleVar(value=float(initial_value))

        self.scale = tk.Scale(
            self,
            from_=spec["from_"],
            to=spec["to"],
            resolution=spec["resolution"],
            orient="horizontal",
            variable=self.var,
            showvalue=False,
            command=self._scale_changed,
            length=240,
            bg=UI_PANEL_BG,
            fg=UI_FG,
            troughcolor=UI_PANEL_ALT_BG,
            highlightthickness=0,
            activebackground=UI_ACCENT,
        )
        self.scale.grid(row=0, column=2, sticky="ew", padx=2)
        self.scale.bind("<Button-1>", lambda e: self.set_active(self))
        self.scale.bind("<FocusIn>", lambda e: self.set_active(self))

        self.btn_plus = ttk.Button(self, text="▶", width=3, command=lambda: self.bump(+1))
        self.btn_plus.grid(row=0, column=3, padx=2)

        self.entry = ttk.Entry(self, width=10)
        self.entry.grid(row=0, column=4, padx=(4, 0))
        self.entry.insert(0, self._fmt_value(self.var.get()))
        self.entry.bind("<Return>", self._entry_commit)
        self.entry.bind("<FocusIn>", lambda e: self.set_active(self))
        self.entry.bind("<FocusOut>", self._entry_commit)

    def _fmt_value(self, v: Any) -> str:
        if self.is_int:
            return str(int(round(float(v))))
        return f"{float(v):.4f}".rstrip("0").rstrip(".")

    def _scale_changed(self, _event=None):
        self.entry.delete(0, tk.END)
        self.entry.insert(0, self._fmt_value(self.var.get()))
        self.on_change(self.key, self.get_value())

    def _entry_commit(self, _event=None):
        try:
            if self.is_int:
                value = int(round(float(self.entry.get().strip())))
            else:
                value = float(self.entry.get().strip())
        except Exception:
            self.entry.delete(0, tk.END)
            self.entry.insert(0, self._fmt_value(self.var.get()))
            return
        lo = float(self.spec["from_"])
        hi = float(self.spec["to"])
        value = max(lo, min(hi, value))
        if self.is_int:
            self.var.set(int(round(value)))
        else:
            self.var.set(float(value))
        self._scale_changed()

    def bump(self, direction: int):
        self.set_active(self)
        value = float(self.var.get()) + direction * float(self.step)
        lo = float(self.spec["from_"])
        hi = float(self.spec["to"])
        value = max(lo, min(hi, value))
        if self.is_int:
            self.var.set(int(round(value)))
        else:
            self.var.set(float(value))
        self._scale_changed()

    def get_value(self):
        return int(self.var.get()) if self.is_int else float(self.var.get())


class ScrollableFrame(ttk.Frame):
    def __init__(self, master, *args, **kwargs):
        super().__init__(master, *args, **kwargs)
        canvas = tk.Canvas(self, borderwidth=0, highlightthickness=0, bg=UI_PANEL_BG)
        vsb = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.inner = ttk.Frame(canvas)
        self.inner.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=self.inner, anchor="nw")
        canvas.configure(yscrollcommand=vsb.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=1)
        self.canvas = canvas
        self.vsb = vsb


# ============================================================================
# Main GUI app
# ============================================================================

class KleinBottleApp:
    def __init__(self, root: tk.Tk, initial_params: Optional[Params] = None):
        self.root = root
        self.root.title("Klein Bottle Shell/Lattice G-code Tool")
        self._configure_theme()
        self.params = sanitize_params(initial_params or Params())
        self.controls: Dict[str, ParameterControl] = {}
        self.active_control: Optional[ParameterControl] = None
        self._pending_redraw = False

        self.calibration_path_var = tk.StringVar(value=self.params.calibration_path)
        self.output_path_var = tk.StringVar(value=self.params.output_path)
        self.status_var = tk.StringVar(value="Ready")
        self.show_wireframe_var = tk.IntVar(value=int(self.params.show_wireframe))
        self.show_normals_var = tk.IntVar(value=int(self.params.show_normals))
        self.lattice_rings_var = tk.IntVar(value=int(self.params.lattice_include_rings))
        self.lattice_meridians_var = tk.IntVar(value=int(self.params.lattice_include_meridians))
        self.lattice_diagonals_var = tk.IntVar(value=int(self.params.lattice_include_diagonals))

        self._build_ui()
        self.root.bind("<Left>", self._arrow_left)
        self.root.bind("<Right>", self._arrow_right)
        self.root.bind("r", lambda e: self.schedule_redraw())
        self.schedule_redraw()

    # ---------- UI ----------

    def _configure_theme(self):
        self.root.configure(bg=UI_BG)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(".", background=UI_BG, foreground=UI_FG)
        style.configure("TFrame", background=UI_BG)
        style.configure("TLabelframe", background=UI_PANEL_BG, foreground=UI_FG, bordercolor=UI_BORDER)
        style.configure("TLabelframe.Label", background=UI_PANEL_BG, foreground=UI_FG)
        style.configure("TLabel", background=UI_BG, foreground=UI_FG)
        style.configure("TButton", background=UI_PANEL_ALT_BG, foreground=UI_FG, bordercolor=UI_BORDER, focusthickness=1, focuscolor=UI_ACCENT)
        style.map("TButton", background=[("active", UI_ACCENT)], foreground=[("active", UI_BG)])
        style.configure("TCheckbutton", background=UI_PANEL_BG, foreground=UI_FG)
        style.map("TCheckbutton", background=[("active", UI_PANEL_BG)], foreground=[("active", UI_FG)])
        style.configure("TEntry", fieldbackground=UI_ENTRY_BG, foreground=UI_FG, bordercolor=UI_BORDER, insertcolor=UI_FG)
        style.configure("Vertical.TScrollbar", background=UI_PANEL_ALT_BG, troughcolor=UI_PANEL_BG, bordercolor=UI_BORDER, arrowcolor=UI_FG)

    def _build_ui(self):
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root, padding=8)
        left.grid(row=0, column=0, sticky="nsw")
        right = ttk.Frame(self.root, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.columnconfigure(0, weight=1)
        right.rowconfigure(1, weight=1)

        # Top controls on left
        path_frame = ttk.LabelFrame(left, text="Files", padding=6)
        path_frame.grid(row=0, column=0, sticky="ew")
        path_frame.columnconfigure(1, weight=1)

        ttk.Label(path_frame, text="Calibration JSON").grid(row=0, column=0, sticky="w")
        ttk.Entry(path_frame, textvariable=self.calibration_path_var, width=36).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(path_frame, text="Browse", command=self.browse_calibration).grid(row=0, column=2)

        ttk.Label(path_frame, text="Output G-code").grid(row=1, column=0, sticky="w", pady=(4, 0))
        ttk.Entry(path_frame, textvariable=self.output_path_var, width=36).grid(row=1, column=1, sticky="ew", padx=4, pady=(4, 0))
        ttk.Button(path_frame, text="Browse", command=self.browse_output).grid(row=1, column=2, pady=(4, 0))

        btns = ttk.Frame(left)
        btns.grid(row=1, column=0, sticky="ew", pady=(6, 6))
        ttk.Button(btns, text="Export G-code", command=self.export_current).grid(row=0, column=0, padx=2)
        ttk.Button(btns, text="Save Preset", command=self.save_preset).grid(row=0, column=1, padx=2)
        ttk.Button(btns, text="Load Preset", command=self.load_preset).grid(row=0, column=2, padx=2)
        ttk.Button(btns, text="Reset", command=self.reset_defaults).grid(row=0, column=3, padx=2)

        toggles = ttk.LabelFrame(left, text="Preview", padding=6)
        toggles.grid(row=2, column=0, sticky="ew")
        ttk.Checkbutton(toggles, text="Show wireframe", variable=self.show_wireframe_var, command=self._toggle_changed).grid(row=0, column=0, sticky="w")
        ttk.Checkbutton(toggles, text="Show normals", variable=self.show_normals_var, command=self._toggle_changed).grid(row=1, column=0, sticky="w")
        ttk.Checkbutton(toggles, text="Lattice rings", variable=self.lattice_rings_var, command=self._toggle_changed).grid(row=0, column=1, sticky="w")
        ttk.Checkbutton(toggles, text="Lattice meridians", variable=self.lattice_meridians_var, command=self._toggle_changed).grid(row=1, column=1, sticky="w")
        ttk.Checkbutton(toggles, text="Lattice diagonals", variable=self.lattice_diagonals_var, command=self._toggle_changed).grid(row=2, column=1, sticky="w")

        self.scroll = ScrollableFrame(left)
        self.scroll.grid(row=3, column=0, sticky="nsew", pady=(6, 0))
        left.rowconfigure(3, weight=1)

        inner = self.scroll.inner
        for group_name, specs in PARAM_GROUPS:
            lf = ttk.LabelFrame(inner, text=group_name, padding=6)
            lf.pack(fill="x", expand=True, pady=4)
            for spec in specs:
                key = spec["key"]
                initial = getattr(self.params, key)
                ctrl = ParameterControl(lf, spec, initial, on_change=self._param_changed, set_active=self._set_active_control)
                ctrl.pack(fill="x", expand=True, pady=2)
                self.controls[key] = ctrl

        info_frame = ttk.Frame(right)
        info_frame.grid(row=0, column=0, sticky="ew")
        info_frame.columnconfigure(0, weight=1)

        self.info_label = ttk.Label(info_frame, text="", justify="left")
        self.info_label.grid(row=0, column=0, sticky="w")

        self.fig = Figure(figsize=(8.0, 7.0), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().configure(bg=UI_BG, highlightthickness=0)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")

        status = ttk.Label(self.root, textvariable=self.status_var, anchor="w", relief="sunken")
        status.grid(row=1, column=0, columnspan=2, sticky="ew")

    # ---------- callbacks ----------

    def _set_active_control(self, ctrl: ParameterControl):
        self.active_control = ctrl

    def _param_changed(self, key: str, value: Any):
        setattr(self.params, key, value)
        self.schedule_redraw()

    def _toggle_changed(self):
        self.params.show_wireframe = int(self.show_wireframe_var.get())
        self.params.show_normals = int(self.show_normals_var.get())
        self.params.lattice_include_rings = int(self.lattice_rings_var.get())
        self.params.lattice_include_meridians = int(self.lattice_meridians_var.get())
        self.params.lattice_include_diagonals = int(self.lattice_diagonals_var.get())
        self.schedule_redraw()

    def _arrow_left(self, _event=None):
        if self.active_control is not None:
            self.active_control.bump(-1)

    def _arrow_right(self, _event=None):
        if self.active_control is not None:
            self.active_control.bump(+1)

    def browse_calibration(self):
        path = filedialog.askopenfilename(
            title="Select calibration JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*")],
        )
        if path:
            self.calibration_path_var.set(path)
            self.params.calibration_path = path
            self.schedule_redraw()

    def browse_output(self):
        path = filedialog.asksaveasfilename(
            title="Save G-code",
            defaultextension=".gcode",
            filetypes=[("G-code", "*.gcode"), ("All files", "*")],
            initialfile=os.path.basename(self.output_path_var.get() or DEFAULT_OUT),
        )
        if path:
            self.output_path_var.set(path)
            self.params.output_path = path

    def save_preset(self):
        path = filedialog.asksaveasfilename(
            title="Save preset JSON",
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*")],
        )
        if not path:
            return
        self.params.calibration_path = self.calibration_path_var.get().strip()
        self.params.output_path = self.output_path_var.get().strip()
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.params), f, indent=2)
        self.status_var.set(f"Saved preset: {path}")

    def load_preset(self):
        path = filedialog.askopenfilename(
            title="Load preset JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*")],
        )
        if not path:
            return
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        self.params = params_from_mapping(data, base=self.params)
        self._sync_controls_from_params()
        self.status_var.set(f"Loaded preset: {path}")
        self.schedule_redraw()

    def reset_defaults(self):
        self.params = Params()
        self._sync_controls_from_params()
        self.status_var.set("Reset to defaults")
        self.schedule_redraw()

    def _sync_controls_from_params(self):
        for key, ctrl in self.controls.items():
            val = getattr(self.params, key)
            ctrl.var.set(val)
            ctrl.entry.delete(0, tk.END)
            ctrl.entry.insert(0, ctrl._fmt_value(val))
        self.calibration_path_var.set(self.params.calibration_path)
        self.output_path_var.set(self.params.output_path)
        self.show_wireframe_var.set(int(self.params.show_wireframe))
        self.show_normals_var.set(int(self.params.show_normals))
        self.lattice_rings_var.set(int(self.params.lattice_include_rings))
        self.lattice_meridians_var.set(int(self.params.lattice_include_meridians))
        self.lattice_diagonals_var.set(int(self.params.lattice_include_diagonals))

    def schedule_redraw(self):
        if self._pending_redraw:
            return
        self._pending_redraw = True
        self.root.after(60, self.redraw)

    # ---------- redraw ----------

    def redraw(self):
        self._pending_redraw = False
        try:
            self.params.calibration_path = self.calibration_path_var.get().strip()
            self.params.output_path = self.output_path_var.get().strip()
            self.params.show_wireframe = int(self.show_wireframe_var.get())
            self.params.show_normals = int(self.show_normals_var.get())
            self.params.lattice_include_rings = int(self.lattice_rings_var.get())
            self.params.lattice_include_meridians = int(self.lattice_meridians_var.get())
            self.params.lattice_include_diagonals = int(self.lattice_diagonals_var.get())
            self.params = sanitize_params(self.params)

            cache = build_surface_cache(self.params, self.params.preview_spine_samples)
            polylines = build_param_polylines(cache, self.params)
            self._draw_scene(cache, polylines)
            self._update_info(cache, polylines)
            self.status_var.set("Preview updated")
        except Exception as exc:
            self.status_var.set(f"Preview error: {exc}")
            self.ax.clear()
            self.ax.text2D(0.02, 0.98, f"Preview error:\n{exc}", transform=self.ax.transAxes, va="top")
            self.canvas.draw_idle()

    def _draw_scene(self, cache: SurfaceCache, polylines: Sequence[ParamPolyline]):
        ax = self.ax
        ax.clear()
        self.fig.patch.set_facecolor(UI_BG)
        ax.set_facecolor(UI_PLOT_BG)
        ax.set_xlabel("X", color=UI_FG)
        ax.set_ylabel("Y", color=UI_FG)
        ax.set_zlabel("Z", color=UI_FG)
        ax.set_title("Klein bottle shell / lattice preview", color=UI_FG)
        ax.tick_params(colors=UI_MUTED_FG)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.line.set_color(UI_MUTED_FG)
            axis.set_pane_color((*matplotlib.colors.to_rgb(UI_PLOT_BG), 1.0))
            axis._axinfo["grid"]["color"] = UI_GRID

        if int(self.params.show_wireframe):
            uu = int(max(8, self.params.preview_wireframe_u))
            vv = int(max(6, self.params.preview_wireframe_v))
            ts = np.linspace(0.0, 1.0, uu)
            phis = np.linspace(0.0, 2.0 * math.pi, vv, endpoint=False)
            X = np.zeros((uu, vv))
            Y = np.zeros((uu, vv))
            Z = np.zeros((uu, vv))
            for i, t in enumerate(ts):
                for j, phi in enumerate(phis):
                    xyz = point_on_surface(cache, self.params, float(t), float(phi))
                    X[i, j], Y[i, j], Z[i, j] = xyz
            ax.plot_wireframe(X, Y, Z, linewidth=0.45, alpha=0.25, color="#8fb8ff")

        ax.plot(cache.spine[:, 0], cache.spine[:, 1], cache.spine[:, 2], linewidth=1.0, alpha=0.7, color="#f59e0b")

        for poly in polylines:
            xyz = np.asarray(polyline_to_xyz(cache, self.params, poly), dtype=float)
            if xyz.shape[0] < 2:
                continue
            lw = 1.8 if poly.name.startswith("shell") else (2.2 if poly.name == "base_ring" else 0.9)
            color = "#4ade80" if poly.name.startswith("shell") else ("#f97316" if poly.name == "base_ring" else "#93c5fd")
            ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], linewidth=lw, color=color)

        if int(self.params.show_normals):
            stride = int(max(1, self.params.normal_stride))
            shells = [poly for poly in polylines if poly.name.startswith("shell") or poly.name == "base_ring"]
            for poly in shells:
                for pt in poly.points[::stride]:
                    pos = param_point_to_xyz(cache, self.params, pt)
                    nrm = param_point_to_direction(cache, self.params, pt)
                    ax.quiver(pos[0], pos[1], pos[2], nrm[0], nrm[1], nrm[2], length=8.0, normalize=True, color="#f43f5e")

        self._set_equal_axes_from_polys(cache, polylines)
        self.canvas.draw_idle()

    def _set_equal_axes_from_polys(self, cache: SurfaceCache, polylines: Sequence[ParamPolyline]):
        pts: List[np.ndarray] = [cache.spine]
        for poly in polylines:
            xyz = np.array(polyline_to_xyz(cache, self.params, poly))
            if xyz.size > 0:
                pts.append(xyz)
        all_pts = np.concatenate(pts, axis=0)
        mins = np.min(all_pts, axis=0)
        maxs = np.max(all_pts, axis=0)
        center = 0.5 * (mins + maxs)
        extent = np.max(maxs - mins)
        if extent < 1e-6:
            extent = 1.0
        half = 0.55 * extent
        self.ax.set_xlim(center[0] - half, center[0] + half)
        self.ax.set_ylim(center[1] - half, center[1] + half)
        self.ax.set_zlim(center[2] - half, center[2] + half)

    def _update_info(self, cache: SurfaceCache, polylines: Sequence[ParamPolyline]):
        total_len = total_print_length(cache, self.params, polylines)
        lat0 = min(self.params.lattice_start, self.params.lattice_end)
        lat1 = max(self.params.lattice_start, self.params.lattice_end)
        base_r = radius_profile(0.0, self.params)
        neck_r = radius_profile(self.params.neck_center, self.params)
        end_r = radius_profile(1.0, self.params)

        cal_text = "Calibration: not loaded"
        if self.params.calibration_path:
            try:
                cal = load_calibration(self.params.calibration_path)
                inv = TipAngleInverter(cal)
                sample_pts = []
                for t in [0.0, max(0.0, lat0 * 0.5), min(1.0, 0.5 * (lat0 + lat1)), min(1.0, 0.5 * (lat1 + 1.0)), 1.0]:
                    d = _normalize(float(self.params.tool_normal_sign) * tangent_dir(cache, t))
                    tilt, _ = direction_to_tilt_azimuth_deg(d, 0.0)
                    b_cmd, clamped = inv.angle_to_b(tilt)
                    tag = "*" if clamped else ""
                    sample_pts.append(f"t={t:.2f}: track={tilt:.1f}° -> B={b_cmd:.3f}{tag}")
                fit_name = cal.selected_offplane_fit_model or cal.selected_fit_model or "legacy"
                cal_text = f"Calibration: loaded | offplane fit={fit_name} | " + " | ".join(sample_pts)
            except Exception as exc:
                cal_text = f"Calibration preview unavailable: {exc}"

        txt = (
            f"Approx print length: {total_len:.1f} mm\n"
            f"Approx extrusion: {total_len * self.params.extrusion_per_mm:.2f} mm\n"
            f"Continuous layer height: {self.params.continuous_layer_height:.2f} mm | Base tube height: {self.params.base_tube_height:.2f} mm | Large base height: {self.params.large_base_height:.2f} mm\n"
            f"Radius samples -> start: {base_r:.2f} mm, neck: {neck_r:.2f} mm, end: {end_r:.2f} mm\n"
            f"Lattice window: {lat0:.3f} to {lat1:.3f} | Polylines: {len(polylines)}\n"
            f"{cal_text}"
        )
        self.info_label.configure(text=txt)

    # ---------- export ----------

    def export_current(self):
        try:
            self.params.calibration_path = self.calibration_path_var.get().strip()
            self.params.output_path = self.output_path_var.get().strip()
            self.params.show_wireframe = int(self.show_wireframe_var.get())
            self.params.show_normals = int(self.show_normals_var.get())
            self.params.lattice_include_rings = int(self.lattice_rings_var.get())
            self.params.lattice_include_meridians = int(self.lattice_meridians_var.get())
            self.params.lattice_include_diagonals = int(self.lattice_diagonals_var.get())
            self.params = sanitize_params(self.params)
            out_path = self.params.output_path or DEFAULT_OUT
            result = export_gcode(self.params, out_path)
            warn_msg = ""
            if result["warning_count"] > 0:
                warn_msg = f"\nBBox clamp warnings: {result['warning_count']}"
            self.status_var.set(f"Exported: {out_path}{warn_msg}")
            messagebox.showinfo(
                "Export complete",
                f"Wrote:\n{out_path}\n\n"
                f"Approx print length: {result['approx_print_length_mm']:.2f} mm\n"
                f"Polylines: {result['polyline_count']}\n"
                f"Warnings: {result['warning_count']}",
            )
        except Exception as exc:
            tb = traceback.format_exc()
            self.status_var.set(f"Export failed: {exc}")
            messagebox.showerror("Export failed", f"{exc}\n\n{tb}")


# ============================================================================
# CLI / main
# ============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Interactive Klein bottle shell/lattice G-code generator.")
    ap.add_argument("--calibration", default="", help="Path to calibration JSON.")
    ap.add_argument("--preset", default="", help="Optional preset JSON to load on startup.")
    return ap.parse_args(argv)


def load_startup_params(args: argparse.Namespace) -> Params:
    p = Params()
    if args.preset:
        with open(args.preset, "r", encoding="utf-8") as f:
            data = json.load(f)
        p = params_from_mapping(data, base=p)
    if args.calibration:
        p.calibration_path = args.calibration
    return sanitize_params(p)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    params = load_startup_params(args)
    if FigureCanvasTkAgg is None:
        raise RuntimeError("TkAgg backend is unavailable in this environment; run on a desktop Python with Tk support.")
    root = tk.Tk()
    app = KleinBottleApp(root, initial_params=params)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
