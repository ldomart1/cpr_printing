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
DEFAULT_CALIBRATION_PATH = "Test_Calibration_2026-05-11_00/processed_image_data_folder/calibrated_robot_gcode_calibration.json"

# Tip-space placement (mm)
DEFAULT_START_X = 100.0
DEFAULT_START_Y = 0.0
DEFAULT_START_Z = -120.0

# Machine stage startup/end poses (raw stage axes)
DEFAULT_MACHINE_START_X = 100.0
DEFAULT_MACHINE_START_Y = 0.0
DEFAULT_MACHINE_START_Z = 0.0
DEFAULT_MACHINE_START_B = 0.0
DEFAULT_MACHINE_START_C = 0.0

DEFAULT_MACHINE_END_X = 100.0
DEFAULT_MACHINE_END_Y = 0.0
DEFAULT_MACHINE_END_Z = 0.0
DEFAULT_MACHINE_END_B = 0.0
DEFAULT_MACHINE_END_C = 0.0

DEFAULT_SAFE_APPROACH_Z = 0.0

# Motion
DEFAULT_TRAVEL_FEED = 1200.0
DEFAULT_PRINT_FEED = 500.0
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
DEFAULT_SPINE_HEIGHT = 56.5
DEFAULT_SPINE_R0 = 36.0
DEFAULT_SPINE_R1 = 13.0
DEFAULT_SPINE_S = 0.0
DEFAULT_SPINE_Z_WOBBLE = 0.0
DEFAULT_TWIST_DEG = 0.0
DEFAULT_CONTINUOUS_LAYER_HEIGHT = 2.45
DEFAULT_SHELL_PHASE_DEG = 0.0
DEFAULT_SPINE_X_SCALE = 1.0
DEFAULT_SPINE_Y_SCALE = 1.0
DEFAULT_INNER_START_OFFSET_MM = 0.0
DEFAULT_RETURN_START_OFFSET_MM = 0.0
DEFAULT_OFFPLANE_SIGN = -1.0
DEFAULT_SMALL_TUBE_COLLISION_CLEARANCE_MM = 0.75
DEFAULT_SMALL_TUBE_DETOUR_SAMPLES = 4
DEFAULT_SMALL_TUBE_B_ANGLE_DEG = 180.0
DEFAULT_OUTER_SHELL_B_ANGLE_DEG = 0.0
DEFAULT_OUTER_SHELL_PRE_SEAM_B_ANGLE_DEG = 90.0
DEFAULT_OUTER_SHELL_B_RAMP_DISTANCE_MM = 18.0
DEFAULT_SEAM_AVOIDANCE_B_ANGLE_DEG = 30.0
DEFAULT_SEAM_AVOIDANCE_BOTTOM_B_ANGLE_DEG = 90.0
DEFAULT_FINAL_PASS_B_ANGLE_DEG = 30.0
DEFAULT_FINAL_PASS_ENABLE = 1
DEFAULT_FINAL_PASS_LENGTH_MM = 12.0
DEFAULT_FINAL_PASS_ARC_SEGMENTS = 36
DEFAULT_FINAL_PASS_X_OFFSET_MM = 0.5
# C plane convention: old tangent-based C schedule, but flipped by 180 deg.
# If your machine convention needs an additional perpendicular offset, adjust this in the GUI.
DEFAULT_C_TANGENT_PLANE_OFFSET_DEG = 180.0

# Rotary layer / bounded-C controls for small tube and base transition
DEFAULT_C_LAYER_START_DEG = -180.0
DEFAULT_C_LAYER_END_DEG = 180.0
# Apply a physical C-plane/azimuth offset to the geometric circle layers while
# preserving direct bounded C commands. Default -180 flips the short-tube plane
# without requiring any unwind/reposition move.
DEFAULT_CIRCLE_LAYER_C_PLANE_OFFSET_DEG = -180.0
DEFAULT_BASE_TRANSITION_RING_STEP_MM = 1.5

# Radius controls
DEFAULT_LARGE_BASE_RADIUS = 30.0
DEFAULT_LARGE_BASE_HEIGHT = 15.5
DEFAULT_BASE_TUBE_RADIUS = 8.5
DEFAULT_BASE_TUBE_HEIGHT = 15.5
DEFAULT_NECK_RADIUS = 7.8
DEFAULT_NECK_CENTER = 0.305
DEFAULT_NECK_WIDTH = 0.24
DEFAULT_NECK_POWER = 3.0
DEFAULT_BASE_FILLET_SPAN = 0.5
DEFAULT_BASE_OUTER_EDGE_FILLET = 16.3

# Dark UI colors
UI_BG = "#000000"
UI_PANEL_BG = "#000000"
UI_PANEL_ALT_BG = "#0a0a0a"
UI_FG = "#e6edf3"
UI_MUTED_FG = "#a8b3c2"
UI_ACCENT = "#5aa9ff"
UI_BORDER = "#3b4350"
UI_ENTRY_BG = "#000000"
UI_PLOT_BG = "#000000"
UI_GRID = "#4b5563"

# Lattice window + density
DEFAULT_LATTICE_START = 0.0
DEFAULT_LATTICE_END = 0.0
DEFAULT_LATTICE_U_COUNT = 11
DEFAULT_LATTICE_V_COUNT = 12
DEFAULT_LATTICE_INCLUDE_RINGS = 0
DEFAULT_LATTICE_INCLUDE_MERIDIANS = 0
DEFAULT_LATTICE_INCLUDE_DIAGONALS = 0
DEFAULT_ENABLE_LATTICE = 0

# Orientation convention
DEFAULT_TOOL_NORMAL_SIGN = 1

# BBox
DEFAULT_BBOX_X_MIN = 0.0
DEFAULT_BBOX_X_MAX = 160.0
DEFAULT_BBOX_Y_MIN = 0.0
DEFAULT_BBOX_Y_MAX = 200.0
DEFAULT_BBOX_Z_MIN = -200.0
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
    r_model: Optional[Dict[str, Any]] = None
    z_model: Optional[Dict[str, Any]] = None
    y_off_model: Optional[Dict[str, Any]] = None
    y_off_extrap_model: Optional[Dict[str, Any]] = None
    tip_angle_model: Optional[Dict[str, Any]] = None
    selected_fit_model: Optional[str] = None
    selected_offplane_fit_model: Optional[str] = None
    active_offplane_fit_model: Optional[str] = None
    active_phase: str = "pull"


@dataclass
class Params:
    calibration_path: str = DEFAULT_CALIBRATION_PATH
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
    spine_x_scale: float = DEFAULT_SPINE_X_SCALE
    spine_y_scale: float = DEFAULT_SPINE_Y_SCALE
    twist_deg: float = DEFAULT_TWIST_DEG
    continuous_layer_height: float = DEFAULT_CONTINUOUS_LAYER_HEIGHT
    shell_phase_deg: float = DEFAULT_SHELL_PHASE_DEG
    inner_start_offset_mm: float = DEFAULT_INNER_START_OFFSET_MM
    return_start_offset_mm: float = DEFAULT_RETURN_START_OFFSET_MM
    small_tube_collision_clearance_mm: float = DEFAULT_SMALL_TUBE_COLLISION_CLEARANCE_MM
    small_tube_detour_samples: int = DEFAULT_SMALL_TUBE_DETOUR_SAMPLES
    small_tube_b_angle_deg: float = DEFAULT_SMALL_TUBE_B_ANGLE_DEG
    outer_shell_b_angle_deg: float = DEFAULT_OUTER_SHELL_B_ANGLE_DEG
    outer_shell_pre_seam_b_angle_deg: float = DEFAULT_OUTER_SHELL_PRE_SEAM_B_ANGLE_DEG
    outer_shell_b_ramp_distance_mm: float = DEFAULT_OUTER_SHELL_B_RAMP_DISTANCE_MM
    seam_avoidance_b_angle_deg: float = DEFAULT_SEAM_AVOIDANCE_B_ANGLE_DEG
    seam_avoidance_bottom_b_angle_deg: float = DEFAULT_SEAM_AVOIDANCE_BOTTOM_B_ANGLE_DEG
    final_pass_b_angle_deg: float = DEFAULT_FINAL_PASS_B_ANGLE_DEG
    final_pass_enable: int = DEFAULT_FINAL_PASS_ENABLE
    final_pass_length_mm: float = DEFAULT_FINAL_PASS_LENGTH_MM
    final_pass_arc_segments: int = DEFAULT_FINAL_PASS_ARC_SEGMENTS
    final_pass_x_offset_mm: float = DEFAULT_FINAL_PASS_X_OFFSET_MM
    c_tangent_plane_offset_deg: float = DEFAULT_C_TANGENT_PLANE_OFFSET_DEG
    c_layer_start_deg: float = DEFAULT_C_LAYER_START_DEG
    c_layer_end_deg: float = DEFAULT_C_LAYER_END_DEG
    circle_layer_c_plane_offset_deg: float = DEFAULT_CIRCLE_LAYER_C_PLANE_OFFSET_DEG
    base_transition_ring_step_mm: float = DEFAULT_BASE_TRANSITION_RING_STEP_MM

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
    enable_lattice: int = DEFAULT_ENABLE_LATTICE
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

    show_wireframe: int = 0
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
    radius_mode: str = "outer"
    c_mode: str = "continuous"
    c_sweep_u: float = 0.0
    c_sweep_start_deg: Optional[float] = None
    c_sweep_end_deg: Optional[float] = None
    c_angle_override_deg: Optional[float] = None
    c_azimuth_sign: float = 1.0
    c_azimuth_offset_deg: float = 0.0
    b_radius_override_mm: Optional[float] = None
    b_angle_override_deg: Optional[float] = None
    xyz_override: Optional[Tuple[float, float, float]] = None


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


@dataclass
class BRadiusInversionResult:
    b_cmd: float
    radius_actual: float
    z_offset: float
    y_off: float
    clamped: bool


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
            {"key": "spine_x_scale", "label": "X Bend Scale", "from_": -2.0, "to": 2.0, "resolution": 0.05, "step": 0.05},
            {"key": "spine_y_scale", "label": "Y Bend Scale", "from_": -2.0, "to": 2.0, "resolution": 0.05, "step": 0.05},
            {"key": "twist_deg", "label": "Twist Deg", "from_": -720.0, "to": 720.0, "resolution": 1.0, "step": 1.0},
            {"key": "continuous_layer_height", "label": "Continuous Layer H", "from_": 0.1, "to": 10.0, "resolution": 0.05, "step": 0.05},
            {"key": "shell_phase_deg", "label": "Shell Phase", "from_": -180.0, "to": 180.0, "resolution": 1.0, "step": 1.0},
            {"key": "inner_start_offset_mm", "label": "Inner Start Offset", "from_": 0.0, "to": 40.0, "resolution": 0.25, "step": 0.25},
            {"key": "return_start_offset_mm", "label": "Return Start Offset", "from_": 0.0, "to": 40.0, "resolution": 0.25, "step": 0.25},
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
            {"key": "base_fillet_span", "label": "Base Fillet Span", "from_": 0.0, "to": 2.0, "resolution": 0.005, "step": 0.005},
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
            {"key": "small_tube_collision_clearance_mm", "label": "Tube Clearance", "from_": 0.0, "to": 5.0, "resolution": 0.05, "step": 0.05},
            {"key": "small_tube_detour_samples", "label": "Tube Detour Samples", "from_": 3, "to": 16, "resolution": 1, "step": 1},
            {"key": "small_tube_b_angle_deg", "label": "Small Tube B Angle", "from_": 0.0, "to": 180.0, "resolution": 1.0, "step": 1.0},
            {"key": "outer_shell_b_angle_deg", "label": "Outer Shell B Angle", "from_": 0.0, "to": 180.0, "resolution": 1.0, "step": 1.0},
            {"key": "outer_shell_pre_seam_b_angle_deg", "label": "Outer Pre-Seam B Angle", "from_": 0.0, "to": 180.0, "resolution": 1.0, "step": 1.0},
            {"key": "outer_shell_b_ramp_distance_mm", "label": "Outer B Ramp Distance", "from_": 0.0, "to": 80.0, "resolution": 0.5, "step": 0.5},
            {"key": "seam_avoidance_bottom_b_angle_deg", "label": "Seam Bottom B Angle", "from_": 30.0, "to": 90.0, "resolution": 1.0, "step": 1.0},
            {"key": "seam_avoidance_b_angle_deg", "label": "Seam Top B Angle", "from_": 30.0, "to": 90.0, "resolution": 1.0, "step": 1.0},
            {"key": "final_pass_b_angle_deg", "label": "Final Junction B Angle", "from_": 0.0, "to": 90.0, "resolution": 1.0, "step": 1.0},
            {"key": "final_pass_length_mm", "label": "Final Pass Length", "from_": 0.0, "to": 40.0, "resolution": 0.25, "step": 0.25},
            {"key": "final_pass_arc_segments", "label": "Final Pass Segments", "from_": 4, "to": 180, "resolution": 1, "step": 1},
            {"key": "c_tangent_plane_offset_deg", "label": "C Tangent Plane Offset", "from_": -180.0, "to": 180.0, "resolution": 1.0, "step": 1.0},
            {"key": "c_layer_start_deg", "label": "Circle C Start", "from_": -180.0, "to": 180.0, "resolution": 1.0, "step": 1.0},
            {"key": "c_layer_end_deg", "label": "Circle C End", "from_": -180.0, "to": 180.0, "resolution": 1.0, "step": 1.0},
            {"key": "circle_layer_c_plane_offset_deg", "label": "Circle C Plane Offset", "from_": -180.0, "to": 180.0, "resolution": 1.0, "step": 1.0},
            {"key": "base_transition_ring_step_mm", "label": "Base Transition Ring Step", "from_": 0.1, "to": 10.0, "resolution": 0.1, "step": 0.1},
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

def clamp(x: float, lo: float, hi: float) -> float:
    return max(float(lo), min(float(hi), float(x)))


def clamp01(x: float) -> float:
    return clamp(float(x), 0.0, 1.0)


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


def wrap_degrees_180(angle_deg: float) -> float:
    a = (float(angle_deg) + 180.0) % 360.0 - 180.0
    if a <= -180.0:
        a += 360.0
    return a


def clamp_degrees(angle_deg: float, lo: float = -360.0, hi: float = 360.0) -> float:
    return max(float(lo), min(float(hi), float(angle_deg)))


def bounded_equivalent_degrees(
    angle_deg: float,
    prev_deg: Optional[float] = None,
    lo: float = -360.0,
    hi: float = 360.0,
) -> float:
    """Return an equivalent C angle inside [lo, hi], preferring continuity."""
    lo = float(lo)
    hi = float(hi)
    if hi <= lo:
        raise ValueError("Invalid C bounds.")
    base = wrap_degrees_180(angle_deg)
    candidates = [base + 360.0 * k for k in range(-4, 5)]
    candidates = [c for c in candidates if lo - 1e-9 <= c <= hi + 1e-9]
    if not candidates:
        return clamp_degrees(base, lo, hi)
    if prev_deg is None:
        target = float(angle_deg)
    else:
        target = float(prev_deg)
    return float(min(candidates, key=lambda c: abs(c - target)))


def surface_azimuth_c_deg(phi_rad: float, sign: float = 1.0, offset_deg: float = 0.0) -> float:
    """Map a surface azimuth phi to a bounded C command in [-180, 180]."""
    return bounded_equivalent_degrees(float(sign) * math.degrees(float(phi_rad)) + float(offset_deg), None, -180.0, 180.0)


def phi_from_surface_azimuth_c_deg(c_deg: float, sign: float = 1.0, offset_deg: float = 0.0) -> float:
    """Inverse of surface_azimuth_c_deg before wrapping, used to flip the physical azimuth plane."""
    sgn = 1.0 if float(sign) >= 0.0 else -1.0
    return math.radians((float(c_deg) - float(offset_deg)) / sgn)


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


def _select_first_model(models_list: Sequence[Optional[Dict[str, Any]]], *keys: str) -> Optional[Dict[str, Any]]:
    for models in models_list:
        if not isinstance(models, dict):
            continue
        for key in keys:
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
                d[k] = (w1 + w2) / (w1 / delta[k - 1] + w2 / delta[k])
        d[0] = _pchip_endpoint_slope(h[0], h[1], delta[0], delta[1])
        d[-1] = _pchip_endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])
    flat = xq.reshape(-1)
    idx = np.clip(np.searchsorted(x, flat, side="right") - 1, 0, x.size - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    y0 = y[idx]
    y1 = y[idx + 1]
    h_i = x1 - x0
    t = np.clip((flat - x0) / h_i, 0.0, 1.0)
    h00 = 2 * t**3 - 3 * t**2 + 1
    h10 = t**3 - 2 * t**2 + t
    h01 = -2 * t**3 + 3 * t**2
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
        return poly_eval(model_spec.get("coefficients", model_spec.get("coeffs")), u, default_if_none)
    raise ValueError(f"Unsupported calibration model_type: {model_type}")


def eval_pchip_with_linear_extrap(
    model_spec: Dict[str, Any],
    extrap_model_spec: Optional[Dict[str, Any]],
    b: Any,
) -> np.ndarray:
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
    shared_aux_models = data.get("shared_aux_fit_models", {}) or {}

    model_sources = (active_models, fit_models, shared_aux_models)

    y_selector = selected_offplane_fit_model or selected_fit_model or "avg_cubic"
    r_model = _select_named_model(active_models, "r", selected_fit_model)
    z_model = _select_named_model(active_models, "z", selected_fit_model)
    tip_angle_model = _select_named_model(active_models, "tip_angle", selected_fit_model)
    y_off_model = _select_first_model(model_sources, "offplane_y_avg_cubic")
    active_offplane_fit_model = "avg_cubic" if y_off_model is not None else None
    if y_off_model is None:
        y_off_model = _select_named_model(active_models, "offplane_y", y_selector)
        active_offplane_fit_model = selected_offplane_fit_model or selected_fit_model
    y_off_extrap_model = _select_first_model(
        model_sources,
        "offplane_y_avg_linear",
        "offplane_y_linear",
        "offplane_y",
    )

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
        r_model=r_model,
        z_model=z_model,
        y_off_model=y_off_model,
        y_off_extrap_model=y_off_extrap_model,
        tip_angle_model=tip_angle_model,
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
        active_offplane_fit_model=active_offplane_fit_model,
        active_phase=active_phase,
    )


def eval_r(cal: Calibration, b: Any) -> np.ndarray:
    return eval_model_spec(cal.r_model, b) if cal.r_model is not None else poly_eval(cal.pr, b)


def eval_z(cal: Calibration, b: Any) -> np.ndarray:
    return eval_model_spec(cal.z_model, b) if cal.z_model is not None else poly_eval(cal.pz, b)


def eval_offplane_y(cal: Calibration, b: Any) -> np.ndarray:
    if cal.y_off_model is not None:
        if str(cal.y_off_model.get("model_type", "")).strip().lower() == "pchip":
            vals = eval_pchip_with_linear_extrap(cal.y_off_model, cal.y_off_extrap_model, b)
        else:
            vals = eval_model_spec(cal.y_off_model, b, default_if_none=0.0)
    else:
        vals = poly_eval(cal.py_off, b, default_if_none=0.0)
    return float(DEFAULT_OFFPLANE_SIGN) * np.asarray(vals, dtype=float)


def eval_tip_angle_deg(cal: Calibration, b: Any) -> np.ndarray:
    if cal.tip_angle_model is not None:
        return eval_model_spec(cal.tip_angle_model, b)
    if cal.pa is None:
        raise ValueError("Calibration is missing tip_angle_coeffs.")
    return poly_eval(cal.pa, b)


def tip_offset_xyz_physical(cal: Calibration, b_cmd: float, c_deg: float) -> np.ndarray:
    """Physical tip offset from the stage XYZ origin at B/C.

    The calibration off-plane value is signed first with offplane_sign = -1:
        y_off(B) = -offplane(B)

    Then the in-plane radius r(B) and signed off-plane offset y_off(B)
    are rotated by C about the stage Z axis:
        X_tip = X_stage + r*cos(C) - y_off*sin(C)
        Y_tip = Y_stage + r*sin(C) + y_off*cos(C)
        Z_tip = Z_stage + z(B)

    With y_off = -offplane, this is equivalently:
        X_tip = X_stage + r*cos(C) + offplane*sin(C)
        Y_tip = Y_stage + r*sin(C) - offplane*cos(C)
        Z_tip = Z_stage + z(B)
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


class TipAngleInverter:
    """Approximate inverse for tip_angle_deg(B_cmd), with branch continuity.

    Some calibrations are not one-to-one: the same requested tip angle can have
    more than one valid B command. The older sorted-angle interpolation could
    silently jump between branches as the requested angle changed, which showed
    up as large B oscillations after the small-tube circle layers. This inverter
    finds all sampled/crossing solutions and, when a previous B value is
    available, chooses the closest branch for continuous motion.
    """

    def __init__(self, cal: Calibration, samples: int = 20001):
        if cal.pa is None and cal.tip_angle_model is None:
            raise ValueError("Calibration JSON has no tip_angle_coeffs / tip_angle model.")
        self.cal = cal
        self.b_samples = np.linspace(float(cal.b_min), float(cal.b_max), int(max(1001, samples)))
        self.a_samples = np.asarray(eval_tip_angle_deg(cal, self.b_samples), dtype=float)
        finite = np.isfinite(self.a_samples) & np.isfinite(self.b_samples)
        if not np.any(finite):
            raise ValueError("Calibration tip-angle samples are all non-finite.")
        self.b_samples = self.b_samples[finite]
        self.a_samples = self.a_samples[finite]
        order = np.argsort(self.a_samples)
        a_sorted = np.asarray(self.a_samples[order], dtype=float)
        b_sorted = np.asarray(self.b_samples[order], dtype=float)
        a_unique, idx = np.unique(a_sorted, return_index=True)
        self.a = a_unique
        self.b = b_sorted[idx]
        self.a_min = float(np.min(self.a_samples))
        self.a_max = float(np.max(self.a_samples))

    def _candidate_bs_for_angle(self, angle_deg: float) -> List[float]:
        target = float(angle_deg)
        diff = self.a_samples - target
        candidates: List[float] = []
        tol = 1e-7

        exact_idx = np.flatnonzero(np.abs(diff) <= tol)
        if exact_idx.size:
            candidates.extend(float(self.b_samples[i]) for i in exact_idx.tolist())

        cross_idx = np.flatnonzero(diff[:-1] * diff[1:] < 0.0)
        for i in cross_idx.tolist():
            a0 = float(self.a_samples[i])
            a1 = float(self.a_samples[i + 1])
            b0 = float(self.b_samples[i])
            b1 = float(self.b_samples[i + 1])
            if abs(a1 - a0) <= tol:
                candidates.append(0.5 * (b0 + b1))
            else:
                u = (target - a0) / (a1 - a0)
                candidates.append((1.0 - u) * b0 + u * b1)

        if not candidates:
            nearest = np.argsort(np.abs(diff))[:8]
            candidates.extend(float(self.b_samples[i]) for i in nearest.tolist())

        # Deduplicate nearby candidates while preserving order.
        deduped: List[float] = []
        for b in candidates:
            if not any(abs(b - existing) < 1e-5 for existing in deduped):
                deduped.append(float(b))
        return deduped

    def angle_to_b(self, angle_deg: float, prefer_b: Optional[float] = None) -> Tuple[float, bool]:
        ang = float(angle_deg)
        clamped = False
        if ang <= self.a_min:
            ang = self.a_min
            clamped = True
        elif ang >= self.a_max:
            ang = self.a_max
            clamped = True

        candidates = self._candidate_bs_for_angle(ang)
        if prefer_b is not None and np.isfinite(float(prefer_b)):
            b_cmd = min(candidates, key=lambda b: abs(float(b) - float(prefer_b)))
        else:
            # Backward-compatible fallback for the first point of a file/polyline.
            b_cmd = float(np.interp(ang, self.a, self.b))
        return float(b_cmd), clamped


def effective_radius(cal: Calibration, b_cmd: Any, use_offplane: bool = True) -> np.ndarray:
    r = np.asarray(eval_r(cal, b_cmd), dtype=float)
    if not use_offplane:
        return r
    y = np.asarray(eval_offplane_y(cal, b_cmd), dtype=float)
    return np.sqrt(r * r + y * y)


class BRadiusInverter:
    """Approximate inverse from desired calibrated rotary tip radius to B command."""

    def __init__(self, cal: Calibration, use_offplane: bool = True, samples: int = 20001):
        self.cal = cal
        self.use_offplane = bool(use_offplane)
        self.b_samples = np.linspace(float(cal.b_min), float(cal.b_max), int(max(1001, samples)))
        self.rad_samples = np.asarray(effective_radius(cal, self.b_samples, self.use_offplane), dtype=float)
        if self.rad_samples.size < 2:
            raise ValueError("Calibration radius range is degenerate; cannot invert radius to B.")
        self.r_min = float(np.min(self.rad_samples))
        self.r_max = float(np.max(self.rad_samples))

    def radius_to_b(self, target_radius: float) -> BRadiusInversionResult:
        target = float(target_radius)
        clamped = False
        if target <= self.r_min:
            target_i = self.r_min
            clamped = True
        elif target >= self.r_max:
            target_i = self.r_max
            clamped = True
        else:
            target_i = target

        diff = self.rad_samples - float(target_i)
        candidates: List[float] = []
        tol = 1e-9

        exact_idx = np.flatnonzero(np.abs(diff) <= tol)
        if exact_idx.size:
            candidates.extend(float(self.b_samples[i]) for i in exact_idx.tolist())

        cross_idx = np.flatnonzero(diff[:-1] * diff[1:] < 0.0)
        for i in cross_idx.tolist():
            r0 = float(self.rad_samples[i])
            r1 = float(self.rad_samples[i + 1])
            b0 = float(self.b_samples[i])
            b1 = float(self.b_samples[i + 1])
            if abs(r1 - r0) <= tol:
                candidates.append(max(b0, b1))
                continue
            a = (float(target_i) - r0) / (r1 - r0)
            candidates.append((1.0 - a) * b0 + a * b1)

        if not candidates:
            nearest = np.argsort(np.abs(diff))[:8]
            candidates.extend(float(self.b_samples[i]) for i in nearest.tolist())

        b_cmd = float(
            sorted(
                candidates,
                key=lambda b: (abs(float(eval_tip_angle_deg(self.cal, b))), -float(b)),
            )[0]
        )
        actual = float(effective_radius(self.cal, b_cmd, self.use_offplane))
        zoff = float(eval_z(self.cal, b_cmd))
        yoff = float(eval_offplane_y(self.cal, b_cmd))
        return BRadiusInversionResult(
            b_cmd=b_cmd,
            radius_actual=actual,
            z_offset=zoff,
            y_off=yoff,
            clamped=clamped,
        )


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
    x_scale: float,
    y_scale: float,
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

    loop_x = float(x_scale) * ((r0 + r1 * math.cos(u)) * math.cos(u) + s_cross * math.sin(2.0 * u))
    loop_y = float(y_scale) * (r0 + r1 * math.cos(u)) * math.sin(u)

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
    # Keep the large-base vertical collar coincident with the small base tube.
    return min(max(0.0, float(p.large_base_height)), max(0.0, float(p.base_tube_height)))


def start_branch_height_at_t(t: float, p: Params) -> float:
    t = clamp01(t)
    base_h = max(0.0, float(p.base_tube_height))
    loop_h = max(0.0, float(p.spine_height))
    frac = base_tube_fraction(p)
    if frac > 1.0 - 1e-9:
        return base_h * t
    if frac > 1e-9 and t <= frac:
        return base_h * (t / frac)
    u = 0.0 if frac >= 1.0 else (t - frac) / max(1e-9, 1.0 - frac)
    return base_h + loop_h * clamp01(u)


def start_branch_t_for_height(height_mm: float, p: Params) -> float:
    base_h = max(0.0, float(p.base_tube_height))
    loop_h = max(0.0, float(p.spine_height))
    frac = base_tube_fraction(p)
    h = max(0.0, float(height_mm))
    if h <= base_h + 1e-9:
        if frac <= 1e-9:
            return 0.0
        return clamp01(frac * (h / max(base_h, 1e-9)))
    if loop_h <= 1e-9:
        return clamp01(frac)
    s = clamp01((h - base_h) / loop_h)
    u = clamp01(math.asin(s) / math.pi)
    return clamp01(frac + (1.0 - frac) * u)


def return_branch_t_for_height(height_mm: float, p: Params) -> float:
    base_h = max(0.0, float(p.base_tube_height))
    loop_h = max(0.0, float(p.spine_height))
    frac = base_tube_fraction(p)
    h = max(base_h, float(height_mm))
    if loop_h <= 1e-9:
        return 1.0
    s = clamp01((h - base_h) / loop_h)
    u = 1.0 - (math.asin(s) / math.pi)
    return clamp01(frac + (1.0 - frac) * u)


def base_fillet_intersection_t(p: Params) -> float:
    """
    First start-branch parameter where the large outer-base profile has relaxed
    back to the body tube radius. This ring is the geometric intersection of the
    inner tube surface and the outer fillet surface.
    """
    ts = np.linspace(0.0, 1.0, 2001)
    diffs = np.array([radius_profile(float(t), p) - body_radius_profile(float(t), p) for t in ts], dtype=float)
    tol = 1e-6
    hit = np.flatnonzero(diffs <= tol)
    if hit.size == 0:
        return 1.0
    idx = int(hit[0])
    if idx <= 0:
        return float(ts[0])
    lo = float(ts[idx - 1])
    hi = float(ts[idx])
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if radius_profile(mid, p) - body_radius_profile(mid, p) <= tol:
            hi = mid
        else:
            lo = mid
    return clamp01(hi)


def return_base_fillet_intersection_t(p: Params) -> float:
    """
    Return-side parameter where the large outer-base profile relaxes back
    to the body/thin tube radius.

    This is the point near t=1 where the outer large-base/fillet surface
    meets the thin tube surface.
    """
    frac = base_tube_fraction(p)

    ts = np.linspace(1.0, frac, 2001)
    diffs = np.array(
        [radius_profile(float(t), p) - body_radius_profile(float(t), p) for t in ts],
        dtype=float,
    )

    tol = 1e-6
    hit = np.flatnonzero(diffs <= tol)

    if hit.size == 0:
        return frac

    idx = int(hit[0])
    if idx <= 0:
        return 1.0

    lo = float(ts[idx - 1])
    hi = float(ts[idx])

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if radius_profile(mid, p) - body_radius_profile(mid, p) <= tol:
            hi = mid
        else:
            lo = mid

    return clamp01(hi)


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
                x_scale=float(p.spine_x_scale),
                y_scale=float(p.spine_y_scale),
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


def t_for_spine_arclength_mm(cache: SurfaceCache, s_mm: float) -> float:
    s = float(np.clip(float(s_mm), float(cache.spine_s_mm[0]), float(cache.spine_s_mm[-1])))
    idx = int(np.searchsorted(cache.spine_s_mm, s, side="left"))
    if idx <= 0:
        return float(cache.ts[0])
    if idx >= len(cache.spine_s_mm):
        return float(cache.ts[-1])
    s0 = float(cache.spine_s_mm[idx - 1])
    s1 = float(cache.spine_s_mm[idx])
    if s1 <= s0 + 1e-15:
        return float(cache.ts[idx])
    a = (s - s0) / (s1 - s0)
    return float((1.0 - a) * cache.ts[idx - 1] + a * cache.ts[idx])


def shell_phi_for_t(cache: SurfaceCache, t: float, p: Params) -> float:
    pitch = max(0.05, float(p.continuous_layer_height))
    turns = spine_arclength_mm(cache, t) / pitch
    return math.radians(float(p.shell_phase_deg)) + 2.0 * math.pi * turns


def effective_surface_radius_for_mode(t: float, p: Params, radius_mode: str) -> float:
    mode = str(radius_mode)
    if mode == "inner":
        return body_radius_profile(t, p)
    if mode == "outer":
        return radius_profile(t, p)
    return radius_profile(t, p)


def surface_spacing_kappa(t: float, p: Params, radius_mode: str) -> float:
    """
    dphi/ds for approximately constant surface spacing between neighboring
    spiral turns.

    For tube metric ds^2 + r^2 dphi^2, spacing between adjacent turns is:
        spacing = 2π / sqrt(kappa^2 + 1/r^2)

    Solve for kappa using spacing = continuous_layer_height.
    """
    h = max(0.05, float(p.continuous_layer_height))
    r = max(1e-6, effective_surface_radius_for_mode(t, p, radius_mode))

    target = 2.0 * math.pi / h
    curvature_term = 1.0 / r

    val = target * target - curvature_term * curvature_term
    if val <= 0.0:
        return 0.0
    return math.sqrt(val)


def shell_phi_for_t_surface_spaced(
    cache: SurfaceCache,
    t: float,
    p: Params,
    radius_mode: str,
) -> float:
    """
    Metric-aware replacement for shell_phi_for_t().
    Integrates dphi/ds over the spine so the adjacent-turn spacing on
    the swept surface is closer to continuous_layer_height.
    """
    t = clamp01(t)

    ts = cache.ts[cache.ts <= t]
    if len(ts) == 0 or ts[-1] < t:
        ts = np.append(ts, t)

    if len(ts) < 2:
        return math.radians(float(p.shell_phase_deg))

    ss = np.asarray([spine_arclength_mm(cache, float(tt)) for tt in ts], dtype=float)
    kk = np.asarray([surface_spacing_kappa(float(tt), p, radius_mode) for tt in ts], dtype=float)

    trap_fn = getattr(np, "trapezoid", None)
    if trap_fn is None:
        trap_fn = np.trapz
    phi = float(trap_fn(kk, ss))
    return math.radians(float(p.shell_phase_deg)) + phi


def point_on_surface(cache: SurfaceCache, p: Params, t: float, phi: float) -> np.ndarray:
    c = spine_point(cache, t)
    n, b = normal_frame(cache, t)
    r = radius_profile(t, p)
    return c + r * (math.cos(phi) * n + math.sin(phi) * b)


def point_on_surface_with_radius(cache: SurfaceCache, p: Params, t: float, phi: float, radius_mode: str, blend_u: float = 0.0) -> np.ndarray:
    c = spine_point(cache, t)
    n, b = normal_frame(cache, t)
    r_inner = body_radius_profile(t, p)
    r_outer = radius_profile(t, p)
    mode = str(radius_mode)
    if mode == "inner":
        r = r_inner
    elif mode == "blend":
        a = 0.5 - 0.5 * math.cos(math.pi * clamp01(blend_u))
        r = (1.0 - a) * r_inner + a * r_outer
    else:
        r = r_outer
    return c + r * (math.cos(phi) * n + math.sin(phi) * b)


def phi_for_xyz_at_t(cache: SurfaceCache, xyz: np.ndarray, t: float) -> float:
    c = spine_point(cache, t)
    n, b = normal_frame(cache, t)
    v = np.asarray(xyz, dtype=float) - c
    return math.atan2(float(np.dot(v, b)), float(np.dot(v, n)))


def ring_extreme_phi(
    cache: SurfaceCache,
    p: Params,
    t: float,
    radius_mode: str,
    axis: int,
    find_min: bool,
) -> float:
    """Find ring azimuth whose surface point is min/max along a global axis."""
    n_dense = 720
    phis = np.linspace(-math.pi, math.pi, n_dense, endpoint=False)
    pts = np.asarray(
        [point_on_surface_with_radius(cache, p, float(t), float(phi), radius_mode) for phi in phis],
        dtype=float,
    )
    idx = int(np.argmin(pts[:, int(axis)]) if bool(find_min) else np.argmax(pts[:, int(axis)]))
    return float(phis[idx])


def unwrap_phi_near(prev: float, cur: float) -> float:
    return unwrap_radians_near(float(prev), float(cur))


def phi_path_via_waypoints(start_phi: float, waypoint_phis: Sequence[float], end_phi: float, samples_each: int) -> List[float]:
    """Piecewise-unwrapped phi path through explicit azimuth waypoints."""
    raw = [float(start_phi), *[float(x) for x in waypoint_phis], float(end_phi)]
    unwrapped = [raw[0]]
    for phi in raw[1:]:
        unwrapped.append(unwrap_phi_near(unwrapped[-1], phi))
    out: List[float] = []
    nseg = max(1, int(samples_each))
    for a0, a1 in zip(unwrapped[:-1], unwrapped[1:]):
        vals = np.linspace(a0, a1, nseg + 1)
        if out:
            vals = vals[1:]
        out.extend(float(v) for v in vals)
    return out


def offset_t_up_return_branch(cache: SurfaceCache, p: Params, t_join: float, offset_mm: float) -> float:
    """
    Move the write-start upward/earlier along the return-side spine,
    away from the base fillet junction.

    On the return branch, moving upward means decreasing spine arclength / t.
    """
    base_frac = base_tube_fraction(p)
    s_join = spine_arclength_mm(cache, t_join)
    s_min = spine_arclength_mm(cache, base_frac)

    s_start = max(s_min, s_join - max(0.0, float(offset_mm)))
    return clamp01(t_for_spine_arclength_mm(cache, s_start))


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


def unwrap_radians_near(prev: float, cur: float) -> float:
    a = float(cur)
    p0 = float(prev)
    while a - p0 > math.pi:
        a -= 2.0 * math.pi
    while a - p0 < -math.pi:
        a += 2.0 * math.pi
    return a


def point_segment_distance(q: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    q = np.asarray(q, dtype=float)
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)

    ab = b - a
    denom = float(np.dot(ab, ab))
    if denom <= 1e-12:
        return float(np.linalg.norm(q - a))

    u = float(np.dot(q - a, ab) / denom)
    u = max(0.0, min(1.0, u))
    closest = a + u * ab
    return float(np.linalg.norm(q - closest))


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
    radius_mode: str = "outer",
    track_mode: str = "centerline_tangent",
    b_angle_override_deg: Optional[float] = None,
    b_angle_for_t: Optional[Any] = None,
    c_mode: str = "surface_azimuth",
    c_azimuth_sign: float = 1.0,
    c_azimuth_offset_deg: float = 0.0,
) -> ParamPolyline:
    t0 = clamp01(t0)
    t1 = clamp01(t1)
    n = int(max(2, steps))
    ts = np.linspace(t0, t1, n)
    pts: List[ParamPoint] = []
    for t in ts:
        tt = float(t)
        b_override = None
        if b_angle_for_t is not None:
            b_override = float(b_angle_for_t(tt))
        elif b_angle_override_deg is not None:
            b_override = float(b_angle_override_deg)
        pts.append(
            ParamPoint(
                t=tt,
                phi=shell_phi_for_t_surface_spaced(cache, tt, p, radius_mode) + float(phase_offset),
                mode="surface",
                comment=f"{name} t={tt:.4f}",
                track_sign=float(track_sign),
                track_mode=str(track_mode),
                radius_mode=str(radius_mode),
                c_mode=str(c_mode),
                c_azimuth_sign=float(c_azimuth_sign),
                c_azimuth_offset_deg=float(c_azimuth_offset_deg),
                b_angle_override_deg=b_override,
            )
        )
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


def make_base_transition_polyline(t: float, phi0: float, phi1: float, samples: int, name: str) -> ParamPolyline:
    n = int(max(3, samples))
    spin_sign = 1.0 if float(phi1) >= float(phi0) else -1.0
    pts: List[ParamPoint] = []
    for i in range(n):
        a = 0.0 if n == 1 else (i / (n - 1))
        phi = (1.0 - a) * float(phi0) + a * float(phi1)
        pts.append(
            ParamPoint(
                t=float(t),
                phi=phi,
                mode="surface",
                comment=f"{name} a={a:.3f}",
                track_mode="bottom_turn",
                track_sign=spin_sign,
                track_u=float(a),
                radius_mode="blend",
                c_mode="bounded",
            )
        )
    return ParamPolyline(name=name, points=pts)


def make_small_tube_to_outer_shell_blend_polyline(
    cache: SurfaceCache,
    p: Params,
    start_pt: ParamPoint,
    t_end: float,
    t_final_switch: float,
    samples: int,
    name: str,
) -> ParamPolyline:
    """
    Blend the final small-tube circle-layer parametrization into the outer-shell
    spiral parametrization with one continuous surface-following connector.
    """
    t_end = clamp01(t_end)
    n = int(max(3, samples))
    phi_start = float(start_pt.phi)
    spin_sign = 1.0 if float(start_pt.track_sign) >= 0.0 else -1.0
    pts: List[ParamPoint] = []
    prev_phi = phi_start
    small_b = clamp(float(p.small_tube_b_angle_deg), 0.0, 180.0)

    for i in range(n):
        u = 0.0 if n == 1 else (i / (n - 1))
        a = smootherstep(u)
        tt = a * t_end
        phi_ring = phi_start + spin_sign * (2.0 * math.pi * a)
        phi_outer = shell_phi_for_t_surface_spaced(cache, tt, p, "outer")
        phi_outer = unwrap_radians_near(phi_ring, phi_outer)
        phi = (1.0 - a) * phi_ring + a * phi_outer
        phi = unwrap_radians_near(prev_phi, phi)
        prev_phi = phi
        outer_b = outer_shell_b_angle_for_t(cache, p, tt, t_final_switch)
        pts.append(
            ParamPoint(
                t=float(tt),
                phi=float(phi),
                mode="surface",
                comment=f"{name} u={u:.3f}",
                track_sign=spin_sign,
                track_mode="fixed_down",
                track_u=float(u),
                radius_mode="blend",
                c_mode="path_tangent_plane_flipped",
                b_angle_override_deg=(1.0 - a) * small_b + a * float(outer_b),
            )
        )

    return ParamPolyline(name=name, points=pts)


def build_small_tube_obstacle_samples(
    cache: SurfaceCache,
    p: Params,
    intervals: Sequence[Tuple[float, float]],
    samples_per_interval: int = 200,
    clearance_mm: float = 0.75,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ts_out: List[float] = []
    centers: List[np.ndarray] = []
    radii: List[float] = []

    for t0, t1 in intervals:
        ts = np.linspace(clamp01(t0), clamp01(t1), int(max(8, samples_per_interval)))
        for t in ts:
            tt = float(t)
            ts_out.append(tt)
            centers.append(spine_point(cache, tt))
            radii.append(body_radius_profile(tt, p) + float(clearance_mm))

    return (
        np.asarray(ts_out, dtype=float),
        np.asarray(centers, dtype=float),
        np.asarray(radii, dtype=float),
    )


def segment_small_tube_hit_info(
    a: np.ndarray,
    b: np.ndarray,
    ts: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
) -> Tuple[bool, float, float]:
    """
    Returns:
        hit?
        obstacle t at strongest hit
        penetration depth in mm
    """
    best_pen = -1.0
    best_t = 0.0

    for tt, c, r in zip(ts, centers, radii):
        d = point_segment_distance(c, a, b)
        pen = float(r) - float(d)
        if pen > best_pen:
            best_pen = pen
            best_t = float(tt)

    return best_pen > 0.0, best_t, best_pen


def nearest_solid_boundary_projection(
    xyz: np.ndarray,
    centers: np.ndarray,
    radii: np.ndarray,
    fallback_normal: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, float, int]:
    """
    Project a point to the boundary of the sampled already-printed solid.

    The small tube is interpreted as a swept solid, approximated by the union
    of sampled balls along its centerline with radius = printed tube radius +
    clearance. signed_distance < 0 means the point is inside already printed
    material/clearance and therefore would break through it.
    """
    xyz = np.asarray(xyz, dtype=float)
    if centers.size == 0 or radii.size == 0:
        return xyz.copy(), float("inf"), -1

    v = xyz[None, :] - np.asarray(centers, dtype=float)
    d = np.linalg.norm(v, axis=1)
    signed = d - np.asarray(radii, dtype=float)
    i = int(np.argmin(signed))
    di = float(d[i])
    ri = float(radii[i])

    if di > 1e-9:
        n = v[i] / di
    elif fallback_normal is not None and float(np.linalg.norm(fallback_normal)) > 1e-9:
        n = _normalize(np.asarray(fallback_normal, dtype=float))
    else:
        n = np.array([0.0, 0.0, 1.0], dtype=float)

    return np.asarray(centers[i], dtype=float) + ri * n, float(signed[i]), i


def make_outer_ring_avoidance_arc(
    cache: SurfaceCache,
    p: Params,
    pt0: ParamPoint,
    pt1: ParamPoint,
    obstacle_t: float,
    clearance_mm: float,
    name: str,
    detour_samples: int = 4,
    obstacle_centers: Optional[np.ndarray] = None,
    obstacle_radii: Optional[np.ndarray] = None,
) -> List[ParamPoint]:
    """
    Seam avoidance with solid-boundary interpretation.

    A hit means the nominal outer-shell segment crosses the already-printed
    small-tube solid. Merely adding more points along that same nominal segment
    still cuts through the print. This routine therefore keeps the path on the
    exposed boundary of the union: samples that are inside the small-tube solid
    are projected to the small-tube boundary plus clearance; samples already
    outside remain on the outer shell. That approximates the actual seam edge
    instead of a centerline chord or synthetic ellipse.
    """
    n = int(max(3, min(10, detour_samples)))
    phi0 = float(pt0.phi)
    phi1 = unwrap_radians_near(phi0, float(pt1.phi))

    centers = np.asarray(obstacle_centers if obstacle_centers is not None else np.zeros((0, 3)), dtype=float)
    radii = np.asarray(obstacle_radii if obstacle_radii is not None else np.zeros(0), dtype=float)

    # Build nominal outer-shell edge samples, then replace penetrating samples
    # with closest solid-boundary points. The returned list omits pt0 because
    # the caller has already emitted it.
    samples: List[Tuple[float, float, np.ndarray, float, int, bool]] = []
    for i in range(0, n + 1):
        u = i / max(1, n)
        a = smootherstep(u)
        tt = (1.0 - a) * float(pt0.t) + a * float(pt1.t)
        phi = (1.0 - a) * phi0 + a * phi1
        tt_c = clamp01(tt)
        nominal_xyz = point_on_surface_with_radius(cache, p, tt_c, phi, "outer", float(pt1.track_u))
        n_frame, b_frame = normal_frame(cache, tt_c)
        fallback_normal = math.cos(phi) * n_frame + math.sin(phi) * b_frame
        boundary_xyz, signed, obs_i = nearest_solid_boundary_projection(
            nominal_xyz, centers, radii, fallback_normal=fallback_normal
        )
        inside = bool(signed < 0.0)
        # Use a small blend just outside the signed boundary to prevent the
        # G-code from grazing the solid due to sampling/calibration error.
        xyz = boundary_xyz if inside else nominal_xyz
        samples.append((tt_c, float(phi), xyz, float(signed), int(obs_i), inside))

    xyzs = np.asarray([x[2] for x in samples], dtype=float)
    if xyzs.shape[0] >= 2:
        bottom_i = int(np.argmin(xyzs[:, 1]))  # smallest global Y
        top_i = int(np.argmax(xyzs[:, 2]))     # highest global Z
        bottom_xyz = xyzs[bottom_i]
        top_xyz = xyzs[top_i]
        v_bt = top_xyz - bottom_xyz
        denom = float(np.dot(v_bt, v_bt))
    else:
        bottom_i = top_i = 0
        bottom_xyz = top_xyz = xyzs[0]
        v_bt = np.zeros(3)
        denom = 0.0

    bottom_b = clamp(float(p.seam_avoidance_bottom_b_angle_deg), 30.0, 90.0)
    top_b = clamp(float(p.seam_avoidance_b_angle_deg), 30.0, 90.0)

    out: List[ParamPoint] = []
    for i, (tt, phi, xyz, signed, obs_i, inside) in enumerate(samples[1:], start=1):
        if denom > 1e-9:
            v = float(np.dot(xyz - bottom_xyz, v_bt) / denom)
            v = clamp01(v)
        else:
            v = i / max(1, n)
        b_angle = (1.0 - v) * bottom_b + v * top_b
        out.append(
            ParamPoint(
                t=float(tt),
                phi=float(phi),
                mode="cartesian",
                comment=(
                    f"{name} solid_seam_boundary B={b_angle:.1f} "
                    f"signed={signed:.3f} projected={int(inside)} obs={obs_i} "
                    f"bottomY_idx={bottom_i} topZ_idx={top_i} i={i}/{n}"
                ),
                track_sign=float(pt1.track_sign),
                track_mode="fixed_down",
                track_u=float(pt1.track_u),
                radius_mode="outer",
                c_mode="path_tangent_plane_flipped",
                b_angle_override_deg=b_angle,
                xyz_override=(float(xyz[0]), float(xyz[1]), float(xyz[2])),
            )
        )
    return out

def midpoint_param_point(pt0: ParamPoint, pt1: ParamPoint, name: str = "surface_mid") -> ParamPoint:
    phi0 = float(pt0.phi)
    phi1 = unwrap_radians_near(phi0, float(pt1.phi))

    return ParamPoint(
        t=0.5 * (float(pt0.t) + float(pt1.t)),
        phi=0.5 * (phi0 + phi1),
        mode="surface",
        comment=name,
        track_sign=float(pt1.track_sign),
        track_mode=str(pt1.track_mode),
        track_u=0.5 * (float(pt0.track_u) + float(pt1.track_u)),
        radius_mode=str(pt1.radius_mode),
        c_mode=str(pt1.c_mode),
        c_sweep_u=0.5 * (float(pt0.c_sweep_u) + float(pt1.c_sweep_u)),
        c_sweep_start_deg=pt1.c_sweep_start_deg,
        c_sweep_end_deg=pt1.c_sweep_end_deg,
        c_angle_override_deg=pt1.c_angle_override_deg,
        c_azimuth_sign=float(pt1.c_azimuth_sign),
        c_azimuth_offset_deg=float(pt1.c_azimuth_offset_deg),
        b_radius_override_mm=pt1.b_radius_override_mm,
        b_angle_override_deg=pt1.b_angle_override_deg,
        xyz_override=pt1.xyz_override,
    )


def append_surface_following_segment(
    cache: SurfaceCache,
    p: Params,
    out: List[ParamPoint],
    pt0: ParamPoint,
    pt1: ParamPoint,
    obstacle_ts: np.ndarray,
    obstacle_centers: np.ndarray,
    obstacle_radii: np.ndarray,
    *,
    clearance_mm: float,
    max_depth: int,
    min_param_span: float,
    name: str,
    detour_samples: int = 6,
) -> None:
    a = param_point_to_xyz(cache, p, pt0)
    b = param_point_to_xyz(cache, p, pt1)

    hit, hit_t, _pen = segment_small_tube_hit_info(
        a,
        b,
        obstacle_ts,
        obstacle_centers,
        obstacle_radii,
    )

    if not hit:
        out.append(pt1)
        return

    param_span = abs(float(pt1.t) - float(pt0.t)) + 0.02 * abs(unwrap_radians_near(pt0.phi, pt1.phi) - pt0.phi)

    if max_depth > 0 and param_span > min_param_span:
        mid = midpoint_param_point(pt0, pt1, f"{name} adaptive_mid")
        append_surface_following_segment(
            cache, p, out, pt0, mid,
            obstacle_ts, obstacle_centers, obstacle_radii,
            clearance_mm=clearance_mm,
            max_depth=max_depth - 1,
            min_param_span=min_param_span,
            name=name,
            detour_samples=detour_samples,
        )
        append_surface_following_segment(
            cache, p, out, mid, pt1,
            obstacle_ts, obstacle_centers, obstacle_radii,
            clearance_mm=clearance_mm,
            max_depth=max_depth - 1,
            min_param_span=min_param_span,
            name=name,
            detour_samples=detour_samples,
        )
        return

    out.extend(
        make_outer_ring_avoidance_arc(
            cache,
            p,
            pt0,
            pt1,
            hit_t,
            clearance_mm,
            name,
            detour_samples=detour_samples,
            obstacle_centers=obstacle_centers,
            obstacle_radii=obstacle_radii,
        )
    )


def collision_aware_outer_polyline(
    cache: SurfaceCache,
    p: Params,
    poly: ParamPolyline,
    printed_small_tube_intervals: Sequence[Tuple[float, float]],
    *,
    clearance_mm: float = 0.75,
    obstacle_samples: int = 240,
    detour_samples: int = 4,
) -> ParamPolyline:
    obstacle_ts, centers, radii = build_small_tube_obstacle_samples(
        cache,
        p,
        printed_small_tube_intervals,
        samples_per_interval=obstacle_samples,
        clearance_mm=clearance_mm,
    )

    if len(poly.points) < 2:
        return poly

    new_pts: List[ParamPoint] = [poly.points[0]]

    for pt0, pt1 in zip(poly.points[:-1], poly.points[1:]):
        append_surface_following_segment(
            cache,
            p,
            new_pts,
            pt0,
            pt1,
            obstacle_ts,
            centers,
            radii,
            clearance_mm=clearance_mm,
            max_depth=1,
            min_param_span=8e-4,
            name=poly.name,
            detour_samples=detour_samples,
        )

    return ParamPolyline(name=f"{poly.name}_surface_following", points=new_pts)


def t_before_by_spine_distance(cache: SurfaceCache, t_end: float, distance_mm: float) -> float:
    s_end = spine_arclength_mm(cache, t_end)
    s0 = max(float(cache.spine_s_mm[0]), s_end - max(0.0, float(distance_mm)))
    return clamp01(t_for_spine_arclength_mm(cache, s0))


def make_final_pass_t_stations(
    cache: SurfaceCache,
    p: Params,
    t0: float,
    t1: float,
) -> List[float]:
    """
    Stations from t0 to t1, spaced approximately by continuous_layer_height
    along the spine, then returned in normal order.
    """
    s0 = spine_arclength_mm(cache, t0)
    s1 = spine_arclength_mm(cache, t1)
    span = max(0.0, s1 - s0)

    h = max(0.05, float(p.continuous_layer_height))
    steps = max(1, int(round(span / h)))

    ss = np.linspace(s0, s1, steps + 1)
    return [clamp01(t_for_spine_arclength_mm(cache, float(s))) for s in ss]


def ring_half_phi_arrays_for_bottom_to_top(
    cache: SurfaceCache,
    p: Params,
    t: float,
    arc_segments: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split a constant-t ring into two halves.
    - start point: bottom of the ring, defined as minimum global Y
    - top point: top of the ring, defined as maximum global Z
    Returned arrays both run bottom->top, one on each side of the ring.
    """
    n = int(max(4, arc_segments))
    phi_bottom = ring_extreme_phi(cache, p, t, "outer", axis=1, find_min=True)
    phi_top = unwrap_phi_near(phi_bottom, ring_extreme_phi(cache, p, t, "outer", axis=2, find_min=False))

    side_a = np.linspace(phi_bottom, phi_top, n + 1)
    if phi_top >= phi_bottom:
        side_b_top_to_bottom = np.linspace(phi_top, phi_bottom + 2.0 * math.pi, n + 1)
    else:
        side_b_top_to_bottom = np.linspace(phi_top, phi_bottom - 2.0 * math.pi, n + 1)
    side_b = side_b_top_to_bottom[::-1]  # bottom->top on the opposite side.
    return side_a, side_b


def make_final_junction_layer_polyline(
    t: float,
    first_bottom_to_top: Sequence[float],
    second_top_to_bottom: Sequence[float],
    layer_idx: int,
    p: Params,
) -> ParamPolyline:
    """
    Final small-tube junction strategy:
    print top-down layers with a shallow B angle. Each layer starts at the
    bottom of the ring, traces one half to the top, does a bounded 180-degree C
    spin at the top, then traces the other half back to the bottom. Adjacent
    layers swap the first/second side so the next first half is on the same side
    as the previous layer's last half.
    """
    spin_dir = 1.0 if int(layer_idx) % 2 == 0 else -1.0
    b_angle = clamp(float(p.final_pass_b_angle_deg), 0.0, 90.0)
    pts: List[ParamPoint] = []

    def add_point(phi: float, c_offset: float, comment: str) -> None:
        c_val = surface_azimuth_c_deg(float(phi), sign=1.0, offset_deg=float(c_offset))
        pts.append(
            ParamPoint(
                t=float(t),
                phi=float(phi),
                mode="surface",
                comment=comment,
                track_sign=1.0,
                track_mode="fixed_down",
                radius_mode="outer",
                c_mode="explicit_value",
                c_angle_override_deg=c_val,
                b_angle_override_deg=b_angle,
            )
        )

    first = [float(x) for x in first_bottom_to_top]
    second = [float(x) for x in second_top_to_bottom]

    for i, phi in enumerate(first):
        add_point(phi, 0.0, f"final_junction_layer_{layer_idx} first_half bottom->top i={i}/{len(first)-1} B={b_angle:.1f}")

    top_phi = first[-1]
    add_point(
        top_phi,
        180.0 * spin_dir,
        f"final_junction_layer_{layer_idx} bounded_C_spin_{'+180' if spin_dir > 0 else '-180'}_at_top B={b_angle:.1f}",
    )

    for i, phi in enumerate(second[1:], start=1):
        add_point(
            phi,
            180.0 * spin_dir,
            f"final_junction_layer_{layer_idx} second_half top->bottom i={i}/{len(second)-1} B={b_angle:.1f}",
        )

    return ParamPolyline(name=f"final_junction_layer_{layer_idx}", points=pts)


def build_final_pass_polylines(
    cache: SurfaceCache,
    p: Params,
    t_final_switch: float,
    t_start: float,
) -> List[ParamPolyline]:
    """
    Final junction between the two small tubes: shallow-tip, top-down ring
    layers split into two halves with a bounded 180-degree C spin at the top.
    """
    stations = make_final_pass_t_stations(cache, p, t_final_switch, t_start)
    stations = list(reversed(stations))

    polylines: List[ParamPolyline] = []
    for layer_idx, t in enumerate(stations):
        side_a, side_b = ring_half_phi_arrays_for_bottom_to_top(
            cache,
            p,
            t,
            int(p.final_pass_arc_segments),
        )
        if layer_idx % 2 == 0:
            first = side_a                 # bottom->top
            second = side_b[::-1]          # top->bottom on opposite side
        else:
            first = side_b                 # bottom->top on same side as previous last half
            second = side_a[::-1]          # top->bottom on opposite side
        polylines.append(make_final_junction_layer_polyline(t, first, second, layer_idx, p))
    return polylines


def build_lattice_polylines(p: Params) -> List[ParamPolyline]:
    if not int(p.enable_lattice):
        return []
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


def layer_t_stations_by_spine_spacing(
    cache: SurfaceCache,
    t0: float,
    t1: float,
    layer_height_mm: float,
) -> List[float]:
    """Return t stations from t0 to t1 spaced by approximate spine distance."""
    t0 = clamp01(t0)
    t1 = clamp01(t1)
    s0 = spine_arclength_mm(cache, t0)
    s1 = spine_arclength_mm(cache, t1)
    span = abs(s1 - s0)
    h = max(0.05, float(layer_height_mm))
    steps = max(1, int(round(span / h)))
    ss = np.linspace(s0, s1, steps + 1)
    return [clamp01(t_for_spine_arclength_mm(cache, float(s))) for s in ss]


def surface_radius_for_mode_and_blend(t: float, p: Params, radius_mode: str, blend_u: float = 0.0) -> float:
    mode = str(radius_mode)
    if mode == "inner":
        return float(body_radius_profile(t, p))
    if mode == "blend":
        a = 0.5 - 0.5 * math.cos(math.pi * clamp01(blend_u))
        return float((1.0 - a) * body_radius_profile(t, p) + a * radius_profile(t, p))
    return float(radius_profile(t, p))


def make_circle_layer_polyline(
    t: float,
    p: Params,
    layer_idx: int,
    name: str,
    *,
    radius_mode: str = "inner",
    blend_u: float = 0.0,
    target_radius_mm: Optional[float] = None,
    b_angle_override_deg: Optional[float] = None,
    azimuth_flip: bool = False,
) -> ParamPolyline:
    """Make one bounded-C circle layer.

    The geometric ring alternates direction layer-to-layer and the commanded C
    angle follows the same explicit sweep: even layers C_start -> C_end, odd
    layers C_end -> C_start. This prevents any C unwind/reposition move between
    short-tube layers because the next layer begins at the C value where the
    previous layer ended.

    The physical/geometric azimuth plane is offset separately from the commanded
    C sweep. With the default circle_layer_c_plane_offset_deg=-180, the tool
    writes the opposite C plane while the emitted C commands still monotonically
    sweep -180->180 or 180->-180. This is the requested plane flip without any
    C unwind/reposition move. The legacy azimuth_flip argument is kept only as
    a marker for callers; it no longer mirrors the direction with sign=-1.
    """
    m = int(max(8, p.base_ring_segments))
    reverse = bool(int(layer_idx) % 2)
    c0 = float(p.c_layer_end_deg if reverse else p.c_layer_start_deg)
    c1 = float(p.c_layer_start_deg if reverse else p.c_layer_end_deg)
    az_sign = 1.0
    c_plane_offset = float(getattr(p, "circle_layer_c_plane_offset_deg", -180.0 if bool(azimuth_flip) else 0.0))
    phi0 = phi_from_surface_azimuth_c_deg(c0, sign=az_sign, offset_deg=c_plane_offset)
    phi1 = phi_from_surface_azimuth_c_deg(c1, sign=az_sign, offset_deg=c_plane_offset)
    phis = np.linspace(phi0, phi1, m + 1)
    spin_sign = 1.0 if phi1 >= phi0 else -1.0
    if target_radius_mm is None:
        target_radius_mm = surface_radius_for_mode_and_blend(float(t), p, radius_mode, blend_u)
    pts: List[ParamPoint] = []
    for i, phi in enumerate(phis):
        u = 0.0 if m == 0 else i / m
        pts.append(
            ParamPoint(
                t=float(t),
                phi=float(phi),
                mode="surface",
                comment=f"{name} layer={layer_idx} C={c0:.1f}->{c1:.1f} plane_offset={c_plane_offset:.1f} i={i}/{m}",
                track_sign=spin_sign,
                track_mode="ring_tangent",
                track_u=float(blend_u),
                radius_mode=str(radius_mode),
                # Short-tube/base circle layers use direct bounded C sweeps.
                # Do not derive C from tangent here; alternate full-circle sweeps
                # layer-to-layer so no C unwind/reposition move is needed.
                c_mode="explicit_sweep",
                c_sweep_u=float(u),
                c_sweep_start_deg=c0,
                c_sweep_end_deg=c1,
                c_azimuth_sign=az_sign,
                c_azimuth_offset_deg=c_plane_offset,
                b_radius_override_mm=float(target_radius_mm),
                b_angle_override_deg=None if b_angle_override_deg is None else float(b_angle_override_deg),
            )
        )
    return ParamPolyline(name=name, points=pts)


def make_small_tube_circle_layer_polylines(
    cache: SurfaceCache,
    p: Params,
    t0: float,
    t1: float,
    name_prefix: str,
    *,
    first_layer_idx: int = 0,
) -> List[ParamPolyline]:
    stations = layer_t_stations_by_spine_spacing(cache, t0, t1, p.continuous_layer_height)
    out: List[ParamPolyline] = []
    for i, t in enumerate(stations):
        layer_idx = int(first_layer_idx) + i
        target_radius = body_radius_profile(float(t), p)
        out.append(
            make_circle_layer_polyline(
                float(t),
                p,
                layer_idx,
                f"{name_prefix}_circle_layer_{i}",
                radius_mode="inner",
                target_radius_mm=target_radius,
                b_angle_override_deg=float(p.small_tube_b_angle_deg),
                azimuth_flip=True,
            )
        )
    return out




def outer_shell_b_angle_for_t(cache: SurfaceCache, p: Params, t: float, t_final_switch: float) -> float:
    """Ramp B earlier before the outer skin reaches the small-tube seam."""
    base_b = clamp(float(p.outer_shell_b_angle_deg), 0.0, 180.0)
    pre_seam_b = clamp(float(p.outer_shell_pre_seam_b_angle_deg), 0.0, 180.0)
    ramp_mm = max(0.0, float(p.outer_shell_b_ramp_distance_mm))
    if ramp_mm <= 1e-9:
        return base_b
    s = spine_arclength_mm(cache, clamp01(t))
    s_end = spine_arclength_mm(cache, clamp01(t_final_switch))
    dist_to_seam = max(0.0, s_end - s)
    a = smootherstep(1.0 - clamp01(dist_to_seam / ramp_mm))
    return (1.0 - a) * base_b + a * pre_seam_b
def make_base_radius_transition_circle_layers(
    p: Params,
    *,
    first_layer_idx: int = 0,
) -> List[ParamPolyline]:
    r0 = body_radius_profile(0.0, p)
    r1 = radius_profile(0.0, p)
    dr = abs(float(r1) - float(r0))
    step = max(0.05, float(p.base_transition_ring_step_mm))
    count = max(2, int(math.ceil(dr / step)) + 1)
    out: List[ParamPolyline] = []
    for j in range(count):
        u = 0.0 if count <= 1 else j / (count - 1)
        a = 0.5 - 0.5 * math.cos(math.pi * clamp01(u))
        target_radius = (1.0 - a) * float(r0) + a * float(r1)
        b_angle = (1.0 - a) * float(p.small_tube_b_angle_deg) + a * float(p.outer_shell_b_angle_deg)
        out.append(
            make_circle_layer_polyline(
                0.0,
                p,
                int(first_layer_idx) + j,
                f"base_radius_transition_circle_layer_{j}",
                radius_mode="blend",
                blend_u=float(u),
                target_radius_mm=target_radius,
                b_angle_override_deg=b_angle,
            )
        )
    return out


def build_param_polylines(
    cache: SurfaceCache,
    p: Params,
    *,
    apply_collision_avoidance: bool = True,
) -> List[ParamPolyline]:
    lat0 = clamp01(min(float(p.lattice_start), float(p.lattice_end)))
    lat1 = clamp01(max(float(p.lattice_start), float(p.lattice_end)))
    lattice_enabled = int(p.enable_lattice) and (lat1 > lat0 + 1e-6)
    base_frac = base_tube_fraction(p)
    t_join = return_base_fillet_intersection_t(p)
    t_start = offset_t_up_return_branch(cache, p, t_join, p.return_start_offset_mm)
    if int(p.final_pass_enable):
        t_final_switch = t_before_by_spine_distance(
            cache,
            t_start,
            float(p.final_pass_length_mm),
        )
    else:
        t_final_switch = t_start

    steps_total = int(max(100, p.path_steps))
    inner_return_steps = max(2, int(round(steps_total * max(0.0, 1.0 - t_start))))
    inner_base_steps = max(2, int(round(steps_total * max(0.0, base_frac))))
    base_turn_steps = max(12, int(round(0.5 * max(16, p.base_ring_segments))))

    polylines: List[ParamPolyline] = []
    outer_phase_offset = 0.0

    # Starting small tube: print constant-t circle layers. The geometric ring
    # and commanded C both alternate layer-to-layer: -180 -> 180, then
    # 180 -> -180. This keeps C bounded and avoids C reposition/unwind moves.
    layer_idx = 0
    inner_return_layers = make_small_tube_circle_layer_polylines(
        cache,
        p,
        t_start,
        1.0,
        "inner_return_small_tube",
        first_layer_idx=layer_idx,
    )
    polylines.extend(inner_return_layers)
    layer_idx += len(inner_return_layers)

    inner_base_layers = make_small_tube_circle_layer_polylines(
        cache,
        p,
        base_frac,
        0.0,
        "inner_base_vertical_tube",
        first_layer_idx=layer_idx,
    )
    polylines.extend(inner_base_layers)
    layer_idx += len(inner_base_layers)

    tail_poly_idx = len(polylines) - 1 if inner_base_layers else None
    t_blend_end = base_fillet_intersection_t(p)
    blend_samples = max(16, int(round(max(0.25, t_blend_end) * float(p.base_ring_segments))))
    blend_poly: Optional[ParamPolyline] = None
    if inner_base_layers and t_blend_end > 1e-6:
        blend_poly = make_small_tube_to_outer_shell_blend_polyline(
            cache,
            p,
            inner_base_layers[-1].points[-1],
            t_blend_end,
            t_final_switch,
            blend_samples,
            "small_tube_to_outer_shell_blend",
        )

    join_ignore_mm = 1.0
    s_start = spine_arclength_mm(cache, t_start)
    s_obstacle_start = min(
        spine_arclength_mm(cache, 1.0),
        s_start + join_ignore_mm,
    )
    t_obstacle_start = t_for_spine_arclength_mm(cache, s_obstacle_start)
    printed_small_tube_intervals = [
        (t_obstacle_start, 1.0),
        (0.0, base_frac),
    ]

    def maybe_collision_aware(poly: ParamPolyline) -> ParamPolyline:
        if not apply_collision_avoidance:
            return poly
        return collision_aware_outer_polyline(
            cache,
            p,
            poly,
            printed_small_tube_intervals,
            clearance_mm=float(p.small_tube_collision_clearance_mm),
            obstacle_samples=240,
            detour_samples=int(p.small_tube_detour_samples),
        )

    if not lattice_enabled:
        if t_final_switch > t_blend_end + 1e-6:
            outer_shell = make_shell_polyline(
                cache,
                p,
                t_blend_end,
                t_final_switch,
                max(2, int(round(steps_total * max(0.0, t_final_switch - t_blend_end)))),
                "outer_shell_to_final_switch",
                phase_offset=outer_phase_offset,
                track_sign=1.0,
                radius_mode="outer",
                track_mode="fixed_down",
                b_angle_for_t=lambda tt: outer_shell_b_angle_for_t(cache, p, tt, t_final_switch),
                c_mode="path_tangent_plane_flipped",
            )
            outer_shell = maybe_collision_aware(outer_shell)
            if blend_poly is not None:
                outer_shell = ParamPolyline(
                    name=outer_shell.name,
                    points=blend_poly.points + outer_shell.points[1:],
                )
            if tail_poly_idx is not None:
                polylines[tail_poly_idx] = ParamPolyline(
                    name="inner_base_to_outer_shell",
                    points=polylines[tail_poly_idx].points + outer_shell.points[1:],
                )
            else:
                polylines.append(outer_shell)
        elif blend_poly is not None and tail_poly_idx is not None:
            polylines[tail_poly_idx] = ParamPolyline(
                name="inner_base_to_outer_shell",
                points=polylines[tail_poly_idx].points + blend_poly.points[1:],
            )
        elif blend_poly is not None:
            polylines.append(blend_poly)
        if int(p.final_pass_enable) and t_final_switch < t_start - 1e-6:
            polylines.extend(
                build_final_pass_polylines(
                    cache,
                    p,
                    t_final_switch,
                    t_start,
                )
            )
        return polylines

    shell_pre_end = min(lat0, t_final_switch)
    if shell_pre_end > t_blend_end + 1e-6:
        shell_pre = make_shell_polyline(
            cache,
            p,
            t_blend_end,
            shell_pre_end,
            max(2, int(round(steps_total * max(0.0, shell_pre_end - t_blend_end)))),
            "shell_pre_lattice",
            phase_offset=outer_phase_offset,
            track_sign=1.0,
            radius_mode="outer",
            track_mode="fixed_down",
            b_angle_for_t=lambda tt: outer_shell_b_angle_for_t(cache, p, tt, t_final_switch),
            c_mode="path_tangent_plane_flipped",
        )
        shell_pre = maybe_collision_aware(shell_pre)
        if blend_poly is not None:
            shell_pre = ParamPolyline(
                name=shell_pre.name,
                points=blend_poly.points + shell_pre.points[1:],
            )
            blend_poly = None
        if tail_poly_idx is not None:
            polylines[tail_poly_idx] = ParamPolyline(
                name="inner_base_to_outer_shell",
                points=polylines[tail_poly_idx].points + shell_pre.points[1:],
            )
        else:
            polylines.append(shell_pre)
    elif blend_poly is not None and tail_poly_idx is not None:
        polylines[tail_poly_idx] = ParamPolyline(
            name="inner_base_to_outer_shell",
            points=polylines[tail_poly_idx].points + blend_poly.points[1:],
        )
        blend_poly = None
    elif blend_poly is not None:
        polylines.append(blend_poly)
        blend_poly = None

    if lat0 < t_final_switch and lat1 > lat0:
        polylines.extend(build_lattice_polylines(p))

    if lat1 < t_final_switch - 1e-6:
        shell_post = make_shell_polyline(
            cache,
            p,
            lat1,
            t_final_switch,
            max(2, int(round(steps_total * max(0.0, t_final_switch - lat1)))),
            "shell_post_lattice_to_final_switch",
            phase_offset=outer_phase_offset,
            track_sign=1.0,
            radius_mode="outer",
            track_mode="fixed_down",
            b_angle_for_t=lambda tt: outer_shell_b_angle_for_t(cache, p, tt, t_final_switch),
            c_mode="path_tangent_plane_flipped",
        )
        polylines.append(maybe_collision_aware(shell_post))

    if int(p.final_pass_enable) and t_final_switch < t_start - 1e-6:
        polylines.extend(
            build_final_pass_polylines(
                cache,
                p,
                t_final_switch,
                t_start,
            )
        )

    return polylines


def param_point_to_xyz(cache: SurfaceCache, p: Params, pt: ParamPoint) -> np.ndarray:
    if pt.mode == "cartesian":
        if pt.xyz_override is None:
            raise ValueError("cartesian ParamPoint requires xyz_override")
        return np.asarray(pt.xyz_override, dtype=float)
    if pt.mode == "base_ring":
        t = 0.0
        c = spine_point(cache, t)
        n, b = normal_frame(cache, t)
        r = radius_profile(0.0, p)
        return c + r * (math.cos(pt.phi) * n + math.sin(pt.phi) * b)
    if pt.mode == "surface":
        return point_on_surface_with_radius(cache, p, pt.t, pt.phi, pt.radius_mode, pt.track_u)
    raise ValueError(f"Unknown point mode: {pt.mode}")


def param_point_to_direction(cache: SurfaceCache, p: Params, pt: ParamPoint) -> np.ndarray:
    tool_sign = 1.0 if int(p.tool_normal_sign) >= 0 else -1.0
    if pt.mode in {"base_ring", "surface", "cartesian"}:
        if pt.track_mode == "fixed_down":
            return _normalize(tool_sign * np.array([0.0, 0.0, -1.0], dtype=float))
        centerline_t = tangent_dir(cache, pt.t)
        if pt.track_mode == "ring_tangent":
            n, b = normal_frame(cache, pt.t)
            ring_tangent = _normalize(-math.sin(pt.phi) * n + math.cos(pt.phi) * b)
            return _normalize(tool_sign * float(pt.track_sign) * ring_tangent)
        if pt.track_mode == "bottom_turn":
            n, b = normal_frame(cache, pt.t)
            ring_tangent = _normalize(-math.sin(pt.phi) * n + math.cos(pt.phi) * b)

            # Smoothly rotate from the inner/down orientation (-T)
            # to the outer/large-tube tangent orientation (+T).
            a = smootherstep(clamp01(pt.track_u))

            d = (
                math.cos(math.pi * a) * (-centerline_t)
                + float(pt.track_sign) * math.sin(math.pi * a) * ring_tangent
            )
            return _normalize(tool_sign * d)
        return _normalize(tool_sign * float(pt.track_sign) * centerline_t)
    raise ValueError(f"Unknown point mode: {pt.mode}")


def polyline_to_xyz(cache: SurfaceCache, p: Params, poly: ParamPolyline) -> List[np.ndarray]:
    return [param_point_to_xyz(cache, p, pt) for pt in poly.points]


def polyline_to_waypoints(
    cache: SurfaceCache,
    p: Params,
    poly: ParamPolyline,
    inverter: TipAngleInverter,
    c_start_deg: float,
    radius_inverter: Optional[BRadiusInverter] = None,
    b_start_cmd: Optional[float] = None,
) -> List[Waypoint]:
    waypoints: List[Waypoint] = []
    prev_c = bounded_equivalent_degrees(float(c_start_deg), None, -180.0, 180.0)
    prev_b: Optional[float] = None if b_start_cmd is None else float(b_start_cmd)

    xyzs = [param_point_to_xyz(cache, p, pt) for pt in poly.points]

    def local_path_tangent(i: int, fallback: np.ndarray) -> np.ndarray:
        if len(xyzs) >= 2:
            if i <= 0:
                d = xyzs[1] - xyzs[0]
            elif i >= len(xyzs) - 1:
                d = xyzs[-1] - xyzs[-2]
            else:
                d = xyzs[i + 1] - xyzs[i - 1]
            if float(np.linalg.norm(d)) > 1e-9:
                return _normalize(d)
        return _normalize(fallback)

    def tangent_plane_c(tangent: np.ndarray, prev: float, flipped: bool = False) -> float:
        # The C plane convention used by the earlier script came from the
        # extrusion tangent. The requested correction is that same tangent-plane
        # schedule, flipped by 180 degrees. A GUI offset is left exposed in case
        # the physical C-plane convention needs a small machine-specific shift.
        _tilt_unused, tangent_az = direction_to_tilt_azimuth_deg(tangent, prev)
        offset = float(getattr(p, "c_tangent_plane_offset_deg", 180.0 if flipped else 0.0))
        if not flipped:
            offset = 0.0
        return bounded_equivalent_degrees(tangent_az + offset, prev, -180.0, 180.0)

    for idx, pt in enumerate(poly.points):
        xyz = xyzs[idx]
        direction = param_point_to_direction(cache, p, pt)
        tilt, az_from_direction = direction_to_tilt_azimuth_deg(direction, prev_c)
        path_tangent = local_path_tangent(idx, direction)

        c_mode = str(pt.c_mode).strip().lower()
        if c_mode == "explicit_sweep":
            c0 = float(p.c_layer_start_deg if pt.c_sweep_start_deg is None else pt.c_sweep_start_deg)
            c1 = float(p.c_layer_end_deg if pt.c_sweep_end_deg is None else pt.c_sweep_end_deg)
            u = clamp01(float(pt.c_sweep_u))
            az = (1.0 - u) * c0 + u * c1
            az = clamp_degrees(az, -180.0, 180.0)
        elif c_mode == "explicit_value":
            if pt.c_angle_override_deg is None:
                raise ValueError("explicit_value C mode requires c_angle_override_deg")
            az = clamp_degrees(float(pt.c_angle_override_deg), -180.0, 180.0)
        elif c_mode in {"path_tangent_plane", "path_tangent"}:
            az = tangent_plane_c(path_tangent, prev_c, flipped=False)
        elif c_mode in {"path_tangent_plane_flipped", "path_tangent_flipped", "tangent_flipped"}:
            az = tangent_plane_c(path_tangent, prev_c, flipped=True)
        elif c_mode == "surface_azimuth":
            az = surface_azimuth_c_deg(
                pt.phi,
                sign=float(pt.c_azimuth_sign),
                offset_deg=float(pt.c_azimuth_offset_deg),
            )
            az = bounded_equivalent_degrees(az, prev_c, -180.0, 180.0)
        elif c_mode == "bounded":
            az = bounded_equivalent_degrees(az_from_direction, prev_c, -180.0, 180.0)
        else:
            # Keep all C output bounded even for the older continuous modes.
            az = bounded_equivalent_degrees(unwrap_degrees(prev_c, az_from_direction), prev_c, -180.0, 180.0)

        prev_c = az

        comment = pt.comment
        if pt.b_angle_override_deg is not None:
            target_b_angle = float(pt.b_angle_override_deg)
            b_cmd, clamped = inverter.angle_to_b(target_b_angle, prefer_b=prev_b)
            if clamped:
                comment += f" (B angle clamped: requested {target_b_angle:.1f})"
        elif pt.b_radius_override_mm is not None:
            if radius_inverter is None:
                raise ValueError("A radius inverter is required for b_radius_override_mm toolpaths.")
            inv = radius_inverter.radius_to_b(float(pt.b_radius_override_mm))
            b_cmd = inv.b_cmd
            if inv.clamped:
                comment += f" (B radius clamped: requested {float(pt.b_radius_override_mm):.3f}, actual {inv.radius_actual:.3f})"
        else:
            b_cmd, clamped = inverter.angle_to_b(tilt, prefer_b=prev_b)
            if clamped:
                comment += " (B clamped)"
        prev_b = float(b_cmd)
        waypoints.append(Waypoint(tip_xyz=xyz, b_cmd=float(b_cmd), c_deg=float(az), comment=comment))
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

        self.pressure_charged = False
        self.cur_stage_xyz: Optional[np.ndarray] = None
        self.cur_tip_xyz: Optional[np.ndarray] = None
        self.cur_b = 0.0
        self.cur_c = 0.0
        self.step_counter = 0

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
            (self.cal.c_axis, clamp_degrees(float(c_deg), -180.0, 180.0)),
        ]
        self.f.write(f"G1 {_fmt_axes_move(axes)} F{float(feed):.0f}\n")
        self.cur_stage_xyz = pc.copy()
        self.cur_b = float(b_cmd)
        self.cur_c = clamp_degrees(float(c_deg), -180.0, 180.0)

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
            (self.cal.c_axis, clamp_degrees(float(wp.c_deg), -180.0, 180.0)),
        ]

        if wp.comment and self.debug_every > 0 and (self.step_counter % self.debug_every == 0):
            tip_ang = float(eval_tip_angle_deg(self.cal, wp.b_cmd))
            self.f.write(f"; step={self.step_counter} {wp.comment} | Bcmd={wp.b_cmd:.4f} C={wp.c_deg:.2f} tipAngle={tip_ang:.2f}\n")

        self.f.write(f"G1 {_fmt_axes_move(axes)} F{float(feed):.0f}\n")
        self.cur_stage_xyz = pc.copy()
        self.cur_tip_xyz = tip_xyz.copy()
        self.cur_b = float(wp.b_cmd)
        self.cur_c = clamp_degrees(float(wp.c_deg), -180.0, 180.0)
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
    q.inner_start_offset_mm = max(0.0, float(q.inner_start_offset_mm))
    q.return_start_offset_mm = max(0.0, float(q.return_start_offset_mm))
    q.small_tube_collision_clearance_mm = max(0.0, float(q.small_tube_collision_clearance_mm))
    q.small_tube_detour_samples = int(max(3, min(16, q.small_tube_detour_samples)))
    q.small_tube_b_angle_deg = clamp(float(q.small_tube_b_angle_deg), 0.0, 180.0)
    q.outer_shell_b_angle_deg = clamp(float(q.outer_shell_b_angle_deg), 0.0, 180.0)
    q.outer_shell_pre_seam_b_angle_deg = clamp(float(q.outer_shell_pre_seam_b_angle_deg), 0.0, 180.0)
    q.outer_shell_b_ramp_distance_mm = max(0.0, float(q.outer_shell_b_ramp_distance_mm))
    q.seam_avoidance_bottom_b_angle_deg = clamp(float(q.seam_avoidance_bottom_b_angle_deg), 30.0, 90.0)
    q.seam_avoidance_b_angle_deg = clamp(float(q.seam_avoidance_b_angle_deg), 30.0, 90.0)
    q.final_pass_b_angle_deg = clamp(float(q.final_pass_b_angle_deg), 0.0, 90.0)
    q.final_pass_enable = 1 if int(q.final_pass_enable) else 0
    q.final_pass_length_mm = max(0.0, float(q.final_pass_length_mm))
    q.final_pass_arc_segments = int(max(4, q.final_pass_arc_segments))
    q.final_pass_x_offset_mm = max(0.0, float(q.final_pass_x_offset_mm))
    q.c_tangent_plane_offset_deg = clamp(float(q.c_tangent_plane_offset_deg), -180.0, 180.0)
    q.c_layer_start_deg = clamp_degrees(float(q.c_layer_start_deg), -180.0, 180.0)
    q.c_layer_end_deg = clamp_degrees(float(q.c_layer_end_deg), -180.0, 180.0)
    q.circle_layer_c_plane_offset_deg = clamp_degrees(float(q.circle_layer_c_plane_offset_deg), -180.0, 180.0)
    if abs(q.c_layer_end_deg - q.c_layer_start_deg) < 1e-9:
        q.c_layer_start_deg = -180.0
        q.c_layer_end_deg = 180.0
    q.base_transition_ring_step_mm = max(0.05, float(q.base_transition_ring_step_mm))
    q.tool_normal_sign = 1 if int(q.tool_normal_sign) >= 0 else -1
    q.lattice_start = clamp01(q.lattice_start)
    q.lattice_end = clamp01(q.lattice_end)
    if q.lattice_end < q.lattice_start:
        q.lattice_start, q.lattice_end = q.lattice_end, q.lattice_start
    q.neck_center = clamp01(q.neck_center)
    q.neck_width = max(1e-6, float(q.neck_width))
    q.base_fillet_span = max(0.0, float(q.base_fillet_span))
    q.base_outer_edge_fillet = max(0.0, float(q.base_outer_edge_fillet))
    q.spine_x_scale = float(q.spine_x_scale)
    q.spine_y_scale = float(q.spine_y_scale)
    q.base_tube_radius = max(0.1, float(q.base_tube_radius))
    q.base_tube_height = max(0.0, float(q.base_tube_height))
    q.large_base_height = min(max(0.0, float(q.large_base_height)), float(q.base_tube_height))
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
    radius_inverter = BRadiusInverter(cal, use_offplane=True)
    cache = build_surface_cache(p, samples=int(max(300, p.path_steps // 4)))
    polylines = build_param_polylines(cache, p, apply_collision_avoidance=True)
    bbox = params_bbox(p)
    emit_extrusion = float(p.extrusion_per_mm) != 0.0
    warn_log: List[str] = []

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    with open(out_path, "w", encoding="utf-8") as f:
        total_len = total_print_length(cache, p, polylines)
        f.write("; generated by klein_bottle_gui_slicer.py\n")
        f.write("; calibrated tip-position planning: stage = tip - offset_tip(B_cmd,C)\n")
        f.write("; tip offset equations: y_off=-offplane; Xtip=Xstage+r*cos(C)-y_off*sin(C); Ytip=Ystage+r*sin(C)+y_off*cos(C); Ztip=Zstage+z(B)\n")
        f.write("; small tube/base transition: final small-tube circle layer blends continuously onto the outer shell surface, with bounded alternating C only on the circle layers\n")
        f.write("; small tube and outer shell use tangent-plane C scheduling with a 180-degree flip; seam avoidance follows the actual seam edge\n")
        f.write("; extrusion actuation: pressure solenoid valve via M42 P0 S1 / M42 P0 S0\n")
        f.write(f"; axes: X->{cal.x_axis}, Y->{cal.y_axis}, Z->{cal.z_axis}, B->{cal.b_axis}, C->{cal.c_axis}\n")
        f.write(f"; calibration B range (command units): [{cal.b_min:.6f}, {cal.b_max:.6f}]\n")
        f.write(f"; selected_offplane_fit_model={cal.active_offplane_fit_model or cal.selected_offplane_fit_model or cal.selected_fit_model or 'avg_cubic'} active_phase={cal.active_phase}\n")
        f.write(f"; start tip-space base = [{p.start_x:.3f}, {p.start_y:.3f}, {p.start_z:.3f}]\n")
        f.write(f"; spine: height={p.spine_height:.3f}, base_tube_height={p.base_tube_height:.3f}, r0={p.spine_r0:.3f}, r1={p.spine_r1:.3f}, s={p.spine_s:.3f}, x_scale={p.spine_x_scale:.3f}, y_scale={p.spine_y_scale:.3f}, z_wobble={p.spine_z_wobble:.3f}\n")
        f.write(f"; twist_deg={p.twist_deg:.3f}, continuous_layer_height={p.continuous_layer_height:.3f}, shell_phase_deg={p.shell_phase_deg:.3f}\n")
        f.write(f"; radii: large_base={p.large_base_radius:.3f}, large_base_height={p.large_base_height:.3f}, base_tube={p.base_tube_radius:.3f}, neck={p.neck_radius:.3f}\n")
        f.write(f"; neck: center={p.neck_center:.3f}, width={p.neck_width:.3f}, power={p.neck_power:.3f}\n")
        f.write(f"; base fillets: base_fillet_span={p.base_fillet_span:.3f}, outer_edge_fillet={p.base_outer_edge_fillet:.3f}, outer_edge_span=base_tube_radius({p.base_tube_radius:.3f})\n")
        f.write(f"; lattice: enabled={int(p.enable_lattice)} start={p.lattice_start:.3f}, end={p.lattice_end:.3f}, u={p.lattice_u_count}, v={p.lattice_v_count}\n")
        f.write(f"; lengths: approx_total_print_length={total_len:.3f} mm, approx_total_extrusion={total_len * p.extrusion_per_mm:.3f}\n")
        f.write(f"; feeds: travel={p.travel_feed:.1f}, print={p.print_feed:.1f}\n")
        f.write(f"; tool_normal_sign={p.tool_normal_sign}\n")
        f.write(f"; short-tube/base circle C: even layers {p.c_layer_start_deg:.3f}->{p.c_layer_end_deg:.3f}, odd layers {p.c_layer_end_deg:.3f}->{p.c_layer_start_deg:.3f}; geometric C-plane offset={p.circle_layer_c_plane_offset_deg:.3f} deg; no C unwind/reposition between adjacent layers; bounded to [-180, 180]\n")
        f.write(f"; B posture targets: small_tube={p.small_tube_b_angle_deg:.1f} deg, outer_shell={p.outer_shell_b_angle_deg:.1f} deg, seam_avoidance={p.seam_avoidance_b_angle_deg:.1f} deg\n")
        f.write(f"; radius inversion range for optional B extension: [{radius_inverter.r_min:.3f}, {radius_inverter.r_max:.3f}] mm\n")
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

        # No forced machine-start reposition is emitted here. The first emitted
        # motion is the normal travel-to-start for the first print polyline.
        # Seed B/C internally only so branch-continuous inverse kinematics and
        # bounded C scheduling start from the configured machine posture.
        g.cur_b = float(p.machine_start_b)
        g.cur_c = clamp_degrees(float(p.machine_start_c), -180.0, 180.0)

        c_seed = float(p.machine_start_c)
        b_seed = float(p.machine_start_b)
        total_waypoints = 0
        for poly in polylines:
            waypoints = polyline_to_waypoints(
                cache,
                p,
                poly,
                inverter,
                c_seed,
                radius_inverter=radius_inverter,
                b_start_cmd=b_seed,
            )
            if not waypoints:
                continue
            c_seed = float(waypoints[-1].c_deg)
            b_seed = float(waypoints[-1].b_cmd)
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
        self.enable_lattice_var = tk.IntVar(value=int(self.params.enable_lattice))
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
        ttk.Checkbutton(toggles, text="Enable lattice", variable=self.enable_lattice_var, command=self._toggle_changed).grid(row=2, column=0, sticky="w")
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
        self.params.enable_lattice = int(self.enable_lattice_var.get())
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
        self.enable_lattice_var.set(int(self.params.enable_lattice))
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
            self.params.enable_lattice = int(self.enable_lattice_var.get())
            self.params.lattice_include_rings = int(self.lattice_rings_var.get())
            self.params.lattice_include_meridians = int(self.lattice_meridians_var.get())
            self.params.lattice_include_diagonals = int(self.lattice_diagonals_var.get())
            self.params = sanitize_params(self.params)

            cache = build_surface_cache(self.params, self.params.preview_spine_samples)
            polylines = build_param_polylines(
                cache,
                self.params,
                apply_collision_avoidance=False,
            )
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
            f"Spine X bend scale: {self.params.spine_x_scale:.2f} | Spine Y bend scale: {self.params.spine_y_scale:.2f} | Lattice enabled: {int(self.params.enable_lattice)}\n"
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
            self.params.enable_lattice = int(self.enable_lattice_var.get())
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
