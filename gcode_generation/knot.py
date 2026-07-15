#!/usr/bin/env python3
"""
mesh_centerline_pretzel_knot_gui_v13.py

GUI + headless G-code generator for tracing a knot centerline extracted from
an STL or 3MF mesh.

Why this version exists
-----------------------
Instead of hand-fitting the pretzel / overhand knot with Bezier controls, this
script can load a mesh of the correct knot, extract the tube centerline, scale
and place that centerline in robot tip space, then export Duet/RRF-friendly
G-code.  The mesh is treated as the source of truth for the geometry.

Supported workflow
------------------
1) Load a watertight STL or 3MF of the knot tube.  A swept tube with caps works
   best.  The default tube diameter helper is 3 mm, but the algorithm uses the
   mesh volume / medial skeleton rather than relying on exact ring topology.
2) By default, compute surface-geodesic rings from the low-Z cap to the high-Z cap
   and average each ring to recover the centerline. This is better for constant-diameter
   pretzel/overhand tubes because it follows the mesh topology instead of jumping
   through close self-contact regions. Voxel distance-path and skeleton fallbacks are
   still available.
3) Scale, rotate, reverse, and place the extracted centerline.
4) Optionally force the bottom and top branch Y offsets; defaults are +15 mm
   for the bottom branch and -15 mm for the top branch.
5) Export calibrated G-code using your robot calibration JSON, or Cartesian
   G-code for preview/testing.

Notes and limitations
---------------------
- For centerline extraction, STL is fine.  3MF is also supported through
  trimesh and can preserve units better, but both are reduced to triangles.
- Best input: one connected, watertight, capped tube mesh.  If the STL is only a
  thin shell or has holes, extraction can still work, but voxel fill may need a
  smaller pitch or mesh repair in your CAD/slicer.
- The centerline is a geometric path only.  B and C are generated from smooth
  arclength schedules that you can tune in the GUI.  Defaults use C0 -> C-180 and hold C-180 on the last pass.
- In v13, trace_scale is still a true final uniform scale, but the
  recommended way to make the knot read larger in the XZ projection is the
  new interior XZ projection shaping controls: Interior X scale and Y->X projection gain. v13 adds a final no-wobble vertical-tail lock so the centerline stays exactly vertical until the controlled bend into the knot.  Extraction, smoothing,
  vertical-tail constraints, flare, and marker offsets are first solved at
  the base shape. Use Final path scale only when you truly want the whole path
  larger; use Interior X scale / Y->X projection gain when you want more XZ overlap
  without stretching Z or changing the vertical tails.
- If you have a STEP file that contains the original sweep/spine curve, exporting
  that curve is ideal.  If the STEP is only a solid tube, this script still uses
  the triangle-mesh/surface-geodesic path after export/import.

Dependencies
------------
Python 3.10+
  numpy, scipy, matplotlib, scikit-image, tkinter
  trimesh optional. If unavailable, STL and basic 3MF are read by a built-in fallback loader.

Install if needed:
  python -m pip install numpy scipy matplotlib scikit-image
  # optional but faster/more robust:
  python -m pip install trimesh

Usage
-----
GUI:
  python mesh_centerline_pretzel_knot_gui.py
  python mesh_centerline_pretzel_knot_gui.py --mesh knot.stl --calibration robot_calibration.json

Headless export:
  python mesh_centerline_pretzel_knot_gui.py --nogui --mesh knot.stl --out knot.gcode
  python mesh_centerline_pretzel_knot_gui.py --nogui --cartesian --preview-png preview.png
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import traceback
import struct
import zipfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, asdict, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

_DISABLE_TRIMESH = os.environ.get("MESH_CENTERLINE_NO_TRIMESH", "").strip().lower() in {"1", "true", "yes"}
try:
    if _DISABLE_TRIMESH:
        raise ImportError("trimesh disabled by MESH_CENTERLINE_NO_TRIMESH")
    import trimesh
except Exception as exc:  # pragma: no cover
    trimesh = None
    _TRIMESH_IMPORT_ERROR = exc
else:
    _TRIMESH_IMPORT_ERROR = None

try:
    from skimage.morphology import skeletonize
except Exception as exc:  # pragma: no cover
    skeletonize = None
    _SKIMAGE_IMPORT_ERROR = exc
else:
    _SKIMAGE_IMPORT_ERROR = None

try:
    from scipy import interpolate
    from scipy.ndimage import gaussian_filter1d, binary_fill_holes, binary_closing, distance_transform_edt
    from scipy.sparse import coo_matrix, csr_matrix
    from scipy.sparse.csgraph import connected_components, dijkstra
    from scipy.spatial import cKDTree
except Exception as exc:  # pragma: no cover
    interpolate = None
    gaussian_filter1d = None
    binary_fill_holes = None
    binary_closing = None
    coo_matrix = None
    csr_matrix = None
    connected_components = None
    dijkstra = None
    cKDTree = None
    _SCIPY_IMPORT_ERROR = exc
else:
    _SCIPY_IMPORT_ERROR = None

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
try:
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:  # pragma: no cover
    matplotlib.use("Agg")
    FigureCanvasTkAgg = None
from matplotlib.figure import Figure


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_OUT = "gcode_generation/tie_it.gcode"
DEFAULT_PREVIEW_PNG = "gcode_generation/tie_it.png"
DEFAULT_CALIBRATION_PATH = ""
DEFAULT_MESH_PATH = "gcode_generation/knot_short_longer.stl"

# Tip-space placement.  Z increases upward; the default bottom point is near
# the previously used knot examples.
DEFAULT_START_X = 100.0
DEFAULT_START_Y = 80.0
DEFAULT_START_Z = -150.0
DEFAULT_BOTTOM_BRANCH_Y_OFFSET = 15.0
DEFAULT_TOP_BRANCH_Y_OFFSET = -15.0

# Mesh/extraction.
DEFAULT_MESH_UNIT_SCALE = 1.0
DEFAULT_TRACE_SCALE = 2.5
DEFAULT_MESH_TUBE_DIAMETER = 3.0
DEFAULT_VOXEL_PITCH = 0.55
DEFAULT_EXTRACTION_METHOD = "surface_geodesic"  # surface_geodesic + cross-section refinement is the default; plane_march remains a fallback
DEFAULT_SURFACE_RING_SPACING_MM = 0.45
DEFAULT_SURFACE_RING_BAND_MM = 0.55
# Local cross-section marching extractor.  This is the new default for constant-diameter
# tube meshes because it fits the tube center from actual circular slices instead of
# using surface-distance rings, which can sit on the side wall in tight knots.
DEFAULT_PLANE_MARCH_STEP_MM = 0.45
DEFAULT_PLANE_MARCH_GATE_RADIUS_MM = 4.2
DEFAULT_PLANE_MARCH_CLUSTER_RADIUS_MM = 0.9
DEFAULT_PLANE_MARCH_MAX_STEPS = 2500
DEFAULT_ENDPOINT_SPAN_MM = 2.5
DEFAULT_CENTERLINE_CLEARANCE_WEIGHT = 18.0
DEFAULT_CENTERLINE_CLEARANCE_POWER = 4.0
DEFAULT_SMOOTH_SIGMA_MM = 1.2
DEFAULT_RESAMPLE_SPACING_MM = 0.45
DEFAULT_PATH_DECIMATE_PREVIEW = 1
DEFAULT_SURFACE_GEODESIC_REFINE_SLICES = 1
DEFAULT_SURFACE_GEODESIC_LOCK_XZ = 1
DEFAULT_VIEW_Z_STRETCH = 1.65
DEFAULT_FORCE_BRANCH_Y = 0
DEFAULT_FORCE_COAXIAL_BRANCHES = 0
DEFAULT_VERTICAL_BRANCH_LENGTH_MM = 7.0
DEFAULT_VERTICAL_BRANCH_BLEND_MM = 16.0
DEFAULT_VERTICAL_BRANCH_FLARE_MM = 5.0
DEFAULT_VERTICAL_TAIL_NO_WOBBLE_MM = 12.0
DEFAULT_VERTICAL_TAIL_NO_WOBBLE_FADE_MM = 8.0
DEFAULT_REVERSE_PATH = 0
DEFAULT_ROT_X_DEG = 0.0
DEFAULT_ROT_Y_DEG = 0.0
DEFAULT_ROT_Z_DEG = 0.0

# Motion and B/C schedule.  C0 -> C-180 and hold C-180 by default.
DEFAULT_B_START_DEG = 0.0
DEFAULT_B_MID_DEG = 170.0
DEFAULT_B_END_DEG = -160.0
DEFAULT_B_RAMP_START = 0.18
DEFAULT_B_RAMP_END = 0.40
DEFAULT_B_RELEASE_START = 1.0
DEFAULT_B_RELEASE_END = 1.0
DEFAULT_C_START_DEG = 0.0
DEFAULT_C_END_DEG = -180.0
DEFAULT_C_RAMP_START = 0.50
DEFAULT_C_RAMP_END = 0.70
# Optional top-loop azimuth: gives the still-C0 upper loop some C angle so the robot lies in the print-direction azimuth plane.
DEFAULT_ENABLE_TOP_LOOP_C_AZIMUTH = 1
DEFAULT_C_TOP_LOOP_DEG = 90.0
DEFAULT_C_TOP_LOOP_START = 0.10
DEFAULT_C_TOP_LOOP_END = 0.50

DEFAULT_TRAVEL_LIFT_Z = 8.0
DEFAULT_SAFE_TRAVEL_Z = -30.0
DEFAULT_POST_PRINT_DROP_Z_MM = 30.0
DEFAULT_POST_PRINT_LEAD_X_MM = -3.0
DEFAULT_POST_PRINT_LEAD_Y_MM = -2.0
DEFAULT_POST_PRINT_SHIFT_X_MM = 20.0
DEFAULT_EXIT_LIFT_Z = -10
DEFAULT_EXIT_Y_MM = -30.0
DEFAULT_EXIT_B_DEG = 100.0
DEFAULT_TRAVEL_FEED = 2000.0
DEFAULT_APPROACH_FEED = 400.0
DEFAULT_FINE_APPROACH_FEED = 200.0
DEFAULT_PRINT_FEED = 400.0
DEFAULT_EXTRUSION_PER_MM = 0.0
DEFAULT_PREFLOW_DWELL_MS = 350
DEFAULT_END_DWELL_MS = 1000
DEFAULT_EMIT_PRESSURE = 1
DEFAULT_EMIT_U_EXTRUSION = 0
DEFAULT_PRESSURE_PIN = 0

# Calibration / compensation.
DEFAULT_WRITE_MODE = "calibrated"  # calibrated or cartesian
DEFAULT_FIT_MODEL = "pchip"
DEFAULT_OFFPLANE_FIT_MODEL = "pchip"
DEFAULT_CALIBRATION_PHASE = "auto"
DEFAULT_OFFPLANE_SIGN = -1.0
DEFAULT_BC_SOLVE_SAMPLES = 12001

# Local XZ shaping anchors. These are applied after centerline extraction and
# endpoint/tail constraints, so they scale/shape the path without modifying the mesh.
DEFAULT_ENABLE_MARKER_X_OFFSETS = 0
DEFAULT_TOP_C0_X_OFFSET_MM = -15.0
DEFAULT_BOTTOM_C_TARGET_DEG = -176.0
DEFAULT_BOTTOM_C_TARGET_X_OFFSET_MM = 15.0
DEFAULT_MARKER_X_OFFSET_WIDTH_MM = 7.0

# XZ projection shaping.  These controls are meant to fix the case where the
# extracted centerline is good in 3D, but the knot does not read strongly enough
# in the robot XZ drawing plane.  They are applied only to the interior knot
# region, with smooth fade-in/out near the vertical tails.
DEFAULT_ENABLE_XZ_PROJECTION_SHAPE = 1
DEFAULT_INTERIOR_X_SCALE = 1.0
DEFAULT_Y_TO_X_PROJECTION_GAIN = 0.0
# Symmetric X expansion of upper/lower lobes, independent of marker offsets.
DEFAULT_SYMMETRIC_LOOP_X_SPREAD_MM = 3.0
DEFAULT_SYMMETRIC_LOOP_X_SPREAD_WIDTH_MM = 16.0

# Bounds clamp for stage commands.
DEFAULT_BBOX_X_MIN = -1e9
DEFAULT_BBOX_X_MAX = 1e9
DEFAULT_BBOX_Y_MIN = -1e9
DEFAULT_BBOX_Y_MAX = 1e9
DEFAULT_BBOX_Z_MIN = -1e9
DEFAULT_BBOX_Z_MAX = 1e9

# Preview appearance.
UI_BG = "#000000"
UI_PANEL_BG = "#060606"
UI_ENTRY_BG = "#000000"
UI_FG = "#e6edf3"
UI_MUTED = "#a8b3c2"
UI_ACCENT = "#5ac8fa"
UI_GRID = "#3d4652"
PRESET_FORMAT_VERSION = 1


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class Params:
    mesh_path: str = DEFAULT_MESH_PATH
    calibration_path: str = DEFAULT_CALIBRATION_PATH
    output_path: str = DEFAULT_OUT
    preview_png_path: str = DEFAULT_PREVIEW_PNG

    write_mode: str = DEFAULT_WRITE_MODE
    fit_model: str = DEFAULT_FIT_MODEL
    offplane_fit_model: str = DEFAULT_OFFPLANE_FIT_MODEL
    calibration_phase: str = DEFAULT_CALIBRATION_PHASE
    offplane_sign: float = DEFAULT_OFFPLANE_SIGN
    bc_solve_samples: int = DEFAULT_BC_SOLVE_SAMPLES

    start_x: float = DEFAULT_START_X
    start_y: float = DEFAULT_START_Y
    start_z: float = DEFAULT_START_Z
    bottom_branch_y_offset: float = DEFAULT_BOTTOM_BRANCH_Y_OFFSET
    top_branch_y_offset: float = DEFAULT_TOP_BRANCH_Y_OFFSET
    force_branch_y: int = DEFAULT_FORCE_BRANCH_Y
    force_coaxial_branches: int = DEFAULT_FORCE_COAXIAL_BRANCHES
    vertical_branch_length_mm: float = DEFAULT_VERTICAL_BRANCH_LENGTH_MM
    vertical_branch_blend_mm: float = DEFAULT_VERTICAL_BRANCH_BLEND_MM
    vertical_branch_flare_mm: float = DEFAULT_VERTICAL_BRANCH_FLARE_MM
    vertical_tail_no_wobble_mm: float = DEFAULT_VERTICAL_TAIL_NO_WOBBLE_MM
    vertical_tail_no_wobble_fade_mm: float = DEFAULT_VERTICAL_TAIL_NO_WOBBLE_FADE_MM
    enable_marker_x_offsets: int = DEFAULT_ENABLE_MARKER_X_OFFSETS
    top_c0_x_offset_mm: float = DEFAULT_TOP_C0_X_OFFSET_MM
    bottom_c_target_deg: float = DEFAULT_BOTTOM_C_TARGET_DEG
    bottom_c_target_x_offset_mm: float = DEFAULT_BOTTOM_C_TARGET_X_OFFSET_MM
    marker_x_offset_width_mm: float = DEFAULT_MARKER_X_OFFSET_WIDTH_MM
    enable_xz_projection_shape: int = DEFAULT_ENABLE_XZ_PROJECTION_SHAPE
    interior_x_scale: float = DEFAULT_INTERIOR_X_SCALE
    y_to_x_projection_gain: float = DEFAULT_Y_TO_X_PROJECTION_GAIN
    symmetric_loop_x_spread_mm: float = DEFAULT_SYMMETRIC_LOOP_X_SPREAD_MM
    symmetric_loop_x_spread_width_mm: float = DEFAULT_SYMMETRIC_LOOP_X_SPREAD_WIDTH_MM

    mesh_unit_scale: float = DEFAULT_MESH_UNIT_SCALE
    trace_scale: float = DEFAULT_TRACE_SCALE
    mesh_tube_diameter: float = DEFAULT_MESH_TUBE_DIAMETER
    voxel_pitch: float = DEFAULT_VOXEL_PITCH
    extraction_method: str = DEFAULT_EXTRACTION_METHOD
    endpoint_span_mm: float = DEFAULT_ENDPOINT_SPAN_MM
    surface_ring_spacing_mm: float = DEFAULT_SURFACE_RING_SPACING_MM
    surface_ring_band_mm: float = DEFAULT_SURFACE_RING_BAND_MM
    surface_geodesic_refine_slices: int = DEFAULT_SURFACE_GEODESIC_REFINE_SLICES
    surface_geodesic_lock_xz: int = DEFAULT_SURFACE_GEODESIC_LOCK_XZ
    plane_march_step_mm: float = DEFAULT_PLANE_MARCH_STEP_MM
    plane_march_gate_radius_mm: float = DEFAULT_PLANE_MARCH_GATE_RADIUS_MM
    plane_march_cluster_radius_mm: float = DEFAULT_PLANE_MARCH_CLUSTER_RADIUS_MM
    plane_march_max_steps: int = DEFAULT_PLANE_MARCH_MAX_STEPS
    centerline_clearance_weight: float = DEFAULT_CENTERLINE_CLEARANCE_WEIGHT
    centerline_clearance_power: float = DEFAULT_CENTERLINE_CLEARANCE_POWER
    smooth_sigma_mm: float = DEFAULT_SMOOTH_SIGMA_MM
    resample_spacing_mm: float = DEFAULT_RESAMPLE_SPACING_MM
    reverse_path: int = DEFAULT_REVERSE_PATH
    rot_x_deg: float = DEFAULT_ROT_X_DEG
    rot_y_deg: float = DEFAULT_ROT_Y_DEG
    rot_z_deg: float = DEFAULT_ROT_Z_DEG

    b_start_deg: float = DEFAULT_B_START_DEG
    b_mid_deg: float = DEFAULT_B_MID_DEG
    b_end_deg: float = DEFAULT_B_END_DEG
    b_ramp_start: float = DEFAULT_B_RAMP_START
    b_ramp_end: float = DEFAULT_B_RAMP_END
    b_release_start: float = DEFAULT_B_RELEASE_START
    b_release_end: float = DEFAULT_B_RELEASE_END
    c_start_deg: float = DEFAULT_C_START_DEG
    c_end_deg: float = DEFAULT_C_END_DEG
    c_ramp_start: float = DEFAULT_C_RAMP_START
    c_ramp_end: float = DEFAULT_C_RAMP_END
    enable_top_loop_c_azimuth: int = DEFAULT_ENABLE_TOP_LOOP_C_AZIMUTH
    c_top_loop_deg: float = DEFAULT_C_TOP_LOOP_DEG
    c_top_loop_start: float = DEFAULT_C_TOP_LOOP_START
    c_top_loop_end: float = DEFAULT_C_TOP_LOOP_END

    travel_lift_z: float = DEFAULT_TRAVEL_LIFT_Z
    safe_travel_z: float = DEFAULT_SAFE_TRAVEL_Z
    post_print_drop_z_mm: float = DEFAULT_POST_PRINT_DROP_Z_MM
    post_print_lead_x_mm: float = DEFAULT_POST_PRINT_LEAD_X_MM
    post_print_lead_y_mm: float = DEFAULT_POST_PRINT_LEAD_Y_MM
    post_print_shift_x_mm: float = DEFAULT_POST_PRINT_SHIFT_X_MM
    exit_lift_z: float = DEFAULT_EXIT_LIFT_Z
    exit_y_mm: float = DEFAULT_EXIT_Y_MM
    exit_b_deg: float = DEFAULT_EXIT_B_DEG
    travel_feed: float = DEFAULT_TRAVEL_FEED
    approach_feed: float = DEFAULT_APPROACH_FEED
    fine_approach_feed: float = DEFAULT_FINE_APPROACH_FEED
    print_feed: float = DEFAULT_PRINT_FEED
    extrusion_per_mm: float = DEFAULT_EXTRUSION_PER_MM
    preflow_dwell_ms: int = DEFAULT_PREFLOW_DWELL_MS
    end_dwell_ms: int = DEFAULT_END_DWELL_MS
    emit_pressure: int = DEFAULT_EMIT_PRESSURE
    emit_u_extrusion: int = DEFAULT_EMIT_U_EXTRUSION
    pressure_pin: int = DEFAULT_PRESSURE_PIN

    bbox_x_min: float = DEFAULT_BBOX_X_MIN
    bbox_x_max: float = DEFAULT_BBOX_X_MAX
    bbox_y_min: float = DEFAULT_BBOX_Y_MIN
    bbox_y_max: float = DEFAULT_BBOX_Y_MAX
    bbox_z_min: float = DEFAULT_BBOX_Z_MIN
    bbox_z_max: float = DEFAULT_BBOX_Z_MAX

    show_mesh: int = 1
    show_raw_skeleton: int = 0
    show_centerline: int = 1
    show_tip_path: int = 1
    show_b_c_markers: int = 1
    preview_stride_mesh: int = 3
    view_z_stretch: float = DEFAULT_VIEW_Z_STRETCH
    plot_dpi: int = 200


@dataclass
class Calibration:
    pr: np.ndarray
    pz: np.ndarray
    py_off: Optional[np.ndarray]
    pa: Optional[np.ndarray]
    b_min: float
    b_max: float
    x_axis: str = "X"
    y_axis: str = "Y"
    z_axis: str = "Z"
    b_axis: str = "B"
    c_axis: str = "C"
    u_axis: str = "U"
    c_180_deg: float = 180.0
    r_model: Optional[Dict[str, Any]] = None
    z_model: Optional[Dict[str, Any]] = None
    y_off_model: Optional[Dict[str, Any]] = None
    y_off_extrap_model: Optional[Dict[str, Any]] = None
    tip_angle_model: Optional[Dict[str, Any]] = None
    offplane_sign: float = DEFAULT_OFFPLANE_SIGN
    active_phase: str = ""
    fit_model_selector: str = ""
    offplane_fit_model_selector: str = ""


@dataclass
class ExtractionResult:
    mesh_vertices: Optional[np.ndarray]
    mesh_faces: Optional[np.ndarray]
    raw_centerline: np.ndarray
    centerline: np.ndarray
    placed_tip_path: np.ndarray
    arclength: np.ndarray
    warnings: List[str]


# =============================================================================
# Generic numeric helpers
# =============================================================================

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))


def smoothstep(x: float) -> float:
    x = clamp01(x)
    return x * x * (3.0 - 2.0 * x)


def smootherstep(x: float) -> float:
    x = clamp01(x)
    return x * x * x * (x * (x * 6.0 - 15.0) + 10.0)


def fmt_float(value: float, decimals: int = 5) -> str:
    s = f"{float(value):.{decimals}f}".rstrip("0").rstrip(".")
    return "0" if s == "-0" else s


def coerce_param_value(field_name: str, value: Any) -> Any:
    field_map = {f.name: f for f in fields(Params)}
    field_def = field_map.get(field_name)
    if field_def is None:
        raise KeyError(field_name)
    target_type = field_def.type
    if target_type is int:
        return int(value)
    if target_type is float:
        return float(value)
    if target_type is str:
        return "" if value is None else str(value)
    return value


def params_to_preset_dict(p: Params) -> Dict[str, Any]:
    return {
        "preset_type": "mesh_centerline_knot_gui_settings",
        "preset_version": PRESET_FORMAT_VERSION,
        "params": asdict(p),
    }


def apply_preset_data(p: Params, data: Dict[str, Any]) -> List[str]:
    payload = data.get("params", data)
    if not isinstance(payload, dict):
        raise ValueError("Preset JSON must contain an object of parameter values.")
    payload = dict(payload)
    if "post_print_drop_z_mm" not in payload and "post_print_end_z" in payload:
        try:
            payload["post_print_drop_z_mm"] = max(0.0, float(getattr(p, "safe_travel_z")) - float(payload["post_print_end_z"]))
        except Exception:
            pass
    ignored: List[str] = []
    for key, raw_value in payload.items():
        if not hasattr(p, key):
            ignored.append(str(key))
            continue
        setattr(p, key, coerce_param_value(key, raw_value))
    return ignored


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    a = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(a))
    if n <= eps:
        return np.zeros_like(a)
    return a / n


def polyline_arclength(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return np.zeros(0, dtype=float)
    out = np.zeros(len(pts), dtype=float)
    if len(pts) > 1:
        out[1:] = np.cumsum(np.linalg.norm(np.diff(pts, axis=0), axis=1))
    return out


def remove_duplicate_points(points: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return pts.copy()
    keep = [pts[0]]
    for p in pts[1:]:
        if float(np.linalg.norm(p - keep[-1])) > tol:
            keep.append(p)
    return np.asarray(keep, dtype=float)


def euler_rotation_matrix(rx_deg: float, ry_deg: float, rz_deg: float) -> np.ndarray:
    rx = math.radians(float(rx_deg))
    ry = math.radians(float(ry_deg))
    rz = math.radians(float(rz_deg))
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]], dtype=float)
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]], dtype=float)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


def resample_polyline(points: np.ndarray, spacing_mm: float, min_points: int = 2) -> np.ndarray:
    pts = remove_duplicate_points(np.asarray(points, dtype=float))
    if len(pts) <= 1:
        return pts.copy()
    s = polyline_arclength(pts)
    total = float(s[-1])
    if total <= 1e-12:
        return pts[:1].copy()
    n = max(int(math.ceil(total / max(1e-6, float(spacing_mm)))) + 1, int(min_points))
    s_new = np.linspace(0.0, total, n)
    out = np.column_stack([np.interp(s_new, s, pts[:, j]) for j in range(3)])
    return out


def smooth_centerline(points: np.ndarray, spacing_mm: float, sigma_mm: float) -> np.ndarray:
    pts = resample_polyline(points, spacing_mm, min_points=4)
    if len(pts) < 5 or float(sigma_mm) <= 1e-9 or gaussian_filter1d is None:
        return pts
    sigma_samples = max(0.0, float(sigma_mm) / max(1e-6, float(spacing_mm)))
    sm = pts.copy()
    for j in range(3):
        sm[:, j] = gaussian_filter1d(pts[:, j], sigma=sigma_samples, mode="nearest")
    # Keep the branch endpoints exact; the center is smoothed.
    sm[0] = pts[0]
    sm[-1] = pts[-1]
    return resample_polyline(sm, spacing_mm, min_points=len(sm))


def branch_y_warp(points: np.ndarray, bottom_y: float, top_y: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float).copy()
    if len(pts) < 2:
        return pts
    s = polyline_arclength(pts)
    u = s / max(1e-9, float(s[-1]))
    delta0 = float(bottom_y) - float(pts[0, 1])
    delta1 = float(top_y) - float(pts[-1, 1])
    pts[:, 1] += (1.0 - u) * delta0 + u * delta1
    pts[0, 1] = float(bottom_y)
    pts[-1, 1] = float(top_y)
    return pts


def enforce_coaxial_vertical_branches(
    points: np.ndarray,
    branch_len_mm: float = DEFAULT_VERTICAL_BRANCH_LENGTH_MM,
    blend_mm: float = DEFAULT_VERTICAL_BRANCH_BLEND_MM,
    flare_mm: float = DEFAULT_VERTICAL_BRANCH_FLARE_MM,
) -> np.ndarray:
    """Force same-XY vertical tails, then flare smoothly into/out of the knot.

    The first/last branch_len_mm of arclength are kept perfectly vertical on a
    common XY anchor.  The following blend_mm is a smooth easing region that
    starts the knot curve earlier than the extracted centerline would.  A small
    optional flare pushes the transition outward in XY, creating a larger loop
    before the tight knot region and reducing kinematic collision risk.
    """
    pts = remove_duplicate_points(np.asarray(points, dtype=float), tol=1e-9)
    if len(pts) < 3:
        return pts.copy()
    out = pts.copy()
    s = polyline_arclength(out)
    total = float(s[-1])
    if total <= 1e-9:
        return out

    branch = max(0.0, float(branch_len_mm))
    blend = max(0.0, float(blend_mm))
    flare = max(0.0, float(flare_mm))

    # Use the current lower endpoint as the machine placement anchor.  Make the
    # top endpoint coaxial with it.  This is intentionally applied after optional
    # Y-warping, so the coaxial constraint wins when both options are enabled.
    anchor_xy = out[0, :2].copy()
    src_xy = pts[:, :2].copy()

    def _dir_from_region(start_side: bool) -> np.ndarray:
        target_s = min(total, branch + blend) if start_side else max(0.0, total - branch - blend)
        idx = int(np.argmin(np.abs(s - target_s)))
        v = src_xy[idx] - anchor_xy
        if float(np.linalg.norm(v)) < 1e-9:
            # Fall back to the strongest lateral direction in the relevant half.
            if start_side:
                candidates = src_xy[s <= min(total, branch + max(blend, 1e-6) * 1.5)] - anchor_xy
            else:
                candidates = src_xy[s >= max(0.0, total - branch - max(blend, 1e-6) * 1.5)] - anchor_xy
            if len(candidates):
                j = int(np.argmax(np.linalg.norm(candidates, axis=1)))
                v = candidates[j]
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return np.array([1.0, 0.0], dtype=float)
        return v / n

    start_dir = _dir_from_region(True)
    end_dir = _dir_from_region(False)

    def _apply_start(i: int, si: float) -> bool:
        if si <= branch:
            out[i, :2] = anchor_xy
            return True
        if blend <= 1e-9 or si >= branch + blend:
            return False
        u = (si - branch) / blend
        a = smootherstep(u)
        # Smooth endpoint-zero bump: more lateral clearance mid-transition,
        # with no kink at the vertical tail or at the extracted knot.
        bump = math.sin(math.pi * clamp01(u)) ** 1.25
        out[i, :2] = anchor_xy + a * (src_xy[i] - anchor_xy) + flare * bump * start_dir
        return True

    def _apply_end(i: int, si: float) -> bool:
        di = total - si
        if di <= branch:
            out[i, :2] = anchor_xy
            return True
        if blend <= 1e-9 or di >= branch + blend:
            return False
        u = (di - branch) / blend
        a = smootherstep(u)
        bump = math.sin(math.pi * clamp01(u)) ** 1.25
        out[i, :2] = anchor_xy + a * (src_xy[i] - anchor_xy) + flare * bump * end_dir
        return True

    for i, si in enumerate(s):
        si = float(si)
        # Prefer the nearer endpoint if the two transition regions overlap.
        if si <= total * 0.5:
            _apply_start(i, si)
        else:
            _apply_end(i, si)

    out[0, :2] = anchor_xy
    out[-1, :2] = anchor_xy
    return out


def apply_final_vertical_tail_no_wobble(points: np.ndarray, p: Params) -> np.ndarray:
    """Remove residual lateral wobble from the straight vertical tails.

    This runs after all geometry shaping and final uniform scaling.  It locks the
    first/last `vertical_tail_no_wobble_mm` of arclength to the coaxial vertical
    anchor, then uses one monotone eased XY transition over
    `vertical_tail_no_wobble_fade_mm`.  The Z values are not altered, so the tail
    remains a true vertical line until the clean bend into the knot begins.
    """
    pts = remove_duplicate_points(np.asarray(points, dtype=float), tol=1e-9)
    if len(pts) < 4:
        return pts.copy()
    out = pts.copy()
    s = polyline_arclength(out)
    total = float(s[-1])
    if total <= 1e-9:
        return out

    lock = max(0.0, float(getattr(p, "vertical_tail_no_wobble_mm", DEFAULT_VERTICAL_TAIL_NO_WOBBLE_MM)))
    fade = max(0.0, float(getattr(p, "vertical_tail_no_wobble_fade_mm", DEFAULT_VERTICAL_TAIL_NO_WOBBLE_FADE_MM)))
    # Avoid consuming the knot when the overall path is very short.
    max_each = 0.42 * total
    lock = min(lock, max_each)
    fade = min(fade, max(0.0, max_each - lock))
    if lock <= 1e-9 and fade <= 1e-9:
        return out

    anchor_xy = out[0, :2].copy()
    out[-1, :2] = anchor_xy

    def target_xy_at(s_target: float) -> np.ndarray:
        s_target = float(np.clip(s_target, 0.0, total))
        return np.array([
            np.interp(s_target, s, pts[:, 0]),
            np.interp(s_target, s, pts[:, 1]),
        ], dtype=float)

    start_target = target_xy_at(min(total, lock + fade))
    end_target = target_xy_at(max(0.0, total - lock - fade))

    for i, si0 in enumerate(s):
        si = float(si0)
        di = total - si
        if si <= total * 0.5:
            if si <= lock:
                out[i, :2] = anchor_xy
            elif fade > 1e-9 and si < lock + fade:
                u = smootherstep((si - lock) / fade)
                out[i, :2] = anchor_xy + u * (start_target - anchor_xy)
        else:
            if di <= lock:
                out[i, :2] = anchor_xy
            elif fade > 1e-9 and di < lock + fade:
                u = smootherstep((di - lock) / fade)
                out[i, :2] = anchor_xy + u * (end_target - anchor_xy)

    out[0, :2] = anchor_xy
    out[-1, :2] = anchor_xy
    return out


def apply_marker_x_offsets(points: np.ndarray, p: Params) -> np.ndarray:
    """Apply local X offsets at B/C marker-defined knot landmarks.

    This is a post-extraction, post-placement shaping layer.  It does not scale
    the mesh and it keeps the first/last coaxial vertical tails fixed.  The two
    defaults do what you asked for:
      - top-most C0 region: X -15 mm
      - lower C≈-176 region: X +15 mm
    """
    if not int(getattr(p, "enable_marker_x_offsets", DEFAULT_ENABLE_MARKER_X_OFFSETS)):
        return np.asarray(points, dtype=float).copy()
    pts = remove_duplicate_points(np.asarray(points, dtype=float), tol=1e-9)
    if len(pts) < 8:
        return pts.copy()
    out = pts.copy()
    s = polyline_arclength(out)
    total = float(s[-1])
    if total <= 1e-9:
        return out
    u = s / total
    c = c_schedule(u, p)

    branch = max(0.0, float(getattr(p, "vertical_branch_length_mm", DEFAULT_VERTICAL_BRANCH_LENGTH_MM)))
    blend = max(0.0, float(getattr(p, "vertical_branch_blend_mm", DEFAULT_VERTICAL_BRANCH_BLEND_MM)))
    guard = min(total * 0.35, branch + 0.25 * blend + 1.0)
    interior = (s > guard) & (s < total - guard)
    if not np.any(interior):
        interior = np.ones(len(out), dtype=bool)
        interior[0] = False
        interior[-1] = False

    def _local_add(center_idx: int, dx: float, width_mm: float) -> None:
        if abs(float(dx)) <= 1e-12 or width_mm <= 1e-9:
            return
        w = np.exp(-0.5 * ((s - float(s[center_idx])) / max(1e-6, float(width_mm))) ** 2)
        # Do not move the perfectly vertical end tails.
        w = np.where(interior, w, 0.0)
        out[:, 0] += float(dx) * w

    # Top-most C0 point: pick the highest-Z interior point still on/near C0.
    c0_band = max(10.0, abs(float(p.c_end_deg) - float(p.c_start_deg)) * 0.08)
    c0_mask = interior & (np.abs(c - float(p.c_start_deg)) <= c0_band)
    if not np.any(c0_mask):
        # If C has already started ramping on the upper lobe, use the highest
        # point in the first half of the interior path.
        c0_mask = interior & (u < min(0.65, float(p.c_ramp_start) + 0.20))
    if np.any(c0_mask):
        idxs = np.flatnonzero(c0_mask)
        idx_top = int(idxs[np.argmax(out[idxs, 2])])
        _local_add(idx_top, float(getattr(p, "top_c0_x_offset_mm", DEFAULT_TOP_C0_X_OFFSET_MM)), float(getattr(p, "marker_x_offset_width_mm", DEFAULT_MARKER_X_OFFSET_WIDTH_MM)))

    # Lower C target point: choose the lowest-Z point near the target C value.
    target_c = float(getattr(p, "bottom_c_target_deg", DEFAULT_BOTTOM_C_TARGET_DEG))
    c_span = max(12.0, abs(float(p.c_end_deg) - float(p.c_start_deg)) * 0.08)
    cmask = interior & (np.abs(c - target_c) <= c_span)
    if not np.any(cmask):
        cmask = interior
    idxs = np.flatnonzero(cmask)
    if len(idxs):
        # Prefer low Z; this is the bottom of the return/wrap lobe.
        idx_bot = int(idxs[np.argmin(out[idxs, 2])])
        _local_add(idx_bot, float(getattr(p, "bottom_c_target_x_offset_mm", DEFAULT_BOTTOM_C_TARGET_X_OFFSET_MM)), float(getattr(p, "marker_x_offset_width_mm", DEFAULT_MARKER_X_OFFSET_WIDTH_MM)))

    # Keep the exact vertical branch endpoints coaxial after shaping.
    out[0, :2] = pts[0, :2]
    out[-1, :2] = pts[-1, :2]
    return out



def _smooth_tail_window(s: np.ndarray, total: float, guard: float, fade: float) -> np.ndarray:
    """Window that is 0 in the straight tails and 1 in the knot interior."""
    if total <= 1e-9:
        return np.zeros_like(s, dtype=float)
    fade = max(1e-6, float(fade))
    guard = max(0.0, min(float(guard), 0.45 * float(total)))
    a = np.clip((s - guard) / fade, 0.0, 1.0)
    b = np.clip((float(total) - guard - s) / fade, 0.0, 1.0)
    a = a * a * (3.0 - 2.0 * a)
    b = b * b * (3.0 - 2.0 * b)
    return a * b


def apply_xz_projection_shaping(points: np.ndarray, p: Params) -> np.ndarray:
    """Increase the knot's projected XZ overlap without stretching the whole path.

    This is different from Final path scale.  Final path scale multiplies X/Y/Z
    and therefore changes the apparent vertical-tail length and the curvature
    distances.  This projection shaping touches only the interior knot region:

      * interior_x_scale expands/compresses X about the coaxial vertical axis.
      * y_to_x_projection_gain maps existing Y offset into X.  This is useful
        when the imported mesh has a correct loop in the YZ view but the XZ
        projection looks too vertical.

    The first and last vertical tails are protected by a smooth arclength window.
    """
    if not int(getattr(p, "enable_xz_projection_shape", DEFAULT_ENABLE_XZ_PROJECTION_SHAPE)):
        return np.asarray(points, dtype=float).copy()

    pts = remove_duplicate_points(np.asarray(points, dtype=float), tol=1e-9)
    if len(pts) < 8:
        return pts.copy()

    x_scale = float(getattr(p, "interior_x_scale", DEFAULT_INTERIOR_X_SCALE))
    y_gain = float(getattr(p, "y_to_x_projection_gain", DEFAULT_Y_TO_X_PROJECTION_GAIN))
    symmetric_spread = float(getattr(p, "symmetric_loop_x_spread_mm", DEFAULT_SYMMETRIC_LOOP_X_SPREAD_MM))
    if abs(x_scale - 1.0) <= 1e-12 and abs(y_gain) <= 1e-12 and abs(symmetric_spread) <= 1e-12:
        return pts.copy()

    out = pts.copy()
    s = polyline_arclength(out)
    total = float(s[-1])
    if total <= 1e-9:
        return out

    branch = max(0.0, float(getattr(p, "vertical_branch_length_mm", DEFAULT_VERTICAL_BRANCH_LENGTH_MM)))
    blend = max(1.0, float(getattr(p, "vertical_branch_blend_mm", DEFAULT_VERTICAL_BRANCH_BLEND_MM)))
    guard = branch + 0.20 * blend + 1.0
    w = _smooth_tail_window(s, total, guard=guard, fade=max(2.0, 0.65 * blend))

    axis_x = float(out[0, 0])
    axis_y = float(out[0, 1])
    # X-only scale about the vertical branch axis.
    out[:, 0] += w * (x_scale - 1.0) * (pts[:, 0] - axis_x)
    # Optional projection of Y loop into the XZ drawing plane.
    out[:, 0] += w * y_gain * (pts[:, 1] - axis_y)

    # Symmetric upper/lower lobe spread.  This is deliberately separate from
    # C-marker offsets: it expands both lobes outward in X while leaving the
    # vertical tails protected.  It gives a little more room for the top loop
    # to return inside itself without adding marker-specific bumps.
    spread = float(getattr(p, "symmetric_loop_x_spread_mm", DEFAULT_SYMMETRIC_LOOP_X_SPREAD_MM))
    if abs(spread) > 1e-12:
        width = max(1e-6, float(getattr(p, "symmetric_loop_x_spread_width_mm", DEFAULT_SYMMETRIC_LOOP_X_SPREAD_WIDTH_MM)))
        interior_mask = w > 1e-3
        if np.any(interior_mask):
            zint = pts[interior_mask, 2]
            z_top = float(np.max(zint))
            z_bot = float(np.min(zint))
            w_lobes = np.exp(-0.5 * ((pts[:, 2] - z_top) / width) ** 2) + np.exp(-0.5 * ((pts[:, 2] - z_bot) / width) ** 2)
            side = np.sign(pts[:, 0] - axis_x)
            # Near the vertical axis, use Y side as a stable outward cue.
            near_axis = np.abs(pts[:, 0] - axis_x) < 0.5
            side = np.where(near_axis, np.sign(pts[:, 1] - axis_y), side)
            side = np.where(np.abs(side) < 1e-9, 1.0, side)
            out[:, 0] += w * w_lobes * spread * side

    # Keep exact endpoints coaxial and untouched.
    out[0, :] = pts[0, :]
    out[-1, 0] = pts[0, 0]
    out[-1, 1] = pts[0, 1]
    return out


def build_demo_centerline() -> np.ndarray:
    """Fallback path for GUI testing when no mesh has been loaded."""
    # Local coordinates.  The first point is the bottom branch, last is top branch.
    controls = np.array([
        [0.0, 15.0, 0.0],
        [0.0, 15.0, 22.0],
        [1.5, 7.0, 34.0],
        [9.5, -8.0, 42.0],
        [24.0, -8.0, 52.0],
        [17.0, -8.0, 74.0],
        [0.0, -7.0, 78.0],
        [-21.0, -7.0, 73.0],
        [-26.0, -7.0, 52.0],
        [-21.0, -4.0, 35.0],
        [-4.0, -1.0, 28.0],
        [17.0, 8.0, 38.0],
        [31.0, 2.0, 48.0],
        [27.0, -4.0, 64.0],
        [12.0, -9.0, 72.0],
        [0.0, -15.0, 82.0],
        [0.0, -15.0, 105.0],
    ], dtype=float)
    return catmull_rom_chain(controls, samples_per_seg=28)


def catmull_rom_chain(points: np.ndarray, samples_per_seg: int = 20, alpha: float = 0.5) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return pts.copy()
    # Duplicate endpoints for endpoint tangent support.
    p = np.vstack([pts[0], pts, pts[-1]])
    out: List[np.ndarray] = []
    for i in range(1, len(p) - 2):
        p0, p1, p2, p3 = p[i - 1], p[i], p[i + 1], p[i + 2]
        def tj(ti: float, a: np.ndarray, b: np.ndarray) -> float:
            return ti + float(np.linalg.norm(b - a)) ** float(alpha)
        t0 = 0.0
        t1 = tj(t0, p0, p1)
        t2 = tj(t1, p1, p2)
        t3 = tj(t2, p2, p3)
        if t1 == t0: t1 += 1e-6
        if t2 == t1: t2 += 1e-6
        if t3 == t2: t3 += 1e-6
        ts = np.linspace(t1, t2, int(max(2, samples_per_seg)), endpoint=False)
        for t in ts:
            A1 = (t1 - t) / (t1 - t0) * p0 + (t - t0) / (t1 - t0) * p1
            A2 = (t2 - t) / (t2 - t1) * p1 + (t - t1) / (t2 - t1) * p2
            A3 = (t3 - t) / (t3 - t2) * p2 + (t - t2) / (t3 - t2) * p3
            B1 = (t2 - t) / (t2 - t0) * A1 + (t - t0) / (t2 - t0) * A2
            B2 = (t3 - t) / (t3 - t1) * A2 + (t - t1) / (t3 - t1) * A3
            C = (t2 - t) / (t2 - t1) * B1 + (t - t1) / (t2 - t1) * B2
            out.append(C)
    out.append(pts[-1].copy())
    return remove_duplicate_points(np.asarray(out, dtype=float))


# =============================================================================
# Calibration / kinematics helpers
# =============================================================================

def _normalize_model_spec(model_spec: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(model_spec, dict):
        return None
    out = dict(model_spec)
    if out.get("model_type") is not None:
        out["model_type"] = str(out["model_type"]).strip().lower()
    return out


def _model_variants(base: str, selector: Optional[str]) -> List[str]:
    sel = None if selector is None else str(selector).strip().lower().replace("-", "_")
    out: List[str] = []
    if sel:
        out.append(f"{base}_{sel}")
        if sel in {"cubic", "pchip", "linear"}:
            out.append(f"{base}_avg_{sel}")
    out.append(base)
    return list(dict.fromkeys(out))


def _select_named_model(models: Dict[str, Any], base_name: str, selector: Optional[str]) -> Optional[Dict[str, Any]]:
    if not isinstance(models, dict):
        return None
    for key in _model_variants(base_name, selector):
        spec = _normalize_model_spec(models.get(key))
        if spec is not None:
            return spec
    return None


def _coeffs_from_models(models: Dict[str, Any], *names: str) -> Optional[np.ndarray]:
    if not isinstance(models, dict):
        return None
    for name in names:
        spec = models.get(name)
        if not isinstance(spec, dict):
            continue
        coeffs = spec.get("coefficients", spec.get("coeffs"))
        if coeffs is not None:
            arr = np.asarray(coeffs, dtype=float).reshape(-1)
            if arr.size > 0:
                return arr
    return None


def poly_eval(coeffs: Any, u: Any, default_if_none: Optional[float] = None) -> np.ndarray:
    u_arr = np.asarray(u, dtype=float)
    if coeffs is None:
        if default_if_none is None:
            raise ValueError("Missing polynomial coefficients.")
        return np.full_like(u_arr, float(default_if_none), dtype=float)
    arr = np.asarray(coeffs, dtype=float).reshape(-1)
    if arr.size == 0:
        if default_if_none is None:
            raise ValueError("Empty polynomial coefficients.")
        return np.full_like(u_arr, float(default_if_none), dtype=float)
    return np.polyval(arr, u_arr)


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
        raise ValueError("PCHIP requires equal-length nonempty x/y knots.")
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
    hi = x1 - x0
    t = np.clip((flat - x0) / hi, 0.0, 1.0)
    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
    h11 = t**3 - t**2
    yq = h00 * y0 + h10 * hi * d[idx] + h01 * y1 + h11 * hi * d[idx + 1]
    return yq.reshape(xq.shape)


def eval_model_spec(model_spec: Optional[Dict[str, Any]], u: Any, default_if_none: Optional[float] = None) -> np.ndarray:
    if model_spec is None:
        if default_if_none is None:
            raise ValueError("Missing calibration model.")
        return np.full_like(np.asarray(u, dtype=float), float(default_if_none), dtype=float)
    model_type = str(model_spec.get("model_type") or "").strip().lower()
    if model_type == "pchip":
        return pchip_eval(model_spec.get("x_knots"), model_spec.get("y_knots"), u)
    if model_type == "polynomial":
        return poly_eval(model_spec.get("coefficients", model_spec.get("coeffs")), u, default_if_none)
    raise ValueError(f"Unsupported calibration model_type: {model_type!r}")


def eval_pchip_with_linear_extrap(model_spec: Dict[str, Any], extrap_model_spec: Optional[Dict[str, Any]], b: Any) -> np.ndarray:
    x_knots = np.asarray(model_spec.get("x_knots", []), dtype=float).reshape(-1)
    if x_knots.size == 0 or extrap_model_spec is None:
        return eval_model_spec(model_spec, b, default_if_none=0.0)
    b_arr = np.asarray(b, dtype=float)
    out = np.asarray(eval_model_spec(model_spec, b_arr, default_if_none=0.0), dtype=float)
    outside = (b_arr < float(np.min(x_knots))) | (b_arr > float(np.max(x_knots)))
    if np.any(outside):
        out = np.where(outside, eval_model_spec(extrap_model_spec, b_arr, default_if_none=0.0), out)
    return out


def available_calibration_phases(json_path: str) -> List[str]:
    """Return phase equation-set names available in a calibration JSON."""
    try:
        p = Path(json_path)
        if not p.exists():
            return []
        data = json.loads(p.read_text(encoding="utf-8"))
        phase_models = data.get("fit_models_by_phase", {}) or {}
        if not isinstance(phase_models, dict):
            return []
        return sorted(str(k) for k in phase_models.keys())
    except Exception:
        return []


def _resolve_calibration_phase(phase_models: Dict[str, Any], requested: Optional[str], default_phase: str) -> str:
    if not isinstance(phase_models, dict) or not phase_models:
        return ""
    req = str(requested or "auto").strip()
    if not req or req.lower() in {"auto", "default", "legacy"}:
        req = default_phase
    # exact first, then case-insensitive/normalized match.
    if req in phase_models:
        return req
    norm_req = req.lower().replace("-", "_").replace(" ", "_")
    for key in phase_models.keys():
        if str(key).lower().replace("-", "_").replace(" ", "_") == norm_req:
            return str(key)
    # Allow compact names such as pull/release when the file has pull_1/release_1.
    starts = [str(k) for k in phase_models.keys() if str(k).lower().replace("-", "_").startswith(norm_req)]
    if len(starts) == 1:
        return starts[0]
    return str(default_phase) if str(default_phase) in phase_models else str(next(iter(phase_models.keys())))


def load_calibration(json_path: str, fit_model: str, offplane_fit_model: str, offplane_sign: float, calibration_phase: str = "auto") -> Calibration:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")
    data = json.loads(p.read_text(encoding="utf-8"))
    cubic = data.get("cubic_coefficients", {}) or {}
    fit_models = data.get("fit_models", {}) or {}
    phase_models = data.get("fit_models_by_phase", {}) or {}
    default_phase = str(data.get("default_phase_for_legacy_access") or "pull").strip()
    active_phase = _resolve_calibration_phase(phase_models, calibration_phase, default_phase)
    active_models = phase_models.get(active_phase) if isinstance(phase_models, dict) else None
    if not isinstance(active_models, dict):
        active_models = fit_models
        active_phase = "fit_models"
    # Force the user-selected fit families from the GUI/CLI. Defaults are pchip for
    # both normal tip tracking and the off-plane Y compensation.
    selector = str(fit_model or "pchip").strip().lower().replace("-", "_")
    y_selector = str(offplane_fit_model or "pchip").strip().lower().replace("-", "_")

    r_model = _select_named_model(active_models, "r", selector) or _select_named_model(fit_models, "r", selector)
    z_model = _select_named_model(active_models, "z", selector) or _select_named_model(fit_models, "z", selector)
    tip_model = _select_named_model(active_models, "tip_angle", selector) or _select_named_model(fit_models, "tip_angle", selector)
    y_model = (_select_named_model(active_models, "offplane_y", y_selector) or
               _select_named_model(fit_models, "offplane_y", y_selector) or
               _select_named_model(active_models, "offplane_y", "avg_cubic") or
               _select_named_model(fit_models, "offplane_y", "avg_cubic"))
    y_linear = (_select_named_model(active_models, "offplane_y", "avg_linear") or
                _select_named_model(fit_models, "offplane_y", "avg_linear") or
                _select_named_model(active_models, "offplane_y_linear", None) or
                _select_named_model(fit_models, "offplane_y_linear", None))

    pr = np.asarray(cubic.get("r_coeffs"), dtype=float) if cubic.get("r_coeffs") is not None else (_coeffs_from_models(active_models, "r_cubic", "r_avg_cubic") or np.zeros(1))
    pz = np.asarray(cubic.get("z_coeffs"), dtype=float) if cubic.get("z_coeffs") is not None else (_coeffs_from_models(active_models, "z_cubic", "z_avg_cubic") or np.zeros(1))
    py = np.asarray(cubic.get("offplane_y_coeffs"), dtype=float) if cubic.get("offplane_y_coeffs") is not None else _coeffs_from_models(active_models, "offplane_y_cubic", "offplane_y_avg_cubic", "offplane_y")
    pa = np.asarray(cubic.get("tip_angle_coeffs"), dtype=float) if cubic.get("tip_angle_coeffs") is not None else _coeffs_from_models(active_models, "tip_angle_cubic", "tip_angle_avg_cubic")

    motor_setup = data.get("motor_setup", {}) or {}
    duet_map = data.get("duet_axis_mapping", {}) or {}
    b_range = motor_setup.get("b_motor_position_range", [-5.4, 0.0])
    b_min, b_max = map(float, b_range)
    if b_min > b_max:
        b_min, b_max = b_max, b_min

    return Calibration(
        pr=np.asarray(pr, dtype=float),
        pz=np.asarray(pz, dtype=float),
        py_off=None if py is None else np.asarray(py, dtype=float),
        pa=None if pa is None else np.asarray(pa, dtype=float),
        r_model=r_model,
        z_model=z_model,
        y_off_model=y_model,
        y_off_extrap_model=y_linear,
        tip_angle_model=tip_model,
        b_min=float(b_min),
        b_max=float(b_max),
        x_axis=str(duet_map.get("horizontal_axis") or motor_setup.get("horizontal_axis") or "X"),
        y_axis=str(duet_map.get("depth_axis") or motor_setup.get("depth_axis") or "Y"),
        z_axis=str(duet_map.get("vertical_axis") or motor_setup.get("vertical_axis") or "Z"),
        b_axis=str(duet_map.get("pull_axis") or motor_setup.get("b_motor_axis") or "B"),
        c_axis=str(duet_map.get("rotation_axis") or motor_setup.get("rotation_axis") or "C"),
        u_axis=str(duet_map.get("extruder_axis") or "U"),
        c_180_deg=float(motor_setup.get("rotation_axis_180_deg", 180.0)),
        offplane_sign=float(offplane_sign),
        active_phase=str(active_phase),
        fit_model_selector=str(selector),
        offplane_fit_model_selector=str(y_selector),
    )


def eval_r(cal: Calibration, b: Any) -> np.ndarray:
    return eval_model_spec(cal.r_model, b) if cal.r_model is not None else poly_eval(cal.pr, b)


def eval_z(cal: Calibration, b: Any) -> np.ndarray:
    return eval_model_spec(cal.z_model, b) if cal.z_model is not None else poly_eval(cal.pz, b)


def eval_offplane_y(cal: Calibration, b: Any) -> np.ndarray:
    if cal.y_off_model is not None:
        if str(cal.y_off_model.get("model_type", "")).lower() == "pchip":
            vals = eval_pchip_with_linear_extrap(cal.y_off_model, cal.y_off_extrap_model, b)
        else:
            vals = eval_model_spec(cal.y_off_model, b, default_if_none=0.0)
    else:
        vals = poly_eval(cal.py_off, b, default_if_none=0.0)
    return float(cal.offplane_sign) * np.asarray(vals, dtype=float)


def eval_tip_angle_deg(cal: Calibration, b: Any) -> np.ndarray:
    if cal.tip_angle_model is not None:
        return eval_model_spec(cal.tip_angle_model, b)
    if cal.pa is None:
        raise ValueError("Calibration is missing tip_angle model/coefficients.")
    return poly_eval(cal.pa, b)


class TipAngleInverter:
    def __init__(self, cal: Calibration, samples: int = 12001):
        b = np.linspace(float(cal.b_min), float(cal.b_max), int(max(1001, samples)))
        a = eval_tip_angle_deg(cal, b)
        order = np.argsort(a)
        aa = np.asarray(a[order], dtype=float)
        bb = np.asarray(b[order], dtype=float)
        aa_unique, idx = np.unique(aa, return_index=True)
        self.angles = aa_unique
        self.commands = bb[idx]
        self.a_min = float(self.angles[0])
        self.a_max = float(self.angles[-1])

    def angle_to_b(self, angle_deg: float) -> Tuple[float, bool]:
        target = float(angle_deg)
        clamped = False
        if target < self.a_min:
            target = self.a_min
            clamped = True
        if target > self.a_max:
            target = self.a_max
            clamped = True
        return float(np.interp(target, self.angles, self.commands)), clamped


def tip_offset_xyz(cal: Calibration, b_cmd: float, c_deg: float) -> np.ndarray:
    r = float(eval_r(cal, b_cmd))
    z = float(eval_z(cal, b_cmd))
    y_off = float(eval_offplane_y(cal, b_cmd))
    c = math.radians(float(c_deg))
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def stage_xyz_for_tip(cal: Calibration, tip_xyz: np.ndarray, b_cmd: float, c_deg: float) -> np.ndarray:
    return np.asarray(tip_xyz, dtype=float) - tip_offset_xyz(cal, b_cmd, c_deg)


# =============================================================================
# Mesh loading and centerline extraction
# =============================================================================

def require_extract_deps(require_skeleton: bool = False) -> None:
    """Dependencies required after a triangle mesh has been loaded.

    The default extractor is an endpoint-guided distance-field path, so
    scikit-image skeletonization is only required when explicitly falling back
    to the old skeleton method. trimesh remains optional.
    """
    if _SCIPY_IMPORT_ERROR is not None:
        raise RuntimeError(f"scipy is required for graph/smoothing/path extraction: {_SCIPY_IMPORT_ERROR}")
    if require_skeleton and skeletonize is None:
        raise RuntimeError(f"scikit-image is required for skeleton fallback: {_SKIMAGE_IMPORT_ERROR}")


@dataclass
class MeshLite:
    vertices: np.ndarray
    faces: np.ndarray
    source: str = "built-in"

    def __post_init__(self) -> None:
        self.vertices = np.asarray(self.vertices, dtype=float).reshape((-1, 3))
        self.faces = np.asarray(self.faces, dtype=int).reshape((-1, 3))

    def copy(self) -> "MeshLite":
        return MeshLite(self.vertices.copy(), self.faces.copy(), self.source)

    def remove_unreferenced_vertices(self) -> None:
        if len(self.faces) == 0:
            self.vertices = np.zeros((0, 3), dtype=float)
            return
        used = np.unique(self.faces.reshape(-1))
        remap = -np.ones(len(self.vertices), dtype=int)
        remap[used] = np.arange(len(used), dtype=int)
        self.vertices = self.vertices[used]
        self.faces = remap[self.faces]

    def apply_scale(self, scale: float) -> None:
        self.vertices *= float(scale)

    @property
    def is_watertight(self) -> bool:
        if len(self.faces) == 0:
            return False
        edges = np.vstack([
            self.faces[:, [0, 1]],
            self.faces[:, [1, 2]],
            self.faces[:, [2, 0]],
        ])
        edges = np.sort(edges, axis=1)
        _, counts = np.unique(edges, axis=0, return_counts=True)
        return bool(np.all(counts == 2))


def _dedupe_vertices(vertices: np.ndarray, faces: np.ndarray, decimals: int = 7) -> MeshLite:
    verts = np.asarray(vertices, dtype=float).reshape((-1, 3))
    f = np.asarray(faces, dtype=int).reshape((-1, 3))
    if len(verts) == 0 or len(f) == 0:
        return MeshLite(verts, f)
    rounded = np.round(verts, int(decimals))
    unique_rounded, first_idx, inv = np.unique(rounded, axis=0, return_index=True, return_inverse=True)
    # Use the first exact coordinate for each rounded vertex.
    verts2 = verts[first_idx]
    f2 = inv[f]
    keep = np.asarray([len(set(map(int, tri))) == 3 for tri in f2], dtype=bool)
    return MeshLite(verts2, f2[keep])


def _load_binary_stl_lite(data: bytes) -> Optional[MeshLite]:
    if len(data) < 84:
        return None
    n_tri = struct.unpack_from("<I", data, 80)[0]
    expected = 84 + 50 * int(n_tri)
    if expected != len(data):
        return None
    verts = np.empty((int(n_tri) * 3, 3), dtype=float)
    faces = np.empty((int(n_tri), 3), dtype=int)
    off = 84
    for i in range(int(n_tri)):
        # normal is bytes off:off+12
        v = struct.unpack_from("<9f", data, off + 12)
        verts[3 * i:3 * i + 3] = np.asarray(v, dtype=float).reshape((3, 3))
        faces[i] = [3 * i, 3 * i + 1, 3 * i + 2]
        off += 50
    return _dedupe_vertices(verts, faces)


def _load_ascii_stl_lite(text: str) -> MeshLite:
    verts: List[List[float]] = []
    faces: List[List[int]] = []
    cur: List[int] = []
    for raw in text.splitlines():
        parts = raw.strip().split()
        if len(parts) == 4 and parts[0].lower() == "vertex":
            try:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                cur.append(len(verts) - 1)
                if len(cur) == 3:
                    faces.append(cur)
                    cur = []
            except ValueError:
                continue
    if not verts or not faces:
        raise ValueError("Could not parse ASCII STL vertices/facets.")
    return _dedupe_vertices(np.asarray(verts, dtype=float), np.asarray(faces, dtype=int))


def load_stl_lite(path: Path) -> MeshLite:
    data = path.read_bytes()
    mesh = _load_binary_stl_lite(data)
    if mesh is not None:
        mesh.source = "built-in binary STL"
        return mesh
    # Some binary STLs start with 'solid', so only fall back after the size check.
    text = data.decode("utf-8", errors="ignore")
    mesh = _load_ascii_stl_lite(text)
    mesh.source = "built-in ASCII STL"
    return mesh


def _xml_local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def load_3mf_lite(path: Path) -> MeshLite:
    verts_all: List[np.ndarray] = []
    faces_all: List[np.ndarray] = []
    v_offset = 0
    with zipfile.ZipFile(path, "r") as zf:
        model_names = [n for n in zf.namelist() if n.lower().endswith(".model")]
        if not model_names:
            raise ValueError("3MF file has no .model XML entry.")
        # Most 3MF files use 3D/3dmodel.model.
        model_names.sort(key=lambda n: ("3d/3dmodel.model" not in n.lower(), len(n)))
        root = ET.fromstring(zf.read(model_names[0]))
    for obj in root.iter():
        if _xml_local(obj.tag) != "object":
            continue
        mesh_elem = None
        for child in obj:
            if _xml_local(child.tag) == "mesh":
                mesh_elem = child
                break
        if mesh_elem is None:
            continue
        vertices_elem = None
        triangles_elem = None
        for child in mesh_elem:
            lname = _xml_local(child.tag)
            if lname == "vertices":
                vertices_elem = child
            elif lname == "triangles":
                triangles_elem = child
        if vertices_elem is None or triangles_elem is None:
            continue
        verts: List[List[float]] = []
        for v in vertices_elem:
            if _xml_local(v.tag) != "vertex":
                continue
            verts.append([float(v.attrib.get("x", "0")), float(v.attrib.get("y", "0")), float(v.attrib.get("z", "0"))])
        faces: List[List[int]] = []
        for tri in triangles_elem:
            if _xml_local(tri.tag) != "triangle":
                continue
            faces.append([int(tri.attrib["v1"]), int(tri.attrib["v2"]), int(tri.attrib["v3"])])
        if verts and faces:
            verts_all.append(np.asarray(verts, dtype=float))
            faces_all.append(np.asarray(faces, dtype=int) + v_offset)
            v_offset += len(verts)
    if not verts_all:
        raise ValueError("Could not parse triangle meshes from 3MF. If it uses components/transforms, install trimesh for full 3MF support.")
    mesh = _dedupe_vertices(np.vstack(verts_all), np.vstack(faces_all))
    mesh.source = "built-in 3MF"
    return mesh


def load_obj_lite(path: Path) -> MeshLite:
    verts: List[List[float]] = []
    faces: List[List[int]] = []
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        if raw.startswith("v "):
            parts = raw.split()
            if len(parts) >= 4:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
        elif raw.startswith("f "):
            idxs = []
            for tok in raw.split()[1:]:
                base = tok.split("/", 1)[0]
                if not base:
                    continue
                ii = int(base)
                idxs.append(ii - 1 if ii > 0 else len(verts) + ii)
            for k in range(1, len(idxs) - 1):
                faces.append([idxs[0], idxs[k], idxs[k + 1]])
    if not verts or not faces:
        raise ValueError("Could not parse OBJ vertices/faces.")
    mesh = _dedupe_vertices(np.asarray(verts, dtype=float), np.asarray(faces, dtype=int))
    mesh.source = "built-in OBJ"
    return mesh


def load_triangle_mesh_lite(path: str, unit_scale: float = 1.0) -> MeshLite:
    p = Path(path)
    ext = p.suffix.lower()
    if ext == ".stl":
        mesh = load_stl_lite(p)
    elif ext == ".3mf":
        mesh = load_3mf_lite(p)
    elif ext == ".obj":
        mesh = load_obj_lite(p)
    else:
        raise RuntimeError(f"Built-in fallback loader supports STL, simple 3MF, and OBJ. Install trimesh for {ext or 'this file type'}.")
    mesh.remove_unreferenced_vertices()
    if float(unit_scale) != 1.0:
        mesh.apply_scale(float(unit_scale))
    return mesh


def load_triangle_mesh(path: str, unit_scale: float = 1.0):
    require_extract_deps()
    if not path:
        raise FileNotFoundError("No mesh path provided.")
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Mesh not found: {path}")

    if trimesh is not None:
        loaded = trimesh.load(str(p), process=False)
        if isinstance(loaded, trimesh.Scene):
            try:
                mesh = loaded.dump(concatenate=True)
            except TypeError:
                mesh = trimesh.util.concatenate(list(loaded.geometry.values()))
        elif isinstance(loaded, (list, tuple)):
            mesh = trimesh.util.concatenate([m for m in loaded if isinstance(m, trimesh.Trimesh)])
        else:
            mesh = loaded
        if not isinstance(mesh, trimesh.Trimesh):
            raise TypeError(f"Could not load a triangle mesh from {path!r}")
        mesh = mesh.copy()
        mesh.remove_unreferenced_vertices()
        if float(unit_scale) != 1.0:
            mesh.apply_scale(float(unit_scale))
        return mesh

    return load_triangle_mesh_lite(str(p), unit_scale=float(unit_scale))


def voxelize_mesh_filled_lite(vertices: np.ndarray, faces: np.ndarray, pitch: float) -> Tuple[np.ndarray, np.ndarray]:
    """Voxelize a closed triangle mesh using +X ray parity filling.

    Returns (volume[x,y,z], origin), where integer voxel index i maps to
    world position origin + i * pitch.  This fallback is slower than trimesh but
    avoids the trimesh dependency for ordinary watertight STL/3MF tube meshes.
    """
    verts = np.asarray(vertices, dtype=float).reshape((-1, 3))
    f = np.asarray(faces, dtype=int).reshape((-1, 3))
    if len(verts) == 0 or len(f) == 0:
        raise ValueError("Input mesh is empty.")
    pitch = max(0.025, float(pitch))
    pad = 4
    bmin = np.min(verts, axis=0)
    bmax = np.max(verts, axis=0)
    origin = bmin - pad * pitch
    shape = np.ceil((bmax - bmin) / pitch).astype(int) + 2 * pad + 3
    shape = np.maximum(shape, 8)
    if int(np.prod(shape)) > 35_000_000:
        raise MemoryError(
            f"Fallback voxel grid would be {tuple(map(int, shape))} ({int(np.prod(shape))} voxels). "
            "Increase voxel_pitch or install trimesh for a faster voxelizer."
        )
    vg = (verts - origin) / pitch
    volume = np.zeros(tuple(int(x) for x in shape), dtype=bool)
    intersections: Dict[Tuple[int, int], List[float]] = defaultdict(list)

    def add_surface_samples(tri: np.ndarray) -> None:
        edges = [np.linalg.norm(tri[1] - tri[0]), np.linalg.norm(tri[2] - tri[1]), np.linalg.norm(tri[0] - tri[2])]
        n = int(max(1, min(20, math.ceil(max(edges) * 1.75))))
        for i in range(n + 1):
            for j in range(n + 1 - i):
                a = i / n
                b = j / n
                c = 1.0 - a - b
                q = a * tri[0] + b * tri[1] + c * tri[2]
                idx = np.rint(q).astype(int)
                if np.all(idx >= 0) and np.all(idx < shape):
                    volume[tuple(idx)] = True

    for face in f:
        tri = vg[face]
        add_surface_samples(tri)
        # Intersections of +X rays with triangle, projected in the YZ plane.
        yz0 = tri[0, 1:3]
        yz1 = tri[1, 1:3]
        yz2 = tri[2, 1:3]
        den = (yz1[1] - yz2[1]) * (yz0[0] - yz2[0]) + (yz2[0] - yz1[0]) * (yz0[1] - yz2[1])
        if abs(float(den)) < 1e-10:
            continue
        ymin = max(0, int(math.floor(float(np.min(tri[:, 1])) - 1)))
        ymax = min(int(shape[1]) - 1, int(math.ceil(float(np.max(tri[:, 1])) + 1)))
        zmin = max(0, int(math.floor(float(np.min(tri[:, 2])) - 1)))
        zmax = min(int(shape[2]) - 1, int(math.ceil(float(np.max(tri[:, 2])) + 1)))
        for iy in range(ymin, ymax + 1):
            py = float(iy)
            for iz in range(zmin, zmax + 1):
                pz = float(iz)
                w0 = ((yz1[1] - yz2[1]) * (py - yz2[0]) + (yz2[0] - yz1[0]) * (pz - yz2[1])) / den
                w1 = ((yz2[1] - yz0[1]) * (py - yz2[0]) + (yz0[0] - yz2[0]) * (pz - yz2[1])) / den
                w2 = 1.0 - w0 - w1
                eps = -1e-8
                if w0 >= eps and w1 >= eps and w2 >= eps:
                    x = w0 * tri[0, 0] + w1 * tri[1, 0] + w2 * tri[2, 0]
                    if -1 <= x <= shape[0] + 1:
                        intersections[(iy, iz)].append(float(x))

    for (iy, iz), xs in intersections.items():
        if len(xs) < 2:
            continue
        xs_sorted = sorted(xs)
        clustered: List[float] = []
        for x in xs_sorted:
            if not clustered or abs(x - clustered[-1]) > 0.45:
                clustered.append(float(x))
            else:
                clustered[-1] = 0.5 * (clustered[-1] + float(x))
        if len(clustered) < 2:
            continue
        if len(clustered) % 2 == 1:
            # Drop the last unpaired edge-hit.  This is usually a tangent/vertex hit.
            clustered = clustered[:-1]
        for a, b in zip(clustered[0::2], clustered[1::2]):
            lo, hi = sorted((float(a), float(b)))
            ix0 = max(0, int(math.ceil(lo - 1e-8)))
            ix1 = min(int(shape[0]) - 1, int(math.floor(hi + 1e-8)))
            if ix1 >= ix0:
                volume[ix0:ix1 + 1, int(iy), int(iz)] = True

    if binary_closing is not None:
        try:
            volume = binary_closing(volume, iterations=1)
        except TypeError:
            volume = binary_closing(volume)
    if binary_fill_holes is not None:
        volume = np.asarray(binary_fill_holes(volume), dtype=bool)
    if int(volume.sum()) < 8:
        raise ValueError("Fallback voxelization produced too few voxels. Try a smaller voxel_pitch or a watertight/capped mesh.")
    return volume, origin


def _build_skeleton_graph(coords: np.ndarray, pitch: float) -> Tuple[csr_matrix, np.ndarray, np.ndarray]:
    coords = np.asarray(coords, dtype=np.int32)
    n = len(coords)
    key_to_idx = {tuple(c): i for i, c in enumerate(coords)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if not (dx == 0 and dy == 0 and dz == 0)]
    for i, c in enumerate(coords):
        cx, cy, cz = int(c[0]), int(c[1]), int(c[2])
        for dx, dy, dz in offsets:
            if dx < 0 or (dx == 0 and dy < 0) or (dx == 0 and dy == 0 and dz < 0):
                # add each undirected edge once, then mirror
                continue
            j = key_to_idx.get((cx + dx, cy + dy, cz + dz))
            if j is None:
                continue
            w = float(pitch) * math.sqrt(dx * dx + dy * dy + dz * dz)
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([w, w])
    graph = coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()
    degree = np.diff(graph.indptr)
    return graph, degree, coords


def _reconstruct_predecessor_path(pred: np.ndarray, src: int, dst: int) -> List[int]:
    path = [int(dst)]
    cur = int(dst)
    guard = 0
    while cur != int(src) and guard < len(pred) + 5:
        cur = int(pred[cur])
        if cur < 0:
            break
        path.append(cur)
        guard += 1
    path.reverse()
    return path


def skeleton_longest_path_indices(coords: np.ndarray, pitch: float) -> np.ndarray:
    graph, degree, coords = _build_skeleton_graph(coords, pitch)
    if graph.shape[0] < 2:
        raise ValueError("Skeleton has fewer than 2 voxels.")
    n_comp, labels = connected_components(graph, directed=False)
    if n_comp > 1:
        counts = np.bincount(labels)
        largest = int(np.argmax(counts))
        keep = np.flatnonzero(labels == largest)
        coords = coords[keep]
        graph, degree, coords = _build_skeleton_graph(coords, pitch)
    endpoints = np.flatnonzero(degree <= 1)
    if len(endpoints) < 2:
        endpoints = np.arange(graph.shape[0])
    if len(endpoints) <= 96:
        # Exact endpoint-pair diameter over the skeleton graph.
        dist = dijkstra(graph, directed=False, indices=endpoints, return_predecessors=False)
        end_cols = endpoints
        best = np.unravel_index(int(np.nanargmax(dist[:, end_cols])), (dist.shape[0], len(end_cols)))
        src = int(endpoints[best[0]])
        dst = int(end_cols[best[1]])
    else:
        # Fast two-sweep approximation if the skeleton has many spurious endpoints.
        d0 = dijkstra(graph, directed=False, indices=int(endpoints[0]), return_predecessors=False)
        src = int(np.nanargmax(d0))
        d1 = dijkstra(graph, directed=False, indices=src, return_predecessors=False)
        dst = int(np.nanargmax(d1))
    _, pred = dijkstra(graph, directed=False, indices=src, return_predecessors=True)
    path_idx = _reconstruct_predecessor_path(pred, src, dst)
    if len(path_idx) < 2:
        raise ValueError("Could not reconstruct a centerline path through skeleton.")
    return coords[np.asarray(path_idx, dtype=int)]


# --- Distance-field centerline extraction -------------------------------------

def infer_vertical_cap_centers(vertices: np.ndarray, tube_diameter_mm: float, span_mm: float) -> Tuple[np.ndarray, np.ndarray]:
    """Infer start/end cap centers from the low-Z and high-Z end clusters.

    The intended input knot has straight vertical start/end branches with a
    constant 3 mm diameter.  Using the extreme-Z cap clusters makes the path
    endpoint-guided instead of relying on the skeleton diameter, which can jump
    through close self-contact regions of the knot.
    """
    v = np.asarray(vertices, dtype=float)
    if len(v) == 0:
        raise ValueError("Cannot infer endpoints from an empty mesh.")
    zmin = float(np.min(v[:, 2]))
    zmax = float(np.max(v[:, 2]))
    span = max(0.05, float(span_mm), 0.35 * float(tube_diameter_mm))
    lo = v[v[:, 2] <= zmin + span]
    hi = v[v[:, 2] >= zmax - span]
    if len(lo) < 3:
        lo = v[np.argsort(v[:, 2])[: max(3, min(64, len(v)))]]
    if len(hi) < 3:
        hi = v[np.argsort(v[:, 2])[-max(3, min(64, len(v))) :]]
    bottom = np.array([float(np.mean(lo[:, 0])), float(np.mean(lo[:, 1])), zmin], dtype=float)
    top = np.array([float(np.mean(hi[:, 0])), float(np.mean(hi[:, 1])), zmax], dtype=float)
    return bottom, top


def _build_surface_edge_graph(vertices: np.ndarray, faces: np.ndarray) -> csr_matrix:
    """Sparse geodesic graph on mesh vertices using triangle edges."""
    v = np.asarray(vertices, dtype=float).reshape((-1, 3))
    f = np.asarray(faces, dtype=int).reshape((-1, 3))
    n = int(len(v))
    if n < 2 or len(f) == 0:
        raise ValueError("Mesh has too few vertices/faces for surface geodesic extraction.")
    edges = np.vstack([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    edges = np.sort(edges, axis=1)
    edges = np.unique(edges, axis=0)
    keep = (edges[:, 0] >= 0) & (edges[:, 1] >= 0) & (edges[:, 0] < n) & (edges[:, 1] < n) & (edges[:, 0] != edges[:, 1])
    edges = edges[keep]
    if len(edges) == 0:
        raise ValueError("Mesh edge graph is empty.")
    w = np.linalg.norm(v[edges[:, 0]] - v[edges[:, 1]], axis=1)
    rows = np.concatenate([edges[:, 0], edges[:, 1]]).astype(int)
    cols = np.concatenate([edges[:, 1], edges[:, 0]]).astype(int)
    data = np.concatenate([w, w]).astype(float)
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def _dijkstra_multisource(graph: csr_matrix, source_indices: np.ndarray) -> np.ndarray:
    """Dijkstra from a virtual source connected to source_indices at zero cost."""
    src = np.unique(np.asarray(source_indices, dtype=int))
    if src.size == 0:
        raise ValueError("No source vertices for multi-source geodesic.")
    n = graph.shape[0]
    rows = np.concatenate([src, np.full(src.size, n, dtype=int)])
    cols = np.concatenate([np.full(src.size, n, dtype=int), src])
    data = np.full(src.size * 2, 1e-12, dtype=float)
    extra = coo_matrix((data, (rows, cols)), shape=(n + 1, n + 1)).tocsr()
    rr, cc = graph.nonzero()
    base = coo_matrix((graph.data, (rr, cc)), shape=(n + 1, n + 1)).tocsr()
    g2 = base + extra
    d = dijkstra(g2, directed=False, indices=n, return_predecessors=False)
    return np.asarray(d[:n], dtype=float)


def surface_geodesic_centerline_path(vertices: np.ndarray, faces: np.ndarray, p: Params) -> np.ndarray:
    """Extract a tube centerline by averaging surface geodesic rings.

    This is the preferred extractor for a constant-diameter capped tube mesh.
    It does not voxelize the solid, so it avoids the failure mode where a medial
    skeleton jumps through close self-contact regions of a knot.  We infer the
    low/high-Z cap vertex sets, compute geodesic distance along the *surface*
    from the bottom cap, and average vertices in equal-distance bands.  On a
    circular tube, those bands are cross-section rings; their centroids recover
    the tube centerline.
    """
    if _SCIPY_IMPORT_ERROR is not None or coo_matrix is None or dijkstra is None:
        raise RuntimeError(f"scipy sparse graph tools are required for surface_geodesic extraction: {_SCIPY_IMPORT_ERROR}")
    v = np.asarray(vertices, dtype=float).reshape((-1, 3))
    f = np.asarray(faces, dtype=int).reshape((-1, 3))
    if len(v) < 8 or len(f) < 4:
        raise ValueError("Mesh is too small for surface geodesic extraction.")

    zmin = float(np.min(v[:, 2]))
    zmax = float(np.max(v[:, 2]))
    span = max(0.05, float(p.endpoint_span_mm), 0.35 * float(p.mesh_tube_diameter))
    start_idx = np.flatnonzero(v[:, 2] <= zmin + span)
    end_idx = np.flatnonzero(v[:, 2] >= zmax - span)
    if start_idx.size < 3:
        start_idx = np.argsort(v[:, 2])[: max(3, min(128, len(v)))]
    if end_idx.size < 3:
        end_idx = np.argsort(v[:, 2])[-max(3, min(128, len(v))) :]

    start_center = np.array([float(np.mean(v[start_idx, 0])), float(np.mean(v[start_idx, 1])), zmin], dtype=float)
    end_center = np.array([float(np.mean(v[end_idx, 0])), float(np.mean(v[end_idx, 1])), zmax], dtype=float)

    graph = _build_surface_edge_graph(v, f)
    dist = _dijkstra_multisource(graph, start_idx)
    finite = np.isfinite(dist)
    if not np.any(finite):
        raise ValueError("No finite surface geodesic distances from start cap.")
    end_finite = end_idx[np.isfinite(dist[end_idx])]
    if end_finite.size == 0:
        raise ValueError("High-Z cap is not connected to low-Z cap on the mesh surface.")
    end_dist = float(np.nanmin(dist[end_finite]))
    if not np.isfinite(end_dist) or end_dist <= 1e-6:
        raise ValueError("Invalid low-to-high cap geodesic length.")

    spacing = max(0.05, float(p.surface_ring_spacing_mm), 0.15 * float(p.mesh_tube_diameter))
    band = max(0.05, float(p.surface_ring_band_mm), 0.15 * float(p.mesh_tube_diameter))
    n_bins = int(max(8, math.ceil(end_dist / spacing)))
    targets = np.linspace(0.0, end_dist, n_bins + 1)
    centers: List[np.ndarray] = [start_center]

    # Skip the end-cap spans; use explicit cap centers at both ends.
    for tdist in targets[1:-1]:
        mask = finite & (np.abs(dist - float(tdist)) <= band)
        idx = np.flatnonzero(mask)
        if idx.size < 6:
            # Wider fallback for sparse triangulation.
            mask = finite & (np.abs(dist - float(tdist)) <= 2.0 * band)
            idx = np.flatnonzero(mask)
        if idx.size < 3:
            continue
        dd = np.abs(dist[idx] - float(tdist))
        weights = np.maximum(0.0, 1.0 - dd / max(1e-9, (2.0 * band)))
        if float(np.sum(weights)) <= 1e-12:
            c = np.mean(v[idx], axis=0)
        else:
            c = np.average(v[idx], axis=0, weights=weights)
        # Avoid adding many nearly identical cap points.
        if np.linalg.norm(c - centers[-1]) > 0.05 * float(p.mesh_tube_diameter):
            centers.append(np.asarray(c, dtype=float))

    centers.append(end_center)
    raw = remove_duplicate_points(np.asarray(centers, dtype=float), tol=1e-7)
    if len(raw) < 4:
        raise ValueError("Surface geodesic ring extraction produced too few center points.")

    if int(getattr(p, "surface_geodesic_refine_slices", 1)):
        raw = refine_centerline_with_mesh_slices(
            vertices=v,
            faces=f,
            approx_path=raw,
            p=p,
            preserve_xz=bool(int(getattr(p, "surface_geodesic_lock_xz", 1))),
        )
    return raw



def _plane_basis_from_normal(normal: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    n = normalize(normal)
    ref = np.array([1.0, 0.0, 0.0], dtype=float)
    if abs(float(np.dot(ref, n))) > 0.9:
        ref = np.array([0.0, 1.0, 0.0], dtype=float)
    u = normalize(ref - float(np.dot(ref, n)) * n)
    w = normalize(np.cross(n, u))
    return u, w


def _triangle_plane_intersection_points(vertices: np.ndarray, faces: np.ndarray, plane_point: np.ndarray, normal: np.ndarray) -> np.ndarray:
    """Return all triangle-edge intersection points with a plane."""
    v = np.asarray(vertices, dtype=float)
    f = np.asarray(faces, dtype=int)
    n = normalize(np.asarray(normal, dtype=float))
    p0 = np.asarray(plane_point, dtype=float)
    if len(v) < 3 or len(f) == 0:
        return np.zeros((0, 3), dtype=float)
    tri = v[f]
    sd = np.tensordot(tri - p0[None, None, :], n, axes=([2], [0]))
    eps = 1e-9
    # Plane must cross the triangle.  Ignore exactly coplanar facets because they
    # are usually cap triangles and can dominate the slice mean.
    mask = (np.min(sd, axis=1) <= eps) & (np.max(sd, axis=1) >= -eps) & (np.max(np.abs(sd), axis=1) > eps)
    if not np.any(mask):
        return np.zeros((0, 3), dtype=float)
    tri = tri[mask]
    sd = sd[mask]
    pts: List[np.ndarray] = []
    for ia, ib in ((0, 1), (1, 2), (2, 0)):
        sa = sd[:, ia]
        sb = sd[:, ib]
        denom = sa - sb
        crosses = (np.abs(denom) > eps) & (((sa <= eps) & (sb >= -eps)) | ((sb <= eps) & (sa >= -eps)))
        if not np.any(crosses):
            continue
        t = sa[crosses] / denom[crosses]
        t = np.clip(t, 0.0, 1.0)
        pa = tri[crosses, ia, :]
        pb = tri[crosses, ib, :]
        pts.append(pa + (pb - pa) * t[:, None])
    if not pts:
        return np.zeros((0, 3), dtype=float)
    out = np.vstack(pts)
    # Dedupe plane/edge duplicates without destroying the local ring distribution.
    if len(out) > 1:
        out = np.unique(np.round(out, 7), axis=0)
    return out


def _largest_near_slice_cluster(points: np.ndarray, pred_center: np.ndarray, cluster_radius: float, min_points: int = 8) -> np.ndarray:
    """Choose the connected cluster nearest pred_center from plane-slice points."""
    pts = np.asarray(points, dtype=float).reshape((-1, 3))
    if len(pts) <= min_points or cKDTree is None or connected_components is None or coo_matrix is None:
        return pts
    tree = cKDTree(pts)
    pairs = np.asarray(list(tree.query_pairs(r=max(1e-6, float(cluster_radius)))), dtype=int)
    if pairs.size == 0:
        return pts
    rows = np.concatenate([pairs[:, 0], pairs[:, 1]])
    cols = np.concatenate([pairs[:, 1], pairs[:, 0]])
    data = np.ones(len(rows), dtype=float)
    graph = coo_matrix((data, (rows, cols)), shape=(len(pts), len(pts))).tocsr()
    n_comp, labels = connected_components(graph, directed=False)
    if n_comp <= 1:
        return pts
    pred = np.asarray(pred_center, dtype=float)
    best_idx = None
    best_score = float("inf")
    for lab in range(n_comp):
        idx = np.flatnonzero(labels == lab)
        if idx.size < min_points:
            continue
        c = np.mean(pts[idx], axis=0)
        score = float(np.linalg.norm(c - pred)) - 0.01 * float(idx.size)
        if score < best_score:
            best_score = score
            best_idx = idx
    if best_idx is None:
        return pts
    return pts[best_idx]


def _fit_slice_center(points: np.ndarray, plane_point: np.ndarray, normal: np.ndarray, tube_radius: float) -> Tuple[np.ndarray, float]:
    """Fit a circle to a cross-section. Falls back to the slice centroid."""
    pts = np.asarray(points, dtype=float).reshape((-1, 3))
    if len(pts) == 0:
        raise ValueError("No points to fit slice center.")
    centroid = np.mean(pts, axis=0)
    if len(pts) < 6:
        return centroid, 0.0
    u, w = _plane_basis_from_normal(normal)
    rel = pts - np.asarray(plane_point, dtype=float)[None, :]
    xy = np.column_stack([rel @ u, rel @ w])
    x = xy[:, 0]
    y = xy[:, 1]
    A = np.column_stack([2.0 * x, 2.0 * y, np.ones_like(x)])
    b = x * x + y * y
    try:
        sol, *_ = np.linalg.lstsq(A, b, rcond=None)
        cx, cy, cc = map(float, sol)
        rad2 = cc + cx * cx + cy * cy
        r = math.sqrt(max(0.0, rad2))
        c3 = np.asarray(plane_point, dtype=float) + cx * u + cy * w
        # Trust the circle fit only when it looks like the expected 3 mm tube.
        if np.isfinite(r) and 0.35 * tube_radius <= r <= 2.4 * tube_radius and np.linalg.norm(c3 - centroid) <= 3.0 * max(tube_radius, 1e-6):
            return c3, r
    except Exception:
        pass
    rr = float(np.mean(np.linalg.norm(pts - centroid[None, :], axis=1)))
    return centroid, rr



def refine_centerline_with_mesh_slices(
    vertices: np.ndarray,
    faces: np.ndarray,
    approx_path: np.ndarray,
    p: Params,
    preserve_xz: bool = True,
) -> np.ndarray:
    """Refine a surface-geodesic centerline using true mesh cross-sections.

    The surface-geodesic bands usually give the right topological order and a
    good XZ projection, but in tight over/under knots their band centroid can
    sit on the tube wall in Y.  For each geodesic sample we slice the mesh by a
    plane perpendicular to the local path tangent, fit the circular 3 mm tube
    section, and use that fitted circle center.  With preserve_xz=True the
    geodesic X/Z coordinates are kept and only Y is corrected; this matches the
    observed failure mode where XZ is good but YZ is wrong.
    """
    path = remove_duplicate_points(np.asarray(approx_path, dtype=float), tol=1e-8)
    if len(path) < 4:
        return path
    tube_radius = max(0.05, 0.5 * float(getattr(p, "mesh_tube_diameter", 3.0)))
    gate = max(2.5 * tube_radius, 2.25 * float(getattr(p, "surface_ring_band_mm", 0.55)), 1.8)
    cluster_radius = max(0.35 * tube_radius, float(getattr(p, "plane_march_cluster_radius_mm", 0.9)))
    half_window = max(2, min(8, int(round(2.0 * tube_radius / max(0.05, float(getattr(p, "surface_ring_spacing_mm", 0.45)))))))

    refined = path.copy()
    # Keep cap centers fixed; cross-section slicing exactly on caps is unstable.
    for i in range(1, len(path) - 1):
        ilo = max(0, i - half_window)
        ihi = min(len(path) - 1, i + half_window)
        tangent = normalize(path[ihi] - path[ilo])
        if float(np.linalg.norm(tangent)) < 1e-9:
            continue
        candidate = path[i]
        all_pts = _triangle_plane_intersection_points(vertices, faces, candidate, tangent)
        if len(all_pts) < 6:
            continue

        best_center = None
        best_score = float("inf")
        for mult in (1.0, 1.6, 2.4, 3.5):
            near = all_pts[np.linalg.norm(all_pts - candidate[None, :], axis=1) <= gate * mult]
            if len(near) < 6:
                continue
            cluster = _largest_near_slice_cluster(near, candidate, cluster_radius=cluster_radius * mult, min_points=6)
            if len(cluster) < 6:
                continue
            center, rad = _fit_slice_center(cluster, candidate, tangent, tube_radius)
            if not np.all(np.isfinite(center)):
                continue
            # Prefer a plausible tube radius and a center close to the geodesic guess.
            radial_error = abs(float(rad) - tube_radius) if rad > 1e-9 else tube_radius
            dist_error = float(np.linalg.norm(center - candidate))
            if dist_error > max(4.5 * tube_radius, gate * mult):
                continue
            score = dist_error + 2.0 * radial_error - 0.001 * len(cluster)
            if score < best_score:
                best_score = score
                best_center = center
        if best_center is None:
            continue
        if preserve_xz:
            refined[i, 1] = float(best_center[1])
        else:
            refined[i] = best_center

    # A light second-pass smoothing only on the corrected Y prevents sawtooth YZ
    # traces from irregular triangulation while preserving the geodesic XZ trace.
    if len(refined) > 7 and gaussian_filter1d is not None:
        sigma_samples = max(0.0, float(getattr(p, "smooth_sigma_mm", 1.2)) / max(0.05, float(getattr(p, "surface_ring_spacing_mm", 0.45))))
        sigma_samples = min(max(sigma_samples, 0.0), 3.0)
        if sigma_samples > 0.25:
            y = refined[:, 1].copy()
            yy = gaussian_filter1d(y, sigma=sigma_samples, mode="nearest")
            yy[0] = y[0]
            yy[-1] = y[-1]
            refined[:, 1] = yy
    return remove_duplicate_points(refined, tol=1e-8)


def _slice_center_near_prediction(
    vertices: np.ndarray,
    faces: np.ndarray,
    plane_point: np.ndarray,
    normal: np.ndarray,
    pred_center: np.ndarray,
    gate_radius: float,
    cluster_radius: float,
    tube_radius: float,
) -> Optional[np.ndarray]:
    all_pts = _triangle_plane_intersection_points(vertices, faces, plane_point, normal)
    if len(all_pts) < 6:
        return None
    pred = np.asarray(pred_center, dtype=float)
    for mult in (1.0, 1.7, 2.6, 4.0):
        gate = max(float(gate_radius) * mult, 1.8 * float(tube_radius))
        near = all_pts[np.linalg.norm(all_pts - pred[None, :], axis=1) <= gate]
        if len(near) < 6:
            continue
        cluster = _largest_near_slice_cluster(near, pred, cluster_radius=max(float(cluster_radius), 0.35 * float(tube_radius)), min_points=6)
        if len(cluster) < 6:
            continue
        c, _r = _fit_slice_center(cluster, plane_point, normal, float(tube_radius))
        if np.linalg.norm(c - pred) <= max(2.5 * gate, 4.0 * float(tube_radius)):
            return np.asarray(c, dtype=float)
    return None


def plane_march_centerline_path(vertices: np.ndarray, faces: np.ndarray, p: Params) -> np.ndarray:
    """Extract centerline by marching through circular plane slices of the tube.

    The tube is constant diameter and has vertical start/end branches.  Starting
    from the low-Z cap center, we repeatedly slice the mesh by a plane normal to
    the current tangent, select the local intersection ring nearest the predicted
    next center, and fit that ring's circle center.  This follows the actual tube
    core through close over/under passes instead of averaging surface-geodesic
    bands that may remain on the side wall.
    """
    v = np.asarray(vertices, dtype=float).reshape((-1, 3))
    f = np.asarray(faces, dtype=int).reshape((-1, 3))
    if len(v) < 8 or len(f) < 4:
        raise ValueError("Mesh is too small for plane-march extraction.")

    start, end = infer_vertical_cap_centers(v, float(p.mesh_tube_diameter), float(p.endpoint_span_mm))
    tube_radius = 0.5 * max(0.05, float(p.mesh_tube_diameter))
    step = max(0.05, float(p.plane_march_step_mm), 0.08 * tube_radius)
    gate = max(float(p.plane_march_gate_radius_mm), 2.3 * tube_radius)
    cluster_r = max(float(p.plane_march_cluster_radius_mm), 0.35 * tube_radius)
    max_steps = int(max(50, float(p.plane_march_max_steps)))

    centers: List[np.ndarray] = [np.asarray(start, dtype=float)]
    # Start branch is vertical by construction.
    tangent = np.array([0.0, 0.0, 1.0], dtype=float)
    if end[2] < start[2]:
        tangent *= -1.0

    misses = 0
    for _ in range(max_steps):
        cur = centers[-1]
        if len(centers) > 6 and float(np.linalg.norm(cur - end)) <= max(1.5 * step, 0.75 * tube_radius):
            break
        pred = cur + tangent * step
        # Do not overshoot the final cap once we are in the last straight branch.
        if float(np.linalg.norm(pred - end)) < float(np.linalg.norm(cur - end)) and abs(float(pred[2] - end[2])) <= step:
            pred = 0.5 * (pred + end)
        c = _slice_center_near_prediction(v, f, pred, tangent, pred, gate, cluster_r, tube_radius)
        if c is None:
            # A short lost-slice run is common around cap triangles or very tight bends.
            misses += 1
            if misses >= 6:
                break
            pred2 = cur + tangent * (step * (1.0 + 0.5 * misses))
            c = _slice_center_near_prediction(v, f, pred2, tangent, pred2, gate * (1.0 + 0.35 * misses), cluster_r, tube_radius)
            if c is None:
                centers.append(pred2)
                continue
        misses = 0
        # One refinement at the fitted center reduces offset from the side wall.
        c2 = _slice_center_near_prediction(v, f, c, tangent, c, gate, cluster_r, tube_radius)
        if c2 is not None and np.linalg.norm(c2 - c) <= 2.5 * tube_radius:
            c = c2
        move = c - cur
        d = float(np.linalg.norm(move))
        if d <= 0.04 * tube_radius:
            continue
        new_tangent = move / d
        # Smooth tangent but keep it responsive through bends.
        tangent = normalize(0.35 * tangent + 0.65 * new_tangent)
        centers.append(np.asarray(c, dtype=float))
        # Prevent accidental runaway after passing the top cap.
        if len(centers) > 10 and ((tangent[2] > 0 and c[2] >= end[2] - 0.25 * tube_radius) or (tangent[2] < 0 and c[2] <= end[2] + 0.25 * tube_radius)):
            if np.linalg.norm(c - end) <= 3.0 * tube_radius:
                break

    if np.linalg.norm(centers[-1] - end) > 0.2 * tube_radius:
        centers.append(np.asarray(end, dtype=float))
    raw = remove_duplicate_points(np.asarray(centers, dtype=float), tol=1e-7)
    if len(raw) < 6:
        raise ValueError("Plane-march extraction produced too few center points. Increase gate radius or reduce march step.")
    return raw

def _volume_world_points(coords: np.ndarray, pitch: float, *, origin: Optional[np.ndarray] = None, vox=None) -> np.ndarray:
    coords = np.asarray(coords, dtype=int)
    if vox is not None and hasattr(vox, "indices_to_points"):
        return np.asarray(vox.indices_to_points(coords), dtype=float)
    if origin is None:
        origin = np.zeros(3, dtype=float)
    return np.asarray(origin, dtype=float) + coords.astype(float) * float(pitch)


def nearest_center_voxel_to_endpoint(
    coords: np.ndarray,
    world_pts: np.ndarray,
    edt_values: np.ndarray,
    endpoint_xyz: np.ndarray,
    radius_mm: float,
) -> int:
    """Pick a voxel close to the cap center, biased toward the tube core."""
    q = np.asarray(endpoint_xyz, dtype=float)
    d = np.linalg.norm(np.asarray(world_pts, dtype=float) - q[None, :], axis=1)
    r = max(0.05, float(radius_mm))
    core_bonus_threshold = 0.55 * r
    wall_pen = np.maximum(0.0, core_bonus_threshold - np.asarray(edt_values, dtype=float))
    score = d + 4.0 * wall_pen
    return int(np.argmin(score))


def build_weighted_volume_graph(coords: np.ndarray, edt_values: np.ndarray, pitch: float, p: Params) -> csr_matrix:
    """Build a 26-neighbor graph through all filled voxels.

    Edge weights penalize voxels close to the wall, so the shortest path stays
    near the medial ridge of the constant-diameter tube.  This is more reliable
    for a pretzel knot than taking the graph diameter of a raw skeleton, because
    skeletons often introduce shortcuts near close self-contacts.
    """
    coords = np.asarray(coords, dtype=np.int32)
    edt_values = np.asarray(edt_values, dtype=float).reshape(-1)
    n = int(len(coords))
    if n < 2:
        raise ValueError("Filled volume has fewer than 2 voxels.")
    key_to_idx = {tuple(c): i for i, c in enumerate(coords)}
    rows: List[int] = []
    cols: List[int] = []
    data: List[float] = []
    radius = max(0.05, 0.5 * float(p.mesh_tube_diameter))
    weight = max(0.0, float(p.centerline_clearance_weight))
    power = max(0.25, float(p.centerline_clearance_power))
    offsets = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1)
               if not (dx == 0 and dy == 0 and dz == 0)]
    for i, c in enumerate(coords):
        cx, cy, cz = int(c[0]), int(c[1]), int(c[2])
        di = float(edt_values[i])
        for dx, dy, dz in offsets:
            if dx < 0 or (dx == 0 and dy < 0) or (dx == 0 and dy == 0 and dz < 0):
                continue
            j = key_to_idx.get((cx + dx, cy + dy, cz + dz))
            if j is None:
                continue
            dj = float(edt_values[j])
            step = float(pitch) * math.sqrt(dx * dx + dy * dy + dz * dz)
            dmid = 0.5 * (di + dj)
            # 0 near ideal centerline, 1 near/at wall.
            wall = max(0.0, min(1.0, (radius - dmid) / radius))
            penalty = 1.0 + weight * (wall ** power)
            w = step * penalty
            rows.extend([i, j])
            cols.extend([j, i])
            data.extend([w, w])
    return coo_matrix((data, (rows, cols)), shape=(n, n)).tocsr()


def distance_field_centerline_path(
    volume: np.ndarray,
    pitch: float,
    vertices: np.ndarray,
    p: Params,
    *,
    origin: Optional[np.ndarray] = None,
    vox=None,
) -> np.ndarray:
    """Endpoint-guided centerline through the filled tube volume.

    The path is constrained to the filled voxel volume and weighted by distance
    from the surface.  Endpoints are inferred from the low/high Z vertical cap
    centers, matching the expected mesh: constant 3 mm tube with vertical start
    and end branches.
    """
    if distance_transform_edt is None:
        raise RuntimeError("scipy.ndimage.distance_transform_edt is unavailable.")
    vol = np.asarray(volume, dtype=bool)
    if int(vol.sum()) < 8:
        raise ValueError("Voxelized mesh is too sparse. Try a smaller voxel pitch or check mesh scale/units.")
    edt = np.asarray(distance_transform_edt(vol, sampling=float(pitch)), dtype=float)
    coords = np.argwhere(vol)
    edt_values = edt[tuple(coords.T)]
    world_pts = _volume_world_points(coords, pitch, origin=origin, vox=vox)

    bottom, top = infer_vertical_cap_centers(vertices, float(p.mesh_tube_diameter), float(p.endpoint_span_mm))
    radius = max(0.05, 0.5 * float(p.mesh_tube_diameter))
    src = nearest_center_voxel_to_endpoint(coords, world_pts, edt_values, bottom, radius)
    dst = nearest_center_voxel_to_endpoint(coords, world_pts, edt_values, top, radius)
    if src == dst:
        raise ValueError("Inferred start/end voxels collapsed to one point; check mesh scale or endpoint span.")

    graph = build_weighted_volume_graph(coords, edt_values, pitch, p)
    _, pred = dijkstra(graph, directed=False, indices=int(src), return_predecessors=True)
    path_idx = _reconstruct_predecessor_path(pred, int(src), int(dst))
    if len(path_idx) < 2 or int(path_idx[-1]) != int(dst):
        raise ValueError("Could not find a connected distance-field path between vertical end caps.")
    path_coords = coords[np.asarray(path_idx, dtype=int)]
    return _volume_world_points(path_coords, pitch, origin=origin, vox=vox)


def _skeleton_centerline_from_volume(
    volume: np.ndarray,
    pitch: float,
    *,
    origin: Optional[np.ndarray] = None,
    vox=None,
) -> np.ndarray:
    """Legacy skeleton/diameter extraction used as a fallback."""
    require_extract_deps(require_skeleton=True)
    vol_pad = np.pad(np.asarray(volume, dtype=bool), 1, mode="constant", constant_values=False)
    try:
        skel = skeletonize(vol_pad, method="lee")
    except TypeError:
        skel = skeletonize(vol_pad)
    skel = np.asarray(skel, dtype=bool)
    coords_pad = np.argwhere(skel)
    if len(coords_pad) < 2:
        raise ValueError("Skeleton extraction produced too few voxels. Try smaller voxel_pitch or repair the mesh.")
    path_coords_pad = skeleton_longest_path_indices(coords_pad, pitch)
    path_coords = path_coords_pad.astype(float) - 1.0
    if vox is not None and hasattr(vox, "indices_to_points"):
        return np.asarray(vox.indices_to_points(path_coords.astype(int)), dtype=float)
    if origin is None:
        origin = np.zeros(3, dtype=float)
    return np.asarray(origin, dtype=float) + path_coords.astype(float) * float(pitch)


def extract_centerline_from_mesh(mesh, p: Params) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Return raw world-space centerline, mesh vertices, mesh faces, warnings."""
    require_extract_deps(require_skeleton=False)
    warnings: List[str] = []
    vertices = np.asarray(mesh.vertices, dtype=float)
    faces = np.asarray(mesh.faces, dtype=int)
    if vertices.shape[0] == 0 or faces.shape[0] == 0:
        raise ValueError("Input mesh is empty.")
    is_watertight = bool(getattr(mesh, "is_watertight", False))
    if not is_watertight:
        warnings.append("Mesh is not watertight; voxel fill may be less reliable. A capped/watertight tube is preferred.")

    pitch = max(0.05, float(p.voxel_pitch))
    method = str(getattr(p, "extraction_method", DEFAULT_EXTRACTION_METHOD) or DEFAULT_EXTRACTION_METHOD).strip().lower().replace("-", "_")
    if method not in {"plane_march", "surface_geodesic", "distance_path", "skeleton"}:
        warnings.append(f"Unknown extraction_method={method!r}; using surface_geodesic.")
        method = "surface_geodesic"

    if method == "plane_march":
        try:
            raw = plane_march_centerline_path(vertices, faces, p)
            warnings.append("Centerline extracted by local circular plane slices through the constant-diameter tube.")
        except Exception as exc:
            warnings.append(f"Plane-march extraction failed ({exc}); falling back to surface-geodesic rings.")
            try:
                raw = surface_geodesic_centerline_path(vertices, faces, p)
                warnings.append("Centerline extracted from surface-geodesic rings between the low/high Z vertical tube caps.")
            except Exception as exc2:
                warnings.append(f"Surface-geodesic extraction failed ({exc2}); falling back to distance-field path.")
                method = "distance_path"
    elif method == "surface_geodesic":
        try:
            raw = surface_geodesic_centerline_path(vertices, faces, p)
            warnings.append("Centerline extracted from surface-geodesic rings between the low/high Z vertical tube caps.")
        except Exception as exc:
            warnings.append(f"Surface-geodesic extraction failed ({exc}); falling back to distance-field path.")
            method = "distance_path"

    if method in {"distance_path", "skeleton"}:
        volume = None
        origin = None
        vox_filled = None
        if trimesh is not None and hasattr(mesh, "voxelized"):
            vox = mesh.voxelized(pitch)
            try:
                vox_filled = vox.fill()
            except Exception as exc:
                warnings.append(f"Voxel fill failed ({exc}); using surface voxels only.")
                vox_filled = vox
            volume = np.asarray(vox_filled.matrix, dtype=bool)
            if volume.sum() < 8:
                raise ValueError("Voxelized mesh is too sparse. Try a smaller voxel_pitch or check mesh scale/units.")
            warnings.append("Mesh loaded/voxelized with trimesh.")
        else:
            volume, origin = voxelize_mesh_filled_lite(vertices, faces, pitch)
            warnings.append("Mesh loaded/voxelized with built-in fallback loader; install trimesh for faster/more robust voxelization if needed.")

        if method == "distance_path":
            try:
                raw = distance_field_centerline_path(volume, pitch, vertices, p, origin=origin, vox=vox_filled)
                warnings.append("Centerline extracted with endpoint-guided distance-field path from vertical end caps.")
            except Exception as exc:
                warnings.append(f"Distance-field path failed ({exc}); falling back to skeleton diameter.")
                raw = _skeleton_centerline_from_volume(volume, pitch, origin=origin, vox=vox_filled)
                warnings.append("Centerline extracted with legacy skeleton fallback.")
        else:
            raw = _skeleton_centerline_from_volume(volume, pitch, origin=origin, vox=vox_filled)
            warnings.append("Centerline extracted with legacy skeleton diameter method.")

    raw = remove_duplicate_points(np.asarray(raw, dtype=float), tol=1e-9)
    if len(raw) < 2:
        raise ValueError("Extracted centerline path is too short.")
    return raw, vertices, faces, warnings


def final_uniform_scale_about_anchor(points: np.ndarray, scale: float, anchor: Optional[np.ndarray] = None) -> np.ndarray:
    """Uniformly scale the finished trace geometry about a fixed anchor.

    Earlier versions multiplied the raw extracted centerline before smoothing,
    coaxial-tail enforcement, flare shaping, and C-marker offsets.  Those
    operations use millimeter lengths, so scaling first changed the effective
    geometry.  This final-stage scale preserves the solved scale-1 silhouette
    and simply makes the whole robot trace larger or smaller.
    """
    pts = np.asarray(points, dtype=float)
    if len(pts) == 0:
        return pts.copy()
    sc = float(scale)
    if abs(sc - 1.0) <= 1e-12:
        return pts.copy()
    if anchor is None:
        anchor_arr = pts[0].copy()
    else:
        anchor_arr = np.asarray(anchor, dtype=float).reshape(3)
    return anchor_arr + sc * (pts - anchor_arr)


def apply_transform_and_place(raw_centerline: np.ndarray, p: Params) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    pts = np.asarray(raw_centerline, dtype=float).copy()
    pts = remove_duplicate_points(pts)
    if len(pts) < 2:
        raise ValueError("Centerline has fewer than 2 points.")
    # Orient from low Z to high Z by default. The GUI reverse checkbox flips after that.
    if pts[0, 2] > pts[-1, 2]:
        pts = pts[::-1].copy()
    if int(p.reverse_path):
        pts = pts[::-1].copy()
    # Rotate around the first point.  Do NOT apply trace_scale here: scaling
    # before smoothing/tail/flare/marker constraints changes the apparent knot
    # geometry because those constraints are defined in millimeters.  The
    # uniform path scale is applied at the very end instead.
    origin = pts[0].copy()
    R = euler_rotation_matrix(p.rot_x_deg, p.rot_y_deg, p.rot_z_deg)
    pts2 = (pts - origin) @ R.T
    # Put the bottom endpoint exactly at requested start_x/start_y/start_z.
    start = np.array([float(p.start_x), float(p.start_y), float(p.start_z)], dtype=float)
    placed = start + pts2
    if int(p.force_branch_y):
        placed = branch_y_warp(placed, float(p.bottom_branch_y_offset), float(p.top_branch_y_offset))
        placed[0] = np.array([float(p.start_x), float(p.bottom_branch_y_offset), float(p.start_z)], dtype=float)
    smoothed = smooth_centerline(placed, float(p.resample_spacing_mm), float(p.smooth_sigma_mm))
    if int(p.force_branch_y):
        smoothed = branch_y_warp(smoothed, float(p.bottom_branch_y_offset), float(p.top_branch_y_offset))
        smoothed[0] = np.array([float(p.start_x), float(p.bottom_branch_y_offset), float(p.start_z)], dtype=float)
    if int(getattr(p, "force_coaxial_branches", DEFAULT_FORCE_COAXIAL_BRANCHES)):
        smoothed = enforce_coaxial_vertical_branches(
            smoothed,
            branch_len_mm=float(getattr(p, "vertical_branch_length_mm", DEFAULT_VERTICAL_BRANCH_LENGTH_MM)),
            blend_mm=float(getattr(p, "vertical_branch_blend_mm", DEFAULT_VERTICAL_BRANCH_BLEND_MM)),
            flare_mm=float(getattr(p, "vertical_branch_flare_mm", DEFAULT_VERTICAL_BRANCH_FLARE_MM)),
        )
        # Keep the placed lower endpoint exact after the endpoint-axis constraint.
        smoothed[0, 0] = float(p.start_x)
        smoothed[0, 1] = float(p.bottom_branch_y_offset) if int(p.force_branch_y) else float(p.start_y)
        smoothed[0, 2] = float(p.start_z)
        smoothed[-1, 0] = smoothed[0, 0]
        smoothed[-1, 1] = smoothed[0, 1]
    smoothed = apply_marker_x_offsets(smoothed, p)
    smoothed = apply_xz_projection_shaping(smoothed, p)
    if int(getattr(p, "force_coaxial_branches", DEFAULT_FORCE_COAXIAL_BRANCHES)):
        smoothed[0, 0] = float(p.start_x)
        smoothed[0, 1] = float(p.bottom_branch_y_offset) if int(p.force_branch_y) else float(p.start_y)
        smoothed[-1, 0] = smoothed[0, 0]
        smoothed[-1, 1] = smoothed[0, 1]

    # Final uniform robot-path scaling.  This gives the larger overlap/loop of
    # scale 3.0 while keeping the smooth scale-1 geometry, because all
    # smoothing, tail, flare, and marker-shaping decisions have already been
    # made at the base geometry.
    anchor = smoothed[0].copy()
    smoothed = final_uniform_scale_about_anchor(smoothed, float(p.trace_scale), anchor=anchor)
    if int(getattr(p, "force_coaxial_branches", DEFAULT_FORCE_COAXIAL_BRANCHES)):
        smoothed[0, 0] = float(p.start_x)
        smoothed[0, 1] = float(p.bottom_branch_y_offset) if int(p.force_branch_y) else float(p.start_y)
        smoothed[0, 2] = float(p.start_z)
        smoothed[-1, 0] = smoothed[0, 0]
        smoothed[-1, 1] = smoothed[0, 1]
        # Final tail cleanup: shaping/marker offsets can leave a small lateral
        # wiggle just after the vertical tails. Lock and fade the tail XY last.
        smoothed = apply_final_vertical_tail_no_wobble(smoothed, p)
        # Re-apply once because the arclength changes slightly when lateral tail
        # motion is removed; this guarantees the final measured tail is vertical.
        smoothed = apply_final_vertical_tail_no_wobble(smoothed, p)
        smoothed[0, 0] = float(p.start_x)
        smoothed[0, 1] = float(p.bottom_branch_y_offset) if int(p.force_branch_y) else float(p.start_y)
        smoothed[0, 2] = float(p.start_z)
        smoothed[-1, 0] = smoothed[0, 0]
        smoothed[-1, 1] = smoothed[0, 1]
    s = polyline_arclength(smoothed)
    return pts, smoothed, s


def build_extraction(p: Params) -> ExtractionResult:
    warnings: List[str] = []
    if p.mesh_path:
        mesh = load_triangle_mesh(p.mesh_path, p.mesh_unit_scale)
        raw, verts, faces, ws = extract_centerline_from_mesh(mesh, p)
        warnings.extend(ws)
    else:
        raw = build_demo_centerline()
        verts = None
        faces = None
        warnings.append("No mesh loaded; showing the built-in demo pretzel centerline.")
    raw_oriented, placed, s = apply_transform_and_place(raw, p)
    if abs(float(p.trace_scale) - 1.0) > 1e-12:
        warnings.append(f"Final uniform path scale applied after smoothing/shaping: {float(p.trace_scale):.4g}x about the bottom endpoint.")
        warnings.append("For a larger knot-looking XZ overlap without stretching Z/tails, prefer Final path scale=1 and increase Interior X scale and/or Y->X projection gain.")
    if int(getattr(p, "enable_xz_projection_shape", DEFAULT_ENABLE_XZ_PROJECTION_SHAPE)) and (abs(float(getattr(p, "interior_x_scale", 1.0)) - 1.0) > 1e-12 or abs(float(getattr(p, "y_to_x_projection_gain", 0.0))) > 1e-12 or abs(float(getattr(p, "symmetric_loop_x_spread_mm", 0.0))) > 1e-12):
        warnings.append(f"Interior XZ projection shaping enabled: interior_x_scale={float(getattr(p, 'interior_x_scale', 1.0)):.4g}, y_to_x_projection_gain={float(getattr(p, 'y_to_x_projection_gain', 0.0)):.4g}, symmetric_loop_x_spread={float(getattr(p, 'symmetric_loop_x_spread_mm', 0.0)):.4g} mm.")
    return ExtractionResult(
        mesh_vertices=verts,
        mesh_faces=faces,
        raw_centerline=raw_oriented,
        centerline=raw_oriented,
        placed_tip_path=placed,
        arclength=s,
        warnings=warnings,
    )


# =============================================================================
# B/C schedule and G-code generation
# =============================================================================

def b_schedule(u: np.ndarray, p: Params) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    b = np.full_like(u, float(p.b_start_deg), dtype=float)
    # Ramp from start to mid.
    u0 = clamp01(p.b_ramp_start)
    u1 = clamp01(p.b_ramp_end)
    if u1 > u0:
        a = np.clip((u - u0) / (u1 - u0), 0.0, 1.0)
        a = a * a * (3.0 - 2.0 * a)
        b = (1.0 - a) * float(p.b_start_deg) + a * float(p.b_mid_deg)
    else:
        b[:] = float(p.b_mid_deg)
    # Optional release/ramp to end.
    r0 = clamp01(p.b_release_start)
    r1 = clamp01(p.b_release_end)
    if r1 > r0 and abs(float(p.b_end_deg) - float(p.b_mid_deg)) > 1e-9:
        a = np.clip((u - r0) / (r1 - r0), 0.0, 1.0)
        a = a * a * (3.0 - 2.0 * a)
        b = (1.0 - a) * b + a * float(p.b_end_deg)
    return b


def c_schedule(u: np.ndarray, p: Params) -> np.ndarray:
    """C schedule with an optional upper-loop azimuth lead-in.

    Earlier versions held C at c_start_deg until the C-end ramp.  For the upper
    lobe this can leave the robot outside the print-direction azimuth plane.
    The top-loop block smoothly raises C toward c_top_loop_deg before the final
    C ramp, then the final ramp continues from that current C to c_end_deg.
    """
    u = np.asarray(u, dtype=float)
    c_start = float(p.c_start_deg)
    c = np.full_like(u, c_start, dtype=float)

    def _pre_ramp_c(x: np.ndarray | float) -> np.ndarray:
        xarr = np.asarray(x, dtype=float)
        out = np.full_like(xarr, c_start, dtype=float)
        if int(getattr(p, "enable_top_loop_c_azimuth", DEFAULT_ENABLE_TOP_LOOP_C_AZIMUTH)):
            t0 = clamp01(getattr(p, "c_top_loop_start", DEFAULT_C_TOP_LOOP_START))
            t1 = clamp01(getattr(p, "c_top_loop_end", DEFAULT_C_TOP_LOOP_END))
            if t1 > t0:
                a = np.clip((xarr - t0) / (t1 - t0), 0.0, 1.0)
                a = a * a * (3.0 - 2.0 * a)
                out = (1.0 - a) * c_start + a * float(getattr(p, "c_top_loop_deg", DEFAULT_C_TOP_LOOP_DEG))
        return out

    c = _pre_ramp_c(u)
    u0 = clamp01(p.c_ramp_start)
    u1 = clamp01(p.c_ramp_end)
    if u1 > u0:
        c_at_u0 = float(_pre_ramp_c(np.asarray([u0], dtype=float))[0])
        a = np.clip((u - u0) / (u1 - u0), 0.0, 1.0)
        a = a * a * (3.0 - 2.0 * a)
        ramped = (1.0 - a) * c_at_u0 + a * float(p.c_end_deg)
        c = np.where(u >= u0, ramped, c)
    else:
        c[:] = float(p.c_end_deg)
    return c


def clamp_stage_xyz(stage: np.ndarray, p: Params) -> np.ndarray:
    return np.array([
        float(np.clip(stage[0], float(p.bbox_x_min), float(p.bbox_x_max))),
        float(np.clip(stage[1], float(p.bbox_y_min), float(p.bbox_y_max))),
        float(np.clip(stage[2], float(p.bbox_z_min), float(p.bbox_z_max))),
    ], dtype=float)


def write_gcode(path_xyz: np.ndarray, p: Params) -> Dict[str, Any]:
    pts = np.asarray(path_xyz, dtype=float)
    if len(pts) < 2:
        raise ValueError("Need at least two path points for G-code.")
    s = polyline_arclength(pts)
    total = float(s[-1])
    u = s / max(1e-9, total)
    b_phys = b_schedule(u, p)
    c_cmd = c_schedule(u, p)

    cal: Optional[Calibration] = None
    inverter: Optional[TipAngleInverter] = None
    write_mode = str(p.write_mode).strip().lower()
    if write_mode == "calibrated":
        if not p.calibration_path:
            raise ValueError("Calibration path is required in calibrated mode. Use cartesian mode for testing.")
        cal = load_calibration(p.calibration_path, p.fit_model, p.offplane_fit_model, p.offplane_sign, p.calibration_phase)
        inverter = TipAngleInverter(cal, p.bc_solve_samples)
    elif write_mode != "cartesian":
        raise ValueError("write_mode must be calibrated or cartesian")

    x_axis = cal.x_axis if cal else "X"
    y_axis = cal.y_axis if cal else "Y"
    z_axis = cal.z_axis if cal else "Z"
    b_axis = cal.b_axis if cal else "B"
    c_axis = cal.c_axis if cal else "C"
    u_axis = cal.u_axis if cal else "U"

    def b_command(target_deg: float) -> Tuple[float, bool]:
        if inverter is None:
            return float(target_deg), False
        return inverter.angle_to_b(float(target_deg))

    def stage_for(i: int, tip_override: Optional[np.ndarray] = None, b_override: Optional[float] = None, c_override: Optional[float] = None) -> Tuple[np.ndarray, float, float, bool]:
        tip = pts[i] if tip_override is None else np.asarray(tip_override, dtype=float)
        btar = float(b_phys[i] if b_override is None else b_override)
        c = float(c_cmd[i] if c_override is None else c_override)
        bcmd, clamped = b_command(btar)
        if cal is None:
            stage = tip.copy()
        else:
            stage = stage_xyz_for_tip(cal, tip, bcmd, c)
        return clamp_stage_xyz(stage, p), bcmd, c, clamped

    out_path = Path(p.output_path or DEFAULT_OUT)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    clamped_count = 0
    u_extr = 0.0

    with out_path.open("w", encoding="utf-8") as fh:
        fh.write("; Mesh-centerline pretzel / overhand knot trace\n")
        fh.write("; Generated by mesh_centerline_pretzel_knot_gui.py\n")
        fh.write(f"; mesh_path={p.mesh_path or 'demo_centerline'}\n")
        fh.write(f"; write_mode={write_mode} c_start={p.c_start_deg:.6f} c_end={p.c_end_deg:.6f}\n")
        fh.write(f"; bottom_branch_y_offset={p.bottom_branch_y_offset:.6f} top_branch_y_offset={p.top_branch_y_offset:.6f}\n")
        fh.write(f"; point_count={len(pts)} total_length_mm={total:.6f}\n")
        fh.write("G90                 ; absolute positioning\n")
        fh.write("G94                 ; units/min feedrate mode\n")
        if int(p.emit_u_extrusion):
            fh.write(f"G92 {u_axis}0\n")

        # Approach from a fixed safe Z, then lower vertically to the print start.
        first_tip = pts[0].copy()
        safe_start = first_tip.copy()
        safe_start[2] = float(p.safe_travel_z)
        st, bc, cc, clamped = stage_for(0, tip_override=safe_start)
        clamped_count += int(clamped)
        fh.write("; approach start at safe Z\n")
        fh.write(f"G1 {x_axis}{st[0]:.3f} {y_axis}{st[1]:.3f} {z_axis}{st[2]:.3f} {b_axis}{bc:.3f} {c_axis}{cc:.3f} F{fmt_float(p.travel_feed,3)}\n")
        st, bc, cc, clamped = stage_for(0)
        clamped_count += int(clamped)
        fh.write("; lower to print start\n")
        fh.write(f"G1 {x_axis}{st[0]:.3f} {y_axis}{st[1]:.3f} {z_axis}{st[2]:.3f} {b_axis}{bc:.3f} {c_axis}{cc:.3f} F{fmt_float(p.travel_feed,3)}\n")
        if int(p.emit_pressure):
            fh.write("; open pressure solenoid\n")
            fh.write(f"M42 P{int(p.pressure_pin)} S1\n")
            if int(p.preflow_dwell_ms) > 0:
                fh.write(f"G4 P{int(p.preflow_dwell_ms)}\n")

        fh.write("; trace extracted centerline\n")
        for i in range(1, len(pts)):
            st, bc, cc, clamped = stage_for(i)
            clamped_count += int(clamped)
            dist = float(np.linalg.norm(pts[i] - pts[i - 1]))
            if int(p.emit_u_extrusion) and float(p.extrusion_per_mm) > 0.0:
                u_extr += dist * float(p.extrusion_per_mm)
                u_part = f" {u_axis}{u_extr:.5f}"
            else:
                u_part = ""
            if i == 1 or i == len(pts) - 1 or i % 100 == 0:
                fh.write(f"; u={u[i]:.5f} Bphys={b_phys[i]:.3f} C={cc:.3f}\n")
            fh.write(f"G1 {x_axis}{st[0]:.3f} {y_axis}{st[1]:.3f} {z_axis}{st[2]:.3f} {b_axis}{bc:.3f} {c_axis}{cc:.3f}{u_part} F{fmt_float(p.print_feed,3)}\n")

        if int(p.emit_pressure):
            fh.write("; close pressure solenoid\n")
            fh.write(f"M42 P{int(p.pressure_pin)} S0\n")
            if int(p.end_dwell_ms) > 0:
                fh.write(f"G4 P{int(p.end_dwell_ms)}\n")

        fh.write("; dwell at print end before pull-out\n")
        fh.write("G4 P2000\n")

        # Pull-out is defined in machine space from the actual final stage pose:
        # first move relatively by (-Z, X, Y), then make a horizontal +X exit move,
        # then rise back to the safe start Z.
        final_stage, final_bc, final_cc, final_clamped = stage_for(len(pts) - 1)
        clamped_count += int(final_clamped)
        exit_stage = final_stage.copy()
        exit_stage[2] -= float(p.post_print_drop_z_mm)
        exit_stage[0] += float(p.post_print_lead_x_mm)
        exit_stage[1] += float(p.post_print_lead_y_mm)
        exit_stage = clamp_stage_xyz(exit_stage, p)
        fh.write("; exit: move down in Z and offset in X/Y\n")
        fh.write(f"G1 {x_axis}{exit_stage[0]:.3f} {y_axis}{exit_stage[1]:.3f} {z_axis}{exit_stage[2]:.3f} {b_axis}{final_bc:.3f} {c_axis}{final_cc:.3f} F{fmt_float(p.approach_feed,3)}\n")
        exit_stage2 = exit_stage.copy()
        exit_stage2[0] += float(p.post_print_shift_x_mm)
        exit_stage2 = clamp_stage_xyz(exit_stage2, p)
        fh.write("; exit: move horizontally in +X\n")
        fh.write(f"G1 {x_axis}{exit_stage2[0]:.3f} {y_axis}{exit_stage2[1]:.3f} {z_axis}{exit_stage2[2]:.3f} {b_axis}{final_bc:.3f} {c_axis}{final_cc:.3f} F{fmt_float(p.approach_feed,3)}\n")
        exit_stage3 = exit_stage2.copy()
        exit_stage3[2] = float(p.safe_travel_z)
        exit_stage3 = clamp_stage_xyz(exit_stage3, p)
        fh.write("; exit: rise to safe Z\n")
        fh.write(f"G1 {x_axis}{exit_stage3[0]:.3f} {y_axis}{exit_stage3[1]:.3f} {z_axis}{exit_stage3[2]:.3f} {b_axis}{final_bc:.3f} {c_axis}{final_cc:.3f} F{fmt_float(p.travel_feed,3)}\n")
        fh.write("; END\n")

    return {
        "out": str(out_path),
        "point_count": int(len(pts)),
        "length_mm": total,
        "b_physical_range": [float(np.min(b_phys)), float(np.max(b_phys))],
        "c_range": [float(np.min(c_cmd)), float(np.max(c_cmd))],
        "tip_xyz_range": {
            "x": [float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))],
            "y": [float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))],
            "z": [float(np.min(pts[:, 2])), float(np.max(pts[:, 2]))],
        },
        "calibration_clamped_b_count": int(clamped_count),
    }


# =============================================================================
# Preview plotting
# =============================================================================

def set_axes_equal_3d(ax, pts: np.ndarray, pad: float = 8.0, z_stretch: float = 1.35) -> None:
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 0:
        return
    mins = np.nanmin(pts, axis=0)
    maxs = np.nanmax(pts, axis=0)
    centers = 0.5 * (mins + maxs)
    radius = 0.5 * float(np.max(maxs - mins)) + float(pad)
    if radius <= 1e-9:
        radius = 10.0
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)
    try:
        ax.set_box_aspect((1.0, 1.0, max(0.5, float(z_stretch))))
    except Exception:
        pass


def _style_3d_axes(ax) -> None:
    ax.set_facecolor((0, 0, 0, 0))
    ax.tick_params(colors=UI_FG, labelsize=7, pad=1)
    # Make the 3D panes transparent instead of the default opaque gray planes.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.set_facecolor((0.0, 0.0, 0.0, 0.0))
            axis.pane.set_edgecolor((0.7, 0.75, 0.8, 0.35))
            axis.pane.fill = False
        except Exception:
            pass
        try:
            axis._axinfo["grid"]["color"] = (0.55, 0.6, 0.65, 0.28)
            axis._axinfo["axisline"]["color"] = (0.75, 0.78, 0.82, 0.55)
            axis._axinfo["tick"]["color"] = UI_FG
        except Exception:
            pass


def _style_2d_axes(ax) -> None:
    ax.set_facecolor(UI_BG)
    ax.tick_params(colors=UI_FG, labelsize=7, pad=1)
    ax.grid(True, color=UI_GRID, alpha=0.34, linewidth=0.55)
    for spine in ax.spines.values():
        spine.set_color("#4d5563")


def plot_preview(fig: Figure, result: ExtractionResult, p: Params, preserve_view: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    fig.clear()
    # Middle panel: 3D view spanning both rows. Right panel: XZ over YZ debugging plots.
    gs = fig.add_gridspec(2, 2, width_ratios=[1.0, 1.0], height_ratios=[1.0, 1.0], wspace=0.24, hspace=0.50)
    ax3 = fig.add_subplot(gs[:, 0], projection="3d")
    ax_xz = fig.add_subplot(gs[0, 1])
    ax_yz = fig.add_subplot(gs[1, 1])
    _style_3d_axes(ax3)
    _style_2d_axes(ax_xz)
    _style_2d_axes(ax_yz)
    fig.patch.set_facecolor(UI_BG)

    pts = np.asarray(result.placed_tip_path, dtype=float)
    fit_clouds: List[np.ndarray] = [pts]
    s = result.arclength
    u = s / max(1e-9, float(s[-1]) if len(s) else 1.0)
    b = b_schedule(u, p)
    c = c_schedule(u, p)

    if int(p.show_mesh) and result.mesh_vertices is not None and result.mesh_faces is not None:
        verts = np.asarray(result.mesh_vertices, dtype=float)
        if len(verts):
            stride = max(1, int(p.preview_stride_mesh))
            vv = verts[::stride]
            if len(vv) > 25000:
                idx = np.linspace(0, len(vv) - 1, 25000).astype(int)
                vv = vv[idx]
            raw = result.raw_centerline
            if len(raw) >= 1:
                raw_origin = raw[0].copy()
                R = euler_rotation_matrix(p.rot_x_deg, p.rot_y_deg, p.rot_z_deg)
                vv2 = (vv - raw_origin) @ R.T
                vv2 *= float(p.trace_scale)
                vv2 = np.array([float(p.start_x), float(p.start_y), float(p.start_z)]) + vv2
                fit_clouds.append(np.asarray(vv2, dtype=float))
                ax3.scatter(vv2[:, 0], vv2[:, 1], vv2[:, 2], s=0.35, alpha=0.075, c="#9aa5b1")

    if int(p.show_tip_path) and len(pts):
        ax3.plot(pts[:, 0], pts[:, 1], pts[:, 2], lw=1.9, color=UI_ACCENT, label="centerline")
        ax_xz.plot(pts[:, 0], pts[:, 2], lw=1.55, color=UI_ACCENT, label="XZ centerline")
        ax_yz.plot(pts[:, 1], pts[:, 2], lw=1.55, color=UI_ACCENT, label="YZ centerline")
    if int(p.show_b_c_markers) and len(pts) > 5:
        idxs = np.linspace(0, len(pts) - 1, 9).astype(int)
        ax3.scatter(pts[idxs, 0], pts[idxs, 1], pts[idxs, 2], s=9, c="#ff2bd6")
        for i in idxs:
            ax_xz.text(pts[i, 0], pts[i, 2], f"C{c[i]:.0f}", color="#e0c000", fontsize=5.5)
            ax_yz.text(pts[i, 1], pts[i, 2], f"C{c[i]:.0f}", color="#e0c000", fontsize=5.5)
    if len(pts):
        ax3.scatter([pts[0, 0]], [pts[0, 1]], [pts[0, 2]], s=34, c="#33ff33", label="start")
        ax3.scatter([pts[-1, 0]], [pts[-1, 1]], [pts[-1, 2]], s=34, c="#ff3333", label="end")
        ax_xz.scatter([pts[0, 0]], [pts[0, 2]], s=22, c="#33ff33")
        ax_xz.scatter([pts[-1, 0]], [pts[-1, 2]], s=22, c="#ff3333")
        ax_yz.scatter([pts[0, 1]], [pts[0, 2]], s=22, c="#33ff33")
        ax_yz.scatter([pts[-1, 1]], [pts[-1, 2]], s=22, c="#ff3333")

    ax3.set_title("3D centerline", color=UI_FG, fontsize=10, pad=5)
    ax3.set_xlabel("X tip (mm)", color=UI_FG, fontsize=8, labelpad=3)
    ax3.set_ylabel("Y tip (mm)", color=UI_FG, fontsize=8, labelpad=3)
    ax3.set_zlabel("Z tip (mm)", color=UI_FG, fontsize=8, labelpad=6)
    ax3.grid(True)

    legend_kw = dict(loc="best", facecolor="#0b0b0b", edgecolor="#333a44", labelcolor=UI_FG, fontsize=6.5, framealpha=0.55, borderpad=0.35, handlelength=1.6)

    ax_xz.set_title("XZ projection", color=UI_FG, fontsize=10, pad=5)
    ax_xz.set_xlabel("X tip (mm)", color=UI_FG, fontsize=8)
    ax_xz.set_ylabel("Z tip (mm)", color=UI_FG, fontsize=8)
    ax_xz.axis("equal")
    ax_xz.legend(**legend_kw)

    ax_yz.set_title("YZ projection", color=UI_FG, fontsize=10, pad=5)
    ax_yz.set_xlabel("Y tip (mm)", color=UI_FG, fontsize=8)
    ax_yz.set_ylabel("Z tip (mm)", color=UI_FG, fontsize=8)
    ax_yz.axis("equal")
    ax_yz.legend(**legend_kw)
    ax3.legend(**legend_kw)

    if preserve_view and preserve_view.get("has_view"):
        try:
            ax3.view_init(elev=preserve_view.get("elev"), azim=preserve_view.get("azim"), roll=preserve_view.get("roll", 0))
        except TypeError:
            ax3.view_init(elev=preserve_view.get("elev"), azim=preserve_view.get("azim"))
        if preserve_view.get("xlim") is not None:
            ax3.set_xlim(preserve_view["xlim"])
            ax3.set_ylim(preserve_view["ylim"])
            ax3.set_zlim(preserve_view["zlim"])
        try:
            ax3.set_box_aspect((1.0, 1.0, max(0.5, float(getattr(p, "view_z_stretch", DEFAULT_VIEW_Z_STRETCH)))))
        except Exception:
            pass
    else:
        fit_pts = np.vstack([cloud for cloud in fit_clouds if cloud is not None and len(cloud)])
        set_axes_equal_3d(ax3, fit_pts, pad=10.0, z_stretch=float(getattr(p, "view_z_stretch", DEFAULT_VIEW_Z_STRETCH)))
        ax3.view_init(elev=24, azim=-58)

    fig.subplots_adjust(left=0.045, right=0.985, bottom=0.075, top=0.925, wspace=0.24, hspace=0.50)
    return {
        "has_view": True,
        "elev": getattr(ax3, "elev", None),
        "azim": getattr(ax3, "azim", None),
        "roll": getattr(ax3, "roll", 0),
        "xlim": ax3.get_xlim(),
        "ylim": ax3.get_ylim(),
        "zlim": ax3.get_zlim(),
    }

def save_preview_png(result: ExtractionResult, p: Params, out_png: str) -> None:
    fig = Figure(figsize=(16, 8), dpi=max(120, int(p.plot_dpi)))
    plot_preview(fig, result, p, preserve_view=None)
    out = Path(out_png)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out), dpi=max(160, int(p.plot_dpi)), facecolor=UI_BG)


# =============================================================================
# GUI
# =============================================================================

PARAM_GROUPS: List[Tuple[str, List[Tuple[str, str, float, float, float]]]] = [
    ("Mesh / extraction", [
        ("mesh_unit_scale", "Mesh unit scale", 0.001, 100.0, 0.001),
        ("trace_scale", "Final path scale", 0.05, 10.0, 0.01),
        ("interior_x_scale", "Interior X scale", 0.2, 5.0, 0.01),
        ("y_to_x_projection_gain", "Y->X projection gain", -3.0, 3.0, 0.01),
        ("symmetric_loop_x_spread_mm", "Symmetric loop X spread", -20.0, 20.0, 0.25),
        ("symmetric_loop_x_spread_width_mm", "Loop spread width", 1.0, 40.0, 0.5),
        ("mesh_tube_diameter", "Tube dia helper", 0.5, 12.0, 0.05),
        ("voxel_pitch", "Voxel pitch", 0.1, 3.0, 0.05),
        ("endpoint_span_mm", "Endpoint cap span", 0.2, 12.0, 0.1),
        ("surface_ring_spacing_mm", "Surface ring spacing", 0.15, 3.0, 0.05),
        ("surface_ring_band_mm", "Surface ring band", 0.15, 3.0, 0.05),
        ("surface_geodesic_refine_slices", "Refine geodesic slices", 0, 1, 1),
        ("surface_geodesic_lock_xz", "Keep geodesic XZ", 0, 1, 1),
        ("vertical_branch_length_mm", "Vertical tail length", 0.0, 40.0, 0.5),
        ("vertical_branch_blend_mm", "Vertical tail blend", 0.0, 40.0, 0.5),
        ("vertical_branch_flare_mm", "Vertical tail flare", 0.0, 16.0, 0.25),
        ("vertical_tail_no_wobble_mm", "Vertical no-wobble", 0.0, 40.0, 0.5),
        ("vertical_tail_no_wobble_fade_mm", "No-wobble fade", 0.0, 30.0, 0.5),
        ("top_c0_x_offset_mm", "Top C0 X offset", -30.0, 30.0, 0.5),
        ("bottom_c_target_x_offset_mm", "Bottom C-target X offset", -30.0, 30.0, 0.5),
        ("bottom_c_target_deg", "Bottom C target", -270.0, 0.0, 1.0),
        ("marker_x_offset_width_mm", "Marker X offset width", 1.0, 30.0, 0.5),
        ("plane_march_step_mm", "March step", 0.15, 2.0, 0.05),
        ("plane_march_gate_radius_mm", "Slice gate radius", 1.8, 12.0, 0.1),
        ("plane_march_cluster_radius_mm", "Slice cluster radius", 0.25, 3.0, 0.05),
        ("plane_march_max_steps", "March max steps", 100, 8000, 50),
        ("centerline_clearance_weight", "Centerline wall penalty", 0.0, 80.0, 1.0),
        ("centerline_clearance_power", "Wall penalty power", 1.0, 8.0, 0.25),
        ("smooth_sigma_mm", "Smooth sigma mm", 0.0, 8.0, 0.05),
        ("resample_spacing_mm", "Resample spacing", 0.1, 3.0, 0.05),
        ("view_z_stretch", "3D Z stretch", 0.8, 2.4, 0.05),
    ]),
    ("Placement", [
        ("start_x", "Start X", 0.0, 200.0, 0.5),
        ("start_y", "Start Y", -80.0, 120.0, 0.5),
        ("start_z", "Start Z", -250.0, 30.0, 0.5),
        ("bottom_branch_y_offset", "Bottom branch Y", -80.0, 120.0, 0.5),
        ("top_branch_y_offset", "Top branch Y", -80.0, 120.0, 0.5),
    ]),
    ("Orientation", [
        ("rot_x_deg", "Rot X deg", -180.0, 180.0, 1.0),
        ("rot_y_deg", "Rot Y deg", -180.0, 180.0, 1.0),
        ("rot_z_deg", "Rot Z deg", -180.0, 180.0, 1.0),
    ]),
    ("B schedule", [
        ("b_start_deg", "B start", 0.0, 180.0, 1.0),
        ("b_mid_deg", "B curled", 0.0, 180.0, 1.0),
        ("b_end_deg", "B end", -180.0, 180.0, 1.0),
        ("b_ramp_start", "B ramp start", 0.0, 1.0, 0.005),
        ("b_ramp_end", "B ramp end", 0.0, 1.0, 0.005),
        ("b_release_start", "B release start", 0.0, 1.0, 0.005),
        ("b_release_end", "B release end", 0.0, 1.0, 0.005),
    ]),
    ("C schedule", [
        ("c_start_deg", "C start", -360.0, 360.0, 1.0),
        ("c_top_loop_deg", "C top-loop azimuth", -180.0, 180.0, 1.0),
        ("c_top_loop_start", "C top-loop start", 0.0, 1.0, 0.005),
        ("c_top_loop_end", "C top-loop end", 0.0, 1.0, 0.005),
        ("c_end_deg", "C final", -540.0, 540.0, 1.0),
        ("c_ramp_start", "C ramp start", 0.0, 1.0, 0.005),
        ("c_ramp_end", "C ramp end", 0.0, 1.0, 0.005),
    ]),
    ("Motion", [
        ("safe_travel_z", "Safe travel Z", -250.0, 30.0, 0.5),
        ("post_print_drop_z_mm", "Post-print drop Z", 0.0, 100.0, 0.5),
        ("post_print_lead_x_mm", "Post-print lead X", -100.0, 100.0, 0.5),
        ("post_print_lead_y_mm", "Post-print lead Y", -100.0, 100.0, 0.5),
        ("post_print_shift_x_mm", "Post-print exit +X", 0.0, 100.0, 0.5),
        ("travel_feed", "Travel feed", 10.0, 5000.0, 10.0),
        ("print_feed", "Print feed", 5.0, 2000.0, 5.0),
        ("extrusion_per_mm", "U extrusion/mm", 0.0, 0.2, 0.0005),
    ]),
]


class ScrollFrame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        canvas = tk.Canvas(self, bg=UI_PANEL_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(self, orient="vertical", command=canvas.yview)
        self.scrollable = ttk.Frame(canvas)
        self.scrollable.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.scrollable, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")


class MeshCenterlineGUI:
    def __init__(self, root: tk.Tk, p: Params):
        self.root = root
        self.p = p
        self.vars: Dict[str, tk.Variable] = {}
        self.result: Optional[ExtractionResult] = None
        self.current_view: Optional[Dict[str, Any]] = None
        self._redraw_after_id: Optional[str] = None
        self._extract_after_id: Optional[str] = None

        self.root.title("Mesh centerline pretzel / overhand knot tracer")
        self.root.configure(bg=UI_BG)
        self._configure_style()

        main = ttk.PanedWindow(root, orient=tk.HORIZONTAL)
        main.pack(fill="both", expand=True)
        # Default layout: 1/3 controls, then a 2/3 Matplotlib area split
        # equally into 3D and XZ subplots.
        left = ttk.Frame(main, width=600)
        right = ttk.Frame(main, width=1200)
        main.add(left, weight=1)
        main.add(right, weight=2)
        self.root.after(250, lambda: self._set_initial_pane_sash(main))

        self._build_controls(left)
        self.fig = Figure(figsize=(13.8, 7.6), dpi=int(self.p.plot_dpi))
        if FigureCanvasTkAgg is None:
            raise RuntimeError("Matplotlib TkAgg backend is unavailable.")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        self.status = tk.StringVar(value="Ready. Load a mesh or use the built-in demo centerline.")
        ttk.Label(right, textvariable=self.status).pack(fill="x", padx=6, pady=4)

        self.extract_and_redraw(reset_view=True)

    def _set_initial_pane_sash(self, main) -> None:
        try:
            w = max(900, int(self.root.winfo_width()))
            main.sashpos(0, int(w / 3))
        except Exception:
            pass

    def _configure_style(self) -> None:
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=UI_PANEL_BG)
        style.configure("TLabelframe", background=UI_PANEL_BG, foreground=UI_FG)
        style.configure("TLabelframe.Label", background=UI_PANEL_BG, foreground=UI_ACCENT)
        style.configure("TLabel", background=UI_PANEL_BG, foreground=UI_FG)
        style.configure("TButton", background="#111111", foreground=UI_FG)
        style.configure("TCheckbutton", background=UI_PANEL_BG, foreground=UI_FG)
        style.configure("TEntry", fieldbackground=UI_ENTRY_BG, foreground=UI_FG)
        style.configure("Horizontal.TScale", background=UI_PANEL_BG)

    def _build_controls(self, parent) -> None:
        top = ttk.Frame(parent)
        top.pack(fill="x", padx=6, pady=6)
        ttk.Label(top, text="Mesh path").pack(anchor="w")
        row = ttk.Frame(top)
        row.pack(fill="x")
        self.vars["mesh_path"] = tk.StringVar(value=self.p.mesh_path)
        ttk.Entry(row, textvariable=self.vars["mesh_path"], width=32).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Browse", command=self.browse_mesh).pack(side="left", padx=3)
        ttk.Button(row, text="Extract", command=lambda: self.extract_and_redraw(reset_view=False)).pack(side="left", padx=3)

        method_row = ttk.Frame(top)
        method_row.pack(fill="x", pady=(6, 0))
        ttk.Label(method_row, text="Extractor", width=10).pack(side="left")
        self.vars["extraction_method"] = tk.StringVar(value=self.p.extraction_method)
        method_box = ttk.Combobox(
            method_row,
            textvariable=self.vars["extraction_method"],
            values=["surface_geodesic", "plane_march", "distance_path", "skeleton"],
            state="readonly",
            width=18,
        )
        method_box.pack(side="left", fill="x", expand=True)
        method_box.bind("<<ComboboxSelected>>", lambda e: self.extract_and_redraw(reset_view=False))

        ttk.Label(top, text="Calibration JSON").pack(anchor="w", pady=(8, 0))
        row2 = ttk.Frame(top)
        row2.pack(fill="x")
        self.vars["calibration_path"] = tk.StringVar(value=self.p.calibration_path)
        ttk.Entry(row2, textvariable=self.vars["calibration_path"], width=32).pack(side="left", fill="x", expand=True)
        ttk.Button(row2, text="Browse", command=self.browse_calibration).pack(side="left", padx=3)

        cal_opts = ttk.Frame(top)
        cal_opts.pack(fill="x", pady=(5, 0))
        ttk.Label(cal_opts, text="Equation", width=10).pack(side="left")
        self.vars["fit_model"] = tk.StringVar(value=self.p.fit_model)
        self.fit_model_box = ttk.Combobox(
            cal_opts,
            textvariable=self.vars["fit_model"],
            values=["pchip", "avg_pchip", "cubic", "avg_cubic", "linear", "avg_linear"],
            state="readonly",
            width=11,
        )
        self.fit_model_box.pack(side="left", padx=(0, 3))
        self.fit_model_box.bind("<<ComboboxSelected>>", lambda e: self._on_param_change())
        ttk.Label(cal_opts, text="Yoff", width=5).pack(side="left")
        self.vars["offplane_fit_model"] = tk.StringVar(value=self.p.offplane_fit_model)
        self.offplane_model_box = ttk.Combobox(
            cal_opts,
            textvariable=self.vars["offplane_fit_model"],
            values=["pchip", "avg_pchip", "cubic", "avg_cubic", "linear", "avg_linear"],
            state="readonly",
            width=11,
        )
        self.offplane_model_box.pack(side="left", padx=(0, 3))
        self.offplane_model_box.bind("<<ComboboxSelected>>", lambda e: self._on_param_change())

        phase_row = ttk.Frame(top)
        phase_row.pack(fill="x", pady=(5, 0))
        ttk.Label(phase_row, text="Phase set", width=10).pack(side="left")
        self.vars["calibration_phase"] = tk.StringVar(value=self.p.calibration_phase)
        self.phase_box = ttk.Combobox(
            phase_row,
            textvariable=self.vars["calibration_phase"],
            values=self._calibration_phase_values(),
            state="readonly",
            width=22,
        )
        self.phase_box.pack(side="left", fill="x", expand=True)
        self.phase_box.bind("<<ComboboxSelected>>", lambda e: self._on_param_change())

        ttk.Label(top, text="Output G-code").pack(anchor="w", pady=(8, 0))
        row3 = ttk.Frame(top)
        row3.pack(fill="x")
        self.vars["output_path"] = tk.StringVar(value=self.p.output_path)
        ttk.Entry(row3, textvariable=self.vars["output_path"], width=32).pack(side="left", fill="x", expand=True)
        ttk.Button(row3, text="Save as", command=self.browse_output).pack(side="left", padx=3)

        mode_row = ttk.Frame(top)
        mode_row.pack(fill="x", pady=(8, 2))
        self.vars["write_mode"] = tk.StringVar(value=self.p.write_mode)
        ttk.Radiobutton(mode_row, text="Calibrated", variable=self.vars["write_mode"], value="calibrated").pack(side="left")
        ttk.Radiobutton(mode_row, text="Cartesian", variable=self.vars["write_mode"], value="cartesian").pack(side="left")

        checks = ttk.Frame(top)
        checks.pack(fill="x", pady=4)
        for key, label in [
            ("force_branch_y", "Force branch Y offsets"),
            ("force_coaxial_branches", "Coaxial vertical tails"),
            ("enable_marker_x_offsets", "C-marker X offsets"),
            ("enable_xz_projection_shape", "XZ projection shaping"),
            ("enable_top_loop_c_azimuth", "Top-loop C azimuth"),
            ("reverse_path", "Reverse trace"),
            ("emit_pressure", "Pressure M42"),
            ("emit_u_extrusion", "Emit U extrusion"),
            ("show_mesh", "Show mesh"),
            ("show_tip_path", "Show path"),
            ("show_b_c_markers", "Show B/C markers"),
        ]:
            self.vars[key] = tk.IntVar(value=int(getattr(self.p, key)))
            cb = ttk.Checkbutton(checks, text=label, variable=self.vars[key], command=self._on_param_change)
            cb.pack(anchor="w")

        sf = ScrollFrame(parent)
        sf.pack(fill="both", expand=True, padx=6, pady=6)
        for group_name, specs in PARAM_GROUPS:
            lf = ttk.LabelFrame(sf.scrollable, text=group_name)
            lf.pack(fill="x", expand=True, pady=4)
            for key, label, vmin, vmax, step in specs:
                self._add_slider(lf, key, label, vmin, vmax, step)

        buttons = ttk.Frame(parent)
        buttons.pack(fill="x", padx=6, pady=6)
        ttk.Button(buttons, text="Export G-code", command=self.export_gcode).pack(fill="x", pady=2)
        ttk.Button(buttons, text="Save preview PNG", command=self.save_png).pack(fill="x", pady=2)
        ttk.Button(buttons, text="Save preset", command=self.save_preset).pack(fill="x", pady=2)
        ttk.Button(buttons, text="Load preset", command=self.load_preset).pack(fill="x", pady=2)

    def _add_slider(self, parent, key: str, label: str, vmin: float, vmax: float, step: float) -> None:
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=2)
        ttk.Label(row, text=label, width=18).pack(side="left")
        var = tk.DoubleVar(value=float(getattr(self.p, key)))
        self.vars[key] = var
        scale = ttk.Scale(row, from_=vmin, to=vmax, variable=var, orient="horizontal", command=lambda _=None: self._on_param_change())
        scale.pack(side="left", fill="x", expand=True, padx=4)
        ent = ttk.Entry(row, width=9, textvariable=var)
        ent.pack(side="left")
        ent.bind("<Return>", lambda e: self._on_param_change(force_extract=True))
        ent.bind("<FocusOut>", lambda e: self._on_param_change(force_extract=True))

    def _calibration_phase_values(self) -> List[str]:
        phases = available_calibration_phases(str(self.vars.get("calibration_path", tk.StringVar(value=self.p.calibration_path)).get()))
        values = ["auto"] + phases
        # A few common aliases are useful before a file is loaded.
        for fallback in ["pull", "release", "pull_1", "release_1", "pull_2", "release_2"]:
            if fallback not in values:
                values.append(fallback)
        return values

    def _refresh_calibration_phase_values(self) -> None:
        try:
            values = self._calibration_phase_values()
            self.phase_box.configure(values=values)
            cur = str(self.vars["calibration_phase"].get())
            if cur not in values:
                self.vars["calibration_phase"].set("auto")
        except Exception:
            pass

    def browse_mesh(self) -> None:
        path = filedialog.askopenfilename(title="Select STL or 3MF mesh", filetypes=[("Mesh files", "*.stl *.3mf *.obj *.ply"), ("All files", "*.*")])
        if path:
            self.vars["mesh_path"].set(path)
            self.extract_and_redraw(reset_view=False)

    def browse_calibration(self) -> None:
        path = filedialog.askopenfilename(title="Select calibration JSON", filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if path:
            self.vars["calibration_path"].set(path)
            self._refresh_calibration_phase_values()

    def browse_output(self) -> None:
        path = filedialog.asksaveasfilename(title="Output G-code", defaultextension=".gcode", filetypes=[("G-code", "*.gcode *.gco *.nc"), ("All files", "*.*")])
        if path:
            self.vars["output_path"].set(path)

    def _read_params_from_vars(self) -> None:
        for f in fields(Params):
            key = f.name
            if key not in self.vars:
                continue
            var = self.vars[key]
            try:
                if isinstance(var, tk.IntVar):
                    value = int(var.get())
                elif isinstance(var, tk.DoubleVar):
                    value = float(var.get())
                else:
                    value = str(var.get())
                setattr(self.p, key, value)
            except Exception:
                pass

    def _capture_view(self) -> Optional[Dict[str, Any]]:
        if not self.fig.axes:
            return self.current_view
        for ax in self.fig.axes:
            if hasattr(ax, "get_zlim"):
                return {
                    "has_view": True,
                    "elev": getattr(ax, "elev", None),
                    "azim": getattr(ax, "azim", None),
                    "roll": getattr(ax, "roll", 0),
                    "xlim": ax.get_xlim(),
                    "ylim": ax.get_ylim(),
                    "zlim": ax.get_zlim(),
                }
        return self.current_view

    def _on_param_change(self, force_extract: bool = False) -> None:
        self._read_params_from_vars()
        # Geometry-affecting params require re-extraction/placement. Toggle-only changes redraw only.
        if force_extract:
            self.extract_and_redraw(reset_view=False)
            return
        if self._redraw_after_id is not None:
            self.root.after_cancel(self._redraw_after_id)
        self._redraw_after_id = self.root.after(180, self.redraw_only)

    def extract_and_redraw(self, reset_view: bool = False) -> None:
        self._read_params_from_vars()
        view = None if reset_view else self._capture_view()
        try:
            self.status.set("Extracting centerline... this can take a few seconds for fine voxel pitch.")
            self.root.update_idletasks()
            self.result = build_extraction(self.p)
            msg = f"Centerline points: {len(self.result.placed_tip_path)}"
            if self.result.warnings:
                msg += " | " + " | ".join(self.result.warnings[:2])
            self.status.set(msg)
            self.current_view = plot_preview(self.fig, self.result, self.p, preserve_view=view)
            self.canvas.draw_idle()
        except Exception as exc:
            self.status.set(f"Error: {exc}")
            traceback.print_exc()
            messagebox.showerror("Extraction error", str(exc))

    def redraw_only(self) -> None:
        self._read_params_from_vars()
        self._redraw_after_id = None
        view = self._capture_view()
        try:
            # Rebuild placement/smoothing in case sliders changed. If mesh path/extraction changes, use Extract.
            if self.result is None:
                self.result = build_extraction(self.p)
            else:
                _, placed, s = apply_transform_and_place(self.result.raw_centerline, self.p)
                self.result.placed_tip_path = placed
                self.result.arclength = s
            self.current_view = plot_preview(self.fig, self.result, self.p, preserve_view=view)
            self.canvas.draw_idle()
        except Exception as exc:
            self.status.set(f"Error: {exc}")
            traceback.print_exc()

    def export_gcode(self) -> None:
        self._read_params_from_vars()
        try:
            if self.result is None:
                self.result = build_extraction(self.p)
            summary = write_gcode(self.result.placed_tip_path, self.p)
            self.status.set(f"Exported {summary['point_count']} points to {summary['out']}")
            messagebox.showinfo("Export complete", json.dumps(summary, indent=2))
        except Exception as exc:
            self.status.set(f"Export error: {exc}")
            traceback.print_exc()
            messagebox.showerror("Export error", str(exc))

    def save_png(self) -> None:
        self._read_params_from_vars()
        path = filedialog.asksaveasfilename(title="Save preview PNG", defaultextension=".png", filetypes=[("PNG", "*.png"), ("All files", "*.*")])
        if not path:
            return
        try:
            if self.result is None:
                self.result = build_extraction(self.p)
            save_preview_png(self.result, self.p, path)
            self.status.set(f"Saved preview {path}")
        except Exception as exc:
            messagebox.showerror("Preview error", str(exc))

    def save_preset(self) -> None:
        self._read_params_from_vars()
        path = filedialog.asksaveasfilename(title="Save preset", defaultextension=".json", filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if path:
            preset = params_to_preset_dict(self.p)
            Path(path).write_text(json.dumps(preset, indent=2), encoding="utf-8")
            self.status.set(f"Saved preset {path}")

    def load_preset(self) -> None:
        path = filedialog.askopenfilename(title="Load preset", filetypes=[("JSON", "*.json"), ("All files", "*.*")])
        if not path:
            return
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        ignored = apply_preset_data(self.p, data)
        for f in fields(Params):
            key = f.name
            if key not in self.vars:
                continue
            try:
                self.vars[key].set(getattr(self.p, key))
            except Exception:
                pass
        self._refresh_calibration_phase_values()
        if ignored:
            self.status.set(f"Loaded preset {path} (ignored unknown keys: {', '.join(ignored[:4])}{'...' if len(ignored) > 4 else ''})")
        else:
            self.status.set(f"Loaded preset {path}")
        self.extract_and_redraw(reset_view=False)


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Extract a knot centerline from STL/3MF and export robot G-code.")
    ap.add_argument("--mesh", dest="mesh_path", default=DEFAULT_MESH_PATH, help="STL/3MF/OBJ/PLY mesh path. If omitted, the demo path is used.")
    ap.add_argument("--calibration", dest="calibration_path", default=DEFAULT_CALIBRATION_PATH, help="Calibration JSON for calibrated mode.")
    ap.add_argument("--out", dest="output_path", default=DEFAULT_OUT, help="Output G-code path.")
    ap.add_argument("--preview-png", dest="preview_png_path", default=DEFAULT_PREVIEW_PNG, help="Optional preview PNG path.")
    ap.add_argument("--nogui", action="store_true", help="Run headless export.")
    ap.add_argument("--cartesian", action="store_true", help="Use Cartesian output, no calibration required.")
    ap.add_argument("--mesh-unit-scale", type=float, default=DEFAULT_MESH_UNIT_SCALE)
    ap.add_argument("--trace-scale", type=float, default=DEFAULT_TRACE_SCALE, help="Final uniform robot-path scale about the bottom endpoint. Use for true whole-path scaling only. For stronger knot overlap in XZ, keep this at 1 and use --interior-x-scale / --y-to-x-projection-gain.")
    ap.add_argument("--interior-x-scale", type=float, default=DEFAULT_INTERIOR_X_SCALE, help="Scale only the interior knot X offsets about the vertical tail axis; does not stretch Z/tails.")
    ap.add_argument("--y-to-x-projection-gain", type=float, default=DEFAULT_Y_TO_X_PROJECTION_GAIN, help="Add gain*(Y-anchor_Y) into X in the knot interior; useful when the loop shows in YZ but not XZ.")
    ap.add_argument("--symmetric-loop-x-spread-mm", type=float, default=DEFAULT_SYMMETRIC_LOOP_X_SPREAD_MM, help="Symmetric outward X spread applied to upper/lower lobes after extraction.")
    ap.add_argument("--symmetric-loop-x-spread-width-mm", type=float, default=DEFAULT_SYMMETRIC_LOOP_X_SPREAD_WIDTH_MM)
    ap.add_argument("--no-xz-projection-shape", action="store_true", help="Disable interior XZ projection shaping.")
    ap.add_argument("--fit-model", type=str, default=DEFAULT_FIT_MODEL, help="Calibration r/z/tip-angle model selector; default pchip.")
    ap.add_argument("--offplane-fit-model", type=str, default=DEFAULT_OFFPLANE_FIT_MODEL, help="Calibration off-plane Y model selector; default pchip.")
    ap.add_argument("--calibration-phase", type=str, default=DEFAULT_CALIBRATION_PHASE, help="Calibration fit_models_by_phase key/equation set. Use auto for the calibration file default.")
    ap.add_argument("--offplane-sign", type=float, default=DEFAULT_OFFPLANE_SIGN, help="Multiplier for off-plane Y calibration component; default -1.")
    ap.add_argument("--voxel-pitch", type=float, default=DEFAULT_VOXEL_PITCH)
    ap.add_argument("--extraction-method", choices=["plane_march", "surface_geodesic", "distance_path", "skeleton"], default=DEFAULT_EXTRACTION_METHOD)
    ap.add_argument("--endpoint-span-mm", type=float, default=DEFAULT_ENDPOINT_SPAN_MM)
    ap.add_argument("--surface-ring-spacing-mm", type=float, default=DEFAULT_SURFACE_RING_SPACING_MM)
    ap.add_argument("--surface-ring-band-mm", type=float, default=DEFAULT_SURFACE_RING_BAND_MM)
    ap.add_argument("--surface-geodesic-refine-slices", type=int, choices=[0, 1], default=DEFAULT_SURFACE_GEODESIC_REFINE_SLICES)
    ap.add_argument("--surface-geodesic-lock-xz", type=int, choices=[0, 1], default=DEFAULT_SURFACE_GEODESIC_LOCK_XZ)
    ap.add_argument("--plane-march-step-mm", type=float, default=DEFAULT_PLANE_MARCH_STEP_MM)
    ap.add_argument("--plane-march-gate-radius-mm", type=float, default=DEFAULT_PLANE_MARCH_GATE_RADIUS_MM)
    ap.add_argument("--plane-march-cluster-radius-mm", type=float, default=DEFAULT_PLANE_MARCH_CLUSTER_RADIUS_MM)
    ap.add_argument("--plane-march-max-steps", type=int, default=DEFAULT_PLANE_MARCH_MAX_STEPS)
    ap.add_argument("--centerline-clearance-weight", type=float, default=DEFAULT_CENTERLINE_CLEARANCE_WEIGHT)
    ap.add_argument("--centerline-clearance-power", type=float, default=DEFAULT_CENTERLINE_CLEARANCE_POWER)
    ap.add_argument("--smooth-sigma-mm", type=float, default=DEFAULT_SMOOTH_SIGMA_MM)
    ap.add_argument("--resample-spacing-mm", type=float, default=DEFAULT_RESAMPLE_SPACING_MM)
    ap.add_argument("--start-x", type=float, default=DEFAULT_START_X)
    ap.add_argument("--start-y", type=float, default=DEFAULT_START_Y)
    ap.add_argument("--start-z", type=float, default=DEFAULT_START_Z)
    ap.add_argument("--bottom-branch-y-offset", type=float, default=DEFAULT_BOTTOM_BRANCH_Y_OFFSET)
    ap.add_argument("--top-branch-y-offset", type=float, default=DEFAULT_TOP_BRANCH_Y_OFFSET)
    ap.add_argument("--no-force-branch-y", action="store_true")
    ap.add_argument("--force-branch-y", action="store_true", help="Enable legacy bottom/top branch Y warp.")
    ap.add_argument("--force-coaxial-branches", action="store_true", help="Enable the same-XY vertical endpoint constraint.")
    ap.add_argument("--no-force-coaxial-branches", action="store_true", help="Disable the same-XY vertical endpoint constraint.")
    ap.add_argument("--vertical-branch-length-mm", type=float, default=DEFAULT_VERTICAL_BRANCH_LENGTH_MM)
    ap.add_argument("--vertical-branch-blend-mm", type=float, default=DEFAULT_VERTICAL_BRANCH_BLEND_MM)
    ap.add_argument("--vertical-branch-flare-mm", type=float, default=DEFAULT_VERTICAL_BRANCH_FLARE_MM)
    ap.add_argument("--vertical-tail-no-wobble-mm", type=float, default=DEFAULT_VERTICAL_TAIL_NO_WOBBLE_MM, help="Final-path length kept perfectly vertical at the bottom and top tails.")
    ap.add_argument("--vertical-tail-no-wobble-fade-mm", type=float, default=DEFAULT_VERTICAL_TAIL_NO_WOBBLE_FADE_MM, help="Smooth fade length from the locked vertical tail into the extracted knot.")
    ap.add_argument("--marker-x-offsets", action="store_true", help="Enable local X offsets tied to C-marker landmarks.")
    ap.add_argument("--no-marker-x-offsets", action="store_true", help="Disable the local X offsets tied to C-marker landmarks.")
    ap.add_argument("--top-c0-x-offset-mm", type=float, default=DEFAULT_TOP_C0_X_OFFSET_MM)
    ap.add_argument("--bottom-c-target-deg", type=float, default=DEFAULT_BOTTOM_C_TARGET_DEG)
    ap.add_argument("--bottom-c-target-x-offset-mm", type=float, default=DEFAULT_BOTTOM_C_TARGET_X_OFFSET_MM)
    ap.add_argument("--marker-x-offset-width-mm", type=float, default=DEFAULT_MARKER_X_OFFSET_WIDTH_MM)
    ap.add_argument("--reverse-path", action="store_true")
    ap.add_argument("--rot-x-deg", type=float, default=DEFAULT_ROT_X_DEG)
    ap.add_argument("--rot-y-deg", type=float, default=DEFAULT_ROT_Y_DEG)
    ap.add_argument("--rot-z-deg", type=float, default=DEFAULT_ROT_Z_DEG)
    ap.add_argument("--c-end-deg", type=float, default=DEFAULT_C_END_DEG, help="Default is C-180 for the last pass.")
    ap.add_argument("--c-top-loop-deg", type=float, default=DEFAULT_C_TOP_LOOP_DEG, help="Upper-loop azimuth C value before the final C ramp; default 90.")
    ap.add_argument("--c-top-loop-start", type=float, default=DEFAULT_C_TOP_LOOP_START)
    ap.add_argument("--c-top-loop-end", type=float, default=DEFAULT_C_TOP_LOOP_END)
    ap.add_argument("--no-top-loop-c-azimuth", action="store_true", help="Disable the C0->C top-loop azimuth lead-in.")
    ap.add_argument("--c-ramp-start", type=float, default=DEFAULT_C_RAMP_START)
    ap.add_argument("--c-ramp-end", type=float, default=DEFAULT_C_RAMP_END)
    ap.add_argument("--b-ramp-start", type=float, default=DEFAULT_B_RAMP_START)
    ap.add_argument("--b-ramp-end", type=float, default=DEFAULT_B_RAMP_END)
    ap.add_argument("--safe-travel-z", type=float, default=DEFAULT_SAFE_TRAVEL_Z)
    ap.add_argument("--post-print-drop-z-mm", type=float, default=DEFAULT_POST_PRINT_DROP_Z_MM)
    ap.add_argument("--post-print-lead-x-mm", type=float, default=DEFAULT_POST_PRINT_LEAD_X_MM)
    ap.add_argument("--post-print-shift-x-mm", type=float, default=DEFAULT_POST_PRINT_SHIFT_X_MM)
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED)
    ap.add_argument("--view-z-stretch", type=float, default=DEFAULT_VIEW_Z_STRETCH)
    return ap


def params_from_args(args: argparse.Namespace) -> Params:
    p = Params()
    for f in fields(Params):
        if hasattr(args, f.name):
            setattr(p, f.name, getattr(args, f.name))
    if getattr(args, "cartesian", False):
        p.write_mode = "cartesian"
    if getattr(args, "no_force_branch_y", False):
        p.force_branch_y = 0
    if getattr(args, "force_branch_y", False):
        p.force_branch_y = 1
    if getattr(args, "force_coaxial_branches", False):
        p.force_coaxial_branches = 1
    if getattr(args, "no_force_coaxial_branches", False):
        p.force_coaxial_branches = 0
    if getattr(args, "reverse_path", False):
        p.reverse_path = 1
    if getattr(args, "marker_x_offsets", False):
        p.enable_marker_x_offsets = 1
    if getattr(args, "no_marker_x_offsets", False):
        p.enable_marker_x_offsets = 0
    if getattr(args, "no_xz_projection_shape", False):
        p.enable_xz_projection_shape = 0
    if getattr(args, "no_top_loop_c_azimuth", False):
        p.enable_top_loop_c_azimuth = 0
    return p


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = build_arg_parser().parse_args(argv)
    p = params_from_args(args)
    if args.nogui:
        result = build_extraction(p)
        if p.preview_png_path:
            save_preview_png(result, p, p.preview_png_path)
        summary = write_gcode(result.placed_tip_path, p)
        print(json.dumps({"summary": summary, "warnings": result.warnings}, indent=2))
        return 0
    root = tk.Tk()
    # Default layout is roughly 1/3 controls, 1/3 3D view, 1/3 XZ plot.
    root.geometry("1800x950")
    MeshCenterlineGUI(root, p)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
