#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code for a rounded π glyph in the XZ plane.

This version matches the visual style of the uploaded sign much more closely:
- rounded left hook on the top bar
- optional right flare after the horizontal top bar
- two hanging vertical legs under the bar
- left bottom curl and right bottom curl
- right bottom curl trimmed to a quarter-circle before a straight tail
- separate print passes so the glyph layout matches the sign
- angle-specific calibration branches for 0-90-0, 0-180-0, and 90-180-90
- travel moves retreat out of plane and above the work before repositioning,
  so travel does not cross already printed material

Write order
-----------
1) left leg, bottom -> top
2) top bar + left hook, left -> right
3) right leg, top -> bottom

Orientation convention
----------------------
- B = 0 deg   -> tool points straight up (+Z)
- B = 90 deg  -> tool is horizontal
- B = 180 deg -> tool points straight down (-Z)

The calibration / tip-offset planning logic follows the same structure as the
reference script you supplied earlier.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------- Defaults ----------------
DEFAULT_OUT = "gcode_generation/pi_sign_image_style_xz.gcode"

# Overall glyph box in tip space
DEFAULT_X_LEFT = 65.0
DEFAULT_X_RIGHT = 115.0
DEFAULT_Y = 52.0
DEFAULT_Z_BOTTOM = -140.0
DEFAULT_Z_TOP = -110.0

# Shape details
DEFAULT_VERTICAL_OVEREXTEND_MM = 1.0
DEFAULT_LEG_SPACING_EXTRA_MM = 2.0
DEFAULT_LEFT_LEG_OFFSET_X_MM = 0 #-2.0
DEFAULT_LEFT_LEG_OFFSET_Y_MM = 0.0
DEFAULT_LEFT_LEG_OFFSET_Z_MM = 0 #-1.0
DEFAULT_RIGHT_LEG_OFFSET_X_MM = 0.0
DEFAULT_RIGHT_LEG_OFFSET_Y_MM = 0 #3.0
DEFAULT_RIGHT_LEG_OFFSET_Z_MM = 0 #2.0
DEFAULT_POINTS_PER_MM = 10.0
DEFAULT_TANGENT_SMOOTH_WINDOW = 2
DEFAULT_SLOW_ZONE_RADIUS_MM = 0.5
DEFAULT_LEFT_LEG_EXTRUSION_START_FRACTION = 0.15
DEFAULT_RIGHT_LEG_EXTRUSION_STOP_FRACTION = 0.80
DEFAULT_TOP_BAR_SHORTEN_MM = 2.0

# Orientation
DEFAULT_WRITE_MODE = "calibrated"
DEFAULT_ORIENTATION_MODE = "tangent"
DEFAULT_FIT_STRATEGY = "phase-pchip"
DEFAULT_FIXED_B = 90.0
DEFAULT_FIXED_C = 180.0
DEFAULT_C_DEG = 180.0
DEFAULT_B_ANGLE_BIAS_DEG = 0.0
DEFAULT_BC_SOLVE_SAMPLES = 5001
DEFAULT_Y_OFFPLANE_SIGN = -1.0
DEFAULT_Y_OFFPLANE_FIT_MODEL = "avg_pchip"

# Motion
DEFAULT_TRAVEL_FEED = 1600.0
DEFAULT_APPROACH_FEED = 400.0
DEFAULT_FINE_APPROACH_FEED = 400.0
DEFAULT_NODE_CONNECTION_FEED = 400.0
DEFAULT_PRINT_FEED = 400.0
DEFAULT_TRAVEL_LIFT_Z = 8.0
DEFAULT_APPROACH_SIDE_MM = 4.0
DEFAULT_RETREAT_SIDE_MM = 4.0
DEFAULT_EDGE_SAMPLES = 1
DEFAULT_BRANCH_TRANSITION_MODE = "smooth"
DEFAULT_PHASE_TRANSITION_STEPS = 12

# Optional right-end flare appended after the horizontal top bar.
# Radius 0 means "infer from the vertical radius of the existing left
# top-bar quarter-circle".
DEFAULT_TOP_FLARE_DEGREES = 0.0
DEFAULT_TOP_FLARE_RADIUS_MM = 0.0

# Pressure actuation
DEFAULT_EMIT_EXTRUSION = True
DEFAULT_PREFLOW_DWELL_MS = 400

# Stage limits
DEFAULT_BBOX_X_MIN = -1e9
DEFAULT_BBOX_X_MAX = 1e9
DEFAULT_BBOX_Y_MIN = -1e9
DEFAULT_BBOX_Y_MAX = 1e9
DEFAULT_BBOX_Z_MIN = -1e9
DEFAULT_BBOX_Z_MAX = 1e9

DEFAULT_MIN_TANGENT_XY = 1e-9
DEFAULT_POINT_MERGE_TOL = 1e-9


# ---------------- Embedded normalized template ----------------
# Extracted from the uploaded sign image's inner π glyph, skeletonized and
# resampled. Coordinates are normalized to the glyph bbox:
#   x_norm in [0, 1], z_norm in [0, 1] with z increasing upward.
LEFT_LEG_TEMPLATE = np.asarray([[0.197436, 0.062315], [0.222808, 0.067409], [0.229381, 0.069169], [0.235876, 0.071157], [0.24868, 0.074972], [0.261393, 0.079835], [0.273822, 0.085318], [0.285871, 0.091631], [0.297397, 0.099088], [0.308431, 0.107619], [0.319062, 0.117034], [0.329064, 0.127319], [0.338415, 0.138316], [0.346976, 0.149837], [0.35508, 0.161758], [0.362189, 0.17405], [0.36863, 0.18659], [0.373962, 0.199543], [0.378628, 0.212744], [0.382412, 0.226273], [0.385708, 0.239984], [0.387992, 0.254072], [0.390189, 0.268191], [0.391661, 0.282581], [0.392846, 0.297078], [0.393854, 0.31164], [0.394832, 0.326213], [0.395565, 0.340878], [0.396247, 0.355562], [0.396613, 0.370363], [0.39698, 0.385165], [0.397346, 0.399966], [0.397436, 0.41487], [0.397436, 0.429808], [0.397436, 0.444745], [0.397436, 0.459683], [0.397436, 0.47462], [0.397436, 0.489558], [0.397436, 0.504495], [0.397436, 0.519433], [0.397436, 0.534371], [0.397436, 0.549308], [0.397436, 0.564246], [0.397436, 0.579183], [0.397436, 0.594121], [0.397436, 0.609059], [0.397436, 0.623996], [0.397436, 0.638934], [0.397436, 0.653871], [0.397436, 0.668809], [0.397436, 0.683746], [0.397436, 0.698684], [0.397436, 0.713622], [0.397436, 0.728559], [0.397436, 0.743497], [0.397436, 0.758434], [0.397436, 0.773372], [0.397436, 0.78831], [0.397436, 0.803247], [0.397436, 0.818185], [0.397436, 0.833122], [0.397436, 0.84806], [0.397436, 0.862998], [0.397436, 0.877935], [0.397436, 0.892873], [0.397436, 0.900341], [0.397436, 0.90781], [0.397436, 0.937685]], dtype=float)
TOP_BAR_TEMPLATE = np.asarray([[0.048718, 0.703264], [0.05389, 0.731168], [0.055165, 0.73815], [0.056782, 0.745006], [0.059881, 0.758766], [0.063453, 0.772351], [0.067482, 0.785766], [0.071968, 0.799011], [0.077234, 0.811965], [0.08325, 0.82464], [0.089952, 0.83706], [0.097492, 0.849157], [0.105834, 0.860507], [0.114872, 0.871367], [0.124703, 0.881473], [0.135065, 0.890921], [0.146084, 0.899434], [0.157551, 0.906968], [0.169452, 0.913554], [0.181753, 0.919266], [0.194471, 0.924067], [0.207479, 0.928235], [0.22097, 0.931344], [0.234625, 0.934097], [0.24857, 0.936217], [0.262812, 0.937685], [0.277144, 0.938957], [0.291671, 0.939805], [0.306391, 0.940229], [0.321111, 0.940653], [0.336026, 0.940653], [0.35094, 0.940653], [0.365854, 0.940653], [0.380769, 0.940653], [0.395683, 0.940653], [0.410597, 0.940653], [0.425511, 0.940653], [0.440426, 0.940653], [0.45534, 0.940653], [0.470254, 0.940653], [0.485169, 0.940653], [0.500083, 0.940653], [0.514997, 0.940653], [0.529911, 0.940653], [0.544826, 0.940653], [0.55974, 0.940653], [0.574654, 0.940653], [0.589569, 0.940653], [0.604483, 0.940653], [0.619397, 0.940653], [0.634311, 0.940653], [0.649226, 0.940653], [0.66414, 0.940653], [0.679054, 0.940653], [0.693969, 0.940653], [0.708883, 0.940653], [0.723797, 0.940653], [0.738711, 0.940653], [0.753626, 0.940653], [0.76854, 0.940653], [0.783454, 0.940653], [0.798369, 0.940653], [0.813283, 0.940653], [0.828197, 0.940653], [0.843111, 0.940653], [0.858026, 0.940653], [0.87294, 0.940653], [0.887854, 0.940653], [0.902575, 0.940229], [0.909999, 0.940158], [0.917411, 0.940059], [0.946154, 0.937685]], dtype=float)
RIGHT_LEG_TEMPLATE = np.asarray([[0.671795, 0.937685], [0.671795, 0.907781], [0.671795, 0.900305], [0.671795, 0.892828], [0.671795, 0.877876], [0.671795, 0.862924], [0.671795, 0.847972], [0.671795, 0.833019], [0.671795, 0.818067], [0.671795, 0.803115], [0.671795, 0.788162], [0.671795, 0.77321], [0.671795, 0.758258], [0.671795, 0.743305], [0.671795, 0.728353], [0.671795, 0.713401], [0.671795, 0.698448], [0.671795, 0.683496], [0.671795, 0.668544], [0.671795, 0.653591], [0.671795, 0.638639], [0.671795, 0.623687], [0.671795, 0.608734], [0.671795, 0.593782], [0.671795, 0.57883], [0.671795, 0.563877], [0.671795, 0.548925], [0.671795, 0.533973], [0.671795, 0.51902], [0.671795, 0.504068], [0.671795, 0.489116], [0.671795, 0.474163], [0.671795, 0.459211], [0.671795, 0.444259], [0.671795, 0.429306], [0.671795, 0.414354], [0.671795, 0.399402], [0.671795, 0.384449], [0.671795, 0.369497], [0.671795, 0.354545], [0.671795, 0.339592], [0.671795, 0.32464], [0.671795, 0.309688], [0.671795, 0.294735], [0.672161, 0.279919], [0.672527, 0.265103], [0.67326, 0.250424], [0.674354, 0.235879], [0.675459, 0.221337], [0.677128, 0.207007], [0.679363, 0.192886], [0.681927, 0.178888], [0.68559, 0.165299], [0.690174, 0.152053], [0.695793, 0.139192], [0.702797, 0.126847], [0.710887, 0.115365], [0.72009, 0.10481], [0.730473, 0.095562], [0.741661, 0.087534], [0.753593, 0.080932], [0.766087, 0.075559], [0.779321, 0.071802], [0.792846, 0.069531], [0.806635, 0.068684], [0.820491, 0.06883], [0.834198, 0.070317], [0.847524, 0.073095], [0.860416, 0.0776], [0.872694, 0.083445], [0.884427, 0.090481], [0.895338, 0.098813], [0.905148, 0.108544], [0.913953, 0.119299], [0.921473, 0.130992], [0.927837, 0.143346], [0.932901, 0.156413], [0.936857, 0.169893], [0.940259, 0.18358], [0.942839, 0.197572], [0.944671, 0.211842], [0.945706, 0.218933], [0.946642, 0.226061], [0.948718, 0.255193]], dtype=float)

# Normalized node locations where the top bar meets the two vertical legs.
LEFT_NODE_NORM = np.asarray([0.397436, 0.937685], dtype=float)
RIGHT_NODE_NORM = np.asarray([0.671795, 0.937685], dtype=float)


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
    c_180_deg: float

    fit_models: Optional[Dict[str, Any]] = None
    r_model: Optional[Dict[str, Any]] = None
    z_model: Optional[Dict[str, Any]] = None
    y_off_model: Optional[Dict[str, Any]] = None
    y_off_extrap_model: Optional[Dict[str, Any]] = None
    tip_angle_model: Optional[Dict[str, Any]] = None
    selected_fit_model: Optional[str] = None
    selected_offplane_fit_model: Optional[str] = None
    requested_offplane_fit_model: Optional[str] = None
    resolved_offplane_fit_model: Optional[str] = None
    active_phase: str = "pull"
    offplane_y_sign: float = 1.0
    phase_models: Optional[Dict[str, Dict[str, Any]]] = None
    angle_phase_models: Optional[Dict[str, Dict[str, Dict[str, Any]]]] = None
    phase_aliases: Optional[Dict[str, str]] = None
    default_motion_phase: str = "pull"
    derived_model_cache: Optional[Dict[Tuple[str, ...], Optional[Dict[str, Any]]]] = None


# ---------------- Math / geometry helpers ----------------
def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if n <= eps:
        return np.zeros_like(arr)
    return arr / n


def deduplicate_polyline_points(points: np.ndarray, tol: float = DEFAULT_POINT_MERGE_TOL) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return pts.copy()
    out = [pts[0].copy()]
    for p in pts[1:]:
        if float(np.linalg.norm(p - out[-1])) > float(tol):
            out.append(np.asarray(p, dtype=float).copy())
    return np.asarray(out, dtype=float)


def polyline_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def cumulative_polyline_lengths(points: np.ndarray) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return np.zeros(len(pts), dtype=float)
    ds = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(ds)])


def tangent_for_index(points: np.ndarray, i: int, smooth_window: int) -> np.ndarray:
    n = len(points)
    i0 = max(0, i - int(smooth_window))
    i1 = min(n - 1, i + int(smooth_window))
    if i1 == i0:
        if i == 0 and n > 1:
            return normalize(points[1] - points[0])
        if i == n - 1 and n > 1:
            return normalize(points[-1] - points[-2])
        return np.array([1.0, 0.0, 0.0], dtype=float)
    return normalize(points[i1] - points[i0])


def build_tangents_for_points(points: np.ndarray, smooth_window: int) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    tangents = np.zeros_like(pts)
    for i in range(len(pts)):
        tangents[i] = tangent_for_index(pts, i, smooth_window=max(1, int(smooth_window)))
    return tangents


def resample_polyline(points: np.ndarray, points_per_mm: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 2:
        return pts.copy()
    s = cumulative_polyline_lengths(pts)
    total = float(s[-1])
    if total <= 1e-12:
        return pts[[0, -1]].copy()
    n = max(2, int(math.ceil(total * float(points_per_mm))) + 1)
    ss = np.linspace(0.0, total, n)
    out = np.empty((n, pts.shape[1]), dtype=float)
    for j in range(pts.shape[1]):
        out[:, j] = np.interp(ss, s, pts[:, j])
    return out


def smooth_polyline(points: np.ndarray, window: int = 5) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 2 or int(window) <= 1:
        return pts.copy()
    w = int(window)
    half = w // 2
    out = pts.copy()
    for i in range(len(pts)):
        i0 = max(0, i - half)
        i1 = min(len(pts), i + half + 1)
        out[i] = pts[i0:i1].mean(axis=0)
    out[0] = pts[0]
    out[-1] = pts[-1]
    return out


def smooth_scalar_profile(values: np.ndarray, window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size <= 2 or int(window) <= 1:
        return arr.copy()

    w = max(3, int(window))
    if w % 2 == 0:
        w += 1
    pad = w // 2
    kernel = np.ones(w, dtype=float) / float(w)
    padded = np.pad(arr, (pad, pad), mode="edge")
    smoothed = np.convolve(padded, kernel, mode="valid")
    smoothed[0] = arr[0]
    smoothed[-1] = arr[-1]
    return smoothed


def smooth_b_angle_profile(values: np.ndarray, window: int = 5) -> np.ndarray:
    arr = np.asarray(values, dtype=float).reshape(-1)
    if arr.size <= 2:
        return arr.copy()

    out = smooth_scalar_profile(arr, window=window)
    delta = float(arr[-1] - arr[0])
    if delta > 1e-9:
        out = np.maximum.accumulate(out)
        out = np.clip(out, float(arr[0]), float(arr[-1]))
    elif delta < -1e-9:
        out = -np.maximum.accumulate(-out)
        out = np.clip(out, float(arr[-1]), float(arr[0]))

    out[0] = arr[0]
    out[-1] = arr[-1]
    return out


def desired_physical_b_angle_from_tangent(tangent: np.ndarray) -> float:
    """
    B-angle convention:
      0 deg   -> +Z (straight up)
      90 deg  -> horizontal
      180 deg -> -Z (straight down)
    """
    t = normalize(np.asarray(tangent, dtype=float))
    tz = float(np.clip(t[2], -1.0, 1.0))
    return float(math.degrees(math.acos(tz)))


def side_vector_from_tangent(
    tangent: np.ndarray,
    fallback: Optional[np.ndarray] = None,
    min_xy: float = DEFAULT_MIN_TANGENT_XY,
) -> np.ndarray:
    xy = np.asarray(tangent[:2], dtype=float)
    nxy = float(np.linalg.norm(xy))
    if nxy < float(min_xy):
        if fallback is not None and float(np.linalg.norm(np.asarray(fallback[:2], dtype=float))) >= float(min_xy):
            xy = np.asarray(fallback[:2], dtype=float)
            nxy = float(np.linalg.norm(xy))
        else:
            return np.array([0.0, 1.0, 0.0], dtype=float)
    tx, ty = xy / nxy
    return np.array([-ty, tx, 0.0], dtype=float)


def prepend_extension(points: np.ndarray, length_mm: float) -> Tuple[np.ndarray, float]:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2 or float(length_mm) <= 0.0:
        return pts.copy(), 0.0
    direction = normalize(pts[0] - pts[1])
    p_ext = pts[0] + float(length_mm) * direction
    out = np.vstack([p_ext, pts])
    return out, float(length_mm)


def append_extension(points: np.ndarray, length_mm: float) -> Tuple[np.ndarray, float]:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2 or float(length_mm) <= 0.0:
        return pts.copy(), 0.0
    direction = normalize(pts[-1] - pts[-2])
    p_ext = pts[-1] + float(length_mm) * direction
    out = np.vstack([pts, p_ext])
    return out, float(polyline_length(pts))


def scale_template_to_world(
    template_xz: np.ndarray,
    x_left: float,
    x_right: float,
    z_bottom: float,
    z_top: float,
    y: float,
) -> np.ndarray:
    txz = np.asarray(template_xz, dtype=float)
    width = float(x_right) - float(x_left)
    height = float(z_top) - float(z_bottom)
    x = float(x_left) + width * txz[:, 0]
    z = float(z_bottom) + height * txz[:, 1]
    yy = np.full_like(x, float(y))
    return np.column_stack([x, yy, z])


def infer_left_hook_radius_mm(top_bar_points: np.ndarray) -> float:
    """Infer the radius of the left top-bar quarter-circle in world mm.

    The template's left hook rises from its first point into the horizontal
    top bar.  Using the vertical rise makes the right flare use the same
    physical radius in the XZ plane.
    """
    pts = np.asarray(top_bar_points, dtype=float)
    if len(pts) < 2:
        return 0.0
    return max(0.0, float(np.max(pts[:, 2]) - pts[0, 2]))


def build_top_bar_flare(
    start_point: np.ndarray,
    radius_mm: float,
    flare_degrees: float,
    points_per_mm: float,
) -> np.ndarray:
    """Append a fourth-quadrant circular flare whose tangent B goes 90 -> 45 deg.

    In XZ tip-space, a tangent B of 90 deg is +X horizontal and B of 45 deg
    points +X/+Z.  The corresponding radius vector sweeps from -90 to -45 deg,
    i.e. the fourth trig quadrant.
    """
    start = np.asarray(start_point, dtype=float).reshape(3)
    r = float(radius_mm)
    deg = float(flare_degrees)
    if r <= 0.0 or deg <= 0.0:
        return start.reshape(1, 3).copy()

    arc_len = r * math.radians(abs(deg))
    n = max(2, int(math.ceil(arc_len * float(points_per_mm))) + 1)
    theta = np.radians(np.linspace(-90.0, -90.0 + deg, n))
    center = start + np.array([0.0, 0.0, r], dtype=float)
    x = center[0] + r * np.cos(theta)
    y = np.full_like(x, start[1])
    z = center[2] + r * np.sin(theta)
    flare = np.column_stack([x, y, z])
    flare[0] = start
    return flare


def widen_leg_spacing(points: np.ndarray, left_node_x: float, right_node_x: float, extra_spacing_mm: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float).copy()
    extra = float(extra_spacing_mm)
    if len(pts) == 0 or abs(extra) <= 1e-12:
        return pts

    x = pts[:, 0].copy()
    left_shift = -0.5 * extra
    right_shift = 0.5 * extra
    span = float(right_node_x) - float(left_node_x)

    left_mask = x <= float(left_node_x)
    right_mask = x >= float(right_node_x)
    mid_mask = ~(left_mask | right_mask)

    pts[left_mask, 0] += left_shift
    pts[right_mask, 0] += right_shift
    if np.any(mid_mask) and span > 1e-12:
        u = (x[mid_mask] - float(left_node_x)) / span
        pts[mid_mask, 0] = (float(left_node_x) + left_shift) + u * (span + extra)

    return pts


def trim_right_leg_to_quarter_arc(points: np.ndarray, points_per_mm: float, stop_b_deg: float = 90.0) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return pts.copy()

    tangents = build_tangents_for_points(pts, smooth_window=max(2, DEFAULT_TANGENT_SMOOTH_WINDOW))
    b_targets = np.asarray([desired_physical_b_angle_from_tangent(t) for t in tangents], dtype=float)
    hits = np.where(b_targets <= float(stop_b_deg))[0]
    if hits.size == 0:
        return pts.copy()

    idx = int(max(1, hits[0]))
    quarter_end = pts[idx].copy()
    tail_end_x = float(pts[-1, 0])
    if tail_end_x <= float(quarter_end[0]) + 1e-12:
        return pts[: idx + 1].copy()

    tail_len = tail_end_x - float(quarter_end[0])
    n_tail = max(2, int(math.ceil(tail_len * float(points_per_mm))) + 1)
    x = np.linspace(float(quarter_end[0]), tail_end_x, n_tail)
    y = np.full_like(x, float(quarter_end[1]))
    z = np.full_like(x, float(quarter_end[2]))
    tail = np.column_stack([x, y, z])
    return deduplicate_polyline_points(np.vstack([pts[: idx + 1], tail[1:]]))


def arclength_at_first_b_below(
    points: np.ndarray,
    threshold_deg: float,
    smooth_window: int = DEFAULT_TANGENT_SMOOTH_WINDOW,
) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    tangents = build_tangents_for_points(pts, smooth_window=smooth_window)
    b_targets = np.asarray([desired_physical_b_angle_from_tangent(t) for t in tangents], dtype=float)
    hits = np.where(b_targets <= float(threshold_deg))[0]
    if hits.size == 0:
        return 0.0
    s = cumulative_polyline_lengths(pts)
    return float(s[int(hits[0])])


def offset_points(points: np.ndarray, dx: float = 0.0, dy: float = 0.0, dz: float = 0.0) -> np.ndarray:
    pts = np.asarray(points, dtype=float).copy()
    if len(pts) == 0:
        return pts
    pts[:, 0] += float(dx)
    pts[:, 1] += float(dy)
    pts[:, 2] += float(dz)
    return pts


def trim_polyline_end_by_length(points: np.ndarray, trim_length_mm: float) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    trim = float(trim_length_mm)
    if len(pts) <= 1 or trim <= 1e-12:
        return pts.copy()

    s = cumulative_polyline_lengths(pts)
    total = float(s[-1])
    if trim >= total:
        return pts[:2].copy()

    keep_until = total - trim
    idx = int(np.searchsorted(s, keep_until, side="right") - 1)
    idx = int(np.clip(idx, 0, len(pts) - 2))
    seg_len = float(s[idx + 1] - s[idx])
    if seg_len <= 1e-12:
        cut_point = pts[idx].copy()
    else:
        u = (keep_until - float(s[idx])) / seg_len
        cut_point = pts[idx] + u * (pts[idx + 1] - pts[idx])
    out = np.vstack([pts[: idx + 1], cut_point])
    return deduplicate_polyline_points(out)


def split_polyline_by_first_b_below(
    points: np.ndarray,
    threshold_deg: float,
    smooth_window: int = DEFAULT_TANGENT_SMOOTH_WINDOW,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split at the first point where tangent-derived B drops below threshold."""
    pts = np.asarray(points, dtype=float)
    if len(pts) < 3:
        return pts.copy(), pts[-1:].copy()
    tangents = build_tangents_for_points(pts, smooth_window=smooth_window)
    b_targets = np.asarray([desired_physical_b_angle_from_tangent(t) for t in tangents], dtype=float)
    hits = np.where(b_targets < float(threshold_deg))[0]
    if hits.size == 0:
        return pts.copy(), pts[-1:].copy()
    idx = int(max(1, hits[0] - 1))
    return pts[: idx + 1].copy(), pts[idx:].copy()


def nearest_arclength_to_world_point(points: np.ndarray, target_point: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    target = np.asarray(target_point, dtype=float)
    idx = int(np.argmin(np.linalg.norm(pts - target, axis=1)))
    return float(cumulative_polyline_lengths(pts)[idx])


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


def _selector_variants(selector: Optional[str]) -> List[str]:
    selected = None if selector is None else str(selector).strip().lower().replace("-", "_")
    if not selected:
        return []
    out = [selected]
    if selected == "pchip":
        out.append("avg_pchip")
    elif selected == "cubic":
        out.append("avg_cubic")
    elif selected == "linear":
        out.append("avg_linear")
    return list(dict.fromkeys(out))


def _select_named_model(models: Dict[str, Any], base_name: str, selected_fit_model: Optional[str]) -> Optional[Dict[str, Any]]:
    candidates: List[str] = []
    for selected in _selector_variants(selected_fit_model):
        candidates.append(f"{base_name}_{selected}")
    candidates.append(base_name)
    for key in candidates:
        spec = _normalize_model_spec(models.get(key))
        if spec is not None:
            return spec
    return None


def _fit_linear_model_from_pchip(model_spec: Optional[Dict[str, Any]], value_name: str) -> Optional[Dict[str, Any]]:
    spec = _normalize_model_spec(model_spec)
    if spec is None or str(spec.get("model_type") or "").strip().lower() != "pchip":
        return None
    x_knots = np.asarray(spec.get("x_knots", []), dtype=float).reshape(-1)
    y_knots = np.asarray(spec.get("y_knots", []), dtype=float).reshape(-1)
    if x_knots.size != y_knots.size or x_knots.size < 2:
        return None
    order = np.argsort(x_knots)
    x_knots = x_knots[order]
    y_knots = y_knots[order]
    keep = np.ones_like(x_knots, dtype=bool)
    keep[1:] = np.diff(x_knots) > 0.0
    x_use = x_knots[keep]
    y_use = y_knots[keep]
    if x_use.size < 2:
        return None
    coeffs = np.polyfit(x_use, y_use, 1)
    return {
        "model_type": "polynomial",
        "coefficients": np.asarray(coeffs, dtype=float).tolist(),
        "source_model_type": "pchip",
        "source_selector": "avg_pchip",
        "value_name": value_name,
        "equation": f"{value_name}(b) = {coeffs[0]:.9g}*b + {coeffs[1]:.9g}",
        "fit_x_range": [float(x_use[0]), float(x_use[-1])],
    }


def _normalize_motion_phase_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _normalize_calibration_set_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    text = text.replace("º", "").replace("°", "")
    text = text.replace("_", "-").replace(" ", "")
    return text or None


def _split_motion_phase_reference(value: Any) -> Tuple[Optional[str], Optional[str]]:
    """Return (calibration_set, phase), accepting refs like '0-90-0:release'."""
    text = _normalize_motion_phase_name(value)
    if text is None:
        return None, None
    for sep in ("::", ":", "/", "|"):
        if sep in text:
            set_part, phase_part = text.split(sep, 1)
            return _normalize_calibration_set_name(set_part), _normalize_motion_phase_name(phase_part)
    return None, text


def _extract_angle_phase_models(data: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    payload = data.get("curl_angle_specific_fit_models") or {}
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not isinstance(payload, dict):
        return out
    for raw_set_name, set_payload in payload.items():
        set_name = _normalize_calibration_set_name(raw_set_name)
        if set_name is None or not isinstance(set_payload, dict):
            continue
        phase_payload = set_payload.get("fit_models_by_phase") or {}
        if not isinstance(phase_payload, dict):
            continue
        out[set_name] = {}
        for raw_phase_name, models in phase_payload.items():
            phase_name = _normalize_motion_phase_name(raw_phase_name)
            if phase_name is not None and isinstance(models, dict):
                out[set_name][phase_name] = dict(models)
    return out


def _select_from_phase_model_dict(
    models_by_phase: Dict[str, Dict[str, Any]],
    phase_name: Optional[str],
    base_name: str,
    selector: Optional[str],
    aliases: Optional[Dict[str, str]] = None,
) -> Optional[Dict[str, Any]]:
    want = _normalize_motion_phase_name(phase_name)
    if want is None:
        return None
    if aliases and want in aliases:
        want = aliases[want]
    candidates: List[str] = []
    if want in models_by_phase:
        candidates.append(want)
    exact_prefix_matches = [k for k in models_by_phase if k.startswith(want)]
    if len(exact_prefix_matches) == 1:
        candidates.append(exact_prefix_matches[0])
    contains_matches = [k for k in models_by_phase if want in k]
    if len(contains_matches) == 1:
        candidates.append(contains_matches[0])
    for phase_key in list(dict.fromkeys(candidates)):
        model = _select_named_model(models_by_phase[phase_key], base_name, selector)
        if model is not None:
            return model
    return None


def _normalize_phase_aliases(data: Dict[str, Any]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for raw_key in ("phase_aliases", "motion_phase_map"):
        payload = data.get(raw_key) or {}
        if not isinstance(payload, dict):
            continue
        for src, dst in payload.items():
            src_key = _normalize_motion_phase_name(src)
            dst_key = _normalize_motion_phase_name(dst)
            if src_key is not None and dst_key is not None:
                aliases[src_key] = dst_key
    return aliases


def _extract_phase_models(data: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], str]:
    phase_payload = data.get("fit_models_by_phase") or {}
    phase_models: Dict[str, Dict[str, Any]] = {}
    if isinstance(phase_payload, dict):
        for raw_phase_name, models in phase_payload.items():
            phase_name = _normalize_motion_phase_name(raw_phase_name)
            if phase_name is None or not isinstance(models, dict):
                continue
            phase_models[phase_name] = dict(models)

    default_phase = _normalize_motion_phase_name(data.get("default_phase_for_legacy_access"))
    if default_phase is None or default_phase not in phase_models:
        if "pull" in phase_models:
            default_phase = "pull"
        elif phase_models:
            default_phase = next(iter(phase_models))
        else:
            default_phase = "pull"
    return phase_models, default_phase


def resolve_phase_name(cal: Calibration, phase_name: Optional[str]) -> str:
    phase_models = cal.phase_models or {}
    default_phase = str(cal.default_motion_phase or cal.active_phase or "pull")
    want = _normalize_motion_phase_name(phase_name)
    if want is None or not phase_models:
        return default_phase

    aliases = cal.phase_aliases or {}
    if want in aliases:
        want = aliases[want]
    if want in phase_models:
        return want

    exact_prefix_matches = [k for k in phase_models if k.startswith(want)]
    if len(exact_prefix_matches) == 1:
        return exact_prefix_matches[0]

    contains_matches = [k for k in phase_models if want in k]
    if len(contains_matches) == 1:
        return contains_matches[0]

    return default_phase


def _select_fit_model(
    cal: Calibration,
    base_name: str,
    motion_phase: Optional[str] = None,
    fit_model_selector: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    selector = fit_model_selector if fit_model_selector is not None else cal.selected_fit_model
    calibration_set, phase_ref = _split_motion_phase_reference(motion_phase)
    aliases = cal.phase_aliases or {}

    # First preference: angle-specific calibration sets, e.g. 0-90-0:release.
    if calibration_set is not None and cal.angle_phase_models:
        models_by_phase = cal.angle_phase_models.get(calibration_set)
        if models_by_phase:
            model = _select_from_phase_model_dict(
                models_by_phase,
                phase_ref,
                base_name,
                selector,
                aliases=aliases,
            )
            if model is not None:
                return model

    # Fallback: legacy/generic phase models.
    phase_models = cal.phase_models or {}
    phase_name = resolve_phase_name(cal, phase_ref if calibration_set is not None else motion_phase) if phase_models else None
    if phase_name is not None and phase_name in phase_models:
        model = _select_named_model(phase_models[phase_name], base_name, selector)
        if model is not None:
            return model

    model = _select_named_model(cal.fit_models or {}, base_name, selector)
    if model is not None:
        return model

    selector_key = None if selector is None else str(selector).strip().lower().replace("-", "_")
    if selector_key == "avg_linear":
        if cal.derived_model_cache is None:
            cal.derived_model_cache = {}
        cache_key = (calibration_set or "", phase_name or phase_ref or "", str(base_name), selector_key)
        if cache_key in cal.derived_model_cache:
            return cal.derived_model_cache[cache_key]  # type: ignore[index]
        source_model = None
        if calibration_set is not None and cal.angle_phase_models:
            models_by_phase = cal.angle_phase_models.get(calibration_set)
            if models_by_phase:
                source_model = _select_from_phase_model_dict(
                    models_by_phase,
                    phase_ref,
                    base_name,
                    "avg_pchip",
                    aliases=aliases,
                )
        if source_model is None and phase_name is not None and phase_name in phase_models:
            source_model = _select_named_model(phase_models[phase_name], base_name, "avg_pchip")
        if source_model is None:
            source_model = _select_named_model(cal.fit_models or {}, base_name, "avg_pchip")
        derived = _fit_linear_model_from_pchip(source_model, value_name=f"{base_name}_avg_linear")
        cal.derived_model_cache[cache_key] = derived  # type: ignore[index]
        if derived is not None:
            return derived
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


def load_calibration(json_path: str, requested_offplane_fit_model: Optional[str] = DEFAULT_Y_OFFPLANE_FIT_MODEL) -> Calibration:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cubic = data.get("cubic_coefficients", {}) or {}
    fit_models = data.get("fit_models", {}) or {}

    def _coeffs_from_model(models: Dict[str, Any], *names: str) -> Optional[np.ndarray]:
        for name in names:
            spec = _normalize_model_spec(models.get(name))
            if spec is None:
                continue
            coeffs = spec.get("coefficients", spec.get("coeffs"))
            if coeffs is not None:
                arr = np.asarray(coeffs, dtype=float).reshape(-1)
                if arr.size > 0:
                    return arr
        return None

    pr_arr = cubic.get("r_coeffs")
    if pr_arr is None:
        pr = _coeffs_from_model(fit_models, "r_cubic", "r_avg_cubic")
        if pr is None:
            pr = np.zeros(1, dtype=float)
    else:
        pr = np.asarray(pr_arr, dtype=float)

    pz_arr = cubic.get("z_coeffs")
    if pz_arr is None:
        pz = _coeffs_from_model(fit_models, "z_cubic", "z_avg_cubic")
        if pz is None:
            pz = np.zeros(1, dtype=float)
    else:
        pz = np.asarray(pz_arr, dtype=float)

    py_off_raw = cubic.get("offplane_y_coeffs", None)
    if py_off_raw is None:
        py_off = _coeffs_from_model(
            fit_models,
            "offplane_y_cubic",
            "offplane_y_avg_cubic",
            "offplane_y",
            "offplane_y_linear",
            "offplane_y_avg_linear",
        )
    else:
        py_off = np.asarray(py_off_raw, dtype=float)

    pa_raw = cubic.get("tip_angle_coeffs", None)
    if pa_raw is None:
        pa = _coeffs_from_model(fit_models, "tip_angle_cubic", "tip_angle_avg_cubic")
    else:
        pa = np.asarray(pa_raw, dtype=float)

    selected_fit_model = data.get("selected_fit_model")
    selected_fit_model = None if selected_fit_model is None else str(selected_fit_model).strip().lower()
    selected_offplane_fit_model = data.get("selected_offplane_fit_model")
    selected_offplane_fit_model = None if selected_offplane_fit_model is None else str(selected_offplane_fit_model).strip().lower()
    phase_models, default_phase = _extract_phase_models(data)
    angle_phase_models = _extract_angle_phase_models(data)
    phase_aliases = _normalize_phase_aliases(data)
    active_phase = resolve_phase_name(
        Calibration(
            pr=pr,
            pz=pz,
            py_off=py_off,
            pa=pa,
            b_min=0.0,
            b_max=0.0,
            x_axis="X",
            y_axis="Y",
            z_axis="Z",
            b_axis="B",
            c_axis="C",
            c_180_deg=180.0,
            fit_models=dict(fit_models),
            phase_models=phase_models,
            angle_phase_models=angle_phase_models,
            phase_aliases=phase_aliases,
            default_motion_phase=default_phase,
        ),
        data.get("default_phase_for_legacy_access"),
    )

    active_phase_models = phase_models.get(active_phase, fit_models) if phase_models else fit_models

    r_model = _select_named_model(active_phase_models, "r", selected_fit_model)
    z_model = _select_named_model(active_phase_models, "z", selected_fit_model)
    requested_offplane_fit_model = (
        None if requested_offplane_fit_model is None else str(requested_offplane_fit_model).strip().lower()
    )
    y_off_selector = requested_offplane_fit_model or selected_offplane_fit_model or selected_fit_model
    y_off_model = _select_named_model(active_phase_models, "offplane_y", y_off_selector)
    resolved_offplane_fit_model = None
    if y_off_model is not None:
        resolved_offplane_fit_model = y_off_selector
    elif selected_offplane_fit_model:
        resolved_offplane_fit_model = selected_offplane_fit_model
    elif selected_fit_model:
        resolved_offplane_fit_model = selected_fit_model
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

    return Calibration(
        pr=pr,
        pz=pz,
        py_off=py_off,
        pa=pa,
        fit_models=dict(fit_models),
        r_model=r_model,
        z_model=z_model,
        y_off_model=y_off_model,
        y_off_extrap_model=y_off_extrap_model,
        tip_angle_model=tip_angle_model,
        selected_fit_model=selected_fit_model,
        selected_offplane_fit_model=selected_offplane_fit_model,
        requested_offplane_fit_model=requested_offplane_fit_model,
        resolved_offplane_fit_model=resolved_offplane_fit_model,
        active_phase=active_phase,
        b_min=b_min,
        b_max=b_max,
        x_axis=str(duet_map.get("horizontal_axis") or motor_setup.get("horizontal_axis") or "X"),
        y_axis=str(duet_map.get("depth_axis") or motor_setup.get("depth_axis") or "Y"),
        z_axis=str(duet_map.get("vertical_axis") or motor_setup.get("vertical_axis") or "Z"),
        b_axis=str(duet_map.get("pull_axis") or motor_setup.get("b_motor_axis") or "B"),
        c_axis=str(duet_map.get("rotation_axis") or motor_setup.get("rotation_axis") or "C"),
        c_180_deg=float(motor_setup.get("rotation_axis_180_deg", 180.0)),
        phase_models=phase_models,
        phase_aliases=phase_aliases,
        default_motion_phase=default_phase,
        derived_model_cache={},
    )


def eval_r(cal: Calibration, b: Any, motion_phase: Optional[str] = None, fit_model_selector: Optional[str] = None) -> np.ndarray:
    model = _select_fit_model(cal, "r", motion_phase=motion_phase, fit_model_selector=fit_model_selector)
    if model is not None:
        return eval_model_spec(model, b)
    if cal.r_model is not None:
        return eval_model_spec(cal.r_model, b)
    return poly_eval(cal.pr, b)


def eval_z(cal: Calibration, b: Any, motion_phase: Optional[str] = None, fit_model_selector: Optional[str] = None) -> np.ndarray:
    model = _select_fit_model(cal, "z", motion_phase=motion_phase, fit_model_selector=fit_model_selector)
    if model is not None:
        return eval_model_spec(model, b)
    if cal.z_model is not None:
        return eval_model_spec(cal.z_model, b)
    return poly_eval(cal.pz, b)


def eval_offplane_y(cal: Calibration, b: Any, motion_phase: Optional[str] = None, fit_model_selector: Optional[str] = None) -> np.ndarray:
    selector = cal.requested_offplane_fit_model or cal.selected_offplane_fit_model or cal.selected_fit_model
    model = _select_fit_model(cal, "offplane_y", motion_phase=motion_phase, fit_model_selector=selector)
    extrap_model = _select_fit_model(cal, "offplane_y_linear", motion_phase=motion_phase, fit_model_selector=None)
    if model is not None:
        if str(model.get("model_type", "")).lower() == "pchip":
            values = eval_pchip_with_linear_extrap(model, extrap_model or cal.y_off_extrap_model, b)
        else:
            values = eval_model_spec(model, b, default_if_none=0.0)
    elif cal.y_off_model is not None:
        if str(cal.y_off_model.get("model_type", "")).lower() == "pchip":
            values = eval_pchip_with_linear_extrap(cal.y_off_model, cal.y_off_extrap_model, b)
        else:
            values = eval_model_spec(cal.y_off_model, b, default_if_none=0.0)
    else:
        values = poly_eval(cal.py_off, b, default_if_none=0.0)
    return float(cal.offplane_y_sign) * np.asarray(values, dtype=float)


def eval_tip_angle_deg(cal: Calibration, b: Any, motion_phase: Optional[str] = None, fit_model_selector: Optional[str] = None) -> np.ndarray:
    model = _select_fit_model(cal, "tip_angle", motion_phase=motion_phase, fit_model_selector=fit_model_selector)
    if model is not None:
        return eval_model_spec(model, b)
    if cal.tip_angle_model is not None:
        return eval_model_spec(cal.tip_angle_model, b)
    if cal.pa is None:
        raise ValueError("Calibration is missing tip_angle_coeffs.")
    return poly_eval(cal.pa, b)


def predict_tip_offset_xyz(
    cal: Calibration,
    b: float,
    c_deg: float,
    motion_phase: Optional[str] = None,
    fit_model_selector: Optional[str] = None,
) -> np.ndarray:
    r = float(eval_r(cal, b, motion_phase=motion_phase, fit_model_selector=fit_model_selector))
    z = float(eval_z(cal, b, motion_phase=motion_phase, fit_model_selector=fit_model_selector))
    y_off = float(eval_offplane_y(cal, b, motion_phase=motion_phase, fit_model_selector=fit_model_selector))
    c = math.radians(float(c_deg))
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def stage_xyz_for_tip(
    cal: Calibration,
    tip_xyz: np.ndarray,
    b: float,
    c_deg: float,
    motion_phase: Optional[str] = None,
    fit_model_selector: Optional[str] = None,
) -> np.ndarray:
    return np.asarray(tip_xyz, dtype=float) - predict_tip_offset_xyz(
        cal,
        b,
        c_deg,
        motion_phase=motion_phase,
        fit_model_selector=fit_model_selector,
    )


def solve_b_for_target_tip_angle(
    cal: Calibration,
    target_angle_deg: float,
    search_samples: int = DEFAULT_BC_SOLVE_SAMPLES,
    motion_phase: Optional[str] = None,
    fit_model_selector: Optional[str] = None,
) -> float:
    b_lo, b_hi = float(cal.b_min), float(cal.b_max)
    bb = np.linspace(b_lo, b_hi, int(max(101, search_samples)))
    aa = eval_tip_angle_deg(cal, bb, motion_phase=motion_phase, fit_model_selector=fit_model_selector) - float(target_angle_deg)
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
            return float(
                eval_tip_angle_deg(cal, x, motion_phase=motion_phase, fit_model_selector=fit_model_selector)
                - float(target_angle_deg)
            )

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


@dataclass(frozen=True)
class MotionProfile:
    motion_phase: Optional[str]
    fit_model_selector: Optional[str]


def profile_for_phase(fit_strategy: str, phase: str, calibration_set: Optional[str] = None) -> MotionProfile:
    strategy = str(fit_strategy).strip().lower().replace("_", "-")
    if strategy == "fit":
        phase_ref = str(phase)
        return MotionProfile(motion_phase=f"0-180-0:{phase_ref}", fit_model_selector="pchip")
    if strategy == "phase-pchip":
        phase_ref = str(phase)
        set_ref = _normalize_calibration_set_name(calibration_set)
        if set_ref:
            phase_ref = f"{set_ref}:{phase_ref}"
        return MotionProfile(motion_phase=phase_ref, fit_model_selector="pchip")
    if strategy == "avg-pchip":
        return MotionProfile(motion_phase=None, fit_model_selector="avg_pchip")
    if strategy == "avg-cubic":
        return MotionProfile(motion_phase=None, fit_model_selector="avg_cubic")
    if strategy == "avg-linear":
        return MotionProfile(motion_phase=None, fit_model_selector="avg_linear")
    raise ValueError("--fit-strategy must be fit, phase-pchip, avg-pchip, avg-cubic, or avg-linear")


def resolve_offplane_fit_model(fit_strategy: str, offplane_fit_model: Optional[str]) -> str:
    if offplane_fit_model is not None:
        return str(offplane_fit_model).strip().lower().replace("-", "_")
    strategy = str(fit_strategy).strip().lower().replace("_", "-")
    if strategy == "avg-linear":
        return "avg_linear"
    return DEFAULT_Y_OFFPLANE_FIT_MODEL


def branch_points_from_live_tip(points: np.ndarray, live_tip_xyz: Optional[np.ndarray], keep_shape: bool) -> np.ndarray:
    pts = np.asarray(points, dtype=float).copy()
    if live_tip_xyz is None or len(pts) == 0:
        return pts
    live_tip = np.asarray(live_tip_xyz, dtype=float)
    if keep_shape:
        pts += live_tip - pts[0]
    else:
        pts[0] = live_tip
    return pts


# ---------------- Glyph construction ----------------
def build_pi_sign_branches(
    x_left: float,
    x_right: float,
    y: float,
    z_bottom: float,
    z_top: float,
    vertical_overextend_mm: float,
    leg_spacing_extra_mm: float,
    left_leg_offset_x_mm: float,
    left_leg_offset_y_mm: float,
    left_leg_offset_z_mm: float,
    right_leg_offset_x_mm: float,
    right_leg_offset_y_mm: float,
    right_leg_offset_z_mm: float,
    left_leg_extrusion_start_fraction: float,
    right_leg_extrusion_stop_fraction: float,
    top_bar_shorten_mm: float,
    points_per_mm: float,
    fit_strategy: str,
    top_flare_radius_mm: float = DEFAULT_TOP_FLARE_RADIUS_MM,
    top_flare_degrees: float = DEFAULT_TOP_FLARE_DEGREES,
) -> List[Dict[str, Any]]:
    x_left = float(x_left)
    x_right = float(x_right)
    y = float(y)
    z_bottom = float(z_bottom)
    z_top = float(z_top)
    vertical_overextend_mm = float(vertical_overextend_mm)
    leg_spacing_extra_mm = float(leg_spacing_extra_mm)
    left_leg_offset_x_mm = float(left_leg_offset_x_mm)
    left_leg_offset_y_mm = float(left_leg_offset_y_mm)
    left_leg_offset_z_mm = float(left_leg_offset_z_mm)
    right_leg_offset_x_mm = float(right_leg_offset_x_mm)
    right_leg_offset_y_mm = float(right_leg_offset_y_mm)
    right_leg_offset_z_mm = float(right_leg_offset_z_mm)
    left_leg_extrusion_start_fraction = float(left_leg_extrusion_start_fraction)
    right_leg_extrusion_stop_fraction = float(right_leg_extrusion_stop_fraction)
    top_bar_shorten_mm = float(top_bar_shorten_mm)
    points_per_mm = float(points_per_mm)
    top_flare_radius_mm = float(top_flare_radius_mm)
    top_flare_degrees = float(top_flare_degrees)

    if x_right <= x_left:
        raise ValueError("--x-right must be greater than --x-left")
    if z_top <= z_bottom:
        raise ValueError("--z-top must be greater than --z-bottom")
    if vertical_overextend_mm < 0.0:
        raise ValueError("--vertical-overextend-mm must be >= 0")
    if points_per_mm <= 0.0:
        raise ValueError("--points-per-mm must be > 0")
    if not (0.0 <= left_leg_extrusion_start_fraction <= 1.0):
        raise ValueError("--left-leg-extrusion-start-fraction must be in [0, 1]")
    if not (0.0 <= right_leg_extrusion_stop_fraction <= 1.0):
        raise ValueError("--right-leg-extrusion-stop-fraction must be in [0, 1]")
    if top_bar_shorten_mm < 0.0:
        raise ValueError("--top-bar-shorten-mm must be >= 0")
    if top_flare_radius_mm < 0.0:
        raise ValueError("--top-flare-radius-mm must be >= 0")
    if top_flare_degrees < 0.0:
        raise ValueError("--top-flare-degrees must be >= 0")

    left_leg = scale_template_to_world(LEFT_LEG_TEMPLATE, x_left, x_right, z_bottom, z_top, y)
    top_bar_raw = scale_template_to_world(TOP_BAR_TEMPLATE, x_left, x_right, z_bottom, z_top, y)
    right_leg = scale_template_to_world(RIGHT_LEG_TEMPLATE, x_left, x_right, z_bottom, z_top, y)

    left_node_x = x_left + (x_right - x_left) * float(LEFT_NODE_NORM[0])
    right_node_x = x_left + (x_right - x_left) * float(RIGHT_NODE_NORM[0])
    left_leg = widen_leg_spacing(left_leg, left_node_x, right_node_x, leg_spacing_extra_mm)
    top_bar_raw = widen_leg_spacing(top_bar_raw, left_node_x, right_node_x, leg_spacing_extra_mm)
    right_leg = widen_leg_spacing(right_leg, left_node_x, right_node_x, leg_spacing_extra_mm)
    left_leg = offset_points(
        left_leg,
        dx=left_leg_offset_x_mm,
        dy=left_leg_offset_y_mm,
        dz=left_leg_offset_z_mm,
    )
    right_leg = trim_right_leg_to_quarter_arc(right_leg, points_per_mm=points_per_mm, stop_b_deg=90.0)
    right_leg = offset_points(
        right_leg,
        dx=right_leg_offset_x_mm,
        dy=right_leg_offset_y_mm,
        dz=right_leg_offset_z_mm,
    )

    # Apply the requested over-extension on the side of the horizontal bar.
    left_leg_ext, _ = append_extension(left_leg, vertical_overextend_mm)
    right_leg_ext, _ = prepend_extension(right_leg, vertical_overextend_mm)

    left_leg_ext = deduplicate_polyline_points(smooth_polyline(resample_polyline(left_leg_ext, points_per_mm), 5))
    top_bar_no_flare = deduplicate_polyline_points(smooth_polyline(resample_polyline(top_bar_raw, points_per_mm), 5))
    right_leg_ext = deduplicate_polyline_points(smooth_polyline(resample_polyline(right_leg_ext, points_per_mm), 5))
    left_leg_curve_end_mm = arclength_at_first_b_below(left_leg_ext, 5.0)
    left_leg_extrusion_start_mm = left_leg_extrusion_start_fraction * left_leg_curve_end_mm

    inferred_radius = infer_left_hook_radius_mm(top_bar_raw)
    flare_radius = top_flare_radius_mm if top_flare_radius_mm > 0.0 else inferred_radius
    top_flare = build_top_bar_flare(
        start_point=top_bar_no_flare[-1],
        radius_mm=flare_radius,
        flare_degrees=top_flare_degrees,
        points_per_mm=points_per_mm,
    )
    top_flare = deduplicate_polyline_points(smooth_polyline(resample_polyline(top_flare, points_per_mm), 3))

    # Split the top bar so the left hook and horizontal run can use the curl
    # branch, then the new 90->45 flare can switch to release without a tip jump.
    z_plateau = float(np.max(top_bar_no_flare[:, 2]))
    left_arc_candidates = np.where(top_bar_no_flare[:, 2] >= z_plateau - 0.05)[0]
    left_arc_end_idx = int(left_arc_candidates[0]) if left_arc_candidates.size else max(1, len(top_bar_no_flare) // 4)
    left_arc_end_idx = int(np.clip(left_arc_end_idx, 1, len(top_bar_no_flare) - 2))
    top_bar_left_arc = top_bar_no_flare[: left_arc_end_idx + 1]
    top_bar_horizontal = top_bar_no_flare[left_arc_end_idx:]
    top_bar_horizontal = trim_polyline_end_by_length(top_bar_horizontal, top_bar_shorten_mm)

    # Split the right leg into 0->180 curl and 180->90 uncurl.  The raw
    # template's extra sub-90 tail is removed so the bottom bend stays a true
    # quarter-circle before transitioning into a straight horizontal run.
    right_to_180, right_uncurl_and_tail = split_polyline_by_first_b_below(right_leg_ext, 170.0)
    right_180_to_90, right_sub90_tail = split_polyline_by_first_b_below(right_uncurl_and_tail, 90.0)
    right_leg_quarter_extrusion_stop_mm = right_leg_extrusion_stop_fraction * polyline_length(right_180_to_90)

    profile_090_release = profile_for_phase(fit_strategy, "release", calibration_set="0-90-0")
    profile_090_pull = profile_for_phase(fit_strategy, "pull", calibration_set="0-90-0")
    profile_0180_pull = profile_for_phase(fit_strategy, "pull", calibration_set="0-180-0")
    profile_18090_release = profile_for_phase(fit_strategy, "release", calibration_set="90-180-90")

    branches: List[Dict[str, Any]] = [
        {
            "name": "left_leg_90_to_0_release",
            "points": left_leg_ext,
            "slow_distances_mm": [polyline_length(left_leg_ext)],
            "extrusion_start_distance_mm": left_leg_extrusion_start_mm,
            "motion_phase": profile_090_release.motion_phase,
            "fit_model_selector": profile_090_release.fit_model_selector,
        },
        {
            "name": "top_bar_left_arc_0_to_90_curl",
            "points": top_bar_left_arc,
            "slow_distances_mm": [0.0],
            "motion_phase": profile_090_pull.motion_phase,
            "fit_model_selector": profile_090_pull.fit_model_selector,
        },
        {
            "name": "top_bar_horizontal_0_to_90_curl",
            "points": top_bar_horizontal,
            "slow_distances_mm": [],
            "motion_phase": profile_090_pull.motion_phase,
            "fit_model_selector": profile_090_pull.fit_model_selector,
            "connect_from_previous": True,
        },
    ]

    if len(top_flare) >= 2 and top_flare_degrees > 0.0 and flare_radius > 0.0:
        branches.append(
            {
                "name": "top_bar_flare_90_to_45_release",
                "points": top_flare,
                "slow_distances_mm": [polyline_length(top_flare)],
                "motion_phase": profile_090_release.motion_phase,
                "fit_model_selector": profile_090_release.fit_model_selector,
                "connect_from_previous": True,
                "flare_radius_mm": flare_radius,
                "flare_degrees": top_flare_degrees,
            }
        )

    branches.extend(
        [
            {
                "name": "right_leg_0_to_180_curl",
                "points": right_to_180,
                "slow_distances_mm": [0.0],
                "motion_phase": profile_0180_pull.motion_phase,
                "fit_model_selector": profile_0180_pull.fit_model_selector,
            },
            {
                "name": "right_leg_180_to_90_uncurl",
                "points": right_180_to_90,
                "slow_distances_mm": [],
                "extrusion_end_distance_mm": right_leg_quarter_extrusion_stop_mm,
                "motion_phase": profile_18090_release.motion_phase,
                "fit_model_selector": profile_18090_release.fit_model_selector,
                "connect_from_previous": True,
            },
        ]
    )

    if len(right_sub90_tail) >= 2:
        branches.append(
            {
                "name": "right_leg_horizontal_release_finish",
                "points": right_sub90_tail,
                "slow_distances_mm": [polyline_length(right_sub90_tail)],
                "extrusion_start_distance_mm": float("inf"),
                "motion_phase": profile_090_release.motion_phase,
                "fit_model_selector": profile_090_release.fit_model_selector,
                "connect_from_previous": True,
            }
        )

    return [b for b in branches if len(np.asarray(b["points"], dtype=float)) >= 2]


# ---------------- G-code helpers ----------------
def _fmt_axes_move(axes_vals: Sequence[Tuple[str, float]]) -> str:
    return " ".join(f"{ax}{val:.3f}" for ax, val in axes_vals)


class GCodeWriter:
    def __init__(
        self,
        fh,
        cal: Optional[Calibration],
        write_mode: str,
        orientation_mode: str,
        fixed_b: float,
        fixed_c: float,
        c_deg: float,
        b_angle_bias_deg: float,
        bc_solve_samples: int,
        bbox: Dict[str, float],
        travel_feed: float,
        approach_feed: float,
        fine_approach_feed: float,
        node_connection_feed: float,
        print_feed: float,
        edge_samples: int,
        emit_extrusion: bool,
        preflow_dwell_ms: int,
        branch_transition_mode: str,
        phase_transition_steps: int = DEFAULT_PHASE_TRANSITION_STEPS,
    ) -> None:
        self.f = fh
        self.cal = cal
        self.write_mode = str(write_mode).strip().lower()
        self.orientation_mode = str(orientation_mode).strip().lower()
        self.fixed_b = float(fixed_b)
        self.fixed_c = float(fixed_c)
        self.c_deg = float(c_deg)
        self.b_angle_bias_deg = float(b_angle_bias_deg)
        self.bc_solve_samples = int(bc_solve_samples)
        self.bbox = dict(bbox)
        self.travel_feed = float(travel_feed)
        self.approach_feed = float(approach_feed)
        self.fine_approach_feed = float(fine_approach_feed)
        self.node_connection_feed = float(node_connection_feed)
        self.print_feed = float(print_feed)
        self.edge_samples = max(1, int(edge_samples))
        self.emit_extrusion = bool(emit_extrusion)
        self.preflow_dwell_ms = int(preflow_dwell_ms)
        self.branch_transition_mode = str(branch_transition_mode).strip().lower()
        self.phase_transition_steps = max(2, int(phase_transition_steps))

        self.pressure_charged = False
        self.cur_stage_xyz: Optional[np.ndarray] = None
        self.cur_tip_xyz: Optional[np.ndarray] = None
        self.cur_b: float = 0.0
        self.cur_c: float = self.c_deg
        self.last_tip_tangent: Optional[np.ndarray] = None
        self.current_motion_phase: Optional[str] = None
        self.current_fit_model_selector: Optional[str] = None
        self.warnings: List[str] = []

        self.stage_min = np.array([np.inf, np.inf, np.inf], dtype=float)
        self.stage_max = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
        self.b_min_used = float("inf")
        self.b_max_used = float("-inf")
        self.c_min_used = float("inf")
        self.c_max_used = float("-inf")

        if self.write_mode == "calibrated" and self.cal is None:
            raise ValueError("Calibration is required for calibrated mode.")

        if self.write_mode == "calibrated" and self.orientation_mode == "fixed" and self.fixed_b is None:
            self.fixed_b = solve_b_for_target_tip_angle(self.cal, 90.0, search_samples=self.bc_solve_samples)

    @property
    def x_axis(self) -> str:
        return self.cal.x_axis if self.cal is not None else "X"

    @property
    def y_axis(self) -> str:
        return self.cal.y_axis if self.cal is not None else "Y"

    @property
    def z_axis(self) -> str:
        return self.cal.z_axis if self.cal is not None else "Z"

    @property
    def b_axis(self) -> str:
        return self.cal.b_axis if self.cal is not None else "B"

    @property
    def c_axis(self) -> str:
        return self.cal.c_axis if self.cal is not None else "C"

    def clamp_stage(self, p_stage: np.ndarray, context: str) -> np.ndarray:
        x = float(np.clip(p_stage[0], self.bbox["x_min"], self.bbox["x_max"]))
        y = float(np.clip(p_stage[1], self.bbox["y_min"], self.bbox["y_max"]))
        z = float(np.clip(p_stage[2], self.bbox["z_min"], self.bbox["z_max"]))
        if abs(x - float(p_stage[0])) > 1e-12 or abs(y - float(p_stage[1])) > 1e-12 or abs(z - float(p_stage[2])) > 1e-12:
            self.warnings.append(f"WARNING: {context} stage point clamped to bbox.")
        return np.array([x, y, z], dtype=float)

    def bc_for_tangent(
        self,
        tangent: np.ndarray,
        motion_phase: Optional[str] = None,
        fit_model_selector: Optional[str] = None,
    ) -> Tuple[float, float]:
        if self.orientation_mode == "fixed":
            return float(self.fixed_b), float(self.fixed_c)

        target_b = desired_physical_b_angle_from_tangent(tangent) + float(self.b_angle_bias_deg)
        target_b = float(np.clip(target_b, 0.0, 180.0))

        if self.write_mode == "calibrated":
            assert self.cal is not None
            b = solve_b_for_target_tip_angle(
                self.cal,
                target_b,
                search_samples=self.bc_solve_samples,
                motion_phase=motion_phase,
                fit_model_selector=fit_model_selector,
            )
        else:
            b = target_b

        return float(b), float(self.c_deg)

    def solve_b_profile(
        self,
        tangents: np.ndarray,
        motion_phase: Optional[str] = None,
        fit_model_selector: Optional[str] = None,
    ) -> np.ndarray:
        tangents_arr = np.asarray(tangents, dtype=float)
        if len(tangents_arr) == 0:
            return np.zeros(0, dtype=float)

        if self.orientation_mode == "fixed":
            return np.full(len(tangents_arr), float(self.fixed_b), dtype=float)

        target_b = np.asarray(
            [desired_physical_b_angle_from_tangent(t) for t in tangents_arr],
            dtype=float,
        )
        target_b = smooth_b_angle_profile(target_b, window=7)
        target_b = np.clip(target_b + float(self.b_angle_bias_deg), 0.0, 180.0)

        if self.write_mode == "calibrated":
            assert self.cal is not None
            solved = np.asarray(
                [
                    solve_b_for_target_tip_angle(
                        self.cal,
                        float(b_target),
                        search_samples=self.bc_solve_samples,
                        motion_phase=motion_phase,
                        fit_model_selector=fit_model_selector,
                    )
                    for b_target in target_b
                ],
                dtype=float,
            )
            return smooth_b_angle_profile(solved, window=5)

        return smooth_b_angle_profile(target_b, window=5)

    def tip_to_stage_with_bc(
        self,
        p_tip: np.ndarray,
        b: float,
        c: float,
        motion_phase: Optional[str] = None,
        fit_model_selector: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, float]:
        if self.write_mode == "calibrated":
            assert self.cal is not None
            p_stage = stage_xyz_for_tip(
                self.cal,
                np.asarray(p_tip, dtype=float),
                float(b),
                float(c),
                motion_phase=motion_phase,
                fit_model_selector=fit_model_selector,
            )
        else:
            p_stage = np.asarray(p_tip, dtype=float)

        return self.clamp_stage(p_stage, "tip_to_stage"), float(b), float(c)

    def tip_to_stage(
        self,
        p_tip: np.ndarray,
        tangent: Optional[np.ndarray],
        motion_phase: Optional[str] = None,
        fit_model_selector: Optional[str] = None,
    ) -> Tuple[np.ndarray, float, float]:
        tangent_arr = np.array([1.0, 0.0, 0.0], dtype=float) if tangent is None else normalize(np.asarray(tangent, dtype=float))
        b, c = self.bc_for_tangent(tangent_arr, motion_phase=motion_phase, fit_model_selector=fit_model_selector)

        return self.tip_to_stage_with_bc(
            p_tip=np.asarray(p_tip, dtype=float),
            b=b,
            c=c,
            motion_phase=motion_phase,
            fit_model_selector=fit_model_selector,
        )

    def write_move(
        self,
        p_stage: np.ndarray,
        b: float,
        c: float,
        feed: float,
        comment: Optional[str] = None,
    ) -> None:
        if comment:
            self.f.write(f"; {comment}\n")
        axes: List[Tuple[str, float]] = [
            (self.x_axis, float(p_stage[0])),
            (self.y_axis, float(p_stage[1])),
            (self.z_axis, float(p_stage[2])),
            (self.b_axis, float(b)),
            (self.c_axis, float(c)),
        ]
        self.f.write(f"G1 {_fmt_axes_move(axes)} F{float(feed):.0f}\n")

        p_stage_arr = np.asarray(p_stage, dtype=float).copy()
        self.cur_stage_xyz = p_stage_arr
        self.cur_b = float(b)
        self.cur_c = float(c)
        self.stage_min = np.minimum(self.stage_min, p_stage_arr)
        self.stage_max = np.maximum(self.stage_max, p_stage_arr)
        self.b_min_used = min(self.b_min_used, float(b))
        self.b_max_used = max(self.b_max_used, float(b))
        self.c_min_used = min(self.c_min_used, float(c))
        self.c_max_used = max(self.c_max_used, float(c))

    def move_to_tip(
        self,
        p_tip: np.ndarray,
        tangent: Optional[np.ndarray],
        feed: float,
        comment: Optional[str] = None,
        motion_phase: Optional[str] = None,
        fit_model_selector: Optional[str] = None,
    ) -> None:
        p_stage, b, c = self.tip_to_stage(
            np.asarray(p_tip, dtype=float),
            tangent=tangent,
            motion_phase=motion_phase,
            fit_model_selector=fit_model_selector,
        )
        self.write_move(p_stage, b, c, feed, comment=comment)
        self.cur_tip_xyz = np.asarray(p_tip, dtype=float).copy()
        self.last_tip_tangent = None if tangent is None else np.asarray(tangent, dtype=float).copy()
        self.current_motion_phase = motion_phase
        self.current_fit_model_selector = fit_model_selector

    def transition_motion_phase(
        self,
        motion_phase: Optional[str],
        fit_model_selector: Optional[str],
        feed: float,
    ) -> None:
        if (
            self.write_mode != "calibrated"
            or self.branch_transition_mode != "smooth"
            or self.cur_tip_xyz is None
            or self.last_tip_tangent is None
            or self.current_motion_phase is None
            or motion_phase is None
            or (
                self.current_motion_phase == motion_phase
                and self.current_fit_model_selector == fit_model_selector
            )
        ):
            self.current_motion_phase = motion_phase
            self.current_fit_model_selector = fit_model_selector
            return

        p_tip = np.asarray(self.cur_tip_xyz, dtype=float)
        tangent = np.asarray(self.last_tip_tangent, dtype=float)
        start_stage, start_b, start_c = self.tip_to_stage(
            p_tip,
            tangent=tangent,
            motion_phase=self.current_motion_phase,
            fit_model_selector=self.current_fit_model_selector,
        )
        end_stage, end_b, end_c = self.tip_to_stage(
            p_tip,
            tangent=tangent,
            motion_phase=motion_phase,
            fit_model_selector=fit_model_selector,
        )

        self.f.write(
            f"; smooth phase transition {self.current_motion_phase} -> {motion_phase} at fixed tip pose\n"
        )
        for i in range(1, self.phase_transition_steps + 1):
            alpha = i / float(self.phase_transition_steps)
            p_stage = (1.0 - alpha) * start_stage + alpha * end_stage
            b = (1.0 - alpha) * float(start_b) + alpha * float(end_b)
            c = (1.0 - alpha) * float(start_c) + alpha * float(end_c)
            self.write_move(p_stage, b, c, feed, comment=None)

        self.cur_tip_xyz = p_tip.copy()
        self.last_tip_tangent = tangent.copy()
        self.current_motion_phase = motion_phase
        self.current_fit_model_selector = fit_model_selector

    def pressure_preload_before_print(self) -> None:
        if self.emit_extrusion and not self.pressure_charged:
            self.pressure_charged = True
            self.f.write("; open pressure solenoid before print pass\n")
            self.f.write("M42 P0 S1\n")
            if self.preflow_dwell_ms > 0:
                self.f.write(f"G4 P{self.preflow_dwell_ms}\n")

    def pressure_release_after_print(self) -> None:
        if self.emit_extrusion and self.pressure_charged:
            self.pressure_charged = False
            self.f.write("; close pressure solenoid after print pass\n")
            self.f.write("M42 P0 S0\n")

    def approach_start(
        self,
        start_tip: np.ndarray,
        start_tangent: np.ndarray,
        travel_lift_z: float,
        approach_side_mm: float,
        retreat_side_mm: float,
        motion_phase: Optional[str],
        fit_model_selector: Optional[str],
    ) -> None:
        start_tip = np.asarray(start_tip, dtype=float)
        start_tangent = normalize(np.asarray(start_tangent, dtype=float))
        start_side = side_vector_from_tangent(start_tangent, fallback=self.last_tip_tangent)

        if self.cur_tip_xyz is not None and self.last_tip_tangent is not None:
            self.transition_motion_phase(
                motion_phase=motion_phase,
                fit_model_selector=fit_model_selector,
                feed=self.approach_feed,
            )
            retreat_side = side_vector_from_tangent(self.last_tip_tangent, fallback=start_side)
            retreat_tip = (
                np.asarray(self.cur_tip_xyz, dtype=float)
                + retreat_side * float(retreat_side_mm)
                + np.array([0.0, 0.0, float(travel_lift_z)], dtype=float)
            )
            self.move_to_tip(
                retreat_tip,
                tangent=self.last_tip_tangent,
                feed=self.approach_feed,
                comment="retreat from previous end before repositioning",
                motion_phase=motion_phase,
                fit_model_selector=fit_model_selector,
            )
        else:
            self.current_motion_phase = motion_phase
            self.current_fit_model_selector = fit_model_selector

        far_tip = start_tip - start_side * float(approach_side_mm) + np.array([0.0, 0.0, float(travel_lift_z)], dtype=float)
        near_tip = start_tip - start_side * (0.5 * float(approach_side_mm)) + np.array([0.0, 0.0, 0.5 * float(travel_lift_z)], dtype=float)
        self.move_to_tip(
            far_tip,
            tangent=start_tangent,
            feed=self.travel_feed,
            comment="travel above and outside the glyph",
            motion_phase=motion_phase,
            fit_model_selector=fit_model_selector,
        )
        self.move_to_tip(
            near_tip,
            tangent=start_tangent,
            feed=self.approach_feed,
            comment="approach toward the next start",
            motion_phase=motion_phase,
            fit_model_selector=fit_model_selector,
        )
        self.move_to_tip(
            start_tip,
            tangent=start_tangent,
            feed=self.fine_approach_feed,
            comment="fine approach to the next start",
            motion_phase=motion_phase,
            fit_model_selector=fit_model_selector,
        )

    def approach_right_vertical_leg(
        self,
        start_tip: np.ndarray,
        start_tangent: np.ndarray,
        travel_lift_z: float,
        approach_side_mm: float,
        retreat_side_mm: float,
        motion_phase: Optional[str],
        fit_model_selector: Optional[str],
    ) -> None:
        if self.cur_tip_xyz is None or self.last_tip_tangent is None:
            self.approach_start(
                start_tip=start_tip,
                start_tangent=start_tangent,
                travel_lift_z=travel_lift_z,
                approach_side_mm=approach_side_mm,
                retreat_side_mm=retreat_side_mm,
                motion_phase=motion_phase,
                fit_model_selector=fit_model_selector,
            )
            return

        start_tip = np.asarray(start_tip, dtype=float)
        start_tangent = normalize(np.asarray(start_tangent, dtype=float))
        self.transition_motion_phase(
            motion_phase=motion_phase,
            fit_model_selector=fit_model_selector,
            feed=self.approach_feed,
        )

        lowered_tip = np.asarray(self.cur_tip_xyz, dtype=float) + np.array([0.0, 0.0, -float(travel_lift_z)], dtype=float)
        self.move_to_tip(
            lowered_tip,
            tangent=self.last_tip_tangent,
            feed=self.approach_feed,
            comment="drop in Z before curling B for right vertical leg",
            motion_phase=motion_phase,
            fit_model_selector=fit_model_selector,
        )
        self.move_to_tip(
            lowered_tip,
            tangent=start_tangent,
            feed=self.approach_feed,
            comment="curl B for right vertical leg while below the start",
            motion_phase=motion_phase,
            fit_model_selector=fit_model_selector,
        )

        approach_tip = start_tip + np.array([float(approach_side_mm), 0.0, -0.5 * float(travel_lift_z)], dtype=float)
        self.move_to_tip(
            approach_tip,
            tangent=start_tangent,
            feed=self.approach_feed,
            comment="move to the +X/-Z side of the right vertical start",
            motion_phase=motion_phase,
            fit_model_selector=fit_model_selector,
        )
        self.move_to_tip(
            start_tip,
            tangent=start_tangent,
            feed=self.fine_approach_feed,
            comment="move -X and +Z into the right vertical start",
            motion_phase=motion_phase,
            fit_model_selector=fit_model_selector,
        )

    def print_polyline(
        self,
        points: np.ndarray,
        tangents: np.ndarray,
        path_name: str,
        motion_phase: Optional[str] = None,
        fit_model_selector: Optional[str] = None,
        slow_distances_mm: Optional[Sequence[float]] = None,
        slow_radius_mm: float = DEFAULT_SLOW_ZONE_RADIUS_MM,
        extrusion_start_distance_mm: float = 0.0,
        extrusion_end_distance_mm: Optional[float] = None,
        close_pressure_at_end: bool = True,
    ) -> None:
        if len(points) < 2:
            return

        self.f.write(
            "; PI_PASS_START "
            f"name={path_name} "
            f"point_count={len(points)} "
            f"tip_start_x={float(points[0, 0]):.6f} tip_start_y={float(points[0, 1]):.6f} tip_start_z={float(points[0, 2]):.6f} "
            f"tip_end_x={float(points[-1, 0]):.6f} tip_end_y={float(points[-1, 1]):.6f} tip_end_z={float(points[-1, 2]):.6f} "
            "tip_angle_convention=0_posZ_90_horizontal_180_negZ\n"
        )

        slow_list = [float(v) for v in (slow_distances_mm or [])]
        extrusion_start_distance_mm = max(0.0, float(extrusion_start_distance_mm))
        extrusion_end_distance = None if extrusion_end_distance_mm is None else max(0.0, float(extrusion_end_distance_mm))
        last_tip = np.asarray(points[0], dtype=float).copy()
        self.cur_tip_xyz = last_tip.copy()
        self.last_tip_tangent = np.asarray(tangents[0], dtype=float).copy()
        self.current_motion_phase = motion_phase
        self.current_fit_model_selector = fit_model_selector

        sample_tips: List[np.ndarray] = [np.asarray(points[0], dtype=float).copy()]
        sample_tangents: List[np.ndarray] = [np.asarray(tangents[0], dtype=float).copy()]
        sample_arcs: List[float] = [0.0]

        cumulative = 0.0
        for i in range(1, len(points)):
            p0 = np.asarray(points[i - 1], dtype=float)
            p1 = np.asarray(points[i], dtype=float)
            t0 = np.asarray(tangents[i - 1], dtype=float)
            t1 = np.asarray(tangents[i], dtype=float)
            edge_len = float(np.linalg.norm(p1 - p0))

            for s in range(1, self.edge_samples + 1):
                u = s / float(self.edge_samples)
                p_tip = p0 + u * (p1 - p0)
                tangent = normalize((1.0 - u) * t0 + u * t1)
                sample_tips.append(p_tip.copy())
                sample_tangents.append(tangent.copy())
                sample_arcs.append(cumulative + u * edge_len)

            cumulative += edge_len

        b_profile = self.solve_b_profile(
            np.asarray(sample_tangents, dtype=float),
            motion_phase=motion_phase,
            fit_model_selector=fit_model_selector,
        )
        c_profile = np.full_like(b_profile, float(self.fixed_c if self.orientation_mode == "fixed" else self.c_deg), dtype=float)
        extrusion_ended = False
        if extrusion_start_distance_mm <= 1e-9:
            self.pressure_preload_before_print()

        for i in range(1, len(sample_tips)):
            p_tip = np.asarray(sample_tips[i], dtype=float)
            tangent = np.asarray(sample_tangents[i], dtype=float)
            p_stage, b, c = self.tip_to_stage_with_bc(
                p_tip,
                b=float(b_profile[i]),
                c=float(c_profile[i]),
                motion_phase=motion_phase,
                fit_model_selector=fit_model_selector,
            )

            arc_here = float(sample_arcs[i])
            feed = self.print_feed
            for slow_d in slow_list:
                if abs(arc_here - slow_d) <= float(slow_radius_mm):
                    feed = min(feed, self.node_connection_feed)
                    break

            if (not extrusion_ended) and (not self.pressure_charged) and arc_here >= extrusion_start_distance_mm:
                self.f.write(
                    f"; open pressure at arclength {arc_here:.3f} mm on {path_name}\n"
                )
                self.pressure_preload_before_print()

            if self.pressure_charged and extrusion_end_distance is not None and arc_here >= extrusion_end_distance:
                self.f.write(
                    f"; close pressure at arclength {arc_here:.3f} mm on {path_name}\n"
                )
                self.pressure_release_after_print()
                extrusion_ended = True

            self.write_move(p_stage, b, c, feed, comment=None)
            self.cur_tip_xyz = p_tip.copy()
            self.last_tip_tangent = tangent.copy()
            last_tip = p_tip.copy()

        if close_pressure_at_end:
            self.pressure_release_after_print()
        self.f.write("; PI_PASS_END\n")

    def finish(self, travel_lift_z: float, retreat_side_mm: float) -> None:
        if self.cur_tip_xyz is None or self.last_tip_tangent is None:
            return
        end_tip = np.asarray(self.cur_tip_xyz, dtype=float)
        x_clear_tip = end_tip + np.array([float(retreat_side_mm), 0.0, 0.0], dtype=float)
        self.move_to_tip(
            x_clear_tip,
            tangent=self.last_tip_tangent,
            feed=self.approach_feed,
            comment="small +X move after pi write",
            motion_phase=self.current_motion_phase,
            fit_model_selector=self.current_fit_model_selector,
        )
        lift_tip = x_clear_tip + np.array([0.0, 0.0, float(travel_lift_z)], dtype=float)
        self.move_to_tip(
            lift_tip,
            tangent=self.last_tip_tangent,
            feed=self.approach_feed,
            comment="+Z travel out after +X clearance",
            motion_phase=self.current_motion_phase,
            fit_model_selector=self.current_fit_model_selector,
        )


# ---------------- Top-level generation ----------------
def write_pi_sign_gcode(
    out: str,
    calibration: Optional[str],
    write_mode: str,
    orientation_mode: str,
    fit_strategy: str,
    offplane_fit_model: Optional[str],
    y_offplane_sign: float,
    x_left: float,
    x_right: float,
    y: float,
    z_bottom: float,
    z_top: float,
    vertical_overextend_mm: float,
    leg_spacing_extra_mm: float,
    left_leg_offset_x_mm: float,
    left_leg_offset_y_mm: float,
    left_leg_offset_z_mm: float,
    right_leg_offset_x_mm: float,
    right_leg_offset_y_mm: float,
    right_leg_offset_z_mm: float,
    left_leg_extrusion_start_fraction: float,
    right_leg_extrusion_stop_fraction: float,
    top_bar_shorten_mm: float,
    points_per_mm: float,
    tangent_smooth_window: int,
    slow_zone_radius_mm: float,
    top_flare_radius_mm: float,
    top_flare_degrees: float,
    fixed_b: float,
    fixed_c: float,
    c_deg: float,
    b_angle_bias_deg: float,
    bc_solve_samples: int,
    travel_feed: float,
    approach_feed: float,
    fine_approach_feed: float,
    node_connection_feed: float,
    print_feed: float,
    travel_lift_z: float,
    approach_side_mm: float,
    retreat_side_mm: float,
    edge_samples: int,
    branch_transition_mode: str,
    emit_extrusion: bool,
    preflow_dwell_ms: int,
    bbox_x_min: float,
    bbox_x_max: float,
    bbox_y_min: float,
    bbox_y_max: float,
    bbox_z_min: float,
    bbox_z_max: float,
) -> Dict[str, Any]:
    write_mode = str(write_mode).strip().lower()
    orientation_mode = str(orientation_mode).strip().lower()
    if write_mode not in {"calibrated", "cartesian"}:
        raise ValueError("--write-mode must be calibrated or cartesian")
    if orientation_mode not in {"tangent", "fixed"}:
        raise ValueError("--orientation-mode must be tangent or fixed")
    fit_strategy = str(fit_strategy).strip().lower().replace("_", "-")
    if fit_strategy not in {"fit", "phase-pchip", "avg-pchip", "avg-cubic", "avg-linear"}:
        raise ValueError("--fit-strategy must be fit, phase-pchip, avg-pchip, avg-cubic, or avg-linear")
    branch_transition_mode = str(branch_transition_mode).strip().lower()
    if branch_transition_mode not in {"jump", "smooth"}:
        raise ValueError("--branch-transition-mode must be jump or smooth")
    if write_mode == "calibrated" and not calibration:
        raise ValueError("--calibration is required when --write-mode calibrated")

    cal: Optional[Calibration]
    if write_mode == "calibrated":
        resolved_offplane_fit_model = resolve_offplane_fit_model(fit_strategy, offplane_fit_model)
        cal = load_calibration(str(calibration), requested_offplane_fit_model=resolved_offplane_fit_model)
        cal.offplane_y_sign = float(y_offplane_sign)
    else:
        cal = None
        resolved_offplane_fit_model = resolve_offplane_fit_model(fit_strategy, offplane_fit_model)

    bbox = {
        "x_min": float(bbox_x_min),
        "x_max": float(bbox_x_max),
        "y_min": float(bbox_y_min),
        "y_max": float(bbox_y_max),
        "z_min": float(bbox_z_min),
        "z_max": float(bbox_z_max),
    }

    branches = build_pi_sign_branches(
        x_left=x_left,
        x_right=x_right,
        y=y,
        z_bottom=z_bottom,
        z_top=z_top,
        vertical_overextend_mm=vertical_overextend_mm,
        leg_spacing_extra_mm=leg_spacing_extra_mm,
        left_leg_offset_x_mm=left_leg_offset_x_mm,
        left_leg_offset_y_mm=left_leg_offset_y_mm,
        left_leg_offset_z_mm=left_leg_offset_z_mm,
        right_leg_offset_x_mm=right_leg_offset_x_mm,
        right_leg_offset_y_mm=right_leg_offset_y_mm,
        right_leg_offset_z_mm=right_leg_offset_z_mm,
        left_leg_extrusion_start_fraction=left_leg_extrusion_start_fraction,
        right_leg_extrusion_stop_fraction=right_leg_extrusion_stop_fraction,
        top_bar_shorten_mm=top_bar_shorten_mm,
        points_per_mm=points_per_mm,
        fit_strategy=fit_strategy,
        top_flare_radius_mm=top_flare_radius_mm,
        top_flare_degrees=top_flare_degrees,
    )
    top_flare_effective_radius_mm = float(top_flare_radius_mm)
    for branch in branches:
        if "flare_radius_mm" in branch:
            top_flare_effective_radius_mm = float(branch["flare_radius_mm"])
            break

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    writer = GCodeWriter(
        fh=None,  # type: ignore[arg-type]
        cal=cal,
        write_mode=write_mode,
        orientation_mode=orientation_mode,
        fixed_b=fixed_b,
        fixed_c=fixed_c,
        c_deg=c_deg,
        b_angle_bias_deg=b_angle_bias_deg,
        bc_solve_samples=bc_solve_samples,
        bbox=bbox,
        travel_feed=travel_feed,
        approach_feed=approach_feed,
        fine_approach_feed=fine_approach_feed,
        node_connection_feed=node_connection_feed,
        print_feed=print_feed,
        edge_samples=edge_samples,
        emit_extrusion=emit_extrusion,
        preflow_dwell_ms=preflow_dwell_ms,
        branch_transition_mode=branch_transition_mode,
    )

    with out_path.open("w", encoding="utf-8") as fh:
        writer.f = fh
        fh.write("; Rounded pi sign in the XZ plane, matched to the supplied sign image\n")
        fh.write("; Generated by pi_sign_image_style_xz_generator.py\n")
        fh.write(
            f"; write_mode={write_mode} orientation_mode={orientation_mode} "
            f"fit_strategy={fit_strategy} branch_transition_mode={branch_transition_mode}\n"
        )
        fh.write(
            "; glyph_box "
            f"x=[{float(x_left):.3f},{float(x_right):.3f}] "
            f"y={float(y):.3f} "
            f"z=[{float(z_bottom):.3f},{float(z_top):.3f}]\n"
        )
        fh.write(
            "; shape_parameters "
            f"vertical_overextend_mm={float(vertical_overextend_mm):.3f} "
            f"leg_spacing_extra_mm={float(leg_spacing_extra_mm):.3f} "
            f"left_leg_offset_xyz_mm=({float(left_leg_offset_x_mm):.3f},{float(left_leg_offset_y_mm):.3f},{float(left_leg_offset_z_mm):.3f}) "
            f"right_leg_offset_xyz_mm=({float(right_leg_offset_x_mm):.3f},{float(right_leg_offset_y_mm):.3f},{float(right_leg_offset_z_mm):.3f}) "
            f"left_leg_extrusion_start_fraction={float(left_leg_extrusion_start_fraction):.3f} "
            f"right_leg_extrusion_stop_fraction={float(right_leg_extrusion_stop_fraction):.3f} "
            f"top_bar_shorten_mm={float(top_bar_shorten_mm):.3f} "
            f"points_per_mm={float(points_per_mm):.3f} "
            f"slow_zone_radius_mm={float(slow_zone_radius_mm):.3f}\n"
        )
        fh.write("G21\n")
        fh.write("G90\n")
        fh.write("; pressure actuation: open with M42 P0 S1, close with M42 P0 S0\n")
        fh.write("; B-angle convention: 0=up, 90=horizontal, 180=down\n")
        fh.write(f"; C held constant at {float(c_deg):.3f} deg by default\n")
        if cal is not None:
            fh.write(
                "; y_offplane_model "
                f"requested={cal.requested_offplane_fit_model or 'calibration_default'} "
                f"resolved={cal.resolved_offplane_fit_model or 'unspecified'} "
                f"calibration_default={cal.selected_offplane_fit_model or cal.selected_fit_model or 'unspecified'} "
                f"active_phase={cal.active_phase}\n"
            )

        for i_branch, branch in enumerate(branches):
            pts = np.asarray(branch["points"], dtype=float)
            connect_from_previous = bool(branch.get("connect_from_previous", False))
            motion_phase = None if cal is None else branch.get("motion_phase")
            fit_model_selector = None if cal is None else branch.get("fit_model_selector")

            if connect_from_previous and writer.cur_tip_xyz is not None:
                # In fit mode, keep the live tip as the reference and translate
                # the full incoming branch so its trajectory stays continuous.
                # Other modes only pin the first point to avoid a tip jump.
                pts = branch_points_from_live_tip(
                    pts,
                    writer.cur_tip_xyz,
                    keep_shape=(fit_strategy == "fit"),
                )
                tangents = build_tangents_for_points(pts, smooth_window=tangent_smooth_window)
                writer.transition_motion_phase(
                    motion_phase=motion_phase,
                    fit_model_selector=fit_model_selector,
                    feed=approach_feed,
                )
            else:
                tangents = build_tangents_for_points(pts, smooth_window=tangent_smooth_window)
                if str(branch["name"]).startswith("right_leg_0_to_180"):
                    writer.approach_right_vertical_leg(
                        start_tip=pts[0],
                        start_tangent=tangents[0],
                        travel_lift_z=travel_lift_z,
                        approach_side_mm=approach_side_mm,
                        retreat_side_mm=retreat_side_mm,
                        motion_phase=motion_phase,
                        fit_model_selector=fit_model_selector,
                    )
                else:
                    writer.approach_start(
                        start_tip=pts[0],
                        start_tangent=tangents[0],
                        travel_lift_z=travel_lift_z,
                        approach_side_mm=approach_side_mm,
                        retreat_side_mm=retreat_side_mm,
                        motion_phase=motion_phase,
                        fit_model_selector=fit_model_selector,
                    )

            next_connects = i_branch + 1 < len(branches) and bool(branches[i_branch + 1].get("connect_from_previous", False))
            writer.print_polyline(
                pts,
                tangents,
                path_name=str(branch["name"]),
                motion_phase=motion_phase,
                fit_model_selector=fit_model_selector,
                slow_distances_mm=branch.get("slow_distances_mm", []),
                slow_radius_mm=slow_zone_radius_mm,
                extrusion_start_distance_mm=float(branch.get("extrusion_start_distance_mm", 0.0)),
                extrusion_end_distance_mm=branch.get("extrusion_end_distance_mm"),
                close_pressure_at_end=not next_connects,
            )

        writer.finish(travel_lift_z=travel_lift_z, retreat_side_mm=retreat_side_mm)
        fh.write("; End of file\n")

    all_points = np.vstack([np.asarray(b["points"], dtype=float) for b in branches])
    tip_b_segments: List[np.ndarray] = []
    b_used_segments: List[np.ndarray] = []
    for branch in branches:
        pts = np.asarray(branch["points"], dtype=float)
        tangents = build_tangents_for_points(pts, smooth_window=tangent_smooth_window)
        tip_b_branch = np.array([desired_physical_b_angle_from_tangent(t) for t in tangents], dtype=float)
        tip_b_segments.append(tip_b_branch)
        motion_phase = branch.get("motion_phase")
        fit_model_selector = branch.get("fit_model_selector")
        b_branch = writer.solve_b_profile(
            tangents,
            motion_phase=motion_phase,
            fit_model_selector=fit_model_selector,
        )
        b_used_segments.append(b_branch)
    tip_b = np.concatenate(tip_b_segments)
    b_used = np.concatenate(b_used_segments)

    summary = {
        "out": str(out_path),
        "write_mode": write_mode,
        "orientation_mode": orientation_mode,
        "fit_strategy": fit_strategy,
        "offplane_fit_model": resolved_offplane_fit_model,
        "branch_transition_mode": branch_transition_mode,
        "leg_spacing_extra_mm": float(leg_spacing_extra_mm),
        "left_leg_offset_xyz_mm": (
            float(left_leg_offset_x_mm),
            float(left_leg_offset_y_mm),
            float(left_leg_offset_z_mm),
        ),
        "right_leg_offset_xyz_mm": (
            float(right_leg_offset_x_mm),
            float(right_leg_offset_y_mm),
            float(right_leg_offset_z_mm),
        ),
        "left_leg_extrusion_start_fraction": float(left_leg_extrusion_start_fraction),
        "right_leg_extrusion_stop_fraction": float(right_leg_extrusion_stop_fraction),
        "top_bar_shorten_mm": float(top_bar_shorten_mm),
        "tip_x_range": (float(np.min(all_points[:, 0])), float(np.max(all_points[:, 0]))),
        "tip_y_range": (float(np.min(all_points[:, 1])), float(np.max(all_points[:, 1]))),
        "tip_z_range": (float(np.min(all_points[:, 2])), float(np.max(all_points[:, 2]))),
        "tip_b_target_range_deg": (float(np.min(tip_b)), float(np.max(tip_b))),
        "b_command_range_deg": (float(np.min(b_used)), float(np.max(b_used))),
        "c_command_deg": float(c_deg if orientation_mode == "tangent" else fixed_c),
        "branch_order": [str(b["name"]) for b in branches],
        "branch_motion_phases": [str(b.get("motion_phase", "")) for b in branches],
        "top_flare_radius_mm": float(top_flare_radius_mm),
        "top_flare_effective_radius_mm": float(top_flare_effective_radius_mm),
        "top_flare_degrees": float(top_flare_degrees),
        "stage_xyz_range": {
            "x": (float(writer.stage_min[0]), float(writer.stage_max[0])),
            "y": (float(writer.stage_min[1]), float(writer.stage_max[1])),
            "z": (float(writer.stage_min[2]), float(writer.stage_max[2])),
        },
        "warnings": list(writer.warnings),
    }
    return summary


# ---------------- CLI ----------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Generate calibrated or Cartesian G-code for a rounded π glyph in the XZ plane. "
            "The geometry is matched to the supplied sign image. "
            "The vertical legs are printed as separate passes, the top bar is printed left-to-right, "
            "travel avoids already printed material, and the leg/bar nodes are slowed down."
        )
    )
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code file.")
    ap.add_argument("--calibration", default=None, help="Calibration JSON. Required for --write-mode calibrated.")
    ap.add_argument("--write-mode", choices=["calibrated", "cartesian"], default=DEFAULT_WRITE_MODE)
    ap.add_argument("--orientation-mode", choices=["tangent", "fixed"], default=DEFAULT_ORIENTATION_MODE)
    ap.add_argument(
        "--fit-strategy",
        choices=["fit", "phase-pchip", "avg-pchip", "avg-cubic", "avg-linear"],
        default=DEFAULT_FIT_STRATEGY,
        help="Use the 0-180-0 pull/release branch pair with continuous branch anchoring ('fit'), separate pull/release PCHIP, average PCHIP, average cubic, or average linear models for r/z/tip-angle tracking.",
    )
    ap.add_argument(
        "--offplane-fit-model",
        default=None,
        help="Selector for offplane_y model, e.g. avg_pchip, avg_cubic, avg_linear, pchip, cubic, linear. Defaults to avg_linear with --fit-strategy avg-linear, otherwise avg_pchip.",
    )
    ap.add_argument("--y-offplane-sign", type=float, default=DEFAULT_Y_OFFPLANE_SIGN, help="Multiplier applied to the calibration off-plane Y term in calibrated mode. Use -1 to flip the sign.")

    ap.add_argument("--x-left", type=float, default=DEFAULT_X_LEFT, help="Left edge of the overall glyph box.")
    ap.add_argument("--x-right", type=float, default=DEFAULT_X_RIGHT, help="Right edge of the overall glyph box.")
    ap.add_argument("--y", type=float, default=DEFAULT_Y)
    ap.add_argument("--z-bottom", type=float, default=DEFAULT_Z_BOTTOM, help="Bottom edge of the overall glyph box.")
    ap.add_argument("--z-top", type=float, default=DEFAULT_Z_TOP, help="Top edge of the overall glyph box.")

    ap.add_argument("--vertical-overextend-mm", type=float, default=DEFAULT_VERTICAL_OVEREXTEND_MM, help="How far the vertical legs extend past the top bar on the bar side.")
    ap.add_argument("--leg-spacing-extra-mm", type=float, default=DEFAULT_LEG_SPACING_EXTRA_MM, help="Extra horizontal separation added between the two vertical legs.")
    ap.add_argument("--left-leg-offset-x-mm", type=float, default=DEFAULT_LEFT_LEG_OFFSET_X_MM, help="X offset applied to the left leg only.")
    ap.add_argument("--left-leg-offset-y-mm", type=float, default=DEFAULT_LEFT_LEG_OFFSET_Y_MM, help="Y offset applied to the left leg only.")
    ap.add_argument("--left-leg-offset-z-mm", type=float, default=DEFAULT_LEFT_LEG_OFFSET_Z_MM, help="Z offset applied to the left leg only.")
    ap.add_argument("--right-leg-offset-x-mm", type=float, default=DEFAULT_RIGHT_LEG_OFFSET_X_MM, help="X offset applied to the right leg only.")
    ap.add_argument("--right-leg-offset-y-mm", type=float, default=DEFAULT_RIGHT_LEG_OFFSET_Y_MM, help="Y offset applied to the right leg only.")
    ap.add_argument("--right-leg-offset-z-mm", type=float, default=DEFAULT_RIGHT_LEG_OFFSET_Z_MM, help="Z offset applied to the right leg only.")
    ap.add_argument("--left-leg-extrusion-start-fraction", type=float, default=DEFAULT_LEFT_LEG_EXTRUSION_START_FRACTION, help="Where to start extrusion through the initial left-leg curl, as a fraction of that curl arclength.")
    ap.add_argument("--right-leg-extrusion-stop-fraction", type=float, default=DEFAULT_RIGHT_LEG_EXTRUSION_STOP_FRACTION, help="Where to stop extrusion through the right-leg 180->90 quarter-circle, as a fraction of that branch arclength.")
    ap.add_argument("--top-bar-shorten-mm", type=float, default=DEFAULT_TOP_BAR_SHORTEN_MM, help="How much to trim from the right end of the horizontal top bar.")
    ap.add_argument("--points-per-mm", type=float, default=DEFAULT_POINTS_PER_MM, help="Polyline resampling density.")
    ap.add_argument("--tangent-smooth-window", type=int, default=DEFAULT_TANGENT_SMOOTH_WINDOW)
    ap.add_argument("--slow-zone-radius-mm", type=float, default=DEFAULT_SLOW_ZONE_RADIUS_MM, help="Distance around each leg/bar node where print feed is reduced.")
    ap.add_argument("--top-flare-radius-mm", type=float, default=DEFAULT_TOP_FLARE_RADIUS_MM, help="Radius for the right-end top-bar flare. Use 0 to match the inferred radius of the left top-bar quarter-circle.")
    ap.add_argument("--top-flare-degrees", type=float, default=DEFAULT_TOP_FLARE_DEGREES, help="Flare sweep in degrees. Default 0 disables the extra right-end flare.")

    ap.add_argument("--fixed-b", type=float, default=DEFAULT_FIXED_B, help="Used only when --orientation-mode fixed.")
    ap.add_argument("--fixed-c", type=float, default=DEFAULT_FIXED_C, help="Used only when --orientation-mode fixed.")
    ap.add_argument("--c-deg", type=float, default=DEFAULT_C_DEG, help="Constant C angle used in tangent mode. Default is 180 deg for an XZ-plane write.")
    ap.add_argument("--b-angle-bias-deg", type=float, default=DEFAULT_B_ANGLE_BIAS_DEG, help="Bias added to the tangent-derived B target before solving / emitting.")
    ap.add_argument("--bc-solve-samples", type=int, default=DEFAULT_BC_SOLVE_SAMPLES)

    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED)
    ap.add_argument("--approach-feed", type=float, default=DEFAULT_APPROACH_FEED)
    ap.add_argument("--fine-approach-feed", type=float, default=DEFAULT_FINE_APPROACH_FEED)
    ap.add_argument("--node-connection-feed", type=float, default=DEFAULT_NODE_CONNECTION_FEED)
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED)
    ap.add_argument("--travel-lift-z", type=float, default=DEFAULT_TRAVEL_LIFT_Z)
    ap.add_argument("--approach-side-mm", type=float, default=DEFAULT_APPROACH_SIDE_MM)
    ap.add_argument("--retreat-side-mm", type=float, default=DEFAULT_RETREAT_SIDE_MM)
    ap.add_argument("--edge-samples", type=int, default=DEFAULT_EDGE_SAMPLES)
    ap.add_argument(
        "--branch-transition-mode",
        choices=["jump", "smooth"],
        default=DEFAULT_BRANCH_TRANSITION_MODE,
        help="How to switch pull/release branch models between passes. 'smooth' blends the phase change at fixed tip pose before repositioning.",
    )

    ap.add_argument("--emit-extrusion", dest="emit_extrusion", action="store_true", default=DEFAULT_EMIT_EXTRUSION)
    ap.add_argument("--no-emit-extrusion", dest="emit_extrusion", action="store_false")
    ap.add_argument("--preflow-dwell-ms", type=int, default=DEFAULT_PREFLOW_DWELL_MS, help="Dwell after opening the pressure valve and before starting the print pass.")

    ap.add_argument("--bbox-x-min", type=float, default=DEFAULT_BBOX_X_MIN)
    ap.add_argument("--bbox-x-max", type=float, default=DEFAULT_BBOX_X_MAX)
    ap.add_argument("--bbox-y-min", type=float, default=DEFAULT_BBOX_Y_MIN)
    ap.add_argument("--bbox-y-max", type=float, default=DEFAULT_BBOX_Y_MAX)
    ap.add_argument("--bbox-z-min", type=float, default=DEFAULT_BBOX_Z_MIN)
    ap.add_argument("--bbox-z-max", type=float, default=DEFAULT_BBOX_Z_MAX)

    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    summary = write_pi_sign_gcode(**vars(args))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
