#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code for a rounded π glyph in the XZ plane.

This version matches the visual style of the uploaded sign much more closely:
- rounded left hook on the top bar
- two hanging vertical legs under the bar
- left bottom curl and right bottom curl
- separate print passes so the glyph layout matches the sign
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
DEFAULT_X_LEFT = 60.0
DEFAULT_X_RIGHT = 110.0
DEFAULT_Y = 52.0
DEFAULT_Z_BOTTOM = -150.0
DEFAULT_Z_TOP = -115.0

# Shape details
DEFAULT_VERTICAL_OVEREXTEND_MM = 1.5
DEFAULT_POINTS_PER_MM = 8.0
DEFAULT_TANGENT_SMOOTH_WINDOW = 4
DEFAULT_SLOW_ZONE_RADIUS_MM = 1.2

# Orientation
DEFAULT_WRITE_MODE = "calibrated"
DEFAULT_ORIENTATION_MODE = "tangent"
DEFAULT_FIXED_B = 90.0
DEFAULT_FIXED_C = 180.0
DEFAULT_C_DEG = 180.0
DEFAULT_B_ANGLE_BIAS_DEG = 0.0
DEFAULT_BC_SOLVE_SAMPLES = 5001
DEFAULT_Y_OFFPLANE_SIGN = -1.0
DEFAULT_Y_OFFPLANE_FIT_MODEL = "avg_cubic"

# Motion
DEFAULT_TRAVEL_FEED = 1000.0
DEFAULT_APPROACH_FEED = 400.0
DEFAULT_FINE_APPROACH_FEED = 80.0
DEFAULT_NODE_CONNECTION_FEED = 35.0
DEFAULT_PRINT_FEED = 250.0
DEFAULT_TRAVEL_LIFT_Z = 8.0
DEFAULT_APPROACH_SIDE_MM = 4.0
DEFAULT_RETREAT_SIDE_MM = 4.0
DEFAULT_EDGE_SAMPLES = 1

# Extrusion / pressure
DEFAULT_EMIT_EXTRUSION = True
DEFAULT_EXTRUSION_PER_MM = 0.0015
DEFAULT_PRESSURE_OFFSET_MM = 4.0
DEFAULT_PRESSURE_ADVANCE_FEED = 120.0
DEFAULT_PRESSURE_RETRACT_FEED = 240.0
DEFAULT_PREFLOW_DWELL_MS = 500

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
    u_axis: str

    c_180_deg: float

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
    active_phase = str(data.get("default_phase_for_legacy_access") or "pull").strip().lower()

    phase_models = data.get("fit_models_by_phase", {}) or {}
    active_phase_models = phase_models.get(active_phase) if isinstance(phase_models, dict) else None
    if not isinstance(active_phase_models, dict):
        active_phase_models = fit_models

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
        u_axis=str(duet_map.get("extruder_axis") or "U"),
        c_180_deg=float(motor_setup.get("rotation_axis_180_deg", 180.0)),
    )


def eval_r(cal: Calibration, b: Any) -> np.ndarray:
    if cal.r_model is not None:
        return eval_model_spec(cal.r_model, b)
    return poly_eval(cal.pr, b)


def eval_z(cal: Calibration, b: Any) -> np.ndarray:
    if cal.z_model is not None:
        return eval_model_spec(cal.z_model, b)
    return poly_eval(cal.pz, b)


def eval_offplane_y(cal: Calibration, b: Any) -> np.ndarray:
    if cal.y_off_model is not None:
        if str(cal.y_off_model.get("model_type", "")).lower() == "pchip":
            values = eval_pchip_with_linear_extrap(cal.y_off_model, cal.y_off_extrap_model, b)
        else:
            values = eval_model_spec(cal.y_off_model, b, default_if_none=0.0)
    else:
        values = poly_eval(cal.py_off, b, default_if_none=0.0)
    return float(cal.offplane_y_sign) * np.asarray(values, dtype=float)


def eval_tip_angle_deg(cal: Calibration, b: Any) -> np.ndarray:
    if cal.tip_angle_model is not None:
        return eval_model_spec(cal.tip_angle_model, b)
    if cal.pa is None:
        raise ValueError("Calibration is missing tip_angle_coeffs.")
    return poly_eval(cal.pa, b)


def predict_tip_offset_xyz(cal: Calibration, b: float, c_deg: float) -> np.ndarray:
    r = float(eval_r(cal, b))
    z = float(eval_z(cal, b))
    y_off = float(eval_offplane_y(cal, b))
    c = math.radians(float(c_deg))
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def stage_xyz_for_tip(cal: Calibration, tip_xyz: np.ndarray, b: float, c_deg: float) -> np.ndarray:
    return np.asarray(tip_xyz, dtype=float) - predict_tip_offset_xyz(cal, b, c_deg)


def solve_b_for_target_tip_angle(cal: Calibration, target_angle_deg: float, search_samples: int = DEFAULT_BC_SOLVE_SAMPLES) -> float:
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


# ---------------- Glyph construction ----------------
def build_pi_sign_branches(
    x_left: float,
    x_right: float,
    y: float,
    z_bottom: float,
    z_top: float,
    vertical_overextend_mm: float,
    points_per_mm: float,
) -> List[Dict[str, Any]]:
    x_left = float(x_left)
    x_right = float(x_right)
    y = float(y)
    z_bottom = float(z_bottom)
    z_top = float(z_top)
    vertical_overextend_mm = float(vertical_overextend_mm)
    points_per_mm = float(points_per_mm)

    if x_right <= x_left:
        raise ValueError("--x-right must be greater than --x-left")
    if z_top <= z_bottom:
        raise ValueError("--z-top must be greater than --z-bottom")
    if vertical_overextend_mm < 0.0:
        raise ValueError("--vertical-overextend-mm must be >= 0")
    if points_per_mm <= 0.0:
        raise ValueError("--points-per-mm must be > 0")

    left_leg = scale_template_to_world(LEFT_LEG_TEMPLATE, x_left, x_right, z_bottom, z_top, y)
    top_bar = scale_template_to_world(TOP_BAR_TEMPLATE, x_left, x_right, z_bottom, z_top, y)
    right_leg = scale_template_to_world(RIGHT_LEG_TEMPLATE, x_left, x_right, z_bottom, z_top, y)

    # Apply the requested 1 mm over-extension on the side of the horizontal bar.
    left_leg_ext, left_node_distance = append_extension(left_leg, vertical_overextend_mm)
    right_leg_ext, right_node_distance = prepend_extension(right_leg, vertical_overextend_mm)

    left_leg_ext = deduplicate_polyline_points(smooth_polyline(resample_polyline(left_leg_ext, points_per_mm), 5))
    top_bar = deduplicate_polyline_points(smooth_polyline(resample_polyline(top_bar, points_per_mm), 5))
    right_leg_ext = deduplicate_polyline_points(smooth_polyline(resample_polyline(right_leg_ext, points_per_mm), 5))

    left_node_world = scale_template_to_world(LEFT_NODE_NORM.reshape(1, 2), x_left, x_right, z_bottom, z_top, y)[0]
    right_node_world = scale_template_to_world(RIGHT_NODE_NORM.reshape(1, 2), x_left, x_right, z_bottom, z_top, y)[0]

    top_bar_slow_distances = [
        nearest_arclength_to_world_point(top_bar, left_node_world),
        nearest_arclength_to_world_point(top_bar, right_node_world),
    ]

    branches: List[Dict[str, Any]] = [
        {
            "name": "left_leg",
            "points": left_leg_ext,
            "slow_distances_mm": [left_node_distance],
        },
        {
            "name": "top_bar",
            "points": top_bar,
            "slow_distances_mm": top_bar_slow_distances,
        },
        {
            "name": "right_leg",
            "points": right_leg_ext,
            "slow_distances_mm": [right_node_distance],
        },
    ]
    return branches


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
        extrusion_per_mm: float,
        pressure_offset_mm: float,
        pressure_advance_feed: float,
        pressure_retract_feed: float,
        preflow_dwell_ms: int,
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
        self.extrusion_per_mm = float(extrusion_per_mm)
        self.pressure_offset_mm = float(pressure_offset_mm)
        self.pressure_advance_feed = float(pressure_advance_feed)
        self.pressure_retract_feed = float(pressure_retract_feed)
        self.preflow_dwell_ms = int(preflow_dwell_ms)

        self.u_material_abs = 0.0
        self.pressure_charged = False
        self.cur_stage_xyz: Optional[np.ndarray] = None
        self.cur_tip_xyz: Optional[np.ndarray] = None
        self.cur_b: float = 0.0
        self.cur_c: float = self.c_deg
        self.last_tip_tangent: Optional[np.ndarray] = None
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

    @property
    def u_axis(self) -> str:
        return self.cal.u_axis if self.cal is not None else "U"

    def clamp_stage(self, p_stage: np.ndarray, context: str) -> np.ndarray:
        x = float(np.clip(p_stage[0], self.bbox["x_min"], self.bbox["x_max"]))
        y = float(np.clip(p_stage[1], self.bbox["y_min"], self.bbox["y_max"]))
        z = float(np.clip(p_stage[2], self.bbox["z_min"], self.bbox["z_max"]))
        if abs(x - float(p_stage[0])) > 1e-12 or abs(y - float(p_stage[1])) > 1e-12 or abs(z - float(p_stage[2])) > 1e-12:
            self.warnings.append(f"WARNING: {context} stage point clamped to bbox.")
        return np.array([x, y, z], dtype=float)

    def u_cmd_actual(self) -> float:
        return self.u_material_abs + (self.pressure_offset_mm if self.pressure_charged else 0.0)

    def bc_for_tangent(self, tangent: np.ndarray) -> Tuple[float, float]:
        if self.orientation_mode == "fixed":
            return float(self.fixed_b), float(self.fixed_c)

        target_b = desired_physical_b_angle_from_tangent(tangent) + float(self.b_angle_bias_deg)
        target_b = float(np.clip(target_b, 0.0, 180.0))

        if self.write_mode == "calibrated":
            assert self.cal is not None
            b = solve_b_for_target_tip_angle(self.cal, target_b, search_samples=self.bc_solve_samples)
        else:
            b = target_b

        return float(b), float(self.c_deg)

    def tip_to_stage(self, p_tip: np.ndarray, tangent: Optional[np.ndarray]) -> Tuple[np.ndarray, float, float]:
        tangent_arr = np.array([1.0, 0.0, 0.0], dtype=float) if tangent is None else normalize(np.asarray(tangent, dtype=float))
        b, c = self.bc_for_tangent(tangent_arr)

        if self.write_mode == "calibrated":
            assert self.cal is not None
            p_stage = stage_xyz_for_tip(self.cal, np.asarray(p_tip, dtype=float), b, c)
        else:
            p_stage = np.asarray(p_tip, dtype=float)

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
        axes: List[Tuple[str, float]] = [
            (self.x_axis, float(p_stage[0])),
            (self.y_axis, float(p_stage[1])),
            (self.z_axis, float(p_stage[2])),
            (self.b_axis, float(b)),
            (self.c_axis, float(c)),
        ]
        if u_value is not None:
            axes.append((self.u_axis, float(u_value)))
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

    def move_to_tip(self, p_tip: np.ndarray, tangent: Optional[np.ndarray], feed: float, comment: Optional[str] = None) -> None:
        p_stage, b, c = self.tip_to_stage(np.asarray(p_tip, dtype=float), tangent=tangent)
        self.write_move(p_stage, b, c, feed, comment=comment)
        self.cur_tip_xyz = np.asarray(p_tip, dtype=float).copy()
        self.last_tip_tangent = None if tangent is None else np.asarray(tangent, dtype=float).copy()

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
    ) -> None:
        start_tip = np.asarray(start_tip, dtype=float)
        start_tangent = normalize(np.asarray(start_tangent, dtype=float))
        start_side = side_vector_from_tangent(start_tangent, fallback=self.last_tip_tangent)

        if self.cur_tip_xyz is not None and self.last_tip_tangent is not None:
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
            )

        far_tip = start_tip - start_side * float(approach_side_mm) + np.array([0.0, 0.0, float(travel_lift_z)], dtype=float)
        near_tip = start_tip - start_side * (0.5 * float(approach_side_mm)) + np.array([0.0, 0.0, 0.5 * float(travel_lift_z)], dtype=float)
        self.move_to_tip(far_tip, tangent=start_tangent, feed=self.travel_feed, comment="travel above and outside the glyph")
        self.move_to_tip(near_tip, tangent=start_tangent, feed=self.approach_feed, comment="approach toward the next start")
        self.move_to_tip(start_tip, tangent=start_tangent, feed=self.fine_approach_feed, comment="fine approach to the next start")

    def approach_right_vertical_leg(
        self,
        start_tip: np.ndarray,
        start_tangent: np.ndarray,
        travel_lift_z: float,
        approach_side_mm: float,
        retreat_side_mm: float,
    ) -> None:
        if self.cur_tip_xyz is None or self.last_tip_tangent is None:
            self.approach_start(
                start_tip=start_tip,
                start_tangent=start_tangent,
                travel_lift_z=travel_lift_z,
                approach_side_mm=approach_side_mm,
                retreat_side_mm=retreat_side_mm,
            )
            return

        start_tip = np.asarray(start_tip, dtype=float)
        start_tangent = normalize(np.asarray(start_tangent, dtype=float))

        lowered_tip = np.asarray(self.cur_tip_xyz, dtype=float) + np.array([0.0, 0.0, -float(travel_lift_z)], dtype=float)
        self.move_to_tip(
            lowered_tip,
            tangent=self.last_tip_tangent,
            feed=self.approach_feed,
            comment="drop in Z before curling B for right vertical leg",
        )
        self.move_to_tip(
            lowered_tip,
            tangent=start_tangent,
            feed=self.approach_feed,
            comment="curl B for right vertical leg while below the start",
        )

        approach_tip = start_tip + np.array([float(approach_side_mm), 0.0, -0.5 * float(travel_lift_z)], dtype=float)
        self.move_to_tip(
            approach_tip,
            tangent=start_tangent,
            feed=self.approach_feed,
            comment="move to the +X/-Z side of the right vertical start",
        )
        self.move_to_tip(
            start_tip,
            tangent=start_tangent,
            feed=self.fine_approach_feed,
            comment="move -X and +Z into the right vertical start",
        )

    def print_polyline(
        self,
        points: np.ndarray,
        tangents: np.ndarray,
        path_name: str,
        slow_distances_mm: Optional[Sequence[float]] = None,
        slow_radius_mm: float = DEFAULT_SLOW_ZONE_RADIUS_MM,
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

        self.pressure_preload_before_print()

        slow_list = [float(v) for v in (slow_distances_mm or [])]
        last_tip = np.asarray(points[0], dtype=float).copy()
        self.cur_tip_xyz = last_tip.copy()
        self.last_tip_tangent = np.asarray(tangents[0], dtype=float).copy()

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
                p_stage, b, c = self.tip_to_stage(p_tip, tangent=tangent)

                arc_here = cumulative + u * edge_len
                feed = self.print_feed
                for slow_d in slow_list:
                    if abs(arc_here - slow_d) <= float(slow_radius_mm):
                        feed = min(feed, self.node_connection_feed)
                        break

                self.write_move(p_stage, b, c, feed, comment=None, u_value=None)
                self.cur_tip_xyz = p_tip.copy()
                self.last_tip_tangent = tangent.copy()
                last_tip = p_tip.copy()

            cumulative += edge_len

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
        )
        lift_tip = x_clear_tip + np.array([0.0, 0.0, float(travel_lift_z)], dtype=float)
        self.move_to_tip(
            lift_tip,
            tangent=self.last_tip_tangent,
            feed=self.approach_feed,
            comment="+Z travel out after +X clearance",
        )


# ---------------- Top-level generation ----------------
def write_pi_sign_gcode(
    out: str,
    calibration: Optional[str],
    write_mode: str,
    orientation_mode: str,
    y_offplane_sign: float,
    x_left: float,
    x_right: float,
    y: float,
    z_bottom: float,
    z_top: float,
    vertical_overextend_mm: float,
    points_per_mm: float,
    tangent_smooth_window: int,
    slow_zone_radius_mm: float,
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
    emit_extrusion: bool,
    extrusion_per_mm: float,
    pressure_offset_mm: float,
    pressure_advance_feed: float,
    pressure_retract_feed: float,
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
    if write_mode == "calibrated" and not calibration:
        raise ValueError("--calibration is required when --write-mode calibrated")

    cal: Optional[Calibration]
    if write_mode == "calibrated":
        cal = load_calibration(str(calibration), requested_offplane_fit_model=DEFAULT_Y_OFFPLANE_FIT_MODEL)
        cal.offplane_y_sign = float(y_offplane_sign)
    else:
        cal = None

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
        points_per_mm=points_per_mm,
    )

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
        extrusion_per_mm=extrusion_per_mm,
        pressure_offset_mm=pressure_offset_mm,
        pressure_advance_feed=pressure_advance_feed,
        pressure_retract_feed=pressure_retract_feed,
        preflow_dwell_ms=preflow_dwell_ms,
    )

    with out_path.open("w", encoding="utf-8") as fh:
        writer.f = fh
        fh.write("; Rounded pi sign in the XZ plane, matched to the supplied sign image\n")
        fh.write("; Generated by pi_sign_image_style_xz_generator.py\n")
        fh.write(f"; write_mode={write_mode} orientation_mode={orientation_mode}\n")
        fh.write(
            "; glyph_box "
            f"x=[{float(x_left):.3f},{float(x_right):.3f}] "
            f"y={float(y):.3f} "
            f"z=[{float(z_bottom):.3f},{float(z_top):.3f}]\n"
        )
        fh.write(
            "; shape_parameters "
            f"vertical_overextend_mm={float(vertical_overextend_mm):.3f} "
            f"points_per_mm={float(points_per_mm):.3f} "
            f"slow_zone_radius_mm={float(slow_zone_radius_mm):.3f}\n"
        )
        fh.write("G21\n")
        fh.write("G90\n")
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

        for branch in branches:
            pts = np.asarray(branch["points"], dtype=float)
            tangents = build_tangents_for_points(pts, smooth_window=tangent_smooth_window)
            if str(branch["name"]) == "right_leg":
                writer.approach_right_vertical_leg(
                    start_tip=pts[0],
                    start_tangent=tangents[0],
                    travel_lift_z=travel_lift_z,
                    approach_side_mm=approach_side_mm,
                    retreat_side_mm=retreat_side_mm,
                )
            else:
                writer.approach_start(
                    start_tip=pts[0],
                    start_tangent=tangents[0],
                    travel_lift_z=travel_lift_z,
                    approach_side_mm=approach_side_mm,
                    retreat_side_mm=retreat_side_mm,
                )
            writer.print_polyline(
                pts,
                tangents,
                path_name=str(branch["name"]),
                slow_distances_mm=branch.get("slow_distances_mm", []),
                slow_radius_mm=slow_zone_radius_mm,
            )

        writer.finish(travel_lift_z=travel_lift_z, retreat_side_mm=retreat_side_mm)
        fh.write("; End of file\n")

    all_points = np.vstack([np.asarray(b["points"], dtype=float) for b in branches])
    all_tangents = np.vstack([build_tangents_for_points(np.asarray(b["points"], dtype=float), smooth_window=tangent_smooth_window) for b in branches])
    tip_b = np.array([desired_physical_b_angle_from_tangent(t) for t in all_tangents], dtype=float)
    if orientation_mode == "fixed":
        b_used = np.full_like(tip_b, float(fixed_b), dtype=float)
    elif write_mode == "calibrated":
        assert cal is not None
        b_used = np.array(
            [
                solve_b_for_target_tip_angle(cal, float(np.clip(bv + b_angle_bias_deg, 0.0, 180.0)), search_samples=bc_solve_samples)
                for bv in tip_b
            ],
            dtype=float,
        )
    else:
        b_used = np.clip(tip_b + float(b_angle_bias_deg), 0.0, 180.0)

    summary = {
        "out": str(out_path),
        "write_mode": write_mode,
        "orientation_mode": orientation_mode,
        "tip_x_range": (float(np.min(all_points[:, 0])), float(np.max(all_points[:, 0]))),
        "tip_y_range": (float(np.min(all_points[:, 1])), float(np.max(all_points[:, 1]))),
        "tip_z_range": (float(np.min(all_points[:, 2])), float(np.max(all_points[:, 2]))),
        "tip_b_target_range_deg": (float(np.min(tip_b)), float(np.max(tip_b))),
        "b_command_range_deg": (float(np.min(b_used)), float(np.max(b_used))),
        "c_command_deg": float(c_deg if orientation_mode == "tangent" else fixed_c),
        "branch_order": [str(b["name"]) for b in branches],
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
    ap.add_argument("--y-offplane-sign", type=float, default=DEFAULT_Y_OFFPLANE_SIGN, help="Multiplier applied to the calibration off-plane Y term in calibrated mode. Use -1 to flip the sign.")

    ap.add_argument("--x-left", type=float, default=DEFAULT_X_LEFT, help="Left edge of the overall glyph box.")
    ap.add_argument("--x-right", type=float, default=DEFAULT_X_RIGHT, help="Right edge of the overall glyph box.")
    ap.add_argument("--y", type=float, default=DEFAULT_Y)
    ap.add_argument("--z-bottom", type=float, default=DEFAULT_Z_BOTTOM, help="Bottom edge of the overall glyph box.")
    ap.add_argument("--z-top", type=float, default=DEFAULT_Z_TOP, help="Top edge of the overall glyph box.")

    ap.add_argument("--vertical-overextend-mm", type=float, default=DEFAULT_VERTICAL_OVEREXTEND_MM, help="How far the vertical legs extend past the top bar on the bar side.")
    ap.add_argument("--points-per-mm", type=float, default=DEFAULT_POINTS_PER_MM, help="Polyline resampling density.")
    ap.add_argument("--tangent-smooth-window", type=int, default=DEFAULT_TANGENT_SMOOTH_WINDOW)
    ap.add_argument("--slow-zone-radius-mm", type=float, default=DEFAULT_SLOW_ZONE_RADIUS_MM, help="Distance around each leg/bar node where print feed is reduced.")

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

    ap.add_argument("--emit-extrusion", dest="emit_extrusion", action="store_true", default=DEFAULT_EMIT_EXTRUSION)
    ap.add_argument("--no-emit-extrusion", dest="emit_extrusion", action="store_false")
    ap.add_argument("--extrusion-per-mm", type=float, default=DEFAULT_EXTRUSION_PER_MM)
    ap.add_argument("--pressure-offset-mm", type=float, default=DEFAULT_PRESSURE_OFFSET_MM)
    ap.add_argument("--pressure-advance-feed", type=float, default=DEFAULT_PRESSURE_ADVANCE_FEED)
    ap.add_argument("--pressure-retract-feed", type=float, default=DEFAULT_PRESSURE_RETRACT_FEED)
    ap.add_argument("--preflow-dwell-ms", type=int, default=DEFAULT_PREFLOW_DWELL_MS)

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
