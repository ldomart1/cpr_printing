
#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code for a stylized surf wave traced in the XZ plane.

Shape
-----
The generated tip path is a single continuous surf-wave stroke:

1) left horizontal lead-in
2) smooth concave backside rise
3) curl into the lip toward a *planned* XZ heading of 220 deg
   (equivalently -140 deg in the local unwrap used here), so the crest hooks
   back down inside like a breaking wave
4) smooth inside lip that goes back down while uncurling
5) right horizontal run-out on the same Z level as the left horizontal

Motion / orientation
--------------------
- Desired tip path lies in the XZ plane at constant Y.
- In tangent mode, B follows the local XZ-path tangent using the same convention
  as the reference generator:
      B = 0 deg   -> tip points straight up (+Z)
      B = 90 deg  -> tip is horizontal
      B = 180 deg -> tip points straight down (-Z)
- Default C is held at 180 deg so the tool stays in the XZ plane.
- In calibrated mode, the stage XYZ is solved from the desired tip XYZ using the
  calibration tip-offset model, so the *robot tip tracks the lip trajectory*
  even during the aggressive curl / lip-writing section.
- The lip geometry is prioritized over exact tangent matching: the path planner
  aims for a 220 deg lip heading in XZ, and the B solve uses as much curl as the
  calibration allows. If the calibration cannot realize the full tangent demand,
  the stage still follows the requested tip trajectory and B simply saturates at
  the nearest achievable value.
- Travel moves are routed through a clearance height above the full printed
  envelope, with start/end anchors outside the wave span, so travel does not
  cross the already printed path.

This script keeps the same calibration loading / tip-offset planning structure
as the uploaded Gaussian-wave reference generator, and adapts only the path
construction and safe travel strategy.
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
DEFAULT_OUT = "gcode_generation/surf_wave_xz.gcode"

# Tip-space placement
DEFAULT_X_START = 50.0
DEFAULT_X_TIP = 98.0
DEFAULT_X_END = 150.0
DEFAULT_Y = 52.0

DEFAULT_Z_LEFT = -170.0
DEFAULT_Z_CREST = -128.0
DEFAULT_Z_RIGHT = DEFAULT_Z_LEFT

DEFAULT_LEFT_FLAT_LEN = 10.0
DEFAULT_RIGHT_FLAT_LEN = 18.0

# Backside shaping
DEFAULT_BACKSIDE_HANDLE_START_FRAC = 0.35
DEFAULT_BACKSIDE_HANDLE_END_FRAC = 0.30

# Curl / lip shaping
DEFAULT_CURL_ARC_LEN = 18.0
DEFAULT_LIP_TARGET_HEADING_DEG = 220.0
DEFAULT_CURL_ANGLE_MARGIN_DEG = 2.0
DEFAULT_MIN_CURL_TIP_ANGLE_DEG = 120.0
DEFAULT_CARTESIAN_CURL_TIP_ANGLE_DEG = 165.0
DEFAULT_LIP_START_HANDLE_MM = 10.0
DEFAULT_LIP_END_HANDLE_MM = 10.0

# Sampling
DEFAULT_SAMPLE_STEP_MM = 0.50
DEFAULT_TANGENT_SMOOTH_WINDOW = 3
DEFAULT_CENTERLINE_SMOOTH_WINDOW = 0
DEFAULT_MIN_TANGENT_XY = 1e-9
DEFAULT_POINT_MERGE_TOL = 1e-9

# Orientation
DEFAULT_WRITE_MODE = "calibrated"
DEFAULT_ORIENTATION_MODE = "tangent"
DEFAULT_FIXED_B = 90.0
DEFAULT_FIXED_C = 180.0
DEFAULT_C_DEG = 180.0
DEFAULT_B_ANGLE_BIAS_DEG = 0.0
DEFAULT_BC_SOLVE_SAMPLES = 5001
DEFAULT_Y_OFFPLANE_SIGN = 1.0

# Motion
DEFAULT_TRAVEL_FEED = 1000.0
DEFAULT_APPROACH_FEED = 400.0
DEFAULT_FINE_APPROACH_FEED = 80.0
DEFAULT_PRINT_FEED = 250.0
DEFAULT_TRAVEL_CLEARANCE_Z = 8.0
DEFAULT_TRAVEL_OUTSIDE_X_MARGIN = 10.0
DEFAULT_APPROACH_SIDE_MM = 4.0
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
    active_phase: str = "pull"
    offplane_y_sign: float = 1.0


@dataclass
class SegmentSpan:
    name: str
    start_idx: int
    end_idx: int
    lip_tracking: bool = False


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


def build_tangents_for_points(
    points: np.ndarray,
    smooth_window: int,
    centerline_smooth_window: int = DEFAULT_CENTERLINE_SMOOTH_WINDOW,
) -> np.ndarray:
    tangent_points = smooth_centerline_points(points, window=centerline_smooth_window)
    tangents = np.zeros_like(tangent_points)
    for i in range(len(tangent_points)):
        tangents[i] = tangent_for_index(tangent_points, i, smooth_window=max(1, int(smooth_window)))
    return tangents


def polyline_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) < 2:
        return 0.0
    return float(np.sum(np.linalg.norm(pts[1:] - pts[:-1], axis=1)))


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


def unwrap_angle_deg_near(target_deg: float, reference_deg: float) -> float:
    target = float(target_deg)
    ref = float(reference_deg)
    while target - ref > 180.0:
        target -= 360.0
    while target - ref < -180.0:
        target += 360.0
    return float(target)


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


def line_points(p0: np.ndarray, p1: np.ndarray, step_mm: float) -> np.ndarray:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    dist = float(np.linalg.norm(p1 - p0))
    n = max(2, int(math.ceil(max(dist, 1e-9) / max(step_mm, 1e-6))) + 1)
    return np.linspace(p0, p1, n)


def cubic_bezier_eval(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float).reshape(-1, 1)
    omt = 1.0 - t
    return (
        (omt ** 3) * p0.reshape(1, 3)
        + (3.0 * omt * omt * t) * p1.reshape(1, 3)
        + (3.0 * omt * t * t) * p2.reshape(1, 3)
        + (t ** 3) * p3.reshape(1, 3)
    )


def cubic_bezier_points(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray, step_mm: float) -> np.ndarray:
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    p2 = np.asarray(p2, dtype=float)
    p3 = np.asarray(p3, dtype=float)
    approx_len = (
        float(np.linalg.norm(p1 - p0))
        + float(np.linalg.norm(p2 - p1))
        + float(np.linalg.norm(p3 - p2))
    )
    n = max(4, int(math.ceil(max(approx_len, 1e-9) / max(step_mm, 1e-6))) + 1)
    tt = np.linspace(0.0, 1.0, n)
    return cubic_bezier_eval(p0, p1, p2, p3, tt)


def cosine_ease(s: np.ndarray) -> np.ndarray:
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    return 0.5 - 0.5 * np.cos(np.pi * s)


def integrate_angle_segment(
    start_point: np.ndarray,
    start_tangent_deg: float,
    end_tangent_deg: float,
    arc_len_mm: float,
    step_mm: float,
) -> np.ndarray:
    """
    Integrate a planar XZ segment whose tangent angle changes smoothly from
    start_tangent_deg to end_tangent_deg.

    Tangent angle here is measured in the XZ plane from +X toward +Z.
    So:
      0 deg   -> +X
      +90 deg -> +Z
      -90 deg -> -Z
    """
    if arc_len_mm <= 0.0:
        return np.asarray(start_point, dtype=float).reshape(1, 3)

    n = max(4, int(math.ceil(float(arc_len_mm) / max(step_mm, 1e-6))) + 1)
    s = np.linspace(0.0, 1.0, n)
    w = cosine_ease(s)
    theta_deg = float(start_tangent_deg) + (float(end_tangent_deg) - float(start_tangent_deg)) * w

    pts = np.zeros((n, 3), dtype=float)
    pts[0] = np.asarray(start_point, dtype=float)

    ds = float(arc_len_mm) / float(n - 1)
    for i in range(1, n):
        th_mid = math.radians(0.5 * (theta_deg[i - 1] + theta_deg[i]))
        dx = ds * math.cos(th_mid)
        dz = ds * math.sin(th_mid)
        pts[i] = pts[i - 1] + np.array([dx, 0.0, dz], dtype=float)

    return pts


def append_segment(parts: List[np.ndarray], spans: List[SegmentSpan], name: str, pts: np.ndarray, lip_tracking: bool = False) -> None:
    pts = np.asarray(pts, dtype=float)
    if len(pts) == 0:
        return

    if not parts:
        start_idx = 0
        parts.append(pts)
        end_idx = len(pts) - 1
    else:
        if float(np.linalg.norm(parts[-1][-1] - pts[0])) <= DEFAULT_POINT_MERGE_TOL:
            pts_use = pts[1:]
            start_idx = sum(len(p) for p in parts) - 1
        else:
            pts_use = pts
            start_idx = sum(len(p) for p in parts)
        if len(pts_use) == 0:
            end_idx = start_idx
        else:
            parts.append(pts_use)
            end_idx = start_idx + len(pts_use) - 1

    spans.append(SegmentSpan(name=name, start_idx=start_idx, end_idx=end_idx, lip_tracking=lip_tracking))


def _orient2d_xz(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    return float((b[0] - a[0]) * (c[2] - a[2]) - (b[2] - a[2]) * (c[0] - a[0]))


def _segments_intersect_xz(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray, tol: float = 1e-9) -> bool:
    def on_segment(p: np.ndarray, q: np.ndarray, r: np.ndarray) -> bool:
        return (
            min(p[0], q[0]) - tol <= r[0] <= max(p[0], q[0]) + tol
            and min(p[2], q[2]) - tol <= r[2] <= max(p[2], q[2]) + tol
        )

    o1 = _orient2d_xz(a, b, c)
    o2 = _orient2d_xz(a, b, d)
    o3 = _orient2d_xz(c, d, a)
    o4 = _orient2d_xz(c, d, b)

    if ((o1 > tol and o2 < -tol) or (o1 < -tol and o2 > tol)) and ((o3 > tol and o4 < -tol) or (o3 < -tol and o4 > tol)):
        return True

    if abs(o1) <= tol and on_segment(a, b, c):
        return True
    if abs(o2) <= tol and on_segment(a, b, d):
        return True
    if abs(o3) <= tol and on_segment(c, d, a):
        return True
    if abs(o4) <= tol and on_segment(c, d, b):
        return True
    return False


def polyline_self_intersections_xz(points: np.ndarray, tol: float = 1e-9) -> List[Tuple[int, int]]:
    pts = np.asarray(points, dtype=float)
    hits: List[Tuple[int, int]] = []
    if len(pts) < 4:
        return hits

    for i in range(len(pts) - 1):
        a = pts[i]
        b = pts[i + 1]
        for j in range(i + 2, len(pts) - 1):
            if j == i or j == i + 1:
                continue
            if i == 0 and j == len(pts) - 2:
                continue
            c = pts[j]
            d = pts[j + 1]
            shared_endpoint = (
                np.linalg.norm(a - c) <= tol
                or np.linalg.norm(a - d) <= tol
                or np.linalg.norm(b - c) <= tol
                or np.linalg.norm(b - d) <= tol
            )
            if shared_endpoint:
                continue
            if _segments_intersect_xz(a, b, c, d, tol=tol):
                hits.append((i, j))
    return hits


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


def load_calibration(json_path: str) -> Calibration:
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
    y_off_selector = selected_offplane_fit_model or selected_fit_model
    y_off_model = _select_named_model(active_phase_models, "offplane_y", y_off_selector)
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


def max_available_tip_angle_deg(cal: Calibration, search_samples: int = DEFAULT_BC_SOLVE_SAMPLES) -> float:
    bb = np.linspace(float(cal.b_min), float(cal.b_max), int(max(101, search_samples)))
    aa = np.asarray(eval_tip_angle_deg(cal, bb), dtype=float)
    return float(np.max(aa))


# ---------------- Surf-wave path construction ----------------
def build_surf_wave_points(
    cal: Optional[Calibration],
    write_mode: str,
    x_start: float,
    x_tip: float,
    x_end: float,
    y: float,
    z_left: float,
    z_crest: float,
    z_right: float,
    left_flat_len: float,
    right_flat_len: float,
    backside_handle_start_frac: float,
    backside_handle_end_frac: float,
    curl_arc_len: float,
    lip_target_heading_deg: float,
    curl_angle_margin_deg: float,
    min_curl_tip_angle_deg: float,
    cartesian_curl_tip_angle_deg: float,
    lip_start_handle_mm: float,
    lip_end_handle_mm: float,
    sample_step_mm: float,
    bc_solve_samples: int,
) -> Tuple[np.ndarray, List[SegmentSpan], Dict[str, float]]:
    write_mode = str(write_mode).strip().lower()

    if not (x_start < x_tip < x_end):
        raise ValueError("Require x_start < x_tip < x_end.")
    if left_flat_len < 0.0 or right_flat_len < 0.0:
        raise ValueError("left_flat_len and right_flat_len must be >= 0.")
    if x_start + left_flat_len >= x_tip:
        raise ValueError("left_flat_len is too large for the requested x_start/x_tip.")
    if x_tip >= x_end - right_flat_len:
        raise ValueError("right_flat_len leaves no room for the lip to exit before x_end.")
    if z_crest <= z_left:
        raise ValueError("z_crest must be greater than z_left for the backside rise.")
    if sample_step_mm <= 0.0:
        raise ValueError("sample_step_mm must be > 0.")
    if curl_arc_len <= 0.0:
        raise ValueError("curl_arc_len must be > 0.")

    target_lip_heading_deg = float(lip_target_heading_deg)
    unwrapped_lip_heading_deg = unwrap_angle_deg_near(target_lip_heading_deg, 0.0)

    if write_mode == "calibrated":
        if cal is None:
            raise ValueError("Calibration is required in calibrated mode.")
        max_tip_angle_deg = max_available_tip_angle_deg(cal, search_samples=bc_solve_samples)
        effective_max_tip_angle_deg = min(max_tip_angle_deg - float(curl_angle_margin_deg), 179.0)
        effective_max_tip_angle_deg = max(effective_max_tip_angle_deg, float(min_curl_tip_angle_deg))
    else:
        max_tip_angle_deg = float(cartesian_curl_tip_angle_deg)
        effective_max_tip_angle_deg = float(cartesian_curl_tip_angle_deg)

    p0 = np.array([float(x_start), float(y), float(z_left)], dtype=float)
    p1 = np.array([float(x_start + left_flat_len), float(y), float(z_left)], dtype=float)
    p_tip = np.array([float(x_tip), float(y), float(z_crest)], dtype=float)
    p_tail = np.array([float(x_end), float(y), float(z_right)], dtype=float)
    p_exit = np.array([float(x_end - right_flat_len), float(y), float(z_right)], dtype=float)

    back_dx = float(p_tip[0] - p1[0])
    c1 = p1 + np.array([float(backside_handle_start_frac) * back_dx, 0.0, 0.0], dtype=float)
    c2 = p_tip - np.array([float(backside_handle_end_frac) * back_dx, 0.0, 0.0], dtype=float)

    parts: List[np.ndarray] = []
    spans: List[SegmentSpan] = []

    append_segment(parts, spans, "left_horizontal", line_points(p0, p1, sample_step_mm), lip_tracking=False)
    append_segment(parts, spans, "backside_rise", cubic_bezier_points(p1, c1, c2, p_tip, sample_step_mm), lip_tracking=False)

    curl_pts = integrate_angle_segment(
        start_point=p_tip,
        start_tangent_deg=0.0,
        end_tangent_deg=float(unwrapped_lip_heading_deg),
        arc_len_mm=float(curl_arc_len),
        step_mm=sample_step_mm,
    )
    append_segment(parts, spans, "curl_to_lip_heading", curl_pts, lip_tracking=True)

    p_curl_end = parts[-1][-1].copy()
    chord_len = float(np.linalg.norm(p_exit - p_curl_end))
    dx_to_exit = float(p_exit[0] - p_curl_end[0])
    if chord_len <= 1e-6 or dx_to_exit <= -0.25 * chord_len:
        raise ValueError(
            "The planned lip hook leaves no clean path to the exit. Increase x_end, "
            "reduce curl_arc_len, or choose a less aggressive lip heading."
        )

    lip_dir = np.array(
        [
            math.cos(math.radians(unwrapped_lip_heading_deg)),
            0.0,
            math.sin(math.radians(unwrapped_lip_heading_deg)),
        ],
        dtype=float,
    )

    best_lip_pts: Optional[np.ndarray] = None
    best_scale = None
    for scale in np.linspace(1.0, 0.20, 17):
        start_handle = min(float(lip_start_handle_mm) * float(scale), 0.48 * max(chord_len, 1e-9))
        end_handle = min(float(lip_end_handle_mm) * float(scale), 0.48 * max(chord_len, 1e-9))
        c3 = p_curl_end + start_handle * lip_dir
        c4 = p_exit - np.array([end_handle, 0.0, 0.0], dtype=float)
        lip_pts = cubic_bezier_points(p_curl_end, c3, c4, p_exit, sample_step_mm)

        test_pts = deduplicate_polyline_points(np.vstack(parts + [lip_pts, line_points(p_exit, p_tail, sample_step_mm)]))
        if not polyline_self_intersections_xz(test_pts):
            best_lip_pts = lip_pts
            best_scale = float(scale)
            break

    if best_lip_pts is None:
        raise ValueError(
            "Unable to build a non-self-intersecting inside lip for the requested lip heading. "
            "Increase x_end, reduce curl_arc_len, or reduce lip handle lengths."
        )

    append_segment(parts, spans, "inside_lip_uncurl", best_lip_pts, lip_tracking=True)
    append_segment(parts, spans, "right_horizontal", line_points(p_exit, p_tail, sample_step_mm), lip_tracking=False)

    pts = deduplicate_polyline_points(np.vstack(parts))
    intersections = polyline_self_intersections_xz(pts)
    if intersections:
        raise ValueError(
            "Generated surf-wave path self-intersects in XZ. Adjust curl_arc_len / lip handles / x_end."
        )

    tangents = build_tangents_for_points(pts, smooth_window=1)
    requested_tip_b_deg = np.array([desired_physical_b_angle_from_tangent(t) for t in tangents], dtype=float)

    info = {
        "lip_target_heading_deg": float(target_lip_heading_deg),
        "lip_unwrapped_heading_deg": float(unwrapped_lip_heading_deg),
        "requested_tip_b_range_deg_min": float(np.min(requested_tip_b_deg)),
        "requested_tip_b_range_deg_max": float(np.max(requested_tip_b_deg)),
        "max_available_tip_angle_deg": float(max_tip_angle_deg),
        "effective_max_tip_angle_deg": float(effective_max_tip_angle_deg),
        "path_len_mm": float(polyline_length(pts)),
        "lip_handle_scale_used": float(best_scale if best_scale is not None else 1.0),
    }
    return pts, spans, info


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
        if self.emit_extrusion and self.pressure_offset_mm > 0.0 and not self.pressure_charged:
            self.pressure_charged = True
            self.f.write("; pressure preload before print pass\n")
            self.f.write(f"G1 {self.u_axis}{self.u_cmd_actual():.3f} F{self.pressure_advance_feed:.0f}\n")
            if self.preflow_dwell_ms > 0:
                self.f.write(f"G4 P{self.preflow_dwell_ms}\n")

    def pressure_release_after_print(self) -> None:
        if self.emit_extrusion and self.pressure_offset_mm > 0.0 and self.pressure_charged:
            self.pressure_charged = False
            self.f.write("; pressure release after print pass\n")
            self.f.write(f"G1 {self.u_axis}{self.u_cmd_actual():.3f} F{self.pressure_retract_feed:.0f}\n")

    def safe_approach_start(
        self,
        start_tip: np.ndarray,
        start_tangent: np.ndarray,
        envelope: Dict[str, float],
        travel_clearance_z: float,
        travel_outside_x_margin: float,
        approach_side_mm: float,
    ) -> None:
        start_tip = np.asarray(start_tip, dtype=float)
        start_tangent = normalize(np.asarray(start_tangent, dtype=float))
        side = side_vector_from_tangent(start_tangent, fallback=self.last_tip_tangent)

        clearance_z = float(envelope["z_max"] + travel_clearance_z)
        left_anchor = np.array([float(envelope["x_min"] - travel_outside_x_margin), start_tip[1], clearance_z], dtype=float)
        side_air = start_tip - side * float(approach_side_mm)
        side_air[2] = clearance_z
        side_near = start_tip - side * (0.5 * float(approach_side_mm))

        self.move_to_tip(left_anchor, tangent=start_tangent, feed=self.travel_feed, comment="safe travel anchor left of printed envelope")
        self.move_to_tip(side_air, tangent=start_tangent, feed=self.travel_feed, comment="approach above start from outside; does not cross printed path")
        self.move_to_tip(side_near, tangent=start_tangent, feed=self.approach_feed, comment="descend beside start from outside")
        self.move_to_tip(start_tip, tangent=start_tangent, feed=self.fine_approach_feed, comment="fine approach to start")

    def safe_retreat_end(
        self,
        end_tip: np.ndarray,
        end_tangent: np.ndarray,
        envelope: Dict[str, float],
        travel_clearance_z: float,
        travel_outside_x_margin: float,
        approach_side_mm: float,
    ) -> None:
        end_tip = np.asarray(end_tip, dtype=float)
        end_tangent = normalize(np.asarray(end_tangent, dtype=float))
        side = side_vector_from_tangent(end_tangent, fallback=self.last_tip_tangent)

        clearance_z = float(envelope["z_max"] + travel_clearance_z)
        side_near = end_tip + side * (0.5 * float(approach_side_mm))
        side_air = end_tip + side * float(approach_side_mm)
        side_air[2] = clearance_z
        right_anchor = np.array([float(envelope["x_max"] + travel_outside_x_margin), end_tip[1], clearance_z], dtype=float)

        self.move_to_tip(side_near, tangent=end_tangent, feed=self.approach_feed, comment="retreat beside end before non-print travel")
        self.move_to_tip(side_air, tangent=end_tangent, feed=self.approach_feed, comment="lift above printed envelope")
        self.move_to_tip(right_anchor, tangent=end_tangent, feed=self.travel_feed, comment="safe travel anchor right of printed envelope")

    def print_polyline(self, points: np.ndarray, tangents: np.ndarray, spans: List[SegmentSpan]) -> None:
        if len(points) < 2:
            return

        boundary_comments: Dict[int, Tuple[str, bool]] = {}
        for s in spans:
            boundary_comments[int(s.start_idx)] = (s.name, s.lip_tracking)

        self.f.write(
            "; SURF_WAVE_WRITE_START "
            f"point_count={len(points)} "
            f"tip_start_x={float(points[0, 0]):.6f} tip_start_y={float(points[0, 1]):.6f} tip_start_z={float(points[0, 2]):.6f} "
            f"tip_end_x={float(points[-1, 0]):.6f} tip_end_y={float(points[-1, 1]):.6f} tip_end_z={float(points[-1, 2]):.6f} "
            "tip_angle_convention=0_posZ_90_horizontal_180_negZ\n"
        )

        self.pressure_preload_before_print()

        if 0 in boundary_comments:
            seg_name, lip_tracking = boundary_comments[0]
            if lip_tracking:
                self.f.write(f"; SEGMENT {seg_name} (lip-tracking enabled; stage solve follows the lip point)\n")
            else:
                self.f.write(f"; SEGMENT {seg_name}\n")

        last_tip = np.asarray(points[0], dtype=float).copy()
        self.cur_tip_xyz = last_tip.copy()
        self.last_tip_tangent = np.asarray(tangents[0], dtype=float).copy()

        for i in range(1, len(points)):
            if i in boundary_comments:
                seg_name, lip_tracking = boundary_comments[i]
                if lip_tracking:
                    self.f.write(f"; SEGMENT {seg_name} (lip-tracking enabled; stage solve follows the lip point)\n")
                else:
                    self.f.write(f"; SEGMENT {seg_name}\n")

            p0 = np.asarray(points[i - 1], dtype=float)
            p1 = np.asarray(points[i], dtype=float)
            t0 = np.asarray(tangents[i - 1], dtype=float)
            t1 = np.asarray(tangents[i], dtype=float)

            for s in range(1, self.edge_samples + 1):
                u = s / float(self.edge_samples)
                p_tip = p0 + u * (p1 - p0)
                tangent = normalize((1.0 - u) * t0 + u * t1)
                p_stage, b, c = self.tip_to_stage(p_tip, tangent=tangent)

                u_value = None
                if self.emit_extrusion:
                    tip_seg_len = float(np.linalg.norm(p_tip - last_tip))
                    self.u_material_abs += self.extrusion_per_mm * tip_seg_len
                    u_value = self.u_cmd_actual()

                self.write_move(p_stage, b, c, self.print_feed, comment=None, u_value=u_value)
                self.cur_tip_xyz = p_tip.copy()
                self.last_tip_tangent = tangent.copy()
                last_tip = p_tip.copy()

        self.pressure_release_after_print()
        self.f.write("; SURF_WAVE_WRITE_END\n")


# ---------------- Top-level generation ----------------
def write_surf_wave_gcode(
    out: str,
    calibration: Optional[str],
    write_mode: str,
    orientation_mode: str,
    y_offplane_sign: float,
    x_start: float,
    x_tip: float,
    x_end: float,
    y: float,
    z_left: float,
    z_crest: float,
    z_right: float,
    left_flat_len: float,
    right_flat_len: float,
    backside_handle_start_frac: float,
    backside_handle_end_frac: float,
    curl_arc_len: float,
    lip_target_heading_deg: float,
    curl_angle_margin_deg: float,
    min_curl_tip_angle_deg: float,
    cartesian_curl_tip_angle_deg: float,
    lip_start_handle_mm: float,
    lip_end_handle_mm: float,
    sample_step_mm: float,
    tangent_smooth_window: int,
    centerline_smooth_window: int,
    fixed_b: float,
    fixed_c: float,
    c_deg: float,
    b_angle_bias_deg: float,
    bc_solve_samples: int,
    travel_feed: float,
    approach_feed: float,
    fine_approach_feed: float,
    print_feed: float,
    travel_clearance_z: float,
    travel_outside_x_margin: float,
    approach_side_mm: float,
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
        cal = load_calibration(str(calibration))
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

    pts, spans, path_info = build_surf_wave_points(
        cal=cal,
        write_mode=write_mode,
        x_start=x_start,
        x_tip=x_tip,
        x_end=x_end,
        y=y,
        z_left=z_left,
        z_crest=z_crest,
        z_right=z_right,
        left_flat_len=left_flat_len,
        right_flat_len=right_flat_len,
        backside_handle_start_frac=backside_handle_start_frac,
        backside_handle_end_frac=backside_handle_end_frac,
        curl_arc_len=curl_arc_len,
        lip_target_heading_deg=lip_target_heading_deg,
        curl_angle_margin_deg=curl_angle_margin_deg,
        min_curl_tip_angle_deg=min_curl_tip_angle_deg,
        cartesian_curl_tip_angle_deg=cartesian_curl_tip_angle_deg,
        lip_start_handle_mm=lip_start_handle_mm,
        lip_end_handle_mm=lip_end_handle_mm,
        sample_step_mm=sample_step_mm,
        bc_solve_samples=bc_solve_samples,
    )
    tangents = build_tangents_for_points(
        pts,
        smooth_window=tangent_smooth_window,
        centerline_smooth_window=centerline_smooth_window,
    )

    envelope = {
        "x_min": float(np.min(pts[:, 0])),
        "x_max": float(np.max(pts[:, 0])),
        "z_min": float(np.min(pts[:, 2])),
        "z_max": float(np.max(pts[:, 2])),
    }

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

        fh.write("; Stylized surf-wave write in the XZ plane\n")
        fh.write("; Generated by surf_wave_xz_generator.py\n")
        fh.write(f"; write_mode={write_mode} orientation_mode={orientation_mode}\n")
        fh.write(f"; B-angle convention: 0=up, 90=horizontal, 180=down\n")
        fh.write(f"; C held constant at {float(c_deg):.3f} deg in tangent mode\n")
        fh.write(
            "; surf_wave_parameters "
            f"x_start={float(x_start):.3f} x_tip={float(x_tip):.3f} x_end={float(x_end):.3f} "
            f"y={float(y):.3f} "
            f"z_left={float(z_left):.3f} z_crest={float(z_crest):.3f} z_right={float(z_right):.3f} "
            f"left_flat_len={float(left_flat_len):.3f} right_flat_len={float(right_flat_len):.3f} "
            f"curl_arc_len={float(curl_arc_len):.3f} "
            f"lip_target_heading_deg={float(path_info['lip_target_heading_deg']):.3f} "
            f"lip_unwrapped_heading_deg={float(path_info['lip_unwrapped_heading_deg']):.3f} "
            f"max_available_tip_angle_deg={float(path_info['max_available_tip_angle_deg']):.3f}\n"
        )
        fh.write(
            "; safe_travel "
            f"clearance_z_above_envelope={float(travel_clearance_z):.3f} "
            f"outside_x_margin={float(travel_outside_x_margin):.3f}\n"
        )
        fh.write("G21\n")
        fh.write("G90\n")

        writer.safe_approach_start(
            start_tip=pts[0],
            start_tangent=tangents[0],
            envelope=envelope,
            travel_clearance_z=travel_clearance_z,
            travel_outside_x_margin=travel_outside_x_margin,
            approach_side_mm=approach_side_mm,
        )
        writer.print_polyline(pts, tangents, spans)
        writer.safe_retreat_end(
            end_tip=pts[-1],
            end_tangent=tangents[-1],
            envelope=envelope,
            travel_clearance_z=travel_clearance_z,
            travel_outside_x_margin=travel_outside_x_margin,
            approach_side_mm=approach_side_mm,
        )

        fh.write("; End of file\n")

    tip_b = np.array([desired_physical_b_angle_from_tangent(t) for t in tangents], dtype=float)
    if orientation_mode == "fixed":
        b_used = np.full_like(tip_b, float(fixed_b), dtype=float)
        c_used = np.full_like(tip_b, float(fixed_c), dtype=float)
    elif write_mode == "calibrated":
        assert cal is not None
        b_used = np.array(
            [
                solve_b_for_target_tip_angle(cal, float(np.clip(bv + b_angle_bias_deg, 0.0, 180.0)), search_samples=bc_solve_samples)
                for bv in tip_b
            ],
            dtype=float,
        )
        c_used = np.full_like(tip_b, float(c_deg), dtype=float)
    else:
        b_used = np.clip(tip_b + float(b_angle_bias_deg), 0.0, 180.0)
        c_used = np.full_like(tip_b, float(c_deg), dtype=float)

    summary = {
        "out": str(out_path),
        "write_mode": write_mode,
        "orientation_mode": orientation_mode,
        "tip_x_range": (float(np.min(pts[:, 0])), float(np.max(pts[:, 0]))),
        "tip_y_range": (float(np.min(pts[:, 1])), float(np.max(pts[:, 1]))),
        "tip_z_range": (float(np.min(pts[:, 2])), float(np.max(pts[:, 2]))),
        "tip_b_target_range_deg": (float(np.min(tip_b)), float(np.max(tip_b))),
        "b_command_range_deg": (float(np.min(b_used)), float(np.max(b_used))),
        "c_command_range_deg": (float(np.min(c_used)), float(np.max(c_used))),
        "lip_target_heading_deg": float(path_info["lip_target_heading_deg"]),
        "lip_unwrapped_heading_deg": float(path_info["lip_unwrapped_heading_deg"]),
        "requested_tip_b_range_deg": (
            float(path_info["requested_tip_b_range_deg_min"]),
            float(path_info["requested_tip_b_range_deg_max"]),
        ),
        "max_available_tip_angle_deg": float(path_info["max_available_tip_angle_deg"]),
        "effective_max_tip_angle_deg": float(path_info["effective_max_tip_angle_deg"]),
        "lip_handle_scale_used": float(path_info["lip_handle_scale_used"]),
        "path_len_mm": float(path_info["path_len_mm"]),
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
            "Generate calibrated or Cartesian G-code for a stylized surf wave in the XZ plane. "
            "The wave starts with a left horizontal run, rises on a smooth concave backside, "
            "then curls the lip toward a planned 220 deg XZ heading while the stage keeps the "
            "tip on that trajectory. If the calibration cannot realize the full tangent demand, "
            "the B solve uses the maximum curl available and the tip trajectory is still tracked. "
            "Travel moves are routed outside and above the printed envelope."
        )
    )
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code file.")
    ap.add_argument("--calibration", default=None, help="Calibration JSON. Required for --write-mode calibrated.")
    ap.add_argument("--write-mode", choices=["calibrated", "cartesian"], default=DEFAULT_WRITE_MODE)
    ap.add_argument("--orientation-mode", choices=["tangent", "fixed"], default=DEFAULT_ORIENTATION_MODE)
    ap.add_argument("--y-offplane-sign", type=float, default=DEFAULT_Y_OFFPLANE_SIGN, help="Multiplier applied to the calibration off-plane Y term in calibrated mode. Use -1 to flip the sign.")

    ap.add_argument("--x-start", type=float, default=DEFAULT_X_START)
    ap.add_argument("--x-tip", type=float, default=DEFAULT_X_TIP, help="Tip/crest X location where the curl begins.")
    ap.add_argument("--x-end", type=float, default=DEFAULT_X_END)
    ap.add_argument("--y", type=float, default=DEFAULT_Y)
    ap.add_argument("--z-left", type=float, default=DEFAULT_Z_LEFT, help="Z level of the left horizontal segment.")
    ap.add_argument("--z-crest", type=float, default=DEFAULT_Z_CREST, help="Z level of the crest / lip point.")
    ap.add_argument("--z-right", type=float, default=DEFAULT_Z_RIGHT, help="Z level of the right horizontal segment after the lip. Defaults to the same level as --z-left.")
    ap.add_argument("--left-flat-len", type=float, default=DEFAULT_LEFT_FLAT_LEN, help="Horizontal lead-in length on the left.")
    ap.add_argument("--right-flat-len", type=float, default=DEFAULT_RIGHT_FLAT_LEN, help="Horizontal run-out length on the right.")
    ap.add_argument("--backside-handle-start-frac", type=float, default=DEFAULT_BACKSIDE_HANDLE_START_FRAC, help="Fraction of backside span used by the first cubic handle.")
    ap.add_argument("--backside-handle-end-frac", type=float, default=DEFAULT_BACKSIDE_HANDLE_END_FRAC, help="Fraction of backside span used by the second cubic handle.")
    ap.add_argument("--curl-arc-len", type=float, default=DEFAULT_CURL_ARC_LEN, help="Arc length used to curl from the crest into the lip hook.")
    ap.add_argument("--lip-target-heading-deg", type=float, default=DEFAULT_LIP_TARGET_HEADING_DEG, help="Preferred full XZ heading of the lip curl. Default 220 deg. Geometry is prioritized; the calibration-limited B solve follows as far as it can.")
    ap.add_argument("--curl-angle-margin-deg", type=float, default=DEFAULT_CURL_ANGLE_MARGIN_DEG, help="Safety margin subtracted when reporting the effective calibration-limited curl capability.")
    ap.add_argument("--min-curl-tip-angle-deg", type=float, default=DEFAULT_MIN_CURL_TIP_ANGLE_DEG, help="Lower bound for the physically requested curl tip angle in calibrated mode.")
    ap.add_argument("--cartesian-curl-tip-angle-deg", type=float, default=DEFAULT_CARTESIAN_CURL_TIP_ANGLE_DEG, help="Fallback max curl tip angle used when --write-mode cartesian.")
    ap.add_argument("--lip-start-handle-mm", type=float, default=DEFAULT_LIP_START_HANDLE_MM, help="Bezier handle length at the start of the inside lip.")
    ap.add_argument("--lip-end-handle-mm", type=float, default=DEFAULT_LIP_END_HANDLE_MM, help="Bezier handle length at the end of the inside lip.")
    ap.add_argument("--sample-step-mm", type=float, default=DEFAULT_SAMPLE_STEP_MM, help="Nominal sample spacing along the planned tip path.")

    ap.add_argument("--tangent-smooth-window", type=int, default=DEFAULT_TANGENT_SMOOTH_WINDOW)
    ap.add_argument("--centerline-smooth-window", type=int, default=DEFAULT_CENTERLINE_SMOOTH_WINDOW)

    ap.add_argument("--fixed-b", type=float, default=DEFAULT_FIXED_B, help="Used only when --orientation-mode fixed.")
    ap.add_argument("--fixed-c", type=float, default=DEFAULT_FIXED_C, help="Used only when --orientation-mode fixed.")
    ap.add_argument("--c-deg", type=float, default=DEFAULT_C_DEG, help="Constant C angle used in tangent mode. Default is 180 deg for an XZ-plane write.")
    ap.add_argument("--b-angle-bias-deg", type=float, default=DEFAULT_B_ANGLE_BIAS_DEG, help="Bias added to the tangent-derived B target before solving / emitting.")
    ap.add_argument("--bc-solve-samples", type=int, default=DEFAULT_BC_SOLVE_SAMPLES)

    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED)
    ap.add_argument("--approach-feed", type=float, default=DEFAULT_APPROACH_FEED)
    ap.add_argument("--fine-approach-feed", type=float, default=DEFAULT_FINE_APPROACH_FEED)
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED)
    ap.add_argument("--travel-clearance-z", type=float, default=DEFAULT_TRAVEL_CLEARANCE_Z, help="Tip-space clearance above the full printed envelope for non-print travel.")
    ap.add_argument("--travel-outside-x-margin", type=float, default=DEFAULT_TRAVEL_OUTSIDE_X_MARGIN, help="Tip-space X margin outside the printed envelope used for safe travel anchors.")
    ap.add_argument("--approach-side-mm", type=float, default=DEFAULT_APPROACH_SIDE_MM)
    ap.add_argument("--edge-samples", type=int, default=DEFAULT_EDGE_SAMPLES, help="Subdivide each polyline segment into this many printed G1 moves.")

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
    summary = write_surf_wave_gcode(**vars(args))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
