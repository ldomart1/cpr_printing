#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code for two calibrated spherical-style structures
using the full 3D calibration model with corrected off-plane sign.

Structure 1:
  - XZ circle first
  - optional YZ circle second
  - XY circle last
  - if YZ is skipped, XY starts at the current tip position after XZ

Structure 2 (top-down only, different sphere center):
  - fixed B at 0 deg
  - bottom quarter arcs first
  - then XY circle
  - then top quarter arcs
  - travel between edges by moving first up in Z to the highest printed tip Z
    so far, then in XY, then back down in Z to the next start point

Corrected off-plane sign convention:
  local transverse vector = [ r(B), -offplane_y(B) ]

World offset:
  x_off = r(B) * cos(C) + offplane_y(B) * sin(C)
  y_off = r(B) * sin(C) - offplane_y(B) * cos(C)
  z_off = z(B)

Exact tip tracking:
  stage_xyz = tip_xyz_desired - offset_xyz(B, C)
"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.optimize import brentq  # type: ignore
except Exception:
    def brentq(f, a, b, maxiter=100, xtol=1e-10, rtol=4*np.finfo(float).eps):
        fa = f(a)
        fb = f(b)
        if np.sign(fa) == np.sign(fb):
            raise ValueError("Root not bracketed.")
        x0, x1 = float(a), float(b)
        f0, f1 = float(fa), float(fb)
        x2 = 0.5 * (x0 + x1)
        for _ in range(int(maxiter)):
            if abs(f1 - f0) > 1e-14:
                x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            else:
                x2 = 0.5 * (x0 + x1)
            if not (min(x0, x1) <= x2 <= max(x0, x1)):
                x2 = 0.5 * (x0 + x1)
            f2 = float(f(x2))
            if abs(f2) < max(xtol, rtol * max(1.0, abs(x2))):
                return float(x2)
            if np.sign(f2) == np.sign(f0):
                x0, f0 = x2, f2
            else:
                x1, f1 = x2, f2
        return float(x2)


# ---------------- Defaults ----------------

DEFAULT_OUT = "circle_xyz_calibrated_final.gcode"

# Structure 1 center
DEFAULT_CENTER_X = 60.0
DEFAULT_CENTER_Y = 20.0
DEFAULT_CENTER_Z = -140.0

# Structure 2 center (different sphere center)
DEFAULT_TOPDOWN_CENTER_X = 60.0
DEFAULT_TOPDOWN_CENTER_Y = 70.0
DEFAULT_TOPDOWN_CENTER_Z = -140.0

DEFAULT_RADIUS = 20.0
DEFAULT_TOPDOWN_RADIUS = 20.0

DEFAULT_SAMPLES_PER_CIRCLE = 360
DEFAULT_SAMPLES_PER_QUARTER_ARC = 90
DEFAULT_RADIUS_SAFETY_MARGIN = 0.5

DEFAULT_MACHINE_START_X = 100.0
DEFAULT_MACHINE_START_Y = 60.0
DEFAULT_MACHINE_START_Z = -20.0
DEFAULT_MACHINE_START_B = 0.0
DEFAULT_MACHINE_START_C = 0.0

DEFAULT_MACHINE_END_X = 100.0
DEFAULT_MACHINE_END_Y = 60.0
DEFAULT_MACHINE_END_Z = -20.0
DEFAULT_MACHINE_END_B = 0.0
DEFAULT_MACHINE_END_C = 0.0
DEFAULT_SAFE_APPROACH_Z = 0.0

DEFAULT_TRAVEL_FEED = 1200.0
DEFAULT_PRINT_FEED = 250.0
DEFAULT_BC_HOLD_FEED = 300.0

DEFAULT_EXTRUSION_PER_MM = 0.0015
DEFAULT_PRIME_MM = 0.0
DEFAULT_PRESSURE_OFFSET_MM = 5.0
DEFAULT_PRESSURE_ADVANCE_FEED = 120.0
DEFAULT_PRESSURE_RETRACT_FEED = 240.0
DEFAULT_PREFLOW_DWELL_MS = 50
DEFAULT_NODE_DWELL_MS = 200
DEFAULT_Y_OFFPLANE_FIT_MODEL = "avg_cubic"

DEFAULT_TANGENTIAL_EXIT_MM = 4.0

DEFAULT_BBOX_X_MIN = -1.0e9
DEFAULT_BBOX_X_MAX = +1.0e9
DEFAULT_BBOX_Y_MIN = -1.0e9
DEFAULT_BBOX_Y_MAX = +1.0e9
DEFAULT_BBOX_Z_MIN = -1.0e9
DEFAULT_BBOX_Z_MAX = +1.0e9


# ---------------- Data classes ----------------

@dataclass
class Calibration:
    pr: np.ndarray
    pz: np.ndarray
    py_off: np.ndarray
    pa: np.ndarray

    b_min: float
    b_max: float
    tip_angle_min: float
    tip_angle_max: float

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


@dataclass
class PathState:
    tip_xyz: np.ndarray
    b_cmd: float
    c_cmd: float
    phase: str = "pull"


# ---------------- Calibration helpers ----------------

def poly_eval(coeffs: Any, u: Any, default_if_none: Optional[float] = None) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    if coeffs is None:
        if default_if_none is None:
            raise ValueError("Missing polynomial coefficients.")
        return np.full_like(u, float(default_if_none), dtype=float)
    arr = np.asarray(coeffs, dtype=float).reshape(-1)
    if arr.size == 0:
        if default_if_none is None:
            raise ValueError("Empty polynomial coefficients.")
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

    with p.open("r") as f:
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
        if py_off is None:
            py_off = np.zeros(1, dtype=float)
    else:
        py_off = np.asarray(py_off_raw, dtype=float)
    pa_raw = cubic.get("tip_angle_coeffs", None)
    if pa_raw is None:
        pa = _coeffs_from_model(fit_models, "tip_angle_cubic", "tip_angle_avg_cubic")
        if pa is None:
            raise ValueError("Calibration JSON is missing tip_angle_coeffs.")
    else:
        pa = np.asarray(pa_raw, dtype=float)

    selected_fit_model = data.get("selected_fit_model")
    selected_fit_model = None if selected_fit_model is None else str(selected_fit_model).strip().lower()
    selected_offplane_fit_model = data.get("selected_offplane_fit_model")
    selected_offplane_fit_model = None if selected_offplane_fit_model is None else str(selected_offplane_fit_model).strip().lower()
    requested_offplane_fit_model = (
        None if requested_offplane_fit_model is None else str(requested_offplane_fit_model).strip().lower()
    )
    active_phase = str(data.get("default_phase_for_legacy_access") or "pull").strip().lower()
    phase_models = data.get("fit_models_by_phase", {}) or {}
    active_phase_models = phase_models.get(active_phase) if isinstance(phase_models, dict) else None
    if not isinstance(active_phase_models, dict):
        active_phase_models = fit_models

    r_model = _select_named_model(active_phase_models, "r", selected_fit_model or "pchip")
    z_model = _select_named_model(active_phase_models, "z", selected_fit_model or "pchip")
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
    tip_angle_model = _select_named_model(active_phase_models, "tip_angle", selected_fit_model or "pchip")

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

    tip_env = data.get("working_envelope", {}).get("tip_angle_range_deg")
    if isinstance(tip_env, list) and len(tip_env) == 2:
        tip_angle_min = float(min(tip_env))
        tip_angle_max = float(max(tip_env))
    else:
        bb = np.linspace(b_min, b_max, 1001)
        aa = poly_eval(pa, bb)
        tip_angle_min = float(np.min(aa))
        tip_angle_max = float(np.max(aa))

    return Calibration(
        pr=pr,
        pz=pz,
        py_off=py_off,
        pa=pa,
        b_min=b_min,
        b_max=b_max,
        tip_angle_min=tip_angle_min,
        tip_angle_max=tip_angle_max,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        b_axis=b_axis,
        c_axis=c_axis,
        u_axis=u_axis,
        c_180_deg=c_180,
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
            return eval_pchip_with_linear_extrap(cal.y_off_model, cal.y_off_extrap_model, b)
        return eval_model_spec(cal.y_off_model, b, default_if_none=0.0)
    return poly_eval(cal.py_off, b, default_if_none=0.0)


def eval_tip_angle_deg(cal: Calibration, b: Any) -> np.ndarray:
    if cal.tip_angle_model is not None:
        return eval_model_spec(cal.tip_angle_model, b)
    return poly_eval(cal.pa, b)


def tip_offset_xyz_physical(cal: Calibration, b: float, c_deg: float) -> np.ndarray:
    """
    Corrected sign convention:
      local transverse = [r(B), -offplane_y(B)]

    Rotated into XY:
      x = r cos(C) + offplane_y sin(C)
      y = r sin(C) - offplane_y cos(C)
    """
    r = float(eval_r(cal, b))
    z = float(eval_z(cal, b))
    y_off_local = float(eval_offplane_y(cal, b))

    c = math.radians(float(c_deg))
    x = r * math.cos(c) + y_off_local * math.sin(c)
    y = r * math.sin(c) - y_off_local * math.cos(c)
    return np.array([x, y, z], dtype=float)


def stage_xyz_for_tip(cal: Calibration, tip_xyz: np.ndarray, b: float, c_deg: float) -> np.ndarray:
    return np.asarray(tip_xyz, dtype=float) - tip_offset_xyz_physical(cal, b, c_deg)


def compute_robot_transverse_radius(cal: Calibration, n: int = 1001) -> float:
    bb = np.linspace(cal.b_min, cal.b_max, n)
    rr = eval_r(cal, bb)
    yy = eval_offplane_y(cal, bb)
    return float(np.max(np.sqrt(rr * rr + yy * yy)))


# ---------------- Root finding ----------------

def _find_roots_for_target(coeffs: np.ndarray, target: float, b_min: float, b_max: float) -> List[float]:
    f = lambda b: float(poly_eval(coeffs, b) - target)
    bs = np.linspace(b_min, b_max, 800)
    fs = np.array([f(b) for b in bs], dtype=float)
    fs = np.where(np.isfinite(fs), fs, 1e30)

    roots: List[float] = []
    for a, b, fa, fb in zip(bs[:-1], bs[1:], fs[:-1], fs[1:]):
        if abs(fa) < 1e-12:
            roots.append(float(a))
        elif np.sign(fa) != np.sign(fb):
            try:
                roots.append(float(brentq(f, float(a), float(b), maxiter=100)))
            except Exception:
                roots.append(float(0.5 * (float(a) + float(b))))

    deduped: List[float] = []
    for r in roots:
        if not deduped or min(abs(r - d) for d in deduped) > 1e-6:
            deduped.append(r)
    return deduped


def solve_b_for_tip_angle(cal: Calibration, angle_target_deg: float, b_prev: Optional[float] = None) -> float:
    roots = _find_roots_for_target(cal.pa, float(angle_target_deg), cal.b_min, cal.b_max)
    if not roots:
        bb = np.linspace(cal.b_min, cal.b_max, 2001)
        aa = eval_tip_angle_deg(cal, bb)
        return float(bb[int(np.argmin(np.abs(aa - float(angle_target_deg))))])

    if b_prev is None or len(roots) == 1:
        return float(roots[0])

    arr = np.asarray(roots, dtype=float)
    return float(arr[np.argmin(np.abs(arr - float(b_prev)))])


# ---------------- Utilities ----------------

def fmt_axes_move(axes_vals: List[Tuple[str, float]]) -> str:
    return " ".join(f"{ax}{val:.3f}" for ax, val in axes_vals)


def clamp_stage_xyz_to_bbox(x: float, y: float, z: float, bbox: dict) -> Tuple[float, float, float]:
    return (
        float(np.clip(x, bbox["x_min"], bbox["x_max"])),
        float(np.clip(y, bbox["y_min"], bbox["y_max"])),
        float(np.clip(z, bbox["z_min"], bbox["z_max"])),
    )


def make_constant_bc_path(points: np.ndarray, b: float, c: float) -> List[PathState]:
    return [PathState(points[i], float(b), float(c)) for i in range(len(points))]


# ---------------- Circle builders ----------------

def build_xz_circle_path(
    cal: Calibration,
    center_xyz: np.ndarray,
    radius: float,
    samples_per_circle: int,
    c0_deg: float,
    c180_deg: float,
) -> List[PathState]:
    """
    No artificial "jump back to first state" at the end.
    The path closes geometrically and ends at leftmost point with C=0, B~180.
    """
    cx, cy, cz = [float(v) for v in center_xyz]
    n_lower = int(samples_per_circle) // 2
    n_upper = int(samples_per_circle) - n_lower

    raw_lower = np.linspace(math.pi, 2.0 * math.pi, n_lower + 1)
    raw_upper = np.linspace(2.0 * math.pi, 3.0 * math.pi, n_upper + 1)

    pts: List[np.ndarray] = []
    b_vals: List[float] = []
    c_vals: List[float] = []

    b_prev: Optional[float] = None

    # lower half: C=180, uncurl 180->0
    for th in raw_lower:
        p = np.array([cx + radius * math.cos(th), cy, cz + radius * math.sin(th)], dtype=float)
        progress = (th - math.pi) / math.pi
        ang = 180.0 * (1.0 - progress)
        ang = float(np.clip(ang, cal.tip_angle_min, cal.tip_angle_max))
        b_i = solve_b_for_tip_angle(cal, ang, b_prev=b_prev)
        pts.append(p)
        b_vals.append(b_i)
        c_vals.append(float(c180_deg))
        b_prev = b_i

    # seam duplicate at same point, new C state
    seam = np.array([cx + radius, cy, cz], dtype=float)
    pts.append(seam.copy())
    b_vals.append(b_vals[-1])
    c_vals.append(float(c0_deg))

    # upper half: C=0, curl 0->180
    for th in raw_upper[1:]:
        p = np.array([cx + radius * math.cos(th), cy, cz + radius * math.sin(th)], dtype=float)
        progress = (th - 2.0 * math.pi) / math.pi
        ang = 180.0 * progress
        ang = float(np.clip(ang, cal.tip_angle_min, cal.tip_angle_max))
        b_i = solve_b_for_tip_angle(cal, ang, b_prev=b_prev)
        pts.append(p)
        b_vals.append(b_i)
        c_vals.append(float(c0_deg))
        b_prev = b_i

    return [PathState(np.asarray(pts[i]), float(b_vals[i]), float(c_vals[i])) for i in range(len(pts))]


def build_yz_circle_path(
    cal: Calibration,
    center_xyz: np.ndarray,
    radius: float,
    samples_per_circle: int,
    c90_deg: float,
    c270_deg: float,
) -> List[PathState]:
    """
    No artificial end jump. Ends at negative-Y point with C=90, B~180.
    """
    cx, cy, cz = [float(v) for v in center_xyz]
    n_lower = int(samples_per_circle) // 2
    n_upper = int(samples_per_circle) - n_lower

    raw_lower = np.linspace(math.pi, 2.0 * math.pi, n_lower + 1)
    raw_upper = np.linspace(2.0 * math.pi, 3.0 * math.pi, n_upper + 1)

    pts: List[np.ndarray] = []
    b_vals: List[float] = []
    c_vals: List[float] = []

    b_prev: Optional[float] = None

    # lower half: C=270, uncurl 180->0
    for th in raw_lower:
        p = np.array([cx, cy + radius * math.cos(th), cz + radius * math.sin(th)], dtype=float)
        progress = (th - math.pi) / math.pi
        ang = 180.0 * (1.0 - progress)
        ang = float(np.clip(ang, cal.tip_angle_min, cal.tip_angle_max))
        b_i = solve_b_for_tip_angle(cal, ang, b_prev=b_prev)
        pts.append(p)
        b_vals.append(b_i)
        c_vals.append(float(c270_deg))
        b_prev = b_i

    seam = np.array([cx, cy + radius, cz], dtype=float)
    pts.append(seam.copy())
    b_vals.append(b_vals[-1])
    c_vals.append(float(c90_deg))

    # upper half: C=90, curl 0->180
    for th in raw_upper[1:]:
        p = np.array([cx, cy + radius * math.cos(th), cz + radius * math.sin(th)], dtype=float)
        progress = (th - 2.0 * math.pi) / math.pi
        ang = 180.0 * progress
        ang = float(np.clip(ang, cal.tip_angle_min, cal.tip_angle_max))
        b_i = solve_b_for_tip_angle(cal, ang, b_prev=b_prev)
        pts.append(p)
        b_vals.append(b_i)
        c_vals.append(float(c90_deg))
        b_prev = b_i

    return [PathState(np.asarray(pts[i]), float(b_vals[i]), float(c_vals[i])) for i in range(len(pts))]


def build_xy_circle_path(
    cal: Calibration,
    center_xyz: np.ndarray,
    radius: float,
    samples_per_circle: int,
    b_fixed: float,
    start_theta: float,
    ccw: bool = True,
) -> List[PathState]:
    """
    XY circle with continuous C and no end jump before dwell.
    """
    cx, cy, cz = [float(v) for v in center_xyz]
    t = np.linspace(0.0, 2.0 * math.pi, int(samples_per_circle) + 1)
    theta = float(start_theta) + (t if ccw else -t)

    x = cx + float(radius) * np.cos(theta)
    y = cy + float(radius) * np.sin(theta)
    z = np.full_like(theta, cz, dtype=float)

    r_fixed = float(eval_r(cal, b_fixed))
    yoff_fixed = float(eval_offplane_y(cal, b_fixed))
    local_alpha_deg = math.degrees(math.atan2(-yoff_fixed, r_fixed))

    if ccw:
        tangent_deg = np.degrees(theta + 0.5 * math.pi)
    else:
        tangent_deg = np.degrees(theta - 0.5 * math.pi)

    c_vals = tangent_deg - local_alpha_deg
    pts = np.column_stack([x, y, z])

    return [PathState(pts[i], float(b_fixed), float(c_vals[i])) for i in range(len(pts))]


# ---------------- Top-down structure builders ----------------

def build_xz_quarter_arc(center_xyz: np.ndarray, radius: float, theta_start: float, theta_end: float, samples: int) -> np.ndarray:
    cx, cy, cz = [float(v) for v in center_xyz]
    tt = np.linspace(theta_start, theta_end, int(samples) + 1)
    return np.column_stack([
        cx + radius * np.cos(tt),
        np.full_like(tt, cy, dtype=float),
        cz + radius * np.sin(tt),
    ])


def build_yz_quarter_arc(center_xyz: np.ndarray, radius: float, theta_start: float, theta_end: float, samples: int) -> np.ndarray:
    cx, cy, cz = [float(v) for v in center_xyz]
    tt = np.linspace(theta_start, theta_end, int(samples) + 1)
    return np.column_stack([
        np.full_like(tt, cx, dtype=float),
        cy + radius * np.cos(tt),
        cz + radius * np.sin(tt),
    ])


def build_xy_circle_fixed_bc(center_xyz: np.ndarray, radius: float, samples: int, start_theta: float) -> np.ndarray:
    cx, cy, cz = [float(v) for v in center_xyz]
    tt = np.linspace(start_theta, start_theta + 2.0 * math.pi, int(samples) + 1)
    return np.column_stack([
        cx + radius * np.cos(tt),
        cy + radius * np.sin(tt),
        np.full_like(tt, cz, dtype=float),
    ])


# ---------------- Writer ----------------

class GCodeWriter:
    def __init__(
        self,
        fh,
        cal: Calibration,
        bbox: dict,
        travel_feed: float,
        print_feed: float,
        bc_hold_feed: float,
        extrusion_per_mm: float,
        pressure_offset_mm: float,
        pressure_advance_feed: float,
        pressure_retract_feed: float,
        preflow_dwell_ms: int,
        node_dwell_ms: int,
    ):
        self.f = fh
        self.cal = cal
        self.bbox = bbox
        self.travel_feed = float(travel_feed)
        self.print_feed = float(print_feed)
        self.bc_hold_feed = float(bc_hold_feed)

        self.extrusion_per_mm = float(extrusion_per_mm)
        self.emit_extrusion = abs(self.extrusion_per_mm) > 0.0
        self.pressure_offset_mm = float(pressure_offset_mm)
        self.pressure_advance_feed = float(pressure_advance_feed)
        self.pressure_retract_feed = float(pressure_retract_feed)
        self.preflow_dwell_ms = int(preflow_dwell_ms)
        self.node_dwell_ms = int(node_dwell_ms)

        self.cur_stage_xyz: Optional[np.ndarray] = None
        self.cur_tip_xyz: Optional[np.ndarray] = None
        self.cur_b = 0.0
        self.cur_c = 0.0

        self.u_material_abs = 0.0
        self.pressure_charged = False

        self.highest_tip_z_printed = -1.0e18

    def _stage_for(self, tip_xyz: np.ndarray, b: float, c: float) -> np.ndarray:
        return stage_xyz_for_tip(self.cal, tip_xyz, b, c)

    def _emit_move(
        self,
        p_stage: np.ndarray,
        b: float,
        c: float,
        feed: float,
        with_u: bool = False,
        seg_len: float = 0.0,
        comment: Optional[str] = None,
    ):
        if comment:
            self.f.write(f"; {comment}\n")

        x, y, z = clamp_stage_xyz_to_bbox(p_stage[0], p_stage[1], p_stage[2], self.bbox)
        axes = [
            (self.cal.x_axis, x),
            (self.cal.y_axis, y),
            (self.cal.z_axis, z),
            (self.cal.b_axis, float(b)),
            (self.cal.c_axis, float(c)),
        ]

        if with_u and self.emit_extrusion:
            self.u_material_abs += self.extrusion_per_mm * float(seg_len)
            axes.append((self.cal.u_axis, self.u_material_abs + (self.pressure_offset_mm if self.pressure_charged else 0.0)))

        self.f.write(f"G1 {fmt_axes_move(axes)} F{float(feed):.0f}\n")
        self.cur_stage_xyz = np.array([x, y, z], dtype=float)
        self.cur_tip_xyz = None
        self.cur_b = float(b)
        self.cur_c = float(c)

    def move_stage_xyzbc(self, stage_xyz: np.ndarray, b: float, c: float, feed: float, comment: Optional[str] = None):
        self._emit_move(stage_xyz, b, c, feed, comment=comment)

    def move_to_tip_state(self, tip_xyz: np.ndarray, b: float, c: float, feed: float, comment: Optional[str] = None):
        stage = self._stage_for(tip_xyz, b, c)
        self._emit_move(stage, b, c, feed, comment=comment)
        self.cur_tip_xyz = np.asarray(tip_xyz, dtype=float).copy()

    def hold_tip_change_state(self, tip_xyz: np.ndarray, b_new: float, c_new: float, feed: float, comment: Optional[str] = None):
        stage = self._stage_for(tip_xyz, b_new, c_new)
        self._emit_move(stage, b_new, c_new, feed, comment=comment)
        self.cur_tip_xyz = np.asarray(tip_xyz, dtype=float).copy()

    def safe_travel_tip_via_z(
        self,
        target_tip_xyz: np.ndarray,
        b: float,
        c: float,
        z_lift_tip: float,
        comment_prefix: str,
    ):
        if self.cur_tip_xyz is None:
            raise RuntimeError("Current tip position unknown for safe tip travel.")

        cur = self.cur_tip_xyz.copy()
        up = cur.copy()
        up[2] = float(z_lift_tip)

        xy = target_tip_xyz.copy()
        xy[2] = float(z_lift_tip)

        self.move_to_tip_state(up, b, c, self.travel_feed, f"{comment_prefix}: up in Z")
        self.move_to_tip_state(xy, b, c, self.travel_feed, f"{comment_prefix}: move in XY")
        self.move_to_tip_state(target_tip_xyz, b, c, self.travel_feed, f"{comment_prefix}: down in Z")

    def pressure_preload_before_print(self):
        if self.emit_extrusion and not self.pressure_charged:
            self.pressure_charged = True
            self.f.write("; open pressure solenoid before print pass\n")
            self.f.write("M42 P0 S1\n")
            if self.preflow_dwell_ms > 0:
                self.f.write(f"G4 P{self.preflow_dwell_ms}\n")

    def pressure_release_after_print(self):
        if self.emit_extrusion and self.pressure_charged:
            if self.node_dwell_ms > 0:
                self.f.write("; end-of-circle dwell\n")
                self.f.write(f"G4 P{self.node_dwell_ms}\n")
            self.pressure_charged = False
            self.f.write("; close pressure solenoid after circle\n")
            self.f.write("M42 P0 S0\n")

    def print_path(self, path: List[PathState], label: str, move_to_start: bool = True):
        if not path:
            return

        if move_to_start:
            self.move_to_tip_state(path[0].tip_xyz, path[0].b_cmd, path[0].c_cmd, self.travel_feed, comment=f"{label}: move to start")

        self.pressure_preload_before_print()
        self.f.write(f"; print {label}\n")

        prev_stage = self._stage_for(path[0].tip_xyz, path[0].b_cmd, path[0].c_cmd)
        self.cur_tip_xyz = path[0].tip_xyz.copy()
        self.highest_tip_z_printed = max(self.highest_tip_z_printed, float(path[0].tip_xyz[2]))

        for i in range(1, len(path)):
            ps = path[i]
            stage_i = self._stage_for(ps.tip_xyz, ps.b_cmd, ps.c_cmd)

            tip_check = stage_i + tip_offset_xyz_physical(self.cal, ps.b_cmd, ps.c_cmd)
            if np.linalg.norm(tip_check - ps.tip_xyz) > 1e-8:
                raise RuntimeError(f"Tip tracking failed in {label} at point {i}.")

            seg_len = float(np.linalg.norm(stage_i - prev_stage))
            self._emit_move(stage_i, ps.b_cmd, ps.c_cmd, self.print_feed, with_u=False, seg_len=seg_len)
            self.cur_tip_xyz = ps.tip_xyz.copy()
            self.highest_tip_z_printed = max(self.highest_tip_z_printed, float(ps.tip_xyz[2]))
            prev_stage = stage_i.copy()

        self.pressure_release_after_print()


# ---------------- Main generation ----------------

def write_gcode(
    out_path: str,
    cal: Calibration,
    center_xyz: np.ndarray,
    topdown_center_xyz: np.ndarray,
    xz_radius: float,
    yz_radius: float,
    xy_radius: float,
    topdown_radius: float,
    write_yz: bool,
    samples_per_circle: int,
    samples_per_quarter_arc: int,
    c0_deg: float,
    c180_deg: float,
    c90_deg: float,
    c270_deg: float,
    tangential_exit_mm: float,
    machine_start_pose: Tuple[float, float, float, float, float],
    machine_end_pose: Tuple[float, float, float, float, float],
    safe_approach_z: float,
    travel_feed: float,
    print_feed: float,
    bc_hold_feed: float,
    extrusion_per_mm: float,
    prime_mm: float,
    pressure_offset_mm: float,
    pressure_advance_feed: float,
    pressure_retract_feed: float,
    preflow_dwell_ms: int,
    node_dwell_ms: int,
    bbox: dict,
):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)

    robot_radius = compute_robot_transverse_radius(cal)
    yz_radius_used = float(yz_radius)
    yz_radius_clamped = False
    if write_yz:
        yz_max = max(0.0, robot_radius - DEFAULT_RADIUS_SAFETY_MARGIN)
        if yz_radius_used >= yz_max:
            yz_radius_used = yz_max
            yz_radius_clamped = True

    b180 = solve_b_for_tip_angle(cal, 180.0)
    b0 = solve_b_for_tip_angle(cal, 0.0, b_prev=b180)
    b90 = solve_b_for_tip_angle(cal, 90.0, b_prev=b0)

    # ---------- Structure 1 ----------
    xz_path = build_xz_circle_path(cal, center_xyz, xz_radius, samples_per_circle, c0_deg, c180_deg)
    yz_path = build_yz_circle_path(cal, center_xyz, yz_radius_used, samples_per_circle, c90_deg, c270_deg) if write_yz else []

    # XY start:
    #   with YZ -> negative-Y point
    #   without YZ -> current XZ endpoint / leftmost point
    xy_start_theta = 1.5 * math.pi if write_yz else math.pi
    xy_path = build_xy_circle_path(cal, center_xyz, xy_radius, samples_per_circle, b90, start_theta=xy_start_theta)

    xz_start_tip = xz_path[0].tip_xyz.copy()
    yz_start_tip = np.array([center_xyz[0], center_xyz[1] - yz_radius_used, center_xyz[2]], dtype=float)
    xy_start_tip = xy_path[0].tip_xyz.copy()

    # Tangential move out in -Y, as requested
    xy_exit_tip = xy_path[-1].tip_xyz.copy() + np.array([0.0, -float(tangential_exit_mm), 0.0], dtype=float)

    # ---------- Structure 2 top-down only ----------
    td_cx, td_cy, td_cz = [float(v) for v in topdown_center_xyz]
    td_R = float(topdown_radius)

    # Fixed B/C for top-down structure
    td_b = float(b0)
    td_c = float(c0_deg)

    # Bottom quarter arcs: from bottom vertex to equator points
    bottom_xz_left = make_constant_bc_path(
        build_xz_quarter_arc(topdown_center_xyz, td_R, 1.5 * math.pi, math.pi, samples_per_quarter_arc),
        td_b, td_c
    )
    bottom_yz_back = make_constant_bc_path(
        build_yz_quarter_arc(topdown_center_xyz, td_R, math.pi, 1.5 * math.pi, samples_per_quarter_arc)[::-1],
        td_b, td_c
    )
    bottom_xz_right = make_constant_bc_path(
        build_xz_quarter_arc(topdown_center_xyz, td_R, 1.5 * math.pi, 2.0 * math.pi, samples_per_quarter_arc),
        td_b, td_c
    )
    bottom_yz_front = make_constant_bc_path(
        build_yz_quarter_arc(topdown_center_xyz, td_R, 1.5 * math.pi, 2.0 * math.pi, samples_per_quarter_arc),
        td_b, td_c
    )

    # Equator XY circle
    td_xy_circle = make_constant_bc_path(
        build_xy_circle_fixed_bc(topdown_center_xyz, td_R, samples_per_circle, start_theta=1.5 * math.pi),
        td_b, td_c
    )

    # Top quarter arcs: equator points to top vertex
    top_yz_front = make_constant_bc_path(
        build_yz_quarter_arc(topdown_center_xyz, td_R, 0.0, 0.5 * math.pi, samples_per_quarter_arc),
        td_b, td_c
    )
    top_xz_right = make_constant_bc_path(
        build_xz_quarter_arc(topdown_center_xyz, td_R, 0.0, 0.5 * math.pi, samples_per_quarter_arc),
        td_b, td_c
    )
    top_yz_back = make_constant_bc_path(
        build_yz_quarter_arc(topdown_center_xyz, td_R, math.pi, 0.5 * math.pi, samples_per_quarter_arc),
        td_b, td_c
    )
    top_xz_left = make_constant_bc_path(
        build_xz_quarter_arc(topdown_center_xyz, td_R, math.pi, 0.5 * math.pi, samples_per_quarter_arc),
        td_b, td_c
    )

    with open(out_path, "w") as f:
        f.write("; generated by circle_xyz_calibrated_final.py\n")
        f.write("; corrected offplane sign convention\n")
        f.write("; structure 1: XZ -> optional YZ -> XY\n")
        f.write("; structure 2: top-down-only spherical wireframe, B fixed at 0deg-equivalent solve\n")
        f.write("; tip-space exact tracking via stage_xyz = tip_xyz - offset_xyz(B,C)\n")
        f.write(f"; axes: X->{cal.x_axis}, Y->{cal.y_axis}, Z->{cal.z_axis}, B->{cal.b_axis}, C->{cal.c_axis}, U->{cal.u_axis}\n")
        f.write(
            "; y_offplane_model "
            f"requested={cal.requested_offplane_fit_model or 'calibration_default'} "
            f"resolved={cal.resolved_offplane_fit_model or 'unspecified'} "
            f"calibration_default={cal.selected_offplane_fit_model or cal.selected_fit_model or 'unspecified'} "
            f"active_phase={cal.active_phase}\n"
        )
        f.write(f"; robot transverse radius = {robot_radius:.6f} mm\n")
        f.write(f"; B(180)={b180:.6f}, B(0)={b0:.6f}, B(90)={b90:.6f}\n")
        f.write(f"; structure1 center = [{center_xyz[0]:.3f}, {center_xyz[1]:.3f}, {center_xyz[2]:.3f}]\n")
        f.write(f"; structure2 center = [{topdown_center_xyz[0]:.3f}, {topdown_center_xyz[1]:.3f}, {topdown_center_xyz[2]:.3f}]\n")
        if yz_radius_clamped:
            f.write("; info: YZ radius was clamped below robot transverse radius\n")
        f.write("G90\n")

        g = GCodeWriter(
            fh=f,
            cal=cal,
            bbox=bbox,
            travel_feed=travel_feed,
            print_feed=print_feed,
            bc_hold_feed=bc_hold_feed,
            extrusion_per_mm=extrusion_per_mm,
            pressure_offset_mm=pressure_offset_mm,
            pressure_advance_feed=pressure_advance_feed,
            pressure_retract_feed=pressure_retract_feed,
            preflow_dwell_ms=preflow_dwell_ms,
            node_dwell_ms=node_dwell_ms,
        )
        msx, msy, msz, msb, msc = [float(v) for v in machine_start_pose]
        mex, mey, mez, meb, mec = [float(v) for v in machine_end_pose]

        # ---- Startup ----
        g.move_stage_xyzbc(np.array([msx, msy, safe_approach_z], dtype=float), msb, msc, travel_feed, "startup: safe approach")
        g.move_stage_xyzbc(np.array([msx, msy, msz], dtype=float), msb, msc, travel_feed, "startup: dive")

        # ==================== Structure 1 ====================
        g.move_to_tip_state(xz_start_tip, b180, c180_deg, travel_feed, "move to XZ start")
        g.print_path(xz_path, "XZ circle", move_to_start=False)

        if write_yz:
            g.move_to_tip_state(yz_start_tip, b180, c90_deg, travel_feed, "move to YZ start with transit orientation")
            g.hold_tip_change_state(yz_start_tip, b180, c270_deg, bc_hold_feed, "rotate C by 180 at fixed tip for YZ start")
            g.print_path(yz_path, "YZ circle", move_to_start=False)

            if np.linalg.norm(yz_path[-1].tip_xyz - xy_start_tip) > 1e-9:
                g.move_to_tip_state(xy_start_tip, b180, c270_deg, travel_feed, "move to XY start")
            g.hold_tip_change_state(xy_start_tip, b0, c270_deg, bc_hold_feed, "hold tip: uncurl B to 0")
            g.hold_tip_change_state(xy_start_tip, b0, c0_deg, bc_hold_feed, "hold tip: rotate C by +90")
            g.hold_tip_change_state(xy_start_tip, b90, xy_path[0].c_cmd, bc_hold_feed, "hold tip: set B to 90 and align C to XY start")
        else:
            # XZ ends at the same tip point where skip-YZ XY starts
            g.hold_tip_change_state(
                xz_path[-1].tip_xyz,
                b_new=float(b90),
                c_new=float(xy_path[0].c_cmd),
                feed=bc_hold_feed,
                comment="skip YZ: hold tip fixed while changing B to 90 and rotating C to XY tangential start"
            )

        g.print_path(xy_path, "XY circle", move_to_start=False)

        # Tangential move out in -Y
        g.move_to_tip_state(xy_exit_tip, b90, xy_path[-1].c_cmd, travel_feed, "tangential move out in -Y")
        g.hold_tip_change_state(xy_exit_tip, b0, xy_path[-1].c_cmd, bc_hold_feed, "hold tip: uncurl B to straight")

        # Before structure 2, set C back to 0 at fixed tip to simplify top-down phase
        g.hold_tip_change_state(xy_exit_tip, b0, c0_deg, bc_hold_feed, "hold tip: rotate C back to 0 for top-down structure")

        # ==================== Structure 2 ====================
        f.write("; ==================== top-down-only structure ====================\n")

        topdown_paths = [
            ("topdown_bottom_xz_left", bottom_xz_left),
            ("topdown_bottom_yz_back", bottom_yz_back),
            ("topdown_bottom_xz_right", bottom_xz_right),
            ("topdown_bottom_yz_front", bottom_yz_front),
            ("topdown_xy_circle", td_xy_circle),
            ("topdown_top_yz_front", top_yz_front),
            ("topdown_top_xz_right", top_xz_right),
            ("topdown_top_yz_back", top_yz_back),
            ("topdown_top_xz_left", top_xz_left),   # finishes at top vertex
        ]

        for idx, (label, path) in enumerate(topdown_paths):
            start_tip = path[0].tip_xyz.copy()
            if g.cur_tip_xyz is None:
                g.move_to_tip_state(start_tip, td_b, td_c, travel_feed, f"{label}: move to start")
            else:
                z_lift = max(g.highest_tip_z_printed, float(start_tip[2]))
                g.safe_travel_tip_via_z(
                    start_tip,
                    td_b,
                    td_c,
                    z_lift_tip=z_lift,
                    comment_prefix=f"{label} travel"
                )
            g.print_path(path, label, move_to_start=False)

        # ---- Final home ----
        if g.cur_tip_xyz is not None:
            z_lift = max(g.highest_tip_z_printed, safe_approach_z)
            safe_tip = g.cur_tip_xyz.copy()
            safe_tip[2] = float(z_lift)
            g.move_to_tip_state(safe_tip, td_b, td_c, travel_feed, "final: raise in tip Z before home")

        g.move_stage_xyzbc(np.array([g.cur_stage_xyz[0], g.cur_stage_xyz[1], safe_approach_z], dtype=float), g.cur_b, g.cur_c, travel_feed, "final: raise stage to safe Z")
        g.move_stage_xyzbc(np.array([mex, mey, safe_approach_z], dtype=float), meb, mec, travel_feed, "final: move home XY at safe Z")
        g.move_stage_xyzbc(np.array([mex, mey, mez], dtype=float), meb, mec, travel_feed, "final: dive to home/end Z")

    print(f"Wrote {out_path}")
    print(f"Axes: X={cal.x_axis}, Y={cal.y_axis}, Z={cal.z_axis}, B={cal.b_axis}, C={cal.c_axis}, U={cal.u_axis}")
    print(f"Robot transverse radius: {robot_radius:.6f} mm")
    print(f"B(180)={b180:.6f}, B(0)={b0:.6f}, B(90)={b90:.6f}")
    print(f"Structure 1 order: XZ -> {'YZ -> ' if write_yz else ''}XY")
    print("Structure 2 order: bottom quarter arcs -> XY circle -> top quarter arcs")
    print("XZ/YZ no longer jump back to the initial C state before dwell.")
    print("End exit move changed to -Y before lifting in Z.")


# ---------------- CLI ----------------

def main(args):
    cal = load_calibration(args.calibration, requested_offplane_fit_model=DEFAULT_Y_OFFPLANE_FIT_MODEL)

    bbox = {
        "x_min": float(args.bbox_x_min),
        "x_max": float(args.bbox_x_max),
        "y_min": float(args.bbox_y_min),
        "y_max": float(args.bbox_y_max),
        "z_min": float(args.bbox_z_min),
        "z_max": float(args.bbox_z_max),
    }

    center_xyz = np.array([float(args.center_x), float(args.center_y), float(args.center_z)], dtype=float)
    topdown_center_xyz = np.array([float(args.topdown_center_x), float(args.topdown_center_y), float(args.topdown_center_z)], dtype=float)

    base_radius = float(args.radius)
    xz_radius = float(args.xz_radius) if args.xz_radius is not None else base_radius
    yz_radius = float(args.yz_radius) if args.yz_radius is not None else base_radius
    xy_radius = float(args.xy_radius) if args.xy_radius is not None else base_radius
    topdown_radius = float(args.topdown_radius)

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

    write_gcode(
        out_path=str(args.out),
        cal=cal,
        center_xyz=center_xyz,
        topdown_center_xyz=topdown_center_xyz,
        xz_radius=xz_radius,
        yz_radius=yz_radius,
        xy_radius=xy_radius,
        topdown_radius=topdown_radius,
        write_yz=not args.skip_yz,
        samples_per_circle=int(args.samples_per_circle),
        samples_per_quarter_arc=int(args.samples_per_quarter_arc),
        c0_deg=float(args.c0_deg),
        c180_deg=float(args.c180_deg if args.c180_deg is not None else cal.c_180_deg),
        c90_deg=float(args.c90_deg),
        c270_deg=float(args.c270_deg),
        tangential_exit_mm=float(args.tangential_exit_mm),
        machine_start_pose=machine_start_pose,
        machine_end_pose=machine_end_pose,
        safe_approach_z=float(args.safe_approach_z),
        travel_feed=float(args.travel_feed),
        print_feed=float(args.print_feed),
        bc_hold_feed=float(args.bc_hold_feed),
        extrusion_per_mm=float(args.extrusion_per_mm),
        prime_mm=float(args.prime_mm),
        pressure_offset_mm=float(args.pressure_offset_mm),
        pressure_advance_feed=float(args.pressure_advance_feed),
        pressure_retract_feed=float(args.pressure_retract_feed),
        preflow_dwell_ms=int(args.preflow_dwell_ms),
        node_dwell_ms=int(args.node_dwell_ms),
        bbox=bbox,
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser(
        description="Generate calibrated spherical-style G-code with corrected off-plane sign, no end-of-circle C jump, -Y exit, and a second top-down structure."
    )
    ap.add_argument("--calibration", required=True)
    ap.add_argument("--out", default=DEFAULT_OUT)

    # Structure 1 center
    ap.add_argument("--center-x", type=float, default=DEFAULT_CENTER_X)
    ap.add_argument("--center-y", type=float, default=DEFAULT_CENTER_Y)
    ap.add_argument("--center-z", type=float, default=DEFAULT_CENTER_Z)

    # Structure 2 center
    ap.add_argument("--topdown-center-x", type=float, default=DEFAULT_TOPDOWN_CENTER_X)
    ap.add_argument("--topdown-center-y", type=float, default=DEFAULT_TOPDOWN_CENTER_Y)
    ap.add_argument("--topdown-center-z", type=float, default=DEFAULT_TOPDOWN_CENTER_Z)

    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    ap.add_argument("--xz-radius", type=float, default=None)
    ap.add_argument("--yz-radius", type=float, default=None)
    ap.add_argument("--xy-radius", type=float, default=None)
    ap.add_argument("--topdown-radius", type=float, default=DEFAULT_TOPDOWN_RADIUS)

    ap.add_argument("--samples-per-circle", type=int, default=DEFAULT_SAMPLES_PER_CIRCLE)
    ap.add_argument("--samples-per-quarter-arc", type=int, default=DEFAULT_SAMPLES_PER_QUARTER_ARC)

    ap.add_argument("--skip-yz", action="store_true", default=False)

    ap.add_argument("--c0-deg", type=float, default=0.0)
    ap.add_argument("--c180-deg", type=float, default=None)
    ap.add_argument("--c90-deg", type=float, default=90.0)
    ap.add_argument("--c270-deg", type=float, default=270.0)

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

    ap.add_argument("--safe-approach-z", type=float, default=DEFAULT_SAFE_APPROACH_Z)

    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED)
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED)
    ap.add_argument("--bc-hold-feed", type=float, default=DEFAULT_BC_HOLD_FEED)
    ap.add_argument("--tangential-exit-mm", type=float, default=DEFAULT_TANGENTIAL_EXIT_MM)

    ap.add_argument("--extrusion-per-mm", type=float, default=DEFAULT_EXTRUSION_PER_MM)
    ap.add_argument("--prime-mm", type=float, default=DEFAULT_PRIME_MM)
    ap.add_argument("--pressure-offset-mm", type=float, default=DEFAULT_PRESSURE_OFFSET_MM)
    ap.add_argument("--pressure-advance-feed", type=float, default=DEFAULT_PRESSURE_ADVANCE_FEED)
    ap.add_argument("--pressure-retract-feed", type=float, default=DEFAULT_PRESSURE_RETRACT_FEED)
    ap.add_argument("--preflow-dwell-ms", type=int, default=DEFAULT_PREFLOW_DWELL_MS)
    ap.add_argument("--node-dwell-ms", type=int, default=DEFAULT_NODE_DWELL_MS)

    ap.add_argument("--bbox-x-min", type=float, default=DEFAULT_BBOX_X_MIN)
    ap.add_argument("--bbox-x-max", type=float, default=DEFAULT_BBOX_X_MAX)
    ap.add_argument("--bbox-y-min", type=float, default=DEFAULT_BBOX_Y_MIN)
    ap.add_argument("--bbox-y-max", type=float, default=DEFAULT_BBOX_Y_MAX)
    ap.add_argument("--bbox-z-min", type=float, default=DEFAULT_BBOX_Z_MIN)
    ap.add_argument("--bbox-z-max", type=float, default=DEFAULT_BBOX_Z_MAX)

    args = ap.parse_args()
    main(args)
