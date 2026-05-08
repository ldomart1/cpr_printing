#!/usr/bin/env python3
"""
Generate calibrated G-code for torus skin printing with both OUTER and INNER
surface families.

Geometry
--------
The torus lies in the XZ plane, so its ring axis is +Y:

    (sqrt((x-cx)^2 + (z-cz)^2) - R)^2 + (y-cy)^2 = r^2

where:
    R = centerline radius = centerline_diameter / 2
    r = tube radius       = tube_diameter / 2

Branch families
---------------
For each Y plane, the torus intersects the plane in one or two circles in XZ:

    outer radius = R + sqrt(r^2 - (y-cy)^2)
    inner radius = R - sqrt(r^2 - (y-cy)^2)

This script can emit the outer family, the inner family, or both.
Default is both.

Build strategy
--------------
For each selected branch family:

Phase 1: front sweep
    * planes from y = cy down to y = cy - r
    * each plane prints the selected slice circle in the XZ plane
    * requested tool orientation is fixed at tip-angle 90 deg, C = -90 deg
    * every point is tip-tracked through the calibration model

Phase 2: side sweep
    * starts again at y = cy - r
    * continues plane-by-plane all the way to y = cy + r
    * each plane prints the same slice circle from the side:
        - start at the top point with requested tip-angle 0 deg, C = -180 deg
        - right half: tip-angle 0 -> 180, C fixed at -180
        - left half:  tip-angle 180 -> 0, C blended -180 -> 0
    * between planes, the tool tracks the top point of the next circle while C
      resets from 0 back to -180
    * every point is tip-tracked through the calibration model

Notes
-----
* Calibration-based tip tracking is used whenever --write-mode calibrated.
* At y = cy +/- r the inner and outer radii are identical (= R). When
  --branches both is used, the inner branch skips those exact endpoint planes to
  avoid duplicating the same circle already written by the outer branch.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np


# ---------------------------- defaults ----------------------------
DEFAULT_OUT = "gcode_generation/torus_1.gcode"
DEFAULT_CENTER_X = 100.0
DEFAULT_CENTER_Y = 52.0
DEFAULT_CENTER_Z = -140.0
DEFAULT_CENTERLINE_DIAMETER = 40.0
DEFAULT_TUBE_DIAMETER = 10.0
DEFAULT_PLANE_STEP = 0.40
DEFAULT_PRINT_FEED = 5000.0
DEFAULT_TRAVEL_FEED = 5000.0
DEFAULT_BRIDGE_FEED = 5000.0
DEFAULT_SAFE_Z_LIFT = 8.0
DEFAULT_MACHINE_START_X = 100.0
DEFAULT_MACHINE_START_Y = 60.0
DEFAULT_MACHINE_START_Z = -100.0
DEFAULT_MACHINE_START_B = 0.0
DEFAULT_MACHINE_START_C = 0.0
DEFAULT_MACHINE_END_X = 100.0
DEFAULT_MACHINE_END_Y = 60.0
DEFAULT_MACHINE_END_Z = -100.0
DEFAULT_MACHINE_END_B = 0.0
DEFAULT_MACHINE_END_C = 0.0
DEFAULT_FRONT_CIRCLE_SAMPLES = 360
DEFAULT_SIDE_HALF_SAMPLES = 180
DEFAULT_BRIDGE_SAMPLES = 60
DEFAULT_BC_SOLVE_SAMPLES = 5001
DEFAULT_Y_OFFPLANE_SIGN = 1.0
DEFAULT_WRITE_MODE = "calibrated"
DEFAULT_EXTRUSION_AXIS = "U"
DEFAULT_EXTRUSION_PER_MM = 0.0015
DEFAULT_PRIME_MM = 4
DEFAULT_BRANCHES = "both"


# ---------------------------- data types ----------------------------
@dataclass(frozen=True)
class TipPose:
    x: float
    y: float
    z: float
    tip_angle_deg: float
    c_deg: float


@dataclass
class Config:
    calibration: Optional[str]
    write_mode: str
    out: str

    center_x: float = DEFAULT_CENTER_X
    center_y: float = DEFAULT_CENTER_Y
    center_z: float = DEFAULT_CENTER_Z
    centerline_diameter: float = DEFAULT_CENTERLINE_DIAMETER
    tube_diameter: float = DEFAULT_TUBE_DIAMETER
    plane_step: float = DEFAULT_PLANE_STEP
    branches: str = DEFAULT_BRANCHES

    front_circle_samples: int = DEFAULT_FRONT_CIRCLE_SAMPLES
    side_half_samples: int = DEFAULT_SIDE_HALF_SAMPLES
    bridge_samples: int = DEFAULT_BRIDGE_SAMPLES
    bc_solve_samples: int = DEFAULT_BC_SOLVE_SAMPLES
    y_offplane_sign: float = DEFAULT_Y_OFFPLANE_SIGN

    print_feed: float = DEFAULT_PRINT_FEED
    travel_feed: float = DEFAULT_TRAVEL_FEED
    bridge_feed: float = DEFAULT_BRIDGE_FEED
    safe_z_lift: float = DEFAULT_SAFE_Z_LIFT

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

    use_extrusion: bool = False
    extrusion_axis: str = DEFAULT_EXTRUSION_AXIS
    extrusion_per_mm: float = DEFAULT_EXTRUSION_PER_MM
    prime_mm: float = DEFAULT_PRIME_MM


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


# ---------------------------- helpers ----------------------------
def frange_inclusive(start: float, stop: float, step: float) -> List[float]:
    if step == 0.0:
        raise ValueError("step must be non-zero")
    vals: List[float] = []
    cur = float(start)
    eps = abs(step) * 1e-6 + 1e-9
    if step > 0:
        while cur <= stop + eps:
            vals.append(cur)
            cur += step
    else:
        while cur >= stop - eps:
            vals.append(cur)
            cur += step
    if not vals:
        vals.append(float(start))
    if abs(vals[-1] - stop) > eps:
        vals.append(float(stop))
    return vals


def lerp(a: float, b: float, t: float) -> float:
    return float(a) + (float(b) - float(a)) * float(t)


def wrap_near(target_deg: float, reference_deg: float) -> float:
    out = float(target_deg)
    ref = float(reference_deg)
    while out - ref > 180.0:
        out -= 360.0
    while out - ref < -180.0:
        out += 360.0
    return out


def dist_xyz(a: Sequence[float], b: Sequence[float]) -> float:
    aa = np.asarray(a, dtype=float)
    bb = np.asarray(b, dtype=float)
    return float(np.linalg.norm(bb - aa))


def normalize_branch_selection(branches: str) -> Tuple[str, ...]:
    raw = str(branches).strip().lower()
    if raw in {"both", "all", "outer,inner", "inner,outer"}:
        return ("outer", "inner")
    pieces = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    if not pieces:
        raise ValueError("--branches must be outer, inner, or both")
    out: List[str] = []
    for piece in pieces:
        if piece not in {"outer", "inner"}:
            raise ValueError("--branches must be outer, inner, or both")
        if piece not in out:
            out.append(piece)
    return tuple(out)


# ---------------------------- torus geometry ----------------------------
def major_radius(cfg: Config) -> float:
    return 0.5 * float(cfg.centerline_diameter)


def tube_radius(cfg: Config) -> float:
    return 0.5 * float(cfg.tube_diameter)


def slice_radius(cfg: Config, y: float, branch: str) -> float:
    R = major_radius(cfg)
    r = tube_radius(cfg)
    dy = float(y) - float(cfg.center_y)
    inside = max(0.0, r * r - dy * dy)
    delta = math.sqrt(inside)
    branch = str(branch).strip().lower()
    if branch == "outer":
        return R + delta
    if branch == "inner":
        return R - delta
    raise ValueError(f"Unsupported branch: {branch}")


def tip_xyz_on_circle(cfg: Config, radius: float, y: float, theta_deg: float) -> np.ndarray:
    th = math.radians(float(theta_deg))
    return np.array(
        [
            float(cfg.center_x) + float(radius) * math.cos(th),
            float(y),
            float(cfg.center_z) + float(radius) * math.sin(th),
        ],
        dtype=float,
    )


def build_front_circle(cfg: Config, y: float, branch: str) -> List[TipPose]:
    radius = slice_radius(cfg, y, branch)
    n = max(12, int(cfg.front_circle_samples))
    out: List[TipPose] = []
    for i in range(n + 1):
        t = i / n
        theta_deg = 90.0 - 360.0 * t  # start at top, move clockwise
        xyz = tip_xyz_on_circle(cfg, radius, y, theta_deg)
        out.append(TipPose(float(xyz[0]), float(xyz[1]), float(xyz[2]), 90.0, -90.0))
    return out


def build_side_circle(cfg: Config, y: float, branch: str) -> List[TipPose]:
    radius = slice_radius(cfg, y, branch)
    n_half = max(8, int(cfg.side_half_samples))
    out: List[TipPose] = []

    # Right half: top -> bottom along the right side.
    for i in range(n_half + 1):
        t = i / n_half
        theta_deg = 90.0 - 180.0 * t
        xyz = tip_xyz_on_circle(cfg, radius, y, theta_deg)
        out.append(TipPose(float(xyz[0]), float(xyz[1]), float(xyz[2]), lerp(0.0, 180.0, t), -180.0))

    # Left half continuation: bottom -> top while C blends -180 -> 0.
    for i in range(1, n_half + 1):
        t = i / n_half
        theta_deg = -90.0 - 180.0 * t
        xyz = tip_xyz_on_circle(cfg, radius, y, theta_deg)
        out.append(TipPose(float(xyz[0]), float(xyz[1]), float(xyz[2]), lerp(180.0, 0.0, t), lerp(-180.0, 0.0, t)))

    return out


def build_plane_sequences(cfg: Config) -> Tuple[List[float], List[float]]:
    """
    Full Y coverage for a selected torus surface family.

    Phase 1: cy -> cy-r
    Phase 2: cy-r -> cy+r
    """
    r = tube_radius(cfg)
    step = abs(float(cfg.plane_step))
    phase1 = frange_inclusive(float(cfg.center_y), float(cfg.center_y) - r, -step)
    phase2 = frange_inclusive(float(cfg.center_y) - r, float(cfg.center_y) + r, step)
    return phase1, phase2


def filter_planes_for_branch(phase1: List[float], phase2: List[float], branch: str, selected_branches: Tuple[str, ...]) -> Tuple[List[float], List[float]]:
    """
    When printing both outer and inner families, skip the exact endpoint planes
    for the inner family because those circles coincide exactly with the outer
    family at radius R.
    """
    branch = str(branch).strip().lower()
    if branch != "inner" or tuple(selected_branches) != ("outer", "inner"):
        return list(phase1), list(phase2)

    # Keep the center plane in phase 1, but drop cy-r from the front sweep.
    phase1_f = list(phase1[:-1]) if len(phase1) > 1 else []
    # Drop both endpoint planes cy-r and cy+r from the side sweep.
    phase2_f = list(phase2[1:-1]) if len(phase2) > 2 else []
    return phase1_f, phase2_f


# ---------------------------- calibration helpers ----------------------------
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
    if pr_arr is not None:
        pr = np.asarray(pr_arr, dtype=float)
    else:
        tmp = _coeffs_from_model(fit_models, "r_cubic", "r_avg_cubic")
        pr = tmp if tmp is not None else np.zeros(1, dtype=float)

    pz_arr = cubic.get("z_coeffs")
    if pz_arr is not None:
        pz = np.asarray(pz_arr, dtype=float)
    else:
        tmp = _coeffs_from_model(fit_models, "z_cubic", "z_avg_cubic")
        pz = tmp if tmp is not None else np.zeros(1, dtype=float)

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
        u_axis=str(duet_map.get("extruder_axis") or DEFAULT_EXTRUSION_AXIS),
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


# ---------------------------- machine planning ----------------------------
def machine_pose_from_tip(cfg: Config, cal: Optional[Calibration], tip_pose: TipPose) -> Tuple[np.ndarray, float, float]:
    c_deg = float(tip_pose.c_deg)
    if str(cfg.write_mode).strip().lower() == "calibrated":
        if cal is None:
            raise ValueError("Calibration is required in calibrated mode.")
        machine_b = solve_b_for_target_tip_angle(cal, float(tip_pose.tip_angle_deg), search_samples=int(cfg.bc_solve_samples))
        stage_xyz = stage_xyz_for_tip(cal, np.array([tip_pose.x, tip_pose.y, tip_pose.z], dtype=float), machine_b, c_deg)
        return stage_xyz, float(machine_b), float(c_deg)
    stage_xyz = np.array([tip_pose.x, tip_pose.y, tip_pose.z], dtype=float)
    return stage_xyz, float(tip_pose.tip_angle_deg), float(c_deg)


# ---------------------------- writer ----------------------------
class GCodeWriter:
    def __init__(self, cfg: Config, cal: Optional[Calibration]):
        self.cfg = cfg
        self.cal = cal
        self.lines: List[str] = []
        self.last_stage_xyz: Optional[np.ndarray] = None
        self.last_tip_xyz: Optional[np.ndarray] = None
        self.last_b: Optional[float] = None
        self.last_c: Optional[float] = None
        self.e_accum: float = 0.0

    def add(self, line: str = "") -> None:
        self.lines.append(line)

    def header(self) -> None:
        cfg = self.cfg
        self.add("; torus skin generator")
        self.add(f"; write_mode={cfg.write_mode}")
        self.add(f"; calibration={cfg.calibration}")
        self.add(f"; branches={cfg.branches}")
        self.add(f"; center=({cfg.center_x:.3f}, {cfg.center_y:.3f}, {cfg.center_z:.3f})")
        self.add(f"; centerline_diameter={cfg.centerline_diameter:.3f}")
        self.add(f"; tube_diameter={cfg.tube_diameter:.3f}")
        self.add(f"; plane_step={cfg.plane_step:.3f}")
        self.add("; phase1: front sweep cy -> cy-r")
        self.add("; phase2: side sweep cy-r -> cy+r")
        if self.cal is not None:
            self.add(f"; axes: {self.cal.x_axis} {self.cal.y_axis} {self.cal.z_axis} {self.cal.b_axis} {self.cal.c_axis} {self.cal.u_axis}")
            self.add(f"; selected_fit_model={self.cal.selected_fit_model}")
            self.add(f"; selected_offplane_fit_model={self.cal.selected_offplane_fit_model}")
            self.add(f"; active_phase={self.cal.active_phase}")
            self.add(f"; y_offplane_sign={self.cal.offplane_y_sign:.1f}")
        self.add("G90")
        if cfg.use_extrusion:
            self.add("M83")
        self.add()

    def axis_names(self) -> Tuple[str, str, str, str, str, str]:
        if self.cal is None:
            return "X", "Y", "Z", "B", "C", self.cfg.extrusion_axis
        return self.cal.x_axis, self.cal.y_axis, self.cal.z_axis, self.cal.b_axis, self.cal.c_axis, self.cal.u_axis

    def emit_stage_move(
        self,
        stage_xyz: np.ndarray,
        b: float,
        c: float,
        feed: float,
        tip_xyz: Optional[np.ndarray] = None,
        extrude: bool = False,
        comment: Optional[str] = None,
    ) -> None:
        xax, yax, zax, bax, cax, uax = self.axis_names()
        if comment:
            self.add(f"; {comment}")
        parts = [
            f"{xax}{float(stage_xyz[0]):.3f}",
            f"{yax}{float(stage_xyz[1]):.3f}",
            f"{zax}{float(stage_xyz[2]):.3f}",
            f"{bax}{float(b):.3f}",
            f"{cax}{float(c):.3f}",
        ]
        if extrude and self.cfg.use_extrusion:
            if tip_xyz is not None and self.last_tip_xyz is not None:
                self.e_accum += dist_xyz(self.last_tip_xyz, tip_xyz) * float(self.cfg.extrusion_per_mm)
            parts.append(f"{uax}{self.e_accum:.5f}")
        self.add(f"G1 {' '.join(parts)} F{float(feed):.1f}")
        self.last_stage_xyz = np.asarray(stage_xyz, dtype=float).copy()
        if tip_xyz is not None:
            self.last_tip_xyz = np.asarray(tip_xyz, dtype=float).copy()
        self.last_b = float(b)
        self.last_c = float(c)

    def move_tip(self, tip_pose: TipPose, feed: float, extrude: bool = False, comment: Optional[str] = None) -> None:
        stage_xyz, machine_b, machine_c = machine_pose_from_tip(self.cfg, self.cal, tip_pose)
        if self.last_c is not None:
            machine_c = wrap_near(machine_c, self.last_c)
        self.emit_stage_move(
            stage_xyz=stage_xyz,
            b=machine_b,
            c=machine_c,
            feed=feed,
            tip_xyz=np.array([tip_pose.x, tip_pose.y, tip_pose.z], dtype=float),
            extrude=extrude,
            comment=comment,
        )

    def lift_and_travel_to_tip(self, tip_pose: TipPose, note: str) -> None:
        target_stage_xyz, target_b, target_c = machine_pose_from_tip(self.cfg, self.cal, tip_pose)
        if self.last_stage_xyz is not None and self.last_b is not None and self.last_c is not None:
            lifted = self.last_stage_xyz.copy()
            lifted[2] += float(self.cfg.safe_z_lift)
            self.emit_stage_move(lifted, self.last_b, self.last_c, self.cfg.travel_feed, tip_xyz=self.last_tip_xyz, extrude=False, comment=f"lift before {note}")
        travel = target_stage_xyz.copy()
        travel[2] += float(self.cfg.safe_z_lift)
        c_travel = target_c if self.last_c is None else wrap_near(target_c, self.last_c)
        self.emit_stage_move(travel, target_b, c_travel, self.cfg.travel_feed, tip_xyz=self.last_tip_xyz, extrude=False, comment=note)
        self.emit_stage_move(target_stage_xyz, target_b, c_travel, self.cfg.travel_feed, tip_xyz=np.array([tip_pose.x, tip_pose.y, tip_pose.z], dtype=float), extrude=False, comment=f"lower after {note}")

    def polyline(self, poses: Sequence[TipPose], feed: float, comment: str) -> None:
        if not poses:
            return
        start = poses[0]
        if self.last_tip_xyz is None:
            self.move_tip(start, self.cfg.travel_feed, extrude=False, comment=f"position for {comment}")
        else:
            start_xyz = np.array([start.x, start.y, start.z], dtype=float)
            if dist_xyz(self.last_tip_xyz, start_xyz) > 1e-9 or (self.last_c is not None and abs(wrap_near(start.c_deg, self.last_c) - self.last_c) > 1e-9):
                self.move_tip(start, self.cfg.travel_feed, extrude=False, comment=f"position for {comment}")
        self.add(f"; begin {comment}")
        for pose in poses[1:]:
            self.move_tip(pose, feed, extrude=self.cfg.use_extrusion)
        self.add(f"; end {comment}")

    def tracked_bridge(self, start: TipPose, end: TipPose, samples: int, feed: float, comment: str) -> None:
        self.add(f"; begin {comment}")
        samples = max(2, int(samples))
        prev_c = start.c_deg if self.last_c is None else self.last_c
        for i in range(1, samples + 1):
            t = i / samples
            pose = TipPose(
                x=lerp(start.x, end.x, t),
                y=lerp(start.y, end.y, t),
                z=lerp(start.z, end.z, t),
                tip_angle_deg=lerp(start.tip_angle_deg, end.tip_angle_deg, t),
                c_deg=wrap_near(lerp(start.c_deg, end.c_deg, t), prev_c),
            )
            prev_c = pose.c_deg
            self.move_tip(pose, feed, extrude=self.cfg.use_extrusion)
        self.add(f"; end {comment}")

    def footer(self) -> None:
        self.add()
        end_pose = TipPose(
            x=float(self.cfg.machine_end_x),
            y=float(self.cfg.machine_end_y),
            z=float(self.cfg.machine_end_z),
            tip_angle_deg=float(self.cfg.machine_end_b),
            c_deg=float(self.cfg.machine_end_c),
        )
        self.lift_and_travel_to_tip(end_pose, "return home")
        self.add("; done")

    def write(self) -> Path:
        path = Path(self.cfg.out)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(self.lines) + "\n", encoding="utf-8")
        return path


# ---------------------------- generation ----------------------------
def generate_branch(writer: GCodeWriter, cfg: Config, branch: str, phase1_planes: List[float], phase2_planes: List[float]) -> None:
    writer.add()
    writer.add(f"; ===== branch: {branch} =====")

    if phase1_planes:
        writer.add(f"; ===== phase 1: fixed-angle front sweep ({branch}) =====")
        first_front = build_front_circle(cfg, phase1_planes[0], branch)
        writer.lift_and_travel_to_tip(first_front[0], f"move to first front circle start ({branch})")
        writer.polyline(first_front, cfg.print_feed, f"front circle {branch} y={phase1_planes[0]:.3f}")
        for y in phase1_planes[1:]:
            circle = build_front_circle(cfg, y, branch)
            writer.lift_and_travel_to_tip(circle[0], f"move to front circle {branch} y={y:.3f}")
            writer.polyline(circle, cfg.print_feed, f"front circle {branch} y={y:.3f}")

    if phase2_planes:
        writer.add()
        writer.add(f"; ===== phase 2: side-oriented full sweep ({branch}) =====")
        current_circle = build_side_circle(cfg, phase2_planes[0], branch)
        writer.lift_and_travel_to_tip(current_circle[0], f"move to side circle {branch} y={phase2_planes[0]:.3f}")
        writer.polyline(current_circle, cfg.print_feed, f"side circle {branch} y={phase2_planes[0]:.3f}")

        for y in phase2_planes[1:]:
            next_circle = build_side_circle(cfg, y, branch)
            writer.tracked_bridge(
                start=current_circle[-1],
                end=next_circle[0],
                samples=cfg.bridge_samples,
                feed=cfg.bridge_feed,
                comment=f"tracked plane change and C reset ({branch}) to y={y:.3f}",
            )
            writer.polyline(next_circle, cfg.print_feed, f"side circle {branch} y={y:.3f}")
            current_circle = next_circle


def generate(cfg: Config) -> Path:
    write_mode = str(cfg.write_mode).strip().lower()
    if write_mode not in {"calibrated", "cartesian"}:
        raise ValueError("--write-mode must be calibrated or cartesian")

    branches = normalize_branch_selection(cfg.branches)

    cal: Optional[Calibration] = None
    if write_mode == "calibrated":
        if not cfg.calibration:
            raise ValueError("--calibration is required in calibrated mode")
        cal = load_calibration(str(cfg.calibration))
        cal.offplane_y_sign = float(cfg.y_offplane_sign)

    phase1_planes, phase2_planes = build_plane_sequences(cfg)

    writer = GCodeWriter(cfg, cal)
    writer.header()

    if cfg.use_extrusion and float(cfg.prime_mm) > 0.0:
        _, _, _, _, _, uax = writer.axis_names()
        writer.add(f"G1 {uax}{float(cfg.prime_mm):.5f} F300.0")
        writer.e_accum = float(cfg.prime_mm)

    start_pose = TipPose(
        x=float(cfg.machine_start_x),
        y=float(cfg.machine_start_y),
        z=float(cfg.machine_start_z),
        tip_angle_deg=float(cfg.machine_start_b),
        c_deg=float(cfg.machine_start_c),
    )
    writer.move_tip(start_pose, cfg.travel_feed, extrude=False, comment="machine start")

    for branch in branches:
        b_phase1, b_phase2 = filter_planes_for_branch(phase1_planes, phase2_planes, branch, branches)
        generate_branch(writer, cfg, branch, b_phase1, b_phase2)

    writer.footer()
    return writer.write()


# ---------------------------- CLI ----------------------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Generate calibrated torus skin G-code.")
    ap.add_argument("--calibration", type=str, default=None, help="Calibration JSON path. Required in calibrated mode.")
    ap.add_argument("--write-mode", choices=["calibrated", "cartesian"], default=DEFAULT_WRITE_MODE)
    ap.add_argument("--out", type=str, default=DEFAULT_OUT)
    ap.add_argument("--output", dest="out", type=str, help=argparse.SUPPRESS)

    ap.add_argument("--center-x", type=float, default=DEFAULT_CENTER_X)
    ap.add_argument("--center-y", type=float, default=DEFAULT_CENTER_Y)
    ap.add_argument("--center-z", type=float, default=DEFAULT_CENTER_Z)
    ap.add_argument("--centerline-diameter", type=float, default=DEFAULT_CENTERLINE_DIAMETER)
    ap.add_argument("--tube-diameter", type=float, default=DEFAULT_TUBE_DIAMETER)
    ap.add_argument("--plane-step", type=float, default=DEFAULT_PLANE_STEP, help="Y offset between planes, e.g. line thickness.")
    ap.add_argument("--branches", type=str, default=DEFAULT_BRANCHES, help="outer, inner, or both")

    ap.add_argument("--front-circle-samples", type=int, default=DEFAULT_FRONT_CIRCLE_SAMPLES)
    ap.add_argument("--side-half-samples", type=int, default=DEFAULT_SIDE_HALF_SAMPLES)
    ap.add_argument("--bridge-samples", type=int, default=DEFAULT_BRIDGE_SAMPLES)
    ap.add_argument("--bc-solve-samples", type=int, default=DEFAULT_BC_SOLVE_SAMPLES)
    ap.add_argument("--y-offplane-sign", type=float, default=DEFAULT_Y_OFFPLANE_SIGN)

    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED)
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED)
    ap.add_argument("--bridge-feed", type=float, default=DEFAULT_BRIDGE_FEED)
    ap.add_argument("--safe-z-lift", type=float, default=DEFAULT_SAFE_Z_LIFT)

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

    ap.add_argument("--use-extrusion", action="store_true")
    ap.add_argument("--extrusion-axis", type=str, default=DEFAULT_EXTRUSION_AXIS)
    ap.add_argument("--extrusion-per-mm", type=float, default=DEFAULT_EXTRUSION_PER_MM)
    ap.add_argument("--prime-mm", type=float, default=DEFAULT_PRIME_MM)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = Config(
        calibration=args.calibration,
        write_mode=args.write_mode,
        out=args.out,
        center_x=args.center_x,
        center_y=args.center_y,
        center_z=args.center_z,
        centerline_diameter=args.centerline_diameter,
        tube_diameter=args.tube_diameter,
        plane_step=args.plane_step,
        branches=args.branches,
        front_circle_samples=args.front_circle_samples,
        side_half_samples=args.side_half_samples,
        bridge_samples=args.bridge_samples,
        bc_solve_samples=args.bc_solve_samples,
        y_offplane_sign=args.y_offplane_sign,
        print_feed=args.print_feed,
        travel_feed=args.travel_feed,
        bridge_feed=args.bridge_feed,
        safe_z_lift=args.safe_z_lift,
        machine_start_x=args.machine_start_x,
        machine_start_y=args.machine_start_y,
        machine_start_z=args.machine_start_z,
        machine_start_b=args.machine_start_b,
        machine_start_c=args.machine_start_c,
        machine_end_x=args.machine_end_x,
        machine_end_y=args.machine_end_y,
        machine_end_z=args.machine_end_z,
        machine_end_b=args.machine_end_b,
        machine_end_c=args.machine_end_c,
        use_extrusion=args.use_extrusion,
        extrusion_axis=args.extrusion_axis,
        extrusion_per_mm=args.extrusion_per_mm,
        prime_mm=args.prime_mm,
    )
    out = generate(cfg)
    print(out)


if __name__ == "__main__":
    main()
