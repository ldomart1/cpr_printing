#!/usr/bin/env python3
"""
rotary_vase_gui_slicer.py

GUI-driven rotary vase G-code generator using the calibration-aware B/C
kinematics from the Klein bottle slicer.

Geometry implemented here
-------------------------
- A layer is traced by holding the XYZ gantry fixed and rotating C from
  C-center - 180 deg to C-center + 180 deg. The C sweep is emitted as
  intermediate segmented G1 commands rather than one endpoint-only move.
- B is selected from the calibration file so the calibrated tip radius matches
  the requested layer radius.
- The gantry Z for each layer is compensated by the calibration's tip Z offset
  so the physical tip height tracks the requested print height.
- The outer shell is printed first. Its radius is the inner radius profile plus
  a configurable offset, default 5 mm.
- A solid base is printed before the outer shell with concentric calibrated
  rotary rings.
- After the outer shell, the robot retracts upward, centers over the vase hole,
  moves down, and prints the inner shell bottom-to-top.
- Final rest move raises the stage to Z = -20 mm by default.

Dependencies
------------
Python 3.10+
NumPy, Matplotlib, Tkinter

Usage
-----
python3 rotary_vase_gui_slicer.py
python3 rotary_vase_gui_slicer.py --calibration path/to/calibration.json
python3 rotary_vase_gui_slicer.py --preset saved_params.json
"""

from __future__ import annotations

import argparse
import json
import math
import traceback
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

import matplotlib
try:
    matplotlib.use("TkAgg")
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
except Exception:  # pragma: no cover - desktop backend availability varies
    matplotlib.use("Agg")
    FigureCanvasTkAgg = None
from matplotlib.figure import Figure


# ============================================================================
# Defaults
# ============================================================================

DEFAULT_OUT = "gcode_generation/rotary_vase_outer_inner_shell.gcode"
DEFAULT_CALIBRATION_PATH = "Test_Calibration_2026-05-22_00/processed_image_data_folder/calibrated_robot_gcode_calibration.json"

# Machine / work placement
DEFAULT_CENTER_X = 100.0
DEFAULT_CENTER_Y = 0.0
DEFAULT_PRINT_BASE_Z = -120.0       # physical tip print Z at vase base
DEFAULT_SAFE_STAGE_Z = -20.0        # upward travel / rest Z in stage coordinates
DEFAULT_MACHINE_START_B = 0.0
DEFAULT_MACHINE_START_C = 0.0
DEFAULT_TRAVEL_B = 0.0
DEFAULT_C_CENTER = 0.0              # layer turn goes C_CENTER-180 -> C_CENTER+180

# Process
DEFAULT_TRAVEL_FEED = 1200.0
DEFAULT_STARTUP_TRAVEL_FEED = 2000.0
DEFAULT_TRANSITION_TRAVEL_FEED = 2000.0
DEFAULT_B_REORIENT_FEED = 400.0
DEFAULT_PRINT_FEED = 5000.0      # high-speed C-only print rotation feed
DEFAULT_UNTURN_FEED = 5000.0     # high-speed C-only unturn feed
DEFAULT_PREFLOW_DWELL_MS = 150
DEFAULT_END_DWELL_MS = 250
DEFAULT_PRESSURE_PIN = 0
DEFAULT_USE_PRESSURE_PIN = 1
DEFAULT_DEBUG_EVERY_N_LAYERS = 10

# Vase geometry
DEFAULT_INNER_HEIGHT = 70.0
DEFAULT_OUTER_OFFSET = 5.0
DEFAULT_OUTER_LAYER_HEIGHT = 2.0
DEFAULT_INNER_LAYER_HEIGHT = 0.5
DEFAULT_BASE_SOLID_HEIGHT = 2.0
DEFAULT_BASE_LAYER_HEIGHT = 1.0
DEFAULT_BASE_FILL_STEP = 1.5
DEFAULT_BASE_EXTRA_RADIUS = 0.0

# Inner radius profile default: 10 -> 15 -> 10 -> 12 mm over 80 mm height
DEFAULT_INNER_R_BASE = 10.0
DEFAULT_INNER_R_BULGE = 15.0
DEFAULT_INNER_R_NECK = 10.0
DEFAULT_INNER_R_TOP = 12.0
DEFAULT_FILLET_UP_Z = 15.0
DEFAULT_BACK_DOWN_Z = 45.0
DEFAULT_TOP_BLEND_START_Z = 65.0

# Preview / output controls
DEFAULT_PREVIEW_Z_SAMPLES = 180
DEFAULT_C_SEGMENTS = 96             # also used as C-command segmentation for G-code
DEFAULT_UNTURN_AFTER_EACH_LAYER = 1
DEFAULT_EFFECTIVE_RADIUS_USES_OFFPLANE = 1
DEFAULT_USE_SMOOTH_BASE_FILLET = 0
DEFAULT_INNER_DESCENT_TIP_ANGLE_DEG = 0.0
DEFAULT_PAUSE_BEFORE_INNER_MS = 10000
DEFAULT_BBOX_X_MIN = -9999.0
DEFAULT_BBOX_X_MAX = 9999.0
DEFAULT_BBOX_Y_MIN = -9999.0
DEFAULT_BBOX_Y_MAX = 9999.0
DEFAULT_BBOX_Z_MIN = -9999.0
DEFAULT_BBOX_Z_MAX = 9999.0

DEFAULT_OFFPLANE_SIGN = -1.0

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

    center_x: float = DEFAULT_CENTER_X
    center_y: float = DEFAULT_CENTER_Y
    print_base_z: float = DEFAULT_PRINT_BASE_Z
    safe_stage_z: float = DEFAULT_SAFE_STAGE_Z
    machine_start_b: float = DEFAULT_MACHINE_START_B
    machine_start_c: float = DEFAULT_MACHINE_START_C
    travel_b: float = DEFAULT_TRAVEL_B
    c_center_deg: float = DEFAULT_C_CENTER

    travel_feed: float = DEFAULT_TRAVEL_FEED
    startup_travel_feed: float = DEFAULT_STARTUP_TRAVEL_FEED
    transition_travel_feed: float = DEFAULT_TRANSITION_TRAVEL_FEED
    b_reorient_feed: float = DEFAULT_B_REORIENT_FEED
    print_feed: float = DEFAULT_PRINT_FEED
    outer_print_feed: float = DEFAULT_PRINT_FEED
    inner_print_feed: float = DEFAULT_PRINT_FEED
    unturn_feed: float = DEFAULT_UNTURN_FEED
    preflow_dwell_ms: int = DEFAULT_PREFLOW_DWELL_MS
    end_dwell_ms: int = DEFAULT_END_DWELL_MS
    pause_before_inner_ms: int = DEFAULT_PAUSE_BEFORE_INNER_MS
    pressure_pin: int = DEFAULT_PRESSURE_PIN
    use_pressure_pin: int = DEFAULT_USE_PRESSURE_PIN
    debug_every_n_layers: int = DEFAULT_DEBUG_EVERY_N_LAYERS

    inner_height: float = DEFAULT_INNER_HEIGHT
    outer_offset: float = DEFAULT_OUTER_OFFSET
    outer_layer_height: float = DEFAULT_OUTER_LAYER_HEIGHT
    inner_layer_height: float = DEFAULT_INNER_LAYER_HEIGHT
    base_solid_height: float = DEFAULT_BASE_SOLID_HEIGHT
    base_layer_height: float = DEFAULT_BASE_LAYER_HEIGHT
    base_fill_step: float = DEFAULT_BASE_FILL_STEP
    base_extra_radius: float = DEFAULT_BASE_EXTRA_RADIUS

    inner_r_base: float = DEFAULT_INNER_R_BASE
    inner_r_bulge: float = DEFAULT_INNER_R_BULGE
    inner_r_neck: float = DEFAULT_INNER_R_NECK
    inner_r_top: float = DEFAULT_INNER_R_TOP
    fillet_up_z: float = DEFAULT_FILLET_UP_Z
    back_down_z: float = DEFAULT_BACK_DOWN_Z
    top_blend_start_z: float = DEFAULT_TOP_BLEND_START_Z

    preview_z_samples: int = DEFAULT_PREVIEW_Z_SAMPLES
    c_segments: int = DEFAULT_C_SEGMENTS
    unturn_after_each_layer: int = DEFAULT_UNTURN_AFTER_EACH_LAYER
    effective_radius_uses_offplane: int = DEFAULT_EFFECTIVE_RADIUS_USES_OFFPLANE
    use_smooth_base_fillet: int = DEFAULT_USE_SMOOTH_BASE_FILLET

    bbox_x_min: float = DEFAULT_BBOX_X_MIN
    bbox_x_max: float = DEFAULT_BBOX_X_MAX
    bbox_y_min: float = DEFAULT_BBOX_Y_MIN
    bbox_y_max: float = DEFAULT_BBOX_Y_MAX
    bbox_z_min: float = DEFAULT_BBOX_Z_MIN
    bbox_z_max: float = DEFAULT_BBOX_Z_MAX


@dataclass
class BRadiusInversionResult:
    b_cmd: float
    radius_actual: float
    z_offset: float
    y_off: float
    clamped: bool


# ============================================================================
# Numeric helpers and calibration model evaluation
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


# ============================================================================
# Calibration / rotary kinematics
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
    y_off_extrap_model = _select_first_model(model_sources, "offplane_y_avg_linear", "offplane_y_linear", "offplane_y")

    pr = (np.asarray(cubic["r_coeffs"], dtype=float) if cubic.get("r_coeffs") is not None
          else (_coeffs_from_models(active_models, "r_cubic", "r_avg_cubic") or np.zeros(1)))
    pz = (np.asarray(cubic["z_coeffs"], dtype=float) if cubic.get("z_coeffs") is not None
          else (_coeffs_from_models(active_models, "z_cubic", "z_avg_cubic") or np.zeros(1)))

    py_off = _coeffs_from_models(active_models, "offplane_y_avg_cubic", "offplane_y_cubic", "offplane_y", "offplane_y_linear", "offplane_y_avg_linear")
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


def tip_offset_xyz_physical(cal: Calibration, b_cmd: float, c_deg: float) -> np.ndarray:
    """Physical tip offset from the XYZ gantry center at B/C."""
    r = float(eval_r(cal, b_cmd))
    z = float(eval_z(cal, b_cmd))
    y_off = float(eval_offplane_y(cal, b_cmd))
    c = math.radians(float(c_deg))
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def eval_tip_angle_deg(cal: Calibration, b: Any) -> np.ndarray:
    if cal.tip_angle_model is not None:
        return eval_model_spec(cal.tip_angle_model, b)
    if cal.pa is None:
        raise ValueError("Calibration missing tip_angle_coeffs / tip_angle model.")
    return poly_eval(cal.pa, b)


def solve_b_for_target_tip_angle(cal: Calibration, target_angle_deg: float, search_samples: int = 20001) -> float:
    bb = np.linspace(float(cal.b_min), float(cal.b_max), int(max(101, search_samples)))
    aa = np.asarray(eval_tip_angle_deg(cal, bb), dtype=float) - float(target_angle_deg)
    i_best = int(np.argmin(np.abs(aa)))
    b_best = float(bb[i_best])

    crossings = np.flatnonzero(aa[:-1] * aa[1:] < 0.0)
    if crossings.size:
        # Prefer the higher-B branch when multiple tip-angle solutions exist.
        i = int(crossings[-1])
        lo = float(bb[i])
        hi = float(bb[i + 1])
        flo = float(eval_tip_angle_deg(cal, lo) - float(target_angle_deg))
        for _ in range(80):
            mid = 0.5 * (lo + hi)
            fmid = float(eval_tip_angle_deg(cal, mid) - float(target_angle_deg))
            if abs(fmid) < 1e-10:
                return float(mid)
            if flo * fmid <= 0.0:
                hi = mid
            else:
                lo, flo = mid, fmid
        return float(0.5 * (lo + hi))
    return b_best


def effective_radius(cal: Calibration, b_cmd: Any, use_offplane: bool = True) -> np.ndarray:
    r = np.asarray(eval_r(cal, b_cmd), dtype=float)
    if not use_offplane:
        return r
    y = np.asarray(eval_offplane_y(cal, b_cmd), dtype=float)
    return np.sqrt(r * r + y * y)


class BRadiusInverter:
    """Approximate inverse from desired physical rotary radius to B command."""

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

        # Prefer the branch with the smallest physical tip angle, then the
        # higher B value if two candidates have effectively the same angle.
        b_cmd = float(
            sorted(
                candidates,
                key=lambda b: (abs(float(eval_tip_angle_deg(self.cal, b))), -float(b)),
            )[0]
        )
        actual = float(effective_radius(self.cal, b_cmd, self.use_offplane))
        zoff = float(eval_z(self.cal, b_cmd))
        yoff = float(eval_offplane_y(self.cal, b_cmd))
        return BRadiusInversionResult(b_cmd=b_cmd, radius_actual=actual, z_offset=zoff, y_off=yoff, clamped=clamped)


# ============================================================================
# Vase geometry
# ============================================================================

def inner_radius_at_z(z_mm: float, p: Params) -> float:
    """Smooth 10 -> 15 -> 10 -> 12 mm profile by default."""
    h = max(1e-6, float(p.inner_height))
    z = clamp(float(z_mm), 0.0, h)

    z1 = clamp(float(p.fillet_up_z), 0.0, h)
    z2 = clamp(float(p.back_down_z), z1, h)
    z3 = clamp(float(p.top_blend_start_z), z2, h)

    r0 = max(0.0, float(p.inner_r_base))
    r1 = max(0.0, float(p.inner_r_bulge))
    r2 = max(0.0, float(p.inner_r_neck))
    r3 = max(0.0, float(p.inner_r_top))

    if z1 > 1e-9 and z <= z1:
        s = z / z1
        if int(getattr(p, "use_smooth_base_fillet", 0)):
            # Quarter-fillet style rise: horizontal tangent at the base,
            # vertical tangent where it blends into the main body.
            a = math.sqrt(max(0.0, 2.0 * s - s * s))
        else:
            a = smootherstep(s)
        return (1.0 - a) * r0 + a * r1
    if z2 > z1 + 1e-9 and z <= z2:
        a = smootherstep((z - z1) / (z2 - z1))
        return (1.0 - a) * r1 + a * r2
    if z <= z3 or z3 >= h - 1e-9:
        return r2
    a = smootherstep((z - z3) / max(1e-9, h - z3))
    return (1.0 - a) * r2 + a * r3


def _inner_profile_samples(p: Params, n: int = 1201) -> Tuple[np.ndarray, np.ndarray]:
    h = max(1e-6, float(p.inner_height))
    count = int(max(101, n))
    zs = np.linspace(0.0, h, count, dtype=float)
    rs = np.array([inner_radius_at_z(float(z), p) for z in zs], dtype=float)
    return zs, rs


def outer_profile_curve(p: Params, n: int = 1201) -> Tuple[np.ndarray, np.ndarray]:
    d = max(0.0, float(p.outer_offset))
    zs, rs = _inner_profile_samples(p, n=n)
    if d <= 1e-12:
        return zs, rs

    dr_dz = np.gradient(rs, zs, edge_order=2)
    norm = np.sqrt(1.0 + dr_dz * dr_dz)

    # Offset the (radius, z) centerline by distance d along the outward normal.
    # Positive radial normal component grows the vase radius; the z component
    # shifts the outer shell above/below the inner profile as needed.
    r_off = rs + d / norm
    z_off = zs - d * dr_dz / norm

    order = np.argsort(z_off)
    z_sorted = np.asarray(z_off[order], dtype=float)
    r_sorted = np.asarray(r_off[order], dtype=float)
    z_unique, idx = np.unique(z_sorted, return_index=True)
    r_unique = r_sorted[idx]
    return z_unique, r_unique


def outer_height_range(p: Params) -> Tuple[float, float]:
    z_outer, _ = outer_profile_curve(p)
    return float(z_outer[0]), float(z_outer[-1])


def outer_radius_at_z(z_mm: float, p: Params) -> float:
    z_outer, r_outer = outer_profile_curve(p)
    z = clamp(float(z_mm), float(z_outer[0]), float(z_outer[-1]))
    return float(np.interp(z, z_outer, r_outer))


def layer_values(height: float, step: float, include_top: bool = True) -> List[float]:
    h = max(0.0, float(height))
    dz = max(1e-6, float(step))
    vals = list(np.arange(0.0, h + 0.5 * dz, dz, dtype=float))
    if include_top and (not vals or abs(vals[-1] - h) > 1e-6):
        vals.append(h)
    return [float(clamp(v, 0.0, h)) for v in vals]


def layer_values_between(z0: float, z1: float, step: float, include_ends: bool = True) -> List[float]:
    lo = float(min(z0, z1))
    hi = float(max(z0, z1))
    dz = max(1e-6, float(step))
    vals = list(np.arange(lo, hi + 0.5 * dz, dz, dtype=float))
    if include_ends:
        if not vals or abs(vals[0] - lo) > 1e-6:
            vals.insert(0, lo)
        else:
            vals[0] = lo
        if abs(vals[-1] - hi) > 1e-6:
            vals.append(hi)
        else:
            vals[-1] = hi
    return [float(v) for v in vals]


def shell_layer_values(which: str, p: Params) -> List[float]:
    if which == "outer":
        z0, z1 = outer_height_range(p)
        return layer_values_between(z0, z1, float(p.outer_layer_height), include_ends=True)
    if which == "inner":
        return layer_values(float(p.inner_height), float(p.inner_layer_height), include_top=True)
    raise ValueError(which)

def preview_points_for_shell(which: str, p: Params) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_z = int(max(8, p.preview_z_samples))
    n_c = int(max(16, p.c_segments))
    if which == "outer":
        z0, z1 = outer_height_range(p)
        zs = np.linspace(z0, z1, n_z)
    else:
        zs = np.linspace(0.0, float(p.inner_height), n_z)
    cs = np.linspace(-math.pi, math.pi, n_c)
    X = np.zeros((n_z, n_c), dtype=float)
    Y = np.zeros_like(X)
    Z = np.zeros_like(X)
    for i, z in enumerate(zs):
        r = outer_radius_at_z(float(z), p) if which == "outer" else inner_radius_at_z(float(z), p)
        X[i, :] = float(p.center_x) + r * np.cos(cs)
        Y[i, :] = float(p.center_y) + r * np.sin(cs)
        Z[i, :] = float(p.print_base_z) + z
    return X, Y, Z

def ring_points_at_z(radius: float, tip_z: float, p: Params, n: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Renderer helper for a full C-180..C+180 rotary ring at fixed XYZ center."""
    count = int(max(16, n if n is not None else p.c_segments))
    theta = np.linspace(math.radians(float(p.c_center_deg) - 180.0), math.radians(float(p.c_center_deg) + 180.0), count + 1)
    r = max(0.0, float(radius))
    x = float(p.center_x) + r * np.cos(theta)
    y = float(p.center_y) + r * np.sin(theta)
    z = np.full_like(x, float(tip_z), dtype=float)
    return x, y, z


def seam_line_points(which: str, p: Params, n: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Renderer helper for the C-start/C-end seam meridian."""
    if which == "outer":
        z0, z1 = outer_height_range(p)
        zs = np.linspace(z0, z1, int(max(2, n)))
    else:
        zs = np.linspace(0.0, float(p.inner_height), int(max(2, n)))
    theta = math.radians(float(p.c_center_deg) - 180.0)
    radii = np.array([outer_radius_at_z(float(z), p) if which == "outer" else inner_radius_at_z(float(z), p) for z in zs])
    x = float(p.center_x) + radii * math.cos(theta)
    y = float(p.center_y) + radii * math.sin(theta)
    z = float(p.print_base_z) + zs
    return x, y, z


def layer_stride_for_preview(layer_count: int, target_count: int = 90) -> int:
    return int(max(1, math.ceil(max(1, int(layer_count)) / max(1, int(target_count)))))


# ============================================================================
# G-code writer
# ============================================================================

def _fmt_axes_move(axes_vals: Sequence[Tuple[str, float]]) -> str:
    return " ".join(f"{ax}{float(val):.3f}" for ax, val in axes_vals)


def params_bbox(p: Params) -> Dict[str, float]:
    return {
        "x_min": float(min(p.bbox_x_min, p.bbox_x_max)),
        "x_max": float(max(p.bbox_x_min, p.bbox_x_max)),
        "y_min": float(min(p.bbox_y_min, p.bbox_y_max)),
        "y_max": float(max(p.bbox_y_min, p.bbox_y_max)),
        "z_min": float(min(p.bbox_z_min, p.bbox_z_max)),
        "z_max": float(max(p.bbox_z_min, p.bbox_z_max)),
    }


def _clamp_stage_xyz_to_bbox(x: float, y: float, z: float, bbox: Dict[str, float], context: str, warn_log: List[str]) -> Tuple[float, float, float]:
    def clamp_one(axis: str, value: float, lo: float, hi: float) -> float:
        if value < lo:
            warn_log.append(f"WARNING: {context} {axis}={value:.3f} below bbox min {lo:.3f}; clamped to {lo:.3f}")
            return lo
        if value > hi:
            warn_log.append(f"WARNING: {context} {axis}={value:.3f} above bbox max {hi:.3f}; clamped to {hi:.3f}")
            return hi
        return value

    return (
        clamp_one("X", float(x), float(bbox["x_min"]), float(bbox["x_max"])),
        clamp_one("Y", float(y), float(bbox["y_min"]), float(bbox["y_max"])),
        clamp_one("Z", float(z), float(bbox["z_min"]), float(bbox["z_max"])),
    )


class RotaryVaseGCodeWriter:
    def __init__(self, fh, cal: Calibration, p: Params, warn_log: List[str]):
        self.f = fh
        self.cal = cal
        self.p = p
        self.warn_log = warn_log
        self.bbox = params_bbox(p)
        self.cur_stage = np.array([float(p.center_x), float(p.center_y), float(p.safe_stage_z)], dtype=float)
        self.cur_b = float(p.machine_start_b)
        self.cur_c = float(p.machine_start_c)
        self.pressure_on_state = False
        self.layer_count = 0

    @property
    def c_start(self) -> float:
        return float(self.p.c_center_deg) - 180.0

    @property
    def c_end(self) -> float:
        return float(self.p.c_center_deg) + 180.0

    def write_comment(self, text: str) -> None:
        self.f.write(f"; {text}\n")

    def move_stage_xyzbc(self, x: float, y: float, z: float, b_cmd: float, c_deg: float, feed: float, comment: str = "") -> None:
        if comment:
            self.write_comment(comment)
        xc, yc, zc = _clamp_stage_xyz_to_bbox(x, y, z, self.bbox, comment or "move", self.warn_log)
        axes = [
            (self.cal.x_axis, xc),
            (self.cal.y_axis, yc),
            (self.cal.z_axis, zc),
            (self.cal.b_axis, float(b_cmd)),
            (self.cal.c_axis, float(c_deg)),
        ]
        self.f.write(f"G1 {_fmt_axes_move(axes)} F{float(feed):.0f}\n")
        self.cur_stage = np.array([xc, yc, zc], dtype=float)
        self.cur_b = float(b_cmd)
        self.cur_c = float(c_deg)

    def move_c_sweep_segmented(
        self,
        x: float,
        y: float,
        z: float,
        b_cmd: float,
        c0_deg: float,
        c1_deg: float,
        feed: float,
        comment: str = "",
    ) -> None:
        """Emit a fixed-XYZ/B rotary C sweep using intermediate G1 commands.

        This is intentionally not a single endpoint move. Some Duet/RRF setups
        stream rotary motion more predictably when the C move is broken into
        explicit intermediate commands, especially with pressure on.
        """
        if comment:
            self.write_comment(comment)
        xc, yc, zc = _clamp_stage_xyz_to_bbox(x, y, z, self.bbox, comment or "C sweep", self.warn_log)
        n = int(max(1, self.p.c_segments))
        for i in range(1, n + 1):
            a = i / n
            c_deg = (1.0 - a) * float(c0_deg) + a * float(c1_deg)
            axes = [
                (self.cal.x_axis, xc),
                (self.cal.y_axis, yc),
                (self.cal.z_axis, zc),
                (self.cal.b_axis, float(b_cmd)),
                (self.cal.c_axis, float(c_deg)),
            ]
            self.f.write(f"G1 {_fmt_axes_move(axes)} F{float(feed):.0f} ; C segment {i}/{n}\n")
        self.cur_stage = np.array([xc, yc, zc], dtype=float)
        self.cur_b = float(b_cmd)
        self.cur_c = float(c1_deg)

    def pressure_on(self) -> None:
        if int(self.p.use_pressure_pin) and not self.pressure_on_state:
            self.f.write("; pressure on\n")
            self.f.write(f"M42 P{int(self.p.pressure_pin)} S1\n")
            if int(self.p.preflow_dwell_ms) > 0:
                self.f.write(f"G4 P{int(self.p.preflow_dwell_ms)}\n")
            self.pressure_on_state = True

    def pressure_off(self) -> None:
        if int(self.p.use_pressure_pin) and self.pressure_on_state:
            if int(self.p.end_dwell_ms) > 0:
                self.f.write(f"G4 P{int(self.p.end_dwell_ms)}\n")
            self.f.write("; pressure off\n")
            self.f.write(f"M42 P{int(self.p.pressure_pin)} S0\n")
            self.pressure_on_state = False

    def stage_z_for_tip_layer(self, layer_z: float, inv: BRadiusInversionResult) -> float:
        return float(self.p.print_base_z) + float(layer_z) - float(inv.z_offset)

    def travel_to_safe_center(
        self,
        b_cmd: Optional[float] = None,
        c_deg: Optional[float] = None,
        feed: Optional[float] = None,
        comment: str = "safe center travel",
    ) -> None:
        self.pressure_off()
        move_feed = float(self.p.travel_feed if feed is None else feed)
        self.move_stage_xyzbc(
            float(self.cur_stage[0]),
            float(self.cur_stage[1]),
            float(self.p.safe_stage_z),
            float(self.cur_b if b_cmd is None else b_cmd),
            float(self.cur_c if c_deg is None else c_deg),
            move_feed,
            comment=f"{comment}: raise to safe Z",
        )
        self.move_stage_xyzbc(
            float(self.p.center_x),
            float(self.p.center_y),
            float(self.p.safe_stage_z),
            float(self.cur_b if b_cmd is None else b_cmd),
            float(self.cur_c if c_deg is None else c_deg),
            move_feed,
            comment=f"{comment}: center XY over hole",
        )

    def print_rotary_layer(
        self,
        name: str,
        layer_z: float,
        target_radius: float,
        inverter: BRadiusInverter,
        section: str,
        radius_index: Optional[int] = None,
        reverse_direction: bool = False,
        print_feed: Optional[float] = None,
    ) -> float:
        inv = inverter.radius_to_b(float(target_radius))
        stage_z = self.stage_z_for_tip_layer(layer_z, inv)
        self.layer_count += 1
        debug_n = int(max(0, self.p.debug_every_n_layers))
        if debug_n == 0 or self.layer_count == 1 or (self.layer_count % debug_n == 0):
            clamp_msg = " CLAMPED" if inv.clamped else ""
            idx_msg = "" if radius_index is None else f" r_index={radius_index}"
            self.write_comment(
                f"{name}: section={section} layer_z={layer_z:.3f} target_radius={target_radius:.3f} "
                f"B={inv.b_cmd:.4f} actual_radius={inv.radius_actual:.3f} tip_z_offset={inv.z_offset:.3f}{idx_msg}{clamp_msg}"
            )
        if inv.clamped:
            self.warn_log.append(
                f"{name}: requested radius {target_radius:.3f} mm outside calibration radius range "
                f"[{inverter.r_min:.3f}, {inverter.r_max:.3f}] mm; used {inv.radius_actual:.3f} mm."
            )

        c0 = self.c_end if reverse_direction else self.c_start
        c1 = self.c_start if reverse_direction else self.c_end

        # Travel to start with XYZ held at the layer's center/stage Z, then rotate C while extruding.
        self.move_stage_xyzbc(
            float(self.p.center_x),
            float(self.p.center_y),
            stage_z,
            inv.b_cmd,
            c0,
            float(self.p.travel_feed),
            comment=f"travel to {name} start {'C+180' if reverse_direction else 'C-180'}",
        )

        self.pressure_on()
        self.move_c_sweep_segmented(
            float(self.p.center_x),
            float(self.p.center_y),
            stage_z,
            inv.b_cmd,
            c0,
            c1,
            float(self.p.print_feed if print_feed is None else print_feed),
            comment=f"PRINT {name}: xyz constant, segmented {'C+180 to C-180' if reverse_direction else 'C-180 to C+180'}",
        )
        self.pressure_off()
        return 2.0 * math.pi * float(inv.radius_actual)


# ============================================================================
# Export
# ============================================================================

def sanitize_params(p: Params) -> Params:
    q = Params(**asdict(p))
    q.center_x = float(q.center_x)
    q.center_y = float(q.center_y)
    q.print_base_z = float(q.print_base_z)
    q.safe_stage_z = float(q.safe_stage_z)
    q.machine_start_b = float(q.machine_start_b)
    q.machine_start_c = float(q.machine_start_c)
    q.travel_b = float(q.travel_b)
    q.c_center_deg = float(q.c_center_deg)

    q.travel_feed = max(1.0, float(q.travel_feed))
    q.startup_travel_feed = max(1.0, float(q.startup_travel_feed))
    q.transition_travel_feed = max(1.0, float(q.transition_travel_feed))
    q.b_reorient_feed = max(1.0, float(q.b_reorient_feed))
    q.print_feed = max(1.0, float(q.print_feed))
    q.outer_print_feed = max(1.0, float(q.outer_print_feed))
    q.inner_print_feed = max(1.0, float(q.inner_print_feed))
    q.unturn_feed = max(1.0, float(q.unturn_feed))
    q.preflow_dwell_ms = int(max(0, q.preflow_dwell_ms))
    q.end_dwell_ms = int(max(0, q.end_dwell_ms))
    q.pause_before_inner_ms = int(max(0, q.pause_before_inner_ms))
    q.pressure_pin = int(max(0, q.pressure_pin))
    q.use_pressure_pin = 1 if int(q.use_pressure_pin) else 0
    q.debug_every_n_layers = int(max(0, q.debug_every_n_layers))

    q.inner_height = max(0.1, float(q.inner_height))
    q.outer_offset = max(0.0, float(q.outer_offset))
    q.outer_layer_height = max(0.01, float(q.outer_layer_height))
    q.inner_layer_height = max(0.01, float(q.inner_layer_height))
    q.base_solid_height = max(0.0, float(q.base_solid_height))
    q.base_layer_height = max(0.01, float(q.base_layer_height))
    q.base_fill_step = max(0.01, float(q.base_fill_step))
    q.base_extra_radius = max(0.0, float(q.base_extra_radius))

    q.inner_r_base = max(0.0, float(q.inner_r_base))
    q.inner_r_bulge = max(0.0, float(q.inner_r_bulge))
    q.inner_r_neck = max(0.0, float(q.inner_r_neck))
    q.inner_r_top = max(0.0, float(q.inner_r_top))
    q.fillet_up_z = clamp(float(q.fillet_up_z), 0.0, q.inner_height)
    q.back_down_z = clamp(float(q.back_down_z), q.fillet_up_z, q.inner_height)
    q.top_blend_start_z = clamp(float(q.top_blend_start_z), q.back_down_z, q.inner_height)

    q.preview_z_samples = int(max(8, q.preview_z_samples))
    q.c_segments = int(max(16, q.c_segments))
    q.unturn_after_each_layer = 1 if int(q.unturn_after_each_layer) else 0
    q.effective_radius_uses_offplane = 1 if int(q.effective_radius_uses_offplane) else 0
    q.use_smooth_base_fillet = 1 if int(q.use_smooth_base_fillet) else 0
    return q


def params_from_mapping(data: Dict[str, Any], base: Optional[Params] = None) -> Params:
    merged = asdict(base or Params())
    allowed = {f.name for f in fields(Params)}
    for key, value in data.items():
        if key in allowed:
            merged[key] = value
    if "outer_print_feed" not in data and "print_feed" in data:
        merged["outer_print_feed"] = data["print_feed"]
    if "inner_print_feed" not in data and "print_feed" in data:
        merged["inner_print_feed"] = data["print_feed"]
    return sanitize_params(Params(**merged))


def export_gcode(params: Params, out_path: str) -> Dict[str, Any]:
    p = sanitize_params(params)
    if not p.calibration_path:
        raise ValueError("Calibration JSON path is required for G-code export.")
    cal = load_calibration(p.calibration_path)
    inverter = BRadiusInverter(cal, use_offplane=bool(p.effective_radius_uses_offplane))
    warn_log: List[str] = []
    total_print_len = 0.0
    total_layers = 0

    out = Path(out_path)
    if out.parent and str(out.parent) != ".":
        out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        f.write("; generated by rotary_vase_gui_slicer.py\n")
        f.write("; geometry: layer = fixed XYZ gantry + rotary C sweep, alternating sweep direction on successive layers\n")
        f.write("; B command selected by calibrated radius inverse; stage Z compensated by calibration tip Z offset\n")
        f.write("; path order: outer shell bottom-to-top -> safe center/down -> inner shell bottom-to-top -> rest Z\n")
        f.write(f"; axes: X->{cal.x_axis}, Y->{cal.y_axis}, Z->{cal.z_axis}, B->{cal.b_axis}, C->{cal.c_axis}\n")
        f.write(f"; calibration B range: [{cal.b_min:.6f}, {cal.b_max:.6f}]\n")
        f.write(f"; calibration radius range used for inversion: [{inverter.r_min:.3f}, {inverter.r_max:.3f}] mm; offplane_comp={int(p.effective_radius_uses_offplane)}\n")
        f.write(f"; active calibration phase={cal.active_phase}; offplane_model={cal.active_offplane_fit_model or cal.selected_offplane_fit_model or cal.selected_fit_model or 'none'}\n")
        f.write(f"; center gantry XY=({p.center_x:.3f}, {p.center_y:.3f}); print_base_tip_Z={p.print_base_z:.3f}; safe/rest_stage_Z={p.safe_stage_z:.3f}\n")
        f.write(f"; C turn: start={p.c_center_deg - 180.0:.3f}, end={p.c_center_deg + 180.0:.3f}, center={p.c_center_deg:.3f}\n")
        f.write(f"; inner height={p.inner_height:.3f}; outer_offset={p.outer_offset:.3f}; outer_layer_height={p.outer_layer_height:.3f}; inner_layer_height={p.inner_layer_height:.3f}\n")
        base_mode = "smooth_fillet" if int(p.use_smooth_base_fillet) else "default_smootherstep"
        f.write(f"; inner radius profile: mode={base_mode}, base={p.inner_r_base:.3f}, bulge={p.inner_r_bulge:.3f} at z={p.fillet_up_z:.3f}, neck={p.inner_r_neck:.3f} at z={p.back_down_z:.3f}, top={p.inner_r_top:.3f} from z={p.top_blend_start_z:.3f} to top\n")
        f.write(
            f"; feeds: startup_travel={p.startup_travel_feed:.1f}, transition_travel={p.transition_travel_feed:.1f}, "
            f"travel={p.travel_feed:.1f}, B-reorient={p.b_reorient_feed:.1f}, "
            f"outer_C_print={p.outer_print_feed:.1f}, inner_C_print={p.inner_print_feed:.1f}; C_segments={int(p.c_segments)}\n"
        )
        f.write(f"; pause_before_inner_ms={int(p.pause_before_inner_ms)}\n")
        f.write("G90\n")
        if int(p.use_pressure_pin):
            f.write(f"M42 P{int(p.pressure_pin)} S0\n")

        g = RotaryVaseGCodeWriter(f, cal, p, warn_log)

        # Startup / approach.
        g.move_stage_xyzbc(
            float(p.center_x),
            float(p.center_y),
            float(p.safe_stage_z),
            float(p.machine_start_b),
            float(p.machine_start_c),
            float(p.startup_travel_feed),
            comment="startup: move to centered safe/rest Z",
        )
        g.move_stage_xyzbc(
            float(p.center_x),
            float(p.center_y),
            float(p.safe_stage_z),
            float(p.travel_b),
            g.c_start,
            float(p.b_reorient_feed),
            comment="startup: set travel B and C-180 start",
        )

        # Outer shell first, bottom-to-top, alternating C sweep direction by layer.
        outer_zs = shell_layer_values("outer", p)
        for i, z in enumerate(outer_zs):
            r = outer_radius_at_z(float(z), p)
            total_print_len += g.print_rotary_layer(
                name=f"outer_shell_z{z:.3f}",
                layer_z=float(z),
                target_radius=float(r),
                inverter=inverter,
                section="outer_shell",
                reverse_direction=bool(i % 2),
                print_feed=float(p.outer_print_feed),
            )
            total_layers += 1

        # Retract, center over the hole, open to a safe physical tip angle while elevated,
        # then descend straight down into the inner-shell start region.
        g.travel_to_safe_center(feed=float(p.transition_travel_feed), comment="after outer shell")
        if int(p.pause_before_inner_ms) > 0:
            f.write(f"; pause before inner shell\nG4 P{int(p.pause_before_inner_ms)}\n")
        inner_descent_b = solve_b_for_target_tip_angle(cal, DEFAULT_INNER_DESCENT_TIP_ANGLE_DEG)
        g.move_stage_xyzbc(
            float(p.center_x),
            float(p.center_y),
            float(p.safe_stage_z),
            float(inner_descent_b),
            g.c_start,
            float(p.b_reorient_feed),
            comment=f"after outer shell: set B for {DEFAULT_INNER_DESCENT_TIP_ANGLE_DEG:.1f} deg tip angle at safe Z before inner descent",
        )
        inner0 = inverter.radius_to_b(inner_radius_at_z(0.0, p))
        inner0_stage_z = g.stage_z_for_tip_layer(0.0, inner0)
        g.move_stage_xyzbc(
            float(p.center_x),
            float(p.center_y),
            inner0_stage_z,
            inner0.b_cmd,
            g.c_start,
            float(p.transition_travel_feed),
            comment="descend in centered hole to inner-shell base",
        )

        # Inner shell second, bottom-to-top, alternating C sweep direction by layer.
        inner_zs = shell_layer_values("inner", p)
        for i, z in enumerate(inner_zs):
            r = inner_radius_at_z(float(z), p)
            total_print_len += g.print_rotary_layer(
                name=f"inner_shell_z{z:.3f}",
                layer_z=float(z),
                target_radius=float(r),
                inverter=inverter,
                section="inner_shell",
                reverse_direction=bool(i % 2),
                print_feed=float(p.inner_print_feed),
            )
            total_layers += 1

        # Final rest: pressure off, raise to Z=-20 by default, optionally reset B/C.
        g.pressure_off()
        g.move_stage_xyzbc(
            float(p.center_x),
            float(p.center_y),
            float(p.safe_stage_z),
            float(p.travel_b),
            float(p.c_center_deg),
            float(p.travel_feed),
            comment="end: move up to rest Z",
        )

        if warn_log:
            f.write("; warnings:\n")
            for msg in warn_log:
                f.write(f"; {msg}\n")
        f.write(f"; total printed rotary rings = {total_layers}\n")
        f.write(f"; approximate printed length = {total_print_len:.3f} mm\n")
        f.write(f"; warning count = {len(warn_log)}\n")
        f.write("; --- end ---\n")

    return {
        "out_path": str(out),
        "warning_count": len(warn_log),
        "warnings": warn_log,
        "printed_ring_count": total_layers,
        "approx_print_length_mm": total_print_len,
        "b_radius_min_mm": inverter.r_min,
        "b_radius_max_mm": inverter.r_max,
    }


# ============================================================================
# GUI
# ============================================================================

PARAM_GROUPS: List[Tuple[str, List[Dict[str, Any]]]] = [
    (
        "Paths",
        [],
    ),
    (
        "Placement / Motion",
        [
            {"key": "center_x", "label": "Center X", "from_": -100.0, "to": 250.0, "resolution": 0.5, "step": 0.5},
            {"key": "center_y", "label": "Center Y", "from_": -100.0, "to": 250.0, "resolution": 0.5, "step": 0.5},
            {"key": "print_base_z", "label": "Print Base Tip Z", "from_": -250.0, "to": 50.0, "resolution": 0.5, "step": 0.5},
            {"key": "safe_stage_z", "label": "Safe / Rest Stage Z", "from_": -250.0, "to": 50.0, "resolution": 0.5, "step": 0.5},
            {"key": "travel_b", "label": "Travel B", "from_": -20.0, "to": 20.0, "resolution": 0.05, "step": 0.05},
            {"key": "c_center_deg", "label": "C Center Deg", "from_": -720.0, "to": 720.0, "resolution": 1.0, "step": 1.0},
        ],
    ),
    (
        "Shell Heights / Layers",
        [
            {"key": "inner_height", "label": "Inner Height", "from_": 1.0, "to": 200.0, "resolution": 0.5, "step": 0.5},
            {"key": "outer_offset", "label": "Outer Offset", "from_": 0.0, "to": 25.0, "resolution": 0.1, "step": 0.1},
            {"key": "outer_layer_height", "label": "Outer Layer Height", "from_": 0.1, "to": 10.0, "resolution": 0.05, "step": 0.05},
            {"key": "inner_layer_height", "label": "Inner Layer Height", "from_": 0.05, "to": 5.0, "resolution": 0.05, "step": 0.05},
            {"key": "base_solid_height", "label": "Solid Base Height", "from_": 0.0, "to": 30.0, "resolution": 0.1, "step": 0.1},
            {"key": "base_layer_height", "label": "Base Layer Height", "from_": 0.1, "to": 5.0, "resolution": 0.05, "step": 0.05},
            {"key": "base_fill_step", "label": "Base Fill Step", "from_": 0.1, "to": 10.0, "resolution": 0.05, "step": 0.05},
            {"key": "base_extra_radius", "label": "Base Extra Radius", "from_": 0.0, "to": 20.0, "resolution": 0.1, "step": 0.1},
        ],
    ),
    (
        "Inner Radius Profile",
        [
            {"key": "inner_r_base", "label": "Base Radius", "from_": 0.0, "to": 60.0, "resolution": 0.1, "step": 0.1},
            {"key": "inner_r_bulge", "label": "Fillet/Bulge Radius", "from_": 0.0, "to": 80.0, "resolution": 0.1, "step": 0.1},
            {"key": "inner_r_neck", "label": "Back Down Radius", "from_": 0.0, "to": 60.0, "resolution": 0.1, "step": 0.1},
            {"key": "inner_r_top", "label": "Top Radius", "from_": 0.0, "to": 60.0, "resolution": 0.1, "step": 0.1},
            {"key": "fillet_up_z", "label": "Fillet Up Z", "from_": 0.0, "to": 200.0, "resolution": 0.5, "step": 0.5},
            {"key": "back_down_z", "label": "Back Down Z", "from_": 0.0, "to": 200.0, "resolution": 0.5, "step": 0.5},
            {"key": "top_blend_start_z", "label": "Top Blend Start Z", "from_": 0.0, "to": 200.0, "resolution": 0.5, "step": 0.5},
        ],
    ),
    (
        "Process",
        [
            {"key": "travel_feed", "label": "XYZ/B Travel Feed", "from_": 10.0, "to": 5000.0, "resolution": 10.0, "step": 10.0},
            {"key": "startup_travel_feed", "label": "Startup Travel Feed", "from_": 10.0, "to": 5000.0, "resolution": 10.0, "step": 10.0},
            {"key": "transition_travel_feed", "label": "Transition Travel Feed", "from_": 10.0, "to": 5000.0, "resolution": 10.0, "step": 10.0},
            {"key": "b_reorient_feed", "label": "B Reorient Feed", "from_": 10.0, "to": 5000.0, "resolution": 10.0, "step": 10.0},
            {"key": "outer_print_feed", "label": "Outer C Print Feed", "from_": 10.0, "to": 12000.0, "resolution": 10.0, "step": 10.0},
            {"key": "inner_print_feed", "label": "Inner C Print Feed", "from_": 10.0, "to": 12000.0, "resolution": 10.0, "step": 10.0},
            {"key": "unturn_feed", "label": "C Unturn Feed", "from_": 10.0, "to": 12000.0, "resolution": 10.0, "step": 10.0},
            {"key": "preflow_dwell_ms", "label": "Preflow Dwell ms", "from_": 0, "to": 2000, "resolution": 10, "step": 10},
            {"key": "end_dwell_ms", "label": "End Dwell ms", "from_": 0, "to": 2000, "resolution": 10, "step": 10},
            {"key": "pause_before_inner_ms", "label": "Pause Before Inner ms", "from_": 0, "to": 60000, "resolution": 100, "step": 100},
            {"key": "pressure_pin", "label": "Pressure Pin", "from_": 0, "to": 99, "resolution": 1, "step": 1},
            {"key": "c_segments", "label": "C Command/Preview Segments", "from_": 8, "to": 720, "resolution": 1, "step": 1},
            {"key": "preview_z_samples", "label": "Preview Z Samples", "from_": 16, "to": 600, "resolution": 1, "step": 1},
            {"key": "debug_every_n_layers", "label": "Debug Every N Layers", "from_": 0, "to": 100, "resolution": 1, "step": 1},
        ],
    ),
]


class ParameterControl(ttk.Frame):
    def __init__(self, master, spec: Dict[str, Any], initial_value: Any, on_change, set_active):
        super().__init__(master)
        self.spec = spec
        self.key = spec["key"]
        self.step = spec.get("step", spec.get("resolution", 1.0))
        self.on_change = on_change
        self.set_active = set_active
        self.is_int = int(spec.get("resolution", 1)) == 1 and isinstance(initial_value, int)
        self.columnconfigure(2, weight=1)

        ttk.Label(self, text=spec["label"], width=24).grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Button(self, text="◀", width=3, command=lambda: self.bump(-1)).grid(row=0, column=1, padx=2)

        self.var = tk.IntVar(value=int(initial_value)) if self.is_int else tk.DoubleVar(value=float(initial_value))
        self.scale = tk.Scale(
            self,
            from_=spec["from_"],
            to=spec["to"],
            resolution=spec["resolution"],
            orient="horizontal",
            variable=self.var,
            showvalue=False,
            command=self._scale_changed,
            length=280,
            bg=UI_PANEL_BG,
            fg=UI_FG,
            troughcolor=UI_PANEL_ALT_BG,
            highlightthickness=0,
            activebackground=UI_ACCENT,
        )
        self.scale.grid(row=0, column=2, sticky="ew", padx=2)
        self.scale.bind("<Button-1>", lambda e: self.set_active(self))
        self.scale.bind("<FocusIn>", lambda e: self.set_active(self))

        ttk.Button(self, text="▶", width=3, command=lambda: self.bump(+1)).grid(row=0, column=3, padx=2)
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
            value = int(round(float(self.entry.get().strip()))) if self.is_int else float(self.entry.get().strip())
        except Exception:
            self.entry.delete(0, tk.END)
            self.entry.insert(0, self._fmt_value(self.var.get()))
            return
        value = clamp(float(value), float(self.spec["from_"]), float(self.spec["to"]))
        self.var.set(int(round(value)) if self.is_int else float(value))
        self._scale_changed()

    def bump(self, direction: int):
        self.set_active(self)
        value = float(self.var.get()) + direction * float(self.step)
        value = clamp(value, float(self.spec["from_"]), float(self.spec["to"]))
        self.var.set(int(round(value)) if self.is_int else float(value))
        self._scale_changed()

    def get_value(self):
        return int(self.var.get()) if self.is_int else float(self.var.get())

    def set_value(self, value: Any):
        numeric = int(round(float(value))) if self.is_int else float(value)
        self.var.set(numeric)
        self.entry.delete(0, tk.END)
        self.entry.insert(0, self._fmt_value(self.var.get()))


class RotaryVaseApp:
    def __init__(self, root: tk.Tk, initial_params: Params):
        self.root = root
        self.params = sanitize_params(initial_params)
        self.controls: Dict[str, ParameterControl] = {}
        self.active_control: Optional[ParameterControl] = None
        self.redraw_after_id: Optional[str] = None
        self.toggle_buttons: List[tk.Checkbutton] = []

        self.root.title("Rotary Vase GUI Slicer")
        self.root.geometry("1660x920")
        self.configure_style()
        self.build_ui()
        self.schedule_redraw(delay_ms=50)

    def configure_style(self):
        self.root.configure(bg=UI_BG)
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TFrame", background=UI_BG)
        style.configure("TLabelframe", background=UI_BG, foreground=UI_FG, bordercolor=UI_BORDER)
        style.configure("TLabelframe.Label", background=UI_BG, foreground=UI_ACCENT)
        style.configure("TLabel", background=UI_BG, foreground=UI_FG)
        style.configure("TButton", background=UI_PANEL_ALT_BG, foreground=UI_FG, bordercolor=UI_BORDER)
        style.configure("TEntry", fieldbackground=UI_ENTRY_BG, foreground=UI_FG, insertcolor=UI_FG)
        style.configure("TCheckbutton", background=UI_BG, foreground=UI_FG)

    def build_ui(self):
        self.root.columnconfigure(0, weight=1, minsize=760)
        self.root.columnconfigure(1, weight=1, minsize=760)
        self.root.rowconfigure(0, weight=1)

        left = ttk.Frame(self.root)
        left.grid(row=0, column=0, sticky="nsew", padx=8, pady=8)
        left.columnconfigure(0, weight=1)
        right = ttk.Frame(self.root)
        right.grid(row=0, column=1, sticky="nsew", padx=(0, 8), pady=8)
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        # Scrollable control panel
        canvas = tk.Canvas(left, width=760, bg=UI_BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(left, orient="vertical", command=canvas.yview)
        self.control_frame = ttk.Frame(canvas)
        self.control_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.control_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.grid(row=0, column=0, sticky="nsew")
        scrollbar.grid(row=0, column=1, sticky="ns")
        left.rowconfigure(0, weight=1)
        self.control_frame.columnconfigure(0, weight=1)

        row = 0
        # Paths group
        path_group = ttk.LabelFrame(self.control_frame, text="Paths")
        path_group.grid(row=row, column=0, sticky="ew", padx=4, pady=6)
        path_group.columnconfigure(1, weight=1)
        row += 1

        ttk.Label(path_group, text="Calibration JSON").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.cal_path_var = tk.StringVar(value=self.params.calibration_path)
        ttk.Entry(path_group, textvariable=self.cal_path_var, width=54).grid(row=0, column=1, sticky="ew", padx=4, pady=3)
        ttk.Button(path_group, text="Browse", command=self.browse_calibration).grid(row=0, column=2, padx=4, pady=3)

        ttk.Label(path_group, text="Output G-code").grid(row=1, column=0, sticky="w", padx=4, pady=3)
        self.out_path_var = tk.StringVar(value=self.params.output_path)
        ttk.Entry(path_group, textvariable=self.out_path_var, width=54).grid(row=1, column=1, sticky="ew", padx=4, pady=3)
        ttk.Button(path_group, text="Save As", command=self.browse_output).grid(row=1, column=2, padx=4, pady=3)

        for group_name, specs in PARAM_GROUPS[1:]:
            lf = ttk.LabelFrame(self.control_frame, text=group_name)
            lf.grid(row=row, column=0, sticky="ew", padx=4, pady=6)
            lf.columnconfigure(0, weight=1)
            row += 1
            for r, spec in enumerate(specs):
                ctrl = ParameterControl(lf, spec, getattr(self.params, spec["key"]), self.on_param_change, self.set_active_control)
                ctrl.grid(row=r, column=0, sticky="ew", padx=4, pady=2)
                self.controls[spec["key"]] = ctrl

        toggle_group = ttk.LabelFrame(self.control_frame, text="Toggles")
        toggle_group.grid(row=row, column=0, sticky="ew", padx=4, pady=6)
        toggle_group.columnconfigure(0, weight=1)
        row += 1
        self.unturn_var = tk.IntVar(value=int(self.params.unturn_after_each_layer))
        self.toggle_buttons.append(
            tk.Checkbutton(
                toggle_group,
                text="Unturn after each layer",
                variable=self.unturn_var,
                command=self.on_toggle_change,
                anchor="w",
                justify="left",
                bg=UI_BG,
                fg=UI_FG,
                activebackground=UI_BG,
                activeforeground=UI_FG,
                selectcolor=UI_PANEL_ALT_BG,
                highlightthickness=0,
                bd=0,
            )
        )
        self.toggle_buttons[-1].grid(row=0, column=0, sticky="ew", padx=4, pady=3)
        self.offplane_var = tk.IntVar(value=int(self.params.effective_radius_uses_offplane))
        self.toggle_buttons.append(
            tk.Checkbutton(
                toggle_group,
                text="Use off-plane Y in radius inversion",
                variable=self.offplane_var,
                command=self.on_toggle_change,
                anchor="w",
                justify="left",
                bg=UI_BG,
                fg=UI_FG,
                activebackground=UI_BG,
                activeforeground=UI_FG,
                selectcolor=UI_PANEL_ALT_BG,
                highlightthickness=0,
                bd=0,
            )
        )
        self.toggle_buttons[-1].grid(row=1, column=0, sticky="ew", padx=4, pady=3)
        self.pressure_var = tk.IntVar(value=int(self.params.use_pressure_pin))
        self.toggle_buttons.append(
            tk.Checkbutton(
                toggle_group,
                text="Use M42 pressure pin",
                variable=self.pressure_var,
                command=self.on_toggle_change,
                anchor="w",
                justify="left",
                bg=UI_BG,
                fg=UI_FG,
                activebackground=UI_BG,
                activeforeground=UI_FG,
                selectcolor=UI_PANEL_ALT_BG,
                highlightthickness=0,
                bd=0,
            )
        )
        self.toggle_buttons[-1].grid(row=2, column=0, sticky="ew", padx=4, pady=3)
        self.base_fillet_var = tk.IntVar(value=int(self.params.use_smooth_base_fillet))
        self.toggle_buttons.append(
            tk.Checkbutton(
                toggle_group,
                text="Smooth base fillet (horizontal tangent at base)",
                variable=self.base_fillet_var,
                command=self.on_toggle_change,
                anchor="w",
                justify="left",
                bg=UI_BG,
                fg=UI_FG,
                activebackground=UI_BG,
                activeforeground=UI_FG,
                selectcolor=UI_PANEL_ALT_BG,
                highlightthickness=0,
                bd=0,
            )
        )
        self.toggle_buttons[-1].grid(row=3, column=0, sticky="ew", padx=4, pady=3)
        toggle_group.bind("<Configure>", self._resize_toggle_wrap)

        button_group = ttk.Frame(self.control_frame)
        button_group.grid(row=row, column=0, sticky="ew", padx=4, pady=10)
        ttk.Button(button_group, text="Update Preview", command=lambda: self.schedule_redraw(delay_ms=1)).grid(row=0, column=0, padx=4, pady=4)
        ttk.Button(button_group, text="Export G-code", command=self.export_clicked).grid(row=0, column=1, padx=4, pady=4)
        ttk.Button(button_group, text="Save JSON", command=self.save_preset).grid(row=0, column=2, padx=4, pady=4)
        ttk.Button(button_group, text="Load JSON", command=self.load_preset).grid(row=0, column=3, padx=4, pady=4)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(self.control_frame, textvariable=self.status_var, foreground=UI_MUTED_FG).grid(row=row + 1, column=0, sticky="ew", padx=4, pady=(0, 8))

        self.fig = Figure(figsize=(8.8, 7.4), dpi=100, facecolor=UI_PLOT_BG)
        gs = self.fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0], wspace=0.18)
        self.ax3d = self.fig.add_subplot(gs[0, 0], projection="3d", facecolor=UI_PLOT_BG)
        self.ax2d = self.fig.add_subplot(gs[0, 1], facecolor=UI_PLOT_BG)
        if FigureCanvasTkAgg is None:
            raise RuntimeError("TkAgg backend unavailable; run this script in a desktop Python with Tk support.")
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

        self.root.bind("<Left>", lambda e: self.bump_active(-1))
        self.root.bind("<Right>", lambda e: self.bump_active(+1))

    def set_active_control(self, ctrl: ParameterControl):
        self.active_control = ctrl

    def _resize_toggle_wrap(self, event=None):
        if event is None:
            return
        wrap = max(180, int(event.width) - 24)
        for btn in self.toggle_buttons:
            btn.configure(wraplength=wrap)

    def bump_active(self, direction: int):
        if self.active_control is not None:
            self.active_control.bump(direction)

    def gather_params(self) -> Params:
        data = asdict(self.params)
        for key, ctrl in self.controls.items():
            data[key] = ctrl.get_value()
        data["calibration_path"] = self.cal_path_var.get().strip()
        data["output_path"] = self.out_path_var.get().strip()
        data["unturn_after_each_layer"] = int(self.unturn_var.get())
        data["effective_radius_uses_offplane"] = int(self.offplane_var.get())
        data["use_pressure_pin"] = int(self.pressure_var.get())
        data["use_smooth_base_fillet"] = int(self.base_fillet_var.get())
        return sanitize_params(Params(**data))

    def apply_params(self, params: Params):
        self.params = sanitize_params(params)
        for key, ctrl in self.controls.items():
            ctrl.set_value(getattr(self.params, key))
        self.cal_path_var.set(self.params.calibration_path)
        self.out_path_var.set(self.params.output_path)
        self.unturn_var.set(int(self.params.unturn_after_each_layer))
        self.offplane_var.set(int(self.params.effective_radius_uses_offplane))
        self.pressure_var.set(int(self.params.use_pressure_pin))
        self.base_fillet_var.set(int(self.params.use_smooth_base_fillet))
        self.schedule_redraw(delay_ms=1)

    def on_param_change(self, _key: str, _value: Any):
        self.params = self.gather_params()
        self.schedule_redraw(delay_ms=120)

    def on_toggle_change(self):
        self.params = self.gather_params()
        self.schedule_redraw(delay_ms=60)

    def schedule_redraw(self, delay_ms: int = 100):
        if self.redraw_after_id is not None:
            try:
                self.root.after_cancel(self.redraw_after_id)
            except Exception:
                pass
        self.redraw_after_id = self.root.after(delay_ms, self.redraw)

    def redraw(self):
        self.redraw_after_id = None
        self.params = self.gather_params()
        p = self.params
        self.ax3d.clear()
        self.ax2d.clear()
        for ax in (self.ax3d, self.ax2d):
            ax.set_facecolor(UI_PLOT_BG)
            ax.tick_params(colors=UI_MUTED_FG)
            for spine in ax.spines.values():
                spine.set_color(UI_BORDER)
        for axis in (self.ax3d.xaxis, self.ax3d.yaxis, self.ax3d.zaxis):
            axis.set_pane_color((0.0, 0.0, 0.0, 0.0))
        self.ax3d.grid(True, color=UI_GRID, alpha=0.25)
        self.ax2d.grid(True, color=UI_GRID, alpha=0.25)

        Xo, Yo, Zo = preview_points_for_shell("outer", p)
        Xi, Yi, Zi = preview_points_for_shell("inner", p)

        # Full geometry renderer: translucent surfaces + layer rings + base fill.
        rstride = max(1, Xo.shape[0] // 90)
        cstride = max(1, Xo.shape[1] // 90)
        try:
            self.ax3d.plot_surface(Xo, Yo, Zo, rstride=rstride, cstride=cstride, alpha=0.16, linewidth=0.0, antialiased=True)
            self.ax3d.plot_surface(Xi, Yi, Zi, rstride=rstride, cstride=cstride, alpha=0.10, linewidth=0.0, antialiased=True)
        except Exception:
            # Very old Matplotlib builds can be finicky with alpha on 3D surfaces;
            # fall back to wire-only rendering instead of failing the GUI.
            pass

        # Wireframe rings and meridians for both shells.
        for arrX, arrY, arrZ in [(Xo, Yo, Zo), (Xi, Yi, Zi)]:
            step_z = max(1, arrX.shape[0] // 28)
            step_c = max(1, arrX.shape[1] // 16)
            for i in range(0, arrX.shape[0], step_z):
                self.ax3d.plot(arrX[i, :], arrY[i, :], arrZ[i, :], linewidth=0.6, alpha=0.70)
            for j in range(0, arrX.shape[1], step_c):
                self.ax3d.plot(arrX[:, j], arrY[:, j], arrZ[:, j], linewidth=0.5, alpha=0.55)

        # Actual exported layer rings. These match the C command segmentation count.
        outer_zs = shell_layer_values("outer", p)
        inner_zs = shell_layer_values("inner", p)
        outer_stride = layer_stride_for_preview(len(outer_zs), target_count=80)
        inner_stride = layer_stride_for_preview(len(inner_zs), target_count=120)
        for z in outer_zs[::outer_stride]:
            x, y, zz = ring_points_at_z(outer_radius_at_z(float(z), p), float(p.print_base_z) + float(z), p)
            self.ax3d.plot(x, y, zz, linewidth=1.0, alpha=0.85)
        for z in inner_zs[::inner_stride]:
            x, y, zz = ring_points_at_z(inner_radius_at_z(float(z), p), float(p.print_base_z) + float(z), p)
            self.ax3d.plot(x, y, zz, linewidth=0.8, alpha=0.75)

        # Seam meridian: C start and C end are geometrically coincident but mark
        # where every rotary layer begins/ends.
        for which in ("outer", "inner"):
            sx, sy, sz = seam_line_points(which, p)
            self.ax3d.plot(sx, sy, sz, linestyle="--", linewidth=1.0, alpha=0.75)

        # 2D radius profile / command-layer renderer.
        zs = np.linspace(0.0, float(p.inner_height), int(max(10, p.preview_z_samples)))
        inner_rs = np.array([inner_radius_at_z(float(z), p) for z in zs])
        outer_zs_curve, outer_rs_curve = outer_profile_curve(p, n=int(max(201, p.preview_z_samples * 4)))
        self.ax2d.plot(inner_rs, zs, label="inner surface")
        self.ax2d.plot(outer_rs_curve, outer_zs_curve, label="outer surface (parallel offset)")
        for z in outer_zs:
            self.ax2d.axhline(z, alpha=0.075, linewidth=0.45)
        for z in inner_zs[::max(1, layer_stride_for_preview(len(inner_zs), 50))]:
            self.ax2d.axhline(z, alpha=0.04, linewidth=0.35)
        self.ax2d.set_xlabel("Radius from rotary center (mm)", color=UI_FG)
        self.ax2d.set_ylabel("Height above base (mm)", color=UI_FG)
        self.ax2d.set_title("Vase geometry and exported layer heights", color=UI_FG)
        self.ax2d.set_aspect("equal", adjustable="box")
        self.ax2d.legend(facecolor=UI_PANEL_ALT_BG, edgecolor=UI_BORDER, labelcolor=UI_FG)

        # Small text readout embedded in renderer.
        c0 = float(p.c_center_deg) - 180.0
        c1 = float(p.c_center_deg) + 180.0
        txt = (
            f"C sweep: {c0:.1f}° → {c1:.1f}° in {int(p.c_segments)} commands\n"
            f"C feeds: outer {float(p.outer_print_feed):.0f}, inner {float(p.inner_print_feed):.0f}\n"
            f"outer dz {float(p.outer_layer_height):.3f} mm, inner dz {float(p.inner_layer_height):.3f} mm"
        )
        self.ax2d.text(0.02, 0.02, txt, transform=self.ax2d.transAxes, va="bottom", ha="left", color=UI_MUTED_FG)

        self.ax3d.set_xlabel("X", color=UI_FG)
        self.ax3d.set_ylabel("Y", color=UI_FG)
        self.ax3d.set_zlabel("Tip Z", color=UI_FG)
        self.ax3d.set_title("3D GUI geometry renderer: outer → inner", color=UI_FG)
        self._set_3d_equalish(Xo, Yo, Zo)
        self.fig.tight_layout()
        self.canvas.draw_idle()

        outer_layers = len(outer_zs)
        inner_layers = len(inner_zs)
        self.status_var.set(
            f"Renderer: outer layers={outer_layers}, inner layers={inner_layers}, "
            f"C commands/layer={int(p.c_segments)}, outer/inner feeds={float(p.outer_print_feed):.0f}/{float(p.inner_print_feed):.0f}"
        )

    def _set_3d_equalish(self, X: np.ndarray, Y: np.ndarray, Z: np.ndarray) -> None:
        xs = X.reshape(-1)
        ys = Y.reshape(-1)
        zs = Z.reshape(-1)
        cx = 0.5 * (float(xs.min()) + float(xs.max()))
        cy = 0.5 * (float(ys.min()) + float(ys.max()))
        cz = 0.5 * (float(zs.min()) + float(zs.max()))
        dx = max(float(xs.max() - xs.min()), 1.0)
        dy = max(float(ys.max() - ys.min()), 1.0)
        dz = max(float(zs.max() - zs.min()), 1.0)
        span_xy = max(dx, dy) * 0.55
        span_z = dz * 0.55
        self.ax3d.set_xlim(cx - span_xy, cx + span_xy)
        self.ax3d.set_ylim(cy - span_xy, cy + span_xy)
        self.ax3d.set_zlim(cz - span_z, cz + span_z)
        self.ax3d.set_box_aspect((1.0, 1.0, dz / max(dx, dy)))

    def browse_calibration(self):
        path = filedialog.askopenfilename(title="Select calibration JSON", filetypes=[("JSON", "*.json"), ("All files", "*")])
        if path:
            self.cal_path_var.set(path)
            self.on_toggle_change()

    def browse_output(self):
        path = filedialog.asksaveasfilename(title="Output G-code", defaultextension=".gcode", filetypes=[("G-code", "*.gcode *.g *.nc"), ("All files", "*")])
        if path:
            self.out_path_var.set(path)
            self.on_toggle_change()

    def save_preset(self):
        try:
            self.params = self.gather_params()
            path = filedialog.asksaveasfilename(title="Save preset JSON", defaultextension=".json", filetypes=[("JSON", "*.json"), ("All files", "*")])
            if not path:
                return
            with open(path, "w", encoding="utf-8") as f:
                json.dump(asdict(self.params), f, indent=2)
            self.status_var.set(f"Saved preset: {path}")
        except Exception as exc:
            messagebox.showerror("Save preset failed", f"{exc}\n\n{traceback.format_exc()}")

    def load_preset(self):
        try:
            path = filedialog.askopenfilename(title="Load preset JSON", filetypes=[("JSON", "*.json"), ("All files", "*")])
            if not path:
                return
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.apply_params(params_from_mapping(data, base=self.params))
            self.status_var.set(f"Loaded preset: {path}")
        except Exception as exc:
            messagebox.showerror("Load preset failed", f"{exc}\n\n{traceback.format_exc()}")

    def export_clicked(self):
        try:
            self.params = self.gather_params()
            out_path = self.params.output_path or DEFAULT_OUT
            result = export_gcode(self.params, out_path)
            warn_msg = f"\nWarnings: {result['warning_count']}" if result["warning_count"] else ""
            self.status_var.set(f"Exported: {result['out_path']}{warn_msg}")
            messagebox.showinfo(
                "Export complete",
                f"Wrote:\n{result['out_path']}\n\n"
                f"Printed rings: {result['printed_ring_count']}\n"
                f"Approx print length: {result['approx_print_length_mm']:.2f} mm\n"
                f"Calibration radius range: {result['b_radius_min_mm']:.2f} to {result['b_radius_max_mm']:.2f} mm\n"
                f"Warnings: {result['warning_count']}",
            )
        except Exception as exc:
            self.status_var.set(f"Export failed: {exc}")
            messagebox.showerror("Export failed", f"{exc}\n\n{traceback.format_exc()}")


# ============================================================================
# CLI / main
# ============================================================================

def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Interactive rotary vase G-code generator.")
    ap.add_argument("--calibration", default="", help="Path to calibration JSON.")
    ap.add_argument("--preset", default="", help="Optional preset JSON to load on startup.")
    ap.add_argument("--export", default="", help="Optional output path. If set, export immediately without GUI.")
    return ap.parse_args(argv)


def load_startup_params(args: argparse.Namespace) -> Params:
    p = Params()
    if args.preset:
        with open(args.preset, "r", encoding="utf-8") as f:
            data = json.load(f)
        p = params_from_mapping(data, base=p)
    if args.calibration:
        p.calibration_path = args.calibration
    if args.export:
        p.output_path = args.export
    return sanitize_params(p)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    params = load_startup_params(args)
    if args.export:
        result = export_gcode(params, params.output_path)
        print(json.dumps(result, indent=2))
        return 0
    if FigureCanvasTkAgg is None:
        raise RuntimeError("TkAgg backend is unavailable; run on a desktop Python with Tk support.")
    root = tk.Tk()
    app = RotaryVaseApp(root, initial_params=params)
    root.mainloop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
