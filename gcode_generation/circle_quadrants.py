#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code for tracing a calibrated circle in the XZ plane.

Path convention
---------------
The printed circle lives in tip space at constant Y:
  X = center_x + radius * cos(theta)
  Y = center_y
  Z = center_z + radius * sin(theta)

Quadrants are printed in this order:
  1: leftmost  -> bottommost, C=180, B physical target 180 -> 90,  phase pull
  2: bottom    -> rightmost,  C=180, B physical target  90 -> 0,   phase pull
     stop extrusion, then fixed-tip C spin C=180 -> C=0 at the rightmost point
  3: rightmost -> topmost,    C=0,   B physical target   0 -> 90,  phase release
  4: top       -> leftmost,   C=0,   B physical target  90 ->180,  phase release

"B physical target" uses the calibration tip_angle model to solve the actual
Duet B motor command in calibrated mode.  Cartesian mode emits the physical B
angle directly and does no point-offset compensation.

Fit strategy
------------
--fit-strategy phase-pchip: q1/q2 use release PCHIP; q3/q4 use pull PCHIP.
--fit-strategy avg-pchip:   all tracking/B solves use average PCHIP models.
--fit-strategy avg-cubic:   all tracking/B solves use average cubic models.
--fit-strategy avg-linear:  all tracking/B solves use average linear models.

Phase branch modes
------------------
--phase-branch-mode half-circle:
    use the 0-180-0 branch mapping when available, but allow fallback.
--phase-branch-mode curl-specific-0-180-0:
    require the calibration to contain curl-specific 0-180-0 pull/release
    models, and use release PCHIP for the 180->0 first half of the circle then
    pull PCHIP for the 0->180 second half.
--phase-branch-mode quadrant-split:
    split the circle into 90-degree curl groups.

The off-plane Y compensation can be selected separately; the default is
--offplane-fit-model avg_cubic, unless --fit-strategy avg-linear is used, in
which case the default becomes avg_linear so all B-pull equations match.

Feedrate modes
--------------
Default is G94 units/min.  With --feedrate-mode inverse-time the file enters
G93 at the start and emits F as inverse minutes for every G1 move, then returns
to G94 at the end.

Example
-------
python circle_trace_generator.py \
  --calibration robot_calibration.json \
  --center-x 80 --center-y 52 --center-z -120 --radius 25 \
  --quadrants 1,2,3,4 \
  --fit-strategy phase-pchip \
  --feedrate-mode inverse-time \
  --out circle.gcode
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np


# ---------------- Defaults ----------------
DEFAULT_OUT = "gcode_generation/circle_trace_xz.gcode"

DEFAULT_CENTER_X = 100.0
DEFAULT_CENTER_Y = 52.0
DEFAULT_CENTER_Z = -130.0
DEFAULT_RADIUS = 20.0
DEFAULT_POINTS_PER_MM = 8.0
DEFAULT_EDGE_SAMPLES = 1

DEFAULT_WRITE_MODE = "calibrated"
DEFAULT_FIT_STRATEGY = "phase-pchip"
DEFAULT_PHASE_BRANCH_MODE = "half-circle"
DEFAULT_OFFPLANE_FIT_MODEL = "avg_cubic"
DEFAULT_Y_OFFPLANE_SIGN = -1.0
DEFAULT_BC_SOLVE_SAMPLES = 5001
DEFAULT_PHASE_TRANSITION_STEPS = 12

DEFAULT_TRAVEL_FEED = 1000.0
DEFAULT_APPROACH_FEED = 400.0
DEFAULT_FINE_APPROACH_FEED = 80.0
DEFAULT_PRINT_FEED = 200.0
DEFAULT_C_SPIN_FEED = 10000.0
DEFAULT_SPIN_STEPS = 24
DEFAULT_TRAVEL_LIFT_Z = 5.0
DEFAULT_END_LEFT_MM = 5.0
DEFAULT_FIRST_INVERSE_F = 60.0

DEFAULT_EMIT_EXTRUSION = True
DEFAULT_PREFLOW_DWELL_MS = 500

DEFAULT_BBOX_X_MIN = -1e9
DEFAULT_BBOX_X_MAX = 1e9
DEFAULT_BBOX_Y_MIN = -1e9
DEFAULT_BBOX_Y_MAX = 1e9
DEFAULT_BBOX_Z_MIN = -1e9
DEFAULT_BBOX_Z_MAX = 1e9


# ---------------- Calibration data ----------------
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

    fit_models: Dict[str, Any]
    phase_models: Dict[str, Dict[str, Any]]
    curl_angle_models: Dict[str, Dict[str, Dict[str, Any]]]
    phase_aliases: Dict[str, str]
    selected_fit_model: Optional[str] = None
    selected_offplane_fit_model: Optional[str] = None
    offplane_y_sign: float = 1.0
    default_motion_phase: str = "pull"
    derived_model_cache: Dict[Tuple[str, str, str], Optional[Dict[str, Any]]] = field(default_factory=dict)


# ---------------- General helpers ----------------
def _clean_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower().replace("-", "_")
    return text or None


def _fmt_float(value: float, decimals: int = 5) -> str:
    text = f"{float(value):.{decimals}f}"
    text = text.rstrip("0").rstrip(".")
    if text == "-0":
        return "0"
    return text


def _fmt_axes_move(axes_vals: Sequence[Tuple[str, float]]) -> str:
    return " ".join(f"{ax}{val:.3f}" for ax, val in axes_vals)


def normalize(v: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    arr = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(arr))
    if n <= eps:
        return np.zeros_like(arr)
    return arr / n


def polyline_length(points: np.ndarray) -> float:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return 0.0
    return float(np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1)))


def deduplicate_polyline_points(points: np.ndarray, tol: float = 1e-9) -> np.ndarray:
    pts = np.asarray(points, dtype=float)
    if len(pts) <= 1:
        return pts.copy()
    out = [pts[0].copy()]
    for p in pts[1:]:
        if float(np.linalg.norm(p - out[-1])) > float(tol):
            out.append(np.asarray(p, dtype=float).copy())
    return np.asarray(out, dtype=float)


# ---------------- PCHIP / model evaluation ----------------
def poly_eval(coeffs: Any, u: Any, default_if_none: Optional[float] = None) -> np.ndarray:
    u_arr = np.asarray(u, dtype=float)
    if coeffs is None:
        if default_if_none is None:
            raise ValueError("Missing required polynomial coefficients.")
        return np.full_like(u_arr, float(default_if_none), dtype=float)
    coeff_arr = np.asarray(coeffs, dtype=float).reshape(-1)
    if coeff_arr.size == 0:
        if default_if_none is None:
            raise ValueError("Polynomial coefficients array is empty.")
        return np.full_like(u_arr, float(default_if_none), dtype=float)
    return np.polyval(coeff_arr, u_arr)


def _normalize_model_spec(model_spec: Any) -> Optional[Dict[str, Any]]:
    if not isinstance(model_spec, dict):
        return None
    out = dict(model_spec)
    if out.get("model_type") is not None:
        out["model_type"] = str(out["model_type"]).strip().lower()
    return out


def _selector_variants(selector: Optional[str]) -> List[str]:
    sel = _clean_name(selector)
    if not sel:
        return []
    out = [sel]
    if sel == "pchip":
        out += ["avg_pchip"]
    elif sel == "cubic":
        out += ["avg_cubic"]
    elif sel == "linear":
        out += ["avg_linear"]
    return list(dict.fromkeys(out))


def _select_named_model(models: Dict[str, Any], base_name: str, selector: Optional[str]) -> Optional[Dict[str, Any]]:
    base = _clean_name(base_name) or str(base_name)
    candidates: List[str] = []
    for sel in _selector_variants(selector):
        candidates.append(f"{base}_{sel}")
    candidates.append(base)
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
    t = np.clip((flat - x0) / h_i, 0.0, 1.0)

    h00 = 2.0 * t**3 - 3.0 * t**2 + 1.0
    h10 = t**3 - 2.0 * t**2 + t
    h01 = -2.0 * t**3 + 3.0 * t**2
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
    raise ValueError(f"Unsupported calibration model_type: {model_type!r}")


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


def _normalize_phase_aliases(data: Dict[str, Any]) -> Dict[str, str]:
    aliases: Dict[str, str] = {}
    for raw_key in ("phase_aliases", "motion_phase_map"):
        payload = data.get(raw_key) or {}
        if not isinstance(payload, dict):
            continue
        for src, dst in payload.items():
            src_key = _clean_name(src)
            dst_key = _clean_name(dst)
            if src_key and dst_key:
                aliases[src_key] = dst_key
    return aliases


def _extract_phase_models(data: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], str]:
    phase_payload = data.get("fit_models_by_phase") or {}
    phase_models: Dict[str, Dict[str, Any]] = {}
    if isinstance(phase_payload, dict):
        for raw_phase_name, models in phase_payload.items():
            phase_name = _clean_name(raw_phase_name)
            if phase_name and isinstance(models, dict):
                phase_models[phase_name] = dict(models)

    default_phase = _clean_name(data.get("default_phase_for_legacy_access"))
    if default_phase is None or default_phase not in phase_models:
        if "pull" in phase_models:
            default_phase = "pull"
        elif phase_models:
            default_phase = next(iter(phase_models))
        else:
            default_phase = "pull"
    return phase_models, default_phase


def _extract_curl_angle_models(data: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    payload = data.get("curl_angle_specific_fit_models") or {}
    out: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if not isinstance(payload, dict):
        return out
    for raw_group_name, group_payload in payload.items():
        group_name = _clean_name(raw_group_name)
        if not group_name or not isinstance(group_payload, dict):
            continue
        fit_models_by_phase = group_payload.get("fit_models_by_phase") or {}
        if not isinstance(fit_models_by_phase, dict):
            continue
        phase_map: Dict[str, Dict[str, Any]] = {}
        for raw_phase_name, models in fit_models_by_phase.items():
            phase_name = _clean_name(raw_phase_name)
            if phase_name and isinstance(models, dict):
                phase_map[phase_name] = dict(models)
        if phase_map:
            out[group_name] = phase_map
    return out


def resolve_phase_name(cal: Calibration, phase_name: Optional[str]) -> Optional[str]:
    want = _clean_name(phase_name)
    if want is None:
        return None
    aliases = cal.phase_aliases or {}
    want = aliases.get(want, want)
    if want in (cal.phase_models or {}):
        return want
    phase_models = cal.phase_models or {}
    prefix = [k for k in phase_models if k.startswith(want)]
    if len(prefix) == 1:
        return prefix[0]
    contains = [k for k in phase_models if want in k]
    if len(contains) == 1:
        return contains[0]
    return want


def resolve_phase_name_in_models(
    phase_models: Dict[str, Dict[str, Any]],
    aliases: Dict[str, str],
    phase_name: Optional[str],
) -> Optional[str]:
    want = _clean_name(phase_name)
    if want is None:
        return None
    want = aliases.get(want, want)
    if want in phase_models:
        return want
    prefix = [k for k in phase_models if k.startswith(want)]
    if len(prefix) == 1:
        return prefix[0]
    contains = [k for k in phase_models if want in k]
    if len(contains) == 1:
        return contains[0]
    return want


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


def load_calibration(json_path: str, y_offplane_sign: float = DEFAULT_Y_OFFPLANE_SIGN) -> Calibration:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")
    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    cubic = data.get("cubic_coefficients", {}) or {}
    fit_models = data.get("fit_models", {}) or {}
    if not isinstance(fit_models, dict):
        fit_models = {}

    pr_arr = cubic.get("r_coeffs")
    if pr_arr is None:
        pr = _coeffs_from_model(fit_models, "r_cubic", "r_avg_cubic", "r_linear", "r_avg_linear")
        if pr is None:
            pr = np.zeros(1, dtype=float)
    else:
        pr = np.asarray(pr_arr, dtype=float)

    pz_arr = cubic.get("z_coeffs")
    if pz_arr is None:
        pz = _coeffs_from_model(fit_models, "z_cubic", "z_avg_cubic", "z_linear", "z_avg_linear")
        if pz is None:
            pz = np.zeros(1, dtype=float)
    else:
        pz = np.asarray(pz_arr, dtype=float)

    py_off_raw = cubic.get("offplane_y_coeffs")
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

    pa_raw = cubic.get("tip_angle_coeffs")
    if pa_raw is None:
        pa = _coeffs_from_model(
            fit_models,
            "tip_angle_cubic",
            "tip_angle_avg_cubic",
            "tip_angle_linear",
            "tip_angle_avg_linear",
        )
    else:
        pa = np.asarray(pa_raw, dtype=float)

    phase_models, default_phase = _extract_phase_models(data)
    curl_angle_models = _extract_curl_angle_models(data)
    phase_aliases = _normalize_phase_aliases(data)

    motor_setup = data.get("motor_setup", {}) or {}
    duet_map = data.get("duet_axis_mapping", {}) or {}
    b_range = motor_setup.get("b_motor_position_range", [-5.4, 0.0])
    b_min, b_max = map(float, b_range)
    if b_min > b_max:
        b_min, b_max = b_max, b_min

    return Calibration(
        pr=pr,
        pz=pz,
        py_off=py_off,
        pa=pa,
        b_min=b_min,
        b_max=b_max,
        x_axis=str(duet_map.get("horizontal_axis") or motor_setup.get("horizontal_axis") or "X"),
        y_axis=str(duet_map.get("depth_axis") or motor_setup.get("depth_axis") or "Y"),
        z_axis=str(duet_map.get("vertical_axis") or motor_setup.get("vertical_axis") or "Z"),
        b_axis=str(duet_map.get("pull_axis") or motor_setup.get("b_motor_axis") or "B"),
        c_axis=str(duet_map.get("rotation_axis") or motor_setup.get("rotation_axis") or "C"),
        c_180_deg=float(motor_setup.get("rotation_axis_180_deg", 180.0)),
        fit_models=dict(fit_models),
        phase_models=phase_models,
        curl_angle_models=curl_angle_models,
        phase_aliases=phase_aliases,
        selected_fit_model=_clean_name(data.get("selected_fit_model")),
        selected_offplane_fit_model=_clean_name(data.get("selected_offplane_fit_model")),
        offplane_y_sign=float(y_offplane_sign),
        default_motion_phase=default_phase,
    )


def _select_fit_model(
    cal: Calibration,
    base_name: str,
    motion_phase: Optional[str] = None,
    fit_model_selector: Optional[str] = None,
    model_group: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    selector = _clean_name(fit_model_selector if fit_model_selector is not None else cal.selected_fit_model)
    phase_models = cal.phase_models or {}
    if model_group is not None:
        group_key = _clean_name(model_group)
        group_models = (cal.curl_angle_models or {}).get(group_key)
        if isinstance(group_models, dict) and group_models:
            phase_models = group_models
    phase = resolve_phase_name_in_models(phase_models, cal.phase_aliases or {}, motion_phase)
    if phase is not None and phase in phase_models:
        model = _select_named_model(phase_models[phase], base_name, selector)
        if model is not None:
            return model
    model = _select_named_model(cal.fit_models or {}, base_name, selector)
    if model is not None:
        return model
    if selector == "avg_linear":
        cache_key = (phase or "", _clean_name(base_name) or str(base_name), selector)
        if cache_key in cal.derived_model_cache:
            return cal.derived_model_cache[cache_key]
        source_model = None
        if phase is not None and phase in phase_models:
            source_model = _select_named_model(phase_models[phase], base_name, "avg_pchip")
        if source_model is None:
            source_model = _select_named_model(cal.fit_models or {}, base_name, "avg_pchip")
        derived = _fit_linear_model_from_pchip(
            source_model,
            value_name=f"{_clean_name(base_name) or str(base_name)}_avg_linear",
        )
        cal.derived_model_cache[cache_key] = derived
        if derived is not None:
            return derived
    return None


def eval_r(cal: Calibration, b: Any, motion_phase: Optional[str], fit_model_selector: Optional[str], model_group: Optional[str] = None) -> np.ndarray:
    model = _select_fit_model(cal, "r", motion_phase, fit_model_selector, model_group=model_group)
    if model is not None:
        return eval_model_spec(model, b)
    return poly_eval(cal.pr, b)


def eval_z(cal: Calibration, b: Any, motion_phase: Optional[str], fit_model_selector: Optional[str], model_group: Optional[str] = None) -> np.ndarray:
    model = _select_fit_model(cal, "z", motion_phase, fit_model_selector, model_group=model_group)
    if model is not None:
        return eval_model_spec(model, b)
    return poly_eval(cal.pz, b)


def eval_offplane_y(
    cal: Calibration,
    b: Any,
    offplane_fit_model: Optional[str],
    motion_phase: Optional[str] = None,
    model_group: Optional[str] = None,
) -> np.ndarray:
    selector = offplane_fit_model or cal.selected_offplane_fit_model or cal.selected_fit_model
    model = _select_fit_model(cal, "offplane_y", motion_phase, selector, model_group=model_group)
    extrap_model = _select_fit_model(cal, "offplane_y_linear", motion_phase, None, model_group=model_group)
    if model is not None:
        if str(model.get("model_type", "")).lower() == "pchip":
            values = eval_pchip_with_linear_extrap(model, extrap_model, b)
        else:
            values = eval_model_spec(model, b, default_if_none=0.0)
    else:
        values = poly_eval(cal.py_off, b, default_if_none=0.0)
    return float(cal.offplane_y_sign) * np.asarray(values, dtype=float)


def eval_tip_angle_deg(cal: Calibration, b: Any, motion_phase: Optional[str], fit_model_selector: Optional[str], model_group: Optional[str] = None) -> np.ndarray:
    model = _select_fit_model(cal, "tip_angle", motion_phase, fit_model_selector, model_group=model_group)
    if model is not None:
        return eval_model_spec(model, b)
    if cal.pa is None:
        raise ValueError("Calibration is missing tip_angle coefficients/model.")
    return poly_eval(cal.pa, b)


def solve_b_for_target_tip_angle(
    cal: Calibration,
    target_angle_deg: float,
    search_samples: int = DEFAULT_BC_SOLVE_SAMPLES,
    motion_phase: Optional[str] = None,
    fit_model_selector: Optional[str] = None,
    model_group: Optional[str] = None,
) -> float:
    b_lo, b_hi = float(cal.b_min), float(cal.b_max)
    bb = np.linspace(b_lo, b_hi, int(max(101, search_samples)))
    aa = eval_tip_angle_deg(cal, bb, motion_phase, fit_model_selector, model_group=model_group) - float(target_angle_deg)
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
            return float(eval_tip_angle_deg(cal, x, motion_phase, fit_model_selector, model_group=model_group) - float(target_angle_deg))

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


def predict_tip_offset_xyz(
    cal: Calibration,
    b_command: float,
    c_deg: float,
    motion_phase: Optional[str],
    fit_model_selector: Optional[str],
    offplane_fit_model: Optional[str],
    model_group: Optional[str] = None,
) -> np.ndarray:
    r = float(eval_r(cal, b_command, motion_phase, fit_model_selector, model_group=model_group))
    z = float(eval_z(cal, b_command, motion_phase, fit_model_selector, model_group=model_group))
    y_off = float(eval_offplane_y(cal, b_command, offplane_fit_model, motion_phase=motion_phase, model_group=model_group))
    c = math.radians(float(c_deg))
    x = r * math.cos(c) - y_off * math.sin(c)
    y = r * math.sin(c) + y_off * math.cos(c)
    return np.array([x, y, z], dtype=float)


def stage_xyz_for_tip(
    cal: Calibration,
    tip_xyz: np.ndarray,
    b_command: float,
    c_deg: float,
    motion_phase: Optional[str],
    fit_model_selector: Optional[str],
    offplane_fit_model: Optional[str],
    model_group: Optional[str] = None,
) -> np.ndarray:
    return np.asarray(tip_xyz, dtype=float) - predict_tip_offset_xyz(
        cal, b_command, c_deg, motion_phase, fit_model_selector, offplane_fit_model, model_group=model_group
    )


# ---------------- Circle geometry ----------------
@dataclass(frozen=True)
class MotionProfile:
    motion_phase: Optional[str]
    fit_model_selector: Optional[str]
    model_group: Optional[str] = None


@dataclass(frozen=True)
class CircleSegment:
    quadrant: int
    name: str
    theta_start_deg: float
    theta_end_deg: float
    c_deg: float
    b_start_deg: float
    b_end_deg: float
    profile: MotionProfile


def profile_for_phase(fit_strategy: str, phase: str) -> MotionProfile:
    strategy = str(fit_strategy).strip().lower().replace("_", "-")
    if strategy == "phase-pchip":
        return MotionProfile(motion_phase=phase, fit_model_selector="pchip")
    if strategy == "avg-pchip":
        return MotionProfile(motion_phase=None, fit_model_selector="avg_pchip")
    if strategy == "avg-cubic":
        return MotionProfile(motion_phase=None, fit_model_selector="avg_cubic")
    if strategy == "avg-linear":
        return MotionProfile(motion_phase=None, fit_model_selector="avg_linear")
    raise ValueError("--fit-strategy must be phase-pchip, avg-pchip, avg-cubic, or avg-linear")


def resolve_offplane_fit_model(fit_strategy: str, offplane_fit_model: Optional[str]) -> str:
    if offplane_fit_model is not None:
        return str(offplane_fit_model)
    strategy = str(fit_strategy).strip().lower().replace("_", "-")
    if strategy == "avg-linear":
        return "avg_linear"
    return DEFAULT_OFFPLANE_FIT_MODEL


def phase_profile_for_group(
    fit_strategy: str,
    phase: str,
    model_group: str,
) -> MotionProfile:
    strategy = str(fit_strategy).strip().lower().replace("_", "-")
    if strategy == "phase-pchip":
        return MotionProfile(motion_phase=phase, fit_model_selector="pchip", model_group=model_group)
    return profile_for_phase(fit_strategy, phase)


def build_segments(args: argparse.Namespace) -> List[CircleSegment]:
    branch_mode = str(getattr(args, "phase_branch_mode", DEFAULT_PHASE_BRANCH_MODE)).strip().lower().replace("_", "-")
    if branch_mode not in {"half-circle", "curl-specific-0-180-0", "quadrant-split"}:
        raise ValueError("--phase-branch-mode must be half-circle, curl-specific-0-180-0, or quadrant-split")

    if branch_mode in {"half-circle", "curl-specific-0-180-0"}:
        q1_profile = phase_profile_for_group(args.fit_strategy, "release", "0_180_0")
        q2_profile = phase_profile_for_group(args.fit_strategy, "release", "0_180_0")
        q3_profile = phase_profile_for_group(args.fit_strategy, "pull", "0_180_0")
        q4_profile = phase_profile_for_group(args.fit_strategy, "pull", "0_180_0")
    else:
        q1_profile = phase_profile_for_group(args.fit_strategy, "release", "90_180_90")
        q2_profile = phase_profile_for_group(args.fit_strategy, "release", "0_90_0")
        q3_profile = phase_profile_for_group(args.fit_strategy, "pull", "0_90_0")
        q4_profile = phase_profile_for_group(args.fit_strategy, "pull", "90_180_90")

    return [
        CircleSegment(1, "left_to_bottom", 180.0, 270.0, 180.0, args.q1_b_start, args.q1_b_end, q1_profile),
        CircleSegment(2, "bottom_to_right", 270.0, 360.0, 180.0, args.q2_b_start, args.q2_b_end, q2_profile),
        CircleSegment(3, "right_to_top", 0.0, 90.0, 0.0, args.q3_b_start, args.q3_b_end, q3_profile),
        CircleSegment(4, "top_to_left", 90.0, 180.0, 0.0, args.q4_b_start, args.q4_b_end, q4_profile),
    ]


def validate_phase_branch_mode(
    cal: Optional[Calibration],
    fit_strategy: str,
    phase_branch_mode: str,
) -> None:
    branch_mode = str(phase_branch_mode).strip().lower().replace("_", "-")
    if branch_mode != "curl-specific-0-180-0":
        return

    strategy = str(fit_strategy).strip().lower().replace("_", "-")
    if strategy != "phase-pchip":
        raise ValueError(
            "--phase-branch-mode curl-specific-0-180-0 requires --fit-strategy phase-pchip "
            "so tip position and tip angle use the per-phase PCHIP equations."
        )
    if cal is None:
        raise ValueError("--phase-branch-mode curl-specific-0-180-0 requires --write-mode calibrated")

    group_models = (cal.curl_angle_models or {}).get("0_180_0")
    if not isinstance(group_models, dict) or not group_models:
        raise ValueError(
            "Calibration is missing curl-specific 0-180-0 fit models. "
            "Use a calibration JSON with curl_angle_specific_fit_models['0-180-0']."
        )
    for phase_name in ("release", "pull"):
        resolved_phase = resolve_phase_name_in_models(group_models, cal.phase_aliases or {}, phase_name)
        if resolved_phase not in group_models:
            raise ValueError(
                "Calibration curl-specific 0-180-0 models must include both release and pull phases."
            )


def circle_point(center_x: float, center_y: float, center_z: float, radius: float, theta_deg: float) -> np.ndarray:
    th = math.radians(float(theta_deg))
    return np.array(
        [
            float(center_x) + float(radius) * math.cos(th),
            float(center_y),
            float(center_z) + float(radius) * math.sin(th),
        ],
        dtype=float,
    )


def sample_segment_points(
    seg: CircleSegment,
    center_x: float,
    center_y: float,
    center_z: float,
    radius: float,
    points_per_mm: float,
) -> Tuple[np.ndarray, np.ndarray]:
    arc_len = abs(math.radians(seg.theta_end_deg - seg.theta_start_deg)) * float(radius)
    n = max(2, int(math.ceil(arc_len * float(points_per_mm))) + 1)
    theta = np.linspace(float(seg.theta_start_deg), float(seg.theta_end_deg), n)
    pts = np.asarray([circle_point(center_x, center_y, center_z, radius, t) for t in theta], dtype=float)
    b_targets = np.linspace(float(seg.b_start_deg), float(seg.b_end_deg), n)
    return deduplicate_polyline_points(pts), b_targets[: len(deduplicate_polyline_points(pts))]


def parse_quadrants(text: str) -> Set[int]:
    raw = str(text).strip().lower()
    if raw in {"all", "*", "1,2,3,4"}:
        return {1, 2, 3, 4}
    aliases = {
        "left_to_bottom": 1,
        "left-bottom": 1,
        "lb": 1,
        "bottom_to_right": 2,
        "bottom-right": 2,
        "br": 2,
        "right_to_top": 3,
        "right-top": 3,
        "rt": 3,
        "top_to_left": 4,
        "top-left": 4,
        "tl": 4,
    }
    out: Set[int] = set()
    for part in re.split(r"[,\s]+", raw):
        if not part:
            continue
        if "-" in part and all(p.strip().isdigit() for p in part.split("-", 1)):
            a, b = [int(v) for v in part.split("-", 1)]
            lo, hi = sorted((a, b))
            out.update(range(lo, hi + 1))
            continue
        if part.isdigit():
            out.add(int(part))
            continue
        part_norm = part.replace(" ", "_")
        if part_norm in aliases:
            out.add(aliases[part_norm])
            continue
        raise ValueError(f"Unknown quadrant selector {part!r}")
    bad = sorted(q for q in out if q not in {1, 2, 3, 4})
    if bad:
        raise ValueError(f"Quadrants must be in 1..4; got {bad}")
    if not out:
        raise ValueError("At least one quadrant must be selected")
    return out


# ---------------- G-code writer ----------------
class GCodeWriter:
    def __init__(
        self,
        fh,
        cal: Optional[Calibration],
        write_mode: str,
        feedrate_mode: str,
        offplane_fit_model: str,
        bc_solve_samples: int,
        bbox: Dict[str, float],
        first_inverse_f: float,
        emit_extrusion: bool,
        preflow_dwell_ms: int,
    ) -> None:
        self.f = fh
        self.cal = cal
        self.write_mode = str(write_mode).strip().lower()
        self.feedrate_mode = str(feedrate_mode).strip().lower().replace("_", "-")
        self.offplane_fit_model = _clean_name(offplane_fit_model)
        self.bc_solve_samples = int(bc_solve_samples)
        self.bbox = dict(bbox)
        self.first_inverse_f = float(first_inverse_f)
        self.emit_extrusion = bool(emit_extrusion)
        self.preflow_dwell_ms = int(preflow_dwell_ms)

        self.cur_stage_xyz: Optional[np.ndarray] = None
        self.cur_tip_xyz: Optional[np.ndarray] = None
        self.cur_b_command: Optional[float] = None
        self.cur_b_physical: Optional[float] = None
        self.cur_c: Optional[float] = None
        self.extruding = False
        self.warnings: List[str] = []

        self.stage_min = np.array([np.inf, np.inf, np.inf], dtype=float)
        self.stage_max = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
        self.tip_min = np.array([np.inf, np.inf, np.inf], dtype=float)
        self.tip_max = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
        self.b_command_min = float("inf")
        self.b_command_max = float("-inf")
        self.c_min = float("inf")
        self.c_max = float("-inf")

        if self.write_mode == "calibrated" and self.cal is None:
            raise ValueError("Calibration is required for --write-mode calibrated")
        if self.write_mode not in {"calibrated", "cartesian"}:
            raise ValueError("--write-mode must be calibrated or cartesian")
        if self.feedrate_mode not in {"units-per-min", "inverse-time"}:
            raise ValueError("--feedrate-mode must be units-per-min or inverse-time")

    @property
    def x_axis(self) -> str:
        return self.cal.x_axis if self.cal else "X"

    @property
    def y_axis(self) -> str:
        return self.cal.y_axis if self.cal else "Y"

    @property
    def z_axis(self) -> str:
        return self.cal.z_axis if self.cal else "Z"

    @property
    def b_axis(self) -> str:
        return self.cal.b_axis if self.cal else "B"

    @property
    def c_axis(self) -> str:
        return self.cal.c_axis if self.cal else "C"

    def header_modal(self) -> None:
        if self.feedrate_mode == "inverse-time":
            self.f.write("G93                 ; inverse time mode\n")
        else:
            self.f.write("G94                 ; units/min feedrate mode\n")

    def footer_modal(self) -> None:
        if self.feedrate_mode == "inverse-time":
            self.f.write("G94                 ; return to normal units/min mode\n")

    def clamp_stage(self, p_stage: np.ndarray, context: str) -> np.ndarray:
        x = float(np.clip(p_stage[0], self.bbox["x_min"], self.bbox["x_max"]))
        y = float(np.clip(p_stage[1], self.bbox["y_min"], self.bbox["y_max"]))
        z = float(np.clip(p_stage[2], self.bbox["z_min"], self.bbox["z_max"]))
        if abs(x - p_stage[0]) > 1e-12 or abs(y - p_stage[1]) > 1e-12 or abs(z - p_stage[2]) > 1e-12:
            self.warnings.append(f"WARNING: {context} stage point clamped to bbox")
        return np.array([x, y, z], dtype=float)

    def b_command_for_physical_angle(self, b_physical_deg: float, profile: MotionProfile) -> float:
        target = float(np.clip(b_physical_deg, 0.0, 180.0))
        if self.write_mode == "calibrated":
            assert self.cal is not None
            return solve_b_for_target_tip_angle(
                self.cal,
                target,
                search_samples=self.bc_solve_samples,
                motion_phase=profile.motion_phase,
                fit_model_selector=profile.fit_model_selector,
                model_group=profile.model_group,
            )
        return target

    def tip_to_stage(self, tip_xyz: np.ndarray, b_physical_deg: float, c_deg: float, profile: MotionProfile) -> Tuple[np.ndarray, float]:
        b_cmd = self.b_command_for_physical_angle(b_physical_deg, profile)
        if self.write_mode == "calibrated":
            assert self.cal is not None
            p_stage = stage_xyz_for_tip(
                self.cal,
                np.asarray(tip_xyz, dtype=float),
                b_cmd,
                float(c_deg),
                profile.motion_phase,
                profile.fit_model_selector,
                self.offplane_fit_model,
                model_group=profile.model_group,
            )
        else:
            p_stage = np.asarray(tip_xyz, dtype=float)
        return self.clamp_stage(p_stage, "tip_to_stage"), float(b_cmd)

    def _inverse_time_f(self, tip_xyz: np.ndarray, stage_xyz: np.ndarray, feed: float, move_kind: str) -> float:
        if self.cur_stage_xyz is None:
            self.warnings.append(
                "WARNING: first inverse-time move has no known previous pose; using --first-inverse-f"
            )
            return self.first_inverse_f
        if move_kind == "spin" and self.cur_c is not None:
            c_delta = abs(float(self.pending_c_for_feed) - float(self.cur_c))  # set by write_move
            if c_delta > 1e-12:
                return max(float(feed) / c_delta, 1e-6)
        if self.cur_tip_xyz is not None:
            dist = float(np.linalg.norm(np.asarray(tip_xyz, dtype=float) - self.cur_tip_xyz))
        else:
            dist = float(np.linalg.norm(np.asarray(stage_xyz, dtype=float) - self.cur_stage_xyz))
        if dist <= 1e-12:
            # Still produce a legal finite F for pure B/C changes or zero-length bookkeeping moves.
            return self.first_inverse_f
        duration_min = dist / max(float(feed), 1e-9)
        return max(1.0 / max(duration_min, 1e-9), 1e-6)

    def write_move(
        self,
        tip_xyz: np.ndarray,
        b_physical_deg: float,
        c_deg: float,
        feed: float,
        profile: MotionProfile,
        comment: Optional[str] = None,
        move_kind: str = "linear",
    ) -> None:
        tip = np.asarray(tip_xyz, dtype=float)
        stage, b_cmd = self.tip_to_stage(tip, b_physical_deg, c_deg, profile)
        if comment:
            self.f.write(f"; {comment}\n")

        axes: List[Tuple[str, float]] = [
            (self.x_axis, float(stage[0])),
            (self.y_axis, float(stage[1])),
            (self.z_axis, float(stage[2])),
            (self.b_axis, float(b_cmd)),
            (self.c_axis, float(c_deg)),
        ]

        if self.feedrate_mode == "inverse-time":
            self.pending_c_for_feed = float(c_deg)
            f_value = self._inverse_time_f(tip, stage, float(feed), move_kind=move_kind)
            f_text = _fmt_float(f_value, 5)
        else:
            f_text = _fmt_float(float(feed), 3)
        self.f.write(f"G1 {_fmt_axes_move(axes)} F{f_text}\n")

        self.cur_stage_xyz = stage.copy()
        self.cur_tip_xyz = tip.copy()
        self.cur_b_command = float(b_cmd)
        self.cur_b_physical = float(b_physical_deg)
        self.cur_c = float(c_deg)

        self.stage_min = np.minimum(self.stage_min, stage)
        self.stage_max = np.maximum(self.stage_max, stage)
        self.tip_min = np.minimum(self.tip_min, tip)
        self.tip_max = np.maximum(self.tip_max, tip)
        self.b_command_min = min(self.b_command_min, float(b_cmd))
        self.b_command_max = max(self.b_command_max, float(b_cmd))
        self.c_min = min(self.c_min, float(c_deg))
        self.c_max = max(self.c_max, float(c_deg))

    def pressure_on(self) -> None:
        if self.emit_extrusion and not self.extruding:
            self.extruding = True
            self.f.write("; open pressure solenoid before print pass\n")
            self.f.write("M42 P0 S1\n")
            if self.preflow_dwell_ms > 0:
                self.f.write(f"G4 P{self.preflow_dwell_ms}\n")

    def pressure_off(self) -> None:
        if self.emit_extrusion and self.extruding:
            self.extruding = False
            self.f.write("; close pressure solenoid after print pass\n")
            self.f.write("M42 P0 S0\n")

    def approach_to_start(
        self,
        start_tip: np.ndarray,
        b_physical_deg: float,
        c_deg: float,
        profile: MotionProfile,
        travel_lift_z: float,
        travel_feed: float,
        approach_feed: float,
        fine_approach_feed: float,
    ) -> None:
        start = np.asarray(start_tip, dtype=float)
        if self.extruding:
            self.pressure_off()
        if self.cur_tip_xyz is not None:
            lifted = np.asarray(self.cur_tip_xyz, dtype=float).copy()
            lifted[2] += float(travel_lift_z)
            self.write_move(
                lifted,
                self.cur_b_physical if self.cur_b_physical is not None else b_physical_deg,
                self.cur_c if self.cur_c is not None else c_deg,
                approach_feed,
                profile,
                comment="lift current tip before travel",
            )
        above = start.copy()
        above[2] += float(travel_lift_z)
        self.write_move(
            above,
            b_physical_deg,
            c_deg,
            travel_feed,
            profile,
            comment="move above circle start at clearance Z",
        )
        self.write_move(
            start,
            b_physical_deg,
            c_deg,
            fine_approach_feed,
            profile,
            comment="go down in Z at circle start",
        )

    def final_retract_left_then_up(
        self,
        end_left_mm: float,
        travel_lift_z: float,
        approach_feed: float,
        profile: MotionProfile,
    ) -> None:
        if self.cur_tip_xyz is None or self.cur_b_physical is None or self.cur_c is None:
            return
        if self.extruding:
            self.pressure_off()
        left = self.cur_tip_xyz.copy()
        left[0] -= float(end_left_mm)
        self.write_move(
            left,
            self.cur_b_physical,
            self.cur_c,
            approach_feed,
            profile,
            comment=f"move left by {float(end_left_mm):.3f} mm after circle end",
        )
        up = left.copy()
        up[2] += float(travel_lift_z)
        self.write_move(
            up,
            self.cur_b_physical,
            self.cur_c,
            approach_feed,
            profile,
            comment="go up in Z after final left move",
        )

    def transition_profile_at_fixed_tip(
        self,
        tip_xyz: np.ndarray,
        b_physical_deg: float,
        c_deg: float,
        from_profile: MotionProfile,
        to_profile: MotionProfile,
        steps: int,
        feed: float,
    ) -> None:
        if self.write_mode != "calibrated":
            return
        if from_profile == to_profile:
            return
        assert self.cal is not None
        tip = np.asarray(tip_xyz, dtype=float)
        start_stage, start_b = self.tip_to_stage(tip, b_physical_deg, c_deg, from_profile)
        end_stage, end_b = self.tip_to_stage(tip, b_physical_deg, c_deg, to_profile)
        n = max(2, int(steps))
        self.f.write("; smooth fixed-tip transition between calibration fit profiles\n")
        for i in range(1, n + 1):
            u = i / float(n)
            stage = (1.0 - u) * start_stage + u * end_stage
            b_cmd = (1.0 - u) * start_b + u * end_b
            axes: List[Tuple[str, float]] = [
                (self.x_axis, float(stage[0])),
                (self.y_axis, float(stage[1])),
                (self.z_axis, float(stage[2])),
                (self.b_axis, float(b_cmd)),
                (self.c_axis, float(c_deg)),
            ]
            if self.feedrate_mode == "inverse-time":
                dist = 0.0 if self.cur_stage_xyz is None else float(np.linalg.norm(stage - self.cur_stage_xyz))
                f_value = self.first_inverse_f if dist <= 1e-12 else max(float(feed) / dist, 1e-6)
                f_text = _fmt_float(f_value, 5)
            else:
                f_text = _fmt_float(float(feed), 3)
            self.f.write(f"G1 {_fmt_axes_move(axes)} F{f_text}\n")
            self.cur_stage_xyz = stage.copy()
            self.cur_tip_xyz = tip.copy()
            self.cur_b_command = float(b_cmd)
            self.cur_b_physical = float(b_physical_deg)
            self.cur_c = float(c_deg)
            self.stage_min = np.minimum(self.stage_min, stage)
            self.stage_max = np.maximum(self.stage_max, stage)
            self.b_command_min = min(self.b_command_min, float(b_cmd))
            self.b_command_max = max(self.b_command_max, float(b_cmd))
            self.c_min = min(self.c_min, float(c_deg))
            self.c_max = max(self.c_max, float(c_deg))

    def spin_c_tracking_fixed_tip(
        self,
        tip_xyz: np.ndarray,
        b_physical_deg: float,
        c_start: float,
        c_end: float,
        profile: MotionProfile,
        steps: int,
        c_spin_feed: float,
    ) -> None:
        if self.extruding:
            self.pressure_off()
        self.f.write(
            f"; fixed-tip C spin at rightmost point: C{float(c_start):.3f} -> C{float(c_end):.3f}\n"
        )
        n = max(1, int(steps))
        for i in range(1, n + 1):
            u = i / float(n)
            c_here = (1.0 - u) * float(c_start) + u * float(c_end)
            self.write_move(
                tip_xyz,
                b_physical_deg,
                c_here,
                c_spin_feed,
                profile,
                comment=None,
                move_kind="spin",
            )

    def print_segment(
        self,
        seg: CircleSegment,
        points: np.ndarray,
        b_targets: np.ndarray,
        print_feed: float,
        edge_samples: int,
    ) -> None:
        if len(points) < 2:
            return
        self.f.write(
            "; CIRCLE_QUADRANT_START "
            f"q={seg.quadrant} name={seg.name} point_count={len(points)} "
            f"theta_start={seg.theta_start_deg:.6f} theta_end={seg.theta_end_deg:.6f} "
            f"c={seg.c_deg:.6f} b_start={seg.b_start_deg:.6f} b_end={seg.b_end_deg:.6f} "
            f"motion_phase={seg.profile.motion_phase or 'none'} fit_model={seg.profile.fit_model_selector or 'none'} "
            f"model_group={seg.profile.model_group or 'default'}\n"
        )
        self.pressure_on()
        n_sub = max(1, int(edge_samples))
        for i in range(1, len(points)):
            p0 = np.asarray(points[i - 1], dtype=float)
            p1 = np.asarray(points[i], dtype=float)
            b0 = float(b_targets[i - 1])
            b1 = float(b_targets[i])
            for s in range(1, n_sub + 1):
                u = s / float(n_sub)
                p = p0 + u * (p1 - p0)
                b = (1.0 - u) * b0 + u * b1
                self.write_move(p, b, seg.c_deg, print_feed, seg.profile)
        self.f.write("; CIRCLE_QUADRANT_END\n")


# ---------------- Top-level generator ----------------
def write_circle_gcode(
    out: str,
    calibration: Optional[str],
    write_mode: str,
    fit_strategy: str,
    phase_branch_mode: str,
    offplane_fit_model: Optional[str],
    y_offplane_sign: float,
    center_x: float,
    center_y: float,
    center_z: float,
    radius: float,
    quadrants: str,
    points_per_mm: float,
    edge_samples: int,
    q1_b_start: float,
    q1_b_end: float,
    q2_b_start: float,
    q2_b_end: float,
    q3_b_start: float,
    q3_b_end: float,
    q4_b_start: float,
    q4_b_end: float,
    bc_solve_samples: int,
    phase_transition_steps: int,
    feedrate_mode: str,
    first_inverse_f: float,
    travel_feed: float,
    approach_feed: float,
    fine_approach_feed: float,
    print_feed: float,
    c_spin_feed: float,
    spin_steps: int,
    travel_lift_z: float,
    end_left_mm: float,
    emit_extrusion: bool,
    preflow_dwell_ms: int,
    bbox_x_min: float,
    bbox_x_max: float,
    bbox_y_min: float,
    bbox_y_max: float,
    bbox_z_min: float,
    bbox_z_max: float,
) -> Dict[str, Any]:
    if float(radius) <= 0.0:
        raise ValueError("--radius must be > 0")
    if float(points_per_mm) <= 0.0:
        raise ValueError("--points-per-mm must be > 0")
    write_mode = str(write_mode).strip().lower()
    if write_mode == "calibrated" and not calibration:
        raise ValueError("--calibration is required when --write-mode calibrated")
    resolved_offplane_fit_model = resolve_offplane_fit_model(fit_strategy, offplane_fit_model)

    cal: Optional[Calibration]
    if write_mode == "calibrated":
        cal = load_calibration(str(calibration), y_offplane_sign=y_offplane_sign)
    else:
        cal = None
    validate_phase_branch_mode(cal, fit_strategy, phase_branch_mode)

    selected = parse_quadrants(quadrants)

    # argparse.Namespace is used so build_segments can share CLI defaults and keyword values.
    ns = argparse.Namespace(
        fit_strategy=fit_strategy,
        phase_branch_mode=phase_branch_mode,
        q1_b_start=q1_b_start,
        q1_b_end=q1_b_end,
        q2_b_start=q2_b_start,
        q2_b_end=q2_b_end,
        q3_b_start=q3_b_start,
        q3_b_end=q3_b_end,
        q4_b_start=q4_b_start,
        q4_b_end=q4_b_end,
    )
    segments = build_segments(ns)
    seg_by_q = {seg.quadrant: seg for seg in segments}

    samples: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for seg in segments:
        samples[seg.quadrant] = sample_segment_points(
            seg, center_x=center_x, center_y=center_y, center_z=center_z, radius=radius, points_per_mm=points_per_mm
        )

    bbox = {
        "x_min": float(bbox_x_min),
        "x_max": float(bbox_x_max),
        "y_min": float(bbox_y_min),
        "y_max": float(bbox_y_max),
        "z_min": float(bbox_z_min),
        "z_max": float(bbox_z_max),
    }

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        writer = GCodeWriter(
            fh=fh,
            cal=cal,
            write_mode=write_mode,
            feedrate_mode=feedrate_mode,
            offplane_fit_model=resolved_offplane_fit_model,
            bc_solve_samples=bc_solve_samples,
            bbox=bbox,
            first_inverse_f=first_inverse_f,
            emit_extrusion=emit_extrusion,
            preflow_dwell_ms=preflow_dwell_ms,
        )

        fh.write("; Calibrated XZ circle trace generator\n")
        fh.write("; Generated by circle_trace_generator.py\n")
        fh.write(
            f"; center_x={float(center_x):.6f} center_y={float(center_y):.6f} "
            f"center_z={float(center_z):.6f} radius={float(radius):.6f}\n"
        )
        fh.write(f"; selected_quadrants={','.join(str(q) for q in sorted(selected))}\n")
        fh.write(
            f"; write_mode={write_mode} fit_strategy={fit_strategy} phase_branch_mode={phase_branch_mode} "
            f"offplane_fit_model={resolved_offplane_fit_model}\n"
        )
        fh.write("G90                 ; absolute positioning\n")
        writer.header_modal()

        ready_for_next_without_approach = False
        last_printed_q: Optional[int] = None
        last_profile: Optional[MotionProfile] = None

        for q in [1, 2, 3, 4]:
            if q not in selected:
                ready_for_next_without_approach = False
                continue

            seg = seg_by_q[q]
            points, b_targets = samples[q]
            points_to_print = np.asarray(points, dtype=float).copy()
            b_targets_to_print = np.asarray(b_targets, dtype=float).copy()
            start_tip = points[0]
            start_b = float(b_targets[0])

            should_approach = not ready_for_next_without_approach
            if should_approach:
                writer.approach_to_start(
                    start_tip=start_tip,
                    b_physical_deg=start_b,
                    c_deg=seg.c_deg,
                    profile=seg.profile,
                    travel_lift_z=travel_lift_z,
                    travel_feed=travel_feed,
                    approach_feed=approach_feed,
                    fine_approach_feed=fine_approach_feed,
                )
            elif writer.cur_tip_xyz is not None and writer.cur_b_physical is not None:
                points_to_print[0] = np.asarray(writer.cur_tip_xyz, dtype=float)
                b_targets_to_print[0] = float(writer.cur_b_physical)

            writer.print_segment(
                seg,
                points_to_print,
                b_targets_to_print,
                print_feed=print_feed,
                edge_samples=edge_samples,
            )
            last_printed_q = q
            last_profile = seg.profile

            if q == 2 and 3 in selected:
                # Track the current tip during the C spin, then let the next quadrant
                # continue from that actual post-handoff tip state rather than snapping
                # to the nominal branch start.
                writer.pressure_off()
                next_seg = seg_by_q[3]
                if writer.cur_tip_xyz is None:
                    raise ValueError("Branch handoff requires a known current tip pose.")
                handoff_tip = np.asarray(writer.cur_tip_xyz, dtype=float).copy()
                right_b = float(b_targets_to_print[-1])
                writer.spin_c_tracking_fixed_tip(
                    tip_xyz=handoff_tip,
                    b_physical_deg=right_b,
                    c_start=seg.c_deg,
                    c_end=next_seg.c_deg,
                    profile=next_seg.profile,
                    steps=spin_steps,
                    c_spin_feed=c_spin_feed,
                )
                last_profile = next_seg.profile
                ready_for_next_without_approach = True
                continue

            # Continue extrusion only across true adjacent quadrant joins with matching C and no spin.
            if (q == 1 and 2 in selected) or (q == 3 and 4 in selected):
                ready_for_next_without_approach = True
                continue

            writer.pressure_off()
            future_selected = any(qq in selected for qq in range(q + 1, 5))
            if future_selected:
                ready_for_next_without_approach = False
            else:
                writer.final_retract_left_then_up(
                    end_left_mm=end_left_mm,
                    travel_lift_z=travel_lift_z,
                    approach_feed=approach_feed,
                    profile=seg.profile,
                )
                ready_for_next_without_approach = False

        writer.pressure_off()
        writer.footer_modal()
        fh.write("; END\n")

        all_points = np.vstack([samples[q][0] for q in sorted(selected)])
        all_b_targets = np.concatenate([samples[q][1] for q in sorted(selected)])

        # Hard validation: this generator is for an XZ-plane circle only.
        # Tip-space Y must remain fixed at center_y; any Y motion in the emitted
        # stage coordinates comes only from calibration tip-offset compensation
        # in stage_xyz_for_tip().
        tip_y_error = float(np.max(np.abs(all_points[:, 1] - float(center_y))))
        if tip_y_error > 1e-9:
            raise AssertionError(
                f"Internal planner error: tip-space Y changed by {tip_y_error:.9g} mm; "
                "the circle must remain in the XZ plane."
            )
        tip_z_span = float(np.max(all_points[:, 2]) - np.min(all_points[:, 2]))
        if len(selected) == 4 and abs(tip_z_span - 2.0 * float(radius)) > 1e-6:
            raise AssertionError(
                f"Internal planner error: full-circle tip Z span is {tip_z_span:.9g} mm; "
                f"expected {2.0 * float(radius):.9g} mm for an XZ-plane circle."
            )

        summary = {
            "circle_plane": "XZ",
            "tip_y_constant_error_mm": tip_y_error,
            "out": str(out_path),
            "write_mode": write_mode,
            "feedrate_mode": str(feedrate_mode),
            "fit_strategy": str(fit_strategy),
            "phase_branch_mode": str(phase_branch_mode),
            "offplane_fit_model": str(resolved_offplane_fit_model),
            "selected_quadrants": sorted(selected),
            "center": [float(center_x), float(center_y), float(center_z)],
            "radius": float(radius),
            "tip_xyz_range": {
                "x": [float(np.min(all_points[:, 0])), float(np.max(all_points[:, 0]))],
                "y": [float(np.min(all_points[:, 1])), float(np.max(all_points[:, 1]))],
                "z": [float(np.min(all_points[:, 2])), float(np.max(all_points[:, 2]))],
            },
            "physical_b_target_range_deg": [float(np.min(all_b_targets)), float(np.max(all_b_targets))],
            "b_command_range": [float(writer.b_command_min), float(writer.b_command_max)],
            "c_command_range": [float(writer.c_min), float(writer.c_max)],
            "stage_xyz_range": {
                "x": [float(writer.stage_min[0]), float(writer.stage_max[0])],
                "y": [float(writer.stage_min[1]), float(writer.stage_max[1])],
                "z": [float(writer.stage_min[2]), float(writer.stage_max[2])],
            },
            "warnings": list(writer.warnings),
        }
        return summary


# ---------------- CLI ----------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Generate calibrated or Cartesian G-code for tracing a circle in the XZ plane. "
            "The planner supports quadrant selection, pull/release PCHIP or average fits, "
            "fixed-tip C spin tracking, and optional G93 inverse-time feedrates."
        )
    )
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code file.")
    ap.add_argument("--calibration", default=None, help="Calibration JSON. Required for --write-mode calibrated.")
    ap.add_argument("--write-mode", choices=["calibrated", "cartesian"], default=DEFAULT_WRITE_MODE)
    ap.add_argument(
        "--fit-strategy",
        choices=["phase-pchip", "avg-pchip", "avg-cubic", "avg-linear"],
        default=DEFAULT_FIT_STRATEGY,
        help="Use separate pull/release PCHIP, average PCHIP, average cubic, or average linear models for r/z/tip-angle tracking.",
    )
    ap.add_argument(
        "--phase-branch-mode",
        choices=["half-circle", "curl-specific-0-180-0", "quadrant-split"],
        default=DEFAULT_PHASE_BRANCH_MODE,
        help=(
            "With --fit-strategy phase-pchip, choose either the generic 180->0 release then 0->180 pull "
            "half-circle mapping, a required curl-specific 0-180-0 variant of that mapping, or split "
            "each half at 90 deg using 180->90 release, 90->0 release, 0->90 pull, 90->180 pull."
        ),
    )
    ap.add_argument(
        "--offplane-fit-model",
        default=None,
        help="Selector for offplane_y model, e.g. avg_cubic, avg_linear, avg_pchip, pchip, cubic. Defaults to avg_linear with --fit-strategy avg-linear, otherwise avg_cubic.",
    )
    ap.add_argument(
        "--y-offplane-sign",
        type=float,
        default=DEFAULT_Y_OFFPLANE_SIGN,
        help="Multiplier applied to calibration off-plane Y term. Use -1 to flip sign.",
    )

    ap.add_argument("--center-x", type=float, default=DEFAULT_CENTER_X)
    ap.add_argument("--center-y", type=float, default=DEFAULT_CENTER_Y)
    ap.add_argument("--center-z", dest="center_z", type=float, default=DEFAULT_CENTER_Z, help="Tip-space Z coordinate of the circle center.")
    ap.add_argument("--print-z", dest="center_z", type=float, help=argparse.SUPPRESS)  # backward-compatible alias
    ap.add_argument("--radius", type=float, default=DEFAULT_RADIUS)
    ap.add_argument(
        "--quadrants",
        default="1,2,3,4",
        help=(
            "Quadrants to print: all, 1,2,3,4, ranges like 1-3, or aliases "
            "lb/br/rt/tl for left-bottom, bottom-right, right-top, top-left."
        ),
    )
    ap.add_argument("--points-per-mm", type=float, default=DEFAULT_POINTS_PER_MM)
    ap.add_argument("--edge-samples", type=int, default=DEFAULT_EDGE_SAMPLES)

    # Physical target B-angle schedule. These are solved through calibration in calibrated mode.
    ap.add_argument("--q1-b-start", type=float, default=180.0, help="Physical B target at leftmost start of quadrant 1.")
    ap.add_argument("--q1-b-end", type=float, default=90.0, help="Physical B target at bottommost end of quadrant 1.")
    ap.add_argument("--q2-b-start", type=float, default=90.0, help="Physical B target at bottommost start of quadrant 2.")
    ap.add_argument("--q2-b-end", type=float, default=0.0, help="Physical B target at rightmost end of quadrant 2.")
    ap.add_argument("--q3-b-start", type=float, default=0.0, help="Physical B target at rightmost start of quadrant 3.")
    ap.add_argument("--q3-b-end", type=float, default=90.0, help="Physical B target at topmost end of quadrant 3.")
    ap.add_argument("--q4-b-start", type=float, default=90.0, help="Physical B target at topmost start of quadrant 4.")
    ap.add_argument(
        "--q4-b-end",
        type=float,
        default=180.0,
        help="Physical B target at leftmost final point. Default follows your request: return to start point at B angle 90.",
    )

    ap.add_argument("--bc-solve-samples", type=int, default=DEFAULT_BC_SOLVE_SAMPLES)
    ap.add_argument(
        "--phase-transition-steps",
        type=int,
        default=DEFAULT_PHASE_TRANSITION_STEPS,
        help="Fixed-tip blend steps when switching pull/release calibration before the C spin.",
    )

    ap.add_argument(
        "--feedrate-mode",
        choices=["units-per-min", "inverse-time"],
        default="units-per-min",
        help="Use G94 units/min or G93 inverse-time F values.",
    )
    ap.add_argument(
        "--first-inverse-f",
        type=float,
        default=DEFAULT_FIRST_INVERSE_F,
        help="F value for the first G93 move, because previous machine pose is unknown.",
    )
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED)
    ap.add_argument("--approach-feed", type=float, default=DEFAULT_APPROACH_FEED)
    ap.add_argument("--fine-approach-feed", type=float, default=DEFAULT_FINE_APPROACH_FEED)
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED)
    ap.add_argument(
        "--c-spin-feed",
        type=float,
        default=DEFAULT_C_SPIN_FEED,
        help="C-axis feed used only for the fixed-tip C spin. Default 10000.",
    )
    ap.add_argument("--spin-steps", type=int, default=DEFAULT_SPIN_STEPS)
    ap.add_argument("--travel-lift-z", type=float, default=DEFAULT_TRAVEL_LIFT_Z)
    ap.add_argument(
        "--end-left-mm",
        type=float,
        default=DEFAULT_END_LEFT_MM,
        help="After the final printed point, move left by this X distance, then lift in Z.",
    )

    ap.add_argument("--emit-extrusion", dest="emit_extrusion", action="store_true", default=DEFAULT_EMIT_EXTRUSION)
    ap.add_argument("--no-emit-extrusion", dest="emit_extrusion", action="store_false")
    ap.add_argument(
        "--preflow-dwell-ms",
        type=int,
        default=DEFAULT_PREFLOW_DWELL_MS,
        help="Dwell after opening the pressure valve and before starting a print pass.",
    )

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
    summary = write_circle_gcode(**vars(args))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
