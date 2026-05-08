#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    from scipy.interpolate import PchipInterpolator  # type: ignore
except Exception:  # pragma: no cover
    PchipInterpolator = None


DEFAULT_OUT = "gcode_generation/dance_valse_40cycles.gcode"
DEFAULT_WRITE_MODE = "calibrated"
DEFAULT_FEEDRATE = 3000.0
DEFAULT_TRAVEL_FEED = 3000.0
DEFAULT_INVERSE_SAMPLES = 20001
DEFAULT_FLIP_RZ_SIGN = True
DEFAULT_OFFPLANE_SIGN = -1.0

DEFAULT_POINT_X = 100.0
DEFAULT_POINT_Y = 52.0
DEFAULT_POINT_Z = -145.0

DEFAULT_DANCE_REPEATS = 40
DEFAULT_DANCE_POINTS_PER_REPEAT = 300

# Waltz / valse defaults: broad sweep with a 3-beat pulse.
DEFAULT_DANCE_X_AMPLITUDE = 10.0
DEFAULT_DANCE_Y_AMPLITUDE = 5.0
DEFAULT_DANCE_Z_AMPLITUDE = 6.0
DEFAULT_DANCE_ORBIT_CYCLES = 1.0
DEFAULT_DANCE_B_CENTER = 88.0
DEFAULT_DANCE_B_AMPLITUDE = 28.0
DEFAULT_DANCE_B_CYCLES = 1.0
DEFAULT_DANCE_C_CENTER = 180.0
DEFAULT_DANCE_C_AMPLITUDE = 180.0
DEFAULT_DANCE_C_CYCLES = 4.5
DEFAULT_DANCE_RETURN_TO_CENTER = True

# Valse rhythm shaping.
DEFAULT_VALSE_SWING = 0.22
DEFAULT_VALSE_PULSE_SHARPNESS = 1.4
DEFAULT_VALSE_BEAT1_WEIGHT = 1.00
DEFAULT_VALSE_BEAT2_WEIGHT = 0.55
DEFAULT_VALSE_BEAT3_WEIGHT = 0.72
DEFAULT_VALSE_RISE = 0.65
DEFAULT_VALSE_SWAY = 0.45
DEFAULT_VALSE_C_ACCENT = 0.35
DEFAULT_VALSE_B_ACCENT = 0.55


@dataclass
class Calibration:
    r_model: Dict[str, Any]
    z_model: Dict[str, Any]
    y_off_model: Optional[Dict[str, Any]]
    tip_angle_model: Optional[Dict[str, Any]]
    phase_models: Dict[str, Dict[str, Any]]
    default_motion_phase: str
    b_min: float
    b_max: float
    x_axis: str
    y_axis: str
    z_axis: str
    b_axis: str
    c_axis: str
    c_180_deg: float


@dataclass
class PlannedPoint:
    tip_xyz: np.ndarray
    stage_xyz: np.ndarray
    b: float
    c: float
    segment: str
    motion_phase: str


def poly_eval(coeffs: Any, u: Any, default_if_none: Optional[float] = None) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    if coeffs is None:
        if default_if_none is None:
            raise ValueError("Missing polynomial coefficients.")
        return np.full_like(u, float(default_if_none), dtype=float)
    arr = np.asarray(coeffs, dtype=float).reshape(-1)
    if arr.size == 0:
        if default_if_none is None:
            raise ValueError("Polynomial coefficients are empty.")
        return np.full_like(u, float(default_if_none), dtype=float)
    return np.polyval(arr, u)


def eval_model(model: Optional[Dict[str, Any]], u: Any, default_if_none: Optional[float] = None) -> np.ndarray:
    u_arr = np.asarray(u, dtype=float)
    if model is None:
        if default_if_none is None:
            raise ValueError("Missing fit model.")
        return np.full_like(u_arr, float(default_if_none), dtype=float)

    model_type = str(model.get("model_type", "polynomial")).strip().lower()
    if model_type == "polynomial":
        return poly_eval(model.get("coefficients"), u_arr, default_if_none=default_if_none)

    if model_type == "pchip":
        x_knots = np.asarray(model.get("x_knots"), dtype=float)
        y_knots = np.asarray(model.get("y_knots"), dtype=float)
        if x_knots.size == 0 or y_knots.size == 0:
            raise ValueError("PCHIP model is missing knots.")
        if PchipInterpolator is not None:
            interp = PchipInterpolator(x_knots, y_knots, extrapolate=True)
            return np.asarray(interp(u_arr), dtype=float)
        return np.interp(u_arr, x_knots, y_knots)

    raise ValueError(f"Unsupported fit model type: {model_type}")


def legacy_poly_model(coeffs: Any, value_name: str) -> Optional[Dict[str, Any]]:
    if coeffs is None:
        return None
    coeff_list = np.asarray(coeffs, dtype=float).reshape(-1).tolist()
    if not coeff_list:
        return None
    return {
        "model_type": "polynomial",
        "basis": "monomial",
        "input_axis": "b_motor",
        "value_name": value_name,
        "coefficients": coeff_list,
    }


def _normalize_phase_name(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().lower()
    return text or None


def _extract_phase_models(data: Dict[str, Any]) -> Tuple[Dict[str, Dict[str, Any]], str]:
    phase_payload = data.get("fit_models_by_phase") or {}
    phase_models: Dict[str, Dict[str, Any]] = {}
    for raw_name, models in phase_payload.items():
        phase_name = _normalize_phase_name(raw_name)
        if phase_name is None or not isinstance(models, dict):
            continue
        phase_models[phase_name] = dict(models)

    default_phase = _normalize_phase_name(data.get("default_phase_for_legacy_access"))
    if default_phase is None or default_phase not in phase_models:
        if "pull" in phase_models:
            default_phase = "pull"
        elif phase_models:
            default_phase = next(iter(phase_models))
        else:
            default_phase = "pull"
    return phase_models, default_phase


def load_calibration(json_path: str) -> Calibration:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")

    with p.open("r", encoding="utf-8") as f:
        data = json.load(f)

    phase_models, default_phase = _extract_phase_models(data)
    fit_models = data.get("fit_models", {}) or {}
    cubic = data.get("cubic_coefficients", {}) or {}
    default_phase_models = phase_models.get(default_phase, {})

    r_model = fit_models.get("r") or default_phase_models.get("r") or legacy_poly_model(cubic.get("r_coeffs"), "r")
    z_model = fit_models.get("z") or default_phase_models.get("z") or legacy_poly_model(cubic.get("z_coeffs"), "z")
    y_off_model = fit_models.get("offplane_y") or default_phase_models.get("offplane_y") or legacy_poly_model(cubic.get("offplane_y_coeffs"), "y_offplane_mm")
    tip_angle_model = fit_models.get("tip_angle") or default_phase_models.get("tip_angle") or legacy_poly_model(cubic.get("tip_angle_coeffs"), "tip_angle_deg")

    if r_model is None or z_model is None or tip_angle_model is None:
        raise ValueError("Calibration JSON is missing required r/z/tip-angle models.")

    motor_setup = data.get("motor_setup", {}) or {}
    duet_map = data.get("duet_axis_mapping", {}) or {}

    b_range = motor_setup.get("b_motor_position_range", [-5.4, 0.0])
    b_min, b_max = map(float, b_range)
    if b_min > b_max:
        b_min, b_max = b_max, b_min

    return Calibration(
        r_model=r_model,
        z_model=z_model,
        y_off_model=y_off_model,
        tip_angle_model=tip_angle_model,
        phase_models=phase_models,
        default_motion_phase=default_phase,
        b_min=b_min,
        b_max=b_max,
        x_axis=str(duet_map.get("horizontal_axis") or motor_setup.get("horizontal_axis") or "X"),
        y_axis=str(duet_map.get("depth_axis") or motor_setup.get("depth_axis") or "Y"),
        z_axis=str(duet_map.get("vertical_axis") or motor_setup.get("vertical_axis") or "Z"),
        b_axis=str(duet_map.get("pull_axis") or motor_setup.get("b_motor_axis") or "B"),
        c_axis=str(duet_map.get("rotation_axis") or motor_setup.get("rotation_axis") or "C"),
        c_180_deg=float(motor_setup.get("rotation_axis_180_deg", 180.0)),
    )


def _select_fit_model(cal: Calibration, model_name: str, motion_phase: Optional[str]) -> Optional[Dict[str, Any]]:
    phase_name = _normalize_phase_name(motion_phase) or cal.default_motion_phase
    phase_model = cal.phase_models.get(phase_name, {}).get(model_name)
    if phase_model is not None:
        return phase_model
    return {
        "r": cal.r_model,
        "z": cal.z_model,
        "offplane_y": cal.y_off_model,
        "tip_angle": cal.tip_angle_model,
    }[model_name]


def eval_r(cal: Calibration, b: Any, flip_rz_sign: bool, motion_phase: Optional[str]) -> np.ndarray:
    sign = -1.0 * (-1.0 if bool(flip_rz_sign) else 1.0)
    return sign * eval_model(_select_fit_model(cal, "r", motion_phase), b)


def eval_z(cal: Calibration, b: Any, motion_phase: Optional[str]) -> np.ndarray:
    return eval_model(_select_fit_model(cal, "z", motion_phase), b)


def eval_offplane_y(cal: Calibration, b: Any, offplane_sign: float, motion_phase: Optional[str]) -> np.ndarray:
    return float(offplane_sign) * eval_model(_select_fit_model(cal, "offplane_y", motion_phase), b, default_if_none=0.0)


def eval_tip_angle_deg(cal: Calibration, b: Any, motion_phase: Optional[str]) -> np.ndarray:
    return eval_model(_select_fit_model(cal, "tip_angle", motion_phase), b)


def predict_tip_offset_xyz(
    cal: Calibration,
    b: float,
    c_deg: float,
    flip_rz_sign: bool,
    offplane_sign: float,
    motion_phase: Optional[str],
) -> np.ndarray:
    r = float(eval_r(cal, b, flip_rz_sign=flip_rz_sign, motion_phase=motion_phase))
    z = float(eval_z(cal, b, motion_phase=motion_phase))
    y_off = float(eval_offplane_y(cal, b, offplane_sign=offplane_sign, motion_phase=motion_phase))
    c_rad = math.radians(float(c_deg))
    x = r * math.cos(c_rad) - y_off * math.sin(c_rad)
    y = r * math.sin(c_rad) + y_off * math.cos(c_rad)
    return np.array([x, y, z], dtype=float)


def stage_xyz_for_fixed_tip(
    cal: Calibration,
    tip_xyz: np.ndarray,
    b: float,
    c_deg: float,
    flip_rz_sign: bool,
    offplane_sign: float,
    motion_phase: Optional[str],
) -> np.ndarray:
    return np.asarray(tip_xyz, dtype=float) - predict_tip_offset_xyz(
        cal=cal,
        b=b,
        c_deg=c_deg,
        flip_rz_sign=flip_rz_sign,
        offplane_sign=offplane_sign,
        motion_phase=motion_phase,
    )


def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def clamp_c(c_deg: float) -> float:
    return clamp(float(c_deg), -360.0, 360.0)


def smoothstep01(u: float) -> float:
    uu = clamp(float(u), 0.0, 1.0)
    return uu * uu * (3.0 - 2.0 * uu)


def build_tip_angle_inverse_table(
    cal: Calibration,
    num_samples: int,
    motion_phase: Optional[str],
) -> Tuple[np.ndarray, np.ndarray]:
    b_samples = np.linspace(float(cal.b_min), float(cal.b_max), max(1000, int(num_samples)), dtype=float)
    angle_samples = eval_tip_angle_deg(cal, b_samples, motion_phase=motion_phase)
    order = np.argsort(angle_samples)
    angle_sorted = np.asarray(angle_samples[order], dtype=float)
    b_sorted = np.asarray(b_samples[order], dtype=float)
    angle_unique, unique_idx = np.unique(angle_sorted, return_index=True)
    b_unique = b_sorted[unique_idx]
    if angle_unique.size < 2:
        raise ValueError("Could not build a usable tip-angle inverse table.")
    return angle_unique, b_unique


def inverse_tip_angle(
    requested_tip_angle_deg: float,
    angle_table_deg: np.ndarray,
    b_table: np.ndarray,
) -> Tuple[float, float]:
    amin = float(angle_table_deg[0])
    amax = float(angle_table_deg[-1])
    used_angle = clamp(float(requested_tip_angle_deg), amin, amax)
    b_val = float(np.interp(used_angle, angle_table_deg, b_table))
    return b_val, used_angle


class BAngleSolver:
    def __init__(self, cal: Calibration, inverse_samples: int) -> None:
        self.cal = cal
        self.tables: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
        for phase in [cal.default_motion_phase, "pull", "release"]:
            if phase in self.tables:
                continue
            try:
                self.tables[phase] = build_tip_angle_inverse_table(cal, num_samples=inverse_samples, motion_phase=phase)
            except Exception:
                continue
        if cal.default_motion_phase not in self.tables:
            self.tables[cal.default_motion_phase] = build_tip_angle_inverse_table(
                cal, num_samples=inverse_samples, motion_phase=None
            )

    def solve(self, target_tip_angle_deg: float, prev_b: Optional[float]) -> float:
        candidates: List[float] = []
        for angle_table, b_table in self.tables.values():
            b_val, _ = inverse_tip_angle(target_tip_angle_deg, angle_table, b_table)
            candidates.append(float(clamp(b_val, self.cal.b_min, self.cal.b_max)))
        if not candidates:
            raise RuntimeError("No inverse tables were available for B solving.")
        if prev_b is None:
            return float(candidates[0])
        arr = np.asarray(candidates, dtype=float)
        return float(arr[np.argmin(np.abs(arr - float(prev_b)))])


def infer_motion_phase(prev_b: Optional[float], curr_b: float, next_b: Optional[float], default_phase: str) -> str:
    deltas: List[float] = []
    if prev_b is not None:
        deltas.append(float(curr_b) - float(prev_b))
    if next_b is not None:
        deltas.append(float(next_b) - float(curr_b))
    for delta in deltas:
        if abs(delta) <= 1e-9:
            continue
        return "pull" if delta < 0.0 else "release"
    return default_phase


def resolve_stage_positions(
    cal: Calibration,
    raw_points: List[Tuple[np.ndarray, float, float, str]],
    flip_rz_sign: bool,
    offplane_sign: float,
) -> List[PlannedPoint]:
    planned: List[PlannedPoint] = []
    b_values = [float(p[1]) for p in raw_points]
    for i, (tip_xyz, b, c, segment) in enumerate(raw_points):
        prev_b = None if i == 0 else b_values[i - 1]
        next_b = None if i + 1 >= len(b_values) else b_values[i + 1]
        motion_phase = infer_motion_phase(prev_b, b, next_b, cal.default_motion_phase)
        stage_xyz = stage_xyz_for_fixed_tip(
            cal=cal,
            tip_xyz=np.asarray(tip_xyz, dtype=float),
            b=float(b),
            c_deg=float(c),
            flip_rz_sign=flip_rz_sign,
            offplane_sign=offplane_sign,
            motion_phase=motion_phase,
        )
        planned.append(
            PlannedPoint(
                tip_xyz=np.asarray(tip_xyz, dtype=float),
                stage_xyz=stage_xyz,
                b=float(b),
                c=float(c),
                segment=segment,
                motion_phase=motion_phase,
            )
        )
    return planned


def valse_pulse(phase_01: float, sharpness: float, beat_weights: Tuple[float, float, float]) -> float:
    phase = float(phase_01) % 1.0
    beat_float = 3.0 * phase
    beat_idx = min(2, int(math.floor(beat_float)))
    local = beat_float - float(beat_idx)
    window = math.sin(math.pi * local)
    window = max(0.0, window) ** max(0.2, float(sharpness))
    return float(beat_weights[beat_idx]) * window


def generate_valse_segment(
    solver: BAngleSolver,
    anchor_tip_xyz: np.ndarray,
    repeats: int,
    points_per_repeat: int,
    x_amplitude: float,
    y_amplitude: float,
    z_amplitude: float,
    orbit_cycles: float,
    b_center: float,
    b_amplitude: float,
    b_cycles: float,
    c_center: float,
    c_amplitude: float,
    c_cycles: float,
    swing: float,
    pulse_sharpness: float,
    beat1_weight: float,
    beat2_weight: float,
    beat3_weight: float,
    rise: float,
    sway: float,
    c_accent: float,
    b_accent: float,
    return_to_center: bool,
) -> List[Tuple[np.ndarray, float, float, str]]:
    raw: List[Tuple[np.ndarray, float, float, str]] = []
    prev_b: Optional[float] = None
    anchor = np.asarray(anchor_tip_xyz, dtype=float)
    n_repeat = max(1, int(repeats))
    n_pts = max(24, int(points_per_repeat))
    beat_weights = (float(beat1_weight), float(beat2_weight), float(beat3_weight))

    for rep in range(n_repeat):
        for i in range(n_pts):
            if raw and i == 0:
                continue

            u = i / float(n_pts - 1)
            if return_to_center:
                # Each repeat is a self-contained waltz bar that returns home.
                bar_phase = u
            else:
                bar_phase = (rep + u) / float(n_repeat)

            # ONE-two-three rhythmic pulse.
            pulse = valse_pulse(bar_phase, sharpness=pulse_sharpness, beat_weights=beat_weights)
            pulse_lag = valse_pulse((bar_phase - 0.08) % 1.0, sharpness=pulse_sharpness, beat_weights=beat_weights)

            theta = 2.0 * math.pi * float(orbit_cycles) * bar_phase
            swing_theta = theta + float(swing) * math.sin(2.0 * math.pi * 3.0 * bar_phase)

            x = anchor[0] + float(x_amplitude) * math.cos(swing_theta)
            y = anchor[1] + float(y_amplitude) * (
                math.sin(swing_theta) + float(sway) * pulse_lag * math.sin(theta + math.pi / 6.0)
            )
            z = anchor[2] + float(z_amplitude) * (
                -0.40 * math.cos(theta)
                + float(rise) * pulse
                - 0.18 * pulse_lag
            )
            target_tip = np.array([x, y, z], dtype=float)

            target_b_angle = clamp(
                float(b_center)
                + float(b_amplitude) * (
                    0.72 * math.sin(2.0 * math.pi * float(b_cycles) * bar_phase - math.pi / 3.0)
                    + float(b_accent) * (pulse - 0.45 * pulse_lag)
                ),
                0.0,
                180.0,
            )

            target_c = clamp_c(
                float(c_center)
                + float(c_amplitude) * (
                    0.78 * math.sin(2.0 * math.pi * float(c_cycles) * bar_phase)
                    + float(c_accent) * math.sin(2.0 * math.pi * 3.0 * bar_phase - math.pi / 8.0)
                    + 0.10 * pulse
                )
            )

            b_cmd = solver.solve(target_b_angle, prev_b=prev_b)
            raw.append((target_tip, b_cmd, target_c, f"valse_{rep + 1:02d}"))
            prev_b = b_cmd

    return raw


def write_gcode(
    out_path: str,
    cal: Calibration,
    points: List[PlannedPoint],
    feedrate: float,
    travel_feed: float,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", encoding="utf-8") as f:
        f.write("; generated by dance_valse_40cycles.py\n")
        f.write(
            f"; axes: X->{cal.x_axis}, Y->{cal.y_axis}, Z->{cal.z_axis}, B->{cal.b_axis}, C->{cal.c_axis}\n"
        )
        f.write(f"; write_mode={args.write_mode} feedrate={feedrate:.1f}\n")
        f.write(f"; valse repeats={args.dance_repeats} points_per_repeat={args.dance_points_per_repeat}\n")
        f.write("; calibrated dance motion only: 3-beat waltz/valse rhythm, B curl/uncurl, faster C ornamentation\n")
        f.write("G90\n")
        f.write("G21\n")

        if not points:
            raise ValueError("No motion points were generated.")

        first = points[0]
        f.write("; ---- move to start ----\n")
        f.write(
            f"G1 {cal.x_axis}{first.stage_xyz[0]:.3f} {cal.y_axis}{first.stage_xyz[1]:.3f} "
            f"{cal.z_axis}{first.stage_xyz[2]:.3f} {cal.b_axis}{first.b:.3f} {cal.c_axis}{first.c:.3f} "
            f"F{travel_feed:.0f}\n"
        )

        current_segment = first.segment
        f.write(f"; ---- segment: {current_segment} ----\n")
        for p in points[1:]:
            if p.segment != current_segment:
                current_segment = p.segment
                f.write(f"; ---- segment: {current_segment} ----\n")
            f.write(
                f"G1 {cal.x_axis}{p.stage_xyz[0]:.3f} {cal.y_axis}{p.stage_xyz[1]:.3f} "
                f"{cal.z_axis}{p.stage_xyz[2]:.3f} {cal.b_axis}{p.b:.3f} {cal.c_axis}{p.c:.3f} "
                f"F{feedrate:.0f}\n"
            )

        last = points[-1]
        f.write("; ---- end ----\n")
        f.write(
            f"; final tip xyz = ({last.tip_xyz[0]:.3f}, {last.tip_xyz[1]:.3f}, {last.tip_xyz[2]:.3f})\n"
        )

    stage_xyz = np.asarray([p.stage_xyz for p in points], dtype=float)
    tip_xyz = np.asarray([p.tip_xyz for p in points], dtype=float)
    b_vals = np.asarray([p.b for p in points], dtype=float)
    c_vals = np.asarray([p.c for p in points], dtype=float)

    return {
        "out": str(out),
        "point_count": int(len(points)),
        "feedrate": float(feedrate),
        "dance_repeats": int(args.dance_repeats),
        "points_per_repeat": int(args.dance_points_per_repeat),
        "tip_xyz_range": {
            "x": [float(np.min(tip_xyz[:, 0])), float(np.max(tip_xyz[:, 0]))],
            "y": [float(np.min(tip_xyz[:, 1])), float(np.max(tip_xyz[:, 1]))],
            "z": [float(np.min(tip_xyz[:, 2])), float(np.max(tip_xyz[:, 2]))],
        },
        "stage_xyz_range": {
            "x": [float(np.min(stage_xyz[:, 0])), float(np.max(stage_xyz[:, 0]))],
            "y": [float(np.min(stage_xyz[:, 1])), float(np.max(stage_xyz[:, 1]))],
            "z": [float(np.min(stage_xyz[:, 2])), float(np.max(stage_xyz[:, 2]))],
        },
        "b_range_deg": [float(np.min(b_vals)), float(np.max(b_vals))],
        "c_range_deg": [float(np.min(c_vals)), float(np.max(c_vals))],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Generate motion-only calibrated valse/waltz G-code: a 3-beat dance with B curl/uncurl and faster C ornamentation."
    )
    ap.add_argument("--calibration", required=True, help="Calibration JSON path.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code file.")
    ap.add_argument("--write-mode", choices=["calibrated", "cartesian"], default=DEFAULT_WRITE_MODE)
    ap.add_argument("--feedrate", type=float, default=DEFAULT_FEEDRATE)
    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED)
    ap.add_argument("--inverse-samples", type=int, default=DEFAULT_INVERSE_SAMPLES)

    ap.add_argument("--flip-rz-sign", action="store_true", default=DEFAULT_FLIP_RZ_SIGN)
    ap.add_argument("--no-flip-rz-sign", dest="flip_rz_sign", action="store_false")
    ap.add_argument("--offplane-sign", type=float, default=DEFAULT_OFFPLANE_SIGN)

    ap.add_argument("--point-x", type=float, default=DEFAULT_POINT_X)
    ap.add_argument("--point-y", type=float, default=DEFAULT_POINT_Y)
    ap.add_argument("--point-z", type=float, default=DEFAULT_POINT_Z)

    ap.add_argument("--dance-repeats", type=int, default=DEFAULT_DANCE_REPEATS)
    ap.add_argument("--dance-points-per-repeat", type=int, default=DEFAULT_DANCE_POINTS_PER_REPEAT)
    ap.add_argument("--dance-x-amplitude", type=float, default=DEFAULT_DANCE_X_AMPLITUDE)
    ap.add_argument("--dance-y-amplitude", type=float, default=DEFAULT_DANCE_Y_AMPLITUDE)
    ap.add_argument("--dance-z-amplitude", type=float, default=DEFAULT_DANCE_Z_AMPLITUDE)
    ap.add_argument("--dance-orbit-cycles", type=float, default=DEFAULT_DANCE_ORBIT_CYCLES)
    ap.add_argument("--dance-b-center", type=float, default=DEFAULT_DANCE_B_CENTER)
    ap.add_argument("--dance-b-amplitude", type=float, default=DEFAULT_DANCE_B_AMPLITUDE)
    ap.add_argument("--dance-b-cycles", type=float, default=DEFAULT_DANCE_B_CYCLES)
    ap.add_argument("--dance-c-center", type=float, default=DEFAULT_DANCE_C_CENTER)
    ap.add_argument("--dance-c-amplitude", type=float, default=DEFAULT_DANCE_C_AMPLITUDE)
    ap.add_argument("--dance-c-cycles", type=float, default=DEFAULT_DANCE_C_CYCLES)
    ap.add_argument("--dance-return-to-center", action="store_true", default=DEFAULT_DANCE_RETURN_TO_CENTER)
    ap.add_argument("--no-dance-return-to-center", dest="dance_return_to_center", action="store_false")

    ap.add_argument("--valse-swing", type=float, default=DEFAULT_VALSE_SWING)
    ap.add_argument("--valse-pulse-sharpness", type=float, default=DEFAULT_VALSE_PULSE_SHARPNESS)
    ap.add_argument("--valse-beat1-weight", type=float, default=DEFAULT_VALSE_BEAT1_WEIGHT)
    ap.add_argument("--valse-beat2-weight", type=float, default=DEFAULT_VALSE_BEAT2_WEIGHT)
    ap.add_argument("--valse-beat3-weight", type=float, default=DEFAULT_VALSE_BEAT3_WEIGHT)
    ap.add_argument("--valse-rise", type=float, default=DEFAULT_VALSE_RISE)
    ap.add_argument("--valse-sway", type=float, default=DEFAULT_VALSE_SWAY)
    ap.add_argument("--valse-c-accent", type=float, default=DEFAULT_VALSE_C_ACCENT)
    ap.add_argument("--valse-b-accent", type=float, default=DEFAULT_VALSE_B_ACCENT)
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()
    cal = load_calibration(args.calibration)
    solver = BAngleSolver(cal, inverse_samples=args.inverse_samples)

    anchor_tip = np.array([args.point_x, args.point_y, args.point_z], dtype=float)
    raw_points = generate_valse_segment(
        solver=solver,
        anchor_tip_xyz=anchor_tip,
        repeats=args.dance_repeats,
        points_per_repeat=args.dance_points_per_repeat,
        x_amplitude=args.dance_x_amplitude,
        y_amplitude=args.dance_y_amplitude,
        z_amplitude=args.dance_z_amplitude,
        orbit_cycles=args.dance_orbit_cycles,
        b_center=args.dance_b_center,
        b_amplitude=args.dance_b_amplitude,
        b_cycles=args.dance_b_cycles,
        c_center=args.dance_c_center,
        c_amplitude=args.dance_c_amplitude,
        c_cycles=args.dance_c_cycles,
        swing=args.valse_swing,
        pulse_sharpness=args.valse_pulse_sharpness,
        beat1_weight=args.valse_beat1_weight,
        beat2_weight=args.valse_beat2_weight,
        beat3_weight=args.valse_beat3_weight,
        rise=args.valse_rise,
        sway=args.valse_sway,
        c_accent=args.valse_c_accent,
        b_accent=args.valse_b_accent,
        return_to_center=args.dance_return_to_center,
    )

    if args.write_mode == "cartesian":
        planned = [
            PlannedPoint(
                tip_xyz=np.asarray(tip_xyz, dtype=float),
                stage_xyz=np.asarray(tip_xyz, dtype=float),
                b=float(b),
                c=float(c),
                segment=segment,
                motion_phase=cal.default_motion_phase,
            )
            for (tip_xyz, b, c, segment) in raw_points
        ]
    else:
        planned = resolve_stage_positions(
            cal=cal,
            raw_points=raw_points,
            flip_rz_sign=args.flip_rz_sign,
            offplane_sign=args.offplane_sign,
        )

    summary = write_gcode(
        out_path=args.out,
        cal=cal,
        points=planned,
        feedrate=args.feedrate,
        travel_feed=args.travel_feed,
        args=args,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
