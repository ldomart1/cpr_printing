#!/usr/bin/env python3
"""
Generate Duet/RRF-friendly G-code for the torus skin strategy described in the
prompt.

Geometry
--------
- Torus major (centerline) diameter: default 30 mm  -> major radius R = 15 mm
- Torus minor (tube) diameter:       default  8 mm  -> minor radius r =  4 mm
- Default torus center: (100, 60, -140)
- Torus axis is +Y, so the torus centerline lies in the XZ plane.

Path strategy implemented
-------------------------
1) "Planar" phase
   - Start on the largest outer XZ circle at y = center_y.
   - Trace clockwise circles in constant-Y planes.
   - Each next circle steps by -y_step until the y-most plane is reached.
   - Orientation is fixed at B = 90 deg, C = -90 deg.

2) "Side-written" phase
   - Starting from the y-most plane, trace circles again while keeping the tool
     perpendicular to the XZ circle using the angle recipe from the request.
   - For a single circle:
       * start at the top point with C = -180, B = 0
       * right half:  top -> bottom with B going 0 -> -180, C = -180
       * bottom spin: +180 C spin at constant XYZ/B, so C: -180 -> 0
       * left half:   bottom -> top with B going -180 -> 0, C = 0
       * top transition to next circle: move tip to next top point while
         spinning C back 0 -> -180 and keeping B = 0.
   - To match the sentence "until reaching our very first drawn plane circle",
     this second phase runs from the y-most plane back up to the initial
     largest-diameter circle.

Important note
--------------
The description has one internal tension: it says to keep decreasing Y, but it
also says to continue until reaching the very first (largest) circle again.
Those cannot both be true for the outer torus branch. This script resolves that
by making the side-written phase climb from y_min back to y_start, because that
is the only way to return to the first circle.

This script uses direct Cartesian stage motion for the tip path. It does not
reintroduce the calibration-compensated tip-offset model from the reference
wave script.
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
DEFAULT_OUT = "gcode_generation/torus_skin.gcode"

DEFAULT_CENTER_X = 100.0
DEFAULT_CENTER_Y = 60.0
DEFAULT_CENTER_Z = -140.0
DEFAULT_MAJOR_DIAMETER = 30.0
DEFAULT_MINOR_DIAMETER = 8.0
DEFAULT_Y_STEP = 0.40

DEFAULT_PLANAR_POINTS = 361
DEFAULT_SIDE_HALF_POINTS = 181
DEFAULT_SPIN_POINTS = 25
DEFAULT_TOP_TRANSITION_POINTS = 25

DEFAULT_PLANAR_B = 90.0
DEFAULT_PLANAR_C = -90.0
DEFAULT_SIDE_RIGHT_C = -180.0
DEFAULT_SIDE_LEFT_C = 0.0
DEFAULT_SIDE_TOP_B = 0.0
DEFAULT_SIDE_BOTTOM_B = -180.0

DEFAULT_TRAVEL_FEED = 1200.0
DEFAULT_APPROACH_FEED = 500.0
DEFAULT_PRINT_FEED = 250.0
DEFAULT_ROTARY_SPIN_FEED = 180.0
DEFAULT_TOP_TRANSITION_FEED = 220.0
DEFAULT_TRAVEL_LIFT_Z = 8.0

DEFAULT_EMIT_EXTRUSION = True
DEFAULT_EXTRUSION_PER_MM = 0.0015
DEFAULT_PRESSURE_OFFSET_MM = 4.0
DEFAULT_PRESSURE_ADVANCE_FEED = 120.0
DEFAULT_PRESSURE_RETRACT_FEED = 240.0
DEFAULT_PREFLOW_DWELL_MS = 500

DEFAULT_HOME_TO_START = True
DEFAULT_RETURN_HOME = True


# ---------------- Data classes ----------------
@dataclass
class MotionPoint:
    x: float
    y: float
    z: float
    b: float
    c: float

    def xyz(self) -> np.ndarray:
        return np.array([self.x, self.y, self.z], dtype=float)


# ---------------- Geometry helpers ----------------
def torus_outer_radius_at_y(y: float, center_y: float, major_radius: float, minor_radius: float) -> float:
    dy = float(y) - float(center_y)
    inside = float(minor_radius) ** 2 - dy ** 2
    if inside < -1e-9:
        raise ValueError(
            f"Requested y={y:.6f} is outside the torus minor-radius band "
            f"[{center_y - minor_radius:.6f}, {center_y + minor_radius:.6f}]"
        )
    inside = max(0.0, inside)
    return float(major_radius) + math.sqrt(inside)


def descending_planes(start_y: float, end_y: float, step: float) -> List[float]:
    if step <= 0.0:
        raise ValueError("y_step must be > 0")
    planes: List[float] = [float(start_y)]
    y = float(start_y)
    while y - float(step) > float(end_y) + 1e-12:
        y -= float(step)
        planes.append(float(y))
    if abs(planes[-1] - float(end_y)) > 1e-9:
        planes.append(float(end_y))
    return planes


def top_point_of_circle(center_x: float, y: float, center_z: float, radius: float) -> np.ndarray:
    return np.array([float(center_x), float(y), float(center_z) + float(radius)], dtype=float)


def circle_clockwise_xz(
    center_x: float,
    y: float,
    center_z: float,
    radius: float,
    n_points: int,
    b_deg: float,
    c_deg: float,
) -> List[MotionPoint]:
    if n_points < 5:
        raise ValueError("planar circle needs at least 5 points")

    # Top point first, then clockwise when viewed from +Y.
    thetas = np.linspace(math.pi / 2.0, math.pi / 2.0 - 2.0 * math.pi, int(n_points), endpoint=True)
    pts: List[MotionPoint] = []
    for th in thetas:
        x = float(center_x) + float(radius) * math.cos(float(th))
        z = float(center_z) + float(radius) * math.sin(float(th))
        pts.append(MotionPoint(x=x, y=float(y), z=z, b=float(b_deg), c=float(c_deg)))
    return pts


def side_written_circle(
    center_x: float,
    y: float,
    center_z: float,
    radius: float,
    half_points: int,
    spin_points: int,
    c_right_deg: float,
    c_left_deg: float,
) -> List[MotionPoint]:
    if half_points < 3:
        raise ValueError("side circle half needs at least 3 points")
    if spin_points < 2:
        raise ValueError("spin_points must be at least 2")

    pts: List[MotionPoint] = []

    # Right half: top -> bottom, with B = theta_deg - 90, theta: +90 -> -90.
    right_thetas = np.linspace(math.pi / 2.0, -math.pi / 2.0, int(half_points), endpoint=True)
    for th in right_thetas:
        x = float(center_x) + float(radius) * math.cos(float(th))
        z = float(center_z) + float(radius) * math.sin(float(th))
        b = math.degrees(float(th)) - 90.0
        pts.append(MotionPoint(x=x, y=float(y), z=z, b=float(b), c=float(c_right_deg)))

    # Bottom C spin at constant XYZ/B.
    xb = float(center_x)
    zb = float(center_z) - float(radius)
    cb = np.linspace(float(c_right_deg), float(c_left_deg), int(spin_points), endpoint=True)
    for c in cb[1:]:
        pts.append(MotionPoint(x=xb, y=float(y), z=zb, b=-180.0, c=float(c)))

    # Left half: bottom -> top, same B law, theta: -90 -> +90.
    left_thetas = np.linspace(-math.pi / 2.0, math.pi / 2.0, int(half_points), endpoint=True)
    for th in left_thetas[1:]:
        x = float(center_x) + float(radius) * math.cos(float(th))
        z = float(center_z) + float(radius) * math.sin(float(th))
        b = math.degrees(float(th)) - 90.0
        pts.append(MotionPoint(x=x, y=float(y), z=z, b=float(b), c=float(c_left_deg)))

    return pts


def top_transition_between_circles(
    center_x: float,
    y0: float,
    z0: float,
    y1: float,
    z1: float,
    n_points: int,
    b_deg: float,
    c0_deg: float,
    c1_deg: float,
) -> List[MotionPoint]:
    if n_points < 2:
        raise ValueError("top transition needs at least 2 points")
    ys = np.linspace(float(y0), float(y1), int(n_points), endpoint=True)
    zs = np.linspace(float(z0), float(z1), int(n_points), endpoint=True)
    cs = np.linspace(float(c0_deg), float(c1_deg), int(n_points), endpoint=True)
    pts: List[MotionPoint] = []
    for i in range(1, int(n_points)):
        pts.append(MotionPoint(x=float(center_x), y=float(ys[i]), z=float(zs[i]), b=float(b_deg), c=float(cs[i])))
    return pts


# ---------------- G-code helpers ----------------
def _fmt_axes_move(axes_vals: Sequence[Tuple[str, float]]) -> str:
    return " ".join(f"{ax}{val:.3f}" for ax, val in axes_vals)


class GCodeWriter:
    def __init__(
        self,
        fh,
        travel_feed: float,
        approach_feed: float,
        print_feed: float,
        rotary_spin_feed: float,
        top_transition_feed: float,
        emit_extrusion: bool,
        extrusion_per_mm: float,
        pressure_offset_mm: float,
        pressure_advance_feed: float,
        pressure_retract_feed: float,
        preflow_dwell_ms: int,
        x_axis: str = "X",
        y_axis: str = "Y",
        z_axis: str = "Z",
        b_axis: str = "B",
        c_axis: str = "C",
        u_axis: str = "U",
    ) -> None:
        self.f = fh
        self.travel_feed = float(travel_feed)
        self.approach_feed = float(approach_feed)
        self.print_feed = float(print_feed)
        self.rotary_spin_feed = float(rotary_spin_feed)
        self.top_transition_feed = float(top_transition_feed)
        self.emit_extrusion = bool(emit_extrusion)
        self.extrusion_per_mm = float(extrusion_per_mm)
        self.pressure_offset_mm = float(pressure_offset_mm)
        self.pressure_advance_feed = float(pressure_advance_feed)
        self.pressure_retract_feed = float(pressure_retract_feed)
        self.preflow_dwell_ms = int(preflow_dwell_ms)
        self.x_axis = str(x_axis)
        self.y_axis = str(y_axis)
        self.z_axis = str(z_axis)
        self.b_axis = str(b_axis)
        self.c_axis = str(c_axis)
        self.u_axis = str(u_axis)

        self.cur_point: Optional[MotionPoint] = None
        self.u_material_abs = 0.0
        self.pressure_charged = False

        self.xyz_min = np.array([np.inf, np.inf, np.inf], dtype=float)
        self.xyz_max = np.array([-np.inf, -np.inf, -np.inf], dtype=float)
        self.b_min = float("inf")
        self.b_max = float("-inf")
        self.c_min = float("inf")
        self.c_max = float("-inf")

    def u_cmd_actual(self) -> float:
        return self.u_material_abs + (self.pressure_offset_mm if self.pressure_charged else 0.0)

    def _write_move(self, pt: MotionPoint, feed: float, comment: Optional[str] = None, extrude: bool = False) -> None:
        if comment:
            self.f.write(f"; {comment}\n")

        u_value: Optional[float] = None
        if extrude and self.emit_extrusion and self.cur_point is not None:
            seg_len = float(np.linalg.norm(pt.xyz() - self.cur_point.xyz()))
            self.u_material_abs += self.extrusion_per_mm * seg_len
            u_value = self.u_cmd_actual()

        axes: List[Tuple[str, float]] = [
            (self.x_axis, pt.x),
            (self.y_axis, pt.y),
            (self.z_axis, pt.z),
            (self.b_axis, pt.b),
            (self.c_axis, pt.c),
        ]
        if u_value is not None:
            axes.append((self.u_axis, u_value))

        self.f.write(f"G1 {_fmt_axes_move(axes)} F{float(feed):.0f}\n")
        self.cur_point = MotionPoint(pt.x, pt.y, pt.z, pt.b, pt.c)

        xyz = pt.xyz()
        self.xyz_min = np.minimum(self.xyz_min, xyz)
        self.xyz_max = np.maximum(self.xyz_max, xyz)
        self.b_min = min(self.b_min, float(pt.b))
        self.b_max = max(self.b_max, float(pt.b))
        self.c_min = min(self.c_min, float(pt.c))
        self.c_max = max(self.c_max, float(pt.c))

    def move_no_extrusion(self, pt: MotionPoint, feed: float, comment: Optional[str] = None) -> None:
        self._write_move(pt, feed=feed, comment=comment, extrude=False)

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

    def print_path(self, pts: Sequence[MotionPoint], feed: float, comment: Optional[str] = None) -> None:
        if not pts:
            return
        if comment:
            self.f.write(f"; {comment}\n")
        self.pressure_preload_before_print()
        for i, pt in enumerate(pts):
            if self.cur_point is None or i == 0:
                # First point is a position-only synchronization move.
                self._write_move(pt, feed=feed, comment=None, extrude=False)
            else:
                self._write_move(pt, feed=feed, comment=None, extrude=True)
        self.pressure_release_after_print()

    def emit_header(self, metadata: Dict[str, Any]) -> None:
        self.f.write("; Torus skin G-code\n")
        self.f.write("; Generated by torus_skin_generator.py\n")
        for key, value in metadata.items():
            self.f.write(f"; {key}={value}\n")
        self.f.write("G21\n")
        self.f.write("G90\n")
        self.f.write("; Angle convention used here:\n")
        self.f.write(";   planar phase: B=90, C=-90\n")
        self.f.write(";   side phase: top B=0, right side B=-90, bottom B=-180\n")
        self.f.write(";   side phase right half C=-180, left half C=0\n")


# ---------------- Top-level generation ----------------
def generate_torus_skin_gcode(
    out: str,
    center_x: float,
    center_y: float,
    center_z: float,
    major_diameter: float,
    minor_diameter: float,
    y_step: float,
    planar_points: int,
    side_half_points: int,
    spin_points: int,
    top_transition_points: int,
    planar_b: float,
    planar_c: float,
    side_right_c: float,
    side_left_c: float,
    travel_feed: float,
    approach_feed: float,
    print_feed: float,
    rotary_spin_feed: float,
    top_transition_feed: float,
    travel_lift_z: float,
    emit_extrusion: bool,
    extrusion_per_mm: float,
    pressure_offset_mm: float,
    pressure_advance_feed: float,
    pressure_retract_feed: float,
    preflow_dwell_ms: int,
    home_to_start: bool,
    return_home: bool,
) -> Dict[str, Any]:
    R = 0.5 * float(major_diameter)
    r = 0.5 * float(minor_diameter)
    if R <= 0.0 or r <= 0.0:
        raise ValueError("major_diameter and minor_diameter must be > 0")
    if y_step <= 0.0:
        raise ValueError("y_step must be > 0")

    y_start = float(center_y)
    y_min = float(center_y) - float(r)
    planes_down = descending_planes(y_start, y_min, y_step)
    radii_down = [torus_outer_radius_at_y(y, center_y, R, r) for y in planes_down]

    out_path = Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with out_path.open("w", encoding="utf-8") as fh:
        writer = GCodeWriter(
            fh=fh,
            travel_feed=travel_feed,
            approach_feed=approach_feed,
            print_feed=print_feed,
            rotary_spin_feed=rotary_spin_feed,
            top_transition_feed=top_transition_feed,
            emit_extrusion=emit_extrusion,
            extrusion_per_mm=extrusion_per_mm,
            pressure_offset_mm=pressure_offset_mm,
            pressure_advance_feed=pressure_advance_feed,
            pressure_retract_feed=pressure_retract_feed,
            preflow_dwell_ms=preflow_dwell_ms,
        )

        writer.emit_header(
            {
                "center": f"({center_x:.3f}, {center_y:.3f}, {center_z:.3f})",
                "major_diameter_mm": f"{major_diameter:.3f}",
                "minor_diameter_mm": f"{minor_diameter:.3f}",
                "major_radius_mm": f"{R:.3f}",
                "minor_radius_mm": f"{r:.3f}",
                "y_step_mm": f"{y_step:.3f}",
                "planar_phase_planes": len(planes_down),
                "side_phase_planes": len(planes_down),
            }
        )

        first_radius = radii_down[0]
        first_top_xyz = top_point_of_circle(center_x, planes_down[0], center_z, first_radius)
        safe_start = MotionPoint(
            x=float(first_top_xyz[0]),
            y=float(first_top_xyz[1]),
            z=float(first_top_xyz[2] + travel_lift_z),
            b=float(planar_b),
            c=float(planar_c),
        )
        first_start = MotionPoint(
            x=float(first_top_xyz[0]),
            y=float(first_top_xyz[1]),
            z=float(first_top_xyz[2]),
            b=float(planar_b),
            c=float(planar_c),
        )

        if home_to_start:
            writer.move_no_extrusion(safe_start, feed=travel_feed, comment="move above torus start")
            writer.move_no_extrusion(first_start, feed=approach_feed, comment="approach first planar circle start")

        # Phase 1: planar circles from y_start down to y_min.
        for i, (y, rho) in enumerate(zip(planes_down, radii_down)):
            circle_pts = circle_clockwise_xz(
                center_x=center_x,
                y=y,
                center_z=center_z,
                radius=rho,
                n_points=planar_points,
                b_deg=planar_b,
                c_deg=planar_c,
            )
            if writer.cur_point is None:
                writer.move_no_extrusion(circle_pts[0], feed=approach_feed, comment="sync to planar circle start")
            elif np.linalg.norm(circle_pts[0].xyz() - writer.cur_point.xyz()) > 1e-9 or abs(circle_pts[0].b - writer.cur_point.b) > 1e-9 or abs(circle_pts[0].c - writer.cur_point.c) > 1e-9:
                writer.move_no_extrusion(circle_pts[0], feed=approach_feed, comment="move to planar circle start")

            writer.print_path(circle_pts, feed=print_feed, comment=f"planar phase circle {i + 1}/{len(planes_down)} at y={y:.3f} radius={rho:.3f}")

            if i < len(planes_down) - 1:
                next_y = planes_down[i + 1]
                next_rho = radii_down[i + 1]
                next_top = MotionPoint(
                    x=center_x,
                    y=next_y,
                    z=center_z + next_rho,
                    b=planar_b,
                    c=planar_c,
                )
                writer.move_no_extrusion(next_top, feed=approach_feed, comment="step to next lower planar circle")

        # Re-orient to the first side-written circle at the y-most plane.
        bottom_y = planes_down[-1]
        bottom_rho = radii_down[-1]
        bottom_top = MotionPoint(
            x=center_x,
            y=bottom_y,
            z=center_z + bottom_rho,
            b=0.0,
            c=side_right_c,
        )
        writer.move_no_extrusion(bottom_top, feed=approach_feed, comment="reorient for side-written phase at y-most circle")

        # Phase 2: side-written circles from y_min back up to y_start.
        planes_up = list(reversed(planes_down))
        radii_up = list(reversed(radii_down))
        for i, (y, rho) in enumerate(zip(planes_up, radii_up)):
            side_pts = side_written_circle(
                center_x=center_x,
                y=y,
                center_z=center_z,
                radius=rho,
                half_points=side_half_points,
                spin_points=spin_points,
                c_right_deg=side_right_c,
                c_left_deg=side_left_c,
            )

            if writer.cur_point is None or np.linalg.norm(side_pts[0].xyz() - writer.cur_point.xyz()) > 1e-9 or abs(side_pts[0].b - writer.cur_point.b) > 1e-9 or abs(side_pts[0].c - writer.cur_point.c) > 1e-9:
                writer.move_no_extrusion(side_pts[0], feed=approach_feed, comment="move to side-written circle start")

            # Break the side-written circle into three motion-rate regions.
            right_half_count = side_half_points
            spin_count = max(0, spin_points - 1)
            left_half_count = side_half_points - 1

            writer.f.write(f"; side phase circle {i + 1}/{len(planes_up)} at y={y:.3f} radius={rho:.3f}\n")
            writer.pressure_preload_before_print()

            for j, pt in enumerate(side_pts):
                if j == 0:
                    writer.move_no_extrusion(pt, feed=approach_feed)
                    continue

                if j < right_half_count:
                    feed = print_feed
                elif j < right_half_count + spin_count:
                    feed = rotary_spin_feed
                else:
                    feed = print_feed
                writer._write_move(pt, feed=feed, comment=None, extrude=True)

            writer.pressure_release_after_print()

            if i < len(planes_up) - 1:
                next_y = planes_up[i + 1]
                next_rho = radii_up[i + 1]
                trans_pts = top_transition_between_circles(
                    center_x=center_x,
                    y0=y,
                    z0=center_z + rho,
                    y1=next_y,
                    z1=center_z + next_rho,
                    n_points=top_transition_points,
                    b_deg=0.0,
                    c0_deg=side_left_c,
                    c1_deg=side_right_c,
                )
                writer.f.write("; top transition to next side-written circle\n")
                writer.pressure_preload_before_print()
                for pt in trans_pts:
                    writer._write_move(pt, feed=top_transition_feed, comment=None, extrude=True)
                writer.pressure_release_after_print()

        if return_home:
            final_safe = MotionPoint(
                x=center_x,
                y=planes_down[0],
                z=center_z + radii_down[0] + travel_lift_z,
                b=planar_b,
                c=planar_c,
            )
            writer.move_no_extrusion(final_safe, feed=travel_feed, comment="lift above torus after final circle")
            writer.move_no_extrusion(safe_start, feed=travel_feed, comment="return to home start position")

        writer.f.write("; End of file\n")

    summary = {
        "out": str(out_path),
        "center": [float(center_x), float(center_y), float(center_z)],
        "major_radius_mm": float(R),
        "minor_radius_mm": float(r),
        "planar_y_planes": [float(v) for v in planes_down],
        "planar_circle_radii_mm": [float(v) for v in radii_down],
        "xyz_range": {
            "x": [float(center_x - (R + r)), float(center_x + (R + r))],
            "y": [float(y_min), float(y_start)],
            "z": [float(center_z - (R + r)), float(center_z + (R + r) + travel_lift_z)],
        },
    }
    return summary


# ---------------- CLI ----------------
def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=(
            "Generate G-code for a torus skin using a two-phase strategy: "
            "(1) planar clockwise XZ circles with fixed B/C, then "
            "(2) side-written circles using the requested B/C choreography."
        )
    )
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code file")

    ap.add_argument("--center-x", type=float, default=DEFAULT_CENTER_X)
    ap.add_argument("--center-y", type=float, default=DEFAULT_CENTER_Y)
    ap.add_argument("--center-z", type=float, default=DEFAULT_CENTER_Z)
    ap.add_argument("--major-diameter", type=float, default=DEFAULT_MAJOR_DIAMETER, help="Torus centerline diameter")
    ap.add_argument("--minor-diameter", type=float, default=DEFAULT_MINOR_DIAMETER, help="Torus tube diameter")
    ap.add_argument("--y-step", type=float, default=DEFAULT_Y_STEP, help="Plane-to-plane offset in Y, e.g. line thickness")

    ap.add_argument("--planar-points", type=int, default=DEFAULT_PLANAR_POINTS, help="Samples per planar circle")
    ap.add_argument("--side-half-points", type=int, default=DEFAULT_SIDE_HALF_POINTS, help="Samples per half of a side-written circle")
    ap.add_argument("--spin-points", type=int, default=DEFAULT_SPIN_POINTS, help="Samples used for the bottom +180 C spin")
    ap.add_argument("--top-transition-points", type=int, default=DEFAULT_TOP_TRANSITION_POINTS, help="Samples used for the top move/C-spin into the next side-written circle")

    ap.add_argument("--planar-b", type=float, default=DEFAULT_PLANAR_B)
    ap.add_argument("--planar-c", type=float, default=DEFAULT_PLANAR_C)
    ap.add_argument("--side-right-c", type=float, default=DEFAULT_SIDE_RIGHT_C)
    ap.add_argument("--side-left-c", type=float, default=DEFAULT_SIDE_LEFT_C)

    ap.add_argument("--travel-feed", type=float, default=DEFAULT_TRAVEL_FEED)
    ap.add_argument("--approach-feed", type=float, default=DEFAULT_APPROACH_FEED)
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED)
    ap.add_argument("--rotary-spin-feed", type=float, default=DEFAULT_ROTARY_SPIN_FEED)
    ap.add_argument("--top-transition-feed", type=float, default=DEFAULT_TOP_TRANSITION_FEED)
    ap.add_argument("--travel-lift-z", type=float, default=DEFAULT_TRAVEL_LIFT_Z)

    ap.add_argument("--emit-extrusion", dest="emit_extrusion", action="store_true", default=DEFAULT_EMIT_EXTRUSION)
    ap.add_argument("--no-emit-extrusion", dest="emit_extrusion", action="store_false")
    ap.add_argument("--extrusion-per-mm", type=float, default=DEFAULT_EXTRUSION_PER_MM)
    ap.add_argument("--pressure-offset-mm", type=float, default=DEFAULT_PRESSURE_OFFSET_MM)
    ap.add_argument("--pressure-advance-feed", type=float, default=DEFAULT_PRESSURE_ADVANCE_FEED)
    ap.add_argument("--pressure-retract-feed", type=float, default=DEFAULT_PRESSURE_RETRACT_FEED)
    ap.add_argument("--preflow-dwell-ms", type=int, default=DEFAULT_PREFLOW_DWELL_MS)

    ap.add_argument("--home-to-start", dest="home_to_start", action="store_true", default=DEFAULT_HOME_TO_START)
    ap.add_argument("--no-home-to-start", dest="home_to_start", action="store_false")
    ap.add_argument("--return-home", dest="return_home", action="store_true", default=DEFAULT_RETURN_HOME)
    ap.add_argument("--no-return-home", dest="return_home", action="store_false")
    return ap


def main() -> None:
    ap = build_arg_parser()
    args = ap.parse_args()
    summary = generate_torus_skin_gcode(**vars(args))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
