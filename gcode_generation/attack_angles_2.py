#!/usr/bin/env python3
"""
Generate G-code for the custom XZ-plane calibration/path geometry described by the user.

What this script does
---------------------
1) Figure 1 on plane Y = start_y (non-tangential):
   - Horizontal line (0 deg) and "roof" polylines with +a then -a slope, a = 5..45 deg
   - B fixed at 0, C fixed at 0
   - Return after each line: Z up +50, X back, Z down

2) Figure 1 repeated on plane Y = start_y + 20 (tangential):
   - Uses B solved from calibration tip-angle polynomial
   - C fixed at 180 deg (from calibration rotation_axis_180_deg if provided)
   - Mapping for polynomial angle:
       +slope  -> poly angle = slope
       -slope  -> poly angle = 90 + abs(slope)
   - Return after each line: Z up +50, X back, Z down

3) Figure 3 on plane Y = start_y (non-tangential):
   - Bottom half (from origin x+100): lines at 135..90 deg
       * 135..95 terminate at z_rel = +40
       * 90 deg extends to z_rel = +80
       * B fixed at 0, C fixed at 180
       * Return to origin between bottom-half lines: X first, then Z down
   - Top half (same figure): lines from ends of 95..135 bottom-half lines to top vertex
       * These are effectively 85..45 deg from endpoint -> top vertex
       * B fixed at 0, C fixed at 0
       * Reposition from top vertex to next startpoint: X first, then Z down

4) Figure 3 bottom half on plane Y = start_y + 20 (tangential):
   - Same bottom-half geometry as step 3
   - B solved tangentially from polynomial angle = (geom_angle - 90), e.g. 135 -> 45
   - DEFAULT C = 180 deg (configurable; see --fig3-bottom-tangent-c-mode)
     NOTE: the prompt has a contradiction for this; default is chosen from the example sentence.
   - Return to origin: X positive first, then Z down

5) Figure 3 top half on plane Y = start_y + 40 (tangential):
   - Start every line from the top common vertex
   - Draw "90-degree line" first, then 95..135 labels (i.e. top directions 270..225 deg)
   - Polynomial mapping for top-half travel direction:
       top_geom_angle = 360 - base_angle
       poly angle = top_geom_angle - 90
       examples: 270 -> 180, 225 -> 135
   - C fixed at 0
   - Return to top vertex between lines: move up in Z, then move X back

Extrusion behavior (kept from your sample pattern)
--------------------------------------------------
Before each printed line:
  - Advance U by pressure_offset_mm
  - Dwell preflow_dwell_ms
During the line:
  - Coordinated U extrusion by extrusion_per_mm (absolute U mode)
After each printed line:
  - Dwell node_dwell_ms
  - Retract U by pressure_offset_mm

Feeds
-----
- C-only moves use c_feed (default 5000)
- XYZ/B print/travel moves use motion_feed/print_feed (default 300)
- U-only pressure preload/retract feeds are separate (defaults from sample style)

Example
-------
python3 generate_custom_xz_geometry.py \
  --calibration calibration.json \
  --out custom_xz_geometry.gcode \
  --start-x 20 --start-y 0 --start-z -110 \
  --extrusion-per-mm 0.03

"""

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

# ---------------- Defaults ----------------
DEFAULT_OUT = "custom_xz_geometry.gcode"

# User-param starting coordinate (TIP-space / design frame)
DEFAULT_START_X = 20.0
DEFAULT_START_Y = 40.0
DEFAULT_START_Z = -110.0

# Geometry
DEFAULT_LINE_LENGTH = 80.0
DEFAULT_HALF_X = 40.0
DEFAULT_ANGLE_STEP = 5
DEFAULT_MAX_ANGLE = 45
DEFAULT_RETURN_LIFT_Z = 50.0

DEFAULT_FIG3_X_OFFSET = 100.0
DEFAULT_FIG3_MID_Z_REL = 40.0
DEFAULT_FIG3_TOP_Z_REL = 80.0

# Motion feeds
DEFAULT_MOTION_FEED = 300.0   # XYZ/B (travel)
DEFAULT_PRINT_FEED = 300.0    # XYZ/B/U coordinated print
DEFAULT_C_FEED = 5000.0       # C-axis only moves

# Extrusion (U axis)
DEFAULT_EXTRUSION_PER_MM = 0.0  # set >0 to enable line extrusion
DEFAULT_PRIME_MM = 0.0
DEFAULT_PRESSURE_OFFSET_MM = 5.0
DEFAULT_PRESSURE_ADVANCE_FEED = 120.0
DEFAULT_PRESSURE_RETRACT_FEED = 240.0
DEFAULT_PREFLOW_DWELL_MS = 150
DEFAULT_NODE_DWELL_MS = 250

# B solve
DEFAULT_B_SEARCH_SAMPLES = 2001
DEFAULT_B_REFINE_PASSES = 3

# Non-tangential fixed B
DEFAULT_B_NON_TANGENT = 0.0

# Tangential figure 3 bottom half C default:
# Prompt had a contradiction; defaulting to 180 based on the explicit example sentence.
DEFAULT_FIG3_BOTTOM_TANGENT_C_MODE = "180"  # choices: "0", "180"

# Syringe/tube math diagnostics (optional)
DEFAULT_SYRINGE_MM_PER_ML = 6.0
DEFAULT_TUBE_ID_INCH = 0.02


# ---------------- Data classes ----------------
@dataclass
class Calibration:
    r_coeffs: Tuple[float, float, float, float]
    z_coeffs: Tuple[float, float, float, float]
    tip_angle_coeffs: Tuple[float, float, float, float]
    b_min: float
    b_max: float
    x_axis: str
    y_axis: str
    z_axis: str
    b_axis: str
    c_axis: str
    u_axis: str
    c_180_deg: float


@dataclass
class LinePiece:
    p0_tip: Tuple[float, float, float]
    p1_tip: Tuple[float, float, float]
    b_cmd: float
    c_cmd: float
    label: str = ""


# ---------------- Math / geometry helpers ----------------
def poly3_eval(coeffs: Sequence[float], x: float) -> float:
    a, b, c, d = [float(v) for v in coeffs]
    return ((a * x + b) * x + c) * x + d


def dist3(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    return math.sqrt((a[0]-b[0])**2 + (a[1]-b[1])**2 + (a[2]-b[2])**2)


def almost_equal(a: float, b: float, tol: float = 1e-9) -> bool:
    return abs(a - b) <= tol


def point(x: float, y: float, z: float) -> Tuple[float, float, float]:
    return (float(x), float(y), float(z))


def endpoint_from_origin_angle_to_zrel(
    origin_tip: Tuple[float, float, float],
    angle_deg: float,
    z_rel: float,
) -> Tuple[float, float, float]:
    """
    Ray from origin in XZ plane at angle (deg from +X, CCW in XZ).
    Return point where delta-Z reaches z_rel.
    """
    ox, oy, oz = origin_tip
    th = math.radians(angle_deg)
    s = math.sin(th)
    c = math.cos(th)
    if abs(s) < 1e-12:
        raise ValueError(f"Angle {angle_deg} deg cannot reach z_rel={z_rel} (sin ~ 0).")
    t = float(z_rel) / s
    dx = t * c
    dz = t * s  # should equal z_rel numerically
    return point(ox + dx, oy, oz + dz)


def tube_area_mm2_from_id_inch(id_inch: float) -> float:
    d_mm = float(id_inch) * 25.4
    r = 0.5 * d_mm
    return math.pi * r * r


# ---------------- Calibration / kinematics ----------------
def load_calibration(json_path: str) -> Calibration:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")

    data = json.loads(p.read_text())

    cubic = data.get("cubic_coefficients", {})
    r_coeffs = tuple(float(v) for v in cubic["r_coeffs"])
    z_coeffs = tuple(float(v) for v in cubic["z_coeffs"])
    tip_angle_coeffs = tuple(float(v) for v in cubic["tip_angle_coeffs"])

    motor_setup = data.get("motor_setup", {})
    duet_map = data.get("duet_axis_mapping", {})

    b_range = motor_setup.get("b_motor_position_range", [-5.4, 0.0])
    b_min = float(b_range[0])
    b_max = float(b_range[1])
    if b_min > b_max:
        b_min, b_max = b_max, b_min

    x_axis = str(duet_map.get("horizontal_axis", "X"))
    y_axis = str(duet_map.get("depth_axis", "Y"))
    z_axis = str(duet_map.get("vertical_axis", "Z"))
    b_axis = str(duet_map.get("pull_axis", "B"))
    c_axis = str(duet_map.get("rotation_axis", "C"))
    u_axis = str(duet_map.get("extruder_axis", "U"))

    c_180 = float(motor_setup.get("rotation_axis_180_deg", 180.0))

    return Calibration(
        r_coeffs=r_coeffs,
        z_coeffs=z_coeffs,
        tip_angle_coeffs=tip_angle_coeffs,
        b_min=b_min,
        b_max=b_max,
        x_axis=x_axis,
        y_axis=y_axis,
        z_axis=z_axis,
        b_axis=b_axis,
        c_axis=c_axis,
        u_axis=u_axis,
        c_180_deg=c_180,
    )


def eval_r(cal: Calibration, b: float) -> float:
    return poly3_eval(cal.r_coeffs, b)


def eval_z(cal: Calibration, b: float) -> float:
    return poly3_eval(cal.z_coeffs, b)


def eval_tip_angle_poly(cal: Calibration, b: float) -> float:
    """Calibration polynomial output angle in degrees (the 'polynomial angle' space)."""
    return poly3_eval(cal.tip_angle_coeffs, b)


def solve_b_for_poly_angle_deg(
    cal: Calibration,
    target_angle_deg: float,
    search_samples: int = DEFAULT_B_SEARCH_SAMPLES,
    refine_passes: int = DEFAULT_B_REFINE_PASSES,
) -> Tuple[float, float]:
    """
    Solve B by dense sampling (robust to non-monotonic cubic), minimizing |tip_angle_poly(B)-target|.
    Returns (b_cmd, abs_error_deg).
    """
    lo = cal.b_min
    hi = cal.b_max
    n = max(101, int(search_samples))

    best_b = lo
    best_err = float("inf")

    # coarse + refinements
    for rp in range(max(1, refine_passes)):
        step = (hi - lo) / (n - 1)
        for i in range(n):
            b = lo + i * step
            err = abs(eval_tip_angle_poly(cal, b) - target_angle_deg)
            if err < best_err:
                best_err = err
                best_b = b

        # narrow around best for next pass
        if rp < refine_passes - 1:
            half_window = max((hi - lo) * 0.08, (cal.b_max - cal.b_min) * 0.002)
            lo = max(cal.b_min, best_b - half_window)
            hi = min(cal.b_max, best_b + half_window)

    return best_b, best_err


def tip_offset_xyz_physical(cal: Calibration, b: float, c_deg: float) -> Tuple[float, float, float]:
    """
    Physical tip offset from stage origin:
      [ r(B)*cos(C), r(B)*sin(C), z(B) ]
    """
    r = eval_r(cal, b)
    z = eval_z(cal, b)
    c = math.radians(c_deg)
    return (r * math.cos(c), r * math.sin(c), z)


def stage_from_tip(cal: Calibration, p_tip: Tuple[float, float, float], b: float, c_deg: float) -> Tuple[float, float, float]:
    ox, oy, oz = tip_offset_xyz_physical(cal, b, c_deg)
    return (p_tip[0] - ox, p_tip[1] - oy, p_tip[2] - oz)


# ---------------- G-code writer ----------------
class GCodeWriter:
    def __init__(
        self,
        out_path: str,
        cal: Calibration,
        motion_feed: float,
        print_feed: float,
        c_feed: float,
        extrusion_per_mm: float,
        prime_mm: float,
        pressure_offset_mm: float,
        pressure_advance_feed: float,
        pressure_retract_feed: float,
        preflow_dwell_ms: int,
        node_dwell_ms: int,
    ):
        self.out_path = out_path
        self.cal = cal

        self.motion_feed = float(motion_feed)
        self.print_feed = float(print_feed)
        self.c_feed = float(c_feed)

        self.extrusion_per_mm = float(extrusion_per_mm)
        self.emit_extrusion = abs(self.extrusion_per_mm) > 0.0

        self.prime_mm = float(prime_mm)
        self.pressure_offset_mm = float(pressure_offset_mm)
        self.pressure_advance_feed = float(pressure_advance_feed)
        self.pressure_retract_feed = float(pressure_retract_feed)
        self.preflow_dwell_ms = int(preflow_dwell_ms)
        self.node_dwell_ms = int(node_dwell_ms)

        self.fh = open(self.out_path, "w", encoding="utf-8")

        # tracked commanded state
        self.current_stage_x: Optional[float] = None
        self.current_stage_y: Optional[float] = None
        self.current_stage_z: Optional[float] = None
        self.current_b: Optional[float] = None
        self.current_c: Optional[float] = None
        self.current_tip: Optional[Tuple[float, float, float]] = None

        # extrusion state (absolute U)
        self.u_material_abs = 0.0
        self.pressure_charged = False

        # stats / warnings
        self.total_print_len = 0.0
        self.line_count = 0
        self.b_solve_warnings: List[str] = []

    def close(self):
        self.fh.close()

    def write(self, s: str):
        self.fh.write(s)

    def comment(self, text: str):
        self.write(f"; {text}\n")

    def _fmt_axes(self, axes_vals: Sequence[Tuple[str, float]]) -> str:
        return " ".join(f"{ax}{val:.3f}" for ax, val in axes_vals)

    def g1(self, axes_vals: Sequence[Tuple[str, float]], feed: Optional[float] = None, comment: Optional[str] = None):
        if not axes_vals:
            return
        line = f"G1 {self._fmt_axes(axes_vals)}"
        if feed is not None:
            line += f" F{float(feed):.0f}"
        if comment:
            line += f" ; {comment}"
        self.write(line + "\n")

    def dwell_ms(self, ms: int, comment: Optional[str] = None):
        if ms <= 0:
            return
        line = f"G4 P{int(ms)}"
        if comment:
            line += f" ; {comment}"
        self.write(line + "\n")

    def gcode_header(self, cli_args: argparse.Namespace):
        self.comment("generated by generate_custom_xz_geometry.py")
        self.comment("Custom XZ-plane geometry with optional tangential B-pull based on calibration tip-angle polynomial")
        self.comment("C-only moves use c_feed; XYZ/B moves use motion/print feed")
        self.comment(f"Axes: X={self.cal.x_axis} Y={self.cal.y_axis} Z={self.cal.z_axis} B={self.cal.b_axis} C={self.cal.c_axis} U={self.cal.u_axis}")
        self.comment(f"Calibration B range: [{self.cal.b_min:.3f}, {self.cal.b_max:.3f}]")
        self.comment(f"C(180) command from calibration: {self.cal.c_180_deg:.3f}")
        self.comment(f"Feeds: motion={self.motion_feed:.1f}, print={self.print_feed:.1f}, C-only={self.c_feed:.1f}")
        if self.emit_extrusion:
            self.comment(
                f"Extrusion enabled: extrusion_per_mm={self.extrusion_per_mm:.6f}, pressure_offset={self.pressure_offset_mm:.3f}, "
                f"preflow_dwell={self.preflow_dwell_ms}ms, node_dwell={self.node_dwell_ms}ms"
            )
        else:
            self.comment("Extrusion disabled (set --extrusion-per-mm > 0 to emit U-axis commands)")
        self.comment("NOTE: Step 4 prompt had contradictory C setting; default bottom-half tangential C uses 180 (configurable)")
        self.write("G90\n")
        if self.emit_extrusion:
            self.write("M82\n")
            self.g1([(self.cal.u_axis, 0.0)], comment="U absolute zero (via G1)")
            self.write(f"G92 {self.cal.u_axis}0\n")
            if abs(self.prime_mm) > 0:
                self.u_material_abs += self.prime_mm
                self.g1([(self.cal.u_axis, self._u_cmd_actual())], feed=self.pressure_advance_feed, comment="prime material")

    def _u_cmd_actual(self) -> float:
        return self.u_material_abs + (self.pressure_offset_mm if self.pressure_charged else 0.0)

    def begin_print_line(self, label: str):
        self.comment(f"BEGIN LINE: {label}")
        self.line_count += 1
        if self.emit_extrusion and self.pressure_offset_mm > 0.0 and not self.pressure_charged:
            self.pressure_charged = True
            self.g1([(self.cal.u_axis, self._u_cmd_actual())], feed=self.pressure_advance_feed, comment="pressure preload")
            self.dwell_ms(self.preflow_dwell_ms, comment="preflow dwell")

    def end_print_line(self):
        if self.emit_extrusion and self.pressure_offset_mm > 0.0 and self.pressure_charged:
            self.dwell_ms(self.node_dwell_ms, comment="node dwell")
            self.pressure_charged = False
            self.g1([(self.cal.u_axis, self._u_cmd_actual())], feed=self.pressure_retract_feed, comment="pressure retract")
        self.comment("END LINE")

    def finalize(self):
        if self.emit_extrusion and self.pressure_charged:
            self.pressure_charged = False
            self.g1([(self.cal.u_axis, self._u_cmd_actual())], feed=self.pressure_retract_feed, comment="final pressure retract")
        self.comment(f"total printed length = {self.total_print_len:.3f} mm")
        self.comment(f"printed line count = {self.line_count}")
        if self.b_solve_warnings:
            self.comment(f"B-solve warnings = {len(self.b_solve_warnings)}")
            for w in self.b_solve_warnings:
                self.comment(w)
        self.comment("end of program")

    def set_c_fast(self, c_cmd: float, comment: Optional[str] = None):
        """
        C-only move at c_feed (fast). Use this when you specifically want a faster C rotation.
        Note: this changes tip position unless compensated in XYZ, so use at safe points.
        """
        if self.current_c is not None and almost_equal(self.current_c, c_cmd, 1e-9):
            return
        self.g1([(self.cal.c_axis, float(c_cmd))], feed=self.c_feed, comment=comment or "C-axis move")
        self.current_c = float(c_cmd)
        self.current_tip = None  # no longer exact-tracked after C-only move

    def move_tip_exact(
        self,
        p_tip: Tuple[float, float, float],
        b_cmd: float,
        c_cmd: float,
        comment: Optional[str] = None,
        prefer_fast_c_split: bool = False,
    ):
        """
        Non-print move to a tip point with exact tip tracking (stage = tip - offset(B,C)).
        If prefer_fast_c_split=True and C changes, emit a C-only move first at c_feed, then XYZ/B move.
        """
        p_stage = stage_from_tip(self.cal, p_tip, b_cmd, c_cmd)

        # Optional fast C split
        if prefer_fast_c_split and (self.current_c is None or not almost_equal(self.current_c, c_cmd, 1e-9)):
            self.set_c_fast(c_cmd, comment="fast C rotate before XYZ/B positioning")

        axes = [
            (self.cal.x_axis, p_stage[0]),
            (self.cal.y_axis, p_stage[1]),
            (self.cal.z_axis, p_stage[2]),
            (self.cal.b_axis, float(b_cmd)),
            (self.cal.c_axis, float(c_cmd)),
        ]
        self.g1(axes, feed=self.motion_feed, comment=comment)

        self.current_stage_x, self.current_stage_y, self.current_stage_z = p_stage
        self.current_b = float(b_cmd)
        self.current_c = float(c_cmd)
        self.current_tip = p_tip

    def travel_tip_waypoints(
        self,
        waypoints_tip: Sequence[Tuple[float, float, float]],
        b_cmd: float,
        c_cmd: float,
        comment_prefix: str,
        prefer_fast_c_split_first: bool = False,
    ):
        for i, p in enumerate(waypoints_tip):
            self.move_tip_exact(
                p,
                b_cmd=b_cmd,
                c_cmd=c_cmd,
                comment=f"{comment_prefix} wp{i+1}",
                prefer_fast_c_split=(prefer_fast_c_split_first and i == 0),
            )

    def print_tip_segment(
        self,
        p0_tip: Tuple[float, float, float],
        p1_tip: Tuple[float, float, float],
        b_cmd: float,
        c_cmd: float,
        label: str = "",
    ):
        # Ensure start point / orientation
        if self.current_tip is None or dist3(self.current_tip, p0_tip) > 1e-6 or self.current_b is None or self.current_c is None \
           or (not almost_equal(self.current_b, b_cmd, 1e-9)) or (not almost_equal(self.current_c, c_cmd, 1e-9)):
            self.move_tip_exact(p0_tip, b_cmd, c_cmd, comment=f"position for segment start ({label})")

        p_stage1 = stage_from_tip(self.cal, p1_tip, b_cmd, c_cmd)

        axes = [
            (self.cal.x_axis, p_stage1[0]),
            (self.cal.y_axis, p_stage1[1]),
            (self.cal.z_axis, p_stage1[2]),
            (self.cal.b_axis, float(b_cmd)),
            (self.cal.c_axis, float(c_cmd)),
        ]

        seg_len = dist3(p0_tip, p1_tip)
        if self.emit_extrusion:
            self.u_material_abs += self.extrusion_per_mm * seg_len
            axes.append((self.cal.u_axis, self._u_cmd_actual()))

        self.g1(axes, feed=self.print_feed, comment=label if label else "print segment")

        self.current_stage_x, self.current_stage_y, self.current_stage_z = p_stage1
        self.current_b = float(b_cmd)
        self.current_c = float(c_cmd)
        self.current_tip = p1_tip
        self.total_print_len += seg_len

    def print_line_pieces(self, pieces: Sequence[LinePiece], line_label: str):
        """
        Prints one logical line (can have multiple pieces).
        Pressure preload/retract is applied once around the entire logical line.
        """
        if not pieces:
            return
        self.begin_print_line(line_label)
        for i, piece in enumerate(pieces):
            # tip-fixed pivot if B/C changes between pieces
            if i > 0:
                prev = pieces[i - 1]
                if (not almost_equal(prev.b_cmd, piece.b_cmd, 1e-9)) or (not almost_equal(prev.c_cmd, piece.c_cmd, 1e-9)):
                    self.move_tip_exact(
                        piece.p0_tip,
                        b_cmd=piece.b_cmd,
                        c_cmd=piece.c_cmd,
                        comment=f"tip-fixed pivot to next piece ({piece.label})",
                        prefer_fast_c_split=False,
                    )
            self.print_tip_segment(piece.p0_tip, piece.p1_tip, piece.b_cmd, piece.c_cmd, label=piece.label)
        self.end_print_line()


# ---------------- Mapping helpers (user-specific) ----------------
def fig1_poly_angle_from_slope_deg(slope_deg: float) -> float:
    """
    User-provided mapping for figure 1 tangential case on Y+20 plane:
      +5 slope  -> 5 deg polynomial angle
      -5 slope  -> 95 deg polynomial angle
      -15 slope -> 105 deg polynomial angle
    """
    if slope_deg >= 0:
        return float(slope_deg)
    return 90.0 + abs(float(slope_deg))


def fig3_bottom_poly_angle_from_geom_deg(geom_angle_deg: float) -> float:
    """
    User example for figure 3 bottom-half tangential:
      135 deg geom -> 45 deg polynomial
    => poly = geom - 90  (for geom in [90, 135])
    """
    return float(geom_angle_deg) - 90.0


def fig3_top_poly_angle_from_base_angle_deg(base_angle_deg: float) -> Tuple[float, float]:
    """
    Top-half tangential (starting from top vertex to midline endpoints).
    User examples:
      top direction 225 -> poly 135
      top direction 270 -> poly 180

    For a given base-angle label (90..135), the top travel direction is:
      top_geom = 360 - base_angle
    then:
      poly = top_geom - 90
    Returns (top_geom, poly).
    """
    top_geom = 360.0 - float(base_angle_deg)
    poly = top_geom - 90.0
    return top_geom, poly


# ---------------- Piece builders ----------------
def make_nontangent_piece(
    p0_tip: Tuple[float, float, float],
    p1_tip: Tuple[float, float, float],
    b_non_tangent: float,
    c_cmd: float,
    label: str,
) -> LinePiece:
    return LinePiece(p0_tip=p0_tip, p1_tip=p1_tip, b_cmd=float(b_non_tangent), c_cmd=float(c_cmd), label=label)


def make_tangent_piece_from_poly_angle(
    cal: Calibration,
    p0_tip: Tuple[float, float, float],
    p1_tip: Tuple[float, float, float],
    poly_angle_deg: float,
    c_cmd: float,
    label: str,
    writer: Optional[GCodeWriter] = None,
    b_search_samples: int = DEFAULT_B_SEARCH_SAMPLES,
) -> LinePiece:
    b_cmd, err = solve_b_for_poly_angle_deg(cal, poly_angle_deg, search_samples=b_search_samples)
    if writer is not None and err > 1.0:
        writer.b_solve_warnings.append(
            f"{label}: target poly angle {poly_angle_deg:.3f} deg, solved B={b_cmd:.4f}, abs error={err:.3f} deg"
        )
    return LinePiece(p0_tip=p0_tip, p1_tip=p1_tip, b_cmd=b_cmd, c_cmd=float(c_cmd), label=f"{label} (poly={poly_angle_deg:.1f})")


# ---------------- Main sequence generation ----------------
def generate_sequence(
    writer: GCodeWriter,
    cal: Calibration,
    start_tip: Tuple[float, float, float],
    angle_step: int,
    max_angle: int,
    line_length: float,
    half_x: float,
    return_lift_z: float,
    fig3_x_offset: float,
    fig3_mid_z_rel: float,
    fig3_top_z_rel: float,
    b_non_tangent: float,
    b_search_samples: int,
    fig3_bottom_tangent_c_mode: str,
):
    x0, y0, z0 = start_tip
    y_plane_1 = y0
    y_plane_2 = y0 + 20.0
    y_plane_3 = y0 + 40.0

    c0 = 0.0
    c180 = cal.c_180_deg

    # Resolve step 4 bottom-half tangential C (prompt conflict)
    if fig3_bottom_tangent_c_mode == "0":
        c_fig3_bottom_tan = c0
    elif fig3_bottom_tangent_c_mode == "180":
        c_fig3_bottom_tan = c180
    else:
        raise ValueError("--fig3-bottom-tangent-c-mode must be '0' or '180'")

    writer.comment("=== STEP 1: Figure 1 on first XZ plane (Y=start_y), non-tangential, B=0, C=0 ===")
    start_fig1_plane1 = point(x0, y_plane_1, z0)

    # Move to first start point
    writer.move_tip_exact(start_fig1_plane1, b_non_tangent, c0, comment="move to step1 start", prefer_fast_c_split=True)

    angles = list(range(0, max_angle + 1, angle_step))
    for idx, a in enumerate(angles):
        s = start_fig1_plane1
        pieces: List[LinePiece] = []

        if a == 0:
            p_end = point(x0 + line_length, y_plane_1, z0)
            pieces.append(make_nontangent_piece(s, p_end, b_non_tangent, c0, label="fig1 a=0"))
        else:
            dz_up = half_x * math.tan(math.radians(a))
            p_mid = point(x0 + half_x, y_plane_1, z0 + dz_up)
            p_end = point(x0 + line_length, y_plane_1, z0)  # symmetric +a then -a returns to base z
            pieces.append(make_nontangent_piece(s, p_mid, b_non_tangent, c0, label=f"fig1 +{a}"))
            pieces.append(make_nontangent_piece(p_mid, p_end, b_non_tangent, c0, label=f"fig1 -{a}"))

        writer.print_line_pieces(pieces, line_label=f"STEP1 Figure1 angle={a} deg")

        # Return after each line: Z up 50, X back, Z down
        # Use B/C for next line start (same here: B=0, C=0)
        cur_end = pieces[-1].p1_tip
        wp1 = point(cur_end[0], cur_end[1], cur_end[2] + return_lift_z)
        wp2 = point(s[0], s[1], wp1[2])
        wp3 = s
        writer.travel_tip_waypoints([wp1, wp2, wp3], b_non_tangent, c0, comment_prefix=f"STEP1 return angle={a}", prefer_fast_c_split_first=False)

    writer.comment("=== STEP 2: Figure 1 on second XZ plane (Y=start_y+20), tangential, C=180 ===")
    start_fig1_plane2 = point(x0, y_plane_2, z0)

    # Move to plane Y+20; choose B/C for first line (a=0 first segment)
    first_poly = fig1_poly_angle_from_slope_deg(0.0)
    first_b, first_err = solve_b_for_poly_angle_deg(cal, first_poly, search_samples=b_search_samples)
    if first_err > 1.0:
        writer.b_solve_warnings.append(f"STEP2 first line a=0 target poly={first_poly:.1f} err={first_err:.3f} deg")
    writer.move_tip_exact(start_fig1_plane2, first_b, c180, comment="move to step2 start (Y+20 plane)", prefer_fast_c_split=True)

    for idx, a in enumerate(angles):
        s = start_fig1_plane2
        pieces = []

        if a == 0:
            p_end = point(x0 + line_length, y_plane_2, z0)
            pieces.append(
                make_tangent_piece_from_poly_angle(
                    cal, s, p_end,
                    poly_angle_deg=fig1_poly_angle_from_slope_deg(0.0),
                    c_cmd=c180,
                    label="fig1_tan a=0",
                    writer=writer,
                    b_search_samples=b_search_samples,
                )
            )
        else:
            dz_up = half_x * math.tan(math.radians(a))
            p_mid = point(x0 + half_x, y_plane_2, z0 + dz_up)
            p_end = point(x0 + line_length, y_plane_2, z0)

            pieces.append(
                make_tangent_piece_from_poly_angle(
                    cal, s, p_mid,
                    poly_angle_deg=fig1_poly_angle_from_slope_deg(+a),
                    c_cmd=c180,
                    label=f"fig1_tan +{a}",
                    writer=writer,
                    b_search_samples=b_search_samples,
                )
            )
            pieces.append(
                make_tangent_piece_from_poly_angle(
                    cal, p_mid, p_end,
                    poly_angle_deg=fig1_poly_angle_from_slope_deg(-a),
                    c_cmd=c180,
                    label=f"fig1_tan -{a}",
                    writer=writer,
                    b_search_samples=b_search_samples,
                )
            )

        # Ensure we are at start with first piece orientation
        writer.move_tip_exact(s, pieces[0].b_cmd, pieces[0].c_cmd, comment=f"STEP2 line {a} start position")
        writer.print_line_pieces(pieces, line_label=f"STEP2 Figure1 tangential angle={a} deg")

        # Return after each line: Z up 50, X back, Z down
        cur_end = pieces[-1].p1_tip
        wp1 = point(cur_end[0], cur_end[1], cur_end[2] + return_lift_z)
        wp2 = point(s[0], s[1], wp1[2])
        wp3 = s

        # If there's a next line, preposition using next line's first-piece B/C; otherwise use current first piece B/C
        if idx + 1 < len(angles):
            a_next = angles[idx + 1]
            poly_next = fig1_poly_angle_from_slope_deg(0.0 if a_next == 0 else a_next)
            b_next, err_next = solve_b_for_poly_angle_deg(cal, poly_next, search_samples=b_search_samples)
            if err_next > 1.0:
                writer.b_solve_warnings.append(f"STEP2 next-line preposition a={a_next} target poly={poly_next:.1f} err={err_next:.3f} deg")
            b_ret = b_next
            c_ret = c180
        else:
            b_ret = pieces[0].b_cmd
            c_ret = pieces[0].c_cmd

        writer.travel_tip_waypoints([wp1, wp2, wp3], b_ret, c_ret, comment_prefix=f"STEP2 return angle={a}", prefer_fast_c_split_first=False)

    writer.comment("=== STEP 3: Figure 3 on first XZ plane (Y=start_y), non-tangential ===")
    fig3_origin_plane1 = point(x0 + fig3_x_offset, y_plane_1, z0)
    fig3_top_plane1 = point(x0 + fig3_x_offset, y_plane_1, z0 + fig3_top_z_rel)

    # Bottom half: C=180, B=0
    writer.move_tip_exact(fig3_origin_plane1, b_non_tangent, c180, comment="STEP3 bottom-half origin (Y=start_y)", prefer_fast_c_split=True)

    bottom_angles = list(range(135, 89, -5))  # 135..90
    bottom_endpoints_plane1: Dict[int, Tuple[float, float, float]] = {}

    for theta in bottom_angles:
        if theta == 90:
            p_end = fig3_top_plane1
        else:
            p_end = endpoint_from_origin_angle_to_zrel(fig3_origin_plane1, theta, fig3_mid_z_rel)
        bottom_endpoints_plane1[theta] = p_end

        piece = make_nontangent_piece(fig3_origin_plane1, p_end, b_non_tangent, c180, label=f"fig3 bottom {theta}deg")
        writer.print_line_pieces([piece], line_label=f"STEP3 bottom-half geom={theta} deg")

        # Return to origin between bottom-half lines except after the 90 deg vertical (we continue to top-half setup)
        if theta != 90:
            # "When going back down, move back in x before going back down in z."
            wp_x = point(fig3_origin_plane1[0], fig3_origin_plane1[1], p_end[2])  # same z, x back to origin
            wp_z = fig3_origin_plane1
            writer.travel_tip_waypoints([wp_x, wp_z], b_non_tangent, c180, comment_prefix=f"STEP3 bottom return {theta}", prefer_fast_c_split_first=False)

    # Top half: switch to C=0, draw lines from endpoints of 95..135 to top vertex
    top_start_base_angles = list(range(95, 136, 5))  # endpoints already generated by bottom-half lines
    writer.comment("STEP3 top-half (non-tangential): move to end of 95deg line, then draw 85..45deg lines to top vertex; C=0")

    # We are currently at top vertex (end of 90deg line)
    # Reposition to end of 95deg line: X first then Z down (collision-avoidance instruction)
    first_start = bottom_endpoints_plane1[95]
    wp_x = point(first_start[0], fig3_top_plane1[1], fig3_top_plane1[2])  # top z, x first
    wp_z = first_start  # then down to z=mid
    writer.travel_tip_waypoints([wp_x, wp_z], b_non_tangent, c0, comment_prefix="STEP3 top reposition to 95 endpoint", prefer_fast_c_split_first=True)

    for i, base_theta in enumerate(top_start_base_angles):
        p_start = bottom_endpoints_plane1[base_theta]
        p_end = fig3_top_plane1
        line_angle = 180 - base_theta  # 95->85, 135->45 (as described)

        # Ensure we're at current startpoint
        if writer.current_tip is None or dist3(writer.current_tip, p_start) > 1e-6:
            # For subsequent lines, move from top vertex to next startpoint: X first then Z down
            wp_x = point(p_start[0], fig3_top_plane1[1], fig3_top_plane1[2])
            wp_z = p_start
            writer.travel_tip_waypoints([wp_x, wp_z], b_non_tangent, c0, comment_prefix=f"STEP3 top reposition to {base_theta} endpoint", prefer_fast_c_split_first=False)

        piece = make_nontangent_piece(p_start, p_end, b_non_tangent, c0, label=f"fig3 top {line_angle}deg (from {base_theta} endpoint)")
        writer.print_line_pieces([piece], line_label=f"STEP3 top-half line={line_angle} deg")

        # Next iteration starts from top vertex and must move to next endpoint with X first then Z down.
        # We do not add an extra return here because line already ends at top vertex.

    writer.comment("=== STEP 4: Figure 3 bottom half on second XZ plane (Y=start_y+20), tangential ===")
    fig3_origin_plane2 = point(x0 + fig3_x_offset, y_plane_2, z0)
    fig3_top_plane2 = point(x0 + fig3_x_offset, y_plane_2, z0 + fig3_top_z_rel)

    # Move to origin; C mode configurable due to prompt contradiction (default 180)
    # Use first bottom angle 135 => poly 45 as example
    first_poly_bottom = fig3_bottom_poly_angle_from_geom_deg(135.0)
    first_b_bottom, first_err_bottom = solve_b_for_poly_angle_deg(cal, first_poly_bottom, search_samples=b_search_samples)
    if first_err_bottom > 1.0:
        writer.b_solve_warnings.append(f"STEP4 first bottom line target poly={first_poly_bottom:.1f} err={first_err_bottom:.3f} deg")
    writer.move_tip_exact(fig3_origin_plane2, first_b_bottom, c_fig3_bottom_tan, comment="STEP4 bottom-half origin (Y+20)", prefer_fast_c_split=True)

    for theta in bottom_angles:
        if theta == 90:
            p_end = fig3_top_plane2
        else:
            p_end = endpoint_from_origin_angle_to_zrel(fig3_origin_plane2, theta, fig3_mid_z_rel)

        poly_angle = fig3_bottom_poly_angle_from_geom_deg(theta)
        piece = make_tangent_piece_from_poly_angle(
            cal,
            fig3_origin_plane2,
            p_end,
            poly_angle_deg=poly_angle,
            c_cmd=c_fig3_bottom_tan,
            label=f"fig3 bottom tan {theta}deg",
            writer=writer,
            b_search_samples=b_search_samples,
        )

        writer.move_tip_exact(fig3_origin_plane2, piece.b_cmd, piece.c_cmd, comment=f"STEP4 line {theta} start position")
        writer.print_line_pieces([piece], line_label=f"STEP4 bottom-half tangential geom={theta} deg")

        # Return to origin every time; "move x positively first then decrease z"
        # (endpoint x is <= origin x for these angles)
        wp_x = point(fig3_origin_plane2[0], fig3_origin_plane2[1], p_end[2])
        wp_z = fig3_origin_plane2

        # Preposition using next line's B if possible
        if theta > 90:
            next_theta = theta - 5
            next_poly = fig3_bottom_poly_angle_from_geom_deg(next_theta)
            b_next, err_next = solve_b_for_poly_angle_deg(cal, next_poly, search_samples=b_search_samples)
            if err_next > 1.0:
                writer.b_solve_warnings.append(f"STEP4 next-line preposition theta={next_theta} target poly={next_poly:.1f} err={err_next:.3f} deg")
            b_ret = b_next
        else:
            b_ret = piece.b_cmd

        writer.travel_tip_waypoints([wp_x, wp_z], b_ret, c_fig3_bottom_tan, comment_prefix=f"STEP4 bottom return {theta}", prefer_fast_c_split_first=False)

    writer.comment("=== STEP 5: Figure 3 top half on third XZ plane (Y=start_y+40), tangential ===")
    fig3_origin_plane3 = point(x0 + fig3_x_offset, y_plane_3, z0)
    fig3_top_plane3 = point(x0 + fig3_x_offset, y_plane_3, z0 + fig3_top_z_rel)

    # Start each line from top common vertex; draw 90-degree line first (base-angle label 90)
    top_base_angles = list(range(90, 136, 5))  # 90,95,...,135

    # First line: base 90 => top_geom 270 => poly 180 (user example)
    _, first_top_poly = fig3_top_poly_angle_from_base_angle_deg(90.0)
    first_b_top, first_err_top = solve_b_for_poly_angle_deg(cal, first_top_poly, search_samples=b_search_samples)
    if first_err_top > 1.0:
        writer.b_solve_warnings.append(f"STEP5 first top line target poly={first_top_poly:.1f} err={first_err_top:.3f} deg")
    writer.move_tip_exact(fig3_top_plane3, first_b_top, c0, comment="STEP5 top-half top vertex (Y+40)", prefer_fast_c_split=True)

    # Build endpoint cache on plane 3
    bottom_endpoints_plane3: Dict[int, Tuple[float, float, float]] = {}
    for theta in top_base_angles:
        if theta == 90:
            p_mid = point(fig3_origin_plane3[0], fig3_origin_plane3[1], fig3_origin_plane3[2] + fig3_mid_z_rel)
        else:
            p_mid = endpoint_from_origin_angle_to_zrel(fig3_origin_plane3, theta, fig3_mid_z_rel)
        bottom_endpoints_plane3[theta] = p_mid

    for i, base_theta in enumerate(top_base_angles):
        p_start = fig3_top_plane3
        p_end = bottom_endpoints_plane3[base_theta]
        top_geom, poly_angle = fig3_top_poly_angle_from_base_angle_deg(base_theta)

        piece = make_tangent_piece_from_poly_angle(
            cal,
            p_start,
            p_end,
            poly_angle_deg=poly_angle,
            c_cmd=c0,
            label=f"fig3 top tan base={base_theta} topgeom={top_geom}",
            writer=writer,
            b_search_samples=b_search_samples,
        )

        # Ensure at top vertex with this B/C
        writer.move_tip_exact(fig3_top_plane3, piece.b_cmd, piece.c_cmd, comment=f"STEP5 line base={base_theta} start position")
        writer.print_line_pieces([piece], line_label=f"STEP5 top-half tangential base={base_theta} deg (top geom={top_geom:.0f})")

        # Return to top vertex between lines:
        # "move up and then move x back to the next print point"
        # (reverse of top->endpoint start travel)
        if i < len(top_base_angles) - 1:
            cur = p_end
            wp_up = point(cur[0], cur[1], fig3_top_plane3[2])     # same x, move up in z
            wp_backx = fig3_top_plane3                            # then x back to top vertex

            next_base = top_base_angles[i + 1]
            _, next_poly = fig3_top_poly_angle_from_base_angle_deg(next_base)
            b_next, err_next = solve_b_for_poly_angle_deg(cal, next_poly, search_samples=b_search_samples)
            if err_next > 1.0:
                writer.b_solve_warnings.append(f"STEP5 next-line preposition base={next_base} target poly={next_poly:.1f} err={err_next:.3f} deg")

            writer.travel_tip_waypoints([wp_up, wp_backx], b_next, c0, comment_prefix=f"STEP5 top return base={base_theta}", prefer_fast_c_split_first=False)


# ---------------- CLI / diagnostics ----------------
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description="Generate G-code for the requested custom XZ-plane line geometry using calibration-based B pull tangential solves."
    )
    ap.add_argument("--calibration", required=True, help="Path to calibration JSON.")
    ap.add_argument("--out", default=DEFAULT_OUT, help="Output G-code path.")

    # Start/design origin
    ap.add_argument("--start-x", type=float, default=DEFAULT_START_X, help="Starting TIP-space X for figure 1.")
    ap.add_argument("--start-y", type=float, default=DEFAULT_START_Y, help="Starting TIP-space Y (first XZ plane).")
    ap.add_argument("--start-z", type=float, default=DEFAULT_START_Z, help="Starting TIP-space Z for figure 1.")

    # Geometry
    ap.add_argument("--line-length", type=float, default=DEFAULT_LINE_LENGTH, help="Figure 1 total line length (default 80 mm).")
    ap.add_argument("--half-x", type=float, default=DEFAULT_HALF_X, help="Figure 1 midpoint x_rel and half length (default 40 mm).")
    ap.add_argument("--angle-step", type=int, default=DEFAULT_ANGLE_STEP, help="Angle increment in degrees (default 5).")
    ap.add_argument("--max-angle", type=int, default=DEFAULT_MAX_ANGLE, help="Maximum angle for figure 1 roof lines (default 45).")
    ap.add_argument("--return-lift-z", type=float, default=DEFAULT_RETURN_LIFT_Z, help="Lift amount for figure 1 returns (default 50 mm).")

    ap.add_argument("--fig3-x-offset", type=float, default=DEFAULT_FIG3_X_OFFSET, help="Figure 3 origin x offset from start point (default 100 mm).")
    ap.add_argument("--fig3-mid-z-rel", type=float, default=DEFAULT_FIG3_MID_Z_REL, help="Figure 3 midline z_rel (default 40 mm).")
    ap.add_argument("--fig3-top-z-rel", type=float, default=DEFAULT_FIG3_TOP_Z_REL, help="Figure 3 top vertex z_rel (default 80 mm).")

    # Motion
    ap.add_argument("--motion-feed", type=float, default=DEFAULT_MOTION_FEED, help="XYZ/B non-print move feed (default 300).")
    ap.add_argument("--print-feed", type=float, default=DEFAULT_PRINT_FEED, help="XYZ/B/U print feed (default 300).")
    ap.add_argument("--c-feed", type=float, default=DEFAULT_C_FEED, help="C-axis only move feed (default 5000).")

    # Extrusion
    ap.add_argument("--extrusion-per-mm", type=float, default=DEFAULT_EXTRUSION_PER_MM, help="U mm per mm of printed path; set >0 to enable extrusion.")
    ap.add_argument("--prime-mm", type=float, default=DEFAULT_PRIME_MM, help="Initial U prime (absolute mode).")
    ap.add_argument("--pressure-offset-mm", type=float, default=DEFAULT_PRESSURE_OFFSET_MM, help="Preload/retract U offset before/after each line.")
    ap.add_argument("--pressure-advance-feed", type=float, default=DEFAULT_PRESSURE_ADVANCE_FEED, help="U-only preload feed.")
    ap.add_argument("--pressure-retract-feed", type=float, default=DEFAULT_PRESSURE_RETRACT_FEED, help="U-only retract feed.")
    ap.add_argument("--preflow-dwell-ms", type=int, default=DEFAULT_PREFLOW_DWELL_MS, help="Dwell after preload before print line.")
    ap.add_argument("--node-dwell-ms", type=int, default=DEFAULT_NODE_DWELL_MS, help="Dwell at line end before pressure retract.")

    # B solve
    ap.add_argument("--b-search-samples", type=int, default=DEFAULT_B_SEARCH_SAMPLES, help="Samples for B solve by tip-angle polynomial.")
    ap.add_argument("--b-non-tangent", type=float, default=DEFAULT_B_NON_TANGENT, help="Fixed B command for non-tangential passes (default 0).")

    # Prompt contradiction handling
    ap.add_argument(
        "--fig3-bottom-tangent-c-mode",
        choices=["0", "180"],
        default=DEFAULT_FIG3_BOTTOM_TANGENT_C_MODE,
        help="Step 4 bottom-half tangential C mode (prompt is contradictory). Default 180.",
    )

    # Diagnostics only
    ap.add_argument("--syringe-mm-per-ml", type=float, default=DEFAULT_SYRINGE_MM_PER_ML, help="For extrusion diagnostics only.")
    ap.add_argument("--tube-id-inch", type=float, default=DEFAULT_TUBE_ID_INCH, help="For extrusion diagnostics only.")

    return ap


def main():
    ap = build_argparser()
    args = ap.parse_args()

    cal = load_calibration(args.calibration)

    writer = GCodeWriter(
        out_path=args.out,
        cal=cal,
        motion_feed=args.motion_feed,
        print_feed=args.print_feed,
        c_feed=args.c_feed,
        extrusion_per_mm=args.extrusion_per_mm,
        prime_mm=args.prime_mm,
        pressure_offset_mm=args.pressure_offset_mm,
        pressure_advance_feed=args.pressure_advance_feed,
        pressure_retract_feed=args.pressure_retract_feed,
        preflow_dwell_ms=args.preflow_dwell_ms,
        node_dwell_ms=args.node_dwell_ms,
    )

    start_tip = point(args.start_x, args.start_y, args.start_z)

    try:
        writer.gcode_header(args)

        generate_sequence(
            writer=writer,
            cal=cal,
            start_tip=start_tip,
            angle_step=args.angle_step,
            max_angle=args.max_angle,
            line_length=args.line_length,
            half_x=args.half_x,
            return_lift_z=args.return_lift_z,
            fig3_x_offset=args.fig3_x_offset,
            fig3_mid_z_rel=args.fig3_mid_z_rel,
            fig3_top_z_rel=args.fig3_top_z_rel,
            b_non_tangent=args.b_non_tangent,
            b_search_samples=args.b_search_samples,
            fig3_bottom_tangent_c_mode=args.fig3_bottom_tangent_c_mode,
        )

        writer.finalize()
    finally:
        writer.close()

    # Console diagnostics
    print(f"Wrote: {args.out}")
    print(f"Printed line count: {writer.line_count}")
    print(f"Total printed path length: {writer.total_print_len:.3f} mm")
    print(f"Extrusion enabled: {writer.emit_extrusion}")
    print(f"Feeds: motion/print={args.motion_feed}/{args.print_feed} mm/min, C-only={args.c_feed} mm/min")

    if writer.emit_extrusion:
        path_speed_mm_s = float(args.print_feed) / 60.0
        u_speed_mm_s = float(args.extrusion_per_mm) * path_speed_mm_s
        q_mm3_s = (1000.0 / float(args.syringe_mm_per_ml)) * u_speed_mm_s if args.syringe_mm_per_ml > 0 else float("nan")
        tube_area = tube_area_mm2_from_id_inch(float(args.tube_id_inch))
        tube_v_mm_s = q_mm3_s / tube_area if tube_area > 0 else float("nan")
        pressure_offset_ml = float(args.pressure_offset_mm) / float(args.syringe_mm_per_ml) if args.syringe_mm_per_ml > 0 else float("nan")
        pressure_offset_ul = pressure_offset_ml * 1000.0 if math.isfinite(pressure_offset_ml) else float("nan")

        print("Extrusion diagnostics:")
        print(f"  extrusion_per_mm = {args.extrusion_per_mm:.6f} U mm / mm path")
        print(f"  path speed = {path_speed_mm_s:.6f} mm/s")
        print(f"  U speed during print = {u_speed_mm_s:.6f} mm/s")
        print(f"  volumetric flow = {q_mm3_s:.6f} mm^3/s (uL/s)")
        print(f"  tube area = {tube_area:.6f} mm^2  -> tube velocity = {tube_v_mm_s:.6f} mm/s")
        print(f"  pressure offset = {args.pressure_offset_mm:.3f} U mm ~= {pressure_offset_ul:.3f} uL")
        print(f"  preflow/node dwells = {args.preflow_dwell_ms}/{args.node_dwell_ms} ms")

    if writer.b_solve_warnings:
        print(f"B-solve warnings ({len(writer.b_solve_warnings)}):")
        for w in writer.b_solve_warnings[:20]:
            print("  -", w)
        if len(writer.b_solve_warnings) > 20:
            print(f"  ... and {len(writer.b_solve_warnings)-20} more")


if __name__ == "__main__":
    main()