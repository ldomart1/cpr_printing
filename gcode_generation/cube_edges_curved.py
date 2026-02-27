#!/usr/bin/env python3
"""
Duet/RRF G-code generator: cube edges + vertical side panels using a bending/rotating tube.

Axes:
  - Motion: X, Y, Z (these are the *pivot* point coordinates)
  - Extrusion: U (relative)
  - Tube bend pull: B
  - Tube rotation: C (turns; 0.5 = 180deg, 1.0 = 360deg)

Tool model at 90deg bend:
  - When B == B90 (default -1.2), nozzle tip offset from pivot:
      +offset_z_mm in Z
      +offset_xy_mm in XY along direction set by C:
          theta = 2*pi*C
          dir = (cos(theta), sin(theta))
          offset = (offset_xy_mm*dir.x, offset_xy_mm*dir.y, +offset_z_mm)

We generate desired paths in *tip space* and convert to pivot XYZ with:
  pivot = tip - offset

Output order:
  1) bottom horizontal square
  2) 4 vertical side panel perimeters (one per cube face)
  3) top horizontal square
  4) retract + dwell + home
"""

import argparse
import math
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Params:
    edge: float
    cx: float
    cy: float
    cz: float

    f_print: float          # mm/min (motion feed during extrusion)
    f_travel: float         # mm/min

    u_per_mm: float         # U units per mm of toolpath length
    u_retract: float        # retract amount (positive; applied as -u_retract)
    dwell_s: float
    z_hop: float

    # Tube model
    b_straight: float       # typically 0.0
    b_90: float             # pull corresponding to 90deg bend (default -1.2)
    offset_xy_mm: float     # in-plane offset at 90deg
    offset_z_mm: float      # +Z offset at 90deg

    # Program
    home_at_end: bool


def gcode_header(p: Params) -> List[str]:
    return [
        "; --- Cube printing (Duet/RRF) with tube B/C compensation ---",
        f"; edge={p.edge}  center=({p.cx},{p.cy},{p.cz})",
        f"; F_print={p.f_print} mm/min  F_travel={p.f_travel} mm/min",
        f"; U_per_mm={p.u_per_mm}  retract={p.u_retract}  dwell={p.dwell_s}s",
        f"; Tube: B90={p.b_90}  offset_xy_mm={p.offset_xy_mm}  offset_z_mm={p.offset_z_mm}  C in turns (0.5=180deg)",
        "; Assumes printer is already homed before running this file.",
        "",
        "G90                         ; absolute positioning (XYZBC)",
        "M83                         ; relative extrusion (U)",
        "G92 U0                      ; reset U",
        "M400                        ; wait",
        "",
    ]


def stop_flow(p: Params) -> List[str]:
    return [
        f"G1 U{-p.u_retract:.3f} F300       ; retract U to stop flow",
        f"G4 S{p.dwell_s:.0f}                ; dwell to let material stop",
    ]


def set_tool_state(b: float, c_turns: float) -> List[str]:
    # You may prefer to separate these; this keeps it simple.
    return [
        f"G1 B{b:.4f} C{c_turns:.5f} F2000    ; set tube bend/rotation",
    ]


def tool_offset_xyz(p: Params, b: float, c_turns: float) -> Tuple[float, float, float]:
    """
    Returns (ox, oy, oz) such that: tip = pivot + offset
    We only model the known 90deg case; otherwise assume straight/no offset.
    """
    # Only apply the offset at the 90deg bend setting (within a small tolerance)
    if abs(b - p.b_90) > 1e-6:
        return (0.0, 0.0, 0.0)

    theta = 2.0 * math.pi * c_turns
    dx = math.cos(theta)
    dy = math.sin(theta)
    return (p.offset_xy_mm * dx, p.offset_xy_mm * dy, p.offset_z_mm)


def pivot_from_tip(p: Params, tip: Tuple[float, float, float], b: float, c_turns: float) -> Tuple[float, float, float]:
    ox, oy, oz = tool_offset_xyz(p, b, c_turns)
    return (tip[0] - ox, tip[1] - oy, tip[2] - oz)


def fmt_xyz(x: float, y: float, z: float) -> str:
    return f"X{x:.3f} Y{y:.3f} Z{z:.3f}"


def travel_to_pivot(p: Params, pivot_xyz: Tuple[float, float, float], comment: str = "") -> str:
    cmt = f" ; {comment}" if comment else ""
    x, y, z = pivot_xyz
    return f"G1 {fmt_xyz(x, y, z)} F{p.f_travel:.0f}{cmt}"


def extrude_to_pivot(p: Params, pivot_xyz: Tuple[float, float, float], u_amt: float, comment: str = "") -> str:
    cmt = f" ; {comment}" if comment else ""
    x, y, z = pivot_xyz
    return f"G1 {fmt_xyz(x, y, z)} U{u_amt:.5f} F{p.f_print:.0f}{cmt}"


def path_length(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> float:
    dx = b[0] - a[0]
    dy = b[1] - a[1]
    dz = b[2] - a[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def square_corners_xy(edge: float, cx: float, cy: float, z: float) -> List[Tuple[float, float, float]]:
    half = edge / 2.0
    return [
        (cx - half, cy - half, z),
        (cx + half, cy - half, z),
        (cx + half, cy + half, z),
        (cx - half, cy + half, z),
    ]


def vertical_face_loop(edge: float, cx: float, cy: float, cz: float, face: str) -> List[Tuple[float, float, float]]:
    """
    Returns a closed loop (without repeating first point) of 4 corners for a vertical face panel perimeter
    in TIP coordinates.

    face in {"X+", "X-", "Y+", "Y-"}.
    """
    half = edge / 2.0
    z0 = cz - half
    z1 = cz + half

    if face == "X+":
        x = cx + half
        return [(x, cy - half, z0), (x, cy + half, z0), (x, cy + half, z1), (x, cy - half, z1)]
    if face == "X-":
        x = cx - half
        return [(x, cy - half, z0), (x, cy + half, z0), (x, cy + half, z1), (x, cy - half, z1)]
    if face == "Y+":
        y = cy + half
        return [(cx - half, y, z0), (cx + half, y, z0), (cx + half, y, z1), (cx - half, y, z1)]
    if face == "Y-":
        y = cy - half
        return [(cx - half, y, z0), (cx + half, y, z0), (cx + half, y, z1), (cx - half, y, z1)]

    raise ValueError(f"Unknown face: {face}")


def c_for_face(face: str, nozzle_rot_ccw: bool = True) -> float:
    """
    We model:
      offset_dir angle = 2*pi*C
      nozzle_dir angle = offset_dir ± 90deg

    If nozzle_rot_ccw=True:
      nozzle_dir = offset_dir + 90deg (CCW)
      => offset_dir = nozzle_dir - 90deg

    If nozzle_rot_ccw=False (CW):
      nozzle_dir = offset_dir - 90deg
      => offset_dir = nozzle_dir + 90deg
    """
    # desired nozzle direction angles (deg) for outward normals:
    nozzle_deg = {"X+": 0.0, "Y+": 90.0, "X-": 180.0, "Y-": 270.0}[face]

    if nozzle_rot_ccw:
        offset_deg = (nozzle_deg - 90.0) % 360.0
    else:
        offset_deg = (nozzle_deg + 90.0) % 360.0

    return offset_deg / 360.0  # turns


def generate(p: Params) -> List[str]:
    g: List[str] = []
    g += gcode_header(p)

    half = p.edge / 2.0
    z0 = p.cz - half
    z1 = p.cz + half

    # ---------- 1) Bottom square (straight tool) ----------
    g += ["; --- Bottom horizontal square (tip path in XY @ Z=z0) ---"]
    b = p.b_straight
    c = 0.0
    g += set_tool_state(b, c)

    bottom = square_corners_xy(p.edge, p.cx, p.cy, z0)
    start_tip = bottom[0]

    g += stop_flow(p)
    # travel above start (optional z-hop in pivot space == tip space here because offset is 0)
    g += [travel_to_pivot(p, (start_tip[0], start_tip[1], start_tip[2] + p.z_hop), "travel above bottom start")]
    if p.z_hop > 0:
        g += [travel_to_pivot(p, (start_tip[0], start_tip[1], start_tip[2]), "drop to bottom Z")]

    prev_tip = start_tip
    for i in range(1, 4):
        tip = bottom[i]
        u = path_length(prev_tip, tip) * p.u_per_mm
        g += [extrude_to_pivot(p, tip, u, f"bottom edge {i}")]
        prev_tip = tip
    # close
    u = path_length(prev_tip, start_tip) * p.u_per_mm
    g += [extrude_to_pivot(p, start_tip, u, "bottom edge close")]

    # ---------- 2) Side panels (90deg bend + C rotation) ----------
    g += ["", "; --- Side panels (each face perimeter), printed with 90deg angle of attack ---"]
    b = p.b_90

    for face in ["X+", "Y+", "X-", "Y-"]:
        c = c_for_face(face, nozzle_rot_ccw=True)
        loop = vertical_face_loop(p.edge, p.cx, p.cy, p.cz, face)

        g += ["", f"; Face {face}  (B={b}, C={c} turns)"]
        g += stop_flow(p)
        g += set_tool_state(b, c)

        # First point (tip), convert to pivot
        start_tip = loop[0]
        start_pivot = pivot_from_tip(p, start_tip, b, c)

        # travel above in pivot coordinates (z-hop acts on pivot Z)
        g += [travel_to_pivot(p, (start_pivot[0], start_pivot[1], start_pivot[2] + p.z_hop), f"travel above {face} start")]
        if p.z_hop > 0:
            g += [travel_to_pivot(p, start_pivot, f"drop to {face} start")]

        prev_tip = start_tip
        prev_pivot = start_pivot

        # trace 3 edges
        for i in range(1, 4):
            tip = loop[i]
            pivot = pivot_from_tip(p, tip, b, c)
            u = path_length(prev_tip, tip) * p.u_per_mm
            g += [extrude_to_pivot(p, pivot, u, f"{face} edge {i}")]
            prev_tip, prev_pivot = tip, pivot

        # close loop
        u = path_length(prev_tip, start_tip) * p.u_per_mm
        g += [extrude_to_pivot(p, start_pivot, u, f"{face} edge close")]

    # ---------- 3) Top square (straight tool) ----------
    g += ["", "; --- Top horizontal square (straight tool) ---"]
    b = p.b_straight
    c = 0.0
    g += stop_flow(p)
    g += set_tool_state(b, c)

    top = square_corners_xy(p.edge, p.cx, p.cy, z1)
    start_tip = top[0]

    g += [travel_to_pivot(p, (start_tip[0], start_tip[1], start_tip[2] + p.z_hop), "travel above top start")]
    if p.z_hop > 0:
        g += [travel_to_pivot(p, (start_tip[0], start_tip[1], start_tip[2]), "drop to top Z")]

    prev_tip = start_tip
    for i in range(1, 4):
        tip = top[i]
        u = path_length(prev_tip, tip) * p.u_per_mm
        g += [extrude_to_pivot(p, tip, u, f"top edge {i}")]
        prev_tip = tip
    u = path_length(prev_tip, start_tip) * p.u_per_mm
    g += [extrude_to_pivot(p, start_tip, u, "top edge close")]

    # ---------- Finish ----------
    g += ["", "; --- Finish / retract / home ---"]
    g += stop_flow(p)
    g += set_tool_state(p.b_straight, 0.0)
    g += [f"G1 Z{(z1 + max(2.0, p.z_hop)):.3f} F{p.f_travel:.0f} ; lift before home"]
    if p.home_at_end:
        g += ["G28                         ; home all axes"]
    g += ["M400                        ; wait",
          "; --- End ---"]
    return g


def compute_u_per_mm(
    syringe_diam_mm: float,
    nozzle_diam_mm: float,
    flow_mult: float = 1.0,
    bead_width_mm: float | None = None,
    bead_height_mm: float | None = None,
) -> float:
    """
    Compute U-per-mm based on volumetric continuity:
      volume_per_length (mm^3/mm) = bead_area
      u_per_mm = bead_area / syringe_area

    By default, bead_area uses the nozzle orifice area.
    """
    syringe_area = math.pi * (syringe_diam_mm * 0.5) ** 2
    if bead_width_mm is None and bead_height_mm is None:
        bead_area = math.pi * (nozzle_diam_mm * 0.5) ** 2
    else:
        w = bead_width_mm if bead_width_mm is not None else nozzle_diam_mm
        h = bead_height_mm if bead_height_mm is not None else nozzle_diam_mm
        bead_area = w * h
    return flow_mult * (bead_area / syringe_area)


def main():
    ap = argparse.ArgumentParser(description="Generate Duet G-code for cube bottom/top + perpendicular side panels using B/C tube model.")
    ap.add_argument("--edge", type=float, default=15.0, help="Cube edge size (mm)")
    ap.add_argument("--center", type=float, nargs=3, default=[25.0, 25.0, -25.0], metavar=("CX", "CY", "CZ"),
                    help="Cube center (mm)")
    ap.add_argument("--fprint", type=float, default=300.0, help="Print feedrate (mm/min)")
    ap.add_argument("--ftravel", type=float, default=600.0, help="Travel feedrate (mm/min)")
    ap.add_argument("--upermm", type=float, default=None, help="Extrusion U per mm of path (omit to auto-compute)")
    ap.add_argument("--syringe-diam", type=float, default=14.0, help="Syringe inner diameter (mm)")
    ap.add_argument("--nozzle-diam-in", type=float, default=0.02, help="Nozzle orifice diameter (inches)")
    ap.add_argument("--bead-width", type=float, default=None, help="Bead width (mm). Default: nozzle diameter")
    ap.add_argument("--bead-height", type=float, default=None, help="Bead height (mm). Default: nozzle diameter")
    ap.add_argument("--flow-mult", type=float, default=10.0, help="Flow multiplier")
    ap.add_argument("--retract", type=float, default=0.100, help="Retract U amount")
    ap.add_argument("--dwell", type=float, default=5.0, help="Dwell seconds after retract")
    ap.add_argument("--zhop", type=float, default=0.0, help="Z hop (mm) for travels")

    ap.add_argument("--bstraight", type=float, default=0.0, help="B value for straight tube")
    ap.add_argument("--b90", type=float, default=-4.0, help="B value for 90deg bend")
    ap.add_argument("--offset", type=float, default=None, help="Legacy: offset mm at 90deg bend (applied in +Z and in-plane)")
    ap.add_argument("--offset-xy", type=float, default=25.0, help="In-plane offset mm at 90deg bend")
    ap.add_argument("--offset-z", type=float, default=18.0, help="Z+ offset mm at 90deg bend")

    ap.add_argument("--no-home", action="store_true", help="Do not home at end")
    ap.add_argument("-o", "--out", default="cube_panels.gcode", help="Output filename")
    args = ap.parse_args()

    nozzle_diam_mm = args.nozzle_diam_in * 25.4
    u_per_mm = args.upermm
    if u_per_mm is None:
        u_per_mm = compute_u_per_mm(
            syringe_diam_mm=args.syringe_diam,
            nozzle_diam_mm=nozzle_diam_mm,
            flow_mult=args.flow_mult,
            bead_width_mm=args.bead_width,
            bead_height_mm=args.bead_height,
        )

    offset_xy = args.offset_xy
    offset_z = args.offset_z
    if args.offset is not None:
        offset_xy = args.offset
        offset_z = args.offset

    p = Params(
        edge=args.edge,
        cx=args.center[0], cy=args.center[1], cz=args.center[2],
        f_print=args.fprint,
        f_travel=args.ftravel,
        u_per_mm=u_per_mm,
        u_retract=args.retract,
        dwell_s=args.dwell,
        z_hop=args.zhop,
        b_straight=args.bstraight,
        b_90=args.b90,
        offset_xy_mm=offset_xy,
        offset_z_mm=offset_z,
        home_at_end=not args.no_home,
    )

    lines = generate(p)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"Wrote {args.out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
