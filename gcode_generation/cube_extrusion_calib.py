#!/usr/bin/env python3
"""
Duet/RRF G-code generator: extrusion calibration by drawing parallel lines.

Pattern:
  - 7 parallel lines along X, each 30mm long
  - spaced 5mm in Y
  - repeated on 4 Z planes, spaced 10mm

Goal:
  - calibrate U extrusion factor (U per mm of travel) and/or compare multiple U settings
  - optionally sweep U_per_mm across the 7 lines

Assumptions:
  - machine is already homed
  - XYZ absolute positioning
  - U relative extrusion (M83)
"""

import argparse
from dataclasses import dataclass
from typing import List, Optional


@dataclass
class Params:
    # geometry
    x0: float
    y0: float
    z0: float
    line_len: float
    n_lines: int
    line_pitch_y: float
    n_planes: int
    plane_pitch_z: float

    # motion/extrusion
    f_print: float       # mm/min
    f_travel: float      # mm/min
    u_per_mm: float      # base U per mm
    u_sweep: bool        # if true, vary U across lines
    u_min: float         # used if u_sweep
    u_max: float         # used if u_sweep

    # stop-flow
    u_retract: float
    dwell_s: float

    # tube state (kept simple / straight)
    b_straight: float
    c_straight: float

    # program end
    home_at_end: bool


def header(p: Params) -> List[str]:
    return [
        "; --- Extrusion calibration lines (Duet/RRF) ---",
        f"; Start: X={p.x0} Y={p.y0} Z={p.z0}",
        f"; Lines: {p.n_lines} x {p.line_len}mm along +X, Y pitch {p.line_pitch_y}mm",
        f"; Planes: {p.n_planes} planes, Z pitch {p.plane_pitch_z}mm",
        f"; F_print={p.f_print} mm/min, F_travel={p.f_travel} mm/min",
        f"; U_per_mm base={p.u_per_mm}",
        (f"; U sweep enabled: {p.u_min} -> {p.u_max} across {p.n_lines} lines"
         if p.u_sweep else "; U sweep disabled (constant U_per_mm)"),
        f"; Stop-flow: retract={p.u_retract}, dwell={p.dwell_s}s",
        "; Assumes machine is already homed.",
        "",
        "G90                         ; absolute positioning (XYZBC)",
        "M83                         ; relative extrusion (U)",
        "G92 U0                      ; reset U",
        f"G1 B{p.b_straight:.4f} C{p.c_straight:.5f} F2000  ; tube straight",
        "M400                        ; wait",
        "",
    ]


def stop_flow(p: Params) -> List[str]:
    return [
        f"G1 U{-p.u_retract:.3f} F300       ; retract U to stop flow",
        f"G4 S{p.dwell_s:.0f}                ; dwell",
    ]


def move_travel(x: float, y: float, z: float, f: float, comment: str = "") -> str:
    cmt = f" ; {comment}" if comment else ""
    return f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} F{f:.0f}{cmt}"


def move_extrude(x: float, y: float, z: float, u: float, f: float, comment: str = "") -> str:
    cmt = f" ; {comment}" if comment else ""
    return f"G1 X{x:.3f} Y{y:.3f} Z{z:.3f} U{u:.5f} F{f:.0f}{cmt}"


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def u_per_mm_for_line(p: Params, i: int) -> float:
    if not p.u_sweep or p.n_lines <= 1:
        return p.u_per_mm
    t = i / (p.n_lines - 1)
    return lerp(p.u_min, p.u_max, t)


def generate(p: Params) -> List[str]:
    g: List[str] = []
    g += header(p)

    for plane in range(p.n_planes):
        z = p.z0 + plane * p.plane_pitch_z
        g += ["", f"; --- Plane {plane+1}/{p.n_planes} at Z={z:.3f} ---"]

        for i in range(p.n_lines):
            y = p.y0 + i * p.line_pitch_y
            uperm = u_per_mm_for_line(p, i)
            u_amt = p.line_len * uperm

            g += stop_flow(p)

            # Travel to line start
            g += [move_travel(p.x0, y, z, p.f_travel, f"travel to line {i+1} start (U/mm={uperm:.5f})")]

            # Extrude line along +X
            x1 = p.x0 + p.line_len
            g += [move_extrude(x1, y, z, u_amt, p.f_print, f"print line {i+1} (U/mm={uperm:.5f})")]

    g += ["", "; --- Finish ---"]
    g += stop_flow(p)
    g += ["G1 Z{:.3f} F{:.0f} ; lift".format(p.z0 + (p.n_planes - 1) * p.plane_pitch_z + 5.0, p.f_travel)]
    if p.home_at_end:
        g += ["G28                         ; home all axes"]
    g += ["M400                        ; wait",
          "; --- End ---"]
    return g


def main():
    ap = argparse.ArgumentParser(description="Generate Duet G-code for extrusion calibration lines.")
    ap.add_argument("-o", "--out", default="extrusion_calibration.gcode", help="Output file name")

    # Start location
    ap.add_argument("--start", type=float, nargs=3, default=[0.0, 0.0, 0.0], metavar=("X0", "Y0", "Z0"),
                    help="Start position for first line (default 0 0 0)")

    # Pattern settings (your requested defaults)
    ap.add_argument("--len", type=float, default=30.0, help="Line length in mm (default 30)")
    ap.add_argument("--lines", type=int, default=7, help="Number of lines (default 7)")
    ap.add_argument("--ypitch", type=float, default=5.0, help="Y spacing between lines (default 5)")
    ap.add_argument("--planes", type=int, default=4, help="Number of Z planes (default 4)")
    ap.add_argument("--zpitch", type=float, default=10.0, help="Z spacing between planes (default 10)")

    # Motion / extrusion
    ap.add_argument("--fprint", type=float, default=1200.0, help="Print feedrate mm/min (default 1200)")
    ap.add_argument("--ftravel", type=float, default=3000.0, help="Travel feedrate mm/min (default 3000)")
    ap.add_argument("--upermm", type=float, default=0.020, help="Base U per mm (default 0.020)")

    # Sweep options
    ap.add_argument("--sweep", action="store_true", help="Sweep U per mm across the 7 lines")
    ap.add_argument("--umin", type=float, default=0.010, help="Min U per mm for sweep (default 0.010)")
    ap.add_argument("--umax", type=float, default=0.040, help="Max U per mm for sweep (default 0.040)")

    # Stop-flow
    ap.add_argument("--retract", type=float, default=0.100, help="Retract amount on U (default 0.1)")
    ap.add_argument("--dwell", type=float, default=5.0, help="Dwell seconds after retract (default 5)")

    # Tube state
    ap.add_argument("--bstraight", type=float, default=0.0, help="B value for straight tube (default 0)")
    ap.add_argument("--cstraight", type=float, default=0.0, help="C value for straight tube (default 0)")

    ap.add_argument("--no-home", action="store_true", help="Do not home at end")

    args = ap.parse_args()

    p = Params(
        x0=args.start[0], y0=args.start[1], z0=args.start[2],
        line_len=args.len,
        n_lines=args.lines,
        line_pitch_y=args.ypitch,
        n_planes=args.planes,
        plane_pitch_z=args.zpitch,
        f_print=args.fprint,
        f_travel=args.ftravel,
        u_per_mm=args.upermm,
        u_sweep=args.sweep,
        u_min=args.umin,
        u_max=args.umax,
        u_retract=args.retract,
        dwell_s=args.dwell,
        b_straight=args.bstraight,
        c_straight=args.cstraight,
        home_at_end=not args.no_home,
    )

    lines = generate(p)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {args.out} ({len(lines)} lines).")


if __name__ == "__main__":
    main()
