#!/usr/bin/env python3
"""
Generate Duet/RRF G-code to extrude the edges of a cube.

Axes:
  - Motion: X, Y, Z
  - Extrusion: U

Order:
  1) Bottom square (Z = z0)
  2) Vertical edges (4, one by one)
  3) Top square (Z = z1)
  4) Stop extrusion, retract, home

Defaults:
  edge = 20 mm
  center = (30, 30, -30)
  print feedrate = 1200 mm/min (20 mm/s)
"""

import argparse
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Params:
    edge: float
    cx: float
    cy: float
    cz: float
    f_print: float          # mm/min
    f_travel: float         # mm/min
    u_per_mm: float         # extrusion units per mm of travel
    u_mult: float           # multiplier applied to u_per_mm
    z_hop: float
    start_u: float
    home_at_end: bool


def fmt_xyz(x: float, y: float, z: float) -> str:
    return f"X{x:.3f} Y{y:.3f} Z{z:.3f}"


def gcode_header(p: Params) -> List[str]:
    return [
        "; --- Cube edge G-code (Duet/RRF) ---",
        f"; edge={p.edge}mm center=({p.cx},{p.cy},{p.cz})",
        f"; F_print={p.f_print} mm/min, F_travel={p.f_travel} mm/min",
        f"; U_per_mm={p.u_per_mm}, U_mult={p.u_mult}",
        "; Assumes machine is already homed.",
        "",
        "G90                         ; absolute positioning (XYZ)",
        "M83                         ; relative extrusion (U)",
        f"G92 U{p.start_u:.3f}                 ; reset extrusion axis position",
        "M400                        ; wait for moves to finish",
        "",
    ]


def travel_to(p: Params, x: float, y: float, z: float, comment: str = "") -> List[str]:
    cmt = f" ; {comment}" if comment else ""
    return [f"G1 {fmt_xyz(x, y, z)} F{p.f_travel:.0f}{cmt}"]


def extrude_line(p: Params,
                 x0: float, y0: float, z0: float,
                 x1: float, y1: float, z1: float,
                 comment: str = "") -> List[str]:
    # For extrusion amount, use path length * u_per_mm
    dx, dy, dz = (x1 - x0), (y1 - y0), (z1 - z0)
    length = (dx*dx + dy*dy + dz*dz) ** 0.5
    u_amt = length * p.u_per_mm * p.u_mult

    cmt = f" ; {comment}" if comment else ""
    return [
        f"G1 {fmt_xyz(x1, y1, z1)} U{u_amt:.5f} F{p.f_print:.0f}{cmt}"
    ]


def u_per_mm_from_diameters(syringe_diam_mm: float, nozzle_diam_mm: float) -> float:
    syringe_area = (syringe_diam_mm ** 2)
    nozzle_area = (nozzle_diam_mm ** 2)
    return nozzle_area / syringe_area


def square_corners(edge: float, cx: float, cy: float, z: float) -> List[Tuple[float, float, float]]:
    half = edge / 2.0
    # Order: around the square, returning to start later
    return [
        (cx - half, cy - half, z),
        (cx + half, cy - half, z),
        (cx + half, cy + half, z),
        (cx - half, cy + half, z),
    ]


def generate(p: Params) -> List[str]:
    g: List[str] = []
    g += gcode_header(p)

    half = p.edge / 2.0
    z0 = p.cz - half
    z1 = p.cz + half

    bottom = square_corners(p.edge, p.cx, p.cy, z0)
    top = square_corners(p.edge, p.cx, p.cy, z1)

    # Helper to do a safe z-hop travel (optional)
    def hop_z(z_current: float) -> float:
        return z_current + p.z_hop if p.z_hop > 0 else z_current

    # --- 1) Bottom square ---
    g += ["; --- Bottom plane (square) ---"]
    x_start, y_start, z_start = bottom[0]
    g += travel_to(p, x_start, y_start, hop_z(z_start), "travel above start")
    if p.z_hop > 0:
        g += travel_to(p, x_start, y_start, z_start, "drop to bottom Z")

    # Trace bottom edges
    x_prev, y_prev, z_prev = x_start, y_start, z_start
    for i in range(1, len(bottom)):
        x, y, z = bottom[i]
        g += extrude_line(p, x_prev, y_prev, z_prev, x, y, z, f"bottom edge {i}")
        x_prev, y_prev, z_prev = x, y, z
    # Close the square
    g += extrude_line(p, x_prev, y_prev, z_prev, x_start, y_start, z_start, "bottom edge close")

    # --- 2) Vertical edges (one by one) ---
    g += ["", "; --- Vertical edges ---"]
    for idx in range(4):
        xb, yb, _ = bottom[idx]
        xt, yt, _ = top[idx]

        g += travel_to(p, xb, yb, hop_z(z0), f"travel to vertical edge {idx+1} base")
        if p.z_hop > 0:
            g += travel_to(p, xb, yb, z0, f"drop to base of vertical edge {idx+1}")

        # Extrude straight up to top
        g += extrude_line(p, xb, yb, z0, xt, yt, z1, f"vertical edge {idx+1}")

    # --- 3) Top square ---
    g += ["", "; --- Top plane (square) ---"]
    x_start, y_start, z_start = top[0]
    g += travel_to(p, x_start, y_start, hop_z(z_start), "travel above top start")
    if p.z_hop > 0:
        g += travel_to(p, x_start, y_start, z_start, "drop to top Z")

    x_prev, y_prev, z_prev = x_start, y_start, z_start
    for i in range(1, len(top)):
        x, y, z = top[i]
        g += extrude_line(p, x_prev, y_prev, z_prev, x, y, z, f"top edge {i}")
        x_prev, y_prev, z_prev = x, y, z
    g += extrude_line(p, x_prev, y_prev, z_prev, x_start, y_start, z_start, "top edge close")

    # --- Finish ---
    g += ["", "; --- Finish / return to origin ---"]
    # lift a bit before returning, to be polite
    g += [f"G1 Z{(z1 + max(2.0, p.z_hop)):.3f} F{p.f_travel:.0f} ; lift before return"]
    if p.home_at_end:
        g += ["G1 X0.000 Y0.000 Z0.000 F{:.0f} ; return to origin".format(p.f_travel)]
    g += ["M400                        ; wait for moves to finish",
          "; --- End ---"]

    return g


def main():
    ap = argparse.ArgumentParser(description="Generate Duet G-code to extrude cube edges (XYZ + U).")
    ap.add_argument("--edge", type=float, default=30.0, help="Cube edge size in mm (default 20)")
    ap.add_argument("--center", type=float, nargs=3, default=[15.0, 15.0, -20.0],
                    metavar=("CX", "CY", "CZ"), help="Cube center in mm (default 30 30 -30)")
    ap.add_argument("--fprint", type=float, default=400.0, help="Print feedrate in mm/min (default 1200)")
    ap.add_argument("--ftravel", type=float, default=800.0, help="Travel feedrate in mm/min (default 3000)")
    ap.add_argument("--syringe-diam", type=float, default=14.0,
                    help="Syringe inner diameter in mm (default 14)")
    ap.add_argument("--nozzle-diam-in", type=float, default=0.02,
                    help="Nozzle diameter in inches (default 0.02)")
    ap.add_argument("--nozzle-diam-mm", type=float, default=None,
                    help="Nozzle diameter in mm (overrides --nozzle-diam-in)")
    ap.add_argument("--upermm", type=float, default=None,
                    help="U extrusion per mm of travel (overrides diameter-based calc)")
    ap.add_argument("--umult", type=float, default=50.0,
                    help="Multiplier applied to U per mm (use >1.0 to push more)")
    ap.add_argument("--zhop", type=float, default=0.0, help="Z hop (mm) on travels (default 0)")
    ap.add_argument("--startu", type=float, default=0.0, help="Initial U reset value for G92 (default 0)")
    ap.add_argument("--no-home", action="store_true", help="Do not home at end")
    ap.add_argument("-o", "--out", default="cube_edges.gcode", help="Output filename (default cube_edges.gcode)")
    args = ap.parse_args()

    nozzle_diam_mm = args.nozzle_diam_mm
    if nozzle_diam_mm is None:
        nozzle_diam_mm = args.nozzle_diam_in * 25.4

    u_per_mm = args.upermm
    if u_per_mm is None:
        u_per_mm = u_per_mm_from_diameters(args.syringe_diam, nozzle_diam_mm)

    p = Params(
        edge=args.edge,
        cx=args.center[0],
        cy=args.center[1],
        cz=args.center[2],
        f_print=args.fprint,
        f_travel=args.ftravel,
        u_per_mm=u_per_mm,
        u_mult=args.umult,
        z_hop=args.zhop,
        start_u=args.startu,
        home_at_end=not args.no_home,
    )

    lines = generate(p)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print(f"Wrote {args.out} ({len(lines)} lines)")


if __name__ == "__main__":
    main()
