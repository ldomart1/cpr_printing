#!/usr/bin/env python3
"""
Interactive 3D tip-position visualizer for G-code using calibration JSON.

Features:
  - True equal-axis rendering (geometry is not distorted) and preserved while scrubbing.
  - Major view plane buttons: XY, XZ, YZ, and ISO.
  - Zoom slider + reset button (scales the equal-axis box around the fixed data center).
  - Dark mode / black background.
  - Slider to scrub through motion points.
  - Current tip/stage markers + live info panel.

Kinematic model (3D rotated radial offset):
  X_tip = X_stage + r(B) * cos(C)
  Y_tip = Y_stage + r(B) * sin(C)
  Z_tip = Z_stage + z(B)

where:
  - r(B), z(B), tip_angle_deg(B) are cubic polynomials from calibration JSON
"""

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, CheckButtons
from mpl_toolkits.mplot3d.art3d import Line3DCollection


# ---------------- Defaults ----------------
DEFAULT_MIN_B = -5.0
DEFAULT_MAX_B = -0.0
DEFAULT_C0_DEG = 0.0
DEFAULT_FIGSIZE = (12.5, 9.0)
# -----------------------------------------


# =========================
# Calibration model
# =========================
@dataclass
class Calibration:
    pr: np.ndarray
    pz: np.ndarray
    pa: np.ndarray
    b_min: float
    b_max: float
    tip_angle_min: float
    tip_angle_max: float
    pull_axis: str
    rot_axis: str
    x_axis: str
    z_axis: str
    c_180_deg: float


def _polyval4(coeffs: np.ndarray, u) -> np.ndarray:
    a, b, c, d = coeffs
    u = np.asarray(u, dtype=float)
    return ((a * u + b) * u + c) * u + d


def load_calibration(json_path: str) -> Calibration:
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(f"Calibration JSON not found: {json_path}")

    with p.open("r") as f:
        data = json.load(f)

    pr = np.array(data["cubic_coefficients"]["r_coeffs"], dtype=float)
    pz = np.array(data["cubic_coefficients"]["z_coeffs"], dtype=float)
    pa = np.array(data["cubic_coefficients"]["tip_angle_coeffs"], dtype=float)

    if pr.shape[0] != 4 or pz.shape[0] != 4 or pa.shape[0] != 4:
        raise ValueError("Expected 4 coeffs for r_coeffs, z_coeffs, and tip_angle_coeffs")

    motor_setup = data.get("motor_setup", {})
    duet_map = data.get("duet_axis_mapping", {})

    b_range = motor_setup.get("b_motor_position_range", [DEFAULT_MIN_B, DEFAULT_MAX_B])
    b_min, b_max = map(float, b_range)
    if b_min > b_max:
        b_min, b_max = b_max, b_min

    pull_axis = str(duet_map.get("pull_axis") or motor_setup.get("b_motor_axis") or "B")
    rot_axis = str(duet_map.get("rotation_axis") or motor_setup.get("rotation_axis") or "C")
    x_axis = str(duet_map.get("horizontal_axis") or motor_setup.get("horizontal_axis") or "X")
    z_axis = str(duet_map.get("vertical_axis") or motor_setup.get("vertical_axis") or "Z")

    c_180 = float(motor_setup.get("rotation_axis_180_deg", 180.0))

    tip_env = data.get("working_envelope", {}).get("tip_angle_range_deg")
    if isinstance(tip_env, list) and len(tip_env) == 2:
        tip_angle_min = float(min(tip_env))
        tip_angle_max = float(max(tip_env))
    else:
        bb = np.linspace(b_min, b_max, 801)
        aa = _polyval4(pa, bb)
        tip_angle_min = float(np.min(aa))
        tip_angle_max = float(np.max(aa))

    return Calibration(
        pr=pr, pz=pz, pa=pa,
        b_min=b_min, b_max=b_max,
        tip_angle_min=tip_angle_min, tip_angle_max=tip_angle_max,
        pull_axis=pull_axis, rot_axis=rot_axis,
        x_axis=x_axis, z_axis=z_axis,
        c_180_deg=c_180
    )


# =========================
# G-code parsing
# =========================
@dataclass
class MotionState:
    idx: int
    gcode_line_no: int
    gcode_raw: str
    motion_code: str
    x_stage: float
    y_stage: float
    z_stage: float
    b_cmd: float
    c_cmd: float
    u_cmd: float
    feed: Optional[float]


_AXVAL_RE = re.compile(r"([A-Za-z])\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+))")


def strip_comment(line: str) -> str:
    no_semicolon = line.split(";", 1)[0]
    no_paren = re.sub(r"\([^)]*\)", "", no_semicolon)
    return no_paren.strip()


def parse_gcode_motion_states(
    gcode_path: str,
    x_axis: str,
    y_axis: str,
    z_axis: str,
    b_axis: str,
    c_axis: str,
    u_axis: str = "U",
    default_c0: float = DEFAULT_C0_DEG,
) -> List[MotionState]:
    p = Path(gcode_path)
    if not p.exists():
        raise FileNotFoundError(f"G-code file not found: {gcode_path}")

    abs_mode = True
    current_motion = "G1"

    pos: Dict[str, float] = {
        x_axis.upper(): 0.0,
        y_axis.upper(): 0.0,
        z_axis.upper(): 0.0,
        b_axis.upper(): 0.0,
        c_axis.upper(): default_c0,
        u_axis.upper(): 0.0,
    }
    current_feed: Optional[float] = None
    states: List[MotionState] = []

    with p.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, raw in enumerate(f, start=1):
            line = strip_comment(raw)
            if not line:
                continue

            uline = line.upper()
            if "G90" in uline:
                abs_mode = True
            if "G91" in uline:
                abs_mode = False

            if re.search(r"(?<!\d)G0(?!\d)", uline):
                current_motion = "G0"
            if re.search(r"(?<!\d)G1(?!\d)", uline):
                current_motion = "G1"

            words = {m.group(1).upper(): float(m.group(2)) for m in _AXVAL_RE.finditer(line)}

            if "F" in words:
                current_feed = float(words["F"])

            pose_axes = {x_axis.upper(), y_axis.upper(), z_axis.upper(), b_axis.upper(), c_axis.upper()}
            is_motion_line = current_motion in ("G0", "G1")
            if not is_motion_line:
                continue

            # Track extrusion axis position even on U-only motion lines so later dU is correct.
            u_axis_u = u_axis.upper()
            if u_axis_u in words:
                if abs_mode:
                    pos[u_axis_u] = float(words[u_axis_u])
                else:
                    pos[u_axis_u] += float(words[u_axis_u])

            has_pose_move = any(ax in words for ax in pose_axes)
            if not has_pose_move:
                continue

            for ax in pose_axes:
                if ax in words:
                    if abs_mode:
                        pos[ax] = float(words[ax])
                    else:
                        pos[ax] += float(words[ax])

            states.append(MotionState(
                idx=len(states),
                gcode_line_no=line_no,
                gcode_raw=raw.rstrip("\n"),
                motion_code=current_motion,
                x_stage=pos[x_axis.upper()],
                y_stage=pos[y_axis.upper()],
                z_stage=pos[z_axis.upper()],
                b_cmd=pos[b_axis.upper()],
                c_cmd=pos[c_axis.upper()],
                u_cmd=pos[u_axis.upper()],
                feed=current_feed,
            ))

    if not states:
        raise RuntimeError("No G0/G1 motion states with tracked axes were found in the G-code file.")

    return states


# =========================
# Tip reconstruction
# =========================
@dataclass
class TipTrajectory:
    x_tip: np.ndarray
    y_tip: np.ndarray
    z_tip: np.ndarray
    x_stage: np.ndarray
    y_stage: np.ndarray
    z_stage: np.ndarray
    b_cmd: np.ndarray
    c_cmd: np.ndarray
    r_of_b: np.ndarray
    z_of_b: np.ndarray
    tip_angle_deg: np.ndarray
    sgn: np.ndarray


def _angle_diff_deg(a: np.ndarray, b: float) -> np.ndarray:
    return ((a - b + 180.0) % 360.0) - 180.0


def infer_sgn_from_c(c_vals: np.ndarray, c0_deg: float, c180_deg: float) -> np.ndarray:
    d0 = np.abs(_angle_diff_deg(c_vals, c0_deg))
    d180 = np.abs(_angle_diff_deg(c_vals, c180_deg))
    return np.where(d0 <= d180, 1.0, -1.0)


def reconstruct_tip_trajectory(states: List[MotionState], cal: Calibration, c0_deg: float = DEFAULT_C0_DEG) -> TipTrajectory:
    x_stage = np.array([s.x_stage for s in states], dtype=float)
    y_stage = np.array([s.y_stage for s in states], dtype=float)
    z_stage = np.array([s.z_stage for s in states], dtype=float)
    b_cmd = np.array([s.b_cmd for s in states], dtype=float)
    c_cmd = np.array([s.c_cmd for s in states], dtype=float)

    r_of_b = _polyval4(cal.pr, b_cmd)
    z_of_b = _polyval4(cal.pz, b_cmd)
    tip_ang = _polyval4(cal.pa, b_cmd)

    # Full 3D tip offset model (continuous C, including unwrapped angles)
    c_rad = np.deg2rad(c_cmd)
    x_tip = x_stage + r_of_b * np.cos(c_rad)
    y_tip = y_stage + r_of_b * np.sin(c_rad)
    z_tip = z_stage + z_of_b

    # Retained for debug display/backward compatibility in the info panel.
    sgn = infer_sgn_from_c(c_cmd, c0_deg=c0_deg, c180_deg=cal.c_180_deg)

    return TipTrajectory(
        x_tip=x_tip, y_tip=y_tip, z_tip=z_tip,
        x_stage=x_stage, y_stage=y_stage, z_stage=z_stage,
        b_cmd=b_cmd, c_cmd=c_cmd,
        r_of_b=r_of_b, z_of_b=z_of_b,
        tip_angle_deg=tip_ang,
        sgn=sgn,
    )


# =========================
# Plot helpers
# =========================
def make_line_segments(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray) -> np.ndarray:
    pts = np.stack([xs, ys, zs], axis=1)
    if len(pts) < 2:
        return np.empty((0, 2, 3), dtype=float)
    return np.stack([pts[:-1], pts[1:]], axis=1)


def compute_equal_box_center_radius(xs: np.ndarray, ys: np.ndarray, zs: np.ndarray, pad_frac: float = 0.04):
    x_min, x_max = float(np.min(xs)), float(np.max(xs))
    y_min, y_max = float(np.min(ys)), float(np.max(ys))
    z_min, z_max = float(np.min(zs)), float(np.max(zs))

    cx = 0.5 * (x_min + x_max)
    cy = 0.5 * (y_min + y_max)
    cz = 0.5 * (z_min + z_max)

    dx = max(x_max - x_min, 1e-9)
    dy = max(y_max - y_min, 1e-9)
    dz = max(z_max - z_min, 1e-9)
    r = 0.5 * max(dx, dy, dz) * (1.0 + pad_frac)
    return (cx, cy, cz), r


def apply_equal_axes_with_zoom(ax, center: Tuple[float, float, float], base_radius: float, zoom: float):
    """
    zoom > 1.0 => zoom in (smaller visible box)
    zoom < 1.0 => zoom out (larger visible box)
    """
    zoom = max(float(zoom), 1e-6)
    cx, cy, cz = center
    r = base_radius / zoom

    ax.set_xlim(cx - r, cx + r)
    ax.set_ylim(cy - r, cy + r)
    ax.set_zlim(cz - r, cz + r)

    try:
        ax.set_box_aspect((1, 1, 1))
    except Exception:
        pass


def set_major_view(ax, view_name: str):
    name = view_name.upper()
    if name == "XY":
        ax.view_init(elev=90, azim=-90)     # looking down +Z
    elif name == "XZ":
        ax.view_init(elev=0, azim=-90)      # looking along +Y
    elif name == "YZ":
        ax.view_init(elev=0, azim=0)        # looking along +X
    elif name == "ISO":
        ax.view_init(elev=25, azim=-60)
    else:
        raise ValueError(f"Unknown view '{view_name}'")


def style_dark_3d_axes(fig, ax):
    # Figure / axes backgrounds
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    # Pane colors (3D walls)
    try:
        ax.xaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
        ax.yaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
        ax.zaxis.set_pane_color((0.0, 0.0, 0.0, 1.0))
    except Exception:
        pass

    # Grid and axis/tick colors (version-compatible best effort)
    try:
        ax.grid(True, color=(0.35, 0.35, 0.35, 0.6))
    except Exception:
        ax.grid(True)

    ax.tick_params(colors="white")
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")
    ax.title.set_color("white")

    # Axis lines (deprecated/private attributes in some versions, guarded)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.line.set_color("white")
        except Exception:
            pass
        try:
            axis._axinfo["grid"]["color"] = (0.35, 0.35, 0.35, 0.6)
            axis._axinfo["tick"]["color"] = (1, 1, 1, 1)
            axis._axinfo["axisline"]["color"] = (1, 1, 1, 1)
        except Exception:
            pass


# =========================
# Interactive UI
# =========================
def launch_interactive_plot(
    states: List[MotionState],
    traj: TipTrajectory,
    cal: Calibration,
    c0_deg: float = DEFAULT_C0_DEG,
    show_stage: bool = False,
):
    n = len(states)
    idx0 = 0

    fig = plt.figure(figsize=DEFAULT_FIGSIZE)
    ax = fig.add_subplot(111, projection="3d")

    # Leave extra room for controls
    plt.subplots_adjust(left=0.05, right=0.98, bottom=0.24, top=0.94)

    # Dark mode
    style_dark_3d_axes(fig, ax)

    xs, ys, zs = traj.x_tip, traj.y_tip, traj.z_tip
    segs = make_line_segments(xs, ys, zs)
    u_cmd = np.array([s.u_cmd for s in states], dtype=float)
    extrude_seg_mask = np.abs(np.diff(u_cmd)) > 1e-12

    # Full reference tip path (uniform color)
    if len(segs) > 0:
        lc = Line3DCollection(segs, colors=["deepskyblue"], linewidths=1.4, alpha=0.45)
        ax.add_collection3d(lc)

    # Optional stage path
    stage_path_line = None
    if show_stage:
        stage_path_line, = ax.plot(
            traj.x_stage, traj.y_stage, traj.z_stage,
            linestyle="--", linewidth=0.7, alpha=0.35, color="0.8", label="Stage path"
        )

    # Start/end markers
    ax.scatter([xs[0]], [ys[0]], [zs[0]], marker="o", s=46, color="lime", label="Start")
    ax.scatter([xs[-1]], [ys[-1]], [zs[-1]], marker="x", s=46, color="red", label="End")

    # Current markers
    current_tip_marker, = ax.plot(
        [xs[idx0]], [ys[idx0]], [zs[idx0]],
        marker="o", markersize=8, linestyle="None", color="white"
    )
    current_stage_marker, = ax.plot(
        [traj.x_stage[idx0]], [traj.y_stage[idx0]], [traj.z_stage[idx0]],
        marker="^", markersize=6, linestyle="None", alpha=0.95, color="magenta"
    )
    # Matplotlib 3D can error when adding an empty Line3DCollection; seed with a degenerate segment.
    p0 = np.array([[xs[idx0], ys[idx0], zs[idx0]], [xs[idx0], ys[idx0], zs[idx0]]], dtype=float)
    current_print_lc = Line3DCollection([p0], colors=["yellow"], linewidths=2.2, alpha=0.95)
    ax.add_collection3d(current_print_lc)

    ax.set_title("Tip Trajectory from G-code and Calibration")
    ax.set_xlabel("X_tip (mm)")
    ax.set_ylabel("Y_tip (mm)")
    ax.set_zlabel("Z_tip (mm)")

    # Legend dark styling
    leg = ax.legend(loc="upper left")
    if leg is not None:
        try:
            leg.get_frame().set_facecolor((0.1, 0.1, 0.1, 0.9))
            leg.get_frame().set_edgecolor((0.7, 0.7, 0.7, 0.5))
            for txt in leg.get_texts():
                txt.set_color("white")
        except Exception:
            pass

    all_x = xs if not show_stage else np.concatenate([xs, traj.x_stage])
    all_y = ys if not show_stage else np.concatenate([ys, traj.y_stage])
    all_z = zs if not show_stage else np.concatenate([zs, traj.z_stage])

    center, base_radius = compute_equal_box_center_radius(all_x, all_y, all_z, pad_frac=0.05)
    current_zoom = {"value": 1.0}

    def apply_view_limits():
        apply_equal_axes_with_zoom(ax, center=center, base_radius=base_radius, zoom=current_zoom["value"])

    apply_view_limits()
    set_major_view(ax, "ISO")

    # ----- Info panel -----
    text_ax = fig.add_axes([0.05, 0.05, 0.60, 0.14])
    text_ax.set_facecolor("black")
    text_ax.axis("off")
    info_text = text_ax.text(
        0.0, 1.0, "",
        va="top", ha="left", family="monospace", fontsize=9, color="white"
    )

    # ----- Index slider -----
    slider_ax = fig.add_axes([0.70, 0.08, 0.25, 0.03], facecolor="#111111")
    idx_slider = Slider(
        ax=slider_ax,
        label="Index",
        valmin=0,
        valmax=n - 1,
        valinit=idx0,
        valstep=1,
        color="#2aa1ff",
    )
    try:
        idx_slider.label.set_color("white")
        idx_slider.valtext.set_color("white")
    except Exception:
        pass

    # ----- Zoom slider -----
    zoom_ax = fig.add_axes([0.70, 0.13, 0.25, 0.03], facecolor="#111111")
    zoom_slider = Slider(
        ax=zoom_ax,
        label="Zoom",
        valmin=0.25,    # zoomed out
        valmax=8.0,     # zoomed in
        valinit=1.0,
        valstep=0.01,
        color="#66d17a",
    )
    try:
        zoom_slider.label.set_color("white")
        zoom_slider.valtext.set_color("white")
    except Exception:
        pass

    # ----- Nav buttons -----
    prev_ax = fig.add_axes([0.70, 0.18, 0.08, 0.04], facecolor="#111111")
    next_ax = fig.add_axes([0.79, 0.18, 0.08, 0.04], facecolor="#111111")
    zrst_ax = fig.add_axes([0.88, 0.18, 0.07, 0.04], facecolor="#111111")
    btn_prev = Button(prev_ax, "Prev", color="#222222", hovercolor="#333333")
    btn_next = Button(next_ax, "Next", color="#222222", hovercolor="#333333")
    btn_zrst = Button(zrst_ax, "Zrst", color="#222222", hovercolor="#333333")

    # ----- Major view plane buttons -----
    xy_ax = fig.add_axes([0.70, 0.225, 0.055, 0.035], facecolor="#111111")
    xz_ax = fig.add_axes([0.762, 0.225, 0.055, 0.035], facecolor="#111111")
    yz_ax = fig.add_axes([0.824, 0.225, 0.055, 0.035], facecolor="#111111")
    iso_ax = fig.add_axes([0.886, 0.225, 0.064, 0.035], facecolor="#111111")

    btn_xy = Button(xy_ax, "XY", color="#222222", hovercolor="#333333")
    btn_xz = Button(xz_ax, "XZ", color="#222222", hovercolor="#333333")
    btn_yz = Button(yz_ax, "YZ", color="#222222", hovercolor="#333333")
    btn_iso = Button(iso_ax, "ISO", color="#222222", hovercolor="#333333")

    # Style button text white
    for b in [btn_prev, btn_next, btn_zrst, btn_xy, btn_xz, btn_yz, btn_iso]:
        try:
            b.label.set_color("white")
        except Exception:
            pass

    # ----- Reference dashed-line visibility -----
    ref_check = None
    if stage_path_line is not None:
        ref_ax = fig.add_axes([0.70, 0.265, 0.16, 0.035], facecolor="#111111")
        ref_check = CheckButtons(ref_ax, ["Ref dashed"], [True])
        try:
            for txt in ref_check.labels:
                txt.set_color("white")
            for rect in ref_check.rectangles:
                rect.set_edgecolor("white")
                rect.set_facecolor("#111111")
            for lines in ref_check.lines:
                for ln in lines:
                    ln.set_color("#66d17a")
        except Exception:
            pass

    help_ax = fig.add_axes([0.70, 0.045, 0.25, 0.02], facecolor="black")
    help_ax.axis("off")
    help_ax.text(
        0.0, 0.5,
        "Keys: ←/→, j/k, 1=XY 2=XZ 3=YZ 0=ISO, +/- zoom",
        va="center", ha="left", fontsize=8, color="white"
    )

    def fmt_state(i: int) -> str:
        s = states[i]
        f_val = s.feed if s.feed is not None else float("nan")
        return (
            f"idx={s.idx:6d}   gcode_line={s.gcode_line_no:6d}   motion={s.motion_code}   F={f_val:.3f}\n"
            f"stage: X={traj.x_stage[i]: .4f}  Y={traj.y_stage[i]: .4f}  Z={traj.z_stage[i]: .4f}  "
            f"{cal.pull_axis}={traj.b_cmd[i]: .4f}  {cal.rot_axis}={traj.c_cmd[i]: .4f}\n"
            f"tip:   X={traj.x_tip[i]: .4f}  Y={traj.y_tip[i]: .4f}  Z={traj.z_tip[i]: .4f}    "
            f"r(B)={traj.r_of_b[i]: .4f}  z(B)={traj.z_of_b[i]: .4f}  tip_angle={traj.tip_angle_deg[i]: .3f}°  sgn={int(traj.sgn[i]):+d}\n"
            f"zoom={current_zoom['value']:.2f}x   raw: {s.gcode_raw}"
        )

    def redraw(i: int):
        i = int(np.clip(i, 0, n - 1))

        current_tip_marker.set_data_3d([traj.x_tip[i]], [traj.y_tip[i]], [traj.z_tip[i]])
        current_stage_marker.set_data_3d([traj.x_stage[i]], [traj.y_stage[i]], [traj.z_stage[i]])
        if i <= 0 or len(segs) == 0:
            current_print_lc.set_segments([])
        else:
            visible_print_segs = segs[:i][extrude_seg_mask[:i]]
            current_print_lc.set_segments(visible_print_segs)

        info_text.set_text(fmt_state(i))

        fig.canvas.draw_idle()

    def on_index_slider(val):
        redraw(int(val))

    def on_zoom_slider(val):
        current_zoom["value"] = float(val)
        apply_view_limits()
        redraw(int(idx_slider.val))

    def on_ref_check(_label):
        if stage_path_line is None:
            return
        stage_path_line.set_visible(not stage_path_line.get_visible())
        fig.canvas.draw_idle()

    def on_prev(event):
        idx_slider.set_val(max(0, int(idx_slider.val) - 1))

    def on_next(event):
        idx_slider.set_val(min(n - 1, int(idx_slider.val) + 1))

    def on_zoom_reset(event):
        zoom_slider.set_val(1.0)

    def make_view_cb(name: str):
        def _cb(event):
            set_major_view(ax, name)
            apply_view_limits()
            fig.canvas.draw_idle()
        return _cb

    btn_xy.on_clicked(make_view_cb("XY"))
    btn_xz.on_clicked(make_view_cb("XZ"))
    btn_yz.on_clicked(make_view_cb("YZ"))
    btn_iso.on_clicked(make_view_cb("ISO"))

    idx_slider.on_changed(on_index_slider)
    zoom_slider.on_changed(on_zoom_slider)
    btn_prev.on_clicked(on_prev)
    btn_next.on_clicked(on_next)
    btn_zrst.on_clicked(on_zoom_reset)
    if ref_check is not None:
        ref_check.on_clicked(on_ref_check)

    def on_key(event):
        key = event.key.lower() if isinstance(event.key, str) else event.key
        i = int(idx_slider.val)

        if key in ("right", "k", "l"):
            idx_slider.set_val(min(n - 1, i + 1))
        elif key in ("left", "j", "h"):
            idx_slider.set_val(max(0, i - 1))
        elif key == "pagedown":
            idx_slider.set_val(min(n - 1, i + 10))
        elif key == "pageup":
            idx_slider.set_val(max(0, i - 10))
        elif key == "home":
            idx_slider.set_val(0)
        elif key == "end":
            idx_slider.set_val(n - 1)
        elif key == "1":
            set_major_view(ax, "XY")
            apply_view_limits()
            fig.canvas.draw_idle()
        elif key == "2":
            set_major_view(ax, "XZ")
            apply_view_limits()
            fig.canvas.draw_idle()
        elif key == "3":
            set_major_view(ax, "YZ")
            apply_view_limits()
            fig.canvas.draw_idle()
        elif key == "0":
            set_major_view(ax, "ISO")
            apply_view_limits()
            fig.canvas.draw_idle()
        elif key in ("+", "="):
            zoom_slider.set_val(min(8.0, float(zoom_slider.val) * 1.15))
        elif key in ("-", "_"):
            zoom_slider.set_val(max(0.25, float(zoom_slider.val) / 1.15))

    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw(idx0)
    plt.show()


# =========================
# CLI / main
# =========================
def main():
    ap = argparse.ArgumentParser(description="Interactive 3D tip-position plot from G-code + calibration JSON.")
    ap.add_argument("--gcode", required=True, help="Path to input G-code file.")
    ap.add_argument("--calibration", required=True, help="Path to calibration JSON (shadow_calibration schema).")
    ap.add_argument("--y-axis", default="Y", help="Stage Y axis letter in G-code (default: Y).")
    ap.add_argument("--c0-deg", type=float, default=DEFAULT_C0_DEG,
                    help="C value corresponding to +X sign in calibration model (default: 0).")
    ap.add_argument("--show-stage", action="store_true", help="Overlay stage path.")
    ap.add_argument("--print-summary", action="store_true", help="Print summary before plotting.")
    args = ap.parse_args()

    cal = load_calibration(args.calibration)
    y_axis = args.y_axis.upper()

    states = parse_gcode_motion_states(
        gcode_path=args.gcode,
        x_axis=cal.x_axis,
        y_axis=y_axis,
        z_axis=cal.z_axis,
        b_axis=cal.pull_axis,
        c_axis=cal.rot_axis,
        default_c0=args.c0_deg,
    )
    traj = reconstruct_tip_trajectory(states, cal=cal, c0_deg=args.c0_deg)

    if args.print_summary:
        c_unique = np.unique(np.round(traj.c_cmd, 6))
        print(f"Parsed {len(states)} motion states from {args.gcode}")
        print(f"Axis mapping: X={cal.x_axis}, Y={y_axis}, Z={cal.z_axis}, B={cal.pull_axis}, C={cal.rot_axis}")
        print(f"C0={args.c0_deg:.3f}, C180(cal)={cal.c_180_deg:.3f}")
        print(f"Unique C values (rounded): {c_unique.tolist()[:20]}{' ...' if len(c_unique) > 20 else ''}")
        print(f"B range used: [{np.min(traj.b_cmd):.4f}, {np.max(traj.b_cmd):.4f}] (cal: [{cal.b_min:.4f}, {cal.b_max:.4f}])")
        print(f"TIP X range: [{np.min(traj.x_tip):.4f}, {np.max(traj.x_tip):.4f}]")
        print(f"TIP Y range: [{np.min(traj.y_tip):.4f}, {np.max(traj.y_tip):.4f}]")
        print(f"TIP Z range: [{np.min(traj.z_tip):.4f}, {np.max(traj.z_tip):.4f}]")

    launch_interactive_plot(
        states=states,
        traj=traj,
        cal=cal,
        c0_deg=args.c0_deg,
        show_stage=args.show_stage,
    )

if __name__ == "__main__":
    main()
