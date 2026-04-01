#!/usr/bin/env python3
import csv
import inspect
import math
import os
import shutil
import sys
import time
from datetime import datetime
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from shadow_calibration import CTR_Shadow_Calibration, refine_tip_parallel_centerline


# =========================
# USER CONFIGURATION
# =========================
PROJECT_NAME = "Drift_Test_2026-04_01_02"
ALLOW_EXISTING_PROJECT = True
ADD_DATE_TO_PROJECT_FOLDER = False

MANUAL_CROP_ADJUSTMENT = True
THRESHOLD = 200
USE_EXACT_CLASS_THRESHOLDING = True  # True = match analyze_data() Otsu behavior from shadow_calibration.py

PULL_B_START = 0.3
PULL_B_STEPS = 24
PULL_B_STEP_SIZE = -0.25
NUM_SEQUENCES = 500

CAMERA_PORT = 0
SHOW_CAMERA_PREVIEW = False
JOGGING_FEEDRATE = 600
CAPTURE_DWELL_S = 0.5
CAMERA_WARMUP_FRAMES = 2
CAPTURE_RETRY_LIMIT = 10
CAPTURE_RETRY_WAIT_S = 0.5
ANNOTATED_IMAGE_EXTENSION = ".jpg"
ANNOTATED_IMAGE_JPEG_QUALITY = 88

ROBOT_FRONT_AXIS_NAME = "X"
ROBOT_STAGE_Y_AXIS_NAME = "Y"
ROBOT_STAGE_Z_AXIS_NAME = "Z"
ROBOT_REAR_AXIS_NAME = "B"

CAMERA_CALIBRATION_FILE = os.path.join(SCRIPT_DIR, "captures/calibration_webcam_20260331_100557.npz")
BOARD_REFERENCE_IMAGE = os.path.join(SCRIPT_DIR, "captures/photo_20260331_100553.png")

PROBE_MODE = "middle"  # "middle" | "five"
RAW_IMAGE_EXTENSION = ".jpg"  # ".jpg" is compressed and reliable with cv2.imread
RAW_IMAGE_JPEG_QUALITY = 95
ARCHIVE_RAW_IMAGE_FOLDER_AT_END = True
RESET_EXISTING_OUTPUTS = True

# If True, keep the class crop/setup behavior exactly like your existing workflow.
USE_CLASS_ANALYSIS_CROP_SETUP = True

if PROBE_MODE == "middle":
    PROBE_POINTS = [(100.0, 52.0, -155.0)]
elif PROBE_MODE == "five":
    PROBE_POINTS = [
        (30.0, 0.0, -70.0),
        (125.0, 0.0, -70.0),
        (30.0, 0.0, -110.0),
        (125.0, 0.0, -110.0),
        (77.5, 0.0, -100.0),
    ]
else:
    raise ValueError(f"Unknown PROBE_MODE: {PROBE_MODE}")


# =========================
# HELPERS
# =========================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def reset_dir(path: str) -> str:
    if os.path.isdir(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)
    return path


def wait_for_duet_motion_complete(cal: CTR_Shadow_Calibration, extra_settle: float = 0.0) -> None:
    try:
        cal.rrf.send_code("M400")
    except Exception as exc:
        print(f"[WARN] M400 wait failed ({exc}). Falling back to timed settle only.")
    if extra_settle > 0:
        time.sleep(extra_settle)


def move_abs(
    cal: CTR_Shadow_Calibration,
    current_pos: Dict[str, float],
    feedrate: float,
    **axes_targets: float,
) -> Dict[str, float]:
    cmd = ["G90", "G1"]
    target_pos = dict(current_pos)
    moved = False
    for axis_name, axis_val in axes_targets.items():
        if axis_val is None:
            continue
        cmd.append(f"{axis_name}{float(axis_val):.5f}")
        target_pos[axis_name] = float(axis_val)
        moved = True
    cmd.append(f"F{float(feedrate):.3f}")
    gcode = " ".join(cmd)
    if moved:
        print(f"[MOVE] {gcode}")
        cal.rrf.send_code(gcode)
    wait_for_duet_motion_complete(cal, extra_settle=0.0)
    return target_pos


def compute_checkpoint_positions(b_start: float, b_steps: int, b_step_size: float) -> Dict[str, float]:
    if b_steps < 1:
        raise ValueError("b_steps must be >= 1")
    end_b = float(b_start + b_steps * b_step_size)
    fraction_by_checkpoint = {
        "pull_25": 0.25,
        "pull_middle": 0.50,
        "pull_75": 0.75,
        "pull_85": 0.85,
        "pull_end": 1.00,
    }
    checkpoint_b = {
        checkpoint_name: float(b_start + (b_steps * fraction) * b_step_size)
        for checkpoint_name, fraction in fraction_by_checkpoint.items()
    }
    checkpoint_b["release_85"] = checkpoint_b["pull_85"]
    checkpoint_b["release_75"] = checkpoint_b["pull_75"]
    checkpoint_b["release_middle"] = checkpoint_b["pull_middle"]
    checkpoint_b["release_25"] = checkpoint_b["pull_25"]
    checkpoint_b["release_end"] = float(b_start)
    return checkpoint_b


CHECKPOINT_ORDER = [
    "pull_25",
    "pull_middle",
    "pull_75",
    "pull_85",
    "pull_end",
    "release_85",
    "release_75",
    "release_middle",
    "release_25",
    "release_end",
]

CHECKPOINT_COLORS = {
    "pull_25": "#4cc9f0",
    "pull_middle": "#7fd6ff",
    "pull_75": "#80ed99",
    "pull_85": "#f8961e",
    "pull_end": "#ffd166",
    "release_85": "#ffb703",
    "release_75": "#f28482",
    "release_middle": "#ff8fab",
    "release_25": "#c77dff",
    "release_end": "#9bffb0",
}


def _make_checkpoint_grid(num_items: int) -> Tuple[plt.Figure, np.ndarray]:
    num_cols = 5
    num_rows = 2
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(5.2 * num_cols, 4.6 * num_rows),
        squeeze=False,
    )
    return fig, axes


def capture_frame(camera: cv2.VideoCapture, warmup_frames: int = 2) -> np.ndarray:
    frame = None
    for _ in range(max(1, int(warmup_frames))):
        _ret, _frame = camera.read()
        if _ret and _frame is not None:
            frame = _frame
    ret, frame2 = camera.read()
    if ret and frame2 is not None:
        frame = frame2
    if frame is None:
        raise RuntimeError("Could not capture frame from camera.")
    return frame


def save_compressed_image(
    image_bgr: np.ndarray,
    output_path: str,
    jpeg_quality: Optional[int] = None,
) -> None:
    ext = os.path.splitext(output_path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        quality = RAW_IMAGE_JPEG_QUALITY if jpeg_quality is None else int(jpeg_quality)
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    else:
        params = []
    ok = cv2.imwrite(output_path, image_bgr, params)
    if not ok:
        raise RuntimeError(f"Failed to save image: {output_path}")


def build_annotated_frame(
    image_bgr: np.ndarray,
    analysis: Dict[str, float],
    probe_idx: int,
    sequence_idx: int,
    checkpoint_name: str,
    b_target: float,
) -> np.ndarray:
    annotated = image_bgr.copy()
    tip_x = int(round(float(analysis["tip_col_px"])))
    tip_y = int(round(float(analysis["tip_row_px"])))

    cv2.drawMarker(
        annotated,
        (tip_x, tip_y),
        color=(0, 215, 255),
        markerType=cv2.MARKER_CROSS,
        markerSize=28,
        thickness=2,
        line_type=cv2.LINE_AA,
    )
    cv2.circle(annotated, (tip_x, tip_y), 13, (255, 150, 40), 2, lineType=cv2.LINE_AA)

    lines = [
        f"Probe {probe_idx:02d} | Seq {sequence_idx:04d} | {checkpoint_name}",
        f"B = {b_target:.3f} | tip(px) = ({analysis['tip_col_px']:.1f}, {analysis['tip_row_px']:.1f})",
        f"tip(mm) = ({analysis['tip_u_mm']:.3f}, {analysis['tip_z_mm']:.3f}) | angle = {analysis['tip_angle_deg']:.2f} deg",
    ]

    banner_h = 96
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (annotated.shape[1], banner_h), (8, 10, 16), thickness=-1)
    annotated = cv2.addWeighted(overlay, 0.82, annotated, 0.18, 0.0)

    y = 28
    for idx, line in enumerate(lines):
        scale = 0.68 if idx == 0 else 0.58
        cv2.putText(
            annotated,
            line,
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (232, 240, 248),
            2,
            lineType=cv2.LINE_AA,
        )
        y += 28

    return annotated


def analyze_frame_with_ctr_class(
    cal: CTR_Shadow_Calibration,
    image_bgr: np.ndarray,
    threshold: int,
) -> Dict[str, float]:
    """
    Lightweight version of analyze_data() for single frames.
    Reuses the CTR_Shadow_Calibration image-analysis logic without generating plots.
    """
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    crop = cal.analysis_crop
    if crop is None:
        raise RuntimeError("Analysis crop is not configured. Run setup_analysis_crop() first.")

    img_h, img_w = image_bgr.shape[:2]
    crop_x_min_img = int(crop["crop_width_min"])
    crop_x_max_img = int(crop["crop_width_max"])
    crop_y_min_img = int(img_h - crop["crop_height_max"])
    crop_y_max_img = int(img_h - crop["crop_height_min"])

    crop_x_min_img = max(0, min(crop_x_min_img, img_w - 1))
    crop_x_max_img = max(crop_x_min_img + 1, min(crop_x_max_img, img_w))
    crop_y_min_img = max(0, min(crop_y_min_img, img_h - 1))
    crop_y_max_img = max(crop_y_min_img + 1, min(crop_y_max_img, img_h))

    cropped_image = image_bgr[crop_y_min_img:crop_y_max_img, crop_x_min_img:crop_x_max_img, :]
    grayscale_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    grayscale_eq = clahe.apply(grayscale_image)
    grayscale_blur = cv2.GaussianBlur(grayscale_eq, (3, 3), 0)

    if USE_EXACT_CLASS_THRESHOLDING:
        _thr, binary_image = cv2.threshold(grayscale_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        _thr, binary_image = cv2.threshold(grayscale_blur, int(threshold), 255, cv2.THRESH_BINARY)

    tip_row, tip_column, tip_angle_deg, _tip_debug = cal.find_ctr_tip_skeleton(
        binary_image,
        min_spur_len=25,
        return_tip_angle=True,
        return_debug=True,
    )

    yy_refined, xx_refined, _tip_refine_dbg = refine_tip_parallel_centerline(
        grayscale=grayscale_image,
        binary_image=binary_image,
        tip_yx=(int(round(float(tip_row))), int(round(float(tip_column)))),
        tip_angle_deg=float(tip_angle_deg),
        section_near_r=float(cal.tip_parallel_section_near_r),
        section_far_r=float(cal.tip_parallel_section_far_r),
        scan_half_r=float(cal.tip_parallel_scan_half_r),
        num_sections=int(cal.tip_parallel_num_sections),
        cross_step_px=float(cal.tip_parallel_cross_step_px),
        ray_step_px=float(cal.tip_parallel_ray_step_px),
        ray_max_len_r=float(cal.tip_parallel_ray_max_len_r),
    )

    tip_row_full_px = float(yy_refined + crop_y_min_img)
    tip_col_full_px = float(xx_refined + crop_x_min_img)

    result = {
        "tip_row_px": tip_row_full_px,
        "tip_col_px": tip_col_full_px,
        "tip_angle_deg": float(tip_angle_deg),
    }

    # Reuse the calibrated board/ruler conversion when available.
    try:
        u_mm, z_mm = cal.pixel_point_to_calibrated_axes(
            x_px=tip_col_full_px,
            y_px=tip_row_full_px,
        )
        result["tip_u_mm"] = float(u_mm)
        result["tip_z_mm"] = float(z_mm)
        result["location_units"] = "mm"
    except Exception:
        result["tip_u_mm"] = float("nan")
        result["tip_z_mm"] = float("nan")
        result["location_units"] = "px"

    return result


def append_result_row(csv_path: str, row: Dict[str, object]) -> None:
    file_exists = os.path.isfile(csv_path)
    fieldnames = list(row.keys())
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)
        f.flush()


def archive_raw_images(raw_dir: str) -> Optional[str]:
    if not os.path.isdir(raw_dir):
        return None
    archive_base = os.path.abspath(raw_dir.rstrip(os.sep))
    archive_path = shutil.make_archive(archive_base, "gztar", root_dir=raw_dir)
    return archive_path


def _apply_dark_axes_style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, color="#f4f7fb", fontsize=13, pad=10, weight="semibold")
    ax.set_xlabel(xlabel, color="#d7e2ee")
    ax.set_ylabel(ylabel, color="#d7e2ee")
    ax.tick_params(colors="#c8d5e3", labelsize=10)
    for spine in ax.spines.values():
        spine.set_color((0.75, 0.84, 0.93, 0.25))
        spine.set_linewidth(1.1)
    ax.grid(True, color=(0.75, 0.84, 0.93, 0.10), linewidth=0.8)
    ax.set_facecolor("#0f1723")


def _make_dark_density_cmap() -> LinearSegmentedColormap:
    return LinearSegmentedColormap.from_list(
        "drift_density_dark",
        [
            "#0a0f18",
            "#122033",
            "#17324d",
            "#1c4f73",
            "#1f6fa8",
            "#26a0b8",
            "#79d9cf",
            "#f3d67a",
        ],
        N=256,
    )


def _save_density_animation_gif(
    plot_df: pd.DataFrame,
    x_col: str,
    y_col: str,
    xlabel: str,
    ylabel: str,
    output_path: str,
) -> Optional[str]:
    if plot_df.empty or "sequence_idx" not in plot_df.columns:
        return None

    seq_vals = pd.to_numeric(plot_df["sequence_idx"], errors="coerce")
    finite_seq = np.isfinite(seq_vals.to_numpy(dtype=float))
    if not np.any(finite_seq):
        return None

    plot_df = plot_df.loc[finite_seq].copy()
    plot_df["sequence_idx"] = seq_vals.loc[finite_seq].astype(int)
    min_seq = int(plot_df["sequence_idx"].min())
    max_seq = int(plot_df["sequence_idx"].max())
    if max_seq < min_seq:
        return None

    fps = 20
    idx_per_second = 100
    idx_per_frame = max(1, int(math.ceil(idx_per_second / float(fps))))
    frame_cutoffs = list(range(min_seq, max_seq + 1, idx_per_frame))
    if not frame_cutoffs or frame_cutoffs[-1] != max_seq:
        frame_cutoffs.append(max_seq)

    global_x = pd.to_numeric(plot_df[x_col], errors="coerce").to_numpy(dtype=float)
    global_y = pd.to_numeric(plot_df[y_col], errors="coerce").to_numpy(dtype=float)
    finite_global = np.isfinite(global_x) & np.isfinite(global_y)
    if not np.any(finite_global):
        return None

    global_x = global_x[finite_global]
    global_y = global_y[finite_global]
    x_span = float(np.ptp(global_x))
    y_span = float(np.ptp(global_y))
    x_pad = max(0.15, 0.12 * max(x_span, 1.0))
    y_pad = max(0.15, 0.12 * max(y_span, 1.0))
    x_limits = (float(np.min(global_x) - x_pad), float(np.max(global_x) + x_pad))
    y_limits = (float(np.min(global_y) - y_pad), float(np.max(global_y) + y_pad))

    fig, axes = _make_checkpoint_grid(len(CHECKPOINT_ORDER))
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    def draw_frame(cutoff: int) -> None:
        for ax, checkpoint_name in zip(axes.flat, CHECKPOINT_ORDER):
            ax.clear()
            checkpoint_df = plot_df[
                (plot_df["checkpoint"] == checkpoint_name) & (plot_df["sequence_idx"] <= cutoff)
            ].copy()
            if not checkpoint_df.empty:
                x_vals = pd.to_numeric(checkpoint_df[x_col], errors="coerce").to_numpy(dtype=float)
                y_vals = pd.to_numeric(checkpoint_df[y_col], errors="coerce").to_numpy(dtype=float)
                finite_mask = np.isfinite(x_vals) & np.isfinite(y_vals)
                x_vals = x_vals[finite_mask]
                y_vals = y_vals[finite_mask]

                if x_vals.size > 0:
                    ax.hexbin(
                        x_vals,
                        y_vals,
                        gridsize=22,
                        mincnt=1,
                        cmap=_make_dark_density_cmap(),
                        linewidths=0.2,
                        edgecolors=(1.0, 1.0, 1.0, 0.12),
                    )
                    ax.scatter(
                        x_vals,
                        y_vals,
                        s=22,
                        color=CHECKPOINT_COLORS.get(checkpoint_name, "#d6dee8"),
                        alpha=0.28,
                        edgecolors="none",
                        zorder=3,
                    )

            _apply_dark_axes_style(
                ax,
                title=checkpoint_name.replace("_", " ").title(),
                xlabel=xlabel,
                ylabel=ylabel,
            )
            ax.set_xlim(*x_limits)
            ax.set_ylim(*y_limits)

        for ax in axes.flat[len(CHECKPOINT_ORDER):]:
            ax.set_visible(False)

        fig.suptitle(
            f"Tracked tip position density by checkpoint | seq <= {cutoff}",
            color="#f7fbff",
            fontsize=15,
            weight="semibold",
            y=0.99,
        )

    animation = FuncAnimation(fig, draw_frame, frames=frame_cutoffs, interval=1000 / fps, repeat=False)
    animation.save(output_path, writer=PillowWriter(fps=fps), dpi=160)
    plt.close(fig)
    return output_path


def make_drift_plots(csv_path: str, plots_dir: str) -> List[str]:
    df = pd.read_csv(csv_path)
    if df.empty:
        raise ValueError("CSV is empty; no drift plots can be generated.")

    output_paths: List[str] = []

    drift_grid_path = os.path.join(plots_dir, "drift_by_checkpoint.png")
    fig, axes = _make_checkpoint_grid(len(CHECKPOINT_ORDER))
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    for ax, checkpoint_name in zip(axes.flat, CHECKPOINT_ORDER):
        dfi = df[df["checkpoint"] == checkpoint_name].copy()
        dfi = dfi.sort_values(["probe_idx", "sequence_idx"])

        if not dfi.empty and np.isfinite(dfi["tip_u_mm"]).any() and np.isfinite(dfi["tip_z_mm"]).any():
            y1_label = "Δu (mm)"
            y2_label = "Δz (mm)"
            u_palette = ["#66d9ff", "#7fd6ff", "#55efc4", "#74b9ff", "#81ecec"]
            z_palette = ["#ff8fab", "#ffb86c", "#ffd166", "#fab1a0", "#f6a6ff"]
            for color_idx, (probe_idx, dfg) in enumerate(dfi.groupby("probe_idx")):
                dfg = dfg.sort_values("sequence_idx")
                u0 = float(dfg.iloc[0]["tip_u_mm"])
                z0 = float(dfg.iloc[0]["tip_z_mm"])
                du = dfg["tip_u_mm"].astype(float) - u0
                dz = dfg["tip_z_mm"].astype(float) - z0
                u_color = u_palette[color_idx % len(u_palette)]
                z_color = z_palette[color_idx % len(z_palette)]
                ax.plot(
                    dfg["sequence_idx"],
                    du,
                    color=u_color,
                    linewidth=2.2,
                    marker="o",
                    markersize=5.5,
                    label=f"Probe {probe_idx} Δu",
                )
                ax.plot(
                    dfg["sequence_idx"],
                    dz,
                    color=z_color,
                    linewidth=1.8,
                    linestyle="--",
                    alpha=0.88,
                    marker="s",
                    markersize=4.5,
                    label=f"Probe {probe_idx} Δz",
                )
        elif not dfi.empty:
            y1_label = "Δx (px)"
            y2_label = "Δy (px)"
            x_palette = ["#66d9ff", "#7fd6ff", "#55efc4", "#74b9ff", "#81ecec"]
            y_palette = ["#ff8fab", "#ffb86c", "#ffd166", "#fab1a0", "#f6a6ff"]
            for color_idx, (probe_idx, dfg) in enumerate(dfi.groupby("probe_idx")):
                dfg = dfg.sort_values("sequence_idx")
                x0 = float(dfg.iloc[0]["tip_col_px"])
                y0 = float(dfg.iloc[0]["tip_row_px"])
                dx = dfg["tip_col_px"].astype(float) - x0
                dy = dfg["tip_row_px"].astype(float) - y0
                x_color = x_palette[color_idx % len(x_palette)]
                y_color = y_palette[color_idx % len(y_palette)]
                ax.plot(
                    dfg["sequence_idx"],
                    dx,
                    color=x_color,
                    linewidth=2.2,
                    marker="o",
                    markersize=5.5,
                    label=f"Probe {probe_idx} Δx",
                )
                ax.plot(
                    dfg["sequence_idx"],
                    dy,
                    color=y_color,
                    linewidth=1.8,
                    linestyle="--",
                    alpha=0.88,
                    marker="s",
                    markersize=4.5,
                    label=f"Probe {probe_idx} Δy",
                )

        else:
            use_mm = np.isfinite(df["tip_u_mm"]).any() and np.isfinite(df["tip_z_mm"]).any()
            y1_label = "Δu (mm)" if use_mm else "Δx (px)"
            y2_label = "Δz (mm)" if use_mm else "Δy (px)"

        _apply_dark_axes_style(
            ax,
            title=checkpoint_name.replace("_", " ").title(),
            xlabel="Sequence index",
            ylabel=f"{y1_label} / {y2_label}",
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(fontsize=6.9, loc="best", frameon=True, ncol=1)
            leg.get_frame().set_facecolor("#121c28")
            leg.get_frame().set_edgecolor((0.75, 0.84, 0.93, 0.20))
            for txt in leg.get_texts():
                txt.set_color("#e5edf6")

    for ax in axes.flat[len(CHECKPOINT_ORDER):]:
        ax.set_visible(False)

    fig.suptitle(
        "Tip drift over time by checkpoint",
        color="#f7fbff",
        fontsize=15,
        weight="semibold",
        y=0.99,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    fig.savefig(drift_grid_path, dpi=200, bbox_inches="tight", transparent=True)
    plt.close(fig)
    output_paths.append(drift_grid_path)

    heatmap_path = os.path.join(plots_dir, "tip_position_heatmap.png")
    fig, ax = plt.subplots(figsize=(8.8, 7.2))
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    use_mm = np.isfinite(df["tip_u_mm"]).any() and np.isfinite(df["tip_z_mm"]).any()
    x_col = "tip_u_mm" if use_mm else "tip_col_px"
    y_col = "tip_z_mm" if use_mm else "tip_row_px"
    xlabel = "u (mm)" if use_mm else "x (px)"
    ylabel = "z (mm)" if use_mm else "y (px)"
    plot_df = df[np.isfinite(df[x_col]) & np.isfinite(df[y_col])].copy()
    if not plot_df.empty:
        for checkpoint_name, dfi in plot_df.groupby("checkpoint"):
            ax.scatter(
                dfi[x_col].astype(float),
                dfi[y_col].astype(float),
                s=40,
                color=CHECKPOINT_COLORS.get(checkpoint_name, "#d6dee8"),
                alpha=0.72,
                edgecolors="#f3f8ff",
                linewidths=0.35,
                label=checkpoint_name.replace("_", " "),
                zorder=3,
            )

    _apply_dark_axes_style(
        ax,
        title="Tracked tip positions",
        xlabel=xlabel,
        ylabel=ylabel,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(loc="best", frameon=True, fontsize=9)
        leg.get_frame().set_facecolor("#121c28")
        leg.get_frame().set_edgecolor((0.75, 0.84, 0.93, 0.18))
        for txt in leg.get_texts():
            txt.set_color("#e8f0f8")
    fig.tight_layout()
    fig.savefig(heatmap_path, dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig)
    output_paths.append(heatmap_path)

    four_panel_heatmap_path = os.path.join(plots_dir, "tip_position_heatmap_by_checkpoint.png")
    fig, axes = _make_checkpoint_grid(len(CHECKPOINT_ORDER))
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    for ax, checkpoint_name in zip(axes.flat, CHECKPOINT_ORDER):
        checkpoint_df = plot_df[plot_df["checkpoint"] == checkpoint_name].copy()
        if not checkpoint_df.empty:
            x_vals = checkpoint_df[x_col].astype(float).to_numpy()
            y_vals = checkpoint_df[y_col].astype(float).to_numpy()
            hb = ax.hexbin(
                x_vals,
                y_vals,
                gridsize=22,
                mincnt=1,
                cmap=_make_dark_density_cmap(),
                linewidths=0.2,
                edgecolors=(1.0, 1.0, 1.0, 0.12),
            )
            ax.scatter(
                x_vals,
                y_vals,
                s=26,
                color=CHECKPOINT_COLORS.get(checkpoint_name, "#d6dee8"),
                alpha=0.28,
                edgecolors="none",
                zorder=3,
                label=checkpoint_name.replace("_", " "),
            )
            x_span = float(np.ptp(x_vals))
            y_span = float(np.ptp(y_vals))
            x_pad = max(0.15, 0.12 * max(x_span, 1.0))
            y_pad = max(0.15, 0.12 * max(y_span, 1.0))
            ax.set_xlim(float(np.min(x_vals) - x_pad), float(np.max(x_vals) + x_pad))
            ax.set_ylim(float(np.min(y_vals) - y_pad), float(np.max(y_vals) + y_pad))
            cbar = fig.colorbar(hb, ax=ax, pad=0.02, shrink=0.88)
            cbar.set_label("Density", color="#d7e2ee")
            cbar.ax.yaxis.set_tick_params(color="#c8d5e3")
            plt.setp(cbar.ax.get_yticklabels(), color="#c8d5e3")
            cbar.outline.set_edgecolor((0.75, 0.84, 0.93, 0.25))

        _apply_dark_axes_style(
            ax,
            title=checkpoint_name.replace("_", " ").title(),
            xlabel=xlabel,
            ylabel=ylabel,
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(loc="best", frameon=True, fontsize=8.5)
            leg.get_frame().set_facecolor("#121c28")
            leg.get_frame().set_edgecolor((0.75, 0.84, 0.93, 0.18))
            for txt in leg.get_texts():
                txt.set_color("#e8f0f8")

    for ax in axes.flat[len(CHECKPOINT_ORDER):]:
        ax.set_visible(False)

    fig.suptitle(
        "Tracked tip position density by checkpoint",
        color="#f7fbff",
        fontsize=15,
        weight="semibold",
        y=0.98,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    fig.savefig(four_panel_heatmap_path, dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig)
    output_paths.append(four_panel_heatmap_path)

    density_gif_path = os.path.join(plots_dir, "tip_position_heatmap_by_checkpoint.gif")
    saved_gif = _save_density_animation_gif(
        plot_df=plot_df,
        x_col=x_col,
        y_col=y_col,
        xlabel=xlabel,
        ylabel=ylabel,
        output_path=density_gif_path,
    )
    if saved_gif:
        output_paths.append(saved_gif)

    error_hist_path = os.path.join(plots_dir, "tip_error_histograms_by_checkpoint.png")
    hist_rows = int(math.ceil(len(CHECKPOINT_ORDER) / 2.0))
    fig = plt.figure(figsize=(15.2, 4.8 * hist_rows))
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    outer_gs = fig.add_gridspec(hist_rows, 2, hspace=0.28, wspace=0.18)

    use_mm = np.isfinite(df["tip_u_mm"]).any() and np.isfinite(df["tip_z_mm"]).any()
    x_col = "tip_u_mm" if use_mm else "tip_col_px"
    y_col = "tip_z_mm" if use_mm else "tip_row_px"
    u_label = "u error (mm)" if use_mm else "x error (px)"
    z_label = "z error (mm)" if use_mm else "y error (px)"
    u_color = "#66d9ff"
    z_color = "#ff8fab"

    for outer_idx, checkpoint_name in enumerate(CHECKPOINT_ORDER):
        row_idx = outer_idx // 2
        col_idx = outer_idx % 2
        inner_gs = outer_gs[row_idx, col_idx].subgridspec(1, 2, wspace=0.20)
        ax_u = fig.add_subplot(inner_gs[0, 0])
        ax_z = fig.add_subplot(inner_gs[0, 1])

        dfi = df[df["checkpoint"] == checkpoint_name].copy()
        dfi = dfi.sort_values(["probe_idx", "sequence_idx"])
        delta_u_parts = []
        delta_z_parts = []

        for _probe_idx, dfg in dfi.groupby("probe_idx"):
            dfg = dfg.sort_values("sequence_idx")
            if dfg.empty:
                continue
            x_vals = pd.to_numeric(dfg[x_col], errors="coerce").to_numpy(dtype=float)
            y_vals = pd.to_numeric(dfg[y_col], errors="coerce").to_numpy(dtype=float)
            finite_x = np.isfinite(x_vals)
            finite_y = np.isfinite(y_vals)
            if np.any(finite_x):
                delta_u_parts.append(x_vals[finite_x] - x_vals[finite_x][0])
            if np.any(finite_y):
                delta_z_parts.append(y_vals[finite_y] - y_vals[finite_y][0])

        delta_u = np.concatenate(delta_u_parts) if delta_u_parts else np.asarray([], dtype=float)
        delta_z = np.concatenate(delta_z_parts) if delta_z_parts else np.asarray([], dtype=float)

        if delta_u.size > 0:
            ax_u.hist(
                delta_u,
                bins=max(8, min(30, int(np.sqrt(delta_u.size) * 2))),
                color=(0.40, 0.85, 1.0, 0.82),
                edgecolor=(0.93, 0.97, 1.0, 0.95),
                linewidth=0.9,
            )
            ax_u.axvline(0.0, color="#eaf6ff", linestyle="--", linewidth=1.1, alpha=0.85)
        if delta_z.size > 0:
            ax_z.hist(
                delta_z,
                bins=max(8, min(30, int(np.sqrt(delta_z.size) * 2))),
                color=(1.0, 0.56, 0.67, 0.82),
                edgecolor=(1.0, 0.95, 0.97, 0.95),
                linewidth=0.9,
            )
            ax_z.axvline(0.0, color="#fff0f5", linestyle="--", linewidth=1.1, alpha=0.85)

        _apply_dark_axes_style(
            ax_u,
            title=f"{checkpoint_name.replace('_', ' ').title()} | {u_label}",
            xlabel=u_label,
            ylabel="Count",
        )
        _apply_dark_axes_style(
            ax_z,
            title=f"{checkpoint_name.replace('_', ' ').title()} | {z_label}",
            xlabel=z_label,
            ylabel="Count",
        )

    total_hist_slots = hist_rows * 2
    for empty_idx in range(len(CHECKPOINT_ORDER), total_hist_slots):
        row_idx = empty_idx // 2
        col_idx = empty_idx % 2
        empty_ax = fig.add_subplot(outer_gs[row_idx, col_idx])
        empty_ax.set_visible(False)

    fig.suptitle(
        "Tip error histograms by checkpoint",
        color="#f7fbff",
        fontsize=15,
        weight="semibold",
        y=0.985,
    )
    fig.tight_layout(rect=[0.0, 0.0, 1.0, 0.965])
    fig.savefig(error_hist_path, dpi=220, bbox_inches="tight", transparent=True)
    plt.close(fig)
    output_paths.append(error_hist_path)

    return output_paths


def prepare_calibration_object() -> CTR_Shadow_Calibration:
    print(f"[DEBUG] SCRIPT_DIR={SCRIPT_DIR}")
    import shadow_calibration
    print(f"[DEBUG] shadow_calibration loaded from: {inspect.getsourcefile(shadow_calibration)}")

    cal = CTR_Shadow_Calibration(
        parent_directory=SCRIPT_DIR,
        project_name=PROJECT_NAME,
        allow_existing=ALLOW_EXISTING_PROJECT,
        add_date=ADD_DATE_TO_PROJECT_FOLDER,
    )
    print("Calibration object created!")

    if os.path.isfile(CAMERA_CALIBRATION_FILE):
        cal.load_camera_calibration(CAMERA_CALIBRATION_FILE)
        if BOARD_REFERENCE_IMAGE is not None:
            cal.estimate_board_reference_from_image(
                BOARD_REFERENCE_IMAGE,
                draw_debug=True,
                save_debug_path=os.path.join(SCRIPT_DIR, "captures", "checkerboard_reference_debug.png"),
            )
    else:
        print(f"[WARN] Camera calibration file not found, continuing without it: {CAMERA_CALIBRATION_FILE}")

    cal.connect_to_camera(cam_port=CAMERA_PORT, show_preview=SHOW_CAMERA_PREVIEW)
    if USE_CLASS_ANALYSIS_CROP_SETUP:
        cal.setup_analysis_crop(enable_manual_adjustment=MANUAL_CROP_ADJUSTMENT)
    cal.connect_to_robot()
    return cal


def run_drift_experiment(cal: CTR_Shadow_Calibration) -> Dict[str, str]:
    if cal.cam is None:
        raise RuntimeError("Camera is not connected.")
    if cal.rrf is None:
        raise RuntimeError("Robot is not connected.")

    checkpoint_b = compute_checkpoint_positions(PULL_B_START, PULL_B_STEPS, PULL_B_STEP_SIZE)
    print(f"[INFO] Checkpoint B positions: {checkpoint_b}")

    project_dir = cal.calibration_data_folder
    if RESET_EXISTING_OUTPUTS:
        raw_dir = reset_dir(os.path.join(project_dir, "raw_image_data_folder"))
        processed_dir = reset_dir(os.path.join(project_dir, "processed_image_data_folder"))
        print("[INFO] Existing raw/processed outputs were cleared before the run.")
    else:
        raw_dir = ensure_dir(os.path.join(project_dir, "raw_image_data_folder"))
        processed_dir = ensure_dir(os.path.join(project_dir, "processed_image_data_folder"))
    plots_dir = ensure_dir(os.path.join(processed_dir, "drift_plots"))
    annotated_dir = ensure_dir(os.path.join(processed_dir, "annotated_outputs"))
    csv_path = os.path.join(processed_dir, "tip_drift_measurements.csv")

    current_pos = {
        ROBOT_FRONT_AXIS_NAME: 0.0,
        ROBOT_STAGE_Y_AXIS_NAME: 0.0,
        ROBOT_STAGE_Z_AXIS_NAME: 0.0,
        ROBOT_REAR_AXIS_NAME: 0.0,
    }

    # Home the pull axis to the configured start before the first sequence.
    current_pos = move_abs(cal, current_pos, JOGGING_FEEDRATE, **{ROBOT_REAR_AXIS_NAME: PULL_B_START})

    sequence_start_time = time.time()

    for probe_idx, (probe_x, probe_y, probe_z) in enumerate(PROBE_POINTS, start=1):
        print("\n" + "=" * 80)
        print(f"[PROBE] {probe_idx}/{len(PROBE_POINTS)} at X={probe_x}, Y={probe_y}, Z={probe_z}")
        print("=" * 80)

        current_pos = move_abs(
            cal,
            current_pos,
            JOGGING_FEEDRATE,
            **{
                ROBOT_FRONT_AXIS_NAME: probe_x,
                ROBOT_STAGE_Y_AXIS_NAME: probe_y,
                ROBOT_STAGE_Z_AXIS_NAME: probe_z,
            },
        )
        current_pos = move_abs(cal, current_pos, JOGGING_FEEDRATE, **{ROBOT_REAR_AXIS_NAME: PULL_B_START})

        for sequence_idx in range(1, NUM_SEQUENCES + 1):
            print(f"\n[SEQ] Probe {probe_idx} | sequence {sequence_idx}/{NUM_SEQUENCES}")

            capture_plan = [(checkpoint_name, checkpoint_b[checkpoint_name]) for checkpoint_name in CHECKPOINT_ORDER]

            for checkpoint_name, b_target in capture_plan:
                frame_bgr = None
                for attempt_idx in range(1, CAPTURE_RETRY_LIMIT + 1):
                    current_pos = move_abs(
                        cal,
                        current_pos,
                        0.30 * JOGGING_FEEDRATE,
                        **{ROBOT_REAR_AXIS_NAME: b_target},
                    )
                    wait_for_duet_motion_complete(cal, extra_settle=0.0)
                    time.sleep(float(max(0.0, CAPTURE_DWELL_S)))

                    try:
                        frame_bgr = capture_frame(cal.cam, warmup_frames=CAMERA_WARMUP_FRAMES)
                        break
                    except RuntimeError as exc:
                        print(
                            f"  [WARN] Capture failed at {checkpoint_name} "
                            f"(attempt {attempt_idx}/{CAPTURE_RETRY_LIMIT}): {exc}"
                        )
                        if attempt_idx >= CAPTURE_RETRY_LIMIT:
                            raise RuntimeError(
                                f"Camera capture failed at checkpoint '{checkpoint_name}' "
                                f"after {CAPTURE_RETRY_LIMIT} attempts."
                            ) from exc
                        time.sleep(float(max(0.0, CAPTURE_RETRY_WAIT_S)))

                if frame_bgr is None:
                    raise RuntimeError(f"Capture did not succeed at checkpoint '{checkpoint_name}'.")

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                image_name = (
                    f"probe{probe_idx:02d}_seq{sequence_idx:04d}_{checkpoint_name}_"
                    f"X{probe_x:.2f}_Y{probe_y:.2f}_Z{probe_z:.2f}_B{b_target:.2f}_{timestamp}"
                    f"{RAW_IMAGE_EXTENSION}"
                )
                image_path = os.path.join(raw_dir, image_name)
                save_compressed_image(frame_bgr, image_path)

                analysis = analyze_frame_with_ctr_class(cal, frame_bgr, THRESHOLD)
                annotated_frame = build_annotated_frame(
                    frame_bgr,
                    analysis,
                    probe_idx=probe_idx,
                    sequence_idx=sequence_idx,
                    checkpoint_name=checkpoint_name,
                    b_target=b_target,
                )
                annotated_name = f"{os.path.splitext(image_name)[0]}_annotated{ANNOTATED_IMAGE_EXTENSION}"
                annotated_path = os.path.join(annotated_dir, annotated_name)
                save_compressed_image(
                    annotated_frame,
                    annotated_path,
                    jpeg_quality=ANNOTATED_IMAGE_JPEG_QUALITY,
                )

                row = {
                    "timestamp": datetime.now().isoformat(timespec="milliseconds"),
                    "probe_idx": probe_idx,
                    "probe_x": float(probe_x),
                    "probe_y": float(probe_y),
                    "probe_z": float(probe_z),
                    "sequence_idx": sequence_idx,
                    "checkpoint": checkpoint_name,
                    "b_target": float(b_target),
                    "tip_row_px": analysis["tip_row_px"],
                    "tip_col_px": analysis["tip_col_px"],
                    "tip_u_mm": analysis["tip_u_mm"],
                    "tip_z_mm": analysis["tip_z_mm"],
                    "tip_angle_deg": analysis["tip_angle_deg"],
                    "location_units": analysis["location_units"],
                    "image_file": image_name,
                }
                append_result_row(csv_path, row)
                print(
                    f"  [CAPTURE] {checkpoint_name} | B={b_target:.3f} | "
                    f"tip(px)=({analysis['tip_col_px']:.2f}, {analysis['tip_row_px']:.2f}) | "
                    f"tip(mm)=({analysis['tip_u_mm']:.3f}, {analysis['tip_z_mm']:.3f}) | "
                    f"angle={analysis['tip_angle_deg']:.3f}"
                )

    elapsed_s = time.time() - sequence_start_time
    print(f"\n[INFO] Acquisition + online analysis complete in {elapsed_s / 60.0:.2f} min")

    drift_plot_paths = make_drift_plots(csv_path, plots_dir)
    archive_path = archive_raw_images(raw_dir) if ARCHIVE_RAW_IMAGE_FOLDER_AT_END else None

    summary = {
        "project_dir": project_dir,
        "raw_dir": raw_dir,
        "processed_dir": processed_dir,
        "csv_path": csv_path,
        "plots_dir": plots_dir,
        "annotated_dir": annotated_dir,
        "archive_path": archive_path or "",
    }

    print("\n[INFO] Output summary")
    for key, value in summary.items():
        if value:
            print(f"  - {key}: {value}")
    if drift_plot_paths:
        print("  - drift plots:")
        for p in drift_plot_paths:
            print(f"      {p}")

    return summary


def main() -> None:
    cal = prepare_calibration_object()
    try:
        run_drift_experiment(cal)
    finally:
        try:
            if cal.cam is not None:
                cal.disconnect_camera()
        except Exception as exc:
            print(f"[WARN] Camera disconnect failed: {exc}")


if __name__ == "__main__":
    main()
