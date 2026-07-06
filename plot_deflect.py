#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from shadow_calibration import (
    CTR_Shadow_Calibration,
    _normalize_tip_angle_deg,
    _normalize_tip_detection_mode,
    _remap_zoom_axes_to_crop_coordinates,
    _select_tip_candidate,
    annotate_tip_geometry_on_axes,
    refine_tip_parallel_centerline,
)


# =========================
# DEFAULT USER CONFIGURATION
# =========================
ALLOW_EXISTING_PROJECT = True
ADD_DATE_TO_PROJECT_FOLDER = False

MANUAL_CROP_ADJUSTMENT = True
THRESHOLD = 200
USE_EXACT_CLASS_THRESHOLDING = True

# Optional calibration assets for mm output
CAMERA_CALIBRATION_FILE = os.path.join(
    SCRIPT_DIR, "captures", "calibration_webcam_20260406_104136.npz"
)
BOARD_REFERENCE_IMAGE = os.path.join(
    SCRIPT_DIR, "captures", "photo_20260624_150531.png"
)
DEFAULT_TIP_REFINER_MODEL = os.path.join(
    SCRIPT_DIR,
    "CNN_calib_2",
    "processed_image_data_folder",
    "tip_refinement_model",
    "best_tip_refiner.pt",
)
DEFAULT_TIP_REFINER_ANCHOR = None
DEFAULT_TIP_REFINER_COMPARE_ONLY = False
DEFAULT_TIP_REFINE_MODE = "coarse"
DEFAULT_TIP_DETECTION_MODE = "classical"

RUN_OUTPUT_SUBDIR = "per_run"
RUN_RESULTS_CSV_NAME = "tip_tracking_results.csv"
RUN_SUMMARY_CSV_NAME = "run_summary.csv"
ALL_RESULTS_CSV_NAME = "all_tip_tracking_results.csv"
SPEED_SUMMARY_CSV_NAME = "speed_summary_by_dataset_and_run_type.csv"
SUMMARY_PLOT_DIRNAME = "summary_plots"

ANNOTATED_IMAGE_EXTENSION = ".jpg"
ANNOTATED_IMAGE_JPEG_QUALITY = 90

PHASE_COLORS = {
    "pre": "#e5e7eb",
    "accel": "#4cc9f0",
    "ss": "#80ed99",
    "decel": "#f28482",
    "pause": "#f9c74f",
    "post": "#f8fafc",
    "unknown": "#cbd5e1",
}
DATASET_COLORS = {
    "forward": "#60a5fa",
    "backward": "#f59e0b",
    "transverse": "#34d399",
}
FIXED_B_SERIES = {
    "fixed_b_0": {"curl_angle_deg": 0.0, "label": "0 deg", "color": "#3d70b2"},
    "fixed_b_m3p2": {"curl_angle_deg": 90.0, "label": "90 deg", "color": "#c83f3f"},
    "fixed_b_m5p1": {"curl_angle_deg": 180.0, "label": "180 deg", "color": "#3aa97a"},
}
VALID_IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


# =========================
# DATA TYPES
# =========================
@dataclass
class RunFolderInfo:
    folder_path: str
    folder_name: str
    capture_log_path: str
    speed_mm_min: float
    direction: str
    dataset_label: str
    orientation_name: str
    orientation_deg: float
    run_type: str
    repeat_idx: int
    repeat_count: int
    daq_mode: str = "speed_series"
    attack_angle_deg: float = float("nan")
    b_command_mm: float = float("nan")


# =========================
# HELPERS
# =========================
def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def natural_key(text: str) -> List[object]:
    return [int(tok) if tok.isdigit() else tok.lower() for tok in re.split(r"(\d+)", text)]


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


def set_science_light_style(font: str = "Arial") -> None:
    mpl.rcParams.update(
        {
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.03,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "font.family": "sans-serif",
            "font.sans-serif": [font, "Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.titlesize": 8.8,
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.6,
            "ytick.labelsize": 7.6,
            "legend.fontsize": 7.1,
            "axes.linewidth": 0.8,
            "axes.edgecolor": "#222222",
            "axes.labelcolor": "#222222",
            "xtick.color": "#222222",
            "ytick.color": "#222222",
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.width": 0.75,
            "ytick.major.width": 0.75,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
        }
    )


def _apply_light_axes_style(ax, title: str, xlabel: str, ylabel: str) -> None:
    ax.set_title(title, color="#222222", fontsize=9.0, pad=8, weight="semibold")
    ax.set_xlabel(xlabel, color="#222222")
    ax.set_ylabel(ylabel, color="#222222")
    ax.tick_params(colors="#222222", labelsize=7.6)
    for spine in ax.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(0.8)
    ax.grid(True, color="#d8d8d8", linewidth=0.6, alpha=0.45)
    ax.set_facecolor("white")


def _style_axes(ax, theme: str, title: str, xlabel: str, ylabel: str) -> None:
    if theme == "dark":
        _apply_dark_axes_style(ax, title, xlabel, ylabel)
    else:
        _apply_light_axes_style(ax, title, xlabel, ylabel)


def _style_legend(leg, theme: str) -> None:
    if leg is None:
        return
    if theme == "dark":
        leg.get_frame().set_facecolor("#121c28")
        leg.get_frame().set_edgecolor((0.75, 0.84, 0.93, 0.18))
        for txt in leg.get_texts():
            txt.set_color("#e8f0f8")
    else:
        leg.get_frame().set_facecolor("white")
        leg.get_frame().set_edgecolor("#d8d8d8")


def _save_plot_outputs(fig, out_base: str, theme: str) -> None:
    fig.savefig(f"{out_base}.png", dpi=220, bbox_inches="tight", transparent=(theme == "dark"))
    if theme == "light":
        fig.savefig(f"{out_base}.pdf", bbox_inches="tight")
        fig.savefig(f"{out_base}.svg", bbox_inches="tight")


def parse_folder_speed_and_direction(folder_name: str) -> Tuple[Optional[float], Optional[str]]:
    direction = None
    if folder_name.endswith("_forward"):
        direction = "forward"
    elif folder_name.endswith("_backward"):
        direction = "backward"

    speed = None
    speed_match = re.search(r"speed_([0-9]+(?:p[0-9]+)?)_mm_min", folder_name)
    if speed_match:
        speed_str = speed_match.group(1).replace("p", ".")
        try:
            speed = float(speed_str)
        except ValueError:
            speed = None

    return speed, direction


def parse_folder_metadata(folder_name: str) -> Dict[str, object]:
    speed, direction = parse_folder_speed_and_direction(folder_name)

    run_type = None
    if "_two_stage_" in folder_name or folder_name.endswith("_two_stage"):
        run_type = "two_stage"
    elif "_direct_" in folder_name or folder_name.endswith("_direct"):
        run_type = "direct"

    dataset_label = None
    for candidate in ("forward", "backward", "transverse"):
        if f"_{candidate}_" in folder_name or folder_name.endswith(f"_{candidate}"):
            dataset_label = candidate
            break

    orientation_name = None
    orientation_deg = float("nan")
    orientation_match = re.search(r"(c\d{3})", folder_name)
    if orientation_match:
        orientation_name = orientation_match.group(1).lower()
        try:
            orientation_deg = float(int(orientation_name[1:]))
        except Exception:
            orientation_deg = float("nan")

    attack_angle_deg = float("nan")
    attack_match = re.search(r"attack_([mp0-9]+(?:p[0-9]+)?)deg", folder_name)
    if attack_match:
        try:
            attack_angle_deg = float(attack_match.group(1).replace("m", "-").replace("p", "."))
        except Exception:
            attack_angle_deg = float("nan")

    b_command_mm = float("nan")
    b_match = re.search(r"bcmd_([mp0-9]+(?:p[0-9]+)?)", folder_name)
    if b_match:
        try:
            b_command_mm = float(b_match.group(1).replace("m", "-").replace("p", "."))
        except Exception:
            b_command_mm = float("nan")

    daq_mode = "b_attack_sweep" if "_b_attack_" in folder_name else "speed_series"

    repeat_idx = 1
    repeat_count = 1
    repeat_match = re.search(r"_pass_(\d+)_of_(\d+)", folder_name)
    if repeat_match:
        try:
            repeat_idx = int(repeat_match.group(1))
            repeat_count = int(repeat_match.group(2))
        except Exception:
            repeat_idx = 1
            repeat_count = 1

    return {
        "speed_mm_min": speed,
        "direction": direction,
        "run_type": run_type,
        "dataset_label": dataset_label,
        "orientation_name": orientation_name,
        "orientation_deg": orientation_deg,
        "repeat_idx": repeat_idx,
        "repeat_count": repeat_count,
        "daq_mode": daq_mode,
        "attack_angle_deg": attack_angle_deg,
        "b_command_mm": b_command_mm,
    }


def save_json(path: str, payload: Dict[str, object]) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_json(path: str) -> Dict[str, object]:
    with open(path, "r") as f:
        return json.load(f)


def resolve_raw_root(project_dir: Optional[str], raw_root: Optional[str]) -> str:
    if (project_dir is None and raw_root is None) or (project_dir is not None and raw_root is not None):
        raise ValueError("Use either --project-dir OR --raw-root")

    if project_dir is not None:
        candidate = os.path.join(os.path.abspath(project_dir), "raw_image_data_folder")
    else:
        candidate = os.path.abspath(raw_root)

    if not os.path.isdir(candidate):
        raise FileNotFoundError(f"raw_image_data_folder not found: {candidate}")
    return candidate


def infer_output_root(project_dir: Optional[str], raw_root: str, output_dir: Optional[str]) -> str:
    if output_dir is not None:
        return ensure_dir(os.path.abspath(output_dir))

    if project_dir is not None:
        return ensure_dir(
            os.path.join(
                os.path.abspath(project_dir),
                "processed_image_data_folder",
                "tip_tracking_analysis",
            )
        )

    raw_root_abs = os.path.abspath(raw_root)
    project_parent = os.path.dirname(raw_root_abs)
    return ensure_dir(
        os.path.join(project_parent, "processed_image_data_folder", "tip_tracking_analysis")
    )


def themed_summary_plot_dir(output_root: str, theme: str) -> str:
    theme = str(theme).strip().lower()
    folder = f"{SUMMARY_PLOT_DIRNAME}_{theme}" if theme in {"light", "dark"} else SUMMARY_PLOT_DIRNAME
    return ensure_dir(os.path.join(output_root, folder))


def roi_to_analysis_crop(roi_xywh: Tuple[int, int, int, int], image_shape: Tuple[int, ...]) -> Dict[str, int]:
    x, y, w, h = [int(v) for v in roi_xywh]
    img_h, img_w = image_shape[:2]

    if w <= 0 or h <= 0:
        raise ValueError("Invalid ROI with non-positive width/height")

    crop_width_min = max(0, x)
    crop_width_max = min(img_w, x + w)

    # convert top-origin ROI to the class crop format used in the provided script
    crop_height_min = max(0, img_h - (y + h))
    crop_height_max = min(img_h, img_h - y)

    return {
        "crop_width_min": int(crop_width_min),
        "crop_width_max": int(crop_width_max),
        "crop_height_min": int(crop_height_min),
        "crop_height_max": int(crop_height_max),
    }


def select_roi_interactively(image_bgr: np.ndarray, window_name: str = "Select analysis crop") -> Tuple[int, int, int, int]:
    display = image_bgr.copy()
    roi = cv2.selectROI(window_name, display, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow(window_name)
    if roi is None or len(roi) != 4 or roi[2] <= 0 or roi[3] <= 0:
        raise RuntimeError("No valid ROI selected.")
    return tuple(int(v) for v in roi)


def get_first_image_for_crop(run_infos: Sequence[RunFolderInfo]) -> str:
    for run_info in run_infos:
        log_df = pd.read_csv(run_info.capture_log_path)
        if "frame_idx" in log_df.columns:
            log_df = log_df.sort_values("frame_idx")
        for _, row in log_df.iterrows():
            image_file = str(row.get("image_file", "")).strip()
            if not image_file:
                continue
            image_path = os.path.join(run_info.folder_path, image_file)
            if os.path.isfile(image_path):
                return image_path
    raise FileNotFoundError("Could not find any image in any run folder.")


def load_or_create_analysis_crop(
    run_infos: Sequence[RunFolderInfo],
    output_root: str,
    crop_json: Optional[str],
    crop_xywh: Optional[Sequence[int]],
    manual_crop: bool,
) -> Dict[str, int]:
    if crop_json is not None:
        crop_data = load_json(os.path.abspath(crop_json))
        required = {"crop_width_min", "crop_width_max", "crop_height_min", "crop_height_max"}
        missing = required - set(crop_data.keys())
        if missing:
            raise ValueError(f"Crop JSON missing keys: {sorted(missing)}")
        return {k: int(crop_data[k]) for k in required}

    first_image_path = get_first_image_for_crop(run_infos)
    first_image = cv2.imread(first_image_path)
    if first_image is None:
        raise RuntimeError(f"Could not read first crop image: {first_image_path}")

    if crop_xywh is not None:
        if len(crop_xywh) != 4:
            raise ValueError("--crop-xywh must have 4 integers: x y w h")
        roi = tuple(int(v) for v in crop_xywh)
    elif manual_crop:
        print(f"[INFO] Select crop ROI using the first image:\n  {first_image_path}")
        roi = select_roi_interactively(first_image)
    else:
        raise ValueError("No crop provided. Use --manual-crop, --crop-json, or --crop-xywh.")

    crop_data = roi_to_analysis_crop(roi, first_image.shape)
    save_json(os.path.join(output_root, "analysis_crop.json"), crop_data)
    return crop_data


def find_run_folders(raw_root: str) -> List[RunFolderInfo]:
    run_infos: List[RunFolderInfo] = []

    for entry in os.scandir(raw_root):
        if not entry.is_dir():
            continue

        folder_path = os.path.abspath(entry.path)
        folder_name = os.path.basename(folder_path)
        capture_log_path = os.path.join(folder_path, "capture_log.csv")
        if not os.path.isfile(capture_log_path):
            continue

        folder_meta = parse_folder_metadata(folder_name)

        try:
            log_df = pd.read_csv(capture_log_path)
        except Exception as exc:
            print(f"[WARN] Could not read {capture_log_path}: {exc}")
            continue

        if log_df.empty:
            print(f"[WARN] Empty capture log: {capture_log_path}")
            continue

        speed = folder_meta["speed_mm_min"]
        direction = folder_meta["direction"]
        dataset_label = folder_meta["dataset_label"]
        orientation_name = folder_meta["orientation_name"]
        orientation_deg = folder_meta["orientation_deg"]
        run_type = folder_meta["run_type"]
        repeat_idx = int(folder_meta.get("repeat_idx", 1))
        repeat_count = int(folder_meta.get("repeat_count", 1))
        daq_mode = str(folder_meta.get("daq_mode", "speed_series"))
        attack_angle_deg = float(folder_meta.get("attack_angle_deg", float("nan")))
        b_command_mm = float(folder_meta.get("b_command_mm", float("nan")))

        if "speed_mm_min" in log_df.columns and pd.notna(log_df["speed_mm_min"].iloc[0]):
            try:
                speed = float(log_df["speed_mm_min"].iloc[0])
            except Exception:
                pass

        if "direction" in log_df.columns and pd.notna(log_df["direction"].iloc[0]):
            direction = str(log_df["direction"].iloc[0]).strip().lower()

        if "dataset_label" in log_df.columns and pd.notna(log_df["dataset_label"].iloc[0]):
            dataset_label = str(log_df["dataset_label"].iloc[0]).strip().lower()

        if "orientation_name" in log_df.columns and pd.notna(log_df["orientation_name"].iloc[0]):
            orientation_name = str(log_df["orientation_name"].iloc[0]).strip().lower()

        if "orientation_deg" in log_df.columns and pd.notna(log_df["orientation_deg"].iloc[0]):
            try:
                orientation_deg = float(log_df["orientation_deg"].iloc[0])
            except Exception:
                pass

        if "run_type" in log_df.columns and pd.notna(log_df["run_type"].iloc[0]):
            run_type = str(log_df["run_type"].iloc[0]).strip().lower()
        if "repeat_idx" in log_df.columns and pd.notna(log_df["repeat_idx"].iloc[0]):
            try:
                repeat_idx = int(log_df["repeat_idx"].iloc[0])
            except Exception:
                pass
        if "repeat_count" in log_df.columns and pd.notna(log_df["repeat_count"].iloc[0]):
            try:
                repeat_count = int(log_df["repeat_count"].iloc[0])
            except Exception:
                pass
        if "daq_mode" in log_df.columns and pd.notna(log_df["daq_mode"].iloc[0]):
            daq_mode = str(log_df["daq_mode"].iloc[0]).strip().lower()
        if "attack_angle_deg" in log_df.columns and pd.notna(log_df["attack_angle_deg"].iloc[0]):
            try:
                attack_angle_deg = float(log_df["attack_angle_deg"].iloc[0])
            except Exception:
                pass
        if "b_command_mm" in log_df.columns and pd.notna(log_df["b_command_mm"].iloc[0]):
            try:
                b_command_mm = float(log_df["b_command_mm"].iloc[0])
            except Exception:
                pass

        if dataset_label is None:
            dataset_label = direction if direction is not None else "unknown"
        if orientation_name is None:
            orientation_name = "unknown"
        if run_type is None:
            run_type = "direct"
        if direction is None:
            direction = "unknown"

        if speed is None:
            print(f"[WARN] Skipping folder with unresolved speed: {folder_path}")
            continue

        run_infos.append(
            RunFolderInfo(
                folder_path=folder_path,
                folder_name=folder_name,
                capture_log_path=capture_log_path,
                speed_mm_min=float(speed),
                direction=direction,
                dataset_label=dataset_label,
                orientation_name=orientation_name,
                orientation_deg=float(orientation_deg) if pd.notna(orientation_deg) else float("nan"),
                run_type=run_type,
                repeat_idx=int(repeat_idx),
                repeat_count=int(repeat_count),
                daq_mode=str(daq_mode),
                attack_angle_deg=float(attack_angle_deg) if pd.notna(attack_angle_deg) else float("nan"),
                b_command_mm=float(b_command_mm) if pd.notna(b_command_mm) else float("nan"),
            )
        )

    run_type_order = {"direct": 0, "two_stage": 1}
    dataset_order = {"forward": 0, "backward": 1, "transverse": 2}
    run_infos.sort(
        key=lambda r: (
            9999.0 if not np.isfinite(r.orientation_deg) else float(r.orientation_deg),
            dataset_order.get(r.dataset_label, 99),
            r.speed_mm_min,
            run_type_order.get(r.run_type, 99),
            9999.0 if not np.isfinite(r.attack_angle_deg) else float(r.attack_angle_deg),
            r.repeat_idx,
            natural_key(r.folder_name),
        )
    )
    return run_infos


def read_capture_log(capture_log_path: str) -> pd.DataFrame:
    df = pd.read_csv(capture_log_path)
    if df.empty:
        raise ValueError(f"Empty capture log: {capture_log_path}")

    if "image_file" not in df.columns:
        raise ValueError(f"{capture_log_path} is missing required column: image_file")

    if "frame_idx" in df.columns:
        df["frame_idx"] = pd.to_numeric(df["frame_idx"], errors="coerce")
        df = df.sort_values("frame_idx", na_position="last")
    else:
        df = df.copy()
        df["frame_idx"] = np.arange(1, len(df) + 1)

    if "elapsed_s_from_motion_start" not in df.columns:
        df["elapsed_s_from_motion_start"] = np.arange(len(df), dtype=float)
    if "elapsed_s_from_run_start" not in df.columns:
        df["elapsed_s_from_run_start"] = pd.to_numeric(df["elapsed_s_from_motion_start"], errors="coerce")
    if "elapsed_s_from_segment_start" not in df.columns:
        df["elapsed_s_from_segment_start"] = np.nan

    if "phase" not in df.columns:
        df["phase"] = "unknown"

    if "sample_type" not in df.columns:
        phase_norm = df["phase"].astype(str).str.lower()
        df["sample_type"] = np.select(
            [phase_norm == "pre", phase_norm == "post"],
            ["idle_before_motion", "idle_after_motion"],
            default="motion",
        )

    if "direction" not in df.columns:
        df["direction"] = "unknown"
    if "dataset_label" not in df.columns:
        df["dataset_label"] = df["direction"]
    if "orientation_name" not in df.columns:
        df["orientation_name"] = "unknown"
    if "orientation_deg" not in df.columns:
        df["orientation_deg"] = np.nan
    if "run_type" not in df.columns:
        df["run_type"] = "direct"
    if "repeat_idx" not in df.columns:
        df["repeat_idx"] = 1
    if "repeat_count" not in df.columns:
        df["repeat_count"] = 1
    if "segment_name" not in df.columns:
        df["segment_name"] = "unknown"
    if "segment_index" not in df.columns:
        df["segment_index"] = np.nan

    if "speed_mm_min" not in df.columns:
        df["speed_mm_min"] = np.nan
    if "daq_mode" not in df.columns:
        df["daq_mode"] = "speed_series"
    if "attack_angle_deg" not in df.columns:
        df["attack_angle_deg"] = np.nan
    if "b_command_mm" not in df.columns:
        df["b_command_mm"] = np.nan

    return df.reset_index(drop=True)


def save_compressed_image(
    image_bgr: np.ndarray,
    output_path: str,
    jpeg_quality: Optional[int] = None,
) -> None:
    ext = os.path.splitext(output_path)[1].lower()
    if ext in (".jpg", ".jpeg"):
        quality = ANNOTATED_IMAGE_JPEG_QUALITY if jpeg_quality is None else int(jpeg_quality)
        params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)]
    elif ext == ".png":
        params = [cv2.IMWRITE_PNG_COMPRESSION, 9]
    else:
        params = []

    ok = cv2.imwrite(output_path, image_bgr, params)
    if not ok:
        raise RuntimeError(f"Failed to save image: {output_path}")


# =========================
# CTR SHADOW CALIBRATION PIPELINE
# =========================
def prepare_calibration_object(
    parent_directory: str,
    analysis_crop: Dict[str, int],
    camera_calibration_file: Optional[str],
    board_reference_image: Optional[str],
    tip_refiner_model: Optional[str],
    tip_refiner_anchor: Optional[str],
    tip_refiner_compare_only: bool,
) -> CTR_Shadow_Calibration:
    print(f"[DEBUG] SCRIPT_DIR={SCRIPT_DIR}")
    import shadow_calibration
    print(f"[DEBUG] shadow_calibration loaded from: {inspect.getsourcefile(shadow_calibration)}")

    cal = CTR_Shadow_Calibration(
        parent_directory=parent_directory,
        project_name="tip_tracking_processing",
        allow_existing=ALLOW_EXISTING_PROJECT,
        add_date=ADD_DATE_TO_PROJECT_FOLDER,
    )
    print("Calibration object created!")

    cal.analysis_crop = dict(analysis_crop)
    cal.tip_refine_mode = DEFAULT_TIP_REFINE_MODE
    cal.tip_detection_mode = DEFAULT_TIP_DETECTION_MODE
    cal.tip_parallel_section_near_r = 0.75
    cal.tip_parallel_section_far_r = 5.0
    cal.tip_parallel_scan_half_r = 2.5
    cal.tip_parallel_num_sections = 7
    cal.tip_parallel_cross_step_px = 0.5
    cal.tip_parallel_ray_step_px = 0.5
    cal.tip_parallel_ray_max_len_r = 10.0

    if camera_calibration_file is not None and os.path.isfile(camera_calibration_file):
        cal.load_camera_calibration(camera_calibration_file)
        if board_reference_image is not None and os.path.isfile(board_reference_image):
            cal.estimate_board_reference_from_image(
                board_reference_image,
                draw_debug=True,
                save_debug_path=os.path.join(parent_directory, "checkerboard_reference_debug.png"),
            )
        else:
            print(f"[WARN] Board reference image not found: {board_reference_image}")
    else:
        print(f"[WARN] Camera calibration file not found, continuing without it: {camera_calibration_file}")

    if tip_refiner_model:
        model_path = Path(tip_refiner_model).expanduser().resolve()
        if not model_path.is_file():
            raise FileNotFoundError(f"Tip refiner model not found: {model_path}")
        cal.load_tip_refiner_model(
            str(model_path),
            anchor_name=tip_refiner_anchor,
            use_as_selected=(not bool(tip_refiner_compare_only)),
        )

    return cal


def analyze_frame_with_ctr_class(
    cal: CTR_Shadow_Calibration,
    image_bgr: np.ndarray,
    threshold: int,
    use_exact_class_thresholding: bool,
) -> Dict[str, float]:
    """
    Same single-frame CTR image-analysis pipeline pattern as the script you provided.
    """
    if image_bgr is None:
        raise ValueError("image_bgr is None")

    crop = cal.analysis_crop
    if crop is None:
        raise RuntimeError("Analysis crop is not configured. Set analysis_crop first.")

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

    if use_exact_class_thresholding:
        _thr, binary_image = cv2.threshold(
            grayscale_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
    else:
        _thr, binary_image = cv2.threshold(
            grayscale_blur, int(threshold), 255, cv2.THRESH_BINARY
        )

    tip_row, tip_column, tip_angle_deg, tip_debug = cal.find_ctr_tip_skeleton(
        binary_image,
        min_spur_len=5,
        return_tip_angle=True,
        return_debug=True,
    )
    tip_angle_deg = _normalize_tip_angle_deg(tip_angle_deg)

    yy_refined, xx_refined, tip_refine_dbg = refine_tip_parallel_centerline(
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
    yy_selected, xx_selected, tip_select_dbg = _select_tip_candidate(
        coarse_tip_yx=(float(tip_row), float(tip_column)),
        refined_tip_yx=(float(yy_refined), float(xx_refined)),
        tip_dbg=tip_refine_dbg,
        mode=cal.tip_refine_mode,
    )
    tip_detection_mode = _normalize_tip_detection_mode(getattr(cal, "tip_detection_mode", "classical"))

    cnn_tip_abs = None
    cnn_tip_dbg = None
    if getattr(cal, "tip_refiner_enabled", False):
        anchor_lookup_abs = {
            "coarse": (float(tip_column + crop_x_min_img), float(tip_row + crop_y_min_img)),
            "refined": (float(xx_refined + crop_x_min_img), float(yy_refined + crop_y_min_img)),
            "selected": (float(xx_selected + crop_x_min_img), float(yy_selected + crop_y_min_img)),
        }
        anchor_x_abs, anchor_y_abs = anchor_lookup_abs[cal.tip_refiner_anchor_name]
        try:
            cnn_tip_dbg = cal.predict_tip_refiner_abs(
                cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY),
                anchor_x_abs=anchor_x_abs,
                anchor_y_abs=anchor_y_abs,
            )
            if cnn_tip_dbg is not None:
                cnn_tip_abs = (float(cnn_tip_dbg["y_abs"]), float(cnn_tip_dbg["x_abs"]))
                if cal.tip_refiner_use_as_selected and tip_detection_mode == "classical":
                    yy_selected = float(cnn_tip_dbg["y_abs"] - crop_y_min_img)
                    xx_selected = float(cnn_tip_dbg["x_abs"] - crop_x_min_img)
                    tip_select_dbg = dict(tip_select_dbg) if isinstance(tip_select_dbg, dict) else {}
                    tip_select_dbg["selected_tip_source"] = "cnn"
                    tip_select_dbg["selected_tip_reason"] = "cnn_tip_refiner"
        except Exception as exc:
            cnn_tip_dbg = {"error": str(exc)}

    tip_row_full_px = float(yy_selected + crop_y_min_img)
    tip_col_full_px = float(xx_selected + crop_x_min_img)

    dbg_local = dict(tip_select_dbg) if isinstance(tip_select_dbg, dict) else {}
    dbg_local["tip_angle_deg"] = float(tip_angle_deg)
    dbg_local["coarse_tip_before_local_xy"] = [float(tip_column), float(tip_row)]
    dbg_local["coarse_tip_after_local_xy"] = [float(xx_refined), float(yy_refined)]
    dbg_local["crop_origin_xy"] = [int(crop_x_min_img), int(crop_y_min_img)]
    dbg_local["tip_detection_mode"] = tip_detection_mode
    if cnn_tip_dbg is not None:
        dbg_local["cnn_tip_refiner"] = cnn_tip_dbg
        dbg_local["cnn_tip_abs_yx"] = None if cnn_tip_abs is None else [float(cnn_tip_abs[0]), float(cnn_tip_abs[1])]
        dbg_local["cnn_anchor_name"] = cal.tip_refiner_anchor_name

    result = {
        "tip_row_px": tip_row_full_px,
        "tip_col_px": tip_col_full_px,
        "tip_angle_deg": float(tip_angle_deg),
        "tip_u_mm": float("nan"),
        "tip_z_mm": float("nan"),
        "location_units": "px",
        "_crop_bounds": (crop_x_min_img, crop_x_max_img, crop_y_min_img, crop_y_max_img),
        "_cropped_image": cropped_image,
        "_binary_image": binary_image,
        "_grayscale_image": grayscale_image,
        "_skeleton": tip_debug.get("skeleton"),
        "_tip_path": tip_debug.get("tip_path", []),
        "_tip_debug": dbg_local,
    }

    try:
        u_mm, z_mm = cal.pixel_point_to_calibrated_axes(
            x_px=tip_col_full_px,
            y_px=tip_row_full_px,
        )
        result["tip_u_mm"] = float(u_mm)
        result["tip_z_mm"] = float(z_mm)
        result["location_units"] = "mm"
    except Exception:
        pass

    return result


def build_annotated_frame(
    image_bgr: np.ndarray,
    analysis: Dict[str, float],
    folder_name: str,
    direction: str,
    speed_mm_min: float,
    phase: str,
    frame_idx: int,
    elapsed_s: float,
    disp_x: float,
    disp_y: float,
    disp_mag: float,
    units: str,
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
        f"{folder_name}",
        f"{direction} | speed={speed_mm_min:.3f} mm/min | phase={phase} | frame={frame_idx}",
        f"t={elapsed_s:.4f} s | tip(px)=({analysis['tip_col_px']:.1f}, {analysis['tip_row_px']:.1f}) | angle={analysis['tip_angle_deg']:.2f} deg",
    ]

    if units == "mm" and np.isfinite(analysis["tip_u_mm"]) and np.isfinite(analysis["tip_z_mm"]):
        lines.append(
            f"tip(mm)=({analysis['tip_u_mm']:.4f}, {analysis['tip_z_mm']:.4f}) | "
            f"disp=({disp_x:.4f}, {disp_y:.4f}) | |disp|={disp_mag:.4f} mm"
        )
    else:
        lines.append(
            f"disp=({disp_x:.4f}, {disp_y:.4f}) | |disp|={disp_mag:.4f} px"
        )

    banner_h = min(128, max(96, 28 * len(lines) + 12))
    overlay = annotated.copy()
    cv2.rectangle(overlay, (0, 0), (annotated.shape[1], banner_h), (8, 10, 16), thickness=-1)
    annotated = cv2.addWeighted(overlay, 0.82, annotated, 0.18, 0.0)

    y = 28
    for idx, line in enumerate(lines):
        scale = 0.68 if idx == 0 else 0.56
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
        y += 26

    return annotated


def save_class_annotated_image(
    image_bgr: np.ndarray,
    analysis: Dict[str, object],
    output_path: str,
    title: str,
) -> None:
    crop_x_min, crop_x_max, crop_y_min, crop_y_max = analysis["_crop_bounds"]
    cropped_image = analysis["_cropped_image"]
    binary_image = analysis["_binary_image"]
    grayscale_image = analysis["_grayscale_image"]
    skeleton = analysis.get("_skeleton")
    tip_path = analysis.get("_tip_path", [])
    dbg = analysis.get("_tip_debug", {})

    fig, axs = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(title, fontsize=13)

    axs[0, 0].imshow(cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB))
    axs[0, 0].set_title("Cropped image")

    axs[0, 1].imshow(binary_image, cmap="gray")
    if skeleton is not None:
        skel_ys, skel_xs = np.where(skeleton == 1)
        axs[0, 1].scatter(skel_xs, skel_ys, s=2, c="cyan", alpha=0.7)
    axs[0, 1].set_title("Thresholded image")

    local_tip_x = float(analysis["tip_col_px"]) - float(crop_x_min)
    local_tip_y = float(analysis["tip_row_px"]) - float(crop_y_min)
    zoom_half = 75
    zoom_x_min = max(0, int(round(local_tip_x)) - zoom_half)
    zoom_x_max = min(binary_image.shape[1] - 1, int(round(local_tip_x)) + zoom_half)
    zoom_y_min = max(0, int(round(local_tip_y)) - zoom_half)
    zoom_y_max = min(binary_image.shape[0] - 1, int(round(local_tip_y)) + zoom_half)

    axs[1, 0].imshow(binary_image[zoom_y_min:zoom_y_max + 1, zoom_x_min:zoom_x_max + 1], cmap="gray")
    axs[1, 0].set_title("Selected tip zoom")

    grayscale_tip = grayscale_image[zoom_y_min:zoom_y_max + 1, zoom_x_min:zoom_x_max + 1]
    axs[1, 1].imshow(grayscale_tip, cmap="gray")
    if len(tip_path) >= 2:
        path_y = np.array([p[0] - zoom_y_min for p in tip_path], dtype=float)
        path_x = np.array([p[1] - zoom_x_min for p in tip_path], dtype=float)
        valid = (
            (path_y >= 0) & (path_y < grayscale_tip.shape[0]) &
            (path_x >= 0) & (path_x < grayscale_tip.shape[1])
        )
        if np.any(valid):
            axs[1, 1].plot(path_x[valid], path_y[valid], "-", color="yellow", linewidth=2)
    axs[1, 1].set_title("CTR class tip geometry")

    for ax in axs.flat:
        ax.set_axis_off()

    _remap_zoom_axes_to_crop_coordinates(axs, zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max)
    annotate_tip_geometry_on_axes(axs, dbg, title_suffix="")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =========================
# PROCESSING
# =========================
def choose_run_units(results_df: pd.DataFrame) -> str:
    if {"tip_u_mm", "tip_z_mm"}.issubset(results_df.columns):
        u = pd.to_numeric(results_df["tip_u_mm"], errors="coerce").to_numpy(dtype=float)
        z = pd.to_numeric(results_df["tip_z_mm"], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(u).sum() >= 1 and np.isfinite(z).sum() >= 1:
            return "mm"
    return "px"


def process_single_run_folder(
    run_info: RunFolderInfo,
    cal: CTR_Shadow_Calibration,
    output_root: str,
    threshold: int,
    use_exact_class_thresholding: bool,
) -> Tuple[pd.DataFrame, Dict[str, object]]:
    print("\n" + "=" * 80)
    print(f"[RUN] Processing {run_info.folder_name}")
    print(
        f"      speed={run_info.speed_mm_min:.3f} mm/min | "
        f"dataset={run_info.dataset_label} | run_type={run_info.run_type} | "
        f"orientation={run_info.orientation_name}"
    )
    print("=" * 80)

    run_output_dir = ensure_dir(os.path.join(output_root, RUN_OUTPUT_SUBDIR, run_info.folder_name))
    annotated_dir = ensure_dir(os.path.join(run_output_dir, "annotated_outputs"))
    log_df = read_capture_log(run_info.capture_log_path)

    rows: List[Dict[str, object]] = []
    missing_images = 0
    units = None
    ref_x = None
    ref_y = None

    for _, log_row in log_df.iterrows():
        image_file = str(log_row["image_file"]).strip()
        image_path = os.path.join(run_info.folder_path, image_file)

        if not os.path.isfile(image_path):
            print(f"[WARN] Missing image: {image_path}")
            missing_images += 1
            continue

        frame_bgr = cv2.imread(image_path)
        if frame_bgr is None:
            print(f"[WARN] Could not read image: {image_path}")
            continue

        try:
            analysis = analyze_frame_with_ctr_class(
                cal=cal,
                image_bgr=frame_bgr,
                threshold=threshold,
                use_exact_class_thresholding=use_exact_class_thresholding,
            )
        except Exception as exc:
            print(f"[WARN] Tip analysis failed for {image_path}: {exc}")
            continue

        current_units = "mm"
        if not (np.isfinite(analysis["tip_u_mm"]) and np.isfinite(analysis["tip_z_mm"])):
            current_units = "px"

        if units is None:
            units = current_units

        if units == "mm":
            cur_x = float(analysis["tip_u_mm"])
            cur_y = float(analysis["tip_z_mm"])
        else:
            cur_x = float(analysis["tip_col_px"])
            cur_y = float(analysis["tip_row_px"])

        if ref_x is None or ref_y is None:
            ref_x = cur_x
            ref_y = cur_y

        disp_x = cur_x - ref_x
        disp_y = cur_y - ref_y
        disp_mag = float(np.sqrt(disp_x ** 2 + disp_y ** 2))

        frame_idx = int(log_row["frame_idx"])
        elapsed_s = float(log_row["elapsed_s_from_motion_start"])
        phase = str(log_row.get("phase", "unknown")).strip().lower()
        if phase not in PHASE_COLORS:
            phase = "unknown"
        sample_type = str(log_row.get("sample_type", "motion")).strip().lower()

        annotated_name = f"{os.path.splitext(image_file)[0]}_annotated{ANNOTATED_IMAGE_EXTENSION}"
        annotated_path = os.path.join(annotated_dir, annotated_name)
        save_class_annotated_image(
            image_bgr=frame_bgr,
            analysis=analysis,
            output_path=annotated_path,
            title=(
                f"{run_info.folder_name} | {sample_type} | {run_info.direction} | "
                f"{run_info.speed_mm_min:.3f} mm/min | phase={phase} | frame={frame_idx}"
            ),
        )

        row = {
            "folder_name": run_info.folder_name,
            "folder_path": run_info.folder_path,
            "image_file": image_file,
            "image_path": image_path,
            "annotated_file": annotated_name,
            "annotated_path": annotated_path,
            "frame_idx": frame_idx,
            "elapsed_s_from_motion_start": elapsed_s,
            "elapsed_s_from_run_start": float(log_row.get("elapsed_s_from_run_start", elapsed_s)),
            "elapsed_s_from_segment_start": float(log_row.get("elapsed_s_from_segment_start", np.nan)),
            "sample_type": sample_type,
            "phase": phase,
            "direction": run_info.direction,
            "dataset_label": run_info.dataset_label,
            "orientation_name": run_info.orientation_name,
            "orientation_deg": run_info.orientation_deg,
            "run_type": run_info.run_type,
            "repeat_idx": int(run_info.repeat_idx),
            "repeat_count": int(run_info.repeat_count),
            "daq_mode": str(log_row.get("daq_mode", run_info.daq_mode)).strip().lower(),
            "attack_angle_deg": float(pd.to_numeric(log_row.get("attack_angle_deg", run_info.attack_angle_deg), errors="coerce")),
            "b_command_mm": float(pd.to_numeric(log_row.get("b_command_mm", run_info.b_command_mm), errors="coerce")),
            "segment_name": str(log_row.get("segment_name", "unknown")).strip().lower(),
            "segment_index": pd.to_numeric(log_row.get("segment_index", np.nan), errors="coerce"),
            "speed_mm_min": float(run_info.speed_mm_min),
            "tip_row_px": float(analysis["tip_row_px"]),
            "tip_col_px": float(analysis["tip_col_px"]),
            "tip_u_mm": float(analysis["tip_u_mm"]),
            "tip_z_mm": float(analysis["tip_z_mm"]),
            "tip_angle_deg": float(analysis["tip_angle_deg"]),
            "location_units": str(analysis["location_units"]),
            "analysis_units": units,
            "ref_x": float(ref_x),
            "ref_y": float(ref_y),
            "disp_x": float(disp_x),
            "disp_y": float(disp_y),
            "disp_mag": float(disp_mag),
        }
        rows.append(row)

        if units == "mm":
            print(
                f"  [TRACK] frame={frame_idx:06d} | phase={phase} | "
                f"tip(px)=({analysis['tip_col_px']:.2f}, {analysis['tip_row_px']:.2f}) | "
                f"tip(mm)=({analysis['tip_u_mm']:.4f}, {analysis['tip_z_mm']:.4f}) | "
                f"disp=({disp_x:.4f}, {disp_y:.4f}) | |disp|={disp_mag:.4f} mm"
            )
        else:
            print(
                f"  [TRACK] frame={frame_idx:06d} | phase={phase} | "
                f"tip(px)=({analysis['tip_col_px']:.2f}, {analysis['tip_row_px']:.2f}) | "
                f"disp=({disp_x:.4f}, {disp_y:.4f}) | |disp|={disp_mag:.4f} px"
            )

    if not rows:
        raise RuntimeError(f"No analyzable images found in {run_info.folder_path}")

    results_df = pd.DataFrame(rows).sort_values("frame_idx").reset_index(drop=True)
    units = choose_run_units(results_df)
    if units == "mm":
        results_df["current_x"] = pd.to_numeric(results_df["tip_u_mm"], errors="coerce")
        results_df["current_y"] = pd.to_numeric(results_df["tip_z_mm"], errors="coerce")
    else:
        results_df["current_x"] = pd.to_numeric(results_df["tip_col_px"], errors="coerce")
        results_df["current_y"] = pd.to_numeric(results_df["tip_row_px"], errors="coerce")

    initial_x = float(results_df.iloc[0]["current_x"])
    initial_y = float(results_df.iloc[0]["current_y"])
    results_df["initial_ref_x"] = initial_x
    results_df["initial_ref_y"] = initial_y
    results_df["disp_x_from_initial"] = results_df["current_x"] - initial_x
    results_df["disp_y_from_initial"] = results_df["current_y"] - initial_y
    results_df["disp_mag_from_initial"] = np.sqrt(
        results_df["disp_x_from_initial"] ** 2 + results_df["disp_y_from_initial"] ** 2
    )

    motion_df = results_df[results_df["sample_type"] == "motion"].copy()
    if motion_df.empty:
        motion_ref_x = float("nan")
        motion_ref_y = float("nan")
    else:
        motion_ref_x = float(motion_df.iloc[0]["current_x"])
        motion_ref_y = float(motion_df.iloc[0]["current_y"])

    results_df["motion_start_ref_x"] = motion_ref_x
    results_df["motion_start_ref_y"] = motion_ref_y
    if np.isfinite(motion_ref_x) and np.isfinite(motion_ref_y):
        results_df["disp_x_from_motion_start"] = results_df["current_x"] - motion_ref_x
        results_df["disp_y_from_motion_start"] = results_df["current_y"] - motion_ref_y
        results_df["disp_mag_from_motion_start"] = np.sqrt(
            results_df["disp_x_from_motion_start"] ** 2 + results_df["disp_y_from_motion_start"] ** 2
        )
    else:
        results_df["disp_x_from_motion_start"] = np.nan
        results_df["disp_y_from_motion_start"] = np.nan
        results_df["disp_mag_from_motion_start"] = np.nan

    middle_pause_df = results_df[results_df["sample_type"] == "idle_mid_pause"].copy()
    if middle_pause_df.empty:
        middle_pause_x = float("nan")
        middle_pause_y = float("nan")
    else:
        middle_pause_x = float(middle_pause_df.iloc[-1]["current_x"])
        middle_pause_y = float(middle_pause_df.iloc[-1]["current_y"])

    results_df["middle_pause_ref_x"] = middle_pause_x
    results_df["middle_pause_ref_y"] = middle_pause_y
    if np.isfinite(middle_pause_x) and np.isfinite(middle_pause_y):
        results_df["disp_x_from_middle_pause"] = results_df["current_x"] - middle_pause_x
        results_df["disp_y_from_middle_pause"] = results_df["current_y"] - middle_pause_y
        results_df["disp_mag_from_middle_pause"] = np.sqrt(
            results_df["disp_x_from_middle_pause"] ** 2 + results_df["disp_y_from_middle_pause"] ** 2
        )
    else:
        results_df["disp_x_from_middle_pause"] = np.nan
        results_df["disp_y_from_middle_pause"] = np.nan
        results_df["disp_mag_from_middle_pause"] = np.nan

    results_df["disp_x"] = results_df["disp_x_from_initial"]
    results_df["disp_y"] = results_df["disp_y_from_initial"]
    results_df["disp_mag"] = results_df["disp_mag_from_initial"]

    results_csv_path = os.path.join(run_output_dir, RUN_RESULTS_CSV_NAME)
    results_df.to_csv(results_csv_path, index=False)

    if units == "mm":
        dx_label = "Δu (mm)"
        dy_label = "Δz (mm)"
        mag_label = "|Δtip| (mm)"
    else:
        dx_label = "Δx (px)"
        dy_label = "Δy (px)"
        mag_label = "|Δtip| (px)"

    make_per_run_plots(
        results_df=results_df,
        output_dir=run_output_dir,
        folder_name=run_info.folder_name,
        speed_mm_min=run_info.speed_mm_min,
        dataset_label=run_info.dataset_label,
        run_type=run_info.run_type,
        dx_label=dx_label,
        dy_label=dy_label,
        mag_label=mag_label,
    )

    ss_df = results_df[results_df["phase"] == "ss"].copy()
    second_leg_ss_df = results_df[
        (results_df["phase"] == "ss") & (results_df["segment_name"] == "mid_to_end")
    ].copy()

    def _mean_std(frame: pd.DataFrame, prefix: str) -> Tuple[float, float, float, float, int]:
        if frame.empty:
            return float("nan"), float("nan"), float("nan"), float("nan"), 0
        mean_x = float(pd.to_numeric(frame[f"disp_x_from_{prefix}"], errors="coerce").mean())
        mean_y = float(pd.to_numeric(frame[f"disp_y_from_{prefix}"], errors="coerce").mean())
        mean_mag = float(pd.to_numeric(frame[f"disp_mag_from_{prefix}"], errors="coerce").mean())
        std_mag = float(pd.to_numeric(frame[f"disp_mag_from_{prefix}"], errors="coerce").std())
        return mean_x, mean_y, mean_mag, std_mag, int(len(frame))

    mean_ss_disp_x_from_initial, mean_ss_disp_y_from_initial, mean_ss_disp_mag_from_initial, std_ss_disp_mag_from_initial, n_ss_all = _mean_std(
        ss_df,
        "initial",
    )
    mean_second_leg_ss_disp_x_from_middle, mean_second_leg_ss_disp_y_from_middle, mean_second_leg_ss_disp_mag_from_middle, std_second_leg_ss_disp_mag_from_middle, n_ss_second_leg = _mean_std(
        second_leg_ss_df,
        "middle_pause",
    )

    if run_info.run_type == "two_stage" and np.isfinite(middle_pause_x) and n_ss_second_leg > 0:
        mean_ss_disp_x = mean_second_leg_ss_disp_x_from_middle
        mean_ss_disp_y = mean_second_leg_ss_disp_y_from_middle
        mean_ss_disp_mag = mean_second_leg_ss_disp_mag_from_middle
        std_ss_disp_mag = std_second_leg_ss_disp_mag_from_middle
        n_ss = n_ss_second_leg
        ss_reference = "middle_pause"
        ss_segment_name = "mid_to_end"
    else:
        mean_ss_disp_x = mean_ss_disp_x_from_initial
        mean_ss_disp_y = mean_ss_disp_y_from_initial
        mean_ss_disp_mag = mean_ss_disp_mag_from_initial
        std_ss_disp_mag = std_ss_disp_mag_from_initial
        n_ss = n_ss_all
        ss_reference = "initial"
        ss_segment_name = "all_ss"

    final_row = results_df.iloc[-1]
    first_motion_row = results_df[results_df["sample_type"] == "motion"].head(1)
    if first_motion_row.empty:
        initial_to_first_motion_disp_x = float("nan")
        initial_to_first_motion_disp_y = float("nan")
        initial_to_first_motion_disp_mag = float("nan")
        first_motion_frame_idx = float("nan")
    else:
        first_motion = first_motion_row.iloc[0]
        initial_to_first_motion_disp_x = float(first_motion["disp_x_from_initial"])
        initial_to_first_motion_disp_y = float(first_motion["disp_y_from_initial"])
        initial_to_first_motion_disp_mag = float(first_motion["disp_mag_from_initial"])
        first_motion_frame_idx = int(first_motion["frame_idx"])

    if middle_pause_df.empty:
        middle_pause_disp_x_from_initial = float("nan")
        middle_pause_disp_y_from_initial = float("nan")
        middle_pause_disp_mag_from_initial = float("nan")
        middle_pause_frame_idx = float("nan")
    else:
        middle_pause_row = middle_pause_df.iloc[-1]
        middle_pause_disp_x_from_initial = float(middle_pause_row["disp_x_from_initial"])
        middle_pause_disp_y_from_initial = float(middle_pause_row["disp_y_from_initial"])
        middle_pause_disp_mag_from_initial = float(middle_pause_row["disp_mag_from_initial"])
        middle_pause_frame_idx = int(middle_pause_row["frame_idx"])

    summary_row: Dict[str, object] = {
        "folder_name": run_info.folder_name,
        "folder_path": run_info.folder_path,
        "results_csv_path": results_csv_path,
        "annotated_dir": annotated_dir,
        "speed_mm_min": float(run_info.speed_mm_min),
        "direction": run_info.direction,
        "dataset_label": run_info.dataset_label,
        "orientation_name": run_info.orientation_name,
        "orientation_deg": float(run_info.orientation_deg),
        "run_type": run_info.run_type,
        "repeat_idx": int(run_info.repeat_idx),
        "repeat_count": int(run_info.repeat_count),
        "daq_mode": str(run_info.daq_mode),
        "attack_angle_deg": float(run_info.attack_angle_deg),
        "b_command_mm": float(run_info.b_command_mm),
        "analysis_units": units,
        "n_frames_analyzed": int(len(results_df)),
        "n_idle_before_frames": int((results_df["sample_type"] == "idle_before_motion").sum()),
        "n_idle_middle_frames": int((results_df["sample_type"] == "idle_mid_pause").sum()),
        "n_idle_after_frames": int((results_df["sample_type"] == "idle_after_motion").sum()),
        "n_idle_frames": int(
            results_df["sample_type"].isin(["idle_before_motion", "idle_mid_pause", "idle_after_motion"]).sum()
        ),
        "n_missing_images": int(missing_images),
        "n_ss_frames": int(n_ss),
        "n_ss_frames_all": int(n_ss_all),
        "n_ss_frames_second_leg": int(n_ss_second_leg),
        "initial_ref_x": initial_x,
        "initial_ref_y": initial_y,
        "motion_start_ref_x": motion_ref_x,
        "motion_start_ref_y": motion_ref_y,
        "middle_pause_ref_x": middle_pause_x,
        "middle_pause_ref_y": middle_pause_y,
        "first_motion_frame_idx": first_motion_frame_idx,
        "middle_pause_frame_idx": middle_pause_frame_idx,
        "initial_to_first_motion_disp_x": initial_to_first_motion_disp_x,
        "initial_to_first_motion_disp_y": initial_to_first_motion_disp_y,
        "initial_to_first_motion_disp_mag": initial_to_first_motion_disp_mag,
        "middle_pause_disp_x_from_initial": middle_pause_disp_x_from_initial,
        "middle_pause_disp_y_from_initial": middle_pause_disp_y_from_initial,
        "middle_pause_disp_mag_from_initial": middle_pause_disp_mag_from_initial,
        "mean_ss_disp_x_from_initial": mean_ss_disp_x_from_initial,
        "mean_ss_disp_y_from_initial": mean_ss_disp_y_from_initial,
        "mean_ss_disp_mag_from_initial": mean_ss_disp_mag_from_initial,
        "std_ss_disp_mag_from_initial": std_ss_disp_mag_from_initial,
        "mean_second_leg_ss_disp_x_from_middle_pause": mean_second_leg_ss_disp_x_from_middle,
        "mean_second_leg_ss_disp_y_from_middle_pause": mean_second_leg_ss_disp_y_from_middle,
        "mean_second_leg_ss_disp_mag_from_middle_pause": mean_second_leg_ss_disp_mag_from_middle,
        "std_second_leg_ss_disp_mag_from_middle_pause": std_second_leg_ss_disp_mag_from_middle,
        "ss_reference": ss_reference,
        "ss_segment_name": ss_segment_name,
        "mean_ss_disp_x": mean_ss_disp_x,
        "mean_ss_disp_y": mean_ss_disp_y,
        "mean_ss_disp_mag": mean_ss_disp_mag,
        "std_ss_disp_mag": std_ss_disp_mag,
        "final_disp_x": float(final_row["disp_x_from_initial"]),
        "final_disp_y": float(final_row["disp_y_from_initial"]),
        "final_disp_mag": float(final_row["disp_mag_from_initial"]),
        "max_disp_mag": float(pd.to_numeric(results_df["disp_mag_from_initial"], errors="coerce").max()),
        "min_elapsed_s": float(pd.to_numeric(results_df["elapsed_s_from_motion_start"], errors="coerce").min()),
        "max_elapsed_s": float(pd.to_numeric(results_df["elapsed_s_from_motion_start"], errors="coerce").max()),
    }

    if np.isfinite(summary_row["mean_ss_disp_mag"]):
        print(
            f"[INFO] {run_info.folder_name} | average steady-state displacement = "
            f"{summary_row['mean_ss_disp_mag']:.6f} {units} | "
            f"ref={summary_row['ss_reference']} | n_ss={n_ss}"
        )
    else:
        print(
            f"[INFO] {run_info.folder_name} | average steady-state displacement = nan "
            f"(no steady-state frames)"
        )

    return results_df, summary_row


# =========================
# PLOTTING
# =========================
def make_per_run_plots(
    results_df: pd.DataFrame,
    output_dir: str,
    folder_name: str,
    speed_mm_min: float,
    dataset_label: str,
    run_type: str,
    dx_label: str,
    dy_label: str,
    mag_label: str,
) -> None:
    title_base = f"{folder_name} | {dataset_label} | {run_type} | {speed_mm_min:.3f} mm/min"

    # magnitude scatter vs time
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    for phase_name, dfi in results_df.groupby("phase"):
        phase_key = phase_name if phase_name in PHASE_COLORS else "unknown"
        ax.scatter(
            pd.to_numeric(dfi["elapsed_s_from_motion_start"], errors="coerce"),
            pd.to_numeric(dfi["disp_mag"], errors="coerce"),
            s=42,
            color=PHASE_COLORS[phase_key],
            alpha=0.82,
            edgecolors="#f8fafc",
            linewidths=0.35,
            label=phase_name,
        )

    _apply_dark_axes_style(
        ax,
        title=f"{title_base}\nTip displacement magnitude over time",
        xlabel="Elapsed time from motion start (s)",
        ylabel=mag_label,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(loc="best", frameon=True, fontsize=9)
        leg.get_frame().set_facecolor("#121c28")
        leg.get_frame().set_edgecolor((0.75, 0.84, 0.93, 0.18))
        for txt in leg.get_texts():
            txt.set_color("#e8f0f8")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "displacement_magnitude_vs_time_scatter.png"),
        dpi=220,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)

    # xy scatter relative to first image
    fig, ax = plt.subplots(figsize=(7.4, 7.0))
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    for phase_name, dfi in results_df.groupby("phase"):
        phase_key = phase_name if phase_name in PHASE_COLORS else "unknown"
        ax.scatter(
            pd.to_numeric(dfi["disp_x"], errors="coerce"),
            pd.to_numeric(dfi["disp_y"], errors="coerce"),
            s=44,
            color=PHASE_COLORS[phase_key],
            alpha=0.84,
            edgecolors="#f8fafc",
            linewidths=0.35,
            label=phase_name,
        )

    ax.axhline(0.0, color="#cbd5e1", linestyle="--", linewidth=1.0, alpha=0.65)
    ax.axvline(0.0, color="#cbd5e1", linestyle="--", linewidth=1.0, alpha=0.65)

    _apply_dark_axes_style(
        ax,
        title=f"{title_base}\nTip displacement scatter relative to first image",
        xlabel=dx_label,
        ylabel=dy_label,
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(loc="best", frameon=True, fontsize=9)
        leg.get_frame().set_facecolor("#121c28")
        leg.get_frame().set_edgecolor((0.75, 0.84, 0.93, 0.18))
        for txt in leg.get_texts():
            txt.set_color("#e8f0f8")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "displacement_xy_scatter.png"),
        dpi=220,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)

    # components over time
    fig, ax = plt.subplots(figsize=(9.2, 6.2))
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    elapsed = pd.to_numeric(results_df["elapsed_s_from_motion_start"], errors="coerce")
    disp_x = pd.to_numeric(results_df["disp_x"], errors="coerce")
    disp_y = pd.to_numeric(results_df["disp_y"], errors="coerce")

    ax.plot(elapsed, disp_x, marker="o", markersize=4.5, linewidth=1.8, alpha=0.92, label=dx_label)
    ax.plot(elapsed, disp_y, marker="s", markersize=4.1, linewidth=1.6, alpha=0.92, linestyle="--", label=dy_label)
    ax.axhline(0.0, color="#cbd5e1", linestyle="--", linewidth=1.0, alpha=0.65)

    _apply_dark_axes_style(
        ax,
        title=f"{title_base}\nTip displacement components over time",
        xlabel="Elapsed time from motion start (s)",
        ylabel=f"{dx_label} / {dy_label}",
    )
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        leg = ax.legend(loc="best", frameon=True, fontsize=9)
        leg.get_frame().set_facecolor("#121c28")
        leg.get_frame().set_edgecolor((0.75, 0.84, 0.93, 0.18))
        for txt in leg.get_texts():
            txt.set_color("#e8f0f8")
    fig.tight_layout()
    fig.savefig(
        os.path.join(output_dir, "displacement_components_vs_time.png"),
        dpi=220,
        bbox_inches="tight",
        transparent=True,
    )
    plt.close(fig)

    middle_pause_df = results_df[
        results_df["sample_type"].astype(str).str.lower() == "idle_mid_pause"
    ].copy()
    second_leg_ss_df = results_df[
        (results_df["phase"].astype(str).str.lower() == "ss")
        & (results_df["segment_name"].astype(str).str.lower() == "mid_to_end")
    ].copy()
    if not middle_pause_df.empty or not second_leg_ss_df.empty:
        fig, ax = plt.subplots(figsize=(9.2, 6.2))
        fig.patch.set_facecolor("none")
        fig.patch.set_alpha(0.0)

        elapsed = pd.to_numeric(results_df["elapsed_s_from_run_start"], errors="coerce")
        disp_mid = pd.to_numeric(results_df["disp_mag_from_middle_pause"], errors="coerce")
        ax.scatter(
            elapsed,
            disp_mid,
            s=42,
            color="#8ecae6",
            alpha=0.72,
            edgecolors="#f8fafc",
            linewidths=0.35,
            label="all frames vs middle pause",
        )
        if not second_leg_ss_df.empty:
            ax.scatter(
                pd.to_numeric(second_leg_ss_df["elapsed_s_from_run_start"], errors="coerce"),
                pd.to_numeric(second_leg_ss_df["disp_mag_from_middle_pause"], errors="coerce"),
                s=58,
                color=PHASE_COLORS["ss"],
                alpha=0.92,
                edgecolors="#f8fafc",
                linewidths=0.4,
                label="second-leg steady state",
            )
        if not middle_pause_df.empty:
            ax.scatter(
                pd.to_numeric(middle_pause_df["elapsed_s_from_run_start"], errors="coerce"),
                pd.to_numeric(middle_pause_df["disp_mag_from_middle_pause"], errors="coerce"),
                s=70,
                color=PHASE_COLORS["pause"],
                alpha=0.95,
                edgecolors="#f8fafc",
                linewidths=0.4,
                label="middle pause",
            )

        _apply_dark_axes_style(
            ax,
            title=f"{title_base}\nDisplacement relative to middle pause",
            xlabel="Elapsed time from run start (s)",
            ylabel=mag_label,
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(loc="best", frameon=True, fontsize=9)
            leg.get_frame().set_facecolor("#121c28")
            leg.get_frame().set_edgecolor((0.75, 0.84, 0.93, 0.18))
            for txt in leg.get_texts():
                txt.set_color("#e8f0f8")
        fig.tight_layout()
        fig.savefig(
            os.path.join(output_dir, "displacement_from_middle_pause_vs_time.png"),
            dpi=220,
            bbox_inches="tight",
            transparent=True,
        )
        plt.close(fig)


def make_summary_plots(summary_df: pd.DataFrame, output_dir: str, theme: str = "dark") -> None:
    if summary_df.empty:
        return

    output_dir = ensure_dir(output_dir)
    averaged_df = build_speed_summary(summary_df)

    units_mode = summary_df["analysis_units"].mode()
    units = str(units_mode.iloc[0]) if not units_mode.empty else "px"
    mag_label = f"|Δtip| ({units})"
    dataset_order = [
        label for label in ["forward", "backward", "transverse"]
        if label in set(summary_df["dataset_label"].astype(str).str.lower())
    ]
    for label in sorted(set(summary_df["dataset_label"].astype(str).str.lower())):
        if label not in dataset_order:
            dataset_order.append(label)
    fig, axes = plt.subplots(
        1,
        max(1, len(dataset_order)),
        figsize=(5.2 * max(1, len(dataset_order)), 4.8),
        squeeze=False,
    )
    if theme == "dark":
        fig.patch.set_facecolor("none")
        fig.patch.set_alpha(0.0)
    else:
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)

    curve_specs = [
        {
            "label": "Plunge Reset",
            "run_type": "direct",
            "color": "#3d70b2",
        },
        {
            "label": "Bath start-stop",
            "run_type": "two_stage",
            "color": "#c83f3f",
        },
    ]

    for col_idx, dataset_label in enumerate(dataset_order):
        ax = axes[0, col_idx]
        for spec in curve_specs:
            dfi = summary_df[
                (summary_df["run_type"].astype(str).str.lower() == spec["run_type"])
                & (summary_df["dataset_label"].astype(str).str.lower() == dataset_label)
            ].copy().sort_values("speed_mm_min")
            avgi = averaged_df[
                (averaged_df["run_type"].astype(str).str.lower() == spec["run_type"])
                & (averaged_df["dataset_label"].astype(str).str.lower() == dataset_label)
            ].copy().sort_values("speed_mm_min")
            if not dfi.empty:
                run_vals = pd.to_numeric(dfi["mean_ss_disp_mag"], errors="coerce")
                run_speed = pd.to_numeric(dfi["speed_mm_min"], errors="coerce")
                mask = run_speed.notna() & run_vals.notna()
                if mask.any():
                    ax.scatter(
                        run_speed[mask],
                        run_vals[mask],
                        s=50,
                        color=spec["color"],
                        alpha=0.22,
                        edgecolors="none",
                        label=None,
                    )
            if not avgi.empty:
                mean_vals = pd.to_numeric(avgi["mean_of_mean_ss_disp_mag"], errors="coerce")
                std_vals = pd.to_numeric(avgi["std_of_mean_ss_disp_mag"], errors="coerce").fillna(0.0)
                speed_vals = pd.to_numeric(avgi["speed_mm_min"], errors="coerce")
                mask = speed_vals.notna() & mean_vals.notna()
                if mask.any():
                    ax.errorbar(
                        speed_vals[mask],
                        mean_vals[mask],
                        yerr=std_vals[mask],
                        linewidth=2.0,
                        marker="o",
                        markersize=5.2,
                        alpha=0.95,
                        color=spec["color"],
                        capsize=3.5,
                        label=spec["label"],
                    )

        _style_axes(
            ax,
            theme,
            title=f"{dataset_label}\nSteady-state displacement vs speed",
            xlabel="Bath speed (mm/min)",
            ylabel=f"{mag_label} | plunge reset / bath start-stop" if col_idx == 0 else "",
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(loc="best", frameon=True, fontsize=9)
            _style_legend(leg, theme)

    fig.tight_layout()
    _save_plot_outputs(
        fig,
        os.path.join(output_dir, "steady_state_displacement_vs_speed_by_dataset_and_run_type"),
        theme,
    )
    plt.close(fig)

    two_stage_df = summary_df[
        summary_df["run_type"].astype(str).str.lower() == "two_stage"
    ].copy()
    if not two_stage_df.empty:
        fig, axes = plt.subplots(1, max(1, len(dataset_order)), figsize=(5.2 * max(1, len(dataset_order)), 5.0), squeeze=False)
        if theme == "dark":
            fig.patch.set_facecolor("none")
            fig.patch.set_alpha(0.0)
        else:
            fig.patch.set_facecolor("white")
            fig.patch.set_alpha(1.0)
        for col_idx, dataset_label in enumerate(dataset_order):
            ax = axes[0, col_idx]
            dfi = two_stage_df[
                two_stage_df["dataset_label"].astype(str).str.lower() == dataset_label
            ].copy().sort_values("speed_mm_min")
            if not dfi.empty:
                color = DATASET_COLORS.get(dataset_label, "#fca5a5")
                ax.scatter(
                    dfi["speed_mm_min"],
                    dfi["middle_pause_disp_mag_from_initial"],
                    s=72,
                    color=color,
                    alpha=0.36,
                    edgecolors="none",
                    label="runs",
                )
                avg_dfi = averaged_df[
                    (averaged_df["run_type"].astype(str).str.lower() == "two_stage")
                    & (averaged_df["dataset_label"].astype(str).str.lower() == dataset_label)
                ].copy().sort_values("speed_mm_min")
                yerr = pd.to_numeric(avg_dfi["mean_of_middle_pause_disp_mag_from_initial"], errors="coerce") * 0.0
                ax.errorbar(
                    avg_dfi["speed_mm_min"],
                    avg_dfi["mean_of_middle_pause_disp_mag_from_initial"],
                    yerr=yerr,
                    linewidth=2.0,
                    marker="o",
                    markersize=5.2,
                    alpha=0.95,
                    color=color,
                    capsize=3.5,
                    label="repeat mean",
                )
            _style_axes(
                ax,
                theme,
                title=f"{dataset_label} | two_stage\nMiddle-pause displacement vs speed",
                xlabel="Bath speed (mm/min)",
                ylabel=f"{mag_label} from initial pose",
            )
            handles, labels = ax.get_legend_handles_labels()
            if handles:
                leg = ax.legend(loc="best", frameon=True, fontsize=9)
                _style_legend(leg, theme)
        fig.tight_layout()
        _save_plot_outputs(
            fig,
            os.path.join(output_dir, "middle_pause_displacement_vs_speed_by_dataset"),
            theme,
        )
        plt.close(fig)


    make_b_attack_summary_plots(summary_df, output_dir, theme=theme)


def make_b_attack_summary_plots(summary_df: pd.DataFrame, output_dir: str, theme: str = "dark") -> None:
    if summary_df.empty or "attack_angle_deg" not in summary_df.columns:
        return

    attack_df = summary_df[pd.to_numeric(summary_df["attack_angle_deg"], errors="coerce").notna()].copy()
    if attack_df.empty:
        return

    attack_df["attack_angle_deg"] = pd.to_numeric(attack_df["attack_angle_deg"], errors="coerce")
    attack_df["mean_ss_disp_mag"] = pd.to_numeric(attack_df["mean_ss_disp_mag"], errors="coerce")
    attack_df["middle_pause_disp_mag_from_initial"] = pd.to_numeric(
        attack_df["middle_pause_disp_mag_from_initial"], errors="coerce"
    )

    units_mode = attack_df["analysis_units"].mode()
    units = str(units_mode.iloc[0]) if not units_mode.empty else "px"
    mag_label = f"|Δtip| ({units})"

    group_cols = [
        "dataset_label",
        "orientation_name",
        "orientation_deg",
        "speed_mm_min",
        "attack_angle_deg",
        "analysis_units",
    ]
    avg_df = (
        attack_df.groupby(group_cols, as_index=False)
        .agg(
            n_runs=("folder_name", "count"),
            mean_of_mean_ss_disp_mag=("mean_ss_disp_mag", "mean"),
            std_of_mean_ss_disp_mag=("mean_ss_disp_mag", "std"),
            mean_middle_pause_disp_mag_from_initial=("middle_pause_disp_mag_from_initial", "mean"),
            std_middle_pause_disp_mag_from_initial=("middle_pause_disp_mag_from_initial", "std"),
            mean_b_command_mm=("b_command_mm", "mean"),
        )
        .sort_values(["orientation_deg", "speed_mm_min", "attack_angle_deg"])
        .reset_index(drop=True)
    )
    avg_csv = os.path.join(output_dir, "b_attack_summary_by_attack_angle.csv")
    avg_df.to_csv(avg_csv, index=False)

    dataset_order = []
    for label in ["forward", "backward", "transverse"]:
        if label in set(attack_df["dataset_label"].astype(str).str.lower()):
            dataset_order.append(label)
    for label in sorted(set(attack_df["dataset_label"].astype(str).str.lower())):
        if label not in dataset_order:
            dataset_order.append(label)

    fig, axes = plt.subplots(
        1,
        max(1, len(dataset_order)),
        figsize=(5.4 * max(1, len(dataset_order)), 5.1),
        squeeze=False,
    )
    if theme == "dark":
        fig.patch.set_facecolor("none")
        fig.patch.set_alpha(0.0)
    else:
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)

    for col_idx, dataset_label in enumerate(dataset_order):
        ax = axes[0, col_idx]
        dfi = attack_df[attack_df["dataset_label"].astype(str).str.lower() == dataset_label].copy()
        avgi = avg_df[avg_df["dataset_label"].astype(str).str.lower() == dataset_label].copy()
        color = DATASET_COLORS.get(dataset_label, "#93c5fd")
        if not dfi.empty:
            ax.scatter(
                dfi["attack_angle_deg"],
                dfi["mean_ss_disp_mag"],
                s=58,
                color=color,
                alpha=0.36,
                edgecolors="none",
                label="runs",
            )
        if not avgi.empty:
            yerr = pd.to_numeric(avgi["std_of_mean_ss_disp_mag"], errors="coerce").fillna(0.0)
            ax.errorbar(
                avgi["attack_angle_deg"],
                avgi["mean_of_mean_ss_disp_mag"],
                yerr=yerr,
                linewidth=2.2,
                marker="o",
                markersize=4.8,
                alpha=0.95,
                color=color,
                capsize=3.5,
                label="repeat mean",
            )
        _style_axes(
            ax,
            theme,
            title=f"{dataset_label} | B-attack sweep\nSteady-state displacement vs attack angle",
            xlabel="B curl attack angle (deg)",
            ylabel=f"{mag_label} | second leg from middle pose",
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(loc="best", frameon=True, fontsize=9)
            _style_legend(leg, theme)
    fig.tight_layout()
    _save_plot_outputs(
        fig,
        os.path.join(output_dir, "b_attack_steady_state_displacement_vs_attack_angle"),
        theme,
    )
    plt.close(fig)

    fig, axes = plt.subplots(
        1,
        max(1, len(dataset_order)),
        figsize=(5.4 * max(1, len(dataset_order)), 5.1),
        squeeze=False,
    )
    if theme == "dark":
        fig.patch.set_facecolor("none")
        fig.patch.set_alpha(0.0)
    else:
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)
    for col_idx, dataset_label in enumerate(dataset_order):
        ax = axes[0, col_idx]
        dfi = attack_df[attack_df["dataset_label"].astype(str).str.lower() == dataset_label].copy()
        avgi = avg_df[avg_df["dataset_label"].astype(str).str.lower() == dataset_label].copy()
        color = DATASET_COLORS.get(dataset_label, "#fca5a5")
        if not dfi.empty:
            ax.scatter(
                dfi["attack_angle_deg"],
                dfi["middle_pause_disp_mag_from_initial"],
                s=58,
                color=color,
                alpha=0.36,
                edgecolors="none",
                label="runs",
            )
        if not avgi.empty:
            yerr = pd.to_numeric(avgi["std_middle_pause_disp_mag_from_initial"], errors="coerce").fillna(0.0) if "std_middle_pause_disp_mag_from_initial" in avgi.columns else pd.Series(np.zeros(len(avgi)))
            ax.errorbar(
                avgi["attack_angle_deg"],
                avgi["mean_middle_pause_disp_mag_from_initial"],
                yerr=yerr,
                linewidth=2.2,
                marker="o",
                markersize=4.8,
                alpha=0.95,
                color=color,
                capsize=3.5,
                label="repeat mean",
            )
        _style_axes(
            ax,
            theme,
            title=f"{dataset_label} | B-attack sweep\nMiddle-pause displacement vs attack angle",
            xlabel="B curl attack angle (deg)",
            ylabel=f"{mag_label} from initial pose",
        )
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            leg = ax.legend(loc="best", frameon=True, fontsize=9)
            _style_legend(leg, theme)
    fig.tight_layout()
    _save_plot_outputs(
        fig,
        os.path.join(output_dir, "b_attack_middle_pause_displacement_vs_attack_angle"),
        theme,
    )
    plt.close(fig)


def build_speed_summary(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    grouped = (
        summary_df.groupby(
            [
                "dataset_label",
                "run_type",
                "orientation_name",
                "orientation_deg",
                "speed_mm_min",
                "analysis_units",
                "repeat_count",
            ],
            as_index=False,
        )
        .agg(
            n_runs=("folder_name", "count"),
            mean_of_mean_ss_disp_mag=("mean_ss_disp_mag", "mean"),
            std_of_mean_ss_disp_mag=("mean_ss_disp_mag", "std"),
            mean_of_mean_ss_disp_x=("mean_ss_disp_x", "mean"),
            mean_of_mean_ss_disp_y=("mean_ss_disp_y", "mean"),
            mean_of_middle_pause_disp_mag_from_initial=("middle_pause_disp_mag_from_initial", "mean"),
            mean_of_initial_to_first_motion_disp_mag=("initial_to_first_motion_disp_mag", "mean"),
            std_of_initial_to_first_motion_disp_mag=("initial_to_first_motion_disp_mag", "std"),
            mean_of_final_disp_mag=("final_disp_mag", "mean"),
            std_of_final_disp_mag=("final_disp_mag", "std"),
            mean_of_max_disp_mag=("max_disp_mag", "mean"),
        )
        .sort_values(["dataset_label", "run_type", "orientation_name", "speed_mm_min"])
        .reset_index(drop=True)
    )
    return grouped


def infer_fixed_b_key_from_path(path_text: str) -> Optional[str]:
    txt = str(path_text)
    for key in FIXED_B_SERIES.keys():
        if f"/{key}/" in txt or txt.endswith(f"/{key}") or f"\\{key}\\" in txt:
            return key
    return None


def attach_fixed_b_metadata(df: pd.DataFrame, fixed_b_key: str) -> pd.DataFrame:
    out = df.copy()
    meta = FIXED_B_SERIES.get(fixed_b_key, {})
    out["fixed_b_key"] = fixed_b_key
    out["curl_angle_deg"] = float(meta.get("curl_angle_deg", np.nan))
    out["curl_angle_label"] = str(meta.get("label", fixed_b_key))
    out["series_color"] = str(meta.get("color", "#3d70b2"))
    return out


def load_existing_analysis_frames(project_dir: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    analysis_root = os.path.join(
        os.path.abspath(project_dir),
        "processed_image_data_folder",
        "tip_tracking_analysis",
    )
    if not os.path.isdir(analysis_root):
        raise FileNotFoundError(f"tip_tracking_analysis not found: {analysis_root}")

    per_run_frames: List[pd.DataFrame] = []
    per_speed_frames: List[pd.DataFrame] = []

    for fixed_b_key in FIXED_B_SERIES.keys():
        subdir = os.path.join(analysis_root, fixed_b_key)
        run_summary_path = os.path.join(subdir, RUN_SUMMARY_CSV_NAME)
        if not os.path.isfile(run_summary_path):
            continue
        run_df = pd.read_csv(run_summary_path)
        per_run_frames.append(attach_fixed_b_metadata(run_df, fixed_b_key))

        speed_summary_path = os.path.join(subdir, SPEED_SUMMARY_CSV_NAME)
        if os.path.isfile(speed_summary_path):
            speed_df = pd.read_csv(speed_summary_path)
        else:
            speed_df = build_speed_summary(run_df)
        per_speed_frames.append(attach_fixed_b_metadata(speed_df, fixed_b_key))

    if not per_run_frames:
        run_summary_path = os.path.join(analysis_root, RUN_SUMMARY_CSV_NAME)
        if not os.path.isfile(run_summary_path):
            raise FileNotFoundError(
                f"No reusable run_summary.csv found under: {analysis_root}"
            )
        run_df = pd.read_csv(run_summary_path)
        key = infer_fixed_b_key_from_path(run_summary_path) or "fixed_b_0"
        per_run_frames.append(attach_fixed_b_metadata(run_df, key))
        speed_summary_path = os.path.join(analysis_root, SPEED_SUMMARY_CSV_NAME)
        if os.path.isfile(speed_summary_path):
            speed_df = pd.read_csv(speed_summary_path)
        else:
            speed_df = build_speed_summary(run_df)
        per_speed_frames.append(attach_fixed_b_metadata(speed_df, key))

    per_run_df = pd.concat(per_run_frames, ignore_index=True)
    per_speed_df = pd.concat(per_speed_frames, ignore_index=True)
    return per_run_df, per_speed_df


def make_fixed_b_overlay_summary_plot(
    per_run_df: pd.DataFrame,
    per_speed_df: pd.DataFrame,
    output_dir: str,
    theme: str = "light",
) -> None:
    if per_run_df.empty or per_speed_df.empty:
        return

    output_dir = ensure_dir(output_dir)
    units_mode = per_run_df["analysis_units"].mode()
    units = str(units_mode.iloc[0]) if not units_mode.empty else "px"
    mag_label = f"|Δtip| ({units})"

    dataset_order = [
        label for label in ["forward", "backward", "transverse"]
        if label in set(per_run_df["dataset_label"].astype(str).str.lower())
    ]
    for label in sorted(set(per_run_df["dataset_label"].astype(str).str.lower())):
        if label not in dataset_order:
            dataset_order.append(label)

    fig, axes = plt.subplots(
        1,
        max(1, len(dataset_order)),
        figsize=(4.8 * max(1, len(dataset_order)), 4.0),
        squeeze=False,
        sharey=False,
    )
    fig.patch.set_facecolor("white")
    fig.patch.set_alpha(1.0)

    for col_idx, dataset_label in enumerate(dataset_order):
        ax = axes[0, col_idx]
        for fixed_b_key in ["fixed_b_0", "fixed_b_m3p2", "fixed_b_m5p1"]:
            meta = FIXED_B_SERIES[fixed_b_key]
            dfi = per_run_df[
                (per_run_df["dataset_label"].astype(str).str.lower() == dataset_label)
                & (per_run_df["fixed_b_key"].astype(str) == fixed_b_key)
            ].copy().sort_values("speed_mm_min")
            avgi = per_speed_df[
                (per_speed_df["dataset_label"].astype(str).str.lower() == dataset_label)
                & (per_speed_df["fixed_b_key"].astype(str) == fixed_b_key)
            ].copy().sort_values("speed_mm_min")
            if not dfi.empty:
                ax.scatter(
                    dfi["speed_mm_min"],
                    dfi["mean_ss_disp_mag"],
                    s=34,
                    color=meta["color"],
                    alpha=0.28,
                    edgecolors="none",
                    label=None,
                )
            if not avgi.empty:
                yerr = pd.to_numeric(avgi["std_of_mean_ss_disp_mag"], errors="coerce").fillna(0.0)
                ax.errorbar(
                    avgi["speed_mm_min"],
                    avgi["mean_of_mean_ss_disp_mag"],
                    yerr=yerr,
                    color=meta["color"],
                    linewidth=1.8,
                    marker="o",
                    markersize=4.2,
                    capsize=3.2,
                    alpha=0.97,
                    label=f"{meta['label']} curl",
                )
        _style_axes(
            ax,
            theme,
            title=f"{dataset_label}",
            xlabel="Bath speed (mm/min)",
            ylabel=f"{mag_label} | steady state" if col_idx == 0 else "",
        )
        leg = ax.legend(loc="best", frameon=True, fontsize=7.0)
        _style_legend(leg, theme)

    fig.tight_layout()
    _save_plot_outputs(
        fig,
        os.path.join(output_dir, "steady_state_displacement_vs_speed_overlay_by_fixed_b"),
        theme,
    )
    plt.close(fig)

    for dataset_label in dataset_order:
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        fig.patch.set_facecolor("white")
        fig.patch.set_alpha(1.0)
        for fixed_b_key in ["fixed_b_0", "fixed_b_m3p2", "fixed_b_m5p1"]:
            meta = FIXED_B_SERIES[fixed_b_key]
            dfi = per_run_df[
                (per_run_df["dataset_label"].astype(str).str.lower() == dataset_label)
                & (per_run_df["fixed_b_key"].astype(str) == fixed_b_key)
            ].copy().sort_values("speed_mm_min")
            avgi = per_speed_df[
                (per_speed_df["dataset_label"].astype(str).str.lower() == dataset_label)
                & (per_speed_df["fixed_b_key"].astype(str) == fixed_b_key)
            ].copy().sort_values("speed_mm_min")
            if not dfi.empty:
                ax.scatter(
                    dfi["speed_mm_min"],
                    dfi["mean_ss_disp_mag"],
                    s=34,
                    color=meta["color"],
                    alpha=0.28,
                    edgecolors="none",
                    label=None,
                )
            if not avgi.empty:
                yerr = pd.to_numeric(avgi["std_of_mean_ss_disp_mag"], errors="coerce").fillna(0.0)
                ax.errorbar(
                    avgi["speed_mm_min"],
                    avgi["mean_of_mean_ss_disp_mag"],
                    yerr=yerr,
                    color=meta["color"],
                    linewidth=1.8,
                    marker="o",
                    markersize=4.2,
                    capsize=3.2,
                    alpha=0.97,
                    label=f"{meta['label']} curl",
                )
        _style_axes(
            ax,
            theme,
            title=f"{dataset_label}",
            xlabel="Bath speed (mm/min)",
            ylabel=f"{mag_label} | steady state",
        )
        leg = ax.legend(loc="best", frameon=True, fontsize=7.0)
        _style_legend(leg, theme)
        fig.tight_layout()
        _save_plot_outputs(
            fig,
            os.path.join(output_dir, f"steady_state_displacement_vs_speed_overlay_{dataset_label}"),
            theme,
        )
        plt.close(fig)


def make_plunge_and_bath_plots(
    summary_df: pd.DataFrame,
    output_dir: str,
    theme: str = "light",
) -> None:
    if summary_df.empty:
        return

    output_dir = ensure_dir(output_dir)
    averaged_df = build_speed_summary(summary_df)
    units_mode = summary_df["analysis_units"].mode()
    units = str(units_mode.iloc[0]) if not units_mode.empty else "px"
    ylabel = f"|Δtip| ({units})"
    dataset_order = [
        label for label in ["forward", "backward", "transverse"]
        if label in set(summary_df["dataset_label"].astype(str).str.lower())
    ]
    for label in sorted(set(summary_df["dataset_label"].astype(str).str.lower())):
        if label not in dataset_order:
            dataset_order.append(label)

    curve_specs = [
        {
            "label": "Plunge Reset",
            "run_type": "direct",
            "run_col": "mean_ss_disp_mag",
            "mean_col": "mean_of_mean_ss_disp_mag",
            "std_col": "std_of_mean_ss_disp_mag",
            "color": "#3d70b2",
        },
        {
            "label": "Bath start-stop",
            "run_type": "two_stage",
            "run_col": "mean_ss_disp_mag",
            "mean_col": "mean_of_mean_ss_disp_mag",
            "std_col": "std_of_mean_ss_disp_mag",
            "color": "#c83f3f",
        },
    ]

    fig, axes = plt.subplots(
        1,
        max(1, len(dataset_order)),
        figsize=(4.8 * max(1, len(dataset_order)), 4.0),
        squeeze=False,
        sharey=True,
    )
    fig.patch.set_facecolor("white" if theme == "light" else "none")
    fig.patch.set_alpha(1.0 if theme == "light" else 0.0)

    for col_idx, dataset_label in enumerate(dataset_order):
        ax = axes[0, col_idx]
        for spec in curve_specs:
            dfi = summary_df[
                (summary_df["dataset_label"].astype(str).str.lower() == dataset_label)
                & (summary_df["run_type"].astype(str).str.lower() == spec["run_type"])
            ].copy().sort_values("speed_mm_min")
            avgi = averaged_df[
                (averaged_df["dataset_label"].astype(str).str.lower() == dataset_label)
                & (averaged_df["run_type"].astype(str).str.lower() == spec["run_type"])
            ].copy().sort_values("speed_mm_min")
            run_vals = pd.to_numeric(dfi[spec["run_col"]], errors="coerce")
            run_speed = pd.to_numeric(dfi["speed_mm_min"], errors="coerce")
            mask = run_speed.notna() & run_vals.notna()
            if mask.any():
                ax.scatter(
                    run_speed[mask],
                    run_vals[mask],
                    s=32,
                    color=spec["color"],
                    alpha=0.22,
                    edgecolors="none",
                    label=None,
                )
            mean_vals = pd.to_numeric(avgi[spec["mean_col"]], errors="coerce")
            std_vals = pd.to_numeric(avgi[spec["std_col"]], errors="coerce").fillna(0.0)
            speed_vals = pd.to_numeric(avgi["speed_mm_min"], errors="coerce")
            mask = speed_vals.notna() & mean_vals.notna()
            if mask.any():
                ax.errorbar(
                    speed_vals[mask],
                    mean_vals[mask],
                    yerr=std_vals[mask],
                    color=spec["color"],
                    linewidth=1.8,
                    marker="o",
                    markersize=4.2,
                    capsize=3.2,
                    alpha=0.97,
                    label=spec["label"],
                )
        _style_axes(
            ax,
            theme,
            title=f"{dataset_label}",
            xlabel="Bath speed (mm/min)",
            ylabel=f"{ylabel}" if col_idx == 0 else "",
        )
        leg = ax.legend(loc="best", frameon=True, fontsize=7.0)
        _style_legend(leg, theme)

    fig.tight_layout()
    _save_plot_outputs(
        fig,
        os.path.join(output_dir, "plunge_reset_and_bath_start_stop_by_dataset"),
        theme,
    )
    plt.close(fig)

    for dataset_label in dataset_order:
        fig, ax = plt.subplots(figsize=(4.8, 4.0))
        fig.patch.set_facecolor("white" if theme == "light" else "none")
        fig.patch.set_alpha(1.0 if theme == "light" else 0.0)
        for spec in curve_specs:
            dfi = summary_df[
                (summary_df["dataset_label"].astype(str).str.lower() == dataset_label)
                & (summary_df["run_type"].astype(str).str.lower() == spec["run_type"])
            ].copy().sort_values("speed_mm_min")
            avgi = averaged_df[
                (averaged_df["dataset_label"].astype(str).str.lower() == dataset_label)
                & (averaged_df["run_type"].astype(str).str.lower() == spec["run_type"])
            ].copy().sort_values("speed_mm_min")
            run_vals = pd.to_numeric(dfi[spec["run_col"]], errors="coerce")
            run_speed = pd.to_numeric(dfi["speed_mm_min"], errors="coerce")
            mask = run_speed.notna() & run_vals.notna()
            if mask.any():
                ax.scatter(
                    run_speed[mask],
                    run_vals[mask],
                    s=32,
                    color=spec["color"],
                    alpha=0.22,
                    edgecolors="none",
                    label=None,
                )
            mean_vals = pd.to_numeric(avgi[spec["mean_col"]], errors="coerce")
            std_vals = pd.to_numeric(avgi[spec["std_col"]], errors="coerce").fillna(0.0)
            speed_vals = pd.to_numeric(avgi["speed_mm_min"], errors="coerce")
            mask = speed_vals.notna() & mean_vals.notna()
            if mask.any():
                ax.errorbar(
                    speed_vals[mask],
                    mean_vals[mask],
                    yerr=std_vals[mask],
                    color=spec["color"],
                    linewidth=1.8,
                    marker="o",
                    markersize=4.2,
                    capsize=3.2,
                    alpha=0.97,
                    label=spec["label"],
                )
        _style_axes(
            ax,
            theme,
            title=f"{dataset_label}",
            xlabel="Bath speed (mm/min)",
            ylabel=ylabel,
        )
        leg = ax.legend(loc="best", frameon=True, fontsize=7.0)
        _style_legend(leg, theme)
        fig.tight_layout()
        _save_plot_outputs(
            fig,
            os.path.join(output_dir, f"plunge_reset_and_bath_start_stop_{dataset_label}"),
            theme,
        )
        plt.close(fig)


# =========================
# ARGPARSE / MAIN
# =========================
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Process all run folders inside raw_image_data_folder, track tip displacement "
            "using the CTR shadow calibration pipeline plus the CNN tip refiner, save "
            "annotated tip outputs, and generate steady-state displacement summaries for "
            "direct and two-stage deflection runs across C0/C180/C90 orientations."
        )
    )

    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument(
        "--project-dir",
        type=str,
        default=None,
        help="Project directory containing raw_image_data_folder and processed_image_data_folder.",
    )
    source_group.add_argument(
        "--raw-root",
        type=str,
        default=None,
        help="Direct path to raw_image_data_folder.",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to save outputs. Default: <project>/processed_image_data_folder/tip_tracking_analysis",
    )

    parser.add_argument(
        "--camera-calibration-file",
        type=str,
        default=CAMERA_CALIBRATION_FILE,
        help="Optional camera calibration .npz for mm conversion.",
    )
    parser.add_argument(
        "--board-reference-image",
        type=str,
        default=BOARD_REFERENCE_IMAGE,
        help="Optional board reference image for mm conversion.",
    )

    parser.add_argument(
        "--threshold",
        type=int,
        default=THRESHOLD,
        help=f"Binary threshold when not using exact class thresholding (default: {THRESHOLD})",
    )
    parser.add_argument(
        "--use-exact-class-thresholding",
        action="store_true",
        default=USE_EXACT_CLASS_THRESHOLDING,
        help="Use Otsu thresholding to match the original CTR class behavior.",
    )
    parser.add_argument(
        "--no-exact-class-thresholding",
        action="store_false",
        dest="use_exact_class_thresholding",
        help="Disable Otsu thresholding and use the fixed --threshold value.",
    )

    parser.add_argument(
        "--crop-json",
        type=str,
        default=None,
        help="Load a saved analysis crop JSON with crop_width_min/max and crop_height_min/max.",
    )
    parser.add_argument(
        "--crop-xywh",
        nargs=4,
        type=int,
        default=None,
        help="Crop as x y w h in image coordinates.",
    )
    parser.add_argument(
        "--manual-crop",
        action="store_true",
        default=MANUAL_CROP_ADJUSTMENT,
        help="Select crop interactively from the first image if no crop is provided.",
    )
    parser.add_argument(
        "--tip-refiner-model",
        type=str,
        default=DEFAULT_TIP_REFINER_MODEL,
        help="Optional CNN tip refiner model. When present, it can replace the selected tip.",
    )
    parser.add_argument(
        "--tip-refiner-anchor",
        type=str,
        default=DEFAULT_TIP_REFINER_ANCHOR,
        choices=["coarse", "selected", "refined"],
        help="Patch anchor used for the CNN tip refiner. Defaults to the checkpoint anchor.",
    )
    parser.add_argument(
        "--tip-refiner-compare-only",
        action="store_true",
        default=DEFAULT_TIP_REFINER_COMPARE_ONLY,
        help="Run the CNN refiner for debug only and keep the classical selected tip.",
    )
    parser.add_argument(
        "--reuse-existing-analysis",
        action="store_true",
        help="Skip image analysis and rebuild plots from existing tip_tracking_analysis CSV summaries.",
    )
    parser.add_argument(
        "--plot-theme",
        choices=["dark", "light"],
        default="light",
        help="Summary plot theme. Default: light.",
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.plot_theme == "light":
        set_science_light_style()

    raw_root = resolve_raw_root(project_dir=args.project_dir, raw_root=args.raw_root)
    output_root = infer_output_root(args.project_dir, raw_root, args.output_dir)
    summary_plot_dir = themed_summary_plot_dir(output_root, args.plot_theme)

    if bool(args.reuse_existing_analysis):
        if args.project_dir is None:
            raise ValueError("--reuse-existing-analysis currently requires --project-dir.")
        summary_df, speed_summary_df = load_existing_analysis_frames(args.project_dir)
        analysis_root = os.path.join(
            os.path.abspath(args.project_dir),
            "processed_image_data_folder",
            "tip_tracking_analysis",
        )
        has_fixed_b_subdirs = any(
            os.path.isdir(os.path.join(analysis_root, key)) for key in FIXED_B_SERIES.keys()
        )
        if has_fixed_b_subdirs:
            for fixed_b_key in sorted(summary_df["fixed_b_key"].astype(str).unique(), key=lambda x: FIXED_B_SERIES.get(x, {}).get("curl_angle_deg", 0.0)):
                per_fixed_dir = ensure_dir(os.path.join(summary_plot_dir, fixed_b_key))
                per_fixed_df = summary_df[summary_df["fixed_b_key"].astype(str) == fixed_b_key].copy()
                make_summary_plots(per_fixed_df, per_fixed_dir, theme=str(args.plot_theme))
            make_fixed_b_overlay_summary_plot(summary_df, speed_summary_df, summary_plot_dir, theme=str(args.plot_theme))
        else:
            make_summary_plots(summary_df, summary_plot_dir, theme=str(args.plot_theme))
            make_plunge_and_bath_plots(summary_df, summary_plot_dir, theme=str(args.plot_theme))
        print("\n[INFO] Reused existing analysis summaries only; images were not reprocessed.")
        print(f"  - project_dir: {os.path.abspath(args.project_dir)}")
        print(f"  - summary_plots_dir: {summary_plot_dir}")
        return

    run_infos = find_run_folders(raw_root)
    if not run_infos:
        raise RuntimeError(f"No valid run folders with capture_log.csv found in: {raw_root}")

    print("[INFO] Found run folders:")
    for run_info in run_infos:
        print(
            f"  - {run_info.folder_name} | speed={run_info.speed_mm_min:.3f} mm/min | "
            f"dataset={run_info.dataset_label} | run_type={run_info.run_type} | "
            f"orientation={run_info.orientation_name}"
        )

    analysis_crop = load_or_create_analysis_crop(
        run_infos=run_infos,
        output_root=output_root,
        crop_json=args.crop_json,
        crop_xywh=args.crop_xywh,
        manual_crop=args.manual_crop,
    )

    cal = prepare_calibration_object(
        parent_directory=output_root,
        analysis_crop=analysis_crop,
        camera_calibration_file=args.camera_calibration_file,
        board_reference_image=args.board_reference_image,
        tip_refiner_model=args.tip_refiner_model,
        tip_refiner_anchor=args.tip_refiner_anchor,
        tip_refiner_compare_only=bool(args.tip_refiner_compare_only),
    )

    all_results: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, object]] = []

    for run_info in run_infos:
        results_df, summary_row = process_single_run_folder(
            run_info=run_info,
            cal=cal,
            output_root=output_root,
            threshold=args.threshold,
            use_exact_class_thresholding=args.use_exact_class_thresholding,
        )
        all_results.append(results_df)
        summary_rows.append(summary_row)

    summary_df = (
        pd.DataFrame(summary_rows)
        .sort_values(["orientation_deg", "dataset_label", "run_type", "speed_mm_min", "folder_name"])
        .reset_index(drop=True)
    )
    summary_csv_path = os.path.join(output_root, RUN_SUMMARY_CSV_NAME)
    summary_df.to_csv(summary_csv_path, index=False)

    all_results_df = pd.concat(all_results, ignore_index=True) if all_results else pd.DataFrame()
    all_results_csv_path = os.path.join(output_root, ALL_RESULTS_CSV_NAME)
    if not all_results_df.empty:
        all_results_df.to_csv(all_results_csv_path, index=False)

    speed_summary_df = build_speed_summary(summary_df)
    speed_summary_csv_path = os.path.join(output_root, SPEED_SUMMARY_CSV_NAME)
    if not speed_summary_df.empty:
        speed_summary_df.to_csv(speed_summary_csv_path, index=False)

    make_summary_plots(summary_df, summary_plot_dir, theme=str(args.plot_theme))

    print("\n" + "=" * 80)
    print("[INFO] Average tip displacement during steady state")
    print("=" * 80)
    for _, row in summary_df.iterrows():
        avg_ss = row["mean_ss_disp_mag"]
        units = row["analysis_units"]
        if pd.isna(avg_ss):
            avg_ss_str = "nan (no ss frames)"
        else:
            avg_ss_str = f"{float(avg_ss):.6f} {units}"
        print(
            f"{row['folder_name']}: "
            f"speed={float(row['speed_mm_min']):.3f} mm/min | "
            f"dataset={row['dataset_label']} | "
            f"run_type={row['run_type']} | "
            f"ref={row['ss_reference']} | "
            f"avg_ss_disp={avg_ss_str}"
        )

    if not speed_summary_df.empty:
        print("\n" + "=" * 80)
        print("[INFO] Mean steady-state displacement by speed, dataset, and run type")
        print("=" * 80)
        for _, row in speed_summary_df.iterrows():
            print(
                f"{row['dataset_label']} | {row['run_type']} | "
                f"speed={float(row['speed_mm_min']):.3f} mm/min | "
                f"mean(mean_ss_disp_mag)={float(row['mean_of_mean_ss_disp_mag']):.6f} {row['analysis_units']} | "
                f"n_runs={int(row['n_runs'])}"
            )

    print("\n[INFO] Output summary")
    print(f"  - raw_root: {raw_root}")
    print(f"  - output_root: {output_root}")
    print(f"  - run_summary_csv: {summary_csv_path}")
    if not all_results_df.empty:
        print(f"  - all_results_csv: {all_results_csv_path}")
    if not speed_summary_df.empty:
        print(f"  - speed_summary_csv: {speed_summary_csv_path}")
    print(f"  - summary_plots_dir: {summary_plot_dir}")
    print(f"  - crop_json: {os.path.join(output_root, 'analysis_crop.json')}")

    print("\n[INFO] Per-run outputs")
    for run_info in run_infos:
        run_output_dir = os.path.join(output_root, RUN_OUTPUT_SUBDIR, run_info.folder_name)
        print(f"  - {run_info.folder_name}: {run_output_dir}")


if __name__ == "__main__":
    main()
