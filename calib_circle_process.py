#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
offline_run_checkerboard_circle_error_analysis.py

Offline CTR shadow calibration runner that:
  1) Opens a project/raw image folder
  2) Lets you choose crop bounds from the FIRST raw image
  3) Uses a checkerboard reference image + camera calibration to define mm axes
  4) Runs analyze_data_batch + existing tracking pipeline
  5) Converts tracked tip pixels to checkerboard-referenced mm coordinates
  6) Computes a reference tracked point = mean of all valid tracked points
  7) Computes per-sample error distance to that mean (in mm)
  8) Saves:
       - CSV of tracked points and errors
       - JSON metrics summary
       - error-vs-sample plot
       - desired-vs-actual circle plot
       - histogram of error distance vs number of samples
       - optional overlay on checkerboard image
       - optional desired lookup from planned_command_sequence.csv when available

Key outputs:
  processed_image_data_folder/
    tracked_tip_positions_mm.csv
    tracked_tip_error_metrics.json
    tracked_tip_error_vs_sample.png
    tracked_tip_error_histogram.png
    checkerboard_reference_annotated_analysis.png   (optional)

Usage example:
  python3 offline_run_checkerboard_circle_error_analysis.py \
    --project_dir "/path/to/project" \
    --camera_calibration_file "/path/to/camera_calibration.npz" \
    --checkerboard_reference_image "/path/to/checkerboard_ref.png" \
    --threshold 200 \
    --save_plots \
    --tip_refine_mode edge_dt

Important:
  - This script patches the filename parser used by shadow_calibration so that files named like:
      00134_tracked_X90.604_Y31.876_Z-173.537_B-3.368_C-52.800_...
    are parsed correctly.
  - It uses X as the "ntnl_pos" placeholder and B as the "ss_pos" placeholder because
    the existing pipeline expects a 3-value tuple from filename parsing.
"""

import argparse
import csv
import json
import os
import re
import shutil
import sys
import types
from heapq import heappop, heappush
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import math
import numpy as np

# Add path to your shadow_calibration script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

import shadow_calibration as shadow_calibration_module  # noqa: E402
from shadow_calibration import CTR_Shadow_Calibration  # noqa: E402

IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
_NEI8_W = [
    (-1, -1, 2 ** 0.5), (-1, 0, 1.0), (-1, 1, 2 ** 0.5),
    (0, -1, 1.0),                             (0, 1, 1.0),
    (1, -1, 2 ** 0.5),  (1, 0, 1.0),  (1, 1, 2 ** 0.5),
]
CIRCLE_CENTER_X_DEFAULT = 100.0
CIRCLE_CENTER_Z_DEFAULT = -125.0
CIRCLE_RADIUS_DEFAULT = 18.0
CIRCLE_SAMPLES_PER_QUARTER_DEFAULT = 50
CIRCLE_RECORDED_PHASES = {"q1_pull", "q2_release", "q3_pull", "q4_release"}
LEGACY_STAR_RECORDED_PHASES = {"right", "left"}
_TRACKED_SAMPLE_RE = re.compile(
    r"(?:^|_)(\d{5})_(q1_pull_start|q1_pull|pull_to_release_1|q2_release_start|q2_release|midpoint_c_flip|q3_pull_start|q3_pull|pull_to_release_2|q4_release_start|q4_release|final_recenter|right_start|right|mirror_flip|left|tracked)(?:_|$)"
)


# -----------------------------
# Utilities: IO and discovery
# -----------------------------
def list_images(folder: Path) -> List[Path]:
    imgs = [p for p in folder.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTS]
    imgs.sort()
    return imgs


def ensure_project_from_raw(raw_dir: Path, project_dir: Path, link_mode: str = "symlink") -> Path:
    """
    Create a project_dir with raw_image_data_folder containing images from raw_dir.
    Returns raw_image_data_folder path.
    """
    raw_dir = Path(raw_dir).expanduser().resolve()
    project_dir = Path(project_dir).expanduser().resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    raw_out = project_dir / "raw_image_data_folder"
    raw_out.mkdir(parents=True, exist_ok=True)

    imgs = list_images(raw_dir)
    if not imgs:
        raise RuntimeError(f"No images found in raw_dir: {raw_dir}")

    for src in imgs:
        dst = raw_out / src.relative_to(raw_dir)
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if link_mode == "symlink":
            try:
                os.symlink(src.resolve(), dst)
            except Exception:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)

    return raw_out


def _json_ready(value: Any):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    return value


def load_optional_tip_refiner(cal: CTR_Shadow_Calibration, args) -> None:
    if not getattr(args, "tip_refiner_model", None):
        return
    model_path = Path(args.tip_refiner_model).expanduser().resolve()
    if not model_path.exists():
        raise FileNotFoundError(f"Tip refiner model not found: {model_path}")
    cal.load_tip_refiner_model(
        str(model_path),
        anchor_name=args.tip_refiner_anchor,
        use_as_selected=(not bool(args.tip_refiner_compare_only)),
    )


def select_tracked_rows_for_analysis(cal: CTR_Shadow_Calibration, args) -> Tuple[np.ndarray, str]:
    source = str(getattr(args, "tracked_tip_source", "auto")).strip().lower()
    if source == "auto":
        if getattr(args, "tip_refiner_model", None) and not bool(getattr(args, "tip_refiner_compare_only", False)):
            source = "selected"
        else:
            source = "coarse"

    if source == "cnn":
        arr = getattr(cal, "tip_locations_array_cnn", None)
        if arr is None:
            raise RuntimeError("tracked_tip_source=cnn requested, but tip_locations_array_cnn is unavailable.")
        return np.asarray(arr, dtype=float), "cnn"

    if source == "selected":
        arr = getattr(cal, "tip_locations_array_selected", None)
        if arr is None:
            raise RuntimeError("tracked_tip_source=selected requested, but tip_locations_array_selected is unavailable.")
        return np.asarray(arr, dtype=float), "selected"

    if source == "coarse":
        arr = getattr(cal, "tip_locations_array_coarse", None)
        if arr is None:
            raise RuntimeError("tracked_tip_source=coarse requested, but tip_locations_array_coarse is unavailable.")
        return np.asarray(arr, dtype=float), "coarse"

    raise ValueError(f"Unsupported tracked_tip_source: {source}")


def _parse_inner_corners_arg(value):
    if value is None:
        return None
    if isinstance(value, (list, tuple)) and len(value) == 2:
        return int(value[0]), int(value[1])

    txt = str(value).strip().lower().replace("x", ",")
    parts = [p.strip() for p in txt.split(",") if p.strip()]
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Expected checkerboard inner corners as 'Nx,Ny' or 'NxXNy', got: {value}"
        )
    return int(parts[0]), int(parts[1])


def _undistort_points_to_image(points_xy: np.ndarray, camera_K: np.ndarray, camera_dist: np.ndarray) -> np.ndarray:
    pts = np.asarray(points_xy, dtype=np.float64).reshape(-1, 1, 2)
    undist = cv2.undistortPoints(pts, camera_K, camera_dist, P=camera_K)
    return undist.reshape(-1, 2)


def _scale_mm_from_px_homography(H: np.ndarray, mm_scale: float) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64).copy()
    S = np.diag([float(mm_scale), float(mm_scale), 1.0]).astype(np.float64)
    return S @ H


def _scale_px_from_mm_homography(H: np.ndarray, mm_scale: float) -> np.ndarray:
    H = np.asarray(H, dtype=np.float64).copy()
    if abs(float(mm_scale)) < 1e-12:
        raise ValueError("mm_scale must be non-zero.")
    S_inv = np.diag([1.0 / float(mm_scale), 1.0 / float(mm_scale), 1.0]).astype(np.float64)
    return H @ S_inv


# -----------------------------------------
# Filename parser patch for tracked_X..._B...
# -----------------------------------------
def robust_parse_positions_from_filename(file_name: str):
    """
    Supports filenames like:
      00134_tracked_X90.604_Y31.876_Z-173.537_B-3.368_C-52.800_20260319_113124_658346.png

    Returns:
      orientation, ntnl_pos, ss_pos

    Mapping used to keep compatibility with the existing pipeline:
      - orientation = inferred from side token / C value / legacy numeric token
      - ntnl_pos    = X value
      - ss_pos      = B value
    """
    base = os.path.splitext(os.path.basename(file_name))[0]
    parts = base.split("_")

    orientation = None
    side_token = None

    values = {}
    for p in parts:
        p_lower = p.lower()
        if p_lower in ("right", "left"):
            side_token = p_lower
        if len(p) >= 2 and p[0] in ("X", "Y", "Z", "B", "C"):
            key = p[0]
            try:
                values[key] = float(p[1:])
            except Exception:
                pass

    if side_token == "right":
        orientation = 0
    elif side_token == "left":
        orientation = 1

    if orientation is None and "C" in values:
        c_val = float(values["C"])
        c_norm = ((c_val + 180.0) % 360.0) - 180.0
        if abs(c_norm - 0.0) <= 5.0:
            orientation = 0
        elif abs(abs(c_norm) - 180.0) <= 5.0:
            orientation = 1
        elif abs(c_norm - 90.0) <= 5.0:
            orientation = 2
        elif abs(c_norm + 90.0) <= 5.0:
            orientation = 3

    if orientation is None and len(parts) > 0:
        try:
            candidate = int(parts[0])
            if candidate in (0, 1, 2, 3):
                orientation = candidate
        except Exception:
            pass

    if orientation is None:
        orientation = 0

    if "X" not in values:
        raise ValueError(f"Could not parse X value from filename: {file_name}")
    if "B" not in values:
        raise ValueError(f"Could not parse B value from filename: {file_name}")

    ntnl_pos = float(values["X"])
    ss_pos = float(values["B"])
    return orientation, ntnl_pos, ss_pos


def extract_named_values_from_filename(file_name: str) -> Dict[str, float]:
    """
    Extract numeric values from filename tokens such as X..., Y..., Z..., B..., C...
    """
    base = os.path.splitext(os.path.basename(file_name))[0]
    values: Dict[str, float] = {}
    for p in base.split("_"):
        if len(p) < 2 or p[0] not in ("X", "Y", "Z", "B", "C"):
            continue
        try:
            values[p[0]] = float(p[1:])
        except Exception:
            continue
    return values


def parse_tracked_sample_metadata(file_name: str) -> Tuple[Optional[int], Optional[str]]:
    m = _TRACKED_SAMPLE_RE.search(os.path.basename(file_name))
    if not m:
        return None, None
    return int(m.group(1)), str(m.group(2))


def _image_sort_key(path: Path):
    sample_idx, phase = parse_tracked_sample_metadata(path.name)
    if sample_idx is not None:
        return (0, int(sample_idx), str(phase), path.name)
    return (1, path.name)


def build_circle_tip_lookup(
    center_x: float,
    center_z: float,
    radius: float,
    samples_per_quarter: int,
    capture_at_start: bool = False,
    capture_every_n_circle_moves: int = 1,
) -> Dict[int, Dict[str, float]]:
    samples_per_quarter = int(max(2, samples_per_quarter))
    capture_every_n_circle_moves = max(1, int(capture_every_n_circle_moves))

    quarter_angles = [
        np.linspace(-0.5 * math.pi, 0.0, samples_per_quarter, endpoint=True),
        np.linspace(0.0, 0.5 * math.pi, samples_per_quarter, endpoint=True),
        np.linspace(0.5 * math.pi, math.pi, samples_per_quarter, endpoint=True),
        np.linspace(math.pi, 1.5 * math.pi, samples_per_quarter, endpoint=True),
    ]

    lookup: Dict[int, Dict[str, float]] = {}
    sample_idx = 0
    circle_move_counter = 0

    if bool(capture_at_start):
        a0 = float(quarter_angles[0][0])
        sample_idx += 1
        lookup[sample_idx] = {
            "desired_tip_x_mm": float(center_x + radius * math.cos(a0)),
            "desired_tip_z_mm": float(center_z + radius * math.sin(a0)),
        }

    for q_angles in quarter_angles:
        for ang in q_angles[1:]:
            circle_move_counter += 1
            if (circle_move_counter % capture_every_n_circle_moves) != 0:
                continue
            sample_idx += 1
            lookup[sample_idx] = {
                "desired_tip_x_mm": float(center_x + radius * math.cos(float(ang))),
                "desired_tip_z_mm": float(center_z + radius * math.sin(float(ang))),
            }

    return lookup


def build_circle_tip_lookup_from_planned_csv(
    planned_csv_path: Path,
    capture_at_start: bool = False,
    capture_every_n_circle_moves: int = 1,
) -> Dict[int, Dict[str, float]]:
    planned_csv_path = Path(planned_csv_path).expanduser().resolve()
    if not planned_csv_path.exists():
        raise FileNotFoundError(f"planned command CSV not found: {planned_csv_path}")

    rows = []
    with open(planned_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    if not rows:
        raise RuntimeError(f"planned command CSV is empty: {planned_csv_path}")

    lookup: Dict[int, Dict[str, float]] = {}
    sample_idx = 0
    circle_move_counter = 0
    capture_every_n_circle_moves = max(1, int(capture_every_n_circle_moves))

    first = rows[0]
    if bool(capture_at_start):
        sample_idx += 1
        lookup[sample_idx] = {
            "desired_tip_x_mm": float(first["tip_x"]),
            "desired_tip_z_mm": float(first["tip_z"]),
        }

    for row in rows[1:]:
        phase = str(row.get("phase", "")).strip()
        if phase not in CIRCLE_RECORDED_PHASES and phase not in LEGACY_STAR_RECORDED_PHASES:
            continue
        circle_move_counter += 1
        if (circle_move_counter % capture_every_n_circle_moves) != 0:
            continue
        sample_idx += 1
        lookup[sample_idx] = {
            "desired_tip_x_mm": float(row["tip_x"]),
            "desired_tip_z_mm": float(row["tip_z"]),
        }

    return lookup


def resolve_desired_tip_lookup(project_dir: Path, args) -> Dict[int, Dict[str, float]]:
    candidate_csvs = []
    if getattr(args, "planned_command_csv", None):
        candidate_csvs.append(Path(args.planned_command_csv).expanduser())
    candidate_csvs.append(Path(project_dir) / "planned_command_sequence.csv")

    for candidate in candidate_csvs:
        try:
            if candidate.exists():
                lookup = build_circle_tip_lookup_from_planned_csv(
                    candidate,
                    capture_at_start=bool(args.capture_at_start),
                    capture_every_n_circle_moves=int(args.capture_every_n_circle_moves),
                )
                print(f"[INFO] Using planned command CSV for desired lookup: {candidate}")
                return lookup
        except Exception as e:
            print(f"[WARN] Failed to use planned command CSV {candidate}: {e}")

    print("[INFO] planned_command_sequence.csv not found; using ideal circle lookup fallback.")
    return build_circle_tip_lookup(
        center_x=float(args.circle_center_x_mm),
        center_z=float(args.circle_center_z_mm),
        radius=float(args.circle_radius_mm),
        samples_per_quarter=int(args.circle_samples_per_quarter),
        capture_at_start=bool(args.capture_at_start),
        capture_every_n_circle_moves=int(args.capture_every_n_circle_moves),
    )


def patch_filename_parser():
    """
    Patch shadow_calibration filename parsing so analyze_data_batch can read
    filenames like:
      00134_tracked_X90.604_Y31.876_Z-173.537_B-3.368_C-52.800_...
    """
    shadow_calibration_module._get_pos_from_file_name = robust_parse_positions_from_filename
    print("[INFO] Patched filename parser for tracked_X..._B... style filenames.")


def normalize_raw_filenames_for_shadow_parser(raw_folder: Path) -> List[Tuple[str, str]]:
    """
    Rename files in-place so shadow_calibration's legacy parser can read them.

    Legacy parser requirement:
      {orientation}_{X}_{B}_...
    """
    rename_pairs: List[Tuple[str, str]] = []

    for src in list_images(raw_folder):
        try:
            orientation, ntnl_pos, ss_pos = robust_parse_positions_from_filename(src.name)
        except Exception:
            continue

        base = src.stem
        legacy_prefix = f"{int(orientation)}_{float(ntnl_pos):.6f}_{float(ss_pos):.6f}_"
        if base.startswith(legacy_prefix):
            continue

        dst_name = f"{legacy_prefix}{base}{src.suffix}"
        dst = src.with_name(dst_name)
        counter = 1
        while dst.exists() and dst != src:
            dst_name = f"{legacy_prefix}{base}__{counter}{src.suffix}"
            dst = src.with_name(dst_name)
            counter += 1

        src.rename(dst)
        rename_pairs.append((src.name, dst.name))

    return rename_pairs


# -----------------------------------------
# Checkerboard reference correction helpers
# -----------------------------------------
def apply_checkerboard_reference_corrections(
    cal: CTR_Shadow_Calibration,
    mm_scale: float = 0.5,
    flip_planar_x: bool = True,
):
    """
    Correct checkerboard-backed calibration axes so they match the ruler-backed convention.

    Requested fixes:
      1) checkerboard mm values are 2x too large -> scale checkerboard mm conversion by 0.5
      2) checkerboard planar x sign is flipped vs motor -> negate planar x
    """
    if hasattr(cal, "apply_checkerboard_reference_corrections"):
        return cal.apply_checkerboard_reference_corrections(
            mm_scale=mm_scale,
            flip_planar_x=flip_planar_x,
        )

    if getattr(cal, "board_pose", None) is None:
        return cal

    correction_meta = {
        "checkerboard_mm_scale_correction": float(mm_scale),
        "checkerboard_planar_x_flipped": bool(flip_planar_x),
    }

    if getattr(cal, "board_mm_per_px_local", None) is not None:
        cal.board_mm_per_px_local = float(cal.board_mm_per_px_local) * float(mm_scale)
    if getattr(cal, "board_px_per_mm_local", None) is not None:
        if abs(float(mm_scale)) < 1e-12:
            raise ValueError("mm_scale must be non-zero.")
        cal.board_px_per_mm_local = float(cal.board_px_per_mm_local) / float(mm_scale)

    if getattr(cal, "board_homography_mm_from_px", None) is not None:
        cal.board_homography_mm_from_px = _scale_mm_from_px_homography(
            cal.board_homography_mm_from_px,
            mm_scale=float(mm_scale),
        )
    if getattr(cal, "board_homography_px_from_mm", None) is not None:
        cal.board_homography_px_from_mm = _scale_px_from_mm_homography(
            cal.board_homography_px_from_mm,
            mm_scale=float(mm_scale),
        )

    if isinstance(cal.board_pose, dict):
        bp = dict(cal.board_pose)
        if "board_homography_mm_from_px" in bp and bp["board_homography_mm_from_px"] is not None:
            bp["board_homography_mm_from_px"] = _scale_mm_from_px_homography(
                bp["board_homography_mm_from_px"], mm_scale=float(mm_scale)
            )
        if "board_homography_px_from_mm" in bp and bp["board_homography_px_from_mm"] is not None:
            bp["board_homography_px_from_mm"] = _scale_px_from_mm_homography(
                bp["board_homography_px_from_mm"], mm_scale=float(mm_scale)
            )
        if "board_mm_per_px_local" in bp and bp["board_mm_per_px_local"] is not None:
            bp["board_mm_per_px_local"] = float(bp["board_mm_per_px_local"]) * float(mm_scale)
        if "board_px_per_mm_local" in bp and bp["board_px_per_mm_local"] is not None:
            bp["board_px_per_mm_local"] = float(bp["board_px_per_mm_local"]) / float(mm_scale)
        bp.setdefault("corrections_applied", {})
        bp["corrections_applied"].update(correction_meta)
        cal.board_pose = bp

    cal.board_reference_correction_meta = correction_meta

    if hasattr(cal, "pixel_point_to_calibrated_axes"):
        original_pixel_point_to_calibrated_axes = cal.pixel_point_to_calibrated_axes

        def pixel_point_to_calibrated_axes_patched(self, *args, **kwargs):
            u_mm, z_mm = original_pixel_point_to_calibrated_axes(*args, **kwargs)
            u_mm = float(u_mm)
            z_mm = float(z_mm)
            if bool(flip_planar_x):
                u_mm = -u_mm
            return u_mm, z_mm

        cal.pixel_point_to_calibrated_axes = types.MethodType(
            pixel_point_to_calibrated_axes_patched, cal
        )

    if hasattr(cal, "calibrated_axes_to_pixel_point"):
        original_calibrated_axes_to_pixel_point = cal.calibrated_axes_to_pixel_point

        def calibrated_axes_to_pixel_point_patched(self, u_mm, z_mm, *args, **kwargs):
            u_in = -float(u_mm) if bool(flip_planar_x) else float(u_mm)
            return original_calibrated_axes_to_pixel_point(u_in, float(z_mm), *args, **kwargs)

        cal.calibrated_axes_to_pixel_point = types.MethodType(
            calibrated_axes_to_pixel_point_patched, cal
        )

    print(
        "[INFO] Applied checkerboard reference corrections: "
        f"mm_scale={float(mm_scale):.6f}, "
        f"flip_planar_x={bool(flip_planar_x)}"
    )
    return cal


def collect_board_reference_info(cal: CTR_Shadow_Calibration) -> Dict[str, Any]:
    return {
        "camera_calib_path": getattr(cal, "camera_calib_path", None),
        "camera_calib_meta": getattr(cal, "camera_calib_meta", None),
        "board_reference_image_path": getattr(cal, "board_reference_image_path", None),
        "board_pose": getattr(cal, "board_pose", None),
        "true_vertical_img_unit": getattr(cal, "true_vertical_img_unit", None),
        "board_homography_px_from_mm": getattr(cal, "board_homography_px_from_mm", None),
        "board_homography_mm_from_px": getattr(cal, "board_homography_mm_from_px", None),
        "board_px_per_mm_local": getattr(cal, "board_px_per_mm_local", None),
        "board_mm_per_px_local": getattr(cal, "board_mm_per_px_local", None),
        "board_reference_correction_meta": getattr(cal, "board_reference_correction_meta", None),
    }


# -------------------------------------------
# GUI: crop selection from one image
# -------------------------------------------
def interactive_crop_from_image(
    image_bgr: np.ndarray,
    default_crop=None,
    window_crop="Manual Crop Setup (OFFLINE)",
):
    """
    Returns analysis_crop dict in the format CTR_Shadow_Calibration expects.
    """
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image passed to interactive GUI.")

    img_h, img_w = image_bgr.shape[:2]

    if default_crop is None:
        default_crop = {
            "crop_width_min": 650,
            "crop_width_max": 2900,
            "crop_height_min": 150,
            "crop_height_max": 1750,
        }

    x_min = int(np.clip(default_crop["crop_width_min"], 0, img_w - 2))
    x_max = int(np.clip(default_crop["crop_width_max"], x_min + 1, img_w - 1))
    y_min = int(np.clip(img_h - default_crop["crop_height_max"], 0, img_h - 2))
    y_max = int(np.clip(img_h - default_crop["crop_height_min"], y_min + 1, img_h - 1))

    corners = {
        "tl": [x_min, y_min],
        "tr": [x_max, y_min],
        "br": [x_max, y_max],
        "bl": [x_min, y_max],
    }
    active_corner = {"name": None}
    drag_threshold_px = 30

    def nearest_corner(mx, my):
        best_name, best_dist = None, 1e18
        for name, (cx, cy) in corners.items():
            d = (mx - cx) ** 2 + (my - cy) ** 2
            if d < best_dist:
                best_dist = d
                best_name = name
        return best_name if best_dist <= drag_threshold_px**2 else None

    def clamp_rect():
        xs = [pt[0] for pt in corners.values()]
        ys = [pt[1] for pt in corners.values()]
        x0 = int(np.clip(min(xs), 0, img_w - 2))
        x1 = int(np.clip(max(xs), x0 + 1, img_w - 1))
        y0 = int(np.clip(min(ys), 0, img_h - 2))
        y1 = int(np.clip(max(ys), y0 + 1, img_h - 1))
        corners["tl"] = [x0, y0]
        corners["tr"] = [x1, y0]
        corners["br"] = [x1, y1]
        corners["bl"] = [x0, y1]

    def on_mouse_crop(event, mx, my, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            active_corner["name"] = nearest_corner(mx, my)
        elif event == cv2.EVENT_MOUSEMOVE and active_corner["name"] is not None:
            name = active_corner["name"]
            mx = int(np.clip(mx, 0, img_w - 1))
            my = int(np.clip(my, 0, img_h - 1))
            if name == "tl":
                corners["tl"] = [mx, my]
                corners["tr"][1] = my
                corners["bl"][0] = mx
            elif name == "tr":
                corners["tr"] = [mx, my]
                corners["tl"][1] = my
                corners["br"][0] = mx
            elif name == "br":
                corners["br"] = [mx, my]
                corners["bl"][1] = my
                corners["tr"][0] = mx
            elif name == "bl":
                corners["bl"] = [mx, my]
                corners["br"][1] = my
                corners["tl"][0] = mx
            clamp_rect()
        elif event == cv2.EVENT_LBUTTONUP:
            active_corner["name"] = None

    accepted_crop = False
    cv2.namedWindow(window_crop, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_crop, on_mouse_crop)

    print("\n[GUI] Manual crop setup (OFFLINE)")
    print("- Drag rectangle corners with left mouse.")
    print("- Press ENTER or SPACE to confirm.")
    print("- Press R to reset to defaults.")
    print("- Press Q or ESC to cancel (uses defaults).")

    try:
        while True:
            if cv2.getWindowProperty(window_crop, cv2.WND_PROP_VISIBLE) < 1:
                break

            clamp_rect()
            x0, y0 = corners["tl"]
            x1, y1 = corners["br"]

            disp = image_bgr.copy()
            cv2.rectangle(disp, (x0, y0), (x1, y1), (0, 255, 0), 2)
            for pt in corners.values():
                cv2.circle(disp, tuple(pt), 6, (0, 255, 255), -1)

            cv2.putText(
                disp,
                f"x:[{x0},{x1}] y:[{y0},{y1}]  (ENTER confirm | R reset | Q/ESC cancel)",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (255, 255, 255),
                2,
            )

            cv2.imshow(window_crop, disp)
            key = cv2.waitKey(20) & 0xFF

            if key in (13, 32):
                accepted_crop = True
                break
            if key in (27, ord("q")):
                break
            if key in (ord("r"), ord("R")):
                corners["tl"] = [x_min, y_min]
                corners["tr"] = [x_max, y_min]
                corners["br"] = [x_max, y_max]
                corners["bl"] = [x_min, y_max]
    finally:
        cv2.setMouseCallback(window_crop, lambda *args: None)
        cv2.destroyWindow(window_crop)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    if accepted_crop:
        x0, y0 = corners["tl"]
        x1, y1 = corners["br"]
        analysis_crop = {
            "crop_width_min": int(x0),
            "crop_width_max": int(x1),
            "crop_height_min": int(img_h - y1),
            "crop_height_max": int(img_h - y0),
        }
        print(f"[GUI] Selected analysis_crop: {analysis_crop}")
    else:
        analysis_crop = dict(default_crop)
        print(f"[GUI] Crop cancelled, using default analysis_crop: {analysis_crop}")

    return analysis_crop


# ------------------------------------------
# Tip refinement helpers
# ------------------------------------------
def _tip_angle_to_direction_xy(tip_angle_deg: float) -> np.ndarray:
    ang = np.deg2rad(float(tip_angle_deg))
    vx = float(np.sin(ang))
    vy = float(np.cos(ang))
    d = np.array([vx, vy], dtype=np.float64)
    n = float(np.linalg.norm(d))
    return d / max(n, 1e-12)


def _inside_mask(mask_fg: np.ndarray, xy: np.ndarray) -> bool:
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    h, w = mask_fg.shape[:2]
    return (0 <= x < w) and (0 <= y < h) and (mask_fg[y, x] > 0)


def _backtrack_point_inside_fg(mask_fg: np.ndarray, p0_xy: np.ndarray, dir_xy: np.ndarray,
                               max_back_px: float = 80.0, step_px: float = 0.5):
    p0 = np.asarray(p0_xy, dtype=np.float64)
    d = np.asarray(dir_xy, dtype=np.float64)
    d /= max(np.linalg.norm(d), 1e-12)

    if _inside_mask(mask_fg, p0):
        return p0.copy(), True, 0.0

    n_steps = int(max_back_px / max(step_px, 1e-6))
    for i in range(1, n_steps + 1):
        q = p0 - i * step_px * d
        if _inside_mask(mask_fg, q):
            return q, True, i * step_px
    return p0.copy(), False, None


def ray_last_inside(mask_fg: np.ndarray, p0_xy: np.ndarray, dir_xy: np.ndarray,
                    step_px: float = 0.5, max_len_px: float = 120.0):
    h, w = mask_fg.shape[:2]
    d = np.asarray(dir_xy, dtype=np.float64)
    d /= max(np.linalg.norm(d), 1e-12)

    last_inside = np.asarray(p0_xy, dtype=np.float64).copy()
    n_steps = int(max_len_px / max(step_px, 1e-6))

    for i in range(1, n_steps + 1):
        p = p0_xy + i * step_px * d
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        if not (0 <= x < w and 0 <= y < h):
            break
        if mask_fg[y, x] == 1:
            last_inside = p
        else:
            break
    return last_inside


def _estimate_radius_along_axis(mask_fg: np.ndarray, dist_img: np.ndarray, p_in_xy: np.ndarray,
                                dir_xy: np.ndarray, back_len_px: float = 60.0, step_px: float = 1.0) -> float:
    d = np.asarray(dir_xy, dtype=np.float64)
    d /= max(np.linalg.norm(d), 1e-12)
    vals = []
    n_steps = int(back_len_px / max(step_px, 1e-6))
    for i in range(n_steps + 1):
        p = p_in_xy - i * step_px * d
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        if 0 <= x < mask_fg.shape[1] and 0 <= y < mask_fg.shape[0] and mask_fg[y, x] == 1:
            v = float(dist_img[y, x])
            if np.isfinite(v) and v > 0:
                vals.append(v)
    if not vals:
        x = int(round(float(p_in_xy[0])))
        y = int(round(float(p_in_xy[1])))
        if 0 <= x < dist_img.shape[1] and 0 <= y < dist_img.shape[0]:
            return max(3.0, float(dist_img[y, x]))
        return 6.0
    return max(3.0, float(np.median(vals)))


def _contiguous_runs_from_bool(mask_bool: np.ndarray):
    m = np.asarray(mask_bool, dtype=bool)
    edges = np.diff(np.r_[False, m, False].astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _cross_section_boundaries(mask_fg: np.ndarray, center_xy: np.ndarray, normal_xy: np.ndarray,
                              scan_half_width_px: float = 20.0, step_px: float = 0.5):
    n = np.asarray(normal_xy, dtype=np.float64)
    n /= max(np.linalg.norm(n), 1e-12)
    ts = np.arange(-scan_half_width_px, scan_half_width_px + 0.5 * step_px, step_px, dtype=np.float64)
    inside = np.zeros_like(ts, dtype=bool)

    h, w = mask_fg.shape[:2]
    for i, t in enumerate(ts):
        p = center_xy + t * n
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        inside[i] = (0 <= x < w) and (0 <= y < h) and (mask_fg[y, x] == 1)

    if not np.any(inside):
        return None

    runs = _contiguous_runs_from_bool(inside)
    if not runs:
        return None

    idx0 = int(np.argmin(np.abs(ts)))
    chosen = None
    for s0, s1 in runs:
        if s0 <= idx0 <= s1:
            chosen = (s0, s1)
            break

    if chosen is None:
        centers = [0.5 * (ts[s0] + ts[s1]) for s0, s1 in runs]
        j = int(np.argmin(np.abs(np.asarray(centers))))
        chosen = runs[j]

    s0, s1 = chosen
    t_left = float(ts[s0])
    t_right = float(ts[s1])
    p_left = center_xy + t_left * n
    p_right = center_xy + t_right * n

    return {
        "t_left": t_left,
        "t_right": t_right,
        "p_left_xy": p_left,
        "p_right_xy": p_right,
        "ts": ts,
        "inside": inside,
    }


def refine_tip_edge_distance_transform(
    binary_image: np.ndarray,
    tip_yx: Tuple[int, int],
    tip_angle_deg: float,
    max_step_px: int = 80,
    step_px: float = 1.0,
    exit_mode: str = "first_background",
):
    h, w = binary_image.shape[:2]
    fg = (binary_image == 0).astype(np.uint8)

    d_xy = _tip_angle_to_direction_xy(tip_angle_deg)
    x0, y0 = float(tip_yx[1]), float(tip_yx[0])

    last_inside = np.array([x0, y0], dtype=np.float64)
    exited = False
    n_steps = int(max(1, max_step_px / max(step_px, 1e-6)))

    for i in range(1, n_steps + 1):
        p = np.array([x0, y0], dtype=np.float64) + i * step_px * d_xy
        x = int(round(float(p[0])))
        y = int(round(float(p[1])))
        if not (0 <= x < w and 0 <= y < h):
            exited = True
            break
        if fg[y, x] == 1:
            last_inside = p
            continue
        exited = True
        if exit_mode == "first_background":
            break

    dbg = {
        "mode": "edge_dt",
        "d_xy": d_xy.tolist(),
        "exited": exited,
        "last_inside_xy": last_inside.tolist(),
        "tip_xy": last_inside.tolist(),
        "center_line_xy": [np.array([x0, y0]).tolist(), last_inside.tolist()],
    }
    return float(last_inside[1]), float(last_inside[0]), dbg


def refine_tip_edge_subpixel_gradient(
    grayscale: np.ndarray,
    binary_image: np.ndarray,
    tip_yx: Tuple[int, int],
    tip_angle_deg: float,
    search_len_px: int = 60,
    step_px: float = 0.25,
):
    h, w = grayscale.shape[:2]
    fg = (binary_image == 0).astype(np.uint8)

    d_xy = _tip_angle_to_direction_xy(tip_angle_deg)
    x0, y0 = float(tip_yx[1]), float(tip_yx[0])
    n = int(max(5, search_len_px / max(step_px, 1e-6)))

    samples = []
    coords = []
    inside_mask = []

    for i in range(n + 1):
        p = np.array([x0, y0], dtype=np.float64) + i * step_px * d_xy
        x = int(np.clip(round(float(p[0])), 0, w - 1))
        y = int(np.clip(round(float(p[1])), 0, h - 1))
        samples.append(float(grayscale[y, x]))
        coords.append(p.copy())
        inside_mask.append(int(fg[y, x] == 1))

    s = np.array(samples, dtype=float)
    g = np.diff(s)
    g_abs = np.abs(g)

    valid = [i for i in range(len(g_abs)) if inside_mask[i] == 1]
    if not valid:
        yy, xx, dbg = refine_tip_edge_distance_transform(binary_image, tip_yx, tip_angle_deg)
        dbg["fallback"] = "edge_dt"
        return yy, xx, dbg

    valid = np.array(valid, dtype=int)
    j = int(valid[np.argmax(g_abs[valid])])
    p_edge = 0.5 * (coords[j] + coords[j + 1])

    dbg = {
        "mode": "edge_grad",
        "d_xy": d_xy.tolist(),
        "j": j,
        "g_max": float(g_abs[j]),
        "step_px": float(step_px),
        "tip_xy": p_edge.tolist(),
        "center_line_xy": [np.array([x0, y0]).tolist(), p_edge.tolist()],
    }
    return float(p_edge[1]), float(p_edge[0]), dbg


def skeletonize_mask(mask_fg: np.ndarray) -> np.ndarray:
    try:
        from skimage.morphology import skeletonize
        skel = skeletonize(mask_fg.astype(bool))
        return skel.astype(np.uint8)
    except Exception:
        if hasattr(cv2, "ximgproc") and hasattr(cv2.ximgproc, "thinning"):
            skel = cv2.ximgproc.thinning((mask_fg * 255).astype(np.uint8))
            return (skel > 0).astype(np.uint8)
        raise RuntimeError(
            "Need either scikit-image (preferred) or cv2.ximgproc.thinning for skeletonization."
        )


def skeleton_degree_image(skel: np.ndarray) -> np.ndarray:
    k = np.ones((3, 3), dtype=np.uint8)
    k[1, 1] = 0
    deg = cv2.filter2D(skel.astype(np.uint8), cv2.CV_16U, k)
    return deg


def build_skeleton_graph(skel: np.ndarray):
    pts = np.argwhere(skel > 0)
    pointset = {tuple(p) for p in pts}
    adj = {}
    for y, x in pointset:
        nbrs = []
        for dy, dx, w in _NEI8_W:
            q = (y + dy, x + dx)
            if q in pointset:
                nbrs.append((q, w))
        adj[(y, x)] = nbrs
    return adj


def dijkstra_skeleton(adj, src):
    dist = {src: 0.0}
    prev = {}
    pq = [(0.0, src)]
    visited = set()

    while pq:
        d, u = heappop(pq)
        if u in visited:
            continue
        visited.add(u)

        for v, w in adj[u]:
            nd = d + w
            if nd < dist.get(v, float("inf")):
                dist[v] = nd
                prev[v] = u
                heappush(pq, (nd, v))
    return dist, prev


def reconstruct_path(prev, dst):
    path = [dst]
    cur = dst
    while cur in prev:
        cur = prev[cur]
        path.append(cur)
    path.reverse()
    return path


def path_to_xy(path_yx):
    return np.array([[x, y] for y, x in path_yx], dtype=np.float64)


def cumulative_arclength(xy: np.ndarray) -> np.ndarray:
    if len(xy) == 0:
        return np.zeros((0,), dtype=np.float64)
    if len(xy) == 1:
        return np.array([0.0], dtype=np.float64)
    ds = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(ds)])


def interp_point_on_path(xy: np.ndarray, s: np.ndarray, target_s: float) -> np.ndarray:
    target_s = float(np.clip(target_s, s[0], s[-1]))
    x = np.interp(target_s, s, xy[:, 0])
    y = np.interp(target_s, s, xy[:, 1])
    return np.array([x, y], dtype=np.float64)


def robust_tangent_from_path_window(xy: np.ndarray, s: np.ndarray, s0: float, s1: float) -> np.ndarray:
    mask = (s >= s0) & (s <= s1)
    pts = xy[mask]

    if pts.shape[0] < 5:
        i1 = max(0, len(xy) - 15)
        i2 = max(i1 + 2, len(xy) - 3)
        pts = xy[i1:i2]

    if pts.shape[0] < 2:
        v = xy[-1] - xy[max(0, len(xy) - 2)]
        n = np.linalg.norm(v)
        return v / max(n, 1e-9)

    mu = pts.mean(axis=0)
    uu, ss, vh = np.linalg.svd(pts - mu, full_matrices=False)
    t = vh[0]
    if np.dot(t, xy[-1] - xy[0]) < 0:
        t = -t
    n = np.linalg.norm(t)
    return t / max(n, 1e-9)


def refine_tip_edge_mainray(
    grayscale: np.ndarray,
    binary_image: np.ndarray,
    tip_yx: tuple,
    fit_back_near_r: float = 1.5,
    fit_back_far_r: float = 6.0,
    anchor_back_r: float = 1.0,
    ray_step_px: float = 0.5,
    ray_max_len_r: float = 8.0,
):
    mask_fg = (binary_image == 0).astype(np.uint8)
    if mask_fg.sum() == 0:
        return float(tip_yx[0]), float(tip_yx[1]), {"mode": "mainray", "fallback": "empty_fg"}

    skel = skeletonize_mask(mask_fg)
    deg = skeleton_degree_image(skel)
    endpoints = np.argwhere((skel > 0) & (deg == 1))
    if len(endpoints) < 2:
        return float(tip_yx[0]), float(tip_yx[1]), {"mode": "mainray", "fallback": "not_enough_endpoints"}

    tip_arr = np.asarray(tip_yx, dtype=np.float64)
    d2 = np.sum((endpoints.astype(np.float64) - tip_arr[None, :]) ** 2, axis=1)
    distal_ep = tuple(endpoints[int(np.argmin(d2))])

    adj = build_skeleton_graph(skel)
    dist_map, prev = dijkstra_skeleton(adj, distal_ep)
    endpoint_tuples = [tuple(p) for p in endpoints]
    proximal_ep = max(endpoint_tuples, key=lambda p: dist_map.get(p, -1.0))

    path_yx = reconstruct_path(prev, proximal_ep)
    if len(path_yx) < 5:
        return float(tip_yx[0]), float(tip_yx[1]), {"mode": "mainray", "fallback": "short_path"}

    path_xy = path_to_xy(path_yx)
    s = cumulative_arclength(path_xy)
    s_end = float(s[-1])

    dist_img = cv2.distanceTransform(mask_fg, cv2.DIST_L2, 5)
    tail = path_yx[max(0, len(path_yx) - 20):max(1, len(path_yx) - 3)]
    r_vals = [float(dist_img[y, x]) for (y, x) in tail if 0 <= y < dist_img.shape[0] and 0 <= x < dist_img.shape[1]]
    r_px = float(np.median(r_vals)) if len(r_vals) else float(dist_img[distal_ep[0], distal_ep[1]])
    r_px = max(r_px, 3.0)

    fit_near = fit_back_near_r * r_px
    fit_far = fit_back_far_r * r_px
    anchor_back = anchor_back_r * r_px
    ray_max_len = ray_max_len_r * r_px

    s0 = max(0.0, s_end - fit_far)
    s1 = max(0.0, s_end - fit_near)

    tangent_xy = robust_tangent_from_path_window(path_xy, s, s0, s1)
    anchor_xy = interp_point_on_path(path_xy, s, max(0.0, s_end - anchor_back))
    edge_xy = ray_last_inside(mask_fg, anchor_xy, tangent_xy, step_px=ray_step_px, max_len_px=ray_max_len)

    dbg = {
        "mode": "mainray",
        "distal_endpoint_yx": list(distal_ep),
        "proximal_endpoint_yx": list(proximal_ep),
        "radius_px": r_px,
        "fit_window_px": [fit_near, fit_far],
        "anchor_xy": anchor_xy.tolist(),
        "tangent_xy": tangent_xy.tolist(),
        "tip_xy": edge_xy.tolist(),
        "path_len_px": s_end,
        "center_line_xy": [anchor_xy.tolist(), edge_xy.tolist()],
        "path_window_xy": path_xy[(s >= s0) & (s <= s1)].tolist(),
    }
    return float(edge_xy[1]), float(edge_xy[0]), dbg


def refine_tip_parallel_centerline(
    grayscale: np.ndarray,
    binary_image: np.ndarray,
    tip_yx: Tuple[int, int],
    tip_angle_deg: float,
    section_near_r: float = 1.0,
    section_far_r: float = 6.0,
    scan_half_r: float = 3.0,
    num_sections: int = 9,
    cross_step_px: float = 0.5,
    ray_step_px: float = 0.5,
    ray_max_len_r: float = 8.0,
):
    mask_fg = (binary_image == 0).astype(np.uint8)
    if mask_fg.sum() == 0:
        return float(tip_yx[0]), float(tip_yx[1]), {"mode": "parallel_centerline", "fallback": "empty_fg"}

    d_xy = _tip_angle_to_direction_xy(tip_angle_deg)
    n_xy = np.array([-d_xy[1], d_xy[0]], dtype=np.float64)
    n_xy /= max(np.linalg.norm(n_xy), 1e-12)

    p_guess_xy = np.array([float(tip_yx[1]), float(tip_yx[0])], dtype=np.float64)
    p_in_xy, found_inside, back_dist = _backtrack_point_inside_fg(mask_fg, p_guess_xy, d_xy, max_back_px=120.0, step_px=0.5)
    if not found_inside:
        return float(tip_yx[0]), float(tip_yx[1]), {
            "mode": "parallel_centerline",
            "fallback": "could_not_backtrack_inside",
        }

    dist_img = cv2.distanceTransform(mask_fg, cv2.DIST_L2, 5)
    r_px = _estimate_radius_along_axis(mask_fg, dist_img, p_in_xy, d_xy, back_len_px=80.0, step_px=1.0)

    section_near_px = max(0.5 * r_px, float(section_near_r) * r_px)
    section_far_px = max(section_near_px + 2.0, float(section_far_r) * r_px)
    scan_half_px = max(2.5 * r_px, float(scan_half_r) * r_px)
    ray_max_len_px = max(20.0, float(ray_max_len_r) * r_px)

    s_samples = np.linspace(section_near_px, section_far_px, int(max(3, num_sections)))
    left_offsets = []
    right_offsets = []
    section_centers = []
    left_points = []
    right_points = []

    for s_back in s_samples:
        c_xy = p_in_xy - s_back * d_xy
        res = _cross_section_boundaries(
            mask_fg,
            center_xy=c_xy,
            normal_xy=n_xy,
            scan_half_width_px=scan_half_px,
            step_px=float(cross_step_px),
        )
        if res is None:
            continue
        t_left = float(res["t_left"])
        t_right = float(res["t_right"])
        if t_left > t_right:
            t_left, t_right = t_right, t_left
        left_offsets.append(t_left)
        right_offsets.append(t_right)
        section_centers.append(c_xy.tolist())
        left_points.append(np.asarray(res["p_left_xy"], dtype=np.float64).tolist())
        right_points.append(np.asarray(res["p_right_xy"], dtype=np.float64).tolist())

    if len(left_offsets) < 2 or len(right_offsets) < 2:
        return float(tip_yx[0]), float(tip_yx[1]), {
            "mode": "parallel_centerline",
            "fallback": "insufficient_cross_sections",
            "radius_px": r_px,
        }

    t_left_med = float(np.median(left_offsets))
    t_right_med = float(np.median(right_offsets))
    if t_left_med > t_right_med:
        t_left_med, t_right_med = t_right_med, t_left_med
    t_center = 0.5 * (t_left_med + t_right_med)
    width_px = float(t_right_med - t_left_med)

    line_back_px = section_far_px + 0.75 * r_px
    base_center_xy = p_in_xy - line_back_px * d_xy

    left_line_start = base_center_xy + t_left_med * n_xy
    right_line_start = base_center_xy + t_right_med * n_xy
    center_line_start = base_center_xy + t_center * n_xy

    if not _inside_mask(mask_fg, center_line_start):
        center_line_start, ok_center, _ = _backtrack_point_inside_fg(mask_fg, center_line_start, d_xy, max_back_px=3.0 * r_px, step_px=0.5)
        if not ok_center:
            center_line_start = p_in_xy + t_center * n_xy

    tip_xy = ray_last_inside(mask_fg, center_line_start, d_xy, step_px=float(ray_step_px), max_len_px=ray_max_len_px + line_back_px)

    line_forward_px = float(np.linalg.norm(tip_xy - base_center_xy)) + 0.5 * r_px
    left_line_end = base_center_xy + line_forward_px * d_xy + t_left_med * n_xy
    right_line_end = base_center_xy + line_forward_px * d_xy + t_right_med * n_xy
    center_line_end = base_center_xy + line_forward_px * d_xy + t_center * n_xy

    dbg = {
        "mode": "parallel_centerline",
        "tip_angle_deg": float(tip_angle_deg),
        "d_xy": d_xy.tolist(),
        "n_xy": n_xy.tolist(),
        "tip_guess_xy": p_guess_xy.tolist(),
        "inside_anchor_xy": p_in_xy.tolist(),
        "backtrack_dist_px": None if back_dist is None else float(back_dist),
        "radius_px": float(r_px),
        "width_px": width_px,
        "left_offset_px": float(t_left_med),
        "right_offset_px": float(t_right_med),
        "center_offset_px": float(t_center),
        "section_centers_xy": section_centers,
        "section_left_points_xy": left_points,
        "section_right_points_xy": right_points,
        "parallel_left_line_xy": [left_line_start.tolist(), left_line_end.tolist()],
        "parallel_right_line_xy": [right_line_start.tolist(), right_line_end.tolist()],
        "center_line_xy": [center_line_start.tolist(), tip_xy.tolist()],
        "center_line_full_xy": [center_line_start.tolist(), center_line_end.tolist()],
        "tip_xy": tip_xy.tolist(),
    }
    return float(tip_xy[1]), float(tip_xy[0]), dbg


def annotate_tip_geometry_on_axes(axs, dbg: Dict[str, Any], title_suffix: str = ""):
    if axs is None or not isinstance(dbg, dict):
        return

    try:
        if isinstance(axs, np.ndarray):
            target_axes = [axs.flat[-1]]
            if axs.size >= 3:
                target_axes.append(axs.flat[-2])
        elif isinstance(axs, (list, tuple)):
            target_axes = [axs[-1]]
        else:
            target_axes = [axs]

        used_labels = set()
        mode = str(dbg.get("mode", "tip_refine"))
        tip_xy = np.asarray(dbg.get("tip_xy", []), dtype=float).reshape(-1)

        for ax in target_axes:
            if ax is None:
                continue

            def _plot_line(key, color, label):
                line = dbg.get(key)
                if line is None:
                    return
                arr = np.asarray(line, dtype=float).reshape(-1, 2)
                if arr.shape[0] < 2:
                    return
                line_label = label if label not in used_labels else None
                if line_label is not None:
                    used_labels.add(label)
                ax.plot(arr[:, 0], arr[:, 1], color=color, linewidth=2.0, label=line_label)

            _plot_line("parallel_left_line_xy", "#00ff66", "tube side lines")
            _plot_line("parallel_right_line_xy", "#00ff66", "tube side lines")
            _plot_line("center_line_xy", "#ffd400", "center parallel line")
            _plot_line("center_line_full_xy", "#ffaa00", "center line (full)")

            if "path_window_xy" in dbg:
                pts = np.asarray(dbg["path_window_xy"], dtype=float).reshape(-1, 2)
                if pts.size > 0:
                    lbl = "tangent fit window" if "tangent fit window" not in used_labels else None
                    if lbl is not None:
                        used_labels.add(lbl)
                    ax.plot(pts[:, 0], pts[:, 1], color="#00e5ff", linewidth=2.0, label=lbl)

            if "section_left_points_xy" in dbg and len(dbg["section_left_points_xy"]) > 0:
                pts = np.asarray(dbg["section_left_points_xy"], dtype=float).reshape(-1, 2)
                lbl = "sampled edge points" if "sampled edge points" not in used_labels else None
                if lbl is not None:
                    used_labels.add(lbl)
                ax.scatter(pts[:, 0], pts[:, 1], s=10, c="#44ff44", label=lbl)

            if "section_right_points_xy" in dbg and len(dbg["section_right_points_xy"]) > 0:
                pts = np.asarray(dbg["section_right_points_xy"], dtype=float).reshape(-1, 2)
                ax.scatter(pts[:, 0], pts[:, 1], s=10, c="#44ff44")

            if "inside_anchor_xy" in dbg:
                p = np.asarray(dbg["inside_anchor_xy"], dtype=float).reshape(-1)
                lbl = "inside anchor" if "inside anchor" not in used_labels else None
                if lbl is not None:
                    used_labels.add(lbl)
                ax.scatter([p[0]], [p[1]], s=40, c="#ffffff", edgecolors="#000000", label=lbl, zorder=5)

            if tip_xy.size == 2 and np.all(np.isfinite(tip_xy)):
                lbl = f"refined tip ({mode})" if f"refined tip ({mode})" not in used_labels else None
                if lbl is not None:
                    used_labels.add(lbl)
                ax.scatter([tip_xy[0]], [tip_xy[1]], s=55, c="#ff3b30", edgecolors="#ffffff", label=lbl, zorder=6)
                ax.annotate(
                    f"tip{title_suffix}",
                    (tip_xy[0], tip_xy[1]),
                    xytext=(6, -8),
                    textcoords="offset points",
                    color="white",
                    fontsize=9,
                    bbox=dict(boxstyle="round,pad=0.2", fc="black", ec="none", alpha=0.55),
                )

            try:
                ax.legend(loc="best", fontsize=8)
            except Exception:
                pass
    except Exception as e:
        print(f"[WARN] Failed to annotate analysis axes: {e}")


def _remap_zoom_axes_to_crop_coordinates(axs, zoom_x_min: int, zoom_x_max: int, zoom_y_min: int, zoom_y_max: int):
    if axs is None or not isinstance(axs, np.ndarray) or axs.size < 4:
        return

    zoom_axes = [axs[1, 0], axs[1, 1]]
    x_offset = float(zoom_x_min)
    y_offset = float(zoom_y_min)
    extent = [float(zoom_x_min), float(zoom_x_max + 1), float(zoom_y_max + 1), float(zoom_y_min)]

    for ax in zoom_axes:
        if ax is None:
            continue

        for image in ax.images:
            image.set_extent(extent)

        for coll in ax.collections:
            try:
                offsets = coll.get_offsets()
            except Exception:
                continue
            if offsets is None:
                continue
            offsets = np.asarray(offsets, dtype=float)
            if offsets.size == 0:
                continue
            shifted = offsets.copy()
            shifted[:, 0] += x_offset
            shifted[:, 1] += y_offset
            coll.set_offsets(shifted)

        for line in ax.lines:
            try:
                xdata = np.asarray(line.get_xdata(), dtype=float)
                ydata = np.asarray(line.get_ydata(), dtype=float)
            except Exception:
                continue
            if xdata.size == 0 or ydata.size == 0:
                continue
            line.set_xdata(xdata + x_offset)
            line.set_ydata(ydata + y_offset)

        ax.set_xlim(float(zoom_x_min), float(zoom_x_max + 1))
        ax.set_ylim(float(zoom_y_max + 1), float(zoom_y_min))
        ax.set_aspect("equal")


def patch_analyze_data_for_tip_refinement(
    cal: CTR_Shadow_Calibration,
    refine_mode: str = "none",
    dt_step_px: float = 1.0,
    dt_max_step_px: int = 80,
    grad_step_px: float = 0.25,
    grad_search_len_px: int = 60,
    mainray_fit_back_near_r: float = 1.5,
    mainray_fit_back_far_r: float = 6.0,
    mainray_anchor_back_r: float = 1.0,
    mainray_ray_step_px: float = 0.5,
    mainray_ray_max_len_r: float = 8.0,
    parallel_section_near_r: float = 1.0,
    parallel_section_far_r: float = 6.0,
    parallel_scan_half_r: float = 3.0,
    parallel_num_sections: int = 9,
    parallel_cross_step_px: float = 0.5,
    parallel_ray_step_px: float = 0.5,
    parallel_ray_max_len_r: float = 8.0,
):
    original_analyze_data = cal.analyze_data
    if not hasattr(cal, "tip_refine_debug_records"):
        cal.tip_refine_debug_records = {}

    def analyze_data_patched(
        image_file_name,
        crop_width_min=None,
        crop_width_max=None,
        crop_height_min=None,
        crop_height_max=None,
        threshold=200,
    ):
        fig, axs, coarse_row, fine_row = original_analyze_data(
            image_file_name,
            crop_width_min,
            crop_width_max,
            crop_height_min,
            crop_height_max,
            threshold,
        )

        raw_data_folder = Path(cal.calibration_data_folder) / "raw_image_data_folder"
        img = cv2.imread(str(raw_data_folder / image_file_name), cv2.IMREAD_COLOR)
        if img is None:
            return fig, axs, coarse_row, fine_row

        if crop_width_min is None:
            crop_width_min = cal.analysis_crop["crop_width_min"]
        if crop_width_max is None:
            crop_width_max = cal.analysis_crop["crop_width_max"]
        if crop_height_min is None:
            crop_height_min = cal.analysis_crop["crop_height_min"]
        if crop_height_max is None:
            crop_height_max = cal.analysis_crop["crop_height_max"]

        crop_x_min_img = int(crop_width_min)
        crop_x_max_img = int(crop_width_max)
        crop_y_min_img = int(img.shape[0] - crop_height_max)
        crop_y_max_img = int(img.shape[0] - crop_height_min)

        cropped = img[crop_y_min_img:crop_y_max_img, crop_x_min_img:crop_x_max_img, :]
        if cropped.size == 0:
            return fig, axs, coarse_row, fine_row

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, int(threshold), 255, cv2.THRESH_BINARY)

        tip_y_full = float(coarse_row[0])
        tip_x_full = float(coarse_row[1])
        tip_angle_deg = float(coarse_row[5])

        tip_y = tip_y_full - crop_y_min_img
        tip_x = tip_x_full - crop_x_min_img
        tip_y = float(np.clip(tip_y, 0, binary.shape[0] - 1))
        tip_x = float(np.clip(tip_x, 0, binary.shape[1] - 1))

        zoom_x_min = int(max(int(round(tip_x)) - 75, 0))
        zoom_x_max = int(min(int(round(tip_x)) + 75, binary.shape[1] - 1))
        zoom_y_min = int(max(int(round(tip_y)) - 75, 0))
        zoom_y_max = int(min(int(round(tip_y)) + 75, binary.shape[0] - 1))

        if refine_mode == "edge_dt":
            yy, xx, _dbg = refine_tip_edge_distance_transform(
                binary_image=binary,
                tip_yx=(int(round(tip_y)), int(round(tip_x))),
                tip_angle_deg=tip_angle_deg,
                max_step_px=int(dt_max_step_px),
                step_px=float(dt_step_px),
            )
        elif refine_mode == "edge_grad":
            yy, xx, _dbg = refine_tip_edge_subpixel_gradient(
                grayscale=gray,
                binary_image=binary,
                tip_yx=(int(round(tip_y)), int(round(tip_x))),
                tip_angle_deg=tip_angle_deg,
                search_len_px=int(grad_search_len_px),
                step_px=float(grad_step_px),
            )
        elif refine_mode == "mainray":
            yy, xx, _dbg = refine_tip_edge_mainray(
                grayscale=gray,
                binary_image=binary,
                tip_yx=(int(round(tip_y)), int(round(tip_x))),
                fit_back_near_r=float(mainray_fit_back_near_r),
                fit_back_far_r=float(mainray_fit_back_far_r),
                anchor_back_r=float(mainray_anchor_back_r),
                ray_step_px=float(mainray_ray_step_px),
                ray_max_len_r=float(mainray_ray_max_len_r),
            )
        elif refine_mode == "parallel_centerline":
            yy, xx, _dbg = refine_tip_parallel_centerline(
                grayscale=gray,
                binary_image=binary,
                tip_yx=(int(round(tip_y)), int(round(tip_x))),
                tip_angle_deg=tip_angle_deg,
                section_near_r=float(parallel_section_near_r),
                section_far_r=float(parallel_section_far_r),
                scan_half_r=float(parallel_scan_half_r),
                num_sections=int(parallel_num_sections),
                cross_step_px=float(parallel_cross_step_px),
                ray_step_px=float(parallel_ray_step_px),
                ray_max_len_r=float(parallel_ray_max_len_r),
            )
        else:
            yy, xx, _dbg = tip_y, tip_x, {"mode": "none", "tip_xy": [tip_x, tip_y]}

        coarse_row_refined = np.array(coarse_row, dtype=float).copy()
        coarse_row_refined[0] = yy + crop_y_min_img
        coarse_row_refined[1] = xx + crop_x_min_img

        dbg_local = dict(_dbg) if isinstance(_dbg, dict) else {}
        dbg_local["image_file_name"] = image_file_name
        dbg_local["tip_angle_deg"] = tip_angle_deg
        dbg_local["coarse_tip_before_local_xy"] = [float(tip_x), float(tip_y)]
        dbg_local["coarse_tip_after_local_xy"] = [float(xx), float(yy)]
        dbg_local["crop_origin_xy"] = [int(crop_x_min_img), int(crop_y_min_img)]
        cal.tip_refine_debug_records[image_file_name] = _json_ready(dbg_local)

        _remap_zoom_axes_to_crop_coordinates(axs, zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max)
        annotate_tip_geometry_on_axes(axs, dbg_local, title_suffix=f" ({refine_mode})")
        return fig, axs, coarse_row_refined, fine_row

    cal.analyze_data = analyze_data_patched
    return cal


# -----------------------------------------
# Visualization helpers
# -----------------------------------------
def draw_checkerboard_analysis_overlay(
    cal: CTR_Shadow_Calibration,
    output_path: Path,
    tracked_rows: np.ndarray,
    image_files: List[Path],
    board_debug_image: np.ndarray,
):
    if board_debug_image is None:
        raise ValueError("board_debug_image is required.")
    if tracked_rows is None or tracked_rows.size == 0:
        raise ValueError("tracked_rows is empty; run analyze_data_batch first.")
    if getattr(cal, "board_pose", None) is None or getattr(cal, "true_vertical_img_unit", None) is None:
        raise RuntimeError("Board reference is unavailable.")

    valid_mask = np.all(np.isfinite(tracked_rows[:, :2]), axis=1)
    if tracked_rows.shape[1] > 3:
        valid_mask &= np.isfinite(tracked_rows[:, 3])
    valid_rows = tracked_rows[valid_mask]
    if valid_rows.size == 0:
        raise RuntimeError("No valid tracked points were found for annotation.")

    pts_xy = np.column_stack([valid_rows[:, 1], valid_rows[:, 0]]).astype(np.float64)
    if bool(cal.board_pose.get("use_undistort", False)):
        pts_xy_draw = _undistort_points_to_image(pts_xy, cal.camera_K, cal.camera_dist)
    else:
        pts_xy_draw = pts_xy.copy()

    overlay = board_debug_image.copy()
    h, w = overlay.shape[:2]
    origin = np.asarray(cal.board_pose["origin_px"], dtype=np.float64).reshape(2)
    v_hat = np.asarray(cal.true_vertical_img_unit, dtype=np.float64).reshape(2)
    v_hat /= max(np.linalg.norm(v_hat), 1e-12)
    u_hat = np.array([v_hat[1], -v_hat[0]], dtype=np.float64)

    arrow_len = int(max(60, min(h, w) * 0.14))
    origin_i = tuple(np.round(origin).astype(int))
    horiz_end = tuple(np.round(origin + u_hat * arrow_len).astype(int))
    vert_end = tuple(np.round(origin + v_hat * arrow_len).astype(int))
    cv2.arrowedLine(overlay, origin_i, horiz_end, (40, 80, 255), 3, tipLength=0.14)
    cv2.arrowedLine(overlay, origin_i, vert_end, (0, 220, 0), 3, tipLength=0.14)
    cv2.putText(overlay, "camera horizontal", (horiz_end[0] + 8, horiz_end[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (40, 80, 255), 2)
    cv2.putText(overlay, "camera vertical", (vert_end[0] + 8, vert_end[1] - 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 0), 2)

    for pt in pts_xy_draw:
        px = int(round(pt[0]))
        py = int(round(pt[1]))
        if 0 <= px < w and 0 <= py < h:
            cv2.circle(overlay, (px, py), 4, (0, 255, 255), -1)
            cv2.circle(overlay, (px, py), 7, (0, 0, 0), 1)

    if pts_xy_draw.shape[0] >= 2:
        pts_poly = np.round(pts_xy_draw).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(overlay, [pts_poly], False, (255, 200, 0), 1, lineType=cv2.LINE_AA)

    summary_lines = [
        f"tracked points: {len(valid_rows)}",
        f"local scale: {float(cal.board_px_per_mm_local):.4f} px/mm",
    ]
    if getattr(cal, "board_reference_correction_meta", None):
        cmeta = cal.board_reference_correction_meta
        summary_lines.append(f"checkerboard mm scale correction: {float(cmeta['checkerboard_mm_scale_correction']):.3f}")
        summary_lines.append(f"checkerboard x sign flipped: {bool(cmeta['checkerboard_planar_x_flipped'])}")

    y_text = 32
    for line in summary_lines:
        cv2.putText(overlay, line, (24, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 3)
        cv2.putText(overlay, line, (24, y_text), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (20, 20, 20), 1)
        y_text += 30

    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(output_path), overlay):
        raise RuntimeError(f"Failed to write annotated checkerboard image: {output_path}")
    return output_path


# -----------------------------------------
# Error analysis in checkerboard mm frame
# -----------------------------------------
def _extract_b_value_from_row(row: np.ndarray) -> float:
    if row is None or len(row) < 4:
        return float("nan")
    try:
        return float(row[3])
    except Exception:
        return float("nan")


def compute_tracked_tip_positions_mm(
    cal: CTR_Shadow_Calibration,
    tracked_rows: np.ndarray,
    image_files: List[Path],
    desired_tip_lookup: Optional[Dict[int, Dict[str, float]]] = None,
) -> Dict[str, Any]:
    """
    Convert tracked tip pixel coordinates to checkerboard-referenced mm.
    """
    if getattr(cal, "board_pose", None) is None:
        raise RuntimeError("Checkerboard board_pose is not available.")
    if tracked_rows is None or tracked_rows.size == 0:
        raise RuntimeError("tracked_rows is empty.")

    origin = np.asarray(cal.board_pose["origin_px"], dtype=np.float64).reshape(2)

    records = []
    mm_points = []
    valid_indices = []
    matched_measured_mm_points = []
    desired_mm_points = []
    matched_valid_indices = []

    for i, row in enumerate(np.asarray(tracked_rows, dtype=float)):
        file_name = image_files[i].name if i < len(image_files) else f"sample_{i:04d}"
        tracked_sample_idx, tracked_phase = parse_tracked_sample_metadata(file_name)
        csv_sample_index = int(tracked_sample_idx) if tracked_sample_idx is not None else int(i)
        if row.size < 2 or not np.all(np.isfinite(row[:2])):
            records.append({
                "sample_index": csv_sample_index,
                "image_name": file_name,
                "tip_y_px": None,
                "tip_x_px": None,
                "u_mm": None,
                "z_mm": None,
                "b_pull_mm": None,
                "phase": tracked_phase,
                "valid": False,
            })
            continue

        y_px = float(row[0])
        x_px = float(row[1])

        try:
            u_mm, z_mm = cal.pixel_point_to_calibrated_axes(
                x_px=x_px,
                y_px=y_px,
                origin_px=origin,
            )
            u_mm = float(u_mm)
            z_mm = float(z_mm)
            valid = np.isfinite(u_mm) and np.isfinite(z_mm)
        except Exception:
            u_mm, z_mm = float("nan"), float("nan")
            valid = False

        rec = {
            "sample_index": csv_sample_index,
            "image_name": file_name,
            "tip_y_px": y_px,
            "tip_x_px": x_px,
            "u_mm": u_mm if valid else None,
            "z_mm": z_mm if valid else None,
            "b_pull_mm": _extract_b_value_from_row(row),
            "phase": tracked_phase,
            "valid": bool(valid),
        }
        named_values = extract_named_values_from_filename(file_name)
        rec["stage_x_mm"] = float(named_values["X"]) if "X" in named_values else None
        rec["stage_y_mm"] = float(named_values["Y"]) if "Y" in named_values else None
        rec["stage_z_mm"] = float(named_values["Z"]) if "Z" in named_values else None
        desired_tip = desired_tip_lookup.get(tracked_sample_idx) if (desired_tip_lookup and tracked_sample_idx is not None) else None
        rec["desired_x_mm"] = float(desired_tip["desired_tip_x_mm"]) if desired_tip is not None else rec["stage_x_mm"]
        rec["desired_y_mm"] = rec["stage_y_mm"]
        rec["desired_z_mm"] = float(desired_tip["desired_tip_z_mm"]) if desired_tip is not None else rec["stage_z_mm"]
        records.append(rec)

        if valid:
            mm_points.append([u_mm, z_mm])
            valid_indices.append(i)
            if rec["desired_x_mm"] is not None and rec["desired_z_mm"] is not None:
                matched_measured_mm_points.append([u_mm, z_mm])
                desired_mm_points.append([float(rec["desired_x_mm"]), float(rec["desired_z_mm"])])
                matched_valid_indices.append(i)

    mm_points = np.asarray(mm_points, dtype=float) if len(mm_points) else np.empty((0, 2), dtype=float)
    matched_measured_mm_points = (
        np.asarray(matched_measured_mm_points, dtype=float)
        if len(matched_measured_mm_points) else np.empty((0, 2), dtype=float)
    )
    desired_mm_points = (
        np.asarray(desired_mm_points, dtype=float) if len(desired_mm_points) else np.empty((0, 2), dtype=float)
    )

    return {
        "records": records,
        "valid_indices": valid_indices,
        "mm_points": mm_points,
        "matched_measured_mm_points": matched_measured_mm_points,
        "desired_mm_points": desired_mm_points,
        "matched_valid_indices": matched_valid_indices,
    }


def _closest_points_on_polyline(points_xy: np.ndarray, polyline_xy: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    points_xy = np.asarray(points_xy, dtype=float).reshape(-1, 2)
    polyline_xy = np.asarray(polyline_xy, dtype=float).reshape(-1, 2)
    if points_xy.shape[0] == 0:
        return np.empty((0, 2), dtype=float), np.empty((0,), dtype=int)
    if polyline_xy.shape[0] == 0:
        raise RuntimeError("Desired curve is empty.")
    if polyline_xy.shape[0] == 1:
        nearest = np.repeat(polyline_xy, points_xy.shape[0], axis=0)
        return nearest, np.zeros((points_xy.shape[0],), dtype=int)

    seg_starts = polyline_xy[:-1]
    seg_ends = polyline_xy[1:]
    seg_vecs = seg_ends - seg_starts
    seg_len_sq = np.sum(seg_vecs * seg_vecs, axis=1)

    nearest_points = np.empty_like(points_xy)
    nearest_seg_indices = np.zeros((points_xy.shape[0],), dtype=int)

    for i, p in enumerate(points_xy):
        best_dist_sq = float("inf")
        best_proj = None
        best_seg_idx = 0
        for seg_idx, (a, v, vv) in enumerate(zip(seg_starts, seg_vecs, seg_len_sq)):
            if vv <= 1e-12:
                proj = a
            else:
                t = float(np.dot(p - a, v) / vv)
                t = min(1.0, max(0.0, t))
                proj = a + t * v
            dist_sq = float(np.sum((p - proj) ** 2))
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_proj = proj
                best_seg_idx = seg_idx
        nearest_points[i] = best_proj
        nearest_seg_indices[i] = best_seg_idx

    return nearest_points, nearest_seg_indices


def compute_error_metrics(measured_mm_points: np.ndarray, desired_mm_points: np.ndarray) -> Dict[str, Any]:
    """
    Align measured points to desired points by matching their mean centers.
    Error for each sample = Euclidean distance to the closest point on the desired curve.
    RMSE = sqrt(mean(error^2))
    """
    measured_mm_points = np.asarray(measured_mm_points, dtype=float).reshape(-1, 2)
    desired_mm_points = np.asarray(desired_mm_points, dtype=float).reshape(-1, 2)
    if measured_mm_points.shape[0] == 0:
        raise RuntimeError("No valid measured mm points available for error analysis.")
    if desired_mm_points.shape[0] == 0:
        raise RuntimeError("No desired mm points available for aligned error analysis.")

    measured_mean = np.mean(measured_mm_points, axis=0)
    desired_mean = np.mean(desired_mm_points, axis=0)
    alignment_offset = desired_mean - measured_mean
    aligned_measured_points = measured_mm_points + alignment_offset[None, :]
    closest_desired_points, closest_segment_indices = _closest_points_on_polyline(
        aligned_measured_points,
        desired_mm_points,
    )
    deltas = aligned_measured_points - closest_desired_points
    dist_mm = np.linalg.norm(deltas, axis=1)

    rmse_mm = float(np.sqrt(np.mean(dist_mm ** 2)))
    mean_err_mm = float(np.mean(dist_mm))
    std_err_mm = float(np.std(dist_mm))
    max_err_mm = float(np.max(dist_mm))
    min_err_mm = float(np.min(dist_mm))
    median_err_mm = float(np.median(dist_mm))

    return {
        "reference_point_mm": {
            "u_mean_mm": float(desired_mean[0]),
            "z_mean_mm": float(desired_mean[1]),
        },
        "measured_mean_mm": {
            "u_mean_mm": float(measured_mean[0]),
            "z_mean_mm": float(measured_mean[1]),
        },
        "desired_mean_mm": {
            "u_mean_mm": float(desired_mean[0]),
            "z_mean_mm": float(desired_mean[1]),
        },
        "alignment_offset_mm": {
            "du_mm": float(alignment_offset[0]),
            "dz_mm": float(alignment_offset[1]),
        },
        "errors_mm": dist_mm,
        "deltas_mm": deltas,
        "aligned_measured_mm_points": aligned_measured_points,
        "desired_mm_points": desired_mm_points,
        "closest_desired_mm_points": closest_desired_points,
        "closest_desired_segment_indices": closest_segment_indices,
        "rmse_mm": rmse_mm,
        "mean_error_mm": mean_err_mm,
        "std_error_mm": std_err_mm,
        "median_error_mm": median_err_mm,
        "min_error_mm": min_err_mm,
        "max_error_mm": max_err_mm,
        "num_samples": int(measured_mm_points.shape[0]),
    }


def attach_errors_to_records(
    records: List[Dict[str, Any]],
    valid_indices: List[int],
    deltas_mm: np.ndarray,
    errors_mm: np.ndarray,
    aligned_measured_mm_points: np.ndarray,
    closest_desired_mm_points: np.ndarray,
):
    idx_to_local = {global_idx: k for k, global_idx in enumerate(valid_indices)}

    for i, rec in enumerate(records):
        if i not in idx_to_local:
            rec["aligned_u_mm"] = None
            rec["aligned_z_mm"] = None
            rec["closest_desired_u_mm"] = None
            rec["closest_desired_z_mm"] = None
            rec["du_aligned_to_desired_mm"] = None
            rec["dz_aligned_to_desired_mm"] = None
            rec["error_distance_mm"] = None
            continue

        k = idx_to_local[i]
        rec["aligned_u_mm"] = float(aligned_measured_mm_points[k, 0])
        rec["aligned_z_mm"] = float(aligned_measured_mm_points[k, 1])
        rec["closest_desired_u_mm"] = float(closest_desired_mm_points[k, 0])
        rec["closest_desired_z_mm"] = float(closest_desired_mm_points[k, 1])
        rec["du_aligned_to_desired_mm"] = float(deltas_mm[k, 0])
        rec["dz_aligned_to_desired_mm"] = float(deltas_mm[k, 1])
        rec["error_distance_mm"] = float(errors_mm[k])

    return records


def _phase_to_quarter_label(phase: Optional[str]) -> Optional[str]:
    phase_str = str(phase).strip() if phase is not None else ""
    if phase_str.startswith("q1_pull"):
        return "q1_pull"
    if phase_str.startswith("q2_release"):
        return "q2_release"
    if phase_str.startswith("q3_pull"):
        return "q3_pull"
    if phase_str.startswith("q4_release"):
        return "q4_release"
    return None


def compute_quarter_error_stats(records: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    quarter_errors: Dict[str, List[float]] = {
        "q1_pull": [],
        "q2_release": [],
        "q3_pull": [],
        "q4_release": [],
    }

    for rec in records:
        err = rec.get("error_distance_mm")
        if err is None:
            continue
        quarter = _phase_to_quarter_label(rec.get("phase"))
        if quarter is None:
            continue
        quarter_errors[quarter].append(float(err))

    stats: Dict[str, Dict[str, Any]] = {}
    for quarter, values in quarter_errors.items():
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            stats[quarter] = {
                "num_samples": 0,
                "rmse_mm": None,
                "mean_error_mm": None,
                "std_error_mm": None,
                "median_error_mm": None,
                "min_error_mm": None,
                "max_error_mm": None,
            }
            continue
        stats[quarter] = {
            "num_samples": int(arr.size),
            "rmse_mm": float(np.sqrt(np.mean(arr ** 2))),
            "mean_error_mm": float(np.mean(arr)),
            "std_error_mm": float(np.std(arr)),
            "median_error_mm": float(np.median(arr)),
            "min_error_mm": float(np.min(arr)),
            "max_error_mm": float(np.max(arr)),
        }
    return stats


def save_tracked_tip_csv(csv_path: Path, records: List[Dict[str, Any]]):
    fieldnames = [
        "sample_index",
        "image_name",
        "phase",
        "tip_y_px",
        "tip_x_px",
        "u_mm",
        "z_mm",
        "b_pull_mm",
        "stage_x_mm",
        "stage_y_mm",
        "stage_z_mm",
        "desired_x_mm",
        "desired_y_mm",
        "desired_z_mm",
        "aligned_u_mm",
        "aligned_z_mm",
        "closest_desired_u_mm",
        "closest_desired_z_mm",
        "du_aligned_to_desired_mm",
        "dz_aligned_to_desired_mm",
        "error_distance_mm",
        "valid",
    ]
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in records:
            writer.writerow(r)


def save_error_plot(error_plot_path: Path, records: List[Dict[str, Any]], title_prefix: str = ""):
    valid_recs = [r for r in records if r.get("valid", False) and r.get("error_distance_mm") is not None]
    if not valid_recs:
        raise RuntimeError("No valid records available for error plot.")
    valid_recs = sorted(valid_recs, key=lambda r: int(r["sample_index"]))

    sample_idx = [int(r["sample_index"]) for r in valid_recs]
    errors_mm = [float(r["error_distance_mm"]) for r in valid_recs]

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="none")
    ax.set_facecolor("none")
    ax.plot(
        sample_idx,
        errors_mm,
        marker="o",
        linestyle="-",
        color="white",
        markerfacecolor="white",
        markeredgecolor="white",
        linewidth=1.8,
        markersize=4.5,
    )
    ax.set_xlabel("Sample index", color="white")
    ax.set_ylabel("Aligned distance to closest desired-curve point (mm)", color="white")
    ax.set_title(f"{title_prefix}Aligned tracked tip error vs sample".strip(), color="white")
    ax.grid(True, alpha=0.22, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    plt.tight_layout()
    plt.savefig(error_plot_path, dpi=200, transparent=True)
    plt.close()


def save_error_histogram(
    hist_path: Path,
    errors_mm: np.ndarray,
    rmse_mm: float,
    title_prefix: str = "",
    bins: int = 20,
):
    errors_mm = np.asarray(errors_mm, dtype=float).reshape(-1)
    if errors_mm.size == 0:
        raise RuntimeError("No error values available for histogram.")

    fig, ax = plt.subplots(figsize=(8, 5), facecolor="none")
    ax.set_facecolor("none")
    ax.hist(
        errors_mm,
        bins=int(max(5, bins)),
        color="white",
        edgecolor="white",
        linewidth=1.0,
        alpha=0.9,
    )
    ax.set_xlabel("Error distance (mm)", color="white")
    ax.set_ylabel("Number of samples", color="white")
    ax.set_title(
        f"{title_prefix}Histogram of tracked tip error | RMSE = {float(rmse_mm):.4f} mm".strip(),
        color="white",
    )
    ax.grid(True, alpha=0.18, color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("white")
    plt.tight_layout()
    plt.savefig(hist_path, dpi=200, transparent=True)
    plt.close()


def save_desired_vs_actual_plot(
    plot_path: Path,
    records: List[Dict[str, Any]],
    reference_point_mm: Dict[str, float],
    alignment_offset_mm: Dict[str, float],
    title_prefix: str = "",
):
    valid_recs = [
        r for r in records
        if r.get("valid", False)
        and r.get("aligned_u_mm") is not None
        and r.get("aligned_z_mm") is not None
    ]
    desired_recs = [r for r in valid_recs if r.get("desired_x_mm") is not None and r.get("desired_z_mm") is not None]
    if not valid_recs:
        raise RuntimeError("No valid tracked records available for desired-vs-actual plot.")
    if not desired_recs:
        raise RuntimeError("Desired tip X/Z values could not be parsed from filenames.")

    valid_recs = sorted(valid_recs, key=lambda r: int(r["sample_index"]))
    desired_recs = sorted(desired_recs, key=lambda r: int(r["sample_index"]))

    actual_u = np.asarray([float(r["aligned_u_mm"]) for r in valid_recs], dtype=float)
    actual_z = np.asarray([float(r["aligned_z_mm"]) for r in valid_recs], dtype=float)
    desired_tip_x = np.asarray([float(r["desired_x_mm"]) for r in desired_recs], dtype=float)
    desired_tip_z = np.asarray([float(r["desired_z_mm"]) for r in desired_recs], dtype=float)

    ref_u = float(reference_point_mm["u_mean_mm"])
    ref_z = float(reference_point_mm["z_mean_mm"])

    desired_u = desired_tip_x
    desired_z = desired_tip_z

    fig, ax = plt.subplots(figsize=(8.5, 7.0), facecolor="none")
    ax.set_facecolor("none")

    ax.plot(
        desired_u,
        desired_z,
        color="#8cf7ff",
        linewidth=2.2,
        alpha=0.95,
        label="Desired circle",
        zorder=3,
    )
    ax.scatter(
        desired_u,
        desired_z,
        s=20,
        color="#f8fafc",
        edgecolors="#8cf7ff",
        linewidths=0.6,
        alpha=0.95,
        label="Desired sampled points",
        zorder=4,
    )
    ax.scatter(
        actual_u,
        actual_z,
        s=26,
        color="#ff8fab",
        edgecolors="#fff1f5",
        linewidths=0.5,
        alpha=0.9,
        label="Measured tracked points (mean-aligned)",
        zorder=5,
    )
    ax.scatter(
        desired_u[:1],
        desired_z[:1],
        s=64,
        color="#f4d35e",
        edgecolors="none",
        label="Trajectory start",
        zorder=6,
    )

    all_u = np.concatenate([actual_u, desired_u, np.asarray([ref_u])])
    all_z = np.concatenate([actual_z, desired_z, np.asarray([ref_z])])
    center_u = 0.5 * float(np.min(all_u) + np.max(all_u))
    center_z = 0.5 * float(np.min(all_z) + np.max(all_z))
    span_u = max(float(np.max(np.abs(all_u - center_u))), 1.0)
    span_z = max(float(np.max(np.abs(all_z - center_z))), 1.0)
    span = 1.08 * max(span_u, span_z)

    ax.set_xlim(center_u - span, center_u + span)
    ax.set_ylim(center_z - span, center_z + span)
    ax.set_aspect("equal", adjustable="box")

    ax.set_xlabel("u / desired X (mm)", color="#e8f3ff")
    ax.set_ylabel("z / desired tip Z (mm)", color="#e8f3ff")
    ax.set_title(
        f"{title_prefix}Desired circle trajectory vs mean-aligned tracked points".strip(),
        color="#f8fbff",
    )
    ax.grid(True, color="#8eb8d8", alpha=0.12, linewidth=0.8)
    ax.tick_params(colors="#dceaf7")
    for spine in ax.spines.values():
        spine.set_color("#6b92b3")

    legend = ax.legend(frameon=True, facecolor="#102131", edgecolor="#6b92b3")
    for txt in legend.get_texts():
        txt.set_color("#f8fbff")

    ax.text(
        0.02,
        0.98,
        (
            "Applied mean alignment "
            f"(du={float(alignment_offset_mm['du_mm']):.3f} mm, "
            f"dz={float(alignment_offset_mm['dz_mm']):.3f} mm)"
        ),
        transform=ax.transAxes,
        va="top",
        ha="left",
        color="#f8fbff",
        fontsize=9.5,
        bbox={"facecolor": "#102131", "edgecolor": "#6b92b3", "alpha": 0.75, "pad": 6},
    )

    plt.tight_layout()
    plt.savefig(plot_path, dpi=220, transparent=True)
    plt.close()


def save_metrics_json(
    json_path: Path,
    metrics: Dict[str, Any],
    cal: CTR_Shadow_Calibration,
    args,
):
    payload = {
        "reference_point_mm": metrics["reference_point_mm"],
        "measured_mean_mm": metrics["measured_mean_mm"],
        "desired_mean_mm": metrics["desired_mean_mm"],
        "alignment_offset_mm": metrics["alignment_offset_mm"],
        "quarter_error_stats": metrics.get("quarter_error_stats", {}),
        "rmse_mm": metrics["rmse_mm"],
        "mean_error_mm": metrics["mean_error_mm"],
        "std_error_mm": metrics["std_error_mm"],
        "median_error_mm": metrics["median_error_mm"],
        "min_error_mm": metrics["min_error_mm"],
        "max_error_mm": metrics["max_error_mm"],
        "num_samples": metrics["num_samples"],
        "analysis_crop": getattr(cal, "analysis_crop", None),
        "board_reference": collect_board_reference_info(cal),
        "settings": {
            "threshold": int(args.threshold),
            "tracked_tip_source": str(getattr(args, "tracked_tip_source", "auto")),
            "tip_refiner_model": None if getattr(args, "tip_refiner_model", None) is None else str(Path(args.tip_refiner_model).expanduser().resolve()),
            "tip_refiner_anchor": getattr(args, "tip_refiner_anchor", None),
            "tip_refiner_compare_only": bool(getattr(args, "tip_refiner_compare_only", False)),
            "tip_refine_mode": str(args.tip_refine_mode),
            "tip_refine_dt_step_px": float(args.tip_refine_dt_step_px),
            "tip_refine_max_step_px": int(args.tip_refine_max_step_px),
            "tip_refine_grad_step_px": float(args.tip_refine_grad_step_px),
            "tip_refine_grad_search_len_px": int(args.tip_refine_grad_search_len_px),
            "tip_refine_mainray_fit_back_near_r": float(args.tip_refine_mainray_fit_back_near_r),
            "tip_refine_mainray_fit_back_far_r": float(args.tip_refine_mainray_fit_back_far_r),
            "tip_refine_mainray_anchor_back_r": float(args.tip_refine_mainray_anchor_back_r),
            "tip_refine_mainray_ray_step_px": float(args.tip_refine_mainray_ray_step_px),
            "tip_refine_mainray_ray_max_len_r": float(args.tip_refine_mainray_ray_max_len_r),
            "tip_refine_parallel_section_near_r": float(args.tip_refine_parallel_section_near_r),
            "tip_refine_parallel_section_far_r": float(args.tip_refine_parallel_section_far_r),
            "tip_refine_parallel_scan_half_r": float(args.tip_refine_parallel_scan_half_r),
            "tip_refine_parallel_num_sections": int(args.tip_refine_parallel_num_sections),
            "tip_refine_parallel_cross_step_px": float(args.tip_refine_parallel_cross_step_px),
            "tip_refine_parallel_ray_step_px": float(args.tip_refine_parallel_ray_step_px),
            "tip_refine_parallel_ray_max_len_r": float(args.tip_refine_parallel_ray_max_len_r),
            "circle_center_x_mm": float(args.circle_center_x_mm),
            "circle_center_z_mm": float(args.circle_center_z_mm),
            "circle_radius_mm": float(args.circle_radius_mm),
            "circle_samples_per_quarter": int(args.circle_samples_per_quarter),
            "planned_command_csv": (None if args.planned_command_csv is None else str(args.planned_command_csv)),
            "capture_at_start": bool(args.capture_at_start),
            "capture_every_n_circle_moves": int(args.capture_every_n_circle_moves),
            "hist_bins": int(args.hist_bins),
        },
    }
    with open(json_path, "w") as f:
        json.dump(_json_ready(payload), f, indent=2)


# -----------------------------
# Main
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project_dir", type=str, default=None,
                    help="Existing project folder containing raw_image_data_folder/")
    ap.add_argument("--raw_dir", type=str, default=None,
                    help="raw_image_data_folder or any folder of images (will be wrapped into a project)")
    ap.add_argument("--threshold", type=int, default=200)
    ap.add_argument("--save_plots", action="store_true")

    ap.add_argument("--camera_calibration_file", type=str, required=True,
                    help="Path to camera calibration .npz for checkerboard-reference analysis.")
    ap.add_argument("--checkerboard_reference_image", type=str, required=True,
                    help="Path to checkerboard reference image.")
    ap.add_argument("--checkerboard_inner_corners", type=_parse_inner_corners_arg, default=None,
                    help="Checkerboard inner-corner grid as 'Nx,Ny' or 'NxXNy'. Defaults to metadata in the calibration file.")
    ap.add_argument("--checkerboard_square_size_mm", type=float, default=None,
                    help="Checkerboard square size in mm. Defaults to metadata in the calibration file.")
    ap.add_argument("--checkerboard_no_undistort", action="store_true",
                    help="Disable undistortion before checkerboard pose estimation.")

    ap.add_argument("--checkerboard_mm_scale_correction", type=float, default=0.5,
                    help="Multiply checkerboard-derived mm values by this factor. Default 0.5 fixes observed 2x overscale.")
    ap.add_argument("--checkerboard_no_flip_planar_x", action="store_true",
                    help="Disable checkerboard planar-x sign flip.")

    ap.add_argument("--link_mode", type=str, default="symlink", choices=["symlink", "copy"])
    ap.add_argument("--save_analysis_config", action="store_true")

    ap.add_argument("--tip_refine_mode", type=str, default="none",
                    choices=["none", "edge_dt", "edge_grad", "mainray", "parallel_centerline"],
                    help="Refine tip position using the same distal tip analysis modes as offline_run_calibration.py.")
    ap.add_argument("--tip_refine_dt_step_px", type=float, default=1.0)
    ap.add_argument("--tip_refine_max_step_px", type=int, default=80)
    ap.add_argument("--tip_refine_grad_step_px", type=float, default=0.25)
    ap.add_argument("--tip_refine_grad_search_len_px", type=int, default=60)
    ap.add_argument("--tip_refine_mainray_fit_back_near_r", type=float, default=1.5)
    ap.add_argument("--tip_refine_mainray_fit_back_far_r", type=float, default=6.0)
    ap.add_argument("--tip_refine_mainray_anchor_back_r", type=float, default=1.0)
    ap.add_argument("--tip_refine_mainray_ray_step_px", type=float, default=0.5)
    ap.add_argument("--tip_refine_mainray_ray_max_len_r", type=float, default=8.0)
    ap.add_argument("--tip_refine_parallel_section_near_r", type=float, default=1.0)
    ap.add_argument("--tip_refine_parallel_section_far_r", type=float, default=6.0)
    ap.add_argument("--tip_refine_parallel_scan_half_r", type=float, default=3.0)
    ap.add_argument("--tip_refine_parallel_num_sections", type=int, default=9)
    ap.add_argument("--tip_refine_parallel_cross_step_px", type=float, default=0.5)
    ap.add_argument("--tip_refine_parallel_ray_step_px", type=float, default=0.5)
    ap.add_argument("--tip_refine_parallel_ray_max_len_r", type=float, default=8.0)
    ap.add_argument("--tip_refiner_model", type=str, default=None,
                    help="Path to cnn/train_tip_refiner.py best_tip_refiner.pt for CNN tip detection.")
    ap.add_argument("--tip_refiner_anchor", type=str, default=None, choices=["coarse", "selected", "refined"],
                    help="Patch anchor for CNN inference. Defaults to the model checkpoint anchor.")
    ap.add_argument("--tip_refiner_compare_only", action="store_true",
                    help="Save CNN tips but keep non-CNN tips as the default tracked source.")
    ap.add_argument("--tracked_tip_source", type=str, default="auto", choices=["auto", "coarse", "selected", "cnn"],
                    help="Which tip rows to convert to mm. auto uses selected/CNN when --tip_refiner_model is active, otherwise coarse.")

    ap.add_argument("--circle_center_x_mm", type=float, default=CIRCLE_CENTER_X_DEFAULT)
    ap.add_argument("--circle_center_z_mm", type=float, default=CIRCLE_CENTER_Z_DEFAULT)
    ap.add_argument("--circle_radius_mm", type=float, default=CIRCLE_RADIUS_DEFAULT)
    ap.add_argument("--circle_samples_per_quarter", type=int, default=CIRCLE_SAMPLES_PER_QUARTER_DEFAULT,
                    help="Fallback ideal-circle sampling used only when planned_command_sequence.csv is unavailable.")
    ap.add_argument("--planned_command_csv", type=str, default=None,
                    help="Optional path to planned_command_sequence.csv from the circle acquisition run. Defaults to <project_dir>/planned_command_sequence.csv when present.")
    ap.add_argument("--capture_at_start", action="store_true",
                    help="Match acquisition runs that also captured the initial q1_pull_start point.")
    ap.add_argument("--capture_every_n_circle_moves", type=int, default=1,
                    help="Match acquisition subsampling: one saved image every N recorded circle moves.")

    ap.add_argument("--hist_bins", type=int, default=20,
                    help="Number of histogram bins.")

    args = ap.parse_args()

    if args.project_dir is None and args.raw_dir is None:
        raise SystemExit("Provide --project_dir or --raw_dir")

    project_dir = Path(args.project_dir).expanduser().resolve() if args.project_dir else None
    raw_dir = Path(args.raw_dir).expanduser().resolve() if args.raw_dir else None

    if project_dir and (project_dir / "raw_image_data_folder").is_dir():
        raw_folder = project_dir / "raw_image_data_folder"
    elif raw_dir:
        if raw_dir.name == "raw_image_data_folder":
            raw_folder = raw_dir
            if project_dir is None:
                project_dir = raw_dir.parent
            elif project_dir != raw_dir.parent:
                raw_folder = ensure_project_from_raw(raw_dir, project_dir, link_mode=args.link_mode)
        else:
            if project_dir is None:
                project_dir = raw_dir.parent / (raw_dir.name + "_offline_project")
            raw_folder = ensure_project_from_raw(raw_dir, project_dir, link_mode=args.link_mode)
    else:
        raise SystemExit("Could not resolve folders. Check paths.")

    project_dir = project_dir.resolve()
    raw_folder = raw_folder.resolve()

    renamed_files = normalize_raw_filenames_for_shadow_parser(raw_folder)
    if renamed_files:
        print(f"[INFO] Normalized {len(renamed_files)} raw image filenames for shadow_calibration.")
        for old_name, new_name in renamed_files[:10]:
            print(f"  - {old_name} -> {new_name}")
        if len(renamed_files) > 10:
            print(f"  ... and {len(renamed_files) - 10} more")

    cal = CTR_Shadow_Calibration(
        parent_directory=str(project_dir.parent),
        project_name=project_dir.name,
        allow_existing=True,
        add_date=False,
    )
    cal.calibration_data_folder = str(project_dir)
    load_optional_tip_refiner(cal, args)

    patch_filename_parser()

    desired_tip_lookup = resolve_desired_tip_lookup(project_dir, args)

    if args.tip_refine_mode != "none":
        patch_analyze_data_for_tip_refinement(
            cal,
            refine_mode=str(args.tip_refine_mode),
            dt_step_px=float(args.tip_refine_dt_step_px),
            dt_max_step_px=int(args.tip_refine_max_step_px),
            grad_step_px=float(args.tip_refine_grad_step_px),
            grad_search_len_px=int(args.tip_refine_grad_search_len_px),
            mainray_fit_back_near_r=float(args.tip_refine_mainray_fit_back_near_r),
            mainray_fit_back_far_r=float(args.tip_refine_mainray_fit_back_far_r),
            mainray_anchor_back_r=float(args.tip_refine_mainray_anchor_back_r),
            mainray_ray_step_px=float(args.tip_refine_mainray_ray_step_px),
            mainray_ray_max_len_r=float(args.tip_refine_mainray_ray_max_len_r),
            parallel_section_near_r=float(args.tip_refine_parallel_section_near_r),
            parallel_section_far_r=float(args.tip_refine_parallel_section_far_r),
            parallel_scan_half_r=float(args.tip_refine_parallel_scan_half_r),
            parallel_num_sections=int(args.tip_refine_parallel_num_sections),
            parallel_cross_step_px=float(args.tip_refine_parallel_cross_step_px),
            parallel_ray_step_px=float(args.tip_refine_parallel_ray_step_px),
            parallel_ray_max_len_r=float(args.tip_refine_parallel_ray_max_len_r),
        )
        print(f"[INFO] Tip refinement enabled: {args.tip_refine_mode}")

    imgs = list_images(raw_folder)
    if not imgs:
        raise SystemExit(f"No images found in: {raw_folder}")

    first_img_path = imgs[0]
    img_bgr = cv2.imread(str(first_img_path), cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise SystemExit(f"Could not read first image: {first_img_path}")

    print(f"[INFO] Using first image for GUI: {first_img_path.name}")

    calib_path = Path(args.camera_calibration_file).expanduser().resolve()
    board_ref_path = Path(args.checkerboard_reference_image).expanduser().resolve()

    print(f"[INFO] Loading camera calibration: {calib_path}")
    cal.load_camera_calibration(str(calib_path))

    processed_dir = project_dir / "processed_image_data_folder"
    processed_dir.mkdir(parents=True, exist_ok=True)
    checkerboard_debug_path = processed_dir / "checkerboard_reference_debug.png"

    board_result = cal.estimate_board_reference_from_image(
        str(board_ref_path),
        inner_corners=args.checkerboard_inner_corners,
        square_size_mm=args.checkerboard_square_size_mm,
        use_undistort=(not args.checkerboard_no_undistort),
        draw_debug=True,
        save_debug_path=str(checkerboard_debug_path),
    )
    board_reference_debug_image = board_result.get("debug_image")

    apply_checkerboard_reference_corrections(
        cal,
        mm_scale=float(args.checkerboard_mm_scale_correction),
        flip_planar_x=(not bool(args.checkerboard_no_flip_planar_x)),
    )

    print(f"[INFO] Checkerboard reference estimated from: {board_ref_path}")

    analysis_crop = interactive_crop_from_image(
        img_bgr,
        default_crop=cal.default_analysis_crop,
    )
    cal.analysis_crop = dict(analysis_crop)

    if args.save_analysis_config:
        cfg = {}
        if hasattr(cal, "get_analysis_reference_info"):
            cfg = cal.get_analysis_reference_info()
        cfg["board_reference"] = collect_board_reference_info(cal)
        cfg_path = project_dir / "analysis_reference.json"
        with open(cfg_path, "w") as f:
            json.dump(_json_ready(cfg), f, indent=2)
        print(f"[INFO] Saved analysis config to: {cfg_path}")

    print("\n[INFO] Running analyze_data_batch (offline)...")
    cal.analyze_data_batch(threshold=int(args.threshold))

    tracked_rows, tracked_source = select_tracked_rows_for_analysis(cal, args)
    if tracked_rows.size == 0:
        raise RuntimeError(f"No tracked tip data found for source={tracked_source} after analyze_data_batch.")
    print(f"[INFO] Using tracked tip source for mm analysis: {tracked_source}")

    valid_mask = np.all(np.isfinite(tracked_rows[:, :2]), axis=1) if tracked_rows.ndim == 2 and tracked_rows.shape[1] >= 2 else np.zeros((0,), dtype=bool)
    num_valid = int(np.sum(valid_mask))

    print(f"[INFO] Total tracked rows: {tracked_rows.shape[0]}")
    print(f"[INFO] Valid tracked rows: {num_valid}")

    if num_valid == 0:
        raise RuntimeError(
            "Batch analysis completed, but no valid tracked tip points were produced. "
            "Filename parsing is now patched, so the next things to check are threshold/crop/tracking quality."
        )

    print("[INFO] Converting tracked tips to checkerboard-referenced mm...")
    tip_data = compute_tracked_tip_positions_mm(
        cal,
        tracked_rows,
        imgs,
        desired_tip_lookup=desired_tip_lookup,
    )

    if tip_data["mm_points"].shape[0] == 0:
        raise RuntimeError(
            "Tracked rows exist, but none could be converted into checkerboard-referenced mm coordinates."
        )

    if tip_data["desired_mm_points"].shape[0] == 0:
        raise RuntimeError(
            "Tracked rows exist, but no desired tip points were available for mean-aligned error analysis."
        )

    print("[INFO] Computing mean-aligned nearest-curve error metrics...")
    metrics = compute_error_metrics(
        tip_data["matched_measured_mm_points"],
        tip_data["desired_mm_points"],
    )

    records = attach_errors_to_records(
        records=tip_data["records"],
        valid_indices=tip_data["matched_valid_indices"],
        deltas_mm=metrics["deltas_mm"],
        errors_mm=metrics["errors_mm"],
        aligned_measured_mm_points=metrics["aligned_measured_mm_points"],
        closest_desired_mm_points=metrics["closest_desired_mm_points"],
    )
    metrics["quarter_error_stats"] = compute_quarter_error_stats(records)

    csv_path = processed_dir / "tracked_tip_positions_mm.csv"
    metrics_json_path = processed_dir / "tracked_tip_error_metrics.json"
    error_plot_path = processed_dir / "tracked_tip_error_vs_sample.png"
    hist_path = processed_dir / "tracked_tip_error_histogram.png"
    desired_vs_actual_plot_path = processed_dir / "tracked_tip_desired_vs_actual.png"

    save_tracked_tip_csv(csv_path, records)
    save_metrics_json(metrics_json_path, metrics, cal, args)
    save_error_plot(
        error_plot_path,
        records,
        title_prefix="Checkerboard-referenced ",
    )
    save_error_histogram(
        hist_path,
        metrics["errors_mm"],
        rmse_mm=metrics["rmse_mm"],
        title_prefix="Checkerboard-referenced ",
        bins=int(args.hist_bins),
    )
    save_desired_vs_actual_plot(
        desired_vs_actual_plot_path,
        records,
        reference_point_mm=metrics["reference_point_mm"],
        alignment_offset_mm=metrics["alignment_offset_mm"],
        title_prefix="Checkerboard-referenced ",
    )

    print(f"[INFO] Saved CSV: {csv_path}")
    print(f"[INFO] Saved metrics JSON: {metrics_json_path}")
    print(f"[INFO] Saved error plot: {error_plot_path}")
    print(f"[INFO] Saved histogram: {hist_path}")
    print(f"[INFO] Saved desired-vs-actual plot: {desired_vs_actual_plot_path}")

    if board_reference_debug_image is not None:
        try:
            annotated_board_path = processed_dir / "checkerboard_reference_annotated_analysis.png"
            draw_checkerboard_analysis_overlay(
                cal=cal,
                output_path=annotated_board_path,
                tracked_rows=tracked_rows,
                image_files=imgs,
                board_debug_image=board_reference_debug_image,
            )
            print(f"[INFO] Saved annotated checkerboard analysis image: {annotated_board_path}")
        except Exception as e:
            print(f"[WARN] Failed to create annotated checkerboard analysis image: {e}")

    print("\n========== RESULTS ==========")
    print(f"Valid samples: {metrics['num_samples']}")
    print(
        "Shared mean reference (desired mean / aligned tracked mean): "
        f"u = {metrics['reference_point_mm']['u_mean_mm']:.6f} mm, "
        f"z = {metrics['reference_point_mm']['z_mean_mm']:.6f} mm"
    )
    print(
        "Applied alignment offset to measured points: "
        f"du = {metrics['alignment_offset_mm']['du_mm']:.6f} mm, "
        f"dz = {metrics['alignment_offset_mm']['dz_mm']:.6f} mm"
    )
    print("Error definition: distance from each aligned measured point to the closest point on the desired curve.")
    print(f"RMSE:         {metrics['rmse_mm']:.6f} mm")
    print(f"Mean error:   {metrics['mean_error_mm']:.6f} mm")
    print(f"Std error:    {metrics['std_error_mm']:.6f} mm")
    print(f"Median error: {metrics['median_error_mm']:.6f} mm")
    print(f"Min error:    {metrics['min_error_mm']:.6f} mm")
    print(f"Max error:    {metrics['max_error_mm']:.6f} mm")
    print("Per-quarter error stats:")
    for quarter in ("q1_pull", "q2_release", "q3_pull", "q4_release"):
        qstats = metrics["quarter_error_stats"].get(quarter, {})
        if int(qstats.get("num_samples", 0)) == 0:
            print(f"  {quarter}: no samples")
            continue
        print(
            f"  {quarter}: n={int(qstats['num_samples'])}, "
            f"rmse={float(qstats['rmse_mm']):.6f} mm, "
            f"mean={float(qstats['mean_error_mm']):.6f} mm, "
            f"std={float(qstats['std_error_mm']):.6f} mm, "
            f"median={float(qstats['median_error_mm']):.6f} mm, "
            f"min={float(qstats['min_error_mm']):.6f} mm, "
            f"max={float(qstats['max_error_mm']):.6f} mm"
        )
    print("=============================\n")

    print("[DONE] Offline checkerboard tip error analysis complete.")
    print(f"Outputs are in: {processed_dir}")


if __name__ == "__main__":
    main()
