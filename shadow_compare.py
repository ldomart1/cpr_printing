#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
compare_ctr_shadow_calibrations.py

Compare two already-processed CTR shadow calibration folders to validate drift.

Typical use
-----------
python compare_ctr_shadow_calibrations.py \
    --cal_a /path/to/Calibration_A \
    --cal_b /path/to/Calibration_B \
    --label_a "before" \
    --label_b "after" \
    --out_dir /path/to/drift_check

You may pass either:
- a calibration project folder containing processed_image_data_folder/
- or processed_image_data_folder itself

Preferred input
---------------
This script first looks for:
    processed_image_data_folder/tip_locations_selected_mm.xlsx

That file is produced by CTR_Shadow_Calibration.postprocess_calibration_data().
If it is missing, the script falls back to tip_locations_selected.csv/.npy in
pixel units, or converts those pixels to mm if --mm_per_px is provided.

What it plots
-------------
1. Overlay of calibration A and B tip trajectories.
2. Overlay of x/z/angle versus shared motion index or B pull.
3. Error between A and B, assuming the rows/images correspond exactly.
4. Optional tracked-skeleton overlays if skeleton_points_selected_xy_px.npy/.csv
   exists in both folders.

The purpose is drift validation: with identical motion and same image count,
A-to-B residuals should stay small and should not show a strong trend versus
motion index or B pull.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


TIP_COLUMNS = [
    "x",
    "z",
    "orientation",
    "b_pull",
    "ntnl_pos",
    "tip_angle_deg",
    "motion_phase_code",
    "pass_idx",
]

PHASE_NAME = {
    0: "pull",
    1: "release",
}


@dataclass
class CalibrationData:
    label: str
    root_dir: Path
    processed_dir: Path
    tip: pd.DataFrame
    units: str
    source: str
    image_files: list[str] | None
    skeleton_xy: np.ndarray | None
    skeleton_units: str | None


def as_path(value: str | Path) -> Path:
    return Path(value).expanduser().resolve()


def resolve_processed_dir(path_like: str | Path) -> tuple[Path, Path]:
    """
    Return (project/root folder, processed_image_data_folder).

    Accepts either the project folder or processed_image_data_folder itself.
    """
    path = as_path(path_like)
    if not path.exists():
        raise FileNotFoundError(f"Folder does not exist: {path}")

    if path.name == "processed_image_data_folder":
        return path.parent, path

    processed = path / "processed_image_data_folder"
    if processed.is_dir():
        return path, processed

    raise FileNotFoundError(
        f"Could not find processed_image_data_folder in {path}. "
        "Pass either the project folder or processed_image_data_folder."
    )


def find_first_existing(folder: Path, names: Iterable[str]) -> Path | None:
    for name in names:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def read_analysis_reference_mm_per_px(project_dir: Path) -> float | None:
    """
    Try to recover manual-ruler scale from analysis_reference.json if present.

    This only applies to pixel fallback mode. Board-homography conversion cannot
    be reconstructed here; use tip_locations_selected_mm.xlsx for board-calibrated
    comparisons.
    """
    path = project_dir / "analysis_reference.json"
    if not path.is_file():
        return None

    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        return None

    for key_path in (
        ("ruler_mm_per_px",),
        ("board_reference", "board_mm_per_px_local"),
        ("board_reference", "camera_calib_meta", "ruler_mm_per_px"),
    ):
        node = data
        ok = True
        for key in key_path:
            if not isinstance(node, dict) or key not in node:
                ok = False
                break
            node = node[key]
        if ok and node is not None:
            try:
                value = float(node)
                if math.isfinite(value) and value > 0:
                    return value
            except Exception:
                pass

    return None


def load_image_file_column(processed_dir: Path, n_rows: int) -> list[str] | None:
    csv_path = processed_dir / "tip_locations_selected.csv"
    if not csv_path.is_file():
        return None
    try:
        df = pd.read_csv(csv_path)
    except Exception:
        return None
    if "image_file" not in df.columns or len(df) != n_rows:
        return None
    return [str(x) for x in df["image_file"].tolist()]


def normalize_tip_frame(df: pd.DataFrame, n_expected_cols: int = 8) -> pd.DataFrame:
    """
    Make sure the tip table has canonical column names.

    The mm xlsx is headerless. The csv already has names:
    tip_row, tip_column, orientation, ss_pos, ntnl_pos, tip_angle_deg,
    motion_phase_code, pass_idx, image_file.

    For comparison, this script canonicalizes:
    x = lateral/u coordinate
    z = vertical/z coordinate
    b_pull = ss_pos
    """
    df = df.copy()

    if {"tip_row", "tip_column"}.issubset(df.columns):
        rename = {
            "tip_column": "x",
            "tip_row": "z",
            "ss_pos": "b_pull",
        }
        df = df.rename(columns=rename)
        if "ntnl_pos" not in df.columns:
            df["ntnl_pos"] = np.nan
        if "tip_angle_deg" not in df.columns:
            df["tip_angle_deg"] = np.nan
        if "motion_phase_code" not in df.columns:
            df["motion_phase_code"] = 0
        if "pass_idx" not in df.columns:
            df["pass_idx"] = 1
        return df

    # Headerless dataframe from xlsx/npy.
    if all(isinstance(c, int) for c in df.columns):
        for idx, name in enumerate(TIP_COLUMNS[: min(n_expected_cols, df.shape[1])]):
            df = df.rename(columns={idx: name})
        return df

    # Already partly canonical.
    if "ss_pos" in df.columns and "b_pull" not in df.columns:
        df = df.rename(columns={"ss_pos": "b_pull"})
    return df


def load_tip_data(
    project_dir: Path,
    processed_dir: Path,
    *,
    mm_per_px: float | None,
    force_px: bool = False,
) -> tuple[pd.DataFrame, str, str, list[str] | None]:
    """
    Load selected tip data.

    Preference:
    1. tip_locations_selected_mm.xlsx, unless force_px=True
    2. tip_locations_selected.csv
    3. tip_locations_selected.npy
    """
    if not force_px:
        xlsx_path = processed_dir / "tip_locations_selected_mm.xlsx"
        if xlsx_path.is_file():
            try:
                df = pd.read_excel(xlsx_path, header=None)
            except ImportError as exc:
                raise RuntimeError(
                    f"Could not read {xlsx_path}. Install openpyxl or re-run with --force_px."
                ) from exc
            df = normalize_tip_frame(df)
            image_files = load_image_file_column(processed_dir, len(df))
            if image_files is not None:
                df["image_file"] = image_files
            return df, "mm", str(xlsx_path), image_files

    csv_path = processed_dir / "tip_locations_selected.csv"
    if csv_path.is_file():
        df = pd.read_csv(csv_path)
        df = normalize_tip_frame(df)
        image_files = [str(x) for x in df["image_file"].tolist()] if "image_file" in df.columns else None
        if mm_per_px is not None:
            # CSV pixel convention: x = tip_column, z = tip_row.
            # Match CTR postprocess fallback convention: z_mm = -tip_row * scale.
            df["x"] = df["x"].astype(float) * float(mm_per_px)
            df["z"] = -df["z"].astype(float) * float(mm_per_px)
            return df, "mm", f"{csv_path} scaled by mm_per_px={mm_per_px}", image_files
        return df, "px", str(csv_path), image_files

    npy_path = processed_dir / "tip_locations_selected.npy"
    if npy_path.is_file():
        arr = np.asarray(np.load(npy_path), dtype=float)
        if arr.ndim != 2 or arr.shape[1] < 5:
            raise ValueError(f"{npy_path} has unexpected shape {arr.shape}; expected Nx>=5")
        df = pd.DataFrame(arr, columns=TIP_COLUMNS[: arr.shape[1]])
        df = normalize_tip_frame(df)
        image_files = load_image_file_column(processed_dir, len(df))
        if image_files is not None:
            df["image_file"] = image_files
        if mm_per_px is not None:
            df["x"] = df["x"].astype(float) * float(mm_per_px)
            df["z"] = -df["z"].astype(float) * float(mm_per_px)
            return df, "mm", f"{npy_path} scaled by mm_per_px={mm_per_px}", image_files
        return df, "px", str(npy_path), image_files

    raise FileNotFoundError(
        f"No selected tip data found in {processed_dir}. Expected one of: "
        "tip_locations_selected_mm.xlsx, tip_locations_selected.csv, tip_locations_selected.npy"
    )


def load_skeleton_points(
    project_dir: Path,
    processed_dir: Path,
    *,
    mm_per_px: float | None,
    force_px: bool = False,
) -> tuple[np.ndarray | None, str | None]:
    """
    Load tracked skeleton points, if present.

    The analyzer saves skeleton_points_selected_xy_px.npy/.csv. If --mm_per_px is
    provided and force_px=False, this converts x_px/y_px to the same simple
    ruler-style convention used by postprocess fallback:
        x_mm = x_px * mm_per_px
        z_mm = -y_px * mm_per_px

    For board-homography-calibrated skeletons, compare pixel skeletons unless
    you export your own skeleton mm arrays.
    """
    npy_path = processed_dir / "skeleton_points_selected_xy_px.npy"
    csv_path = processed_dir / "skeleton_points_selected_xy_px.csv"

    arr = None
    if npy_path.is_file():
        arr = np.asarray(np.load(npy_path), dtype=float)
    elif csv_path.is_file():
        df = pd.read_csv(csv_path)
        xy_cols = [c for c in df.columns if c.startswith("p") and (c.endswith("_x_px") or c.endswith("_y_px"))]
        if xy_cols:
            # Sort by p00_x, p00_y, p01_x, p01_y, ...
            xy_cols = sorted(xy_cols)
            raw = df[xy_cols].to_numpy(dtype=float)
            if raw.shape[1] % 2 == 0:
                arr = raw.reshape(raw.shape[0], raw.shape[1] // 2, 2)

    if arr is None:
        return None, None

    if arr.ndim != 3 or arr.shape[2] != 2:
        print(f"[WARN] Ignoring skeleton data in {processed_dir}; unexpected shape {arr.shape}", file=sys.stderr)
        return None, None

    if mm_per_px is not None and not force_px:
        scaled = arr.copy()
        scaled[:, :, 0] = scaled[:, :, 0] * float(mm_per_px)
        scaled[:, :, 1] = -scaled[:, :, 1] * float(mm_per_px)
        return scaled, "mm"

    return arr, "px"


def load_calibration(
    folder: str | Path,
    label: str,
    *,
    mm_per_px: float | None,
    force_px: bool = False,
) -> CalibrationData:
    project_dir, processed_dir = resolve_processed_dir(folder)

    scale = mm_per_px
    if scale is None:
        scale = read_analysis_reference_mm_per_px(project_dir)

    tip, units, source, image_files = load_tip_data(
        project_dir,
        processed_dir,
        mm_per_px=scale,
        force_px=force_px,
    )
    skeleton, skeleton_units = load_skeleton_points(
        project_dir,
        processed_dir,
        mm_per_px=scale if units == "mm" else None,
        force_px=force_px,
    )

    return CalibrationData(
        label=label,
        root_dir=project_dir,
        processed_dir=processed_dir,
        tip=tip,
        units=units,
        source=source,
        image_files=image_files,
        skeleton_xy=skeleton,
        skeleton_units=skeleton_units,
    )


def circular_angle_delta_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Signed minimal angular difference b - a in degrees, in [-180, 180).
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return (b - a + 180.0) % 360.0 - 180.0


def phase_label(code: float, pass_idx: float | None = None) -> str:
    try:
        c = int(round(float(code)))
    except Exception:
        c = 999999
    name = PHASE_NAME.get(c, f"phase_{code}")
    if pass_idx is None or not np.isfinite(pass_idx):
        return name
    try:
        return f"{name} pass {int(round(float(pass_idx)))}"
    except Exception:
        return name


def finite_rms(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(v * v)))


def finite_mean(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.mean(v))


def finite_max_abs(values: np.ndarray) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    return float(np.max(np.abs(v)))


def check_same_length(a: CalibrationData, b: CalibrationData) -> None:
    if len(a.tip) != len(b.tip):
        raise ValueError(
            f"Tip row count mismatch: {a.label} has {len(a.tip)} rows; "
            f"{b.label} has {len(b.tip)} rows. This script assumes same movement "
            "and same number/order of images."
        )


def warn_if_motion_metadata_differs(a: CalibrationData, b: CalibrationData, atol: float = 1e-6) -> None:
    cols = ["orientation", "b_pull", "motion_phase_code", "pass_idx"]
    for col in cols:
        if col not in a.tip.columns or col not in b.tip.columns:
            continue
        av = pd.to_numeric(a.tip[col], errors="coerce").to_numpy(dtype=float)
        bv = pd.to_numeric(b.tip[col], errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(av) & np.isfinite(bv)
        if valid.any() and not np.allclose(av[valid], bv[valid], atol=atol, rtol=0):
            maxdiff = finite_max_abs(bv[valid] - av[valid])
            print(
                f"[WARN] Motion metadata column {col!r} differs between folders. "
                f"max |delta|={maxdiff:.6g}. Row-by-row comparison will still run.",
                file=sys.stderr,
            )

    if a.image_files and b.image_files:
        n_mismatch = sum(aa != bb for aa, bb in zip(a.image_files, b.image_files))
        if n_mismatch:
            print(
                f"[WARN] image_file ordering differs for {n_mismatch}/{len(a.image_files)} rows. "
                "This script still compares by row index.",
                file=sys.stderr,
            )


def apply_alignment(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """
    Align x/z coordinates to isolate different drift components.

    none:
        Absolute A-B drift.
    first:
        Subtract each dataset's first valid point.
    mean:
        Subtract each dataset's mean x/z point.
    group_first:
        Subtract each phase/pass/orientation group's first valid point.
    """
    df = df.copy()
    mode = str(mode).lower().strip()

    if mode == "none":
        return df

    if mode == "first":
        valid = np.isfinite(df["x"].to_numpy(dtype=float)) & np.isfinite(df["z"].to_numpy(dtype=float))
        if not valid.any():
            return df
        idx = int(np.where(valid)[0][0])
        df["x"] = df["x"] - float(df.iloc[idx]["x"])
        df["z"] = df["z"] - float(df.iloc[idx]["z"])
        return df

    if mode == "mean":
        df["x"] = df["x"] - finite_mean(df["x"].to_numpy(dtype=float))
        df["z"] = df["z"] - finite_mean(df["z"].to_numpy(dtype=float))
        return df

    if mode == "group_first":
        group_cols = [c for c in ["motion_phase_code", "pass_idx", "orientation"] if c in df.columns]
        if not group_cols:
            return apply_alignment(df, "first")

        for _, idxs in df.groupby(group_cols, dropna=False).groups.items():
            idxs = list(idxs)
            sub = df.loc[idxs]
            valid = np.isfinite(sub["x"].to_numpy(dtype=float)) & np.isfinite(sub["z"].to_numpy(dtype=float))
            if not valid.any():
                continue
            first_idx = idxs[int(np.where(valid)[0][0])]
            x0 = float(df.loc[first_idx, "x"])
            z0 = float(df.loc[first_idx, "z"])
            df.loc[idxs, "x"] = df.loc[idxs, "x"].astype(float) - x0
            df.loc[idxs, "z"] = df.loc[idxs, "z"].astype(float) - z0
        return df

    raise ValueError(f"Unknown alignment mode: {mode}")


def build_error_table(a: CalibrationData, b: CalibrationData, alignment: str) -> pd.DataFrame:
    check_same_length(a, b)
    warn_if_motion_metadata_differs(a, b)

    if a.units != b.units:
        raise ValueError(
            f"Unit mismatch: {a.label} is {a.units}, {b.label} is {b.units}. "
            "Use --force_px for both, or provide --mm_per_px / postprocess both folders."
        )

    df_a = apply_alignment(a.tip, alignment)
    df_b = apply_alignment(b.tip, alignment)

    out = pd.DataFrame()
    out["row_index"] = np.arange(len(df_a), dtype=int)

    for col in ["image_file", "orientation", "b_pull", "ntnl_pos", "motion_phase_code", "pass_idx"]:
        if col in df_a.columns:
            out[col] = df_a[col].to_numpy()
        elif col in df_b.columns:
            out[col] = df_b[col].to_numpy()

    out[f"x_{a.label}"] = pd.to_numeric(df_a["x"], errors="coerce").to_numpy(dtype=float)
    out[f"z_{a.label}"] = pd.to_numeric(df_a["z"], errors="coerce").to_numpy(dtype=float)
    out[f"x_{b.label}"] = pd.to_numeric(df_b["x"], errors="coerce").to_numpy(dtype=float)
    out[f"z_{b.label}"] = pd.to_numeric(df_b["z"], errors="coerce").to_numpy(dtype=float)

    out["dx"] = out[f"x_{b.label}"] - out[f"x_{a.label}"]
    out["dz"] = out[f"z_{b.label}"] - out[f"z_{a.label}"]
    out["error_norm"] = np.sqrt(out["dx"] ** 2 + out["dz"] ** 2)

    if "tip_angle_deg" in df_a.columns and "tip_angle_deg" in df_b.columns:
        aa = pd.to_numeric(df_a["tip_angle_deg"], errors="coerce").to_numpy(dtype=float)
        bb = pd.to_numeric(df_b["tip_angle_deg"], errors="coerce").to_numpy(dtype=float)
        out[f"tip_angle_deg_{a.label}"] = aa
        out[f"tip_angle_deg_{b.label}"] = bb
        out["d_tip_angle_deg"] = circular_angle_delta_deg(aa, bb)

    return out


def summarize_errors(error_df: pd.DataFrame, units: str, alignment: str) -> pd.DataFrame:
    rows = []

    def add_row(scope: str, data: pd.DataFrame):
        rows.append(
            {
                "scope": scope,
                "n_rows": int(len(data)),
                "units": units,
                "alignment": alignment,
                "mean_dx": finite_mean(data["dx"].to_numpy(dtype=float)),
                "mean_dz": finite_mean(data["dz"].to_numpy(dtype=float)),
                "rms_dx": finite_rms(data["dx"].to_numpy(dtype=float)),
                "rms_dz": finite_rms(data["dz"].to_numpy(dtype=float)),
                "mean_error_norm": finite_mean(data["error_norm"].to_numpy(dtype=float)),
                "rms_error_norm": finite_rms(data["error_norm"].to_numpy(dtype=float)),
                "max_error_norm": finite_max_abs(data["error_norm"].to_numpy(dtype=float)),
                "rms_angle_deg": finite_rms(data["d_tip_angle_deg"].to_numpy(dtype=float))
                if "d_tip_angle_deg" in data.columns
                else np.nan,
                "max_abs_angle_deg": finite_max_abs(data["d_tip_angle_deg"].to_numpy(dtype=float))
                if "d_tip_angle_deg" in data.columns
                else np.nan,
            }
        )

    add_row("all", error_df)

    group_cols = [c for c in ["motion_phase_code", "pass_idx", "orientation"] if c in error_df.columns]
    if group_cols:
        for key, sub in error_df.groupby(group_cols, dropna=False):
            if not isinstance(key, tuple):
                key = (key,)
            parts = []
            key_map = dict(zip(group_cols, key))
            if "motion_phase_code" in key_map:
                parts.append(phase_label(key_map["motion_phase_code"], key_map.get("pass_idx", np.nan)))
            if "orientation" in key_map:
                try:
                    parts.append(f"orientation={int(round(float(key_map['orientation'])))}")
                except Exception:
                    parts.append(f"orientation={key_map['orientation']}")
            add_row(" / ".join(parts) if parts else str(key), sub)

    return pd.DataFrame(rows)


def choose_x_axis(error_df: pd.DataFrame, mode: str) -> tuple[np.ndarray, str]:
    mode = str(mode).lower().strip()
    if mode == "b":
        if "b_pull" not in error_df.columns:
            print("[WARN] --x_axis b requested, but b_pull column is missing. Falling back to row index.", file=sys.stderr)
        else:
            b_vals = pd.to_numeric(error_df["b_pull"], errors="coerce").to_numpy(dtype=float)
            if np.isfinite(b_vals).any():
                return b_vals, "B pull (mm)"
    if mode == "index":
        return error_df["row_index"].to_numpy(dtype=float), "row / image index"
    raise ValueError(f"Unknown --x_axis {mode!r}; choose index or b")


def configure_axes(ax, title: str, xlabel: str | None = None, ylabel: str | None = None) -> None:
    ax.set_title(title, fontsize=11)
    if xlabel:
        ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25, linewidth=0.7)
    ax.tick_params(axis="both", labelsize=9)


def plot_by_groups(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    meta_df: pd.DataFrame,
    *,
    label: str,
    color: str,
    linestyle: str = "-",
    linewidth: float = 1.8,
    alpha: float = 0.88,
    marker: str | None = None,
    markersize: float = 3.0,
) -> None:
    """
    Plot continuous lines within phase/pass/orientation groups so release/pull
    transitions do not create unreadable long jump lines.
    """
    group_cols = [c for c in ["motion_phase_code", "pass_idx", "orientation"] if c in meta_df.columns]
    if not group_cols:
        ax.plot(x, y, label=label, color=color, linestyle=linestyle, linewidth=linewidth, alpha=alpha, marker=marker, markersize=markersize)
        return

    used = False
    for _, idxs in meta_df.groupby(group_cols, dropna=False, sort=False).groups.items():
        idxs = np.asarray(list(idxs), dtype=int)
        if idxs.size == 0:
            continue
        order = np.argsort(x[idxs])
        ii = idxs[order]
        ax.plot(
            x[ii],
            y[ii],
            label=label if not used else None,
            color=color,
            linestyle=linestyle,
            linewidth=linewidth,
            alpha=alpha,
            marker=marker,
            markersize=markersize,
        )
        used = True


def make_main_plot(
    error_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    *,
    a: CalibrationData,
    b: CalibrationData,
    x_axis_mode: str,
    output_path: Path,
    dpi: int,
    show: bool,
) -> None:
    x_axis, x_label = choose_x_axis(error_df, x_axis_mode)
    units = a.units

    fig, axs = plt.subplots(2, 3, figsize=(18, 10), constrained_layout=True)
    ax_traj, ax_x, ax_z, ax_vec, ax_norm, ax_ang = axs.ravel()

    # Colors intentionally high contrast for paper/lab readability.
    color_a = "#1f77b4"
    color_b = "#d62728"
    color_dx = "#9467bd"
    color_dz = "#2ca02c"
    color_norm = "#111111"

    xa = error_df[f"x_{a.label}"].to_numpy(dtype=float)
    za = error_df[f"z_{a.label}"].to_numpy(dtype=float)
    xb = error_df[f"x_{b.label}"].to_numpy(dtype=float)
    zb = error_df[f"z_{b.label}"].to_numpy(dtype=float)

    # X-Z overlay.
    plot_by_groups(ax_traj, xa, za, error_df, label=a.label, color=color_a, linestyle="-", marker="o", markersize=2.8)
    plot_by_groups(ax_traj, xb, zb, error_df, label=b.label, color=color_b, linestyle="--", marker="s", markersize=2.6)
    ax_traj.set_aspect("equal", adjustable="datalim")
    configure_axes(ax_traj, "Tip trajectory overlay", f"x / lateral ({units})", f"z / vertical ({units})")
    ax_traj.legend(fontsize=9, loc="best")

    # X and Z versus motion.
    plot_by_groups(ax_x, x_axis, xa, error_df, label=a.label, color=color_a, linestyle="-")
    plot_by_groups(ax_x, x_axis, xb, error_df, label=b.label, color=color_b, linestyle="--")
    configure_axes(ax_x, "Lateral coordinate overlay", x_label, f"x ({units})")
    ax_x.legend(fontsize=9, loc="best")

    plot_by_groups(ax_z, x_axis, za, error_df, label=a.label, color=color_a, linestyle="-")
    plot_by_groups(ax_z, x_axis, zb, error_df, label=b.label, color=color_b, linestyle="--")
    configure_axes(ax_z, "Vertical coordinate overlay", x_label, f"z ({units})")
    ax_z.legend(fontsize=9, loc="best")

    # Component errors.
    ax_vec.axhline(0.0, color="0.4", linewidth=0.9)
    ax_vec.plot(x_axis, error_df["dx"], label=f"dx = {b.label} - {a.label}", color=color_dx, linewidth=1.7)
    ax_vec.plot(x_axis, error_df["dz"], label=f"dz = {b.label} - {a.label}", color=color_dz, linewidth=1.7)
    configure_axes(ax_vec, "Signed component error", x_label, f"error ({units})")
    ax_vec.legend(fontsize=9, loc="best")

    # Norm error.
    ax_norm.plot(x_axis, error_df["error_norm"], label="2D error norm", color=color_norm, linewidth=2.0)
    ax_norm.axhline(float(summary_df.iloc[0]["rms_error_norm"]), color="0.35", linestyle=":", linewidth=1.4, label="RMS")
    configure_axes(ax_norm, "A-to-B drift magnitude", x_label, f"|error| ({units})")
    ax_norm.legend(fontsize=9, loc="best")

    s0 = summary_df.iloc[0]
    stats_text = (
        f"All rows, alignment={s0['alignment']}\n"
        f"N = {int(s0['n_rows'])}\n"
        f"mean dx = {s0['mean_dx']:.4g} {units}\n"
        f"mean dz = {s0['mean_dz']:.4g} {units}\n"
        f"RMS |err| = {s0['rms_error_norm']:.4g} {units}\n"
        f"max |err| = {s0['max_error_norm']:.4g} {units}"
    )
    ax_norm.text(
        0.02,
        0.98,
        stats_text,
        transform=ax_norm.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.82, edgecolor="0.6"),
    )

    # Angle.
    if "d_tip_angle_deg" in error_df.columns:
        aa = error_df.get(f"tip_angle_deg_{a.label}", pd.Series(np.nan, index=error_df.index)).to_numpy(dtype=float)
        bb = error_df.get(f"tip_angle_deg_{b.label}", pd.Series(np.nan, index=error_df.index)).to_numpy(dtype=float)
        ax_ang.plot(x_axis, aa, label=f"{a.label} tip angle", color=color_a, linestyle="-", linewidth=1.4, alpha=0.85)
        ax_ang.plot(x_axis, bb, label=f"{b.label} tip angle", color=color_b, linestyle="--", linewidth=1.4, alpha=0.85)
        ax_ang_twin = ax_ang.twinx()
        ax_ang_twin.plot(x_axis, error_df["d_tip_angle_deg"], label="angle delta", color="0.05", linewidth=1.6, alpha=0.75)
        ax_ang_twin.set_ylabel("angle delta (deg)", fontsize=9)
        ax_ang_twin.tick_params(axis="y", labelsize=9)
        configure_axes(ax_ang, "Tip angle overlay + delta", x_label, "tip angle (deg)")
        lines1, labels1 = ax_ang.get_legend_handles_labels()
        lines2, labels2 = ax_ang_twin.get_legend_handles_labels()
        ax_ang.legend(lines1 + lines2, labels1 + labels2, fontsize=8, loc="best")
    else:
        ax_ang.text(0.5, 0.5, "No tip angle columns found", ha="center", va="center", transform=ax_ang.transAxes)
        configure_axes(ax_ang, "Tip angle overlay + delta", x_label, "deg")

    for ax in axs.ravel():
        ax.xaxis.set_major_locator(MaxNLocator(nbins=7))

    fig.suptitle(
        f"CTR shadow calibration drift check: {a.label} vs {b.label}\n"
        f"{a.processed_dir}  |  {b.processed_dir}",
        fontsize=13,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved main overlay/error plot: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def evenly_spaced_indices(n: int, count: int) -> np.ndarray:
    if n <= 0:
        return np.asarray([], dtype=int)
    count = int(max(1, min(count, n)))
    return np.unique(np.linspace(0, n - 1, count).round().astype(int))


def skeleton_error_table(a: CalibrationData, b: CalibrationData) -> pd.DataFrame | None:
    if a.skeleton_xy is None or b.skeleton_xy is None:
        return None
    if a.skeleton_xy.shape != b.skeleton_xy.shape:
        print(
            f"[WARN] Skeleton shape mismatch: {a.skeleton_xy.shape} vs {b.skeleton_xy.shape}. "
            "Skipping skeleton error table.",
            file=sys.stderr,
        )
        return None
    if a.skeleton_units != b.skeleton_units:
        print(
            f"[WARN] Skeleton unit mismatch: {a.skeleton_units} vs {b.skeleton_units}. "
            "Skipping skeleton error table.",
            file=sys.stderr,
        )
        return None

    delta = b.skeleton_xy - a.skeleton_xy
    norm = np.linalg.norm(delta, axis=2)

    rows = []
    for i in range(norm.shape[0]):
        rows.append(
            {
                "row_index": i,
                "mean_skeleton_error": finite_mean(norm[i]),
                "rms_skeleton_error": finite_rms(norm[i]),
                "max_skeleton_error": finite_max_abs(norm[i]),
                "units": a.skeleton_units,
            }
        )
    return pd.DataFrame(rows)


def make_skeleton_plot(
    a: CalibrationData,
    b: CalibrationData,
    *,
    output_path: Path,
    sample_count: int,
    dpi: int,
    show: bool,
) -> None:
    if a.skeleton_xy is None or b.skeleton_xy is None:
        return
    if a.skeleton_xy.shape != b.skeleton_xy.shape:
        return
    if a.skeleton_units != b.skeleton_units:
        return

    n_frames = a.skeleton_xy.shape[0]
    sample_idxs = evenly_spaced_indices(n_frames, sample_count)
    if sample_idxs.size == 0:
        return

    ncols = min(3, sample_idxs.size)
    nrows = int(math.ceil(sample_idxs.size / ncols))
    fig, axs = plt.subplots(nrows, ncols, figsize=(5.8 * ncols, 4.8 * nrows), squeeze=False, constrained_layout=True)

    units = a.skeleton_units or "px"
    color_a = "#1f77b4"
    color_b = "#d62728"

    for ax in axs.ravel():
        ax.axis("off")

    delta = b.skeleton_xy - a.skeleton_xy
    norm = np.linalg.norm(delta, axis=2)

    for ax, idx in zip(axs.ravel(), sample_idxs):
        pa = a.skeleton_xy[idx]
        pb = b.skeleton_xy[idx]
        valid_a = np.all(np.isfinite(pa), axis=1)
        valid_b = np.all(np.isfinite(pb), axis=1)

        if valid_a.any():
            ax.plot(pa[valid_a, 0], pa[valid_a, 1], "-o", color=color_a, linewidth=2.0, markersize=3.0, label=a.label, alpha=0.88)
        if valid_b.any():
            ax.plot(pb[valid_b, 0], pb[valid_b, 1], "--s", color=color_b, linewidth=2.0, markersize=3.0, label=b.label, alpha=0.88)

        valid_both = valid_a & valid_b
        if valid_both.any():
            # Draw sparse displacement connectors so drift direction is readable.
            connector_idxs = np.where(valid_both)[0]
            if connector_idxs.size > 8:
                connector_idxs = connector_idxs[np.linspace(0, connector_idxs.size - 1, 8).round().astype(int)]
            for pidx in connector_idxs:
                ax.plot([pa[pidx, 0], pb[pidx, 0]], [pa[pidx, 1], pb[pidx, 1]], color="0.25", alpha=0.45, linewidth=1.0)

        mean_err = finite_mean(norm[idx])
        max_err = finite_max_abs(norm[idx])
        ax.set_title(f"row {idx}: mean={mean_err:.3g}, max={max_err:.3g} {units}", fontsize=10)
        ax.set_xlabel(f"x ({units})")
        ax.set_ylabel(f"z/y ({units})")
        ax.grid(True, alpha=0.25)
        ax.axis("equal")
        ax.legend(fontsize=8, loc="best")
        ax.axis("on")

    fig.suptitle(f"Tracked skeleton overlays: {a.label} vs {b.label}", fontsize=13)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    print(f"Saved skeleton overlay plot: {output_path}")

    if show:
        plt.show()
    else:
        plt.close(fig)


def write_readme(
    output_dir: Path,
    *,
    a: CalibrationData,
    b: CalibrationData,
    error_csv: Path,
    summary_csv: Path,
    main_plot: Path,
    skeleton_csv: Path | None,
    skeleton_plot: Path | None,
    alignment: str,
) -> None:
    readme = output_dir / "README_drift_check.txt"
    lines = [
        "CTR shadow calibration drift check",
        "==================================",
        "",
        f"Calibration A label: {a.label}",
        f"Calibration A root: {a.root_dir}",
        f"Calibration A processed: {a.processed_dir}",
        f"Calibration A tip source: {a.source}",
        "",
        f"Calibration B label: {b.label}",
        f"Calibration B root: {b.root_dir}",
        f"Calibration B processed: {b.processed_dir}",
        f"Calibration B tip source: {b.source}",
        "",
        f"Tip units: {a.units}",
        f"Alignment: {alignment}",
        "",
        "Outputs:",
        f"- Per-row tip error CSV: {error_csv.name}",
        f"- Summary CSV: {summary_csv.name}",
        f"- Overlay/error plot: {main_plot.name}",
    ]
    if skeleton_csv is not None:
        lines.append(f"- Per-row skeleton error CSV: {skeleton_csv.name}")
    if skeleton_plot is not None:
        lines.append(f"- Skeleton overlay plot: {skeleton_plot.name}")
    lines.extend(
        [
            "",
            "Interpretation notes:",
            "- dx, dz, and error_norm are Calibration B minus Calibration A.",
            "- A fixed nonzero mean dx/dz suggests registration/camera/setup offset.",
            "- A trend in error_norm versus row index or B pull suggests drift during the sweep.",
            "- group_first alignment removes per-phase/pass/orientation starting offsets and highlights shape mismatch.",
        ]
    )
    readme.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved README: {readme}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare two CTR shadow calibration processed folders to validate drift."
    )
    parser.add_argument("--cal_a", required=True, help="Calibration A project folder or processed_image_data_folder.")
    parser.add_argument("--cal_b", required=True, help="Calibration B project folder or processed_image_data_folder.")
    parser.add_argument("--label_a", default="cal_A", help="Label for first calibration.")
    parser.add_argument("--label_b", default="cal_B", help="Label for second calibration.")
    parser.add_argument(
        "--out_dir",
        default=None,
        help="Output directory. Default: ./ctr_drift_check_<label_a>_vs_<label_b>",
    )
    parser.add_argument(
        "--alignment",
        default="none",
        choices=["none", "first", "mean", "group_first"],
        help=(
            "Coordinate alignment before computing error. "
            "none=absolute drift; first/mean remove global offsets; "
            "group_first removes each phase/pass/orientation start offset."
        ),
    )
    parser.add_argument(
        "--x_axis",
        default="index",
        choices=["index", "b"],
        help="X-axis for time-series plots: row index or B pull.",
    )
    parser.add_argument(
        "--mm_per_px",
        type=float,
        default=None,
        help=(
            "Fallback scale for pixel files when tip_locations_selected_mm.xlsx is absent. "
            "If omitted, the script tries analysis_reference.json; otherwise it compares in pixels."
        ),
    )
    parser.add_argument(
        "--force_px",
        action="store_true",
        help="Ignore tip_locations_selected_mm.xlsx and compare raw selected tip pixels.",
    )
    parser.add_argument(
        "--skeleton_samples",
        type=int,
        default=6,
        help="Number of representative skeleton frames to overlay.",
    )
    parser.add_argument("--dpi", type=int, default=180, help="Figure DPI.")
    parser.add_argument("--show", action="store_true", help="Show figures interactively after saving.")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    out_dir = as_path(args.out_dir) if args.out_dir else as_path(
        f"ctr_drift_check_{args.label_a}_vs_{args.label_b}"
    )
    out_dir.mkdir(parents=True, exist_ok=True)

    a = load_calibration(
        args.cal_a,
        args.label_a,
        mm_per_px=args.mm_per_px,
        force_px=bool(args.force_px),
    )
    b = load_calibration(
        args.cal_b,
        args.label_b,
        mm_per_px=args.mm_per_px,
        force_px=bool(args.force_px),
    )

    print("\nLoaded calibration folders")
    print(f"  {a.label}: {a.processed_dir}")
    print(f"    source={a.source}")
    print(f"    rows={len(a.tip)}, units={a.units}")
    print(f"  {b.label}: {b.processed_dir}")
    print(f"    source={b.source}")
    print(f"    rows={len(b.tip)}, units={b.units}")

    error_df = build_error_table(a, b, args.alignment)
    summary_df = summarize_errors(error_df, a.units, args.alignment)

    error_csv = out_dir / "drift_error_by_row.csv"
    summary_csv = out_dir / "drift_summary.csv"
    error_df.to_csv(error_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)

    print(f"\nSaved per-row error CSV: {error_csv}")
    print(f"Saved summary CSV: {summary_csv}")
    print("\nSummary:")
    with pd.option_context("display.max_columns", None, "display.width", 180):
        print(summary_df.head(12).to_string(index=False))

    main_plot = out_dir / "drift_overlay_and_error.png"
    make_main_plot(
        error_df,
        summary_df,
        a=a,
        b=b,
        x_axis_mode=args.x_axis,
        output_path=main_plot,
        dpi=int(args.dpi),
        show=bool(args.show),
    )

    skel_csv = None
    skel_plot = None
    skel_df = skeleton_error_table(a, b)
    if skel_df is not None:
        skel_csv = out_dir / "drift_skeleton_error_by_row.csv"
        skel_df.to_csv(skel_csv, index=False)
        print(f"Saved skeleton error CSV: {skel_csv}")

        skel_plot = out_dir / "drift_skeleton_overlay.png"
        make_skeleton_plot(
            a,
            b,
            output_path=skel_plot,
            sample_count=int(args.skeleton_samples),
            dpi=int(args.dpi),
            show=bool(args.show),
        )

    write_readme(
        out_dir,
        a=a,
        b=b,
        error_csv=error_csv,
        summary_csv=summary_csv,
        main_plot=main_plot,
        skeleton_csv=skel_csv,
        skeleton_plot=skel_plot,
        alignment=args.alignment,
    )

    print("\nDone.")
    print(f"Output directory: {out_dir}")


if __name__ == "__main__":
    main()
