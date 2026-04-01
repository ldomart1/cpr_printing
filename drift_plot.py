#!/usr/bin/env python3
import argparse
import glob
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.colors import LinearSegmentedColormap

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


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


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
):
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
    idx_per_frame = max(1, int(np.ceil(idx_per_second / float(fps))))
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


def _make_checkpoint_grid(num_items: int):
    num_cols = 5
    num_rows = 2
    fig, axes = plt.subplots(
        num_rows,
        num_cols,
        figsize=(5.2 * num_cols, 4.6 * num_rows),
        squeeze=False,
    )
    return fig, axes


def validate_dataframe(df: pd.DataFrame) -> None:
    required_columns = {
        "checkpoint",
        "probe_idx",
        "sequence_idx",
        "tip_row_px",
        "tip_col_px",
        "tip_u_mm",
        "tip_z_mm",
    }
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")


def find_csv_in_folder(folder: str, csv_name: str = None) -> str:
    folder = os.path.abspath(folder)
    if not os.path.isdir(folder):
        raise FileNotFoundError(f"Folder not found: {folder}")

    if csv_name is not None:
        candidate = os.path.join(folder, csv_name)
        if not os.path.isfile(candidate):
            raise FileNotFoundError(f"CSV not found in folder: {candidate}")
        return candidate

    exact = os.path.join(folder, "tip_drift_measurements.csv")
    if os.path.isfile(exact):
        return exact

    csvs = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if len(csvs) == 0:
        raise FileNotFoundError(f"No CSV files found in folder: {folder}")
    if len(csvs) == 1:
        return csvs[0]

    raise ValueError(
        f"Multiple CSV files found in {folder}. Use --csv-name to specify which one.\n"
        f"Found: {csvs}"
    )


def combine_csv_files(csv_paths: List[str], combined_csv_path: str) -> pd.DataFrame:
    dfs = []
    seq_offset = 0

    for dataset_idx, csv_path in enumerate(csv_paths, start=1):
        df = pd.read_csv(csv_path)
        if df.empty:
            continue

        validate_dataframe(df)

        df = df.copy()
        df["source_csv"] = os.path.abspath(csv_path)
        df["source_dataset_idx"] = dataset_idx

        # Offset sequence index so appended datasets continue in time
        if "sequence_idx" in df.columns:
            df["sequence_idx"] = pd.to_numeric(df["sequence_idx"], errors="coerce")
            if df["sequence_idx"].notna().any():
                df["sequence_idx"] = df["sequence_idx"] + seq_offset
                seq_offset = int(df["sequence_idx"].max())

        dfs.append(df)

    if not dfs:
        raise ValueError("No valid data found across input CSV files.")

    combined_df = pd.concat(dfs, ignore_index=True)
    combined_df.to_csv(combined_csv_path, index=False)
    return combined_df


def make_drift_plots_from_df(df: pd.DataFrame, plots_dir: str) -> List[str]:
    if df.empty:
        raise ValueError("Combined dataframe is empty; no drift plots can be generated.")

    output_paths: List[str] = []

    drift_grid_path = os.path.join(plots_dir, "drift_by_checkpoint.png")
    fig, axes = _make_checkpoint_grid(len(CHECKPOINT_ORDER))
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)

    for ax, checkpoint_name in zip(axes.flat, CHECKPOINT_ORDER):
        dfi = df[df["checkpoint"] == checkpoint_name].copy()
        dfi = dfi.sort_values(["probe_idx", "sequence_idx"])

        use_mm = np.isfinite(dfi["tip_u_mm"]).any() and np.isfinite(dfi["tip_z_mm"]).any()

        if not dfi.empty and use_mm:
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
                    markersize=4.8,
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
                    markersize=4.0,
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
                    markersize=4.8,
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
                    markersize=4.0,
                    label=f"Probe {probe_idx} Δy",
                )

        else:
            use_mm = np.isfinite(df["tip_u_mm"]).any() and np.isfinite(df["tip_z_mm"]).any()
            y1_label = "Δu (mm)" if use_mm else "Δx (px)"
            y2_label = "Δz (mm)" if use_mm else "Δy (px)"

        _apply_dark_axes_style(
            ax,
            title=checkpoint_name.replace("_", " ").title(),
            xlabel="Combined sequence index",
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

    # Combined scatter plot
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

    _apply_dark_axes_style(ax, "Tracked tip positions", xlabel, ylabel)

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

    # 4-panel density plot
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
                s=24,
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
    hist_rows = int(np.ceil(len(CHECKPOINT_ORDER) / 2.0))
    fig = plt.figure(figsize=(15.2, 4.8 * hist_rows))
    fig.patch.set_facecolor("none")
    fig.patch.set_alpha(0.0)
    outer_gs = fig.add_gridspec(hist_rows, 2, hspace=0.28, wspace=0.18)

    use_mm = np.isfinite(df["tip_u_mm"]).any() and np.isfinite(df["tip_z_mm"]).any()
    x_col = "tip_u_mm" if use_mm else "tip_col_px"
    y_col = "tip_z_mm" if use_mm else "tip_row_px"
    u_label = "u error (mm)" if use_mm else "x error (px)"
    z_label = "z error (mm)" if use_mm else "y error (px)"

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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Combine one or more drift CSV files and generate plots."
    )

    parser.add_argument(
        "--csv",
        type=str,
        nargs="+",
        default=None,
        help="One or more input CSV files.",
    )

    parser.add_argument(
        "--folders",
        nargs="+",
        default=None,
        help="One or more folders containing CSV files to combine.",
    )

    parser.add_argument(
        "--csv-name",
        type=str,
        default=None,
        help="CSV filename to look for inside each folder. Example: tip_drift_measurements.csv",
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save combined CSV and plots.",
    )

    parser.add_argument(
        "--combined-name",
        type=str,
        default="combined_tip_drift_measurements.csv",
        help="Filename for the combined CSV.",
    )

    args = parser.parse_args()

    if (args.csv is None and args.folders is None) or (args.csv is not None and args.folders is not None):
        parser.error("Use either --csv <file1> [file2 ...] OR --folders <folder1> [folder2 ...]")

    return args


def main() -> None:
    args = parse_args()
    output_dir = ensure_dir(os.path.abspath(args.output_dir))
    combined_csv_path = os.path.join(output_dir, args.combined_name)

    if args.csv is not None:
        csv_paths = [os.path.abspath(path) for path in args.csv]
    else:
        csv_paths = [find_csv_in_folder(folder, csv_name=args.csv_name) for folder in args.folders]

    print("Input CSV files:")
    for p in csv_paths:
        print(f"  {p}")

    combined_df = combine_csv_files(csv_paths, combined_csv_path)
    plot_paths = make_drift_plots_from_df(combined_df, output_dir)

    print("\nCombined CSV:")
    print(f"  {combined_csv_path}")

    print("\nGenerated plot files:")
    for p in plot_paths:
        print(f"  {p}")


if __name__ == "__main__":
    main()
