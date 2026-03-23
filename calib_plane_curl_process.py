#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dual_c_hysteresis_analysis.py

Full standalone analysis / plotting script for Dual-C B-only hysteresis data.

Implements:
1) Predicted path is flipped on the x-axis in the mirrored display panel so it overlays correctly.
2) Predicted trajectory is drawn as an OPEN polyline (no first<->last closure).
3) Predicted line is thinner.
4) Legend is smaller.
5) Each analysis image is annotated only once.
6) Tip is found with a parallel-centerline technique, not by taking the terminal skeleton pixel.

Expected workflow
-----------------
- Input images are binary or near-binary masks of the catheter / strip / sample.
- Metadata (B and C) are parsed from the filename, matching names like:
      00001_tracked_block_X100.000_Y20.000_Z-155.000_B-2.345_C180.000_....png
- The script extracts a centerline from each mask, finds the tip using a
  parallel-centerline method, converts to mm, groups by C and motion direction,
  and plots measured curling / uncurling against the predicted polynomial path.

Dependencies
------------
pip install numpy matplotlib opencv-python scipy scikit-image pandas

Example
-------
python dual_c_hysteresis_analysis.py \
    --input-dir ./raw_image_data_folder \
    --calibration ./calibration.json \
    --output-dir ./analysis_out \
    --px-per-mm 12.3 \
    --origin-x-px 1920 \
    --origin-z-px 1080 \
    --c180-display-mode mirrored

Notes
-----
- Coordinate convention used in plotting:
    x_mm =  (x_px - origin_x_px) / px_per_mm
    z_mm = -(z_px - origin_z_px) / px_per_mm
  so image-down becomes +up in plotted vertical position.
- If your mask polarity is reversed, use --invert-mask.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from scipy import ndimage as ndi
from scipy.spatial import cKDTree
from skimage.morphology import remove_small_objects, skeletonize


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_MIN_COMPONENT_AREA = 300
DEFAULT_SKELETON_MIN_POINTS = 20
DEFAULT_DISTAL_WINDOW_PTS = 25
DEFAULT_DISTAL_SEARCH_EXTENSION_PX = 40.0
DEFAULT_DIRECTION_SMOOTH_WINDOW = 5
DEFAULT_ANNOTATION_RADIUS_PX = 7
DEFAULT_MEASURED_LINEWIDTH = 2.2
DEFAULT_PREDICTED_LINEWIDTH = 1.4   # thinner than measured
DEFAULT_MARKER_SIZE = 95
DEFAULT_LEGEND_FONTSIZE = 9         # smaller legend
DEFAULT_TITLE_FONTSIZE = 17
DEFAULT_SUBTITLE_FONTSIZE = 11
DEFAULT_PANEL_TITLE_FONTSIZE = 15

FILENAME_B_RE = re.compile(r"_B(-?\d+(?:\.\d+)?)")
FILENAME_C_RE = re.compile(r"_C(-?\d+(?:\.\d+)?)")


# =============================================================================
# Data classes
# =============================================================================

@dataclass
class Calibration:
    r_coeffs: np.ndarray
    z_coeffs: np.ndarray
    offplane_y_coeffs: Optional[np.ndarray]
    tip_angle_coeffs: Optional[np.ndarray]
    b_min: float
    b_max: float
    c_180_deg: float


@dataclass
class SampleResult:
    image_path: Path
    b: float
    c_deg: float
    x_tip_px: float
    z_tip_px: float
    x_tip_mm: float
    z_tip_mm: float
    path_length_px: float
    direction: str   # "curling" or "uncurling"
    block_name: str
    annotated_path: Optional[Path] = None


# =============================================================================
# Calibration loading / polynomial evaluation
# =============================================================================

def load_calibration(json_path: Path) -> Calibration:
    with open(json_path, "r") as f:
        data = json.load(f)

    cubic = data["cubic_coefficients"]
    motor_setup = data.get("motor_setup", {})

    b_range = motor_setup.get("b_motor_position_range", [-5.4, 0.0])
    b_min, b_max = float(min(b_range)), float(max(b_range))

    return Calibration(
        r_coeffs=np.asarray(cubic["r_coeffs"], dtype=float),
        z_coeffs=np.asarray(cubic["z_coeffs"], dtype=float),
        offplane_y_coeffs=(
            None
            if cubic.get("offplane_y_coeffs") is None
            else np.asarray(cubic["offplane_y_coeffs"], dtype=float)
        ),
        tip_angle_coeffs=(
            None
            if cubic.get("tip_angle_coeffs") is None
            else np.asarray(cubic["tip_angle_coeffs"], dtype=float)
        ),
        b_min=b_min,
        b_max=b_max,
        c_180_deg=float(motor_setup.get("rotation_axis_180_deg", 180.0)),
    )


def poly_eval(coeffs: np.ndarray, x: np.ndarray) -> np.ndarray:
    return np.polyval(np.asarray(coeffs, dtype=float), np.asarray(x, dtype=float))


def predict_path_from_b(
    cal: Calibration,
    b_vals: np.ndarray,
    flip_rz_sign: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Returns predicted x,z in mm for the C=0 geometry.
    """
    sign = -1.0 if flip_rz_sign else 1.0
    x_mm = sign * poly_eval(cal.r_coeffs, b_vals)
    z_mm = sign * poly_eval(cal.z_coeffs, b_vals)
    return x_mm, z_mm


# =============================================================================
# Mask / centerline / tip analysis
# =============================================================================

def load_mask_image(path: Path, invert_mask: bool = False) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise RuntimeError(f"Could not read image: {path}")

    # Robust near-binary threshold
    _, mask = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    if invert_mask:
        mask = 255 - mask
    return mask > 0


def largest_component(mask: np.ndarray, min_area: int) -> np.ndarray:
    nlab, labels, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), 8)
    if nlab <= 1:
        return mask

    best_lab = 0
    best_area = 0
    for lab in range(1, nlab):
        area = int(stats[lab, cv2.CC_STAT_AREA])
        if area > best_area:
            best_area = area
            best_lab = lab

    out = labels == best_lab
    out = remove_small_objects(out, min_size=min_area)
    return out.astype(bool)


def skeleton_to_points(skel: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(skel)
    return np.column_stack([xs.astype(float), ys.astype(float)])


def build_neighbor_graph(points: np.ndarray, radius: float = 1.45) -> Dict[int, List[int]]:
    tree = cKDTree(points)
    pairs = tree.query_pairs(r=radius)
    graph: Dict[int, List[int]] = {i: [] for i in range(len(points))}
    for i, j in pairs:
        graph[i].append(j)
        graph[j].append(i)
    return graph


def graph_endpoints(graph: Dict[int, List[int]]) -> List[int]:
    return [i for i, nbrs in graph.items() if len(nbrs) == 1]


def shortest_path(graph: Dict[int, List[int]], start: int, end: int) -> List[int]:
    from collections import deque
    q = deque([start])
    prev = {start: None}
    while q:
        u = q.popleft()
        if u == end:
            break
        for v in graph[u]:
            if v not in prev:
                prev[v] = u
                q.append(v)

    if end not in prev:
        return []

    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = prev[cur]
    path.reverse()
    return path


def longest_endpoint_path(points: np.ndarray) -> np.ndarray:
    graph = build_neighbor_graph(points)
    eps = graph_endpoints(graph)

    if len(eps) < 2:
        # fallback: sort by y then x if no good endpoints
        order = np.lexsort((points[:, 0], points[:, 1]))
        return points[order]

    best_path: List[int] = []
    best_len = -1.0

    for i in range(len(eps)):
        for j in range(i + 1, len(eps)):
            path_idx = shortest_path(graph, eps[i], eps[j])
            if len(path_idx) < 2:
                continue
            pp = points[path_idx]
            seg = np.diff(pp, axis=0)
            plen = float(np.sum(np.linalg.norm(seg, axis=1)))
            if plen > best_len:
                best_len = plen
                best_path = path_idx

    if not best_path:
        order = np.lexsort((points[:, 0], points[:, 1]))
        return points[order]

    return points[np.asarray(best_path, dtype=int)]


def orient_centerline_base_to_tip(centerline: np.ndarray) -> np.ndarray:
    """
    Heuristic orientation:
    - base is usually lower in the image (larger y), tip higher in image (smaller y)
    """
    if centerline[0, 1] > centerline[-1, 1]:
        return centerline
    return centerline[::-1].copy()


def cumulative_arclength(polyline: np.ndarray) -> np.ndarray:
    if len(polyline) < 2:
        return np.zeros(len(polyline), dtype=float)
    seg = np.linalg.norm(np.diff(polyline, axis=0), axis=1)
    return np.concatenate([[0.0], np.cumsum(seg)])


def local_tangent(points: np.ndarray) -> np.ndarray:
    if len(points) < 2:
        return np.array([1.0, 0.0], dtype=float)
    v = points[-1] - points[0]
    n = np.linalg.norm(v)
    if n < 1e-9:
        return np.array([1.0, 0.0], dtype=float)
    return v / n


def unit_normal(tangent: np.ndarray) -> np.ndarray:
    return np.array([-tangent[1], tangent[0]], dtype=float)


def march_to_boundary(
    mask: np.ndarray,
    start_xy: np.ndarray,
    direction_xy: np.ndarray,
    max_dist_px: float,
    step_px: float = 0.5,
) -> np.ndarray:
    """
    March from start point in a direction until leaving the mask.
    Returns last in-mask point.
    """
    h, w = mask.shape
    d = direction_xy / max(np.linalg.norm(direction_xy), 1e-9)

    last_inside = start_xy.copy()
    steps = int(max_dist_px / step_px)
    for k in range(1, steps + 1):
        p = start_xy + d * (k * step_px)
        x, y = int(round(p[0])), int(round(p[1]))
        if x < 0 or x >= w or y < 0 or y >= h or not mask[y, x]:
            break
        last_inside = p
    return last_inside


def find_tip_parallel_centerline(
    mask: np.ndarray,
    centerline: np.ndarray,
    distal_window_pts: int = DEFAULT_DISTAL_WINDOW_PTS,
    distal_search_extension_px: float = DEFAULT_DISTAL_SEARCH_EXTENSION_PX,
) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    Parallel-centerline tip finder.

    Steps:
    - Uses the distal part of the ordered centerline.
    - Estimates local tangent from the distal window.
    - Uses the distance transform to estimate half-width along that distal section.
    - Builds two parallel tracks offset by +/- local normal * local radius.
    - Marches both parallel tracks forward along the tangent until they leave the mask.
    - Tip = midpoint of the two foremost boundary points.

    This avoids using the raw terminal skeleton pixel as the tip.
    """
    if len(centerline) < 5:
        raise RuntimeError("Centerline too short for tip finding.")

    dist_map = ndi.distance_transform_edt(mask)
    n = len(centerline)
    k0 = max(0, n - distal_window_pts)
    distal = centerline[k0:]
    tangent = local_tangent(distal)
    normal = unit_normal(tangent)

    radii = []
    for p in distal:
        x, y = int(round(p[0])), int(round(p[1]))
        x = np.clip(x, 0, mask.shape[1] - 1)
        y = np.clip(y, 0, mask.shape[0] - 1)
        radii.append(float(dist_map[y, x]))
    radius = float(np.median(radii)) if radii else 1.0

    distal_center = distal.copy()
    left_track = distal_center + normal[None, :] * radius
    right_track = distal_center - normal[None, :] * radius

    left_seed = left_track[-1]
    right_seed = right_track[-1]

    left_tip = march_to_boundary(mask, left_seed, tangent, distal_search_extension_px)
    right_tip = march_to_boundary(mask, right_seed, tangent, distal_search_extension_px)
    tip = 0.5 * (left_tip + right_tip)

    debug = {
        "distal_center": distal_center,
        "left_track": left_track,
        "right_track": right_track,
        "left_tip": left_tip,
        "right_tip": right_tip,
        "tip": tip,
        "tangent": tangent,
        "normal": normal,
    }
    return tip, debug


def analyze_single_mask(
    image_path: Path,
    invert_mask: bool,
    min_component_area: int,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
    mask = load_mask_image(image_path, invert_mask=invert_mask)
    mask = largest_component(mask, min_component_area=min_component_area)

    skel = skeletonize(mask)
    points = skeleton_to_points(skel)
    if len(points) < DEFAULT_SKELETON_MIN_POINTS:
        raise RuntimeError(f"Skeleton too short for {image_path.name}")

    centerline = longest_endpoint_path(points)
    centerline = orient_centerline_base_to_tip(centerline)

    tip_xy, dbg = find_tip_parallel_centerline(mask, centerline)
    return mask, centerline, {**dbg, "tip": tip_xy}


# =============================================================================
# Metadata / coordinates / direction
# =============================================================================

def parse_b_c_from_filename(path: Path) -> Tuple[float, float]:
    name = path.name
    mb = FILENAME_B_RE.search(name)
    mc = FILENAME_C_RE.search(name)
    if mb is None or mc is None:
        raise RuntimeError(f"Could not parse B/C from filename: {name}")
    return float(mb.group(1)), float(mc.group(1))


def px_to_mm(
    xy_px: np.ndarray,
    px_per_mm: float,
    origin_x_px: float,
    origin_z_px: float,
) -> Tuple[float, float]:
    x_px, z_px = float(xy_px[0]), float(xy_px[1])
    x_mm = (x_px - origin_x_px) / px_per_mm
    z_mm = -(z_px - origin_z_px) / px_per_mm
    return x_mm, z_mm


def classify_direction_from_pathlength(
    samples: List[SampleResult],
    smooth_window: int = DEFAULT_DIRECTION_SMOOTH_WINDOW,
) -> None:
    """
    Classify curling / uncurling inside each C block from path-length progression.
    """
    if not samples:
        return

    s = np.array([r.path_length_px for r in samples], dtype=float)
    if len(s) >= smooth_window:
        kernel = np.ones(smooth_window, dtype=float) / smooth_window
        ss = np.convolve(s, kernel, mode="same")
    else:
        ss = s

    ds = np.gradient(ss)
    for i, r in enumerate(samples):
        r.direction = "curling" if ds[i] >= 0 else "uncurling"


# =============================================================================
# Annotation
# =============================================================================

class Annotator:
    """
    Ensures each image is annotated only once.
    """
    def __init__(self) -> None:
        self._done: set[str] = set()

    def annotate_once(
        self,
        image_path: Path,
        out_path: Path,
        mask: np.ndarray,
        centerline: np.ndarray,
        tip_xy: np.ndarray,
        debug: Dict[str, np.ndarray],
        radius_px: int = DEFAULT_ANNOTATION_RADIUS_PX,
    ) -> Optional[Path]:
        key = str(image_path.resolve())
        if key in self._done:
            return None
        self._done.add(key)

        gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            return None
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        # overlay mask contour
        cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(rgb, cnts, -1, (0, 180, 255), 1)

        # centerline
        cl = np.round(centerline).astype(np.int32)
        for i in range(len(cl) - 1):
            cv2.line(rgb, tuple(cl[i]), tuple(cl[i + 1]), (255, 255, 0), 1)

        # distal center and parallels
        for key_name, color in [
            ("distal_center", (255, 128, 0)),
            ("left_track", (0, 255, 0)),
            ("right_track", (0, 200, 255)),
        ]:
            pts = debug.get(key_name)
            if pts is None or len(pts) < 2:
                continue
            pts = np.round(pts).astype(np.int32)
            for i in range(len(pts) - 1):
                cv2.line(rgb, tuple(pts[i]), tuple(pts[i + 1]), color, 1)

        left_tip = np.round(debug["left_tip"]).astype(int)
        right_tip = np.round(debug["right_tip"]).astype(int)
        tip = np.round(tip_xy).astype(int)

        cv2.circle(rgb, tuple(left_tip), radius_px, (0, 255, 0), 2)
        cv2.circle(rgb, tuple(right_tip), radius_px, (0, 200, 255), 2)
        cv2.circle(rgb, tuple(tip), radius_px + 1, (0, 0, 255), 2)
        cv2.putText(rgb, "TIP", tuple(tip + np.array([8, -8])), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)

        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), rgb)
        return out_path


# =============================================================================
# Plotting helpers
# =============================================================================

def open_polyline_segments(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    pts = np.column_stack([x, y])
    if len(pts) < 2:
        return np.zeros((0, 2, 2), dtype=float)
    return np.stack([pts[:-1], pts[1:]], axis=1)


def plot_open_polyline(
    ax,
    x: np.ndarray,
    y: np.ndarray,
    color: str,
    linewidth: float,
    label: Optional[str] = None,
    zorder: int = 2,
    alpha: float = 1.0,
) -> None:
    """
    Draw an OPEN path only between consecutive samples.
    Never closes first->last.
    """
    segs = open_polyline_segments(np.asarray(x), np.asarray(y))
    if len(segs) == 0:
        return
    lc = LineCollection(segs, colors=color, linewidths=linewidth, alpha=alpha, zorder=zorder)
    ax.add_collection(lc)
    if label is not None:
        ax.plot([], [], color=color, lw=linewidth, label=label, alpha=alpha)


def prepare_panel_dataframe(results: List[SampleResult]) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "image_path": str(r.image_path),
                "b": r.b,
                "c_deg": r.c_deg,
                "x_tip_mm": r.x_tip_mm,
                "z_tip_mm": r.z_tip_mm,
                "direction": r.direction,
                "block_name": r.block_name,
            }
        )
    df = pd.DataFrame(rows).sort_values(["c_deg", "b"]).reset_index(drop=True)
    return df


def split_curl_uncurl(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    curling = df[df["direction"] == "curling"].copy()
    uncurling = df[df["direction"] == "uncurling"].copy()

    # keep acquisition order locally within each subset
    return curling.reset_index(drop=True), uncurling.reset_index(drop=True)


def mirrored_x(x: np.ndarray) -> np.ndarray:
    return -np.asarray(x, dtype=float)


def plot_panel(
    ax,
    df_panel: pd.DataFrame,
    cal: Calibration,
    panel_title: str,
    mirror_x_for_display: bool,
    flip_rz_sign: bool,
) -> None:
    curling, uncurling = split_curl_uncurl(df_panel)

    def tx(vals: pd.Series) -> np.ndarray:
        xx = vals.to_numpy(dtype=float)
        return mirrored_x(xx) if mirror_x_for_display else xx

    # measured
    if len(curling):
        plot_open_polyline(
            ax,
            tx(curling["x_tip_mm"]),
            curling["z_tip_mm"].to_numpy(dtype=float),
            color="#4FC3F7",
            linewidth=DEFAULT_MEASURED_LINEWIDTH,
            label="Measured curling",
            zorder=3,
        )
        ax.scatter(
            tx(curling["x_tip_mm"]),
            curling["z_tip_mm"],
            s=18,
            color="#4FC3F7",
            zorder=4,
        )

    if len(uncurling):
        plot_open_polyline(
            ax,
            tx(uncurling["x_tip_mm"]),
            uncurling["z_tip_mm"].to_numpy(dtype=float),
            color="#FF2DA2",
            linewidth=DEFAULT_MEASURED_LINEWIDTH,
            label="Measured uncurling",
            zorder=3,
        )
        ax.scatter(
            tx(uncurling["x_tip_mm"]),
            uncurling["z_tip_mm"],
            s=18,
            color="#FF2DA2",
            zorder=4,
        )

    # predicted path evaluated at the measured B values
    b_all = df_panel["b"].to_numpy(dtype=float)
    x_pred, z_pred = predict_path_from_b(cal, b_all, flip_rz_sign=flip_rz_sign)

    # FIX 1: mirrored panel also mirrors predicted x for correct overlay
    if mirror_x_for_display:
        x_pred = mirrored_x(x_pred)

    # FIX 2: open polyline only, no first->last closing
    # FIX 3: predicted line thinner
    plot_open_polyline(
        ax,
        x_pred,
        z_pred,
        color="white",
        linewidth=DEFAULT_PREDICTED_LINEWIDTH,
        label="Predicted polynomial path",
        zorder=2,
        alpha=0.95,
    )

    # start / end markers for predicted path
    if len(x_pred):
        ax.scatter([x_pred[0]], [z_pred[0]], marker="o", s=DEFAULT_MARKER_SIZE,
                   facecolors="none", edgecolors="0.85", linewidths=2, label="Start", zorder=5)
        ax.scatter([x_pred[-1]], [z_pred[-1]], marker="x", s=DEFAULT_MARKER_SIZE,
                   color="#FFE04D", linewidths=2.2, label="End", zorder=5)

    ax.set_title(panel_title, fontsize=DEFAULT_PANEL_TITLE_FONTSIZE, weight="bold", pad=10)
    ax.set_xlabel("Mirrored horizontal position (mm)", fontsize=11, weight="bold")
    ax.set_ylabel("Vertical position (mm)", fontsize=11, weight="bold")
    ax.grid(True, alpha=0.28)
    ax.set_aspect("equal", adjustable="box")

    # FIX 4: smaller legend
    leg = ax.legend(
        loc="upper right",
        fontsize=DEFAULT_LEGEND_FONTSIZE,
        framealpha=0.88,
        borderpad=0.35,
        handlelength=2.0,
    )
    for txt in leg.get_texts():
        txt.set_fontsize(DEFAULT_LEGEND_FONTSIZE)


def plot_dual_c_figure(
    df: pd.DataFrame,
    cal: Calibration,
    output_path: Path,
    flip_rz_sign: bool,
) -> None:
    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 2, figsize=(16, 8), constrained_layout=True)

    fig.suptitle(
        "Dual-C B-only hysteresis: measured curling / uncurling vs predicted polynomial path",
        fontsize=DEFAULT_TITLE_FONTSIZE,
        weight="bold",
        y=1.02,
    )
    fig.text(
        0.03,
        0.93,
        "Measured hysteresis: curling vs uncurling",
        fontsize=DEFAULT_PANEL_TITLE_FONTSIZE,
        weight="bold",
        ha="left",
    )
    fig.text(
        0.03,
        0.905,
        "C = 180° panel is mirrored about the vertical axis (x -> -x) for visual comparison with C = 0°\n"
        "Predicted curve is from calibration polynomials (x,z) evaluated at the measured B values",
        fontsize=DEFAULT_SUBTITLE_FONTSIZE,
        ha="left",
    )

    # Split by orientation
    tol = 20.0
    df0 = df[np.abs(df["c_deg"] - 0.0) <= tol].copy()
    df180 = df[np.abs(df["c_deg"] - cal.c_180_deg) <= tol].copy()

    plot_panel(
        axes[0],
        df0,
        cal,
        panel_title="C = 0°",
        mirror_x_for_display=False,
        flip_rz_sign=flip_rz_sign,
    )
    plot_panel(
        axes[1],
        df180,
        cal,
        panel_title=f"C = {int(round(cal.c_180_deg))}°  (mirrored)",
        mirror_x_for_display=True,
        flip_rz_sign=flip_rz_sign,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


# =============================================================================
# Main pipeline
# =============================================================================

def collect_image_files(input_dir: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
    return sorted([p for p in input_dir.iterdir() if p.suffix.lower() in exts])


def infer_block_name(c_deg: float, cal: Calibration) -> str:
    return "C0" if abs(c_deg - 0.0) < abs(c_deg - cal.c_180_deg) else "C180"


def analyze_folder(
    input_dir: Path,
    cal: Calibration,
    output_dir: Path,
    px_per_mm: float,
    origin_x_px: float,
    origin_z_px: float,
    invert_mask: bool,
    min_component_area: int,
    annotate_images: bool,
) -> List[SampleResult]:
    files = collect_image_files(input_dir)
    if not files:
        raise RuntimeError(f"No image files found in: {input_dir}")

    annotator = Annotator()
    results: List[SampleResult] = []

    for image_path in files:
        b, c_deg = parse_b_c_from_filename(image_path)

        mask, centerline, debug = analyze_single_mask(
            image_path=image_path,
            invert_mask=invert_mask,
            min_component_area=min_component_area,
        )

        tip_xy = debug["tip"]
        x_tip_mm, z_tip_mm = px_to_mm(
            tip_xy,
            px_per_mm=px_per_mm,
            origin_x_px=origin_x_px,
            origin_z_px=origin_z_px,
        )

        arclen = cumulative_arclength(centerline)[-1] if len(centerline) else 0.0

        annotated_path = None
        if annotate_images:
            annotated_path = output_dir / "annotated" / image_path.name
            annotated_path = annotator.annotate_once(
                image_path=image_path,
                out_path=annotated_path,
                mask=mask,
                centerline=centerline,
                tip_xy=tip_xy,
                debug=debug,
            )

        results.append(
            SampleResult(
                image_path=image_path,
                b=b,
                c_deg=c_deg,
                x_tip_px=float(tip_xy[0]),
                z_tip_px=float(tip_xy[1]),
                x_tip_mm=float(x_tip_mm),
                z_tip_mm=float(z_tip_mm),
                path_length_px=float(arclen),
                direction="curling",  # filled later
                block_name=infer_block_name(c_deg, cal),
                annotated_path=annotated_path,
            )
        )

    # Direction assignment inside each C block
    for block_name in sorted(set(r.block_name for r in results)):
        subset = [r for r in results if r.block_name == block_name]
        subset.sort(key=lambda r: r.b)
        classify_direction_from_pathlength(subset)

    return results


def save_results_csv(results: List[SampleResult], out_csv: Path) -> pd.DataFrame:
    rows = []
    for r in results:
        rows.append(
            {
                "image_path": str(r.image_path),
                "annotated_path": "" if r.annotated_path is None else str(r.annotated_path),
                "b": r.b,
                "c_deg": r.c_deg,
                "x_tip_px": r.x_tip_px,
                "z_tip_px": r.z_tip_px,
                "x_tip_mm": r.x_tip_mm,
                "z_tip_mm": r.z_tip_mm,
                "path_length_px": r.path_length_px,
                "direction": r.direction,
                "block_name": r.block_name,
            }
        )
    df = pd.DataFrame(rows).sort_values(["c_deg", "b"]).reset_index(drop=True)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    return df


# =============================================================================
# CLI
# =============================================================================

def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Dual-C hysteresis analysis with parallel-centerline tip finding.")
    ap.add_argument("--input-dir", type=Path, required=True, help="Folder containing mask images.")
    ap.add_argument("--calibration", type=Path, required=True, help="Calibration JSON.")
    ap.add_argument("--output-dir", type=Path, required=True, help="Output folder.")
    ap.add_argument("--px-per-mm", type=float, required=True, help="Image scale.")
    ap.add_argument("--origin-x-px", type=float, required=True, help="Image x-origin in pixels.")
    ap.add_argument("--origin-z-px", type=float, required=True, help="Image z-origin in pixels.")
    ap.add_argument("--invert-mask", action="store_true", help="Invert the binary mask polarity.")
    ap.add_argument("--flip-rz-sign", action="store_true", default=True,
                    help="Use same sign convention as prior analysis/calibration plots.")
    ap.add_argument("--min-component-area", type=int, default=DEFAULT_MIN_COMPONENT_AREA)
    ap.add_argument("--no-annotate-images", action="store_true", help="Skip annotated image outputs.")
    return ap


def main() -> None:
    args = build_arg_parser().parse_args()

    cal = load_calibration(args.calibration)
    output_dir: Path = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    results = analyze_folder(
        input_dir=args.input_dir,
        cal=cal,
        output_dir=output_dir,
        px_per_mm=float(args.px_per_mm),
        origin_x_px=float(args.origin_x_px),
        origin_z_px=float(args.origin_z_px),
        invert_mask=bool(args.invert_mask),
        min_component_area=int(args.min_component_area),
        annotate_images=(not bool(args.no_annotate_images)),
    )

    df = save_results_csv(results, output_dir / "tip_measurements.csv")
    plot_dual_c_figure(
        df=df,
        cal=cal,
        output_path=output_dir / "dual_c_hysteresis_plot.png",
        flip_rz_sign=bool(args.flip_rz_sign),
    )

    print(f"Saved CSV:  {output_dir / 'tip_measurements.csv'}")
    print(f"Saved plot: {output_dir / 'dual_c_hysteresis_plot.png'}")
    if not args.no_annotate_images:
        print(f"Saved annotated images in: {output_dir / 'annotated'}")


if __name__ == "__main__":
    main()