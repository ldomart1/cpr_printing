# -*- coding: utf-8 -*-
"""
Created on Tue Sep 9 13:59:28 2025

@author: skylar-scott-lab
"""

# importing libraries
import cv2
import os
import time
import pandas as pd
import pickle
import inspect
import json
from collections import deque
from pathlib import Path

try:
    from duetwebapi import DuetWebAPI
except:
    print("Error! It appears you don't have the DuetWebAPI package installed! Please use 'pip install duetwebapi==1.1.0'.")

import matplotlib.pyplot as plt
import numpy as np
from datetime import date
from skimage.morphology import skeletonize
import math
from scipy.interpolate import CubicSpline
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

def _to_json_compatible(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, dict):
        return {str(k): _to_json_compatible(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_compatible(v) for v in value]
    return value

def _neighbor_count_8(skel_u8: np.ndarray) -> np.ndarray:
    # skel_u8 is 0/1
    k = np.ones((3, 3), np.uint8)
    return cv2.filter2D(skel_u8, -1, k, borderType=cv2.BORDER_CONSTANT) - skel_u8


def _endpoints_8(skel_u8: np.ndarray) -> np.ndarray:
    nbr = _neighbor_count_8(skel_u8)
    # returns Nx2 array of (y,x)
    return np.column_stack(np.where((skel_u8 == 1) & (nbr == 1)))


def _prune_short_spurs(skel_u8: np.ndarray, min_branch_len: int = 15, max_iters: int = 200) -> np.ndarray:
    """
    Remove short endpoint branches from a skeleton.
    Works well when the tube is basically one long centerline with small spurs/noise.
    """
    sk = skel_u8.copy().astype(np.uint8)
    h, w = sk.shape
    dirs = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]

    for _ in range(max_iters):
        eps = _endpoints_8(sk)
        if len(eps) == 0:
            break

        removed_any = False
        nbr = _neighbor_count_8(sk)

        # For each endpoint, walk forward until junction/endpoint, tracking path length.
        for sy, sx in eps:
            if sk[sy, sx] == 0:
                continue

            path = [(sy, sx)]
            py, px = -1, -1
            cy, cx = int(sy), int(sx)

            # Walk until we hit a junction (deg>=3) or another endpoint, or dead end.
            while True:
                # find skeleton neighbors excluding the previous pixel
                nxt = None
                ncnt = 0
                for dy, dx in dirs:
                    ny, nx = cy + dy, cx + dx
                    if 0 <= ny < h and 0 <= nx < w and sk[ny, nx] == 1:
                        if ny == py and nx == px:
                            continue
                        ncnt += 1
                        nxt = (ny, nx)

                deg_here = int(nbr[cy, cx])  # degree in 8-neighborhood

                # stop if junction (deg>=3) or no forward neighbor
                if deg_here >= 3 or ncnt == 0:
                    break

                # advance
                py, px = cy, cx
                cy, cx = nxt
                path.append((cy, cx))

                # stop if we arrived at an endpoint (degree==1) (the other end of a tiny twig)
                if nbr[cy, cx] == 1:
                    break

                # safety
                if len(path) > min_branch_len + 50:
                    break

            # If this walked segment is short and ends at a junction, it's a spur -> delete it (except the junction pixel).
            if len(path) < min_branch_len:
                endy, endx = path[-1]
                if nbr[endy, endx] >= 3:
                    for y, x in path[:-1]:
                        sk[y, x] = 0
                    removed_any = True

        if not removed_any:
            break

    return sk


def _multisource_bfs_geodesic(skel_u8: np.ndarray, seeds_yx: np.ndarray) -> np.ndarray:
    """
    BFS geodesic distance along skeleton pixels (0/1).
    seeds_yx: Nx2 array of (y,x)
    """
    h, w = skel_u8.shape
    dist = -np.ones((h, w), dtype=np.int32)
    q = deque()

    for y, x in seeds_yx:
        y = int(y)
        x = int(x)
        if skel_u8[y, x]:
            dist[y, x] = 0
            q.append((y, x))

    dirs = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]

    while q:
        y, x = q.popleft()
        d0 = dist[y, x]
        for dy, dx in dirs:
            ny, nx = y + dy, x + dx
            if 0 <= ny < h and 0 <= nx < w and skel_u8[ny, nx] and dist[ny, nx] < 0:
                dist[ny, nx] = d0 + 1
                q.append((ny, nx))

    return dist


def _extract_main_trunk_skeleton(skel_u8: np.ndarray, seeds_yx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Keep only the dominant base-to-tip trunk of a skeleton.
    This suppresses side branches in both debug plots and distal-pose estimation.
    """
    sk = (np.asarray(skel_u8, dtype=np.uint8) > 0).astype(np.uint8)
    if sk.sum() == 0:
        return sk, -np.ones_like(sk, dtype=np.int32)

    dist = _multisource_bfs_geodesic(sk, np.asarray(seeds_yx, dtype=int).reshape(-1, 2))
    valid = (sk == 1) & (dist >= 0)
    if not np.any(valid):
        return sk, dist

    endpoints = _endpoints_8(sk)
    if endpoints.size > 0:
        ep_dist = dist[endpoints[:, 0], endpoints[:, 1]]
        valid_ep = ep_dist >= 0
        if np.any(valid_ep):
            tip_y, tip_x = endpoints[valid_ep][int(np.argmax(ep_dist[valid_ep]))]
        else:
            ys, xs = np.where(valid)
            idx = int(np.argmax(dist[ys, xs]))
            tip_y, tip_x = int(ys[idx]), int(xs[idx])
    else:
        ys, xs = np.where(valid)
        idx = int(np.argmax(dist[ys, xs]))
        tip_y, tip_x = int(ys[idx]), int(xs[idx])

    trunk = np.zeros_like(sk, dtype=np.uint8)
    nbr = _neighbor_count_8(sk)
    dirs = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]
    cy, cx = int(tip_y), int(tip_x)
    trunk[cy, cx] = 1

    max_steps = int(sk.sum()) + 5
    for _ in range(max_steps):
        d_here = int(dist[cy, cx])
        if d_here <= 0:
            break

        best = None
        best_score = None
        for dy, dx in dirs:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < sk.shape[0] and 0 <= nx < sk.shape[1] and sk[ny, nx] == 1:
                nd = int(dist[ny, nx])
                if nd < 0 or nd >= d_here:
                    continue
                score = (nd, -int(nbr[ny, nx]))
                if best_score is None or score > best_score:
                    best_score = score
                    best = (ny, nx)

        if best is None:
            break

        cy, cx = best
        trunk[cy, cx] = 1

    trunk = _prune_short_spurs(trunk, min_branch_len=3, max_iters=20)
    trunk_dist = _multisource_bfs_geodesic(trunk, np.asarray(seeds_yx, dtype=int).reshape(-1, 2))
    return trunk, trunk_dist


def _tip_angle_from_vertical_deg(skel_u8: np.ndarray, dist: np.ndarray, tip_y: int, tip_x: int, path_len: int = 12, return_path: bool = False):
    """
    Estimate local tip tangent angle with respect to vertical (degrees).
    Positive angle means the tip leans toward +x (image right).
    """
    h, w = skel_u8.shape
    dirs = [(-1, -1), (-1, 0), (-1, 1),
            (0, -1),           (0, 1),
            (1, -1),  (1, 0),  (1, 1)]

    cy, cx = int(tip_y), int(tip_x)
    path = [(cy, cx)]
    steps = max(2, int(path_len))

    for _ in range(steps - 1):
        d_here = int(dist[cy, cx])
        if d_here <= 0:
            break

        best = None
        best_dist = d_here
        for dy, dx in dirs:
            ny, nx = cy + dy, cx + dx
            if 0 <= ny < h and 0 <= nx < w and skel_u8[ny, nx] == 1:
                nd = int(dist[ny, nx])
                if nd >= 0 and nd < best_dist:
                    best_dist = nd
                    best = (ny, nx)

        if best is None:
            break

        cy, cx = best
        path.append((cy, cx))

    if len(path) < 2:
        if return_path:
            return float("nan"), path
        return float("nan")

    y0, x0 = path[0]
    y1, x1 = path[-1]
    dy = float(y1 - y0)
    dx = float(x1 - x0)

    if abs(dx) < 1e-9 and abs(dy) < 1e-9:
        if return_path:
            return float("nan"), path
        return float("nan")

    # 0 deg = vertical-up direction in image coordinates.
    angle = float(np.degrees(np.arctan2(dx, -dy)))

    if return_path:
        return angle, path
    return angle


def _pca_tip_tangent_xy(points_yx: np.ndarray, dist_vals=None, weights=None) -> np.ndarray:
    """
    PCA direction (unit) in XY (x,y) from skeleton points near the tip.

    - If weights is provided (e.g. distance-transform radius), uses weighted PCA so
      thick tube dominates thin wire.
    - If dist_vals is provided (distance-from-base), orients the direction as tip -> base.
    """
    if points_yx.shape[0] < 3:
        return np.array([np.nan, np.nan], dtype=float)

    pts_xy = points_yx[:, ::-1].astype(float)  # (x, y)

    if weights is None:
        w = np.ones((pts_xy.shape[0],), dtype=float)
    else:
        w = np.asarray(weights, dtype=float).ravel()
        if w.size != pts_xy.shape[0]:
            w = np.ones((pts_xy.shape[0],), dtype=float)
        w = np.clip(w, 0.0, np.inf)

    wsum = float(np.sum(w))
    if wsum < 1e-12:
        w = np.ones((pts_xy.shape[0],), dtype=float)
        wsum = float(np.sum(w))
    w = w / wsum

    mean_xy = np.sum(pts_xy * w[:, None], axis=0, keepdims=True)
    X = pts_xy - mean_xy

    # Weighted PCA via SVD of sqrt(w) * X
    Xw = X * np.sqrt(w)[:, None]
    _, _, vt = np.linalg.svd(Xw, full_matrices=False)
    v = vt[0].astype(float)
    nv = np.linalg.norm(v)
    if nv < 1e-9:
        return np.array([np.nan, np.nan], dtype=float)
    v /= nv

    # If cov(proj, dist) > 0, v points base -> tip; flip to tip -> base.
    if dist_vals is not None:
        dist_vals = np.asarray(dist_vals, dtype=float).ravel()
        if dist_vals.size == pts_xy.shape[0] and np.all(np.isfinite(dist_vals)):
            proj = (X @ v).ravel()
            mproj = float(np.sum(w * proj))
            mdist = float(np.sum(w * dist_vals))
            cov = float(np.sum(w * (proj - mproj) * (dist_vals - mdist)))
            if cov > 0:
                v = -v

    return v


def _tip_pose_from_distal_pca(
    skel_u8: np.ndarray,
    dist: np.ndarray,
    mask_u8: np.ndarray,
    distal_window=50,      # was 80
    roi_margin=18,
    tangent_len=40,
    radius_frac=0.30,      # was 0.55
    tip_keep_frac=0.65,    # was 0.85
    weight_power=0.5,      # was 2.0
):
    """
    Returns:
        tip_y, tip_x: distal-most point on thick mask along the dominant distal direction
        angle_deg: signed deg in [-180, 180) (0 = vertical-down, +right, -left)
        tip_path: list[(y,x)] points along tangent (debug plotting)
    """
    h, w = skel_u8.shape
    ys, xs = np.where(skel_u8 == 1)

    if ys.size == 0:
        return 0, 0, float("nan"), []

    dvals = dist[ys, xs]
    valid = dvals >= 0
    ys, xs, dvals = ys[valid], xs[valid], dvals[valid]

    if ys.size == 0:
        return 0, 0, float("nan"), []

    # Thickness map in pixel radius; thicker tube should dominate over thin wire.
    dt = cv2.distanceTransform(mask_u8.astype(np.uint8), cv2.DIST_L2, 5)
    r_skel = dt[ys, xs]

    # Robust tube-radius estimate and thick-pixel threshold.
    r_ref = float(np.percentile(r_skel[np.isfinite(r_skel)], 75)) if r_skel.size else 0.0
    r_thresh = max(0.75, radius_frac * r_ref)

    thick = r_skel >= r_thresh

    # Pick distal skeleton tip only among thick points; fall back to all skeleton points.
    if np.any(thick):
        idx_local = int(np.argmax(dvals[thick]))
        cand_ys = ys[thick]
        cand_xs = xs[thick]
        cand_d = dvals[thick]
        tip_y_s, tip_x_s = int(cand_ys[idx_local]), int(cand_xs[idx_local])
        tip_dist = float(cand_d[idx_local])
    else:
        idx_far = int(np.argmax(dvals))
        tip_y_s, tip_x_s = int(ys[idx_far]), int(xs[idx_far])
        tip_dist = float(dvals[idx_far])

    keep = dvals >= (tip_dist - int(distal_window))
    if np.any(thick):
        keep &= (r_skel >= (r_thresh * tip_keep_frac))
    if not np.any(keep):
        keep = dvals >= (tip_dist - int(distal_window))
    if not np.any(keep):
        keep = np.zeros_like(dvals, dtype=bool)
        keep[int(np.argmax(dvals))] = True
    tip_cloud_yx = np.column_stack([ys[keep], xs[keep]])
    tip_cloud_d = dvals[keep]
    tip_cloud_r = r_skel[keep]

    # PCA tangent in XY, oriented tip -> base.
    weights = np.power(np.clip(tip_cloud_r, 0.0, np.inf), weight_power)
    v_xy = _pca_tip_tangent_xy(tip_cloud_yx, dist_vals=tip_cloud_d, weights=weights)
    if not np.all(np.isfinite(v_xy)):
        return tip_y_s, tip_x_s, float("nan"), [(tip_y_s, tip_x_s)]

    vx, vy = float(v_xy[0]), float(v_xy[1])
    # v_xy is oriented tip -> base; flip to base -> tip so vertical-down is ~0 deg.
    vx, vy = -vx, -vy
    # Signed angle relative to vertical-down.
    # 0 deg = down, +90 = right, -90 = left, +/-180 = up.
    angle_deg = float(np.degrees(np.arctan2(vx, vy)))

    # Start from the coarse skeleton tip, then extend on the foreground mask
    # along the estimated distal tangent so slight skeleton truncation does not
    # stop the tip estimate early.
    tip_xy_seed = np.array([float(tip_x_s), float(tip_y_s)], dtype=np.float64)
    tip_dir_xy = np.array([vx, vy], dtype=np.float64)
    tip_dir_norm = float(np.linalg.norm(tip_dir_xy))
    if tip_dir_norm > 1e-9:
        tip_dir_xy /= tip_dir_norm
        tip_xy_in, found_inside, _ = _backtrack_point_inside_fg(
            mask_u8.astype(np.uint8),
            tip_xy_seed,
            tip_dir_xy,
            max_back_px=max(6.0, 0.5 * max(r_ref, 1.0)),
            step_px=0.5,
        )
        if found_inside:
            tip_xy_ext = ray_last_inside(
                mask_u8.astype(np.uint8),
                tip_xy_in,
                tip_dir_xy,
                step_px=0.25,
                max_len_px=max(float(roi_margin), 8.0 * max(r_ref, 1.0)),
            )
            tip_x = int(round(float(tip_xy_ext[0])))
            tip_y = int(round(float(tip_xy_ext[1])))
        else:
            tip_y, tip_x = tip_y_s, tip_x_s
    else:
        tip_y, tip_x = tip_y_s, tip_x_s

    tip_path = []
    for t in range(int(max(1, tangent_len))):
        yy = int(round(tip_y + t * vy))
        xx = int(round(tip_x + t * vx))
        if 0 <= yy < h and 0 <= xx < w:
            tip_path.append((yy, xx))

    return tip_y, tip_x, angle_deg, tip_path


def _tip_angle_to_direction_xy(tip_angle_deg):
    ang = np.deg2rad(float(tip_angle_deg))
    vx = float(np.sin(ang))
    vy = float(np.cos(ang))
    d = np.array([vx, vy], dtype=np.float64)
    n = float(np.linalg.norm(d))
    return d / max(n, 1e-12)


def _normalize_tip_angle_deg(angle_deg):
    angle_arr = np.asarray(angle_deg, dtype=float)
    normalized = np.where(np.isfinite(angle_arr) & (angle_arr < -90.0), angle_arr + 360.0, angle_arr)
    if np.ndim(normalized) == 0:
        return float(normalized)
    return normalized


def _inside_mask(mask_fg, xy):
    x = int(round(float(xy[0])))
    y = int(round(float(xy[1])))
    h, w = mask_fg.shape[:2]
    return (0 <= x < w) and (0 <= y < h) and (mask_fg[y, x] > 0)


def _backtrack_point_inside_fg(mask_fg, p0_xy, dir_xy, max_back_px=80.0, step_px=0.5):
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


def ray_last_inside(mask_fg, p0_xy, dir_xy, step_px=0.5, max_len_px=120.0):
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


def _estimate_radius_along_axis(mask_fg, dist_img, p_in_xy, dir_xy, back_len_px=60.0, step_px=0.5):
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


def _contiguous_runs_from_bool(mask_bool):
    m = np.asarray(mask_bool, dtype=bool)
    edges = np.diff(np.r_[False, m, False].astype(np.int8))
    starts = np.where(edges == 1)[0]
    ends = np.where(edges == -1)[0] - 1
    return list(zip(starts.tolist(), ends.tolist()))


def _cross_section_boundaries(mask_fg, center_xy, normal_xy, scan_half_width_px=20.0, step_px=0.5):
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
    }


def _trace_centerline_midpoints(
    mask_fg,
    start_xy,
    dir_xy,
    normal_xy,
    scan_half_width_px=20.0,
    cross_step_px=0.5,
    forward_step_px=1.0,
    max_len_px=120.0,
    min_width_px=2.0,
):
    d = np.asarray(dir_xy, dtype=np.float64)
    d /= max(np.linalg.norm(d), 1e-12)
    n = np.asarray(normal_xy, dtype=np.float64)
    n /= max(np.linalg.norm(n), 1e-12)

    cur_xy = np.asarray(start_xy, dtype=np.float64).copy()
    samples = []
    n_steps = int(max_len_px / max(forward_step_px, 1e-6))

    for _ in range(max(1, n_steps + 1)):
        res = _cross_section_boundaries(
            mask_fg,
            center_xy=cur_xy,
            normal_xy=n,
            scan_half_width_px=scan_half_width_px,
            step_px=float(cross_step_px),
        )
        if res is None:
            break

        t_left = float(res["t_left"])
        t_right = float(res["t_right"])
        if t_left > t_right:
            t_left, t_right = t_right, t_left

        width_px = float(t_right - t_left)
        if width_px < float(min_width_px):
            break

        p_left = np.asarray(res["p_left_xy"], dtype=np.float64)
        p_right = np.asarray(res["p_right_xy"], dtype=np.float64)
        center_offset_px = 0.5 * (t_left + t_right)
        mid_xy = 0.5 * (p_left + p_right)
        samples.append(
            {
                "center_xy": cur_xy.copy(),
                "mid_xy": mid_xy.copy(),
                "left_xy": p_left.copy(),
                "right_xy": p_right.copy(),
                "center_offset_px": float(center_offset_px),
                "width_px": width_px,
            }
        )
        cur_xy = mid_xy + float(forward_step_px) * d

    return samples


def refine_tip_parallel_centerline(
    grayscale,
    binary_image,
    tip_yx,
    tip_angle_deg,
    section_near_r=1.0,
    section_far_r=6.0,
    scan_half_r=3.0,
    num_sections=9,
    cross_step_px=0.5,
    ray_step_px=0.25,
    ray_max_len_r=12,
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
    r_px = _estimate_radius_along_axis(mask_fg, dist_img, p_in_xy, d_xy, back_len_px=60.0, step_px=1.0)

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

    centerline_samples = _trace_centerline_midpoints(
        mask_fg,
        start_xy=center_line_start,
        dir_xy=d_xy,
        normal_xy=n_xy,
        scan_half_width_px=scan_half_px,
        cross_step_px=float(cross_step_px),
        forward_step_px=max(0.75, float(ray_step_px)),
        max_len_px=ray_max_len_px + line_back_px,
        min_width_px=max(1.5, 0.15 * width_px),
    )

    if centerline_samples:
        ref_count = min(6, len(centerline_samples))
        center_offset_ref = float(np.median([centerline_samples[i]["center_offset_px"] for i in range(ref_count)]))
        center_offset_limit = max(1.5, 0.12 * width_px)
        stable_samples = []
        for sample in centerline_samples:
            if abs(float(sample["center_offset_px"]) - center_offset_ref) > center_offset_limit:
                break
            stable_samples.append(sample)
        if stable_samples:
            centerline_samples = stable_samples

        center_trace_xy = [s["mid_xy"].tolist() for s in centerline_samples]
        tip_center_seed_xy = np.asarray(centerline_samples[-1]["mid_xy"], dtype=np.float64)
    else:
        center_trace_xy = [center_line_start.tolist()]
        tip_center_seed_xy = center_line_start

    tip_xy = ray_last_inside(
        mask_fg,
        tip_center_seed_xy,
        d_xy,
        step_px=float(ray_step_px),
        max_len_px=max(2.0, 1.5 * r_px),
    )

    line_forward_px = float(np.linalg.norm(tip_xy - base_center_xy)) + 1.5 * r_px
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
        "centerline_trace_xy": center_trace_xy,
        "centerline_center_offset_px": [float(s["center_offset_px"]) for s in centerline_samples],
        "parallel_left_line_xy": [left_line_start.tolist(), left_line_end.tolist()],
        "parallel_right_line_xy": [right_line_start.tolist(), right_line_end.tolist()],
        "center_line_xy": [center_line_start.tolist(), tip_xy.tolist()],
        "center_line_full_xy": [center_line_start.tolist(), center_line_end.tolist()],
        "tip_xy": tip_xy.tolist(),
    }
    return float(tip_xy[1]), float(tip_xy[0]), dbg


def _point_to_polyline_distance_xy(point_xy, polyline_xy):
    p = np.asarray(point_xy, dtype=np.float64).reshape(2)
    pts = np.asarray(polyline_xy, dtype=np.float64).reshape(-1, 2)
    if pts.shape[0] == 0:
        return float("inf"), None
    if pts.shape[0] == 1:
        q = pts[0].copy()
        return float(np.linalg.norm(p - q)), q

    best_dist = float("inf")
    best_proj = None
    for a, b in zip(pts[:-1], pts[1:]):
        ab = b - a
        denom = float(np.dot(ab, ab))
        if denom <= 1e-12:
            q = a
        else:
            t = float(np.clip(np.dot(p - a, ab) / denom, 0.0, 1.0))
            q = a + t * ab
        dist = float(np.linalg.norm(p - q))
        if dist < best_dist:
            best_dist = dist
            best_proj = q.copy()

    return best_dist, best_proj


def _select_tip_candidate(coarse_tip_yx, refined_tip_yx, tip_dbg, mode="auto"):
    dbg = dict(tip_dbg) if isinstance(tip_dbg, dict) else {}

    coarse_yx = np.asarray(coarse_tip_yx, dtype=np.float64).reshape(2)
    refined_yx = np.asarray(refined_tip_yx, dtype=np.float64).reshape(2)
    coarse_xy = np.array([coarse_yx[1], coarse_yx[0]], dtype=np.float64)
    refined_xy = np.array([refined_yx[1], refined_yx[0]], dtype=np.float64)

    mode_norm = str(mode or "auto").strip().lower().replace("-", "_").replace(" ", "_")
    if mode_norm in ("skeleton", "coarse_only"):
        mode_norm = "coarse"
    elif mode_norm in ("refined", "parallel_centerline_only"):
        mode_norm = "parallel_centerline"
    elif mode_norm not in ("coarse", "parallel_centerline", "auto"):
        mode_norm = "auto"

    dbg["coarse_tip_xy"] = coarse_xy.tolist()
    dbg["refined_tip_candidate_xy"] = refined_xy.tolist()
    dbg["tip_selection_mode"] = mode_norm

    selected_xy = refined_xy.copy()
    selected_source = "parallel_centerline"
    selected_reason = "explicit_parallel_centerline_mode"

    if mode_norm == "coarse":
        selected_xy = coarse_xy.copy()
        selected_source = "coarse"
        selected_reason = "explicit_coarse_mode"
    elif mode_norm == "auto":
        fallback_reason = dbg.get("fallback")
        radius_px = max(1.0, float(dbg.get("radius_px", 3.0) or 3.0))

        centerline_xy = dbg.get("centerline_trace_xy", [])
        if len(centerline_xy) < 2:
            centerline_xy = dbg.get("center_line_xy", [])
        centerline_dist_px, projected_xy = _point_to_polyline_distance_xy(refined_xy, centerline_xy)

        delta_xy = refined_xy - coarse_xy
        d_xy = np.asarray(dbg.get("d_xy", [np.nan, np.nan]), dtype=np.float64).reshape(-1)
        n_xy = np.asarray(dbg.get("n_xy", [np.nan, np.nan]), dtype=np.float64).reshape(-1)
        along_shift_px = float("nan")
        lateral_shift_px = float("nan")

        if d_xy.size == 2 and np.all(np.isfinite(d_xy)):
            d_norm = float(np.linalg.norm(d_xy))
            if d_norm > 1e-12:
                d_xy = d_xy / d_norm
                along_shift_px = float(np.dot(delta_xy, d_xy))

        if n_xy.size == 2 and np.all(np.isfinite(n_xy)):
            n_norm = float(np.linalg.norm(n_xy))
            if n_norm > 1e-12:
                n_xy = n_xy / n_norm
                lateral_shift_px = float(abs(np.dot(delta_xy, n_xy)))

        max_centerline_dist_px = max(1.5, 0.35 * radius_px)
        max_lateral_shift_px = max(1.5, 0.50 * radius_px)
        min_along_shift_px = -max(1.0, 0.35 * radius_px)
        max_along_shift_px = max(4.0, 2.0 * radius_px)

        accept_refined = fallback_reason in (None, "", False)
        accept_refined &= np.isfinite(centerline_dist_px) and (centerline_dist_px <= max_centerline_dist_px)
        if np.isfinite(lateral_shift_px):
            accept_refined &= (lateral_shift_px <= max_lateral_shift_px)
        if np.isfinite(along_shift_px):
            accept_refined &= (min_along_shift_px <= along_shift_px <= max_along_shift_px)

        dbg["refined_tip_distance_to_centerline_px"] = (
            None if not np.isfinite(centerline_dist_px) else float(centerline_dist_px)
        )
        dbg["refined_tip_projected_xy"] = (
            None if projected_xy is None else np.asarray(projected_xy, dtype=np.float64).tolist()
        )
        dbg["refined_tip_lateral_shift_px"] = (
            None if not np.isfinite(lateral_shift_px) else float(lateral_shift_px)
        )
        dbg["refined_tip_along_shift_px"] = (
            None if not np.isfinite(along_shift_px) else float(along_shift_px)
        )
        dbg["refined_tip_acceptance_limits"] = {
            "max_centerline_dist_px": float(max_centerline_dist_px),
            "max_lateral_shift_px": float(max_lateral_shift_px),
            "min_along_shift_px": float(min_along_shift_px),
            "max_along_shift_px": float(max_along_shift_px),
        }

        if not accept_refined:
            selected_xy = coarse_xy.copy()
            selected_source = "coarse"
            if fallback_reason not in (None, "", False):
                selected_reason = f"refine_fallback:{fallback_reason}"
            else:
                selected_reason = "auto_rejected_refined_tip"
        else:
            selected_source = "parallel_centerline"
            selected_reason = "auto_accepted_refined_tip"

    dbg["tip_xy"] = selected_xy.tolist()
    dbg["selected_tip_xy"] = selected_xy.tolist()
    dbg["selected_tip_source"] = selected_source
    dbg["selected_tip_reason"] = selected_reason
    dbg["mode"] = selected_source

    return float(selected_xy[1]), float(selected_xy[0]), dbg


def _remap_zoom_axes_to_crop_coordinates(axs, zoom_x_min: int, zoom_x_max: int, zoom_y_min: int, zoom_y_max: int):
    """
    Re-express the lower zoom panels in the same cropped-image pixel coordinates
    used by the refinement geometry so overlays land on the underlying image.
    """
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


def annotate_tip_geometry_on_axes(axs, dbg, title_suffix=""):
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
        selected_tip_source = str(dbg.get("selected_tip_source", mode))
        tip_xy = np.asarray(dbg.get("tip_xy", []), dtype=float).reshape(-1)
        coarse_tip_xy = np.asarray(dbg.get("coarse_tip_xy", []), dtype=float).reshape(-1)

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
                if p.size == 2:
                    lbl = "inside anchor" if "inside anchor" not in used_labels else None
                    if lbl is not None:
                        used_labels.add(lbl)
                    ax.scatter([p[0]], [p[1]], s=40, c="#ffffff", edgecolors="#000000", label=lbl, zorder=5)

            if coarse_tip_xy.size == 2 and np.all(np.isfinite(coarse_tip_xy)):
                if tip_xy.size != 2 or not np.allclose(coarse_tip_xy, tip_xy, atol=0.25):
                    lbl = "coarse tip" if "coarse tip" not in used_labels else None
                    if lbl is not None:
                        used_labels.add(lbl)
                    ax.scatter(
                        [coarse_tip_xy[0]],
                        [coarse_tip_xy[1]],
                        s=45,
                        c="none",
                        edgecolors="#66e0ff",
                        linewidths=1.5,
                        label=lbl,
                        zorder=6,
                    )

            if tip_xy.size == 2 and np.all(np.isfinite(tip_xy)):
                lbl_txt = f"selected tip ({selected_tip_source})"
                lbl = lbl_txt if lbl_txt not in used_labels else None
                if lbl is not None:
                    used_labels.add(lbl_txt)
                ax.scatter([tip_xy[0]], [tip_xy[1]], s=55, c="#ff3b30", edgecolors="#ffffff", label=lbl, zorder=6)

            try:
                handles, labels = ax.get_legend_handles_labels()
                visible = [(h, lbl) for h, lbl in zip(handles, labels) if lbl and not lbl.startswith("_")]
                if visible:
                    ax.legend(
                        [item[0] for item in visible],
                        [item[1] for item in visible],
                        loc="upper right",
                        fontsize=6.5,
                    )
            except Exception:
                pass
    except Exception as e:
        print(f"[WARN] Failed to annotate analysis axes: {e}")


class CTR_Shadow_Calibration:
    """
    The following code is designed to acquire image data and track the tip of a curve in 2-D space using computer vision (non-AI) methods.
    A camera takes picture(s) of the curve against a uniform background (ideally a bright white screen).
    The resulting image is then analyzed to return the position of the tip in world space.
    The curve is generated by a Continuous Tube Robot (CTR), commonly used in academic investigations of collaborative 3D-printing and minimally invasive surgery.

    The class includes methods for connecting to the camera, robot, etc.
    All the data is kept in a root folder whose default label is the calibration project name (recommended to be the name of the robot you are calibrating, or some other identifier so you can link the physical machine with the code.) plus the date the calibration was run.

    The folder structure is as follows:
    - parent_directory
    - project_name_and_todays_date
        - focus_testing_folder
        - raw_image_data_folder
        - processed_image_data_folder
    """

    def __init__(self, parent_directory=os.getcwd(), project_name=None, allow_existing=False, add_date=False):
        """ Class constructor. """

        # first, change to the parent directory
        if not os.path.isdir(parent_directory):
            print("Error -- the parent directory you specified does not exist")
            return
        else:
            os.chdir(parent_directory)

        # next, create the "home folder" where everything for this lies
        if project_name is None:
            project_name = "New Calibration Project"

        if add_date:
            self.calibration_data_folder = project_name + " " + str(date.today())
        else:
            self.calibration_data_folder = project_name

        # create the calibration data folder in the parent_directory
        if not os.path.isdir(self.calibration_data_folder):
            os.mkdir(self.calibration_data_folder)
            print(f"Created new project folder: {self.calibration_data_folder}")
        else:
            if allow_existing:
                print(f"Using existing project folder: {self.calibration_data_folder}")
            else:
                print("Warning -- the project folder you specified already exists. "
                      "To avoid overwriting existing data, please use a different name "
                      "or set allow_existing=True.")
                return

        self.calibration_data_folder = os.path.join(parent_directory, self.calibration_data_folder)
        os.chdir(self.calibration_data_folder)

        # initiate all the different attributes of the class with default values
        self.parent_directory = parent_directory
        self.cam_port = None
        self.cam = None
        self.rrf = None

        # On 3840x2160 frames this maps to the GUI box:
        # x:[1005,2897] y:[330,1628]
        self.default_analysis_crop = {
            "crop_width_min": 1005,
            "crop_width_max": 2897,
            "crop_height_min": 532,
            "crop_height_max": 1830,
        }
        self.analysis_crop = dict(self.default_analysis_crop)

        # Optional camera/board calibration state (CTR shadow calibration workflow)
        self.camera_calib_path = None
        self.camera_K = None
        self.camera_dist = None
        self.camera_calib_meta = None
        self.board_pose = None
        self.true_vertical_img_unit = None
        self.board_homography_px_from_mm = None
        self.board_homography_mm_from_px = None
        self.board_px_per_mm_local = None
        self.board_mm_per_px_local = None
        self.board_planar_x_sign = 1.0
        self.board_reference_correction_meta = None
        self.board_reference_image_path = None
        self.default_charuco_config = {
            "squares_x": 10,
            "squares_y": 14,
            "square_size_mm": 15.0,
            "marker_size_mm": 11.0,
            "aruco_dictionary": "DICT_4X4",
        }
        self.ruler_ref_p1_px = None
        self.ruler_ref_p2_px = None
        self.ruler_ref_distance_mm = None
        self.ruler_mm_per_px = None
        self.ruler_px_per_mm = None
        self.ruler_axis_unit = None
        self.ruler_axis_perp_unit = None
        self.ruler_calib_meta = None
        self.tip_refine_debug_records = {}
        self.tip_refine_mode = "coarse"
        self.tip_parallel_section_near_r = 0.75
        self.tip_parallel_section_far_r = 5.0
        self.tip_parallel_scan_half_r = 2.5
        self.tip_parallel_num_sections = 7
        self.tip_parallel_cross_step_px = 0.5
        self.tip_parallel_ray_step_px = 0.5
        self.tip_parallel_ray_max_len_r = 10.0
        self.tip_refiner_model_path = None
        self.tip_refiner_model = None
        self.tip_refiner_device = None
        self.tip_refiner_patch_size = None
        self.tip_refiner_anchor_name = "selected"
        self.tip_refiner_use_as_selected = True
        self.tip_refiner_softargmax_temperature = None
        self.tip_refiner_enabled = False
        self._last_tip_locations_cnn = None

        print("Calibration object initialized successfully!")

    # Usage example:
    # sc.load_camera_calibration("/path/to/webcam_calibration.npz")
    # sc.estimate_board_reference_from_image("/path/to/checkerboard_reference.png", draw_debug=True)
    # u_mm, z_mm = sc.pixel_point_to_calibrated_axes(x_px=1800, y_px=900)
    @staticmethod
    def _normalize_board_type(board_type):
        if board_type is None:
            return "checkerboard"
        txt = str(board_type).strip().lower().replace("-", "_").replace(" ", "_")
        if "charuco" in txt:
            return "charuco"
        return "checkerboard"

    @staticmethod
    def _resolve_aruco_dictionary_id(dictionary_spec):
        if dictionary_spec is None:
            dictionary_spec = "DICT_4X4_50"

        if isinstance(dictionary_spec, (np.integer, int)):
            return int(dictionary_spec), str(int(dictionary_spec))

        txt = str(dictionary_spec).strip().upper().replace("-", "_").replace(" ", "_")
        txt = txt.replace("ARUCO_", "")
        if txt == "DICT4X4" or txt == "DICT_4X4":
            txt = "DICT_4X4_50"
        elif txt == "4X4" or txt == "4X4_50":
            txt = "DICT_4X4_50"
        elif txt.startswith("DICT4X4_"):
            txt = "DICT_" + txt[len("DICT"):]
        if not txt.startswith("DICT_"):
            txt = f"DICT_{txt}"

        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV ArUco module is unavailable. Install opencv-contrib-python.")
        if not hasattr(cv2.aruco, txt):
            raise ValueError(f"Unsupported ArUco dictionary spec: {dictionary_spec}")
        return int(getattr(cv2.aruco, txt)), txt

    def _get_aruco_dictionary(self, dictionary_spec):
        dictionary_id, dictionary_name = self._resolve_aruco_dictionary_id(dictionary_spec)
        if hasattr(cv2.aruco, "getPredefinedDictionary"):
            return cv2.aruco.getPredefinedDictionary(dictionary_id), dictionary_id, dictionary_name
        if hasattr(cv2.aruco, "Dictionary_get"):
            return cv2.aruco.Dictionary_get(dictionary_id), dictionary_id, dictionary_name
        raise RuntimeError("This OpenCV ArUco build cannot construct predefined dictionaries.")

    def _build_charuco_board(self, squares_x, squares_y, square_length_mm, marker_length_mm, dictionary_spec, use_legacy_pattern=False):
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV ArUco module is unavailable. Install opencv-contrib-python.")

        aruco_dict, dictionary_id, dictionary_name = self._get_aruco_dictionary(dictionary_spec)
        sx = int(squares_x)
        sy = int(squares_y)
        square_len = float(square_length_mm)
        marker_len = float(marker_length_mm)

        if sx < 2 or sy < 2:
            raise ValueError(f"Charuco board squares must each be >= 2, got {(sx, sy)}")
        if square_len <= 0 or marker_len <= 0:
            raise ValueError("Charuco square and marker sizes must be positive.")
        if marker_len >= square_len:
            raise ValueError("Charuco marker size must be smaller than checker square size.")

        if hasattr(cv2.aruco, "CharucoBoard"):
            try:
                board = cv2.aruco.CharucoBoard((sx, sy), square_len, marker_len, aruco_dict)
            except Exception:
                board = cv2.aruco.CharucoBoard.create(sx, sy, square_len, marker_len, aruco_dict)
        elif hasattr(cv2.aruco, "CharucoBoard_create"):
            board = cv2.aruco.CharucoBoard_create(sx, sy, square_len, marker_len, aruco_dict)
        else:
            raise RuntimeError("This OpenCV ArUco build cannot create Charuco boards.")

        if bool(use_legacy_pattern) and hasattr(board, "setLegacyPattern"):
            board.setLegacyPattern(True)

        return board, {
            "squares_x": sx,
            "squares_y": sy,
            "square_size_mm": square_len,
            "marker_size_mm": marker_len,
            "aruco_dictionary_id": dictionary_id,
            "aruco_dictionary_name": dictionary_name,
            "legacy_pattern": bool(use_legacy_pattern),
        }

    def _get_charuco_chessboard_corners(self, board):
        if hasattr(board, "getChessboardCorners"):
            corners = board.getChessboardCorners()
        elif hasattr(board, "chessboardCorners"):
            corners = board.chessboardCorners
        else:
            raise RuntimeError("Could not extract Charuco chessboard corners from board object.")
        return np.asarray(corners, dtype=np.float64).reshape(-1, 3)

    def _detect_charuco_corners(self, gray, board, dictionary_spec=None):
        if gray is None or gray.ndim != 2:
            raise ValueError("gray must be a valid grayscale image.")
        if not hasattr(cv2, "aruco"):
            raise RuntimeError("OpenCV ArUco module is unavailable. Install opencv-contrib-python.")

        aruco_dict, _, _ = self._get_aruco_dictionary(dictionary_spec)
        charuco_corners = None
        charuco_ids = None
        marker_corners = None
        marker_ids = None

        if hasattr(cv2.aruco, "DetectorParameters"):
            try:
                detector_params = cv2.aruco.DetectorParameters()
            except Exception:
                detector_params = cv2.aruco.DetectorParameters_create()
        elif hasattr(cv2.aruco, "DetectorParameters_create"):
            detector_params = cv2.aruco.DetectorParameters_create()
        else:
            detector_params = None

        if hasattr(cv2.aruco, "CharucoDetector"):
            try:
                charuco_detector = cv2.aruco.CharucoDetector(board)
                charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
            except Exception:
                charuco_corners = None
                charuco_ids = None
                marker_corners = None
                marker_ids = None

        if charuco_corners is None or charuco_ids is None or len(charuco_ids) < 4:
            if hasattr(cv2.aruco, "ArucoDetector"):
                detector = cv2.aruco.ArucoDetector(aruco_dict, detector_params) if detector_params is not None else cv2.aruco.ArucoDetector(aruco_dict)
                marker_corners, marker_ids, _ = detector.detectMarkers(gray)
            else:
                marker_corners, marker_ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=detector_params)

            if marker_ids is not None and len(marker_ids) > 0:
                if hasattr(cv2.aruco, "interpolateCornersCharuco"):
                    interp = cv2.aruco.interpolateCornersCharuco(
                        marker_corners,
                        marker_ids,
                        gray,
                        board,
                    )
                    if interp is not None and len(interp) >= 3:
                        _, charuco_corners, charuco_ids = interp[:3]

        if charuco_corners is None or charuco_ids is None:
            return False, None, None, marker_corners, marker_ids

        charuco_corners = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
        charuco_ids = np.asarray(charuco_ids, dtype=np.int32).reshape(-1, 1)
        if charuco_ids.shape[0] < 4:
            return False, None, None, marker_corners, marker_ids

        return True, charuco_corners, charuco_ids, marker_corners, marker_ids

    def load_camera_calibration(self, calibration_npz_path):
        """
        Load camera intrinsics/distortion and board metadata from a saved .npz file.

        Supported metadata:
        - Checkerboard: inner_corners, square_size_m
        - Charuco: charuco_squares_x / squares_x, charuco_squares_y / squares_y,
          marker_size_m / marker_length_m, aruco_dictionary / dictionary
        """
        if calibration_npz_path is None:
            raise ValueError("Calibration file path cannot be None.")

        calib_path = os.path.abspath(os.path.expanduser(str(calibration_npz_path)))
        if not os.path.isfile(calib_path):
            raise FileNotFoundError(f"Camera calibration file not found: {calib_path}")

        with np.load(calib_path, allow_pickle=False) as data:
            required_keys = ("K", "dist", "rms")
            missing = [k for k in required_keys if k not in data]
            if missing:
                raise KeyError(
                    f"Calibration file is missing required keys: {missing}. "
                    f"Found keys: {list(data.keys())}"
                )

            K = np.asarray(data["K"], dtype=np.float64)
            dist = np.asarray(data["dist"], dtype=np.float64)
            rms = float(np.asarray(data["rms"]).reshape(-1)[0])
            board_type = self._normalize_board_type(data["board_type"][()] if "board_type" in data else None)
            if (
                "charuco_squares_x" in data or "squares_x" in data
                or "charuco_squares_y" in data or "squares_y" in data
                or "marker_size_m" in data or "marker_length_m" in data
                or "aruco_dictionary" in data or "dictionary" in data or "dictionary_name" in data
                or "inner_corners" not in data
            ):
                board_type = "charuco"

            square_size_m = None
            square_size_mm = None
            inner_corners = None
            squares_x = None
            squares_y = None
            marker_size_m = None
            marker_size_mm = None
            aruco_dictionary = None
            charuco_legacy_pattern = False

            if "square_size_m" in data:
                square_size_m = float(np.asarray(data["square_size_m"]).reshape(-1)[0])
                square_size_mm = square_size_m * 1000.0
            elif "square_length_m" in data:
                square_size_m = float(np.asarray(data["square_length_m"]).reshape(-1)[0])
                square_size_mm = square_size_m * 1000.0
            elif board_type == "charuco":
                square_size_mm = float(self.default_charuco_config["square_size_mm"])
                square_size_m = square_size_mm / 1000.0

            if board_type == "checkerboard":
                if "inner_corners" not in data or square_size_m is None:
                    raise KeyError(
                        "Checkerboard calibration file must contain inner_corners and square_size_m."
                    )
                inner_corners_arr = np.asarray(data["inner_corners"]).reshape(-1)
                if inner_corners_arr.size < 2:
                    raise ValueError(
                        f"inner_corners must have at least 2 values, got {inner_corners_arr}"
                    )
                inner_corners = (int(inner_corners_arr[0]), int(inner_corners_arr[1]))
            else:
                if "charuco_squares_x" in data:
                    squares_x = int(np.asarray(data["charuco_squares_x"]).reshape(-1)[0])
                elif "squares_x" in data:
                    squares_x = int(np.asarray(data["squares_x"]).reshape(-1)[0])
                elif "charuco_squares_xy" in data:
                    squares_xy = np.asarray(data["charuco_squares_xy"]).reshape(-1)
                    if squares_xy.size >= 2:
                        squares_x = int(squares_xy[0])
                if "charuco_squares_y" in data:
                    squares_y = int(np.asarray(data["charuco_squares_y"]).reshape(-1)[0])
                elif "squares_y" in data:
                    squares_y = int(np.asarray(data["squares_y"]).reshape(-1)[0])
                elif "charuco_squares_xy" in data:
                    squares_xy = np.asarray(data["charuco_squares_xy"]).reshape(-1)
                    if squares_xy.size >= 2:
                        squares_y = int(squares_xy[1])
                if squares_x is None:
                    squares_x = int(self.default_charuco_config["squares_x"])
                if squares_y is None:
                    squares_y = int(self.default_charuco_config["squares_y"])

                if "marker_size_m" in data:
                    marker_size_m = float(np.asarray(data["marker_size_m"]).reshape(-1)[0])
                elif "marker_length_m" in data:
                    marker_size_m = float(np.asarray(data["marker_length_m"]).reshape(-1)[0])
                else:
                    marker_size_mm = float(self.default_charuco_config["marker_size_mm"])
                    marker_size_m = marker_size_mm / 1000.0
                if marker_size_mm is None:
                    marker_size_mm = marker_size_m * 1000.0

                if "aruco_dictionary" in data:
                    aruco_dictionary = data["aruco_dictionary"][()]
                elif "dictionary" in data:
                    aruco_dictionary = data["dictionary"][()]
                elif "dictionary_name" in data:
                    aruco_dictionary = data["dictionary_name"][()]
                else:
                    aruco_dictionary = self.default_charuco_config["aruco_dictionary"]

                if "charuco_legacy_pattern" in data:
                    charuco_legacy_pattern = bool(np.asarray(data["charuco_legacy_pattern"]).reshape(-1)[0])

        if K.shape != (3, 3):
            raise ValueError(f"K must be shape (3,3), got {K.shape}")

        self.camera_K = K
        self.camera_dist = dist
        self.camera_calib_path = calib_path
        self.camera_calib_meta = {
            "board_type": board_type,
            "rms": rms,
            "square_size_m": square_size_m,
            "square_size_mm": square_size_mm,
        }
        if inner_corners is not None:
            self.camera_calib_meta["inner_corners"] = inner_corners
        if squares_x is not None:
            self.camera_calib_meta["squares_x"] = squares_x
        if squares_y is not None:
            self.camera_calib_meta["squares_y"] = squares_y
        if marker_size_m is not None:
            self.camera_calib_meta["marker_size_m"] = marker_size_m
        if marker_size_mm is not None:
            self.camera_calib_meta["marker_size_mm"] = marker_size_mm
        if aruco_dictionary is not None:
            self.camera_calib_meta["aruco_dictionary"] = str(aruco_dictionary)
        if board_type == "charuco":
            self.camera_calib_meta["charuco_legacy_pattern"] = bool(charuco_legacy_pattern)

        print(
            "Loaded camera calibration: "
            f"RMS={rms:.6f}, board_type={board_type}, "
            + (
                f"inner_corners={inner_corners}, square_size={square_size_mm:.3f} mm"
                if board_type == "checkerboard"
                else (
                    f"squares=({squares_x},{squares_y}), square_size={square_size_mm:.3f} mm, "
                    f"marker_size={marker_size_mm:.3f} mm, dictionary={self.camera_calib_meta['aruco_dictionary']}, "
                    f"legacy_pattern={bool(self.camera_calib_meta['charuco_legacy_pattern'])}"
                )
            )
        )
        return dict(self.camera_calib_meta)

    def _detect_checkerboard_corners(self, gray, inner_corners=None, use_sb=True):
        """
        Detect checkerboard inner corners in a grayscale image.

        Returns:
            found (bool), corners (N,1,2 float32) or None.
        """
        if gray is None or gray.ndim != 2:
            raise ValueError("gray must be a valid grayscale image.")

        if inner_corners is None:
            if self.camera_calib_meta is None or "inner_corners" not in self.camera_calib_meta:
                raise ValueError(
                    "inner_corners was not provided and no camera calibration metadata is loaded."
                )
            inner_corners = self.camera_calib_meta["inner_corners"]

        pattern_size = (int(inner_corners[0]), int(inner_corners[1]))
        found = False
        corners = None

        if use_sb and hasattr(cv2, "findChessboardCornersSB"):
            try:
                flags_sb = cv2.CALIB_CB_EXHAUSTIVE + cv2.CALIB_CB_ACCURACY
                found, corners = cv2.findChessboardCornersSB(gray, pattern_size, flags=flags_sb)
            except Exception:
                found = False
                corners = None

        if not found:
            flags = (
                cv2.CALIB_CB_ADAPTIVE_THRESH
                + cv2.CALIB_CB_NORMALIZE_IMAGE
                + cv2.CALIB_CB_FAST_CHECK
            )
            found, corners = cv2.findChessboardCorners(gray, pattern_size, flags)
            if found:
                criteria = (
                    cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                    40,
                    1e-4,
                )
                corners = cv2.cornerSubPix(
                    gray,
                    corners,
                    winSize=(11, 11),
                    zeroZone=(-1, -1),
                    criteria=criteria,
                )

        if not found or corners is None:
            return False, None

        corners = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
        return True, corners

    @staticmethod
    def _scale_mm_from_px_homography(H, mm_scale):
        """
        If mm_new = mm_scale * mm_old and H_old maps px -> mm_old,
        then H_new = S * H_old with S = diag(mm_scale, mm_scale, 1).
        """
        H = np.asarray(H, dtype=np.float64).copy()
        S = np.diag([float(mm_scale), float(mm_scale), 1.0]).astype(np.float64)
        return S @ H

    @staticmethod
    def _scale_px_from_mm_homography(H, mm_scale):
        """
        If mm_new = mm_scale * mm_old and H_old maps mm_old -> px,
        then H_new = H_old * S_inv where S_inv rescales mm_new back to mm_old.
        """
        if abs(float(mm_scale)) < 1e-12:
            raise ValueError("mm_scale must be non-zero.")
        H = np.asarray(H, dtype=np.float64).copy()
        S_inv = np.diag([1.0 / float(mm_scale), 1.0 / float(mm_scale), 1.0]).astype(np.float64)
        return H @ S_inv

    def _board_reference_measurement_scale_correction(self):
        board_type = None
        if isinstance(self.board_pose, dict):
            board_type = self.board_pose.get("board_type")
            if board_type is None and "inner_corners" in self.board_pose:
                board_type = "checkerboard"
        if board_type is None and isinstance(self.camera_calib_meta, dict):
            board_type = self.camera_calib_meta.get("board_type")
        board_type = self._normalize_board_type(board_type)

        if board_type != "checkerboard":
            return 1.0

        correction_meta = self.board_reference_correction_meta
        if isinstance(correction_meta, dict):
            applied = correction_meta.get("checkerboard_mm_scale_correction")
            if applied is not None and np.isfinite(applied):
                return 1.0
        return 0.5

    def apply_checkerboard_reference_corrections(self, mm_scale=0.5, flip_planar_x=True):
        """
        Correct checkerboard-backed reference axes so they match the ruler-backed convention.

        Requested fixes:
        - checkerboard mm values are 2x too large -> scale checkerboard mm conversion by 0.5
        - checkerboard planar x sign is flipped vs motor -> negate planar x

        This is applied at the checkerboard reference layer so downstream post-processing
        uses the corrected convention on the same images.
        """
        if self.board_pose is None:
            raise RuntimeError(
                "Board reference is unavailable. Call estimate_board_reference_from_image(...) first."
            )

        mm_scale = float(mm_scale)
        flip_planar_x = bool(flip_planar_x)

        current_meta = self.board_reference_correction_meta
        if current_meta is not None:
            same_scale = abs(float(current_meta.get("checkerboard_mm_scale_correction", 1.0)) - mm_scale) < 1e-12
            same_flip = bool(current_meta.get("checkerboard_planar_x_flipped", False)) == flip_planar_x
            if same_scale and same_flip:
                return dict(current_meta)
            raise RuntimeError(
                "Checkerboard reference corrections are already applied with different parameters. "
                "Re-estimate the board reference before applying a different correction."
            )

        if abs(mm_scale) < 1e-12:
            raise ValueError("mm_scale must be non-zero.")

        if self.board_mm_per_px_local is not None:
            self.board_mm_per_px_local = float(self.board_mm_per_px_local) * mm_scale
        if self.board_px_per_mm_local is not None:
            self.board_px_per_mm_local = float(self.board_px_per_mm_local) / mm_scale

        if self.board_homography_mm_from_px is not None:
            self.board_homography_mm_from_px = self._scale_mm_from_px_homography(
                self.board_homography_mm_from_px,
                mm_scale=mm_scale,
            )
        if self.board_homography_px_from_mm is not None:
            self.board_homography_px_from_mm = self._scale_px_from_mm_homography(
                self.board_homography_px_from_mm,
                mm_scale=mm_scale,
            )

        self.board_planar_x_sign = -1.0 if flip_planar_x else 1.0
        self.board_reference_correction_meta = {
            "checkerboard_mm_scale_correction": mm_scale,
            "checkerboard_planar_x_flipped": flip_planar_x,
        }

        if isinstance(self.board_pose, dict):
            self.board_pose["board_homography_px_from_mm"] = None if self.board_homography_px_from_mm is None else self.board_homography_px_from_mm.copy()
            self.board_pose["board_homography_mm_from_px"] = None if self.board_homography_mm_from_px is None else self.board_homography_mm_from_px.copy()
            self.board_pose["board_px_per_mm_local"] = self.board_px_per_mm_local
            self.board_pose["board_mm_per_px_local"] = self.board_mm_per_px_local
            self.board_pose["board_planar_x_sign"] = self.board_planar_x_sign
            self.board_pose["corrections_applied"] = dict(self.board_reference_correction_meta)

        print(
            "Applied checkerboard reference corrections: "
            f"mm scale x{mm_scale:.3f}, planar x flipped={flip_planar_x}"
        )
        return dict(self.board_reference_correction_meta)

    def estimate_board_reference_from_image(
        self,
        image_or_path,
        inner_corners=None,
        square_size_mm=None,
        marker_size_mm=None,
        squares_x=None,
        squares_y=None,
        aruco_dictionary=None,
        use_undistort=True,
        draw_debug=False,
        save_debug_path=None,
    ):
        """
        Estimate planar board pose and calibrated board reference from a single image.

        Convention:
        - Image coordinates are (x, y) = (column, row) in pixels.
        - Board coordinates are (x_mm, y_mm) in the board plane.
        - "True vertical" in image is defined as projection of board +Y axis.
          Therefore +z_mm returned by pixel_point_to_calibrated_axes corresponds to +Y_board.
        """
        if self.camera_K is None or self.camera_dist is None:
            raise RuntimeError(
                "Camera calibration is not loaded. Call load_camera_calibration(...) first."
            )

        if isinstance(image_or_path, str):
            image_path = os.path.abspath(os.path.expanduser(image_or_path))
            image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
            if image_bgr is None:
                raise FileNotFoundError(f"Could not read board reference image: {image_path}")
            self.board_reference_image_path = image_path
        elif isinstance(image_or_path, np.ndarray):
            image_bgr = image_or_path.copy()
            self.board_reference_image_path = None
        else:
            raise TypeError("image_or_path must be an image path or a BGR numpy array.")

        board_type = "checkerboard"
        if self.camera_calib_meta is not None:
            board_type = self._normalize_board_type(self.camera_calib_meta.get("board_type"))

        if square_size_mm is None:
            if self.camera_calib_meta is None or "square_size_mm" not in self.camera_calib_meta:
                raise ValueError("square_size_mm is required when calibration metadata is unavailable.")
            square_size_mm = float(self.camera_calib_meta["square_size_mm"])
        else:
            square_size_mm = float(square_size_mm)

        if square_size_mm <= 0:
            raise ValueError(f"square_size_mm must be positive, got {square_size_mm}")

        if use_undistort:
            image_for_detection = cv2.undistort(image_bgr, self.camera_K, self.camera_dist)
            dist_for_pnp = np.zeros((1, 5), dtype=np.float64)
        else:
            image_for_detection = image_bgr
            dist_for_pnp = self.camera_dist

        gray = cv2.cvtColor(image_for_detection, cv2.COLOR_BGR2GRAY)
        board_debug_meta = {}
        corners = None
        charuco_ids = None
        marker_corners = None
        marker_ids = None

        if board_type == "charuco":
            if squares_x is None:
                if self.camera_calib_meta is None or "squares_x" not in self.camera_calib_meta:
                    raise ValueError("squares_x is required for Charuco board reference.")
                squares_x = int(self.camera_calib_meta["squares_x"])
            else:
                squares_x = int(squares_x)

            if squares_y is None:
                if self.camera_calib_meta is None or "squares_y" not in self.camera_calib_meta:
                    raise ValueError("squares_y is required for Charuco board reference.")
                squares_y = int(self.camera_calib_meta["squares_y"])
            else:
                squares_y = int(squares_y)

            if marker_size_mm is None:
                if self.camera_calib_meta is None or "marker_size_mm" not in self.camera_calib_meta:
                    raise ValueError("marker_size_mm is required for Charuco board reference.")
                marker_size_mm = float(self.camera_calib_meta["marker_size_mm"])
            else:
                marker_size_mm = float(marker_size_mm)

            if aruco_dictionary is None:
                if self.camera_calib_meta is not None and "aruco_dictionary" in self.camera_calib_meta:
                    aruco_dictionary = self.camera_calib_meta["aruco_dictionary"]
                else:
                    aruco_dictionary = "DICT_4X4_50"
            preferred_legacy_pattern = False
            if self.camera_calib_meta is not None and "charuco_legacy_pattern" in self.camera_calib_meta:
                preferred_legacy_pattern = bool(self.camera_calib_meta["charuco_legacy_pattern"])

            candidate_patterns = [preferred_legacy_pattern]
            if not preferred_legacy_pattern:
                candidate_patterns.append(True)
            else:
                candidate_patterns.append(False)

            found = False
            board_obj = None
            for use_legacy_pattern in candidate_patterns:
                board_obj, board_debug_meta = self._build_charuco_board(
                    squares_x=squares_x,
                    squares_y=squares_y,
                    square_length_mm=square_size_mm,
                    marker_length_mm=marker_size_mm,
                    dictionary_spec=aruco_dictionary,
                    use_legacy_pattern=use_legacy_pattern,
                )
                found, corners, charuco_ids, marker_corners, marker_ids = self._detect_charuco_corners(
                    gray,
                    board_obj,
                    dictionary_spec=aruco_dictionary,
                )
                if found:
                    if self.camera_calib_meta is not None:
                        self.camera_calib_meta["charuco_legacy_pattern"] = bool(use_legacy_pattern)
                    break
            if not found:
                raise RuntimeError(
                    "Charuco board not found in reference image "
                    f"using squares=({squares_x},{squares_y}), square_size_mm={square_size_mm}, "
                    f"marker_size_mm={marker_size_mm}, dictionary={board_debug_meta['aruco_dictionary_name']}."
                )

            all_obj_points = self._get_charuco_chessboard_corners(board_obj)
            ids_flat = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
            obj_points = all_obj_points[ids_flat]
            img_points = corners.reshape(-1, 2).astype(np.float64)
            board_xy = obj_points[:, :2].astype(np.float64)
        else:
            if inner_corners is None:
                if self.camera_calib_meta is None or "inner_corners" not in self.camera_calib_meta:
                    raise ValueError("inner_corners is required when calibration metadata is unavailable.")
                inner_corners = self.camera_calib_meta["inner_corners"]
            inner_corners = (int(inner_corners[0]), int(inner_corners[1]))

            found, corners = self._detect_checkerboard_corners(gray, inner_corners=inner_corners, use_sb=True)
            if not found:
                raise RuntimeError(
                    f"Checkerboard not found in reference image using inner_corners={inner_corners}."
                )

            nx, ny = int(inner_corners[0]), int(inner_corners[1])
            board_xy = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2).astype(np.float64) * square_size_mm
            obj_points = np.zeros((board_xy.shape[0], 3), dtype=np.float64)
            obj_points[:, :2] = board_xy
            img_points = corners.reshape(-1, 2).astype(np.float64)

        pnp_ok, rvec, tvec = cv2.solvePnP(
            obj_points,
            img_points,
            self.camera_K,
            dist_for_pnp,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not pnp_ok:
            pnp_ok, rvec, tvec, _ = cv2.solvePnPRansac(
                obj_points,
                img_points,
                self.camera_K,
                dist_for_pnp,
            )
        if not pnp_ok:
            raise RuntimeError("Failed to estimate checkerboard pose (solvePnP failed).")

        R, _ = cv2.Rodrigues(rvec)
        board_normal_cam = R[:, 2].copy()

        axis_len = float(square_size_mm * 3.0)
        axis_points_3d = np.array(
            [
                [0.0, 0.0, 0.0],
                [axis_len, 0.0, 0.0],
                [0.0, axis_len, 0.0],
            ],
            dtype=np.float64,
        )
        proj_axes, _ = cv2.projectPoints(
            axis_points_3d,
            rvec,
            tvec,
            self.camera_K,
            dist_for_pnp,
        )
        proj_axes = proj_axes.reshape(-1, 2)
        origin_px = proj_axes[0].astype(np.float64)
        x_axis_px = proj_axes[1].astype(np.float64)
        y_axis_px = proj_axes[2].astype(np.float64)

        vertical_vec = y_axis_px - origin_px
        v_norm = float(np.linalg.norm(vertical_vec))
        if v_norm < 1e-9:
            raise RuntimeError("Projected board +Y axis is degenerate; cannot define true vertical.")
        true_vertical_img_unit = vertical_vec / v_norm

        H_px_from_mm, _ = cv2.findHomography(board_xy, img_points, method=0)
        if H_px_from_mm is None:
            raise RuntimeError("Failed to compute board-plane homography (mm -> px).")
        H_mm_from_px = np.linalg.inv(H_px_from_mm)

        d_all = []
        if board_type == "charuco":
            ids_flat = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
            id_to_img = {
                int(cid): img_points[idx]
                for idx, cid in enumerate(ids_flat)
            }
            id_to_xy = {
                int(cid): board_xy[idx]
                for idx, cid in enumerate(ids_flat)
            }
            tol = float(square_size_mm) * 0.25
            for ida, pxa in id_to_img.items():
                xya = id_to_xy[ida]
                for idb, pxb in id_to_img.items():
                    if idb <= ida:
                        continue
                    dxy = id_to_xy[idb] - xya
                    same_row = abs(float(dxy[1])) <= tol and abs(abs(float(dxy[0])) - square_size_mm) <= tol
                    same_col = abs(float(dxy[0])) <= tol and abs(abs(float(dxy[1])) - square_size_mm) <= tol
                    if same_row or same_col:
                        d_all.append(float(np.linalg.norm(pxb - pxa)))
            d_all = np.asarray(d_all, dtype=float)
        else:
            corners_grid = img_points.reshape(ny, nx, 2)
            d_h = np.linalg.norm(corners_grid[:, 1:, :] - corners_grid[:, :-1, :], axis=2).reshape(-1)
            d_v = np.linalg.norm(corners_grid[1:, :, :] - corners_grid[:-1, :, :], axis=2).reshape(-1)
            d_all = np.concatenate([d_h, d_v])
        d_all = np.asarray(d_all, dtype=float)
        d_all = d_all[np.isfinite(d_all) & (d_all > 0)]
        if d_all.size == 0:
            raise RuntimeError("Could not compute local board px/mm scale from detected corners.")

        board_px_per_mm_local = float(np.mean(d_all) / square_size_mm)
        board_mm_per_px_local = 1.0 / board_px_per_mm_local

        self.true_vertical_img_unit = true_vertical_img_unit.astype(np.float64)
        self.board_homography_px_from_mm = H_px_from_mm.astype(np.float64)
        self.board_homography_mm_from_px = H_mm_from_px.astype(np.float64)
        self.board_px_per_mm_local = float(board_px_per_mm_local)
        self.board_mm_per_px_local = float(board_mm_per_px_local)
        self.board_planar_x_sign = 1.0
        self.board_reference_correction_meta = None
        self.board_pose = {
            "board_type": board_type,
            "rvec": np.asarray(rvec, dtype=np.float64).reshape(3, 1),
            "tvec": np.asarray(tvec, dtype=np.float64).reshape(3, 1),
            "R": np.asarray(R, dtype=np.float64),
            "normal_cam": np.asarray(board_normal_cam, dtype=np.float64).reshape(3),
            "origin_px": origin_px.astype(np.float64),
            "x_axis_px": x_axis_px.astype(np.float64),
            "y_axis_px": y_axis_px.astype(np.float64),
            "image_shape": tuple(image_for_detection.shape),
            "square_size_mm": float(square_size_mm),
            "use_undistort": bool(use_undistort),
            "board_homography_px_from_mm": self.board_homography_px_from_mm.copy(),
            "board_homography_mm_from_px": self.board_homography_mm_from_px.copy(),
            "board_px_per_mm_local": self.board_px_per_mm_local,
            "board_mm_per_px_local": self.board_mm_per_px_local,
            "board_planar_x_sign": self.board_planar_x_sign,
        }
        if board_type == "checkerboard":
            self.board_pose["inner_corners"] = inner_corners
        else:
            self.board_pose.update({
                "squares_x": int(squares_x),
                "squares_y": int(squares_y),
                "marker_size_mm": float(marker_size_mm),
                "aruco_dictionary": board_debug_meta["aruco_dictionary_name"],
                "charuco_legacy_pattern": bool(board_debug_meta.get("legacy_pattern", False)),
            })

        if board_type == "checkerboard":
            self.apply_checkerboard_reference_corrections(
                mm_scale=self._board_reference_measurement_scale_correction(),
                flip_planar_x=True,
            )

        result = {
            "camera_calib_path": self.camera_calib_path,
            "board_reference_image_path": self.board_reference_image_path,
            "board_type": board_type,
            "rvec": self.board_pose["rvec"],
            "tvec": self.board_pose["tvec"],
            "R": self.board_pose["R"],
            "normal_cam": self.board_pose["normal_cam"],
            "origin_px": self.board_pose["origin_px"],
            "x_axis_px": self.board_pose["x_axis_px"],
            "y_axis_px": self.board_pose["y_axis_px"],
            "true_vertical_img_unit": self.true_vertical_img_unit,
            "board_homography_px_from_mm": self.board_homography_px_from_mm,
            "board_homography_mm_from_px": self.board_homography_mm_from_px,
            "board_px_per_mm_local": self.board_px_per_mm_local,
            "board_mm_per_px_local": self.board_mm_per_px_local,
            "square_size_mm": float(square_size_mm),
            "image_shape": tuple(image_for_detection.shape),
            "board_planar_x_sign": self.board_planar_x_sign,
            "board_reference_correction_meta": None if self.board_reference_correction_meta is None else dict(self.board_reference_correction_meta),
            "undistorted_image": image_for_detection.copy(),
        }
        if board_type == "checkerboard":
            result["inner_corners"] = inner_corners
        else:
            result.update({
                "squares_x": int(squares_x),
                "squares_y": int(squares_y),
                "marker_size_mm": float(marker_size_mm),
                "aruco_dictionary": board_debug_meta["aruco_dictionary_name"],
                "charuco_legacy_pattern": bool(board_debug_meta.get("legacy_pattern", False)),
            })

        if draw_debug:
            debug_img = image_for_detection.copy()
            if board_type == "checkerboard":
                cv2.drawChessboardCorners(debug_img, (nx, ny), corners, True)
            else:
                if marker_corners is not None and marker_ids is not None and len(marker_ids) > 0:
                    cv2.aruco.drawDetectedMarkers(debug_img, marker_corners, marker_ids)
                if corners is not None and charuco_ids is not None and len(charuco_ids) > 0 and hasattr(cv2.aruco, "drawDetectedCornersCharuco"):
                    cv2.aruco.drawDetectedCornersCharuco(debug_img, corners, charuco_ids, (255, 0, 255))

            o = tuple(np.round(origin_px).astype(int))
            x_end = tuple(np.round(x_axis_px).astype(int))
            y_end = tuple(np.round(y_axis_px).astype(int))
            cv2.circle(debug_img, o, 7, (255, 255, 0), -1)
            cv2.arrowedLine(debug_img, o, x_end, (0, 0, 255), 3, tipLength=0.15)  # X axis
            cv2.arrowedLine(debug_img, o, y_end, (0, 255, 0), 3, tipLength=0.15)  # Y axis / true vertical

            txt1 = f"px/mm(local): {self.board_px_per_mm_local:.4f}"
            txt2 = f"mm/px(local): {self.board_mm_per_px_local:.6f}"
            cv2.putText(debug_img, txt1, (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 255, 40), 2)
            cv2.putText(debug_img, txt2, (30, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (40, 255, 40), 2)

            if save_debug_path is not None:
                save_debug_path = os.path.abspath(os.path.expanduser(str(save_debug_path)))
                cv2.imwrite(save_debug_path, debug_img)
                print(f"Saved board reference debug image: {save_debug_path}")
                debug_root, debug_ext = os.path.splitext(save_debug_path)
                undistorted_debug_path = f"{debug_root}_undistorted{debug_ext if debug_ext else '.png'}"
                cv2.imwrite(undistorted_debug_path, image_for_detection)
                print(f"Saved board reference undistorted image: {undistorted_debug_path}")

            result["debug_image"] = debug_img

        print(
            "Estimated board reference: "
            + (
                f"checkerboard inner_corners={inner_corners}, square={square_size_mm:.3f} mm, "
                if board_type == "checkerboard"
                else (
                    f"charuco squares=({int(squares_x)},{int(squares_y)}), "
                    f"square={square_size_mm:.3f} mm, marker={float(marker_size_mm):.3f} mm, "
                    f"dictionary={board_debug_meta['aruco_dictionary_name']}, "
                )
            )
            + f"local scale={self.board_px_per_mm_local:.4f} px/mm"
        )
        print("True vertical convention: +Z image-axis is projection of board +Y.")
        return result

    def board_mm_to_px(self, x_mm, y_mm):
        """Map board-plane mm coordinates -> image px using homography."""
        if self.board_homography_px_from_mm is None:
            raise RuntimeError(
                "Board homography is unavailable. Call estimate_board_reference_from_image(...) first."
            )

        pts_mm = np.array([[[float(x_mm), float(y_mm)]]], dtype=np.float64)
        pts_px = cv2.perspectiveTransform(pts_mm, self.board_homography_px_from_mm)
        return pts_px.reshape(2)

    def board_px_to_mm(self, x_px, y_px):
        """Map image px coordinates -> board-plane mm using inverse homography."""
        if self.board_homography_mm_from_px is None:
            raise RuntimeError(
                "Board inverse homography is unavailable. Call estimate_board_reference_from_image(...) first."
            )

        pts_px = np.array([[[float(x_px), float(y_px)]]], dtype=np.float64)
        pts_mm = cv2.perspectiveTransform(pts_px, self.board_homography_mm_from_px)
        return pts_mm.reshape(2)

    def mm_to_px_local(self, mm_val):
        """Convert millimeters to pixels using local checkerboard scale near board center."""
        if self.board_px_per_mm_local is None:
            raise RuntimeError("Local px/mm scale unavailable. Estimate board reference first.")
        return float(mm_val) * float(self.board_px_per_mm_local)

    def px_to_mm_local(self, px_val):
        """Convert pixels to millimeters using local checkerboard scale near board center."""
        if self.board_mm_per_px_local is None:
            raise RuntimeError("Local mm/px scale unavailable. Estimate board reference first.")
        return float(px_val) * float(self.board_mm_per_px_local)

    def mm_to_px_ruler(self, mm_val):
        """Convert millimeters to pixels using manual ruler-reference scale."""
        if self.ruler_px_per_mm is None:
            raise RuntimeError("Ruler calibration unavailable. Run setup_analysis_crop(enable_manual_adjustment=True) and complete ruler picking.")
        return float(mm_val) * float(self.ruler_px_per_mm)

    def px_to_mm_ruler(self, px_val):
        """Convert pixels to millimeters using manual ruler-reference scale."""
        if self.ruler_mm_per_px is None:
            raise RuntimeError("Ruler calibration unavailable. Run setup_analysis_crop(enable_manual_adjustment=True) and complete ruler picking.")
        return float(px_val) * float(self.ruler_mm_per_px)

    def project_pixel_delta_onto_ruler_axis(self, dx_px, dy_px):
        """Project a pixel delta onto the ruler axis unit vector."""
        if self.ruler_axis_unit is None:
            raise RuntimeError("Ruler axis unavailable. Run setup_analysis_crop(enable_manual_adjustment=True) and complete ruler picking.")
        delta = np.array([float(dx_px), float(dy_px)], dtype=np.float64)
        return float(np.dot(delta, self.ruler_axis_unit))

    def project_pixel_delta_onto_ruler_perp(self, dx_px, dy_px):
        """Project a pixel delta onto the ruler-axis perpendicular unit vector."""
        if self.ruler_axis_perp_unit is None:
            raise RuntimeError("Ruler perpendicular axis unavailable. Run setup_analysis_crop(enable_manual_adjustment=True) and complete ruler picking.")
        delta = np.array([float(dx_px), float(dy_px)], dtype=np.float64)
        return float(np.dot(delta, self.ruler_axis_perp_unit))

    def get_analysis_reference_info(self):
        """Return analysis crop + optional ruler-reference state."""
        return {
            "analysis_crop": dict(self.analysis_crop) if self.analysis_crop is not None else None,
            "ruler_ref_p1_px": None if self.ruler_ref_p1_px is None else tuple(self.ruler_ref_p1_px),
            "ruler_ref_p2_px": None if self.ruler_ref_p2_px is None else tuple(self.ruler_ref_p2_px),
            "ruler_ref_distance_mm": self.ruler_ref_distance_mm,
            "ruler_mm_per_px": self.ruler_mm_per_px,
            "ruler_px_per_mm": self.ruler_px_per_mm,
            "ruler_axis_unit": None if self.ruler_axis_unit is None else np.asarray(self.ruler_axis_unit, dtype=float).copy(),
            "ruler_axis_perp_unit": None if self.ruler_axis_perp_unit is None else np.asarray(self.ruler_axis_perp_unit, dtype=float).copy(),
            "ruler_calib_meta": None if self.ruler_calib_meta is None else dict(self.ruler_calib_meta),
            "board_reference_correction_meta": None if self.board_reference_correction_meta is None else dict(self.board_reference_correction_meta),
        }

    @staticmethod
    def _safe_output_stem(path_text):
        stem = str(path_text).strip().replace("\\", "/")
        root, ext = os.path.splitext(stem)
        if ext.lower() in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
            stem = root
        out = []
        for ch in stem:
            if ch.isalnum() or ch in ("_", "-", "."):
                out.append(ch)
            else:
                out.append("_")
        return "".join(out).strip("._") or "image"

    def load_tip_refiner_model(self, model_path, anchor_name=None, use_as_selected=True, device=None):
        """Load the optional CNN tip refiner trained by cnn/train_tip_refiner.py."""
        try:
            import torch
            import torch.nn as nn
            import torch.nn.functional as F
        except Exception as exc:
            raise RuntimeError(
                "PyTorch is required to use --tip_refiner_model. Install torch in this environment."
            ) from exc

        class ConvBlock(nn.Module):
            def __init__(self, in_ch, out_ch):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_ch, out_ch, 3, padding=1),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(inplace=True),
                )

            def forward(self, x):
                return self.net(x)

        class TinyUNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.enc1 = ConvBlock(1, 32)
                self.enc2 = ConvBlock(32, 64)
                self.enc3 = ConvBlock(64, 128)
                self.pool = nn.MaxPool2d(2)
                self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
                self.dec2 = ConvBlock(128, 64)
                self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
                self.dec1 = ConvBlock(64, 32)
                self.head = nn.Conv2d(32, 1, kernel_size=1)

            def forward(self, x):
                x1 = self.enc1(x)
                x2 = self.enc2(self.pool(x1))
                x3 = self.enc3(self.pool(x2))
                y = self.up2(x3)
                if y.shape[-2:] != x2.shape[-2:]:
                    y = F.interpolate(y, size=x2.shape[-2:], mode="bilinear", align_corners=False)
                y = self.dec2(torch.cat([y, x2], dim=1))
                y = self.up1(y)
                if y.shape[-2:] != x1.shape[-2:]:
                    y = F.interpolate(y, size=x1.shape[-2:], mode="bilinear", align_corners=False)
                y = self.dec1(torch.cat([y, x1], dim=1))
                return self.head(y)

        model_path = str(Path(model_path).expanduser().resolve())
        dev = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        ckpt = torch.load(model_path, map_location=dev)
        cfg = ckpt.get("config", {})
        model = TinyUNet().to(dev)
        model.load_state_dict(ckpt["model_state_dict"])
        model.eval()

        ckpt_anchor = str(ckpt.get("anchor_name", "selected")).strip().lower()
        chosen_anchor = str(anchor_name if anchor_name is not None else ckpt_anchor).strip().lower()
        if chosen_anchor not in {"coarse", "selected", "refined"}:
            raise ValueError("--tip_refiner_anchor must be one of: coarse, selected, refined")

        self.tip_refiner_model_path = model_path
        self.tip_refiner_model = model
        self.tip_refiner_device = dev
        self.tip_refiner_patch_size = int(ckpt.get("patch_size", 128))
        self.tip_refiner_anchor_name = chosen_anchor
        self.tip_refiner_use_as_selected = bool(use_as_selected)
        self.tip_refiner_softargmax_temperature = (
            float(cfg["softargmax_temperature"])
            if isinstance(cfg, dict) and "softargmax_temperature" in cfg
            else None
        )
        self.tip_refiner_enabled = True
        print(
            "Loaded CNN tip refiner: "
            f"{model_path} | patch_size={self.tip_refiner_patch_size} "
            f"anchor={self.tip_refiner_anchor_name} device={dev}"
        )

    @staticmethod
    def _extract_tip_refiner_patch(gray_image, center_x, center_y, patch_size):
        patch_size = int(patch_size)
        half = patch_size // 2
        x0 = int(round(float(center_x))) - half
        y0 = int(round(float(center_y))) - half
        x1 = x0 + patch_size
        y1 = y0 + patch_size

        h, w = gray_image.shape[:2]
        src_x0 = max(0, x0)
        src_y0 = max(0, y0)
        src_x1 = min(w, x1)
        src_y1 = min(h, y1)

        dst_x0 = src_x0 - x0
        dst_y0 = src_y0 - y0
        dst_x1 = dst_x0 + (src_x1 - src_x0)
        dst_y1 = dst_y0 + (src_y1 - src_y0)

        patch = np.full((patch_size, patch_size), 255, dtype=np.uint8)
        patch[dst_y0:dst_y1, dst_x0:dst_x1] = gray_image[src_y0:src_y1, src_x0:src_x1]
        return patch, x0, y0

    def predict_tip_refiner_abs(self, gray_image, anchor_x_abs, anchor_y_abs):
        if not self.tip_refiner_enabled or self.tip_refiner_model is None:
            return None
        import torch

        patch, x0, y0 = self._extract_tip_refiner_patch(
            gray_image=gray_image,
            center_x=anchor_x_abs,
            center_y=anchor_y_abs,
            patch_size=self.tip_refiner_patch_size,
        )
        tensor = torch.from_numpy((patch.astype(np.float32) / 255.0)[None, None, :, :]).to(self.tip_refiner_device)
        with torch.no_grad():
            logits = self.tip_refiner_model(tensor)
            _, _, h, w = logits.shape
            if self.tip_refiner_softargmax_temperature is not None:
                probs = torch.softmax(logits.view(1, -1) * float(self.tip_refiner_softargmax_temperature), dim=1)
                ys = torch.arange(h, device=logits.device, dtype=logits.dtype).repeat_interleave(w)
                xs = torch.arange(w, device=logits.device, dtype=logits.dtype).repeat(h)
                x_patch = float(torch.sum(probs * xs[None, :], dim=1).detach().cpu().numpy()[0])
                y_patch = float(torch.sum(probs * ys[None, :], dim=1).detach().cpu().numpy()[0])
            else:
                probs = torch.sigmoid(logits)
                idx = torch.argmax(probs.view(1, -1), dim=1)
                y_patch = float((idx // w).detach().cpu().numpy()[0])
                x_patch = float((idx % w).detach().cpu().numpy()[0])
        return {
            "x_abs": float(x0 + x_patch),
            "y_abs": float(y0 + y_patch),
            "x_patch": x_patch,
            "y_patch": y_patch,
            "patch_x0": int(x0),
            "patch_y0": int(y0),
        }

    def project_pixel_delta_onto_true_vertical(self, dx_px, dy_px):
        """
        Return signed component in pixels along estimated true vertical image direction.
        """
        if self.true_vertical_img_unit is None:
            raise RuntimeError(
                "True vertical image direction is unavailable. Estimate board reference first."
            )
        delta = np.array([float(dx_px), float(dy_px)], dtype=np.float64)
        return float(np.dot(delta, self.true_vertical_img_unit))

    def project_points_onto_true_vertical(self, pts_xy_px, origin_xy_px=None):
        """
        Project points onto true vertical image direction.

        Returns signed pixel distances along true vertical from origin.
        """
        if self.true_vertical_img_unit is None:
            raise RuntimeError(
                "True vertical image direction is unavailable. Estimate board reference first."
            )

        pts = np.asarray(pts_xy_px, dtype=np.float64).reshape(-1, 2)
        if origin_xy_px is None:
            if self.board_pose is not None and "origin_px" in self.board_pose:
                origin = np.asarray(self.board_pose["origin_px"], dtype=np.float64).reshape(1, 2)
            else:
                origin = np.zeros((1, 2), dtype=np.float64)
        else:
            origin = np.asarray(origin_xy_px, dtype=np.float64).reshape(1, 2)

        deltas = pts - origin
        return (deltas @ self.true_vertical_img_unit.reshape(2, 1)).reshape(-1)

    def pixel_point_to_calibrated_axes(self, x_px, y_px, origin_px=None):
        """
        Convert a pixel point to board-referenced calibrated axes (u_mm, z_mm).

        - u_mm: transverse coordinate in board-plane X direction
        - z_mm: coordinate along "true vertical", defined as checkerboard +Y direction

        Uses full homography when available. Falls back to local projection + local scale.
        """
        pt_px = np.array([float(x_px), float(y_px)], dtype=np.float64)

        if origin_px is None:
            if self.board_pose is not None and "origin_px" in self.board_pose:
                origin_px_vec = np.asarray(self.board_pose["origin_px"], dtype=np.float64).reshape(2)
            else:
                origin_px_vec = np.array([0.0, 0.0], dtype=np.float64)
        else:
            origin_px_vec = np.asarray(origin_px, dtype=np.float64).reshape(2)

        scale_fix = float(self._board_reference_measurement_scale_correction())

        if self.board_homography_mm_from_px is not None:
            pt_mm = self.board_px_to_mm(pt_px[0], pt_px[1])
            origin_mm = self.board_px_to_mm(origin_px_vec[0], origin_px_vec[1])
            delta_mm = pt_mm - origin_mm
            u_mm = float(delta_mm[0]) * float(self.board_planar_x_sign) * scale_fix
            z_mm = float(delta_mm[1]) * scale_fix
            return u_mm, z_mm

        if self.true_vertical_img_unit is None or self.board_mm_per_px_local is None:
            raise RuntimeError(
                "Calibrated conversion unavailable: need board homography or (true_vertical + local mm/px)."
            )

        delta_px = pt_px - origin_px_vec
        v_hat = self.true_vertical_img_unit.astype(np.float64).reshape(2)
        u_hat = np.array([v_hat[1], -v_hat[0]], dtype=np.float64)

        u_mm = float(np.dot(delta_px, u_hat) * self.board_mm_per_px_local * float(self.board_planar_x_sign) * scale_fix)
        z_mm = float(np.dot(delta_px, v_hat) * self.board_mm_per_px_local * scale_fix)
        return u_mm, z_mm

    def setup_analysis_crop(self, enable_manual_adjustment=False, cam_port=None):
        """
        Configure analysis crop bounds.

        If enable_manual_adjustment=True, opens an interactive window where the user can drag rectangle corners.
        The rectangle starts from default crop values and is stored in self.analysis_crop when confirmed.
        """
        if not enable_manual_adjustment:
            self.analysis_crop = dict(self.default_analysis_crop)
            print(f"Using default analysis crop: {self.analysis_crop}")
            return dict(self.analysis_crop)

        if cam_port is None:
            cam_port = self.cam_port if self.cam_port is not None else 0

        cam_opened_here = False
        cam = self.cam
        if cam is None:
            cam = cv2.VideoCapture(cam_port)
            cam_opened_here = True

        if not cam.isOpened():
            print("Could not open camera for manual crop setup. Falling back to default crop.")
            self.analysis_crop = dict(self.default_analysis_crop)
            return dict(self.analysis_crop)

        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

        # Flush a couple of frames to stabilize exposure.
        _ = cam.read()
        ret, frame = cam.read()
        if not ret or frame is None:
            print("Could not capture frame for manual crop setup. Falling back to default crop.")
            if cam_opened_here:
                cam.release()
            self.analysis_crop = dict(self.default_analysis_crop)
            return dict(self.analysis_crop)

        img_h, img_w = frame.shape[:2]
        defaults = dict(self.default_analysis_crop)
        x_min = int(np.clip(defaults["crop_width_min"], 0, img_w - 1))
        x_max = int(np.clip(defaults["crop_width_max"], x_min + 1, img_w))
        y_min = int(np.clip(img_h - defaults["crop_height_max"], 0, img_h - 1))
        y_max = int(np.clip(img_h - defaults["crop_height_min"], y_min + 1, img_h))

        corners = {
            "tl": [x_min, y_min],
            "tr": [x_max, y_min],
            "br": [x_max, y_max],
            "bl": [x_min, y_max],
        }

        active_corner = {"name": None}
        drag_threshold_px = 70
        window_name = "Manual Crop Setup"

        def nearest_corner(mx, my):
            best_name = None
            best_dist = 1e9
            for name, (cx, cy) in corners.items():
                d = (mx - cx) ** 2 + (my - cy) ** 2
                if d < best_dist:
                    best_dist = d
                    best_name = name
            if best_dist <= drag_threshold_px ** 2:
                return best_name
            return None

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

        def on_mouse(event, mx, my, flags, param):
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

        accepted = False
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(window_name, on_mouse)

        print("Manual crop setup:")
        print("- Drag rectangle corners with left mouse.")
        print("- Press ENTER or SPACE to confirm.")
        print("- Press R to reset to defaults.")
        print("- Press Q or ESC to cancel and keep defaults.")

        try:
            while True:
                # Allow closing via window manager "X" button.
                if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
                    break

                display = frame.copy()
                clamp_rect()
                x0, y0 = corners["tl"]
                x1, y1 = corners["br"]

                cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 0), 2)
                for pt in corners.values():
                    cv2.circle(display, tuple(pt), 10, (0, 100, 255), -1)

                cv2.putText(
                    display,
                    f"x:[{x0},{x1}] y:[{y0},{y1}]",
                    (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                cv2.imshow(window_name, display)
                key = cv2.waitKey(20) & 0xFF

                if key in (13, 32):  # Enter or Space
                    accepted = True
                    break
                if key in (27, ord('q')):  # Esc or q
                    break
                if key in (ord('r'), ord('R')):
                    corners["tl"] = [x_min, y_min]
                    corners["tr"] = [x_max, y_min]
                    corners["br"] = [x_max, y_max]
                    corners["bl"] = [x_min, y_max]
        finally:
            # Ensure HighGUI fully closes before returning to main execution.
            cv2.setMouseCallback(window_name, lambda *args: None)
            cv2.destroyWindow(window_name)
            cv2.waitKey(1)
            cv2.destroyAllWindows()
            cv2.waitKey(1)

        if accepted:
            x0, y0 = corners["tl"]
            x1, y1 = corners["br"]
            selected_crop = {
                "crop_width_min": int(x0),
                "crop_width_max": int(x1),
                "crop_height_min": int(img_h - y1),
                "crop_height_max": int(img_h - y0),
            }
            self.analysis_crop = selected_crop
            print(f"Selected analysis crop: {self.analysis_crop}")

            # Reset prior ruler state; this session may repick or skip.
            self.ruler_ref_p1_px = None
            self.ruler_ref_p2_px = None
            self.ruler_ref_distance_mm = None
            self.ruler_mm_per_px = None
            self.ruler_px_per_mm = None
            self.ruler_axis_unit = None
            self.ruler_axis_perp_unit = None
            self.ruler_calib_meta = None

            # Step B: ruler reference picking (150 mm known distance).
            ruler_window_name = "Ruler Reference Setup"
            ruler_points = []
            ruler_known_mm = 150.0
            ruler_confirmed = False
            ruler_skipped = False
            min_valid_dist_px = 5.0

            def on_ruler_mouse(event, mx, my, flags, param):
                if event != cv2.EVENT_LBUTTONDOWN:
                    return
                mx = int(np.clip(mx, 0, img_w - 1))
                my = int(np.clip(my, 0, img_h - 1))
                if len(ruler_points) >= 2:
                    return
                ruler_points.append((mx, my))

            print("Ruler reference setup:")
            print("- Click two points on the physical ruler (known distance = 100.0 mm).")
            print("- Press ENTER or SPACE to confirm once two points are selected.")
            print("- Press R to reset ruler points.")
            print("- Press Q or ESC to skip ruler calibration (crop remains accepted).")

            cv2.namedWindow(ruler_window_name, cv2.WINDOW_NORMAL)
            cv2.setMouseCallback(ruler_window_name, on_ruler_mouse)
            try:
                while True:
                    if cv2.getWindowProperty(ruler_window_name, cv2.WND_PROP_VISIBLE) < 1:
                        ruler_skipped = True
                        break

                    display = frame.copy()
                    cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 0), 2)

                    # Draw picked ruler points and optional line.
                    for idx, pt in enumerate(ruler_points):
                        cv2.circle(display, tuple(pt), 7, (0, 255, 255), -1)
                        cv2.putText(
                            display,
                            f"P{idx + 1}",
                            (pt[0] + 8, pt[1] - 8),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (0, 255, 255),
                            2
                        )

                    dist_px = None
                    mm_per_px = None
                    if len(ruler_points) == 2:
                        p1 = np.asarray(ruler_points[0], dtype=float)
                        p2 = np.asarray(ruler_points[1], dtype=float)
                        dist_px = float(np.linalg.norm(p2 - p1))
                        if dist_px >= 1e-9:
                            mm_per_px = ruler_known_mm / dist_px
                        cv2.line(display, ruler_points[0], ruler_points[1], (50, 200, 255), 2)

                    cv2.putText(
                        display,
                        "Pick 2 ruler points (150.0 mm): ENTER confirm | R reset | Q/ESC skip",
                        (20, 35),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (255, 255, 255),
                        2
                    )
                    cv2.putText(
                        display,
                        f"Crop x:[{x0},{x1}] y:[{y0},{y1}]",
                        (20, 68),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.65,
                        (0, 255, 0),
                        2
                    )
                    if dist_px is not None and mm_per_px is not None:
                        cv2.putText(
                            display,
                            f"dist_px={dist_px:.2f} | mm/px={mm_per_px:.6f}",
                            (20, 100),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.65,
                            (50, 255, 120),
                            2
                        )

                    cv2.imshow(ruler_window_name, display)
                    key = cv2.waitKey(20) & 0xFF

                    if key in (13, 32):  # Enter/Space
                        if len(ruler_points) < 2:
                            print("Ruler confirmation rejected: select two points first.")
                            continue
                        p1 = np.asarray(ruler_points[0], dtype=float)
                        p2 = np.asarray(ruler_points[1], dtype=float)
                        dist_px = float(np.linalg.norm(p2 - p1))
                        if dist_px < min_valid_dist_px:
                            print(
                                "Ruler confirmation rejected: selected points are too close "
                                f"({dist_px:.3f} px < {min_valid_dist_px:.1f} px). Re-pick points."
                            )
                            continue
                        ruler_confirmed = True
                        break
                    if key in (27, ord('q')):  # Esc/q
                        ruler_skipped = True
                        break
                    if key in (ord('r'), ord('R')):
                        ruler_points.clear()
            finally:
                cv2.setMouseCallback(ruler_window_name, lambda *args: None)
                cv2.destroyWindow(ruler_window_name)
                cv2.waitKey(1)
                cv2.destroyAllWindows()
                cv2.waitKey(1)

            if ruler_confirmed:
                p1 = np.asarray(ruler_points[0], dtype=np.float64)
                p2 = np.asarray(ruler_points[1], dtype=np.float64)
                axis_vec = p2 - p1
                pixel_dist = float(np.linalg.norm(axis_vec))
                axis_unit = axis_vec / pixel_dist
                axis_perp_unit = np.array([axis_unit[1], -axis_unit[0]], dtype=np.float64)
                mm_per_px = float(ruler_known_mm / pixel_dist)
                px_per_mm = float(pixel_dist / ruler_known_mm)

                self.ruler_ref_p1_px = tuple(np.round(p1).astype(int))
                self.ruler_ref_p2_px = tuple(np.round(p2).astype(int))
                self.ruler_ref_distance_mm = float(ruler_known_mm)
                self.ruler_mm_per_px = mm_per_px
                self.ruler_px_per_mm = px_per_mm
                self.ruler_axis_unit = axis_unit.astype(np.float64)
                self.ruler_axis_perp_unit = axis_perp_unit.astype(np.float64)
                self.ruler_calib_meta = {
                    "source": "manual_setup_analysis_crop",
                    "known_distance_mm": float(ruler_known_mm),
                    "pixel_distance": float(pixel_dist),
                    "crop": dict(self.analysis_crop),
                    "p1_px": [int(self.ruler_ref_p1_px[0]), int(self.ruler_ref_p1_px[1])],
                    "p2_px": [int(self.ruler_ref_p2_px[0]), int(self.ruler_ref_p2_px[1])],
                }

                print("Ruler-reference scale set successfully.")
                print(f"  p1_px={self.ruler_ref_p1_px}, p2_px={self.ruler_ref_p2_px}")
                print(f"  pixel_distance={pixel_dist:.6f} px")
                print(f"  mm_per_px={self.ruler_mm_per_px:.9f}")
                print(f"  px_per_mm={self.ruler_px_per_mm:.9f}")
            elif ruler_skipped:
                print("Crop accepted; ruler calibration skipped. Ruler scale remains unset.")
        else:
            self.analysis_crop = dict(self.default_analysis_crop)
            print(f"Manual crop cancelled. Using default analysis crop: {self.analysis_crop}")

        if cam_opened_here:
            cam.release()

        return dict(self.analysis_crop)

    def connect_to_camera(self, cam_port=None, show_preview=False, enable_manual_focus=False, manual_focus_val=60):
        """
        Connects to the camera in the specified port and optionally shows a live preview.
        """
        if cam_port is None:
            cam_port = 0
        self.cam_port = cam_port

        # Removed cv2.CAP_DSHOW for Mac compatibility
        self.cam = cv2.VideoCapture(self.cam_port)

        # display showing preview
        if show_preview:
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

            print("Showing live preview. Press 'q' to exit.")
            while True:
                # Capture frame-by-frame
                ret, frame = self.cam.read()

                # If frame is read correctly, ret is True
                if not ret:
                    print("Can't receive frame (stream end?). Exiting ...")
                    break

                # Display the resulting frame
                cv2.imshow('frame', frame)

                # Break the loop if 'q' key is pressed
                if cv2.waitKey(1) == ord('q'):
                    break

            # When everything done, release the capture
            cv2.destroyAllWindows()

    def disconnect_camera(self):
        """ Disconnects the camera attached to the calibration object. """
        if self.cam is None:
            print("Error: camera object doesn't exist. Did you connect to it?")
            return
        else:
            self.cam.release()

    def test_focus(self, cam_port=None):
        """ Focus sweep utility. Captures images at a range of focus values. """
        if cam_port is None:
            cam_port = self.cam_port

        if self.cam_port is None:
            print("Error: no camera found. Did you forget to connect to the camera?")
            return 0

        self.cam_port = cam_port

        if self.cam is None:
            self.cam = cv2.VideoCapture(self.cam_port)

        # Disable autofocus and sweep manual focus
        self.cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
        focus_values = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]

        # make sure we are in the correct directory
        os.chdir(self.calibration_data_folder)

        if not os.path.isdir("focus_folder"):
            os.mkdir("focus_folder")
        else:
            print("You've already tested the focus of the camera for this project. This will overwrite the current data.")

        os.chdir("focus_folder")

        for focus in focus_values:
            print(focus, end=" ")
            self.cam.set(cv2.CAP_PROP_FOCUS, focus)
            self.cam.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
            self.cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

            result, image = self.cam.read()
            filename = "focus of " + str(focus) + ".png"

            if result:
                cv2.imwrite(filename, image)

        os.chdir("../")
        return 1

    def connect_to_robot(self, duet_web_address=r"http://192.168.2.21"):
        """ Connects to the robot via the Duet interface. """
        self.rrf = DuetWebAPI(duet_web_address)
        print("Connection attempted. Requesting diagnostics.")
        resp = self.rrf.send_code("M122")  # synchronous
        print("Returned diagnostics data:")
        print(resp)
        return 1

    def calibrate(self,
                  manual_focus_val=60,
                  jogging_feedrate=200,
                  # Axis names
                  robot_front_axis_name="X",
                  robot_stage_y_axis_name="Y",
                  robot_stage_z_axis_name="Z",
                  robot_rear_axis_name="B",
                  robot_rotation_axis_name="C",
                  robot_rotation_axis_180_deg=180.0,
                  # Pull plan
                  b_start=0.0,
                  b_steps=25,
                  b_step_size=-0.2,
                  # Probe points
                  probe_points=None,
                  enable_manual_focus=True,
                  # New: capture release images after pull sequence
                  enable_release_imaging=True,
                  num_curl_uncurl_cycles=1,
                  capture_dwell_s=0.5,
                  combine_redundant_passes=True,
                  redundant_pass_combination="average",
                  save_pass_index_in_filename=True):
        """
        Probes at 5 XYZ locations and for each location:
        - Capture pull images while moving B from b_start toward the pulled state.
        - Optionally capture release images while moving B back toward b_start.
        - Repeat for all 4 camera orientations.

        Filenames begin with numeric tokens compatible with the downstream parser:
            "{orientation}_{X}_{B}_..."
        and now also include a phase token:
            DIRpull  or  DIRrelease
        Optional redundant capture cycles append PASS1 / PASS2.
        """
        if self.cam is None or self.rrf is None or self.cam_port is None:
            print("Not connected to camera or robot. Please run connect_to_camera followed by connect_to_robot.")
            print("Exiting...")
            return

        if probe_points is None:
            probe_points = [
                (30.0, 0.0, -80.0),
                (90.0, 0.0, -80.0),
                (90.0, 0.0, -110.0),
                (30.0, 0.0, -110.0),
                (75.0, 0.0, -90.0),
            ]

        if not hasattr(self, 'calibration_data_folder') or not os.path.exists(self.calibration_data_folder):
            print("Error: Calibration data folder not found. Make sure the class was initialized properly.")
            return

        os.chdir(self.calibration_data_folder)

        num_curl_uncurl_cycles = int(num_curl_uncurl_cycles)
        if num_curl_uncurl_cycles not in (1, 2):
            raise ValueError("num_curl_uncurl_cycles must be 1 or 2.")
        capture_dwell_s = max(0.0, float(capture_dwell_s))
        redundant_pass_combination = str(redundant_pass_combination).strip().lower()
        if redundant_pass_combination not in {"average", "keep_separate"}:
            raise ValueError("redundant_pass_combination must be 'average' or 'keep_separate'.")
        combine_redundant_passes = bool(combine_redundant_passes)
        save_pass_index_in_filename = bool(save_pass_index_in_filename)
        self._combine_redundant_passes = combine_redundant_passes
        self._redundant_pass_combination = redundant_pass_combination

        settling_time_large = 5.0
        settling_time_small = 0.2
        rotation_settling_time = 2.0
        small_move_threshold = 2.0

        if not os.path.isdir("raw_image_data_folder"):
            os.mkdir("raw_image_data_folder")
        else:
            print("You already have raw image data for this project. This will overwrite the current data.")

        os.chdir("raw_image_data_folder")

        cam = self.cam if self.cam is not None else cv2.VideoCapture(self.cam_port if self.cam_port is not None else 0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

        def capture_and_save(orientation: int, x: float, y: float, z: float, b: float,
                             probe_idx: int, step_idx: int, motion_phase: str, pass_idx: int = 1):
            _ = cam.read()
            ret, image = cam.read()
            if not ret:
                ret, image = cam.read()

            filename = (
                f"{orientation}_{x:.2f}_{b:.2f}"
                f"_Y{y:.2f}_Z{z:.2f}_P{probe_idx:02d}_S{step_idx:02d}_DIR{motion_phase}"
            )
            if save_pass_index_in_filename:
                filename += f"_PASS{int(pass_idx)}"
            filename += ".png"
            if ret:
                cv2.imwrite(filename, image)
                print(f" ✓ Saved {filename}")
            else:
                print(f" ✗ ERROR: Could not save {filename}")

        def wait_for_duet_motion_complete(extra_settle: float = 0.0):
            try:
                self.rrf.send_code("M400")
            except Exception as e:
                print(f" Warning: M400 wait failed ({e}); falling back to timed settle only.")
            if extra_settle > 0:
                time.sleep(extra_settle)

        def wait_for_move(cur: dict, tgt: dict, feedrate: float):
            diffs = []
            for ax in tgt:
                if ax in cur and tgt[ax] is not None and cur[ax] is not None and tgt[ax] != cur[ax]:
                    diffs.append(tgt[ax] - cur[ax])
            if not diffs:
                time.sleep(settling_time_small)
                return
            distance = math.sqrt(sum(d * d for d in diffs))
            is_small = distance <= small_move_threshold
            move_type = "small" if is_small else "large"
            settle = settling_time_small if is_small else settling_time_large
            print(f" {move_type} move: {distance:.2f}mm at {feedrate}mm/min - waiting for Duet (M400) + {settle:.1f}s settle")
            wait_for_duet_motion_complete(extra_settle=settle)

        def move_abs(fr: float, cur: dict, **axes_targets):
            cmd = ["G90", "G1"]
            tgt = dict(cur)
            for ax, val in axes_targets.items():
                if val is None:
                    continue
                cmd.append(f"{ax}{val}")
                tgt[ax] = float(val)
            cmd.append(f"F{fr}")
            g = " ".join(cmd)
            print(f" Command: {g}")
            self.rrf.send_code(g)
            wait_for_move(cur, tgt, fr)
            return tgt

        def rotate_rel(axis_name: str, c_units: float, fr=None):
            if fr is None:
                fr = jogging_feedrate / 0.025
            g = f"G91 G1 {axis_name}{c_units} F{fr}"
            print(f" Command: {g}")
            self.rrf.send_code(g)
            angle_deg = (c_units / robot_rotation_axis_180_deg) * 180.0 if robot_rotation_axis_180_deg else 0.0
            print(f" Rotating {angle_deg:.1f}° - waiting for Duet (M400) + {rotation_settling_time:.1f}s settle")
            wait_for_duet_motion_complete(extra_settle=rotation_settling_time)
            self.rrf.send_code("G90")

        def build_phase_schedule():
            phases = []
            for cycle_idx in range(1, num_curl_uncurl_cycles + 1):
                phases.append(("pull", cycle_idx))
                if enable_release_imaging:
                    phases.append(("release", cycle_idx))
            return phases

        def run_motion_pass(orientation_id: int, x: float, y: float, z: float, probe_idx: int,
                            motion_phase: str, pass_idx: int, b_values, label_total: int):
            nonlocal pos, total_images
            motion_phase = str(motion_phase).strip().lower()
            if motion_phase == "pull":
                print(f" Starting pull pass {pass_idx}/{label_total}")
            else:
                print(f" Starting release pass {pass_idx}/{label_total}")
            for step_idx, b_val in enumerate(b_values):
                print(
                    f" {motion_phase.capitalize()} step {step_idx:02d}/{len(b_values) - 1:02d}: "
                    f"{robot_rear_axis_name}={b_val:.2f}"
                )
                pos = move_abs(0.3 * jogging_feedrate, pos, **{robot_rear_axis_name: b_val})
                wait_for_duet_motion_complete(extra_settle=capture_dwell_s)
                capture_and_save(orientation_id, x, y, z, b_val, probe_idx, step_idx, motion_phase, pass_idx=pass_idx)
                total_images += 1

        def run_pull_release_sequence_for_orientation(orientation_id: int, x: float, y: float, z: float,
                                                      probe_idx: int, max_pull_steps=None):
            nonlocal pos
            steps_to_run = b_steps if max_pull_steps is None else int(max_pull_steps)
            steps_to_run = int(np.clip(steps_to_run, 0, b_steps))
            pos = move_abs(0.3 * jogging_feedrate, pos, **{robot_rear_axis_name: b_start})
            pull_b_values = [b_start + step_idx * b_step_size for step_idx in range(0, steps_to_run + 1)]
            release_b_values = list(reversed(pull_b_values))
            if len(release_b_values) > 1:
                release_b_values = release_b_values[1:]

            total_pull_passes = num_curl_uncurl_cycles
            total_release_passes = num_curl_uncurl_cycles if enable_release_imaging else 0
            for motion_phase, pass_idx in build_phase_schedule():
                if motion_phase == "pull":
                    run_motion_pass(
                        orientation_id, x, y, z, probe_idx,
                        motion_phase="pull", pass_idx=pass_idx,
                        b_values=pull_b_values, label_total=total_pull_passes,
                    )
                elif len(release_b_values) > 0:
                    run_motion_pass(
                        orientation_id, x, y, z, probe_idx,
                        motion_phase="release", pass_idx=pass_idx,
                        b_values=release_b_values, label_total=total_release_passes,
                    )

            if not enable_release_imaging:
                pos = move_abs(jogging_feedrate, pos, **{robot_rear_axis_name: b_start})

        pos = {
            robot_front_axis_name: 0.0,
            robot_stage_y_axis_name: 0.0,
            robot_stage_z_axis_name: 0.0,
            robot_rear_axis_name: 0.0,
        }

        print("Starting probe sequence (4-orientation capture: 0/180/+90/-90)...")
        print(f"Probe points (X,Y,Z): {probe_points}")
        print(f"Pull axis: {robot_rear_axis_name} | steps={b_steps} | step_size={b_step_size}mm | start={b_start}mm")
        print(f"Rotation axis: {robot_rotation_axis_name} | 180deg={robot_rotation_axis_180_deg} axis units")
        print(f"Release imaging enabled: {enable_release_imaging}")
        print(f"Redundant curl/uncurl cycles: {num_curl_uncurl_cycles}")
        print(f"Capture dwell after M400: {capture_dwell_s:.2f}s")
        print(f"Combine redundant passes: {combine_redundant_passes} ({redundant_pass_combination})")

        print("\nMoving pull axis to start position...")
        pos = move_abs(jogging_feedrate, pos, **{robot_rear_axis_name: b_start})

        offplane_step_fraction = 0.4
        n_partial_steps = max(1, int(np.floor(offplane_step_fraction * b_steps)))
        print(
            f"±90 acquisition truncated to first {n_partial_steps}/{b_steps} pull steps "
            f"({100.0 * offplane_step_fraction:.0f}%) due to tip detectability limits"
        )

        total_images = 0
        for probe_idx, (x, y, z) in enumerate(probe_points, start=1):
            print(f"\n{'=' * 70}")
            print(f"PROBE {probe_idx}/{len(probe_points)}: X={x}, Y={y}, Z={z}")
            print(f"{'=' * 70}")

            print(" Moving to probe XYZ...")
            pos = move_abs(
                jogging_feedrate,
                pos,
                **{
                    robot_front_axis_name: x,
                    robot_stage_y_axis_name: y,
                    robot_stage_z_axis_name: z
                }
            )

            print(f" Setting {robot_rear_axis_name} to start ({b_start:.2f})...")
            pos = move_abs(jogging_feedrate, pos, **{robot_rear_axis_name: b_start})

            print(" Phase 1: orientation 0 (C = 0 deg)")
            run_pull_release_sequence_for_orientation(0, x, y, z, probe_idx)

            print(" Rotating to orientation 1 (C = 180 deg)...")
            rotate_rel(robot_rotation_axis_name, robot_rotation_axis_180_deg)
            print(" Phase 2: orientation 1 (C = 180 deg)")
            run_pull_release_sequence_for_orientation(1, x, y, z, probe_idx)

            print(" Rotating back to C = 0 deg...")
            rotate_rel(robot_rotation_axis_name, -robot_rotation_axis_180_deg)

            c_quarter = robot_rotation_axis_180_deg / 2.0
            print(" Rotating to orientation 2 (C = +90 deg)...")
            rotate_rel(robot_rotation_axis_name, +c_quarter)
            print(" Phase 3: orientation 2 (C = +90 deg)")
            run_pull_release_sequence_for_orientation(2, x, y, z, probe_idx, max_pull_steps=n_partial_steps)

            print(" Rotating to orientation 3 (C = -90 deg)...")
            rotate_rel(robot_rotation_axis_name, -robot_rotation_axis_180_deg)
            print(" Phase 4: orientation 3 (C = -90 deg)")
            run_pull_release_sequence_for_orientation(3, x, y, z, probe_idx, max_pull_steps=n_partial_steps)

            print(" Rotating back to C = 0 deg...")
            rotate_rel(robot_rotation_axis_name, +c_quarter)

        print("\n" + "=" * 70)
        print("ALL PROBES FINISHED!")
        print("=" * 70)
        print(f"Total images captured: {total_images}")
        print(f"Final position: {pos}")

        os.chdir(self.calibration_data_folder)
        return 1

    def find_ctr_tip_skeleton(self, binary_image, base_band_frac=0.05, do_erosion_break=False, prune_spurs=False, min_spur_len=20, return_tip_angle=False, tip_angle_path_len=75, return_debug=False):
        """
        binary_image: uint8 0/255 where tube is dark (0) and background is white (255)

        Returns: (tip_row, tip_col) in binary_image coordinates
        """
        fg = (binary_image == 0).astype(np.uint8)

        if do_erosion_break:
            fg = cv2.erode(fg, np.ones((3, 3), np.uint8), iterations=1)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=1)
        #fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)

        n, lab, stats, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
        if n <= 1:
            raise ValueError("No foreground component found (check threshold/crop).")

        h, w = fg.shape
        top_band = max(1, int(base_band_frac * h))
        candidates = np.unique(lab[:top_band, :])
        candidates = candidates[candidates != 0]

        if len(candidates) == 0:
            target = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        else:
            target = candidates[np.argmax(stats[candidates, cv2.CC_STAT_AREA])]

        mask = (lab == target).astype(np.uint8)
        skel = skeletonize(mask > 0).astype(np.uint8)

        if skel.sum() == 0:
            raise ValueError("Skeleton empty after cleanup (try different morphology/threshold).")

        if prune_spurs:
            skel = _prune_short_spurs(skel, min_branch_len=int(min_spur_len))
            if skel.sum() == 0:
                raise ValueError("Skeleton vanished after spur pruning (min_spur_len too big?).")

        # Base selection: ANY skeleton pixels in the top band (not just endpoints).
        base_pixels = np.column_stack(np.where(skel[:top_band, :] == 1))
        if len(base_pixels) == 0:
            # fallback: topmost skeleton pixel(s)
            ys, xs = np.where(skel == 1)
            y0 = ys.min()
            base_pixels = np.column_stack(np.where((skel == 1) & (np.arange(h)[:, None] == y0)))

        # Keep the full skeleton available so tip detection can still reach the distal end.
        full_skel = skel.copy()
        full_dist = _multisource_bfs_geodesic(full_skel, base_pixels)
        if (full_dist >= 0).sum() == 0:
            raise ValueError("Geodesic BFS failed (disconnected skeleton?)")

        skel_trunk, trunk_dist = _extract_main_trunk_skeleton(full_skel, base_pixels)
        if skel_trunk.sum() > 0 and (trunk_dist >= 0).sum() > 0:
            full_max = float(np.max(full_dist))
            trunk_max = float(np.max(trunk_dist))
            # Only trust the pruned trunk if it still reaches most of the distal extent.
            if full_max <= 0 or trunk_max >= 0.97 * full_max:
                skel = skel_trunk
                dist = trunk_dist
            else:
                skel = full_skel
                dist = full_dist
        else:
            skel = full_skel
            dist = full_dist

        # Robust distal pose from dominant PCA axis near tip (avoids Y-split branch issues).
        if return_tip_angle:
            tip_y, tip_x, tip_angle_deg, tip_path = _tip_pose_from_distal_pca(
                skel_u8=skel,
                dist=dist,
                mask_u8=mask,
                distal_window=70,
                roi_margin=18,
                tangent_len=tip_angle_path_len,
                radius_frac=0.18,
                tip_keep_frac=0.40,
                weight_power=0.35,
            )
            if return_debug:
                debug_data = {
                    "skeleton": skel.copy(),
                    "dist": dist.copy(),
                    "tip_path": tip_path,
                }
                return tip_y, tip_x, tip_angle_deg, debug_data
            return tip_y, tip_x, tip_angle_deg

        tip_y, tip_x, _, tip_path = _tip_pose_from_distal_pca(
            skel_u8=skel,
            dist=dist,
            mask_u8=mask,
            distal_window=70,
            roi_margin=18,
            tangent_len=tip_angle_path_len,
            radius_frac=0.10,
            tip_keep_frac=0.20,
            weight_power=0.05,
        )
        if return_debug:
            debug_data = {
                "skeleton": skel.copy(),
                "dist": dist.copy(),
                "tip_path": tip_path,
            }
            return tip_y, tip_x, debug_data
        return tip_y, tip_x


    @staticmethod
    def _parse_capture_context_from_filename(file_name):
        """Return parsed filename metadata for calibration images."""
        base = os.path.splitext(os.path.basename(file_name))[0]
        parts = base.split("_")

        orientation = 0
        if len(parts) > 0:
            try:
                candidate = int(parts[0])
                if candidate in (0, 1, 2, 3):
                    orientation = candidate
            except Exception:
                pass

        values = {}
        motion_phase = "pull"
        pass_idx = 1
        for p in parts:
            matched_prefixed = False
            if p.startswith("DIR"):
                phase_val = p[3:].strip().lower()
                if phase_val in ("pull", "release"):
                    motion_phase = phase_val
                continue
            if p.startswith("PASS"):
                try:
                    pass_idx = max(1, int(float(p[4:])))
                except Exception:
                    pass
                continue
            for key in ("tipX", "tipY", "tipZ", "stageX", "stageY", "stageZ", "reqA", "useA"):
                if p.startswith(key):
                    try:
                        values[key] = float(p[len(key):])
                        matched_prefixed = True
                    except Exception:
                        pass
                    break
            if matched_prefixed:
                continue
            if len(p) >= 2 and p[0] in ("X", "Y", "Z", "B", "C"):
                try:
                    values[p[0]] = float(p[1:])
                except Exception:
                    pass

        x_value = values["tipX"] if "tipX" in values else values.get("X")
        if x_value is not None and "B" in values:
            ntnl_pos = float(x_value)
            ss_pos = float(values["B"])
        elif len(parts) >= 3:
            ntnl_pos = float(parts[1])
            ss_pos = float(parts[2])
        else:
            raise ValueError(
                "Could not parse X/B values from filename: "
                f"{file_name}. Expected either '<ori>_<X>_<B>_...' or tokens like 'X...' and 'B...'."
            )

        return {
            "orientation": int(orientation),
            "ntnl_pos": ntnl_pos,
            "ss_pos": ss_pos,
            "motion_phase": motion_phase,
            "motion_phase_code": 0 if motion_phase == "pull" else 1,
            "pass_idx": int(pass_idx),
        }

    def analyze_data(self, image_file_name, crop_width_min=None, crop_width_max=None, crop_height_min=None, crop_height_max=None, threshold=200, ):
        """ Analyzes the data for the calibrations. Only analyzes one image. """

        file_ctx = self._parse_capture_context_from_filename(image_file_name)

        # Fixed path handling for Mac
        raw_data_folder = os.path.join(self.calibration_data_folder, "raw_image_data_folder")
        tip_locations_array_coarse = np.zeros((len(os.listdir(raw_data_folder)), 8))
        tip_locations_array_fine = np.zeros((len(os.listdir(raw_data_folder)), 8))
        i = 0

        # Fixed path handling for Mac
        image = cv2.imread(os.path.join(raw_data_folder, image_file_name))
        if image is None:
            raise FileNotFoundError(f"Could not read image: {os.path.join(raw_data_folder, image_file_name)}")
        full_grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        if crop_width_min is None:
            crop_width_min = self.analysis_crop["crop_width_min"]
        if crop_width_max is None:
            crop_width_max = self.analysis_crop["crop_width_max"]
        if crop_height_min is None:
            crop_height_min = self.analysis_crop["crop_height_min"]
        if crop_height_max is None:
            crop_height_max = self.analysis_crop["crop_height_max"]

        crop_x_min_img = crop_width_min
        crop_x_max_img = crop_width_max
        crop_y_min_img = np.shape(image)[0] - crop_height_max
        crop_y_max_img = np.shape(image)[0] - crop_height_min

        cropped_image = image[crop_y_min_img:crop_y_max_img, crop_x_min_img:crop_x_max_img, :]
        grayscale_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        grayscale_eq = clahe.apply(grayscale_image)
        grayscale_blur = cv2.GaussianBlur(grayscale_eq, (3, 3), 0)
        _, binary_image = cv2.threshold(
            grayscale_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

        tip_row, tip_column, tip_angle_deg, tip_debug = self.find_ctr_tip_skeleton(
            binary_image,
            min_spur_len=5,
            return_tip_angle=True,
            return_debug=True,
        )
        tip_angle_deg = _normalize_tip_angle_deg(tip_angle_deg)

        skel = tip_debug["skeleton"]
        dist = tip_debug["dist"]
        tip_path = tip_debug["tip_path"]

        ys, xs = np.where(skel == 1)
        dvals = dist[ys, xs]
        idx = int(np.argmax(dvals))
        tip_row_legacy, tip_col_legacy = int(ys[idx]), int(xs[idx])

        endpoints = _endpoints_8(skel)  # Nx2 (y, x)
        if endpoints.size > 0:
            ed = dist[endpoints[:, 0], endpoints[:, 1]]
            eidx = int(np.argmax(ed))
            if ed[eidx] >= dist[tip_row_legacy, tip_col_legacy] - 1:
                tip_row_legacy = int(endpoints[eidx, 0])
                tip_col_legacy = int(endpoints[eidx, 1])

        tip_y = float(np.clip(tip_row, 0, binary_image.shape[0] - 1))
        tip_x = float(np.clip(tip_column, 0, binary_image.shape[1] - 1))

        zoom_x_min = int(max(int(round(tip_x)) - 75, 0))
        zoom_x_max = int(min(int(round(tip_x)) + 75, binary_image.shape[1] - 1))
        zoom_y_min = int(max(int(round(tip_y)) - 75, 0))
        zoom_y_max = int(min(int(round(tip_y)) + 75, binary_image.shape[0] - 1))

        yy_refined, xx_refined, tip_refine_dbg = refine_tip_parallel_centerline(
            grayscale=grayscale_image,
            binary_image=binary_image,
            tip_yx=(int(round(tip_y)), int(round(tip_x))),
            tip_angle_deg=tip_angle_deg,
            section_near_r=float(self.tip_parallel_section_near_r),
            section_far_r=float(self.tip_parallel_section_far_r),
            scan_half_r=float(self.tip_parallel_scan_half_r),
            num_sections=int(self.tip_parallel_num_sections),
            cross_step_px=float(self.tip_parallel_cross_step_px),
            ray_step_px=float(self.tip_parallel_ray_step_px),
            ray_max_len_r=float(self.tip_parallel_ray_max_len_r),
        )
        yy_selected, xx_selected, tip_select_dbg = _select_tip_candidate(
            coarse_tip_yx=(tip_y, tip_x),
            refined_tip_yx=(yy_refined, xx_refined),
            tip_dbg=tip_refine_dbg,
            mode=self.tip_refine_mode,
        )

        cnn_tip_abs = None
        cnn_tip_dbg = None
        if self.tip_refiner_enabled:
            anchor_lookup_abs = {
                "coarse": (float(tip_x + crop_x_min_img), float(tip_y + crop_y_min_img)),
                "refined": (float(xx_refined + crop_x_min_img), float(yy_refined + crop_y_min_img)),
                "selected": (float(xx_selected + crop_x_min_img), float(yy_selected + crop_y_min_img)),
            }
            anchor_x_abs, anchor_y_abs = anchor_lookup_abs[self.tip_refiner_anchor_name]
            try:
                cnn_tip_dbg = self.predict_tip_refiner_abs(
                    full_grayscale_image,
                    anchor_x_abs=anchor_x_abs,
                    anchor_y_abs=anchor_y_abs,
                )
                if cnn_tip_dbg is not None:
                    cnn_tip_abs = (float(cnn_tip_dbg["y_abs"]), float(cnn_tip_dbg["x_abs"]))
                    if self.tip_refiner_use_as_selected:
                        yy_selected = float(cnn_tip_dbg["y_abs"] - crop_y_min_img)
                        xx_selected = float(cnn_tip_dbg["x_abs"] - crop_x_min_img)
                        tip_select_dbg = dict(tip_select_dbg) if isinstance(tip_select_dbg, dict) else {}
                        tip_select_dbg["selected_tip_source"] = "cnn"
                        tip_select_dbg["selected_tip_reason"] = "cnn_tip_refiner"
            except Exception as exc:
                cnn_tip_dbg = {"error": str(exc)}

        orientation = int(file_ctx['orientation'])
        ntnl_pos = float(file_ctx['ntnl_pos'])
        ss_pos = float(file_ctx['ss_pos'])
        motion_phase_code = float(file_ctx['motion_phase_code'])
        pass_idx = float(file_ctx.get('pass_idx', 1))

        tip_locations_array_coarse[i, :] = np.array(
            [
                tip_y + crop_y_min_img,
                tip_x + crop_x_min_img,
                float(orientation),
                float(ss_pos),
                float(ntnl_pos),
                float(tip_angle_deg),
                motion_phase_code,
                pass_idx,
            ]
        )

        crop_x_min = zoom_x_min
        crop_x_max = zoom_x_max
        crop_y_min = zoom_y_min
        crop_y_max = zoom_y_max

        # Use inclusive max indices when slicing, then compute a clamped seed point.
        zoomed_tip = binary_image[crop_y_min:crop_y_max + 1, crop_x_min:crop_x_max + 1]
        h, w = zoomed_tip.shape[:2]
        mask = np.zeros((h + 2, w + 2), np.uint8)
        floodfilled = zoomed_tip.copy()

        seed_x = int(tip_column - crop_x_min)
        seed_y = int(tip_row - crop_y_min)
        seed_x = max(0, min(seed_x, w - 1))
        seed_y = max(0, min(seed_y, h - 1))

        cv2.floodFill(floodfilled, mask, (seed_x, seed_y), 128)

        isolated_needle = np.where(floodfilled == 128, 0, 255).astype(np.uint8)
        if isolated_needle[0, -1] != 255:
            isolated_needle = np.where(floodfilled == 128, 255, 0).astype(np.uint8)

        cols = np.arange(np.shape(isolated_needle)[1])
        rows = np.arange(np.shape(isolated_needle)[0])
        x_grid, y_grid = np.meshgrid(cols, rows)

        x_points = np.multiply(x_grid, (255 - isolated_needle) / 255).flatten()
        y_points = np.multiply(y_grid, (255 - isolated_needle) / 255).flatten()

        x_points = x_points[isolated_needle.flatten() != 255]
        y_points = y_points[isolated_needle.flatten() != 255]

        tip_location = np.array([0, 0], dtype=float)
        p_1 = None
        p_2 = None
        residual_1 = None
        residual_2 = None

        # Fit 1 / Fit 2 and fine tip detection disabled: use coarse tip only.
        # if x_points.size == 0 or y_points.size == 0:
        # # Fallback to coarse tip when no needle pixels are detected.
        # tip_location = np.array([tip_column - crop_x_min, tip_row - crop_y_min], dtype=float)
        # else:
        # coefficients_1, residual_1, _, _, _ = np.polyfit(x_points, y_points, 1, full=True)
        # coefficients_2, residual_2, _, _, _ = np.polyfit(y_points, x_points, 1, full=True)
        # # p_1 = np.poly1d(coefficients_1)
        # # p_2 = np.poly1d(coefficients_2)

        tip_location = np.array([tip_column - crop_x_min, tip_row - crop_y_min], dtype=float)

        grayscale_tip = grayscale_image[crop_y_min:crop_y_max + 1, crop_x_min:crop_x_max + 1]
        if grayscale_tip.size == 0:
            raise ValueError(
                "Crop resulted in empty grayscale image; check crop bounds against image size."
            )

        # box_length = 5
        # search_radius = 10
        # foundPoint = False
        # # if p_1 is not None and p_2 is not None and residual_1 > residual_2 and not np.isclose(ntnl_pos, ss_pos):
        # if orientation == 1:
        # search_pos_min = np.max(y_points).astype(int) - search_radius
        # search_pos_max = np.max(y_points).astype(int) + search_radius
        # else:
        # search_pos_min = np.min(y_points).astype(int) - search_radius
        # search_pos_max = np.min(y_points).astype(int) + search_radius
        # # for row in range(search_pos_min, search_pos_max):
        # col = p_2(row)
        # # if col < box_length:
        # continue
        # if col > np.shape(zoomed_tip)[1] - box_length:
        # continue
        # # col_idx_min = max(np.round(col - box_length).astype(int), 0)
        # col_idx_max = min(np.round(col + box_length).astype(int), np.shape(zoomed_tip)[1] - 1)
        # row_idx_min = max(np.round(row - box_length).astype(int), 0)
        # row_idx_max = min(np.round(row + box_length).astype(int), np.shape(zoomed_tip)[0] - 1)
        # # if row_idx_max <= row_idx_min or col_idx_max <= col_idx_min:
        # continue
        # # if not foundPoint:
        # foundPoint = True
        # baseline = np.min(grayscale_tip[row_idx_min:row_idx_max, col_idx_min:col_idx_max])
        # look_for_dark_pixels = baseline > 240
        # # if foundPoint and look_for_dark_pixels:
        # if np.min(grayscale_tip[row_idx_min:row_idx_max, col_idx_min:col_idx_max]) < 200:
        # tip_location = np.array([col, row])
        # break
        # elif foundPoint and not look_for_dark_pixels:
        # if np.min(grayscale_tip[row_idx_min:row_idx_max, col_idx_min:col_idx_max]) > 200:
        # tip_location = np.array([col, row])
        # break
        # elif p_1 is not None and p_2 is not None:
        # search_pos_min = (np.max(x_points).astype(int) - search_radius)
        # search_pos_max = (np.max(x_points).astype(int) + search_radius)
        # # for col in range(search_pos_min, search_pos_max):
        # row = p_1(col)
        # # if row < box_length:
        # continue
        # if row > np.shape(zoomed_tip)[0] - box_length:
        # continue
        # # col_idx_min = max(np.round(col - box_length).astype(int), 0)
        # col_idx_max = min(np.round(col + box_length).astype(int), np.shape(zoomed_tip)[1] - 1)
        # row_idx_min = max(np.round(row - box_length).astype(int), 0)
        # row_idx_max = min(np.round(row + box_length).astype(int), np.shape(zoomed_tip)[0] - 1)
        # # if row_idx_max <= row_idx_min or col_idx_max <= col_idx_min:
        # continue
        # # if not foundPoint:
        # foundPoint = True
        # baseline = np.min(grayscale_tip[row_idx_min:row_idx_max, col_idx_min:col_idx_max])
        # look_for_dark_pixels = baseline > 240
        # # if foundPoint and look_for_dark_pixels:
        # if np.min(grayscale_tip[row_idx_min:row_idx_max, col_idx_min:col_idx_max]) < 200:
        # tip_location = np.array([col, row])
        # break
        # elif foundPoint and not look_for_dark_pixels:
        # if np.min(grayscale_tip[row_idx_min:row_idx_max, col_idx_min:col_idx_max]) > 200:
        # tip_location = np.array([col, row])
        # break

        fig, axs = plt.subplots(2, 2, figsize=(12, 10))

        axs[0, 0].imshow(cropped_image)
        axs[0, 0].set_title('Cropped image')

        axs[0, 1].imshow(binary_image)
        skel_ys, skel_xs = np.where(skel == 1)
        axs[0, 1].scatter(skel_xs, skel_ys, s=2, c='cyan', alpha=0.7)
        axs[0, 1].set_title('thresholded image')

        axs[1, 0].imshow(binary_image[crop_y_min:crop_y_max + 1, crop_x_min:crop_x_max + 1])
        skel_in_zoom = (skel[crop_y_min:crop_y_max + 1, crop_x_min:crop_x_max + 1] == 1)
        zys, zxs = np.where(skel_in_zoom)
        axs[1, 0].scatter(zxs, zys, s=3, c='cyan', alpha=0.8)
        axs[1, 0].scatter([tip_column - crop_x_min], [tip_row - crop_y_min])
        axs[1, 0].set_title('Identified coarse tip')

        axs[1, 1].imshow(grayscale_tip, cmap='gray')
        axs[1, 1].scatter(zxs, zys, s=3, c='cyan', alpha=0.8)

        if len(tip_path) >= 2:
            path_y = np.array([p[0] - crop_y_min for p in tip_path], dtype=float)
            path_x = np.array([p[1] - crop_x_min for p in tip_path], dtype=float)
            valid = (
                (path_y >= 0) & (path_y < grayscale_tip.shape[0]) &
                (path_x >= 0) & (path_x < grayscale_tip.shape[1])
            )
            if np.any(valid):
                axs[1, 1].plot(path_x[valid], path_y[valid], '-', color='yellow', linewidth=2)

        axs[1, 1].scatter([tip_column - crop_x_min], [tip_row - crop_y_min])
        if cnn_tip_abs is not None:
            axs[1, 1].scatter(
                [cnn_tip_abs[1] - crop_x_min_img - crop_x_min],
                [cnn_tip_abs[0] - crop_y_min_img - crop_y_min],
                c="magenta",
                marker="x",
                s=80,
            )

        # if p_1 is not None and x_points.size > 0:
        # axs[1, 1].scatter(np.unique(x_points), p_1(np.unique(x_points)))
        # if p_2 is not None and y_points.size > 0:
        # axs[1, 1].scatter(p_2(np.unique(y_points)), np.unique(y_points))
        # axs[1, 1].scatter(tip_location[0], tip_location[1])
        # axs[1, 1].legend(["coarse tip", "fit 1", "fit 2", "fine tip"])
        # axs[1, 1].set_title('Identified fine tip')

        tip_locations_array_fine[i, :] = np.array(
            [
                yy_selected + crop_y_min_img,
                xx_selected + crop_x_min_img,
                float(orientation),
                float(ss_pos),
                float(ntnl_pos),
                float(tip_angle_deg),
                motion_phase_code,
                pass_idx,
            ]
        )

        if cnn_tip_abs is not None:
            self._last_tip_locations_cnn = np.array(
                [
                    cnn_tip_abs[0],
                    cnn_tip_abs[1],
                    float(orientation),
                    float(ss_pos),
                    float(ntnl_pos),
                    float(tip_angle_deg),
                    motion_phase_code,
                    pass_idx,
                ],
                dtype=float,
            )
        else:
            self._last_tip_locations_cnn = None

        dbg_local = dict(tip_select_dbg) if isinstance(tip_select_dbg, dict) else {}
        dbg_local["image_file_name"] = image_file_name
        dbg_local["tip_angle_deg"] = float(tip_angle_deg)
        dbg_local["coarse_tip_before_local_xy"] = [float(tip_x), float(tip_y)]
        dbg_local["coarse_tip_after_local_xy"] = [float(xx_refined), float(yy_refined)]
        dbg_local["legacy_tip_xy"] = [float(tip_col_legacy), float(tip_row_legacy)]
        dbg_local["crop_origin_xy"] = [int(crop_x_min_img), int(crop_y_min_img)]
        if cnn_tip_dbg is not None:
            dbg_local["cnn_tip_refiner"] = cnn_tip_dbg
            dbg_local["cnn_tip_abs_yx"] = None if cnn_tip_abs is None else [float(cnn_tip_abs[0]), float(cnn_tip_abs[1])]
            dbg_local["cnn_anchor_name"] = self.tip_refiner_anchor_name
        self.tip_refine_debug_records[image_file_name] = dbg_local

        _remap_zoom_axes_to_crop_coordinates(axs, zoom_x_min, zoom_x_max, zoom_y_min, zoom_y_max)
        selected_tip_source = str(dbg_local.get("selected_tip_source", self.tip_refine_mode))
        annotate_tip_geometry_on_axes(axs, dbg_local, title_suffix=f" ({selected_tip_source})")

        calibrated_tip_axes = None
        if (
            self.board_homography_mm_from_px is not None
            or (
                self.true_vertical_img_unit is not None
                and self.board_mm_per_px_local is not None
            )
        ):
            try:
                calibrated_tip_axes = self.pixel_point_to_calibrated_axes(
                    x_px=float(tip_locations_array_fine[i, 1]),
                    y_px=float(tip_locations_array_fine[i, 0]),
                )
            except Exception:
                calibrated_tip_axes = None

        if calibrated_tip_axes is not None:
            u_mm, z_mm = calibrated_tip_axes
            axs[1, 1].set_title(
                f'Identified selected tip [{selected_tip_source}] (angle={tip_angle_deg:.2f} deg)\n'
                f'Calibrated: u={u_mm:.2f} mm, z={z_mm:.2f} mm'
            )
        else:
            axs[1, 1].set_title(
                f'Identified selected tip [{selected_tip_source}] (angle={tip_angle_deg:.2f} deg)'
            )

        return fig, axs, tip_locations_array_coarse[i, :], tip_locations_array_fine[i, :]

    def analyze_data_batch(self, crop_width_min=None, crop_width_max=None, crop_height_min=None, crop_height_max=None, threshold=200, ):
        """ Analyzes the data for the calibrations. Analyzes all images in batch. """
        current_dir = os.getcwd()
        print(f"Current working directory: {current_dir}")

        if current_dir.endswith("raw_image_data_folder"):
            self.calibration_data_folder = os.path.dirname(current_dir)
            print(f"Detected we're in raw_image_data_folder, using parent: {self.calibration_data_folder}")

        print(f"Looking for calibration folder: {self.calibration_data_folder}")

        if not hasattr(self, 'calibration_data_folder') or self.calibration_data_folder is None:
            print("Error: No calibration data folder specified.")
            return

        if not os.path.exists(self.calibration_data_folder):
            print(f"Calibration folder not found at: {self.calibration_data_folder}")
            search_dirs = [current_dir]
            if not current_dir.endswith("raw_image_data_folder"):
                search_dirs.append(os.path.dirname(current_dir))

            potential_folders = []
            for search_dir in search_dirs:
                print(f"Searching in: {search_dir}")
                try:
                    for item in os.listdir(search_dir):
                        item_path = os.path.join(search_dir, item)
                        if os.path.isdir(item_path):
                            raw_folder = os.path.join(item_path, "raw_image_data_folder")
                            if os.path.exists(raw_folder):
                                potential_folders.append(item_path)
                                print(f" Found potential calibration folder: {item_path}")
                except OSError as e:
                    print(f" Could not search {search_dir}: {e}")

            if potential_folders:
                potential_folders.sort(key=lambda x: os.path.getmtime(x), reverse=True)
                self.calibration_data_folder = potential_folders[0]
                print(f"Using calibration folder: {self.calibration_data_folder}")
            else:
                print("Error: No valid calibration folder found.")
                print("Make sure you have run calibrate() first to generate the required folder structure.")
                return

        raw_data_folder = os.path.join(self.calibration_data_folder, "raw_image_data_folder")
        print(f"Looking for raw images in: {raw_data_folder}")

        if not os.path.exists(raw_data_folder):
            print(f"Error: Raw image folder not found at: {raw_data_folder}")
            return

        if crop_width_min is None:
            crop_width_min = self.analysis_crop["crop_width_min"]
        if crop_width_max is None:
            crop_width_max = self.analysis_crop["crop_width_max"]
        if crop_height_min is None:
            crop_height_min = self.analysis_crop["crop_height_min"]
        if crop_height_max is None:
            crop_height_max = self.analysis_crop["crop_height_max"]

        print("Analysis crop bounds:")
        print(f" crop_width_min={crop_width_min}, crop_width_max={crop_width_max}")
        print(f" crop_height_min={crop_height_min}, crop_height_max={crop_height_max}")

        try:
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
            image_files = []
            for root, _, files in os.walk(raw_data_folder):
                for file_name in files:
                    if file_name.lower().endswith(image_extensions):
                        image_files.append(os.path.relpath(os.path.join(root, file_name), raw_data_folder))
            image_files = sorted(image_files)

            if len(image_files) == 0:
                print(f"Error: No image files found in {raw_data_folder}")
                return

            print(f"Found {len(image_files)} image files to process")
        except OSError as e:
            print(f"Error reading raw data folder: {e}")
            return

        processed_folder = os.path.join(self.calibration_data_folder, "processed_image_data_folder")
        analysis_output_folder = os.path.join(processed_folder, "analysis_outputs")

        try:
            if not os.path.exists(processed_folder):
                os.makedirs(processed_folder)
                print(f"Created processed data folder: {processed_folder}")
            else:
                print(f"Using existing processed data folder: {processed_folder}")

            if not os.path.exists(analysis_output_folder):
                os.makedirs(analysis_output_folder)
                print(f"Created analysis output folder: {analysis_output_folder}")
            else:
                print(f"Using existing analysis output folder: {analysis_output_folder}")
        except OSError as e:
            print(f"Error creating processed folder: {e}")
            return

        num_images = len(image_files)
        self.tip_locations_array_coarse = np.zeros((num_images, 8))
        self.tip_locations_array_selected = np.zeros((num_images, 8))
        self.tip_locations_array_cnn = np.full((num_images, 8), np.nan)

        successful_analyses = 0
        failed_analyses = 0

        for i, image_file in enumerate(image_files):
            print(f"\nProcessing image {i + 1}/{num_images}: {image_file}")
            try:
                fig, axs, coarse_tip, selected_tip = self.analyze_data(
                    image_file,
                    crop_width_min,
                    crop_width_max,
                    crop_height_min,
                    crop_height_max,
                    threshold
                )

                self.tip_locations_array_coarse[i, :] = coarse_tip
                self.tip_locations_array_selected[i, :] = selected_tip
                if self.tip_refiner_enabled and self._last_tip_locations_cnn is not None:
                    self.tip_locations_array_cnn[i, :] = self._last_tip_locations_cnn
                    if self.tip_refiner_use_as_selected:
                        self.tip_locations_array_selected[i, :] = self._last_tip_locations_cnn

                output_filename = f"{self._safe_output_stem(image_file)}_analysis.png"
                output_path = os.path.join(analysis_output_folder, output_filename)
                self._apply_dark_theme_to_figure(fig)
                fig.savefig(output_path, dpi=150, bbox_inches='tight', transparent=True, facecolor='none')
                plt.close(fig)

                successful_analyses += 1
                print(f" ✓ Successfully processed and saved to analysis_outputs/{output_filename}")

            except Exception as e:
                print(f" ✗ Error processing {image_file}: {e}")
                failed_analyses += 1
                self.tip_locations_array_coarse[i, :] = np.nan
                self.tip_locations_array_selected[i, :] = np.nan

        try:
            coarse_file = os.path.join(processed_folder, "tip_locations_coarse.npy")
            selected_file = os.path.join(processed_folder, "tip_locations_selected.npy")
            np.save(coarse_file, self.tip_locations_array_coarse)
            np.save(selected_file, self.tip_locations_array_selected)
            print(f"\n✓ Saved coarse tip locations to: {coarse_file}")
            print(f"✓ Saved selected tip locations to: {selected_file}")
            if self.tip_refiner_enabled:
                cnn_file = os.path.join(processed_folder, "tip_locations_cnn.npy")
                np.save(cnn_file, self.tip_locations_array_cnn)
                print(f"✓ Saved CNN tip locations to: {cnn_file}")

            try:
                columns = ['tip_row', 'tip_column', 'orientation', 'ss_pos', 'ntnl_pos', 'tip_angle_deg', 'motion_phase_code', 'pass_idx']
                df_coarse = pd.DataFrame(self.tip_locations_array_coarse, columns=columns)
                df_coarse['image_file'] = image_files
                coarse_csv = os.path.join(processed_folder, "tip_locations_coarse.csv")
                df_coarse.to_csv(coarse_csv, index=False)
                df_selected = pd.DataFrame(self.tip_locations_array_selected, columns=columns)
                df_selected['image_file'] = image_files
                selected_csv = os.path.join(processed_folder, "tip_locations_selected.csv")
                df_selected.to_csv(selected_csv, index=False)
                print(f"✓ Saved CSV file: {coarse_csv}")
                print(f"✓ Saved CSV file: {selected_csv}")
                if self.tip_refiner_enabled:
                    df_cnn = pd.DataFrame(self.tip_locations_array_cnn, columns=columns)
                    df_cnn['image_file'] = image_files
                    cnn_csv = os.path.join(processed_folder, "tip_locations_cnn.csv")
                    df_cnn.to_csv(cnn_csv, index=False)
                    print(f"✓ Saved CSV file: {cnn_csv}")
            except Exception as e:
                print(f"Warning: Could not save CSV files: {e}")

        except Exception as e:
            print(f"Error saving results: {e}")

        print(f"\n" + "=" * 50)
        print(f"BATCH ANALYSIS COMPLETE")
        print(f"=" * 50)
        print(f"Total images processed: {num_images}")
        print(f"Successful analyses: {successful_analyses}")
        print(f"Failed analyses: {failed_analyses}")
        print(f"Success rate: {successful_analyses / num_images * 100:.1f}%")
        print(f"Results saved in: {processed_folder}")

        return successful_analyses, failed_analyses

    def append_numbers_to_file_names(self, target_name, directory):
        file_list = [x.split('.')[0] for x in os.listdir(directory)]
        num_list = []
        for x in file_list:
            if x.startswith(target_name):
                try:
                    num_list.append(int(x.split(' ')[1][1:len(x.split(' ')[1]) - 1]))
                except:
                    num_list.append(0)
        num_file = max(num_list) + 1
        return f"{target_name} ({num_file})"

    @staticmethod
    def _polyval(coeffs, b):
        return np.polyval(np.asarray(coeffs, dtype=float), np.asarray(b, dtype=float))

    @staticmethod
    def _prepare_fit_samples(x_vals, y_vals):
        x_vals = np.asarray(x_vals, dtype=float).ravel()
        y_vals = np.asarray(y_vals, dtype=float).ravel()
        valid = np.isfinite(x_vals) & np.isfinite(y_vals)
        x_use = x_vals[valid]
        y_use = y_vals[valid]
        if x_use.size == 0:
            return x_use, y_use

        order = np.argsort(x_use)
        x_use = x_use[order]
        y_use = y_use[order]

        unique_x, inverse = np.unique(x_use, return_inverse=True)
        if unique_x.size != x_use.size:
            y_sum = np.zeros(unique_x.shape, dtype=float)
            counts = np.zeros(unique_x.shape, dtype=float)
            np.add.at(y_sum, inverse, y_use)
            np.add.at(counts, inverse, 1.0)
            x_use = unique_x
            y_use = y_sum / np.maximum(counts, 1.0)

        return x_use, y_use

    @staticmethod
    def _enforce_phase_endpoint_continuity(datasets, value_keys=None, atol=1e-9):
        """
        Force all phase branches to share the same endpoint knot values on the
        overlapping B-domain. This gives exact C0 continuity for interpolating
        fits such as PCHIP when traversing the hysteresis loop.
        """
        if not isinstance(datasets, dict) or len(datasets) < 2:
            return datasets

        phase_names = list(datasets.keys())
        shared_b = None
        for phase_name in phase_names:
            ds = datasets.get(phase_name)
            if ds is None or ds.get("common_b") is None:
                return datasets
            b_vals = np.asarray(ds["common_b"], dtype=float).ravel()
            if b_vals.size == 0:
                return datasets
            shared_b = b_vals if shared_b is None else np.intersect1d(shared_b, b_vals)
            if shared_b.size == 0:
                return datasets

        endpoint_b = [float(shared_b[0])]
        if shared_b.size > 1 and not np.isclose(shared_b[-1], shared_b[0], atol=atol):
            endpoint_b.append(float(shared_b[-1]))

        coord_keys = ("r_coords", "z_coords", "x_raw", "z_raw")
        angle_keys = ("tip_angle",)
        keys_to_use = tuple(value_keys) if value_keys is not None else coord_keys + angle_keys

        for b_target in endpoint_b:
            phase_indices = {}
            for phase_name in phase_names:
                b_vals = np.asarray(datasets[phase_name]["common_b"], dtype=float).ravel()
                match = np.where(np.isclose(b_vals, b_target, atol=atol))[0]
                if match.size == 0:
                    phase_indices = {}
                    break
                phase_indices[phase_name] = int(match[0])

            if len(phase_indices) != len(phase_names):
                continue

            for key in keys_to_use:
                samples = []
                for phase_name, idx in phase_indices.items():
                    arr = datasets[phase_name].get(key)
                    if arr is None:
                        continue
                    arr = np.asarray(arr, dtype=float).ravel()
                    if idx >= arr.size:
                        continue
                    val = float(arr[idx])
                    if np.isfinite(val):
                        samples.append(val)

                if not samples:
                    continue

                shared_val = float(np.mean(samples))
                for phase_name, idx in phase_indices.items():
                    arr = datasets[phase_name].get(key)
                    if arr is None:
                        continue
                    arr = np.asarray(arr, dtype=float).copy().ravel()
                    if idx < arr.size:
                        arr[idx] = shared_val
                        datasets[phase_name][key] = arr

        return datasets

    @staticmethod
    def _curve_fit_label(model_descriptor):
        if model_descriptor is None:
            return "Unknown Fit"
        model_type = str(model_descriptor.get("model_type", "")).lower()
        if model_type == "pchip":
            return "PCHIP Fit"
        degree = model_descriptor.get("degree")
        if degree is None:
            return "Polynomial Fit"
        if int(degree) == 3:
            return "Cubic Fit"
        return f"Degree-{int(degree)} Polynomial Fit"

    @classmethod
    def _evaluate_curve_model(cls, model_descriptor, b_vals):
        if model_descriptor is None:
            return None

        b_arr = np.asarray(b_vals, dtype=float)
        model_type = str(model_descriptor.get("model_type", "polynomial")).lower()

        if model_type == "polynomial":
            coeffs = model_descriptor.get("coefficients")
            if coeffs is None:
                raise ValueError("Polynomial fit descriptor missing coefficients.")
            return cls._polyval(coeffs, b_arr)

        if model_type == "pchip":
            x_knots = model_descriptor.get("x_knots")
            y_knots = model_descriptor.get("y_knots")
            if x_knots is None or y_knots is None:
                raise ValueError("PCHIP fit descriptor missing knots.")
            interpolator = PchipInterpolator(
                np.asarray(x_knots, dtype=float),
                np.asarray(y_knots, dtype=float),
                extrapolate=True,
            )
            return interpolator(b_arr)

        raise ValueError(f"Unsupported curve model_type: {model_type}")

    @classmethod
    def _fit_curve_model(cls, x_vals, y_vals, model_kind, value_name, min_points=None):
        model_kind = str(model_kind).strip().lower()
        x_use, y_use = cls._prepare_fit_samples(x_vals, y_vals)

        if model_kind == "cubic":
            min_required = 4 if min_points is None else max(4, int(min_points))
            if x_use.size < min_required:
                return None
            coeffs = np.polyfit(x_use, y_use, 3)
            return {
                "model_type": "polynomial",
                "basis": "monomial",
                "degree": 3,
                "input_axis": "b_motor",
                "value_name": str(value_name),
                "coefficients": coeffs.tolist(),
                "equation": cls._format_polynomial_equation(coeffs, str(value_name)),
                "sample_count": int(x_use.size),
                "fit_x_range": [float(np.min(x_use)), float(np.max(x_use))],
            }

        if model_kind == "linear":
            min_required = 2 if min_points is None else max(2, int(min_points))
            if x_use.size < min_required:
                return None
            coeffs = np.polyfit(x_use, y_use, 1)
            return {
                "model_type": "polynomial",
                "basis": "monomial",
                "degree": 1,
                "input_axis": "b_motor",
                "value_name": str(value_name),
                "coefficients": coeffs.tolist(),
                "equation": cls._format_polynomial_equation(coeffs, str(value_name)),
                "sample_count": int(x_use.size),
                "fit_x_range": [float(np.min(x_use)), float(np.max(x_use))],
            }

        if model_kind == "pchip":
            min_required = 2 if min_points is None else max(2, int(min_points))
            if x_use.size < min_required:
                return None
            return {
                "model_type": "pchip",
                "input_axis": "b_motor",
                "value_name": str(value_name),
                "x_knots": x_use.tolist(),
                "y_knots": y_use.tolist(),
                "equation": f"{value_name}(b) = PCHIP interpolation through {int(x_use.size)} calibration knots",
                "sample_count": int(x_use.size),
                "fit_x_range": [float(np.min(x_use)), float(np.max(x_use))],
            }

        raise ValueError(f"Unsupported fit model: {model_kind}")

    @staticmethod
    def _format_polynomial_equation(coeffs, var_name):
        coeffs = np.asarray(coeffs, dtype=float).ravel()
        degree = coeffs.size - 1
        pieces = [f"{var_name} ="]
        for idx, coef in enumerate(coeffs):
            power = degree - idx
            sign = "+" if coef >= 0 else "-"
            mag = abs(float(coef))
            if idx == 0:
                sign = "" if coef >= 0 else "-"
            term = f"{mag:.6f}"
            if power >= 1:
                term += "*b"
                if power >= 2:
                    term += f"^{power}"
            pieces.append(f"{sign} {term}".strip())
        return " ".join(pieces)

    @classmethod
    def _build_polynomial_model_descriptor(cls, coeffs, value_name, fit_x_range=None, sample_count=None):
        coeffs = np.asarray(coeffs, dtype=float).ravel()
        return {
            "model_type": "polynomial",
            "basis": "monomial",
            "degree": int(coeffs.size - 1),
            "input_axis": "b_motor",
            "value_name": str(value_name),
            "coefficients": coeffs.tolist(),
            "equation": cls._format_polynomial_equation(coeffs, str(value_name)),
            "sample_count": None if sample_count is None else int(sample_count),
            "fit_x_range": None if fit_x_range is None else [float(fit_x_range[0]), float(fit_x_range[1])],
        }

    @classmethod
    def _average_polynomial_models(cls, model_descriptors, value_name):
        valid_models = [
            m for m in model_descriptors
            if m is not None and str(m.get("model_type", "")).lower() == "polynomial"
        ]
        if not valid_models:
            return None

        coeff_arrays = [np.asarray(m.get("coefficients", []), dtype=float).ravel() for m in valid_models]
        coeff_len = coeff_arrays[0].size
        if coeff_len == 0 or any(arr.size != coeff_len for arr in coeff_arrays):
            raise ValueError(f"Cannot average incompatible polynomial models for {value_name}.")

        avg_coeffs = np.mean(np.vstack(coeff_arrays), axis=0)
        fit_ranges = [m.get("fit_x_range") for m in valid_models if m.get("fit_x_range") is not None]
        fit_x_range = None
        if fit_ranges:
            fit_x_range = (
                float(np.min([fr[0] for fr in fit_ranges])),
                float(np.max([fr[1] for fr in fit_ranges])),
            )
        sample_count = sum(int(m.get("sample_count") or 0) for m in valid_models)
        return cls._build_polynomial_model_descriptor(
            avg_coeffs,
            value_name=value_name,
            fit_x_range=fit_x_range,
            sample_count=sample_count if sample_count > 0 else None,
        )

    def get_pass_dataset(self, phase, pass_idx=None, combined=False):
        datasets = getattr(self, "_postprocessed_datasets", None)
        if not isinstance(datasets, dict) or not datasets:
            raise ValueError("No postprocessed datasets are available. Run postprocess_calibration_data() first.")
        phase_key = str(phase).strip().lower()
        if combined:
            lookup_key = f"{phase_key}_combined"
        elif pass_idx is not None:
            lookup_key = f"{phase_key}_{int(pass_idx)}"
        else:
            lookup_key = phase_key
        if lookup_key not in datasets:
            raise KeyError(f"Unknown dataset '{lookup_key}'. Available: {list(datasets.keys())}")
        return datasets[lookup_key]

    def get_fit_model(self, phase, fit_family="pchip", pass_idx=None, combined=False):
        fit_models_by_phase = getattr(self, "_postprocessed_fit_models_by_phase", None)
        if not isinstance(fit_models_by_phase, dict) or not fit_models_by_phase:
            raise ValueError("No postprocessed fit models are available. Run postprocess_calibration_data() first.")
        dataset = self.get_pass_dataset(phase=phase, pass_idx=pass_idx, combined=combined)
        dataset_key = str(dataset.get("phase_name", phase)).strip().lower()
        fit_family = str(fit_family).strip().lower()
        if fit_family not in {"pchip", "linear", "cubic"}:
            raise ValueError("fit_family must be 'pchip', 'linear', or 'cubic'.")
        key_map = {
            "pchip": {"r": "r", "z": "z", "tip_angle": "tip_angle", "offplane_y": "offplane_y_pchip"},
            "linear": {"r": "r", "z": "z", "tip_angle": "tip_angle", "offplane_y": "offplane_y_linear"},
            "cubic": {"r": "r_cubic", "z": "z_cubic", "tip_angle": "tip_angle_cubic", "offplane_y": "offplane_y_cubic"},
        }
        if dataset_key not in fit_models_by_phase:
            raise KeyError(f"Unknown fit model group '{dataset_key}'. Available: {list(fit_models_by_phase.keys())}")
        return {
            quantity: (
                fit_models_by_phase[dataset_key].get(model_key)
                or (fit_models_by_phase[dataset_key].get("offplane_y") if quantity == "offplane_y" else None)
            )
            for quantity, model_key in key_map[fit_family].items()
        }

    @staticmethod
    def _apply_dark_theme_to_figure(fig):
        fig.patch.set_facecolor('none')
        fig.patch.set_alpha(0.0)

        for ax in fig.axes:
            ax.set_facecolor('none')
            ax.title.set_color('white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            ax.tick_params(colors='white')
            for spine in ax.spines.values():
                spine.set_color((1.0, 1.0, 1.0, 0.7))
            ax.grid(True, color=(1.0, 1.0, 1.0, 0.18))

            legend = ax.get_legend()
            if legend is not None:
                legend.get_frame().set_facecolor((0.05, 0.05, 0.05, 0.55))
                legend.get_frame().set_edgecolor((1.0, 1.0, 1.0, 0.25))
                for text in legend.get_texts():
                    text.set_color('white')

    @classmethod
    def _legacy_curve_model_from_calibration_json(cls, cal_json, curve_key):
        coeffs = cal_json.get("cubic_coefficients", {})
        key_map = {
            "r": ("r_coeffs", "r_equation"),
            "z": ("z_coeffs", "z_equation"),
            "tip_angle": ("tip_angle_coeffs", "tip_angle_equation"),
            "offplane_y": ("offplane_y_coeffs", "offplane_y_equation"),
        }
        coeff_key, equation_key = key_map[curve_key]
        curve_coeffs = coeffs.get(coeff_key)
        if curve_coeffs is None:
            return None
        degree = len(curve_coeffs) - 1
        return {
            "model_type": "polynomial",
            "basis": "monomial",
            "degree": int(degree),
            "input_axis": "b_motor",
            "value_name": curve_key,
            "coefficients": curve_coeffs,
            "equation": coeffs.get(equation_key),
        }

    @classmethod
    def _sample_curve_xyz_from_calibration_json(cls, cal_json, b_vals):
        fit_models = cal_json.get("fit_models", {})
        r_model = fit_models.get("r") or cls._legacy_curve_model_from_calibration_json(cal_json, "r")
        z_model = fit_models.get("z") or cls._legacy_curve_model_from_calibration_json(cal_json, "z")
        y_model = fit_models.get("offplane_y") or cls._legacy_curve_model_from_calibration_json(cal_json, "offplane_y")

        if r_model is None or z_model is None:
            raise ValueError("Missing r/z fit models in calibration JSON.")

        x = cls._evaluate_curve_model(r_model, b_vals)
        z = cls._evaluate_curve_model(z_model, b_vals)
        if y_model is not None:
            y = cls._evaluate_curve_model(y_model, b_vals)
        else:
            y = np.zeros_like(x, dtype=float)

        return np.column_stack([x, y, z]).astype(float)

    @staticmethod
    def _compute_link_frames(points_xyz):
        """
        For link i between P[i] -> P[i+1]:
        - origin at P[i]
        - z-axis along the link direction
        - x/y axes are chosen to be stable with a world-up fallback
        """
        pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
        frames = []
        world_up = np.array([0.0, 0.0, 1.0], dtype=float)

        for i in range(pts.shape[0] - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            d = p1 - p0
            length_mm = float(np.linalg.norm(d))

            if length_mm < 1e-9:
                frames.append({
                    "origin_mm": p0.tolist(),
                    "direction_unit": [0.0, 0.0, 1.0],
                    "length_mm": 0.0,
                    "R_world_from_link": np.eye(3).tolist(),
                })
                continue

            z_axis = d / length_mm
            up = world_up
            if abs(float(np.dot(up, z_axis))) > 0.95:
                up = np.array([0.0, 1.0, 0.0], dtype=float)

            x_axis = np.cross(up, z_axis)
            x_norm = float(np.linalg.norm(x_axis))
            if x_norm < 1e-12:
                x_axis = np.array([1.0, 0.0, 0.0], dtype=float)
            else:
                x_axis = x_axis / x_norm

            y_axis = np.cross(z_axis, x_axis)
            y_norm = float(np.linalg.norm(y_axis))
            if y_norm < 1e-12:
                y_axis = np.array([0.0, 1.0, 0.0], dtype=float)
            else:
                y_axis = y_axis / y_norm

            R = np.column_stack([x_axis, y_axis, z_axis])
            frames.append({
                "origin_mm": p0.tolist(),
                "direction_unit": z_axis.tolist(),
                "length_mm": length_mm,
                "R_world_from_link": R.tolist(),
            })

        return frames

    @staticmethod
    def _unit_vector(v):
        n = float(np.linalg.norm(v))
        if n < 1e-12:
            return np.zeros_like(v)
        return v / n

    @classmethod
    def _build_orthonormal_basis(cls, direction):
        d = cls._unit_vector(np.asarray(direction, dtype=float))
        if np.linalg.norm(d) < 1e-12:
            return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])

        a = np.array([0.0, 0.0, 1.0], dtype=float)
        if abs(float(np.dot(d, a))) > 0.9:
            a = np.array([0.0, 1.0, 0.0], dtype=float)

        u = cls._unit_vector(np.cross(d, a))
        v = cls._unit_vector(np.cross(d, u))
        return u, v

    @classmethod
    def polyline_to_cylinder_mesh(cls, points_xyz, radius_mm=1.5, sides=16):
        pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
        if pts.shape[0] < 2:
            raise ValueError("Need at least two points for a polyline mesh.")

        verts = []
        faces = []

        for i in range(pts.shape[0] - 1):
            p0 = pts[i]
            p1 = pts[i + 1]
            d = p1 - p0
            if float(np.linalg.norm(d)) < 1e-9:
                continue

            u, v = cls._build_orthonormal_basis(d)

            ring0 = []
            ring1 = []
            for k in range(int(sides)):
                theta = 2.0 * np.pi * (k / float(sides))
                offset = float(radius_mm) * (np.cos(theta) * u + np.sin(theta) * v)
                ring0.append(p0 + offset)
                ring1.append(p1 + offset)

            base_idx = len(verts)
            verts.extend(ring0)
            verts.extend(ring1)

            for k in range(int(sides)):
                k2 = (k + 1) % int(sides)
                a0 = base_idx + k
                b0 = base_idx + k2
                a1 = base_idx + int(sides) + k
                b1 = base_idx + int(sides) + k2

                faces.append([a0, a1, b1])
                faces.append([a0, b1, b0])

        return np.asarray(verts, dtype=np.float32), np.asarray(faces, dtype=np.int32)

    @classmethod
    def write_binary_stl(cls, path, vertices, faces, solid_name="robot_skeleton"):
        path = Path(path)
        vertices = np.asarray(vertices, dtype=np.float32)
        faces = np.asarray(faces, dtype=np.int32)

        header = bytearray(80)
        name_bytes = str(solid_name).encode("ascii", errors="ignore")[:80]
        header[:len(name_bytes)] = name_bytes

        tri_count = faces.shape[0]
        with open(path, "wb") as f:
            f.write(header)
            f.write(np.uint32(tri_count).tobytes())
            for tri in faces:
                p0 = vertices[tri[0]]
                p1 = vertices[tri[1]]
                p2 = vertices[tri[2]]
                normal = np.cross(p1 - p0, p2 - p0)
                normal = cls._unit_vector(normal).astype(np.float32)
                f.write(normal.tobytes())
                f.write(p0.astype(np.float32).tobytes())
                f.write(p1.astype(np.float32).tobytes())
                f.write(p2.astype(np.float32).tobytes())
                f.write(np.uint16(0).tobytes())

    def export_parametric_skeleton_model(self, robot_name, n_links=6, diameter_mm=3.0, b_ref=0.0, stl_reference_pose=True):
        """
        Export the fitted calibration curve as a parametric link-chain model plus predictor.

        Outputs in processed_image_data_folder:
        - <robot_name>_skeleton_parametric.json
        - <robot_name>_skeleton_predict.py
        - optionally <robot_name>_robot_skeleton_reference.stl
        """
        processed = Path(self.calibration_data_folder) / "processed_image_data_folder"
        gcode_json_path = processed / f"{robot_name}_gcode_calibration.json"
        if not gcode_json_path.is_file():
            raise FileNotFoundError(f"Missing calibration JSON: {gcode_json_path}")

        with open(gcode_json_path, "r") as f:
            cal_json = json.load(f)

        fit_models = cal_json.get("fit_models", {})
        r_model = fit_models.get("r") or self._legacy_curve_model_from_calibration_json(cal_json, "r")
        z_model = fit_models.get("z") or self._legacy_curve_model_from_calibration_json(cal_json, "z")
        y_model = fit_models.get("offplane_y") or self._legacy_curve_model_from_calibration_json(cal_json, "offplane_y")
        if r_model is None or z_model is None:
            raise ValueError("Calibration JSON missing r/z fit models.")

        motor_setup = cal_json.get("motor_setup", {})
        coeffs = cal_json.get("cubic_coefficients", {})
        b_rng = motor_setup.get("b_motor_position_range") or coeffs.get("b_motor_range")
        if b_rng is None or len(b_rng) != 2:
            raise ValueError("Calibration JSON missing b_motor_position_range.")

        b_min = float(b_rng[0])
        b_max = float(b_rng[1])
        n_links = int(max(1, n_links))
        diameter_mm = float(diameter_mm)
        t = np.linspace(0.0, 1.0, n_links + 1)
        b_knots = b_min + t * (b_max - b_min)

        pts_ref = self._sample_curve_xyz_from_calibration_json(cal_json, b_knots)

        origin = np.array([0.0, 0.0, 0.0], dtype=float)
        if np.linalg.norm(pts_ref[-1] - origin) > 1e-9:
            pts_ref_with_origin = np.vstack([pts_ref, origin])
        else:
            pts_ref_with_origin = pts_ref.copy()

        frames_ref = self._compute_link_frames(pts_ref_with_origin)

        skel_param = {
            "robot_name": robot_name,
            "type": "parametric_link_chain",
            "units": "mm",
            "diameter_mm": diameter_mm,
            "radius_mm": diameter_mm / 2.0,
            "n_links_curve": n_links,
            "unknown_tail_to_origin": True,
            "b_range": [b_min, b_max],
            "reference_b_pull_mm": float(b_ref),
            "knot_definition": {
                "t_i": t.tolist(),
                "b_i": b_knots.tolist(),
                "meaning": "Curve knots are sampled along the full calibrated B range; each knot is a point on the fitted curve.",
            },
            "curve_equations": {
                "x_mm": {"fit_model": r_model, "definition": "x = r(b) signed planar radial deflection"},
                "z_mm": {"fit_model": z_model, "definition": "z = z(b) axial position"},
                "y_mm": {
                    "fit_model": y_model,
                    "definition": "y = y_offplane(b) if available else 0",
                    "available": y_model is not None,
                },
            },
            "reference_pose": {
                "points_xyz_mm": pts_ref_with_origin.round(6).tolist(),
                "link_frames": frames_ref,
                "note": "Reference pose uses knots sampled across B-range and a final link to origin. Use predictor to get poses for arbitrary b.",
            },
            "predictor_convention": {
                "inputs": ["b_pull_mm"],
                "outputs": [
                    "points_xyz_mm (N+2 points including origin)",
                    "link_frames (per-link origin, direction, length, rotation matrix)",
                ],
                "base_rule": "Append final segment to (0,0,0) if last point isn't already origin.",
            },
        }

        param_path = processed / f"{robot_name}_skeleton_parametric.json"
        with open(param_path, "w") as f:
            json.dump(skel_param, f, indent=2)

        py_path = processed / f"{robot_name}_skeleton_predict.py"
        predictor_code = f"""# Auto-generated parametric skeleton predictor for {robot_name}
# Units: mm
import numpy as np
from scipy.interpolate import PchipInterpolator

R_MODEL = {json.dumps(r_model)}
Z_MODEL = {json.dumps(z_model)}
Y_MODEL = {json.dumps(y_model)}

B_MIN = {b_min:.10f}
B_MAX = {b_max:.10f}

T_KNOTS = np.array({json.dumps(t.tolist())}, dtype=float)
B_KNOTS = B_MIN + T_KNOTS * (B_MAX - B_MIN)

DIAMETER_MM = {diameter_mm:.6f}
RADIUS_MM = {diameter_mm / 2.0:.6f}

def _polyval(c, b):
    if c is None:
        return None
    return np.polyval(np.asarray(c, dtype=float), np.asarray(b, dtype=float))

def _evaluate_curve_model(model_descriptor, b):
    if model_descriptor is None:
        return None
    b_arr = np.asarray(b, dtype=float)
    model_type = str(model_descriptor.get("model_type", "polynomial")).lower()
    if model_type == "polynomial":
        coeffs = model_descriptor.get("coefficients")
        if coeffs is None:
            raise ValueError("Polynomial fit model missing coefficients.")
        return _polyval(coeffs, b_arr)
    if model_type == "pchip":
        x_knots = model_descriptor.get("x_knots")
        y_knots = model_descriptor.get("y_knots")
        if x_knots is None or y_knots is None:
            raise ValueError("PCHIP fit model missing knots.")
        return PchipInterpolator(np.asarray(x_knots, dtype=float), np.asarray(y_knots, dtype=float), extrapolate=True)(b_arr)
    raise ValueError(f"Unsupported model_type: {{model_type}}")

def curve_point_xyz(b):
    x = float(_evaluate_curve_model(R_MODEL, b))
    z = float(_evaluate_curve_model(Z_MODEL, b))
    if Y_MODEL is None:
        y = 0.0
    else:
        y = float(_evaluate_curve_model(Y_MODEL, b))
    return np.array([x, y, z], dtype=float)

def skeleton_points(b_pull_mm, include_origin=True):
    pts = np.stack([curve_point_xyz(bi) for bi in B_KNOTS], axis=0)
    if include_origin:
        if np.linalg.norm(pts[-1] - np.array([0.0, 0.0, 0.0])) > 1e-9:
            pts = np.vstack([pts, np.array([0.0, 0.0, 0.0])])
    return pts

def _compute_link_frames(points_xyz):
    pts = np.asarray(points_xyz, dtype=float).reshape(-1, 3)
    frames = []
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    for i in range(pts.shape[0] - 1):
        p0 = pts[i]
        p1 = pts[i+1]
        d = p1 - p0
        L = float(np.linalg.norm(d))
        if L < 1e-9:
            frames.append({{
                "origin_mm": p0.tolist(),
                "direction_unit": [0.0, 0.0, 1.0],
                "length_mm": 0.0,
                "R_world_from_link": np.eye(3).tolist(),
            }})
            continue
        z = d / L
        up = world_up
        if abs(float(np.dot(up, z))) > 0.95:
            up = np.array([0.0, 1.0, 0.0], dtype=float)
        x = np.cross(up, z)
        xn = float(np.linalg.norm(x))
        if xn < 1e-12:
            x = np.array([1.0, 0.0, 0.0], dtype=float)
        else:
            x /= xn
        y = np.cross(z, x)
        yn = float(np.linalg.norm(y))
        if yn < 1e-12:
            y = np.array([0.0, 1.0, 0.0], dtype=float)
        else:
            y /= yn
        R = np.column_stack([x, y, z])
        frames.append({{
            "origin_mm": p0.tolist(),
            "direction_unit": z.tolist(),
            "length_mm": L,
            "R_world_from_link": R.tolist(),
        }})
    return frames

def skeleton_link_frames(b_pull_mm, include_origin=True):
    pts = skeleton_points(b_pull_mm, include_origin=include_origin)
    return _compute_link_frames(pts)
"""
        with open(py_path, "w") as f:
            f.write(predictor_code)

        stl_path = None
        if stl_reference_pose:
            verts, faces = self.polyline_to_cylinder_mesh(
                pts_ref_with_origin,
                radius_mm=diameter_mm / 2.0,
                sides=16,
            )
            stl_path = processed / f"{robot_name}_robot_skeleton_reference.stl"
            self.write_binary_stl(
                stl_path,
                verts,
                faces,
                solid_name=f"{robot_name}_skeleton_ref",
            )

        cal_json.setdefault("exported_models", {})
        cal_json["exported_models"]["robot_skeleton_parametric"] = {
            "format": "json+py",
            "parametric_json": param_path.name,
            "predictor_py": py_path.name,
            "reference_stl": stl_path.name if stl_path is not None else None,
            "diameter_mm": diameter_mm,
            "n_links": n_links,
            "note": "Use predictor to generate link endpoints/frames for any B pull; includes final link to origin for unknown parts.",
        }
        with open(gcode_json_path, "w") as f:
            json.dump(cal_json, f, indent=2)

        return param_path, py_path, stl_path, gcode_json_path

    def postprocess_calibration_data(self, width_in_pixels=3025, width_in_mm=140, robot_name="calibrated_robot", save_plots=True, export_skeleton=False, skeleton_diameter_mm=3.0, skeleton_links=6, skeleton_reference_stl=False, fit_model="cubic", offplane_fit_model="linear"):
        """Postprocess calibration data and export separate pull/release fits."""
        if not hasattr(self, 'tip_locations_array_selected'):
            print("Tip location data not found in memory, attempting to load from saved files...")
            processed_folder = os.path.join(self.calibration_data_folder, "processed_image_data_folder")
            selected_file = os.path.join(processed_folder, "tip_locations_selected.npy")
            coarse_file = os.path.join(processed_folder, "tip_locations_coarse.npy")
            if os.path.exists(selected_file):
                self.tip_locations_array_selected = np.load(selected_file)
                print(f"✓ Loaded selected tip data from {selected_file}")
                print(f" Selected data shape: {self.tip_locations_array_selected.shape}")
            elif os.path.exists(coarse_file):
                self.tip_locations_array_selected = np.load(coarse_file)
                print(f"✓ Loaded fallback tip data from {coarse_file}")
                print(f" Fallback data shape: {self.tip_locations_array_selected.shape}")
            else:
                print("Error: Required .npy files not found. Run analyze_data_batch() first.")
                return None

        if not hasattr(self, 'tip_locations_array_coarse'):
            processed_folder = os.path.join(self.calibration_data_folder, "processed_image_data_folder")
            coarse_file = os.path.join(processed_folder, "tip_locations_coarse.npy")
            if os.path.exists(coarse_file):
                self.tip_locations_array_coarse = np.load(coarse_file)

        processed_folder = os.path.join(self.calibration_data_folder, "processed_image_data_folder")
        os.makedirs(processed_folder, exist_ok=True)
        original_dir = os.getcwd()
        os.chdir(processed_folder)

        try:
            print("\n" + "=" * 50)
            print("Starting calibration data postprocessing...")

            arr = np.asarray(self.tip_locations_array_selected, dtype=float)
            if arr.ndim != 2 or arr.shape[1] < 5:
                raise ValueError("tip_locations_selected must have at least 5 columns.")

            has_tip_angle = arr.shape[1] >= 6
            has_phase = arr.shape[1] >= 7
            has_pass_idx = arr.shape[1] >= 8
            if not has_phase:
                arr = np.column_stack([arr, np.zeros((arr.shape[0],), dtype=float)])
                print("Motion phase column not found; assuming all images are pull-phase.")
            if not has_pass_idx:
                arr = np.column_stack([arr, np.ones((arr.shape[0],), dtype=float)])
                print("Pass index column not found; assuming PASS1 for all images.")

            ORI_COL = 2
            B_PULL_COL = 3
            ANGLE_COL = 5
            PHASE_COL = 6
            PASS_COL = 7

            has_calibrated_axes = (
                self.board_homography_mm_from_px is not None
                or (self.true_vertical_img_unit is not None and self.board_mm_per_px_local is not None)
            )
            board_type = self._normalize_board_type(
                None if self.board_pose is None else self.board_pose.get("board_type", None)
            )
            if board_type is None and isinstance(self.camera_calib_meta, dict):
                board_type = self._normalize_board_type(self.camera_calib_meta.get("board_type"))

            conversion_mode = "legacy_linear_scale"
            pixel_to_mm_scale = float(width_in_mm) / float(width_in_pixels)
            # Board-reference corrections are already applied at the calibration
            # layer. Do not rescale the calibrated mm coordinates again before
            # fitting, or all exported mm-valued fits end up 2x too small.
            board_measurement_scale = 1.0
            tip_mm = arr.copy()

            if has_calibrated_axes:
                conversion_mode = "board_reference_calibrated"
                if self.board_mm_per_px_local is not None:
                    pixel_to_mm_scale = float(self.board_mm_per_px_local) * board_measurement_scale
                if self.board_pose is not None and "origin_px" in self.board_pose:
                    origin_px = np.asarray(self.board_pose["origin_px"], dtype=float).reshape(2)
                else:
                    origin_px = None
                u_mm = np.zeros((tip_mm.shape[0],), dtype=float)
                z_mm = np.zeros((tip_mm.shape[0],), dtype=float)
                for idx, row in enumerate(arr):
                    uu, zz = self.pixel_point_to_calibrated_axes(
                        x_px=float(row[1]), y_px=float(row[0]), origin_px=origin_px
                    )
                    u_mm[idx] = uu * board_measurement_scale
                    z_mm[idx] = zz * board_measurement_scale
            elif self.ruler_mm_per_px is not None:
                conversion_mode = "ruler_reference_scale"
                pixel_to_mm_scale = float(self.ruler_mm_per_px)
                u_mm = arr[:, 1].astype(float) * pixel_to_mm_scale
                z_mm = -arr[:, 0].astype(float) * pixel_to_mm_scale
            else:
                u_mm = arr[:, 1].astype(float) / float(width_in_pixels) * float(width_in_mm)
                z_mm = -arr[:, 0].astype(float) / float(width_in_pixels) * float(width_in_mm)

            tip_mm[:, 0] = u_mm
            tip_mm[:, 1] = z_mm
            pd.DataFrame(tip_mm).to_excel('tip_locations_selected_mm.xlsx', index=False, header=False)
            print(f"Physical conversion mode: {conversion_mode} | representative mm/px={pixel_to_mm_scale:.6f}")

            valid_rows = tip_mm[np.all(np.isfinite(tip_mm[:, [0, 1, ORI_COL, B_PULL_COL, PHASE_COL, PASS_COL]]), axis=1)].copy()
            if valid_rows.size == 0:
                raise ValueError("No valid calibration rows after filtering NaNs.")
            print("Applying orientation processing (two-capture averaging) for rotation/runout compensation...")

            def circular_mean_deg(a):
                a = np.asarray(a, dtype=float)
                a = a[np.isfinite(a)]
                if a.size == 0:
                    return float('nan')
                rad = np.deg2rad(a)
                return _normalize_tip_angle_deg(
                    float(np.rad2deg(np.arctan2(np.mean(np.sin(rad)), np.mean(np.cos(rad)))))
                )

            def collapse_by_bpull(arr_phase):
                if arr_phase.size == 0:
                    return arr_phase
                uniq = np.unique(arr_phase[:, B_PULL_COL])
                out = []
                for bu in np.sort(uniq):
                    m = np.isclose(arr_phase[:, B_PULL_COL], bu, atol=1e-12)
                    row = arr_phase[m]
                    r = np.nanmean(row, axis=0)
                    r[B_PULL_COL] = bu
                    if has_tip_angle and row.shape[1] > ANGLE_COL:
                        r[ANGLE_COL] = circular_mean_deg(row[:, ANGLE_COL])
                    out.append(r)
                return np.vstack(out) if out else np.zeros((0, arr_phase.shape[1]))

            def process_orientation_pair(
                arr_a,
                arr_b,
                *,
                primary_col=0,
                secondary_col=1,
                angle_col=None,
                primary_mirror_sign=-1.0,
                angle_mirror_sign=-1.0,
                pair_name="orientation_pair",
            ):
                """
                Run the two-capture orientation-processing step used for machine
                rotation/runout compensation:
                1. collapse each orientation by B position
                2. align them on the common B grid
                3. mirror one orientation into the other's frame
                4. average the mirrored pair
                """
                arr_a = collapse_by_bpull(arr_a)
                arr_b = collapse_by_bpull(arr_b)
                if arr_a.size == 0 or arr_b.size == 0:
                    return None

                common_b = np.intersect1d(arr_a[:, B_PULL_COL], arr_b[:, B_PULL_COL])
                if common_b.size == 0:
                    return None

                idx_a = np.searchsorted(arr_a[:, B_PULL_COL], common_b)
                idx_b = np.searchsorted(arr_b[:, B_PULL_COL], common_b)
                arr_ap = arr_a[idx_a].copy()
                arr_bp = arr_b[idx_b].copy()

                mirror_line = float(np.mean(np.vstack([arr_a[:, primary_col], arr_b[:, primary_col]])))
                arr_b_primary_mirrored = mirror_line + (
                    primary_mirror_sign * (arr_bp[:, primary_col] - mirror_line)
                )
                avg_primary = 0.5 * (arr_ap[:, primary_col] + arr_b_primary_mirrored)
                avg_secondary = 0.5 * (arr_ap[:, secondary_col] + arr_bp[:, secondary_col])

                avg_angle = None
                arr_b_angle_mirrored = None
                if (
                    angle_col is not None
                    and arr_ap.shape[1] > angle_col
                    and arr_bp.shape[1] > angle_col
                ):
                    arr_b_angle_mirrored = angle_mirror_sign * arr_bp[:, angle_col]
                    avg_angle = np.array(
                        [
                            circular_mean_deg([a0, a1])
                            for a0, a1 in zip(arr_ap[:, angle_col], arr_b_angle_mirrored)
                        ],
                        dtype=float,
                    )

                return {
                    'pair_name': pair_name,
                    'common_b': common_b.astype(float),
                    'mirror_line': mirror_line,
                    'primary_mirror_sign': float(primary_mirror_sign),
                    'a_collapsed': arr_a,
                    'b_collapsed': arr_b,
                    'a_aligned': arr_ap,
                    'b_aligned': arr_bp,
                    'b_primary_mirrored': arr_b_primary_mirrored.astype(float),
                    'avg_primary': avg_primary.astype(float),
                    'avg_secondary': avg_secondary.astype(float),
                    'b_angle_mirrored': None if arr_b_angle_mirrored is None else arr_b_angle_mirrored.astype(float),
                    'avg_angle': None if avg_angle is None else avg_angle.astype(float),
                }

            phase_name_map = {0: "pull", 1: "release"}

            def build_phase_dataset(phase_code, pass_idx=1):
                phase_rows = valid_rows[
                    np.isclose(valid_rows[:, PHASE_COL], phase_code)
                    & np.isclose(valid_rows[:, PASS_COL], pass_idx)
                ]
                if phase_rows.size == 0:
                    return None
                planar_pair = process_orientation_pair(
                    phase_rows[np.isclose(phase_rows[:, ORI_COL], 0)],
                    phase_rows[np.isclose(phase_rows[:, ORI_COL], 1)],
                    primary_col=0,
                    secondary_col=1,
                    angle_col=ANGLE_COL if has_tip_angle else None,
                    primary_mirror_sign=-1.0,
                    angle_mirror_sign=-1.0,
                    pair_name='C0_C180',
                )
                if planar_pair is None:
                    return None
                o0 = planar_pair['a_collapsed']
                o1 = planar_pair['b_collapsed']
                o0p = planar_pair['a_aligned']
                o1p = planar_pair['b_aligned']
                common_b = planar_pair['common_b']
                x_ref_mirror_line = float(planar_pair['mirror_line'])
                o1_x_mirrored = planar_pair['b_primary_mirrored']
                avg_x_raw = planar_pair['avg_primary']
                avg_z_raw = planar_pair['avg_secondary']
                avg_ang = planar_pair['avg_angle']
                if avg_ang is None:
                    avg_ang = np.full((common_b.shape[0],), np.nan, dtype=float)
                zero_mask = np.isclose(common_b, 0.0, atol=1e-9)
                zero_idx = int(np.where(zero_mask)[0][0]) if np.any(zero_mask) else int(np.argmin(np.abs(common_b)))
                x0_ref = float(x_ref_mirror_line)
                z0_ref = float(avg_z_raw[zero_idx])
                x_avg = avg_x_raw - x0_ref
                z_avg = avg_z_raw - z0_ref
                tip_angle_avg = avg_ang.copy()
                finite_mask = np.isfinite(tip_angle_avg)
                if np.any(finite_mask):
                    tip_angle_avg[finite_mask] = _normalize_tip_angle_deg(tip_angle_avg[finite_mask])
                dataset = {
                    'phase_code': int(phase_code),
                    'pass_idx': int(pass_idx),
                    'phase_base_name': phase_name_map.get(int(phase_code), f"phase_{int(phase_code)}"),
                    'phase_name': f"{phase_name_map.get(int(phase_code), f'phase_{int(phase_code)}')}_{int(pass_idx)}",
                    'common_b': common_b.astype(float),
                    'x_raw': avg_x_raw.astype(float),
                    'z_raw': avg_z_raw.astype(float),
                    'r_coords': x_avg.astype(float),
                    'z_coords': z_avg.astype(float),
                    'tip_angle': tip_angle_avg.astype(float),
                    'zero_idx': zero_idx,
                    'x_ref_mirror_line_mm': x_ref_mirror_line,
                    'z0_ref_mm': z0_ref,
                    'x0_ref_mm': x0_ref,
                    'orientation_processing_planar': {
                        'pair_name': planar_pair['pair_name'],
                        'common_b': common_b.astype(float),
                        'mirror_line_mm': x_ref_mirror_line,
                        'orientation_0_x_mm': o0p[:, 0].astype(float),
                        'orientation_0_z_mm': o0p[:, 1].astype(float),
                        'orientation_180_x_mirrored_mm': o1_x_mirrored.astype(float),
                        'orientation_180_z_mm': o1p[:, 1].astype(float),
                        'avg_x_mm': avg_x_raw.astype(float),
                        'avg_z_mm': avg_z_raw.astype(float),
                    },
                    'raw_planar_trajectories': {
                        'C0': {
                            'b': o0[:, B_PULL_COL].astype(float).tolist(),
                            'x': o0[:, 0].astype(float).tolist(),
                            'z': o0[:, 1].astype(float).tolist(),
                        },
                        'C180': {
                            'b': o1[:, B_PULL_COL].astype(float).tolist(),
                            'x': o1[:, 0].astype(float).tolist(),
                            'z': o1[:, 1].astype(float).tolist(),
                        },
                    },
                    'mirrored_planar_overlay': {
                        'b': common_b.astype(float),
                        'C0_x': o0p[:, 0].astype(float),
                        'C0_z': o0p[:, 1].astype(float),
                        'C180_x_mirrored': o1_x_mirrored.astype(float),
                        'C180_z': o1p[:, 1].astype(float),
                        'avg_x': avg_x_raw.astype(float),
                        'avg_z': avg_z_raw.astype(float),
                    },
                }
                offplane_pair = process_orientation_pair(
                    phase_rows[np.isclose(phase_rows[:, ORI_COL], 2)],
                    phase_rows[np.isclose(phase_rows[:, ORI_COL], 3)],
                    primary_col=0,
                    secondary_col=1,
                    angle_col=None,
                    primary_mirror_sign=-1.0,
                    angle_mirror_sign=-1.0,
                    pair_name='C90_C-90',
                )
                if offplane_pair is not None:
                    o2 = offplane_pair['a_collapsed']
                    o3 = offplane_pair['b_collapsed']
                    o2p = offplane_pair['a_aligned']
                    o3p = offplane_pair['b_aligned']
                    common_b_off = offplane_pair['common_b']
                    y_ref_offplane = float(offplane_pair['mirror_line'])
                    o3_y_mirrored = offplane_pair['b_primary_mirrored']
                    y_off_raw = offplane_pair['avg_primary']
                    dataset['offplane_b'] = common_b_off.astype(float)
                    dataset['offplane_y'] = (y_off_raw - y_ref_offplane).astype(float)
                    dataset['y_ref_offplane_mirror_line_mm'] = y_ref_offplane
                    dataset['orientation_processing_offplane'] = {
                        'pair_name': offplane_pair['pair_name'],
                        'common_b': common_b_off.astype(float),
                        'mirror_line_mm': y_ref_offplane,
                        'orientation_pos90_y_mm': o2p[:, 0].astype(float),
                        'orientation_pos90_z_mm': o2p[:, 1].astype(float),
                        'orientation_neg90_y_mirrored_mm': o3_y_mirrored.astype(float),
                        'orientation_neg90_z_mm': o3p[:, 1].astype(float),
                        'avg_y_mm': y_off_raw.astype(float),
                    }
                    dataset['raw_offplane_trajectories'] = {
                        'C90': {
                            'b': o2[:, B_PULL_COL].astype(float).tolist(),
                            'y_proxy': o2[:, 0].astype(float).tolist(),
                            'z': o2[:, 1].astype(float).tolist(),
                        },
                        'C-90': {
                            'b': o3[:, B_PULL_COL].astype(float).tolist(),
                            'y_proxy': o3[:, 0].astype(float).tolist(),
                            'z': o3[:, 1].astype(float).tolist(),
                        },
                    }
                else:
                    dataset['offplane_b'] = None
                    dataset['offplane_y'] = None
                    dataset['y_ref_offplane_mirror_line_mm'] = None
                    dataset['orientation_processing_offplane'] = None
                    dataset['raw_offplane_trajectories'] = None
                return dataset

            datasets = {}
            available_phase_codes = sorted(np.unique(valid_rows[:, PHASE_COL]).astype(int))
            available_pass_indices = sorted(np.unique(valid_rows[:, PASS_COL]).astype(int))
            for phase_code in available_phase_codes:
                for pass_idx in available_pass_indices:
                    ds = build_phase_dataset(phase_code, pass_idx=pass_idx)
                    if ds is not None:
                        datasets[ds['phase_name']] = ds

            if not datasets:
                raise ValueError("Could not construct any valid pull/release paired datasets.")

            combine_redundant_passes = bool(getattr(self, "_combine_redundant_passes", True))
            redundant_pass_combination = str(getattr(self, "_redundant_pass_combination", "average")).strip().lower()
            if redundant_pass_combination not in {"average", "keep_separate"}:
                redundant_pass_combination = "average"

            def clone_dataset_with_updates(base_ds, updates):
                merged = dict(base_ds)
                merged.update(updates)
                return merged

            def combine_phase_pair(base_phase_name):
                ds_1 = datasets.get(f"{base_phase_name}_1")
                ds_2 = datasets.get(f"{base_phase_name}_2")
                if ds_1 is None or ds_2 is None:
                    return None
                common_b = np.intersect1d(np.asarray(ds_1['common_b'], dtype=float), np.asarray(ds_2['common_b'], dtype=float))
                if common_b.size == 0:
                    return None
                idx_1 = np.searchsorted(np.asarray(ds_1['common_b'], dtype=float), common_b)
                idx_2 = np.searchsorted(np.asarray(ds_2['common_b'], dtype=float), common_b)

                def avg_or_none(key, use_common_b=True):
                    a = ds_1.get(key)
                    b = ds_2.get(key)
                    if a is None or b is None:
                        return None
                    a = np.asarray(a, dtype=float)
                    b = np.asarray(b, dtype=float)
                    if use_common_b:
                        a = a[idx_1]
                        b = b[idx_2]
                    if key == 'tip_angle':
                        return np.array([circular_mean_deg([va, vb]) for va, vb in zip(a, b)], dtype=float)
                    return 0.5 * (a + b)

                combined = clone_dataset_with_updates(
                    ds_1,
                    {
                        'phase_base_name': base_phase_name,
                        'phase_name': f"{base_phase_name}_combined",
                        'pass_idx': None,
                        'combined_from_passes': [1, 2],
                        'common_b': common_b.astype(float),
                        'x_raw': avg_or_none('x_raw'),
                        'z_raw': avg_or_none('z_raw'),
                        'r_coords': avg_or_none('r_coords'),
                        'z_coords': avg_or_none('z_coords'),
                        'tip_angle': avg_or_none('tip_angle'),
                        'offplane_b': None,
                        'offplane_y': None,
                    },
                )
                zero_mask = np.isclose(common_b, 0.0, atol=1e-9)
                combined['zero_idx'] = int(np.where(zero_mask)[0][0]) if np.any(zero_mask) else int(np.argmin(np.abs(common_b)))
                combined['x_ref_mirror_line_mm'] = float(np.nanmean([ds_1.get('x_ref_mirror_line_mm'), ds_2.get('x_ref_mirror_line_mm')]))
                combined['z0_ref_mm'] = float(np.nanmean([ds_1.get('z0_ref_mm'), ds_2.get('z0_ref_mm')]))
                combined['x0_ref_mm'] = float(np.nanmean([ds_1.get('x0_ref_mm'), ds_2.get('x0_ref_mm')]))
                if ds_1.get('offplane_b') is not None and ds_2.get('offplane_b') is not None:
                    common_b_off = np.intersect1d(np.asarray(ds_1['offplane_b'], dtype=float), np.asarray(ds_2['offplane_b'], dtype=float))
                    if common_b_off.size > 0:
                        id1 = np.searchsorted(np.asarray(ds_1['offplane_b'], dtype=float), common_b_off)
                        id2 = np.searchsorted(np.asarray(ds_2['offplane_b'], dtype=float), common_b_off)
                        combined['offplane_b'] = common_b_off.astype(float)
                        combined['offplane_y'] = 0.5 * (
                            np.asarray(ds_1['offplane_y'], dtype=float)[id1]
                            + np.asarray(ds_2['offplane_y'], dtype=float)[id2]
                        )
                combined['orientation_processing_planar'] = None
                combined['orientation_processing_offplane'] = None
                combined['raw_planar_trajectories'] = {
                    'source_passes': [ds_1.get('phase_name'), ds_2.get('phase_name')]
                }
                combined['raw_offplane_trajectories'] = {
                    'source_passes': [ds_1.get('phase_name'), ds_2.get('phase_name')]
                } if combined.get('offplane_b') is not None else None
                combined['mirrored_planar_overlay'] = None
                return combined

            def compute_redundancy_diagnostics(base_phase_name):
                ds_1 = datasets.get(f"{base_phase_name}_1")
                ds_2 = datasets.get(f"{base_phase_name}_2")
                if ds_1 is None or ds_2 is None:
                    return None
                common_b = np.intersect1d(np.asarray(ds_1['common_b'], dtype=float), np.asarray(ds_2['common_b'], dtype=float))
                if common_b.size == 0:
                    return None
                idx_1 = np.searchsorted(np.asarray(ds_1['common_b'], dtype=float), common_b)
                idx_2 = np.searchsorted(np.asarray(ds_2['common_b'], dtype=float), common_b)

                def diff_for(key):
                    a = ds_1.get(key)
                    b = ds_2.get(key)
                    if a is None or b is None:
                        return None
                    return np.asarray(a, dtype=float)[idx_1] - np.asarray(b, dtype=float)[idx_2]

                diffs = {
                    'b': common_b.astype(float),
                    'x': None if diff_for('x_raw') is None else diff_for('x_raw').astype(float).tolist(),
                    'z': None if diff_for('z_raw') is None else diff_for('z_raw').astype(float).tolist(),
                    'r_coords': None if diff_for('r_coords') is None else diff_for('r_coords').astype(float).tolist(),
                    'z_coords': None if diff_for('z_coords') is None else diff_for('z_coords').astype(float).tolist(),
                    'tip_angle': None if diff_for('tip_angle') is None else diff_for('tip_angle').astype(float).tolist(),
                }
                if ds_1.get('offplane_b') is not None and ds_2.get('offplane_b') is not None:
                    common_off_b = np.intersect1d(np.asarray(ds_1['offplane_b'], dtype=float), np.asarray(ds_2['offplane_b'], dtype=float))
                    if common_off_b.size > 0:
                        id1 = np.searchsorted(np.asarray(ds_1['offplane_b'], dtype=float), common_off_b)
                        id2 = np.searchsorted(np.asarray(ds_2['offplane_b'], dtype=float), common_off_b)
                        diffs['offplane_b'] = common_off_b.astype(float).tolist()
                        diffs['offplane_y'] = (
                            np.asarray(ds_1['offplane_y'], dtype=float)[id1]
                            - np.asarray(ds_2['offplane_y'], dtype=float)[id2]
                        ).astype(float).tolist()

                rms_terms = []
                for key in ('r_coords', 'z_coords', 'tip_angle'):
                    d = diff_for(key)
                    if d is not None and d.size > 0:
                        rms_terms.append(np.square(d))
                rms_mismatch = None
                if rms_terms:
                    rms_mismatch = float(np.sqrt(np.mean(np.concatenate([x.ravel() for x in rms_terms]))))
                return {
                    'phase': base_phase_name,
                    'shared_b': common_b.astype(float).tolist(),
                    'differences': diffs,
                    'rms_mismatch': rms_mismatch,
                }

            redundancy_diagnostics = {}
            if 'pull_2' in datasets:
                redundancy_diagnostics['pull'] = compute_redundancy_diagnostics('pull')
            if 'release_2' in datasets:
                redundancy_diagnostics['release'] = compute_redundancy_diagnostics('release')

            combined_datasets = {}
            if combine_redundant_passes and redundant_pass_combination == "average":
                if 'pull_2' in datasets:
                    print("Combining redundant pull passes by averaging")
                    combined_ds = combine_phase_pair('pull')
                    if combined_ds is not None:
                        combined_datasets[combined_ds['phase_name']] = combined_ds
                if 'release_2' in datasets:
                    print("Combining redundant release passes by averaging")
                    combined_ds = combine_phase_pair('release')
                    if combined_ds is not None:
                        combined_datasets[combined_ds['phase_name']] = combined_ds

            datasets_for_fitting = dict(datasets)
            datasets_for_fitting.update(combined_datasets)
            dataset_aliases = {}
            if 'pull_combined' in combined_datasets:
                dataset_aliases['pull'] = 'pull_combined'
            elif 'pull_1' in datasets:
                dataset_aliases['pull'] = 'pull_1'
            if 'release_combined' in combined_datasets:
                dataset_aliases['release'] = 'release_combined'
            elif 'release_1' in datasets:
                dataset_aliases['release'] = 'release_1'

            fit_model = str(fit_model).strip().lower()
            if fit_model not in {'cubic', 'pchip'}:
                raise ValueError(f"Unsupported fit_model '{fit_model}'. Use 'cubic' or 'pchip'.")
            offplane_fit_model = str(offplane_fit_model).strip().lower()
            if offplane_fit_model not in {'linear', 'cubic', 'pchip'}:
                raise ValueError(
                    f"Unsupported offplane_fit_model '{offplane_fit_model}'. Use 'linear', 'cubic', or 'pchip'."
                )

            if fit_model == 'pchip':
                datasets = self._enforce_phase_endpoint_continuity(
                    {k: v for k, v in datasets_for_fitting.items() if isinstance(v, dict) and v.get('common_b') is not None},
                    value_keys=('r_coords', 'z_coords', 'tip_angle'),
                )
                datasets_for_fitting.update(datasets)

            def r2_score_safe(y_true, y_pred):
                y_true = np.asarray(y_true, dtype=float)
                y_pred = np.asarray(y_pred, dtype=float)
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                if ss_tot == 0:
                    return 1.0 if ss_res == 0 else float('-inf')
                return float(1.0 - (ss_res / ss_tot))

            def pretty_model_name(model_descriptor):
                if model_descriptor is None:
                    return "Fit"
                model_type = str(model_descriptor.get('model_type', fit_model)).strip().lower()
                if model_type == 'pchip':
                    return 'PCHIP'
                if model_type == 'polynomial':
                    degree = int(model_descriptor.get('degree', -1))
                    if degree == 1:
                        return 'Linear'
                    if degree == 3:
                        return 'Cubic'
                    return f'Polynomial (deg {degree})'
                return model_type.upper()

            fit_results = {}
            legacy_phase = dataset_aliases.get('pull', list(datasets_for_fitting.keys())[0])

            for phase_name, ds in datasets_for_fitting.items():
                b = ds['common_b']
                r_coords = ds['r_coords']
                z_coords = ds['z_coords']
                tip_ang = ds['tip_angle']
                r_model = self._fit_curve_model(b, r_coords, fit_model, f"r_{phase_name}")
                z_model = self._fit_curve_model(b, z_coords, fit_model, f"z_{phase_name}")
                r_cubic_model = self._fit_curve_model(b, r_coords, 'cubic', f"r_{phase_name}_avg_cubic")
                z_cubic_model = self._fit_curve_model(b, z_coords, 'cubic', f"z_{phase_name}_avg_cubic")
                if r_model is None or z_model is None:
                    raise ValueError(f"Insufficient points for {phase_name} planar fit.")
                r_pred = self._evaluate_curve_model(r_model, b)
                z_pred = self._evaluate_curve_model(z_model, b)
                angle_model = None
                angle_pred = None
                angle_r2 = None
                angle_cubic_model = None
                if has_tip_angle and np.any(np.isfinite(tip_ang)):
                    valid_ang = np.isfinite(tip_ang)
                    if np.sum(valid_ang) >= 4:
                        angle_model = self._fit_curve_model(b[valid_ang], tip_ang[valid_ang], fit_model, f"tip_angle_{phase_name}")
                        if angle_model is not None:
                            angle_pred = self._evaluate_curve_model(angle_model, b[valid_ang])
                            angle_r2 = r2_score_safe(tip_ang[valid_ang], angle_pred)
                        angle_cubic_model = self._fit_curve_model(b[valid_ang], tip_ang[valid_ang], 'cubic', f"tip_angle_{phase_name}_avg_cubic")
                off_model = None
                off_pred = None
                off_r2 = None
                off_plot_b = None
                off_plot_pred = None
                off_linear_model = None
                off_cubic_model = None
                off_pchip_model = None
                if ds.get('offplane_b') is not None and ds.get('offplane_y') is not None:
                    off_b = ds['offplane_b']
                    off_y = ds['offplane_y']
                    if off_b.size >= 2:
                        off_linear_model = self._fit_curve_model(off_b, off_y, 'linear', f"offplane_y_{phase_name}_linear")
                        off_pchip_model = self._fit_curve_model(off_b, off_y, 'pchip', f"offplane_y_{phase_name}_pchip")
                    if off_b.size >= 4:
                        off_cubic_model = self._fit_curve_model(off_b, off_y, 'cubic', f"offplane_y_{phase_name}_cubic")
                    off_model = {
                        'linear': off_linear_model,
                        'cubic': off_cubic_model,
                        'pchip': off_pchip_model,
                    }.get(offplane_fit_model)
                    if off_model is None:
                        off_model = off_linear_model or off_cubic_model or off_pchip_model
                    if off_model is not None:
                        off_pred = self._evaluate_curve_model(off_model, off_b)
                        off_r2 = r2_score_safe(off_y, off_pred)
                        off_plot_b = ds['common_b']
                        off_plot_pred = self._evaluate_curve_model(off_model, off_plot_b)
                fit_results[phase_name] = {
                    'dataset': ds,
                    'r_model': r_model,
                    'z_model': z_model,
                    'r_pchip_model': self._fit_curve_model(b, r_coords, 'pchip', f"r_{phase_name}_pchip"),
                    'z_pchip_model': self._fit_curve_model(b, z_coords, 'pchip', f"z_{phase_name}_pchip"),
                    'r_cubic_model': r_cubic_model,
                    'z_cubic_model': z_cubic_model,
                    'tip_angle_model': angle_model,
                    'tip_angle_pchip_model': self._fit_curve_model(b[np.isfinite(tip_ang)], tip_ang[np.isfinite(tip_ang)], 'pchip', f"tip_angle_{phase_name}_pchip") if has_tip_angle and np.any(np.isfinite(tip_ang)) and np.sum(np.isfinite(tip_ang)) >= 2 else None,
                    'tip_angle_cubic_model': angle_cubic_model,
                    'offplane_y_model': off_model,
                    'offplane_y_linear_model': off_linear_model,
                    'offplane_y_cubic_model': off_cubic_model,
                    'offplane_y_pchip_model': off_pchip_model,
                    'r_pred': r_pred,
                    'z_pred': z_pred,
                    'tip_angle_pred': angle_pred,
                    'r_r2': r2_score_safe(r_coords, r_pred),
                    'z_r2': r2_score_safe(z_coords, z_pred),
                    'tip_angle_r2': angle_r2,
                    'offplane_y_r2': off_r2,
                    'offplane_plot_b': off_plot_b,
                    'offplane_plot_pred': off_plot_pred,
                }

            selected_phase_fit_results = {}
            for phase_alias in ('pull', 'release'):
                canonical_name = dataset_aliases.get(phase_alias, phase_alias)
                if canonical_name in fit_results:
                    selected_phase_fit_results[phase_alias] = fit_results[canonical_name]

            shared_tip_angle_model = self._average_polynomial_models(
                [fr['tip_angle_cubic_model'] for fr in selected_phase_fit_results.values()],
                value_name='tip_angle_avg',
            )
            shared_r_cubic_model = self._average_polynomial_models(
                [fr['r_cubic_model'] for fr in selected_phase_fit_results.values()],
                value_name='r_avg',
            )
            shared_z_cubic_model = self._average_polynomial_models(
                [fr['z_cubic_model'] for fr in selected_phase_fit_results.values()],
                value_name='z_avg',
            )
            shared_offplane_y_model = self._average_polynomial_models(
                [fr['offplane_y_model'] for fr in selected_phase_fit_results.values()],
                value_name='offplane_y_avg',
            )
            shared_offplane_y_linear_model = self._average_polynomial_models(
                [fr['offplane_y_linear_model'] for fr in selected_phase_fit_results.values()],
                value_name='offplane_y_avg_linear',
            )
            shared_offplane_y_cubic_model = self._average_polynomial_models(
                [fr['offplane_y_cubic_model'] for fr in selected_phase_fit_results.values()],
                value_name='offplane_y_avg_cubic',
            )

            def shared_pair_average_xy(x_key, y_key, circular=False):
                fr_pull = selected_phase_fit_results.get('pull')
                fr_release = selected_phase_fit_results.get('release')
                if fr_pull is None or fr_release is None:
                    return None, None

                ds_pull = fr_pull['dataset']
                ds_release = fr_release['dataset']
                x_pull = ds_pull.get(x_key)
                y_pull = ds_pull.get(y_key)
                x_release = ds_release.get(x_key)
                y_release = ds_release.get(y_key)
                if x_pull is None or y_pull is None or x_release is None or y_release is None:
                    return None, None

                x_pull = np.asarray(x_pull, dtype=float)
                y_pull = np.asarray(y_pull, dtype=float)
                x_release = np.asarray(x_release, dtype=float)
                y_release = np.asarray(y_release, dtype=float)
                common_x = np.intersect1d(x_pull, x_release)
                if common_x.size == 0:
                    return None, None

                idx_pull = np.searchsorted(x_pull, common_x)
                idx_release = np.searchsorted(x_release, common_x)
                a = y_pull[idx_pull]
                b = y_release[idx_release]
                valid = np.isfinite(a) & np.isfinite(b)
                if not np.any(valid):
                    return None, None

                common_x = common_x[valid]
                a = a[valid]
                b = b[valid]
                if circular:
                    y_avg = np.array([circular_mean_deg([va, vb]) for va, vb in zip(a, b)], dtype=float)
                else:
                    y_avg = 0.5 * (a + b)
                return common_x.astype(float), y_avg.astype(float)

            shared_r_cubic_r2 = None
            shared_z_cubic_r2 = None
            shared_tip_angle_r2 = None
            shared_offplane_y_r2 = None
            shared_offplane_y_linear_r2 = None
            shared_offplane_y_cubic_r2 = None

            if shared_r_cubic_model is not None:
                r_b_avg, r_y_avg = shared_pair_average_xy('common_b', 'r_coords', circular=False)
                if r_b_avg is not None:
                    shared_r_cubic_r2 = r2_score_safe(
                        r_y_avg,
                        self._evaluate_curve_model(shared_r_cubic_model, r_b_avg),
                    )
                else:
                    r_b_all = []
                    r_y_all = []
                    for fr in fit_results.values():
                        ds = fr['dataset']
                        r_b_all.append(np.asarray(ds['common_b'], dtype=float))
                        r_y_all.append(np.asarray(ds['r_coords'], dtype=float))
                    if r_b_all:
                        r_b_all = np.concatenate(r_b_all)
                        r_y_all = np.concatenate(r_y_all)
                        shared_r_cubic_r2 = r2_score_safe(
                            r_y_all,
                            self._evaluate_curve_model(shared_r_cubic_model, r_b_all),
                        )

            if shared_z_cubic_model is not None:
                z_b_avg, z_y_avg = shared_pair_average_xy('common_b', 'z_coords', circular=False)
                if z_b_avg is not None:
                    shared_z_cubic_r2 = r2_score_safe(
                        z_y_avg,
                        self._evaluate_curve_model(shared_z_cubic_model, z_b_avg),
                    )
                else:
                    z_b_all = []
                    z_y_all = []
                    for fr in fit_results.values():
                        ds = fr['dataset']
                        z_b_all.append(np.asarray(ds['common_b'], dtype=float))
                        z_y_all.append(np.asarray(ds['z_coords'], dtype=float))
                    if z_b_all:
                        z_b_all = np.concatenate(z_b_all)
                        z_y_all = np.concatenate(z_y_all)
                        shared_z_cubic_r2 = r2_score_safe(
                            z_y_all,
                            self._evaluate_curve_model(shared_z_cubic_model, z_b_all),
                        )

            if shared_tip_angle_model is not None:
                tip_b_avg, tip_y_avg = shared_pair_average_xy('common_b', 'tip_angle', circular=True)
                if tip_b_avg is not None:
                    shared_tip_angle_r2 = r2_score_safe(
                        tip_y_avg,
                        self._evaluate_curve_model(shared_tip_angle_model, tip_b_avg),
                    )
                else:
                    tip_b_all = []
                    tip_y_all = []
                    for fr in fit_results.values():
                        ds = fr['dataset']
                        valid_ang = np.isfinite(ds['tip_angle'])
                        if np.any(valid_ang):
                            tip_b_all.append(ds['common_b'][valid_ang])
                            tip_y_all.append(ds['tip_angle'][valid_ang])
                    if tip_b_all:
                        tip_b_all = np.concatenate(tip_b_all)
                        tip_y_all = np.concatenate(tip_y_all)
                        shared_tip_angle_r2 = r2_score_safe(
                            tip_y_all,
                            self._evaluate_curve_model(shared_tip_angle_model, tip_b_all),
                        )

            off_b_avg, off_y_avg = shared_pair_average_xy('offplane_b', 'offplane_y', circular=False)
            if shared_offplane_y_model is not None:
                if off_b_avg is not None:
                    shared_offplane_y_r2 = r2_score_safe(
                        off_y_avg,
                        self._evaluate_curve_model(shared_offplane_y_model, off_b_avg),
                    )
                else:
                    off_b_all = []
                    off_y_all = []
                    for fr in fit_results.values():
                        ds = fr['dataset']
                        if ds.get('offplane_b') is not None and ds.get('offplane_y') is not None:
                            off_b_all.append(np.asarray(ds['offplane_b'], dtype=float))
                            off_y_all.append(np.asarray(ds['offplane_y'], dtype=float))
                    if off_b_all:
                        off_b_all = np.concatenate(off_b_all)
                        off_y_all = np.concatenate(off_y_all)
                        shared_offplane_y_r2 = r2_score_safe(
                            off_y_all,
                            self._evaluate_curve_model(shared_offplane_y_model, off_b_all),
                        )
                        off_b_avg = off_b_all
                        off_y_avg = off_y_all
            if shared_offplane_y_linear_model is not None and off_b_avg is not None:
                shared_offplane_y_linear_r2 = r2_score_safe(
                    off_y_avg,
                    self._evaluate_curve_model(shared_offplane_y_linear_model, off_b_avg),
                )
            if shared_offplane_y_cubic_model is not None and off_b_avg is not None:
                shared_offplane_y_cubic_r2 = r2_score_safe(
                    off_y_avg,
                    self._evaluate_curve_model(shared_offplane_y_cubic_model, off_b_avg),
                )

            shared_offplane_plot_model = (
                shared_offplane_y_model
                or shared_offplane_y_cubic_model
                or shared_offplane_y_linear_model
            )
            shared_offplane_plot_r2 = (
                shared_offplane_y_r2
                if shared_offplane_y_model is not None
                else shared_offplane_y_cubic_r2
                if shared_offplane_y_cubic_model is not None
                else shared_offplane_y_linear_r2
            )

            def model_plot_b(model_descriptor, fallback_values, num_points=250):
                if model_descriptor is None:
                    return None
                fit_x_range = model_descriptor.get('fit_x_range')
                if fit_x_range is not None and len(fit_x_range) == 2:
                    lo, hi = float(fit_x_range[0]), float(fit_x_range[1])
                elif fallback_values:
                    lo, hi = float(min(fallback_values)), float(max(fallback_values))
                else:
                    return None
                if not np.isfinite(lo) or not np.isfinite(hi):
                    return None
                if hi < lo:
                    lo, hi = hi, lo
                if np.isclose(lo, hi):
                    return np.array([lo], dtype=float)
                return np.linspace(lo, hi, int(num_points))

            legacy_ds = fit_results[legacy_phase]['dataset']
            r_legacy_cubic_descriptor = self._fit_curve_model(legacy_ds['common_b'], legacy_ds['r_coords'], 'cubic', 'r')
            z_legacy_cubic_descriptor = self._fit_curve_model(legacy_ds['common_b'], legacy_ds['z_coords'], 'cubic', 'z')
            r_coefficients = np.asarray(r_legacy_cubic_descriptor['coefficients'], dtype=float)
            z_coefficients = np.asarray(z_legacy_cubic_descriptor['coefficients'], dtype=float)
            shared_r_cubic_coefficients = None
            shared_z_cubic_coefficients = None
            tip_angle_coefficients = None
            y_off_coefficients = None
            if shared_r_cubic_model is not None:
                shared_r_cubic_coefficients = np.asarray(shared_r_cubic_model['coefficients'], dtype=float)
            if shared_z_cubic_model is not None:
                shared_z_cubic_coefficients = np.asarray(shared_z_cubic_model['coefficients'], dtype=float)
            if shared_tip_angle_model is not None:
                tip_angle_coefficients = np.asarray(shared_tip_angle_model['coefficients'], dtype=float)
            if shared_offplane_y_model is not None:
                y_off_coefficients = np.asarray(shared_offplane_y_model['coefficients'], dtype=float)
            y_off_linear_coefficients = None
            y_off_cubic_coefficients = None
            if shared_offplane_y_linear_model is not None:
                y_off_linear_coefficients = np.asarray(shared_offplane_y_linear_model['coefficients'], dtype=float)
            if shared_offplane_y_cubic_model is not None:
                y_off_cubic_coefficients = np.asarray(shared_offplane_y_cubic_model['coefficients'], dtype=float)

            phase_metrics_rows = []
            for phase_name, fr in fit_results.items():
                phase_metrics_rows.extend([
                    {'Phase': phase_name, 'Channel': 'r', 'R2': fr['r_r2'], 'Equation': fr['r_model'].get('equation')},
                    {'Phase': phase_name, 'Channel': 'z', 'R2': fr['z_r2'], 'Equation': fr['z_model'].get('equation')},
                    {'Phase': phase_name, 'Channel': 'tip_angle', 'R2': fr['tip_angle_r2'], 'Equation': None if fr['tip_angle_model'] is None else fr['tip_angle_model'].get('equation')},
                    {'Phase': phase_name, 'Channel': 'offplane_y', 'R2': fr['offplane_y_r2'], 'Equation': None if fr['offplane_y_model'] is None else fr['offplane_y_model'].get('equation')},
                ])
            phase_metrics_rows.extend([
                {'Phase': 'averaged', 'Channel': 'r_cubic', 'R2': shared_r_cubic_r2, 'Equation': None if shared_r_cubic_model is None else shared_r_cubic_model.get('equation')},
                {'Phase': 'averaged', 'Channel': 'z_cubic', 'R2': shared_z_cubic_r2, 'Equation': None if shared_z_cubic_model is None else shared_z_cubic_model.get('equation')},
                {'Phase': 'averaged', 'Channel': 'tip_angle', 'R2': shared_tip_angle_r2, 'Equation': None if shared_tip_angle_model is None else shared_tip_angle_model.get('equation')},
                {'Phase': 'averaged', 'Channel': 'offplane_y', 'R2': shared_offplane_y_r2, 'Equation': None if shared_offplane_y_model is None else shared_offplane_y_model.get('equation')},
                {'Phase': 'averaged', 'Channel': 'offplane_y_linear', 'R2': shared_offplane_y_linear_r2, 'Equation': None if shared_offplane_y_linear_model is None else shared_offplane_y_linear_model.get('equation')},
                {'Phase': 'averaged', 'Channel': 'offplane_y_cubic', 'R2': shared_offplane_y_cubic_r2, 'Equation': None if shared_offplane_y_cubic_model is None else shared_offplane_y_cubic_model.get('equation')},
            ])
            pd.DataFrame(phase_metrics_rows).to_excel(f"{robot_name}_phase_fit_summary.xlsx", index=False)

            if save_plots:
                fig = plt.figure(figsize=(20, 15))
                gs = fig.add_gridspec(3, 3)
                ax_ref = fig.add_subplot(gs[0, 0])
                ax_planar = fig.add_subplot(gs[0, 1])
                ax_r_fit = fig.add_subplot(gs[0, 2])
                ax_z_fit = fig.add_subplot(gs[1, 0])
                ax_r_res = fig.add_subplot(gs[1, 1])
                ax_z_res = fig.add_subplot(gs[1, 2])
                ax_angle = fig.add_subplot(gs[2, 0])
                ax_off = fig.add_subplot(gs[2, 1])
                raw_subgs = gs[2, 2].subgridspec(1, 2, wspace=0.18)
                ax_raw_xz = fig.add_subplot(raw_subgs[0, 0])
                ax_raw_yz = fig.add_subplot(raw_subgs[0, 1])

                phase_colors = {
                    'pull': '#79c7ff',
                    'release': '#ffb38a',
                }
                phase_markers = {
                    'pull': 'o',
                    'release': 's',
                }
                avg_cubic_colors = {
                    'r': '#8ef0d2',
                    'z': '#c6f36b',
                    'angle': '#ff8fb1',
                    'offplane': '#9fd6ff',
                }

                all_b_ranges = []
                all_tip_b = []
                all_off_b = []
                all_off_y = []
                all_tip_y = []
                for phase_name, fr in fit_results.items():
                    ds = fr['dataset']
                    color = phase_colors.get(phase_name, None)
                    marker = phase_markers.get(phase_name, 'o')
                    all_b_ranges.extend(ds['common_b'].astype(float).tolist())

                    ax_ref.plot(
                        ds['x_raw'],
                        ds['z_raw'],
                        marker=marker,
                        linestyle='-',
                        color=color,
                        linewidth=1.8,
                        markersize=4,
                        label=f"{phase_name} referenced",
                    )
                    ax_planar.plot(
                        ds['r_coords'],
                        ds['z_coords'],
                        marker=marker,
                        linestyle='-',
                        color=color,
                        linewidth=1.8,
                        markersize=4,
                        label=f"{phase_name} measured",
                    )
                    ax_planar.plot(
                        fr['r_pred'],
                        fr['z_pred'],
                        '--',
                        color=color,
                        linewidth=2.0,
                        label=f"{phase_name} fit",
                    )
                    ax_r_fit.plot(
                        ds['common_b'],
                        ds['r_coords'],
                        marker=marker,
                        linestyle='None',
                        color=color,
                        markersize=5,
                        label=f"{phase_name} measured",
                    )
                    ax_r_fit.plot(
                        ds['common_b'],
                        fr['r_pred'],
                        color=color,
                        linewidth=2.0,
                        marker=marker,
                        markersize=3.5,
                        label=f"{phase_name} {pretty_model_name(fr['r_model'])} fit",
                    )
                    ax_z_fit.plot(
                        ds['common_b'],
                        ds['z_coords'],
                        marker=marker,
                        linestyle='None',
                        color=color,
                        markersize=5,
                        label=f"{phase_name} measured",
                    )
                    ax_z_fit.plot(
                        ds['common_b'],
                        fr['z_pred'],
                        color=color,
                        linewidth=2.0,
                        marker=marker,
                        markersize=3.5,
                        label=f"{phase_name} {pretty_model_name(fr['z_model'])} fit",
                    )
                    ax_r_res.plot(
                        ds['common_b'],
                        ds['r_coords'] - fr['r_pred'],
                        marker=marker,
                        color=color,
                        linewidth=1.6,
                        label=f"{phase_name} residuals",
                    )
                    ax_z_res.plot(
                        ds['common_b'],
                        ds['z_coords'] - fr['z_pred'],
                        marker=marker,
                        color=color,
                        linewidth=1.6,
                        label=f"{phase_name} residuals",
                    )

                    raw_planar = ds.get('raw_planar_trajectories') or {}
                    if 'C0' in raw_planar:
                        ax_raw_xz.plot(
                            raw_planar['C0']['x'],
                            raw_planar['C0']['z'],
                            marker='o',
                            linestyle='-',
                            linewidth=1.5,
                            markersize=3.5,
                            label=f"{phase_name} C=0",
                        )
                    if 'C180' in raw_planar:
                        ax_raw_xz.plot(
                            raw_planar['C180']['x'],
                            raw_planar['C180']['z'],
                            marker='o',
                            linestyle='-',
                            linewidth=1.5,
                            markersize=3.5,
                            label=f"{phase_name} C=180",
                        )

                    raw_offplane = ds.get('raw_offplane_trajectories') or {}
                    if 'C90' in raw_offplane:
                        ax_raw_yz.plot(
                            raw_offplane['C90']['y_proxy'],
                            raw_offplane['C90']['z'],
                            marker='o',
                            linestyle='-',
                            linewidth=1.5,
                            markersize=3.5,
                            label=f"{phase_name} C=+90",
                        )
                    if 'C-90' in raw_offplane:
                        ax_raw_yz.plot(
                            raw_offplane['C-90']['y_proxy'],
                            raw_offplane['C-90']['z'],
                            marker='o',
                            linestyle='-',
                            linewidth=1.5,
                            markersize=3.5,
                            label=f"{phase_name} C=-90",
                        )
                    if np.any(np.isfinite(ds['tip_angle'])):
                        valid_ang = np.isfinite(ds['tip_angle'])
                        all_tip_b.extend(ds['common_b'][valid_ang].astype(float).tolist())
                        all_tip_y.extend(ds['tip_angle'][valid_ang].astype(float).tolist())
                        ax_angle.plot(
                            ds['common_b'][valid_ang],
                            ds['tip_angle'][valid_ang],
                            marker=marker,
                            linestyle='None',
                            color=color,
                            markersize=5,
                            label=f"{phase_name} measured",
                        )
                        if fr['tip_angle_model'] is not None and fr.get('tip_angle_pred') is not None:
                            ax_angle.plot(
                                ds['common_b'][valid_ang],
                                fr['tip_angle_pred'],
                                color=color,
                                linewidth=2.0,
                                marker=marker,
                                markersize=3.5,
                                label=f"{phase_name} {pretty_model_name(fr['tip_angle_model'])} fit",
                            )
                    if ds.get('offplane_b') is not None and ds.get('offplane_y') is not None:
                        off_b = np.asarray(ds['offplane_b'], dtype=float)
                        off_y = np.asarray(ds['offplane_y'], dtype=float)
                        all_off_b.extend(off_b.tolist())
                        all_off_y.extend(off_y.tolist())
                        ax_off.plot(
                            off_b,
                            off_y,
                            marker=marker,
                            linestyle='None',
                            color=color,
                            markersize=5,
                            label=f"{phase_name} measured",
                        )

                if shared_tip_angle_model is not None and all_tip_b:
                    tip_plot_b = model_plot_b(shared_tip_angle_model, all_tip_b)
                    if tip_plot_b is not None:
                        ax_angle.plot(
                            tip_plot_b,
                            self._evaluate_curve_model(shared_tip_angle_model, tip_plot_b),
                            color=avg_cubic_colors['angle'],
                            linewidth=2.2,
                            marker='s',
                            markersize=3.0,
                            markevery=max(1, len(tip_plot_b) // 22),
                            label='Averaged cubic fit',
                        )

                if shared_r_cubic_model is not None and all_b_ranges:
                    fit_plot_b = model_plot_b(shared_r_cubic_model, all_b_ranges)
                    if fit_plot_b is not None:
                        ax_r_fit.plot(
                            fit_plot_b,
                            self._evaluate_curve_model(shared_r_cubic_model, fit_plot_b),
                            color=avg_cubic_colors['r'],
                            linewidth=2.4,
                            alpha=0.95,
                            label='Averaged cubic fit',
                        )

                if shared_z_cubic_model is not None and all_b_ranges:
                    fit_plot_b = model_plot_b(shared_z_cubic_model, all_b_ranges)
                    if fit_plot_b is not None:
                        ax_z_fit.plot(
                            fit_plot_b,
                            self._evaluate_curve_model(shared_z_cubic_model, fit_plot_b),
                            color=avg_cubic_colors['z'],
                            linewidth=2.4,
                            alpha=0.95,
                            label='Averaged cubic fit',
                        )

                if shared_offplane_plot_model is not None and all_off_b:
                    off_plot_b = model_plot_b(shared_offplane_plot_model, all_off_b)
                    if off_plot_b is not None:
                        ax_off.plot(
                            off_plot_b,
                            self._evaluate_curve_model(shared_offplane_plot_model, off_plot_b),
                            color=avg_cubic_colors['offplane'],
                            linewidth=2.2,
                            label=f"Averaged {pretty_model_name(shared_offplane_plot_model).lower()} fit",
                        )

                mirror_x_vals = [
                    float(fr['dataset']['x_ref_mirror_line_mm'])
                    for fr in fit_results.values()
                    if fr['dataset'].get('x_ref_mirror_line_mm') is not None
                ]
                if mirror_x_vals:
                    ax_raw_xz.axvline(float(np.mean(mirror_x_vals)), color='white', linestyle='--', linewidth=1.5, alpha=0.8, label='Mirror line')
                mirror_y_vals = [
                    float(fr['dataset']['y_ref_offplane_mirror_line_mm'])
                    for fr in fit_results.values()
                    if fr['dataset'].get('y_ref_offplane_mirror_line_mm') is not None
                ]
                if mirror_y_vals:
                    ax_raw_yz.axvline(float(np.mean(mirror_y_vals)), color='white', linestyle='--', linewidth=1.5, alpha=0.8, label='Mirror line')

                max_r_r2 = max(fr['r_r2'] for fr in fit_results.values()) if fit_results else float('nan')
                max_z_r2 = max(fr['z_r2'] for fr in fit_results.values()) if fit_results else float('nan')
                ax_ref.set_title('Referenced Planar Trajectory (X vs Z)')
                ax_planar.set_title('Planar Bending Coordinates (r=x, z)')
                ax_r_fit.set_title(f'Planar X Deflection vs Motor (phase fit max R² = {max_r_r2:.4f}; avg cubic R² = {0.0 if shared_r_cubic_r2 is None else shared_r_cubic_r2:.4f})')
                ax_z_fit.set_title(f'Axial Position vs Motor (phase fit max R² = {max_z_r2:.4f}; avg cubic R² = {0.0 if shared_z_cubic_r2 is None else shared_z_cubic_r2:.4f})')
                ax_r_res.set_title('Planar X Deflection Fit Residuals')
                ax_z_res.set_title('Axial Fit Residuals')
                max_angle_r2 = max(
                    (fr['tip_angle_r2'] for fr in fit_results.values() if fr['tip_angle_r2'] is not None),
                    default=float('nan'),
                )
                ax_angle.set_title(f'Attack Angle vs Motor (phase fit max R² = {0.0 if not np.isfinite(max_angle_r2) else max_angle_r2:.4f}; avg cubic R² = {0.0 if shared_tip_angle_r2 is None else shared_tip_angle_r2:.4f})')
                ax_off.set_title(
                    f"B-offset {pretty_model_name(shared_offplane_plot_model) if shared_offplane_plot_model is not None else offplane_fit_model.title()} "
                    f"Fit (averaged) (R² = {0.0 if shared_offplane_plot_r2 is None else shared_offplane_plot_r2:.4f})"
                )
                ax_raw_xz.set_title('XZ before mirroring', fontsize=11)
                ax_raw_yz.set_title('YZ before mirroring (±90)', fontsize=11)

                for ax in [ax_ref, ax_planar, ax_raw_xz, ax_raw_yz]:
                    ax.axis('equal')
                for ax in [ax_ref, ax_planar, ax_r_fit, ax_z_fit, ax_r_res, ax_z_res, ax_angle, ax_off, ax_raw_xz, ax_raw_yz]:
                    ax.legend(fontsize=8)
                ax_r_res.axhline(0.0, color='white', linestyle='--', linewidth=1.2, alpha=0.55)
                ax_z_res.axhline(0.0, color='white', linestyle='--', linewidth=1.2, alpha=0.55)

                ax_ref.set_xlabel('X transverse deflection (mm, ref. B=0)')
                ax_ref.set_ylabel('Z axial position (mm, ref. B=0)')
                ax_planar.set_xlabel('R = X (signed transverse deflection, mm)')
                ax_planar.set_ylabel('Z (axial) Position (mm)')
                ax_r_fit.set_xlabel('B Motor Position')
                ax_r_fit.set_ylabel('R = X (signed transverse deflection, mm)')
                ax_z_fit.set_xlabel('B Motor Position')
                ax_z_fit.set_ylabel('Z (axial) Position (mm)')
                ax_r_res.set_xlabel('B Motor Position')
                ax_r_res.set_ylabel('R = X Residuals (mm)')
                ax_z_res.set_xlabel('B Motor Position')
                ax_z_res.set_ylabel('Z Residuals (mm)')
                ax_angle.set_xlabel('B Motor Position')
                ax_angle.set_ylabel('Tip Angle vs Vertical (deg)')
                ax_off.set_xlabel('B Motor Position')
                ax_off.set_ylabel("Off-plane Y' (mm, ref.)")
                ax_raw_xz.set_xlabel('X (mm)')
                ax_raw_xz.set_ylabel('Z (mm)')
                ax_raw_yz.set_xlabel('Y proxy (mm)')
                ax_raw_yz.set_ylabel('Z (mm)')
                fig.tight_layout()
                self._apply_dark_theme_to_figure(fig)
                fig.savefig('10_dual_phase_fits.png', dpi=150, bbox_inches='tight', transparent=True, facecolor='none')
                plt.close(fig)

                xz_fig, xz_ax = plt.subplots(figsize=(8, 8))
                for phase_name, fr in fit_results.items():
                    ds = fr['dataset']
                    color = phase_colors.get(phase_name, None)
                    xz_ax.plot(
                        ds['r_coords'],
                        ds['z_coords'],
                        'o-',
                        color=color,
                        linewidth=1.8,
                        markersize=4,
                        label=f"{phase_name} measured",
                    )
                    xz_ax.plot(
                        fr['r_pred'],
                        fr['z_pred'],
                        '--',
                        color=color,
                        linewidth=2.0,
                        label=f"{phase_name} fit",
                    )
                    xz_ax.scatter(
                        ds['r_coords'][0],
                        ds['z_coords'][0],
                        color=color,
                        marker='s',
                        s=45,
                        zorder=3,
                    )
                xz_ax.set_title('X-Z trajectory hysteresis')
                xz_ax.set_xlabel('X / R (mm)')
                xz_ax.set_ylabel('Z (mm)')
                xz_ax.axis('equal')
                xz_ax.legend(fontsize=8)
                xz_fig.tight_layout()
                self._apply_dark_theme_to_figure(xz_fig)
                xz_fig.savefig('11_xz_trajectory_hysteresis.png', dpi=150, bbox_inches='tight', transparent=True, facecolor='none')
                plt.close(xz_fig)

                overlay_fig, overlay_axes = plt.subplots(1, 3, figsize=(19, 6.5))
                ax_overlay_xz, ax_overlay_x, ax_overlay_z = overlay_axes
                orientation_colors = {
                    'C0': '#6ec5ff',
                    'C180_mirrored': '#ff9f68',
                }
                phase_markers_overlay = {
                    'pull': 'o',
                    'release': 's',
                }
                avg_measured_style = {
                    'color': '#f2f2f2',
                    'linestyle': '-',
                    'linewidth': 2.8,
                    'alpha': 0.95,
                }
                avg_pred_style = {
                    'color': '#d7ff70',
                    'linestyle': ':',
                    'linewidth': 3.0,
                    'alpha': 1.0,
                }

                for phase_name, fr in fit_results.items():
                    ds = fr['dataset']
                    phase_marker = phase_markers_overlay.get(phase_name, 'o')
                    overlay = ds.get('mirrored_planar_overlay') or {}
                    if not overlay:
                        continue

                    b_common = np.asarray(overlay.get('b', []), dtype=float)
                    x0 = np.asarray(overlay.get('C0_x', []), dtype=float)
                    z0 = np.asarray(overlay.get('C0_z', []), dtype=float)
                    x180m = np.asarray(overlay.get('C180_x_mirrored', []), dtype=float)
                    z180 = np.asarray(overlay.get('C180_z', []), dtype=float)
                    xavg = np.asarray(overlay.get('avg_x', []), dtype=float)
                    zavg = np.asarray(overlay.get('avg_z', []), dtype=float)

                    if b_common.size == 0:
                        continue

                    x_ref = float(ds['x_ref_mirror_line_mm'])
                    z_ref = float(ds['z0_ref_mm'])
                    c0_x_defl = x0 - x_ref
                    c180_x_defl = x180m - x_ref
                    c0_z_defl = z0 - z_ref
                    c180_z_defl = z180 - z_ref
                    avg_x_defl = xavg - x_ref
                    avg_z_defl = zavg - z_ref

                    ax_overlay_xz.plot(
                        x0,
                        z0,
                        color=orientation_colors['C0'],
                        linewidth=1.8,
                        linestyle='-',
                        alpha=0.92,
                        marker=phase_marker,
                        markersize=3.8,
                        label=f"{phase_name} C=0",
                    )
                    ax_overlay_xz.plot(
                        x180m,
                        z180,
                        color=orientation_colors['C180_mirrored'],
                        linewidth=1.8,
                        linestyle='-',
                        alpha=0.92,
                        marker=phase_marker,
                        markersize=3.8,
                        label=f"{phase_name} C=180 mirrored",
                    )
                    ax_overlay_xz.plot(
                        xavg,
                        zavg,
                        color=avg_measured_style['color'],
                        linewidth=avg_measured_style['linewidth'],
                        linestyle=avg_measured_style['linestyle'],
                        alpha=avg_measured_style['alpha'],
                        marker=phase_marker,
                        markersize=3.6,
                        label=f"{phase_name} averaged",
                    )

                    phase_x_pred = x_ref + np.asarray(fr['r_pred'], dtype=float)
                    phase_z_pred = z_ref + np.asarray(fr['z_pred'], dtype=float)
                    ax_overlay_xz.plot(
                        phase_x_pred,
                        phase_z_pred,
                        color=avg_pred_style['color'],
                        linewidth=avg_pred_style['linewidth'],
                        linestyle=avg_pred_style['linestyle'],
                        alpha=avg_pred_style['alpha'],
                        marker=phase_marker,
                        markersize=3.2,
                        label=f"{phase_name} predicted",
                    )

                    ax_overlay_x.plot(
                        b_common,
                        c0_x_defl,
                        color=orientation_colors['C0'],
                        linewidth=1.6,
                        linestyle='-',
                        alpha=0.92,
                        marker=phase_marker,
                        markersize=3.6,
                        label=f"{phase_name} C=0",
                    )
                    ax_overlay_x.plot(
                        b_common,
                        c180_x_defl,
                        color=orientation_colors['C180_mirrored'],
                        linewidth=1.6,
                        linestyle='-',
                        alpha=0.92,
                        marker=phase_marker,
                        markersize=3.6,
                        label=f"{phase_name} C=180 mirrored",
                    )
                    ax_overlay_x.plot(
                        b_common,
                        avg_x_defl,
                        color=avg_measured_style['color'],
                        linewidth=avg_measured_style['linewidth'],
                        linestyle=avg_measured_style['linestyle'],
                        alpha=avg_measured_style['alpha'],
                        marker=phase_marker,
                        markersize=3.4,
                        label=f"{phase_name} averaged",
                    )
                    ax_overlay_x.plot(
                        b_common,
                        np.asarray(fr['r_pred'], dtype=float),
                        color=avg_pred_style['color'],
                        linewidth=avg_pred_style['linewidth'],
                        linestyle=avg_pred_style['linestyle'],
                        alpha=avg_pred_style['alpha'],
                        marker=phase_marker,
                        markersize=3.0,
                        label=f"{phase_name} predicted",
                    )

                    ax_overlay_z.plot(
                        b_common,
                        c0_z_defl,
                        color=orientation_colors['C0'],
                        linewidth=1.6,
                        linestyle='-',
                        alpha=0.92,
                        marker=phase_marker,
                        markersize=3.6,
                        label=f"{phase_name} C=0",
                    )
                    ax_overlay_z.plot(
                        b_common,
                        c180_z_defl,
                        color=orientation_colors['C180_mirrored'],
                        linewidth=1.6,
                        linestyle='-',
                        alpha=0.92,
                        marker=phase_marker,
                        markersize=3.6,
                        label=f"{phase_name} C=180 mirrored",
                    )
                    ax_overlay_z.plot(
                        b_common,
                        avg_z_defl,
                        color=avg_measured_style['color'],
                        linewidth=avg_measured_style['linewidth'],
                        linestyle=avg_measured_style['linestyle'],
                        alpha=avg_measured_style['alpha'],
                        marker=phase_marker,
                        markersize=3.4,
                        label=f"{phase_name} averaged",
                    )
                    ax_overlay_z.plot(
                        b_common,
                        np.asarray(fr['z_pred'], dtype=float),
                        color=avg_pred_style['color'],
                        linewidth=avg_pred_style['linewidth'],
                        linestyle=avg_pred_style['linestyle'],
                        alpha=avg_pred_style['alpha'],
                        marker=phase_marker,
                        markersize=3.0,
                        label=f"{phase_name} predicted",
                    )

                if shared_r_cubic_model is not None and shared_z_cubic_model is not None and all_b_ranges:
                    overlay_b = np.linspace(min(all_b_ranges), max(all_b_ranges), 250)
                    shared_r_pred = self._evaluate_curve_model(shared_r_cubic_model, overlay_b)
                    shared_z_pred = self._evaluate_curve_model(shared_z_cubic_model, overlay_b)
                    shared_x_refs = [
                        float(fr['dataset']['x_ref_mirror_line_mm'])
                        for fr in fit_results.values()
                        if fr['dataset'].get('x_ref_mirror_line_mm') is not None
                    ]
                    shared_z_refs = [
                        float(fr['dataset']['z0_ref_mm'])
                        for fr in fit_results.values()
                        if fr['dataset'].get('z0_ref_mm') is not None
                    ]
                    if shared_x_refs and shared_z_refs:
                        ax_overlay_xz.plot(
                            float(np.mean(shared_x_refs)) + shared_r_pred,
                            float(np.mean(shared_z_refs)) + shared_z_pred,
                            color=avg_pred_style['color'],
                            linewidth=3.0,
                            linestyle='--',
                            alpha=0.95,
                            label='Shared averaged prediction',
                        )
                    ax_overlay_x.plot(
                        overlay_b,
                        shared_r_pred,
                        color=avg_pred_style['color'],
                        linewidth=3.0,
                        linestyle='--',
                        alpha=0.95,
                        label='Shared averaged prediction',
                    )
                    ax_overlay_z.plot(
                        overlay_b,
                        shared_z_pred,
                        color=avg_pred_style['color'],
                        linewidth=3.0,
                        linestyle='--',
                        alpha=0.95,
                        label='Shared averaged prediction',
                    )

                ax_overlay_xz.set_title('Mirrored C=0 / C=180 Overlay in XZ')
                ax_overlay_x.set_title('Mirrored Planar X Deflection vs Motor')
                ax_overlay_z.set_title('Mirrored Axial Z Deflection vs Motor')
                ax_overlay_xz.set_xlabel('X (mm)')
                ax_overlay_xz.set_ylabel('Z (mm)')
                ax_overlay_x.set_xlabel('B Motor Position')
                ax_overlay_x.set_ylabel('Planar X Deflection (mm)')
                ax_overlay_z.set_xlabel('B Motor Position')
                ax_overlay_z.set_ylabel('Axial Z Deflection (mm)')
                ax_overlay_xz.axis('equal')
                ax_overlay_x.axhline(0.0, color='white', linestyle='--', linewidth=1.1, alpha=0.45)
                ax_overlay_z.axhline(0.0, color='white', linestyle='--', linewidth=1.1, alpha=0.45)
                for ax in overlay_axes:
                    ax.legend(fontsize=8)
                overlay_fig.tight_layout()
                self._apply_dark_theme_to_figure(overlay_fig)
                overlay_fig.savefig('12_mirrored_phase_overlays.png', dpi=150, bbox_inches='tight', transparent=True, facecolor='none')
                plt.close(overlay_fig)

            fit_models_by_phase = {}
            for phase, fr in fit_results.items():
                fit_models_by_phase[phase] = {
                    'r': fr['r_model'],
                    'z': fr['z_model'],
                    'r_pchip': fr['r_pchip_model'],
                    'z_pchip': fr['z_pchip_model'],
                    'r_cubic': fr['r_cubic_model'],
                    'z_cubic': fr['z_cubic_model'],
                    'tip_angle': fr['tip_angle_model'],
                    'tip_angle_pchip': fr['tip_angle_pchip_model'],
                    'tip_angle_cubic': fr['tip_angle_cubic_model'],
                    'tip_angle_avg_cubic': shared_tip_angle_model,
                    'offplane_y': fr['offplane_y_model'],
                    'offplane_y_linear': fr['offplane_y_linear_model'],
                    'offplane_y_cubic': fr['offplane_y_cubic_model'],
                    'offplane_y_pchip': fr['offplane_y_pchip_model'],
                    'offplane_y_avg_linear': shared_offplane_y_linear_model,
                    'offplane_y_avg_cubic': shared_offplane_y_cubic_model,
                    'r_avg_cubic': shared_r_cubic_model,
                    'z_avg_cubic': shared_z_cubic_model,
                }
            datasets_with_aliases = dict(datasets_for_fitting)
            fit_models_with_aliases = dict(fit_models_by_phase)
            for alias_name, canonical_name in dataset_aliases.items():
                if canonical_name in datasets_for_fitting:
                    datasets_with_aliases[alias_name] = datasets_for_fitting[canonical_name]
                if canonical_name in fit_models_by_phase:
                    fit_models_with_aliases[alias_name] = fit_models_by_phase[canonical_name]
            default_fit_models = fit_models_with_aliases[legacy_phase]
            self._postprocessed_datasets = datasets_with_aliases
            self._postprocessed_fit_models_by_phase = fit_models_with_aliases
            self._postprocessed_redundancy_diagnostics = redundancy_diagnostics

            cubic_calibration_data = {
                'selected_fit_model': fit_model,
                'selected_offplane_fit_model': offplane_fit_model,
                'default_phase_for_legacy_access': legacy_phase,
                'fit_models': default_fit_models,
                'fit_models_by_phase': fit_models_with_aliases,
                'datasets_by_phase': datasets_with_aliases,
                'redundancy_diagnostics': redundancy_diagnostics,
                'shared_aux_fit_models': {
                    'r_avg_cubic': shared_r_cubic_model,
                    'z_avg_cubic': shared_z_cubic_model,
                    'tip_angle_avg_cubic': shared_tip_angle_model,
                    'offplane_y': shared_offplane_y_model,
                    'offplane_y_avg_linear': shared_offplane_y_linear_model,
                    'offplane_y_avg_cubic': shared_offplane_y_cubic_model,
                },
                'phase_fit_metrics': {
                    phase: {
                        'r_r2': fr['r_r2'],
                        'z_r2': fr['z_r2'],
                        'r_avg_cubic_r2': shared_r_cubic_r2,
                        'z_avg_cubic_r2': shared_z_cubic_r2,
                        'tip_angle_r2': fr['tip_angle_r2'],
                        'tip_angle_avg_cubic_r2': shared_tip_angle_r2,
                        'offplane_y_r2': shared_offplane_y_r2,
                        'offplane_y_linear_r2': shared_offplane_y_linear_r2,
                        'offplane_y_cubic_r2': shared_offplane_y_cubic_r2,
                    }
                    for phase, fr in fit_results.items()
                },
                'r_coefficients': r_coefficients,
                'z_coefficients': z_coefficients,
                'shared_r_cubic_coefficients': shared_r_cubic_coefficients,
                'shared_z_cubic_coefficients': shared_z_cubic_coefficients,
                'tip_angle_coefficients': tip_angle_coefficients,
                'offplane_y_coefficients': y_off_coefficients,
                'offplane_y_linear_coefficients': y_off_linear_coefficients,
                'offplane_y_cubic_coefficients': y_off_cubic_coefficients,
            }
            with open(f"{robot_name}_cubic_calibration.pkl", "wb") as f:
                pickle.dump(cubic_calibration_data, f)

            predictor_code = """import numpy as np
from scipy.interpolate import PchipInterpolator

FIT_MODELS_BY_PHASE = {fit_models_json}
DEFAULT_PHASE = {default_phase_json}

def _polyval(c, b):
    return np.polyval(np.asarray(c, dtype=float), np.asarray(b, dtype=float))

def _evaluate_curve_model(model_descriptor, b_vals):
    if model_descriptor is None:
        return None
    b_arr = np.asarray(b_vals, dtype=float)
    model_type = str(model_descriptor.get('model_type', 'polynomial')).lower()
    if model_type == 'polynomial':
        return _polyval(model_descriptor.get('coefficients'), b_arr)
    if model_type == 'pchip':
        return PchipInterpolator(np.asarray(model_descriptor['x_knots'], dtype=float), np.asarray(model_descriptor['y_knots'], dtype=float), extrapolate=True)(b_arr)
    raise ValueError(f'Unsupported model_type: {{model_type}}')

def _get_phase_models(phase=None):
    phase_key = DEFAULT_PHASE if phase is None else str(phase).lower()
    if phase_key not in FIT_MODELS_BY_PHASE:
        raise KeyError(f'Unknown phase {{phase_key}}. Available: {{list(FIT_MODELS_BY_PHASE)}}')
    return FIT_MODELS_BY_PHASE[phase_key]

def get_fit_model(phase, fit_family='pchip'):
    phase_key = str(phase).lower()
    if phase_key not in FIT_MODELS_BY_PHASE:
        raise KeyError(f'Unknown phase {{phase_key}}. Available: {{list(FIT_MODELS_BY_PHASE)}}')
    m = FIT_MODELS_BY_PHASE[phase_key]
    fit_family = str(fit_family).lower()
    if fit_family == 'pchip':
        return {{
            'r': m.get('r_pchip') or m.get('r'),
            'z': m.get('z_pchip') or m.get('z'),
            'tip_angle': m.get('tip_angle_pchip') or m.get('tip_angle'),
            'offplane_y': m.get('offplane_y_pchip') or m.get('offplane_y'),
        }}
    if fit_family == 'linear':
        return {{
            'r': m.get('r'),
            'z': m.get('z'),
            'tip_angle': m.get('tip_angle'),
            'offplane_y': m.get('offplane_y_linear') or m.get('offplane_y'),
        }}
    if fit_family == 'cubic':
        return {{
            'r': m.get('r_cubic') or m.get('r_avg_cubic'),
            'z': m.get('z_cubic') or m.get('z_avg_cubic'),
            'tip_angle': m.get('tip_angle_cubic') or m.get('tip_angle_avg_cubic'),
            'offplane_y': m.get('offplane_y_cubic') or m.get('offplane_y_avg_cubic') or m.get('offplane_y'),
        }}
    raise ValueError(f'Unsupported fit_family: {{fit_family}}')

def predict_tip_position_from_b(b_motor_pos, phase=None):
    m = _get_phase_models(phase)
    b_motor = np.atleast_1d(b_motor_pos)
    return _evaluate_curve_model(m['r'], b_motor), _evaluate_curve_model(m['z'], b_motor)

def predict_avg_cubic_tip_position_from_b(b_motor_pos, phase=None):
    m = _get_phase_models(phase)
    b_motor = np.atleast_1d(b_motor_pos)
    return _evaluate_curve_model(m['r_avg_cubic'], b_motor), _evaluate_curve_model(m['z_avg_cubic'], b_motor)

def predict_tip_angle_from_b(b_motor_pos, phase=None):
    m = _get_phase_models(phase)
    if m.get('tip_angle') is None:
        return None
    return _evaluate_curve_model(m['tip_angle'], np.atleast_1d(b_motor_pos))

def predict_avg_cubic_tip_angle_from_b(b_motor_pos, phase=None):
    m = _get_phase_models(phase)
    if m.get('tip_angle_avg_cubic') is None:
        return None
    return _evaluate_curve_model(m['tip_angle_avg_cubic'], np.atleast_1d(b_motor_pos))

def predict_offplane_y_from_b(b_motor_pos, phase=None):
    m = _get_phase_models(phase)
    if m.get('offplane_y') is None:
        return None
    return _evaluate_curve_model(m['offplane_y'], np.atleast_1d(b_motor_pos))

def predict_cartesian_from_b(b_motor_pos, phase=None):
    return predict_tip_position_from_b(b_motor_pos, phase=phase)
""".format(
                fit_models_json=json.dumps(fit_models_with_aliases, indent=2),
                default_phase_json=json.dumps(legacy_phase),
            )
            with open(f"{robot_name}_cubic_prediction_functions.py", "w") as f:
                f.write(predictor_code)

            gcode_calibration_data = {
                'robot_name': robot_name,
                'calibration_date': pd.Timestamp.now().isoformat(),
                'selected_fit_model': fit_model,
                'selected_offplane_fit_model': offplane_fit_model,
                'default_phase_for_legacy_access': legacy_phase,
                'orientation_map': {'0': 'C=0 deg', '1': 'C=180 deg', '2': 'C=+90 deg', '3': 'C=-90 deg'},
                'motion_phase_map': {'0': 'pull', '1': 'release'},
                'redundant_pass_combination': redundant_pass_combination,
                'fit_models': default_fit_models,
                'fit_models_by_phase': fit_models_with_aliases,
                'datasets_by_phase': datasets_with_aliases,
                'redundancy_diagnostics': redundancy_diagnostics,
                'shared_aux_fit_models': {
                    'r_avg_cubic': shared_r_cubic_model,
                    'z_avg_cubic': shared_z_cubic_model,
                    'tip_angle_avg_cubic': shared_tip_angle_model,
                    'offplane_y': shared_offplane_y_model,
                    'offplane_y_avg_linear': shared_offplane_y_linear_model,
                    'offplane_y_avg_cubic': shared_offplane_y_cubic_model,
                },
                'phase_fit_metrics': {
                    phase: {
                        'r_r_squared': fr['r_r2'],
                        'z_r_squared': fr['z_r2'],
                        'r_avg_cubic_r_squared': shared_r_cubic_r2,
                        'z_avg_cubic_r_squared': shared_z_cubic_r2,
                        'tip_angle_r_squared': fr['tip_angle_r2'],
                        'tip_angle_avg_cubic_r_squared': shared_tip_angle_r2,
                        'offplane_y_r_squared': shared_offplane_y_r2,
                        'offplane_y_linear_r_squared': shared_offplane_y_linear_r2,
                        'offplane_y_cubic_r_squared': shared_offplane_y_cubic_r2,
                    }
                    for phase, fr in fit_results.items()
                },
                'cubic_coefficients': {
                    'default_phase': legacy_phase,
                    'r_coeffs': r_coefficients.tolist(),
                    'z_coeffs': z_coefficients.tolist(),
                    'r_avg_coeffs': shared_r_cubic_coefficients.tolist() if shared_r_cubic_coefficients is not None else None,
                    'z_avg_coeffs': shared_z_cubic_coefficients.tolist() if shared_z_cubic_coefficients is not None else None,
                    'tip_angle_coeffs': tip_angle_coefficients.tolist() if tip_angle_coefficients is not None else None,
                    'offplane_y_coeffs': y_off_coefficients.tolist() if y_off_coefficients is not None else None,
                    'offplane_y_linear_coeffs': y_off_linear_coefficients.tolist() if y_off_linear_coefficients is not None else None,
                    'offplane_y_cubic_coeffs': y_off_cubic_coefficients.tolist() if y_off_cubic_coefficients is not None else None,
                    'r_avg_equation': None if shared_r_cubic_model is None else shared_r_cubic_model.get('equation'),
                    'z_avg_equation': None if shared_z_cubic_model is None else shared_z_cubic_model.get('equation'),
                    'tip_angle_equation': None if shared_tip_angle_model is None else shared_tip_angle_model.get('equation'),
                    'offplane_y_equation': None if shared_offplane_y_model is None else shared_offplane_y_model.get('equation'),
                    'offplane_y_linear_equation': None if shared_offplane_y_linear_model is None else shared_offplane_y_linear_model.get('equation'),
                    'offplane_y_cubic_equation': None if shared_offplane_y_cubic_model is None else shared_offplane_y_cubic_model.get('equation'),
                },
                'motor_setup': {
                    'b_motor_axis': 'B',
                    'b_motor_home_position': 0.0,
                    'b_motor_position_range': [float(np.min(legacy_ds['common_b'])), float(np.max(legacy_ds['common_b']))],
                    'rotation_axis': 'C',
                    'rotation_axis_180_deg': 180.0,
                    'horizontal_axis': 'X',
                    'vertical_axis': 'Z',
                    'depth_axis': 'Y',
                },
            }
            with open(f"{robot_name}_gcode_calibration.json", "w") as f:
                json.dump(_to_json_compatible(gcode_calibration_data), f, indent=2)

            print("\nDual-phase calibration export complete:")
            print(f" - {robot_name}_cubic_calibration.pkl")
            print(f" - {robot_name}_cubic_prediction_functions.py")
            print(f" - {robot_name}_gcode_calibration.json")
            print(" - 10_dual_phase_fits.png")
            print(" - 11_xz_trajectory_hysteresis.png")
            print(" - 12_mirrored_phase_overlays.png")
            print(f" - {robot_name}_phase_fit_summary.xlsx")

            if export_skeleton:
                param_path, py_path, stl_path, _patched_json = self.export_parametric_skeleton_model(
                    robot_name=robot_name,
                    n_links=int(skeleton_links),
                    diameter_mm=float(skeleton_diameter_mm),
                    stl_reference_pose=bool(skeleton_reference_stl),
                )
                print("\nParametric skeleton export complete:")
                print(f" - {param_path.name}")
                print(f" - {py_path.name}")
                if stl_path is not None:
                    print(f" - {stl_path.name}")

            return {
                'fit_models_by_phase': fit_models_with_aliases,
                'datasets_by_phase': datasets_with_aliases,
                'default_phase': legacy_phase,
                'redundancy_diagnostics': redundancy_diagnostics,
                'phase_fit_metrics': gcode_calibration_data['phase_fit_metrics'],
                'json_path': os.path.join(processed_folder, f"{robot_name}_gcode_calibration.json"),
            }
        finally:
            os.chdir(original_dir)
