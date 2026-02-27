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
from scipy.interpolate import RegularGridInterpolator
from scipy.optimize import curve_fit

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
    distal_window: int = 80,
    roi_margin: int = 25,
    tangent_len: int = 30,
    radius_frac: float = 0.55,
    tip_keep_frac: float = 0.85,
    weight_power: float = 2.0,
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

    # Use coarse skeleton tip for coordinates.
    tip_y, tip_x = tip_y_s, tip_x_s

    tip_path = []
    for t in range(int(max(1, tangent_len))):
        yy = int(round(tip_y + t * vy))
        xx = int(round(tip_x + t * vx))
        if 0 <= yy < h and 0 <= xx < w:
            tip_path.append((yy, xx))

    return tip_y, tip_x, angle_deg, tip_path


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

        self.default_analysis_crop = {
            "crop_width_min": 650,
            "crop_width_max": 2900,
            "crop_height_min": 150,
            "crop_height_max": 1750,
        }
        self.analysis_crop = dict(self.default_analysis_crop)

        print("Calibration object initialized successfully!")

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
        drag_threshold_px = 30
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
                    cv2.circle(display, tuple(pt), 6, (0, 255, 255), -1)

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
        else:
            self.analysis_crop = dict(self.default_analysis_crop)
            print(f"Manual crop cancelled. Using default analysis crop: {self.analysis_crop}")

        if cam_opened_here:
            cam.release()

        return dict(self.analysis_crop)

    def connect_to_camera(self, cam_port=None, show_preview=False, enable_manual_focus=False, manual_focus_val=60):
        """
        Connects to the camera in the specified port and shows a live preview.
        Optionally enables manual focus and sets the focus value.
        """
        if cam_port is None:
            cam_port = 0
        self.cam_port = cam_port

        # Removed cv2.CAP_DSHOW for Mac compatibility
        self.cam = cv2.VideoCapture(self.cam_port)

        # Optional: configure manual focus immediately on connect
        if enable_manual_focus:
            try:
                self.cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                self.cam.set(cv2.CAP_PROP_FOCUS, float(manual_focus_val))
                print(f"Manual focus enabled (FOCUS={manual_focus_val}).")
            except Exception as e:
                print(f"Warning: could not enable manual focus on connect: {e}")

        # display showing preview
        if show_preview:
            # For preview, autofocus can help the user frame the shot; keep as-is
            try:
                if not enable_manual_focus:
                    self.cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            except Exception:
                pass

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
                  jogging_feedrate=1000,
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
                  # Camera focus control
                  enable_manual_focus=True):
        """
        Probes at 5 XYZ locations and for each location:
        - Capture orientation 0 pull (B axis microsteps)
        - Rotate C by robot_rotation_axis_180_deg
        - Capture orientation 1 pull
        - Rotate C back by robot_rotation_axis_180_deg (legacy-style two-orientation capture)

        Filenames begin with: "{orientation}_{X}_{B}_..." so downstream processing can parse orientation/X/B,
        while extra suffix tokens avoid overwriting between different Z points.
        """
        # --- connectivity checks ---
        if self.cam is None or self.rrf is None or self.cam_port is None:
            print("Not connected to camera or robot. Please run connect_to_camera followed by connect_to_robot.")
            print("Exiting...")
            return

        # Default probe points per your request
        if probe_points is None:
            probe_points = [
                (30.0, 0.0, -80.0),
                (90.0, 0.0, -80.0),
                (90.0, 0.0, -110.0),
                (30.0, 0.0, -110.0),
                (75.0, 0.0, -90.0),
            ]

        # Validate calibration folder
        if not hasattr(self, 'calibration_data_folder') or not os.path.exists(self.calibration_data_folder):
            print("Error: Calibration data folder not found. Make sure the class was initialized properly.")
            return

        os.chdir(self.calibration_data_folder)

        # Timing parameters
        large_move_safety = 0.0
        small_move_safety = 0.3
        rotation_safety = 0.0
        settling_time_large = 3.0
        settling_time_small = 0.2
        rotation_settling_time = 1.8
        small_move_threshold = 0.5

        # Prepare output folder
        if not os.path.isdir("raw_image_data_folder"):
            os.mkdir("raw_image_data_folder")
        else:
            print("You already have raw image data for this project. This will overwrite the current data.")

        os.chdir("raw_image_data_folder")

        # Camera setup
        cam = self.cam if self.cam is not None else cv2.VideoCapture(self.cam_port if self.cam_port is not None else 0)
        cam.set(cv2.CAP_PROP_FRAME_WIDTH, 3840)
        cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 2160)

        # Manual focus (re-enabled)
        if enable_manual_focus:
            try:
                cam.set(cv2.CAP_PROP_AUTOFOCUS, 0)
                cam.set(cv2.CAP_PROP_FOCUS, float(manual_focus_val))
                print(f"Camera manual focus enabled: FOCUS={manual_focus_val}")
            except Exception as e:
                print(f"Warning: could not set manual focus (FOCUS={manual_focus_val}): {e}")
        else:
            try:
                cam.set(cv2.CAP_PROP_AUTOFOCUS, 1)
            except Exception:
                pass

        def capture_and_save(orientation: int, x: float, y: float, z: float, b: float, probe_idx: int, step_idx: int):
            # Minimal buffer flush
            _ = cam.read()
            ret, image = cam.read()
            if not ret:
                ret, image = cam.read()

            # IMPORTANT: first 3 underscore tokens are numeric: orientation, X, B
            # Suffix avoids overwriting between different Y/Z/probe/step
            filename = (
                f"{orientation}_{x:.2f}_{b:.2f}"
                f"_Y{y:.2f}_Z{z:.2f}_P{probe_idx:02d}_S{step_idx:02d}.png"
            )

            if ret:
                cv2.imwrite(filename, image)
                print(f" ✓ Saved {filename}")
            else:
                print(f" ✗ ERROR: Could not save {filename}")

        def wait_for_duet_motion_complete(extra_settle: float = 0.0):
            """
            Block until Duet reports queued motion is complete, then optionally settle.
            Uses M400 (wait for moves to finish) rather than estimating a sleep duration.
            """
            try:
                self.rrf.send_code("M400")
            except Exception as e:
                # Fallback keeps acquisition working if M400/API call fails unexpectedly.
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
                fr = jogging_feedrate / 0.025  # faster rotation

            g = f"G91 G1 {axis_name}{c_units} F{fr}"
            print(f" Command: {g}")
            self.rrf.send_code(g)
            angle_deg = (c_units / robot_rotation_axis_180_deg) * 180.0 if robot_rotation_axis_180_deg else 0.0
            print(f" Rotating {angle_deg:.1f}° - waiting for Duet (M400) + {rotation_settling_time:.1f}s settle")
            wait_for_duet_motion_complete(extra_settle=rotation_settling_time)

            self.rrf.send_code("G90")

        # Track current robot position on relevant axes
        pos = {
            robot_front_axis_name: 0.0,
            robot_stage_y_axis_name: 0.0,
            robot_stage_z_axis_name: 0.0,
            robot_rear_axis_name: 0.0,
        }

        print("Starting probe sequence (legacy rotate behavior)...")
        print(f"Probe points (X,Y,Z): {probe_points}")
        print(f"Pull axis: {robot_rear_axis_name} | steps={b_steps} | step_size={b_step_size}mm | start={b_start}mm")
        print(f"Rotation axis: {robot_rotation_axis_name} | 180deg={robot_rotation_axis_180_deg} axis units")

        # Set pull axis to start
        print("\nMoving pull axis to start position...")
        pos = move_abs(jogging_feedrate, pos, **{robot_rear_axis_name: b_start})

        total_images = 0
        for probe_idx, (x, y, z) in enumerate(probe_points, start=1):
            print(f"\n{'=' * 70}")
            print(f"PROBE {probe_idx}/{len(probe_points)}: X={x}, Y={y}, Z={z}")
            print(f"{'=' * 70}")

            # Move to probe XYZ
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

            # Ensure B at start
            print(f" Setting {robot_rear_axis_name} to start ({b_start:.2f})...")
            pos = move_abs(jogging_feedrate, pos, **{robot_rear_axis_name: b_start})

            # --- Phase 1: orientation 0 ---
            print(" Phase 1: orientation 0 (no rotation)")
            # Baseline capture at B start (typically B=0.0)
            print(f" Baseline capture: {robot_rear_axis_name}={b_start:.2f}")
            pos = move_abs(jogging_feedrate, pos, **{robot_rear_axis_name: b_start})
            capture_and_save(0, x, y, z, b_start, probe_idx, 0)
            total_images += 1

            # Pull sequence
            for step_idx in range(1, b_steps + 1):
                b_val = b_start + step_idx * b_step_size
                print(f" Step {step_idx:02d}/{b_steps}: {robot_rear_axis_name}={b_val:.2f}")
                pos = move_abs(jogging_feedrate, pos, **{robot_rear_axis_name: b_val})
                capture_and_save(0, x, y, z, b_val, probe_idx, step_idx)
                total_images += 1

            # Return B to start
            pos = move_abs(jogging_feedrate, pos, **{robot_rear_axis_name: b_start})

            # Rotate to orientation 1
            print(" Rotating to orientation 1...")
            rotate_rel(robot_rotation_axis_name, robot_rotation_axis_180_deg)

            # --- Phase 2: orientation 1 ---
            print(" Phase 2: orientation 1 (after rotation)")
            # Baseline capture at B start (typically B=0.0)
            print(f" Baseline capture: {robot_rear_axis_name}={b_start:.2f}")
            pos = move_abs(jogging_feedrate, pos, **{robot_rear_axis_name: b_start})
            capture_and_save(1, x, y, z, b_start, probe_idx, 0)
            total_images += 1

            # Pull sequence
            for step_idx in range(1, b_steps + 1):
                b_val = b_start + step_idx * b_step_size
                print(f" Step {step_idx:02d}/{b_steps}: {robot_rear_axis_name}={b_val:.2f}")
                pos = move_abs(jogging_feedrate, pos, **{robot_rear_axis_name: b_val})
                capture_and_save(1, x, y, z, b_val, probe_idx, step_idx)
                total_images += 1

            # Return B to start
            pos = move_abs(jogging_feedrate, pos, **{robot_rear_axis_name: b_start})

            # Rotate back to orientation 0 (so every probe starts consistent)
            print(" Rotating back to orientation 0...")
            rotate_rel(robot_rotation_axis_name, -robot_rotation_axis_180_deg)

        print("\n" + "=" * 70)
        print("ALL PROBES FINISHED!")
        print("=" * 70)
        print(f"Total images captured: {total_images}")
        print(f"Final position: {pos}")

        os.chdir(self.calibration_data_folder)
        return 1

    def find_ctr_tip_skeleton(self, binary_image, base_band_frac=0.05, do_erosion_break=False, prune_spurs=True, min_spur_len=15, return_tip_angle=False, tip_angle_path_len=40, return_debug=False):
        """
        binary_image: uint8 0/255 where tube is dark (0) and background is white (255)

        Returns: (tip_row, tip_col) in binary_image coordinates
        """
        fg = (binary_image == 0).astype(np.uint8)

        if do_erosion_break:
            fg = cv2.erode(fg, np.ones((3, 3), np.uint8), iterations=1)

        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, k, iterations=2)
        fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, k, iterations=1)

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

        # Multi-source BFS from base band -> robust against wrong single base endpoint.
        dist = _multisource_bfs_geodesic(skel, base_pixels)
        if (dist >= 0).sum() == 0:
            raise ValueError("Geodesic BFS failed (disconnected skeleton?)")

        # Robust distal pose from dominant PCA axis near tip (avoids Y-split branch issues).
        if return_tip_angle:
            tip_y, tip_x, tip_angle_deg, tip_path = _tip_pose_from_distal_pca(
                skel_u8=skel,
                dist=dist,
                mask_u8=mask,
                distal_window=80,
                roi_margin=25,
                tangent_len=tip_angle_path_len,
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
            distal_window=80,
            roi_margin=25,
            tangent_len=tip_angle_path_len,
        )
        if return_debug:
            debug_data = {
                "skeleton": skel.copy(),
                "dist": dist.copy(),
                "tip_path": tip_path,
            }
            return tip_y, tip_x, debug_data
        return tip_y, tip_x

    def analyze_data(self, image_file_name, crop_width_min=None, crop_width_max=None, crop_height_min=None, crop_height_max=None, threshold=200, ):
        """ Analyzes the data for the calibrations. Only analyzes one image. """

        def get_pos_from_file_name(file_name):
            base = os.path.splitext(os.path.basename(file_name))[0]
            parts = base.split("_")
            orientation = int(parts[0])
            ntnl_pos = float(parts[1])
            ss_pos = float(parts[2])
            return [orientation, ntnl_pos, ss_pos]

        # Fixed path handling for Mac
        raw_data_folder = os.path.join(self.calibration_data_folder, "raw_image_data_folder")
        tip_locations_array_coarse = np.zeros((len(os.listdir(raw_data_folder)), 6))
        tip_locations_array_fine = np.zeros((len(os.listdir(raw_data_folder)), 6))
        i = 0

        # Fixed path handling for Mac
        image = cv2.imread(os.path.join(raw_data_folder, image_file_name))

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
        _, binary_image = cv2.threshold(grayscale_image, threshold, 255, cv2.THRESH_BINARY)

        # # finding the approximate tip for a vertical tube, scanning from the bottom
        # for x in np.linspace(np.shape(binary_image)[0] - 1, 0, np.shape(binary_image)[0]).astype(int):
        # current_row = binary_image[x, :]
        # current_range = np.max(current_row) - np.min(current_row)
        # if current_range > 128:
        # tip_row = x
        # break
        # tip_column = np.average(np.where(binary_image[tip_row, :] <= 128)).astype(int)

        tip_row, tip_column, tip_angle_deg, tip_debug = self.find_ctr_tip_skeleton(
            binary_image,
            return_tip_angle=True,
            return_debug=True,
        )

        skel = tip_debug["skeleton"]
        tip_path = tip_debug["tip_path"]

        orientation, ntnl_pos, ss_pos = get_pos_from_file_name(image_file_name)

        tip_locations_array_coarse[i, :] = np.array(
            [
                tip_row + crop_y_min_img,
                tip_column + crop_x_min_img,
                float(orientation),
                float(ss_pos),
                float(ntnl_pos),
                float(tip_angle_deg),
            ]
        )

        crop_x_min = int(max(tip_column - 75, 0))
        crop_x_max = int(min(tip_column + 75, np.shape(binary_image)[1] - 1))
        crop_y_min = int(max(tip_row - 75, 0))
        crop_y_max = int(min(tip_row + 75, np.shape(binary_image)[0] - 1))

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

        axs[1, 0].imshow(binary_image[crop_y_min:crop_y_max, crop_x_min:crop_x_max])
        skel_in_zoom = (skel[crop_y_min:crop_y_max, crop_x_min:crop_x_max] == 1)
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

        # if p_1 is not None and x_points.size > 0:
        # axs[1, 1].scatter(np.unique(x_points), p_1(np.unique(x_points)))
        # if p_2 is not None and y_points.size > 0:
        # axs[1, 1].scatter(p_2(np.unique(y_points)), np.unique(y_points))
        # axs[1, 1].scatter(tip_location[0], tip_location[1])
        # axs[1, 1].legend(["coarse tip", "fit 1", "fit 2", "fine tip"])
        # axs[1, 1].set_title('Identified fine tip')

        if len(tip_path) >= 2:
            axs[1, 1].legend(["skeleton", "tip tangent path", "coarse tip"])
        else:
            axs[1, 1].legend(["skeleton", "coarse tip"])

        axs[1, 1].set_title(f'Identified coarse tip (angle={tip_angle_deg:.2f} deg)')

        tip_locations_array_fine[i, :] = np.array(
            [tip_location[1] + crop_y_min_img + crop_y_min,
             tip_location[0] + crop_x_min_img + crop_x_min,
             float(orientation),
             float(ss_pos),
             float(ntnl_pos),
             float(tip_angle_deg)]
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
            all_files = os.listdir(raw_data_folder)
            image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif')
            image_files = [f for f in all_files if f.lower().endswith(image_extensions)]

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
        self.tip_locations_array_coarse = np.zeros((num_images, 6))
        # self.tip_locations_array_fine = np.zeros((num_images, 5))

        successful_analyses = 0
        failed_analyses = 0

        for i, image_file in enumerate(image_files):
            print(f"\nProcessing image {i + 1}/{num_images}: {image_file}")
            try:
                # fig, axs, coarse_tip, fine_tip = self.analyze_data(
                fig, axs, coarse_tip, _ = self.analyze_data(
                    image_file,
                    crop_width_min,
                    crop_width_max,
                    crop_height_min,
                    crop_height_max,
                    threshold
                )

                self.tip_locations_array_coarse[i, :] = coarse_tip
                # self.tip_locations_array_fine[i, :] = fine_tip

                output_filename = f"{os.path.splitext(image_file)[0]}_analysis.png"
                output_path = os.path.join(analysis_output_folder, output_filename)
                fig.savefig(output_path, dpi=150, bbox_inches='tight')
                plt.close(fig)

                successful_analyses += 1
                print(f" ✓ Successfully processed and saved to analysis_outputs/{output_filename}")

            except Exception as e:
                print(f" ✗ Error processing {image_file}: {e}")
                failed_analyses += 1
                self.tip_locations_array_coarse[i, :] = np.nan
                # self.tip_locations_array_fine[i, :] = np.nan

        try:
            coarse_file = os.path.join(processed_folder, "tip_locations_coarse.npy")
            np.save(coarse_file, self.tip_locations_array_coarse)
            # np.save(fine_file, self.tip_locations_array_fine)
            print(f"\n✓ Saved coarse tip locations to: {coarse_file}")
            # print(f"✓ Saved fine tip locations to: {fine_file}")

            try:
                columns = ['tip_row', 'tip_column', 'orientation', 'ss_pos', 'ntnl_pos', 'tip_angle_deg']
                df_coarse = pd.DataFrame(self.tip_locations_array_coarse, columns=columns)
                df_coarse['image_file'] = image_files
                coarse_csv = os.path.join(processed_folder, "tip_locations_coarse.csv")
                df_coarse.to_csv(coarse_csv, index=False)
                # df_fine = pd.DataFrame(self.tip_locations_array_fine, columns=columns)
                # df_fine['image_file'] = image_files
                # fine_csv = os.path.join(processed_folder, "tip_locations_fine.csv")
                # df_fine.to_csv(fine_csv, index=False)
                print(f"✓ Saved CSV file: {coarse_csv}")
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

    def postprocess_calibration_data(self, width_in_pixels=3025, width_in_mm=140, robot_name="calibrated_robot", save_plots=True):
        """ Postprocesses calibration data starting from step 3 (assumes pixel data already exists). This method starts from converting pixel coordinates to physical coordinates and continues with all subsequent steps. """
        # Check if we have the required data in memory, if not try to load from files
        # if not hasattr(self, 'tip_locations_array_fine') or not hasattr(self, 'tip_locations_array_coarse'):
        if not hasattr(self, 'tip_locations_array_coarse'):
            print("Tip location data not found in memory, attempting to load from saved files...")

            # Find the processed data folder
            processed_folder = os.path.join(self.calibration_data_folder, "processed_image_data_folder")
            if not os.path.exists(processed_folder):
                print("Error: Processed data folder not found. Run analyze_data_batch() first.")
                return None

            # Try to load the numpy arrays
            coarse_file = os.path.join(processed_folder, "tip_locations_coarse.npy")
            # fine_file = os.path.join(processed_folder, "tip_locations_fine.npy")

            try:
                if os.path.exists(coarse_file):
                    self.tip_locations_array_coarse = np.load(coarse_file)
                    # self.tip_locations_array_fine = np.load(fine_file)
                    print(f"✓ Loaded existing data from {coarse_file}")
                    print(f" Coarse data shape: {self.tip_locations_array_coarse.shape}")
                    # print(f" Fine data shape: {self.tip_locations_array_fine.shape}")
                else:
                    print("Error: Required .npy files not found. Run analyze_data_batch() first.")
                    print(f"Looking for: {coarse_file}")
                    # print(f"Looking for: {fine_file}")
                    return None
            except Exception as e:
                print(f"Error loading saved data: {e}")
                print("Run analyze_data_batch() first to generate the required data.")
                return None

        # Work in the processed data folder
        processed_folder = os.path.join(self.calibration_data_folder, "processed_image_data_folder")
        if not os.path.exists(processed_folder):
            print("Error: Processed data folder not found. Run analyze_data_batch() first.")
            return None

        original_dir = os.getcwd()
        os.chdir(processed_folder)

        try:
            print(f"\n" + "=" * 50)
            print("Starting calibration data postprocessing from step 3...")

            if self.tip_locations_array_coarse.shape[1] < 5:
                print("Error: tip_locations_coarse must have at least 5 columns.")
                return None

            has_tip_angle = self.tip_locations_array_coarse.shape[1] >= 6
            if has_tip_angle:
                print("Tip-angle data detected and will be included in calibration fitting.")
            else:
                print("Warning: Tip-angle data not found. Re-run analyze_data_batch() to enable angle fitting.")

            # ---------- Step 3: Convert to physical coordinates + canonicalize axes ----------
            print(f"Converting to physical coordinates (scale: {width_in_pixels} px = {width_in_mm} mm)")

            tip_locations_array_fine_mm = self.tip_locations_array_coarse.copy()
            tip_locations_array_fine_mm[:, 0:2] = self.tip_locations_array_coarse[:, 0:2] / width_in_pixels * width_in_mm

            # Canonical camera frame (matches your desired Plot 03):
            # X = image column (mm) = original col1
            # Z = -image row (mm)   = -original col0  (flip Z negative ONCE here)
            x_mm = tip_locations_array_fine_mm[:, 1].copy()
            z_mm = -tip_locations_array_fine_mm[:, 0].copy()

            # Overwrite first two columns so the rest of the pipeline is consistent:
            # col0 := X, col1 := Z
            tip_locations_array_fine_mm[:, 0] = x_mm
            tip_locations_array_fine_mm[:, 1] = z_mm

            # Optional: save mm data
            df_mm = pd.DataFrame(tip_locations_array_fine_mm)
            df_mm.to_excel('tip_locations_coarse_mm.xlsx', index=False, header=False)

            # ---------- Step 4: Fit line and plot (in canonical Z-X) ----------
            print("Fitting alignment line (X as a function of Z)...")
            # Fit X = mZ + b so the reference alignment line is vertical (along Z axis).
            coefficients = np.polyfit(tip_locations_array_fine_mm[:, 1], tip_locations_array_fine_mm[:, 0], 1)  # X = mZ + b
            p = np.poly1d(coefficients)

            if save_plots:
                z_sorted = np.sort(tip_locations_array_fine_mm[:, 1])
                plt.figure(figsize=(10, 8))
                plt.scatter(tip_locations_array_fine_mm[:, 0], tip_locations_array_fine_mm[:, 1], alpha=0.7)
                plt.plot(p(z_sorted), z_sorted, 'r-', linewidth=2)
                plt.axis("equal")
                plt.title(f"Tip Locations with Fitted Vertical Alignment Line (dX/dZ slope: {coefficients[0]:.4f})")
                plt.xlabel("X (mm)")
                plt.ylabel("Z (mm) [flipped]")
                plt.savefig("02_tip_locations_with_line_fitted.png", dpi=150, bbox_inches='tight')
                plt.close()

            # ---------- Step 5: Change of basis (rotate to align) ----------
            print("Performing coordinate alignment (rotate to align with Z axis)...")

            arr0 = tip_locations_array_fine_mm.copy()
            arr0[:, 0] -= np.mean(arr0[:, 0])  # X
            arr0[:, 1] -= np.mean(arr0[:, 1])  # Z

            # If slope = dX/dZ, rotate by +atan(slope) to align fitted line with Z axis.
            theta = math.atan(coefficients[0])
            print(f"Rotation angle: {theta:.4f} rad ({theta * 180 / math.pi:.2f} deg)")

            R = np.array([[math.cos(theta), -math.sin(theta)],
                        [math.sin(theta),  math.cos(theta)]])

            arr_rot = arr0.copy()
            arr_rot[:, 0:2] = (R @ arr0[:, 0:2].T).T

            if save_plots:
                plt.figure(figsize=(12, 6))
                plt.subplot(1, 2, 1)
                plt.scatter(arr0[:, 0], arr0[:, 1], alpha=0.7)
                plt.title("03: Before Alignment (Canonical X-Z)")
                plt.axis("equal")
                plt.xlabel("X (mm)")
                plt.ylabel("Z (mm) [flipped]")
                plt.subplot(1, 2, 2)
                plt.scatter(arr_rot[:, 0], arr_rot[:, 1], alpha=0.7)
                plt.title("03: After Alignment (Aligned X-Z)")
                plt.axis("equal")
                plt.xlabel("X (mm)")
                plt.ylabel("Z (mm) [flipped]")
                plt.tight_layout()
                plt.savefig("03_tip_locations_alignment.png", dpi=150, bbox_inches='tight')
                plt.close()

            # ---------- Step 6: Orientation processing ----------
            # Mirror + average the X component between orientations 0/1, paired by b_pull.
            print("Processing orientations: mirror + average X components by paired b_pull values...")

            # Aliases for columns (rotated array keeps original extra columns)
            # arr_rot columns:
            #   0 X_mm, 1 Z_mm, 2 orientation, 3 b_pull (was ss_pos), 4 ntnl_pos (unused), 5 tip_angle_deg
            ORI_COL = 2
            B_PULL_COL = 3
            ANGLE_COL = 5

            arr_valid_rows = arr_rot[np.isfinite(arr_rot[:, 0])].copy()
            if arr_valid_rows.size == 0:
                raise ValueError("No valid aligned X values found in arr_rot to compute planar mirror-line reference.")

            x_ref_mirror_line = float(np.mean(arr_valid_rows[:, 0]))
            b_ref_mirror_line_mean = float(np.mean(arr_valid_rows[:, B_PULL_COL]))
            b_ref_mirror_line_min = float(np.min(arr_valid_rows[:, B_PULL_COL]))
            b_ref_mirror_line_max = float(np.max(arr_valid_rows[:, B_PULL_COL]))

            print(
                "Planar radial zero (mirror line) from aligned unmirrored data (all B, all orientations): "
                f"samples={arr_valid_rows.shape[0]}, X_ref_mirror_line={x_ref_mirror_line:.6f} mm, "
                f"B range=[{b_ref_mirror_line_min:.6f}, {b_ref_mirror_line_max:.6f}], "
                f"B mean={b_ref_mirror_line_mean:.6f}"
            )

            o0 = arr_rot[arr_rot[:, ORI_COL] == 0].copy()
            o1 = arr_rot[arr_rot[:, ORI_COL] == 1].copy()
            if o0.size == 0 or o1.size == 0:
                raise ValueError("Missing orientation 0 or 1 data; cannot mirror/average X between orientations.")

            def circular_mean_deg(a):
                a = np.asarray(a, dtype=float)
                valid = np.isfinite(a)
                if not np.any(valid):
                    return float("nan")
                rad = np.deg2rad(a[valid])
                s = np.mean(np.sin(rad))
                c = np.mean(np.cos(rad))
                return float((np.rad2deg(np.arctan2(s, c)) + 360.0) % 360.0)

            def collapse_by_bpull(arr):
                b = arr[:, B_PULL_COL]
                uniq = np.unique(b)
                out = []
                for bu in uniq:
                    m = (b == bu)
                    row = arr[m].copy()
                    r = row[0].copy()
                    r[0] = np.mean(row[:, 0])  # X
                    r[1] = np.mean(row[:, 1])  # Z
                    r[B_PULL_COL] = bu
                    if has_tip_angle and row.shape[1] > ANGLE_COL:
                        r[ANGLE_COL] = circular_mean_deg(row[:, ANGLE_COL])
                    out.append(r)
                out = np.vstack(out)
                return out[np.argsort(out[:, B_PULL_COL])]

            o0c = collapse_by_bpull(o0)
            o1c = collapse_by_bpull(o1)

            b0 = o0c[:, B_PULL_COL]
            b1 = o1c[:, B_PULL_COL]
            common_b = np.intersect1d(b0, b1)
            if common_b.size == 0:
                raise ValueError("No matching b_pull values between orientations 0 and 1.")

            idx0 = np.searchsorted(b0, common_b)
            idx1 = np.searchsorted(b1, common_b)
            o0p = o0c[idx0].copy()
            o1p = o1c[idx1].copy()

            # Requested behavior: mirror and average X between orientations.
            o1_x_mirrored = -o1p[:, 0]
            avg_x = (o0p[:, 0] + o1_x_mirrored) / 2.0

            # Keep a single paired trajectory for fitting.
            avg_z = (o0p[:, 1] + o1p[:, 1]) / 2.0
            if has_tip_angle and o0p.shape[1] > ANGLE_COL and o1p.shape[1] > ANGLE_COL:
                # Mirror the orientation-1 angle consistently with X reflection, then circular-average.
                o1_ang_mirrored = (360.0 - o1p[:, ANGLE_COL]) % 360.0
                avg_ang = np.array(
                    [circular_mean_deg([a0, a1]) for a0, a1 in zip(o0p[:, ANGLE_COL], o1_ang_mirrored)],
                    dtype=float
                )
            else:
                avg_ang = np.full_like(common_b, np.nan, dtype=float)

            # [X_mm, Z_mm, b_pull, tip_angle_deg]
            tip_locations_final = np.column_stack([avg_x, avg_z, common_b, avg_ang])

            print("After orientation X mirroring/averaging (aligned frame):")
            print(f"X range: {tip_locations_final[:, 0].min():.3f} to {tip_locations_final[:, 0].max():.3f} mm")
            print(f"Z range: {tip_locations_final[:, 1].min():.3f} to {tip_locations_final[:, 1].max():.3f} mm")
            print(f"b_pull range: {tip_locations_final[:, 2].min():.3f} to {tip_locations_final[:, 2].max():.3f}")
            if np.all(np.isfinite(tip_locations_final[:, 3])):
                print(f"tip_angle range: {tip_locations_final[:, 3].min():.3f} to {tip_locations_final[:, 3].max():.3f} deg")
            else:
                print("tip_angle range: unavailable")

            if save_plots:
                plt.figure(figsize=(10, 8))
                plt.scatter(tip_locations_final[:, 0], tip_locations_final[:, 1], alpha=0.7)
                plt.xlabel("X (mm)")
                plt.ylabel("Z (mm) [flipped]")
                plt.title("03b: After Orientation X Mirroring + Averaging")
                plt.axis("equal")
                plt.grid(True, alpha=0.3)
                plt.savefig("03b_after_orientation_processing.png", dpi=150, bbox_inches='tight')
                plt.close()

            # ---------- Step 7: Physical scaling correction (still skipped) ----------
            print("Skipping physical scaling correction (disabled for current setup).")

            # ---------- Step 8: Simplified calibration data for fitting ----------
            # Since ntnl_pos is irrelevant now, simplified data is sorted by b_pull.
            print("Creating simplified calibration data (sorted by b_pull only)...")

            # tip_locations_final columns: [X_mm, Z_mm, b_pull, tip_angle_deg]
            sort_idx = np.argsort(tip_locations_final[:, 2])
            tip_locations_final_simplified = tip_locations_final[sort_idx].copy()

            print(f"Simplified data shape: {tip_locations_final_simplified.shape}")
            print(f"b_pull unique count: {len(np.unique(tip_locations_final_simplified[:, 2]))}")

            # Step 8.5: Convert to planar bending coordinates and fit cubic equations
            print("\nConverting to planar bending coordinates and fitting cubic equations...")

            # --- Bridge for Step 8.5 (NEW pipeline: b_pull only) ---
            # tip_locations_final_simplified columns: [X_mm, Z_mm, b_pull, tip_angle_deg]
            x_avg_raw = tip_locations_final_simplified[:, 0].copy()
            z_avg_raw = tip_locations_final_simplified[:, 1].copy()
            delta_motor = tip_locations_final_simplified[:, 2].copy()  # b_pull
            tip_angle_avg = tip_locations_final_simplified[:, 3].copy() if has_tip_angle else None

            # -------------------------------------------------------------------------
            # ZERO REFERENCE (requested):
            # Use the FIRST measured tip location at B = 0.0 as the zero reference for x, z, and r.
            # If exact 0.0 is not present (floating-point), use np.isclose and fall back to nearest B.
            # -------------------------------------------------------------------------
            b_zero_target = 0.0
            zero_mask = np.isclose(delta_motor, b_zero_target, atol=1e-9)

            if np.any(zero_mask):
                # "first measured" among the rows that match B=0.0 in current ordering
                zero_idx = np.where(zero_mask)[0][0]
            else:
                # Fallback: nearest B to 0.0
                zero_idx = int(np.argmin(np.abs(delta_motor - b_zero_target)))
                print(f"Warning: No exact B=0.0 found; using nearest point at B={delta_motor[zero_idx]:.6f} as zero reference.")

            # X/r reference uses mirror-line X from aligned unmirrored data at B=0 (or nearest B)
            x0_ref = float(x_ref_mirror_line)
            z0_ref = float(z_avg_raw[zero_idx])

            # Re-reference cartesian trajectory so first B=0 point is (0,0)
            x_avg = x_avg_raw - x0_ref
            z_avg = z_avg_raw - z0_ref

            # Strict planar bending definition:
            # r is the signed transverse/radial deflection, i.e. r = x (not Euclidean radius).
            # Radial zero is defined by the mirror line (mean unmirrored aligned X across all B and both orientations).
            r_coords_raw = x_avg_raw.copy()
            r_coords = x_avg.copy()

            # Axial coordinate for fitting/plots (re-referenced)
            z_coords = z_avg.copy()

            if save_plots:
                # Referenced X/Z plot for simplified trajectory
                plt.figure(figsize=(10, 8))
                plt.plot(x_avg, z_avg, 'o-', alpha=0.8, linewidth=2, markersize=6)
                plt.xlabel("X (mm, referenced to first B=0 point)")
                plt.ylabel("Z (mm, referenced to first B=0 point)")
                plt.title("04: Final Calibrated Locations (Referenced to first B=0 tip)")
                plt.axis("equal")
                plt.grid(True, alpha=0.3)
                plt.savefig("04_final_calibrated_locations.png", dpi=150, bbox_inches='tight')
                plt.close()

            print(f"Using {len(x_avg)} averaged data points for planar-coordinate conversion and cubic fitting")
            print(f"Zero reference index: {zero_idx} | B_ref = {delta_motor[zero_idx]:.6f}")
            print(f"Reference tip (raw): x_ref_mirror_line = {x0_ref:.6f} mm, z0 = {z0_ref:.6f} mm")
            print(
                "Radial reference source: mirror line = mean unmirrored aligned X across all measured B values "
                f"and both orientations (B range {b_ref_mirror_line_min:.6f} to {b_ref_mirror_line_max:.6f}, "
                f"B mean {b_ref_mirror_line_mean:.6f})"
            )
            print("Reference tip (shifted): x=0.000000 mm, z=0.000000 mm, r=0.000000 mm")
            print("Referenced ranges:")
            print(f"  X range: {x_avg.min():.3f} to {x_avg.max():.3f} mm")
            print(f"  Z range: {z_coords.min():.3f} to {z_coords.max():.3f} mm")
            print(f"  R (signed planar X deflection) range: {r_coords.min():.3f} to {r_coords.max():.3f} mm")
            print(f"  b_pull range: {delta_motor.min():.3f} to {delta_motor.max():.3f}")

            # Fit cubic polynomials: r = f(B_motor) and z = f(B_motor)
            print("\nFitting cubic equations:")
            print("r = f(B_motor) - signed planar transverse deflection (r = x) as function of B-axis translation")
            print("z = f(B_motor) - axial position as function of B-axis translation")

            # Fit cubic polynomial for r-coordinate
            r_coefficients = np.polyfit(delta_motor, r_coords, 3)
            r_polynomial = np.poly1d(r_coefficients)
            r_predicted = r_polynomial(delta_motor)

            # Fit cubic polynomial for z-coordinate
            z_coefficients = np.polyfit(delta_motor, z_coords, 3)
            z_polynomial = np.poly1d(z_coefficients)
            z_predicted = z_polynomial(delta_motor)

            tip_angle_coefficients = None
            tip_angle_predicted = None
            tip_angle_r2 = float("nan")
            tip_angle_equation = None

            if has_tip_angle and np.all(np.isfinite(tip_angle_avg)):
                tip_angle_coefficients = np.polyfit(delta_motor, tip_angle_avg, 3)
                tip_angle_polynomial = np.poly1d(tip_angle_coefficients)
                tip_angle_predicted = tip_angle_polynomial(delta_motor)

            # Calculate R² scores
            def r2_score_safe(y_true, y_pred):
                """Calculate R² score with safety checks for edge cases"""
                ss_res = np.sum((y_true - y_pred) ** 2)
                ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
                if ss_tot == 0:
                    return 1.0 if ss_res == 0 else float('-inf')
                return 1 - (ss_res / ss_tot)

            r_r2 = r2_score_safe(r_coords, r_predicted)
            z_r2 = r2_score_safe(z_coords, z_predicted)

            if tip_angle_predicted is not None:
                tip_angle_r2 = r2_score_safe(tip_angle_avg, tip_angle_predicted)

            print(f"\nCubic polynomial fit results:")
            print(f"R-coordinate (signed planar X deflection) R² score: {r_r2:.6f}")
            print(f"Z-coordinate R² score: {z_r2:.6f}")
            if tip_angle_predicted is not None:
                print(f"Tip-angle R² score: {tip_angle_r2:.6f}")

            # Create polynomial equation strings
            def format_cubic_equation(coeffs, var_name):
                """Format cubic polynomial coefficients into readable equation"""
                a, b, c, d = coeffs
                equation = f"{var_name} = {a:.6f}*b³"
                if b >= 0:
                    equation += f" + {b:.6f}*b²"
                else:
                    equation += f" - {abs(b):.6f}*b²"
                if c >= 0:
                    equation += f" + {c:.6f}*b"
                else:
                    equation += f" - {abs(c):.6f}*b"
                if d >= 0:
                    equation += f" + {d:.6f}"
                else:
                    equation += f" - {abs(d):.6f}"
                return equation

            r_equation = format_cubic_equation(r_coefficients, "r")
            z_equation = format_cubic_equation(z_coefficients, "z")
            if tip_angle_coefficients is not None:
                tip_angle_equation = format_cubic_equation(tip_angle_coefficients, "tip_angle_deg")

            print(f"\nCubic equations:")
            print(f"R: {r_equation}")
            print(f"Z: {z_equation}")
            if tip_angle_equation is not None:
                print(f"Tip angle: {tip_angle_equation}")

            # Create comprehensive results DataFrame
            coefficients_data = []

            # R-coordinate coefficients (signed planar X deflection, r = x)
            for i, (power, coef) in enumerate(zip(['b^3', 'b^2', 'b^1', 'b^0'], r_coefficients)):
                coefficients_data.append({
                    'Coordinate': 'R (signed planar X deflection, r=x)',
                    'Term': power,
                    'Coefficient': coef,
                    'Description': f'Coefficient for {power} term in planar transverse-deflection equation (r=x)'
                })

            # Z-coordinate coefficients
            for i, (power, coef) in enumerate(zip(['b^3', 'b^2', 'b^1', 'b^0'], z_coefficients)):
                coefficients_data.append({
                    'Coordinate': 'Z (axial)',
                    'Term': power,
                    'Coefficient': coef,
                    'Description': f'Coefficient for {power} term in axial equation'
                })

            if tip_angle_coefficients is not None:
                for i, (power, coef) in enumerate(zip(['b^3', 'b^2', 'b^1', 'b^0'], tip_angle_coefficients)):
                    coefficients_data.append({
                        'Coordinate': 'Tip angle (deg vs vertical)',
                        'Term': power,
                        'Coefficient': coef,
                        'Description': f'Coefficient for {power} term in tip-angle equation'
                    })

            df_coefficients = pd.DataFrame(coefficients_data)

            # Create fit quality DataFrame
            fit_info_data = [
                {'Metric': 'R_coordinate_R_squared', 'Value': r_r2, 'Description': 'R² score for planar transverse-deflection coordinate fit (r=x)'},
                {'Metric': 'Z_coordinate_R_squared', 'Value': z_r2, 'Description': 'R² score for axial coordinate fit'},
                {'Metric': 'Max_R_Error', 'Value': np.max(np.abs(r_predicted - r_coords)), 'Description': 'Maximum absolute error in planar transverse-deflection prediction (r=x)'},
                {'Metric': 'Max_Z_Error', 'Value': np.max(np.abs(z_predicted - z_coords)), 'Description': 'Maximum absolute error in axial prediction'},
                {'Metric': 'Mean_R_Error', 'Value': np.mean(np.abs(r_predicted - r_coords)), 'Description': 'Mean absolute error in planar transverse-deflection prediction (r=x)'},
                {'Metric': 'Mean_Z_Error', 'Value': np.mean(np.abs(z_predicted - z_coords)), 'Description': 'Mean absolute error in axial prediction'}
            ]
            if tip_angle_predicted is not None:
                fit_info_data.extend([
                    {'Metric': 'Tip_Angle_R_squared', 'Value': tip_angle_r2, 'Description': 'R² score for tip-angle fit'},
                    {'Metric': 'Max_Tip_Angle_Error_deg', 'Value': np.max(np.abs(tip_angle_predicted - tip_angle_avg)), 'Description': 'Maximum absolute tip-angle prediction error'},
                    {'Metric': 'Mean_Tip_Angle_Error_deg', 'Value': np.mean(np.abs(tip_angle_predicted - tip_angle_avg)), 'Description': 'Mean absolute tip-angle prediction error'},
                ])
            df_fit_info = pd.DataFrame(fit_info_data)

            # Create equations DataFrame
            equations_data = [
                {'Coordinate': 'R (signed planar X deflection, r=x)', 'Cubic_Equation': r_equation},
                {'Coordinate': 'Z (axial)', 'Cubic_Equation': z_equation}
            ]
            if tip_angle_equation is not None:
                equations_data.append({'Coordinate': 'Tip angle (deg vs vertical)', 'Cubic_Equation': tip_angle_equation})
            df_equations = pd.DataFrame(equations_data)

            # Plotting
            if save_plots:
                plt.figure(figsize=(15, 10))

                # Plot 1: Referenced X,Z trajectory
                plt.subplot(2, 3, 1)
                plt.plot(x_avg, z_avg, 'o-', linewidth=2, markersize=6)
                plt.xlabel('X transverse deflection (mm, ref. B=0)')
                plt.ylabel('Z axial position (mm, ref. B=0)')
                plt.title('Referenced Planar Trajectory (X vs Z)')
                plt.axis('equal')
                plt.grid(True, alpha=0.3)

                # Plot 2: Planar bending coordinates (r = x, z)
                plt.subplot(2, 3, 2)
                plt.plot(r_coords, z_coords, 'o-', linewidth=2, markersize=6, color='green')
                plt.xlabel('R = X (signed transverse deflection, mm)')
                plt.ylabel('Z (axial) Position (mm)')
                plt.title('Planar Bending Coordinates (r=x, z)')
                plt.grid(True, alpha=0.3)

                # Plot 3: R vs B Motor
                plt.subplot(2, 3, 3)
                plt.plot(delta_motor, r_coords, 'o', linewidth=2, markersize=6, label='Measured')
                plt.plot(delta_motor, r_predicted, 's-', linewidth=2, markersize=4, color='red', label='Cubic Fit')
                plt.xlabel('B Motor Position')
                plt.ylabel('R = X (signed transverse deflection, mm)')
                plt.title(f'Planar X Deflection vs Motor (R² = {r_r2:.4f})')
                plt.grid(True, alpha=0.3)
                plt.legend()

                # Plot 4: Z vs B Motor
                plt.subplot(2, 3, 4)
                plt.plot(delta_motor, z_coords, 'o', linewidth=2, markersize=6, label='Measured')
                plt.plot(delta_motor, z_predicted, 's-', linewidth=2, markersize=4, color='red', label='Cubic Fit')
                plt.xlabel('B Motor Position')
                plt.ylabel('Z (axial) Position (mm)')
                plt.title(f'Axial Position vs Motor (R² = {z_r2:.4f})')
                plt.grid(True, alpha=0.3)
                plt.legend()

                # Plot 5: Residuals for R (planar X deflection)
                plt.subplot(2, 3, 5)
                r_residuals = r_coords - r_predicted
                plt.plot(delta_motor, r_residuals, 'o-', linewidth=2, markersize=6, color='purple')
                plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                plt.xlabel('B Motor Position')
                plt.ylabel('R = X Residuals (mm)')
                plt.title('Planar X Deflection Fit Residuals')
                plt.grid(True, alpha=0.3)

                # Plot 6: Residuals for Z
                plt.subplot(2, 3, 6)
                z_residuals = z_coords - z_predicted
                plt.plot(delta_motor, z_residuals, 'o-', linewidth=2, markersize=6, color='orange')
                plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
                plt.xlabel('B Motor Position')
                plt.ylabel('Z Residuals (mm)')
                plt.title('Axial Fit Residuals')
                plt.grid(True, alpha=0.3)

                plt.tight_layout()
                plt.savefig("10_polar_cubic_fits.png", dpi=150, bbox_inches='tight')
                plt.close()
                print("Added plot: 10_polar_cubic_fits.png")

                if tip_angle_predicted is not None:
                    sort_idx = np.argsort(delta_motor)
                    plt.figure(figsize=(9, 6))
                    plt.plot(delta_motor[sort_idx], tip_angle_avg[sort_idx], 'o', markersize=6, label='Measured')
                    plt.plot(delta_motor[sort_idx], tip_angle_predicted[sort_idx], 's-', linewidth=2, markersize=4, color='red', label='Cubic Fit')
                    plt.xlabel('B Motor Position')
                    plt.ylabel('Tip Angle vs Vertical (deg)')
                    plt.title(f'Tip Angle vs B Pull (R² = {tip_angle_r2:.4f})')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    plt.tight_layout()
                    plt.savefig("11_tip_angle_vs_b_pull.png", dpi=150, bbox_inches='tight')
                    plt.close()
                    print("Added plot: 11_tip_angle_vs_b_pull.png")

            # Save to Excel
            excel_filename = f"{robot_name}_cubic_polar_coefficients.xlsx"
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                df_coefficients.to_excel(writer, sheet_name='Cubic_Coefficients', index=False)
                df_fit_info.to_excel(writer, sheet_name='Fit_Quality', index=False)
                df_equations.to_excel(writer, sheet_name='Equations', index=False)

                df_raw_data = pd.DataFrame({
                    'X_mm_raw': x_avg_raw,
                    'Z_mm_raw': z_avg_raw,
                    'R_planarX_mm_raw': r_coords_raw,
                    'X_mm_ref_B0': x_avg,
                    'Z_mm_ref_B0': z_avg,
                    'R_planarX_mm_ref_B0': r_coords,
                    'X_ref_mirror_line_mm': np.full_like(delta_motor, x_ref_mirror_line, dtype=float),
                    'B_Pull': delta_motor,
                    'Tip_Angle_deg': tip_angle_avg if tip_angle_avg is not None else np.full_like(delta_motor, np.nan, dtype=float)
                })
                df_raw_data.to_excel(writer, sheet_name='Raw_Data', index=False)

                df_validation = pd.DataFrame({
                    'B_Pull': delta_motor,
                    'Actual_R_planarX_ref_B0': r_coords,
                    'Predicted_R_planarX_ref_B0': r_predicted,
                    'R_Error': r_predicted - r_coords,
                    'X_ref_mirror_line_mm': np.full_like(delta_motor, x_ref_mirror_line, dtype=float),
                    'Actual_Z_ref_B0': z_coords,
                    'Predicted_Z_ref_B0': z_predicted,
                    'Z_Error': z_predicted - z_coords,
                    'Actual_Tip_Angle_deg': tip_angle_avg if tip_angle_avg is not None else np.full_like(delta_motor, np.nan, dtype=float),
                    'Predicted_Tip_Angle_deg': tip_angle_predicted if tip_angle_predicted is not None else np.full_like(delta_motor, np.nan, dtype=float),
                    'Tip_Angle_Error_deg': (tip_angle_predicted - tip_angle_avg) if tip_angle_predicted is not None else np.full_like(delta_motor, np.nan, dtype=float)
                })
                df_validation.to_excel(writer, sheet_name='Validation', index=False)

            print(f"\nCubic polar coefficients saved to: {excel_filename}")
            print(f"Excel file contains 5 sheets: Cubic_Coefficients, Fit_Quality, Equations, Raw_Data, and Validation")

            # Save coefficients as numpy arrays
            np.save(f"{robot_name}_r_cubic_coefficients.npy", r_coefficients)
            np.save(f"{robot_name}_z_cubic_coefficients.npy", z_coefficients)
            print(f"R cubic coefficients saved to: {robot_name}_r_cubic_coefficients.npy")
            print(f"Z cubic coefficients saved to: {robot_name}_z_cubic_coefficients.npy")

            if tip_angle_coefficients is not None:
                np.save(f"{robot_name}_tip_angle_cubic_coefficients.npy", tip_angle_coefficients)
                print(f"Tip-angle cubic coefficients saved to: {robot_name}_tip_angle_cubic_coefficients.npy")

            # Save the cubic model data
            cubic_model_data = {
                'r_coefficients': r_coefficients,
                'z_coefficients': z_coefficients,
                'r_r2': r_r2,
                'z_r2': z_r2,
                'b_motor_range': [float(np.min(delta_motor)), float(np.max(delta_motor))],
                'r_equation': r_equation,
                'r_definition': 'signed planar transverse deflection in strict planar bending calibration (r = x in the B0-referenced frame)',
                'z_equation': z_equation,
                'tip_angle_coefficients': tip_angle_coefficients,
                'tip_angle_r2': tip_angle_r2,
                'tip_angle_equation': tip_angle_equation,
                'reference_point': {
                    'b_ref': float(delta_motor[zero_idx]),
                    'x0_ref_mm': x0_ref,
                    'z0_ref_mm': z0_ref,
                    'radial_reference_b_mean_mm': b_ref_mirror_line_mean,
                    'radial_reference_b_min_mm': b_ref_mirror_line_min,
                    'radial_reference_b_max_mm': b_ref_mirror_line_max,
                    'x_ref_mirror_line_mm': x_ref_mirror_line,
                    'reference_definition': 'z is referenced using the first averaged B=0.0 point (or nearest B in averaged data); radial r is defined as signed planar transverse deflection (r=x) and referenced to the mirror line, computed as the mean unmirrored aligned X across all measured B values and both orientations in the aligned frame'
                }
            }
            with open(f"{robot_name}_cubic_polar_calibration.pkl", "wb") as f:
                pickle.dump(cubic_model_data, f)
            print(f"Complete cubic model saved to: {robot_name}_cubic_polar_calibration.pkl")

            tip_angle_coeffs_list = tip_angle_coefficients.tolist() if tip_angle_coefficients is not None else None

            # Create prediction functions
            def predict_tip_position(b_motor_pos):
                """
                Predict tip position in planar bending coordinates (r=x, z) given B motor position.
                Parameters:
                    b_motor_pos: float or array-like, B motor position(s)
                Returns:
                    r, z: predicted planar coordinates where r is signed transverse deflection (r=x) and z is axial position, referenced to the first measured B=0.0 tip location
                """
                b_motor = np.atleast_1d(b_motor_pos)
                # Calculate r and z using cubic polynomials
                r_pred = np.polyval(r_coefficients, b_motor)
                z_pred = np.polyval(z_coefficients, b_motor)
                return r_pred, z_pred

            def predict_tip_angle(b_motor_pos):
                """Predict tip angle (deg vs vertical) from B motor position."""
                if tip_angle_coefficients is None:
                    return None
                b_motor = np.atleast_1d(b_motor_pos)
                return np.polyval(tip_angle_coefficients, b_motor)

            def predict_tip_position_cartesian(b_motor_pos):
                """
                Predict tip position in Cartesian coordinates given B motor position.
                Parameters:
                    b_motor_pos: float or array-like, B motor position(s)
                Returns:
                    x, y: predicted Cartesian coordinates referenced to the first measured B=0.0 tip location
                """
                r_pred, z_pred = predict_tip_position(b_motor_pos)
                # For this application, assuming x = r and y = z
                # (adjust this conversion based on your coordinate system)
                x_pred = r_pred
                y_pred = z_pred
                return x_pred, y_pred

            # Save prediction functions
            function_code = f'''import numpy as np
# Cubic polynomial coefficients
r_coefficients = {r_coefficients.tolist()}
z_coefficients = {z_coefficients.tolist()}
tip_angle_coefficients = {tip_angle_coeffs_list}

def predict_tip_position(b_motor_pos):
    """
    Predict tip position in planar bending coordinates (r=x, z) given B motor position.

    Parameters:
        b_motor_pos: float or array-like, B motor position(s)

    Returns:
        r, z: predicted planar coordinates where r is signed transverse deflection (r=x) and z is axial position, referenced to the first measured B=0.0 tip location

    Equations:
        {r_equation}
        {z_equation}
    """
    b_motor = np.atleast_1d(b_motor_pos)
    # Calculate r and z using cubic polynomials
    r_pred = np.polyval(r_coefficients, b_motor)
    z_pred = np.polyval(z_coefficients, b_motor)
    return r_pred, z_pred

def predict_tip_angle(b_motor_pos):
    """
    Predict tip angle (deg vs vertical) given B motor position.
    Returns None if angle calibration is unavailable.
    """
    if tip_angle_coefficients is None:
        return None
    b_motor = np.atleast_1d(b_motor_pos)
    return np.polyval(tip_angle_coefficients, b_motor)

def predict_tip_position_cartesian(b_motor_pos):
    """
    Predict tip position in Cartesian coordinates given B motor position.

    Parameters:
        b_motor_pos: float or array-like, B motor position(s)

    Returns:
        x, y: predicted Cartesian coordinates referenced to the first measured B=0.0 tip location
    """
    r_pred, z_pred = predict_tip_position(b_motor_pos)
    # In strict planar bending calibration, r is already the referenced X deflection.
    x_pred = r_pred  # or r_pred * cos(theta) if you have angular component
    y_pred = z_pred  # axial coordinate
    return x_pred, y_pred

# Example usage:
# r, z = predict_tip_position(-1.2)           # Predict planar coordinates (r=x, z)
# x, y = predict_tip_position_cartesian(-1.2) # Predict Cartesian coordinates
# angle_deg = predict_tip_angle(-1.2)
'''
            with open(f"{robot_name}_cubic_prediction_functions.py", "w") as f:
                f.write(function_code)
            print(f"Prediction functions saved to: {robot_name}_cubic_prediction_functions.py")

            print("\n" + "="*60)
            print("PLANAR-BENDING CUBIC FITTING COMPLETE!")
            print("="*60)
            print(f"Planar transverse-deflection equation (r=x) R² score: {r_r2:.6f}")
            print(f"Axial equation R² score: {z_r2:.6f}")
            if tip_angle_equation is not None:
                print(f"Tip-angle equation R² score: {tip_angle_r2:.6f}")
            print(f"B motor fit range: {float(np.min(delta_motor)):.6f} to {float(np.max(delta_motor)):.6f}")

            print(f"\nCubic equations fitted:")
            print(f"R: {r_equation}")
            print(f"Z: {z_equation}")
            if tip_angle_equation is not None:
                print(f"Tip angle: {tip_angle_equation}")

            print(f"\nSaved files:")
            print(f" - {robot_name}_cubic_polar_calibration.pkl (complete model)")
            print(f" - {robot_name}_cubic_prediction_functions.py (ready-to-use functions)")
            print(f" - {robot_name}_r_cubic_coefficients.npy & {robot_name}_z_cubic_coefficients.npy")
            print(f" - {robot_name}_cubic_polar_coefficients.xlsx (detailed Excel file)")
            print(f" - 10_polar_cubic_fits.png (comprehensive plots)")
            if tip_angle_equation is not None:
                print(f" - {robot_name}_tip_angle_cubic_coefficients.npy")
                print(f" - 11_tip_angle_vs_b_pull.png")

            # Step 9: Save final results
            print("Saving final calibration results...")

            # Save the cubic model and function
            cubic_calibration_data = {
                'r_coefficients': r_coefficients,
                'z_coefficients': z_coefficients,
                'r_r2': r_r2,
                'z_r2': z_r2,
                'b_motor_range': [float(np.min(delta_motor)), float(np.max(delta_motor))],
                'r_equation': r_equation,
                'r_definition': 'signed planar transverse deflection in strict planar bending calibration (r = x in the B0-referenced frame)',
                'z_equation': z_equation,
                'tip_angle_coefficients': tip_angle_coefficients,
                'tip_angle_r2': tip_angle_r2,
                'tip_angle_equation': tip_angle_equation,
                'reference_point': {
                    'b_ref': float(delta_motor[zero_idx]),
                    'x0_ref_mm': x0_ref,
                    'z0_ref_mm': z0_ref,
                    'radial_reference_b_mean_mm': b_ref_mirror_line_mean,
                    'radial_reference_b_min_mm': b_ref_mirror_line_min,
                    'radial_reference_b_max_mm': b_ref_mirror_line_max,
                    'x_ref_mirror_line_mm': x_ref_mirror_line,
                    'reference_definition': 'z is referenced using the first averaged B=0.0 point (or nearest B in averaged data); radial r is defined as signed planar transverse deflection (r=x) and referenced to the mirror line, computed as the mean unmirrored aligned X across all measured B values and both orientations in the aligned frame'
                }
            }
            with open(f"{robot_name}_cubic_calibration.pkl", "wb") as f:
                pickle.dump(cubic_calibration_data, f)
            print(f"\nCubic calibration saved to {robot_name}_cubic_calibration.pkl")

            # Create prediction functions that can be called with B motor positions
            def predict_tip_position_from_b(b_motor_pos):
                """
                Predict tip position in planar bending coordinates (r=x, z) given B motor position.
                Parameters:
                    b_motor_pos: float or array-like, B motor position(s)
                Returns:
                    r, z: predicted planar coordinates where r is signed transverse deflection (r=x) and z is axial position, referenced to the first measured B=0.0 tip location
                """
                # Ensure inputs are arrays
                b_motor = np.atleast_1d(b_motor_pos)
                # Calculate r and z using cubic polynomials
                r_pred = np.polyval(r_coefficients, b_motor)
                z_pred = np.polyval(z_coefficients, b_motor)
                return r_pred, z_pred

            def predict_tip_angle_from_b(b_motor_pos):
                if tip_angle_coefficients is None:
                    return None
                b_motor = np.atleast_1d(b_motor_pos)
                return np.polyval(tip_angle_coefficients, b_motor)

            def predict_cartesian_from_b(b_motor_pos):
                """
                Predict tip position in Cartesian coordinates given B motor position.
                Parameters:
                    b_motor_pos: float or array-like, B motor position(s)
                Returns:
                    x, y: predicted Cartesian coordinates referenced to the first measured B=0.0 tip location
                """
                r_pred, z_pred = predict_tip_position_from_b(b_motor_pos)
                # Convert to Cartesian (adjust based on your coordinate system)
                x_pred = r_pred  # or use trigonometric conversion if needed
                y_pred = z_pred
                return x_pred, y_pred

            # Save the function definition as a string
            function_code = f'''import numpy as np

            # Cubic polynomial coefficients from calibration
            r_coefficients = {r_coefficients.tolist()}
            z_coefficients = {z_coefficients.tolist()}
            tip_angle_coefficients = {tip_angle_coeffs_list}

            # Cubic equations:
            # R: {r_equation}
            # Z: {z_equation}
            # Tip Angle: {tip_angle_equation if tip_angle_equation is not None else "N/A"}

            def predict_tip_position_from_b(b_motor_pos):
                """
                Predict tip position in planar bending coordinates (r=x, z) given B motor position.

                Parameters:
                    b_motor_pos: float or array-like, B motor position(s)

                Returns:
                    r, z: predicted planar coordinates where r is signed transverse deflection (r=x) and z is axial position, referenced to the first measured B=0.0 tip location
                """
                b_motor = np.atleast_1d(b_motor_pos)
                # Calculate r and z using cubic polynomials
                r_pred = np.polyval(r_coefficients, b_motor)
                z_pred = np.polyval(z_coefficients, b_motor)
                return r_pred, z_pred

            def predict_tip_angle_from_b(b_motor_pos):
                """
                Predict tip angle (deg vs vertical) given B motor position.
                Returns None if angle calibration is unavailable.
                """
                if tip_angle_coefficients is None:
                    return None
                b_motor = np.atleast_1d(b_motor_pos)
                return np.polyval(tip_angle_coefficients, b_motor)

            def predict_cartesian_from_b(b_motor_pos):
                """
                Predict tip position in Cartesian coordinates given B motor position.

                Parameters:
                    b_motor_pos: float or array-like, B motor position(s)

                Returns:
                    x, y: predicted Cartesian coordinates referenced to the first measured B=0.0 tip location
                """
                r_pred, z_pred = predict_tip_position_from_b(b_motor_pos)
                # Convert to Cartesian
                x_pred = r_pred
                y_pred = z_pred
                return x_pred, y_pred

            # Backward-compatible aliases
            def predict_tip_position_from_delta(delta_motor_pos):
                return predict_tip_position_from_b(delta_motor_pos)

            def predict_cartesian_from_delta(delta_motor_pos):
                return predict_cartesian_from_b(delta_motor_pos)

            def predict_tip_angle_from_delta(delta_motor_pos):
                return predict_tip_angle_from_b(delta_motor_pos)

            # Example usage:
            # r, z = predict_tip_position_from_b(-1.2)
            # x, y = predict_cartesian_from_b(-1.2)
            # angle_deg = predict_tip_angle_from_b(-1.2)
            '''
            with open(f"{robot_name}_cubic_prediction_functions.py", "w") as f:
                f.write(function_code)
            print(f"Prediction function saved to {robot_name}_cubic_prediction_functions.py")

            # Save coefficient arrays for easy loading
            np.save(f"{robot_name}_r_cubic_coefficients.npy", r_coefficients)
            np.save(f"{robot_name}_z_cubic_coefficients.npy", z_coefficients)
            if tip_angle_coefficients is not None:
                np.save(f"{robot_name}_tip_angle_cubic_coefficients.npy", tip_angle_coefficients)

            print(f"Cubic coefficients saved to {robot_name}_r_cubic_coefficients.npy and {robot_name}_z_cubic_coefficients.npy")

            print("\n" + "="*60)
            print("POSTPROCESSING COMPLETE!")
            print("="*60)
            print(f"Final cubic calibration completed")
            print(f"Planar transverse-deflection fit (r=x) R² score: {r_r2:.6f}")
            print(f"Axial fit R² score: {z_r2:.6f}")
            if tip_angle_equation is not None:
                print(f"Tip-angle fit R² score: {tip_angle_r2:.6f}")
            print(f"B motor fit range: {float(np.min(delta_motor)):.6f} to {float(np.max(delta_motor)):.6f}")

            print(f"\nSaved files:")
            print(f" - {robot_name}_cubic_calibration.pkl (complete calibration data)")
            print(f" - {robot_name}_cubic_prediction_functions.py (ready-to-use Python functions)")
            print(f" - {robot_name}_r_cubic_coefficients.npy & {robot_name}_z_cubic_coefficients.npy")
            print(f" - {robot_name}_cubic_polar_coefficients.xlsx (Excel file with detailed data)")
            print(f" - 10_polar_cubic_fits.png (comprehensive visualization)")
            if tip_angle_equation is not None:
                print(f" - {robot_name}_tip_angle_cubic_coefficients.npy")
                print(f" - 11_tip_angle_vs_b_pull.png")

            # Print the cubic equations for reference
            print(f"\nCubic equations fitted:")
            print(f"Planar transverse deflection (r=x): {r_equation}")
            print(f"Axial: {z_equation}")
            if tip_angle_equation is not None:
                print(f"Tip angle: {tip_angle_equation}")

            # Print ready-to-use function example
            print(f"\nReady-to-use functions:")
            print("# Load the calibration:")
            print(f"# import pickle")
            print(f"# with open('{robot_name}_cubic_calibration.pkl', 'rb') as f:")
            print(f"# calibration = pickle.load(f)")
            print("# ")
            print("# Use the functions:")
            print("# r, z = predict_tip_position_from_b(-1.2)")
            print("# x, y = predict_cartesian_from_b(-1.2)")
            print("# angle_deg = predict_tip_angle_from_b(-1.2)")

            # Step 10: Export calibration data for G-code generation
            print("Exporting calibration data for G-code generation...")

            # Create comprehensive Excel export
            calibration_summary = {
                'Calibration_Info': pd.DataFrame([
                    {'Parameter': 'Robot_Name', 'Value': robot_name},
                    {'Parameter': 'Calibration_Date', 'Value': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')},
                    {'Parameter': 'R_Equation_R_Squared', 'Value': r_r2},
                    {'Parameter': 'Z_Equation_R_Squared', 'Value': z_r2},
                    {'Parameter': 'Tip_Angle_Equation_R_Squared', 'Value': tip_angle_r2},
                    {'Parameter': 'B_Motor_Min', 'Value': float(np.min(delta_motor))},
                    {'Parameter': 'B_Motor_Max', 'Value': float(np.max(delta_motor))},
                    {'Parameter': 'Pixel_to_MM_Scale', 'Value': width_in_mm/width_in_pixels},
                ]),
                'Cubic_Coefficients': pd.DataFrame([
                    {'Coordinate': 'R (signed planar X deflection, r=x)', 'Power': 3, 'Coefficient': r_coefficients[0]},
                    {'Coordinate': 'R (signed planar X deflection, r=x)', 'Power': 2, 'Coefficient': r_coefficients[1]},
                    {'Coordinate': 'R (signed planar X deflection, r=x)', 'Power': 1, 'Coefficient': r_coefficients[2]},
                    {'Coordinate': 'R (signed planar X deflection, r=x)', 'Power': 0, 'Coefficient': r_coefficients[3]},
                    {'Coordinate': 'Z', 'Power': 3, 'Coefficient': z_coefficients[0]},
                    {'Coordinate': 'Z', 'Power': 2, 'Coefficient': z_coefficients[1]},
                    {'Coordinate': 'Z', 'Power': 1, 'Coefficient': z_coefficients[2]},
                    {'Coordinate': 'Z', 'Power': 0, 'Coefficient': z_coefficients[3]},
                    {'Coordinate': 'Tip_Angle_deg', 'Power': 3, 'Coefficient': tip_angle_coefficients[0] if tip_angle_coefficients is not None else np.nan},
                    {'Coordinate': 'Tip_Angle_deg', 'Power': 2, 'Coefficient': tip_angle_coefficients[1] if tip_angle_coefficients is not None else np.nan},
                    {'Coordinate': 'Tip_Angle_deg', 'Power': 1, 'Coefficient': tip_angle_coefficients[2] if tip_angle_coefficients is not None else np.nan},
                    {'Coordinate': 'Tip_Angle_deg', 'Power': 0, 'Coefficient': tip_angle_coefficients[3] if tip_angle_coefficients is not None else np.nan},
                ]),
                'Working_Ranges': pd.DataFrame([
                    {'Parameter': 'B_Motor_Min', 'Value': delta_motor.min()},
                    {'Parameter': 'B_Motor_Max', 'Value': delta_motor.max()},
                    {'Parameter': 'R_planarX_Min_mm', 'Value': r_coords.min()},
                    {'Parameter': 'R_planarX_Max_mm', 'Value': r_coords.max()},
                    {'Parameter': 'Z_Min_mm', 'Value': z_coords.min()},
                    {'Parameter': 'Z_Max_mm', 'Value': z_coords.max()},
                    {'Parameter': 'Tip_Angle_Min_deg', 'Value': tip_angle_avg.min() if tip_angle_avg is not None else np.nan},
                    {'Parameter': 'Tip_Angle_Max_deg', 'Value': tip_angle_avg.max() if tip_angle_avg is not None else np.nan},
                ])
            }

            # Export to Excel
            excel_filename = f"{robot_name}_calibration_for_gcode.xlsx"
            with pd.ExcelWriter(excel_filename, engine='openpyxl') as writer:
                for sheet_name, df in calibration_summary.items():
                    df.to_excel(writer, sheet_name=sheet_name, index=False)

            # Export JSON for easy Python loading
            b_min = float(np.min(delta_motor))
            b_max = float(np.max(delta_motor))
            b_cmd_min = min(b_min, b_max)
            b_cmd_max = max(b_min, b_max)

            gcode_calibration_data = {
                'robot_name': robot_name,
                'calibration_date': pd.Timestamp.now().isoformat(),
                'reference_frame': {
                    'type': 'B0_tip_referenced',
                    'b_reference': float(delta_motor[zero_idx]),
                    'x0_ref_mm': x0_ref,
                    'z0_ref_mm': z0_ref,
                    'radial_reference_b_mean_mm': b_ref_mirror_line_mean,
                    'radial_reference_b_min_mm': b_ref_mirror_line_min,
                    'radial_reference_b_max_mm': b_ref_mirror_line_max,
                    'x_ref_mirror_line_mm': x_ref_mirror_line,
                    'r_definition': 'signed planar transverse deflection (r = x) in strict planar bending calibration',
                    'notes': 'Z is shifted so the first averaged B=0.0 tip is z=0 (or nearest B fallback). Radial r is defined as signed planar transverse deflection (r=x) and referenced to the mirror line, computed as the mean unmirrored aligned X across all measured B values and both orientations in the aligned frame, so r=0 at that radial reference.'
                },
                'cubic_coefficients': {
                    'r_coeffs': r_coefficients.tolist(),  # [u³, u², u¹, u⁰]
                    'z_coeffs': z_coefficients.tolist(),
                    'tip_angle_coeffs': tip_angle_coefficients.tolist() if tip_angle_coefficients is not None else None,
                    'r_equation': r_equation,
                    'r_definition': 'signed planar transverse deflection (r = x), referenced to the mirror line = mean unmirrored aligned X across all measured B values and both orientations in the aligned frame',
                    'z_equation': z_equation,
                    'tip_angle_equation': tip_angle_equation,
                    'r_r_squared': float(r_r2),
                    'z_r_squared': float(z_r2),
                    'tip_angle_r_squared': float(tip_angle_r2) if np.isfinite(tip_angle_r2) else None
                },
                'motor_setup': {
                    # Current machine setup: single pull motor on B axis, homed at 0.
                    'b_motor_axis': 'B',
                    'b_motor_home_position': 0.0,
                    'b_motor_position_range': [b_cmd_min, b_cmd_max],
                    'rotation_axis': 'C',
                    'rotation_axis_180_deg': 180.0,
                    'horizontal_axis': 'X',
                    'vertical_axis': 'Z',
                    'depth_axis': 'Y',
                },
                'working_envelope': {
                    'radius_range_mm': [float(r_coords.min()), float(r_coords.max())],
                    'radius_range_definition': 'signed planar transverse deflection range (r = x), not Euclidean radius',
                    'z_range_mm': [float(z_coords.min()), float(z_coords.max())],
                    'tip_angle_range_deg': [float(tip_angle_avg.min()), float(tip_angle_avg.max())] if tip_angle_avg is not None else None,
                    'max_radius_mm': float(r_coords.max())
                },
                'duet_axis_mapping': {
                    # Canonical mapping for current setup
                    'horizontal_axis': 'X',
                    'vertical_axis': 'Z',
                    'depth_axis': 'Y',
                    'pull_axis': 'B',
                    'rotation_axis': 'C',
                    'extruder_axis': 'U',
                }
            }

            with open(f"{robot_name}_gcode_calibration.json", "w") as f:
                json.dump(gcode_calibration_data, f, indent=2)

            print(f"\nCalibration data exported for G-code generation:")
            print(f" - {excel_filename} (human-readable)")
            print(f" - {robot_name}_gcode_calibration.json (for Python G-code class)")

        finally:
            # Return to original directory
            os.chdir(original_dir)
            
