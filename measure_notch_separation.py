#!/usr/bin/env python3
"""
Interactive notch-separation measurement tool.

Folder structure expected:

root_folder/
  curl/
    0/
      image_001.png
      image_002.png
    30/
      image_001.png
  uncurl/
    0/
      image_001.png
    45/
      image_001.png

The first level is the comparison mode: curl and/or uncurl.
Each numeric subfolder inside curl/uncurl is interpreted as one orientation/curl angle.
Missing angle folders are allowed: a curl angle can exist without the matching uncurl angle, and vice versa.
For every image, the user clicks 2 points for each notch, from base to tip.
The Euclidean distance between the two clicked points is the notch separation.

Outputs are written into root_folder/notch_measurement_outputs/:
  - notch_measurement_progress.json
  - notch_measurements_long.csv
  - notch_measurements_summary_by_mode_angle.csv
  - notch_measurements_wide_mean.csv
  - notch_measurements.xlsx, if openpyxl is available
  - notch_separation_vs_position_light.png
  - notch_separation_vs_position_dark_transparent.png

GUI note:
  The image viewport is locked while clicking/selecting notch points.
  The viewport only changes when you use the crop/navigation controls: arrows, Q, or W.

Usage:
  python measure_notch_separation.py
  python measure_notch_separation.py /path/to/root_folder
  python measure_notch_separation.py /path/to/root_folder --normalization image_width
  python measure_notch_separation.py /path/to/root_folder --normalization manual_scale

Normalization modes:
  image_width      normalized distance = notch distance px / image width px
  image_height     normalized distance = notch distance px / image height px
  image_diagonal   normalized distance = notch distance px / image diagonal px
  manual_scale     click a 2-point reference length once per image; distances are divided by that length
  none             no normalization; normalized distance equals raw px distance

For images that only differ by resolution/resampling, image_width or image_diagonal usually works well.
For images with different crops or fields of view, use manual_scale.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

import tkinter as tk
from tkinter import filedialog, messagebox


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
DEFAULT_N_NOTCHES = 25

Point = Tuple[float, float]


@dataclass(frozen=True)
class ImageTask:
    mode: str
    angle_value: float
    angle_label: str
    angle_folder: Path
    image_path: Path
    rel_image_path: str


def parse_angle_folder_name(name: str) -> Optional[float]:
    try:
        return float(name)
    except ValueError:
        return None


def angle_label_from_value(value: float) -> str:
    if float(value).is_integer():
        return str(int(value))
    return f"{value:g}"


def find_angle_image_tasks(root_folder: Path) -> List[ImageTask]:
    root_folder = Path(root_folder).expanduser().resolve()
    if not root_folder.exists():
        raise FileNotFoundError(f"Root folder does not exist: {root_folder}")

    mode_dirs: List[Tuple[str, Path]] = []
    for mode in ("curl", "uncurl"):
        mode_dir = root_folder / mode
        if mode_dir.is_dir():
            mode_dirs.append((mode, mode_dir))

    if not mode_dirs:
        raise ValueError(
            f"No 'curl' or 'uncurl' subfolders found in {root_folder}.\n"
            "Expected folders like root/curl/0, root/curl/30, root/uncurl/0, etc."
        )

    tasks: List[ImageTask] = []
    for mode, mode_dir in mode_dirs:
        angle_dirs: List[Tuple[float, Path]] = []
        for p in mode_dir.iterdir():
            if not p.is_dir():
                continue
            angle = parse_angle_folder_name(p.name)
            if angle is not None:
                angle_dirs.append((angle, p))

        if not angle_dirs:
            print(f"Warning: no numeric angle subfolders found in {mode_dir}", file=sys.stderr)
            continue

        for angle_value, angle_dir in sorted(angle_dirs, key=lambda x: x[0]):
            image_paths = sorted(
                [p for p in angle_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS],
                key=lambda p: p.name.lower(),
            )
            if not image_paths:
                print(f"Warning: no images found in angle folder {angle_dir}", file=sys.stderr)
                continue

            for image_path in image_paths:
                rel = str(image_path.relative_to(root_folder))
                tasks.append(
                    ImageTask(
                        mode=mode,
                        angle_value=angle_value,
                        angle_label=angle_label_from_value(angle_value),
                        angle_folder=angle_dir,
                        image_path=image_path,
                        rel_image_path=rel,
                    )
                )

    if not tasks:
        raise ValueError("curl/uncurl folders were found, but no supported images were found inside numeric angle folders.")

    # Order all curl tasks first, then uncurl; within each mode, increasing angle and filename.
    mode_order = {"curl": 0, "uncurl": 1}
    tasks.sort(key=lambda t: (mode_order.get(t.mode, 99), t.angle_value, t.image_path.name.lower()))
    return tasks


def point_distance_px(p1: Point, p2: Point) -> float:
    return float(math.hypot(p2[0] - p1[0], p2[1] - p1[1]))


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        return float(value)
    except Exception:
        return default


def json_point(pt: Optional[Point]) -> Optional[List[float]]:
    if pt is None:
        return None
    return [float(pt[0]), float(pt[1])]


def tuple_point(value: Any) -> Optional[Point]:
    if value is None:
        return None
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    return (float(value[0]), float(value[1]))


def ensure_record(progress: Dict[str, Any], task: ImageTask, n_notches: int) -> Dict[str, Any]:
    images = progress.setdefault("images", {})
    rec = images.setdefault(task.rel_image_path, {})

    rec.setdefault("mode", task.mode)
    rec.setdefault("angle_value", task.angle_value)
    rec.setdefault("angle_label", task.angle_label)
    rec.setdefault("image_path", task.rel_image_path)
    rec.setdefault("normalization_px", None)
    rec.setdefault("scale_p1", None)
    rec.setdefault("scale_p2", None)

    notches = rec.setdefault("notches", [])
    while len(notches) < n_notches:
        notches.append({"p1": None, "p2": None})
    if len(notches) > n_notches:
        del notches[n_notches:]

    return rec


def image_normalization_px(img_shape: Tuple[int, int, int], mode: str, rec: Dict[str, Any]) -> Optional[float]:
    h, w = img_shape[:2]
    if mode == "none":
        return 1.0
    if mode == "image_width":
        return float(w)
    if mode == "image_height":
        return float(h)
    if mode == "image_diagonal":
        return float(math.hypot(w, h))
    if mode == "manual_scale":
        p1 = tuple_point(rec.get("scale_p1"))
        p2 = tuple_point(rec.get("scale_p2"))
        if p1 is None or p2 is None:
            return None
        d = point_distance_px(p1, p2)
        if d <= 0:
            return None
        return float(d)
    raise ValueError(f"Unknown normalization mode: {mode}")


def load_progress(path: Path, reset: bool = False) -> Dict[str, Any]:
    if reset or not path.exists():
        return {"version": 2, "images": {}}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_progress(path: Path, progress: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2)
    tmp.replace(path)


class NotchMeasurementApp:
    def __init__(
        self,
        root: tk.Tk,
        root_folder: Path,
        tasks: List[ImageTask],
        output_folder: Path,
        progress_path: Path,
        progress: Dict[str, Any],
        n_notches: int = DEFAULT_N_NOTCHES,
        normalization: str = "image_width",
    ):
        self.root = root
        self.root_folder = Path(root_folder)
        self.tasks = tasks
        self.output_folder = Path(output_folder)
        self.progress_path = Path(progress_path)
        self.progress = progress
        self.n_notches = int(n_notches)
        self.normalization = normalization

        self.current_task_idx = 0
        self.current_notch_idx = 0

        self.current_image_array: Optional[np.ndarray] = None
        self.current_image_size: Optional[Tuple[int, int]] = None

        self.p1: Optional[Point] = None
        self.p2: Optional[Point] = None
        self.click_stage = 0  # 0: p1, 1: p2, 2: pair complete/drag

        self.scale_p1: Optional[Point] = None
        self.scale_p2: Optional[Point] = None
        self.scale_stage = 0  # manual scale: 0 p1, 1 p2, 2 done

        self.drag_target: Optional[str] = None

        self.held_keys = set()
        self.hold_frames: Dict[str, int] = {}
        self.nav_timer_running = False
        self.last_nav_interval_ms = 30
        self._view_initialized = False
        self._view_xlim: Optional[Tuple[float, float]] = None
        self._view_ylim: Optional[Tuple[float, float]] = None

        self.root.title("Notch Separation Measurement")
        self.root.configure(bg="black")
        try:
            self.root.state("zoomed")
        except tk.TclError:
            self.root.geometry("1500x950")

        self._build_ui()
        self._bind_keys()
        self.load_current_task_and_notch(reset_view=True)
        self.redraw()

    def _build_ui(self) -> None:
        # Fixed-height toolbar: prevents canvas resize/jump when status text changes
        # after Space / first click on the next notch.
        toolbar = tk.Frame(self.root, bg="black", height=82)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(3, 2))
        toolbar.pack_propagate(False)

        self.status_var = tk.StringVar()
        self.info_var = tk.StringVar()

        button_style = {
            "bg": "#030303",
            "fg": "white",
            "activebackground": "#080808",
            "activeforeground": "white",
            "relief": tk.FLAT,
            "bd": 0,
            "padx": 7,
            "pady": 2,
            "highlightthickness": 0,
        }

        btn_frame = tk.Frame(toolbar, bg="black", height=28)
        btn_frame.pack(side=tk.TOP, fill=tk.X, pady=(0, 2))
        btn_frame.pack_propagate(False)

        tk.Button(btn_frame, text="Reset notch", command=self.reset_current_notch, **button_style).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Prev notch", command=self.prev_notch, **button_style).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Save + next notch", command=self.next_notch, **button_style).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Prev image", command=self.prev_image, **button_style).pack(side=tk.LEFT, padx=(10, 2))
        tk.Button(btn_frame, text="Next image", command=self.next_image, **button_style).pack(side=tk.LEFT, padx=2)
        tk.Button(btn_frame, text="Reset scale", command=self.reset_scale, **button_style).pack(side=tk.LEFT, padx=(10, 2))
        tk.Button(btn_frame, text="Finish + export", command=self.finish_and_export, **button_style).pack(side=tk.RIGHT, padx=2)

        tk.Label(
            toolbar,
            textvariable=self.status_var,
            font=("Arial", 10, "bold"),
            bg="black",
            fg="white",
            anchor="w",
            height=1,
        ).pack(side=tk.TOP, fill=tk.X)

        tk.Label(
            toolbar,
            textvariable=self.info_var,
            justify="left",
            bg="black",
            fg="white",
            anchor="w",
            height=2,
        ).pack(side=tk.TOP, fill=tk.X)

        self.fig, self.ax = plt.subplots(figsize=(24, 18), facecolor="black")
        self.fig.subplots_adjust(left=0.01, right=0.995, top=0.985, bottom=0.02)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

    def _bind_keys(self) -> None:
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)
        self.root.bind("<space>", lambda event: self.next_notch())
        self.root.bind("<Return>", lambda event: self.next_image())
        self.root.bind("<KP_Enter>", lambda event: self.next_image())
        self.root.bind("<BackSpace>", lambda event: self.prev_notch())
        self.root.focus_set()

    def current_task(self) -> ImageTask:
        return self.tasks[self.current_task_idx]

    def current_record(self) -> Dict[str, Any]:
        return ensure_record(self.progress, self.current_task(), self.n_notches)

    def current_notch_record(self) -> Dict[str, Any]:
        return self.current_record()["notches"][self.current_notch_idx]

    def load_current_task_and_notch(self, reset_view: bool = False) -> None:
        task = self.current_task()
        rec = self.current_record()

        img = Image.open(task.image_path).convert("RGB")
        self.current_image_array = np.array(img)
        h, w = self.current_image_array.shape[:2]
        self.current_image_size = (w, h)

        self.scale_p1 = tuple_point(rec.get("scale_p1"))
        self.scale_p2 = tuple_point(rec.get("scale_p2"))
        self.scale_stage = 2 if self.scale_p1 is not None and self.scale_p2 is not None else (1 if self.scale_p1 is not None else 0)

        norm_px = image_normalization_px(self.current_image_array.shape, self.normalization, rec)
        rec["normalization_px"] = norm_px

        if reset_view:
            self._view_initialized = False
            self._view_xlim = None
            self._view_ylim = None

        self.load_current_notch()

    def load_current_notch(self) -> None:
        notch = self.current_notch_record()
        self.p1 = tuple_point(notch.get("p1"))
        self.p2 = tuple_point(notch.get("p2"))
        if self.p1 is None:
            self.click_stage = 0
        elif self.p2 is None:
            self.click_stage = 1
        else:
            self.click_stage = 2
        self.update_status_text()

    def save_current_notch(self, allow_incomplete: bool = False) -> bool:
        rec = self.current_record()

        if self.normalization == "manual_scale":
            rec["scale_p1"] = json_point(self.scale_p1)
            rec["scale_p2"] = json_point(self.scale_p2)
            norm_px = image_normalization_px(self.current_image_array.shape, self.normalization, rec)  # type: ignore[arg-type]
            rec["normalization_px"] = norm_px
            if norm_px is None and not allow_incomplete:
                return False

        notch = self.current_notch_record()
        if self.p1 is None or self.p2 is None:
            if allow_incomplete:
                notch["p1"] = json_point(self.p1)
                notch["p2"] = json_point(self.p2)
                save_progress(self.progress_path, self.progress)
            return False

        notch["p1"] = json_point(self.p1)
        notch["p2"] = json_point(self.p2)
        save_progress(self.progress_path, self.progress)
        return True

    def _current_axes_view(self) -> Optional[Tuple[Tuple[float, float], Tuple[float, float]]]:
        """Return the current image viewport/crop limits.

        Keeping this explicit prevents matplotlib from autoscaling or recentering the
        image whenever a point is clicked and the axes are redrawn.
        """
        if self.current_image_array is None:
            return None
        try:
            xlim = tuple(float(v) for v in self.ax.get_xlim())
            ylim = tuple(float(v) for v in self.ax.get_ylim())
        except Exception:
            return None
        return (xlim, ylim)

    def _store_axes_view(self) -> None:
        view = self._current_axes_view()
        if view is None:
            return
        self._view_xlim, self._view_ylim = view
        self._view_initialized = True

    def _set_full_image_view(self) -> None:
        if self.current_image_array is None:
            return
        h, w = self.current_image_array.shape[:2]
        self._view_xlim = (0.0, float(w))
        self._view_ylim = (float(h), 0.0)
        self._view_initialized = True

    def _apply_stored_view(self) -> None:
        if self.current_image_array is None:
            return
        if not self._view_initialized or self._view_xlim is None or self._view_ylim is None:
            self._set_full_image_view()
        assert self._view_xlim is not None
        assert self._view_ylim is not None
        self.ax.set_xlim(self._view_xlim)
        self.ax.set_ylim(self._view_ylim)
        self.ax.set_autoscale_on(False)

    def update_status_text(self) -> None:
        task = self.current_task()
        rec = self.current_record()
        norm_px = rec.get("normalization_px")

        status = (
            f"Image {self.current_task_idx + 1}/{len(self.tasks)} | "
            f"Mode {task.mode} | Angle {task.angle_label}° | "
            f"Notch {self.current_notch_idx + 1}/{self.n_notches} | "
            f"{task.rel_image_path}"
        )
        self.status_var.set(status)

        if self.normalization == "manual_scale" and norm_px is None:
            if self.scale_stage == 0:
                stage_text = "Click scale point 1."
            elif self.scale_stage == 1:
                stage_text = "Click scale point 2."
            else:
                stage_text = "Invalid scale; reset scale."
        else:
            stage_text = {
                0: "Click notch point 1.",
                1: "Click notch point 2.",
                2: "Pair complete; Space for next or drag points to adjust.",
            }[self.click_stage]

        if self.p1 is not None and self.p2 is not None:
            raw = point_distance_px(self.p1, self.p2)
            if norm_px is not None and norm_px > 0:
                dist_text = f"distance {raw:.2f}px | norm {raw / norm_px:.6f}"
            else:
                dist_text = f"distance {raw:.2f}px"
        else:
            dist_text = "distance --"

        norm_text = f"norm: {self.normalization}"
        if norm_px is not None:
            norm_text += f" ({norm_px:.2f}px)"

        # Always exactly two lines, because the label has fixed height=2.
        info = (
            f"{stage_text} | {dist_text} | {norm_text}\n"
            f"Crop controls only: Q/W zoom, arrows pan. Measurement: Space next notch, Backspace previous, Enter next image."
        )
        self.info_var.set(info)

    def redraw(self, fixed_view: Optional[Tuple[Tuple[float, float], Tuple[float, float]]] = None) -> None:
        """Redraw overlays without changing the current image viewport.

        fixed_view is used after mouse clicks/drags so the selected crop/zoom is
        restored exactly, even if matplotlib would otherwise autoscale after clear().
        The only places that intentionally change the stored viewport are the
        keyboard crop/navigation controls in _nav_tick() and loading a new image.
        """
        if self.current_image_array is None:
            return

        if fixed_view is not None:
            self._view_xlim, self._view_ylim = fixed_view
            self._view_initialized = True
        elif self._view_initialized:
            # Capture the current crop once before clearing the axes.
            # This is the key guard against view jumps during point selection.
            self._store_axes_view()

        self.ax.clear()
        self.fig.subplots_adjust(left=0.01, right=0.995, top=0.985, bottom=0.02)
        img = self.current_image_array

        self.fig.patch.set_facecolor("black")
        self.ax.set_facecolor("black")
        self.ax.imshow(img)
        self.ax.set_aspect("equal")
        self.ax.set_title("Click two points for each notch, from base to tip", color="white")
        self.ax.tick_params(colors="white")
        for spine in self.ax.spines.values():
            spine.set_color("white")

        self._apply_stored_view()

        self.draw_scale_overlay()
        self.draw_notch_overlays()

        # Re-apply after drawing labels/overlays so matplotlib cannot expand limits
        # to include off-screen annotations or markers.
        self._apply_stored_view()

        self.update_status_text()
        self.canvas.draw_idle()

    def draw_scale_overlay(self) -> None:
        if self.normalization != "manual_scale":
            return

        if self.scale_p1 is not None:
            self.ax.plot(self.scale_p1[0], self.scale_p1[1], marker="o", color="lime", markersize=4)
        if self.scale_p2 is not None:
            self.ax.plot(self.scale_p2[0], self.scale_p2[1], marker="o", color="lime", markersize=4)
        if self.scale_p1 is not None and self.scale_p2 is not None:
            self.ax.plot(
                [self.scale_p1[0], self.scale_p2[0]],
                [self.scale_p1[1], self.scale_p2[1]],
                color="lime",
                linewidth=1.2,
                linestyle="--",
                label="Normalization length",
            )

    def draw_notch_overlays(self) -> None:
        rec = self.current_record()
        notches = rec.get("notches", [])

        for i, notch in enumerate(notches):
            p1 = tuple_point(notch.get("p1"))
            p2 = tuple_point(notch.get("p2"))
            if i == self.current_notch_idx:
                continue
            if p1 is None or p2 is None:
                continue
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color="cyan", alpha=0.35, linewidth=0.8)
            self.ax.plot([p1[0], p2[0]], [p1[1], p2[1]], "o", color="cyan", alpha=0.45, markersize=2)
            mx = 0.5 * (p1[0] + p2[0])
            my = 0.5 * (p1[1] + p2[1])
            self.ax.text(mx, my, str(i + 1), color="white", fontsize=7, alpha=0.7)

        if self.p1 is not None:
            self.ax.plot(self.p1[0], self.p1[1], marker="o", color="red", markersize=4)
        if self.p2 is not None:
            self.ax.plot(self.p2[0], self.p2[1], marker="o", color="red", markersize=4)
        if self.p1 is not None and self.p2 is not None:
            self.ax.plot([self.p1[0], self.p2[0]], [self.p1[1], self.p2[1]], color="red", linewidth=1.4)
            mx = 0.5 * (self.p1[0] + self.p2[0])
            my = 0.5 * (self.p1[1] + self.p2[1])
            self.ax.text(
                mx,
                my,
                f"{self.current_notch_idx + 1}",
                color="yellow",
                fontsize=10,
                fontweight="bold",
                bbox=dict(facecolor="black", alpha=0.55, edgecolor="none", pad=2),
            )

        handles, labels = self.ax.get_legend_handles_labels()
        if handles:
            leg = self.ax.legend(loc="upper right")
            if leg is not None:
                leg.get_frame().set_facecolor("#020202")
                leg.get_frame().set_edgecolor("#111111")
                leg.get_frame().set_alpha(0.95)
                for text in leg.get_texts():
                    text.set_color("white")

    def reset_current_notch(self) -> None:
        self.p1 = None
        self.p2 = None
        self.click_stage = 0
        notch = self.current_notch_record()
        notch["p1"] = None
        notch["p2"] = None
        save_progress(self.progress_path, self.progress)
        self.redraw()

    def reset_scale(self) -> None:
        rec = self.current_record()
        self.scale_p1 = None
        self.scale_p2 = None
        self.scale_stage = 0
        rec["scale_p1"] = None
        rec["scale_p2"] = None
        rec["normalization_px"] = None
        save_progress(self.progress_path, self.progress)
        self.redraw()

    def prev_notch(self) -> None:
        self.save_current_notch(allow_incomplete=True)
        if self.current_notch_idx > 0:
            self.current_notch_idx -= 1
            self.load_current_notch()
            self.redraw()
        elif self.current_task_idx > 0:
            self.current_task_idx -= 1
            self.current_notch_idx = self.n_notches - 1
            self.load_current_task_and_notch(reset_view=True)
            self.redraw()

    def next_notch(self) -> None:
        if self.normalization == "manual_scale" and image_normalization_px(self.current_image_array.shape, self.normalization, self.current_record()) is None:  # type: ignore[arg-type]
            messagebox.showwarning("Missing scale", "Click the two reference-scale points before measuring notches on this image.")
            return

        ok = self.save_current_notch()
        if not ok:
            messagebox.showwarning("Incomplete notch", "Please click both points for the current notch before continuing.")
            return

        if self.current_notch_idx < self.n_notches - 1:
            self.current_notch_idx += 1
            self.load_current_notch()
            self.redraw()
        else:
            self.next_image(require_current_complete=False)

    def image_has_all_notches(self, task_idx: Optional[int] = None) -> bool:
        if task_idx is None:
            task_idx = self.current_task_idx
        task = self.tasks[task_idx]
        rec = ensure_record(self.progress, task, self.n_notches)
        for notch in rec.get("notches", []):
            if tuple_point(notch.get("p1")) is None or tuple_point(notch.get("p2")) is None:
                return False
        if self.normalization == "manual_scale":
            if tuple_point(rec.get("scale_p1")) is None or tuple_point(rec.get("scale_p2")) is None:
                return False
        return True

    def prev_image(self) -> None:
        self.save_current_notch(allow_incomplete=True)
        if self.current_task_idx > 0:
            self.current_task_idx -= 1
            self.current_notch_idx = 0
            self.load_current_task_and_notch(reset_view=True)
            self.redraw()

    def next_image(self, require_current_complete: bool = True) -> None:
        self.save_current_notch(allow_incomplete=True)
        if require_current_complete and not self.image_has_all_notches(self.current_task_idx):
            ok = messagebox.askyesno(
                "Incomplete image",
                "This image does not have all notches measured. Move to the next image anyway?",
            )
            if not ok:
                return

        if self.current_task_idx < len(self.tasks) - 1:
            self.current_task_idx += 1
            self.current_notch_idx = 0
            self.load_current_task_and_notch(reset_view=True)
            self.redraw()
        else:
            if messagebox.askyesno("Done", "You are on the last image. Export results now?"):
                self.finish_and_export()

    def finish_and_export(self) -> None:
        self.save_current_notch(allow_incomplete=True)
        save_progress(self.progress_path, self.progress)
        try:
            outputs = export_results(
                root_folder=self.root_folder,
                output_folder=self.output_folder,
                tasks=self.tasks,
                progress=self.progress,
                n_notches=self.n_notches,
                normalization=self.normalization,
            )
        except Exception as e:
            messagebox.showerror("Export error", str(e))
            return

        msg = "Export complete:\n\n" + "\n".join(str(p) for p in outputs)
        messagebox.showinfo("Export finished", msg)

    def _event_to_data(self, event) -> Optional[Point]:
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return None
        return (float(event.xdata), float(event.ydata))

    def _nearest_handle(self, screen_x: float, screen_y: float) -> Optional[str]:
        handles = {
            "p1": self.p1,
            "p2": self.p2,
        }
        if self.normalization == "manual_scale":
            handles["scale_p1"] = self.scale_p1
            handles["scale_p2"] = self.scale_p2

        threshold_screen_px = 8.0
        best_name = None
        best_dist = float("inf")
        for name, pt in handles.items():
            if pt is None:
                continue
            sx, sy = self.ax.transData.transform(pt)
            d = math.hypot(sx - screen_x, sy - screen_y)
            if d < best_dist:
                best_dist = d
                best_name = name
        if best_dist <= threshold_screen_px:
            return best_name
        return None

    def on_mouse_press(self, event) -> None:
        data = self._event_to_data(event)
        if data is None:
            return

        fixed_view = self._current_axes_view()

        rec = self.current_record()
        norm_px = image_normalization_px(self.current_image_array.shape, self.normalization, rec)  # type: ignore[arg-type]

        if self.normalization == "manual_scale" and norm_px is None:
            if self.scale_stage == 0:
                self.scale_p1 = data
                rec["scale_p1"] = json_point(self.scale_p1)
                self.scale_stage = 1
            elif self.scale_stage == 1:
                self.scale_p2 = data
                rec["scale_p2"] = json_point(self.scale_p2)
                self.scale_stage = 2
                norm_px = image_normalization_px(self.current_image_array.shape, self.normalization, rec)  # type: ignore[arg-type]
                rec["normalization_px"] = norm_px
            save_progress(self.progress_path, self.progress)
            self.redraw(fixed_view=fixed_view)
            return

        if self.click_stage == 0:
            self.p1 = data
            self.click_stage = 1
        elif self.click_stage == 1:
            self.p2 = data
            self.click_stage = 2
            self.save_current_notch(allow_incomplete=True)
        else:
            self.drag_target = self._nearest_handle(event.x, event.y)

        self.redraw(fixed_view=fixed_view)

    def on_mouse_move(self, event) -> None:
        if self.drag_target is None:
            return
        data = self._event_to_data(event)
        if data is None:
            return

        fixed_view = self._current_axes_view()

        if self.drag_target == "p1":
            self.p1 = data
        elif self.drag_target == "p2":
            self.p2 = data
        elif self.drag_target == "scale_p1":
            self.scale_p1 = data
        elif self.drag_target == "scale_p2":
            self.scale_p2 = data

        rec = self.current_record()
        rec["scale_p1"] = json_point(self.scale_p1)
        rec["scale_p2"] = json_point(self.scale_p2)
        rec["normalization_px"] = image_normalization_px(self.current_image_array.shape, self.normalization, rec)  # type: ignore[arg-type]
        self.save_current_notch(allow_incomplete=True)
        self.redraw(fixed_view=fixed_view)

    def on_mouse_release(self, event) -> None:
        self.drag_target = None
        self.save_current_notch(allow_incomplete=True)

    def on_key_press(self, event) -> None:
        key = event.keysym.lower()
        valid = {"left", "right", "up", "down", "q", "w"}
        if key not in valid:
            return
        self.held_keys.add(key)
        self.hold_frames.setdefault(key, 0)
        if not self.nav_timer_running:
            self.nav_timer_running = True
            self._nav_tick()

    def on_key_release(self, event) -> None:
        key = event.keysym.lower()
        self.held_keys.discard(key)
        self.hold_frames.pop(key, None)

    @staticmethod
    def _zoom_axis_preserve_direction(a: float, b: float, zoom_factor: float) -> List[float]:
        center = (a + b) / 2.0
        half = abs(b - a) * zoom_factor / 2.0
        if b >= a:
            return [center - half, center + half]
        return [center + half, center - half]

    def _nav_tick(self) -> None:
        if not self.held_keys:
            self.nav_timer_running = False
            return

        for key in list(self.held_keys):
            self.hold_frames[key] = self.hold_frames.get(key, 0) + 1

        xlim = list(self.ax.get_xlim())
        ylim = list(self.ax.get_ylim())
        xspan = abs(xlim[1] - xlim[0])
        yspan = abs(ylim[1] - ylim[0])

        pan_x = 0.0
        pan_y = 0.0
        zoom_factor = 1.0

        def ramp(frames: int) -> float:
            return min(1.0 + frames / 8.0, 12.0)

        if "left" in self.held_keys:
            pan_x -= 0.01 * xspan * ramp(self.hold_frames["left"])
        if "right" in self.held_keys:
            pan_x += 0.01 * xspan * ramp(self.hold_frames["right"])
        if "up" in self.held_keys:
            pan_y -= 0.01 * yspan * ramp(self.hold_frames["up"])
        if "down" in self.held_keys:
            pan_y += 0.01 * yspan * ramp(self.hold_frames["down"])

        if "q" in self.held_keys:
            zoom_factor *= 1 / (1.0 + 0.02 * ramp(self.hold_frames["q"]))
        if "w" in self.held_keys:
            zoom_factor *= 1.0 + 0.02 * ramp(self.hold_frames["w"])

        xlim = [x + pan_x for x in xlim]
        ylim = [y + pan_y for y in ylim]

        self.ax.set_xlim(self._zoom_axis_preserve_direction(xlim[0], xlim[1], zoom_factor))
        self.ax.set_ylim(self._zoom_axis_preserve_direction(ylim[0], ylim[1], zoom_factor))
        self._store_axes_view()
        self.canvas.draw_idle()
        self.root.after(self.last_nav_interval_ms, self._nav_tick)


def build_long_dataframe(
    tasks: List[ImageTask],
    progress: Dict[str, Any],
    n_notches: int,
    normalization: str,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    for task in tasks:
        rec = ensure_record(progress, task, n_notches)
        norm_px = rec.get("normalization_px")
        norm_px_float = safe_float(norm_px)

        for notch_idx in range(n_notches):
            notch = rec["notches"][notch_idx]
            p1 = tuple_point(notch.get("p1"))
            p2 = tuple_point(notch.get("p2"))

            distance_px = float("nan")
            distance_norm = float("nan")
            distance_norm_percent = float("nan")

            if p1 is not None and p2 is not None:
                distance_px = point_distance_px(p1, p2)
                if math.isfinite(norm_px_float) and norm_px_float > 0:
                    distance_norm = distance_px / norm_px_float
                    distance_norm_percent = 100.0 * distance_norm

            rows.append(
                {
                    "mode": task.mode,
                    "angle_deg": task.angle_value,
                    "angle_label": task.angle_label,
                    "image_path": task.rel_image_path,
                    "notch_position": notch_idx + 1,
                    "notch_position_normalized_base0_tip1": notch_idx / (n_notches - 1) if n_notches > 1 else 0.0,
                    "p1_x_px": p1[0] if p1 is not None else np.nan,
                    "p1_y_px": p1[1] if p1 is not None else np.nan,
                    "p2_x_px": p2[0] if p2 is not None else np.nan,
                    "p2_y_px": p2[1] if p2 is not None else np.nan,
                    "distance_px": distance_px,
                    "normalization_mode": normalization,
                    "normalization_px": norm_px_float,
                    "distance_normalized": distance_norm,
                    "distance_normalized_percent": distance_norm_percent,
                    "scale_p1_x_px": tuple_point(rec.get("scale_p1"))[0] if tuple_point(rec.get("scale_p1")) is not None else np.nan,
                    "scale_p1_y_px": tuple_point(rec.get("scale_p1"))[1] if tuple_point(rec.get("scale_p1")) is not None else np.nan,
                    "scale_p2_x_px": tuple_point(rec.get("scale_p2"))[0] if tuple_point(rec.get("scale_p2")) is not None else np.nan,
                    "scale_p2_y_px": tuple_point(rec.get("scale_p2"))[1] if tuple_point(rec.get("scale_p2")) is not None else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_summary_dataframe(long_df: pd.DataFrame) -> pd.DataFrame:
    if long_df.empty:
        return pd.DataFrame()

    g = long_df.groupby(["mode", "angle_deg", "angle_label", "notch_position"], dropna=False)
    summary = g.agg(
        n_images=("distance_normalized", lambda s: int(s.notna().sum())),
        mean_distance_px=("distance_px", "mean"),
        std_distance_px=("distance_px", "std"),
        mean_distance_normalized=("distance_normalized", "mean"),
        std_distance_normalized=("distance_normalized", "std"),
        mean_distance_normalized_percent=("distance_normalized_percent", "mean"),
        std_distance_normalized_percent=("distance_normalized_percent", "std"),
    ).reset_index()

    summary = summary.sort_values(["mode", "angle_deg", "notch_position"])
    return summary


def build_wide_mean_dataframe(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()
    wide = summary_df.pivot_table(
        index="notch_position",
        columns=["mode", "angle_label"],
        values="mean_distance_normalized",
        aggfunc="first",
    )
    wide.columns = [f"{mode}_angle_{angle}_mean_normalized" for mode, angle in wide.columns]
    return wide.reset_index()


def plot_notch_curves(
    long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_path: Path,
    dark: bool = False,
    y_column: str = "distance_normalized",
) -> None:
    """Plot one curve for every available (mode, angle).

    This intentionally does not require curl and uncurl to have matching angle entries.
    If only curl/30 exists, curl/30 is plotted. If only uncurl/45 exists, uncurl/45 is plotted.
    """
    fig, ax = plt.subplots(figsize=(12, 7))

    if dark:
        fig.patch.set_facecolor((0, 0, 0, 0))
        ax.set_facecolor((0, 0, 0, 0.72))
        text_color = "white"
        grid_color = "white"
        spine_color = "white"
        legend_face = (0, 0, 0, 0.65)
        scatter_alpha = 0.35
        line_alpha = 1.0
    else:
        fig.patch.set_facecolor("white")
        ax.set_facecolor("white")
        text_color = "black"
        grid_color = "black"
        spine_color = "black"
        legend_face = "white"
        scatter_alpha = 0.30
        line_alpha = 0.95

    if summary_df.empty:
        return

    mode_order = {"curl": 0, "uncurl": 1}
    pairs = (
        summary_df[["mode", "angle_deg", "angle_label"]]
        .drop_duplicates()
        .sort_values(by=["angle_deg", "mode"], key=lambda col: col.map(mode_order) if col.name == "mode" else col)
    )

    cmap = plt.get_cmap("tab20")
    markers = {"curl": "o", "uncurl": "^"}
    linestyles = {"curl": "-", "uncurl": "--"}

    for idx, row in enumerate(pairs.itertuples(index=False)):
        mode = str(row.mode)
        angle = float(row.angle_deg)
        angle_label = str(row.angle_label)
        color = cmap(idx % cmap.N)

        mask_long = (long_df["mode"] == mode) & (long_df["angle_deg"] == angle)
        mask_summary = (summary_df["mode"] == mode) & (summary_df["angle_deg"] == angle)
        pair_long = long_df[mask_long].copy()
        pair_summary = summary_df[mask_summary].copy()
        if pair_summary.empty:
            continue

        ax.scatter(
            pair_long["notch_position"],
            pair_long[y_column],
            s=18,
            alpha=scatter_alpha,
            color=color,
            edgecolors="none",
        )
        ax.plot(
            pair_summary["notch_position"],
            pair_summary[f"mean_{y_column}"],
            marker=markers.get(mode, "o"),
            linestyle=linestyles.get(mode, "-"),
            markersize=4.5,
            linewidth=1.8,
            alpha=line_alpha,
            color=color,
            label=f"{mode} {angle_label}°",
        )

    ax.set_title("Notch separation distance vs notch position", color=text_color, fontsize=15)
    ax.set_xlabel("Notch position from base to tip", color=text_color, fontsize=12)

    if y_column == "distance_normalized":
        ax.set_ylabel("Normalized notch separation distance", color=text_color, fontsize=12)
    elif y_column == "distance_normalized_percent":
        ax.set_ylabel("Notch separation distance (% of normalization length)", color=text_color, fontsize=12)
    else:
        ax.set_ylabel("Notch separation distance (px)", color=text_color, fontsize=12)

    ax.set_xlim(0.5, max(25.5, float(long_df["notch_position"].max()) + 0.5))
    ax.set_xticks(sorted(long_df["notch_position"].dropna().unique()))
    ax.tick_params(axis="x", colors=text_color, rotation=0)
    ax.tick_params(axis="y", colors=text_color)

    for spine in ax.spines.values():
        spine.set_color(spine_color)

    ax.grid(True, color=grid_color, alpha=0.12 if dark else 0.18)

    legend = ax.legend(title="Mode / angle", ncols=2 if len(pairs) > 8 else 1)
    if legend is not None:
        legend.get_frame().set_facecolor(legend_face)
        legend.get_frame().set_edgecolor(spine_color)
        legend.get_frame().set_alpha(0.82 if dark else 0.92)
        legend.get_title().set_color(text_color)
        for text in legend.get_texts():
            text.set_color(text_color)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if dark:
        fig.savefig(output_path, dpi=300, transparent=True, facecolor="none", edgecolor="none")
    else:
        fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_notch_curves_by_angle(
    long_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    output_folder: Path,
    dark: bool = False,
    y_column: str = "distance_normalized",
) -> List[Path]:
    """Create one comparison plot per angle.

    If an angle exists only in curl or only in uncurl, it is still plotted.
    """
    outputs: List[Path] = []
    if summary_df.empty:
        return outputs

    suffix = "dark_transparent" if dark else "light"
    angles = summary_df[["angle_deg", "angle_label"]].drop_duplicates().sort_values("angle_deg")

    for angle_row in angles.itertuples(index=False):
        angle = float(angle_row.angle_deg)
        label = str(angle_row.angle_label)
        angle_long = long_df[long_df["angle_deg"] == angle].copy()
        angle_summary = summary_df[summary_df["angle_deg"] == angle].copy()
        if angle_summary.empty:
            continue

        fig, ax = plt.subplots(figsize=(9, 6))
        if dark:
            fig.patch.set_facecolor((0, 0, 0, 0))
            ax.set_facecolor((0, 0, 0, 0.72))
            text_color = "white"
            grid_color = "white"
            spine_color = "white"
            legend_face = (0, 0, 0, 0.65)
        else:
            fig.patch.set_facecolor("white")
            ax.set_facecolor("white")
            text_color = "black"
            grid_color = "black"
            spine_color = "black"
            legend_face = "white"

        # Fixed mapping for direct curl vs uncurl comparison.
        mode_specs = {
            "curl": {"marker": "o", "linestyle": "-"},
            "uncurl": {"marker": "^", "linestyle": "--"},
        }
        cmap = plt.get_cmap("tab10")
        mode_colors = {"curl": cmap(0), "uncurl": cmap(1)}

        for mode in ["curl", "uncurl"]:
            m_long = angle_long[angle_long["mode"] == mode]
            m_summary = angle_summary[angle_summary["mode"] == mode]
            if m_summary.empty:
                continue
            ax.scatter(
                m_long["notch_position"],
                m_long[y_column],
                s=22,
                alpha=0.34 if dark else 0.30,
                color=mode_colors[mode],
                edgecolors="none",
            )
            ax.plot(
                m_summary["notch_position"],
                m_summary[f"mean_{y_column}"],
                marker=mode_specs[mode]["marker"],
                linestyle=mode_specs[mode]["linestyle"],
                markersize=5,
                linewidth=2.0,
                color=mode_colors[mode],
                label=mode,
            )

        ax.set_title(f"Notch separation vs position — {label}°", color=text_color, fontsize=14)
        ax.set_xlabel("Notch position from base to tip", color=text_color, fontsize=12)
        ax.set_ylabel("Normalized notch separation distance", color=text_color, fontsize=12)
        ax.set_xlim(0.5, max(25.5, float(angle_long["notch_position"].max()) + 0.5))
        ax.set_xticks(sorted(angle_long["notch_position"].dropna().unique()))
        ax.tick_params(axis="x", colors=text_color)
        ax.tick_params(axis="y", colors=text_color)
        for spine in ax.spines.values():
            spine.set_color(spine_color)
        ax.grid(True, color=grid_color, alpha=0.12 if dark else 0.18)
        legend = ax.legend(title="Mode")
        if legend is not None:
            legend.get_frame().set_facecolor(legend_face)
            legend.get_frame().set_edgecolor(spine_color)
            legend.get_frame().set_alpha(0.82 if dark else 0.92)
            legend.get_title().set_color(text_color)
            for text in legend.get_texts():
                text.set_color(text_color)

        plt.tight_layout()
        out = output_folder / f"notch_separation_vs_position_angle_{label}_{suffix}.png"
        if dark:
            fig.savefig(out, dpi=300, transparent=True, facecolor="none", edgecolor="none")
        else:
            fig.savefig(out, dpi=300)
        plt.close(fig)
        outputs.append(out)

    return outputs


def export_results(
    root_folder: Path,
    output_folder: Path,
    tasks: List[ImageTask],
    progress: Dict[str, Any],
    n_notches: int,
    normalization: str,
) -> List[Path]:
    output_folder.mkdir(parents=True, exist_ok=True)

    long_df = build_long_dataframe(tasks, progress, n_notches, normalization)
    summary_df = build_summary_dataframe(long_df)
    wide_df = build_wide_mean_dataframe(summary_df)

    long_csv = output_folder / "notch_measurements_long.csv"
    summary_csv = output_folder / "notch_measurements_summary_by_mode_angle.csv"
    wide_csv = output_folder / "notch_measurements_wide_mean.csv"
    light_png = output_folder / "notch_separation_vs_position_light.png"
    dark_png = output_folder / "notch_separation_vs_position_dark_transparent.png"
    excel_path = output_folder / "notch_measurements.xlsx"

    long_df.to_csv(long_csv, index=False)
    summary_df.to_csv(summary_csv, index=False)
    wide_df.to_csv(wide_csv, index=False)

    outputs = [long_csv, summary_csv, wide_csv]

    if not summary_df.empty and long_df["distance_normalized"].notna().any():
        plot_notch_curves(long_df, summary_df, light_png, dark=False, y_column="distance_normalized")
        plot_notch_curves(long_df, summary_df, dark_png, dark=True, y_column="distance_normalized")
        outputs.extend([light_png, dark_png])
        by_angle_dir = output_folder / "by_angle_plots"
        outputs.extend(plot_notch_curves_by_angle(long_df, summary_df, by_angle_dir, dark=False, y_column="distance_normalized"))
        outputs.extend(plot_notch_curves_by_angle(long_df, summary_df, by_angle_dir, dark=True, y_column="distance_normalized"))

    try:
        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            long_df.to_excel(writer, sheet_name="Long", index=False)
            summary_df.to_excel(writer, sheet_name="Summary", index=False)
            wide_df.to_excel(writer, sheet_name="Wide mean", index=False)
        outputs.append(excel_path)
    except Exception as e:
        print(f"Warning: could not write Excel file: {e}", file=sys.stderr)

    return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Interactively measure notch separation for angle-folder image sets.")
    parser.add_argument("folder", nargs="?", help="Root folder containing curl/ and/or uncurl/ folders, each with numeric angle subfolders.")
    parser.add_argument("--notches", type=int, default=DEFAULT_N_NOTCHES, help="Number of notches per image. Default: 25.")
    parser.add_argument(
        "--normalization",
        choices=["image_width", "image_height", "image_diagonal", "manual_scale", "none"],
        default="image_width",
        help="How to normalize pixel distances between images. Default: image_width.",
    )
    parser.add_argument("--reset_progress", action="store_true", help="Ignore previous saved measurements and start fresh.")
    parser.add_argument("--export_only", action="store_true", help="Do not open the GUI; export plots/tables from the existing progress JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    root = tk.Tk()
    root.withdraw()

    folder = args.folder
    if not folder:
        messagebox.showinfo(
            "Select root folder",
            "Choose the root folder containing numeric angle subfolders, e.g. 0, 30, 60, 90, 180.\n\n"
            "Each angle subfolder should contain the pictures for that curl angle.",
        )
        folder = filedialog.askdirectory(title="Select root folder with angle subfolders")
        if not folder:
            root.destroy()
            return

    root_folder = Path(folder).expanduser().resolve()
    output_folder = root_folder / "notch_measurement_outputs"
    progress_path = output_folder / "notch_measurement_progress.json"

    try:
        tasks = find_angle_image_tasks(root_folder)
        progress = load_progress(progress_path, reset=args.reset_progress)
    except Exception as e:
        messagebox.showerror("Folder error", str(e))
        root.destroy()
        return

    progress["normalization"] = args.normalization
    progress["n_notches"] = args.notches

    if args.export_only:
        try:
            outputs = export_results(
                root_folder=root_folder,
                output_folder=output_folder,
                tasks=tasks,
                progress=progress,
                n_notches=args.notches,
                normalization=args.normalization,
            )
        except Exception as e:
            print(f"Export failed: {e}", file=sys.stderr)
            root.destroy()
            raise SystemExit(1)
        print("Export complete:")
        for p in outputs:
            print(p)
        root.destroy()
        return

    root.deiconify()
    app = NotchMeasurementApp(
        root=root,
        root_folder=root_folder,
        tasks=tasks,
        output_folder=output_folder,
        progress_path=progress_path,
        progress=progress,
        n_notches=args.notches,
        normalization=args.normalization,
    )
    root.mainloop()


if __name__ == "__main__":
    main()
