import math
from pathlib import Path
import sys
import tkinter as tk
from tkinter import filedialog, messagebox

import numpy as np
import pandas as pd
from PIL import Image

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"]
ANGLES = list(range(-180, 181, 5))
EXPECTED_ITEMS = [1, 2]


def find_images_by_item_number(folder: str):
    folder = Path(folder)
    found = {}

    for item in EXPECTED_ITEMS:
        matches = []
        for ext in IMAGE_EXTS:
            p = folder / f"{item}{ext}"
            if p.exists():
                matches.append(p)

        if len(matches) == 0:
            raise ValueError(
                f"Missing image for item {item}. Expected a file like '{item}.jpg' "
                f"(or .png, .tif, etc.) in the selected folder."
            )
        if len(matches) > 1:
            raise ValueError(
                f"Multiple files found for item {item}: {', '.join(str(m.name) for m in matches)}. "
                f"Keep only one image per item number."
            )

        found[item] = matches[0]

    return [found[item] for item in EXPECTED_ITEMS], EXPECTED_ITEMS.copy()


def line_angle_degrees(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    if dx == 0 and dy == 0:
        return None
    ang = math.degrees(math.atan2(dy, dx))
    return ang % 180


def perpendicular_distance_px(p1, p2, p3):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    dx = x2 - x1
    dy = y2 - y1
    denom = math.hypot(dx, dy)
    if denom == 0:
        return None
    return abs(dx * (y1 - y3) - (x1 - x3) * dy) / denom


class LineMeasurementApp:
    def __init__(self, root, image_paths, sample_numbers, output_folder):
        self.root = root
        self.root.title("Parallel Line Distance Measurement")
        self.root.configure(bg="black")
        self.root.state("zoomed")  # maximize window where supported

        self.image_paths = image_paths
        self.sample_numbers = sample_numbers
        self.sample_labels = [str(n) for n in sample_numbers]
        self.output_folder = Path(output_folder)

        self.images = [np.array(Image.open(p).convert("RGB")) for p in image_paths]
        self.results = {
            label: {angle: np.nan for angle in ANGLES} for label in self.sample_labels
        }

        self.current_image_idx = 0
        self.current_angle_idx = 0

        self.p1 = None
        self.p2 = None
        self.p3 = None
        self.click_stage = 0
        self.drag_target = None

        self.geometry_store = {label: {} for label in self.sample_labels}

        self.held_keys = set()
        self.hold_frames = {}
        self.nav_timer_running = False
        self.last_nav_interval_ms = 30

        self._view_initialized = False

        self._build_ui()
        self._bind_keys()
        self.load_current_measurement()
        self.redraw()

    def _build_ui(self):
        top = tk.Frame(self.root, bg="black")
        top.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        self.status_var = tk.StringVar()
        self.info_var = tk.StringVar()

        tk.Label(
            top,
            textvariable=self.status_var,
            font=("Arial", 12, "bold"),
            bg="black",
            fg="white"
        ).pack(anchor="w")

        tk.Label(
            top,
            textvariable=self.info_var,
            justify="left",
            bg="black",
            fg="white"
        ).pack(anchor="w", pady=(4, 0))

        btn_frame = tk.Frame(self.root, bg="black")
        btn_frame.pack(side=tk.TOP, fill=tk.X, padx=8, pady=6)

        button_style = {
            "bg": "#030303",
            "fg": "white",
            "activebackground": "#080808",
            "activeforeground": "white",
            "relief": tk.FLAT,
            "bd": 0,
            "padx": 8,
            "pady": 4,
            "highlightthickness": 0,
        }

        tk.Button(btn_frame, text="Reset current angle", command=self.reset_current_angle, **button_style).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="Previous angle", command=self.prev_angle, **button_style).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="Save + Next angle", command=self.next_angle, **button_style).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="Previous image", command=self.prev_image, **button_style).pack(side=tk.LEFT, padx=12)
        tk.Button(btn_frame, text="Next image", command=self.next_image, **button_style).pack(side=tk.LEFT, padx=3)
        tk.Button(btn_frame, text="Finish + Export", command=self.finish_and_export, **button_style).pack(side=tk.RIGHT, padx=3)

        # Larger figure: about 2x previous visual space
        self.fig, self.ax = plt.subplots(figsize=(22, 16), facecolor="black")
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.root)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.canvas.mpl_connect("button_press_event", self.on_mouse_press)
        self.canvas.mpl_connect("motion_notify_event", self.on_mouse_move)
        self.canvas.mpl_connect("button_release_event", self.on_mouse_release)

    def _bind_keys(self):
        self.root.bind("<KeyPress>", self.on_key_press)
        self.root.bind("<KeyRelease>", self.on_key_release)

        self.root.bind("<space>", lambda event: self.next_angle())
        self.root.bind("<Return>", lambda event: self.next_image())
        self.root.bind("<KP_Enter>", lambda event: self.next_image())

        self.root.focus_set()

    def current_image(self):
        return self.images[self.current_image_idx]

    def current_sample_label(self):
        return self.sample_labels[self.current_image_idx]

    def current_sample_number(self):
        return self.sample_numbers[self.current_image_idx]

    def current_angle(self):
        return ANGLES[self.current_angle_idx]

    def load_current_measurement(self):
        label = self.current_sample_label()
        angle = self.current_angle()

        saved = self.geometry_store[label].get(angle)
        if saved is None:
            self.p1 = None
            self.p2 = None
            self.p3 = None
            self.click_stage = 0
        else:
            self.p1 = tuple(saved["p1"])
            self.p2 = tuple(saved["p2"])
            self.p3 = tuple(saved["p3"])
            self.click_stage = 3

        self.update_status_text()

    def save_current_measurement(self):
        label = self.current_sample_label()
        angle = self.current_angle()

        if self.p1 is None or self.p2 is None or self.p3 is None:
            return False

        dist = perpendicular_distance_px(self.p1, self.p2, self.p3)
        if dist is None:
            return False

        self.results[label][angle] = dist
        self.geometry_store[label][angle] = {
            "p1": self.p1,
            "p2": self.p2,
            "p3": self.p3,
        }
        return True

    def update_status_text(self):
        ang = self.current_angle()
        img_name = self.image_paths[self.current_image_idx].name

        line_ang = None
        if self.p1 is not None and self.p2 is not None:
            line_ang = line_angle_degrees(self.p1, self.p2)

        dist = None
        if self.p1 is not None and self.p2 is not None and self.p3 is not None:
            dist = perpendicular_distance_px(self.p1, self.p2, self.p3)

        status = (
            f"Image {self.current_image_idx + 1}/{len(self.images)}: {img_name} | "
            f"Item #{self.current_sample_number()} | "
            f"Target angle: {ang}°"
        )
        self.status_var.set(status)

        stage_text = {
            0: "Click first endpoint of main line.",
            1: "Click second endpoint of main line.",
            2: "Click a point to define the parallel line.",
            3: "Drag endpoints or parallel handle to adjust."
        }[self.click_stage]

        info = (
            f"{stage_text}\n"
            f"Controls: Q zoom in, W zoom out, arrow keys pan.\n"
            f"Space = save + next angle, Enter = next image.\n"
            f"Actual line angle: {line_ang:.2f}°   Distance: {dist:.2f} px"
            if line_ang is not None and dist is not None
            else
            f"{stage_text}\n"
            f"Controls: Q zoom in, W zoom out, arrow keys pan.\n"
            f"Space = save + next angle, Enter = next image."
        )
        self.info_var.set(info)

    def get_long_parallel_segment(self, p1, p2, through_point, width, height):
        x1, y1 = p1
        x2, y2 = p2
        x3, y3 = through_point

        vx = x2 - x1
        vy = y2 - y1
        norm = math.hypot(vx, vy)
        if norm == 0:
            return None

        ux = vx / norm
        uy = vy / norm

        diag = math.hypot(width, height) * 2.0
        xa = x3 - ux * diag
        ya = y3 - uy * diag
        xb = x3 + ux * diag
        yb = y3 + uy * diag
        return (xa, ya), (xb, yb)

    def redraw(self):
        current_xlim = self.ax.get_xlim()
        current_ylim = self.ax.get_ylim()
        preserve_view = self._view_initialized

        self.ax.clear()
        img = self.current_image()
        h, w = img.shape[:2]

        self.fig.patch.set_facecolor("black")
        self.ax.set_facecolor("black")
        self.ax.imshow(img)
        self.ax.set_aspect("equal")
        self.ax.set_title("Place/adjust line and parallel line", color="white")
        self.ax.tick_params(colors="white")
        for spine in self.ax.spines.values():
            spine.set_color("white")

        if preserve_view:
            self.ax.set_xlim(current_xlim)
            self.ax.set_ylim(current_ylim)
        else:
            self.ax.set_xlim(0, w)
            self.ax.set_ylim(h, 0)
            self._view_initialized = True

        handle_size = 2.0
        main_line_width = 0.9
        parallel_line_width = 0.8

        if self.p1 is not None:
            self.ax.plot(self.p1[0], self.p1[1], "ro", markersize=handle_size)
        if self.p2 is not None:
            self.ax.plot(self.p2[0], self.p2[1], "ro", markersize=handle_size)
        if self.p1 is not None and self.p2 is not None:
            self.ax.plot(
                [self.p1[0], self.p2[0]],
                [self.p1[1], self.p2[1]],
                "r-",
                linewidth=main_line_width,
                label="Main line"
            )

        if self.p3 is not None:
            self.ax.plot(self.p3[0], self.p3[1], "co", markersize=handle_size)

        if self.p1 is not None and self.p2 is not None and self.p3 is not None:
            seg = self.get_long_parallel_segment(self.p1, self.p2, self.p3, w, h)
            if seg is not None:
                a, b = seg
                self.ax.plot(
                    [a[0], b[0]],
                    [a[1], b[1]],
                    "c--",
                    linewidth=parallel_line_width,
                    label="Parallel line"
                )

            dist = perpendicular_distance_px(self.p1, self.p2, self.p3)
            line_ang = line_angle_degrees(self.p1, self.p2)
            if dist is not None and line_ang is not None:
                self.ax.text(
                    0.01, 0.02,
                    f"Distance = {dist:.2f} px | Actual line angle = {line_ang:.2f}°",
                    transform=self.ax.transAxes,
                    fontsize=11,
                    color="white",
                    bbox=dict(facecolor="#020202", alpha=0.96, edgecolor="#111111", pad=6)
                )

        leg = self.ax.legend(loc="upper right")
        if leg is not None:
            leg.get_frame().set_facecolor("#020202")
            leg.get_frame().set_edgecolor("#111111")
            leg.get_frame().set_alpha(0.97)
            for text in leg.get_texts():
                text.set_color("white")

        self.update_status_text()
        self.canvas.draw_idle()

    def reset_current_angle(self):
        self.p1 = None
        self.p2 = None
        self.p3 = None
        self.click_stage = 0

        label = self.current_sample_label()
        angle = self.current_angle()
        if angle in self.geometry_store[label]:
            del self.geometry_store[label][angle]
        self.results[label][angle] = np.nan

        # Intentionally keep current zoom/pan
        self.redraw()

    def prev_angle(self):
        self.save_current_measurement()
        if self.current_angle_idx > 0:
            self.current_angle_idx -= 1
            self.load_current_measurement()
            self.redraw()

    def next_angle(self):
        ok = self.save_current_measurement()
        if not ok:
            messagebox.showwarning(
                "Incomplete measurement",
                "Please define the main line with two points and the parallel line with one point before continuing."
            )
            return

        if self.current_angle_idx < len(ANGLES) - 1:
            self.current_angle_idx += 1
            self.load_current_measurement()
            self.redraw()
        else:
            if self.current_image_idx < len(self.images) - 1:
                self.current_image_idx += 1
                self.current_angle_idx = 0
                self.load_current_measurement()
                self.redraw()
            else:
                if messagebox.askyesno("Done", "All angles for all images are complete. Export results now?"):
                    self.finish_and_export()

    def prev_image(self):
        self.save_current_measurement()
        if self.current_image_idx > 0:
            self.current_image_idx -= 1
            self.current_angle_idx = 0
            self.load_current_measurement()
            self.redraw()

    def next_image(self):
        ok = self.save_current_measurement()
        if not ok:
            messagebox.showwarning(
                "Incomplete measurement",
                "Please define the main line with two points and the parallel line with one point before changing image."
            )
            return

        if self.current_image_idx < len(self.images) - 1:
            self.current_image_idx += 1
            self.current_angle_idx = 0
            self.load_current_measurement()
            self.redraw()
        else:
            if messagebox.askyesno("Done", "You are on the last image. Export results now?"):
                self.finish_and_export()

    def export_excel_and_chart(self):
        df = pd.DataFrame(index=ANGLES)
        df.index.name = "Line angle (deg)"
        for label in self.sample_labels:
            df[f"Sample {label}"] = [self.results[label][ang] for ang in ANGLES]

        excel_path = self.output_folder / "parallel_line_distances.xlsx"
        chart_path = self.output_folder / "parallel_line_distances_chart.png"

        with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name="Distances_px")

        fig, ax = plt.subplots(figsize=(16, 7), facecolor="none")
        ax.set_facecolor("#020202")

        df.plot(kind="bar", ax=ax)

        ax.set_xlabel("Line angle (deg)", color="white")
        ax.set_ylabel("Distance between parallel lines (px)", color="white")
        ax.set_title("Distance vs. Line Angle by Sample", color="white")

        ax.tick_params(axis="x", colors="white", rotation=0)
        ax.tick_params(axis="y", colors="white")

        for spine in ax.spines.values():
            spine.set_color("white")

        legend = ax.legend()
        if legend is not None:
            legend.get_frame().set_facecolor("#020202")
            legend.get_frame().set_edgecolor("#111111")
            legend.get_frame().set_alpha(0.97)
            for text in legend.get_texts():
                text.set_color("white")

        ax.grid(True, axis="y", color="white", alpha=0.12)

        plt.tight_layout()
        plt.savefig(
            chart_path,
            dpi=200,
            transparent=True,
            facecolor="none",
            edgecolor="none"
        )
        plt.close(fig)

        return excel_path, chart_path, df

    def finish_and_export(self):
        self.save_current_measurement()
        excel_path, chart_path, df = self.export_excel_and_chart()

        missing = int(df.isna().sum().sum())
        if missing > 0:
            msg = (
                f"Export complete.\n\n"
                f"Excel: {excel_path}\n"
                f"Chart: {chart_path}\n\n"
                f"Warning: {missing} measurements are still missing."
            )
        else:
            msg = (
                f"Export complete.\n\n"
                f"Excel: {excel_path}\n"
                f"Chart: {chart_path}"
            )
        messagebox.showinfo("Export finished", msg)

    def _event_to_data(self, event):
        if event.inaxes != self.ax or event.xdata is None or event.ydata is None:
            return None
        return (float(event.xdata), float(event.ydata))

    def _nearest_handle(self, x, y):
        handles = {"p1": self.p1, "p2": self.p2, "p3": self.p3}
        threshold_px = 5

        best_name = None
        best_dist = float("inf")

        for name, pt in handles.items():
            if pt is None:
                continue
            sx, sy = self.ax.transData.transform(pt)
            d = math.hypot(sx - x, sy - y)
            if d < best_dist:
                best_dist = d
                best_name = name

        if best_dist <= threshold_px:
            return best_name
        return None

    def on_mouse_press(self, event):
        data = self._event_to_data(event)
        if data is None:
            return

        x, y = data

        if self.click_stage == 0:
            self.p1 = (x, y)
            self.click_stage = 1
        elif self.click_stage == 1:
            self.p2 = (x, y)
            self.click_stage = 2
        elif self.click_stage == 2:
            self.p3 = (x, y)
            self.click_stage = 3
        else:
            self.drag_target = self._nearest_handle(event.x, event.y)

        self.redraw()

    def on_mouse_move(self, event):
        if self.drag_target is None:
            return

        data = self._event_to_data(event)
        if data is None:
            return

        x, y = data
        if self.drag_target == "p1":
            self.p1 = (x, y)
        elif self.drag_target == "p2":
            self.p2 = (x, y)
        elif self.drag_target == "p3":
            self.p3 = (x, y)

        self.redraw()

    def on_mouse_release(self, event):
        self.drag_target = None

    def on_key_press(self, event):
        key = event.keysym.lower()
        valid = {"left", "right", "up", "down", "q", "w"}
        if key not in valid:
            return

        self.held_keys.add(key)
        if key not in self.hold_frames:
            self.hold_frames[key] = 0

        if not self.nav_timer_running:
            self.nav_timer_running = True
            self._nav_tick()

    def on_key_release(self, event):
        key = event.keysym.lower()
        self.held_keys.discard(key)
        self.hold_frames.pop(key, None)

    def _zoom_axis_preserve_direction(self, a, b, zoom_factor):
        center = (a + b) / 2.0
        half = abs(b - a) * zoom_factor / 2.0
        if b >= a:
            return [center - half, center + half]
        return [center + half, center - half]

    def _nav_tick(self):
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

        def ramp(frames):
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
            zoom_factor *= (1.0 + 0.02 * ramp(self.hold_frames["w"]))

        xlim = [x + pan_x for x in xlim]
        ylim = [y + pan_y for y in ylim]

        xlim = self._zoom_axis_preserve_direction(xlim[0], xlim[1], zoom_factor)
        ylim = self._zoom_axis_preserve_direction(ylim[0], ylim[1], zoom_factor)

        self.ax.set_xlim(xlim)
        self.ax.set_ylim(ylim)
        self.canvas.draw_idle()

        self.root.after(self.last_nav_interval_ms, self._nav_tick)


def main():
    root = tk.Tk()
    root.withdraw()

    folder = sys.argv[1] if len(sys.argv) > 1 else None

    if not folder:
        messagebox.showinfo(
            "Select folder",
            "Choose the folder containing image files named 1.jpg and 2.jpg "
            "(or 1.png, 2.tif, etc. - one image per item number).\n\n"
            "You will be prompted through target angles -180 to 180 degrees in 5 degree increments."
        )

        folder = filedialog.askdirectory(title="Select folder with images 1..2")
        if not folder:
            return

    try:
        image_paths, sample_numbers = find_images_by_item_number(folder)
    except Exception as e:
        messagebox.showerror("Folder error", str(e))
        return

    root.deiconify()
    app = LineMeasurementApp(root, image_paths, sample_numbers, folder)
    root.mainloop()


if __name__ == "__main__":
    main()
