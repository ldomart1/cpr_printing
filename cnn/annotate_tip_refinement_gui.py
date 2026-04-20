from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np
import pandas as pd

from common_ctr_tip_refinement import read_manifest, write_manifest

WINDOW_NAME = "tip_refinement_annotation"
DEFAULT_DATASET_NAME = "tip_refinement_dataset"
DEFAULT_DISPLAY_SCALE = 8
DEFAULT_WINDOW_WIDTH = 1200
DEFAULT_WINDOW_HEIGHT = 1000


def default_dataset_dir(project_dir: str) -> Path:
    return Path(project_dir).expanduser().resolve() / "processed_image_data_folder" / DEFAULT_DATASET_NAME


class Annotator:
    def __init__(
        self,
        df: pd.DataFrame,
        output_path: str,
        display_scale: int = DEFAULT_DISPLAY_SCALE,
        window_width: int = DEFAULT_WINDOW_WIDTH,
        window_height: int = DEFAULT_WINDOW_HEIGHT,
    ) -> None:
        self.df = df.copy()
        self.output_path = str(Path(output_path).expanduser().resolve())
        self.display_scale = max(2, int(display_scale))
        self.window_width = max(320, int(window_width))
        self.window_height = max(240, int(window_height))
        self.index = 0
        self.manual_xy_patch: Optional[Tuple[float, float]] = None
        self.last_render = None

        self.normalize_columns()

    def normalize_columns(self) -> None:
        if "annotation_status" not in self.df.columns:
            self.df["annotation_status"] = "pending"
        if "annotation_source" not in self.df.columns:
            self.df["annotation_source"] = ""
        if "manual_label_x_patch" not in self.df.columns:
            self.df["manual_label_x_patch"] = np.nan
        if "manual_label_y_patch" not in self.df.columns:
            self.df["manual_label_y_patch"] = np.nan
        if "manual_label_x_abs" not in self.df.columns:
            self.df["manual_label_x_abs"] = np.nan
        if "manual_label_y_abs" not in self.df.columns:
            self.df["manual_label_y_abs"] = np.nan
        if "notes" not in self.df.columns:
            self.df["notes"] = ""

        text_defaults = {
            "annotation_status": "pending",
            "annotation_source": "",
            "notes": "",
        }
        for column, default in text_defaults.items():
            values = self.df[column].astype("object")
            values = values.where(pd.notna(values), default)
            values = values.replace("nan", default)
            if column == "annotation_status":
                values = values.replace("", default)
            self.df[column] = values.astype("object")

        numeric_columns = [
            "manual_label_x_patch",
            "manual_label_y_patch",
            "manual_label_x_abs",
            "manual_label_y_abs",
        ]
        for column in numeric_columns:
            self.df[column] = pd.to_numeric(self.df[column], errors="coerce")

    def save(self) -> None:
        write_manifest(self.df, self.output_path)

    def current_row(self) -> pd.Series:
        return self.df.iloc[self.index]

    def set_manual_point(self, x_patch: float, y_patch: float) -> None:
        self.manual_xy_patch = (float(x_patch), float(y_patch))

    def persist_manual_point(self) -> None:
        if self.manual_xy_patch is None:
            return
        row = self.current_row()
        x_patch, y_patch = self.manual_xy_patch
        x_abs = float(row["patch_requested_x0"] + x_patch)
        y_abs = float(row["patch_requested_y0"] + y_patch)
        self.df.at[self.index, "manual_label_x_patch"] = x_patch
        self.df.at[self.index, "manual_label_y_patch"] = y_patch
        self.df.at[self.index, "manual_label_x_abs"] = x_abs
        self.df.at[self.index, "manual_label_y_abs"] = y_abs
        self.df.at[self.index, "annotation_status"] = "done"
        self.df.at[self.index, "annotation_source"] = "manual"

    def use_existing_point(self, source: str) -> None:
        row = self.current_row()
        source = source.lower()
        if source == "anchor":
            x_patch, y_patch = float(row["anchor_x_patch"]), float(row["anchor_y_patch"])
            x_abs, y_abs = float(row["anchor_x_abs"]), float(row["anchor_y_abs"])
        elif source == "coarse":
            x_patch, y_patch = float(row["coarse_x_patch"]), float(row["coarse_y_patch"])
            x_abs, y_abs = float(row["coarse_x_abs"]), float(row["coarse_y_abs"])
        elif source == "selected":
            x_patch, y_patch = float(row["selected_x_patch"]), float(row["selected_y_patch"])
            x_abs, y_abs = float(row["selected_x_abs"]), float(row["selected_y_abs"])
        elif source == "refined":
            x_patch, y_patch = float(row["refined_x_patch"]), float(row["refined_y_patch"])
            x_abs, y_abs = float(row["refined_x_abs"]), float(row["refined_y_abs"])
        else:
            raise ValueError(f"Unsupported source: {source}")

        self.df.at[self.index, "manual_label_x_patch"] = x_patch
        self.df.at[self.index, "manual_label_y_patch"] = y_patch
        self.df.at[self.index, "manual_label_x_abs"] = x_abs
        self.df.at[self.index, "manual_label_y_abs"] = y_abs
        self.df.at[self.index, "annotation_status"] = "done"
        self.df.at[self.index, "annotation_source"] = source
        self.manual_xy_patch = None

    def mark_bad(self) -> None:
        self.df.at[self.index, "annotation_status"] = "bad"
        self.df.at[self.index, "annotation_source"] = "bad"
        self.df.at[self.index, "manual_label_x_patch"] = np.nan
        self.df.at[self.index, "manual_label_y_patch"] = np.nan
        self.df.at[self.index, "manual_label_x_abs"] = np.nan
        self.df.at[self.index, "manual_label_y_abs"] = np.nan
        self.manual_xy_patch = None

    def clear_label(self) -> None:
        self.df.at[self.index, "annotation_status"] = "pending"
        self.df.at[self.index, "annotation_source"] = ""
        self.df.at[self.index, "manual_label_x_patch"] = np.nan
        self.df.at[self.index, "manual_label_y_patch"] = np.nan
        self.df.at[self.index, "manual_label_x_abs"] = np.nan
        self.df.at[self.index, "manual_label_y_abs"] = np.nan
        self.manual_xy_patch = None

    def move(self, delta: int) -> None:
        self.index = int(np.clip(self.index + int(delta), 0, len(self.df) - 1))
        row = self.current_row()
        if np.isfinite(row.get("manual_label_x_patch", np.nan)) and np.isfinite(row.get("manual_label_y_patch", np.nan)):
            self.manual_xy_patch = (float(row["manual_label_x_patch"]), float(row["manual_label_y_patch"]))
        else:
            self.manual_xy_patch = None

    def load_patch(self, row: pd.Series) -> np.ndarray:
        patch = cv2.imread(str(row["patch_path"]), cv2.IMREAD_GRAYSCALE)
        if patch is None:
            raise FileNotFoundError(f"Could not read patch: {row['patch_path']}")
        return patch

    def draw_point(self, canvas: np.ndarray, xy_patch: Tuple[float, float], color: Tuple[int, int, int], label: str) -> None:
        x_patch, y_patch = xy_patch
        x = int(round(x_patch * self.display_scale))
        y = int(round(y_patch * self.display_scale))
        radius = max(3, self.display_scale)
        cv2.circle(canvas, (x, y), radius, color, 1, lineType=cv2.LINE_AA)
        cv2.drawMarker(canvas, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=radius * 2, thickness=1)
        cv2.putText(
            canvas,
            label,
            (x + 6, max(12, y - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            color,
            1,
            cv2.LINE_AA,
        )

    def render(self) -> np.ndarray:
        row = self.current_row()
        patch = self.load_patch(row)
        canvas = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
        canvas = cv2.resize(
            canvas,
            (patch.shape[1] * self.display_scale, patch.shape[0] * self.display_scale),
            interpolation=cv2.INTER_NEAREST,
        )

        # Grid
        for x in range(0, canvas.shape[1], self.display_scale * 8):
            cv2.line(canvas, (x, 0), (x, canvas.shape[0]), (230, 230, 230), 1)
        for y in range(0, canvas.shape[0], self.display_scale * 8):
            cv2.line(canvas, (0, y), (canvas.shape[1], y), (230, 230, 230), 1)

        self.draw_point(canvas, (float(row["anchor_x_patch"]), float(row["anchor_y_patch"])), (0, 255, 255), "anchor")
        self.draw_point(canvas, (float(row["coarse_x_patch"]), float(row["coarse_y_patch"])), (255, 255, 0), "coarse")
        self.draw_point(canvas, (float(row["selected_x_patch"]), float(row["selected_y_patch"])), (0, 165, 255), "selected")

        if np.isfinite(row.get("manual_label_x_patch", np.nan)) and np.isfinite(row.get("manual_label_y_patch", np.nan)):
            self.draw_point(
                canvas,
                (float(row["manual_label_x_patch"]), float(row["manual_label_y_patch"])),
                (0, 0, 255),
                "label",
            )
        if self.manual_xy_patch is not None:
            self.draw_point(canvas, self.manual_xy_patch, (0, 0, 180), "click")

        image_label = row.get("image_relative_path", row.get("image_file", ""))
        text_lines = [
            f"{self.index + 1}/{len(self.df)}  {image_label}",
            f"status={row['annotation_status']} source={row['annotation_source']}",
            "mouse: left click set point | n next | p prev | q save+quit",
            "space anchor | c coarse | s selected | r refined | m commit click | u clear | x bad",
        ]
        y0 = 20
        for line in text_lines:
            cv2.putText(canvas, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (40, 40, 40), 2, cv2.LINE_AA)
            cv2.putText(canvas, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            y0 += 22

        return canvas

    def on_mouse(self, event, x, y, flags, param) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        row = self.current_row()
        patch = self.load_patch(row)
        x_patch = np.clip(x / self.display_scale, 0, patch.shape[1] - 1)
        y_patch = np.clip(y / self.display_scale, 0, patch.shape[0] - 1)
        self.set_manual_point(x_patch, y_patch)

    def run(self) -> None:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, self.window_width, self.window_height)
        cv2.setMouseCallback(WINDOW_NAME, self.on_mouse)

        while True:
            canvas = self.render()
            self.last_render = canvas
            cv2.imshow(WINDOW_NAME, canvas)
            key = cv2.waitKey(30) & 0xFF

            if key == 255:
                continue
            if key == ord("q"):
                if self.manual_xy_patch is not None:
                    self.persist_manual_point()
                self.save()
                break
            if key == ord("n"):
                if self.manual_xy_patch is not None:
                    self.persist_manual_point()
                self.save()
                if self.index < len(self.df) - 1:
                    self.move(+1)
                continue
            if key == ord("p"):
                if self.manual_xy_patch is not None:
                    self.persist_manual_point()
                self.save()
                if self.index > 0:
                    self.move(-1)
                continue
            if key == ord(" "):
                self.use_existing_point("anchor")
                self.save()
                continue
            if key == ord("c"):
                self.use_existing_point("coarse")
                self.save()
                continue
            if key == ord("s"):
                self.use_existing_point("selected")
                self.save()
                continue
            if key == ord("r"):
                self.use_existing_point("refined")
                self.save()
                continue
            if key == ord("m"):
                self.persist_manual_point()
                self.save()
                continue
            if key == ord("u"):
                self.clear_label()
                self.save()
                continue
            if key == ord("x"):
                self.mark_bad()
                self.save()
                continue

        cv2.destroyAllWindows()


def main() -> None:
    parser = argparse.ArgumentParser(description="Review patch crops and correct tip locations with a simple OpenCV GUI.")
    parser.add_argument("--project_dir", type=str, default=None, help="Calibration project folder. Used to find the default CNN manifest.")
    parser.add_argument("--manifest", default=None, help="Path to manifest.csv created by prepare_tip_refinement_dataset.py")
    parser.add_argument(
        "--output-manifest",
        default=None,
        help="Where to save annotations. Defaults to <manifest stem>_annotated.csv",
    )
    parser.add_argument("--display-scale", type=int, default=DEFAULT_DISPLAY_SCALE, help="GUI zoom factor.")
    parser.add_argument("--window-width", type=int, default=DEFAULT_WINDOW_WIDTH, help="Initial GUI window width in pixels.")
    parser.add_argument("--window-height", type=int, default=DEFAULT_WINDOW_HEIGHT, help="Initial GUI window height in pixels.")
    parser.add_argument("--start-index", type=int, default=0, help="Optional starting row.")
    args = parser.parse_args()

    if args.manifest is None and args.project_dir is None:
        raise SystemExit("Provide --project_dir or --manifest.")

    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else default_dataset_dir(args.project_dir) / "manifest.csv"
    )
    output_manifest = (
        Path(args.output_manifest).expanduser().resolve()
        if args.output_manifest
        else manifest_path.with_name(f"{manifest_path.stem}_annotated.csv")
    )

    input_manifest = manifest_path
    if args.manifest is None and args.output_manifest is None and output_manifest.exists():
        input_manifest = output_manifest
        print(f"Resuming annotations from: {input_manifest}")

    df = read_manifest(str(input_manifest))
    if len(df) == 0:
        raise ValueError("Manifest is empty.")

    annotator = Annotator(
        df=df,
        output_path=str(output_manifest),
        display_scale=args.display_scale,
        window_width=args.window_width,
        window_height=args.window_height,
    )
    annotator.index = int(np.clip(args.start_index, 0, len(df) - 1))
    annotator.move(0)
    annotator.run()
    print(f"Saved annotations to: {output_manifest}")


if __name__ == "__main__":
    main()
