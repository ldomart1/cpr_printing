#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Offline runner for CTR shadow calibration.

This script keeps the offline workflow:
- choose or pass a folder of images
- create/reuse a calibration project with `raw_image_data_folder`
- use the first image for crop and ruler setup

The actual image analysis and postprocessing are performed by
`shadow_calibration.CTR_Shadow_Calibration` without monkey-patching or
re-implementing its analysis pipeline.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from shadow_calibration import CTR_Shadow_Calibration


IMAGE_EXTS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")


def list_images(folder: Path) -> list[Path]:
    images = [path for path in folder.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTS]
    images.sort()
    return images


def ensure_project_from_raw(raw_dir: Path, project_dir: Path, link_mode: str = "symlink") -> Path:
    project_dir.mkdir(parents=True, exist_ok=True)
    raw_out = project_dir / "raw_image_data_folder"
    raw_out.mkdir(parents=True, exist_ok=True)

    images = list_images(raw_dir)
    if not images:
        raise RuntimeError(f"No images found in {raw_dir}")

    for src in images:
        dst = raw_out / src.name
        if dst.exists():
            continue
        if link_mode == "symlink":
            try:
                os.symlink(src.resolve(), dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)

    return raw_out


def _json_ready(value: Any) -> Any:
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


def choose_folder_dialog() -> Path | None:
    try:
        import tkinter as tk
        from tkinter import filedialog
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)
    try:
        selected = filedialog.askdirectory(title="Select calibration project folder or raw image folder")
    finally:
        root.destroy()

    if not selected:
        return None
    return Path(selected).expanduser().resolve()


def set_ruler_reference(
    cal: CTR_Shadow_Calibration,
    p1: np.ndarray | None,
    p2: np.ndarray | None,
    known_distance_mm: float,
) -> None:
    cal.ruler_ref_p1_px = None
    cal.ruler_ref_p2_px = None
    cal.ruler_ref_distance_mm = None
    cal.ruler_mm_per_px = None
    cal.ruler_px_per_mm = None
    cal.ruler_axis_unit = None
    cal.ruler_axis_perp_unit = None
    cal.ruler_calib_meta = None

    if p1 is None or p2 is None:
        return

    axis_vec = np.asarray(p2, dtype=np.float64) - np.asarray(p1, dtype=np.float64)
    pixel_dist = float(np.linalg.norm(axis_vec))
    if pixel_dist <= 1e-9:
        raise ValueError("Ruler points are too close together.")

    axis_unit = axis_vec / pixel_dist
    axis_perp_unit = np.array([axis_unit[1], -axis_unit[0]], dtype=np.float64)

    cal.ruler_ref_p1_px = tuple(np.round(p1).astype(int))
    cal.ruler_ref_p2_px = tuple(np.round(p2).astype(int))
    cal.ruler_ref_distance_mm = float(known_distance_mm)
    cal.ruler_mm_per_px = float(known_distance_mm / pixel_dist)
    cal.ruler_px_per_mm = float(pixel_dist / known_distance_mm)
    cal.ruler_axis_unit = axis_unit
    cal.ruler_axis_perp_unit = axis_perp_unit
    cal.ruler_calib_meta = {
        "source": "offline_first_image_manual_setup",
        "known_distance_mm": float(known_distance_mm),
        "pixel_distance": float(pixel_dist),
        "crop": dict(cal.analysis_crop),
        "p1_px": [int(cal.ruler_ref_p1_px[0]), int(cal.ruler_ref_p1_px[1])],
        "p2_px": [int(cal.ruler_ref_p2_px[0]), int(cal.ruler_ref_p2_px[1])],
    }


def interactive_crop_and_ruler_from_image(
    image_bgr: np.ndarray,
    default_crop: dict[str, int],
    ruler_known_mm: float,
) -> tuple[dict[str, int], np.ndarray | None, np.ndarray | None]:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image passed to setup GUI.")

    img_h, img_w = image_bgr.shape[:2]

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

    def nearest_corner(mx: int, my: int) -> str | None:
        best_name = None
        best_dist = 1e18
        for name, (cx, cy) in corners.items():
            dist = (mx - cx) ** 2 + (my - cy) ** 2
            if dist < best_dist:
                best_dist = dist
                best_name = name
        if best_dist <= drag_threshold_px ** 2:
            return best_name
        return None

    def clamp_rect() -> None:
        xs = [point[0] for point in corners.values()]
        ys = [point[1] for point in corners.values()]
        x0 = int(np.clip(min(xs), 0, img_w - 2))
        x1 = int(np.clip(max(xs), x0 + 1, img_w - 1))
        y0 = int(np.clip(min(ys), 0, img_h - 2))
        y1 = int(np.clip(max(ys), y0 + 1, img_h - 1))
        corners["tl"] = [x0, y0]
        corners["tr"] = [x1, y0]
        corners["br"] = [x1, y1]
        corners["bl"] = [x0, y1]

    def on_mouse_crop(event: int, mx: int, my: int, flags: int, param: Any) -> None:
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

    crop_window = "Offline Crop Setup"
    crop_accepted = False

    cv2.namedWindow(crop_window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(crop_window, on_mouse_crop)

    print("\nManual crop setup:")
    print("- Drag crop rectangle corners with the mouse.")
    print("- Press ENTER or SPACE to confirm.")
    print("- Press R to reset.")
    print("- Press Q or ESC to keep the default crop.")

    try:
        while True:
            if cv2.getWindowProperty(crop_window, cv2.WND_PROP_VISIBLE) < 1:
                break

            clamp_rect()
            x0, y0 = corners["tl"]
            x1, y1 = corners["br"]

            display = image_bgr.copy()
            cv2.rectangle(display, (x0, y0), (x1, y1), (0, 255, 0), 2)
            for point in corners.values():
                cv2.circle(display, tuple(point), 6, (0, 255, 255), -1)
            cv2.putText(
                display,
                f"x:[{x0},{x1}] y:[{y0},{y1}]  ENTER confirm | R reset | Q default",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            cv2.imshow(crop_window, display)

            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32):
                crop_accepted = True
                break
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("r"), ord("R")):
                corners["tl"] = [x_min, y_min]
                corners["tr"] = [x_max, y_min]
                corners["br"] = [x_max, y_max]
                corners["bl"] = [x_min, y_max]
    finally:
        cv2.setMouseCallback(crop_window, lambda *args: None)
        cv2.destroyWindow(crop_window)
        cv2.waitKey(1)

    if crop_accepted:
        x0, y0 = corners["tl"]
        x1, y1 = corners["br"]
        analysis_crop = {
            "crop_width_min": int(x0),
            "crop_width_max": int(x1),
            "crop_height_min": int(img_h - y1),
            "crop_height_max": int(img_h - y0),
        }
    else:
        analysis_crop = dict(default_crop)

    ruler_points: list[tuple[int, int]] = []
    ruler_window = "Offline Ruler Setup"
    ruler_confirmed = False
    ruler_skipped = False

    def on_mouse_ruler(event: int, mx: int, my: int, flags: int, param: Any) -> None:
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if len(ruler_points) >= 2:
            return
        mx = int(np.clip(mx, 0, img_w - 1))
        my = int(np.clip(my, 0, img_h - 1))
        ruler_points.append((mx, my))

    print("\nRuler setup:")
    print(f"- Click two points spanning {float(ruler_known_mm):.3f} mm.")
    print("- Press ENTER or SPACE to confirm.")
    print("- Press R to reset the two points.")
    print("- Press Q or ESC to skip ruler scaling.")

    cv2.namedWindow(ruler_window, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(ruler_window, on_mouse_ruler)

    try:
        while True:
            if cv2.getWindowProperty(ruler_window, cv2.WND_PROP_VISIBLE) < 1:
                ruler_skipped = True
                break

            display = image_bgr.copy()
            cx0 = analysis_crop["crop_width_min"]
            cx1 = analysis_crop["crop_width_max"]
            cy0 = img_h - analysis_crop["crop_height_max"]
            cy1 = img_h - analysis_crop["crop_height_min"]
            cv2.rectangle(display, (cx0, cy0), (cx1, cy1), (0, 255, 0), 2)

            dist_px = None
            mm_per_px = None
            for idx, point in enumerate(ruler_points):
                cv2.circle(display, point, 7, (0, 255, 255), -1)
                cv2.putText(
                    display,
                    f"P{idx + 1}",
                    (point[0] + 8, point[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 255),
                    2,
                )

            if len(ruler_points) == 2:
                p1 = np.asarray(ruler_points[0], dtype=float)
                p2 = np.asarray(ruler_points[1], dtype=float)
                dist_px = float(np.linalg.norm(p2 - p1))
                if dist_px > 1e-9:
                    mm_per_px = float(ruler_known_mm / dist_px)
                cv2.line(display, ruler_points[0], ruler_points[1], (50, 200, 255), 2)

            cv2.putText(
                display,
                "Pick 2 ruler points: ENTER confirm | R reset | Q skip",
                (20, 35),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )
            if dist_px is not None and mm_per_px is not None:
                cv2.putText(
                    display,
                    f"dist_px={dist_px:.2f} | mm/px={mm_per_px:.6f}",
                    (20, 68),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.65,
                    (50, 255, 120),
                    2,
                )

            cv2.imshow(ruler_window, display)
            key = cv2.waitKey(20) & 0xFF
            if key in (13, 32):
                if len(ruler_points) < 2:
                    print("Select two ruler points first.")
                    continue
                p1 = np.asarray(ruler_points[0], dtype=float)
                p2 = np.asarray(ruler_points[1], dtype=float)
                if float(np.linalg.norm(p2 - p1)) < 5.0:
                    print("Selected ruler points are too close. Re-pick them.")
                    continue
                ruler_confirmed = True
                break
            if key in (27, ord("q"), ord("Q")):
                ruler_skipped = True
                break
            if key in (ord("r"), ord("R")):
                ruler_points.clear()
    finally:
        cv2.setMouseCallback(ruler_window, lambda *args: None)
        cv2.destroyWindow(ruler_window)
        cv2.waitKey(1)
        cv2.destroyAllWindows()
        cv2.waitKey(1)

    if ruler_confirmed:
        return analysis_crop, np.asarray(ruler_points[0], dtype=np.float64), np.asarray(ruler_points[1], dtype=np.float64)
    if ruler_skipped:
        print("Ruler calibration skipped.")
    return analysis_crop, None, None


def resolve_input_folders(
    project_dir: str | None,
    raw_dir: str | None,
    link_mode: str,
) -> tuple[Path, Path]:
    project_path = Path(project_dir).expanduser().resolve() if project_dir else None
    raw_path = Path(raw_dir).expanduser().resolve() if raw_dir else None

    if project_path is None and raw_path is None:
        selected = choose_folder_dialog()
        if selected is None:
            raise SystemExit("No folder selected. Provide --project_dir or --raw_dir.")
        if (selected / "raw_image_data_folder").is_dir():
            project_path = selected
        else:
            raw_path = selected

    if project_path is not None and (project_path / "raw_image_data_folder").is_dir():
        return project_path, (project_path / "raw_image_data_folder").resolve()

    if raw_path is None:
        raise SystemExit("Could not resolve input folder.")

    if raw_path.name == "raw_image_data_folder":
        if project_path is None:
            project_path = raw_path.parent
        elif project_path != raw_path.parent:
            return project_path.resolve(), ensure_project_from_raw(raw_path, project_path, link_mode=link_mode).resolve()
        return project_path.resolve(), raw_path.resolve()

    if project_path is None:
        project_path = raw_path.parent / f"{raw_path.name}_offline_project"

    raw_folder = ensure_project_from_raw(raw_path, project_path, link_mode=link_mode)
    return project_path.resolve(), raw_folder.resolve()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Offline runner for shadow_calibration.py")
    parser.add_argument("--project_dir", type=str, default=None, help="Existing project folder containing raw_image_data_folder")
    parser.add_argument("--raw_dir", type=str, default=None, help="Folder of raw images or raw_image_data_folder")
    parser.add_argument("--link_mode", type=str, default="symlink", choices=["symlink", "copy"])
    parser.add_argument("--threshold", type=int, default=200)
    parser.add_argument("--robot_name", type=str, default="calibrated_robot")
    parser.add_argument("--width_in_pixels", type=float, default=3025.0)
    parser.add_argument("--width_in_mm", type=float, default=140.0)
    parser.add_argument("--fit_model", type=str, default="pchip", choices=["cubic", "pchip"])
    parser.add_argument("--save_plots", action="store_true")
    parser.add_argument("--ruler_mm", type=float, default=150.0)
    parser.add_argument("--save_analysis_config", action="store_true")

    parser.add_argument("--camera_calibration_file", type=str, default=None, help="Path to camera calibration .npz")
    parser.add_argument("--board_reference_image", type=str, default=None, help="Path to checkerboard/board reference image")
    parser.add_argument("--board_inner_corners", type=str, default=None, help="Checkerboard inner corners as Nx,Ny")
    parser.add_argument("--board_square_size_mm", type=float, default=None)
    parser.add_argument("--board_xz_axis_sign", type=int, choices=[-1, 1], default=1, help="Set to -1 to flip the calibrated checkerboard-reference x and z axes.")
    parser.add_argument("--board_no_undistort", action="store_true")

    parser.add_argument("--tip_parallel_section_near_r", type=float, default=1.0)
    parser.add_argument("--tip_parallel_section_far_r", type=float, default=8.0)
    parser.add_argument("--tip_parallel_scan_half_r", type=float, default=3.0)
    parser.add_argument("--tip_parallel_num_sections", type=int, default=9)
    parser.add_argument("--tip_parallel_cross_step_px", type=float, default=0.5)
    parser.add_argument("--tip_parallel_ray_step_px", type=float, default=0.5)
    parser.add_argument("--tip_parallel_ray_max_len_r", type=float, default=16.0)
    parser.add_argument("--tip_refiner_model", type=str, default=None, help="Path to cnn/train_tip_refiner.py best_tip_refiner.pt")
    parser.add_argument("--tip_refiner_anchor", type=str, default=None, choices=["coarse", "selected", "refined"], help="Patch anchor for CNN inference. Defaults to the model checkpoint anchor.")
    parser.add_argument("--tip_refiner_compare_only", action="store_true", help="Save tip_locations_cnn.* but keep classical selected tips for postprocessing.")

    parser.add_argument("--export_skeleton", action="store_true")
    parser.add_argument("--skeleton_diameter_mm", type=float, default=1.51)
    parser.add_argument("--skeleton_links", type=int, default=6)
    parser.add_argument("--skeleton_reference_stl", action="store_true")
    return parser


def parse_inner_corners(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("x", ",")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected inner corners as 'Nx,Ny', got {value!r}")
    return int(parts[0]), int(parts[1])


def main() -> None:
    args = build_arg_parser().parse_args()

    project_dir, raw_folder = resolve_input_folders(args.project_dir, args.raw_dir, args.link_mode)
    images = list_images(raw_folder)
    if not images:
        raise SystemExit(f"No images found in {raw_folder}")

    first_image_path = images[0]
    first_image = cv2.imread(str(first_image_path), cv2.IMREAD_COLOR)
    if first_image is None:
        raise SystemExit(f"Could not read first image: {first_image_path}")

    cal = CTR_Shadow_Calibration(
        parent_directory=str(project_dir.parent),
        project_name=project_dir.name,
        allow_existing=True,
        add_date=False,
    )
    cal.calibration_data_folder = str(project_dir)

    cal.tip_parallel_section_near_r = float(args.tip_parallel_section_near_r)
    cal.tip_parallel_section_far_r = float(args.tip_parallel_section_far_r)
    cal.tip_parallel_scan_half_r = float(args.tip_parallel_scan_half_r)
    cal.tip_parallel_num_sections = int(args.tip_parallel_num_sections)
    cal.tip_parallel_cross_step_px = float(args.tip_parallel_cross_step_px)
    cal.tip_parallel_ray_step_px = float(args.tip_parallel_ray_step_px)
    cal.tip_parallel_ray_max_len_r = float(args.tip_parallel_ray_max_len_r)

    if args.tip_refiner_model:
        cal.load_tip_refiner_model(
            str(Path(args.tip_refiner_model).expanduser().resolve()),
            anchor_name=args.tip_refiner_anchor,
            use_as_selected=(not args.tip_refiner_compare_only),
        )

    if args.camera_calibration_file:
        cal.load_camera_calibration(str(Path(args.camera_calibration_file).expanduser().resolve()))
        if args.board_reference_image:
            processed_dir = project_dir / "processed_image_data_folder"
            processed_dir.mkdir(parents=True, exist_ok=True)
            board_result = cal.estimate_board_reference_from_image(
                str(Path(args.board_reference_image).expanduser().resolve()),
                inner_corners=parse_inner_corners(args.board_inner_corners),
                square_size_mm=args.board_square_size_mm,
                board_xz_axis_sign=float(args.board_xz_axis_sign),
                use_undistort=(not args.board_no_undistort),
                draw_debug=True,
                save_debug_path=str(processed_dir / "board_reference_debug.png"),
            )
            print(f"Board reference estimated: {board_result.get('board_type', 'board')}")
        elif args.board_reference_image is None:
            print("Camera calibration loaded; board reference image not provided.")
    elif args.board_reference_image:
        print("Board reference image was provided without camera calibration; skipping board-reference setup.")

    print(f"\nUsing first image for setup: {first_image_path.name}")
    analysis_crop, ruler_p1, ruler_p2 = interactive_crop_and_ruler_from_image(
        first_image,
        default_crop=dict(cal.default_analysis_crop),
        ruler_known_mm=float(args.ruler_mm),
    )
    cal.analysis_crop = dict(analysis_crop)
    set_ruler_reference(cal, ruler_p1, ruler_p2, known_distance_mm=float(args.ruler_mm))

    if args.save_analysis_config:
        config = cal.get_analysis_reference_info()
        config["board_reference"] = {
            "camera_calib_path": cal.camera_calib_path,
            "camera_calib_meta": cal.camera_calib_meta,
            "board_reference_image_path": cal.board_reference_image_path,
            "board_pose": cal.board_pose,
            "true_vertical_img_unit": cal.true_vertical_img_unit,
            "board_homography_px_from_mm": cal.board_homography_px_from_mm,
            "board_homography_mm_from_px": cal.board_homography_mm_from_px,
            "board_px_per_mm_local": cal.board_px_per_mm_local,
            "board_mm_per_px_local": cal.board_mm_per_px_local,
            "board_xz_axis_sign": cal.board_xz_axis_sign,
        }
        config_path = project_dir / "analysis_reference.json"
        with open(config_path, "w", encoding="utf-8") as handle:
            json.dump(_json_ready(config), handle, indent=2)
        print(f"Saved analysis config to {config_path}")

    print("\nRunning analyze_data_batch...")
    cal.analyze_data_batch(threshold=int(args.threshold))

    print("\nRunning postprocess_calibration_data...")
    cal.postprocess_calibration_data(
        width_in_pixels=float(args.width_in_pixels),
        width_in_mm=float(args.width_in_mm),
        robot_name=str(args.robot_name),
        save_plots=bool(args.save_plots),
        fit_model=str(args.fit_model),
        export_skeleton=bool(args.export_skeleton),
        skeleton_diameter_mm=float(args.skeleton_diameter_mm),
        skeleton_links=int(args.skeleton_links),
        skeleton_reference_stl=bool(args.skeleton_reference_stl),
    )

    print(f"\nDone. Outputs are in {project_dir / 'processed_image_data_folder'}")


if __name__ == "__main__":
    main()
