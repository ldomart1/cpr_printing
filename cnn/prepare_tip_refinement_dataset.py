from __future__ import annotations

import argparse
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from common_ctr_tip_refinement import (
    CTRSourceAdapter,
    detection_to_patch_record,
    list_images,
    parse_crop_string,
    safe_path_stem,
    write_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CTR_SOURCE = REPO_ROOT / "shadow_calibration.py"
DEFAULT_DATASET_NAME = "tip_refinement_dataset"


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


def ensure_project_from_raw(raw_dir: Path, project_dir: Path, link_mode: str = "symlink") -> Path:
    raw_dir = Path(raw_dir).expanduser().resolve()
    project_dir = Path(project_dir).expanduser().resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    raw_out = project_dir / "raw_image_data_folder"
    raw_out.mkdir(parents=True, exist_ok=True)

    images = list_images(str(raw_dir), recursive=True)
    if not images:
        raise RuntimeError(f"No images found in {raw_dir}")

    for src in images:
        rel = src.relative_to(raw_dir)
        dst = raw_out / rel
        if dst.exists():
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        if link_mode == "symlink":
            try:
                os.symlink(src.resolve(), dst)
            except OSError:
                shutil.copy2(src, dst)
        else:
            shutil.copy2(src, dst)

    return raw_out


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


def parse_inner_corners(value: str | None) -> tuple[int, int] | None:
    if value is None:
        return None
    text = str(value).strip().lower().replace("x", ",")
    parts = [part.strip() for part in text.split(",") if part.strip()]
    if len(parts) != 2:
        raise ValueError(f"Expected inner corners as 'Nx,Ny', got {value!r}")
    return int(parts[0]), int(parts[1])


def load_analysis_reference_crop(project_dir: Path) -> dict[str, int] | None:
    config_path = project_dir / "analysis_reference.json"
    if not config_path.exists():
        return None

    with config_path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    crop = data.get("analysis_crop")
    if crop is None:
        return None

    required = ("crop_width_min", "crop_width_max", "crop_height_min", "crop_height_max")
    if not all(key in crop for key in required):
        raise ValueError(f"Invalid analysis_crop in {config_path}")

    return {key: int(crop[key]) for key in required}


def json_ready(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    return value


def resolve_cli_path(value: str | None, base_dir: Path) -> Path | None:
    if value is None:
        return None
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = base_dir / path
    return path.resolve()


def interactive_crop_from_image(
    image_bgr: np.ndarray,
    default_crop: dict[str, int],
    label: str,
) -> dict[str, int]:
    if image_bgr is None or image_bgr.size == 0:
        raise ValueError("Empty image passed to crop GUI.")

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

    def reset_corners() -> None:
        corners["tl"] = [x_min, y_min]
        corners["tr"] = [x_max, y_min]
        corners["br"] = [x_max, y_max]
        corners["bl"] = [x_min, y_max]

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

    def on_mouse_crop(event: int, mx: int, my: int, flags: int, param) -> None:
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

    crop_window = "CNN Crop Setup"
    crop_accepted = False

    cv2.namedWindow(crop_window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(crop_window, 1400, 900)
    cv2.setMouseCallback(crop_window, on_mouse_crop)

    print(f"\nCrop setup for: {label}")
    print("- Drag crop rectangle corners with the mouse.")
    print("- Press ENTER or SPACE to confirm.")
    print("- Press R to reset.")
    print("- Press Q or ESC to keep the default crop for this folder.")

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
                f"{label} | x:[{x0},{x1}] y:[{y0},{y1}]  ENTER confirm | R reset | Q default",
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
                reset_corners()
    finally:
        cv2.setMouseCallback(crop_window, lambda *args: None)
        cv2.destroyWindow(crop_window)
        cv2.waitKey(1)

    if not crop_accepted:
        return dict(default_crop)

    x0, y0 = corners["tl"]
    x1, y1 = corners["br"]
    return {
        "crop_width_min": int(x0),
        "crop_width_max": int(x1),
        "crop_height_min": int(img_h - y1),
        "crop_height_max": int(img_h - y0),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Process a shadow-calibration project or raw image folder and export patch crops + a manifest for manual tip correction."
    )
    parser.add_argument("--ctr-source", default=str(DEFAULT_CTR_SOURCE), help="Path to the Python file containing CTR_Shadow_Calibration.")
    parser.add_argument("--project_dir", type=str, default=None, help="Existing calibration project folder containing raw_image_data_folder.")
    parser.add_argument("--raw_dir", type=str, default=None, help="Folder of raw images or raw_image_data_folder.")
    parser.add_argument("--link_mode", type=str, default="symlink", choices=["symlink", "copy"])
    parser.add_argument("--image-dir", default=None, help="Deprecated alias for --raw_dir.")
    parser.add_argument("--output-dir", default=None, help="Folder where patches and manifest.csv will be written. Defaults to <project>/processed_image_data_folder/tip_refinement_dataset.")
    parser.add_argument("--workspace-dir", default=None, help="Scratch workspace for the class constructor. Defaults to <output-dir>/workspace.")
    parser.add_argument("--project-name", default=None, help="Workspace project name passed into CTR_Shadow_Calibration.")
    parser.add_argument("--crop", default=None, help="Optional crop override as xmin,xmax,ymin,ymax using the class convention.")
    parser.add_argument(
        "--crop_gui_per_folder",
        "--crop-gui-per-folder",
        action="store_true",
        help="Open a crop GUI on the first image of each relative subfolder before processing its images.",
    )
    parser.add_argument(
        "--no-analysis-reference",
        action="store_true",
        help="Ignore <project>/analysis_reference.json and use --crop or the class/default crop.",
    )
    parser.add_argument("--save_analysis_config", action="store_true", help="Write the crop/board reference used by this run to <project>/analysis_reference.json.")
    parser.add_argument("--camera_calibration_file", type=str, default=None, help="Optional camera calibration .npz, accepted for parity with offline_run_calibration.py.")
    parser.add_argument("--board_reference_image", type=str, default=None, help="Optional checkerboard/board reference image.")
    parser.add_argument("--board_inner_corners", type=str, default=None, help="Checkerboard inner corners as Nx,Ny.")
    parser.add_argument("--board_square_size_mm", type=float, default=None)
    parser.add_argument("--board_no_undistort", action="store_true")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=128,
        help="Patch size in pixels. 96 or 128 are good starting points.",
    )
    parser.add_argument(
        "--anchor",
        choices=["coarse", "selected", "refined"],
        default="coarse",
        help="Point used as the patch center. 'coarse' matches the original plan; 'selected' uses your current local refinement output.",
    )
    parser.add_argument(
        "--threshold",
        type=int,
        default=-1,
        help="Binary threshold. Use -1 to mirror the current Otsu-based pipeline.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional max number of images to process.")
    args = parser.parse_args()

    invocation_cwd = Path.cwd()
    ctr_source_path = resolve_cli_path(args.ctr_source, invocation_cwd)
    camera_calibration_path = resolve_cli_path(args.camera_calibration_file, invocation_cwd)
    board_reference_path = resolve_cli_path(args.board_reference_image, invocation_cwd)

    raw_arg = args.raw_dir if args.raw_dir is not None else args.image_dir
    project_dir, raw_folder = resolve_input_folders(args.project_dir, raw_arg, args.link_mode)

    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else project_dir / "processed_image_data_folder" / DEFAULT_DATASET_NAME
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    workspace_dir = Path(args.workspace_dir).expanduser().resolve() if args.workspace_dir else output_dir / "workspace"
    patch_dir = output_dir / "patches"
    patch_dir.mkdir(parents=True, exist_ok=True)

    crop = parse_crop_string(args.crop)
    crop_source = "cli"
    if crop is None and not args.no_analysis_reference:
        crop = load_analysis_reference_crop(project_dir)
        crop_source = "analysis_reference.json" if crop is not None else "adapter_default"
    elif crop is None:
        crop_source = "adapter_default"

    adapter = CTRSourceAdapter(
        ctr_source_path=str(ctr_source_path),
        workspace_dir=str(workspace_dir),
        project_name=args.project_name or f"{project_dir.name}_tip_refinement_workspace",
        allow_existing=True,
    )

    if camera_calibration_path is not None:
        adapter.processor.load_camera_calibration(str(camera_calibration_path))
        if board_reference_path is not None:
            board_result = adapter.processor.estimate_board_reference_from_image(
                str(board_reference_path),
                inner_corners=parse_inner_corners(args.board_inner_corners),
                square_size_mm=args.board_square_size_mm,
                use_undistort=(not args.board_no_undistort),
                draw_debug=True,
                save_debug_path=str(output_dir / "board_reference_debug.png"),
            )
            print(f"Board reference estimated: {board_result.get('board_type', 'board')}")
        else:
            print("Camera calibration loaded; board reference image not provided.")
    elif board_reference_path is not None:
        print("Board reference image was provided without camera calibration; skipping board-reference setup.")

    image_paths = list_images(str(raw_folder), recursive=True)
    if args.limit is not None:
        image_paths = image_paths[: int(args.limit)]

    crop_by_folder: dict[str, dict[str, int]] = {}
    if args.crop_gui_per_folder:
        if not image_paths:
            print("No images found; skipping crop GUI setup.")
        else:
            grouped_paths: dict[str, list[Path]] = defaultdict(list)
            for image_path in image_paths:
                folder_key = image_path.relative_to(raw_folder).parent.as_posix()
                grouped_paths[folder_key].append(image_path)

            base_crop = dict(crop) if crop is not None else dict(adapter.processor.analysis_crop)
            current_default_crop = dict(base_crop)
            print(f"\nOpening crop GUI once per subfolder ({len(grouped_paths)} folders).")
            for folder_key in sorted(grouped_paths):
                first_image_path = grouped_paths[folder_key][0]
                first_image = cv2.imread(str(first_image_path), cv2.IMREAD_COLOR)
                if first_image is None:
                    raise RuntimeError(f"Could not read first image for crop setup: {first_image_path}")

                display_key = folder_key if folder_key != "." else "<raw_image_data_folder>"
                selected_crop = interactive_crop_from_image(
                    first_image,
                    default_crop=current_default_crop,
                    label=display_key,
                )
                crop_by_folder[folder_key] = dict(selected_crop)
                current_default_crop = dict(selected_crop)

    records = []
    failures = []

    for idx, image_path in enumerate(image_paths, start=1):
        image_rel_path = image_path.relative_to(raw_folder).as_posix()
        image_folder_key = image_path.relative_to(raw_folder).parent.as_posix()
        image_crop = crop_by_folder.get(image_folder_key, crop)
        patch_stem = f"{idx:06d}_{safe_path_stem(image_rel_path)}"
        print(f"[{idx}/{len(image_paths)}] {image_rel_path}")
        try:
            detection = adapter.detect_tip(
                image_path=str(image_path),
                crop=image_crop,
                threshold=args.threshold,
                anchor_name=args.anchor,
            )
            gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
            if gray is None:
                raise FileNotFoundError(f"Could not read grayscale image: {image_path}")

            record = detection_to_patch_record(
                detection=detection,
                image_gray=gray,
                patch_dir=str(patch_dir),
                patch_size=args.patch_size,
                manifest_prefix=args.anchor,
                patch_stem=patch_stem,
            )
            record["image_relative_path"] = image_rel_path
            record["image_folder"] = image_folder_key
            records.append(record)
        except Exception as exc:
            failures.append({"image_file": image_path.name, "image_relative_path": image_rel_path, "error": str(exc)})
            print(f"  ! failed: {exc}")

    manifest = pd.DataFrame.from_records(records)
    manifest_path = output_dir / "manifest.csv"
    write_manifest(manifest, str(manifest_path))

    summary = {
        "ctr_source": str(ctr_source_path),
        "project_dir": str(project_dir),
        "raw_folder": str(raw_folder),
        "output_dir": str(output_dir),
        "workspace_dir": str(workspace_dir),
        "anchor": args.anchor,
        "patch_size": int(args.patch_size),
        "threshold": int(args.threshold),
        "crop": crop,
        "crop_source": crop_source,
        "crop_gui_per_folder": bool(args.crop_gui_per_folder),
        "crop_by_folder": crop_by_folder,
        "camera_calibration_file": None if camera_calibration_path is None else str(camera_calibration_path),
        "board_reference_image": None if board_reference_path is None else str(board_reference_path),
        "num_images": len(image_paths),
        "recursive": True,
        "num_success": int(len(records)),
        "num_failures": int(len(failures)),
        "num_unique_patch_paths": int(manifest["patch_path"].nunique()) if "patch_path" in manifest else 0,
        "manifest_path": str(manifest_path),
    }

    (output_dir / "prepare_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    if failures:
        (output_dir / "prepare_failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")

    if args.save_analysis_config:
        config = adapter.processor.get_analysis_reference_info()
        if crop is not None:
            config["analysis_crop"] = dict(crop)
        if crop_by_folder:
            config["cnn_crop_by_folder"] = crop_by_folder
        config["board_reference"] = {
            "camera_calib_path": adapter.processor.camera_calib_path,
            "camera_calib_meta": adapter.processor.camera_calib_meta,
            "board_reference_image_path": adapter.processor.board_reference_image_path,
            "board_pose": adapter.processor.board_pose,
            "true_vertical_img_unit": adapter.processor.true_vertical_img_unit,
            "board_homography_px_from_mm": adapter.processor.board_homography_px_from_mm,
            "board_homography_mm_from_px": adapter.processor.board_homography_mm_from_px,
            "board_px_per_mm_local": adapter.processor.board_px_per_mm_local,
            "board_mm_per_px_local": adapter.processor.board_mm_per_px_local,
        }
        config_path = project_dir / "analysis_reference.json"
        config_path.write_text(json.dumps(json_ready(config), indent=2), encoding="utf-8")
        print(f"Saved analysis config to {config_path}")

    print("\nDone.")
    print(f"Project: {project_dir}")
    print(f"Raw images: {raw_folder}")
    print(f"Manifest: {manifest_path}")
    print(f"Success: {len(records)} / {len(image_paths)}")
    if failures:
        print(f"Failures logged to: {output_dir / 'prepare_failures.json'}")


if __name__ == "__main__":
    main()
