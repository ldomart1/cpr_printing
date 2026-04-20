#!/usr/bin/env python3
import argparse
import inspect
import os
import sys


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
if SCRIPT_DIR not in sys.path:
    sys.path.append(SCRIPT_DIR)

from shadow_calibration import CTR_Shadow_Calibration
import shadow_calibration


DEFAULT_PROJECT_NAME = "Test_Calibration_2026-04-28_05"
DEFAULT_MANUAL_CROP_ADJUSTMENT = True
DEFAULT_THRESHOLD = 220
DEFAULT_PULL_B_START = 0.0
DEFAULT_PULL_B_STEPS = 40
DEFAULT_PULL_B_STEP_SIZE = -0.125
DEFAULT_CAMERA_CALIBRATION_FILE = os.path.join(SCRIPT_DIR, "captures/calibration_webcam_20260406_104136.npz")
DEFAULT_BOARD_REFERENCE_IMAGE = os.path.join(SCRIPT_DIR, "captures/photo_20260428_162904.png")
DEFAULT_BOARD_XZ_AXIS_SIGN = 1
DEFAULT_PROBE_MODE = "middle"
DEFAULT_FIT_MODEL = "pchip"
DEFAULT_TIP_REFINER_MODEL = os.path.join(
    SCRIPT_DIR,
    "CNN_Calib",
    "processed_image_data_folder",
    "tip_refinement_model",
    "best_tip_refiner.pt",
)
DEFAULT_TIP_REFINER_ANCHOR = None
DEFAULT_TIP_REFINER_COMPARE_ONLY = False


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Online runner for shadow_calibration.py")
    parser.add_argument("--project_name", type=str, default=DEFAULT_PROJECT_NAME)
    parser.add_argument("--manual_crop_adjustment", action="store_true", default=DEFAULT_MANUAL_CROP_ADJUSTMENT)
    parser.add_argument("--no_manual_crop_adjustment", dest="manual_crop_adjustment", action="store_false")
    parser.add_argument("--threshold", type=int, default=DEFAULT_THRESHOLD)
    parser.add_argument("--pull_b_start", type=float, default=DEFAULT_PULL_B_START)
    parser.add_argument("--pull_b_steps", type=int, default=DEFAULT_PULL_B_STEPS)
    parser.add_argument("--pull_b_step_size", type=float, default=DEFAULT_PULL_B_STEP_SIZE)
    parser.add_argument("--camera_calibration_file", type=str, default=DEFAULT_CAMERA_CALIBRATION_FILE)
    parser.add_argument("--board_reference_image", type=str, default=DEFAULT_BOARD_REFERENCE_IMAGE)
    parser.add_argument("--board_xz_axis_sign", type=int, choices=[-1, 1], default=DEFAULT_BOARD_XZ_AXIS_SIGN)
    parser.add_argument("--probe_mode", type=str, default=DEFAULT_PROBE_MODE, choices=["middle", "five"])
    parser.add_argument("--fit_model", type=str, default=DEFAULT_FIT_MODEL, choices=["cubic", "pchip"])
    parser.add_argument("--tip_refiner_model", type=str, default=DEFAULT_TIP_REFINER_MODEL)
    parser.add_argument("--tip_refiner_anchor", type=str, default=DEFAULT_TIP_REFINER_ANCHOR)
    parser.add_argument("--tip_refiner_compare_only", action="store_true", default=DEFAULT_TIP_REFINER_COMPARE_ONLY)
    parser.add_argument("--export_skeleton", dest="export_skeleton", action="store_true", default=True)
    parser.add_argument("--no_export_skeleton", dest="export_skeleton", action="store_false")
    parser.add_argument("--skeleton_diameter_mm", type=float, default=1.51)
    parser.add_argument("--skeleton_links", type=int, default=6)
    parser.add_argument("--skeleton_reference_stl", action="store_true")
    return parser


def probe_points_for_mode(probe_mode: str):
    if probe_mode == "middle":
        return [(100.0, 50.0, -170.0)]
    if probe_mode == "five":
        return [
            (30.0, 0.0, -70.0),
            (125.0, 0.0, -70.0),
            (30.0, 0.0, -110.0),
            (125.0, 0.0, -110.0),
            (77.5, 0.0, -100.0),
        ]
    raise ValueError(f"Unknown PROBE_MODE: {probe_mode}")


def main(args: argparse.Namespace) -> None:
    print(f"[DEBUG] SCRIPT_DIR={SCRIPT_DIR}")
    print(f"[DEBUG] shadow_calibration loaded from: {inspect.getsourcefile(shadow_calibration)}")

    cal = CTR_Shadow_Calibration(
        parent_directory=SCRIPT_DIR,
        project_name=str(args.project_name),
        allow_existing=True,
        add_date=False,
    )
    print("Calibration object created!")

    probe_points = probe_points_for_mode(str(args.probe_mode))

    camera_calibration_file = None if args.camera_calibration_file is None else os.path.expanduser(str(args.camera_calibration_file))
    board_reference_image = None if args.board_reference_image is None else os.path.expanduser(str(args.board_reference_image))
    if camera_calibration_file and os.path.isfile(camera_calibration_file):
        cal.load_camera_calibration(camera_calibration_file)
        if board_reference_image is not None:
            cal.estimate_board_reference_from_image(
                board_reference_image,
                board_xz_axis_sign=float(args.board_xz_axis_sign),
                draw_debug=True,
                save_debug_path=os.path.join(SCRIPT_DIR, "captures", "checkerboard_reference_debug.png"),
            )
    else:
        print(f"[WARN] Camera calibration file not found, continuing without it: {camera_calibration_file}")

    cal.tip_refine_mode = "coarse"
    cal.tip_parallel_section_near_r = 0.75
    cal.tip_parallel_section_far_r = 5.0
    cal.tip_parallel_scan_half_r = 2.5
    cal.tip_parallel_num_sections = 7
    cal.tip_parallel_cross_step_px = 0.5
    cal.tip_parallel_ray_step_px = 0.5
    cal.tip_parallel_ray_max_len_r = 10.0

    if args.tip_refiner_model:
        tip_refiner_model = os.path.expanduser(str(args.tip_refiner_model))
        if os.path.isfile(tip_refiner_model):
            cal.load_tip_refiner_model(
                tip_refiner_model,
                anchor_name=args.tip_refiner_anchor,
                use_as_selected=(not bool(args.tip_refiner_compare_only)),
            )
        else:
            raise FileNotFoundError(f"TIP_REFINER_MODEL not found: {tip_refiner_model}")

    cal.connect_to_camera(cam_port=0, show_preview=False)
    cal.setup_analysis_crop(enable_manual_adjustment=bool(args.manual_crop_adjustment))
    cal.connect_to_robot()
    cal.calibrate(
        jogging_feedrate=200,
        probe_points=probe_points,
        b_start=float(args.pull_b_start),
        b_steps=int(args.pull_b_steps),
        b_step_size=float(args.pull_b_step_size),
    )
    cal.analyze_data_batch(
        threshold=int(args.threshold),
    )
    cal.postprocess_calibration_data(
        save_plots=True,
        fit_model=str(args.fit_model),
        export_skeleton=bool(args.export_skeleton),
        skeleton_diameter_mm=float(args.skeleton_diameter_mm),
        skeleton_links=int(args.skeleton_links),
        skeleton_reference_stl=bool(args.skeleton_reference_stl),
    )


if __name__ == "__main__":
    parser = build_arg_parser()
    main(parser.parse_args())
