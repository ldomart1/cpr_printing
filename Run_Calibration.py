#!/usr/bin/env python3
import sys
import os

# Add the path to your shadow_calibration script
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(SCRIPT_DIR)

from shadow_calibration import CTR_Shadow_Calibration
print(f"[DEBUG] SCRIPT_DIR={SCRIPT_DIR}")
import inspect, shadow_calibration
print(f"[DEBUG] shadow_calibration loaded from: {inspect.getsourcefile(shadow_calibration)}")

# Create calibration object
cal = CTR_Shadow_Calibration(
    parent_directory= SCRIPT_DIR, 
    project_name='Test_Calibration_2026-04-20_00',
    allow_existing= True,
    add_date=False
)
print("Calibration object created!")


# --- CONFIG ---p
MANUAL_CROP_ADJUSTMENT = True
THRESHOLD = 220
PULL_B_START = 0.0
PULL_B_STEPS = 24
PULL_B_STEP_SIZE = -0.2
CAMERA_CALIBRATION_FILE = os.path.join(SCRIPT_DIR, "captures/calibration_webcam_20260406_104136.npz")
BOARD_REFERENCE_IMAGE = os.path.join(SCRIPT_DIR, "captures/photo_20260406_104134.png")

PROBE_MODE = "middle"  # "middle" | "five"
FIT_MODEL = "pchip"  # "pchip" | "cubic"

# Optional CNN tip refiner. Set TIP_REFINER_MODEL to None to use only the
# classical selected/coarse tip pipeline.
TIP_REFINER_MODEL = os.path.join(
    SCRIPT_DIR,
    "Test_Calibration_2026-04-07_02_daq",
    "processed_image_data_folder",
    "tip_refinement_model",
    "best_tip_refiner.pt",
)
TIP_REFINER_ANCHOR = None  # None uses the checkpoint anchor; or "coarse", "selected", "refined"
TIP_REFINER_COMPARE_ONLY = False  # True saves CNN tips but keeps classical tips for postprocessing.

if PROBE_MODE == "middle":
    probe_points = [(100.0, 52.0, -130.0)]
elif PROBE_MODE == "five":
    probe_points = [
        (30.0, 0.0, -70.0),
        (125.0, 0.0, -70.0),
        (30.0, 0.0, -110.0),
        (125.0, 0.0, -110.0),
        (77.5, 0.0, -100.0),
    ]
else:
    raise ValueError(f"Unknown PROBE_MODE: {PROBE_MODE}")

# Optional camera/board calibration load for calibrated vertical/mm conversion.
if os.path.isfile(CAMERA_CALIBRATION_FILE):
    cal.load_camera_calibration(CAMERA_CALIBRATION_FILE)
    if BOARD_REFERENCE_IMAGE is not None:
        cal.estimate_board_reference_from_image(
            BOARD_REFERENCE_IMAGE,
            draw_debug=True,
            save_debug_path=os.path.join(SCRIPT_DIR, "captures", "checkerboard_reference_debug.png"),
        )
else:
    print(f"[WARN] Camera calibration file not found, continuing without it: {CAMERA_CALIBRATION_FILE}")

# Tip selection
cal.tip_refine_mode = "coarse"
# other options: "auto", "parallel_centerline"

# Parallel-centerline tuning
cal.tip_parallel_section_near_r = 0.75
cal.tip_parallel_section_far_r = 5.0
cal.tip_parallel_scan_half_r = 2.5
cal.tip_parallel_num_sections = 7
cal.tip_parallel_cross_step_px = 0.5
cal.tip_parallel_ray_step_px = 0.5
cal.tip_parallel_ray_max_len_r = 10.0

if TIP_REFINER_MODEL is not None:
    if os.path.isfile(TIP_REFINER_MODEL):
        cal.load_tip_refiner_model(
            TIP_REFINER_MODEL,
            anchor_name=TIP_REFINER_ANCHOR,
            use_as_selected=(not TIP_REFINER_COMPARE_ONLY),
        )
    else:
        raise FileNotFoundError(f"TIP_REFINER_MODEL not found: {TIP_REFINER_MODEL}")

# Example usage:
cal.connect_to_camera(cam_port=0, show_preview=False)
cal.setup_analysis_crop(enable_manual_adjustment=MANUAL_CROP_ADJUSTMENT)
#cal.disconnect_camera()
cal.connect_to_robot()
cal.calibrate(
    jogging_feedrate=200,
    probe_points=probe_points,
    b_start=PULL_B_START,
    b_steps=PULL_B_STEPS,
    b_step_size=PULL_B_STEP_SIZE,
)
cal.analyze_data_batch(
    threshold=THRESHOLD,
)
cal.postprocess_calibration_data(save_plots=True, fit_model=FIT_MODEL)
