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
    project_name='Test_Calibration_2026-02-27_00',
    allow_existing= True,
    add_date=False
)

print("Calibration object created!")

# --- CONFIG ---
MANUAL_CROP_ADJUSTMENT = True
THRESHOLD = 200
PULL_B_START = 0.0
PULL_B_STEPS = 25
PULL_B_STEP_SIZE = -0.2


PROBE_MODE = "middle"  # "middle" | "five"

if PROBE_MODE == "middle":
    probe_points = [(70.0, 0.0, -140.0)]
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

# Example usage:
cal.connect_to_camera(cam_port=0, show_preview=False)
cal.setup_analysis_crop(enable_manual_adjustment=MANUAL_CROP_ADJUSTMENT)
#cal.disconnect_camera()
cal.connect_to_robot()
cal.calibrate(
    jogging_feedrate=600,
    probe_points=probe_points,
    b_start=PULL_B_START,
    b_steps=PULL_B_STEPS,
    b_step_size=PULL_B_STEP_SIZE,
)
cal.analyze_data_batch(
    threshold=THRESHOLD,
)
cal.postprocess_calibration_data(save_plots=True)
