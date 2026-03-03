# -*- coding: utf-8 -*-
"""
Long-run webcam monitor w/ raw video recording.

What it does:
  - Captures frames from your webcam (OpenCV decoded frames).
  - Records "raw" video at the camera's *captured* resolution/FPS (no resizing on record).
  - Saves one file per start/stop recording session.
  - Optional disk-space guard to stop starting new recordings if low on space.

Keys:
  p → photo
  v → start/stop monitoring recording
  c/SPACE/k/u/r/f/←/→ → same as your calibration/focus controls
  q → quit

Notes:
  - "Native camera format" isn’t directly available via OpenCV; OpenCV gives decoded frames.
    This records “as captured” (resolution unchanged).
"""

import cv2
import numpy as np
import shutil
from pathlib import Path
from datetime import datetime

# --- Hard-disable OpenCL (common macOS crash workaround) ---
try:
    cv2.ocl.setUseOpenCL(False)
    print("[INFO] OpenCL disabled:", cv2.ocl.useOpenCL())
except Exception as e:
    print("[INFO] Could not disable OpenCL:", e)

# -----------------------
# User settings
# -----------------------
PORT = 0

# Base folder
BASE_DIR = Path("captures")
BASE_DIR.mkdir(parents=True, exist_ok=True)

# Video folder
RAW_DIR = BASE_DIR / "videos_raw"
RAW_DIR.mkdir(exist_ok=True)

# Disk guard (stop starting new recordings if free space below this)
MIN_FREE_GB = 5.0

# -----------------------
# Calibration configuration (unchanged)
# -----------------------
CHESSBOARD_INNER_CORNERS = (11, 7)
SQUARE_SIZE_M = 0.020
MIN_CALIB_FRAMES = 12
USE_SB_DETECTOR = False

# ---- Choose backend ----
cap = cv2.VideoCapture(PORT, cv2.CAP_AVFOUNDATION)
# Alternatives if needed:
# cap = cv2.VideoCapture(PORT)
# cap = cv2.VideoCapture(PORT, cv2.CAP_QT)

if not cap.isOpened():
    raise RuntimeError("Could not open camera")

print("""
Controls:
  p → take photo
  v → start/stop video
  c → toggle calibration mode (and capture if board found)
  SPACE → capture calibration frame (only in calibration mode, only if board found)
  k → compute calibration (requires enough captured calibration frames)
  u → toggle undistortion preview (requires calibration)
  r → reset calibration captures

  f → toggle FOCUS calibration mode
  ← / → → adjust focus (300–900) while in focus mode

  q → quit
""")

# ---- Diagnostics for focus/autofocus capability ----
try:
    backend = cap.getBackendName()
except Exception:
    backend = "unknown"
print(f"[INFO] Capture backend: {backend}")

try:
    init_af = cap.get(cv2.CAP_PROP_AUTOFOCUS)
    init_f = cap.get(cv2.CAP_PROP_FOCUS)
    print(f"[INFO] Initial readback: AF={init_af}  Focus={init_f}")
except Exception as e:
    print("[INFO] Could not read autofocus/focus properties:", e)

# -----------------------
# Helpers
# -----------------------
def now_stamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(str(path))
    return usage.free / (1024 ** 3)

# -----------------------
# Video recording state
# -----------------------
recording = False
video_writer = None
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
current_raw_path: Path | None = None

def start_recording(frame_shape_hw):
    """
    Start a new raw video file.
    Records frames at captured resolution (no resizing).
    Writer FPS: we attempt to use camera FPS readback; fallback to 30.
    """
    global video_writer, current_raw_path

    # Disk guard before starting a recording
    gb = free_gb(BASE_DIR)
    if gb < MIN_FREE_GB:
        raise RuntimeError(f"Low disk space: {gb:.2f} GB free (< {MIN_FREE_GB} GB). Not starting recording.")

    h, w = frame_shape_hw
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps != fps or fps < 1:
        fps = 30.0  # fallback if camera doesn't report FPS

    current_raw_path = RAW_DIR / f"raw_{now_stamp()}.mp4"
    video_writer = cv2.VideoWriter(str(current_raw_path), fourcc, float(fps), (w, h))
    if not video_writer.isOpened():
        video_writer = None
        raise RuntimeError("Could not open VideoWriter for raw recording.")
    print(f"🎥 Recording started → {current_raw_path}  (writer_fps≈{fps:.2f}, size={w}x{h})")

def stop_recording():
    """
    Close current writer.
    """
    global video_writer, current_raw_path
    if video_writer is not None:
        try:
            video_writer.release()
        except Exception:
            pass
        video_writer = None
    if current_raw_path is not None:
        print(f"💾 Saved video → {current_raw_path}")
        current_raw_path = None

# -----------------------
# Calibration state (unchanged)
# -----------------------
calib_mode = False
undistort_preview = False
objpoints = []
imgpoints = []
calib_images = 0
K = None
dist = None

focus_mode = False
FOCUS_MIN = 300
FOCUS_MAX = 900
FOCUS_STEP = 10
focus_value = 600
last_focus_set_ok = None
last_af_set_ok = None

def apply_focus_controls_every_frame():
    global last_focus_set_ok, last_af_set_ok
    try:
        last_af_set_ok = bool(cap.set(cv2.CAP_PROP_AUTOFOCUS, 0))
    except Exception:
        last_af_set_ok = None
    try:
        last_focus_set_ok = bool(cap.set(cv2.CAP_PROP_FOCUS, float(focus_value)))
    except Exception:
        last_focus_set_ok = None

def try_detect_checkerboard(gray):
    if USE_SB_DETECTOR and hasattr(cv2, "findChessboardCornersSB"):
        found, corners = cv2.findChessboardCornersSB(gray, CHESSBOARD_INNER_CORNERS, None)
        if found:
            corners = corners.reshape(-1, 1, 2).astype(np.float32)
        return found, corners
    else:
        flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE
        found, corners = cv2.findChessboardCorners(gray, CHESSBOARD_INNER_CORNERS, flags)
        if not found:
            return False, None
        subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), subpix_criteria)
        return True, corners

def save_calibration(K, dist, rms):
    filename = datetime.now().strftime("calibration_webcam_%Y%m%d_%H%M%S.npz")
    path = BASE_DIR / filename
    np.savez(
        str(path),
        K=K,
        dist=dist,
        rms=rms,
        inner_corners=np.array(CHESSBOARD_INNER_CORNERS),
        square_size_m=SQUARE_SIZE_M,
    )
    return str(path)

objp = np.zeros((CHESSBOARD_INNER_CORNERS[0] * CHESSBOARD_INNER_CORNERS[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHESSBOARD_INNER_CORNERS[0], 0:CHESSBOARD_INNER_CORNERS[1]].T.reshape(-1, 2)
objp *= SQUARE_SIZE_M

LEFT_KEYS  = {81, 2424832, 65361}
RIGHT_KEYS = {83, 2555904, 65363}

# -----------------------
# Main loop with safe cleanup
# -----------------------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        display = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Apply focus controls continuously while in focus mode
        if focus_mode:
            apply_focus_controls_every_frame()

        # Calibration overlay
        found, corners = (False, None)
        if calib_mode:
            found, corners = try_detect_checkerboard(gray)
            if found:
                cv2.drawChessboardCorners(display, CHESSBOARD_INNER_CORNERS, corners, found)

            status = f"CALIB MODE  frames={calib_images}  found={found}"
            cv2.putText(display, status, (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)

            cv2.putText(display, "SPACE/c: capture   k: calibrate   r: reset",
                        (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Undistort preview
        if undistort_preview and K is not None and dist is not None:
            display = cv2.undistort(display, K, dist)

        # Recording behavior
        if recording:
            # Start recording if needed
            if video_writer is None:
                try:
                    h, w = frame.shape[:2]
                    start_recording((h, w))
                except Exception as e:
                    print(f"Could not start recording: {e}")
                    recording = False  # stop trying

            # Write frame if writer exists
            if video_writer is not None:
                cv2.putText(display, "● REC", (20, 110),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                try:
                    video_writer.write(frame)
                except Exception as e:
                    print(f"Write failed; stopping recording. {e}")
                    recording = False
                    stop_recording()

        # Focus overlay
        if focus_mode:
            try:
                af_read = cap.get(cv2.CAP_PROP_AUTOFOCUS)
            except Exception:
                af_read = float("nan")
            try:
                f_read = cap.get(cv2.CAP_PROP_FOCUS)
            except Exception:
                f_read = float("nan")

            af_ok_txt = "?" if last_af_set_ok is None else ("OK" if last_af_set_ok else "IGNORED")
            f_ok_txt = "?" if last_focus_set_ok is None else ("OK" if last_focus_set_ok else "IGNORED")

            cv2.putText(display,
                        f"FOCUS MODE  target={focus_value}  read={f_read:.1f}  set={f_ok_txt}",
                        (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            cv2.putText(display,
                        f"AUTOFOCUS  read={af_read:.1f}  set={af_ok_txt}   (f to exit)",
                        (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            cv2.putText(display,
                        "Use ←/→ to adjust focus (300–900)",
                        (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

        cv2.imshow("Camera", display)
        key = cv2.waitKeyEx(1)

        # Take photo
        if key == ord('p'):
            filename = f"photo_{now_stamp()}.png"
            path = BASE_DIR / filename
            cv2.imwrite(str(path), frame)
            print(f"📸 Saved {path}")

        # Toggle recording
        elif key == ord('v'):
            if not recording:
                recording = True
                print("Monitoring recording ON")
                print(f"    Raw → {RAW_DIR}")
            else:
                recording = False
                print("Monitoring recording OFF")
                stop_recording()

        # Toggle calibration mode (also captures if board found)
        elif key == ord('c'):
            if not calib_mode:
                calib_mode = True
                print("Calibration mode ON")
            else:
                if found and corners is not None:
                    objpoints.append(objp.copy())
                    imgpoints.append(corners.copy())
                    calib_images += 1
                    print(f"Captured calibration frame {calib_images}")
                else:
                    print("Checkerboard not found; move/tilt board and try again.")

        # Capture calibration frame with SPACE
        elif key == 32:
            if calib_mode:
                if found and corners is not None:
                    objpoints.append(objp.copy())
                    imgpoints.append(corners.copy())
                    calib_images += 1
                    print(f"Captured calibration frame {calib_images}")
                else:
                    print("Checkerboard not found; move/tilt board and try again.")

        # Compute calibration
        elif key == ord('k'):
            if len(objpoints) < MIN_CALIB_FRAMES:
                print(f"Need at least {MIN_CALIB_FRAMES} good frames; currently have {len(objpoints)}.")
            else:
                img_size = (frame.shape[1], frame.shape[0])
                rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
                    objpoints, imgpoints, img_size, None, None
                )
                path = save_calibration(K, dist, rms)
                print("🎯 Calibration complete")
                print("  RMS reprojection error:", rms)
                print("  Saved:", path)

        # Toggle undistortion preview
        elif key == ord('u'):
            if K is None or dist is None:
                print("No calibration yet. Press 'k' after capturing frames.")
            else:
                undistort_preview = not undistort_preview
                print("🪄 Undistort preview:", "ON" if undistort_preview else "OFF")

        # Reset calibration captures
        elif key == ord('r'):
            objpoints.clear()
            imgpoints.clear()
            calib_images = 0
            print("🔄 Reset calibration captures")

        # Toggle focus calibration mode
        elif key == ord('f'):
            focus_mode = not focus_mode
            if focus_mode:
                print("🔎 Focus calibration mode ON (use ←/→ to adjust)")
                apply_focus_controls_every_frame()
            else:
                print("🔎 Focus calibration mode OFF")

        # Adjust focus
        elif focus_mode and key in LEFT_KEYS:
            focus_value = max(FOCUS_MIN, focus_value - FOCUS_STEP)
            apply_focus_controls_every_frame()

        elif focus_mode and key in RIGHT_KEYS:
            focus_value = min(FOCUS_MAX, focus_value + FOCUS_STEP)
            apply_focus_controls_every_frame()

        # Quit
        elif key == ord('q'):
            break

finally:
    # Stop recording cleanly
    try:
        recording = False
        stop_recording()
    except Exception:
        pass

    # Release camera + UI
    try:
        cap.release()
    except Exception:
        pass
    try:
        cv2.destroyAllWindows()
    except Exception:
        pass

    print("✅ Exited cleanly.")
