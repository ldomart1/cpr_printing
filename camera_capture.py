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
# Calibration configuration
# -----------------------
CHARUCO_SQUARES_X = 12
CHARUCO_SQUARES_Y = 8
CHARUCO_SQUARE_SIZE_M = 0.015
CHARUCO_MARKER_SIZE_M = 0.011
CHARUCO_DICT_NAME = "DICT_4X4_50"
MIN_CALIB_FRAMES = 12

aruco = cv2.aruco
ARUCO_DICT = aruco.getPredefinedDictionary(getattr(aruco, CHARUCO_DICT_NAME))

def make_charuco_board(use_legacy_pattern=False):
    board = aruco.CharucoBoard(
        (CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y),
        CHARUCO_SQUARE_SIZE_M,
        CHARUCO_MARKER_SIZE_M,
        ARUCO_DICT,
    )
    if use_legacy_pattern and hasattr(board, "setLegacyPattern"):
        board.setLegacyPattern(True)
    return board

CHARUCO_BOARD = make_charuco_board(use_legacy_pattern=False)
CHARUCO_BOARD_LEGACY = make_charuco_board(use_legacy_pattern=True)
CHARUCO_BOARD_LEGACY_SIZE = (CHARUCO_SQUARES_X - 1, CHARUCO_SQUARES_Y - 1)
ACTIVE_CHARUCO_BOARD = CHARUCO_BOARD
ACTIVE_CHARUCO_USE_LEGACY = False
legacy_pattern_notice_printed = False

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

def try_detect_charuco(gray):
    global ACTIVE_CHARUCO_BOARD, ACTIVE_CHARUCO_USE_LEGACY, legacy_pattern_notice_printed

    if hasattr(aruco, "DetectorParameters"):
        try:
            detector_params = aruco.DetectorParameters()
        except Exception:
            detector_params = aruco.DetectorParameters_create() if hasattr(aruco, "DetectorParameters_create") else None
    elif hasattr(aruco, "DetectorParameters_create"):
        detector_params = aruco.DetectorParameters_create()
    else:
        detector_params = None

    if hasattr(aruco, "ArucoDetector"):
        aruco_detector = aruco.ArucoDetector(ARUCO_DICT, detector_params) if detector_params is not None else aruco.ArucoDetector(ARUCO_DICT)
        marker_corners, marker_ids, _ = aruco_detector.detectMarkers(gray)
    else:
        marker_corners, marker_ids, _ = aruco.detectMarkers(gray, ARUCO_DICT, parameters=detector_params)

    if marker_ids is None or len(marker_ids) == 0:
        return False, None, None, marker_corners, marker_ids, ACTIVE_CHARUCO_BOARD

    candidate_boards = [(ACTIVE_CHARUCO_BOARD, ACTIVE_CHARUCO_USE_LEGACY)]
    if ACTIVE_CHARUCO_USE_LEGACY:
        candidate_boards.append((CHARUCO_BOARD, False))
    else:
        candidate_boards.append((CHARUCO_BOARD_LEGACY, True))

    for board, use_legacy_pattern in candidate_boards:
        charuco_corners = None
        charuco_ids = None

        if hasattr(aruco, "CharucoDetector"):
            try:
                charuco_detector = aruco.CharucoDetector(board)
                charuco_corners, charuco_ids, marker_corners, marker_ids = charuco_detector.detectBoard(gray)
            except Exception:
                charuco_corners = None
                charuco_ids = None

        if (charuco_ids is None or charuco_corners is None or len(charuco_ids) < 4) and hasattr(aruco, "interpolateCornersCharuco"):
            interp = aruco.interpolateCornersCharuco(marker_corners, marker_ids, gray, board)
            if interp is not None and len(interp) >= 3:
                _, charuco_corners, charuco_ids = interp[:3]

        if charuco_ids is None or charuco_corners is None or len(charuco_ids) < 4:
            continue

        ACTIVE_CHARUCO_BOARD = board
        ACTIVE_CHARUCO_USE_LEGACY = use_legacy_pattern
        if use_legacy_pattern and not legacy_pattern_notice_printed:
            print("[INFO] Charuco detection required legacy board pattern mode.")
            legacy_pattern_notice_printed = True
        return True, charuco_corners, charuco_ids, marker_corners, marker_ids, board

    return False, None, None, marker_corners, marker_ids, ACTIVE_CHARUCO_BOARD

def get_charuco_object_points(board, charuco_ids, charuco_corners):
    if hasattr(board, "getChessboardCorners"):
        all_obj_points = board.getChessboardCorners()
    elif hasattr(board, "chessboardCorners"):
        all_obj_points = board.chessboardCorners
    else:
        raise RuntimeError("Could not extract Charuco chessboard corners from board object.")

    all_obj_points = np.asarray(all_obj_points, dtype=np.float32).reshape(-1, 3)
    ids_flat = np.asarray(charuco_ids, dtype=np.int32).reshape(-1)
    img_points = np.asarray(charuco_corners, dtype=np.float32).reshape(-1, 2)

    if ids_flat.shape[0] != img_points.shape[0]:
        raise RuntimeError("Charuco ids/corners size mismatch.")

    obj_points = all_obj_points[ids_flat]
    return obj_points, img_points

def calibrate_charuco_camera(charuco_ids_list, charuco_corners_list, image_size, board):
    if hasattr(aruco, "calibrateCameraCharuco"):
        return aruco.calibrateCameraCharuco(
            charucoCorners=charuco_corners_list,
            charucoIds=charuco_ids_list,
            board=board,
            imageSize=image_size,
            cameraMatrix=None,
            distCoeffs=None,
        )

    calib_objpoints = []
    calib_imgpoints = []
    for charuco_ids, charuco_corners in zip(charuco_ids_list, charuco_corners_list):
        obj_pts, img_pts = get_charuco_object_points(board, charuco_ids, charuco_corners)
        if obj_pts.shape[0] < 4:
            continue
        calib_objpoints.append(obj_pts.astype(np.float32))
        calib_imgpoints.append(img_pts.astype(np.float32).reshape(-1, 1, 2))

    if len(calib_objpoints) < MIN_CALIB_FRAMES:
        raise RuntimeError(
            f"Only {len(calib_objpoints)} valid Charuco frames remained for calibration; need at least {MIN_CALIB_FRAMES}."
        )

    return cv2.calibrateCamera(
        calib_objpoints,
        calib_imgpoints,
        image_size,
        None,
        None,
    )

def save_calibration(K, dist, rms):
    filename = datetime.now().strftime("calibration_webcam_%Y%m%d_%H%M%S.npz")
    path = BASE_DIR / filename
    np.savez(
        str(path),
        K=K,
        dist=dist,
        rms=rms,
        inner_corners=np.array(CHARUCO_BOARD_LEGACY_SIZE),
        square_size_m=CHARUCO_SQUARE_SIZE_M,
        board_type="charuco",
        charuco_squares_xy=np.array((CHARUCO_SQUARES_X, CHARUCO_SQUARES_Y)),
        charuco_squares_x=CHARUCO_SQUARES_X,
        charuco_squares_y=CHARUCO_SQUARES_Y,
        charuco_square_size_m=CHARUCO_SQUARE_SIZE_M,
        charuco_marker_size_m=CHARUCO_MARKER_SIZE_M,
        aruco_dictionary=CHARUCO_DICT_NAME,
        charuco_legacy_pattern=bool(ACTIVE_CHARUCO_USE_LEGACY),
    )
    return str(path)

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
        found, charuco_corners, charuco_ids, marker_corners, marker_ids, detected_board = (False, None, None, None, None, ACTIVE_CHARUCO_BOARD)
        if calib_mode:
            found, charuco_corners, charuco_ids, marker_corners, marker_ids, detected_board = try_detect_charuco(gray)
            if marker_ids is not None and len(marker_ids) > 0:
                aruco.drawDetectedMarkers(display, marker_corners, marker_ids)
            if found:
                aruco.drawDetectedCornersCharuco(display, charuco_corners, charuco_ids)

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
                if found and charuco_corners is not None and charuco_ids is not None:
                    objpoints.append(charuco_ids.copy())
                    imgpoints.append(charuco_corners.copy())
                    calib_images += 1
                    print(f"Captured calibration frame {calib_images}")
                else:
                    print("Charuco board not found; move/tilt board and try again.")

        # Capture calibration frame with SPACE
        elif key == 32:
            if calib_mode:
                if found and charuco_corners is not None and charuco_ids is not None:
                    objpoints.append(charuco_ids.copy())
                    imgpoints.append(charuco_corners.copy())
                    calib_images += 1
                    print(f"Captured calibration frame {calib_images}")
                else:
                    print("Charuco board not found; move/tilt board and try again.")

        # Compute calibration
        elif key == ord('k'):
            if len(objpoints) < MIN_CALIB_FRAMES:
                print(f"Need at least {MIN_CALIB_FRAMES} good frames; currently have {len(objpoints)}.")
            else:
                img_size = (frame.shape[1], frame.shape[0])
                rms, K, dist, rvecs, tvecs = calibrate_charuco_camera(
                    charuco_ids_list=objpoints,
                    charuco_corners_list=imgpoints,
                    image_size=img_size,
                    board=ACTIVE_CHARUCO_BOARD,
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
