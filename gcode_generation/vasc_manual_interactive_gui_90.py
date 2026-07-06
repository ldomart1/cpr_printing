#!/usr/bin/env python3
"""
Interactive vasculature printer wrapper for gcode_generation/vasc_manual_90.py.

This script keeps the existing vasc_manual_90 planning pipeline, but adds two modes:

  1) --run-mode save-gcode
     Exactly writes a normal G-code file using vasc_manual_90.write_vessel_gcode.

  2) --run-mode print
     Physically sends the same generated commands to a Duet/RRF printer via
     duetwebapi. M400 is intentionally not used for motion synchronization;
     the script estimates the required wait time from the commanded move length
     and feedrate before sending the next motion command.

Interactive behavior
--------------------
Before every generated print group/path, the robot approaches the start using
that group's saved node offset from --group-displacements-file, then pauses for
manual tuning.

Node XY/Z tune keys:
    a  : X - step
    d  : X + step
    w  : Y - step
    s  : Y + step
    up : Z + step
    down: Z - step
    space: accept and save
    q or Esc: abort

The saved manual text file uses the existing vasc_manual_90 displacement format:
    group dx dy dz db dc feed enabled node_dx node_dy node_dz node_db node_dc # note

Where:
  - dx/dy/dz and db/dc remain present for file-format compatibility and default to travel-feed values.
  - node_dx/node_dy/node_dz are start-node XYZ stage offsets applied to the full group path.
  - node_db/node_dc are saved node-start rotary offsets used when starting the branch.

Typical usage is the same as vasc_manual_90.py, plus one of:
    --run-mode print --duet-web-address http://192.168.2.21
    --run-mode save-gcode

Important safety notes:
  - The script cannot preempt a single G1 already accepted by the controller.
  - Interactive mode follows the same planned branch ordering and travel/approach
    sequence as vasc_manual_90.py, with only a pause for node-start tuning before each branch prints.
"""

from __future__ import annotations

import argparse
import contextlib
import importlib
import importlib.util
import io
import math
import os
import queue
import re
import select
import sys
import tempfile
import termios
import threading
import time
import tty
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

try:
    import tkinter as tk
    from tkinter import filedialog, ttk
except Exception:
    tk = None
    ttk = None
    filedialog = None

try:
    from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
    from matplotlib.figure import Figure
except Exception:
    FigureCanvasTkAgg = None
    Figure = None

try:
    from duetwebapi import DuetWebAPI
except Exception:
    DuetWebAPI = None  # imported lazily for save-gcode mode


def import_vasc_manual_module() -> Any:
    """Import the sibling/original vasc_manual_90.py robustly.

    Works when this script is run from the repository root as
    `python gcode_generation/vasc_manual_interactive_gui_90.py`, when run from
    inside `gcode_generation`, or when copied next to `vasc_manual_90.py`.
    """
    here = Path(__file__).resolve().parent
    repo_root = here.parent

    for candidate in (repo_root, here):
        candidate_s = str(candidate)
        if candidate_s not in sys.path:
            sys.path.insert(0, candidate_s)

    errors: List[str] = []
    for module_name in ("gcode_generation.vasc_manual_90", "vasc_manual_90"):
        try:
            return importlib.import_module(module_name)
        except Exception as exc:
            errors.append(f"{module_name}: {exc!r}")

    for module_path in (
        here / "vasc_manual_90.py",
        repo_root / "gcode_generation" / "vasc_manual_90.py",
    ):
        if not module_path.exists():
            continue
        spec = importlib.util.spec_from_file_location("vasc_manual_90", module_path)
        if spec is None or spec.loader is None:
            continue
        module = importlib.util.module_from_spec(spec)
        sys.modules.setdefault("vasc_manual_90", module)
        spec.loader.exec_module(module)
        return module

    raise ModuleNotFoundError(
        "Could not import vasc_manual_90.py. Place this script next to vasc_manual_90.py "
        "or run it from the repository root. Tried: " + "; ".join(errors)
    )


DEFAULT_DUET_WEB_ADDRESS = "http://192.168.2.21"
DEFAULT_MANUAL_FILE = "vessels_manual_offsets.txt"
DEFAULT_VESSELS_FILE = "vessel_table_raw_Louis.txt"
DEFAULT_CALIBRATION_FILE = "Test_Calibration_2026-06-05_00/processed_image_data_folder/calibrated_robot_gcode_calibration.json"
DEFAULT_ROBOT_SKELETON_FILE = "Test_Calibration_2026-06-05_00/processed_image_data_folder/calibrated_robot_skeleton_parametric.json"
DEFAULT_OUT_FILE = "vessels_5.gcode"
DEFAULT_GROUPS_OUT_FILE = "vessels_5_groups.txt"
DEFAULT_GROUP_DISPLACEMENTS_FILE = "vessels_5_group_displacements.txt"
DEFAULT_NODE_TUNE_STEP_MM = 0.25
DEFAULT_TRAVEL_TUNE_STEP_MM = 0.25
DEFAULT_B_TUNE_STEP_DEG = 0.1
DEFAULT_C_TUNE_STEP_DEG = 1.0
DEFAULT_TUNE_FEED = 1000.0
DEFAULT_ROTARY_TUNE_FEED = 20000.0
DEFAULT_MAX_B_MOVE_FEED = 1000.0
DEFAULT_APPROACH_B_MOVE_FEED = 600.0
DEFAULT_START_STOP_B_MOVE_FEED = 600.0
DEFAULT_START_STOP_SLOW_SAMPLES = 4
DEFAULT_MIN_C_MOVE_FEED = 4000.0
DEFAULT_TRAVEL_SEGMENT_MM = 0.5
DEFAULT_TRAVEL_REWIND_INDICES = 4
DEFAULT_PRINT_REWIND_SAMPLES = 12
DEFAULT_INTER_COMMAND_DELAY_S = 0.005
DEFAULT_MIN_MOVE_WAIT_S = 0.02
DEFAULT_GUI_REFRESH_MS = 200
DEFAULT_GUI_PLOT_REFRESH_MS = 600
DEFAULT_TRAVEL_FEED_OVERRIDE = 3000.0
DEFAULT_PRINT_FEED_OVERRIDE = 500.0


# ------------------------- manual displacement persistence -------------------------

DEFAULT_MANUAL_FEED_FALLBACK = float(DEFAULT_TRAVEL_FEED_OVERRIDE)

@dataclass
class ManualGroupOffset:
    group_number: int
    delta_tip_xyz: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    delta_b_deg: float = 0.0
    delta_c_deg: float = 0.0
    feed: Optional[float] = None
    enabled: bool = True
    node_gantry_offset_mm: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=float))
    node_delta_b_deg: float = 0.0
    node_delta_c_deg: float = 0.0
    note: str = ""

    @classmethod
    def from_vm(cls, value: Any) -> "ManualGroupOffset":
        return cls(
            group_number=int(value.group_number),
            delta_tip_xyz=np.asarray(value.delta_tip_xyz, dtype=float).reshape(3),
            delta_b_deg=float(getattr(value, "delta_b_deg", 0.0)),
            delta_c_deg=float(getattr(value, "delta_c_deg", 0.0)),
            feed=getattr(value, "feed", None),
            enabled=bool(getattr(value, "enabled", True)),
            node_gantry_offset_mm=np.asarray(getattr(value, "node_gantry_offset_mm", np.zeros(3)), dtype=float).reshape(3),
            node_delta_b_deg=float(getattr(value, "node_delta_b_deg", 0.0)),
            node_delta_c_deg=float(getattr(value, "node_delta_c_deg", 0.0)),
            note=str(getattr(value, "note", "") or ""),
        )

    def to_vm(self, vm: Any) -> Any:
        return vm.GroupDisplacement(
            group_number=int(self.group_number),
            delta_tip_xyz=np.asarray(self.delta_tip_xyz, dtype=float).reshape(3),
            delta_b_deg=float(self.delta_b_deg),
            delta_c_deg=float(self.delta_c_deg),
            node_gantry_offset_mm=np.asarray(self.node_gantry_offset_mm, dtype=float).reshape(3),
            node_delta_b_deg=float(self.node_delta_b_deg),
            node_delta_c_deg=float(self.node_delta_c_deg),
            feed=self.feed,
            enabled=bool(self.enabled),
            note=str(self.note or ""),
        )


def default_manual_group_offset(group_number: int) -> "ManualGroupOffset":
    return ManualGroupOffset(group_number=int(group_number))


def _normalized_feed_value(feed: Optional[float], default_feed: float = DEFAULT_MANUAL_FEED_FALLBACK) -> float:
    if feed is None:
        return float(default_feed)
    try:
        value = float(feed)
    except Exception:
        return float(default_feed)
    if not math.isfinite(value):
        return float(default_feed)
    return value


def normalize_manual_offset_feeds(
    offsets: Dict[int, ManualGroupOffset],
    default_feed: float = DEFAULT_MANUAL_FEED_FALLBACK,
) -> Dict[int, ManualGroupOffset]:
    for offset in offsets.values():
        offset.feed = _normalized_feed_value(offset.feed, default_feed)
    return offsets


def ensure_manual_offsets_cover_paths(
    offsets: Dict[int, ManualGroupOffset],
    paths: Sequence[Any],
) -> Dict[int, ManualGroupOffset]:
    out: Dict[int, ManualGroupOffset] = {}
    for group_idx, _path in enumerate(paths, start=1):
        existing = offsets.get(group_idx)
        out[group_idx] = existing if existing is not None else default_manual_group_offset(group_idx)
    for group_idx, offset in offsets.items():
        if int(group_idx) not in out:
            out[int(group_idx)] = offset
    return out


def path_metadata_note(path: Optional[Any]) -> str:
    if path is None:
        return ""
    vessel_ids = ",".join(str(int(v)) for v in getattr(path, "source_vessel_ids", []))
    return f"path_id={getattr(path, 'path_id', '')} vessels={vessel_ids}".strip()


def compose_manual_note(base_note: str, path: Optional[Any]) -> str:
    base = str(base_note or "").strip()
    meta = path_metadata_note(path)
    if not meta:
        return base
    if not base:
        return meta
    if meta in base:
        return base
    return f"{base} | {meta}"


def parse_manual_offsets(path: Optional[str], vm: Any) -> Dict[int, ManualGroupOffset]:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        return {}
    return parse_manual_offsets_text(p.read_text(), vm, source_label=str(p))


def parse_manual_offsets_text(text: str, vm: Any, source_label: str = "<memory>") -> Dict[int, ManualGroupOffset]:
    out: Dict[int, ManualGroupOffset] = {}
    tmp_path: Optional[str] = None
    try:
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as tmp:
            tmp.write(text)
            tmp_path = tmp.name
        parsed = vm.parse_group_displacements_file(str(tmp_path))
        for k, v in parsed.items():
            out[int(k)] = ManualGroupOffset.from_vm(v)
        return out
    except Exception:
        out = {}
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass

    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        note = ""
        if "#" in line:
            line, note = line.split("#", 1)
            line = line.strip()
            note = note.strip()
        parts = [x for x in re.split(r"[\s,]+", line) if x]
        if len(parts) < 4:
            raise ValueError(f"Invalid manual offset row {source_label}:{line_no}: {raw}")
        group = int(parts[0])
        out[group] = ManualGroupOffset(
            group_number=group,
            delta_tip_xyz=np.array([float(parts[1]), float(parts[2]), float(parts[3])], dtype=float),
            delta_b_deg=float(parts[4]) if len(parts) >= 5 else 0.0,
            delta_c_deg=float(parts[5]) if len(parts) >= 6 else 0.0,
            feed=float(parts[6]) if len(parts) >= 7 else None,
            enabled=(str(parts[7]).lower() not in {"0", "false", "no", "off", "disabled"}) if len(parts) >= 8 else True,
            node_gantry_offset_mm=np.array([
                float(parts[8]) if len(parts) >= 9 else 0.0,
                float(parts[9]) if len(parts) >= 10 else 0.0,
                float(parts[10]) if len(parts) >= 11 else 0.0,
            ], dtype=float),
            node_delta_b_deg=float(parts[11]) if len(parts) >= 12 else 0.0,
            node_delta_c_deg=float(parts[12]) if len(parts) >= 13 else 0.0,
            note=note,
        )
    return out


def save_manual_offsets(path: str, offsets: Dict[int, ManualGroupOffset], paths: Optional[Sequence[Any]] = None) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# Manual group offsets for vasc_manual_interactive.py",
        "# Format:",
        "# group dx dy dz db dc feed enabled node_dx node_dy node_dz node_db node_dc # note",
        "# dx/dy/dz are stage XYZ offsets for travel/manual pose tuning; db/dc are rotary offsets.",
        "# node_dx/node_dy/node_dz are stage XYZ offsets applied to the full group path before printing.",
        "# node_db/node_dc are node-start rotary offsets used when starting the branch.",
    ]
    ordered_offsets = ensure_manual_offsets_cover_paths(offsets, paths or [])
    groups = list(range(1, len(paths) + 1)) if paths is not None else sorted(ordered_offsets)
    for group in groups:
        o = ordered_offsets.get(group, default_manual_group_offset(group))
        feed = f"{_normalized_feed_value(o.feed):.6f}"
        path_ref = None if paths is None or group < 1 or group > len(paths) else paths[group - 1]
        note_text = compose_manual_note(o.note, path_ref)
        note = f" # {note_text}" if note_text else ""
        lines.append(
            f"{int(group)} "
            f"{float(o.delta_tip_xyz[0]):.6f} {float(o.delta_tip_xyz[1]):.6f} {float(o.delta_tip_xyz[2]):.6f} "
            f"{float(o.delta_b_deg):.6f} {float(o.delta_c_deg):.6f} "
            f"{feed} {1 if o.enabled else 0} "
            f"{float(o.node_gantry_offset_mm[0]):.6f} {float(o.node_gantry_offset_mm[1]):.6f} {float(o.node_gantry_offset_mm[2]):.6f} "
            f"{float(o.node_delta_b_deg):.6f} {float(o.node_delta_c_deg):.6f}"
            f"{note}"
        )
    tmp = p.with_suffix(p.suffix + ".tmp")
    tmp.write_text("\n".join(lines) + "\n")
    tmp.replace(p)



# ------------------------- GUI / shared print state -------------------------

class SharedPrintState:
    """Thread-safe bridge between the print worker and the Tk GUI.

    The object also acts as a keyboard provider: writer.keyboard.get_key(timeout)
    reads keys placed in the queue by GUI buttons or Tk key bindings.
    """

    def __init__(self, manual_file: str = "") -> None:
        self.lock = threading.RLock()
        self.key_queue: "queue.Queue[str]" = queue.Queue()
        self.pause_requested = False
        self.abort_requested = False
        self.done = False
        self.error_text = ""
        self.manual_file = str(manual_file or "")
        self.plan_paths: List[Any] = []
        self.mode = "idle"
        self.status = "Idle"
        self.current_group: Optional[int] = None
        self.total_groups: Optional[int] = None
        self.current_path_id = ""
        self.current_source_vessels: List[int] = []
        self.current_manual_line = ""
        self.upcoming_manual_lines: List[str] = []
        self.next_command = ""
        self.next_reason = ""
        self.last_command = ""
        self.last_reason = ""
        self.current_stage_xyz: Optional[np.ndarray] = None
        self.current_tip_xyz: Optional[np.ndarray] = None
        self.current_b: Optional[float] = None
        self.current_c: Optional[float] = None
        self.current_point_index: int = 0
        self.current_point_total: int = 0
        self.printed_points: List[Tuple[float, float, float]] = []
        self.current_path_points: List[Tuple[float, float, float]] = []
        self.current_travel_points: List[Tuple[float, float, float]] = []
        self.manual_travel_points: List[Tuple[float, float, float]] = []
        self.last_saved_offset_text = ""
        self.last_key = ""
        self.branch_rewind_requested = False

    def post_key(self, key: str) -> None:
        self.key_queue.put(str(key))
        with self.lock:
            self.last_key = str(key)

    def get_key(self, timeout_s: float = 0.0) -> Optional[str]:
        try:
            return self.key_queue.get(timeout=max(0.0, float(timeout_s)))
        except queue.Empty:
            return None

    def request_pause(self) -> None:
        with self.lock:
            self.pause_requested = True
            if not self.status.lower().startswith("paused"):
                self.status = "Pause requested; will pause before the next command."

    def resume(self) -> None:
        with self.lock:
            self.pause_requested = False
            if self.mode == "paused":
                self.mode = "printing"
            self.status = "Resumed."

    def request_abort(self) -> None:
        with self.lock:
            self.abort_requested = True
            self.status = "Abort requested; no more commands will be sent after the current wait."

    def request_branch_rewind(self) -> bool:
        with self.lock:
            self.branch_rewind_requested = True
            self.status = "Branch rewind requested; the tool will move back to the previous branch end for travel retuning."
            return self.branch_rewind_requested

    def consume_branch_rewind_request(self) -> bool:
        with self.lock:
            requested = bool(self.branch_rewind_requested)
            self.branch_rewind_requested = False
            return requested

    def set_mode(self, mode: str, status: Optional[str] = None) -> None:
        with self.lock:
            self.mode = str(mode)
            if status is not None:
                self.status = str(status)

    def set_done(self, status: str = "Complete") -> None:
        with self.lock:
            self.done = True
            self.mode = "complete"
            self.status = str(status)

    def set_error(self, text: str) -> None:
        with self.lock:
            self.done = True
            self.mode = "error"
            self.error_text = str(text)
            self.status = str(text)

    def set_group(self, group: int, total: int, path_id: str, source_vessels: Sequence[int], path_points: np.ndarray) -> None:
        pts = np.asarray(path_points, dtype=float)
        with self.lock:
            self.current_group = int(group)
            self.total_groups = int(total)
            self.current_path_id = str(path_id)
            self.current_source_vessels = [int(x) for x in source_vessels]
            self.current_path_points = [tuple(map(float, p)) for p in pts[:, :3]] if pts.ndim == 2 and pts.shape[1] >= 3 else []
            self.current_point_index = 0
            self.current_point_total = int(len(pts))
            self.current_travel_points = []
            self.manual_travel_points = []
            self.status = f"Group {group}/{total}: {path_id}"

    def set_travel_preview_points(self, points: Sequence[Sequence[float]]) -> None:
        with self.lock:
            self.current_travel_points = [tuple(map(float, p[:3])) for p in points]

    def clear_travel_preview_points(self) -> None:
        with self.lock:
            self.current_travel_points = []

    def begin_manual_travel_trace(self, start_point: Optional[np.ndarray]) -> None:
        with self.lock:
            self.manual_travel_points = []
            if start_point is not None:
                p = np.asarray(start_point, dtype=float).reshape(3)
                self.manual_travel_points.append((float(p[0]), float(p[1]), float(p[2])))

    def append_manual_travel_point(self, point: Optional[np.ndarray]) -> None:
        if point is None:
            return
        p = np.asarray(point, dtype=float).reshape(3)
        with self.lock:
            self.manual_travel_points.append((float(p[0]), float(p[1]), float(p[2])))
            self.manual_travel_points = self.manual_travel_points[-2000:]

    def set_manual_window(self, current_line: str, upcoming_lines: Sequence[str]) -> None:
        with self.lock:
            self.current_manual_line = str(current_line or "")
            self.upcoming_manual_lines = [str(x) for x in upcoming_lines]

    def set_next_command(self, code: str, reason: str) -> None:
        with self.lock:
            self.next_command = str(code)
            self.next_reason = str(reason)

    def set_last_command(self, code: str, reason: str) -> None:
        with self.lock:
            self.last_command = str(code)
            self.last_reason = str(reason)
            if self.next_command == code:
                self.next_command = ""
                self.next_reason = ""

    def update_pose(self, stage_xyz: Optional[np.ndarray], b: Optional[float], c: Optional[float], tip_xyz: Optional[np.ndarray] = None) -> None:
        with self.lock:
            self.current_stage_xyz = None if stage_xyz is None else np.asarray(stage_xyz, dtype=float).copy()
            self.current_tip_xyz = None if tip_xyz is None else np.asarray(tip_xyz, dtype=float).copy()
            self.current_b = None if b is None else float(b)
            self.current_c = None if c is None else float(c)

    def record_print_point(self, tip_xyz: Optional[np.ndarray], index: Optional[int] = None, total: Optional[int] = None) -> None:
        if tip_xyz is None:
            return
        p = np.asarray(tip_xyz, dtype=float).reshape(3)
        with self.lock:
            self.printed_points.append((float(p[0]), float(p[1]), float(p[2])))
            self.printed_points = self.printed_points[-20000:]
            if index is not None:
                self.current_point_index = int(index)
            else:
                self.current_point_index += 1
            if total is not None:
                self.current_point_total = int(total)

    def wait_if_paused(self) -> None:
        while True:
            with self.lock:
                if self.abort_requested:
                    raise KeyboardInterrupt("Abort requested from GUI.")
                paused = self.pause_requested
                if paused:
                    self.mode = "paused"
                    self.status = "Paused. Press Resume to continue sending commands."
            if not paused:
                return
            time.sleep(0.05)

    def sleep_checked(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, float(seconds))
        while time.monotonic() < deadline:
            with self.lock:
                if self.abort_requested:
                    raise KeyboardInterrupt("Abort requested from GUI.")
            time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))

    def snapshot(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "pause_requested": self.pause_requested,
                "abort_requested": self.abort_requested,
                "done": self.done,
                "error_text": self.error_text,
                "manual_file": self.manual_file,
                "mode": self.mode,
                "status": self.status,
                "current_group": self.current_group,
                "total_groups": self.total_groups,
                "current_path_id": self.current_path_id,
                "current_source_vessels": list(self.current_source_vessels),
                "current_manual_line": self.current_manual_line,
                "upcoming_manual_lines": list(self.upcoming_manual_lines),
                "next_command": self.next_command,
                "next_reason": self.next_reason,
                "last_command": self.last_command,
                "last_reason": self.last_reason,
                "current_stage_xyz": None if self.current_stage_xyz is None else self.current_stage_xyz.copy(),
                "current_tip_xyz": None if self.current_tip_xyz is None else self.current_tip_xyz.copy(),
                "current_b": self.current_b,
                "current_c": self.current_c,
                "current_point_index": self.current_point_index,
                "current_point_total": self.current_point_total,
                "printed_points": list(self.printed_points),
                "current_path_points": list(self.current_path_points),
                "current_travel_points": list(self.current_travel_points),
                "manual_travel_points": list(self.manual_travel_points),
                "last_saved_offset_text": self.last_saved_offset_text,
                "last_key": self.last_key,
                "branch_rewind_requested": self.branch_rewind_requested,
            }


def _manual_offset_line_for_group(offsets: Dict[int, ManualGroupOffset], group: int, path: Optional[Any] = None) -> str:
    o = offsets.get(group, default_manual_group_offset(group))
    feed = f"{_normalized_feed_value(o.feed):.6f}"
    note_text = compose_manual_note(o.note, path)
    note = f" # {note_text}" if note_text else ""
    return (
        f"{int(group)} "
        f"{float(o.delta_tip_xyz[0]):.6f} {float(o.delta_tip_xyz[1]):.6f} {float(o.delta_tip_xyz[2]):.6f} "
        f"{float(o.delta_b_deg):.6f} {float(o.delta_c_deg):.6f} "
        f"{feed} {1 if o.enabled else 0} "
        f"{float(o.node_gantry_offset_mm[0]):.6f} {float(o.node_gantry_offset_mm[1]):.6f} {float(o.node_gantry_offset_mm[2]):.6f} "
        f"{float(o.node_delta_b_deg):.6f} {float(o.node_delta_c_deg):.6f}"
        f"{note}"
    )


def update_manual_window_state(
    state: Optional[SharedPrintState],
    offsets: Dict[int, ManualGroupOffset],
    current_group: int,
    paths: Optional[Sequence[Any]] = None,
    lookahead: int = 5,
) -> None:
    if state is None:
        return
    current_path = None if paths is None or not (1 <= int(current_group) <= len(paths)) else paths[int(current_group) - 1]
    current = _manual_offset_line_for_group(offsets, int(current_group), current_path)
    upcoming: List[str] = []
    for i in range(1, int(lookahead) + 1):
        group = int(current_group) + i
        path = None if paths is None or not (1 <= group <= len(paths)) else paths[group - 1]
        line = _manual_offset_line_for_group(offsets, group, path)
        upcoming.append(line)
    state.set_manual_window(current, upcoming)


def current_tip_matches_start(writer: Any, start_tip: np.ndarray, tol_mm: float = 0.05) -> bool:
    if getattr(writer.base, "cur_tip_xyz", None) is None:
        return False
    cur_tip = np.asarray(writer.base.cur_tip_xyz, dtype=float).reshape(3)
    start_tip = np.asarray(start_tip, dtype=float).reshape(3)
    return float(np.linalg.norm(cur_tip - start_tip)) <= float(tol_mm)


class GuiKeyboardProxy:
    def __init__(self, state: SharedPrintState) -> None:
        self.state = state

    def get_key(self, timeout_s: float = 0.0) -> Optional[str]:
        return self.state.get_key(timeout_s)


class HybridKeyboard:
    """Read GUI-posted keys first, then fall back to terminal keys."""

    def __init__(self, state: Optional[SharedPrintState], terminal_keyboard: Optional["TerminalKeyboard"]) -> None:
        self.state = state
        self.terminal_keyboard = terminal_keyboard

    def get_key(self, timeout_s: float = 0.0) -> Optional[str]:
        if self.state is not None:
            key = self.state.get_key(0.0)
            if key is not None:
                return key
        if self.terminal_keyboard is None:
            if self.state is not None:
                return self.state.get_key(timeout_s)
            return None
        key = self.terminal_keyboard.get_key(timeout_s)
        if key is not None:
            return key
        return None


class PrintGUIApp:
    def __init__(self, vm: Any, args: argparse.Namespace, state: SharedPrintState, plan: Dict[str, Any], manual_file: str, refresh_ms: int = DEFAULT_GUI_REFRESH_MS, plot_refresh_ms: int = DEFAULT_GUI_PLOT_REFRESH_MS) -> None:
        if tk is None or ttk is None:
            raise RuntimeError("Tkinter is not available in this Python environment.")
        if Figure is None or FigureCanvasTkAgg is None:
            raise RuntimeError("Matplotlib TkAgg is not available; install matplotlib or use --no-gui.")
        self.vm = vm
        self.args = args
        self.state = state
        self.plan = plan
        self.manual_file = str(manual_file)
        self.refresh_ms = int(refresh_ms)
        self.plot_refresh_ms = int(plot_refresh_ms)
        self.root = tk.Tk()
        self.root.title("Interactive Vasculature Print")
        self.root.geometry(f"1560x{min(900, max(760, int(self.root.winfo_screenheight() * 0.9)))}")
        self._last_plot_update = 0.0
        self._last_printed_count = -1
        self._build_widgets()
        self._bind_keys()
        self._draw_static_paths()
        self.root.after(self.refresh_ms, self._refresh)

    def _build_widgets(self) -> None:
        self.root.columnconfigure(0, weight=0)
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        left_outer = ttk.Frame(self.root, padding=8)
        left_outer.grid(row=0, column=0, sticky="nsew")
        left_outer.rowconfigure(0, weight=1)
        left_outer.columnconfigure(0, weight=1)
        left_canvas = tk.Canvas(left_outer, highlightthickness=0, width=620)
        left_scroll = ttk.Scrollbar(left_outer, orient="vertical", command=left_canvas.yview)
        left_canvas.configure(yscrollcommand=left_scroll.set)
        left_canvas.grid(row=0, column=0, sticky="nsew")
        left_scroll.grid(row=0, column=1, sticky="ns")
        left = ttk.Frame(left_canvas, padding=0)
        left_canvas_window = left_canvas.create_window((0, 0), window=left, anchor="nw")

        def sync_left_scroll_region(_event: Any) -> None:
            left_canvas.configure(scrollregion=left_canvas.bbox("all"))

        def sync_left_canvas_width(event: Any) -> None:
            left_canvas.itemconfigure(left_canvas_window, width=event.width)

        left.bind("<Configure>", sync_left_scroll_region)
        left_canvas.bind("<Configure>", sync_left_canvas_width)
        right = ttk.Frame(self.root, padding=8)
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(0, weight=1)
        right.columnconfigure(0, weight=1)

        self.status_var = tk.StringVar(value="Idle")
        self.group_var = tk.StringVar(value="Group: -")
        self.path_var = tk.StringVar(value="Path: -")
        self.pose_var = tk.StringVar(value="Pose: -")
        self.point_var = tk.StringVar(value="Point: -")
        self.last_key_var = tk.StringVar(value="Last key: -")
        self.manual_file_var = tk.StringVar(value=f"Manual file: {self.manual_file}")
        self.manual_edit_status_var = tk.StringVar(value="Manual file editor: ready")
        self.export_status_var = tk.StringVar(value="Teach mode: Accept saves the taught node; export uses the current taught state.")

        row = 0
        ttk.Label(left, text="Print status", font=("TkDefaultFont", 13, "bold")).grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 4)); row += 1
        ttk.Label(left, textvariable=self.status_var, wraplength=560).grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 8)); row += 1
        ttk.Label(left, textvariable=self.group_var).grid(row=row, column=0, columnspan=4, sticky="w"); row += 1
        ttk.Label(left, textvariable=self.path_var, wraplength=560).grid(row=row, column=0, columnspan=4, sticky="w"); row += 1
        ttk.Label(left, textvariable=self.point_var).grid(row=row, column=0, columnspan=4, sticky="w"); row += 1
        ttk.Label(left, textvariable=self.pose_var, wraplength=560).grid(row=row, column=0, columnspan=4, sticky="w"); row += 1
        ttk.Label(left, textvariable=self.last_key_var).grid(row=row, column=0, columnspan=4, sticky="w", pady=(0, 8)); row += 1

        ttk.Separator(left, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=6); row += 1
        ttk.Button(left, text="Pause", command=self.state.request_pause).grid(row=row, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(left, text="Resume", command=self.state.resume).grid(row=row, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(left, text="Accept / Space", command=lambda: self.state.post_key("SPACE")).grid(row=row, column=2, sticky="ew", padx=2, pady=2)
        ttk.Button(left, text="Abort", command=self.state.request_abort).grid(row=row, column=3, sticky="ew", padx=2, pady=2); row += 1
        ttk.Label(left, text="Jog buttons mirror the keyboard controls", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, columnspan=4, sticky="w"); row += 1
        for col in range(4):
            left.columnconfigure(col, weight=1)
        ttk.Button(left, text="X-  a", command=lambda: self.state.post_key("a")).grid(row=row, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(left, text="X+  d", command=lambda: self.state.post_key("d")).grid(row=row, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(left, text="Y-  w", command=lambda: self.state.post_key("w")).grid(row=row, column=2, sticky="ew", padx=2, pady=2)
        ttk.Button(left, text="Y+  s", command=lambda: self.state.post_key("s")).grid(row=row, column=3, sticky="ew", padx=2, pady=2); row += 1
        ttk.Button(left, text="Z+  ↑", command=lambda: self.state.post_key("UP")).grid(row=row, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(left, text="Z-  ↓", command=lambda: self.state.post_key("DOWN")).grid(row=row, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(left, text="B+  i", command=lambda: self.state.post_key("i")).grid(row=row, column=2, sticky="ew", padx=2, pady=2)
        ttk.Button(left, text="B-  k", command=lambda: self.state.post_key("k")).grid(row=row, column=3, sticky="ew", padx=2, pady=2); row += 1
        ttk.Button(left, text="C-  j", command=lambda: self.state.post_key("j")).grid(row=row, column=0, sticky="ew", padx=2, pady=2)
        ttk.Button(left, text="C+  l", command=lambda: self.state.post_key("l")).grid(row=row, column=1, sticky="ew", padx=2, pady=2)
        ttk.Button(left, text="q / Esc", command=lambda: self.state.post_key("ESC")).grid(row=row, column=2, columnspan=2, sticky="ew", padx=2, pady=2); row += 1

        ttk.Separator(left, orient="horizontal").grid(row=row, column=0, columnspan=4, sticky="ew", pady=6); row += 1
        ttk.Label(left, text="Next command", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, columnspan=4, sticky="w"); row += 1
        self.next_command_text = tk.Text(left, width=74, height=4, wrap="word")
        self.next_command_text.grid(row=row, column=0, columnspan=4, sticky="ew"); row += 1
        ttk.Label(left, text="Last command", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, columnspan=4, sticky="w", pady=(8, 0)); row += 1
        self.last_command_text = tk.Text(left, width=74, height=3, wrap="word")
        self.last_command_text.grid(row=row, column=0, columnspan=4, sticky="ew"); row += 1

        ttk.Label(left, textvariable=self.manual_file_var, wraplength=560).grid(row=row, column=0, columnspan=4, sticky="w", pady=(8, 0)); row += 1
        ttk.Label(left, text="Teach / Export", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, columnspan=4, sticky="w"); row += 1
        ttk.Button(left, text="Save Taught Offsets", command=self._save_taught_offsets).grid(row=row, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        ttk.Button(left, text="Export Taught G-code", command=self._export_taught_gcode).grid(row=row, column=2, columnspan=2, sticky="ew", padx=2, pady=2); row += 1
        ttk.Button(left, text="Export Top-Down G-code", command=self._export_top_down_gcode).grid(row=row, column=0, columnspan=4, sticky="ew", padx=2, pady=2); row += 1
        ttk.Label(left, textvariable=self.export_status_var, wraplength=560).grid(row=row, column=0, columnspan=4, sticky="w"); row += 1
        ttk.Label(left, text="Current and upcoming group-displacement lines", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, columnspan=4, sticky="w"); row += 1
        self.manual_text = tk.Text(left, width=74, height=6, wrap="none")
        self.manual_text.grid(row=row, column=0, columnspan=4, sticky="nsew")
        row += 1
        ttk.Label(left, text="Editable travel/node offset file", font=("TkDefaultFont", 11, "bold")).grid(row=row, column=0, columnspan=4, sticky="w", pady=(8, 0)); row += 1
        self.manual_edit_text = tk.Text(left, width=74, height=10, wrap="none")
        self.manual_edit_text.grid(row=row, column=0, columnspan=4, sticky="nsew"); row += 1
        ttk.Button(left, text="Reload manual file", command=self._reload_manual_editor_from_disk).grid(row=row, column=0, columnspan=2, sticky="ew", padx=2, pady=2)
        ttk.Button(left, text="Save manual file", command=self._save_manual_editor).grid(row=row, column=2, columnspan=2, sticky="ew", padx=2, pady=2); row += 1
        ttk.Label(left, textvariable=self.manual_edit_status_var, wraplength=560).grid(row=row, column=0, columnspan=4, sticky="w"); row += 1
        left.rowconfigure(row - 2, weight=1)
        left.rowconfigure(row - 5, weight=1)
        self._reload_manual_editor_from_disk()

        self.fig = Figure(figsize=(8, 7), dpi=100)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self._configure_dark_plot()
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.printed_line = None
        self.current_point_artist = None
        self.current_path_line = None
        self.current_travel_line = None
        self.manual_travel_line = None
        self.plot_mode_text = self.ax.text2D(
            0.03, 0.97, "", transform=self.ax.transAxes,
            color="#facc15", fontsize=12, fontweight="bold",
            va="top", ha="left",
            bbox=dict(boxstyle="round,pad=0.25", facecolor="#111827", edgecolor="#facc15", alpha=0.9),
        )

    def _configure_dark_plot(self) -> None:
        bg = "#070b11"
        fg = "#d8dee9"
        grid = "#334155"
        self.fig.patch.set_facecolor(bg)
        self.ax.set_facecolor(bg)
        self.ax.set_xlabel("X mm", color=fg)
        self.ax.set_ylabel("Y mm", color=fg)
        self.ax.set_zlabel("Z mm", color=fg)
        self.ax.set_title("3D print progress", color=fg)
        self.ax.tick_params(colors=fg)
        for axis in (self.ax.xaxis, self.ax.yaxis, self.ax.zaxis):
            axis.set_pane_color((0.05, 0.07, 0.10, 1.0))
            try:
                axis.line.set_color(fg)
            except Exception:
                pass
        try:
            self.ax.xaxis._axinfo["grid"]["color"] = grid
            self.ax.yaxis._axinfo["grid"]["color"] = grid
            self.ax.zaxis._axinfo["grid"]["color"] = grid
        except Exception:
            pass

    def _bind_keys(self) -> None:
        def handler(event: Any) -> str:
            key = event.keysym
            char = event.char
            mapped = None
            if key == "space": mapped = "SPACE"
            elif key == "Up": mapped = "UP"
            elif key == "Down": mapped = "DOWN"
            elif key == "Right": mapped = "FORWARD"
            elif key == "Escape": mapped = "ESC"
            elif char in {"a", "d", "w", "s", "i", "k", "j", "l", "q"}:
                mapped = char
            if mapped is not None:
                self.state.post_key(mapped)
            return "break"
        self.root.bind_all("<Key>", handler)

    def _draw_static_paths(self) -> None:
        all_pts = []
        for path in self.plan.get("paths", []):
            pts = np.asarray(path.points, dtype=float)
            if pts.ndim != 2 or pts.shape[0] < 2 or pts.shape[1] < 3:
                continue
            self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#4b5563", linewidth=0.8, alpha=0.85)
            all_pts.append(pts[:, :3])
        if all_pts:
            pts_all = np.vstack(all_pts)
            self._set_equalish_axes(pts_all)
        self.canvas.draw_idle()

    def _set_equalish_axes(self, pts: np.ndarray) -> None:
        mins = np.min(pts, axis=0)
        maxs = np.max(pts, axis=0)
        spans = np.maximum(maxs - mins, 1.0)
        center = 0.5 * (mins + maxs)
        radius = 0.55 * float(np.max(spans))
        self.ax.set_xlim(center[0] - radius, center[0] + radius)
        self.ax.set_ylim(center[1] - radius, center[1] + radius)
        self.ax.set_zlim(center[2] - radius, center[2] + radius)

    def _set_text(self, widget: Any, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    def _reload_manual_editor_from_disk(self) -> None:
        try:
            text = Path(self.manual_file).read_text() if Path(self.manual_file).exists() else ""
        except Exception as exc:
            self.manual_edit_status_var.set(f"Reload failed: {exc}")
            return
        self.manual_edit_text.delete("1.0", "end")
        self.manual_edit_text.insert("1.0", text)
        self.manual_edit_status_var.set("Manual file editor reloaded from disk.")

    def _save_manual_editor(self) -> None:
        text = self.manual_edit_text.get("1.0", "end")
        try:
            parsed = parse_manual_offsets_text(text, self.vm, source_label=self.manual_file)
            normalize_manual_offset_feeds(parsed, default_feed=float(self.plan.get("travel_feed", DEFAULT_MANUAL_FEED_FALLBACK)))
            Path(self.manual_file).write_text(text.rstrip() + "\n")
            manual_offsets = self.plan["manual_offsets"]
            manual_offsets.clear()
            manual_offsets.update(parsed)
            snap = self.state.snapshot()
            current_group = int(snap["current_group"] or 1)
            update_manual_window_state(self.state, manual_offsets, current_group, self.plan.get("paths"))
            self.manual_edit_status_var.set(f"Saved manual file with {len(parsed)} group rows.")
        except Exception as exc:
            self.manual_edit_status_var.set(f"Save failed: {exc}")

    def _save_taught_offsets(self) -> None:
        try:
            save_manual_offsets(self.manual_file, self.plan["manual_offsets"], self.plan.get("paths"))
            self._reload_manual_editor_from_disk()
            self.export_status_var.set(f"Saved taught offsets to {self.manual_file}")
        except Exception as exc:
            self.export_status_var.set(f"Save taught offsets failed: {exc}")

    def _default_export_path(self, suffix_tag: str = "") -> str:
        out_path = Path(str(getattr(self.args, "out", DEFAULT_OUT_FILE)))
        suffix = out_path.suffix if out_path.suffix else ".gcode"
        stem = out_path.stem if out_path.stem else out_path.name
        if suffix_tag:
            stem = f"{stem}_{suffix_tag}"
        return str(out_path.with_name(stem + suffix))

    def _choose_export_path(self, title: str, suffix_tag: str = "") -> Optional[str]:
        initial_path = self._default_export_path(suffix_tag)
        if filedialog is None:
            return initial_path
        return filedialog.asksaveasfilename(
            parent=self.root,
            title=title,
            initialfile=Path(initial_path).name,
            initialdir=str(Path(initial_path).parent),
            defaultextension=".gcode",
            filetypes=[("G-code", "*.gcode"), ("Text files", "*.txt"), ("All files", "*.*")],
        ) or None

    def _export_gcode_from_taught_state(self, orientation_mode: Optional[str], suffix_tag: str, title: str) -> None:
        out_path = self._choose_export_path(title=title, suffix_tag=suffix_tag)
        if not out_path:
            self.export_status_var.set("Export cancelled.")
            return
        export_args = argparse.Namespace(**vars(self.args))
        export_args.run_mode = "save-gcode"
        export_args.out = str(out_path)
        if orientation_mode is not None:
            export_args.orientation_mode = str(orientation_mode)
        try:
            save_gcode(self.vm, export_args, self.plan)
            self.export_status_var.set(f"Exported taught G-code to {out_path}")
        except Exception as exc:
            self.export_status_var.set(f"Export failed: {exc}")

    def _export_taught_gcode(self) -> None:
        self._export_gcode_from_taught_state(
            orientation_mode=None,
            suffix_tag="taught",
            title="Export taught G-code",
        )

    def _export_top_down_gcode(self) -> None:
        self._export_gcode_from_taught_state(
            orientation_mode="fixed",
            suffix_tag="topdown",
            title="Export top-down taught G-code",
        )

    def _refresh(self) -> None:
        snap = self.state.snapshot()
        group = snap["current_group"]
        total = snap["total_groups"]
        vessels = ",".join(str(x) for x in snap["current_source_vessels"])
        self.status_var.set(f"{snap['mode'].upper()} - {snap['status']}")
        self.group_var.set(f"Branch/group: {group if group is not None else '-'} / {total if total is not None else '-'}")
        self.path_var.set(f"Path: {snap['current_path_id'] or '-'}    source vessels: {vessels or '-'}")
        self.point_var.set(f"Point progress in group: {snap['current_point_index']} / {snap['current_point_total']}")
        self.last_key_var.set(f"Last key/button: {snap['last_key'] or '-'}")
        stage = snap["current_stage_xyz"]
        tip = snap["current_tip_xyz"]
        b = snap["current_b"]
        c = snap["current_c"]
        stage_txt = "stage XYZ=-" if stage is None else f"stage XYZ=({stage[0]:.3f}, {stage[1]:.3f}, {stage[2]:.3f})"
        tip_txt = "tip XYZ=-" if tip is None else f"tip XYZ=({tip[0]:.3f}, {tip[1]:.3f}, {tip[2]:.3f})"
        bc_txt = f"B={b:.3f} C={c:.3f}" if b is not None and c is not None else "B/C=-"
        self.pose_var.set(f"{stage_txt}    {tip_txt}    {bc_txt}")
        next_text = f"Reason: {snap['next_reason']}\n{snap['next_command']}" if snap["next_command"] else "-"
        last_text = f"Reason: {snap['last_reason']}\n{snap['last_command']}" if snap["last_command"] else "-"
        self._set_text(self.next_command_text, next_text)
        self._set_text(self.last_command_text, last_text)
        manual_lines = []
        if snap["current_manual_line"]:
            manual_lines.append("CURRENT: " + snap["current_manual_line"])
        for i, line in enumerate(snap["upcoming_manual_lines"], start=1):
            manual_lines.append(f"+{i}:      " + line)
        self._set_text(self.manual_text, "\n".join(manual_lines) if manual_lines else "-")
        now = time.monotonic()
        if now - self._last_plot_update >= self.plot_refresh_ms / 1000.0:
            self._refresh_plot(snap)
            self._last_plot_update = now
        if snap["done"]:
            # Keep window open, but slow refresh down.
            self.root.after(1000, self._refresh)
        else:
            self.root.after(self.refresh_ms, self._refresh)

    def _refresh_plot(self, snap: Dict[str, Any]) -> None:
        printed = snap["printed_points"]
        current_path = snap["current_path_points"]
        current_travel = snap["current_travel_points"]
        manual_travel = snap["manual_travel_points"]
        mode = str(snap.get("mode") or "")
        if mode == "node_tune":
            self.plot_mode_text.set_text("NODE TUNING MODE")
        else:
            self.plot_mode_text.set_text("")
        if self.current_path_line is not None:
            self.current_path_line.remove()
            self.current_path_line = None
        if current_path:
            pts = np.asarray(current_path, dtype=float)
            self.current_path_line, = self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#f59e0b", linewidth=1.5)
        if self.current_travel_line is not None:
            self.current_travel_line.remove()
            self.current_travel_line = None
        if current_travel:
            pts = np.asarray(current_travel, dtype=float)
            self.current_travel_line, = self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#94a3b8", linewidth=2.0, alpha=0.30)
        if self.manual_travel_line is not None:
            self.manual_travel_line.remove()
            self.manual_travel_line = None
        if manual_travel:
            pts = np.asarray(manual_travel, dtype=float)
            self.manual_travel_line, = self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#1d4ed8", linewidth=2.4)
        if self.printed_line is not None:
            self.printed_line.remove()
            self.printed_line = None
        if printed:
            pts = np.asarray(printed, dtype=float)
            self.printed_line, = self.ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#38bdf8", linewidth=2.2)
        if self.current_point_artist is not None:
            try:
                self.current_point_artist.remove()
            except Exception:
                pass
            self.current_point_artist = None
        tip = snap["current_tip_xyz"]
        if tip is not None:
            p = np.asarray(tip, dtype=float).reshape(3)
            self.current_point_artist = self.ax.scatter([p[0]], [p[1]], [p[2]], color="#f43f5e", s=32)
        self.canvas.draw_idle()

    def run(self) -> None:
        self.root.mainloop()

# ------------------------- terminal keyboard helper -------------------------

class TerminalKeyboard:
    def __init__(self) -> None:
        self._fd: Optional[int] = None
        self._old_attrs: Optional[list] = None

    def __enter__(self) -> "TerminalKeyboard":
        if not sys.stdin.isatty():
            raise RuntimeError("Interactive keyboard control requires a TTY/stdin terminal.")
        self._fd = sys.stdin.fileno()
        self._old_attrs = termios.tcgetattr(self._fd)
        tty.setcbreak(self._fd)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if self._fd is not None and self._old_attrs is not None:
            termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old_attrs)

    def get_key(self, timeout_s: float = 0.0) -> Optional[str]:
        if self._fd is None:
            return None
        r, _, _ = select.select([sys.stdin], [], [], max(0.0, float(timeout_s)))
        if not r:
            return None
        ch = sys.stdin.read(1)
        if ch == "\x1b":
            # Arrow keys arrive as ESC [ A/B/C/D.
            r2, _, _ = select.select([sys.stdin], [], [], 0.001)
            if r2:
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    return {"A": "UP", "B": "DOWN", "C": "RIGHT", "D": "LEFT"}.get(ch3, "ESC")
            return "ESC"
        if ch == " ":
            return "SPACE"
        return ch


# ------------------------- direct robot writer -------------------------

class InteractiveRobotWriter:
    """Subclass-like wrapper around vm.GCodeWriter that sends commands instead of writing a file."""

    def __init__(self, vm: Any, rrf: Any, keyboard: Any, manual_file: str,
                 manual_offsets: Dict[int, ManualGroupOffset], node_tune_step_mm: float,
                 travel_tune_step_mm: float, b_tune_step_deg: float, c_tune_step_deg: float,
                 tune_feed: float, rotary_tune_feed: float, travel_segment_mm: float,
                 travel_rewind_indices: int, inter_command_delay_s: float,
                 min_move_wait_s: float, ui_state: Optional[SharedPrintState] = None, *args: Any, **kwargs: Any) -> None:
        self.vm = vm
        self.rrf = rrf
        self.keyboard = keyboard
        self.ui_state = ui_state
        self.manual_file = str(manual_file)
        self.manual_offsets = manual_offsets
        self.node_tune_step_mm = float(node_tune_step_mm)
        self.travel_tune_step_mm = float(travel_tune_step_mm)
        self.b_tune_step_deg = float(b_tune_step_deg)
        self.c_tune_step_deg = float(c_tune_step_deg)
        self.tune_feed = float(tune_feed)
        self.rotary_tune_feed = float(rotary_tune_feed)
        self.travel_segment_mm = max(0.05, float(travel_segment_mm))
        self.travel_rewind_indices = max(0, int(travel_rewind_indices))
        self.inter_command_delay_s = max(0.0, float(inter_command_delay_s))
        self.min_move_wait_s = max(0.0, float(min_move_wait_s))
        self.max_b_move_feed = float(DEFAULT_MAX_B_MOVE_FEED)
        self.approach_b_move_feed = max(0.0, float(kwargs.pop("approach_b_move_feed", DEFAULT_APPROACH_B_MOVE_FEED)))
        self.start_stop_b_move_feed = max(0.0, float(kwargs.pop("start_stop_b_move_feed", DEFAULT_START_STOP_B_MOVE_FEED)))
        self.start_stop_slow_samples = max(0, int(kwargs.pop("start_stop_slow_samples", DEFAULT_START_STOP_SLOW_SAMPLES)))
        self.min_c_move_feed = max(DEFAULT_MIN_C_MOVE_FEED, float(kwargs.get("c_feed", DEFAULT_MIN_C_MOVE_FEED)))
        self.estimated_motion_done_at = time.monotonic()
        self.current_group_number: Optional[int] = None
        self.current_path_id: Optional[str] = None
        self.in_print_polyline = False
        self.in_tip_tracking_interpolation = False
        self._travel_history: List[Tuple[np.ndarray, float, float, Optional[np.ndarray]]] = []
        self.previous_branch_end_pose: Optional[Tuple[np.ndarray, float, float, Optional[np.ndarray]]] = None
        self._accepted_node_start_group: Optional[int] = None

        # Construct the original writer against a StringIO sink. We reuse all
        # its planning, collision, tangent, pressure, and state logic, then
        # override command emission methods by assigning bound methods below.
        self.base = vm.GCodeWriter(io.StringIO(), *args, **kwargs)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.base, name)

    # ----- command/wait primitives -----

    def _axis_order(self) -> List[str]:
        cal = self.base.cal
        axes = [cal.x_axis, cal.y_axis, cal.z_axis]
        if self.base.write_mode == "calibrated":
            axes.extend([cal.b_axis, cal.c_axis])
        return axes

    def _current_axes(self) -> Dict[str, float]:
        cal = self.base.cal
        out: Dict[str, float] = {}
        if self.base.cur_stage_xyz is not None:
            out[cal.x_axis] = float(self.base.cur_stage_xyz[0])
            out[cal.y_axis] = float(self.base.cur_stage_xyz[1])
            out[cal.z_axis] = float(self.base.cur_stage_xyz[2])
        if self.base.write_mode == "calibrated":
            out[cal.b_axis] = float(self.base.cur_b)
            out[cal.c_axis] = float(self.base.cur_c)
        return out

    def _estimate_wait_s(self, previous_axes: Dict[str, float], target_axes: Dict[str, float], feed: float) -> float:
        deltas_sq = 0.0
        for ax, target in target_axes.items():
            if ax not in previous_axes:
                continue
            d = float(target) - float(previous_axes[ax])
            deltas_sq += d * d
        dist = math.sqrt(deltas_sq)
        if dist <= 1e-12:
            return 0.0
        return max(self.min_move_wait_s, dist / max(1e-9, float(feed) / 60.0))

    def _cap_feed_for_b_axis_component(
        self,
        requested_feed: float,
        stage_dist: float,
        b_dist: float,
        c_dist: float,
        max_b_axis_feed: Optional[float],
    ) -> float:
        feed = max(1.0, float(requested_feed))
        if max_b_axis_feed is None:
            return feed
        max_b = float(max_b_axis_feed)
        if max_b <= 0.0 or float(b_dist) <= 1e-12:
            return feed
        axis_dist = float(np.linalg.norm(np.array([float(stage_dist), float(b_dist), float(c_dist)], dtype=float)))
        if axis_dist <= 1e-12:
            return feed
        return min(feed, float(max_b) * axis_dist / float(b_dist))

    def _is_approach_reason(self, reason: Optional[str]) -> bool:
        text = str(reason or "").strip().lower()
        if not text:
            return False
        markers = (
            "approach",
            "pre-spin",
            "travel to node start",
            "fine-approach",
            "fine approach",
            "node start resync",
        )
        return any(marker in text for marker in markers)

    def _b_feed_limit_for_reason(self, reason: Optional[str]) -> float:
        if self._is_approach_reason(reason):
            return float(self.approach_b_move_feed)
        return float(self.max_b_move_feed)

    def _check_for_immediate_branch_rewind(self) -> None:
        return

    def _sleep_interruptible(self, seconds: float) -> None:
        deadline = time.monotonic() + max(0.0, float(seconds))
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                break
            if self.ui_state is not None:
                self.ui_state.wait_if_paused()
                self._check_for_immediate_branch_rewind()
            time.sleep(min(0.05, remaining))
        if self.ui_state is not None:
            self._check_for_immediate_branch_rewind()

    def _wait_estimated_complete(self, extra_s: float = 0.0, reason: str = "motion") -> None:
        wait_s = max(0.0, self.estimated_motion_done_at - time.monotonic()) + max(0.0, extra_s)
        if wait_s > 0:
            print(f" Estimated wait for {reason}: {wait_s:.3f} s")
            self._sleep_interruptible(wait_s)
        self.estimated_motion_done_at = max(self.estimated_motion_done_at, time.monotonic())

    def send_code(self, code: str, wait_s: float = 0.0, reason: str = "command") -> None:
        self._check_for_immediate_branch_rewind()
        if self.ui_state is not None:
            self.ui_state.set_next_command(code, reason)
            self.ui_state.wait_if_paused()
            current_mode = str(self.ui_state.snapshot().get("mode") or "")
            if self.in_print_polyline:
                self.ui_state.set_mode("printing", f"Sending: {reason}")
            elif current_mode not in {"node_tune", "travel_tune", "travel_rewind"}:
                self.ui_state.set_mode("moving", f"Sending: {reason}")
        print(f" -> {code}")
        self.rrf.send_code(code)
        if self.ui_state is not None:
            self.ui_state.set_last_command(code, reason)
        if self.inter_command_delay_s > 0:
            self._sleep_interruptible(self.inter_command_delay_s)
        if wait_s > 0:
            self.estimated_motion_done_at = max(time.monotonic(), self.estimated_motion_done_at) + float(wait_s)
            self._wait_estimated_complete(reason=reason)

    def send_absolute_move_axes(self, axes: Dict[str, float], feed: float, reason: str = "move") -> None:
        previous = self._current_axes()
        parts = ["G90", "G1"]
        for ax in self._axis_order():
            if ax in axes:
                parts.append(f"{ax}{float(axes[ax]):.3f}")
        parts.append(f"F{float(feed):.0f}")
        code = " ".join(parts)
        wait_s = self._estimate_wait_s(previous, {str(k): float(v) for k, v in axes.items()}, feed)
        self.send_code(code, wait_s=wait_s, reason=reason)

    def _commit_state(self, p_stage: np.ndarray, b: float, c: float) -> None:
        p_stage = np.asarray(p_stage, dtype=float)
        self.base.cur_stage_xyz = p_stage.copy()
        self.base.cur_b = float(b)
        self.base.cur_c = float(c)
        self.base.command_min = np.minimum(self.base.command_min, p_stage)
        self.base.command_max = np.maximum(self.base.command_max, p_stage)
        self.base.b_min_used = min(self.base.b_min_used, float(b))
        self.base.b_max_used = max(self.base.b_max_used, float(b))
        self.base.c_min_used = min(self.base.c_min_used, float(c))
        self.base.c_max_used = max(self.base.c_max_used, float(c))
        if self.ui_state is not None:
            tip_xyz = None
            try:
                if self.base.write_mode == "calibrated":
                    tip_xyz = self.vm.tip_xyz_for_stage(self.base.cal, p_stage, float(b), float(c))
                else:
                    tip_xyz = p_stage.copy()
            except Exception:
                tip_xyz = None
            self.ui_state.update_pose(p_stage, float(b), float(c), tip_xyz=tip_xyz)
            if self.in_print_polyline and tip_xyz is not None:
                self.ui_state.record_print_point(tip_xyz)

    def _commit_rotary_only_state(self, b: Optional[float] = None, c: Optional[float] = None) -> None:
        if self.base.cur_stage_xyz is None:
            return
        self._commit_state(
            np.asarray(self.base.cur_stage_xyz, dtype=float).copy(),
            float(self.base.cur_b if b is None else b),
            float(self.base.cur_c if c is None else c),
        )

    def _move_to_tip_with_explicit_bc(self, p_tip: np.ndarray, b: float, c: float, feed: float, reason: str) -> None:
        p_tip = np.asarray(p_tip, dtype=float).copy()
        if self.base.write_mode == "cartesian":
            p_stage = self.base.clamp_stage(p_tip, "cartesian_tip_to_stage_explicit_bc")
            self._send_atomic_stage_pose(p_stage, float(b), float(c), feed, reason=reason)
            self.base.cur_tip_xyz = p_tip.copy()
            self.base.last_tip_tangent = None
            return

        c_cmd = float(self.vm.unwrap_angle_deg_near(float(c), float(self.base.cur_c)))
        phase_key = self.vm._normalize_phase_key(getattr(self.base, "current_motion_phase", None))
        p_stage = self.base.clamp_stage(
            self.vm.stage_xyz_for_tip(self.base.cal, p_tip, float(b), c_cmd, motion_phase=phase_key),
            "tip_to_stage_explicit_bc",
        )
        self._send_atomic_stage_pose(p_stage, float(b), c_cmd, feed, reason=reason)
        self.base.cur_tip_xyz = p_tip.copy()
        self.base.last_tip_tangent = None
        if phase_key is not None:
            self.base.current_motion_phase = phase_key

    def _move_to_tip_with_explicit_bc_synced(
        self,
        p_tip: np.ndarray,
        b: float,
        c: float,
        feed: float,
        reason: str,
    ) -> np.ndarray:
        p_tip = np.asarray(p_tip, dtype=float).copy()
        if self.base.write_mode == "cartesian":
            p_stage = self.base.clamp_stage(p_tip, "cartesian_tip_to_stage_explicit_bc_synced")
            axes = {
                self.base.cal.x_axis: float(p_stage[0]),
                self.base.cal.y_axis: float(p_stage[1]),
                self.base.cal.z_axis: float(p_stage[2]),
            }
            self.send_absolute_move_axes(axes, float(feed), reason=reason)
            self._commit_state(p_stage, float(b), float(c))
            self.base.cur_tip_xyz = p_tip.copy()
            self.base.last_tip_tangent = None
            return p_stage

        c_cmd = float(self.vm.unwrap_angle_deg_near(float(c), float(self.base.cur_c)))
        phase_key = self.vm._normalize_phase_key(getattr(self.base, "current_motion_phase", None))
        p_stage = self.base.clamp_stage(
            self.vm.stage_xyz_for_tip(self.base.cal, p_tip, float(b), c_cmd, motion_phase=phase_key),
            "tip_to_stage_explicit_bc_synced",
        )
        axes = {
            self.base.cal.x_axis: float(p_stage[0]),
            self.base.cal.y_axis: float(p_stage[1]),
            self.base.cal.z_axis: float(p_stage[2]),
            self.base.cal.b_axis: float(b),
            self.base.cal.c_axis: float(c_cmd),
        }
        self.send_absolute_move_axes(axes, float(feed), reason=reason)
        self._commit_state(p_stage, float(b), c_cmd)
        self.base.cur_tip_xyz = p_tip.copy()
        self.base.last_tip_tangent = None
        if phase_key is not None:
            self.base.current_motion_phase = phase_key
        return p_stage

    def _tip_and_rotary_match_target(self, p_tip: np.ndarray, b: float, c: float, tol_mm: float = 1e-6, tol_deg: float = 1e-6) -> bool:
        if self.base.cur_tip_xyz is None:
            return False
        cur_tip = np.asarray(self.base.cur_tip_xyz, dtype=float).reshape(3)
        target_tip = np.asarray(p_tip, dtype=float).reshape(3)
        if float(np.linalg.norm(cur_tip - target_tip)) > float(tol_mm):
            return False
        c_target = float(self.vm.unwrap_angle_deg_near(float(c), float(self.base.cur_c)))
        return (
            abs(float(self.base.cur_b) - float(b)) <= float(tol_deg)
            and abs(float(self.base.cur_c) - c_target) <= float(tol_deg)
        )

    def mark_node_start_accepted(self, group_number: int) -> None:
        self._accepted_node_start_group = int(group_number)

    def consume_node_start_accepted(self, group_number: int) -> bool:
        if self._accepted_node_start_group is None or int(self._accepted_node_start_group) != int(group_number):
            return False
        self._accepted_node_start_group = None
        return True

    # ----- overrides for original writer methods -----

    def set_travel_acceleration(self, accel_mm_s2: float, comment: Optional[str] = None) -> None:
        accel = max(0.0, float(accel_mm_s2))
        if accel <= 0.0:
            return
        if self.base.active_travel_accel_mm_s2 is not None and abs(self.base.active_travel_accel_mm_s2 - accel) <= 1e-9:
            return
        if comment:
            print(f"; {comment}")
        self.send_code(f"M204 T{accel:.0f}", reason="acceleration")
        self.base.active_travel_accel_mm_s2 = accel

    def pressure_preload_before_print(self) -> None:
        if not self.base.emit_extrusion or self.base.pressure_charged:
            return
        self.send_code("M42 P0 S1", reason="pressure on")
        if self.base.pressure_offset_mm > 1e-9:
            self.send_code(f"G91", reason="relative mode")
            self.send_code(
                f"G1 {self.base.cal.u_axis}{float(self.base.pressure_offset_mm):.3f} F{float(self.base.pressure_advance_feed):.0f}",
                wait_s=abs(float(self.base.pressure_offset_mm)) / max(1e-9, float(self.base.pressure_advance_feed) / 60.0),
                reason="pressure preload",
            )
            self.send_code("G90", reason="absolute mode")
        if self.base.preflow_dwell_ms > 0:
            time.sleep(float(self.base.preflow_dwell_ms) / 1000.0)
        self.base.pressure_charged = True

    def pressure_release_after_print(self) -> None:
        if not self.base.emit_extrusion or not self.base.pressure_charged:
            return
        if self.base.pressure_offset_mm > 1e-9:
            self.send_code("G91", reason="relative mode")
            self.send_code(
                f"G1 {self.base.cal.u_axis}{-float(self.base.pressure_offset_mm):.3f} F{float(self.base.pressure_retract_feed):.0f}",
                wait_s=abs(float(self.base.pressure_offset_mm)) / max(1e-9, float(self.base.pressure_retract_feed) / 60.0),
                reason="pressure retract",
            )
            self.send_code("G90", reason="absolute mode")
        self.send_code("M42 P0 S0", reason="pressure off")
        self.base.pressure_charged = False

    def emit_vasculature_write_start(self, label: str, points: np.ndarray, tangents: np.ndarray, extrusion_multiplier: float) -> None:
        print(f"; VASCULATURE_WRITE_START path_id={label} point_count={len(points)}")

    def emit_vasculature_write_end(self, label: str) -> None:
        print(f"; VASCULATURE_WRITE_END path_id={label}")

    def _feed_with_rotary_limits(self, requested_feed: float, current_b: float, target_b: float, current_c: float, target_c: float) -> float:
        feed = float(requested_feed)
        if abs(float(target_b) - float(current_b)) > 1e-9:
            feed = min(feed, self.max_b_move_feed)
        if abs(float(target_c) - float(current_c)) > 1e-9:
            feed = max(feed, self.min_c_move_feed)
        return feed

    def write_move(self, p_stage: np.ndarray, b: float, c: float, feed: float,
                   comment: Optional[str] = None, u_value: Optional[float] = None) -> None:
        if comment:
            print(f"; {comment}")
        p_stage = np.asarray(p_stage, dtype=float)

        if (self.base.write_mode == "calibrated"
                and (not self.in_print_polyline)
                and abs(float(c) - float(self.base.cur_c)) > 1e-9):
            self.send_absolute_move_axes(
                {self.base.cal.c_axis: float(c)},
                self._feed_with_rotary_limits(self.base.c_feed, self.base.cur_b, self.base.cur_b, self.base.cur_c, c),
                reason="C move",
            )
            self.base.cur_c = float(c)
            self.base.c_min_used = min(self.base.c_min_used, float(c))
            self.base.c_max_used = max(self.base.c_max_used, float(c))

        if (self.base.write_mode == "calibrated"
                and (not self.in_print_polyline)
                and abs(float(b) - float(self.base.cur_b)) > 1e-9):
            self.send_absolute_move_axes(
                {self.base.cal.b_axis: float(b)},
                self._feed_with_rotary_limits(self._b_feed_limit_for_reason(comment), self.base.cur_b, b, self.base.cur_c, self.base.cur_c),
                reason="B move",
            )
            self._commit_rotary_only_state(b=float(b))

        axes = {
            self.base.cal.x_axis: float(p_stage[0]),
            self.base.cal.y_axis: float(p_stage[1]),
            self.base.cal.z_axis: float(p_stage[2]),
        }
        if self.base.write_mode == "calibrated":
            if self.in_print_polyline:
                axes[self.base.cal.b_axis] = float(b)
                axes[self.base.cal.c_axis] = float(c)
        if u_value is not None:
            axes[self.base.cal.u_axis] = float(u_value)
        move_feed = float(feed)
        if self.base.write_mode == "calibrated" and not self.in_print_polyline:
            move_feed = self._feed_with_rotary_limits(float(feed), self.base.cur_b, b, self.base.cur_c, c)
        self.send_absolute_move_axes(axes, move_feed, reason="print move" if self.in_print_polyline else "move")
        self._commit_state(p_stage, b, c)

    def write_tip_tracking_pose_move(self, target_tip: np.ndarray, target_b: float, target_c: float, feed: float,
                                     comment: Optional[str] = None, max_tip_step_mm: Optional[float] = None,
                                     max_b_step_deg: Optional[float] = None, max_c_step_deg: Optional[float] = None) -> None:
        if self.base.cur_stage_xyz is None:
            raise RuntimeError("Cannot emit tip-tracking pose move before machine pose is established.")
        if comment:
            print(f"; {comment}")
        if self.base.write_mode != "calibrated":
            self.write_move(np.asarray(target_tip, dtype=float), target_b, target_c, feed)
            self.base.cur_tip_xyz = np.asarray(target_tip, dtype=float).copy()
            self.base.last_tip_tangent = None
            return

        start_tip = (
            np.asarray(self.base.cur_tip_xyz, dtype=float).copy()
            if self.base.cur_tip_xyz is not None
            else self.vm.tip_xyz_for_stage(self.base.cal, np.asarray(self.base.cur_stage_xyz, dtype=float), self.base.cur_b, self.base.cur_c)
        )
        target_tip = np.asarray(target_tip, dtype=float).copy()
        start_b = float(self.base.cur_b)
        start_c = float(self.base.cur_c)
        end_b = float(target_b)
        end_c = float(self.vm.unwrap_angle_deg_near(float(target_c), start_c))

        tip_span = float(np.linalg.norm(target_tip - start_tip))
        b_span = abs(end_b - start_b)
        c_span = abs(end_c - start_c)
        tip_step = max(0.25, float(self.base.skeleton_collision_sample_step_mm) if max_tip_step_mm is None else float(max_tip_step_mm))
        b_step = max(0.25, float(self.base.b_max_step_deg) if max_b_step_deg is None else float(max_b_step_deg))
        c_step = max(0.25, float(self.base.c_max_step_deg) if max_c_step_deg is None else float(max_c_step_deg))
        n_steps = max(1,
                      int(math.ceil(tip_span / tip_step)) if tip_span > 1e-12 else 1,
                      int(math.ceil(b_span / b_step)) if b_span > 1e-12 else 1,
                      int(math.ceil(c_span / c_step)) if c_span > 1e-12 else 1)
        old = self.in_tip_tracking_interpolation
        self.in_tip_tracking_interpolation = True
        try:
            for i in range(1, n_steps + 1):
                t = i / float(n_steps)
                tip_i = start_tip + t * (target_tip - start_tip)
                b_i = start_b + t * (end_b - start_b)
                c_i = start_c + t * (end_c - start_c)
                p_stage_i = self.base.clamp_stage(self.vm.stage_xyz_for_tip(self.base.cal, tip_i, b_i, c_i), "tip_tracking_pose_move")
                self.write_move(p_stage_i, b_i, c_i, feed)
                self.base.cur_tip_xyz = tip_i.copy()
        finally:
            self.in_tip_tracking_interpolation = old

    def write_tip_speed_move(
        self,
        p_stage: np.ndarray,
        b: float,
        c: float,
        desired_tip_feed: float,
        start_tip_xyz: np.ndarray,
        end_tip_xyz: np.ndarray,
        max_b_axis_feed: Optional[float] = None,
        motion_phase: Optional[str] = None,
        requested_tip_angle: Optional[float] = None,
        comment: Optional[str] = None,
    ) -> None:
        if self.base.cur_stage_xyz is None:
            raise RuntimeError("Cannot emit tip-speed move before machine pose is established.")
        if comment:
            print(f"; {comment}")

        p_stage = np.asarray(p_stage, dtype=float)
        start_tip = np.asarray(start_tip_xyz, dtype=float)
        end_tip = np.asarray(end_tip_xyz, dtype=float)
        c_cmd = float(self.vm.unwrap_angle_deg_near(float(c), float(self.base.cur_c)))

        tip_dist = float(np.linalg.norm(end_tip - start_tip))
        stage_dist = float(np.linalg.norm(p_stage - np.asarray(self.base.cur_stage_xyz, dtype=float)))
        b_dist = abs(float(b) - float(self.base.cur_b))
        c_dist = abs(float(c_cmd) - float(self.base.cur_c))
        axis_dist = float(np.linalg.norm(np.array([stage_dist, b_dist, c_dist], dtype=float)))

        if tip_dist <= 1e-12 or axis_dist <= 1e-12:
            command_feed = float(max(desired_tip_feed, 1.0))
        else:
            desired_time_min = tip_dist / float(max(desired_tip_feed, 1e-9))
            command_feed = float(max(axis_dist / max(desired_time_min, 1e-12), 1.0))
        command_feed = self._cap_feed_for_b_axis_component(
            command_feed,
            stage_dist=stage_dist,
            b_dist=b_dist,
            c_dist=c_dist,
            max_b_axis_feed=max_b_axis_feed,
        )

        axes = {
            self.base.cal.x_axis: float(p_stage[0]),
            self.base.cal.y_axis: float(p_stage[1]),
            self.base.cal.z_axis: float(p_stage[2]),
        }
        if self.base.write_mode == "calibrated":
            axes[self.base.cal.b_axis] = float(b)
            axes[self.base.cal.c_axis] = float(c_cmd)
        self.send_absolute_move_axes(axes, command_feed, reason="print move")
        self._commit_state(p_stage, b, c_cmd)
        self.base.cur_tip_xyz = end_tip.copy()
        if motion_phase is not None:
            self.base.current_motion_phase = self.vm._normalize_phase_key(motion_phase)
        if requested_tip_angle is not None:
            self.base.last_requested_tip_angle = float(requested_tip_angle)

    def _build_print_samples(self, points: np.ndarray, tangents: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
        samples: List[Tuple[np.ndarray, np.ndarray]] = []
        for i in range(1, len(points)):
            p0 = np.asarray(points[i - 1], dtype=float)
            p1 = np.asarray(points[i], dtype=float)
            seg_t0 = np.asarray(tangents[i - 1], dtype=float)
            seg_t1 = np.asarray(tangents[i], dtype=float)
            for s in range(1, self.edge_samples + 1):
                t = s / float(self.edge_samples)
                p_tip = p0 + t * (p1 - p0)
                tangent = self.vm.normalize((1.0 - t) * seg_t0 + t * seg_t1)
                samples.append((p_tip, tangent))
        return samples

    def print_polyline(
        self,
        points: np.ndarray,
        tangents: np.ndarray,
        extrusion_multiplier: float,
        label: str,
        path_radius_mm: float = 0.0,
        print_feed_override: Optional[float] = None,
        radius_profile_mm: Optional[np.ndarray] = None,
    ) -> None:
        old = self.in_print_polyline
        self.in_print_polyline = True
        if self.ui_state is not None:
            self.ui_state.set_mode("printing", f"Printing {label}")
            self.ui_state.current_point_total = int(len(points))
            self.ui_state.current_point_index = 0
        try:
            if len(points) < 2:
                return
            active_print_feed = float(self.base.print_feed if print_feed_override is None else print_feed_override)
            self.pressure_preload_before_print()
            mode_note = "Cartesian centerline" if self.base.write_mode == "cartesian" else "calibration-based exact tip tracking"
            print(f"; print {label} ({mode_note}); active_print_feed={active_print_feed:.3f} mm/min; path_radius_mm={float(path_radius_mm):.6f}")
            self.emit_vasculature_write_start(label, points, tangents, extrusion_multiplier)

            points = np.asarray(points, dtype=float)
            tangents = np.asarray(tangents, dtype=float)
            rr = None if radius_profile_mm is None else np.asarray(radius_profile_mm, dtype=float).reshape(-1)
            if rr is not None and len(rr) != len(points):
                raise ValueError(f"Radius profile length mismatch for {label}: {len(rr)} radii for {len(points)} points.")
            if self.base.cur_tip_xyz is None:
                self.base.cur_tip_xyz = np.asarray(points[0], dtype=float).copy()
            if self.base.last_tip_tangent is None:
                self.base.last_tip_tangent = np.asarray(tangents[0], dtype=float).copy()
            samples: List[Tuple[np.ndarray, np.ndarray, float]] = []
            for idx in range(1, len(points)):
                p0 = np.asarray(points[idx - 1], dtype=float)
                p1 = np.asarray(points[idx], dtype=float)
                seg_t0 = np.asarray(tangents[idx - 1], dtype=float)
                seg_t1 = np.asarray(tangents[idx], dtype=float)
                r0 = float(path_radius_mm if rr is None else rr[idx - 1])
                r1 = float(path_radius_mm if rr is None else rr[idx])
                for s in range(1, self.edge_samples + 1):
                    t = s / float(self.edge_samples)
                    p_tip = p0 + t * (p1 - p0)
                    tangent = self.vm.normalize((1.0 - t) * seg_t0 + t * seg_t1)
                    local_radius_mm = float((1.0 - t) * r0 + t * r1)
                    samples.append((p_tip, tangent, local_radius_mm))
            total_samples = len(samples)
            if self.ui_state is not None:
                self.ui_state.current_point_total = total_samples
                self.ui_state.current_point_index = 0

            i = 0
            last_tip = np.asarray(points[0], dtype=float).copy()
            while i < total_samples:
                p_tip, tangent, local_radius_mm = samples[i]
                p_stage, b, c, motion_phase, used_tip_angle = self.base.tip_to_stage_with_phase(
                    p_tip,
                    tangent=tangent,
                    prev_c=self.base.cur_c,
                    prev_b=self.base.cur_b,
                )
                local_feed = active_print_feed
                if self.base.extrusion_feed_mode == "weighted" and self.base.radius_feed_map:
                    local_feed = self.vm.feed_from_radius_map(local_radius_mm, self.base.radius_feed_map, active_print_feed)
                boundary_b_feed = None
                if self.start_stop_slow_samples > 0 and self.start_stop_b_move_feed > 0.0:
                    if i < self.start_stop_slow_samples or (total_samples - 1 - i) < self.start_stop_slow_samples:
                        boundary_b_feed = self.start_stop_b_move_feed
                self.write_tip_speed_move(
                    p_stage=p_stage,
                    b=b,
                    c=c,
                    desired_tip_feed=local_feed,
                    start_tip_xyz=last_tip,
                    end_tip_xyz=p_tip,
                    max_b_axis_feed=boundary_b_feed,
                    motion_phase=motion_phase,
                    requested_tip_angle=used_tip_angle,
                )
                self.base.last_tip_tangent = tangent.copy()
                last_tip = p_tip.copy()
                if self.ui_state is not None:
                    self.ui_state.current_point_index = i + 1
                i += 1

            self._wait_estimated_complete(reason="print path completion")
            self.emit_vasculature_write_end(label)
            self.pressure_release_after_print()
            self.base.record_printed_path(points, radius_mm=path_radius_mm)
            self.previous_branch_end_pose = (
                np.asarray(self.base.cur_stage_xyz, dtype=float).copy(),
                float(self.base.cur_b),
                float(self.base.cur_c),
                None if self.base.cur_tip_xyz is None else np.asarray(self.base.cur_tip_xyz, dtype=float).copy(),
            )
            if self.base.end_overextension_mm > 1e-9:
                extension_dir = self.base.terminal_overextension_direction(points, tangents)
                extension_tip = np.asarray(points[-1], dtype=float) + extension_dir * float(self.base.end_overextension_mm)
                self._with_patched_base(lambda: self.base.move_to_tip(
                    extension_tip,
                    tangent=extension_dir,
                    feed=self.base.end_overextension_feed,
                    comment=(
                        f"{label}: pressure-off terminal overextension "
                        f"{float(self.base.end_overextension_mm):.3f} mm along terminal branch direction"
                    ),
                ))
        finally:
            self.in_print_polyline = old

    def _with_patched_base(self, func):
        names = [
            "write_move", "write_tip_tracking_pose_move", "set_travel_acceleration",
            "pressure_preload_before_print", "pressure_release_after_print",
            "emit_vasculature_write_start", "emit_vasculature_write_end",
            "move_to_tip_with_explicit_bc", "move_to_tip_aorta_cylinder_azimuth_plane",
        ]
        old = {n: getattr(self.base, n) for n in names}
        for n in names:
            setattr(self.base, n, getattr(self, n))
        try:
            return func()
        finally:
            for n, v in old.items():
                setattr(self.base, n, v)

    def move_to_tip_with_explicit_bc(
        self,
        p_tip: np.ndarray,
        b: float,
        c: float,
        feed: float,
        comment: Optional[str] = None,
        motion_phase: Optional[str] = None,
        requested_tip_angle: Optional[float] = None,
    ) -> None:
        if motion_phase is not None:
            self.base.current_motion_phase = self.vm._normalize_phase_key(motion_phase)
        self._move_to_tip_with_explicit_bc(
            np.asarray(p_tip, dtype=float),
            float(b),
            float(c),
            float(feed),
            reason="move" if comment is None else str(comment),
        )
        if requested_tip_angle is not None:
            self.base.last_requested_tip_angle = float(requested_tip_angle)

    def move_to_tip_aorta_cylinder_azimuth_plane(
        self,
        p_tip: np.ndarray,
        axis_tangent: np.ndarray,
        radial_unit: np.ndarray,
        feed: float,
        comment: Optional[str] = None,
    ) -> None:
        self._with_patched_base(
            lambda: self.base.move_to_tip_aorta_cylinder_azimuth_plane(
                np.asarray(p_tip, dtype=float),
                np.asarray(axis_tangent, dtype=float),
                np.asarray(radial_unit, dtype=float),
                float(feed),
                comment=comment,
            )
        )

    def approach_start_from_aorta_cylinder(
        self,
        start_tip: np.ndarray,
        start_tangent: np.ndarray,
        label: str,
        final_approach_feed: Optional[float] = None,
    ) -> None:
        self._with_patched_base(
            lambda: self.base.approach_start_from_aorta_cylinder(
                np.asarray(start_tip, dtype=float),
                np.asarray(start_tangent, dtype=float),
                label,
                final_approach_feed=final_approach_feed,
            )
        )

    def approach_start_direct(self, start_tip: np.ndarray, start_tangent: np.ndarray, label: str, final_approach_feed: Optional[float] = None) -> None:
        self._with_patched_base(lambda: self.base.approach_start_direct(start_tip, start_tangent, label, final_approach_feed=final_approach_feed))

    def approach_start_from_side(self, start_tip: np.ndarray, start_tangent: np.ndarray, far_clearance: float,
                                 near_clearance: float, retreat_clearance: float, side_lift_z: float, label: str,
                                 final_approach_feed: Optional[float] = None) -> None:
        self._with_patched_base(lambda: self.base.approach_start_from_side(start_tip, start_tangent, far_clearance, near_clearance, retreat_clearance, side_lift_z, label, final_approach_feed=final_approach_feed))

    def apply_group_displacement(self, displacement: Any, default_feed: float, label: str) -> None:
        if not displacement.enabled:
            return
        if self.base.cur_stage_xyz is None:
            raise RuntimeError(f"Cannot apply group displacement for {label} before any stage pose has been established.")

        delta_stage = np.asarray(displacement.delta_tip_xyz, dtype=float).reshape(3)
        delta_b = float(displacement.delta_b_deg)
        delta_c = float(displacement.delta_c_deg)
        if float(np.linalg.norm(delta_stage)) <= 1e-12 and abs(delta_b) <= 1e-12 and abs(delta_c) <= 1e-12:
            return

        current_tip = (
            np.asarray(self.base.cur_tip_xyz, dtype=float).copy()
            if self.base.cur_tip_xyz is not None
            else self.vm.tip_xyz_for_stage(
                self.base.cal,
                np.asarray(self.base.cur_stage_xyz, dtype=float),
                self.base.cur_b,
                self.base.cur_c,
                motion_phase=getattr(self.base, "current_motion_phase", None),
            )
        )
        target_tip = np.asarray(current_tip, dtype=float).copy() + delta_stage
        target_b = float(self.base.cur_b) + delta_b
        target_c = float(self.base.cur_c) + delta_c
        feed = float(default_feed if displacement.feed is None else displacement.feed)
        note_suffix = f" ({displacement.note})" if displacement.note else ""
        self._move_to_tip_with_explicit_bc_synced(
            target_tip,
            target_b,
            target_c,
            feed,
            reason=f"{label}: manual stage displacement{note_suffix}",
        )

    def _send_atomic_stage_pose(self, p_stage: np.ndarray, b: float, c: float, feed: float, reason: str = "move") -> None:
        p_stage = np.asarray(p_stage, dtype=float)
        if self.base.write_mode == "calibrated":
            if abs(float(b) - float(self.base.cur_b)) > 1e-9:
                self.send_absolute_move_axes(
                    {self.base.cal.b_axis: float(b)},
                    self._feed_with_rotary_limits(self._b_feed_limit_for_reason(reason), self.base.cur_b, b, self.base.cur_c, self.base.cur_c),
                    reason="B move",
                )
                self._commit_rotary_only_state(b=float(b))
            if abs(float(c) - float(self.base.cur_c)) > 1e-9:
                self.send_absolute_move_axes(
                    {self.base.cal.c_axis: float(c)},
                    self._feed_with_rotary_limits(self.min_c_move_feed, self.base.cur_b, self.base.cur_b, self.base.cur_c, c),
                    reason="C move",
                )
                self._commit_rotary_only_state(c=float(c))
        axes = {
            self.base.cal.x_axis: float(p_stage[0]),
            self.base.cal.y_axis: float(p_stage[1]),
            self.base.cal.z_axis: float(p_stage[2]),
        }
        self.send_absolute_move_axes(axes, float(feed), reason=reason)
        self._commit_state(p_stage, b, c)

    def apply_node_rotary_offset(
        self,
        offset: ManualGroupOffset,
        start_tip: np.ndarray,
        start_tangent: np.ndarray,
        feed: Optional[float] = None,
    ) -> None:
        if abs(float(offset.node_delta_b_deg)) <= 1e-12 and abs(float(offset.node_delta_c_deg)) <= 1e-12:
            return
        _, base_b, base_c = self.base.tip_to_stage(
            np.asarray(start_tip, dtype=float),
            tangent=np.asarray(start_tangent, dtype=float),
            prev_c=self.base.cur_c,
            prev_b=self.base.cur_b,
        )
        target_b = float(base_b) + float(offset.node_delta_b_deg)
        target_c = float(base_c) + float(offset.node_delta_c_deg)
        target_c = float(self.vm.unwrap_angle_deg_near(target_c, float(self.base.cur_c)))
        if abs(target_b - float(self.base.cur_b)) <= 1e-9 and abs(target_c - float(self.base.cur_c)) <= 1e-9:
            return
        self._move_to_tip_with_explicit_bc_synced(
            np.asarray(start_tip, dtype=float),
            target_b,
            target_c,
            self.tune_feed if feed is None else float(feed),
            reason="node rotary tune",
        )

    # ----- manual tuning loops -----

    def tune_node_start(self, group_number: int, path_id: str, base_start_tip: np.ndarray, start_tangent: np.ndarray,
                        offset: ManualGroupOffset) -> ManualGroupOffset:
        print("\n" + "=" * 72)
        print(f"Manual node tuning for group {group_number}: {path_id}")
        print("Keys: a/d X-/X+, w/s Y-/Y+, ↑/↓ Z+/Z-, i/k B+/B-, l/j C+/C-, Right/Space accept, q/Esc abort")
        print(f"Current node offset: {offset.node_gantry_offset_mm}")
        print("=" * 72)
        if self.ui_state is not None:
            self.ui_state.set_mode("node_tune", f"Tune start node for group {group_number}. Use keys/buttons, then Space/Accept.")
            update_manual_window_state(self.ui_state, self.manual_offsets, group_number, getattr(self.ui_state, "plan_paths", None))

        # Tune node start by moving machine XYZ directly while keeping B/C fixed.
        base_stage, base_b, base_c = self.base.tip_to_stage(
            np.asarray(base_start_tip, dtype=float),
            tangent=np.asarray(start_tangent, dtype=float),
            prev_c=self.base.cur_c,
            prev_b=self.base.cur_b,
        )
        working = np.asarray(offset.node_gantry_offset_mm, dtype=float).copy()
        working_b = float(offset.node_delta_b_deg)
        working_c = float(offset.node_delta_c_deg)
        target_tip = np.asarray(base_start_tip, dtype=float).copy() + working
        target_b = base_b + working_b
        target_c = base_c + working_c
        if not self._tip_and_rotary_match_target(target_tip, target_b, target_c):
            self._move_to_tip_with_explicit_bc_synced(target_tip, target_b, target_c, self.tune_feed, reason="manual node tune")
        self.base.last_tip_tangent = np.asarray(start_tangent, dtype=float).copy()

        while True:
            key = self.keyboard.get_key(0.1)
            if key is None:
                continue
            if key in {"q", "Q", "ESC"}:
                raise KeyboardInterrupt("Manual tuning aborted by user.")
            if key in {"SPACE", "RIGHT", "FORWARD"}:
                offset.node_gantry_offset_mm = working.copy()
                offset.node_delta_b_deg = float(working_b)
                offset.node_delta_c_deg = float(working_c)
                offset.note = f"node taught {time.strftime('%Y-%m-%d %H:%M:%S')}"
                self.manual_offsets[group_number] = offset
                save_manual_offsets(self.manual_file, self.manual_offsets, getattr(self.ui_state, "plan_paths", None))
                self.mark_node_start_accepted(group_number)
                update_manual_window_state(self.ui_state, self.manual_offsets, group_number, getattr(self.ui_state, "plan_paths", None))
                if self.ui_state is not None:
                    self.ui_state.set_mode("moving", f"Accepted and saved node offset for group {group_number}.")
                print(f"Accepted node offset for group {group_number}: {working}")
                return offset
            delta = np.zeros(3, dtype=float)
            if key == "a":
                delta[0] -= self.node_tune_step_mm
            elif key == "d":
                delta[0] += self.node_tune_step_mm
            elif key == "w":
                delta[1] -= self.node_tune_step_mm
            elif key == "s":
                delta[1] += self.node_tune_step_mm
            elif key == "UP":
                delta[2] += self.node_tune_step_mm
            elif key == "DOWN":
                delta[2] -= self.node_tune_step_mm
            elif key == "i":
                working_b += self.b_tune_step_deg
            elif key == "k":
                working_b -= self.b_tune_step_deg
            elif key == "l":
                working_c += self.c_tune_step_deg
            elif key == "j":
                working_c -= self.c_tune_step_deg
            else:
                continue
            working += delta
            if self.ui_state is not None:
                self.ui_state.set_mode("node_tune", f"Node offset group {group_number}: [{working[0]:.3f}, {working[1]:.3f}, {working[2]:.3f}] dB={working_b:.3f} dC={working_c:.3f}")
            target_tip = np.asarray(base_start_tip, dtype=float).copy() + working
            self._move_to_tip_with_explicit_bc_synced(target_tip, base_b + working_b, base_c + working_c, self.tune_feed, reason="manual node tune")
            self.base.last_tip_tangent = np.asarray(start_tangent, dtype=float).copy()
            print(f" node_offset = [{working[0]:.3f}, {working[1]:.3f}, {working[2]:.3f}] dB={working_b:.3f} dC={working_c:.3f}", end="\r", flush=True)

    def tune_group_travel(self, group_number: int, path_id: str, offset: ManualGroupOffset) -> ManualGroupOffset:
        print("\n" + "=" * 72)
        print(f"Manual travel tuning after group {group_number}: {path_id}")
        print("Keys: a/d X-/X+, w/s Y-/Y+, ↑/↓ Z+/Z-, i/k B+/B-, l/j C+/C-, Right/Space accept, q/Esc abort")
        print(f"Current travel offset: {offset.delta_tip_xyz} dB={offset.delta_b_deg:.3f} dC={offset.delta_c_deg:.3f}")
        print("=" * 72)
        if self.base.cur_stage_xyz is None:
            raise RuntimeError("Cannot tune travel before any stage pose has been established.")

        base_stage = np.asarray(self.base.cur_stage_xyz, dtype=float).copy()
        base_b = float(self.base.cur_b)
        base_c = float(self.base.cur_c)
        base_tip = (
            np.asarray(self.base.cur_tip_xyz, dtype=float).copy()
            if self.base.cur_tip_xyz is not None
            else self.vm.tip_xyz_for_stage(
                self.base.cal,
                base_stage,
                base_b,
                base_c,
                motion_phase=getattr(self.base, "current_motion_phase", None),
            )
        )
        working = np.asarray(offset.delta_tip_xyz, dtype=float).copy()
        working_b = float(offset.delta_b_deg)
        working_c = float(offset.delta_c_deg)

        if self.ui_state is not None:
            self.ui_state.set_mode("travel_tune", f"Tune post-branch travel for group {group_number}. Use keys/buttons, then Space/Accept.")
            self.ui_state.begin_manual_travel_trace(base_stage)

        def move_to_working_target() -> None:
            target_tip = np.asarray(base_tip, dtype=float).copy() + working
            target_b = base_b + working_b
            target_c = base_c + working_c
            target_stage = self._move_to_tip_with_explicit_bc_synced(
                target_tip,
                target_b,
                target_c,
                self.tune_feed,
                reason="manual travel tune",
            )
            if self.ui_state is not None:
                self.ui_state.append_manual_travel_point(target_stage)

        if float(np.linalg.norm(working)) > 1e-12 or abs(working_b) > 1e-12 or abs(working_c) > 1e-12:
            move_to_working_target()

        while True:
            key = self.keyboard.get_key(0.1)
            if key is None:
                continue
            if key in {"q", "Q", "ESC"}:
                raise KeyboardInterrupt("Manual travel tuning aborted by user.")
            if key in {"SPACE", "RIGHT", "FORWARD"}:
                offset.delta_tip_xyz = working.copy()
                offset.delta_b_deg = float(working_b)
                offset.delta_c_deg = float(working_c)
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                base_note = str(offset.note or "").strip()
                travel_note = f"travel taught {timestamp}"
                offset.note = travel_note if not base_note else f"{base_note} | {travel_note}"
                self.manual_offsets[group_number] = offset
                save_manual_offsets(self.manual_file, self.manual_offsets, getattr(self.ui_state, "plan_paths", None))
                update_manual_window_state(self.ui_state, self.manual_offsets, group_number, getattr(self.ui_state, "plan_paths", None))
                if self.ui_state is not None:
                    self.ui_state.set_mode("moving", f"Accepted and saved travel offset for group {group_number}.")
                print(f"Accepted travel offset for group {group_number}: {working} dB={working_b:.3f} dC={working_c:.3f}")
                return offset

            delta = np.zeros(3, dtype=float)
            if key == "a":
                delta[0] -= self.travel_tune_step_mm
            elif key == "d":
                delta[0] += self.travel_tune_step_mm
            elif key == "w":
                delta[1] -= self.travel_tune_step_mm
            elif key == "s":
                delta[1] += self.travel_tune_step_mm
            elif key == "UP":
                delta[2] += self.travel_tune_step_mm
            elif key == "DOWN":
                delta[2] -= self.travel_tune_step_mm
            elif key == "i":
                working_b += self.b_tune_step_deg
            elif key == "k":
                working_b -= self.b_tune_step_deg
            elif key == "l":
                working_c += self.c_tune_step_deg
            elif key == "j":
                working_c -= self.c_tune_step_deg
            else:
                continue

            working += delta
            if self.ui_state is not None:
                self.ui_state.set_mode("travel_tune", f"Travel offset group {group_number}: [{working[0]:.3f}, {working[1]:.3f}, {working[2]:.3f}] dB={working_b:.3f} dC={working_c:.3f}")
            move_to_working_target()
            print(f" travel_offset = [{working[0]:.3f}, {working[1]:.3f}, {working[2]:.3f}] dB={working_b:.3f} dC={working_c:.3f}", end="\r", flush=True)


# ------------------------- planning shared with vasc_manual.main -------------------------

def build_plan_from_args(vm: Any, args: argparse.Namespace):
    write_mode = str(args.write_mode).strip().lower()
    manual_offsets = parse_manual_offsets(args.group_displacements_file, vm)
    normalize_manual_offset_feeds(manual_offsets, default_feed=float(args.travel_feed))

    vessels = vm.parse_vessel_file(args.vessels)
    selected_vessel_ids = vm.parse_vessel_id_selection(args.vessel_ids)
    vessels = vm.filter_vessels_by_id(vessels, selected_vessel_ids)
    preferred_main_vessel_ids = None if args.main_vessel_ids is None else vm.parse_vessel_id_selection(args.main_vessel_ids)
    rotate_origin = (float(args.rotate_origin_x), float(args.rotate_origin_y), float(args.rotate_origin_z))
    transform_info = vm.transform_vessel_geometry_with_z(
        vessels,
        rotate_y_deg=float(args.rotate_y_deg),
        rotate_z_deg=float(args.rotate_z_deg),
        rotate_origin=rotate_origin,
    )
    scale_origin, geometry_scale = vm.scale_vessel_geometry(vessels, float(args.geometry_scale))
    alignment_info = None
    if args.align_lowest_centroid is not None:
        alignment_info = vm.align_vessel_lowest_centroid(vessels, args.align_lowest_centroid)
    vm.infer_parent_relationships(vessels, attach_threshold=float(args.node_attach_threshold))
    vm.orient_vessels(vessels)

    if write_mode == "calibrated":
        if not args.calibration:
            raise ValueError("--calibration is required when --write-mode calibrated")
        cal = vm.load_calibration(args.calibration, requested_offplane_fit_mode=str(args.offplane_fit_mode))
    else:
        cal = vm.Calibration(
            pr=np.zeros(1), pz=np.zeros(1), py_off=None, pa=None,
            b_min=0.0, b_max=180.0,
            x_axis="X", y_axis="Y", z_axis="Z", b_axis="B", c_axis="C", u_axis="U",
            c_180_deg=180.0,
            requested_offplane_fit_mode=str(args.offplane_fit_mode),
        )

    robot_skeleton_path = None if args.robot_skeleton_file is None else Path(str(args.robot_skeleton_file)).expanduser()
    if robot_skeleton_path is None and args.calibration:
        robot_skeleton_path = vm.resolve_robot_skeleton_from_calibration(str(args.calibration))
    robot_skeleton_ref = vm.load_robot_skeleton_reference(str(robot_skeleton_path)) if robot_skeleton_path is not None else None

    bbox = {
        "x_min": float(args.bbox_x_min), "x_max": float(args.bbox_x_max),
        "y_min": float(args.bbox_y_min), "y_max": float(args.bbox_y_max),
        "z_min": float(args.bbox_z_min), "z_max": float(args.bbox_z_max),
    }

    paths = vm.build_print_paths(
        vessels=vessels,
        vessel_order_mode=str(args.vessel_order_mode),
        simplify_paths=bool(args.simplify_endpoint_chains),
        chain_merge_threshold=float(args.chain_merge_threshold),
        path_resample_spacing=float(args.path_resample_spacing),
        path_geometry_smooth_window=int(args.path_geometry_smooth_window),
        point_merge_tol=float(args.point_merge_tol),
        min_group_length_mm=float(args.min_group_length_mm),
        force_bottom_to_top=bool(args.force_bottom_to_top),
        branch_start_overlap_mm=float(args.branch_start_overlap_mm),
        branch_overlap_tangent_window_mm=float(args.branch_overlap_tangent_window_mm),
        preferred_main_vessel_ids=preferred_main_vessel_ids,
    )
    manual_offsets = ensure_manual_offsets_cover_paths(manual_offsets, paths)
    normalize_manual_offset_feeds(manual_offsets, default_feed=float(args.travel_feed))
    group_displacements = {k: v.to_vm(vm) for k, v in manual_offsets.items()}

    return {
        "vessels": vessels,
        "paths": paths,
        "group_parent_map": vm.build_group_parent_map(paths),
        "cal": cal,
        "bbox": bbox,
        "robot_skeleton_ref": robot_skeleton_ref,
        "manual_offsets": manual_offsets,
        "group_displacements": group_displacements,
        "preferred_main_vessel_ids": preferred_main_vessel_ids,
        "rotate_origin": rotate_origin,
        "rotate_z_origin": np.asarray(transform_info["rotate_z_origin"], dtype=float).copy(),
        "scale_origin": scale_origin,
        "geometry_scale": geometry_scale,
        "alignment_info": alignment_info,
        "travel_feed": float(args.travel_feed),
    }


def save_gcode(vm: Any, args: argparse.Namespace, plan: Dict[str, Any]) -> None:
    group_displacements = {k: v.to_vm(vm) for k, v in plan["manual_offsets"].items()}
    if str(args.orientation_mode).strip().lower() == "fixed":
        for disp in group_displacements.values():
            disp.node_gantry_offset_mm = np.zeros(3, dtype=float)
            disp.node_delta_b_deg = 0.0
            disp.node_delta_c_deg = 0.0
    radius_feed_map = vm.parse_radius_feed_map(
        raw=str(args.radius_feed_map or ""),
        file_path=None if args.radius_feed_map_file is None else str(args.radius_feed_map_file),
    )
    ranges = vm.write_vessel_gcode(
        out_path=args.out,
        vessels=plan["vessels"],
        cal=plan["cal"],
        bbox=plan["bbox"],
        travel_feed=float(args.travel_feed),
        approach_feed=float(args.approach_feed),
        fine_approach_feed=float(args.fine_approach_feed),
        print_feed=float(args.print_feed),
        c_feed=float(args.c_feed),
        extrusion_per_mm=0.0 if bool(args.no_extrusion) else float(args.extrusion_per_mm),
        prime_mm=float(args.prime_mm),
        pressure_offset_mm=float(args.pressure_offset_mm),
        pressure_advance_feed=float(args.pressure_advance_feed),
        pressure_retract_feed=float(args.pressure_retract_feed),
        preflow_dwell_ms=int(args.preflow_dwell_ms),
        start_node_dwell_ms=int(args.start_node_dwell_ms),
        node_dwell_ms=int(args.node_dwell_ms),
        edge_samples=int(args.edge_samples),
        write_mode=str(args.write_mode),
        orientation_mode=str(args.orientation_mode),
        bc_solve_samples=int(args.bc_solve_samples),
        b_max_step_deg=float(args.b_max_step_deg),
        c_max_step_deg=float(args.c_max_step_deg),
        b_smoothing_alpha=float(args.b_smoothing_alpha),
        c_smoothing_alpha=float(args.c_smoothing_alpha),
        travel_c_max_step_deg=None if args.travel_c_max_step_deg is None else float(args.travel_c_max_step_deg),
        travel_c_smoothing_alpha=None if args.travel_c_smoothing_alpha is None else float(args.travel_c_smoothing_alpha),
        c_min_deg=float(args.c_min_deg),
        c_max_deg=float(args.c_max_deg),
        min_tangent_xy_for_c=float(args.min_tangent_xy_for_c),
        tangent_smooth_window=int(args.tangent_smooth_window),
        centerline_smooth_window=int(args.centerline_smooth_window),
        rotate_y_deg=float(args.rotate_y_deg),
        rotate_z_deg=float(args.rotate_z_deg),
        rotate_origin=plan["rotate_origin"],
        rotate_z_origin=plan["rotate_z_origin"],
        geometry_scale=float(plan["geometry_scale"]),
        vessel_order_mode=str(args.vessel_order_mode),
        simplify_paths=bool(args.simplify_endpoint_chains),
        chain_merge_threshold=float(args.chain_merge_threshold),
        path_resample_spacing=float(args.path_resample_spacing),
        path_geometry_smooth_window=int(args.path_geometry_smooth_window),
        point_merge_tol=float(args.point_merge_tol),
        min_group_length_mm=float(args.min_group_length_mm),
        force_bottom_to_top=bool(args.force_bottom_to_top),
        branch_start_overlap_mm=float(args.branch_start_overlap_mm),
        branch_overlap_tangent_window_mm=float(args.branch_overlap_tangent_window_mm),
        preferred_main_vessel_ids=plan["preferred_main_vessel_ids"],
        extrusion_multiplier_main=float(args.extrusion_multiplier_main),
        extrusion_multiplier_branch=float(args.extrusion_multiplier_branch),
        side_approach_far=float(args.side_approach_far),
        side_approach_near=float(args.side_approach_near),
        side_retreat=float(args.side_retreat),
        side_lift_z=float(args.side_lift_z),
        travel_clearance_above_printed_z=float(args.travel_clearance_above_printed_z),
        travel_bbox_margin=float(args.travel_bbox_margin),
        travel_edge_clearance=float(args.travel_edge_clearance),
        enable_travel_bbox_clearance=bool(args.enable_travel_bbox_clearance),
        enable_side_approach=bool(args.enable_side_approach),
        fine_approach_distance=float(args.fine_approach_distance),
        robot_skeleton_ref=plan["robot_skeleton_ref"],
        skeleton_collision_clearance=float(args.skeleton_collision_clearance),
        skeleton_collision_sample_step_mm=float(args.skeleton_collision_sample_step_mm),
        travel_accel_mm_s2=float(args.travel_accel_mm_s2),
        post_print_travel_accel_scale=float(args.post_print_travel_accel_scale),
        travel_strategy=str(args.travel_strategy),
        enable_aorta_cylinder_travel=bool(args.enable_aorta_cylinder_travel),
        aorta_cylinder_clearance_mm=float(args.aorta_cylinder_clearance_mm),
        aorta_cylinder_min_radius_mm=float(args.aorta_cylinder_min_radius_mm),
        aorta_cylinder_samples=int(args.aorta_cylinder_samples),
        machine_start_pose=(float(args.machine_start_x), float(args.machine_start_y), float(args.machine_start_z), float(args.machine_start_b), float(args.machine_start_c)),
        machine_end_pose=(float(args.machine_end_x), float(args.machine_end_y), float(args.machine_end_z), float(args.machine_end_b), float(args.machine_end_c)),
        extrusion_feed_mode=str(args.extrusion_feed_mode),
        radius_feed_map=radius_feed_map,
        group_displacements=group_displacements,
    )
    print(f"Wrote G-code to {args.out}")
    print(f"Ranges: {ranges}")


def run_interactive_print(vm: Any, args: argparse.Namespace, plan: Dict[str, Any], ui_state: Optional[SharedPrintState] = None) -> None:
    """Run the physical print. If ui_state is supplied, GUI keys/buttons are used.

    Without ui_state, this remains the terminal/keyboard mode from the previous
    version. With ui_state, Tk runs on the main thread and this function runs in
    a worker thread.
    """
    if DuetWebAPI is None:
        raise ImportError("Missing duetwebapi. Install with: pip install duetwebapi==1.1.0")
    manual_file = args.group_displacements_file or DEFAULT_MANUAL_FILE
    if not Path(manual_file).exists():
        save_manual_offsets(manual_file, plan["manual_offsets"], plan["paths"])
    if ui_state is not None:
        ui_state.manual_file = str(manual_file)
        ui_state.plan_paths = list(plan["paths"])
        ui_state.total_groups = int(len(plan["paths"]))
        ui_state.set_mode("connecting", f"Connecting to Duet at {args.duet_web_address}")
        update_manual_window_state(ui_state, plan["manual_offsets"], 1, plan["paths"])

    rrf = DuetWebAPI(str(args.duet_web_address))
    print("Connection attempted. Requesting diagnostics with M122.")
    diag = rrf.send_code("M122")
    print(diag)
    print("Robot connected.")
    if ui_state is not None:
        ui_state.set_mode("connected", "Robot connected. Initializing printer state.")

    terminal_ctx = contextlib.nullcontext(None) if ui_state is not None else TerminalKeyboard()
    with terminal_ctx as term_kb:
        kb = HybridKeyboard(ui_state, term_kb)
        radius_feed_map = vm.parse_radius_feed_map(
            raw=str(args.radius_feed_map or ""),
            file_path=None if args.radius_feed_map_file is None else str(args.radius_feed_map_file),
        )
        feed_mode = str(args.extrusion_feed_mode).strip().lower()
        writer = InteractiveRobotWriter(
            vm, rrf, kb, manual_file, plan["manual_offsets"],
            float(args.node_tune_step_mm), float(args.travel_tune_step_mm),
            float(args.b_tune_step_deg), float(args.c_tune_step_deg),
            float(args.tune_feed), float(args.rotary_tune_feed),
            float(args.travel_segment_mm), int(args.travel_rewind_indices),
            float(args.inter_command_delay_s), float(args.min_move_wait_s),
            ui_state=ui_state,
            approach_b_move_feed=float(args.approach_b_move_feed),
            start_stop_b_move_feed=float(args.start_stop_b_move_feed),
            start_stop_slow_samples=int(args.start_stop_slow_samples),
            cal=plan["cal"], bbox=plan["bbox"],
            travel_feed=float(args.travel_feed), approach_feed=float(args.approach_feed),
            fine_approach_feed=float(args.fine_approach_feed), print_feed=float(args.print_feed),
            c_feed=float(args.c_feed), extrusion_per_mm=float(args.extrusion_per_mm),
            pressure_offset_mm=float(args.pressure_offset_mm),
            pressure_advance_feed=float(args.pressure_advance_feed),
            pressure_retract_feed=float(args.pressure_retract_feed),
            preflow_dwell_ms=int(args.preflow_dwell_ms),
            start_node_dwell_ms=int(args.start_node_dwell_ms),
            node_dwell_ms=int(args.node_dwell_ms),
            edge_samples=int(args.edge_samples), emit_extrusion=(False if bool(args.no_extrusion) else float(args.extrusion_per_mm) != 0.0),
            write_mode=str(args.write_mode), orientation_mode=str(args.orientation_mode),
            bc_solve_samples=int(args.bc_solve_samples),
            b_max_step_deg=float(args.b_max_step_deg), c_max_step_deg=float(args.c_max_step_deg),
            b_smoothing_alpha=float(args.b_smoothing_alpha), c_smoothing_alpha=float(args.c_smoothing_alpha),
            travel_c_max_step_deg=None if args.travel_c_max_step_deg is None else float(args.travel_c_max_step_deg),
            travel_c_smoothing_alpha=None if args.travel_c_smoothing_alpha is None else float(args.travel_c_smoothing_alpha),
            c_min_deg=float(args.c_min_deg), c_max_deg=float(args.c_max_deg),
            min_tangent_xy_for_c=float(args.min_tangent_xy_for_c),
            travel_clearance_above_printed_z=float(args.travel_clearance_above_printed_z),
            travel_bbox_margin=float(args.travel_bbox_margin),
            travel_edge_clearance=float(args.travel_edge_clearance),
            enable_travel_bbox_clearance=bool(args.enable_travel_bbox_clearance),
            fine_approach_distance=float(args.fine_approach_distance),
            robot_skeleton_ref=plan["robot_skeleton_ref"],
            skeleton_collision_clearance=float(args.skeleton_collision_clearance),
            skeleton_collision_sample_step_mm=float(args.skeleton_collision_sample_step_mm),
            travel_accel_mm_s2=float(args.travel_accel_mm_s2),
            post_print_travel_accel_scale=float(args.post_print_travel_accel_scale),
            enable_aorta_cylinder_travel=bool(args.enable_aorta_cylinder_travel),
            aorta_cylinder_clearance_mm=float(args.aorta_cylinder_clearance_mm),
            aorta_cylinder_min_radius_mm=float(args.aorta_cylinder_min_radius_mm),
            aorta_cylinder_samples=int(args.aorta_cylinder_samples),
        )

        group_displacements_vm = {k: v.to_vm(vm) for k, v in plan["manual_offsets"].items()}
        planned_vasculature_paths: List[Any] = []
        for group_idx, path in enumerate(plan["paths"], start=1):
            displacement = group_displacements_vm.get(group_idx)
            cumulative_offset = vm.cumulative_group_node_offset_mm(
                group_idx,
                group_displacements_vm,
                plan.get("group_parent_map"),
            )
            planned_points = vm.prepend_group_node_gantry_offset(
                points=np.asarray(path.points, dtype=float),
                displacement=displacement,
                point_merge_tol=float(args.point_merge_tol),
                cumulative_offset=cumulative_offset,
            )
            if len(planned_points) < 2:
                continue
            path_radius_mm = max(0.0, float(path.radius_like) * float(plan["geometry_scale"]))
            planned_vasculature_paths.append(
                vm.PrintedVasculaturePath(
                    points_xyz_mm=np.asarray(planned_points, dtype=float).copy(),
                    radius_mm=path_radius_mm,
                )
            )
        writer.set_planned_vasculature_paths(planned_vasculature_paths)

        try:
            writer.send_code("G90", reason="absolute mode")
            if not bool(args.no_extrusion):
                writer.send_code("M42 P0 S0", reason="pressure off")
            if float(args.travel_accel_mm_s2) > 0.0:
                writer.set_travel_acceleration(float(args.travel_accel_mm_s2), comment="startup: nominal travel acceleration")

            msx, msy, msz, msb, msc = [float(v) for v in (args.machine_start_x, args.machine_start_y, args.machine_start_z, args.machine_start_b, args.machine_start_c)]
            if str(args.write_mode).strip().lower() == "calibrated":
                writer.send_absolute_move_axes({writer.base.cal.c_axis: float(msc)}, float(args.c_feed), reason="startup C sync before Z move")
                writer.base.cur_c = float(msc)
                writer.base.c_min_used = min(writer.base.c_min_used, float(msc))
                writer.base.c_max_used = max(writer.base.c_max_used, float(msc))
            writer.write_move(np.array([msx, msy, msz], dtype=float), msb, msc, float(args.travel_feed), comment="startup: move to machine start pose")

            total_paths = int(len(plan["paths"]))
            for group_idx, path in enumerate(plan["paths"], start=1):
                writer.current_group_number = group_idx
                writer.current_path_id = str(path.path_id)
                offset = plan["manual_offsets"].get(group_idx, ManualGroupOffset(group_number=group_idx))

                raw_points = np.asarray(path.points, dtype=float)
                if len(raw_points) < 2:
                    continue
                if ui_state is not None:
                    ui_state.set_group(group_idx, total_paths, str(path.path_id), getattr(path, "source_vessel_ids", []), raw_points)
                    update_manual_window_state(ui_state, plan["manual_offsets"], group_idx, plan["paths"])
                base_tangents = vm.build_tangents_for_points(raw_points, smooth_window=int(args.tangent_smooth_window), centerline_smooth_window=int(args.centerline_smooth_window))
                group_displacements_vm = {k: v.to_vm(vm) for k, v in plan["manual_offsets"].items()}
                cumulative_offset = vm.cumulative_group_node_offset_mm(
                    group_idx,
                    group_displacements_vm,
                    plan.get("group_parent_map"),
                )
                current_node_offset = np.zeros(3, dtype=float)
                current_disp = group_displacements_vm.get(group_idx)
                if current_disp is not None:
                    current_node_offset = np.asarray(current_disp.node_gantry_offset_mm, dtype=float).reshape(3)
                inherited_node_offset = np.asarray(cumulative_offset, dtype=float).reshape(3) - current_node_offset
                base_start_tip = np.asarray(raw_points[0], dtype=float).copy()
                base_start_tip = base_start_tip + inherited_node_offset
                points = vm.prepend_group_node_gantry_offset(
                    points=raw_points,
                    displacement=group_displacements_vm.get(group_idx),
                    point_merge_tol=float(args.point_merge_tol),
                    cumulative_offset=cumulative_offset,
                )
                if len(points) < 2:
                    continue
                tangents = vm.build_tangents_for_points(points, smooth_window=int(args.tangent_smooth_window), centerline_smooth_window=int(args.centerline_smooth_window))
                mult = float(args.extrusion_multiplier_main) if bool(path.is_main) else float(args.extrusion_multiplier_branch)
                path_radius_mm = max(0.0, float(path.radius_like) * float(plan["geometry_scale"]))
                active_path_print_feed = (
                    vm.feed_from_radius_map(path_radius_mm, radius_feed_map, float(args.print_feed))
                    if feed_mode == "weighted"
                    else float(args.print_feed)
                )
                if group_idx == 1:
                    writer.set_aorta_reference(points)

                print("\n" + "-" * 72)
                print(f"Printing group {group_idx}: {path.path_id} source_vessels={path.source_vessel_ids}")
                print("-" * 72)
                if ui_state is not None:
                    ui_state.set_mode("approach", f"Approaching group {group_idx}: {path.path_id}")

                if (
                    str(args.orientation_mode).strip().lower() in {"90mode", "90", "ninety", "tangent"}
                    and group_idx > 1
                    and bool(args.enable_aorta_cylinder_travel)
                ):
                    writer.approach_start_from_aorta_cylinder(points[0], tangents[0], str(path.path_id), final_approach_feed=active_path_print_feed)
                elif bool(args.enable_side_approach):
                    writer.approach_start_from_side(points[0], tangents[0], float(args.side_approach_far), float(args.side_approach_near), float(args.side_retreat), float(args.side_lift_z), str(path.path_id), final_approach_feed=active_path_print_feed)
                else:
                    writer.approach_start_direct(points[0], tangents[0], str(path.path_id), final_approach_feed=active_path_print_feed)
                writer.apply_node_rotary_offset(offset, points[0], tangents[0], feed=float(args.fine_approach_feed))

                tuned_offset = writer.tune_node_start(group_idx, str(path.path_id), base_start_tip, base_tangents[0], offset)
                plan["manual_offsets"][group_idx] = tuned_offset
                normalize_manual_offset_feeds(plan["manual_offsets"], default_feed=float(args.travel_feed))
                update_manual_window_state(ui_state, plan["manual_offsets"], group_idx, plan["paths"])

                group_displacements_vm = {k: v.to_vm(vm) for k, v in plan["manual_offsets"].items()}
                cumulative_offset = vm.cumulative_group_node_offset_mm(
                    group_idx,
                    group_displacements_vm,
                    plan.get("group_parent_map"),
                )
                points = vm.prepend_group_node_gantry_offset(
                    points=raw_points,
                    displacement=group_displacements_vm.get(group_idx),
                    point_merge_tol=float(args.point_merge_tol),
                    cumulative_offset=cumulative_offset,
                )
                if len(points) < 2:
                    continue
                tangents = vm.build_tangents_for_points(points, smooth_window=int(args.tangent_smooth_window), centerline_smooth_window=int(args.centerline_smooth_window))

                if writer.consume_node_start_accepted(group_idx) or current_tip_matches_start(writer, points[0]):
                    if ui_state is not None:
                        ui_state.set_mode("ready_to_print", f"At node start for group {group_idx}; starting print.")
                else:
                    writer._move_to_tip_with_explicit_bc(points[0], writer.base.cur_b, writer.base.cur_c, float(args.fine_approach_feed), reason="node start resync")

                writer.print_polyline(
                    points,
                    tangents,
                    extrusion_multiplier=mult,
                    label=str(path.path_id),
                    path_radius_mm=path_radius_mm,
                    print_feed_override=active_path_print_feed,
                    radius_profile_mm=np.asarray(path.radius_profile, dtype=float) * float(plan["geometry_scale"]),
                )
                if group_idx < total_paths:
                    tuned_offset = writer.tune_group_travel(group_idx, str(path.path_id), plan["manual_offsets"][group_idx])
                    plan["manual_offsets"][group_idx] = tuned_offset
                    normalize_manual_offset_feeds(plan["manual_offsets"], default_feed=float(args.travel_feed))
                update_manual_window_state(ui_state, plan["manual_offsets"], group_idx + 1 if group_idx < total_paths else group_idx, plan["paths"])

            if writer.base.cur_stage_xyz is None:
                raise RuntimeError("No machine pose established; cannot shut down.")
            if ui_state is not None:
                ui_state.set_mode("shutdown", "Moving to shutdown pose.")
            shutdown_stage = np.asarray(writer.base.cur_stage_xyz, dtype=float).copy()
            shutdown_stage[2] = float(args.machine_end_z)
            writer.write_move(shutdown_stage, float(writer.base.cur_b), float(writer.base.cur_c), float(args.travel_feed), comment=f"shutdown: lift to Z={float(args.machine_end_z):.3f}")
            if not bool(args.no_extrusion):
                writer.send_code("M42 P0 S0", reason="pressure off")

            print("\nInteractive print complete.")
            print(f"Taught offsets saved to {manual_file}")
            print(f"Command ranges: {writer.base.command_ranges()}")
            if ui_state is not None:
                ui_state.set_done(f"Interactive print complete. Taught offsets saved to {manual_file}.")
        except KeyboardInterrupt as exc:
            if ui_state is not None:
                ui_state.set_error(f"Stopped: {exc}")
            raise
        except Exception as exc:
            if ui_state is not None:
                ui_state.set_error(f"Error: {exc}")
            raise


def run_interactive_print_gui(vm: Any, args: argparse.Namespace, plan: Dict[str, Any]) -> None:
    manual_file = args.group_displacements_file or DEFAULT_MANUAL_FILE
    state = SharedPrintState(manual_file=str(manual_file))
    state.plan_paths = list(plan["paths"])
    update_manual_window_state(state, plan["manual_offsets"], 1, plan["paths"])
    app = PrintGUIApp(
        vm=vm,
        args=args,
        state=state,
        plan=plan,
        manual_file=str(manual_file),
        refresh_ms=int(args.gui_refresh_ms),
        plot_refresh_ms=int(args.gui_plot_refresh_ms),
    )

    def worker() -> None:
        try:
            run_interactive_print(vm, args, plan, ui_state=state)
        except KeyboardInterrupt:
            # State has already been updated. Keep GUI open for inspection.
            pass
        except Exception as exc:
            state.set_error(str(exc))

    t = threading.Thread(target=worker, name="vasculature-print-worker", daemon=True)
    t.start()
    app.run()



# ------------------------- CLI -------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    vm = import_vasc_manual_module()
    ap = vm.build_arg_parser()
    for action in ap._actions:
        if "--vessels" in getattr(action, "option_strings", []):
            action.required = False
        elif "--calibration" in getattr(action, "option_strings", []):
            action.required = False
        elif "--robot-skeleton-file" in getattr(action, "option_strings", []):
            action.required = False
        elif "--out" in getattr(action, "option_strings", []):
            action.required = False
        elif "--groups-out" in getattr(action, "option_strings", []):
            action.required = False
        elif "--group-displacements-file" in getattr(action, "option_strings", []):
            action.required = False
    ap.set_defaults(
        vessels=DEFAULT_VESSELS_FILE,
        calibration=DEFAULT_CALIBRATION_FILE,
        robot_skeleton_file=DEFAULT_ROBOT_SKELETON_FILE,
        out=DEFAULT_OUT_FILE,
        groups_out=DEFAULT_GROUPS_OUT_FILE,
        group_displacements_file=DEFAULT_GROUP_DISPLACEMENTS_FILE,
        write_mode="calibrated",
        orientation_mode="tangent",
        rotate_y_deg=180.0,
        align_lowest_centroid=[100.0, 100.0, -160.0],
        simplify_endpoint_chains=True,
        branch_start_overlap_mm=1.5,
        travel_clearance_above_printed_z=10.0,
        fine_approach_distance=3.0,
        geometry_scale=1.0,
        tangent_smooth_window=16,
        centerline_smooth_window=4,
        path_resample_spacing=0.8,
        b_max_step_deg=3.0,
        c_max_step_deg=3.0,
        b_smoothing_alpha=0.15,
        c_smoothing_alpha=0.15,
        min_tangent_xy_for_c=0.30,
        min_group_length_mm=8.0,
        enable_travel_bbox_clearance=False,
        enable_side_approach=False,
        offplane_fit_mode="avg_cubic",
        travel_feed=DEFAULT_TRAVEL_FEED_OVERRIDE,
        print_feed=DEFAULT_PRINT_FEED_OVERRIDE,
    )
    ap.description = "Interactive physical runner for gcode_generation/vasc_manual_90.py, with optional save-gcode mode."
    ap.add_argument("--run-mode", choices=["print", "save-gcode"], default="print", help="print sends commands to the robot; save-gcode writes --out only.")
    ap.add_argument("--no-extrusion", action="store_true", default=False, help="Do not actuate pressure/extrusion commands during --run-mode print; in save-gcode mode, sets effective extrusion_per_mm to 0.")
    ap.add_argument("--duet-web-address", default=DEFAULT_DUET_WEB_ADDRESS, help="Duet Web Control address used in --run-mode print.")
    ap.add_argument("--node-tune-step-mm", type=float, default=DEFAULT_NODE_TUNE_STEP_MM)
    ap.add_argument("--travel-tune-step-mm", type=float, default=DEFAULT_TRAVEL_TUNE_STEP_MM)
    ap.add_argument("--b-tune-step-deg", type=float, default=DEFAULT_B_TUNE_STEP_DEG)
    ap.add_argument("--c-tune-step-deg", type=float, default=DEFAULT_C_TUNE_STEP_DEG, help="C tuning step. Default 1 deg.")
    ap.add_argument("--tune-feed", type=float, default=DEFAULT_TUNE_FEED)
    ap.add_argument("--rotary-tune-feed", type=float, default=DEFAULT_ROTARY_TUNE_FEED, help="Higher feedrate used for B/C manual tuning moves.")
    ap.add_argument("--approach-b-move-feed", type=float, default=DEFAULT_APPROACH_B_MOVE_FEED, help="Maximum standalone B-axis feed, in deg/min, during approach/reorientation moves before printing.")
    ap.add_argument("--start-stop-b-move-feed", type=float, default=DEFAULT_START_STOP_B_MOVE_FEED, help="Maximum B-axis component speed, in deg/min, for the first/last print samples of each branch.")
    ap.add_argument("--start-stop-slow-samples", type=int, default=DEFAULT_START_STOP_SLOW_SAMPLES, help="How many print samples at the start and end of each branch use the slower B-axis cap.")
    ap.add_argument("--travel-segment-mm", type=float, default=DEFAULT_TRAVEL_SEGMENT_MM, help="Maximum stage-space segment length for interruptible travel moves.")
    ap.add_argument("--travel-rewind-indices", type=int, default=DEFAULT_TRAVEL_REWIND_INDICES, help="How many internal travel segments to move back after R.")
    ap.add_argument("--print-rewind-samples", type=int, default=DEFAULT_PRINT_REWIND_SAMPLES, help="How many print samples to rewind when the print-rewind toggle is armed.")
    ap.add_argument("--inter-command-delay-s", type=float, default=DEFAULT_INTER_COMMAND_DELAY_S)
    ap.add_argument("--min-move-wait-s", type=float, default=DEFAULT_MIN_MOVE_WAIT_S)
    ap.add_argument("--gui", dest="gui", action="store_true", default=True, help="Show the live Tk GUI in --run-mode print. Default: enabled.")
    ap.add_argument("--no-gui", dest="gui", action="store_false", help="Use terminal-only keyboard interaction instead of the GUI.")
    ap.add_argument("--gui-refresh-ms", type=int, default=DEFAULT_GUI_REFRESH_MS, help="GUI status refresh interval in milliseconds.")
    ap.add_argument("--gui-plot-refresh-ms", type=int, default=DEFAULT_GUI_PLOT_REFRESH_MS, help="3D plot refresh interval in milliseconds.")
    return ap


def main() -> None:
    # Import once here so this script can live outside the package but still use
    # the current local vasc_manual_90.py implementation.
    vm = import_vasc_manual_module()
    ap = build_arg_parser()
    args = ap.parse_args()

    if args.group_displacements_file is None:
        args.group_displacements_file = DEFAULT_MANUAL_FILE

    plan = build_plan_from_args(vm, args)
    if args.groups_out:
        vm.export_print_groups(str(args.groups_out), plan["paths"])
        print(f"Wrote print groups to {args.groups_out}")

    # Keep the manual file in sync before doing anything risky.
    save_manual_offsets(str(args.group_displacements_file), plan["manual_offsets"], plan["paths"])

    summary = vm.summarize_vessels(plan["vessels"])
    path_summary = vm.summarize_paths(plan["paths"])
    print(f"Parsed {summary['vessel_count']} vessels into {path_summary['path_count']} printable paths.")
    print(f"Manual offset file: {args.group_displacements_file}")

    if args.run_mode == "save-gcode":
        save_gcode(vm, args, plan)
    else:
        if bool(args.gui):
            if tk is None or Figure is None or FigureCanvasTkAgg is None:
                raise RuntimeError("GUI mode requires tkinter and matplotlib TkAgg. Re-run with --no-gui for terminal-only mode.")
            run_interactive_print_gui(vm, args, plan)
        else:
            run_interactive_print(vm, args, plan)


if __name__ == "__main__":
    main()
