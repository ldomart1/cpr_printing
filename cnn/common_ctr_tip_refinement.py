from __future__ import annotations

import argparse
import json
import math
import os
import re
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import pandas as pd

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}


@dataclass
class DetectionResult:
    image_path: str
    image_file: str
    image_width: int
    image_height: int
    crop_x_min_img: int
    crop_x_max_img: int
    crop_y_min_img: int
    crop_y_max_img: int
    coarse_x_abs: float
    coarse_y_abs: float
    selected_x_abs: float
    selected_y_abs: float
    refined_x_abs: float
    refined_y_abs: float
    tip_angle_deg: float
    coarse_x_crop: float
    coarse_y_crop: float
    selected_x_crop: float
    selected_y_crop: float
    refined_x_crop: float
    refined_y_crop: float
    anchor_name: str
    anchor_x_abs: float
    anchor_y_abs: float
    anchor_x_crop: float
    anchor_y_crop: float
    orientation: Optional[float] = None
    ss_pos: Optional[float] = None
    ntnl_pos: Optional[float] = None
    motion_phase_code: Optional[float] = None
    motion_phase: Optional[str] = None
    pass_idx: Optional[float] = None
    selected_tip_source: Optional[str] = None
    selected_tip_reason: Optional[str] = None
    status: str = "ok"
    error_message: Optional[str] = None


@dataclass
class PatchInfo:
    patch_path: str
    patch_size: int
    patch_requested_x0: int
    patch_requested_y0: int
    patch_requested_x1: int
    patch_requested_y1: int
    patch_pad_left: int
    patch_pad_top: int
    anchor_x_patch: float
    anchor_y_patch: float
    coarse_x_patch: float
    coarse_y_patch: float
    selected_x_patch: float
    selected_y_patch: float
    refined_x_patch: float
    refined_y_patch: float


def parse_crop_string(crop: Optional[str]) -> Optional[Dict[str, int]]:
    if crop is None:
        return None
    parts = [p.strip() for p in str(crop).split(",") if p.strip()]
    if len(parts) != 4:
        raise ValueError("Crop must be 'xmin,xmax,ymin,ymax' in the class convention.")
    xmin, xmax, ymin, ymax = map(int, parts)
    return {
        "crop_width_min": xmin,
        "crop_width_max": xmax,
        "crop_height_min": ymin,
        "crop_height_max": ymax,
    }


class CTRSourceAdapter:
    def __init__(
        self,
        ctr_source_path: str,
        workspace_dir: str,
        project_name: str = "tip_refinement_workspace",
        allow_existing: bool = True,
    ) -> None:
        self.ctr_source_path = str(Path(ctr_source_path).expanduser().resolve())
        self.workspace_dir = Path(workspace_dir).expanduser().resolve()
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.module = self._load_source_module(self.ctr_source_path)
        if not hasattr(self.module, "CTR_Shadow_Calibration"):
            raise AttributeError(
                f"Could not find CTR_Shadow_Calibration in {self.ctr_source_path}."
            )

        cls = self.module.CTR_Shadow_Calibration
        self.processor = cls(
            parent_directory=str(self.workspace_dir),
            project_name=project_name,
            allow_existing=allow_existing,
            add_date=False,
        )

        # Restore cwd to the caller's shell expectation after the class constructor changes it.
        os.chdir(str(self.workspace_dir))

        self.normalize_angle = getattr(self.module, "_normalize_tip_angle_deg")
        self.refine_tip_parallel_centerline = getattr(self.module, "refine_tip_parallel_centerline")
        self.select_tip_candidate = getattr(self.module, "_select_tip_candidate")

    @staticmethod
    def _load_source_module(source_path: str) -> types.ModuleType:
        path = Path(source_path)
        if not path.exists():
            raise FileNotFoundError(f"CTR source file not found: {path}")

        code = path.read_text(encoding="utf-8")
        module = types.ModuleType(path.stem)
        module.__file__ = str(path)
        exec(compile(code, str(path), "exec"), module.__dict__)
        return module

    def set_analysis_crop(self, crop: Optional[Dict[str, int]]) -> None:
        if crop is None:
            return
        self.processor.analysis_crop = dict(crop)

    def parse_capture_context(self, image_file_name: str) -> Dict[str, Any]:
        parser = getattr(self.processor, "_parse_capture_context_from_filename", None)
        if parser is None:
            return {}
        try:
            return dict(parser(image_file_name))
        except Exception:
            return {}

    def detect_tip(
        self,
        image_path: str,
        crop: Optional[Dict[str, int]] = None,
        threshold: int = 200,
        anchor_name: str = "coarse",
    ) -> DetectionResult:
        anchor_name = str(anchor_name).strip().lower()
        if anchor_name not in {"coarse", "selected", "refined"}:
            raise ValueError("anchor_name must be one of: coarse, selected, refined")

        self.set_analysis_crop(crop)
        image_path = str(Path(image_path).expanduser().resolve())
        image_file = Path(image_path).name
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        h_img, w_img = image.shape[:2]
        crop_cfg = dict(self.processor.analysis_crop)
        # If the class defaults are meant for 4K data but the current image is smaller,
        # fall back to the full frame unless the caller explicitly overrides the crop.
        if crop is None:
            implied_w = int(crop_cfg.get("crop_width_max", w_img)) - int(crop_cfg.get("crop_width_min", 0))
            implied_h = int(crop_cfg.get("crop_height_max", h_img)) - int(crop_cfg.get("crop_height_min", 0))
            if (
                int(crop_cfg.get("crop_width_max", w_img)) > w_img
                or int(crop_cfg.get("crop_height_max", h_img)) > h_img
                or implied_w < max(32, int(0.20 * w_img))
                or implied_h < max(32, int(0.20 * h_img))
            ):
                crop_cfg = {
                    "crop_width_min": 0,
                    "crop_width_max": int(w_img),
                    "crop_height_min": 0,
                    "crop_height_max": int(h_img),
                }

        crop_x_min_img = max(0, min(int(crop_cfg["crop_width_min"]), w_img - 1))
        crop_x_max_img = max(crop_x_min_img + 1, min(int(crop_cfg["crop_width_max"]), w_img))
        crop_y_min_img = max(0, min(h_img - int(crop_cfg["crop_height_max"]), h_img - 1))
        crop_y_max_img = max(crop_y_min_img + 1, min(h_img - int(crop_cfg["crop_height_min"]), h_img))

        cropped = image[crop_y_min_img:crop_y_max_img, crop_x_min_img:crop_x_max_img, :]
        grayscale = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        grayscale_eq = clahe.apply(grayscale)
        grayscale_blur = cv2.GaussianBlur(grayscale_eq, (3, 3), 0)

        if threshold is None or threshold < 0:
            _, binary_image = cv2.threshold(
                grayscale_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
        else:
            # Mirror the current script's behavior by defaulting to Otsu, but allow a fixed threshold.
            _, binary_image = cv2.threshold(
                grayscale_blur, int(threshold), 255, cv2.THRESH_BINARY
            )

        # The current class expects dark foreground = 0 and bright background = 255.
        if np.mean(binary_image) < 127:
            binary_image = cv2.bitwise_not(binary_image)

        tip_row, tip_col, tip_angle_deg, tip_debug = self.processor.find_ctr_tip_skeleton(
            binary_image,
            min_spur_len=5,
            return_tip_angle=True,
            return_debug=True,
        )
        tip_angle_deg = float(self.normalize_angle(tip_angle_deg))

        if not np.isfinite(tip_angle_deg):
            tip_angle_deg = 0.0

        try:
            yy_refined, xx_refined, tip_refine_dbg = self.refine_tip_parallel_centerline(
                grayscale=grayscale,
                binary_image=binary_image,
                tip_yx=(int(round(tip_row)), int(round(tip_col))),
                tip_angle_deg=tip_angle_deg,
                section_near_r=float(self.processor.tip_parallel_section_near_r),
                section_far_r=float(self.processor.tip_parallel_section_far_r),
                scan_half_r=float(self.processor.tip_parallel_scan_half_r),
                num_sections=int(self.processor.tip_parallel_num_sections),
                cross_step_px=float(self.processor.tip_parallel_cross_step_px),
                ray_step_px=float(self.processor.tip_parallel_ray_step_px),
                ray_max_len_r=float(self.processor.tip_parallel_ray_max_len_r),
            )
        except Exception as exc:
            yy_refined, xx_refined = float(tip_row), float(tip_col)
            tip_refine_dbg = {"fallback": f"refine_error:{exc}", "mode": "coarse"}
        yy_selected, xx_selected, tip_select_dbg = self.select_tip_candidate(
            coarse_tip_yx=(float(tip_row), float(tip_col)),
            refined_tip_yx=(float(yy_refined), float(xx_refined)),
            tip_dbg=tip_refine_dbg,
            mode=getattr(self.processor, "tip_refine_mode", "coarse"),
        )

        coarse_x_abs = float(tip_col + crop_x_min_img)
        coarse_y_abs = float(tip_row + crop_y_min_img)
        refined_x_abs = float(xx_refined + crop_x_min_img)
        refined_y_abs = float(yy_refined + crop_y_min_img)
        selected_x_abs = float(xx_selected + crop_x_min_img)
        selected_y_abs = float(yy_selected + crop_y_min_img)

        anchor_lookup = {
            "coarse": (coarse_x_abs, coarse_y_abs, float(tip_col), float(tip_row)),
            "refined": (refined_x_abs, refined_y_abs, float(xx_refined), float(yy_refined)),
            "selected": (selected_x_abs, selected_y_abs, float(xx_selected), float(yy_selected)),
        }
        anchor_x_abs, anchor_y_abs, anchor_x_crop, anchor_y_crop = anchor_lookup[anchor_name]

        ctx = self.parse_capture_context(image_file)
        return DetectionResult(
            image_path=image_path,
            image_file=image_file,
            image_width=w_img,
            image_height=h_img,
            crop_x_min_img=crop_x_min_img,
            crop_x_max_img=crop_x_max_img,
            crop_y_min_img=crop_y_min_img,
            crop_y_max_img=crop_y_max_img,
            coarse_x_abs=coarse_x_abs,
            coarse_y_abs=coarse_y_abs,
            selected_x_abs=selected_x_abs,
            selected_y_abs=selected_y_abs,
            refined_x_abs=refined_x_abs,
            refined_y_abs=refined_y_abs,
            tip_angle_deg=tip_angle_deg,
            coarse_x_crop=float(tip_col),
            coarse_y_crop=float(tip_row),
            selected_x_crop=float(xx_selected),
            selected_y_crop=float(yy_selected),
            refined_x_crop=float(xx_refined),
            refined_y_crop=float(yy_refined),
            anchor_name=anchor_name,
            anchor_x_abs=anchor_x_abs,
            anchor_y_abs=anchor_y_abs,
            anchor_x_crop=anchor_x_crop,
            anchor_y_crop=anchor_y_crop,
            orientation=_safe_float(ctx.get("orientation")),
            ss_pos=_safe_float(ctx.get("ss_pos")),
            ntnl_pos=_safe_float(ctx.get("ntnl_pos")),
            motion_phase_code=_safe_float(ctx.get("motion_phase_code")),
            motion_phase=ctx.get("motion_phase"),
            pass_idx=_safe_float(ctx.get("pass_idx")),
            selected_tip_source=tip_select_dbg.get("selected_tip_source") if isinstance(tip_select_dbg, dict) else None,
            selected_tip_reason=tip_select_dbg.get("selected_tip_reason") if isinstance(tip_select_dbg, dict) else None,
        )


def _safe_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def list_images(image_dir: str, recursive: bool = True) -> list[Path]:
    image_dir = Path(image_dir).expanduser().resolve()
    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")
    if not image_dir.is_dir():
        raise NotADirectoryError(f"Not a directory: {image_dir}")
    iterator = image_dir.rglob("*") if recursive else image_dir.iterdir()
    return sorted([p for p in iterator if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def safe_path_stem(path_text: str) -> str:
    stem = str(path_text).strip().replace("\\", "/")
    path = Path(stem)
    if path.suffix.lower() in IMAGE_EXTENSIONS:
        stem = path.with_suffix("").as_posix()
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem)
    stem = stem.strip("._")
    return stem or "image"


def extract_padded_patch(gray_image: np.ndarray, center_x: float, center_y: float, patch_size: int) -> tuple[np.ndarray, Dict[str, int]]:
    patch_size = int(patch_size)
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")

    half = patch_size // 2
    x0 = int(round(center_x)) - half
    y0 = int(round(center_y)) - half
    x1 = x0 + patch_size
    y1 = y0 + patch_size

    h, w = gray_image.shape[:2]
    src_x0 = max(0, x0)
    src_y0 = max(0, y0)
    src_x1 = min(w, x1)
    src_y1 = min(h, y1)

    dst_x0 = src_x0 - x0
    dst_y0 = src_y0 - y0
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    dst_y1 = dst_y0 + (src_y1 - src_y0)

    patch = np.full((patch_size, patch_size), 255, dtype=np.uint8)
    patch[dst_y0:dst_y1, dst_x0:dst_x1] = gray_image[src_y0:src_y1, src_x0:src_x1]

    meta = {
        "patch_requested_x0": int(x0),
        "patch_requested_y0": int(y0),
        "patch_requested_x1": int(x1),
        "patch_requested_y1": int(y1),
        "patch_pad_left": int(dst_x0),
        "patch_pad_top": int(dst_y0),
    }
    return patch, meta


def detection_to_patch_record(
    detection: DetectionResult,
    image_gray: np.ndarray,
    patch_dir: str,
    patch_size: int,
    manifest_prefix: str = "patch",
    patch_stem: str | None = None,
) -> Dict[str, Any]:
    patch_dir = Path(patch_dir)
    patch_dir.mkdir(parents=True, exist_ok=True)

    patch, meta = extract_padded_patch(
        gray_image=image_gray,
        center_x=detection.anchor_x_abs,
        center_y=detection.anchor_y_abs,
        patch_size=patch_size,
    )

    stem = safe_path_stem(patch_stem) if patch_stem is not None else safe_path_stem(detection.image_file)
    patch_file = patch_dir / f"{stem}_{manifest_prefix}_{int(patch_size)}.png"
    cv2.imwrite(str(patch_file), patch)

    x0 = meta["patch_requested_x0"]
    y0 = meta["patch_requested_y0"]

    patch_info = PatchInfo(
        patch_path=str(patch_file),
        patch_size=int(patch_size),
        patch_requested_x0=int(meta["patch_requested_x0"]),
        patch_requested_y0=int(meta["patch_requested_y0"]),
        patch_requested_x1=int(meta["patch_requested_x1"]),
        patch_requested_y1=int(meta["patch_requested_y1"]),
        patch_pad_left=int(meta["patch_pad_left"]),
        patch_pad_top=int(meta["patch_pad_top"]),
        anchor_x_patch=float(detection.anchor_x_abs - x0),
        anchor_y_patch=float(detection.anchor_y_abs - y0),
        coarse_x_patch=float(detection.coarse_x_abs - x0),
        coarse_y_patch=float(detection.coarse_y_abs - y0),
        selected_x_patch=float(detection.selected_x_abs - x0),
        selected_y_patch=float(detection.selected_y_abs - y0),
        refined_x_patch=float(detection.refined_x_abs - x0),
        refined_y_patch=float(detection.refined_y_abs - y0),
    )

    record = detection.__dict__.copy()
    record.update(patch_info.__dict__)
    record["auto_label_x_patch"] = patch_info.anchor_x_patch
    record["auto_label_y_patch"] = patch_info.anchor_y_patch
    record["manual_label_x_patch"] = np.nan
    record["manual_label_y_patch"] = np.nan
    record["manual_label_x_abs"] = np.nan
    record["manual_label_y_abs"] = np.nan
    record["annotation_status"] = "pending"
    record["annotation_source"] = ""
    record["notes"] = ""
    return record


def read_manifest(manifest_path: str) -> pd.DataFrame:
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return pd.read_csv(manifest_path)


def write_manifest(df: pd.DataFrame, manifest_path: str) -> None:
    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(manifest_path, index=False)


__all__ = [
    "CTRSourceAdapter",
    "DetectionResult",
    "PatchInfo",
    "parse_crop_string",
    "list_images",
    "safe_path_stem",
    "extract_padded_patch",
    "detection_to_patch_record",
    "read_manifest",
    "write_manifest",
]
