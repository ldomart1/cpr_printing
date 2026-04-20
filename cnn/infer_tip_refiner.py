from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from common_ctr_tip_refinement import read_manifest, write_manifest
from train_tip_refiner import TinyUNet, decode_heatmap_argmax, decode_heatmap_softargmax

DEFAULT_DATASET_NAME = "tip_refinement_dataset"
DEFAULT_MODEL_NAME = "tip_refinement_model"


def default_dataset_dir(project_dir: str) -> Path:
    return Path(project_dir).expanduser().resolve() / "processed_image_data_folder" / DEFAULT_DATASET_NAME


def default_model_dir(project_dir: str) -> Path:
    return Path(project_dir).expanduser().resolve() / "processed_image_data_folder" / DEFAULT_MODEL_NAME


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a trained local tip refiner on a manifest of patch crops.")
    parser.add_argument("--project_dir", type=str, default=None, help="Calibration project folder. Used for default manifest/model paths.")
    parser.add_argument("--manifest", default=None, help="Manifest CSV with patch_path and patch geometry columns. Defaults to the project CNN manifest.")
    parser.add_argument("--model", default=None, help="Path to best_tip_refiner.pt. Defaults to the project CNN model.")
    parser.add_argument("--output-manifest", default=None, help="Path to write predictions CSV. Defaults next to the input manifest.")
    args = parser.parse_args()

    if (args.manifest is None or args.model is None or args.output_manifest is None) and args.project_dir is None:
        raise SystemExit("Provide --project_dir, or provide --manifest, --model, and --output-manifest.")

    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else default_dataset_dir(args.project_dir) / "manifest.csv"
    )
    model_path = (
        Path(args.model).expanduser().resolve()
        if args.model
        else default_model_dir(args.project_dir) / "best_tip_refiner.pt"
    )
    output_manifest = (
        Path(args.output_manifest).expanduser().resolve()
        if args.output_manifest
        else manifest_path.with_name("manifest_with_model_preds.csv")
    )

    ckpt = torch.load(model_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    softargmax_temperature = float(cfg.get("softargmax_temperature", 20.0))
    use_softargmax = "softargmax_temperature" in cfg
    model = TinyUNet()
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    df = read_manifest(str(manifest_path)).copy()
    pred_x_patch = []
    pred_y_patch = []
    pred_x_abs = []
    pred_y_abs = []

    with torch.no_grad():
        for _, row in df.iterrows():
            patch = cv2.imread(str(row["patch_path"]), cv2.IMREAD_GRAYSCALE)
            if patch is None:
                pred_x_patch.append(np.nan)
                pred_y_patch.append(np.nan)
                pred_x_abs.append(np.nan)
                pred_y_abs.append(np.nan)
                continue
            tensor = torch.from_numpy((patch.astype(np.float32) / 255.0)[None, None, :, :])
            logits = model(tensor)
            if use_softargmax:
                xy = decode_heatmap_softargmax(logits, temperature=softargmax_temperature)[0].cpu().numpy()
            else:
                xy = decode_heatmap_argmax(logits)[0].cpu().numpy()
            x_patch, y_patch = float(xy[0]), float(xy[1])
            pred_x_patch.append(x_patch)
            pred_y_patch.append(y_patch)
            pred_x_abs.append(float(row["patch_requested_x0"] + x_patch))
            pred_y_abs.append(float(row["patch_requested_y0"] + y_patch))

    df["model_label_x_patch"] = pred_x_patch
    df["model_label_y_patch"] = pred_y_patch
    df["model_label_x_abs"] = pred_x_abs
    df["model_label_y_abs"] = pred_y_abs
    write_manifest(df, str(output_manifest))
    print(f"Saved predictions to: {output_manifest}")


if __name__ == "__main__":
    main()
