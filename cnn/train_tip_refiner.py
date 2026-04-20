from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from common_ctr_tip_refinement import read_manifest

DEFAULT_DATASET_NAME = "tip_refinement_dataset"
DEFAULT_MODEL_NAME = "tip_refinement_model"


def default_dataset_dir(project_dir: str) -> Path:
    return Path(project_dir).expanduser().resolve() / "processed_image_data_folder" / DEFAULT_DATASET_NAME


def default_model_dir(project_dir: str) -> Path:
    return Path(project_dir).expanduser().resolve() / "processed_image_data_folder" / DEFAULT_MODEL_NAME


@dataclass
class TrainConfig:
    epochs: int = 30
    batch_size: int = 16
    learning_rate: float = 1e-3
    val_fraction: float = 0.15
    seed: int = 42
    heatmap_sigma_px: float = 2.5
    heatmap_loss_weight: float = 1.0
    coord_loss_weight: float = 0.05
    softargmax_temperature: float = 20.0
    use_done_only: bool = True
    num_workers: int = 0


class TipPatchDataset(Dataset):
    def __init__(self, df: pd.DataFrame, sigma_px: float = 2.5, augment: bool = False):
        self.df = df.reset_index(drop=True).copy()
        self.sigma_px = float(sigma_px)
        self.augment = bool(augment)

    def __len__(self) -> int:
        return len(self.df)

    def _make_heatmap(self, h: int, w: int, x: float, y: float) -> np.ndarray:
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        heat = np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2.0 * self.sigma_px ** 2))
        return heat.astype(np.float32)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        patch = cv2.imread(str(row["patch_path"]), cv2.IMREAD_GRAYSCALE)
        if patch is None:
            raise FileNotFoundError(f"Could not read patch: {row['patch_path']}")
        patch = patch.astype(np.float32) / 255.0

        x = float(row["manual_label_x_patch"])
        y = float(row["manual_label_y_patch"])

        if self.augment:
            # Mild photometric jitter.
            gain = np.random.uniform(0.9, 1.1)
            bias = np.random.uniform(-0.05, 0.05)
            patch = np.clip(patch * gain + bias, 0.0, 1.0)

            if np.random.rand() < 0.5:
                patch = np.fliplr(patch).copy()
                x = (patch.shape[1] - 1) - x
            if np.random.rand() < 0.5:
                patch = np.flipud(patch).copy()
                y = (patch.shape[0] - 1) - y

        heatmap = self._make_heatmap(patch.shape[0], patch.shape[1], x=x, y=y)

        patch_tensor = torch.from_numpy(patch[None, :, :])
        heatmap_tensor = torch.from_numpy(heatmap[None, :, :])
        target_xy = torch.tensor([x, y], dtype=torch.float32)
        return patch_tensor, heatmap_tensor, target_xy


class ProgressBar:
    def __init__(self, total: int, label: str, width: int = 28, enabled: bool = True) -> None:
        self.total = max(1, int(total))
        self.label = str(label)
        self.width = max(10, int(width))
        self.enabled = bool(enabled)
        self.start_time = time.time()
        self.last_len = 0

    def update(self, current: int, **metrics: float) -> None:
        if not self.enabled:
            return

        current = int(np.clip(current, 0, self.total))
        frac = current / self.total
        filled = int(round(frac * self.width))
        bar = "#" * filled + "-" * (self.width - filled)
        elapsed = max(0.0, time.time() - self.start_time)
        rate = current / elapsed if elapsed > 0 else 0.0
        eta = (self.total - current) / rate if rate > 0 else 0.0
        metric_text = " ".join(f"{key}={value:.4g}" for key, value in metrics.items())
        line = (
            f"\r{self.label} [{bar}] {current}/{self.total} "
            f"{100.0 * frac:5.1f}% eta={eta:5.1f}s"
        )
        if metric_text:
            line += f" | {metric_text}"

        padding = " " * max(0, self.last_len - len(line))
        sys.stdout.write(line + padding)
        sys.stdout.flush()
        self.last_len = len(line)

    def finish(self) -> None:
        if self.enabled:
            sys.stdout.write("\n")
            sys.stdout.flush()


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinyUNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1 = ConvBlock(1, 32)
        self.enc2 = ConvBlock(32, 64)
        self.enc3 = ConvBlock(64, 128)
        self.pool = nn.MaxPool2d(2)
        self.up2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec2 = ConvBlock(128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.dec1 = ConvBlock(64, 32)
        self.head = nn.Conv2d(32, 1, kernel_size=1)

    def forward(self, x):
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.enc3(self.pool(x2))
        y = self.up2(x3)
        if y.shape[-2:] != x2.shape[-2:]:
            y = F.interpolate(y, size=x2.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec2(torch.cat([y, x2], dim=1))
        y = self.up1(y)
        if y.shape[-2:] != x1.shape[-2:]:
            y = F.interpolate(y, size=x1.shape[-2:], mode="bilinear", align_corners=False)
        y = self.dec1(torch.cat([y, x1], dim=1))
        return self.head(y)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def decode_heatmap_argmax(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    b, _, h, w = probs.shape
    flat = probs.view(b, -1)
    idx = torch.argmax(flat, dim=1)
    y = (idx // w).float()
    x = (idx % w).float()
    return torch.stack([x, y], dim=1)


def decode_heatmap_softargmax(logits: torch.Tensor, temperature: float = 20.0) -> torch.Tensor:
    b, _, h, w = logits.shape
    flat = logits.view(b, -1) * float(temperature)
    probs = torch.softmax(flat, dim=1)
    ys = torch.arange(h, device=logits.device, dtype=logits.dtype).repeat_interleave(w)
    xs = torch.arange(w, device=logits.device, dtype=logits.dtype).repeat(h)
    x = torch.sum(probs * xs[None, :], dim=1)
    y = torch.sum(probs * ys[None, :], dim=1)
    return torch.stack([x, y], dim=1)


def tip_refiner_loss(
    logits: torch.Tensor,
    heatmaps: torch.Tensor,
    target_xy: torch.Tensor,
    heatmap_weight: float,
    coord_weight: float,
    softargmax_temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    heatmap_loss = F.binary_cross_entropy_with_logits(logits, heatmaps)
    pred_xy = decode_heatmap_softargmax(logits, temperature=softargmax_temperature)
    patch_scale = float(max(logits.shape[-2], logits.shape[-1], 1))
    coord_loss = F.smooth_l1_loss(pred_xy / patch_scale, target_xy / patch_scale)
    total_loss = float(heatmap_weight) * heatmap_loss + float(coord_weight) * coord_loss
    return total_loss, heatmap_loss, coord_loss, pred_xy


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, cfg: TrainConfig) -> Tuple[float, float, float, float]:
    model.eval()
    total_loss = 0.0
    total_heatmap_loss = 0.0
    total_coord_loss = 0.0
    total_dist = 0.0
    count = 0
    with torch.no_grad():
        for patches, heatmaps, target_xy in loader:
            patches = patches.to(device)
            heatmaps = heatmaps.to(device)
            target_xy = target_xy.to(device)
            logits = model(patches)
            loss, heatmap_loss, coord_loss, pred_xy = tip_refiner_loss(
                logits=logits,
                heatmaps=heatmaps,
                target_xy=target_xy,
                heatmap_weight=cfg.heatmap_loss_weight,
                coord_weight=cfg.coord_loss_weight,
                softargmax_temperature=cfg.softargmax_temperature,
            )
            dist = torch.norm(pred_xy - target_xy, dim=1).mean()
            total_loss += float(loss.item()) * patches.size(0)
            total_heatmap_loss += float(heatmap_loss.item()) * patches.size(0)
            total_coord_loss += float(coord_loss.item()) * patches.size(0)
            total_dist += float(dist.item()) * patches.size(0)
            count += patches.size(0)
    if count == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    return total_loss / count, total_dist / count, total_heatmap_loss / count, total_coord_loss / count


def split_dataframe(df: pd.DataFrame, val_fraction: float, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = np.arange(len(df))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)
    n_val = max(1, int(round(len(df) * val_fraction))) if len(df) > 1 else 0
    val_idx = idx[:n_val]
    train_idx = idx[n_val:]
    if len(train_idx) == 0:
        train_idx = val_idx
        val_idx = idx[:0]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a patch-based heatmap model for local tip refinement.")
    parser.add_argument("--project_dir", type=str, default=None, help="Calibration project folder. Used for default manifest/model paths.")
    parser.add_argument("--manifest", default=None, help="Annotated manifest CSV. Defaults to <project>/processed_image_data_folder/tip_refinement_dataset/manifest_annotated.csv.")
    parser.add_argument("--output-dir", default=None, help="Directory to save model artifacts. Defaults to <project>/processed_image_data_folder/tip_refinement_model.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--heatmap-sigma-px", type=float, default=2.5)
    parser.add_argument("--heatmap-loss-weight", type=float, default=1.0)
    parser.add_argument("--coord-loss-weight", type=float, default=0.05)
    parser.add_argument("--softargmax-temperature", type=float, default=20.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--no-progress", action="store_true", help="Disable the per-epoch terminal progress bar.")
    args = parser.parse_args()

    cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        val_fraction=args.val_fraction,
        seed=args.seed,
        heatmap_sigma_px=args.heatmap_sigma_px,
        heatmap_loss_weight=args.heatmap_loss_weight,
        coord_loss_weight=args.coord_loss_weight,
        softargmax_temperature=args.softargmax_temperature,
        num_workers=args.num_workers,
    )

    if (args.manifest is None or args.output_dir is None) and args.project_dir is None:
        raise SystemExit("Provide --project_dir, or provide both --manifest and --output-dir.")

    manifest_path = (
        Path(args.manifest).expanduser().resolve()
        if args.manifest
        else default_dataset_dir(args.project_dir) / "manifest_annotated.csv"
    )
    output_dir = (
        Path(args.output_dir).expanduser().resolve()
        if args.output_dir
        else default_model_dir(args.project_dir)
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(cfg.seed)
    df = read_manifest(str(manifest_path))
    keep = (
        df["annotation_status"].astype(str).eq("done")
        & np.isfinite(df["manual_label_x_patch"])
        & np.isfinite(df["manual_label_y_patch"])
    )
    df = df.loc[keep].reset_index(drop=True)
    if len(df) < 10:
        raise ValueError("Need at least 10 annotated samples with annotation_status=done to train a useful model.")

    train_df, val_df = split_dataframe(df, cfg.val_fraction, cfg.seed)

    train_ds = TipPatchDataset(train_df, sigma_px=cfg.heatmap_sigma_px, augment=True)
    val_ds = TipPatchDataset(val_df, sigma_px=cfg.heatmap_sigma_px, augment=False)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TinyUNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.learning_rate)

    history = []
    best_val_dist = float("inf")
    best_path = output_dir / "best_tip_refiner.pt"

    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_loss = 0.0
        total_heatmap_loss = 0.0
        total_coord_loss = 0.0
        total_dist = 0.0
        count = 0
        progress = ProgressBar(
            total=len(train_loader),
            label=f"epoch {epoch:03d}/{cfg.epochs:03d}",
            enabled=(not args.no_progress),
        )
        for batch_idx, (patches, heatmaps, target_xy) in enumerate(train_loader, start=1):
            patches = patches.to(device)
            heatmaps = heatmaps.to(device)
            target_xy = target_xy.to(device)

            optimizer.zero_grad(set_to_none=True)
            logits = model(patches)
            loss, heatmap_loss, coord_loss, pred_xy = tip_refiner_loss(
                logits=logits,
                heatmaps=heatmaps,
                target_xy=target_xy,
                heatmap_weight=cfg.heatmap_loss_weight,
                coord_weight=cfg.coord_loss_weight,
                softargmax_temperature=cfg.softargmax_temperature,
            )
            loss.backward()
            optimizer.step()

            dist = torch.norm(pred_xy - target_xy, dim=1).mean()
            total_loss += float(loss.item()) * patches.size(0)
            total_heatmap_loss += float(heatmap_loss.item()) * patches.size(0)
            total_coord_loss += float(coord_loss.item()) * patches.size(0)
            total_dist += float(dist.item()) * patches.size(0)
            count += patches.size(0)
            progress.update(
                batch_idx,
                loss=total_loss / max(count, 1),
                hm=total_heatmap_loss / max(count, 1),
                xy=total_coord_loss / max(count, 1),
                px=total_dist / max(count, 1),
            )

        progress.finish()

        train_loss = total_loss / max(count, 1)
        train_heatmap_loss = total_heatmap_loss / max(count, 1)
        train_coord_loss = total_coord_loss / max(count, 1)
        train_dist = total_dist / max(count, 1)
        val_loss, val_dist, val_heatmap_loss, val_coord_loss = evaluate(model, val_loader, device, cfg)

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_heatmap_loss": train_heatmap_loss,
            "train_coord_loss": train_coord_loss,
            "train_mean_px_error": train_dist,
            "val_loss": val_loss,
            "val_heatmap_loss": val_heatmap_loss,
            "val_coord_loss": val_coord_loss,
            "val_mean_px_error": val_dist,
        }
        history.append(row)
        print(
            f"epoch {epoch:03d} | "
            f"train_loss={train_loss:.5f} train_hm={train_heatmap_loss:.5f} "
            f"train_xy={train_coord_loss:.5f} train_px={train_dist:.3f} | "
            f"val_loss={val_loss:.5f} val_hm={val_heatmap_loss:.5f} "
            f"val_xy={val_coord_loss:.5f} val_px={val_dist:.3f}"
        )

        score = val_dist if np.isfinite(val_dist) else train_dist
        if score < best_val_dist:
            best_val_dist = score
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": asdict(cfg),
                    "patch_size": int(train_df.iloc[0]["patch_size"]),
                    "anchor_name": str(train_df.iloc[0]["anchor_name"]),
                },
                best_path,
            )

    pd.DataFrame(history).to_csv(output_dir / "training_history.csv", index=False)
    (output_dir / "training_config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")
    summary = {
        "num_total": int(len(df)),
        "num_train": int(len(train_df)),
        "num_val": int(len(val_df)),
        "best_val_mean_px_error": None if not np.isfinite(best_val_dist) else float(best_val_dist),
        "best_model_path": str(best_path),
        "device": str(device),
    }
    (output_dir / "training_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("\nTraining complete.")
    print(f"Best model: {best_path}")


if __name__ == "__main__":
    main()
