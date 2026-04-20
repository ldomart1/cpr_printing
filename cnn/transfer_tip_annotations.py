from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

DEFAULT_DATASET_NAME = "tip_refinement_dataset"


def default_dataset_dir(project_dir: str) -> Path:
    return Path(project_dir).expanduser().resolve() / "processed_image_data_folder" / DEFAULT_DATASET_NAME


def read_manifest(path: str | Path) -> pd.DataFrame:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Manifest not found: {path}")
    return pd.read_csv(path)


def write_manifest(df: pd.DataFrame, path: str | Path, overwrite: bool = False) -> None:
    path = Path(path).expanduser().resolve()
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {path}. Pass --overwrite to replace it.")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def normalize_annotation_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    text_defaults = {
        "annotation_status": "pending",
        "annotation_source": "",
        "notes": "",
    }
    numeric_defaults = [
        "manual_label_x_patch",
        "manual_label_y_patch",
        "manual_label_x_abs",
        "manual_label_y_abs",
    ]

    for column, default in text_defaults.items():
        if column not in df.columns:
            df[column] = default
        values = df[column].astype("object")
        values = values.where(pd.notna(values), default)
        values = values.replace("nan", default)
        if column == "annotation_status":
            values = values.replace("", default)
        df[column] = values.astype("object")

    for column in numeric_defaults:
        if column not in df.columns:
            df[column] = np.nan
        df[column] = pd.to_numeric(df[column], errors="coerce")

    return df


def choose_match_column(old_df: pd.DataFrame, new_df: pd.DataFrame, requested: str = "auto") -> str:
    if requested != "auto":
        if requested not in old_df.columns or requested not in new_df.columns:
            raise ValueError(f"Requested match column {requested!r} is not present in both manifests.")
        return requested

    candidates = ["image_relative_path", "image_path", "image_file"]
    for column in candidates:
        if column in old_df.columns and column in new_df.columns:
            return column
    raise ValueError("Could not find a shared match column. Expected image_relative_path, image_path, or image_file.")


def first_valid_row_by_key(df: pd.DataFrame, key_column: str) -> dict[str, pd.Series]:
    result: dict[str, pd.Series] = {}
    for _, row in df.iterrows():
        key = str(row.get(key_column, ""))
        if not key or key == "nan":
            continue
        if key not in result:
            result[key] = row
    return result


def row_abs_label(row: pd.Series) -> tuple[float, float] | None:
    x_abs = pd.to_numeric(row.get("manual_label_x_abs", np.nan), errors="coerce")
    y_abs = pd.to_numeric(row.get("manual_label_y_abs", np.nan), errors="coerce")
    if np.isfinite(x_abs) and np.isfinite(y_abs):
        return float(x_abs), float(y_abs)

    x_patch = pd.to_numeric(row.get("manual_label_x_patch", np.nan), errors="coerce")
    y_patch = pd.to_numeric(row.get("manual_label_y_patch", np.nan), errors="coerce")
    patch_x0 = pd.to_numeric(row.get("patch_requested_x0", np.nan), errors="coerce")
    patch_y0 = pd.to_numeric(row.get("patch_requested_y0", np.nan), errors="coerce")
    if all(np.isfinite(v) for v in (x_patch, y_patch, patch_x0, patch_y0)):
        return float(patch_x0 + x_patch), float(patch_y0 + y_patch)

    return None


def patch_contains_label(row: pd.Series, x_patch: float, y_patch: float, margin_px: float = 0.0) -> bool:
    patch_size = pd.to_numeric(row.get("patch_size", np.nan), errors="coerce")
    if not np.isfinite(patch_size):
        return True
    return (
        -float(margin_px) <= float(x_patch) < float(patch_size) + float(margin_px)
        and -float(margin_px) <= float(y_patch) < float(patch_size) + float(margin_px)
    )


def transfer_annotations(
    old_df: pd.DataFrame,
    new_df: pd.DataFrame,
    match_column: str,
    transfer_bad: bool = True,
    allow_out_of_patch: bool = False,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    old_df = normalize_annotation_columns(old_df)
    out_df = normalize_annotation_columns(new_df)
    old_by_key = first_valid_row_by_key(old_df, match_column)

    stats: dict[str, Any] = {
        "match_column": match_column,
        "old_rows": int(len(old_df)),
        "new_rows": int(len(out_df)),
        "matched_rows": 0,
        "transferred_done": 0,
        "transferred_bad": 0,
        "left_pending_new_or_unmatched": 0,
        "skipped_not_labeled": 0,
        "skipped_missing_abs_label": 0,
        "skipped_out_of_patch": 0,
        "duplicate_old_keys": int(old_df[match_column].astype(str).duplicated().sum()),
    }
    skipped_examples: list[dict[str, str]] = []

    for idx, new_row in out_df.iterrows():
        key = str(new_row.get(match_column, ""))
        old_row = old_by_key.get(key)
        if old_row is None:
            stats["left_pending_new_or_unmatched"] += 1
            continue

        stats["matched_rows"] += 1
        status = str(old_row.get("annotation_status", "pending")).strip().lower()

        if status == "bad":
            if transfer_bad:
                out_df.at[idx, "annotation_status"] = "bad"
                out_df.at[idx, "annotation_source"] = "bad"
                out_df.at[idx, "manual_label_x_patch"] = np.nan
                out_df.at[idx, "manual_label_y_patch"] = np.nan
                out_df.at[idx, "manual_label_x_abs"] = np.nan
                out_df.at[idx, "manual_label_y_abs"] = np.nan
                out_df.at[idx, "notes"] = old_row.get("notes", "")
                stats["transferred_bad"] += 1
            else:
                stats["skipped_not_labeled"] += 1
            continue

        if status != "done":
            stats["skipped_not_labeled"] += 1
            continue

        abs_label = row_abs_label(old_row)
        if abs_label is None:
            stats["skipped_missing_abs_label"] += 1
            skipped_examples.append({"key": key, "reason": "missing_abs_label"})
            continue

        patch_x0 = pd.to_numeric(new_row.get("patch_requested_x0", np.nan), errors="coerce")
        patch_y0 = pd.to_numeric(new_row.get("patch_requested_y0", np.nan), errors="coerce")
        if not np.isfinite(patch_x0) or not np.isfinite(patch_y0):
            stats["skipped_missing_abs_label"] += 1
            skipped_examples.append({"key": key, "reason": "missing_new_patch_origin"})
            continue

        x_abs, y_abs = abs_label
        x_patch = float(x_abs - patch_x0)
        y_patch = float(y_abs - patch_y0)
        if not allow_out_of_patch and not patch_contains_label(new_row, x_patch, y_patch):
            stats["skipped_out_of_patch"] += 1
            skipped_examples.append({"key": key, "reason": "out_of_patch"})
            continue

        out_df.at[idx, "manual_label_x_abs"] = float(x_abs)
        out_df.at[idx, "manual_label_y_abs"] = float(y_abs)
        out_df.at[idx, "manual_label_x_patch"] = x_patch
        out_df.at[idx, "manual_label_y_patch"] = y_patch
        out_df.at[idx, "annotation_status"] = "done"
        out_df.at[idx, "annotation_source"] = old_row.get("annotation_source", "transferred") or "transferred"
        out_df.at[idx, "notes"] = old_row.get("notes", "")
        stats["transferred_done"] += 1

    stats["skipped_examples"] = skipped_examples[:20]
    return out_df, stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Transfer manual tip annotations from an old manifest to a regenerated manifest."
    )
    parser.add_argument("--project_dir", type=str, default=None, help="Calibration project folder for default new/output manifest paths.")
    parser.add_argument("--old-manifest", required=True, help="Existing annotated manifest to transfer from.")
    parser.add_argument(
        "--new-manifest",
        default=None,
        help="New manifest.csv to transfer into. Defaults to <project>/processed_image_data_folder/tip_refinement_dataset/manifest.csv.",
    )
    parser.add_argument(
        "--output-manifest",
        default=None,
        help="Output annotated manifest. Defaults to <new manifest folder>/manifest_annotated.csv.",
    )
    parser.add_argument(
        "--match-column",
        default="auto",
        help="Column used to match rows. Defaults to auto: image_relative_path, then image_path, then image_file.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output manifest.")
    parser.add_argument("--dry-run", action="store_true", help="Print transfer stats without writing the output manifest.")
    parser.add_argument("--no-transfer-bad", action="store_true", help="Do not transfer rows marked annotation_status=bad.")
    parser.add_argument("--allow-out-of-patch", action="store_true", help="Transfer labels even if they land outside the new patch.")
    args = parser.parse_args()

    if args.new_manifest is None and args.project_dir is None:
        raise SystemExit("Provide --project_dir or --new-manifest.")

    old_manifest = Path(args.old_manifest).expanduser().resolve()
    new_manifest = (
        Path(args.new_manifest).expanduser().resolve()
        if args.new_manifest
        else default_dataset_dir(args.project_dir) / "manifest.csv"
    )
    output_manifest = (
        Path(args.output_manifest).expanduser().resolve()
        if args.output_manifest
        else new_manifest.with_name("manifest_annotated.csv")
    )

    old_df = read_manifest(old_manifest)
    new_df = read_manifest(new_manifest)
    match_column = choose_match_column(old_df, new_df, requested=args.match_column)
    out_df, stats = transfer_annotations(
        old_df=old_df,
        new_df=new_df,
        match_column=match_column,
        transfer_bad=(not args.no_transfer_bad),
        allow_out_of_patch=bool(args.allow_out_of_patch),
    )

    print(json.dumps(stats, indent=2))
    if args.dry_run:
        print(f"Dry run only. Would write: {output_manifest}")
        return

    write_manifest(out_df, output_manifest, overwrite=bool(args.overwrite))
    print(f"Saved transferred annotations to: {output_manifest}")


if __name__ == "__main__":
    main()
