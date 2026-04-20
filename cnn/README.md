# CTR tip refinement package

This package wraps `shadow_calibration.py` / `CTR_Shadow_Calibration` and adds three stages:

1. process an image folder and save patch crops around the auto-detected tip
2. review/correct those patches in a simple GUI
3. train a small heatmap CNN for local tip refinement

## Files

- `common_ctr_tip_refinement.py` — shared loader and patch utilities
- `prepare_tip_refinement_dataset.py` — runs the current classical pipeline on a folder and writes `manifest.csv`
- `annotate_tip_refinement_gui.py` — patch review / correction GUI
- `transfer_tip_annotations.py` — transfers existing labels into a regenerated manifest
- `train_tip_refiner.py` — trains a small U-Net style heatmap model
- `infer_tip_refiner.py` — optional batch inference on an existing manifest

## Requirements

```bash
pip install opencv-python numpy pandas torch matplotlib scikit-image
```

Run the commands from the repository root. The scripts default to `./shadow_calibration.py` and use the same project layout as `offline_run_calibration.py`:

- `<project>/raw_image_data_folder`
- `<project>/processed_image_data_folder`
- `<project>/analysis_reference.json` when present

If `analysis_reference.json` exists, dataset preparation reuses its `analysis_crop`. Pass `--crop xmin,xmax,ymin,ymax` to override it.

Dataset preparation scans `raw_image_data_folder` recursively. If two subfolders contain files with the same basename, patches are still written with unique names based on the relative path, and the manifest includes `image_relative_path`.

To pick a separate crop for each subfolder, add `--crop_gui_per_folder`. The GUI opens on the first image in each relative subfolder before processing that folder.

## Step 1: prepare a patch dataset

```bash
python3 cnn/prepare_tip_refinement_dataset.py \
  --project_dir "Test_Calibration_2026-04-03_00_off_tuned" \
  --threshold 200 \
  --crop_gui_per_folder \
  --anchor coarse \
  --patch-size 128
```

This writes by default to:

- `Test_Calibration_2026-04-03_00_off_tuned/processed_image_data_folder/tip_refinement_dataset/manifest.csv`
- `Test_Calibration_2026-04-03_00_off_tuned/processed_image_data_folder/tip_refinement_dataset/patches/*.png`
- `Test_Calibration_2026-04-03_00_off_tuned/processed_image_data_folder/tip_refinement_dataset/prepare_summary.json`

To mirror an `offline_run_calibration.py` invocation with camera/board reference arguments:

```bash
python3 cnn/prepare_tip_refinement_dataset.py \
  --project_dir "Test_Calibration_2026-04-03_00_off_tuned" \
  --threshold 200 \
  --anchor coarse \
  --patch-size 128 \
  --save_analysis_config \
  --camera_calibration_file "../captures/calibration_webcam_20260401_163844.npz" \
  --board_reference_image "../captures/photo_20260401_163839.png"
```

Other useful options:

- `--raw_dir /path/to/images` creates or fills a calibration project like `offline_run_calibration.py`.
- Nested folders under `--raw_dir` are preserved inside `raw_image_data_folder`.
- `--link_mode copy` copies images instead of symlinking when using `--raw_dir`.
- `--crop_gui_per_folder` opens the crop GUI once per relative image subfolder.
- `--threshold -1` uses Otsu thresholding.
- `--anchor selected` centers patches on the selected output from the current local refiner.
- `--output-dir /path/to/dataset` overrides the default project output folder.

## Step 2: correct the detections

```bash
python3 cnn/annotate_tip_refinement_gui.py \
  --project_dir "Test_Calibration_2026-04-03_00_off_tuned"
```

GUI controls:

- left click: place a manual point
- `m`: commit the clicked point
- `space`: accept the current anchor point
- `c`: accept the coarse point
- `s`: accept the selected point
- `r`: accept the refined point
- `u`: clear label / set back to pending
- `x`: mark sample as bad
- `n`: save and go next
- `p`: save and go previous
- `q`: save and quit

This writes `manifest_annotated.csv` next to `manifest.csv` by default.

## Optional: transfer old annotations after regenerating patches

If you add a new raw-image subfolder or regenerate patches with a different anchor/patch size, keep a backup of the old annotated manifest first:

```bash
cp \
  "Test_Calibration_2026-04-03_00_off_tuned/processed_image_data_folder/tip_refinement_dataset/manifest_annotated.csv" \
  "Test_Calibration_2026-04-03_00_off_tuned/processed_image_data_folder/tip_refinement_dataset/manifest_annotated_before_regen.csv"
```

Regenerate `manifest.csv` with `prepare_tip_refinement_dataset.py`, then dry-run the transfer:

```bash
python3 cnn/transfer_tip_annotations.py \
  --project_dir "Test_Calibration_2026-04-03_00_off_tuned" \
  --old-manifest "Test_Calibration_2026-04-03_00_off_tuned/processed_image_data_folder/tip_refinement_dataset/manifest_annotated_before_regen.csv" \
  --dry-run
```

If the counts look right, write the transferred annotations:

```bash
python3 cnn/transfer_tip_annotations.py \
  --project_dir "Test_Calibration_2026-04-03_00_off_tuned" \
  --old-manifest "Test_Calibration_2026-04-03_00_off_tuned/processed_image_data_folder/tip_refinement_dataset/manifest_annotated_before_regen.csv" \
  --overwrite
```

Rows that existed before keep their labels. New rows remain `pending`, so the GUI only needs annotation for the new data.

## Step 3: train the local refiner

```bash
python3 cnn/train_tip_refiner.py \
  --project_dir "Test_Calibration_2026-04-03_00_off_tuned" \
  --epochs 30 \
  --batch-size 16
```

Outputs:

- `processed_image_data_folder/tip_refinement_model/best_tip_refiner.pt`
- `processed_image_data_folder/tip_refinement_model/training_history.csv`
- `processed_image_data_folder/tip_refinement_model/training_config.json`
- `processed_image_data_folder/tip_refinement_model/training_summary.json`

## Optional: run inference on the manifest patches

```bash
python3 cnn/infer_tip_refiner.py \
  --project_dir "Test_Calibration_2026-04-03_00_off_tuned"
```

This writes `manifest_with_model_preds.csv` next to the dataset manifest.

## Suggested workflow

Start with `--anchor coarse` if you want the CNN to learn the full local correction from the geodesic tip.

If that correction is larger than you want, switch to `--anchor selected` so the model only learns the last few pixels of refinement from your current parallel-centerline step.
