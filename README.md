# AMC PyTorch Baseline

Standalone PyTorch mainline for automatic modulation classification.

## Status

This repository is the only active execution baseline for the project.

- formal training entry: `scripts/train_fast.py`
- formal slice-building entry: `scripts/build_stratified_slices.py`
- compatibility slice-building entry: `scripts/build_fast_slices.py`
- formal config entry: `configs/fast.yaml`

The file names still contain `fast`, but within project governance they now mean the official baseline path rather than a temporary side path.

## Scope

This repository is the active PyTorch mainline for:

- baseline training
- post-train test evaluation
- SNR-sweep reporting
- checkpoint saving
- future pruning, quantization, and architecture experiments

## Data Contract

This project does not prepare raw datasets by default. It consumes already-prepared processed assets in the current fast-baseline format.

Default semantic locations for the current v2 stratified dataset:

- `data/processed_v2_stratified_64_16_20/`
- `data/processed_v2_stratified_64_16_20/metadata/`
- `data/splits_v2_stratified_64_16_20/`

Expected processed layout:

```text
data/
  splits_v2_stratified_64_16_20/
    train_indexes.csv
    validation_indexes.csv
    test_indexes.csv
  processed_v2_stratified_64_16_20/
    train.pt
    validation.pt
    test.pt
    metadata/
      classes-fixed.json
      slice_manifest.json
      split_summary.csv
      snr_class_distribution.csv
```

Each split `.pt` file stores:

- `observations`
- `labels`
- `snrs`

## Quick Start

### 1. Build v2 stratified `.pt` slices

```bash
python scripts/build_stratified_slices.py \
  --src-hdf5 <path-to-GOLD_XYZ_OSC.0001_1024.hdf5> \
  --class-names-path data/classes-fixed.json \
  --output-root data/processed_v2_stratified_64_16_20 \
  --split-root data/splits_v2_stratified_64_16_20 \
  --train-ratio 0.64 \
  --validation-ratio 0.16 \
  --test-ratio 0.20 \
  --clean-output
```

### 2. Train baseline

```bash
python scripts/train_fast.py \
  --config configs/fast.yaml \
  --data-root data/processed_v2_stratified_64_16_20
```

Training writes a new run under `runs/<run_name>/`.

### 3. Outputs

Typical run outputs include:

- `runs/<run_name>/config.json`
- `runs/<run_name>/train_log.json`
- `runs/<run_name>/metrics.json`
- `runs/<run_name>/snr_metrics.json`
- `runs/<run_name>/summary.md`
- `runs/<run_name>/checkpoints/best.pt`
- `runs/<run_name>/checkpoints/last.pt`

## Remote Use

For AutoDL or Kaggle:

- place processed data where `train_fast.py` can read `train.pt`, `validation.pt`, and `test.pt`
- keep each training run in a new `runs/<run_name>/`
- do not overwrite existing control runs

## Compatibility

`scripts/build_fast_slices.py` is retained only for reproducing older local runs that already depend on the previous index files. New baseline and ablation runs should use `scripts/build_stratified_slices.py` and the v2 stratified data root.
