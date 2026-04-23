# Data Slice Contract

The active PyTorch baseline consumes processed `.pt` bundle files.

## Active Supported Format

- `.pt`

## Required Layout

```text
data/processed_v2_stratified_64_16_20/
  train.pt
  validation.pt
  test.pt
  metadata/
    classes-fixed.json
    slice_manifest.json
    split_summary.csv
    snr_class_distribution.csv

data/splits_v2_stratified_64_16_20/
  train_indexes.csv
  validation_indexes.csv
  test_indexes.csv
```

## Bundle Keys

Each `.pt` split file must contain:

- `observations`
- `labels`
- `snrs`

## Tensor Shapes

- observations: `(N, 2, 1024)` or `(N, 1024, 2)`
- labels: `(N,)` integer labels
- snrs: `(N,)`

The dataset loader normalizes observations into channel-first `(N, 2, 1024)` semantics during loading.

## Metadata

- `data/processed_v2_stratified_64_16_20/metadata/classes-fixed.json` stores class names for reporting
- `data/processed_v2_stratified_64_16_20/metadata/slice_manifest.json` stores source paths, split sizes, and slicing settings
- `data/processed_v2_stratified_64_16_20/metadata/split_summary.csv` stores split-level counts
- `data/processed_v2_stratified_64_16_20/metadata/snr_class_distribution.csv` stores modulation x SNR counts per split

## Split Rule

The current formal split is stratified by `modulation x SNR`:

- train: `64%`
- validation: `16%`
- test: `20%`

Every modulation and SNR group is split independently before the final train, validation, and test sets are merged.

## Compatibility

The previous `data/processed/` location can still be used to reproduce older runs, but new experiments should use `data/processed_v2_stratified_64_16_20/`.
