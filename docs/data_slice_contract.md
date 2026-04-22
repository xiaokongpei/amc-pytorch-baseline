# Data Slice Contract

The active PyTorch baseline consumes processed `.pt` bundle files.

## Active Supported Format

- `.pt`

## Required Layout

```text
data/processed/
  train.pt
  validation.pt
  test.pt
  metadata/
    classes-fixed.json
    slice_manifest.json
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

- `data/processed/metadata/classes-fixed.json` stores class names for reporting
- `data/processed/metadata/slice_manifest.json` stores source paths, split sizes, and slicing settings

## Index Files

The canonical train/test split index files live at:

- `data/train_indexes.csv`
- `data/test_indexes.csv`

Validation is derived from the train index set using `val_ratio`.

## Historical Note

Older shard-style `train/validation/test/` directory contracts are no longer the active default for this repository.
