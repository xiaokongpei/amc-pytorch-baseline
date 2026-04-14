# AMC PyTorch Baseline

Standalone PyTorch baseline for automatic modulation classification.

## Scope

This repository is the new PyTorch mainline for:

- baseline training
- baseline evaluation
- SNR-sweep reporting
- future pruning, quantization, and architecture experiments

## Data

This project does not prepare datasets. It consumes already-prepared sliced assets.

Default semantic locations:

- `data/processed/`
- `data/processed/metadata/`

Expected split layout:

```text
data/processed/
  train/
    shard_000_observations.npy | .npz | .pt
    shard_000_labels.npy | .npz | .pt
    shard_000_snrs.npy | .npz | .pt
    shard_001_observations.npy | .npz | .pt
    ...
  validation/
    ...
  test/
    ...
  metadata/
    classes-fixed.json
```

Recommended shard size follows the original Harper split strategy:

- about `2000` samples per shard
- train and validation may be derived from the original train split
- test keeps its own shard set

If `validation/` is missing, validation is derived from the training split by `val_ratio`.

You can also point the scripts to external data roots with CLI arguments.

## Quick Start

```bash
python scripts/train_baseline.py --config configs/baseline.yaml
python scripts/eval_baseline.py --config configs/baseline.yaml --checkpoint runs/<run_name>/checkpoints/best.pt
```
