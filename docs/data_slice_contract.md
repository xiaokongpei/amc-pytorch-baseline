# Data Slice Contract

The PyTorch baseline consumes sliced data only.

## Supported formats

- `.npy`
- `.npz`
- `.pt`

## Supported split layout

```text
data/processed/
  train/
  validation/
  test/
  metadata/
```

Each split directory must contain one of these equivalent triplets:

- `shard_000_observations.*`, `shard_000_labels.*`, `shard_000_snrs.*`
- `shard_001_data.*`, `shard_001_labels.*`, `shard_001_snrs.*`
- `part_000_features.*`, `part_000_targets.*`, `part_000_snr.*`

Single-file split triplets such as `observations.npy`, `labels.npy`, and `snrs.npy` are also accepted for debugging and smoke runs, but the recommended production format is sharded.

Each split directory may also contain a single bundle file:

- `dataset.npz`
- `bundle.npz`
- `<split_name>.npz`
- `dataset.pt`
- `bundle.pt`
- `<split_name>.pt`

Bundle keys must resolve to:

- observations or data or features
- labels or targets
- snrs or snr

## Tensor shapes

- observations: `(N, 2, 1024)` or `(N, 1024, 2)`
- labels: `(N,)` integer labels or `(N, C)` one-hot labels
- snrs: `(N,)`

The loader normalizes observations into `(N, 2, 1024)` internally.

## Recommended shard count

Follow the original Harper slicing granularity:

- about `2000` samples per shard

For the original Harper split sizes this implies approximately:

- train: `500` shards if using the original full train split directly
- test: `778` shards if using the original full test split directly

If validation is derived from train with `val_ratio=0.1`, the practical target becomes approximately:

- train: `450` shards
- validation: `50` shards
- test: `778` shards

## Metadata

`data/processed/metadata/classes-fixed.json` stores class names for reporting.

If `validation/` is absent, validation is derived from `train/` using `val_ratio`.
