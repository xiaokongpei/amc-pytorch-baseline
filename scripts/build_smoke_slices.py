from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-hdf5", required=True)
    parser.add_argument("--train-index-path", required=True)
    parser.add_argument("--test-index-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--train-samples", type=int, default=4000)
    parser.add_argument("--test-samples", type=int, default=2000)
    parser.add_argument("--shard-size", type=int, default=2000)
    return parser.parse_args()


def read_indices(path: str | Path, limit: int) -> np.ndarray:
    indices = np.genfromtxt(path, delimiter=",", dtype=np.int64)
    if indices.ndim == 0:
        indices = np.asarray([int(indices)])
    return np.asarray(indices[:limit], dtype=np.int64)


def write_shards(split_dir: Path, observations: np.ndarray, labels: np.ndarray, snrs: np.ndarray, shard_size: int) -> None:
    split_dir.mkdir(parents=True, exist_ok=True)
    n_shards = int(np.ceil(len(labels) / shard_size))
    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = min((shard_idx + 1) * shard_size, len(labels))
        prefix = f"shard_{shard_idx:03d}"
        np.save(split_dir / f"{prefix}_observations.npy", observations[start:end])
        np.save(split_dir / f"{prefix}_labels.npy", labels[start:end])
        np.save(split_dir / f"{prefix}_snrs.npy", snrs[start:end])


def extract_split(handle: h5py.File, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    observations = np.asarray(handle["X"][indices], dtype=np.float32)
    labels = np.argmax(np.asarray(handle["Y"][indices]), axis=1).astype(np.int64)
    snrs = np.asarray(handle["Z"][indices]).reshape(-1).astype(np.int64)
    return observations, labels, snrs


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    train_dir = output_root / "train"
    test_dir = output_root / "test"
    validation_dir = output_root / "validation"
    validation_dir.mkdir(parents=True, exist_ok=True)

    train_indices = read_indices(args.train_index_path, args.train_samples)
    test_indices = read_indices(args.test_index_path, args.test_samples)

    with h5py.File(args.src_hdf5, "r") as handle:
        train_obs, train_labels, train_snrs = extract_split(handle, train_indices)
        test_obs, test_labels, test_snrs = extract_split(handle, test_indices)

    write_shards(train_dir, train_obs, train_labels, train_snrs, args.shard_size)
    write_shards(test_dir, test_obs, test_labels, test_snrs, args.shard_size)


if __name__ == "__main__":
    main()
