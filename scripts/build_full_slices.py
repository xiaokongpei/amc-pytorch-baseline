from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-hdf5", required=True)
    parser.add_argument("--train-index-path", required=True)
    parser.add_argument("--test-index-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--class-names-path", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--shard-size", type=int, default=2000)
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def read_indices(path: str | Path) -> np.ndarray:
    indices = np.genfromtxt(path, delimiter=",", dtype=np.int64)
    if indices.ndim == 0:
        indices = np.asarray([int(indices)])
    return np.asarray(indices, dtype=np.int64)


def split_train_validation_indices(
    train_indices: np.ndarray, val_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    if not 0.0 <= val_ratio < 1.0:
        raise ValueError(f"val_ratio must be in [0, 1), got {val_ratio}")
    shuffled = np.asarray(train_indices, dtype=np.int64).copy()
    rng = np.random.default_rng(seed)
    rng.shuffle(shuffled)
    if val_ratio == 0.0:
        return shuffled, np.asarray([], dtype=np.int64)
    val_count = max(int(len(shuffled) * val_ratio), 1)
    return shuffled[val_count:], shuffled[:val_count]


def ensure_clean_split_dir(path: Path, clean_output: bool) -> None:
    if path.exists() and clean_output:
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def extract_split(handle: h5py.File, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(indices)
    sorted_indices = indices[order]
    restore_order = np.argsort(order)
    observations = np.asarray(handle["X"][sorted_indices], dtype=np.float32)[restore_order]
    labels = np.argmax(np.asarray(handle["Y"][sorted_indices]), axis=1).astype(np.int64)[restore_order]
    snrs = np.asarray(handle["Z"][sorted_indices]).reshape(-1).astype(np.int64)[restore_order]
    return observations, labels, snrs


def _print_progress(message: str) -> None:
    print(message, flush=True)


def write_shard_files(
    split_dir: Path,
    observations: np.ndarray,
    labels: np.ndarray,
    snrs: np.ndarray,
    shard_idx: int,
) -> None:
    prefix = f"shard_{shard_idx:03d}"
    np.save(split_dir / f"{prefix}_observations.npy", observations)
    np.save(split_dir / f"{prefix}_labels.npy", labels)
    np.save(split_dir / f"{prefix}_snrs.npy", snrs)


def write_split_shards_from_indices(
    handle: h5py.File,
    split_name: str,
    split_dir: Path,
    indices: np.ndarray,
    shard_size: int,
) -> dict:
    if shard_size <= 0:
        raise ValueError(f"shard_size must be positive, got {shard_size}")

    sample_count = int(len(indices))
    if sample_count == 0:
        _print_progress(f"[{split_name}] no samples to write")
        return {"samples": 0, "shards": 0}

    n_shards = int(np.ceil(sample_count / shard_size))
    _print_progress(f"[{split_name}] start: samples={sample_count}, shard_size={shard_size}, shards={n_shards}")

    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = min((shard_idx + 1) * shard_size, sample_count)
        shard_indices = indices[start:end]
        observations, labels, snrs = extract_split(handle, shard_indices)
        write_shard_files(split_dir, observations, labels, snrs, shard_idx)
        _print_progress(f"[{split_name}] wrote shard {shard_idx + 1}/{n_shards} ({end}/{sample_count} samples)")

    return {"samples": sample_count, "shards": n_shards}


def write_manifest(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    metadata_dir = output_root / "metadata"
    train_dir = output_root / "train"
    validation_dir = output_root / "validation"
    test_dir = output_root / "test"

    for split_dir in (train_dir, validation_dir, test_dir):
        ensure_clean_split_dir(split_dir, clean_output=args.clean_output)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    train_indices = read_indices(args.train_index_path)
    test_indices = read_indices(args.test_index_path)
    train_indices, validation_indices = split_train_validation_indices(
        train_indices=train_indices,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    _print_progress(
        f"Preparing slices with train={len(train_indices)}, validation={len(validation_indices)}, test={len(test_indices)}"
    )

    with h5py.File(args.src_hdf5, "r") as handle:
        train_manifest = write_split_shards_from_indices(
            handle=handle,
            split_name="train",
            split_dir=train_dir,
            indices=train_indices,
            shard_size=args.shard_size,
        )
        validation_manifest = write_split_shards_from_indices(
            handle=handle,
            split_name="validation",
            split_dir=validation_dir,
            indices=validation_indices,
            shard_size=args.shard_size,
        )
        test_manifest = write_split_shards_from_indices(
            handle=handle,
            split_name="test",
            split_dir=test_dir,
            indices=test_indices,
            shard_size=args.shard_size,
        )

    shutil.copy2(args.class_names_path, metadata_dir / "classes-fixed.json")

    manifest = {
        "source_hdf5": str(Path(args.src_hdf5).resolve()),
        "train_index_path": str(Path(args.train_index_path).resolve()),
        "test_index_path": str(Path(args.test_index_path).resolve()),
        "class_names_path": str(Path(args.class_names_path).resolve()),
        "output_root": str(output_root.resolve()),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
        "shard_size": args.shard_size,
        "splits": {
            "train": train_manifest,
            "validation": validation_manifest,
            "test": test_manifest,
        },
    }
    write_manifest(metadata_dir / "slice_manifest.json", manifest)
    _print_progress(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
