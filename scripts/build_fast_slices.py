"""
Build the active PyTorch baseline data format from the original HDF5 dataset.

Output structure:
    data/processed/
        train.pt
        validation.pt
        test.pt
        metadata/
            classes-fixed.json
            slice_manifest.json
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Tuple

import h5py
import numpy as np
import torch


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-hdf5", required=True)
    parser.add_argument("--train-index-path", required=True)
    parser.add_argument("--test-index-path", required=True)
    parser.add_argument("--class-names-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
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


def extract_sorted(handle: h5py.File, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(indices)
    sorted_indices = indices[order]
    observations = np.asarray(handle["X"][sorted_indices], dtype=np.float32)
    labels = np.argmax(np.asarray(handle["Y"][sorted_indices]), axis=1).astype(np.int64)
    snrs = np.asarray(handle["Z"][sorted_indices]).reshape(-1).astype(np.int64)
    return observations[order], labels[order], snrs[order]


def write_pt_file(
    output_path: Path,
    observations: np.ndarray,
    labels: np.ndarray,
    snrs: np.ndarray,
) -> dict:
    data = {
        "observations": torch.from_numpy(observations),
        "labels": torch.from_numpy(labels),
        "snrs": torch.from_numpy(snrs),
    }
    torch.save(data, output_path)

    size_mb = output_path.stat().st_size / (1024 * 1024)
    return {
        "path": str(output_path),
        "num_samples": len(labels),
        "size_mb": round(size_mb, 2),
    }


def main():
    args = parse_args()
    output_root = Path(args.output_root)
    metadata_dir = output_root / "metadata"

    if args.clean_output and output_root.exists():
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    print("Reading index files...")
    train_indices = read_indices(args.train_index_path)
    test_indices = read_indices(args.test_index_path)
    train_indices, validation_indices = split_train_validation_indices(
        train_indices=train_indices,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(
        f"Split sizes: train={len(train_indices)}, validation={len(validation_indices)}, test={len(test_indices)}"
    )

    manifests = {}
    with h5py.File(args.src_hdf5, "r") as handle:
        for split_name, indices in [
            ("train", train_indices),
            ("validation", validation_indices),
            ("test", test_indices),
        ]:
            print(f"\nProcessing {split_name}...")
            obs, labels, snrs = extract_sorted(handle, indices)
            print(f"  Shape: observations {obs.shape}, labels {labels.shape}")
            manifest = write_pt_file(output_root / f"{split_name}.pt", obs, labels, snrs)
            manifests[split_name] = manifest
            print(f"  Written: {manifest['path']} ({manifest['size_mb']} MB)")

    shutil.copy2(args.class_names_path, metadata_dir / "classes-fixed.json")

    manifest = {
        "format": "pytorch_pt",
        "splits": manifests,
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }
    (metadata_dir / "slice_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("\n" + "=" * 50)
    print("Done! Output files:")
    for split_name, split_manifest in manifests.items():
        print(f"  {split_name}: {split_manifest['size_mb']} MB, {split_manifest['num_samples']} samples")
    print("=" * 50)


if __name__ == "__main__":
    main()
