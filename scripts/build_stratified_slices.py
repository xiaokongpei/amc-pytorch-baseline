"""
Build stratified PyTorch slices from the original RadioML2018 HDF5 dataset.

The split is performed inside every modulation x SNR group, then merged into
train/validation/test files. This keeps class and SNR coverage balanced across
all evaluation splits.

Default output:
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
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import h5py
import numpy as np
import torch


DEFAULT_OUTPUT_ROOT = "data/processed_v2_stratified_64_16_20"
DEFAULT_SPLIT_ROOT = "data/splits_v2_stratified_64_16_20"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src-hdf5", required=True)
    parser.add_argument("--class-names-path", default="data/classes-fixed.json")
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--split-root", default=DEFAULT_SPLIT_ROOT)
    parser.add_argument("--train-ratio", type=float, default=0.64)
    parser.add_argument("--validation-ratio", type=float, default=0.16)
    parser.add_argument("--test-ratio", type=float, default=0.20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean-output", action="store_true")
    return parser.parse_args()


def validate_ratios(train_ratio: float, validation_ratio: float, test_ratio: float) -> None:
    ratios = [train_ratio, validation_ratio, test_ratio]
    if any(ratio < 0 for ratio in ratios):
        raise ValueError(f"Split ratios must be non-negative, got {ratios}")
    total = sum(ratios)
    if not np.isclose(total, 1.0):
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.8f}")


def ensure_clean_dir(path: Path, clean_output: bool) -> None:
    if clean_output and path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def grouped_indices(labels: np.ndarray, snrs: np.ndarray) -> dict[tuple[int, int], list[int]]:
    groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for index, (label, snr) in enumerate(zip(labels, snrs)):
        groups[(int(label), int(snr))].append(index)
    return groups


def split_group(
    indices: Iterable[int],
    rng: np.random.Generator,
    train_ratio: float,
    validation_ratio: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    shuffled = np.asarray(list(indices), dtype=np.int64)
    rng.shuffle(shuffled)

    n_total = len(shuffled)
    n_train = int(np.floor(n_total * train_ratio))
    n_validation = int(np.floor(n_total * validation_ratio))

    train = shuffled[:n_train]
    validation = shuffled[n_train : n_train + n_validation]
    test = shuffled[n_train + n_validation :]
    return train, validation, test


def write_indices(path: Path, indices: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(path, np.asarray(indices, dtype=np.int64), fmt="%d", delimiter=",")


def extract_sorted(handle: h5py.File, indices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(indices)
    sorted_indices = indices[order]
    observations = np.asarray(handle["X"][sorted_indices], dtype=np.float32)
    labels = np.argmax(np.asarray(handle["Y"][sorted_indices]), axis=1).astype(np.int64)
    snrs = np.asarray(handle["Z"][sorted_indices]).reshape(-1).astype(np.int64)
    reverse_order = np.argsort(order)
    return observations[reverse_order], labels[reverse_order], snrs[reverse_order]


def write_pt_file(
    handle: h5py.File,
    indices: np.ndarray,
    output_path: Path,
) -> dict[str, object]:
    observations, labels, snrs = extract_sorted(handle, indices)
    torch.save(
        {
            "observations": torch.from_numpy(observations),
            "labels": torch.from_numpy(labels),
            "snrs": torch.from_numpy(snrs),
        },
        output_path,
    )
    size_mb = output_path.stat().st_size / (1024 * 1024)
    return {
        "path": str(output_path),
        "num_samples": int(len(labels)),
        "size_mb": round(size_mb, 2),
    }


def write_split_summary(
    path: Path,
    split_indices: dict[str, np.ndarray],
    labels: np.ndarray,
    snrs: np.ndarray,
) -> None:
    rows = []
    for split_name, indices in split_indices.items():
        split_labels = labels[indices]
        split_snrs = snrs[indices]
        rows.append(
            {
                "split": split_name,
                "num_samples": int(len(indices)),
                "num_classes": int(len(np.unique(split_labels))),
                "num_snrs": int(len(np.unique(split_snrs))),
                "min_snr": int(np.min(split_snrs)),
                "max_snr": int(np.max(split_snrs)),
            }
        )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_distribution(
    path: Path,
    split_indices: dict[str, np.ndarray],
    labels: np.ndarray,
    snrs: np.ndarray,
) -> None:
    rows = []
    for split_name, indices in split_indices.items():
        counts: dict[tuple[int, int], int] = defaultdict(int)
        for label, snr in zip(labels[indices], snrs[indices]):
            counts[(int(label), int(snr))] += 1
        for (label, snr), count in sorted(counts.items()):
            rows.append(
                {
                    "split": split_name,
                    "label": label,
                    "snr": snr,
                    "num_samples": count,
                }
            )

    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "label", "snr", "num_samples"])
        writer.writeheader()
        writer.writerows(rows)


def write_class_names(class_names_path: str | None, metadata_dir: Path, num_classes: int) -> str:
    output_path = metadata_dir / "classes-fixed.json"
    if class_names_path:
        source_path = Path(class_names_path)
        if source_path.exists():
            shutil.copy2(source_path, output_path)
            return str(source_path)
        print(f"Class names file not found, generating numeric class names: {source_path}")

    class_names = [f"class_{index}" for index in range(num_classes)]
    output_path.write_text(json.dumps(class_names, indent=2), encoding="utf-8")
    return "generated_numeric_class_names"


def main() -> None:
    args = parse_args()
    validate_ratios(args.train_ratio, args.validation_ratio, args.test_ratio)

    output_root = Path(args.output_root)
    split_root = Path(args.split_root)
    metadata_dir = output_root / "metadata"

    ensure_clean_dir(output_root, args.clean_output)
    ensure_clean_dir(split_root, args.clean_output)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)

    with h5py.File(args.src_hdf5, "r") as handle:
        print("Reading labels and SNR values...")
        labels = np.argmax(np.asarray(handle["Y"]), axis=1).astype(np.int64)
        snrs = np.asarray(handle["Z"]).reshape(-1).astype(np.int64)

        groups = grouped_indices(labels, snrs)
        split_parts: dict[str, list[np.ndarray]] = {"train": [], "validation": [], "test": []}

        print(f"Stratifying {len(labels)} samples across {len(groups)} modulation x SNR groups...")
        for indices in groups.values():
            train, validation, test = split_group(
                indices,
                rng=rng,
                train_ratio=args.train_ratio,
                validation_ratio=args.validation_ratio,
            )
            split_parts["train"].append(train)
            split_parts["validation"].append(validation)
            split_parts["test"].append(test)

        split_indices = {
            split_name: np.sort(np.concatenate(parts)).astype(np.int64)
            for split_name, parts in split_parts.items()
        }

        print(
            "Split sizes: "
            + ", ".join(f"{name}={len(indices)}" for name, indices in split_indices.items())
        )

        for split_name, indices in split_indices.items():
            index_path = split_root / f"{split_name}_indexes.csv"
            write_indices(index_path, indices)
            print(f"  Wrote indices: {index_path}")

        manifests = {}
        for split_name, indices in split_indices.items():
            print(f"\nWriting {split_name}.pt...")
            manifests[split_name] = write_pt_file(handle, indices, output_root / f"{split_name}.pt")
            print(
                f"  Written {split_name}: {manifests[split_name]['num_samples']} samples, "
                f"{manifests[split_name]['size_mb']} MB"
            )

    class_names_source = write_class_names(
        args.class_names_path,
        metadata_dir=metadata_dir,
        num_classes=int(len(np.unique(labels))),
    )

    write_split_summary(metadata_dir / "split_summary.csv", split_indices, labels, snrs)
    write_distribution(metadata_dir / "snr_class_distribution.csv", split_indices, labels, snrs)

    manifest = {
        "format": "pytorch_pt",
        "split_strategy": "stratified_by_modulation_and_snr",
        "ratios": {
            "train": args.train_ratio,
            "validation": args.validation_ratio,
            "test": args.test_ratio,
        },
        "seed": args.seed,
        "source_hdf5": str(args.src_hdf5),
        "class_names_source": class_names_source,
        "split_root": str(split_root),
        "splits": manifests,
    }
    (metadata_dir / "slice_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    print("\nDone. Use this data root for training:")
    print(f"  --data-root {output_root}")


if __name__ == "__main__":
    main()
