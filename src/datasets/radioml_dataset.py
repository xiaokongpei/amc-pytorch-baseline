from __future__ import annotations

# Keep all path assumptions in this file so local, AutoDL, and Kaggle migration
# changes stay isolated from engine and model code.

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import bisect
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset


OBSERVATION_NAMES = ("observations", "data", "features", "x")
LABEL_NAMES = ("labels", "targets", "y")
SNR_NAMES = ("snrs", "snr", "z")
SUPPORTED_EXTENSIONS = (".npy", ".npz", ".pt")


@dataclass
class DatasetPaths:
    root: Path
    metadata_dir: Path
    train_dir: Path
    test_dir: Path
    val_dir: Optional[Path]
    class_names: Path


@dataclass
class ShardSpec:
    prefix: str
    observations_path: Path
    labels_path: Path
    snrs_path: Path
    length: int


def _load_numpy_payload(path: Path, preferred_keys: Sequence[str]) -> np.ndarray:
    payload = np.load(path, allow_pickle=False)
    if isinstance(payload, np.ndarray):
        return payload

    for key in preferred_keys:
        if key in payload:
            return np.asarray(payload[key])

    first_key = next(iter(payload.files))
    return np.asarray(payload[first_key])


def _load_pt_payload(path: Path, preferred_keys: Sequence[str]) -> np.ndarray:
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        for key in preferred_keys:
            if key in payload:
                value = payload[key]
                return value.cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)
        first_key = next(iter(payload.keys()))
        value = payload[first_key]
        return value.cpu().numpy() if isinstance(value, torch.Tensor) else np.asarray(value)

    if isinstance(payload, torch.Tensor):
        return payload.cpu().numpy()

    return np.asarray(payload)


def _load_array(path: Path, preferred_keys: Sequence[str]) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path, allow_pickle=False)
    if path.suffix == ".npz":
        return _load_numpy_payload(path, preferred_keys)
    if path.suffix == ".pt":
        return _load_pt_payload(path, preferred_keys)
    raise ValueError(f"Unsupported slice file format: {path}")


def _find_by_stems(split_dir: Path, stems: Sequence[str]) -> Optional[Path]:
    for stem in stems:
        for extension in SUPPORTED_EXTENSIONS:
            candidate = split_dir / f"{stem}{extension}"
            if candidate.exists():
                return candidate
    return None


def _normalize_observations(observations: np.ndarray) -> np.ndarray:
    observations = np.asarray(observations, dtype=np.float32)
    if observations.ndim != 3:
        raise ValueError(f"Observations must be rank-3, got shape {observations.shape}")

    if observations.shape[1] == 2 and observations.shape[2] != 2:
        return observations
    if observations.shape[2] == 2:
        return observations.transpose(0, 2, 1)
    raise ValueError(f"Expected channel dimension of size 2, got shape {observations.shape}")


def _normalize_labels(labels: np.ndarray) -> np.ndarray:
    labels = np.asarray(labels)
    if labels.ndim == 2:
        return labels.argmax(axis=1).astype(np.int64)
    return labels.astype(np.int64).reshape(-1)


def _normalize_snrs(snrs: np.ndarray) -> np.ndarray:
    return np.asarray(snrs).astype(np.int64).reshape(-1)


def _infer_length(labels_path: Path) -> int:
    labels = _load_array(labels_path, LABEL_NAMES)
    return int(_normalize_labels(labels).shape[0])


def _build_single_shard(split_dir: Path) -> Optional[ShardSpec]:
    observations_path = _find_by_stems(split_dir, OBSERVATION_NAMES)
    labels_path = _find_by_stems(split_dir, LABEL_NAMES)
    snrs_path = _find_by_stems(split_dir, SNR_NAMES)
    if observations_path and labels_path and snrs_path:
        return ShardSpec(
            prefix="single",
            observations_path=observations_path,
            labels_path=labels_path,
            snrs_path=snrs_path,
            length=_infer_length(labels_path),
        )
    return None


def _discover_sharded_triplets(split_dir: Path, field_names: Sequence[str]) -> Dict[str, Path]:
    discovered: Dict[str, Path] = {}
    for path in split_dir.iterdir():
        if path.suffix not in SUPPORTED_EXTENSIONS or not path.is_file():
            continue
        for field in field_names:
            suffix = f"_{field}"
            if path.stem.endswith(suffix):
                prefix = path.stem[: -len(suffix)]
                discovered[prefix] = path
    return discovered


def _discover_shard_specs(split_dir: Path) -> list[ShardSpec]:
    single = _build_single_shard(split_dir)
    if single is not None:
        return [single]

    observation_map = _discover_sharded_triplets(split_dir, OBSERVATION_NAMES)
    label_map = _discover_sharded_triplets(split_dir, LABEL_NAMES)
    snr_map = _discover_sharded_triplets(split_dir, SNR_NAMES)
    prefixes = sorted(set(observation_map) & set(label_map) & set(snr_map))
    if not prefixes:
        raise FileNotFoundError(
            f"No slice triplets found in {split_dir}. Expected either single files "
            "like observations.npy/labels.npy/snrs.npy or sharded files like "
            "shard_000_observations.npy/shard_000_labels.npy/shard_000_snrs.npy."
        )

    shard_specs = []
    for prefix in prefixes:
        labels_path = label_map[prefix]
        shard_specs.append(
            ShardSpec(
                prefix=prefix,
                observations_path=observation_map[prefix],
                labels_path=labels_path,
                snrs_path=snr_map[prefix],
                length=_infer_length(labels_path),
            )
        )
    return shard_specs


def _split_train_validation(
    train_count: int, val_ratio: float, seed: int
) -> Tuple[np.ndarray, np.ndarray]:
    indices = np.arange(train_count, dtype=np.int64)
    if val_ratio <= 0.0:
        return indices, np.asarray([], dtype=np.int64)
    rng = np.random.default_rng(seed)
    rng.shuffle(indices)
    val_count = max(int(train_count * val_ratio), 1)
    return indices[val_count:], indices[:val_count]


class ShardedRadioMLDataset(Dataset):
    def __init__(self, shard_specs: Sequence[ShardSpec]):
        self.shard_specs = list(shard_specs)
        self.cumulative_sizes = []
        total = 0
        for spec in self.shard_specs:
            total += spec.length
            self.cumulative_sizes.append(total)
        self._cached_shard_index: Optional[int] = None
        self._cached_observations: Optional[np.ndarray] = None
        self._cached_labels: Optional[np.ndarray] = None
        self._cached_snrs: Optional[np.ndarray] = None

    def __len__(self) -> int:
        return self.cumulative_sizes[-1] if self.cumulative_sizes else 0

    def _load_shard(self, shard_index: int) -> None:
        if self._cached_shard_index == shard_index:
            return

        spec = self.shard_specs[shard_index]
        observations = _normalize_observations(_load_array(spec.observations_path, OBSERVATION_NAMES))
        labels = _normalize_labels(_load_array(spec.labels_path, LABEL_NAMES))
        snrs = _normalize_snrs(_load_array(spec.snrs_path, SNR_NAMES))

        if not (len(observations) == len(labels) == len(snrs) == spec.length):
            raise ValueError(
                f"Shard {spec.prefix} has mismatched lengths: "
                f"{len(observations)}, {len(labels)}, {len(snrs)}, expected {spec.length}"
            )

        self._cached_shard_index = shard_index
        self._cached_observations = observations
        self._cached_labels = labels
        self._cached_snrs = snrs

    def __getitem__(self, index: int):
        shard_index = bisect.bisect_right(self.cumulative_sizes, index)
        shard_start = 0 if shard_index == 0 else self.cumulative_sizes[shard_index - 1]
        local_index = index - shard_start
        self._load_shard(shard_index)
        assert self._cached_observations is not None
        assert self._cached_labels is not None
        assert self._cached_snrs is not None
        features = torch.from_numpy(self._cached_observations[local_index])
        label = torch.tensor(int(self._cached_labels[local_index]), dtype=torch.long)
        snr = torch.tensor(int(self._cached_snrs[local_index]), dtype=torch.long)
        return features, label, snr


def resolve_dataset_paths(data_root: str | Path, metadata_dir: str | Path, data_config: Dict[str, object]) -> DatasetPaths:
    root = Path(data_root)
    meta = Path(metadata_dir)
    train_dir = root / str(data_config["train_dir"])
    test_dir = root / str(data_config["test_dir"])
    val_name = data_config.get("val_dir")
    val_dir = (root / str(val_name)) if val_name else None
    class_names = meta / str(data_config["class_names"])

    required = [root, meta, train_dir, test_dir, class_names]
    missing = [path for path in required if not path.exists()]
    if missing:
        joined = ", ".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing sliced dataset assets: {joined}")
    return DatasetPaths(root=root, metadata_dir=meta, train_dir=train_dir, test_dir=test_dir, val_dir=val_dir, class_names=class_names)


def build_datasets(config: Dict[str, object]) -> Dict[str, Dataset]:
    data_cfg = config["data"]
    paths = resolve_dataset_paths(data_cfg["root"], data_cfg["metadata_dir"], data_cfg)

    train_dataset: Dataset = ShardedRadioMLDataset(_discover_shard_specs(paths.train_dir))
    test_dataset: Dataset = ShardedRadioMLDataset(_discover_shard_specs(paths.test_dir))

    if paths.val_dir is not None and paths.val_dir.exists():
        try:
            val_dataset: Dataset = ShardedRadioMLDataset(_discover_shard_specs(paths.val_dir))
        except FileNotFoundError:
            val_dataset = Subset(train_dataset, [])
    else:
        val_dataset = Subset(train_dataset, [])

    if len(val_dataset) == 0:
        train_indices, val_indices = _split_train_validation(
            train_count=len(train_dataset),
            val_ratio=float(data_cfg.get("val_ratio", 0.0)),
            seed=int(config["training"]["seed"]),
        )
        train_dataset = Subset(train_dataset, train_indices.tolist())
        val_dataset = Subset(train_dataset.dataset if isinstance(train_dataset, Subset) else train_dataset, val_indices.tolist())

    return {"train": train_dataset, "val": val_dataset, "test": test_dataset}


def build_dataloaders(config: Dict[str, object]) -> Dict[str, DataLoader]:
    datasets = build_datasets(config)
    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    pin_memory = bool(config["data"].get("pin_memory", False))
    return {
        "train": DataLoader(
            datasets["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "val": DataLoader(
            datasets["val"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
        "test": DataLoader(
            datasets["test"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
        ),
    }
