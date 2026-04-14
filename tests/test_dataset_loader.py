from pathlib import Path

import numpy as np

from src.datasets.radioml_dataset import build_datasets


def _write_shard(split_dir: Path, prefix: str, n_samples: int):
    split_dir.mkdir(parents=True, exist_ok=True)
    observations = np.random.randn(n_samples, 1024, 2).astype(np.float32)
    labels = np.arange(n_samples, dtype=np.int64) % 24
    snrs = np.full((n_samples,), -20, dtype=np.int64)
    np.save(split_dir / f"{prefix}_observations.npy", observations)
    np.save(split_dir / f"{prefix}_labels.npy", labels)
    np.save(split_dir / f"{prefix}_snrs.npy", snrs)


def test_build_datasets_from_sliced_shards(tmp_path: Path):
    data_root = tmp_path / "processed"
    metadata_dir = data_root / "metadata"
    metadata_dir.mkdir(parents=True, exist_ok=True)
    (metadata_dir / "classes-fixed.json").write_text('["A", "B"]', encoding="utf-8")
    _write_shard(data_root / "train", "shard_000", 6)
    _write_shard(data_root / "train", "shard_001", 4)
    _write_shard(data_root / "test", "shard_000", 6)

    config = {
        "data": {
            "root": str(data_root),
            "metadata_dir": str(metadata_dir),
            "train_dir": "train",
            "val_dir": "validation",
            "test_dir": "test",
            "class_names": "classes-fixed.json",
            "val_ratio": 0.2,
            "pin_memory": False,
        },
        "training": {"seed": 42, "batch_size": 4, "num_workers": 0},
    }

    datasets = build_datasets(config)
    train_features, train_label, train_snr = datasets["train"][0]
    assert len(datasets["train"]) == 8
    assert len(datasets["val"]) == 2
    assert len(datasets["test"]) == 6
    assert tuple(train_features.shape) == (2, 1024)
    assert int(train_label) >= 0
    assert int(train_snr) == -20
