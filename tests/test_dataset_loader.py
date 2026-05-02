from pathlib import Path

import torch

from src.datasets.fast_dataset import FastDataset


def _write_pt(path: Path, n_samples: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    observations = torch.randn(n_samples, 2, 1024)
    labels = torch.arange(n_samples, dtype=torch.long) % 24
    snrs = torch.full((n_samples,), -20, dtype=torch.long)
    torch.save({"observations": observations, "labels": labels, "snrs": snrs}, path)


def test_fast_dataset_loads_pt(tmp_path: Path):
    _write_pt(tmp_path / "train.pt", 10)
    dataset = FastDataset(tmp_path / "train.pt", mode="preload")
    assert len(dataset) == 10
    features, label, snr = dataset[0]
    assert tuple(features.shape) == (2, 1024)
    assert int(label) >= 0
    assert int(snr) == -20


def test_fast_dataset_transposes_channel_last(tmp_path: Path):
    path = tmp_path / "train.pt"
    path.parent.mkdir(parents=True, exist_ok=True)
    observations = torch.randn(5, 1024, 2)
    torch.save({
        "observations": observations,
        "labels": torch.zeros(5, dtype=torch.long),
        "snrs": torch.zeros(5, dtype=torch.long),
    }, path)
    dataset = FastDataset(path, mode="preload")
    features, _, _ = dataset[0]
    assert tuple(features.shape) == (2, 1024)
