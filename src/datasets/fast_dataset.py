"""
Fast PyTorch Dataset using memory-mapped .pt files.

Three modes:
1. mmap: Memory-map the file, OS handles caching. Low RAM usage.
2. preload: Load everything into CPU memory. Needs RAM.
3. gpu: Load everything into GPU memory. Fastest, needs GPU memory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

import torch
from torch.utils.data import DataLoader, Dataset, TensorDataset


class FastDataset(Dataset):
    """Dataset that loads from a single .pt file."""

    def __init__(self, pt_path: Path | str, mode: str = "mmap", device: torch.device = None):
        self.pt_path = Path(pt_path)
        self.mode = mode
        self.device = device or torch.device("cuda")

        if mode == "gpu":
            data = torch.load(pt_path, map_location=self.device, weights_only=True)
            self._on_gpu = True
        elif mode == "preload":
            data = torch.load(pt_path, map_location="cpu", weights_only=True)
            self._on_gpu = False
        else:
            data = torch.load(pt_path, map_location="cpu", mmap=True, weights_only=True)
            self._on_gpu = False

        self._observations = data["observations"]
        if self._observations.shape[-1] == 2:
            self._observations = self._observations.permute(0, 2, 1)
        self._labels = data["labels"]
        self._snrs = data["snrs"]
        self._length = len(self._labels)
        location = "GPU" if mode == "gpu" else "CPU"
        print(f"Loaded {self._length} samples from {self.pt_path.name} to {location} (mode={mode})")

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        if self._on_gpu:
            return self._observations[index], self._labels[index], self._snrs[index]
        else:
            return (
                self._observations[index].clone(),
                self._labels[index].clone(),
                self._snrs[index].clone(),
            )


def build_fast_dataloaders(config: Dict, mode: str = "mmap") -> Dict[str, DataLoader]:
    """Build dataloaders from fast .pt format."""
    root = Path(config["data"]["root"])
    device = torch.device("cuda" if mode == "gpu" else "cpu")

    train_path = root / "train.pt"
    val_path = root / "validation.pt"
    test_path = root / "test.pt"

    missing = [p for p in [train_path, val_path, test_path] if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing: {missing}")

    batch_size = int(config["training"]["batch_size"])
    num_workers = int(config["training"]["num_workers"])
    pin_memory = bool(config["data"].get("pin_memory", False))

    dataloaders = {}

    if mode == "gpu":
        # GPU mode: load all data to GPU
        for split, path, shuffle in [("train", train_path, True), ("val", val_path, False), ("test", test_path, False)]:
            data = torch.load(path, map_location=device, weights_only=True)
            obs = data["observations"]
            if obs.shape[-1] == 2:
                obs = obs.permute(0, 2, 1)

            dataset = TensorDataset(obs, data["labels"], data["snrs"])
            print(f"Loaded {len(dataset)} samples from {path.name} to GPU")
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
            dataloaders[split] = loader
    else:
        # CPU modes: mmap or preload
        for split, path in [("train", train_path), ("val", val_path), ("test", test_path)]:
            dataset = FastDataset(path, mode=mode, device=device)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=0 if mode == "preload" else num_workers,
                pin_memory=pin_memory if mode != "preload" else False,
            )
            dataloaders[split] = loader

    return dataloaders
