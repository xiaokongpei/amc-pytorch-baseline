from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn

from src.engine.checkpointing import save_checkpoint


@dataclass
class EpochResult:
    loss: float
    accuracy: float


def run_epoch(
    model: nn.Module,
    loader,
    optimizer: Optional[torch.optim.Optimizer],
    device: torch.device,
) -> EpochResult:
    criterion = nn.CrossEntropyLoss()
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for features, labels, _snrs in loader:
        features = features.to(device)
        labels = labels.to(device)

        if is_train:
            optimizer.zero_grad()

        logits = model(features)
        loss = criterion(logits, labels)

        if is_train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * labels.size(0)
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_items += labels.size(0)

    return EpochResult(
        loss=total_loss / max(total_items, 1),
        accuracy=total_correct / max(total_items, 1),
    )


def fit(
    model: nn.Module,
    train_loader,
    val_loader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    epochs: int,
    device: torch.device,
    run_dir,
) -> List[Dict[str, float]]:
    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")

    for epoch in range(1, epochs + 1):
        train_result = run_epoch(model, train_loader, optimizer, device)
        val_result = run_epoch(model, val_loader, optimizer=None, device=device)

        if scheduler is not None:
            scheduler.step(val_result.loss)

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "train_accuracy": train_result.accuracy,
            "val_loss": val_result.loss,
            "val_accuracy": val_result.accuracy,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(epoch_log)

        save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "best_val_loss": best_val_loss,
            },
        )

        if val_result.loss <= best_val_loss:
            best_val_loss = val_result.loss
            save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                    "best_val_loss": best_val_loss,
                },
            )

    return history
