from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import nn
from torch.amp import GradScaler, autocast

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
    max_grad_norm: float = 1.0,
    scaler: Optional[GradScaler] = None,
) -> EpochResult:
    criterion = nn.CrossEntropyLoss()
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    total_correct = 0
    total_items = 0

    for features, labels, _snrs in loader:
        features = features.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        if is_train:
            optimizer.zero_grad()

        # 混合精度前向传播
        with autocast(device_type=device.type, enabled=scaler is not None):
            logits = model(features)
            loss = criterion(logits, labels)

        if is_train:
            # 混合精度反向传播
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
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
    early_stopping_patience: int = 10,
    use_amp: bool = True,
) -> List[Dict[str, float]]:
    history: List[Dict[str, float]] = []
    best_val_loss = float("inf")
    epochs_without_improvement = 0

    # 混合精度训练
    scaler = GradScaler(device.type) if use_amp and device.type == "cuda" else None
    if scaler:
        print("Using Automatic Mixed Precision (AMP)")

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        train_result = run_epoch(model, train_loader, optimizer, device, scaler=scaler)
        val_result = run_epoch(model, val_loader, optimizer=None, device=device, scaler=None)

        if scheduler is not None:
            scheduler.step(val_result.loss)

        epoch_time = time.time() - epoch_start

        epoch_log = {
            "epoch": epoch,
            "train_loss": train_result.loss,
            "train_accuracy": train_result.accuracy,
            "val_loss": val_result.loss,
            "val_accuracy": val_result.accuracy,
            "lr": float(optimizer.param_groups[0]["lr"]),
            "epoch_time": round(epoch_time, 2),
        }
        history.append(epoch_log)

        print(
            f"Epoch {epoch} ({epoch_time:.1f}s): train_loss={train_result.loss:.4f}, train_acc={train_result.accuracy:.4f}, "
            f"val_loss={val_result.loss:.4f}, val_acc={val_result.accuracy:.4f}, lr={optimizer.param_groups[0]['lr']:.6f}",
            flush=True,
        )

        save_checkpoint(
            run_dir / "checkpoints" / "last.pt",
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "history": history,
                "best_val_loss": best_val_loss,
                "scaler_state_dict": scaler.state_dict() if scaler else None,
            },
        )

        if val_result.loss < best_val_loss:
            best_val_loss = val_result.loss
            epochs_without_improvement = 0
            save_checkpoint(
                run_dir / "checkpoints" / "best.pt",
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "history": history,
                    "best_val_loss": best_val_loss,
                    "scaler_state_dict": scaler.state_dict() if scaler else None,
                },
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stopping_patience:
                print(f"Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)", flush=True)
                break

    return history
