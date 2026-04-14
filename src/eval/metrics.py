from __future__ import annotations

from collections import defaultdict
from typing import Dict, List

import torch
from torch import nn


def compute_accuracy(predictions, labels) -> float:
    total = len(labels)
    correct = sum(int(pred == label) for pred, label in zip(predictions, labels))
    return correct / max(total, 1)


def compute_snr_accuracy(predictions, labels, snrs) -> Dict[str, float]:
    buckets = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred, label, snr in zip(predictions, labels, snrs):
        buckets[int(snr)]["correct"] += int(pred == label)
        buckets[int(snr)]["total"] += 1
    return {
        str(snr): values["correct"] / max(values["total"], 1)
        for snr, values in sorted(buckets.items(), key=lambda item: item[0])
    }


def evaluate_model(model, loader, device: torch.device):
    criterion = nn.CrossEntropyLoss()
    model.eval()

    predictions: List[int] = []
    labels_out: List[int] = []
    snrs_out: List[int] = []
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for features, labels, snrs in loader:
            features = features.to(device)
            labels = labels.to(device)
            logits = model(features)
            loss = criterion(logits, labels)

            batch_predictions = logits.argmax(dim=1).cpu().tolist()
            predictions.extend(batch_predictions)
            labels_out.extend(labels.cpu().tolist())
            snrs_out.extend(snrs.cpu().tolist())
            total_loss += loss.item() * labels.size(0)
            total_items += labels.size(0)

    metrics = {
        "loss": total_loss / max(total_items, 1),
        "accuracy": compute_accuracy(predictions, labels_out),
    }
    snr_metrics = compute_snr_accuracy(predictions, labels_out, snrs_out)
    return metrics, snr_metrics
