from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn


# Group mapping for 24 modulation classes
GROUP_ORDER = ["Amplitude", "Phase", "Amplitude and Phase", "Frequency"]
GROUP_DICT = {
    0: 0,   # OOK
    1: 0,   # BPSK
    2: 0,   # 4-PAM
    3: 1,   # 4-PSK
    4: 1,   # 8-PSK
    5: 1,   # QPSK
    6: 1,   # 16-APSK
    7: 1,   # 32-APSK
    8: 2,   # 16-QAM
    9: 2,   # 32-QAM
    10: 2,  # 64-QAM
    11: 2,  # 128-QAM
    12: 2,  # 256-QAM
    13: 2,  # AM-DSB-SC
    14: 2,  # AM-SSB-SC
    15: 2,  # 16-QAM (duplicate?)
    16: 2,  # 64-QAM (duplicate?)
    17: 0,  # FM
    18: 0,  # GMSK
    19: 0,  # OQPSK
    20: 0,  # 8-PSK (duplicate?)
    21: 3,  # OFDM-64-QAM
    22: 3,  # OFDM-BPSK
    23: 1,  # OFDM-QPSK
}


def compute_accuracy(predictions: List[int], labels: List[int]) -> float:
    total = len(labels)
    correct = sum(int(pred == label) for pred, label in zip(predictions, labels))
    return correct / max(total, 1)


def compute_snr_accuracy(predictions: List[int], labels: List[int], snrs: List[int]) -> Dict[str, float]:
    buckets = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred, label, snr in zip(predictions, labels, snrs):
        buckets[int(snr)]["correct"] += int(pred == label)
        buckets[int(snr)]["total"] += 1
    return {
        str(snr): values["correct"] / max(values["total"], 1)
        for snr, values in sorted(buckets.items(), key=lambda item: item[0])
    }


def compute_per_class_accuracy(
    predictions: List[int], labels: List[int], num_classes: int, class_names: List[str] = None
) -> Dict[str, float]:
    """Compute accuracy for each class."""
    buckets = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred, label in zip(predictions, labels):
        buckets[label]["total"] += 1
        if pred == label:
            buckets[label]["correct"] += 1

    results = {}
    for i in range(num_classes):
        acc = buckets[i]["correct"] / max(buckets[i]["total"], 1)
        if class_names and i < len(class_names):
            results[class_names[i]] = acc
        else:
            results[f"class_{i}"] = acc
    return results


def compute_confusion_matrix(
    predictions: List[int], labels: List[int], num_classes: int
) -> np.ndarray:
    """Compute confusion matrix. Rows = true labels, Cols = predictions."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    for pred, label in zip(predictions, labels):
        cm[label, pred] += 1
    return cm


def compute_group_accuracy(predictions: List[int], labels: List[int]) -> Dict[str, float]:
    """Compute accuracy for each modulation group."""
    buckets = defaultdict(lambda: {"correct": 0, "total": 0})
    for pred, label in zip(predictions, labels):
        true_group = GROUP_DICT[label]
        pred_group = GROUP_DICT[pred]
        buckets[true_group]["total"] += 1
        if true_group == pred_group:
            buckets[true_group]["correct"] += 1

    return {
        GROUP_ORDER[g]: buckets[g]["correct"] / max(buckets[g]["total"], 1)
        for g in sorted(buckets.keys())
    }


def compute_group_snr_accuracy(
    predictions: List[int], labels: List[int], snrs: List[int]
) -> Dict[str, Dict[str, float]]:
    """Compute group accuracy for each SNR level."""
    # First compute per-group, per-SNR stats
    buckets = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "total": 0}))
    for pred, label, snr in zip(predictions, labels, snrs):
        true_group = GROUP_DICT[label]
        pred_group = GROUP_DICT[pred]
        buckets[true_group][int(snr)]["total"] += 1
        if true_group == pred_group:
            buckets[true_group][int(snr)]["correct"] += 1

    # Format results
    results = {}
    for g in sorted(buckets.keys()):
        group_name = GROUP_ORDER[g]
        results[group_name] = {
            str(snr): buckets[g][snr]["correct"] / max(buckets[g][snr]["total"], 1)
            for snr in sorted(buckets[g].keys())
        }
    return results


def compute_topk_accuracy(
    logits: torch.Tensor, labels: torch.Tensor, k: int
) -> float:
    """Compute top-k accuracy."""
    _, topk_indices = logits.topk(k, dim=1)
    correct = topk_indices.eq(labels.unsqueeze(1).expand_as(topk_indices)).sum().item()
    return correct / labels.size(0)


def compute_snr_confusion_matrices(
    predictions: List[int], labels: List[int], snrs: List[int], num_classes: int
) -> Dict[str, np.ndarray]:
    """Compute confusion matrix for each SNR level."""
    snr_data = defaultdict(lambda: {"preds": [], "labels": []})
    for pred, label, snr in zip(predictions, labels, snrs):
        snr_data[snr]["preds"].append(pred)
        snr_data[snr]["labels"].append(label)

    return {
        str(snr): compute_confusion_matrix(data["preds"], data["labels"], num_classes)
        for snr, data in sorted(snr_data.items())
    }


def evaluate_model(
    model: nn.Module, loader, device: torch.device, num_classes: int = 24,
    class_names: List[str] = None
) -> Tuple[Dict, Dict, Dict]:
    """
    Comprehensive model evaluation.

    Returns:
        metrics: Overall metrics (loss, accuracy, top-2 accuracy)
        snr_metrics: SNR-wise accuracy
        detailed_metrics: Confusion matrix, per-class metrics, group metrics, etc.
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()

    predictions: List[int] = []
    labels_out: List[int] = []
    snrs_out: List[int] = []
    all_logits: List[torch.Tensor] = []
    total_loss = 0.0
    total_items = 0

    with torch.no_grad():
        for features, labels, snrs in loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = model(features)
            loss = criterion(logits, labels)

            batch_predictions = logits.argmax(dim=1).cpu().tolist()
            predictions.extend(batch_predictions)
            labels_out.extend(labels.cpu().tolist())
            snrs_out.extend(snrs.cpu().tolist())
            all_logits.append(logits.cpu())
            total_loss += loss.item() * labels.size(0)
            total_items += labels.size(0)

    # Concatenate all logits for top-k accuracy
    all_logits = torch.cat(all_logits, dim=0)
    all_labels = torch.tensor(labels_out)

    # Compute all metrics
    accuracy = compute_accuracy(predictions, labels_out)
    snr_accuracy = compute_snr_accuracy(predictions, labels_out, snrs_out)

    # Confusion matrix
    confusion_matrix = compute_confusion_matrix(predictions, labels_out, num_classes)

    # Per-class accuracy
    per_class_acc = compute_per_class_accuracy(predictions, labels_out, num_classes, class_names)

    # Group accuracy
    group_acc = compute_group_accuracy(predictions, labels_out)
    group_snr_acc = compute_group_snr_accuracy(predictions, labels_out, snrs_out)

    # Top-2 accuracy
    top2_acc = compute_topk_accuracy(all_logits, all_labels, k=2)

    metrics = {
        "loss": total_loss / max(total_items, 1),
        "accuracy": accuracy,
        "top2_accuracy": top2_acc,
    }

    detailed_metrics = {
        "confusion_matrix": confusion_matrix.tolist(),
        "per_class_accuracy": per_class_acc,
        "group_accuracy": group_acc,
        "group_snr_accuracy": group_snr_acc,
        "snr_confusion_matrices": {k: v.tolist() for k, v in compute_snr_confusion_matrices(predictions, labels_out, snrs_out, num_classes).items()},
    }

    return metrics, snr_accuracy, detailed_metrics


def plot_confusion_matrix(matrix: np.ndarray, class_names: List[str], save_path: Path) -> None:
    """Plot normalized confusion matrix as heatmap."""
    normalized = matrix.astype(float) / np.maximum(matrix.sum(axis=1, keepdims=True), 1)

    plt.figure(figsize=(14, 12))
    plt.imshow(normalized, interpolation="nearest", cmap="Blues", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Normalized Accuracy")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.yticks(range(len(class_names)), class_names)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("Normalized Confusion Matrix")
    plt.tight_layout()

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_accuracy_vs_snr(snr_accuracy: Dict[str, float], save_path: Path) -> None:
    """Plot accuracy vs SNR curve."""
    snrs = sorted(int(snr) for snr in snr_accuracy.keys())
    accuracies = [snr_accuracy[str(snr)] for snr in snrs]

    plt.figure(figsize=(10, 6))
    plt.plot(snrs, accuracies, "b-o", linewidth=2, markersize=6)
    plt.xlabel("SNR (dB)")
    plt.ylabel("Classification Accuracy")
    plt.title("Accuracy vs SNR")
    plt.grid(True, alpha=0.3)
    plt.ylim([0, 1.05])

    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_class_accuracy(per_class_accuracy: Dict[str, float], save_path: Path) -> None:
    """Plot per-class accuracy as bar chart."""
    class_names = list(per_class_accuracy.keys())
    accuracies = list(per_class_accuracy.values())

    plt.figure(figsize=(14, 6))
    bars = plt.bar(range(len(class_names)), accuracies, color="steelblue")
    plt.xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    plt.xlabel("Modulation Type")
    plt.ylabel("Accuracy")
    plt.title("Per-Class Classification Accuracy")
    plt.ylim([0, 1.05])

    for bar, accuracy in zip(bars, accuracies):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.01,
            f"{accuracy:.1%}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.tight_layout()
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
