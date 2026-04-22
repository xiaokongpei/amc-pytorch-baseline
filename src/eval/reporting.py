from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import numpy as np

from src.eval.metrics import plot_confusion_matrix, plot_accuracy_vs_snr, plot_class_accuracy


def write_json(path: Path | str, payload: Dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_train_log(path: Path | str, history: List[Dict]) -> None:
    write_json(path, {"history": list(history)})


def write_summary(
    path: Path | str,
    metrics: Dict[str, float],
    snr_metrics: Dict[str, float],
    detailed_metrics: Dict = None,
    class_names: List[str] = None,
) -> None:
    """Write a comprehensive summary report in Markdown format."""
    lines = [
        "# Training Summary",
        "",
        "## Overall Metrics",
        "",
        f"- **Accuracy**: {metrics.get('accuracy', 0.0):.4f}",
        f"- **Loss**: {metrics.get('loss', 0.0):.4f}",
        f"- **Top-2 Accuracy**: {metrics.get('top2_accuracy', 0.0):.4f}",
        "",
        "## SNR-wise Accuracy",
        "",
        "| SNR (dB) | Accuracy |",
        "|----------|----------|",
    ]

    for snr, acc in sorted(snr_metrics.items(), key=lambda x: int(x[0])):
        lines.append(f"| {snr} | {acc:.4f} |")

    # Average SNR accuracy
    avg_acc = np.mean(list(snr_metrics.values()))
    lines.extend([
        "",
        f"**Average SNR Accuracy**: {avg_acc:.4f}",
    ])

    # Group accuracy
    if detailed_metrics:
        group_acc = detailed_metrics.get("group_accuracy", {})
        if group_acc:
            lines.extend([
                "",
                "## Group Accuracy",
                "",
                "| Group | Accuracy |",
                "|-------|----------|",
            ])
            for group, acc in group_acc.items():
                lines.append(f"| {group} | {acc:.4f} |")

        # Per-class accuracy
        per_class_acc = detailed_metrics.get("per_class_accuracy", {})
        if per_class_acc:
            lines.extend([
                "",
                "## Per-Class Accuracy",
                "",
                "| Class | Accuracy |",
                "|-------|----------|",
            ])
            for class_name, acc in per_class_acc.items():
                lines.append(f"| {class_name} | {acc:.4f} |")

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")


def generate_all_reports(
    run_dir: Path,
    metrics: Dict[str, float],
    snr_metrics: Dict[str, float],
    detailed_metrics: Dict,
    class_names: List[str] = None,
) -> None:
    """Generate all evaluation reports."""
    run_dir = Path(run_dir)

    # Basic reports
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "snr_metrics.json", snr_metrics)

    # Detailed reports
    if detailed_metrics:
        write_json(run_dir / "detailed_metrics.json", detailed_metrics)

        confusion_matrix = np.array(detailed_metrics.get("confusion_matrix", []))
        if confusion_matrix.size > 0 and class_names:
            # Save confusion matrix JSON
            write_json(
                run_dir / "confusion_matrix.json",
                {
                    "class_names": class_names,
                    "matrix": confusion_matrix.tolist(),
                    "description": "Rows = True labels, Columns = Predicted labels",
                }
            )
            # Plot confusion matrix
            plot_confusion_matrix(confusion_matrix, class_names, run_dir / "confusion_matrix.png")

        # Plot accuracy vs SNR
        plot_accuracy_vs_snr(snr_metrics, run_dir / "accuracy_vs_snr.png")

        # Plot per-class accuracy
        per_class_acc = detailed_metrics.get("per_class_accuracy", {})
        if per_class_acc:
            plot_class_accuracy(per_class_acc, run_dir / "class_accuracy.png")

        # SNR confusion matrices
        snr_cm = detailed_metrics.get("snr_confusion_matrices")
        if snr_cm:
            write_json(
                run_dir / "snr_analysis.json",
                {
                    "snr_accuracy": snr_metrics,
                    "snr_confusion_matrices": snr_cm,
                    "class_names": class_names,
                }
            )

    # Summary
    write_summary(
        run_dir / "summary.md",
        metrics,
        snr_metrics,
        detailed_metrics,
        class_names,
    )
