from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable


def write_json(path: str | Path, payload: Dict) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_train_log(path: str | Path, history: Iterable[Dict]) -> None:
    write_json(path, {"history": list(history)})


def write_summary(path: str | Path, metrics: Dict[str, float], snr_metrics: Dict[str, float]) -> None:
    lines = [
        "# Run Summary",
        "",
        f"- accuracy: {metrics.get('accuracy', 0.0):.4f}",
        f"- loss: {metrics.get('loss', 0.0):.4f}",
        "",
        "## SNR Accuracy",
    ]
    lines.extend(f"- {snr}: {value:.4f}" for snr, value in snr_metrics.items())
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("\n".join(lines) + "\n", encoding="utf-8")
