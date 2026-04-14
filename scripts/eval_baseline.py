from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.radioml_dataset import build_dataloaders
from src.engine.checkpointing import load_checkpoint
from src.eval.metrics import evaluate_model
from src.eval.reporting import write_json, write_summary
from src.models.harper_baseline import HarperBaseline
from src.utils.config import apply_overrides, load_config
from src.utils.runtime import resolve_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--metadata-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.override)

    if args.data_root is not None:
        config["data"]["root"] = args.data_root
    if args.metadata_dir is not None:
        config["data"]["metadata_dir"] = args.metadata_dir
    if args.device is not None:
        config["runtime"]["device"] = args.device

    device = resolve_device(config["runtime"]["device"])
    dataloaders = build_dataloaders(config)
    model = HarperBaseline(
        input_channels=int(config["model"]["input_channels"]),
        num_classes=int(config["model"]["num_classes"]),
        use_se=bool(config["model"]["use_se"]),
        use_dilation=bool(config["model"]["use_dilation"]),
    ).to(device)

    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics, snr_metrics = evaluate_model(model, dataloaders["test"], device)

    output_dir = Path(args.output_dir) if args.output_dir else Path(args.checkpoint).resolve().parents[1]
    write_json(output_dir / "metrics.json", metrics)
    write_json(output_dir / "snr_metrics.json", snr_metrics)
    write_summary(output_dir / "summary.md", metrics, snr_metrics)


if __name__ == "__main__":
    main()
