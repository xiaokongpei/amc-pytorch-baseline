from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.radioml_dataset import build_dataloaders
from src.engine.checkpointing import load_checkpoint
from src.engine.trainer import fit
from src.eval.metrics import evaluate_model
from src.eval.reporting import write_json, write_summary, write_train_log
from src.models.harper_baseline import HarperBaseline
from src.utils.config import apply_overrides, load_config
from src.utils.paths import make_run_name, prepare_run_dir
from src.utils.runtime import dump_json, resolve_device, set_seed


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/baseline.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--metadata-dir", default=None)
    parser.add_argument("--run-name", default=None)
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

    set_seed(int(config["training"]["seed"]))
    device = resolve_device(config["runtime"]["device"])
    run_name = args.run_name or make_run_name(config["experiment_name"])

    if args.output_dir is not None:
        run_dir = Path(args.output_dir)
        (run_dir / "checkpoints").mkdir(parents=True, exist_ok=False)
    else:
        run_dir = prepare_run_dir(config["runtime"]["run_root"], run_name)

    dataloaders = build_dataloaders(config)
    model = HarperBaseline(
        input_channels=int(config["model"]["input_channels"]),
        num_classes=int(config["model"]["num_classes"]),
        use_se=bool(config["model"]["use_se"]),
        use_dilation=bool(config["model"]["use_dilation"]),
    ).to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(config["training"]["lr"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(config["training"]["scheduler_factor"]),
        patience=int(config["training"]["scheduler_patience"]),
        min_lr=float(config["training"]["min_lr"]),
    )

    dump_json(run_dir / "config.json", config)
    history = fit(
        model=model,
        train_loader=dataloaders["train"],
        val_loader=dataloaders["val"],
        optimizer=optimizer,
        scheduler=scheduler,
        epochs=int(config["training"]["epochs"]),
        device=device,
        run_dir=run_dir,
    )
    write_train_log(run_dir / "train_log.json", history)

    checkpoint = load_checkpoint(run_dir / "checkpoints" / "best.pt", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    metrics, snr_metrics = evaluate_model(model, dataloaders["test"], device)
    write_json(run_dir / "metrics.json", metrics)
    write_json(run_dir / "snr_metrics.json", snr_metrics)
    write_summary(run_dir / "summary.md", metrics, snr_metrics)


if __name__ == "__main__":
    main()
