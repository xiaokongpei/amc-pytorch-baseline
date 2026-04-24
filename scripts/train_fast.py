"""
Training entry for the active PyTorch baseline using the fast .pt data format.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.datasets.fast_dataset import build_fast_dataloaders
from src.engine.checkpointing import load_checkpoint
from src.engine.trainer import fit
from src.eval.metrics import evaluate_model
from src.eval.reporting import generate_all_reports, write_train_log
from src.models.cldnn import CLDNN
from src.models.harper_baseline import HarperBaseline
from src.utils.config import apply_overrides, load_config
from src.utils.paths import make_run_name, prepare_run_dir
from src.utils.runtime import dump_json, resolve_device, set_seed


def _normalize_state_dict_keys(state_dict: dict) -> dict:
    if any(key.startswith("_orig_mod.") for key in state_dict):
        return {
            key.replace("_orig_mod.", "", 1): value
            for key, value in state_dict.items()
        }
    return state_dict


def _build_model(config: dict, device: torch.device) -> nn.Module:
    model_name = str(config["model"].get("name", "harper")).lower()

    if model_name == "harper":
        model = HarperBaseline(
            input_channels=int(config["model"]["input_channels"]),
            num_classes=int(config["model"]["num_classes"]),
            use_se=bool(config["model"]["use_se"]),
            se_policy=str(config["model"].get("se_policy", "all")),
            use_dilation=bool(config["model"]["use_dilation"]),
        )
    elif model_name == "cldnn":
        model = CLDNN(
            input_channels=int(config["model"]["input_channels"]),
            num_classes=int(config["model"]["num_classes"]),
            conv_channels=tuple(config["model"].get("conv_channels", [32, 64, 96, 128])),
            conv_kernels=tuple(config["model"].get("conv_kernels", [7, 5, 5, 3])),
            pool_sizes=tuple(config["model"].get("pool_sizes", [2, 2, 2, 2])),
            lstm_hidden_size=int(config["model"].get("lstm_hidden_size", 128)),
            lstm_layers=int(config["model"].get("lstm_layers", 1)),
            bidirectional=bool(config["model"].get("bidirectional", True)),
            classifier_hidden_dims=tuple(config["model"].get("classifier_hidden_dims", [128, 128])),
            dropout=float(config["model"].get("dropout", 0.2)),
            denoise_type=str(config["model"].get("denoise_type", "none")),
            denoise_position=str(config["model"].get("denoise_position", "after_conv")),
            denoise_reduction=int(config["model"].get("denoise_reduction", 4)),
        )
    else:
        raise ValueError(f"Unsupported model.name={model_name!r}")

    return model.to(device)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/fast.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--mode", choices=["mmap", "preload", "gpu"], default=None)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    config = apply_overrides(config, args.override)

    if args.data_root is not None:
        config["data"]["root"] = args.data_root
    if args.device is not None:
        config["runtime"]["device"] = args.device

    mode = args.mode or config["data"].get("mode", "mmap")

    set_seed(int(config["training"]["seed"]))
    device = resolve_device(config["runtime"]["device"])
    run_name = args.run_name or make_run_name(config["experiment_name"])
    run_dir = prepare_run_dir(config["runtime"]["run_root"], run_name)

    print(f"Device: {device}")
    print(f"Mode: {mode}")
    print(f"Run dir: {run_dir}")
    print("Loading data...")
    dataloaders = build_fast_dataloaders(config, mode=mode)
    print(f"Train: {len(dataloaders['train'].dataset)}")
    print(f"Val: {len(dataloaders['val'].dataset)}")
    print(f"Test: {len(dataloaders['test'].dataset)}")

    model = _build_model(config, device)

    torch.set_float32_matmul_precision("high")
    model = torch.compile(model)
    print("Enabled TF32 and torch.compile()")

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
        early_stopping_patience=int(config["training"].get("early_stopping_patience", 10)),
        use_amp=bool(config["training"].get("use_amp", True)),
    )
    write_train_log(run_dir / "train_log.json", history)

    checkpoint = load_checkpoint(run_dir / "checkpoints" / "best.pt", map_location=device)
    eval_model = _build_model(config, device)
    eval_model.load_state_dict(_normalize_state_dict_keys(checkpoint["model_state_dict"]))
    eval_model.eval()

    class_names_path = Path(config["data"]["root"]) / "metadata" / "classes-fixed.json"
    class_names = None
    if class_names_path.exists():
        class_names = json.loads(class_names_path.read_text())

    print("\nEvaluating model on test set...")
    metrics, snr_metrics, detailed_metrics = evaluate_model(
        eval_model,
        dataloaders["test"],
        device,
        num_classes=int(config["model"]["num_classes"]),
        class_names=class_names,
    )

    generate_all_reports(run_dir, metrics, snr_metrics, detailed_metrics, class_names)

    print(f"\nDone: {run_dir}")


if __name__ == "__main__":
    main()
