from __future__ import annotations

from datetime import datetime
from pathlib import Path


def make_run_name(prefix: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{stamp}"


def prepare_run_dir(run_root: str | Path, run_name: str) -> Path:
    run_dir = Path(run_root) / run_name
    checkpoints = run_dir / "checkpoints"
    checkpoints.mkdir(parents=True, exist_ok=False)
    return run_dir


def resolve_repo_relative(path_like: str | Path, repo_root: str | Path) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return Path(repo_root) / path
