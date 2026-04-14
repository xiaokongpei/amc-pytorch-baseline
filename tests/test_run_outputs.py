from pathlib import Path

from src.eval.reporting import write_json, write_summary, write_train_log


def test_expected_run_artifact_names(tmp_path: Path):
    write_json(tmp_path / "metrics.json", {"accuracy": 0.5})
    write_json(tmp_path / "snr_metrics.json", {"0": 0.5})
    write_train_log(tmp_path / "train_log.json", [{"epoch": 1}])
    write_summary(tmp_path / "summary.md", {"accuracy": 0.5, "loss": 1.0}, {"0": 0.5})

    expected = {
        "metrics.json",
        "snr_metrics.json",
        "train_log.json",
        "summary.md",
    }
    actual = {path.name for path in tmp_path.iterdir()}
    assert actual == expected
