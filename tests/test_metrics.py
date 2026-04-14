import torch

from src.eval.metrics import compute_accuracy, compute_snr_accuracy


def test_tensor_layout_for_conv1d():
    batch = torch.randn(4, 2, 1024)
    assert batch.shape == (4, 2, 1024)


def test_metric_helpers():
    accuracy = compute_accuracy([0, 1, 2], [0, 2, 2])
    snr_accuracy = compute_snr_accuracy([0, 1, 2], [0, 2, 2], [0, 0, 10])
    assert accuracy == 2 / 3
    assert snr_accuracy["0"] == 0.5
    assert snr_accuracy["10"] == 1.0
