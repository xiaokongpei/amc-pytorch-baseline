import torch

from src.models.harper_baseline import HarperBaseline


def test_harper_baseline_output_shape():
    model = HarperBaseline(input_channels=2, num_classes=24, use_se=True, use_dilation=True)
    batch = torch.randn(8, 2, 1024)
    logits = model(batch)
    assert logits.shape == (8, 24)
