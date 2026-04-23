import torch

from src.models.cldnn import CLDNN
from src.models.harper_baseline import HarperBaseline


def test_harper_baseline_output_shape():
    model = HarperBaseline(input_channels=2, num_classes=24, use_se=True, use_dilation=True)
    batch = torch.randn(8, 2, 1024)
    logits = model(batch)
    assert logits.shape == (8, 24)


def test_cldnn_output_shape():
    model = CLDNN(input_channels=2, num_classes=24)
    batch = torch.randn(8, 2, 1024)
    logits = model(batch)
    assert logits.shape == (8, 24)
