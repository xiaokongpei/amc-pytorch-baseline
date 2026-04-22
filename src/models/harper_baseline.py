from __future__ import annotations

from torch import nn

from src.models.blocks import SEBlock, StatisticalPooling


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int):
        super().__init__()
        padding = ((kernel_size - 1) // 2) * dilation
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class HarperBaseline(nn.Module):
    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 24,
        use_se: bool = True,
        se_reduction: int = 2,
        use_dilation: bool = True,
    ):
        super().__init__()
        dilations = [1, 2, 3, 2, 2, 2, 1] if use_dilation else [1] * 7
        channels = [32, 48, 64, 72, 84, 96, 108]
        kernels = [7, 5, 7, 5, 3, 3, 3]

        layers = []
        current_channels = input_channels
        for index, (out_channels, kernel, dilation) in enumerate(zip(channels, kernels, dilations), start=1):
            layers.append(ConvBlock(current_channels, out_channels, kernel, dilation))
            if use_se and index < 7:
                layers.append(SEBlock(out_channels, reduction=se_reduction))
            current_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.pool = StatisticalPooling()
        self.classifier = nn.Sequential(
            nn.Linear(108 * 2, 128),
            nn.SELU(inplace=True),
            nn.Linear(128, 128),
            nn.SELU(inplace=True),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.classifier(x)
