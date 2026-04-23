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
    SE_POLICIES = {
        "all": {1, 2, 3, 4, 5, 6},
        "front": {1, 2, 3},
        "back": {4, 5, 6},
        "none": set(),
    }

    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 24,
        use_se: bool = True,
        se_policy: str = "all",
        use_dilation: bool = True,
    ):
        super().__init__()
        if se_policy not in self.SE_POLICIES:
            valid = ", ".join(sorted(self.SE_POLICIES))
            raise ValueError(f"Unsupported se_policy={se_policy!r}. Expected one of: {valid}")

        dilations = [1, 2, 3, 2, 2, 2, 1] if use_dilation else [1] * 7
        channels = [32, 48, 64, 72, 84, 96, 108]
        kernels = [7, 5, 7, 5, 3, 3, 3]
        se_layers = self.SE_POLICIES[se_policy] if use_se else set()

        layers = []
        current_channels = input_channels
        for index, (out_channels, kernel, dilation) in enumerate(zip(channels, kernels, dilations), start=1):
            layers.append(ConvBlock(current_channels, out_channels, kernel, dilation))
            if index in se_layers:
                layers.append(SEBlock(out_channels))
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
