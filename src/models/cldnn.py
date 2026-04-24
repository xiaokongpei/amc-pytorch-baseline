from __future__ import annotations

from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


class CLDNNConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        pool_size: int = 2,
    ) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.block = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(kernel_size=pool_size, stride=pool_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class TemporalStatsPooling(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, time, channels)
        mean = x.mean(dim=1)
        max_values, _ = x.max(dim=1)
        return torch.cat([mean, max_values], dim=1)


class ShrinkDenoise1D(nn.Module):
    """Learnable channel-wise shrinkage denoise block for 1D feature maps."""

    def __init__(
        self,
        channels: int,
        shrinkage: str = "garrote",
        reduction: int = 4,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        shrinkage = shrinkage.lower()
        if shrinkage not in {"soft", "garrote"}:
            raise ValueError("ShrinkDenoise1D shrinkage must be 'soft' or 'garrote'.")

        hidden = max(channels // reduction, 1)
        self.shrinkage = shrinkage
        self.eps = eps
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)
        self.threshold = nn.Sequential(
            nn.Conv1d(channels * 2, hidden, kernel_size=1, bias=False),
            nn.BatchNorm1d(hidden),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        magnitude = x.abs()
        avg_stats = self.avg_pool(magnitude)
        max_stats = self.max_pool(magnitude)
        scale = self.threshold(torch.cat([avg_stats, max_stats], dim=1))
        tau = scale * avg_stats

        if self.shrinkage == "soft":
            return torch.sign(x) * F.relu(magnitude - tau)

        keep = magnitude > tau
        safe_denominator = torch.where(x >= 0, x + self.eps, x - self.eps)
        shrunk = x - (tau.square() / safe_denominator)
        return torch.where(keep, shrunk, torch.zeros_like(x))


class CLDNN(nn.Module):
    def __init__(
        self,
        input_channels: int = 2,
        num_classes: int = 24,
        conv_channels: Sequence[int] = (32, 64, 96, 128),
        conv_kernels: Sequence[int] = (7, 5, 5, 3),
        pool_sizes: Sequence[int] = (2, 2, 2, 2),
        lstm_hidden_size: int = 128,
        lstm_layers: int = 1,
        bidirectional: bool = True,
        classifier_hidden_dims: Sequence[int] = (128, 128),
        dropout: float = 0.2,
        denoise_type: str = "none",
        denoise_position: str = "after_conv",
        denoise_reduction: int = 4,
    ) -> None:
        super().__init__()
        if not (
            len(conv_channels) == len(conv_kernels) == len(pool_sizes) == 4
        ):
            raise ValueError("CLDNN expects exactly 4 convolution blocks.")
        if len(classifier_hidden_dims) != 2:
            raise ValueError("CLDNN expects exactly 2 classifier hidden layers.")
        denoise_type = denoise_type.lower()
        denoise_position = denoise_position.lower()
        if denoise_type not in {"none", "soft", "garrote"}:
            raise ValueError("denoise_type must be one of: none, soft, garrote.")
        if denoise_position != "after_conv":
            raise ValueError("CLDNN currently supports only denoise_position='after_conv'.")

        layers = []
        current_channels = input_channels
        for out_channels, kernel_size, pool_size in zip(conv_channels, conv_kernels, pool_sizes):
            layers.append(
                CLDNNConvBlock(
                    in_channels=current_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    pool_size=pool_size,
                )
            )
            current_channels = out_channels

        self.features = nn.Sequential(*layers)
        self.denoise = (
            nn.Identity()
            if denoise_type == "none"
            else ShrinkDenoise1D(
                channels=current_channels,
                shrinkage=denoise_type,
                reduction=denoise_reduction,
            )
        )
        self.lstm = nn.LSTM(
            input_size=current_channels,
            hidden_size=lstm_hidden_size,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if lstm_layers > 1 else 0.0,
        )
        self.pool = TemporalStatsPooling()

        lstm_output_size = lstm_hidden_size * (2 if bidirectional else 1)
        pooled_size = lstm_output_size * 2
        hidden_dim_1, hidden_dim_2 = [int(dim) for dim in classifier_hidden_dims]

        self.classifier = nn.Sequential(
            nn.Linear(pooled_size, hidden_dim_1),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim_1, hidden_dim_2),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim_2, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.denoise(x)
        x = x.transpose(1, 2)
        x, _ = self.lstm(x)
        x = self.pool(x)
        return self.classifier(x)
