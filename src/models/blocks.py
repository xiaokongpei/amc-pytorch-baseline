from __future__ import annotations

import torch
from torch import nn


class SEBlock(nn.Module):
    def __init__(self, channels: int, reduction: int = 2):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.reduce = nn.Linear(channels, hidden, bias=False)
        self.act = nn.ReLU(inplace=True)
        self.expand = nn.Linear(hidden, channels, bias=False)
        self.gate = nn.Sigmoid()
        nn.init.kaiming_normal_(self.reduce.weight, nonlinearity="relu")
        nn.init.kaiming_normal_(self.expand.weight, nonlinearity="relu")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.pool(x).squeeze(-1)
        weights = self.reduce(weights)
        weights = self.act(weights)
        weights = self.expand(weights)
        weights = self.gate(weights).unsqueeze(-1)
        return x * weights


class StatisticalPooling(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=-1)
        var = x.var(dim=-1, unbiased=False)
        return torch.cat([mean, var], dim=1)
