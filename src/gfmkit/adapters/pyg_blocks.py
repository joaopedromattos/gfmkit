from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import BatchNorm1d, Linear, ReLU, Sequential
from torch_geometric.nn import GINConv


class GINEncoder(nn.Module):
    """Two-layer GIN encoder used by the local LP baselines."""

    def __init__(self, in_channels: int = 128, hidden_channels: int = 128, out_channels: int = 64, train_eps: bool = True):
        super().__init__()
        mlp1 = Sequential(
            Linear(in_channels, hidden_channels),
            ReLU(),
            Linear(hidden_channels, hidden_channels),
        )
        mlp2 = Sequential(
            Linear(hidden_channels, out_channels),
            ReLU(),
            Linear(out_channels, out_channels),
        )
        self.conv1 = GINConv(mlp1, train_eps=train_eps)
        self.bn1 = BatchNorm1d(hidden_channels)
        self.conv2 = GINConv(mlp2, train_eps=train_eps)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x, edge_index)
        x = self.bn1(x)
        x = F.relu(x)
        return self.conv2(x, edge_index)


class PromptedGINEncoder(GINEncoder):
    """GIN encoder that supports GPF-style input prompting."""

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, prompt=None) -> torch.Tensor:
        if prompt is not None:
            x = prompt.add(x)
        return super().forward(x, edge_index)


class SimplePrompt(nn.Module):
    """Learn one additive feature prompt shared by all nodes."""

    def __init__(self, dim: int):
        super().__init__()
        self.prompt = nn.Parameter(torch.zeros(1, dim))

    def add(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.prompt


class GPFPlusPrompt(nn.Module):
    """Small GPF-plus style basis prompt with feature-conditioned attention."""

    def __init__(self, dim: int, pnum: int = 5):
        super().__init__()
        self.basis = nn.Parameter(torch.empty(pnum, dim))
        self.scorer = nn.Linear(dim, pnum)
        nn.init.xavier_uniform_(self.basis)

    def add(self, x: torch.Tensor) -> torch.Tensor:
        weights = torch.softmax(self.scorer(x), dim=-1)
        return x + weights @ self.basis


class FeatureWeightedPrompt(nn.Module):
    """GraphPrompt-style per-feature output reweighting."""

    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return z * self.weight


class LinearPrompt(nn.Module):
    """GraphPrompt-style learnable linear output transform."""

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.linear(z)

