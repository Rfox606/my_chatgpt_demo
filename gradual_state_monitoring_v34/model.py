from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn

from .config import GradualStateMonitoringV34Config


class ResidualAdapter(nn.Module):
    """The shared/source or independent/target 6 -> 16 -> 6 residual adapter."""

    def __init__(self, config: GradualStateMonitoringV34Config) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(6, config.adapter_hidden_dim), nn.ReLU(),
            nn.Linear(config.adapter_hidden_dim, 6),
        )
        # An identity adapter is a stable starting point while still trainable.
        nn.init.zeros_(self.network[-1].weight)
        nn.init.zeros_(self.network[-1].bias)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return values + self.network(values)


class CausalResidualBlock(nn.Module):
    def __init__(self, inputs: int, outputs: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        padding = dilation * (kernel_size - 1)
        self.padding = padding
        self.conv1 = nn.Conv1d(inputs, outputs, kernel_size=kernel_size, dilation=dilation)
        self.conv2 = nn.Conv1d(outputs, outputs, kernel_size=kernel_size, dilation=dilation)
        self.residual = nn.Identity() if inputs == outputs else nn.Conv1d(inputs, outputs, kernel_size=1)
        self.activation = nn.ReLU()
        self.dropout = nn.Dropout(dropout)

    def _causal(self, value: torch.Tensor, layer: nn.Conv1d) -> torch.Tensor:
        return layer(torch.nn.functional.pad(value, (self.padding, 0)))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        residual = self.residual(values)
        output = self.dropout(self.activation(self._causal(values, self.conv1)))
        output = self.dropout(self.activation(self._causal(output, self.conv2)))
        return self.activation(output + residual)


class CausalTCNEncoder(nn.Module):
    def __init__(self, config: GradualStateMonitoringV34Config) -> None:
        super().__init__()
        if len(config.tcn_channels) != len(config.tcn_dilations):
            raise ValueError("TCN channels and dilations must have equal lengths")
        layers: list[nn.Module] = []
        previous = 6
        for channels, dilation in zip(config.tcn_channels, config.tcn_dilations):
            layers.append(CausalResidualBlock(previous, channels, config.kernel_size, dilation, config.dropout))
            previous = channels
        self.blocks = nn.ModuleList(layers)
        self.projection = nn.Linear(previous, config.embedding_dim)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        output = values.transpose(1, 2)
        for block in self.blocks:
            output = block(output)
        return self.projection(output[:, :, -1])


class GradualStateTCN(nn.Module):
    """Causal encoder plus source transition and h=20 delta heads."""

    def __init__(self, config: GradualStateMonitoringV34Config) -> None:
        super().__init__()
        self.config = config
        self.adapter = ResidualAdapter(config)
        self.encoder = CausalTCNEncoder(config)
        self.transition_head = nn.Linear(config.embedding_dim, 1)
        self.prediction_head = nn.Linear(config.embedding_dim, 6)

    def encode(self, history: torch.Tensor, *, adapter: nn.Module | None = None) -> torch.Tensor:
        return self.encoder((adapter or self.adapter)(history))

    def forward(self, history: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        embedding = self.encode(history)
        return embedding, self.transition_head(embedding).squeeze(-1), self.prediction_head(embedding)

    def target_copy(self) -> "GradualStateTCN":
        """Copy source weights but create a physically independent target adapter."""
        copied = deepcopy(self)
        copied.adapter = deepcopy(self.adapter)
        return copied

    def freeze_for_target_adaptation(self) -> None:
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)
        for parameter in self.transition_head.parameters():
            parameter.requires_grad_(False)
        for parameter in self.prediction_head.parameters():
            parameter.requires_grad_(False)
        for parameter in self.adapter.parameters():
            parameter.requires_grad_(True)


def set_deterministic_seed(seed: int) -> None:
    import numpy as np

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.set_num_threads(1)
