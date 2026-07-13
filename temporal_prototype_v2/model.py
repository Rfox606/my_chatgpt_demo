from __future__ import annotations

import torch
from torch import nn


def ordinal_probabilities(logits: torch.Tensor) -> torch.Tensor:
    survival = torch.sigmoid(logits)
    # CORAL thresholds must be cumulative P(y > threshold) values.
    survival = torch.cummin(survival, dim=-1).values
    probs = torch.cat((1.0 - survival[:, :1], survival[:, :-1] - survival[:, 1:], survival[:, -1:]), dim=-1)
    return probs.clamp_min(1e-7) / probs.sum(dim=-1, keepdim=True).clamp_min(1e-7)


class TemporalPrototypeNet(nn.Module):
    def __init__(self, input_dim: int = 17) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, 32)
        self.norm = nn.LayerNorm(32)
        self.activation = nn.GELU()
        self.gru = nn.GRU(32, 32, batch_first=True, bidirectional=False)
        self.adapter_down = nn.Linear(32, 8)
        self.adapter_up = nn.Linear(8, 32)
        self.adapter_scale = nn.Parameter(torch.zeros(()))
        self.embedding_head = nn.Linear(32, 16)
        self.ordinal_base = nn.Linear(16, 1)
        self.ordinal_thresholds = nn.Parameter(torch.linspace(1.5, -1.5, 4))
        self.ordinal_bias = nn.Parameter(torch.zeros(4))
        self.health_head = nn.Linear(16, 1)

    def encode(self, sequence: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.activation(self.norm(self.input_projection(sequence)))
        hidden, _ = self.gru(x)
        h = hidden[:, -1, :]
        adapter = self.adapter_up(self.activation(self.adapter_down(h)))
        h = h + self.adapter_scale * adapter
        embedding = nn.functional.normalize(self.embedding_head(h), dim=-1)
        return embedding, h

    def forward(self, sequence: torch.Tensor) -> dict[str, torch.Tensor]:
        embedding, hidden = self.encode(sequence)
        base = self.ordinal_base(embedding)
        logits = base + self.ordinal_thresholds.unsqueeze(0) + self.ordinal_bias.unsqueeze(0)
        return {
            "embedding": embedding,
            "hidden": hidden,
            "ordinal_logits": logits,
            "stage_probs": ordinal_probabilities(logits),
            "health": torch.sigmoid(self.health_head(embedding)).squeeze(-1),
        }

    def assert_unidirectional(self) -> None:
        assert not self.gru.bidirectional, "Bidirectional recurrent encoders are forbidden."

    def online_parameters(self, tent_lite: bool = False) -> list[nn.Parameter]:
        for parameter in self.parameters():
            parameter.requires_grad = False
        if tent_lite:
            self.norm.weight.requires_grad = True
            self.norm.bias.requires_grad = True
            self.ordinal_bias.requires_grad = True
            return [self.norm.weight, self.norm.bias, self.ordinal_bias]
        for module in (self.adapter_down, self.adapter_up, self.embedding_head):
            for parameter in module.parameters():
                parameter.requires_grad = True
        self.adapter_scale.requires_grad = True
        self.ordinal_bias.requires_grad = True
        return [p for p in self.parameters() if p.requires_grad]
