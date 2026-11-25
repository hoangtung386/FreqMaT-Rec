"""Normalisation layers."""

import torch
import torch.nn as nn

NORM_TYPES = ("rmsnorm", "layernorm")


class RMSNorm(nn.Module):
    """``x / rms(x) * weight`` (no mean subtraction, no bias), as in LLaMA-style Transformers."""

    def __init__(self, hidden: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        scale = torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.eps)
        return (x.float() * scale).to(x.dtype) * self.weight


def make_norm(kind: str, hidden: int, eps: float) -> nn.Module:
    """``RMSNorm`` or ``LayerNorm``; RMSNorm uses at least 1e-6 because the layer-norm default (1e-12) is unstable."""
    if kind == "rmsnorm":
        return RMSNorm(hidden, max(eps, 1e-6))
    return nn.LayerNorm(hidden, eps=eps)


class OutputHead(nn.Module):
    """Residual MLP on the final state before it is scored against the item embeddings: ``x + MLP(norm(x))``."""

    def __init__(self, hidden: int, dropout: float, norm: nn.Module):
        super().__init__()
        self.norm = norm
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden * 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden * 2, hidden)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(self.norm(x))
