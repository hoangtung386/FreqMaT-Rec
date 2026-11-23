"""Causal multi-head self-attention with rotary position embeddings."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_rope(x: torch.Tensor, base: float = 10000.0) -> torch.Tensor:
    """Rotary embedding. ``x``: ``(b, h, l, d)`` with even ``d``."""
    length, dim = x.shape[-2], x.shape[-1]
    inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, device=x.device).float() / dim))
    angle = torch.arange(length, device=x.device).float()[:, None] * inv_freq[None]
    cos, sin = angle.cos().to(x.dtype), angle.sin().to(x.dtype)
    x1, x2 = x[..., 0::2], x[..., 1::2]
    return torch.stack([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1).flatten(-2)


class RoPEAttention(nn.Module):
    """Causal multi-head self-attention with rotary positions and an optional local window."""

    def __init__(self, hidden: int, heads: int, dropout: float, window: int = 0):
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden must be divisible by heads")
        self.heads, self.head_dim, self.window = heads, hidden // heads, window
        self.qkv = nn.Linear(hidden, hidden * 3)
        self.out = nn.Linear(hidden, hidden)
        self.dropout = dropout

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, length, _ = x.shape
        q, k, v = self.qkv(x).view(batch, length, 3, self.heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k = apply_rope(q), apply_rope(k)
        pos = torch.arange(length, device=x.device)
        mask = pos[None, :] <= pos[:, None]
        if self.window > 0:
            mask &= (pos[:, None] - pos[None, :]) < self.window
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask, dropout_p=self.dropout if self.training else 0.0)
        return self.out(y.transpose(1, 2).reshape(batch, length, -1))
