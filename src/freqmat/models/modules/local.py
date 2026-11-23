"""Local sequential branch: causal depthwise convolution followed by a GRU."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalGRUBranch(nn.Module):
    """Causal conv (left padding) -> SiLU -> GRU. Padding sits at the end of each sequence, so the
    states at real positions do not depend on it."""

    def __init__(self, hidden: int, kernel: int = 4):
        super().__init__()
        self.conv = nn.Conv1d(hidden, hidden, kernel, groups=hidden, padding=kernel - 1)
        self.gru = nn.GRU(hidden, hidden, batch_first=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        local = F.silu(self.conv(x.transpose(1, 2))[..., :length].transpose(1, 2))
        return self.gru(local)[0]


def partial_flip_index(lengths: torch.Tensor, size: int, keep_last: int) -> torch.Tensor:
    """``(B, N)`` gather index: reverse the first ``len - keep_last`` real items, keep the rest in place.
    The permutation is an involution, so the same index flips in and out."""
    t = torch.arange(size, device=lengths.device)[None].expand(len(lengths), size)
    n = (lengths - keep_last).clamp_min(0)[:, None]
    return torch.where(t < n, n - 1 - t, t)
