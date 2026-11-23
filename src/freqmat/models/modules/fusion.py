"""Learned gating over parallel branches."""

import torch
import torch.nn as nn


class GatedFusion(nn.Module):
    """Learned gate over parallel branches.

    2 branches: sigmoid gate ``g*h1 + (1-g)*h2`` (element-wise).
    3 branches: softmax over branches per channel ``sum_i g_i * h_i`` (``g_1+g_2+g_3 = 1``).
    ``last_gate`` keeps the mean gate value per branch of the last forward pass (for analysis).
    """

    def __init__(self, hidden: int, n_branches: int = 2):
        super().__init__()
        if n_branches not in (2, 3):
            raise ValueError("GatedFusion supports 2 or 3 branches")
        self.n, self.hidden = n_branches, hidden
        self.gate = nn.Linear(hidden * n_branches, hidden * (1 if n_branches == 2 else n_branches))
        self.last_gate = None

    def forward(self, *branches: torch.Tensor) -> torch.Tensor:
        z = torch.cat(branches, -1)
        if self.n == 2:
            g = torch.sigmoid(self.gate(z))
            self.last_gate = torch.stack([g.mean(), 1 - g.mean()]).detach()
            return g * branches[0] + (1 - g) * branches[1]
        g = torch.softmax(self.gate(z).view(*z.shape[:-1], self.n, self.hidden), dim=-2)
        self.last_gate = g.mean(dim=tuple(range(g.dim() - 2)) + (-1,)).detach()
        return sum(g[..., i, :] * branches[i] for i in range(self.n))
