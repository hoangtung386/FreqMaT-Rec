"""Learned gating over parallel branches."""

import torch
import torch.nn as nn

GATE_GRANULARITIES = ("channel", "branch")


class GatedFusion(nn.Module):
    """Learned gate over ``n >= 2`` parallel branches.

    2 branches: sigmoid gate ``g*h1 + (1-g)*h2``; 3+ branches: softmax over branches, ``sum_i g_i h_i``.
    ``granularity="channel"``: one gate value per channel (``Linear(nD, D*k)``); ``"branch"``: one scalar
    per branch and token (``Linear(nD, k)``), far fewer parameters for many branches.
    ``last_gate`` keeps the mean gate value per branch of the last forward pass (for analysis).
    """

    def __init__(self, hidden: int, n_branches: int = 2, granularity: str = "channel"):
        super().__init__()
        if n_branches < 2:
            raise ValueError("GatedFusion needs at least 2 branches")
        if granularity not in GATE_GRANULARITIES:
            raise ValueError(f"granularity must be one of {GATE_GRANULARITIES}, got {granularity!r}")
        self.n, self.hidden, self.granularity = n_branches, hidden, granularity
        per = hidden if granularity == "channel" else 1
        self.per = per
        self.gate = nn.Linear(hidden * n_branches, per * (1 if n_branches == 2 else n_branches))
        self.last_gate = None

    def forward(self, *branches: torch.Tensor) -> torch.Tensor:
        z = torch.cat(branches, -1)
        if self.n == 2:
            g = torch.sigmoid(self.gate(z))
            self.last_gate = torch.stack([g.mean(), 1 - g.mean()]).detach()
            return g * branches[0] + (1 - g) * branches[1]
        g = torch.softmax(self.gate(z).view(*z.shape[:-1], self.n, self.per), dim=-2)
        self.last_gate = g.mean(dim=tuple(range(g.dim() - 2)) + (-1,)).detach()
        return sum(g[..., i, :] * branches[i] for i in range(self.n))
