"""Position-wise feed-forward networks: dense and mixture-of-experts."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FFN(nn.Module):
    def __init__(self, hidden: int, inner: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden, inner), nn.GELU(), nn.Dropout(dropout), nn.Linear(inner, hidden), nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MoEFFN(nn.Module):
    """Mixture-of-Experts feed-forward: top-k routed expert FFNs + Switch-style load-balancing loss.

    Reference implementation: experts are evaluated densely (E is small) and combined with sparse top-k
    gates, so the result equals sparse routing but compute scales with E. ``self.aux_loss`` holds the
    balance loss of the last forward pass.
    """

    def __init__(self, hidden: int, expert_inner: int, n_experts: int, top_k: int, dropout: float):
        super().__init__()
        self.n_experts, self.top_k = n_experts, top_k
        self.router = nn.Linear(hidden, n_experts, bias=False)
        self.w1 = nn.Parameter(torch.randn(n_experts, hidden, expert_inner) * 0.02)
        self.b1 = nn.Parameter(torch.zeros(n_experts, expert_inner))
        self.w2 = nn.Parameter(torch.randn(n_experts, expert_inner, hidden) * 0.02)
        self.b2 = nn.Parameter(torch.zeros(n_experts, hidden))
        self.drop = nn.Dropout(dropout)
        self.aux_loss = torch.zeros(())

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        probs = F.softmax(self.router(x).float(), dim=-1)  # (b, l, E)
        top_p, top_i = probs.topk(self.top_k, dim=-1)
        gates = torch.zeros_like(probs).scatter(-1, top_i, top_p / top_p.sum(-1, keepdim=True))
        # load-balancing loss over real (non-padding) tokens: E * sum_e f_e * P_e
        valid = torch.ones_like(probs[..., 0]) if mask is None else mask.float()
        n_tokens = valid.sum().clamp_min(1.0)
        frac = (F.one_hot(top_i, self.n_experts).sum(-2).float() * valid.unsqueeze(-1)).sum((0, 1)) / (
            n_tokens * self.top_k
        )
        mean_prob = (probs * valid.unsqueeze(-1)).sum((0, 1)) / n_tokens
        self.aux_loss = self.n_experts * (frac * mean_prob).sum()
        h = F.gelu(torch.einsum("bld,edi->blei", x, self.w1) + self.b1)
        y = torch.einsum("blei,eih->bleh", self.drop(h), self.w2) + self.b2
        return self.drop((y * gates.unsqueeze(-1).to(y.dtype)).sum(2))


class SharedMoEFFN(nn.Module):
    """``y = shared(x) + sigmoid(a) * ramp * routed(x)``: an always-on dense expert plus routed experts.

    ``ramp`` grows linearly from 0 to 1 over ``warmup_steps`` training steps (0 disables the ramp).
    """

    def __init__(
        self,
        hidden: int,
        shared_inner: int,
        expert_inner: int,
        n_experts: int,
        top_k: int,
        dropout: float,
        warmup_steps: int = 0,
    ):
        super().__init__()
        self.shared = FFN(hidden, shared_inner, dropout)
        self.routed = MoEFFN(hidden, expert_inner, n_experts, top_k, dropout)
        self.alpha_raw = nn.Parameter(torch.zeros(()))
        self.warmup = warmup_steps
        self.register_buffer("step", torch.zeros((), dtype=torch.long))

    @property
    def aux_loss(self) -> torch.Tensor:
        return self.routed.aux_loss

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.training:
            self.step += 1
        ramp = 1.0 if self.warmup <= 0 else (self.step.float() / self.warmup).clamp(max=1.0)
        return self.shared(x) + torch.sigmoid(self.alpha_raw) * ramp * self.routed(x, mask)
