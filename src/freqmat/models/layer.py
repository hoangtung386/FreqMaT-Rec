"""One FreqMaT block: frequency denoising, parallel Mamba / attention mixing, gated fusion, FFN / MoE."""

import torch
import torch.nn as nn

from freqmat.models.config import FreqMaTConfig
from freqmat.models.modules import FFN, FreqFilter, GatedFusion, Mamba3Block, MambaBlock, MoEFFN, RoPEAttention


class FreqMaTLayer(nn.Module):
    """Freq denoise -> [Mamba || RoPE-Attn] -> gated fusion -> FFN/MoE (post-LN residuals).

    ``freq_position="pre"``: the frequency filter runs sequentially before the other branches.
    ``freq_position="branch"``: the filter is a *parallel* branch (as in BSARec) and is fused by the
    learned gate together with Mamba and attention (3-way softmax gate).
    """

    def __init__(self, cfg: FreqMaTConfig):
        super().__init__()
        hidden, dropout, eps = cfg.hidden_size, cfg.hidden_dropout_prob, cfg.layer_norm_eps
        self.freq_kind = cfg.freq_kind
        self.freq_branch = cfg.freq_as_branch
        self.use_mamba, self.use_attn = cfg.use_mamba, cfg.use_attn
        if self.freq_kind != "none":
            self.freq = FreqFilter(hidden, cfg.max_seq_len, dropout, eps, self.freq_kind, cfg.freq_cutoff)
        if self.use_mamba:
            self.mamba = self._build_mamba(cfg)
        if self.use_attn:
            self.attn = RoPEAttention(hidden, cfg.n_heads, cfg.attn_dropout_prob, cfg.attn_window)
        if cfg.n_branches >= 2:
            self.fusion = GatedFusion(hidden, cfg.n_branches)
        self.drop = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(hidden, eps=eps)
        self.use_moe = cfg.ffn_type == "moe"
        if self.use_moe:  # expert size defaults to inner_size / n_experts -> ~same params as dense FFN
            expert_inner = cfg.moe_inner or cfg.inner_size // cfg.moe_experts
            self.ffn = MoEFFN(hidden, expert_inner, cfg.moe_experts, cfg.moe_topk, dropout)
        else:
            self.ffn = FFN(hidden, cfg.inner_size, dropout)
        self.norm2 = nn.LayerNorm(hidden, eps=eps)

    @staticmethod
    def _build_mamba(cfg: FreqMaTConfig) -> nn.Module:
        if cfg.mamba_type == "mamba3":
            return Mamba3Block(cfg.hidden_size, cfg.d_state, cfg.headdim, cfg.expand, cfg.chunk_size)
        return MambaBlock(cfg.hidden_size, cfg.d_state, cfg.d_conv, cfg.expand)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None, basis: torch.Tensor | None = None
    ) -> torch.Tensor:
        has_freq = self.freq_kind != "none"
        if has_freq and not self.freq_branch:
            x = self.freq(x, basis)
        residual = x
        branches = []
        if has_freq and self.freq_branch:
            branches.append(self.freq.filter(x, basis))
        if self.use_mamba:
            branches.append(self.mamba(x))
        if self.use_attn:
            branches.append(self.attn(x))
        mixed = branches[0] if len(branches) == 1 else self.fusion(*branches)
        x = self.norm1(residual + self.drop(mixed))
        ffn_out = self.ffn(x, mask) if self.use_moe else self.ffn(x)
        return self.norm2(x + ffn_out)
