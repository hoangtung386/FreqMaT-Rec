"""One FreqMaT block: frequency denoising, parallel Mamba / attention mixing, gated fusion, FFN / MoE."""

import torch
import torch.nn as nn

from freqmat.models.config import FreqMaTConfig
from freqmat.models.modules import (
    FFN,
    FreqFilter,
    GatedFusion,
    LocalGRUBranch,
    Mamba3Block,
    MambaBlock,
    MoEFFN,
    OscFilter,
    RoPEAttention,
    SharedMoEFFN,
    make_norm,
    partial_flip_index,
)


class FreqMaTLayer(nn.Module):
    """Freq denoise -> [Mamba || RoPE-Attn || ...] -> gated fusion -> FFN/MoE, with pre- or post-norm residuals.

    ``norm_position="pre"`` (default): ``x = x + mix(norm1(x)); x = x + ffn(norm2(x))``; the model applies one final
    norm after the last layer. ``"post"``: ``x = norm1(x + mix(x)); x = norm2(x + ffn(x))``.

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
            if self.freq_kind == "osc":
                self.freq = OscFilter(hidden, cfg.max_seq_len, dropout, eps, cfg.osc_hf, cfg.osc_beta_init)
            else:
                self.freq = FreqFilter(hidden, cfg.max_seq_len, dropout, eps, self.freq_kind, cfg.freq_cutoff)
        if self.use_mamba:
            self.mamba = self._build_mamba(cfg)
        self.use_flip, self.keep_last = cfg.flip_active, cfg.flip_keep_last
        self.flip_shared = cfg.flip_share_weights
        if self.use_flip and not self.flip_shared:
            self.mamba_flip = self._build_mamba(cfg)
        self.use_local = cfg.use_local
        if self.use_local:
            self.local = LocalGRUBranch(hidden, cfg.local_kernel)
        if self.use_attn:
            self.attn = RoPEAttention(hidden, cfg.n_heads, cfg.attn_dropout_prob, cfg.attn_window)
        if cfg.n_branches >= 2:
            self.fusion = GatedFusion(hidden, cfg.n_branches, cfg.gate_granularity)
        self.drop = nn.Dropout(dropout)
        self.pre_norm = cfg.norm_position == "pre"
        self.norm1 = make_norm(cfg.norm_type, hidden, eps)
        self.use_moe = cfg.ffn_type == "moe"
        if self.use_moe and cfg.moe_shared:
            shared_inner = cfg.moe_shared_inner or cfg.inner_size // 2
            expert_inner = cfg.moe_inner or max(1, (cfg.inner_size - shared_inner) // cfg.moe_experts)
            self.ffn = SharedMoEFFN(
                hidden,
                shared_inner,
                expert_inner,
                cfg.moe_experts,
                cfg.moe_topk,
                dropout,
                cfg.moe_warmup_steps,
                cfg.ffn_act,
            )
        elif self.use_moe:  # expert size defaults to inner_size / n_experts -> ~same params as dense FFN
            expert_inner = cfg.moe_inner or cfg.inner_size // cfg.moe_experts
            self.ffn = MoEFFN(hidden, expert_inner, cfg.moe_experts, cfg.moe_topk, dropout, cfg.ffn_act)
        else:
            self.ffn = FFN(hidden, cfg.inner_size, dropout, cfg.ffn_act)
        self.norm2 = make_norm(cfg.norm_type, hidden, eps)

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
        h = self.norm1(x) if self.pre_norm else x
        branches = []
        if has_freq and self.freq_branch:
            branches.append(self.freq.filter(h, basis))
        if self.use_mamba:
            branches.append(self.mamba(h))
            if self.use_flip:
                branches.append(self._flipped(h, mask))
        if self.use_attn:
            branches.append(self.attn(h))
        if self.use_local:
            branches.append(self.local(h))
        mixed = branches[0] if len(branches) == 1 else self.fusion(*branches)
        x = residual + self.drop(mixed)
        if not self.pre_norm:
            x = self.norm1(x)
        h = self.norm2(x) if self.pre_norm else x
        ffn_out = self.ffn(h, mask) if self.use_moe else self.ffn(h)
        x = x + ffn_out
        return x if self.pre_norm else self.norm2(x)

    def _flipped(self, x: torch.Tensor, mask: torch.Tensor | None) -> torch.Tensor:
        """Mamba over the partially flipped sequence, flipped back so positions line up with ``x``."""
        lengths = x.new_full((x.shape[0],), x.shape[1], dtype=torch.long) if mask is None else mask.sum(1)
        idx = partial_flip_index(lengths, x.shape[1], self.keep_last)[..., None].expand_as(x)
        block = self.mamba if self.flip_shared else self.mamba_flip
        return block(x.gather(1, idx)).gather(1, idx)
