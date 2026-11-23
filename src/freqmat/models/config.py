"""Typed, validated hyper-parameters of the FreqMaT architecture."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

FREQ_TYPES = ("none", "learned", "bsa", "both", "gft", "osc")
FREQ_POSITIONS = ("pre", "branch")
MAMBA_TYPES = ("mamba1", "mamba3")
FFN_TYPES = ("dense", "moe")
GATE_GRANULARITIES = ("channel", "branch")


@dataclass(frozen=True)
class FreqMaTConfig:
    """Architecture options. Defaults mirror ``configs/model/freqmat.yaml`` except ``mamba_type``,
    which defaults to the dependency-free ``mamba1`` (the YAML selects ``mamba3``)."""

    hidden_size: int = 256
    inner_size: int = 1024
    n_layers: int = 3
    max_seq_len: int = 50
    hidden_dropout_prob: float = 0.3
    attn_dropout_prob: float = 0.1
    layer_norm_eps: float = 1e-12
    # feed-forward
    ffn_type: str = "moe"
    moe_experts: int = 8
    moe_topk: int = 2
    moe_inner: int | None = None
    moe_aux_weight: float = 0.01
    # attention branch
    use_attn: bool = True
    n_heads: int = 2
    attn_window: int = 0
    # SSM branch
    use_mamba: bool = True
    mamba_type: str = "mamba1"
    d_state: int = 16
    headdim: int = 32
    chunk_size: int = 64
    d_conv: int = 4
    expand: int = 2
    # frequency module
    use_freq: bool = True
    freq_type: str = "osc"
    freq_position: str = "pre"
    freq_cutoff: int = 5
    gft_gamma: float = 0.5
    osc_hf: bool = True
    osc_beta_init: float = 0.5
    # extra branches
    use_local: bool = True
    local_kernel: int = 4
    mamba_flip: bool = True
    flip_keep_last: int = 5
    flip_share_weights: bool = False
    gate_granularity: str = "branch"
    # shared-expert MoE
    moe_shared: bool = True
    moe_shared_inner: int | None = None
    moe_warmup_steps: int = 1000
    # output
    extrap: bool = True
    deep_sup_weight: float = 0.1
    # training objective
    loss_type: str = "CE"
    loss_scope: str = "last"  # "last": target at the last item only; "all": every position predicts its next item
    pop_debias: bool = True  # logit query minus lambda * popularity centroid (lambda learned, 0 at init)
    pop_align_weight: float = 0.05  # pull head-item and tail-item embeddings of one sequence together (0 = off)
    pop_head_frac: float = 0.2  # share of items counted as head

    def __post_init__(self) -> None:
        _check("freq_type", self.freq_type, FREQ_TYPES)
        _check("freq_position", self.freq_position, FREQ_POSITIONS)
        _check("mamba_type", self.mamba_type, MAMBA_TYPES)
        _check("ffn_type", self.ffn_type, FFN_TYPES)
        _check("loss_type", self.loss_type, ("CE", "BPR"))
        _check("gate_granularity", self.gate_granularity, GATE_GRANULARITIES)
        _check("loss_scope", self.loss_scope, ("last", "all"))
        if self.loss_scope == "all":
            if self.loss_type != "CE":
                raise ValueError("loss_scope=all needs loss_type=CE")
            if self.flip_active or self.freq_kind not in ("none", "osc"):
                raise ValueError(
                    "loss_scope=all needs causal branches only: set mamba_flip=false and freq_type in {none, osc}"
                )
        if self.pop_align_weight < 0 or not 0 < self.pop_head_frac < 1:
            raise ValueError("pop_align_weight must be >= 0 and pop_head_frac in (0, 1)")
        if self.deep_sup_weight < 0:
            raise ValueError("deep_sup_weight must be >= 0")
        if not 0 < self.osc_beta_init <= 1:
            raise ValueError("osc_beta_init (beta^2) must be in (0, 1]")
        if self.hidden_size % self.n_heads:
            raise ValueError("hidden_size must be divisible by n_heads")
        if self.n_branches < 1:
            raise ValueError("need at least one of mamba / attn / freq-branch")

    @property
    def freq_kind(self) -> str:
        """Effective frequency filter type ("none" when disabled)."""
        return self.freq_type if self.use_freq else "none"

    @property
    def flip_active(self) -> bool:
        """The flipped-Mamba branch exists only together with the Mamba branch."""
        return self.mamba_flip and self.use_mamba

    @property
    def deep_sup_active(self) -> bool:
        """Deep supervision uses the full-softmax CE head (ignored for BPR)."""
        return self.deep_sup_weight > 0 and self.loss_type == "CE"

    @property
    def freq_as_branch(self) -> bool:
        return self.freq_kind != "none" and self.freq_position == "branch"

    @property
    def n_branches(self) -> int:
        """Number of parallel branches fused by the gate."""
        return (
            int(self.use_mamba)
            + int(self.flip_active)
            + int(self.use_attn)
            + int(self.freq_as_branch)
            + int(self.use_local)
        )

    @classmethod
    def from_recbole(cls, config: Mapping[str, Any]) -> "FreqMaTConfig":
        """Build from a RecBole ``Config`` (``MAX_ITEM_LIST_LENGTH`` maps to ``max_seq_len``)."""
        values = {f.name: config[f.name] for f in fields(cls) if f.name in config}
        values["max_seq_len"] = config["MAX_ITEM_LIST_LENGTH"]
        return cls(**values)


def _check(name: str, value: str, allowed: tuple) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
