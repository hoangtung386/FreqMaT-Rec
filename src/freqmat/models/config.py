"""Typed, validated hyper-parameters of the FreqMaT architecture."""

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

FREQ_TYPES = ("none", "learned", "bsa", "both", "gft")
FREQ_POSITIONS = ("pre", "branch")
MAMBA_TYPES = ("mamba1", "mamba3")
FFN_TYPES = ("dense", "moe")


@dataclass(frozen=True)
class FreqMaTConfig:
    """Architecture options. Defaults mirror ``configs/model/freqmat.yaml`` except ``mamba_type``,
    which defaults to the dependency-free ``mamba1`` (the YAML selects ``mamba3``)."""

    hidden_size: int = 64
    inner_size: int = 256
    n_layers: int = 2
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
    freq_type: str = "learned"
    freq_position: str = "pre"
    freq_cutoff: int = 5
    gft_gamma: float = 0.5
    # training objective
    loss_type: str = "CE"

    def __post_init__(self) -> None:
        _check("freq_type", self.freq_type, FREQ_TYPES)
        _check("freq_position", self.freq_position, FREQ_POSITIONS)
        _check("mamba_type", self.mamba_type, MAMBA_TYPES)
        _check("ffn_type", self.ffn_type, FFN_TYPES)
        _check("loss_type", self.loss_type, ("CE", "BPR"))
        if self.hidden_size % self.n_heads:
            raise ValueError("hidden_size must be divisible by n_heads")
        if self.n_branches < 1:
            raise ValueError("need at least one of mamba / attn / freq-branch")

    @property
    def freq_kind(self) -> str:
        """Effective frequency filter type ("none" when disabled)."""
        return self.freq_type if self.use_freq else "none"

    @property
    def freq_as_branch(self) -> bool:
        return self.freq_kind != "none" and self.freq_position == "branch"

    @property
    def n_branches(self) -> int:
        """Number of parallel branches fused by the gate."""
        return int(self.use_mamba) + int(self.use_attn) + int(self.freq_as_branch)

    @classmethod
    def from_recbole(cls, config: Mapping[str, Any]) -> "FreqMaTConfig":
        """Build from a RecBole ``Config`` (``MAX_ITEM_LIST_LENGTH`` maps to ``max_seq_len``)."""
        values = {f.name: config[f.name] for f in fields(cls) if f.name in config}
        values["max_seq_len"] = config["MAX_ITEM_LIST_LENGTH"]
        return cls(**values)


def _check(name: str, value: str, allowed: tuple) -> None:
    if value not in allowed:
        raise ValueError(f"{name} must be one of {allowed}, got {value!r}")
