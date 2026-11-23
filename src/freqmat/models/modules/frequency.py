"""Frequency-domain filtering along the sequence axis."""

import torch
import torch.nn as nn

FILTER_KINDS = ("learned", "bsa", "both", "gft")


class FreqFilter(nn.Module):
    """Frequency-domain filter along the sequence axis.

    ``kind``:
        learned: FMLP-Rec style learnable complex filter (+ learnable scalar mix with the raw signal).
        bsa:     BSARec "frequency rescaler": keep the ``cutoff`` lowest bins, scale the rest by a
                 learnable ``beta`` vector (``beta = sqrt_beta ** 2``, D params, no complex weights).
        both:    cascade in the spectrum: BSA rescaler, then a learnable complex filter (identity init).
        gft:     like ``bsa`` but in the *graph* Fourier basis of a time-weighted path graph
                 (see :func:`freqmat.models.modules.graph_fourier.time_graph_basis`).
    """

    def __init__(
        self, hidden: int, max_len: int, dropout: float, eps: float = 1e-12, kind: str = "learned", cutoff: int = 5
    ):
        super().__init__()
        if kind not in FILTER_KINDS:
            raise ValueError(f"kind must be one of {FILTER_KINDS}, got {kind!r}")
        n_freq = max_len // 2 + 1
        self.kind, self.max_len, self.cutoff = kind, max_len, cutoff
        if kind in ("learned", "both"):
            weight = torch.randn(1, n_freq, hidden, 2) * 0.02
            if kind == "both":
                weight[..., 0] += 1.0  # identity complex filter at init -> starts as a pure BSA rescaler
            self.weight = nn.Parameter(weight)
        if kind == "learned":
            self.alpha = nn.Parameter(torch.tensor(0.0))  # sigmoid -> mixing weight with raw x
        if kind in ("bsa", "both", "gft"):
            self.sqrt_beta = nn.Parameter(torch.randn(1, 1, hidden))  # as in the BSARec code
        if kind in ("bsa", "both"):
            self.register_buffer("low", (torch.arange(n_freq) < cutoff).float().view(1, -1, 1))
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden, eps=eps)

    def filter(self, x: torch.Tensor, basis: torch.Tensor | None = None) -> torch.Tensor:
        """Filtered signal only (no residual / norm). ``basis`` is ``(B, N, c)`` for ``gft``."""
        if self.kind == "gft":
            basis = basis.to(x.dtype)
            low_pass = basis @ (basis.transpose(1, 2) @ x)
            return low_pass + self.sqrt_beta.square() * (x - low_pass)
        length = x.shape[1]
        spec = torch.fft.rfft(x.float(), n=self.max_len, dim=1, norm="ortho")
        if self.kind in ("bsa", "both"):
            spec = spec * (self.low + (1 - self.low) * self.sqrt_beta.square())
        if self.kind in ("learned", "both"):
            spec = spec * torch.view_as_complex(self.weight)
        y = torch.fft.irfft(spec, n=self.max_len, dim=1, norm="ortho")[:, :length].to(x.dtype)
        if self.kind == "learned":
            mix = torch.sigmoid(self.alpha)
            y = mix * y + (1 - mix) * x
        return y

    def forward(self, x: torch.Tensor, basis: torch.Tensor | None = None) -> torch.Tensor:
        """Sequential ("pre") use: residual + LayerNorm."""
        return self.norm(self.drop(self.filter(x, basis)) + x)
