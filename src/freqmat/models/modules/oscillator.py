"""Causal damped-oscillator filter along the sequence axis."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def _inv_softplus(value: float) -> float:
    return math.log(math.expm1(value))


class OscFilter(nn.Module):
    """Damped-oscillator low-pass (plus a high-frequency rescaler), causal along the sequence axis.

    Per channel: mass ``m``, damping ``c``, stiffness ``k`` (softplus-positive); transfer function
    ``H(w) = k / (k - m w^2 + i c w)`` with unit DC gain. The kernel ``h = IFFT(H)`` keeps only
    non-negative lags and is applied with a zero-padded (length ``2T``) FFT, so the convolution is
    linear and position ``t`` never sees positions ``> t``.
    ``y = LP(x) + beta^2 (x - LP(x))`` (``hf_beta``); ``beta^2`` starts below 1 so that ``m, c, k`` get gradient.
    """

    def __init__(
        self,
        hidden: int,
        max_len: int,
        dropout: float,
        eps: float = 1e-12,
        hf_beta: bool = True,
        beta_sq_init: float = 0.5,
        floor: float = 1e-4,
    ):
        super().__init__()
        self.max_len, self.floor = max_len, floor
        for name in ("m_raw", "c_raw", "k_raw"):
            setattr(self, name, nn.Parameter(torch.full((hidden,), _inv_softplus(1.0))))
        self.sqrt_beta = nn.Parameter(torch.full((hidden,), beta_sq_init**0.5)) if hf_beta else None
        self.register_buffer("omega", 2 * math.pi * torch.arange(max_len + 1) / (2 * max_len), persistent=False)
        self.drop = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden, eps=eps)

    def physical_params(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        m = F.softplus(self.m_raw) + self.floor
        c = F.softplus(self.c_raw)
        k = F.softplus(self.k_raw) + self.floor
        return m, c, k

    def kernel(self) -> torch.Tensor:
        """Causal impulse response ``(2T, D)``."""
        m, c, k = self.physical_params()
        w = self.omega[:, None]
        h = torch.fft.irfft(k / (k - m * w**2 + 1j * c * w), n=2 * self.max_len, dim=0)
        return torch.cat([h[: self.max_len], torch.zeros_like(h[self.max_len :])], dim=0)

    def filter(self, x: torch.Tensor, basis: torch.Tensor | None = None) -> torch.Tensor:  # basis: interface parity
        length, n = x.shape[1], 2 * self.max_len
        spec = torch.fft.rfft(x.float(), n=n, dim=1) * torch.fft.rfft(self.kernel(), n=n, dim=0)[None]
        low = torch.fft.irfft(spec, n=n, dim=1)[:, :length].to(x.dtype)
        return low if self.sqrt_beta is None else low + self.sqrt_beta.square() * (x - low)

    def forward(self, x: torch.Tensor, basis: torch.Tensor | None = None) -> torch.Tensor:
        return self.norm(self.drop(self.filter(x, basis)) + x)
