"""State-space (Mamba) sequence mixers."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

try:  # optional fused Mamba-1 CUDA kernel (only if selective_scan_cuda was built)
    import selective_scan_cuda  # noqa: F401
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
except Exception:  # pragma: no cover
    selective_scan_fn = None


class Mamba3Block(nn.Module):
    """Official Mamba-3 (SISO, Triton kernels) from ``state-spaces/mamba``, run under bf16 autocast.

    Mamba-3 = trapezoidal discretization + complex (RoPE) state update + QK-norm'd B/C biases.
    Requires CUDA and ``mamba-ssm`` built from source (see README).
    """

    def __init__(self, d_model: int, d_state: int = 32, headdim: int = 32, expand: int = 2, chunk_size: int = 64):
        super().__init__()
        try:
            from mamba_ssm import Mamba3
        except ImportError as err:
            raise ImportError(
                "mamba_type=mamba3 needs `mamba-ssm` (see README); use mamba_type=mamba1 for the pure-PyTorch scan"
            ) from err
        self.m = Mamba3(
            d_model=d_model, d_state=d_state, headdim=headdim, expand=expand, is_mimo=False, chunk_size=chunk_size
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.autocast(x.device.type, dtype=torch.bfloat16):
            y = self.m(x.to(torch.bfloat16))
        return y.to(x.dtype)


class MambaBlock(nn.Module):
    """Selective SSM (Mamba-1) mixer. Uses the ``mamba_ssm`` kernel if installed, else a PyTorch scan."""

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_inner = d_model * expand
        self.d_state = d_state
        self.dt_rank = math.ceil(d_model / 16)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, d_conv, groups=self.d_inner, padding=d_conv - 1, bias=True)
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        dt = torch.exp(torch.rand(self.d_inner) * (math.log(0.1) - math.log(1e-3)) + math.log(1e-3))
        with torch.no_grad():
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))  # inverse softplus
        a_init = torch.arange(1, d_state + 1, dtype=torch.float32).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(a_init))
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _scan(self, u, delta, A, B, C):
        """``u, delta``: (b, l, d); ``A``: (d, n); ``B, C``: (b, l, n)."""
        if selective_scan_fn is not None and u.is_cuda:
            y = selective_scan_fn(
                u.transpose(1, 2).contiguous(),
                delta.transpose(1, 2).contiguous(),
                A,
                B.transpose(1, 2).contiguous(),
                C.transpose(1, 2).contiguous(),
                self.D.float(),
                z=None,
                delta_bias=None,
                delta_softplus=False,
            )
            return y.transpose(1, 2)
        batch, length, d_inner = u.shape
        delta_a = torch.exp(delta.unsqueeze(-1) * A)  # (b, l, d, n)
        delta_bu = delta.unsqueeze(-1) * B.unsqueeze(2) * u.unsqueeze(-1)  # (b, l, d, n)
        state = u.new_zeros(batch, d_inner, self.d_state)
        outputs = []
        for t in range(length):
            state = delta_a[:, t] * state + delta_bu[:, t]
            outputs.append((state * C[:, t].unsqueeze(1)).sum(-1))
        return torch.stack(outputs, 1) + u * self.D

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        length = x.shape[1]
        x_in, gate = self.in_proj(x).chunk(2, dim=-1)
        x_in = F.silu(self.conv1d(x_in.transpose(1, 2))[..., :length].transpose(1, 2))
        dt, B, C = torch.split(self.x_proj(x_in), [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = F.softplus(self.dt_proj(dt))
        A = -torch.exp(self.A_log.float())
        y = self._scan(x_in.float(), delta.float(), A, B.float(), C.float()).to(x.dtype)
        return self.out_proj(y * F.silu(gate))
