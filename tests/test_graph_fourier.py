import torch

from freqmat.models.modules import FreqFilter, time_graph_basis

B, N, C, REAL = 4, 50, 5, 30


def _inputs(device):
    ts = torch.cumsum(torch.rand(B, N, device=device) * 1e5, 1) + 1.6e9
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    mask[:, REAL:] = False
    return ts, mask


def test_basis_is_orthonormal_and_decouples_padding(device):
    ts, mask = _inputs(device)
    basis = time_graph_basis(ts, mask, C, 0.5)
    assert basis.shape == (B, N, C)
    eye = torch.eye(C, device=device).expand(B, C, C)
    assert torch.allclose(basis.transpose(1, 2) @ basis, eye, atol=1e-4)
    assert basis[:, REAL:, :].abs().max() < 1e-3, "padding nodes must not carry low-frequency energy"


def test_constant_signal_lives_in_low_band(device):
    ts, mask = _inputs(device)
    basis = time_graph_basis(ts, mask, C, 0.5)
    ones = mask.float().unsqueeze(-1)
    ones = ones / ones.norm(dim=1, keepdim=True)
    assert (basis @ (basis.transpose(1, 2) @ ones) - ones).abs().max() < 1e-3


def test_gft_filter_gradient(x, device):
    ts, mask = _inputs(device)
    basis = time_graph_basis(ts, mask, C, 0.5)
    f = FreqFilter(64, 50, 0.1, kind="gft", cutoff=C).to(device)
    f.filter(x.detach().clone().requires_grad_(), basis).sum().backward()
    assert f.sqrt_beta.grad is not None
