import torch

from freqmat.models.modules import RoPEAttention


def test_shape_and_grad(x, device):
    m = RoPEAttention(64, 2, 0.1).to(device)
    y = m(x)
    assert y.shape == x.shape
    y.sum().backward()


def test_causal(device):
    m = RoPEAttention(64, 2, 0.0).to(device).eval()
    a = torch.randn(1, 20, 64, device=device)
    b = a.clone()
    b[:, 15:] += 1
    assert torch.allclose(m(a)[:, :15], m(b)[:, :15], atol=1e-5)


def test_local_window_limits_receptive_field(device):
    m = RoPEAttention(64, 2, 0.0, window=3).to(device).eval()
    a = torch.randn(1, 20, 64, device=device)
    b = a.clone()
    b[:, 0] += 1  # token 0 is outside the window of tokens >= 3
    assert torch.allclose(m(a)[:, 3:], m(b)[:, 3:], atol=1e-5)
