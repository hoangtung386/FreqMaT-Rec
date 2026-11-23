import pytest
import torch

from freqmat.models.modules import LocalGRUBranch, OscFilter, partial_flip_index


@pytest.mark.parametrize("hf", [True, False])
def test_osc_filter_is_causal(hf):
    torch.manual_seed(0)
    f = OscFilter(16, 20, 0.0, hf_beta=hf).eval()
    x = torch.randn(3, 20, 16)
    x2 = x.clone()
    x2[:, 10:] = torch.randn(3, 10, 16)
    assert torch.allclose(f.filter(x)[:, :10], f.filter(x2)[:, :10], atol=1e-5)
    assert not torch.allclose(f.filter(x)[:, 10:], f.filter(x2)[:, 10:])


def test_osc_filter_dc_gain_and_shorter_input():
    f = OscFilter(8, 50, 0.0, hf_beta=False).eval()
    assert f.filter(torch.ones(1, 50, 8)).shape == (1, 50, 8)
    assert f.filter(torch.ones(1, 12, 8)).shape == (1, 12, 8)
    h = f.kernel()
    assert torch.allclose(h.sum(0), torch.ones(8), atol=0.2)  # DC gain ~1 (kernel truncated at T lags)


def test_osc_params_get_gradient_at_init():
    f = OscFilter(8, 20, 0.0)
    f.filter(torch.randn(2, 20, 8)).square().sum().backward()
    for name in ("m_raw", "c_raw", "k_raw"):
        assert getattr(f, name).grad.abs().sum() > 0


def test_osc_param_count():
    assert sum(p.numel() for p in OscFilter(64, 50, 0.0).parameters()) == 64 * 4 + 128


def test_local_branch_is_causal():
    torch.manual_seed(0)
    b = LocalGRUBranch(16, 4).eval()
    x = torch.randn(2, 12, 16)
    x2 = x.clone()
    x2[:, 6:] = torch.randn(2, 6, 16)
    assert torch.allclose(b(x)[:, :6], b(x2)[:, :6], atol=1e-5)


@pytest.mark.parametrize("length", [1, 3, 5, 7, 12])
def test_flip_is_involution_and_keeps_tail(length):
    lengths = torch.tensor([length])
    idx = partial_flip_index(lengths, 12, 5)
    assert torch.equal(idx.gather(1, idx), torch.arange(12)[None])
    n = max(length - 5, 0)
    assert torch.equal(idx[0, n:], torch.arange(n, 12))
    assert torch.equal(idx[0, :n], torch.arange(n - 1, -1, -1))
