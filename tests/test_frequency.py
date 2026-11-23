import pytest
import torch

from freqmat.models.modules import FreqFilter


@pytest.mark.parametrize("kind", ["learned", "bsa", "both"])
def test_shape_and_grad(x, device, kind):
    f = FreqFilter(64, 50, 0.1, kind=kind, cutoff=5).to(device)
    assert f.filter(x).shape == x.shape
    f(x).sum().backward()
    assert x.grad is not None


def test_invalid_kind():
    with pytest.raises(ValueError):
        FreqFilter(64, 50, 0.0, kind="nope")


def test_bsa_beta1_is_identity_and_beta0_is_low_pass(x, device):
    f = FreqFilter(64, 50, 0.0, kind="bsa", cutoff=5).to(device).eval()
    with torch.no_grad():
        f.sqrt_beta.fill_(1.0)
        assert torch.allclose(f.filter(x), x, atol=1e-4)
        f.sqrt_beta.fill_(0.0)
        spec = torch.fft.rfft(f.filter(x).float(), dim=1, norm="ortho")
    assert spec[:, 5:].abs().max() < 1e-4
    assert spec[:, :5].abs().max() > 1e-2
