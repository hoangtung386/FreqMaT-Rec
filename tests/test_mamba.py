import pytest
import torch

from freqmat.models.modules import Mamba3Block, MambaBlock


def test_mamba1_shape_and_grad(x, device):
    m = MambaBlock(64, 16).to(device)
    y = m(x)
    assert y.shape == x.shape
    y.sum().backward()


def _assert_causal(module, device, atol):
    module = module.to(device).eval()
    a = torch.randn(1, 20, 64, device=device)
    b = a.clone()
    b[:, 15:] += 1
    assert torch.allclose(module(a)[:, :15], module(b)[:, :15], atol=atol)


def test_mamba1_causal(device):
    _assert_causal(MambaBlock(64, 16), device, 1e-5)


@pytest.mark.cuda
def test_mamba3_shape_and_causal(x, device):
    m = Mamba3Block(64, 32, 32).to(device)
    assert m(x).shape == x.shape
    _assert_causal(Mamba3Block(64, 32, 32), device, 1e-2)


def test_mamba3_without_package_raises_clear_error():
    try:
        import mamba_ssm  # noqa: F401

        pytest.skip("mamba_ssm installed")
    except ImportError:
        with pytest.raises(ImportError, match="mamba-ssm"):
            Mamba3Block(64)
