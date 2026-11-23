import pytest
import torch

from freqmat.models.modules import GatedFusion


@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("granularity", ["channel", "branch"])
def test_gated_fusion_shapes_and_weights(n, granularity):
    f = GatedFusion(16, n, granularity)
    branches = [torch.ones(2, 5, 16) * (i + 1) for i in range(n)]
    out = f(*branches)
    assert out.shape == (2, 5, 16)
    assert abs(f.last_gate.sum().item() - 1.0) < 1e-5
    assert out.min() >= 1 - 1e-5 and out.max() <= n + 1e-5  # convex combination


def test_gated_fusion_param_counts():
    assert sum(p.numel() for p in GatedFusion(64, 4, "channel").parameters()) == 65792
    assert sum(p.numel() for p in GatedFusion(64, 4, "branch").parameters()) == 1028


def test_gated_fusion_rejects_bad_args():
    with pytest.raises(ValueError):
        GatedFusion(8, 1)
    with pytest.raises(ValueError):
        GatedFusion(8, 2, "x")


@pytest.mark.parametrize("n", [2, 3])
def test_initial_gates_near_uniform(n):
    torch.manual_seed(0)
    f = GatedFusion(32, n)
    f(*[torch.randn(4, 10, 32) for _ in range(n)])
    assert torch.allclose(f.last_gate, torch.full((n,), 1 / n), atol=0.05)


@pytest.mark.parametrize("granularity", ["channel", "branch"])
def test_gated_fusion_backpropagates(device, granularity):
    fusion = GatedFusion(64, 3, granularity).to(device)
    branches = [torch.randn(4, 50, 64, device=device, requires_grad=True) for _ in range(3)]
    fusion(*branches).sum().backward()
    assert all(b.grad is not None for b in branches)
