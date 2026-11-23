import pytest
import torch

from freqmat.models.modules import GatedFusion


@pytest.mark.parametrize("n", [2, 3])
def test_gates_sum_to_one(device, n):
    fusion = GatedFusion(64, n).to(device)
    branches = [torch.randn(4, 50, 64, device=device) for _ in range(n)]
    out = fusion(*branches)
    out.sum().backward()
    assert out.shape == branches[0].shape
    assert abs(fusion.last_gate.sum().item() - 1) < 1e-4


def test_rejects_unsupported_branch_count():
    with pytest.raises(ValueError):
        GatedFusion(64, 4)
