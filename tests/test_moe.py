import torch

from freqmat.models.modules import FFN, MoEFFN


def _n_params(module):
    return sum(p.numel() for p in module.parameters())


def test_grads_reach_router_and_experts_and_aux_loss_is_finite(x, device):
    moe = MoEFFN(64, 64, 4, 2, 0.1).to(device)
    mask = torch.ones(4, 50, dtype=torch.bool, device=device)
    mask[:, 40:] = False
    y = moe(x.detach(), mask)
    (y.sum() + moe.aux_loss).backward()
    assert y.shape == x.shape
    assert moe.router.weight.grad is not None and moe.w1.grad is not None
    assert torch.isfinite(moe.aux_loss)


def test_param_matched_with_dense_ffn():
    # 8 experts of inner 32 ~ one dense FFN of inner 256 (plus the router)
    moe, dense = MoEFFN(64, 32, 8, 2, 0.1), FFN(64, 256, 0.1)
    assert abs(_n_params(moe) - _n_params(dense)) < 0.1 * _n_params(dense)
