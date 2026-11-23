"""Whole-model forward / backward on a tiny synthetic batch (no RecBole dataset needed)."""

import pytest
import torch

from freqmat.models import FreqMaT, FreqMaTConfig

N_ITEMS, BATCH, LENGTH = 30, 6, 12


class FakeDataset:
    @staticmethod
    def num(_field):
        return N_ITEMS


def make_config(**overrides):
    base = dict(
        USER_ID_FIELD="user_id",
        ITEM_ID_FIELD="item_id",
        LIST_SUFFIX="_list",
        ITEM_LIST_LENGTH_FIELD="item_length",
        NEG_PREFIX="neg_",
        MAX_ITEM_LIST_LENGTH=LENGTH,
        TIME_FIELD="timestamp",
        device="cpu",
        train_neg_sample_args=None,
        hidden_size=32,
        inner_size=64,
        n_layers=2,
        moe_experts=4,
        d_state=8,
        n_heads=2,
        mamba_type="mamba1",
        freq_type="learned",
        freq_position="pre",
        ffn_type="moe",
        use_freq=True,
        use_mamba=True,
        use_attn=True,
        loss_type="CE",
        moe_aux_weight=0.01,
        moe_topk=2,
        moe_inner=None,
        hidden_dropout_prob=0.1,
        attn_dropout_prob=0.1,
        layer_norm_eps=1e-12,
        freq_cutoff=3,
        gft_gamma=0.5,
        headdim=32,
        chunk_size=64,
        d_conv=4,
        expand=2,
        attn_window=0,
    )
    return {**base, **overrides}


def make_batch():
    torch.manual_seed(0)
    lengths = torch.randint(2, LENGTH + 1, (BATCH,))
    seq = torch.zeros(BATCH, LENGTH, dtype=torch.long)
    for i, n in enumerate(lengths):
        seq[i, :n] = torch.randint(1, N_ITEMS, (n,))
    ts = torch.cumsum(torch.rand(BATCH, LENGTH) * 1e5, 1) + 1.6e9
    return {
        "item_id_list": seq,
        "item_length": lengths,
        "item_id": torch.randint(1, N_ITEMS, (BATCH,)),
        "timestamp_list": ts,
    }


VARIANTS = [
    {},
    {"freq_type": "bsa"},
    {"freq_type": "both"},
    {"freq_type": "gft"},
    {"freq_type": "gft", "freq_position": "branch"},
    {"freq_type": "learned", "freq_position": "branch"},
    {"use_freq": False},
    {"use_mamba": False},
    {"use_attn": False},
    {"ffn_type": "dense"},
    {"freq_type": "none"},
]


@pytest.mark.parametrize("overrides", VARIANTS, ids=lambda o: ",".join(f"{k}={v}" for k, v in o.items()) or "default")
def test_loss_is_finite_and_backpropagates(overrides):
    model = FreqMaT(make_config(**overrides), FakeDataset())
    batch = make_batch()
    loss = model.calculate_loss(batch)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.item_embedding.weight.grad is not None
    assert model.full_sort_predict(batch).shape == (BATCH, N_ITEMS)
    assert model.predict(batch).shape == (BATCH,)


def test_padding_positions_are_zeroed():
    model = FreqMaT(make_config(), FakeDataset()).eval()
    batch = make_batch()
    states = model.hidden_states(batch["item_id_list"], batch["timestamp_list"])
    assert len(states) == 3
    assert states[0][batch["item_id_list"] == 0].abs().max() == 0


def test_gft_without_timestamps_raises():
    model = FreqMaT(make_config(freq_type="gft"), FakeDataset())
    with pytest.raises(ValueError, match="timestamp"):
        model.hidden_states(make_batch()["item_id_list"], None)


def test_bpr_requires_negative_sampling():
    with pytest.raises(ValueError, match="train_neg_sample_args"):
        FreqMaT(make_config(loss_type="BPR"), FakeDataset())


@pytest.mark.parametrize(
    "bad",
    [
        {"freq_type": "x"},
        {"mamba_type": "x"},
        {"ffn_type": "x"},
        {"use_mamba": False, "use_attn": False, "use_freq": False},
    ],
)
def test_invalid_config_rejected(bad):
    with pytest.raises(ValueError):
        FreqMaTConfig(**bad)
