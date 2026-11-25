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
    {"pos_embedding": False},
    {"norm_position": "post", "norm_type": "layernorm", "ffn_act": "gelu", "head_mlp": False},
    {"norm_position": "post", "ffn_act": "swiglu", "moe_shared": False},
    {"ffn_type": "dense", "ffn_act": "swiglu"},
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
    {"freq_type": "osc"},
    {"freq_type": "osc", "osc_hf": False},
    {"freq_type": "osc", "freq_position": "branch"},
    {"freq_type": "osc", "use_local": True},
    {"freq_type": "osc", "use_local": True, "gate_granularity": "branch"},
    {"freq_type": "osc", "mamba_flip": True},
    {
        "freq_type": "osc",
        "freq_position": "branch",
        "use_local": True,
        "mamba_flip": True,
        "gate_granularity": "branch",
    },
    {"freq_type": "osc", "mamba_flip": True, "flip_share_weights": True},
    {"moe_shared": True, "moe_warmup_steps": 5},
    {"deep_sup_weight": 0.1, "n_layers": 3},
    {"extrap": True},
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
        {"use_mamba": False, "use_attn": False, "use_freq": False, "use_local": False},
    ],
)
def test_invalid_config_rejected(bad):
    with pytest.raises(ValueError):
        FreqMaTConfig(**bad)


NEW_KEYS = ("osc", "local", "flip", "shared", "alpha_raw", "extrap", "pos_embedding", "final_norm", "head.")
OFF = dict(
    freq_type="learned",
    use_local=False,
    mamba_flip=False,
    gate_granularity="channel",
    moe_shared=False,
    moe_warmup_steps=0,
    extrap=False,
    deep_sup_weight=0.0,
    pos_embedding=False,
    norm_position="post",
    norm_type="layernorm",
    ffn_act="gelu",
    head_mlp=False,
)


def test_all_new_switches_off_adds_no_new_parameters():
    keys = FreqMaT(make_config(**OFF), FakeDataset()).state_dict().keys()
    assert not [k for k in keys if any(n in k for n in NEW_KEYS)]


def test_defaults_enable_new_modules():
    keys = FreqMaT(make_config(freq_type="osc"), FakeDataset()).state_dict().keys()
    for name in (
        "freq.m_raw",
        "local.",
        "mamba_flip.",
        "ffn.shared.",
        "extrap_dt",
        "pos_embedding.weight",
        "final_norm.weight",
        "head.net.0.weight",
    ):
        assert any(name in k for k in keys), name


def test_extrap_is_identity_at_init():
    base = FreqMaT(make_config(extrap=False), FakeDataset()).eval()
    ext = FreqMaT(make_config(extrap=True), FakeDataset()).eval()
    ext.load_state_dict(base.state_dict(), strict=False)
    b = make_batch()
    assert torch.allclose(base.full_sort_predict(b), ext.full_sort_predict(b), atol=1e-6)


def test_deep_supervision_changes_loss():
    plain = FreqMaT(make_config(n_layers=3, deep_sup_weight=0.0), FakeDataset()).eval()
    deep = FreqMaT(make_config(n_layers=3, deep_sup_weight=0.5), FakeDataset()).eval()
    deep.load_state_dict(plain.state_dict())
    b = make_batch()
    assert not torch.allclose(deep.calculate_loss(b), plain.calculate_loss(b))


def test_shared_moe_step_counts_only_in_training():
    model = FreqMaT(make_config(moe_shared=True, moe_warmup_steps=10), FakeDataset())
    ffn = model.layers[0].ffn
    model.train()
    model.calculate_loss(make_batch())
    assert ffn.step.item() == 1
    model.eval()
    model.full_sort_predict(make_batch())
    assert ffn.step.item() == 1
    assert ffn.aux_loss.ndim == 0


@pytest.mark.parametrize(
    "bad",
    [{"gate_granularity": "x"}, {"deep_sup_weight": -1.0}, {"osc_beta_init": 0.0}],
)
def test_invalid_new_config_rejected(bad):
    with pytest.raises(ValueError):
        FreqMaTConfig(**bad)


class PopDataset(FakeDataset):
    """Fake training split: one row per training sample, ``item_id`` is the sample's target."""

    uid_field, iid_field = "user_id", "item_id"
    inter_feat = {"user_id": [1, 2, 1, 3, 2], "item_id": [5, 5, 5, 9, 6]}


def test_popularity_counts_are_target_counts_of_the_training_split():
    from freqmat.models.popularity import popularity_counts

    c = popularity_counts(PopDataset(), N_ITEMS)
    assert c[5] == 3 and c[6] == 1 and c[9] == 1 and c[7] == 0 and c.sum() == 5


def pop_model(**kw):
    return FreqMaT(make_config(pop_debias=True, pop_align_weight=0.1, pop_head_frac=0.2, **kw), PopDataset())


def test_pop_terms_train_and_identity_at_init():
    torch.manual_seed(0)
    m = pop_model()
    assert m.pop_active
    batch = make_batch()
    m.eval()
    q0 = m._debias(torch.randn(3, 32))
    assert q0.shape == (3, 32)
    m.train()
    loss = m.calculate_loss(batch)
    loss.backward()
    assert m.pop_lambda.grad is not None
    assert torch.isfinite(loss)


def test_loss_scope_all_rejects_noncausal_and_matches_shape():
    with pytest.raises(ValueError):
        FreqMaT(make_config(loss_scope="all"), FakeDataset())  # learned filter + flip are not causal
    m = FreqMaT(make_config(loss_scope="all", freq_type="osc", mamba_flip=False, use_local=True), FakeDataset())
    loss = m.calculate_loss(make_batch())
    loss.backward()
    assert torch.isfinite(loss)


def test_rmsnorm_matches_definition_and_swiglu_keeps_parameter_count():
    from freqmat.models.modules import FFN, MoEFFN, RMSNorm

    x = torch.randn(3, 5, 16)
    out = RMSNorm(16, eps=1e-6)(x)
    expected = x / x.pow(2).mean(-1, keepdim=True).add(1e-6).sqrt()
    assert torch.allclose(out, expected, atol=1e-5)
    count = lambda m: sum(p.numel() for p in m.parameters())  # noqa: E731
    assert abs(count(FFN(64, 256, 0.0, "swiglu")) / count(FFN(64, 256, 0.0, "gelu")) - 1) < 0.05
    assert abs(count(MoEFFN(64, 96, 4, 2, 0.0, "swiglu")) / count(MoEFFN(64, 96, 4, 2, 0.0, "gelu")) - 1) < 0.1
    assert FFN(64, 256, 0.0, "swiglu")(x.new_zeros(2, 3, 64)).shape == (2, 3, 64)


def test_pre_norm_stack_has_final_norm_and_post_norm_does_not():
    pre = FreqMaT(make_config(norm_position="pre"), FakeDataset())
    post = FreqMaT(make_config(norm_position="post"), FakeDataset())
    assert pre.final_norm is not None and post.final_norm is None


def test_output_head_is_residual_and_changes_the_score():
    torch.manual_seed(0)
    with_head = FreqMaT(make_config(head_mlp=True), FakeDataset()).eval()
    without = FreqMaT(make_config(head_mlp=False), FakeDataset()).eval()
    without.load_state_dict(with_head.state_dict(), strict=False)
    batch = make_batch()
    a = with_head.full_sort_predict(batch)
    b = without.full_sort_predict(batch)
    assert a.shape == b.shape and not torch.allclose(a, b)


def test_invalid_block_options_rejected():
    for bad in ({"norm_position": "mid"}, {"norm_type": "batchnorm"}, {"ffn_act": "relu"}):
        with pytest.raises(ValueError):
            FreqMaTConfig(**bad)


def test_yaml_defaults_match_dataclass_and_enable_every_block():
    from dataclasses import fields

    import yaml

    from freqmat import paths

    y = yaml.safe_load((paths.CONFIGS / "model" / "freqmat.yaml").read_text())
    defaults = {f.name: f.default for f in fields(FreqMaTConfig)}
    intended = {"mamba_type", "d_state"}  # the YAML targets the Mamba-3 kernels; the dataclass stays dependency-free
    assert {k: v for k, v in y.items() if k in defaults and k not in intended and v != defaults[k]} == {}
    for name in (
        "pos_embedding",
        "head_mlp",
        "use_attn",
        "use_mamba",
        "use_local",
        "mamba_flip",
        "moe_shared",
        "extrap",
    ):
        assert y[name] is True, name
    assert (y["norm_position"], y["norm_type"], y["ffn_act"], y["freq_type"], y["n_layers"]) == (
        "pre",
        "rmsnorm",
        "swiglu",
        "osc",
        2,
    )
    assert y["deep_sup_weight"] > 0 and y["pop_debias"] and y["pop_align_weight"] > 0
