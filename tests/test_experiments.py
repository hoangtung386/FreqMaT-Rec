import torch

from freqmat.analysis.oversmoothing import singular_spectrum, token_cosine
from freqmat.experiments.ablation import load_grid, summarise, write_tsv
from freqmat.training.runner import make_run_dir


def test_shipped_grid_is_well_formed():
    grid = load_grid()
    assert grid["dataset"] and grid["seeds"] and len(grid["variants"]) >= 10
    assert all(isinstance(v, dict) for v in grid["variants"].values())


def test_summarise_mean_std_and_failures(tmp_path):
    runs = [
        {"variant": "a", "status": "ok", "params": 10, "test": {"ndcg@10": 0.1}},
        {"variant": "a", "status": "ok", "params": 10, "test": {"ndcg@10": 0.3}},
        {"variant": "a", "status": "failed: X", "params": "", "test": {}},
        {"variant": "b", "status": "failed: X", "params": "", "test": {}},
    ]
    a, b = summarise(runs)
    assert (a["n_ok"], a["n_failed"], a["ndcg@10_mean"]) == (2, 1, 0.2)
    assert (b["n_ok"], b["n_failed"]) == (0, 1)
    write_tsv([a, b], tmp_path / "t.tsv")
    assert (tmp_path / "t.tsv").read_text().splitlines()[0].startswith("variant\tn_ok")


def test_make_run_dir_is_unique_per_overrides(tmp_path, monkeypatch):
    from freqmat import paths

    monkeypatch.setattr(paths, "OUTPUTS", tmp_path)
    one = make_run_dir("exp", "FreqMaT", "ds", {"a": 1})
    two = make_run_dir("exp", "FreqMaT", "ds", {"a": 2})
    assert one != two and one.parent == tmp_path / "exp"


def test_token_cosine_identical_tokens_is_one():
    states = torch.ones(2, 4, 8)
    mask = torch.tensor([[1, 1, 1, 0], [1, 0, 0, 0]], dtype=torch.bool)
    total, count = token_cosine(states, mask)
    assert count == 1 and abs(total - 1.0) < 1e-5  # the single-token sequence has no pairs


def test_singular_spectrum_is_normalised_and_decreasing():
    s = singular_spectrum(torch.randn(20, 6))
    assert s[0] == 1.0 and all(a >= b for a, b in zip(s, s[1:], strict=False))
