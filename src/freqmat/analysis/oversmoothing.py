"""Oversmoothing analysis (BSARec Fig. 3) for trained FreqMaT checkpoints.

For every layer ``l = 0`` (embeddings) .. ``L`` it measures, on test sequences:

* token cosine similarity: mean pairwise cosine between the hidden states of the real items of a
  sequence (-> 1 means all tokens collapsed to one vector = oversmoothing);
* singular values: normalised spectrum of the (users x D) matrix of last-position states
  (fast decay / rank collapse = oversmoothing).
"""

import json
from pathlib import Path

import numpy as np
import torch

from freqmat import paths
from freqmat.compat import allow_unsafe_torch_load


def load_model_from_checkpoint(path: str | Path, device: str):
    """Rebuild ``(model, test_loader)`` from a RecBole checkpoint produced by this project."""
    allow_unsafe_torch_load()
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_seed

    from freqmat.models import FreqMaT

    ckpt = torch.load(path, map_location="cpu")
    config = ckpt["config"]
    config["device"] = device
    # checkpoints may carry stale absolute data paths (e.g. from before data/ was reorganised)
    config["data_path"] = str(paths.DATA_PROCESSED / config["dataset"])  # RecBole stores <data_path>/<dataset>
    init_seed(config["seed"], config["reproducibility"])
    train, _, test = data_preparation(config, create_dataset(config))
    model = FreqMaT(config, train.dataset).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval(), test


def token_cosine(states: torch.Tensor, mask: torch.Tensor) -> tuple[float, int]:
    """Sum over sequences of the mean off-diagonal cosine among real tokens; also the sequence count."""
    normed = torch.nn.functional.normalize(states.float(), dim=-1)
    sim = normed @ normed.transpose(1, 2)  # (B, N, N)
    eye = torch.eye(mask.shape[1], dtype=torch.bool, device=mask.device)
    pair = (mask.unsqueeze(2) & mask.unsqueeze(1)) & ~eye
    n_pairs = pair.flatten(1).sum(1)
    ok = n_pairs > 0
    per_seq = (sim * pair).flatten(1).sum(1)[ok] / n_pairs[ok]
    return per_seq.sum().item(), int(ok.sum().item())


def singular_spectrum(last_states: torch.Tensor) -> list[float]:
    """Singular values of ``(users, D)`` normalised by the largest one."""
    s = torch.linalg.svdvals(last_states)
    return (s / s[0]).tolist()


@torch.no_grad()
def analyse(path: str | Path, n_seq: int, device: str) -> dict:
    """Per-layer token cosine and last-position singular spectrum for one checkpoint."""
    model, test = load_model_from_checkpoint(path, device)
    cos_sum = cos_cnt = None
    last: list[list[torch.Tensor]] = []
    seen = 0
    for batch in test:
        inter = batch[0].to(device)
        seq, length = inter[model.ITEM_SEQ], inter[model.ITEM_SEQ_LEN]
        states = model.hidden_states(seq, model._timestamps(inter))
        if cos_sum is None:
            cos_sum, cos_cnt, last = np.zeros(len(states)), np.zeros(len(states)), [[] for _ in states]
        mask = seq != 0
        for layer, hidden in enumerate(states):
            total, count = token_cosine(hidden, mask)
            cos_sum[layer] += total
            cos_cnt[layer] += count
            last[layer].append(model.gather_indexes(hidden, length - 1).float().cpu())
        seen += seq.shape[0]
        if seen >= n_seq:
            break
    return {
        "cosine": (cos_sum / np.maximum(cos_cnt, 1)).tolist(),
        "singular_values": [singular_spectrum(torch.cat(chunks)[:n_seq]) for chunks in last],
        "n_sequences": seen,
    }


def plot(results: dict[str, dict], out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 2, figsize=(9, 3.4))
    for label, res in results.items():
        ax[0].plot(res["singular_values"][-1], label=label)
        ax[1].plot(range(len(res["cosine"])), res["cosine"], marker="o", label=label)
    ax[0].set(xlabel="singular value index", ylabel="normalised singular value", title="(a) last-layer spectrum")
    ax[1].set(xlabel="layer (0 = embedding)", ylabel="mean token cosine similarity", title="(b) token similarity")
    ax[1].set_xticks(range(len(next(iter(results.values()))["cosine"])))
    ax[0].legend()
    fig.tight_layout()
    fig.savefig(out, dpi=160)


def run(checkpoints: dict[str, str], n_seq: int, out_stem: Path) -> dict[str, dict]:
    """Analyse every ``label -> checkpoint``; write ``<out_stem>.json`` and ``<out_stem>.png``."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    results = {}
    for label, ckpt in checkpoints.items():
        results[label] = analyse(ckpt, n_seq, device)
        print(label, "cosine per layer:", [round(c, 3) for c in results[label]["cosine"]])
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    with open(out_stem.with_suffix(".json"), "w") as f:
        json.dump(results, f)
    plot(results, out_stem.with_suffix(".png"))
    print(f"saved {out_stem}.png / .json")
    return results
