# FreqMaT-Rec

**Frequency-Aware Mamba-Transformer for Sequential Recommendation.**

Each layer: `FreqFilter (denoise) -> [Mamba-3 || RoPE-Attention] -> GatedFusion -> MoE (8 experts, top-2, replaces FFN)`.
Use `-s ffn_type=dense` for the plain-FFN ablation, `-s freq_position=branch` to make the frequency filter a
parallel branch (3-way gate). See [docs/architecture.md](docs/architecture.md).

| Problem | Component |
|---|---|
| Behavioral noise | frequency-domain filter (`FreqFilter`) |
| Long-range dependency, O(N) | Mamba SSM (`MambaBlock`, `Mamba3Block`) |
| Precise local interactions | causal RoPE attention (`RoPEAttention`) |
| Combining the views | `GatedFusion` |

## Setup

```bash
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu130 -e ".[dev]"
.venv/bin/python scripts/prepare_data.py --datasets Beauty Sports_and_Outdoors Toys_and_Games ml-1m
.venv/bin/pytest -q          
```

### Mamba-3 (optional, CUDA only; official `state-spaces/mamba`)

`configs/model/freqmat.yaml` selects `mamba_type: mamba3`. Without CUDA / `mamba-ssm` use `-s mamba_type=mamba1`
(pure-PyTorch scan, no extra dependencies).

```bash
uv pip install --python .venv/bin/python -e ".[mamba]"
uv pip install --python .venv/bin/python --no-build-isolation git+https://github.com/state-spaces/mamba.git
uv pip uninstall --python .venv/bin/python tilelang   # only needed for MIMO; its import breaks Mamba-3 SISO here
```

This installs Mamba-3 SISO (Triton kernels, no `nvcc` needed). Tested with mamba-ssm 2.3.2.post1 (git commit `e9594ce`), torch 2.14.1+cu130, triton 3.8.0.

## Run

```bash
.venv/bin/python scripts/train.py -m SASRec  -d Amazon_Beauty         # baseline (RecBole)
.venv/bin/python scripts/train.py -m FreqMaT -d Amazon_Beauty
.venv/bin/python scripts/train.py -m FreqMaT -d Amazon_Beauty -s use_freq=false     # ablation switches
.venv/bin/python scripts/run_ablation.py --epochs 1 --seeds 2024 --only freq_bsa    # quick grid check
.venv/bin/python scripts/analyze_oversmoothing.py --ckpt a=outputs/.../x.pth b=outputs/.../y.pth
```

Every run writes `outputs/<experiment>/<timestamp>_<model>_<dataset>_<hash>/` containing `config.json`,
`metrics.json`, the best checkpoint, and RecBole `log/` + `log_tensorboard/`.
Protocol (`configs/protocol.yaml`): leave-one-out, **full-ranking**, HR/NDCG/Recall@{5,10}, 5-core.

## Layout

```
src/freqmat/
  models/        FreqMaT (RecBole model), FreqMaTLayer, FreqMaTConfig, modules/ (frequency, mamba, attention, fusion, ffn, graph_fourier)
  data/          raw -> .inter converters (registry), downloads
  training/      train_and_eval() -> RunResult
  experiments/   ablation grid runner
  analysis/      oversmoothing analysis
scripts/         thin CLIs: train, prepare_data, download_raw, run_ablation, analyze_oversmoothing
configs/         protocol.yaml, model/, dataset/ (per-dataset overrides), ablation/
tests/           pytest suite (CPU; Mamba-3 tests need CUDA)
data/            raw/<dataset>/ and processed/<Name>/<Name>.inter   (git-ignored)
outputs/         run artefacts (git-ignored); outputs/legacy = runs from before the refactor
results/         tracked result tables / figures
docs/            architecture, datasets, experiments, references
```


## Development

```bash
.venv/bin/pytest -q                       # tests (Mamba-3 tests need CUDA and are skipped otherwise)
.venv/bin/ruff check . && .venv/bin/ruff format src tests scripts
```

Conventions: logic lives in `src/freqmat/`, `scripts/` only parse arguments; use `freqmat.paths` instead of
hard-coded paths. New datasets: add a module under `src/freqmat/data/` decorated with `@register("<name>")` and
import it in `data/__init__.py`. New models: register in `freqmat.models.CUSTOM_MODELS` and add
`configs/model/<name>.yaml`. Renaming an `nn.Module` attribute or parameter breaks old checkpoints.

## Known limitations
- `MoEFFN` evaluates all experts densely (cost grows with the number of experts); `RoPEAttention` rebuilds RoPE
  tables and the causal mask every forward.
- The pre-refactor ablation table (`results/legacy_ablation_Amazon_Beauty.tsv`) is single-seed and its differences
  (~0.001 NDCG) are within seed noise; re-run `scripts/run_ablation.py` (3 seeds) before drawing conclusions.
- Old command `python run.py ...` is now `python scripts/train.py ...`; old runs live in `outputs/legacy/`.

## TODO
- Baselines: GRU4Rec, BERT4Rec, FMLP-Rec, BSARec, Mamba4Rec, LRURec (register in `freqmat.models.CUSTOM_MODELS`)
- Datasets: Sports, Toys, Yelp; analysis by sequence length / long-tail; params & speed
