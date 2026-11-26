# FreqMaT-Rec

**Frequency-Aware Mamba-Transformer for Sequential Recommendation.**

Each layer: `OscFilter (causal denoise) -> [Mamba-3 || flipped Mamba || RoPE-Attention || local GRU] -> GatedFusion ->
shared-expert MoE`, with one-step extrapolation of the final state, deep supervision and popularity debiasing. Every new
module is on by default (about 9.2M parameters on Beauty) and has a switch for its `w/o X` ablation, e.g.
`-s use_local=false`, `-s moe_shared=false`, `-s freq_type=learned`. See [docs/architecture.md](docs/architecture.md).

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
./scripts/install_mamba.sh
```
Re-run it after every `uv sync` / environment rebuild (uv removes packages that are not in `uv.lock`). The script pins the
commit, keeps torch/triton untouched (`--no-deps`) and removes `tilelang` (MIMO only; its import breaks Mamba-3 SISO here).

This installs Mamba-3 SISO (Triton kernels, no `nvcc` needed). Tested with mamba-ssm 2.3.2.post1 (git commit `e9594ce`), torch 2.14.1+cu130, triton 3.8.0.

## Run

Datasets must exist under `data/processed/` first (see [docs/datasets.md](docs/datasets.md)). Names for `-d`:
`Amazon_Beauty`, `Amazon_Toys_and_Games`, `Amazon_Video_Games`, `Amazon_Sports_and_Outdoors`, `MassiveSTEPS_NewYork`,
`Gowalla_CA`, `Gowalla_TX`, `ml-1m`, and the reduced sets `KuaiRand_1K_sub`, `LastFM_sub`, `ml-20m_sub`, `Yelp_sub`,
`Amazon_Books_sub`.

**1. Build the reduced datasets (once)** — the full LastFM, ML-20M, Books, Yelp and KuaiRand are too large to train on:

```bash
.venv/bin/python scripts/make_subsets.py
```

**2. Check that a dataset fits in memory and estimate its epoch time** (a few real training and evaluation steps):

```bash
.venv/bin/python scripts/check_data.py -d Amazon_Beauty
```

**3. Train one dataset:**

```bash
.venv/bin/python scripts/train.py -m FreqMaT -d Amazon_Beauty                       # defaults: 100 epochs, early stopping 10
.venv/bin/python scripts/train.py -m FreqMaT -d Amazon_Beauty -s use_local=false    # ablation switch
.venv/bin/python scripts/train.py -m SASRec  -d Amazon_Beauty                       # RecBole baseline
```

**4. Train every dataset, 100 epochs each** (one fresh process per dataset, so the GPU is emptied and the model
re-initialised between datasets; an out-of-memory failure is retried once with half the batch size):

```bash
nohup .venv/bin/python scripts/run_all.py > run_all.log 2>&1 &                      # all 13, 100 epochs, no early stopping
.venv/bin/python scripts/run_all.py -d Amazon_Beauty ml-1m --epochs 100 --patience 10   # subset, with early stopping
```

**GPU choice:** training uses physical GPU 1 when the machine has one, otherwise GPU 0 (`src/freqmat/gpu.py`). Force a
GPU with `FREQMAT_GPU=0 python scripts/run_all.py` or `-s gpu_id=0`. Every run prints `DEVICE ...` at the start.

Summary table: `results/full_100.tsv`. Progress: `tail -f run_all.log`, or `outputs/full_100/<run>/history.csv`.

**5. Plot learning curves for the paper:**

```bash
.venv/bin/python scripts/plot_history.py outputs/full_100/* --out figures
```

**6. Ablations and analysis:**

```bash
.venv/bin/python scripts/run_ablation.py --epochs 1 --seeds 2024 --only freq_bsa    # quick grid check
.venv/bin/python scripts/run_ablation.py --grid configs/ablation/freqmat_v2_grid.yaml
.venv/bin/python scripts/analyze_oversmoothing.py --ckpt a=outputs/.../x.pth b=outputs/.../y.pth
```

Every run writes `outputs/<experiment>/<timestamp>_<model>_<dataset>_<hash>/` containing `config.json`, `metrics.json`,
`history.csv` (one row per epoch), the best checkpoint, and RecBole `log/` + `log_tensorboard/`.
Protocol (`configs/protocol.yaml`): leave-one-out split (per user: last item test, second-to-last validation, rest train), **full-ranking**,
HR/NDCG/Recall@{5,10}, 5-core. The best epoch is chosen on validation NDCG@10; test is evaluated once on that checkpoint.

## Layout

```
src/freqmat/
  models/        FreqMaT (RecBole model), FreqMaTLayer, FreqMaTConfig, modules/ (oscillator, frequency, mamba, local, attention, fusion, ffn, graph_fourier)
  data/          raw -> .inter converters (registry), downloads
  training/      train_and_eval() -> RunResult, HistoryTrainer (history.csv)
  experiments/   ablation grid runner
  analysis/      oversmoothing analysis
scripts/         thin CLIs: train, run_all, check_data, make_subsets, plot_history, prepare_data, download_raw, run_ablation, analyze_oversmoothing
configs/         protocol.yaml, model/, dataset/ (per-dataset overrides), ablation/
tests/           pytest suite (CPU; Mamba-3 tests need CUDA)
data/            raw/<dataset>/ and processed/<Name>/<Name>.inter   (git-ignored)
outputs/         run artefacts (git-ignored)
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

## TODO
- Baselines: GRU4Rec, BERT4Rec, FMLP-Rec, BSARec, Mamba4Rec, LRURec (register in `freqmat.models.CUSTOM_MODELS`)
- Analysis by sequence length / long-tail; params & speed
