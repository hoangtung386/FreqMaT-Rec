# Experiments

## Protocol
Leave-one-out split, **full ranking** (no sampled metrics), 5-core filtering, metrics HR / NDCG / Recall @ {5, 10},
model selection on validation NDCG@10, early stopping after 10 epochs, cross-entropy loss, default seed 2024
(`configs/protocol.yaml`). Config layers, lowest to highest priority: `protocol.yaml` -> `model/<model>.yaml` ->
`dataset/<dataset>.yaml` -> command-line `-s key=value`.

## Single run
```bash
.venv/bin/python scripts/train.py -m FreqMaT -d Amazon_Beauty -e my_experiment -s freq_type=bsa
```
Artefacts go to `outputs/my_experiment/<timestamp>_FreqMaT_Amazon_Beauty_<hash>/`.

## Ablation grid
`configs/ablation/freqmat_grid.yaml` defines variants (config overrides) and seeds.
```bash
.venv/bin/python scripts/run_ablation.py -d Amazon_Beauty                      # variants x seeds
```
Outputs: `results/ablation_<dataset>.tsv` (mean/std per variant) and `results/ablation_<dataset>_runs.tsv` (per run,
with status and relative `run_dir`). A failing run is recorded and the grid continues.

`results/legacy_ablation_Amazon_Beauty.tsv` is the single-seed result from before the refactor (checkpoints under
`outputs/legacy/saved/`); `dense_ffn` and `attn_only` are missing there and differences between variants (~0.001 NDCG)
are within seed noise, so re-run with several seeds before drawing conclusions.

## Oversmoothing
```bash
.venv/bin/python scripts/analyze_oversmoothing.py --ckpt attn_only=<ckpt> freqmat=<ckpt>
```
Writes `results/oversmoothing.{json,png}`. Legacy checkpoints (`outputs/legacy/saved/`) still load.

## Baselines (TODO)
GRU4Rec, BERT4Rec, FMLP-Rec, BSARec, Mamba4Rec, LRURec: add as modules under `src/freqmat/models/` and register in
`CUSTOM_MODELS` (`src/freqmat/models/__init__.py`).
