# Experiments

## Protocol
Global temporal split **80 : 10 : 10** (all interactions sorted by time: the first 80% train, the next 10% validation,
the last 10% test), **full ranking** (no sampled metrics), 5-core filtering, metrics HR / NDCG / Recall @ {5, 10},
model selection on validation NDCG@10, cross-entropy loss, default seed 2024
(`configs/protocol.yaml`). Config layers, lowest to highest priority: `protocol.yaml` -> `model/<model>.yaml` ->
`dataset/<dataset>.yaml` -> command-line `-s key=value`.

## Single run
```bash
.venv/bin/python scripts/train.py -m FreqMaT -d Amazon_Beauty -e my_experiment -s freq_type=bsa
```
Artefacts go to `outputs/my_experiment/<timestamp>_FreqMaT_Amazon_Beauty_<hash>/`.

## Data split and leakage
- Every sample is "history before time t, predict the item at time t". A validation or test sample's history may contain
  training-period items (that is what the model would know at that time), but no training target is newer than a
  validation target, and no validation target is newer than a test target.
- The test split is evaluated once, on the checkpoint with the best **validation** NDCG@10. It is never used to pick
  the epoch, the checkpoint or any hyper-parameter, and `history.csv` logs validation only.
- Popularity statistics (`pop_debias`, `pop_align_weight`) are counted from training targets only.
- 5-core filtering is applied to the whole table before splitting (standard practice); it keeps only items and users
  with at least 5 interactions overall.
- Timestamps in the Amazon sets have day resolution, so interactions that share the boundary day can fall on either side.
- Items that appear only after the training period (about 6% of validation and 19% of test targets on Toys) have an
  untrained embedding, which caps the achievable metric. This is the cost of a temporal split.

## Full training: 200 epochs on every dataset
```bash
.venv/bin/python scripts/run_all.py                          # all 13 datasets, 200 epochs each, no early stopping
.venv/bin/python scripts/run_all.py -d Amazon_Beauty --epochs 200 --patience 10
```
Each dataset runs in a fresh process, so the GPU is empty and the model is re-initialised before the next one. The best
checkpoint (validation NDCG@10) is saved at every improvement and evaluated on test at the end. A CUDA out-of-memory
failure is retried once with half the batch size. Summary: `results/full_200.tsv`.

Per run, under `outputs/<experiment>/<timestamp>_<model>_<dataset>_<hash>/`:

| File | Content |
|---|---|
| `history.csv` | one row per epoch: train loss, epoch time, every validation metric, best score so far, peak GPU memory, learned `extrap_dt` and `pop_lambda` |
| `metrics.json` | final validation / test metrics of the best checkpoint, parameter count, peak GPU memory |
| `config.json` | the full resolved configuration |
| `*.pth` | best checkpoint |
| `log/` | RecBole text log (every epoch line, parameter summary) |
| `log_tensorboard/` | TensorBoard scalars (`Loss/Train`, validation score) |

Learning curves for the paper: `python scripts/plot_history.py outputs/full_200/* --out figures` (PDF and PNG per run).

Check memory and speed of a dataset before a long run: `python scripts/check_data.py -d <dataset>`.

## Ablation grid
`configs/ablation/freqmat_grid.yaml` defines variants (config overrides) and seeds.
```bash
.venv/bin/python scripts/run_ablation.py -d Amazon_Beauty                      # variants x seeds
```
Outputs: `results/ablation_<dataset>.tsv` (mean/std per variant) and `results/ablation_<dataset>_runs.tsv` (per run,
with status and relative `run_dir`). A failing run is recorded and the grid continues.

Differences between variants are typically ~0.001 NDCG, i.e. within seed noise: use several seeds before drawing conclusions.

## Oversmoothing
```bash
.venv/bin/python scripts/analyze_oversmoothing.py --ckpt attn_only=<ckpt> freqmat=<ckpt>
```
Writes `results/oversmoothing.{json,png}`.

## Baselines (TODO)
GRU4Rec, BERT4Rec, FMLP-Rec, BSARec, Mamba4Rec, LRURec: add as modules under `src/freqmat/models/` and register in
`CUSTOM_MODELS` (`src/freqmat/models/__init__.py`).
