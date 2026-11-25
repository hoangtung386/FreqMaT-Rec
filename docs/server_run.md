# Running every dataset on the server

Goal: train FreqMaT on all 13 datasets (100 epochs each) on **GPU 1** of the server. The code picks physical GPU 1 when
the machine has one and falls back to GPU 0 otherwise (`src/freqmat/gpu.py`), so nothing has to be edited.

## 1. Set up (once)

```bash
git clone git@github.com:hoangtung386/FreqMaT-Rec.git && cd FreqMaT-Rec
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu130 -e ".[dev]"   # match the server's CUDA version
./scripts/install_mamba.sh                                               # Mamba-3 kernels (needs CUDA)
nvidia-smi -L                                                            # expect at least two GPUs
.venv/bin/python -m pytest -q                                            # expect all passed (CUDA tests need a GPU)
```

If the server has no usable `mamba-ssm`, add `-s mamba_type=mamba1` to every command below (slower pure-PyTorch scan).

## 2. Data

`data/` is git-ignored, so copy `data/processed/` from the development machine (the folder layout must stay
`data/processed/<Name>/<Name>.inter`):

```bash
rsync -avP data/processed/ <server>:<path>/FreqMaT-Rec/data/processed/
```

Needed: `Amazon_Beauty`, `Amazon_Toys_and_Games`, `Amazon_Video_Games`, `Amazon_Sports_and_Outdoors`,
`MassiveSTEPS_NewYork`, `Gowalla_CA`, `Gowalla_TX`, `ml-1m` and the reduced sets `KuaiRand_1K_sub`, `LastFM_sub`,
`ml-20m_sub`, `Yelp_sub`, `Amazon_Books_sub`. Copy the `_sub` folders as they are, so the server trains on identical
data. Only if one is missing, build it from the full file with `.venv/bin/python scripts/make_subsets.py <name>`.

## 3. Check, then run

```bash
.venv/bin/python scripts/check_data.py -d Amazon_Beauty --steps 10    # must print CHECK {...}; look at params_M and memory
nohup .venv/bin/python scripts/run_all.py > outputs/run_all.log 2>&1 &
```

- The first lines of each run print `DEVICE cuda (physical GPU id 1)`: confirm it says **1**.
- Every dataset runs in its own process, so GPU memory is emptied between datasets. A CUDA out-of-memory failure is retried
  once with half the batch size. If the GPU itself fails, the script stops; start it again and datasets that already
  finished are skipped.
- Do not stop or kill processes that belong to other users of the server. If GPU 1 is busy, report it and wait.

## 4. Results

| What | Where |
|---|---|
| progress | `tail -f outputs/run_all.log` |
| table (one row per dataset, status and minutes) | `results/full_100.tsv` |
| per-epoch curve of each run | `outputs/full_100/<run>/history.csv` |
| final validation / test metrics of the best epoch | `outputs/full_100/<run>/metrics.json` |
| best checkpoint | `outputs/full_100/<run>/*.pth` |
| figures | `.venv/bin/python scripts/plot_history.py outputs/full_100/* --out figures` |

The best epoch is chosen on validation NDCG@10. Test is evaluated once on that checkpoint and is never used for selection.
`outputs/` is git-ignored: copy it back with `rsync`, do not commit it.

Expect roughly 2 to 3 days for all 13 datasets at 100 epochs (ml-1m alone takes about half of that). Earlier runs on the
development machine reached their best validation score within the first few epochs and then overfit, so
`run_all.py --patience 10` finishes in a fraction of the time; use it only if that trade-off is accepted.

## 5. Rules for whoever runs it

- Do not change code, configs or the protocol (`configs/`, `src/`) without asking; fix only the environment (packages, paths).
  Keep `train_batch_size` as configured even though the server GPU may be larger, so results stay comparable.
- Never use the test split to choose an epoch, a checkpoint or a hyper-parameter. The protocol is fixed: temporal 80:10:10
  split, checkpoint chosen on validation NDCG@10, test scored once.
- Do not delete `outputs/`. Report errors verbatim.
- If the rows of `make_subsets.py` differ from these, report it: `ml-20m_sub` 528,031 interactions / 8,000 users / 6,481 items;
  `Amazon_Books_sub` 860,869 / 60,351 / 62,984; `Yelp_sub` 603,571 / 40,271 / 35,447; `KuaiRand_1K_sub` 91,962 / 998 / 10,513;
  `LastFM_sub` 159,890 / 987 / 7,063.

## 6. When it finishes, report per dataset

Best epoch, validation NDCG@10 and Recall@10, test NDCG@10 / HR@10 / Recall@10 (from `metrics.json`), parameter count,
run time, and any dataset that failed with the reason. Do not judge model quality without a baseline run under the same
protocol.
