"""Train one model on every dataset, one fresh process per dataset, and tabulate the results.

e.g.  python scripts/run_all.py                       # all 13 datasets, 100 epochs each, full run
      python scripts/run_all.py -d Amazon_Beauty ml-1m --epochs 100 --patience 10

Each dataset runs in its own Python process (``scripts/train.py``). When the process exits the CUDA context is
destroyed, so the GPU is empty before the next dataset is loaded and the model is initialised again from scratch.
Checkpoints, ``history.csv``, ``metrics.json``, the RecBole log and TensorBoard files stay on disk under
``outputs/<experiment>/``. A run that ends with CUDA out-of-memory is retried once with half the batch size.
A dataset whose run directory already holds ``metrics.json`` is skipped, so the script can simply be started
again after an interruption, and it stops at once if the GPU itself fails (driver / hardware error)
instead of failing every remaining dataset.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

from freqmat import paths
from freqmat.gpu import choose_gpu

DATASETS = [
    "Amazon_Beauty",
    "Amazon_Toys_and_Games",
    "Amazon_Video_Games",
    "Amazon_Sports_and_Outdoors",
    "MassiveSTEPS_NewYork",
    "KuaiRand_1K_sub",
    "LastFM_sub",
    "ml-20m_sub",
    "Yelp_sub",
    "Gowalla_CA",
    "Gowalla_TX",
    "Amazon_Books_sub",
    "ml-1m",
]


def gpu_used_mb() -> int | None:
    """Memory in use on the GPU the runs will use, according to the driver (``None`` without ``nvidia-smi``)."""
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", choose_gpu() or "0"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return int(out.strip().splitlines()[0])
    except (OSError, subprocess.CalledProcessError, ValueError):
        return None


GPU_FAILURES = ("unspecified launch failure", "invalid argument to exchangeDevice", "CUDA error", "no CUDA GPUs")


def finished_run(experiment: str, dataset: str) -> Path | None:
    """A completed run of ``dataset`` under ``outputs/<experiment>/`` (one that wrote ``metrics.json``)."""
    for run in sorted((paths.OUTPUTS / experiment).glob(f"*_{dataset}_*")):
        if (run / "metrics.json").exists():
            return run
    return None


def batch_size_for(dataset: str, model: str) -> int:
    from freqmat.config import build_config

    return int(build_config(model, dataset)["train_batch_size"])


def run_one(dataset: str, args, log_dir: Path, batch_size: int | None = None) -> tuple[int, Path]:
    overrides = [f"epochs={args.epochs}", f"stopping_step={args.patience}", f"seed={args.seed}", *args.set]
    if batch_size:
        overrides.append(f"train_batch_size={batch_size}")
    log_file = log_dir / f"{dataset}{'_bs' + str(batch_size) if batch_size else ''}.log"
    cmd = [sys.executable, str(paths.ROOT / "scripts" / "train.py"), "-m", args.model, "-d", dataset]
    cmd += ["-e", args.experiment, "-s", *overrides]
    with open(log_file, "w") as f:
        code = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False).returncode
    return code, log_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-d", "--datasets", nargs="+", default=DATASETS)
    parser.add_argument("-m", "--model", default="FreqMaT")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument(
        "--patience",
        type=int,
        default=None,
        help="early-stopping patience in epochs; default = --epochs, i.e. always train the full number of epochs",
    )
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("-e", "--experiment", default="full_100")
    parser.add_argument("-s", "--set", nargs="*", default=[], metavar="KEY=VALUE")
    args = parser.parse_args()
    args.patience = args.epochs if args.patience is None else args.patience

    out_dir = paths.OUTPUTS / args.experiment
    log_dir = out_dir / "stdout"
    log_dir.mkdir(parents=True, exist_ok=True)
    summary = paths.RESULTS / f"{args.experiment}.tsv"
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    rows = ["dataset\tstatus\tgpu_mb_before\tminutes\tbatch_size\tstdout"]
    for dataset in args.datasets:
        done = finished_run(args.experiment, dataset)
        if done is not None:
            print(f"== {dataset}  already finished ({done.name}), skipped", flush=True)
            rows.append(f"{dataset}\tdone\t\t\t\t{done}")
            continue
        before = gpu_used_mb()
        if before is None:
            raise SystemExit("nvidia-smi cannot reach the GPU: fix the driver / reboot, then start run_all.py again")
        print(f"== {dataset}  (GPU in use before: {before} MB)", flush=True)
        start = time.time()
        code, log_file = run_one(dataset, args, log_dir)
        batch = batch_size_for(dataset, args.model)
        if code != 0 and "out of memory" in log_file.read_text():
            batch //= 2
            print(f"   out of memory, retrying with train_batch_size={batch}", flush=True)
            code, log_file = run_one(dataset, args, log_dir, batch)
        status = "ok" if code == 0 else f"failed (exit {code})"
        gpu_broken = code != 0 and any(msg in log_file.read_text() for msg in GPU_FAILURES)
        print(f"   {status} in {(time.time() - start) / 60:.1f} min", flush=True)
        rows.append(f"{dataset}\t{status}\t{before}\t{(time.time() - start) / 60:.1f}\t{batch}\t{log_file}")
        summary.write_text("\n".join(rows) + "\n")
        if gpu_broken:
            raise SystemExit(
                f"GPU failure while training {dataset}; stopping. Start run_all.py again once the GPU is healthy."
            )


if __name__ == "__main__":
    main()
