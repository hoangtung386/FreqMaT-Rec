"""Train and evaluate a model on a dataset.  e.g.  python scripts/train.py -m FreqMaT -d ml-1m -s epochs=1"""

import argparse

from freqmat.config import parse_overrides
from freqmat.training.runner import train_and_eval


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("-m", "--model", default="FreqMaT")
    parser.add_argument("-d", "--dataset", default="ml-1m")
    parser.add_argument(
        "-s", "--set", nargs="*", default=[], metavar="KEY=VALUE", help="config overrides, e.g. use_freq=false epochs=1"
    )
    parser.add_argument("-e", "--experiment", default="adhoc", help="outputs/<experiment>/ group name")
    args = parser.parse_args()
    res = train_and_eval(args.model, args.dataset, parse_overrides(args.set), args.experiment)
    print("RUN_DIR   ", res.run_dir)
    print("CHECKPOINT", res.checkpoint)
    print(f"PEAK_GPU_MEM_GB {res.peak_gpu_gb:.2f}  PARAMS {res.params:,}")
    print("VALID", res.valid)
    print("TEST ", res.test)


if __name__ == "__main__":
    main()
