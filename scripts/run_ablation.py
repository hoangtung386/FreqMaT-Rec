"""Run the FreqMaT ablation grid into results/ablation_<dataset>.tsv.

python scripts/run_ablation.py                                                    # full grid from the YAML
python scripts/run_ablation.py --epochs 1 --seeds 2024 --only freq_bsa freq_gft   # quick check
"""

import argparse
from pathlib import Path

from freqmat.experiments.ablation import DEFAULT_GRID, load_grid, run_grid


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--grid", type=Path, default=DEFAULT_GRID)
    parser.add_argument("-d", "--dataset", default=None, help="override the dataset of the grid")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--seeds", type=int, nargs="*", default=None)
    parser.add_argument("--only", nargs="*", default=None, help="variant names to run")
    args = parser.parse_args()
    grid = load_grid(args.grid)
    if args.dataset:
        grid["dataset"] = args.dataset
    run_grid(grid, only=args.only, seeds=args.seeds, epochs=args.epochs)


if __name__ == "__main__":
    main()
