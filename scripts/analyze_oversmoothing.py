"""Oversmoothing analysis of trained checkpoints (BSARec Fig. 3).

python scripts/analyze_oversmoothing.py --ckpt attn_only=outputs/.../A.pth freqmat=outputs/.../B.pth
"""

import argparse
from pathlib import Path

from freqmat import paths
from freqmat.analysis.oversmoothing import run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ckpt", nargs="+", required=True, metavar="LABEL=PATH")
    parser.add_argument("--n-seq", type=int, default=2000)
    parser.add_argument(
        "--out", type=Path, default=paths.RESULTS / "oversmoothing", help="output path without extension"
    )
    args = parser.parse_args()
    run(dict(item.split("=", 1) for item in args.ckpt), args.n_seq, args.out)


if __name__ == "__main__":
    main()
