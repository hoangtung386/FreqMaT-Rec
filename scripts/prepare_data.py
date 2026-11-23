"""Convert raw datasets to RecBole .inter files.  e.g.  python scripts/prepare_data.py --datasets Beauty ml-1m"""

import argparse
import logging

from freqmat import data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["Beauty", "ml-1m"],
        help=f"registered: {sorted(data.REGISTRY)}; otherwise an Amazon category "
        "(Beauty, Video_Games, Books, Sports_and_Outdoors, Toys_and_Games, ...)",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in args.datasets:
        data.convert(name)


if __name__ == "__main__":
    main()
