"""Build the derived training sets listed in freqmat.data.reduce.SUBSETS.  e.g.  python scripts/make_subsets.py"""

import argparse
import logging

from freqmat.data.reduce import SUBSETS, build_subset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", default=sorted(SUBSETS), help=f"default: all of {sorted(SUBSETS)}")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    for name in args.names:
        build_subset(name)


if __name__ == "__main__":
    main()
