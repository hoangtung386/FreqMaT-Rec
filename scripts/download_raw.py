"""Resumable raw downloads into data/raw/<dataset>/.  e.g.  python scripts/download_raw.py ml-20m gowalla"""

import argparse
import logging

from freqmat.data.download import download
from freqmat.data.sources import SOURCES


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("names", nargs="*", help=f"any of {sorted(SOURCES)} (default: all)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    download(args.names or None)


if __name__ == "__main__":
    main()
