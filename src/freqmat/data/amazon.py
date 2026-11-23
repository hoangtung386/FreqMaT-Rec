"""Amazon-2014 5-core reviews."""

import gzip
import json

import pandas as pd

from freqmat import paths
from freqmat.data.download import fetch
from freqmat.data.io import COLUMNS, save_inter
from freqmat.data.sources import AMAZON_URL


def convert(category: str) -> None:
    """``reviews_<category>_5.json.gz`` -> ``data/processed/Amazon_<category>`` (downloads if missing)."""
    filename = f"reviews_{category}_5.json.gz"
    path = fetch(AMAZON_URL.format(category), paths.DATA_RAW / "amazon" / filename)
    rows = []
    with gzip.open(path, "rt") as f:
        for line in f:
            review = json.loads(line)
            rows.append((review["reviewerID"], review["asin"], review["overall"], review["unixReviewTime"]))
    save_inter(pd.DataFrame(rows, columns=COLUMNS), f"Amazon_{category}")
