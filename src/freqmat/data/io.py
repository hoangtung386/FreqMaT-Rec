"""Shared helpers for converting raw datasets into RecBole atomic files."""

import logging
from pathlib import Path
from typing import Callable

import pandas as pd

from freqmat import paths

LOGGER = logging.getLogger(__name__)
HEADER = "user_id:token\titem_id:token\trating:float\ttimestamp:float"
COLUMNS = ["user_id", "item_id", "rating", "timestamp"]

Converter = Callable[[], None]
REGISTRY: dict[str, Converter] = {}


def register(*names: str):
    """Decorator registering a converter under one or more CLI dataset names."""

    def decorator(func: Converter) -> Converter:
        for name in names:
            REGISTRY[name] = func
        return func

    return decorator


def raw_file(dataset_dir: str, filename: str) -> Path:
    """Path of a raw download, exiting with a helpful message when it is missing."""
    path = paths.DATA_RAW / dataset_dir / filename
    if not path.exists():
        raise SystemExit(f"missing {path} -- run scripts/download_raw.py {dataset_dir} (or see docs/datasets.md)")
    return path


def to_unix_seconds(values: pd.Series) -> pd.Series:
    """Datetime-like strings -> integer unix seconds (UTC)."""
    return pd.to_datetime(values, utc=True).astype("int64") // 10**9


def save_inter(df: pd.DataFrame, name: str) -> Path:
    """Write ``data/processed/<name>/<name>.inter`` sorted by (user, time)."""
    out_dir = paths.DATA_PROCESSED / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{name}.inter"
    df = df[COLUMNS].sort_values(["user_id", "timestamp"])
    with open(out, "w") as f:
        f.write(HEADER + "\n")
        df.to_csv(f, sep="\t", header=False, index=False)
    LOGGER.info("%s: %d interactions, %d users, %d items", name, len(df), df.user_id.nunique(), df.item_id.nunique())
    return out
