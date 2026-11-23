"""Derived training sets: the same interactions, cut down to a size one GPU can train in reasonable time.

RecBole turns every prefix of a user's history into a training sample, so the cost of an epoch is the number of
interactions, and a 20M-row table needs tens of GB of RAM just to build the samples. A derived set applies, in order:
collapse consecutive repeats of the same item, k-core filtering, a seeded random sample of users, keeping only each
user's most recent ``max_history`` interactions, and k-core filtering again. Source files are never modified.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from freqmat import paths
from freqmat.data.io import COLUMNS, save_inter

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class Subset:
    source: str
    collapse_repeats: bool = False
    max_users: int | None = None
    max_history: int | None = None
    min_inter: int = 5
    seed: int = 2024


SUBSETS: dict[str, Subset] = {
    "ml-20m_sub": Subset("ml-20m", max_users=8_000, max_history=100),
    "Amazon_Books_sub": Subset("Amazon_Books", max_users=100_000),
    "Yelp_sub": Subset("Yelp", max_users=50_000),
    "KuaiRand_1K_sub": Subset("KuaiRand_1K", max_history=200),
    "LastFM_sub": Subset("LastFM", collapse_repeats=True, max_history=200),
}


def read_inter(name: str) -> pd.DataFrame:
    path = paths.DATA_PROCESSED / name / f"{name}.inter"
    df = pd.read_csv(path, sep="\t", skiprows=1, header=None, names=COLUMNS, dtype={"user_id": str, "item_id": str})
    return df.sort_values(["user_id", "timestamp"], kind="stable")


def collapse_repeats(df: pd.DataFrame) -> pd.DataFrame:
    """Drop an interaction when the same user's previous one was the same item (replays, not new choices)."""
    same = (df.user_id == df.user_id.shift()) & (df.item_id == df.item_id.shift())
    return df[~same]


def k_core(df: pd.DataFrame, k: int) -> pd.DataFrame:
    """Iteratively drop users and items with fewer than ``k`` interactions."""
    while True:
        n = len(df)
        df = df[df.item_id.map(df.item_id.value_counts()) >= k]
        df = df[df.user_id.map(df.user_id.value_counts()) >= k]
        if len(df) == n:
            return df


def reduce_frame(df: pd.DataFrame, spec: Subset) -> pd.DataFrame:
    """Apply ``spec`` to a ``(user, time)``-sorted interaction frame."""
    if spec.collapse_repeats:
        df = collapse_repeats(df)
    df = k_core(df, spec.min_inter)
    if spec.max_users is not None:
        users = df.user_id.unique()
        if len(users) > spec.max_users:
            keep = np.random.default_rng(spec.seed).choice(users, spec.max_users, replace=False)
            df = df[df.user_id.isin(keep)]
    if spec.max_history is not None:
        df = df.groupby("user_id", sort=False).tail(spec.max_history)
    return k_core(df, spec.min_inter)


def build_subset(name: str) -> None:
    spec = SUBSETS[name]
    df = reduce_frame(read_inter(spec.source), spec)
    save_inter(df, name)


__all__ = ["SUBSETS", "Subset", "build_subset", "reduce_frame"]
