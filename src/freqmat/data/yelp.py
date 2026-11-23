"""Yelp Open Dataset reviews (manual download, see docs/datasets.md)."""

import pandas as pd

from freqmat.data.io import raw_file, register, save_inter

CHUNK_ROWS = 500_000


@register("Yelp")
def convert() -> None:
    """Reviews as interactions, streamed in chunks to avoid holding the 5 GB JSON in memory."""
    path = raw_file("yelp", "yelp_academic_dataset_review.json")
    chunks = []
    for chunk in pd.read_json(path, lines=True, chunksize=CHUNK_ROWS, dtype=False, convert_dates=False):
        chunks.append(
            chunk[["user_id", "business_id", "stars", "date"]].set_axis(
                ["user_id", "item_id", "rating", "timestamp"], axis=1
            )
        )
    df = pd.concat(chunks, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**9
    save_inter(df, "Yelp")
