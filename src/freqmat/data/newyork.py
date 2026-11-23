"""Massive-STEPS New York (Foursquare) check-ins."""

import pandas as pd

from freqmat.data.io import raw_file, register, save_inter


@register("NewYork")
def convert() -> None:
    df = pd.read_csv(raw_file("newyork", "new_york_checkins.csv"), usecols=["user_id", "venue_id", "timestamp"])
    df = df.rename(columns={"venue_id": "item_id"})
    df["timestamp"] = pd.to_datetime(df["timestamp"]).astype("int64") // 10**9
    df["rating"] = 1.0
    save_inter(df, "MassiveSTEPS_NewYork")
