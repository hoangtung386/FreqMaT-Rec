"""Gowalla check-ins cut into California / Texas."""

import pandas as pd

from freqmat.data.io import raw_file, register, save_inter, to_unix_seconds

# Approximate bounding boxes (lat_min, lat_max, lon_min, lon_max). NOT exact state borders: a few points of
# neighbouring states (NV, NM, OK, ...) leak in, so numbers are not comparable with papers' "Gowalla-CA/TX".
BOXES = {
    "Gowalla_CA": (32.5, 42.0, -124.5, -114.1),
    "Gowalla_TX": (25.8, 36.5, -106.7, -93.5),
}


@register("Gowalla")
def convert() -> None:
    df = pd.read_csv(
        raw_file("gowalla", "loc-gowalla_totalCheckins.txt.gz"),
        sep="\t",
        header=None,
        names=["user_id", "ts", "lat", "lon", "item_id"],
    ).dropna()
    df["timestamp"] = to_unix_seconds(df["ts"])
    df["rating"] = 1.0
    for name, (lat_min, lat_max, lon_min, lon_max) in BOXES.items():
        save_inter(df[df.lat.between(lat_min, lat_max) & df.lon.between(lon_min, lon_max)], name)
