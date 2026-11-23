"""MovieLens 1M / 20M."""

import zipfile

import pandas as pd

from freqmat import paths
from freqmat.data.download import fetch
from freqmat.data.io import raw_file, register, save_inter
from freqmat.data.sources import SOURCES


@register("ml-1m")
def ml1m() -> None:
    subdir, url = SOURCES["ml-1m"]
    path = fetch(url, paths.DATA_RAW / subdir / "ml-1m.zip")
    with zipfile.ZipFile(path) as z:
        df = pd.read_csv(
            z.open("ml-1m/ratings.dat"), sep="::", engine="python", names=["user_id", "item_id", "rating", "timestamp"]
        )
    save_inter(df, "ml-1m")


@register("ml-20m")
def ml20m() -> None:
    with zipfile.ZipFile(raw_file("ml-20m", "ml-20m.zip")) as z:
        df = pd.read_csv(z.open("ml-20m/ratings.csv"))
    save_inter(df.rename(columns={"userId": "user_id", "movieId": "item_id"}), "ml-20m")
