"""LastFM-1K listening events."""

import tarfile

import pandas as pd

from freqmat.data.io import raw_file, register, save_inter


@register("LastFM")
def convert() -> None:
    """Item = artist (standard for sequential rec); timestamp = listening time."""
    with tarfile.open(raw_file("lastfm", "lastfm-dataset-1K.tar.gz")) as tar:
        member = next(m for m in tar.getmembers() if "timestamp-artid" in m.name)
        df = pd.read_csv(
            tar.extractfile(member),
            sep="\t",
            header=None,
            quoting=3,
            on_bad_lines="skip",
            usecols=[0, 1, 2, 3],
            names=["user_id", "ts", "mbid", "artist"],
        )
    df["item_id"] = df["mbid"].fillna(df["artist"]).astype(str)
    df["timestamp"] = pd.to_datetime(df["ts"], utc=True, errors="coerce").astype("int64") // 10**9
    df = df.dropna(subset=["timestamp"])
    df["rating"] = 1.0
    save_inter(df, "LastFM")
