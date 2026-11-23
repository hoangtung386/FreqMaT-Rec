"""KuaiRand-1K clicks."""

import glob
import tarfile

import pandas as pd

from freqmat import paths
from freqmat.data.io import raw_file, register, save_inter


@register("KuaiRand")
def convert() -> None:
    """Clicked videos from the two *standard* logs (the random-exposure log is excluded: it is for debiasing)."""
    folder = paths.DATA_RAW / "kuairand"
    pattern = str(folder / "*" / "data" / "log_standard_*_1k.csv")
    if not glob.glob(pattern):  # extract once
        with tarfile.open(raw_file("kuairand", "KuaiRand-1K.tar.gz")) as tar:
            tar.extractall(folder)
    frames = [
        pd.read_csv(f, usecols=["user_id", "video_id", "time_ms", "is_click"]) for f in sorted(glob.glob(pattern))
    ]
    df = pd.concat(frames)
    df = df[df.is_click == 1]
    df = df.assign(item_id=df.video_id, rating=1.0, timestamp=df.time_ms // 1000)
    save_inter(df, "KuaiRand_1K")
