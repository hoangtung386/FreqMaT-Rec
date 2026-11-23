import gzip
import json

import pandas as pd
import pytest

from freqmat import data, paths
from freqmat.data import io


@pytest.fixture
def sandbox(tmp_path, monkeypatch):
    """Redirect raw / processed data directories into a temp dir."""
    raw, processed = tmp_path / "raw", tmp_path / "processed"
    monkeypatch.setattr(paths, "DATA_RAW", raw)
    monkeypatch.setattr(paths, "DATA_PROCESSED", processed)
    return raw, processed


def read_inter(path):
    with open(path) as f:
        header = f.readline().rstrip("\n")
    return header, pd.read_csv(path, sep="\t", skiprows=1, header=None, names=io.COLUMNS)


def test_save_inter_sorts_and_writes_recbole_header(sandbox):
    _, processed = sandbox
    df = pd.DataFrame({"user_id": ["b", "a", "a"], "item_id": ["x", "y", "z"], "rating": 1.0, "timestamp": [5, 9, 1]})
    out = io.save_inter(df, "Toy")
    header, got = read_inter(out)
    assert out == processed / "Toy" / "Toy.inter"
    assert header == io.HEADER
    assert got.user_id.tolist() == ["a", "a", "b"] and got.timestamp.tolist() == [1, 9, 5]


def test_missing_raw_file_gives_actionable_error(sandbox):
    with pytest.raises(SystemExit, match="download_raw"):
        io.raw_file("gowalla", "nope.gz")


def test_amazon_converter(sandbox):
    raw, processed = sandbox
    (raw / "amazon").mkdir(parents=True)
    with gzip.open(raw / "amazon" / "reviews_Toy_5.json.gz", "wt") as f:
        for u, i, t in [("u1", "i1", 10), ("u1", "i2", 20)]:
            f.write(json.dumps({"reviewerID": u, "asin": i, "overall": 5.0, "unixReviewTime": t}) + "\n")
    data.convert("Toy")  # not registered -> treated as an Amazon category
    _, got = read_inter(processed / "Amazon_Toy" / "Amazon_Toy.inter")
    assert got.item_id.tolist() == ["i1", "i2"]


def test_gowalla_split_by_bounding_box(sandbox):
    raw, processed = sandbox
    (raw / "gowalla").mkdir(parents=True)
    rows = ["1\t2010-01-01T00:00:00Z\t37.0\t-122.0\tloc_ca", "2\t2010-01-02T00:00:00Z\t30.0\t-97.0\tloc_tx"]
    with gzip.open(raw / "gowalla" / "loc-gowalla_totalCheckins.txt.gz", "wt") as f:
        f.write("\n".join(rows) + "\n")
    data.convert("Gowalla")
    _, ca = read_inter(processed / "Gowalla_CA" / "Gowalla_CA.inter")
    _, tx = read_inter(processed / "Gowalla_TX" / "Gowalla_TX.inter")
    assert ca.item_id.tolist() == ["loc_ca"] and tx.item_id.tolist() == ["loc_tx"]


def test_registry_contains_all_documented_datasets():
    assert {"ml-1m", "ml-20m", "LastFM", "Gowalla", "NewYork", "Yelp", "KuaiRand"} <= set(data.REGISTRY)
