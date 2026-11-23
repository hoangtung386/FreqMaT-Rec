"""Public download locations of the raw datasets: ``name -> (subdirectory of data/raw, url)``."""

SOURCES: dict[str, tuple[str, str]] = {
    "ml-1m": ("ml-1m", "https://files.grouplens.org/datasets/movielens/ml-1m.zip"),
    "ml-20m": ("ml-20m", "https://files.grouplens.org/datasets/movielens/ml-20m.zip"),
    "lastfm1k": ("lastfm", "https://zenodo.org/records/6090214/files/lastfm-dataset-1K.tar.gz"),
    "kuairand1k": ("kuairand", "https://zenodo.org/records/10439422/files/KuaiRand-1K.tar.gz"),
    "gowalla": ("gowalla", "https://snap.stanford.edu/data/loc-gowalla_totalCheckins.txt.gz"),
    "video_games": (
        "amazon",
        "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Video_Games_5.json.gz",
    ),
    "books": ("amazon", "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz"),
}
AMAZON_URL = "https://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_{}_5.json.gz"
