"""Dataset conversion: raw downloads -> RecBole atomic files (``data/processed/<name>/<name>.inter``).

Importing this package registers every converter; Amazon categories are addressed by bare category name
(``Beauty``, ``Video_Games``, ...).
"""

from freqmat.data import gowalla, kuairand, lastfm, movielens, newyork, yelp  # noqa: F401  (registration)
from freqmat.data.amazon import convert as convert_amazon
from freqmat.data.io import REGISTRY


def convert(name: str) -> None:
    """Convert dataset ``name`` (registered name, otherwise an Amazon category)."""
    if name in REGISTRY:
        REGISTRY[name]()
    else:
        convert_amazon(name)


__all__ = ["REGISTRY", "convert"]
