"""Model zoo: custom models registered here take precedence over RecBole's built-ins."""

from freqmat.models.config import FreqMaTConfig
from freqmat.models.freqmat import FreqMaT

CUSTOM_MODELS = {"FreqMaT": FreqMaT}


def resolve_model(name: str):
    """Return the model class for ``name`` (custom first, then RecBole)."""
    if name in CUSTOM_MODELS:
        return CUSTOM_MODELS[name]
    from recbole.utils import get_model

    return get_model(name)


__all__ = ["CUSTOM_MODELS", "FreqMaT", "FreqMaTConfig", "resolve_model"]
