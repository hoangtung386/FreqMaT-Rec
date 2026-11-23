"""Compatibility shims for third-party libraries."""

import torch

_PATCHED = False


def allow_unsafe_torch_load() -> None:
    """Make ``torch.load`` default to ``weights_only=False`` (idempotent).

    RecBole 1.2 loads its own, trusted checkpoints, but torch>=2.6 defaults to
    ``weights_only=True`` which breaks that. Only call this for checkpoints you produced.
    """
    global _PATCHED
    if _PATCHED:
        return
    original = torch.load
    torch.load = lambda *args, **kwargs: original(*args, **{"weights_only": False, **kwargs})
    _PATCHED = True
