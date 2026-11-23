"""Assemble a RecBole ``Config`` from the layered YAML files under ``configs/``."""

import re
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import yaml

from freqmat import paths

_SCIENTIFIC_FLOAT = re.compile(r"[+-]?(\d+\.?\d*|\.\d+)[eE][+-]?\d+")


def _parse_value(text: str) -> Any:
    """YAML scalar, except that ``1e-3`` is a float (PyYAML's YAML 1.1 resolver would return a string)."""
    value = yaml.safe_load(text)
    if isinstance(value, str) and _SCIENTIFIC_FLOAT.fullmatch(value):
        return float(value)
    return value


def parse_overrides(items: Iterable[str]) -> dict[str, Any]:
    """Parse ``["key=value", ...]`` (values are YAML scalars, e.g. ``use_freq=false``)."""
    overrides = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep:
            raise ValueError(f"override must look like key=value, got {item!r}")
        overrides[key] = _parse_value(value)
    return overrides


def config_files(model: str, dataset: str) -> list[Path]:
    """Existing config layers, lowest to highest priority: protocol, model, dataset."""
    candidates = [
        paths.CONFIGS / "protocol.yaml",
        paths.CONFIGS / "model" / f"{model.lower()}.yaml",
        paths.CONFIGS / "dataset" / f"{dataset}.yaml",
    ]
    return [path for path in candidates if path.exists()]


def build_config(model: str, dataset: str, overrides: dict[str, Any] | None = None):
    """Return a RecBole ``Config`` for ``model`` on ``dataset`` with ``overrides`` applied last."""
    from recbole.config import Config

    from freqmat.models import resolve_model

    values = {"data_path": str(paths.DATA_PROCESSED), "checkpoint_dir": str(paths.OUTPUTS / "adhoc")}
    values.update(overrides or {})
    return Config(
        model=resolve_model(model),
        dataset=dataset,
        config_file_list=[str(f) for f in config_files(model, dataset)],
        config_dict=values,
    )
