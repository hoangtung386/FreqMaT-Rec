import pytest

from freqmat import paths
from freqmat.config import config_files, parse_overrides
from freqmat.models import FreqMaT, resolve_model


def test_parse_overrides_uses_yaml_scalars():
    assert parse_overrides(["use_freq=false", "epochs=3", "freq_type=bsa", "lr=1e-3"]) == {
        "use_freq": False,
        "epochs": 3,
        "freq_type": "bsa",
        "lr": 1e-3,
    }


def test_parse_overrides_rejects_malformed():
    with pytest.raises(ValueError):
        parse_overrides(["oops"])


def test_config_layers_are_ordered_and_optional():
    files = config_files("FreqMaT", "LastFM")
    assert [f.parent.name for f in files] == ["configs", "model", "dataset"]
    assert [f.name for f in config_files("SASRec", "ml-1m")] == ["protocol.yaml"]


def test_paths_point_into_the_repository():
    assert (paths.ROOT / "pyproject.toml").exists()
    assert paths.CONFIGS.is_dir()


def test_resolve_model_prefers_custom():
    assert resolve_model("FreqMaT") is FreqMaT
    assert resolve_model("SASRec").__name__ == "SASRec"
