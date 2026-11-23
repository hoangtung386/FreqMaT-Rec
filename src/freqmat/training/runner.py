"""Train and evaluate one model on one dataset, writing every artefact into a dedicated run directory."""

import contextlib
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from freqmat import paths
from freqmat.compat import allow_unsafe_torch_load
from freqmat.config import build_config


@dataclass
class RunResult:
    run_dir: Path
    checkpoint: Path
    valid: dict[str, float]
    test: dict[str, float]
    params: int
    peak_gpu_gb: float

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["run_dir"] = _relative(self.run_dir)
        data["checkpoint"] = _relative(self.checkpoint)
        return data


def _relative(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(paths.ROOT))
    except ValueError:
        return str(path)


def make_run_dir(experiment: str, model: str, dataset: str, overrides: dict[str, Any]) -> Path:
    """``outputs/<experiment>/<timestamp>_<model>_<dataset>_<hash of overrides>``."""
    digest = hashlib.md5(json.dumps(overrides, sort_keys=True, default=str).encode()).hexdigest()[:6]
    name = f"{time.strftime('%Y%m%d-%H%M%S')}_{model}_{dataset}_{digest}"
    run_dir = paths.OUTPUTS / experiment / name
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


@contextlib.contextmanager
def working_directory(path: Path):
    """Temporarily ``chdir``; RecBole writes ``log/`` and ``log_tensorboard/`` relative to the cwd."""
    previous = os.getcwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def train_and_eval(
    model: str,
    dataset: str,
    overrides: dict[str, Any] | None = None,
    experiment: str = "adhoc",
    show_progress: bool = True,
) -> RunResult:
    """Fit ``model`` on ``dataset`` (leave-one-out, full ranking), evaluate the best checkpoint on test."""
    allow_unsafe_torch_load()
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import get_trainer, init_logger, init_seed

    from freqmat.models import resolve_model

    overrides = dict(overrides or {})
    run_dir = make_run_dir(experiment, model, dataset, overrides)
    config = build_config(model, dataset, {**overrides, "checkpoint_dir": str(run_dir)})
    (run_dir / "config.json").write_text(json.dumps(config.final_config_dict, indent=2, default=str))

    with working_directory(run_dir):
        init_seed(config["seed"], config["reproducibility"])
        init_logger(config)
        data = create_dataset(config)
        train, valid, test = data_preparation(config, data)
        net = resolve_model(model)(config, train.dataset).to(config["device"])
        trainer = get_trainer(config["MODEL_TYPE"], config["model"])(config, net)
        _, valid_result = trainer.fit(train, valid, saved=True, show_progress=show_progress)
        test_result = trainer.evaluate(test, load_best_model=True, show_progress=show_progress)

    result = RunResult(
        run_dir=run_dir,
        checkpoint=Path(trainer.saved_model_file),
        valid=dict(valid_result),
        test=dict(test_result),
        params=sum(p.numel() for p in net.parameters()),
        peak_gpu_gb=torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0,
    )
    (run_dir / "metrics.json").write_text(json.dumps(result.to_json(), indent=2))
    return result
