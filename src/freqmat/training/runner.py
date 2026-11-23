"""Train and evaluate one model on one dataset, writing every artefact into a dedicated run directory."""

import contextlib
import gc
import hashlib
import json
import logging
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

from freqmat import paths
from freqmat.compat import allow_unsafe_torch_load
from freqmat.config import build_config
from freqmat.training.history import HistoryTrainer


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


def release_gpu(tensorboard: Any = None) -> None:
    """Close the run's loggers and TensorBoard writer, then return the cached GPU memory to the driver.

    The caller must have dropped its own references to the model, optimizer and data loaders first; checkpoints are
    already on disk. Handlers are removed because RecBole attaches them to the root logger, and a second run in the
    same process would otherwise keep writing into the first run's log file.
    """
    if tensorboard is not None:
        tensorboard.close()
    root = logging.getLogger()
    for handler in root.handlers[:]:
        handler.close()
        root.removeHandler(handler)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def train_and_eval(
    model: str,
    dataset: str,
    overrides: dict[str, Any] | None = None,
    experiment: str = "adhoc",
    show_progress: bool = True,
) -> RunResult:
    """Fit ``model`` on ``dataset`` (temporal 80:10:10 split, full ranking), evaluate the best checkpoint on test."""
    allow_unsafe_torch_load()
    from recbole.data import create_dataset, data_preparation
    from recbole.utils import init_logger, init_seed

    from freqmat.models import resolve_model

    overrides = dict(overrides or {})
    run_dir = make_run_dir(experiment, model, dataset, overrides)
    config = build_config(model, dataset, {**overrides, "checkpoint_dir": str(run_dir)})
    (run_dir / "config.json").write_text(json.dumps(config.final_config_dict, indent=2, default=str))

    peak_gpu_gb = params = 0
    trainer = None
    try:
        with working_directory(run_dir):
            init_seed(config["seed"], config["reproducibility"])
            init_logger(config)
            data = create_dataset(config)
            train, valid, test = data_preparation(config, data)
            net = resolve_model(model)(config, train.dataset).to(config["device"])
            params = sum(p.numel() for p in net.parameters())
            trainer = HistoryTrainer(config, net, run_dir / "history.csv")
            _, valid_result = trainer.fit(train, valid, saved=True, show_progress=show_progress)
            test_result = trainer.evaluate(test, load_best_model=True, show_progress=show_progress)
            checkpoint = Path(trainer.saved_model_file)
            peak_gpu_gb = torch.cuda.max_memory_allocated() / 2**30 if torch.cuda.is_available() else 0.0
    finally:
        writer = getattr(trainer, "tensorboard", None)
        trainer = net = train = valid = test = data = None  # drop every GPU reference before emptying the cache
        release_gpu(writer)

    result = RunResult(
        run_dir=run_dir,
        checkpoint=checkpoint,
        valid=dict(valid_result),
        test=dict(test_result),
        params=params,
        peak_gpu_gb=peak_gpu_gb,
    )
    (run_dir / "metrics.json").write_text(json.dumps(result.to_json(), indent=2))
    return result
