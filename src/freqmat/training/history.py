"""Per-epoch training log (``history.csv``) for plotting learning curves.

RecBole only prints the epoch lines to a text log and returns the best validation result. :class:`HistoryTrainer`
also appends one CSV row per epoch, flushed immediately, so an interrupted run still leaves a usable curve. Only the
validation split is logged here; the test split is evaluated once, on the best checkpoint, after training.
"""

import csv
import time
from pathlib import Path

import torch
from recbole.trainer import Trainer

LEARNED_SCALARS = ("extrap_dt", "pop_lambda")


class HistoryTrainer(Trainer):
    """RecBole ``Trainer`` that records epoch, loss, validation metrics, timings, memory and learned scalars."""

    def __init__(self, config, model, history_path: Path):
        super().__init__(config, model)
        self.history_path = Path(history_path)
        self.history: list[dict] = []
        self._row: dict = {}
        self._fieldnames: list[str] | None = None

    def _train_epoch(self, train_data, epoch_idx, loss_func=None, show_progress=False):
        start = time.time()
        total = super()._train_epoch(train_data, epoch_idx, loss_func, show_progress)
        batches = max(len(train_data), 1)
        mean = (sum(total) if isinstance(total, tuple) else total) / batches
        self._row = {
            "epoch": epoch_idx,
            "train_loss": round(mean, 6),
            "train_time_s": round(time.time() - start, 2),
            "learning_rate": self.optimizer.param_groups[0]["lr"],
        }
        return total

    def _valid_epoch(self, valid_data, show_progress=False):
        start = time.time()
        score, result = super()._valid_epoch(valid_data, show_progress)
        row = {
            **self._row,
            "valid_time_s": round(time.time() - start, 2),
            "valid_score": float(score),
            **{f"valid_{k}": float(v) for k, v in result.items()},
            "best_valid_score": max([float(score)] + [h["valid_score"] for h in self.history]),
            "peak_gpu_gb": round(torch.cuda.max_memory_allocated() / 2**30, 3) if torch.cuda.is_available() else 0.0,
        }
        for name in LEARNED_SCALARS:
            param = getattr(self.model, name, None)
            if param is not None:
                row[name] = float(param)
        self.history.append(row)
        self._append(row)
        return score, result

    def _append(self, row: dict) -> None:
        new_file = self._fieldnames is None
        if new_file:
            self._fieldnames = list(row)
        with open(self.history_path, "a" if not new_file else "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self._fieldnames, extrasaction="ignore")
            if new_file:
                writer.writeheader()
            writer.writerow(row)
