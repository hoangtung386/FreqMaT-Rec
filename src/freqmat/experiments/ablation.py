"""Run the FreqMaT ablation grid (variants x seeds) and collect test metrics into ``results/``."""

import statistics
import traceback
from pathlib import Path
from typing import Any

import yaml

from freqmat import paths
from freqmat.training.runner import train_and_eval

DEFAULT_GRID = paths.CONFIGS / "ablation" / "freqmat_grid.yaml"


def load_grid(path: Path = DEFAULT_GRID) -> dict[str, Any]:
    return yaml.safe_load(Path(path).read_text())


def summarise(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Mean / std of every test metric per variant over its successful runs."""
    rows = []
    for variant in dict.fromkeys(r["variant"] for r in runs):
        ok = [r for r in runs if r["variant"] == variant and r["status"] == "ok"]
        row: dict[str, Any] = {
            "variant": variant,
            "n_ok": len(ok),
            "n_failed": sum(r["variant"] == variant for r in runs) - len(ok),
        }
        if ok:
            row["params"] = ok[0]["params"]
            for key in sorted(ok[0]["test"]):
                values = [r["test"][key] for r in ok]
                row[f"{key}_mean"] = round(statistics.fmean(values), 4)
                row[f"{key}_std"] = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
        rows.append(row)
    return rows


def write_tsv(rows: list[dict[str, Any]], path: Path) -> None:
    keys = list(dict.fromkeys(k for row in rows for k in row))
    lines = ["\t".join(keys)] + ["\t".join(str(row.get(k, "")) for k in keys) for row in rows]
    path.write_text("\n".join(lines) + "\n")


def run_grid(
    grid: dict[str, Any], only: list[str] | None = None, seeds: list[int] | None = None, epochs: int | None = None
) -> list[dict[str, Any]]:
    """Execute the grid, writing ``results/ablation_<dataset>[_runs].tsv`` after every run."""
    dataset = grid["dataset"]
    seeds = seeds or grid["seeds"]
    epochs = epochs or grid["epochs"]
    paths.RESULTS.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for name, overrides in grid["variants"].items():
        if only and name not in only:
            continue
        for seed in seeds:
            print(f"== {name} seed={seed}", flush=True)
            record: dict[str, Any] = {"variant": name, "seed": seed}
            try:
                res = train_and_eval(
                    "FreqMaT",
                    dataset,
                    {**overrides, "epochs": epochs, "seed": seed},
                    experiment=f"ablation_{dataset}",
                    show_progress=False,
                )
                record.update(status="ok", params=res.params, test=res.test, run_dir=res.to_json()["run_dir"])
                print("  ", res.test, flush=True)
            except Exception as err:  # keep going: one broken variant must not stop the grid
                traceback.print_exc()
                record.update(status=f"failed: {type(err).__name__}", test={}, params="", run_dir="")
            runs.append(record)
            flat = [
                {k: v for k, v in r.items() if k != "test"} | {f"test_{k}": v for k, v in r["test"].items()}
                for r in runs
            ]
            write_tsv(flat, paths.RESULTS / f"ablation_{dataset}_runs.tsv")
            write_tsv(summarise(runs), paths.RESULTS / f"ablation_{dataset}.tsv")
    return runs
