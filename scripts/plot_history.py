"""Plot learning curves from the ``history.csv`` files written during training.

e.g.  python scripts/plot_history.py outputs/full_200/*Amazon_Beauty*          # one run directory
      python scripts/plot_history.py outputs/full_200/* --out figures/curves   # every run, one figure each

Writes ``<out>/<run name>.pdf`` and ``.png``: training loss, validation NDCG@10 / Recall@10 with the best epoch
marked, and the learned scalars (``extrap_dt``, ``pop_lambda``) when present.
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402


def plot_run(run_dir: Path, out_dir: Path) -> Path | None:
    history = run_dir / "history.csv"
    if not history.exists():
        return None
    df = pd.read_csv(history)
    scalars = [c for c in ("extrap_dt", "pop_lambda") if c in df]
    fig, axes = plt.subplots(
        1, 3 if scalars else 2, figsize=(4.2 * (3 if scalars else 2), 3.2), constrained_layout=True
    )
    axes[0].plot(df.epoch, df.train_loss)
    axes[0].set(xlabel="epoch", ylabel="training loss (mean per batch)", title="Training loss")
    best = df.loc[df.valid_score.idxmax()]
    for col, label in (("valid_ndcg@10", "NDCG@10"), ("valid_recall@10", "Recall@10")):
        if col in df:
            axes[1].plot(df.epoch, df[col], label=label)
    axes[1].axvline(best.epoch, color="grey", linestyle="--", linewidth=0.8, label=f"best epoch {int(best.epoch)}")
    axes[1].set(xlabel="epoch", ylabel="validation metric", title="Validation")
    axes[1].legend(frameon=False)
    if scalars:
        for col in scalars:
            axes[2].plot(df.epoch, df[col], label=col)
        axes[2].set(xlabel="epoch", title="Learned scalars")
        axes[2].legend(frameon=False)
    out_dir.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(out_dir / f"{run_dir.name}.{ext}", dpi=200)
    plt.close(fig)
    return out_dir / f"{run_dir.name}.pdf"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+", type=Path, help="run directories (each holds a history.csv)")
    parser.add_argument("--out", type=Path, default=Path("figures"))
    args = parser.parse_args()
    for run in args.runs:
        written = plot_run(run, args.out)
        print(f"{run.name}: {written or 'no history.csv, skipped'}")


if __name__ == "__main__":
    main()
