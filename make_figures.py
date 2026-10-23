"""Draw the summary figure from results/*_summary.csv.

    python make_figures.py              # study results (results/*_study_summary.csv)
    python make_figures.py --synthetic  # synthetic results
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1", "#fcfcfb"
SERIES = {"balanced_accuracy": ("Balanced accuracy (threshold 0.5)", "#2a78d6"),
          "roc_auc": ("ROC AUC (threshold-free)", "#eb6834")}
LABELS = {"complications": "Complications", "relapses": "Relapses"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--results", type=Path, default=Path("results"))
    p.add_argument("--out", type=Path, default=Path("figures"))
    a = p.parse_args()
    tag = "synthetic" if a.synthetic else "study"

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "text.color": INK,
                         "axes.edgecolor": GRID, "axes.labelcolor": INK_2,
                         "xtick.color": INK_2, "ytick.color": INK_2})
    targets = ["complications", "relapses"]
    data = {t: pd.read_csv(a.results / f"{t}_{tag}_summary.csv") for t in targets}
    models = list(dict.fromkeys(data[targets[0]]["model"]))
    strategies = list(dict.fromkeys(data[targets[0]]["strategy"]))

    fig, axes = plt.subplots(len(targets), len(models), figsize=(13, 6.4), sharex=True, sharey=True,
                             facecolor=SURFACE)
    for r, t in enumerate(targets):
        for c, m in enumerate(models):
            ax = axes[r, c]
            ax.set_facecolor(SURFACE)
            d = data[t][data[t]["model"] == m].set_index("strategy").loc[strategies]
            ys = range(len(strategies))
            ax.axvline(0.5, color=MUTED, lw=1, ls=(0, (3, 3)), zorder=1)
            for k, (metric, (label, color)) in enumerate(SERIES.items()):
                off = -0.14 if k == 0 else 0.14
                y = [i + off for i in ys]
                ax.errorbar(d[metric], y, xerr=d[metric + "_sd"], fmt="o", ms=6, color=color,
                            mec=SURFACE, mew=1.5, elinewidth=2, capsize=0, label=label, zorder=3)
            ax.set_yticks(list(ys), strategies)
            ax.invert_yaxis() if c == 0 and r == 0 else None
            ax.set_xlim(0.42, 0.82)
            ax.grid(axis="x", color=GRID, lw=0.8)
            ax.tick_params(length=0)
            for s in ("top", "right", "left"):
                ax.spines[s].set_visible(False)
            if r == 0:
                ax.set_title(m, fontsize=10, fontweight="bold", color=INK, loc="left")
            if c == 0:
                ax.set_ylabel(LABELS[t], fontsize=10, fontweight="bold", color=INK)

    h, l = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="upper left", ncol=2, frameon=False, bbox_to_anchor=(0.003, 0.955))
    fig.suptitle("Rebalancing moves the decision threshold; it barely changes how well patients are ranked",
                 x=0.01, y=0.99, ha="left", fontsize=12, fontweight="bold", color=INK)
    fig.text(0.01, 0.005, "Mean ± SD over 5×5 repeated stratified cross-validation. "
             "Resampling applied to training folds only. Dashed line: chance (0.5).", fontsize=8, color=INK_2)
    fig.tight_layout(rect=(0, 0.02, 1, 0.925))
    a.out.mkdir(exist_ok=True)
    out = a.out / f"cv_comparison_{tag}.png"
    fig.savefig(out, dpi=170, facecolor=SURFACE)
    print("wrote", out)


if __name__ == "__main__":
    main()
