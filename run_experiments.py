"""Compare models and class-imbalance strategies with repeated stratified CV.

    python run_experiments.py --synthetic                 # runs anywhere
    python run_experiments.py --data-dir path/to/study    # needs the study data

--data-dir expects one sub-folder per target ("Complication Binary",
"Relapses Binary"), each holding X_train/X_test/Y_train/Y_test CSVs.
"""
import argparse
from pathlib import Path

from postop.data import load_study_data, make_synthetic
from postop.evaluate import run_cv, summarise

SUBFOLDER = {"complications": "Complication Binary", "relapses": "Relapses Binary"}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", action="store_true", help="use the synthetic stand-in data")
    src.add_argument("--data-dir", type=Path, help="folder with the study data")
    p.add_argument("--targets", nargs="+", default=["complications", "relapses"], choices=list(SUBFOLDER))
    p.add_argument("--repeats", type=int, default=5, help="repeats of 5-fold CV (default 5)")
    p.add_argument("--out", type=Path, default=Path("results"))
    a = p.parse_args()

    tag = "synthetic" if a.synthetic else "study"
    a.out.mkdir(parents=True, exist_ok=True)
    for target in a.targets:
        X, y = make_synthetic(target) if a.synthetic else load_study_data(a.data_dir / SUBFOLDER[target], target)
        print(f"\n{target}: {len(y)} patients, {y.sum()} positive ({y.mean():.1%})")
        folds = run_cv(X, y, n_repeats=a.repeats)
        summary = summarise(folds)
        folds.to_csv(a.out / f"{target}_{tag}_folds.csv", index=False)
        summary.to_csv(a.out / f"{target}_{tag}_summary.csv", index=False)
        cols = ["model", "strategy", "balanced_accuracy", "recall_pos", "precision_pos", "roc_auc", "pr_auc"]
        print(summary[cols].to_string(index=False))


if __name__ == "__main__":
    main()
