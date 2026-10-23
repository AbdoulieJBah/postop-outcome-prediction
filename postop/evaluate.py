"""Repeated stratified cross-validation of model x imbalance-strategy pairs.

Every resampling step happens inside the training folds, so no synthetic or
duplicated patient ever reaches a validation fold.
"""
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, balanced_accuracy_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight

from .resampling import random_oversample, smote

MODELS = {
    "Logistic regression": LogisticRegression(max_iter=5000),
    "SVM (RBF)": SVC(kernel="rbf", C=1.0, gamma="scale"),
    "Random forest": RandomForestClassifier(n_estimators=300, min_samples_leaf=3, random_state=0),
    "Gradient boosting": GradientBoostingClassifier(n_estimators=150, learning_rate=0.05,
                                                    max_depth=3, random_state=0),
}

# name -> (kind, ratio)
STRATEGIES = {
    "No rebalancing": ("none", None),
    "Class weights": ("weights", None),
    "Random oversampling 0.5": ("random", 0.5),
    "Random oversampling 1.0": ("random", 1.0),
    "SMOTE 0.5": ("smote", 0.5),
    "SMOTE 1.0": ("smote", 1.0),
}


def _fit(model, strategy, X, y, seed):
    kind, ratio = STRATEGIES[strategy]
    model = clone(model)
    if kind == "random":
        X, y = random_oversample(X, y, ratio, rng=seed)
    elif kind == "smote":
        X, y = smote(X, y, ratio, rng=seed)
    elif kind == "weights":
        if "class_weight" in model.get_params():
            model.set_params(class_weight="balanced")
        else:  # gradient boosting has no class_weight; weight the samples instead
            return model.fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    return model.fit(X, y)


def _score(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)


def _one_fold(model_name, strategy, X, y, train, test, fold):
    m = _fit(MODELS[model_name], strategy, X[train], y[train], seed=fold)
    pred, score = m.predict(X[test]), _score(m, X[test])
    yt = y[test]
    return {
        "model": model_name, "strategy": strategy, "fold": fold,
        "balanced_accuracy": balanced_accuracy_score(yt, pred),
        "recall_pos": recall_score(yt, pred, zero_division=0),
        "precision_pos": precision_score(yt, pred, zero_division=0),
        "roc_auc": roc_auc_score(yt, score),
        "pr_auc": average_precision_score(yt, score),
    }


def run_cv(X, y, n_splits=5, n_repeats=5, seed=42, n_jobs=-1):
    X, y = np.asarray(X, float), np.asarray(y, int)
    cv = RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)
    folds = list(cv.split(X, y))
    rows = Parallel(n_jobs=n_jobs)(
        delayed(_one_fold)(m, s, X, y, tr, te, f)
        for m in MODELS for s in STRATEGIES for f, (tr, te) in enumerate(folds))
    return pd.DataFrame(rows)


def summarise(folds: pd.DataFrame) -> pd.DataFrame:
    metrics = ["balanced_accuracy", "recall_pos", "precision_pos", "roc_auc", "pr_auc"]
    g = folds.groupby(["model", "strategy"], sort=False)[metrics]
    out = g.mean().round(3).join(g.std().round(3), rsuffix="_sd")
    return out.reset_index()
