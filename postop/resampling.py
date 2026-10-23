"""Oversampling of the minority class (label 1), applied to training data only.

`ratio` is the size of the minority class after resampling, relative to the
majority class (the same meaning as imbalanced-learn's float
`sampling_strategy`): 0.5 means one positive for every two negatives,
1.0 means fully balanced.
"""
import numpy as np
from sklearn.neighbors import NearestNeighbors


def _n_new(y, ratio):
    n_pos, n_neg = int((y == 1).sum()), int((y == 0).sum())
    return max(int(round(ratio * n_neg)) - n_pos, 0)


def random_oversample(X, y, ratio=1.0, rng=None):
    """Duplicate randomly chosen minority rows."""
    rng = np.random.default_rng(rng)
    X, y = np.asarray(X, float), np.asarray(y)
    pos = np.flatnonzero(y == 1)
    extra = rng.choice(pos, _n_new(y, ratio), replace=True)
    return np.vstack([X, X[extra]]), np.concatenate([y, y[extra]])


def smote(X, y, ratio=1.0, k=5, rng=None):
    """SMOTE (Chawla et al., 2002): new minority points on the segment between
    a minority sample and one of its k nearest minority neighbours,
    x_new = x_i + lambda * (x_j - x_i), lambda ~ U(0, 1).

    Note: on binary columns the interpolation produces values between 0 and 1.
    SMOTE-NC is the variant designed for mixed numeric/categorical data.
    """
    rng = np.random.default_rng(rng)
    X, y = np.asarray(X, float), np.asarray(y)
    Xp = X[y == 1]
    n_new = _n_new(y, ratio)
    if n_new == 0:
        return X, y
    k = min(k, len(Xp) - 1)
    neigh = NearestNeighbors(n_neighbors=k + 1).fit(Xp).kneighbors(Xp, return_distance=False)[:, 1:]
    i = rng.integers(0, len(Xp), n_new)
    j = neigh[i, rng.integers(0, k, n_new)]
    lam = rng.random((n_new, 1))
    synth = Xp[i] + lam * (Xp[j] - Xp[i])
    return np.vstack([X, synth]), np.concatenate([y, np.ones(n_new, dtype=y.dtype)])
