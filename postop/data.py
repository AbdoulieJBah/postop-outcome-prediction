"""Loading the study data (when you have access to it) or a synthetic stand-in.

The clinical dataset is not part of this repository. See data/README.md.
"""
from pathlib import Path

import numpy as np
import pandas as pd

# Original (Italian) column names -> English names used in the code.
# Whitespace inside the original names is normalised before matching.
COLUMNS = {
    "ETA'": "age",
    "SESSO M=0; f=1": "sex_female",
    "PROLASSO_EMORR Sì=1 NO=0": "haemorrhoid_prolapse",
    "tipo_interv": "intervention_type",
    "Ligasure Sì=1 NO=0": "ligasure",
    "Focus Sì=1 NO=0": "focus",
    "Diatermica Sì=1 NO=0": "diathermy",
    "anestesia": "anaesthesia",
    "SANGUINAMENTO pre SI/NO": "bleeding_pre",
    "RIDUZIONEMANUALE pre SI/NO": "manual_reduction_pre",
    "DOLORE-FASTIDIOpre SI/NO": "pain_discomfort_pre",
    "scoretotale_pre": "preop_total_score",
}

TARGETS = {
    # target name -> (label column in the study files, features used)
    "complications": ("Complicanze_Binary", list(COLUMNS.values())),
    # The relapse subset has no Ligasure column.
    "relapses": ("Recidive_Binary", [c for c in COLUMNS.values() if c != "ligasure"]),
}


def _rename(df: pd.DataFrame) -> pd.DataFrame:
    df = df.loc[:, [c for c in df.columns if not str(c).startswith("Unnamed")]]
    return df.rename(columns=lambda c: COLUMNS.get(" ".join(str(c).split()), c))


def load_study_data(folder: str | Path, target: str):
    """Load the pre-processed study split (X_train/X_test/Y_train/Y_test CSVs)
    from `folder` and pool it, so that cross-validation can re-split it.

    Features in these files were already cleaned (missing rows dropped, age
    outliers removed by the IQR rule) and min-max scaled to [0, 1].
    """
    folder = Path(folder)
    label, features = TARGETS[target]
    X = pd.concat([_rename(pd.read_csv(folder / f"X_{s}.csv")) for s in ("train", "test")])
    y = pd.concat([pd.read_csv(folder / f"Y_{s}.csv")[label] for s in ("train", "test")])
    return X[features].reset_index(drop=True), y.astype(int).reset_index(drop=True)


def make_synthetic(target: str, n: int | None = None, seed: int = 0):
    """A synthetic dataset with the same columns, value ranges and class
    balance as the study data, so the code runs end to end without patient data.

    Nothing here is derived from patient records: the marginal distributions
    and the outcome model are invented. Only the cohort sizes and outcome rates
    (published in the thesis) are reused. Results on this data mean nothing
    clinically.
    """
    rng = np.random.default_rng(seed)
    label, features = TARGETS[target]
    n = n or (1726 if target == "complications" else 1250)
    rate = 275 / 1726 if target == "complications" else 116 / 1250

    age = np.clip(rng.normal(52, 14, n), 18, 90)
    score = rng.integers(0, 23, n)
    df = pd.DataFrame({
        "age": (age - age.min()) / (age.max() - age.min()),
        "sex_female": rng.binomial(1, 0.45, n),
        "haemorrhoid_prolapse": rng.binomial(1, 0.6, n),
        "intervention_type": rng.binomial(1, 0.5, n),
        "ligasure": rng.binomial(1, 0.3, n),
        "focus": rng.binomial(1, 0.2, n),
        "diathermy": rng.binomial(1, 0.4, n),
        "anaesthesia": rng.choice([0.0, 0.5, 1.0], n, p=[0.3, 0.4, 0.3]),
        "bleeding_pre": rng.binomial(1, 0.7, n),
        "manual_reduction_pre": rng.binomial(1, 0.3, n),
        "pain_discomfort_pre": rng.binomial(1, 0.6, n),
        "preop_total_score": score / 22,
    })[features].astype(float)

    # Invented, weak signal, so the task is about as hard as the real one.
    logit = (1.2 * df["preop_total_score"] + 0.8 * df["haemorrhoid_prolapse"]
             - 0.6 * df["age"] + 0.5 * df["intervention_type"] + rng.normal(0, 1.2, n))
    y = (logit > np.quantile(logit, 1 - rate)).astype(int)
    return df, pd.Series(y, name=label)
