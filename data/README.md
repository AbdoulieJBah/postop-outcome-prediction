# Data

The study data are clinical records of patients operated for haemorrhoids in an Italian
hospital, shared for this thesis through SICCR. They are not public, and no patient-level
data or outputs are included in this repository.

## Running without the data

```bash
python run_experiments.py --synthetic
```

`postop.data.make_synthetic()` generates invented patients with the same columns, value
ranges, cohort sizes and outcome rates as the study data. The feature distributions and the
outcome model are made up, so results on synthetic data say nothing about real patients.

## Running with the data

If you have authorised access, point `--data-dir` at a folder with this layout:

```
<data-dir>/
  Complication Binary/  X_train.csv  X_test.csv  Y_train.csv  Y_test.csv
  Relapses Binary/      X_train.csv  X_test.csv  Y_train.csv  Y_test.csv
```

The train and test files are pooled and re-split by the cross-validation. `.gitignore`
excludes CSV and Excel files under `data/` so they cannot be committed by accident.
