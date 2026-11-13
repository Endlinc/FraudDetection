# Transaction-only baselines

Train Logistic Regression, LightGBM, and CatBoost on `data/train_transaction.csv`.
`isFraud` is the target. `TransactionID` and `TransactionDT` never enter the
predictor matrix. The identity table is not used.

From the repository root:

```bash
python -m venv .venv
.venv/bin/python -m pip install -r requirements-training.txt
.venv/bin/python src/training/train_baselines.py
```

The default run uses six CPU threads, seed 42, up to 1,000 boosting rounds and
1,000 Logistic Regression iterations. Each invocation creates new directories:

- `experiments/<timestamp>/`: feature manifest, package versions, parameters,
  metrics, learning curves, warnings, and a training log.
- `data/model_runs/<timestamp>/`: fitted models, exact split membership, and
  validation probabilities. This generated directory is ignored by Git by default.
  Model files use Git LFS when published; run `git lfs pull` after cloning to
  download any published models.

To reproduce a named run, install the exact versions in `requirements-training.lock.txt`
and use explicit new output paths (existing directories are not overwritten):

```bash
.venv/bin/python src/training/train_baselines.py \
  --output experiments/reproduction \
  --artifacts data/model_runs/reproduction
```

## Validation

Sort by `TransactionDT`; train on approximately the earliest 80% and validate on
the latest 20%. A timestamp that crosses the boundary belongs entirely to
validation. The timestamp is used for this partition only, then removed from
all model inputs. Report ROC-AUC, average precision, and log loss on the same
validation transactions for every model.

Boosting early stopping uses validation ROC-AUC with a patience of 60 rounds.
These scores support exploratory comparison; validation also selects the number
of boosting rounds, so it is not an untouched final test set. No class weighting,
resampling, feature engineering, or hyperparameter search is applied.

## Preprocessing

String columns, `card1`–`card6`, and `addr1`/`addr2` are treated as categorical
codes. Missing categories use `__MISSING__`; numeric features use float32.

- Logistic Regression: train-only median imputation with missing indicators,
  standard scaling, and sparse one-hot encoding. Categories occurring fewer than
  20 times are grouped; unseen validation categories are ignored. The model uses
  L2 regularization with C=1 and the L-BFGS solver. Convergence warnings are saved.
- LightGBM: native numeric missing-value handling and pandas categorical columns.
  Category vocabularies come only from training; unseen values become missing.
- CatBoost: native numeric missing-value handling and categorical features.

Preprocessing and models are serialized together with the feature schema.

```bash
.venv/bin/python src/training/predict.py \
  --model data/model_runs/<timestamp>/lightgbm.joblib \
  --data data/train_transaction.csv \
  --output data/model_runs/<timestamp>/scores.csv
```

Use the saved validation predictions to reproduce reported metrics; the example
above also scores training rows and is not a new validation evaluation.
Load only model files you trust: joblib uses Python pickle serialization.

## Checks

```bash
.venv/bin/python -m unittest discover -s tests -v
```

The tests cover time-boundary ties, predictor exclusion, duplicate IDs, and invalid
labels. A small end-to-end run and saved-model prediction checks should precede
full-data training when dependencies or training code change.

API references: [scikit-learn LogisticRegression](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html),
[LightGBM classifier](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.LGBMClassifier.html),
[CatBoost fit](https://catboost.ai/docs/en/concepts/python-reference_catboostclassifier_fit).
