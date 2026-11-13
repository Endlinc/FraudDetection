"""Train transaction-only CPU baselines on a shared chronological holdout."""
import argparse
import gc
import hashlib
import importlib.metadata
import json
import logging
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier, early_stopping, log_evaluation, record_evaluation
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, log_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[2]
EXCLUDED = ['TransactionID', 'TransactionDT', 'isFraud']
MODELS = ['logistic_regression', 'lightgbm', 'catboost']


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def prepare_data(frame, validation_fraction):
    """Split by time, keeping equal timestamps together, then remove metadata."""
    if not 0 < validation_fraction < 1:
        raise ValueError('validation_fraction must be between zero and one')
    if not set(EXCLUDED).issubset(frame.columns):
        raise ValueError(f'Input must include {EXCLUDED}')
    if frame.TransactionID.isna().any() or frame.TransactionID.duplicated().any():
        raise ValueError('TransactionID must be populated and unique')
    if frame.TransactionDT.isna().any() or not np.isfinite(frame.TransactionDT).all():
        raise ValueError('TransactionDT must be finite and populated for splitting')
    if not frame.isFraud.isin([0, 1]).all():
        raise ValueError('isFraud must contain only 0 and 1')
    ordered = frame.sort_values('TransactionDT', kind='stable').reset_index(drop=True)
    index = int(len(ordered) * (1 - validation_fraction))
    if not 0 < index < len(ordered):
        raise ValueError('Insufficient rows for the requested split')
    cutoff = ordered.TransactionDT.iloc[index]
    mask = ordered.TransactionDT < cutoff
    if not mask.any() or mask.all():
        raise ValueError('Need distinct time periods on both sides of the split')
    train, valid = ordered.loc[mask].copy(), ordered.loc[~mask].copy()
    if train.isFraud.nunique() != 2 or valid.isFraud.nunique() != 2:
        raise ValueError('Both split partitions must contain both classes')
    features = [c for c in frame.columns if c not in EXCLUDED]
    categorical = [c for c in features if pd.api.types.is_object_dtype(frame[c])
                   or c in [f'card{i}' for i in range(1, 7)] + ['addr1', 'addr2']]
    numeric = [c for c in features if c not in categorical]
    x_train, x_valid = train[features].copy(), valid[features].copy()
    for c in categorical:
        for x in [x_train, x_valid]:
            x[c] = x[c].astype('string').fillna('__MISSING__').astype(str)
    for x in [x_train, x_valid]:
        x[numeric] = x[numeric].astype(np.float32)
        if np.isinf(x[numeric].to_numpy()).any():
            raise ValueError('Numeric features contain infinity')
    assert not set(EXCLUDED) & set(x_train.columns)
    return train, valid, x_train, x_valid, features, categorical, numeric


def train_models(args):
    args.output.mkdir(parents=True, exist_ok=False)
    args.artifacts.mkdir(parents=True, exist_ok=False)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                        handlers=[logging.StreamHandler(sys.stdout),
                                  logging.FileHandler(args.output/'training.txt')])
    logger = logging.getLogger(__name__)
    logger.info('Reading %s', args.data)
    digest = hashlib.sha256()
    with args.data.open('rb') as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            digest.update(block)
    frame = pd.read_csv(args.data, low_memory=False)
    train, valid, x_train, x_valid, features, cats, nums = prepare_data(frame, args.validation_fraction)
    numeric_category_dtypes = {c: 'float' if pd.api.types.is_float_dtype(frame[c]) else 'integer'
                               for c in cats if pd.api.types.is_numeric_dtype(frame[c])}
    del frame
    y_train, y_valid = train.isFraud.to_numpy(), valid.isFraud.to_numpy()
    manifest = {
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'input': str(args.data.resolve()), 'input_sha256': digest.hexdigest(),
        'feature_count': len(features), 'features': features,
        'excluded_predictors': EXCLUDED, 'categorical_features': cats,
        'numeric_category_dtypes': numeric_category_dtypes,
        'seed': args.seed, 'threads': args.threads, 'max_boosting_iterations': args.iterations,
        'max_logistic_iterations': args.lr_iterations,
        'class_weight': None,
        'split': {'method': 'chronological; equal timestamps kept together',
                  'requested_validation_fraction': args.validation_fraction,
                  'train_rows': len(train), 'validation_rows': len(valid),
                  'train_fraud': int(y_train.sum()), 'validation_fraud': int(y_valid.sum()),
                  'train_time_range': [float(train.TransactionDT.min()), float(train.TransactionDT.max())],
                  'validation_time_range': [float(valid.TransactionDT.min()), float(valid.TransactionDT.max())]},
        'versions': {p: importlib.metadata.version(p) for p in ['numpy', 'pandas', 'scikit-learn', 'lightgbm', 'catboost', 'joblib']},
        'python': platform.python_version(),
        'limitations': ['Validation is also used for boosting early stopping; scores are exploratory, not an independent final test.',
                        'No feature engineering, resampling, class weights, or hyperparameter search.'],
        'preprocessing': {'logistic': 'Training medians plus missing indicators; standardize numeric features; one-hot categorical codes (min_frequency=20, unknown values ignored).',
                          'trees': 'Native numeric NaN handling; categorical missing values use __MISSING__. LightGBM categories are learned from training only.'},
    }
    write_json(args.output/'config.json', manifest)
    predictions = valid[['TransactionID', 'TransactionDT', 'isFraud']].copy()
    membership = pd.concat([train[['TransactionID', 'TransactionDT']].assign(partition='train'),
                            valid[['TransactionID', 'TransactionDT']].assign(partition='validation')])
    membership.to_csv(args.artifacts/'split.csv', index=False)
    del membership, train, valid
    gc.collect()
    logger.info('%d features (%d categorical); train=%d, validation=%d', len(features), len(cats), len(y_train), len(y_valid))
    results = []
    for name in args.models:
        logger.info('Starting %s', name)
        started = time.perf_counter()
        history = {}
        train_input, valid_input = x_train, x_valid
        if name == 'logistic_regression':
            preprocessor = ColumnTransformer([
                ('numeric', Pipeline([('imputer', SimpleImputer(strategy='median', add_indicator=True, keep_empty_features=True)),
                                      ('scale', StandardScaler())]), nums),
                ('categorical', OneHotEncoder(handle_unknown='ignore', min_frequency=20, dtype=np.float32), cats)
            ], sparse_threshold=1.0)
            estimator = LogisticRegression(solver='lbfgs', C=1.0, max_iter=args.lr_iterations,
                                           tol=1e-4, random_state=args.seed)
            model = Pipeline([('preprocessor', preprocessor), ('classifier', estimator)])
            params = estimator.get_params()
        elif name == 'lightgbm':
            train_input, valid_input = x_train.copy(), x_valid.copy()
            for c in cats:
                dtype = pd.CategoricalDtype(categories=pd.unique(x_train[c]))
                train_input[c] = train_input[c].astype(dtype)
                valid_input[c] = valid_input[c].astype(dtype)
            model = LGBMClassifier(n_estimators=args.iterations, learning_rate=.05, num_leaves=31,
                                   max_depth=-1, min_child_samples=50, colsample_bytree=.8,
                                   reg_lambda=1.0, random_state=args.seed, n_jobs=args.threads,
                                   deterministic=True, force_col_wise=True, verbosity=-1, metric='auc')
            params = model.get_params()
        else:
            model = CatBoostClassifier(iterations=args.iterations, learning_rate=.05, depth=6,
                                       loss_function='Logloss', eval_metric='AUC', random_seed=args.seed,
                                       thread_count=args.threads, task_type='CPU', allow_writing_files=False)
            params = model.get_params()
        write_json(args.output/f'{name}_parameters.json', params)
        with warnings.catch_warnings(record=True) as caught, threadpool_limits(limits=args.threads):
            warnings.simplefilter('always', ConvergenceWarning)
            if name == 'logistic_regression':
                model.fit(train_input, y_train)
            elif name == 'lightgbm':
                model.fit(train_input, y_train, eval_set=[(valid_input, y_valid)], eval_metric='auc',
                          categorical_feature=cats,
                          callbacks=[early_stopping(60, first_metric_only=True), log_evaluation(100), record_evaluation(history)])
            else:
                model.fit(train_input, y_train, cat_features=cats, eval_set=(valid_input, y_valid),
                          early_stopping_rounds=60, use_best_model=True, verbose=100)
                history = model.get_evals_result()
            scores = model.predict_proba(valid_input)[:, 1]
        if not np.isfinite(scores).all() or np.any((scores < 0) | (scores > 1)):
            raise ValueError('Invalid predicted probabilities')
        notes = sorted(set(str(w.message) for w in caught))
        for note in notes:
            logger.warning('%s: %s', name, note)
        iterations = (int(model.named_steps['classifier'].n_iter_[0]) if name == 'logistic_regression'
                      else int(model.best_iteration_) if name == 'lightgbm' else int(model.tree_count_))
        result = {'model': name, 'roc_auc': float(roc_auc_score(y_valid, scores)),
                  'average_precision': float(average_precision_score(y_valid, scores)),
                  'log_loss': float(log_loss(y_valid, scores)),
                  'fit_and_predict_seconds': time.perf_counter()-started,
                  'iterations': iterations, 'warnings': notes}
        predictions[name] = scores
        results.append(result)
        joblib.dump({'model': model, 'features': features, 'categorical_features': cats,
                     'numeric_category_dtypes': numeric_category_dtypes,
                     'category_levels': {c: train_input[c].cat.categories.tolist() for c in cats} if name == 'lightgbm' else None,
                     'excluded_predictors': EXCLUDED}, args.artifacts/f'{name}.joblib', compress=3)
        write_json(args.output/'metrics.json', results)
        pd.DataFrame([{k:v for k,v in r.items() if k!='warnings'} for r in results]).to_csv(args.output/'metrics.csv', index=False)
        write_json(args.output/f'{name}_history.json', history)
        predictions.to_csv(args.artifacts/'validation_predictions.csv', index=False)
        logger.info('%s ROC-AUC=%.6f AP=%.6f time=%.1fs', name, result['roc_auc'], result['average_precision'], result['fit_and_predict_seconds'])
        del model, train_input, valid_input
        gc.collect()
    logger.info('Finished. Results: %s; models/predictions: %s', args.output, args.artifacts)


def main():
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=ROOT/'data/train_transaction.csv')
    parser.add_argument('--output', type=Path, default=ROOT/'experiments'/stamp)
    parser.add_argument('--artifacts', type=Path, default=ROOT/'data/model_runs'/stamp)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=MODELS)
    parser.add_argument('--validation-fraction', type=float, default=.2)
    parser.add_argument('--iterations', type=int, default=1000)
    parser.add_argument('--lr-iterations', type=int, default=1000)
    parser.add_argument('--threads', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    if min(args.iterations,args.lr_iterations,args.threads)<=0:
        parser.error('Iteration and thread counts must be positive')
    train_models(args)

if __name__ == '__main__':
    main()
