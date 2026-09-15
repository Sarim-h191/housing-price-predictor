"""Reproducible five-feature housing regression comparison."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from xgboost import XGBRegressor

FEATURES = ['GrLivArea', 'OverallQual', 'TotalBsmtSF', 'GarageCars', 'YearBuilt']


def run(train, test):
    for label, frame, required in (
        ('train', train, FEATURES + ['SalePrice']),
        ('test', test, FEATURES + ['Id']),
    ):
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f'{label} is missing columns: {sorted(missing)}')
        numeric = frame[FEATURES].apply(pd.to_numeric, errors='raise')
        if np.isinf(numeric.to_numpy(dtype=float)).any():
            raise ValueError(f'{label} contains infinite feature values')
    y = pd.to_numeric(train['SalePrice'], errors='raise')
    if not np.isfinite(y).all():
        raise ValueError('SalePrice must contain only finite numbers')
    if test['Id'].isna().any() or test['Id'].duplicated().any():
        raise ValueError('Test Id values must be present and unique')
    X = train[FEATURES].apply(pd.to_numeric, errors='raise')
    # Split raw rows BEFORE learning medians. Validation stays unseen during fitting.
    X_fit, X_valid, y_fit, y_valid = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    estimators = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=1),
        'XGBoost': XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=1),
    }
    fitted, scores, validation_predictions = {}, {}, {}
    for name, estimator in estimators.items():
        pipeline = make_pipeline(SimpleImputer(strategy='median', keep_empty_features=True), estimator)
        pipeline.fit(X_fit, y_fit)
        predictions = pipeline.predict(X_valid)
        scores[name] = float(np.sqrt(mean_squared_error(y_valid, predictions)))
        fitted[name] = pipeline
        validation_predictions[name] = predictions
    selected = min(scores, key=scores.get)
    # Start a fresh copy, then learn medians AND model parameters from ALL labeled rows.
    final_model = clone(fitted[selected]).fit(X, y)
    predictions = final_model.predict(test[FEATURES].apply(pd.to_numeric, errors='raise'))
    if not np.isfinite(predictions).all():
        raise ValueError('Model produced non-finite predictions')
    submission = pd.DataFrame({'Id': test['Id'], 'SalePrice': predictions})
    metrics = {
        'validation_rmse': scores, 'selected_model': selected,
        'fit_rows': len(X_fit), 'validation_rows': len(X_valid),
        'final_training_rows': len(X), 'prediction_rows': len(test),
        'random_state': 42,
        'evaluation': 'One 80/20 validation split; selection scores are not independent test performance.',
    }
    return metrics, submission, final_model, fitted, y_valid, validation_predictions


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=Path('data'))
    parser.add_argument('--output-dir', type=Path, default=Path('outputs'))
    args = parser.parse_args()
    for name in ('train.csv', 'test.csv'):
        if not (args.data_dir / name).is_file():
            parser.error(f'Missing {args.data_dir / name}. Download the Kaggle House Prices files; see README.md.')
    metrics, submission, *_ = run(pd.read_csv(args.data_dir / 'train.csv'), pd.read_csv(args.data_dir / 'test.csv'))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output_dir / 'submission.csv', index=False)
    (args.output_dir / 'metrics.json').write_text(json.dumps(metrics, indent=2) + '\n')
    print(json.dumps(metrics, indent=2))


if __name__ == '__main__':
    main()
