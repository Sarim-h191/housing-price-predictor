# Verification notes

## Environment

The release was checked in a fresh GitHub checkout with a new Python 3.12 virtual environment on Linux. Installation from requirements.txt succeeded. Tested direct dependencies:

- NumPy 2.5.3
- pandas 2.3.3
- scikit-learn 1.8.0
- XGBoost 3.4.1
- Matplotlib 3.11.2
- JupyterLab 4.6.3

## Automated checks

All three unittest cases passed. The checks cover validation-isolated median imputation, full-data preprocessing refit, candidate selection, consistent prediction preprocessing, preserved IDs, deterministic repeated runs, and invalid inputs.

## Dataset run

The command-line workflow completed on the Kaggle-format CSV files:

- Training data: 1,460 rows and 81 columns.
- Prediction data: 1,459 rows and 80 columns.
- Validation split: 1,168 fitting rows and 292 validation rows.
- Validation RMSE: Linear Regression 39,763.29526578072; Random Forest 28,887.958918999644; XGBoost 28,436.0436066623.
- Selected XGBoost and refitted its pipeline on all 1,460 labeled rows.
- Generated 1,459 finite predictions with the original test IDs in order.

The saved results/metrics.json and results/sample_predictions.csv match this run. Validation scores are not independent test or leaderboard scores.

SHA-256 input fingerprints:

```text
train.csv 1e18addf81e5e4d347cc17ee6075bbe4a42b7fa26b9e5b063e8f692a5f929d41
test.csv 8fdd3d829d4d986b58f845c9553b225e67dd8383624d90fb6ca1d4bed5798c1e
```

## Notebook verification

All notebook code cells executed sequentially on the real CSV files and matched the command-line metrics. A Jupyter kernel session and browser interface were not verified because kernel socket startup was blocked in the validation environment. macOS and Windows were not independently tested.

## Reproduce

Follow README.md to install dependencies, obtain the source CSV files, run tests, and generate outputs. Generated files under outputs/ and local datasets under data/ are excluded from Git.
