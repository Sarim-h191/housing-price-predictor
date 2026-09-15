# Repair verification

Verified on Python 3.12 with NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0 and XGBoost 3.4.1.

Three automated tests passed using generated synthetic rows. They check preprocessing isolation from validation, full-data preprocessing refit, selected-model choice, consistent predictions and preserved IDs, deterministic repeated runs, and rejection of invalid input.

The user subsequently supplied the original-format Kaggle CSV files. The repaired command-line workflow completed successfully on these files:

- Train shape: 1,460 rows, 81 columns. Test shape: 1,459 rows, 80 columns.
- Shared split: 1,168 fitting rows, 292 validation rows.
- Validation RMSE: Linear Regression 39,763.29526578072; Random Forest 28,887.958918999644; XGBoost 28,436.0436066623.
- Selected XGBoost, then refitted its complete pipeline on all 1,460 labeled rows.
- Generated 1,459 predictions with preserved unique IDs and no missing predictions.
- These are model-selection validation scores, not independent test or leaderboard scores.

SHA-256 input fingerprints:

```text
train.csv 1e18addf81e5e4d347cc17ee6075bbe4a42b7fa26b9e5b063e8f692a5f929d41
test.csv 8fdd3d829d4d986b58f845c9553b225e67dd8383624d90fb6ca1d4bed5798c1e
```

See README.md for explanations and the demo walkthrough.

All notebook code cells also executed in sequence using synthetic CSV inputs and wrote both output files. A normal Jupyter-kernel run could not be verified because this execution environment blocks the kernel's network sockets; direct code-cell execution passed.


## Release verification

A fresh GitHub checkout and a new Python 3.12 virtual environment successfully installed the requirements, passed all three tests, and ran the command-line workflow on the provided real CSVs. Direct dependencies are now pinned to the tested versions:

- numpy 2.5.3
- pandas 2.3.3
- scikit-learn 1.8.0
- xgboost 3.4.1
- matplotlib 3.11.2
- jupyterlab 4.6.3

Saved metrics, a chart, and 10 sample predictions are included in results/ for inspection without running code. Full predictions are generated locally under outputs/.

All notebook code cells also executed sequentially on the real CSVs in the fresh environment and matched the command-line metrics. This checks cell logic; Jupyter kernel/UI execution remains unverified in this environment.
