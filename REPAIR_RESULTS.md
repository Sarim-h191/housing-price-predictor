# Repair verification

Verified on Python 3.12 with NumPy 2.3.5, pandas 2.2.3, scikit-learn 1.8.0 and XGBoost 3.4.1.

Three automated tests passed using generated synthetic rows. They check preprocessing isolation from validation, full-data preprocessing refit, selected-model choice, consistent predictions and preserved IDs, deterministic repeated runs, and rejection of invalid input.

The original Kaggle CSV files are absent from this repository. No updated real-data RMSE, model winner, leaderboard result, or generated real-data submission is claimed. Old saved notebook outputs were cleared. Upload the original train.csv and test.csv to compute verified results.

See README.md for explanations and the demo walkthrough.

All notebook code cells also executed in sequence using synthetic CSV inputs and wrote both output files. A normal Jupyter-kernel run could not be verified because this execution environment blocks the kernel's network sockets; direct code-cell execution passed.
