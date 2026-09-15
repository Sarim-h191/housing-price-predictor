# Housing Price Predictor

A Python regression project comparing Linear Regression, Random Forest, and XGBoost using five numeric house features: GrLivArea, OverallQual, TotalBsmtSF, GarageCars, and YearBuilt.

## Setup (Python 3.11 or 3.12)

```bash
git clone https://github.com/Sarim-h191/housing-price-predictor.git
cd housing-price-predictor
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

Obtain `train.csv` and `test.csv` from the Kaggle House Prices: Advanced Regression Techniques competition and place them in a local `data/` directory. The original dataset is not bundled. Do not substitute an unrelated dataset when reporting performance.

```bash
python housing.py --data-dir data --output-dir outputs
```

This writes `outputs/metrics.json` and `outputs/submission.csv`. Alternatively, run `jupyter lab housing_price_predictor.ipynb` from the repository root and run all cells. The notebook calls the same workflow and adds model-comparison and actual-versus-predicted plots. Change `DATA_DIR` if using a Kaggle folder.

## Repairs explained

- Split raw rows first. Computing medians before the split lets validation data influence training; the repaired workflow learns medians from fitting rows only.
- Each model uses a pipeline: first fill missing values using medians, then fit the regressor. The same fitted preprocessing is applied to validation and prediction data, replacing the old switch to zero filling.
- All three models use the same five features, split, and RMSE metric. The random seed is 42. Lower RMSE means smaller errors, measured in the same units as SalePrice; it is not a percentage accuracy.
- Choose the lowest validation RMSE, then create a fresh copy of that pipeline and fit it on all labeled rows. Final predictions now benefit from all available training examples.
- Removed hard-coded Kaggle paths, notebook package installation, stale notebook outputs, and unsupported feature-importance and superiority claims.

## Results and limits

No updated real-data RMSE is claimed until the repaired workflow is run on the original CSV files. The previous notebook's saved scores used the old workflow and should not be used as repaired-model results. XGBoost is a candidate, not a guaranteed winner.

The 80/20 holdout is used for model selection. Its RMSE is not an independent final test score. Kaggle's unlabeled test file cannot be used to compute local RMSE. Dollar-scale RMSE here is also different from a log-price competition metric. No hyperparameter search or feature-importance plot is implemented.

## Tests and short demo

```bash
python -m unittest -v
```

Synthetic-data tests verify that validation rows do not influence fitted medians, the final pipeline relearns medians on all labeled rows, prediction preprocessing is consistent, output IDs are preserved, repeated runs agree, and invalid data is rejected. Synthetic scores are not evidence of real housing prediction performance.

Demo: run the command above, open `metrics.json`, explain the three RMSE values and selected model, and show the first rows of `submission.csv`. In the notebook, show the comparison chart. Explain: “I held out rows to compare models, then retrained the selected pipeline on all labeled data to generate predictions.”
