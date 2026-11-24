# Streamflow · NB1 Hyperparameter tuning (Optuna)

[← Streamflow README](../README.md) · [method guide](../../../notebooks/01_hyperparameter_tuning/README.md) · [report (PDF)](../../../papers/streamflow/paper.pdf)

| | |
|---|---|
| **What** | Tunes ten regressors with Optuna (5-fold CV on the training set only) and saves the settings every later notebook loads. |
| **Notebook** | [`Optuna_autosampler.ipynb`](<Optuna_autosampler.ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_15_16](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-15-16) |
| **Executed output** | [`results/`](results/) (5 files: executed notebook(s) and Excel workbooks) |

## Tuned models

| Model | Best pruner | 5-fold CV RMSE | Test RMSE | Test r |
|---|---|--:|--:|--:|
| GradientBoosting | HyperbandPruner | 102.4 | 115.4 | 0.820 |
| NGBoost | HyperbandPruner | 104.6 | 112.9 | 0.819 |
| RandomForest | SuccessiveHalvingPruner | 105.9 | 118.6 | 0.799 |
| TabNet | NopPruner | 106.9 | 116.8 | 0.867 |
| CatBoost | WilcoxonPruner | 109.1 | 114.2 | 0.815 |
| PGBM | SuccessiveHalvingPruner | 109.5 | 112.9 | 0.823 |
| GPBoost | ThresholdPruner | 110.0 | 109.2 | 0.833 |
| LightGBM | SuccessiveHalvingPruner | 110.1 | 125.3 | 0.781 |
| HistGB | HyperbandPruner | 115.2 | 117.8 | 0.802 |
| XGBoost | ThresholdPruner | 120.7 | 118.0 | 0.802 |

Best model by CV: **GradientBoosting**. The selected settings are in `results/best_params.json` / `.pkl` and are loaded by notebooks 2–7.

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/01_hyperparameter_tuning/README.md).*
