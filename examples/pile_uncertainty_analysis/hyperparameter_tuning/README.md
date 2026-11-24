# Pile · NB1 Hyperparameter tuning (Optuna)

[← Pile README](../README.md) · [method guide](../../../notebooks/01_hyperparameter_tuning/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Tunes ten regressors with Optuna (5-fold CV on the training set only) and saves the settings every later notebook loads. |
| **Notebook** | [`Optuna_autosampler.ipynb`](<Optuna_autosampler.ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_01_58](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-01-58) |
| **Executed output** | [`results/`](results/) (5 files: executed notebook(s) and Excel workbooks) |

## Tuned models

| Model | Best pruner | 5-fold CV RMSE | Test RMSE | Test r |
|---|---|--:|--:|--:|
| LightGBM | HyperbandPruner | 0.567 | 0.761 | 0.901 |
| XGBoost | NopPruner | 0.567 | 0.715 | 0.921 |
| GPBoost | HyperbandPruner | 0.582 | 0.674 | 0.908 |
| CatBoost | SuccessiveHalvingPruner | 0.604 | 0.506 | 0.938 |
| GradientBoosting | PercentilePruner | 0.609 | 0.682 | 0.925 |
| NGBoost | NopPruner | 0.611 | 0.712 | 0.923 |
| PGBM | HyperbandPruner | 0.623 | 0.492 | 0.951 |
| HistGB | HyperbandPruner | 0.636 | 0.692 | 0.915 |
| RandomForest | SuccessiveHalvingPruner | 0.643 | 0.439 | 0.955 |
| TabNet | MedianPruner | 0.817 | 1.14 | 0.836 |

Best model by CV: **LightGBM**. The selected settings are in `results/best_params.json` / `.pkl` and are loaded by notebooks 2–7.

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/01_hyperparameter_tuning/README.md).*
