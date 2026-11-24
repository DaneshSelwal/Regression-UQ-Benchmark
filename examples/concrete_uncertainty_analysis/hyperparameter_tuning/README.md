# Concrete · NB1 Hyperparameter tuning (Optuna)

[← Concrete README](../README.md) · [method guide](../../../notebooks/01_hyperparameter_tuning/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Tunes ten regressors with Optuna (5-fold CV on the training set only) and saves the settings every later notebook loads. |
| **Notebook** | [`Optuna_autosampler.ipynb`](<Optuna_autosampler.ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_17_57](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-17-57) |
| **Executed output** | [`results/`](results/) (5 files: executed notebook(s) and Excel workbooks) |

## Tuned models

| Model | Best pruner | 5-fold CV RMSE | Test RMSE | Test r |
|---|---|--:|--:|--:|
| CatBoost | NopPruner | 3.66 | 2.99 | 0.954 |
| GradientBoosting | MedianPruner | 3.78 | 3.26 | 0.945 |
| NGBoost | MedianPruner | 3.79 | 3.19 | 0.948 |
| PGBM | WilcoxonPruner | 3.85 | 3.04 | 0.953 |
| XGBoost | WilcoxonPruner | 3.85 | 2.89 | 0.958 |
| LightGBM | HyperbandPruner | 3.87 | 2.89 | 0.958 |
| GPBoost | WilcoxonPruner | 3.97 | 3.18 | 0.948 |
| HistGB | HyperbandPruner | 4.14 | 2.81 | 0.960 |
| RandomForest | WilcoxonPruner | 4.43 | 3.39 | 0.943 |
| TabNet | HyperbandPruner | 7.29 | 5.75 | 0.826 |

Best model by CV: **CatBoost**. The selected settings are in `results/best_params.json` / `.pkl` and are loaded by notebooks 2–7.

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/01_hyperparameter_tuning/README.md).*
