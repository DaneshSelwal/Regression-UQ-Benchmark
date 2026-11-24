# Pile settlement · NB1 Hyperparameter tuning (Optuna)

[← Pile settlement README](../README.md) · [method guide](../../../notebooks/01_hyperparameter_tuning/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Tunes ten regressors with Optuna (5-fold CV on the training set only) and saves the settings every later notebook loads. |
| **Notebook** | [`Optuna_autosampler.ipynb`](<Optuna_autosampler.ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_17_56](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-17-56) |
| **Executed output** | [`results/`](results/) (5 files: executed notebook(s) and Excel workbooks) |

## Tuned models

| Model | Best pruner | 5-fold CV RMSE | Test RMSE | Test r |
|---|---|--:|--:|--:|
| GPBoost | HyperbandPruner | 9.53 | 11.41 | 0.909 |
| XGBoost | HyperbandPruner | 9.77 | 10.53 | 0.926 |
| GradientBoosting | MedianPruner | 9.89 | 10.33 | 0.931 |
| CatBoost | WilcoxonPruner | 10.03 | 10.62 | 0.924 |
| PGBM | WilcoxonPruner | 10.21 | 10.41 | 0.927 |
| LightGBM | WilcoxonPruner | 10.23 | 10.24 | 0.933 |
| TabNet | PercentilePruner | 10.34 | 16.81 | 0.854 |
| HistGB | WilcoxonPruner | 10.74 | 10.99 | 0.919 |
| NGBoost | MedianPruner | 10.83 | 13.54 | 0.874 |
| RandomForest | WilcoxonPruner | 10.97 | 12.33 | 0.911 |

Best model by CV: **GPBoost**. The selected settings are in `results/best_params.json` / `.pkl` and are loaded by notebooks 2–7.

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/01_hyperparameter_tuning/README.md).*
