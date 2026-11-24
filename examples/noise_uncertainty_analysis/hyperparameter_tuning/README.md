# Noise · NB1 Hyperparameter tuning (Optuna)

[← Noise README](../README.md) · [method guide](../../../notebooks/01_hyperparameter_tuning/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Tunes ten regressors with Optuna (5-fold CV on the training set only) and saves the settings every later notebook loads. |
| **Notebook** | [`Optuna_autosampler.ipynb`](<Optuna_autosampler.ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_17_59](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-17-59) |
| **Executed output** | [`results/`](results/) (5 files: executed notebook(s) and Excel workbooks) |

## Tuned models

| Model | Best pruner | 5-fold CV RMSE | Test RMSE | Test r |
|---|---|--:|--:|--:|
| LightGBM | HyperbandPruner | 3.97 | 4.06 | 0.962 |
| XGBoost | MedianPruner | 4.15 | 3.95 | 0.965 |
| GradientBoosting | HyperbandPruner | 4.15 | 4.34 | 0.962 |
| CatBoost | WilcoxonPruner | 4.17 | 4.26 | 0.959 |
| RandomForest | WilcoxonPruner | 4.24 | 3.92 | 0.966 |
| PGBM | SuccessiveHalvingPruner | 4.29 | 4.44 | 0.958 |
| NGBoost | ThresholdPruner | 4.35 | 4.32 | 0.958 |
| GPBoost | SuccessiveHalvingPruner | 4.46 | 3.39 | 0.974 |
| HistGB | NopPruner | 4.57 | 3.94 | 0.968 |
| TabNet | ThresholdPruner | 7.44 | 9.12 | 0.933 |

Best model by CV: **LightGBM**. The selected settings are in `results/best_params.json` / `.pkl` and are loaded by notebooks 2–7.

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/01_hyperparameter_tuning/README.md).*
