# Water quality (EWQI) · NB1 Hyperparameter tuning (Optuna)

[← Water quality (EWQI) README](../README.md) · [method guide](../../../notebooks/01_hyperparameter_tuning/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Tunes ten regressors with Optuna (5-fold CV on the training set only) and saves the settings every later notebook loads. |
| **Notebook** | [`Optuna_autosampler.ipynb`](<Optuna_autosampler.ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_14_57](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-14-57) |
| **Executed output** | [`results/`](results/) (5 files: executed notebook(s) and Excel workbooks) |

## Tuned models

| Model | Best pruner | 5-fold CV RMSE | Test RMSE | Test r |
|---|---|--:|--:|--:|
| XGBoost | HyperbandPruner | 22.43 | 20.34 | 0.953 |
| LightGBM | PercentilePruner | 22.88 | 18.14 | 0.956 |
| GPBoost | SuccessiveHalvingPruner | 22.90 | 17.95 | 0.957 |
| GradientBoosting | SuccessiveHalvingPruner | 23.22 | 22.27 | 0.943 |
| NGBoost | MedianPruner | 23.38 | 20.81 | 0.951 |
| CatBoost | NopPruner | 24.40 | 19.41 | 0.950 |
| PGBM | ThresholdPruner | 24.51 | 19.53 | 0.947 |
| HistGB | WilcoxonPruner | 25.50 | 18.72 | 0.954 |
| TabNet | WilcoxonPruner | 28.13 | 46.99 | 0.876 |
| RandomForest | SuccessiveHalvingPruner | 28.14 | 22.67 | 0.956 |

Best model by CV: **XGBoost**. The selected settings are in `results/best_params.json` / `.pkl` and are loaded by notebooks 2–7.

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/01_hyperparameter_tuning/README.md).*
