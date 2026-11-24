# Water quality (WQI) · NB1 Hyperparameter tuning (Optuna)

[← Water quality (WQI) README](../README.md) · [method guide](../../../notebooks/01_hyperparameter_tuning/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Tunes ten regressors with Optuna (5-fold CV on the training set only) and saves the settings every later notebook loads. |
| **Notebook** | [`Optuna_autosampler.ipynb`](<Optuna_autosampler.ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_19_53](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-19-53) |
| **Executed output** | [`results/`](results/) (5 files: executed notebook(s) and Excel workbooks) |

## Tuned models

| Model | Best pruner | 5-fold CV RMSE | Test RMSE | Test r |
|---|---|--:|--:|--:|
| XGBoost | WilcoxonPruner | 22.55 | 28.74 | 0.946 |
| PGBM | PercentilePruner | 22.73 | 29.38 | 0.939 |
| LightGBM | HyperbandPruner | 22.80 | 31.90 | 0.925 |
| GPBoost | MedianPruner | 22.97 | 34.91 | 0.911 |
| HistGB | SuccessiveHalvingPruner | 23.54 | 34.61 | 0.915 |
| NGBoost | WilcoxonPruner | 23.61 | 29.19 | 0.946 |
| GradientBoosting | HyperbandPruner | 24.30 | 31.41 | 0.931 |
| CatBoost | SuccessiveHalvingPruner | 26.32 | 27.16 | 0.943 |
| RandomForest | WilcoxonPruner | 29.13 | 31.83 | 0.938 |
| TabNet | HyperbandPruner | 30.88 | 30.87 | 0.930 |

Best model by CV: **XGBoost**. The selected settings are in `results/best_params.json` / `.pkl` and are loaded by notebooks 2–7.

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/01_hyperparameter_tuning/README.md).*
