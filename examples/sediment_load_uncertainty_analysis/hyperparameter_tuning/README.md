# Sediment load · NB1 Hyperparameter tuning (Optuna)

[← Sediment load README](../README.md) · [method guide](../../../notebooks/01_hyperparameter_tuning/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Tunes ten regressors with Optuna (5-fold CV on the training set only) and saves the settings every later notebook loads. |
| **Notebook** | [`Optuna_autosampler.ipynb`](<Optuna_autosampler.ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_17_58](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-17-58) |
| **Executed output** | [`results/`](results/) (5 files: executed notebook(s) and Excel workbooks) |

## Tuned models

| Model | Best pruner | 5-fold CV RMSE | Test RMSE | Test r |
|---|---|--:|--:|--:|
| TabNet | NopPruner | 1,925 | 728.5 | 0.962 |
| GradientBoosting | NopPruner | 2,364 | 328.6 | 0.965 |
| RandomForest | WilcoxonPruner | 2,476 | 543.8 | 0.956 |
| GPBoost | WilcoxonPruner | 2,586 | 338.2 | 0.965 |
| LightGBM | HyperbandPruner | 2,596 | 454.0 | 0.963 |
| PGBM | WilcoxonPruner | 2,620 | 440.9 | 0.938 |
| HistGB | MedianPruner | 2,631 | 420.2 | 0.965 |
| NGBoost | WilcoxonPruner | 2,640 | 673.3 | 0.932 |
| XGBoost | WilcoxonPruner | 2,723 | 696.3 | 0.958 |
| CatBoost | NopPruner | 2,762 | 482.7 | 0.925 |

Best model by CV: **TabNet**. The selected settings are in `results/best_params.json` / `.pkl` and are loaded by notebooks 2–7.

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/01_hyperparameter_tuning/README.md).*
