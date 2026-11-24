# Scour · NB1 Hyperparameter tuning (Optuna)

[← Scour README](../README.md) · [method guide](../../../notebooks/01_hyperparameter_tuning/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Tunes ten regressors with Optuna (5-fold CV on the training set only) and saves the settings every later notebook loads. |
| **Notebook** | [`Optuna_autosampler.ipynb`](<Optuna_autosampler.ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_14_35](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-14-35) |
| **Executed output** | [`results/`](results/) (5 files: executed notebook(s) and Excel workbooks) |

## Tuned models

| Model | Best pruner | 5-fold CV RMSE | Test RMSE | Test r |
|---|---|--:|--:|--:|
| XGBoost | MedianPruner | 0.544 | 0.404 | 0.929 |
| CatBoost | NopPruner | 0.550 | 0.378 | 0.934 |
| GradientBoosting | WilcoxonPruner | 0.551 | 0.425 | 0.917 |
| PGBM | MedianPruner | 0.561 | 0.311 | 0.956 |
| RandomForest | WilcoxonPruner | 0.563 | 0.457 | 0.903 |
| GPBoost | ThresholdPruner | 0.566 | 0.413 | 0.924 |
| LightGBM | NopPruner | 0.568 | 0.401 | 0.927 |
| HistGB | HyperbandPruner | 0.568 | 0.488 | 0.890 |
| NGBoost | MedianPruner | 0.579 | 0.402 | 0.925 |
| TabNet | WilcoxonPruner | 0.606 | 0.438 | 0.937 |

Best model by CV: **XGBoost**. The selected settings are in `results/best_params.json` / `.pkl` and are loaded by notebooks 2–7.

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/01_hyperparameter_tuning/README.md).*
