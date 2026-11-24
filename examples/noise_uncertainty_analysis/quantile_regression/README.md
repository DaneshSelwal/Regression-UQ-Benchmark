# Noise · NB2 Quantile regression

[← Noise README](../README.md) · [method guide](../../../notebooks/02_quantile_regression/README.md) · [report (PDF)](../../../papers/noise/paper.pdf)

| | |
|---|---|
| **What** | Fits one model per quantile level (13 levels) for each regressor; reports the raw 90% interval, quantile CRPS and quantile crossing. |
| **Notebook** | [`Quantile_Regression.ipynb`](<Quantile_Regression.ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_20_53](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-20-53) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## 90% interval [q̂₀.₀₅, q̂₀.₉₅] per model

| Model | Coverage | Width | Width / std(y) | CRPS | Crossing (before sorting) |
|---|--:|--:|--:|--:|--:|
| GPBoost | 71.7% | 9.09 | 0.60 | 1.78 | 100.0% |
| LightGBM | 73.3% | 6.76 | 0.45 | 1.90 | 100.0% |
| CatBoost | 90.0% | 12.11 | 0.81 | 1.93 | 100.0% |
| HistGB | 91.7% | 13.15 | 0.88 | 2.10 | 100.0% |
| NGBoost | 86.7% | 11.47 | 0.76 | 2.25 | 0.0% |
| XGBoost | 98.3% | 25.01 | 1.66 | 2.34 | 100.0% |
| TabNet | 100.0% | 29.57 | 1.97 | 2.40 | 100.0% |
| GradientBoosting | 91.7% | 32.31 | 2.15 | 2.68 | 100.0% |
| PGBM | 6.7% | 0.262 | 0.02 | 2.80 | 0.0% |

Validity threshold for n = 60: 0.836.

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/02_quantile_regression/README.md).*
