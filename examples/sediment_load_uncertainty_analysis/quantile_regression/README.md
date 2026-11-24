# Sediment load · NB2 Quantile regression

[← Sediment load README](../README.md) · [method guide](../../../notebooks/02_quantile_regression/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Fits one model per quantile level (13 levels) for each regressor; reports the raw 90% interval, quantile CRPS and quantile crossing. |
| **Notebook** | [`Quantile_Regression.ipynb`](<Quantile_Regression.ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_01_24](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-24) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## 90% interval [q̂₀.₀₅, q̂₀.₉₅] per model

| Model | Coverage | Width | Width / std(y) | CRPS | Crossing (before sorting) |
|---|--:|--:|--:|--:|--:|
| TabNet | 89.3% | 597.2 | 0.48 | 69.97 | 98.6% |
| CatBoost | 78.6% | 559.6 | 0.45 | 71.48 | 92.9% |
| GradientBoosting | 78.1% | 512.3 | 0.41 | 73.86 | 89.9% |
| GPBoost | 84.1% | 767.9 | 0.62 | 75.59 | 68.5% |
| HistGB | 77.0% | 523.1 | 0.42 | 79.67 | 99.5% |
| XGBoost | 81.1% | 1,176 | 0.95 | 83.76 | 97.3% |
| LightGBM | 83.8% | 1,001 | 0.81 | 88.00 | 23.0% |
| PGBM | 5.8% | 8.32 | 0.01 | 113.4 | 0.0% |
| NGBoost | 84.9% | 277.7 | 0.22 | 128.5 | 0.0% |

Validity threshold for n = 365: 0.874.

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/02_quantile_regression/README.md).*
