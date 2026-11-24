# Concrete · NB2 Quantile regression

[← Concrete README](../README.md) · [method guide](../../../notebooks/02_quantile_regression/README.md) · [report (PDF)](../../../papers/concrete/paper.pdf)

| | |
|---|---|
| **What** | Fits one model per quantile level (13 levels) for each regressor; reports the raw 90% interval, quantile CRPS and quantile crossing. |
| **Notebook** | [`Quantile_Regression.ipynb`](<Quantile_Regression.ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_19_52](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-19-52) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## 90% interval [q̂₀.₀₅, q̂₀.₉₅] per model

| Model | Coverage | Width | Width / std(y) | CRPS | Crossing (before sorting) |
|---|--:|--:|--:|--:|--:|
| LightGBM | 79.0% | 10.35 | 1.03 | 1.77 | 97.9% |
| GPBoost | 73.7% | 8.05 | 0.80 | 1.79 | 98.9% |
| CatBoost | 81.0% | 10.63 | 1.06 | 1.81 | 97.9% |
| HistGB | 77.9% | 11.34 | 1.13 | 1.83 | 100.0% |
| GradientBoosting | 81.0% | 12.54 | 1.25 | 1.84 | 100.0% |
| NGBoost | 32.6% | 1.89 | 0.19 | 1.90 | 0.0% |
| PGBM | 16.8% | 1.05 | 0.10 | 1.97 | 0.0% |
| TabNet | 95.8% | 19.18 | 1.91 | 2.70 | 100.0% |
| XGBoost | 90.5% | 31.67 | 3.16 | 3.00 | 78.9% |

Validity threshold for n = 95: 0.849.

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/02_quantile_regression/README.md).*
