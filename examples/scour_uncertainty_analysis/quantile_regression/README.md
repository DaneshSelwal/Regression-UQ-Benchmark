# Scour · NB2 Quantile regression

[← Scour README](../README.md) · [method guide](../../../notebooks/02_quantile_regression/README.md) · [report (PDF)](../../../papers/scour/paper.pdf)

| | |
|---|---|
| **What** | Fits one model per quantile level (13 levels) for each regressor; reports the raw 90% interval, quantile CRPS and quantile crossing. |
| **Notebook** | [`Quantile_Regression.ipynb`](<Quantile_Regression.ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_21_03](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-21-03) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## 90% interval [q̂₀.₀₅, q̂₀.₉₅] per model

| Model | Coverage | Width | Width / std(y) | CRPS | Crossing (before sorting) |
|---|--:|--:|--:|--:|--:|
| PGBM | 98.7% | 1.91 | 1.81 | 0.188 | 0.0% |
| TabNet | 85.9% | 1.57 | 1.48 | 0.196 | 100.0% |
| CatBoost | 70.5% | 1.02 | 0.96 | 0.200 | 60.3% |
| XGBoost | 76.9% | 0.861 | 0.81 | 0.202 | 98.7% |
| GPBoost | 87.2% | 1.22 | 1.15 | 0.207 | 97.4% |
| GradientBoosting | 84.6% | 1.25 | 1.18 | 0.210 | 98.7% |
| HistGB | 85.9% | 1.21 | 1.14 | 0.211 | 97.4% |
| LightGBM | 76.9% | 1.16 | 1.09 | 0.212 | 100.0% |
| NGBoost | 66.7% | 0.588 | 0.55 | 0.214 | 0.0% |

Validity threshold for n = 78: 0.844.

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/02_quantile_regression/README.md).*
