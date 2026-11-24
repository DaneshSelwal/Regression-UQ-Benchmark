# Pile · NB2 Quantile regression

[← Pile README](../README.md) · [method guide](../../../notebooks/02_quantile_regression/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Fits one model per quantile level (13 levels) for each regressor; reports the raw 90% interval, quantile CRPS and quantile crossing. |
| **Notebook** | [`Quantile_Regression.ipynb`](<Quantile_Regression.ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_13_18](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-13-18) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## 90% interval [q̂₀.₀₅, q̂₀.₉₅] per model

| Model | Coverage | Width | Width / std(y) | CRPS | Crossing (before sorting) |
|---|--:|--:|--:|--:|--:|
| CatBoost | 50.0% | 0.726 | 0.55 | 0.255 | 100.0% |
| LightGBM | 58.3% | 1.12 | 0.84 | 0.260 | 100.0% |
| GPBoost | 62.5% | 1.05 | 0.80 | 0.271 | 100.0% |
| HistGB | 91.7% | 2.34 | 1.76 | 0.286 | 100.0% |
| PGBM | 37.5% | 0.295 | 0.22 | 0.330 | 0.0% |
| XGBoost | 58.3% | 0.865 | 0.65 | 0.330 | 100.0% |
| GradientBoosting | 58.3% | 0.980 | 0.74 | 0.333 | 100.0% |
| NGBoost | 0.0% | 0.003 | 0.00 | 0.460 | 0.0% |
| TabNet | 87.5% | 5.91 | 4.46 | 0.627 | 100.0% |

Validity threshold for n = 24: 0.799.

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/02_quantile_regression/README.md).*
