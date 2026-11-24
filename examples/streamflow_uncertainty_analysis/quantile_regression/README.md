# Streamflow · NB2 Quantile regression

[← Streamflow README](../README.md) · [method guide](../../../notebooks/02_quantile_regression/README.md) · [report (PDF)](../../../papers/streamflow/paper.pdf)

| | |
|---|---|
| **What** | Fits one model per quantile level (13 levels) for each regressor; reports the raw 90% interval, quantile CRPS and quantile crossing. |
| **Notebook** | [`Quantile_Regression.ipynb`](<Quantile_Regression.ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_01_13](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-13) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## 90% interval [q̂₀.₀₅, q̂₀.₉₅] per model

| Model | Coverage | Width | Width / std(y) | CRPS | Crossing (before sorting) |
|---|--:|--:|--:|--:|--:|
| CatBoost | 83.3% | 233.4 | 1.18 | 48.76 | 52.1% |
| TabNet | 86.5% | 218.4 | 1.11 | 50.01 | 100.0% |
| GPBoost | 91.7% | 298.7 | 1.52 | 50.27 | 49.0% |
| HistGB | 92.7% | 323.7 | 1.64 | 51.61 | 57.3% |
| NGBoost | 91.7% | 255.0 | 1.29 | 52.14 | 0.0% |
| LightGBM | 74.0% | 180.3 | 0.91 | 52.17 | 100.0% |
| XGBoost | 83.3% | 243.5 | 1.24 | 53.18 | 90.6% |
| GradientBoosting | 94.8% | 406.6 | 2.06 | 54.79 | 82.3% |
| PGBM | 13.5% | 21.44 | 0.11 | 71.15 | 0.0% |

Validity threshold for n = 96: 0.850.

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/02_quantile_regression/README.md).*
