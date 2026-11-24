# Water quality (WQI) · NB2 Quantile regression

[← Water quality (WQI) README](../README.md) · [method guide](../../../notebooks/02_quantile_regression/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Fits one model per quantile level (13 levels) for each regressor; reports the raw 90% interval, quantile CRPS and quantile crossing. |
| **Notebook** | [`Quantile_Regression.ipynb`](<Quantile_Regression.ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_08_17](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-08-17) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## 90% interval [q̂₀.₀₅, q̂₀.₉₅] per model

| Model | Coverage | Width | Width / std(y) | CRPS | Crossing (before sorting) |
|---|--:|--:|--:|--:|--:|
| TabNet | 92.7% | 104.9 | 1.44 | 13.50 | 100.0% |
| XGBoost | 87.8% | 123.3 | 1.69 | 16.11 | 100.0% |
| CatBoost | 51.2% | 53.63 | 0.74 | 17.13 | 100.0% |
| LightGBM | 85.4% | 98.92 | 1.36 | 17.26 | 100.0% |
| GPBoost | 82.9% | 89.88 | 1.23 | 17.32 | 100.0% |
| GradientBoosting | 79.3% | 133.4 | 1.83 | 17.65 | 100.0% |
| HistGB | 62.2% | 74.33 | 1.02 | 18.81 | 100.0% |
| PGBM | 15.8% | 13.63 | 0.19 | 20.91 | 0.0% |
| NGBoost | 2.4% | 0.721 | 0.01 | 22.48 | 0.0% |

Validity threshold for n = 82: 0.846.

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/02_quantile_regression/README.md).*
