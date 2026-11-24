# Pile settlement · NB2 Quantile regression

[← Pile settlement README](../README.md) · [method guide](../../../notebooks/02_quantile_regression/README.md) · [report (PDF)](../../../papers/pile_settlement/paper.pdf)

| | |
|---|---|
| **What** | Fits one model per quantile level (13 levels) for each regressor; reports the raw 90% interval, quantile CRPS and quantile crossing. |
| **Notebook** | [`Quantile_Regression.ipynb`](<Quantile_Regression.ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_21_09](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-21-09) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## 90% interval [q̂₀.₀₅, q̂₀.₉₅] per model

| Model | Coverage | Width | Width / std(y) | CRPS | Crossing (before sorting) |
|---|--:|--:|--:|--:|--:|
| TabNet | 95.2% | 16.53 | 0.61 | 2.37 | 100.0% |
| GradientBoosting | 93.0% | 25.09 | 0.92 | 4.29 | 100.0% |
| HistGB | 84.4% | 17.42 | 0.64 | 4.34 | 99.5% |
| GPBoost | 82.8% | 23.01 | 0.85 | 4.37 | 100.0% |
| LightGBM | 76.3% | 14.01 | 0.52 | 4.57 | 100.0% |
| PGBM | 6.5% | 0.510 | 0.02 | 4.61 | 0.0% |
| CatBoost | 78.5% | 21.52 | 0.79 | 4.71 | 99.5% |
| NGBoost | 58.1% | 5.13 | 0.19 | 5.73 | 0.0% |
| XGBoost | 93.0% | 31.34 | 1.15 | 6.11 | 98.4% |

Validity threshold for n = 186: 0.864.

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/02_quantile_regression/README.md).*
