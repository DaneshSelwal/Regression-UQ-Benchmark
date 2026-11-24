# Water quality (EWQI) · NB2 Quantile regression

[← Water quality (EWQI) README](../README.md) · [method guide](../../../notebooks/02_quantile_regression/README.md) · [report (PDF)](../../../papers/water_quality_ewqi/paper.pdf)

| | |
|---|---|
| **What** | Fits one model per quantile level (13 levels) for each regressor; reports the raw 90% interval, quantile CRPS and quantile crossing. |
| **Notebook** | [`Quantile_Regression.ipynb`](<Quantile_Regression.ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_21_12](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-21-12) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## 90% interval [q̂₀.₀₅, q̂₀.₉₅] per model

| Model | Coverage | Width | Width / std(y) | CRPS | Crossing (before sorting) |
|---|--:|--:|--:|--:|--:|
| CatBoost | 63.8% | 36.41 | 0.63 | 12.64 | 100.0% |
| GPBoost | 66.0% | 66.42 | 1.16 | 12.95 | 100.0% |
| XGBoost | 93.6% | 142.7 | 2.49 | 14.35 | 100.0% |
| LightGBM | 53.2% | 56.43 | 0.98 | 14.37 | 97.9% |
| HistGB | 83.0% | 97.66 | 1.70 | 14.55 | 100.0% |
| NGBoost | 0.0% | 0.000 | 0.00 | 14.77 | 0.0% |
| PGBM | 4.3% | 1.90 | 0.03 | 15.40 | 0.0% |
| TabNet | 89.4% | 152.5 | 2.65 | 17.07 | 100.0% |
| GradientBoosting | 89.4% | 182.9 | 3.18 | 30.05 | 80.9% |

Validity threshold for n = 47: 0.828.

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/02_quantile_regression/README.md).*
