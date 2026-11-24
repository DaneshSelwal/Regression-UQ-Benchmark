# Streamflow · NB3 Distributional (NGBoost, PGBM, IBUG, Treeffuser)

[← Streamflow README](../README.md) · [method guide](../../../notebooks/03_probabilistic_distribution/README.md) · [report (PDF)](../../../papers/streamflow/paper.pdf)

| | |
|---|---|
| **What** | NGBoost and PGBM (native Gaussian), IBUG (XGBoost, LightGBM) and Treeffuser: full predictive distributions, CRPS, PIT and calibration. |
| **Notebook** | [`Probabilistic__Distribution.ipynb`](<Probabilistic__Distribution.ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_01_14](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-14) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Predictive distributions (Gaussian 90% interval μ̂ ± 1.645σ̂)

| Method | Model | Coverage | Width / std(y) | CRPS | PIT diagnosis |
|---|---|--:|--:|--:|---|
| native | NGBoost | 89.6% | 1.28 | 51.20 | Well-calibrated (Uniform PIT) |
| IBUG | LightGBM | 89.6% | 1.70 | 57.87 | – |
| IBUG | XGBoost | 91.7% | 2.10 | 60.06 | – |
| native | PGBM | 16.7% | 0.13 | 70.09 | Under-dispersed (Over-confident) |
| Treeffuser | LightGBM | 100.0% | 5.82 | 96.99 | Over-dispersed (Under-confident) |
| Treeffuser | XGBoost | 99.0% | 7.30 | 119.4 | Over-dispersed (Under-confident) |
| Treeffuser | CatBoost | 100.0% | 14.57 | 217.9 | Over-dispersed (Under-confident) |
| Treeffuser | GPBoost | 100.0% | 19.88 | 294.1 | Over-dispersed (Under-confident) |
| Treeffuser | GradientBoosting | 100.0% | 32.76 | 482.9 | Over-dispersed (Under-confident) |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/03_probabilistic_distribution/README.md).*
