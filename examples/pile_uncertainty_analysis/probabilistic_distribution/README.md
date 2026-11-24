# Pile · NB3 Distributional (NGBoost, PGBM, IBUG, Treeffuser)

[← Pile README](../README.md) · [method guide](../../../notebooks/03_probabilistic_distribution/README.md) · [report (PDF)](../../../papers/pile/paper.pdf)

| | |
|---|---|
| **What** | NGBoost and PGBM (native Gaussian), IBUG (XGBoost, LightGBM) and Treeffuser: full predictive distributions, CRPS, PIT and calibration. |
| **Notebook** | [`Probabilistic__Distribution.ipynb`](<Probabilistic__Distribution.ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_13_25](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-13-25) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Predictive distributions (Gaussian 90% interval μ̂ ± 1.645σ̂)

| Method | Model | Coverage | Width / std(y) | CRPS | PIT diagnosis |
|---|---|--:|--:|--:|---|
| native | PGBM | 41.7% | 0.23 | 0.323 | Under-dispersed (Over-confident) |
| IBUG | XGBoost | 100.0% | 2.11 | 0.330 | – |
| IBUG | LightGBM | 100.0% | 1.93 | 0.337 | – |
| native | NGBoost | 0.0% | 0.00 | 0.498 | Under-dispersed (Over-confident) |
| Treeffuser | XGBoost | 100.0% | 5.27 | 0.562 | Over-dispersed (Under-confident) |
| Treeffuser | LightGBM | 100.0% | 5.22 | 0.586 | Over-dispersed (Under-confident) |
| Treeffuser | GPBoost | 100.0% | 6.13 | 0.626 | Over-dispersed (Under-confident) |
| Treeffuser | GradientBoosting | 100.0% | 8.08 | 0.847 | Over-dispersed (Under-confident) |
| Treeffuser | CatBoost | 100.0% | 39.57 | 3.85 | Over-dispersed (Under-confident) |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/03_probabilistic_distribution/README.md).*
