# Water quality (EWQI) · NB3 Distributional (NGBoost, PGBM, IBUG, Treeffuser)

[← Water quality (EWQI) README](../README.md) · [method guide](../../../notebooks/03_probabilistic_distribution/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | NGBoost and PGBM (native Gaussian), IBUG (XGBoost, LightGBM) and Treeffuser: full predictive distributions, CRPS, PIT and calibration. |
| **Notebook** | [`Probabilistic__Distribution.ipynb`](<Probabilistic__Distribution.ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_21_19](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-21-19) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Predictive distributions (Gaussian 90% interval μ̂ ± 1.645σ̂)

| Method | Model | Coverage | Width / std(y) | CRPS | PIT diagnosis |
|---|---|--:|--:|--:|---|
| native | PGBM | 2.1% | 0.03 | 15.34 | Under-dispersed (Over-confident) |
| IBUG | LightGBM | 95.7% | 2.35 | 15.84 | – |
| native | NGBoost | 0.0% | 0.00 | 16.55 | Under-dispersed (Over-confident) |
| IBUG | XGBoost | 95.7% | 2.36 | 16.70 | – |
| Treeffuser | GPBoost | 100.0% | 10.54 | 46.73 | Over-dispersed (Under-confident) |
| Treeffuser | CatBoost | 100.0% | 10.93 | 47.68 | Over-dispersed (Under-confident) |
| Treeffuser | LightGBM | 100.0% | 11.78 | 52.53 | Over-dispersed (Under-confident) |
| Treeffuser | XGBoost | 100.0% | 14.89 | 64.99 | Over-dispersed (Under-confident) |
| Treeffuser | GradientBoosting | 100.0% | 62.82 | 296.2 | Over-dispersed (Under-confident) |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/03_probabilistic_distribution/README.md).*
