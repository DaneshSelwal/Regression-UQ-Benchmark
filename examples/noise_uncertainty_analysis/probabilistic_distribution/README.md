# Noise · NB3 Distributional (NGBoost, PGBM, IBUG, Treeffuser)

[← Noise README](../README.md) · [method guide](../../../notebooks/03_probabilistic_distribution/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | NGBoost and PGBM (native Gaussian), IBUG (XGBoost, LightGBM) and Treeffuser: full predictive distributions, CRPS, PIT and calibration. |
| **Notebook** | [`Probabilistic__Distribution.ipynb`](<Probabilistic__Distribution.ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_21_02](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-21-02) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Predictive distributions (Gaussian 90% interval μ̂ ± 1.645σ̂)

| Method | Model | Coverage | Width / std(y) | CRPS | PIT diagnosis |
|---|---|--:|--:|--:|---|
| IBUG | LightGBM | 98.3% | 1.09 | 2.08 | – |
| native | NGBoost | 86.7% | 0.76 | 2.22 | Well-calibrated (Uniform PIT) |
| Treeffuser | LightGBM | 93.3% | 1.04 | 2.32 | Over-dispersed (Under-confident) |
| IBUG | XGBoost | 90.0% | 0.95 | 2.39 | – |
| native | PGBM | 6.7% | 0.02 | 2.79 | Under-dispersed (Over-confident) |
| Treeffuser | GPBoost | 95.0% | 1.83 | 2.88 | Over-dispersed (Under-confident) |
| Treeffuser | CatBoost | 100.0% | 2.77 | 3.74 | Over-dispersed (Under-confident) |
| Treeffuser | XGBoost | 100.0% | 7.21 | 8.12 | Over-dispersed (Under-confident) |
| Treeffuser | GradientBoosting | 100.0% | 21.17 | 23.04 | Over-dispersed (Under-confident) |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/03_probabilistic_distribution/README.md).*
