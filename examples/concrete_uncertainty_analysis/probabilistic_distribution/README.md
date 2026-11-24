# Concrete · NB3 Distributional (NGBoost, PGBM, IBUG, Treeffuser)

[← Concrete README](../README.md) · [method guide](../../../notebooks/03_probabilistic_distribution/README.md) · [report (PDF)](../../../papers/concrete/paper.pdf)

| | |
|---|---|
| **What** | NGBoost and PGBM (native Gaussian), IBUG (XGBoost, LightGBM) and Treeffuser: full predictive distributions, CRPS, PIT and calibration. |
| **Notebook** | [`Probabilistic__Distribution.ipynb`](<Probabilistic__Distribution.ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_19_59](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-19-59) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Predictive distributions (Gaussian 90% interval μ̂ ± 1.645σ̂)

| Method | Model | Coverage | Width / std(y) | CRPS | PIT diagnosis |
|---|---|--:|--:|--:|---|
| IBUG | LightGBM | 86.3% | 1.03 | 1.80 | – |
| IBUG | XGBoost | 95.8% | 1.63 | 1.84 | – |
| native | NGBoost | 33.7% | 0.19 | 1.87 | Under-dispersed (Over-confident) |
| native | PGBM | 21.1% | 0.13 | 1.92 | Under-dispersed (Over-confident) |
| Treeffuser | GPBoost | 98.9% | 2.13 | 2.46 | Over-dispersed (Under-confident) |
| Treeffuser | GradientBoosting | 93.7% | 1.81 | 2.48 | Well-calibrated (Uniform PIT) |
| Treeffuser | LightGBM | 100.0% | 3.33 | 3.27 | Over-dispersed (Under-confident) |
| Treeffuser | XGBoost | 100.0% | 7.36 | 6.15 | Over-dispersed (Under-confident) |
| Treeffuser | CatBoost | 100.0% | 31.90 | 23.33 | Over-dispersed (Under-confident) |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/03_probabilistic_distribution/README.md).*
