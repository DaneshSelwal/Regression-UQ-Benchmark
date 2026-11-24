# Pile settlement · NB3 Distributional (NGBoost, PGBM, IBUG, Treeffuser)

[← Pile settlement README](../README.md) · [method guide](../../../notebooks/03_probabilistic_distribution/README.md) · [report (PDF)](../../../papers/pile_settlement/paper.pdf)

| | |
|---|---|
| **What** | NGBoost and PGBM (native Gaussian), IBUG (XGBoost, LightGBM) and Treeffuser: full predictive distributions, CRPS, PIT and calibration. |
| **Notebook** | [`Probabilistic__Distribution.ipynb`](<Probabilistic__Distribution.ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_21_39](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-21-39) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Predictive distributions (Gaussian 90% interval μ̂ ± 1.645σ̂)

| Method | Model | Coverage | Width / std(y) | CRPS | PIT diagnosis |
|---|---|--:|--:|--:|---|
| IBUG | LightGBM | 96.8% | 1.22 | 4.07 | – |
| IBUG | XGBoost | 95.2% | 1.31 | 4.35 | – |
| native | PGBM | 8.1% | 0.02 | 4.58 | Under-dispersed (Over-confident) |
| native | NGBoost | 61.3% | 0.19 | 5.71 | Under-dispersed (Over-confident) |
| Treeffuser | LightGBM | 95.2% | 1.77 | 6.36 | Over-dispersed (Under-confident) |
| Treeffuser | GPBoost | 96.2% | 1.97 | 6.76 | Over-dispersed (Under-confident) |
| Treeffuser | CatBoost | 96.8% | 2.70 | 8.08 | Over-dispersed (Under-confident) |
| Treeffuser | GradientBoosting | 98.4% | 3.50 | 8.90 | Over-dispersed (Under-confident) |
| Treeffuser | XGBoost | 98.9% | 4.72 | 11.66 | Over-dispersed (Under-confident) |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/03_probabilistic_distribution/README.md).*
