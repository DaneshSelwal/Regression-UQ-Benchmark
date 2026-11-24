# Scour · NB3 Distributional (NGBoost, PGBM, IBUG, Treeffuser)

[← Scour README](../README.md) · [method guide](../../../notebooks/03_probabilistic_distribution/README.md) · [report (PDF)](../../../papers/scour/paper.pdf)

| | |
|---|---|
| **What** | NGBoost and PGBM (native Gaussian), IBUG (XGBoost, LightGBM) and Treeffuser: full predictive distributions, CRPS, PIT and calibration. |
| **Notebook** | [`Probabilistic__Distribution.ipynb`](<Probabilistic__Distribution.ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_01_48](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-48) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Predictive distributions (Gaussian 90% interval μ̂ ± 1.645σ̂)

| Method | Model | Coverage | Width / std(y) | CRPS | PIT diagnosis |
|---|---|--:|--:|--:|---|
| native | PGBM | 100.0% | 1.87 | 0.198 | Over-dispersed (Under-confident) |
| native | NGBoost | 65.4% | 0.55 | 0.212 | Under-dispersed (Over-confident) |
| IBUG | LightGBM | 96.2% | 1.67 | 0.220 | – |
| IBUG | XGBoost | 89.7% | 1.76 | 0.227 | – |
| Treeffuser | LightGBM | 100.0% | 4.48 | 0.394 | Over-dispersed (Under-confident) |
| Treeffuser | GPBoost | 100.0% | 4.75 | 0.419 | Over-dispersed (Under-confident) |
| Treeffuser | XGBoost | 100.0% | 6.81 | 0.562 | Over-dispersed (Under-confident) |
| Treeffuser | GradientBoosting | 100.0% | 7.25 | 0.593 | Over-dispersed (Under-confident) |
| Treeffuser | CatBoost | 100.0% | 8.53 | 0.698 | Over-dispersed (Under-confident) |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/03_probabilistic_distribution/README.md).*
