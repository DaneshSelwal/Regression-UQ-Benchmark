# Sediment load · NB3 Distributional (NGBoost, PGBM, IBUG, Treeffuser)

[← Sediment load README](../README.md) · [method guide](../../../notebooks/03_probabilistic_distribution/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | NGBoost and PGBM (native Gaussian), IBUG (XGBoost, LightGBM) and Treeffuser: full predictive distributions, CRPS, PIT and calibration. |
| **Notebook** | [`Probabilistic__Distribution.ipynb`](<Probabilistic__Distribution.ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_01_25](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-25) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Predictive distributions (Gaussian 90% interval μ̂ ± 1.645σ̂)

| Method | Model | Coverage | Width / std(y) | CRPS | PIT diagnosis |
|---|---|--:|--:|--:|---|
| native | PGBM | 6.6% | 0.01 | 113.2 | Under-dispersed (Over-confident) |
| native | NGBoost | 85.8% | 0.24 | 166.7 | Under-dispersed (Over-confident) |
| IBUG | LightGBM | 98.6% | 1.52 | 178.8 | – |
| IBUG | XGBoost | 98.4% | 1.98 | 260.4 | – |
| Treeffuser | CatBoost | 99.7% | 4.55 | 433.3 | Over-dispersed (Under-confident) |
| Treeffuser | GradientBoosting | 100.0% | 4.58 | 441.2 | Over-dispersed (Under-confident) |
| Treeffuser | XGBoost | 100.0% | 5.45 | 553.2 | Over-dispersed (Under-confident) |
| Treeffuser | GPBoost | 100.0% | 18.49 | 1,687 | Over-dispersed (Under-confident) |
| Treeffuser | LightGBM | 100.0% | 40.65 | 3,660 | Over-dispersed (Under-confident) |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/03_probabilistic_distribution/README.md).*
