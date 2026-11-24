# Water quality (WQI) · NB3 Distributional (NGBoost, PGBM, IBUG, Treeffuser)

[← Water quality (WQI) README](../README.md) · [method guide](../../../notebooks/03_probabilistic_distribution/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | NGBoost and PGBM (native Gaussian), IBUG (XGBoost, LightGBM) and Treeffuser: full predictive distributions, CRPS, PIT and calibration. |
| **Notebook** | [`Probabilistic__Distribution.ipynb`](<Probabilistic__Distribution.ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_08_48](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-08-48) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Predictive distributions (Gaussian 90% interval μ̂ ± 1.645σ̂)

| Method | Model | Coverage | Width / std(y) | CRPS | PIT diagnosis |
|---|---|--:|--:|--:|---|
| IBUG | XGBoost | 68.3% | 0.75 | 14.96 | – |
| IBUG | LightGBM | 91.5% | 1.52 | 16.32 | – |
| Treeffuser | GPBoost | 93.9% | 2.24 | 19.26 | Over-dispersed (Under-confident) |
| native | PGBM | 14.6% | 0.19 | 20.52 | Under-dispersed (Over-confident) |
| native | NGBoost | 1.2% | 0.01 | 22.41 | Under-dispersed (Over-confident) |
| Treeffuser | GradientBoosting | 100.0% | 4.22 | 26.67 | Over-dispersed (Under-confident) |
| Treeffuser | XGBoost | 100.0% | 4.91 | 29.30 | Over-dispersed (Under-confident) |
| Treeffuser | LightGBM | 100.0% | 5.91 | 35.77 | Over-dispersed (Under-confident) |
| Treeffuser | CatBoost | 100.0% | 8.49 | 48.26 | Over-dispersed (Under-confident) |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/03_probabilistic_distribution/README.md).*
