# Water quality (WQI) · NB4 Distributional (CARD diffusion)

[← Water quality (WQI) README](../README.md) · [method guide](../../../notebooks/04_probabilistic_distribution_card/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | CARD: a denoising diffusion model guided by each tuned regressor (official settings, 1000 samples per test point). |
| **Notebook** | [`Probabilistic__Distribution(CARD).ipynb`](<Probabilistic__Distribution(CARD).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_08_49](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-08-49) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## CARD guided by each tuned model

| Guide model | Coverage | Width / std(y) | CRPS | PIT diagnosis |
|---|--:|--:|--:|---|
| TabNet | 34.1% | 0.21 | 12.78 | Under-dispersed (Over-confident) |
| CatBoost | 25.6% | 0.18 | 19.03 | Under-dispersed (Over-confident) |
| GradientBoosting | 26.8% | 0.23 | 19.66 | Under-dispersed (Over-confident) |
| NGBoost | 25.6% | 0.18 | 20.71 | Under-dispersed (Over-confident) |
| PGBM | 22.0% | 0.18 | 20.73 | Under-dispersed (Over-confident) |
| XGBoost | 20.7% | 0.18 | 20.83 | Under-dispersed (Over-confident) |
| LightGBM | 18.3% | 0.19 | 22.08 | Under-dispersed (Over-confident) |
| HistGB | 22.0% | 0.19 | 23.17 | Under-dispersed (Over-confident) |
| GPBoost | 25.6% | 0.18 | 23.29 | Under-dispersed (Over-confident) |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/04_probabilistic_distribution_card/README.md).*
