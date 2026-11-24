# Pile · NB4 Distributional (CARD diffusion)

[← Pile README](../README.md) · [method guide](../../../notebooks/04_probabilistic_distribution_card/README.md) · [report (PDF)](../../../papers/pile/paper.pdf)

| | |
|---|---|
| **What** | CARD: a denoising diffusion model guided by each tuned regressor (official settings, 1000 samples per test point). |
| **Notebook** | [`Probabilistic__Distribution(CARD).ipynb`](<Probabilistic__Distribution(CARD).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_13_26](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-13-26) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## CARD guided by each tuned model

| Guide model | Coverage | Width / std(y) | CRPS | PIT diagnosis |
|---|--:|--:|--:|---|
| CatBoost | 41.7% | 0.20 | 0.311 | Under-dispersed (Over-confident) |
| PGBM | 50.0% | 0.39 | 0.335 | Under-dispersed (Over-confident) |
| TabNet | 20.8% | 0.20 | 0.400 | Under-dispersed (Over-confident) |
| GPBoost | 16.7% | 0.19 | 0.446 | Under-dispersed (Over-confident) |
| HistGB | 62.5% | 0.72 | 0.450 | Under-dispersed (Over-confident) |
| NGBoost | 16.7% | 0.16 | 0.467 | Under-dispersed (Over-confident) |
| XGBoost | 16.7% | 0.16 | 0.472 | Under-dispersed (Over-confident) |
| LightGBM | 29.2% | 0.17 | 0.472 | Under-dispersed (Over-confident) |
| GradientBoosting | 12.5% | 0.16 | 0.476 | Under-dispersed (Over-confident) |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/04_probabilistic_distribution_card/README.md).*
