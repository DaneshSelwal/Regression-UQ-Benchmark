# Pile settlement · NB4 Distributional (CARD diffusion)

[← Pile settlement README](../README.md) · [method guide](../../../notebooks/04_probabilistic_distribution_card/README.md) · [report (PDF)](../../../papers/pile_settlement/paper.pdf)

| | |
|---|---|
| **What** | CARD: a denoising diffusion model guided by each tuned regressor (official settings, 1000 samples per test point). |
| **Notebook** | [`Probabilistic__Distribution(CARD).ipynb`](<Probabilistic__Distribution(CARD).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_21_49](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-21-49) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## CARD guided by each tuned model

| Guide model | Coverage | Width / std(y) | CRPS | PIT diagnosis |
|---|--:|--:|--:|---|
| TabNet | 75.8% | 0.13 | 2.08 | Well-calibrated (Uniform PIT) |
| GradientBoosting | 61.3% | 0.11 | 3.88 | Under-dispersed (Over-confident) |
| CatBoost | 69.9% | 0.21 | 3.95 | Under-dispersed (Over-confident) |
| LightGBM | 61.3% | 0.12 | 4.01 | Under-dispersed (Over-confident) |
| GPBoost | 76.9% | 0.24 | 4.10 | Well-calibrated (Uniform PIT) |
| XGBoost | 59.1% | 0.13 | 4.16 | Under-dispersed (Over-confident) |
| PGBM | 48.9% | 0.12 | 4.20 | Under-dispersed (Over-confident) |
| HistGB | 54.8% | 0.16 | 4.23 | Under-dispersed (Over-confident) |
| NGBoost | 73.1% | 0.22 | 5.26 | Under-dispersed (Over-confident) |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/04_probabilistic_distribution_card/README.md).*
