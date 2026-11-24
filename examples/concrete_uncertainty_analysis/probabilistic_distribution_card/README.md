# Concrete · NB4 Distributional (CARD diffusion)

[← Concrete README](../README.md) · [method guide](../../../notebooks/04_probabilistic_distribution_card/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | CARD: a denoising diffusion model guided by each tuned regressor (official settings, 1000 samples per test point). |
| **Notebook** | [`Probabilistic__Distribution(CARD).ipynb`](<Probabilistic__Distribution(CARD).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_20_08](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-20-08) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## CARD guided by each tuned model

| Guide model | Coverage | Width / std(y) | CRPS | PIT diagnosis |
|---|--:|--:|--:|---|
| XGBoost | 60.0% | 0.34 | 1.61 | Under-dispersed (Over-confident) |
| LightGBM | 57.9% | 0.32 | 1.62 | Under-dispersed (Over-confident) |
| HistGB | 50.5% | 0.28 | 1.67 | Under-dispersed (Over-confident) |
| CatBoost | 48.4% | 0.26 | 1.76 | Under-dispersed (Over-confident) |
| PGBM | 42.1% | 0.22 | 1.81 | Under-dispersed (Over-confident) |
| NGBoost | 41.1% | 0.24 | 1.85 | Under-dispersed (Over-confident) |
| GradientBoosting | 56.8% | 0.38 | 1.89 | Under-dispersed (Over-confident) |
| GPBoost | 27.4% | 0.16 | 2.01 | Under-dispersed (Over-confident) |
| TabNet | 60.0% | 0.65 | 2.77 | Under-dispersed (Over-confident) |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/04_probabilistic_distribution_card/README.md).*
