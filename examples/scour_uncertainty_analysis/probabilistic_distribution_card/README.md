# Scour · NB4 Distributional (CARD diffusion)

[← Scour README](../README.md) · [method guide](../../../notebooks/04_probabilistic_distribution_card/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | CARD: a denoising diffusion model guided by each tuned regressor (official settings, 1000 samples per test point). |
| **Notebook** | [`Probabilistic__Distribution(CARD).ipynb`](<Probabilistic__Distribution(CARD).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_21_11](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-21-11) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## CARD guided by each tuned model

| Guide model | Coverage | Width / std(y) | CRPS | PIT diagnosis |
|---|--:|--:|--:|---|
| PGBM | 74.4% | 0.65 | 0.186 | Well-calibrated (Uniform PIT) |
| CatBoost | 82.1% | 0.66 | 0.196 | Well-calibrated (Uniform PIT) |
| TabNet | 57.7% | 0.41 | 0.214 | Under-dispersed (Over-confident) |
| GPBoost | 80.8% | 0.81 | 0.216 | Well-calibrated (Uniform PIT) |
| LightGBM | 74.4% | 0.74 | 0.217 | Under-dispersed (Over-confident) |
| NGBoost | 67.9% | 0.59 | 0.219 | Well-calibrated (Uniform PIT) |
| XGBoost | 65.4% | 0.60 | 0.234 | Under-dispersed (Over-confident) |
| GradientBoosting | 75.6% | 0.76 | 0.248 | Well-calibrated (Uniform PIT) |
| HistGB | 65.4% | 0.55 | 0.251 | Under-dispersed (Over-confident) |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/04_probabilistic_distribution_card/README.md).*
