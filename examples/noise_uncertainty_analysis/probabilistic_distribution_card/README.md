# Noise · NB4 Distributional (CARD diffusion)

[← Noise README](../README.md) · [method guide](../../../notebooks/04_probabilistic_distribution_card/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | CARD: a denoising diffusion model guided by each tuned regressor (official settings, 1000 samples per test point). |
| **Notebook** | [`Probabilistic__Distribution(CARD).ipynb`](<Probabilistic__Distribution(CARD).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_01_19](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-01-19) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## CARD guided by each tuned model

| Guide model | Coverage | Width / std(y) | CRPS | PIT diagnosis |
|---|--:|--:|--:|---|
| GradientBoosting | 85.0% | 0.65 | 1.94 | Well-calibrated (Uniform PIT) |
| GPBoost | 30.0% | 0.11 | 2.00 | Under-dispersed (Over-confident) |
| HistGB | 75.0% | 0.52 | 2.03 | Under-dispersed (Over-confident) |
| TabNet | 83.3% | 0.63 | 2.19 | Well-calibrated (Uniform PIT) |
| CatBoost | 73.3% | 0.39 | 2.25 | Well-calibrated (Uniform PIT) |
| XGBoost | 46.7% | 0.21 | 2.30 | Under-dispersed (Over-confident) |
| NGBoost | 70.0% | 0.40 | 2.38 | Well-calibrated (Uniform PIT) |
| LightGBM | 16.7% | 0.08 | 2.46 | Under-dispersed (Over-confident) |
| PGBM | 55.0% | 0.23 | 2.51 | Under-dispersed (Over-confident) |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/04_probabilistic_distribution_card/README.md).*
