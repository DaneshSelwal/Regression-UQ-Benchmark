# Streamflow · NB4 Distributional (CARD diffusion)

[← Streamflow README](../README.md) · [method guide](../../../notebooks/04_probabilistic_distribution_card/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | CARD: a denoising diffusion model guided by each tuned regressor (official settings, 1000 samples per test point). |
| **Notebook** | [`Probabilistic__Distribution(CARD).ipynb`](<Probabilistic__Distribution(CARD).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_01_15](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-15) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## CARD guided by each tuned model

| Guide model | Coverage | Width / std(y) | CRPS | PIT diagnosis |
|---|--:|--:|--:|---|
| GPBoost | 86.5% | 1.16 | 49.73 | Well-calibrated (Uniform PIT) |
| HistGB | 86.5% | 1.22 | 51.79 | Well-calibrated (Uniform PIT) |
| CatBoost | 83.3% | 1.06 | 53.72 | Well-calibrated (Uniform PIT) |
| TabNet | 80.2% | 1.11 | 53.80 | Well-calibrated (Uniform PIT) |
| NGBoost | 78.1% | 0.99 | 55.60 | Well-calibrated (Uniform PIT) |
| GradientBoosting | 75.0% | 0.90 | 55.87 | Well-calibrated (Uniform PIT) |
| PGBM | 52.1% | 0.58 | 67.27 | Under-dispersed (Over-confident) |
| XGBoost | 58.3% | 0.67 | 67.28 | Under-dispersed (Over-confident) |
| LightGBM | 55.2% | 0.61 | 70.10 | Under-dispersed (Over-confident) |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/04_probabilistic_distribution_card/README.md).*
