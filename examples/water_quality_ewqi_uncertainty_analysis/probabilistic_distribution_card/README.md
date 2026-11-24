# Water quality (EWQI) · NB4 Distributional (CARD diffusion)

[← Water quality (EWQI) README](../README.md) · [method guide](../../../notebooks/04_probabilistic_distribution_card/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | CARD: a denoising diffusion model guided by each tuned regressor (official settings, 1000 samples per test point). |
| **Notebook** | [`Probabilistic__Distribution(CARD).ipynb`](<Probabilistic__Distribution(CARD).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_01_20](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-01-20) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## CARD guided by each tuned model

| Guide model | Coverage | Width / std(y) | CRPS | PIT diagnosis |
|---|--:|--:|--:|---|
| LightGBM | 40.4% | 0.25 | 10.91 | Under-dispersed (Over-confident) |
| GPBoost | 40.4% | 0.24 | 10.96 | Under-dispersed (Over-confident) |
| CatBoost | 46.8% | 0.24 | 11.03 | Under-dispersed (Over-confident) |
| HistGB | 34.0% | 0.27 | 11.30 | Under-dispersed (Over-confident) |
| PGBM | 34.0% | 0.28 | 12.25 | Under-dispersed (Over-confident) |
| XGBoost | 36.2% | 0.24 | 13.01 | Under-dispersed (Over-confident) |
| NGBoost | 25.5% | 0.24 | 14.55 | Under-dispersed (Over-confident) |
| GradientBoosting | 23.4% | 0.25 | 14.68 | Under-dispersed (Over-confident) |
| TabNet | 27.7% | 0.25 | 17.62 | Under-dispersed (Over-confident) |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/04_probabilistic_distribution_card/README.md).*
