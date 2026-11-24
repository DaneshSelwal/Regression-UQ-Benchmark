# Sediment load · NB4 Distributional (CARD diffusion)

[← Sediment load README](../README.md) · [method guide](../../../notebooks/04_probabilistic_distribution_card/README.md) · [report (PDF)](../../../papers/sediment_load/paper.pdf)

| | |
|---|---|
| **What** | CARD: a denoising diffusion model guided by each tuned regressor (official settings, 1000 samples per test point). |
| **Notebook** | [`Probabilistic__Distribution(CARD).ipynb`](<Probabilistic__Distribution(CARD).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_01_34](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-34) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## CARD guided by each tuned model

| Guide model | Coverage | Width / std(y) | CRPS | PIT diagnosis |
|---|--:|--:|--:|---|
| GPBoost | 94.2% | 0.45 | 64.54 | Over-dispersed (Under-confident) |
| GradientBoosting | 94.2% | 0.40 | 70.60 | Over-dispersed (Under-confident) |
| HistGB | 94.2% | 0.37 | 71.78 | Over-dispersed (Under-confident) |
| CatBoost | 95.6% | 0.43 | 76.39 | Over-dispersed (Under-confident) |
| LightGBM | 94.2% | 0.50 | 76.54 | Over-dispersed (Under-confident) |
| TabNet | 94.8% | 0.51 | 77.00 | Over-dispersed (Under-confident) |
| XGBoost | 95.1% | 0.44 | 78.74 | Over-dispersed (Under-confident) |
| PGBM | 91.8% | 0.28 | 103.9 | Over-dispersed (Under-confident) |
| NGBoost | 89.9% | 0.15 | 141.3 | Over-dispersed (Under-confident) |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/04_probabilistic_distribution_card/README.md).*
