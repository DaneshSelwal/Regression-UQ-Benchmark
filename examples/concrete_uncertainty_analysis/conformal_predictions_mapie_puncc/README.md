# Concrete · NB5 Conformal (MAPIE, PUNCC)

[← Concrete README](../README.md) · [method guide](../../../notebooks/05_conformal_mapie_puncc/README.md) · [report (PDF)](../../../papers/concrete/paper.pdf)

| | |
|---|---|
| **What** | Split conformal, CV+ and J+aB (MAPIE); split conformal, CV+ and CQR (PUNCC), wrapped around every tuned model. |
| **Notebook** | [`Conformal Predictions(MAPIE,PUNCC).ipynb`](<Conformal Predictions(MAPIE,PUNCC).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_20_15](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-20-15) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Coverage / relative width, every method × model

⚠ = below the validity threshold (0.849).

| Method | CatBoost | XGBoost | LightGBM | GPBoost | GradientBoosting | HistGB | NGBoost | PGBM | TabNet |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| J+aB (MAPIE) | 94% / 1.15 | 92% / 1.10 | 92% / 1.02 | 92% / 1.08 | 94% / 1.40 | 86% / 0.99 | 89% / 1.07 | 92% / 1.12 | 94% / 2.08 |
| CV+ (MAPIE) | 89% / 1.03 | 89% / 0.99 | 94% / 1.12 | 95% / 1.16 | 97% / 1.26 | 89% / 1.05 | 92% / 1.08 | 95% / 1.31 | 98% / 2.76 |
| Split CP (PUNCC) | 87% / 0.97 | 92% / 1.29 | 85% / 0.90 | 93% / 1.19 | 86% / 1.38 | 85% / 1.13 | 92% / 1.10 | 94% / 1.25 | 88% / 2.46 |
| CV+ (PUNCC) | 94% / 1.11 | 94% / 1.21 | 95% / 1.13 | 96% / 1.29 | 97% / 1.41 | 98% / 1.30 | 95% / 1.14 | 95% / 1.26 | 100% / 3.43 |
| Split CP (MAPIE) | 99% / 1.75 | 97% / 1.49 | 95% / 1.33 | 95% / 1.44 | 93% / 1.47 | 96% / 1.50 | 96% / 1.42 | 95% / 1.43 | 91% / 2.07 |
| CQR (PUNCC) | 95% / 2.93 | 94% / 3.77 | 97% / 3.79 | 96% / 3.44 | 94% / 2.71 | 97% / 2.29 | 95% / 2.45 | 98% / 3.13 | 97% / 3.72 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/05_conformal_mapie_puncc/README.md).*
