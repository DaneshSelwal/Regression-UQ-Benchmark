# Noise · NB5 Conformal (MAPIE, PUNCC)

[← Noise README](../README.md) · [method guide](../../../notebooks/05_conformal_mapie_puncc/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Split conformal, CV+ and J+aB (MAPIE); split conformal, CV+ and CQR (PUNCC), wrapped around every tuned model. |
| **Notebook** | [`Conformal Predictions(MAPIE,PUNCC).ipynb`](<Conformal Predictions(MAPIE,PUNCC).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_21_25](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-21-25) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Coverage / relative width, every method × model

⚠ = below the validity threshold (0.836).

| Method | CatBoost | XGBoost | LightGBM | GPBoost | GradientBoosting | HistGB | NGBoost | PGBM | TabNet |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| J+aB (MAPIE) | 88% / 0.90 | 87% / 0.91 | 92% / 0.91 | 90% / 0.89 | 90% / 0.91 | 85% / 0.87 | 87% / 0.87 | 87% / 0.90 | 88% / 1.80 |
| CV+ (PUNCC) | 92% / 1.00 | 93% / 1.03 | 95% / 1.03 | 97% / 0.99 | 92% / 0.99 | 93% / 1.23 | 93% / 1.08 | 92% / 1.02 | 93% / 2.48 |
| CQR (PUNCC) | 92% / 1.21 | 92% / 3.02 | 88% / 1.10 | 87% / 1.25 | 88% / 1.66 | 88% / 0.90 | 87% / 0.77 | 87% / 0.93 | **83% / 1.81** ⚠ |
| Split CP (MAPIE) | 92% / 1.11 | 92% / 1.12 | 93% / 1.03 | 95% / 1.21 | 92% / 1.31 | 95% / 1.25 | 92% / 1.14 | 92% / 1.25 | **82% / 1.40** ⚠ |
| CV+ (MAPIE) | 88% / 0.99 | 93% / 1.17 | 97% / 1.31 | 98% / 1.47 | 95% / 1.26 | 97% / 1.30 | 90% / 1.02 | 90% / 1.11 | 95% / 2.46 |
| Split CP (PUNCC) | 92% / 1.52 | 95% / 1.43 | 95% / 2.06 | 95% / 2.02 | 92% / 1.50 | 87% / 1.07 | 93% / 1.19 | 95% / 1.41 | 92% / 3.22 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/05_conformal_mapie_puncc/README.md).*
