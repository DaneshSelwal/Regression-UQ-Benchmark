# Pile · NB5 Conformal (MAPIE, PUNCC)

[← Pile README](../README.md) · [method guide](../../../notebooks/05_conformal_mapie_puncc/README.md) · [report (PDF)](../../../papers/pile/paper.pdf)

| | |
|---|---|
| **What** | Split conformal, CV+ and J+aB (MAPIE); split conformal, CV+ and CQR (PUNCC), wrapped around every tuned model. |
| **Notebook** | [`Conformal Predictions(MAPIE,PUNCC).ipynb`](<Conformal Predictions(MAPIE,PUNCC).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_13_33](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-13-33) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Coverage / relative width, every method × model

⚠ = below the validity threshold (0.799).

| Method | CatBoost | XGBoost | LightGBM | GPBoost | GradientBoosting | HistGB | NGBoost | PGBM | TabNet |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| J+aB (MAPIE) | 96% / 1.64 | 92% / 1.43 | 92% / 1.48 | 92% / 1.49 | 96% / 1.60 | 92% / 1.52 | 96% / 1.66 | 96% / 1.72 | 96% / 1.78 |
| CV+ (PUNCC) | 100% / 1.83 | 100% / 1.87 | 100% / 1.80 | 100% / 1.91 | 100% / 2.20 | 100% / 2.29 | 96% / 1.78 | 100% / 1.96 | 100% / 3.15 |
| CV+ (MAPIE) | 100% / 1.93 | 92% / 1.83 | 92% / 1.92 | 92% / 1.93 | 96% / 1.95 | 92% / 1.77 | 96% / 1.82 | 96% / 1.98 | 100% / 3.72 |
| Split CP (MAPIE) | 100% / 2.42 | 92% / 1.84 | 92% / 2.06 | 96% / 2.31 | **79% / 1.43** ⚠ | 88% / 1.53 | 92% / 2.09 | 96% / 1.73 | 88% / 3.42 |
| CQR (PUNCC) | 100% / 2.06 | 100% / 1.99 | 100% / 2.51 | 100% / 1.54 | 96% / 2.41 | 92% / 3.09 | 100% / 2.81 | 100% / 2.26 | 88% / 2.63 |
| Split CP (PUNCC) | 96% / 2.43 | 92% / 3.06 | 96% / 2.53 | 92% / 2.15 | 92% / 2.82 | 96% / 2.86 | 96% / 3.17 | 92% / 1.82 | 88% / 3.64 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/05_conformal_mapie_puncc/README.md).*
