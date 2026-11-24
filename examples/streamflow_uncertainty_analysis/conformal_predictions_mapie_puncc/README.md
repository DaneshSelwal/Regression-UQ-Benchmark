# Streamflow · NB5 Conformal (MAPIE, PUNCC)

[← Streamflow README](../README.md) · [method guide](../../../notebooks/05_conformal_mapie_puncc/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Split conformal, CV+ and J+aB (MAPIE); split conformal, CV+ and CQR (PUNCC), wrapped around every tuned model. |
| **Notebook** | [`Conformal Predictions(MAPIE,PUNCC).ipynb`](<Conformal Predictions(MAPIE,PUNCC).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_01_16](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-16) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Coverage / relative width, every method × model

⚠ = below the validity threshold (0.850).

| Method | CatBoost | XGBoost | LightGBM | GPBoost | GradientBoosting | HistGB | NGBoost | PGBM | TabNet |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| J+aB (MAPIE) | 89% / 1.31 | 88% / 1.48 | 89% / 1.40 | 90% / 1.32 | 88% / 1.31 | 89% / 1.27 | 89% / 1.32 | 90% / 1.43 | 88% / 1.42 |
| CV+ (MAPIE) | 90% / 1.35 | 86% / 1.54 | 91% / 1.66 | 90% / 1.29 | 89% / 1.49 | 94% / 1.47 | 88% / 1.32 | 88% / 1.48 | 96% / 2.00 |
| CV+ (PUNCC) | 89% / 1.40 | 94% / 1.77 | 92% / 1.72 | 92% / 1.40 | 89% / 1.48 | 92% / 1.53 | 90% / 1.42 | 95% / 1.61 | 97% / 2.11 |
| CQR (PUNCC) | 95% / 1.83 | 95% / 2.17 | 88% / 1.38 | 92% / 1.60 | 93% / 1.64 | 90% / 1.47 | 89% / 1.35 | 92% / 1.64 | 97% / 2.93 |
| Split CP (PUNCC) | 97% / 1.88 | 92% / 1.79 | 95% / 1.89 | 97% / 2.48 | 97% / 2.36 | 97% / 2.25 | 95% / 1.84 | 95% / 1.98 | 94% / 2.62 |
| Split CP (MAPIE) | 90% / 1.37 | 97% / 2.40 | 97% / 2.88 | 97% / 2.65 | 97% / 2.45 | 97% / 3.11 | 97% / 2.36 | 95% / 2.22 | 97% / 2.87 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/05_conformal_mapie_puncc/README.md).*
