# Scour · NB5 Conformal (MAPIE, PUNCC)

[← Scour README](../README.md) · [method guide](../../../notebooks/05_conformal_mapie_puncc/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Split conformal, CV+ and J+aB (MAPIE); split conformal, CV+ and CQR (PUNCC), wrapped around every tuned model. |
| **Notebook** | [`Conformal Predictions(MAPIE,PUNCC).ipynb`](<Conformal Predictions(MAPIE,PUNCC).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_21_20](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-21-20) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Coverage / relative width, every method × model

⚠ = below the validity threshold (0.844).

| Method | CatBoost | XGBoost | LightGBM | GPBoost | GradientBoosting | HistGB | NGBoost | PGBM | TabNet |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| Split CP (MAPIE) | 92% / 1.17 | 90% / 1.34 | 94% / 1.44 | 96% / 1.69 | 94% / 1.80 | 91% / 1.18 | 95% / 1.48 | 96% / 1.49 | 96% / 1.84 |
| J+aB (MAPIE) | 95% / 1.45 | 92% / 1.54 | 95% / 1.67 | 95% / 1.64 | 92% / 1.62 | 95% / 1.68 | 95% / 1.63 | 94% / 1.56 | 96% / 1.65 |
| CV+ (MAPIE) | 95% / 1.51 | 92% / 1.54 | 96% / 1.66 | 95% / 1.61 | 96% / 1.78 | 95% / 1.64 | 95% / 1.78 | 96% / 1.60 | 99% / 2.71 |
| Split CP (PUNCC) | 95% / 1.64 | 94% / 1.54 | 94% / 1.74 | 94% / 1.85 | 96% / 1.78 | 90% / 1.50 | 95% / 1.77 | 95% / 1.69 | 94% / 2.45 |
| CQR (PUNCC) | 96% / 1.60 | 94% / 1.48 | 96% / 1.86 | 95% / 1.54 | 95% / 1.56 | 96% / 2.47 | 96% / 2.36 | 95% / 2.30 | 94% / 1.92 |
| CV+ (PUNCC) | 97% / 1.88 | 97% / 1.95 | 95% / 1.86 | 95% / 1.82 | 95% / 1.91 | 96% / 1.89 | 97% / 1.93 | 97% / 1.86 | 100% / 2.29 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/05_conformal_mapie_puncc/README.md).*
