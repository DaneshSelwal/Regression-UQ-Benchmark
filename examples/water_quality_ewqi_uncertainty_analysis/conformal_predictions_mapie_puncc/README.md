# Water quality (EWQI) · NB5 Conformal (MAPIE, PUNCC)

[← Water quality (EWQI) README](../README.md) · [method guide](../../../notebooks/05_conformal_mapie_puncc/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Split conformal, CV+ and J+aB (MAPIE); split conformal, CV+ and CQR (PUNCC), wrapped around every tuned model. |
| **Notebook** | [`Conformal Predictions(MAPIE,PUNCC).ipynb`](<Conformal Predictions(MAPIE,PUNCC).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_01_54](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-01-54) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Coverage / relative width, every method × model

⚠ = below the validity threshold (0.828).

| Method | CatBoost | XGBoost | LightGBM | GPBoost | GradientBoosting | HistGB | NGBoost | PGBM | TabNet |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| J+aB (MAPIE) | 91% / 1.38 | 89% / 1.29 | 96% / 1.39 | 98% / 1.41 | 96% / 1.42 | 98% / 1.59 | 100% / 1.71 | 98% / 1.46 | 96% / 1.70 |
| CV+ (PUNCC) | 98% / 1.74 | 96% / 1.78 | 98% / 1.73 | 98% / 1.67 | 98% / 1.80 | 96% / 1.80 | 100% / 2.01 | 100% / 2.01 | 100% / 3.84 |
| CV+ (MAPIE) | 96% / 1.59 | 100% / 1.84 | 96% / 1.96 | 96% / 1.93 | 98% / 1.96 | 100% / 2.08 | 100% / 2.02 | 100% / 1.98 | 98% / 3.27 |
| Split CP (PUNCC) | 94% / 1.30 | 94% / 2.07 | 91% / 1.66 | 98% / 1.70 | 98% / 1.99 | 98% / 2.12 | 100% / 2.32 | 100% / 2.35 | 91% / 2.69 |
| Split CP (MAPIE) | 100% / 2.40 | 100% / 2.41 | 100% / 2.39 | 100% / 2.32 | 100% / 2.30 | 100% / 2.32 | 89% / 1.51 | 100% / 2.48 | 91% / 2.58 |
| CQR (PUNCC) | 98% / 2.53 | 94% / 4.17 | 85% / 1.58 | 96% / 2.57 | 87% / 3.25 | 96% / 2.49 | 98% / 2.26 | 98% / 1.59 | 96% / 3.75 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/05_conformal_mapie_puncc/README.md).*
