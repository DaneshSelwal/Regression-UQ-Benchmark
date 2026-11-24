# Sediment load · NB5 Conformal (MAPIE, PUNCC)

[← Sediment load README](../README.md) · [method guide](../../../notebooks/05_conformal_mapie_puncc/README.md) · [report (PDF)](../../../papers/sediment_load/paper.pdf)

| | |
|---|---|
| **What** | Split conformal, CV+ and J+aB (MAPIE); split conformal, CV+ and CQR (PUNCC), wrapped around every tuned model. |
| **Notebook** | [`Conformal Predictions(MAPIE,PUNCC).ipynb`](<Conformal Predictions(MAPIE,PUNCC).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_01_37](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-37) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Coverage / relative width, every method × model

⚠ = below the validity threshold (0.874).

| Method | CatBoost | XGBoost | LightGBM | GPBoost | GradientBoosting | HistGB | NGBoost | PGBM | TabNet |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CQR (PUNCC) | 92% / 0.61 | 90% / 1.03 | **76% / 0.75** ⚠ | **83% / 0.68** ⚠ | 91% / 0.58 | 94% / 0.59 | 94% / 0.55 | 93% / 0.85 | 96% / 0.59 |
| J+aB (MAPIE) | 94% / 0.80 | 93% / 0.97 | 94% / 0.86 | 95% / 0.87 | 96% / 0.60 | 96% / 0.76 | 93% / 0.69 | 95% / 0.75 | 97% / 1.30 |
| Split CP (PUNCC) | 93% / 0.77 | 92% / 1.19 | 94% / 0.91 | 93% / 0.94 | 93% / 0.69 | 94% / 1.08 | 93% / 0.88 | 93% / 0.72 | **81% / 1.20** ⚠ |
| CV+ (PUNCC) | 97% / 1.03 | 95% / 1.15 | 95% / 0.93 | 95% / 1.01 | 97% / 0.67 | 97% / 0.95 | 96% / 0.92 | 97% / 1.05 | 100% / 2.41 |
| Split CP (MAPIE) | 95% / 1.00 | 93% / 1.38 | 94% / 1.10 | 96% / 1.09 | 96% / 0.84 | 96% / 1.52 | 97% / 1.17 | 96% / 1.17 | **84% / 1.10** ⚠ |
| CV+ (MAPIE) | 96% / 1.25 | 98% / 2.21 | 94% / 1.30 | 96% / 1.38 | 98% / 1.08 | 98% / 1.82 | 98% / 1.69 | 98% / 1.66 | 100% / 1.79 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/05_conformal_mapie_puncc/README.md).*
