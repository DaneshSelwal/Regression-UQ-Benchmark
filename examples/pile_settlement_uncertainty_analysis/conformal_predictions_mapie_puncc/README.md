# Pile settlement · NB5 Conformal (MAPIE, PUNCC)

[← Pile settlement README](../README.md) · [method guide](../../../notebooks/05_conformal_mapie_puncc/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Split conformal, CV+ and J+aB (MAPIE); split conformal, CV+ and CQR (PUNCC), wrapped around every tuned model. |
| **Notebook** | [`Conformal Predictions(MAPIE,PUNCC).ipynb`](<Conformal Predictions(MAPIE,PUNCC).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_22_02](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-22-02) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Coverage / relative width, every method × model

⚠ = below the validity threshold (0.864).

| Method | CatBoost | XGBoost | LightGBM | GPBoost | GradientBoosting | HistGB | NGBoost | PGBM | TabNet |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CV+ (MAPIE) | 88% / 0.77 | 88% / 0.75 | 88% / 0.74 | 88% / 0.80 | 90% / 0.81 | 91% / 0.96 | 87% / 0.99 | 90% / 0.83 | 96% / 1.06 |
| J+aB (MAPIE) | **85% / 0.83** ⚠ | **85% / 0.81** ⚠ | **86% / 0.91** ⚠ | **84% / 0.79** ⚠ | **85% / 0.85** ⚠ | 88% / 0.96 | **82% / 0.94** ⚠ | 88% / 0.89 | 90% / 0.75 |
| CV+ (PUNCC) | 88% / 0.78 | 90% / 0.86 | 89% / 0.84 | 89% / 1.00 | 92% / 0.96 | 91% / 0.97 | 87% / 1.00 | 90% / 0.92 | 96% / 0.94 |
| CQR (PUNCC) | 90% / 1.29 | 95% / 2.62 | 87% / 0.94 | 90% / 0.87 | 87% / 0.98 | **85% / 1.05** ⚠ | 89% / 1.58 | 89% / 1.11 | **81% / 0.75** ⚠ |
| Split CP (PUNCC) | 89% / 0.92 | 88% / 0.96 | 90% / 1.06 | 89% / 1.14 | 87% / 1.23 | 88% / 1.57 | **85% / 1.19** ⚠ | **86% / 1.07** ⚠ | 88% / 1.08 |
| Split CP (MAPIE) | 95% / 1.13 | 90% / 1.26 | 92% / 1.35 | 89% / 1.20 | 92% / 1.06 | 93% / 1.53 | 90% / 1.49 | 93% / 1.49 | **78% / 0.74** ⚠ |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/05_conformal_mapie_puncc/README.md).*
