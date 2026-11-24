# Water quality (WQI) · NB5 Conformal (MAPIE, PUNCC)

[← Water quality (WQI) README](../README.md) · [method guide](../../../notebooks/05_conformal_mapie_puncc/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Split conformal, CV+ and J+aB (MAPIE); split conformal, CV+ and CQR (PUNCC), wrapped around every tuned model. |
| **Notebook** | [`Conformal Predictions(MAPIE,PUNCC).ipynb`](<Conformal Predictions(MAPIE,PUNCC).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_08_50](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-08-50) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Coverage / relative width, every method × model

⚠ = below the validity threshold (0.846).

| Method | CatBoost | XGBoost | LightGBM | GPBoost | GradientBoosting | HistGB | NGBoost | PGBM | TabNet |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| CV+ (PUNCC) | 94% / 1.33 | 89% / 1.16 | 87% / 1.11 | **82% / 1.15** ⚠ | **80% / 1.14** ⚠ | 87% / 1.09 | **83% / 1.15** ⚠ | **84% / 1.07** ⚠ | 99% / 1.79 |
| J+aB (MAPIE) | 91% / 1.27 | 87% / 1.12 | 89% / 1.24 | **82% / 1.03** ⚠ | 90% / 1.21 | 87% / 1.14 | 89% / 1.18 | 87% / 1.15 | 98% / 1.46 |
| Split CP (MAPIE) | 89% / 1.16 | **83% / 1.06** ⚠ | 87% / 1.25 | **78% / 1.23** ⚠ | 88% / 1.29 | **79% / 1.27** ⚠ | 89% / 1.27 | 85% / 1.11 | 93% / 1.28 |
| CV+ (MAPIE) | 94% / 1.35 | 89% / 1.32 | 89% / 1.37 | **83% / 1.28** ⚠ | 95% / 1.56 | **83% / 1.22** ⚠ | 90% / 1.32 | 87% / 1.18 | 100% / 2.04 |
| Split CP (PUNCC) | 94% / 1.55 | 91% / 1.38 | 91% / 1.67 | 90% / 1.75 | 96% / 2.17 | 96% / 1.87 | 96% / 1.83 | 94% / 1.65 | 91% / 1.52 |
| CQR (PUNCC) | 94% / 1.96 | 99% / 2.25 | 99% / 2.04 | 88% / 1.77 | 95% / 2.58 | 91% / 1.69 | 89% / 1.69 | 87% / 1.62 | **83% / 1.80** ⚠ |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/05_conformal_mapie_puncc/README.md).*
