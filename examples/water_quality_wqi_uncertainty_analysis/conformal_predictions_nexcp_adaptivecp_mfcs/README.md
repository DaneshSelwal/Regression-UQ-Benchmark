# Water quality (WQI) · NB6 NexCP and online conformal

[← Water quality (WQI) README](../README.md) · [method guide](../../../notebooks/06_conformal_nexcp_online/README.md) · [report (PDF)](../../../papers/water_quality_wqi/paper.pdf)

| | |
|---|---|
| **What** | Weighted conformal (NexCP split, full, CV+, J+aB), online conformal (ACI, FACI, SF-OGD, SAOCP, online split/CV+/J+aB, MFCS), COP and Extreme CP. One Kaggle job per base model. |
| **Notebook** | [`Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb`](<Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_08_59](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-08-59), [REG23_11_2026_09_10](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-09-10), [REG23_11_2026_10_06](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-10-06), [REG23_11_2026_10_09](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-10-09), [REG23_11_2026_10_18](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-10-18), [REG23_11_2026_10_23](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-10-23), [REG23_11_2026_11_07](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-11-07), [REG23_11_2026_11_12](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-11-12), [REG23_11_2026_11_21](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-11-21) |
| **Executed output** | [`results/`](results/) (19 files: executed notebook(s) and Excel workbooks) |

> **Large notebooks.** These executed notebooks are mostly training logs (e.g. PGBM prints one line per tree) and may be too large for GitHub's notebook viewer. They are stored unchanged; open them with nbviewer, the Kaggle run, or download them: [`EXECUTED_PGBM.ipynb`](https://nbviewer.org/github/DaneshSelwal/Regression-UQ-Benchmark/blob/version2.0/examples/water_quality_wqi_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/results/EXECUTED_PGBM.ipynb) (43 MB)

## Coverage / relative width, every method × model

⚠ = below the validity threshold (0.846). n/r = not run.

| Method | CatBoost | XGBoost | LightGBM | GPBoost | GradientBoosting | HistGB | NGBoost | PGBM | TabNet |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NexCP J+aB | 94% / 1.33 | **84% / 1.04** ⚠ | 89% / 1.16 | **84% / 1.07** ⚠ | 89% / 1.21 | 87% / 1.15 | **84% / 1.14** ⚠ | 85% / 1.09 | 93% / 1.35 |
| NexCP Split | 93% / 1.38 | **72% / 0.86** ⚠ | 87% / 1.23 | **80% / 1.38** ⚠ | **84% / 1.17** ⚠ | **74% / 1.05** ⚠ | 87% / 1.23 | **83% / 1.04** ⚠ | 94% / 1.57 |
| NexCP CV+ | 95% / 1.39 | 85% / 1.06 | 87% / 1.24 | 91% / 1.41 | 89% / 1.27 | 85% / 1.14 | 85% / 1.21 | 85% / 1.13 | 99% / 1.82 |
| NexCP Full | **79% / 1.61** ⚠ | 89% / 1.22 | **79% / 0.86** ⚠ | 88% / 1.66 | **80% / 1.46** ⚠ | 87% / 1.47 | 90% / 1.90 | 87% / 1.16 | 99% / 3.04 |
| MFCS Split | 91% / 1.34 | **72% / 0.86** ⚠ | **83% / 1.15** ⚠ | **80% / 1.37** ⚠ | **84% / 1.18** ⚠ | **73% / 1.04** ⚠ | 87% / 1.23 | **83% / 1.04** ⚠ | 95% / 1.70 |
| Online Split | 91% / 1.34 | **72% / 0.86** ⚠ | **83% / 1.15** ⚠ | **80% / 1.37** ⚠ | **84% / 1.18** ⚠ | **73% / 1.04** ⚠ | 87% / 1.23 | **83% / 1.04** ⚠ | 95% / 1.70 |
| MFCS Full | 91% / 1.19 | 89% / 1.11 | 94% / 1.22 | 93% / 1.31 | 93% / 1.27 | 89% / 1.25 | 91% / 1.21 | 88% / 1.13 | 91% / 1.63 |
| Online J+aB | 95% / 1.46 | 89% / 1.18 | 89% / 1.27 | 89% / 1.17 | 90% / 1.29 | 89% / 1.24 | 90% / 1.26 | 89% / 1.26 | 93% / 1.46 |
| ACI | 90% / 1.37 | 87% / 1.18 | 87% / 1.26 | 87% / 1.65 | 87% / 1.26 | **80% / 1.38** ⚠ | 87% / 1.22 | 85% / 1.17 | 94% / 1.63 |
| Online CV+ | 94% / 1.38 | 88% / 1.15 | 89% / 1.28 | 91% / 1.58 | 90% / 1.31 | 90% / 1.32 | 87% / 1.25 | 88% / 1.17 | 99% / 1.89 |
| FACI | 91% / 1.38 | **84% / 1.24** ⚠ | 87% / 1.32 | 87% / 1.75 | 87% / 1.31 | 85% / 1.47 | 88% / 1.26 | 88% / 1.21 | 91% / 1.39 |
| SAOCP | 91% / 1.43 | 89% / 1.48 | 89% / 1.51 | 93% / 1.89 | 91% / 1.63 | 89% / 1.86 | 90% / 1.54 | 91% / 1.59 | 91% / 1.51 |
| SF-OGD | 90% / 1.34 | 89% / 1.34 | 89% / 1.83 | 90% / 1.82 | 89% / 1.62 | 89% / 1.72 | 89% / 1.68 | 90% / 1.66 | 90% / 1.62 |
| COP | 90% / 1.41 | 85% / 1.19 | 87% / 1.34 | 87% / 1.67 | 89% / 1.34 | **83% / 1.45** ⚠ | 88% / 1.19 | 88% / 1.31 | 94% / 1.47 |

## Extreme CP (one-sided upper bound y ≤ ŷ + q)

| Model | Coverage | q |
|---|--:|--:|
| CatBoost | 96.3% | 19.95 |
| GPBoost | 93.9% | 23.84 |
| GradientBoosting | 92.7% | 17.15 |
| HistGB | 98.8% | 46.02 |
| LightGBM | 92.7% | 26.39 |
| NGBoost | 93.9% | 23.60 |
| PGBM | 95.1% | 24.76 |
| TabNet | 98.8% | 45.16 |
| XGBoost | 95.1% | 23.83 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/06_conformal_nexcp_online/README.md).*
