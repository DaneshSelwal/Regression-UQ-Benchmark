# Water quality (EWQI) · NB6 NexCP and online conformal

[← Water quality (EWQI) README](../README.md) · [method guide](../../../notebooks/06_conformal_nexcp_online/README.md) · [report (PDF)](../../../papers/water_quality_ewqi/paper.pdf)

| | |
|---|---|
| **What** | Weighted conformal (NexCP split, full, CV+, J+aB), online conformal (ACI, FACI, SF-OGD, SAOCP, online split/CV+/J+aB, MFCS), COP and Extreme CP. One Kaggle job per base model. |
| **Notebook** | [`Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb`](<Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_01_18](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-01-18) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

> **Large notebooks.** These executed notebooks are mostly training logs (e.g. PGBM prints one line per tree) and may be too large for GitHub's notebook viewer. They are stored unchanged; open them with nbviewer, the Kaggle run, or download them: [`EXECUTED_Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb`](https://nbviewer.org/github/DaneshSelwal/Regression-UQ-Benchmark/blob/version2.0/examples/water_quality_ewqi_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/results/EXECUTED_Conformal_Predictions%28NEXCP%2C%20Adaptive%20CP%2C%20mfcs%29.ipynb) (61 MB)

## Coverage / relative width, every method × model

⚠ = below the validity threshold (0.828). n/r = not run.

| Method | CatBoost | XGBoost | LightGBM | GPBoost | GradientBoosting | HistGB | NGBoost | PGBM | TabNet |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NexCP J+aB | 94% / 1.51 | 96% / 1.54 | 96% / 1.35 | 94% / 1.35 | 96% / 1.50 | 94% / 1.59 | 96% / 1.76 | 98% / 1.51 | 98% / 1.90 |
| NexCP CV+ | 98% / 1.99 | 98% / 1.70 | 96% / 1.57 | 94% / 1.55 | 96% / 1.89 | 100% / 2.17 | 100% / 2.33 | 98% / 1.83 | 100% / 3.23 |
| NexCP Split | 98% / 2.04 | 96% / 1.42 | 94% / 1.46 | 94% / 1.40 | 98% / 1.96 | 100% / 1.96 | 98% / 2.28 | 100% / 1.77 | 91% / 2.38 |
| NexCP Full | 89% / 3.35 | 98% / 2.60 | 98% / 2.24 | 98% / 2.30 | 94% / 2.88 | 96% / 2.79 | 100% / 4.42 | 100% / 2.58 | 98% / 4.46 |
| FACI | 94% / 1.36 | 94% / 1.29 | 91% / 1.23 | 91% / 1.22 | 91% / 1.33 | 98% / 1.36 | 91% / 1.82 | 94% / 1.26 | 91% / 2.35 |
| ACI | 94% / 1.43 | 94% / 1.32 | 91% / 1.25 | 91% / 1.24 | 91% / 1.38 | 98% / 1.56 | 94% / 1.88 | 94% / 1.33 | 89% / 2.36 |
| Online J+aB | 89% / 1.50 | 94% / 1.42 | 94% / 1.35 | 91% / 1.34 | 96% / 1.42 | 96% / 1.54 | 94% / 1.55 | 98% / 1.38 | 98% / 1.66 |
| SAOCP | 91% / 1.55 | 94% / 1.50 | 94% / 1.48 | 94% / 1.37 | 91% / 1.64 | 94% / 1.45 | 87% / 1.76 | 91% / 1.33 | 91% / 2.81 |
| MFCS Full | 89% / 1.37 | 94% / 1.43 | 94% / 1.53 | 94% / 1.53 | 100% / 1.51 | 96% / 1.58 | 98% / 1.61 | 98% / 1.41 | 94% / 2.13 |
| SF-OGD | 91% / 1.74 | 91% / 1.62 | 91% / 1.34 | 91% / 1.38 | 91% / 1.74 | 91% / 1.50 | 91% / 2.24 | 91% / 1.59 | 94% / 3.14 |
| Online CV+ | 96% / 1.71 | 98% / 1.50 | 94% / 1.44 | 94% / 1.44 | 96% / 1.67 | 96% / 1.84 | 100% / 2.07 | 98% / 1.54 | 100% / 3.22 |
| MFCS Split | 98% / 2.04 | 96% / 1.42 | 94% / 1.46 | 94% / 1.40 | 98% / 1.96 | 100% / 1.96 | 98% / 2.28 | 100% / 1.77 | 91% / 2.38 |
| Online Split | 98% / 2.04 | 96% / 1.42 | 94% / 1.46 | 94% / 1.40 | 98% / 1.96 | 100% / 1.96 | 98% / 2.28 | 100% / 1.77 | 91% / 2.38 |
| COP | 94% / 1.41 | 91% / 1.28 | 91% / 1.26 | 91% / 1.23 | 91% / 1.35 | 96% / 1.82 | 94% / 1.90 | 94% / 1.33 | 91% / 2.40 |

## Extreme CP (one-sided upper bound y ≤ ŷ + q)

| Model | Coverage | q |
|---|--:|--:|
| CatBoost | 95.7% | 50.32 |
| GPBoost | 74.5% | 19.77 |
| GradientBoosting | 87.2% | 27.34 |
| HistGB | 78.7% | 21.63 |
| LightGBM | 78.7% | 21.75 |
| NGBoost | 93.6% | 37.43 |
| PGBM | 89.4% | 23.48 |
| TabNet | 95.7% | 54.76 |
| XGBoost | 66.0% | 15.49 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/06_conformal_nexcp_online/README.md).*
