# Pile · NB6 NexCP and online conformal

[← Pile README](../README.md) · [method guide](../../../notebooks/06_conformal_nexcp_online/README.md) · [report (PDF)](../../../papers/pile/paper.pdf)

| | |
|---|---|
| **What** | Weighted conformal (NexCP split, full, CV+, J+aB), online conformal (ACI, FACI, SF-OGD, SAOCP, online split/CV+/J+aB, MFCS), COP and Extreme CP. One Kaggle job per base model. |
| **Notebook** | [`Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb`](<Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_13_17](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-13-17) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

> **Large notebooks.** These executed notebooks are mostly training logs (e.g. PGBM prints one line per tree) and may be too large for GitHub's notebook viewer. They are stored unchanged; open them with nbviewer, the Kaggle run, or download them: [`EXECUTED_Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb`](https://nbviewer.org/github/DaneshSelwal/Regression_Uncertainty_Quantification_Analysis/blob/version2.0/examples/pile_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/results/EXECUTED_Conformal_Predictions%28NEXCP%2C%20Adaptive%20CP%2C%20mfcs%29.ipynb) (31 MB)

## Coverage / relative width, every method × model

⚠ = below the validity threshold (0.799). n/r = not run.

| Method | CatBoost | XGBoost | LightGBM | GPBoost | GradientBoosting | HistGB | NGBoost | PGBM | TabNet |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NexCP J+aB | 100% / 1.78 | 92% / 1.72 | 92% / 1.62 | 92% / 1.61 | 96% / 1.76 | 96% / 1.85 | 96% / 1.89 | 100% / 1.77 | 100% / 1.90 |
| NexCP Split | 96% / 1.70 | 92% / 1.79 | 92% / 1.75 | 92% / 1.80 | 88% / 1.94 | 92% / 2.02 | 96% / 2.31 | 96% / 1.67 | 88% / 2.73 |
| NexCP CV+ | 96% / 1.72 | 92% / 1.90 | 92% / 1.86 | 96% / 1.97 | 100% / 2.01 | 100% / 2.05 | 100% / 2.21 | 96% / 1.78 | 100% / 3.19 |
| NexCP Full | 88% / 1.75 | 88% / 1.58 | 83% / 2.57 | **75% / 1.84** ⚠ | 100% / 3.88 | 100% / 2.09 | 96% / 4.23 | 92% / 1.26 | 100% / 4.81 |
| ACI | 96% / 1.52 | 92% / 1.72 | 88% / 1.62 | 92% / 1.62 | 83% / 1.85 | 92% / 1.68 | 88% / 1.90 | 96% / 1.52 | 88% / 2.74 |
| FACI | 92% / 1.46 | 92% / 1.71 | 83% / 1.69 | 92% / 1.60 | 88% / 1.90 | 92% / 1.68 | 88% / 1.88 | 96% / 1.42 | 88% / 2.85 |
| Online J+aB | 96% / 1.75 | 92% / 1.64 | 92% / 1.57 | 92% / 1.63 | 96% / 1.50 | 96% / 1.77 | 96% / 1.70 | 96% / 1.72 | 96% / 1.88 |
| MFCS Split | 96% / 1.57 | 92% / 1.75 | **79% / 1.50** ⚠ | 92% / 1.69 | 83% / 1.84 | 92% / 1.96 | 92% / 2.14 | 96% / 1.67 | 88% / 2.73 |
| Online Split | 96% / 1.57 | 92% / 1.75 | **79% / 1.50** ⚠ | 92% / 1.69 | 83% / 1.84 | 92% / 1.96 | 92% / 2.14 | 96% / 1.67 | 88% / 2.73 |
| SAOCP | 88% / 1.35 | 88% / 1.96 | 88% / 1.83 | **79% / 1.58** ⚠ | 88% / 2.11 | 88% / 1.75 | 88% / 1.93 | 88% / 1.43 | 92% / 3.45 |
| Online CV+ | 96% / 1.60 | 92% / 1.83 | 92% / 1.79 | 92% / 1.81 | 100% / 1.95 | 96% / 1.91 | 96% / 2.06 | 96% / 1.67 | 100% / 3.32 |
| MFCS Full | 96% / 1.71 | 92% / 1.75 | 92% / 1.79 | 96% / 1.87 | 100% / 1.83 | 96% / 1.87 | 100% / 1.89 | 100% / 1.80 | 96% / 2.62 |
| SF-OGD | 92% / 1.95 | 88% / 1.80 | 92% / 1.90 | 92% / 2.11 | 88% / 2.08 | 88% / 1.72 | 88% / 2.03 | 88% / 1.60 | 88% / 4.85 |
| COP | 96% / 1.43 | 92% / 1.73 | 83% / 1.54 | 92% / 1.67 | 83% / 1.76 | 92% / 1.59 | 88% / 1.81 | 96% / 1.43 | 88% / 2.74 |

## Extreme CP (one-sided upper bound y ≤ ŷ + q)

| Model | Coverage | q |
|---|--:|--:|
| CatBoost | 75.0% | 0.375 |
| GPBoost | 79.2% | 0.637 |
| GradientBoosting | 83.3% | 0.512 |
| HistGB | 79.2% | 0.726 |
| LightGBM | 70.8% | 0.378 |
| NGBoost | 66.7% | 0.429 |
| PGBM | 70.8% | 0.431 |
| TabNet | 87.5% | 1.78 |
| XGBoost | 75.0% | 0.428 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/06_conformal_nexcp_online/README.md).*
