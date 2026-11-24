# Noise · NB6 NexCP and online conformal

[← Noise README](../README.md) · [method guide](../../../notebooks/06_conformal_nexcp_online/README.md) · [report (PDF)](../../../papers/noise/paper.pdf)

| | |
|---|---|
| **What** | Weighted conformal (NexCP split, full, CV+, J+aB), online conformal (ACI, FACI, SF-OGD, SAOCP, online split/CV+/J+aB, MFCS), COP and Extreme CP. One Kaggle job per base model. |
| **Notebook** | [`Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb`](<Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_06_10](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-06-10), [REG23_11_2026_11_26](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-11-26), [REG23_11_2026_12_12](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-12-12), [REG23_11_2026_13_13](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-13-13), [REG23_11_2026_13_52](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-13-52), [REG23_11_2026_14_02](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-02), [REG23_11_2026_14_12](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-12), [REG23_11_2026_14_23](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-23), [REG23_11_2026_14_40](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-40) |
| **Executed output** | [`results/`](results/) (19 files: executed notebook(s) and Excel workbooks) |

> **Large notebooks.** These executed notebooks are mostly training logs (e.g. PGBM prints one line per tree) and may be too large for GitHub's notebook viewer. They are stored unchanged; open them with nbviewer, the Kaggle run, or download them: [`EXECUTED_PGBM.ipynb`](https://nbviewer.org/github/DaneshSelwal/Regression_Uncertainty_Quantification_Analysis/blob/version2.0/examples/noise_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/results/EXECUTED_PGBM.ipynb) (10 MB)

## Coverage / relative width, every method × model

⚠ = below the validity threshold (0.836). n/r = not run.

| Method | CatBoost | XGBoost | LightGBM | GPBoost | GradientBoosting | HistGB | NGBoost | PGBM | TabNet |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NexCP Split | 92% / 1.00 | 92% / 1.06 | 95% / 1.21 | 93% / 1.03 | 92% / 1.27 | 97% / 1.34 | 87% / 1.04 | 90% / 0.99 | 92% / 2.14 |
| NexCP CV+ | 92% / 1.03 | 92% / 1.09 | 93% / 1.26 | 97% / 1.05 | 95% / 1.25 | 98% / 1.49 | 93% / 1.10 | 95% / 1.05 | 90% / 2.24 |
| NexCP J+aB | 95% / 1.09 | 93% / 1.04 | 90% / 1.00 | 93% / 1.04 | 93% / 1.21 | 95% / 1.21 | 93% / 1.16 | 90% / 1.14 | 90% / 1.93 |
| NexCP Full | 90% / 0.83 | 95% / 0.99 | 98% / 2.27 | 98% / 1.54 | 97% / 1.14 | 98% / 1.30 | 97% / 0.92 | 95% / 1.10 | 97% / 3.37 |
| FACI | 87% / 0.81 | 85% / 0.79 | 85% / 0.74 | 88% / 0.65 | **83% / 0.71** ⚠ | 87% / 0.90 | 85% / 0.93 | **83% / 0.86** ⚠ | 90% / 2.01 |
| ACI | 88% / 0.83 | 87% / 0.80 | 88% / 0.71 | 88% / 0.64 | 87% / 0.85 | 87% / 0.99 | 87% / 0.93 | 85% / 0.84 | 90% / 1.93 |
| MFCS Split | 88% / 0.86 | 90% / 0.85 | 90% / 0.72 | 90% / 0.64 | 87% / 0.90 | 95% / 1.11 | 87% / 0.93 | **83% / 0.86** ⚠ | 90% / 1.95 |
| Online Split | 88% / 0.86 | 90% / 0.85 | 90% / 0.72 | 90% / 0.64 | 87% / 0.90 | 95% / 1.11 | 87% / 0.93 | **83% / 0.86** ⚠ | 90% / 1.95 |
| Online CV+ | 87% / 0.88 | 88% / 0.86 | 92% / 0.88 | 93% / 0.81 | 92% / 0.92 | 93% / 1.11 | 88% / 0.94 | 87% / 0.90 | 93% / 2.19 |
| Online J+aB | 87% / 0.92 | 87% / 0.94 | 88% / 0.87 | 88% / 0.86 | 88% / 0.87 | 85% / 0.91 | 85% / 0.90 | **83% / 0.90** ⚠ | 90% / 1.95 |
| MFCS Full | 87% / 0.96 | 87% / 0.88 | 87% / 0.88 | 88% / 0.87 | 85% / 0.94 | 90% / 1.01 | 87% / 0.98 | 85% / 0.88 | 92% / 2.12 |
| SAOCP | 92% / 1.24 | 90% / 1.07 | 95% / 1.05 | 92% / 1.05 | 93% / 1.12 | 93% / 1.11 | 90% / 1.15 | 88% / 1.27 | 88% / 2.40 |
| SF-OGD | 92% / 1.20 | 90% / 1.10 | 92% / 1.14 | 92% / 1.19 | 92% / 1.15 | 92% / 1.07 | 92% / 1.15 | 90% / 1.35 | 90% / 2.34 |
| COP | 88% / 0.81 | 87% / 0.83 | 88% / 0.75 | 90% / 0.68 | 87% / 0.83 | 87% / 0.98 | 87% / 1.01 | 85% / 0.88 | 90% / 1.95 |

## Extreme CP (one-sided upper bound y ≤ ŷ + q)

| Model | Coverage | q |
|---|--:|--:|
| CatBoost | 83.3% | 3.55 |
| GPBoost | 90.0% | 4.74 |
| GradientBoosting | 93.3% | 4.43 |
| HistGB | 88.3% | 3.68 |
| LightGBM | 86.7% | 4.93 |
| NGBoost | 88.3% | 3.81 |
| PGBM | 91.7% | 4.19 |
| TabNet | 86.7% | 11.51 |
| XGBoost | 85.0% | 3.18 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/06_conformal_nexcp_online/README.md).*
