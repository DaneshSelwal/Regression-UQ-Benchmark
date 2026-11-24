# Scour · NB6 NexCP and online conformal

[← Scour README](../README.md) · [method guide](../../../notebooks/06_conformal_nexcp_online/README.md) · [report (PDF)](../../../papers/scour/paper.pdf)

| | |
|---|---|
| **What** | Weighted conformal (NexCP split, full, CV+, J+aB), online conformal (ACI, FACI, SF-OGD, SAOCP, online split/CV+/J+aB, MFCS), COP and Extreme CP. One Kaggle job per base model. |
| **Notebook** | [`Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb`](<Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_06_40](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-06-40), [REG23_11_2026_11_31](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-11-31), [REG23_11_2026_12_35](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-12-35), [REG23_11_2026_13_26](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-13-26), [REG23_11_2026_06_14](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-06-14), [REG23_11_2026_06_15](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-06-15), [REG23_11_2026_14_19](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-19), [REG23_11_2026_14_30](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-30), [REG23_11_2026_14_41](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-41) |
| **Executed output** | [`results/`](results/) (19 files: executed notebook(s) and Excel workbooks) |

> **Large notebooks.** These executed notebooks are mostly training logs (e.g. PGBM prints one line per tree) and may be too large for GitHub's notebook viewer. They are stored unchanged; open them with nbviewer, the Kaggle run, or download them: [`EXECUTED_PGBM.ipynb`](https://nbviewer.org/github/DaneshSelwal/Regression_Uncertainty_Quantification_Analysis/blob/version2.0/examples/scour_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/results/EXECUTED_PGBM.ipynb) (48 MB)

## Coverage / relative width, every method × model

⚠ = below the validity threshold (0.844). n/r = not run.

| Method | CatBoost | XGBoost | LightGBM | GPBoost | GradientBoosting | HistGB | NGBoost | PGBM | TabNet |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NexCP Full | 94% / 1.32 | 97% / 1.46 | 94% / 1.47 | 95% / 1.58 | 96% / 1.59 | 95% / 1.51 | 94% / 1.47 | 96% / 1.31 | 100% / 4.83 |
| NexCP J+aB | 97% / 1.91 | 95% / 1.86 | 95% / 1.86 | 95% / 1.82 | 96% / 1.85 | 96% / 2.20 | 96% / 2.07 | 99% / 1.87 | 97% / 1.97 |
| NexCP CV+ | 97% / 1.91 | 97% / 1.98 | 96% / 2.25 | 96% / 2.13 | 96% / 2.03 | 97% / 1.97 | 96% / 1.96 | 99% / 1.89 | 100% / 3.16 |
| NexCP Split | 97% / 1.92 | 96% / 1.99 | 97% / 2.24 | 97% / 2.04 | 95% / 2.00 | 96% / 1.98 | 97% / 1.98 | 100% / 1.76 | 99% / 3.12 |
| FACI | 92% / 1.29 | 90% / 1.51 | 91% / 1.58 | 94% / 1.64 | 91% / 1.66 | 92% / 1.52 | 92% / 1.40 | 94% / 1.19 | 92% / 1.64 |
| ACI | 94% / 1.35 | 91% / 1.53 | 97% / 1.71 | 96% / 1.73 | 92% / 1.73 | 94% / 1.53 | 95% / 1.45 | 96% / 1.48 | 95% / 1.68 |
| MFCS Full | 95% / 1.42 | 95% / 1.49 | 97% / 1.63 | 97% / 1.65 | 94% / 1.68 | 95% / 1.54 | 95% / 1.58 | 91% / 1.49 | 92% / 1.77 |
| SF-OGD | 90% / 1.64 | 90% / 1.87 | 91% / 1.61 | 91% / 1.61 | 91% / 1.66 | 90% / 1.89 | 91% / 1.44 | 90% / 1.33 | 90% / 1.62 |
| Online J+aB | 95% / 1.48 | 94% / 1.61 | 95% / 1.65 | 95% / 1.63 | 94% / 1.66 | 94% / 1.69 | 94% / 1.64 | 96% / 1.54 | 96% / 1.67 |
| Online CV+ | 95% / 1.48 | 95% / 1.64 | 95% / 1.90 | 95% / 1.86 | 94% / 1.80 | 96% / 1.69 | 95% / 1.59 | 96% / 1.66 | 97% / 2.28 |
| MFCS Split | 95% / 1.43 | 95% / 1.68 | 97% / 2.04 | 97% / 1.95 | 94% / 1.80 | 94% / 1.64 | 95% / 1.59 | 97% / 1.70 | 99% / 2.42 |
| Online Split | 95% / 1.43 | 95% / 1.68 | 97% / 2.04 | 97% / 1.95 | 94% / 1.80 | 94% / 1.64 | 95% / 1.59 | 97% / 1.70 | 99% / 2.42 |
| SAOCP | 91% / 1.66 | 90% / 1.79 | 91% / 1.62 | 90% / 1.77 | 92% / 1.83 | 92% / 1.85 | 94% / 1.72 | 92% / 1.38 | 92% / 2.09 |
| COP | 94% / 1.28 | 91% / 1.56 | 96% / 1.82 | 95% / 1.73 | 94% / 1.94 | 94% / 1.48 | 94% / 1.36 | 96% / 1.33 | 95% / 1.70 |

## Extreme CP (one-sided upper bound y ≤ ŷ + q)

| Model | Coverage | q |
|---|--:|--:|
| CatBoost | 87.2% | 0.370 |
| GPBoost | 83.3% | 0.422 |
| GradientBoosting | 85.9% | 0.451 |
| HistGB | 87.2% | 0.348 |
| LightGBM | 82.1% | 0.347 |
| NGBoost | 89.7% | 0.516 |
| PGBM | 88.5% | 0.407 |
| TabNet | 96.2% | 0.776 |
| XGBoost | 85.9% | 0.469 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/06_conformal_nexcp_online/README.md).*
