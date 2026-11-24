# Streamflow · NB6 NexCP and online conformal

[← Streamflow README](../README.md) · [method guide](../../../notebooks/06_conformal_nexcp_online/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Weighted conformal (NexCP split, full, CV+, J+aB), online conformal (ACI, FACI, SF-OGD, SAOCP, online split/CV+/J+aB, MFCS), COP and Extreme CP. One Kaggle job per base model. |
| **Notebook** | [`Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb`](<Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_06_59](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-06-59), [REG23_11_2026_11_40](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-11-40), [REG23_11_2026_13_10](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-13-10), [REG23_11_2026_13_35](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-13-35), [REG23_11_2026_14_01](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-01), [REG23_11_2026_14_11](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-11), [REG23_11_2026_14_22](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-22), [REG23_11_2026_14_33](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-33), [REG23_11_2026_06_25](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-06-25) |
| **Executed output** | [`results/`](results/) (19 files: executed notebook(s) and Excel workbooks) |

> **Large notebooks.** These executed notebooks are mostly training logs (e.g. PGBM prints one line per tree) and may be too large for GitHub's notebook viewer. They are stored unchanged; open them with nbviewer, the Kaggle run, or download them: [`EXECUTED_PGBM.ipynb`](https://nbviewer.org/github/DaneshSelwal/Regression-UQ-Benchmark/blob/version2.0/examples/streamflow_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/results/EXECUTED_PGBM.ipynb) (20 MB); [`EXECUTED_TabNet.ipynb`](https://nbviewer.org/github/DaneshSelwal/Regression-UQ-Benchmark/blob/version2.0/examples/streamflow_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/results/EXECUTED_TabNet.ipynb) (11 MB)

## Coverage / relative width, every method × model

⚠ = below the validity threshold (0.850). n/r = not run.

| Method | CatBoost | XGBoost | LightGBM | GPBoost | GradientBoosting | HistGB | NGBoost | PGBM | TabNet |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NexCP Full | 91% / 1.29 | **83% / 1.41** ⚠ | 88% / 1.66 | 90% / 1.26 | 88% / 1.37 | 91% / 1.37 | 86% / 1.27 | 85% / 1.34 | 97% / 2.75 |
| NexCP J+aB | 93% / 1.52 | 96% / 1.94 | 93% / 1.79 | 95% / 1.77 | 94% / 1.73 | 92% / 1.41 | 94% / 1.61 | 93% / 1.58 | 92% / 1.57 |
| NexCP CV+ | 95% / 1.72 | 96% / 1.98 | 92% / 1.62 | 92% / 1.53 | 94% / 1.91 | 96% / 1.98 | 96% / 1.88 | 93% / 1.76 | 98% / 2.43 |
| NexCP Split | 95% / 1.72 | 94% / 1.90 | 86% / 1.45 | 92% / 1.44 | 95% / 1.93 | 96% / 2.02 | 97% / 1.89 | 91% / 1.72 | 97% / 2.05 |
| Online J+aB | 90% / 1.35 | 92% / 1.60 | 91% / 1.44 | 89% / 1.35 | 88% / 1.39 | 89% / 1.29 | 89% / 1.37 | 91% / 1.44 | 92% / 1.65 |
| MFCS Split | 91% / 1.44 | 88% / 1.64 | 85% / 1.42 | 89% / 1.30 | 90% / 1.62 | 93% / 1.46 | 91% / 1.40 | 89% / 1.54 | 97% / 2.03 |
| Online Split | 91% / 1.44 | 88% / 1.64 | 85% / 1.42 | 89% / 1.30 | 90% / 1.62 | 93% / 1.46 | 91% / 1.40 | 89% / 1.54 | 97% / 2.03 |
| MFCS Full | 90% / 1.34 | 89% / 1.55 | 89% / 1.60 | 92% / 1.32 | 89% / 1.47 | 91% / 1.36 | 89% / 1.40 | 90% / 1.49 | 93% / 1.88 |
| ACI | 91% / 1.33 | 89% / 1.65 | 88% / 1.64 | 88% / 1.28 | 90% / 1.61 | 91% / 1.29 | 90% / 1.39 | 89% / 1.57 | 91% / 1.74 |
| FACI | 90% / 1.35 | 86% / 1.71 | 88% / 1.64 | 88% / 1.28 | 90% / 1.61 | 89% / 1.33 | 90% / 1.41 | 90% / 1.62 | 91% / 1.62 |
| Online CV+ | 91% / 1.44 | 92% / 1.71 | 92% / 1.62 | 92% / 1.34 | 91% / 1.65 | 92% / 1.41 | 91% / 1.46 | 91% / 1.66 | 98% / 2.27 |
| SF-OGD | 90% / 1.71 | 91% / 2.15 | 90% / 2.03 | 90% / 1.40 | 90% / 1.51 | 90% / 1.74 | 90% / 1.66 | 90% / 1.81 | 90% / 2.30 |
| SAOCP | 92% / 1.86 | 91% / 2.33 | 91% / 2.05 | 92% / 1.83 | 91% / 1.94 | 92% / 2.07 | 91% / 1.84 | 91% / 1.98 | 93% / 2.44 |
| COP | 90% / 1.44 | 90% / 1.79 | 89% / 1.65 | 89% / 1.33 | 90% / 1.60 | 90% / 1.51 | 89% / 1.40 | 89% / 1.73 | 90% / 1.60 |

## Extreme CP (one-sided upper bound y ≤ ŷ + q)

| Model | Coverage | q |
|---|--:|--:|
| CatBoost | 91.7% | 126.9 |
| GPBoost | 89.6% | 120.3 |
| GradientBoosting | 89.6% | 166.9 |
| HistGB | 92.7% | 146.7 |
| LightGBM | 89.6% | 117.9 |
| NGBoost | 91.7% | 124.2 |
| PGBM | 91.7% | 134.7 |
| TabNet | 97.9% | 161.6 |
| XGBoost | 96.9% | 234.1 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/06_conformal_nexcp_online/README.md).*
