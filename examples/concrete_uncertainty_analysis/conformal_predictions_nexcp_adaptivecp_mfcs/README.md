# Concrete · NB6 NexCP and online conformal

[← Concrete README](../README.md) · [method guide](../../../notebooks/06_conformal_nexcp_online/README.md) · [report (PDF)](../../../papers/concrete/paper.pdf)

| | |
|---|---|
| **What** | Weighted conformal (NexCP split, full, CV+, J+aB), online conformal (ACI, FACI, SF-OGD, SAOCP, online split/CV+/J+aB, MFCS), COP and Extreme CP. One Kaggle job per base model. |
| **Notebook** | [`Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb`](<Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_07_27](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-07-27), [REG23_11_2026_06_07](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-06-07), [REG23_11_2026_08_52](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-08-52), [REG23_11_2026_11_54](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-11-54), [REG23_11_2026_06_08](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-06-08), [REG23_11_2026_13_09](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-13-09), [REG23_11_2026_06_09](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-06-09), [REG23_11_2026_14_44](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-14-44), [REG23_11_2026_15_02](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-15-02) |
| **Executed output** | [`results/`](results/) (19 files: executed notebook(s) and Excel workbooks) |

> **Large notebooks.** These executed notebooks are mostly training logs (e.g. PGBM prints one line per tree) and may be too large for GitHub's notebook viewer. They are stored unchanged; open them with nbviewer, the Kaggle run, or download them: [`EXECUTED_PGBM.ipynb`](https://nbviewer.org/github/DaneshSelwal/Regression-UQ-Benchmark/blob/version2.0/examples/concrete_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/results/EXECUTED_PGBM.ipynb) (62 MB); [`EXECUTED_TabNet.ipynb`](https://nbviewer.org/github/DaneshSelwal/Regression-UQ-Benchmark/blob/version2.0/examples/concrete_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/results/EXECUTED_TabNet.ipynb) (10 MB)

## Coverage / relative width, every method × model

⚠ = below the validity threshold (0.849). n/r = not run.

| Method | CatBoost | XGBoost | LightGBM | GPBoost | GradientBoosting | HistGB | NGBoost | PGBM | TabNet |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| NexCP Full | 93% / 1.06 | 94% / 0.94 | 93% / 0.90 | 93% / 2.11 | 93% / 1.07 | 87% / 0.78 | 93% / 1.10 | 85% / 0.86 | 98% / 3.69 |
| NexCP J+aB | 94% / 1.15 | 92% / 1.12 | 93% / 1.12 | 94% / 1.16 | 95% / 1.56 | 87% / 0.98 | 91% / 1.04 | 93% / 1.14 | 95% / 2.15 |
| NexCP Split | 98% / 1.55 | 97% / 1.39 | 94% / 1.20 | 96% / 1.45 | 95% / 1.52 | 95% / 1.15 | 94% / 1.38 | 93% / 1.26 | 97% / 2.79 |
| NexCP CV+ | 96% / 1.53 | 96% / 1.44 | 96% / 1.30 | 98% / 1.51 | 95% / 1.51 | 91% / 1.21 | 96% / 1.38 | 95% / 1.38 | 99% / 3.29 |
| MFCS Full | 91% / 1.00 | 91% / 1.00 | 93% / 1.07 | 88% / 1.07 | 91% / 1.09 | 92% / 1.00 | 91% / 0.96 | 92% / 1.00 | 94% / 2.50 |
| FACI | 92% / 1.11 | 91% / 1.01 | 89% / 0.95 | 92% / 1.08 | 92% / 1.18 | 89% / 0.95 | 91% / 1.02 | 89% / 1.03 | 92% / 2.19 |
| Online J+aB | 92% / 1.12 | 93% / 1.09 | 92% / 1.07 | 93% / 1.05 | 93% / 1.40 | 87% / 0.98 | 91% / 1.02 | 89% / 1.06 | 95% / 2.38 |
| ACI | 92% / 1.23 | 92% / 1.14 | 93% / 1.02 | 93% / 1.10 | 92% / 1.18 | 91% / 0.95 | 93% / 1.13 | 91% / 1.07 | 94% / 2.27 |
| SAOCP | 92% / 1.16 | 94% / 1.24 | 92% / 1.14 | 94% / 1.21 | 94% / 1.39 | 91% / 1.12 | 91% / 1.25 | 91% / 1.21 | 92% / 7.13 |
| MFCS Split | 97% / 1.48 | 96% / 1.24 | 93% / 1.16 | 93% / 1.19 | 93% / 1.21 | 93% / 1.09 | 94% / 1.26 | 93% / 1.14 | 97% / 2.47 |
| Online Split | 97% / 1.48 | 96% / 1.24 | 93% / 1.16 | 93% / 1.19 | 93% / 1.21 | 93% / 1.09 | 94% / 1.26 | 93% / 1.14 | 97% / 2.47 |
| Online CV+ | 95% / 1.38 | 93% / 1.25 | 95% / 1.17 | 96% / 1.24 | 93% / 1.35 | 92% / 1.10 | 95% / 1.26 | 95% / 1.22 | 98% / 2.82 |
| SF-OGD | 91% / 1.04 | 91% / 1.15 | 91% / 1.31 | 92% / 1.29 | 91% / 1.28 | 91% / 1.17 | 91% / 1.30 | 91% / 1.24 | 91% / 6.15 |
| COP | 92% / 1.22 | 91% / 1.05 | 91% / 0.91 | 91% / 1.07 | 91% / 1.15 | 89% / 0.87 | 91% / 1.04 | 88% / 1.04 | 91% / 2.15 |

## Extreme CP (one-sided upper bound y ≤ ŷ + q)

| Model | Coverage | q |
|---|--:|--:|
| CatBoost | 91.6% | 7.81 |
| GPBoost | 92.6% | 8.61 |
| GradientBoosting | 93.7% | 7.57 |
| HistGB | 92.6% | 6.96 |
| LightGBM | 91.6% | 6.95 |
| NGBoost | 93.7% | 7.64 |
| PGBM | 92.6% | 8.66 |
| TabNet | 93.7% | 15.02 |
| XGBoost | 90.5% | 6.84 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/06_conformal_nexcp_online/README.md).*
