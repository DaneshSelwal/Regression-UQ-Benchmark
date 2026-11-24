# Pile settlement · NB6 NexCP and online conformal

[← Pile settlement README](../README.md) · [method guide](../../../notebooks/06_conformal_nexcp_online/README.md) · [report (PDF)](../../../papers/pile_settlement/paper.pdf)

| | |
|---|---|
| **What** | Weighted conformal (NexCP split, full, CV+, J+aB), online conformal (ACI, FACI, SF-OGD, SAOCP, online split/CV+/J+aB, MFCS), COP and Extreme CP. One Kaggle job per base model. |
| **Notebook** | [`Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb`](<Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_06_12](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-06-12), [REG23_11_2026_12_28](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-12-28), [REG23_11_2026_13_04](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-13-04), [REG23_11_2026_13_20](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-13-20), [REG23_11_2026_13_36](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-13-36), [REG23_11_2026_14_51](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-14-51), [REG23_11_2026_06_19](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-06-19) |
| **Executed output** | [`results/`](results/) (15 files: executed notebook(s) and Excel workbooks) |

> **Large notebooks.** These executed notebooks are mostly training logs (e.g. PGBM prints one line per tree) and may be too large for GitHub's notebook viewer. They are stored unchanged; open them with nbviewer, the Kaggle run, or download them: [`EXECUTED_TabNet.ipynb`](https://nbviewer.org/github/DaneshSelwal/Regression_Uncertainty_Quantification_Analysis/blob/version2.0/examples/pile_settlement_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/results/EXECUTED_TabNet.ipynb) (16 MB)

## Coverage / relative width, every method × model

⚠ = below the validity threshold (0.864). n/r = not run.

| Method | CatBoost | XGBoost | LightGBM | GPBoost | GradientBoosting | HistGB | TabNet |
|---|--:|--:|--:|--:|--:|--:|--:|
| NexCP Full | 90% / 0.53 | 90% / 0.51 | **83% / 0.35** ⚠ | 88% / 0.52 | 91% / 0.57 | 93% / 0.98 | 90% / 1.46 |
| NexCP Split | 93% / 1.08 | 89% / 1.11 | 91% / 1.03 | 87% / 0.96 | 91% / 0.93 | 93% / 1.50 | **83% / 0.98** ⚠ |
| NexCP CV+ | 91% / 1.14 | 91% / 1.19 | 91% / 1.13 | 92% / 1.07 | 93% / 0.99 | 94% / 1.54 | 96% / 1.28 |
| NexCP J+aB | 91% / 1.25 | 89% / 1.16 | 90% / 1.35 | 88% / 1.06 | 91% / 1.23 | 91% / 1.29 | 94% / 1.31 |
| MFCS Full | 89% / 0.69 | 90% / 0.74 | 90% / 0.72 | 90% / 0.79 | 90% / 0.69 | 89% / 0.86 | 90% / 0.74 |
| MFCS Split | **84% / 0.66** ⚠ | 88% / 0.83 | 88% / 0.82 | **84% / 0.79** ⚠ | 88% / 0.77 | 91% / 0.98 | **83% / 0.97** ⚠ |
| Online Split | **84% / 0.66** ⚠ | 88% / 0.83 | 88% / 0.82 | **84% / 0.79** ⚠ | 88% / 0.77 | 91% / 0.98 | **83% / 0.97** ⚠ |
| ACI | 89% / 0.82 | 89% / 1.01 | 90% / 0.86 | 88% / 1.07 | 90% / 0.85 | 91% / 0.91 | **86% / 1.57** ⚠ |
| FACI | 88% / 0.87 | 89% / 1.02 | 90% / 0.90 | 88% / 1.09 | 89% / 0.87 | 90% / 0.93 | 87% / 1.71 |
| Online J+aB | 87% / 0.91 | 88% / 0.96 | **86% / 1.01** ⚠ | 87% / 0.96 | 88% / 1.00 | 88% / 0.94 | 91% / 0.83 |
| Online CV+ | 90% / 0.85 | 90% / 0.96 | 90% / 0.94 | 90% / 1.02 | 92% / 0.88 | 92% / 1.08 | 96% / 1.43 |
| SF-OGD | 90% / 1.15 | 91% / 1.35 | 90% / 1.29 | 90% / 1.30 | 90% / 1.09 | 90% / 1.28 | 90% / 2.41 |
| SAOCP | 93% / 1.64 | 92% / 1.66 | 92% / 1.57 | 92% / 1.70 | 93% / 1.52 | 92% / 1.63 | 90% / 2.80 |
| COP | 90% / 0.89 | 90% / 1.07 | 90% / 0.97 | 89% / 1.15 | 90% / 0.93 | 90% / 0.93 | 88% / 1.73 |

## Extreme CP (one-sided upper bound y ≤ ŷ + q)

| Model | Coverage | q |
|---|--:|--:|
| CatBoost | 93.0% | 10.03 |
| GPBoost | 90.3% | 5.22 |
| GradientBoosting | 93.5% | 11.37 |
| HistGB | 91.9% | 13.99 |
| LightGBM | 90.9% | 7.32 |
| TabNet | 95.2% | 20.31 |
| XGBoost | 91.4% | 6.50 |

**Not run (not enough compute):** models NGBoost, PGBM (12-hour limit).

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/06_conformal_nexcp_online/README.md).*
