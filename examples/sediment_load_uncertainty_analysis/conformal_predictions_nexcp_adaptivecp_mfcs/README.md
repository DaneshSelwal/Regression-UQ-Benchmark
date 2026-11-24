# Sediment load · NB6 NexCP and online conformal

[← Sediment load README](../README.md) · [method guide](../../../notebooks/06_conformal_nexcp_online/README.md) · [report (PDF)](../../../papers/sediment_load/paper.pdf)

| | |
|---|---|
| **What** | Weighted conformal (NexCP split, full, CV+, J+aB), online conformal (ACI, FACI, SF-OGD, SAOCP, online split/CV+/J+aB, MFCS), COP and Extreme CP. One Kaggle job per base model. |
| **Notebook** | [`Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb`](<Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_13_40](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-13-40), [REG23_11_2026_14_42](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-42), [REG23_11_2026_14_49](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-49), [REG23_11_2026_14_54](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-54), [REG23_11_2026_15_01](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-15-01), [REG23_11_2026_15_04](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-15-04), [REG23_11_2026_15_09](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-15-09) |
| **Executed output** | [`results/`](results/) (15 files: executed notebook(s) and Excel workbooks) |

## Coverage / relative width, every method × model

⚠ = below the validity threshold (0.874). n/r = not run.

| Method | CatBoost | XGBoost | LightGBM | GPBoost | GradientBoosting | HistGB | NGBoost |
|---|--:|--:|--:|--:|--:|--:|--:|
| NexCP Split | **87% / 0.24** ⚠ | **87% / 0.23** ⚠ | 89% / 0.20 | 90% / 0.25 | 89% / 0.13 | 88% / 0.18 | 90% / 0.22 |
| NexCP J+aB | 88% / 0.28 | 88% / 0.28 | 90% / 0.23 | 89% / 0.21 | 91% / 0.17 | 90% / 0.24 | 89% / 0.24 |
| NexCP CV+ | 91% / 0.40 | 92% / 0.50 | 92% / 0.32 | 93% / 0.46 | 92% / 0.21 | 95% / 0.34 | 94% / 0.38 |
| ACI | 93% / 0.51 | 92% / 0.57 | 92% / 0.45 | 93% / 0.38 | 92% / 0.32 | 93% / 0.42 | 92% / 0.48 |
| FACI | 92% / 0.59 | 92% / 1.05 | 93% / 0.80 | 93% / 0.43 | 92% / 0.41 | 91% / 0.71 | 93% / 0.75 |
| Online J+aB | 95% / 0.72 | 94% / 0.86 | 94% / 0.71 | 94% / 0.72 | 95% / 0.51 | 95% / 0.85 | 94% / 0.72 |
| MFCS Full | 93% / 0.73 | 94% / 0.85 | 94% / 0.72 | 94% / 0.74 | 94% / 0.50 | 93% / 0.80 | 93% / 0.70 |
| MFCS Split | 94% / 0.89 | 93% / 0.95 | 93% / 0.77 | 96% / 0.89 | 95% / 0.58 | 95% / 0.98 | 94% / 0.81 |
| Online Split | 94% / 0.89 | 93% / 0.95 | 93% / 0.77 | 96% / 0.89 | 95% / 0.58 | 95% / 0.98 | 94% / 0.81 |
| Online CV+ | 96% / 0.97 | 96% / 1.10 | 95% / 0.79 | 95% / 0.93 | 96% / 0.57 | 98% / 0.97 | 96% / 0.84 |
| SF-OGD | 90% / 4.43 | 90% / 4.28 | 90% / 4.32 | 90% / 3.98 | 90% / 4.10 | 90% / 3.78 | 90% / 3.49 |
| SAOCP | 99% / 8.78 | 99% / 8.93 | 100% / 8.58 | 99% / 8.16 | 100% / 8.53 | 100% / 7.81 | 99% / 6.86 |
| COP | 96% / 0.80 | 95% / 1.54 | 96% / 1.04 | 97% / 0.83 | 96% / 0.67 | 96% / 1.17 | 96% / 0.58 |

## Extreme CP (one-sided upper bound y ≤ ŷ + q)

| Model | Coverage | q |
|---|--:|--:|
| CatBoost | 97.5% | 425.0 |
| GPBoost | 97.8% | 222.6 |
| GradientBoosting | 97.3% | 233.7 |
| HistGB | 95.1% | 404.3 |
| LightGBM | 98.6% | 563.2 |
| NGBoost | 97.5% | 331.2 |
| XGBoost | 97.3% | 279.8 |

**Not run (not enough compute):** models TabNet, PGBM (12-hour limit). NexCP Full for all models (FIX B16).

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/06_conformal_nexcp_online/README.md).*
