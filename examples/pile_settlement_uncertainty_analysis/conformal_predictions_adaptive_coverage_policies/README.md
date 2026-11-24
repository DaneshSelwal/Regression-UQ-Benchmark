# Pile settlement · NB7 Adaptive coverage policies (ACP)

[← Pile settlement README](../README.md) · [method guide](../../../notebooks/07_adaptive_coverage_policies/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Adaptive coverage policies: a small network chooses each point's coverage level; e-value conformal sets give the guarantee at that level. |
| **Notebook** | [`Adaptive_Coverage_Policies(ACP).ipynb`](<Adaptive_Coverage_Policies(ACP).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_22_21](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-22-21) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Learned coverage per model (λ = 50)

| Model | Learned target 1 − α̃ | Coverage | Width / std(y) | Saturated? |
|---|--:|--:|--:|---|
| CatBoost | 65.4% | 82.3% | 0.66 | no |
| GPBoost | 63.9% | 79.0% | 0.69 | no |
| GradientBoosting | 64.9% | 84.4% | 0.66 | no |
| HistGB | 59.5% | 82.3% | 0.76 | no |
| LightGBM | 67.0% | 82.8% | 0.62 | no |
| NGBoost | 57.6% | 81.7% | 0.80 | no |
| PGBM | 64.7% | 82.8% | 0.67 | no |
| TabNet | 61.5% | 86.0% | 0.70 | no |
| XGBoost | 64.1% | 83.3% | 0.69 | no |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/07_adaptive_coverage_policies/README.md).*
