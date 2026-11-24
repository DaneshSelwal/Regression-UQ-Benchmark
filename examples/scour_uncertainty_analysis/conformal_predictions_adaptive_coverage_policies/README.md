# Scour · NB7 Adaptive coverage policies (ACP)

[← Scour README](../README.md) · [method guide](../../../notebooks/07_adaptive_coverage_policies/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Adaptive coverage policies: a small network chooses each point's coverage level; e-value conformal sets give the guarantee at that level. |
| **Notebook** | [`Adaptive_Coverage_Policies(ACP).ipynb`](<Adaptive_Coverage_Policies(ACP).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_21_45](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-21-45) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Learned coverage per model (λ = 50)

| Model | Learned target 1 − α̃ | Coverage | Width / std(y) | Saturated? |
|---|--:|--:|--:|---|
| CatBoost | 87.5% | 100.0% | 5.28 | no |
| GPBoost | 87.2% | 100.0% | 5.40 | no |
| GradientBoosting | 86.9% | 100.0% | 5.62 | no |
| HistGB | 87.6% | 100.0% | 5.18 | no |
| LightGBM | 86.7% | 100.0% | 5.67 | no |
| NGBoost | 87.5% | 100.0% | 5.22 | no |
| PGBM | 87.7% | 100.0% | 5.11 | no |
| TabNet | 84.6% | 100.0% | 6.57 | no |
| XGBoost | 87.8% | 100.0% | 5.12 | no |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/07_adaptive_coverage_policies/README.md).*
