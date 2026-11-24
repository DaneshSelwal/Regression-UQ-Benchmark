# Noise · NB7 Adaptive coverage policies (ACP)

[← Noise README](../README.md) · [method guide](../../../notebooks/07_adaptive_coverage_policies/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Adaptive coverage policies: a small network chooses each point's coverage level; e-value conformal sets give the guarantee at that level. |
| **Notebook** | [`Adaptive_Coverage_Policies(ACP).ipynb`](<Adaptive_Coverage_Policies(ACP).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_21_46](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-21-46) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Learned coverage per model (λ = 50)

| Model | Learned target 1 − α̃ | Coverage | Width / std(y) | Saturated? |
|---|--:|--:|--:|---|
| CatBoost | 64.0% | 88.3% | 1.16 | no |
| GPBoost | 64.8% | 88.3% | 1.12 | no |
| GradientBoosting | 64.2% | 91.7% | 1.16 | no |
| HistGB | 62.8% | 83.3% | 1.22 | no |
| LightGBM | 62.1% | 93.3% | 1.21 | no |
| NGBoost | 64.4% | 88.3% | 1.15 | no |
| PGBM | 66.6% | 93.3% | 1.11 | no |
| TabNet | 40.2% | 88.3% | 1.97 | no |
| XGBoost | 65.4% | 88.3% | 1.11 | no |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/07_adaptive_coverage_policies/README.md).*
