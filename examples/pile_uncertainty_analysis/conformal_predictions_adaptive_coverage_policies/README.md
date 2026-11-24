# Pile · NB7 Adaptive coverage policies (ACP)

[← Pile README](../README.md) · [method guide](../../../notebooks/07_adaptive_coverage_policies/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Adaptive coverage policies: a small network chooses each point's coverage level; e-value conformal sets give the guarantee at that level. |
| **Notebook** | [`Adaptive_Coverage_Policies(ACP).ipynb`](<Adaptive_Coverage_Policies(ACP).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_13_34](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-13-34) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Learned coverage per model (λ = 50)

| Model | Learned target 1 − α̃ | Coverage | Width / std(y) | Saturated? |
|---|--:|--:|--:|---|
| CatBoost | 82.9% | 100.0% | 5.13 | no |
| GPBoost | 83.0% | 100.0% | 5.23 | no |
| GradientBoosting | 81.8% | 100.0% | 5.42 | no |
| HistGB | 81.7% | 100.0% | 5.43 | no |
| LightGBM | 82.5% | 100.0% | 5.20 | no |
| NGBoost | 82.8% | 100.0% | 5.17 | no |
| PGBM | 82.6% | 100.0% | 5.10 | no |
| TabNet | 78.4% | 100.0% | 7.00 | no |
| XGBoost | 82.8% | 100.0% | 5.15 | no |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/07_adaptive_coverage_policies/README.md).*
