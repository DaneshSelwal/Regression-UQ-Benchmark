# Concrete · NB7 Adaptive coverage policies (ACP)

[← Concrete README](../README.md) · [method guide](../../../notebooks/07_adaptive_coverage_policies/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Adaptive coverage policies: a small network chooses each point's coverage level; e-value conformal sets give the guarantee at that level. |
| **Notebook** | [`Adaptive_Coverage_Policies(ACP).ipynb`](<Adaptive_Coverage_Policies(ACP).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_20_52](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-20-52) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Learned coverage per model (λ = 50)

| Model | Learned target 1 − α̃ | Coverage | Width / std(y) | Saturated? |
|---|--:|--:|--:|---|
| CatBoost | 66.9% | 95.8% | 1.67 | no |
| GPBoost | 66.3% | 97.9% | 1.67 | no |
| GradientBoosting | 61.3% | 95.8% | 1.93 | no |
| HistGB | 64.4% | 95.8% | 1.75 | no |
| LightGBM | 65.7% | 97.9% | 1.69 | no |
| NGBoost | 66.1% | 96.8% | 1.67 | no |
| PGBM | 66.1% | 97.9% | 1.70 | no |
| TabNet | 45.5% | 84.2% | 2.76 | no |
| XGBoost | 66.0% | 97.9% | 1.68 | no |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/07_adaptive_coverage_policies/README.md).*
