# Water quality (EWQI) · NB7 Adaptive coverage policies (ACP)

[← Water quality (EWQI) README](../README.md) · [method guide](../../../notebooks/07_adaptive_coverage_policies/README.md) · [report (PDF)](../../../papers/water_quality_ewqi/paper.pdf)

| | |
|---|---|
| **What** | Adaptive coverage policies: a small network chooses each point's coverage level; e-value conformal sets give the guarantee at that level. |
| **Notebook** | [`Adaptive_Coverage_Policies(ACP).ipynb`](<Adaptive_Coverage_Policies(ACP).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_21_42](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-21-42) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Learned coverage per model (λ = 50)

| Model | Learned target 1 − α̃ | Coverage | Width / std(y) | Saturated? |
|---|--:|--:|--:|---|
| CatBoost | 0.1% | 76.6% | 0.87 | yes (α̃ = 0.999) |
| GPBoost | 0.1% | 59.6% | 0.87 | yes (α̃ = 0.999) |
| GradientBoosting | 3.3% | 53.2% | 0.76 | no |
| HistGB | 0.1% | 78.7% | 0.99 | yes (α̃ = 0.999) |
| LightGBM | 0.1% | 63.8% | 0.87 | yes (α̃ = 0.999) |
| NGBoost | 1.8% | 72.3% | 0.77 | no |
| PGBM | 3.6% | 59.6% | 0.75 | no |
| TabNet | 11.5% | 55.3% | 0.68 | no |
| XGBoost | 7.0% | 59.6% | 0.72 | no |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/07_adaptive_coverage_policies/README.md).*
