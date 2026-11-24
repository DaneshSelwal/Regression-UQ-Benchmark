# Sediment load · NB7 Adaptive coverage policies (ACP)

[← Sediment load README](../README.md) · [method guide](../../../notebooks/07_adaptive_coverage_policies/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Adaptive coverage policies: a small network chooses each point's coverage level; e-value conformal sets give the guarantee at that level. |
| **Notebook** | [`Adaptive_Coverage_Policies(ACP).ipynb`](<Adaptive_Coverage_Policies(ACP).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_01_38](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-38) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Learned coverage per model (λ = 50)

| Model | Learned target 1 − α̃ | Coverage | Width / std(y) | Saturated? |
|---|--:|--:|--:|---|
| CatBoost | 0.1% | 92.1% | 0.57 | yes (α̃ = 0.999) |
| GPBoost | 0.1% | 92.6% | 0.46 | yes (α̃ = 0.999) |
| GradientBoosting | 0.1% | 93.4% | 0.42 | yes (α̃ = 0.999) |
| HistGB | 0.1% | 90.4% | 0.55 | yes (α̃ = 0.999) |
| LightGBM | 0.1% | 93.2% | 0.49 | yes (α̃ = 0.999) |
| NGBoost | 0.1% | 91.8% | 0.43 | yes (α̃ = 0.999) |
| PGBM | 0.1% | 93.7% | 0.60 | yes (α̃ = 0.999) |
| TabNet | 0.1% | 93.7% | 0.73 | yes (α̃ = 0.999) |
| XGBoost | 0.1% | 91.5% | 0.64 | yes (α̃ = 0.999) |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/07_adaptive_coverage_policies/README.md).*
