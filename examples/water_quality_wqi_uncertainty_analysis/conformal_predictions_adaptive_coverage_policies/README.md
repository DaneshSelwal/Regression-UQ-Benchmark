# Water quality (WQI) · NB7 Adaptive coverage policies (ACP)

[← Water quality (WQI) README](../README.md) · [method guide](../../../notebooks/07_adaptive_coverage_policies/README.md) · [report (PDF)](../../../papers/water_quality_wqi/paper.pdf)

| | |
|---|---|
| **What** | Adaptive coverage policies: a small network chooses each point's coverage level; e-value conformal sets give the guarantee at that level. |
| **Notebook** | [`Adaptive_Coverage_Policies(ACP).ipynb`](<Adaptive_Coverage_Policies(ACP).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_08_51](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-08-51) |
| **Executed output** | [`results/`](results/) (11 files: executed notebook(s) and Excel workbooks) |

## Learned coverage per model (λ = 50)

| Model | Learned target 1 − α̃ | Coverage | Width / std(y) | Saturated? |
|---|--:|--:|--:|---|
| CatBoost | 20.9% | 54.9% | 0.53 | no |
| GPBoost | 30.5% | 54.9% | 0.47 | no |
| GradientBoosting | 21.9% | 52.4% | 0.53 | no |
| HistGB | 23.9% | 52.4% | 0.52 | no |
| LightGBM | 25.6% | 54.9% | 0.50 | no |
| NGBoost | 30.8% | 46.3% | 0.47 | no |
| PGBM | 24.0% | 51.2% | 0.51 | no |
| TabNet | 1.2% | 65.9% | 0.66 | no |
| XGBoost | 26.5% | 56.1% | 0.50 | no |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/07_adaptive_coverage_policies/README.md).*
