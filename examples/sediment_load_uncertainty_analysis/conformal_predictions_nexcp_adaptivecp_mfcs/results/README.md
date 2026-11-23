# Notebook 6 results: sediment_load (NexCP, Adaptive CP, MFCS)

Run with FIX B15 (faster NexCP-Full: same intervals) and split into **one Kaggle job per model**,
so each model fits inside Kaggle's 12-hour limit. Every job runs the notebook's common cells,
then only that model's section. Each executed notebook is renamed after its model.

| Model | Kaggle run | Files |
|---|---|---|
| NGBoost | REG23_11_2026_13_40 | `EXECUTED_NGBoost.ipynb`, `NGBoost.xlsx` |
| GradientBoosting | REG23_11_2026_14_42 | `EXECUTED_GradientBoosting.ipynb`, `Gradient Boosting.xlsx` |
| CatBoost | REG23_11_2026_14_49 | `EXECUTED_CatBoost.ipynb`, `CatBoost.xlsx` |
| XGBoost | REG23_11_2026_14_54 | `EXECUTED_XGBoost.ipynb`, `XGBoost.xlsx` |
| Lightgbm | REG23_11_2026_15_01 | `EXECUTED_Lightgbm.ipynb`, `LightGBM.xlsx` |
| GPBoost | REG23_11_2026_15_04 | `EXECUTED_GPBoost.ipynb`, `GPBoost.xlsx` |
| HistGradientBoosting | REG23_11_2026_15_09 | `EXECUTED_HistGradientBoosting.ipynb`, `HGBM.xlsx` |

**Not run (out of compute):** TabNet, PGBM.

NexCP-Full was not run for any model (out of compute, FIX B16): its rows show empty values with the note 'not run: out of compute'. Every other method was run. TabNet and PGBM were skipped by decision (slow models, out of compute).
