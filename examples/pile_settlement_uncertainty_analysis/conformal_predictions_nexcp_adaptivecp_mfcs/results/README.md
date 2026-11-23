# Notebook 6 results: pile_settlement (NexCP, Adaptive CP, MFCS)

Run with FIX B15 (faster NexCP-Full: same intervals) and split into **one Kaggle job per model**,
so each model fits inside Kaggle's 12-hour limit. Every job runs the notebook's common cells,
then only that model's section. Each executed notebook is renamed after its model.

| Model | Kaggle run | Files |
|---|---|---|
| TabNet | REG23_11_2026_06_12 | `EXECUTED_TabNet.ipynb`, `TabNet.xlsx` |
| GradientBoosting | REG23_11_2026_12_28 | `EXECUTED_GradientBoosting.ipynb`, `Gradient Boosting.xlsx` |
| CatBoost | REG23_11_2026_13_04 | `EXECUTED_CatBoost.ipynb`, `CatBoost.xlsx` |
| XGBoost | REG23_11_2026_13_20 | `EXECUTED_XGBoost.ipynb`, `XGBoost.xlsx` |
| Lightgbm | REG23_11_2026_13_36 | `EXECUTED_Lightgbm.ipynb`, `LightGBM.xlsx` |
| GPBoost | REG23_11_2026_14_51 | `EXECUTED_GPBoost.ipynb`, `GPBoost.xlsx` |
| HistGradientBoosting | REG23_11_2026_06_19 | `EXECUTED_HistGradientBoosting.ipynb`, `HGBM.xlsx` |

**Not run (out of compute):** NGBoost, PGBM.

NGBoost and PGBM were not run to completion: not enough compute (Kaggle's 12-hour limit).
