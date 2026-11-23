# Notebook 6 results: scour (NexCP, Adaptive CP, MFCS)

Run with FIX B15 (faster NexCP-Full: same intervals) and split into **one Kaggle job per model**,
so each model fits inside Kaggle's 12-hour limit. Every job runs the notebook's common cells,
then only that model's section. Each executed notebook is renamed after its model.

| Model | Kaggle run | Files |
|---|---|---|
| NGBoost | REG23_11_2026_06_40 | `EXECUTED_NGBoost.ipynb`, `NGBoost.xlsx` |
| TabNet | REG23_11_2026_11_31 | `EXECUTED_TabNet.ipynb`, `TabNet.xlsx` |
| PGBM | REG23_11_2026_12_35 | `EXECUTED_PGBM.ipynb`, `PGBM.xlsx` |
| GradientBoosting | REG23_11_2026_13_26 | `EXECUTED_GradientBoosting.ipynb`, `Gradient Boosting.xlsx` |
| CatBoost | REG23_11_2026_06_14 | `EXECUTED_CatBoost.ipynb`, `CatBoost.xlsx` |
| XGBoost | REG23_11_2026_06_15 | `EXECUTED_XGBoost.ipynb`, `XGBoost.xlsx` |
| Lightgbm | REG23_11_2026_14_19 | `EXECUTED_Lightgbm.ipynb`, `LightGBM.xlsx` |
| GPBoost | REG23_11_2026_14_30 | `EXECUTED_GPBoost.ipynb`, `GPBoost.xlsx` |
| HistGradientBoosting | REG23_11_2026_14_41 | `EXECUTED_HistGradientBoosting.ipynb`, `HGBM.xlsx` |
