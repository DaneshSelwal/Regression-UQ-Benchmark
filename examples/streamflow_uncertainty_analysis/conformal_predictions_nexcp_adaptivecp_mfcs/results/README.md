# Notebook 6 results: streamflow (NexCP, Adaptive CP, MFCS)

Run with FIX B15 (faster NexCP-Full: same intervals) and split into **one Kaggle job per model**,
so each model fits inside Kaggle's 12-hour limit. Every job runs the notebook's common cells,
then only that model's section. Each executed notebook is renamed after its model.

| Model | Kaggle run | Files |
|---|---|---|
| NGBoost | REG23_11_2026_06_59 | `EXECUTED_NGBoost.ipynb`, `NGBoost.xlsx` |
| TabNet | REG23_11_2026_11_40 | `EXECUTED_TabNet.ipynb`, `TabNet.xlsx` |
| PGBM | REG23_11_2026_13_10 | `EXECUTED_PGBM.ipynb`, `PGBM.xlsx` |
| GradientBoosting | REG23_11_2026_13_35 | `EXECUTED_GradientBoosting.ipynb`, `Gradient Boosting.xlsx` |
| CatBoost | REG23_11_2026_14_01 | `EXECUTED_CatBoost.ipynb`, `CatBoost.xlsx` |
| XGBoost | REG23_11_2026_14_11 | `EXECUTED_XGBoost.ipynb`, `XGBoost.xlsx` |
| Lightgbm | REG23_11_2026_14_22 | `EXECUTED_Lightgbm.ipynb`, `LightGBM.xlsx` |
| GPBoost | REG23_11_2026_14_33 | `EXECUTED_GPBoost.ipynb`, `GPBoost.xlsx` |
| HistGradientBoosting | REG23_11_2026_06_25 | `EXECUTED_HistGradientBoosting.ipynb`, `HGBM.xlsx` |
