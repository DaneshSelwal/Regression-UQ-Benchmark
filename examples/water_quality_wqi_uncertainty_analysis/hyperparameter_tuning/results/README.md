# Final tuning results — water_quality_wqi

- Kaggle run: `REG22_11_2026_19_53` (`daneshselwal/reg22-11-2026-19-53`), see `docs/run_registry.tsv`.
- Notebook: the fixed pile `Optuna_autosampler.ipynb` (FIX B13 + B14 fast tuning) with this dataset's data and labels.
- `best_params.json`: the parameters the other notebooks use; `EXECUTED_*.ipynb` holds every cell output.
- **Run stopped at Kaggle's 12-hour limit in the last cell** (cell 21: model explanations, SHAP / InterpretML).
  The tuning itself finished in 254 minutes (all 10 models × 8 pruners, 80 entries in `best_params.json`)
  and its results were saved (cell 17) before the stop. The explanation step is slow here because this
  dataset has 151 features; its output in the executed notebook is incomplete. The other notebooks only
  use the tuned parameters.
