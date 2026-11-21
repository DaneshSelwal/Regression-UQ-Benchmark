# Change log: pile notebooks (branch `fix/pile-notebooks`)

Each fix is reviewed and approved one at a time before it is applied.
Every change in the notebooks is marked with a `# FIX <id>` comment, so `grep "FIX"` shows all of them.

## Before any fix: saved outputs did not come from pile
The outputs saved in all 8 pile notebooks show data shapes of (189, 8) / (95, 8) (the concrete
dataset) or (168, 6) / (60, 6) (the noise dataset). The pile data has 81 train / 24 test rows.
These notebooks had therefore never been run on pile. The old outputs are cleared in the fixed
notebooks and replaced by the Kaggle runs. Git history keeps the old outputs.

---

## Notebook 1: `hyperparameter_tuning/Optuna_autosampler.ipynb`

### Fix 1: hyperparameters were chosen on the test set (approved 2026-11-21)
- **Before:** each Optuna trial was trained on train and scored on **test**
  (`mean_squared_error(y_test, model.predict(X_test))`). The setting with the lowest test error
  (best of 8 pruners × 50 trials) was kept, and that same test error was reported as the result.
  Each model's "best pruner" was also picked by `test_mse`.
- **Why it's a problem:** the test set must stay unseen until the final check. Choosing among
  about 400 settings by their score on 24 test rows makes the reported test error look better
  than it really is. All later notebooks use these settings.
- **After:** each trial is scored by **5-fold cross-validation on the training data only**
  (`KFold(5, shuffle=True, random_state=42)`). The best setting is refit on all 81 training rows
  and evaluated **once** on test. `best_score` now holds the CV MSE (new column `cv_rmse`).
  `test_mse`, `test_rmse` and `test_corr_coef` are still reported. Each model's best pruner is
  chosen by CV score.
- **Expected effect:** the test RMSE may come out higher (it is now an honest estimate), and the
  tuned settings may change. Tuning takes about 5× longer.

### Fix 2: pruners never pruned, and runs were not repeatable (approved 2026-11-21)
- **Before:** trials never reported intermediate scores (`trial.report` / `should_prune` were
  never called), so none of the 8 pruners could stop a trial. The 8 "pruner" runs were really
  8 repeats of the same thing. `AutoSampler()` had no seed, so the ranking of pruners changed by
  chance from run to run.
  `ThresholdPruner(lower=0.1)` would, once pruning works, stop trials whose error falls
  **below** 0.1, which are the best ones.
- **After:**
  - After each CV fold the trial reports its score: the running mean MSE, or the per-fold MSE
    for `WilcoxonPruner`, which is designed to compare per-fold values. Bad trials can now be
    stopped early. A pruned trial appears as an empty point (NaN) in the RMSE-per-trial plot and
    sheet.
  - `AutoSampler(seed=42)`: reruns give the same results, and all pruners see the same sequence
    of suggested settings, so differences between pruners are real.
  - `ThresholdPruner(upper=var(y_train))`: stops a trial whose error is worse than simply
    predicting the training mean.
  - **New Excel sheet `Pruning_Summary`** (in `test.xlsx`). For each model × pruner: number of
    trials, completed, pruned, % pruned, best CV RMSE, test RMSE and tuning time.

### Fix 3: Gradient Boosting could be tuned with quantile loss (approved 2026-11-21)
- **Before:** the GBR search space included `loss='quantile'` (sklearn default level 0.9), and
  two saved GBR winners used it.
- **Why it's a problem:** with quantile loss the model deliberately predicts high (about the
  90th percentile) instead of the typical value. The conformal notebooks build intervals around
  this prediction, so the intervals become wider and lopsided.
- **After:** `loss` ∈ {squared_error, absolute_error, huber}. Quantile models remain in the
  quantile-regression notebook, which is not touched.

### K1: run on Kaggle (setup only, no change to the math)
- Cell 0: on Kaggle, the pile data (`daneshselwal/reguq-uq-data`: `d08_a.csv` = train,
  `d08_b.csv` = test) is copied to `./drive/MyDrive/Regression/pile_uncertainty_analysis/data/`,
  so every original path in the notebook works unchanged. On Colab the Drive mount runs as before.
- Cell 1: the original Colab install block is kept and runs only on Colab. On Kaggle (main
  account, internet on) only the missing packages are installed, and the kernel is not
  restarted with `os._exit(0)`.

### K2: tuning results saved to a file (approved 2026-11-21)
- **Before:** the chosen settings were copied by hand into 6 other notebooks
  (a pasted `best_scores_autosampler = {...}` block).
- **After:** a new cell right after tuning saves them to
  `hyperparameter_tuning/best_params.pkl`, plus a readable `best_params.json`. The other
  notebooks will load this file instead of using a pasted copy (applied notebook by notebook).

### B1: missing helper made the notebook crash (approved 2026-11-21)
- **Before:** `_ensure_excel_file(...)` was called while writing the Excel results but never
  defined in this notebook, so the first Kaggle run stopped with `NameError` after Random Forest.
  The other 7 notebooks define it.
- **After:** the same helper (copied unchanged) is defined in the helper cell. Results are not
  affected.
- First Kaggle run, before this fix: the Random Forest tuning finished with the new CV scoring,
  and **47 trials were pruned**. Pruning works now, where before 0 trials were ever pruned.

### Found, not yet discussed (no change made)
- The interpretability cell uses `model_classes`, which is never defined, so it fails with
  `NameError`.
- The interpretability cell passes the **unscaled** `x_test` to models trained on scaled data.
