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

---

## Notebook 2: `quantile_regression/Quantile_Regression.ipynb`

### K1: run on Kaggle (setup only; same as Notebook 1)
On Colab the notebook runs exactly as before. On Kaggle the pile data is copied into the
original paths, and the missing packages are installed at the Colab versions.

### K2: load the tuned settings instead of a pasted copy (approved 2026-11-21)
- **Before:** a hand-pasted `best_scores_autosampler = {...}` from an older tuning run, which had
  been selected on the test set.
- **After:** loads `hyperparameter_tuning/best_params.pkl`, written by Notebook 1. On Kaggle it
  is read from the Notebook 1 run's output.

### Q1: XGBoost was trained with LightGBM's settings (approved 2026-11-21)
- **Before:** the XGBoost cell stored its settings in `best_paramst`, but the training loop used
  `best_params`, which still held LightGBM's settings from the cell above. XGBoost silently
  ignored the LightGBM-only settings and used the ones whose names matched.
- **After:** the loop uses `best_paramst`, the tuned XGBoost settings.

### Q2: quantile crossing (approved 2026-11-21)
- **Before:** each quantile level is a separate model, so for some rows the 13 predictions were
  out of order (old output for LightGBM, row 2: q0.05 = 34.48 > q0.10 = 33.39 > ... > q0.40 = 32.45).
  This gives negative interval widths, which make the average width look smaller, and flips the
  sign in the calibration step. Only the CRPS step sorted the values.
- **After:** a new helper `rearrange_quantiles` sorts each row's quantiles (monotone
  rearrangement, Chernozhukov et al. 2010) at the start of all 5 reporting functions. It never
  makes the quantile predictions worse. The values are reordered, not changed.
  **New in Excel:** the `Values` sheet of each model has a row "Rows with quantile crossing
  (before sorting)", giving the count and %. The same is printed in the notebook.

### Q3: TabNet early stopping checked the wrong thing (approved 2026-11-21)
- **Before:** `eval_set=[(X_train, y_train)]` scored with MSE. Checking on the training data
  itself means the score always improves, so early stopping never triggered (old log: "reached
  max_epochs = 200 with best_epoch = 199"). MSE also pulls every quantile model toward the middle.
- **After:** 20% of the **training** rows are held out as validation (`random_state=42`, same
  split for all quantiles). Each quantile's model is checked with its own pinball loss. Max
  epochs, patience and batch sizes are unchanged. The test set is not touched.

### Q4: CRPS from quantiles (approved 2026-11-21)
- **Before:** `ps.crps_ensemble(y, 13 quantiles)` treats the 13 levels as 13 equally likely
  random samples. The levels are unevenly spaced (5% apart at the edges, 10% in the middle), so
  the edges got too much weight. The result was biased differently for each model.
- **After:** CRPS = 2 · Σ wᵢ · pinballᵢ(y, qᵢ), the quantile form of CRPS (Gneiting & Raftery
  2007). wᵢ is the share of [0, 1] that level i represents, so the weights sum to 1. The new
  values are not directly comparable with the old CRPS numbers.

---

## Notebook 3: `probabilistic_distribution/Probabilistic__Distribution.ipynb`

### K1: run on Kaggle (setup only)
- The same environment cell as Notebooks 1–2.
- The Treeffuser fork pins numpy<2, scikit-learn==1.5.0 and lightgbm==4.3.0, which can't be
  installed on Kaggle's Python 3.13. On Kaggle the same fork is installed with `--no-deps`, plus
  its three small helpers (jaxtyping, einops, ml-collections). On Colab the install is unchanged.
- The IBUG cells run `%cd ibug`, `%cd ..`, `%cd ..` and then write to `./content/drive/...` and
  `/content/drive/...` (the Colab layout). On Kaggle, `/content` and `/kaggle/content` are linked
  to `/kaggle/working`, so all Excel files go to the same output folder. No path in the notebook
  was changed.

### K2: load the tuned settings instead of a pasted copy (approved 2026-11-21)
Same as Notebook 2.

### P2: IBUG uncertainty was square-rooted again (approved 2026-11-21)
- **Before:** `dev = scale**0.5` (LightGBM-IBUG) and `std_dev = scale ** 0.5` (XGBoost-IBUG),
  treating IBUG's `scale` as a variance.
- **Verified in IBUG's source** (github.com/jjbrophy47/ibug, `ibug/classes.py`): the value is
  computed as `scale = max(np.std(train_vals), min_scale)`, a **standard deviation**, and IBUG's
  own scoring uses it as "std. dev. values". Some of IBUG's comments loosely call it "variance",
  which is the likely source of the mix-up. IBUG's math itself is correct.
- **After:** `dev = scale` / `std_dev = scale`. The 30% validation split for IBUG now has
  `random_state=42`, so results can be repeated.

### P1: the CRPS decomposition carried no information (approved 2026-11-21)
- **Before:** "Reliability" = the model's own CRPS averaged over 10 groups of test rows, which is
  the total CRPS again. So Resolution = Uncertainty always (old outputs: Reliability ≈ total
  CRPS, Resolution ≈ Uncertainty, for every model).
- **Checked:** the total CRPS uses `properscoring.crps_gaussian`, which is correct and unchanged.
  `properscoring` has no decomposition function, and the repo docs cite no source for this split.
- **After:** Reliability, Resolution and Uncertainty from **Hersbach (2000)**, *Weather and
  Forecasting* 15: each predicted Gaussian is represented by 100 members at its quantile levels,
  then Reliability = Σ gᵢ(oᵢ − pᵢ)², CRPS_pot = Σ gᵢoᵢ(1 − oᵢ), Uncertainty = CRPS of the
  observations' own spread, and Resolution = Uncertainty − CRPS_pot.
  Reliability − Resolution + Uncertainty equals the 100-member CRPS, which is very close to
  Total_CRPS.
- **Verified on Kaggle** (kernel `hersbach-check`, synthetic data, n = 24 and 365):
  Reliability + CRPS_pot equals `properscoring.crps_ensemble` to within about 1e-15, and is within
  0.0004 of Total_CRPS. A calibrated forecast gets Reliability ≈ 0.0003; biased, overconfident
  or underconfident forecasts get 0.13–0.36.
- **Output format unchanged:** the same function and arguments, the same sheets
  (`CRPS_Summary_<model>`, `CRPS_Bins_<model>`), the same columns, the same per-bin table and the
  same prints.

### T1: Treeffuser's Gradient Boosting score model must use squared error (approved 2026-11-21)
- **Checked against the original repo** (github.com/blei-lab/treeffuser): the fork's diffusion
  math (noise, training targets, score) is identical to the original. The original documents the
  score model's objective as a least-squares (denoising) loss.
- **Before:** the GBR score model got the tuned GBR settings, which can include
  `loss='absolute_error'` or `'huber'`, so it learned the wrong target.
- **After:** `model_params={**best_params, 'loss': 'squared_error'}`, for the Treeffuser GBR
  score model only.
- **Not changed (to look at after the run):** the original sets up the score model with 3,000
  trees, early stopping and 30 noise repeats. Here it gets the tuned point-model settings and
  10 repeats, with no early stopping. If the Treeffuser results look broken after this run, a
  proposal will be made before anything changes.

### P3: XGBoost-IBUG's plots used LightGBM's predictions (approved 2026-11-21)
- **Before:** `generate_and_save_plots(predictions_LGBM_df, ..., "XGB_Prob.xlsx")`, while the
  other three reports in that file used `predictions_XGB_df`.
- **After:** `predictions_XGB_df`.

---

## Notebook 4: `probabilistic_distribution_card/Probabilistic__Distribution(CARD).ipynb`

### K1 / K2: Kaggle setup and loading the tuned settings
Same as Notebook 3.

### C1: CARD rewritten to follow the official code (approved 2026-11-21)
- **Before:** the network got only `[X, t]` and was trained to predict pure noise. It never saw
  the noisy target or the guide model's prediction (`residuals` was computed but unused). There
  was no noise schedule, and prediction did a single step at t = 0. So every output was
  `base_pred + N(0, 1)` noise, whatever the data.
- **After:** `CARDRegressor` follows the official CARD code, **github.com/XzwHan/CARD**
  (`regression/diffusion_utils.py`, `model.py`, `card_regression.py`, `configs/uci_*.yml`):
  - linear β schedule from 1e-4 to 0.02 over T steps
  - `q_sample`: y_t = √ᾱ_t·y₀ + (1 − √ᾱ_t)·ŷ₀ + √(1 − ᾱ_t)·ε
  - `ConditionalGuidedModel`: ε_θ(x, y_t, ŷ₀, t), 3 × 128 time-conditioned layers with softplus
  - noise-estimation loss with antithetic timestep sampling
  - the `p_sample_loop` reverse process (γ₀, γ₁, γ₂ posterior and the final t = 1→0 step)
  - the target standardised as `normalize_y: True`, and converted back afterwards
- **Official settings** (on your instruction): T = 1000, epochs = 5000, batch size 32,
  1000 samples per test point, lr = 1e-3.
- The class name and API are unchanged: `CARDRegressor(base_model=..., ...)`, `.fit`, and
  `.predict` → (mean, std). All 9 guide models and all outputs are unchanged.

### C2: EMA warm-up (approved 2026-11-21)
- The official code samples with an EMA of the weights (rate 0.9999). That needs 100k+ updates
  before the average stops being dominated by the random initial weights.
- **Kaggle test** (kernel `card-check`, synthetic heteroscedastic data with known std, n = 300):
  the official EMA without warm-up gave a mean predicted std of 26.4 (true 0.955) and coverage
  1.00. With warm-up it gave std 1.055, 90% coverage 0.915, and RMSE 1.173 (guide alone: 1.159).
- **After:** EMA decay = min(0.9999, (1 + k)/(10 + k)) at update k. This is the standard
  warm-up, and identical to the official EMA once training is long.

### P1: CRPS decomposition
Same function as Notebook 3, so the same approved Hersbach (2000) fix.

### B2: TabNet calibration curve was written into `CatBoost.xlsx` (file-path bug)
It now goes to `TabNet.xlsx`.

---

## Notebook 7 — Adaptive Coverage Policies: `conformal_predictions_adaptive_coverage_policies/Adaptive_Coverage_Policies(ACP).ipynb`

Checked against Gauthier, Bach & Jordan (AISTATS 2026), "Adaptive Coverage Policies in Conformal
Prediction", and the official code **github.com/GauthierE/adaptive-coverage-policies**
(`regression/regression.ipynb`). The following match and are unchanged: the width formula
2·Σscores / (α̃(n+1) − 1), the leave-one-out sums as input, the frozen base model, the 32-unit ReLU +
sigmoid network, and Adam with lr 1e-3 and batch size 32. The notebook defines the pipeline twice
(`conformal_predictions_ACP` and `run_full_acp_pipeline`); every fix is applied to both copies.

### K1 / K2: Kaggle setup and loading the tuned settings
Same as Notebook 2: the Kaggle env/data cell, a Kaggle-only pinned install (the Colab block is kept),
and the tuning results loaded from `best_params.pkl` instead of the pasted dict.

### A1: λ chosen by comparing losses across λ (approved 2026-11-21)
- **Before:** `_select_best_alpha_net` compared the final training loss *across* λ = 10, 20, 50.
  The loss is width + λ·α̃, so the best achievable loss always grows with λ. That means λ = 10, the
  narrowest and lowest-coverage setting, was always picked. The λ = 20/50 runs were trained and never used.
- **Why:** λ is the user's size/coverage trade-off. The paper reports each λ separately, and the
  official `selection-lambda.ipynb` picks λ to hit a target mean size, never by comparing losses across λ.
- **After:** new parameter `lambda_main=50` (the λ the paper reports numbers for). The best run is
  chosen **within** λ_main by its final training loss, and the main plots, `conformal_predictions_ACP_values`
  and the printout use it. New sheet `acp_lambda_comparison` evaluates every λ (its best run) on the test set:
  λ, α̃, target coverage 1 − α̃, test coverage, width, and miss / α̃.
- **Reporting:** the guarantee (eq. 3) is E[miss / α̃] ≤ 1, so `Target_coverage = 1 − α̃` and
  `Miss_over_alpha` are added to the values sheet, `acp_final_metrics` and the printout.

### A2: AlphaNet input not scaled, random initial output (approved 2026-11-21)
- **Before:** the network input (the sum of leave-one-out errors) was raw. On datasets with large errors or
  many calibration points it is in the thousands, where the sigmoid is flat, so α̃ stays stuck at its
  starting value. The output started at a random α̃.
- **After:** the training inputs are divided by their mean (≈ 1 on every dataset). The test input (the sum
  over all n) is divided by the same number, stored on the network as `feature_scale_`. The width formula
  still uses the raw errors. The output starts near α̃ ≈ 1 (final bias 5, weights N(0, 0.01)), as in the
  official `initialize_to_one`. Input scaling is our addition for generality; the official code feeds the raw sum.

### M4: one ALPHA
`ALPHA = 0.1` is defined once. AlphaNet learns its own α̃, so ALPHA is only the nominal reference.

### B5 (non-math): tuned parameters silently replaced by defaults
`_resolve_model_params` returned `{}` without any message when a model rejected its tuned parameters. It
now prints a warning naming the model and the error before it falls back.

### Note (no change)
Both cells define `_select_best_alpha_net` / `evaluate_acp`. The second definition overrides the first,
so `conformal_predictions_ACP` (never called; all runs use `run_full_acp_pipeline`) would get the
dict-returning version if it were called after the second cell. This was already the case before.

---

## Notebook 8 — HCM: `hyperspherical_confidence_mapping_hcm/Hyperspherical_Confidence_Mapping(HCM).ipynb`

Checked against Choi et al. (ICLR 2026), "Uncertainty Estimation via Hyperspherical Confidence
Mapping", and the official code **github.com/Abandoned-Puppy/HCM** (`Toy_example.py`). The network, the
loss, the prediction (R·d)[0], σ = |R|·√|‖d‖²−1|, the validation temperature scaling to 68/95/99.7%,
Cov@kσ and ECE_reg all match, so **no math changes**. The fixes below are all non-math.

### K1 (non-math): run on Kaggle
The Kaggle env/data cell is the same as in the other notebooks. HCM does not use treeffuser, and its
fork's pins cannot be installed on Python 3.13, so the treeffuser install is skipped on Kaggle. On Kaggle
only `properscoring` and `XlsxWriter` are installed. The Colab blocks are unchanged.

### B6 (non-math): silent switch to made-up data
- **Before:** if the train or test CSV was missing or empty, a warning was printed and the whole notebook ran
  on a generated demo dataset. The results could then look like real results for this dataset.
- **After:** it stops with `FileNotFoundError` and lists where it looked.

### B7 (non-math): output folder on Kaggle
On Kaggle, outputs now go to the same `./drive/MyDrive/Regression/<dataset>/...` tree as the other notebooks.
