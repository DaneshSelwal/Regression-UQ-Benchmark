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

### B10 (non-math): last cell crashed with NameError
`generate_interpretml_explanations_summary_pruners` looks models up in `model_classes`, which this
notebook never defined (also true in v1.2), so the interpretability summary (last cell) always failed.
The same 9-model mapping the other notebooks use is now defined in that cell. Found in the 5-trial
test run. The tuning results themselves were not affected (they are saved two cells earlier).

### Found, not yet discussed (no change made)
- The interpretability cell uses `model_classes`, which is never defined, so it fails with
  `NameError`.
- The interpretability cell passes the **unscaled** `x_test` to models trained on scaled data.

---

### B13: all trials pruned → crash (found by the 5-trial smoke runs, 2026-11-22)
- **Problem:** `ThresholdPruner(upper=var(y_train))` stops a trial whose running CV error is worse than predicting
  the mean. On sediment_load, noise, water_quality_ewqi and pile_settlement it stopped all 5 NGBoost trials, so
  no `NGBoost_ThresholdPruner_BEST.pkl` was saved and loading it raised `FileNotFoundError`.
- **Fix (non-math):** a pair with no completed trial is recorded in `Pruning_Summary` (Completed 0, RMSE NaN)
  and skipped. Every model keeps its `NopPruner` entry, which never prunes, so the later notebooks always find
  params. Pile is unaffected (no pile pair had every trial pruned).

### B14: faster tuning, same search (approved 2026-11-22)
- **Problem:** the 80 studies (10 models × 8 pruners) ran one after another on one core, and every repeated
  parameter set was trained again: on pile 66% of completed trials repeated an already-scored set. Big
  datasets needed 13–15 h (Kaggle stops a run at 12 h).
- **Fix (speed only):** (1) each model keeps a table parameter set → its 5 fold MSEs and its full-training-set
  refit; a repeated set takes the stored values instead of being trained again. The study receives the same
  fold values, so the pruners, Optuna's next suggestions and the selected best model are unchanged.
  (2) models are tuned 2 at a time in separate processes (Kaggle CPU = 2 cores × 2 threads), longest first;
  logs, Excel sheets, plots, `best_params` and saved models are gathered in the original order.
- **Check (scour, 5 trials, REG22_11_2026_17_30 vs the old code REG22_11_2026_13_04):** tuning 57.2 → 5.8 min.
  For all 10 models the overall best parameters (lowest CV RMSE over the 8 pruners) are identical; test RMSE
  identical for 9. The differences come from models whose own training is slightly random (NGBoost, GPBoost,
  CatBoost): the old code re-trained the same set several times and kept the luckiest score (NGBoost: one set
  scored 0.599–0.622 across pruners); now each set is trained once (NGBoost 0.630 in every study).

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

### B8 (non-math): TabNet metric crashed on Kaggle
`pytorch-tabnet2` passes sample weights to custom metrics as a third argument; the pinball metric added
in Q3 now accepts `weights=None`. Found in the first demo run (cell 61).

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

## Notebook 5: `conformal_predictions_mapie_puncc/Conformal Predictions(MAPIE,PUNCC).ipynb`

### K1 / K2: Kaggle setup and loading the tuned settings
Same as Notebook 2. On Kaggle only the missing packages are installed (adds `mapie==0.6.0`,
`puncc==0.8.0`); the Colab install block is kept unchanged under `else:`.

### M1: CQR used ordinary (mean) models (approved 2026-11-21)
- **Before:** the CQR strategy received the same mean-regression model for the lower, middle and upper
  estimators, so it was not quantile regression at all.
- **After:** new `QuantileModel(model_class, params, q)` wrapper. Native quantile loss where the library
  has one (LightGBM / GPBoost `objective="quantile"`, XGBoost `reg:quantileerror`, Gradient Boosting
  `loss="quantile"`, CatBoost `Quantile:alpha=q`, TabNet with a pinball `loss_fn`); distribution-based
  quantiles for NGBoost (`pred_dist.ppf`) and PGBM (`predict_dist`). CQR now uses q = α/2 and 1 − α/2.

### M2: MAPIE interval array had an extra axis (approved 2026-11-21)
MAPIE 0.6 returns intervals of shape (n, 2, n_alpha). The notebook compared (n, 2, 1) arrays with
(n,) targets, which broadcast to an n × n coverage matrix. **After:** `[:, :, 0]` after each predict.

### M3: Split and J+aB strategies were not what their names say (approved 2026-11-21)
- Split: `cv=ShuffleSplit(n_splits=1, test_size=0.5, random_state=SEED)` — one fit / calibration split
  (was `method="base", cv=KFold(n_splits=5)`, i.e. a 5-fold jackknife, not split conformal).
- J+aB: `method="plus"` with `Subsample(n_resamplings=50, random_state=SEED)` (was `method="minmax"`,
  which is the J+aB *minmax* variant, not J+aB).

### M4: one ALPHA
`ALPHA = 0.1` for MAPIE and PUNCC (was 0.05 for MAPIE and 0.1 for PUNCC, so results were not comparable).

### M5 / M6: tuned settings were not always used
- M5: the CatBoost PUNCC call now passes `best_params` (was defaults).
- M6: `PGBMWrapper` gets `get_params` / `set_params`, so MAPIE's `clone` keeps the tuned settings.

### B3 / B4 (non-math)
- B3: the Gradient Boosting MAPIE run was labelled "GPBoost".
- B4: `KFold(..., shuffle=True)` without a seed → `random_state=42`.

### B9 (non-math): MAPIE 0.6 rejected the split from M3
The first demo run stopped at the first MAPIE call: MAPIE 0.6 accepts only `BaseCrossValidator`
objects, and `ShuffleSplit` is not one. The same split (a seeded random half of the training rows for
calibration, the other half to fit) is now passed as a `PredefinedSplit` via `_split_cv(n, SEED)`.

## Notebook 6: `conformal_predictions_nexcp_adaptivecp_mfcs/Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb`

All changes are in one helper cell (inserted before `conformal_predictions_NEXCP`) plus the method
sections of every copy (generic, `_PGBM`, `_tabnet`). Metrics, plots and Excel layout are unchanged
except for the added columns/sheets named below. Sources: see REFERENCES.md (NEXCP section).

### K1 / K2: Kaggle setup and loading the tuned settings
Same as Notebook 5.

### N1: calibration residuals came from the training fit (approved 2026-11-21)
- **Before:** every method measured errors on the same rows the model was trained on; such errors are
  much smaller than errors on new data, so intervals were too narrow.
- **After:** out-of-fold residuals (5-fold, seed 123): each training row is scored by a model that never
  saw it. Used by NexCP Split / CV+, all online methods, COP's starting value and MFCS-Full.

### N2: weighted quantile (approved 2026-11-21)
- **Before:** `np.interp` between residuals, no mass for the test point.
- **After:** NexCP eq. (10–11): step quantile of Σ w̃ᵢ δ_Rᵢ + w̃ₙ₊₁ δ₊∞ with w̃ = w / (Σw + 1). If the
  quantile is +∞ the interval is capped at the largest residual and the **Note** column says so.
  The old function is kept, renamed `_old_weighted_quantile_interp`.

### N3: NexCP Full / J+aB / CV+ were not the NexCP methods (approved 2026-11-21, option A)
- Full: was identical to an unweighted split. Now eq. (12): for each test point, refit on train + (x, y)
  for `full_grid_size=50` candidate y values in ŷ ± 2·q_split and keep the y values that conform.
- J+aB: out-of-bag predictions (rows not in a bootstrap sample) + eq. (33) weighted jackknife+ bounds
  (was min/max of in-bag intervals).
- CV+: eq. (33) with K folds (was averaged bounds).

### N4–N6: online methods (approved 2026-11-21)
Ported from the authors' code (salesforce/online_conformal) and the papers:
- SF-OGD (gradient sign and look-ahead fixed), SAOCP (coin-betting experts), FACI (expert weights,
  I = 100), **ACI added** (γ = 0.005, N5).
- Every interval is output **before** that test point's error is used (no look-ahead).
- Online CV+ / J+aB: jackknife+ with past test errors added to the calibration set as they arrive.
- The sliding-window variants were removed (N6). MFCS Split / Full kept as written (no published source);
  MFCS-Full now uses out-of-fold residuals of each refit.

### N8: COP step size (approved 2026-11-21)
`lr_t = lr × (max − min)` of the last `T_burnin` scores (training OOF errors followed by past test
errors), as in the official COP code. Was a fixed 0.05 in target units. Start value from OOF residuals.

### N9: Extreme CP (approved 2026-11-21)
As in the official `ExtremeConformal` package (`conformalize_GPD_max`): signed scores y − ŷ on all
calibration points; thresholds 0.80–0.99 (capped so ≥ 10 scores exceed them); official GPD quantile
formula; maximum over thresholds. One-sided: coverage = P(y ≤ ŷ + q), width = q, `ymin` empty,
plot shows the upper bound. New sheet `EXTREMECP_GPD_Thresholds` with the fit at every threshold.

### N10: notes
New **Note** column in the NEXCP / ONLINECP / EXTREMECP summaries (caps at +∞, grid-edge hits, fallbacks).

### A1 / A2: AlphaNet (approved 2026-11-21, same as Notebook 7)
- Input = sum of LOO scores divided by its mean (was the raw sum).
- Summary: `Alpha_nominal (not used for the interval)`, `Target_coverage = 1 − α_learned`,
  `Miss_over_alpha = (1 − coverage) / α_learned`.

### A3: AlphaNet training budget (2026-11-22, same as Notebook 7)
`lr_alpha` 1e-3 → 1e-2, and `num_epochs` raised so training has at least 2000 optimizer steps. The
leave-one-out scores were already precomputed here. The progress print is now about 10 lines per run.

### B11 (non-math): `.to(device)` inside a comment
An earlier A2 edit left `.to(device)` inside the `# FIX A2` comment on the test-feature line, so the test
input was not moved to the GPU. That's harmless on CPU and would crash on a GPU. It's now code again.

### B15 (non-math): NexCP-Full speed-up, same intervals (approved 2026-11-23)
NexCP-Full (N3) refits the model for each of 50 candidate y values at every test point, and keeps the
interval [smallest kept y, largest kept y]. On pile it took 163 of the notebook's 207 minutes, and the
larger datasets would have needed 18–700 hours (Kaggle stops a run after 12).
- **Edge scan:** for each test point, check candidates from the bottom of the grid up to the first kept
  y, then from the top down to the first kept y. The keep rule is unchanged, so lo and hi are exactly
  the values the full scan gives, as are the "reaches the grid edge" and "empty set" notes. Candidates
  in the middle are skipped.
- **Workers:** test points can run on 2 worker processes. The code times 3 points with 1 worker and 6 with
  2, then uses the faster way for the rest. A model that already uses every core (LightGBM) gets slower
  with 2 workers (1.0 vs 26.8 s per point on pile), so it stays on 1. If the workers fail, it falls back to 1.
- **Split runs:** for large datasets, the notebook can run as one Kaggle job per model section (shared
  setup cells, the definitions from earlier sections, then that model's cells); each model writes its
  own workbook, so the merged result is the same set of files.
- **Check on pile (REG23_11_2026_05_10 … 05_51, one job per model, against REG22_11_2026_13_17):** every
  NexCP, online CP, COP and Extreme CP value is identical for 8 models. NGBoost's NexCP-Full and J+aB
  values differ by up to 1.0 because NGBoost is not exactly repeatable from run to run (J+aB is not
  touched by B15). AlphaNet differs only in run time and by at most 1.5e-6 (CatBoost) from run-to-run
  variation in its training. NexCP-Full time: 163 → 19 min in total (NGBoost 79.7 → 3.5, TabNet
  50.6 → 4.9, Gradient Boosting 10.7 → 0.2).

### M4: one ALPHA
`ALPHA = 0.1` in the helper cell; all 45 method calls use it (were 0.05 / 0.1 mixed).

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

### A3: AlphaNet hardly trained, so λ had no effect (decided 2026-11-22 under the user's "change it only if the impact is big" rule)
- **Before:** 50 epochs at lr 1e-3. On pile the calibration set is 29 points with batch size 32, so one
  epoch is one optimizer step: 50 steps in total. Starting from α̃ ≈ 0.99 (bias 5), α̃ ended at about 0.98
  for λ = 10, 20 and 50 alike, so the λ comparison showed nothing. (The official code gets away with 50
  epochs because it feeds the raw sum, about 80 there, which makes each step large. After A2's scaling the
  input is about 1, so the steps are small.)
- **After:** at least 2000 optimizer steps (`epochs = max(epochs, ceil(2000 / batches per epoch))`) at lr 1e-2.
  The leave-one-out score vectors are computed once and reused, giving the same numbers as before; it was
  one `predict()` per row per step, which would be far too slow with 2000 steps.
- **Check (Kaggle run REG22_11_2026_08_09, LightGBM and XGBoost, pile):** the trained α̃ matches the exact
  minimiser of the same loss for every λ (0.347 / 0.256 / 0.174 for λ = 10 / 20 / 50, against 0.347 /
  0.256 / 0.174). Target coverage is now 65% / 74% / 83% (was 1.8% for every λ).

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

---

## All notebooks

### B12 (non-math): plot labels named the concrete dataset's target on every dataset
Your template says "Replace (str) with actual data label name", but every dataset still used `'str'`
(concrete's target, compressive strength) in about 70 plot labels across the 8 notebooks, e.g.
`set_ylabel('str')` or `'Actual str'`. Each notebook now has `TARGET_LABEL = '...'` next to
`feature_names`, and every plot label uses it (`f'Actual {TARGET_LABEL}'`). Pile: `'Total'`. The value is
the only per-dataset edit, together with the folder name, the data file names and `feature_names`.
Pile's target column is just `Total` and its meaning isn't recorded, so the label is `'Pile response (Total)'`.

### D4: number of bins follows the test-set size (approved 2026-11-22)
- **Before:** every binned metric used 10 bins on every dataset (CRPS decomposition table, PIT histogram,
  conditional coverage, coverage-by-width, regression ECE). With pile's 24 test points that is 2–3 points per bin.
- **After:** `N_BINS = max(2, min(10, n_test // 10))`, defined once after the test data is loaded and used by all
  of them (`n_bins=N_BINS`, `num_bins=N_BINS`). Pile 2, ewqi 4, noise 6, scour 7, wqi 8, concrete / streamflow 9,
  pile_settlement / sediment 10. Display-only histograms (20 or 30 bins) are unchanged.

### D5: COP burn-in follows the test-set size (approved 2026-11-22, Notebook 6)
- **Before:** `T_burnin=50`. COP's distribution-informed correction (and the window for its step size) starts
  after 50 test points, so on pile (24) and water_quality_ewqi (47) COP was plain online gradient descent.
- **After:** `T_burnin=COP_T_BURNIN`, with `COP_T_BURNIN = min(50, n_test // 2)`. Pile: 12; the 7 datasets with
  at least 100 test points keep 50.

NexCP `decay = 0.99` (Barber et al. 2023) is kept on every dataset (decided 2026-11-22).

