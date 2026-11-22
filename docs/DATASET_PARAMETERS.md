# Dataset-dependent settings

What has to change, or behaves differently, when the pipeline is moved from one dataset to
another. Section 1 is the evidence from the v1.2 notebooks. Section 2 lists the settings that must
be edited per dataset. Section 3 lists the settings that stay fixed but whose effect depends on
dataset size. Section 4 contains proposals; none of them is applied yet.

Values below were read from the pile notebooks on branch `version2.0`
(`examples/pile_uncertainty_analysis/<phase>/*.ipynb`). "NBn" numbers follow the run order in the
main README.

---

## 1. Evidence from v1.2: what was edited per dataset

The v1.2 notebooks of every dataset were compared with the pile notebooks, after replacing the
dataset folder name and leaving out the hand-pasted `best_scores_autosampler` cell. Only three
things differed:

1. the dataset folder name (`<dataset>_uncertainty_analysis`) in all paths;
2. `feature_names = [...]`;
3. the pasted tuned parameters (now loaded from `best_params.pkl`, fix K2).

Nothing else had been adapted:

- **Plot y-axis label.** The template instruction says "Replace (str) with actual data label
  name", but every dataset still uses `'str'`, which is the concrete dataset's target. It
  appears in NB2, NB5, NB6 and NB7, for example `set_ylabel('str')` 13× in NB5 and 11× in NB6.
- **HCM `feature_names`.** The HCM notebooks were never customised: every dataset, pile
  included, has `['A', 'B', 'C', 'D', 'O']`, the noise dataset's columns. In HCM the names were
  used only by the synthetic-data fallback, which v2.0 removes.

## 2. Settings that must be edited per dataset

| Setting | Where | Pile value | Notes |
|---|---|---|---|
| Dataset folder name | every path in all 8 notebooks | `pile_uncertainty_analysis` | Find-and-replace, as the template instructs. |
| Kaggle input file names | `FIX K1` cell (cell 0) of each notebook | `d08_a.csv` (train), `d08_b.csv` (test) | Names of this dataset's files inside `daneshselwal/reguq-uq-data`. |
| `feature_names` | cell 4 or 5 (NB1–NB7); also used by NB2's per-feature plots and NB1's SHAP/LIME | `['F1','F2','Zv1','Zv2','phi','Th','L']` | Must match the CSV header, minus the target. |
| Target label in plots | `set_ylabel('str')` / `ylabel('str')` | `'str'` (wrong for pile, whose target is `Total`) | Cosmetic, but it affects every figure. |
| Tuned parameters | `best_params.pkl` | produced by NB1 | Loaded automatically (K2). Downstream jobs need the matching tuning job as input. |

Targets by dataset (last CSV column): concrete `str`, noise `Leq`, pile `Total`, pile_settlement
`S_exp`, scour `Scour`, sediment_load `St`, streamflow `TDS`, water_quality_ewqi
`Pre monsoon EWQI`, water_quality_wqi `WQI`.

## 3. Settings that are fixed but whose effect depends on dataset size

Dataset sizes: train/test rows range from 47/47 (water_quality_ewqi) and 81/24 (pile) up to
1095/365 (sediment_load).

### Splits and calibration-set sizes

| Setting | Where | Value | Pile (81 train / 24 test) | Large data (sediment 1095 / 365) |
|---|---|---|---|---|
| Tuning CV folds | NB1 `KFold(n_splits=5)` | 5 | ~16 rows per validation fold | ~219 rows per fold |
| MAPIE split conformal calibration | NB5 `ShuffleSplit(n_splits=1, test_size=0.5)` | 50% | 40 calibration rows | 547 rows |
| MAPIE CV+ folds | NB5 `'cv': 10` | 10 | ~8 rows per fold | ~110 rows per fold |
| PUNCC SplitCP / CQR split | NB5 `fit_ratio=0.5`, `test_size=0.5` | 50% | 40 fitting rows per model | — |
| PUNCC CV+ | NB5 `K=5` | 5 | — | — |
| MAPIE analysis grid | NB5 `KFold(5)`, `Subsample(n_resamplings=20/50)` | — | — | Bootstraps scale cost linearly. |
| Extreme CP calibration | NB6 `calib_size=0.3` in calls (default 0.2) | 30% | 25 calibration scores | 329 scores |
| AlphaNet calibration (NB6) | `calib_fraction=0.5` | 50% | 41 calibration points | 548 |
| ACP calibration (NB7) | `train_test_split(test_size=0.35)` | 35% | 29 calibration points | 384 |
| HCM validation | NB8 `test_size=0.20` | 20% | 17 rows; temperature T and early stopping both use these 17 rows | 219 rows |
| TabNet early stopping | NB2 `test_size=0.2` | 20% | 17 rows | 219 rows |
| IBUG validation | NB3 `test_size=0.3` | 30% | 25 rows | 329 rows |

Conformal guarantees hold at any size, but small calibration sets make the intervals noisy.
With n calibration points the attainable coverage levels are steps of 1/(n+1). At α = 0.1 with
n < 9, the conformal quantile is +∞; v2.0 caps such intervals and flags them (fix N2).

### Method-specific settings

| Setting | Where | Value | Why it depends on the data |
|---|---|---|---|
| NexCP decay | NB6 `decay=0.99` | 0.99 | Weight of the k-th most recent training row is 0.99^k, so the effective window is ~100 rows. Pile (81 rows): weights fall only to 0.45, almost uniform. Sediment (1095 rows): the oldest rows get weight ~10⁻⁵, so the method effectively uses the last few hundred rows. The rows' order must be meaningful (time order) for the weights to make sense. |
| NexCP / online J+aB bootstraps | NB6 `n_bootstrap=50` (NexCP), `30` (online) | 50 / 30 | Each training row needs to be out-of-bag in some bootstrap; about 37% of bootstraps leave a given row out, so 30–50 is enough for any size. Cost grows linearly with the count. |
| NexCP-Full grid | NB6, approved fix N3 | 50 candidate values | Cost = n_test × 50 refits per model. Pile: 1,200 refits; sediment: 18,250 refits. Slow models (CatBoost, TabNet) dominate the run time. |
| COP burn-in | NB6 `T_burnin=50` | 50 | The distribution-informed ("optimistic") correction only starts after 50 test points. With pile's 24 test points it never activates, so COP reduces to plain online gradient descent. |
| COP learning rate | NB6 `lr=0.05` | 0.05 | v2.0 multiplies it by the range of recent errors (official proportional step, fix N8), so it no longer depends on the target's units. |
| Extreme CP thresholds | NB6, fix N9 | levels 0.80–0.99, at least 10 exceedances | With 25 calibration scores, all threshold levels are capped at 1 − 10/25 = 0.60, so a single threshold is used. Larger calibration sets use the full range. |
| Online windows | NB6 `window_min=50`, `window_max=500`, `window_grid=(25,…,400)` | — | Used only by the v1.2 "SAOCP" and "FACI" stand-ins, which v2.0 replaces (fixes N5, N6). |
| CARD batch / epochs | NB4 `batch_size=32`, `epochs=5000` (official settings) | — | Number of gradient updates = ceil(n/32) × 5000. Pile: 15,000 updates. Sediment: 175,000 updates, roughly 12× the run time. |
| CARD samples | NB4 `n_samples=1000` | 1000 | Memory and time scale with n_test × 1000. |
| TabNet batch | NB2/NB4 `batch_size=1024`, `virtual_batch_size=128`, `max_epochs=200`, `patience=30` | — | With fewer than 1024 rows, each epoch is a single batch. On pile, one batch with ghost batch norm over ~65 rows. |
| AlphaNet epochs / runs | NB6 `num_epochs=50`, `num_runs=4`; NB7 `epochs=50`, `num_runs=3`, `lambdas=(10,20,50)` | — | The size term scales with the error sum. v2.0 rescales the network input (fix A2), so the same settings work for any target scale. |
| HCM training | NB8 `batch_size=64`, `epochs=200`, `patience=40` | — | Pile (64 training rows): 1 batch per epoch. |
| Number of bins | `n_bins=10` (CRPS decomposition table, ECE_reg), `num_bins=10` (conditional coverage) | 10 | With pile's 24 test points, each bin holds 2–3 points, so per-bin numbers are very noisy. With 365 points, each bin holds ~36. |
| Optuna trials | NB1 `n_trials=50` per model × pruner | 50 | Independent of size; cost per trial grows with n. |
| Quantile levels | NB2 / NB3 | 13 levels 0.05–0.95 | Fixed. The extreme levels (0.05, 0.95) rest on very few points when n is small. |

## 4. Proposals (decided 2026-11-22: 1–3 done by the notebook generator and FIX B12; 4 → FIX D4, 5 → FIX D5 applied; 6 rejected, decay stays 0.99)

Changes to bookkeeping only, with no effect on any method's mathematics:

1. **One `DATASET` configuration cell** at the top of each notebook, holding the dataset folder
   name and the Kaggle train/test file names. All paths would be built from it, replacing
   find-and-replace.
2. **`feature_names` from the CSV header**: `feature_names = list(train_data.columns[:-1])`.
3. **Target label from the CSV header**: `TARGET_NAME = train_data.columns[-1]`, used in place of
   the literal `'str'` in every `set_ylabel` / `ylabel` call.

Changes that would affect results, and therefore need an explicit decision:

4. Size-aware number of bins, for example `n_bins = max(2, min(10, n_test // 10))`.
5. Whether COP's `T_burnin` should scale with the test-set length when n_test < 50.
6. Whether NexCP's `decay` should be expressed as an effective window (for example
   `decay = 1 - 1/window`) so that it means the same on every dataset.

The current notebooks use the fixed values in Section 3. Proposals 4–6 would be applied only
after they are approved, and would be logged in `CHANGELOG_FIXES.md`.
