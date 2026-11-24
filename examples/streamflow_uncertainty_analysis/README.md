# Streamflow: how far can the predictions be trusted?

[← main README](../../README.md) · [all datasets](../../README.md#5-the-nine-datasets) · [report (PDF)](../../papers/streamflow/paper.pdf) · [method guides](../../notebooks/)

| | |
|---|---|
| **Data** | 192 training / 96 test rows, 4 features, target `TDS` |
| **Key result** | Conformal intervals are valid in **179 of 180** model × method cases; the sharpest valid method is **J+aB (MAPIE)** (88.5% coverage, 1.32 × std(y) wide) |
| **Watch out** | Raw quantile regression covers only 86.5% (median); native Gaussian intervals: NGBoost 89.6%, PGBM 16.7% |
| **Best point model** | GradientBoosting (5-fold CV RMSE 102.4) |
| **Full report** | [`papers/streamflow/paper.pdf`](../../papers/streamflow/paper.pdf): methods with maths, all tables, appendix |

## Contents

1. [The data](#1-the-data)
2. [What was run](#2-what-was-run)
3. [Results, question by question](#3-results-question-by-question)
4. [All interval methods](#4-all-interval-methods)
5. [Folder map and Kaggle runs](#5-folder-map-and-kaggle-runs)

---

## 1. The data

<p align="center"><img src="../../papers/streamflow/figures/fig_data.png" width="90%" alt="Target distribution and strongest features"></p>

*__Figure 1.__ Left: the target on the training and test sets. Right: the features most correlated with it (Spearman, training set).*

| Statistic | Train | Test |
|---|--:|--:|
| Rows | 192.0 | 96.00 |
| Mean | 259.5 | 233.1 |
| Std | 285.3 | 197.1 |
| Min | 52.00 | 54.00 |
| Median | 180.0 | 180.0 |
| Max | 2,000 | 1,500 |

**Features:** `pH`, `Salinity`, `Turbidity `, `Water Temperature`.

**Reading Figure 1.** The target's skewness is 4.07 (a long right tail: a few large values are the hardest to predict and to cover). The test mean differs from the training mean by -0.09 training standard deviations and the test spread is 0.69× the training spread — the two sets are not identically distributed, which strains methods that assume exchangeability. With 48 training rows per feature, there is enough data per feature for stable tree models.

---

## 2. What was run

The same eight notebooks as every other dataset ([pipeline](../../README.md#4-the-pipeline)); every phase folder has its own README with this dataset's numbers.

| Phase | What it computes | Results | Kaggle run(s) |
|---|---|---|---|
| [NB1](hyperparameter_tuning/README.md) | Hyperparameter tuning (Optuna) | [`hyperparameter_tuning/results/`](hyperparameter_tuning/results/) | [REG22_11_2026_15_16](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-15-16) |
| [NB2](quantile_regression/README.md) | Quantile regression | [`quantile_regression/results/`](quantile_regression/results/) | [REG23_11_2026_01_13](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-13) |
| [NB3](probabilistic_distribution/README.md) | Distributional (NGBoost, PGBM, IBUG, Treeffuser) | [`probabilistic_distribution/results/`](probabilistic_distribution/results/) | [REG23_11_2026_01_14](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-14) |
| [NB4](probabilistic_distribution_card/README.md) | Distributional (CARD diffusion) | [`probabilistic_distribution_card/results/`](probabilistic_distribution_card/results/) | [REG23_11_2026_01_15](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-15) |
| [NB5](conformal_predictions_mapie_puncc/README.md) | Conformal (MAPIE, PUNCC) | [`conformal_predictions_mapie_puncc/results/`](conformal_predictions_mapie_puncc/results/) | [REG23_11_2026_01_16](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-16) |
| [NB6](conformal_predictions_nexcp_adaptivecp_mfcs/README.md) | NexCP and online conformal | [`conformal_predictions_nexcp_adaptivecp_mfcs/results/`](conformal_predictions_nexcp_adaptivecp_mfcs/results/) | [REG23_11_2026_06_59](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-06-59), [REG23_11_2026_11_40](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-11-40), [REG23_11_2026_13_10](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-13-10), [REG23_11_2026_13_35](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-13-35), [REG23_11_2026_14_01](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-01), [REG23_11_2026_14_11](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-11), [REG23_11_2026_14_22](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-22), [REG23_11_2026_14_33](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-33), [REG23_11_2026_06_25](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-06-25) |
| [NB7](conformal_predictions_adaptive_coverage_policies/README.md) | Adaptive coverage policies (ACP) | [`conformal_predictions_adaptive_coverage_policies/results/`](conformal_predictions_adaptive_coverage_policies/results/) | [REG23_11_2026_01_17](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-17) |
| [NB8](hyperspherical_confidence_mapping_hcm/README.md) | Hyperspherical confidence mapping (HCM) | [`hyperspherical_confidence_mapping_hcm/results/`](hyperspherical_confidence_mapping_hcm/results/) | [REG23_11_2026_01_23](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-23) |

---

## 3. Results, question by question

Every interval targets **90%**. With 96 test points a correct method can land anywhere above **0.850** by chance, so that is the validity threshold ([why](../../README.md#73-when-is-a-coverage-good-enough)). Relative width = interval width / std of the test target.

### RQ1 · How accurate are the tuned point predictors?

<p align="center"><img src="../../papers/streamflow/figures/fig_accuracy.png" width="75%" alt="Accuracy"></p>

**How to read it.** Grey bars: 5-fold CV RMSE on the training set (used to choose the model). Blue bars: RMSE on the untouched test set; dark blue = best CV model.

**What it shows.**

- **GradientBoosting** has the lowest 5-fold CV RMSE (102.4); on the test set the lowest RMSE is **GPBoost** (109.2).
- The target's test standard deviation is 197.1, so the best test RMSE explains about 69% of the variance (1 − RMSE²/var); the median model explains 66%.
- CV and test rankings agree only roughly: with 96 test points the test RMSE of one model has a sampling error of several percent.

> **Takeaway RQ1.** The tree ensembles are close to each other in accuracy, so the choice of UQ method matters more for the quality of the intervals than the choice among the boosted models.

### RQ2 · Does raw quantile regression give a valid 90% interval?

**How to read it.** Quantile regression fits one model per level; its 90% interval is the 5% and 95% models' predictions. No calibration step is involved.

**What it shows.**

- The raw 90% interval [q̂₀.₀₅, q̂₀.₉₅] has a median coverage of **86.5%** over 9 models (range 13.5%–94.8%); 5 of 9 are valid (≥ 0.850).
- Quantiles cross on a median **57.3%** of test points before sorting: the 13 separately fitted models disagree on the order of their own quantiles.
- Its median width is 1.24 × std(y), narrower than most conformal intervals; the narrowness is what makes it undercover.
- Best quantile CRPS: CatBoost (48.76).

> **Takeaway RQ2.** Raw quantile regression happens to reach the target on this dataset, but it has no guarantee and its quantiles cross; a conformal method gives the same width with a guarantee.

### RQ3 · How good are the predictive distributions?

<p align="center"><img src="../../papers/streamflow/figures/fig_crps.png" width="75%" alt="CRPS"></p>

**How to read it.** CRPS (log scale, lower is better) scores a whole predictive distribution: it rewards a sharp distribution centred on the truth. Coverage, in the text, checks only the 90% interval.

**What it shows.**

- NGBoost (native Gaussian): coverage **89.6%**, width 1.28 × std(y), PIT diagnosis: well-calibrated (uniform pit).
- PGBM (native Gaussian): coverage **16.7%**, width 0.13 × std(y), PIT diagnosis: under-dispersed (over-confident).
- IBUG on XGBoost: coverage 91.7%, width 2.10 × std(y).
- IBUG on LightGBM: coverage 89.6%, width 1.70 × std(y).
- Treeffuser (median of 5 models): coverage 100.0%, width 14.57 × std(y).
- CARD (median of 9 guides): coverage 78.1%, width 0.99 × std(y).
- Lowest CRPS of all predictive distributions: **CatBoost (Quantile regression)**, 48.76 (target units).

> **Takeaway RQ3.** The native PGBM distributions are over-confident on this dataset: their σ̂ is too small. A good CRPS does not imply a valid interval: CRPS rewards a sharp distribution centred near the truth, while coverage checks the tails.

### RQ4 · Do conformal methods keep their 90% promise?

<p align="center"><img src="../../papers/streamflow/figures/fig_method_league.png" width="85%" alt="Interval methods"></p>

<p align="center"><img src="../../papers/streamflow/figures/fig_coverage_width.png" width="75%" alt="Coverage vs width"></p>

**How to read it.** Top: one row per method; dot = median coverage over the base models it wraps, bar = min–max, green band = valid region; right panel = median relative width. Bottom: every model × method pair; good intervals sit in the green band, far to the left.

**What it shows.**

- Conformal intervals reach the target in **179 of 180** model × method cases (99.4%).
- Misses: NexCP Full (1). Lowest coverage: NexCP Full on XGBoost, 83.3%.
- NexCP Full intervals sometimes reach the edge of their y-grid, so their width is a lower bound; they are not ranked (†).

> **Takeaway RQ4.** Conformal calibration delivers the promised coverage on almost every model, whatever the model's accuracy.

### RQ5 · Which valid method is the sharpest?

<p align="center"><img src="../../papers/streamflow/figures/fig_intervals.png" width="75%" alt="Intervals"></p>

**How to read it.** The CV+ intervals of GradientBoosting on every test point, sorted by the observed value; red crosses are misses. Marginal coverage allows misses to cluster, typically at the extremes.

**What it shows.**

- Recommended: **J+aB (MAPIE)**, median coverage 88.5%, median width 260.4 = **1.32 × std(y)**.
- Next narrowest valid methods: Online J+aB 1.39; MFCS Split 1.46; Online Split 1.46.
- The widest valid method (Treeffuser, 14.57) is 11.0 × wider than the recommended one: validity alone does not pick a method.
- Raw quantile regression is 1.24 × std(y) wide but covers only 86.5%.
- Median width of valid intervals by family: CARD 1.19; Quantile regression 1.52; Online conformal 1.57; COP 1.60; Conformal (MAPIE/PUNCC) 1.62; Weighted conformal (NexCP) 1.72; Distributional 6.56.

> **Takeaway RQ5.** On this dataset the sharpest valid interval comes from **J+aB (MAPIE)**.

### RQ6 · What do the learned-coverage and single-network methods add?

**How to read it.** ACP coverage must be compared with its own learned target 1 − α̃, not with 90%. HCM reports a 95% interval and the correlation between its predicted spread and the actual error.

**What it shows.**

- For 9 of 9 models (CatBoost, GPBoost, GradientBoosting, HistGB, LightGBM, NGBoost, PGBM, TabNet, XGBoost) the ACP policy saturated at α̃ = 0.999: the sets have constant width and the learned level carries no information (realised coverage 59.4%–66.7%).
- HCM (one network, 95% interval): coverage 95.8%, RMSE 192.4, Spearman(|error|, σ̂) = -0.15 — its uncertainty is not informative here.
- Extreme CP (one-sided upper bound): coverage 89.6%–97.9% of the bound y ≤ ŷ + q.

> **Takeaway RQ6.** Learned-level and single-network methods answer different questions from a fixed 90% interval; they are reported for completeness and are not ranked.

---

## 4. All interval methods

Aggregated over the base models each method wraps. *Valid* = share of models with coverage ≥ 0.850. † = some intervals hit the edge of the label grid, so the width is a lower bound (not ranked). **Bold** = recommended.

| Family | Method | Models | Coverage (median) | Coverage range | Width (median) | Relative width | Valid |
|---|---|--:|--:|--:|--:|--:|--:|
| Quantile regression | Quantile regression (raw) | 9 | 86.5% | 13.5%–94.8% | 243.5 | 1.24 | 56% |
| Distributional | PGBM (native) | 1 | 16.7% | 16.7%–16.7% | 25.51 | 0.13 | 0% |
| Distributional | NGBoost (native) | 1 | 89.6% | 89.6%–89.6% | 251.8 | 1.28 | 100% |
| Distributional | IBUG | 2 | 90.6% | 89.6%–91.7% | 374.0 | 1.90 | 100% |
| Distributional | Treeffuser | 5 | 100.0% | 99.0%–100.0% | 2,872 | 14.57 | 100% |
| CARD | CARD | 9 | 78.1% | 52.1%–86.5% | 194.4 | 0.99 | 22% |
| Conformal (MAPIE/PUNCC) | **J+aB (MAPIE)** | 9 | 88.5% | 87.5%–89.6% | 260.4 | 1.32 | 100% |
| Conformal (MAPIE/PUNCC) | CV+ (MAPIE) | 9 | 89.6% | 86.5%–95.8% | 291.4 | 1.48 | 100% |
| Conformal (MAPIE/PUNCC) | CV+ (PUNCC) | 9 | 91.7% | 88.5%–96.9% | 302.1 | 1.53 | 100% |
| Conformal (MAPIE/PUNCC) | CQR (PUNCC) | 9 | 91.7% | 87.5%–96.9% | 322.6 | 1.64 | 100% |
| Conformal (MAPIE/PUNCC) | Split CP (PUNCC) | 9 | 94.8% | 91.7%–96.9% | 389.8 | 1.98 | 100% |
| Conformal (MAPIE/PUNCC) | Split CP (MAPIE) | 9 | 96.9% | 89.6%–96.9% | 483.5 | 2.45 | 100% |
| Weighted conformal (NexCP) | NexCP Full † | 9 | 87.5% | 83.3%–96.9% | 269.2 | 1.37 | 89% |
| Weighted conformal (NexCP) | NexCP J+aB | 9 | 92.7% | 91.7%–95.8% | 317.5 | 1.61 | 100% |
| Weighted conformal (NexCP) | NexCP CV+ | 9 | 94.8% | 91.7%–97.9% | 371.0 | 1.88 | 100% |
| Weighted conformal (NexCP) | NexCP Split | 9 | 94.8% | 86.5%–96.9% | 373.3 | 1.89 | 100% |
| Online conformal | Online J+aB | 9 | 89.6% | 87.5%–91.7% | 274.6 | 1.39 | 100% |
| Online conformal | MFCS Split | 9 | 89.6% | 85.4%–96.9% | 288.6 | 1.46 | 100% |
| Online conformal | Online Split | 9 | 89.6% | 85.4%–96.9% | 288.6 | 1.46 | 100% |
| Online conformal | MFCS Full | 9 | 89.6% | 88.5%–92.7% | 290.5 | 1.47 | 100% |
| Online conformal | ACI | 9 | 89.6% | 87.5%–90.6% | 309.9 | 1.57 | 100% |
| Online conformal | FACI | 9 | 89.6% | 86.5%–90.6% | 317.4 | 1.61 | 100% |
| Online conformal | Online CV+ | 9 | 91.7% | 90.6%–97.9% | 318.6 | 1.62 | 100% |
| Online conformal | SF-OGD | 9 | 89.6% | 89.6%–90.6% | 342.4 | 1.74 | 100% |
| Online conformal | SAOCP | 9 | 90.6% | 90.6%–92.7% | 390.3 | 1.98 | 100% |
| COP | COP | 9 | 89.6% | 88.5%–89.6% | 315.2 | 1.60 | 100% |

<details><summary><b>Tuned models (NB1)</b></summary>

| Model | Pruner | CV RMSE | Test RMSE | Test r |
|---|---|--:|--:|--:|
| GradientBoosting | HyperbandPruner | 102.4 | 115.4 | 0.820 |
| NGBoost | HyperbandPruner | 104.6 | 112.9 | 0.819 |
| RandomForest | SuccessiveHalvingPruner | 105.9 | 118.6 | 0.799 |
| TabNet | NopPruner | 106.9 | 116.8 | 0.867 |
| CatBoost | WilcoxonPruner | 109.1 | 114.2 | 0.815 |
| PGBM | SuccessiveHalvingPruner | 109.5 | 112.9 | 0.823 |
| GPBoost | ThresholdPruner | 110.0 | 109.2 | 0.833 |
| LightGBM | SuccessiveHalvingPruner | 110.1 | 125.3 | 0.781 |
| HistGB | HyperbandPruner | 115.2 | 117.8 | 0.802 |
| XGBoost | ThresholdPruner | 120.7 | 118.0 | 0.802 |

</details>

---

## 5. Folder map and Kaggle runs

```
streamflow_uncertainty_analysis/
├── README.md                 ← you are here
├── data/                     ← train.csv, test.csv (last column = target)
├── hyperparameter_tuning/                            ← NB1
├── quantile_regression/                              ← NB2
├── probabilistic_distribution/                       ← NB3
├── probabilistic_distribution_card/                  ← NB4
├── conformal_predictions_mapie_puncc/                ← NB5
├── conformal_predictions_nexcp_adaptivecp_mfcs/      ← NB6
├── conformal_predictions_adaptive_coverage_policies/ ← NB7
├── hyperspherical_confidence_mapping_hcm/            ← NB8
│   ├── README.md             ← this phase on this dataset
│   ├── <notebook>.ipynb      ← v2.0 code with this dataset's paths (no outputs)
│   └── results/              ← EXECUTED_*.ipynb (every cell output) + Excel files
```

Every run used for the results is linked in the table of [section 2](#2-what-was-run) and in each `results/README.md`. To reproduce, run the eight notebooks in order ([how](../../README.md#11-reproducing-the-results)). Machine-readable numbers: [`results/benchmark_long.csv`](../../results/benchmark_long.csv) (rows with `dataset = streamflow`).
