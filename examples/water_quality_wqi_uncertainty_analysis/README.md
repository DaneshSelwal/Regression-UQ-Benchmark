# Water quality (WQI): how far can the predictions be trusted?

[← main README](../../README.md) · [all datasets](../../README.md#5-the-nine-datasets) · [report (PDF)](../../papers/water_quality_wqi/paper.pdf) · [method guides](../../notebooks/)

| | |
|---|---|
| **Data** | 190 training / 82 test rows, 151 features, target `WQI` |
| **Key result** | Conformal intervals are valid in **143 of 180** model × method cases; the sharpest valid method is **J+aB (MAPIE)** (89.0% coverage, 1.18 × std(y) wide) |
| **Watch out** | Raw quantile regression covers only 79.3% (median); native Gaussian intervals: NGBoost 1.2%, PGBM 14.6% |
| **Best point model** | XGBoost (5-fold CV RMSE 22.55) |
| **Full report** | [`papers/water_quality_wqi/paper.pdf`](../../papers/water_quality_wqi/paper.pdf): methods with maths, all tables, appendix |

## Contents

1. [The data](#1-the-data)
2. [What was run](#2-what-was-run)
3. [Results, question by question](#3-results-question-by-question)
4. [All interval methods](#4-all-interval-methods)
5. [Folder map and Kaggle runs](#5-folder-map-and-kaggle-runs)

---

## 1. The data

<p align="center"><img src="../../papers/water_quality_wqi/figures/fig_data.png" width="90%" alt="Target distribution and strongest features"></p>

*__Figure 1.__ Left: the target on the training and test sets. Right: the features most correlated with it (Spearman, training set).*

| Statistic | Train | Test |
|---|--:|--:|
| Rows | 190.0 | 82.00 |
| Mean | 174.0 | 175.4 |
| Std | 82.17 | 72.93 |
| Min | 55.00 | 81.00 |
| Median | 149.5 | 164.0 |
| Max | 467.0 | 516.0 |

**Features:** `R1` … `R151` (151 columns).

**Reading Figure 1.** The target's skewness is 1.26 (a long right tail: a few large values are the hardest to predict and to cover). The test mean differs from the training mean by +0.02 training standard deviations and the test spread is 0.89× the training spread — the two sets look alike. With 1 training rows per feature, the problem is data-poor.

---

## 2. What was run

The same eight notebooks as every other dataset ([pipeline](../../README.md#4-the-pipeline)); every phase folder has its own README with this dataset's numbers.

| Phase | What it computes | Results | Kaggle run(s) |
|---|---|---|---|
| [NB1](hyperparameter_tuning/README.md) | Hyperparameter tuning (Optuna) | [`hyperparameter_tuning/results/`](hyperparameter_tuning/results/) | [REG22_11_2026_19_53](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-19-53) |
| [NB2](quantile_regression/README.md) | Quantile regression | [`quantile_regression/results/`](quantile_regression/results/) | [REG23_11_2026_08_17](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-08-17) |
| [NB3](probabilistic_distribution/README.md) | Distributional (NGBoost, PGBM, IBUG, Treeffuser) | [`probabilistic_distribution/results/`](probabilistic_distribution/results/) | [REG23_11_2026_08_48](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-08-48) |
| [NB4](probabilistic_distribution_card/README.md) | Distributional (CARD diffusion) | [`probabilistic_distribution_card/results/`](probabilistic_distribution_card/results/) | [REG23_11_2026_08_49](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-08-49) |
| [NB5](conformal_predictions_mapie_puncc/README.md) | Conformal (MAPIE, PUNCC) | [`conformal_predictions_mapie_puncc/results/`](conformal_predictions_mapie_puncc/results/) | [REG23_11_2026_08_50](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-08-50) |
| [NB6](conformal_predictions_nexcp_adaptivecp_mfcs/README.md) | NexCP and online conformal | [`conformal_predictions_nexcp_adaptivecp_mfcs/results/`](conformal_predictions_nexcp_adaptivecp_mfcs/results/) | [REG23_11_2026_08_59](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-08-59), [REG23_11_2026_09_10](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-09-10), [REG23_11_2026_10_06](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-10-06), [REG23_11_2026_10_09](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-10-09), [REG23_11_2026_10_18](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-10-18), [REG23_11_2026_10_23](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-10-23), [REG23_11_2026_11_07](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-11-07), [REG23_11_2026_11_12](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-11-12), [REG23_11_2026_11_21](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-11-21) |
| [NB7](conformal_predictions_adaptive_coverage_policies/README.md) | Adaptive coverage policies (ACP) | [`conformal_predictions_adaptive_coverage_policies/results/`](conformal_predictions_adaptive_coverage_policies/results/) | [REG23_11_2026_08_51](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-08-51) |
| [NB8](hyperspherical_confidence_mapping_hcm/README.md) | Hyperspherical confidence mapping (HCM) | [`hyperspherical_confidence_mapping_hcm/results/`](hyperspherical_confidence_mapping_hcm/results/) | [REG23_11_2026_08_58](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-08-58) |

---

## 3. Results, question by question

Every interval targets **90%**. With 82 test points a correct method can land anywhere above **0.846** by chance, so that is the validity threshold ([why](../../README.md#73-when-is-a-coverage-good-enough)). Relative width = interval width / std of the test target.

### RQ1 · How accurate are the tuned point predictors?

<p align="center"><img src="../../papers/water_quality_wqi/figures/fig_accuracy.png" width="75%" alt="Accuracy"></p>

**How to read it.** Grey bars: 5-fold CV RMSE on the training set (used to choose the model). Blue bars: RMSE on the untouched test set; dark blue = best CV model.

**What it shows.**

- **XGBoost** has the lowest 5-fold CV RMSE (22.55); on the test set the lowest RMSE is **CatBoost** (27.16).
- The target's test standard deviation is 72.93, so the best test RMSE explains about 86% of the variance (1 − RMSE²/var); the median model explains 82%.
- CV and test rankings agree only roughly: with 82 test points the test RMSE of one model has a sampling error of several percent.

> **Takeaway RQ1.** The tree ensembles are close to each other in accuracy, so the choice of UQ method matters more for the quality of the intervals than the choice among the boosted models.

### RQ2 · Does raw quantile regression give a valid 90% interval?

**How to read it.** Quantile regression fits one model per level; its 90% interval is the 5% and 95% models' predictions. No calibration step is involved.

**What it shows.**

- The raw 90% interval [q̂₀.₀₅, q̂₀.₉₅] has a median coverage of **79.3%** over 9 models (range 2.4%–92.7%); 3 of 9 are valid (≥ 0.846).
- Quantiles cross on a median **100.0%** of test points before sorting: the 13 separately fitted models disagree on the order of their own quantiles.
- Its median width is 1.23 × std(y), narrower than most conformal intervals; the narrowness is what makes it undercover.
- Best quantile CRPS: TabNet (13.50).

> **Takeaway RQ2.** Raw quantile regression is not a reliable 90% interval here; it needs conformal calibration (CQR) or replacement by a conformal method.

### RQ3 · How good are the predictive distributions?

<p align="center"><img src="../../papers/water_quality_wqi/figures/fig_crps.png" width="75%" alt="CRPS"></p>

**How to read it.** CRPS (log scale, lower is better) scores a whole predictive distribution: it rewards a sharp distribution centred on the truth. Coverage, in the text, checks only the 90% interval.

**What it shows.**

- NGBoost (native Gaussian): coverage **1.2%**, width 0.01 × std(y), PIT diagnosis: under-dispersed (over-confident).
- PGBM (native Gaussian): coverage **14.6%**, width 0.19 × std(y), PIT diagnosis: under-dispersed (over-confident).
- IBUG on XGBoost: coverage 68.3%, width 0.75 × std(y).
- IBUG on LightGBM: coverage 91.5%, width 1.52 × std(y).
- Treeffuser (median of 5 models): coverage 100.0%, width 4.91 × std(y).
- CARD (median of 9 guides): coverage 25.6%, width 0.18 × std(y).
- Lowest CRPS of all predictive distributions: **TabNet (CARD)**, 12.78 (target units).

> **Takeaway RQ3.** The native NGBoost and PGBM distributions are over-confident on this dataset: their σ̂ is too small. A good CRPS does not imply a valid interval: CRPS rewards a sharp distribution centred near the truth, while coverage checks the tails.

### RQ4 · Do conformal methods keep their 90% promise?

<p align="center"><img src="../../papers/water_quality_wqi/figures/fig_method_league.png" width="85%" alt="Interval methods"></p>

<p align="center"><img src="../../papers/water_quality_wqi/figures/fig_coverage_width.png" width="75%" alt="Coverage vs width"></p>

**How to read it.** Top: one row per method; dot = median coverage over the base models it wraps, bar = min–max, green band = valid region; right panel = median relative width. Bottom: every model × method pair; good intervals sit in the green band, far to the left.

**What it shows.**

- Conformal intervals reach the target in **143 of 180** model × method cases (79.4%).
- Misses: MFCS Split (6), Online Split (6), NexCP Split (5), CV+ (PUNCC) (4), NexCP Full (3), NexCP J+aB (3), Split CP (MAPIE) (3), CV+ (MAPIE) (2), ACI (1), COP (1), CQR (PUNCC) (1), FACI (1), J+aB (MAPIE) (1). Lowest coverage: MFCS Split on XGBoost, 72.0%.
- NexCP Full intervals sometimes reach the edge of their y-grid, so their width is a lower bound; they are not ranked (†).

> **Takeaway RQ4.** Conformal methods are valid on most, but not all, models here; the misses show where the exchangeability assumption or a small calibration set bites.

### RQ5 · Which valid method is the sharpest?

<p align="center"><img src="../../papers/water_quality_wqi/figures/fig_intervals.png" width="75%" alt="Intervals"></p>

**How to read it.** The CV+ intervals of XGBoost on every test point, sorted by the observed value; red crosses are misses. Marginal coverage allows misses to cluster, typically at the extremes.

**What it shows.**

- Recommended: **J+aB (MAPIE)**, median coverage 89.0%, median width 86.39 = **1.18 × std(y)**.
- Next narrowest valid methods: MFCS Full 1.22; NexCP CV+ 1.24; Online J+aB 1.26.
- The widest valid method (Treeffuser, 4.91) is 4.1 × wider than the recommended one: validity alone does not pick a method.
- Raw quantile regression is 1.23 × std(y) wide but covers only 79.3%.
- Median width of valid intervals by family: Weighted conformal (NexCP) 1.24; COP 1.34; Online conformal 1.34; Conformal (MAPIE/PUNCC) 1.37; Quantile regression 1.44; Distributional 4.57.

> **Takeaway RQ5.** On this dataset the sharpest valid interval comes from **J+aB (MAPIE)**.

### RQ6 · What do the learned-coverage and single-network methods add?

**How to read it.** ACP coverage must be compared with its own learned target 1 − α̃, not with 90%. HCM reports a 95% interval and the correlation between its predicted spread and the actual error.

**What it shows.**

- ACP learns its own level: mean learned target 1 − α̃ = 22.8%; realised coverage 46.3%–65.9% across 9 models.
- HCM (one network, 95% interval): coverage 81.7%, RMSE 39.88, Spearman(|error|, σ̂) = 0.30 — its uncertainty tracks the error.
- Extreme CP (one-sided upper bound): coverage 92.7%–98.8% of the bound y ≤ ŷ + q.

> **Takeaway RQ6.** Learned-level and single-network methods answer different questions from a fixed 90% interval; they are reported for completeness and are not ranked.

---

## 4. All interval methods

Aggregated over the base models each method wraps. *Valid* = share of models with coverage ≥ 0.846. † = some intervals hit the edge of the label grid, so the width is a lower bound (not ranked). **Bold** = recommended.

| Family | Method | Models | Coverage (median) | Coverage range | Width (median) | Relative width | Valid |
|---|---|--:|--:|--:|--:|--:|--:|
| Quantile regression | Quantile regression (raw) | 9 | 79.3% | 2.4%–92.7% | 89.88 | 1.23 | 33% |
| Distributional | NGBoost (native) | 1 | 1.2% | 1.2%–1.2% | 0.719 | 0.01 | 0% |
| Distributional | PGBM (native) | 1 | 14.6% | 14.6%–14.6% | 14.00 | 0.19 | 0% |
| Distributional | IBUG | 2 | 79.9% | 68.3%–91.5% | 82.83 | 1.14 | 50% |
| Distributional | Treeffuser | 5 | 100.0% | 93.9%–100.0% | 358.3 | 4.91 | 100% |
| CARD | CARD | 9 | 25.6% | 18.3%–34.1% | 13.27 | 0.18 | 0% |
| Conformal (MAPIE/PUNCC) | CV+ (PUNCC) | 9 | 86.6% | 80.5%–98.8% | 83.87 | 1.15 | 56% |
| Conformal (MAPIE/PUNCC) | **J+aB (MAPIE)** | 9 | 89.0% | 81.7%–97.6% | 86.39 | 1.18 | 89% |
| Conformal (MAPIE/PUNCC) | Split CP (MAPIE) | 9 | 86.6% | 78.0%–92.7% | 90.94 | 1.25 | 67% |
| Conformal (MAPIE/PUNCC) | CV+ (MAPIE) | 9 | 89.0% | 82.9%–100.0% | 96.18 | 1.32 | 78% |
| Conformal (MAPIE/PUNCC) | Split CP (PUNCC) | 9 | 93.9% | 90.2%–96.3% | 122.0 | 1.67 | 100% |
| Conformal (MAPIE/PUNCC) | CQR (PUNCC) | 9 | 91.5% | 82.9%–98.8% | 131.5 | 1.80 | 89% |
| Weighted conformal (NexCP) | NexCP J+aB | 9 | 86.6% | 84.1%–93.9% | 83.61 | 1.15 | 67% |
| Weighted conformal (NexCP) | NexCP Split | 9 | 84.1% | 72.0%–93.9% | 89.50 | 1.23 | 44% |
| Weighted conformal (NexCP) | NexCP CV+ | 9 | 86.6% | 85.4%–98.8% | 90.60 | 1.24 | 100% |
| Weighted conformal (NexCP) | NexCP Full † | 9 | 86.6% | 79.3%–98.8% | 107.0 | 1.47 | 67% |
| Online conformal | MFCS Split | 9 | 82.9% | 72.0%–95.1% | 85.75 | 1.18 | 33% |
| Online conformal | Online Split | 9 | 82.9% | 72.0%–95.1% | 85.75 | 1.18 | 33% |
| Online conformal | MFCS Full | 9 | 91.5% | 87.8%–93.9% | 89.11 | 1.22 | 100% |
| Online conformal | Online J+aB | 9 | 89.0% | 89.0%–95.1% | 92.02 | 1.26 | 100% |
| Online conformal | ACI | 9 | 86.6% | 80.5%–93.9% | 92.18 | 1.26 | 89% |
| Online conformal | Online CV+ | 9 | 90.2% | 86.6%–98.8% | 95.52 | 1.31 | 100% |
| Online conformal | FACI | 9 | 86.6% | 84.1%–91.5% | 96.42 | 1.32 | 89% |
| Online conformal | SAOCP | 9 | 91.5% | 89.0%–92.7% | 112.6 | 1.54 | 100% |
| Online conformal | SF-OGD | 9 | 89.0% | 89.0%–90.2% | 121.0 | 1.66 | 100% |
| COP | COP | 9 | 87.8% | 82.9%–93.9% | 97.77 | 1.34 | 89% |

<details><summary><b>Tuned models (NB1)</b></summary>

| Model | Pruner | CV RMSE | Test RMSE | Test r |
|---|---|--:|--:|--:|
| XGBoost | WilcoxonPruner | 22.55 | 28.74 | 0.946 |
| PGBM | PercentilePruner | 22.73 | 29.38 | 0.939 |
| LightGBM | HyperbandPruner | 22.80 | 31.90 | 0.925 |
| GPBoost | MedianPruner | 22.97 | 34.91 | 0.911 |
| HistGB | SuccessiveHalvingPruner | 23.54 | 34.61 | 0.915 |
| NGBoost | WilcoxonPruner | 23.61 | 29.19 | 0.946 |
| GradientBoosting | HyperbandPruner | 24.30 | 31.41 | 0.931 |
| CatBoost | SuccessiveHalvingPruner | 26.32 | 27.16 | 0.943 |
| RandomForest | WilcoxonPruner | 29.13 | 31.83 | 0.938 |
| TabNet | HyperbandPruner | 30.88 | 30.87 | 0.930 |

</details>

---

## 5. Folder map and Kaggle runs

```
water_quality_wqi_uncertainty_analysis/
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

Every run used for the results is linked in the table of [section 2](#2-what-was-run) and in each `results/README.md`. To reproduce, run the eight notebooks in order ([how](../../README.md#11-reproducing-the-results)). Machine-readable numbers: [`results/benchmark_long.csv`](../../results/benchmark_long.csv) (rows with `dataset = water_quality_wqi`).
