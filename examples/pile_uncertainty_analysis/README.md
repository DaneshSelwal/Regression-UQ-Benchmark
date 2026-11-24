# Pile: how far can the predictions be trusted?

[← main README](../../README.md) · [all datasets](../../README.md#5-the-nine-datasets) · [report (PDF)](../../papers/pile/paper.pdf) · [method guides](../../notebooks/)

| | |
|---|---|
| **Data** | 81 training / 24 test rows, 7 features, target `Total` |
| **Key result** | Conformal intervals are valid in **175 of 180** model × method cases; the sharpest valid method is **J+aB (MAPIE)** (95.8% coverage, 1.60 × std(y) wide) |
| **Watch out** | Raw quantile regression covers only 58.3% (median); native Gaussian intervals: NGBoost 0.0%, PGBM 41.7% |
| **Best point model** | LightGBM (5-fold CV RMSE 0.567) |
| **Full report** | [`papers/pile/paper.pdf`](../../papers/pile/paper.pdf): methods with maths, all tables, appendix |

## Contents

1. [The data](#1-the-data)
2. [What was run](#2-what-was-run)
3. [Results, question by question](#3-results-question-by-question)
4. [All interval methods](#4-all-interval-methods)
5. [Folder map and Kaggle runs](#5-folder-map-and-kaggle-runs)

---

## 1. The data

<p align="center"><img src="../../papers/pile/figures/fig_data.png" width="90%" alt="Target distribution and strongest features"></p>

*__Figure 1.__ Left: the target on the training and test sets. Right: the features most correlated with it (Spearman, training set).*

| Statistic | Train | Test |
|---|--:|--:|
| Rows | 81.00 | 24.00 |
| Mean | 5.89 | 5.87 |
| Std | 1.40 | 1.33 |
| Min | 2.72 | 3.70 |
| Median | 5.78 | 5.78 |
| Max | 8.47 | 8.21 |

**Features:** `F1`, `F2`, `Zv1`, `Zv2`, `phi`, `Th`, `L`.

**Reading Figure 1.** The target's skewness is -0.16. The test mean differs from the training mean by -0.02 training standard deviations and the test spread is 0.95× the training spread — the two sets look alike. With 12 training rows per feature, the problem is data-poor.

---

## 2. What was run

The same eight notebooks as every other dataset ([pipeline](../../README.md#4-the-pipeline)); every phase folder has its own README with this dataset's numbers.

| Phase | What it computes | Results | Kaggle run(s) |
|---|---|---|---|
| [NB1](hyperparameter_tuning/README.md) | Hyperparameter tuning (Optuna) | [`hyperparameter_tuning/results/`](hyperparameter_tuning/results/) | [REG22_11_2026_01_58](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-01-58) |
| [NB2](quantile_regression/README.md) | Quantile regression | [`quantile_regression/results/`](quantile_regression/results/) | [REG22_11_2026_13_18](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-13-18) |
| [NB3](probabilistic_distribution/README.md) | Distributional (NGBoost, PGBM, IBUG, Treeffuser) | [`probabilistic_distribution/results/`](probabilistic_distribution/results/) | [REG22_11_2026_13_25](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-13-25) |
| [NB4](probabilistic_distribution_card/README.md) | Distributional (CARD diffusion) | [`probabilistic_distribution_card/results/`](probabilistic_distribution_card/results/) | [REG22_11_2026_13_26](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-13-26) |
| [NB5](conformal_predictions_mapie_puncc/README.md) | Conformal (MAPIE, PUNCC) | [`conformal_predictions_mapie_puncc/results/`](conformal_predictions_mapie_puncc/results/) | [REG22_11_2026_13_33](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-13-33) |
| [NB6](conformal_predictions_nexcp_adaptivecp_mfcs/README.md) | NexCP and online conformal | [`conformal_predictions_nexcp_adaptivecp_mfcs/results/`](conformal_predictions_nexcp_adaptivecp_mfcs/results/) | [REG22_11_2026_13_17](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-13-17) |
| [NB7](conformal_predictions_adaptive_coverage_policies/README.md) | Adaptive coverage policies (ACP) | [`conformal_predictions_adaptive_coverage_policies/results/`](conformal_predictions_adaptive_coverage_policies/results/) | [REG22_11_2026_13_34](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-13-34) |
| [NB8](hyperspherical_confidence_mapping_hcm/README.md) | Hyperspherical confidence mapping (HCM) | [`hyperspherical_confidence_mapping_hcm/results/`](hyperspherical_confidence_mapping_hcm/results/) | [REG22_11_2026_14_03](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-14-03) |

---

## 3. Results, question by question

Every interval targets **90%**. With 24 test points a correct method can land anywhere above **0.799** by chance, so that is the validity threshold ([why](../../README.md#73-when-is-a-coverage-good-enough)). Relative width = interval width / std of the test target.

### RQ1 · How accurate are the tuned point predictors?

<p align="center"><img src="../../papers/pile/figures/fig_accuracy.png" width="75%" alt="Accuracy"></p>

**How to read it.** Grey bars: 5-fold CV RMSE on the training set (used to choose the model). Blue bars: RMSE on the untouched test set; dark blue = best CV model.

**What it shows.**

- **LightGBM** has the lowest 5-fold CV RMSE (0.567); on the test set the lowest RMSE is **RandomForest** (0.439).
- The target's test standard deviation is 1.33, so the best test RMSE explains about 89% of the variance (1 − RMSE²/var); the median model explains 73%.
- CV and test rankings agree only roughly: with 24 test points the test RMSE of one model has a sampling error of several percent.
- TabNet, the only neural base model, is among the least accurate, as is common for small tabular data.

> **Takeaway RQ1.** The tree ensembles are close to each other in accuracy, so the choice of UQ method matters more for the quality of the intervals than the choice among the boosted models.

### RQ2 · Does raw quantile regression give a valid 90% interval?

**How to read it.** Quantile regression fits one model per level; its 90% interval is the 5% and 95% models' predictions. No calibration step is involved.

**What it shows.**

- The raw 90% interval [q̂₀.₀₅, q̂₀.₉₅] has a median coverage of **58.3%** over 9 models (range 0.0%–91.7%); 2 of 9 are valid (≥ 0.799).
- Quantiles cross on a median **100.0%** of test points before sorting: the 13 separately fitted models disagree on the order of their own quantiles.
- Its median width is 0.74 × std(y), narrower than most conformal intervals; the narrowness is what makes it undercover.
- Best quantile CRPS: CatBoost (0.255).

> **Takeaway RQ2.** Raw quantile regression is not a reliable 90% interval here; it needs conformal calibration (CQR) or replacement by a conformal method.

### RQ3 · How good are the predictive distributions?

<p align="center"><img src="../../papers/pile/figures/fig_crps.png" width="75%" alt="CRPS"></p>

**How to read it.** CRPS (log scale, lower is better) scores a whole predictive distribution: it rewards a sharp distribution centred on the truth. Coverage, in the text, checks only the 90% interval.

**What it shows.**

- NGBoost (native Gaussian): coverage **0.0%**, width 0.00 × std(y), PIT diagnosis: under-dispersed (over-confident).
- PGBM (native Gaussian): coverage **41.7%**, width 0.23 × std(y), PIT diagnosis: under-dispersed (over-confident).
- IBUG on XGBoost: coverage 100.0%, width 2.11 × std(y).
- IBUG on LightGBM: coverage 100.0%, width 1.93 × std(y).
- Treeffuser (median of 5 models): coverage 100.0%, width 6.13 × std(y).
- CARD (median of 9 guides): coverage 20.8%, width 0.19 × std(y).
- Lowest CRPS of all predictive distributions: **CatBoost (Quantile regression)**, 0.255 (target units).

> **Takeaway RQ3.** The native NGBoost and PGBM distributions are over-confident on this dataset: their σ̂ is too small. A good CRPS does not imply a valid interval: CRPS rewards a sharp distribution centred near the truth, while coverage checks the tails.

### RQ4 · Do conformal methods keep their 90% promise?

<p align="center"><img src="../../papers/pile/figures/fig_method_league.png" width="85%" alt="Interval methods"></p>

<p align="center"><img src="../../papers/pile/figures/fig_coverage_width.png" width="75%" alt="Coverage vs width"></p>

**How to read it.** Top: one row per method; dot = median coverage over the base models it wraps, bar = min–max, green band = valid region; right panel = median relative width. Bottom: every model × method pair; good intervals sit in the green band, far to the left.

**What it shows.**

- Conformal intervals reach the target in **175 of 180** model × method cases (97.2%).
- Misses: MFCS Split (1), NexCP Full (1), Online Split (1), SAOCP (1), Split CP (MAPIE) (1). Lowest coverage: NexCP Full on GPBoost, 75.0%.
- NexCP Full intervals sometimes reach the edge of their y-grid, so their width is a lower bound; they are not ranked (†).

> **Takeaway RQ4.** Conformal calibration delivers the promised coverage on almost every model, whatever the model's accuracy.

### RQ5 · Which valid method is the sharpest?

<p align="center"><img src="../../papers/pile/figures/fig_intervals.png" width="75%" alt="Intervals"></p>

**How to read it.** The CV+ intervals of LightGBM on every test point, sorted by the observed value; red crosses are misses. Marginal coverage allows misses to cluster, typically at the extremes.

**What it shows.**

- Recommended: **J+aB (MAPIE)**, median coverage 95.8%, median width 2.12 = **1.60 × std(y)**.
- Next narrowest valid methods: COP 1.67; ACI 1.68; FACI 1.69.
- The widest valid method (Treeffuser, 6.13) is 3.8 × wider than the recommended one: validity alone does not pick a method.
- Raw quantile regression is 0.74 × std(y) wide but covers only 58.3%.
- Median width of valid intervals by family: COP 1.67; Online conformal 1.79; Weighted conformal (NexCP) 1.86; Conformal (MAPIE/PUNCC) 1.96; Quantile regression 3.11; Distributional 5.27.

> **Takeaway RQ5.** On this dataset the sharpest valid interval comes from **J+aB (MAPIE)**.

### RQ6 · What do the learned-coverage and single-network methods add?

**How to read it.** ACP coverage must be compared with its own learned target 1 − α̃, not with 90%. HCM reports a 95% interval and the correlation between its predicted spread and the actual error.

**What it shows.**

- ACP learns its own level: mean learned target 1 − α̃ = 82.1%; realised coverage 100.0%–100.0% across 9 models.
- HCM (one network, 95% interval): coverage 100.0%, RMSE 1.30, Spearman(|error|, σ̂) = -0.00 — its uncertainty is not informative here.
- Extreme CP (one-sided upper bound): coverage 66.7%–87.5% of the bound y ≤ ŷ + q.

> **Takeaway RQ6.** Learned-level and single-network methods answer different questions from a fixed 90% interval; they are reported for completeness and are not ranked.

---

## 4. All interval methods

Aggregated over the base models each method wraps. *Valid* = share of models with coverage ≥ 0.799. † = some intervals hit the edge of the label grid, so the width is a lower bound (not ranked). **Bold** = recommended.

| Family | Method | Models | Coverage (median) | Coverage range | Width (median) | Relative width | Valid |
|---|---|--:|--:|--:|--:|--:|--:|
| Quantile regression | Quantile regression (raw) | 9 | 58.3% | 0.0%–91.7% | 0.980 | 0.74 | 22% |
| Distributional | NGBoost (native) | 1 | 0.0% | 0.0%–0.0% | 0.004 | 0.00 | 0% |
| Distributional | PGBM (native) | 1 | 41.7% | 41.7%–41.7% | 0.306 | 0.23 | 0% |
| Distributional | IBUG | 2 | 100.0% | 100.0%–100.0% | 2.68 | 2.02 | 100% |
| Distributional | Treeffuser | 5 | 100.0% | 100.0%–100.0% | 8.13 | 6.13 | 100% |
| CARD | CARD | 9 | 20.8% | 12.5%–62.5% | 0.255 | 0.19 | 0% |
| Conformal (MAPIE/PUNCC) | **J+aB (MAPIE)** | 9 | 95.8% | 91.7%–95.8% | 2.12 | 1.60 | 100% |
| Conformal (MAPIE/PUNCC) | CV+ (PUNCC) | 9 | 100.0% | 95.8%–100.0% | 2.53 | 1.91 | 100% |
| Conformal (MAPIE/PUNCC) | CV+ (MAPIE) | 9 | 95.8% | 91.7%–100.0% | 2.55 | 1.93 | 100% |
| Conformal (MAPIE/PUNCC) | Split CP (MAPIE) | 9 | 91.7% | 79.2%–100.0% | 2.73 | 2.06 | 89% |
| Conformal (MAPIE/PUNCC) | CQR (PUNCC) | 9 | 100.0% | 87.5%–100.0% | 3.19 | 2.41 | 100% |
| Conformal (MAPIE/PUNCC) | Split CP (PUNCC) | 9 | 91.7% | 87.5%–95.8% | 3.74 | 2.82 | 100% |
| Weighted conformal (NexCP) | NexCP J+aB | 9 | 95.8% | 91.7%–100.0% | 2.35 | 1.77 | 100% |
| Weighted conformal (NexCP) | NexCP Split | 9 | 91.7% | 87.5%–95.8% | 2.38 | 1.80 | 100% |
| Weighted conformal (NexCP) | NexCP CV+ | 9 | 95.8% | 91.7%–100.0% | 2.61 | 1.97 | 100% |
| Weighted conformal (NexCP) | NexCP Full † | 9 | 91.7% | 75.0%–100.0% | 2.77 | 2.09 | 89% |
| Online conformal | ACI | 9 | 91.7% | 83.3%–95.8% | 2.23 | 1.68 | 100% |
| Online conformal | FACI | 9 | 91.7% | 83.3%–95.8% | 2.24 | 1.69 | 100% |
| Online conformal | Online J+aB | 9 | 95.8% | 91.7%–95.8% | 2.25 | 1.70 | 100% |
| Online conformal | MFCS Split | 9 | 91.7% | 79.2%–95.8% | 2.33 | 1.75 | 89% |
| Online conformal | Online Split | 9 | 91.7% | 79.2%–95.8% | 2.33 | 1.75 | 89% |
| Online conformal | SAOCP | 9 | 87.5% | 79.2%–91.7% | 2.42 | 1.83 | 89% |
| Online conformal | Online CV+ | 9 | 95.8% | 91.7%–100.0% | 2.43 | 1.83 | 100% |
| Online conformal | MFCS Full | 9 | 95.8% | 91.7%–100.0% | 2.43 | 1.83 | 100% |
| Online conformal | SF-OGD | 9 | 87.5% | 87.5%–91.7% | 2.59 | 1.95 | 100% |
| COP | COP | 9 | 91.7% | 83.3%–95.8% | 2.22 | 1.67 | 100% |

<details><summary><b>Tuned models (NB1)</b></summary>

| Model | Pruner | CV RMSE | Test RMSE | Test r |
|---|---|--:|--:|--:|
| LightGBM | HyperbandPruner | 0.567 | 0.761 | 0.901 |
| XGBoost | NopPruner | 0.567 | 0.715 | 0.921 |
| GPBoost | HyperbandPruner | 0.582 | 0.674 | 0.908 |
| CatBoost | SuccessiveHalvingPruner | 0.604 | 0.506 | 0.938 |
| GradientBoosting | PercentilePruner | 0.609 | 0.682 | 0.925 |
| NGBoost | NopPruner | 0.611 | 0.712 | 0.923 |
| PGBM | HyperbandPruner | 0.623 | 0.492 | 0.951 |
| HistGB | HyperbandPruner | 0.636 | 0.692 | 0.915 |
| RandomForest | SuccessiveHalvingPruner | 0.643 | 0.439 | 0.955 |
| TabNet | MedianPruner | 0.817 | 1.14 | 0.836 |

</details>

---

## 5. Folder map and Kaggle runs

```
pile_uncertainty_analysis/
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

Every run used for the results is linked in the table of [section 2](#2-what-was-run) and in each `results/README.md`. To reproduce, run the eight notebooks in order ([how](../../README.md#11-reproducing-the-results)). Machine-readable numbers: [`results/benchmark_long.csv`](../../results/benchmark_long.csv) (rows with `dataset = pile`).
