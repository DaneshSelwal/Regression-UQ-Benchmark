# Pile settlement: how far can the predictions be trusted?

[← main README](../../README.md) · [all datasets](../../README.md#3-results-on-all-nine-datasets) · [report (PDF)](report/paper.pdf) · [method guides](../../notebooks/)

| | |
|---|---|
| **Data** | 371 training / 186 test rows, 8 features, target `S_exp ` |
| **Key result** | Conformal intervals are valid in **131 of 152** model × method cases; the sharpest valid method is **MFCS Full** (89.8% coverage, 0.74 × std(y) wide) |
| **Watch out** | Raw quantile regression covers only 82.8% (median); native Gaussian intervals: NGBoost 61.3%, PGBM 8.1% |
| **Best point model** | GPBoost (5-fold CV RMSE 9.53) |
| **Full report** | [`report/paper.pdf`](report/paper.pdf): methods with maths, all tables, appendix |

## Contents

1. [The data](#1-the-data)
2. [What was run](#2-what-was-run)
3. [Results, question by question](#3-results-question-by-question)
4. [All interval methods](#4-all-interval-methods)
5. [Folder map and Kaggle runs](#5-folder-map-and-kaggle-runs)

---

## 1. The data

<p align="center"><img src="report/figures/fig_data.png" width="90%" alt="Target distribution and strongest features"></p>

*__Figure 1.__ Left: the target on the training and test sets. Right: the features most correlated with it (Spearman, training set).*

| Statistic | Train | Test |
|---|--:|--:|
| Rows | 371.0 | 186.0 |
| Mean | 13.37 | 14.97 |
| Std | 23.93 | 27.20 |
| Min | 0.000 | 0.000 |
| Median | 3.69 | 4.03 |
| Max | 128.7 | 129.9 |

**Features:** `D`, `L`, `N30_Ava`, `Cu_Ava`, `N30_Base`, `Cu_Base`, `UL`, `Q `.

**Reading Figure 1.** The target's skewness is 2.44 (a long right tail: a few large values are the hardest to predict and to cover). The test mean differs from the training mean by +0.07 training standard deviations and the test spread is 1.14× the training spread — the two sets look alike. With 46 training rows per feature, there is enough data per feature for stable tree models.

---

## 2. What was run

The same eight notebooks as every other dataset ([pipeline](../../README.md#1-overview-and-pipeline)); every phase folder has its own README with this dataset's numbers.

| Phase | What it computes | Results | Kaggle run(s) |
|---|---|---|---|
| [NB1](hyperparameter_tuning/README.md) | Hyperparameter tuning (Optuna) | [`hyperparameter_tuning/results/`](hyperparameter_tuning/results/) | [REG22_11_2026_17_56](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-17-56) |
| [NB2](quantile_regression/README.md) | Quantile regression | [`quantile_regression/results/`](quantile_regression/results/) | [REG22_11_2026_21_09](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-21-09) |
| [NB3](probabilistic_distribution/README.md) | Distributional (NGBoost, PGBM, IBUG, Treeffuser) | [`probabilistic_distribution/results/`](probabilistic_distribution/results/) | [REG22_11_2026_21_39](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-21-39) |
| [NB4](probabilistic_distribution_card/README.md) | Distributional (CARD diffusion) | [`probabilistic_distribution_card/results/`](probabilistic_distribution_card/results/) | [REG22_11_2026_21_49](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-21-49) |
| [NB5](conformal_predictions_mapie_puncc/README.md) | Conformal (MAPIE, PUNCC) | [`conformal_predictions_mapie_puncc/results/`](conformal_predictions_mapie_puncc/results/) | [REG22_11_2026_22_02](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-22-02) |
| [NB6](conformal_predictions_nexcp_adaptivecp_mfcs/README.md) | NexCP and online conformal | [`conformal_predictions_nexcp_adaptivecp_mfcs/results/`](conformal_predictions_nexcp_adaptivecp_mfcs/results/) | [REG23_11_2026_06_12](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-06-12), [REG23_11_2026_12_28](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-12-28), [REG23_11_2026_13_04](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-13-04), [REG23_11_2026_13_20](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-13-20), [REG23_11_2026_13_36](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-13-36), [REG23_11_2026_14_51](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-14-51), [REG23_11_2026_06_19](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-06-19) |
| [NB7](conformal_predictions_adaptive_coverage_policies/README.md) | Adaptive coverage policies (ACP) | [`conformal_predictions_adaptive_coverage_policies/results/`](conformal_predictions_adaptive_coverage_policies/results/) | [REG22_11_2026_22_21](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-22-21) |
| [NB8](hyperspherical_confidence_mapping_hcm/README.md) | Hyperspherical confidence mapping (HCM) | [`hyperspherical_confidence_mapping_hcm/results/`](hyperspherical_confidence_mapping_hcm/results/) | [REG24_11_2026_15_21](https://www.kaggle.com/code/daneshselwal/reg24-11-2026-15-21), [REG22_11_2026_22_24](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-22-24) |

**Not run (not enough compute).** NB6 for NGBoost, PGBM: these jobs exceed Kaggle's 12-hour limit.

---

## 3. Results, question by question

Every interval targets **90%**. With 186 test points a correct method can land anywhere above **0.864** by chance, so that is the validity threshold ([why](../../README.md#52-validity-of-a-coverage-value)). Relative width = interval width / std of the test target.

### RQ1 · How accurate are the tuned point predictors?

<p align="center"><img src="report/figures/fig_accuracy.png" width="75%" alt="Accuracy"></p>

**How to read it.** Grey bars: 5-fold CV RMSE on the training set (used to choose the model). Blue bars: RMSE on the untouched test set; dark blue = best CV model.

**What it shows.**

- **GPBoost** has the lowest 5-fold CV RMSE (9.53); on the test set the lowest RMSE is **LightGBM** (10.24).
- The target's test standard deviation is 27.20, so the best test RMSE explains about 86% of the variance (1 − RMSE²/var); the median model explains 85%.
- CV and test rankings agree only roughly: with 186 test points the test RMSE of one model has a sampling error of several percent.
- TabNet, the only neural base model, is among the least accurate, as is common for small tabular data.

> **Takeaway RQ1.** The tree ensembles are close to each other in accuracy, so the choice of UQ method matters more for the quality of the intervals than the choice among the boosted models.

### RQ2 · Does raw quantile regression give a valid 90% interval?

**How to read it.** Quantile regression fits one model per level; its 90% interval is the 5% and 95% models' predictions. No calibration step is involved.

**What it shows.**

- The raw 90% interval [q̂₀.₀₅, q̂₀.₉₅] has a median coverage of **82.8%** over 9 models (range 6.5%–95.2%); 3 of 9 are valid (≥ 0.864).
- Quantiles cross on a median **99.5%** of test points before sorting: the 13 separately fitted models disagree on the order of their own quantiles.
- Its median width is 0.64 × std(y), narrower than most conformal intervals; the narrowness is what makes it undercover.
- Best quantile CRPS: TabNet (2.37).

> **Takeaway RQ2.** Raw quantile regression is not a reliable 90% interval here; it needs conformal calibration (CQR) or replacement by a conformal method.

### RQ3 · How good are the predictive distributions?

<p align="center"><img src="report/figures/fig_crps.png" width="75%" alt="CRPS"></p>

**How to read it.** CRPS (log scale, lower is better) scores a whole predictive distribution: it rewards a sharp distribution centred on the truth. Coverage, in the text, checks only the 90% interval.

**What it shows.**

- NGBoost (native Gaussian): coverage **61.3%**, width 0.19 × std(y), PIT diagnosis: under-dispersed (over-confident).
- PGBM (native Gaussian): coverage **8.1%**, width 0.02 × std(y), PIT diagnosis: under-dispersed (over-confident).
- IBUG on XGBoost: coverage 95.2%, width 1.31 × std(y).
- IBUG on LightGBM: coverage 96.8%, width 1.22 × std(y).
- Treeffuser (median of 5 models): coverage 96.8%, width 2.70 × std(y).
- CARD (median of 9 guides): coverage 61.3%, width 0.13 × std(y).
- Lowest CRPS of all predictive distributions: **TabNet (CARD)**, 2.08 (target units).

> **Takeaway RQ3.** The native NGBoost and PGBM distributions are over-confident on this dataset: their σ̂ is too small. A good CRPS does not imply a valid interval: CRPS rewards a sharp distribution centred near the truth, while coverage checks the tails.

### RQ4 · Do conformal methods keep their 90% promise?

<p align="center"><img src="report/figures/fig_method_league.png" width="85%" alt="Interval methods"></p>

<p align="center"><img src="report/figures/fig_coverage_width.png" width="75%" alt="Coverage vs width"></p>

**How to read it.** Top: one row per method; dot = median coverage over the base models it wraps, bar = min–max, green band = valid region; right panel = median relative width. Bottom: every model × method pair; good intervals sit in the green band, far to the left.

**What it shows.**

- Conformal intervals reach the target in **131 of 152** model × method cases (86.2%).
- Misses: J+aB (MAPIE) (6), MFCS Split (3), Online Split (3), CQR (PUNCC) (2), Split CP (PUNCC) (2), ACI (1), NexCP Full (1), NexCP Split (1), Online J+aB (1), Split CP (MAPIE) (1). Lowest coverage: Split CP (MAPIE) on TabNet, 78.0%.
- NexCP Full intervals sometimes reach the edge of their y-grid, so their width is a lower bound; they are not ranked (†).

> **Takeaway RQ4.** Conformal methods are valid on most, but not all, models here; the misses show where the exchangeability assumption or a small calibration set bites.

### RQ5 · Which valid method is the sharpest?

<p align="center"><img src="report/figures/fig_intervals.png" width="75%" alt="Intervals"></p>

**How to read it.** The CV+ intervals of GPBoost on every test point, sorted by the observed value; red crosses are misses. Marginal coverage allows misses to cluster, typically at the extremes.

**What it shows.**

- Recommended: **MFCS Full**, median coverage 89.8%, median width 20.22 = **0.74 × std(y)**.
- Next narrowest valid methods: CV+ (MAPIE) 0.81; ACI 0.91; FACI 0.93.
- The widest valid method (Treeffuser, 2.70) is 3.6 × wider than the recommended one: validity alone does not pick a method.
- Raw quantile regression is 0.64 × std(y) wide but covers only 82.8%.
- Median width of valid intervals by family: Quantile regression 0.92; Online conformal 0.96; COP 0.97; Conformal (MAPIE/PUNCC) 0.98; Weighted conformal (NexCP) 1.12; Distributional 1.97.

> **Takeaway RQ5.** On this dataset the sharpest valid interval comes from **MFCS Full**.

### RQ6 · What do the learned-coverage and single-network methods add?

**How to read it.** ACP coverage must be compared with its own learned target 1 − α̃, not with 90%. HCM reports a 95% interval and the correlation between its predicted spread and the actual error.

**What it shows.**

- ACP learns its own level: mean learned target 1 − α̃ = 63.2%; realised coverage 79.0%–86.0% across 9 models.
- HCM (one network, 95% interval): coverage 90.3%, RMSE 9.61, Spearman(|error|, σ̂) = 0.43 — its uncertainty tracks the error.
- Extreme CP (one-sided upper bound): coverage 90.3%–95.2% of the bound y ≤ ŷ + q.

> **Takeaway RQ6.** Learned-level and single-network methods answer different questions from a fixed 90% interval; they are reported for completeness and are not ranked.

---

## 4. All interval methods

Aggregated over the base models each method wraps. *Valid* = share of models with coverage ≥ 0.864. † = some intervals hit the edge of the label grid, so the width is a lower bound (not ranked). **Bold** = recommended.

| Family | Method | Models | Coverage (median) | Coverage range | Width (median) | Relative width | Valid |
|---|---|--:|--:|--:|--:|--:|--:|
| Quantile regression | Quantile regression (raw) | 9 | 82.8% | 6.5%–95.2% | 17.42 | 0.64 | 33% |
| Distributional | PGBM (native) | 1 | 8.1% | 8.1%–8.1% | 0.626 | 0.02 | 0% |
| Distributional | NGBoost (native) | 1 | 61.3% | 61.3%–61.3% | 5.30 | 0.19 | 0% |
| Distributional | IBUG | 2 | 96.0% | 95.2%–96.8% | 34.46 | 1.27 | 100% |
| Distributional | Treeffuser | 5 | 96.8% | 95.2%–98.9% | 73.40 | 2.70 | 100% |
| CARD | CARD | 9 | 61.3% | 48.9%–76.9% | 3.67 | 0.13 | 0% |
| Conformal (MAPIE/PUNCC) | CV+ (MAPIE) | 9 | 88.2% | 86.6%–96.2% | 22.17 | 0.81 | 100% |
| Conformal (MAPIE/PUNCC) | J+aB (MAPIE) | 9 | 85.5% | 82.3%–90.3% | 22.99 | 0.85 | 33% |
| Conformal (MAPIE/PUNCC) | CV+ (PUNCC) | 9 | 90.3% | 87.1%–95.7% | 25.58 | 0.94 | 100% |
| Conformal (MAPIE/PUNCC) | CQR (PUNCC) | 9 | 88.7% | 81.2%–94.6% | 28.52 | 1.05 | 78% |
| Conformal (MAPIE/PUNCC) | Split CP (PUNCC) | 9 | 88.2% | 85.5%–89.8% | 29.25 | 1.08 | 78% |
| Conformal (MAPIE/PUNCC) | Split CP (MAPIE) | 9 | 91.9% | 78.0%–94.6% | 34.37 | 1.26 | 89% |
| Weighted conformal (NexCP) | NexCP Full † | 7 | 90.3% | 82.8%–93.0% | 14.44 | 0.53 | 86% |
| Weighted conformal (NexCP) | NexCP Split | 7 | 90.9% | 83.3%–93.0% | 28.13 | 1.03 | 86% |
| Weighted conformal (NexCP) | NexCP CV+ | 7 | 91.9% | 90.9%–96.2% | 30.90 | 1.14 | 100% |
| Weighted conformal (NexCP) | NexCP J+aB | 7 | 90.9% | 87.6%–93.5% | 34.11 | 1.25 | 100% |
| Online conformal | **MFCS Full** | 7 | 89.8% | 88.7%–90.3% | 20.22 | 0.74 | 100% |
| Online conformal | MFCS Split | 7 | 88.2% | 82.8%–91.4% | 22.42 | 0.82 | 57% |
| Online conformal | Online Split | 7 | 88.2% | 82.8%–91.4% | 22.42 | 0.82 | 57% |
| Online conformal | ACI | 7 | 89.2% | 86.0%–90.9% | 24.84 | 0.91 | 86% |
| Online conformal | FACI | 7 | 89.2% | 86.6%–90.3% | 25.41 | 0.93 | 100% |
| Online conformal | Online J+aB | 7 | 87.6% | 86.0%–91.4% | 26.13 | 0.96 | 86% |
| Online conformal | Online CV+ | 7 | 90.3% | 89.8%–95.7% | 26.16 | 0.96 | 100% |
| Online conformal | SF-OGD | 7 | 90.3% | 89.8%–90.9% | 35.04 | 1.29 | 100% |
| Online conformal | SAOCP | 7 | 91.9% | 90.3%–93.0% | 44.62 | 1.64 | 100% |
| COP | COP | 7 | 89.8% | 87.6%–90.3% | 26.44 | 0.97 | 100% |

<details><summary><b>Tuned models (NB1)</b></summary>

| Model | Pruner | CV RMSE | Test RMSE | Test r |
|---|---|--:|--:|--:|
| GPBoost | HyperbandPruner | 9.53 | 11.41 | 0.909 |
| XGBoost | HyperbandPruner | 9.77 | 10.53 | 0.926 |
| GradientBoosting | MedianPruner | 9.89 | 10.33 | 0.931 |
| CatBoost | WilcoxonPruner | 10.03 | 10.62 | 0.924 |
| PGBM | WilcoxonPruner | 10.21 | 10.41 | 0.927 |
| LightGBM | WilcoxonPruner | 10.23 | 10.24 | 0.933 |
| TabNet | PercentilePruner | 10.34 | 16.81 | 0.854 |
| HistGB | WilcoxonPruner | 10.74 | 10.99 | 0.919 |
| NGBoost | MedianPruner | 10.83 | 13.54 | 0.874 |
| RandomForest | WilcoxonPruner | 10.97 | 12.33 | 0.911 |

</details>

---

## 5. Folder map and Kaggle runs

```
pile_settlement_uncertainty_analysis/
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

Every run used for the results is linked in the table of [section 2](#2-what-was-run) and in each `results/README.md`. To reproduce, run the eight notebooks in order ([how](../../README.md#6-reproducing-the-results)). Machine-readable numbers: [`results/benchmark_long.csv`](../../results/benchmark_long.csv) (rows with `dataset = pile_settlement`).
