# Noise: how far can the predictions be trusted?

[← main README](../../README.md) · [all datasets](../../README.md#3-results-on-all-nine-datasets) · [report (PDF)](report/paper.pdf) · [method guides](../../notebooks/)

| | |
|---|---|
| **Data** | 168 training / 60 test rows, 5 features, target `Leq` |
| **Key result** | Conformal intervals are valid in **173 of 180** model × method cases; the sharpest valid method is **FACI** (85.0% coverage, 0.81 × std(y) wide) |
| **Watch out** | Raw quantile regression covers only 90.0% (median); native Gaussian intervals: NGBoost 86.7%, PGBM 6.7% |
| **Best point model** | LightGBM (5-fold CV RMSE 3.97) |
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
| Rows | 168.0 | 60.00 |
| Mean | 68.06 | 64.30 |
| Std | 14.24 | 15.02 |
| Min | 28.22 | 32.02 |
| Median | 72.92 | 70.57 |
| Max | 83.59 | 81.00 |

**Features:** `A`, `B`, `C`, `D`, `O`.

**Reading Figure 1.** The target's skewness is -1.24. The test mean differs from the training mean by -0.26 training standard deviations and the test spread is 1.05× the training spread — the two sets are not identically distributed, which strains methods that assume exchangeability. With 34 training rows per feature, there is enough data per feature for stable tree models.

---

## 2. What was run

The same eight notebooks as every other dataset ([pipeline](../../README.md#1-overview-and-pipeline)); every phase folder has its own README with this dataset's numbers.

| Phase | What it computes | Results | Kaggle run(s) |
|---|---|---|---|
| [NB1](hyperparameter_tuning/README.md) | Hyperparameter tuning (Optuna) | [`hyperparameter_tuning/results/`](hyperparameter_tuning/results/) | [REG22_11_2026_17_59](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-17-59) |
| [NB2](quantile_regression/README.md) | Quantile regression | [`quantile_regression/results/`](quantile_regression/results/) | [REG22_11_2026_20_53](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-20-53) |
| [NB3](probabilistic_distribution/README.md) | Distributional (NGBoost, PGBM, IBUG, Treeffuser) | [`probabilistic_distribution/results/`](probabilistic_distribution/results/) | [REG22_11_2026_21_02](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-21-02) |
| [NB4](probabilistic_distribution_card/README.md) | Distributional (CARD diffusion) | [`probabilistic_distribution_card/results/`](probabilistic_distribution_card/results/) | [REG23_11_2026_01_19](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-01-19) |
| [NB5](conformal_predictions_mapie_puncc/README.md) | Conformal (MAPIE, PUNCC) | [`conformal_predictions_mapie_puncc/results/`](conformal_predictions_mapie_puncc/results/) | [REG22_11_2026_21_25](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-21-25) |
| [NB6](conformal_predictions_nexcp_adaptivecp_mfcs/README.md) | NexCP and online conformal | [`conformal_predictions_nexcp_adaptivecp_mfcs/results/`](conformal_predictions_nexcp_adaptivecp_mfcs/results/) | [REG23_11_2026_06_10](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-06-10), [REG23_11_2026_11_26](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-11-26), [REG23_11_2026_12_12](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-12-12), [REG23_11_2026_13_13](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-13-13), [REG23_11_2026_13_52](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-13-52), [REG23_11_2026_14_02](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-02), [REG23_11_2026_14_12](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-12), [REG23_11_2026_14_23](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-23), [REG23_11_2026_14_40](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-40) |
| [NB7](conformal_predictions_adaptive_coverage_policies/README.md) | Adaptive coverage policies (ACP) | [`conformal_predictions_adaptive_coverage_policies/results/`](conformal_predictions_adaptive_coverage_policies/results/) | [REG22_11_2026_21_46](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-21-46) |
| [NB8](hyperspherical_confidence_mapping_hcm/README.md) | Hyperspherical confidence mapping (HCM) | [`hyperspherical_confidence_mapping_hcm/results/`](hyperspherical_confidence_mapping_hcm/results/) | [REG24_11_2026_15_19](https://www.kaggle.com/code/daneshselwal/reg24-11-2026-15-19), [REG22_11_2026_21_59](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-21-59) |

---

## 3. Results, question by question

Every interval targets **90%**. With 60 test points a correct method can land anywhere above **0.836** by chance, so that is the validity threshold ([why](../../README.md#52-validity-of-a-coverage-value)). Relative width = interval width / std of the test target.

### RQ1 · How accurate are the tuned point predictors?

<p align="center"><img src="report/figures/fig_accuracy.png" width="75%" alt="Accuracy"></p>

**How to read it.** Grey bars: 5-fold CV RMSE on the training set (used to choose the model). Blue bars: RMSE on the untouched test set; dark blue = best CV model.

**What it shows.**

- **LightGBM** has the lowest 5-fold CV RMSE (3.97); on the test set the lowest RMSE is **GPBoost** (3.39).
- The target's test standard deviation is 15.02, so the best test RMSE explains about 95% of the variance (1 − RMSE²/var); the median model explains 92%.
- CV and test rankings agree only roughly: with 60 test points the test RMSE of one model has a sampling error of several percent.
- TabNet, the only neural base model, is among the least accurate, as is common for small tabular data.

> **Takeaway RQ1.** The tree ensembles are close to each other in accuracy, so the choice of UQ method matters more for the quality of the intervals than the choice among the boosted models.

### RQ2 · Does raw quantile regression give a valid 90% interval?

**How to read it.** Quantile regression fits one model per level; its 90% interval is the 5% and 95% models' predictions. No calibration step is involved.

**What it shows.**

- The raw 90% interval [q̂₀.₀₅, q̂₀.₉₅] has a median coverage of **90.0%** over 9 models (range 6.7%–100.0%); 6 of 9 are valid (≥ 0.836).
- Quantiles cross on a median **100.0%** of test points before sorting: the 13 separately fitted models disagree on the order of their own quantiles.
- Its median width is 0.81 × std(y), narrower than most conformal intervals; the narrowness is what makes it undercover.
- Best quantile CRPS: GPBoost (1.78).

> **Takeaway RQ2.** Raw quantile regression happens to reach the target on this dataset, but it has no guarantee and its quantiles cross; a conformal method gives the same width with a guarantee.

### RQ3 · How good are the predictive distributions?

<p align="center"><img src="report/figures/fig_crps.png" width="75%" alt="CRPS"></p>

**How to read it.** CRPS (log scale, lower is better) scores a whole predictive distribution: it rewards a sharp distribution centred on the truth. Coverage, in the text, checks only the 90% interval.

**What it shows.**

- NGBoost (native Gaussian): coverage **86.7%**, width 0.76 × std(y), PIT diagnosis: well-calibrated (uniform pit).
- PGBM (native Gaussian): coverage **6.7%**, width 0.02 × std(y), PIT diagnosis: under-dispersed (over-confident).
- IBUG on XGBoost: coverage 90.0%, width 0.95 × std(y).
- IBUG on LightGBM: coverage 98.3%, width 1.09 × std(y).
- Treeffuser (median of 5 models): coverage 100.0%, width 2.77 × std(y).
- CARD (median of 9 guides): coverage 70.0%, width 0.39 × std(y).
- Lowest CRPS of all predictive distributions: **GPBoost (Quantile regression)**, 1.78 (target units).

> **Takeaway RQ3.** The native PGBM distributions are over-confident on this dataset: their σ̂ is too small. A good CRPS does not imply a valid interval: CRPS rewards a sharp distribution centred near the truth, while coverage checks the tails.

### RQ4 · Do conformal methods keep their 90% promise?

<p align="center"><img src="report/figures/fig_method_league.png" width="85%" alt="Interval methods"></p>

<p align="center"><img src="report/figures/fig_coverage_width.png" width="75%" alt="Coverage vs width"></p>

**How to read it.** Top: one row per method; dot = median coverage over the base models it wraps, bar = min–max, green band = valid region; right panel = median relative width. Bottom: every model × method pair; good intervals sit in the green band, far to the left.

**What it shows.**

- Conformal intervals reach the target in **173 of 180** model × method cases (96.1%).
- Misses: FACI (2), CQR (PUNCC) (1), MFCS Split (1), Online J+aB (1), Online Split (1), Split CP (MAPIE) (1). Lowest coverage: Split CP (MAPIE) on TabNet, 81.7%.
- NexCP Full intervals sometimes reach the edge of their y-grid, so their width is a lower bound; they are not ranked (†).

> **Takeaway RQ4.** Conformal calibration delivers the promised coverage on almost every model, whatever the model's accuracy.

### RQ5 · Which valid method is the sharpest?

<p align="center"><img src="report/figures/fig_intervals.png" width="75%" alt="Intervals"></p>

**How to read it.** The CV+ intervals of LightGBM on every test point, sorted by the observed value; red crosses are misses. Marginal coverage allows misses to cluster, typically at the extremes.

**What it shows.**

- Recommended: **FACI**, median coverage 85.0%, median width 12.17 = **0.81 × std(y)**.
- Next narrowest valid methods: COP 0.83; ACI 0.84; Online Split 0.86.
- The widest valid method (Treeffuser, 2.77) is 3.4 × wider than the recommended one: validity alone does not pick a method.
- Raw quantile regression is 0.81 × std(y) wide but covers only 90.0%.
- Median width of valid intervals by family: CARD 0.65; COP 0.83; Online conformal 0.93; Weighted conformal (NexCP) 1.12; Conformal (MAPIE/PUNCC) 1.13; Quantile regression 1.27; Distributional 1.46.

> **Takeaway RQ5.** On this dataset the sharpest valid interval comes from **FACI**.

### RQ6 · What do the learned-coverage and single-network methods add?

**How to read it.** ACP coverage must be compared with its own learned target 1 − α̃, not with 90%. HCM reports a 95% interval and the correlation between its predicted spread and the actual error.

**What it shows.**

- ACP learns its own level: mean learned target 1 − α̃ = 61.6%; realised coverage 83.3%–93.3% across 9 models.
- HCM (one network, 95% interval): coverage 86.7%, RMSE 5.14, Spearman(|error|, σ̂) = 0.56 — its uncertainty tracks the error.
- Extreme CP (one-sided upper bound): coverage 83.3%–93.3% of the bound y ≤ ŷ + q.

> **Takeaway RQ6.** Learned-level and single-network methods answer different questions from a fixed 90% interval; they are reported for completeness and are not ranked.

---

## 4. All interval methods

Aggregated over the base models each method wraps. *Valid* = share of models with coverage ≥ 0.836. † = some intervals hit the edge of the label grid, so the width is a lower bound (not ranked). **Bold** = recommended.

| Family | Method | Models | Coverage (median) | Coverage range | Width (median) | Relative width | Valid |
|---|---|--:|--:|--:|--:|--:|--:|
| Quantile regression | Quantile regression (raw) | 9 | 90.0% | 6.7%–100.0% | 12.11 | 0.81 | 67% |
| Distributional | PGBM (native) | 1 | 6.7% | 6.7%–6.7% | 0.272 | 0.02 | 0% |
| Distributional | NGBoost (native) | 1 | 86.7% | 86.7%–86.7% | 11.48 | 0.76 | 100% |
| Distributional | IBUG | 2 | 94.2% | 90.0%–98.3% | 15.32 | 1.02 | 100% |
| Distributional | Treeffuser | 5 | 100.0% | 93.3%–100.0% | 41.57 | 2.77 | 100% |
| CARD | CARD | 9 | 70.0% | 16.7%–85.0% | 5.89 | 0.39 | 11% |
| Conformal (MAPIE/PUNCC) | J+aB (MAPIE) | 9 | 88.3% | 85.0%–91.7% | 13.58 | 0.90 | 100% |
| Conformal (MAPIE/PUNCC) | CV+ (PUNCC) | 9 | 93.3% | 91.7%–96.7% | 15.45 | 1.03 | 100% |
| Conformal (MAPIE/PUNCC) | CQR (PUNCC) | 9 | 88.3% | 83.3%–91.7% | 18.14 | 1.21 | 89% |
| Conformal (MAPIE/PUNCC) | Split CP (MAPIE) | 9 | 91.7% | 81.7%–95.0% | 18.19 | 1.21 | 89% |
| Conformal (MAPIE/PUNCC) | CV+ (MAPIE) | 9 | 95.0% | 88.3%–98.3% | 18.87 | 1.26 | 100% |
| Conformal (MAPIE/PUNCC) | Split CP (PUNCC) | 9 | 93.3% | 86.7%–95.0% | 22.49 | 1.50 | 100% |
| Weighted conformal (NexCP) | NexCP Split | 9 | 91.7% | 86.7%–96.7% | 15.91 | 1.06 | 100% |
| Weighted conformal (NexCP) | NexCP CV+ | 9 | 93.3% | 90.0%–98.3% | 16.47 | 1.10 | 100% |
| Weighted conformal (NexCP) | NexCP J+aB | 9 | 93.3% | 90.0%–95.0% | 17.19 | 1.14 | 100% |
| Weighted conformal (NexCP) | NexCP Full † | 9 | 96.7% | 90.0%–98.3% | 17.19 | 1.14 | 100% |
| Online conformal | **FACI** | 9 | 85.0% | 83.3%–90.0% | 12.17 | 0.81 | 78% |
| Online conformal | ACI | 9 | 86.7% | 85.0%–90.0% | 12.67 | 0.84 | 100% |
| Online conformal | MFCS Split | 9 | 90.0% | 83.3%–95.0% | 12.88 | 0.86 | 89% |
| Online conformal | Online Split | 9 | 90.0% | 83.3%–95.0% | 12.88 | 0.86 | 89% |
| Online conformal | Online CV+ | 9 | 91.7% | 86.7%–93.3% | 13.53 | 0.90 | 100% |
| Online conformal | Online J+aB | 9 | 86.7% | 83.3%–90.0% | 13.56 | 0.90 | 89% |
| Online conformal | MFCS Full | 9 | 86.7% | 85.0%–91.7% | 14.18 | 0.94 | 100% |
| Online conformal | SAOCP | 9 | 91.7% | 88.3%–95.0% | 16.78 | 1.12 | 100% |
| Online conformal | SF-OGD | 9 | 91.7% | 90.0%–91.7% | 17.33 | 1.15 | 100% |
| COP | COP | 9 | 86.7% | 85.0%–90.0% | 12.52 | 0.83 | 100% |

<details><summary><b>Tuned models (NB1)</b></summary>

| Model | Pruner | CV RMSE | Test RMSE | Test r |
|---|---|--:|--:|--:|
| LightGBM | HyperbandPruner | 3.97 | 4.06 | 0.962 |
| XGBoost | MedianPruner | 4.15 | 3.95 | 0.965 |
| GradientBoosting | HyperbandPruner | 4.15 | 4.34 | 0.962 |
| CatBoost | WilcoxonPruner | 4.17 | 4.26 | 0.959 |
| RandomForest | WilcoxonPruner | 4.24 | 3.92 | 0.966 |
| PGBM | SuccessiveHalvingPruner | 4.29 | 4.44 | 0.958 |
| NGBoost | ThresholdPruner | 4.35 | 4.32 | 0.958 |
| GPBoost | SuccessiveHalvingPruner | 4.46 | 3.39 | 0.974 |
| HistGB | NopPruner | 4.57 | 3.94 | 0.968 |
| TabNet | ThresholdPruner | 7.44 | 9.12 | 0.933 |

</details>

---

## 5. Folder map and Kaggle runs

```
noise_uncertainty_analysis/
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

Every run used for the results is linked in the table of [section 2](#2-what-was-run) and in each `results/README.md`. To reproduce, run the eight notebooks in order ([how](../../README.md#6-reproducing-the-results)). Machine-readable numbers: [`results/benchmark_long.csv`](../../results/benchmark_long.csv) (rows with `dataset = noise`).
