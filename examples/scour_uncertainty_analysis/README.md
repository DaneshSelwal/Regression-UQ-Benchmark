# Scour: how far can the predictions be trusted?

[← main README](../../README.md) · [all datasets](../../README.md#3-results-on-all-nine-datasets) · [report (PDF)](report/paper.pdf) · [method guides](../../notebooks/)

| | |
|---|---|
| **Data** | 154 training / 78 test rows, 7 features, target ` Scour  ` |
| **Key result** | Conformal intervals are valid in **180 of 180** model × method cases; the sharpest valid method is **Split CP (MAPIE)** (93.6% coverage, 1.48 × std(y) wide) |
| **Watch out** | Raw quantile regression covers only 84.6% (median); native Gaussian intervals: NGBoost 65.4%, PGBM 100.0% |
| **Best point model** | XGBoost (5-fold CV RMSE 0.544) |
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
| Rows | 154.0 | 78.00 |
| Mean | 1.12 | 0.938 |
| Std | 1.27 | 1.06 |
| Min | 0.100 | 0.100 |
| Median | 0.700 | 0.600 |
| Max | 7.10 | 6.20 |

**Features:** `Ps`, `Pw`, ` Skew  `, ` Velocity  `, ` Depth  `, ` D50  `, ` Gradation  `.

**Reading Figure 1.** The target's skewness is 2.63 (a long right tail: a few large values are the hardest to predict and to cover). The test mean differs from the training mean by -0.14 training standard deviations and the test spread is 0.83× the training spread — the two sets look alike. With 22 training rows per feature, there is enough data per feature for stable tree models.

---

## 2. What was run

The same eight notebooks as every other dataset ([pipeline](../../README.md#1-overview-and-pipeline)); every phase folder has its own README with this dataset's numbers.

| Phase | What it computes | Results | Kaggle run(s) |
|---|---|---|---|
| [NB1](hyperparameter_tuning/README.md) | Hyperparameter tuning (Optuna) | [`hyperparameter_tuning/results/`](hyperparameter_tuning/results/) | [REG22_11_2026_14_35](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-14-35) |
| [NB2](quantile_regression/README.md) | Quantile regression | [`quantile_regression/results/`](quantile_regression/results/) | [REG22_11_2026_21_03](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-21-03) |
| [NB3](probabilistic_distribution/README.md) | Distributional (NGBoost, PGBM, IBUG, Treeffuser) | [`probabilistic_distribution/results/`](probabilistic_distribution/results/) | [REG23_11_2026_01_48](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-48) |
| [NB4](probabilistic_distribution_card/README.md) | Distributional (CARD diffusion) | [`probabilistic_distribution_card/results/`](probabilistic_distribution_card/results/) | [REG22_11_2026_21_11](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-21-11) |
| [NB5](conformal_predictions_mapie_puncc/README.md) | Conformal (MAPIE, PUNCC) | [`conformal_predictions_mapie_puncc/results/`](conformal_predictions_mapie_puncc/results/) | [REG22_11_2026_21_20](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-21-20) |
| [NB6](conformal_predictions_nexcp_adaptivecp_mfcs/README.md) | NexCP and online conformal | [`conformal_predictions_nexcp_adaptivecp_mfcs/results/`](conformal_predictions_nexcp_adaptivecp_mfcs/results/) | [REG23_11_2026_06_40](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-06-40), [REG23_11_2026_11_31](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-11-31), [REG23_11_2026_12_35](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-12-35), [REG23_11_2026_13_26](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-13-26), [REG23_11_2026_06_14](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-06-14), [REG23_11_2026_06_15](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-06-15), [REG23_11_2026_14_19](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-19), [REG23_11_2026_14_30](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-30), [REG23_11_2026_14_41](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-41) |
| [NB7](conformal_predictions_adaptive_coverage_policies/README.md) | Adaptive coverage policies (ACP) | [`conformal_predictions_adaptive_coverage_policies/results/`](conformal_predictions_adaptive_coverage_policies/results/) | [REG22_11_2026_21_45](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-21-45) |
| [NB8](hyperspherical_confidence_mapping_hcm/README.md) | Hyperspherical confidence mapping (HCM) | [`hyperspherical_confidence_mapping_hcm/results/`](hyperspherical_confidence_mapping_hcm/results/) | [REG24_11_2026_15_22](https://www.kaggle.com/code/daneshselwal/reg24-11-2026-15-22), [REG22_11_2026_21_48](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-21-48) |

---

## 3. Results, question by question

Every interval targets **90%**. With 78 test points a correct method can land anywhere above **0.844** by chance, so that is the validity threshold ([why](../../README.md#52-validity-of-a-coverage-value)). Relative width = interval width / std of the test target.

### RQ1 · How accurate are the tuned point predictors?

<p align="center"><img src="report/figures/fig_accuracy.png" width="75%" alt="Accuracy"></p>

**How to read it.** Grey bars: 5-fold CV RMSE on the training set (used to choose the model). Blue bars: RMSE on the untouched test set; dark blue = best CV model.

**What it shows.**

- **XGBoost** has the lowest 5-fold CV RMSE (0.544); on the test set the lowest RMSE is **PGBM** (0.311).
- The target's test standard deviation is 1.06, so the best test RMSE explains about 91% of the variance (1 − RMSE²/var); the median model explains 85%.
- CV and test rankings agree only roughly: with 78 test points the test RMSE of one model has a sampling error of several percent.
- TabNet, the only neural base model, is among the least accurate, as is common for small tabular data.

> **Takeaway RQ1.** The tree ensembles are close to each other in accuracy, so the choice of UQ method matters more for the quality of the intervals than the choice among the boosted models.

### RQ2 · Does raw quantile regression give a valid 90% interval?

**How to read it.** Quantile regression fits one model per level; its 90% interval is the 5% and 95% models' predictions. No calibration step is involved.

**What it shows.**

- The raw 90% interval [q̂₀.₀₅, q̂₀.₉₅] has a median coverage of **84.6%** over 9 models (range 66.7%–98.7%); 5 of 9 are valid (≥ 0.844).
- Quantiles cross on a median **97.4%** of test points before sorting: the 13 separately fitted models disagree on the order of their own quantiles.
- Its median width is 1.14 × std(y), narrower than most conformal intervals; the narrowness is what makes it undercover.
- Best quantile CRPS: PGBM (0.188).

> **Takeaway RQ2.** Raw quantile regression happens to reach the target on this dataset, but it has no guarantee and its quantiles cross; a conformal method gives the same width with a guarantee.

### RQ3 · How good are the predictive distributions?

<p align="center"><img src="report/figures/fig_crps.png" width="75%" alt="CRPS"></p>

**How to read it.** CRPS (log scale, lower is better) scores a whole predictive distribution: it rewards a sharp distribution centred on the truth. Coverage, in the text, checks only the 90% interval.

**What it shows.**

- NGBoost (native Gaussian): coverage **65.4%**, width 0.55 × std(y), PIT diagnosis: under-dispersed (over-confident).
- PGBM (native Gaussian): coverage **100.0%**, width 1.87 × std(y), PIT diagnosis: over-dispersed (under-confident).
- IBUG on XGBoost: coverage 89.7%, width 1.76 × std(y).
- IBUG on LightGBM: coverage 96.2%, width 1.67 × std(y).
- Treeffuser (median of 5 models): coverage 100.0%, width 6.81 × std(y).
- CARD (median of 9 guides): coverage 74.4%, width 0.65 × std(y).
- Lowest CRPS of all predictive distributions: **PGBM (CARD)**, 0.186 (target units).

> **Takeaway RQ3.** The native NGBoost distributions are over-confident on this dataset: their σ̂ is too small. A good CRPS does not imply a valid interval: CRPS rewards a sharp distribution centred near the truth, while coverage checks the tails.

### RQ4 · Do conformal methods keep their 90% promise?

<p align="center"><img src="report/figures/fig_method_league.png" width="85%" alt="Interval methods"></p>

<p align="center"><img src="report/figures/fig_coverage_width.png" width="75%" alt="Coverage vs width"></p>

**How to read it.** Top: one row per method; dot = median coverage over the base models it wraps, bar = min–max, green band = valid region; right panel = median relative width. Bottom: every model × method pair; good intervals sit in the green band, far to the left.

**What it shows.**

- Conformal intervals reach the target in **180 of 180** model × method cases (100.0%).
- No conformal interval falls below the validity threshold.
- NexCP Full intervals sometimes reach the edge of their y-grid, so their width is a lower bound; they are not ranked (†).

> **Takeaway RQ4.** Conformal calibration delivers the promised coverage on almost every model, whatever the model's accuracy.

### RQ5 · Which valid method is the sharpest?

<p align="center"><img src="report/figures/fig_intervals.png" width="75%" alt="Intervals"></p>

**How to read it.** The CV+ intervals of XGBoost on every test point, sorted by the observed value; red crosses are misses. Marginal coverage allows misses to cluster, typically at the extremes.

**What it shows.**

- Recommended: **Split CP (MAPIE)**, median coverage 93.6%, median width 1.56 = **1.48 × std(y)**.
- Next narrowest valid methods: FACI 1.52; ACI 1.53; COP 1.56.
- The widest valid method (Treeffuser, 6.81) is 4.6 × wider than the recommended one: validity alone does not pick a method.
- Raw quantile regression is 1.14 × std(y) wide but covers only 84.6%.
- Median width of valid intervals by family: Quantile regression 1.18; COP 1.56; Online conformal 1.64; Conformal (MAPIE/PUNCC) 1.68; Weighted conformal (NexCP) 1.94; Distributional 4.61.

> **Takeaway RQ5.** On this dataset the sharpest valid interval comes from **Split CP (MAPIE)**.

### RQ6 · What do the learned-coverage and single-network methods add?

**How to read it.** ACP coverage must be compared with its own learned target 1 − α̃, not with 90%. HCM reports a 95% interval and the correlation between its predicted spread and the actual error.

**What it shows.**

- ACP learns its own level: mean learned target 1 − α̃ = 87.1%; realised coverage 100.0%–100.0% across 9 models.
- HCM (one network, 95% interval): coverage 92.3%, RMSE 0.394, Spearman(|error|, σ̂) = 0.14 — its uncertainty tracks the error.
- Extreme CP (one-sided upper bound): coverage 82.1%–96.2% of the bound y ≤ ŷ + q.

> **Takeaway RQ6.** Learned-level and single-network methods answer different questions from a fixed 90% interval; they are reported for completeness and are not ranked.

---

## 4. All interval methods

Aggregated over the base models each method wraps. *Valid* = share of models with coverage ≥ 0.844. † = some intervals hit the edge of the label grid, so the width is a lower bound (not ranked). **Bold** = recommended.

| Family | Method | Models | Coverage (median) | Coverage range | Width (median) | Relative width | Valid |
|---|---|--:|--:|--:|--:|--:|--:|
| Quantile regression | Quantile regression (raw) | 9 | 84.6% | 66.7%–98.7% | 1.21 | 1.14 | 56% |
| Distributional | NGBoost (native) | 1 | 65.4% | 65.4%–65.4% | 0.586 | 0.55 | 0% |
| Distributional | IBUG | 2 | 92.9% | 89.7%–96.2% | 1.81 | 1.71 | 100% |
| Distributional | PGBM (native) | 1 | 100.0% | 100.0%–100.0% | 1.98 | 1.87 | 100% |
| Distributional | Treeffuser | 5 | 100.0% | 100.0%–100.0% | 7.21 | 6.81 | 100% |
| CARD | CARD | 9 | 74.4% | 57.7%–82.1% | 0.687 | 0.65 | 0% |
| Conformal (MAPIE/PUNCC) | **Split CP (MAPIE)** | 9 | 93.6% | 89.7%–96.2% | 1.56 | 1.48 | 100% |
| Conformal (MAPIE/PUNCC) | J+aB (MAPIE) | 9 | 94.9% | 92.3%–96.2% | 1.72 | 1.63 | 100% |
| Conformal (MAPIE/PUNCC) | CV+ (MAPIE) | 9 | 94.9% | 92.3%–98.7% | 1.74 | 1.64 | 100% |
| Conformal (MAPIE/PUNCC) | Split CP (PUNCC) | 9 | 93.6% | 89.7%–96.2% | 1.84 | 1.74 | 100% |
| Conformal (MAPIE/PUNCC) | CQR (PUNCC) | 9 | 94.9% | 93.6%–96.2% | 1.97 | 1.86 | 100% |
| Conformal (MAPIE/PUNCC) | CV+ (PUNCC) | 9 | 97.4% | 94.9%–100.0% | 2.00 | 1.89 | 100% |
| Weighted conformal (NexCP) | NexCP Full † | 9 | 94.9% | 93.6%–100.0% | 1.56 | 1.47 | 100% |
| Weighted conformal (NexCP) | NexCP J+aB | 9 | 96.2% | 94.9%–98.7% | 1.98 | 1.87 | 100% |
| Weighted conformal (NexCP) | NexCP CV+ | 9 | 97.4% | 96.2%–100.0% | 2.10 | 1.98 | 100% |
| Weighted conformal (NexCP) | NexCP Split | 9 | 97.4% | 94.9%–100.0% | 2.11 | 1.99 | 100% |
| Online conformal | FACI | 9 | 92.3% | 89.7%–93.6% | 1.61 | 1.52 | 100% |
| Online conformal | ACI | 9 | 94.9% | 91.0%–97.4% | 1.62 | 1.53 | 100% |
| Online conformal | MFCS Full | 9 | 94.9% | 91.0%–97.4% | 1.68 | 1.58 | 100% |
| Online conformal | SF-OGD | 9 | 89.7% | 89.7%–91.0% | 1.71 | 1.62 | 100% |
| Online conformal | Online J+aB | 9 | 94.9% | 93.6%–96.2% | 1.74 | 1.64 | 100% |
| Online conformal | Online CV+ | 9 | 94.9% | 93.6%–97.4% | 1.79 | 1.69 | 100% |
| Online conformal | MFCS Split | 9 | 94.9% | 93.6%–98.7% | 1.80 | 1.70 | 100% |
| Online conformal | Online Split | 9 | 94.9% | 93.6%–98.7% | 1.80 | 1.70 | 100% |
| Online conformal | SAOCP | 9 | 92.3% | 89.7%–93.6% | 1.88 | 1.77 | 100% |
| COP | COP | 9 | 93.6% | 91.0%–96.2% | 1.65 | 1.56 | 100% |

<details><summary><b>Tuned models (NB1)</b></summary>

| Model | Pruner | CV RMSE | Test RMSE | Test r |
|---|---|--:|--:|--:|
| XGBoost | MedianPruner | 0.544 | 0.404 | 0.929 |
| CatBoost | NopPruner | 0.550 | 0.378 | 0.934 |
| GradientBoosting | WilcoxonPruner | 0.551 | 0.425 | 0.917 |
| PGBM | MedianPruner | 0.561 | 0.311 | 0.956 |
| RandomForest | WilcoxonPruner | 0.563 | 0.457 | 0.903 |
| GPBoost | ThresholdPruner | 0.566 | 0.413 | 0.924 |
| LightGBM | NopPruner | 0.568 | 0.401 | 0.927 |
| HistGB | HyperbandPruner | 0.568 | 0.488 | 0.890 |
| NGBoost | MedianPruner | 0.579 | 0.402 | 0.925 |
| TabNet | WilcoxonPruner | 0.606 | 0.438 | 0.937 |

</details>

---

## 5. Folder map and Kaggle runs

```
scour_uncertainty_analysis/
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

Every run used for the results is linked in the table of [section 2](#2-what-was-run) and in each `results/README.md`. To reproduce, run the eight notebooks in order ([how](../../README.md#6-reproducing-the-results)). Machine-readable numbers: [`results/benchmark_long.csv`](../../results/benchmark_long.csv) (rows with `dataset = scour`).
