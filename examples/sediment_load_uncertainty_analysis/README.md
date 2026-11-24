# Sediment load: how far can the predictions be trusted?

[← main README](../../README.md) · [all datasets](../../README.md#3-results-on-all-nine-datasets) · [report (PDF)](report/paper.pdf) · [method guides](../../notebooks/)

| | |
|---|---|
| **Data** | 1095 training / 365 test rows, 3 features, target `St` |
| **Key result** | Conformal intervals are valid in **139 of 145** model × method cases; the sharpest valid method is **NexCP J+aB** (89.3% coverage, 0.24 × std(y) wide) |
| **Watch out** | Raw quantile regression covers only 81.1% (median); native Gaussian intervals: NGBoost 85.8%, PGBM 6.6% |
| **Best point model** | TabNet (5-fold CV RMSE 1,925) |
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
| Rows | 1,095 | 365.0 |
| Mean | 991.8 | 369.0 |
| Std | 4,860 | 1,235 |
| Min | 3.40 | 0.130 |
| Median | 47.00 | 23.00 |
| Max | 84,400 | 7,400 |

**Features:** `Qt`, `Qt-1`, `St-1`.

**Reading Figure 1.** The target's skewness is 12.92 (a long right tail: a few large values are the hardest to predict and to cover). The test mean differs from the training mean by -0.13 training standard deviations and the test spread is 0.25× the training spread — the two sets are not identically distributed, which strains methods that assume exchangeability. With 365 training rows per feature, there is enough data per feature for stable tree models.

---

## 2. What was run

The same eight notebooks as every other dataset ([pipeline](../../README.md#1-overview-and-pipeline)); every phase folder has its own README with this dataset's numbers.

| Phase | What it computes | Results | Kaggle run(s) |
|---|---|---|---|
| [NB1](hyperparameter_tuning/README.md) | Hyperparameter tuning (Optuna) | [`hyperparameter_tuning/results/`](hyperparameter_tuning/results/) | [REG22_11_2026_17_58](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-17-58) |
| [NB2](quantile_regression/README.md) | Quantile regression | [`quantile_regression/results/`](quantile_regression/results/) | [REG23_11_2026_01_24](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-24) |
| [NB3](probabilistic_distribution/README.md) | Distributional (NGBoost, PGBM, IBUG, Treeffuser) | [`probabilistic_distribution/results/`](probabilistic_distribution/results/) | [REG23_11_2026_01_25](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-25) |
| [NB4](probabilistic_distribution_card/README.md) | Distributional (CARD diffusion) | [`probabilistic_distribution_card/results/`](probabilistic_distribution_card/results/) | [REG23_11_2026_01_34](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-34) |
| [NB5](conformal_predictions_mapie_puncc/README.md) | Conformal (MAPIE, PUNCC) | [`conformal_predictions_mapie_puncc/results/`](conformal_predictions_mapie_puncc/results/) | [REG23_11_2026_01_37](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-37) |
| [NB6](conformal_predictions_nexcp_adaptivecp_mfcs/README.md) | NexCP and online conformal | [`conformal_predictions_nexcp_adaptivecp_mfcs/results/`](conformal_predictions_nexcp_adaptivecp_mfcs/results/) | [REG23_11_2026_13_40](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-13-40), [REG23_11_2026_14_42](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-42), [REG23_11_2026_14_49](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-49), [REG23_11_2026_14_54](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-14-54), [REG23_11_2026_15_01](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-15-01), [REG23_11_2026_15_04](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-15-04), [REG23_11_2026_15_09](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-15-09) |
| [NB7](conformal_predictions_adaptive_coverage_policies/README.md) | Adaptive coverage policies (ACP) | [`conformal_predictions_adaptive_coverage_policies/results/`](conformal_predictions_adaptive_coverage_policies/results/) | [REG23_11_2026_01_38](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-38) |
| [NB8](hyperspherical_confidence_mapping_hcm/README.md) | Hyperspherical confidence mapping (HCM) | [`hyperspherical_confidence_mapping_hcm/results/`](hyperspherical_confidence_mapping_hcm/results/) | [REG24_11_2026_15_23](https://www.kaggle.com/code/daneshselwal/reg24-11-2026-15-23), [REG23_11_2026_01_45](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-45) |

**Not run (not enough compute).** NB6 for TabNet, PGBM: these jobs exceed Kaggle's 12-hour limit. NexCP Full conformal was switched off for this dataset (`RUN_NEXCP_FULL = False`, FIX B16) for the same reason.

---

## 3. Results, question by question

Every interval targets **90%**. With 365 test points a correct method can land anywhere above **0.874** by chance, so that is the validity threshold ([why](../../README.md#52-validity-of-a-coverage-value)). Relative width = interval width / std of the test target.

### RQ1 · How accurate are the tuned point predictors?

<p align="center"><img src="report/figures/fig_accuracy.png" width="75%" alt="Accuracy"></p>

**How to read it.** Grey bars: 5-fold CV RMSE on the training set (used to choose the model). Blue bars: RMSE on the untouched test set; dark blue = best CV model.

**What it shows.**

- **TabNet** has the lowest 5-fold CV RMSE (1,925); on the test set the lowest RMSE is **GradientBoosting** (328.6).
- The target's test standard deviation is 1,235, so the best test RMSE explains about 93% of the variance (1 − RMSE²/var); the median model explains 86%.
- CV and test rankings agree only roughly: with 365 test points the test RMSE of one model has a sampling error of several percent.
- TabNet, the only neural base model, is among the least accurate, as is common for small tabular data.

> **Takeaway RQ1.** The tree ensembles are close to each other in accuracy, so the choice of UQ method matters more for the quality of the intervals than the choice among the boosted models.

### RQ2 · Does raw quantile regression give a valid 90% interval?

**How to read it.** Quantile regression fits one model per level; its 90% interval is the 5% and 95% models' predictions. No calibration step is involved.

**What it shows.**

- The raw 90% interval [q̂₀.₀₅, q̂₀.₉₅] has a median coverage of **81.1%** over 9 models (range 5.8%–89.3%); 1 of 9 are valid (≥ 0.874).
- Quantiles cross on a median **89.9%** of test points before sorting: the 13 separately fitted models disagree on the order of their own quantiles.
- Its median width is 0.45 × std(y), narrower than most conformal intervals; the narrowness is what makes it undercover.
- Best quantile CRPS: TabNet (69.97).

> **Takeaway RQ2.** Raw quantile regression is not a reliable 90% interval here; it needs conformal calibration (CQR) or replacement by a conformal method.

### RQ3 · How good are the predictive distributions?

<p align="center"><img src="report/figures/fig_crps.png" width="75%" alt="CRPS"></p>

**How to read it.** CRPS (log scale, lower is better) scores a whole predictive distribution: it rewards a sharp distribution centred on the truth. Coverage, in the text, checks only the 90% interval.

**What it shows.**

- NGBoost (native Gaussian): coverage **85.8%**, width 0.24 × std(y), PIT diagnosis: under-dispersed (over-confident).
- PGBM (native Gaussian): coverage **6.6%**, width 0.01 × std(y), PIT diagnosis: under-dispersed (over-confident).
- IBUG on XGBoost: coverage 98.4%, width 1.98 × std(y).
- IBUG on LightGBM: coverage 98.6%, width 1.52 × std(y).
- Treeffuser (median of 5 models): coverage 100.0%, width 5.45 × std(y).
- CARD (median of 9 guides): coverage 94.2%, width 0.43 × std(y).
- Lowest CRPS of all predictive distributions: **GPBoost (CARD)**, 64.54 (target units).

> **Takeaway RQ3.** The native NGBoost and PGBM distributions are over-confident on this dataset: their σ̂ is too small. A good CRPS does not imply a valid interval: CRPS rewards a sharp distribution centred near the truth, while coverage checks the tails.

### RQ4 · Do conformal methods keep their 90% promise?

<p align="center"><img src="report/figures/fig_method_league.png" width="85%" alt="Interval methods"></p>

<p align="center"><img src="report/figures/fig_coverage_width.png" width="75%" alt="Coverage vs width"></p>

**How to read it.** Top: one row per method; dot = median coverage over the base models it wraps, bar = min–max, green band = valid region; right panel = median relative width. Bottom: every model × method pair; good intervals sit in the green band, far to the left.

**What it shows.**

- Conformal intervals reach the target in **139 of 145** model × method cases (95.9%).
- Misses: CQR (PUNCC) (2), NexCP Split (2), Split CP (MAPIE) (1), Split CP (PUNCC) (1). Lowest coverage: CQR (PUNCC) on LightGBM, 75.6%.

> **Takeaway RQ4.** Conformal calibration delivers the promised coverage on almost every model, whatever the model's accuracy.

### RQ5 · Which valid method is the sharpest?

<p align="center"><img src="report/figures/fig_intervals.png" width="75%" alt="Intervals"></p>

**How to read it.** The CV+ intervals of TabNet on every test point, sorted by the observed value; red crosses are misses. Marginal coverage allows misses to cluster, typically at the extremes.

**What it shows.**

- Recommended: **NexCP J+aB**, median coverage 89.3%, median width 298.8 = **0.24 × std(y)**.
- Next narrowest valid methods: NexCP CV+ 0.38; CARD 0.43; ACI 0.45.
- The widest valid method (SAOCP, 8.53) is 35.3 × wider than the recommended one: validity alone does not pick a method.
- Raw quantile regression is 0.45 × std(y) wide but covers only 81.1%.
- Median width of valid intervals by family: Weighted conformal (NexCP) 0.24; CARD 0.43; Quantile regression 0.48; Online conformal 0.81; COP 0.83; Conformal (MAPIE/PUNCC) 0.99; Distributional 4.58.

> **Takeaway RQ5.** On this dataset the sharpest valid interval comes from **NexCP J+aB**.

### RQ6 · What do the learned-coverage and single-network methods add?

**How to read it.** ACP coverage must be compared with its own learned target 1 − α̃, not with 90%. HCM reports a 95% interval and the correlation between its predicted spread and the actual error.

**What it shows.**

- For 9 of 9 models (CatBoost, GPBoost, GradientBoosting, HistGB, LightGBM, NGBoost, PGBM, TabNet, XGBoost) the ACP policy saturated at α̃ = 0.999: the sets have constant width and the learned level carries no information (realised coverage 90.4%–93.7%).
- HCM (one network, 95% interval): coverage 90.1%, RMSE 520.2, Spearman(|error|, σ̂) = 0.43 — its uncertainty tracks the error.
- Extreme CP (one-sided upper bound): coverage 95.1%–98.6% of the bound y ≤ ŷ + q.

> **Takeaway RQ6.** Learned-level and single-network methods answer different questions from a fixed 90% interval; they are reported for completeness and are not ranked.

---

## 4. All interval methods

Aggregated over the base models each method wraps. *Valid* = share of models with coverage ≥ 0.874. † = some intervals hit the edge of the label grid, so the width is a lower bound (not ranked). **Bold** = recommended.

| Family | Method | Models | Coverage (median) | Coverage range | Width (median) | Relative width | Valid |
|---|---|--:|--:|--:|--:|--:|--:|
| Quantile regression | Quantile regression (raw) | 9 | 81.1% | 5.8%–89.3% | 559.6 | 0.45 | 11% |
| Distributional | PGBM (native) | 1 | 6.6% | 6.6%–6.6% | 8.65 | 0.01 | 0% |
| Distributional | NGBoost (native) | 1 | 85.8% | 85.8%–85.8% | 297.1 | 0.24 | 0% |
| Distributional | IBUG | 2 | 98.5% | 98.4%–98.6% | 2,163 | 1.75 | 100% |
| Distributional | Treeffuser | 5 | 100.0% | 99.7%–100.0% | 6,735 | 5.45 | 100% |
| CARD | CARD | 9 | 94.2% | 89.9%–95.6% | 536.6 | 0.43 | 100% |
| Conformal (MAPIE/PUNCC) | CQR (PUNCC) | 9 | 91.5% | 75.6%–96.2% | 748.1 | 0.61 | 78% |
| Conformal (MAPIE/PUNCC) | J+aB (MAPIE) | 9 | 94.5% | 93.2%–96.7% | 988.0 | 0.80 | 100% |
| Conformal (MAPIE/PUNCC) | Split CP (PUNCC) | 9 | 93.4% | 81.4%–93.7% | 1,127 | 0.91 | 89% |
| Conformal (MAPIE/PUNCC) | CV+ (PUNCC) | 9 | 96.7% | 94.5%–99.7% | 1,242 | 1.01 | 100% |
| Conformal (MAPIE/PUNCC) | Split CP (MAPIE) | 9 | 95.6% | 84.4%–97.0% | 1,364 | 1.10 | 89% |
| Conformal (MAPIE/PUNCC) | CV+ (MAPIE) | 9 | 97.8% | 94.0%–99.7% | 2,046 | 1.66 | 100% |
| Weighted conformal (NexCP) | NexCP Split | 7 | 88.8% | 86.6%–90.4% | 270.6 | 0.22 | 71% |
| Weighted conformal (NexCP) | **NexCP J+aB** | 7 | 89.3% | 87.9%–91.0% | 298.8 | 0.24 | 100% |
| Weighted conformal (NexCP) | NexCP CV+ | 7 | 92.1% | 91.2%–94.8% | 467.1 | 0.38 | 100% |
| Online conformal | ACI | 7 | 92.3% | 92.1%–93.4% | 551.6 | 0.45 | 100% |
| Online conformal | FACI | 7 | 92.3% | 91.0%–92.6% | 871.5 | 0.71 | 100% |
| Online conformal | Online J+aB | 7 | 94.0% | 93.7%–95.3% | 885.5 | 0.72 | 100% |
| Online conformal | MFCS Full | 7 | 93.7% | 92.9%–94.2% | 903.3 | 0.73 | 100% |
| Online conformal | MFCS Split | 7 | 94.2% | 92.9%–95.6% | 1,104 | 0.89 | 100% |
| Online conformal | Online Split | 7 | 94.2% | 92.9%–95.6% | 1,104 | 0.89 | 100% |
| Online conformal | Online CV+ | 7 | 95.9% | 94.8%–98.1% | 1,153 | 0.93 | 100% |
| Online conformal | SF-OGD | 7 | 90.1% | 89.9%–90.1% | 5,064 | 4.10 | 100% |
| Online conformal | SAOCP | 7 | 99.5% | 99.5%–100.0% | 10,539 | 8.53 | 100% |
| COP | COP | 7 | 95.9% | 94.8%–96.7% | 1,029 | 0.83 | 100% |

<details><summary><b>Tuned models (NB1)</b></summary>

| Model | Pruner | CV RMSE | Test RMSE | Test r |
|---|---|--:|--:|--:|
| TabNet | NopPruner | 1,925 | 728.5 | 0.962 |
| GradientBoosting | NopPruner | 2,364 | 328.6 | 0.965 |
| RandomForest | WilcoxonPruner | 2,476 | 543.8 | 0.956 |
| GPBoost | WilcoxonPruner | 2,586 | 338.2 | 0.965 |
| LightGBM | HyperbandPruner | 2,596 | 454.0 | 0.963 |
| PGBM | WilcoxonPruner | 2,620 | 440.9 | 0.938 |
| HistGB | MedianPruner | 2,631 | 420.2 | 0.965 |
| NGBoost | WilcoxonPruner | 2,640 | 673.3 | 0.932 |
| XGBoost | WilcoxonPruner | 2,723 | 696.3 | 0.958 |
| CatBoost | NopPruner | 2,762 | 482.7 | 0.925 |

</details>

---

## 5. Folder map and Kaggle runs

```
sediment_load_uncertainty_analysis/
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

Every run used for the results is linked in the table of [section 2](#2-what-was-run) and in each `results/README.md`. To reproduce, run the eight notebooks in order ([how](../../README.md#6-reproducing-the-results)). Machine-readable numbers: [`results/benchmark_long.csv`](../../results/benchmark_long.csv) (rows with `dataset = sediment_load`).
