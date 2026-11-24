# Concrete: how far can the predictions be trusted?

[← main README](../../README.md) · [all datasets](../../README.md#3-results-on-all-nine-datasets) · [report (PDF)](report/paper.pdf) · [method guides](../../notebooks/)

| | |
|---|---|
| **Data** | 189 training / 95 test rows, 7 features, target `str` |
| **Key result** | Conformal intervals are valid in **180 of 180** model × method cases; the sharpest valid method is **MFCS Full** (90.5% coverage, 1.00 × std(y) wide) |
| **Watch out** | Raw quantile regression covers only 79.0% (median); native Gaussian intervals: NGBoost 33.7%, PGBM 21.1% |
| **Best point model** | CatBoost (5-fold CV RMSE 3.66) |
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
| Rows | 189.0 | 95.00 |
| Mean | 35.70 | 36.24 |
| Std | 10.00 | 10.03 |
| Min | 19.00 | 21.60 |
| Median | 34.00 | 34.00 |
| Max | 66.00 | 65.34 |

**Features:** `C`, `mp`, `FA`, `CA`, `F`, `W_P`, `Adm`.

**Reading Figure 1.** The target's skewness is 0.92. The test mean differs from the training mean by +0.05 training standard deviations and the test spread is 1.00× the training spread — the two sets look alike. With 27 training rows per feature, there is enough data per feature for stable tree models.

---

## 2. What was run

The same eight notebooks as every other dataset ([pipeline](../../README.md#1-overview-and-pipeline)); every phase folder has its own README with this dataset's numbers.

| Phase | What it computes | Results | Kaggle run(s) |
|---|---|---|---|
| [NB1](hyperparameter_tuning/README.md) | Hyperparameter tuning (Optuna) | [`hyperparameter_tuning/results/`](hyperparameter_tuning/results/) | [REG22_11_2026_17_57](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-17-57) |
| [NB2](quantile_regression/README.md) | Quantile regression | [`quantile_regression/results/`](quantile_regression/results/) | [REG22_11_2026_19_52](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-19-52) |
| [NB3](probabilistic_distribution/README.md) | Distributional (NGBoost, PGBM, IBUG, Treeffuser) | [`probabilistic_distribution/results/`](probabilistic_distribution/results/) | [REG22_11_2026_19_59](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-19-59) |
| [NB4](probabilistic_distribution_card/README.md) | Distributional (CARD diffusion) | [`probabilistic_distribution_card/results/`](probabilistic_distribution_card/results/) | [REG22_11_2026_20_08](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-20-08) |
| [NB5](conformal_predictions_mapie_puncc/README.md) | Conformal (MAPIE, PUNCC) | [`conformal_predictions_mapie_puncc/results/`](conformal_predictions_mapie_puncc/results/) | [REG22_11_2026_20_15](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-20-15) |
| [NB6](conformal_predictions_nexcp_adaptivecp_mfcs/README.md) | NexCP and online conformal | [`conformal_predictions_nexcp_adaptivecp_mfcs/results/`](conformal_predictions_nexcp_adaptivecp_mfcs/results/) | [REG23_11_2026_07_27](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-07-27), [REG23_11_2026_06_07](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-06-07), [REG23_11_2026_08_52](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-08-52), [REG23_11_2026_11_54](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-11-54), [REG23_11_2026_06_08](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-06-08), [REG23_11_2026_13_09](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-13-09), [REG23_11_2026_06_09](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-06-09), [REG23_11_2026_14_44](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-14-44), [REG23_11_2026_15_02](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-15-02) |
| [NB7](conformal_predictions_adaptive_coverage_policies/README.md) | Adaptive coverage policies (ACP) | [`conformal_predictions_adaptive_coverage_policies/results/`](conformal_predictions_adaptive_coverage_policies/results/) | [REG22_11_2026_20_52](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-20-52) |
| [NB8](hyperspherical_confidence_mapping_hcm/README.md) | Hyperspherical confidence mapping (HCM) | [`hyperspherical_confidence_mapping_hcm/results/`](hyperspherical_confidence_mapping_hcm/results/) | [REG24_11_2026_15_14](https://www.kaggle.com/code/daneshselwal/reg24-11-2026-15-14), [REG22_11_2026_21_44](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-21-44) |

---

## 3. Results, question by question

Every interval targets **90%**. With 95 test points a correct method can land anywhere above **0.849** by chance, so that is the validity threshold ([why](../../README.md#52-validity-of-a-coverage-value)). Relative width = interval width / std of the test target.

### RQ1 · How accurate are the tuned point predictors?

<p align="center"><img src="report/figures/fig_accuracy.png" width="75%" alt="Accuracy"></p>

**How to read it.** Grey bars: 5-fold CV RMSE on the training set (used to choose the model). Blue bars: RMSE on the untouched test set; dark blue = best CV model.

**What it shows.**

- **CatBoost** has the lowest 5-fold CV RMSE (3.66); on the test set the lowest RMSE is **HistGB** (2.81).
- The target's test standard deviation is 10.03, so the best test RMSE explains about 92% of the variance (1 − RMSE²/var); the median model explains 91%.
- CV and test rankings agree only roughly: with 95 test points the test RMSE of one model has a sampling error of several percent.
- TabNet, the only neural base model, is among the least accurate, as is common for small tabular data.

> **Takeaway RQ1.** The tree ensembles are close to each other in accuracy, so the choice of UQ method matters more for the quality of the intervals than the choice among the boosted models.

### RQ2 · Does raw quantile regression give a valid 90% interval?

**How to read it.** Quantile regression fits one model per level; its 90% interval is the 5% and 95% models' predictions. No calibration step is involved.

**What it shows.**

- The raw 90% interval [q̂₀.₀₅, q̂₀.₉₅] has a median coverage of **79.0%** over 9 models (range 16.8%–95.8%); 2 of 9 are valid (≥ 0.849).
- Quantiles cross on a median **97.9%** of test points before sorting: the 13 separately fitted models disagree on the order of their own quantiles.
- Its median width is 1.06 × std(y), narrower than most conformal intervals; the narrowness is what makes it undercover.
- Best quantile CRPS: LightGBM (1.77).

> **Takeaway RQ2.** Raw quantile regression is not a reliable 90% interval here; it needs conformal calibration (CQR) or replacement by a conformal method.

### RQ3 · How good are the predictive distributions?

<p align="center"><img src="report/figures/fig_crps.png" width="75%" alt="CRPS"></p>

**How to read it.** CRPS (log scale, lower is better) scores a whole predictive distribution: it rewards a sharp distribution centred on the truth. Coverage, in the text, checks only the 90% interval.

**What it shows.**

- NGBoost (native Gaussian): coverage **33.7%**, width 0.19 × std(y), PIT diagnosis: under-dispersed (over-confident).
- PGBM (native Gaussian): coverage **21.1%**, width 0.13 × std(y), PIT diagnosis: under-dispersed (over-confident).
- IBUG on XGBoost: coverage 95.8%, width 1.63 × std(y).
- IBUG on LightGBM: coverage 86.3%, width 1.03 × std(y).
- Treeffuser (median of 5 models): coverage 100.0%, width 3.33 × std(y).
- CARD (median of 9 guides): coverage 50.5%, width 0.28 × std(y).
- Lowest CRPS of all predictive distributions: **XGBoost (CARD)**, 1.61 (target units).

> **Takeaway RQ3.** The native NGBoost and PGBM distributions are over-confident on this dataset: their σ̂ is too small. A good CRPS does not imply a valid interval: CRPS rewards a sharp distribution centred near the truth, while coverage checks the tails.

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

**How to read it.** The CV+ intervals of CatBoost on every test point, sorted by the observed value; red crosses are misses. Marginal coverage allows misses to cluster, typically at the extremes.

**What it shows.**

- Recommended: **MFCS Full**, median coverage 90.5%, median width 10.05 = **1.00 × std(y)**.
- Next narrowest valid methods: FACI 1.03; COP 1.05; Online J+aB 1.07.
- The widest valid method (Treeffuser, 3.33) is 3.3 × wider than the recommended one: validity alone does not pick a method.
- Raw quantile regression is 1.06 × std(y) wide but covers only 79.0%.
- Median width of valid intervals by family: COP 1.05; Online conformal 1.17; Weighted conformal (NexCP) 1.28; Conformal (MAPIE/PUNCC) 1.30; Distributional 2.13; Quantile regression 2.53.

> **Takeaway RQ5.** On this dataset the sharpest valid interval comes from **MFCS Full**.

### RQ6 · What do the learned-coverage and single-network methods add?

**How to read it.** ACP coverage must be compared with its own learned target 1 − α̃, not with 90%. HCM reports a 95% interval and the correlation between its predicted spread and the actual error.

**What it shows.**

- ACP learns its own level: mean learned target 1 − α̃ = 63.1%; realised coverage 84.2%–97.9% across 9 models.
- HCM (one network, 95% interval): coverage 78.9%, RMSE 5.62, Spearman(|error|, σ̂) = 0.18 — its uncertainty tracks the error.
- Extreme CP (one-sided upper bound): coverage 90.5%–93.7% of the bound y ≤ ŷ + q.

> **Takeaway RQ6.** Learned-level and single-network methods answer different questions from a fixed 90% interval; they are reported for completeness and are not ranked.

---

## 4. All interval methods

Aggregated over the base models each method wraps. *Valid* = share of models with coverage ≥ 0.849. † = some intervals hit the edge of the label grid, so the width is a lower bound (not ranked). **Bold** = recommended.

| Family | Method | Models | Coverage (median) | Coverage range | Width (median) | Relative width | Valid |
|---|---|--:|--:|--:|--:|--:|--:|
| Quantile regression | Quantile regression (raw) | 9 | 79.0% | 16.8%–95.8% | 10.63 | 1.06 | 22% |
| Distributional | PGBM (native) | 1 | 21.1% | 21.1%–21.1% | 1.28 | 0.13 | 0% |
| Distributional | NGBoost (native) | 1 | 33.7% | 33.7%–33.7% | 1.88 | 0.19 | 0% |
| Distributional | IBUG | 2 | 91.1% | 86.3%–95.8% | 13.34 | 1.33 | 100% |
| Distributional | Treeffuser | 5 | 100.0% | 93.7%–100.0% | 33.41 | 3.33 | 100% |
| CARD | CARD | 9 | 50.5% | 27.4%–60.0% | 2.81 | 0.28 | 0% |
| Conformal (MAPIE/PUNCC) | J+aB (MAPIE) | 9 | 91.6% | 86.3%–93.7% | 11.05 | 1.10 | 100% |
| Conformal (MAPIE/PUNCC) | CV+ (MAPIE) | 9 | 93.7% | 89.5%–97.9% | 11.29 | 1.12 | 100% |
| Conformal (MAPIE/PUNCC) | Split CP (PUNCC) | 9 | 88.4% | 85.3%–93.7% | 11.92 | 1.19 | 100% |
| Conformal (MAPIE/PUNCC) | CV+ (PUNCC) | 9 | 94.7% | 93.7%–100.0% | 12.62 | 1.26 | 100% |
| Conformal (MAPIE/PUNCC) | Split CP (MAPIE) | 9 | 94.7% | 90.5%–98.9% | 14.77 | 1.47 | 100% |
| Conformal (MAPIE/PUNCC) | CQR (PUNCC) | 9 | 95.8% | 93.7%–97.9% | 31.36 | 3.13 | 100% |
| Weighted conformal (NexCP) | NexCP Full † | 9 | 92.6% | 85.3%–97.9% | 10.67 | 1.06 | 100% |
| Weighted conformal (NexCP) | NexCP J+aB | 9 | 92.6% | 87.4%–94.7% | 11.42 | 1.14 | 100% |
| Weighted conformal (NexCP) | NexCP Split | 9 | 94.7% | 92.6%–97.9% | 13.90 | 1.39 | 100% |
| Weighted conformal (NexCP) | NexCP CV+ | 9 | 95.8% | 90.5%–98.9% | 14.45 | 1.44 | 100% |
| Online conformal | **MFCS Full** | 9 | 90.5% | 88.4%–93.7% | 10.05 | 1.00 | 100% |
| Online conformal | FACI | 9 | 90.5% | 89.5%–91.6% | 10.34 | 1.03 | 100% |
| Online conformal | Online J+aB | 9 | 91.6% | 87.4%–94.7% | 10.77 | 1.07 | 100% |
| Online conformal | ACI | 9 | 91.6% | 90.5%–93.7% | 11.33 | 1.13 | 100% |
| Online conformal | SAOCP | 9 | 91.6% | 90.5%–93.7% | 12.15 | 1.21 | 100% |
| Online conformal | MFCS Split | 9 | 92.6% | 92.6%–96.8% | 12.19 | 1.21 | 100% |
| Online conformal | Online Split | 9 | 92.6% | 92.6%–96.8% | 12.19 | 1.21 | 100% |
| Online conformal | Online CV+ | 9 | 94.7% | 91.6%–97.9% | 12.52 | 1.25 | 100% |
| Online conformal | SF-OGD | 9 | 90.5% | 90.5%–91.6% | 12.84 | 1.28 | 100% |
| COP | COP | 9 | 90.5% | 88.4%–91.6% | 10.56 | 1.05 | 100% |

<details><summary><b>Tuned models (NB1)</b></summary>

| Model | Pruner | CV RMSE | Test RMSE | Test r |
|---|---|--:|--:|--:|
| CatBoost | NopPruner | 3.66 | 2.99 | 0.954 |
| GradientBoosting | MedianPruner | 3.78 | 3.26 | 0.945 |
| NGBoost | MedianPruner | 3.79 | 3.19 | 0.948 |
| PGBM | WilcoxonPruner | 3.85 | 3.04 | 0.953 |
| XGBoost | WilcoxonPruner | 3.85 | 2.89 | 0.958 |
| LightGBM | HyperbandPruner | 3.87 | 2.89 | 0.958 |
| GPBoost | WilcoxonPruner | 3.97 | 3.18 | 0.948 |
| HistGB | HyperbandPruner | 4.14 | 2.81 | 0.960 |
| RandomForest | WilcoxonPruner | 4.43 | 3.39 | 0.943 |
| TabNet | HyperbandPruner | 7.29 | 5.75 | 0.826 |

</details>

---

## 5. Folder map and Kaggle runs

```
concrete_uncertainty_analysis/
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

Every run used for the results is linked in the table of [section 2](#2-what-was-run) and in each `results/README.md`. To reproduce, run the eight notebooks in order ([how](../../README.md#6-reproducing-the-results)). Machine-readable numbers: [`results/benchmark_long.csv`](../../results/benchmark_long.csv) (rows with `dataset = concrete`).
