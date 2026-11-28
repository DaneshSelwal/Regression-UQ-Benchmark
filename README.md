<div align="center">

# Regression Uncertainty Quantification Benchmark

[![License: MIT](https://img.shields.io/badge/license-MIT-2a78d6)](LICENSE)
[![Release](https://img.shields.io/badge/release-v2.0.0-1baf7a)](https://github.com/DaneshSelwal/Regression-UQ-Benchmark/releases)
![Python](https://img.shields.io/badge/python-3-2a78d6)
![Base models](https://img.shields.io/badge/regressors-10-2a78d6)
![UQ methods](https://img.shields.io/badge/UQ%20methods-30%2B-eb6834)
![Compute](https://img.shields.io/badge/compute-Kaggle%20CPU-20BEFF)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-b31b1b)](CITATION.cff)

</div>

<p align="center"><em>A reproducible benchmark of more than thirty uncertainty-quantification methods for tabular regression, wrapped around ten tuned regressors and presented through a case study on concrete compressive strength.</em></p>

---

## Contents

1. [Overview](#1-overview)
2. [Highlights](#2-highlights)
3. [Repository structure](#3-repository-structure)
4. [Installation](#4-installation)
5. [Quick start](#5-quick-start)
6. [Pipeline](#6-pipeline)
7. [Methods](#7-methods)
8. [Evaluation protocol](#8-evaluation-protocol)
9. [Case study: Concrete compressive strength](#9-case-study-concrete-compressive-strength)
10. [Additional datasets](#10-additional-datasets)
11. [Reproducibility](#11-reproducibility)
12. [Limitations](#12-limitations)
13. [Roadmap](#13-roadmap)
14. [Citation](#14-citation)
15. [License](#15-license)
16. [Acknowledgements and contact](#16-acknowledgements-and-contact)

---

## 1. Overview

A regression model returns a point prediction without saying how far that prediction can be trusted. In engineering
practice, where predictions feed into design margins and regulatory decisions, a **prediction interval with a known
coverage** is often more useful than the point prediction itself.

Uncertainty-quantification (UQ) methods differ in what they assume and what they guarantee. Quantile regression and
predictive distributions are correct only if the fitted model is correct. Conformal prediction guarantees coverage in
finite samples under exchangeability, whatever the model. Weighted and online conformal methods relax exchangeability, and
learned coverage policies and single-network methods trade guarantees for flexibility. Published comparisons usually use
large benchmark collections, whereas engineering datasets are small (tens to a few hundred test points), heterogeneous and
often heavy-tailed.

This repository applies more than thirty UQ methods to the same set of tuned regressors and asks, for a given dataset:

| # | Question |
|:-:|---|
| Q1 | How accurate is each regressor as a point predictor? |
| Q2 | Does each regressor give valid 90% intervals under each UQ method? |
| Q3 | Among valid intervals, which regressor gives the sharpest (narrowest) intervals? |
| Q4 | Do model-based intervals (quantile regression, predictive distributions) reach their nominal coverage for each regressor? |

**Scope.** Tabular, single-output regression. Each dataset has one fixed train/test split. Every interval method targets
90% coverage (α = 0.1), except HCM, which reports Gaussian 1σ/2σ/3σ intervals. The main case study is the **Concrete**
dataset (Section 9). Eight further datasets are provided as worked examples (Section 10).

## 2. Highlights

- **Model comparison under a common protocol.** Nine tuned regressors, each wrapped by the same 20 two-sided conformal
  methods and by quantile regression, predictive distributions, CARD and ACP, so regressors can be compared method by method.
- **Breadth.** More than thirty UQ methods from five families: quantile regression, predictive distributions, split and
  resampling conformal prediction, weighted and online conformal prediction, and learned or single-network uncertainty.
- **Verified implementations.** Every method was checked against its publication and, where available, the authors'
  reference code. Each correction is marked `# FIX <id>` in the code and documented in
  [`docs/CHANGELOG_FIXES.md`](docs/CHANGELOG_FIXES.md).
- **No test-set leakage.** Hyperparameters are selected by 5-fold cross-validation on the training set; the test set is
  used once, for evaluation.
- **Sample-size-aware validity.** Coverage is judged with a one-sided binomial tolerance, so that small test sets are not
  penalised for sampling variation.
- **Full provenance.** Every number comes from an executed notebook stored next to its Excel outputs, with its Kaggle run
  recorded in [`docs/run_registry.tsv`](docs/run_registry.tsv).

## 3. Repository structure

```
.
├── README.md
├── CITATION.cff
├── LICENSE
├── notebooks/
│   ├── 01_hyperparameter_tuning/ … 08_hyperspherical_confidence_mapping/
│   │                               generalised pipeline (no outputs) and a method guide per notebook
│   └── concrete_uncertainty_analysis/
│       ├── data/                   train.csv, test.csv
│       └── <phase>/                notebook and results/ (EXECUTED_*.ipynb, *.xlsx, run record)
├── examples/<dataset>_uncertainty_analysis/
│                                   the eight additional datasets, organised like the Concrete folder
├── docs/
│   ├── CHANGELOG_FIXES.md          every v2.0 correction: state before, reason, state after
│   ├── DATASET_PARAMETERS.md       dataset-specific settings
│   ├── KAGGLE_RUNS.md              how the Kaggle jobs are built and executed
│   └── run_registry.tsv            every Kaggle run: code name, notebook, time, status
├── references/                     REFERENCES.md and key papers
└── tools/
    ├── kaggle/                     Kaggle job builder and push queue
    └── reports/                    summary, table and figure generators
```

## 4. Installation

The notebooks were executed with Python 3 on Kaggle CPU sessions. The principal packages and the versions used:

| Purpose | Packages (version) |
|---|---|
| Core | numpy (1.26.4), scikit-learn (1.6.1), torch (2.9.0) |
| Base regressors | xgboost (3.1.2), lightgbm (4.6.0), catboost (1.2.8), gpboost (1.6.1), ngboost (0.5.8), pgbm (2.2.0), pytorch-tabnet2 (4.5.3) |
| Tuning | optuna (4.6.0), optunahub (0.4.0), cmaes (0.12.0) |
| Interpretability | shap (0.44.0), lime (0.2.0.1), interpret (0.7.4) |
| Conformal prediction | mapie (0.6.0), puncc (0.8.0) |
| Distributions and scoring | treeffuser (fork), ibug, properscoring (0.1) |
| Output | openpyxl (3.1.5), XlsxWriter (3.2.9), matplotlib, seaborn (0.13.2), plotly (5.24.1) |

No separate installation step is needed. The first cell of every notebook (`FIX K1`) detects the environment (Kaggle or
Google Colab) and installs any missing package at these versions.

## 5. Quick start

**On Kaggle**

1. Attach the Kaggle dataset `daneshselwal/reguq-uq-data`, or upload the dataset's `data/` folder.
2. Run **notebook 1** (hyperparameter tuning). The `FIX K1` cell copies the CSV files into the expected layout.
3. Run **notebooks 2–8**, adding the notebook 1 output as an input. The `FIX K2` cell locates `best_params.pkl`.
4. For batch execution, [`tools/kaggle/`](tools/kaggle/README.md) wraps a notebook in a papermill job, names the run
   `REG<DD>_<MM>_<YYYY>_<HH>_<MM>` and records it in [`docs/run_registry.tsv`](docs/run_registry.tsv). Notebook 6 is run as
   one job per regressor to stay within Kaggle's 12-hour session limit.

**On Google Colab.** Place the dataset folder at `MyDrive/Regression/<dataset>_uncertainty_analysis/` and run the notebooks in
the same order. The Colab installation block in the first cell runs only on Colab.

**On a new dataset**

1. Provide `train.csv` and `test.csv` with the target in the last column.
2. Copy the notebooks from [`notebooks/`](notebooks/README.md).
3. In the first cells, set the dataset folder name, the data file names, `feature_names` and `TARGET_LABEL`.
4. Run notebook 1, then notebooks 2–8.

## 6. Pipeline

The pipeline has eight notebooks. Notebook 1 produces the tuned hyperparameters (`best_params.pkl`), which notebooks 2–7
load; otherwise the notebooks are independent. Notebook 8 trains its own network.

```
 train.csv / test.csv
        │
        ▼
 NB1  Hyperparameter tuning ── best_params.pkl ──┬──► NB2  Quantile regression
      (5-fold CV, Optuna)                        ├──► NB3  Predictive distributions
                                                 ├──► NB4  CARD (diffusion)
                                                 ├──► NB5  Conformal: MAPIE / PUNCC
                                                 ├──► NB6  Conformal: NexCP / online / MFCS / COP
                                                 └──► NB7  Adaptive coverage policies
 NB8  Hyperspherical confidence mapping (own network, independent of NB1)
```

| # | Notebook | Methods | Principal outputs | Method guide |
|:-:|---|---|---|---|
| 1 | Hyperparameter tuning | Optuna AutoSampler, 5-fold CV, 8 pruners; SHAP, LIME, EBM | Tuned settings, CV and test RMSE, interpretability plots | [NB1](notebooks/01_hyperparameter_tuning/README.md) |
| 2 | Quantile regression | 13 quantile levels per model | Coverage, width, CRPS, crossing rate | [NB2](notebooks/02_quantile_regression/README.md) |
| 3 | Predictive distributions | NGBoost, PGBM, IBUG, Treeffuser | μ̂, σ̂, NLL, CRPS, PIT, calibration | [NB3](notebooks/03_probabilistic_distribution/README.md) |
| 4 | CARD | Guided diffusion, 1,000 samples per point | Predictive distribution, CRPS, calibration | [NB4](notebooks/04_probabilistic_distribution_card/README.md) |
| 5 | Conformal (MAPIE, PUNCC) | Split, CV+, J+aB, CQR | Intervals, coverage, width | [NB5](notebooks/05_conformal_mapie_puncc/README.md) |
| 6 | Conformal (NexCP, online, MFCS) | NexCP (4 variants), ACI, FACI, SF-OGD, SAOCP, online split/CV+/J+aB, MFCS, COP, AlphaNet, Extreme CP | Intervals per family | [NB6](notebooks/06_conformal_nexcp_online/README.md) |
| 7 | Adaptive coverage policies | ACP with e-values; λ ∈ {10, 20, 50} | Learned level, coverage, width | [NB7](notebooks/07_adaptive_coverage_policies/README.md) |
| 8 | Hyperspherical confidence mapping | HCM (single network) | Coverage at 1/2/3σ, error–uncertainty correlation | [NB8](notebooks/08_hyperspherical_confidence_mapping/README.md) |

## 7. Methods

### 7.1 Base regressors

Ten regressors are tuned in notebook 1. Nine of them are carried into the UQ notebooks; RandomForest is used for tuning
only.

| Model | Type | Reference | Used in |
|---|---|---|---|
| RandomForest | Bagged trees | Breiman (2001) | NB1 only |
| GradientBoosting | Gradient-boosted trees | Friedman (2001) | NB1–NB7 |
| XGBoost | Gradient-boosted trees | Chen & Guestrin (2016) | NB1–NB7 |
| LightGBM | Gradient-boosted trees | Ke et al. (2017) | NB1–NB7 |
| CatBoost | Gradient-boosted trees | Prokhorenkova et al. (2018) | NB1–NB7 |
| GPBoost | Gradient-boosted trees with Gaussian process | Sigrist (2022) | NB1–NB7 |
| HistGradientBoosting | Histogram gradient boosting | Pedregosa et al. (2011) | NB1–NB7 |
| NGBoost | Probabilistic boosting (natural gradient) | Duan et al. (2020) | NB1–NB7 |
| PGBM | Probabilistic gradient boosting | Sprangers et al. (2021) | NB1–NB7 |
| TabNet | Attentive neural network for tabular data | Arik & Pfister (2021) | NB1–NB7 |

### 7.2 Uncertainty-quantification methods

| Family | Method | Reference | Implementation | Guarantee |
|---|---|---|---|---|
| Quantile regression | Pinball-loss quantile models | Koenker & Bassett (1978) | Native model objectives | None |
| Predictive distribution | NGBoost | Duan et al. (2020) | `ngboost` | None |
| | PGBM | Sprangers et al. (2021) | `pgbm` | None |
| | IBUG | Brophy & Lowd (2022) | `ibug` | None |
| | Treeffuser | Beltran-Velez et al. (2024) | `treeffuser` (fork) | None |
| | CARD | Han, Zheng & Zhou (2022) | Re-implemented from the official code | None |
| Conformal: split and resampling | Split conformal | Vovk et al. (2005); Lei et al. (2018) | MAPIE, PUNCC | Finite-sample, marginal |
| | CV+ | Barber et al. (2021) | MAPIE, PUNCC | Finite-sample (1 − 2α) |
| | Jackknife+-after-bootstrap (J+aB) | Kim, Xu & Barber (2020) | MAPIE | Finite-sample (1 − 2α) |
| | Conformalized quantile regression (CQR) | Romano, Patterson & Candès (2019) | PUNCC | Finite-sample, marginal |
| Conformal: weighted | NexCP (split, full, CV+, J+aB) | Barber et al. (2023) | Notebook implementation | Coverage gap bounded by weighted total variation |
| Conformal: online | ACI | Gibbs & Candès (2021) | Notebook implementation | Long-run coverage |
| | FACI | Gibbs & Candès (2024) | Ported from reference code | Long-run coverage |
| | SF-OGD, SAOCP | Bhatnagar et al. (2023) | Ported from reference code | Long-run coverage |
| | Online split / CV+ / J+aB, MFCS | — | Notebook implementation | Long-run coverage |
| | COP | Hu et al. (2026) | Following the official code | Long-run coverage |
| Conformal: tail | Extreme CP (one-sided) | Pasche, Lam & Engelke (2025) | Following the official code | Tail coverage |
| Learned coverage | AlphaNet / ACP with e-values | Gauthier, Bach & Jordan (2025, 2026) | Following the official code | At the learned level |
| Single network | HCM | Choi et al. (2026) | Following the official code | None (calibrated scale) |

Full bibliographic entries, with links to the reference implementations, are in
[`references/REFERENCES.md`](references/REFERENCES.md). The derivation of each method is in the
[method guides](notebooks/README.md).

**Split conformal prediction, the common building block.** The model is fitted on part of the training data, and the
absolute residuals $`s_i=|y_i-\hat f(x_i)|`$ are computed on the remaining $`m`$ calibration points. With

```math
\hat q = s_{(k)},\qquad k=\bigl\lceil (m+1)(1-\alpha)\bigr\rceil,\qquad \alpha=0.1,
```

the interval for a new point is $`\hat C(x)=[\hat f(x)-\hat q,\ \hat f(x)+\hat q]`$, and under exchangeability
$`\Pr\{Y\in\hat C(X)\}\ge 1-\alpha`$ whatever the model. The other conformal methods change one element of this construction:
the reuse of training points for calibration (CV+, J+aB), the score (CQR), the weighting of residuals (NexCP), or a
sequential update of the radius (online methods).

## 8. Evaluation protocol

**Data split.** One fixed train/test split per dataset, stored in the dataset's `data/` folder. Hyperparameters are selected
by 5-fold cross-validation on the training set (`KFold(shuffle=True, random_state=42)`); the test set is used only for
evaluation.

**Target coverage.** 90% (α = 0.1) for every interval method except HCM.

**Metrics**

| Metric | Definition | Desired value |
|---|---|---|
| **Coverage** | Share of test points inside their interval | ≥ 90% within sampling tolerance |
| **Width** | Mean upper minus lower bound, in target units | Small, given valid coverage |
| **Relative width** | Width divided by the standard deviation of the test target | Small; comparable across datasets |
| **CRPS** | Continuous ranked probability score | Low |
| **NLL** | Negative log-likelihood of the observation under the predictive distribution | Low |
| **PIT** | Predicted CDF evaluated at the observation | Uniform histogram |
| **Crossing rate** | Share of points whose separately fitted quantiles are not monotone | 0 |
| **RMSE** | Root mean squared error of the point prediction | Low |

**Validity.** The number of covered test points is binomial, so the coverage of a correctly calibrated 90% method has
standard error $`\sqrt{0.09/n}`$. An interval is **valid** when its coverage is not significantly below 90% in a one-sided
test at the 5% level:

```math
\hat c\ \ge\ 0.9-1.645\sqrt{\frac{0.09}{n_{\text{test}}}} .
```

| Test points | 24 | 47 | 95 | 186 | 365 |
|---|--:|--:|--:|--:|--:|
| Validity threshold | 79.9% | 82.8% | 84.9% | 86.4% | 87.4% |

**Comparing regressors.** For each regressor the 20 two-sided conformal methods of notebooks 5 and 6 are evaluated. A
regressor's **sharpness** is the median width over the methods for which its intervals are valid. NexCP Full intervals that
reach the edge of the label grid are only lower bounds on the true width and are not ranked. Extreme CP (one-sided) and ACP
(learned coverage level) are reported separately.

## 9. Case study: Concrete compressive strength

All numbers in this section are read from the executed notebooks and Excel files in
[`notebooks/concrete_uncertainty_analysis/`](notebooks/concrete_uncertainty_analysis/). The comparison is between
**regressors**: for each UQ method, which model gives valid and sharp intervals.

### 9.1 Dataset

| Property | Value |
|---|---|
| Domain | Civil engineering: concrete mix design |
| Source | Provided by the project supervisor; original source to be documented |
| Target | Compressive strength (`str`, MPa) |
| Features (7) | Seven mix-design variables (see below) |
| Train / test | 189 / 95 |
| Test target | mean 36.2 MPa, standard deviation 10.0 MPa, range 21.6–65.3 MPa |
| Validity threshold (n = 95) | 84.9% |

| Column | Interpreted meaning | Range (train + test) |
|---|---|---|
| `C` | Cement content (kg/m³) | 220–600 |
| `mp` | Mineral powder content (kg/m³) | 0–200 |
| `FA` | Fine aggregate content (kg/m³) | 0–975 |
| `CA` | Coarse aggregate content (kg/m³) | 525–1,278 |
| `F` | Filler / fly ash content (kg/m³) | 0–220 |
| `W_P` | Water-to-powder ratio | 0.28–0.57 |
| `Adm` | Chemical admixture (superplasticiser) dosage | 0–4,095 ¹ |

The meanings are inferred from the column codes and value ranges; the water-to-powder ratio suggests self-compacting
concrete mixes. They have not been confirmed against the original data record.
¹ All values but one are at most 10.3; a single training row has 4,095, probably a data-entry error, and is kept as supplied.

### 9.2 Point accuracy of the regressors

Tuned with Optuna (AutoSampler) by 5-fold CV on the training set; the pruner shown is the one selected for each model.
Ranked by test RMSE (Kaggle run `REG22_11_2026_17_57`).

| Regressor | Pruner | CV RMSE (MPa) | Test RMSE (MPa) | Test r |
|---|---|--:|--:|--:|
| HistGradientBoosting | Hyperband | 4.14 | **2.81** | 0.960 |
| LightGBM | Hyperband | 3.87 | 2.89 | 0.958 |
| XGBoost | Wilcoxon | 3.85 | 2.89 | 0.958 |
| CatBoost | None | **3.66** | 2.99 | 0.954 |
| PGBM | Wilcoxon | 3.85 | 3.04 | 0.953 |
| GPBoost | Wilcoxon | 3.97 | 3.18 | 0.948 |
| NGBoost | Median | 3.79 | 3.19 | 0.948 |
| GradientBoosting | Median | 3.78 | 3.26 | 0.945 |
| RandomForest | Wilcoxon | 4.43 | 3.39 | 0.943 |
| TabNet | Hyperband | 7.29 | 5.75 | 0.826 |

The eight boosted-tree models lie within 0.45 MPa of each other on the test set; TabNet is about twice as far from the
target. The CV and test rankings do not agree (CatBoost is best in CV, HistGradientBoosting on the test set), which with 95
test points is within sampling variation.

### 9.3 Regressor comparison under conformal prediction

Twenty two-sided conformal methods (6 from notebook 5, 14 from notebook 6), applied to each of the nine regressors. Ranked
by median valid width.

| Rank | Regressor | Test RMSE | Valid methods | Coverage range | Median width (MPa) | Relative width | Narrowest valid interval (width, coverage) |
|:-:|---|--:|:-:|:-:|--:|--:|---|
| 1 | **HistGradientBoosting** | 2.81 | 20 / 20 | 85.3% – 97.9% | **10.98** | **1.09** | COP (8.69, 89.5%) |
| 2 | LightGBM | 2.89 | 20 / 20 | 85.3% – 96.8% | 11.31 | 1.13 | PUNCC split CP (9.07, 85.3%) |
| 3 | NGBoost | 3.19 | 20 / 20 | 87.4% – 95.8% | 11.39 | 1.13 | MFCS Full (9.60, 90.5%) |
| 4 | CatBoost | 2.99 | 20 / 20 | 87.4% – 98.9% | 11.63 | 1.16 | PUNCC split CP (9.77, 87.4%) |
| 5 | PGBM | 3.04 | 20 / 20 | 88.4% – 97.9% | 11.77 | 1.17 | MFCS Full (10.00, 91.6%) |
| 6 | XGBoost | 2.89 | 20 / 20 | 88.4% – 96.8% | 11.84 | 1.18 | MAPIE CV+ (9.90, 89.5%) |
| 7 | GPBoost | 3.18 | 20 / 20 | 85.3% – 97.9% | 11.92 | 1.19 | Online J+aB (10.57, 92.6%) |
| 8 | GradientBoosting | 3.26 | 19 / 20 | 82.1% – 96.8% | 13.82 | 1.38 | MFCS Full (10.96, 90.5%) |
| 9 | TabNet | 5.75 | 19 / 20 | 70.5% – 100.0% | 24.79 | 2.47 | MAPIE split CP (20.75, 90.5%) |

The only invalid conformal intervals are AlphaNet conformal on GradientBoosting (82.1%) and on TabNet (70.5%).

**Which regressor is sharpest under each method.** For every conformal method, the regressor with the narrowest valid
interval and the runner-up:

| Method | Narrowest regressor | Width (MPa) | Coverage | Runner-up (width) |
|---|---|--:|--:|---|
| MAPIE split CP | LightGBM | 13.38 | 94.7% | NGBoost (14.24) |
| MAPIE CV+ | XGBoost | 9.90 | 89.5% | CatBoost (10.37) |
| MAPIE J+aB | HistGradientBoosting | 9.97 | 86.3% | LightGBM (10.25) |
| PUNCC split CP | LightGBM | 9.07 | 85.3% | CatBoost (9.77) |
| PUNCC CV+ | CatBoost | 11.19 | 93.7% | LightGBM (11.33) |
| PUNCC CQR | HistGradientBoosting | 23.00 | 96.8% | NGBoost (24.63) |
| NexCP split | HistGradientBoosting | 11.49 | 94.7% | LightGBM (12.02) |
| NexCP J+aB | HistGradientBoosting | 9.84 | 87.4% | NGBoost (10.47) |
| NexCP CV+ | HistGradientBoosting | 12.11 | 90.5% | LightGBM (13.05) |
| Online split | HistGradientBoosting | 10.94 | 92.6% | PGBM (11.45) |
| Online CV+ | HistGradientBoosting | 11.02 | 91.6% | LightGBM (11.71) |
| Online J+aB | HistGradientBoosting | 9.78 | 87.4% | NGBoost (10.25) |
| ACI | HistGradientBoosting | 9.49 | 90.5% | LightGBM (10.19) |
| FACI | HistGradientBoosting | 9.50 | 89.5% | LightGBM (9.50) |
| SF-OGD | CatBoost | 10.39 | 90.5% | XGBoost (11.56) |
| SAOCP | HistGradientBoosting | 11.27 | 90.5% | LightGBM (11.46) |
| MFCS split | HistGradientBoosting | 10.94 | 92.6% | PGBM (11.45) |
| MFCS full | NGBoost | 9.60 | 90.5% | PGBM (10.00) |
| COP | HistGradientBoosting | 8.69 | 89.5% | LightGBM (9.10) |
| AlphaNet conformal | CatBoost | 9.89 | 88.4% | PGBM (10.07) |
| *NexCP full* ¹ | *HistGradientBoosting* | *7.87* | *87.4%* | *PGBM (8.67)* |
| *Extreme CP (one-sided)* ² | *XGBoost* | *6.84* | *90.5%* | *LightGBM (6.95)* |

¹ Not ranked: for every regressor between 7 and 66 of the 95 intervals reach the edge of the label grid, so the widths are
lower bounds. ² One-sided interval; widths are not comparable with two-sided ones. Extreme CP is valid for all nine
regressors (90.5%–93.7%).

HistGradientBoosting gives the narrowest valid interval for 13 of the 20 ranked methods, CatBoost for 3, LightGBM for 2, and
XGBoost and NGBoost for 1 each.

### 9.4 Regressor comparison under model-based and learned intervals

90% intervals from quantile regression (notebook 2), CARD guided by each regressor (notebook 4) and adaptive coverage
policies (notebook 7). **Bold ✗** marks coverage below the 84.9% validity threshold.

| Regressor | QR coverage | QR width | QR crossing rate | CARD coverage | CARD width | ACP coverage | ACP width | ACP learned level |
|---|--:|--:|--:|--:|--:|--:|--:|--:|
| XGBoost | 90.5% | 31.67 | 79% | **60.0%** ✗ | 3.37 | 97.9% | 16.88 | 66.0% |
| LightGBM | **79.0%** ✗ | 10.35 | 98% | **57.9%** ✗ | 3.21 | 97.9% | 16.95 | 65.7% |
| CatBoost | **81.0%** ✗ | 10.63 | 98% | **48.4%** ✗ | 2.65 | 95.8% | 16.73 | 66.9% |
| GPBoost | **73.7%** ✗ | 8.05 | 99% | **27.4%** ✗ | 1.56 | 97.9% | 16.75 | 66.3% |
| HistGradientBoosting | **77.9%** ✗ | 11.34 | 100% | **50.5%** ✗ | 2.81 | 95.8% | 17.51 | 64.4% |
| GradientBoosting | **81.0%** ✗ | 12.54 | 100% | **56.8%** ✗ | 3.83 | 95.8% | 19.41 | 61.3% |
| NGBoost | **32.6%** ✗ | 1.89 | 0% | **41.1%** ✗ | 2.45 | 96.8% | 16.80 | 66.1% |
| PGBM | **16.8%** ✗ | 1.05 | 0% | **42.1%** ✗ | 2.25 | 97.9% | 17.02 | 66.1% |
| TabNet | 95.8% | 19.18 | 100% | **60.0%** ✗ | 6.55 | **84.2%** ✗ | 27.69 | 45.5% |

**Predictive distributions (notebook 3).** Gaussian 90% intervals from each distribution:

| Distribution | RMSE | Coverage | Width (MPa) | CRPS | PIT diagnosis |
|---|--:|--:|--:|--:|---|
| IBUG on LightGBM | 3.37 | 86.3% | 10.33 | 1.80 | — |
| IBUG on XGBoost | 3.25 | 95.8% | 16.35 | 1.84 | — |
| Treeffuser on GradientBoosting | 4.50 | 93.7% | 18.14 | 2.48 | Well calibrated |
| Treeffuser on GPBoost | 4.49 | 98.9% | 21.38 | 2.46 | Over-dispersed |
| Treeffuser on LightGBM | 5.34 | 100.0% | 33.41 | 3.27 | Over-dispersed |
| Treeffuser on XGBoost | 7.87 | 100.0% | 73.88 | 6.15 | Over-dispersed |
| Treeffuser on CatBoost | 12.04 | 100.0% | 320.12 | 23.33 | Over-dispersed |
| NGBoost (native) | 3.15 | **33.7%** ✗ | 1.88 | 1.87 | Under-dispersed |
| PGBM (native) | 3.03 | **21.1%** ✗ | 1.28 | 1.92 | Under-dispersed |

**Hyperspherical confidence mapping (notebook 8)** uses its own network rather than a tuned regressor: RMSE 5.62 MPa,
coverage 51.6% / 80.0% / 91.6% at 1σ / 2σ / 3σ (Gaussian ideal 68.3% / 95.4% / 99.7%), Spearman correlation between error
and predicted uncertainty 0.18.

### 9.5 Findings

1. **Choice of regressor.** With a conformal wrapper, the seven gradient-boosted models (HistGradientBoosting, LightGBM,
   NGBoost, CatBoost, PGBM, XGBoost, GPBoost) give valid intervals under all 20 methods, with median widths within 9% of each
   other (10.98–11.92 MPa). **HistGradientBoosting** is the sharpest overall and under 13 of the 20 methods. It also has the
   lowest test RMSE.
2. **Point accuracy carries over to interval width.** The two weakest point predictors give the widest intervals:
   GradientBoosting (median 13.82 MPa) and TabNet (24.79 MPa, 2.3 times HistGradientBoosting).
3. **Model-based intervals depend on the regressor and mostly under-cover.** Quantile regression is valid only for XGBoost
   (90.5%, but 31.7 MPa wide) and TabNet. The tree models produce crossing quantiles at 79–100% of test points, and NGBoost and
   PGBM quantiles cover only 33% and 17%. CARD under-covers whichever regressor guides it (27–60%).
4. **Predictive distributions.** NGBoost and PGBM are strongly over-confident. IBUG on LightGBM is the sharpest valid
   distributional interval (86.3%, 10.33 MPa), close to the conformal intervals. Treeffuser is over-dispersed for every
   regressor except GradientBoosting.
5. **ACP.** Every regressor except TabNet covers 95.8–97.9% of test points, at a learned level of 61–67%, with widths of
   16.7–19.4 MPa: 40–50% wider than the conformal intervals of the same regressor.
6. **Recommendation for Concrete.** Use HistGradientBoosting (or LightGBM) with a conformal wrapper. For example, ACI gives
   90.5% coverage at 9.49 MPa with HistGradientBoosting.

## 10. Additional datasets

The same pipeline was run on eight further engineering and environmental datasets. Each folder holds its data, notebooks,
executed copies and Excel results, organised like the Concrete folder.

| Dataset | Domain | Target (column) | Features | Train / test | Folder |
|---|---|---|--:|--:|---|
| Noise | Environmental acoustics | Equivalent continuous sound level (`Leq`) | 5 | 168 / 60 | [`noise_uncertainty_analysis`](examples/noise_uncertainty_analysis/) |
| Pile | Geotechnical engineering: piles | `Total` | 7 | 81 / 24 | [`pile_uncertainty_analysis`](examples/pile_uncertainty_analysis/) |
| Pile settlement | Geotechnical engineering: piles | Measured settlement (`S_exp`) | 8 | 371 / 186 | [`pile_settlement_uncertainty_analysis`](examples/pile_settlement_uncertainty_analysis/) |
| Scour | Hydraulic engineering | Scour depth (`Scour`) | 7 | 154 / 78 | [`scour_uncertainty_analysis`](examples/scour_uncertainty_analysis/) |
| Sediment load | Hydrology | Sediment load at time t (`St`) | 3 | 1,095 / 365 | [`sediment_load_uncertainty_analysis`](examples/sediment_load_uncertainty_analysis/) |
| Streamflow | Hydrology / water quality | Total dissolved solids (`TDS`) | 4 | 192 / 96 | [`streamflow_uncertainty_analysis`](examples/streamflow_uncertainty_analysis/) |
| Water quality (EWQI) | Groundwater quality | Pre-monsoon entropy water quality index (`Pre monsoon EWQI`) | 14 | 47 / 47 | [`water_quality_ewqi_uncertainty_analysis`](examples/water_quality_ewqi_uncertainty_analysis/) |
| Water quality (WQI) | Water quality | Water quality index (`WQI`) | 151 | 190 / 82 | [`water_quality_wqi_uncertainty_analysis`](examples/water_quality_wqi_uncertainty_analysis/) |

Dataset-specific settings are listed in [`docs/DATASET_PARAMETERS.md`](docs/DATASET_PARAMETERS.md).

## 11. Reproducibility

| Element | Practice |
|---|---|
| Data split | One fixed train/test split per dataset, stored in its `data/` folder |
| Model selection | 5-fold CV on the training set (`random_state=42`); the test set is not used |
| Tuned parameters | Saved once by notebook 1 (`best_params.pkl`, `best_params.json`) and loaded by notebooks 2–7 |
| Executed notebooks | Stored with all cell outputs as `EXECUTED_*.ipynb` next to the Excel outputs |
| Run identifiers | Every Kaggle run is named `REG<DD>_<MM>_<YYYY>_<HH>_<MM>` and listed in [`docs/run_registry.tsv`](docs/run_registry.tsv); each `results/README.md` names its run |
| Code changes | Every change against v1.2 is marked `# FIX <id>` and documented in [`docs/CHANGELOG_FIXES.md`](docs/CHANGELOG_FIXES.md) |
| Result summaries | [`tools/reports/build_summary.py`](tools/reports/README.md) extracts the headline numbers of every notebook into `tools/reports/summary/<dataset>.json` without re-running any model |
| Known non-determinism | Optuna's search is not bit-reproducible; neural models (TabNet, CARD, ACP, HCM) depend on their seeds |

**Release history**

| Version | Date | Description |
|---|---|---|
| **v2.0.0** | November 2026 | Correctness release: every method verified against its publication and reference code; model selection without the test set; all notebooks re-executed on Kaggle |
| v1.2 | June 2026 | Colab notebooks for nine datasets (branch `version1.2`) |
| v1.1 | April 2026 | Branch `version-1.1` |
| v1.0 | March 2026 | Branch `version-1.0` |

Outputs saved in the v1.2 notebooks were not produced on their stated datasets, so v1.2 numbers should not be compared
with v2.0.

## 12. Limitations

| # | Limitation | Consequence |
|:-:|---|---|
| 1 | One train/test split and one run per notebook | No confidence intervals across splits or seeds; with 95 Concrete test points, differences of a few percentage points in coverage are not significant |
| 2 | Online methods process the test rows in their stored order | Their widths partly reflect learning from earlier test labels |
| 3 | All guarantees are marginal | A method can cover 90% overall and less for extreme target values |
| 4 | Optuna search is not bit-reproducible; neural models depend on seeds | Re-running gives slightly different numbers; the stored tuning run is the one every notebook uses |
| 5 | Kaggle's 12-hour session limit | Not run: NB6 TabNet and PGBM on sediment load (and NexCP full there); NB6 NGBoost and PGBM on pile settlement |
| 6 | NexCP full uses a finite label grid | Intervals that reach the grid edge are lower bounds and are not ranked |
| 7 | MFCS has no identified published source | It is reported as implemented in the original notebooks |

## 13. Roadmap

The notebooks are a research pipeline whose methods cannot be imported into other code. The next stage is **`reguq`**
(regression uncertainty quantification), a Python package that provides the verified methods of this benchmark as a tested,
documented library:

- a common interface (`fit`, `calibrate`, `predict_interval`, `predict_distribution`) for every method family, wrapping any
  scikit-learn-compatible regressor;
- the evaluation protocol of Section 8 (coverage with sampling tolerance, relative width, CRPS, PIT, crossing rate) and the
  regressor comparison as functions;
- tests that check each method against the authors' reference implementation and against its theoretical coverage on
  synthetic data;
- repeated random splits with confidence intervals, and conditional-coverage diagnostics.

## 14. Citation

If you use this benchmark, please cite it (GitHub's *Cite this repository* button uses [`CITATION.cff`](CITATION.cff)),
together with the original publications of the methods you use:

```bibtex
@misc{selwal_pal_regression_uq_2026,
  author       = {Selwal, Danesh and Pal, Mahesh},
  title        = {Regression Uncertainty Quantification Benchmark: Over Thirty Methods on
                  Nine Engineering and Environmental Datasets},
  year         = {2026},
  howpublished = {\url{https://github.com/DaneshSelwal/Regression-UQ-Benchmark}},
  note         = {Version 2.0.0}
}
```

## 15. License

The code and documentation are released under the [MIT License](LICENSE). The datasets are provided for research
reproducibility; please acknowledge the original data sources when publishing results obtained on them.

## 16. Acknowledgements and contact

**Authors.** Danesh Selwal and Dr. Mahesh Pal.

**Acknowledgements.** Computation was carried out on Kaggle Notebooks; the original notebooks were developed on Google
Colab. The benchmark builds on the open-source libraries listed in Section 4 and on the reference implementations cited in
[`references/REFERENCES.md`](references/REFERENCES.md).

**Contact.** Please open an issue on this repository for questions or corrections.
