<div align="center">

# Regression Uncertainty Quantification Benchmark

### A reproducible benchmark of more than thirty uncertainty-quantification methods for tabular regression, evaluated with nine tuned regressors on nine engineering and environmental datasets

[![License: MIT](https://img.shields.io/badge/license-MIT-2a78d6)](LICENSE)
[![Release](https://img.shields.io/badge/release-v2.0.0-1baf7a)](https://github.com/DaneshSelwal/Regression-UQ-Benchmark/releases)
![Python](https://img.shields.io/badge/python-3-2a78d6)
![Datasets](https://img.shields.io/badge/datasets-9-2a78d6)
![Base models](https://img.shields.io/badge/regressors-10-2a78d6)
![UQ methods](https://img.shields.io/badge/UQ%20methods-30%2B-eb6834)
![Compute](https://img.shields.io/badge/compute-Kaggle%20CPU-20BEFF)
[![Cite](https://img.shields.io/badge/cite-CITATION.cff-b31b1b)](CITATION.cff)

</div>

<sub>Repository formerly named <code>Regression_Uncertainty_Quantification_Analysis</code>; former links redirect here.</sub>

> **Abstract.** A regression model returns a point prediction without indicating how far that prediction can be trusted. In
> engineering and environmental practice, where predictions inform design margins and regulatory decisions, a prediction
> interval with a known coverage is often more useful than the point prediction itself. This repository provides a
> reproducible benchmark of **more than thirty uncertainty-quantification (UQ) methods**, applied to **ten tuned regressors** on
> **nine engineering and environmental datasets**. The methods cover quantile regression, parametric and generative predictive
> distributions, split, cross-validation, bootstrap, weighted and online conformal prediction, learned coverage policies and
> single-network uncertainty. Every 90% prediction interval is assessed on held-out data for **validity** (coverage within the
> sampling tolerance of 90%) and **sharpness** (width). Every reported number is produced by an executed notebook stored in
> this repository, with the corresponding Kaggle run recorded in a run registry.

---

## Contents

1. [Motivation and scope](#1-motivation-and-scope)
2. [Key features](#2-key-features)
3. [Pipeline](#3-pipeline)
4. [Datasets](#4-datasets)
5. [Models and methods](#5-models-and-methods)
6. [Evaluation protocol](#6-evaluation-protocol)
7. [Getting started](#7-getting-started)
8. [Reproducibility and provenance](#8-reproducibility-and-provenance)
9. [Repository structure](#9-repository-structure)
10. [Version history](#10-version-history)
11. [Limitations](#11-limitations)
12. [Roadmap: the `reguq` library](#12-roadmap-the-reguq-library)
13. [Citation](#13-citation)
14. [License, data and acknowledgements](#14-license-data-and-acknowledgements)

---

## 1. Motivation and scope

Uncertainty-quantification methods for regression differ in what they assume and in what they guarantee. Quantile regression
and predictive distributions are correct only if the fitted model is correct; conformal prediction guarantees coverage in
finite samples under exchangeability, whatever the model; online and weighted conformal methods relax exchangeability; learned
coverage policies and single-network methods trade guarantees for flexibility. Published comparisons are typically carried out
on large benchmark collections, whereas engineering and environmental datasets are small (tens to a few hundred test points),
heterogeneous and frequently heavy-tailed.

This benchmark addresses the following research questions on such data:

| # | Research question |
|:-:|---|
| RQ1 | Do model-based intervals (quantile regression, predictive distributions) attain their nominal coverage? |
| RQ2 | Do conformal methods attain valid coverage for every base model and dataset? |
| RQ3 | Among valid methods, which yields the sharpest intervals, and does the answer depend on the dataset? |
| RQ4 | What do learned coverage levels (ACP) and single-network uncertainty (HCM) offer relative to conformal prediction? |

**Scope.** Tabular, single-output regression; a fixed train/test split per dataset; a nominal coverage of 90% (α = 0.1) for
every interval method except HCM, which reports Gaussian 1/2/3σ intervals.

---

## 2. Key features

- **Breadth.** More than thirty UQ methods from five families, each wrapped around up to nine tuned base regressors.
- **Verified implementations.** Every method was checked against its original publication and, where available, the
  authors' reference code; each correction is identified in the code (`# FIX <id>`) and documented in
  [`docs/CHANGELOG_FIXES.md`](docs/CHANGELOG_FIXES.md) with the state before, the reason and the state after.
- **No test-set leakage.** Hyperparameters are selected by 5-fold cross-validation on the training set only; the test set is
  used once, for evaluation.
- **Sample-size-aware evaluation.** Validity is judged with a one-sided binomial test, so that small test sets are not
  penalised for sampling variation.
- **Full provenance.** Each executed notebook, its Excel outputs and its Kaggle run identifier are stored with the results.
- **Portability.** One generalised notebook per stage; a new dataset requires four settings (folder name, file names,
  feature names, target label).

---

## 3. Pipeline

The pipeline consists of eight notebooks. Notebook 1 produces the tuned hyperparameters (`best_params.pkl`), which every
subsequent notebook loads; notebooks 2–8 are otherwise independent of one another.

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
| 3 | Predictive distributions | NGBoost, PGBM, IBUG, Treeffuser | μ̂, σ̂, NLL, CRPS (Hersbach decomposition), PIT, calibration | [NB3](notebooks/03_probabilistic_distribution/README.md) |
| 4 | CARD | Guided diffusion, 1,000 samples per point | Predictive distribution, CRPS, calibration | [NB4](notebooks/04_probabilistic_distribution_card/README.md) |
| 5 | Conformal (MAPIE, PUNCC) | Split, CV+, J+aB, CQR | Intervals, coverage, width | [NB5](notebooks/05_conformal_mapie_puncc/README.md) |
| 6 | Conformal (NexCP, online, MFCS) | NexCP (4 variants), ACI, FACI, SF-OGD, SAOCP, online split/CV+/J+aB, MFCS, COP, AlphaNet, Extreme CP | Intervals per family; generalised Pareto tail fits | [NB6](notebooks/06_conformal_nexcp_online/README.md) |
| 7 | Adaptive coverage policies | ACP with e-values; λ ∈ {10, 20, 50} | Learned level, coverage, width | [NB7](notebooks/07_adaptive_coverage_policies/README.md) |
| 8 | Hyperspherical confidence mapping | HCM (single network) | Coverage at 1/2/3σ, error–uncertainty correlation | [NB8](notebooks/08_hyperspherical_confidence_mapping/README.md) |

---

## 4. Datasets

All datasets are tabular with a single continuous target in the last column of `train.csv` and `test.csv`. They range from 47
to 1,095 training rows and from 3 to 151 input features.

| Dataset | Domain | Target (column) | Features | Train / test | Folder |
|---|---|---|--:|--:|---|
| Concrete | Civil engineering: concrete mix design | Compressive strength (`str`, MPa) | 7 | 189 / 95 | [`concrete_uncertainty_analysis`](examples/concrete_uncertainty_analysis/) |
| Noise | Environmental acoustics | Equivalent continuous sound level (`Leq`) | 5 | 168 / 60 | [`noise_uncertainty_analysis`](examples/noise_uncertainty_analysis/) |
| Pile | Geotechnical engineering: piles | `Total` | 7 | 81 / 24 | [`pile_uncertainty_analysis`](examples/pile_uncertainty_analysis/) |
| Pile settlement | Geotechnical engineering: piles | Measured settlement (`S_exp`) | 8 | 371 / 186 | [`pile_settlement_uncertainty_analysis`](examples/pile_settlement_uncertainty_analysis/) |
| Scour | Hydraulic engineering | Scour depth (`Scour`) | 7 | 154 / 78 | [`scour_uncertainty_analysis`](examples/scour_uncertainty_analysis/) |
| Sediment load | Hydrology | Sediment load at time t (`St`) | 3 | 1,095 / 365 | [`sediment_load_uncertainty_analysis`](examples/sediment_load_uncertainty_analysis/) |
| Streamflow | Hydrology / water quality | Total dissolved solids (`TDS`) | 4 | 192 / 96 | [`streamflow_uncertainty_analysis`](examples/streamflow_uncertainty_analysis/) |
| Water quality (EWQI) | Groundwater quality | Pre-monsoon entropy water quality index (`Pre monsoon EWQI`) | 14 | 47 / 47 | [`water_quality_ewqi_uncertainty_analysis`](examples/water_quality_ewqi_uncertainty_analysis/) |
| Water quality (WQI) | Water quality | Water quality index (`WQI`) | 151 | 190 / 82 | [`water_quality_wqi_uncertainty_analysis`](examples/water_quality_wqi_uncertainty_analysis/) |

Dataset-specific settings, and settings whose effect depends on dataset size, are listed in
[`docs/DATASET_PARAMETERS.md`](docs/DATASET_PARAMETERS.md).

---

## 5. Models and methods

### 5.1 Base regressors

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

### 5.2 Uncertainty-quantification methods

| Family | Method | Reference | Implementation | Guarantee |
|---|---|---|---|---|
| Quantile regression | Pinball-loss quantile models, monotone rearrangement | Koenker & Bassett (1978); Chernozhukov et al. (2010) | Native model objectives | None |
| Predictive distribution | NGBoost | Duan et al. (2020) | `ngboost` | None |
| | PGBM | Sprangers et al. (2021) | `pgbm` | None |
| | IBUG | Brophy & Lowd (2022) | Notebook implementation | None |
| | Treeffuser | Beltran-Velez et al. (2024) | `treeffuser` (fork) | None |
| | CARD | Han, Zheng & Zhou (2022) | Re-implemented from the official code | None |
| Conformal: split and resampling | Split conformal | Vovk et al. (2005); Lei et al. (2018) | MAPIE, PUNCC | Finite-sample, marginal |
| | CV+ | Barber et al. (2021) | MAPIE, PUNCC | Finite-sample (1 − 2α) |
| | Jackknife+-after-bootstrap (J+aB) | Kim, Xu & Barber (2020) | MAPIE | Finite-sample (1 − 2α) |
| | Conformalized quantile regression (CQR) | Romano, Patterson & Candès (2019) | MAPIE, PUNCC | Finite-sample, marginal |
| Conformal: weighted | NexCP (split, full, CV+, J+aB) | Barber et al. (2023) | Notebook implementation | Coverage gap bounded by weighted total variation |
| Conformal: online | ACI | Gibbs & Candès (2021) | Notebook implementation | Long-run coverage |
| | FACI | Gibbs & Candès (2024) | Ported from reference code | Long-run coverage |
| | SF-OGD, SAOCP | Bhatnagar et al. (2023) | Ported from reference code | Long-run coverage |
| | Online split / CV+ / J+aB, MFCS | — | Notebook implementation | Long-run coverage |
| | COP | Hu et al. (2026) | Following the official code | Long-run coverage |
| Conformal: tail | Extreme CP (one-sided) | Pasche, Lam & Engelke (2025) | Following the official code | Tail coverage |
| Learned coverage | AlphaNet / ACP with e-values | Gauthier, Bach & Jordan (2025, 2026) | Following the official code | At the learned level |
| Single network | HCM | Choi et al. (2026) | Following the official code | None (calibrated scale) |

Full bibliographic entries, with links to the reference implementations, are given in
[`references/REFERENCES.md`](references/REFERENCES.md). The derivation of each method is given in the
[method guides](notebooks/README.md).

### 5.3 Split conformal prediction

The model is fitted on part of the training data, and the absolute residuals $`s_i=|y_i-\hat f(x_i)|`$ are computed on the
remaining $`m`$ calibration points. With

```math
\hat q = s_{(k)},\qquad k=\bigl\lceil (m+1)(1-\alpha)\bigr\rceil,\qquad \alpha=0.1,
```

the interval for a new point is $`\hat C(x)=[\hat f(x)-\hat q,\ \hat f(x)+\hat q]`$. Under exchangeability the residual of the new
point is equally likely to occupy any rank among the $`m+1`$ residuals, so that
$`\Pr\{Y\in\hat C(X)\}\ge 1-\alpha`$ irrespective of the model. The other conformal methods modify one element of this
construction: the reuse of training points for calibration (CV+, J+aB), the score (CQR), the weighting of residuals (NexCP),
or the sequential update of the radius (online methods).

---

## 6. Evaluation protocol

### 6.1 Metrics

| Metric | Definition | Desired value |
|---|---|---|
| **Coverage** | Share of test points inside their interval | ≥ 90% within sampling tolerance |
| **Width** | Mean difference between upper and lower bound, in target units | Small, given valid coverage |
| **Relative width** | Width divided by the standard deviation of the test target | Small; comparable across datasets |
| **CRPS** | Continuous ranked probability score; the pinball loss integrated over all quantile levels | Low |
| **NLL** | Negative log-likelihood of the observation under the predictive distribution | Low |
| **PIT** | Predicted CDF evaluated at the observation | Uniform histogram |
| **Crossing rate** | Share of points whose separately fitted quantiles are not monotone | 0 |
| **Spearman (error, σ̂)** | Rank correlation between predicted uncertainty and absolute error | > 0 |
| **RMSE** | Root mean squared error of the point prediction | Low |

### 6.2 Validity of a coverage value

The number of covered test points is binomial, so the coverage of a correctly calibrated 90% method has standard error
$`\sqrt{0.09/n}`$. A method is considered **valid** when its coverage is not significantly below 90% in a one-sided test at the
5% level:

```math
\hat c\ \ge\ 0.9-1.645\sqrt{\frac{0.09}{n_{\text{test}}}} .
```

| Test points | 24 | 47 | 95 | 186 | 365 |
|---|--:|--:|--:|--:|--:|
| Validity threshold | 79.9% | 82.8% | 84.9% | 86.4% | 87.4% |

### 6.3 Recommended method

A method wraps up to nine base models. It is **recommended** on a dataset if it has the smallest median relative width among
methods that are valid on at least 75% of the models they ran on, ran on at least three models, and report no width that is
only a lower bound.

---

## 7. Getting started

### 7.1 Requirements

The notebooks were executed with Python 3 on Kaggle CPU sessions. The principal packages and the versions used:

| Purpose | Packages (version) |
|---|---|
| Core | numpy (1.26.4), scikit-learn (1.6.1), torch (2.9.0) |
| Base regressors | xgboost (3.1.2), lightgbm (4.6.0), catboost (1.2.8), gpboost (1.6.1), ngboost (0.5.8), pgbm (2.2.0), pytorch-tabnet2 (4.5.3) |
| Tuning | optuna (4.6.0), optunahub (0.4.0), cmaes (0.12.0) |
| Interpretability | shap (0.44.0), lime (0.2.0.1), interpret (0.7.4) |
| Conformal prediction | mapie (0.6.0), puncc (0.8.0) |
| Distributions and scoring | treeffuser (fork), properscoring (0.1) |
| Output | openpyxl (3.1.5), XlsxWriter (3.2.9), matplotlib, seaborn (0.13.2), plotly (5.24.1) |

The first cell of every notebook (`FIX K1`) installs the packages missing from the environment at these versions.

### 7.2 Running on Kaggle

1. Attach the Kaggle dataset `daneshselwal/reguq-uq-data`, or upload `examples/<dataset>_uncertainty_analysis/data/`.
2. Run **notebook 1** first. The `FIX K1` cell detects the Kaggle environment, copies the CSV files into the expected layout
   and installs missing packages.
3. Run **notebooks 2–8**, adding the output of the notebook 1 run as an input; the `FIX K2` cell locates `best_params.pkl`.
4. For batch execution, [`tools/kaggle/`](tools/kaggle/README.md) wraps a notebook in a papermill job, assigns a run name of
   the form `REG<DD>_<MM>_<YYYY>_<HH>_<MM>` and records it in [`docs/run_registry.tsv`](docs/run_registry.tsv). Notebook 6 is run
   as one job per base model to remain within the 12-hour session limit.

### 7.3 Running on Google Colab

Place the dataset folder at `MyDrive/Regression/<dataset>_uncertainty_analysis/` and run the notebooks in the order above. The
original Colab installation block is retained in the first cell and is executed only on Colab.

### 7.4 Applying the pipeline to a new dataset

1. Provide `train.csv` and `test.csv` with the target in the last column.
2. Copy the notebooks from [`notebooks/`](notebooks/README.md).
3. In the first cells, set the dataset folder name, the data file names, `feature_names` and `TARGET_LABEL`.
4. Run notebook 1, then notebooks 2–8.

---

## 8. Reproducibility and provenance

| Element | Practice |
|---|---|
| Data split | A fixed train/test split per dataset, stored in `examples/<dataset>_uncertainty_analysis/data/` |
| Model selection | 5-fold cross-validation on the training set (`KFold(shuffle=True, random_state=42)`); the test set is not used |
| Tuned parameters | Saved once by notebook 1 (`best_params.pkl`, `best_params.json`) and loaded by every later notebook |
| Executed notebooks | Stored with all cell outputs as `EXECUTED_*.ipynb` next to the Excel outputs |
| Run identifiers | Every Kaggle run is named `REG<DD>_<MM>_<YYYY>_<HH>_<MM>` and listed in [`docs/run_registry.tsv`](docs/run_registry.tsv) |
| Code changes | Every change against v1.2 is marked `# FIX <id>` and documented in [`docs/CHANGELOG_FIXES.md`](docs/CHANGELOG_FIXES.md) |
| Known non-determinism | Optuna's search is not bit-reproducible; neural models (TabNet, CARD, ACP, HCM) depend on their seeds |

---

## 9. Repository structure

```
.
├── README.md
├── CITATION.cff
├── LICENSE
├── docs/
│   ├── CHANGELOG_FIXES.md        every v2.0 correction: before, reason, after
│   ├── DATASET_PARAMETERS.md     dataset-specific settings
│   ├── KAGGLE_RUNS.md            construction and execution of the Kaggle jobs
│   └── run_registry.tsv          every Kaggle run: code name, notebook, time, status
├── examples/<dataset>_uncertainty_analysis/
│   ├── data/                     train.csv, test.csv
│   └── <phase>/                  notebook and results/ (EXECUTED_*.ipynb, *.xlsx, run record)
├── notebooks/                    generalised pipeline (no outputs) and a method guide per notebook
├── references/                   REFERENCES.md and key papers
└── tools/
    ├── kaggle/                   Kaggle job builder and push queue
    └── reports/                  report and figure generators
```

---

## 10. Version history

| Version | Date | Description |
|---|---|---|
| **v2.0.0** | November 2026 | Correctness release: every method verified against its publication and reference code; model selection without the test set; all notebooks re-executed on Kaggle for nine datasets |
| v1.2 | June 2026 | Colab notebooks for nine datasets (branch `version1.2`) |
| v1.1 | April 2026 | Branch `version-1.1` |
| v1.0 | March 2026 | Branch `version-1.0` |

**Principal changes in v2.0**

| Area | Change |
|---|---|
| Tuning | Model selection by 5-fold CV on the training set (previously on the test set); tuned settings saved and loaded by every later notebook |
| Quantile regression | Quantile crossing measured and removed; CRPS with correct level weights; TabNet early stopping on a validation split |
| Distributions | Hersbach CRPS decomposition; IBUG variance no longer square-rooted twice; Treeffuser score model uses squared error |
| CARD | Re-implemented following the official CARD code |
| MAPIE / PUNCC | Correct interval shapes; genuine split conformal and J+aB; CQR with quantile models; a single α = 0.1 |
| NexCP / online | Out-of-fold calibration residuals; NexCP weighted quantile as published; weighted full, CV+ and J+aB; SF-OGD, SAOCP and FACI ported from the reference code; ACI added; COP and Extreme CP following their official code |
| ACP | λ no longer selected by comparing losses across λ; coverage reported against the learned level |
| HCM | Silent fallback to synthetic data removed; training schedule extended so that the network converges |
| Infrastructure | Execution on Kaggle or Colab; executed notebooks stored with the results; one NB6 job per model; run registry |

Outputs saved in the v1.2 notebooks were not produced on their stated datasets; v1.2 numbers should therefore not be compared
with v2.0.

---

## 11. Limitations

| # | Limitation | Consequence |
|:-:|---|---|
| 1 | One train/test split and one run per notebook | No confidence intervals across splits or seeds; small test sets detect only large departures from 90% |
| 2 | Online methods process the test rows in their stored order | Their widths partly reflect learning from earlier test labels |
| 3 | All guarantees are marginal | A method can cover 90% overall and less for extreme target values |
| 4 | Optuna search is not bit-reproducible; neural models depend on seeds | Re-running yields slightly different numbers; the stored tuning run is the one used by every notebook |
| 5 | Kaggle's 12-hour session limit | Not run: NB6 TabNet and PGBM on sediment load (and NexCP Full there); NB6 NGBoost and PGBM on pile settlement |
| 6 | MFCS has no identified published source | It is reported as implemented in the original notebooks |

---

## 12. Roadmap: the `reguq` library

The notebooks constitute a research pipeline whose methods cannot be imported into other code. The next stage of the project
is **`reguq`** (regression uncertainty quantification), a Python package that provides the verified methods of this benchmark
as a tested and documented library:

- a common interface (`fit`, `calibrate`, `predict_interval`, `predict_distribution`) for every method family, wrapping any
  scikit-learn-compatible regressor;
- the evaluation protocol of Section 6 (coverage with sampling tolerance, relative width, CRPS, PIT, crossing rate) and the
  recommendation rule as functions;
- tests that verify each method against the authors' reference implementation and against its theoretical coverage on
  synthetic data;
- repeated random splits with confidence intervals, and conditional-coverage diagnostics.

---

## 13. Citation

If you use this benchmark, please cite it (GitHub's *Cite this repository* button uses [`CITATION.cff`](CITATION.cff)),
together with the original publications of the methods used:

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

---

## 14. License, data and acknowledgements

**License.** The code and documentation are released under the [MIT License](LICENSE).

**Data.** The datasets are provided for research reproducibility. Users should acknowledge the original data sources when
publishing results obtained on them.

**Acknowledgements.** Computation was carried out on Kaggle Notebooks; the original notebooks were developed on Google Colab.
The benchmark builds on the open-source libraries listed in Section 7.1 and on the reference implementations cited in
[`references/REFERENCES.md`](references/REFERENCES.md).

**Authors.** Danesh Selwal and Dr. Mahesh Pal.
