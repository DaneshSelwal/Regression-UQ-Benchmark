<div align="center">

# Regression Uncertainty Quantification Benchmark

### Over 30 uncertainty-quantification methods evaluated with 9 tuned regressors on 9 engineering and environmental datasets

[![License: MIT](https://img.shields.io/badge/license-MIT-2a78d6)](LICENSE)
![Datasets](https://img.shields.io/badge/datasets-9-2a78d6)
![Base models](https://img.shields.io/badge/tuned%20regressors-9%2B1-2a78d6)
![UQ methods](https://img.shields.io/badge/UQ%20methods-30%2B-eb6834)
![Compute](https://img.shields.io/badge/compute-Kaggle%20CPU-20BEFF)
[![Version](https://img.shields.io/badge/version-2.0-1baf7a)](https://github.com/DaneshSelwal/Regression-UQ-Benchmark/releases)

</div>

<sub>Repository formerly named <code>Regression_Uncertainty_Quantification_Analysis</code>; former links redirect here.</sub>

> **Abstract.** A regression model returns a point prediction without indicating how far that prediction can be trusted.
> This repository evaluates **more than thirty uncertainty-quantification (UQ) methods**, applied to **nine tuned regressors**,
> on **nine engineering and environmental datasets**. The methods cover quantile regression, parametric and generative
> predictive distributions (NGBoost, PGBM, IBUG, Treeffuser, CARD), split, cross-validation, bootstrap, weighted and online
> conformal prediction, learned coverage policies and single-network uncertainty. Every 90% prediction interval is assessed
> on held-out data for validity (coverage within sampling tolerance of 90%) and sharpness (width). All results are produced
> by executed Kaggle notebooks stored in this repository.

---

## Contents

1. [Overview and pipeline](#1-overview-and-pipeline)
2. [Datasets](#2-datasets)
3. [Methods](#3-methods)
4. [Evaluation protocol](#4-evaluation-protocol)
5. [Reproducing the results](#5-reproducing-the-results)
6. [Repository structure](#6-repository-structure)
7. [Changes in version 2.0](#7-changes-in-version-20)
8. [Limitations](#8-limitations)
9. [Future work: the `reguq` library](#9-future-work-the-reguq-library)
10. [Citation, license and authors](#10-citation-license-and-authors)

---

## 1. Overview and pipeline

The pipeline consists of eight notebooks, applied without modification (apart from paths, file names and column names) to
every dataset.

| # | Notebook | Methods | Principal outputs | Method guide |
|:-:|---|---|---|---|
| 1 | Hyperparameter tuning | Optuna AutoSampler (TPE), 5-fold CV, 8 pruners; SHAP, LIME, EBM | Tuned settings (`best_params.json`), CV and test RMSE | [NB1](notebooks/01_hyperparameter_tuning/README.md) |
| 2 | Quantile regression | 13 quantile levels per model | Coverage, width, CRPS, crossing rate | [NB2](notebooks/02_quantile_regression/README.md) |
| 3 | Predictive distributions | NGBoost, PGBM, IBUG, Treeffuser | μ̂, σ̂, NLL, CRPS (Hersbach), PIT, calibration | [NB3](notebooks/03_probabilistic_distribution/README.md) |
| 4 | CARD | Guided diffusion, 1,000 samples per point | Predictive distribution, CRPS, calibration | [NB4](notebooks/04_probabilistic_distribution_card/README.md) |
| 5 | Conformal (MAPIE, PUNCC) | Split, CV+, J+aB, CQR | Intervals, coverage, width | [NB5](notebooks/05_conformal_mapie_puncc/README.md) |
| 6 | Conformal (NexCP, online, MFCS) | NexCP (4 variants), ACI, FACI, SF-OGD, SAOCP, online split/CV+/J+aB, MFCS, COP, AlphaNet, Extreme CP | One workbook sheet per family; GPD fits | [NB6](notebooks/06_conformal_nexcp_online/README.md) |
| 7 | Adaptive coverage policies | ACP with e-values; λ ∈ {10, 20, 50} | Learned level, coverage, width | [NB7](notebooks/07_adaptive_coverage_policies/README.md) |
| 8 | Hyperspherical confidence mapping | HCM (single network) | Coverage at 1/2/3σ, error–uncertainty correlation | [NB8](notebooks/08_hyperspherical_confidence_mapping/README.md) |

Base models: RandomForest (tuning only), GradientBoosting, XGBoost, LightGBM, GPBoost, CatBoost, HistGradientBoosting, NGBoost,
PGBM and TabNet.

---

## 2. Datasets

Each dataset folder in [`examples/`](examples/) contains the data, the eight notebooks with this dataset's paths, and for each
notebook a `results/` folder with the executed notebook and its Excel outputs.

| Dataset | Train / test | Folder |
|---|---|---|
| Concrete | 189 / 95 | [`examples/concrete_uncertainty_analysis/`](examples/concrete_uncertainty_analysis/) |
| Noise | 168 / 60 | [`examples/noise_uncertainty_analysis/`](examples/noise_uncertainty_analysis/) |
| Pile | 81 / 24 | [`examples/pile_uncertainty_analysis/`](examples/pile_uncertainty_analysis/) |
| Pile settlement | 371 / 186 | [`examples/pile_settlement_uncertainty_analysis/`](examples/pile_settlement_uncertainty_analysis/) |
| Scour | 154 / 78 | [`examples/scour_uncertainty_analysis/`](examples/scour_uncertainty_analysis/) |
| Sediment load | 1095 / 365 | [`examples/sediment_load_uncertainty_analysis/`](examples/sediment_load_uncertainty_analysis/) |
| Streamflow | 192 / 96 | [`examples/streamflow_uncertainty_analysis/`](examples/streamflow_uncertainty_analysis/) |
| Water quality (EWQI) | 47 / 47 | [`examples/water_quality_ewqi_uncertainty_analysis/`](examples/water_quality_ewqi_uncertainty_analysis/) |
| Water quality (WQI) | 190 / 82 | [`examples/water_quality_wqi_uncertainty_analysis/`](examples/water_quality_wqi_uncertainty_analysis/) |

---

## 3. Methods

| Family | Principle | Output | Coverage guarantee |
|---|---|---|---|
| **Quantile regression** | Separate models estimate the lower and upper conditional quantiles by minimising the pinball loss | interval | none (holds only if both quantile models are correct) |
| **Predictive distributions**: NGBoost, PGBM, IBUG, Treeffuser, CARD | A parametric or generative model of the conditional distribution of the target | distribution → interval | none (holds only if the noise model is correct) |
| **Conformal prediction**: split, CV+, J+aB, CQR, NexCP, online (ACI, FACI, SF-OGD, SAOCP, MFCS), COP | Calibrates the interval radius on held-out residuals so that it exceeds 90% of them | interval | **finite-sample**: ≥ 90% on average under exchangeability (online methods: long-run, without assumptions on the data) |
| **Learned level / single network**: ACP, HCM | A network learns the miscoverage level (ACP) or derives uncertainty from the geometry of its output (HCM) | interval at a learned level / σ̂ | ACP: at its learned level; HCM: none |

**Split conformal prediction.** The model is fitted on part of the training data, and the absolute residuals
$`s_i=|y_i-\hat f(x_i)|`$ are computed on the remaining $`m`$ calibration points. With

```math
\hat q = s_{(k)},\qquad k=\bigl\lceil (m+1)(1-\alpha)\bigr\rceil,\qquad \alpha=0.1,
```

the interval for a new point is $`\hat C(x)=[\hat f(x)-\hat q,\ \hat f(x)+\hat q]`$. Under exchangeability the residual of the new
point is equally likely to occupy any rank among the $`m+1`$ residuals, so it does not exceed $`\hat q`$ with probability at
least $`1-\alpha`$, irrespective of the model. The other conformal methods modify one element of this construction: the reuse
of training points for calibration (CV+, J+aB), the score (CQR), the weighting of residuals (NexCP), or the sequential update
of the radius (online methods). Each method is derived in the [method guides](notebooks/README.md).

---

## 4. Evaluation protocol

### 4.1 Metrics

| Metric | Definition | Desired value |
|---|---|---|
| **Coverage** | Share of test points inside their interval | ≥ 90% within sampling tolerance |
| **Width** | Mean difference between upper and lower bound, in target units | small, given valid coverage |
| **Relative width** | Width divided by the standard deviation of the test target | small; comparable across datasets |
| **CRPS** | Continuous ranked probability score; equals the pinball loss integrated over all quantile levels | low |
| **PIT** | Predicted CDF evaluated at the observation; uniform if calibrated | uniform histogram |
| **Crossing rate** | Share of points whose separately fitted quantiles are not monotone | 0 |
| **Spearman (error, σ̂)** | Rank correlation between predicted uncertainty and absolute error | > 0 |

### 4.2 Validity of a coverage value

The number of covered test points is binomial, so the coverage of a correctly calibrated 90% method has standard error
$`\sqrt{0.09/n}`$. A method is considered **valid** when its coverage is not significantly below 90% in a one-sided test at the
5% level:

```math
\hat c\ \ge\ 0.9-1.645\sqrt{\frac{0.09}{n_{\text{test}}}} .
```

The threshold is 79.9% for 24 test points (pile), 84.9% for 95 (concrete) and 87.4% for 365 (sediment load).

### 4.3 Recommended method

A method wraps up to nine base models. It is **recommended** on a dataset if it has the smallest median relative width among
methods that are valid on at least 75% of the models they ran on, ran on at least three models, and report no width that is
only a lower bound.

---

## 5. Reproducing the results

All results were computed on **Kaggle CPU sessions**; no model was trained locally.

1. **Data.** Attach the Kaggle dataset `daneshselwal/reguq-uq-data` or upload `examples/<dataset>_uncertainty_analysis/data/`.
2. **Notebook 1 first.** The `FIX K1` cell detects the Kaggle environment, copies the CSV files into the expected layout and
   installs only missing packages at pinned versions.
3. **Notebooks 2–8.** Add the output of the tuning run as an input; the `FIX K2` cell locates `best_params.pkl`.
4. **Batch execution.** [`tools/kaggle/`](tools/kaggle/README.md) wraps a notebook in a papermill job, names each run
   `REG<DD>_<MM>_<YYYY>_<HH>_<MM>` and records it in [`docs/run_registry.tsv`](docs/run_registry.tsv). Notebook 6 runs as one job
   per base model to remain within the 12-hour session limit.

The notebooks also run on **Google Colab** with the dataset folder at `MyDrive/Regression/<dataset>_uncertainty_analysis/`. To
apply the pipeline to a new dataset, provide `train.csv` and `test.csv` with the target in the last column, copy the notebooks
from `notebooks/`, and set the folder name, file names, `feature_names` and target label in the first cell
([`docs/DATASET_PARAMETERS.md`](docs/DATASET_PARAMETERS.md)).

---

## 6. Repository structure

```
.
├── README.md                     ← this document
├── docs/
│   ├── CHANGELOG_FIXES.md        ← every v2.0 correction: before, reason, after
│   ├── DATASET_PARAMETERS.md     ← dataset-specific settings
│   ├── KAGGLE_RUNS.md            ← construction and execution of the Kaggle jobs
│   └── run_registry.tsv          ← every Kaggle run: code name, notebook, time
├── examples/<dataset>_uncertainty_analysis/
│   ├── data/                     ← train.csv, test.csv
│   └── <phase>/                  ← notebook, results/ (EXECUTED_*.ipynb, *.xlsx, run record)
├── notebooks/                    ← the generalised pipeline (without outputs) and a method guide per notebook
├── references/                   ← REFERENCES.md and key papers (PDF)
└── tools/
    ├── kaggle/                   ← job builder and push queue
    └── reports/                  ← report and figure generators
```

---

## 7. Changes in version 2.0

Version 2.0 is a **correctness release**. Each notebook was checked method by method against the original publication and,
where available, the authors' reference implementation. Each confirmed defect was corrected with the smallest possible change,
approved individually and marked in the code with `# FIX <id>`; [`docs/CHANGELOG_FIXES.md`](docs/CHANGELOG_FIXES.md) records the
state before the change, the reason and the state after.

| Area | Change |
|---|---|
| Tuning | Model selection by 5-fold CV on the training set (previously on the test set); tuned settings saved and loaded by every later notebook |
| Quantile regression | Quantile crossing measured and removed; CRPS with correct level weights; TabNet early stopping on a validation split |
| Distributions | Hersbach CRPS decomposition; IBUG variance no longer square-rooted twice; Treeffuser score model uses squared error |
| CARD | Re-implemented following the official CARD code |
| MAPIE / PUNCC | Correct interval shapes; genuine split conformal and J+aB; CQR with quantile models; a single α = 0.1 |
| NexCP / online | Out-of-fold calibration residuals; NexCP weighted quantile as published; weighted full, CV+ and J+aB; SF-OGD, SAOCP and FACI ported from the reference code; ACI added; COP and Extreme CP following their official code |
| ACP | λ no longer selected by comparing losses across λ; coverage reported against the learned level |
| HCM | Silent fallback to synthetic data removed; training schedule extended so that the network converges (H1) |
| Infrastructure | Execution on Kaggle or Colab; executed notebooks stored with the results; one NB6 job per model; run registry |

Outputs saved in the v1.2 notebooks were not produced on their stated datasets, so v1.2 numbers should not be compared with
v2.0. Earlier versions are preserved on the branches `version1.2`, `version-1.1` and `version-1.0`.

---

## 8. Limitations

| # | Limitation | Consequence |
|:-:|---|---|
| 1 | One train/test split and one run per notebook | No confidence intervals across splits or seeds; small test sets detect only large departures from 90% |
| 2 | Online methods process the test rows in their stored order | Their widths partly reflect learning from earlier test labels |
| 3 | All guarantees are marginal | A method can cover 90% overall and less for extreme target values |
| 4 | Optuna search is not bit-reproducible; neural models depend on seeds | Re-running yields slightly different numbers; the stored tuning run is the one used by every notebook |
| 5 | Kaggle's 12-hour session limit | Not run: NB6 TabNet and PGBM on sediment load (and NexCP Full there); NB6 NGBoost and PGBM on pile settlement |

---

## 9. Future work: the `reguq` library

The notebooks constitute a research pipeline: long, dataset-specific copies whose methods cannot be imported into other code.
The next stage of this project is **`reguq`** (regression uncertainty quantification), a Python package that provides the
verified methods of this benchmark as a tested and documented library:

- a common interface (`fit` / `calibrate` / `predict_interval` / `predict_distribution`) for every method family studied here,
  wrapping any scikit-learn-compatible regressor;
- the evaluation protocol of Section 4 (coverage with sampling tolerance, relative width, CRPS, PIT, crossing rate) and the
  recommendation rule as functions, so that a new dataset can be benchmarked with minimal code;
- tests that verify each method against the authors' reference implementation and against its theoretical coverage on
  synthetic data;
- repeated random splits with confidence intervals, and conditional-coverage diagnostics.

---

## 10. Citation, license and authors

| Resource | Location |
|---|---|
| Notebooks and method guides | [`notebooks/`](notebooks/README.md) |
| Executed notebooks and outputs | `examples/<dataset>_uncertainty_analysis/<phase>/results/` |
| Kaggle runs | complete log in [`docs/run_registry.tsv`](docs/run_registry.tsv) |
| References | [`references/REFERENCES.md`](references/REFERENCES.md) |
| Libraries | scikit-learn, XGBoost, LightGBM, CatBoost, GPBoost, NGBoost, PGBM, pytorch-tabnet, Optuna, MAPIE, PUNCC, IBUG, Treeffuser, PyTorch, SHAP, LIME, InterpretML |

**Citation.** If you use this benchmark or its results, please cite it (GitHub's *Cite this repository* button uses
[`CITATION.cff`](CITATION.cff)) together with the original publications of the methods used:

```bibtex
@misc{selwal_pal_regression_uq_2026,
  author       = {Selwal, Danesh and Pal, Mahesh},
  title        = {Regression Uncertainty Quantification Benchmark: Over Thirty Methods on
                  Nine Engineering and Environmental Datasets},
  year         = {2026},
  howpublished = {\url{https://github.com/DaneshSelwal/Regression-UQ-Benchmark}},
  note         = {Version 2.0}
}
```

**License.** [MIT](LICENSE).

**Authors.** Danesh Selwal and Dr. Mahesh Pal.
