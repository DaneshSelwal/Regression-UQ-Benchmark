# Regression Uncertainty Quantification Analysis — Version 2.0

An end-to-end, notebook-based pipeline for regression with **uncertainty quantification (UQ)**:
every point prediction comes with a prediction interval or a full predictive distribution, and the
quality of that uncertainty (coverage, width, calibration, CRPS) is measured and exported to Excel.
The pipeline tunes ten gradient-boosting / neural regressors, then runs them through quantile
regression, probabilistic (distributional and diffusion) models, standard and adaptive conformal
prediction, learned coverage policies and hyperspherical confidence mapping. It is domain-agnostic
and is applied here to nine civil- and environmental-engineering datasets.

> **Branch:** `version2.0` (default). The previous release is preserved unchanged on
> [`version1.2`](https://github.com/DaneshSelwal/Regression_Uncertainty_Quantification_Analysis/tree/version1.2)
> (and `version-1.1`, `version-1.0`).

---

## Contents

1. [What's new in 2.0](#whats-new-in-20)
2. [Repository layout](#repository-layout)
3. [Pipeline (run order)](#pipeline-run-order)
4. [Datasets](#datasets)
5. [Running on Kaggle](#running-on-kaggle)
6. [Running on Google Colab](#running-on-google-colab)
7. [Using your own dataset](#using-your-own-dataset)
8. [Validation status](#validation-status)
9. [References and citation](#references-and-citation)
10. [License and authors](#license-and-authors)

---

## What's new in 2.0

Version 2.0 is a **correctness release**. Every notebook was reviewed method by method against the
original paper and, where one exists, the authors' official code. Each confirmed problem was fixed
with the smallest possible change, approved individually, and marked in the code with a
`# FIX <id>` comment (`grep -n "FIX" *.ipynb` lists them all). The structure, model list, variable
names and Excel output formats of the notebooks are kept.

Main changes (full before/after/why for each fix in
[`examples/pile_uncertainty_analysis/CHANGELOG_FIXES.md`](examples/pile_uncertainty_analysis/CHANGELOG_FIXES.md)):

| Area | Change |
|---|---|
| Hyperparameter tuning | Model selection by 5-fold cross-validation on the training set only (the test set was previously used to pick the best model); pruners now receive per-fold reports; tuned settings are saved to `best_params.pkl` and **loaded** by every downstream notebook instead of being copy-pasted. |
| Quantile regression | Quantile crossing removed by monotone rearrangement (counts reported); CRPS computed from quantiles with the correct level weights; XGBoost uses its own tuned settings; TabNet early stopping on a validation split with the pinball loss. |
| Probabilistic distribution | Hersbach (2000) CRPS decomposition (reliability / resolution / uncertainty); IBUG scale no longer square-rooted twice; Treeffuser's score model uses squared error; plot/data mix-ups fixed. |
| CARD | Re-implemented to follow the official CARD code (noise schedule, conditional guided model, full reverse diffusion, EMA with warm-up). |
| MAPIE / PUNCC | Correct MAPIE interval shapes; true split conformal and J+aB; CQR with real quantile models; one miscoverage level (`ALPHA = 0.1`) for all methods; tuned settings reach CatBoost and PGBM. |
| NEXCP / online CP | Out-of-fold calibration residuals; NexCP weighted quantile per eq. (10)–(11); real weighted full conformal, CV+ and J+aB; SF-OGD, SAOCP and FACI ported from the authors' reference code; ACI added; COP step size per the official code; Extreme CP per the official `ExtremeConformal` package. |
| ACP (AlphaNet) | λ no longer chosen by comparing losses across λ values; input scaling and near-1 initialisation; coverage reported against the learned level 1 − α̃. |
| HCM | Verified against the paper and official code; the silent fallback to synthetic data was removed. |
| Infrastructure | Notebooks run unchanged on Kaggle or Colab; Kaggle jobs execute notebooks with papermill so the executed notebook (all cell outputs) is stored with the results; job registry in [`docs/KAGGLE_RUNS.md`](docs/KAGGLE_RUNS.md). |

All sources used to verify the fixes are listed in
[`examples/pile_uncertainty_analysis/REFERENCES.md`](examples/pile_uncertainty_analysis/REFERENCES.md)
(entries marked **[fix]**).

---

## Repository layout

```
.
├── README.md
├── LICENSE
├── Regression/                         # template pipeline (copy to Google Drive / use as a starting point)
│   ├── data/                           # train.csv, test.csv (last column = target)
│   ├── hyperparameter_tuning/          # Optuna_autosampler.ipynb (+ .md theory notes)
│   ├── quantile_regression/
│   ├── probabilistic_distribution/
│   ├── probabilistic_distribution_card/
│   ├── conformal_predictions_mapie_puncc/
│   ├── conformal_predictions_nexcp_adaptivecp_mfcs/
│   ├── conformal_predictions_adaptive_coverage_policies/
│   └── hyperspherical_confidence_mapping_hcm/
├── examples/
│   ├── <dataset>_uncertainty_analysis/ # one folder per dataset: data/ + the same 8 phase sub-folders
│   └── pile_uncertainty_analysis/      # pilot dataset for v2.0: also CHANGELOG_FIXES.md, REFERENCES.md, README.md
├── resources/                          # PDFs: NexCP, ACI, Gentle Introduction to Conformal Prediction
├── docs/
│   ├── KAGGLE_RUNS.md                  # registry of Kaggle jobs: which notebook runs where, status, outputs
│   └── DATASET_PARAMETERS.md           # settings that depend on the dataset (names, sizes, splits)
└── tools/
    └── kaggle/                         # make_kernel.py (papermill job builder), queue_push.sh, README.md
```

---

## Pipeline (run order)

Run the notebooks of a dataset in this order. Notebook 1 must run first: it writes
`hyperparameter_tuning/best_params.pkl`, which all other notebooks load.

| # | Folder / notebook | Methods | Main outputs |
|---|---|---|---|
| 1 | `hyperparameter_tuning/Optuna_autosampler.ipynb` | Optuna AutoSampler × 8 pruners, 5-fold CV on train; Random Forest, Gradient Boosting, XGBoost, LightGBM, GPBoost, CatBoost, HistGradientBoosting (PGBM), NGBoost, PGBM, TabNet; SHAP / LIME / EBM interpretability | `test_results.xlsx`, `best_params.pkl` / `.json` |
| 2 | `quantile_regression/Quantile_Regression.ipynb` | One model per quantile level (13 levels, 0.05–0.95) for each regressor | one Excel per model: quantiles, coverage, width, pinball loss, CRPS, crossing counts, plots |
| 3 | `probabilistic_distribution/Probabilistic__Distribution.ipynb` | NGBoost, PGBM, IBUG, Treeffuser | one Excel per model: μ/σ, NLL, CRPS + Hersbach decomposition, PIT, calibration |
| 4 | `probabilistic_distribution_card/Probabilistic__Distribution(CARD).ipynb` | CARD diffusion regression guided by each tuned regressor | one Excel per model: predictive samples, CRPS decomposition, calibration |
| 5 | `conformal_predictions_mapie_puncc/Conformal Predictions(MAPIE,PUNCC).ipynb` | MAPIE: split CP, CV+, J+aB; PUNCC: SplitCP, CV+, CQR | one Excel per model: intervals, coverage, width, CWC, conditional coverage |
| 6 | `conformal_predictions_nexcp_adaptivecp_mfcs/Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb` | NexCP (split, full, CV+, J+aB), online split / CV+ / J+aB, SF-OGD, SAOCP, FACI, ACI, MFCS, COP, Extreme CP, AlphaNet | one Excel per model with one sheet group per method |
| 7 | `conformal_predictions_adaptive_coverage_policies/Adaptive_Coverage_Policies(ACP).ipynb` | Adaptive Coverage Policies (AlphaNet, e-value conformal) | one Excel per model: intervals, learned α̃, λ comparison, training curves |
| 8 | `hyperspherical_confidence_mapping_hcm/Hyperspherical_Confidence_Mapping(HCM).ipynb` | Hyperspherical Confidence Mapping | `HCM.xlsx`, `Matrix Evaluation.xlsx`: Cov@1/2/3σ, ECE_reg, Pearson/Spearman |

Each folder also contains a `.md` theory note for its method.

---

## Datasets

Each dataset lives in `examples/<name>_uncertainty_analysis/data/` as `train.csv` / `test.csv`;
the last column is the target.

| Dataset | Train rows | Test rows | Target column |
|---|---:|---:|---|
| concrete | 189 | 95 | `str` |
| noise | 168 | 60 | `Leq` |
| pile (pilot for v2.0) | 81 | 24 | `Total` |
| pile_settlement | 371 | 186 | `S_exp` |
| scour | 154 | 78 | `Scour` |
| sediment_load | 1095 | 365 | `St` |
| streamflow | 192 | 96 | `TDS` |
| water_quality_ewqi | 47 | 47 | `Pre monsoon EWQI` |
| water_quality_wqi | 190 | 82 | `WQI` |

---

## Running on Kaggle

All computation for v2.0 runs on Kaggle (CPU, **internet on** — required for `pip install`).

1. Attach the dataset `daneshselwal/reguq-uq-data` (the train/test CSV files).
2. Open notebook 1 (tuning) with internet enabled and run it. The first cell (`FIX K1`) detects
   Kaggle, copies the CSV files into the folder layout the notebooks expect
   (`./drive/MyDrive/Regression/<dataset>_uncertainty_analysis/data/`), and the install cell
   installs only the missing packages at pinned versions.
3. For notebooks 2–8, add the output of the tuning job as an input (`kernel_sources`); the `FIX K2`
   cell finds `hyperparameter_tuning/best_params.pkl` there automatically.
4. To get the executed notebook (with every cell output) back through the CLI, run notebooks as
   papermill jobs with `tools/kaggle/make_kernel.py`; see [`tools/kaggle/README.md`](tools/kaggle/README.md).
   Every job, its inputs and where its results are stored are recorded in
   [`docs/KAGGLE_RUNS.md`](docs/KAGGLE_RUNS.md).

Kaggle allows 5 concurrent CPU sessions per account; `tools/kaggle/queue_push.sh` waits for a free
slot before pushing.

## Running on Google Colab

The same notebooks still run on Colab (the original Colab install blocks are kept under
`else:` in the `FIX K1` cells).

1. Upload the dataset folder (e.g. `examples/pile_uncertainty_analysis/`, renamed or placed as
   `MyDrive/Regression/pile_uncertainty_analysis/`) to Google Drive.
2. Open a notebook in Colab; the first cell mounts Drive.
3. Run notebook 1 first, then the others in the order above.

## Using your own dataset

1. Create `train.csv` and `test.csv` with the target as the last column.
2. Copy `Regression/` (or an `examples/` folder) and replace the data files.
3. Update the dataset folder name, `feature_names` and the plot target label as described in
   [`docs/DATASET_PARAMETERS.md`](docs/DATASET_PARAMETERS.md), which also lists the settings whose
   effect depends on dataset size.

---

## Validation status

| Dataset | Status |
|---|---|
| pile | Pilot dataset. All 8 notebooks fixed; demo runs and the full tuning run on Kaggle in progress; results under review by the authors. |
| all others | Pending. Will be run with the v2.0 notebooks once the pile results are confirmed. |

The saved outputs in the v1.2 example notebooks were not produced on their own datasets (see the
change log), so v1.2 numbers should not be compared directly with v2.0 results.

---

## References and citation

The complete reference list — base models, tuning, interpretability, quantile regression,
probabilistic models, conformal prediction (standard, non-exchangeable, online, extreme, learned
coverage), HCM and software — is in
[`examples/pile_uncertainty_analysis/REFERENCES.md`](examples/pile_uncertainty_analysis/REFERENCES.md).
Local copies of three key papers are in `resources/`:

- Barber, Candès, Ramdas & Tibshirani (2023). *Conformal prediction beyond exchangeability.* [PDF](resources/NEXCP_beyond_exchangeability.pdf)
- Gibbs & Candès (2021). *Adaptive conformal inference under distribution shift.* [PDF](resources/Adaptive_Conformal_Inference.pdf)
- Angelopoulos & Bates (2023). *A gentle introduction to conformal prediction.* [PDF](resources/Gentle_Intro_to_Conformal_Prediction.pdf)

If you use this pipeline, please cite this repository together with the original papers of the
methods you use.

## License and authors

Released under the [MIT License](LICENSE).

Developed under the guidance of Dr. Mahesh Pal by Prakriti Bisht and Danesh Selwal.
