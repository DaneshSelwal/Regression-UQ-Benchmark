# Kaggle run registry

Every Kaggle job used for Version 2.0, with its inputs and where its results are kept.
Account: **daneshselwal** (main account; CPU, internet on). Update the *Status* column when a
job's state changes.

## How jobs are run

- **Runner.** Notebooks are executed by `tools/kaggle/make_kernel.py`, which builds a Kaggle
  *script* job that runs the notebook with papermill and saves `EXECUTED_<notebook>.ipynb`, with
  every cell output, in `/kaggle/working`. `kaggle kernels output` then downloads that notebook
  together with the Excel files. A plain notebook job returns only the output files and a log.
  The tuning job `pile-optuna-autosampler` predates the runner and is a plain notebook job.
- **Inputs.** Dataset `daneshselwal/reguq-uq-data` (train/test CSV files). For pile, `d08_a.csv`
  is the train file and `d08_b.csv` the test file. Downstream notebooks also take the output of a
  parameters job as `kernel_sources`, which provides
  `drive/MyDrive/Regression/<dataset>_uncertainty_analysis/hyperparameter_tuning/best_params.pkl`.
- **Concurrency.** At most **5 concurrent CPU sessions** per Kaggle account, shared with any other
  jobs on the account. `tools/kaggle/queue_push.sh` waits for a free slot.
- **Naming.** `<dataset>-<demo|real>-nb<N>-<short>`, for example `pile-demo-nb5-mapie` or
  `pile-real-nb2-quantile`. *demo* = run with demo (pre-assumed) tuned parameters to check that a
  fixed notebook runs end to end. *real* = run with the parameters from the real tuning job.
- **Where results go.** Executed notebook and Excel files are committed to
  `examples/<dataset>_uncertainty_analysis/<phase>/results/<demo|real>/` on branch `version2.0`.
  Large model pickles are not committed; `best_params.pkl` / `best_params.json` are.

## Notebook numbers

| N | Phase folder | Short name |
|---|---|---|
| 1 | `hyperparameter_tuning` | tuning / optuna |
| 2 | `quantile_regression` | quantile |
| 3 | `probabilistic_distribution` | probdist |
| 4 | `probabilistic_distribution_card` | card |
| 5 | `conformal_predictions_mapie_puncc` | mapie |
| 6 | `conformal_predictions_nexcp_adaptivecp_mfcs` | nexcp |
| 7 | `conformal_predictions_adaptive_coverage_policies` | acp |
| 8 | `hyperspherical_confidence_mapping_hcm` | hcm |

## Jobs

| Slug | Notebook | Dataset | Params source | Purpose | Status | Output stored at |
|---|---|---|---|---|---|---|
| `pile-optuna-autosampler` | NB1 `Optuna_autosampler.ipynb` | pile (d08_a / d08_b) | — (produces them) | **Real** tuning run: CV-selected parameters for all models × pruners | running | `examples/pile_uncertainty_analysis/hyperparameter_tuning/results/real/` (after completion) |
| `pile-demo-params` | script (no notebook) | — | old hand-pasted `best_scores_autosampler` dict from the v1.2 pile notebooks | Writes a demo `best_params.pkl` so notebooks 2–8 can be test-run before the real tuning finishes | complete | Kaggle output only (input to the demo jobs) |
| `pile-demo-nb2-quantile` | NB2 `Quantile_Regression.ipynb` | pile | `pile-demo-params` | Demo end-to-end run of the fixed notebook | queued/running | `examples/pile_uncertainty_analysis/quantile_regression/results/demo/` |
| `pile-demo-nb3-probdist` | NB3 `Probabilistic__Distribution.ipynb` | pile | `pile-demo-params` | Demo run | queued/running | `examples/pile_uncertainty_analysis/probabilistic_distribution/results/demo/` |
| `pile-demo-nb4-card` | NB4 `Probabilistic__Distribution(CARD).ipynb` | pile | `pile-demo-params` | Demo run | queued/running | `examples/pile_uncertainty_analysis/probabilistic_distribution_card/results/demo/` |
| `pile-demo-nb5-mapie` | NB5 `Conformal Predictions(MAPIE,PUNCC).ipynb` | pile | `pile-demo-params` | Demo run | queued/running | `examples/pile_uncertainty_analysis/conformal_predictions_mapie_puncc/results/demo/` |
| `pile-demo-nb6-nexcp` | NB6 `Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb` | pile | `pile-demo-params` | Demo run | queued/running | `examples/pile_uncertainty_analysis/conformal_predictions_nexcp_adaptivecp_mfcs/results/demo/` |
| `pile-demo-nb7-acp` | NB7 `Adaptive_Coverage_Policies(ACP).ipynb` | pile | `pile-demo-params` | Demo run | queued/running | `examples/pile_uncertainty_analysis/conformal_predictions_adaptive_coverage_policies/results/demo/` |
| `pile-demo-nb8-hcm` | NB8 `Hyperspherical_Confidence_Mapping(HCM).ipynb` | pile | — (HCM does not use tuned params) | Demo run | queued/running | `examples/pile_uncertainty_analysis/hyperspherical_confidence_mapping_hcm/results/demo/` |
| `hersbach-check` | script | synthetic | — | Validation probe for the Hersbach CRPS decomposition (fix P1) | complete | Kaggle output only |
| `card-check` | script | synthetic | — | Validation probe for the re-implemented CARD (fix C1/C2: coverage 0.915, σ 1.055 vs true 0.955) | complete | Kaggle output only |
| `rq-env-probe` | script | — | — | Kaggle image probe: Python 3.13, package availability with internet | complete | Kaggle output only |

Planned after the real tuning run: `pile-real-nb2-quantile` … `pile-real-nb8-hcm` with
`kernel_sources = daneshselwal/pile-optuna-autosampler`, then the same set for the other datasets.
