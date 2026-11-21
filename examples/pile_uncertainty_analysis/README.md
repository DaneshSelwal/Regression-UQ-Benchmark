# Pile dataset (pilot for Version 2.0)

The pile dataset is the smallest clean dataset in the project. It is used to verify every
Version 2.0 fix end to end before the other datasets are run.

| | |
|---|---|
| Training set | `data/train.csv`: 81 rows |
| Test set | `data/test.csv`: 24 rows |
| Features (7) | `F1`, `F2`, `Zv1`, `Zv2`, `phi`, `Th`, `L` |
| Target | `Total` (last column) |
| Kaggle input | `daneshselwal/reguq-uq-data`: `d08_a.csv` (train), `d08_b.csv` (test) |

## Run order

1. `hyperparameter_tuning/Optuna_autosampler.ipynb`: writes `hyperparameter_tuning/best_params.pkl`.
2. `quantile_regression/Quantile_Regression.ipynb`
3. `probabilistic_distribution/Probabilistic__Distribution.ipynb`
4. `probabilistic_distribution_card/Probabilistic__Distribution(CARD).ipynb`
5. `conformal_predictions_mapie_puncc/Conformal Predictions(MAPIE,PUNCC).ipynb`
6. `conformal_predictions_nexcp_adaptivecp_mfcs/Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb`
7. `conformal_predictions_adaptive_coverage_policies/Adaptive_Coverage_Policies(ACP).ipynb`
8. `hyperspherical_confidence_mapping_hcm/Hyperspherical_Confidence_Mapping(HCM).ipynb`

Notebooks 2–7 load the tuned parameters from step 1. Notebook 8 (HCM) trains its own network.

## Documents

- [`CHANGELOG_FIXES.md`](CHANGELOG_FIXES.md): every fix, with before, why and after, and its
  approval. Each fix is marked `# FIX <id>` in the code.
- [`REFERENCES.md`](REFERENCES.md): all papers, repositories and libraries used. Sources used to
  verify a fix are marked **[fix]**.
- [`../../docs/KAGGLE_RUNS.md`](../../docs/KAGGLE_RUNS.md): which Kaggle job runs which notebook,
  its status and where its outputs are stored.
- [`../../docs/DATASET_PARAMETERS.md`](../../docs/DATASET_PARAMETERS.md): settings that depend on
  the dataset.

## Results

Executed notebooks (with all cell outputs) and Excel files are stored per phase:

- `<phase>/results/demo/`: runs with demo parameters, to check that each fixed notebook runs end
  to end.
- `<phase>/results/real/`: runs with the parameters from the real tuning job. These are the
  results to review.

The outputs saved in the v1.2 notebooks were not produced on pile: they show the concrete and
noise datasets' shapes. They were cleared in Version 2.0.
