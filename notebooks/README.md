# The pipeline: eight notebooks

[← main README](../README.md)

These are the **generalised v2.0 notebooks** with no outputs, so the code is easy to read. Each folder in
[`examples/`](../examples/) holds the same notebooks with one dataset's paths, plus the executed copies and results.

| # | Folder | Method guide (theory, maths, settings) | What it computes |
|---|---|---|---|
| 1 | [`01_hyperparameter_tuning/`](01_hyperparameter_tuning/) | [guide](01_hyperparameter_tuning/README.md) | Optuna tuning of ten regressors by 5-fold CV on the training set; `best_params.pkl` |
| 2 | [`02_quantile_regression/`](02_quantile_regression/) | [guide](02_quantile_regression/README.md) | Pinball-loss models at 13 levels; raw 90% interval, quantile CRPS, crossing rate |
| 3 | [`03_probabilistic_distribution/`](03_probabilistic_distribution/) | [guide](03_probabilistic_distribution/README.md) | NGBoost, PGBM, IBUG, Treeffuser distributions; CRPS, PIT, calibration |
| 4 | [`04_probabilistic_distribution_card/`](04_probabilistic_distribution_card/) | [guide](04_probabilistic_distribution_card/README.md) | CARD diffusion guided by each tuned model |
| 5 | [`05_conformal_mapie_puncc/`](05_conformal_mapie_puncc/) | [guide](05_conformal_mapie_puncc/README.md) | Split CP, CV+, J+aB (MAPIE); split CP, CV+, CQR (PUNCC) |
| 6 | [`06_conformal_nexcp_online/`](06_conformal_nexcp_online/) | [guide](06_conformal_nexcp_online/README.md) | NexCP; ACI, FACI, SF-OGD, SAOCP, online split/CV+/J+aB, MFCS; COP; Extreme CP |
| 7 | [`07_adaptive_coverage_policies/`](07_adaptive_coverage_policies/) | [guide](07_adaptive_coverage_policies/README.md) | Learned per-point coverage with e-value conformal sets |
| 8 | [`08_hyperspherical_confidence_mapping/`](08_hyperspherical_confidence_mapping/) | [guide](08_hyperspherical_confidence_mapping/README.md) | Single-network uncertainty (HCM) |

Run notebook 1 first; notebooks 2–7 load its `best_params.pkl`; notebook 8 is independent. The `.md` file next to each
notebook is the older v1.2 theory note, kept for reference.
