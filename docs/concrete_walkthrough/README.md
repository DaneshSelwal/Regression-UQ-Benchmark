# Concrete case study: uncertainty quantification step by step

[← Main README](../../README.md)

The concrete compressive-strength dataset is used to present the complete analysis in detail. It is the only dataset on which
every notebook ran for every model without restriction, and its test set (95 mixes) is large enough for coverage to be assessed
meaningfully. Each step corresponds to one notebook of the pipeline. Every step compares all models within each method,
illustrates the results with figures, and closes with a conclusion.

| | |
|---|---|
| **Data** | 189 training and 95 test mixes; 7 inputs: cement, mineral powder, fine aggregate, coarse aggregate, fly ash, water/powder ratio, admixture |
| **Target** | Compressive strength (MPa); test mean 36.2 MPa, standard deviation 10.0 MPa |
| **Interval target** | 90% (α = 0.1); valid if coverage ≥ 84.9% (one-sided binomial test at the 5% level with 95 test mixes) |
| **Source** | [`examples/concrete_uncertainty_analysis/`](../../examples/concrete_uncertainty_analysis/README.md) (data, executed notebooks, Excel outputs) · [dataset report (PDF)](../../examples/concrete_uncertainty_analysis/report/paper.pdf) |

| Step | Notebook | Question addressed | Principal result |
|:-:|---|---|---|
| 1 | [Hyperparameter tuning](01_tuning.md) | Which base model is most accurate, and how was it selected? | CatBoost (CV RMSE 3.66 MPa); 8 boosting models within 0.5 MPa |
| 2 | [Quantile regression](02_quantile_regression.md) | Do directly estimated quantiles give a reliable 90% interval? | No: 7 of 9 models cover 17–81% |
| 3 | [Predictive distributions](03_predictive_distributions.md) | Are NGBoost, PGBM, IBUG and Treeffuser distributions calibrated? | Only IBUG (LightGBM); NGBoost and PGBM over-confident |
| 4 | [CARD](04_card.md) | Does a guided diffusion model improve the distribution? | Lowest CRPS, but intervals cover 27–60% |
| 5 | [Split, CV+, J+aB, CQR](05_conformal_split_cv_jab.md) | Does conformal prediction deliver valid intervals for every model? | Yes: 54 of 54 valid; J+aB sharpest |
| 6 | [NexCP, online, MFCS, COP](06_conformal_weighted_online.md) | Do weighted and online methods improve sharpness? | Online methods sharpest (MFCS Full 1.00 × std) |
| 7 | [Adaptive coverage policies](07_adaptive_coverage_policies.md) | What does a learned coverage level offer? | Valid at its learned level; conservative and wider |
| 8 | [HCM](08_hcm.md) | Can a single network estimate its own uncertainty? | Weakly: σ̂ tracks the error (Spearman +0.18) after fix H1, but the intervals undercover (78.9% at 95%) |
| – | [Summary](09_summary.md) | Which method should be used on this dataset? | Conformal methods; J+aB, or MFCS Full / FACI for sequential data |

The figures are generated from the stored results by [`tools/reports/showcase_concrete.py`](../../tools/reports/showcase_concrete.py);
no model is re-run.
