# Papers

[← Main README](../README.md)

| Document | Scope | PDF | Source |
|---|---|---|---|
| **Benchmark paper** | Cross-dataset analysis of all nine datasets: validity and sharpness of every method family, with the mathematics of each method | [paper.pdf](benchmark/paper.pdf) | [paper.tex](benchmark/paper.tex) |

**Dataset reports.** Each dataset report is stored with its data and results in the corresponding example folder:

| Dataset | Rows (train / test), features | Report |
|---|---|---|
| Concrete (case study) | 189 / 95, 7 | [`examples/concrete_uncertainty_analysis/report/`](../examples/concrete_uncertainty_analysis/report/paper.pdf) |
| Noise | 168 / 60, 5 | [`examples/noise_uncertainty_analysis/report/`](../examples/noise_uncertainty_analysis/report/paper.pdf) |
| Pile | 81 / 24, 7 | [`examples/pile_uncertainty_analysis/report/`](../examples/pile_uncertainty_analysis/report/paper.pdf) |
| Pile settlement | 371 / 186, 8 | [`examples/pile_settlement_uncertainty_analysis/report/`](../examples/pile_settlement_uncertainty_analysis/report/paper.pdf) |
| Scour | 154 / 78, 7 | [`examples/scour_uncertainty_analysis/report/`](../examples/scour_uncertainty_analysis/report/paper.pdf) |
| Sediment load | 1095 / 365, 3 | [`examples/sediment_load_uncertainty_analysis/report/`](../examples/sediment_load_uncertainty_analysis/report/paper.pdf) |
| Streamflow | 192 / 96, 4 | [`examples/streamflow_uncertainty_analysis/report/`](../examples/streamflow_uncertainty_analysis/report/paper.pdf) |
| Water quality (EWQI) | 47 / 47, 14 | [`examples/water_quality_ewqi_uncertainty_analysis/report/`](../examples/water_quality_ewqi_uncertainty_analysis/report/paper.pdf) |
| Water quality (WQI) | 190 / 82, 151 | [`examples/water_quality_wqi_uncertainty_analysis/report/`](../examples/water_quality_wqi_uncertainty_analysis/report/paper.pdf) |

**Structure of a dataset report:** introduction and research questions; problem set-up and notation; data analysis; pipeline;
methods (tuning, quantile regression, predictive distributions, split, cross-validation and bootstrap conformal prediction,
NexCP, online conformal prediction, COP, Extreme CP, ACP, HCM); evaluation protocol; results for RQ1–RQ6; discussion,
limitations and future work; conclusions; Kaggle runs; appendix with every model × method result.

All documents are generated from the stored results by [`tools/reports/`](../tools/reports/README.md); the numbers in the text,
tables and figures are those of the Excel files and executed notebooks.
