# Streamflow · NB8 Hyperspherical confidence mapping (HCM)

[← Streamflow README](../README.md) · [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md) · [report (PDF)](../../../papers/streamflow/paper.pdf)

| | |
|---|---|
| **What** | Hyperspherical confidence mapping: one network predicts magnitude and direction; the direction's distance from the unit circle is the uncertainty. |
| **Notebook** | [`Hyperspherical_Confidence_Mapping(HCM).ipynb`](<Hyperspherical_Confidence_Mapping(HCM).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_01_23](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-23) |
| **Executed output** | [`results/`](results/) (4 files: executed notebook(s) and Excel workbooks) |

## HCM metrics

| Metric | Value |
|---|--:|
| Cov@1Sigma | 62.5% |
| Cov@2Sigma | 95.8% |
| Cov@3Sigma | 96.9% |
| Coverage_95Interval | 95.8% |
| Average_95Interval_Width | 513.8 |
| Pearson | 0.107 |
| Spearman | -0.147 |
| ECE_reg | 28.07 |
| RMSE | 192.4 |
| MAE | 125.2 |
| Mean_CRPS | 90.44 |
| Mean_Negative_LogLikelihood | 6.71 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md).*
