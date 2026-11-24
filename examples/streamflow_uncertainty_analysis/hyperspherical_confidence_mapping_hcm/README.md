# Streamflow · NB8 Hyperspherical confidence mapping (HCM)

[← Streamflow README](../README.md) · [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Hyperspherical confidence mapping: one network predicts magnitude and direction; the direction's distance from the unit circle is the uncertainty. |
| **Notebook** | [`Hyperspherical_Confidence_Mapping(HCM).ipynb`](<Hyperspherical_Confidence_Mapping(HCM).ipynb>) |
| **Kaggle run(s)** | [REG24_11_2026_15_24](https://www.kaggle.com/code/daneshselwal/reg24-11-2026-15-24), [REG23_11_2026_01_23](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-23) |
| **Executed output** | [`results/`](results/) (4 files: executed notebook(s) and Excel workbooks) |

## HCM metrics

| Metric | Value |
|---|--:|
| Cov@1Sigma | 65.6% |
| Cov@2Sigma | 93.8% |
| Cov@3Sigma | 99.0% |
| Coverage_95Interval | 92.7% |
| Average_95Interval_Width | 414.6 |
| Pearson | 0.557 |
| Spearman | 0.274 |
| ECE_reg | 27.43 |
| RMSE | 114.9 |
| MAE | 80.01 |
| Mean_CRPS | 57.18 |
| Mean_Negative_LogLikelihood | 5.99 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md).*
