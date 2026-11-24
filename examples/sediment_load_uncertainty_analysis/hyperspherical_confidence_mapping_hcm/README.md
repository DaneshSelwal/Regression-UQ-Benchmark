# Sediment load · NB8 Hyperspherical confidence mapping (HCM)

[← Sediment load README](../README.md) · [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Hyperspherical confidence mapping: one network predicts magnitude and direction; the direction's distance from the unit circle is the uncertainty. |
| **Notebook** | [`Hyperspherical_Confidence_Mapping(HCM).ipynb`](<Hyperspherical_Confidence_Mapping(HCM).ipynb>) |
| **Kaggle run(s)** | [REG24_11_2026_15_23](https://www.kaggle.com/code/daneshselwal/reg24-11-2026-15-23), [REG23_11_2026_01_45](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-45) |
| **Executed output** | [`results/`](results/) (4 files: executed notebook(s) and Excel workbooks) |

## HCM metrics

| Metric | Value |
|---|--:|
| Cov@1Sigma | 67.7% |
| Cov@2Sigma | 90.1% |
| Cov@3Sigma | 96.4% |
| Coverage_95Interval | 90.1% |
| Average_95Interval_Width | 459.7 |
| Pearson | 0.892 |
| Spearman | 0.431 |
| ECE_reg | 72.01 |
| RMSE | 520.2 |
| MAE | 140.9 |
| Mean_CRPS | 109.3 |
| Mean_Negative_LogLikelihood | 7.23 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md).*
