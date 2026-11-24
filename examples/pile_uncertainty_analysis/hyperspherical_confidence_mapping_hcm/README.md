# Pile · NB8 Hyperspherical confidence mapping (HCM)

[← Pile README](../README.md) · [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Hyperspherical confidence mapping: one network predicts magnitude and direction; the direction's distance from the unit circle is the uncertainty. |
| **Notebook** | [`Hyperspherical_Confidence_Mapping(HCM).ipynb`](<Hyperspherical_Confidence_Mapping(HCM).ipynb>) |
| **Kaggle run(s)** | [REG24_11_2026_15_20](https://www.kaggle.com/code/daneshselwal/reg24-11-2026-15-20), [REG22_11_2026_14_03](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-14-03) |
| **Executed output** | [`results/`](results/) (4 files: executed notebook(s) and Excel workbooks) |

## HCM metrics

| Metric | Value |
|---|--:|
| Cov@1Sigma | 91.7% |
| Cov@2Sigma | 95.8% |
| Cov@3Sigma | 95.8% |
| Coverage_95Interval | 95.8% |
| Average_95Interval_Width | 5.58 |
| Pearson | -0.068 |
| Spearman | 0.123 |
| ECE_reg | 1.06 |
| RMSE | 0.440 |
| MAE | 0.367 |
| Mean_CRPS | 0.417 |
| Mean_Negative_LogLikelihood | 9.83 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md).*
