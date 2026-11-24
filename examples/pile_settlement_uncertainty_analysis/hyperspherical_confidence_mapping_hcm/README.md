# Pile settlement · NB8 Hyperspherical confidence mapping (HCM)

[← Pile settlement README](../README.md) · [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Hyperspherical confidence mapping: one network predicts magnitude and direction; the direction's distance from the unit circle is the uncertainty. |
| **Notebook** | [`Hyperspherical_Confidence_Mapping(HCM).ipynb`](<Hyperspherical_Confidence_Mapping(HCM).ipynb>) |
| **Kaggle run(s)** | [REG24_11_2026_15_21](https://www.kaggle.com/code/daneshselwal/reg24-11-2026-15-21), [REG22_11_2026_22_24](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-22-24) |
| **Executed output** | [`results/`](results/) (4 files: executed notebook(s) and Excel workbooks) |

## HCM metrics

| Metric | Value |
|---|--:|
| Cov@1Sigma | 79.0% |
| Cov@2Sigma | 90.9% |
| Cov@3Sigma | 94.6% |
| Coverage_95Interval | 90.3% |
| Average_95Interval_Width | 21.99 |
| Pearson | 0.566 |
| Spearman | 0.429 |
| ECE_reg | 1.84 |
| RMSE | 9.61 |
| MAE | 4.16 |
| Mean_CRPS | 3.32 |
| Mean_Negative_LogLikelihood | 3.72 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md).*
