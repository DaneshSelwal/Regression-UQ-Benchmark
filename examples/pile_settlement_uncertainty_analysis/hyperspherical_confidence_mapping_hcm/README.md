# Pile settlement · NB8 Hyperspherical confidence mapping (HCM)

[← Pile settlement README](../README.md) · [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md) · [report (PDF)](../../../papers/pile_settlement/paper.pdf)

| | |
|---|---|
| **What** | Hyperspherical confidence mapping: one network predicts magnitude and direction; the direction's distance from the unit circle is the uncertainty. |
| **Notebook** | [`Hyperspherical_Confidence_Mapping(HCM).ipynb`](<Hyperspherical_Confidence_Mapping(HCM).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_22_24](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-22-24) |
| **Executed output** | [`results/`](results/) (4 files: executed notebook(s) and Excel workbooks) |

## HCM metrics

| Metric | Value |
|---|--:|
| Cov@1Sigma | 74.7% |
| Cov@2Sigma | 86.6% |
| Cov@3Sigma | 91.9% |
| Coverage_95Interval | 86.6% |
| Average_95Interval_Width | 54.66 |
| Pearson | 0.264 |
| Spearman | 0.511 |
| ECE_reg | 3.42 |
| RMSE | 23.07 |
| MAE | 14.91 |
| Mean_CRPS | 11.20 |
| Mean_Negative_LogLikelihood | 5.06 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md).*
