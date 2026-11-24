# Water quality (WQI) · NB8 Hyperspherical confidence mapping (HCM)

[← Water quality (WQI) README](../README.md) · [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md) · [report (PDF)](../../../papers/water_quality_wqi/paper.pdf)

| | |
|---|---|
| **What** | Hyperspherical confidence mapping: one network predicts magnitude and direction; the direction's distance from the unit circle is the uncertainty. |
| **Notebook** | [`Hyperspherical_Confidence_Mapping(HCM).ipynb`](<Hyperspherical_Confidence_Mapping(HCM).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_08_58](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-08-58) |
| **Executed output** | [`results/`](results/) (4 files: executed notebook(s) and Excel workbooks) |

## HCM metrics

| Metric | Value |
|---|--:|
| Cov@1Sigma | 45.1% |
| Cov@2Sigma | 86.6% |
| Cov@3Sigma | 98.8% |
| Coverage_95Interval | 81.7% |
| Average_95Interval_Width | 130.7 |
| Pearson | 0.538 |
| Spearman | 0.304 |
| ECE_reg | 7.89 |
| RMSE | 39.88 |
| MAE | 33.60 |
| Mean_CRPS | 23.21 |
| Mean_Negative_LogLikelihood | 5.29 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md).*
