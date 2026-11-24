# Water quality (WQI) · NB8 Hyperspherical confidence mapping (HCM)

[← Water quality (WQI) README](../README.md) · [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Hyperspherical confidence mapping: one network predicts magnitude and direction; the direction's distance from the unit circle is the uncertainty. |
| **Notebook** | [`Hyperspherical_Confidence_Mapping(HCM).ipynb`](<Hyperspherical_Confidence_Mapping(HCM).ipynb>) |
| **Kaggle run(s)** | [REG24_11_2026_15_26](https://www.kaggle.com/code/daneshselwal/reg24-11-2026-15-26), [REG23_11_2026_08_58](https://www.kaggle.com/code/daneshselwal/reg23-11-2026-08-58) |
| **Executed output** | [`results/`](results/) (4 files: executed notebook(s) and Excel workbooks) |

## HCM metrics

| Metric | Value |
|---|--:|
| Cov@1Sigma | 64.6% |
| Cov@2Sigma | 89.0% |
| Cov@3Sigma | 97.6% |
| Coverage_95Interval | 89.0% |
| Average_95Interval_Width | 107.4 |
| Pearson | 0.348 |
| Spearman | 0.218 |
| ECE_reg | 7.63 |
| RMSE | 29.83 |
| MAE | 23.63 |
| Mean_CRPS | 16.81 |
| Mean_Negative_LogLikelihood | 4.93 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md).*
