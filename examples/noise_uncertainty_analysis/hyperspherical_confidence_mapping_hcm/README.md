# Noise · NB8 Hyperspherical confidence mapping (HCM)

[← Noise README](../README.md) · [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md) · [report (PDF)](../../../papers/noise/paper.pdf)

| | |
|---|---|
| **What** | Hyperspherical confidence mapping: one network predicts magnitude and direction; the direction's distance from the unit circle is the uncertainty. |
| **Notebook** | [`Hyperspherical_Confidence_Mapping(HCM).ipynb`](<Hyperspherical_Confidence_Mapping(HCM).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_21_59](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-21-59) |
| **Executed output** | [`results/`](results/) (4 files: executed notebook(s) and Excel workbooks) |

## HCM metrics

| Metric | Value |
|---|--:|
| Cov@1Sigma | 66.7% |
| Cov@2Sigma | 80.0% |
| Cov@3Sigma | 91.7% |
| Coverage_95Interval | 80.0% |
| Average_95Interval_Width | 43.86 |
| Pearson | -0.319 |
| Spearman | -0.124 |
| ECE_reg | 6.62 |
| RMSE | 15.26 |
| MAE | 11.45 |
| Mean_CRPS | 8.66 |
| Mean_Negative_LogLikelihood | 4.39 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md).*
