# Noise · NB8 Hyperspherical confidence mapping (HCM)

[← Noise README](../README.md) · [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Hyperspherical confidence mapping: one network predicts magnitude and direction; the direction's distance from the unit circle is the uncertainty. |
| **Notebook** | [`Hyperspherical_Confidence_Mapping(HCM).ipynb`](<Hyperspherical_Confidence_Mapping(HCM).ipynb>) |
| **Kaggle run(s)** | [REG24_11_2026_15_19](https://www.kaggle.com/code/daneshselwal/reg24-11-2026-15-19), [REG22_11_2026_21_59](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-21-59) |
| **Executed output** | [`results/`](results/) (4 files: executed notebook(s) and Excel workbooks) |

## HCM metrics

| Metric | Value |
|---|--:|
| Cov@1Sigma | 63.3% |
| Cov@2Sigma | 88.3% |
| Cov@3Sigma | 95.0% |
| Coverage_95Interval | 86.7% |
| Average_95Interval_Width | 17.08 |
| Pearson | 0.646 |
| Spearman | 0.565 |
| ECE_reg | 1.17 |
| RMSE | 5.14 |
| MAE | 3.63 |
| Mean_CRPS | 2.54 |
| Mean_Negative_LogLikelihood | 3.00 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md).*
