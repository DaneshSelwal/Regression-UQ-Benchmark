# Concrete · NB8 Hyperspherical confidence mapping (HCM)

[← Concrete README](../README.md) · [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Hyperspherical confidence mapping: one network predicts magnitude and direction; the direction's distance from the unit circle is the uncertainty. |
| **Notebook** | [`Hyperspherical_Confidence_Mapping(HCM).ipynb`](<Hyperspherical_Confidence_Mapping(HCM).ipynb>) |
| **Kaggle run(s)** | [REG24_11_2026_15_14](https://www.kaggle.com/code/daneshselwal/reg24-11-2026-15-14), [REG22_11_2026_21_44](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-21-44) |
| **Executed output** | [`results/`](results/) (4 files: executed notebook(s) and Excel workbooks) |

## HCM metrics

| Metric | Value |
|---|--:|
| Cov@1Sigma | 51.6% |
| Cov@2Sigma | 80.0% |
| Cov@3Sigma | 91.6% |
| Coverage_95Interval | 78.9% |
| Average_95Interval_Width | 15.75 |
| Pearson | 0.221 |
| Spearman | 0.182 |
| ECE_reg | 1.36 |
| RMSE | 5.62 |
| MAE | 4.46 |
| Mean_CRPS | 3.22 |
| Mean_Negative_LogLikelihood | 4.64 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md).*
