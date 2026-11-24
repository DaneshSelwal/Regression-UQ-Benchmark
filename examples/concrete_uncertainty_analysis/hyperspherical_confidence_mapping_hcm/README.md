# Concrete · NB8 Hyperspherical confidence mapping (HCM)

[← Concrete README](../README.md) · [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md) · [report (PDF)](../../../papers/concrete/paper.pdf)

| | |
|---|---|
| **What** | Hyperspherical confidence mapping: one network predicts magnitude and direction; the direction's distance from the unit circle is the uncertainty. |
| **Notebook** | [`Hyperspherical_Confidence_Mapping(HCM).ipynb`](<Hyperspherical_Confidence_Mapping(HCM).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_21_44](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-21-44) |
| **Executed output** | [`results/`](results/) (4 files: executed notebook(s) and Excel workbooks) |

## HCM metrics

| Metric | Value |
|---|--:|
| Cov@1Sigma | 56.8% |
| Cov@2Sigma | 84.2% |
| Cov@3Sigma | 94.7% |
| Coverage_95Interval | 84.2% |
| Average_95Interval_Width | 31.00 |
| Pearson | -0.323 |
| Spearman | -0.375 |
| ECE_reg | 3.10 |
| RMSE | 9.83 |
| MAE | 7.65 |
| Mean_CRPS | 5.70 |
| Mean_Negative_LogLikelihood | 4.05 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md).*
