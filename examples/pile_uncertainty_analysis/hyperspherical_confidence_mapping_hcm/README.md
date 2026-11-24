# Pile · NB8 Hyperspherical confidence mapping (HCM)

[← Pile README](../README.md) · [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md) · [report (PDF)](../../../papers/pile/paper.pdf)

| | |
|---|---|
| **What** | Hyperspherical confidence mapping: one network predicts magnitude and direction; the direction's distance from the unit circle is the uncertainty. |
| **Notebook** | [`Hyperspherical_Confidence_Mapping(HCM).ipynb`](<Hyperspherical_Confidence_Mapping(HCM).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_14_03](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-14-03) |
| **Executed output** | [`results/`](results/) (4 files: executed notebook(s) and Excel workbooks) |

## HCM metrics

| Metric | Value |
|---|--:|
| Cov@1Sigma | 75.0% |
| Cov@2Sigma | 100.0% |
| Cov@3Sigma | 100.0% |
| Coverage_95Interval | 100.0% |
| Average_95Interval_Width | 6.76 |
| Pearson | 0.237 |
| Spearman | -0.003 |
| ECE_reg | 0.706 |
| RMSE | 1.30 |
| MAE | 1.02 |
| Mean_CRPS | 0.756 |
| Mean_Negative_LogLikelihood | 1.75 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md).*
