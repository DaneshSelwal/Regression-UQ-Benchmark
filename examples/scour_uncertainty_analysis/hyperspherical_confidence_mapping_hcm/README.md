# Scour · NB8 Hyperspherical confidence mapping (HCM)

[← Scour README](../README.md) · [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md) · [report (PDF)](../../../papers/scour/paper.pdf)

| | |
|---|---|
| **What** | Hyperspherical confidence mapping: one network predicts magnitude and direction; the direction's distance from the unit circle is the uncertainty. |
| **Notebook** | [`Hyperspherical_Confidence_Mapping(HCM).ipynb`](<Hyperspherical_Confidence_Mapping(HCM).ipynb>) |
| **Kaggle run(s)** | [REG22_11_2026_21_48](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-21-48) |
| **Executed output** | [`results/`](results/) (4 files: executed notebook(s) and Excel workbooks) |

## HCM metrics

| Metric | Value |
|---|--:|
| Cov@1Sigma | 64.1% |
| Cov@2Sigma | 91.0% |
| Cov@3Sigma | 91.0% |
| Coverage_95Interval | 91.0% |
| Average_95Interval_Width | 2.40 |
| Pearson | -0.157 |
| Spearman | -0.111 |
| ECE_reg | 0.276 |
| RMSE | 1.04 |
| MAE | 0.662 |
| Mean_CRPS | 0.507 |
| Mean_Negative_LogLikelihood | 2.09 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md).*
