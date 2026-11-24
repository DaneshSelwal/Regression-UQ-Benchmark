# Scour · NB8 Hyperspherical confidence mapping (HCM)

[← Scour README](../README.md) · [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Hyperspherical confidence mapping: one network predicts magnitude and direction; the direction's distance from the unit circle is the uncertainty. |
| **Notebook** | [`Hyperspherical_Confidence_Mapping(HCM).ipynb`](<Hyperspherical_Confidence_Mapping(HCM).ipynb>) |
| **Kaggle run(s)** | [REG24_11_2026_15_22](https://www.kaggle.com/code/daneshselwal/reg24-11-2026-15-22), [REG22_11_2026_21_48](https://www.kaggle.com/code/daneshjangra/reg22-11-2026-21-48) |
| **Executed output** | [`results/`](results/) (4 files: executed notebook(s) and Excel workbooks) |

## HCM metrics

| Metric | Value |
|---|--:|
| Cov@1Sigma | 78.2% |
| Cov@2Sigma | 92.3% |
| Cov@3Sigma | 97.4% |
| Coverage_95Interval | 92.3% |
| Average_95Interval_Width | 2.28 |
| Pearson | 0.520 |
| Spearman | 0.135 |
| ECE_reg | 0.333 |
| RMSE | 0.394 |
| MAE | 0.272 |
| Mean_CRPS | 0.226 |
| Mean_Negative_LogLikelihood | 0.559 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md).*
