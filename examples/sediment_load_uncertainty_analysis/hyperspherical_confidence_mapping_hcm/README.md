# Sediment load · NB8 Hyperspherical confidence mapping (HCM)

[← Sediment load README](../README.md) · [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md) · [report (PDF)](../../../papers/sediment_load/paper.pdf)

| | |
|---|---|
| **What** | Hyperspherical confidence mapping: one network predicts magnitude and direction; the direction's distance from the unit circle is the uncertainty. |
| **Notebook** | [`Hyperspherical_Confidence_Mapping(HCM).ipynb`](<Hyperspherical_Confidence_Mapping(HCM).ipynb>) |
| **Kaggle run(s)** | [REG23_11_2026_01_45](https://www.kaggle.com/code/daneshjangra/reg23-11-2026-01-45) |
| **Executed output** | [`results/`](results/) (4 files: executed notebook(s) and Excel workbooks) |

## HCM metrics

| Metric | Value |
|---|--:|
| Cov@1Sigma | 44.4% |
| Cov@2Sigma | 100.0% |
| Cov@3Sigma | 100.0% |
| Coverage_95Interval | 100.0% |
| Average_95Interval_Width | 3,819 |
| Pearson | 0.723 |
| Spearman | -0.573 |
| ECE_reg | 279.8 |
| RMSE | 911.4 |
| MAE | 755.9 |
| Mean_CRPS | 489.0 |
| Mean_Negative_LogLikelihood | 8.06 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md).*
