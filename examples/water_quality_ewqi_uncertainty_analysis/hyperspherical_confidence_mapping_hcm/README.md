# Water quality (EWQI) · NB8 Hyperspherical confidence mapping (HCM)

[← Water quality (EWQI) README](../README.md) · [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md) · [report (PDF)](../report/paper.pdf)

| | |
|---|---|
| **What** | Hyperspherical confidence mapping: one network predicts magnitude and direction; the direction's distance from the unit circle is the uncertainty. |
| **Notebook** | [`Hyperspherical_Confidence_Mapping(HCM).ipynb`](<Hyperspherical_Confidence_Mapping(HCM).ipynb>) |
| **Kaggle run(s)** | [REG24_11_2026_15_25](https://www.kaggle.com/code/daneshselwal/reg24-11-2026-15-25), [REG22_11_2026_21_43](https://www.kaggle.com/code/daneshselwal/reg22-11-2026-21-43) |
| **Executed output** | [`results/`](results/) (4 files: executed notebook(s) and Excel workbooks) |

## HCM metrics

| Metric | Value |
|---|--:|
| Cov@1Sigma | 83.0% |
| Cov@2Sigma | 97.9% |
| Cov@3Sigma | 100.0% |
| Coverage_95Interval | 97.9% |
| Average_95Interval_Width | 266.2 |
| Pearson | 0.430 |
| Spearman | 0.306 |
| ECE_reg | 26.52 |
| RMSE | 58.77 |
| MAE | 41.39 |
| Mean_CRPS | 30.78 |
| Mean_Negative_LogLikelihood | 5.39 |

*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/08_hyperspherical_confidence_mapping/README.md).*
