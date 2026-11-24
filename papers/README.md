# Papers

[← main README](../README.md)

Ten arXiv-style reports by Danesh Selwal and Dr. Mahesh Pal. Each explains every method with its mathematics and the reasoning behind
its guarantee, describes the data and the pipeline, answers the research questions with figures and tables (each with a *how to
read it* paragraph and a takeaway), and lists the Kaggle runs that produced the numbers.

| Report | Scope | PDF | Source |
|---|---|---|---|
| **Benchmark** | All nine datasets: which UQ methods keep their promise, and which are the sharpest | [paper.pdf](benchmark/paper.pdf) | [paper.tex](benchmark/paper.tex) |
| Concrete | 189 / 95 rows, 7 features | [paper.pdf](concrete/paper.pdf) | [paper.tex](concrete/paper.tex) |
| Noise | 168 / 60 rows, 5 features | [paper.pdf](noise/paper.pdf) | [paper.tex](noise/paper.tex) |
| Pile | 81 / 24 rows, 7 features | [paper.pdf](pile/paper.pdf) | [paper.tex](pile/paper.tex) |
| Pile settlement | 371 / 186 rows, 8 features | [paper.pdf](pile_settlement/paper.pdf) | [paper.tex](pile_settlement/paper.tex) |
| Scour | 154 / 78 rows, 7 features | [paper.pdf](scour/paper.pdf) | [paper.tex](scour/paper.tex) |
| Sediment load | 1095 / 365 rows, 3 features | [paper.pdf](sediment_load/paper.pdf) | [paper.tex](sediment_load/paper.tex) |
| Streamflow | 192 / 96 rows, 4 features | [paper.pdf](streamflow/paper.pdf) | [paper.tex](streamflow/paper.tex) |
| Water quality (EWQI) | 47 / 47 rows, 14 features | [paper.pdf](water_quality_ewqi/paper.pdf) | [paper.tex](water_quality_ewqi/paper.tex) |
| Water quality (WQI) | 190 / 82 rows, 151 features | [paper.pdf](water_quality_wqi/paper.pdf) | [paper.tex](water_quality_wqi/paper.tex) |

**Structure of a dataset report:** introduction and research questions → problem set-up and notation → data analysis →
pipeline (figure) → methods (tuning, quantile regression, predictive distributions, split / cross / bootstrap conformal, NexCP,
online conformal, COP, Extreme CP, ACP, HCM) → evaluation protocol (validity rule, recommendation rule) → results RQ1–RQ6 →
discussion, limitations, future work (`reguq`) → conclusions → Kaggle runs → appendix with every model × method result.

Every report is generated from the stored results by [`tools/reports/`](../tools/reports/README.md), so the numbers in the text,
tables and figures are the ones in the Excel files and executed notebooks.
