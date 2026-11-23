# Report tools

Scripts that turn the stored results (`examples/<dataset>_uncertainty_analysis/*/results/`) into
the per-dataset reports in `papers/`, the dataset READMEs and the cross-dataset figures. They read
only the Excel files, executed notebooks, `best_params.json` and the results READMEs; no model is
re-run.

| Script | Purpose |
|---|---|
| `build_summary.py <repo> <out_dir> [dataset ...]` | Extracts the headline numbers of every notebook into `<out_dir>/<dataset>.json`. |
| `analysis.py` | Builds the coverage/width table of every model × interval method, the sampling tolerance and the recommendation rule. |
| `figures.py` | Matplotlib figures (PNG for READMEs, PDF for the papers). |
| `make_reports.py <repo> dataset ...` | Writes `papers/<dataset>/` (LaTeX, figures, PDF via [tectonic](https://tectonic-typesetting.github.io)) and `examples/<dataset>_uncertainty_analysis/README.md`. |
| `cross.py <repo> dataset ...` | Prints the cross-dataset table and writes `docs/figures/*_heatmap.png`. |
| `references.bib` | Bibliography used by every report. |

```bash
cd tools/reports
python build_summary.py ../.. summary            # all datasets
python make_reports.py ../.. concrete noise ...
python cross.py ../.. concrete noise ...
```

**Rules used in the reports.** Target coverage 90% (α = 0.1). An interval is *valid* when its
test coverage is at least 0.9 − 1.645·√(0.09/n_test). The recommended method is the narrowest
(median width over models) method that is valid for at least 75% of at least three base models;
methods whose widths are lower bounds (NexCP Full intervals that reach the edge of the label grid)
are not ranked. Extreme CP (one-sided) and ACP (learned level) are reported separately.
