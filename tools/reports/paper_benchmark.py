"""paper_benchmark.py <R>: the cross-dataset benchmark paper (papers/benchmark/paper.pdf).

Run after study.py (docs/figures, results/*.csv) and cross.py (heatmaps).
"""
import sys, os, json, math, shutil
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import analysis as A, texblocks as TB, study as ST
from make_reports import tex, pct, f, md2tex, tab, build_pdf, REPO, CONF, NAMES

def main(R):
    D = ST.long_table(); Mt = pd.read_csv(f"{R}/results/benchmark_methods.csv"); DT = pd.read_csv(f"{R}/results/benchmark_datasets.csv")
    out = f"{R}/papers/benchmark"; fd = f"{out}/figures"; os.makedirs(fd, exist_ok=True)
    for x in ["fig_pipeline", "fig_datasets", "fig_conformal_explainer", "fig_validity_tolerance", "fig_family_validity", "fig_model_based_coverage",
              "fig_conformal_ranking", "fig_acp_hcm", "fig_recommended"]:
        shutil.copy(f"{R}/docs/figures/{x}.pdf", f"{fd}/{x}.pdf")
    for x in ["coverage_heatmap", "width_heatmap"]: shutil.copy(f"{R}/docs/figures/{x}.png", f"{fd}/{x}.png")
    S = {ds: ST.load(ds) for ds in ST.DS}
    c = D[D.family.isin(CONF)]; nv, nc = int(c.valid.sum()), len(c)
    miss = c[~c.valid].groupby("dataset").size().sort_values(ascending=False)
    q = D[D.family == "Quantile regression"]
    # --- tables
    t_data = tab("The nine datasets. $p$: number of features; sd: standard deviation of the test target; threshold: lowest coverage accepted as valid (eq.~\\ref{eq:valid}).",
                 "tab:datasets", "lrrrlrr", "Dataset & Train & Test & $p$ & Target & sd & Threshold",
                 [f"{tex(ST.NAMES[r.dataset])} & {r.n_train} & {r.n_test} & {r.n_features} & \\texttt{{{tex(r.target)}}} & {f(r.target_std_test)} & {r.valid_threshold:.3f}\\\\" for r in DT.itertuples()])
    acc = []
    for ds in ST.DS:
        t = S[ds]["tuning"]; b = A.best_model(S[ds]); bt = min(t, key=lambda m: t[m]["test_rmse"]); sd = S[ds]["data"]["target_stats"]["std"]
        acc.append(f"{tex(ST.NAMES[ds])} & {tex(b)} & {f(t[b]['test_rmse'])} & {tex(bt)} & {f(t[bt]['test_rmse'])} & {t[bt]['test_rmse']/sd:.2f}\\\\")
    t_acc = tab("Point accuracy. Best model by 5-fold CV (training set) and best model on the test set; last column: best test RMSE divided by the test-target standard deviation.",
                "tab:acc", "lllllr", "Dataset & Best (CV) & its test RMSE & Best (test) & RMSE & RMSE/sd", acc)
    mb = []
    for ds in ST.DS:
        g = D[D.dataset == ds]; cr = [v.get("crossing_rate") for v in S[ds]["quantile_regression"].values() if v.get("crossing_rate") is not None]
        get = lambda m: g[g.method == m].coverage
        mb.append(f"{tex(ST.NAMES[ds])} & {pct(get('Quantile regression (raw)').median())} & {pct(np.median(cr)) if cr else '--'} & {pct(get('NGBoost (native)').median())} & "
                  f"{pct(get('PGBM (native)').median())} & {pct(get('IBUG').median())} & {pct(get('Treeffuser').median())} & {pct(get('CARD').median())}\\\\")
    t_mb = tab("Coverage of the model-based 90\\% intervals (median over base models where several). Crossing: median share of test points with crossing quantiles.",
               "tab:modelbased", "lrrrrrrr", "Dataset & QR & Crossing & NGBoost & PGBM & IBUG & Treeffuser & CARD", mb, resize=True)
    fam = D.groupby("family").agg(n=("valid", "size"), valid=("valid", "mean"), relw=("rel_width", "median")).reindex(A.FAMILY_ORDER).dropna()
    t_fam = tab("All model $\\times$ method $\\times$ dataset intervals pooled by family: number of intervals, share valid and median relative width.",
                "tab:family", "lrrr", "Family & Intervals & Valid & Median width/sd", [f"{tex(k)} & {int(r.n)} & {pct(r.valid)} & {r.relw:.2f}\\\\" for k, r in fam.iterrows()])
    pool = c.groupby("label").agg(n=("valid", "size"), valid=("valid", "mean"), relw=("rel_width", "median"), nds=("dataset", "nunique")).sort_values("relw")
    rec_count = DT.recommended_method.value_counts()
    t_pool = tab("Conformal methods pooled over all datasets and models: number of intervals, datasets, share valid, median relative width, and on how many datasets the method is recommended.",
                 "tab:pool", "lrrrrr", "Method & Intervals & Datasets & Valid & Median width/sd & Recommended",
                 [f"{tex(k)} & {int(r.n)} & {int(r.nds)} & {pct(r.valid)} & {r.relw:.2f} & {int(rec_count.get(k, 0))}\\\\" for k, r in pool.iterrows()])
    t_rec = tab("Recommended method per dataset: the narrowest method that is valid on at least 75\\% of the base models it wraps.", "tab:rec", "llrrr",
                "Dataset & Best model (CV) & Raw QR coverage & Recommended & Coverage / width",
                [f"{tex(ST.NAMES[r.dataset])} & {tex(r.best_model_cv)} & {pct(r.qr_coverage_median)} & {tex(r.recommended_method)} & {pct(r.recommended_coverage)} / {r.recommended_rel_width:.2f}\\\\" for r in DT.itertuples()])
    ah = []
    for ds in ST.DS:
        ac = S[ds].get("acp") or {}; h = S[ds].get("hcm") or {}
        sat = sum((v.get("Target_coverage") or 1) < 0.01 for v in ac.values())
        ok = [v for v in ac.values() if (v.get("Target_coverage") or 1) >= 0.01]
        ah.append(f"{tex(ST.NAMES[ds])} & {sat}/{len(ac)} & {pct(np.mean([v['Target_coverage'] for v in ok])) if ok else '--'} & {pct(np.mean([v['coverage'] for v in ok])) if ok else '--'} & "
                  f"{pct(h.get('Coverage_95Interval'))} & {f(h.get('Spearman'), 2)}\\\\")
    t_ah = tab("Adaptive coverage policies (ACP) and HCM per dataset. Saturated: models whose policy sits at $\\tilde\\alpha=0.999$; target and coverage are means over the other models.",
               "tab:acphcm", "lrrrrr", "Dataset & ACP saturated & ACP target & ACP coverage & HCM 95\\% cov. & HCM Spearman", ah, resize=True)
    nr = ("Two sets of jobs exceeded Kaggle's 12-hour limit and were not re-run (not enough compute): Notebook 6 for TabNet and PGBM on sediment load "
          "(where NexCP Full was also switched off for every model) and Notebook 6 for NGBoost and PGBM on pile settlement. These cells are missing, not estimated.")
    qmed = q.groupby("dataset").coverage.median()
    abstract = (f"Prediction intervals are the simplest honest answer to the question ``how far can this regression prediction be trusted?''. We benchmark "
                f"more than thirty uncertainty quantification methods, wrapped around nine tuned gradient-boosted, probabilistic and neural regressors, on nine "
                f"engineering and environmental datasets with 24 to 365 test points. Each method's 90\\% interval is checked with a sample-size-aware validity rule "
                f"and compared on width relative to the spread of the target. Model-based intervals are unreliable: raw quantile regression reaches a median "
                f"coverage of {pct(qmed.min())}--{pct(qmed.max())} across datasets with crossing quantiles on most test points, and native NGBoost and PGBM Gaussian "
                f"intervals are severely over-confident on most datasets. Conformal methods are valid in {nv} of {nc} dataset--model--method cases "
                f"({pct(nv/nc)}), with the misses concentrated on two datasets. Among valid methods widths differ by a factor of two or more, and the sharpest "
                f"valid method changes with the data: jackknife+-after-bootstrap, an online method (FACI, MFCS) or a weighted method (NexCP J+aB). "
                f"Every number is read from executed Kaggle notebooks released with the code, and each method is explained with its mathematics.")
    doc = TB.PREAMBLE + r"""
\title{How Far Can a Regression Prediction Be Trusted?\\[2pt] \large A Benchmark of Conformal, Distributional and Learned-Coverage Uncertainty Quantification on Nine Engineering and Environmental Datasets}
""" + TB.AUTHORS + r"""
\date{Regression UQ Benchmark v2.0 --- November 2026}
\begin{document}\maketitle
\begin{abstract}""" + abstract + r"""\end{abstract}
\tableofcontents
\newpage
\section{Introduction}
Regression models are used in engineering and environmental practice to predict quantities such as concrete strength, pile capacity and
settlement, scour depth, sediment load, streamflow salinity, noise level and water-quality indices. In these applications a prediction is
used to make a decision, and the decision depends on how wrong the prediction could be. A point prediction cannot say this.

Uncertainty quantification (UQ) adds an interval or a distribution to each prediction, and the literature offers dozens of ways to do so.
They differ in what they promise. \emph{Conformal prediction} gives a finite-sample guarantee of marginal coverage under exchangeability
\cite{vovk2005alrw,lei2018distribution,angelopoulos2023gentle}; \emph{weighted} and \emph{online} conformal methods relax that assumption
\cite{barber2023beyond,gibbs2021aci}; \emph{quantile regression} and \emph{predictive distributions} are correct only if the model is
\cite{koenker1978quantile,duan2020ngboost}; and newer methods learn their own coverage level \cite{gauthier2026acp} or extract uncertainty from a
single network \cite{choi2026hcm}. Practitioners rarely see these methods compared on the same small, tabular, real datasets they work with.

This paper runs one fixed pipeline of eight notebooks on nine datasets and asks:
\begin{enumerate}[label=\textbf{Q\arabic*.}]
\item Do model-based intervals (quantile regression, NGBoost, PGBM, IBUG, Treeffuser, CARD) reach their nominal coverage?
\item Do conformal methods keep their promise on real, small datasets, and where do they fail?
\item Among valid methods, which are the sharpest, and is the answer the same on every dataset?
\item What do learned-coverage policies and single-network uncertainty add?
\end{enumerate}
\paragraph{Contributions.} (i) An executed benchmark of $>30$ UQ methods $\times$ 9 base models $\times$ 9 datasets (""" + f"{len(D)}" + r""" two-sided intervals), with every executed notebook and raw output released; (ii) a self-contained explanation of the
mathematics of each method and of why its guarantee holds or fails; (iii) a sample-size-aware validity rule and a recommendation rule that turn the
results into a per-dataset choice; (iv) a correctness audit of the original notebooks, with each fix documented. A separate report for each
dataset gives the full detail.
""" + TB.SETUP + r"""
\section{Datasets}\label{sec:data}
""" + t_data + r"""
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/fig_datasets.pdf}\caption{Target distribution of each dataset on the training (filled) and test (outline) sets.}\label{fig:datasets}\end{figure}
Table~\ref{tab:datasets} and Fig.~\ref{fig:datasets} summarise the nine datasets. They are small (47--1095 training rows) and varied: four
have strongly right-skewed targets (sediment load, streamflow, scour, pile settlement), one has more features than a tree model can use reliably with
its sample size (WQI: 151 features, 190 rows), and two have tiny test sets (pile: 24, EWQI: 47) on which only large departures from 90\% coverage can
be detected. The train/test split is the one supplied with each dataset.

\section{Pipeline and protocol}\label{sec:pipeline}
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/fig_pipeline.pdf}\caption{The eight-notebook pipeline, run unchanged on every dataset.}\label{fig:pipeline}\end{figure}
Figure~\ref{fig:pipeline} shows the pipeline. Notebook 1 tunes the base models by 5-fold cross-validation on the training set and writes the
settings that notebooks 2--7 load; the test set is used only once, for evaluation. All runs used Kaggle CPU sessions; long notebooks were split into
one job per base model to fit the 12-hour session limit. """ + nr + r""" The v1.2 notebooks were audited method by method against the original papers
and reference code; every fix is listed in \texttt{docs/CHANGELOG\_FIXES.md}.
""" + TB.METHODS + TB.evaluation() + r"""
\begin{figure}[h]\centering\includegraphics[width=0.75\linewidth]{figures/fig_validity_tolerance.pdf}\caption{Lowest coverage accepted as valid, eq.~\eqref{eq:valid}, as a function of the test-set size, with the nine datasets marked.}\label{fig:tol}\end{figure}
\begin{figure}[h]\centering\includegraphics[width=\linewidth]{figures/fig_conformal_explainer.pdf}\caption{Split conformal prediction in three steps on synthetic data: score, cut-off, interval.}\label{fig:explainer}\end{figure}

\section{Results}\label{sec:results}
\subsection{Point accuracy}
""" + t_acc + r"""
Table~\ref{tab:acc} shows that the best model differs between datasets and that the CV choice is often, but not always, the best on the test set.
The tree ensembles are close to each other on every dataset; the differences between UQ methods are much larger than the differences between the
boosted base models, which is why the rest of the paper aggregates over base models.

\subsection{Q1: model-based intervals do not reach their nominal coverage}
""" + t_mb + r"""
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/fig_model_based_coverage.pdf}\caption{Coverage of the model-based 90\% intervals on each dataset. Small dots: individual base models; large dots: median; green bar: validity threshold.}\label{fig:modelbased}\end{figure}
\paragraph{How to read it.} In Fig.~\ref{fig:modelbased} each colour is one method; a method is reliable if its dots sit at or above the green bar
of every dataset. \paragraph{What it shows.}
\begin{itemize}
\item \textbf{Raw quantile regression undercovers}: its median coverage is below the threshold on """ + str(int((qmed < DT.set_index('dataset').valid_threshold.reindex(qmed.index)).sum())) + r""" of nine datasets, and the separately fitted quantiles cross on most test points of almost every dataset (Table~\ref{tab:modelbased}).
\item \textbf{Native NGBoost and PGBM Gaussians are over-confident}: their $\hat\sigma$ is far too small on most datasets (PGBM covers 2--42\% on eight of nine). Their PIT histograms are U-shaped (under-dispersed).
\item \textbf{IBUG and Treeffuser go the other way}: IBUG is usually close to or above 90\%; Treeffuser covers almost everything with very wide intervals.
\item \textbf{CARD} is sharp but undercovers on most datasets; it reaches the target only on the large sediment-load dataset.
\end{itemize}
\begin{insight}[Takeaway Q1]
A predictive distribution that scores well on CRPS can still give a badly calibrated interval. Without a calibration step, no model-based method is reliable across datasets.
\end{insight}

\subsection{Q2: conformal methods are valid almost everywhere}
""" + t_fam + r"""
\begin{figure}[t]\centering\includegraphics[width=0.95\linewidth]{figures/fig_family_validity.pdf}\caption{Share of model $\times$ method intervals that are valid, by family and dataset.}\label{fig:famvalid}\end{figure}
\begin{figure}[t]\centering\includegraphics[width=0.9\linewidth]{figures/coverage_heatmap.png}\caption{Median coverage (\%) of every two-sided method on every dataset. n/r: not run.}\label{fig:covheat}\end{figure}
\paragraph{What it shows.}
\begin{itemize}
\item Conformal intervals (MAPIE/PUNCC, NexCP, online, COP) are valid in \textbf{""" + f"{nv} of {nc}" + r"""} cases (""" + pct(nv / nc) + r""").
\item The misses concentrate on """ + ", ".join(f"{tex(ST.NAMES[k])} ({v})" for k, v in miss.items()) + r""". WQI has 151 features for 190 training rows, so the out-of-fold
residuals used for calibration come from models fitted on even fewer rows than the final model; pile settlement has a heavy right tail (skewness 2.4), and its few very large
settlements are hard to cover with residuals calibrated mostly on small ones.
\item Model-based families are valid far less often (Table~\ref{tab:family}).
\end{itemize}
\begin{insight}[Takeaway Q2]
On small real datasets the conformal guarantee is not just a theorem: it holds in practice for 95\% of the intervals, and the failures point to the
data (very high dimension, heavy tails), not to the methods.
\end{insight}

\subsection{Q3: the sharpest valid method depends on the data}
""" + t_pool + t_rec + r"""
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/fig_conformal_ranking.pdf}\caption{Median relative width of each conformal method on each dataset; colour gives the rank within the dataset (dark = sharpest).}\label{fig:rank}\end{figure}
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/fig_recommended.pdf}\caption{Relative width of every conformal method (grey), raw quantile regression (red) and the recommended method (star) per dataset.}\label{fig:rec}\end{figure}
\paragraph{What it shows.}
\begin{itemize}
\item Valid methods differ in width by a factor of two or more on the same dataset (Fig.~\ref{fig:rec}).
\item No method is the sharpest everywhere (Fig.~\ref{fig:rank}, Table~\ref{tab:rec}). J+aB (MAPIE) is recommended on """ + str(int(rec_count.get("J+aB (MAPIE)", 0))) + r""" datasets; the online
methods FACI and MFCS Full on """ + str(int(rec_count.get("FACI", 0) + rec_count.get("MFCS Full", 0))) + r"""; NexCP J+aB on the large, heavy-tailed sediment data, where recent-weighted residuals adapt to the scale of the target.
\item Jackknife+-after-bootstrap, which reuses every training point for calibration, is usually sharper than split conformal, which calibrates on half the data; CV+ is close to split conformal in width but more often valid.
\item CQR is valid but wide here: with a few hundred points the conformal correction has to repair poorly estimated quantile models.
\end{itemize}
\begin{insight}[Takeaway Q3]
Validity is easy to get; sharpness is not. A practitioner should run a small panel of valid methods (J+aB, CV+, and, if labels arrive sequentially,
FACI or MFCS) and pick the narrowest on held-out data, rather than commit to one method in advance.
\end{insight}

\subsection{Q4: learned coverage and single-network uncertainty}
""" + t_ah + r"""
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/fig_acp_hcm.pdf}\caption{Left: ACP realised coverage against its learned target. Right: HCM's Spearman correlation between absolute error and predicted spread.}\label{fig:acphcm}\end{figure}
ACP's guarantee is relative to its own learned level, and its realised coverage is always above that level (Fig.~\ref{fig:acphcm}, left). On
sediment load and streamflow, and for four models on EWQI, the policy saturated at $\tilde\alpha=0.999$: the learned level then carries no information and
every set has the same width. HCM's 95\% interval covers 80--100\%, but its predicted spread is negatively correlated with the error, or not at all, on six of
nine datasets (Fig.~\ref{fig:acphcm}, right), so on these small tabular problems a single network does not know when it is wrong.

\section{Discussion}\label{sec:discussion}
\paragraph{A practical recipe.} (1) Tune the base model by cross-validation only. (2) Wrap it in jackknife+-after-bootstrap or CV+; these are valid
on almost every dataset and model here and need no extra data. (3) If labels arrive one at a time, add an online method (FACI, MFCS) and let it adapt.
(4) Check coverage against the sample-size threshold of eq.~\eqref{eq:valid}, not against exactly 90\%. (5) Use predictive distributions for their CRPS
and shape, not for their raw intervals, unless they are conformalised.
\paragraph{When the guarantee bends.} The two datasets with most conformal misses share a feature: the calibration residuals do not represent the test
residuals (too few rows per feature on WQI; a heavy right tail on pile settlement). Weighted (NexCP) and online methods are designed for
exactly this, and they recover coverage on pile settlement (MFCS Full is recommended there).
""" + TB.LIMITATIONS + TB.FUTURE + r"""
\section{Conclusions}
\begin{itemize}
\item Raw quantile regression and native Gaussian predictive distributions do not deliver their nominal 90\% coverage on these datasets.
\item Conformal methods are valid in """ + f"{nv} of {nc}" + r""" cases; their failures are explained by the data.
\item The sharpest valid method depends on the dataset; J+aB, FACI, MFCS Full, NexCP J+aB and split conformal are each recommended at least once.
\item Learned-coverage policies can saturate, and single-network uncertainty is not reliably informative on small tabular data.
\end{itemize}
\paragraph{Per-dataset reports.} \texttt{papers/<dataset>/paper.pdf} for """ + ", ".join(tex(ST.NAMES[d]) for d in ST.DS) + r""".
\paragraph{Reproducibility.} Code, executed notebooks and raw outputs: \url{""" + REPO + r"""} (branch \texttt{version2.0}). Machine-readable results:
\texttt{results/benchmark\_long.csv} (every interval), \texttt{benchmark\_methods.csv}, \texttt{benchmark\_datasets.csv}. Every Kaggle run is listed in
\texttt{docs/run\_registry.tsv} and in each dataset's report.
\paragraph{Acknowledgements.} Computations were run on Kaggle's free CPU sessions.
\FloatBarrier
\bibliographystyle{plainnat}\bibliography{references}
\end{document}
"""
    rc, err, warn = build_pdf(R, "benchmark", doc, folder=out)
    print("benchmark pdf rc", rc, err if rc else "", "| warnings:", warn[:5])

if __name__ == "__main__":
    main(sys.argv[1])
