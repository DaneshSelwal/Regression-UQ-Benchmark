"""make_reports.py <R> <ds...>: figures + LaTeX paper (papers/<ds>/) + example README for each dataset."""
import sys, os, re, json, math, shutil, subprocess, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import analysis as A, figures as F

NAMES = {"concrete": "Concrete", "noise": "Noise", "pile": "Pile", "pile_settlement": "Pile settlement", "scour": "Scour",
         "sediment_load": "Sediment load", "streamflow": "Streamflow", "water_quality_ewqi": "Water quality (EWQI)",
         "water_quality_wqi": "Water quality (WQI)"}
PHASES = [("hyperparameter_tuning", "NB1 Hyperparameter tuning (Optuna)"), ("quantile_regression", "NB2 Quantile regression"),
          ("probabilistic_distribution", "NB3 Distributional (NGBoost, PGBM, IBUG, Treeffuser)"),
          ("probabilistic_distribution_card", "NB4 Distributional (CARD diffusion)"),
          ("conformal_predictions_mapie_puncc", "NB5 Conformal (MAPIE, PUNCC)"),
          ("conformal_predictions_nexcp_adaptivecp_mfcs", "NB6 NexCP and online conformal"),
          ("conformal_predictions_adaptive_coverage_policies", "NB7 Adaptive coverage policies (ACP)"),
          ("hyperspherical_confidence_mapping_hcm", "NB8 Hyperspherical confidence mapping (HCM)")]
GH = "https://github.com/DaneshSelwal/Regression_Uncertainty_Quantification_Analysis/blob/version2.0"

def f(x, d=3):
    if x is None or (isinstance(x, float) and math.isnan(x)): return "--"
    x = float(x); a = abs(x)
    if a >= 1000: return f"{x:,.0f}"
    if a >= 100: return f"{x:.1f}"
    if a >= 1: return f"{x:.{max(d-1,1)}f}" if a < 10 else f"{x:.2f}"
    return f"{x:.{d}f}"
def pct(x): return "--" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100*x:.1f}\\%"
def pmd(x): return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{100*x:.1f}%"
def tex(s): return str(s).replace("\\", "/").replace("_", "\\_").replace("%", "\\%").replace("&", "\\&").replace("#", "\\#")

def registry(R):
    reg = pd.read_csv(f"{R}/docs/run_registry.tsv", sep="\t", dtype=str)
    return dict(zip(reg.code_name, reg.slug))

def not_run_models(S):
    t = S.get("nb6_readme") or ""; m = re.search(r"\*\*Not run \(out of compute\):\*\*\s*(.+)", t)
    return [x.strip(" .") for x in m.group(1).split(",")] if m else []

def facts(R, ds):
    S = json.load(open(f"{HERE}/summary/{ds}.json")); df = A.rows(S); L = A.league(df)
    rec, combo, ok = A.recommend(df, L); n = S["data"]["n_test"]; tl = A.tol(n)
    t = S["tuning"]; best = A.best_model(S); best_test = min(t, key=lambda m: t[m]["test_rmse"])
    qr = df[df.family == "Quantile regression"]
    dist = []
    for k, v in S["quantile_regression"].items():
        if v.get("crps") is not None: dist.append(("Quantile regression", k, v["crps"]))
    for k, v in S["probabilistic"].items():
        if v.get("crps_gauss") is not None:
            b = k.replace(" (TF)", ""); dist.append(("Treeffuser" if "(TF)" in k else ("native" if b in ("NGBoost", "PGBM") else "IBUG"), b, v["crps_gauss"]))
    for k, v in S["card"].items():
        if v.get("crps_gauss") is not None: dist.append(("CARD", k, v["crps_gauss"]))
    D = pd.DataFrame(dist, columns=["method", "model", "crps"]).sort_values("crps")
    fam = df[df.valid].groupby("family").rel_width.median().sort_values()
    nexfull_off = all((d.get("NexCP Full", {}).get("coverage") is None) for d in S["nexcp_online"].values()) and len(S["nexcp_online"]) > 0
    return dict(S=S, df=df, L=L, rec=rec, combo=combo, ok=ok, n=n, tol=tl, best=best, best_test=best_test, qr=qr, D=D, fam=fam,
                notrun=not_run_models(S), nexfull_off=nexfull_off, sd=S["data"]["target_stats"]["std"])

def findings(F_):
    """Short, number-driven conclusions (shared by paper and README)."""
    S, df, L, n = F_["S"], F_["df"], F_["L"], F_["n"]; t = S["tuning"]; out = []
    b, bt = F_["best"], F_["best_test"]
    out.append(f"Point accuracy: {b} has the lowest 5-fold CV RMSE ({f(t[b]['cv_rmse'])}); on the test set the lowest RMSE is {bt} ({f(t[bt]['test_rmse'])}, target std {f(F_['sd'])}).")
    q = F_["qr"]
    if len(q):
        cr = [v.get("crossing_rate") for v in S["quantile_regression"].values() if v.get("crossing_rate") is not None]
        out.append(f"Raw quantile regression undercovers: median 90%-interval coverage {pmd(q.coverage.median())} (range {pmd(q.coverage.min())}–{pmd(q.coverage.max())}); "
                   f"{int(q.valid.sum())} of {len(q)} models reach the target. Quantile crossing occurs on a median {pmd(np.median(cr)) if cr else '–'} of test points.")
    D = F_["D"]
    if len(D):
        r = D.iloc[0]; out.append(f"Best probabilistic score: {r.model} ({r.method}) with CRPS {f(r.crps)}.")
    for nm in ("NGBoost (native)", "PGBM (native)"):
        g = df[df.label == nm]
        if len(g) and g.coverage.iloc[0] < A.TARGET - F_["tol"]:
            out.append(f"{nm} predictive distributions are over-confident here: the Gaussian 90% interval covers only {pmd(g.coverage.iloc[0])}.")
    c = df[df.family == "CARD"]
    if len(c): out.append(f"CARD (diffusion) intervals: median coverage {pmd(c.coverage.median())} at {f(c.rel_width.median(),2)}× the target std.")
    rec = F_["rec"]
    if rec is not None:
        out.append(f"Recommended interval method: **{rec['label']}** – the narrowest method that reaches the 90% target (within sampling tolerance, n={n}) on ≥75% of models: "
                   f"median coverage {pmd(rec.cov_med)}, median width {f(rec.w_med)} ({f(rec.relw_med,2)}× target std).")
    cb = F_["combo"]
    if cb is not None:
        out.append(f"Narrowest single valid interval (coverage within tolerance of 90%): {cb['label']} on {cb.model}, coverage {pmd(cb.coverage)}, mean width {f(cb.width)}.")
    fam = F_["fam"]
    if len(fam) >= 2:
        out.append("Among valid intervals, median width / target std by family: " + "; ".join(f"{k} {v:.2f}" for k, v in fam.items()) + ".")
    ac = S.get("acp") or {}
    if ac:
        sat = sorted(k for k, v in ac.items() if (v.get("Target_coverage") or 1) < 0.01)
        ok_ = {k: v for k, v in ac.items() if k not in sat}
        if ok_:
            cv = [v["coverage"] for v in ok_.values()]; tg = [v["Target_coverage"] for v in ok_.values() if v.get("Target_coverage") is not None]
            out.append(f"ACP learns its own coverage level (mean learned target {pmd(np.mean(tg)) if tg else '–'}); realised coverage of its e-value sets is {pmd(min(cv))}–{pmd(max(cv))} across models.")
        if sat:
            cv = [ac[k]["coverage"] for k in sat]
            out.append(f"ACP policy saturated at the boundary α̃ = 0.999 for {len(sat)} of {len(ac)} models ({', '.join(sat)}): the sets have constant width and the learned level carries no information; realised coverage {pmd(min(cv))}–{pmd(max(cv))}.")
    h = S.get("hcm") or {}
    if h.get("Coverage_95Interval") is not None:
        out.append(f"HCM (single network, 95% interval): coverage {pmd(h['Coverage_95Interval'])}, RMSE {f(h.get('RMSE'))}, error–uncertainty Spearman correlation {f(h.get('Spearman'),2)}.")
    if F_["notrun"]: out.append(f"Not run (out of compute, Kaggle 12-hour limit): NB6 for {', '.join(F_['notrun'])}.")
    if F_["nexfull_off"]: out.append("NexCP Full conformal was not run on this dataset (out of compute).")
    return out

# ---------------------------------------------------------------- figures
def make_figs(R, ds, F_):
    d = f"{R}/papers/{ds}/figures"; S = F_["S"]
    F.accuracy(S, d); F.cov_width(F_["df"], S, d); F.league_plot(F_["L"], S, d); F.crps_plot(S, d)
    m = F_["best"]; meth = F.intervals(R, S, d, m)
    if meth is None:
        for mm in sorted(S["mapie_puncc"]):
            meth = F.intervals(R, S, d, mm)
            if meth: m = mm; break
    F_["int_model"] = m

# ---------------------------------------------------------------- run links
def runs(R, ds):
    reg = registry(R); rn = A.run_names(R, ds); out = []; tune = set(rn.get("hyperparameter_tuning", []))
    for ph, title in PHASES:
        for c in rn.get(ph, []):
            if ph != "hyperparameter_tuning" and c in tune: continue   # the tuning run is an input, listed once
            sl = reg.get(c)
            if sl: out.append((ph, title, c, sl, f"https://www.kaggle.com/code/{sl}"))
    return out

# ---------------------------------------------------------------- LaTeX
def latex(R, ds, F_):
    S = F_["S"]; dd = S["data"]; nm = NAMES[ds]; t = S["tuning"]; n = F_["n"]; L = F_["L"]
    fx = dd["features"] if len(dd["features"]) <= 20 else dd["features"][:3] + ["...", dd["features"][-1]]
    feats = ", ".join(f"\\texttt{{{tex(x)}}}" if x != "..." else "\\dots" for x in fx); ts = dd["target_stats"]
    fl = findings(F_)
    clean = lambda s: tex(s).replace("**", "").replace("–", "--").replace("≥", "$\\geq$").replace("×", "$\\times$").replace("α̃", "$\\tilde\\alpha$")
    rec = F_["rec"]
    abstract = (f"We benchmark uncertainty quantification (UQ) methods for tabular regression on the {nm.lower()} dataset "
                f"({dd['n_train']} training and {n} test samples, {len(dd['features'])} features). Ten gradient-boosted, probabilistic and neural "
                f"regressors are tuned with Optuna and then wrapped by quantile regression, distributional methods (NGBoost, PGBM, IBUG, Treeffuser, CARD), "
                f"split, cross and bootstrap conformal prediction (MAPIE, PUNCC), weighted and online conformal methods (NexCP, ACI, FACI, SF-OGD, SAOCP, MFCS, COP, Extreme CP), "
                f"adaptive coverage policies (ACP) and hyperspherical confidence mapping (HCM). At a nominal 90\\% level, "
                + (f"raw quantile regression reaches a median coverage of only {pct(F_['qr'].coverage.median())}, while conformal methods restore validity; "
                   if len(F_['qr']) else "")
                + (f"the narrowest method that is valid on most models is {tex(rec['label'])} (median coverage {pct(rec.cov_med)}, median width {f(rec.relw_med,2)} target standard deviations). "
                   if rec is not None else "")
                + "All experiments ran on Kaggle; the executed notebooks and raw outputs are linked from this report.")
    T = []
    # tuning table
    rows = "\n".join(f"{tex(m)} & {tex(t[m].get('pruner') or '--')} & {f(t[m]['cv_rmse'])} & {f(t[m]['test_rmse'])} & {f(t[m].get('test_r'))} \\\\"
                     for m in sorted(t, key=lambda m: t[m]["cv_rmse"]))
    T.append(r"""\begin{table}[t]\centering\small
\caption{Tuned point predictors, sorted by 5-fold cross-validated RMSE on the training set. $r$: Pearson correlation on the test set.}\label{tab:tuning}
\begin{tabular}{llrrr}\toprule Model & Pruner & CV RMSE & Test RMSE & Test $r$\\\midrule
""" + rows + r"""
\bottomrule\end{tabular}\end{table}""")
    # QR + distributional table
    dr = []
    for m, v in sorted(S["quantile_regression"].items(), key=lambda kv: kv[1].get("crps") or 9e9):
        dr.append(f"QR & {tex(m)} & {pct(v.get('coverage90'))} & {f(v.get('width90'))} & {f(v.get('crps'))} & {pct(v.get('crossing_rate'))}\\\\")
    for k, v in sorted(S["probabilistic"].items(), key=lambda kv: kv[1].get("crps_gauss") or 9e9):
        b = k.replace(" (TF)", ""); meth = "Treeffuser" if "(TF)" in k else ("Native" if b in ("NGBoost", "PGBM") else "IBUG")
        dr.append(f"{meth} & {tex(b)} & {pct(v.get('coverage90_gauss'))} & {f(v.get('width90_gauss'))} & {f(v.get('crps_gauss'))} & --\\\\")
    for m, v in sorted(S["card"].items(), key=lambda kv: kv[1].get("crps_gauss") or 9e9):
        dr.append(f"CARD & {tex(m)} & {pct(v.get('coverage90_gauss'))} & {f(v.get('width90_gauss'))} & {f(v.get('crps_gauss'))} & --\\\\")
    T.append(r"""\begin{table}[t]\centering\small
\caption{Quantile regression and distributional methods at the 90\% level. For distributional methods the interval is $\hat\mu\pm1.645\hat\sigma$ and the CRPS is the closed-form Gaussian CRPS of $(\hat\mu,\hat\sigma)$; for quantile regression the CRPS is the pinball-loss approximation over the predicted quantiles. Crossing: share of test points whose predicted quantiles are not monotone.}\label{tab:dist}
\begin{tabular}{llrrrr}\toprule Method & Model & Coverage & Width & CRPS & Crossing\\\midrule
""" + "\n".join(dr) + r"""
\bottomrule\end{tabular}\end{table}""")
    # league table
    lr = "\n".join(f"{tex(r.family)} & {tex(r.label)}{'$^\\dagger$' if r.truncated else ''} & {r.n} & {pct(r.cov_med)} & {pct(r.cov_min)}--{pct(r.cov_max)} & {f(r.w_med)} & {r.relw_med:.2f} & {int(round(100*r.share_valid))}\\% \\\\"
                   for r in L.itertuples())
    T.append(r"""\begin{table}[p]\centering\small
\caption{All interval methods at the 90\% target, aggregated over the base models they wrap ($n$). Width: median over models of the mean test-interval width; rel.: divided by the standard deviation of the test target. $^\dagger$: some intervals reach the edge of the label grid, so the reported width is a lower bound; the method is not ranked. Valid: share of models whose coverage is at least $0.9-1.645\sqrt{0.09/n_{\text{test}}}=""" + f"{A.TARGET-F_['tol']:.3f}" + r"""$.}\label{tab:league}
\resizebox{\textwidth}{!}{\begin{tabular}{llrrcrrr}\toprule Family & Method & $n$ & Coverage (median) & Range & Width & Rel. & Valid\\\midrule
""" + lr + r"""
\bottomrule\end{tabular}}\end{table}""")
    ex = A.extreme(S)
    if ex:
        er = "\n".join(f"{tex(m)} & {pct(v['coverage'])} & {f(v['width'])}\\\\" for m, v in sorted(ex.items()))
        T.append(r"""\begin{table}[t]\centering\small
\caption{Extreme conformal prediction: one-sided upper bound $\hat y+q$ (no lower bound). Coverage is $P(y\le \hat y+q)$; the last column is $q$.}\label{tab:extreme}
\begin{tabular}{lrr}\toprule Model & Coverage & $q$\\\midrule
""" + er + r"""
\bottomrule\end{tabular}\end{table}""")
    ac = S.get("acp") or {}
    if ac:
        ar = "\n".join(f"{tex(m)} & {pct(v.get('Target_coverage'))} & {pct(v['coverage'])} & {f(v.get('interval_width'))} & {f(v.get('rmse'))} & {f(v.get('best_lambda'))}\\\\" for m, v in sorted(ac.items()))
        T.append(r"""\begin{table}[t]\centering\small
\caption{Adaptive coverage policies: mean learned target coverage $1-\tilde\alpha$, realised coverage and width of the e-value prediction sets.}\label{tab:acp}
\begin{tabular}{lrrrrr}\toprule Model & Target & Coverage & Width & RMSE & $\lambda$\\\midrule
""" + ar + r"""
\bottomrule\end{tabular}\end{table}""")
    h = S.get("hcm") or {}
    hc = ""
    if h:
        keys = [("Coverage_95Interval", "Coverage of 95\\% interval", pct), ("Average_95Interval_Width", "Mean 95\\% width", f), ("Cov@1Sigma", "Coverage at $1\\sigma$", pct),
                ("Cov@2Sigma", "Coverage at $2\\sigma$", pct), ("Cov@3Sigma", "Coverage at $3\\sigma$", pct), ("RMSE", "RMSE", f), ("MAE", "MAE", f),
                ("Mean_CRPS", "CRPS", f), ("Mean_Negative_LogLikelihood", "NLL", f), ("Spearman", "Spearman (|error|, $\\sigma$)", f), ("ECE_reg", "Regression ECE", f)]
        hc = "\n".join(f"{lab} & {fn(h.get(k))}\\\\" for k, lab, fn in keys if h.get(k) is not None)
        T.append(r"""\begin{table}[t]\centering\small
\caption{Hyperspherical confidence mapping (one network).}\label{tab:hcm}
\begin{tabular}{lr}\toprule Metric & Value\\\midrule
""" + hc + r"""
\bottomrule\end{tabular}\end{table}""")
    rl = runs(R, ds)
    rt = "\n".join(f"{tex(title.split(' ',1)[0])} & \\texttt{{{tex(c)}}} & \\url{{{u}}}\\\\" for ph, title, c, sl, u in rl)
    exc = []
    if F_["notrun"]: exc.append(f"NB6 was not run for {', '.join(F_['notrun'])}: these jobs exceed Kaggle's 12-hour limit (not enough compute).")
    if F_["nexfull_off"]: exc.append("NexCP Full conformal (a grid search over candidate labels) was switched off for this dataset for the same reason.")
    findings_tex = "\n".join(f"\\item {clean(x)}" for x in fl)
    best = F_["best"]
    doc = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}\usepackage{amsmath,amssymb,booktabs,graphicx,xcolor}\usepackage[numbers,sort&compress]{natbib}
\usepackage[colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}\usepackage{url}
\usepackage{caption}\captionsetup{font=small,labelfont=bf}\usepackage{placeins}
\title{Uncertainty Quantification for Tabular Regression:\\ a Benchmark on the """ + nm + r""" Dataset}
\author{Prakriti Bisht \and Danesh Selwal}
\date{Regression UQ Analysis v2.0 --- """ + "November 2026" + r"""}
\begin{document}\maketitle
\begin{abstract}""" + abstract + r"""\end{abstract}

\section{Introduction}
A point prediction says nothing about how far the truth may be from it. Uncertainty quantification attaches to each prediction either a predictive distribution or a prediction interval $[\ell(x),u(x)]$ that should contain the true response with a chosen probability $1-\alpha$. Methods differ in their guarantees: quantile regression and parametric distributional models are only asymptotically or approximately calibrated, while conformal prediction gives finite-sample, distribution-free marginal coverage under exchangeability \citep{vovk2005alrw,lei2018distribution,angelopoulos2023gentle}. This report applies one common, fixed pipeline of eight notebooks to the """ + nm.lower() + r""" dataset and compares the methods on coverage, interval width and proper scoring rules. Companion reports with the same structure cover the other eight datasets of the project.

\section{Data}
The training set has """ + str(dd["n_train"]) + r""" rows and the test set """ + str(n) + r""" rows. The """ + str(len(dd["features"])) + r""" input features are """ + feats + r"""; the target is \texttt{""" + tex(dd["target"]) + r"""}. On the test set the target has mean """ + f(ts["mean"]) + r""", standard deviation """ + f(ts["std"]) + r""", median """ + f(ts["50%"]) + r""" and range """ + f(ts["min"]) + r"""--""" + f(ts["max"]) + r""". The train/test split is the one supplied with the data; the test set is used only for the final evaluation (never for tuning or calibration).

\section{Methods}
\subsection{Point predictors and tuning}
Ten regressors are used as base models: random forest \citep{breiman2001rf}, gradient boosting \citep{friedman2001gbm}, histogram gradient boosting \citep{pedregosa2011sklearn}, XGBoost \citep{chen2016xgboost}, LightGBM \citep{ke2017lightgbm}, CatBoost \citep{prokhorenkova2018catboost}, GPBoost \citep{sigrist2022gpboost}, NGBoost \citep{duan2020ngboost}, PGBM \citep{sprangers2021pgbm} and TabNet \citep{arik2021tabnet}. Hyperparameters are tuned with Optuna \citep{akiba2019optuna} by minimising 5-fold cross-validated RMSE on the training set; for each model the best of several pruners (median, Hyperband \citep{li2018hyperband}, Wilcoxon, none) is kept. Random forest is used only in tuning.

\subsection{Quantile regression}
Each boosted model is fitted to the pinball loss $\rho_\tau(r)=r\,(\tau-\mathbb{1}\{r<0\})$ \citep{koenker1978quantile} for $\tau\in\{0.05,\dots,0.95\}$; the 90\% interval is $[\hat q_{0.05}(x),\hat q_{0.95}(x)]$. Separately fitted quantiles can cross \citep{chernozhukov2010rearrangement}; we report the crossing rate.

\subsection{Distributional methods}
NGBoost and PGBM output a Gaussian $\mathcal N(\hat\mu(x),\hat\sigma^2(x))$. IBUG \citep{brophy2022ibug} estimates $\hat\sigma(x)$ from the training targets of the $k$ most affine training points of a boosted ensemble. Treeffuser \citep{beltranvelez2024treeffuser} learns a conditional diffusion with boosted trees and CARD \citep{han2022card} a conditional denoising diffusion model \citep{ho2020ddpm}; samples are summarised by their mean and standard deviation. Every predictive distribution is scored with the continuous ranked probability score \citep{gneiting2007scoring,hersbach2000crps}, which for a Gaussian is
\begin{equation}
\mathrm{CRPS}(\mathcal N(\mu,\sigma^2),y)=\sigma\Big[z\,(2\Phi(z)-1)+2\varphi(z)-\tfrac{1}{\sqrt\pi}\Big],\quad z=\tfrac{y-\mu}{\sigma},
\end{equation}
and the 90\% interval is $\hat\mu\pm1.645\hat\sigma$. PIT histograms and calibration curves \citep{gneiting2007sharpness,kuleshov2018calibrated} are in the notebooks.

\subsection{Conformal prediction}
\emph{Split conformal} \citep{vovk2005alrw,lei2018distribution} holds out a calibration set $\mathcal C$, computes scores $s_i=|y_i-\hat f(x_i)|$ and returns $\hat f(x)\pm\hat q$, with $\hat q$ the $\lceil(1-\alpha)(|\mathcal C|+1)\rceil$-th smallest score; then $P(Y\in\hat C(X))\ge1-\alpha$ for exchangeable data. \emph{CV+} and \emph{jackknife+-after-bootstrap} (J+aB) reuse all data through out-of-fold residuals \citep{barber2021jackknife,kim2020jab}, with guarantee $1-2\alpha$. \emph{CQR} conformalises quantile regression with score $\max\{\hat q_{\alpha/2}(x)-y,\,y-\hat q_{1-\alpha/2}(x)\}$ \citep{romano2019cqr}. We use MAPIE \citep{taquet2022mapie} and PUNCC \citep{mendil2023puncc}.

\emph{NexCP} \citep{barber2023beyond} replaces the empirical quantile by a weighted one, $\hat q=Q_{1-\alpha}\big(\sum_i\tilde w_i\delta_{s_i}+\tilde w_{n+1}\delta_{+\infty}\big)$ with weights decaying in time, and keeps a coverage guarantee degraded by the total-variation drift; it is applied in split, full, CV+ and J+aB form. The \emph{online} methods process the test points sequentially and update the miscoverage level after each observation: ACI \citep{gibbs2021aci} uses
\begin{equation}
\alpha_{t+1}=\alpha_t+\gamma\,(\alpha-\mathrm{err}_t),\qquad \mathrm{err}_t=\mathbb 1\{y_t\notin\hat C_t(x_t)\},
\end{equation}
FACI \citep{gibbs2024faci} aggregates a grid of step sizes $\gamma$, SF-OGD and SAOCP \citep{bhatnagar2023saocp,orabona2018scalefree} run scale-free and strongly adaptive online gradient descent on the quantile directly, MFCS is a split/full online variant, and COP \citep{hu2026cop} uses the predicted distribution to inform the online update. Extreme CP \citep{pasche2025extreme} fits a generalised Pareto tail \citep{pickands1975gpd} to the scores and returns a one-sided upper bound. AlphaNet results are kept in the notebooks only.

\subsection{Adaptive coverage policies and HCM}
ACP \citep{gauthier2026acp} lets a policy choose a per-point level $\tilde\alpha(x)$ and builds the set from conformal e-values \citep{vovk2021evalues,gauthier2025evalues}, so the guarantee holds for the data-dependent level; the realised coverage must be compared with the learned target, not with 90\%. HCM \citep{choi2026hcm} trains a single network whose output direction on a hypersphere encodes the prediction and whose norm encodes confidence; we report its 95\% interval.

\subsection{Evaluation}
With $n=""" + str(n) + r"""$ test points the empirical coverage of a correct 90\% method has standard error $\sqrt{0.09/n}=""" + f"{math.sqrt(0.09/n):.3f}" + r"""$, so we call an interval \emph{valid} if its coverage is at least $0.9-1.645\sqrt{0.09/n}=""" + f"{A.TARGET-F_['tol']:.3f}" + r"""$. Among valid methods, narrower is better. Widths are also given relative to the test-target standard deviation so that datasets can be compared. All runs use $\alpha=0.1$ unless stated.

\section{Experimental setup}
All notebooks ran on Kaggle CPU machines with the generalised v2.0 code (every change is listed in \texttt{docs/CHANGELOG\_FIXES.md}). Downstream notebooks load the parameters of the tuning run. Optuna search is not bit-reproducible between runs; the stored tuning run is the one used by every other phase. """ + " ".join(exc) + r""" The executed notebooks are linked in Table~\ref{tab:runs}.

\section{Results}
""" + "\n".join(T[:2]) + r"""
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/fig_accuracy.pdf}\caption{Cross-validated and test RMSE of the tuned models. Dark bar: best CV model (""" + tex(best) + r""").}\label{fig:acc}\end{figure}
\begin{figure}[t]\centering\includegraphics[width=0.9\linewidth]{figures/fig_crps.pdf}\caption{CRPS of every predictive distribution (log scale).}\label{fig:crps}\end{figure}
\paragraph{Point accuracy (Table~\ref{tab:tuning}, Fig.~\ref{fig:acc}).} """ + clean(fl[0]) + r"""
\paragraph{Quantile and distributional methods (Table~\ref{tab:dist}, Fig.~\ref{fig:crps}).} """ + " ".join(clean(x) for x in fl[1:] if x.startswith(("Raw quantile", "Best probabilistic", "NGBoost (native)", "PGBM (native)", "CARD"))) + r"""

""" + T[2] + r"""
\begin{figure}[t]\centering\includegraphics[width=0.95\linewidth]{figures/fig_method_league.pdf}\caption{Coverage (left; band: valid region) and relative width (right) of every interval method, aggregated over base models.}\label{fig:league}\end{figure}
\begin{figure}[t]\centering\includegraphics[width=0.9\linewidth]{figures/fig_coverage_width.pdf}\caption{Coverage against relative width for every base model $\times$ method pair. Good methods sit in the shaded band at the far left.}\label{fig:cw}\end{figure}
\begin{figure}[t]\centering\includegraphics[width=0.9\linewidth]{figures/fig_intervals.pdf}\caption{MAPIE CV+ 90\% intervals of """ + tex(F_.get("int_model", best)) + r""" on the test set, sorted by observed value; red crosses are misses.}\label{fig:int}\end{figure}
\paragraph{Interval methods (Table~\ref{tab:league}, Figs.~\ref{fig:league}--\ref{fig:int}).} """ + " ".join(clean(x) for x in fl if x.startswith(("Recommended", "Narrowest", "Among valid"))) + r"""

""" + "\n".join(T[3:]) + r"""
\paragraph{Extreme CP, ACP and HCM.} """ + " ".join(clean(x) for x in fl if x.startswith(("ACP", "HCM"))) + (r" Extreme CP returns only an upper bound (Table~\ref{tab:extreme}), so it is not ranked with two-sided intervals." if ex else "") + r"""

\FloatBarrier
\section{Conclusions}
\begin{itemize}\setlength\itemsep{2pt}
""" + findings_tex + r"""
\end{itemize}
\paragraph{Limitations.} One fixed train/test split is used, so all coverages carry the sampling error stated above; widths are compared on the same split. The online methods treat the test set as a sequence in its stored order. Optuna tuning is not bit-reproducible. CARD, HCM and TabNet are neural and sensitive to their seeds.

\paragraph{Reproducibility.} Code: \url{https://github.com/DaneshSelwal/Regression_Uncertainty_Quantification_Analysis} (branch \texttt{version2.0}), folder \texttt{examples/""" + tex(ds) + r"""\_uncertainty\_analysis}. Every executed notebook is stored next to its Excel outputs and is listed below.
\begin{table}[h]\centering\scriptsize\caption{Kaggle runs whose outputs are reported here.}\label{tab:runs}
\begin{tabular}{lll}\toprule Notebook & Run & Kaggle URL\\\midrule
""" + rt + r"""
\bottomrule\end{tabular}\end{table}

\paragraph{Acknowledgements.} This work was carried out under the guidance of Dr.\ Mahesh Pal.

\FloatBarrier
\bibliographystyle{plainnat}\bibliography{references}
\end{document}
"""
    return doc

def build_pdf(R, ds, doc):
    d = f"{R}/papers/{ds}"; os.makedirs(d, exist_ok=True)
    open(f"{d}/paper.tex", "w").write(doc); shutil.copy(f"{HERE}/references.bib", f"{d}/references.bib")
    r = subprocess.run([shutil.which("tectonic") or "/opt/anaconda3/bin/tectonic", "-X", "compile", "--keep-logs", "paper.tex"], cwd=d, capture_output=True, text=True)
    for x in glob.glob(f"{d}/*.log") + glob.glob(f"{d}/*.aux") + glob.glob(f"{d}/*.bbl") + glob.glob(f"{d}/*.blg") + glob.glob(f"{d}/*.out"): os.remove(x)
    return r.returncode, (r.stderr or "")[-1500:]

# ---------------------------------------------------------------- example README
def readme(R, ds, F_):
    S = F_["S"]; dd = S["data"]; nm = NAMES[ds]; n = F_["n"]; L = F_["L"]; fl = findings(F_)
    fig = f"../../papers/{ds}/figures"
    lines = [f"# {nm} dataset: uncertainty quantification results", "",
             f"Results of the full v2.0 pipeline (8 notebooks) on the {nm.lower()} dataset. The full write-up with the methods, "
             f"maths and all tables is the paper [`papers/{ds}/paper.pdf`](../../papers/{ds}/paper.pdf).", "",
             "| | |", "|---|---|", f"| Training set | `data/` – {dd['n_train']} rows |", f"| Test set | {n} rows |",
             f"| Features ({len(dd['features'])}) | " + (", ".join(f"`{x}`" for x in dd["features"]) if len(dd["features"]) <= 20 else f"`{dd['features'][0]}` … `{dd['features'][-1]}`") + " |",
             f"| Target | `{dd['target']}` (test mean {f(dd['target_stats']['mean'])}, std {f(dd['target_stats']['std'])}) |",
             f"| Nominal level | 90% (α = 0.1); HCM reports 95% |", "", "## Key findings", ""]
    lines += [f"- {x}" for x in fl]
    lines += ["", "## Figures", "", f"![Coverage vs width]({fig}/fig_coverage_width.png)", "",
              f"![Interval methods]({fig}/fig_method_league.png)", "", f"![Intervals]({fig}/fig_intervals.png)", "",
              f"<details><summary>Point accuracy and CRPS</summary>", "", f"![Accuracy]({fig}/fig_accuracy.png)", "", f"![CRPS]({fig}/fig_crps.png)", "", "</details>", "",
              "## Interval methods at the 90% target", "",
              f"Aggregated over the base models each method wraps. *Valid* = share of models with coverage ≥ {A.TARGET-F_['tol']:.3f} "
              f"(90% minus the one-sided 95% sampling tolerance for n = {n}). Relative width = width / std of the test target. † = some intervals hit the edge of the label grid, so the width is a lower bound (not ranked).", "",
              "| Family | Method | Models | Coverage (median) | Coverage range | Width (median) | Relative width | Valid |", "|---|---|---:|---:|---:|---:|---:|---:|"]
    for r in L.itertuples():
        lines.append(f"| {r.family} | {r.label}{' †' if r.truncated else ''} | {r.n} | {pmd(r.cov_med)} | {pmd(r.cov_min)}–{pmd(r.cov_max)} | {f(r.w_med)} | {r.relw_med:.2f} | {round(100*r.share_valid)}% |")
    t = S["tuning"]
    lines += ["", "## Tuned models", "", "| Model | CV RMSE | Test RMSE | Test r |", "|---|---:|---:|---:|"]
    for m in sorted(t, key=lambda m: t[m]["cv_rmse"]): lines.append(f"| {m} | {f(t[m]['cv_rmse'])} | {f(t[m]['test_rmse'])} | {f(t[m].get('test_r'))} |")
    lines += ["", "## Folder contents", "", "| Folder | Notebook | Kaggle run(s) |", "|---|---|---|"]
    rl = runs(R, ds)
    for ph, title in PHASES:
        rs = [x for x in rl if x[0] == ph]
        links = ", ".join(f"[{c}]({u})" for _, _, c, _, u in rs) or "–"
        lines.append(f"| [`{ph}/`]({ph}/) | {title} | {links} |")
    lines += ["", "Each phase folder contains:", "",
              "- the notebook: the current v2.0 code for this dataset (same code as `notebooks/`, with this dataset's paths, file names and feature names);",
              "- `results/`: what the Kaggle run produced – `EXECUTED_*.ipynb` (the exact notebook that ran, with all cell outputs), the Excel files with every "
              "prediction, interval and metric, and a `README.md` naming the run.", "",
              "`data/` holds the train and test CSV files. To reproduce, run the eight notebooks in order (see the main README); notebooks 2–7 read "
              "`hyperparameter_tuning/best_params.pkl` written by notebook 1."]
    if F_["notrun"] or F_["nexfull_off"]:
        lines += ["", "## Not run (not enough compute)", ""]
        if F_["notrun"]: lines.append(f"- NB6 for {', '.join(F_['notrun'])}: the jobs exceed Kaggle's 12-hour limit.")
        if F_["nexfull_off"]: lines.append("- NexCP Full conformal (switch `RUN_NEXCP_FULL = False`, FIX B16): same reason.")
    open(f"{R}/examples/{ds}_uncertainty_analysis/README.md", "w").write("\n".join(lines) + "\n")

if __name__ == "__main__":
    R = sys.argv[1]
    for ds in sys.argv[2:]:
        F_ = facts(R, ds); make_figs(R, ds, F_); rc, err = build_pdf(R, ds, latex(R, ds, F_)); readme(R, ds, F_)
        print(ds, "pdf rc", rc, "" if rc == 0 else err)
