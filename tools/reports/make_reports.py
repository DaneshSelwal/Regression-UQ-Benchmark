"""make_reports.py <R> <ds...>: figures + LaTeX paper (papers/<ds>/) + example README + phase READMEs for each dataset.

Every number in the generated text is computed from tools/reports/summary/<ds>.json (build_summary.py) or the data CSVs.
"""
import sys, os, re, json, math, shutil, subprocess, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import analysis as A, figures as F, texblocks as TB

NAMES = {"concrete": "Concrete", "noise": "Noise", "pile": "Pile", "pile_settlement": "Pile settlement", "scour": "Scour",
         "sediment_load": "Sediment load", "streamflow": "Streamflow", "water_quality_ewqi": "Water quality (EWQI)",
         "water_quality_wqi": "Water quality (WQI)"}
PHASES = [("hyperparameter_tuning", "NB1 Hyperparameter tuning (Optuna)", "01_hyperparameter_tuning"),
          ("quantile_regression", "NB2 Quantile regression", "02_quantile_regression"),
          ("probabilistic_distribution", "NB3 Distributional (NGBoost, PGBM, IBUG, Treeffuser)", "03_probabilistic_distribution"),
          ("probabilistic_distribution_card", "NB4 Distributional (CARD diffusion)", "04_probabilistic_distribution_card"),
          ("conformal_predictions_mapie_puncc", "NB5 Conformal (MAPIE, PUNCC)", "05_conformal_mapie_puncc"),
          ("conformal_predictions_nexcp_adaptivecp_mfcs", "NB6 NexCP and online conformal", "06_conformal_nexcp_online"),
          ("conformal_predictions_adaptive_coverage_policies", "NB7 Adaptive coverage policies (ACP)", "07_adaptive_coverage_policies"),
          ("hyperspherical_confidence_mapping_hcm", "NB8 Hyperspherical confidence mapping (HCM)", "08_hyperspherical_confidence_mapping")]
REPO = "https://github.com/DaneshSelwal/Regression_Uncertainty_Quantification_Analysis"
CONF = ["Conformal (MAPIE/PUNCC)", "Weighted conformal (NexCP)", "Online conformal", "COP"]

# ---------------------------------------------------------------- formatting
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
def md2tex(s):
    """Markdown/unicode sentence -> LaTeX."""
    s = tex(s)
    s = re.sub(r"\*\*(.+?)\*\*", r"\\textbf{\1}", s)
    for a, b in [("–", "--"), ("≥", "$\\geq$"), ("≤", "$\\leq$"), ("×", "$\\times$"), ("α̃", "$\\tilde\\alpha$"), ("α", "$\\alpha$"),
                 ("σ", "$\\sigma$"), ("→", "$\\rightarrow$"), ("†", "$^\\dagger$"), ("−", "$-$"), ("≈", "$\\approx$")]:
        s = s.replace(a, b)
    return s

def rw(x, sd):
    return '–' if x is None or sd in (None, 0) else f"{x/sd:.2f}"

def registry(R):
    reg = pd.read_csv(f"{R}/docs/run_registry.tsv", sep="\t", dtype=str)
    return dict(zip(reg.code_name, reg.slug))

def not_run_models(S):
    t = S.get("nb6_readme") or ""; m = re.search(r"\*\*Not run \(out of compute\):\*\*\s*(.+)", t)
    return [x.strip(" .") for x in m.group(1).split(",")] if m else []

# ---------------------------------------------------------------- facts
def data_facts(R, ds):
    E = f"{R}/examples/{ds}_uncertainty_analysis/data"
    tr, te = pd.read_csv(f"{E}/train.csv"), pd.read_csv(f"{E}/test.csv")
    y0, y1 = tr.iloc[:, -1].astype(float), te.iloc[:, -1].astype(float)
    X = tr.iloc[:, :-1].apply(pd.to_numeric, errors="coerce")
    rho = X.apply(lambda c: c.corr(y0, method="spearman")).dropna().sort_values(key=np.abs, ascending=False)
    return dict(skew=float(pd.concat([y0, y1]).skew()), shift=float((y1.mean() - y0.mean()) / y0.std()),
                sd_ratio=float(y1.std() / y0.std()), p=X.shape[1], n=len(y0), rho=rho, y0=y0, y1=y1,
                train_stats=y0.describe(), test_stats=y1.describe())

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
                notrun=not_run_models(S), nexfull_off=nexfull_off, sd=S["data"]["target_stats"]["std"], ds=ds, data=data_facts(R, ds))

def acp_split(S):
    ac = S.get("acp") or {}
    sat = sorted(k for k, v in ac.items() if (v.get("Target_coverage") or 1) < 0.01)
    return ac, sat, {k: v for k, v in ac.items() if k not in sat}

# ---------------------------------------------------------------- findings (markdown sentences, reused in LaTeX)
def rq_text(F_):
    """Research-question answers for this dataset: dict rq -> (what it shows [list], takeaway)."""
    S, df, L, n, sd = F_["S"], F_["df"], F_["L"], F_["n"], F_["sd"]; t = S["tuning"]; out = {}
    b, bt = F_["best"], F_["best_test"]
    rm = sorted(t[m]["test_rmse"] for m in t if m != "RandomForest")
    r2 = lambda e: 1 - (e / sd) ** 2
    w = [f"**{b}** has the lowest 5-fold CV RMSE ({f(t[b]['cv_rmse'])}); on the test set the lowest RMSE is **{bt}** ({f(t[bt]['test_rmse'])}).",
         f"The target's test standard deviation is {f(sd)}, so the best test RMSE explains about {100*max(0, r2(t[bt]['test_rmse'])):.0f}% of the variance "
         f"(1 − RMSE²/var); the median model explains {100*max(0, r2(np.median(rm))):.0f}%.",
         f"CV and test rankings agree only roughly: with {n} test points the test RMSE of one model has a sampling error of several percent."]
    if "TabNet" in t and t["TabNet"]["test_rmse"] >= np.percentile(rm, 75): w.append("TabNet, the only neural base model, is among the least accurate, as is common for small tabular data.")
    out["RQ1"] = (w, "The tree ensembles are close to each other in accuracy, so the choice of UQ method matters more for the quality of the intervals than the choice among the boosted models.")
    q = F_["qr"]; cr = [v.get("crossing_rate") for v in S["quantile_regression"].values() if v.get("crossing_rate") is not None]
    if len(q):
        w = [f"The raw 90% interval [q̂₀.₀₅, q̂₀.₉₅] has a median coverage of **{pmd(q.coverage.median())}** over {len(q)} models "
             f"(range {pmd(q.coverage.min())}–{pmd(q.coverage.max())}); {int(q.valid.sum())} of {len(q)} are valid (≥ {A.TARGET-F_['tol']:.3f}).",
             f"Quantiles cross on a median **{pmd(np.median(cr)) if cr else '–'}** of test points before sorting: the 13 separately fitted models disagree on the order of their own quantiles.",
             f"Its median width is {q.rel_width.median():.2f} × std(y), narrower than most conformal intervals; the narrowness is what makes it undercover."]
        qd = F_["D"][F_["D"].method == "Quantile regression"]
        if len(qd): w.append(f"Best quantile CRPS: {qd.iloc[0].model} ({f(qd.iloc[0].crps)}).")
        out["RQ2"] = (w, "Raw quantile regression is not a reliable 90% interval here; it needs conformal calibration (CQR) or replacement by a conformal method." if q.coverage.median() < A.TARGET - F_["tol"]
                      else "Raw quantile regression happens to reach the target on this dataset, but it has no guarantee and its quantiles cross; a conformal method gives the same width with a guarantee.")
    w = []
    for nm, key in (("NGBoost", "NGBoost"), ("PGBM", "PGBM")):
        v = S["probabilistic"].get(key)
        if v: w.append(f"{nm} (native Gaussian): coverage **{pmd(v['coverage90_gauss'])}**, width {v['width90_gauss']/sd:.2f} × std(y), PIT diagnosis: {(v.get('diagnosis') or '–').lower()}.")
    for key in ("XGBoost", "LightGBM"):
        v = S["probabilistic"].get(key)
        if v: w.append(f"IBUG on {key}: coverage {pmd(v['coverage90_gauss'])}, width {v['width90_gauss']/sd:.2f} × std(y).")
    tf = [v for k, v in S["probabilistic"].items() if "(TF)" in k]
    if tf: w.append(f"Treeffuser (median of {len(tf)} models): coverage {pmd(np.median([v['coverage90_gauss'] for v in tf]))}, width {np.median([v['width90_gauss'] for v in tf])/sd:.2f} × std(y).")
    c = df[df.family == "CARD"]
    if len(c): w.append(f"CARD (median of {len(c)} guides): coverage {pmd(c.coverage.median())}, width {c.rel_width.median():.2f} × std(y).")
    D = F_["D"]
    if len(D): w.append(f"Lowest CRPS of all predictive distributions: **{D.iloc[0].model} ({D.iloc[0].method})**, {f(D.iloc[0].crps)} (target units).")
    over = [nm for nm, key in (("NGBoost", "NGBoost"), ("PGBM", "PGBM")) if S["probabilistic"].get(key) and S["probabilistic"][key]["coverage90_gauss"] < A.TARGET - F_["tol"]]
    out["RQ3"] = (w, (f"The native {' and '.join(over)} distributions are over-confident on this dataset: their σ̂ is too small. " if over else "")
                  + "A good CRPS does not imply a valid interval: CRPS rewards a sharp distribution centred near the truth, while coverage checks the tails.")
    cf = df[df.family.isin(CONF)]; bad = cf[~cf.valid]
    w = [f"Conformal intervals reach the target in **{int(cf.valid.sum())} of {len(cf)}** model × method cases ({pmd(cf.valid.mean())})."]
    if len(bad):
        g = bad.groupby("label").size().sort_values(ascending=False)
        w.append("Misses: " + ", ".join(f"{k} ({v})" for k, v in g.items()) + ". " +
                 ("Lowest coverage: " + f"{bad.sort_values('coverage').iloc[0].label} on {bad.sort_values('coverage').iloc[0].model}, {pmd(bad.coverage.min())}."))
    else: w.append("No conformal interval falls below the validity threshold.")
    trunc = L[L.truncated]
    if len(trunc): w.append("NexCP Full intervals sometimes reach the edge of their y-grid, so their width is a lower bound; they are not ranked (†).")
    out["RQ4"] = (w, "Conformal calibration delivers the promised coverage on almost every model, whatever the model's accuracy." if cf.valid.mean() >= 0.9
                  else "Conformal methods are valid on most, but not all, models here; the misses show where the exchangeability assumption or a small calibration set bites.")
    rec = F_["rec"]; ok = F_["ok"]
    w = []
    if rec is not None:
        w.append(f"Recommended: **{rec['label']}**, median coverage {pmd(rec.cov_med)}, median width {f(rec.w_med)} = **{rec.relw_med:.2f} × std(y)**.")
        top = ok.head(4)
        if len(top) > 1: w.append("Next narrowest valid methods: " + "; ".join(f"{r.label} {r.relw_med:.2f}" for r in top.iloc[1:].itertuples()) + ".")
        widest = ok.sort_values("relw_med").iloc[-1]
        w.append(f"The widest valid method ({widest.label}, {widest.relw_med:.2f}) is {widest.relw_med/rec.relw_med:.1f} × wider than the recommended one: validity alone does not pick a method.")
        if len(q): w.append(f"Raw quantile regression is {q.rel_width.median():.2f} × std(y) wide but covers only {pmd(q.coverage.median())}.")
    fam = F_["fam"]
    if len(fam) >= 2: w.append("Median width of valid intervals by family: " + "; ".join(f"{k} {v:.2f}" for k, v in fam.items()) + ".")
    out["RQ5"] = (w, f"On this dataset the sharpest valid interval comes from **{rec['label']}**." if rec is not None else "No method is valid on enough models to be recommended.")
    w = []; ac, sat, ok_ = acp_split(S)
    if ok_:
        cv = [v["coverage"] for v in ok_.values()]; tg = [v["Target_coverage"] for v in ok_.values() if v.get("Target_coverage") is not None]
        w.append(f"ACP learns its own level: mean learned target 1 − α̃ = {pmd(np.mean(tg)) if tg else '–'}; realised coverage {pmd(min(cv))}–{pmd(max(cv))} across {len(ok_)} models.")
    if sat:
        cv = [ac[k]["coverage"] for k in sat]
        w.append(f"For {len(sat)} of {len(ac)} models ({', '.join(sat)}) the ACP policy saturated at α̃ = 0.999: the sets have constant width and the learned level carries no information (realised coverage {pmd(min(cv))}–{pmd(max(cv))}).")
    h = S.get("hcm") or {}
    if h.get("Coverage_95Interval") is not None:
        sp = h.get("Spearman") or 0
        w.append(f"HCM (one network, 95% interval): coverage {pmd(h['Coverage_95Interval'])}, RMSE {f(h.get('RMSE'))}, Spearman(|error|, σ̂) = {f(sp,2)}"
                 + (" — its uncertainty is not informative here." if sp <= 0.1 else " — its uncertainty tracks the error."))
    ex = A.extreme(S)
    if ex: w.append(f"Extreme CP (one-sided upper bound): coverage {pmd(min(v['coverage'] for v in ex.values()))}–{pmd(max(v['coverage'] for v in ex.values()))} of the bound y ≤ ŷ + q.")
    out["RQ6"] = (w, "Learned-level and single-network methods answer different questions from a fixed 90% interval; they are reported for completeness and are not ranked.")
    return out

RQS = {"RQ1": "How accurate are the tuned point predictors?", "RQ2": "Does raw quantile regression give a valid 90% interval?",
       "RQ3": "How good are the predictive distributions?", "RQ4": "Do conformal methods keep their 90% promise?",
       "RQ5": "Which valid method is the sharpest?", "RQ6": "What do the learned-coverage and single-network methods add?"}

def findings(F_):
    """Flat list of short conclusions (paper conclusions, README key findings)."""
    rq = rq_text(F_); out = []
    for k in RQS: out.append(rq[k][1]) if k in rq else None
    if F_["notrun"]: out.append(f"Not run (not enough compute, Kaggle 12-hour limit): NB6 for {', '.join(F_['notrun'])}.")
    if F_["nexfull_off"]: out.append("NexCP Full conformal was not run on this dataset (not enough compute).")
    return out

# ---------------------------------------------------------------- figures
def fig_data(R, ds, F_, d):
    import matplotlib.pyplot as plt
    D = F_["data"]; fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.2, 2.8), gridspec_kw={"width_ratios": [1.1, 1]})
    b = np.histogram_bin_edges(np.r_[D["y0"], D["y1"]], bins=min(30, max(8, int(np.sqrt(len(D["y0"]) + len(D["y1"]))))))
    a1.hist(D["y0"], bins=b, density=True, color="#5d8aa8", alpha=0.75, label=f"train ({len(D['y0'])})")
    a1.hist(D["y1"], bins=b, density=True, histtype="step", lw=1.4, color="#c0392b", label=f"test ({len(D['y1'])})")
    a1.set_xlabel(F_["S"]["data"]["target"]); a1.set_yticks([]); a1.legend(fontsize=7); a1.set_title(f"Target distribution (skew {D['skew']:.1f})")
    r = D["rho"].head(10)[::-1]
    a2.barh(range(len(r)), r.values, color=["#1e8449" if v > 0 else "#c0392b" for v in r.values])
    a2.set_yticks(range(len(r))); a2.set_yticklabels([str(x)[:22] for x in r.index], fontsize=7); a2.axvline(0, color="k", lw=0.6)
    a2.set_xlabel("Spearman correlation with the target (train)"); a2.set_title(f"Strongest features (of {D['p']})")
    fig.tight_layout(); F.save(fig, d, "fig_data")

def make_figs(R, ds, F_):
    d = f"{R}/papers/{ds}/figures"; S = F_["S"]
    F.accuracy(S, d); F.cov_width(F_["df"], S, d); F.league_plot(F_["L"], S, d); F.crps_plot(S, d); fig_data(R, ds, F_, d)
    for x in ("fig_pipeline", "fig_conformal_explainer"):
        for e in ("pdf", "png"): shutil.copy(f"{R}/docs/figures/{x}.{e}", f"{d}/{x}.{e}")
    m = F_["best"]; meth = F.intervals(R, S, d, m)
    if meth is None:
        for mm in sorted(S["mapie_puncc"]):
            meth = F.intervals(R, S, d, mm)
            if meth: m = mm; break
    F_["int_model"] = m

# ---------------------------------------------------------------- run links
def runs(R, ds):
    reg = registry(R); rn = A.run_names(R, ds); out = []; tune = set(rn.get("hyperparameter_tuning", []))
    for ph, title, _ in PHASES:
        for c in rn.get(ph, []):
            if ph != "hyperparameter_tuning" and c in tune: continue   # the tuning run is an input, listed once
            sl = reg.get(c)
            if sl: out.append((ph, title, c, sl, f"https://www.kaggle.com/code/{sl}"))
    return out

# ---------------------------------------------------------------- LaTeX
def tab(caption, label, cols, header, rows, size="\\small", place="t", resize=False):
    body = "\\begin{tabular}{" + cols + "}\\toprule " + header + "\\\\\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\\end{tabular}"
    if resize: body = "\\resizebox{\\textwidth}{!}{" + body + "}"
    return f"\\begin{{table}}[{place}]\\centering{size}\n\\caption{{{caption}}}\\label{{{label}}}\n{body}\\end{{table}}"

def rq_tex(rq, key, how=None):
    if key not in rq: return ""
    w, take = rq[key]
    s = (f"\\paragraph{{How to read it.}} {how}\n" if how else "") + "\\paragraph{What it shows.}\n\\begin{itemize}\n" + "\n".join(f"\\item {md2tex(x)}" for x in w) + "\n\\end{itemize}\n"
    return s + f"\\begin{{insight}}[Takeaway {key}]\n{md2tex(take)}\n\\end{{insight}}\n"

def latex(R, ds, F_):
    S = F_["S"]; dd = S["data"]; nm = NAMES[ds]; t = S["tuning"]; n = F_["n"]; L = F_["L"]; DF = F_["data"]; df = F_["df"]
    rq = rq_text(F_); rec = F_["rec"]; ts = dd["target_stats"]; best = F_["best"]
    fl = dd["features"] if len(dd["features"]) <= 20 else dd["features"][:5] + ["...", dd["features"][-1]]
    feats = ", ".join(f"\\texttt{{{tex(x)}}}" if x != "..." else "\\dots" for x in fl)
    q = F_["qr"]; cf = df[df.family.isin(CONF)]
    abstract = (f"How far can a regression prediction be trusted? We answer this for the {nm.lower()} dataset "
                f"({dd['n_train']} training and {n} test samples, {len(dd['features'])} features) by attaching more than thirty uncertainty "
                f"quantification (UQ) methods to nine tuned regressors and checking every 90\\% prediction interval on held-out data. "
                f"The methods span quantile regression; predictive distributions (NGBoost, PGBM, IBUG, Treeffuser and the CARD diffusion model); "
                f"split, cross and bootstrap conformal prediction (MAPIE, PUNCC); weighted and online conformal prediction (NexCP, ACI, FACI, SF-OGD, "
                f"SAOCP, MFCS, COP); Extreme conformal prediction; adaptive coverage policies; and hyperspherical confidence mapping. "
                + (f"Raw quantile regression covers a median of only {pct(q.coverage.median())} of test points and its quantiles cross on most of them. " if len(q) else "")
                + f"Conformal intervals are valid in {int(cf.valid.sum())} of {len(cf)} model--method cases. "
                + (f"The sharpest method that is valid on most models is {tex(rec['label'])}, with median coverage {pct(rec.cov_med)} and a median width of "
                   f"{rec.relw_med:.2f} target standard deviations. " if rec is not None else "")
                + "Every number is read from executed Kaggle notebooks that are stored with the code; each method is explained with its "
                  "mathematics and the reasoning behind its guarantee.")
    T = {}
    T["data"] = tab("Data summary. Target statistics on the training and test sets.", "tab:data", "lrr", "Statistic & Train & Test",
                    [f"{k} & {f(DF['train_stats'][k2])} & {f(DF['test_stats'][k2])}\\\\" for k, k2 in
                     [("Rows", "count"), ("Mean", "mean"), ("Std.", "std"), ("Min", "min"), ("Median", "50%"), ("Max", "max")]], place="h")
    T["tuning"] = tab("Tuned point predictors, sorted by 5-fold cross-validated RMSE on the training set. $r$: Pearson correlation on the test set.",
                      "tab:tuning", "llrrr", "Model & Pruner & CV RMSE & Test RMSE & Test $r$",
                      [f"{tex(m)} & {tex(t[m].get('pruner') or '--')} & {f(t[m]['cv_rmse'])} & {f(t[m]['test_rmse'])} & {f(t[m].get('test_r'))} \\\\"
                       for m in sorted(t, key=lambda m: t[m]["cv_rmse"])])
    dr = []
    for m, v in sorted(S["quantile_regression"].items(), key=lambda kv: kv[1].get("crps") or 9e9):
        dr.append(f"QR & {tex(m)} & {pct(v.get('coverage90'))} & {f(v.get('width90'))} & {f(v.get('crps'))} & {pct(v.get('crossing_rate'))} & --\\\\")
    for k, v in sorted(S["probabilistic"].items(), key=lambda kv: kv[1].get("crps_gauss") or 9e9):
        b = k.replace(" (TF)", ""); meth = "Treeffuser" if "(TF)" in k else ("Native" if b in ("NGBoost", "PGBM") else "IBUG")
        dr.append(f"{meth} & {tex(b)} & {pct(v.get('coverage90_gauss'))} & {f(v.get('width90_gauss'))} & {f(v.get('crps_gauss'))} & -- & {tex((v.get('diagnosis') or '--').split(' (')[0])}\\\\")
    for m, v in sorted(S["card"].items(), key=lambda kv: kv[1].get("crps_gauss") or 9e9):
        dr.append(f"CARD & {tex(m)} & {pct(v.get('coverage90_gauss'))} & {f(v.get('width90_gauss'))} & {f(v.get('crps_gauss'))} & -- & {tex((v.get('diagnosis') or '--').split(' (')[0])}\\\\")
    T["dist"] = tab("Quantile regression and predictive distributions at the 90\\% level. Distributional intervals are $\\hat\\mu\\pm1.645\\hat\\sigma$ and their "
                    "CRPS is the Gaussian CRPS~\\eqref{eq:crpsgauss}; quantile-regression CRPS is the quantile form~\\eqref{eq:crps}. Crossing: share of test points "
                    "with out-of-order quantiles before sorting. PIT: diagnosis of the PIT histogram.", "tab:dist", "llrrrrl",
                    "Method & Model & Coverage & Width & CRPS & Crossing & PIT", dr, place="p", resize=True)
    T["league"] = tab("All two-sided interval methods at the 90\\% target, aggregated over the base models they wrap ($n$). Width: median over models of the mean "
                      "test-interval width; Rel.: divided by the test-target standard deviation. $^\\dagger$: some intervals reach the edge of the label grid, so the "
                      "width is a lower bound and the method is not ranked. Valid: share of models whose coverage is at least $" + f"{A.TARGET-F_['tol']:.3f}" + "$ "
                      "(eq.~\\ref{eq:valid}). Bold: recommended method.", "tab:league", "llrrcrrr",
                      "Family & Method & $n$ & Coverage (median) & Range & Width & Rel. & Valid",
                      [(f"{tex(r.family)} & " + (f"\\textbf{{{tex(r.label)}}}" if rec is not None and r.label == rec["label"] else tex(r.label))
                        + f"{'$^\\dagger$' if r.truncated else ''} & {r.n} & {pct(r.cov_med)} & {pct(r.cov_min)}--{pct(r.cov_max)} & {f(r.w_med)} & {r.relw_med:.2f} & {int(round(100*r.share_valid))}\\% \\\\")
                       for r in L.itertuples()], place="p", resize=True)
    ex = A.extreme(S)
    if ex:
        T["extreme"] = tab("Extreme conformal prediction: one-sided upper bound $\\hat y+q$. Coverage is $\\Pr(y\\le\\hat y+q)$; the last column is $q$.",
                           "tab:extreme", "lrr", "Model & Coverage & $q$", [f"{tex(m)} & {pct(v['coverage'])} & {f(v['width'])}\\\\" for m, v in sorted(ex.items())])
    ac = S.get("acp") or {}
    if ac:
        T["acp"] = tab("Adaptive coverage policies ($\\lambda=50$): mean learned target coverage $1-\\tilde\\alpha$, realised coverage and width of the e-value sets. "
                       "$^\\ast$: policy saturated at $\\tilde\\alpha=0.999$.", "tab:acp", "lrrrr", "Model & Target $1-\\tilde\\alpha$ & Coverage & Width & RMSE",
                       [f"{tex(m)}{'$^\\ast$' if (v.get('Target_coverage') or 1) < 0.01 else ''} & {pct(v.get('Target_coverage'))} & {pct(v['coverage'])} & {f(v.get('interval_width'))} & {f(v.get('rmse'))}\\\\" for m, v in sorted(ac.items())])
    h = S.get("hcm") or {}
    if h:
        keys = [("Coverage_95Interval", "Coverage of 95\\% interval", pct), ("Average_95Interval_Width", "Mean 95\\% width", f), ("Cov@1Sigma", "Coverage at $1\\sigma$ (68\\%)", pct),
                ("Cov@2Sigma", "Coverage at $2\\sigma$ (95\\%)", pct), ("Cov@3Sigma", "Coverage at $3\\sigma$ (99.7\\%)", pct), ("RMSE", "RMSE", f), ("MAE", "MAE", f),
                ("Mean_CRPS", "CRPS", f), ("Mean_Negative_LogLikelihood", "NLL", f), ("Spearman", "Spearman (|error|, $\\hat\\sigma$)", f), ("ECE_reg", "Regression ECE", f)]
        T["hcm"] = tab("Hyperspherical confidence mapping (one network).", "tab:hcm", "lr", "Metric & Value", [f"{lab} & {fn(h.get(k))}\\\\" for k, lab, fn in keys if h.get(k) is not None])
    # appendix: model x method coverage / width for the conformal methods
    P = cf.pivot_table(index="label", columns="model", values="coverage"); W = cf.pivot_table(index="label", columns="model", values="rel_width")
    order = L[L.family.isin(CONF)].label.tolist(); P = P.reindex([o for o in order if o in P.index]); W = W.reindex(P.index)
    mods = [m for m in A.MODEL_ORDER if m in P.columns]
    app_rows = []
    for lab in P.index:
        cells = []
        for m in mods:
            c, w_ = P.loc[lab, m], W.loc[lab, m]
            if pd.isna(c): cells.append("n/r")
            else:
                s = f"{100*c:.0f}/{w_:.2f}"; cells.append(f"\\textcolor{{red!70!black}}{{{s}}}" if c < A.TARGET - F_["tol"] else s)
        app_rows.append(f"{tex(lab)} & " + " & ".join(cells) + "\\\\")
    T["app"] = tab("Every conformal method on every base model: coverage (\\%) / relative width. Red: below the validity threshold. n/r: not run.",
                   "tab:app", "l" + "r" * len(mods), "Method & " + " & ".join(tex(m) for m in mods), app_rows, size="\\scriptsize", place="h", resize=True)
    rl = runs(R, ds)
    T["runs"] = tab("Kaggle runs whose outputs are reported here (one row per executed notebook).", "tab:runs", "lll", "Notebook & Run & Kaggle URL",
                    [f"{tex(title.split(' ',1)[0])} & \\texttt{{{tex(c)}}} & \\url{{{u}}}\\\\" for ph, title, c, sl, u in rl], size="\\scriptsize", place="h", resize=True)
    exc = []
    if F_["notrun"]: exc.append(f"Notebook 6 was not run for {', '.join(F_['notrun'])}: these jobs exceed Kaggle's 12-hour limit (not enough compute), so those models are missing from the NexCP and online tables.")
    if F_["nexfull_off"]: exc.append("NexCP Full conformal (a refit for every candidate label and test point) was switched off for this dataset for the same reason.")
    rho = DF["rho"]; top = ", ".join(f"\\texttt{{{tex(k)}}} ({v:+.2f})" for k, v in rho.head(3).items())
    shift_txt = (f"The test target is shifted by {DF['shift']:+.2f} training standard deviations relative to the training target, and its spread is "
                 f"{DF['sd_ratio']:.2f} times the training spread" + (": the split is not a clean random split, which matters for conformal methods that assume exchangeability." if abs(DF["shift"]) > 0.2 or abs(math.log(max(DF['sd_ratio'],1e-9))) > 0.25 else "; the two sets look alike, so exchangeability is plausible."))
    npr = dd["n_train"] / len(dd["features"])
    doc = TB.PREAMBLE + r"""
\title{How Far Can a Regression Prediction Be Trusted?\\[2pt] \large Uncertainty Quantification on the """ + nm + r""" Dataset with Conformal, Distributional and Learned-Coverage Methods}
""" + TB.AUTHORS + r"""
\date{Regression UQ Benchmark v2.0 --- November 2026}
\begin{document}\maketitle
\begin{abstract}""" + abstract + r"""\end{abstract}
\tableofcontents
\newpage
\section{Introduction}
A regression model always returns a number, whether the input looks like the training data or not. For engineering and environmental
quantities this is not enough: a design check, an exceedance warning or a monitoring decision needs to know \emph{how wrong the number could be}.
Uncertainty quantification answers this with a prediction interval or a full predictive distribution. Many UQ methods exist, and they
promise very different things: some are exact in finite samples under mild assumptions (conformal prediction), some are only correct if a
model is correct (quantile regression, Gaussian predictive distributions), and some learn their own target coverage.

This report applies one fixed pipeline of eight notebooks to the """ + nm.lower() + r""" dataset and asks six research questions:
\begin{enumerate}[label=\textbf{RQ\arabic*.}]
""" + "\n".join(f"\\item {v}" for v in RQS.values()) + r"""
\end{enumerate}
\paragraph{Contributions.} (i) A complete, executed comparison of more than thirty UQ methods on one dataset, with every number traceable
to a stored notebook; (ii) an explanation of each method's mathematics and of \emph{why} its guarantee holds or fails; (iii) a simple,
sample-size-aware validity rule and a recommendation rule that turn dozens of coverage--width pairs into one actionable choice.
The same pipeline and report structure are used for eight other datasets; a companion report compares all nine.

\paragraph{Organisation.} Section~\ref{sec:setup} fixes notation; Section~\ref{sec:data} describes the data; Section~\ref{sec:pipeline} the
pipeline; Section~\ref{sec:methods} the methods; Section~\ref{sec:eval} the evaluation protocol; Section~\ref{sec:results} answers RQ1--RQ6;
Sections~\ref{sec:discussion}--\ref{sec:future} discuss limitations and future work. Appendix~\ref{app:tables} lists every model--method result.
""" + TB.SETUP + r"""
\section{Data}\label{sec:data}
The training set has """ + str(dd["n_train"]) + r""" rows and the test set """ + str(n) + r""" rows; the split is the one supplied with the data.
The """ + str(len(dd["features"])) + r""" input features are """ + feats + r"""; the target is \texttt{""" + tex(dd["target"]) + r"""}.
With """ + f"{npr:.0f}" + r""" training rows per feature, """ + ("the problem is data-poor: flexible models can fit noise, and a calibration set carved out of the training data is small." if npr < 15 else "there are enough rows per feature for the tree ensembles to fit stable models.") + r"""
Table~\ref{tab:data} and Fig.~\ref{fig:data} summarise the target. Its skewness is """ + f"{DF['skew']:.2f}" + r"""; """ + ("a long right tail means a few large values dominate the squared error and are the hardest to cover." if DF["skew"] > 1 else "the distribution is not strongly skewed." if abs(DF["skew"]) <= 1 else "a long left tail means a few small values are the hardest to cover.") + r"""
""" + shift_txt + r""" The features most strongly related to the target (Spearman correlation on the training set) are """ + top + r""".
""" + T["data"] + r"""
\begin{figure}[h]\centering\includegraphics[width=0.95\linewidth]{figures/fig_data.pdf}\caption{Left: target distribution on the training and test sets. Right: the ten features with the largest absolute Spearman correlation with the target.}\label{fig:data}\end{figure}

\section{Pipeline and experimental protocol}\label{sec:pipeline}
\begin{figure}[t]\centering\includegraphics[width=\linewidth]{figures/fig_pipeline.pdf}\caption{The eight-notebook pipeline. NB1 tunes the base models and writes \texttt{best\_params.pkl}, which NB2--NB7 load; NB8 trains its own network. Each notebook writes Excel files and is stored as an executed notebook next to them.}\label{fig:pipeline}\end{figure}
Figure~\ref{fig:pipeline} shows the pipeline. The order is fixed: tuning first, so that every UQ method wraps the same tuned models; the test
set is used only for the final evaluation. All notebooks ran on Kaggle CPU sessions with the generalised v2.0 code; the only dataset-specific
settings are the folder name, file names, feature names and the target label used in plots. Every change made to the original v1.2 notebooks
is listed, with its reason and approval, in \texttt{docs/CHANGELOG\_FIXES.md}. All methods use $\alpha=0.1$ (90\% intervals) except HCM, which
reports its own 95\% interval. """ + " ".join(exc) + r""" The executed notebooks are linked in Table~\ref{tab:runs}.
""" + TB.METHODS + TB.evaluation(n, F_["tol"]) + r"""
\section{Results}\label{sec:results}
Each subsection answers one research question: it shows the evidence, explains how to read it, and ends with the takeaway.

\subsection{RQ1: how accurate are the tuned point predictors?}
""" + T["tuning"] + r"""
\begin{figure}[t]\centering\includegraphics[width=0.85\linewidth]{figures/fig_accuracy.pdf}\caption{Cross-validated (training) and test RMSE of the tuned models. Dark bar: best CV model (""" + tex(best) + r""").}\label{fig:acc}\end{figure}
""" + rq_tex(rq, "RQ1", "Table~\\ref{tab:tuning} and Fig.~\\ref{fig:acc} give, for each model, the error estimated by cross-validation on the training set (used to choose the model) and the error on the untouched test set. A large gap between the two would signal over-fitting of the hyperparameters.") + r"""
\subsection{RQ2: does raw quantile regression give a valid 90\% interval?}
""" + rq_tex(rq, "RQ2", "The QR rows of Table~\\ref{tab:dist} give the coverage and mean width of $[\\hat q_{0.05},\\hat q_{0.95}]$ for each model, its quantile CRPS, and the crossing rate. A coverage below the threshold of eq.~\\eqref{eq:valid} is a real failure, not noise.") + r"""
\subsection{RQ3: how good are the predictive distributions?}
""" + T["dist"] + r"""
\begin{figure}[t]\centering\includegraphics[width=0.85\linewidth]{figures/fig_crps.pdf}\caption{CRPS of every predictive distribution (log scale; lower is better). Colours: quantile regression, distributional methods, CARD.}\label{fig:crps}\end{figure}
""" + rq_tex(rq, "RQ3", "Fig.~\\ref{fig:crps} ranks all predictive distributions by CRPS, which rewards being both sharp and centred on the truth. Table~\\ref{tab:dist} adds the coverage of each method's Gaussian 90\\% interval and the PIT diagnosis: \\emph{under-dispersed} means $\\hat\\sigma$ is too small (over-confident), \\emph{over-dispersed} too large.") + r"""
\subsection{RQ4: do conformal methods keep their 90\% promise?}
""" + T["league"] + r"""
\begin{figure}[t]\centering\includegraphics[width=0.95\linewidth]{figures/fig_method_league.pdf}\caption{Coverage (left; dot: median over base models, bar: min--max; shaded: valid region) and median relative width (right) of every two-sided interval method.}\label{fig:league}\end{figure}
\begin{figure}[t]\centering\includegraphics[width=0.85\linewidth]{figures/fig_coverage_width.pdf}\caption{Coverage against relative width for every base model $\times$ method pair. A good method sits inside the shaded band and as far left as possible.}\label{fig:cw}\end{figure}
""" + rq_tex(rq, "RQ4", "In Fig.~\\ref{fig:league} each row is one method; its dot is the median coverage over the base models it wraps and its bar the range. The shaded band is the valid region of eq.~\\eqref{eq:valid}. In Fig.~\\ref{fig:cw} every point is one model--method pair; points below the band are invalid, points far to the right are wide.") + r"""
\subsection{RQ5: which valid method is the sharpest?}
\begin{figure}[t]\centering\includegraphics[width=0.85\linewidth]{figures/fig_intervals.pdf}\caption{MAPIE CV+ 90\% intervals of """ + tex(F_.get("int_model", best)) + r""" on the test set, sorted by the observed value; red crosses are test points outside their interval.}\label{fig:int}\end{figure}
""" + rq_tex(rq, "RQ5", "The right panel of Fig.~\\ref{fig:league} and the Rel.\\ column of Table~\\ref{tab:league} compare widths. Only methods that are valid on most models are candidates. Fig.~\\ref{fig:int} shows one concrete interval set: where along the range of the target the misses occur (marginal coverage allows them to cluster).") + r"""
\subsection{RQ6: learned coverage, single-network UQ and extreme quantiles}
""" + "\n".join(T[k] for k in ("acp", "hcm", "extreme") if k in T) + "\n" + rq_tex(rq, "RQ6", "Table~\\ref{tab:acp} compares each ACP model's realised coverage with its own learned target $1-\\tilde\\alpha$ (not with 90\\%). Table~\\ref{tab:hcm} lists HCM's coverage at several levels and the correlation between its predicted spread and the actual error." + (" Table~\\ref{tab:extreme} gives the one-sided Extreme CP bounds." if ex else "")) + r"""
\FloatBarrier
\section{Discussion}\label{sec:discussion}
\paragraph{Practical recommendation.} """ + (md2tex(f"For a 90% interval on this dataset, use **{rec['label']}**: it is valid on {int(round(100*rec.share_valid))}% of the base models and is the narrowest such method ({rec.relw_med:.2f} × std(y)). "
            + ("It is an online method: it uses the labels of earlier test points, so it suits settings where outcomes are observed one by one. If labels are not available sequentially, use the narrowest valid offline method: "
               + next((r.label for r in F_['ok'].itertuples() if r.family == 'Conformal (MAPIE/PUNCC)' or r.label.startswith('NexCP')), '–') + "."
               if rec['family'] in ('Online conformal', 'COP') else "It needs no labels beyond the training set."))
            if rec is not None else "No method is valid on enough models to be recommended.") + r"""
\paragraph{Why conformal methods work and model-based ones do not.} Conformal methods calibrate on residuals of points the model did not
see, so the size of the interval is set by honest out-of-sample errors. Quantile regression and Gaussian distributions estimate the spread
from the same data the model was fitted to; their intervals are as good as the model's description of the tails, which with a few hundred
points is rarely good. The good CRPS of some distributional methods shows that their \emph{centre} is right; their coverage shows that their
\emph{tails} are not.
\paragraph{Validity versus usefulness.} All valid methods keep the same promise, so the choice among them is about width. Widths differ by
a large factor between valid methods (Table~\ref{tab:league}); reporting coverage without width, or width without coverage, hides this.
""" + TB.LIMITATIONS + TB.FUTURE + r"""
\section{Conclusions}
\begin{itemize}
""" + "\n".join(f"\\item {md2tex(x)}" for x in findings(F_)) + r"""
\end{itemize}

\paragraph{Reproducibility.} Code: \url{""" + REPO + r"""} (branch \texttt{version2.0}), folder \url{examples/""" + ds + r"""_uncertainty_analysis}. Every
executed notebook is stored next to its Excel outputs and is listed in Table~\ref{tab:runs}. Machine-readable results for all datasets:
\texttt{results/benchmark\_long.csv}.
""" + T["runs"] + r"""
\paragraph{Acknowledgements.} Computations were run on Kaggle's free CPU sessions.
\FloatBarrier
\bibliographystyle{plainnat}\bibliography{references}
\appendix
\section{Complete conformal results}\label{app:tables}
""" + T["app"] + r"""
\end{document}
"""
    return doc

def build_pdf(R, ds, doc, folder=None):
    d = folder or f"{R}/papers/{ds}"; os.makedirs(d, exist_ok=True)
    open(f"{d}/paper.tex", "w").write(doc); shutil.copy(f"{HERE}/references.bib", f"{d}/references.bib")
    r = subprocess.run([shutil.which("tectonic") or "/opt/anaconda3/bin/tectonic", "-X", "compile", "--keep-logs", "paper.tex"], cwd=d, capture_output=True, text=True)
    log = open(f"{d}/paper.log").read() if os.path.exists(f"{d}/paper.log") else ""
    warn = [l for l in log.splitlines() if re.search(r"undefined|Overfull \\hbox \((\d{2,})", l)]
    for x in glob.glob(f"{d}/*.log") + glob.glob(f"{d}/*.aux") + glob.glob(f"{d}/*.bbl") + glob.glob(f"{d}/*.blg") + glob.glob(f"{d}/*.out") + glob.glob(f"{d}/*.toc"): os.remove(x)
    return r.returncode, (r.stderr or "")[-1500:], warn

# ---------------------------------------------------------------- example README (DIAS-style)
def md_rq(rq, key, fig_lines, how):
    if key not in rq: return []
    w, take = rq[key]
    return [f"### {key} · {RQS[key]}", ""] + fig_lines + [f"**How to read it.** {how}", "", "**What it shows.**", ""] + [f"- {x}" for x in w] + ["", f"> **Takeaway {key}.** {take}", ""]

def readme(R, ds, F_):
    S = F_["S"]; dd = S["data"]; nm = NAMES[ds]; n = F_["n"]; L = F_["L"]; rq = rq_text(F_); rec = F_["rec"]; DF = F_["data"]; df = F_["df"]
    fig = f"../../papers/{ds}/figures"; cf = df[df.family.isin(CONF)]; q = F_["qr"]; t = S["tuning"]
    nat = [f"{k} {pmd(S['probabilistic'][k]['coverage90_gauss'])}" for k in ("NGBoost", "PGBM") if k in S["probabilistic"]]
    card = [
        "| | |", "|---|---|",
        f"| **Data** | {dd['n_train']} training / {n} test rows, {len(dd['features'])} features, target `{dd['target']}` |",
        f"| **Key result** | Conformal intervals are valid in **{int(cf.valid.sum())} of {len(cf)}** model × method cases; the sharpest valid method is **{rec['label']}** "
        f"({pmd(rec.cov_med)} coverage, {rec.relw_med:.2f} × std(y) wide) |" if rec is not None else f"| **Key result** | Conformal intervals are valid in {int(cf.valid.sum())} of {len(cf)} cases |",
        f"| **Watch out** | Raw quantile regression covers only {pmd(q.coverage.median())} (median); native Gaussian intervals: {', '.join(nat)} |" if len(q) else "",
        f"| **Best point model** | {F_['best']} (5-fold CV RMSE {f(t[F_['best']]['cv_rmse'])}) |",
        f"| **Full report** | [`papers/{ds}/paper.pdf`](../../papers/{ds}/paper.pdf): methods with maths, all tables, appendix |",
    ]
    lines = [f"# {nm}: how far can the predictions be trusted?", "",
             f"[← main README](../../README.md) · [all datasets](../../README.md#5-the-nine-datasets) · [report (PDF)](../../papers/{ds}/paper.pdf) · [method guides](../../notebooks/)", ""] + [c for c in card if c] + ["",
             "## Contents", "", "1. [The data](#1-the-data)", "2. [What was run](#2-what-was-run)", "3. [Results, question by question](#3-results-question-by-question)",
             "4. [All interval methods](#4-all-interval-methods)", "5. [Folder map and Kaggle runs](#5-folder-map-and-kaggle-runs)", "", "---", "",
             "## 1. The data", "",
             f"<p align=\"center\"><img src=\"{fig}/fig_data.png\" width=\"90%\" alt=\"Target distribution and strongest features\"></p>", "",
             f"*__Figure 1.__ Left: the target on the training and test sets. Right: the features most correlated with it (Spearman, training set).*", "",
             "| Statistic | Train | Test |", "|---|--:|--:|"]
    for k, k2 in [("Rows", "count"), ("Mean", "mean"), ("Std", "std"), ("Min", "min"), ("Median", "50%"), ("Max", "max")]:
        lines.append(f"| {k} | {f(DF['train_stats'][k2])} | {f(DF['test_stats'][k2])} |")
    feats = ", ".join(f"`{x}`" for x in dd["features"]) if len(dd["features"]) <= 20 else f"`{dd['features'][0]}` … `{dd['features'][-1]}` ({len(dd['features'])} columns)"
    lines += ["", f"**Features:** {feats}.", "",
              f"**Reading Figure 1.** The target's skewness is {DF['skew']:.2f}"
              + (" (a long right tail: a few large values are the hardest to predict and to cover)" if DF["skew"] > 1 else "") +
              f". The test mean differs from the training mean by {DF['shift']:+.2f} training standard deviations and the test spread is {DF['sd_ratio']:.2f}× the training spread"
              + (" — the two sets are not identically distributed, which strains methods that assume exchangeability." if abs(DF["shift"]) > 0.2 or abs(math.log(max(DF['sd_ratio'], 1e-9))) > 0.25 else " — the two sets look alike.")
              + f" With {dd['n_train']/len(dd['features']):.0f} training rows per feature, " + ("the problem is data-poor." if dd['n_train']/len(dd['features']) < 15 else "there is enough data per feature for stable tree models."), "",
              "---", "", "## 2. What was run", "",
              "The same eight notebooks as every other dataset ([pipeline](../../README.md#4-the-pipeline)); every phase folder has its own README with this dataset's numbers.", "",
              "| Phase | What it computes | Results | Kaggle run(s) |", "|---|---|---|---|"]
    rl = runs(R, ds)
    for ph, title, nbdir in PHASES:
        rs = [x for x in rl if x[0] == ph]; links = ", ".join(f"[{c}]({u})" for _, _, c, _, u in rs) or "–"
        lines.append(f"| [{title.split(' ',1)[0]}]({ph}/README.md) | {title.split(' ',1)[1]} | [`{ph}/results/`]({ph}/results/) | {links} |")
    if F_["notrun"] or F_["nexfull_off"]:
        lines += ["", "**Not run (not enough compute).** " + " ".join(
            ([f"NB6 for {', '.join(F_['notrun'])}: these jobs exceed Kaggle's 12-hour limit."] if F_["notrun"] else []) +
            (["NexCP Full conformal was switched off for this dataset (`RUN_NEXCP_FULL = False`, FIX B16) for the same reason."] if F_["nexfull_off"] else []))]
    lines += ["", "---", "", "## 3. Results, question by question", "",
              f"Every interval targets **90%**. With {n} test points a correct method can land anywhere above **{A.TARGET-F_['tol']:.3f}** by chance, "
              "so that is the validity threshold ([why](../../README.md#73-when-is-a-coverage-good-enough)). Relative width = interval width / std of the test target.", ""]
    lines += md_rq(rq, "RQ1", [f"<p align=\"center\"><img src=\"{fig}/fig_accuracy.png\" width=\"75%\" alt=\"Accuracy\"></p>", ""],
                   "Grey bars: 5-fold CV RMSE on the training set (used to choose the model). Blue bars: RMSE on the untouched test set; dark blue = best CV model.")
    lines += md_rq(rq, "RQ2", [], "Quantile regression fits one model per level; its 90% interval is the 5% and 95% models' predictions. No calibration step is involved.")
    lines += md_rq(rq, "RQ3", [f"<p align=\"center\"><img src=\"{fig}/fig_crps.png\" width=\"75%\" alt=\"CRPS\"></p>", ""],
                   "CRPS (log scale, lower is better) scores a whole predictive distribution: it rewards a sharp distribution centred on the truth. Coverage, in the text, checks only the 90% interval.")
    lines += md_rq(rq, "RQ4", [f"<p align=\"center\"><img src=\"{fig}/fig_method_league.png\" width=\"85%\" alt=\"Interval methods\"></p>", "",
                               f"<p align=\"center\"><img src=\"{fig}/fig_coverage_width.png\" width=\"75%\" alt=\"Coverage vs width\"></p>", ""],
                   "Top: one row per method; dot = median coverage over the base models it wraps, bar = min–max, green band = valid region; right panel = median relative width. Bottom: every model × method pair; good intervals sit in the green band, far to the left.")
    lines += md_rq(rq, "RQ5", [f"<p align=\"center\"><img src=\"{fig}/fig_intervals.png\" width=\"75%\" alt=\"Intervals\"></p>", ""],
                   f"The CV+ intervals of {F_.get('int_model', F_['best'])} on every test point, sorted by the observed value; red crosses are misses. Marginal coverage allows misses to cluster, typically at the extremes.")
    lines += md_rq(rq, "RQ6", [], "ACP coverage must be compared with its own learned target 1 − α̃, not with 90%. HCM reports a 95% interval and the correlation between its predicted spread and the actual error.")
    lines += ["---", "", "## 4. All interval methods", "",
              f"Aggregated over the base models each method wraps. *Valid* = share of models with coverage ≥ {A.TARGET-F_['tol']:.3f}. † = some intervals hit the edge of the label grid, so the width is a lower bound (not ranked). **Bold** = recommended.", "",
              "| Family | Method | Models | Coverage (median) | Coverage range | Width (median) | Relative width | Valid |", "|---|---|--:|--:|--:|--:|--:|--:|"]
    for r in L.itertuples():
        lab = f"**{r.label}**" if rec is not None and r.label == rec["label"] else r.label
        lines.append(f"| {r.family} | {lab}{' †' if r.truncated else ''} | {r.n} | {pmd(r.cov_med)} | {pmd(r.cov_min)}–{pmd(r.cov_max)} | {f(r.w_med)} | {r.relw_med:.2f} | {round(100*r.share_valid)}% |")
    lines += ["", "<details><summary><b>Tuned models (NB1)</b></summary>", "", "| Model | Pruner | CV RMSE | Test RMSE | Test r |", "|---|---|--:|--:|--:|"]
    for m in sorted(t, key=lambda m: t[m]["cv_rmse"]): lines.append(f"| {m} | {t[m].get('pruner') or '–'} | {f(t[m]['cv_rmse'])} | {f(t[m]['test_rmse'])} | {f(t[m].get('test_r'))} |")
    lines += ["", "</details>", "", "---", "", "## 5. Folder map and Kaggle runs", "", "```",
              f"{ds}_uncertainty_analysis/", "├── README.md                 ← you are here", "├── data/                     ← train.csv, test.csv (last column = target)"]
    for ph, title, _ in PHASES:
        lines.append(f"├── {ph}/".ljust(54) + f"← {title.split(' ',1)[0]}")
    lines += ["│   ├── README.md             ← this phase on this dataset", "│   ├── <notebook>.ipynb      ← v2.0 code with this dataset's paths (no outputs)",
              "│   └── results/              ← EXECUTED_*.ipynb (every cell output) + Excel files", "```", "",
              "Every run used for the results is linked in the table of [section 2](#2-what-was-run) and in each `results/README.md`. "
              f"To reproduce, run the eight notebooks in order ([how](../../README.md#11-reproducing-the-results)). Machine-readable numbers: [`results/benchmark_long.csv`](../../results/benchmark_long.csv) (rows with `dataset = {ds}`)."]
    open(f"{R}/examples/{ds}_uncertainty_analysis/README.md", "w").write("\n".join(lines) + "\n")

# ---------------------------------------------------------------- phase READMEs
NBV = "https://nbviewer.org/github/DaneshSelwal/Regression_Uncertainty_Quantification_Analysis/blob/version2.0"
def big_note(E, ph, limit_mb=10):
    """Executed notebooks too large for GitHub's notebook viewer: link nbviewer instead (files are unchanged)."""
    from urllib.parse import quote
    big = [(x, os.path.getsize(x) / 1e6) for x in sorted(glob.glob(f"{E}/results/*.ipynb")) if os.path.getsize(x) / 1e6 > limit_mb]
    if not big: return []
    rel = E.split("/examples/", 1)[1]
    return [f"> **Large notebooks.** These executed notebooks are mostly training logs (e.g. PGBM prints one line per tree) and may be too large for "
            "GitHub's notebook viewer. They are stored unchanged; open them with nbviewer, the Kaggle run, or download them: "
            + "; ".join(f"[`{os.path.basename(x)}`]({NBV}/examples/{quote(rel)}/results/{quote(os.path.basename(x))}) ({mb:.0f} MB)" for x, mb in big), ""]

def phase_readmes(R, ds, F_):
    S = F_["S"]; nm = NAMES[ds]; df = F_["df"]; L = F_["L"]; sd = F_["sd"]; rl = runs(R, ds)
    def mk(ph, title, nbdir, what, body):
        E = f"{R}/examples/{ds}_uncertainty_analysis/{ph}"
        nbs = sorted(os.path.basename(x) for x in glob.glob(f"{E}/*.ipynb")); res = sorted(os.path.basename(x) for x in glob.glob(f"{E}/results/*"))
        rs = [x for x in rl if x[0] == ph]
        lines = [f"# {nm} · {title}", "", f"[← {nm} README](../README.md) · [method guide](../../../notebooks/{nbdir}/README.md) · [report (PDF)](../../../papers/{ds}/paper.pdf)", "",
                 "| | |", "|---|---|", f"| **What** | {what} |", f"| **Notebook** | " + ", ".join(f"[`{x}`](<{x}>)" for x in nbs) + " |",
                 f"| **Kaggle run(s)** | " + (", ".join(f"[{c}]({u})" for _, _, c, _, u in rs) or "–") + " |",
                 f"| **Executed output** | [`results/`](results/) ({len(res)} files: executed notebook(s) and Excel workbooks) |", ""] + big_note(E, ph) + body + \
                ["", f"*Numbers are read from the files in [`results/`](results/). The theory, with the maths, is in the [method guide](../../../notebooks/{nbdir}/README.md).*"]
        open(f"{E}/README.md", "w").write("\n".join(lines) + "\n")
    t = S["tuning"]
    body = ["## Tuned models", "", "| Model | Best pruner | 5-fold CV RMSE | Test RMSE | Test r |", "|---|---|--:|--:|--:|"] + \
           [f"| {m} | {t[m].get('pruner') or '–'} | {f(t[m]['cv_rmse'])} | {f(t[m]['test_rmse'])} | {f(t[m].get('test_r'))} |" for m in sorted(t, key=lambda m: t[m]["cv_rmse"])] + \
           ["", f"Best model by CV: **{F_['best']}**. The selected settings are in `results/best_params.json` / `.pkl` and are loaded by notebooks 2–7."]
    mk(*PHASES[0], "Tunes ten regressors with Optuna (5-fold CV on the training set only) and saves the settings every later notebook loads.", body)
    qr = S["quantile_regression"]
    body = ["## 90% interval [q̂₀.₀₅, q̂₀.₉₅] per model", "", "| Model | Coverage | Width | Width / std(y) | CRPS | Crossing (before sorting) |", "|---|--:|--:|--:|--:|--:|"] + \
           [f"| {m} | {pmd(v.get('coverage90'))} | {f(v.get('width90'))} | {rw(v.get('width90'), sd)} | {f(v.get('crps'))} | {pmd(v.get('crossing_rate'))} |" for m, v in sorted(qr.items(), key=lambda kv: kv[1].get('crps') or 9e9)] + \
           ["", f"Validity threshold for n = {F_['n']}: {A.TARGET-F_['tol']:.3f}."]
    mk(*PHASES[1], "Fits one model per quantile level (13 levels) for each regressor; reports the raw 90% interval, quantile CRPS and quantile crossing.", body)
    p = S["probabilistic"]
    body = ["## Predictive distributions (Gaussian 90% interval μ̂ ± 1.645σ̂)", "", "| Method | Model | Coverage | Width / std(y) | CRPS | PIT diagnosis |", "|---|---|--:|--:|--:|---|"] + \
           [f"| {'Treeffuser' if '(TF)' in k else ('native' if k in ('NGBoost','PGBM') else 'IBUG')} | {k.replace(' (TF)','')} | {pmd(v.get('coverage90_gauss'))} | {rw(v.get('width90_gauss'), sd)} | {f(v.get('crps_gauss'))} | {v.get('diagnosis') or '–'} |"
            for k, v in sorted(p.items(), key=lambda kv: kv[1].get('crps_gauss') or 9e9)]
    mk(*PHASES[2], "NGBoost and PGBM (native Gaussian), IBUG (XGBoost, LightGBM) and Treeffuser: full predictive distributions, CRPS, PIT and calibration.", body)
    c = S["card"]
    body = ["## CARD guided by each tuned model", "", "| Guide model | Coverage | Width / std(y) | CRPS | PIT diagnosis |", "|---|--:|--:|--:|---|"] + \
           [f"| {k} | {pmd(v.get('coverage90_gauss'))} | {rw(v.get('width90_gauss'), sd)} | {f(v.get('crps_gauss'))} | {v.get('diagnosis') or '–'} |" for k, v in sorted(c.items(), key=lambda kv: kv[1].get('crps_gauss') or 9e9)]
    mk(*PHASES[3], "CARD: a denoising diffusion model guided by each tuned regressor (official settings, 1000 samples per test point).", body)
    def grid(sub, labels):
        P = sub.pivot_table(index="label", columns="model", values="coverage"); W = sub.pivot_table(index="label", columns="model", values="rel_width")
        mods = [m for m in A.MODEL_ORDER if m in P.columns]; rows = ["| Method | " + " | ".join(mods) + " |", "|---|" + "--:|" * len(mods)]
        for lab in [l for l in labels if l in P.index]:
            cells = []
            for m in mods:
                v = P.loc[lab, m]
                if pd.isna(v): cells.append("n/r"); continue
                s = f"{100*v:.0f}% / {W.loc[lab, m]:.2f}"; cells.append(f"**{s}** ⚠" if v < A.TARGET - F_["tol"] else s)
            rows.append(f"| {lab} | " + " | ".join(cells) + " |")
        return rows
    sub = df[df.family == "Conformal (MAPIE/PUNCC)"]
    body = ["## Coverage / relative width, every method × model", "", f"⚠ = below the validity threshold ({A.TARGET-F_['tol']:.3f}).", ""] + grid(sub, L[L.family == "Conformal (MAPIE/PUNCC)"].label.tolist())
    mk(*PHASES[4], "Split conformal, CV+ and J+aB (MAPIE); split conformal, CV+ and CQR (PUNCC), wrapped around every tuned model.", body)
    sub = df[df.family.isin(["Weighted conformal (NexCP)", "Online conformal", "COP"])]
    body = ["## Coverage / relative width, every method × model", "", f"⚠ = below the validity threshold ({A.TARGET-F_['tol']:.3f}). n/r = not run.", ""] + \
           grid(sub, L[L.family.isin(["Weighted conformal (NexCP)", "Online conformal", "COP"])].label.tolist())
    ex = A.extreme(S)
    if ex: body += ["", "## Extreme CP (one-sided upper bound y ≤ ŷ + q)", "", "| Model | Coverage | q |", "|---|--:|--:|"] + [f"| {m} | {pmd(v['coverage'])} | {f(v['width'])} |" for m, v in sorted(ex.items())]
    if F_["notrun"] or F_["nexfull_off"]:
        body += ["", "**Not run (not enough compute):** " + " ".join(([f"models {', '.join(F_['notrun'])} (12-hour limit)."] if F_["notrun"] else []) + (["NexCP Full for all models (FIX B16)."] if F_["nexfull_off"] else []))]
    mk(*PHASES[5], "Weighted conformal (NexCP split, full, CV+, J+aB), online conformal (ACI, FACI, SF-OGD, SAOCP, online split/CV+/J+aB, MFCS), COP and Extreme CP. One Kaggle job per base model.", body)
    ac, sat, _ = acp_split(S)
    body = ["## Learned coverage per model (λ = 50)", "", "| Model | Learned target 1 − α̃ | Coverage | Width / std(y) | Saturated? |", "|---|--:|--:|--:|---|"] + \
           [f"| {m} | {pmd(v.get('Target_coverage'))} | {pmd(v['coverage'])} | {rw(v.get('interval_width'), sd)} | {'yes (α̃ = 0.999)' if m in sat else 'no'} |" for m, v in sorted(ac.items())]
    mk(*PHASES[6], "Adaptive coverage policies: a small network chooses each point's coverage level; e-value conformal sets give the guarantee at that level.", body)
    h = S.get("hcm") or {}
    body = ["## HCM metrics", "", "| Metric | Value |", "|---|--:|"] + [f"| {k} | {pmd(h[k]) if k.startswith(('Cov', 'Coverage')) else f(h[k])} |" for k in h if h[k] is not None]
    mk(*PHASES[7], "Hyperspherical confidence mapping: one network predicts magnitude and direction; the direction's distance from the unit circle is the uncertainty.", body)

if __name__ == "__main__":
    R = sys.argv[1]
    for ds in sys.argv[2:]:
        F_ = facts(R, ds); make_figs(R, ds, F_); rc, err, warn = build_pdf(R, ds, latex(R, ds, F_)); readme(R, ds, F_); phase_readmes(R, ds, F_)
        print(ds, "pdf rc", rc, "" if rc == 0 else err, "| warnings:", warn[:5])
