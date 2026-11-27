"""study.py <R>: cross-dataset tables (results/*.csv) and figures (docs/figures/*) for the whole benchmark.

Reads tools/reports/summary/<ds>.json (written by build_summary.py) and the example data folders.
Writes:
  results/benchmark_long.csv      one row per dataset x model x interval method (coverage, width, validity)
  results/benchmark_methods.csv   one row per dataset x method (median over models, share valid)
  results/benchmark_datasets.csv  one row per dataset (sizes, best model, recommended method)
  docs/figures/*.png|pdf          figures used by the main README and the benchmark paper
"""
import sys, os, json, math
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Patch
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
import analysis as A
from figures import FAM_COL, save

DS = ["concrete", "noise", "pile", "pile_settlement", "scour", "sediment_load", "streamflow", "water_quality_ewqi", "water_quality_wqi"]
NAMES = {"concrete": "Concrete", "noise": "Noise", "pile": "Pile", "pile_settlement": "Pile settlement", "scour": "Scour",
         "sediment_load": "Sediment load", "streamflow": "Streamflow", "water_quality_ewqi": "Water quality (EWQI)",
         "water_quality_wqi": "Water quality (WQI)"}
SHORT = {"concrete": "Concrete", "noise": "Noise", "pile": "Pile", "pile_settlement": "Pile settl.", "scour": "Scour",
         "sediment_load": "Sediment", "streamflow": "Streamflow", "water_quality_ewqi": "EWQI", "water_quality_wqi": "WQI"}
CONF = ["Conformal (MAPIE/PUNCC)", "Weighted conformal (NexCP)", "Online conformal", "COP"]

def load(ds): return json.load(open(f"{HERE}/summary/{ds}.json"))

def long_table():
    out = []
    for ds in DS:
        S = load(ds); df = A.rows(S); df.insert(0, "dataset", ds); df["n_test"] = S["data"]["n_test"]
        df["valid_threshold"] = A.TARGET - A.tol(S["data"]["n_test"]); out.append(df)
    return pd.concat(out, ignore_index=True)

# ------------------------------------------------------------------ tables
def tables(R, D):
    os.makedirs(f"{R}/results", exist_ok=True)
    D.round(6).to_csv(f"{R}/results/benchmark_long.csv", index=False)
    rows, M = [], []
    for ds in DS:
        S = load(ds); df = D[D.dataset == ds]; L = A.league(df); rec, _, _ = A.recommend(df, L)
        for r in L.itertuples():
            M.append(dict(dataset=ds, family=r.family, method=r.label, n_models=r.n, coverage_median=r.cov_med, coverage_min=r.cov_min,
                          coverage_max=r.cov_max, width_median=r.w_med, rel_width_median=r.relw_med, share_valid=r.share_valid,
                          width_is_lower_bound=bool(r.truncated), recommended=(rec is not None and r.label == rec["label"])))
        q = df[df.family == "Quantile regression"]
        rows.append(dict(dataset=ds, n_train=S["data"]["n_train"], n_test=S["data"]["n_test"], n_features=len(S["data"]["features"]),
                         target=S["data"]["target"], target_std_test=S["data"]["target_stats"]["std"], best_model_cv=A.best_model(S),
                         valid_threshold=A.TARGET - A.tol(S["data"]["n_test"]), qr_coverage_median=q.coverage.median(),
                         recommended_method=rec["label"] if rec is not None else None, recommended_coverage=rec.cov_med if rec is not None else None,
                         recommended_rel_width=rec.relw_med if rec is not None else None))
    pd.DataFrame(M).round(6).to_csv(f"{R}/results/benchmark_methods.csv", index=False)
    pd.DataFrame(rows).round(6).to_csv(f"{R}/results/benchmark_datasets.csv", index=False)
    return pd.DataFrame(M), pd.DataFrame(rows)

# ------------------------------------------------------------------ figures
def fig_pipeline(d):
    """The 8-notebook pipeline as a figure (for papers and the README)."""
    fig, ax = plt.subplots(figsize=(10, 4.6)); ax.set_xlim(0, 10); ax.set_ylim(0, 4.6); ax.axis("off")
    def box(x, y, w, h, txt, fc, ec="#333", fs=8, bold=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=0.9))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center", fontsize=fs, fontweight="bold" if bold else "normal", wrap=True)
    def arrow(x0, y0, x1, y1, txt=None):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", lw=0.9, color="#444"))
        if txt: ax.text((x0 + x1) / 2, (y0 + y1) / 2 + 0.08, txt, fontsize=6.5, ha="center", color="#444", style="italic")
    box(0.05, 1.9, 1.5, 0.9, "train.csv\ntest.csv\n(target = last column)", "#f2f3f4", fs=7)
    box(1.8, 1.9, 1.75, 0.9, "NB1  Tuning\nOptuna AutoSampler\n5-fold CV on train only", "#d6eaf8", fs=7.5)
    arrow(1.55, 2.35, 1.8, 2.35)
    nb = [("NB2  Quantile regression\npinball loss, 13 levels", FAM_COL["Quantile regression"]),
          ("NB3  Distributional\nNGBoost, PGBM, IBUG, Treeffuser", FAM_COL["Distributional"]),
          ("NB4  CARD\nguided diffusion", FAM_COL["CARD"]),
          ("NB5  Conformal\nSplit, CV+, J+aB, CQR", FAM_COL["Conformal (MAPIE/PUNCC)"]),
          ("NB6  Weighted + online CP\nNexCP, ACI, FACI, SAOCP, MFCS, COP", FAM_COL["Weighted conformal (NexCP)"]),
          ("NB7  Adaptive coverage\npolicies (ACP)", "#5d6d7e")]
    ys = np.linspace(4.15, 1.05, len(nb))
    for (t, c), y in zip(nb, ys):
        box(4.3, y - 0.27, 2.55, 0.54, t, c + "33", ec=c, fs=7)
        arrow(3.55, 2.35, 4.3, y, None)
    ax.text(3.75, 3.05, "best_params.pkl", fontsize=6.5, style="italic", color="#444", ha="center")
    box(4.3, 0.12, 2.55, 0.5, "NB8  HCM\n(own network, no tuning input)", "#eaecee", fs=7)
    ax.annotate("", xy=(4.3, 0.37), xytext=(0.78, 1.9), arrowprops=dict(arrowstyle="-|>", lw=0.8, color="#999", connectionstyle="arc3,rad=0.25"))
    box(7.45, 1.75, 1.15, 1.2, "Excel outputs\n+ EXECUTED\nnotebooks\n(results/)", "#fef9e7", fs=7.5)
    for y in list(ys) + [0.37]: arrow(6.85, y, 7.45, 2.35)
    box(8.85, 1.75, 1.1, 1.2, "tools/reports\n→ READMEs,\npapers (PDF),\nresults/*.csv", "#e9f7ef", fs=7.5)
    arrow(8.6, 2.35, 8.85, 2.35)
    ax.set_title("One fixed pipeline of eight notebooks, applied unchanged to nine datasets (all runs on Kaggle CPU)", fontsize=9.5, fontweight="bold")
    save(fig, d, "fig_pipeline")

def fig_datasets(R, d):
    fig, axs = plt.subplots(3, 3, figsize=(9, 6.6))
    for ax, ds in zip(axs.ravel(), DS):
        tr = pd.read_csv(f"{A.dsdir(R, ds)}/data/train.csv"); te = pd.read_csv(f"{A.dsdir(R, ds)}/data/test.csv")
        y0, y1 = tr.iloc[:, -1].astype(float), te.iloc[:, -1].astype(float)
        b = np.histogram_bin_edges(np.r_[y0, y1], bins=min(30, max(8, int(np.sqrt(len(y0) + len(y1))))))
        ax.hist(y0, bins=b, color="#5d8aa8", alpha=0.75, label=f"train ({len(y0)})", density=True)
        ax.hist(y1, bins=b, histtype="step", color="#c0392b", lw=1.4, label=f"test ({len(y1)})", density=True)
        sk = pd.Series(np.r_[y0, y1]).skew()
        ax.set_title(f"{NAMES[ds]}  ·  {tr.shape[1]-1} features", fontsize=8.5)
        ax.set_xlabel(f"target {tr.columns[-1]}  (skew {sk:.1f})", fontsize=7.5); ax.set_yticks([]); ax.legend(fontsize=6.5)
    fig.suptitle("The nine datasets: distribution of the target in the training and test sets", fontweight="bold", fontsize=10)
    fig.tight_layout(); save(fig, d, "fig_datasets")

def fig_conformal_explainer(d):
    """Split conformal in three panels, synthetic illustrative data."""
    rng = np.random.default_rng(0); n = 200; x = np.sort(rng.uniform(0, 10, n)); f = lambda t: 2 + 0.8 * t + np.sin(t)
    y = f(x) + rng.normal(0, 0.6 + 0.12 * x, n); fit = f(x) + 0.15
    cal = rng.random(n) < 0.5; s = np.abs(y[cal] - fit[cal]); k = math.ceil((len(s) + 1) * 0.9); q = np.sort(s)[min(k, len(s)) - 1]
    fig, axs = plt.subplots(1, 3, figsize=(10, 3.0))
    a = axs[0]; a.scatter(x[cal], y[cal], s=8, color="#5d8aa8"); a.plot(x, fit, color="k", lw=1)
    for xi, yi, fi in list(zip(x[cal], y[cal], fit[cal]))[::6]: a.plot([xi, xi], [fi, yi], color="#c0392b", lw=0.7)
    a.set_title("1. Score calibration points\n$s_i=|y_i-\\hat f(x_i)|$", fontsize=8.5); a.set_xlabel("x"); a.set_ylabel("y")
    a = axs[1]; a.hist(s, bins=25, color="#aab7c4"); a.axvline(q, color="#c0392b", lw=1.5)
    a.text(q, a.get_ylim()[1] * 0.85, f"  $\\hat q$ = {q:.2f}\n  (score {k} of {len(s)} sorted)", color="#c0392b", fontsize=7.5)
    a.set_title("2. Cut-off: the $\\lceil(n+1)(1-\\alpha)\\rceil$-th\nsmallest score ($\\alpha$ = 0.1)", fontsize=8.5); a.set_xlabel("score")
    a = axs[2]; te = ~cal; cov = np.abs(y[te] - fit[te]) <= q
    a.fill_between(x, fit - q, fit + q, color="#aed6f1", label="$\\hat f(x)\\pm\\hat q$"); a.plot(x, fit, color="k", lw=1)
    a.scatter(x[te][cov], y[te][cov], s=8, color="k", label="covered"); a.scatter(x[te][~cov], y[te][~cov], s=16, marker="x", color="#c0392b", label="missed")
    a.set_title(f"3. Interval for new points\ncoverage {cov.mean():.0%} (target 90%)", fontsize=8.5); a.set_xlabel("x"); a.legend(fontsize=6.5, loc="upper left")
    fig.suptitle("Box 1. Split conformal prediction in three steps (synthetic data, for illustration only)", fontweight="bold", fontsize=9.5)
    fig.tight_layout(); save(fig, d, "fig_conformal_explainer")

def fig_tolerance(d, DT):
    n = np.arange(15, 420); fig, ax = plt.subplots(figsize=(6.4, 3.0))
    ax.fill_between(n, 0.9 - 1.645 * np.sqrt(0.09 / n), 0.9, color="#e8f6ef", label="accepted as valid (one-sided 95%)")
    ax.plot(n, 0.9 - 1.645 * np.sqrt(0.09 / n), color="#1e8449"); ax.axhline(0.9, ls="--", color="k", lw=0.8)
    for r in DT.itertuples():
        ax.scatter(r.n_test, r.valid_threshold, color="#1f4e79", s=14, zorder=3)
        ax.annotate(SHORT[r.dataset], (r.n_test, r.valid_threshold), xytext=(3, -9), textcoords="offset points", fontsize=6.5)
    ax.set_xlabel("number of test points $n$"); ax.set_ylabel("lowest coverage accepted")
    ax.set_title("Why 85% can still be 'valid': sampling tolerance of a 90% coverage estimate", fontsize=9, fontweight="bold")
    ax.legend(fontsize=7, loc="lower right"); save(fig, d, "fig_validity_tolerance")

def fig_family_validity(d, D):
    fams = A.FAMILY_ORDER; M = np.full((len(fams), len(DS)), np.nan)
    for j, ds in enumerate(DS):
        for i, f_ in enumerate(fams):
            g = D[(D.dataset == ds) & (D.family == f_)]
            if len(g): M[i, j] = g.valid.mean()
    fig, ax = plt.subplots(figsize=(8, 3.3)); im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, "–" if np.isnan(M[i, j]) else f"{100*M[i,j]:.0f}", ha="center", va="center", fontsize=7.5)
    ax.set_xticks(range(len(DS))); ax.set_xticklabels([SHORT[x] for x in DS], rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(len(fams))); ax.set_yticklabels(fams, fontsize=8)
    ax.set_title("Share (%) of model × method intervals that reach the 90% target, by family and dataset", fontsize=9, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02); fig.tight_layout(); save(fig, d, "fig_family_validity")

def fig_model_based(d, D):
    """Coverage of the non-conformal intervals per dataset: raw QR, native NGBoost/PGBM, IBUG, Treeffuser, CARD."""
    meths = [("Quantile regression (raw)", FAM_COL["Quantile regression"]), ("NGBoost (native)", "#8e44ad"), ("PGBM (native)", "#c39bd3"),
             ("IBUG", "#5b2c6f"), ("Treeffuser", "#a569bd"), ("CARD", FAM_COL["CARD"])]
    fig, ax = plt.subplots(figsize=(8.4, 3.4)); off = np.linspace(-0.33, 0.33, len(meths))
    for j, ds in enumerate(DS):
        thr = D[D.dataset == ds].valid_threshold.iloc[0]; ax.plot([j - 0.45, j + 0.45], [thr, thr], color="#1e8449", lw=1)
    for (m, c), o in zip(meths, off):
        for j, ds in enumerate(DS):
            g = D[(D.dataset == ds) & (D.method == m)]
            if len(g):
                ax.scatter(np.full(len(g), j + o), g.coverage, s=9, color=c, alpha=0.75, edgecolor="none")
                ax.scatter(j + o, g.coverage.median(), s=30, color=c, edgecolor="k", lw=0.5, zorder=3)
    ax.axhline(0.9, ls="--", color="k", lw=0.7)
    ax.set_xticks(range(len(DS))); ax.set_xticklabels([SHORT[x] for x in DS], fontsize=8); ax.set_ylabel("test coverage of the 90% interval")
    ax.legend(handles=[Patch(color=c, label=m) for m, c in meths] + [Patch(color="#1e8449", label="validity threshold")], fontsize=6.5, ncol=4, loc="lower left")
    ax.set_title("Model-based intervals have no guarantee: coverage of raw QR and predictive distributions", fontsize=9, fontweight="bold")
    save(fig, d, "fig_model_based_coverage")

def fig_conformal_ranking(d, Mt):
    """Median relative width of each conformal method per dataset (only valid-majority methods), as ranks."""
    C = Mt[Mt.family.isin(CONF) & (Mt.n_models >= 3) & ~Mt.width_is_lower_bound]
    P = C.pivot_table(index="method", columns="dataset", values="rel_width_median"); P = P[[x for x in DS if x in P.columns]]
    order = P.median(axis=1).sort_values().index; P = P.loc[order]; Rk = P.rank(axis=0)
    V = C.pivot_table(index="method", columns="dataset", values="share_valid").reindex(index=order, columns=P.columns)
    fig, ax = plt.subplots(figsize=(8.4, 0.3 * len(P) + 1.5)); im = ax.imshow(Rk.values, cmap="viridis", aspect="auto")
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            v = P.values[i, j]
            if np.isnan(v): ax.text(j, i, "n/r", ha="center", va="center", fontsize=6.5); continue
            bad = V.values[i, j] < 0.75
            ax.text(j, i, f"{v:.2f}" + ("*" if bad else ""), ha="center", va="center", fontsize=6.5, color="w" if Rk.values[i, j] < Rk.shape[0] * 0.5 else "k")
    ax.set_xticks(range(P.shape[1])); ax.set_xticklabels([SHORT[x] for x in P.columns], rotation=25, ha="right", fontsize=8)
    ax.set_yticks(range(P.shape[0])); ax.set_yticklabels(P.index, fontsize=8)
    ax.set_title("Conformal methods: median width / std(y) per dataset (colour = rank, dark = sharpest; * = valid on < 75% of models)", fontsize=8.5, fontweight="bold")
    fig.tight_layout(); save(fig, d, "fig_conformal_ranking")

def fig_acp_hcm(d):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.2))
    for j, ds in enumerate(DS):
        ac = load(ds).get("acp") or {}
        for v in ac.values():
            t = v.get("Target_coverage")
            if t is None: continue
            sat = t < 0.01
            a1.scatter(t, v["coverage"], s=16, color="#c0392b" if sat else "#1f4e79", alpha=0.7, marker="x" if sat else "o")
    a1.plot([0, 1], [0, 1], ls="--", color="k", lw=0.7); a1.set_xlabel("learned target coverage $1-\\tilde\\alpha$ (mean)"); a1.set_ylabel("realised coverage")
    a1.set_title("ACP: realised vs learned coverage\n(red × = policy saturated at $\\tilde\\alpha$ = 0.999)", fontsize=8.5)
    sp = [load(ds)["hcm"].get("Spearman") for ds in DS]; cv = [load(ds)["hcm"].get("Coverage_95Interval") for ds in DS]
    y = np.arange(len(DS)); a2.barh(y, sp, color=["#1e8449" if (s or 0) > 0.2 else "#c0392b" for s in sp]); a2.axvline(0, color="k", lw=0.7)
    a2.set_yticks(y); a2.set_yticklabels([f"{SHORT[x]} (cov {c:.0%})" for x, c in zip(DS, cv)], fontsize=7.5); a2.invert_yaxis()
    a2.set_xlabel("Spearman correlation between |error| and predicted σ")
    a2.set_title("HCM: does the predicted uncertainty track the error?\n(label: coverage of its 95% interval)", fontsize=8.5)
    fig.tight_layout(); save(fig, d, "fig_acp_hcm")

def fig_recommended(d, DT, Mt):
    fig, ax = plt.subplots(figsize=(8.4, 3.2)); y = np.arange(len(DS))
    for i, ds in enumerate(DS):
        g = Mt[(Mt.dataset == ds) & Mt.family.isin(CONF) & (Mt.n_models >= 3)]
        ax.scatter(g.rel_width_median, np.full(len(g), i), s=10, color="#aab7c4", zorder=2)
        q = Mt[(Mt.dataset == ds) & (Mt.method == "Quantile regression (raw)")]
        if len(q): ax.scatter(q.rel_width_median, [i], marker="s", s=22, color=FAM_COL["Quantile regression"], zorder=3)
        r = DT[DT.dataset == ds].iloc[0]
        ax.scatter(r.recommended_rel_width, i, s=60, marker="*", color="#1e8449", zorder=4)
        ax.text(r.recommended_rel_width, i + 0.3, r.recommended_method, fontsize=6.5, ha="center", color="#1e8449")
    ax.set_yticks(y); ax.set_yticklabels([NAMES[x] for x in DS], fontsize=8); ax.invert_yaxis(); ax.set_xlim(0, 3.2)
    ax.set_xlabel("median interval width / std of the test target")
    ax.legend(handles=[Patch(color="#aab7c4", label="conformal methods"), Patch(color=FAM_COL["Quantile regression"], label="raw quantile regression (not valid on most)"),
                       Patch(color="#1e8449", label="recommended (narrowest valid)")], fontsize=6.5, loc="lower right")
    ax.set_title("The price of a guarantee: width of every conformal method, and the recommended one, per dataset", fontsize=9, fontweight="bold")
    save(fig, d, "fig_recommended")

if __name__ == "__main__":
    R = sys.argv[1]; d = f"{R}/docs/figures"
    D = long_table(); Mt, DT = tables(R, D)
    fig_pipeline(d); fig_datasets(R, d); fig_conformal_explainer(d); fig_tolerance(d, DT); fig_family_validity(d, D)
    fig_model_based(d, D); fig_conformal_ranking(d, Mt); fig_acp_hcm(d); fig_recommended(d, DT, Mt)
    c = D[D.family.isin(CONF)]
    print("conformal cases", len(c), "valid", int(c.valid.sum()), f"{c.valid.mean():.3f}")
    print(DT.to_string())
