"""figures.py: matplotlib figures for one dataset (PNG for READMEs, PDF for the papers)."""
import os, glob
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import analysis as A

plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 200,
                     "axes.titlesize": 10, "axes.titleweight": "bold", "legend.frameon": False, "font.family": "DejaVu Sans"})
FAM_COL = {"Quantile regression": "#c0392b", "Distributional": "#8e44ad", "CARD": "#d35400", "Conformal (MAPIE/PUNCC)": "#2471a3",
           "Weighted conformal (NexCP)": "#17a589", "Online conformal": "#7f8c8d", "COP": "#b7950b"}

def save(fig, d, name):
    os.makedirs(d, exist_ok=True)
    fig.savefig(f"{d}/{name}.png", bbox_inches="tight"); fig.savefig(f"{d}/{name}.pdf", bbox_inches="tight"); plt.close(fig)

def accuracy(S, d):
    t = S["tuning"]; ms = [m for m in A.MODEL_ORDER if m in t]; ms.sort(key=lambda m: t[m]["cv_rmse"])
    best = A.best_model(S); fig, ax = plt.subplots(figsize=(6.4, 2.8)); x = np.arange(len(ms))
    ax.bar(x - 0.2, [t[m]["cv_rmse"] for m in ms], 0.4, label="5-fold CV RMSE (train)", color="#aab7c4")
    ax.bar(x + 0.2, [t[m]["test_rmse"] for m in ms], 0.4, label="Test RMSE", color=["#1f4e79" if m == best else "#5d8aa8" for m in ms])
    ax.set_xticks(x); ax.set_xticklabels(ms, rotation=30, ha="right"); ax.set_ylabel(f"RMSE ({S['data']['target']})")
    ax.set_title("Point accuracy of the tuned models (sorted by CV RMSE)"); ax.legend(loc="upper left")
    save(fig, d, "fig_accuracy")

def cov_width(df, S, d):
    n = S["data"]["n_test"]; tl = A.tol(n); fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.axhspan(A.TARGET - tl, 1.0, color="#e8f6ef", zorder=0, label=f"coverage consistent with 90% (n={n})")
    ax.axhline(A.TARGET, color="k", lw=0.8, ls="--")
    for fam in A.FAMILY_ORDER:
        g = df[df.family == fam]
        if len(g): ax.scatter(g.rel_width.clip(upper=6), g.coverage, s=16, alpha=0.8, color=FAM_COL[fam], label=fam, edgecolor="none")
    ax.set_xlabel("Mean interval width / std of the test target (clipped at 6)"); ax.set_ylabel("Empirical coverage (test)")
    ax.set_ylim(max(0, df.coverage.min() - 0.05), 1.02); ax.set_title("Coverage vs. width of every model × method (target 90%)")
    ax.legend(fontsize=7, loc="lower right", ncol=2); save(fig, d, "fig_coverage_width")

def league_plot(L, S, d):
    L = L[L.n >= 3].copy(); L = L.iloc[::-1]; y = np.arange(len(L)); tl = A.tol(S["data"]["n_test"])
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.0, 0.22 * len(L) + 1.0), sharey=True, gridspec_kw={"width_ratios": [1.2, 1]})
    cols = [FAM_COL[f] for f in L.family]
    a1.hlines(y, L.cov_min, L.cov_max, color=cols, lw=2, alpha=0.5); a1.scatter(L.cov_med, y, color=cols, s=18, zorder=3)
    a1.axvline(A.TARGET, color="k", ls="--", lw=0.8); a1.axvspan(A.TARGET - tl, 1, color="#e8f6ef", zorder=0)
    a1.set_yticks(y); a1.set_yticklabels(L.label); a1.set_xlabel("Coverage: median (dot), min–max over models")
    a2.barh(y, L.relw_med.clip(upper=6), color=cols, alpha=0.85); a2.set_xlabel("Median width / std(y_test)")
    fig.suptitle("Interval methods at the 90% target", fontweight="bold", fontsize=10); fig.tight_layout()
    save(fig, d, "fig_method_league")

def intervals(R, S, d, model, method="MAPIE CV+"):
    f = glob.glob(f"{A.dsdir(R, S['dataset'])}/conformal_predictions_mapie_puncc/results/*.xlsx")
    f = [x for x in f if A.canon_file(x) == model]
    if not f: return None
    xl = pd.read_excel(f[0], sheet_name=None); a = next((v for k, v in xl.items() if k.startswith("all_pred_values")), None)
    strat = method.split(" ", 1)[1]; g = a[a.strategy == strat].sort_values("y_true").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 2.8)); i = np.arange(len(g))
    ax.fill_between(i, g.ymin, g.ymax, color="#aed6f1", label=f"90% {method} interval")
    ax.plot(i, g.y_pred, color="#1f4e79", lw=1, label="prediction"); out = ~g.is_covered.astype(bool)
    ax.scatter(i[~out], g.y_true[~out], s=8, color="k", label="observed (covered)"); ax.scatter(i[out], g.y_true[out], s=14, color="#c0392b", marker="x", label="observed (missed)")
    ax.set_xlabel("Test points sorted by observed value"); ax.set_ylabel(S["data"]["target"])
    ax.set_title(f"{model}: {method} intervals, coverage {g.is_covered.mean():.1%}"); ax.legend(fontsize=7, loc="upper left")
    save(fig, d, "fig_intervals"); return f"{method}"

def crps_plot(S, d):
    rows = []
    for m, v in S["quantile_regression"].items():
        if v.get("crps") is not None: rows.append(("Quantile regression", m, v["crps"]))
    for tag, v in S["probabilistic"].items():
        if v.get("crps_gauss") is not None:
            b = tag.replace(" (TF)", ""); meth = "Treeffuser" if "(TF)" in tag else ("native" if b in ("NGBoost", "PGBM") else "IBUG")
            rows.append((f"Distributional", f"{b} ({meth})", v["crps_gauss"]))
    for m, v in S["card"].items():
        if v.get("crps_gauss") is not None: rows.append(("CARD", m, v["crps_gauss"]))
    df = pd.DataFrame(rows, columns=["family", "model", "crps"]); fams = ["Quantile regression", "Distributional", "CARD"]
    df["o"] = df.family.map({f: i for i, f in enumerate(fams)}); df = df.sort_values(["o", "crps"]).reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(6.4, 0.17 * len(df) + 1.0)); y = np.arange(len(df))[::-1]
    ax.barh(y, df.crps, color=[FAM_COL[f] for f in df.family]); ax.set_yticks(y); ax.set_yticklabels(df.model, fontsize=7)
    ax.set_xscale("log"); ax.set_xlabel("CRPS (target units, log scale; lower is better)")
    from matplotlib.patches import Patch
    from matplotlib.ticker import LogLocator, FormatStrFormatter
    ax.xaxis.set_major_locator(LogLocator(subs=(1, 2, 5))); ax.xaxis.set_major_formatter(FormatStrFormatter("%g"))
    ax.legend(handles=[Patch(color=FAM_COL[f], label=f) for f in fams], fontsize=7, loc="lower right"); ax.set_title("Probabilistic accuracy: CRPS of every predictive distribution")
    save(fig, d, "fig_crps"); return df
