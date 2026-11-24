"""Concrete showcase: one section per notebook, with comparison figures.

Usage:  python showcase_concrete.py <repo>
Writes docs/figures/concrete/*.png. Reads only stored outputs (no model is re-run).
"""
import json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = sys.argv[1] if len(sys.argv) > 1 else "../.."
E = f"{R}/examples/concrete_uncertainty_analysis"
OUT = f"{R}/docs/figures/concrete"; os.makedirs(OUT, exist_ok=True)

BLUE, ORANGE, AQUA, GREY, INK, INK2 = "#2a78d6", "#eb6834", "#1baf7a", "#c9c8c2", "#0b0b0b", "#52514e"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": INK2, "xtick.color": INK2,
                     "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#ecebe7", "grid.linewidth": 0.8, "axes.axisbelow": True,
                     "figure.facecolor": "white", "savefig.dpi": 160, "savefig.bbox": "tight"})
FEAT = {"C": "Cement (C)", "mp": "Mineral powder (mp)", "FA": "Fine aggregate (FA)", "CA": "Coarse aggregate (CA)",
        "F": "Fly ash (F)", "W_P": "Water/powder ratio (W_P)", "Adm": "Admixture (Adm)"}
SHORT = {"Random Forest": "RandomForest", "Gradient Boosting": "GradientBoosting", "HistGradientBoosting": "HistGB"}


def save(fig, name):
    fig.savefig(f"{OUT}/{name}.png"); plt.close(fig)


# ------------------------------------------------------------------ NB1: tuning
def nb1():
    d = json.load(open(f"{E}/hyperparameter_tuning/results/best_params.json"))
    rows = [dict(model=SHORT.get(k.split(" | ")[0], k.split(" | ")[0]), pruner=k.split(" | ")[1].replace("Pruner", ""),
                 cv=v["cv_rmse"], test=v["test_rmse"], r=v["test_corr_coef"]) for k, v in d.items()]
    df = pd.DataFrame(rows)
    best = df.loc[df.groupby("model").cv.idxmin()].set_index("model").sort_values("cv")
    sd = pd.read_csv(f"{E}/data/test.csv").iloc[:, -1].std()

    # Fig 1: CV RMSE (used to choose) vs test RMSE (checked once), per model
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    y = np.arange(len(best))[::-1]
    for yi, (m, r) in zip(y, best.iterrows()):
        ax.plot([r.cv, r.test], [yi, yi], color=GREY, lw=2, zorder=1)
    ax.scatter(best.cv, y, s=70, color=BLUE, zorder=3, label="CV RMSE on training data (used to choose)")
    ax.scatter(best.test, y, s=70, facecolor="white", edgecolor=ORANGE, lw=2, zorder=3, label="Test RMSE (checked once, after choosing)")
    ax.set_yticks(y); ax.set_yticklabels([f"★ {m}" if i == 0 else m for i, m in enumerate(best.index)])
    ax.get_yticklabels()[0].set_fontweight("bold")
    ax.set_xlabel("RMSE (MPa) — lower is better"); ax.set_xlim(0, 8.5)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_title("Tuned models on concrete, sorted by cross-validated RMSE", loc="left", fontsize=11, color=INK)
    save(fig, "c1_rmse")

    # Fig 2: does the pruner matter?  CV RMSE of each model x pruner, relative to the model's best
    piv = df.pivot(index="model", columns="pruner", values="cv").loc[best.index]
    rel = (piv.div(piv.min(axis=1), axis=0) - 1) * 100
    fig, ax = plt.subplots(figsize=(8, 4.4))
    im = ax.imshow(rel.values, cmap="Blues", vmin=0, vmax=8, aspect="auto")
    ax.set_xticks(range(rel.shape[1])); ax.set_xticklabels(rel.columns, rotation=30, ha="right")
    ax.set_yticks(range(rel.shape[0])); ax.set_yticklabels(rel.index); ax.grid(False)
    for i in range(rel.shape[0]):
        for j in range(rel.shape[1]):
            v = rel.values[i, j]
            ax.text(j, i, "best" if v < 0.05 else f"+{v:.0f}%" if v >= 1 else f"+{v:.1f}%", ha="center", va="center",
                    fontsize=8, color="white" if v > 6 else INK)
    cb = fig.colorbar(im, ax=ax, fraction=0.03); cb.set_label("CV RMSE above the model's best pruner (%, capped at 8)")
    ax.set_title("Effect of the pruner on the best CV RMSE (same model, same sampler)", loc="left", fontsize=11, color=INK)
    save(fig, "c1_pruners")

    # Fig 3: predicted vs measured strength for the chosen model, with the median of the others
    t = pd.read_excel(f"{E}/hyperparameter_tuning/results/test_results.xlsx", sheet_name="data")
    yt = t.iloc[:, 7]; pc = t["CatBoost Predictions"]
    fig, ax = plt.subplots(figsize=(5, 5))
    lo, hi = yt.min() - 3, yt.max() + 3
    ax.fill_between([lo, hi], [lo - 3, hi - 3], [lo + 3, hi + 3], color="#e6eef9", lw=0, label="±3 MPa band")
    ax.plot([lo, hi], [lo, hi], color=INK2, lw=1, ls="--")
    ax.scatter(yt, pc, s=28, color=BLUE, edgecolor="white", lw=0.6, label=f"CatBoost (test RMSE {best.loc['CatBoost','test']:.2f})", zorder=3)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("Measured compressive strength (MPa)"); ax.set_ylabel("Predicted (MPa)")
    ax.legend(loc="upper left", frameon=False, fontsize=9)
    ax.set_title("Selected model: predictions on the 95 test mixes", loc="left", fontsize=11, color=INK)
    save(fig, "c1_pred")

    # Fig 4: which inputs matter? share of importance, mean over the tree models (TabNet excluded: unscaled values)
    xl = pd.read_excel(f"{E}/hyperparameter_tuning/results/test_results.xlsx", sheet_name=None)
    sh, li = [], []
    for k, v in xl.items():
        if not k.endswith(" Scores") or k.startswith("TabNet"): continue
        a = v.iloc[:7].set_index("Feature")["LIME Importance"].astype(float)
        b = v.iloc[8:15].set_index(v.columns[0]).iloc[:, 0].astype(float)
        li.append(a / a.sum()); sh.append(b / b.sum())
    S = pd.concat(sh, axis=1).mean(axis=1); L = pd.concat(li, axis=1).mean(axis=1).reindex(S.index)
    order = S.sort_values().index
    names = FEAT
    fig, ax = plt.subplots(figsize=(7.5, 4))
    yy = np.arange(len(order)); h = 0.38
    ax.barh(yy + h / 2, S[order] * 100, h, color=BLUE, label="SHAP (average effect over all test mixes)")
    ax.barh(yy - h / 2, L[order] * 100, h, color=ORANGE, label="LIME (local linear fit around each mix)")
    ax.set_yticks(yy); ax.set_yticklabels([names.get(f, f) for f in order])
    ax.set_xlabel("Share of total importance (%), mean over 9 tree models")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Feature importance: SHAP and LIME", loc="left", fontsize=11, color=INK)
    save(fig, "c1_importance")

    # Fig 5: why not grid search? size of each search space vs the trials actually run
    grid = {"GradientBoosting": 3.19e8, "LightGBM": 4.42e6, "RandomForest": 2.56e6, "GPBoost": 1.47e6, "XGBoost": 5.18e5,
            "HistGB": 1.38e5, "CatBoost": 2.59e4, "TabNet": 1.38e4, "NGBoost": 384}
    fig, ax = plt.subplots(figsize=(7.5, 3.8))
    ks = list(grid)[::-1]
    ax.barh(ks, [grid[k] for k in ks], color=GREY, height=0.6, label="combinations a full grid search would have to try")
    ax.axvline(50, color=BLUE, lw=2); ax.text(55, len(ks) - 0.6, "50 trials per study\n(what the AutoSampler used)", color=BLUE, fontsize=9, va="top")
    ax.set_xscale("log"); ax.set_xlabel("Number of hyperparameter combinations (log scale)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    ax.set_title("Search-space size relative to the trial budget", loc="left", fontsize=11, color=INK)
    save(fig, "c1_search")
    return best, rel, S, L



# ------------------------------------------------------------------ shared helpers for steps 2-8
S_ = json.load(open(f"{os.path.dirname(os.path.abspath(__file__))}/summary/concrete.json"))
MODELS = ["CatBoost", "GradientBoosting", "NGBoost", "PGBM", "XGBoost", "LightGBM", "GPBoost", "HistGB", "TabNet"]  # CV order
N = 95; THR = 0.9 - 1.645 * np.sqrt(0.09 / N)
SD = 10.035
FILE = {"HistGB": "HGBM", "GradientBoosting": "Gradient Boosting"}
GREEN_BAND = "#e3f4ec"


def cov_axis(ax, label=True):
    ax.axvspan(THR * 100, 100.5, color=GREEN_BAND, lw=0, zorder=0)
    ax.axvline(90, color=INK2, lw=1, ls="--", zorder=1)
    ax.set_xlim(0, 101)


def dot_panels(rows, cols, titles, fname, colors=None, xlabels=None, suptitle=None, height=None, cov_cols=(0,), logx=()):
    """rows: list of labels; cols: list of arrays (one per panel)."""
    k = len(cols)
    fig, axs = plt.subplots(1, k, figsize=(3.3 * k + 1.6, height or 0.36 * len(rows) + 1.3), sharey=True)
    axs = np.atleast_1d(axs); y = np.arange(len(rows))[::-1]
    for i, (ax, v) in enumerate(zip(axs, cols)):
        v = np.asarray(v, float)
        if i in cov_cols: cov_axis(ax)
        ax.hlines(y, 0.5 if i in logx else 0, v, color=GREY, lw=1.5, zorder=2)
        if i in logx: ax.set_xscale("log")
        ax.scatter(v, y, s=55, color=colors if colors is not None else BLUE, zorder=3, edgecolor="white", lw=0.8)
        ax.set_title(titles[i], loc="left", fontsize=10, color=INK)
        if xlabels: ax.set_xlabel(xlabels[i])
        ax.grid(axis="y", visible=False)
    axs[0].set_yticks(y); axs[0].set_yticklabels(rows)
    if suptitle: fig.suptitle(suptitle, x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); save(fig, fname)


def heat(ax, M, rows, cols, fmt, valid=None, cmap="Blues", vmin=None, vmax=None):
    im = ax.imshow(M, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(cols, rotation=35, ha="right")
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows); ax.grid(False)
    lo_ = vmin if vmin is not None else np.nanmin(M); hi_ = vmax if vmax is not None else np.nanmax(M)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            if np.isnan(v): continue
            bad = valid is not None and not valid[i, j]
            ax.text(j, i, fmt(v) + (" ✗" if bad else ""), ha="center", va="center", fontsize=7.5,
                    color="white" if (v - lo_) / (hi_ - lo_) > 0.6 else INK, fontweight="bold" if bad else "normal")
    return im


# ------------------------------------------------------------------ NB2: quantile regression
def nb2():
    Q = S_["quantile_regression"]
    rows = [m for m in MODELS if m in Q]
    cov = [Q[m]["coverage90"] * 100 for m in rows]; w = [Q[m]["width90"] for m in rows]
    cr = [Q[m]["crossing_rate"] * 100 for m in rows]; crps = [Q[m]["crps"] for m in rows]
    col = [AQUA if c >= THR * 100 else ORANGE for c in cov]
    dot_panels(rows, [cov, w, cr, crps], ["Coverage of the 90% interval (%)", "Mean width (MPa)", "Points with crossed quantiles (%)",
               "CRPS (MPa, lower = better)"], "c2_qr", colors=col, xlabels=["dashed = 90% target, green = statistically OK", "", "", ""])
    return pd.DataFrame(dict(coverage=cov, width=w, crossing=cr, crps=crps), index=rows)


# ------------------------------------------------------------------ NB3: predictive distributions
PROB = [("NGBoost", "native", "NGBoost_Prob"), ("PGBM", "native", "PGBM_Prob"), ("XGBoost", "IBUG", "XGB_Prob"),
        ("LightGBM", "IBUG", "LGBM_Prob"), ("CatBoost", "Treeffuser", "CatBoost_Prob_TF"), ("GradientBoosting", "Treeffuser", "GBoosting_Prob_TF"),
        ("XGBoost", "Treeffuser", "XGB_Prob_TF"), ("LightGBM", "Treeffuser", "LGBM_Prob_TF"), ("GPBoost", "Treeffuser", "GPBoost_Prob_TF")]
FAMC = {"native": ORANGE, "IBUG": BLUE, "Treeffuser": AQUA}


def nb3():
    from scipy.stats import norm
    P = S_["probabilistic"]; rows, cov, w, crps, rmse, col, pits = [], [], [], [], [], [], []
    for m, fam, f in PROB:
        key = m + (" (TF)" if fam == "Treeffuser" else "")
        v = P[key]; rows.append(f"{m} · {fam}"); col.append(FAMC[fam])
        cov.append(v["coverage90_gauss"] * 100); w.append(v["width90_gauss"]); crps.append(v.get("crps_gauss") or v.get("crps")); rmse.append(v["rmse"])
        d = pd.read_excel(f"{E}/probabilistic_distribution/results/{f}.xlsx", sheet_name="Predictions")
        pits.append(norm.cdf((d.Actual - d.Mean) / d.StdDev))
    dot_panels(rows, [cov, w, crps, rmse], ["Coverage of the 90% interval (%)", "Mean width (MPa)", "CRPS (MPa)", "RMSE of the mean (MPa)"],
               "c3_prob", colors=col, xlabels=["orange = native, blue = IBUG, aqua = Treeffuser", "log scale", "log scale", ""], logx=(1, 2))
    fig, axs = plt.subplots(3, 3, figsize=(9, 7), sharex=True)
    for ax, r, p, c in zip(axs.flat, rows, pits, col):
        ax.hist(p, bins=10, range=(0, 1), color=c, edgecolor="white", density=True)
        ax.axhline(1, color=INK2, ls="--", lw=1); ax.set_title(r, fontsize=9, loc="left", color=INK); ax.set_ylim(0, 4.5); ax.grid(axis="x", visible=False)
    for ax in axs[-1]: ax.set_xlabel("PIT value")
    fig.suptitle("PIT histograms (dashed line: uniform reference; U-shape: under-dispersed; hump: over-dispersed)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); save(fig, "c3_pit")
    return pd.DataFrame(dict(coverage=cov, width=w, crps=crps, rmse=rmse), index=rows)


# ------------------------------------------------------------------ NB4: CARD
def nb4():
    from scipy.stats import norm
    C = S_["card"]; Q = S_["quantile_regression"]; rows = [m for m in MODELS if m in C]
    cov = [C[m]["coverage90_gauss"] * 100 for m in rows]; w = [C[m]["width90_gauss"] for m in rows]
    crps = [C[m]["crps"] for m in rows]; ks = [C[m]["pit_ks"] for m in rows]
    dot_panels(rows, [cov, w, crps], ["Coverage of the 90% interval (%)", "Mean width (MPa)", "CRPS (MPa)"], "c4_card",
               colors=[AQUA if c >= THR * 100 else ORANGE for c in cov])
    # CRPS comparison: same model, quantile regression vs CARD
    fig, ax = plt.subplots(figsize=(7.5, 3.9)); y = np.arange(len(rows))[::-1]
    qc = [Q[m]["crps"] for m in rows]
    for yi, a, b in zip(y, qc, crps): ax.plot([a, b], [yi, yi], color=GREY, lw=2, zorder=1)
    ax.scatter(qc, y, s=60, color=ORANGE, label="Quantile regression (step 2)", zorder=3)
    ax.scatter(crps, y, s=60, color=BLUE, label="CARD (step 4)", zorder=3)
    ax.set_yticks(y); ax.set_yticklabels(rows); ax.set_xlabel("CRPS (MPa) — lower is better"); ax.legend(frameon=False, fontsize=9, loc="upper right")
    ax.set_title("CRPS by base model: quantile regression and CARD", loc="left", fontsize=11, color=INK)
    save(fig, "c4_crps_vs_qr")
    d = pd.read_excel(f"{E}/probabilistic_distribution_card/results/CatBoost.xlsx", sheet_name="Predictions").sort_values("Actual").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8, 3.6)); x = np.arange(len(d))
    lo, hi = d.Mean - 1.645 * d.StdDev, d.Mean + 1.645 * d.StdDev; inside = (d.Actual >= lo) & (d.Actual <= hi)
    ax.vlines(x, lo, hi, color=BLUE, lw=2.2, alpha=0.55, label="CARD 90% interval")
    ax.scatter(x[inside], d.Actual[inside], s=12, color=INK, zorder=3, label="measured, inside")
    ax.scatter(x[~inside], d.Actual[~inside], s=16, color=ORANGE, zorder=3, label="measured, outside")
    ax.set_xlabel("Test mixes, sorted by measured strength"); ax.set_ylabel("Strength (MPa)"); ax.legend(frameon=False, fontsize=8.5)
    ax.set_title(f"CARD intervals, CatBoost guide ({inside.mean()*100:.0f}% coverage)", loc="left", fontsize=11, color=INK)
    save(fig, "c4_band")
    return pd.DataFrame(dict(coverage=cov, width=w, crps=crps, crps_qr=qc, pit_ks=ks), index=rows)


# ------------------------------------------------------------------ NB5 / NB6: conformal heatmaps
def conformal_heat(key, methods, labels, fname, title):
    D = S_[key]; rows = [m for m in MODELS if m in D]
    cov = np.array([[D[m][k]["coverage"] if k in D[m] else np.nan for k in methods] for m in rows]) * 100
    wid = np.array([[D[m][k]["width"] if k in D[m] else np.nan for k in methods] for m in rows]) / SD
    fig, axs = plt.subplots(2, 1, figsize=(1.0 * len(methods) + 2.5, 8.6))
    im = heat(axs[0], cov, rows, labels, lambda v: f"{v:.0f}", valid=cov >= THR * 100, cmap="Greens", vmin=70, vmax=100)
    fig.colorbar(im, ax=axs[0], fraction=0.025).set_label("coverage (%)")
    axs[0].set_title("Coverage (%) of the 90% interval · ✗ = below the 84.9% validity threshold", loc="left", fontsize=10, color=INK)
    im = heat(axs[1], wid, rows, labels, lambda v: f"{v:.2f}", cmap="Blues", vmin=0.6, vmax=2.5)
    fig.colorbar(im, ax=axs[1], fraction=0.025).set_label("width / std(y)")
    axs[1].set_title("Width relative to the spread of strength (std = 10.0 MPa) · lower = sharper", loc="left", fontsize=10, color=INK)
    fig.suptitle(title, x=0.01, ha="left", fontsize=11, color=INK); fig.tight_layout(); save(fig, fname)
    return pd.DataFrame(cov, index=rows, columns=labels), pd.DataFrame(wid, index=rows, columns=labels)


def band(ax, d, lo, hi, color, title):
    x = np.arange(len(d)); inside = (d.y >= d[lo]) & (d.y <= d[hi])
    ax.vlines(x, d[lo], d[hi], color=color, lw=2.2, alpha=0.55)
    ax.scatter(x[inside], d.y[inside], s=10, color=INK, zorder=3); ax.scatter(x[~inside], d.y[~inside], s=14, color=ORANGE, zorder=3)
    ax.set_title(f"{title}: coverage {inside.mean()*100:.0f}%, mean width {(d[hi]-d[lo]).mean():.1f} MPa", loc="left", fontsize=9.5, color=INK)
    ax.set_ylim(5, 80)


def nb5():
    c, w = conformal_heat("mapie_puncc", ["MAPIE SCP", "MAPIE CV+", "MAPIE J+aB", "PUNCC SplitCP", "PUNCC CVPlus", "PUNCC CQR"],
                          ["Split (MAPIE)", "CV+ (MAPIE)", "J+aB (MAPIE)", "Split (PUNCC)", "CV+ (PUNCC)", "CQR (PUNCC)"], "c5_heat",
                          "Step 5 · split, cross and bootstrap conformal on every model")
    m = pd.read_excel(f"{E}/conformal_predictions_mapie_puncc/results/CatBoost.xlsx", sheet_name="all_pred_values")
    pu = pd.read_excel(f"{E}/conformal_predictions_mapie_puncc/results/CatBoost.xlsx", sheet_name="conformal_predictions_PUNCC_values")
    d = pd.DataFrame({"y": m[m.strategy == "SCP"].y_true.values})
    for st in ["SCP", "J+aB"]:
        q = m[m.strategy == st]; d[st + "_lo"] = q.ymin.values; d[st + "_hi"] = q.ymax.values
    d["CQR_lo"] = pu.CQR_lower.values; d["CQR_hi"] = pu.CQR_upper.values
    d = d.sort_values("y").reset_index(drop=True)
    fig, axs = plt.subplots(3, 1, figsize=(8, 7.5), sharex=True)
    for ax, (k, t, c_) in zip(axs, [("SCP", "Split conformal (MAPIE)", BLUE), ("J+aB", "Jackknife+-after-bootstrap (MAPIE)", AQUA), ("CQR", "CQR (PUNCC)", ORANGE)]):
        band(ax, d, k + "_lo", k + "_hi", c_, t); ax.set_ylabel("MPa")
    axs[-1].set_xlabel("Test mixes sorted by measured strength (orange: outside the interval)")
    fig.suptitle("CatBoost: split, J+aB and CQR intervals on the test set", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); save(fig, "c5_band")
    return c, w


NB6M = ["NexCP Split", "NexCP Full", "NexCP CV+", "NexCP Jackknife+ aB", "Online Split", "Online CV+", "Online Jackknife+ aB",
        "ACI", "FACI", "SF-OGD", "SAOCP", "MFCS Split", "MFCS Full", "COP", "AlphaNet Conformal"]
NB6L = ["NexCP Split", "NexCP Full†", "NexCP CV+", "NexCP J+aB", "Online Split", "Online CV+", "Online J+aB",
        "ACI", "FACI", "SF-OGD", "SAOCP", "MFCS Split", "MFCS Full", "COP", "AlphaNet"]


def nb6():
    c, w = conformal_heat("nexcp_online", NB6M, NB6L, "c6_heat", "Step 6 · weighted (NexCP), online, MFCS and COP conformal on every model")
    v = pd.read_excel(f"{E}/conformal_predictions_nexcp_adaptivecp_mfcs/results/CatBoost.xlsx", sheet_name="ONLINECP_Values")
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.9)); t = np.arange(1, len(v) + 1)
    for k, col, lab in [("Online-Split", GREY, "Online split (fixed cut-off)"), ("ACI", ORANGE, "ACI"), ("FACI", BLUE, "FACI"), ("MFCS-Full", AQUA, "MFCS Full")]:
        y, lo, hi = v[k + "_true"], v[k + "_lower"], v[k + "_upper"]
        hit = ((y >= lo) & (y <= hi)).astype(float)
        axs[0].plot(t, np.cumsum(hit) / t * 100, color=col, lw=2, label=lab)
        axs[1].plot(t, hi - lo, color=col, lw=2 if k != "Online-Split" else 2.5, label=lab)
    axs[0].axhline(90, color=INK2, ls="--", lw=1); axs[0].set_ylim(60, 101); axs[0].set_xlabel("test mixes revealed so far"); axs[0].set_ylabel("running coverage (%)")
    axs[1].set_xlabel("test mix (in stored order)"); axs[1].set_ylabel("interval width (MPa)")
    axs[0].set_title("Running coverage", loc="left", fontsize=10, color=INK)
    axs[1].set_title("Interval width per test mix", loc="left", fontsize=10, color=INK)
    axs[1].legend(frameon=False, fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(-0.1, -0.2))
    fig.suptitle("CatBoost: online conformal methods with sequentially revealed labels", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); save(fig, "c6_online")
    ex = {m: S_["nexcp_online"][m]["EXTREME CP"] for m in MODELS}
    return c, w, ex


# ------------------------------------------------------------------ NB7: ACP
def nb7():
    A = S_["acp"]; rows = [m for m in MODELS if m in A]
    fig, axs = plt.subplots(1, 2, figsize=(11, 4), sharey=True); y = np.arange(len(rows))[::-1]
    tg = [A[m]["Target_coverage"] * 100 for m in rows]; cv = [A[m]["coverage"] * 100 for m in rows]
    for yi, a, b in zip(y, tg, cv): axs[0].plot([a, b], [yi, yi], color=GREY, lw=2, zorder=1)
    axs[0].scatter(tg, y, s=60, color=ORANGE, label="learned target 1 − α̃", zorder=3); axs[0].scatter(cv, y, s=60, color=BLUE, label="realised coverage", zorder=3)
    axs[0].axvline(90, color=INK2, ls="--", lw=1); axs[0].set_xlim(40, 101); axs[0].legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    axs[0].set_title("Learned target and realised coverage", loc="left", fontsize=10, color=INK)
    jab = [S_["mapie_puncc"][m]["MAPIE J+aB"]["width"] for m in rows]; aw = [A[m]["interval_width"] for m in rows]
    h = 0.38
    axs[1].barh(y + h / 2, aw, h, color=BLUE, label="ACP"); axs[1].barh(y - h / 2, jab, h, color=GREY, label="J+aB (step 5, 90%)")
    axs[1].set_xlabel("mean width (MPa)"); axs[1].legend(frameon=False, fontsize=8.5, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=2)
    axs[1].set_title("Mean width: ACP and J+aB", loc="left", fontsize=10, color=INK)
    axs[0].set_yticks(y); axs[0].set_yticklabels(rows)
    fig.suptitle("Step 7 · adaptive coverage policies (λ = 50 chosen for every model)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); save(fig, "c7_acp")
    L = pd.read_excel(f"{E}/conformal_predictions_adaptive_coverage_policies/results/CatBoost.xlsx", sheet_name="acp_lambda_comparison")
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    ax.plot(L.Target_coverage * 100, L.test_coverage * 100, "-o", color=BLUE, ms=8)
    for _, r in L.iterrows(): ax.annotate(f"λ = {r['lambda']:.0f}\nwidth {r.width:.1f} MPa", (r.Target_coverage * 100, r.test_coverage * 100), textcoords="offset points", xytext=(8, -18), fontsize=8.5, color=INK2)
    ax.set_xlim(20, 80); ax.set_ylim(60, 100)
    ax.set_xlabel("learned target coverage (%)"); ax.set_ylabel("realised test coverage (%)")
    ax.set_title("CatBoost: effect of λ on coverage and width", loc="left", fontsize=11, color=INK)
    save(fig, "c7_lambda")
    return pd.DataFrame(dict(target=tg, coverage=cv, width=aw, jab_width=jab), index=rows)


# ------------------------------------------------------------------ NB8: HCM
def nb8():
    H = S_["hcm"]; d = pd.read_excel(f"{E}/hyperspherical_confidence_mapping_hcm/results/HCM.xlsx", sheet_name="Predictions")
    pa = pd.read_excel(f"{E}/hyperspherical_confidence_mapping_hcm/results/HCM.xlsx", sheet_name="Predicted_vs_Actual")
    fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.9))
    lv = ["±1σ̂", "±2σ̂", "±3σ̂"]; nom = [68.3, 95.4, 99.7]; got = [H["Cov@1Sigma"] * 100, H["Cov@2Sigma"] * 100, H["Cov@3Sigma"] * 100]
    x = np.arange(3); axs[0].bar(x - 0.2, nom, 0.4, color=GREY, label="nominal"); axs[0].bar(x + 0.2, got, 0.4, color=BLUE, label="HCM")
    axs[0].set_xticks(x); axs[0].set_xticklabels(lv); axs[0].set_ylim(0, 105); axs[0].legend(frameon=False, fontsize=8.5); axs[0].set_ylabel("coverage (%)")
    axs[0].set_title("Coverage at 1, 2, 3 σ̂", loc="left", fontsize=10, color=INK)
    lo, hi = 15, 70
    axs[1].plot([lo, hi], [lo, hi], ls="--", color=INK2, lw=1); axs[1].scatter(pa.Actual, pa.Mean, s=16, color=BLUE)
    axs[1].set_xlim(lo, hi); axs[1].set_ylim(lo, hi); axs[1].set_xlabel("measured (MPa)"); axs[1].set_ylabel("HCM prediction (MPa)")
    axs[1].set_title(f"Point prediction (RMSE {H['RMSE']:.1f} vs CatBoost 3.0)", loc="left", fontsize=10, color=INK)
    axs[2].scatter(d.Sigma_Hat, d.Absolute_Error, s=16, color=ORANGE); axs[2].set_xlabel("predicted uncertainty σ̂ (MPa)"); axs[2].set_ylabel("actual |error| (MPa)")
    axs[2].set_title(f"σ̂ against absolute error (Spearman {H['Spearman']:+.2f})", loc="left", fontsize=10, color=INK)
    fig.suptitle("Step 8 · hyperspherical confidence mapping (one network, no base model)", x=0.01, ha="left", fontsize=11, color=INK)
    fig.tight_layout(); save(fig, "c8_hcm")
    return H


# ------------------------------------------------------------------ overall: every interval in one picture
def overall():
    pts = []
    for m, v in S_["quantile_regression"].items(): pts.append(("Quantile regression", m, v["coverage90"], v["width90"]))
    for k, v in S_["probabilistic"].items(): pts.append(("Predictive distribution", k, v["coverage90_gauss"], v["width90_gauss"]))
    for m, v in S_["card"].items(): pts.append(("CARD", m, v["coverage90_gauss"], v["width90_gauss"]))
    for m, d in S_["mapie_puncc"].items():
        for k, v in d.items(): pts.append(("Conformal (steps 5–6)", m, v["coverage"], v["width"]))
    for m, d in S_["nexcp_online"].items():
        for k, v in d.items():
            if k in ("EXTREME CP", "AlphaNet Conformal") or v.get("note"): continue
            pts.append(("Conformal (steps 5–6)", m, v["coverage"], v["width"]))
    D = pd.DataFrame(pts, columns=["family", "model", "cov", "w"]); D["cov"] *= 100; D["rw"] = D.w / SD
    fig, ax = plt.subplots(figsize=(8.5, 5))
    ax.axvspan(THR * 100, 101, color=GREEN_BAND, lw=0); ax.axvline(90, color=INK2, ls="--", lw=1)
    for fam, col, mk in [("Conformal (steps 5–6)", BLUE, "o"), ("Quantile regression", ORANGE, "s"), ("Predictive distribution", "#6250d6", "D"), ("CARD", "#eda100", "^")]:
        q = D[D.family == fam]; ax.scatter(q["cov"], q.rw, s=26 if fam.startswith("Conf") else 42, color=col, marker=mk, alpha=0.75 if fam.startswith("Conf") else 0.95,
                                           edgecolor="white", lw=0.5, label=f"{fam} ({len(q)})")
    ax.set_yscale("log"); ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 4, 8, 16, 32]); ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "4", "8", "16", "32"])
    ax.set_xlim(10, 101); ax.set_xlabel("coverage of the 90% interval (%) — green band = statistically OK")
    ax.set_ylabel("width / std(strength), log scale — lower = sharper")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Coverage and relative width of every 90% interval on the concrete dataset", loc="left", fontsize=11, color=INK)
    save(fig, "c9_overall")
    return D


if __name__ == "__main__":
    best, rel, S, L = nb1()
    print(nb2().round(2).to_string()); print(nb3().round(2).to_string()); print(nb4().round(2).to_string())
    c, w = nb5(); print(c.round(1).to_string()); print(w.round(2).to_string())
    c, w, ex = nb6(); print(c.round(1).to_string()); print(w.round(2).to_string()); print({k: (v["coverage"], round(v["width"], 1)) for k, v in ex.items()})
    print(nb7().round(2).to_string()); nb8()
    D = overall(); print(D.groupby("family").apply(lambda q: pd.Series(dict(n=len(q), valid=(q["cov"] >= THR * 100).mean(), med_rw=q.rw.median()))).round(2))
