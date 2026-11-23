"""build_summary.py <R> <out_dir>: extract headline numbers of every notebook, per dataset, into <out_dir>/<ds>.json."""
import json, re, sys, glob, os, warnings
import pandas as pd, numpy as np
warnings.filterwarnings("ignore")
R, OUT = sys.argv[1], sys.argv[2]; os.makedirs(OUT, exist_ok=True)
CANON = [("TabNet","TabNet"),("NGB","NGBoost"),("PGBM","PGBM"),("GPBoost","GPBoost"),("CatBoost","CatBoost"),
         ("HistGradient","HistGB"),("HGB","HistGB"),("LightGBM","LightGBM"),("Lightgbm","LightGBM"),("LGBM","LightGBM"),
         ("XGB","XGBoost"),("Random Forest","RandomForest"),("Gradient Boosting","GradientBoosting"),
         ("GradientBoosting","GradientBoosting"),("GBR","GradientBoosting"),("GBoosting","GradientBoosting"),("Boosting","GradientBoosting")]
def canon(s):
    for k, v in CANON:
        if k.lower() in s.lower(): return v
    return s
def f(x):
    try:
        x = float(x); return None if np.isnan(x) else round(x, 6)
    except Exception: return None
def nb_streams(path):
    nb = json.load(open(path))
    for c in nb["cells"]:
        yield "".join(c["source"]), "".join("".join(o.get("text", "")) for o in c.get("outputs", []) if o.get("output_type") == "stream")

def summarise(ds):
    E = f"{R}/examples/{ds}_uncertainty_analysis"; S = {"dataset": ds}
    tr, te = pd.read_csv(f"{E}/data/train.csv"), pd.read_csv(f"{E}/data/test.csv")
    S["data"] = {"n_train": len(tr), "n_test": len(te), "features": list(tr.columns[:-1]), "target": tr.columns[-1],
                 "target_stats": {k: f(v) for k, v in te.iloc[:, -1].describe().items()}}
    # NB1 tuning: per model, the pruner with the lowest CV RMSE (the notebooks' own selection rule)
    bp = json.load(open(f"{E}/hyperparameter_tuning/results/best_params.json")); tun = {}
    for k, v in bp.items():
        m = canon(k.split(" | ")[0])
        if v.get("cv_rmse") is None: continue
        if m not in tun or v["cv_rmse"] < tun[m]["cv_rmse"]:
            tun[m] = {"pruner": v.get("pruner"), "cv_rmse": f(v["cv_rmse"]), "test_rmse": f(v.get("test_rmse")), "test_r": f(v.get("test_corr_coef"))}
    S["tuning"] = tun
    # NB2 quantile regression
    qr = {}; d = f"{E}/quantile_regression/results"
    exe = glob.glob(f"{d}/EXECUTED_*.ipynb")
    if exe:
        for src, out in nb_streams(exe[0]):
            m1 = re.search(r"quantile_regression_print\((\w+?)_df", src); m2 = re.search(r"Coverage of 90% prediction interval: ([\d.]+)%", out)
            if m1 and m2: qr.setdefault(canon(m1[1]), {})["coverage90"] = round(float(m2[1]) / 100, 6)
    for x in glob.glob(f"{d}/*.xlsx"):
        m = canon(os.path.basename(x)); v = pd.read_excel(x, "Values", header=None)
        e = qr.setdefault(m, {})
        for _, r in v.iterrows():
            t = " ".join(str(c) for c in r if str(c) != "nan")
            a = re.search(r"Coverage level: 90%.*Average Interval Length: ([\d.]+)", t)
            if a: e["width90"] = f(a[1])
            if "Rows with quantile crossing" in t: e["crossing_rate"] = f(r.iloc[1]) / f(r.iloc[3]) if f(r.iloc[3]) else None
            if t.startswith("Average Continuous Ranked Probability Score"): e["crps"] = f(r.iloc[1])
            if t.startswith("Average Negative Log-Likelihood"): e["nll"] = f(r.iloc[1])
    S["quantile_regression"] = qr
    # NB3 / NB4 probabilistic (NGBoost-style / CARD)
    for key, sub in (("probabilistic", "probabilistic_distribution"), ("card", "probabilistic_distribution_card")):
        res = {}
        for x in glob.glob(f"{E}/{sub}/results/*.xlsx"):
            xl = pd.read_excel(x, sheet_name=None); tag0 = canon(os.path.basename(x)) + (" (TF)" if "_TF" in os.path.basename(x) else "")
            pr = xl.get("Predictions")
            if pr is not None and {"Mean", "StdDev", "Actual"} <= set(pr.columns):
                mu, sd, y = pr["Mean"].to_numpy(float), np.maximum(pr["StdDev"].to_numpy(float), 1e-12), pr["Actual"].to_numpy(float)
                from scipy.stats import norm
                z = (y - mu) / sd; crps = sd * (z * (2 * norm.cdf(z) - 1) + 2 * norm.pdf(z) - 1 / np.sqrt(np.pi))
                e = res.setdefault(tag0, {})
                e.update({"rmse": f(np.sqrt(np.mean((y - mu) ** 2))), "coverage90_gauss": f(np.mean(np.abs(z) <= 1.6448536)),
                          "width90_gauss": f(np.mean(2 * 1.6448536 * sd)), "crps_gauss": f(np.mean(crps))})
            for n, df in xl.items():
                if df.empty: continue
                for pre, cols in (("CRPS_Summary_", {"crps": "Total_CRPS"}), ("PIT_Summary_", {"pit_ks": "KS_Statistic", "pit_var": "Variance_PIT", "diagnosis": "Calibration_Diagnosis"}),
                                  ("Calibration_Summary_", {"mce": "Mean_Calibration_Error (MCE)", "rmsce": "RMS_Calibration_Error (RMSCE)"})):
                    if n.startswith(pre):
                        tag = tag0
                        for k, c in cols.items():
                            if c in df: res.setdefault(tag, {})[k] = df[c].iloc[0] if k == "diagnosis" else f(df[c].iloc[0])
        S[key] = res
    # NB5 MAPIE + PUNCC
    cp = {}
    for x in glob.glob(f"{E}/conformal_predictions_mapie_puncc/results/*.xlsx"):
        m = canon(os.path.basename(x)); xl = pd.read_excel(x, sheet_name=None)
        a = next((v for k, v in xl.items() if k.startswith("all_pred_values")), None)
        if a is not None and not a.empty:
            for s, g in a.groupby("strategy"):
                cp.setdefault(m, {})[f"MAPIE {s}"] = {"coverage": f(g.is_covered.mean()), "width": f(g.width.mean())}
        p = next((v for k, v in xl.items() if k.startswith("conformal_predictions_PUNCC") and k.endswith("_values")), None)
        if p is not None and not p.empty:
            for s in sorted({c.rsplit("_", 1)[0] for c in p.columns}):
                y, lo, hi = p[f"{s}_true"], p[f"{s}_lower"], p[f"{s}_upper"]
                cp.setdefault(m, {})[f"PUNCC {s}"] = {"coverage": f(((y >= lo) & (y <= hi)).mean()), "width": f((hi - lo).mean())}
    S["mapie_puncc"] = cp
    # NB6 NexCP / online CP / COP / Extreme CP / AlphaNet
    nx = {}
    for x in glob.glob(f"{E}/conformal_predictions_nexcp_adaptivecp_mfcs/results/*.xlsx"):
        m = canon(os.path.basename(x))
        for n, df in pd.read_excel(x, sheet_name=None).items():
            if n.endswith("_Summary") and "Strategy" in df and "Coverage" in df:
                for _, r in df.iterrows():
                    note = r.get("Note"); note = None if not isinstance(note, str) else note
                    nx.setdefault(m, {})[r["Strategy"]] = {"coverage": f(r["Coverage"]), "width": f(r["Width"]), "rmse": f(r.get("RMSE")), "note": note}
    S["nexcp_online"] = nx
    rd = f"{E}/conformal_predictions_nexcp_adaptivecp_mfcs/results/README.md"
    S["nb6_readme"] = open(rd).read() if os.path.exists(rd) else ""
    # NB7 ACP
    acp = {}
    for x in glob.glob(f"{E}/conformal_predictions_adaptive_coverage_policies/results/*.xlsx"):
        df = pd.read_excel(x, sheet_name=None).get("acp_final_metrics")
        if df is not None and not df.empty:
            r = df.iloc[0]; acp[canon(os.path.basename(x))] = {k: f(r[k]) for k in ("coverage", "interval_width", "rmse", "best_lambda", "Target_coverage") if k in r}
    S["acp"] = acp
    # NB8 HCM
    h = glob.glob(f"{E}/hyperspherical_confidence_mapping_hcm/results/HCM.xlsx"); S["hcm"] = {}
    if h:
        xl = pd.read_excel(h[0], sheet_name=None); pm = xl.get("Paper_Metrics"); ss = xl.get("Secondary_Summary")
        if pm is not None: S["hcm"] = {k: f(v) for k, v in pm.iloc[0].items() if k != "Model"}
        if ss is not None: S["hcm"].update({k: f(v) for k, v in ss.iloc[0].items() if k != "Model"})
    json.dump(S, open(f"{OUT}/{ds}.json", "w"), indent=1, default=str)
    return S

for ds in sys.argv[3:] or [p.split("/")[-1].replace("_uncertainty_analysis", "") for p in sorted(glob.glob(f"{R}/examples/*_uncertainty_analysis"))]:
    try:
        S = summarise(ds)
        print(f"{ds:20s} train {S['data']['n_train']:5d} test {S['data']['n_test']:4d} | tune {len(S['tuning'])} | QR {len(S['quantile_regression'])} | prob {len(S['probabilistic'])} | card {len(S['card'])} | cp {len(S['mapie_puncc'])} | nb6 {len(S['nexcp_online'])} | acp {len(S['acp'])} | hcm {len(S['hcm'])}")
    except Exception as e:
        print(ds, "ERROR", type(e).__name__, e)
