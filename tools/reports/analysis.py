"""analysis.py: derived tables for one dataset summary (coverage/width rows, method league, recommendation)."""
import math, re, glob, os
import numpy as np, pandas as pd

TARGET = 0.90

def dsdir(R, ds):
    """Folder of one dataset study: concrete (main case study) lives in notebooks/, the others in examples/."""
    for base in ("notebooks", "examples"):
        d = f"{R}/{base}/{ds}_uncertainty_analysis"
        if os.path.isdir(d): return d
    return f"{R}/examples/{ds}_uncertainty_analysis"

def all_datasets(R):
    return sorted(os.path.basename(p).replace("_uncertainty_analysis", "")
                  for b in ("notebooks", "examples") for p in glob.glob(f"{R}/{b}/*_uncertainty_analysis"))

def dsrel(R, ds):
    return os.path.relpath(dsdir(R, ds), R)
MODEL_ORDER = ["CatBoost", "XGBoost", "LightGBM", "GPBoost", "GradientBoosting", "HistGB", "NGBoost", "PGBM", "TabNet", "RandomForest"]
FAMILY_ORDER = ["Quantile regression", "Distributional", "CARD", "Conformal (MAPIE/PUNCC)", "Weighted conformal (NexCP)",
                "Online conformal", "COP"]
NICE = {"MAPIE SCP": "Split CP (MAPIE)", "MAPIE CV+": "CV+ (MAPIE)", "MAPIE J+aB": "J+aB (MAPIE)", "PUNCC SplitCP": "Split CP (PUNCC)",
        "PUNCC CVPlus": "CV+ (PUNCC)", "PUNCC CQR": "CQR (PUNCC)", "NexCP Jackknife+ aB": "NexCP J+aB", "Online Jackknife+ aB": "Online J+aB",
        "EXTREME CP": "Extreme CP"}

def tol(n):
    """One-sided 95% sampling tolerance of an empirical coverage around 0.9 on n test points."""
    return 1.645 * math.sqrt(TARGET * (1 - TARGET) / n)

def family_of(method):
    if method.startswith(("MAPIE", "PUNCC")): return "Conformal (MAPIE/PUNCC)"
    if method.startswith("NexCP"): return "Weighted conformal (NexCP)"
    if method == "COP": return "COP"
    return "Online conformal"

def rows(S):
    """Every (family, method, model) with an interval at the 90% target."""
    out = []
    for m, v in S["quantile_regression"].items():
        if v.get("coverage90") is not None: out.append(("Quantile regression", "Quantile regression (raw)", m, v["coverage90"], v.get("width90")))
    for tag, v in S["probabilistic"].items():
        base = tag.replace(" (TF)", "")
        meth = "Treeffuser" if "(TF)" in tag else ("NGBoost (native)" if base == "NGBoost" else "PGBM (native)" if base == "PGBM" else "IBUG")
        if v.get("coverage90_gauss") is not None: out.append(("Distributional", meth, base, v["coverage90_gauss"], v["width90_gauss"]))
    for m, v in S["card"].items():
        if v.get("coverage90_gauss") is not None: out.append(("CARD", "CARD", m, v["coverage90_gauss"], v["width90_gauss"]))
    for m, d in S["mapie_puncc"].items():
        for meth, v in d.items():
            if v["coverage"] is not None: out.append(("Conformal (MAPIE/PUNCC)", meth, m, v["coverage"], v["width"]))
    for m, d in S["nexcp_online"].items():
        for meth, v in d.items():
            if meth.startswith("AlphaNet") or meth == "EXTREME CP" or v["coverage"] is None: continue   # one-sided / learned level: own tables
            out.append((family_of(meth), meth, m, v["coverage"], v["width"], "edge" in (v.get("note") or "")))
    out = [r if len(r) == 6 else r + (False,) for r in out]
    df = pd.DataFrame(out, columns=["family", "method", "model", "coverage", "width", "truncated"])
    sd = S["data"]["target_stats"]["std"] or 1.0
    df["rel_width"] = df["width"] / sd
    df["valid"] = df["coverage"] >= TARGET - tol(S["data"]["n_test"])
    df["label"] = df["method"].map(lambda x: NICE.get(x, x))
    return df

def league(df):
    g = df.groupby(["family", "label"]).agg(n=("model", "count"), cov_med=("coverage", "median"), cov_min=("coverage", "min"),
                                             cov_max=("coverage", "max"), relw_med=("rel_width", "median"), w_med=("width", "median"),
                                             share_valid=("valid", "mean"), truncated=("truncated", "any")).reset_index()
    g["fam_rank"] = g["family"].map({f: i for i, f in enumerate(FAMILY_ORDER)})
    return g.sort_values(["fam_rank", "relw_med"]).reset_index(drop=True)

def recommend(df, L):
    """Narrowest method (median over models) whose coverage is valid for >= 75% of the models it ran on."""
    ok = L[(L.share_valid >= 0.75) & (L.n >= 3) & ~L.truncated].sort_values("relw_med")   # grid-truncated widths are lower bounds
    rec = ok.iloc[0] if len(ok) else None
    v = df[df.valid & ~df.truncated].sort_values("width")
    best_combo = v.iloc[0] if len(v) else None
    return rec, best_combo, ok

def best_model(S):
    t = {m: v for m, v in S["tuning"].items() if v.get("cv_rmse") is not None}
    return min(t, key=lambda m: t[m]["cv_rmse"])

def run_names(R, ds):
    """Kaggle run code names per phase, read from the results READMEs."""
    E = dsdir(R, ds); out = {}
    for rd in sorted(glob.glob(f"{E}/*/results/README.md")):
        ph = rd.split("/")[-3]; t = open(rd).read()
        codes = list(dict.fromkeys(re.findall(r"REG\d{2}_\d{2}_\d{4}_\d{2}_\d{2}", t)))
        out[ph] = codes
    return out

def extreme(S):
    """Extreme CP: one-sided upper bound (y <= prediction + q); 'width' is the distance to that bound."""
    return {m: d["EXTREME CP"] for m, d in S["nexcp_online"].items() if "EXTREME CP" in d}

def alphanet(S):
    return {m: d[k] for m, d in S["nexcp_online"].items() for k in d if k.startswith("AlphaNet")}

def canon_file(path):
    n = os.path.basename(path).replace(".xlsx", "")
    for k, v in [("TabNet","TabNet"),("NGB","NGBoost"),("PGBM","PGBM"),("GPBoost","GPBoost"),("CatBoost","CatBoost"),("HGB","HistGB"),
                 ("HistGradient","HistGB"),("LightGBM","LightGBM"),("LGBM","LightGBM"),("XGB","XGBoost"),("Boosting","GradientBoosting")]:
        if k.lower() in n.lower(): return v
    return n
