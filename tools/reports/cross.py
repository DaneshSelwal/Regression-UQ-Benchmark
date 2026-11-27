"""cross.py <R> ds...: cross-dataset table (markdown) + heatmap figure docs/figures/."""
import sys, json, os
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analysis as A, make_reports as M
R = sys.argv[1]; dss = sys.argv[2:]; rows = []; cov = {}; wid = {}
for ds in dss:
    F_ = M.facts(R, ds); S = F_["S"]; L = F_["L"]; rec = F_["rec"]
    rows.append(f"| [{M.NAMES[ds]}]({A.dsrel(R, ds)}/) | {S['data']['n_train']} / {F_['n']} | {F_['best']} | "
                f"{M.pmd(F_['qr'].coverage.median())} | {rec['label']} | {M.pmd(rec.cov_med)} | {rec.relw_med:.2f} | [PDF]({A.dsrel(R, ds)}/report/paper.pdf) |")
    g = L[L.n >= 3].set_index("label"); cov[M.NAMES[ds]] = g.cov_med; wid[M.NAMES[ds]] = g.relw_med
print("\n".join(rows))
C = pd.DataFrame(cov); W = pd.DataFrame(wid)
order = [l for l in A.league(A.rows(json.load(open(f"summary/{dss[0]}.json")))).query("n>=3").label]
C = C.reindex([o for o in order if o in C.index]); W = W.reindex(C.index)
fig, ax = plt.subplots(figsize=(7.2, 0.27 * len(C) + 1.6))
im = ax.imshow(C.values, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
for i in range(C.shape[0]):
    for j in range(C.shape[1]):
        v = C.values[i, j]
        ax.text(j, i, "n/r" if np.isnan(v) else f"{100*v:.0f}", ha="center", va="center", fontsize=7, color="k")
ax.set_xticks(range(C.shape[1])); ax.set_xticklabels(C.columns, rotation=30, ha="right", fontsize=8)
ax.set_yticks(range(C.shape[0])); ax.set_yticklabels(C.index, fontsize=8)
ax.set_title("Median empirical coverage (%) of each 90% interval method, per dataset", fontsize=10, fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); fig.tight_layout()
os.makedirs(f"{R}/docs/figures", exist_ok=True); fig.savefig(f"{R}/docs/figures/coverage_heatmap.png", dpi=170, bbox_inches="tight"); plt.close(fig)
fig, ax = plt.subplots(figsize=(7.2, 0.27 * len(W) + 1.6)); Wc = W.clip(upper=3)
im = ax.imshow(Wc.values, cmap="viridis_r", vmin=0, vmax=3, aspect="auto")
for i in range(W.shape[0]):
    for j in range(W.shape[1]):
        v = W.values[i, j]
        ax.text(j, i, "n/r" if np.isnan(v) else f"{v:.2f}", ha="center", va="center", fontsize=6.5, color="w" if v > 1.5 else "k")
ax.set_xticks(range(W.shape[1])); ax.set_xticklabels(W.columns, rotation=30, ha="right", fontsize=8)
ax.set_yticks(range(W.shape[0])); ax.set_yticklabels(W.index, fontsize=8)
ax.set_title("Median interval width / std of test target (lower = sharper)", fontsize=10, fontweight="bold")
fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02); fig.tight_layout()
fig.savefig(f"{R}/docs/figures/width_heatmap.png", dpi=170, bbox_inches="tight"); plt.close(fig)
