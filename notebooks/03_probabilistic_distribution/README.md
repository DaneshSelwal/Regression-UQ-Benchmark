# NB3 · Predictive distributions (NGBoost, PGBM, IBUG, Treeffuser)

[← all notebooks](../README.md) · previous: [NB2](../02_quantile_regression/README.md) · next: [NB4 CARD](../04_probabilistic_distribution_card/README.md)

| | |
|---|---|
| **What** | Predict a whole distribution $`\hat F_x`$ for each test point, not only an interval |
| **Guarantee** | None: the distribution is right only if the model describes the noise correctly |
| **Reports** | μ̂, σ̂, Gaussian 90% interval μ̂ ± 1.645σ̂, NLL, CRPS with Hersbach decomposition, PIT histogram, calibration curve |
| **Notebook** | [`Probabilistic__Distribution.ipynb`](Probabilistic__Distribution.ipynb) |

## 1. The four methods

| Method | How it gets a spread σ̂(x) |
|---|---|
| **NGBoost** | Boosts the Gaussian's $`(\mu,\log\sigma)`$ along the *natural* gradient of the negative log-likelihood |
| **PGBM** | Each tree leaf weight is a random variable; means and variances of the leaves add up over the trees |
| **IBUG** (on XGBoost, LightGBM) | Finds the $`k`$ training points that share the most leaves with the test point and uses the spread of their targets |
| **Treeffuser** | A conditional diffusion model whose score is learned by gradient-boosted trees; samples give a non-parametric $`\hat F_x`$ |

## 2. Scoring a distribution

**CRPS.** For a Gaussian forecast the CRPS has a closed form:

```math
\mathrm{CRPS}\bigl(\mathcal N(\mu,\sigma^2),y\bigr)=\sigma\Bigl[z\bigl(2\Phi(z)-1\bigr)+2\varphi(z)-\tfrac{1}{\sqrt\pi}\Bigr],\qquad z=\frac{y-\mu}{\sigma}
```

**Interpretation.** A sharp distribution centred on the truth scores low; a wide one, or one centred in the wrong place, scores high.
CRPS is *proper*: in expectation the true distribution gets the best score. The **Hersbach decomposition** splits it into
reliability (calibration error), resolution and uncertainty (FIX P1).

**PIT.** If $`\hat F`$ is right, $`\mathrm{PIT}=\hat F_X(Y)`$ is uniform on $`[0,1]`$. A **U-shaped** histogram means σ̂ is too
small (*under-dispersed, over-confident*); a **hump** means σ̂ is too large (*over-dispersed*).

## 3. What to expect

A good CRPS does not imply a valid interval: CRPS rewards the centre, coverage tests the tails. Across the nine datasets the
native NGBoost and PGBM Gaussians are usually **over-confident** (PGBM covers 2–42% on eight of nine), IBUG is close to or
above 90%, and Treeffuser covers almost everything with very wide intervals.

**Fixes in v2.0:** P1 (Hersbach CRPS decomposition), P2 (IBUG σ was square-rooted twice), T1 (Treeffuser's Gradient Boosting
score model uses squared error), P3 (plot used the wrong model). **References:** Duan et al. 2020; Sprangers et al. 2021;
Brophy & Lowd 2022; Beltran-Velez et al. 2024; Gneiting & Raftery 2007; Hersbach 2000.
