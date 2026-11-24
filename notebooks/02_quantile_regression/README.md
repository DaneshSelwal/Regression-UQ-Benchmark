# NB2 · Quantile regression

[← all notebooks](../README.md) · previous: [NB1](../01_hyperparameter_tuning/README.md) · next: [NB3 Distributional](../03_probabilistic_distribution/README.md)

| | |
|---|---|
| **What** | Fit one model per quantile level (13 levels, 0.05 … 0.95) for each tuned regressor; the 90% interval is $`[\hat q_{0.05}(x),\hat q_{0.95}(x)]`$ |
| **Guarantee** | None in finite samples: the interval is right only if both quantile models are right |
| **Reports** | Coverage, width, pinball loss, CRPS from quantiles, quantile-crossing rate, calibration |
| **Notebook** | [`Quantile_Regression.ipynb`](Quantile_Regression.ipynb) |

## 1. The pinball loss

```math
\rho_\tau(r)=r\,\bigl(\tau-\mathbb 1\{r<0\}\bigr)=\begin{cases}\tau\,r & r\ge 0\\ (\tau-1)\,r & r<0\end{cases},\qquad r=y-q
```

**Interpretation.** A prediction below the truth costs $`\tau`$ per unit, a prediction above it costs $`1-\tau`$. For
$`\tau=0.95`$ under-predicting is 19 times as expensive as over-predicting, so the model is pushed up until only 5% of the
points lie above it. Formally, the derivative of $`\mathbb E\,\rho_\tau(Y-q)`$ in $`q`$ is $`\Pr(Y<q)-\tau`$, which is zero exactly at
the $`\tau`$-quantile.

> **Worked example.** True value 10, prediction 12, $`\tau=0.95`$: $`r=-2`$, loss $`(0.95-1)(-2)=0.1`$. Prediction 8:
> $`r=2`$, loss $`0.95\cdot2=1.9`$. Missing low is 19× worse.

## 2. Quantile crossing

Each level is a separate model, so for one point the 13 predictions can be out of order, e.g.
$`\hat q_{0.05}(x)>\hat q_{0.10}(x)`$. The notebook counts the share of test points with at least one inversion
(*crossing rate*) and then **sorts** each point's 13 values (monotone rearrangement, Chernozhukov et al. 2010), which never
increases the pinball loss.

## 3. CRPS from quantiles

The CRPS of a predictive distribution $`F`$ equals twice the pinball loss averaged over all levels:

```math
\mathrm{CRPS}(F,y)=\int_{-\infty}^{\infty}\bigl(F(z)-\mathbb 1\{z\ge y\}\bigr)^2\,dz=2\int_0^1\rho_\tau\bigl(y-F^{-1}(\tau)\bigr)\,d\tau
\;\approx\;2\sum_{j}w_j\,\rho_{\tau_j}\bigl(y-\hat q_{\tau_j}\bigr)
```

with $`w_j`$ the share of $`[0,1]`$ that level $`j`$ represents (FIX Q4). CRPS is in the units of the target; lower is better.

## 4. What to expect

Separately fitted extreme quantiles are learned from very few points (5% of a few hundred), so the raw interval typically
**undercovers**. Across the nine datasets its median coverage is 58–90%, and the quantiles cross on most test points.
Conformalising the quantile models (CQR, [NB5](../05_conformal_mapie_puncc/README.md)) fixes the coverage.

**Fixes in v2.0:** Q1 (XGBoost used LightGBM's settings), Q2 (crossing measured and removed), Q3 (TabNet early stopping on a
validation split with its own pinball loss), Q4 (CRPS level weights). **References:** Koenker & Bassett 1978; Chernozhukov et al. 2010; Gneiting & Raftery 2007.
