# NB6 · Weighted (NexCP), online, COP and Extreme conformal prediction

[← all notebooks](../README.md) · previous: [NB5](../05_conformal_mapie_puncc/README.md) · next: [NB7 ACP](../07_adaptive_coverage_policies/README.md)

| | |
|---|---|
| **What** | Conformal methods that do **not** need exchangeable data: NexCP (split, full, CV+, J+aB); online ACI, FACI, SF-OGD, SAOCP, online split / CV+ / J+aB, MFCS; COP; Extreme CP (one-sided) |
| **Guarantee** | NexCP: $`1-\alpha`$ minus a drift term. Online: long-run average miscoverage → $`\alpha`$ for *any* sequence |
| **Reports** | One summary sheet per family (coverage, width, CWC, notes), interval plots, GPD fits |
| **Notebook** | [`Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb`](<Conformal_Predictions(NEXCP, Adaptive CP, mfcs).ipynb>), run as one Kaggle job per base model |

All methods start from **out-of-fold** training residuals (5-fold): each training row is scored by a model that never saw it
(FIX N1).

## 1. NexCP: weight recent points more

```math
\hat q=\mathrm{Quantile}_{1-\alpha}\Bigl(\sum_{i=1}^n\tilde w_i\,\delta_{s_i}+\tilde w_{n+1}\,\delta_{+\infty}\Bigr),\qquad
\tilde w_i=\frac{w_i}{\sum_j w_j+1},\quad w_i=\rho^{\,n+1-i},\ \rho=0.99
```

**Interpretation.** A weighted version of split conformal in which old residuals count less; the mass at $`+\infty`$ plays the role of
the "+1". Barber et al. (2023) show coverage $`\ge 1-\alpha-\sum_i\tilde w_i\,d_{\mathrm{TV}}(Z,Z^i)`$: exact under
exchangeability, and only slightly below under slow drift. **Full** NexCP refits the model with each of 50 candidate labels for
every test point and keeps those that conform (expensive: switched off on sediment load, FIX B16). If a Full interval reaches the
edge of its grid, its width is a lower bound (marked †, not ranked).

## 2. Online conformal: learn from each revealed label

The test set is processed as a stream: interval for point $`t`$ first, then its label is revealed and used.

**ACI** adjusts the level it asks for ($`\gamma=0.005`$):

```math
\alpha_{t+1}=\alpha_t+\gamma\,(\alpha-\mathrm{err}_t),\qquad \mathrm{err}_t=\mathbb 1\{y_t\notin\hat C_t\}
```

After a miss the requested level drops, so the next interval is wider. Summing the update gives
$`\bigl|\tfrac1T\sum_t\mathrm{err}_t-\alpha\bigr|\le\frac{\max(\alpha_1,1-\alpha_1)+\gamma}{\gamma T}`$ for **any** data sequence.

| Method | Idea |
|---|---|
| **FACI** | Runs ACI for 8 step sizes $`\gamma=0.001\cdot2^k`$ and mixes them with exponential weights on their pinball loss |
| **SF-OGD** | Online gradient descent on the radius itself, $`q_{t+1}=q_t-\eta\,\nabla_t/\sqrt{\sum_{s\le t}\nabla_s^2}`$, $`\nabla_t=\alpha-\mathbb 1\{s_t>q_t\}`$ |
| **SAOCP** | SF-OGD experts started at different times, combined by coin betting: low regret on every time window |
| **Online split / MFCS Split** | Fixed radius: conformal quantile of the out-of-fold residuals (identical in the code) |
| **Online CV+ / J+aB** | CV+ / J+aB whose calibration set grows with each revealed test residual |
| **MFCS Full** | Refits the model on training data + revealed test points; radius from that refit's out-of-fold residuals (no published reference; kept as written) |
| **COP** | Online quantile update plus an *optimistic* correction from the empirical distribution of the last $`T_{\text{burnin}}`$ scores: $`\tilde g_t=\hat F_t(\tilde q_t-\eta_tg_t)-(1-\alpha)`$, $`q_{t+1}=\tilde q_{t+1}-0.5\,\eta_t\tilde g_t`$ |

> **Reading online results.** These methods use earlier test labels, which is legitimate when outcomes really arrive one by
> one (monitoring, forecasting). On a static test set the row order matters.

## 3. Extreme CP: a one-sided bound for the tail

A generalised Pareto distribution is fitted to the residuals above high thresholds (0.80–0.99 quantiles):
$`\Pr(s-u>z\mid s>u)\approx(1+\xi z/\sigma)^{-1/\xi}`$, and inverted at the required level; the largest bound over thresholds is
used. The result is $`y\le\hat f(x)+\hat q`$, so it is reported separately from two-sided intervals.

## 4. What to expect

Across the nine datasets the online methods are valid on 86–100% of model × dataset pairs, and FACI, MFCS Full and NexCP J+aB are
the recommended (sharpest valid) method on five of the nine datasets.

**Fixes in v2.0:** N1 (out-of-fold residuals), N2 (NexCP weighted quantile), N3 (real NexCP Full/CV+/J+aB), N4–N6 (online methods
ported from the reference code, ACI added, no look-ahead), N8 (COP step), N9 (Extreme CP per official code), B15/B16 (speed,
on/off switch). **References:** Barber et al. 2023; Gibbs & Candès 2021, 2024; Bhatnagar et al. 2023; Orabona & Pál 2018; Hu et al. 2026; Pasche, Lam & Engelke 2025.
