# NB5 · Conformal prediction with MAPIE and PUNCC

[← all notebooks](../README.md) · previous: [NB4](../04_probabilistic_distribution_card/README.md) · next: [NB6 NexCP and online](../06_conformal_nexcp_online/README.md)

| | |
|---|---|
| **What** | Turn each tuned regressor into a 90% interval with a **finite-sample guarantee**: split conformal, CV+ and J+aB (MAPIE); split conformal, CV+ and CQR (PUNCC) |
| **Guarantee** | $`\Pr(Y\in\hat C(X))\ge 1-\alpha`$ (split, CQR) or $`\ge1-2\alpha`$ (CV+, J+aB) if the data are exchangeable, whatever the model |
| **Reports** | Intervals, coverage, width, conditional coverage by target bins, one workbook per model |
| **Notebook** | [`Conformal Predictions(MAPIE,PUNCC).ipynb`](<Conformal Predictions(MAPIE,PUNCC).ipynb>) |

*__Box 1.__ Split conformal prediction in three steps, on synthetic data.*

## 1. Split conformal: the foundation

**Step 1: score.** Fit $`\hat f`$ on one half of the training data. On the other half (the calibration set, $`m`$ points)
measure how wrong it is: $`s_i=|y_i-\hat f(x_i)|`$.

**Step 2: cut-off.** Sort the scores and take the one at position $`k=\lceil(m+1)(1-\alpha)\rceil`$:

```math
\hat q=s_{(k)},\qquad k=\bigl\lceil (m+1)(1-\alpha)\bigr\rceil
```

**Step 3: interval.** $`\hat C(x)=[\hat f(x)-\hat q,\ \hat f(x)+\hat q]`$.

**Why it works.** $`\hat f`$ never saw the calibration points or the test point, so the $`m+1`$ scores are exchangeable and the
test score is equally likely to have any rank among them. It falls at or below the $`k`$-th smallest with probability
$`k/(m+1)\ge1-\alpha`$. The "+1" counts the test point itself. Nothing is assumed about the model: a bad model gets a valid but
wide interval.

> **Worked example.** $`m=19`$ calibration scores, $`\alpha=0.1`$: $`k=\lceil 20\times0.9\rceil=18`$, so $`\hat q`$ is the 18th
> smallest score. If it is 4.2 and $`\hat f(x)=30`$, the interval is [25.8, 34.2].

## 2. CV+ and jackknife+-after-bootstrap (J+aB)

Split conformal spends half the data on calibration. **CV+** fits $`K=5`$ models, each without one fold, scores every
training point with the model that did not see it ($`R_i`$), and combines:

```math
\hat C(x)=\Bigl[\ \hat q^{-}_{\alpha}\bigl\{\hat f^{(-k(i))}(x)-R_i\bigr\},\ \ \hat q^{+}_{\alpha}\bigl\{\hat f^{(-k(i))}(x)+R_i\bigr\}\ \Bigr]
```

($`\hat q^-`$: the $`\lfloor\alpha(n+1)\rfloor`$-th smallest; $`\hat q^+`$: the $`\lceil(1-\alpha)(n+1)\rceil`$-th smallest). **J+aB**
does the same with 50 bootstrap models and out-of-bag residuals. The guarantee on paper is $`1-2\alpha`$; in practice coverage is
close to $`1-\alpha`$ and the intervals are narrower because every point is used to fit *and* to calibrate.

## 3. CQR: conformalised quantile regression

Start from quantile models $`\hat q_{0.05},\hat q_{0.95}`$ (their band already widens where the data are noisy) and correct it:

```math
s_i=\max\bigl\{\hat q_{0.05}(x_i)-y_i,\ y_i-\hat q_{0.95}(x_i)\bigr\},\qquad \hat C(x)=\bigl[\hat q_{0.05}(x)-\hat q,\ \hat q_{0.95}(x)+\hat q\bigr]
```

The score is negative inside the band, so $`\hat q`$ can shrink a band that is too wide as well as widen one that is too narrow.

## 4. What to expect

Across the nine datasets these methods are valid on 91–98% of model × dataset pairs. J+aB is usually the sharpest; CQR is
valid but wide here, because the conformal step has to repair quantile models fitted on a few hundred points.

**Fixes in v2.0:** M1 (CQR now uses real quantile models), M2 (interval array shape), M3 (split and J+aB are what their names
say), M4 (one α = 0.1), M5/M6 (tuned settings always used). **References:** Vovk et al. 2005; Lei et al. 2018; Barber et al.
2021; Kim, Xu & Barber 2020; Romano, Patterson & Candès 2019; Angelopoulos & Bates 2023.
