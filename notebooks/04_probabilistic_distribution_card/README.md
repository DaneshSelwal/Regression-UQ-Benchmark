# NB4 · CARD: diffusion-based predictive distributions

[← all notebooks](../README.md) · previous: [NB3](../03_probabilistic_distribution/README.md) · next: [NB5 Conformal](../05_conformal_mapie_puncc/README.md)

| | |
|---|---|
| **What** | A denoising diffusion model generates samples of $`Y\mid x`$, guided by each tuned regressor's point prediction |
| **Guarantee** | None |
| **Reports** | Samples → mean, std, Gaussian 90% interval, CRPS, NLL, PIT, calibration, one workbook per guide model |
| **Notebook** | [`Probabilistic__Distribution(CARD).ipynb`](<Probabilistic__Distribution(CARD).ipynb>) |

## 1. Forward process: noise that drifts towards the guide

CARD (Han et al. 2022) adds Gaussian noise to the true $`y_0`$ over $`T=1000`$ steps while pulling it towards the guide prediction $`\hat f(x)`$:

```math
y_t=\sqrt{\bar\alpha_t}\,y_0+\bigl(1-\sqrt{\bar\alpha_t}\bigr)\hat f(x)+\sqrt{1-\bar\alpha_t}\,\varepsilon,\qquad
\varepsilon\sim\mathcal N(0,1),\quad \bar\alpha_t=\prod_{s\le t}(1-\beta_s)
```

with $`\beta`$ rising linearly from $`10^{-4}`$ to $`0.02`$. At $`t=T`$, $`y_T`$ is pure noise centred on $`\hat f(x)`$.

## 2. Learning to remove the noise

A network $`\varepsilon_\theta(x,y_t,\hat f(x),t)`$ (three time-conditioned layers of 128 units) learns to predict the noise:
$`\min_\theta\ \mathbb E\,\|\varepsilon-\varepsilon_\theta(x,y_t,\hat f(x),t)\|^2`$. Running the process backwards from noise gives one
sample of $`Y\mid x`$; 1000 samples per test point give the predictive distribution.

*In words: the guide says roughly where the answer is; the diffusion model learns how the truth scatters around it.*

## 3. Settings and what to expect

Official settings (5000 epochs, batch 32, lr 10⁻³, EMA of the weights with warm-up). Each of the nine tuned models is used as
the guide once. CARD intervals are **sharp but usually undercover**: across the nine datasets the median coverage reaches 90%
only on sediment load.

**Fixes in v2.0:** C1 (rewritten to follow the official CARD code; v1.2 returned the guide plus unit noise), C2 (EMA warm-up),
B2 (output file). **References:** Han, Zheng & Zhou 2022; Ho, Jain & Abbeel 2020.
