# NB8 · Hyperspherical confidence mapping (HCM)

[← all notebooks](../README.md) · previous: [NB7](../07_adaptive_coverage_policies/README.md)

| | |
|---|---|
| **What** | One neural network predicts the target and its uncertainty in a single forward pass, without sampling |
| **Guarantee** | None; a scale factor is calibrated on a validation split |
| **Reports** | Coverage at 1/2/3σ, 95% interval coverage and width, RMSE, MAE, CRPS, NLL, regression ECE, Spearman correlation between absolute error and σ̂ |
| **Notebook** | [`Hyperspherical_Confidence_Mapping(HCM).ipynb`](<Hyperspherical_Confidence_Mapping(HCM).ipynb>) |

## 1. Lift, split, predict

The scalar target is lifted to $`\mathbf y=(y,y)\in\mathbb R^2`$ and split into a magnitude and a unit direction,
$`R=\|\mathbf y\|`$, $`\mathbf d=\mathbf y/R`$. A network (3 hidden layers of 20 LeakyReLU units) predicts $`\hat R`$ and $`\hat{\mathbf d}`$ with

```math
\mathcal L=\mathrm{MSE}\bigl(\hat R\,\mathbf d,\ \mathbf y\bigr)+\mathrm{MSE}\bigl(R\,\hat{\mathbf d},\ \mathbf y\bigr)
```

## 2. Uncertainty from geometry

A true direction lies on the unit circle. When the network is unsure, its predicted direction falls inside it:

```math
\hat\sigma_{\text{raw}}(x)=\sqrt{\bigl|\,\|\hat{\mathbf d}(x)\|^2-1\bigr|}\ \,|\hat R(x)|
```

One scale factor is fitted on a validation split so that $`\pm1,2,3\hat\sigma`$ cover 68/95/99.7%.

## 3. What to expect

Across the nine datasets the 95% interval covers 80–100%, but the predicted spread is negatively correlated with the actual error,
or not at all, on six of the nine (Spearman between −0.57 and 0.0): on small tabular problems this single network does not reliably
know when it is wrong.

**Fixes in v2.0:** K1, B6 (silent fallback to synthetic data removed), B7. **Reference:** Choi et al. 2026 (ICLR).
