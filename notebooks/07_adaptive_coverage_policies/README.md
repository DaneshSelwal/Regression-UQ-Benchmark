# NB7 · Adaptive coverage policies (ACP)

[← all notebooks](../README.md) · previous: [NB6](../06_conformal_nexcp_online/README.md) · next: [NB8 HCM](../08_hyperspherical_confidence_mapping/README.md)

| | |
|---|---|
| **What** | A small network (AlphaNet) chooses the coverage level $`1-\tilde\alpha(x)`$ for each test point; conformal e-values make the set valid at that learned level |
| **Guarantee** | $`\mathbb E\bigl[\mathbb 1\{Y\notin\hat C\}/\tilde\alpha(X)\bigr]\le1`$: compare coverage with the *learned* target, not with 90% |
| **Reports** | Learned α̃, target coverage, realised coverage, width, λ comparison, training curves |
| **Notebook** | [`Adaptive_Coverage_Policies(ACP).ipynb`](<Adaptive_Coverage_Policies(ACP).ipynb>) |

## 1. The set for a given level

With calibration scores $`s_1,\dots,s_n`$ a candidate $`y`$ is excluded when its conformal e-value is large:

```math
e(y)=\frac{(n+1)\,s(x,y)}{\sum_{i}s_i+s(x,y)}\ \ge\ \frac{1}{\tilde\alpha(x)}
\quad\Longleftrightarrow\quad s(x,y)\ \ge\ \frac{\sum_i s_i}{\tilde\alpha(x)(n+1)-1}
```

so the set is $`\hat f(x)\pm\sum_is_i/(\tilde\alpha(n+1)-1)`$. A small $`\tilde\alpha`$ (high coverage) gives a wide set.

## 2. Learning the level

AlphaNet (32 ReLU units, sigmoid output) maps the scaled sum of leave-one-out scores to $`\tilde\alpha`$ and is trained on
$`\text{width}+\lambda\,\tilde\alpha`$: λ is the price of a miss. λ = 50 is reported; λ = 10, 20, 50 are compared in a separate sheet.

> **Saturation.** If the policy pushes $`\tilde\alpha`$ to its upper limit (0.999) for every point, the target coverage is
> 0.1%, every set has the same width, and the learned level means nothing. This happens on sediment load and streamflow (all
> models) and on EWQI (4 of 9 models); those models are flagged.

**Fixes in v2.0:** A1 (λ is no longer chosen by comparing losses across λ, which always picked λ = 10), A2 (scaled input,
initial α̃ ≈ 1 as in the official code), A3 (enough optimizer steps for λ to matter). **References:** Gauthier, Bach & Jordan
2026 (AISTATS); Gauthier, Bach & Jordan 2025 (e-values); Vovk & Wang 2021.
