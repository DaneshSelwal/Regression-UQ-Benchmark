# NB1 · Hyperparameter tuning

[← all notebooks](../README.md) · next: [NB2 Quantile regression](../02_quantile_regression/README.md)

| | |
|---|---|
| **What** | Tune ten regressors with Optuna, choose each model's settings by 5-fold cross-validation **on the training set only**, and save them for every later notebook |
| **Why first** | Every UQ method in NB2–NB7 wraps these tuned models, so they must be fixed before any interval is computed |
| **Output** | `best_params.pkl` / `best_params.json` (settings + CV and test RMSE per model), `test_results.xlsx`, SHAP / LIME / EBM plots |
| **Notebook** | [`Optuna_autosampler.ipynb`](Optuna_autosampler.ipynb) |

## 1. The ten regressors

| Model | Idea in one line |
|---|---|
| Random forest | Average of deep trees grown on bootstrap samples (used in tuning only) |
| Gradient boosting, HistGB | Trees added one at a time, each fitting the residual errors of the ensemble so far |
| XGBoost, LightGBM, CatBoost | Fast, regularised gradient boosting; LightGBM grows leaf-wise, CatBoost uses ordered boosting |
| GPBoost | Gradient boosting combined with a Gaussian-process / random-effects term |
| NGBoost | Boosts the *parameters* of a predictive distribution (here a Gaussian) along the natural gradient |
| PGBM | Boosting whose leaf weights are random variables, giving a mean and a variance per prediction |
| TabNet | Attention-based neural network for tables |

## 2. What is optimised

For a candidate setting $`\theta`$ the objective is the 5-fold cross-validated RMSE on the training data:

```math
\mathrm{CV}(\theta)=\frac15\sum_{k=1}^{5}\sqrt{\frac{1}{|I_k|}\sum_{i\in I_k}\bigl(y_i-\hat f^{(-k)}_\theta(x_i)\bigr)^2}
```

*In words: train on four folds, measure the error on the fifth, rotate, average.* The test set never enters this
number. Optuna's **AutoSampler** chooses the search strategy, and a **pruner** stops trials whose early folds are already
worse than the median (MedianPruner), than a successive-halving bracket (Hyperband) or than the best trial in a Wilcoxon
test (WilcoxonPruner). Each model is tuned with each pruner and with none; the setting with the lowest CV RMSE is kept.

> **Why this matters for uncertainty.** If the test set is used to choose hyperparameters (as in v1.2), the test error is
> optimistic and every coverage computed on it is too. v2.0 FIX 1 removed this.

## 3. Interpretability

The notebook also explains the tuned models with SHAP values, LIME and an explainable boosting machine (EBM). These do not
feed the UQ notebooks.

## 4. Settings and fixes

5 folds (`KFold(shuffle=True, random_state=42)`), RMSE objective, one Optuna study per model × pruner. v2.0 changes: FIX 1
(CV instead of test set), FIX 2 (pruners receive per-fold reports, seeded), FIX 3 (no quantile loss for Gradient Boosting),
K2 (settings saved to `best_params.pkl`), B13/B14 (robustness and speed). Details: [`docs/CHANGELOG_FIXES.md`](../../docs/CHANGELOG_FIXES.md).

**References:** Akiba et al. 2019 (Optuna); Li et al. 2018 (Hyperband); model papers in [`references/REFERENCES.md`](../../references/REFERENCES.md).
