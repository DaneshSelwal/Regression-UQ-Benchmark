# References used to verify the notebook fixes

Every source the fixes in `CHANGELOG_FIXES.md` were checked against, grouped by notebook.
Kept for the final reference list.

## Hyperparameter tuning (Optuna AutoSampler)
- Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD. https://github.com/optuna/optuna
- OptunaHub AutoSampler: https://hub.optuna.org/samplers/auto_sampler/

## Quantile regression
- Chernozhukov, V., Fernández-Val, I. & Galichon, A. (2010). *Quantile and Probability Curves Without Crossing.* Econometrica 78(3). Used for monotone rearrangement (fix Q2).
- Gneiting, T. & Raftery, A. E. (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation.* JASA 102. Used for CRPS = 2 × the integral of the pinball loss over the quantile levels (fix Q4).
- pytorch-tabnet (`Metric` API for custom early-stopping metrics): https://github.com/dreamquark-ai/tabnet

## Probabilistic distribution
- Hersbach, H. (2000). *Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems.* Weather and Forecasting 15(5). Used for the reliability / resolution / uncertainty decomposition (fix P1).
- properscoring (checked: it has no decomposition): https://github.com/properscoring/properscoring
- Brophy, J. & Lowd, D. (2022). *Instance-Based Uncertainty Estimation for Gradient-Boosted Regression Trees (IBUG).* NeurIPS. https://github.com/jjbrophy47/ibug. Used for fix P2 (`scale` is already a standard deviation).
- Beltran-Velez, N. et al. (2024). *Treeffuser: Probabilistic Predictions via Conditional Diffusions with Gradient-Boosted Trees.* NeurIPS. https://github.com/blei-lab/treeffuser (the fork used is https://github.com/DaneshSelwal/treeffuser). Used for fix T1.

## CARD
- Han, X., Zheng, H. & Zhou, M. (2022). *CARD: Classification and Regression Diffusion Models.* NeurIPS. https://github.com/XzwHan/CARD (`regression/`: diffusion_utils.py, model.py, card_regression.py, configs/uci_*.yml). Used for fix C1.

## Conformal prediction (MAPIE, PUNCC)
- Romano, Y., Patterson, E. & Candès, E. (2019). *Conformalized Quantile Regression.* NeurIPS. Used for fix M1.
- Barber, R. F., Candès, E., Ramdas, A. & Tibshirani, R. J. (2021). *Predictive Inference with the Jackknife+.* Annals of Statistics 49(1).
- Kim, B., Xu, C. & Barber, R. F. (2020). *Predictive Inference Is Free with the Jackknife+-after-Bootstrap.* NeurIPS.
- MAPIE: https://github.com/scikit-learn-contrib/MAPIE
- PUNCC: https://github.com/deel-ai/puncc

## NEXCP / Adaptive CP / MFCS
- Barber, R. F., Candès, E. J., Ramdas, A. & Tibshirani, R. J. (2023). *Conformal Prediction Beyond Exchangeability.* Annals of Statistics 51(2). Copy in `resources/NEXCP_beyond_exchangeability.pdf`. Eq. (10)–(11): weighted split; eq. (12): weighted full; eq. (33): weighted jackknife+ / CV+.
- Gibbs, I. & Candès, E. (2021). *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS. Copy in `resources/Adaptive_Conformal_Inference.pdf` (ACI).
- Gibbs, I. & Candès, E. (2024). *Conformal Inference for Online Prediction with Arbitrary Distribution Shifts.* JMLR 25 (FACI).
- Bhatnagar, A., Wang, H., Xiong, C. & Bai, Y. (2023). *Improved Online Conformal Prediction via Strongly Adaptive Online Learning.* ICML (SF-OGD, SAOCP). Reference code: https://github.com/salesforce/online_conformal
- Hu, Wu, Xia & Zou (2026). *Distribution-informed Online Conformal Prediction* (COP). ICLR. https://arxiv.org/abs/2512.07770. Code: https://github.com/creator-xi/Conformal-Optimistic-Prediction (`core/methods.py`, `COP`).
- Pasche, O. C., Lam, H. & Engelke, S. (2025). *Extreme Conformal Prediction: Reliable Intervals for High-Impact Events.* https://arxiv.org/abs/2505.08578. Code: https://github.com/opasche/ExtremeConformal (CRAN package `ExtremeConformal`, `R/Conformal_EQR.R`: `GPD_quantiles`, `conformalize_GPD_max`).
- Gauthier, E., Bach, F. & Jordan, M. I. (2026). *Adaptive Coverage Policies in Conformal Prediction.* AISTATS, PMLR 300:3196–3204. https://arxiv.org/abs/2510.04318. Code link in the paper: https://github.com/GauthierE/adaptive-coverage-policies (AlphaNet, eq. 2–9).
- Gauthier, E., Bach, F. & Jordan, M. I. (2025). *E-Values Expand the Scope of Conformal Prediction.* https://arxiv.org/abs/2503.13050. Code: https://github.com/GauthierE/evalues-expand-cp
- MFCS: no source found in the repo; kept as written at the user's request.
