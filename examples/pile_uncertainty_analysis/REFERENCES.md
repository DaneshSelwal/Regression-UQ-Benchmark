# References

Every method, model, library and paper used across the 8 pile notebooks. Kept for the final reference list.
Entries marked **[fix]** are the sources the changes in `CHANGELOG_FIXES.md` were checked against.

## 1. Base regression models (all notebooks)
- Breiman, L. (2001). *Random Forests.* Machine Learning 45(1), 5–32.
- Friedman, J. H. (2001). *Greedy Function Approximation: A Gradient Boosting Machine.* Annals of Statistics 29(5). (GradientBoostingRegressor; also introduces partial dependence plots)
- Chen, T. & Guestrin, C. (2016). *XGBoost: A Scalable Tree Boosting System.* KDD. https://github.com/dmlc/xgboost
- Ke, G. et al. (2017). *LightGBM: A Highly Efficient Gradient Boosting Decision Tree.* NeurIPS. https://github.com/microsoft/LightGBM
- Prokhorenkova, L. et al. (2018). *CatBoost: Unbiased Boosting with Categorical Features.* NeurIPS. https://github.com/catboost/catboost
- Sigrist, F. (2022). *Gaussian Process Boosting.* JMLR 23. https://github.com/fabsig/GPBoost
- Duan, T. et al. (2020). *NGBoost: Natural Gradient Boosting for Probabilistic Prediction.* ICML. https://github.com/stanfordmlgroup/ngboost
- Sprangers, O., Schelter, S. & de Rijke, M. (2021). *Probabilistic Gradient Boosting Machines for Large-Scale Probabilistic Regression.* KDD. https://github.com/elephaint/pgbm (also provides the `pgbm.sklearn` HistGradientBoostingRegressor)
- Arik, S. Ö. & Pfister, T. (2021). *TabNet: Attentive Interpretable Tabular Learning.* AAAI. https://github.com/dreamquark-ai/tabnet (pytorch-tabnet; its `Metric` API is used for early stopping in fix Q3) **[fix]**
- Drucker, H. et al. (1997). *Support Vector Regression Machines.* NeurIPS. (sklearn SVR)
- Pedregosa, F. et al. (2011). *Scikit-learn: Machine Learning in Python.* JMLR 12. https://scikit-learn.org

## 2. Hyperparameter tuning (Optuna AutoSampler notebook)
- Akiba, T., Sano, S., Yanase, T., Ohta, T. & Koyama, M. (2019). *Optuna: A Next-generation Hyperparameter Optimization Framework.* KDD. https://github.com/optuna/optuna
- OptunaHub AutoSampler: https://hub.optuna.org/samplers/auto_sampler/
- Bergstra, J., Bardenet, R., Bengio, Y. & Kégl, B. (2011). *Algorithms for Hyper-Parameter Optimization.* NeurIPS. (TPE sampler)
- Hansen, N. (2016). *The CMA Evolution Strategy: A Tutorial.* arXiv:1604.00772. (CMA-ES sampler)
- Deb, K. et al. (2002). *A Fast and Elitist Multiobjective Genetic Algorithm: NSGA-II.* IEEE TEC 6(2).
- Deb, K. & Jain, H. (2014). *An Evolutionary Many-Objective Optimization Algorithm Using Reference-Point-Based Nondominated Sorting (NSGA-III).* IEEE TEC 18(4).
- Pruners:
  - Li, L. et al. (2018). *Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization.* JMLR 18. (HyperbandPruner)
  - Jamieson, K. & Talwalkar, A. (2016). *Non-stochastic Best Arm Identification and Hyperparameter Optimization.* AISTATS. (successive halving: SuccessiveHalvingPruner, HalvingGridSearchCV)
  - Li, L. et al. (2020). *A System for Massively Parallel Hyperparameter Tuning (ASHA).* MLSys.
  - Median, Percentile, Patient, Threshold, Wilcoxon and Nop pruners: Optuna documentation, https://optuna.readthedocs.io/en/stable/reference/pruners.html
- Bayesian Optimization package: Nogueira, F. (2014–). https://github.com/bayesian-optimization/BayesianOptimization
- Cross-validation for tuning without touching the test set (fixes 1–3): Hastie, T., Tibshirani, R. & Friedman, J. (2009). *The Elements of Statistical Learning*, 2nd ed., §7.10. **[fix]**

## 3. Interpretability (tuning notebook)
- Lundberg, S. M. & Lee, S.-I. (2017). *A Unified Approach to Interpreting Model Predictions (SHAP).* NeurIPS. (ShapKernel)
- Ribeiro, M. T., Singh, S. & Guestrin, C. (2016). *"Why Should I Trust You?": Explaining the Predictions of Any Classifier (LIME).* KDD. https://github.com/marcotcr/lime
- Nori, H., Jenkins, S., Koch, P. & Caruana, R. (2019). *InterpretML: A Unified Framework for Machine Learning Interpretability.* arXiv:1909.09223. https://github.com/interpretml/interpret (ExplainableBoostingRegressor, LimeTabular, ShapKernel)
- Lou, Y., Caruana, R., Gehrke, J. & Hooker, G. (2013). *Accurate Intelligible Models with Pairwise Interactions.* KDD. (EBM / GA2M)
- Goldstein, A., Kapelner, A., Bleich, J. & Pitkin, E. (2015). *Peeking Inside the Black Box: Individual Conditional Expectation (ICE) plots.* JCGS 24(1).

## 4. Quantile regression
- Koenker, R. & Bassett, G. (1978). *Regression Quantiles.* Econometrica 46(1). (pinball / quantile loss)
- Chernozhukov, V., Fernández-Val, I. & Galichon, A. (2010). *Quantile and Probability Curves Without Crossing.* Econometrica 78(3). (monotone rearrangement, fix Q2) **[fix]**
- Gneiting, T. & Raftery, A. E. (2007). *Strictly Proper Scoring Rules, Prediction, and Estimation.* JASA 102. (CRPS = 2 × the integral of the pinball loss over the levels, fix Q4; interval score) **[fix]**

## 5. Probabilistic distribution notebooks (NGBoost / PGBM / IBUG / Treeffuser / CARD)
- Hersbach, H. (2000). *Decomposition of the Continuous Ranked Probability Score for Ensemble Prediction Systems.* Weather and Forecasting 15(5). (reliability / resolution / uncertainty, fix P1) **[fix]**
- Matheson, J. E. & Winkler, R. L. (1976). *Scoring Rules for Continuous Probability Distributions.* Management Science 22(10). (CRPS)
- properscoring (CRPS; checked: it has no decomposition): https://github.com/properscoring/properscoring **[fix]**
- Brophy, J. & Lowd, D. (2022). *Instance-Based Uncertainty Estimation for Gradient-Boosted Regression Trees (IBUG).* NeurIPS. https://github.com/jjbrophy47/ibug (fix P2) **[fix]**
- Beltran-Velez, N. et al. (2024). *Treeffuser: Probabilistic Predictions via Conditional Diffusions with Gradient-Boosted Trees.* NeurIPS. https://github.com/blei-lab/treeffuser (fork used: https://github.com/DaneshSelwal/treeffuser; fix T1) **[fix]**
- Han, X., Zheng, H. & Zhou, M. (2022). *CARD: Classification and Regression Diffusion Models.* NeurIPS. https://github.com/XzwHan/CARD (fix C1) **[fix]**
- Ho, J., Jain, A. & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models.* NeurIPS. (diffusion background for CARD / Treeffuser)
- Kuleshov, V., Fenner, N. & Ermon, S. (2018). *Accurate Uncertainties for Deep Learning Using Calibrated Regression.* ICML. (calibration curves, PIT)
- Gneiting, T., Balabdaoui, F. & Raftery, A. E. (2007). *Probabilistic Forecasts, Calibration and Sharpness.* JRSS-B 69(2). (sharpness, PIT)

## 6. Conformal prediction: general and MAPIE / PUNCC notebook
- Vovk, V., Gammerman, A. & Shafer, G. (2005). *Algorithmic Learning in a Random World.* Springer.
- Lei, J., G'Sell, M., Rinaldo, A., Tibshirani, R. J. & Wasserman, L. (2018). *Distribution-Free Predictive Inference for Regression.* JASA 113. (split conformal)
- Angelopoulos, A. N. & Bates, S. (2023). *Conformal Prediction: A Gentle Introduction.* Foundations and Trends in ML. Copy in `resources/Gentle_Intro_to_Conformal_Prediction.pdf`.
- Romano, Y., Patterson, E. & Candès, E. (2019). *Conformalized Quantile Regression.* NeurIPS. (fix M1) **[fix]**
- Barber, R. F., Candès, E. J., Ramdas, A. & Tibshirani, R. J. (2021). *Predictive Inference with the Jackknife+.* Annals of Statistics 49(1). (CV+, Jackknife+, fix M3) **[fix]**
- Kim, B., Xu, C. & Barber, R. F. (2020). *Predictive Inference Is Free with the Jackknife+-after-Bootstrap.* NeurIPS. (fixes M3, N3) **[fix]**
- Taquet, V. et al. (2022). *MAPIE: an Open-Source Library for Distribution-Free Uncertainty Quantification.* arXiv:2207.12274. https://github.com/scikit-learn-contrib/MAPIE **[fix]**
- Mendil, M., Mossina, L. & Vigouroux, D. (2023). *PUNCC: a Python Library for Predictive Uncertainty Calibration and Conformalization.* COPA (PMLR 204). https://github.com/deel-ai/puncc **[fix]**
- Khosravi, A., Nahavandi, S., Creighton, D. & Atiya, A. F. (2011). *Lower Upper Bound Estimation Method for Construction of Neural Network-Based Prediction Intervals.* IEEE TNN 22(3). (PICP, MPIW, CWC)

## 7. NEXCP / Adaptive CP / MFCS notebook
- Barber, R. F., Candès, E. J., Ramdas, A. & Tibshirani, R. J. (2023). *Conformal Prediction Beyond Exchangeability.* Annals of Statistics 51(2). Copy in `resources/NEXCP_beyond_exchangeability.pdf`. Eq. (10)–(11): weighted split; eq. (12): weighted full; eq. (33): weighted jackknife+ / CV+ (fixes N1–N3). **[fix]**
- Gibbs, I. & Candès, E. (2021). *Adaptive Conformal Inference Under Distribution Shift.* NeurIPS. Copy in `resources/Adaptive_Conformal_Inference.pdf` (ACI, fix N5). **[fix]**
- Gibbs, I. & Candès, E. (2024). *Conformal Inference for Online Prediction with Arbitrary Distribution Shifts.* JMLR 25 (FACI, fix N5). **[fix]**
- Bhatnagar, A., Wang, H., Xiong, C. & Bai, Y. (2023). *Improved Online Conformal Prediction via Strongly Adaptive Online Learning.* ICML (SF-OGD, SAOCP; fixes N4, N6). Reference code: https://github.com/salesforce/online_conformal **[fix]**
- Orabona, F. & Pál, D. (2018). *Scale-Free Online Learning.* Theoretical Computer Science 716. (scale-free OGD background)
- Hu, Wu, Xia & Zou (2026). *Distribution-informed Online Conformal Prediction* (COP). ICLR. https://arxiv.org/abs/2512.07770. Code: https://github.com/creator-xi/Conformal-Optimistic-Prediction (`core/methods.py`, `COP`; fix N8). **[fix]**
- Pasche, O. C., Lam, H. & Engelke, S. (2025). *Extreme Conformal Prediction: Reliable Intervals for High-Impact Events.* https://arxiv.org/abs/2505.08578. Code: https://github.com/opasche/ExtremeConformal (CRAN `ExtremeConformal`, `R/Conformal_EQR.R`: `GPD_quantiles`, `conformalize_GPD_max`; fix N9). **[fix]**
- Pickands, J. (1975). *Statistical Inference Using Extreme Order Statistics.* Annals of Statistics 3(1). (Generalized Pareto tail; scipy `genpareto`)
- MFCS: no source found in the repo; kept as written at the user's request.

## 8. Adaptive Coverage Policies (AlphaNet) notebook and AlphaNet in the NEXCP notebook
- Gauthier, E., Bach, F. & Jordan, M. I. (2026). *Adaptive Coverage Policies in Conformal Prediction.* AISTATS, PMLR 300:3196–3204. https://arxiv.org/abs/2510.04318. Official code (verified): https://github.com/GauthierE/adaptive-coverage-policies (`regression/regression.ipynb`; `classification/selection-lambda.ipynb` picks λ by bisection to a target mean size, never by comparing losses across λ). Eq. 2–9; fixes N10, A1, A2. **[fix]**
- Gauthier, E., Bach, F. & Jordan, M. I. (2025). *E-Values Expand the Scope of Conformal Prediction.* https://arxiv.org/abs/2503.13050. Code: https://github.com/GauthierE/evalues-expand-cp
- Vovk, V. & Wang, R. (2021). *E-values: Calibration, Combination and Applications.* Annals of Statistics 49(3).

## 9. Hyperspherical Confidence Mapping (HCM) notebook
- Choi, E., Kim, H.-Y., Lee, J. et al. (2026). *Uncertainty Estimation via Hyperspherical Confidence Mapping.* ICLR. https://arxiv.org/abs/2605.05964. Official code (verified): https://github.com/Abandoned-Puppy/HCM (`Toy_example.py`: 1D regression, σ = |R|·sqrt(|‖d‖²−1|); paper Appendix A.2 defines Cov@kσ and ECE_reg, A.4 the σ formula, Sec. 3.3 validation temperature scaling).
- Guo, C., Pleiss, G., Sun, Y. & Weinberger, K. Q. (2017). *On Calibration of Modern Neural Networks.* ICML. (temperature scaling, ECE)
- Amini, A., Schwarting, W., Soleimany, A. & Rus, D. (2020). *Deep Evidential Regression.* NeurIPS. (baseline the HCM paper compares against)

## 10. Software used for running and reporting
- PyTorch: Paszke, A. et al. (2019). *PyTorch: An Imperative Style, High-Performance Deep Learning Library.* NeurIPS.
- NumPy (Harris et al. 2020, Nature 585), SciPy (Virtanen et al. 2020, Nature Methods 17), pandas (McKinney 2010), Matplotlib (Hunter 2007), seaborn (Waskom 2021, JOSS), Plotly, openpyxl / XlsxWriter.
- Kaggle Notebooks (compute), Google Colab (original runs).
