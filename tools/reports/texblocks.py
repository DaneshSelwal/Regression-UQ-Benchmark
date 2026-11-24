r"""texblocks.py: shared LaTeX text of the reports (problem set-up, methods with maths, evaluation, future work).

The text describes the v2.0 notebooks exactly as they ran (see docs/CHANGELOG_FIXES.md for every change
against v1.2). It is shared by the per-dataset papers (make_reports.py) and the benchmark paper (paper_benchmark.py).
"""

PREAMBLE = r"""\documentclass[11pt]{article}
\usepackage[margin=1in]{geometry}\usepackage{amsmath,amssymb,amsthm,booktabs,graphicx,xcolor,longtable}
\usepackage[numbers,sort&compress]{natbib}
\usepackage[colorlinks=true,linkcolor=blue!50!black,citecolor=blue!50!black,urlcolor=blue!50!black]{hyperref}\usepackage{xurl}\emergencystretch=3em
\usepackage{caption}\captionsetup{font=small,labelfont=bf}\usepackage{placeins}\usepackage{enumitem}
\usepackage[most]{tcolorbox}
\newtcolorbox{insight}[1][]{colback=blue!3!white,colframe=blue!40!black,fonttitle=\bfseries\small,title=#1,boxrule=0.5pt,arc=2pt,left=4pt,right=4pt,top=2pt,bottom=2pt}
\newtheorem{proposition}{Proposition}
\setlist{itemsep=2pt,topsep=3pt}
\renewcommand{\floatpagefraction}{0.85}\renewcommand{\topfraction}{0.9}\renewcommand{\textfraction}{0.08}
\newcommand{\cov}{\mathrm{Cov}}\newcommand{\1}{\mathbb{1}}
"""

AUTHORS = r"""\author{Danesh Selwal \and Mahesh Pal}"""

# --------------------------------------------------------------------------------------------- set-up
SETUP = r"""
\section{Problem set-up and notation}\label{sec:setup}
We observe training pairs $(X_i,Y_i)$, $i=1,\dots,n$, with features $X_i\in\mathbb R^d$ and a real-valued response
$Y_i$, and want to say something about the response $Y_{n+1}$ of a new point $X_{n+1}$. A \emph{point predictor}
$\hat f$ returns one number $\hat f(X_{n+1})$. An \emph{uncertainty quantification} (UQ) method returns more:
\begin{itemize}
\item a \textbf{prediction interval} $\hat C(x)=[\ell(x),u(x)]$ that should contain the response with probability
$1-\alpha$ (here $1-\alpha=0.9$), or
\item a \textbf{predictive distribution} $\hat F_x$, a full cumulative distribution function for $Y$ given $X=x$, from which
intervals, quantiles and probabilities of exceedance can all be read.
\end{itemize}
Two properties are wanted, and they pull in opposite directions.

\paragraph{Validity (calibration).} The interval should contain the truth as often as it claims:
\begin{equation}\label{eq:marginal}
\Pr\big(Y_{n+1}\in\hat C(X_{n+1})\big)\;\ge\;1-\alpha .
\end{equation}
This is \emph{marginal} coverage: the probability is taken over the random training data \emph{and} the random new point, so it
is an average over the whole population of test points. It does not promise $1-\alpha$ for every value of $x$
(\emph{conditional} coverage, $\Pr(Y\in\hat C(X)\mid X=x)\ge1-\alpha$ for all $x$), which is impossible to guarantee
in finite samples without further assumptions \cite{lei2018distribution}.

\paragraph{Sharpness (efficiency).} Subject to validity, the interval should be as narrow as possible. An interval
$(-\infty,\infty)$ is perfectly valid and useless. We therefore always report coverage \emph{and} width
together, and compare widths only among methods that are valid.

\paragraph{Exchangeability.} Conformal guarantees rely on one assumption: the $n+1$ pairs
$(X_i,Y_i)$, $i=1,\dots,n+1$, are \emph{exchangeable}, meaning their joint distribution does not change when the
pairs are permuted. Independent and identically distributed (i.i.d.) data are exchangeable. Data with a time trend,
or a test set drawn from a different region than the training set, are not.

\paragraph{Notation.} $\alpha$ is the miscoverage level ($\alpha=0.1$ throughout, so intervals target 90\%);
$\hat f$ is a fitted regressor; $s(x,y)$ is a \emph{nonconformity score}, usually the absolute residual
$s(x,y)=|y-\hat f(x)|$; $\lceil\cdot\rceil$ rounds up; $\1\{\cdot\}$ is the indicator function; $\Phi$ and
$\varphi$ are the standard normal CDF and density; $n_{\text{test}}$ is the number of test points.
For a finite list of numbers $v_1,\dots,v_m$ we write $v_{(k)}$ for the $k$-th smallest.
"""

# --------------------------------------------------------------------------------------------- methods
METHODS = r"""
\section{Methods}\label{sec:methods}
The pipeline has eight notebooks (Fig.~\ref{fig:pipeline}). This section explains what each method computes and
why it should, or should not, be expected to give 90\% coverage. Each subsection ends with the exact settings used.

\subsection{Base regressors and hyperparameter tuning (NB1)}
Ten regressors are used: random forest \cite{breiman2001rf}, gradient boosting \cite{friedman2001gbm}, histogram
gradient boosting (HistGB, \cite{pedregosa2011sklearn}), XGBoost \cite{chen2016xgboost}, LightGBM \cite{ke2017lightgbm},
CatBoost \cite{prokhorenkova2018catboost}, GPBoost \cite{sigrist2022gpboost}, NGBoost \cite{duan2020ngboost},
PGBM \cite{sprangers2021pgbm} and TabNet \cite{arik2021tabnet}. All but TabNet are tree ensembles; TabNet is an
attention-based neural network for tables. Random forest is used only in tuning; the other nine are the \emph{base
models} that every UQ method wraps.

Each model's hyperparameters $\theta$ are chosen by Optuna \cite{akiba2019optuna} to minimise the 5-fold
cross-validated root mean squared error on the \emph{training set only}:
\begin{equation}
\hat\theta=\arg\min_\theta\;\frac15\sum_{k=1}^5\sqrt{\frac{1}{|I_k|}\sum_{i\in I_k}\big(Y_i-\hat f^{(-k)}_\theta(X_i)\big)^2},
\end{equation}
where $I_k$ is the $k$-th fold and $\hat f^{(-k)}_\theta$ is fitted on the other four folds. The sampler is Optuna's
AutoSampler, which picks a search strategy (for example a tree-structured Parzen estimator) from the search space.
Unpromising trials are stopped early by a \emph{pruner}; each model is tuned with several pruners (median,
Hyperband \cite{li2018hyperband}, Wilcoxon, none) and the setting with the lowest CV error is kept. The selected
settings are written to \texttt{best\_params.pkl} and loaded by every later notebook, so all UQ methods wrap the same
models. The test set is never used for tuning or calibration; it is used once, for the final evaluation.

\begin{insight}[Why this matters]
In v1.2 the tuning objective was the test-set error, so every later coverage number was optimistic. Tuning on
cross-validation only (FIX~1) is a precondition for any honest coverage estimate.
\end{insight}

\subsection{Quantile regression (NB2)}
The $\tau$-quantile $q_\tau(x)$ of $Y$ given $X=x$ minimises the expected \emph{pinball loss}
\cite{koenker1978quantile}
\begin{equation}
\rho_\tau(r)=r\,\big(\tau-\1\{r<0\}\big)=\begin{cases}\tau\,r & r\ge0\\(\tau-1)\,r & r<0\end{cases},\qquad r=y-q .
\end{equation}
\emph{Why this works:} the derivative of $\mathbb E\,\rho_\tau(Y-q)$ with respect to $q$ is $\Pr(Y<q)-\tau$, which is
zero exactly when $q$ is the $\tau$-quantile. Under-predictions are charged $\tau$ per unit and over-predictions
$1-\tau$, so for $\tau=0.95$ the model is pushed up until only 5\% of points lie above it.

Every boosted model (and TabNet, with a pinball loss) is fitted separately for 13 levels
$\tau\in\{0.05,0.1,0.15,0.2,0.3,\dots,0.8,0.85,0.9,0.95\}$. The 90\% interval is $[\hat q_{0.05}(x),\hat q_{0.95}(x)]$.
This interval has \emph{no finite-sample guarantee}: it is correct only if each quantile model is correct, and with a
few hundred points the extreme quantiles are learned from very few observations.

\paragraph{Quantile crossing.} Because each level is a separate model, the predicted quantiles of one point can
be out of order (e.g.\ $\hat q_{0.05}(x)>\hat q_{0.10}(x)$) \cite{chernozhukov2010rearrangement}. We report the
\emph{crossing rate}, the share of test points with at least one inversion, and then sort the 13 values of each
point (monotone rearrangement). Sorting never makes a quantile prediction worse in pinball loss
\cite{chernozhukov2010rearrangement}.

\paragraph{CRPS from quantiles.} The continuous ranked probability score of a predictive CDF $F$ is
\begin{equation}\label{eq:crps}
\mathrm{CRPS}(F,y)=\int_{-\infty}^{\infty}\big(F(z)-\1\{z\ge y\}\big)^2dz
=2\int_0^1\rho_\tau\big(y-F^{-1}(\tau)\big)\,d\tau ,
\end{equation}
the second form showing that CRPS is the pinball loss averaged over all levels \cite{gneiting2007scoring}. With 13 levels
we use $\mathrm{CRPS}\approx2\sum_jw_j\,\rho_{\tau_j}(y-\hat q_{\tau_j})$, where $w_j$ is the share of $[0,1]$ that level $j$
represents ($\sum_jw_j=1$). CRPS is in the units of the target, is \emph{proper} (it is minimised in expectation by the
true distribution) and reduces to the absolute error for a point forecast.

\subsection{Predictive distributions (NB3, NB4)}
These methods output a full distribution $\hat F_x$. Where it is Gaussian, $\hat F_x=\mathcal N(\hat\mu(x),\hat\sigma^2(x))$,
the 90\% interval is $\hat\mu\pm z_{0.95}\hat\sigma$ with $z_{0.95}=1.645$, and the CRPS has the closed form
\begin{equation}\label{eq:crpsgauss}
\mathrm{CRPS}\big(\mathcal N(\mu,\sigma^2),y\big)=\sigma\Big[z\,\big(2\Phi(z)-1\big)+2\varphi(z)-\tfrac{1}{\sqrt\pi}\Big],\qquad z=\frac{y-\mu}{\sigma}.
\end{equation}
\begin{itemize}
\item \textbf{NGBoost} \cite{duan2020ngboost} boosts the parameters $(\mu,\log\sigma)$ of a Gaussian, moving them along the
\emph{natural} gradient of the negative log-likelihood, which makes the step size invariant to how the distribution is
parameterised.
\item \textbf{PGBM} \cite{sprangers2021pgbm} treats each tree leaf weight as a random variable whose mean and variance
are estimated from the gradients and Hessians that fall into the leaf; summing over trees gives $\hat\mu(x)$ and
$\hat\sigma^2(x)$ in one pass.
\item \textbf{IBUG} \cite{brophy2022ibug} wraps any tree ensemble. For a test point it finds the $k$ training points that
share the most leaves with it (highest \emph{affinity}), and uses the spread of their targets as $\hat\sigma(x)$; the
mean comes from the ensemble. We use IBUG with XGBoost and LightGBM.
\item \textbf{Treeffuser} \cite{beltranvelez2024treeffuser} learns a conditional diffusion model whose score function is
estimated by gradient-boosted trees; samples from the reverse diffusion are non-parametric draws from $\hat F_x$.
\item \textbf{CARD} \cite{han2022card} (NB4) is a classification-and-regression diffusion model guided by a point predictor
$\hat f(x)$. The forward process adds noise while drifting towards the guide:
\begin{equation}
y_t=\sqrt{\bar\alpha_t}\,y_0+\big(1-\sqrt{\bar\alpha_t}\big)\hat f(x)+\sqrt{1-\bar\alpha_t}\,\varepsilon,\qquad\varepsilon\sim\mathcal N(0,1),
\end{equation}
with $\bar\alpha_t=\prod_{s\le t}(1-\beta_s)$ and a linear schedule $\beta_1=10^{-4},\dots,\beta_T=0.02$, $T=1000$. A network
$\varepsilon_\theta(x,y_t,\hat f(x),t)$ is trained to predict the noise, $\min\mathbb E\|\varepsilon-\varepsilon_\theta\|^2$
\cite{ho2020ddpm}, and the reverse process turns pure noise centred on $\hat f(x)$ into samples of $Y\mid x$. Each of the
nine tuned models is used once as the guide; 1000 samples per test point are drawn (official settings: 5000 epochs,
batch 32, learning rate $10^{-3}$, EMA of the weights with warm-up).
\end{itemize}
For sample-based methods (Treeffuser, CARD) we summarise the samples by their mean and standard deviation and report the
Gaussian CRPS~\eqref{eq:crpsgauss} and Gaussian 90\% interval, so that every distributional method is scored the same way.
The notebooks also report the negative log-likelihood, the Hersbach decomposition of CRPS into reliability, resolution and
uncertainty \cite{hersbach2000crps}, and the probability integral transform $\mathrm{PIT}=\hat F_X(Y)$, which is uniform on
$[0,1]$ for a calibrated forecast \cite{gneiting2007sharpness,kuleshov2018calibrated}. A U-shaped PIT histogram means the
distributions are too narrow (over-confident); a hump means they are too wide.

\begin{insight}[What to expect]
None of these distributions comes with a coverage guarantee. A Gaussian assumption fails when errors are skewed or
heavy-tailed, and a $\hat\sigma(x)$ fitted on training data is typically too small, because training residuals are
smaller than test residuals. Their intervals are therefore a \emph{test} of the model, not a guarantee.
\end{insight}

\subsection{Split, cross and bootstrap conformal prediction (NB5)}
Conformal prediction \cite{vovk2005alrw,lei2018distribution,angelopoulos2023gentle} turns any point predictor into an
interval with a finite-sample guarantee. Box~1 in the repository README illustrates the three steps.

\paragraph{Split conformal (SCP).} Split the training data into a fitting part and a calibration part
$\mathcal C$ with $m=|\mathcal C|$ points. Fit $\hat f$ on the first part, compute scores $s_i=|Y_i-\hat f(X_i)|$ on
$\mathcal C$, and set
\begin{equation}\label{eq:scp}
\hat q=s_{(k)},\quad k=\big\lceil(m+1)(1-\alpha)\big\rceil,\qquad \hat C(x)=\big[\hat f(x)-\hat q,\;\hat f(x)+\hat q\big].
\end{equation}
\begin{proposition}[Coverage of split conformal]
If the calibration pairs and the test pair are exchangeable, $\Pr(Y_{n+1}\in\hat C(X_{n+1}))\ge1-\alpha$. If the scores
have no ties, also $\Pr(Y_{n+1}\in\hat C(X_{n+1}))\le1-\alpha+\frac1{m+1}$.
\end{proposition}
\emph{Why.} $\hat f$ was fitted without the calibration and test points, so the $m+1$ scores $s_1,\dots,s_m,s_{n+1}$ are
exchangeable. The rank of $s_{n+1}$ among them is then uniform on $\{1,\dots,m+1\}$, and $Y_{n+1}\in\hat C$ exactly when
$s_{n+1}\le s_{(k)}$, which happens with probability $k/(m+1)\ge1-\alpha$. The ``$+1$'' accounts for the test point
itself. Nothing is assumed about the model or the noise; a poor model gives a valid but wide interval.

\emph{Settings.} MAPIE \cite{taquet2022mapie}: one seeded random half of the training rows fits the model, the other half
calibrates. PUNCC \cite{mendil2023puncc}: its own split. Both use $\alpha=0.1$.

\paragraph{CV+ and jackknife+-after-bootstrap (J+aB).} Split conformal wastes half the data on calibration. CV+
\cite{barber2021jackknife} fits $K$ models $\hat f^{(-k)}$, each without fold $k$, and scores every training point
with the model that did not see it: $R_i=|Y_i-\hat f^{(-k(i))}(X_i)|$. The interval is
\begin{equation}\label{eq:cvplus}
\hat C(x)=\Big[\,\hat q^-_{\alpha}\big\{\hat f^{(-k(i))}(x)-R_i\big\},\;\hat q^+_{\alpha}\big\{\hat f^{(-k(i))}(x)+R_i\big\}\Big],
\end{equation}
where $\hat q^-_\alpha$ is the $\lfloor\alpha(n+1)\rfloor$-th smallest and $\hat q^+_\alpha$ the
$\lceil(1-\alpha)(n+1)\rceil$-th smallest of the $n$ values. Its guarantee is weaker on paper, $1-2\alpha$ (up to a small
term for $K$-fold CV+), but in practice coverage is close to $1-\alpha$ and the interval narrower, because every point
is used both to fit and to calibrate. J+aB \cite{kim2020jab} replaces the folds by $B$ bootstrap samples: $R_i$ uses the
average of the models whose bootstrap sample did not contain $i$ (out-of-bag), and the test prediction aggregates the same
models. \emph{Settings:} $K=5$; J+aB with 50 resamplings (MAPIE \texttt{method="plus"}).

\paragraph{Conformalised quantile regression (CQR).} Symmetric intervals $\hat f\pm\hat q$ have the same width everywhere.
CQR \cite{romano2019cqr} starts from quantile models $\hat q_{\alpha/2},\hat q_{1-\alpha/2}$, whose band already adapts to
$x$, and corrects it with the score
\begin{equation}
s_i=\max\big\{\hat q_{\alpha/2}(X_i)-Y_i,\;Y_i-\hat q_{1-\alpha/2}(X_i)\big\},\qquad
\hat C(x)=\big[\hat q_{\alpha/2}(x)-\hat q,\;\hat q_{1-\alpha/2}(x)+\hat q\big],
\end{equation}
with $\hat q$ from~\eqref{eq:scp}. The score is negative when the point is inside the band, so $\hat q$ can shrink an
over-wide band as well as widen a narrow one; the guarantee is that of split conformal. \emph{Settings:} PUNCC CQR with
native quantile losses (LightGBM, GPBoost, XGBoost, Gradient Boosting, CatBoost, TabNet) or distribution quantiles (NGBoost, PGBM).

\subsection{Weighted conformal prediction: NexCP (NB6)}
When the data drift, old calibration points describe the new point less well. Non-exchangeable conformal prediction
(NexCP, \cite{barber2023beyond}) gives each calibration score a fixed weight $w_i\in[0,1]$ chosen \emph{before} seeing
the data, and uses the weighted quantile
\begin{equation}
\hat q=\mathrm{Quantile}_{1-\alpha}\Big(\sum_{i=1}^n\tilde w_i\,\delta_{s_i}+\tilde w_{n+1}\,\delta_{+\infty}\Big),\qquad
\tilde w_i=\frac{w_i}{\sum_{j=1}^nw_j+1},\;\tilde w_{n+1}=\frac{1}{\sum_jw_j+1},
\end{equation}
where $\delta_v$ is a point mass at $v$. The mass at $+\infty$ plays the role of the ``$+1$'' in~\eqref{eq:scp}. With
$w_i=\rho^{\,n+1-i}$ (here $\rho=0.99$) recent points count more. The guarantee degrades gracefully:
\begin{equation}
\Pr\big(Y_{n+1}\in\hat C(X_{n+1})\big)\;\ge\;1-\alpha-\sum_{i=1}^n\tilde w_i\,d_{\mathrm{TV}}\big(Z,Z^{i}\big),
\end{equation}
where $d_{\mathrm{TV}}(Z,Z^i)$ measures how much the data change when point $i$ is swapped with the test point. If the data are
exchangeable every term is zero and~\eqref{eq:marginal} holds; under drift, down-weighting old points keeps the gap small.
NexCP is run in four forms: \emph{Split} (out-of-fold residuals as scores), \emph{CV+} and \emph{J+aB} (weighted
versions of~\eqref{eq:cvplus}), and \emph{Full} conformal, which for each test point refits the model on the training data
plus a candidate $(x,y)$ for 50 values of $y$ on a grid and keeps the values of $y$ that conform. If the weighted quantile is
$+\infty$ the interval is capped at the largest residual and flagged; if a Full interval reaches the edge of its grid the width
is a lower bound and the method is not ranked (marked $\dagger$).

\subsection{Online conformal prediction (NB6)}
Online methods treat the test set as a \emph{stream}: the interval for point $t$ is issued, then $Y_t$ is revealed and used to
update the next interval. They do not need exchangeability; instead they guarantee that the \emph{long-run average}
miscoverage is close to $\alpha$, even under arbitrary distribution shift. Every method here starts from 5-fold out-of-fold
training residuals and issues each interval \emph{before} its own error is used (no look-ahead).

\paragraph{ACI.} Adaptive conformal inference \cite{gibbs2021aci} adjusts the level it asks for:
\begin{equation}\label{eq:aci}
\alpha_{t+1}=\alpha_t+\gamma\,(\alpha-\mathrm{err}_t),\qquad \mathrm{err}_t=\1\{Y_t\notin\hat C_t(X_t)\},
\end{equation}
and $\hat C_t=\hat f(x)\pm\hat q_t$ uses the $(1-\alpha_t)$ quantile of the out-of-fold training residuals plus the test
residuals revealed so far. After a miss ($\mathrm{err}_t=1$) the level $\alpha_t$ drops, so the next
interval is wider; after a hit it rises slightly. Summing~\eqref{eq:aci} over $t$ gives
$\big|\frac1T\sum_{t=1}^T\mathrm{err}_t-\alpha\big|\le\frac{\max(\alpha_1,1-\alpha_1)+\gamma}{\gamma T}$, so the average
miscoverage converges to $\alpha$ for any data sequence. \emph{Settings:} $\gamma=0.005$.

\paragraph{FACI.} The best $\gamma$ is unknown. Fully adaptive conformal inference \cite{gibbs2024faci} runs ACI for a grid of
$\gamma$ values ($0.001\cdot2^k$, $k=0,\dots,7$) as experts and combines their levels with exponential weights based on each
expert's pinball loss (reference settings, interval length $I=100$).

\paragraph{SF-OGD and SAOCP.} Instead of the level, these methods learn the radius $q_t$ directly by online gradient descent
on the pinball loss of the scores,
\begin{equation}
q_{t+1}=q_t-\eta\,\frac{\nabla_t}{\sqrt{\sum_{s\le t}\nabla_s^2}},\qquad \nabla_t=\alpha-\1\{s_t>q_t\},
\end{equation}
so $q$ grows after a miss and shrinks slowly after a hit; the scale-free step needs no tuning of the learning rate to the
units of $Y$ \cite{orabona2018scalefree} ($\eta$ is the largest calibration residual, as in the reference code, and both methods
are warm-started on the calibration residuals). SAOCP \cite{bhatnagar2023saocp} runs SF-OGD experts started at different times and
combines them by coin betting (expert lifetime 8), which gives low regret on \emph{every} time interval and lets it react faster to a change.

\paragraph{Online split, CV+, J+aB and MFCS.} \emph{Online split} (and \emph{MFCS Split}, identical in the code) uses a fixed
radius, the conformal quantile of the out-of-fold training residuals. \emph{Online CV+} and \emph{Online J+aB} apply~\eqref{eq:cvplus}
and add each revealed test residual to the calibration set. \emph{MFCS Full} refits the model on the training data plus all test
points already revealed, and takes its radius from the out-of-fold residuals of that refit; it is the only method that also
updates the point prediction with past test labels. (MFCS has no published reference; it is kept as written in the original
notebooks.)

\paragraph{COP.} Distribution-informed online conformal prediction \cite{hu2026cop} runs the plain online quantile update
$\tilde q_{t+1}=\tilde q_t-\eta_t g_t$, with $g_t=\alpha$ after a hit and $g_t=-(1-\alpha)$ after a miss, and adds an
\emph{optimistic} correction that looks at the empirical distribution $\hat F_t$ of the last $T_{\text{burnin}}$ scores:
\begin{equation}
\tilde g_t=\hat F_t\big(\tilde q_t-\eta_tg_t\big)-(1-\alpha),\qquad q_{t+1}=\max\big(0,\;\tilde q_{t+1}-c\,\eta_t\,\tilde g_t\big).
\end{equation}
Interpretation: if the step about to be taken would already put more than $1-\alpha$ of the recent scores below the radius, the
correction pulls it back, and vice versa, so the radius anticipates where the coverage is heading. The learning rate is
$\eta_t=\eta\cdot(\max-\min)$ of the recent scores, as in the official code. \emph{Settings:} $\eta=0.05$, $c=0.5$, start value the
90\% quantile of the out-of-fold training residuals, $T_{\text{burnin}}$ set from the test-set size (FIX D5).

\begin{insight}[Reading online results]
Online methods use the labels of earlier test points. Their coverage is an average over the stream and their width can
shrink as they learn, which is legitimate when labels really do arrive one by one (monitoring, forecasting). On a static test set
the order of the rows matters; we use the stored order.
\end{insight}

\subsection{Extreme conformal prediction (NB6)}
For a high level the empirical quantile is the largest residual, and it cannot extrapolate. Extreme CP \cite{pasche2025extreme}
fits a generalised Pareto distribution (GPD) \cite{pickands1975gpd} to the exceedances $s_i-u$ of the signed scores
$s_i=Y_i-\hat f(X_i)$ over a high threshold $u$,
\begin{equation}
\Pr(s-u>z\mid s>u)\approx\Big(1+\frac{\xi z}{\sigma}\Big)^{-1/\xi},
\end{equation}
and inverts it for the required level; the procedure is repeated for thresholds at the 0.80--0.99 quantiles (keeping at least 10
exceedances) and the largest bound is used. The result is a \emph{one-sided} bound $Y\le\hat f(x)+\hat q$, so its coverage is
$\Pr(Y\le\hat f+\hat q)$ and it is reported separately from the two-sided intervals.

\subsection{Adaptive coverage policies (NB7)}
All methods above fix $\alpha$ in advance. ACP \cite{gauthier2026acp} lets a small network (AlphaNet: 32 ReLU units, sigmoid
output) choose a level $\tilde\alpha(x)$ for each test point, trading width against coverage through the loss
$\text{width}+\lambda\,\tilde\alpha$ ($\lambda=50$ reported; $\lambda\in\{10,20,50\}$ compared). Validity for a data-dependent level
comes from conformal \emph{e-values} \cite{vovk2021evalues,gauthier2025evalues}: with calibration scores $s_1,\dots,s_n$ the
candidate $y$ is excluded when
\begin{equation}
e(y)=\frac{(n+1)\,s(x,y)}{\sum_{i=1}^ns_i+s(x,y)}\;\ge\;\frac1{\tilde\alpha(x)}
\quad\Longleftrightarrow\quad s(x,y)\ge\frac{\sum_is_i}{\tilde\alpha(x)(n+1)-1},
\end{equation}
so the set is $\hat f(x)\pm\sum_is_i/(\tilde\alpha(n+1)-1)$. The guarantee is $\mathbb E\big[\1\{Y\notin\hat C\}/\tilde\alpha(X)\big]\le1$:
coverage must be compared with the \emph{learned} target $1-\tilde\alpha$, not with 90\%. If the policy pushes $\tilde\alpha$ to its
upper limit (0.999) for every point, the target coverage is 0.1\%, every set has the same width, and the learned level carries no
information; we flag such models as \emph{saturated}.

\subsection{Hyperspherical confidence mapping (NB8)}
HCM \cite{choi2026hcm} is a single, sampling-free network. The scalar target is lifted to $\mathbf y=(y,y)\in\mathbb R^2$ and
split into a magnitude $R=\|\mathbf y\|$ and a unit direction $\mathbf d=\mathbf y/R$. A network with three hidden layers of 20
units predicts $\hat R$ and $\hat{\mathbf d}$ with the symmetric loss
$\mathrm{MSE}(\hat R\,\mathbf d,\mathbf y)+\mathrm{MSE}(R\,\hat{\mathbf d},\mathbf y)$. At test time the predicted direction should lie on the
unit circle; how far it falls off is the uncertainty,
\begin{equation}
\hat\sigma_{\text{raw}}(x)=\sqrt{\big|\,\|\hat{\mathbf d}(x)\|^2-1\big|}\;|\hat R(x)|,
\end{equation}
and a single scale factor is fitted on a validation split so that $\pm1,2,3\,\hat\sigma$ cover 68/95/99.7\%. We report the 95\%
interval, the coverage at $1,2,3\sigma$, and the Spearman correlation between $|y-\hat y|$ and $\hat\sigma$ (positive means the
network knows when it is wrong).
"""

# --------------------------------------------------------------------------------------------- evaluation
def evaluation(n=None, tol=None):
    s = r"""
\section{Evaluation protocol}\label{sec:eval}
\paragraph{Coverage and width.} For an interval method and base model, the empirical coverage on the test set is
$\hat c=\frac1{n_{\text{test}}}\sum_{t}\1\{\ell(X_t)\le Y_t\le u(X_t)\}$ and the mean width is
$\bar w=\frac1{n_{\text{test}}}\sum_t\big(u(X_t)-\ell(X_t)\big)$. To compare datasets with different units we also report the
\emph{relative width} $\bar w/\mathrm{sd}(Y_{\text{test}})$: a relative width of 1 means the interval is as wide as one standard
deviation of the target, and an interval that ignored $x$ altogether would need about $2\times1.645=3.3$ for a Gaussian target.

\paragraph{When is a coverage ``valid''?} Even a perfect 90\% method will not cover exactly 90\% of a finite test set. If each test
point is covered independently with probability $0.9$, then $n_{\text{test}}\hat c\sim\mathrm{Binomial}(n_{\text{test}},0.9)$, so
$\hat c$ has standard error $\sqrt{0.9\cdot0.1/n_{\text{test}}}$. We call a method \emph{valid} when $\hat c$ is not significantly
below 90\% in a one-sided test at the 5\% level:
\begin{equation}\label{eq:valid}
\hat c\;\ge\;0.9-1.645\sqrt{\frac{0.09}{n_{\text{test}}}} .
\end{equation}"""
    if n is not None:
        s += r"""
On this dataset $n_{\text{test}}=""" + str(n) + r"""$, so the standard error is $""" + f"{(0.09/n)**0.5:.3f}" + r"""$ and the threshold is
$""" + f"{0.9-tol:.3f}" + r"""$: a coverage of, say, """ + f"{100*(0.9-tol/2):.0f}" + r"""\% is statistically consistent with 90\%."""
    s += r"""
The test is approximate (online methods produce dependent errors, and many methods are tested at once), so it is used as a
screening rule, not a formal decision.

\paragraph{Recommended method.} A method wraps up to nine base models. It is \emph{recommended} on a dataset if it is the
narrowest method (median relative width over models) among those that are valid for at least 75\% of the models they ran on,
ran on at least three models, and have no width that is a lower bound. Extreme CP (one-sided), ACP (learned level) and HCM (95\%,
own network) are discussed separately.

\paragraph{Scores for distributions.} CRPS (eqs.~\ref{eq:crps}--\ref{eq:crpsgauss}; lower is better, in target units), the negative
log-likelihood, PIT diagnostics and, for HCM, the Spearman correlation between absolute error and predicted spread.
"""
    return s

# --------------------------------------------------------------------------------------------- closing
LIMITATIONS = r"""
\paragraph{Limitations and threats to validity.}
\begin{itemize}
\item \textbf{One split, one run.} Every number comes from one fixed train/test split and one run, so coverages carry the sampling
error of~\eqref{eq:valid} and no confidence intervals across splits or seeds are available. Small test sets (fewer than 100 points)
can only distinguish large departures from 90\%.
\item \textbf{Order of the test set.} Online methods depend on the order of the stored test rows, which has no time meaning on most
datasets; their good widths partly reflect learning from earlier test labels.
\item \textbf{Marginal, not conditional.} All guarantees are marginal. A method can cover 90\% overall and much less for large target
values; the interval figures show where misses concentrate.
\item \textbf{Approximate tuning reproducibility.} Optuna search is not bit-reproducible; the stored tuning run is the one every
later notebook used. CARD, TabNet, HCM and ACP are neural and depend on their seeds.
\item \textbf{Compute limits.} Kaggle stops a session after 12 hours. Jobs that did not finish are listed as not run; their methods are
missing from the tables rather than estimated.
\end{itemize}
"""

FUTURE = r"""
\section{Future work: the \texttt{reguq} library}\label{sec:future}
The notebooks are a research pipeline: they are long, dataset-specific copies of one another, and their methods cannot be imported
into other code. The next step of this project is a Python package, \textbf{\texttt{reguq}} (regression uncertainty quantification),
that turns the verified methods of this benchmark into a tested, documented library:
\begin{itemize}
\item one interface, \texttt{fit / calibrate / predict\_interval / predict\_distribution}, for every family studied here (quantile
regression, distributional models, split/cross/bootstrap conformal, NexCP, online conformal, Extreme CP, ACP, HCM), wrapping any
scikit-learn-style regressor;
\item the evaluation protocol of Section~\ref{sec:eval} (coverage with its sampling tolerance, relative width, CRPS, PIT, crossing
rate) and the recommendation rule as functions, so that a new dataset can be benchmarked in a few lines;
\item unit tests that check each method against the authors' reference code and against its theoretical coverage on synthetic data;
\item repeated random splits and confidence intervals for coverage and width, which the notebooks do not provide, and conditional
coverage diagnostics (coverage by target range and by feature bins).
\end{itemize}
"""
