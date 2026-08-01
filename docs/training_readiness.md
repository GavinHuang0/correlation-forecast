# Quant training status and model ladder

This document is the current status record for the first quantitative
stock-sector correlation experiment. The T0 proxy was intentionally skipped.
T1 and T2 were constructed for both the tradable ETF benchmark and the
leave-one-out (LOO) peer basket, and rungs 1–4 were trained on the same locked
walk-forward protocol.

The machine-readable protocol is
[`config/quant_training_protocol_v1.json`](../config/quant_training_protocol_v1.json).
Audits and compact results are under
[`experiments/quant_training/v1`](../experiments/quant_training/v1).
Large Parquet panels, predictions, and fit records are generated locally under
`data/features/quant/training_v1` and `outputs/quant_training/v1`.

## Current readiness

| Component | Status | Evidence |
|---|---|---|
| 22 non-factor Bollerslev-style ETF features | Ready | 58,500 complete rows; input hashes verified |
| 22 LOO features | Ready | 14 stock–LOO pair features plus the same eight sector-state features |
| Additional dense, volatility, and extended-hours blocks | Ready | 79,110-row hash-verified panel with explicit availability indicators |
| T1 ETF and LOO targets | Ready | 78,840 valid rows each |
| T2 ETF and LOO targets | Ready | 77,760 valid rows each |
| Matched modeling panel | Ready | 27,510 rows, 30 stocks, 917 dates |
| Chronological splitter and T2 boundary purge | Ready | Three locked date-blocked folds |
| Rungs 1–4 | Complete | Linear, practical linear, GPU XGBoost, and DCC-GARCH |
| Deterministic daily-news v1 panel | Complete exploratory artifact | 27,510 exact Q+D joins; 44 materialized and 40 recommended fields; not historical-version-safe |
| D2-Normalized v2 | Complete exploratory artifact | 71,760 complete stock-days; 27,510 exact Q56+D2 joins; 30 normalized plus five level features; non-version-safe |
| Deterministic-news v2 training | Complete exploratory development run | D0/D1/D3/D4/D5 and stale/wrong-stock controls fit; matched D43 skipped; no D3 target passed every useful-news gate |
| Shared v2 semantic input corpus | Complete exploratory artifact | 466,902 article-target assignments, 55,197 assigned articles, 27,510 stock-days, and 204 no-candidate stock-days; bounded shared text and routing hashes are frozen |
| `W17__flan_t5_xl` | Construction pipeline ready; inference not run | Full tokenizer preflight passed 466,902 assignments and 4,202,118 logical prompts with zero violations (field maxima 398/427/360 under 512); one-record CUDA float16 smoke passed on the RTX 3070 Ti; no full-corpus predictions or daily W17 panel |
| `W17__gpt_5_6_sol` | Not pursued in the current construction pass | Its future design remains deterministic coarsening from accepted GPT R70 fine labels; no daily panel or training exists |
| `R70__flan_t5_xl` | Not pursued in the current construction pass | Fine taxonomy exists, but no pinned fine-XL workflow, calibration, daily panel, or training exists |
| `R70__gpt_5_6_sol` | Offline construction pipeline ready; paid inference blocked | Final hash-bound preflight passed at 933,804 two-view requests, 3,613,888,330 bytes, maximum request size 4,458 bytes, and 934 conservative files; no paid call or authorization gates satisfied |
| `WL17__flan_t5_xl` v3 | Complete exploratory failed-gate run | 50,488/50,488 articles inferred with zero failures; 65.1244% choice-order agreement missed the 85% gate; 27,510 daily rows built; S0-S4 plus ten controls/sensitivities trained; no linear target passed every useful-semantic gate |
| `G40__gpt_5_6_sol` v3 | Ordered global-only design; construction contract incomplete | Forty named article-global semantic columns; request/acceptance/aggregation rules still require a freeze and paid pilot |
| `R70-Lite__gpt_5_6_sol` v3 | Cost plan only; schema incomplete | Factorized target projection remains blocked on an ordered feature contract, projection/conflict rules, and universal grounding |
| Economic hedge evaluation and dependence-aware inference | Not yet run | Forecast metrics are complete; economic/statistical follow-up remains |

## Target design

### Forecast-origin notation

A row dated official session $t$ is a forecast issued at 09:00 ET on $t$;
its target window starts at 09:30. Let $\mathcal F_{t,09{:}00}$ denote the
information available at that cutoff. Every completed-session price feature
is capped by `asof_session = prev(t) < t`; some publication-sensitive fields
are deliberately lagged farther. Current-date extended-hours inputs contain
only bars whose starts are before 09:00, and scheduled-calendar fields contain
only information known by that cutoff.

The supervised models estimate

$$
\widehat z_{i,b,t}
=
f\!\left(X_{i,b,t};\widehat\theta\right),
\qquad
z_{i,b,t}
=
\operatorname{atanh}\!\left(
\operatorname{clip}(\rho_{i,b,t},-0.995,0.995)
\right),
$$

where $X_{i,b,t}$ is measurable with respect to
$\mathcal F_{t,09{:}00}$. No regular-session observation from $t$ enters a
feature. The structural `asof_session < forecast_date` assertion covers
completed-session features; the source builders separately enforce the 09:00
cutoff for extended-hours inputs.

ETF and LOO are parallel targets because they answer different questions:

- **ETF:** tradable hedge coupling. It directly measures the relationship to
  the instrument available for hedging, but the ETF may contain the target
  stock and therefore includes mechanical self-exposure.
- **LOO:** five-peer research coupling. It excludes the target stock and is the
  cleaner test of common sector movement, but it is an equal-weight basket of
  the other five configured peers rather than a complete point-in-time sector
  index.

Neither target should silently replace the other. The ETF result supports an
operational hedging claim; the LOO result supports the cleaner sector-coupling
claim. The first experiment therefore trains and reports both on identical
eligible stock-date observations.

The ETF's mechanical self-exposure can be seen from the first-order
decomposition

$$
R^{ETF}_t
\approx
w_{i,t}R_{i,t}+(1-w_{i,t})R^{-i}_t,
$$

which implies

$$
\operatorname{Cov}(R_i,R^{ETF})
\approx
w_i\operatorname{Var}(R_i)
+(1-w_i)\operatorname{Cov}(R_i,R^{-i}).
$$

The first term is present even if the stock has no economic relationship with
the other constituents. LOO removes this term, at the cost of being a
five-stock equal-weight research factor rather than a directly tradable
index.

### T1: same-day regular-session realized correlation

Let $\mathcal K_t^{RTH}$ be the official regular-session 15-minute schedule:
26 intervals on a normal day and 14 on an official 13:00 close. For asset
$a$,

$$
r_{a,t,1}=\log\!\left(\frac{C_{a,t,1}}{O_{a,t,1}}\right),
\qquad
r_{a,t,k}=\log\!\left(\frac{C_{a,t,k}}{C_{a,t,k-1}}\right),\quad k>1,
$$

where the later return exists only across an exact 15-minute timestamp gap.
Targets contain no close-to-open overnight return. For stock $i$ and
benchmark $b$, define

$$
C_{i,b,t}=\sum_{k\in\mathcal K_t^{RTH}}r_{i,t,k}r_{b,t,k},
\qquad
V_{a,t}=\sum_{k\in\mathcal K_t^{RTH}}r_{a,t,k}^{2}.
$$

Then T1 is

$$
\rho^{(1)}_{i,b,t}
=
\frac{C_{i,b,t}}{\sqrt{V_{i,t}V_{b,t}}}.
$$

This is non-demeaned realized correlation, equivalently cosine normalization
of the two return vectors, not Pearson sample correlation.

ETF and LOO targets require the same complete official interval set across the
stock, sector ETF, and all five peers: 26 returns on a normal session and 14 on
an official 13:00 close. A missing constituent interval invalidates the target
day rather than changing the LOO composition. By design, an ETF gap also
invalidates the LOO target so ETF and LOO comparisons use identical
observations. A zero or nonfinite variance makes the correlation missing.

At each interval, the LOO benchmark is constructed in simple-return space:

$$
r^{LOO}_{i,t,k} =
\log\left(
1+\frac{1}{5}\sum_{j\ne i}
\left[\exp(r_{j,t,k})-1\right]
\right).
$$

The weights reset at every interval. LOO is therefore neither a buy-and-hold
daily peer portfolio nor an ETF-minus-stock reconstruction.

### T2: current-plus-next-four-session realized correlation

Let $d_0=t,d_1,\ldots,d_4$ be five successive official sessions. T2 sums
daily covariance and variance components before normalization:

$$
\rho^{(5)}_{i,b,t} =
\frac{\sum_{\ell=0}^{4} C_{i,b,d_\ell}}
{\sqrt{
\left(\sum_{\ell=0}^{4}V_{i,d_\ell}\right)
\left(\sum_{\ell=0}^{4}V_{b,d_\ell}\right)
}}.
$$

It is not an average of five daily correlations. All five sessions must be
valid, and `target_end_date` records $d_4$. T2 is the current session plus
the next four official sessions, not $t+1,\ldots,t+5$.

All regression targets use
`atanh(clip(correlation, -0.995, 0.995))`; predictions are converted back with
`tanh`.

### Persistence forecasts

The persistence benchmark is target-specific and strictly backward-looking.
For $H$ completed sessions,

$$
\rho^{persist(H)}_{i,b,t}
=
\frac{\sum_{\ell=1}^{H}C_{i,b,t-\ell}}
{\sqrt{
\left(\sum_{\ell=1}^{H}V_{i,t-\ell}\right)
\left(\sum_{\ell=1}^{H}V_{b,t-\ell}\right)
}}.
$$

Here $t-\ell$ denotes official-session indexing. T1 uses $H=1$; T2 uses
$H=5$. Thus T2 persistence is the latest fully observable five-session
component-aggregated correlation $t-5,\ldots,t-1$, not a lagged,
future-overlapping T2 label.

## Construction audit

The exact audit in
[`target_integrity.json`](../experiments/quant_training/v1/construction/audits/target_integrity.json)
passed all of the following:

- component-to-correlation reconstruction for all four targets;
- Fisher transformation;
- exact T2 five-session component sums and end dates;
- exact backward T1 persistence;
- identical ETF/LOO eligibility and stock variance;
- expected normal and early-close interval counts; and
- correlation bounds.

The construction also corrected the old additional-feature implementation:

- the first 09:30 bar's open-to-close return is now included in realized
  volatility;
- close-to-close returns are not formed across missing 15-minute bars;
- lags do not jump across a missing official session;
- premarket overnight returns require the exact preceding official close;
- sector dispersion requires the configured sector coverage; and
- explicit premarket/aftermarket availability indicators accompany sparse
  extended-hours inputs.

The Bollerslev pair-coverage denominator is now the official expected interval
count rather than the more complete observed leg. The historical core feature
builder retains its declared 80% threshold; the T1/T2 targets use the stricter
complete-schedule contract above.

## Feature blocks

### Target-specific core 22

Each target uses its own 14 pair-history features:

```text
rc_d, rc_w, rc_m
rc_negative_d, rc_negative_w, rc_negative_m
exp_rc_d, exp_rc_w, exp_rc_m, exp_rc_q
exp_rc_negative_d, exp_rc_negative_w,
exp_rc_negative_m, exp_rc_negative_q
```

ETF models receive the stock–ETF versions. LOO models receive independently
constructed stock–LOO versions. Both receive the same eight within-sector
state features:

```text
sector_exp_rc_d, sector_exp_rc_w, sector_exp_rc_m, sector_exp_rc_q
sector_exp_rc_negative_d, sector_exp_rc_negative_w,
sector_exp_rc_negative_m, sector_exp_rc_negative_q
```

For forecast date $t$, HAR features aggregate daily covariance and variance
components over $t-H,\ldots,t-1$, with $H\in\{1,5,21\}$, and normalize
only after summing. Exponential features use a finite 500-session window,
centers of mass $h\in\{1,5,21,63\}$, and decay
$q_h=h/(h+1)$. Full equations, downside definitions, missing-component
rules, and the 15-pair sector-state formula are in
[`bollerslev_core_features.md`](bollerslev_core_features.md).

### Dense context

```text
lagged relative stock and ETF volume
lagged six-stock sector dispersion
lagged VIX level/change
conservatively lagged 2y/5y/10y Treasury levels/changes
scheduled macro and pre-open counts
BLS and FOMC day indicators
```

`bea_release_day` is excluded because its historical coverage is incomplete.
The simplified factor-implied correlation is excluded because it is not one
of the omitted characteristic-projection features in the paper and its
historical factor inputs are not point-in-time vintages.

### Corrected volatility and extended hours

The next nested blocks add corrected lagged stock/ETF realized volatility, then
overnight, premarket, prior-aftermarket, volume, bar-count, and availability
features. Missing extended-hours values are median-imputed using training data
only; the availability fields preserve the missingness state.

Exact realized-volatility, relative-volume, dispersion, extended-hours,
market-context, and optional factor formulas are in
[`additional_quant_features.md`](additional_quant_features.md). All
`sector_*` context features refer to the configured sector ETF even when the
response is LOO.

## Locked folds

All stocks on a date remain in the same fold.

| Fold | Train | Validation | Outer test |
|---|---|---|---|
| 1 | 2022-11-01–2024-06-30 | 2024-H2 | 2025-H1 |
| 2 | 2022-11-01–2024-12-31 | 2025-H1 | 2025-H2 |
| 3 | 2022-11-01–2025-06-30 | 2025-H2 | 2026-H1 |

Scaling, imputation, regularization, early stopping, XGBoost configuration,
and eligible ensemble weights are selected without outer-test outcomes. The
v1.1 protocol record transparently restores a pre-specified validation-only
regularization-grid boundary expansion that was omitted from the initial JSON
transcription.

For a block $B=[s_B,e_B]$, the implemented eligible sets are

$$
\mathcal I_B^{(1)}
=
\{(i,t):s_B\le t\le e_B,\ z^{(1)}_{i,t}\text{ is valid}\},
$$

$$
\mathcal I_B^{(5)}
=
\{(i,t):s_B\le t\le e_B,\ z^{(5)}_{i,t}\text{ is valid},
\ \texttt{target\_end}_{i,t}\le e_B\}.
$$

Consequently, the final four forecast sessions of every train, validation,
and test block are excluded for T2. No T2 label crosses a block boundary. No
leading embargo is needed because every target begins on its own in-block
forecast date. Hyperparameters are selected on the preceding validation
block; the final outer-test fit uses the union of the already-purged train and
validation rows, with preprocessing refit on that union. All stocks from the
same date remain together. The last four training-session rows are not
restored when train and validation are later joined for the final fit.

## Completed quant-only ladder (v1)

| Rung | Targets | Estimators | Features | Status |
|---|---|---|---|---|
| 1 | T1/T2 ETF and LOO | Persistence, HAR OLS, SHAR OLS, core OLS, core LASSO | Target-specific core 22 | Complete |
| 2 | T1/T2 ETF and LOO | LASSO and Elastic Net | Core+dense; then volatility; then extended hours | Complete |
| 3 | T1/T2 ETF and LOO | Shallow XGBoost and eligible validation-weighted ensemble | Full rung-2 block | Complete; CUDA used |
| 4 | T1/T2 ETF and LOO | Bivariate Gaussian GARCH(1,1)-DCC(1,1) | Lagged daily RTH returns | Complete |

The historical quant-v1 ladder ends at rung 4. Deterministic-news work was
completed as a separately versioned, exploratory
[quant plus deterministic-news experiment](../experiments/quant_deterministic_news/v1/README.md).
It compared joint Q+D estimators with corrections to saved out-of-sample
quant forecasts without modifying the models or results above. Exact Q-only
parity replays passed. A5's T2 ETF gain versus matched Elastic Net did not beat
the stale-news placebo, and neither placebo-tested linear news specification
passed the final falsification gate. The separate
[v2 feature design](../experiments/quant_deterministic_news/v2/README.md)
has now produced and trained the available D2-Normalized deterministic branch.
No target passed every final falsification gate. The shared semantic input
corpus and the FLAN W17 and GPT R70 construction pipelines now exist, but
full-corpus semantic inference, daily W17/R70 panels, and downstream semantic
training have not occurred for the full v2 contracts. Those full workloads
remain deferred. The
[v3 cost-bounded semantic design](../experiments/quant_deterministic_news/v3/README.md)
preserves v2 and specifies separately named W17-Lite, ordered global-only G40,
and schema-incomplete factorized R70-Lite arms. The FLAN W17-Lite arm is now
complete through inference, daily aggregation, matched training, ten controls,
and final comparison; the GPT arms remain unconstructed.

## V2 news ladder

The canonical design and execution record is
[`experiments/quant_deterministic_news/v2/TRAINING_LADDER.md`](../experiments/quant_deterministic_news/v2/TRAINING_LADDER.md).
It is a separate experiment and does not create quant-v1 rungs 5 or higher.

The deterministic branch is:

| V2 rung | Estimator | Inputs | Raw columns |
|---|---|---|---:|
| `V2-D0` | Elastic Net | Matched Q56 | 56 |
| `V2-D1` | Elastic Net | D2-Normalized only | 30 |
| `V2-D2` | Elastic Net | Q56 + matched D43-recomputed | 99 |
| `V2-D3` | Elastic Net | Q56 + D2-Normalized | 86 |
| `V2-D4` | Elastic Net | Q56 + D2-Normalized + D2-Levels | 91 |
| `V2-D5` | Shallow XGBoost | Q56 + D2-Normalized | 86 |

`V2-D3` is the primary redesigned Q+D endpoint. `V2-D2` is required because
the completed D43 result used a different construction/row universe; D43 must
be recomputed on the exact D2 source profile and eligible rows. `V2-D4` is a
provider-level sensitivity, and `V2-D5` is eligible only after validation
supports the linear endpoint.

The semantic branch is repeated for all four immutable arms:

```text
W17__flan_t5_xl
W17__gpt_5_6_sol
R70__flan_t5_xl
R70__gpt_5_6_sol
```

| V2 template | Inputs | W17 raw columns | R70 raw columns |
|---|---|---:|---:|
| `V2-S0__<arm>` | Matched Q56 | 56 | 56 |
| `V2-S1__<arm>` | Q56 + D2-Normalized | 86 | 86 |
| `V2-S2__<arm>` | Q56 + L | 73 | 126 |
| `V2-S3__<arm>` | Q56 + D2-Normalized + L | 103 | 156 |
| `V2-S4__<arm>` | Shallow XGBoost on the `S3` inputs | 103 | 156 |

The decisive comparisons are Q+L versus matched Q and Q+D+L versus matched
Q+D. Counts exclude any training-fold-derived semantic missingness indicators,
which must be named and counted separately. If uncalibrated quality weights
make `rllm_mean_accepted_quality_weight` constant, it remains an audit column
and is excluded from fitting.

Only FLAN-T5-XL and GPT-5.6 Sol enter this ladder. GPT W17 is deterministically
coarsened from its fine R70 labels; FLAN W17 uses the active coarse extractor,
while FLAN R70 requires a separate fine-schema XL workflow. GPT results are
explicitly future-knowledge-contaminated oracle diagnostics, not OOS evidence.
The shared extractor view is the complete trimmed headline plus, when
available, two line feeds and a deterministic leading description excerpt of
at most 512 Unicode code points, preferring a whitespace boundary within the
last 64 code points. Candidate routing may use complete source/provider
metadata, but provider tickers, keywords, and full-source candidate roles are
not exposed to either extractor. Extractor-visible entities and role flags are
recomputed only from the bounded shared view. The pinned FLAN tokenizer must
preflight every actual prompt; silent runner-side truncation is forbidden.

The ladder reuses the three chronological date blocks only as development
evaluation, preserves T1/T2 × ETF/LOO, and refits matched Q controls on each
eligible row set. It repeats the 20-session stale, fixed within-sector
wrong-stock, semantic coverage-only, and label-permutation controls. Residual
correction remains deferred.

## V3 cost-bounded semantic execution

V3 is a separately versioned, cost-bounded successor to the unexecuted full v2
semantic arms.
Its machine-readable design is
[`config/news_semantic_lite_design_v1.json`](../config/news_semantic_lite_design_v1.json).

| V3 design | Primary workload | Fallback / audit | State |
|---|---|---|---|
| `WL17__flan_t5_xl` | Materialized role-balanced top-16 direct/sector/macro union: 50,488 unique articles | Canonical, long-description, stale, wrong-stock, coverage-only, and date-sector permutation comparisons | Inference, 17 daily features, joins, training, and final comparison complete |
| `G40__gpt_5_6_sol` | One global semantic annotation for 55,197 unique articles | Fixed 5,520-item second-view audit | Designed only |
| `R70-Lite__gpt_5_6_sol` | Future factorized article graph plus target projection | Schema, count, projection, and grounding rules still TBD | Design incomplete |

The FLAN all-history nominal estimate is 12.945 hours and 14.239 hours after
the required 10% reserve, so it currently fails the 13.5-hour
gate. Top-21 is the largest profiled selector below it at 13.44 reserved hours
and 98.57% of aggregation weight; top-16 and top-8 provide 13.03- and
11.66-hour headroom modes. All preserve 917 dates and every candidate-bearing
stock-day. Scope is derived from the
mutually exclusive C/I/P roles. The new FLAN field is a four-way event group
with `other_or_unclear` as abstention; calibrated low-text information status
is excluded. GPT remains `gpt-5.6-sol` only and retains the
future-knowledge-contaminated oracle label. Neither G40 nor future factorized
R70-Lite is semantically equivalent to exact v2 R70.

W17-Lite's canonical/reversed event-label agreement was 65.1244%, below its
frozen 85% gate, so the user-authorized training is exploratory only. Q+WL17
improved matched Q56 only for T2 ETF (1.2015% incremental MSE \(R^2\), paired
95% interval `[-0.001196, -0.000332]`) and worsened the other three targets.
That T2 ETF gain did not beat the stale-event or date-sector permutation
controls and did not decisively beat coverage-only or wrong-stock controls.
Q+D2+WL17 passed no complete useful-semantic gate. The validation-gated
Q+D2+WL17 XGBoost fit ran only for T2 ETF and improved MSE by 6.9374% versus
linear Q+D2, but it does not isolate semantic value because there is no
matched Q+D2-only XGBoost. See the
[final v3 report](../experiments/quant_deterministic_news/v3/training/comparisons/final/RESULTS.md).

## Model definitions

All supervised models are pooled across stock-date rows and use one common
coefficient function per target. The current feature blocks contain no stock
identifier or sector dummy. DCC-GARCH is the exception: it is estimated
separately for every stock-benchmark-fold system.

### OLS, LASSO, and Elastic Net

After fixed `log1p` transformations for selected nonnegative variables, a
linear prediction is

$$
\widehat z_n=\beta_0+x_n^\top\beta.
$$

OLS minimizes residual sum of squares. LASSO uses the scikit-learn objective

$$
\min_{\beta_0,\beta}
\frac{1}{2N}\sum_{n=1}^{N}
\left(z_n-\beta_0-x_n^\top\beta\right)^2
+\alpha\|\beta\|_1.
$$

Elastic Net uses

$$
\min_{\beta_0,\beta}
\frac{1}{2N}\sum_{n=1}^{N}
\left(z_n-\beta_0-x_n^\top\beta\right)^2
+\alpha\gamma\|\beta\|_1
+\frac{\alpha(1-\gamma)}{2}\|\beta\|_2^2,
$$

where $\gamma$ is `l1_ratio`. The $L_1$ term performs sparse selection;
the $L_2$ term stabilizes groups of correlated correlation and volume
features.

For tuning, missing values are replaced with training-block feature medians
and the resulting variables are standardized:

$$
x^*_{n,j}=\frac{x_{n,j}-\mu_j^{train}}{s_j^{train}}.
$$

$\alpha$ and $\gamma$ minimize validation Fisher-$z$ MSE. The base
$\alpha$ grid is
$\{10^{-4},3\!\times\!10^{-4},10^{-3},3\!\times\!10^{-3}, 10^{-2},3\!\times\!10^{-2},10^{-1}\}$, and
$\gamma\in\{0.1,0.5,0.9,1\}$. If validation selects an $\alpha$-grid
boundary, the documented outward expansion is evaluated using validation
data only. The chosen model is then refit on train plus validation before
outer-test prediction.

Rung 1 has these nested linear specifications:

$$
\begin{aligned}
\text{HAR} &: \{\rho_d,\rho_w,\rho_m\},\\
\text{SHAR} &: \text{HAR plus three downside correlations},\\
\text{Core22} &: \text{14 target-specific pair features plus 8 sector-state
features}.
\end{aligned}
$$

These HAR/SHAR/Core22 predictors are the full-day historical features,
including the reconstructed overnight component. They should not be confused
with the strict-RTH `*_rth_rc_d/w` columns used only for persistence.

Rung 2 applies both LASSO and Elastic Net to three cumulative blocks: 37
core-plus-dense features, 39 after adding volatility, and 56 after adding
extended-hours and availability features.

### Gradient-boosted trees and the ensemble

XGBoost represents the Fisher-$z$ forecast as an additive tree model,

$$
\widehat z_n=\sum_{m=1}^{M}\eta f_m(x_n),
$$

and sequentially chooses trees to reduce squared error plus tree-complexity
regularization. The four locked candidates vary shallow depth, learning rate,
minimum child weight, row and column subsampling, and $L_2$ leaf
regularization. Missing values use learned default branches; XGBoost is not
median-imputed or standardized.

Each candidate is fit on train with at most 2,000 trees and 75-round
validation early stopping. The configuration and stopping iteration with the
lowest validation MSE are refit on train plus validation. The optional
ensemble is formed in Fisher space:

$$
\widehat z^{ens}
=
w\widehat z^{XGB}+(1-w)\widehat z^{EN},
\qquad
w\in\{0,0.25,0.5,0.75,1\}.
$$

The weight is selected separately in each fold. A target exposes an ensemble
only when an interior weight strictly beats both components on every
validation fold. T1 ETF and T2 ETF passed this gate; the T1 ETF ensemble won
its outer development comparison, while standalone XGBoost remained better
for T2 ETF.

### Gaussian GARCH(1,1)-DCC(1,1)

For stock and benchmark daily RTH returns, each marginal follows

$$
\epsilon_{j,t}=100r_{j,t}-\mu_j,
\qquad
h_{j,t}
=
\omega_j+\alpha_j\epsilon_{j,t-1}^2+\beta_jh_{j,t-1},
$$

with $\omega_j>0$, $\alpha_j,\beta_j\ge0$, and
$\alpha_j+\beta_j<0.999$. Multiplication by 100 improves numerical scaling
and does not change correlation; $\mu_j$ is the estimation-sample mean.
Standardized residuals are
$u_{j,t}=\epsilon_{j,t}/\sqrt{h_{j,t}}$.

The DCC state evolves as

$$
Q_t=(1-a-b)\overline Q+a\,u_{t-1}u_{t-1}^{\top}+bQ_{t-1},
$$

$$
R_t
=
\operatorname{diag}(Q_t)^{-1/2}
Q_t
\operatorname{diag}(Q_t)^{-1/2},
\qquad
a,b\ge0,\quad a+b<0.999.
$$

The marginal Gaussian quasi-likelihood, with constants omitted, is

$$
\frac12\sum_t
\left(\log h_{j,t}+\frac{\epsilon_{j,t}^2}{h_{j,t}}\right),
$$

and the DCC step minimizes

$$
\frac12\sum_t
\left(\log|R_t|+u_t^\top R_t^{-1}u_t\right).
$$

Three optimizer starting points are tried for each marginal and DCC fit. Let
$s=t-1$ denote the last observed official return session at the forecast
origin. T1 is the one-step conditional correlation
$R_{s+1,12}=R_{t,12}$. For T2, five variance-correlation states are
recursively forecast and combined like the realized target:

$$
\widehat\rho^{DCC(5)}_t
=
\frac{
\sum_{h=1}^{5}
\widehat\rho_{s+h}
\sqrt{\widehat h_{i,s+h}\widehat h_{b,s+h}}
}{
\sqrt{
\left(\sum_{h=1}^{5}\widehat h_{i,s+h}\right)
\left(\sum_{h=1}^{5}\widehat h_{b,s+h}\right)
}
}.
$$

Beyond the first step, the unknown standardized-residual outer product is
replaced by its conditional expectation $R_{s+h}$, while each marginal
variance mean-reverts through
$\widehat h_{j,s+h+1}=\omega_j+(\alpha_j+\beta_j) \widehat h_{j,s+h}$.

Parameters are fit once before each outer block using only return information
available at its first actual test-session cutoff. The first test row may
enter parameter fitting only because its stored history is the preceding
session's lagged return. States then update causally as new lagged returns
become available; parameters are not re-estimated inside the block. DCC
produces raw conditional correlations, which are Fisher-transformed only for
the common evaluation.

## Evaluation formulas

For the $N$ concatenated outer-test stock-date predictions, define
$\widehat\rho_n=\tanh(\widehat z_n)$, Fisher-space error
$e^z_n=z_n-\widehat z_n$, and raw-space error
$e^\rho_n=\rho_n-\widehat\rho_n$. The metrics are

$$
\operatorname{RMSE}_z
=
\sqrt{\frac1N\sum_{n=1}^N(e^z_n)^2},
\qquad
\operatorname{MAE}_z
=
\frac1N\sum_{n=1}^N|e^z_n|,
$$

$$
\operatorname{RMSE}_{\rho}
=
\sqrt{\frac1N\sum_{n=1}^N(e^\rho_n)^2},
\qquad
\operatorname{MAE}_{\rho}
=
\frac1N\sum_{n=1}^N|e^\rho_n|.
$$

Benchmark-relative out-of-sample $R^2$ is

$$
R^2_{OOS}
=
1-
\frac{\sum_{n=1}^{N}(z_n-\widehat z_n)^2}
{\sum_{n=1}^{N}(z_n-z^{persist}_n)^2}.
$$

This is not ordinary in-sample $R^2$ around a sample mean. Zero matches
persistence, a positive value improves on it, and a negative value is worse.
Every stock-date receives equal weight and the aggregate metric pools the
three outer blocks; fold-specific metrics are also retained.

Fisher-$z$ RMSE is the ranking metric because the response is trained in
Fisher space. Raw-correlation errors are included for interpretation. No
standard errors or $p$-values are attached yet: consecutive T2 labels
overlap by four sessions, and stock errors share date and sector shocks.
Dependence-aware inference must preserve those structures.

## Development comparison results

The ranking metric is aggregate outer-test Fisher-$z$ RMSE. Because those
outer blocks are now used to choose among rungs, these are development
comparison estimates rather than an unbiased estimate for a final selected
model. A future period must be reserved as a new confirmatory holdout.

| Target | Best rung/model | Fisher-z RMSE | Raw-correlation RMSE | OOS $R^2$ vs persistence |
|---|---|---:|---:|---:|
| T1 ETF | Rung 3 `elastic_xgboost_ensemble` | 0.3686 | 0.2073 | 0.4154 |
| T1 LOO | Rung 3 XGBoost | 0.3814 | 0.2291 | 0.3964 |
| T2 ETF | Rung 3 XGBoost | 0.2348 | 0.1248 | 0.2885 |
| T2 LOO | Rung 3 XGBoost | 0.2567 | 0.1488 | 0.2517 |

Rung 2 improved both T1 targets over the core-only linear rung, but did not
improve the two T2 targets. XGBoost was the strongest rung overall. The
DCC-GARCH benchmark was weaker: it retained small positive value for T1 but
was below persistence on T2. This is consistent with the limitation that the
DCC benchmark models conditional dependence in one daily RTH return per
session rather than the intraday realized-covariance estimand directly.

The rung-3 ensemble also reuses each validation block for early stopping,
configuration selection, Elastic Net tuning, and ensemble-weight selection.
That does not leak the outer block, but cross-fitted validation predictions or
a separate calibration slice would give a less optimistic ensemble gate.

No statistical-significance or trading-value claim is made yet. Date-block
inference and an explicit tradable hedge evaluation remain quant-v1
follow-ups. The completed deterministic-news v1 experiment and the separately
versioned v2 design/execution record remain separate and do not extend or
renumber this completed ladder.

## Reproduction

Run the builders and trainers in this order:

```powershell
python scripts/build_bollerslev_core_features.py <locked v1 arguments>
python scripts/build_additional_quant_features.py <locked v1 arguments>
python -m scripts.correlation_training.build_targets
python -m scripts.correlation_training.audit_targets
python -m scripts.correlation_training.build_modeling_panel
python -m scripts.correlation_training.run_rung_01
python -m scripts.correlation_training.run_rung_02
python -m scripts.correlation_training.run_rung_03
python -m scripts.correlation_training.run_rung_04
python -m scripts.correlation_training.summarize_ladder
```

Exact paths, grids, feature lists, and fold dates are recorded in the protocol,
manifests, fit records, and per-rung summaries rather than inferred from this
overview.

## Remaining limitations

- The six stocks per sector are a fixed liquid universe, not point-in-time
  constituents; survivorship and membership bias remain.
- The LOO basket is five equal-weight peers, not an exact ETF-minus-stock
  basket.
- Alpaca trade bars replace TAQ midquotes.
- The three paper factor-projection features remain intentionally omitted.
- Point-in-time ETF weights, options features, SPY market controls, and safe
  earnings-timing features are still absent.
