# Methodology

This document describes the methodology behind the selected long-Q models and
the RRES-C6 semantic overlay. Frozen implementation details remain in the
versioned protocols linked below.

## Universe and information cutoff

The modeling universe contains 30 liquid U.S. stocks across five sectors,
with one sector ETF and five non-target peer stocks available for each stock.
Price inputs are derived from 15-minute Alpaca SIP bars adjusted for
corporate actions. Targets use each official regular session, including early
closes; additional features also use the defined extended-hours windows.

The raw price window begins in 2016. Core features require a 500-session
warm-up, so the modeling panel begins on 2017-12-28 and ends on 2026-06-30.
Completed-session features are lagged. Same-day premarket and news inputs
use information available before the 09:00 ET forecast cutoff, with additional
publication lags applied to slower market-context sources.

## Targets

For intraday intervals $`m`$ on date $`t`$, realized stock/reference correlation
is formed from consistently aligned return intervals. The response is clipped
away from $`\pm1`$ and transformed as

```math
z_{i,t}=\mathrm{atanh}(\rho_{i,t}).
```

The four target series are T1 ETF, T1 LOO, T2 ETF, and T2 LOO. T1 is a
same-session correlation target. T2 aggregates covariance and variance
components over the current and next four official sessions before taking
the correlation. ETF uses the tradable sector fund; LOO uses an equal-weight
basket of the other five sector stocks.

Fisher-z space is the primary training and selection scale. Raw-correlation
MAE and RMSE are reported after applying $`\tanh`$ to predictions.

## Long-Q features

The core-22 block contains:

- six HAR/downside realized-correlation features;
- eight finite-500-session exponential stock/reference pair features; and
- eight within-sector exponential state features.

The extended quantitative blocks add overnight and premarket returns,
relative volumes, realized volatility, sector dispersion, market and rate
context, release-calendar indicators, earnings state, and simplified
factor-implied correlation. Availability indicators distinguish a genuine
zero from a missing source. Early folds cannot learn from extended-hours
features before those data begin in November 2022.

## Model ladder

The four long-Q rungs compare:

| Rung | Families |
|---|---|
| 1 | Persistence, HAR OLS, SHAR OLS, core-22 OLS, core-22 LASSO |
| 2 | LASSO and Elastic Net over increasingly rich quantitative blocks |
| 3 | Shallow XGBoost and validation-qualified Elastic Net/XGBoost ensemble |
| 4 | Causal DCC-GARCH benchmark |

All tunable preprocessing and model choices are selected within each fold.
The final test block is excluded from preprocessing and hyperparameter
selection.

## Expanding-window evaluation

Quant v2 fixes 13 six-month outer tests from 2020-H1 through 2026-H1. Each
outer test is preceded by a six-month validation block and an expanding
training block. Every stock on a date stays in one partition. T2 rows are
purged whenever their forward five-session target crosses a validation or
test boundary.

The pooled target-level selection metric is Fisher-z RMSE:

```math
\mathrm{RMSE}_z=
\sqrt{\frac{1}{N}\sum_{i,t}(z_{i,t}-\widehat z_{i,t})^2}.
```

Out-of-sample improvement over persistence is

```math
R^2_{\mathrm{OOS}}=
1-\frac{\sum_{i,t}(z_{i,t}-\widehat z_{i,t})^2}
{\sum_{i,t}(z_{i,t}-\widehat z^{\mathrm{persist}}_{i,t})^2}.
```

Because the outer tests were used to rank model families, these are
development estimates rather than an untouched confirmation.

## Semantic score construction

V4 reuses cached FLAN-T5-XL candidate log scores from canonical and reversed
option orders. Within each order, scores are normalized with a softmax; the
consensus weight is the arithmetic mean:

```math
p_{a,k}=\frac{1}{2}\left[
\mathrm{softmax}(s^{\mathrm{can}}_a)_k+
\mathrm{softmax}(s^{\mathrm{rev}}_a)_k
\right].
```

These weights preserve relative uncertainty but are not calibrated class
probabilities.

Articles are routed relative to each target stock as target-idiosyncratic
($`I`$), peer-idiosyncratic ($`P`$), or common ($`C`$). With frozen article weight
$`w_{i,a,t}`$, the role/event mass is

```math
M_{i,t,r,k}=
\frac{\sum_a w_{i,a,t}\,\mathbf{1}[r_{i,a,t}=r]p_{a,k}}
{\sum_a w_{i,a,t}}.
```

The denominator includes all selected roles and all four score classes.
`other_or_unclear` remains in the denominator but is not a primary event
predictor.

For each mass, a prior-only 63-session exponentially weighted baseline with a
21-session half-life is

```math
B_{i,t,r,k}=
\frac{\sum_{h=1}^{63}2^{-(h-1)/21}M_{i,t-h,r,k}}
{\sum_{h=1}^{63}2^{-(h-1)/21}},
\qquad
\Delta M_{i,t,r,k}=M_{i,t,r,k}-B_{i,t,r,k}.
```

The current session never enters its own baseline.

## RRES-C6

For each of three event families, Coupling6 forms a current and innovation
contrast:

```math
K_{i,t,k}=M_{i,t,C,k}-M_{i,t,I,k}-M_{i,t,P,k},
```

```math
\Delta K_{i,t,k}=\Delta M_{i,t,C,k}
-\Delta M_{i,t,I,k}-\Delta M_{i,t,P,k}.
```

An Elastic Net predicts the residual of a saved, already out-of-sample
long-Q XGBoost forecast. For test fold $`j`$, only earlier OOS residual blocks
through $`j-2`$ are training data; fold $`j-1`$ is validation. The final
correction is refit on training plus validation. Validation selects both
Elastic-Net hyperparameters and correction shrinkage
$`\lambda\in\{0,0.25,0.5,0.75,1\}`$.

## Paired inference

Candidate/base comparisons use squared Fisher-z loss. Incremental MSE R² is

```math
R^2_{\mathrm{inc}}=
1-\frac{\mathrm{MSE}_{\mathrm{candidate}}}
{\mathrm{MSE}_{\mathrm{base}}}.
```

Confidence intervals use 2,000 paired moving-block resamples of whole
forecast dates with ten-session blocks. Blocks are sampled separately within
each outer fold and cannot cross a regime boundary. The RRES-C6 matched test
requires a favorable point loss, an upper 95% loss-delta bound below zero,
and improvement in at least three of five folds.

## Claim limits

The stored tests establish retrospective development performance. Strict
confirmation requires a future period that was not used for design or model
selection, plus a version-preserving point-in-time news source. No result in
this repository is a production or investment-performance claim.
