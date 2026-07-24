# Quant training readiness and model ladder

This page records the repository state audited on **2026-07-25**. It
distinguishes:

- **ready**: implemented, covered by focused tests, materialized, and usable in
  the current matched-period panel;
- **provisional**: materialized, but requiring a documented correction,
  missing-value policy, coverage restriction, or provenance hardening before a
  final experiment; and
- **not ready**: not implemented or not materialized in a form that the
  training workflow can consume.

This distinction is important: feature files exist, but the repository does
not yet contain a target builder, joined modeling-table builder,
chronological splitter, model trainer, or forecast evaluator.
`requirements-quant.txt` likewise contains only the current numerical and
storage dependencies; it does not yet declare scikit-learn, a boosting
library, or a GARCH package.

## Audit evidence

The current generated artifacts are:

| Artifact | Current evidence |
|---|---|
| `data/features/quant/bollerslev_core_features_training.csv.gz` | 30,732 unique stock-date rows, 30 stocks, 22 complete feature columns, 2022-01-03 through 2026-06-30 |
| `data/features/quant/bollerslev_core_features.csv.gz` | Strict 500/500 robustness panel with 20,676 complete rows |
| `data/features/quant/additional_quant_features.parquet` | 79,105 unique stock-date rows and 32 feature columns, 2016-01-04 through 2026-06-30 |

For the common FLAN-period window, 2022-11-01 through 2026-06-30:

- all 27,510 stock-date rows join one-to-one between the Bollerslev training
  panel and the additional-quant panel;
- all 22 Bollerslev features are complete;
- the dense, non-extended-hours context block is complete for all 27,510 rows;
- 24,817 rows are complete across the whole extended-hours block; and
- extended-hours missingness is concentrated in less-active names, so
  complete-case deletion would change the cross-section materially.

The six quant-data and feature test modules contain 42 tests, all of which
passed in the 2026-07-25 audit.

## Bollerslev-Li-Tang feature readiness

The paper's three characteristic-projection features are outside this matrix
because the project intentionally excludes them.

| Feature family | Columns | Status | Remaining action |
|---|---|---|---|
| HAR correlation | `rc_d`, `rc_w`, `rc_m` | **Ready** | None for matched-period exploratory training |
| HAR negative semicorrelation | `rc_negative_d`, `rc_negative_w`, `rc_negative_m` | **Ready** | None for matched-period exploratory training |
| Exponential pair correlation | `exp_rc_d`, `exp_rc_w`, `exp_rc_m`, `exp_rc_q` | **Ready** | Report the 490/500 build as a near-faithful training specification, not an exact replication |
| Exponential pair negative semicorrelation | `exp_rc_negative_d`, `exp_rc_negative_w`, `exp_rc_negative_m`, `exp_rc_negative_q` | **Ready** | Same disclosure as above |
| Sector-state exponential correlation | `sector_exp_rc_d`, `sector_exp_rc_w`, `sector_exp_rc_m`, `sector_exp_rc_q` | **Ready** | Disclose the fixed six-stock sector groups |
| Sector-state exponential negative semicorrelation | `sector_exp_rc_negative_d`, `sector_exp_rc_negative_w`, `sector_exp_rc_negative_m`, `sector_exp_rc_negative_q` | **Ready** | Disclose the fixed six-stock sector groups |

Two provenance tasks remain before a locked final experiment:

1. The current core manifests record
   `actual_hashes_verified: false`. Rebuild the final artifacts with
   `--verify-input-hashes`.
2. The materialized core panels begin in 2022. A long-history quant-only
   experiment should rebuild the same features from the earliest
   post-500-session warm-up date; the raw regular-session bars already begin
   in 2016.

## Additional quant feature readiness

### Ready dense block

These columns are materialized, leakage-lagged, complete over the matched
2022-11-01 through 2026-06-30 panel, and ready for an initial model:

```text
lagged_relative_daily_volume_20d
sector_lagged_relative_daily_volume_20d
lagged_sector_return_dispersion
vix_lag1
vix_change_lag1
treasury_2y_lag2
treasury_2y_change_lag2
treasury_5y_lag2
treasury_5y_change_lag2
treasury_10y_lag2
treasury_10y_change_lag2
scheduled_macro_event_count
scheduled_preopen_macro_count
bls_release_day
bea_release_day
fomc_decision_day
```

The BEA flag is structurally zero before the live machine-readable calendar
begins in 2025. Treat it as incomplete historical coverage rather than as
evidence that no BEA event occurred.

### Provisional block

| Feature family | Columns | Why provisional | Required treatment |
|---|---|---|---|
| Lagged realized volatility | `lagged_realized_volatility`, `sector_lagged_realized_volatility` | The current aggregation differences consecutive bar closes and therefore omits the first regular-session bar's open-to-close return | Correct the aggregation, add a regression test with a nonzero first-bar return, and rebuild |
| Current premarket | `stock_overnight_return`, `premarket_return`, `premarket_volume`, `relative_premarket_volume_20d`, `premarket_bar_count` and sector equivalents | Economically usable, but sparse when no qualifying extended-hours bars exist | Add explicit availability indicators and choose the imputation policy on training data only |
| Overnight spread | `stock_minus_sector_overnight_return` | Missing whenever either stock or sector premarket input is missing | Same treatment as the premarket block |
| Prior aftermarket | `prior_aftermarket_return`, `prior_aftermarket_volume`, `prior_relative_aftermarket_volume_20d` | Mostly complete in the matched period, but still undefined on no-trade windows | Add availability indicators; never replace missing returns or volumes with zero without distinguishing no trade from a true zero |
| Simplified factor correlation | `factor_implied_correlation` | Not the paper's characteristic-projection feature; current Ken French files may contain revisions; 2.29% missing in the matched period | Use only as a separately labeled robustness feature |

The current additional-feature manifest records input path lists but not input
or output hashes, builder parameters, or column-level coverage. Strengthen
that manifest before the final locked run.

### Not ready

The following planned or useful quant inputs are not yet ready:

- a reported-earnings indicator: the cached Alpha Vantage archive covers only
  20 of 30 stocks and does not contain announcement time or the schedule as it
  was known before the event;
- a DCC-GARCH forecast;
- broad-market controls derived from the downloaded SPY bars, such as market
  return, market volatility, stock-market beta, or sector-market correlation;
- dynamic stock-sector beta or a Kalman-filter beta;
- options-implied volatility, options volume, skew, or put-call features;
- point-in-time ETF constituent weights and a mechanical self-inclusion
  control; and
- point-in-time sector membership. The current universe is a fixed research
  panel.

## Targets: current status and ladder

No explicit target panel is currently materialized.

### T0: paper-style stock-ETF correlation proxy

**Status: derivable now for an end-to-end smoke test, but not implemented as
an audited target.**

For a stock row dated \(t\), the same stock's `rc_d` on the next official
forecast-date row is the realized paper-style stock-ETF correlation from
session \(t\). This target includes the prior-close-to-open component and
regular-session returns. It is useful for checking joins, splits, scaling,
and model code, but it is not the preferred economic target.

### T1: same-day regular-session stock-ETF correlation

**Status: not ready; implement first.**

Construct synchronized 15-minute stock and ETF returns from 09:30 through the
official close, including the first bar's open-to-close return. Store the
realized covariance and both realized variances in addition to correlation.
Create:

```text
target_rth_correlation
target_fisher_z
target_realized_covariance
target_stock_realized_variance
target_sector_realized_variance
```

The primary regression target should be:

\[
z_{i,t} =
\operatorname{atanh}
\left(
  \operatorname{clip}(\rho_{i,t}, -0.995, 0.995)
\right).
\]

### T2: next-five-session realized correlation

**Status: not ready.**

Aggregate covariance and variance components over the next five sessions and
normalize afterward. Do not average five daily correlations. This is the
lower-noise robustness target.

### T3: high-versus-low coupling regime

**Status: not ready.**

Derive a classification target from T1 or T2 using a threshold estimated
from the training window only, such as a stock-specific rolling median. This
is a secondary task, not a replacement for continuous correlation forecasts.

### T4: leave-one-out sector-basket correlation

**Status: not ready; preferred research target once implemented.**

The current ETF target mechanically contains the stock when that stock is an
ETF constituent. Build an equal-weight or point-in-time-weighted sector
basket excluding the target stock. Keep stock-ETF correlation as the
tradable-hedge target and leave-one-out correlation as the cleaner scientific
coupling target.

## Training ladder

Every rung must use the same target observations and chronological folds.
All stocks on the same date stay in the same fold. Scaling, imputation,
feature selection, and hyperparameter choice use training data only.

| Rung | Target | Estimator | Feature block | Purpose | Can run now? |
|---|---|---|---|---|---|
| 0A | T0 proxy | Persistence | Prior `rc_d` | Verify target alignment and evaluation code | **After minimal target/table code** |
| 0B | T0 proxy | OLS | `rc_d`, `rc_w`, `rc_m` | HAR smoke baseline | **After minimal target/table code** |
| 0C | T0 proxy | LASSO | All 22 Bollerslev columns | Exercise sparse modeling end to end | **After minimal target/table code** |
| 1A | T1 | Persistence and expanding historical mean | Prior correlation only | Main statistical baselines | No: T1 missing |
| 1B | T1 | OLS | HAR 3, then SHAR 6 | Published-logic baseline ladder | No: T1 missing |
| 1C | T1 | OLS and LASSO | All 22 Bollerslev columns | Primary paper-style quant benchmark | No: T1 missing |
| 2A | T1 | LASSO and Elastic Net | 22 core + ready dense extra block | Primary practical quant model | No: T1/table builder missing |
| 2B | T1 | Elastic Net | Rung 2A + corrected lagged volatility | Test volatility's incremental value | No: volatility correction missing |
| 2C | T1 | Elastic Net | Rung 2B + extended-hours block + missingness indicators | Test point-in-time premarket information | No: missing-data contract missing |
| 3A | T1 | Shallow gradient-boosted trees | Same features as 2C | Nonlinear challenger | No: trainer/splits missing |
| 3B | T1 | Validation-weighted linear/tree average | Forecasts from 2C and 3A | Use only if errors are complementary out of sample | No |
| 4 | T1 and T2 | DCC-GARCH | Return history | Established covariance/correlation benchmark | No: DCC missing |
| 5 | T1 and T2 | Best validated quant estimator | Quant + deterministic daily news | Test observable news activity and breadth | No: production daily-news panel missing |
| 6 | T1 and T2 | Same estimator and folds as rung 5 | Quant + deterministic + validated LLM features | Isolate incremental semantic value | No: production LLM extractor/panel not ready |

The primary scientific comparison is rung 6 versus rung 5, not rung 6 versus
persistence. Until a stronger semantic extractor passes its external
evaluation gate, the current FLAN and Llama outputs belong only in a
separately labeled weak-feature ablation.

## Missing training infrastructure

Before any reported training result, implement and test:

1. an explicit target builder with interval-alignment and early-close tests;
2. a joined modeling-table builder with one-to-one key assertions and a
   feature-availability report;
3. expanding or rolling date-block splits, plus a purge/embargo equal to the
   target horizon;
4. training-only scaling, imputation, missingness indicators, and
   hyperparameter selection;
5. persistence, historical-mean, HAR, SHAR, OLS, LASSO, and Elastic Net
   estimators;
6. MAE and RMSE in Fisher-\(z\) and raw-correlation space, out-of-sample
   \(R^2\), date-aggregated loss comparisons, and calibration by forecast
   decile;
7. date- or sector-date-block bootstrap inference; and
8. hedge evaluation using forecast covariance/variance components, hedge
   variance, turnover, and tail errors.

Do not describe the repository as training-ready end to end until at least
items 1 through 6 exist and pass focused tests.
