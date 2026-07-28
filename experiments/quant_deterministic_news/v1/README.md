# Quant plus deterministic-news training experiment v1

Status: **complete exploratory development experiment**. See
[`STATUS.md`](STATUS.md) for the live bundle ledger.

> **Frozen historical design.** This document records the exact D43 features
> and models that were actually trained. The proposed redesign is separate in
> [v2](../v2/README.md); it does not revise this experiment's protocol,
> artifacts, or conclusions. Forward-looking language below is retained as
> historical methodology unless the completed-outcome section says otherwise.

This is a separate exploratory experiment. It does not add rungs to, overwrite,
or reinterpret the completed
[quant-only v1 ladder](../../quant_training/v1/README.md). The
machine-readable protocol that was proposed and then locked before training is
[`config/quant_deterministic_news_protocol_v1.json`](../../../config/quant_deterministic_news_protocol_v1.json).
Training artifacts are written only to this experiment's bundle directories
and `outputs/quant_deterministic_news/v1`; quant-only v1 remains read-only.

The protocol is now locked before training. Changing its feature sets,
selection rules, or evaluation order after a result is observed requires a
dated amendment; progress and results belong in `STATUS.md` and the individual
model bundles rather than edits to this methodology.

## Completed outcome

All joint, residual, winner-base sensitivity, and placebo bundles completed.
The matched Q-only replays were exact: A0-L and A0-T reproduced all 44,040
frozen predictions with zero numerical difference, and every XGBoost fit
confirmed CUDA execution.

The main findings are:

- Full Q56+D43 Elastic Net (A5) improved T2 ETF Fisher-z MSE by 3.65% versus
  matched Q-only Elastic Net; its 10-session date-block interval for paired
  loss was `[-0.003467, -0.001474]`.
- A5 did not beat the 20-session stale-news placebo on any target by point
  estimate. It did beat the fixed wrong-stock placebo, so target assignment
  contains some information, but the gain is not uniquely attributable to
  timely news.
- Full Q56+D43 XGBoost (A6) had no matched-base interval excluding zero.
- Residual Elastic Net (B4) and residual XGBoost (B5) did not robustly improve
  the primary frozen XGBoost base.
- The T1 ETF winner-base S-B5 sensitivity improved MSE by 0.91%, with paired
  loss interval `[-0.002512, -0.000172]`. It remains descriptive because no
  matching nonlinear placebo suite was preregistered.
- Neither placebo-tested linear specification, A5 or B4, passed the
  conservative final-stage matched-base, calibration/placebo, and
  fold-consistency gate.

The observed-no-news slice has only 66 Track-A rows and 12 Track-B rows and no
fold-3 observations, so it is descriptive only. The practical decision is to
retain the quant-only model and treat all deterministic-news results as
hypothesis-generating until a prospective, version-preserving news sample is
available.

The complete 118 paired target comparisons, 236,000 fold-contained bootstrap
draws, model table, decision gate, and slice results are in the
[final results report](comparisons/final/RESULTS.md).

## Decision summary

Two architectures should be tested:

1. **Joint forecast:** fit the target directly from the quant and
   deterministic-news features.
2. **Residual correction:** leave the saved quant forecast unchanged and fit a
   chronologically valid correction from earlier out-of-sample quant errors.

The joint architecture must be fitted because its design matrix changes. It
does **not** require a shorter calendar than quant v1: both the completed quant
panel and the joined Q+D panel contain the same 27,510 rows from 2022-11-01
through 2026-06-30. A matched Q-only replay remains an important parity
control, but not because quant v1 used more history.

The residual architecture can compare directly with saved quant predictions,
provided the correction is learned only from earlier **out-of-sample** quant
errors. It must never learn from residuals of a base model fitted on those
same observations.

## Readiness and claim boundary

The source panel is
`data/features/q_plus_d/massive_v1/modeling_panel_q_plus_d.parquet`, bound by
SHA-256
`352ba2635febfa0bc36bc6f797c00036a8aee9879327a8176e9accf180e50150`.
The [joined-panel documentation](../../../docs/q_plus_d_massive.md) reports:

| Component | Current status |
|---|---|
| Rows and joins | 27,510/27,510 quant rows matched one-to-one |
| Coverage | 30 stocks, five sectors, 917 dates |
| Targets | T1/T2 for ETF and LOO |
| Quant inputs | 56 target-specific inputs |
| Deterministic inputs | 44 materialized; 40 recommended |
| Derived missingness flags | Three required at fit time |
| Full joint design | 99 inputs |
| Strict point-in-time news | **Not available** |

T1 ETF and LOO are valid on all 27,510 rows. T2 ETF and LOO are valid on
27,390 rows; the final four dates cannot start a complete five-session target.

The reusable quant controls are also pinned:

| Artifact | SHA-256 |
|---|---|
| Rung-2 predictions | `a1812ccf17096f38e2262072136579608e6e21c224ff6d4e5e92d6e1fadfe3fa` |
| Rung-3 predictions | `9c5355aad392c4902838d407748ba8daf39a8ae7cef02d3c4577f604898a857e` |
| Quant-v1 winner summary | `056d960ce2c42aa757a4aaf43247337af7e19cb42db060e89261cba87f3026b9` |

This experiment is mechanically ready but scientifically **exploratory**.
Massive ordinary News lacks first-seen timestamps, update history, and
historical article versions. Untickered macro coverage is incomplete.
Publication time and the currently returned description are proxies for what
was known historically. Results may support a weak-data/provider-feasibility
ablation, but not a strict point-in-time, causal, or production claim.

The following inputs remain unavailable or intentionally excluded:

- the paper's three point-in-time characteristic-projection factors;
- historical ETF weights and effective-dated sector membership;
- options/implied-volatility features;
- leakage-safe historical earnings announcement timing;
- SPY-derived market controls, which have not been emitted;
- complete untickered macro-news coverage; and
- validated LLM-derived daily features.

## Targets and folds

Train four separate pooled stock-date models:

```text
T1 ETF
T1 LOO
T2 ETF
T2 LOO
```

The response is the existing Fisher transform:

$$
z_{i,t}
=
\operatorname{atanh}\!\left(
\operatorname{clip}(\rho_{i,t},-0.995,0.995)
\right).
$$

Track A reuses the three quant-v1 train/validation/test blocks and the existing
T2 boundary purge. For every T2 train, validation, and test block, require the
applicable `target_*_t2_end_date` to be no later than that block's end; this
removes the final four forecast sessions and those rows remain excluded when
train and validation are joined for the outer-test refit. All stocks on a date
remain in the same block. Track B has a narrower exact-reuse design described
below.

Expected train/validation/test row counts are:

| Fold | T1 rows | T2 rows after boundary purge |
|---|---:|---:|
| 1 | 12,480 / 3,840 / 3,660 | 12,360 / 3,720 / 3,540 |
| 2 | 16,320 / 3,660 / 3,840 | 16,200 / 3,540 / 3,720 |
| 3 | 19,980 / 3,840 / 3,690 | 19,860 / 3,720 / 3,570 |

## Feature contract

### Quant block: Q56

Every target uses 56 quant inputs:

- 14 target-specific pair-history features;
- eight common sector-state features;
- 15 dense market and macro features;
- two lagged realized-volatility features; and
- 17 extended-hours and availability features.

ETF targets use:

```text
etf_rc_d, etf_rc_w, etf_rc_m
etf_rc_negative_d, etf_rc_negative_w, etf_rc_negative_m
etf_exp_rc_d, etf_exp_rc_w, etf_exp_rc_m, etf_exp_rc_q
etf_exp_rc_negative_d, etf_exp_rc_negative_w
etf_exp_rc_negative_m, etf_exp_rc_negative_q
```

LOO targets replace `etf_` with `loo_`. Both use:

```text
sector_exp_rc_d, sector_exp_rc_w, sector_exp_rc_m, sector_exp_rc_q
sector_exp_rc_negative_d, sector_exp_rc_negative_w
sector_exp_rc_negative_m, sector_exp_rc_negative_q

lagged_relative_daily_volume_20d
sector_lagged_relative_daily_volume_20d
lagged_sector_return_dispersion
vix_lag1, vix_change_lag1
treasury_2y_lag2, treasury_2y_change_lag2
treasury_5y_lag2, treasury_5y_change_lag2
treasury_10y_lag2, treasury_10y_change_lag2
scheduled_macro_event_count, scheduled_preopen_macro_count
bls_release_day, fomc_decision_day

lagged_realized_volatility
sector_lagged_realized_volatility

stock_overnight_return
premarket_return, premarket_volume
relative_premarket_volume_20d, premarket_bar_count
sector_overnight_return
sector_premarket_return, sector_premarket_volume
sector_relative_premarket_volume_20d, sector_premarket_bar_count
stock_minus_sector_overnight_return
prior_aftermarket_return, prior_aftermarket_volume
prior_relative_aftermarket_volume_20d
stock_premarket_available
sector_premarket_available
stock_prior_aftermarket_available
```

These are exactly the full rung-2/rung-3 quant inputs. Do not add target
components, identifiers, audit fields, `bea_release_day`, or the optional
factor-implied correlation.

### Deterministic block: D43

The joined file contains 44 deterministic columns, but only 40 are recommended
for fitting. Add three missingness indicators at model time, giving D43.

#### D1: activity, breadth, and commonality (18)

```text
observed_relevant_article_count
observed_direct_target_article_count
observed_target_only_article_count
observed_peer_article_count
observed_peer_specific_article_count
observed_common_proxy_article_count
observed_macro_article_count
observed_mixed_target_common_article_count
observed_no_relevant_news
observed_multi_ticker_article_share
observed_target_peer_co_mention_share
observed_unique_peer_count
observed_peer_coverage_ratio
observed_sector_firms_with_news_share
observed_sector_entity_hhi
observed_target_share_of_sector_entity_mentions
observed_common_shock_balance
observed_firm_common_imbalance
```

#### D2: high-precision lexical cues (8)

```text
positive_surprise_cue_article_count
negative_surprise_cue_article_count
earnings_guidance_cue_article_count
product_demand_cue_article_count
supply_capacity_cue_article_count
regulation_legal_cue_article_count
corporate_analyst_cue_article_count
macro_market_cue_article_count
```

#### D3: timing and recency (6)

```text
hours_since_latest_precise_target_article
hours_since_latest_precise_common_article
recency_weighted_precise_target_count_12h
recency_weighted_precise_common_count_12h
same_day_premarket_precise_article_share
prior_session_afterhours_precise_article_share
```

The first two fields are nullable. Add:

```text
missing_hours_since_latest_precise_target_article
missing_hours_since_latest_precise_common_article
```

#### D4: source, title propagation, and burst (8)

```text
unique_source_count
source_entropy
normalized_title_cluster_count
normalized_title_duplicate_ratio
max_normalized_title_cluster_size
max_normalized_title_cluster_source_count
press_release_wire_source_share
observed_target_news_burst_60_session
```

The burst is nullable. Add:

```text
missing_observed_target_news_burst_60_session
```

Exclude the two constants:

```text
timing_imprecise_article_count
government_primary_source_share
```

and the two exact redundancies:

```text
timing_eligible_article_count
timing_eligible_share
```

All 13 provenance/audit columns and `news_row_matched` are also excluded from
the model matrix.

### Preprocessing

Create missingness indicators before imputation. For linear models, fit
medians and standardization parameters using the current training partition
only. Use fixed `log1p` transformations for nonnegative article/cue counts,
recency-weighted counts, positive recency ages, source counts, and title
cluster counts/sizes. Leave shares, HHI, entropy, signed balances, and the
signed burst score untransformed.

XGBoost keeps native missing values and also receives the three explicit
news-missingness indicators. The existing quant transformations remain
unchanged.

Reuse the quant-v1 tuning budgets: the existing Elastic Net alpha and
`l1_ratio` grids, validation-only boundary expansion, four shallow XGBoost
candidates, at most 2,000 trees, 75-round early stopping, and CUDA when
available. This controls search-budget differences between Q-only and Q+D.

## Track A: joint target prediction

The joint model estimates

$$
\widehat z^{joint}_{i,t}
=
f(Q_{i,t},D_{i,t}).
$$

Use Elastic Net for the nested feature ladder because it is interpretable and
handles correlated count/share families. Use the existing shallow XGBoost
grid only for the Q-only and full-Q+D endpoints.

| Rung | Estimator | Inputs | Count | Purpose |
|---|---|---|---:|---|
| A0-L | Frozen/parity-replayed Elastic Net | Q56 | 56 | Linear quant control |
| A0-T | Frozen/parity-replayed XGBoost | Q56 | 56 | Nonlinear quant control |
| A1 | Elastic Net | D43 | 43 | News-only diagnostic |
| A2 | Elastic Net | Q56 + D1 | 74 | Activity/scope increment |
| A3 | Elastic Net | Q56 + D1 + D2 | 82 | Lexical-cue increment |
| A4 | Elastic Net | Q56 + D1 + D2 + D3 + two flags | 90 | Timing increment |
| A5 | Elastic Net | Q56 + D43 | 99 | Full joint linear model |
| A6 | Shallow XGBoost | Q56 + D43 | 99 | Full joint nonlinear model |
| A7 | Optional EN/XGB blend | Q56 + D43 | 99 | Validation-gated combination |

The optional blend is eligible only if an interior weight strictly beats both
components on every validation fold, matching the quant-v1 rule.

A0 may import the saved quant predictions after exact key, target, model,
source-hash, and row-count checks. A Q-only replay is still recommended as a
parity gate: if it differs materially from the frozen artifacts, stop and
resolve the implementation difference before interpreting A2-A7.

## Track B: residual correction with frozen quant forecasts

For a saved out-of-sample quant forecast, define

$$
e^{Q}_{i,t}=z_{i,t}-\widehat z^{Q}_{i,t},
$$

then fit

$$
\widehat z^{corrected}_{i,t}
=
\widehat z^{Q}_{i,t}
+
g(D_{i,t},\widehat z^{Q}_{i,t}).
$$

The primary base is the saved rung-3 XGBoost forecast for all four targets.
Using one base family avoids changing the residual task by target. A secondary
sensitivity will repeat B0-B5 for T1 ETF using its recorded ensemble winner;
the other three target winners are already XGBoost and therefore equal the
primary base.

The saved files contain outer-test forecasts only, beginning in 2025.
Therefore the initial no-quant-refit design is prequential:

1. Split fold-1 outer predictions into 2025-Q1 residual-development training
   and 2025-Q2 validation. Purge T2 labels crossing 2025-03-31 or the fold-1
   endpoint.
2. Select correction settings using only that internal validation portion.
3. Refit on all fold-1 out-of-sample errors and predict corrections for fold 2.
4. Keep settings fixed, refit on fold-1 plus fold-2 out-of-sample errors, and
   predict corrections for fold 3.
5. Report the corrected-versus-unchanged comparison on folds 2 and 3 only.

This directly compares against the already trained quant forecasts without
fitting the correction to in-sample base residuals. It cannot produce a valid
fold-1 correction. An all-three-fold residual study would require new
forward-chained, cross-fitted base forecasts within every original training
pool and is outside this initial reuse experiment.

The internal fold-1 development split contains 1,800/1,860 T1
train/validation rows per target. After the same five-session boundary purge,
T2 contains 1,680/1,740 rows.

| Rung | Correction model | Inputs | Purpose |
|---|---|---|---|
| B0 | Zero correction | Frozen quant forecast | Exact base control |
| B1 | Historical mean error | Intercept | Bias-only control |
| B2 | Linear calibration | Frozen quant forecast | Calibration-only control |
| B3 | Elastic Net | D43 | News-only residual |
| B4 | Elastic Net | Frozen quant forecast + D43 | State-aware linear residual |
| B5 | Shallow XGBoost | Frozen quant forecast + D43 | Nonlinear residual challenger |

Every predeclared B rung will be reported. No single residual model will be
selected and relabeled as a winner after fold-2 or fold-3 results are seen.
For B5, the XGBoost candidate and best iteration selected on the fold-1
Q1-to-Q2 development split remain fixed for both later refits.

Track A can report all three original folds. Track B reports folds 2 and 3.
Compare the two architectures only on their common fold-2/fold-3 keys; never
compare Track A's three-fold pooled RMSE with Track B's two-fold pooled RMSE.
Even on common keys, this is a comparison of final forecast performance, not
a controlled architecture comparison: Track A learns from each original
train/validation history, while Track B can learn corrections only from
earlier saved outer-test residuals beginning in 2025. A clean joint-versus-
residual architecture test would require new forward-chained out-of-fold quant
forecasts throughout every Track-A fit pool, followed by training both
architectures on the same eligible dates.

## Evaluation and controls

Retain the quant-v1 metrics:

```text
Fisher-z RMSE and MAE
raw-correlation RMSE and MAE
OOS R-squared versus persistence
```

The primary incremental metric is

$$
R^2_{news\mid Q}
=
1-
\frac{
\sum_n(z_n-\widehat z^{Q+D}_n)^2
}{
\sum_n(z_n-\widehat z^{Q}_n)^2
}.
$$

Also report the paired Fisher-space squared-loss difference,

$$
\Delta L
=
\frac{1}{N}\sum_n
\left[
(z_n-\widehat z^{Q+D}_n)^2
-
(z_n-\widehat z^{Q}_n)^2
\right],
$$

where a negative value favors news. Use date-block bootstrap intervals that
resample whole dates with all stocks together. Build blocks separately within
each outer fold so a resample cannot cross a model-refit boundary. Use 2,000
moving-block resamples, a ten-session block, and seed 1729; the ten-session
block covers the five-session overlap in T2. Report fold-specific results,
news-day versus observed-no-news slices, and improvement consistency rather
than only one pooled number.

Required controls:

- D43-only direct forecast;
- intercept-only and forecast-only residual calibration;
- deterministic features lagged by 20 official sessions;
- a fixed same-date wrong-stock rotation within each sector; and
- nested D1/D2 versus full-D43 ablations.

The two placebo feature constructions will be fitted for A5 and B4, which
tests them under the principal linear joint and residual architectures without
expanding the nonlinear search after results are observed.

A useful news result must improve the matched quant base, beat calibration and
placebo controls, and not depend entirely on one fold.

## Interpretation limits

The quant-v1 outer blocks were already inspected while selecting the existing
ladder. Every result from this experiment is therefore a development estimate,
even when the residual correction itself does not inspect the later block.

Confirmation requires observations strictly after 2026-06-30 and a
version-preserving news archive. No result from this experiment should be
described as production-ready, causal, or a strict point-in-time news
backtest.
