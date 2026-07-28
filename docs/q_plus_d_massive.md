# Massive quant + deterministic-news panel

> **Completed v1 artifact.** This document preserves the exact deterministic
> feature construction used by the completed Q+D v1 experiment. The proposed
> normalized deterministic redesign and the two semantic feature contracts are
> documented separately in
> [`experiments/quant_deterministic_news/v2/README.md`](../experiments/quant_deterministic_news/v2/README.md).
> Their matched Q+D, Q+L, and Q+D+L ladder is in the
> [v2 training ladder](../experiments/quant_deterministic_news/v2/TRAINING_LADDER.md).
> The v2 D2 panel and available-data deterministic branch are complete; the
> semantic panels remain construction-blocked and untrained.

`scripts/join_quant_deterministic_news.py` joins the frozen quant modeling
panel to the Massive deterministic stock-day features on:

```text
forecast_date, sector, stock, benchmark
```

The join is one-to-one and left-preserving. It rejects duplicate keys,
payload-column collisions, source-hash mismatches, incomplete
stock/benchmark/control collections, and—by default—any unmatched quant row.
It never treats a missing news row as zero news.

Build the deterministic panel from both cursor-complete query roots, then join
it to the frozen quant panel:

```powershell
$trainingPython = '.\.venv-training\Scripts\python.exe'

& $trainingPython scripts\build_massive_deterministic_news_features.py `
  --raw-root data/raw/massive/ordinary_news/free_v1 `
  --raw-root data/raw/massive/ordinary_news/free_v1_benchmarks `
  --overwrite

& $trainingPython scripts\join_quant_deterministic_news.py --overwrite
```

The generated local artifacts are:

```text
data/features/q_plus_d/massive_v1/modeling_panel_q_plus_d.parquet
data/features/q_plus_d/massive_v1/manifest.json
data/features/q_plus_d/massive_v1/manifest.sha256
```

The materialized panel contains 27,510 rows and 170 columns over 917 trading
dates from 2022-11-01 through 2026-06-30. It covers 30 stocks across five
sectors. All 27,510 quant rows have exactly one news match; all 112 quant
columns, values, dtypes, and row order are preserved. The joined additions are
44 materialized deterministic-news columns, 13 audit fields, and one match
flag. The manifest's modeling preflight identifies two constant columns and
two exact timing redundancies, leaving 40 recommended fitting columns.

The manifest binds both source artifacts and manifests, the builder, the
output, and the ordered schema with SHA-256 hashes. `manifest.sha256` binds the
manifest itself. The current panel SHA-256 is
`352ba2635febfa0bc36bc6f797c00036a8aee9879327a8176e9accf180e50150`.

Before fitting, use
`manifest.json -> modeling_preflight.recommended_feature_columns`. The two
constant columns are `timing_imprecise_article_count` and
`government_primary_source_share`. `timing_eligible_article_count` is exactly
equal to `observed_relevant_article_count`, while `timing_eligible_share` is
exactly `1 - observed_no_relevant_news` in this archive. Keeping all four in
the Parquet file preserves the audited construction; excluding them from the
design matrix prevents zero-variance scaling and duplicate interpretation.

Three recommended columns remain nullable:

| Column | Missing rows | Missing rate |
|---|---:|---:|
| `hours_since_latest_precise_target_article` | 9,057 | 32.92% |
| `hours_since_latest_precise_common_article` | 324 | 1.18% |
| `observed_target_news_burst_60_session` | 6,441 | 23.41% |

Fit imputation values on each training fold only and retain missingness
indicators. A missing recency value means no eligible article of that type was
observed in the window; a missing burst value means the trailing-history/MAD
condition was not available.

## Claim boundary

This artifact is **exploratory only**. Massive ordinary news does not provide
first-seen timestamps, update timestamps, historical article versions, or full
article bodies. The deterministic features use publication time and the
currently returned description as proxies. Macro coverage is also incomplete
because the archive was collected using stock and benchmark queries rather
than an exhaustive untickered macro feed. The collection-completeness audit
means every cursor exposed for the configured 30 stock, five benchmark, and SPY
queries was exhausted. It does not mean the provider tagged every relevant
article or that the archive is an exhaustive sector-wide information set.
Accordingly, the manifest marks this output:

```text
status = complete_exploratory_non_version_safe
point_in_time_version_safe = false
primary_training_eligible = false
```

Appropriate uses are pipeline integration, weak-data ablations, sensitivity
analysis, and provider feasibility work. It must not be presented as a strict
point-in-time backtest or as evidence of causal news effects.

## Completed exploratory training experiment

The separate
[quant plus deterministic-news v1 experiment](../experiments/quant_deterministic_news/v1/README.md)
completed two exploratory tracks:

- a joint estimator trained on Q56 plus the deterministic-news blocks; and
- a residual corrector trained only from earlier out-of-sample quant errors.

The completed quant-v1 ladder uses the same 917-date,
2022-11-01-through-2026-06-30 calendar, so the joint track does not require a
shorter sample than those reported models. Both matched Q-only parity controls
reproduced the frozen forecasts exactly. Full Q56+D43 Elastic Net improved T2
ETF MSE by 3.65% versus matched Elastic Net, but the 20-session stale-news
placebo did better. Residual news models did not robustly improve the primary
XGBoost base. See the
[final comparison report](../experiments/quant_deterministic_news/v1/comparisons/final/RESULTS.md)
for all rungs, placebo tests, and date-block intervals.

## Exact deterministic feature contract

Every feature is calculated over the news window:

```text
(previous trading-session 09:00 ET, current trading-session 09:00 ET]
```

Articles are deduplicated by Massive provider ID before stock-day
aggregation. “Sector” always means the target stock's fixed six-stock sector,
not the full 30-stock universe.

### Activity, breadth, and commonality (18)

| Feature | Construction |
|---|---|
| `observed_relevant_article_count` | Direct-target, same-sector-peer, or common-proxy articles |
| `observed_direct_target_article_count` | Articles with target ticker/company evidence |
| `observed_target_only_article_count` | Direct-target articles without a common proxy or same-sector peer |
| `observed_peer_article_count` | Articles with at least one known same-sector peer |
| `observed_peer_specific_article_count` | Peer articles without a common proxy |
| `observed_common_proxy_article_count` | Macro, explicit target-sector, or at least two target-sector entities |
| `observed_macro_article_count` | Ticker-tagged articles with deterministic macro evidence |
| `observed_mixed_target_common_article_count` | Articles that are both direct-target and common-proxy |
| `observed_no_relevant_news` | One when the retrospective archive has no relevant article |
| `observed_multi_ticker_article_share` | Relevant articles with at least two provider ticker tags / relevant articles |
| `observed_target_peer_co_mention_share` | Relevant articles mentioning/tagging target and peer / relevant articles |
| `observed_unique_peer_count` | Distinct known peers appearing in relevant articles |
| `observed_peer_coverage_ratio` | Distinct peers / five known peers |
| `observed_sector_firms_with_news_share` | Distinct mentioned firms in the target's sector / six |
| `observed_sector_entity_hhi` | HHI of target-sector entity mentions for the stock-day |
| `observed_target_share_of_sector_entity_mentions` | Target mentions / all target-sector entity mentions |
| `observed_common_shock_balance` | `(common_proxy_count - target_only_count) / (1 + relevant_count)` |
| `observed_firm_common_imbalance` | `log1p(target_only_count) - log1p(common_proxy_count)` |

The `observed_` prefix is intentional: these are measurements of the current
retrospective provider response, not claims that the identical article
versions were visible historically.

### High-precision lexical cues (8)

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

These are overlapping headline-plus-description regex/dictionary cues. They
are sparse high-precision indicators, not exhaustive semantic labels; absence
of a cue does not imply absence of the event.

### Publication-time timing features (9)

| Feature | Construction |
|---|---|
| `timing_eligible_article_count` | Relevant articles with second-precision publication time |
| `timing_imprecise_article_count` | Relevant articles lacking that precision |
| `timing_eligible_share` | Timing-eligible / relevant |
| `hours_since_latest_precise_target_article` | Age of newest precise direct-target article at cutoff |
| `hours_since_latest_precise_common_article` | Age of newest precise common-proxy article at cutoff |
| `recency_weighted_precise_target_count_12h` | Direct-target count with a 12-hour exponential half-life |
| `recency_weighted_precise_common_count_12h` | Common-proxy count with a 12-hour exponential half-life |
| `same_day_premarket_precise_article_share` | Precise articles from 04:00–09:00 ET / precise articles |
| `prior_session_afterhours_precise_article_share` | Precise articles from prior close–20:00 ET / precise articles |

All timing is publication-time based because ordinary Massive News does not
expose first-seen or last-updated timestamps.

### Source and title-propagation proxies (8)

| Feature | Construction |
|---|---|
| `unique_source_count` | Distinct publisher names |
| `source_entropy` | Natural-log Shannon entropy of publisher counts |
| `normalized_title_cluster_count` | Distinct exact normalized-title groups |
| `normalized_title_duplicate_ratio` | `(relevant_article_count - title_cluster_count) / relevant_article_count` |
| `max_normalized_title_cluster_size` | Largest normalized-title group |
| `max_normalized_title_cluster_source_count` | Distinct sources in the largest cross-source title group |
| `government_primary_source_share` | Government/primary-domain articles / relevant articles |
| `press_release_wire_source_share` | Recognized press-release-wire articles / relevant articles |

These are duplication/propagation proxies, not causal event clusters.

### Trailing abnormal activity (1)

`observed_target_news_burst_60_session` is a robust z-score of
`log1p(observed_direct_target_article_count)` against the prior 60 sessions:

```text
(current - trailing median) / (1.4826 * trailing MAD)
```

It is null until 60 prior sessions exist or when the trailing MAD is zero.

## Data this provider cannot supply for the requested features

The frozen manifest rejects these nine fields rather than silently
approximating them:

```text
version_safe_no_news
first_seen_based_recency
last_updated_based_recency
historical_article_version_features
full_body_deterministic_cues
causal_event_count
causal_event_cluster_size
causal_cross_source_propagation
complete_untagged_macro_news_count
```

The paid Benzinga schema could add optional bodies and `last_updated`, but its
documentation does not establish complete body coverage, first-seen time, or
retrieval of every historical version.
