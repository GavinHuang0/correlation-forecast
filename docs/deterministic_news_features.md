# Deterministic news feature readiness

> **Legacy pilot (superseded).** This document records the original bounded
> 2024 Alpha Vantage semiconductor pilot. It is not the current provider or
> training-readiness assessment. The current 30-stock Massive ordinary-news
> archive, 75,570-row stock-day feature panel, 44 materialized / 40
> recommended deterministic columns, and 27,510-row Q+D join are documented
> in [news_provider_experiment_results.md](news_provider_experiment_results.md)
> and [q_plus_d_massive.md](q_plus_d_massive.md). Those current artifacts are
> still marked exploratory and non-version-safe. The v2 deterministic contract
> has since been built and trained, while the semantic contracts remain
> construction-blocked, in
> [`experiments/quant_deterministic_news/v2/README.md`](../experiments/quant_deterministic_news/v2/README.md).

## Decision

The repository does **not** yet contain a complete, point-in-time news archive
for production correlation-model training. It does contain enough cached data
for a bounded 2024 semiconductor pipeline pilot.

The pilot is built by:

```powershell
C:\Users\gavin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  scripts\build_deterministic_news_features.py
```

It is CPU-only and isolated from every FLAN-T5 and FLAN-T5-XL path.

## Current data inventory

### News

The local Alpha Vantage archive has:

- 11 query payloads;
- 8,084 query-result rows;
- 6,839 unique URLs after removing 1,245 cross-query duplicates;
- publication time, headline, summary, URL, publisher, topics, and ticker tags;
- no stable provider article ID, first-seen time, update/version time, archived
  article version, per-record retrieval time, or full article body.

Coverage is incomplete:

- AMD, AVGO, INTC, and MU have uncapped 2024 ticker responses;
- NVDA hits the 1,000-result limit and ends on December 3;
- QCOM has no dedicated ticker response;
- most macro, monetary, and technology topic responses hit 1,000 and cover
  only fragments of 2024;
- no dedicated queries exist for the other four sectors in the 30-stock price
  universe.

Of the 6,839 unique articles, 1,715 have an exact `T000000` timestamp. The
pilot treats these as date-only records, delays availability to the next UTC
midnight, and excludes them from intraday timing calculations.

The historical responses contain only the currently returned summary and
metadata. They cannot prove which text version was available at the forecast
cutoff. One article stamped `20240102T085135` even contains a banner-image path
under `/2025/10/`, concrete evidence that at least some historical metadata was
mutated later. Consequently, all pilot rows have:

```text
point_in_time_version_safe = 0
primary_training_eligible = 0
```

### Quant and calendar data

The quant block is substantially more complete:

- the Bollerslev-style core feature panel is already built;
- the additional quant panel has 79,110 rows for 2016-01-04 through
  2026-06-30;
- the exchange calendar has 2,637 sessions and official early closes;
- BLS and FOMC schedules cover the research period;
- the macro panel already contains `scheduled_macro_event_count`,
  `scheduled_preopen_macro_count`, `bls_release_day`, `bea_release_day`, and
  `fomc_decision_day`.

Do not duplicate those macro columns in the deterministic-news block. More
specific CPI, PPI, and employment flags can be derived from the existing
official calendar when useful. BEA coverage before 2025 remains incomplete.

The local earnings archive covers 20 of 30 stocks and contains reported dates,
but no actual announcement time or point-in-time schedule. It is unsuitable
for primary pre-open earnings flags.

## Feature feasibility

| Feature family | Current status | Correct interpretation |
|---|---|---|
| Exact-deduplicated article counts | Pilot-ready | Article counts, not event counts |
| Target/peer tags and co-mentions | Pilot-ready | Fixed semiconductor peer universe |
| Multi-ticker share | Pilot-ready | Based on vendor ticker tags |
| Unique peer count and coverage | Pilot-ready | Coverage of configured peers only |
| Source count and entropy | Pilot-ready | Across observed articles |
| Explicit positive/negative surprise cues | Pilot-ready | Unattributed article-level lexical cues |
| Nonexclusive event-family cues | Pilot-ready | Dictionary/vendor-topic cues, not semantic labels |
| Conservative recency and timing | Partial | Only non-midnight publication timestamps |
| Target-news burst | Partial | Computable, but archive completeness is unproven |
| Sector breadth, HHI, common/firm balance | Proxy only | Fixed-six-name, observed-feed proxies |
| `no_news` | Proxy only | Means no article in the incomplete observed archive |
| Cross-source propagation | Proxy only | Normalized-title clusters, not true event clusters |
| Full sector breadth/HHI | Blocked | Needs complete point-in-time constituents and news |
| True event counts/syndication | Blocked | Needs frozen causal event clustering |
| Strict point-in-time timing | Blocked | Needs first-seen/update/version timestamps |
| Earnings-before-cutoff flags | Blocked | Needs scheduled and actual announcement times |
| Historical policy-deadline flags | Blocked | No versioned policy calendar exists locally |

## Generated pilot artifacts

All artifacts live under:

```text
data/features/news_deterministic/v1_0/
```

| File | Rows | Purpose |
|---|---:|---|
| `normalized_articles.csv.gz` | 6,839 | Exact-URL-deduplicated metadata and deterministic cues; raw text is not duplicated |
| `article_target_features.csv.gz` | 26,185 | Relevant article-target proxy features |
| `stock_day_features.csv.gz` | 1,260 | Five targets × 252 trading sessions |
| `coverage_audit.csv` | 11 | Query caps, time spans, missing request manifests, and source hashes |
| `manifest.json` | 1 | Input/output hashes, counts, timing contract, and limitations |

The daily panel includes:

- observed article activity by target, peer, peer-specific, macro, target-only,
  common-proxy, and mixed-proxy route;
- multi-ticker and target-peer co-mention shares;
- configured peer coverage and fixed-six-name breadth/HHI proxies;
- common-shock balance and firm/common imbalance proxies;
- un-attributed article-level explicit surprise cue counts;
- six nonexclusive event-family cue counts;
- conservative recency, premarket, and prior-after-hours features for precise
  timestamps only;
- source count/entropy, normalized-title duplication, and narrow primary-source
  flags;
- archive coverage and training-eligibility flags.

All count-derived commonality columns use an `observed_*` prefix so they cannot
be mistaken for complete information-set measurements.

The pilot does not assign a surprise phrase to a particular company or to a
common sector component. Entity/clause attribution is required before
constructing target-versus-common cue alignment.

## Production retrieval plan

### Option A: continue with Alpha Vantage summaries

Build a resumable downloader that:

1. queries every stock in the selected universe, not only the target stocks;
2. uses bounded `time_from`/`time_to` intervals;
3. bisects any interval that returns 1,000 records until every leaf request is
  below the limit;
4. stores the query parameters, retrieval time, response hash, rate-limit/error
  state, and raw payload;
5. repeats the process for sector and macro topics;
6. exact-deduplicates query overlap using canonical URL and text hashes.

Alpha Vantage documents historical time filters and a maximum of 1,000 results
per request:

[https://www.alphavantage.co/documentation/#news-sentiment](https://www.alphavantage.co/documentation/#news-sentiment)

This can produce a broad summary-based archive, but it will remain a
publication-time proxy unless the provider supplies historical revision or
first-seen metadata. It should not be described as a version-safe news
backtest.

### Option B: use a versioned news source

For the primary point-in-time experiment, prefer records containing:

```text
provider_article_id
published_at
first_seen_at
last_updated_at
retrieved_at
available_at
headline
summary_or_teaser
body, when licensed and available
tickers
topics
publisher
text_hash
raw_payload_hash
```

Define conservatively:

```text
available_at = max(published_at, first_seen_at, last_updated_at)
```

Massive's Benzinga News endpoint currently documents a `benzinga_id`,
`published`, `last_updated`, tickers, tags/channels, teaser, and optional body:

[https://massive.com/docs/rest/partners/benzinga/news](https://massive.com/docs/rest/partners/benzinga/news)

Confirm retention and research-use rights before storing or redistributing
article bodies.

### Event clustering

After a complete archive exists:

1. exact-deduplicate provider IDs/canonical URLs;
2. cluster only records available before the current forecast cutoff;
3. require entity overlap, a bounded time gap, and frozen title/summary
  similarity rules;
4. prevent later articles from bridging two earlier clusters;
5. manually audit a stratified sample of candidate matches/non-matches;
6. freeze thresholds before looking at forecast results.

Only then rename article counts to event counts and construct syndication,
independent-source propagation, and maximum event-cluster-size features.

### Security master and calendars

For a multi-sector production panel, add an effective-dated security master:

```text
permanent_security_id
ticker and company-name aliases with start/end dates
sector/industry with effective dates
peer-universe membership
sector benchmark
ETF/index constituent weight, when used
```

If historical constituent data is unavailable, retain a fixed, preregistered
peer basket and name every feature accordingly; do not call it the complete
sector.

Acquire a separate earnings calendar with scheduled time, actual release time,
status, and revision history before constructing target- or peer-earnings
features. The current reported-date archive may be used only as an
ambiguous-timing robustness check.

## Verification

Run:

```powershell
C:\Users\gavin\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe `
  -m unittest tests.test_build_deterministic_news_features -v
```

The tests cover cutoff alignment, conservative handling of midnight
timestamps, exact cross-query deduplication, peer-specific versus common
routing, the exact 60-session burst requirement, primary-training exclusion,
deterministic rebuilds, and rejection of any FLAN output destination.
