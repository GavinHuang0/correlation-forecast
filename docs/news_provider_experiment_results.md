# News-provider and full-text experiment results

Collection date: 2026-07-26. Local inference and locked evaluation continued
on 2026-07-27.

This report separates documented entitlements from access observed with the
configured credentials. Provider behavior observed today is not a contractual
guarantee. Generated payloads and licensed text remain under Git-ignored
`data/` and `outputs/` paths.

## Decision summary

- The free Massive ordinary Stock News endpoint supplied enough historical
  volume to build the exploratory deterministic-news block. A cursor-exhausted
  30-stock collection returned 82,558 unique documents. The separate
  benchmark/SPY control collection returned 11,257 unique documents, including
  7,732 not already present in the stock collection. After 3,525 cross-root
  overlaps were removed, the combined archive contained 90,290 unique provider
  documents.
- The existing adjusted Alpaca SIP 15-minute archive remains the quant source.
  Massive Basic REST returned 64 adjusted AMD 15-minute bars for 2026-07-20,
  but returned HTTP 403 at the panel's 2022-11-01 start and in 2016. Flat-file
  credentials could list keys, but both tested objects returned HTTP 403 to
  bounded `HEAD` and Range `GET` validation.
- The free ordinary-news archive was converted into 44 deterministic columns
  and joined exactly one-to-one with every row of the 27,510-row quant panel.
  Two columns are constant and two are exact timing redundancies, so the
  manifest recommends 40 columns for fitting. The joined panel is complete for
  exploratory model integration.
- The panel is **not strict point-in-time training data**. Ordinary Massive
  News has publication timestamps but no first-seen time, update timestamp,
  article-version history, or body. The returned current description is
  therefore only a retrospective proxy for what was visible at the forecast
  cutoff.
- Alpha Vantage was not used as a full fallback because Massive ordinary news
  supplied the usable description archive. The collector nevertheless used
  every free request available that day: a frozen 24-ticker overlap diagnostic
  reached the documented 25-request quota and produced 4,120 unique summaries
  from 23 complete slices covering 22 unique tickers, with two ticker slices
  still pending. The collection consumed the configured free daily budget; a
  resumable retry after the local calendar date changed was also quota-limited
  before any new record was cached. Its redacted audit record is
  `experiments/news_provider_fulltext/v1_0/alpha_free_quota_retry_2026-07-27.json`.
  The collection yielded only three exact cross-provider
  matches across the provider corpora and only one match in the selected
  300-document benchmark.
- Public-page retrieval produced a balanced 300-document full-text benchmark,
  but that sample is selection-biased and retrieval-date text. It is suitable
  for the post-cutoff extractor ablation, not historical model training.
- On the locked 228-document evaluation, full text reduced mean macro-F1
  versus Massive descriptions by 0.0908 for FLAN-T5-XL and 0.0593 for Llama
  3.1. Both paired 95% bootstrap intervals were below zero. This does not
  support buying the `$99` Benzinga feed for the current extraction pipeline.

## Provider access

Provider capabilities and official links are maintained in
[news_provider_source_matrix.md](news_provider_source_matrix.md).

### Massive ordinary Stock News

The 30 modeled stocks were collected from the ordinary
[`/v2/reference/news`](https://massive.com/docs/rest/stocks/news) endpoint with
ascending cursor pagination until every cursor was exhausted.

| Measure | Result |
|---|---:|
| Collection status | cursor-complete snapshot at collection time |
| Requested range | 2016-06-22 through 2026-07-26 |
| Completed target tickers | 30 / 30 |
| Response pages | 131 |
| Query rows | 116,983 |
| Unique provider documents | 82,558 |
| Duplicate query rows | 34,425 |
| Duplicate-query-row rate | 29.43% |
| Earliest observed publication | 2016-06-24 15:06 UTC |
| Latest observed publication | 2026-07-26 12:30 UTC |
| Nonempty descriptions | 80,674 / 82,558 |
| Description coverage | 97.72% |
| Public article URL coverage | 100% |
| Unique documents after 2026-03-01 | 4,983 |

“Cursor-complete” means every `next_url` exposed by the API was exhausted at
retrieval time. It does not claim that the still-open July 26 UTC calendar day
was complete; the latest observed article was published before the collection
finished. This qualification does not affect the joined Q+D panel, whose
forecast dates end on 2026-06-30.

The separate SOXX/XLF/XLE/XLV/XLI/SPY control collection also completed:

| Measure | Result |
|---|---:|
| Completed query tickers | 6 / 6 |
| Response pages | 15 |
| Query rows | 12,173 |
| Unique provider documents | 11,257 |
| Unique documents not already in the stock corpus | 7,732 |
| Earliest observed publication | 2020-04-01 13:15 UTC |
| Latest observed publication | 2026-07-25 17:11 UTC |
| Description coverage | 96.41% |
| Unique documents after 2026-03-01 | 366 |

The free key exposed records much older than the two years advertised for
Stocks Basic on Massive's
[pricing page](https://massive.com/pricing?product=stocks). That older access
is preserved locally but must be described as observed, undocumented access.

The paid Benzinga partner endpoint returned HTTP 403 under the free
credentials. Its official
[schema](https://massive.com/docs/rest/partners/benzinga/news) is materially
richer—optional body/teaser plus `published` and `last_updated`—but body
coverage is explicitly not guaranteed and the endpoint does not document
retrieval of every historical article revision.

### Massive flat files

The credentials successfully listed documented minute-aggregate keys. Bounded
object validation then attempted only:

- `us_stocks_sip/minute_aggs_v1/2026/07/2026-07-21.csv.gz`;
- `us_stocks_sip/minute_aggs_v1/2016/01/2016-01-04.csv.gz`.

For both objects:

```text
HEAD:              HTTP 403
Range GET 0-65535: HTTP 403
```

No object body was downloaded. The audit is in
`reports/generated/massive_flat_minute_object_validation_2026-07-26.json`.
The repository's complete adjusted Alpaca archive therefore remains the
source for intraday returns, realized-correlation targets, and quant features.

### Massive REST aggregate bars

The free Stocks key was also tested against the documented adjusted custom-bars
endpoint for AMD 15-minute bars:

| Date | HTTP | Result |
|---|---:|---|
| 2026-07-20 | 200 | 64 adjusted bars |
| 2022-11-01 | 403 | panel start inaccessible |
| 2016-07-20 | 403 | older history inaccessible |

This is consistent with the
[custom-bars documentation](https://massive.com/docs/rest/stocks/aggregates/custom-bars),
which gives Stocks Basic two years of history. Basic can support a recent
sample, but it cannot reproduce the full 2022-11-01 through 2026-06-30 quant
panel. The exact non-secret probe record is
`experiments/news_provider_fulltext/v1_0/massive_rest_aggregate_probe.json`.

### Alpha Vantage overlap diagnostic

Alpha's
[`NEWS_SENTIMENT`](https://www.alphavantage.co/documentation/#news-sentiment)
endpoint documents a maximum of 1,000 rows per call, while the
[free service](https://www.alphavantage.co/support/) documents 25 calls/day.
The bounded March 1-7 diagnostic produced:

| Measure | Result |
|---|---:|
| Frozen ticker scope | 24 / 30 modeled stocks |
| Cached responses | 24 |
| Complete slices / unique completed tickers | 23 / 22 |
| Pending slices | 2 (`UNP`, `BA`) |
| Unique summaries from complete slices | 4,120 |
| Summary/URL/source coverage | 100% |
| Broad windows requiring a split | 1 (`JPM`) |

Only three documents matched the Massive stock corpus by exact normalized URL;
none matched by normalized title plus exact publication second. The selected
300-document benchmark contains one exact Alpha-summary match. A
300-document Alpha-summary versus Massive-description comparison therefore
fails closed rather than using fuzzy or unpaired articles.

Alpha premium would buy backfill throughput. Its official documentation does
not say that premium adds article bodies, revision history, or a higher
per-request news-result ceiling.

## Deterministic-news features

The materialized deterministic archive contains:

| Measure | Result |
|---|---:|
| Raw stock/benchmark query rows | 129,156 |
| Unique provider articles | 90,290 |
| Provider-ID duplicates removed | 38,866 |
| Cross-query articles | 24,106 |
| Article-target rows | 880,466 |
| Stock-day rows | 75,570 |
| Targets / sectors | 30 / 5 |

The following 44 deterministic columns are constructable now:

- 18 activity, ticker-breadth, and commonality proxies;
- 8 high-precision lexical event/surprise cue counts;
- 9 publication-time recency and session-placement features;
- 8 source/title-cluster propagation proxies; and
- 1 trailing 60-session target-news-burst feature.

Only 42 vary in the materialized archive. In addition,
`timing_eligible_article_count` exactly duplicates
`observed_relevant_article_count`, and `timing_eligible_share` is exactly
`1 - observed_no_relevant_news`. The joined manifest retains all 44 for
auditing and provides a 40-column recommended fitting list.

Three recommended columns are nullable: target-article recency is missing on
9,057 rows (32.92%), common-article recency on 324 rows (1.18%), and the
60-session target-news burst on 6,441 rows (23.41%). Any downstream model must
fit imputation values on the training fold only and preserve missingness
indicators.

Important examples include target-only article share, peer coverage, sector
entity HHI, target share of sector mentions, a transparent common-shock
balance, firm/common imbalance, precise publication-time recency, source
entropy, and normalized-title duplication.

The following are not honestly constructable from ordinary Massive News:

- first-seen- or update-aware recency;
- historical article-version features;
- version-safe no-news indicators;
- full-body cue features;
- causal event clusters and causal cross-source propagation; and
- complete untickered macro-news coverage.

See [q_plus_d_massive.md](q_plus_d_massive.md) for the joined artifact and
[news_provider_source_matrix.md](news_provider_source_matrix.md) for the
provider-specific reason each missing field remains unavailable.

## Quant + deterministic-news panel

The frozen quant panel and deterministic stock-day panel were joined on:

```text
forecast_date, sector, stock, benchmark
```

| Measure | Result |
|---|---:|
| Rows / columns | 27,510 / 170 |
| Dates | 917 |
| Date range | 2022-11-01 through 2026-06-30 |
| Stocks / sectors | 30 / 5 |
| Exact news-key matches | 27,510 / 27,510 |
| Preserved quant columns | 112 / 112 |
| Deterministic columns materialized / recommended for fitting | 44 / 40 |
| News audit fields | 13 |
| Output SHA-256 | `352ba2635febfa0bc36bc6f797c00036a8aee9879327a8176e9accf180e50150` |

The manifest status is:

```text
complete_exploratory_non_version_safe
point_in_time_version_safe = false
primary_training_eligible = false
```

This distinction is deliberate: data volume and join completeness are
sufficient, while historical text-version provenance is not.

The isolated
[quant plus deterministic-news v1 experiment](../experiments/quant_deterministic_news/v1/README.md)
is complete. Exact matched Q-only replays passed. The principal A5 linear
model improved T2 ETF MSE versus matched Elastic Net, but failed the
stale-news placebo; the principal residual models did not robustly improve
their frozen XGBoost base. The result supports dataset/pipeline feasibility,
not a reliable historical news-alpha claim.

## Public full-text side experiment

The retriever respected `robots.txt`, did not bypass paywalls or login
controls, capped each response at 2 MB, and retained provenance and hashes.

| Measure | Result |
|---|---:|
| Candidate queue | 5,180 unique post-cutoff documents |
| Retrieval attempts | 554 |
| Unique URLs attempted | 553 |
| Retrieved text | 399 |
| Success per attempt | 72.02% |
| Success per unique URL | 72.15% |
| Attempted / successful domains | 4 / 2 |
| `main` paragraph extraction | 271 |
| JSON-LD `articleBody` extraction | 128 |
| No article text | 141 |
| Robots denied | 7 |
| HTTP 404 | 2 |
| Below minimum length | 2 |
| Timeout | 3 |

The frozen benchmark has:

| Property | Result |
|---|---:|
| Documents | 300 |
| Per stock / stocks / sectors | 10 / 30 / 5 |
| Development / evaluation bookkeeping split | 72 / 228 |
| Publication rule | strictly after 2026-03-01 00:00 UTC |
| Publication range | 2026-03-01 06:30 through 2026-07-25 16:03 UTC |
| Massive descriptions | 300 |
| Retrieved full texts | 300 |
| Alpha summaries | 1 |
| Full-text chunks | 1,675 |
| Mean / median chunks per document | 5.58 / 5 |
| Mean description length | 453 characters |
| Mean reconstructed full-text input length | 3,695 characters |
| Publishers represented | 2 (204 Motley Fool, 96 Benzinga) |
| Assigned target provider-tagged | 73 / 300 |
| Assigned target or known peer provider-tagged | 266 / 300 |
| Sector/market-eligibility-only assignments | 34 / 300 |

Assignment evidence is also uneven across the bookkeeping split:

| Assignment basis | All 300 | Development 72 | Evaluation 228 |
|---|---:|---:|---:|
| Assigned target provider-tagged | 73 | 15 | 58 |
| Target absent, known peer provider-tagged | 193 | 52 | 141 |
| Expanded sector/market eligibility only | 34 | 5 | 29 |

The source and assignment mixes are confounded. Benzinga contributes 27 of
the 34 expanded-only assignments but only 15 of the 73 direct-target
assignments. A Benzinga-only sensitivity result is therefore not a clean
publisher effect.

The date boundary is after GPT-5.6 Sol's documented 2026-02-16
[knowledge cutoff](https://developers.openai.com/api/docs/models/gpt-5.6-sol),
after Llama 3.1's documented December 2023 cutoff and 2024 release, and after
FLAN-T5-XL's 2022 release proxy. FLAN's
[model card](https://huggingface.co/google/flan-t5-xl) does not publish a true
training-data cutoff, so “post-release” is the only defensible claim.

Both local tokenizer preflights passed with zero violations:

| Runner | Input records checked | Maximum operational input | Violations |
|---|---:|---:|---:|
| FLAN-T5-XL | 1,976 | 512 tokens | 0 |
| Llama 3.1 8B Instruct | 1,976 | 1,024 tokens | 0 |

The full-text variant is chunked. Each chunk is inferred independently and
repeats the headline; the local model does not see the reconstructed parent
article in one context. One global chunk reducer per local model is selected
from four candidates on the 72-document development split, locked with input
and output hashes, and then applied unchanged to the 228-document evaluation
split. Mean aggregation weights chunks equally, while max-score and
best-margin reducers have more opportunities to find extreme scores in longer
articles. Score-map coverage is therefore reported explicitly. GPT-5.6 Sol
receives the reconstructed retrieved parent text. Thus this measures chunked
retrieved text versus short provider descriptions—not verified original
historical bodies.

## Model agreement with GPT-5.6 Sol

GPT labels are a silver reference, not human truth. No human audit was
performed in this run, so the tables report agreement only. The 300 labels
were generated in 15 Codex collaboration batches with the requested
`gpt-5.6-sol` model, no external tools, and the frozen annotation protocol.
This was not a dated API snapshot, and platform system instructions were not
replaced. Evidence validation proves only that each nonempty evidence string
was an exact substring of the supplied headline or parent text; it does not
prove that the excerpt entails the label.

The comparison is intentionally an information-retention test, not a symmetric
model contest. GPT saw headline, reconstructed full parent text, and target
metadata, but no provider ticker tags. Each local model saw the applicable text
variant, target metadata, and the provider's actual ticker tags. The same
full-text-derived GPT labels score both the short-description and full-text
arms, which structurally favors an arm containing the information used by the
reference. No expanded sector/market assignment ticker was exposed as if it
were a provider tag.

Paired bootstrap intervals and McNemar tests resample or compare articles as
if they were independent. Shared publishers, dates, and potentially repeated
underlying events make those uncertainty estimates optimistic. Rare classes
can also disappear from a bootstrap resample, changing the supported-class
macro-F1 estimand. The intervals and p-values are descriptive and unadjusted
for the number of fields, models, and sensitivity slices.

## Evidence artifacts and checksums

The generated provider payloads and licensed text are intentionally
Git-ignored. The following SHA-256 values bind the local evidence used in this
report as it existed after benchmark rebuild on 2026-07-27. They are
reproducibility identifiers, not provider guarantees.

| Evidence | Local artifact | SHA-256 |
|---|---|---|
| Massive free ordinary-news stock collection | `data/raw/massive/ordinary_news/free_v1/manifest.json` | `8419ddc03ec68861d818745393dcaf102f919f5086aa1d0f7f9291c253a2db63` |
| Massive benchmark/SPY control collection | `data/raw/massive/ordinary_news/free_v1_benchmarks/manifest.json` | `99538f46727661c6181e98615979ee151768da14978d10d6ca34dfba3b20c652` |
| Massive access and Benzinga-403 probes | `experiments/news_provider_fulltext/v1_0/access_probe_summary.json` | `6516f00a63469b175a730db6069a08c3056ca85370d4006aae579498c58c11a2` |
| Massive flat-file object validation | `reports/generated/massive_flat_minute_object_validation_2026-07-26.json` | `50b8df3a1387d52c2750a3f157d87fb5d34b2fef7816639620add7510753b020` |
| Massive REST aggregate probe | `experiments/news_provider_fulltext/v1_0/massive_rest_aggregate_probe.json` | `83e32204ab39e0a343f71f881c739772481478b7740f798f7d95b92508fac020` |
| Alpha bounded diagnostic manifest | `data/raw/alpha_vantage_news/postcutoff_overlap_v1/manifest.json` | `838c0e92a7f727295e9c704b264bd82b6e34ed0ade374304df840d1275c20a6e` |
| Alpha quota-retry audit | `experiments/news_provider_fulltext/v1_0/alpha_free_quota_retry_2026-07-27.json` | `4116fa60fd6b1de61b5f9511e15dd48506b3c0ee24bd3190caeac1d4589ef937` |
| Alpha/Massive exact-overlap report | `data/external/news_provider_comparison/v1_0/provider_overlap_final.json` | `ea3fd8cef8e8d6d40eb709e56079b79f13a829a44779b27ad3f69c1082c2f9ba` |
| Deterministic-news feature manifest | `data/features/news_deterministic/massive_v1/manifest.json` | `01f3d16df1ed7caf9ba582ad489b366e17aed89df1689e626ae22a0e32fa1a52` |
| Q+D modeling panel | `data/features/q_plus_d/massive_v1/modeling_panel_q_plus_d.parquet` | `352ba2635febfa0bc36bc6f797c00036a8aee9879327a8176e9accf180e50150` |
| Q+D manifest | `data/features/q_plus_d/massive_v1/manifest.json` | `ac161d48901fae669fdc500099df92d218513d2e7d8d383539b8ed326445d86f` |
| Public-page retrieval ledger | `data/external/news_provider_comparison/v1_0/fulltext/manifest.jsonl` | `62a30bf70767d713616a1111f67d279b692d64ceb70ec358c3e8fd297f3261fe` |
| Frozen 300-document benchmark manifest | `data/benchmarks/news_text_ablation_300/v1/manifest.json` | `dc3b8fcefe3c48effff901d3b1cfd2b461f1bd4db0b44f1bcc9ea2e367145857` |
| Massive-description benchmark input | `data/benchmarks/news_text_ablation_300/v1/massive_description.jsonl` | `24dc9de18dc6123ad26ad5fd21b4e46175b695a220fbd85a089d9394c74ce2a4` |
| Full-text chunk benchmark input | `data/benchmarks/news_text_ablation_300/v1/fulltext_evidence_chunks.jsonl` | `d8ca8d83e78686383d01246dd341128df9bdc978abf948559e214b1a13e940b9` |
| GPT-5.6 Sol silver annotations | `annotations/news_provider_fulltext/v1_0/gpt_5_6_sol_silver.jsonl` | `6536f25228706ce5d70c9556a266bdae74f21be613573c3fc2a2ba5a8bd0fb70` |

<!-- MODEL_RESULTS_START -->

Both local models completed on the frozen inputs. Development-only selection
locked `max_score` as the article-level chunk reducer for both models. That
choice followed the preregistered priority on shock-scope and directional-
alignment macro-F1; directional alignment is hierarchy-derived and invariant
to the reducer, so the varying part of the first criterion is effectively
shock scope. `max_score` must not be described as the best overall reducer:
FLAN plurality and Llama mean-score aggregation had higher all-field
development macro-F1 and accuracy. For Llama, the development shock-scope
macro-F1 difference between `max_score` and `best_margin` was only 0.000032.

### Locked 228-document evaluation

| Model | Description mean accuracy | Full-text mean accuracy | Full-text delta (95% paired bootstrap CI) | Description mean macro-F1 | Full-text mean macro-F1 | Full-text delta (95% paired bootstrap CI) |
|---|---:|---:|---:|---:|---:|---:|
| FLAN-T5-XL | 0.4397 | 0.3772 | **-0.0625** [-0.0910, -0.0329] | 0.3562 | 0.2653 | **-0.0908** [-0.1283, -0.0561] |
| Llama 3.1 8B Instruct | 0.4638 | 0.4265 | **-0.0373** [-0.0658, -0.0088] | 0.3661 | 0.3069 | **-0.0593** [-0.0927, -0.0285] |

The paired difference is always:

```text
chunked retrieved full text minus Massive description
```

Thus, negative values favor the Massive description. On this benchmark and
with the locked aggregation protocol, full text reduced agreement for both
models.

Per-field macro-F1 results show that the aggregate finding was not driven by
one field:

| Model / field | Description | Full text | Delta | 95% paired bootstrap CI |
|---|---:|---:|---:|---:|
| FLAN / shock scope | 0.3404 | 0.2562 | -0.0842 | [-0.1409, -0.0310] |
| FLAN / event family | 0.3604 | 0.2804 | -0.0800 | [-0.1368, -0.0281] |
| FLAN / information status | 0.4044 | 0.3057 | -0.0987 | [-0.1777, -0.0539] |
| FLAN / directional alignment | 0.3194 | 0.2191 | -0.1004 | [-0.2061, -0.0069] |
| Llama / shock scope | 0.2516 | 0.1956 | -0.0560 | [-0.0958, -0.0164] |
| Llama / event family | 0.5915 | 0.5061 | -0.0854 | [-0.1599, -0.0082] |
| Llama / information status | 0.4052 | 0.3752 | -0.0299 | [-0.0885, 0.0140] |
| Llama / directional alignment | 0.2163 | 0.1505 | -0.0658 | [-0.1094, -0.0223] |

The result also remained negative in the main sensitivity slices:

| Slice | Documents | FLAN macro-F1 delta | Llama macro-F1 delta |
|---|---:|---:|---:|
| GPT non-abstained | 154 | -0.1037 | -0.0502 |
| Benzinga only | 77 | -0.0497 | -0.0586 |
| Benzinga and GPT non-abstained | 54 | -0.0419 | -0.0775 |
| Assigned target directly provider-tagged | 58 | -0.0987 | -0.0656 |
| Peer-only provider tag | 141 | -0.0871 | -0.0723 |
| Expanded sector/market assignment | 29 | -0.0150 | -0.0184 |

The FLAN Benzinga/non-abstained and both expanded-assignment confidence
intervals included zero. The Llama Benzinga/non-abstained interval remained
below zero. These are small, overlapping, post-selection slices and are not
independent confirmations.

The full-text result was not caused by more missing predictions. For each
model, descriptions had complete score maps for 216/228 evaluation articles
and no score maps for 12. Full text had all chunks scored for 203 articles,
partial chunk scores for 22, and no chunk scores for only 3; 1,187/1,288
evaluation chunks had complete score maps. Directional alignment has no score
map by design because it is hierarchy-derived.

The most plausible interpretation is specific to this implementation:
provider descriptions concentrate the primary event, while retrieved bodies
add boilerplate, secondary companies, and multiple candidate passages.
`max_score` also gives longer documents more opportunities to produce an
extreme score. The experiment therefore rejects purchasing full text for the
**current frozen extraction pipeline**; it does not establish that full text
would fail with a long-context article model, learned attention, a
target-conditioned body extractor, or human reference labels.

One stale FLAN description-only report,
`evaluation_description_only_all.json`, exposed description-arm holdout
metrics before the full-text lock was created. It was not used for reducer
selection and no Llama equivalent was generated, but the broader protocol is
therefore not fully blinded. Combined with the silver reference, two-publisher
selection, current-page bodies, assignment imbalance, and optimistic
article-level uncertainty, all inferential claims remain exploratory.

### Model-output evidence

| Evidence | Local artifact | SHA-256 |
|---|---|---|
| FLAN descriptions | `outputs/news_provider_fulltext/v1_0/flan_t5_xl/massive_description.jsonl` | `89ade123043109858f5e24e818470c2ff2c8dac79b7b798e9cb555b63cc8251c` |
| FLAN full-text chunks | `outputs/news_provider_fulltext/v1_0/flan_t5_xl/fulltext_evidence_chunks.jsonl` | `04a2df536d8b16d139148e74afebb150d9ae77afe46f509fcc7f9e03d9d38bf4` |
| FLAN aggregation lock | `outputs/news_provider_fulltext/v1_0/flan_t5_xl/aggregation_lock.json` | `fa8fc93cd40fb4ed3baa94acf54b725aea5f623f38660b188f14d3745fab261c` |
| FLAN primary evaluation | `outputs/news_provider_fulltext/v1_0/flan_t5_xl/evaluation_primary.json` | `c1c25d892b2263351071f82a3647d9d24e8cfc4a428e897282f852a3fffdc69e` |
| Llama descriptions | `outputs/news_provider_fulltext/v1_0/llama_3_1/massive_description.jsonl` | `66642a78f1694059522485571fa04679ae8d2c620c11899fcfd4e0179df7f342` |
| Llama full-text chunks | `outputs/news_provider_fulltext/v1_0/llama_3_1/fulltext_evidence_chunks.jsonl` | `9a1a32e647381c91412b5da4b203fa009c9a9bfd6986bf42ddf589b677109291` |
| Llama aggregation lock | `outputs/news_provider_fulltext/v1_0/llama_3_1/aggregation_lock.json` | `d4a80348e1265eecb1c71d173c390ba6c07c395778202ba9a6a0420fbc0ac0b3` |
| Llama primary evaluation | `outputs/news_provider_fulltext/v1_0/llama_3_1/evaluation_primary.json` | `080b3f30545642f1767b50f9062c56b5e04e7c70f7dcb37592c9d3b3965505cf` |

<!-- MODEL_RESULTS_END -->

## Provider recommendation

The access and data-readiness evidence already supports three procurement
decisions:

1. **Do not buy a Massive stock plan for quant data now.** Basic REST bars cover
   only the recent entitlement window and free flat-file objects were not
   retrievable, but the adjusted Alpaca quant archive and quant panel are
   already complete.
2. **Do not buy Massive Stocks Starter solely for ordinary-news schema.** The
   free ordinary endpoint already supplied and preserved enough data for the
   exploratory deterministic block. Starter may still be worthwhile for a
   contractual history/throughput guarantee if Massive confirms the
   endpoint-specific entitlement in writing.
3. **Do not buy Alpha premium for presumed text richness.** It addresses the
   25-call bottleneck, but official documentation does not establish fuller
   text or revision-aware data.

4. **Do not buy the `$99` Benzinga feed for the current extraction
   pipeline.** Both local models performed better from ordinary Massive
   descriptions than from retrieved bodies, and the result stayed negative in
   the Benzinga-only slice. The paid feed may have cleaner bodies and metadata
   than public pages, but this experiment supplies no evidence that those
   fields would improve the frozen extractor enough to justify the cost.

The public-body result is not a direct validation of the paid Benzinga feed:
the benchmark contains 204 Motley Fool pages and 96 Benzinga pages, and it
measures current public retrieval rather than paid-feed body coverage. Revisit
the purchase only if a different long-context or target-conditioned pipeline
first shows a need for body text, then require a matched paid-feed sample and
written answers from Massive about:

- historical body coverage by year and ticker;
- the fraction of headline-only records;
- whether the returned body is the final version only;
- retention rights for private research; and
- whether earlier article versions are retrievable.

Even if a later extractor benefits from full text, a purchase is not justified
for strict historical training unless the version/availability problem is
solved. The ordinary feed is sufficient for exploratory deterministic
features today; the paid Benzinga feed would be a text-quality/provenance
purchase, not a volume purchase.

For a no-purchase path, begin an hourly forward collector now. Store every raw
record version immutably under provider ID plus text hash, record local
`first_seen_at`, `last_seen_at`, `retrieved_at`, and provider timestamps, and
never overwrite an earlier body or description. That can create a genuinely
point-in-time archive for future tests, but it cannot repair the historical
version uncertainty in the current retrospective backfill.
