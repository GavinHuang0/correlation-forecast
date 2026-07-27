# News-provider and model source matrix

Verified against official provider documentation and official model cards on
2026-07-26. This file separates documented entitlements from behavior observed
with this repository's credentials. Observed access is not a contractual
entitlement and may change.

## Provider documentation

| Product | Documented access, history, and throughput | Documented content | Relevance to this experiment |
|---|---|---|---|
| Massive ordinary Stock News - Stocks Basic | The [`/v2/reference/news` endpoint](https://massive.com/docs/rest/stocks/news) is included in all Stocks plans, updates hourly, and documents **2 years** of history for Basic. Results date back to 2016-06-22 at the dataset level. A page is limited to 1,000 records and `next_url` provides cursor pagination. The [Stocks pricing page](https://massive.com/pricing?product=stocks) lists Basic at $0, 5 API calls/minute, and 2 years of historical data. | Stable article `id`, exact `published_utc`, title, article URL, publisher, ticker tags, optional `description`, optional keywords, and optional per-ticker sentiment insights. The published response schema does **not** include article body text or an article update/version timestamp. | Sufficient fields for deterministic counts, timing, vendor-tag breadth, source breadth, keywords, and vendor-sentiment controls. It is not a documented full-text or point-in-time article-version archive. |
| Massive ordinary Stock News - Stocks Starter | The [news endpoint page](https://massive.com/docs/rest/stocks/news) documents **all history** for Starter and says records date back to 2016-06-22. The [general Stocks pricing page](https://massive.com/pricing?product=stocks) lists Starter at $29/month, unlimited API calls, and **5 years** of historical data. | Same ordinary-news schema as Basic; paying for Starter does not document a richer article-text field. | The two official pages disagree on the history label. The endpoint-specific page is more directly relevant to News, but confirm historical News entitlement with Massive before purchasing. Starter appears to buy stable throughput and documented historical access, not full text. |
| Massive Benzinga News partner feed | The separate [`/benzinga/v2/news` endpoint](https://massive.com/docs/rest/partners/benzinga/news) is documented at $99/month, real time, with all history back to 2001-12-05. Maximum page size is 50,000 with `next_url` pagination. The [Massive pricing page](https://massive.com/pricing?product=stocks) describes Benzinga partner data as $99/month **per dataset**. | `benzinga_id`, original `published`, `last_updated`, author, title, URL, optional teaser, optional body, optional channels/tags/tickers/images. Massive explicitly says some time-sensitive records are headline-only. | This is the only Massive product here that documents body text and update timestamps. Body coverage is not guaranteed. The docs do not say that every historical version of an edited article can be retrieved, so `last_updated` is useful for conservative availability dating but is not proof of a versioned archive. |
| Massive Stocks REST custom bars | The [custom-bars endpoint](https://massive.com/docs/rest/stocks/aggregates/custom-bars) is included in all Stocks plans. It documents end-of-day recency and two years of history for Basic, five years for Starter, ten years for Developer, and all history for Advanced. Results are adjusted for splits by default and support custom intervals including 15 minutes. | Per-ticker OHLCV aggregates, not news. | Basic can supply a recent quant sample, but its documented two-year history cannot reproduce this repository's 2022-11-01 through 2026-06-30 panel. |
| Massive Stocks minute-aggregate flat files | The [minute-aggregate flat-file page](https://massive.com/docs/flat-files/stocks/minute-aggregates) documents one-minute OHLCV files across U.S. equities. It currently says Basic is **not included**, while Starter has end-of-day access and five years of history. The data is unadjusted, so corporate actions must be applied separately. | Market bars, not news articles. No news flat-file dataset is listed. | Useful in a paid quant-data backfill, but flat-file credentials do not solve news retrieval or article-text coverage. |
| Alpha Vantage `NEWS_SENTIMENT` - free key | The [Market News & Sentiment documentation](https://www.alphavantage.co/documentation/#news-sentiment) describes live and historical multi-outlet news, ticker/topic filters, exact `time_from`/`time_to`, and `LATEST`, `EARLIEST`, or `RELEVANCE` ordering. Multiple tickers/topics are an **AND** filter ("simultaneously mention/cover"), not a union query. Default limit is 50 and maximum limit is 1,000 per request. Alpha's [support page](https://www.alphavantage.co/support/) documents a standard free limit of 25 API requests/day. | The endpoint is described as news and sentiment data. In this experiment's live payloads it returned headline, URL, publication time, source, topics, ticker relevance/sentiment, and summary - not full article body or revision history. The public endpoint page does not publish an article-version guarantee. | Usable for summary-based deterministic and LLM features, but dense per-ticker backfills require narrow time slicing because 1,000 results can truncate a broad query. The free daily quota makes a large multi-year, multi-stock backfill slow. |
| Alpha Vantage `NEWS_SENTIMENT` - premium key | Alpha's [premium page](https://www.alphavantage.co/premium/) says premium has no daily limits and unlocks the other premium features. It does not document a different `NEWS_SENTIMENT` per-request maximum, richer text, or a different news-history start. Current price/request-per-minute choices are rendered in the page's interactive selector and should be checked at purchase time. | No official statement found that premium changes a summary record into full article text or adds historical article versions. | Premium is justified primarily for backfill throughput. Do not buy it on the assumption that it supplies full text or revision-aware records without written confirmation from Alpha Vantage. |

### Documented schema limitations

- Massive ordinary News documents `published_utc` but not `last_updated`,
  `first_seen`, body text, or historical article versions.
- Massive Benzinga documents both `published` and `last_updated`, but its body and
  teaser fields are optional and some records are headline-only.
- Alpha Vantage documents time filters and a 1,000-result ceiling but does not
  document cursor pagination for `NEWS_SENTIMENT`. Time-window slicing is
  therefore required whenever a query reaches 1,000.
- None of the official pages above guarantees that the text returned today is
  exactly the version visible to traders at the original publication timestamp.

## Live access observed in this experiment

These are credential probes performed on 2026-07-26, not advertised plan rights.

| Probe | Observed result | Interpretation |
|---|---|---|
| Massive ordinary News with the Basic key | HTTP 200 for recent 2026 records, January 2024, January 2023, and a query beginning at the documented dataset start on 2016-06-22. The full collection's earliest observed publication was 2016-06-24. Sample records included nonempty descriptions. | The current key exposed substantially more than the documented Basic two-year history. Extract it while available, but do not describe 2016 history as a guaranteed free entitlement or make reproducibility depend on it remaining open. |
| Massive Benzinga with the Basic key | HTTP 403. | The $99 partner feed is not included in the current free entitlement. |
| Massive flat-file credentials | Authenticated prefix listing succeeded, but bounded validation of the documented current minute-aggregate object (`2026-07-21`) and a historical object (`2016-01-04`) returned HTTP 403 for both `HEAD` and a 64 KiB Range `GET`. | The free credentials can enumerate keys but cannot retrieve the tested objects. They are not usable as a quant-data source in this environment, and there is no news flat-file product. |
| Massive REST 15-minute aggregates with the Basic key | An adjusted AMD request for 2026-07-20 returned HTTP 200 and 64 bars. The same request for the quant panel start on 2022-11-01 and for 2016-07-20 returned HTTP 403. | The free key can retrieve recent adjusted bars, consistent with the documented two-year Basic history, but cannot recreate the full existing quant panel. |
| Alpha Vantage AMD query, 2026-03-01 through 2026-07-26 | Returned exactly 1,000 records; with `sort=EARLIEST`, the latest returned item was only around 2026-04-01. | A single broad request was truncated by the documented 1,000 cap. Month-sized or adaptively bisected windows are necessary even for one active ticker. |

The ordinary-news result conflicts with the advertised two-year Basic
entitlement; the REST-bar and flat-file probes do not. Treat the older ordinary
news as undocumented access, preserve raw responses and retrieval timestamps,
and retain an Alpha or paid-plan recovery path.

### Local evidence for the observed-access claims

The non-secret probe summaries are tracked, while raw responses and
provider-licensed text remain Git-ignored. The main evidence bindings are:

| Observation | Artifact | SHA-256 |
|---|---|---|
| Ordinary-news, Benzinga, flat-listing, and Alpha broad-window probes | `experiments/news_provider_fulltext/v1_0/access_probe_summary.json` | `6516f00a63469b175a730db6069a08c3056ca85370d4006aae579498c58c11a2` |
| Free ordinary-news stock collection | `data/raw/massive/ordinary_news/free_v1/manifest.json` | `8419ddc03ec68861d818745393dcaf102f919f5086aa1d0f7f9291c253a2db63` |
| Free ordinary-news benchmark/SPY collection | `data/raw/massive/ordinary_news/free_v1_benchmarks/manifest.json` | `99538f46727661c6181e98615979ee151768da14978d10d6ca34dfba3b20c652` |
| Flat-file bounded object validation | `reports/generated/massive_flat_minute_object_validation_2026-07-26.json` | `50b8df3a1387d52c2750a3f157d87fb5d34b2fef7816639620add7510753b020` |
| REST aggregate entitlement probe | `experiments/news_provider_fulltext/v1_0/massive_rest_aggregate_probe.json` | `83e32204ab39e0a343f71f881c739772481478b7740f798f7d95b92508fac020` |
| Alpha bounded diagnostic | `data/raw/alpha_vantage_news/postcutoff_overlap_v1/manifest.json` | `838c0e92a7f727295e9c704b264bd82b6e34ed0ade374304df840d1275c20a6e` |
| Alpha quota-retry audit | `experiments/news_provider_fulltext/v1_0/alpha_free_quota_retry_2026-07-27.json` | `4116fa60fd6b1de61b5f9511e15dd48506b3c0ee24bd3190caeac1d4589ef937` |

Counts and downstream artifact hashes are reported in
[news_provider_experiment_results.md](news_provider_experiment_results.md).
These hashes identify the local observations; they do not convert observed
free-key behavior into a documented entitlement.

## Model date controls

| Model | Official date evidence | Benchmark rule |
|---|---|---|
| GPT-5.6 Sol | The [OpenAI model page](https://developers.openai.com/api/docs/models/gpt-5.6-sol) gives a **2026-02-16 knowledge cutoff**, supports Structured Outputs, and says the API free tier is not supported. It describes snapshots as behavior locks, but the currently rendered snapshot list shows only the `gpt-5.6-sol` identifier rather than a dated snapshot. | Use documents published strictly after 2026-03-01 00:00 UTC, disable tools/retrieval, provide only the supplied text and metadata, cache outputs, and call the result a silver reference rather than ground truth. Record whether the run used the API or the Codex model surface. |
| `google/flan-t5-xl` | The official [FLAN-T5-XL model card](https://huggingface.co/google/flan-t5-xl) does **not** disclose a training/knowledge cutoff. It links the paper published 2022-10-20. The official [repository commit history](https://huggingface.co/google/flan-t5-xl/commits/main) shows the initial weights uploaded 2022-10-21; the pinned `7d6315d...` revision is a 2023-11-28 safetensors conversion, not documented retraining. | A date after the 2022 release is a conservative **release-date proxy**, not an official cutoff. Say this explicitly. Pinning the revision and running offline freezes weights but does not create missing provenance. |
| `meta-llama/Llama-3.1-8B-Instruct` | Meta's official [model card](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) gives a **December 2023 pretraining-data cutoff**, a 2024-07-23 release date, and calls it a static model trained on an offline dataset. | Post-release documents are conservative. For the three-model matched experiment, the stricter GPT cutoff dominates, so use 2026-03-01 or later. |

Using 2026-03-01 as the shared lower bound is after the documented GPT-5.6 Sol
cutoff, after Llama 3.1's documented cutoff and release, and after FLAN-T5-XL's
release proxy. It reduces direct temporal contamination risk; it does not cure
later-edited article text, retrospective articles, duplicates, or reference-model
annotation error.

The actual silver-reference inputs were not symmetric with the local-model
inputs. GPT received the headline, reconstructed parent full text, and target
metadata, but no provider ticker tags. Each local model receives its applicable
text variant, the same target metadata, and only ticker tags actually returned
by the provider. The same full-text-derived GPT labels are used to score both
the description and full-text arms. Full text therefore has a structural
information advantage relative to the silver reference; the experiment
measures information-retention agreement, not objective accuracy or a symmetric
model contest.

## Procurement interpretation

1. **Quant data:** use the repository's complete adjusted Alpaca SIP 15-minute
   archive. Massive Basic REST bars work only for the recent entitlement
   window, and the free flat credentials did not permit object retrieval, so
   neither path can reproduce the full existing panel.
2. **Deterministic news:** every record exposed through every cursor in the
   configured 30-stock, five-benchmark, and SPY query scope has been
   materialized. Its stable IDs, timestamps, tags, descriptions, and pagination
   are sufficient for the exploratory Q+D block. They are not sufficient for
   a strict point-in-time training claim: cursor completeness is not exhaustive
   market-wide coverage, and the feed lacks first-seen times, update history,
   and historical article versions. The undocumented pre-two-year access also
   needs a reproducibility warning.
3. **If ordinary-history access closes:** Stocks Starter is the least expensive
   Massive upgrade, subject to written confirmation of endpoint-specific News
   history because the two official pages conflict.
4. **Full-text LLM features:** do not buy the Benzinga partner feed for the
   current frozen extraction pipeline. On the locked 228-document evaluation,
   chunked retrieved bodies reduced mean macro-F1 versus ordinary Massive
   descriptions by 0.0908 for FLAN-T5-XL and 0.0593 for Llama 3.1. The
   Benzinga-only slices were also negative. Only Benzinga documents an article
   body and update timestamp, but body coverage is optional and version history
   is not guaranteed. The public-page benchmark mixes 204 Motley Fool and 96
   Benzinga pages, so it cannot validate paid-feed coverage. Revisit the
   partner feed only after a different long-context or target-conditioned
   extractor demonstrates a need for bodies and a matched paid-feed sample
   shows incremental value, with written confirmation of historical body
   coverage, article-version behavior, and retention rights.
5. **Alpha premium:** buy for throughput, not presumed text quality. The official
   documentation does not establish that premium supplies fuller articles than
   the free `NEWS_SENTIMENT` payload.
6. **Forward-only alternative:** poll the ordinary endpoint on a fixed cadence
   and retain every provider-ID/text-hash version with local first-seen,
   last-seen, and retrieval timestamps. This can make future observations
   point-in-time auditable, but it does not make the retrospective archive
   version safe.
