# Deterministic and semantic news feature design v2

Status: **D2-Normalized construction and the available-data deterministic
training ladder are complete. The shared semantic input corpus and the FLAN
W17 and GPT R70 construction pipelines are ready; full semantic inference,
daily L panels, and downstream semantic training have not run**.

Execution disposition (2026-07-29): the full v2 FLAN W17 and GPT R70
workloads are deferred because their compute/cost exceeds the desired budget.
Their definitions and ready pipelines remain frozen. The separately named
[cost-bounded v3 design](../v3/README.md) proposes W17-Lite, ordered
global-only G40, and a later schema-incomplete factorized R70-Lite without
reinterpreting v2 or authorizing inference/training.

This document is the canonical design contract for the next news experiment.
It does not change the inputs, protocol, artifacts, or conclusions of the
completed [Q+D v1 experiment](../v1/README.md). The exact v1 construction
remains documented in
[`docs/q_plus_d_massive.md`](../../../docs/q_plus_d_massive.md).

The v2 design has three blocks:

1. `D2-Normalized`: 30 deterministic features designed to be less sensitive
   to ticker popularity and source coverage than v1 D43, without treating
   source/title counts as independent propagation signal.
2. `WLLM17`: a deliberately small, abstention-aware semantic contract that can
   be populated by either a weak or strong extractor.
3. `RLLM70`: a richer theoretical-reference contract that can likewise be
   attempted by either extractor, subject to much stricter completeness and
   quality gates.

The feature contract and the extractor are separate dimensions. The v2
training ladder crosses both contracts with FLAN-T5-XL and GPT-5.6 Sol. No
other LLM enters v2. `RLLM70` is a design target, not a claim about either
model's accuracy; even a strong model's labels remain silver annotations until
they pass human-grounded, point-in-time validation.

Here `D2` means deterministic contract version 2. It is unrelated to the T2
forecast horizon or the v1 ladder's internal D2 cue block.

The design and execution record for Q+D, Q+L, and Q+D+L is in the
[v2 training ladder](TRAINING_LADDER.md). The completed deterministic
comparison is in the
[final results](training/comparisons/final/RESULTS.md), and the semantic
blockers are recorded in the
[semantic readiness ledger](training/semantic/STATUS.md).

The retrospective ordinary-Massive archive produced 71,760 complete D2
stock-days and an exact 27,510-row Q56+D2 modeling panel. D0, D1, D3, D4,
the validation-gated D5 challenger, and both falsification controls were
trained. The matched D43 comparator was skipped because its exact historical
contract was not rebuilt on the v2 routing universe. The shared semantic
corpus now contains 466,902 article-target assignments, 55,197 assigned
articles, and 27,510 stock-days, including 204 no-candidate stock-days. No
complete WLLM17 or RLLM70 article inference or daily panel exists, so no
semantic model was fit.

## Extractor-contract arms and readiness

The four immutable semantic arms are:

| Arm | Current evidence | Readiness |
|---|---|---|
| `W17__flan_t5_xl` | Pinned coarse XL runner, resumable inference, fail-closed daily aggregator, exact silver-only acceptance lock, passed full tokenizer preflight, and passed CUDA smoke | Construction pipeline ready; full inference/daily panel/training have not run |
| `W17__gpt_5_6_sol` | Fine/coarse bounded references prove the schema shape | Not pursued in the current construction pass; future W17 remains derived from accepted GPT R70 fine labels |
| `R70__flan_t5_xl` | A generic fine FLAN runner exists, but the active XL wrapper is coarse-only | Not pursued in the current construction pass; no pinned fine-XL workflow, daily panel, or training |
| `R70__gpt_5_6_sol` | Full offline two-view Batch, retry, collection, cleanup, merge, adjudication, and daily aggregation pipeline; final hash-bound preflight passed | Offline pipeline ready; no paid call, complete inference, daily panel, or training |

No model-qualified daily semantic panel currently exists. The decision-safe
union of forecast windows contains 57,181 unique documents, including 56,264
with descriptions, from 2022-10-31 09:00 ET through strictly before
2026-06-30 09:00 ET. Deterministic C/I/P routing assigns 55,197 of those
articles to 466,902 target-specific records over the exact 27,510-stock-day
scope. The candidate assignment and bounded common text artifacts are
complete. What remains missing is complete model inference and daily semantic
aggregation. The current peer universe is static rather than effective-dated,
and the archive remains non-version-safe and incomplete for untickered macro
news.

The historical fine-schema local result used FLAN-T5-Large, not XL, and
reported only 0.177 mean macro-F1 across nine fine fields. It proves that the
schema can be emitted, not that `R70__flan_t5_xl` is usable. The active
FLAN-T5-XL coarse result is stronger but still missed every frozen field
threshold.

Primary semantic comparisons use identical hashed input bytes. The shared view
keeps the complete trimmed headline and, when a description exists, appends
two line feeds plus a deterministic leading excerpt of at most 512 Unicode
code points. It prefers a whitespace boundary within the last 64 code points;
otherwise it cuts exactly at the cap. Missing descriptions use a separately
recorded headline-only view. Source, retained, and omitted character counts
and hashes are recorded. The whole corpus passed preflight with the pinned
FLAN tokenizer, and silent runner-side truncation is forbidden. The same
bounded `model_text` bytes are supplied unchanged to GPT.

The completed FLAN tokenizer preflight covered all 466,902 assignments:
4,202,118 logical prompts, 3,070,758 actually tokenized prompts after exact
target-invariant caching, and zero violations. Maximum observed prompt lengths
were 398 tokens for shock scope, 427 for event family, and 360 for information
status, all below the 512-token limit. Its one-record CUDA float16 smoke
completed on the RTX 3070 Ti with one terminal success, no failure, and no
truncation.

Candidate routing may use complete source descriptions, provider tickers,
keywords, and other provider metadata. Those fields and the resulting
full-source candidate-role flags are not exposed to either extractor.
Extractor-visible entities and role flags are recomputed solely from bounded
`model_text`; those visible flags, not full-source routing metadata, govern
field applicability after an article has entered the candidate corpus.

The GPT R70 workload preflight covers 933,804 two-view requests totaling
3,613,888,330 JSONL bytes and projects 934 conservative 1,000-request files.
The final hash-bound manifest is complete; its largest request is 4,458 bytes.
This sizing result is not authorization to spend. No paid request has been
sent. A paid pilot or full run additionally requires an API credential,
explicit confirmation that licensed text may be processed, a hard request
budget, and separate user authorization.

GPT-5.6 Sol is an explicitly future-contaminated exploratory oracle on the
historical panel. OpenAI currently documents a 2026-02-16 knowledge cutoff for
[`gpt-5.6-sol`](https://developers.openai.com/api/docs/models/gpt-5.6-sol).
Every GPT feature/result therefore remains non-confirmatory even when prompts
contain only eligible article text and model tools are disabled.

## Why v1 is being redesigned

The v1 experiment established pipeline feasibility, but not reliable
incremental news signal:

- A5, the full Q56+D43 Elastic Net, reduced T2 ETF mean squared error by
  3.65% relative to matched Q-only Elastic Net.
- Twenty-session-stale D43 outperformed contemporaneous D43 on every A5
  target. The gain therefore did not isolate timely news.
- Correct-stock news beat the fixed wrong-stock control, so D43 retained some
  target identity, but persistent coverage and attention could explain part of
  that result.
- Mean relevant articles per stock-day fell from about 30.7 in 2023 to 10.0 in
  2025, then rose to 15.8 in 2026.
- Direct-target coverage differed by roughly 49 times between the least- and
  most-covered stocks.
- One third of stock-days had no direct-target article, even though only 84 of
  27,510 rows had no broadly "relevant" article.
- Each provider article was assigned to about 9.75 stock rows on average.
  Twenty-six of the 40 materialized v1 fitting columns were constant across
  the six stocks of a sector on at least 90% of sector-dates.
- `observed_relevant_article_count` and exact normalized-title cluster count
  had correlation 0.9999; macro article count and the macro lexical-cue count
  had correlation 0.9981; unique-peer count and peer coverage were exact
  transformations of one another.

The precise diagnostics and source hashes are recorded in the
[v2 design audit](DESIGN_AUDIT.md). That audit was recomputed read-only from
the frozen v1 panels and results and created no v2 data artifact at the time;
the later D2 and semantic-corpus constructions are separately manifested.

These facts point to four design problems: broad assignment diluted
target-specific news, raw levels encoded provider/ticker identity, several
features duplicated the same measurement, and current final article text could
not establish historical availability.

## Provider assumption and claim boundary

For design purposes, the separate Massive Benzinga partner feed is treated as
the most feature-rich documented historical candidate the project could
potentially acquire.
Massive's current
[endpoint documentation](https://massive.com/docs/rest/partners/benzinga/news)
advertises history since 2009, more than 600 Benzinga-authored articles per
day, and the structured fields described below; its
[pricing page](https://massive.com/pricing?product=stocks) lists Benzinga News
as a separate partner dataset. Those are product-level descriptions, not an
audit of the records available under this project's eventual entitlement. The
feed has not been locally validated: the access probe returned 403, so actual
accessible completeness by year, ticker, and text field remains unknown. This
is a planning assumption under the user's data constraint, not evidence that
the feed is complete. See the
[provider source matrix](../../../docs/news_provider_source_matrix.md).

The documented partner schema may provide a richer timestamp/text candidate
through stable article identity, `last_updated` metadata, provider
channels/tags, and optional teaser or body text. It does not by itself prove:

- when a record was first visible to this project;
- that every historical revision can be retrieved;
- that the returned body is the exact version visible at a past cutoff; or
- that body, ticker, and channel coverage are stable through time.

V2 therefore defines three separate source profiles. They must never be
silently concatenated into one panel.

| Profile | Availability rule | Permitted claim |
|---|---|---|
| `D2-ordinary-retrospective` | Use today's returned headline/description with publication time as the availability proxy | Exploratory only; neither text-version-safe nor `primary_training_eligible` |
| `D2-benzinga-retrospective` | Use the final-record rule below when `last_updated` exists | Potentially better timestamped proxy, conditional on an acquisition/coverage audit; still not strict point-in-time evidence |
| `D2-prospective-versioned` | Immutable locally observed versions and first-seen times | Eligible for confirmatory work if collection audits pass |

If a paid archive is acquired, compare it with ordinary Massive on their exact
matched stock-date intersection before attributing any change to feature
quality. Use headline plus provider teaser/description as the primary text
contract. Treat article bodies as a separately named sensitivity because body
availability and revision behavior may differ from teaser availability.

## Information-time contract

For official session \(t\), let \(c_t\) be 09:00 ET and let \(u_{av}\) be the
availability time of article version \(v\). The v2 news window is

$$
\mathcal W_t=\{(a,v):c_{t-1}\leq u_{av}<c_t\}.
$$

The right boundary is strict. A record first available exactly at 09:00 is not
used for that day's forecast.

Availability is defined by source profile:

1. For a prospectively collected immutable version,

   $$
   u_{av}=\max(published_{av},lastUpdated_{av},localFirstSeen_{av}).
   $$

   The maximum is over non-null timestamps, and `lastUpdated` must belong to
   the exact immutable text hash observed for version \(v\). A final provider
   timestamp must never be back-applied to an earlier locally saved version.
2. For a retrospective Benzinga final record,

   $$
   u_a=\max(published_a,lastUpdated_a),
   $$

   and the row is labeled `historical_final_version_conservative`. This can
   exclude an article that existed earlier but was revised later; it still
   cannot reconstruct the earlier text and is not version-safe. If
   `last_updated` is null, fall back to publication time and label the record
   `historical_published_proxy_unversioned`; do not call it final-version
   conservative.
3. For ordinary retrospective Massive news, \(u_a=published_a\) is an
   explicitly non-version-safe proxy. A preregistered 15-minute embargo is a
   useful sensitivity, not a cure.

Every zero-valued news measurement is valid only when every query root that
could contribute to that stock-day was cursor-complete. Otherwise the
measurement is missing, not zero. Even a valid zero means "no observed record
in the frozen provider/product/query universe," never "no public news
existed." In particular, ordinary Massive does not establish complete
untickered macro coverage.

## Article identity, assignment, and roles

The primary unit is a provider-ID-deduplicated article version. Cross-query
duplicates do not create new observations. Exact or near-duplicate title
groups may later be used as propagation sensitivities, but they must not be
called causal events.

Entity aliases, sector membership, and benchmark membership must be
effective-dated. If historical membership is unavailable, outputs must say
`research_peer_set` rather than imply point-in-time index membership.

For target stock \(i\), article \(a\), and session \(t\), define:

- \(D_{iat}=1\), direct target, when the provider tags \(i\), or the headline
  or eligible provider text contains an exact effective-dated ticker/company
  alias for \(i\).
- \(S_{iat}=1\), sector common, when the article has an explicit sector
  benchmark tag, an unambiguous curated multi-word sector expression, or at
  least two distinct effective-dated entities from the research peer set.
  Bare words such as "energy" are insufficient.
- \(M_{iat}=M_{at}=1\), macro common, when a provider macro channel/tag or a curated
  macro phrase is present. An SPY tag or the fact that an SPY query retrieved
  the article is never sufficient by itself.
- \(C_{iat}=S_{iat}\lor M_{at}\), any common article.
- \(I_{iat}=D_{iat}(1-C_{iat})\), target-idiosyncratic.
- \(TC_{iat}=D_{iat}C_{iat}\), target plus common.
- \(P_{iat}=1\) when the article is neither direct nor common and identifies
  exactly one non-target peer, and zero otherwise.

Thus \(C\), \(I\), and \(P\) are disjoint article roles. \(S\) and \(M\) may
overlap, \(TC\) is a subset of \(C\), and \(D=I+TC\) in count form. An article
identifying two peer-set firms is sector-common, not peer-idiosyncratic. Query
provenance can establish collection coverage, but never semantic scope.

Let

$$
N_X(i,t)=\sum_{a\in\mathcal W_t}X_{iat}
$$

for \(X\in\{D,I,TC,P,S,M,C\}\). All article-level cue flags below count each
provider article at most once.

## `D2-Normalized`: primary deterministic block

The primary deterministic block has exactly 30 features. Raw provider counts,
publisher identity, and collection-completeness flags remain in audit tables;
they are not primary predictors.

A primary D2 row is eligible only when the current session and all prior 126
official sessions needed by the ranks are complete within the same frozen
source profile, and when the effective peer set has \(K_{it}\geq2\). This also
covers the 63-session attention history. Incomplete-history or
incomplete-query rows are excluded from the primary D2 comparison rather than
imputed or represented by extra hidden flags, so `D2-Normalized` remains an
exact 30-feature complete-case contract.

### Normalized activity and commonality (6)

For a nonnegative measurement \(x_t\), define its trailing 126-session midrank
using only prior official sessions:

$$
R_{126}(x_t)=\frac{
\sum_{h=1}^{126}\mathbf 1[x_{t-h}<x_t]
+\frac12\sum_{h=1}^{126}\mathbf 1[x_{t-h}=x_t]
}{126}.
$$

A full 126-session history is required; otherwise the D2 row is ineligible.
Ranks are computed separately within source profile and target/sector series.

| Feature | Definition |
|---|---|
| `d2_target_direct_intensity_midrank_126` | \(R_{126}(N_D)\) for the target |
| `d2_peer_idio_intensity_midrank_126` | \(R_{126}(N_P)\) for the target's peer set |
| `d2_common_intensity_midrank_126` | \(R_{126}(N_C)\) for the target sector |
| `d2_observed_no_direct_target_article` | \(\mathbf 1[N_D=0]\) |
| `d2_common_article_share` | \(N_C/\max(1,N_C+N_I+N_P)\) |
| `d2_target_common_share` | \(N_{TC}/\max(1,N_D)\) |

The explicit no-direct flag distinguishes a genuine zero denominator from a
small share. It is non-null only when collection coverage is complete.

### Entity breadth and relative attention (6)

Let \(m_j\) count articles with entity evidence for peer-set entity \(j\),
counting an entity at most once per provider article. Entity evidence is the
same union used by the role rules: a provider ticker tag or an exact
effective-dated ticker/company alias in eligible text. Let \(J_{it}\) be the
effective research peer set, including the target, and
\(K_{it}=|J_{it}|\geq2\).

| Feature | Definition |
|---|---|
| `d2_observed_no_peer_set_entity_mention` | \(\mathbf 1[\sum_jm_j=0]\) |
| `d2_peer_set_entity_hhi` | \(\sum_j(m_j/\sum_km_k)^2\), or 0 when \(\sum_km_k=0\) |
| `d2_peer_coverage_ratio` | Mentioned non-target peers / \((K_{it}-1)\) |
| `d2_target_peer_co_mention_share` | Direct articles naming at least one peer / \(\max(1,N_D)\) |
| `d2_target_attention_share_delta_63` | Current target entity share minus its median over the prior 63 sessions |
| `d2_macro_share_of_common` | \(N_M/\max(1,N_C)\) |

For the attention delta, the current target entity share is
\(m_i/\sum_jm_j\), defined as zero when the denominator is zero. A complete
63-session trailing history is required. The explicit no-peer-set-entity flag
must be interpreted jointly with HHI and target attention: when it is one,
their zero convention means no entity activity, not diffuse concentration or
zero economic attention.

### Availability-aware timing (4)

For an eligible article with age \(h_a=(c_t-u_a)\) hours, define

$$
w_a=\exp\left(-\log(2)h_a/12\right).
$$

| Feature | Definition |
|---|---|
| `d2_target_mean_recency_weight_12h` | \(\sum_{a:D=1}w_a/\max(1,N_D)\) |
| `d2_common_mean_recency_weight_12h` | \(\sum_{a:C=1}w_a/\max(1,N_C)\) |
| `d2_target_premarket_article_share` | Direct articles available from 04:00 through before 09:00 ET on \(t\) / \(\max(1,N_D)\) |
| `d2_common_premarket_article_share` | Common articles available from 04:00 through before 09:00 ET on \(t\) / \(\max(1,N_C)\) |

Mean weights deliberately separate freshness from volume. V1's
hours-since-latest fields and their imputation flags are not retained.

### Scope-conditioned lexical composition (14)

Retain the auditable v1 dictionaries, but apply each cue to explicit article
roles and normalize by the relevant role count. For

```text
earnings_guidance
product_demand
supply_capacity
regulation_legal
corporate_analyst
```

construct:

```text
d2_target_<family>_article_share
d2_common_<family>_article_share
```

where the target denominator is \(\max(1,N_D)\) and the common denominator is
\(\max(1,N_C)\). These ten fields are joined by:

```text
d2_target_positive_surprise_article_share
d2_target_negative_surprise_article_share
d2_common_positive_surprise_article_share
d2_common_negative_surprise_article_share
```

The macro cue is removed because strict \(M\) already defines macro-common
articles. Cue absence means "not detected by this dictionary," not semantic
absence. The 14 shares are not a simplex: cue dictionaries may overlap, a
direct common article enters both target and common denominators, and both
positive and negative cue flags may fire on one mixed article.

### Optional deterministic sensitivities

`D2-Levels` adds five provider-sensitive measurements only in a separately
named ablation:

```text
d2_log1p_target_idio_article_count
d2_log1p_target_common_article_count
d2_log1p_peer_idio_article_count
d2_log1p_sector_common_article_count
d2_log1p_macro_common_article_count
```

These are respectively \(\log(1+N_I)\), \(\log(1+N_{TC})\),
\(\log(1+N_P)\), \(\log(1+N_S)\), and \(\log(1+N_M)\).

They are never substituted silently for `D2-Normalized`.

After a leakage-safe near-duplicate method has been validated, an optional
three-feature `D2-Propagation` sensitivity may add:

```text
d2_target_near_duplicate_ratio
d2_common_near_duplicate_ratio
d2_common_max_cluster_source_count
```

Clusters may use only records available before \(c_t\), may not be bridged by a
future article, and remain duplication groups rather than inferred events.

## Relationship to the completed v1 D43

V1 remains the historical record: 40 recommended materialized fields plus
three model-time missingness flags. V2 does not overwrite or rename that
artifact.

| V1 concept | V2 treatment |
|---|---|
| Window `(previous 09:00, current 09:00]` | Use the non-overlapping, decision-safe window `[previous 09:00, current 09:00)` |
| Broad relevant/direct/target-only/peer/common/mixed raw counts | Recompute strict \(D,I,TC,P,S,M,C\); use three 126-session midranks in the primary block |
| SPY-derived or broadly inferred macro-common assignment | Require explicit macro metadata/text; SPY retrieval alone is audit provenance |
| Multi-ticker share, unique-peer count, peer coverage, two balance formulas | Remove redundancies; keep one peer coverage measure, a no-entity flag, HHI, target co-mention share, and common share |
| Raw target share of entity mentions | Replace with the target's deviation from its own trailing 63-session median |
| Eight unscoped lexical cue counts | Replace with 14 target/common role-conditioned shares; remove redundant macro cue |
| Hours since latest, recency-weighted counts, and two session shares | Replace with four role-specific mean-recency and premarket-share fields |
| Source count/entropy and exact-title propagation fields | Remove from the primary predictive block; retain provider/source data for audits and optional validated propagation |
| Sixty-session MAD burst | Replace with full-history-required 126-session midranks |
| Missingness indicators | Make current/prior source completeness and 126-session history primary row-eligibility conditions; D2-30 has no history-start imputation flags |
| `observed_no_relevant_news` | Replace with the more informative no-direct-target flag |

The stale-news result requires each fitted v2 endpoint to repeat lagged and
wrong-stock falsification controls. The completed D2 branch did so; the
untrained semantic branches must do the same after daily L panels exist.

## Shared semantic aggregation contract

Both LLM designs operate on target-article assignments already admitted by
deterministic routing. LLM output never decides whether an article existed,
when it was available, or whether a query completed.

Classify headline plus provider teaser/description first. Full body is a
separate sensitivity. To reduce the weight of syndicated copies, an online
near-duplicate group \(g(a,t)\) may be used only as a duplication weight:

$$
\omega_{a,t}=
\frac{\exp[-\log(2)h_a/12]}{|g(a,t)|}.
$$

Singleton articles have \(|g|=1\). The group contains only versions available
before \(c_t\). A future panel must hash and name its duplication method. Until
a cutoff-safe near-duplicate method is validated, every provider-ID-deduplicated
article is a singleton; source profiles using different methods cannot be
compared as if their semantic aggregation were identical.

Let \(\mathcal A_{it}\) be the deterministic \(C\cup I\cup P\) candidate set.
For semantic field \(k\):

- \(E_{a,k}=1\) means the field is structurally applicable before its label is
  scored;
- \(A_{a,k}=1\) means a usable predictive label passed its frozen
  field/class-specific acceptance rule; and
- \(q_{a,k}\) is its quality weight.

For WLLM, all three fields are applicable to every candidate. For RLLM,
relevance is applicable to every candidate; accepted `irrelevant` is a valid
structural rejection that stops downstream routing. Scope, event, breadth,
surprise, status, and channels apply to routed relevant articles. Target
direction additionally requires extractor-visible \(D=1\), sector direction
requires extractor-visible \(C=1\), and peer effect requires
extractor-visible direct-target plus peer-entity evidence. These
extractor-visible flags are recomputed from bounded `model_text`; the
full-source `candidate_roles` that admitted an article are provenance and do
not silently broaden applicability. Applicability is frozen from those
visible deterministic roles and any accepted upstream relevance route before
that field is extracted. When \(E=0\),
`not_applicable` is the expected structural audit state. When \(E=1\), a
returned `not_applicable` is unusable for that field and sets \(A=0\); it never
retroactively rewrites \(E\).

For a single-label field and accepted predictive class \(\ell\), the daily
class share is

$$
L_{i,t,k,\ell}=
\frac{\sum_{a\in\mathcal A_{it}}
\omega_{a,t}E_{a,k}q_{a,k}A_{a,k}
\mathbf 1[\widehat y_{a,k}=\ell]}
{\sum_{a\in\mathcal A_{it}}
\omega_{a,t}E_{a,k}q_{a,k}A_{a,k}}.
$$

Transmission channels are multi-label. Let \(\mathcal L_{ch}\) be the 12
predictive channels. Use a share of accepted channel claims:

$$
ChannelShare_{i,t,\ell}=
\frac{\sum_a\omega_{a,t}E_{a,ch}
A_{a,ch,\ell}q_{a,ch,\ell}}
{\sum_a\omega_{a,t}E_{a,ch}
\sum_{r\in\mathcal L_{ch}}A_{a,ch,r}q_{a,ch,r}}.
$$

Each channel has its own acceptance and quality weight. These 12 values sum to
one over accepted predictive channel claims; they are all missing when the
denominator is zero. For channel coverage, \(A_{a,ch}=1\) when at least one
predictive, non-`unclear` channel is accepted.

Usable-label coverage deliberately excludes \(q\):

$$
Coverage_{i,t,k}=
\frac{\sum_a\omega_{a,t}E_{a,k}A_{a,k}}
{\sum_a\omega_{a,t}E_{a,k}}.
$$

The mean accepted \(q\) is recorded separately; it must not be called
coverage. `insufficient`, `unclear`, and `unknown` are epistemic abstentions.
`irrelevant` and `not_applicable` are valid structural routing states. All raw
states and rates remain in the audit output even when they are excluded from
predictive class shares.

For each field, coverage is missing when
\(\sum_a\omega_{a,t}E_{a,k}=0\). It is zero only when applicable mass exists
but every applicable prediction abstains. If \(\mathcal A_{it}\) is empty,
class shares and semantic coverage are missing, and the semantic block's own
no-eligible-article feature equals one. If candidates exist but every
applicable prediction abstains, class shares are missing, applicable-field
coverage is zero, and the no-eligible feature equals zero. An unavailable
class share is never encoded as zero.

If any contributing source/query scope is incomplete, the entire semantic
stock-day is missing and ineligible, including for a `Q + L` comparison.
Partial nonzero aggregates are never admitted. The no-eligible flag is missing
in that case rather than zero or one.

WLLM17 and RLLM70 are raw pre-imputation block sizes. If a future model adds
training-fold-derived missingness indicators for nullable semantic fields,
those columns are outside the named 17- and 70-feature contracts and must be
reported separately.

## `WLLM17`: noisy-extractor design

Contract ID: `weak-news-semantics-v1`.

This block assumes extraction quality similar to the current FLAN-T5-XL v1.1
evaluation. Against deterministically coarsened GPT-5.6 Sol silver labels, its
mean macro-F1 was 0.464. Field macro-F1 was 0.468 for scope, 0.478 for event
family, 0.580 for information status, and 0.328 for directional alignment.
None passed its frozen threshold. The deterministic relevance gate scored
0.935 F1.

Consequently:

- relevance remains deterministic;
- `directional_alignment` is excluded;
- hard labels are never used without field/class-specific acceptance;
- `other_or_unclear` and `unclear` are abstentions, not predictive classes;
- raw generation scores and margins are diagnostics, not probabilities; and
- this block defaults to `primary_training_eligible = false` until evaluated
  against human labels rather than only silver annotations.

The ladder distinguishes primary eligibility from a separately
flagged exploratory downstream fit. After complete corpus inference, frozen
silver-derived acceptance rules, and coverage checks, W17 may be fitted only
as `exploratory_silver_fit`; that does not promote it to confirmatory or
production training.

Each article-field record stores the predicted value, accepted flag, raw and
adjusted scores, top-two margin, calibration split, estimated agreement or
precision lower bound, and one of:

```text
gate_inapplicable
unclear
low_margin
unvalidated_class
low_precision
invalid
truncated
missing
```

Until a human calibration sample exists, the quality statistic must be named
`estimated silver agreement`, never accuracy or precision.

Thresholds must be frozen by field and class on an earlier annotated split.
For the weak contract, accepted labels use \(q_{a,k}=1\); model scores are not
silently treated as calibrated correctness probabilities. Every WLLM slot
also carries an audit-only `enabled_by_calibration` flag; under the current
silver-only evaluation no slot is approved for confirmatory training.

### WLLM17 feature list

Twelve accepted-class shares:

```text
wllm_scope_share_idiosyncratic
wllm_scope_share_common
wllm_scope_share_mixed

wllm_event_share_earnings_guidance
wllm_event_share_product_demand
wllm_event_share_supply_capacity
wllm_event_share_regulation_legal
wllm_event_share_corporate_analyst
wllm_event_share_macro_market

wllm_status_share_confirmed
wllm_status_share_anticipated
wllm_status_share_rumor_or_opinion
```

Five derived/coverage fields:

```text
wllm_scope_common_minus_idiosyncratic
wllm_observed_no_eligible_semantic_article
wllm_coverage_shock_scope
wllm_coverage_event_family
wllm_coverage_information_status
```

The difference is `common share - idiosyncratic share`. The no-eligible flag
is one only when the cutoff-safe deterministic candidate set is empty and its
source/query scope is complete; otherwise it is zero or missing when coverage
cannot be established.

Do not derive weak-model direction, target/sector effects, peer effects,
surprise, affected breadth, transmission channels, entities, evidence,
novelty, materiality, causality, or LLM relevance features.

## `RLLM70`: high-accuracy reference design

Contract ID: `reference-news-semantics-v1`.

This theoretical block assumes a substantially more accurate extractor, such
as a carefully controlled GPT-5.6 Sol workflow. The output schema is the
existing fine-grained
[`stock_sector_news_semantics`](../../../config/news_feature_schema.json)
contract:

```text
relevance
event_scope
event_type
affected_breadth
target_direction
sector_direction
peer_effect
explicit_surprise
information_status
transmission_channels
affected_companies
affected_sectors
evidence
```

The extractor must use two independently designed, pinned prompt/views with
deterministic decoding, validate every enum/entity/evidence span, and
adjudicate disagreements without using future market data. Re-running the same
greedy prompt is not an independent annotation. Fine outputs are
deterministically coarsened when a coarse label is needed; a second
coarse-model call is not made. Self-reported confidence is not accepted as
calibration.

Training eligibility requires:

- human labels sampled across time, ticker, publisher, and event family;
- cross-fitted field/class calibration and frozen acceptance thresholds;
- a pinned model revision, tokenizer, prompt, decoder, and schema;
- point-in-time eligible input text;
- stable evidence-substring and entity-grounding checks; and
- abstention when the text does not support a label.

With human calibration, \(q_{a,k}\) is the cross-fitted estimated probability
that accepted field \(k\) is correct. Without it, \(q=1\) and the entire block
remains a silver-only research artifact. Multi-label channels use a separately
calibrated \(q_{a,ch,\ell}\) for each accepted channel.

### RLLM70 class and accepted-channel-claim shares (55)

The suffixes below define one feature per listed accepted class or channel.

| Prefix | Accepted suffixes | Count |
|---|---|---:|
| `rllm_relevance_share_` | `direct_target`, `sector_or_peer`, `macro_relevant` | 3 |
| `rllm_scope_share_` | `firm_specific`, `peer_specific`, `sector_wide`, `macro_market`, `mixed` | 5 |
| `rllm_event_share_` | `earnings`, `guidance`, `product_technology`, `demand_customer_contract`, `supply_chain_capacity`, `regulation_trade_policy`, `analyst_action`, `corporate_action`, `legal_governance_operations`, `macro_market`, `other` | 11 |
| `rllm_breadth_share_` | `single_firm`, `several_same_sector`, `cross_sector`, `broad_market` | 4 |
| `rllm_target_direction_share_` | `positive`, `negative`, `neutral`, `mixed` | 4 |
| `rllm_sector_direction_share_` | `positive`, `negative`, `neutral`, `mixed` | 4 |
| `rllm_peer_effect_share_` | `same_direction`, `opposite_direction`, `mixed`, `none_stated` | 4 |
| `rllm_surprise_share_` | `positive`, `negative`, `mixed`, `none` | 4 |
| `rllm_status_share_` | `confirmed`, `scheduled_or_expected`, `rumor_or_unconfirmed`, `analysis_or_opinion` | 4 |
| `rllm_channel_accepted_claim_share_` | `demand`, `pricing_margin`, `supply_capacity`, `technology_product`, `competition`, `regulation_trade`, `rates_financing`, `macro_growth`, `geopolitical`, `capital_allocation`, `legal_operational`, `other` | 12 |

`insufficient`, `unclear`, and `unknown` are epistemic abstention states.
`irrelevant` and `not_applicable` are valid structural states that control
routing/applicability. All five are excluded from the listed predictive
classes; none means a zero economic effect.

### RLLM70 coverage and derived fields (15)

Ten field-group coverage features:

```text
rllm_coverage_relevance
rllm_coverage_event_scope
rllm_coverage_event_type
rllm_coverage_affected_breadth
rllm_coverage_target_direction
rllm_coverage_sector_direction
rllm_coverage_peer_effect
rllm_coverage_explicit_surprise
rllm_coverage_information_status
rllm_coverage_transmission_channels
```

Five derived/quality features:

```text
rllm_scope_common_minus_idiosyncratic
rllm_target_sector_same_direction_share
rllm_target_sector_opposite_direction_share
rllm_observed_no_eligible_semantic_article
rllm_mean_accepted_quality_weight
```

`scope_common_minus_idiosyncratic` is the sum of sector-wide and macro-market
scope shares minus the sum of firm- and peer-specific shares.

The two target-sector direction shares are deliberately narrower than the
repository's coarse directional-alignment mapping; explicit peer relationships
remain in the four `peer_effect` features. Define

$$
J_a=E_{a,target}E_{a,sector}A_{a,target}A_{a,sector}
\mathbf 1[\widehat y_{a,target},\widehat y_{a,sector}
\in\{positive,negative\}]
$$

and \(q_a^J=\min(q_{a,target},q_{a,sector})\). The same-direction share is the
\(\omega_aq_a^J J_a\)-weighted fraction with equal signs; the
opposite-direction share is the fraction with opposite signs. Neutral and
mixed labels do not enter this joint denominator. Both shares are missing when
the joint positive/negative denominator is zero.

The no-eligible feature follows the WLLM rule. The mean accepted quality
feature is

$$
\overline q_{i,t}=
\frac{\sum_{a,k}\omega_{a,t}E_{a,k}A_{a,k}q_{a,k}}
{\sum_{a,k}\omega_{a,t}E_{a,k}A_{a,k}},
$$

counting each single-label field once. For channels, first average the
accepted per-channel quality weights within article so the multi-label field
also contributes once. The feature is missing when no usable label is
accepted.

For an exploratory extractor profile with uncalibrated \(q=1\),
`rllm_mean_accepted_quality_weight` is identically one whenever any label is
accepted. Retain it in the raw RLLM70 schema and audit output, but exclude it
as a zero-variance fitting column and report an effective 69-column semantic
matrix. It may re-enter only under a frozen, cross-fitted calibration that
makes \(q\) nonconstant.

Exact evidence grounding is a QA gate and audit statistic, not a predictor.
Its denominator includes only accepted claims for which the schema requires an
evidence span; for example, `explicit_surprise = none` does not require
surprise evidence.

After separate entity-resolution validation, an optional `RLLM-Entity4`
sensitivity may add target mention share, peer mention share, effective peer
coverage, and accepted affected-company count. It is not part of RLLM70.

## Provenance required before construction

Every article version and model output must be recoverable from immutable
metadata. At minimum record:

- provider product/dataset, stable article ID, query root, cursor/page, and raw
  payload hash;
- published, provider `last_updated`, local `first_seen`, retrieval, computed
  availability, source-profile, and cutoff timestamps;
- exact headline/teaser/body version hashes and which fields entered each
  deterministic or semantic computation;
- raw provider ticker/channel/tag payload and effective-dated entity-map hash;
- article-role flags, assignment reason, query-completeness state, and
  duplication-group ID/membership hash;
- semantic eligible-article count, field-applicability masks, accepted-label
  masks, structural-routing rates, and epistemic-abstention rates;
- semantic contract/schema/feature-list hashes;
- model and tokenizer revisions, prompt, decoder, calibration, threshold,
  routing, code, environment, and hardware identifiers;
- raw generation, parsed output, validation result, abstention reason, and
  evidence offsets/hashes; and
- `point_in_time_version_safe`, `historical_final_version_conservative`, and
  `primary_training_eligible` flags.

Raw coverage levels, publisher identity, query completion, provider product,
year, and version-safety flags are evaluation strata and audit variables. They
must not enter the primary predictive matrix.

## Training comparison and execution boundary

The ladder design and execution record are maintained separately in
[`TRAINING_LADDER.md`](TRAINING_LADDER.md). It includes:

```text
Q
Q + D43-recomputed
Q + D2-Normalized
Q + L
Q + D2-Normalized + L
```

where \(L\) is instantiated separately as `W17__flan_t5_xl`,
`W17__gpt_5_6_sol`, `R70__flan_t5_xl`, and `R70__gpt_5_6_sol`.
`D43-recomputed` was intended to be rebuilt on the exact D2 source profile and
row set; the completed v1 result is not a matched substitute. That exact
comparator was not materialized and its rung was therefore skipped.

The ladder makes joint models primary, retrains exact matched Q and Q+D
controls, repeats the 20-session stale and fixed within-sector wrong-stock
falsifications, and reserves shallow XGBoost for validation-gated challengers.
Residual correction remains deferred until a joint semantic branch succeeds
and forward-chained cross-fitted quant forecasts cover the same dates.

All retrospective profiles and every reused 2022-11-01 through 2026-06-30
block are development-only. GPT arms are additionally labeled
future-contaminated oracle diagnostics. Confirmatory evaluation requires
feature/protocol hashes frozen before inspection and an untouched,
prospectively versioned period beginning strictly after 2026-06-30; no
provider upgrade can turn the inspected v1 dates into a final holdout.

The D2 builder, joined feature Parquet, locked deterministic protocol,
available-data deterministic fits, controls, and final comparison now exist.
The hash-bound shared semantic corpus, the FLAN W17 construction pipeline and
silver-only acceptance lock, and the GPT R70 offline pipeline and workload
preflight also exist. FLAN's full tokenizer preflight and CUDA execution smoke
passed, and GPT's final hash-bound preflight manifest is complete.
Neither arm has complete article predictions, a daily L panel, a Q+L fit, or a
Q+D+L fit. GPT W17 and FLAN R70 were not pursued in this construction pass. No
bounded benchmark fragment is promoted into a training panel.

The next execution steps, including the paid GPT gates, are specified in the
[semantic construction runbook](training/semantic/CONSTRUCTION_RUNBOOK.md).
