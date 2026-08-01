# Article-Level News Feature Protocol

Semantic schema version: `0.1.0`

Current FLAN prompt/parser contract: `flan-stock-sector-news-v0.2.0`

> **Schema history and proposed use.** This document and its JSON files remain
> the source contracts for the completed annotation/extractor evaluations.
> The fine schema supplies the article-level ontology for RLLM70, while its
> frozen fine-to-coarse mapping supplies WLLM17. The v2 experiment
> crosses both daily contracts with FLAN-T5-XL and GPT-5.6 Sol only; GPT W17 is
> derived from its fine labels rather than a second coarse call. See the
> [v2 feature design](../experiments/quant_deterministic_news/v2/README.md) and
> [training ladder](../experiments/quant_deterministic_news/v2/TRAINING_LADDER.md).
> The deterministic branch is trained, and the shared v2 semantic input corpus
> is materialized at 466,902 article-target assignments, 55,197 assigned
> articles, and 27,510 stock-days. It is an input/assignment corpus, not a
> labeled semantic panel: no complete W17 or R70 article inference, daily L
> panel, or semantic downstream model exists.
>
> The unexecuted full workloads remain frozen in v2. A separately named,
> documentation-only
> [cost-bounded v3 design](../experiments/quant_deterministic_news/v3/README.md)
> proposes a four-way FLAN event-group contract, an ordered 40-column GPT
> global-semantic subset, and a later factorized GPT R70-Lite event graph.
> R70-Lite's ordered target-relative contract is not frozen, and neither Lite
> arm modifies this schema or claims to be exact R70.

This protocol measures semantic properties of a supplied financial-news headline and summary for stock-sector coupling research. It does not ask either language model to forecast returns, volatility, correlation, beta, or trading outcomes.

The machine-readable enum source of truth is [`config/news_feature_schema.json`](../config/news_feature_schema.json).

The FLAN accuracy-improvement experiment also defines a reduced schema in [`config/news_feature_schema_coarse.json`](../config/news_feature_schema_coarse.json). It deterministically coarsens the original silver labels into four LLM fields—shock scope, event family, information status, and directional alignment—while moving relevance and explicit surprise into an auditable non-LLM layer. The original schema remains the annotation source of truth; the coarse schema is the small-model evaluation contract.

## V2 construction input profile

The current v2 shared view is constructed before either extractor runs:

1. Retain the complete normalized headline.
2. When the normalized description is nonempty, append two line feeds and a
   deterministic leading excerpt of at most 512 Unicode code points. Prefer a
   whitespace boundary within the last 64 code points; otherwise cut exactly
   at the cap.
3. Store the source-description, retained-description, and final-model-text
   hashes and their source, retained, and omitted character counts.
4. Preflight every actual FLAN prompt with the pinned tokenizer. Runner-side
   or silent tokenizer truncation is forbidden.
5. Supply the identical bounded `model_text` bytes to GPT for the common
   comparison.

Complete source descriptions, provider tickers, keywords, and other provider
metadata may be used by the deterministic router to decide which
article-target assignments enter the corpus. They are not extractor inputs.
The corpus stores separate full-source `candidate_roles` and
`routing_detected_entities` for provenance, while extractor-visible `roles`
and `detected_entities` are recomputed solely from bounded `model_text`.
Neither FLAN nor GPT receives provider tickers, keywords, or candidate-role
flags.

The FLAN W17 runner extracts only shock scope, event family, and information
status under the coarse contract; deterministic relevance remains outside the
LLM and directional alignment is excluded from WLLM17. The GPT R70 runner uses
the complete fine schema below through two independently designed,
schema-constrained views, followed by validation and deterministic
adjudication. GPT output remains a future-contaminated exploratory silver
oracle, never ground truth or confirmatory evidence.

The FLAN code, daily aggregator, and exact permissive silver-only acceptance
lock are implemented. Its full tokenizer preflight passed all 466,902
assignments and 4,202,118 logical prompts with zero violations; the maximum
shock-scope, event-family, and information-status prompts were 398, 427, and
360 tokens under the 512-token limit. A one-record CUDA float16 smoke passed
on the RTX 3070 Ti with no failure or truncation; full inference has not
started. The GPT offline pipeline's final hash-bound full-corpus preflight
passed over 933,804 two-view requests. No request has been submitted:
paid execution requires an API credential, explicit licensed-text
confirmation, a hard request budget, and separate user authorization.

## Legacy benchmark inputs

Each benchmark record contains:

- the headline and supplied Alpha Vantage summary;
- the target company and ticker;
- the target sector and sector benchmark;
- a fixed list of known sector peers; and
- vendor-assigned tickers, which may be used only as contemporaneous entity metadata.

The v0.2 FLAN closed-label and channel prompts receive the target company, ticker, sector, known peers, headline, and article text. They intentionally omit the sector benchmark and vendor tickers because those fields encouraged copying rather than article-based classification in the legacy run. Entity and evidence prompts receive only the headline and article text so their outputs can be checked directly against the supplied source.

The annotator must not use external knowledge, later events, remembered price performance, vendor sentiment, or the benchmark sampling stratum.

For firm-query candidates, the target is the candidate's original focus ticker. Sector and macro candidates do not have a unique original focus ticker, so this benchmark assigns `AMD` as their target before annotation. This deterministic assignment makes every record target-relative while preserving peer-, sector-, macro-, irrelevant-, and ambiguous examples.

## Required output

```json
{
  "relevance": "",
  "event_scope": "",
  "event_type": "",
  "affected_breadth": "",
  "target_direction": "",
  "sector_direction": "",
  "peer_effect": "",
  "explicit_surprise": "",
  "information_status": "",
  "transmission_channels": [],
  "affected_companies": [],
  "affected_sectors": [],
  "evidence": {
    "scope": "",
    "direction": "",
    "surprise": ""
  },
  "abstain_reason": null
}
```

## Closed-label fields

### `relevance`

- `direct_target`: the target company is directly discussed or directly affected.
- `sector_or_peer`: a sector peer or the target sector is directly discussed.
- `macro_relevant`: a broad economic or market event is directly relevant to the target sector.
- `irrelevant`: no meaningful connection to the target or its sector.
- `insufficient`: the supplied text is too limited to determine relevance.

### `event_scope`

- `firm_specific`: the primary shock concerns the target company.
- `peer_specific`: the primary shock concerns one non-target sector peer.
- `sector_wide`: the event explicitly applies to several sector firms or the sector as a whole.
- `macro_market`: the event applies broadly across the economy or market.
- `mixed`: two or more scopes are materially present.
- `unclear`: scope cannot be determined.

Sector membership alone is not evidence of a sector-wide event.

### `event_type`

- `earnings`
- `guidance`
- `product_technology`
- `demand_customer_contract`
- `supply_chain_capacity`
- `regulation_trade_policy`
- `analyst_action`
- `corporate_action`: M&A, partnerships, financing, buybacks, dividends, management changes, or restructuring.
- `legal_governance_operations`
- `macro_market`
- `other`
- `unclear`

Choose the primary event. Use `other` only when the event is supported but falls outside the taxonomy; use `unclear` when the supplied text does not establish an event type.

### `affected_breadth`

- `single_firm`
- `several_same_sector`
- `cross_sector`
- `broad_market`
- `unclear`

### `target_direction` and `sector_direction`

- `positive`
- `negative`
- `neutral`
- `mixed`
- `unknown`
- `not_applicable`

Direction is the effect stated or clearly entailed by the text, not an annotator's market forecast. Use `unknown` when the relationship is relevant but its direction is unsupported; use `not_applicable` when the target or sector relationship is not relevant.

### `peer_effect`

- `same_direction`: the target and peers are stated to be affected similarly.
- `opposite_direction`: the target benefits at peers' expense, or vice versa.
- `mixed`: different peers are affected differently.
- `none_stated`: peers are mentioned, but no comparative effect is stated.
- `unknown`: the relationship is relevant but insufficiently described.
- `not_applicable`: a target-peer relationship is not relevant.

### `explicit_surprise`

- `positive`: explicit beat, upside surprise, stronger-than-expected result, unexpected improvement, or upward revision.
- `negative`: explicit miss, downside surprise, weaker-than-expected result, unexpected deterioration, or downward revision.
- `mixed`
- `none`: an event is described but no explicit comparison appears.
- `unknown`: the text is insufficient to tell whether a surprise comparison appears.

Do not infer a beat or miss unless the text compares the event with estimates, expectations, guidance, or a prior benchmark.

### `information_status`

- `confirmed`: completed, officially announced, filed, reported, or enacted.
- `scheduled_or_expected`: planned, forecast, expected, or upcoming.
- `rumor_or_unconfirmed`: reportedly under consideration, rumored, or unconfirmed.
- `analysis_or_opinion`: primarily commentary, interpretation, screening, or opinion.
- `unclear`

## Multi-label and open fields

`transmission_channels` contains no more than two of:

```text
demand
pricing_margin
supply_capacity
technology_product
competition
regulation_trade
rates_financing
macro_growth
geopolitical
capital_allocation
legal_operational
other
unclear
```

`affected_companies` and `affected_sectors` list only entities explicitly named or explicitly covered. Do not expand a company name into unmentioned subsidiaries, customers, competitors, or sector members.

Each evidence string must be a short, exact substring of the headline or summary. Empty evidence is permitted only when the corresponding fact is unsupported or not applicable. `abstain_reason` must explain an `insufficient` relevance label; otherwise it is `null` in version 0.

## Conservative annotation rules

1. Use only the supplied input.
2. Do not infer future market outcomes.
3. Do not infer sector scope from sector membership.
4. Do not infer surprise without an explicit benchmark.
5. Prefer `unknown`, `unclear`, `insufficient`, or `not_applicable` over unsupported certainty.
6. For `irrelevant` or `insufficient` records, use conservative labels for all downstream fields.
7. Treat GPT-5.6 Sol annotations as a silver reference, not unquestioned ground truth.

## FLAN-T5 comparison fields

The primary agreement test covers the nine closed-label fields in nine independent deterministic generations, one field per prompt:

1. `relevance`
2. `event_scope`
3. `affected_breadth`
4. `event_type`
5. `information_status`
6. `explicit_surprise`
7. `target_direction`
8. `sector_direction`
9. `peer_effect`

For the current `flan-stock-sector-news-v0.2.0` contract, greedy decoding is constrained to the legal labels for the field. This prevents format failures without changing the underlying model or silently repairing invalid prose after generation.

The local runner's optional `full` mode adds six secondary generations: transmission channels; companies; sectors; and separate exact-evidence spans for scope, direction, and surprise. Splitting them prevents a malformed entity or evidence output from invalidating unrelated fields. Open outputs remain subject to exact entity/source and evidence-substring validation, and they remain secondary because FLAN-T5-Large is materially less reliable at formatting and grounding free-form lists and copied spans.

## Evaluation language

Report `agreement with GPT-5.6 Sol reference annotations`, not `accuracy against ground truth`. Use macro-F1, Cohen's kappa, confusion matrices, multi-label F1/Jaccard, entity precision/recall/F1, evidence-substring validity, and overall output validity.
