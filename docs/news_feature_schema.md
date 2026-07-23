# Article-Level News Feature Protocol

Semantic schema version: `0.1.0`

Current FLAN prompt/parser contract: `flan-stock-sector-news-v0.2.0`

This protocol measures semantic properties of a supplied financial-news headline and summary for stock-sector coupling research. It does not ask either language model to forecast returns, volatility, correlation, beta, or trading outcomes.

The machine-readable enum source of truth is [`config/news_feature_schema.json`](../config/news_feature_schema.json).

The FLAN accuracy-improvement experiment also defines a reduced schema in [`config/news_feature_schema_coarse.json`](../config/news_feature_schema_coarse.json). It deterministically coarsens the original silver labels into four LLM fields—shock scope, event family, information status, and directional alignment—while moving relevance and explicit surprise into an auditable non-LLM layer. The original schema remains the annotation source of truth; the coarse schema is the small-model evaluation contract.

## Inputs

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
