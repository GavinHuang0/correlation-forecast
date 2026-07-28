# V2 feature-redesign evidence audit

Status: **documentation-only read-only audit**.

This note binds the empirical diagnostics used to motivate the proposed
[v2 design](README.md). It did not construct features, fit a model, change a
v1 artifact, or authorize a v2 experiment.

## Frozen sources

| Source | SHA-256 |
|---|---|
| `data/features/q_plus_d/massive_v1/modeling_panel_q_plus_d.parquet` | `352ba2635febfa0bc36bc6f797c00036a8aee9879327a8176e9accf180e50150` |
| `data/features/q_plus_d/massive_v1/manifest.json` | `ac161d48901fae669fdc500099df92d218513d2e7d8d383539b8ed326445d86f` |
| `data/features/news_deterministic/massive_v1/manifest.json` | `01f3d16df1ed7caf9ba582ad489b366e17aed89df1689e626ae22a0e32fa1a52` |
| [V1 final report](../v1/comparisons/final/RESULTS.md) | `962765fbbda151b01384c3b5281775569da4220783999cc5bf2645e53f8d3bfe` |
| `outputs/quant_deterministic_news/v1/comparisons/final/paired_comparisons.parquet` | `a22337a0bfd7928fd4cfaceb7989b8771aed7948665da0048ba6fba58834dcd8` |
| [FLAN-T5-XL v1.1 evaluation](../../flan_t5_xl/v1_1/evaluation_summary.json) | `6387c7a8b23feabede33214d17a46908be6ab4b54cd75ebae4b6158d175734da` |

The panel contains 27,510 stock-date rows. The 40 deterministic fitting
columns are the ordered union of `d_activity_scope_18`, `d_cues_8`,
`d_timing_6`, `d_source_title_7`, and `d_burst_1` in the frozen v1 protocol.

## Semantic-readiness sources

| Source | SHA-256 |
|---|---|
| [`experiments/active_extractor.json`](../../active_extractor.json) | `101d33c2b99ab0f8680cd23c299c2786a6356077a45190a740fe247319f8509d` |
| [`scripts/run_flan_t5_xl_active.py`](../../../scripts/run_flan_t5_xl_active.py) | `3f26adc188df884c19dd42a41544a518315e476b672beb0a8a592cfd98788d65` |
| [`scripts/extract_flan_t5.py`](../../../scripts/extract_flan_t5.py) | `92927cc438540275d88ae4bc102e6983a2c16e2436e5ce6edc7cf6d51d9ccd69` |
| [`config/news_feature_schema.json`](../../../config/news_feature_schema.json) | `69798bd423f1775c640b2168bb96b2b1fe8cf2b2aab72c4e701416e8ee2597a7` |
| [`config/news_feature_schema_coarse.json`](../../../config/news_feature_schema_coarse.json) | `201e19a0f2b9723e690c3a62eb145213b7e6d8c106a09e0da7374b71830d05fa` |
| [`annotations/chatgpt_5_6_sol_reference.jsonl`](../../../annotations/chatgpt_5_6_sol_reference.jsonl) | `94ae2481611fc2a88ce3ff2a16c6c2945c4083be2ef787810c1d1bb1f50967b7` |

The readiness audit found:

- 57,147 unique Massive documents in 2022-11-01 through 2026-06-30,
  including 56,232 with descriptions (98.40%);
- a cursor-complete collection only for the configured 30 stocks, five sector
  benchmarks, and SPY—not a complete untickered macro feed;
- a static research peer universe, which must be replaced or explicitly
  labeled before effective-dated routing claims;
- no reusable v2 article-target panel, because v1's 880,466 assignments use
  the superseded routing rules;
- an active FLAN-T5-XL pipeline that emits the three WLLM17 fields but no
  full-corpus v2 predictions or daily aggregation;
- a generic fine FLAN runner but no pinned, evaluated fine-schema XL pipeline;
  the completed fine run used FLAN-T5-Large and achieved 0.177 mean macro-F1
  across nine fields; and
- exactly 300 existing fine GPT annotations covering AMD, AVGO, INTC, MU, and
  NVDA, not the 30-stock historical corpus.

The current-window length audit also found 155 headline-plus-description
inputs above 400 whitespace-delimited words and some above 2,000. Therefore
the FLAN 512-token limit requires a full tokenizer preflight and a frozen
common text-view rule; it cannot be handled by silent truncation.

These findings establish that WLLM17/RLLM70 mathematics are complete while all
four model-qualified daily panels remain unbuilt. The detailed proposed
construction and fitting gates are in
[`TRAINING_LADDER.md`](TRAINING_LADDER.md).

## Recomputed panel diagnostics

| Diagnostic | Result | Reproduction rule |
|---|---:|---|
| Mean `observed_relevant_article_count`, 2022 | 27.4294 | Extract calendar year from `forecast_date`; take the stock-day mean |
| Mean `observed_relevant_article_count`, 2023 | 30.7243 | Same |
| Mean `observed_relevant_article_count`, 2024 | 20.5331 | Same |
| Mean `observed_relevant_article_count`, 2025 | 9.9913 | Same |
| Mean `observed_relevant_article_count`, 2026 | 15.8195 | Same |
| Lowest ticker mean direct-target count | UNP, 0.4351 | Group `observed_direct_target_article_count` by `stock` |
| Highest ticker mean direct-target count | NVDA, 21.3839 | Same |
| Maximum/minimum direct-target ratio | 49.1454 | Divide the preceding two means |
| Rows with no direct-target article | 9,057 / 27,510 = 32.9226% | Count `observed_direct_target_article_count == 0` |
| Rows with no broadly relevant article | 84 | Count `observed_relevant_article_count == 0` |
| Provider articles | 90,290 | Frozen deterministic manifest |
| Article-target assignments | 880,466 | Frozen deterministic manifest |
| Assignments per provider article | 9.7515 | `880466 / 90290` |
| D40 fields sector-date invariant in at least 90% of groups | 26 / 40 | For each D40 field, group by `forecast_date, sector`; count groups with `nunique(dropna=False) == 1` |
| Relevant count/title-cluster count Pearson correlation | 0.999901 | Pairwise complete rows |
| Macro count/macro-cue count Pearson correlation | 0.998073 | Pairwise complete rows |
| Unique-peer count/peer-coverage Pearson correlation | 1.000000 within floating precision | Pairwise complete rows |

These measurements describe the retrospective provider response, not the
latent amount of public news. Invariance is not automatically a defect for a
true common shock; the concern is that broad assignment and coverage variables
made many nominal stock rows repeat the same provider state.

## Model-result evidence

The completed v1 report and its hash-bound paired-comparison table establish:

- A5 Q56+D43 Elastic Net reduced T2 ETF mean squared error by 3.65% relative
  to matched Q-only Elastic Net.
- The 20-session-stale A5 control had a lower point-estimate loss than
  contemporaneous A5 on all four targets.
- Correct-stock A5 beat the fixed wrong-stock control on all four targets.
- Neither placebo-tested linear specification passed the conservative final
  useful-news gate.

The interpretation used by v2 is deliberately narrow: D43 contained some
target/attention information, but the completed experiment did not isolate a
stable, timely deterministic-news effect.

## Extractor evidence

The frozen FLAN-T5-XL v1.1 evaluation used 228 documents and
deterministically coarsened GPT-5.6 Sol silver labels. It reported:

| Field | Macro-F1 | Frozen threshold | Passed |
|---|---:|---:|---|
| Shock scope | 0.4684 | 0.70 | No |
| Event family | 0.4776 | 0.65 | No |
| Information status | 0.5797 | 0.70 | No |
| Directional alignment | 0.3284 | 0.65 | No |

Mean field macro-F1 was 0.4635. The deterministic relevance gate had F1
0.9351. These are agreement scores against a silver reference, not accuracy
against human ground truth. They motivate WLLM17's deterministic relevance,
direction exclusion, field/class abstention, and default
`training_eligible = false`.
