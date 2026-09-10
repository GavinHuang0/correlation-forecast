# V4 cached-score redesign evidence audit

Status: **read-only diagnosis of frozen v3 and quant-v2 artifacts**.

This audit motivated v4 before any v4 model result was inspected. It did not
modify v1–v3 artifacts or rerun FLAN-T5-XL.

The subsequently locked construction and training are complete. Their results
are documented separately in [training/README.md](training/README.md); this
pre-training audit remains unchanged as design provenance.

## Frozen inputs

| Artifact | SHA-256 |
|---|---|
| `data/features/news_semantic/massive_v3/flan_w17_lite_k16/predictions.jsonl` | `9496cc382fa77c68edc05768af22ffd55a92e3721d6d4ddd2b846a41b09059e1` |
| `data/features/news_semantic/massive_v3/flan_w17_lite_k16/selected_assignments.parquet` | `e9b134ba99a187ae4ef93f684589da221b62a30a8a25b52ff834ec550cffc01a` |
| `config/quant_deterministic_news_protocol_v3.json` | `2455e3a1bbef5ac9b358e7f0aaa92ed209f51cee2a0936de4093e23799c53def` |
| `config/quant_training_protocol_v2.json` | `fa0ab9ca6992f480d7e8c000644cc95c12cfabbc1c4d98bed90e381e5ec39014` |
| `outputs/quant_training/v2/rung_03/predictions.parquet` | `1b4a6c60ed6d673b8b4a6e1fdf39e32a2272f07849a1b07960f28dd6aaa6e87b` |

## Inference information loss

| Diagnostic | Frozen result |
|---|---:|
| Completed selected articles | 50,488 |
| Canonical/reversed top-label agreements | 32,880 |
| Choice-order agreement | 65.124386% |
| Articles discarded by the hard agreement rule | 17,608 (34.88%) |
| Accepted `policy_corporate` labels | 189 |
| Selected assignment rows | 438,522 |
| Target-idiosyncratic assignment rows | 36,900 (8.41%) |
| Peer-idiosyncratic assignment rows | 184,500 |
| Common assignment rows | 217,122 |

The disagreements were structured rather than random. Canonical predictions
were heavily biased toward `other_or_unclear`, while reversed predictions were
heavily biased toward firm and macro classes. After normalizing the four
scores within each order and averaging orders, mean article score weights
were approximately 0.4221 firm, 0.0738 policy, 0.2708 macro, and 0.2333 other.
Mean normalized canonical/reversed Jensen-Shannon divergence was small enough
to show that many hard-label disagreements were close-boundary decisions.

## Daily-feature dilution

- V3 event shares were constant across all six sector stocks on roughly 76%
  to 99% of sector-dates.
- Status and cue fields were sector-date invariant on roughly 84% to 98% of
  groups; selection/text diagnostics on roughly 89% to 98%.
- Maximum absolute correlations with D2 reached about 0.75 for route fields
  and 0.65 for concentration, confirming substantial duplication.
- Only four of 17 primary W17 columns were model-derived. The remaining
  thirteen were deterministic routing/status or provider/selection/text state.

A read-only prototype of role-conditioned soft aggregation reduced
sector-date invariance of I/P event interactions to about 5%, while common
interactions remained appropriately sector-common.

## V3 outcome evidence

- `S2` Q56+W17 versus Q56 improved only T2 ETF, by 1.2015% incremental MSE
  $`R^2`$; the other three targets worsened.
- `S3` Q56+D2+W17 beyond Q56+D2 improved T2 ETF by only 0.2362%, with a paired
  interval crossing zero; three targets worsened.
- The T2 ETF `S2` gain was indistinguishable from the coverage/text-only
  control and was beaten by stale-20 and date-sector-permuted semantics.
- Mean foldwise selected-feature Jaccard ranged from 0.0588 to 0.4151. Most
  target/bundle pairs had zero or one W17 feature selected in every fold.
- No linear useful-semantic gate passed.

These results support a redesign, not a claim that cached soft scores are
known to forecast correlation.

## Long-history quant reuse

The quant-v2 modeling panel has 58,500 rows and 2,136 dates from 2017-12-28
through 2026-06-30. Its saved rung-3 XGBoost outer predictions overlap every
T1 semantic row (27,510) and 26,550 already-purged T2 semantic rows. Actual
Fisher-z values match bit-for-bit.

On the exact common 2025-H1–2026-H1 sample, the uniform long-history XGBoost
base had lower MSE than the short-history Q56 Elastic Net by about 2.02%,
4.86%, 7.14%, and 10.39% for T1 ETF, T1 LOO, T2 ETF, and T2 LOO. This makes it
a materially stronger and more honest hurdle for semantic correction.

The target-specific quant-v2 winners were selected after inspecting pooled
outer results, so v4 uses one fixed XGBoost family for all targets. Saved
forecasts are an expanding-fit out-of-sample stream, not a single model
checkpoint. They support cheap leakage-safe correction today; producing a new
future forecast will still require refitting the quant model.
