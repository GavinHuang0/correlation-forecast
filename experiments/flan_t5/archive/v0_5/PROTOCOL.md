# FLAN-T5-Large v0.5 Final Candidate Protocol

Status: locked before any v0.5 model scoring.

This is the last planned experiment on the frozen `google/flan-t5-large`
checkpoint. It may be promoted as the final FLAN research extractor only under
the rule in `promotion_rule.json`. It is not a claim of production readiness.

## Fixed model and shared inputs

- Model revision: `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a`
- Runtime target: CUDA/FP16
- Coarse schema: `config/news_feature_schema_coarse.json`, version 0.2.0
- Development assignment: the existing fixed 72-record split
- Engineering evaluation assignment: the existing 228-record split
- Relevance gate and explicit-surprise rules: unchanged from v0.4
- Information-status label: unchanged v0.4 order-averaged prediction

The 228-record split has already informed diagnosis of v0.4. It is therefore an
engineering comparison, not a pristine scientific holdout. A promoted v0.5
still requires a new human-audited test set before any external accuracy claim.

## New frozen questions

All new questions are binary, article-first, and scored using the mean token log
probability of `A` and `B` in both canonical and reversed Yes/No order. Free-form
generation is not used.

### Shock scope

1. Does the primary news contain information unique to one company, beyond
   merely naming that company among several affected firms?
2. Does the primary event explicitly apply to several sector firms, the sector
   as a whole, or the broad economy or market?

Development-only thresholds turn the two log-odds into:

| Firm component | Common component | Label |
|---|---|---|
| yes | no | `idiosyncratic` |
| no | yes | `common` |
| yes | yes | `mixed` |
| no | no | `unclear` |

### Event family

Six separate one-vs-rest questions ask whether the article explicitly contains
each supported family:

- `earnings_guidance`
- `product_demand`
- `supply_capacity`
- `regulation_legal`
- `corporate_analyst`
- `macro_market`

Each class receives one threshold fitted only on the development split. If
several classes clear their thresholds, the largest calibrated margin selects
the single event-family label. If none clears its threshold, the output is
`other_or_unclear`.

### Directional alignment

Two questions ask whether the article explicitly states:

1. the target and sector or peers are affected in the same economic direction;
2. the target benefits at peers' expense, or vice versa.

An explicit relation is accepted only when both Yes/No option orders select
Yes and the order-averaged log-odds is positive. Otherwise alignment defaults
conservatively from predicted scope:

- `idiosyncratic` -> `single_firm_only`
- `common` or `mixed` -> `common_direction_unclear`
- `unclear` -> `unclear`

If both relation questions are positive, the conservative scope-derived default
is retained.

## Calibration and selection

- Scope thresholds are selected jointly on development macro-F1.
- Event thresholds are selected separately using one-vs-rest binary macro-F1.
- Threshold candidates are observed-score midpoints plus exterior sentinels.
- Ties use the deterministic rules recorded in the calibration artifact.
- Evaluation labels and priors are never read during calibration.
- The calibration artifact is written and hash-locked before materializing the
  evaluation predictions.
- No post-evaluation field substitution or threshold change is permitted.

## Promotion

Promotion requires every condition in `promotion_rule.json`. Failure leaves
v0.5 under the rejected-candidate archive and does not create a frozen pointer.
