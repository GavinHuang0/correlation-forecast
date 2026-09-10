# Cached-score semantic redesign and training v4

Status: **feature construction, exploratory training, controls, and paired
evaluation complete; RRES-C6 is selected as the T2 ETF semantic research
overlay**.

The selection is recorded in
[`../../../models/active/registry.json`](../../../models/active/registry.json).
All other v4 forecast candidates and all RRES-C6 target routes except T2 ETF
are archived by status in
[`../../../models/archive/registry.json`](../../../models/archive/registry.json).

This experiment is separate from the completed [v3 W17-Lite
experiment](../v3/README.md). It does not alter v3's locked protocol, feature
panels, predictions, controls, or conclusions. V4 reuses the completed 50,488
article FLAN-T5-XL inference ledger so no language-model inference is repeated.

All v4 results are development evidence. The underlying ordinary-Massive text
is retrospective and not article-version safe, the dates have already been
inspected in earlier experiments, and the research peer map is not fully
effective-dated.

## Outcome

V4 successfully reused all 50,488 cached FLAN-T5-XL article inferences, built
role-conditioned soft features without another LLM run, and trained 20
separately recorded live/control bundles. The broad SoftRoute19 designs did
**not** establish a reliable incremental semantic-news signal. One compact,
target-specific result was selected: RRES-C6 for T2 ETF.

- `J1` Q56+SoftRoute19 improved T2 ETF MSE by 0.7110% versus matched Q56,
  but its 95% interval crossed zero and it was 1.3219% worse than its stale-20
  control. It significantly worsened T2 LOO by 1.3561%.
- `J2` Q56+D2+SoftRoute19 improved T2 ETF MSE by only 0.3003% versus matched
  Q56+D2, with an interval crossing zero, and significantly worsened T2 LOO
  by 1.2908%.
- The current-mass-only `J3` diagnostic improved T2 ETF by 0.6195%, with a
  paired 95% interval of [0.0013%, 1.3428%] and wins in two of three folds.
  It has no architecture-matched control set and is therefore a screening
  result, not a passed semantic claim.
- The compact long-Q residual `RRES-C6` improved T2 ETF by 1.3844% versus the
  unchanged long-history Q anchor, with a paired 95% interval of
  [0.3429%, 2.6228%] and wins in three of five folds. Its point-loss,
  confidence-interval, and fold-count tests passed, so the repository selects
  it as the T2 ETF semantic research overlay. It worsened two other targets
  and was not assigned its own complete post-selection control ladder;
  promotion therefore does not constitute prospective confirmation.
- The full `RRES-L19` residual correction improved long-Q T2 ETF by 0.9342%,
  but its interval crossed zero; the other three target gains were at most
  0.0935%, and T2 LOO worsened. The direct `RSTACK` model worsened all four
  targets, including T2 LOO by 6.4956%.
- No `J1`, `J2`, or `RRES-L19` target passed the complete matched-base, stale,
  wrong-stock, probability-permutation, quality-only, confidence-interval,
  and fold-count gate.

The canonical interpretation and compact result tables are in the
[v4 training report](training/README.md). The complete 92-row paired table is
preserved in the generated [comparison report](training/comparisons/final/RESULTS.md).

## What remains weak after the redesign

Soft aggregation fixed the largest v3 information-loss bug, but it cannot add
semantic distinctions that FLAN never extracted:

- The consensus score maps remain diffuse: mean normalized entropy is 0.8293,
  and canonical/reversed hard choices agreed on only 65.12% of articles. The
  softmax values are relative candidate scores, not calibrated probabilities.
- The joint-mass construction still combines event-class weight with article
  routing/volume. Common macro mass correlates 0.8143 with D2's common-article
  share. The maximum absolute SoftRoute19/D2 correlation is 0.8143.
- Target-idiosyncratic event mass averages only 0.0598, compared with 0.2992
  for peer-idiosyncratic and 0.4125 for common mass. All three target-specific
  current event masses are exactly zero on 46.99% of stock-days.
- Current masses and their innovations have correlations from 0.8077 to
  0.9612. Seven principal components explain 90.20% of the standardized
  18-field current/innovation variance, so 19 columns do not represent 19
  independent semantic dimensions.
- Role conditioning materially improved target specificity: target and peer
  fields are sector-date invariant on only about 6.27% of groups. Common
  fields remain invariant on about 62.16%, as expected for common news. The
  remaining failure is therefore no longer merely the old all-sector copying
  bug.
- No SoftRoute19 coefficient sign was stable across all five long-Q folds for
  any target. Validation-selected residual shrinkage also varied sharply;
  `RRES-L19` selected zero correction in three fits and full correction in
  five of 20 fits.
- The ontology still has no target/sector direction, surprise magnitude,
  materiality, novelty, causal link, or explicit correlation-transmission
  label. A coarse event family can map to either stronger or weaker
  stock-sector comovement.

These diagnostics explain why the larger semantic designs failed even though
the compact C6 projection produced a statistically supported, narrowly scoped
T2 ETF gain.

The 2017 history belongs to the Q anchor, not to L: the first residual
correction has 166 independent training dates from the post-inference era.
That is the leakage-safe way to exploit a long-history Q model, but it leaves
the semantic correction itself data-limited.

## Completed artifacts

- Protocol SHA-256:
  `361c235e46bd9322525534da20df98a2f72820670a027739465f620134bc63fe`.
- Four daily semantic panels: 27,510 rows and 45 columns each; daily manifest
  SHA-256 `2e97ea196a0d304df08787fc26cd48cf027db7a2a3cbf4b290dbdec9e68c23c1`.
- Four Q+D2+L joined panels: 27,510 rows and 186 columns each; joined manifest
  SHA-256 `de282591555d3fcf6d0ea74c139bc0f64573d2b05d64d743562042f1cb04e3f5`.
- 11 short-history and nine long-Q/control bundles, each with predictions,
  validation predictions, fits, reviews, dependencies, and manifests.
- Final comparison manifest SHA-256:
  `fa74ed173b9fb60bbe0c399bffaf63f4ca8ef8839a057c0d2f2814fd7dcaed9f`.

## Why W17-Lite barely improved Q-only

V3 established that the weak semantic block was mechanically usable, but it
did not isolate useful timely semantics:

- Only four of 17 predictors were FLAN-derived: three hard event shares and
  their entropy. The rest were routing rules, status dictionaries, or
  provider/selection/text diagnostics.
- Canonical and reversed option orders agreed for only 32,880 of 50,488
  articles (65.124386%). The hard acceptance rule discarded 17,608 articles,
  including many close firm-versus-macro and unclear-versus-event decisions.
- The accepted ontology nearly deleted `policy_corporate`: only 189 articles
  survived with that label. The cached score maps retain materially more soft
  policy evidence.
- Event labels were pooled before target-relative routing. Only 8.41% of the
  438,522 selected assignment rows were target-idiosyncratic; most semantic
  and status fields were therefore identical across all six stocks on 75% to
  99% of sector-dates.
- Many W17 fields duplicated D2/provider state. They encoded persistent
  coverage, text length, and common article flow more strongly than a new
  target-relative event signal.
- The apparent T2 ETF improvement failed every useful-semantic gate: stale
  event labels and date-sector-permuted labels performed better, the
  wrong-stock comparison was inconclusive, and coefficients were unstable.
- The four-class extractor contains no direction, surprise magnitude,
  materiality, novelty, causal link, or target-versus-sector alignment. Event
  family alone is heterogeneous with respect to realized correlation.
- There are only 917 market dates, the 30 same-day rows are dependent, T2
  outcomes overlap, and Q already explains much of the persistent state.

The hash-bound numerical evidence is preserved in [DESIGN_AUDIT.md](DESIGN_AUDIT.md).

## Cached article-score contract

For each article and each prompt order, normalize the four cached candidate
mean log scores with a within-order softmax. The v4 consensus score weight is
the arithmetic mean of the canonical and reversed distributions:

```math
p_{a,k}=\frac{1}{2}\left[
\mathrm{softmax}(s^{can}_a)_k+
\mathrm{softmax}(s^{rev}_a)_k\right].
```

This transform is invariant to which option order is called canonical. It
uses every completed article, including top-label disagreements. The values
are relative candidate-score weights, not calibrated probabilities.
Temperature tuning, confidence thresholding, or choosing one prompt order
from forecast outcomes is prohibited.

## SoftRoute19

For stock-day $`(i,t)`$, let selected assignments have deterministic,
mutually exclusive roles $`r\in\{I,P,C\}`$: target-idiosyncratic,
peer-idiosyncratic, and common. Let $`w_{iat}`$ be the already frozen
recency/duplication weight and let

```math
W_{it}=\sum_a w_{iat}.
```

For event classes `firm_operating_financial`, `policy_corporate`, and
`macro_market`, the nine current joint masses are

```math
M_{it,r,k}=\frac{\sum_a w_{iat}\mathbf 1[r_{iat}=r]p_{a,k}}
{W_{it}}.
```

The denominator includes every selected role and all four score classes, so
the nine masses are true event×route composition measurements and sum to at
most one. `other_or_unclear` remains in the denominator and audit output but
is not a primary predictor. On a source-complete stock-day with no selected
assignment, all nine current masses are zero and the explicit no-selected
flag is one. Incomplete query scope is missing/ineligible, never zero.

For each current mass, define a prior-only innovation relative to the previous
63 official stock sessions, using a 21-session half-life:

```math
B_{it,r,k}=\frac{\sum_{h=1}^{63}2^{-(h-1)/21}M_{i,t-h,r,k}}
{\sum_{h=1}^{63}2^{-(h-1)/21}},\qquad
\Delta M_{it,r,k}=M_{it,r,k}-B_{it,r,k}.
```

All 63 prior complete sessions are required. The current session never enters
its own baseline. The primary block therefore contains exactly:

- nine current role×event masses;
- nine prior-only innovations; and
- `lsoft_observed_no_selected_article`.

No status regex, text-length flag, publisher/source identity, raw article
count, selection coverage, entropy, order disagreement, or concentration
measure enters SoftRoute19. Those remain audit variables or matched nuisance
controls.

## Coupling6 residual projection

The low-dimensional residual model uses six predeclared linear projections,
not SoftRoute19 and Coupling6 together. For each of the three event classes:

```math
K_{it,k}=M_{it,C,k}-M_{it,I,k}-M_{it,P,k},\qquad
\Delta K_{it,k}=\Delta M_{it,C,k}-\Delta M_{it,I,k}-\Delta M_{it,P,k}.
```

This encodes common semantic pressure versus target/peer-idiosyncratic
pressure. It is intentionally compact because the long-Q corrector has far
fewer independent dates than the quant base.

## Audit and controls

Four score/provider diagnostics are excluded from the primary block:

- weighted canonical/reversed Jensen-Shannon divergence, normalized by
  $`\log 2`$;
- weighted consensus entropy, normalized by $`\log 4`$;
- weighted `other_or_unclear` score mass; and
- selected/full assignment-weight coverage.

The minimum falsification set is:

- `L20`: shift all 18 current/innovation semantic fields 20 stock sessions;
  keep the current no-selected and audit state;
- `WS`: use a fixed alphabetically-next stock donor within sector for the I/P
  fields while retaining genuinely common C fields;
- `PERM`: deterministically permute each unique article's complete four-class
  consensus score vector within forecast-date and sector before reaggregation,
  preserving recipient weights, routes, and class dependence; and
- `QUALITY`: the four diagnostics plus no-selected, without event identity.

The stale and permutation controls are decisive because both stale and
permuted W17 beat live W17 in v3. All controls share the exact folds,
estimator family, and preprocessing of their live counterpart.

The implemented short controls are `C-J1-L20`, `C-J2-L20`, `C-J1-WS`,
`C-J2-WS`, `C-J1-PERM`, `C-J2-PERM`, `C-J1-QUALITY`, and
`C-J2-QUALITY`. The quality-only controls use exactly the four diagnostics
above plus `lsoft_observed_no_selected_article`; they contain no event-class
or route×event feature. The permuted controls read a separately materialized,
hash-bound panel rather than permuting data inside a model-fitting process.

## Training ladder

### Short-history joint models

The three chronological folds remain exactly those used in v3: test blocks
2025-H1, 2025-H2, and 2026-H1, each preceded by an immediately prior
validation half-year. Training begins 2022-11-01 and expands.

| Rung | Model | Inputs |
|---|---|---|
| `J0` | Bound historical base | Existing v3 Q56 Elastic Net predictions |
| `J0D` | Bound historical base | Existing v3 Q56+D2 Elastic Net predictions |
| `J1` | Elastic Net | Q56 + SoftRoute19 |
| `J2` | Elastic Net | Q56 + D2 + SoftRoute19 |
| `J3` | Elastic Net diagnostic | Q56 + nine current role×event masses |
| `C-J1-QUALITY` | Elastic Net nuisance control | Q56 + Quality4 + no-selected |
| `C-J2-QUALITY` | Elastic Net nuisance control | Q56 + D2 + Quality4 + no-selected |
| `C-J1-PERM` | Elastic Net falsification | Q56 + permuted SoftRoute19 |
| `C-J2-PERM` | Elastic Net falsification | Q56 + D2 + permuted SoftRoute19 |

`J1` is compared with `J0`; `J2` with `J0D`. Reusing the hash-verified v3
bases prevents a needless refit from changing the comparator. Hyperparameters
are chosen on the validation block only, preprocessing is fit on train only
for selection and train+validation only for the final test fit, and every
stock on a date stays in one block.

### Long-history quant anchor and residual correction

The long base is the saved `xgboost` outer-test forecast stream from quant-v2
rung 3. Each forecast comes from an expanding quant fit whose estimation
history starts 2017-12-28; 2016 prices supply feature warm-up. It is not one
serialized frozen booster, and v4 does not claim otherwise.

V4 never fills 2017–2022 semantic features with zero. It uses only saved
out-of-sample quant forecasts and forms

```math
e_{it}=z_{it}-\widehat z^{Q,long}_{it}.
```

Five prequential evaluations use quant-v2 folds 09–13 (2024-H1 through
2026-H1). For each test fold, all earlier news-era out-of-sample residual
blocks through $`j-2`$ are training data and fold $`j-1`$ is validation. The
final corrector is refit on training plus validation. T2 consumes only the
already purged long-Q rows.

| Rung | Forecast construction |
|---|---|
| `R0` | unchanged long-Q XGBoost forecast |
| `RCAL` | base-only OLS calibration control |
| `RRES-C6` | long-Q + Elastic-Net correction from Coupling6 |
| `RRES-L19` | long-Q + Elastic-Net correction from SoftRoute19 |
| `RSTACK` | Elastic Net on long-Q forecast state + SoftRoute19 |
| `C-RRES-L19-QUALITY` | long-Q + residual correction from Quality4 + no-selected |
| `C-RRES-L19-PERM` | long-Q + residual correction from permuted SoftRoute19 |

For residual rungs, validation also chooses correction shrinkage from
`{0, 0.25, 0.5, 0.75, 1}`. A shrinkage of zero is a legitimate decision to
retain the base forecast. No in-sample quant residual is ever used.
All Elastic-Net rungs exhaustively evaluate exactly the locked 7×4
alpha/L1-ratio grid. No architecture may adaptively expand that grid at a
boundary.

## Evaluation and claim gate

Loss is squared error in Fisher-z space. Report RMSE, MAE, correlation-scale
metrics, per-fold results, incremental MSE $`R^2`$, and paired moving-block
whole-date bootstrap intervals using ten-session blocks. Blocks are sampled
separately inside each outer fold and then pooled, so no bootstrap block can
cross a train/test regime boundary. The live semantic model is not called
useful unless it beats its matched base and every predeclared stale,
wrong-stock, permutation, and quality-only control with a negative point loss
delta, an upper 95% loss-delta bound below zero, and improvement in at least
two of three short folds or three of five long-anchor folds. These conditions
are computed and saved by the comparison runner, not assessed manually.

`RSTACK` is compared with `R0` and `RCAL`, but it has no
architecture-matched falsification controls. Its automated record therefore
marks the full useful-semantic gate as not evaluable and cannot report a pass.

Even a passed v4 gate would be exploratory. Confirmation requires a protocol
frozen before outcomes and a prospectively versioned news period strictly
after the already inspected data.
