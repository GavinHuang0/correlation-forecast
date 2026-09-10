# Cost-bounded semantic news design v3

Status: **the FLAN K-16 W17-Lite path is complete through inference, daily
feature construction, matched training, controls, and final comparison**.
Paid GPT requests and the G40/R70-Lite branches have not run.

This is a separate successor design. It does not overwrite the complete
deterministic results, the unexecuted full W17/R70 contracts, or any artifact
under [v2](../v2/README.md). The v2 pipelines remain reproducible references,
but their full workloads are no longer the recommended execution path:

- cached full FLAN W17 still requires about 577,296 field classifications; and
- exact GPT R70 requires 933,804 two-view target-article annotations.

The machine-readable companion is
[`config/news_semantic_lite_design_v1.json`](../../../config/news_semantic_lite_design_v1.json).
Throughout this document, **W17-Lite** means exactly the arm
`WL17__flan_t5_xl`; it is not a second feature contract.

## K-16 execution readiness

The executable K-16 path is now
[`scripts/semantic_news_v3/flan_w17_lite.py`](../../../scripts/semantic_news_v3/flan_w17_lite.py),
with the one-command Windows wrapper
[`run_flan_w17_lite_k16.ps1`](../../../scripts/semantic_news_v3/run_flan_w17_lite_k16.ps1).
The materialized selection contains exactly 50,488 unique articles and
438,522 selected assignment rows, retaining 97.3261% of assignment weight.
Its sorted article-ID SHA-256 is
`c676de4ada043e66d44497fdb693daebc04467303e689208cb2e0c802f568da2`.
The tokenizer preflight checked all 100,976 canonical/reversed prompts with
zero violations; the maximum observed prompt was 431 tokens under the
512-token cap. That preflight did not load FLAN. The subsequent resumable
inference completed all 50,488 articles with zero failures or truncations.
Canonical/reversed labels agreed for 32,880 articles (65.124386%), below the
frozen 85% semantic-stability gate. The resulting daily canonical,
long-description, and date-sector-permuted panels each cover all 27,510
stock-days and are joined to the matched Q56+D2 panel.

From the repository root, reproduce or verify the resumable run with:

```powershell
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\semantic_news_v3\run_flan_w17_lite_k16.ps1
```

The process-local execution-policy flag is necessary on the current Windows
installation and does not change the machine-wide policy. If the process is
stopped, rerun the exact same command. The prediction ledger is append-only
and uncompressed; every completed article is flushed and `fsync`-committed
before the next one starts. On restart, the runner:

1. verifies the source selection, schema, model revision, tokenizer preflight,
   runtime, and implementation hashes;
2. rejects a concurrent live writer and automatically recovers a dead
   same-host lock;
3. repairs only an incomplete final JSONL line, if one exists;
4. validates every retained terminal record; and
5. loads FLAN only when selected article IDs remain.

A bounded end-to-end smoke/resume check can be run first by adding
`-MaxNewArticles 25`. The unrestricted command then continues from article 26
under the same immutable run identity. Inspect progress without starting
inference with:

```powershell
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe -NoProfile -ExecutionPolicy Bypass -File .\scripts\semantic_news_v3\run_flan_w17_lite_k16.ps1 -StatusOnly
```

Canonical artifacts live under
`data/features/news_semantic/massive_v3/flan_w17_lite_k16`. Partial inference
is never eligible for daily aggregation or training.

## Decision

The recommended FLAN arm is `WL17__flan_t5_xl`, a new 17-column contract that:

1. classifies only one target-invariant four-way event group per unique
   article;
2. derives target-relative scope from the existing deterministic C/I/P routing;
3. replaces model-generated information status with conservative,
   independently firing deterministic cues;
4. preserves text-length, selection, abstention, and option-order diagnostics;
5. uses a role-balanced quota that preserves the 917-date calendar and every
   candidate-bearing stock-day under the current runtime evidence; and
6. permits all 55,197 unique articles only if a fresh pilot projects at most
   13.5 hours end to end.

The recommended GPT path does not begin with exact v2 R70. It starts with the
separately named 40-column `G40__gpt_5_6_sol` global-semantic block:

- infer article-global semantics once per unique article;
- use one primary view and a stratified second-view audit rather than two
  views for every target assignment; and
- pack several articles into one structured request after a paid pilot
  validates pack-level reliability.

A later factorized `R70-Lite__gpt_5_6_sol` design may add universally grounded
entity relations and deterministic target projection, but its schema and
feature count remain to be frozen. Exact v2 R70 remains the target-relative
oracle contract and must not be silently relabeled as either Lite result.

## Evidence that changes the design

The
[length-stratified Massive experiment](../../flan_t5_xl/length_stratified_massive_v1/README.md)
contained 200 unique articles and 360 model views, including paired
headline-only ablations for 160 articles. All views were resolved without
truncation, but there were no independent labels. It measures operational
stability and text sensitivity, not accuracy.

Important findings are:

| Finding | Design consequence |
|---|---|
| Event-family calibrated agreement was 85.0% for sub-150 native/headline pairs and 86.25% across all paired articles | Retain a reduced FLAN event field |
| Shock-scope option-order agreement was only 50% in both low-text strata | Derive scope from deterministic C/I/P routing |
| Raw information status abstained on 65% of headlines and 72.5% of sub-150 descriptions, while calibration removed every abstention | Do not use calibrated low-text status as a primary FLAN feature |
| Only 41.875% of all paired articles preserved all four calibrated labels | Do not treat headline-only inference as equivalent to description-backed inference |
| All 40 native headline-only records were from Investing.com and 38/40 records in the 600+ stratum were from Benzinga | Cross-length differences are source-confounded |

The separate 228-record
[GPT-silver evaluation](../../flan_t5_xl/v1_1/evaluation_summary.json)
reported macro-F1 of 0.468 for scope, 0.478 for event family, 0.580 for
information status, and 0.328 for directional alignment. These are not human
ground truth, and every semantic field missed its historical threshold.

Consequently, W17-Lite is an exploratory weak-label block. Candidate-score
margins are not correctness probabilities and are not used as quality
weights.

## Workload profile

The frozen v2 semantic corpus contains:

| Quantity | Count |
|---|---:|
| Article-target assignments | 466,902 |
| Assigned unique articles | 55,197 |
| Forecast dates | 917 |
| Stocks | 30 |
| Stock-days | 27,510 |
| Stock-days with no routed candidate | 204 |
| Headline-only unique articles | 848 |
| Description length 1-149 characters | 21,776 |
| Description length at least 150 characters | 32,573 |

The mean article is assigned to 8.46 targets. Repeating target-invariant event
and status inference for every assignment is therefore unnecessary. The exact
v2 FLAN cache already avoids that duplication for event/status, but its
target-conditioned scope field still runs once per assignment.

## W17-Lite article contract

### Text

FLAN receives the existing bounded, hashed `model_text`:

```text
trimmed headline

leading description excerpt, when present
```

The description excerpt remains capped at 512 Unicode code points and every
actual prompt must pass the pinned 512-token FLAN tokenizer check. Descriptions
are not discarded merely to make the corpus uniformly headline-only. The
paired experiment shows that descriptions change labels; it does not show
that removing them improves accuracy.

### One four-way FLAN field

The proposed target-invariant field is:

```text
event_group =
  firm_operating_financial
  policy_corporate
  macro_market
  other_or_unclear
```

The mapping from the old seven-way event-family ontology is:

| W17-Lite group | Existing concepts |
|---|---|
| `firm_operating_financial` | earnings/guidance, product/demand, supply/capacity |
| `policy_corporate` | regulation/legal, corporate/analyst |
| `macro_market` | macro/market |
| `other_or_unclear` | other or insufficiently clear; treated as abstention |

The scorer retains canonical and reversed choice order. A primary event label
is accepted only when:

- both orderings are valid and untruncated;
- both orderings select the same semantic group; and
- the agreed group is not `other_or_unclear`.

No v1.1 calibration is inherited because the label space and prompt are new.
The raw order-averaged scores remain audit data, not empirical probabilities.

### Deterministic target-relative scope

For article $`a`$, stock $`i`$, and forecast date $`t`$, reuse the already
frozen, mutually exclusive pre-cutoff assignment role
$`R_{ait}\in\{I,P,C\}`$:

- $`I`$: direct target-idiosyncratic;
- $`P`$: single-peer idiosyncratic; and
- $`C`$: sector- or macro-common.

The predictive matrix includes the weighted $`I`$ and $`P`$ shares; $`C`$ is
the reference category. This avoids both a dead "mixed" feature and the exact
dummy sum created by including all three shares with an intercept. These are
deterministic routing roles, not LLM claims about economic scope.

### Deterministic information-status cues

Information status is not forced into a single label. Four
high-specificity-intent cue families fire independently on the same bounded
text:

- confirmed action: explicit announcement, filing, approval, completion,
  signing, launch, declaration, or reported financial result;
- scheduled/expected: explicit schedule, earnings date, "will report",
  "set to", or "due to report";
- rumor/unconfirmed: explicit rumor, unconfirmed report, "people familiar",
  sourced report, talks, consideration, or exploration;
- analysis/opinion: explicit analyst rating, upgrade/downgrade, price target,
  initiated coverage, or identified opinion/analysis.

No precision claim is made yet. The eventual builder must freeze
word-boundary patterns and a normalization hash, then audit labeled precision,
conflict rate, and all-zero coverage before construction. Cues are multi-hot:
conflicting cues may coexist, and all-zero means "not detected by the frozen
rules," not confirmed absence. Ambiguous modal words such as `may` or `could`
cannot fire alone.

### Aggregation

Let $`\mathcal A_{it}`$ be all routed candidate assignments for stock $`i`$ at
date $`t`$, and let $`\mathcal S_{it}\subseteq\mathcal A_{it}`$ be assignments
whose unique article was selected for inference. Preserve the existing
12-hour half-life and duplication adjustment:

```math
w_{ait}
=
\frac{\exp\{-\log(2)\,\mathrm{age}_{ait}/12\}}
     {\mathrm{duplication\_group\_size}_a}.
```

For predictive event group $`c`$, let $`A_{a,c}=1`$ when both prompt orderings
are valid, untruncated, agree, and select $`c`$. Let $`U_a=1`$ when both
orderings agree on `other_or_unclear`, and let $`D_a=1`$ when both are valid
but disagree. Any invalid or truncated selected article fails the panel
rather than becoming a zero. Therefore
$`\sum_c A_{a,c}+U_a+D_a=1`$.

The event shares are:

```math
\mathrm{eventShare}_{it,c}
=
\frac{
  \sum_{a\in\mathcal S_{it}} w_{ait}
    A_{a,c}
}{
  \sum_{a\in\mathcal S_{it}} w_{ait}
}.
```

Routing and rule-cue shares use the same selected-weight denominator. The
selector's retained mass is:

```math
\mathrm{selectionCoverage}_{it}
=
\frac{\sum_{a\in\mathcal S_{it}} w_{ait}}
     {\sum_{a\in\mathcal A_{it}} w_{ait}}.
```

Omitted articles are therefore not treated as semantic zeros.

All share and concentration fields are missing when selected weight is zero;
`wlite_observed_no_selected_article` is then one. Otherwise that indicator is
zero. Training-only imputation must preserve this missingness distinction.
The two text-length shares are disjoint: headline-only means no description,
while sub-150 means a nonempty description shorter than 150 characters;
description length at least 150 is the reference category.

## Ordered W17-Lite features

The contract contains exactly 17 columns:

| Block | Feature |
|---|---|
| FLAN event | `wlite_event_share_firm_operating_financial` |
| FLAN event | `wlite_event_share_policy_corporate` |
| FLAN event | `wlite_event_share_macro_market` |
| Deterministic scope | `wlite_route_share_target_idiosyncratic` |
| Deterministic scope | `wlite_route_share_peer_idiosyncratic` |
| Deterministic status cue | `wlite_rule_status_cue_share_confirmed_action` |
| Deterministic status cue | `wlite_rule_status_cue_share_scheduled_expected` |
| Deterministic status cue | `wlite_rule_status_cue_share_rumor_unconfirmed` |
| Deterministic status cue | `wlite_rule_status_cue_share_analysis_opinion` |
| Deterministic status cue | `wlite_rule_status_cue_conflict_weight_share` |
| Selection | `wlite_selection_weight_coverage` |
| Text quality | `wlite_selected_headline_only_weight_share` |
| Text quality | `wlite_selected_sub150_weight_share` |
| Model stability | `wlite_event_order_disagreement_weight_share` |
| Availability | `wlite_observed_no_selected_article` |
| Semantic mixture | `wlite_event_entropy_accepted` |
| Article concentration | `wlite_selected_weight_hhi` |

Let $`m_{it}=\sum_c\mathrm{eventShare}_{it,c}`$ and, when $`m_{it}>0`$,
$`p_{it,c}=\mathrm{eventShare}_{it,c}/m_{it}`$. Then

```math
\mathrm{eventEntropy}_{it}
=-\frac{\sum_c p_{it,c}\log p_{it,c}}{\log 3}.
```

It is zero when selected weight is positive but accepted event mass is zero,
or when only one accepted group has positive mass; it is missing when selected
weight is zero. `wlite_selected_weight_hhi` is
$`\sum_a (w_{ait}/\sum_b w_{bit})^2`$ on selected assignments.
`wlite_rule_status_cue_conflict_weight_share` is:

```math
\frac{\sum_{a\in\mathcal S_{it}}w_{ait}
  \mathbf 1\{\sum_j\mathrm{cue}_{a,j}\ge2\}}
 {\sum_{a\in\mathcal S_{it}}w_{ait}},
```

the selected-weight share of articles on which at least two frozen cue
families fire.

The builder also records, as audit-only nonpredictors, common-route share,
accepted-event coverage, agreed-`other_or_unclear` share, status-cue all-zero
share, invalid/truncated counts, and the identity

```math
\sum_c \mathrm{eventShare}_{it,c}
 + \mathrm{otherUnclearShare}_{it}
 + \mathrm{orderDisagreementShare}_{it}=1
```

whenever selected weight is positive. Keeping the redundant components out of
the 17-column matrix prevents exact linear dependence while retaining the
diagnostic decomposition.

## Runtime-bounded execution policy

### Profiled execution modes

The full-history upper-bound mode would run the four-way `event_group` once on
all 55,197 unique articles, preserving the complete article history and
avoiding selection bias. It is conditional on the runtime gate below and is
not admitted by the current estimate.

The recent 360-view run used 1,102 field calls and approximately 1,163 seconds
including startup, inferred from filesystem timestamps because elapsed time
was not written to its manifest. The current low-memory scorer evaluates every
candidate in canonical and reversed order. Candidate-sequence scaling gives
the following planning estimates:

| Design | Unique FLAN articles | Nominal estimate |
|---|---:|---:|
| Four-way event group, all articles | 55,197 | 12.945 hours; 14.239 with 10% reserve |
| Four-way event group, 21/21/21 budget-max quota | 52,100 | 12.22 hours; 13.44 with 10% reserve |
| Four-way event group, 16/16/16 headroom quota | 50,488 | 11.84 hours; 13.03 with 10% reserve |
| Four-way event group, 8/8/8 safety quota | 45,192 | 10.60 hours; 11.66 with 10% reserve |
| Existing seven-way event family, 2/2/2 fallback | 26,197 | 10.75 hours; 12.90 with 20% reserve |

These are planning estimates, not guaranteed benchmarks.
The 12.945-hour all-article estimate becomes 14.239 hours after the
documented 10% reserve, so current evidence does **not** admit full-history
execution under the 13.5-hour gate. A materially faster fresh pilot would be
needed. The 21/21/21 selector is the largest profiled set under the gate, but
its estimated margin is only about 3.5 minutes. It is eligible only if the
fresh pilot confirms the bound. The 16/16/16 and 8/8/8 modes provide
progressively more runtime headroom.

### Runtime and semantic gates

Before full inference:

1. Design and iterate the four-way prompt and seven-to-four mapping on the
   frozen 72-record development set only; freeze prompt, mapping, tokenizer,
   model, and code hashes before touching evaluation results.
2. Require 100% valid, untruncated outputs.
3. Run one evaluation pass on the frozen 228-record evaluation set and
   materialize both mapped four-way and mapped frozen-seven-way macro-F1;
   require noninferiority within 0.02.
4. Require at least 85% canonical/reversed choice agreement overall.
   Native/headline agreement is reported only as text-sensitivity stability,
   not as evidence of accuracy.
5. Run a fixed 500-1,000 article CUDA pilot stratified by year, sector,
   C/I/P role, publisher, and text-length stratum.
6. Project end-to-end time, including model load, output writes, validation,
   and a 10% I/O reserve. Full-history execution is eligible only when that
   projection is at most 13.5 hours.

The prompt and acceptance rules must be frozen before the runtime pilot. No
forecast target or downstream validation loss enters the choice.

### Quota fallback

If the four-way prompt passes semantic checks but the all-article projection
exceeds 13.5 hours, select the deterministic union of:

- the top $`K`$ direct articles per stock-day;
- the top $`K`$ sector-common articles per sector-day; and
- the top $`K`$ macro-common articles per forecast date.

Rank within each pool by pre-cutoff aggregation weight descending, publication
time descending, then article ID ascending; deduplicate the union before
inference. Here "direct" is the existing `candidate_roles.direct` flag,
including both target-idiosyncratic and target-common articles; it is not the
mutually exclusive role $`I`$ alone.

The predeclared runtime cascade is:

| Mode | $`K`$ | Unique articles | Aggregation-weight mass | Reserved time |
|---|---:|---:|---:|---:|
| Budget-max four-way | 21 | 52,100 | 98.569% | 13.44 hours |
| Headroom four-way | 16 | 50,488 | 97.326% | 13.03 hours |
| Safety four-way | 8 | 45,192 | 91.779% | 11.66 hours |
| Frozen seven-way semantic fallback | 2 | 26,197 | 64.285% | 12.90 hours with a 20% reserve |

The next symmetric selector, $`K=22`$, projects to 13.504 hours and is
therefore outside the 13.5-hour gate.

Every profiled $`K\ge1`$ represents all 27,306 candidate-bearing stock-days,
all 18,534 direct-bearing stock-days, and all 917 dates. Choose the largest
predeclared four-way mode whose fresh pilot remains at or below 13.5 hours;
the choice cannot use forecast outcomes or downstream validation loss. The
eventual selector must materialize and hash its exact IDs and coverage audit
before inference.

If the new four-way prompt fails its semantic gate, use the already frozen
seven-way event-family prompt only on the 2/2/2 fallback. Its larger label
space would exceed the compute budget on the 8/8/8, 16/16/16, or 21/21/21
sets. Do not mix four-way and seven-way labels in one panel.

## GPT-5.6 Sol R70 cost plans

### Why exact R70 is expensive

The exact v2 contract is target-relative. Relevance, scope, target direction,
sector direction, and peer effect can change when the same article is paired
with a different stock. Two views over 466,902 assignments therefore produce
933,804 logical annotations.

Naively copying one unique-article answer to every target would change the
meaning of those fields. Any deduplicated design must be named
`R70-Lite`, not `R70`.

### Recommended cost ladder

The lowest-risk first paid arm is `G40__gpt_5_6_sol`: one annotation item per
unique article emits only the five article-global semantic blocks below. Up to
eight annotation items may share one API request after pack validation:

- event type;
- affected breadth;
- explicit surprise;
- information status;
- up to two transmission channels;

The globally reusable v2 blocks account for 40 of the 70 daily features:

- event type: 11 shares plus coverage;
- affected breadth: four shares plus coverage;
- explicit surprise: four shares plus coverage;
- information status: four shares plus coverage; and
- transmission channels: 12 shares plus coverage.

This ordered 40-feature global-only design preserves the full calendar and avoids
making target-relative claims that the one-article prompt cannot support.
Its ordered `g40_` feature names are frozen in the machine-readable companion;
they copy the existing v2 event, breadth, surprise, status, and channel
suffixes in that order.
It is not yet construction-ready. The reduced article-request schema and
target-free prompt are derived from existing v2 fields but still must be
frozen and pilot-validated. Primary-output acceptance, field applicability,
daily coverage denominators, no-candidate/missingness behavior, and
multi-label channel aggregation must also be frozen and hashed before a paid
corpus run.

Each global label is inferred once, then reused across the existing
pre-cutoff article-target assignments. Daily stock features remain different
because each stock-day has its own routed article set and 12-hour/duplication
weights; the LLM does not decide assignment relevance or target direction.
Rows with no routed article retain the deterministic availability state rather
than receiving fabricated semantic zeros.

`R70-Lite__gpt_5_6_sol` is a possible second arm, not yet a constructable
feature contract. Its article-level output would add affected-company and
affected-sector records plus explicit entity/sector direction and relation.
A deterministic target projection could then derive relevance, scope, target
direction, sector direction, peer effect, and joint direction. Every projected
entity, direction, or relation must be grounded in an exact substring or
validated character span from the bounded input for **every** article—not
merely on an audit subset. Provider tags may route an article but cannot
manufacture semantic direction.

Before any R70-Lite construction, freeze and hash:

- the exact structured article schema and label enums;
- the ordered daily feature names and final feature count;
- entity normalization and target/peer/sector projection formulas;
- conflict precedence, applicability, unknown, and missingness rules; and
- universal grounding validation and fail-closed behavior.

Until those items exist, do not claim that the 30 target-relative v2 columns
can be reconstructed or that R70-Lite contains 70 features. The existing
40-feature global block is the only ordered GPT-Lite daily design in this
document; it is still construction-incomplete for the reasons above.

### Ranked execution options

Counts below are item annotations. Packed API request counts assume up to
eight annotation items per structured request and are subject to pilot
validation.

| Plan | Item annotations | Approximate packed requests | Reduction from 933,804 | Use |
|---|---:|---:|---:|---|
| Exact v2, two views per assignment | 933,804 | not recommended | baseline | Preserve as unrun oracle contract |
| R70 schema, single-view assignment approximation | 466,902 | about 58,363 | 50.0% | Same 70-column schema, but no two-view adjudication |
| G40 global-only, all unique articles, one view | 55,197 | about 6,900 | 94.09% | Recommended first full-history plan |
| G40 global-only plus fixed 10% second-view audit | 60,717 | about 7,590 | 93.50% | Recommended reliability plan |
| G40 global-only on the FLAN 21/21/21 selector | 52,100 | about 6,513 | 94.42% | Matched budget-max FLAN/GPT comparison |
| G40 global-only on the FLAN 16/16/16 selector | 50,488 | about 6,311 | 94.59% | Matched headroom comparison |
| G40 global-only on the FLAN 8/8/8 selector | 45,192 | about 5,649 | 95.16% | Matched safety comparison |
| Factorized full global plus extractor-visible direct follow-ups | 88,961 | about 11,121 | 90.47% | Better target-direction fidelity |
| Factorized full global plus all deterministic-direct follow-ups | 130,327 | about 16,291 | 86.04% | Higher-cost hybrid |

The hybrid counts are planning options for a future frozen R70-Lite schema.
Each follow-up would return only grounded target direction, sector direction,
and peer effect for a direct target-article pair; all global fields would be
reused.

A separate sector-day top-10 union contains 26,738 unique articles and
retains about 75% of aggregate article weight, but the full 55,197-article
G40 plan is preferred when its measured cost is acceptable because it
avoids top-K selection.

### Request and output reduction

The current exact v2 structured schema is reproducibly 2,207 bytes before
article text. An unmaterialized exploratory profile estimated that:

- removing entity/evidence audit fields reduced it about 20.8%;
- the global five-field schema reduced it about 30.3%; and
- a minimal event/status/channel schema reduced it about 62.7%.

Packing eight articles was estimated to reduce local input-body bytes about
71.2% relative to eight separate requests. The reduced schema transforms and
packing profile are not yet frozen artifacts, so these percentages are
provisional engineering estimates, not reproducible cost evidence. They are
JSON-byte results, not guaranteed token or price reductions.
Output/reasoning work still scales with the number of annotation items, and a
failed pack needs item-addressable retries.

Production responses should contain closed labels and entity IDs, not prose.
Exact evidence grounding is mandatory for every entity/direction/relation
used by projection. A second independent view is drawn for a fixed,
pre-output stratified sample of exactly 5,520 articles. Ambiguous or
high-weight escalation is a separate, explicitly capped diagnostic budget and
is not included in the 60,717 primary item count.

The production G40 panel always uses the predeclared primary view. The sampled
second view is a dataset-level reliability audit, not selective adjudication;
it must not change only the audited articles. Freeze an agreement threshold
from the paid pilot, and fail the whole panel if the production audit misses
it.

### Model, Batch, and pricing policy

No other LLM enters this design. The model remains `gpt-5.6-sol`, with tools
and retrieval disabled and with the existing future-contamination label.

As of 2026-07-29, OpenAI lists GPT-5.6 Sol short-context standard rates of
<span>$</span>5.00 per million uncached input tokens, <span>$</span>0.50 per million cached-read input
tokens, <span>$</span>6.25 per million cache-write tokens, and <span>$</span>30.00 per million billed
output tokens; the model supports structured outputs and the Batch endpoint.
[Official model page](https://developers.openai.com/api/docs/models/gpt-5.6-sol)

OpenAI documents Batch completion within 24 hours at a 50% discount.
[Official Batch guide](https://developers.openai.com/api/docs/guides/batch)

For measured uncached input $`I_u`$, cached-read input $`I_c`$, cache-write
input $`I_w`$, and total billed output $`O`$, including reasoning tokens, the
short-context standard planning formula is:

```math
\mathrm{cost}_{standard}
=
\frac{5 I_u + 0.5 I_c + 6.25 I_w + 30 O}{10^6}.
```

The corresponding published Batch-rate formula is:

```math
\mathrm{cost}_{batch}
=
\frac{2.5 I_u + 0.25 I_c + 3.125 I_w + 15 O}{10^6}.
```

The standard, Batch, short-context, and long-context tables are on the
[official API pricing page](https://developers.openai.com/api/docs/pricing).
Use those Batch rates directly; do not halve them again. GPT-5.6 Sol applies
long-context multipliers when a request exceeds 272,000 input tokens. Every
pack must therefore preflight below 272,000 tokens; otherwise price it under
the official long-context rates for the whole request.

### Paid pilot gate

Before any production request:

1. obtain explicit licensed-text processing confirmation and a hard dollar and
   item budget;
2. for the G40 pilot, draw 300 unique source articles across year, sector,
   role, publisher, and text length;
3. create a new high-reasoning, pack-one, two-view G40 reference
   (600 annotation items);
4. run exactly four one-view challenger cells on the same 300 articles:
   medium/pack-one, high/pack-four, high/pack-eight, and medium/pack-eight
   (1,200 more annotation items; 1,800 total);
5. use the legacy 300-document fine-schema labels only as a non-independent
   stability comparison, not as the reference for the new target-free G40
   schema;
6. record uncached-input, cached-read, cache-write, billed output/reasoning,
   and retry tokens;
7. require exact item-ID recovery, the evidence validation specified by the
   frozen G40 schema, fewer than 272,000 input tokens per pack, and 100%
   schema validity after retries;
   and
8. project full cost with at least 15% retry/token headroom.

Use `medium` reasoning or packed requests only if the frozen pilot shows
acceptable agreement and grounding validity relative to the new
high-reasoning, pack-one reference. The production second-view audit is a
pre-output stratified sample of exactly 5,520 articles; any diagnostic
escalation requires a separate cap and authorization. The pilot is a
separately authorized paid action; this document does not authorize it. An
optional expansion to at most 1,000 unique articles scales the same exact
cells and requires a new item/dollar cap. Future R70-Lite must repeat a
separately authorized pilot after its own schema is frozen; G40 pilot results
cannot silently qualify that different contract.

## Executed FLAN ladder and future GPT ladder

The complete, hash-bound FLAN matched ladder was executed under an explicit
failed-semantic-gate exploratory override:

| Rung | Inputs | Raw columns | Purpose |
|---|---|---:|---|
| `V3-S0` | Q56 | 56 | Exact-row quant control |
| `V3-S1` | Q56 + D2-Normalized | 86 | Exact-row deterministic control |
| `V3-S2` | Q56 + `WL17__flan_t5_xl` | 73 | Q+L contribution |
| `V3-S3` | Q56 + D2-Normalized + `WL17__flan_t5_xl` | 103 | Semantics beyond Q+D |
| `V3-S4` | Shallow XGBoost on S3 inputs | 103 | Validation-gated nonlinear challenger |

All S0-S3 linear rungs and ten matched controls/sensitivities completed. No S2
or S3 target passed the complete useful-semantic gate. Only T2 ETF passed the
S4 validation gate; its XGBoost RMSE was 0.233668, versus 0.242221 for linear
S1 and 0.241935 for linear S3. This nonlinear comparison cannot isolate WL17
because the ladder contains no matched Q56+D2-only XGBoost. See the
[complete training report](training/comparisons/final/RESULTS.md) and
[execution status](training/STATUS.md).

The planned GPT global-only ladder is:

| Rung | Inputs | Raw columns | Purpose |
|---|---|---:|---|
| `V3-G0` | Q56 | 56 | Exact-row quant control |
| `V3-G1` | Q56 + D2-Normalized | 86 | Exact-row deterministic control |
| `V3-G2` | Q56 + `G40__gpt_5_6_sol` | 96 | Q+global semantics |
| `V3-G3` | Q56 + D2-Normalized + `G40__gpt_5_6_sol` | 126 | Global semantics beyond Q+D |
| `V3-G4` | Shallow XGBoost on G3 inputs | 126 | Validation-gated nonlinear challenger |

Repeat the matched template for a future R70-Lite plan only after that plan's
ordered feature contract and count are frozen; its raw-column totals are
therefore deliberately TBD.
The GPT and FLAN feature contracts are not interchangeable merely because they
share input articles.

Required controls are:

- matched Q and Q+D refits on the exact selected row set;
- coverage/text-quality-only L;
- 20-session-stale event labels with contemporaneous coverage;
- fixed within-sector wrong-stock projection;
- within-date/sector semantic-label permutation;
- description-length-at-least-150 sensitivity; and
- full-history versus quota sensitivity if both panels are eventually built.

Residual correction remains deferred until a joint semantic model passes its
matched base and falsification controls.

## Claim and provenance limits

- The Massive archive is retrospective and not article-version safe.
- Entity/peer maps are not effective-dated.
- Untickered macro coverage is incomplete.
- The length experiment has no independent reference labels.
- FLAN W17-Lite is exploratory silver evidence.
- GPT-5.6 Sol has future-knowledge contamination on the historical period and
  remains an exploratory oracle even with tools disabled.
- A planned selector is not missing inference: omitted mass must be explicit.
- An interrupted or incomplete selected-article run invalidates the daily
  panel; partial inference is never converted to zeros.

No existing v1/v2 feature, prediction, result, protocol, or model artifact is
changed by this design.
