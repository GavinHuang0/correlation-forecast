# V2 Q+D+L training ladder and execution record

Status: **the available-data deterministic branch is complete under a locked
exploratory protocol. The shared semantic corpus and the requested FLAN W17
and GPT R70 construction pipelines are ready, but complete semantic inference,
daily L panels, and downstream semantic training have not run**.

Execution disposition (2026-07-29): do not launch the full semantic arms as
the default next step. They are preserved as unrun references. The separate
[cost-bounded v3 design](../v3/README.md) defines the current W17-Lite and
global-only G40 designs plus a later schema-incomplete R70-Lite proposal; no
v3 construction or training has run.

This is a new experiment namespace. It does not extend, renumber, overwrite,
or reinterpret either the completed
[quant-only v1 ladder](../../quant_training/v1/README.md) or the completed
[Q+D v1 experiment](../v1/README.md). Feature definitions remain canonical in
the [v2 design](README.md).

Only two semantic extractors enter this ladder:

- the pinned local `google/flan-t5-xl` research extractor; and
- the explicit OpenAI model `gpt-5.6-sol`.

Llama and every other LLM are excluded. Their completed historical evaluations
remain evidence about earlier pipelines, not entrants in v2.

## Fixed prediction task

Every rung retains the existing four targets:

```text
T1 ETF
T1 leave-one-out peer basket
T2 ETF
T2 leave-one-out peer basket
```

The response remains Fisher-transformed realized correlation, with the same
09:00 ET forecast origin, target construction, T2 boundary purge, and
stock-date pooling used by quant v1. `Q56` means the exact 56-column matched
quant block used by Q+D v1. A Q-only model must be retrained on every new
eligible row intersection; saved long-history quant predictions are parity
diagnostics, not a fair primary comparator after D2 or semantic eligibility
changes the sample.

## Four semantic arms

WLLM17 and RLLM70 are feature contracts, not model names. Crossing the two
contracts with the two permitted extractors gives four arms:

| Arm ID | Extractor | Daily contract | Intended role |
|---|---|---:|---|
| `W17__flan_t5_xl` | FLAN-T5-XL | WLLM17 | Realistic weak-extractor arm |
| `W17__gpt_5_6_sol` | GPT-5.6 Sol | WLLM17 | Accuracy-ceiling arm at the coarse ontology |
| `R70__flan_t5_xl` | FLAN-T5-XL | RLLM70 | Fine-schema stress test for a weak extractor |
| `R70__gpt_5_6_sol` | GPT-5.6 Sol | RLLM70 | Rich exploratory oracle arm |

The primary text input is headline plus provider teaser/description. Full body
is a separately named sensitivity, because the completed 300-document
ablation found that retrieved bodies reduced agreement for both tested local
models under the frozen chunking pipeline.

The common text view is constructed before either model runs:

1. Start with the exact normalized headline.
2. If a description exists, append two line feeds and a deterministic leading
   excerpt of at most 512 Unicode code points. Prefer a whitespace boundary
   within the final 64 code points; otherwise cut exactly at the cap.
3. If no description exists, use headline only and retain
   `description_present = 0` as an audit stratum, not a primary predictor.
4. Store source-description, retained-description, and final-view hashes plus
   source, retained, and omitted character counts. Never rely on tokenizer-side
   truncation.
5. Supply the identical final-view bytes to GPT for the common comparison.

Complete source descriptions, provider tickers, keywords, and other provider
metadata may route candidates, but they are not extractor inputs.
Extractor-visible entities and role flags are recomputed from the bounded
shared view; provider fields and full-source candidate roles are not exposed
to FLAN or GPT.

An uncapped GPT description or body view may be reported only as a separately
named sensitivity; it cannot enter the four-arm common comparison.

### Extractor-specific construction rule

`W17__flan_t5_xl` uses the active coarse FLAN-T5-XL pipeline for shock scope,
event family, and information status. Directional alignment is not admitted
to WLLM17.

`R70__flan_t5_xl` requires a new, pinned FLAN-T5-XL run against the fine
article schema, including the closed fields, channels, entities, and evidence
needed by RLLM70. The old generic FLAN fine runner makes this mechanically
possible, but the active XL wrapper does not expose that contract and no
locked XL fine-schema evaluation exists. A deterministic W17 mapping from
this fine run should be retained as a prompt-contract audit, but it does not
replace the primary active-coarse FLAN W17 arm.

`R70__gpt_5_6_sol` uses the fine schema first. `W17__gpt_5_6_sol` is then
derived from the same adjudicated fine output using the frozen fine-to-coarse
mapping in
[`config/news_feature_schema_coarse.json`](../../../config/news_feature_schema_coarse.json).
The coarsened fields then pass their own W17 field/class acceptance rules;
W17 and R70 acceptance need not be identical. There is no second,
independently prompted coarse GPT call. This prevents W17-versus-R70
differences from being driven by two inconsistent GPT annotations of the same
article.

For eventual API execution, pin the explicit `gpt-5.6-sol` model rather than
the moving `gpt-5.6` alias, use schema-constrained structured output, disable
tools and retrieval, and preserve the full request/response and model
fingerprint. OpenAI currently documents structured-output support and a
2026-02-16 knowledge cutoff for
[`gpt-5.6-sol`](https://developers.openai.com/api/docs/models/gpt-5.6-sol).
The current offline preflight does not authorize a request. A paid pilot or
full run requires an API credential, explicit licensed-text-processing
confirmation, a hard maximum request budget, and separate user authorization.
No paid call has been made.

## Current completeness

“Definition complete” means the daily columns and formulas are specified. It
does not mean the full historical inputs, article labels, aggregation, or
training panel exist.

| Component | Definition | Article extraction | Full corpus | Daily panel | Training status |
|---|---|---|---|---|---|
| D2-Normalized | Complete: 30 columns | Complete deterministic routing | 90,290 provider documents | 71,760 stock-days; 27,510 exact quant matches | D0/D1/D3/D4/D5 and two controls complete |
| Shared semantic corpus | Complete routing/text contract | 466,902 target-specific assignments from 55,197 articles | Complete over 27,510 stock-days; 204 have no candidate | Not an L panel | No semantic training |
| `W17__flan_t5_xl` | Complete: 17 columns | Pinned runner, resumable output, fail-closed aggregator, exact silver-only acceptance lock | Full tokenizer preflight and one-record RTX 3070 Ti CUDA float16 smoke passed; inference not run | Missing | Pipeline ready; not trained |
| `W17__gpt_5_6_sol` | Complete: 17 columns | Bounded fine/coarse references exist | Not pursued in this construction pass | Missing | Not trained |
| `R70__flan_t5_xl` | Complete: 70 columns | No active/pinned XL fine-schema output | Not pursued in this construction pass | Missing | Not trained |
| `R70__gpt_5_6_sol` | Complete: 70 columns | Offline two-view Batch/retry/merge/adjudication/aggregation pipeline; final hash-bound preflight passed | 933,804 requests; 3,613,888,330 bytes; maximum request size 4,458 bytes; paid inference not authorized or run | Missing | Offline pipeline ready; paid execution blocked; not trained |

The remaining semantic gaps are:

- no full-corpus W17 or R70 article inference for either requested extractor;
- no completed daily W17/R70 feature panel or model-ready L manifest;
- no human-grounded, cross-fitted field/class acceptance calibration; and
- no retrospective provider archive with recoverable historical text
  versions.

The shared corpus, both requested runners, and both daily aggregators exist.
FLAN W17 deliberately uses a frozen permissive \(q=1\) silver-only acceptance
profile because no human calibration exists; it remains ineligible for primary
or confirmatory claims. GPT R70 also remains an uncalibrated \(q=1\) oracle
unless an earlier human calibration is added.

The current FLAN-T5-XL evaluation covers 228 benchmark documents and reports
0.464 mean macro-F1 against GPT silver labels; every field missed its frozen
threshold. The earlier fine-schema run used FLAN-T5-Large, not XL, and its
fine semantics were unusable. Those results do not establish R70 readiness.

## Completed available-data execution

The deterministic panel and protocol were hash-locked before fitting. The
model panel has 27,510 rows, 917 dates, and 30 stocks. All results use the same
three chronological folds and are development-only because the retrospective
news archive is not historical-version safe.

| Bundle | Inputs / estimator | Execution |
|---|---|---|
| `V2-D0` | Q56 Elastic Net | Complete matched quant control |
| `V2-D1` | D2-Normalized Elastic Net | Complete deterministic-only diagnostic |
| `V2-D2` | Q56 + matched D43 Elastic Net | Skipped: exact D43 was not materialized under v2 routing |
| `V2-D3` | Q56 + D2-Normalized Elastic Net | Complete primary Q+D fit |
| `V2-D4` | Q56 + D2-Normalized + D2-Levels Elastic Net | Complete raw-level sensitivity |
| `V2-D5` | Q56 + D2-Normalized shallow XGBoost | Complete for validation-gated `t1_etf` only; CUDA confirmed |
| `C-D3-L20` | Q56 + 20-session-stale D2 | Complete falsification control |
| `C-D3-WS` | Q56 + fixed wrong-stock D2 | Complete falsification control |

Against D0, D3 changed MSE by -0.31%, +0.04%, +2.13%, and +0.67% for
T1 ETF, T1 LOO, T2 ETF, and T2 LOO respectively. The two T2 improvements had
paired 95% block-bootstrap intervals below zero, but neither beat the stale-D2
control with a confidence interval below zero. No target passed the complete
useful-news gate. D5 improved T1 ETF MSE by 1.16% versus D0, but its paired
interval included zero.

See the [bundle status](training/STATUS.md), the
[final paired comparison](training/comparisons/final/RESULTS.md), and the
[semantic readiness ledger](training/semantic/STATUS.md).

### Eligibility vocabulary

The future artifact manifests must keep these states separate:

```text
definition_complete
construction_complete
exploratory_fit_eligible
primary_training_eligible
confirmatory_eligible
```

All four L arms are definition-complete, but their execution states differ:
FLAN W17 and GPT R70 have construction pipelines while GPT W17 and FLAN R70
were not pursued in the current pass. None is article-inference complete or
daily-panel ready. After full-corpus inference, audit completion, and the
arm-specific gates, a later authorization may set
`exploratory_fit_eligible = 1` for the requested development experiment.
That permission would not imply primary or confirmatory eligibility.

- FLAN W17 remains an `exploratory_silver_fit` unless human calibration passes.
- FLAN R70 remains a fine-schema stress test unless its output validity,
  coverage, and field-quality gates pass.
- Both GPT arms remain `exploratory_future_contaminated_oracle`, regardless of
  apparent downstream performance.

## Leakage and claim labels

GPT-5.6 Sol has a documented 2026-02-16 knowledge cutoff. Applying it to
articles and forecast dates at or before that cutoff risks latent knowledge of
later outcomes even when the prompt supplies only contemporaneous text. The
user has requested this arm as an exploratory accuracy reference, so every GPT
artifact and result must carry:

```text
extractor_model_knowledge_contaminated = 1
confirmatory_eligible = 0
result_role = exploratory_future_contaminated_oracle
```

Prompts may receive the eligible article text and contemporaneous entity
metadata only. They may not receive prices, realized correlations, residuals,
future returns, model losses, or later news, and they may not use web search,
file search, or other tools. Evidence grounding limits unsupported
annotation, but it cannot remove latent pretraining contamination.

FLAN results are also development-only: the project has a release-boundary
proxy rather than a documented checkpoint knowledge cutoff, and the
retrospective news text is not version-safe. Chronological folds cure neither
extractor-knowledge contamination nor provider-version uncertainty.

## Source profiles and sample scopes

Run each provider product as a separate source profile:

```text
ordinary_massive_retrospective
benzinga_retrospective
prospective_versioned
```

Never concatenate those profiles into one nominally homogeneous panel. If
ordinary Massive and paid Benzinga are compared, use the exact matched
stock-date and article-input intersection in addition to each profile's
maximal sample.

Two sample scopes are proposed:

1. `D2-extended`: Q, matched D43, and D2 on the longest source-complete
   deterministic period. This estimates whether normalization helps when LLM
   availability is not the binding constraint.
2. `semantic-common`: every Q/D/L arm on one common FLAN-compatible period,
   provisionally 2022-11-01 through 2026-06-30, with identical provider
   profile, article input hashes, stocks, dates, and targets across all four
   semantic arms.

The final row counts must be recomputed. V1 counts cannot be copied because
D2 requires the current and 126 prior source-complete sessions, while semantic
construction adds full-inference and source-completeness requirements.

Model acceptance must not decide whether a stock-day exists. A source-complete
day with no eligible article remains a row. A source-complete day on which all
applicable predictions abstain also remains a row, with missing class shares
and zero applicable-field coverage. An incomplete query or inference batch
makes the entire semantic stock-day ineligible; partial nonzero aggregates
are forbidden.

Report both:

- each arm's maximal source-complete matched panel; and
- a four-arm common-intersection panel for direct FLAN/GPT and W17/R70
  comparisons.

## Chronological development folds

Reuse the quant-v1 dates only as already-inspected development blocks:

| Fold | Train | Validation | Development evaluation |
|---|---|---|---|
| 1 | 2022-11-01–2024-06-30 | 2024-07-01–2024-12-31 | 2025-01-01–2025-06-30 |
| 2 | 2022-11-01–2024-12-31 | 2025-01-01–2025-06-30 | 2025-07-01–2025-12-31 |
| 3 | 2022-11-01–2025-06-30 | 2025-07-01–2025-12-31 | 2026-01-01–2026-06-30 |

All stocks on a date stay together. T2 removes the final four forecast
sessions from each train, validation, and evaluation block. Scaling,
imputation, missingness-indicator creation, tuning, and early stopping use the
training/validation partitions only.

All dates through 2026-06-30 have already informed project decisions.
Consequently, these are development evaluations even when an internal block
is named “evaluation.” Confirmatory evidence requires a protocol frozen
before inspection and a prospectively versioned forecast period beginning
strictly after 2026-06-30. The GPT oracle arms remain non-confirmatory even
then unless the extractor predates the observations under a defensible
knowledge-time rule.

## Deterministic ladder

Elastic Net is primary. Counts below exclude an intercept and any
training-fold-derived missingness indicators.

| Rung | Estimator | Inputs | Raw columns | Purpose |
|---|---|---|---:|---|
| `V2-D0` | Elastic Net | Q56 | 56 | Matched quant control |
| `V2-D1` | Elastic Net | D2-Normalized | 30 | Deterministic-only diagnostic |
| `V2-D2` | Elastic Net | Q56 + D43-recomputed | 99 | Exact v1-design comparator on the v2 source/row universe |
| `V2-D3` | Elastic Net | Q56 + D2-Normalized | 86 | Primary redesigned Q+D model |
| `V2-D4` | Elastic Net | Q56 + D2-Normalized + D2-Levels | 91 | Provider-level sensitivity only |
| `V2-D5` | Shallow XGBoost | Q56 + D2-Normalized | 86 | Validation-gated nonlinear challenger |

`D43-recomputed` means the exact v1 43-column model contract rebuilt using the
same provider profile, cutoff rules, dates, and eligible rows as D2. The
completed v1 A5 result is historical evidence, not a substitute for `V2-D2`.

Primary deterministic contrasts are:

- `V2-D3` versus `V2-D0`: total redesigned deterministic increment;
- `V2-D3` versus `V2-D2`: normalization/redesign versus the matched D43
  contract; and
- `V2-D4` versus `V2-D3`: whether raw provider levels add information or
  merely reintroduce ticker/provider coverage identity.

`D2-Propagation` is excluded until a cutoff-safe near-duplicate method passes
its own validation. If later admitted, it receives a separately named
sensitivity rung and never changes `V2-D3`.

## Semantic ladder

Repeat the table below for each of the four arm IDs. \(L_{E,C}\) denotes the
17- or 70-column daily block created by extractor \(E\) under contract \(C\).

| Rung template | Estimator | Inputs | W17 columns | R70 columns | Purpose |
|---|---|---|---:|---:|---|
| `V2-S0__<arm>` | Elastic Net | Q56 | 56 | 56 | Exact matched Q control |
| `V2-S1__<arm>` | Elastic Net | Q56 + D2-Normalized | 86 | 86 | Exact matched Q+D control |
| `V2-S2__<arm>` | Elastic Net | Q56 + \(L_{E,C}\) | 73 | 126 | Q+L total semantic-pipeline contribution |
| `V2-S3__<arm>` | Elastic Net | Q56 + D2-Normalized + \(L_{E,C}\) | 103 | 156 | Q+D+L semantic increment beyond D2 |
| `V2-S4__<arm>` | Shallow XGBoost | Same inputs as `S3` | 103 | 156 | Validation-gated nonlinear endpoint |

These are raw contract counts. In an uncalibrated \(q=1\) R70 arm,
`rllm_mean_accepted_quality_weight` is constant whenever labels are present
and is excluded from fitting, producing effective Q+L and Q+D+L counts of 125
and 155 before missingness indicators. The raw 70-column artifact remains
unchanged.

The decisive within-arm contrasts are:

```text
S2 - S0    Q+L versus matched Q
S3 - S1    Q+D+L versus matched Q+D
S3 - S2    deterministic increment after semantics
S1 - S0    deterministic increment on the exact semantic row set
```

The cross-arm contrasts answer different questions:

- GPT versus FLAN within W17 estimates the effect of the complete extraction
  pipeline at a fixed daily ontology; it is not a clean model-architecture
  experiment because prompts and calibration differ.
- GPT versus FLAN within R70 is the analogous fine-ontology comparison, if the
  FLAN fine arm passes execution gates.
- R70 versus W17 within one extractor tests whether added semantic detail is
  useful. It is not a pure accuracy comparison because the feature spaces
  differ.
- `W17__gpt_5_6_sol` versus `R70__gpt_5_6_sol` is the cleanest schema-detail
  comparison because both come from the same GPT fine labels.

Q and Q+D predictions may be reused across arms only after exact equality of
row keys, targets, source profile, features, preprocessing, folds, and input
hashes. Otherwise their matched controls are refit.

## Missingness and preprocessing

The named 17 and 70 counts are raw pre-imputation contract sizes.

For Elastic Net:

- fit scaling and imputation on the current training partition only;
- add one explicit missingness indicator per nullable semantic field group;
- count and name those indicators outside WLLM17/RLLM70 in every manifest;
- keep no-eligible, all-abstain, structurally inapplicable, and unavailable
  states distinct; and
- never turn a missing class-share vector into a genuine all-zero semantic
  observation.

W17 has three nullable semantic field groups: scope, event family, and
information status. R70 has the ten field groups named by its coverage
features. Any additional derived-field missingness rule must be frozen and
counted before fitting.

XGBoost may use native missing values, but it receives the same explicit
availability/coverage fields and may not infer source incompleteness from a
partial aggregate.

Reuse the fixed quant-v1 Elastic Net and shallow-XGBoost search budgets. Do not
expand a grid because an oracle arm underperforms. `V2-D5` or a given
`V2-S4` becomes eligible only when its corresponding linear endpoint improves
matched validation loss in at least two of the three validation blocks,
without consulting a development-evaluation block.

## Required falsification controls

Run the following for every primary linear deterministic or semantic endpoint:

1. **Twenty-session stale block.** Shift the complete D2 or L block by 20
   official sessions within ticker and source profile.
2. **Wrong-stock block.** Apply one frozen same-date within-sector ticker
   rotation. Do not invent an after-the-fact permutation that is easier to
   beat.
3. **Coverage-only L control.** Retain semantic availability, applicability,
   abstention, and quality/coverage fields while removing semantic class
   shares. For W17 this is the no-eligible flag plus three coverage fields; for
   R70 it is the no-eligible flag, ten coverage fields, and mean accepted
   quality.
4. **Semantic-label permutation.** Permute accepted semantic share vectors
   within date and sector while preserving applicability, missingness, and
   field coverage.
5. **Matched D43 comparator.** Always report the exact `V2-D2` result beside
   redesigned D2 results.

For `V2-S3`, stale and wrong-stock controls shift or rotate L while retaining
the contemporaneous D2 block, so they test semantic content rather than
destroying the deterministic comparator. A separate D2-control set shifts or
rotates D2 with Q held fixed.

A useful branch must improve its exact matched base, beat its stale and
wrong-stock controls, beat the L coverage-only control where applicable, and
improve more than one development fold. All four targets are reported
separately; pooled improvement cannot conceal a loss on ETF or LOO.

## Metrics and inference

Retain the completed protocols' metrics:

```text
Fisher-z RMSE and MAE
raw-correlation RMSE and MAE
OOS R-squared versus persistence
incremental R-squared versus the exact matched Q or Q+D base
```

Report paired Fisher-space squared-loss differences with moving date-block
bootstrap intervals. Resample whole dates with all stocks together, keep
blocks inside each model-refit fold, and retain the existing 2,000 resamples,
ten-session block length, and seed 1729 for continuity. Also report:

- fold-specific results;
- source profile, ticker, sector, calendar year, and direct/no-direct-news
  slices;
- semantic coverage and all-abstain rates by field and extractor;
- feature-selection stability for Elastic Net; and
- common-intersection results beside maximal-panel results.

These intervals measure uncertainty within an already-inspected development
period. They are not confirmatory p-values.

## Why residual correction is deferred

The completed v1 residual branch had fewer independent correction-training
dates, and the strong quant model left little residual signal. V2 therefore
makes joint Q+D, Q+L, and Q+D+L models primary.

A residual branch becomes eligible only after a joint semantic arm passes its
validation and falsification gates and forward-chained, cross-fitted quant
forecasts exist throughout the same training dates. Saved outer-test
predictions beginning in 2025 are insufficient for a controlled v2
joint-versus-residual comparison.

## Remaining artifacts before semantic training

The D2 artifacts, shared semantic input corpus, ordered W17/R70 feature
contracts, FLAN W17 construction code and silver-only acceptance lock, and GPT
R70 offline construction code and preflight now exist. They do not make either
arm daily-panel ready or authorize downstream fitting.

Before a Q+L or Q+D+L run, each participating arm still requires:

- a complete hash-bound article-inference manifest over all 466,902
  assignments, with terminal success/failure coverage and no partial aggregate;
- a complete daily W17 or R70 panel and exact stock-date
  eligibility/common-intersection report;
- final extractor runtime, preflight, prompt/view, acceptance/adjudication, and
  daily-panel hashes;
- a locked machine-readable semantic training protocol with folds,
  transformations, controls, metrics, and random seeds; and
- explicit flags distinguishing development, silver-only,
  future-contaminated oracle, primary, and confirmatory eligibility.

FLAN W17's full tokenizer preflight passed all 466,902 assignments and
4,202,118 logical prompts with zero violations; its one-record CUDA smoke also
passed with no failure or truncation. GPT R70's final hash-bound preflight is
complete, but it has not made a paid call; a paid pilot/full run remains gated
by an API credential, licensed-text confirmation, an explicit request budget,
and separate user authorization. No W17/R70 daily feature, semantic
prediction model, fit, or evaluation output exists.

Use the [semantic construction runbook](training/semantic/CONSTRUCTION_RUNBOOK.md)
for the exact execution sequence and completion gates.
