# Forecasting Stock-Sector Coupling

Research project testing whether point-in-time financial news improves
forecasts of how strongly an individual stock will move with its sector.

## Research question

Do LLM-extracted event features add predictive and economic value beyond
quantitative correlation models when forecasting daily stock-sector coupling?

The main hypothesis is that event scope matters:

- firm-specific events may make a stock move more idiosyncratically;
- sector-wide events may strengthen stock-sector coupling; and
- macroeconomic events may strengthen broad common-factor exposure.

## Operational model paths

FLAN-T5-XL v1.1 is now the active research extractor:

| Extractor | Status | Protocol/results |
|---|---|---|
| **FLAN-T5-XL v1.1** | **Active research extractor**; best tested mean macro-F1 and accuracy, below every production threshold | [`experiments/flan_t5_xl/v1_1/README.md`](experiments/flan_t5_xl/v1_1/README.md) |
| FLAN-T5-Large v0.4 | Frozen prior baseline retained for paired comparisons | [`experiments/flan_t5/v0_4/README.md`](experiments/flan_t5/v0_4/README.md) |
| Llama 2 7B Chat v1.2 | Historical research candidate; event-family specialist, not an all-field replacement | [`experiments/llama_2/README.md`](experiments/llama_2/README.md) |
| Llama 3.1 8B Instruct v1.0 | Archived after failing all semantic development thresholds; holdout preserved | [`experiments/llama_3_1/archive/v1_0/README.md`](experiments/llama_3_1/archive/v1_0/README.md) |

XL v1.1 applies a fieldwise score calibration selected only on the fixed
72-document development split. On the locked 228-document comparison it
achieved 0.612 mean accuracy and 0.464 mean macro-F1, versus 0.494 and 0.446
for FLAN-T5-Large v0.4. All four semantic threshold checks still failed, so
“active” means the default research path, not production approval.

Machine-readable contracts:

- [`experiments/active_extractor.json`](experiments/active_extractor.json)
- [`experiments/flan_t5_xl/registry.json`](experiments/flan_t5_xl/registry.json)
- [`experiments/flan_t5/active_baseline.json`](experiments/flan_t5/active_baseline.json)
- [`experiments/flan_t5/registry.json`](experiments/flan_t5/registry.json)
- [`experiments/llama_2/protocol.json`](experiments/llama_2/protocol.json)
- [`experiments/llama_2/benchmark_manifest.json`](experiments/llama_2/benchmark_manifest.json)
- [`experiments/llama_2/research_candidate.json`](experiments/llama_2/research_candidate.json)
- [`experiments/llama_3_1/protocol.json`](experiments/llama_3_1/protocol.json)
- [`experiments/llama_3_1/benchmark_manifest.json`](experiments/llama_3_1/benchmark_manifest.json)
- [`experiments/llama_3_1/registry.json`](experiments/llama_3_1/registry.json)
- [`config/quant_training_protocol_v1.json`](config/quant_training_protocol_v1.json)
- [`experiments/quant_training/v1/README.md`](experiments/quant_training/v1/README.md)

The previous FLAN experiments, rejected v0.5 candidate, historical scripts,
tests, and reorganization records are kept separately in
[`experiments/flan_t5/archive/`](experiments/flan_t5/archive/README.md).

Generated local data follows the same separation:

```text
outputs/
  flan_t5_xl/
    v1_1/         # active calibrated XL predictions/results
    v1_0/         # superseded raw XL development baseline
  flan_t5/
    shared/       # fixed 300-article benchmark
    v0_4/         # prior FLAN-Large baseline
    archive/      # prior and rejected FLAN outputs
  llama_2/
    shared/       # prepared byte-identical 300-article benchmark
    v1_0/         # frozen failed baseline and diagnostics
    v1_1/         # corrected answer-boundary decoder
    v1_2/         # development-selected fieldwise hybrid
  llama_3_1/
    shared/       # post-cutoff byte-identical 72/228 benchmark
    v1_0/         # completed development screen; evaluation holdout preserved
  quant_training/
    v1/
      rung_01/ ... rung_04/
      comparisons/
```

`outputs/`, licensed/raw news, model weights, and secrets remain Git-ignored.

## Original extractor-selection benchmark and labels

The original extractor-selection benchmark contains 300 Alpha Vantage
headline/summary records from 2024:

- development: 72 records;
- evaluation: 228 records;
- evaluation records relevant under the reference: 174.

GPT-5.6 Sol produced the silver reference:

- fine labels:
  [`annotations/chatgpt_5_6_sol_reference.jsonl`](annotations/chatgpt_5_6_sol_reference.jsonl);
- deterministic coarse labels:
  [`annotations/chatgpt_5_6_sol_reference_coarse_v0_2.jsonl`](annotations/chatgpt_5_6_sol_reference_coarse_v0_2.jsonl).

The metrics are agreement with GPT silver annotations, not objective accuracy
against human ground truth. Exact evidence is retained for manual audits.

## FLAN-T5 v0.4 result

The original grouped FLAN prompt often echoed option lists, so all 300 strict
core parses were null. Later versions used one-field constrained scoring,
coarse economically motivated labels, deterministic relevance/surprise rules,
article-first prompts, hierarchical routing, and option-order averaging.

The selected v0.4 hybrid achieved on the original fixed 228-record split:

| Metric | Mapped v0.2 | Active v0.4 |
|---|---:|---:|
| Mean field accuracy | 0.422 | **0.494** |
| Mean semantic macro-F1 | 0.341 | **0.446** |

v0.4 remains below every semantic go/no-go threshold, so it is an exploratory
baseline rather than a production extractor. The final v0.5 experiment fell
to 0.348 mean macro-F1 and was rejected under its preregistered promotion
rule.

The active v0.4 score is a hybrid: two fields use the v0.4 coarse prompt and
two use mappings from the same pinned FLAN model's v0.2 fine prompt. The exact
fine-compatibility prediction and manifest have been hash-verified into
`outputs/flan_t5/v0_4/dependencies/v0_2_fine/`; the rest of v0.2 is archived.

## FLAN-T5-XL v1.1 result

The XL experiment reused the same schema, prompts, deterministic routing, and
2024 benchmark with the larger pinned `google/flan-t5-xl` checkpoint. The raw
XL decoder achieved 0.407 mean macro-F1 on the locked split. A bounded
development experiment then selected one of 27 score-source/log-prior rules
per semantic field and froze those choices before evaluating the 228 records.

| Metric | Raw XL v1.0 | Active XL v1.1 |
|---|---:|---:|
| Mean field accuracy | 0.546 | **0.612** |
| Mean semantic macro-F1 | 0.407 | **0.464** |

Use the frozen wrapper for new point-in-time inputs:

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv-flan-t5-xl\Scripts\python.exe scripts\run_flan_t5_xl_active.py `
  --input <POINT_IN_TIME_INPUT.jsonl> `
  --output <CALIBRATED_OUTPUT.jsonl>
```

The wrapper verifies the active schema and calibration hashes, runs the pinned
CUDA/FP16 XL configuration, retains raw scores, and applies the frozen
postprocessor. See the
[`XL runbook`](experiments/flan_t5_xl/v1_1/README.md)
for validation, resumption, reproduction, and claim limits.

## Llama 2 comparison

The local comparison uses:

```text
meta-llama/Llama-2-7b-chat-hf
revision f5db02db724555f92da89c216ac04704f23d4590
```

Meta documents a September 2022 pretraining cutoff and some tuning data
through July 2023. Every benchmark article is from 2024, so the unchanged
dataset is eligible under the repository's conservative July 31, 2023
boundary. No new GPT labeling is required.

The frozen runtime is CUDA/FP16 with NF4 double quantization, eager attention,
batch size 1, a 4,096-token no-truncation limit, and the official Llama 2
single-turn `[INST]` format. It does not generate prose and has no network
access during extraction.

All 300 documents were processed: 72 development records and 228 evaluation
records. The 228 records were a held-out comparison for v1.0, but became a
post-hoc comparison for v1.2 after v1.0 results and failure modes had been
inspected. Of those records, 174 are reference-relevant.

| Metric | FLAN-T5 v0.4 | Llama v1.0 | Llama v1.2 hybrid |
|---|---:|---:|---:|
| Mean field accuracy | 0.494 | 0.445 | **0.537** |
| Mean semantic macro-F1 | **0.446** | 0.188 | 0.280 |

v1.0 failed primarily because it scored option letters before Llama emitted
its assistant prefix. v1.1 derives the exact partial assistant `Answer:`
continuation from the pinned tokenizer and averages every label across every
cyclic option position. As a full replacement it was rejected on development:
event-family macro-F1 improved from 0.059 to 0.548, but scope and alignment
worsened.

v1.2 therefore freezes a fieldwise hybrid selected on the 72-document
development split: v1.1 supplies event family and information status, while
v1.0 supplies scope and alignment. On the remaining 228 records, reported
only as a post-hoc engineering comparison, it improves over v1.0 by 0.092
mean accuracy and 0.092 mean macro-F1. Its event-family result is competitive
with FLAN (0.449 versus 0.452 macro-F1), but scope and status remain strongly
majority-dominated. FLAN v0.4 remained the better all-field baseline in that
comparison. A fresh external holdout is required before promoting Llama v1.2
beyond a research candidate.

See the
[`Llama 2 runbook`](experiments/llama_2/README.md)
for the exact runs, evaluation, diagnosis, and completed decoder experiment.

## Llama 3.1 8B Instruct development screen

Because XL v1.1 still misses all semantic thresholds, the pinned
`meta-llama/Llama-3.1-8B-Instruct` checkpoint was screened locally. Meta
documents a December 2023 knowledge cutoff; the fixed articles begin January
2, 2024, so the same benchmark and existing silver labels were eligible.

The completed pipeline:

- cache and hash the gated immutable checkpoint;
- create byte-identical, label-free development/evaluation inputs;
- enforce the official chat template and exact assistant boundary;
- average all cyclic option positions;
- support complete multi-token candidate scoring; and
- run NF4/FP16 at batch size one for the local 8 GiB GPU.

The 13-file, 16.07 GB snapshot was verified offline. Smoke extraction and all
72 development records completed with no truncation. The deterministic
relevance gate passed at 0.932 F1, but the semantic model failed all four
field thresholds:

| Metric | Llama 3.1 v1.0 | Active XL v1.1 development |
|---|---:|---:|
| Mean field accuracy | 0.576 | **0.616** |
| Mean semantic macro-F1 | 0.387 | **0.472** |

Scope macro-F1 was only 0.177 and alignment macro-F1 was 0.240. The model
classified 52 of 56 reference-relevant records as `idiosyncratic`, so the
hierarchy also forced most alignment outputs to `single_firm_only`.

v1.0 was rejected without running the 228-document evaluation split. This
preserves the holdout for a future configuration whose scope rule and
scope-dependent alignment routing are frozen before inference. The active
research extractor remains FLAN-T5-XL v1.1.

See the [`Llama 3.1 runbook`](experiments/llama_3_1/README.md) and
[`archived development result`](experiments/llama_3_1/archive/v1_0/README.md).

## Schemas

- Human-readable feature definitions:
  [`docs/news_feature_schema.md`](docs/news_feature_schema.md)
- Original fine schema:
  [`config/news_feature_schema.json`](config/news_feature_schema.json)
- Active coarse schema:
  [`config/news_feature_schema_coarse.json`](config/news_feature_schema_coarse.json)
- Target universe metadata:
  [`config/target_universe.json`](config/target_universe.json)

The active semantic fields are:

```text
shock_scope
event_family
information_status
directional_alignment
```

Relevance and explicit-surprise features are produced by deterministic,
auditable rules.

## Research design

1. Measure realized stock-sector correlation from intraday returns.
2. Use a pooled panel of liquid stocks and sector benchmarks.
3. Establish lagged-correlation, HAR, exponential, regularized-linear,
  shallow-tree, and causal DCC-GARCH quantitative baselines.
4. Preserve the completed exploratory deterministic-news ablation and design
   a separately versioned normalized deterministic/semantic experiment, while
   reserving strict point-in-time claims for a version-preserving archive.
5. Keep every stock observed on the same date in the same chronological fold.
6. Preserve the completed D43 ablations and placebo controls, then evaluate
   any future normalized deterministic and semantic blocks through separately
   versioned, nested comparisons; conventional sentiment remains a separate
   control.
7. Test economic value through hedge error, portfolio risk, and
  coupling-aware filters, not forecast error alone.

## Historical price data

The fixed price universe is in
[`config/price_universe.json`](config/price_universe.json). It contains 30
stocks, five sector ETFs, and downloaded SPY bars reserved for a future market
control. No SPY-derived feature is currently emitted. The initial data contract
is:

```text
source:              Alpaca Market Data API v2
feed:                SIP
bar interval:        15 minutes
adjustment:          all
session:             09:30-16:00 America/New_York
recommended range:   2016-01-01 through 2026-06-30
```

Regular-session filtering uses Alpaca's historical market calendar rather
than a fixed wall-clock close. This removes bars after official early closes.
The normalized calendar response is cached beside the price data and its hash
is recorded in each chunk manifest.

The downloader reads `ALPACA_PUBLIC_KEY` and `ALPACA_SECRET_KEY` from the
process environment or the Git-ignored `.env` file. It never writes credential
values into logs or manifests.

Validate the full plan without making a request:

```powershell
python scripts/fetch_alpaca_bars.py --dry-run
```

Run a small authenticated smoke test first:

```powershell
python scripts/fetch_alpaca_bars.py `
  --symbols AMD,SOXX `
  --start 2024-01-02 `
  --end 2024-01-05 `
  --output-dir data/prices/alpaca-smoke
```

Run the resumable full-universe backfill:

```powershell
python scripts/fetch_alpaca_bars.py
```

Results are stored as one gzip-compressed CSV and one audit manifest per
symbol-year. Complete chunks are hash-checked and skipped on subsequent runs.
Use `--overwrite` only when intentionally replacing an existing chunk.

## Additional quantitative features

The extended-hours and external-context pipeline is documented in
[`docs/additional_quant_features.md`](docs/additional_quant_features.md).
It provides:

- narrow premarket and aftermarket Alpaca requests that do not re-download
  regular-session bars;
- official FRED VIX/Treasury series;
- Ken French daily factors;
- BLS, BEA, and Federal Reserve release-calendar inputs;
- a resumable Alpha Vantage historical reported-earnings archive with explicit
  unknown-timing warnings;
- an offline builder for overnight/premarket returns, relative volumes,
  realized volatility, sector dispersion, macro flags, and simplified
  factor-implied correlation.

The extended-hours downloader writes only to
`data/prices/alpaca-extended/`. The feature builder reads the regular bar
directory without modifying it and writes a separate Parquet panel under
`data/features/quant/`. The locked training artifact includes corrected
first-bar realized volatility, exact official-session lagging, and explicit
extended-hours availability indicators.

## Bollerslev core features

The offline builder in
[`scripts/build_bollerslev_core_features.py`](scripts/build_bollerslev_core_features.py)
constructs the 22 feasible price-derived features from Bollerslev, Li, and
Tang's correlation-forecasting framework:

```text
6 HAR/downside correlation features
8 finite-500-session exponential pair features
8 within-sector exponential state features
```

It uses only completed regular-session chunks, reconstructs the paper-style
overnight component from prior close to current open, and assigns every
forecast-date row information through the preceding official session only.
It does not read the extended-hours download directory.

See
[`docs/bollerslev_core_features.md`](docs/bollerslev_core_features.md)
for formulas, commands, the exact 22-column schema, verification, and the
documented differences between the published monthly stock-pair design and
this daily stock-sector adaptation.

## Quant training

The first locked quant experiment is complete through rung 4. It skips the T0
proxy and trains the following four targets in parallel:

```text
T1 ETF: same-day regular-session stock-ETF correlation
T1 LOO: same-day correlation with an equal-weight five-peer basket
T2 ETF: current-plus-next-four-session stock-ETF correlation
T2 LOO: current-plus-next-four-session five-peer correlation
```

ETF is the tradable **hedge-coupling** target; LOO removes mechanical ETF
self-inclusion and is the cleaner research target. Both use the same strict
official interval set. T2 sums covariance and variance components over five
sessions rather than averaging daily correlations.

The matched modeling panel contains 27,510 stock-date rows from 2022-11-01
through 2026-06-30. Three chronological folds, training-only preprocessing,
and target-end purging are locked in
[`config/quant_training_protocol_v1.json`](config/quant_training_protocol_v1.json).

The best development models are the rung-3 shallow XGBoost challenger or, for
T1 ETF, its validation-qualified `elastic_xgboost_ensemble`:

| Target | Best model | Fisher-z RMSE | OOS R² vs persistence |
|---|---|---:|---:|
| T1 ETF | `elastic_xgboost_ensemble` | 0.3686 | 0.4154 |
| T1 LOO | XGBoost | 0.3814 | 0.3964 |
| T2 ETF | XGBoost | 0.2348 | 0.2885 |
| T2 LOO | XGBoost | 0.2567 | 0.2517 |

Because these outer blocks were used to rank the rungs, the winning metrics
are development estimates, not unbiased final-holdout estimates. A future
period must remain untouched for confirmation.

The audited construction, exact feature and target equations, model
objectives, evaluation formulas, completed ladder, remaining limitations, and
links to all per-rung results are in
[`docs/training_readiness.md`](docs/training_readiness.md) and
[`experiments/quant_training/v1/README.md`](experiments/quant_training/v1/README.md).

## News-provider and deterministic-feature experiment

The Massive/Alpha/full-text provider audit is isolated under
[`experiments/news_provider_fulltext/v1_0/`](experiments/news_provider_fulltext/v1_0/README.md).
Its generated provider payloads, licensed text, Q+D panel, local-model
predictions, and GPT silver-label batches remain Git-ignored.

The free Massive ordinary-news endpoint supplied a cursor-exhausted snapshot
within the configured 30-stock, five-sector-benchmark, and SPY query scope.
This does not claim exhaustive market-wide coverage, a complete still-open
July 26 UTC day, or a contractual right to the observed pre-two-year history.
After cross-query deduplication the snapshot contains 90,290 unique provider
documents. It has been converted into 44 auditable deterministic-news columns
and joined one-to-one to the 27,510-row quant panel. Two columns are constant
and two are exact timing redundancies, so the manifest recommends 40 columns
for fitting. The resulting artifact is sufficient for exploratory Q+D model
integration, but not for a strict point-in-time training claim: ordinary
Massive news lacks first-seen timestamps, update/version history, and article
bodies. The panel and manifest are bound by SHA-256
`352ba2635febfa0bc36bc6f797c00036a8aee9879327a8176e9accf180e50150`
and
`ac161d48901fae669fdc500099df92d218513d2e7d8d383539b8ed326445d86f`.

The separate
[quant plus deterministic-news experiment](experiments/quant_deterministic_news/v1/README.md)
is complete, while the quant-only ladder remains unchanged. Exact Q-only
replays passed, 25 model/control bundles were fitted, and the final report
contains 118 paired target comparisons with 236,000 fold-contained
date-block bootstrap draws. A5 improved T2 ETF MSE by 3.65% versus matched
Elastic Net, but a 20-session stale-news placebo performed better. Residual
news models did not robustly improve the primary XGBoost base. The current
decision is therefore to retain the quant-only model and treat the news
results as exploratory until prospective version-safe confirmation.

The
[deterministic and semantic v2 design](experiments/quant_deterministic_news/v2/README.md)
preserves that v1 record and specifies a 30-feature normalized deterministic
block, a 17-feature
abstention-aware block, and a 70-feature rich semantic block. Its separate
[training ladder](experiments/quant_deterministic_news/v2/TRAINING_LADDER.md)
crosses both semantic contracts with FLAN-T5-XL and GPT-5.6 Sol, adding matched
Q+L and Q+D+L designs while excluding every other LLM. The D2 panel and
available-data deterministic ladder are now complete: D3 modestly improved
both T2 targets versus matched Q56, but no target passed the stale-news,
wrong-stock, fold-consistency, and paired-inference gate together. The shared
semantic corpus is also complete at 466,902 article-target assignments,
55,197 assigned articles, and 27,510 stock-days, including 204 stock-days with
no candidate. The FLAN-T5-XL W17 construction pipeline and its silver-only
acceptance lock are implemented; its full tokenizer preflight passed all
4,202,118 logical prompts with zero over-512 violations, and its one-record
CUDA float16 smoke completed on the RTX 3070 Ti with no failure or truncation.
The GPT-5.6 Sol R70 offline Batch, adjudication, and aggregation pipeline has a
final hash-bound full-corpus preflight covering 933,804 requests in 934
conservative files.
Neither pipeline has produced a complete article-inference corpus or daily
semantic panel, so no FLAN/GPT downstream semantic model was trained. GPT
remains an explicitly future-contaminated oracle design rather than OOS
evidence. No paid call has been made, and any pilot/full run remains gated by an API
credential, licensed-text confirmation, an explicit request budget, and
separate user authorization. The redesign separates direct,
peer-idiosyncratic, sector-common, and macro-common articles; removes raw
coverage/source proxies from the primary matrix; and keeps ordinary Massive,
retrospective Benzinga, and prospective versioned panels distinct. See the
[final v2 comparison](experiments/quant_deterministic_news/v2/training/comparisons/final/RESULTS.md).
The exact construction commands and fail-closed completion gates are in the
[semantic construction runbook](experiments/quant_deterministic_news/v2/training/semantic/CONSTRUCTION_RUNBOOK.md).

The separate full-text benchmark retrieved 399 documents in 554 attempts
covering 553 unique public URLs from a 5,180-document queue, then selected 300
assignment-balanced documents from two publishers, all published strictly
after 2026-03-01 00:00 UTC. Only 73 assigned targets are provider-tagged; 266
documents tag the target or a known peer, and 34 rely on sector/market
eligibility. Alpha Vantage matched only one selected document, so no Alpha
summary comparison is estimable. GPT silver labels use reconstructed parent
full text without provider ticker tags, whereas local models receive the
applicable text variant plus honest provider tags. Reported model results are
therefore agreement and information-retention tests, not objective accuracy,
strict historical point-in-time validity, or a symmetric model contest.

On the locked 228-document evaluation, chunked retrieved full text reduced
mean macro-F1 relative to Massive descriptions by 0.0908 for FLAN-T5-XL and
0.0593 for Llama 3.1; both paired 95% intervals were below zero. The current
recommendation is therefore to use free Massive ordinary News for exploratory
deterministic features, retain Alpaca for quant data, and buy neither Alpha
premium nor the `$99` Benzinga feed for the current extraction pipeline.
Strict point-in-time news training still requires a forward version-preserving
collector or a provider contract that supplies auditable historical versions.

See:

- [`docs/news_provider_experiment_results.md`](docs/news_provider_experiment_results.md)
  for the empirical access, coverage, full-text, and model-ablation results;
- [`docs/news_provider_source_matrix.md`](docs/news_provider_source_matrix.md)
  for documented versus observed provider capabilities; and
- [`docs/q_plus_d_massive.md`](docs/q_plus_d_massive.md) for the joined-panel
  contract and claim boundary.

## Reproducibility principles

1. Use only information available before each forecast cutoff.
2. Pin model and tokenizer revisions.
3. Freeze prompts, schemas, routing, precision, quantization, and runtimes.
4. Hash and cache every input and model output.
5. Keep semantic extraction separate from the downstream forecast model.
6. Never tune against the final evaluation split.
7. Keep licensed news, raw provider payloads, model weights, and credentials
  out of source control.
