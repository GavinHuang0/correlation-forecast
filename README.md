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

Only two extractor paths are current:

| Extractor | Status | Protocol/results |
|---|---|---|
| FLAN-T5-Large v0.4 | Active research baseline; usable for exploratory comparisons, below production thresholds | [`experiments/flan_t5/v0_4/README.md`](experiments/flan_t5/v0_4/README.md) |
| Llama 2 7B Chat v1.2 | Development-selected research candidate; useful as an event-family specialist, not an all-field replacement | [`experiments/llama_2/README.md`](experiments/llama_2/README.md) |

A separate, not-yet-evaluated FLAN-T5-XL comparison is prepared under
[`experiments/flan_t5_xl/README.md`](experiments/flan_t5_xl/README.md). It
reuses the fixed development benchmark and v0.4 protocol without modifying the
active FLAN-T5-Large implementation.

Machine-readable contracts:

- [`experiments/flan_t5/active_baseline.json`](experiments/flan_t5/active_baseline.json)
- [`experiments/flan_t5/registry.json`](experiments/flan_t5/registry.json)
- [`experiments/llama_2/protocol.json`](experiments/llama_2/protocol.json)
- [`experiments/llama_2/benchmark_manifest.json`](experiments/llama_2/benchmark_manifest.json)
- [`experiments/llama_2/research_candidate.json`](experiments/llama_2/research_candidate.json)

The previous FLAN experiments, rejected v0.5 candidate, historical scripts,
tests, and reorganization records are kept separately in
[`experiments/flan_t5/archive/`](experiments/flan_t5/archive/README.md).

Generated local data follows the same separation:

```text
outputs/
  flan_t5/
    shared/       # fixed 300-article benchmark
    v0_4/         # active FLAN predictions/results
    archive/      # prior and rejected FLAN outputs
  llama_2/
    shared/       # prepared byte-identical 300-article benchmark
    v1_0/         # frozen failed baseline and diagnostics
    v1_1/         # corrected answer-boundary decoder
    v1_2/         # development-selected fieldwise hybrid
```

`outputs/`, licensed/raw news, model weights, and secrets remain Git-ignored.

## Benchmark and labels

The fixed benchmark contains 300 Alpha Vantage headline/summary records from
2024:

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

The selected v0.4 hybrid achieved on the fixed 228-record split:

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
majority-dominated. FLAN v0.4 remains the better all-field semantic baseline.
A fresh external holdout is required before promoting Llama v1.2 beyond a
research candidate.

See the
[`Llama 2 runbook`](experiments/llama_2/README.md)
for the exact runs, evaluation, diagnosis, and completed decoder experiment.

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
3. Establish lagged-correlation, HAR, exponential, and regularized
   quantitative baselines. DCC-GARCH remains planned and is not implemented.
4. Add frozen point-in-time news features.
5. Keep every stock observed on the same date in the same chronological fold.
6. Compare against news counts and conventional sentiment.
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
`data/features/quant/`.

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

## Training readiness

The current repository is **feature-ready, not end-to-end training-ready**:

- all 22 non-factor Bollerslev-Li-Tang columns are materialized and complete
  for the matched 2022-11-01 through 2026-06-30 panel;
- a dense block of additional quant features is also materialized;
- extended-hours features need an explicit missingness contract;
- the lagged realized-volatility extras need a first-bar correction; and
- target construction, joined modeling-table assembly, chronological splits,
  estimators, and forecast evaluation are not yet implemented.

The audited readiness matrix, explicit list of missing items, target ladder,
and model/feature ladder are in
[`docs/training_readiness.md`](docs/training_readiness.md).

## Reproducibility principles

1. Use only information available before each forecast cutoff.
2. Pin model and tokenizer revisions.
3. Freeze prompts, schemas, routing, precision, quantization, and runtimes.
4. Hash and cache every input and model output.
5. Keep semantic extraction separate from the downstream forecast model.
6. Never tune against the final evaluation split.
7. Keep licensed news, raw provider payloads, model weights, and credentials
   out of source control.
