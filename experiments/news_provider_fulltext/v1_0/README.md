# News provider and full-text experiment v1.0

This experiment answers two related questions:

1. Can the available free provider credentials support the historical
   deterministic-news panel required by the correlation model?
2. On a fresh post-cutoff benchmark, do full article bodies improve coarse
   semantic extraction relative to provider descriptions? An Alpha-summary
   comparison is retained only if enough exact document matches exist.

The experiment is isolated from all existing FLAN-T5-XL and Llama 3.1
experiment inputs, outputs, registries, and active pointers. Raw news, licensed
text, model predictions, and generated reports remain under Git-ignored
directories.

The provider collection, deterministic-feature build, Q+D join, public-page
retrieval, GPT silver annotation, both local-model arms, reducer locks, and
evaluation reports are complete. The bounded Alpha diagnostic ended at the
free quota with two ticker slices pending, as documented below.

Key outcomes:

- Free Massive ordinary News yielded 90,290 unique provider documents after
  cross-root deduplication and supported a 27,510-row Q+D join with 40
  recommended deterministic fitting columns.
- The joined panel is exploratory and non-version-safe, not primary-training
  eligible.
- Alpha produced only one exact match in the selected benchmark, so an
  Alpha-summary provider comparison is not estimable.
- On the locked 228-document evaluation, chunked retrieved full text reduced
  mean macro-F1 relative to Massive descriptions by 0.0908 for FLAN-T5-XL and
  0.0593 for Llama 3.1. Both paired 95% bootstrap intervals were below zero.
- The result does not support buying the Massive/Benzinga full-text feed for
  the current frozen extractor. It does not rule out bodies for a different
  long-context or target-conditioned design.

Primary reports:

```text
outputs/news_provider_fulltext/v1_0/flan_t5_xl/evaluation_primary.json
outputs/news_provider_fulltext/v1_0/llama_3_1/evaluation_primary.json
docs/news_provider_experiment_results.md
```

## Frozen scope

- Price universe: the 30 stocks and five sectors in
  `config/price_universe.json`. Company names, aliases, peers, and deterministic
  news-target metadata come from `config/news_target_universe_30.json`.
- Massive ordinary-news collection: every cursor exposed at collection time
  for the configured 30 stock queries plus SOXX/XLF/XLE/XLV/XLI/SPY, from
  2016-06-22 through 2026-07-26, using at most five calls per minute. This is
  configured-query completeness, not proof of exhaustive market-wide news
  coverage or perfect provider tagging. The final UTC calendar day was still
  open, so “cursor-complete” is a snapshot claim rather than a claim of a
  complete July 26 calendar day.
- Alpha Vantage: a bounded 24-ticker diagnostic with one ticker per request;
  never a comma-separated union query. It produced 23 complete slices covering
  22 unique tickers, with UNP and BA pending after the free quota was
  exhausted. A resumable retry after the local calendar date changed was still
  quota-blocked; the redacted response audit is
  `alpha_free_quota_retry_2026-07-27.json`. This is not a multi-year Alpha
  fallback. That fallback was not started because free Massive had already
  satisfied the exploratory-volume criterion, while Alpha's documented
  summary schema would not repair the historical version-safety failure.
- Text benchmark: 300 distinct documents, ten per stock, strictly after
  2026-03-01. The balanced assignment is not a direct-target balance:
  73 documents have the assigned target in provider ticker metadata, 266 have
  the target or a known peer, and 34 rely only on expanded sector/market
  eligibility. The selected sources are 204 Motley Fool and 96 Benzinga pages.
- Reference: GPT-5.6 Sol silver annotations based on the supplied headline,
  reconstructed parent full text, and target metadata. GPT did not receive
  provider ticker tags.
- Local extractors: the repository's frozen FLAN-T5-XL v1.1 wrapper and pinned
  Llama 3.1 8B Instruct runtime. They receive the applicable text variant,
  target metadata, and only the ticker tags actually supplied by the provider.
- Metrics: agreement accuracy and macro-F1 per field, plus paired deltas on
  exact matched document sets. They are not human-ground-truth accuracy.

The March 1 cutoff is deliberately later than GPT-5.6 Sol's documented
February 16, 2026 knowledge cutoff. FLAN-T5-XL has no documented checkpoint
knowledge cutoff, so its 2022 release is treated only as a conservative
release boundary, not as an official training-data claim.

## Data products

```text
data/raw/massive/ordinary_news/free_v1/
  pages/<TICKER>/*.json.gz
  state.json
  manifest.json

data/raw/massive/ordinary_news/free_v1_benchmarks/
  pages/<SOXX|XLF|XLE|XLV|XLI|SPY>/*.json.gz
  state.json
  manifest.json

data/raw/alpha_vantage_news/postcutoff_overlap_v1/
  responses/<TICKER>/*.json.gz
  state.json
  manifest.json

data/external/news_provider_comparison/v1_0/
  fulltext_candidates.jsonl
  fulltext/
  provider_overlap_final.json

data/features/news_deterministic/massive_v1/
  normalized_articles.csv.gz
  article_target_features.csv.gz
  stock_day_features.csv.gz
  coverage_audit.csv
  manifest.json

data/features/q_plus_d/massive_v1/
  modeling_panel_q_plus_d.parquet
  manifest.json
  manifest.sha256

data/benchmarks/news_text_ablation_300/v1/
  fixed_assignment_source_manifest.json
  massive_description.jsonl
  alpha_summary.jsonl
  fulltext_evidence_chunks.jsonl
  manifest.json

experiments/news_provider_fulltext/v1_0/
  access_probe_summary.json
  massive_rest_aggregate_probe.json

reports/generated/
  massive_flat_minute_object_validation_2026-07-26.json
```

The generated `data/` paths are ignored because they can contain
provider-licensed text. The tracked probe summaries contain no credentials or
article text.

### As-built integrity ledger

These checksums identify the completed provider/data phase after the benchmark
was rebuilt on 2026-07-27. They are not entitlement guarantees.

| Artifact | SHA-256 |
|---|---|
| `data/raw/massive/ordinary_news/free_v1/manifest.json` | `8419ddc03ec68861d818745393dcaf102f919f5086aa1d0f7f9291c253a2db63` |
| `data/raw/massive/ordinary_news/free_v1_benchmarks/manifest.json` | `99538f46727661c6181e98615979ee151768da14978d10d6ca34dfba3b20c652` |
| `data/raw/alpha_vantage_news/postcutoff_overlap_v1/manifest.json` | `838c0e92a7f727295e9c704b264bd82b6e34ed0ade374304df840d1275c20a6e` |
| `data/features/news_deterministic/massive_v1/manifest.json` | `01f3d16df1ed7caf9ba582ad489b366e17aed89df1689e626ae22a0e32fa1a52` |
| `data/features/q_plus_d/massive_v1/modeling_panel_q_plus_d.parquet` | `352ba2635febfa0bc36bc6f797c00036a8aee9879327a8176e9accf180e50150` |
| `data/features/q_plus_d/massive_v1/manifest.json` | `ac161d48901fae669fdc500099df92d218513d2e7d8d383539b8ed326445d86f` |
| `data/external/news_provider_comparison/v1_0/fulltext/manifest.jsonl` | `62a30bf70767d713616a1111f67d279b692d64ceb70ec358c3e8fd297f3261fe` |
| `data/benchmarks/news_text_ablation_300/v1/manifest.json` | `dc3b8fcefe3c48effff901d3b1cfd2b461f1bd4db0b44f1bcc9ea2e367145857` |
| `annotations/news_provider_fulltext/v1_0/gpt_5_6_sol_silver.jsonl` | `6536f25228706ce5d70c9556a266bdae74f21be613573c3fc2a2ba5a8bd0fb70` |

The benchmark manifest additionally binds the byte-identical model inputs:

```text
Massive descriptions:
24dc9de18dc6123ad26ad5fd21b4e46175b695a220fbd85a089d9394c74ce2a4

Alpha exact-match audit row:
998f052fb77b178f10c3a4074140d673ee2a8727d564780b84e3a9ede0a1036f

Full-text evidence chunks:
d8ca8d83e78686383d01246dd341128df9bdc978abf948559e214b1a13e940b9
```

## Reproducible commands

```powershell
$trainingPython = '.\.venv-training\Scripts\python.exe'

& $trainingPython scripts/fetch_massive_news.py `
  --start 2016-06-22 --end 2026-07-26 `
  --max-requests 10000 --calls-per-minute 5 `
  --output-dir data/raw/massive/ordinary_news/free_v1

& $trainingPython scripts/fetch_massive_news.py `
  --tickers SOXX,XLF,XLE,XLV,XLI,SPY `
  --start 2016-06-22 --end 2026-07-26 `
  --max-requests 10000 --calls-per-minute 5 `
  --output-dir data/raw/massive/ordinary_news/free_v1_benchmarks

& $trainingPython scripts/probe_massive_flatfiles.py `
  --prefix us_stocks_sip/minute_aggs_v1/2026/07/ `
  --current-minute-object us_stocks_sip/minute_aggs_v1/2026/07/2026-07-21.csv.gz `
  --historical-minute-object us_stocks_sip/minute_aggs_v1/2016/01/2016-01-04.csv.gz `
  --sample-bytes 65536 `
  --output reports/generated/massive_flat_minute_object_validation_2026-07-26.json

& $trainingPython scripts/fetch_alpha_vantage_news.py `
  --tickers AMD,NVDA,INTC,MU,AVGO,JPM,BAC,WFC,C,GS,XOM,CVX,COP,EOG,SLB,JNJ,MRK,PFE,ABBV,AMGN,CAT,DE,UNP,BA `
  --start 2026-03-01 --end 2026-03-07 `
  --slice-days 7 --max-calls 25 --calls-per-minute 5 `
  --output-dir data/raw/alpha_vantage_news/postcutoff_overlap_v1

& $trainingPython scripts/build_massive_deterministic_news_features.py `
  --raw-root data/raw/massive/ordinary_news/free_v1 `
  --raw-root data/raw/massive/ordinary_news/free_v1_benchmarks `
  --overwrite

& $trainingPython scripts/join_quant_deterministic_news.py --overwrite

& $trainingPython scripts/build_fulltext_candidate_queue.py `
  --input-root data/raw/massive/ordinary_news/free_v1 `
  --input-root data/raw/massive/ordinary_news/free_v1_benchmarks

& $trainingPython scripts/retrieve_article_fulltext.py `
  data/external/news_provider_comparison/v1_0/fulltext_candidates.jsonl `
  --max-documents 600

& $trainingPython scripts/build_news_text_ablation_benchmark.py `
  --massive data/raw/massive/ordinary_news/free_v1 `
  --massive data/external/news_provider_comparison/v1_0/fulltext_candidates.jsonl `
  --alpha data/raw/alpha_vantage_news/postcutoff_overlap_v1 `
  --fulltext data/external/news_provider_comparison/v1_0/fulltext `
  --max-chunk-characters 850 `
  --fixed-assignment-manifest data/benchmarks/news_text_ablation_300/v1/fixed_assignment_source_manifest.json `
  --overwrite
```

The candidate queue is deliberately supplied as an additional `--massive`
input when the benchmark is built. The raw stock-query pages remain the
canonical source for Massive descriptions, while the deduplicated queue adds
auditable `eligible_target_tickers` metadata. Those targets include direct
stock queries/tags, same-sector peer documents, all sector stocks for
SOXX/XLF/XLE/XLV/XLI query results, and all 30 targets for SPY query results.
The benchmark builder merges the duplicate records by provider ID or URL and
unions the queue's eligible targets; omitting the queue would discard these
economically valid peer, sector, and market assignments.

Assignment eligibility and provider metadata are deliberately separate.
`eligible_target_tickers` is used only to balance articles across target
stocks. The local-model input's `vendor_tickers` contains only tickers actually
supplied in provider fields. Expanded peer/sector/market eligibility must never
be presented to an extractor as if the provider had tagged the assigned
target. This contract is recorded in benchmark manifest v1.1 and covered by a
regression test. In the selected benchmark, 73/300 assigned targets are
provider-tagged, 266/300 documents tag the assigned target or at least one known
peer, and the remaining 34 are sector/market-eligibility assignments.

Benchmark construction also applies a conservative quality gate to matched
retrieved bodies before balanced assignment. It rejects bodies clearly
dominated by a non-Latin script (for example, a `zh-hans` page discovered from
an English provider description) and bodies with severe control-character,
replacement-character, or mojibake corruption. Isolated mojibake is retained.
The raw retrieval files are never edited: the benchmark manifest records
exclusion counts, hashed benchmark article IDs, reasons, and character-level
diagnostics under `fulltext_quality_gate`.

All network collectors are resumable. Their manifests contain hashes and
request scope but never credential values. The Alpha command deliberately
freezes the actual 24-ticker bounded diagnostic scope; using the repository's
30-stock default against that same output root would be rejected as a scope
change. A complete 30-stock Alpha collection must use a new versioned root.
Only one selected benchmark document has an exact Alpha-summary match. That is
an audit record, not a sample from which provider-level agreement can be
estimated.

## Claim limits

- Live access beyond a provider's documented entitlement is an observation,
  not a contractual guarantee.
- GPT labels are a silver reference, not human ground truth.
- Full-text retrieval is limited to public pages allowed by `robots.txt`; the
  workflow does not bypass paywalls, login requirements, or access controls.
- The 72.02% per-attempt retrieval rate is 399 successful documents among 554
  attempts covering 553 unique URLs, not coverage of the 5,180-document
  candidate queue. One URL was retried after a timeout. Retrieval stopped after
  a balanced benchmark became feasible, and the resulting two-publisher sample
  is selection-biased.
- Provider comparisons use exact matched documents. Unmatched Alpha rows are
  excluded from paired metric deltas; with one selected Alpha match, no
  Alpha-summary provider comparison is estimable.
- The Alpha manifest remains `in_progress` by design because its frozen
  24-ticker diagnostic exhausted the free quota with 23 complete slices and
  two pending ticker slices. It is a maximum-use overlap diagnostic, not a
  claim of full-universe or multi-year Alpha completeness.
- GPT silver labels were generated from reconstructed parent full text, while
  the local description arm sees only the short description. The full-text arm
  therefore has an information advantage relative to the reference. Local
  inputs also include honest provider ticker tags that GPT did not receive.
  Results measure information retention and agreement with this silver
  reference, not objective extractor accuracy or a symmetric model contest.

Final access counts, full-text success rates, paired model metrics, and the
provider recommendation are recorded in
`docs/news_provider_experiment_results.md`.

## Local-model compatibility and exact commands

The benchmark rows use the existing local-runner input contract:
`row_number`, unique `article_id`, `time_published_utc`, `source`, `headline`,
`article_text`, `vendor_tickers`, and complete `target` metadata. Full-text
rows use unique chunk IDs; their parent mapping remains in the benchmark
manifest for deterministic score aggregation.

The frozen runners refuse input truncation. Their ordinary `--validate-only`
paths validate schemas and prompts but do not tokenize them. The original base
interpreter behind both preserved Python 3.11 environments is no longer
available, so the broken virtual-environment launchers must not be invoked
directly. The ignored, checksum-recorded embedded Python recovery runtime loads
each environment's preserved `site-packages` without modifying that
environment or an active experiment pointer.

Run the tokenizer-only preflight through the recovery launcher:

```powershell
$py311 = 'data\runtime\python311-embed\base\python.exe'
$launcher = 'scripts\run_with_isolated_python311.py'

& $py311 $launcher `
  --site-packages .venv-flan-t5-xl\Lib\site-packages `
  --script scripts\preflight_news_text_ablation_inputs.py -- `
  --runner flan-t5-xl `
  --input data\benchmarks\news_text_ablation_300\v1\massive_description.jsonl `
  --input data\benchmarks\news_text_ablation_300\v1\alpha_summary.jsonl `
  --input data\benchmarks\news_text_ablation_300\v1\fulltext_evidence_chunks.jsonl `
  --output outputs\news_provider_fulltext\v1_0\flan_context_preflight.json

& $py311 $launcher `
  --site-packages .venv-llama31\Lib\site-packages `
  --script scripts\preflight_news_text_ablation_inputs.py -- `
  --runner llama-3.1 `
  --input data\benchmarks\news_text_ablation_300\v1\massive_description.jsonl `
  --input data\benchmarks\news_text_ablation_300\v1\alpha_summary.jsonl `
  --input data\benchmarks\news_text_ablation_300\v1\fulltext_evidence_chunks.jsonl `
  --output outputs\news_provider_fulltext\v1_0\llama_context_preflight.json
```

Both reports must have `passes_context_limit: true`. The preflight loads only
hash-verified tokenizer files and explicitly records that no model weights or
inference were used.

For FLAN-T5-XL, do not run `run_flan_t5_xl_active.py` through the recovery
launcher: that wrapper starts child Python processes without the recovered
`site-packages` path. Instead, run the frozen raw extractor and the frozen v1.1
calibration as two explicit isolated steps for every variant:

```powershell
$variants = @(
  'massive_description',
  'fulltext_evidence_chunks'
)

foreach ($variant in $variants) {
  $input = "data\benchmarks\news_text_ablation_300\v1\$variant.jsonl"
  $raw = "outputs\news_provider_fulltext\v1_0\flan_t5_xl\$variant.raw.jsonl"
  $output = "outputs\news_provider_fulltext\v1_0\flan_t5_xl\$variant.jsonl"

  & $py311 $launcher `
    --site-packages .venv-flan-t5-xl\Lib\site-packages `
    --script scripts\run_flan_t5_xl_coarse.py -- `
    --input $input --output $raw `
    --schema config\news_feature_schema_coarse.json `
    --decoding order_averaged_letter_score `
    --prompt-profile zero_shot `
    --device cuda --precision float16 --batch-size 32

  & $py311 $launcher `
    --site-packages .venv-flan-t5-xl\Lib\site-packages `
    --script scripts\experiment_flan_t5_xl_v1_1.py -- `
    apply --predictions $raw `
    --schema config\news_feature_schema_coarse.json `
    --config experiments\flan_t5_xl\v1_1\calibration.json `
    --output $output
}
```

`alpha_summary.jsonl` contains one matched row and is retained only as an audit
artifact. Running either model on that single row would not create an
estimable provider comparison, so it is excluded from the main loops and
metrics.

Llama 3.1's direct frozen runner can be invoked through the recovery launcher:

```powershell
foreach ($variant in $variants) {
  $input = "data\benchmarks\news_text_ablation_300\v1\$variant.jsonl"
  $output = "outputs\news_provider_fulltext\v1_0\llama_3_1\$variant.jsonl"

  & $py311 $launcher `
    --site-packages .venv-llama31\Lib\site-packages `
    --script scripts\extract_llama_3_1_coarse.py -- `
    --input $input --output $output
}
```

The ablation evaluator recognizes the active FLAN calibration-adjusted scores
and Llama 3.1 rotation-averaged scores. Full-text chunks support four
article-level reducers: `mean_score`, `max_score`, `best_margin`, and
`plurality`. Select the reducer using only the 72-document development split,
then freeze it before evaluating the 228-document split. The all-300 report is
descriptive only.

```powershell
$trainingPython = '.\.venv-training\Scripts\python.exe'
$reference = 'annotations\news_provider_fulltext\v1_0\gpt_5_6_sol_silver.jsonl'

foreach ($modelDirectory in @('flan_t5_xl', 'llama_3_1')) {
  $predictionRoot = "outputs\news_provider_fulltext\v1_0\$modelDirectory"

  foreach ($aggregation in @(
    'mean_score',
    'max_score',
    'best_margin',
    'plurality'
  )) {
    & $trainingPython scripts\evaluate_news_text_ablation.py `
      --reference $reference `
      --predictions "massive_description=$predictionRoot\massive_description.jsonl" `
      --predictions "fulltext_evidence_chunks=$predictionRoot\fulltext_evidence_chunks.jsonl" `
      --split development `
      --chunk-aggregation $aggregation `
      --output "$predictionRoot\evaluation_development_$aggregation.json"
  }
}
```

Create a hash-bound lock from the four development reports. The lock script
applies the preregistered lexicographic rule and refuses stale benchmark
inputs, prediction outputs, sidecars, schema, extractor identity, or report
hashes. Do not inspect evaluation results before this step.

```powershell
$trainingPython = '.\.venv-training\Scripts\python.exe'
$reference = 'annotations\news_provider_fulltext\v1_0\gpt_5_6_sol_silver.jsonl'
$modelDirectory = 'flan_t5_xl' # Repeat for llama_3_1.
$modelName = 'google/flan-t5-xl' # Use meta-llama/Llama-3.1-8B-Instruct for llama_3_1.
$predictionRoot = "outputs\news_provider_fulltext\v1_0\$modelDirectory"
$lock = "$predictionRoot\aggregation_lock.json"

& $trainingPython scripts\lock_news_text_ablation_aggregation.py `
  --model-name $modelName `
  --reference $reference `
  --prediction "massive_description=$predictionRoot\massive_description.jsonl" `
  --prediction "fulltext_evidence_chunks=$predictionRoot\fulltext_evidence_chunks.jsonl" `
  --development-report "mean_score=$predictionRoot\evaluation_development_mean_score.json" `
  --development-report "max_score=$predictionRoot\evaluation_development_max_score.json" `
  --development-report "best_margin=$predictionRoot\evaluation_development_best_margin.json" `
  --development-report "plurality=$predictionRoot\evaluation_development_plurality.json" `
  --output $lock

$selectedAggregation = (
  Get-Content -LiteralPath $lock -Raw | ConvertFrom-Json
).selected_chunk_aggregation

& $trainingPython scripts\evaluate_news_text_ablation.py `
  --reference $reference `
  --predictions "massive_description=$predictionRoot\massive_description.jsonl" `
  --predictions "fulltext_evidence_chunks=$predictionRoot\fulltext_evidence_chunks.jsonl" `
  --split evaluation `
  --chunk-aggregation $selectedAggregation `
  --aggregation-lock $lock `
  --output "$predictionRoot\evaluation_primary.json"

& $trainingPython scripts\evaluate_news_text_ablation.py `
  --reference $reference `
  --predictions "massive_description=$predictionRoot\massive_description.jsonl" `
  --predictions "fulltext_evidence_chunks=$predictionRoot\fulltext_evidence_chunks.jsonl" `
  --split evaluation `
  --chunk-aggregation $selectedAggregation `
  --aggregation-lock $lock `
  --reference-filter non_abstained `
  --output "$predictionRoot\evaluation_non_abstained.json"

& $trainingPython scripts\evaluate_news_text_ablation.py `
  --reference $reference `
  --predictions "massive_description=$predictionRoot\massive_description.jsonl" `
  --predictions "fulltext_evidence_chunks=$predictionRoot\fulltext_evidence_chunks.jsonl" `
  --split evaluation `
  --chunk-aggregation $selectedAggregation `
  --aggregation-lock $lock `
  --source-filter Benzinga `
  --output "$predictionRoot\evaluation_benzinga.json"

& $trainingPython scripts\evaluate_news_text_ablation.py `
  --reference $reference `
  --predictions "massive_description=$predictionRoot\massive_description.jsonl" `
  --predictions "fulltext_evidence_chunks=$predictionRoot\fulltext_evidence_chunks.jsonl" `
  --split evaluation `
  --chunk-aggregation $selectedAggregation `
  --aggregation-lock $lock `
  --source-filter Benzinga `
  --reference-filter non_abstained `
  --output "$predictionRoot\evaluation_benzinga_non_abstained.json"

foreach ($basis in @(
  'direct_target_tag',
  'peer_only_tag',
  'expanded_sector_or_market_only'
)) {
  & $trainingPython scripts\evaluate_news_text_ablation.py `
    --reference $reference `
    --predictions "massive_description=$predictionRoot\massive_description.jsonl" `
    --predictions "fulltext_evidence_chunks=$predictionRoot\fulltext_evidence_chunks.jsonl" `
    --split evaluation `
    --chunk-aggregation $selectedAggregation `
    --aggregation-lock $lock `
    --assignment-filter $basis `
    --output "$predictionRoot\evaluation_assignment_$basis.json"
}
```
