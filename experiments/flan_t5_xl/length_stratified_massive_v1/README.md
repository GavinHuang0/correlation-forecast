# FLAN-T5-XL length-stratified Massive validation

## Outcome

The frozen `flan-t5-xl-v1.1` extractor completed the benchmark without a
runtime or truncation failure:

- 200 unique Massive provider articles;
- 40 articles in each of five source-description-length strata;
- 8 articles per sector inside every stratum;
- 360 total model views after adding a paired headline-only ablation to every
  nonempty-description article;
- 360/360 untruncated inputs;
- 360/360 semantically applicable inputs with every frozen-schema field
  resolved.

This is an **operational and sensitivity validation**, not an accuracy
evaluation. The sampled articles do not have independent reference labels.
Agreement between the native and headline-only views measures stability, not
correctness.

## Sample contract

The source is the cached Massive ordinary-news snapshot completed on
2026-07-26. Selection is restricted to records published on or after
2022-11-01, uses one provider-tagged in-universe target per distinct provider
article, and is balanced across:

| Native source-description stratum | Articles |
|---|---:|
| Headline only | 40 |
| 1–149 characters | 40 |
| 150–299 characters | 40 |
| 300–599 characters | 40 |
| 600+ characters | 40 |

Stratification uses the original provider-description length. The extractor
receives the repository's explicit deployment-time lede excerpt, capped at
512 Unicode code points with an audited omitted-character count. There is no
tokenizer-side truncation. The largest observed frozen prompt was 376 tokens,
below the 512-token limit.

The native strata are not source- or date-matched. In this sample, all 40
headline-only records are from Investing.com, while 38 of 40 records in the
600+ stratum are from Benzinga. Cross-stratum differences therefore cannot be
interpreted as causal length effects. The paired ablation is the primary
length-sensitivity result because it holds the article, publisher, target, and
date fixed.

## Requested low-text focus

### Native headline-only articles

| Field | Largest calibrated label share | Raw choice-order agreement | Raw abstention | Calibrated abstention |
|---|---:|---:|---:|---:|
| Shock scope | 82.5% `idiosyncratic` | 50.0% | 2.5% | 10.0% |
| Event family | 80.0% `macro_market` | 85.0% | 0.0% | 0.0% |
| Information status | 62.5% `anticipated` | 35.0% | 65.0% | 0.0% |
| Directional alignment | 97.5% `single_firm_only` | n/a—mostly derived | 2.5% | 2.5% |

The concentrated event/scope outputs and weak order agreement for scope and
status are warning signs. Most importantly, the frozen v1.1 calibration
overrode the raw model's `information_status=unclear` behavior in 65% of these
records and left no calibrated status abstentions.

### Native descriptions below 150 characters

| Field | Largest calibrated label share | Raw choice-order agreement | Raw abstention | Calibrated abstention |
|---|---:|---:|---:|---:|
| Shock scope | 75.0% `idiosyncratic` | 50.0% | 10.0% | 22.5% |
| Event family | 62.5% `macro_market` | 75.0% | 2.5% | 2.5% |
| Information status | 45.0% `anticipated` | 62.5% | 72.5% | 0.0% |
| Directional alignment | 87.5% `single_firm_only` | 100.0% when modeled | 10.0% | 12.5% |

Again, calibration removed every raw information-status abstention. It changed
the status label on 75% of the sub-150 records.

### Sub-150 paired ablation

Each short-description article was run both with its native description and
with the description removed:

| Field | Raw label agreement | Calibrated label agreement |
|---|---:|---:|
| Shock scope | 85.0% | 77.5% |
| Event family | 85.0% | 85.0% |
| Information status | 85.0% | 72.5% |
| Directional alignment | 85.0% | 85.0% |

Only 47.5% of the 40 pairs retained the same calibrated label on all four
fields. Across all 160 paired articles, all-field calibrated agreement was
41.875%; fieldwise agreement was 73.125% for scope, 86.25% for event family,
68.125% for information status, and 81.25% for directional alignment.

The short descriptions therefore contain information that changes the
extractor output, but this unlabeled experiment cannot determine whether the
change is an improvement.

## Decision

The active extractor remains operationally valid for exploratory use, but this
experiment does **not** validate headline-only or sub-150 semantic accuracy.
For forecast-model construction:

1. Preserve `headline_only`, original description length, bounded-description
   status, and semantic-field coverage as explicit features.
2. Do not treat candidate-score margins as correctness probabilities.
3. Do not use calibrated `information_status` from headline-only or sub-150
   inputs as a primary feature. Mark it unavailable, or retain a separate raw
   abstention-aware sensitivity arm, until a length-stratified reference set is
   labeled.
4. Treat headline-only semantic features as an exploratory ablation because of
   label concentration and order sensitivity.
5. Obtain independent labels within the headline-only and sub-150 strata
   before claiming accuracy or promoting these strata into a primary
   forecasting specification.

No extractor selection, prompt, calibration rule, or training feature contract
was changed from these unlabeled results. The active pointer's implementation
hashes were refreshed only because the two XL entry points now explicitly add
their sibling `scripts/` directory under the repository's isolated Windows
Python runtime. That portability fix does not change model semantics.

## Reproduction

```powershell
.\.venv-training\Scripts\python.exe `
  scripts\build_massive_length_stratified_benchmark.py --overwrite

.\.venv-flan-t5-xl\Scripts\python.exe `
  scripts\run_flan_t5_xl_active.py `
  --input data\benchmarks\massive_length_stratified_current\v1\model_inputs.jsonl `
  --output outputs\flan_t5_xl\length_stratified_massive_v1\predictions.jsonl `
  --raw-output outputs\flan_t5_xl\length_stratified_massive_v1\predictions.raw.jsonl

.\.venv-training\Scripts\python.exe `
  scripts\analyze_flan_length_stratified.py `
  --inputs data\benchmarks\massive_length_stratified_current\v1\model_inputs.jsonl `
  --predictions outputs\flan_t5_xl\length_stratified_massive_v1\predictions.jsonl `
  --benchmark-manifest data\benchmarks\massive_length_stratified_current\v1\manifest.json `
  --output outputs\flan_t5_xl\length_stratified_massive_v1\analysis.json `
  --overwrite
```

The provider text, model inputs, and predictions remain under ignored
`data/` and `outputs/` paths. The tracked summary contains no article text.
