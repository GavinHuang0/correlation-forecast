# FLAN-T5-Large v0.4 Accuracy-Improvement Experiment

## Outcome

The repository now implements the recommended FLAN-T5-Large improvements without changing the model:

1. reduce the ontology to four economically relevant semantic fields;
2. move relevance and explicit-surprise detection to auditable deterministic rules;
3. put the article before the task instructions;
4. ask one field-specific question at a time;
5. route questions hierarchically so inapplicable fields are not forced;
6. score closed choices directly instead of relying on free-form generation;
7. average scores across canonical and reversed option orders;
8. use a fixed 72-record development split for prompt/field selection; and
9. evaluate the frozen choice on the other 228 records.

The changes materially improved agreement, especially for `event_family` and `information_status`, but did not make FLAN-T5-Large accurate enough for production use.

## Frozen configuration

- Model: `google/flan-t5-large`
- Model revision: `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a`
- Device and precision: CUDA/FP16
- Coarse schema: `stock_sector_news_semantics_coarse` v0.2.0
- Pure prompt: `flan-stock-sector-news-v0.4.0`
- Hybrid prompt manifest: `flan-stock-sector-news-hybrid-v0.4.0`
- Prompt profile: zero-shot
- New decoding: order-averaged letter-label scoring
- Network access during extraction: disabled

The model, tokenizer, schema, inputs, deterministic-rule module, prompt-building source, output, environment, and decoding configuration are hashed in the run manifests.

## Implementation map

| File | Role |
|---|---|
| [`scripts/coarse_news_features.py`](../scripts/coarse_news_features.py) | Fine-to-coarse mapping, deterministic relevance gate, explicit-surprise rules, and stable stratified split |
| [`scripts/prepare_flan_coarse_benchmark.py`](../scripts/prepare_flan_coarse_benchmark.py) | Recreate the coarsened silver reference and 72/228 input split |
| [`scripts/extract_flan_t5_coarse.py`](../scripts/extract_flan_t5_coarse.py) | Run hierarchical FLAN inference and all supported decoding ablations |
| [`scripts/evaluate_flan_coarse_sanity.py`](../scripts/evaluate_flan_coarse_sanity.py) | Evaluate the checked-in synthetic sanity set and enforce its optional failure gate |
| [`scripts/evaluate_flan_coarse.py`](../scripts/evaluate_flan_coarse.py) | Compute gate, surprise, semantic, majority-baseline, and mapped-v0.2 comparisons |
| [`scripts/combine_flan_coarse_hybrid.py`](../scripts/combine_flan_coarse_hybrid.py) | Apply the field choice frozen on the 72-record development split |
| [`scripts/summarize_flan_coarse_report.py`](../scripts/summarize_flan_coarse_report.py) | Produce the compact checked-in result from the locked pure and hybrid reports |

Version numbers refer to different artifacts: the original fine schema is v0.1.0, the coarse schema is v0.2.0, the fixed split retains the internal `coarse_v0_3` filename/seed, and the final FLAN prompt contract is v0.4.0.

## Reduced semantic schema

The LLM now predicts only:

| Field | Labels |
|---|---|
| `shock_scope` | `idiosyncratic`, `common`, `mixed`, `unclear` |
| `event_family` | `earnings_guidance`, `product_demand`, `supply_capacity`, `regulation_legal`, `corporate_analyst`, `macro_market`, `other_or_unclear` |
| `information_status` | `confirmed`, `anticipated`, `rumor_or_opinion`, `unclear` |
| `directional_alignment` | `single_firm_only`, `same_direction`, `opposite_direction`, `common_direction_unclear`, `unclear` |

The complete definitions and deterministic mapping from the original fine labels are in [`config/news_feature_schema_coarse.json`](../config/news_feature_schema_coarse.json).

The non-LLM layer records:

- target, peer, sector, and macro text matches;
- vendor target and peer ticker tags;
- the relevance-gate route and provenance;
- explicit positive and negative surprise cues; and
- a conservative `positive`, `negative`, `mixed`, or `none` surprise label.

Short tickers use case-sensitive token boundaries, so a symbol such as `MU` does not match ordinary words such as “much.”

## Development decisions

The 300 GPT silver annotations were deterministically coarsened, then stratified by `shock_scope` into:

- development: 72 articles;
- evaluation: 228 articles.

The development experiments found:

- constrained label generation fixed formatting but did not solve semantic errors;
- order-averaged label scoring improved `event_family` and `information_status`;
- the older v0.2 FLAN predictions remained better for `shock_scope` and `directional_alignment`; and
- few-shot examples reduced the synthetic sanity score from 0.625 to 0.500, so the final profile remained zero-shot.

The field choice was frozen before evaluating the 228-record split:

| Field | Selected source |
|---|---|
| `shock_scope` | v0.2 FLAN `event_scope`, mapped to the coarse schema |
| `event_family` | v0.4 order-averaged FLAN prompt |
| `information_status` | v0.4 order-averaged FLAN prompt |
| `directional_alignment` | v0.2 FLAN scope/direction fields, mapped to the coarse schema |

This is still one model: both sources use the identical pinned FLAN-T5-Large checkpoint. The hybrid only retains the prompt/decision rule that performed better for each field on the development split.

## Evaluation result

The evaluation split contains 228 records, of which 174 are semantically relevant under the coarsened silver reference.

### Semantic fields

| Field | v0.2 accuracy | Hybrid accuracy | v0.2 macro-F1 | Hybrid macro-F1 |
|---|---:|---:|---:|---:|
| `shock_scope` | 0.511 | 0.506 | 0.381 | 0.372 |
| `event_family` | 0.259 | 0.483 | 0.224 | 0.452 |
| `information_status` | 0.460 | 0.529 | 0.429 | 0.623 |
| `directional_alignment` | 0.460 | 0.460 | 0.331 | 0.335 |
| **Mean** | **0.422** | **0.494** | **0.341** | **0.446** |

The mean macro-F1 gain is 0.105 absolute, or 30.6% relative to the mapped v0.2 baseline. Macro-F1 is the primary aggregate because the labels are imbalanced. The pure v0.4 run had higher mean accuracy, 0.529, but only 0.373 mean macro-F1 because `shock_scope` and `directional_alignment` moved toward majority labels.

The same caveat applies within fields: hybrid `information_status` accuracy, 0.529, is below the 0.557 majority-label accuracy even though its macro-F1 is much stronger than the majority macro-F1. The protocol therefore does not use raw accuracy as its sole selection criterion.

### Deterministic components

| Component | Accuracy | Precision | Recall | F1 or macro-F1 |
|---|---:|---:|---:|---:|
| Relevance gate | 0.895 | 0.883 | 0.994 | F1 0.935 |
| Explicit-surprise rule | 0.930 | macro 0.663 | macro 0.459 | macro-F1 0.523 |

The surprise rule is deliberately conservative: it has high precision but misses rare implicit or unusually phrased surprises.

### Go/no-go result

| Criterion | Threshold | Observed | Pass |
|---|---:|---:|---|
| Relevance-gate F1 | 0.90 | 0.935 | Yes |
| Scope macro-F1 | 0.70 | 0.372 | No |
| Event-family macro-F1 | 0.65 | 0.452 | No |
| Information-status macro-F1 | 0.70 | 0.623 | No |
| Directional-alignment macro-F1 | 0.65 | 0.335 | No |

The honest conclusion is that v0.4 is better than v0.2, but FLAN-T5-Large is not ready to supply the main forecasting features. It should remain a documented weak baseline until a later fine-tuning or model-comparison stage.

## Interpretation limits

These values measure agreement with a deterministically coarsened GPT-5.6 Sol silver reference, not accuracy against human ground truth. The original 300-record benchmark already informed protocol design, so even the fixed 228-record split is not a pristine external scientific test. A final claim requires a new post-protocol, human-audited benchmark.

The synthetic sanity set also failed its near-perfect acceptance gate. The best zero-shot order-averaged run achieved 0.625 all-field accuracy, with no input truncation. This rules out presenting the current extractor as reliable on even every obvious constructed case.

## Reproduction

### 1. Recreate the coarse reference and fixed split

```powershell
.\.venv\Scripts\python.exe scripts\prepare_flan_coarse_benchmark.py `
  --reference annotations\chatgpt_5_6_sol_reference.jsonl `
  --inputs outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\annotation_batches\all_inputs.jsonl `
  --coarse-reference-output annotations\chatgpt_5_6_sol_reference_coarse_v0_2.jsonl `
  --development-input-output outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\annotation_batches\coarse_v0_3_development_inputs.jsonl `
  --evaluation-input-output outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\annotation_batches\coarse_v0_3_evaluation_inputs.jsonl
```

### 2. Run the pure v0.4 evaluation extraction

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv\Scripts\python.exe scripts\extract_flan_t5_coarse.py `
  --input outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\annotation_batches\coarse_v0_3_evaluation_inputs.jsonl `
  --output outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\flan_t5_v0_4_evaluation_zero_order_averaged.jsonl `
  --schema config\news_feature_schema_coarse.json `
  --revision 0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a `
  --decoding order_averaged_letter_score `
  --prompt-profile zero_shot `
  --device cuda `
  --precision float16 `
  --batch-size 4 `
  --local-files-only `
  --overwrite
```

Use `--validate-only` first to validate all records and rendered prompts without loading the model.

### 3. Build the development-selected hybrid

```powershell
.\.venv\Scripts\python.exe scripts\combine_flan_coarse_hybrid.py `
  --coarse-predictions outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\flan_t5_v0_4_evaluation_zero_order_averaged.jsonl `
  --fine-v0-2-predictions outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\flan_t5_v0_2_core_predictions_cuda_fp16.jsonl `
  --schema config\news_feature_schema_coarse.json `
  --output outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\flan_t5_v0_4_evaluation_hybrid.jsonl `
  --overwrite
```

### 4. Evaluate and summarize

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_flan_coarse.py `
  --reference annotations\chatgpt_5_6_sol_reference.jsonl `
  --predictions outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\flan_t5_v0_4_evaluation_hybrid.jsonl `
  --inputs outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\annotation_batches\all_inputs.jsonl `
  --schema config\news_feature_schema_coarse.json `
  --old-fine-predictions outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\flan_t5_v0_2_core_predictions_cuda_fp16.jsonl `
  --split evaluation `
  --output outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\flan_t5_v0_4_evaluation_hybrid_agreement.json

.\.venv\Scripts\python.exe scripts\summarize_flan_coarse_report.py `
  --hybrid-report outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\flan_t5_v0_4_evaluation_hybrid_agreement.json `
  --pure-report outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\flan_t5_v0_4_evaluation_zero_order_averaged_agreement.json `
  --output reports\flan_t5_v0_4_evaluation_summary.json
```

All generated model outputs remain Git-ignored. The compact result and source-code/configuration needed to reproduce it are checked in.
