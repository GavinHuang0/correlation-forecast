# FLAN-T5 Agreement Evaluation

This benchmark measures agreement between frozen `google/flan-t5-large` predictions and 300 GPT-5.6 Sol silver-reference annotations. It does not treat GPT-5.6 Sol as objective ground truth.

The null-output fix described below is the v0.2 baseline. The subsequent accuracy-improvement implementation, fixed-split results, and complete reproduction commands are documented in the active [`v0.4 README`](../../v0_4/README.md).

## Why the first run returned null labels

The archived `flan-stock-sector-news-v0.1.0` contract asked FLAN-T5-Large to emit three labels at once in a pipe-delimited record. The model did not reliably follow that grouped output contract:

- 0 of 300 records had all nine primary labels in valid form;
- none of the 900 grouped core responses contained a safely recoverable complete three-field tuple;
- scope responses frequently echoed the literal output template;
- event and direction responses frequently copied headlines or lists of allowed choices; and
- the separate one-field channel prompt remained strictly parseable for all 300 records.

This was not input truncation: the largest legacy prompt was below the 512-token model limit. It was also not an evaluator defect. The strict parser rejected ambiguous model prose rather than guessing labels from it.

## Corrected contract

New runs use `flan-stock-sector-news-v0.2.0`:

1. Each of the nine primary fields has its own prompt and concise label definitions.
2. Each prompt ends in a single `Answer:` cue.
3. Greedy decoding is constrained to the legal labels for that field.
4. The parser still accepts only a complete legal label; it does not infer a label from copied option lists or prose.
5. The legacy prompts and parsers remain available under `flan-stock-sector-news-v0.1.0`, so the original failed run can still be reproduced and evaluated exactly.

Constrained decoding guarantees schema-valid output, not semantic correctness. Agreement metrics must still determine whether FLAN-T5 chose the same meanings as the silver reference.

## Verified 300-record result

The corrected CUDA/FP16 core run was completed on July 23, 2026 using batch size 4 and the pinned checkpoint above.

| Contract | Records | Primary records valid | Strict format valid | Invalid field predictions |
|---|---:|---:|---:|---:|
| Legacy v0.1 grouped generation | 300 | 0% | 0% | 2,700 |
| v0.2 one-field constrained generation | 300 | 100% | 100% | 0 |

The null problem is therefore resolved as an output-contract problem. The semantic comparison still fails the project thresholds:

| Field | Accuracy | Macro-F1 | Cohen's kappa |
|---|---:|---:|---:|
| `relevance` | 0.033 | 0.039 | 0.002 |
| `event_scope` | 0.280 | 0.123 | 0.126 |
| `affected_breadth` | 0.247 | 0.138 | 0.042 |
| `event_type` | 0.310 | 0.219 | 0.227 |
| `information_status` | 0.463 | 0.538 | 0.250 |
| `explicit_surprise` | 0.083 | 0.105 | 0.035 |
| `target_direction` | 0.387 | 0.167 | 0.112 |
| `sector_direction` | 0.270 | 0.139 | 0.048 |
| `peer_effect` | 0.647 | 0.131 | -0.004 |

Mean field accuracy is 0.302 and mean macro-F1 is 0.177. The apparently high `peer_effect` accuracy is driven by predicting `not_applicable` for 299 of 300 articles, so macro-F1 and the confusion matrix are the more informative measures. The v0.2 conclusion was that every closed-label response had become measurable, but its fine-grained semantics were unusable.

The later v0.4 coarse hybrid raises evaluation-split mean macro-F1 from 0.341 to 0.446 on the same coarse task. That is a real improvement but remains below all four semantic go/no-go thresholds; FLAN-T5-Large is still a research baseline rather than the production extractor.

## Frozen model configuration

- Model: `google/flan-t5-large`
- Immutable Hugging Face revision: `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a`
- Prompt/parser contract: `flan-stock-sector-news-v0.2.0`
- Primary mode: nine independent closed-label passes
- Primary decoding: greedy constrained decoding, `do_sample=False`, `num_beams=1`
- Optional full mode: channels, two entity lists, and three exact-evidence spans
- Network access during extraction: disabled with `--local-files-only`

The extractor hashes the local weight, tokenizer, configuration, input, schema, rendered prompt corpus, and prompt-building code. Its manifest also records the Python, PyTorch, Transformers, hardware, device, precision, decoding mode, and generation settings.

The GPT-5.6 Sol labels are a silver development reference. The user explicitly allowed the interactive annotator to ignore model-cutoff concerns, so the labels may contain memorized-future contamination even though annotation instructions limited judgments to supplied text. A final scientific evaluation should use a genuinely post-cutoff sample and a reproducibly pinned reference process.

## Install and cache the model

Install the PyTorch build appropriate for the local CPU/GPU, then install the remaining dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-llm.txt
```

Download the pinned snapshot once:

```powershell
hf download google/flan-t5-large `
  config.json generation_config.json model.safetensors `
  special_tokens_map.json spiece.model tokenizer.json tokenizer_config.json `
  --revision 0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a
```

## Validate without loading the model

This validates all 300 records and constructs the 2,700 primary prompts without importing Transformers or loading the model:

```powershell
.\.venv\Scripts\python.exe scripts\extract_flan_t5.py `
  --input outputs\flan_t5\shared\benchmark_300\annotation_batches\all_inputs.jsonl `
  --output outputs\flan_t5\archive\v0_2\flan_t5_v0_2_core_predictions.jsonl `
  --revision 0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a `
  --prompt-version flan-stock-sector-news-v0.2.0 `
  --closed-label-decoding constrained `
  --mode core `
  --validate-only
```

## Run a 20-record smoke test

Use a new output path; do not overwrite the archived v0.1 predictions.

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv\Scripts\python.exe scripts\extract_flan_t5.py `
  --input outputs\flan_t5\shared\benchmark_300\annotation_batches\all_inputs.jsonl `
  --output outputs\flan_t5\archive\v0_2\flan_t5_v0_2_core_smoke20.jsonl `
  --revision 0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a `
  --prompt-version flan-stock-sector-news-v0.2.0 `
  --closed-label-decoding constrained `
  --mode core `
  --device cpu `
  --precision float32 `
  --batch-size 1 `
  --local-files-only `
  --limit 20 `
  --overwrite
```

Evaluate the subset with:

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_flan_agreement.py `
  --reference annotations\chatgpt_5_6_sol_reference.jsonl `
  --predictions outputs\flan_t5\archive\v0_2\flan_t5_v0_2_core_smoke20.jsonl `
  --inputs outputs\flan_t5\shared\benchmark_300\annotation_batches\all_inputs.jsonl `
  --output outputs\flan_t5\archive\v0_2\flan_t5_v0_2_core_smoke20_agreement.json `
  --allow-subset
```

Do not start the 300-record run unless the smoke report has a 100% primary valid-output rate. This gate tests the extraction contract, not semantic agreement.

## Run the 300-record primary extraction

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv\Scripts\python.exe scripts\extract_flan_t5.py `
  --input outputs\flan_t5\shared\benchmark_300\annotation_batches\all_inputs.jsonl `
  --output outputs\flan_t5\archive\v0_2\flan_t5_v0_2_core_predictions.jsonl `
  --revision 0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a `
  --prompt-version flan-stock-sector-news-v0.2.0 `
  --closed-label-decoding constrained `
  --mode core `
  --device cpu `
  --precision float32 `
  --batch-size 1 `
  --local-files-only `
  --overwrite
```

CPU/FP32 and CUDA/FP16 are distinct extractor configurations and must use different output paths. Keep device and precision constant across one run. Use `--resume` after an interruption; the script refuses to resume when the model files, prompts, schema, input selection, runtime, device, precision, batch size, or generation settings differ.

Legacy v0.1 artifacts remain evaluator-compatible, but an in-progress legacy extraction is not resumable with the refactored runner because the old manifest predates the decoder/parser implementation fingerprint. Finish legacy reproduction under the archived code environment or start a new v0.2 output instead of mixing contracts.

The primary run performs 2,700 field-level generations. `--mode full` performs 4,500 total generations and adds secondary open-field outputs. Full mode is not the default because entity-list generation remains much less reliable than closed-label classification.

`--closed-label-decoding score` is available as a documented ablation using complete-label mean token log probability. `--closed-label-decoding generate` is useful only for diagnosing unconstrained instruction following; it can reintroduce invalid formats.

## Calculate full-sample agreement

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_flan_agreement.py `
  --reference annotations\chatgpt_5_6_sol_reference.jsonl `
  --predictions outputs\flan_t5\archive\v0_2\flan_t5_v0_2_core_predictions.jsonl `
  --inputs outputs\flan_t5\shared\benchmark_300\annotation_batches\all_inputs.jsonl `
  --output outputs\flan_t5\archive\v0_2\flan_t5_v0_2_core_agreement.json
```

The report includes macro-F1, Cohen's kappa, per-class metrics, confusion matrices, and output-format validity. Channel, entity, and evidence metrics are reported only for a `full` extraction.
