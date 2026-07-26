# FLAN-T5-XL v1.1 Active Research Extractor

## Outcome

v1.1 is the strongest extractor tested in this repository, but it remains
below the predefined field-level acceptance thresholds. It is frozen for
exploratory forecasting experiments and model comparisons; it is not approved
as a production labeler.

On the locked 228-document evaluation split:

| Field | Accuracy | Macro-F1 | Required macro-F1 |
|---|---:|---:|---:|
| Shock scope | 0.649 | 0.468 | 0.70 |
| Event family | 0.511 | 0.478 | 0.65 |
| Information status | 0.615 | 0.580 | 0.70 |
| Directional alignment | 0.672 | 0.328 | 0.65 |
| **Mean** | **0.612** | **0.464** | — |

The deterministic relevance gate achieved F1 0.935. The semantic metrics use
174 reference-relevant articles and end-to-end routing, including the single
relevant article missed by the gate.

## What changed from raw XL

The model was not retrained and the prompt was not changed. v1.1 uses the
candidate score maps already emitted by raw XL v1.0. On the fixed
72-document development split it tested 27 bounded rules per field:

```text
score source: averaged, canonical, or reversed option order
log-prior strength: -1.00 through +1.00 in increments of 0.25
```

Selection maximized field macro-F1, followed by accuracy and Cohen's kappa.
The resulting configuration was serialized and hashed before the evaluation
split was run.

| Field | Frozen score source | Frozen strength |
|---|---|---:|
| Shock scope | averaged | 0.25 |
| Event family | canonical | 0.75 |
| Information status | canonical | -1.00 |
| Directional alignment | canonical | -0.75 |

The exact priors, candidate rankings, source hashes, and tie-break rules are in
[`calibration.json`](calibration.json).

## Locked comparison

| Extractor | Mean accuracy | Mean macro-F1 |
|---|---:|---:|
| FLAN-T5-Large v0.4 | 0.494 | 0.446 |
| Raw FLAN-T5-XL v1.0 | 0.546 | 0.407 |
| **Calibrated FLAN-T5-XL v1.1** | **0.612** | **0.464** |

Against raw XL, calibration added 0.066 mean accuracy and 0.056 mean
macro-F1. Against the previous active FLAN-T5-Large hybrid, it added 0.118
accuracy and 0.018 macro-F1. Macro-F1 is the primary selection measure because
raw accuracy overstates models that collapse to common classes.

The improvement is real on this benchmark but modest relative to the
threshold gap. In particular, directional alignment remains poorly balanced.
Do not turn the 0.672 raw accuracy for that field into a reliability claim:
its macro-F1 is only 0.328.

## Frozen runtime

```text
model:             google/flan-t5-xl
revision:          7d6315df2c2fb742f0f5b556879d730926ca9001
device/precision:  CUDA / FP16
quantization:      none
raw decoder:       canonical/reversed order-averaged closed-label scoring
postprocessor:     frozen fieldwise source/log-prior adjustment
prompt profile:    zero-shot
record batch:      1
candidate batch:   1
network:           disabled
```

The active machine-readable pointer is
[`../../active_extractor.json`](../../active_extractor.json). The compact
result is [`evaluation_summary.json`](evaluation_summary.json).

## Run the active extractor

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv-flan-t5-xl\Scripts\python.exe scripts\run_flan_t5_xl_active.py `
  --input <POINT_IN_TIME_INPUT.jsonl> `
  --output <CALIBRATED_OUTPUT.jsonl>
```

Validate without inference:

```powershell
.\.venv-flan-t5-xl\Scripts\python.exe scripts\run_flan_t5_xl_active.py `
  --input <POINT_IN_TIME_INPUT.jsonl> `
  --output <CALIBRATED_OUTPUT.jsonl> `
  --validate-only
```

Resume only the deterministic postprocessing step when the raw extraction
already completed:

```powershell
.\.venv-flan-t5-xl\Scripts\python.exe scripts\run_flan_t5_xl_active.py `
  --input <POINT_IN_TIME_INPUT.jsonl> `
  --raw-output <RAW_OUTPUT.jsonl> `
  --output <CALIBRATED_OUTPUT.jsonl> `
  --apply-only
```

The wrapper verifies the active schema and calibration hashes before use and
forces the pinned CUDA/FP16/batch-1 runtime.

## Reproduce the locked comparison

The raw evaluation artifact was produced once with:

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv-flan-t5-xl\Scripts\python.exe scripts\run_flan_t5_xl_coarse.py `
  --input outputs\flan_t5\shared\benchmark_300\annotation_batches\coarse_v0_3_evaluation_inputs.jsonl `
  --output outputs\flan_t5_xl\v1_1\evaluation_raw_predictions.jsonl `
  --schema config\news_feature_schema_coarse.json `
  --decoding order_averaged_letter_score `
  --prompt-profile zero_shot `
  --device cuda `
  --precision float16 `
  --batch-size 1
```

The already-frozen rule was then applied without reading labels:

```powershell
.\.venv-flan-t5-xl\Scripts\python.exe scripts\experiment_flan_t5_xl_v1_1.py apply `
  --predictions outputs\flan_t5_xl\v1_1\evaluation_raw_predictions.jsonl `
  --schema config\news_feature_schema_coarse.json `
  --config experiments\flan_t5_xl\v1_1\calibration.json `
  --output outputs\flan_t5_xl\v1_1\evaluation_predictions.jsonl
```

Finally:

```powershell
.\.venv-flan-t5-xl\Scripts\python.exe scripts\evaluate_flan_coarse.py `
  --reference annotations\chatgpt_5_6_sol_reference.jsonl `
  --predictions outputs\flan_t5_xl\v1_1\evaluation_predictions.jsonl `
  --inputs outputs\flan_t5\shared\benchmark_300\annotation_batches\all_inputs.jsonl `
  --schema config\news_feature_schema_coarse.json `
  --split evaluation `
  --output outputs\flan_t5_xl\v1_1\evaluation_agreement.json
```

## Interpretation limit

The 300-document corpus has already informed schema and prompt development.
Its GPT-5.6 annotations are silver labels rather than human ground truth.
Accordingly, even this locked comparison is an engineering benchmark, not a
final scientific estimate. A new human-audited external set is required
before making a reliability claim.
