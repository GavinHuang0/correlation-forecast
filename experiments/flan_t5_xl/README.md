# FLAN-T5-XL Local Comparison

This directory documents the separate FLAN-T5-XL experiment. It does not
replace or modify the active FLAN-T5-Large v0.4 baseline.

## Development outcome

The frozen configuration completed all 72 development documents and produced
valid, untruncated records. It is **rejected as a production feature
extractor**. Only the deterministic relevance gate passed its preregistered
threshold; all four model-produced semantic fields failed.

The development split contains 56 reference-relevant articles:

| Field | Accuracy | Macro-F1 | Required macro-F1 |
|---|---:|---:|---:|
| Shock scope | 0.571 | 0.343 | 0.70 |
| Event family | 0.518 | 0.424 | 0.65 |
| Information status | 0.643 | 0.612 | 0.70 |
| Directional alignment | 0.518 | 0.266 | 0.65 |
| **Mean** | **0.563** | **0.411** | — |

The relevance gate achieved F1 0.932 and semantic coverage 55/56, but those
are produced by deterministic rules rather than FLAN-T5-XL. The model itself
showed serious class collapse:

- `mixed` scope recall was zero across 13 reference examples;
- `supply_capacity` and `other_or_unclear` event recall were both zero;
- `rumor_or_opinion` recall was 1/15;
- `same_direction` and `common_direction_unclear` recall were both zero; and
- canonical/reversed option-order agreement was only 53.2% for scope and
  43.5% for information status.

On the identical development protocol, XL improved mean macro-F1 only from
0.388 for the pure FLAN-T5-Large order-averaged run to 0.411. The gain was
concentrated in scope and alignment; XL was worse on event family and
information status. It remains a useful scaling experiment, not a reliable
source of hard labels for forecast training. The 228-document evaluation split
should remain unrun unless it is needed solely as a locked, documented model
comparison.

The checked-in machine-readable result is
[`development_summary.json`](development_summary.json). Generated predictions
and the full agreement report remain under `outputs/flan_t5_xl/v1_0/`.

## Frozen tested configuration

```text
model:          google/flan-t5-xl
revision:       7d6315df2c2fb742f0f5b556879d730926ca9001
protocol:       FLAN coarse v0.4, zero-shot
decoder:        order-averaged letter scoring
device:         CUDA
precision:      FP16
record batch:   1
candidate batch: 1
quantization:   none
```

FLAN-T5-XL is public and Apache-2.0 licensed. No paid plan, gated-model
approval, or Hugging Face login is required. The pinned snapshot contains two
safetensors weight shards totaling about 11.4 GB on disk. The runtime converts
the weights to FP16 on the GPU.

The experiment uses the same fixed 2024 articles and GPT-5.6 Sol silver labels
as FLAN-T5-Large. The checkpoint predates November 2022, so the existing 2024
benchmark satisfies the repository's conservative model-weight leakage rule.

## 1. Create the environment

Run from the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv-flan-t5-xl
.\.venv-flan-t5-xl\Scripts\python.exe -m pip install --upgrade pip
.\.venv-flan-t5-xl\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv-flan-t5-xl\Scripts\python.exe -m pip install -r requirements-flan-t5-xl.txt
```

Confirm CUDA:

```powershell
.\.venv-flan-t5-xl\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
```

## 2. Validate and cache the pinned checkpoint

Configuration-only validation:

```powershell
.\.venv-flan-t5-xl\Scripts\python.exe scripts\cache_flan_t5_xl.py --validate-only
```

Download and hash only the required safetensors/tokenizer files:

```powershell
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DOWNLOAD_TIMEOUT = "300"
.\.venv-flan-t5-xl\Scripts\python.exe scripts\cache_flan_t5_xl.py
```

If interrupted, rerun the same command. Hugging Face resumes cached partial
files. If the completed manifest already exists and a local verification is
desired, run:

```powershell
.\.venv-flan-t5-xl\Scripts\python.exe scripts\cache_flan_t5_xl.py `
  --local-files-only `
  --overwrite
```

## 3. Validate the development extraction without loading the model

```powershell
.\.venv-flan-t5-xl\Scripts\python.exe scripts\run_flan_t5_xl_coarse.py `
  --input outputs\flan_t5\shared\benchmark_300\annotation_batches\coarse_v0_3_development_inputs.jsonl `
  --output outputs\flan_t5_xl\v1_0\development_predictions.jsonl `
  --schema config\news_feature_schema_coarse.json `
  --decoding order_averaged_letter_score `
  --prompt-profile zero_shot `
  --device cuda `
  --precision float16 `
  --batch-size 1 `
  --validate-only
```

## 4. Run a 12-document smoke extraction

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv-flan-t5-xl\Scripts\python.exe scripts\run_flan_t5_xl_coarse.py `
  --input outputs\flan_t5\shared\benchmark_300\annotation_batches\coarse_v0_3_development_inputs.jsonl `
  --output outputs\flan_t5_xl\v1_0\development_smoke_12.jsonl `
  --schema config\news_feature_schema_coarse.json `
  --decoding order_averaged_letter_score `
  --prompt-profile zero_shot `
  --device cuda `
  --precision float16 `
  --batch-size 1 `
  --limit 12 `
  --overwrite
```

The runner scores one candidate at a time so the 3B model has a better chance
of fitting on the local 8 GB GPU. It does not change prompt semantics or use
quantized weights.

## 5. Run and evaluate all 72 development records

```powershell
.\.venv-flan-t5-xl\Scripts\python.exe scripts\run_flan_t5_xl_coarse.py `
  --input outputs\flan_t5\shared\benchmark_300\annotation_batches\coarse_v0_3_development_inputs.jsonl `
  --output outputs\flan_t5_xl\v1_0\development_predictions.jsonl `
  --schema config\news_feature_schema_coarse.json `
  --decoding order_averaged_letter_score `
  --prompt-profile zero_shot `
  --device cuda `
  --precision float16 `
  --batch-size 1 `
  --overwrite

.\.venv-flan-t5-xl\Scripts\python.exe scripts\evaluate_flan_coarse.py `
  --reference annotations\chatgpt_5_6_sol_reference.jsonl `
  --predictions outputs\flan_t5_xl\v1_0\development_predictions.jsonl `
  --inputs outputs\flan_t5\shared\benchmark_300\annotation_batches\all_inputs.jsonl `
  --schema config\news_feature_schema_coarse.json `
  --split development `
  --output outputs\flan_t5_xl\v1_0\development_agreement.json
```

Do not run the 228 evaluation records for promotion: the development result
already failed the frozen acceptance thresholds. Preserve that split for a
future locked comparison or a different extractor.

If CUDA reports out of memory even with candidate and record batch sizes of
one, stop rather than silently changing precision. The next controlled
experiment would be a separately versioned INT8 run.
