# FLAN-T5-XL Local Comparison

This directory documents the separate FLAN-T5-XL experiment. It does not
replace or modify the active FLAN-T5-Large v0.4 baseline.

## Frozen candidate

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
$env:PYTORCH_CUDA_ALLOC_CONF = "expandable_segments:True"
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

Do not run the 228 evaluation records yet. First inspect the smoke run, complete
the 72-record development run, and decide whether the frozen configuration is
worth evaluating. This preserves the remaining split from another avoidable
round of post-hoc tuning.

If CUDA reports out of memory even with candidate and record batch sizes of
one, stop rather than silently changing precision. The next controlled
experiment would be a separately versioned INT8 run.
