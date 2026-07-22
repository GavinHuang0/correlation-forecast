# FLAN-T5 Agreement Evaluation

This benchmark measures agreement between frozen `google/flan-t5-large` predictions and 300 GPT-5.6 Sol silver-reference annotations. It does not claim that GPT-5.6 Sol is objective ground truth.

## Frozen model configuration

- Model: `google/flan-t5-large`
- Immutable Hugging Face revision: `0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a`
- Decoding: greedy, `do_sample=False`, `num_beams=1`
- Primary comparison: nine closed-label fields in three short generations
- Extended comparison: channels, entities, and exact evidence in three additional generations
- Network access during extraction: disabled with `--local-files-only`

The pinned revision is the model repository's July 17, 2023 `main` commit. The script hashes the local weight, tokenizer, and configuration files and writes the actual Python, PyTorch, Transformers, hardware, precision, rendered prompts, schema, input, and decoding configuration to a sidecar manifest. Pinning the repository is not treated as proof of a training cutoff by itself; the recorded weight hash is the auditable checkpoint identity.

The 2024 GPT-5.6 Sol annotations were produced interactively under the user's explicit instruction to ignore the cutoff issue. They may therefore contain memorized-future contamination or interactive-model variability even though annotation was restricted to the supplied text. Treat them only as a silver development reference. A later final evaluation should use a pinned API snapshot and a genuinely post-cutoff sample.

## Install and cache the model

Install PyTorch using the build appropriate for the local CPU/GPU, then install the remaining dependencies:

```powershell
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements-llm.txt
```

Download the pinned model snapshot once:

```powershell
hf download google/flan-t5-large `
  config.json generation_config.json model.safetensors `
  special_tokens_map.json spiece.model tokenizer.json tokenizer_config.json `
  --revision 0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a
```

## Validate without loading the model

This command validates all 300 records and constructs every prompt, but it does not import Transformers or load FLAN-T5:

```powershell
.\.venv\Scripts\python.exe scripts\extract_flan_t5.py `
  --input outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\annotation_batches\all_inputs.jsonl `
  --output outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\flan_t5_large_predictions.jsonl `
  --revision 0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a `
  --mode full `
  --validate-only
```

## Run the full frozen extraction

After the snapshot is cached, run all six narrow passes offline:

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv\Scripts\python.exe scripts\extract_flan_t5.py `
  --input outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\annotation_batches\all_inputs.jsonl `
  --output outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\flan_t5_large_predictions.jsonl `
  --revision 0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a `
  --mode full `
  --device cpu `
  --precision float32 `
  --batch-size 1 `
  --local-files-only `
  --overwrite
```

The command fixes CPU/FP32 so the benchmark configuration does not silently change across machines. A CUDA/FP16 run is allowed as a separate experiment, but it must use a different output path and be reported as a distinct extractor configuration.

Use `--mode core` for the protocol's primary three-pass, nine-field comparison. That produces 900 generations instead of 1,800. Use `--resume` instead of `--overwrite` after an interrupted run; the script refuses to resume if the model files, prompts, schema, input, runtime, device, precision, batch size, or generation settings differ.

The extractor never receives price data, realized correlation, vendor sentiment, sampling strata, later news, web search, or retrieval tools.

## Calculate agreement after extraction

```powershell
.\.venv\Scripts\python.exe scripts\evaluate_flan_agreement.py `
  --reference annotations\chatgpt_5_6_sol_reference.jsonl `
  --predictions outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\flan_t5_large_predictions.jsonl `
  --inputs outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\annotation_batches\all_inputs.jsonl `
  --output outputs\019f85b5-0b78-7bf1-8eb8-df198bfb385d\flan_t5_agreement_report.json
```

The report includes macro-F1, Cohen's kappa, per-class metrics, confusion matrices, multi-label channel/entity scores, exact evidence validity, and output-format validity.
