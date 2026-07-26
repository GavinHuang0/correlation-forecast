# Llama 3.1 8B Instruct: Development-Screened Candidate

## Decision

`meta-llama/Llama-3.1-8B-Instruct` v1.0 was tested because XL v1.1 still
failed every semantic threshold. The checkpoint has an officially documented
December 2023 knowledge cutoff, while every fixed benchmark article is from
2024. The existing 300-document set was therefore eligible without asking
GPT-5.6 to label a new corpus.

The model and tokenizer are pinned to:

```text
meta-llama/Llama-3.1-8B-Instruct
0e9e39f249a16976918f6564b8830bc894c89659
```

Official references:

- [model card and access request](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct)
- [immutable pinned snapshot](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct/tree/0e9e39f249a16976918f6564b8830bc894c89659)
- [Transformers bitsandbytes guide](https://huggingface.co/docs/transformers/main/en/quantization/bitsandbytes)

The checkpoint download is complete: 13 files totaling 16,069,779,031 bytes
were hash-verified offline. Smoke extraction and all 72 development documents
completed without truncation or GPU offload.

The model did not pass the development gate:

| Field | Macro-F1 | Required |
|---|---:|---:|
| Shock scope | 0.177 | 0.70 |
| Event family | 0.590 | 0.65 |
| Information status | 0.540 | 0.70 |
| Directional alignment | 0.240 | 0.65 |

Mean semantic macro-F1 was 0.387, compared with 0.472 for active XL v1.1 on
the same development split. Scope collapsed toward `idiosyncratic`, which
also forced most alignment outputs to `single_firm_only`.

v1.0 is therefore rejected as an all-field extractor. The 228-document
holdout remains unmaterialized. See
[`archive/v1_0/development_summary.json`](archive/v1_0/development_summary.json)
for exact metrics, hashes, and the decision record.

## Account and license

The checkpoint is gated. Before downloading:

1. Sign in to Hugging Face.
2. Open the model page and accept the Llama 3.1 Community License.
3. Agree to share the requested contact details with Meta and obtain access.
4. Create a read-only Hugging Face token.
5. Authenticate the machine with
  `.\.venv-llama31\Scripts\hf.exe auth login`.

Llama 2 access does not necessarily grant Llama 3.1 access. The weights are
free to download after approval; there is no per-call model charge for local
inference.

## Why this model

Meta documents Llama 3.1 8B as a static offline model with:

- 8 billion parameters;
- supervised instruction tuning and RLHF;
- grouped-query attention;
- a 128K context window; and
- a December 2023 knowledge cutoff.

The experiment deliberately caps context at 1,024 tokens. It uses NF4 double
quantization, FP16 compute, and batch size one to target the local RTX 3070 Ti
8 GiB GPU. Candidate scoring requests only the logits positions actually
needed, avoiding a full sequence-by-vocabulary logits tensor. This is still a
tight configuration, so run the 12-document smoke test before a full
development extraction.

Qwen2.5/Qwen3 were not selected because their official model cards do not
state a sufficiently precise knowledge cutoff for this 2024 benchmark.
Phi-4-mini's June 2024 cutoff would invalidate the earlier portion of the set.

## Decoder

The script does not ask the chat model to emit JSON or explanatory prose. It:

1. reuses the same article-first coarse schema and deterministic hierarchy;
2. renders prompts with the checkpoint's official chat template;
3. derives the exact assistant-side `Answer:` boundary from complete chat
  renders;
4. averages every semantic label across all cyclic option positions; and
5. scores the complete candidate token sequence if a choice is not one token.

This directly addresses the assistant-boundary and option-position failures
observed in Llama 2.

## 1. Create the environment

Run from the repository root in PowerShell:

```powershell
py -3.11 -m venv .venv-llama31
.\.venv-llama31\Scripts\python.exe -m pip install --upgrade pip
.\.venv-llama31\Scripts\python.exe -m pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
.\.venv-llama31\Scripts\python.exe -m pip install -r requirements-llama-3-1.txt
```

The PyTorch command and requirements file pin the exact top-level versions
already exercised by the repository's local Llama runtime. The extraction
manifest records the versions actually used.

Confirm CUDA:

```powershell
.\.venv-llama31\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.version.cuda); print(torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
```

## 2. Validate the pinned cache configuration

This does not download or load the model:

```powershell
.\.venv-llama31\Scripts\python.exe scripts\cache_llama_3_1.py --validate-only
```

## 3. Download the gated pinned snapshot

```powershell
.\.venv-llama31\Scripts\hf.exe auth login
$env:HF_HUB_DISABLE_XET = "1"
$env:HF_HUB_DOWNLOAD_TIMEOUT = "300"
.\.venv-llama31\Scripts\python.exe scripts\cache_llama_3_1.py
```

The pinned source weights occupy roughly 16 GB. If the download is
interrupted before the manifest is written, rerun the same command; Hugging
Face will reuse complete blobs and resume partial cache state. If the manifest
already exists, the scripted download completed. Verify it without network:

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv-llama31\Scripts\python.exe scripts\cache_llama_3_1.py `
  --local-files-only `
  --overwrite
```

## 4. Prepare the test data

The data is already materialized locally, but this command reproduces and
verifies it:

```powershell
.\.venv-llama31\Scripts\python.exe scripts\prepare_llama_3_1_benchmark.py `
  --overwrite
```

It writes:

```text
outputs/llama_3_1/shared/benchmark_300/development_inputs.jsonl
outputs/llama_3_1/shared/benchmark_300/evaluation_inputs.jsonl
outputs/llama_3_1/shared/benchmark_300/manifest.json
```

The copies contain no GPT labels. Their hashes, counts, split integrity, and
post-cutoff eligibility are recorded in the manifest.

## 5. Validate prompts without loading the model

```powershell
.\.venv-llama31\Scripts\python.exe scripts\extract_llama_3_1_coarse.py `
  --input outputs\llama_3_1\shared\benchmark_300\development_inputs.jsonl `
  --output outputs\llama_3_1\v1_0\development_predictions.jsonl `
  --validate-only
```

## 6. Run a 12-document smoke extraction

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv-llama31\Scripts\python.exe scripts\extract_llama_3_1_coarse.py `
  --input outputs\llama_3_1\shared\benchmark_300\development_inputs.jsonl `
  --output outputs\llama_3_1\v1_0\development_smoke_12.jsonl `
  --limit 12 `
  --overwrite
```

Stop if CUDA runs out of memory. Do not silently change precision,
quantization, context, or offload policy inside the v1.0 result.

## 7. Run all 72 development documents

```powershell
.\.venv-llama31\Scripts\python.exe scripts\extract_llama_3_1_coarse.py `
  --input outputs\llama_3_1\shared\benchmark_300\development_inputs.jsonl `
  --output outputs\llama_3_1\v1_0\development_predictions.jsonl `
  --overwrite

.\.venv-llama31\Scripts\python.exe scripts\evaluate_flan_coarse.py `
  --reference annotations\chatgpt_5_6_sol_reference.jsonl `
  --predictions outputs\llama_3_1\v1_0\development_predictions.jsonl `
  --inputs outputs\flan_t5\shared\benchmark_300\annotation_batches\all_inputs.jsonl `
  --schema config\news_feature_schema_coarse.json `
  --split development `
  --output outputs\llama_3_1\v1_0\development_agreement.json
```

Review macro-F1, confusion matrices, option-rotation agreement, truncation,
and VRAM behavior. Freeze any bounded development-only change before touching
the 228 evaluation labels.

## 8. Locked 228-document evaluation: intentionally reserved

Do not run the holdout for v1.0. The frozen configuration failed all four
development thresholds, so spending the holdout would add no promotion
evidence and would reduce its value for a future preregistered candidate.

Any future v1.1 experiment must freeze its prompt, scope decision rule,
hierarchical alignment routing, and acceptance rule using development data
only before running the holdout once.
