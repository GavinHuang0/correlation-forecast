# Llama 2 7B Chat Local Extraction Benchmark

## Status

This is the operational local-model comparison against the active
FLAN-T5-Large v0.4 baseline. The pinned checkpoint was cached and all 300
documents were processed under both the original and corrected decoders.
Version 1.0 is the frozen failed baseline. Version 1.1 fixes its answer
boundary but is rejected as a full-field replacement. Version 1.2 is the
best development-selected Llama candidate and should be used only as an
event-family specialist until it passes a fresh external holdout.

- Model: `meta-llama/Llama-2-7b-chat-hf`
- Immutable revision: `f5db02db724555f92da89c216ac04704f23d4590`
- v1.0 prompt: `llama-2-stock-sector-news-v1.0.0`
- v1.1 prompt: `llama-2-stock-sector-news-v1.1.0-development`
- v1.0 extraction manifest: `llama-2-extraction-v1`
- v1.1 extraction manifest: `llama-2-development-experiment-v1`
- Model configuration: CUDA, FP16 compute, NF4, double quantization, eager
  attention, batch size 1, 4,096-token hard limit
- v1.0 decoder: canonical-and-reversed immediate option-letter score average
- v1.1 decoder: exact partial-assistant `Answer:` cue plus all cyclic rotations
- Training: none
- Free-form generation: none
- Network during extraction: disabled after the snapshot is cached

The complete frozen contract is in [`protocol.json`](protocol.json), and the
fixed-data audit is in [`benchmark_manifest.json`](benchmark_manifest.json).
Compact results are in
[`evaluation_summary.json`](evaluation_summary.json).
The selected research-candidate pointer is
[`research_candidate.json`](research_candidate.json); v1.1 and v1.2 have
separate protocols and compact summaries under their own directories.

## Frozen v1.0 baseline result

The local artifacts contain:

```text
outputs/llama_2/v1_0/development_predictions.jsonl       # 72
outputs/llama_2/v1_0/evaluation_predictions.jsonl        # 228
outputs/llama_2/v1_0/evaluation_agreement.json
outputs/llama_2/v1_0/failure_diagnostics.json
outputs/llama_2/v1_0/next_token_probe_dev_row1_scope.json
outputs/llama_2/v1_1/development_predictions.jsonl       # 72
outputs/llama_2/v1_1/evaluation_predictions.jsonl        # 228, post-hoc use
outputs/llama_2/v1_2/development_predictions.jsonl       # 72 hybrid
outputs/llama_2/v1_2/evaluation_predictions.jsonl        # 228 hybrid, post-hoc
```

The headline comparison remains the held-out 228-record evaluation split, of
which 174 records are semantically relevant under the GPT-5.6 Sol silver
reference. The 72 development records are not blended into the score because
FLAN v0.4 used that split for protocol selection.

| Field | Llama accuracy | Llama macro-F1 | FLAN v0.4 macro-F1 |
|---|---:|---:|---:|
| Shock scope | 0.546 | 0.195 | 0.372 |
| Event family | 0.138 | 0.102 | 0.452 |
| Information status | 0.552 | 0.238 | 0.623 |
| Directional alignment | 0.546 | 0.219 | 0.335 |
| **Mean** | **0.445** | **0.188** | **0.446** |

Llama's mean accuracy is 0.049 below the FLAN hybrid. Its primary mean
macro-F1 is 0.257 below FLAN, a 57.7% relative reduction. A post-hoc paired
20,000-replicate article bootstrap gives:

```text
accuracy delta 95% interval: [-0.102,  0.004]
macro-F1 delta 95% interval: [-0.304, -0.205]
```

The macro-F1 interval is wholly below zero; the accuracy interval crosses
zero. This bootstrap was added after the frozen evaluation and is diagnostic,
not a preregistered significance test.

These are agreement scores against silver annotations, not human-ground-truth
accuracy.

## Failure diagnosis

The run completed cleanly, so this is not an output-parser or truncation
failure:

- every saved label is schema-valid;
- no free-form output is parsed;
- all recorded prompts are untruncated;
- the largest prompt has 468 tokens against a 4,096-token limit; and
- the model and data hashes match their manifests.

The failure occurs at the answer boundary. On a pinned development probe, the
model's immediate next token is whitespace token `29871` with log probability
`-0.0000019`, while the v1.0 decoder immediately scores `A` through `D` at
roughly `-22` to `-25`. Greedy output begins:

```text
 Based on the provided article and target...
```

After consuming the leading whitespace, candidate letters become plausible
(`-3.47` to `-4.74` in the probe), but the model still prefers the explanatory
word `Based` (`-0.13`). The option token IDs themselves match the rendered
assistant continuations, so the problem is not a mislabeled token ID. The
decoder is scoring at a position where the chat model does not naturally emit
the requested letter.

This boundary mismatch exposes a severe option-position prior:

- canonical/reversed agreement is 0/196 for shock scope;
- canonical/reversed agreement is 3/196 for event family;
- canonical/reversed agreement is 0/196 for information status;
- canonical scope chooses `idiosyncratic` 196/196 times; and
- canonical status chooses `confirmed` 196/196 times.

The averaged predictions therefore collapse toward `idiosyncratic` and
`confirmed`. Scope collapse also propagates through the hierarchy, which
derives many alignment values as `single_firm_only`.

## Completed answer-boundary experiment

v1.1 keeps the checkpoint, NF4 weights, schema, deterministic gate, hierarchy,
and inputs fixed. It changes only the classification boundary:

1. Complete replies such as `Answer: A` are rendered with the pinned
   tokenizer.
2. The exact shared partial-assistant `Answer:` token prefix is derived and
   verified.
3. Only the candidate-specific letter token is scored.
4. Every semantic label is placed in every option position using all cyclic
   rotations, and semantic log probabilities are averaged.

The preregistered full-replacement rule rejected v1.1 on the 56
reference-relevant development records:

| Metric | v1.0 | v1.1 full replacement | Delta |
|---|---:|---:|---:|
| Mean field accuracy | 0.424 | 0.388 | -0.036 |
| Mean macro-F1 | 0.173 | 0.263 | +0.090 |

The aggregate hides a useful field-specific result. Event-family accuracy
rose from 0.089 to 0.607 and macro-F1 from 0.059 to 0.548. Scope and
alignment declined, so replacing the full decoder would be inappropriate.

v1.2 is a deterministic fieldwise hybrid selected on that development
result:

```text
shock_scope            <- v1.0
event_family           <- v1.1 corrected decoder
information_status     <- v1.1 corrected decoder
directional_alignment  <- v1.0
```

Development mean accuracy is 0.554 and mean macro-F1 is 0.310, improvements
of 0.129 and 0.137 over v1.0. The frozen mapping was then applied to the
remaining 228 records as an explicitly post-hoc engineering comparison:

| Field | v1.2 accuracy | v1.2 macro-F1 | FLAN v0.4 macro-F1 |
|---|---:|---:|---:|
| Shock scope | 0.546 | 0.195 | 0.372 |
| Event family | 0.517 | 0.449 | 0.452 |
| Information status | 0.540 | 0.257 | 0.623 |
| Directional alignment | 0.546 | 0.219 | 0.335 |
| **Mean** | **0.537** | **0.280** | **0.446** |

The v1.2 mean improves over Llama v1.0 by 0.092 accuracy and 0.092 macro-F1.
Its raw accuracy is 0.043 above FLAN, but that is driven by frequent classes:
model-origin dominant shares are 97.7% for scope and 96.0% for status. Mean
macro-F1 remains 0.166 below FLAN.

The defensible conclusion is narrow: the corrected Llama decoder is useful
for `event_family`; Llama v1.2 is not an all-field FLAN replacement. These are
silver-label agreement scores, and the 228 records were previously inspected
for v1.0. Report observed post-hoc improvement, not statistical significance,
and require a fresh external holdout for confirmation.

## Why the existing benchmark is eligible

Meta documents a September 2022 pretraining cutoff and says that some tuning
data is as recent as July 2023. This repository therefore uses July 31, 2023
as the conservative cutoff. All 300 benchmark articles are from 2024
(`2024-01-02` through `2024-12-27`), so the exact FLAN benchmark and GPT-5.6
Sol silver labels can be reused without new labeling.

This establishes a model-weight leakage boundary. It does not turn the GPT
labels into human ground truth, and it does not resolve the source dataset's
article-version/timestamp limitations.

## One-time access requirement

The official checkpoint is gated. Before downloading it:

1. Create or sign in to a free Hugging Face account.
2. Open
   [`meta-llama/Llama-2-7b-chat-hf`](https://huggingface.co/meta-llama/Llama-2-7b-chat-hf).
3. Accept the Llama 2 Community License and agree to share the requested
   contact information with Meta.
4. Wait until that same Hugging Face account has access.
5. Create a read-only Hugging Face token. Do not place the token in `.env`,
   source control, command history, or a script.

No paid Hugging Face plan and no per-call model fee are required for local
inference.

## Install on this Windows/NVIDIA workstation

The existing `.venv` in this workspace points to a Python installation that
is no longer present, and the Windows Python launcher currently finds no
installed Python. Install Python 3.11 if needed, reopen PowerShell, and create
a separate environment:

```powershell
winget install --exact --id Python.Python.3.11
py -3.11 -m venv .venv-llama2
.\.venv-llama2\Scripts\python.exe -m pip install --upgrade pip
.\.venv-llama2\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.\.venv-llama2\Scripts\python.exe -m pip install -r requirements-llama.txt
```

Authenticate with the approved Hugging Face account:

```powershell
.\.venv-llama2\Scripts\hf.exe auth login
```

Verify the GPU runtime before any download:

```powershell
.\.venv-llama2\Scripts\python.exe -c "import torch; print(torch.__version__); print(torch.cuda.get_device_name(0)); assert torch.cuda.is_available()"
```

The checked machine has an NVIDIA GeForce RTX 3070 Ti with 8 GiB VRAM, so the
frozen NF4 configuration is intentional. The official FP16 weights occupy
about 13.5 GB on disk; reserve additional space for the Hugging Face cache.

## Validate or rebuild the local test dataset

The byte-identical Llama 2 copies are already present under
`outputs/llama_2/shared/benchmark_300/`. Verify every fixed hash, ID, split,
label-separation rule, and cutoff without writing:

```powershell
.\.venv-llama2\Scripts\python.exe scripts\prepare_llama_2_benchmark.py --validate-only
```

Only if those prepared files are missing, create them with:

```powershell
.\.venv-llama2\Scripts\python.exe scripts\prepare_llama_2_benchmark.py
```

To deliberately reconstruct an existing prepared copy from its hash-locked
FLAN sources, add `--overwrite`. The preparation command writes:

```text
outputs/llama_2/shared/benchmark_300/development_inputs.jsonl
outputs/llama_2/shared/benchmark_300/evaluation_inputs.jsonl
outputs/llama_2/shared/benchmark_300/manifest.json
```

The extractor inputs contain article text and target metadata, but no GPT
reference labels.

## Cache the immutable checkpoint

This is the only step that needs Hub network access:

```powershell
.\.venv-llama2\Scripts\python.exe scripts\cache_llama_2.py `
  --manifest outputs\llama_2\model_snapshot_manifest.json
```

The cache script downloads only the pinned safetensors/model/tokenizer files
and hashes every cached file. A 401/403 error means the logged-in account has
not yet received access or is not the account that accepted the license.

## Validate the extraction contract without loading the model

```powershell
.\.venv-llama2\Scripts\python.exe scripts\extract_llama_2_coarse.py `
  --input outputs\llama_2\shared\benchmark_300\development_inputs.jsonl `
  --output outputs\llama_2\v1_0\contract_check.jsonl `
  --validate-only
```

`--validate-only` does not create predictions, import the model runtime, or
load weights.

## Recommended development smoke run

After caching, force offline model access and classify 12 development rows:

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv-llama2\Scripts\python.exe scripts\extract_llama_2_coarse.py `
  --input outputs\llama_2\shared\benchmark_300\development_inputs.jsonl `
  --output outputs\llama_2\v1_0\development_smoke12.jsonl `
  --limit 12 `
  --local-files-only
```

Inspect the prediction JSONL and adjacent manifest before the full run. Do not
use the evaluation reference labels to change the prompt, label ordering,
quantization, or routing.

Run the full development split separately:

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv-llama2\Scripts\python.exe scripts\extract_llama_2_coarse.py `
  --input outputs\llama_2\shared\benchmark_300\development_inputs.jsonl `
  --output outputs\llama_2\v1_0\development_predictions.jsonl `
  --local-files-only
```

## Frozen evaluation extraction

Run the full 228-record evaluation once the smoke output is valid:

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv-llama2\Scripts\python.exe scripts\extract_llama_2_coarse.py `
  --input outputs\llama_2\shared\benchmark_300\evaluation_inputs.jsonl `
  --output outputs\llama_2\v1_0\evaluation_predictions.jsonl `
  --local-files-only
```

The extractor refuses:

- a different model ID or revision;
- prompt truncation;
- CPU or disk offload in the NF4 run;
- an invalid schema or input record;
- output overwrite without `--overwrite`; or
- an incomplete/ambiguous option-token contract.

It writes a manifest beside the predictions with snapshot hashes, source
hashes, prompt hashes, tokenizer/runtime versions, routing details, and the
final output hash.

## Evaluate against GPT silver labels and FLAN v0.4

```powershell
.\.venv-llama2\Scripts\python.exe scripts\evaluate_llama_2_coarse.py `
  --predictions outputs\llama_2\v1_0\evaluation_predictions.jsonl `
  --output outputs\llama_2\v1_0\evaluation_agreement.json
```

The evaluator loads no model. It rejects a changed benchmark, model revision,
prompt, chat template, quantization, input order, output hash, schema, or
deterministic routing implementation before calculating:

- per-field accuracy, macro-F1, kappa, and confusion matrices;
- mean semantic accuracy and macro-F1;
- deterministic relevance/surprise metrics;
- majority baselines; and
- paired deltas against the checked-in FLAN-T5 v0.4 summary.

These values are agreement with GPT-5.6 Sol silver annotations, not
human-ground-truth accuracy.

Reproduce the saved-result decoder and paired-bootstrap diagnostic without
loading either model:

```powershell
.\.venv-llama2\Scripts\python.exe scripts\diagnose_llama_2_results.py `
  --output outputs\llama_2\v1_0\failure_diagnostics.json `
  --bootstrap-reps 20000 `
  --seed 20260724 `
  --overwrite
```

Probe one development prompt with the pinned local model:

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv-llama2\Scripts\python.exe scripts\probe_llama_2_next_tokens.py `
  --input outputs\llama_2\shared\benchmark_300\development_inputs.jsonl `
  --row 1 `
  --field shock_scope `
  --top-k 20 `
  --output outputs\llama_2\v1_0\next_token_probe_dev_row1_scope.json `
  --overwrite
```

## Reproduce the v1.1/v1.2 experiment

Run the corrected decoder on the development split:

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv-llama2\Scripts\python.exe scripts\experiment_llama_2_v1_1.py `
  --input outputs\llama_2\shared\benchmark_300\development_inputs.jsonl `
  --output outputs\llama_2\v1_1\development_predictions.jsonl `
  --local-files-only

.\.venv-llama2\Scripts\python.exe scripts\evaluate_llama_2_v1_1.py `
  --predictions outputs\llama_2\v1_1\development_predictions.jsonl `
  --output outputs\llama_2\v1_1\development_agreement.json
```

Build and evaluate the development-selected hybrid without loading a model:

```powershell
.\.venv-llama2\Scripts\python.exe scripts\build_llama_2_v1_2_hybrid.py `
  --v1-0-predictions outputs\llama_2\v1_0\development_predictions.jsonl `
  --v1-1-predictions outputs\llama_2\v1_1\development_predictions.jsonl `
  --output outputs\llama_2\v1_2\development_predictions.jsonl

.\.venv-llama2\Scripts\python.exe scripts\evaluate_llama_2_v1_2.py `
  --split development `
  --predictions outputs\llama_2\v1_2\development_predictions.jsonl `
  --output outputs\llama_2\v1_2\development_agreement.json
```

The 228-record invocation is retained for reproduction but must remain
post-hoc:

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv-llama2\Scripts\python.exe scripts\experiment_llama_2_v1_1.py `
  --input outputs\llama_2\shared\benchmark_300\evaluation_inputs.jsonl `
  --output outputs\llama_2\v1_1\evaluation_predictions.jsonl `
  --local-files-only

.\.venv-llama2\Scripts\python.exe scripts\build_llama_2_v1_2_hybrid.py `
  --v1-0-predictions outputs\llama_2\v1_0\evaluation_predictions.jsonl `
  --v1-1-predictions outputs\llama_2\v1_1\evaluation_predictions.jsonl `
  --output outputs\llama_2\v1_2\evaluation_predictions.jsonl

.\.venv-llama2\Scripts\python.exe scripts\evaluate_llama_2_v1_2.py `
  --split posthoc_evaluation `
  --predictions outputs\llama_2\v1_2\evaluation_predictions.jsonl `
  --output outputs\llama_2\v1_2\evaluation_agreement.json
```

The shared v1.1 runner writes `development_only=true` in both manifests. For
the 228-record invocation this generic flag is inaccurate; the input hash,
228-row count, [`v1_2/posthoc_run_metadata.json`](v1_2/posthoc_run_metadata.json),
v1.2 protocol, and evaluator identify it as post-hoc. A future runner version
should expose an explicit run-role flag.

## Operational files

| File | Role |
|---|---|
| [`../../scripts/prepare_llama_2_benchmark.py`](../../scripts/prepare_llama_2_benchmark.py) | Verify and byte-copy the fixed inputs |
| [`../../scripts/cache_llama_2.py`](../../scripts/cache_llama_2.py) | Cache and hash the gated immutable snapshot |
| [`../../scripts/extract_llama_2_coarse.py`](../../scripts/extract_llama_2_coarse.py) | Run deterministic local extraction |
| [`../../scripts/evaluate_llama_2_coarse.py`](../../scripts/evaluate_llama_2_coarse.py) | Validate and score the result against GPT/FLAN |
| [`../../scripts/diagnose_llama_2_results.py`](../../scripts/diagnose_llama_2_results.py) | Reproduce class-collapse and paired-bootstrap diagnostics |
| [`../../scripts/probe_llama_2_next_tokens.py`](../../scripts/probe_llama_2_next_tokens.py) | Inspect the frozen chat answer boundary and next-token behavior |
| [`../../scripts/experiment_llama_2_v1_1.py`](../../scripts/experiment_llama_2_v1_1.py) | Run the exact `Answer:` cue and cyclic-rotation experiment |
| [`../../scripts/evaluate_llama_2_v1_1.py`](../../scripts/evaluate_llama_2_v1_1.py) | Apply the preregistered development promotion checks |
| [`../../scripts/build_llama_2_v1_2_hybrid.py`](../../scripts/build_llama_2_v1_2_hybrid.py) | Build the hash-validated fieldwise hybrid |
| [`../../scripts/evaluate_llama_2_v1_2.py`](../../scripts/evaluate_llama_2_v1_2.py) | Score development or post-hoc hybrid results |

Generated datasets, weights, predictions, and reports stay under ignored
`outputs/` or the Hugging Face cache. Only code, frozen contracts, compact
results, and documentation belong in source control.
