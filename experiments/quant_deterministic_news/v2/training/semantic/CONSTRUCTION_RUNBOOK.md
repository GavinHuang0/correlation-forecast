# W17 and R70 construction runbook

Status: **construction-ready, not inference-complete, and not authorized for
downstream training**.

This runbook covers only `W17__flan_t5_xl` and `R70__gpt_5_6_sol`. It does not
promote the retrospective archive to point-in-time evidence, authorize a paid
GPT request, or authorize Q+L/Q+D+L training.

> **Execution disposition (2026-07-29):** retain this runbook for
> reproducibility, but do not use the full commands as the default next run.
> The workloads are deferred on compute/cost grounds. Use the separately named
> [cost-bounded v3 design](../../../v3/README.md) for the current W17-Lite,
> global-only G40, and later schema-incomplete R70-Lite proposals. No v3
> inference or training is authorized by that design.

## Frozen inputs and completed gates

- Shared assignments:
  `data/features/news_semantic/massive_v2/article_target_assignments.jsonl.gz`
  (`52ade4cee38fc56d91556ec0f568f8dfcb16d981102bd23075075735def13d7c`).
- Shared corpus manifest:
  `data/features/news_semantic/massive_v2/manifest.json`
  (`0c37cf044a4e5e9942398f332db4c09f6746ba7cd6a9af2ae2656e3802a30daf`).
- FLAN acceptance file:
  `config/flan_w17_acceptance_v1.json`
  (`9a1cca6bdf6b0a5bc0c2c5e1f9b6b9dcbb01cca4850e1225aa8449c7650f6072`).
- FLAN tokenizer preflight:
  `data/features/news_semantic/massive_v2/flan_w17/preflight.json`
  (`39cd7a595451d142bca367e99c000a9768ced8d8784e8158187c6cf811bfb1b7`).
  It passed 466,902 assignments and 4,202,118 logical prompts with zero
  over-512 violations.
- FLAN CUDA smoke manifest:
  `data/features/news_semantic/massive_v2/flan_w17/smoke/predictions.jsonl.manifest.json`
  (`26aa5ade4a01fbde40960a0b617d91f6205ca0f8d034664fc458657357dc1c21`).
  One RTX 3070 Ti float16 record completed with no failure or truncation.
- GPT workload preflight:
  `data/features/news_semantic/massive_v2/gpt_r70/preflight.json`
  (`79a8e9aa3e19433d7917371f9e994a2a3ba34f3602da6659a09050c73ab3e96f`).
  It covers 933,804 two-view requests, 3,613,888,330 request bytes, and 934
  conservative files.

Rebuilding the shared corpus is unnecessary unless an upstream input changes.
If it does, rebuild with:

```powershell
.\.venv-training\Scripts\python.exe scripts\build_v2_semantic_corpus.py --overwrite
```

Every downstream preflight and inference artifact must then be regenerated;
never reuse a manifest whose recorded input hash differs.

## FLAN-T5-XL W17

Reproduce the completed tokenizer preflight:

```powershell
.\.venv-flan-t5-xl\Scripts\python.exe scripts\semantic_news_v2\flan_w17.py preflight --input data\features\news_semantic\massive_v2\article_target_assignments.jsonl.gz --output data\features\news_semantic\massive_v2\flan_w17\preflight.json --prompt-batch-size 2048
```

Reproduce the one-record CUDA smoke:

```powershell
.\.venv-flan-t5-xl\Scripts\python.exe scripts\semantic_news_v2\flan_w17.py infer --input data\features\news_semantic\massive_v2\article_target_assignments.jsonl.gz --output data\features\news_semantic\massive_v2\flan_w17\smoke\predictions.jsonl --acceptance-config config\flan_w17_acceptance_v1.json --preflight-manifest data\features\news_semantic\massive_v2\flan_w17\preflight.json --shard-count 32 --shard-index 0 --limit 1
```

For full inference, run 32 separate jobs with `--shard-count 32`,
`--shard-index 0` through `31`, a unique output path such as
`flan_w17/shards/shard_00.jsonl`, and no `--limit`. Preserve every prediction
file and sidecar manifest. Resume an interrupted shard against the same path;
use `--retry-failed` only after reviewing its terminal failures.

Do not aggregate until all 32 manifests:

- bind the frozen assignment, preflight, runtime, model-revision, and
  acceptance hashes;
- collectively cover every assignment exactly once;
- report a terminal record for every selected assignment; and
- report no silent truncation or unreviewed systemic failure.

Then run `flan_w17.py aggregate` with the assignments, stock-day scope,
corpus/preflight manifests, acceptance file, and an explicit list of all 32
prediction paths. The aggregator is fail-closed: missing inference invalidates
the affected stock-day rather than becoming a semantic zero.

The resulting W17 panel remains an `exploratory_silver_fit`;
`primary_training_eligible` and `confirmatory_eligible` stay false.

## GPT-5.6 Sol R70

Reproduce the completed offline workload preflight:

```powershell
.\.venv-training\Scripts\python.exe scripts\semantic_news_v2\gpt_r70.py preflight --assignments data\features\news_semantic\massive_v2\article_target_assignments.jsonl.gz --output data\features\news_semantic\massive_v2\gpt_r70\preflight.json --schema config\news_feature_schema.json --requests-per-file 1000
```

Offline request preparation is expected to materialize about 3.6 GB of JSONL.
Run it only after choosing a paid-pilot scope and confirming adequate local
storage:

```powershell
.\.venv-training\Scripts\python.exe scripts\semantic_news_v2\gpt_r70.py prepare --assignments data\features\news_semantic\massive_v2\article_target_assignments.jsonl.gz --output-root data\features\news_semantic\massive_v2\gpt_r70\batches --schema config\news_feature_schema.json --preflight data\features\news_semantic\massive_v2\gpt_r70\preflight.json --max-requests-per-file 1000
```

No `submit` command may run until all four gates are explicit:

1. an OpenAI API credential is available without writing its value to an
   artifact;
2. licensed-text processing is confirmed;
3. a hard maximum paid-request budget is stated; and
4. the paid execution is explicitly approved within the research process.

The first authorized action should be a small pilot, never the full 934-file
workload. `submit` additionally requires both confirmation flags and
`--max-paid-requests`; keep `--max-new-shards` small. The preflight byte count
is not a token-cost estimate.

After submission, use the runner's `collect`, `merge`, `prepare-retry`,
`adjudicate`, and `aggregate` commands in that order, consulting
`gpt_r70.py <command> --help` for the paths created by the preceding stage.
Retry only exact failed request bodies. Run `cleanup-remote-files` only after
download hashes and the merged local manifest have been verified, and only
with its explicit deletion confirmation.

R70 aggregation must fail closed unless both prompt views are terminal for
every assignment and validation/adjudication coverage is complete. Every GPT
article, daily feature, fit, and comparison must retain:

```text
claim_label = exploratory_future_contaminated_oracle
extractor_model_knowledge_contaminated = 1
confirmatory_eligible = 0
```

## Gate before training

Do not start Q+L or Q+D+L training until the selected arm has:

1. complete hash-bound article predictions over all 466,902 assignments;
2. a complete daily W17 or R70 panel with explicit no-candidate versus
   failed/incomplete-source handling;
3. an exact stock-date intersection report against Q56 and D2;
4. frozen feature order, transformations, missingness rules, folds, controls,
   random seeds, and artifact hashes; and
5. the claim labels required by the FLAN-silver or GPT-oracle contract.

The current authoritative readiness record is [`STATUS.md`](STATUS.md), with
machine-readable evidence in [`status.json`](status.json).
