# V2 semantic construction and training status

Audit date: **2026-07-28**

Status: **the shared semantic input corpus is complete; the FLAN-T5-XL W17
construction pipeline is ready and its full tokenizer preflight passed, with
its one-record CUDA smoke also complete; the GPT-5.6 Sol R70 offline pipeline
is ready and its final hash-bound workload preflight passed. No complete
article inference, daily L panel, or downstream semantic model exists**.

The machine-readable record is [`status.json`](status.json). `Pipeline ready`
means the bounded input contract, runner, validation, resumability, and
fail-closed aggregation path exist. It does not mean inference is complete,
a daily panel is ready, or training is authorized.

## Decision

| Arm | What exists | Missing before fitting | Current result |
|---|---|---|---|
| Shared semantic corpus | Hash-bound bounded text, complete C/I/P candidate routing, row-universe audit | Model inference and daily semantic aggregation | Complete exploratory input artifact: 466,902 assignments, 55,197 articles, 27,510 stock-days |
| `W17__flan_t5_xl` | Pinned coarse XL runner, resumable inference, fail-closed daily aggregator, exact permissive silver-only acceptance lock, passed full tokenizer preflight and CUDA smoke | Run full inference; build daily W17 and Q/L join | Pipeline ready; execution proof complete; not inferred or trained |
| `W17__gpt_5_6_sol` | Bounded fine/coarse reference fragments | Accepted GPT R70 fine labels, deterministic coarsening, daily W17 derivation | Not pursued in this construction pass |
| `R70__flan_t5_xl` | Fine contract and historical runner concept | Pinned fine XL workflow, validation, complete inference, daily R70 | Not pursued in this construction pass |
| `R70__gpt_5_6_sol` | Passed final hash-bound full-corpus workload preflight plus offline two-view Batch/retry/merge/adjudication/aggregation pipeline | API credential, licensed-text confirmation, explicit request budget, separate paid-run authorization, full inference, daily R70 and Q/L join | Offline pipeline ready; paid inference blocked; not trained |

Execution disposition (2026-07-29): the ready full W17/R70 pipelines are
deferred rather than launched. The separate
[cost-bounded v3 design](../../../v3/README.md) proposes a four-way,
unique-article FLAN contract, an ordered global-only GPT block, and a later
schema-incomplete factorized GPT contract. This status remains the factual
readiness record for the unchanged v2 pipelines.

No partial benchmark was promoted into a stock-day feature panel. That is the
important fail-closed result: the local fragments establish schemas and
execution mechanics, but they do not cover the historical modeling corpus.

## Exact available fragments

- The normalized Massive archive contains **90,290** unique provider articles,
  with descriptions on **88,148**.
- The semantic-common modeling window from
  `2022-10-31T13:00:00Z` through, but excluding,
  `2026-06-30T13:00:00Z` contains **57,181** unique pre-routing articles,
  including **56,264** with descriptions.
- Those bounds are the exact information-time union for forecast sessions
  **2022-11-01 through 2026-06-30**: filter `published_at_utc` at or after the
  previous official session's 09:00 ET cutoff for the first forecast and
  strictly before 09:00 ET on the final forecast date. Both endpoints are
  13:00Z because daylight saving time is in effect on those dates.
- The **57,147 / 56,232** figures in `DESIGN_AUDIT.md` use a different,
  descriptive UTC calendar-date rule:
  `2022-11-01T00:00:00Z <= published_at_utc < 2026-07-01T00:00:00Z`.
  That rule omits 55 articles (53 with descriptions) from the first
  forecast's prior-session lead-in and includes 21 articles (all with
  descriptions) published after the final 09:00 ET decision cutoff. The net
  semantic-window difference is therefore **+34 articles and +32
  descriptions**. Use 57,181 / 56,264 for semantic construction; retain
  57,147 / 56,232 only as a calendar-date archive description.
- The exact materialized semantic input corpus contains **55,197 assigned
  unique articles**, **466,902 article-target assignments**, and **27,510
  stock-day scope rows**. Exactly **204 stock-days have no candidate**.
- Candidate routing may use complete source descriptions, provider tickers,
  keywords, and other provider metadata. Extractors receive none of those
  routing fields. Their visible entity and role flags are recomputed only from
  the bounded shared `model_text`.
- The legacy GPT fine reference has **300 articles across five tickers**:
  AMD, AVGO, INTC, MU, and NVDA. Its deterministic coarse version contains the
  same 300 articles.
- The active FLAN-T5-XL benchmark has **72 development + 228 evaluation =
  300 documents** across the same five tickers. It made **201 + 627 = 828**
  model-field calls.
- The independent post-cutoff provider benchmark has **300 documents across
  all 30 stocks**. FLAN produced 300 coarse predictions, with 286
  semantic-applicable records and 952 model-field calls. Its GPT reference is
  another 300-document **coarse** silver set; it is not RLLM70.
- The body sensitivity contains **1,675 chunks for those same 300 parent
  articles**. It is not a full-corpus semantic input.
- Full-corpus v2 WLLM17 predictions: **0**.
- Full-corpus v2 RLLM70 predictions: **0**.
- Daily v2 WLLM17 or RLLM70 stock-day rows: **0**.
- Semantic Q+L or Q+D+L training outputs: **0**.

The constructed D2 context is complete for exploratory fitting and matches all
**27,510** quant rows, but it remains ordinary-retrospective and
non-version-safe. The semantic input corpus has the same claim limitation.
Neither input readiness state makes article inference, a daily L panel, or
semantic training complete.

The shared extractor view contains the complete trimmed headline and, for a
nonempty description, two line feeds plus a deterministic leading excerpt of
at most 512 Unicode code points. It prefers a whitespace boundary within the
last 64 code points. Source, retained, and omitted character counts and hashes
are recorded. Every actual FLAN prompt must still pass the pinned tokenizer;
silent runner-side truncation is forbidden. GPT receives the identical bounded
bytes.

## FLAN-T5-XL evidence

The active extractor is pinned to `google/flan-t5-xl` revision
`7d6315df2c2fb742f0f5b556879d730926ca9001`, using CUDA float16 on the
observed NVIDIA GeForce RTX 3070 Ti.

On the frozen 228-document evaluation it achieved:

| Metric | Result |
|---|---:|
| Relevance-gate F1 | 0.9351 |
| Shock-scope macro-F1 | 0.4684 |
| Event-family macro-F1 | 0.4776 |
| Information-status macro-F1 | 0.5797 |
| Directional-alignment macro-F1 | 0.3284 |
| Mean macro-F1 | 0.4635 |

These are agreement scores against GPT silver labels, not human-ground-truth
accuracy. Every semantic field missed its historical frozen threshold. For
the requested exploratory run, the repository therefore freezes an explicitly
permissive silver-only \(q=1\) acceptance profile:
`config/flan_w17_acceptance_v1.json`. It admits valid predictive classes,
treats `unclear`/`other_or_unclear` as abstentions, and keeps
`primary_training_eligible = false` and `confirmatory_eligible = false`.

The full-corpus tokenizer preflight passed all **466,902 assignments** and
**4,202,118 logical prompts**. Target-invariant caching reduced actual
tokenizations to **3,070,758** without changing prompt hashes. There were
**zero** over-limit prompts; maxima were **398** tokens for shock scope,
**427** for event family, and **360** for information status under the
512-token limit. The preflight runtime SHA-256 is
`dfbb8685909d2cd9567e3ca7543dff90f4fd50618855992a469f569ba2415fee`.

The one-record CUDA float16 smoke completed on the RTX 3070 Ti with one
terminal success, no failure, and no truncation. Its prediction hash is
`d1fbc90ba4e612e9c115190b701983845e9f9a4b59591ed6b664c27ed9c6f200`;
the smoke-manifest hash is
`26aa5ade4a01fbde40960a0b617d91f6205ca0f8d034664fc458657357dc1c21`.
Full inference over the 466,902 assignments has not started, and no daily W17
panel exists. FLAN R70 was not pursued and still requires a distinct pinned
fine-schema XL workflow.

## GPT-5.6 Sol evidence and claim boundary

The `.env` file was inspected for variable names only. It contains:

```text
ALPACA_PUBLIC_KEY
ALPACA_SECRET_KEY
ALPHA_VANTAGE_KEY
MASSIVE_API_KEY
MASSIVE_FLAT_KEY
MASSIVE_FLAT_SECRET
```

No values were read, recorded, or exposed. No named OpenAI API credential is
present.

The completed offline R70 workload preflight validated **466,902 assignments**
under two independent views: **933,804 requests**, **3,613,888,330 JSONL
bytes**, **4,458 maximum request bytes**, and **934** conservative files at
1,000 requests per file. Its final hash-bound preflight manifest passed with
SHA-256
`79a8e9aa3e19433d7917371f9e994a2a3ba34f3602da6659a09050c73ab3e96f`
and runtime-contract SHA-256
`8a6aa10f1ecb101024ae112d84e82141dd5d45ebc30926e06a555521d3b6e029`.
The implemented pipeline covers offline preparation, paid-gated
submission, collection, resumable remote-file cleanup, exact-body retries,
disk-backed merge, evidence/entity validation, deterministic adjudication, and
fail-closed 70-feature daily aggregation.

No paid request has been sent, and no full request corpus was materialized
merely to prove the size. A paid pilot or full run requires all of:

- an OpenAI API credential;
- explicit confirmation that the licensed text may be processed;
- a hard maximum paid-request budget; and
- separate user authorization for the paid action.

The user has authorized GPT as an exploratory accuracy reference despite
knowledge-cutoff leakage. Any later GPT W17 or R70 artifact must therefore
carry:

```text
claim_label = exploratory_future_contaminated_oracle
extractor_model_knowledge_contaminated = 1
confirmatory_eligible = 0
```

The existing 300-document fine reference does not replace full-corpus
two-view inference. Without human calibration, GPT R70 uses \(q=1\) and
remains a silver-only oracle. A future GPT W17 panel should be derived from
accepted GPT fine labels; it should not use a second independently prompted
coarse call. GPT W17 was not pursued in this construction pass.

## Remaining execution order

Completed prerequisites:

1. Materialized and hashed strict v2 C/I/P target-article candidates.
2. Materialized one bounded shared headline-plus-description view with
   identical extractor bytes and separate full-source versus visible routing
   metadata.
3. Implemented resumable, hash-bound FLAN W17 and GPT R70 construction paths.
4. Frozen an explicit silver-only FLAN W17 acceptance choice; GPT R70 remains
   \(q=1\) unless human calibration is later supplied.
5. Passed and hash-bound the full FLAN tokenizer preflight and CUDA smoke.
6. Passed and hash-bound the final GPT R70 offline workload preflight.

Remaining:

1. Run one arm at a time over the complete corpus. GPT execution remains
   separately paid-gated.
2. Aggregate complete daily W17/R70 panels. Incomplete source or inference
   coverage invalidates the entire stock-day; it is never a zero.
3. Join to the exact semantic-common Q/D row set and lock features,
   missingness handling, splits, placebos, and protocol hashes.
4. Only then fit the Q+L and Q+D+L ladder. GPT results remain exploratory and
   non-confirmatory regardless of downstream performance.

## Primary evidence

| Artifact | SHA-256 / state |
|---|---|
| `data/features/news_semantic/massive_v2/manifest.json` | `0c37cf044a4e5e9942398f332db4c09f6746ba7cd6a9af2ae2656e3802a30daf` |
| `data/features/news_semantic/massive_v2/article_target_assignments.jsonl.gz` | `52ade4cee38fc56d91556ec0f568f8dfcb16d981102bd23075075735def13d7c` |
| `config/flan_w17_acceptance_v1.json` | `9a1cca6bdf6b0a5bc0c2c5e1f9b6b9dcbb01cca4850e1225aa8449c7650f6072` |
| `data/features/news_semantic/massive_v2/flan_w17/preflight.json` | `39cd7a595451d142bca367e99c000a9768ced8d8784e8158187c6cf811bfb1b7` |
| `data/features/news_semantic/massive_v2/flan_w17/smoke/predictions.jsonl` | `d1fbc90ba4e612e9c115190b701983845e9f9a4b59591ed6b664c27ed9c6f200` |
| `data/features/news_semantic/massive_v2/flan_w17/smoke/predictions.jsonl.manifest.json` | `26aa5ade4a01fbde40960a0b617d91f6205ca0f8d034664fc458657357dc1c21` |
| `data/features/news_semantic/massive_v2/gpt_r70/preflight.json` | `79a8e9aa3e19433d7917371f9e994a2a3ba34f3602da6659a09050c73ab3e96f` |
| `data/features/news_deterministic/massive_v2/manifest.json` | `2d43247e51d9c28e91d9ee70379b33e3ab88400239d862cb72768dcf8a4bc686` |
| `experiments/flan_t5_xl/v1_1/evaluation_summary.json` | `6387c7a8b23feabede33214d17a46908be6ab4b54cd75ebae4b6158d175734da` |
| `annotations/chatgpt_5_6_sol_reference.jsonl` | `94ae2481611fc2a88ce3ff2a16c6c2945c4083be2ef787810c1d1bb1f50967b7` |

Mutable documentation and runner hashes are recorded in `status.json`. The
construction commands and fail-closed completion gates are in
[`CONSTRUCTION_RUNBOOK.md`](CONSTRUCTION_RUNBOOK.md).
