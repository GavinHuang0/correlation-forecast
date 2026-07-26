# Llama 3.1 v1.0 Development Result

This version is a completed development-screen experiment, not an active
extractor.

The pinned `meta-llama/Llama-3.1-8B-Instruct` snapshot was fully cached and
hash-verified offline. The 72 development articles were then processed with
the official chat template, exact `Answer:` assistant boundary, NF4/FP16
inference, and every cyclic option rotation. No input was truncated.

## Result

| Field | Accuracy | Macro-F1 | Required | Passed |
|---|---:|---:|---:|---|
| Shock scope | 0.518 | 0.177 | 0.70 | No |
| Event family | 0.625 | 0.590 | 0.65 | No |
| Information status | 0.661 | 0.540 | 0.70 | No |
| Directional alignment | 0.500 | 0.240 | 0.65 | No |
| **Mean** | **0.576** | **0.387** | — | — |

The deterministic relevance gate passed at 0.932 F1. The semantic extractor
failed all four field thresholds and was worse overall than active
FLAN-T5-XL v1.1 on the same development split (0.472 mean macro-F1).

The principal failure was label collapse:

- 52 of 56 relevant records were classified as `idiosyncratic`;
- no record was classified as `mixed`; and
- 48 of 56 alignment labels became `single_firm_only`.

Event-family extraction was the strongest component, but it still missed its
0.65 threshold.

## Decision

v1.0 is rejected as an all-field extractor. The fixed 228-document evaluation
split was not materialized, preserving it for a future preregistered
configuration.

A quick score-only calibration was not applied. Scope determines whether the
alignment prompt runs, so changing scope scores after inference would make the
stored alignment decision internally inconsistent. A valid follow-up would
need a new version with its scope decision rule frozen before extraction.

The active research path remains
[`../../../active_extractor.json`](../../../active_extractor.json).
Exact development metrics and ignored local-artifact hashes are in
[`development_summary.json`](development_summary.json).
