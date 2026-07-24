# FLAN-T5-Large v0.5 Legacy Experiment — Rejected

This was the last planned accuracy-improvement experiment on the pinned
`google/flan-t5-large` checkpoint. It was not promoted or frozen.

The candidate replaced direct multiclass questions with ten binary component
scores:

- two compositional scope questions;
- six one-vs-rest event-family questions; and
- two explicit directional-relation questions.

Each question was scored in canonical and reversed Yes/No option order. Scope
and event thresholds were fitted only on the fixed 72-record development
split. The resulting calibration was hash-locked before the 228-record
evaluation inference began. Information status remained the frozen v0.4
prediction.

## Result

The evaluation contained 228 records, including 174 reference-relevant
articles.

| Metric | v0.4 baseline | v0.5 candidate | Change |
|---|---:|---:|---:|
| Mean macro-F1 | 0.4457 | 0.3480 | -0.0978 |
| Mean accuracy | 0.4943 | 0.4210 | -0.0733 |

| Field | v0.4 macro-F1 | v0.5 macro-F1 |
|---|---:|---:|
| Shock scope | 0.3724 | 0.2634 |
| Event family | 0.4523 | 0.3212 |
| Information status | 0.6232 | 0.6232 |
| Directional alignment | 0.3351 | 0.1841 |

The paired 10,000-sample bootstrap favored v0.5 in 0% of samples. Its 95%
percentile interval for the mean macro-F1 difference was approximately
[-0.1388, -0.0539].

Only two promotion checks passed:

- all outputs were valid and non-truncated; and
- mixed-scope F1 was nonzero.

All accuracy, macro-F1, field-loss, target-field-gain, and bootstrap checks
failed. The local artifact finalizer originally placed the complete run in the
rejected namespace. It was subsequently archived with the other historical
runs now archived at `outputs/flan_t5/archive/v0_5/` and did not create
`outputs/flan_t5/active_frozen.json`.

## Audit files

- [Frozen protocol](PROTOCOL.md)
- [Preregistered promotion rule](promotion_rule.json)
- [Pre-score source and split lock](protocol_lock.json)
- [Compact evaluation result](evaluation_summary.json)
- [Active v0.4 research baseline](../../v0_4/README.md)

Implementation:

- [`implementation/config/flan_t5_final_candidate_v0_5.json`](implementation/config/flan_t5_final_candidate_v0_5.json)
- [`implementation/scripts/extract_flan_t5_final_candidate.py`](implementation/scripts/extract_flan_t5_final_candidate.py)
- [`implementation/scripts/select_flan_t5_final_candidate.py`](implementation/scripts/select_flan_t5_final_candidate.py)
- [`implementation/scripts/lock_flan_t5_final_candidate.py`](implementation/scripts/lock_flan_t5_final_candidate.py)
- [`implementation/scripts/finalize_flan_t5_candidate.py`](implementation/scripts/finalize_flan_t5_candidate.py)

The implementation and its tests are retained as historical source snapshots,
not operational entry points. Paths recorded inside `protocol_lock.json`
describe their original pre-score locations and are intentionally unchanged.

The 228-record evaluation split had already informed the diagnosis of v0.4.
This result is therefore an engineering comparison, not a pristine scientific
holdout. A future extractor should be evaluated on a new human-audited set.
