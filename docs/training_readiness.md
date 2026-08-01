# Training and publication status

Status: **the selected research suite is complete and cataloged; prospective
confirmation remains outstanding**.

This page replaces the earlier pre-training checklist. The detailed target,
feature, and fitting specifications are preserved in the versioned experiment
directories and in [`methodology.md`](methodology.md).

## Current model status

| Route | Selected model | Evidence status | Public status |
|---|---|---|---|
| T1 ETF | Long-Q Rung 3 XGBoost | 13-fold retrospective expanding-window result | Active quantitative research model |
| T1 LOO | Long-Q Rung 3 XGBoost | 13-fold retrospective expanding-window result | Active quantitative research model |
| T2 ETF | Long-Q Rung 3 Elastic Net/XGBoost ensemble | 13-fold retrospective expanding-window result | Active quantitative benchmark and fallback |
| T2 ETF semantic | RRES-C6 residual correction | Significant matched-base paired result on five semantic-era folds | Active semantic research overlay |
| T2 LOO | Long-Q Rung 1 core-22 LASSO | 13-fold retrospective expanding-window result | Active quantitative research model |

The machine-readable source of truth is
[`../models/active/registry.json`](../models/active/registry.json).

## Completed quantitative work

The long-Q experiment completed:

- a 58,500-row panel covering 30 stocks and 2,136 represented dates;
- 13 expanding training/validation/test folds;
- 44,058 pooled T1 outer-test rows;
- 42,030 pooled, boundary-purged T2 outer-test rows;
- four model rungs from persistence and linear baselines through XGBoost,
  ensembles, and DCC-GARCH;
- training-only preprocessing and validation-only tuning;
- target-integrity, row-key, timing, and artifact-completeness audits; and
- a target-specific winner rather than one model forced across all targets.

| Target | Winner | Fisher-z RMSE | OOS R² vs persistence |
|---|---|---:|---:|
| T1 ETF | Rung 3 XGBoost | 0.3577 | 0.3921 |
| T1 LOO | Rung 3 XGBoost | 0.3657 | 0.3891 |
| T2 ETF | Rung 3 Elastic Net/XGBoost ensemble | 0.2330 | 0.2200 |
| T2 LOO | Rung 1 core-22 LASSO | 0.2427 | 0.2404 |

The full ranking and exact metrics are in
[`../experiments/quant_training/v2/comparisons/summary.json`](../experiments/quant_training/v2/comparisons/summary.json).

## Completed deterministic and semantic work

The project constructed and evaluated:

- deterministic news counts, routing, recency, coverage, concentration, and
  event-calendar fields;
- normalized D2 features intended to reduce raw provider-volume leakage;
- weak FLAN W17-Lite semantic features;
- Q+L and Q+D+L joint models;
- residual news corrections and direct stacks;
- stale, wrong-stock, date/sector or score-permutation, coverage, quality, and
  text-sensitivity controls; and
- cached-score SoftRoute19 and Coupling6 representations.

The v4 experiment reused 50,488 completed FLAN-T5-XL article score maps,
created 27,510-row role-conditioned daily panels, joined them to the
quantitative features, and trained 20 live/control bundles. All expected
bundle manifests and the 92-row final paired comparison were completed.

## RRES-C6 selection

RRES-C6 uses six predictors: current and prior-only innovation contrasts of
common versus target/peer-idiosyncratic mass for each of three event families.
It corrects saved out-of-sample long-Q XGBoost forecasts with a
validation-selected, shrunk Elastic Net.

For T2 ETF:

| Measure | Result |
|---|---:|
| Incremental Fisher-z MSE R² | **+1.3844%** |
| Paired 95% interval | **[+0.3429%, +2.6228%]** |
| Folds improved | **3/5** |
| Fisher-z RMSE | 0.2278 |
| OOS R² vs persistence | 0.2750 |

The stored matched comparison passed its point-loss, interval, and fold-count
tests. RRES-C6 is therefore selected for the T2 ETF semantic research route.
It is not selected for T1 ETF, T1 LOO, or T2 LOO.

## Archived work

The archive registry covers:

- quant v1, superseded by the longer quant-v2 result;
- nonwinning long-Q target/model combinations;
- deterministic/semantic forecast experiments v1-v3;
- v4 J1, J2, J3, RCAL, RRES-L19, and RSTACK;
- all v4 nuisance and falsification controls; and
- non-T2-ETF RRES-C6 target routes.

Original evidence stays at its existing path. See
[`../models/archive/registry.json`](../models/archive/registry.json).

## Completed verification

The final v4 audit recorded:

- 20 expected bundle manifests;
- 160 declared bundle artifacts passing SHA-256, size, and row checks;
- 44,040 outer-test and 44,640 validation rows in each short bundle;
- 73,800 outer-test and 74,160 validation rows in each long bundle;
- exact R0 identity with the locked quant-v2 XGBoost anchor;
- 92 unique target/comparison rows;
- 16 unique useful-gate records; and
- prior v1-v3 artifacts unchanged under their available hash checks.

At v4 completion, the repository had 457 passing tests. The publication pass
added four active/archive registry consistency tests; the complete current
suite passes **461/461 tests**.

## What remains before confirmation

The suite is ready for public inspection and continued research, but not for a
confirmatory claim. Required next evidence is:

1. a protocol frozen before observing the new outcomes;
2. dates strictly after 2026-06-30;
3. a point-in-time, version-preserving news source;
4. an architecture-matched falsification ladder for RRES-C6; and
5. ideally, validated direction, surprise, materiality, novelty, and
   stock-versus-sector transmission labels.

The current ordinary-news archive lacks first-seen and full version history.
Consequently, all semantic findings remain retrospective even when the paired
statistical test is significant.

## Navigation

- [Project overview](../README.md)
- [Methodology](methodology.md)
- [Detailed results](results.md)
- [Reproducibility](reproducibility.md)
- [Active models](../models/active/README.md)
- [Archived models](../models/archive/README.md)
