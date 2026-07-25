# Quant correlation training v1

This directory contains the human-readable and machine-readable audit trail
for the locked ETF/LOO T1/T2 training ladder. Generated predictions and fit
metadata live under `outputs/quant_training/v1`; reproducible summaries live
here.

The initial model families, folds, and feature blocks were locked in
`config/quant_training_protocol_v1.json` before outer-test results were
inspected. Every model uses date-blocked folds, and T2 rows are purged whenever
their five-session target crosses a train, validation, or test boundary.

Protocol v1.1 records one documentation amendment: the validation-only
regularization-grid boundary expansion had been specified in the design audit
but omitted from the first JSON transcription. It was restored without using
outer-test loss and does not change a model family or feature block.

Directory layout:

```text
construction/
  targets/audit.json
  audits/target_integrity.json
  audits/modeling_panel.json
rung_01/                  # persistence, HAR/SHAR, core OLS/LASSO
  t1_etf/ ... t2_loo/
rung_02/                  # nested practical LASSO/Elastic Net
  t1_etf/ ... t2_loo/
rung_03/                  # GPU XGBoost and eligible ensembles
  t1_etf/ ... t2_loo/
rung_04/                  # causal bivariate DCC-GARCH
  t1_etf/ ... t2_loo/
comparisons/
  summary.json
  README.md

outputs/quant_training/v1/
  rung_01/ ... rung_04/
    predictions.parquet
    fold_metrics.json
    fits.json
```

Each rung contains a machine-readable `review.json`, a compact `summary.json`,
and target-specific `metrics.json` files. Larger fold-level predictions and
fit records are intentionally kept under `outputs/quant_training/v1`.

## Target construction

- T1 is same-day regular-session non-demeaned realized correlation.
- T2 sums covariance and variance components over the current and next four
  official sessions before normalizing.
- ETF and LOO targets share the exact same official interval set.
- LOO is an interval-rebalanced equal-weight basket of the other five peers in
  simple-return space.
- The construction audit passed with zero numerical reconstruction error.

In prediction files, `target` identifies ETF versus LOO. The generic
`benchmark` column always retains the sector ETF ticker as sector metadata; it
does not imply that an LOO row was forecast against that ETF.

The complete mathematical contract—including interval returns, ETF/LOO
targets, persistence, feature equations, linear and tree objectives,
DCC-GARCH recursions, fold eligibility, and evaluation formulas—is in
[`docs/training_readiness.md`](../../../docs/training_readiness.md). Feature-
specific derivations are in
[`docs/bollerslev_core_features.md`](../../../docs/bollerslev_core_features.md)
and
[`docs/additional_quant_features.md`](../../../docs/additional_quant_features.md).

## Best development results

| Target | Model | Fisher-z RMSE | Raw RMSE | OOS R² vs persistence |
|---|---|---:|---:|---:|
| T1 ETF | Rung 3 `elastic_xgboost_ensemble` | 0.3686 | 0.2073 | 0.4154 |
| T1 LOO | Rung 3 XGBoost | 0.3814 | 0.2291 | 0.3964 |
| T2 ETF | Rung 3 XGBoost | 0.2348 | 0.1248 | 0.2885 |
| T2 LOO | Rung 3 XGBoost | 0.2567 | 0.1488 | 0.2517 |

CUDA was confirmed for every final XGBoost fit. DCC-GARCH converged for all
180 stock/benchmark/fold systems, but it was weaker than the supervised ladder
and fell below persistence for both T2 targets. The outer blocks were used for
cross-rung ranking, so the winning metrics are development estimates; a future
untouched period is required for confirmation. See
[`comparisons/README.md`](comparisons/README.md) for the rung-by-rung table and
interpretation limits.
