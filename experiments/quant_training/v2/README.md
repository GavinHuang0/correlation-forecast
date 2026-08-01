# Quant correlation training v2: full price history

This experiment reuses the completed v1 quant-only targets, feature blocks,
model families, tuning grids, and four-rung ladder while extending the
expanding-window evaluation backward across the available price history.
It is isolated from v1 under:

```text
config/quant_training_protocol_v2.json
data/features/quant/training_v2/
experiments/quant_training/v2/
outputs/quant_training/v2/
```

The raw price archive begins in 2016. The unchanged core feature contract uses
a 500-session window, requires at least 490 valid sessions, and retains at
least 99 percent of exponential weight. Consequently, 2016 prices provide
warm-up history and the first model-ready row is 2017-12-28.

The protocol fixes 13 expanding folds before inspecting v2 results. Validation
and test blocks remain six calendar months each. Outer tests run from 2020-H1
through 2026-H1, and the final three folds exactly retain the v1 validation and
test periods while adding the earlier training history.

This is a retrospective development and robustness study, not an untouched
confirmation. The ladder was selected during v1, folds 11 through 13 reuse the
v1 outer periods, and aggregate outer-test loss is used to compare v2 rungs.

The historical complete-core filter can make the early panel cross-sectionally
unbalanced when one or more configured stocks lack a complete feature row.
Every trained model continues to use the same v1 information-timing, target-end
purging, training-only preprocessing, validation-only tuning, and output audit
contracts.

The 14 substantive extended-hours features begin on 2022-11-01 (the prior
aftermarket series begins one session later). In the early folds those columns
are retained but entirely missing, so training-fold imputation leaves them
with zero influence; the associated availability flags remain explicit. A
fold cannot learn an extended-hours effect until those values exist in its
training block.

Rung 4 estimates and filters GARCH/DCC states on the separate dense daily
target/history panel beginning in 2016, including expected state updates on
missing-return sessions. Forecasts are then joined onto the filtered modeling
panel for evaluation. This prevents the early core-feature gaps from being
mistaken for consecutive daily observations.

## Reproduction

```powershell
$python = '.\.venv-training\Scripts\python.exe'
$protocol = 'config\quant_training_protocol_v2.json'

& $python -m scripts.correlation_training.build_modeling_panel `
  --core data\features\quant\training_v1\bollerslev_core_features.parquet `
  --context data\features\quant\training_v1\additional_quant_features.parquet `
  --targets data\features\quant\training_v1\correlation_targets_and_loo_features.parquet `
  --output data\features\quant\training_v2\modeling_panel.parquet `
  --start 2017-12-28 --end 2026-06-30 `
  --audit-output experiments\quant_training\v2\construction\audits\modeling_panel.json

& $python -m scripts.correlation_training.run_rung_01 --protocol $protocol
& $python -m scripts.correlation_training.run_rung_02 --protocol $protocol
& $python -m scripts.correlation_training.run_rung_03 --protocol $protocol --device cuda
& $python -m scripts.correlation_training.run_rung_04 --protocol $protocol
& $python -m scripts.correlation_training.summarize_ladder --protocol $protocol
```

<!-- RESULTS_START -->

The complete ladder passed. The modeling panel contains 58,500 stock-date
rows, 30 stocks, and 2,136 represented dates from 2017-12-28 through
2026-06-30. The 13 non-overlapping outer tests contain 44,058 T1 rows and
42,030 boundary-purged T2 rows.

| Target | Best full-history rung/model | Fisher-z RMSE | Raw RMSE | OOS R² vs persistence |
|---|---|---:|---:|---:|
| T1 ETF | Rung 3 XGBoost | 0.3577 | 0.1894 | 0.3921 |
| T1 LOO | Rung 3 XGBoost | 0.3657 | 0.2118 | 0.3891 |
| T2 ETF | Rung 3 Elastic Net/XGBoost ensemble | 0.2330 | 0.1180 | 0.2200 |
| T2 LOO | Rung 1 core-22 LASSO | 0.2427 | 0.1382 | 0.2404 |

The longer evaluation changes one headline relative to v1: a simple core
LASSO beats XGBoost for T2 LOO when all 13 outer blocks are pooled. XGBoost
remains best for both T1 targets. The validation-qualified ensemble enters
only for T2 ETF.

On the identical 2025-H1 through 2026-H1 outer periods, comparing XGBoost with
XGBoost isolates the effect of adding the earlier training history:

| Target | v1 XGBoost RMSE | v2 longer-history XGBoost RMSE | Relative change | v2 OOS R² |
|---|---:|---:|---:|---:|
| T1 ETF | 0.3700 | 0.3672 | -0.75% | 0.4199 |
| T1 LOO | 0.3814 | 0.3763 | -1.35% | 0.4126 |
| T2 ETF | 0.2348 | 0.2359 | +0.48% | 0.2817 |
| T2 LOO | 0.2567 | 0.2493 | -2.85% | 0.2938 |

For T2 ETF, the v2 qualified ensemble scores 0.2348 on those same recent
blocks, effectively level with the old v1 XGBoost result. These comparisons
remain development evidence: v1 influenced the ladder, and the recent outer
periods are reused.

Rung 3 used XGBoost 3.3.0 on CPU because a pre-existing user workload occupied
the GPU; all candidate configurations, seeds, early stopping, feature blocks,
and ensemble gates were unchanged. Rung 4 completed all 780 dense-history
GARCH/DCC systems with zero failures and stationary fitted DCC parameters, but
it was weaker than the supervised ladder.

Every rung review is `passed`, the target-integrity audit has zero numerical
error, and the comparison summary requires every expected artifact before it
can report completion. The pre-existing v1 artifacts remained byte-identical:
the aggregate hashes stayed
`da98bdf5ec1e9c61164434b960488625a549cfadc9f2cd678cff779cc12a85a2`
for `outputs/quant_training/v1` and
`f85a59d6439299bbbeaf9aca54d49af6d5f73510b6a429284514e48c5dba3b33`
for `experiments/quant_training/v1`.

See [`comparisons/README.md`](comparisons/README.md) for interpretation and
the recent-period comparison, and [`comparisons/summary.json`](comparisons/summary.json)
for the complete machine-readable ranking and artifact hashes.

<!-- RESULTS_END -->
