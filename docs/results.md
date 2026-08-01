# Results and model-selection record

## Selected suite

The selected suite is defined in
[`../models/active/registry.json`](../models/active/registry.json). It contains
four primary quantitative routes and one T2 ETF semantic overlay.

### Long-Q winners

| Target | Rung/model | Rows | Fisher-z MAE | Fisher-z RMSE | Raw MAE | Raw RMSE | OOS R² |
|---|---|---:|---:|---:|---:|---:|---:|
| T1 ETF | Rung 3 XGBoost | 44,058 | 0.2837 | 0.3577 | 0.1322 | 0.1894 | 0.3921 |
| T1 LOO | Rung 3 XGBoost | 44,058 | 0.2898 | 0.3657 | 0.1506 | 0.2118 | 0.3891 |
| T2 ETF | Rung 3 Elastic Net/XGBoost ensemble | 42,030 | 0.1834 | 0.2330 | 0.0866 | 0.1180 | 0.2200 |
| T2 LOO | Rung 1 core-22 LASSO | 42,030 | 0.1912 | 0.2427 | 0.1020 | 0.1382 | 0.2404 |

The longer history changed the T2 LOO selection from the shorter-history
XGBoost result to core-22 LASSO. On the common recent 2025-H1 through 2026-H1
periods, adding earlier training history improved XGBoost RMSE by 0.75% for
T1 ETF, 1.35% for T1 LOO, and 2.85% for T2 LOO; T2 ETF XGBoost worsened by
0.48%, while the qualified ensemble approximately matched the older result.

Rung 4 completed all 780 dense-history GARCH/DCC systems with no failures and
stationary fitted DCC parameters, but it did not beat the selected supervised
models.

### RRES-C6 T2 ETF

| Statistic | Value |
|---|---:|
| Rows / dates | 18,150 / 605 |
| Fisher-z MAE / RMSE | 0.1791 / 0.2278 |
| Raw-correlation MAE / RMSE | 0.0929 / 0.1251 |
| OOS R² vs persistence | 0.2750 |
| Matched-base incremental MSE R² | **+1.3844%** |
| Paired moving-block 95% interval | **[+0.3429%, +2.6228%]** |
| Folds improved | **3/5** |
| Bootstrap probability gain $\le 0$ | 0.45% |

The mean improvement was uneven across folds: approximately +0.317%,
-0.152%, 0.000%, +5.182%, and +2.150%. The aggregate paired result passed,
but much of the effect came from the final two folds.

RRES-C6 outcomes on the other targets were not selected:

| Target | Incremental MSE R² vs matched long-Q anchor | 95% interval | Folds improved | Decision |
|---|---:|---:|---:|---|
| T1 ETF | +0.1146% | [-0.0690%, +0.3383%] | 2/5 | Archive target route |
| T1 LOO | -0.2545% | [-0.5134%, +0.0580%] | 0/5 | Archive target route |
| T2 ETF | **+1.3844%** | **[+0.3429%, +2.6228%]** | **3/5** | **Promote overlay** |
| T2 LOO | -0.9851% | [-2.3553%, +0.1826%] | 1/5 | Archive target route |

## Newest Q+L experiment

V4 trained 20 live and control bundles after reusing 50,488 cached FLAN score
maps. The primary matched-base results were:

| Candidate | T1 ETF | T1 LOO | T2 ETF | T2 LOO |
|---|---:|---:|---:|---:|
| J1: Q56 + SoftRoute19 | -0.2183% | +0.1740% | +0.7110% | **-1.3561%** |
| J2: Q56 + D2 + SoftRoute19 | -0.0099% | +0.1943% | +0.3003% | **-1.2908%** |
| J3: Q56 + current soft masses | -0.1491% | +0.2266% | **+0.6195%** | +0.1904% |
| RRES-C6 | +0.1146% | -0.2545% | **+1.3844%** | -0.9851% |
| RRES-L19 | +0.0935% | +0.0461% | +0.9342% | -0.1121% |
| RSTACK | -0.5438% | **-1.4675%** | -0.3566% | **-6.4956%** |

Values are incremental Fisher-z MSE R² versus the matched quantitative base;
positive is better. Bold cells have paired 95% intervals excluding zero in
the indicated direction.

J1, J2, and RRES-L19 each had architecture-matched stale, wrong-stock,
probability-permutation, and quality-only comparisons. No target passed every
required comparison. Important failures include:

- J1 T2 ETF was 1.3219% worse than its stale-20 control.
- Quality-only controls beat several LOO live models, showing that extraction
  confidence and coverage could be as predictive as event identity.
- Live/permuted and live/wrong-stock comparisons did not consistently favor
  correctly assigned current semantics.
- RRES-L19's +0.9342% T2 ETF estimate had a 95% interval crossing zero.
- RSTACK worsened all four targets.

J3 T2 ETF produced a significant +0.6195% estimate with interval
[+0.0013%, +1.3428%] and two of three folds improved, but J3 was a diagnostic
without an architecture-matched control set. It remains archived.

## Why only RRES-C6 was promoted

RRES-C6 is the only v4 candidate/target pair that met all three tests in its
stored matched-base comparison:

1. favorable point loss;
2. paired upper confidence bound below zero loss difference; and
3. improvement in at least three of five folds.

Its low dimensionality is also consistent with the observed data limitations:
the full semantic representation is sparse, uncertain, and collinear. The
promotion does not erase those limitations, so the active registry records it
as a semantic research overlay rather than a confirmed primary model.

## Archive decision

All other forecast candidates are archived by status in
[`../models/archive/registry.json`](../models/archive/registry.json). Their
experiment records, controls, and negative results remain at their original
paths so hashes and citations continue to resolve.

## Evidence files

- Long-Q result narrative:
  [`../experiments/quant_training/v2/README.md`](../experiments/quant_training/v2/README.md)
- Long-Q complete ranking:
  [`../experiments/quant_training/v2/comparisons/summary.json`](../experiments/quant_training/v2/comparisons/summary.json)
- V4 training report:
  [`../experiments/quant_deterministic_news/v4/training/README.md`](../experiments/quant_deterministic_news/v4/training/README.md)
- V4 complete paired table:
  [`../experiments/quant_deterministic_news/v4/training/comparisons/final/RESULTS.md`](../experiments/quant_deterministic_news/v4/training/comparisons/final/RESULTS.md)
