# Stock-sector-ETF convergence backtest on quant full-history v2

Status: **complete; the strategy fails the economic backtest and should not be
deployed**.

This run applies the unchanged
[stock-ETF convergence v1 protocol](../v1/README.md) to the declared T2 ETF
winner from the
[full-history quant v2 ladder](../../quant_training/v2/comparisons/README.md):
rung-3 `elastic_xgboost_ensemble`.

The new protocol is
[`config/stock_etf_spread_backtest_quant_v2.json`](../../../config/stock_etf_spread_backtest_quant_v2.json).
It changes only the forecast experiment, selected model, outer-test sample,
and output namespace. Signal construction, gates, portfolio weights,
five-session RTH-only sleeves, costs, controls, and inference are identical to
v1. The v2 protocol was frozen before the v2 strategy returns were inspected.

Reporting amendment: the `balanced_2023_plus` and `matched_recent` fold lists
were declared from the coverage audit before P&L was inspected, then
transcribed into the protocol as reporting-only metadata after the primary
run. No signal, portfolio, cost, control, or inference setting changed; the
subsample tables were recomputed from the frozen daily-return artifact.

## Execution

1. Verified the quant-v2 pooled winner against the comparison manifest.
2. Selected exactly 42,030 fold-OOS `t2_etf` ensemble forecasts.
3. Hash-verified all 35 stock/ETF bar histories and the official calendar.
4. Rebuilt strictly complete regular-session returns and backward-only
   five-session divergence signals.
5. Constructed sector-neutral stock books with beta-offsetting ETF legs.
6. Ran five overlapping RTH-only sleeves and the frozen 0/1/2/5 bp cost
   ladder.
7. Repeated the unconditional, persistence-level, ML-level, strengthening,
   and weakening gates.
8. Ran Newey-West and fold-contained paired moving-block inference.
9. Reported the canonical 13-fold result, a fully balanced 2023+ sample, and a
   matched folds-11–13 comparison with v1.

All steps completed without an execution-data or integrity failure.

## Sample and coverage

- 13 half-year outer-test folds from 2020-H1 through 2026-H1;
- 1,579 forecast dates and 1,631 trading dates after sleeves finish;
- 30 stocks and five sector ETFs;
- 42,030 unique stock-date forecasts;
- zero missing returns across the 245,175 required stock/ETF holding-window
  observations;
- no sleeve crosses a fold boundary.

The early modeling panel is not cross-sectionally balanced. Of 1,579 forecast
dates, 1,342 contain all 30 stocks and 215 contain only six. Fold 5 is the
most severe case: after 2022-01-14, only the six semiconductor stocks remain
eligible. Folds 7–13, beginning in 2023, have all 30 stocks on every date and
are reported separately.

## Full 13-fold result

### Primary `ml_strengthening` strategy

| Cost per side | Ann. arithmetic return | Ann. volatility | Sharpe | Compounded return | Max drawdown | Ann. modeled cost | NW t-stat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 bp | -2.73% | 4.52% | -0.60 | -16.77% | -25.04% | 0.00% | -1.70 |
| 1 bp | -6.26% | 4.52% | -1.38 | -33.74% | -38.89% | 3.52% | -3.87 |
| **2 bp** | **-9.78%** | **4.53%** | **-2.16** | **-47.26%** | **-50.47%** | **7.05%** | **-6.01** |
| 5 bp | -20.35% | 4.54% | -4.48 | -73.40% | -73.77% | 17.62% | -12.18 |

The strategy loses money before transaction costs. There is therefore no
positive full-sample break-even trading-cost estimate.

### Controls at 2 bp per side

| Gate | Ann. arithmetic return | Sharpe | Max drawdown |
|---|---:|---:|---:|
| Unconditional | -10.56% | -2.46 | -52.20% |
| Historical level | -10.27% | -2.37 | -50.63% |
| ML level | -10.89% | -2.46 | -53.36% |
| **ML strengthening** | **-9.78%** | **-2.16** | **-50.47%** |
| ML weakening | -10.14% | -2.44 | -49.46% |

The primary gate improves annualized mean return over unconditional
divergence by only 0.78 percentage points. The fold-contained 95% bootstrap
interval is -1.42 to 3.15 percentage points, with a 74.95% probability that
the gated return is higher. The incremental effect is not resolved and both
strategies lose heavily.

## Balanced 2023+ sensitivity

Folds 7–13 contain all 30 stocks on every forecast date:

| Cost per side | Ann. arithmetic return | Sharpe | Max drawdown |
|---:|---:|---:|---:|
| 0 bp | 0.35% | 0.09 | -8.75% |
| 1 bp | -3.37% | -0.87 | -17.91% |
| **2 bp** | **-7.10%** | **-1.83** | **-26.60%** |
| 5 bp | -18.29% | -4.69 | -47.79% |

Removing the early one-sector dates eliminates the negative gross return but
does not produce a usable edge. At zero cost, unconditional divergence earns
1.20% annualized versus 0.35% for the ML-strengthening gate. At 2 bp, they
return -6.12% and -7.10% respectively. The gate therefore does not add value
on the balanced sample.

## Fold stability

| Fold | Outer test | Gross ann. return | Net ann. return at 2 bp |
|---|---|---:|---:|
| 01 | 2020-H1 | -12.39% | -19.76% |
| 02 | 2020-H2 | -12.25% | -19.88% |
| 03 | 2021-H1 | -5.10% | -11.66% |
| 04 | 2021-H2 | -6.92% | -13.92% |
| 05 | 2022-H1 | -4.34% | -7.94% |
| 06 | 2022-H2 | 3.18% | -4.00% |
| 07 | 2023-H1 | -8.93% | -16.45% |
| 08 | 2023-H2 | -1.26% | -8.78% |
| 09 | 2024-H1 | 1.46% | -6.01% |
| 10 | 2024-H2 | -0.53% | -8.12% |
| 11 | 2025-H1 | -8.49% | -15.50% |
| 12 | 2025-H2 | 3.08% | -4.48% |
| 13 | 2026-H1 | 17.10% | 9.58% |

Only four of 13 folds are positive before costs, and only fold 13 remains
positive at 2 bp per side.

## Matched comparison with v1

Folds 11–13 contain exactly the same 10,830 stock-date keys as the original
v1 backtest. Unconditional and historical-level strategy returns reproduce v1
exactly, which verifies that prices, divergence, fold handling, and execution
are unchanged. Differences in the ML-gated strategies therefore come from
the new predictions.

| Primary strategy on 2025-H1–2026-H1 | Gross ann. return | Net ann. return at 2 bp | Net Sharpe |
|---|---:|---:|---:|
| v1 T2 ETF XGBoost | 7.96% | 0.77% | 0.18 |
| v2 full-history ensemble | 3.92% | -3.45% | -0.84 |
| **v2 minus v1** | **-4.04 pp** | **-4.22 pp** | — |

The paired 10-session block-bootstrap interval for the annualized v2-minus-v1
net-return difference is **-8.04 to -0.42 percentage points**; only 1.55% of
resamples favor v2. The result compares the selected v2 ensemble with the
selected v1 XGBoost model, so it combines the effects of longer training
history and the change in selected estimator.

## Interpretation

The convergence strategy is rejected in this expanded experiment:

- the base divergence return is negative over the complete period;
- the balanced 2023+ gross edge is approximately zero;
- transaction costs are about 7% per year at the primary assumption;
- the ML-strengthening gate has no resolved incremental value;
- the new selected model makes the matched recent trading result worse;
- 12 of 13 folds lose money after costs.

Forecast-error ranking and trading utility are not interchangeable. The v2
ensemble is the best T2 ETF correlation forecaster under pooled Fisher-z
RMSE, but that improvement does not identify when a stock/ETF return spread
will converge profitably.

The early sparse panel also produces excessive concentration: the largest
absolute daily stock weight is 47.83% of reference capital, versus 37.04% even
in the balanced 2023+ period. The frozen protocol has no single-name cap.

Risk targeting cannot repair negative gross expectancy and would magnify
costs and concentration. Reversing the signal after observing this result
would be data mining. Any next experiment should instead be separately
pre-registered around a direct expected-spread-return target, lower turnover,
position caps, and a genuinely future holdout.

This remains retrospective development evidence. Pooled outer-test forecast
loss selected the v2 model, v1 influenced the reused ladder, folds 11–13 were
already examined in v1, the universe is static, and no untouched holdout
remains.

## Reproduction and artifacts

Run:

```powershell
.\.venv-training\Scripts\python.exe -m scripts.stock_etf_spread_backtest `
  --protocol config\stock_etf_spread_backtest_quant_v2.json `
  --verify-input-hashes

.\.venv-training\Scripts\python.exe -m unittest `
  tests.test_stock_etf_spread_backtest -v
```

The compact result is in [`summary.json`](summary.json). Larger artifacts are
under `outputs/stock_etf_spread/v2/`:

- `signals.parquet`;
- `sleeve_positions.parquet`;
- `daily_positions.parquet`;
- `daily_returns.parquet`;
- `metrics.parquet`;
- `paired_bootstrap.parquet`;
- `subsample_metrics.parquet`;
- `subsample_paired_bootstrap.parquet`;
- `summary.json`.
