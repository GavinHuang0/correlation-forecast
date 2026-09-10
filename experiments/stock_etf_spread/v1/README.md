# Stock-sector-ETF convergence backtest v1

Status: **complete exploratory development backtest; not ready for live
trading**.

This experiment converts the existing five-session stock-sector-ETF
correlation forecasts into a gate for a separate return-convergence signal.
The protocol in
[`config/stock_etf_spread_backtest_v1.json`](../../../config/stock_etf_spread_backtest_v1.json)
was frozen before strategy returns were inspected.

## Execution checklist

1. **Lock the test.** Select the fold-OOS T2 ETF XGBoost forecasts, define the
   signal, controls, execution convention, cost ladder, and inference method.
2. **Build point-in-time signals.** Reconstruct exact official regular-session
   returns, then estimate beta and five-session divergence using data through
   the prior close only.
3. **Apply the forecast gate.** Require predicted correlation of at least 0.50
   and a positive change relative to persistence; retain unconditional and
   alternative gates as controls.
4. **Construct executable spreads.** Make stock weights neutral within sector,
   add the corresponding beta-offsetting ETF leg, equalize active sectors, and
   normalize every new sleeve to 1.0 combined gross.
5. **Align trading with T2.** Run five overlapping sleeves at 20% allocation
   each, entering at the official open and exiting at the official close on
   every holding-window session. No overnight exposure is used.
6. **Charge costs and test robustness.** Evaluate 0, 1, 2, and 5 basis points
   per side, report fold results, use Newey-West statistics, and run a
   fold-contained paired moving-block bootstrap.
7. **Audit and reproduce.** Verify source-bar hashes, reject incomplete
   official sessions, test lagging and portfolio invariants, and save the full
   signal, position, return, metric, and bootstrap artifacts.

All seven steps are complete.

## Frozen strategy

For stock $`i`$, sector ETF $`E`$, and forecast date $`t`$, the pre-open
signals are

```math
\beta_{i,t}
=
\frac{\mathrm{Cov}_{252}(r_i,r_E)}
     {\mathrm{Var}_{252}(r_E)},
\qquad
D_{i,t}
=
\beta_{i,t}\sum_{s=t-5}^{t-1}r_{E,s}
-
\sum_{s=t-5}^{t-1}r_{i,s},
```

where beta uses at least 126 paired daily observations, is clipped to
$`[0,3]`$, and every input ends at $`t-1`$. The primary gate is

```math
\widehat{\rho}^{T2}_{i,t} \ge 0.50,
\qquad
G_{i,t}
=
\widehat{\rho}^{T2}_{i,t}
-
\rho^{\mathrm{persistence}}_{i,t}
>0.
```

Eligible stocks are ranked by $`D`$: recent underperformers are long and
recent outperformers are short. A percentile rank of $`G`$ continuously
scales conviction. The stock scores are demeaned within sector, and the ETF
leg is the negative beta-weighted sum of the stock legs. This makes the stock
book sector-dollar-neutral and the combined stock/ETF book approximately
sector-beta-neutral. A residual ETF dollar balance is financed by cash, which
is excluded from risky-asset gross exposure and earns zero in this backtest.

The controls are:

- `unconditional`: divergence without a correlation gate;
- `historical_level`: persistence correlation at least 0.50;
- `ml_level`: predicted correlation at least 0.50;
- `ml_strengthening`: the primary level-plus-positive-change gate;
- `ml_weakening`: predicted correlation at least 0.50 but negative change.

## Data coverage

- 30 stocks and five sector ETFs;
- 361 forecast dates from 2025-01-02 through 2026-06-24;
- three fold-OOS development blocks;
- 373 trading dates after allowing end-of-fold sleeves to finish;
- 10,830 unique stock-date forecasts and 54,150 variant-annotated signal
  records.

The primary gate admits 49.57% of stock-date observations and produces a
tradable sleeve on 344 of 361 forecast dates.

The saved fold-OOS artifact contains forecasts only where the full T2 label
stays inside its fold. Consequently, no new sleeve starts during the four
purged sessions at the end of folds 1 and 2. A sleeve already opened on the
last eligible date is allowed to finish, is flat by the boundary close, and
no position crosses into the next fold. A live test should instead generate
label-free forecasts on every feature-eligible date.

## Results

### Primary `ml_strengthening` cost sensitivity

| Cost per side | Ann. arithmetic return | Ann. volatility | Sharpe | Compounded return | Max drawdown | Ann. modeled cost | NW t-stat |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 bp | 7.96% | 4.27% | 1.86 | 12.35% | -3.40% | 0.00% | 2.21 |
| 1 bp | 4.37% | 4.27% | 1.02 | 6.53% | -4.61% | 3.60% | 1.21 |
| **2 bp** | **0.77%** | **4.27%** | **0.18** | **1.01%** | **-7.20%** | **7.19%** | **0.21** |
| 5 bp | -10.02% | 4.28% | -2.34 | -13.90% | -16.80% | 17.98% | -2.77 |

The gross edge breaks even at about 2.21 bp per side under the linear cost
assumption. Mean deployed gross is 0.714 because overlapping sleeves can
partially offset; entering and closing every active position still produces
about 360 times capital of modeled annual round-trip turnover.

### Primary-cost comparison

At the frozen 2 bp-per-side assumption:

| Gate | Ann. arithmetic return | Sharpe | Max drawdown |
|---|---:|---:|---:|
| Unconditional | -3.01% | -0.93 | -8.87% |
| Historical level | -3.94% | -1.20 | -8.89% |
| ML level | -3.41% | -1.03 | -9.96% |
| **ML strengthening** | **0.77%** | **0.18** | **-7.20%** |
| ML weakening | -5.44% | -1.34 | -10.35% |

The primary gate improves annualized mean return over the unconditional
strategy by 3.78 percentage points. Its fold-contained 95% moving-block
bootstrap interval is -0.86 to 9.77 percentage points, so the incremental
effect is not statistically resolved. Every other paired comparison also has
a confidence interval spanning zero.

### Fold stability at 2 bp per side

| Fold | Ann. arithmetic return | Sharpe | Max drawdown |
|---|---:|---:|---:|
| 1 | -8.08% | -2.14 | -5.00% |
| 2 | -1.45% | -0.38 | -2.39% |
| 3 | 11.85% | 2.36 | -1.59% |

Only the third fold is profitable after the primary cost assumption. The
pooled positive return is therefore regime-dependent rather than stable
evidence of a deployable strategy.

## Interpretation

The forecast gate is directionally useful: before costs, `ml_strengthening`
has a 1.86 Sharpe versus 1.33 for unconditional divergence, while
`ml_weakening` falls to 0.47. That ordering is consistent with the intended
economic mechanism.

It does not yet survive the implementation burden robustly. The primary net
Sharpe is 0.18, the Newey-West t-stat is 0.21, the paired confidence intervals
include zero, and two of three folds lose money. Short availability, fees,
spread variation, and market impact are not separately modeled, so the 2 bp
case should not be treated as conservative proof of profitability.

The frozen constructor also has no single-name cap. The largest absolute
daily stock weight is 26.63% of reference capital, which is another reason to
avoid interpreting v1 as a production portfolio.

Risk targeting would rescale exposure but would not repair this weak,
cost-sensitive Sharpe. It should be added only in a separately frozen
experiment after a lower-turnover trading rule or a direct forecast of
expected spread return establishes positive net edge.

The model family was selected using the same outer periods represented here.
These results are therefore post-selection development evidence, not an
untouched confirmation. Preserve v1 unchanged and require a future
pre-registered period before making a trading claim.

The fixed 30-stock universe is not effective-dated, so membership and
survivorship bias remain. Because the sector ETFs themselves hold the stocks,
the result is evidence about convergence against a tradable hedge, not an
independent sector-factor effect.

## Reproduction and artifacts

Run:

```powershell
.\.venv-training\Scripts\python.exe -m scripts.stock_etf_spread_backtest --verify-input-hashes
.\.venv-training\Scripts\python.exe -m unittest tests.test_stock_etf_spread_backtest -v
```

The compact machine-readable result is in
[`summary.json`](summary.json). Larger artifacts are under
`outputs/stock_etf_spread/v1/`:

- `signals.parquet`;
- `sleeve_positions.parquet`;
- `daily_positions.parquet`;
- `daily_returns.parquet`;
- `metrics.parquet`;
- `paired_bootstrap.parquet`;
- `summary.json`.
