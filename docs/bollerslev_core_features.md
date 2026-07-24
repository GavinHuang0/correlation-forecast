# Bollerslev core correlation features

`scripts/build_bollerslev_core_features.py` is an offline, daily
stock-sector adaptation of the price-derived features in Bollerslev, Li, and
Tang, *Forecasting and Managing Correlation Risks* (Management Science,
2026; DOI `10.1287/mnsc.2024.08294`).

## Current readiness

As audited on 2026-07-25, all 22 non-factor columns are implemented,
unit-tested, and materialized. The near-faithful training artifact contains
30,732 complete unique stock-date rows, and its full 2022-11-01 through
2026-06-30 matched period contains 27,510 complete rows.

This makes the **feature block** ready for training. It does not make the
repository end-to-end training-ready: no explicit target panel,
modeling-table builder, chronological splitter, estimator, or evaluator has
been implemented. See
[`training_readiness.md`](training_readiness.md)
for the missing work and the proposed target/model/feature ladder.

The current generated manifests record `actual_hashes_verified: false`.
Before a final locked experiment, rebuild with `--verify-input-hashes`. The
currently materialized core artifacts also begin in 2022; rebuild from the
earliest post-warm-up date for a long-history quant-only experiment.

It reads only completed files under:

```text
data/prices/alpaca/15min/sip/all/
```

It never reads or writes the active extended-hours download directory. The
regular-bar file list is snapshotted once at startup, and a chunk is eligible
only if its sibling manifest is complete and matches the 15-minute/SIP/all-
adjustments/regular-session contract. The recorded source-calendar file is
hash-checked, and its date/open/close records must match the build calendar
over that chunk's exact date range.

## The 22 implemented features

| Family | Columns | Count |
|---|---|---:|
| HAR correlation | `rc_d`, `rc_w`, `rc_m` | 3 |
| HAR negative semicorrelation | `rc_negative_d`, `rc_negative_w`, `rc_negative_m` | 3 |
| Exponential pair correlation | `exp_rc_d`, `exp_rc_w`, `exp_rc_m`, `exp_rc_q` | 4 |
| Exponential pair negative semicorrelation | `exp_rc_negative_d`, `exp_rc_negative_w`, `exp_rc_negative_m`, `exp_rc_negative_q` | 4 |
| Sector-state exponential correlation | `sector_exp_rc_d`, `sector_exp_rc_w`, `sector_exp_rc_m`, `sector_exp_rc_q` | 4 |
| Sector-state exponential negative semicorrelation | `sector_exp_rc_negative_d`, `sector_exp_rc_negative_w`, `sector_exp_rc_negative_m`, `sector_exp_rc_negative_q` | 4 |

The exact paper contains three additional characteristic-projection features,
`FRCd`, `FRCw`, and `FRCm`. They are intentionally omitted. Reproducing them
requires a full cross-sectional realized covariance matrix and 15
point-in-time firm characteristics; a rolling Fama-French regression is not
the same construction.

## Return and correlation construction

For each completed day, the builder reconstructs:

1. prior regular-session close to current regular-session open;
2. first 15-minute bar open to close; and
3. subsequent contiguous 15-minute close-to-close returns.

The first item is the paper-style overnight component and needs no
extended-hours download. If a bar is missing, the resulting 30-minute move is
discarded instead of being paired with another asset's 15-minute return.
Official early-close sessions remain valid. The first observed session for a
symbol is retained only as the prior-close anchor; it is not treated as a
complete full-day component because no preceding close is available. If an
entire official session is absent, the builder also refuses to turn the next
observed open into a multi-session "overnight" return.

For synchronized stock and benchmark returns \(r_{i,k}\) and \(r_{s,k}\):

\[
RCov_{is}=\sum_k r_{i,k}r_{s,k},\quad
RV_i=\sum_k r_{i,k}^2,\quad
RC_{is}=\frac{RCov_{is}}{\sqrt{RV_iRV_s}}.
\]

The negative semicovariance uses only jointly negative returns:

\[
RCov^-_{is}=\sum_k r_{i,k}r_{s,k}
\mathbf{1}(r_{i,k}<0)\mathbf{1}(r_{s,k}<0),
\]

where juxtaposition denotes multiplication. Its denominator uses each
asset's own negative semivariance. In code this is implemented directly as:

```text
sum((stock_return * sector_return) where both are negative)
/
sqrt(
    sum(stock_return^2 where stock_return is negative)
    *
    sum(sector_return^2 where sector_return is negative)
)
```

The 5- and 21-session features aggregate covariance and variance components
first and normalize afterward. They do **not** average daily correlations.

## Exponential weights

The implementation follows the paper's center-of-mass equation exactly.
For center \(h\in\{1,5,21,63\}\):

\[
\lambda=\log(1+1/h),\qquad q=e^{-\lambda}=\frac{h}{h+1}.
\]

The 500 most recent daily component observations receive weights
proportional to:

```text
q^499, q^498, ..., q, 1
```

The four weighted covariance and variance sets are normalized into
correlations only after weighting. This is not the same as passing `h` to a
library `halflife` argument. The default requires all 500 sessions.

For each sector, the eight exponential sector-state features are the mean of
the already-normalized features across all unique stock-stock pairs in the
configured six-stock peer set. The ETF is not included in that average.

## Timing

Every row contains:

```text
sector
stock
benchmark
asof_session
forecast_date
22 feature columns
```

For `forecast_date = t`, `asof_session` is the immediately preceding official
market session. No regular-session return from `t` enters the features.

## Commands

Install the declared numeric dependencies:

```powershell
python -m pip install -r requirements-quant.txt
```

Preview the immutable input snapshot:

```powershell
python scripts/build_bollerslev_core_features.py --dry-run
```

Build the full Parquet panel:

```powershell
python scripts/build_bollerslev_core_features.py
```

That command is the strict paper-style audit specification: all 500 daily
components must be present in every exponential window. The Alpaca panel has a
small number of isolated invalid pair-days. Because one missing day affects a
500-session rolling window, strict complete-case filtering is unnecessarily
costly for the downstream matched-period panel.

For training, use an explicitly named near-faithful specification that
requires at least 490 of 500 components and renormalizes the same finite
paper weights over those available observations. It also requires the current
component and at least 99% of the finite-window weight to remain:

```powershell
python scripts/build_bollerslev_core_features.py `
  --start 2022-01-03 `
  --end 2026-06-30 `
  --min-exponential-valid 490 `
  --min-exponential-weight-fraction 0.99 `
  --output data/features/quant/bollerslev_core_features_training.parquet
```

Do not report this as an exact 500/500 replication. Keep the strict build as a
robustness result.

The verified CSV.GZ artifacts reported below used these exact output options:

```powershell
# Strict robustness panel
python scripts/build_bollerslev_core_features.py `
  --start 2022-01-03 `
  --end 2026-06-30 `
  --output data/features/quant/bollerslev_core_features.csv.gz

# Matched-period training panel
python scripts/build_bollerslev_core_features.py `
  --start 2022-01-03 `
  --end 2026-06-30 `
  --min-exponential-valid 490 `
  --min-exponential-weight-fraction 0.99 `
  --output data/features/quant/bollerslev_core_features_training.csv.gz
```

Build a bounded CSV smoke sample without requiring a Parquet engine:

```powershell
python scripts/build_bollerslev_core_features.py `
  --stocks AMD `
  --start 2024-01-02 `
  --end 2024-03-31 `
  --output data/features/quant/smoke/bollerslev_core_amd_2024q1.csv.gz
```

Selecting one output stock still loads its six configured peers, because the
sector-state features require all 15 unique peer pairs.

Useful robustness options:

```text
--exclude-overnight
--keep-incomplete
--verify-input-hashes
--min-alignment-ratio
--min-exponential-weight-fraction
--minimum-sector-pair-fraction
```

The default output is:

```text
data/features/quant/bollerslev_core_features.parquet
data/features/quant/bollerslev_core_features.manifest.json
```

The manifest records the feature schema, timing contract, input snapshot
digest, coverage diagnostics, parameters, and known deviations from the
published design.

## Important differences from the paper

- The paper predicts next-month stock-stock correlations; this project uses
  daily stock-sector-ETF pairs.
- Alpaca trade-bar OHLC prices replace TAQ midquotes.
- The five configured six-stock sector groups are fixed, not point-in-time
  complete index memberships.
- Sector-state features are attached to the stock's sector even though the
  paired benchmark is an ETF outside the paper's GICS stock-pair setup.
- The three characteristic-projection factor features are omitted.

These distinctions make this a transparent adaptation, not an exact
replication.

## Verification

The focused unit suite checks:

- overnight inclusion exactly once;
- synchronized return construction;
- rejection of mismatched 30-minute gaps;
- rejection of multi-session overnight returns after a missing session;
- pair coverage measured against the more complete return leg;
- covariance-component aggregation before normalization;
- negative semicorrelation denominators;
- the exact center-of-mass weights;
- retained exponential-weight enforcement under relaxed counts;
- unique within-sector pair averaging;
- strict `t-1` feature timing;
- input-manifest snapshot and calendar-identity filtering; and
- the exact 22-column schema.

A real-data smoke build for AMD/SOXX over 2024 Q1 produced 61 complete rows,
22 bounded features, no missing values, and `asof_session < forecast_date` on
every row. The strict 500/500 CSV contains 20,676 complete stock-day rows
(61.21% of the fully balanced panel) over 2022-01-03 through 2026-06-30.

The 490/500 plus 99%-retained-weight training CSV contains 30,732 complete
stock-day rows (90.98% overall balanced-panel coverage). Its entire intended
FLAN-T5 period from 2022-11-01 onward is complete for all 30 stocks (917 dates
and 27,510 rows), as is the Llama 2 period from 2023-09-01 onward (708 dates
and 21,240 rows). Every strict row is present in the training panel and has
identical feature values there.
