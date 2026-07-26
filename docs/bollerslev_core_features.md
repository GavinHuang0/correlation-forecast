# Bollerslev core correlation features

`scripts/build_bollerslev_core_features.py` is an offline, daily
stock-sector adaptation of the price-derived features in Bollerslev, Li, and
Tang, *Forecasting and Managing Correlation Risks* (Management Science,
2026; DOI `10.1287/mnsc.2024.08294`).

## Current readiness

As audited on 2026-07-25, all 22 non-factor columns are implemented,
unit-tested, and materialized. The locked full-history artifact contains
58,500 complete unique stock-date rows from 2017-12-28 through 2026-06-30.
Its complete matched training period contains 27,510 rows covering all 30
stocks on 917 dates from 2022-11-01 through 2026-06-30. The manifest records
`actual_hashes_verified: true`.

T1/T2 targets, a joined modeling panel, chronological splitters, and rungs
1–4 are now implemented. See
[`training_readiness.md`](training_readiness.md)
for the exact target contracts and results.

ETF and leave-one-out (LOO) are trained in parallel. ETF correlation is
**tradable hedge coupling** and can include mechanical exposure to the target
stock. The LOO target uses the other five configured peers, removing
self-inclusion but not reconstructing the complete point-in-time sector.
The modeling panel therefore contains separate 14-feature stock–ETF and
stock–LOO pair histories; both reuse the same eight sector-state features.

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

Let $d$ denote an official trading session and let
$\mathcal K_d^{full}$ be its official expected full-day component schedule.
For the historical core features this set contains:

$$
\mathcal K_d^{full}
=
\{\text{prior close}\rightarrow\text{open}\}
\cup
\{\text{regular-session 15-minute intervals}\}.
$$

For a traded stock or ETF, the first regular-session interval is an
open-to-close log return,

$$
r_{a,d,1}=\log\!\left(\frac{C_{a,d,1}}{O_{a,d,1}}\right),
$$

and every later observed interval is close-to-close only when its timestamps
are exactly 15 minutes apart:

$$
r_{a,d,k}
=
\log\!\left(\frac{C_{a,d,k}}{C_{a,d,k-1}}\right).
$$

The overnight component is

$$
r^{ON}_{a,d}
=
\log\!\left(\frac{O_{a,d,1}}{C_{a,d-1,\mathrm{last}}}\right),
$$

and exists only when $d-1$ is the immediately preceding official session.
This price-ratio expression applies to a traded stock or ETF. A synthetic LOO
basket instead aggregates its peers' already-constructed interval returns in
simple-return space, including the overnight interval, as defined below.

Thus $|\mathcal K_d^{full}|=27$ on a normal session and 15 on an official
13:00 close. These counts differ from the strict RTH target counts of 26 and
14 because historical core features include the overnight component.

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

Pair coverage is measured against the official expected component count, not
the more complete observed leg. This prevents two assets with the same missing
bars from appearing fully aligned. The core feature artifact keeps the
declared 80% historical-feature threshold; the T1/T2 target builder separately
requires the complete regular-session schedule.

More precisely, a historical pair-day is valid only when

$$
|\mathcal A_{is,d}|\ge15,\qquad
\frac{|\mathcal A_{is,d}|}{|\mathcal K_d^{full}|}\ge0.80,
\qquad
N^{overnight}_d=1.
$$

Here
$\mathcal A_{is,d}\subseteq\mathcal K_d^{full}$ is the observed aligned
intersection of the two legs. For strict target construction,
$\mathcal A_{is,d}=\mathcal K_d^{RTH}$; the historical feature builder
allows the partial coverage above.

A normal session therefore needs at least 22 of 27 aligned components, while
an early close effectively needs all 15. Failure sets every covariance and
variance component for that pair-day to missing.

For synchronized stock and benchmark returns $r_{i,k}$ and $r_{s,k}$:

$$
C_{is,d}=\sum_{k\in\mathcal A_{is,d}} r_{i,d,k}r_{s,d,k},
\qquad
V_{i,d}=\sum_{k\in\mathcal A_{is,d}}r_{i,d,k}^2,
$$

$$
\rho_{is,d}
=
\frac{C_{is,d}}{\sqrt{V_{i,d}V_{s,d}}}.
$$

This is a non-demeaned realized correlation: it treats the zero-return point
as the origin and does not subtract an intraday sample mean.

The negative semicovariance uses only jointly negative returns, while each
denominator leg uses all negative returns for that asset:

$$
C^-_{is,d}
=
\sum_{k\in\mathcal A_{is,d}} r_{i,d,k}r_{s,d,k}
\mathbf 1\{r_{i,d,k}<0,\ r_{s,d,k}<0\},
$$

$$
V^-_{i,d}
=
\sum_{k\in\mathcal A_{is,d}}
r_{i,d,k}^2\mathbf 1\{r_{i,d,k}<0\},
\qquad
\rho^-_{is,d}
=
\frac{C^-_{is,d}}{\sqrt{V^-_{i,d}V^-_{s,d}}}.
$$

In code this is implemented directly as:

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

## HAR features

For a forecast formed before session $t$, all historical components end at
$t-1$. For $H\in\{1,5,21\}$, the HAR correlation is

$$
\rho^{HAR(H)}_{is,t}
=
\frac{
\sum_{d=t-H}^{t-1} C_{is,d}
}{
\sqrt{
\left(\sum_{d=t-H}^{t-1}V_{i,d}\right)
\left(\sum_{d=t-H}^{t-1}V_{s,d}\right)
}
}.
$$

The downside HAR feature replaces $C,V_i,V_s$ with
$C^-,V_i^-,V_s^-$. The suffixes `d`, `w`, and `m` correspond to
$H=1,5,21$. A window is valid only when all required session components are
finite. The three horizons are a parsimonious heterogeneous-memory
representation: yesterday captures fast adjustment, while weekly and monthly
aggregates capture slower persistence. Downside features allow joint declines
to carry information not present in symmetric correlation.

## Exponential weights

The implementation follows the paper's center-of-mass equation exactly. For
center $h\in\{1,5,21,63\}$:

$$
\lambda=\log(1+1/h),\qquad q=e^{-\lambda}=\frac{h}{h+1}.
$$

For a component $X_d$, its finite-window estimate at forecast date $t$ is

$$
\widetilde X^{(h)}_{t}
=
\frac{
\sum_{\ell=0}^{499}q_h^\ell I_{t-1-\ell}X_{t-1-\ell}
}{
\sum_{\ell=0}^{499}q_h^\ell I_{t-1-\ell}
},
$$

where $I_d$ is one when that daily component is available. The 500 most
recent observations therefore receive weights proportional to:

```text
q^499, q^498, ..., q, 1
```

The weighted covariance and variance components are normalized only after
weighting:

$$
\rho^{EW(h)}_{is,t}
=
\frac{\widetilde C^{(h)}_{is,t}}
{\sqrt{\widetilde V^{(h)}_{i,t}\widetilde V^{(h)}_{s,t}}}.
$$

The downside version applies the same operator to negative components. This
is not the same as passing $h$ to a library `halflife` argument.
Exponential weighting reacts more smoothly to recent changes than a hard
rolling window while retaining a persistent long-run state.

The strict default requires all 500 sessions. The locked training build
allows at least 490 valid sessions, requires the most recent component, and
requires at least 99% of the finite-window weight to remain. Available weights
are renormalized as shown above; missing components are never replaced with
zero.

For each sector, the eight exponential sector-state features are the mean of
the already-normalized features across all unique stock-stock pairs in the
configured six-stock peer set. If $\mathcal P_s$ is the set of its
$\binom 62=15$ peer pairs, then

$$
\mathrm{SectorEW}^{(h)}_{s,t}
=
\frac{1}{15}
\sum_{(j,\ell)\in\mathcal P_s}\rho^{EW(h)}_{j\ell,t}.
$$

The downside sector state replaces each term with
$\rho^{EW(h),-}_{j\ell,t}$. The ETF is not included in either average. The
locked build requires all 15 pair features to be present.

## ETF and leave-one-out pair histories

The original core builder produces the stock–ETF pair features. The separate
target builder constructs the corresponding stock–LOO history. For stock
$i$ in a configured six-stock sector, the five-peer interval return is
formed in simple-return space:

$$
r^{-i}_{d,k}
=
\log\!\left[
1+\frac{1}{5}
\sum_{\substack{j\in s(i)\\j\ne i}}
\left(e^{r_{j,d,k}}-1\right)
\right].
$$

The LOO pair components and all 14 pair-history features are then calculated
with the same equations above and shifted so the row for forecast date $t$
ends at session $t-1$. ETF and LOO models share the eight sector-state
features, but never share their 14 pair-history features. The peer basket
itself exists only when all five peers have the exact full-day official
component set. The subsequent stock-versus-basket pair still follows the
historical 80%-alignment-plus-overnight rule; that is intentionally less
strict than T1/T2 target eligibility.

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

The locked v1 training artifact uses an explicitly named near-faithful
specification that
requires at least 490 of 500 components and renormalizes the same finite
paper weights over those available observations. It also requires the current
component and at least 99% of the finite-window weight to remain:

```powershell
python scripts/build_bollerslev_core_features.py `
  --min-exponential-valid 490 `
  --min-exponential-weight-fraction 0.99 `
  --verify-input-hashes `
  --output data/features/quant/training_v1/bollerslev_core_features.parquet
```

Do not report this as an exact 500/500 replication. Keep the strict build as a
robustness result.

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
- The original core artifact's pair-level HAR, downside, and exponential
  features use the sector ETF. The training panel also contains independently
  constructed LOO versions using the fixed five-peer basket.
- The three characteristic-projection factor features are omitted.

These distinctions make this a transparent adaptation, not an exact
replication.

## Verification

The focused unit suite checks:

- overnight inclusion exactly once;
- synchronized return construction;
- rejection of mismatched 30-minute gaps;
- rejection of multi-session overnight returns after a missing session;
- pair coverage measured against the official expected interval count;
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

The earlier 490/500 plus 99%-retained-weight training CSV contained 30,732
complete stock-day rows. The locked Parquet rebuild extends this specification
back to the earliest post-warm-up date and contains 58,500 rows. Its entire
matched period from 2022-11-01 onward is complete for all 30 stocks (917 dates
and 27,510 rows). Target construction and all four trained rungs use that
matched panel.
