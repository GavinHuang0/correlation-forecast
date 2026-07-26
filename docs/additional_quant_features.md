# Additional quantitative feature pipeline

This pipeline is intentionally isolated from the main regular-session Alpaca
backfill.

## Current readiness

As rebuilt and audited on 2026-07-25, the locked training artifact exists at:

```text
data/features/quant/training_v1/additional_quant_features.parquet
```

It contains 79,110 unique stock-date rows from 2016-01-04 through 2026-06-30.
All 27,510 rows in the matched 2022-11-01 through 2026-06-30 period join
one-to-one to the Bollerslev training panel.

The dense context and corrected realized-volatility blocks are complete over
that matched period. Extended-hours fields remain naturally sparse, so the
artifact now includes explicit stock/ETF premarket and prior-aftermarket
availability indicators. Rungs 2 and 3 use training-only median imputation
rather than complete-case deletion.

Targets, modeling-table assembly, chronological splits, and rungs 1–4 are now
implemented. See
[`training_readiness.md`](training_readiness.md)
for the audited construction and results.

## Scripts

| Script | Network source | Writes to regular-bar directory? | Purpose |
|---|---|---:|---|
| `scripts/fetch_alpaca_extended_bars.py` | Alpaca | No | Premarket/aftermarket supplement |
| `scripts/fetch_official_quant_data.py` | FRED, Ken French, BLS, BEA, Federal Reserve | No | Market context, factors, macro calendars |
| `scripts/fetch_alpha_vantage_earnings.py` | Alpha Vantage | No | Partial historical reported-earnings dates |
| `scripts/build_additional_quant_features.py` | None | No | Derive a stock-date feature panel |

The extended-hours downloader reuses the completed Alpaca calendar cache
read-only. It makes one multi-symbol request for each date/window:

```text
premarket:   04:00 <= bar start < 09:00 ET
aftermarket: official market close <= bar start < 20:00 ET
```

It therefore does not request regular-session bars. The completed matched
LLM-period backfill contains 917 market dates, 917 premarket files, and 917
aftermarket files for 2022-11-01 through 2026-06-30. The 09:00 cutoff is
deliberately earlier than the opening bell so every premarket input is known
when the forecast is formed.

## Commands

Validate the extended-hours plan without credentials or network traffic:

```powershell
python scripts/fetch_alpaca_extended_bars.py --dry-run
```

To verify or resume the matched-period download:

```powershell
python scripts/fetch_alpaca_extended_bars.py
```

If only premarket inputs are wanted:

```powershell
python scripts/fetch_alpaca_extended_bars.py --sessions premarket
```

The official-data job has no Alpaca dependency and can run independently:

```powershell
python scripts/fetch_official_quant_data.py `
  --start 2016-01-01 `
  --end 2026-06-30
```

Fetch the partial Alpha Vantage reported-date archive. The free-tier-friendly
default requests at most 20 new symbols and resumes from cached responses, so
run it again on a later quota day to finish the 30-stock universe:

```powershell
python scripts/fetch_alpha_vantage_earnings.py
```

Install feature-builder dependencies:

```powershell
python -m pip install -r requirements-quant.txt
```

Preview the completed-file-list snapshot:

```powershell
python scripts/build_additional_quant_features.py --dry-run
```

Build the Parquet panel:

```powershell
python scripts/build_additional_quant_features.py
```

## Produced features

| Requested feature | Output/status |
|---|---|
| Stock overnight return | `stock_overnight_return` |
| Sector overnight return | `sector_overnight_return` |
| Stock-minus-sector overnight return | `stock_minus_sector_overnight_return` |
| Premarket return | `premarket_return` |
| Premarket volume | `premarket_volume` |
| Relative premarket volume | `relative_premarket_volume_20d` |
| Prior aftermarket return/volume | `prior_aftermarket_return`, `prior_aftermarket_volume` |
| Relative daily volume | `lagged_relative_daily_volume_20d` |
| Stock realized volatility | `lagged_realized_volatility`; includes the first bar's open-to-close return |
| Sector realized volatility | `sector_lagged_realized_volatility`; same corrected interval contract |
| VIX level/change | `vix_lag1`, `vix_change_lag1` |
| 2/5/10-year yield level/change | `treasury_*_lag2`, `treasury_*_change_lag2` |
| Sector return dispersion | `lagged_sector_return_dispersion` |
| BLS/BEA/FOMC event indicators | release counts and agency/day flags |
| Simplified factor-implied correlation | `factor_implied_correlation` |
| Earnings-event indicator | Partial archive in `reported_earnings.csv`; not in primary panel |

## Mathematical definitions

Let $t$ be the forecast session, $t-1$ its immediately preceding official
session, and $a$ either a stock or its configured sector ETF. Additional
features labelled `sector_*` are ETF quantities even in an LOO-target model.
Only the 14 pair-history columns and the response change from ETF to LOO.

### Regular-session volatility, volume, and dispersion

For a complete regular session $d$, define the first interval return and
subsequent exact-gap returns as

$$
u_{a,d,1}=\log\!\left(\frac{C_{a,d,1}}{O_{a,d,1}}\right),
\qquad
u_{a,d,k}=\log\!\left(\frac{C_{a,d,k}}{C_{a,d,k-1}}\right).
$$

The second equation is used only when the two bar timestamps are exactly
15 minutes apart. A normal session requires 26 returns and an official 13:00
close requires 14. An incomplete session makes its return, volume, and
realized-variance fields missing.

The unannualized realized volatility is

$$
\sigma^{RV}_{a,d}
=
\sqrt{\sum_{k\in\mathcal K_d^{RTH}}u_{a,d,k}^{2}},
\qquad
\texttt{lagged\_realized\_volatility}_{a,t}
=
\sigma^{RV}_{a,t-1}.
$$

If $V_{a,d}$ is complete-session share volume, relative daily volume is

$$
\mathrm{RelVol}^{20}_{a,t}
=
\frac{V_{a,t-1}}
{\frac{1}{n_t}\sum_{\ell=2}^{21}I_{a,t-\ell}V_{a,t-\ell}},
\qquad
n_t=\sum_{\ell=2}^{21}I_{a,t-\ell}\ge 10.
$$

The numerator session is deliberately excluded from the 20-session reference
window. Official-session reindexing ensures that a lag never jumps over a
missing market session.

Let

$$
R^{CC}_{i,d}=\frac{C^{RTH}_{i,d}}{C^{RTH}_{i,d-1}}-1
$$

be a stock's close-to-close simple return. For the six configured stocks
$\mathcal S_s$ in sector $s$, the lagged dispersion feature is the sample
standard deviation

$$
\mathrm{Disp}_{s,t}
=
\sqrt{
\frac{1}{5}
\sum_{i\in\mathcal S_s}
\left(R^{CC}_{i,t-1}-\overline R^{CC}_{s,t-1}\right)^2
}.
$$

All six returns must be present; otherwise the feature is missing.

### Premarket, overnight, and prior aftermarket features

The premarket window is $04{:}00\le\text{bar start}<09{:}00$ ET on forecast
date $t$. For its first open $O^{PM}_{a,t}$, last close
$C^{PM}_{a,t}$, bar volumes $v^{PM}_{a,t,k}$, and bar count $N^{PM}_{a,t}$:

$$
R^{PM}_{a,t}
=
\frac{C^{PM}_{a,t}}{O^{PM}_{a,t}}-1,
\qquad
V^{PM}_{a,t}
=
\sum_k v^{PM}_{a,t,k}.
$$

There is no complete-grid requirement for an extended-hours window. If no
usable bar exists, its values stay missing rather than becoming zero.
Relative premarket volume is

$$
\mathrm{RelPMVol}^{20}_{a,t}
=
\frac{V^{PM}_{a,t}}
{\frac{1}{n_t}\sum_{\ell=1}^{20}
I^{PM}_{a,t-\ell}V^{PM}_{a,t-\ell}},
\qquad n_t\ge 10.
$$

The feature named `overnight_return` stops at the last pre-09:00 close:

$$
R^{ON}_{a,t}
=
\frac{C^{PM}_{a,t}}{C^{RTH}_{a,t-1}}-1.
$$

It requires the exact preceding official RTH close. The cross-leg feature is

$$
R^{ON,\mathrm{relative}}_{i,t}
=
R^{ON}_{i,t}-R^{ON}_{ETF,t}.
$$

For the prior aftermarket window, the observation attached to forecast date
$t$ is the window following official session $t-1$:

$$
R^{AM,prior}_{i,t}
=
\frac{C^{AM}_{i,t-1}}{O^{AM}_{i,t-1}}-1,
$$

$$
\mathrm{RelAMVol}^{20}_{i,t}
=
\frac{V^{AM}_{i,t-1}}
{\frac{1}{n_t}\sum_{\ell=2}^{21}
I^{AM}_{i,t-\ell}V^{AM}_{i,t-\ell}},
\qquad n_t\ge10.
$$

The availability indicators are deterministic missingness flags: stock and
ETF premarket availability mean a bar count is present; prior-stock
aftermarket availability means the prior-window return is present.

### Market and macro context

After reindexing FRED observations to official sessions and forward-filling
published levels,

$$
\texttt{vix\_lag1}_t=VIX_{t-1},
\qquad
\texttt{vix\_change\_lag1}_t=VIX_{t-1}-VIX_{t-2}.
$$

For Treasury maturity $m\in\{2,5,10\}$ years, the conservative publication
lag is two target sessions:

$$
\texttt{treasury}_{m,\mathrm{lag2},t}=Y_{m,t-2},
\qquad
\Delta Y^{lag2}_{m,t}=Y_{m,t-2}-Y_{m,t-3}.
$$

Yields remain in the percentage-point units supplied by FRED. Scheduled macro
counts are same-date calendar quantities known at the forecast cutoff:

$$
N^{macro}_t=\#\{\text{scheduled releases on }t\},
\qquad
N^{preopen}_t=\#\{\text{those flagged before 09:00}\}.
$$

The agency indicators are Boolean maxima over the same schedule rows. They do
not contain the realized macroeconomic surprise.

### Optional factor-implied correlation

This feature is generated for robustness but excluded from the primary
training blocks. For asset $a$, a rolling regression estimates

$$
r_{a,d}-r_{f,d}
=
\alpha_a+\beta_a^\top f_d+\varepsilon_{a,d},
$$

where $f_d=(Mkt-RF,SMB,HML,Mom)^\top$. The window contains at most 252 rows,
requires at least 126, and leaves the two immediately preceding factor rows
outside the estimation slice before forecast date $t$. With factor
covariance $\Sigma_f$, the implied correlation is

$$
\rho^{factor}_{i,b,t}
=
\frac{\beta_i^\top\Sigma_f\beta_b}
{\sqrt{
\left(\beta_i^\top\Sigma_f\beta_i+\sigma_{\varepsilon_i}^2\right)
\left(\beta_b^\top\Sigma_f\beta_b+\sigma_{\varepsilon_b}^2\right)
}}.
$$

Residual covariance is assumed zero. This is not the characteristic-
projection feature in Bollerslev, Li, and Tang.

### Transformations used by the trained models

Before rungs 2 and 3, nonnegative count, bar-count, volume, and relative-volume
features listed in `LOG1P_FEATURES` receive the fixed transformation

$$
x^*=\log(1+x).
$$

During hyperparameter selection, linear models use training-block median
imputation and training-block standardization. After selection, the final
outer-test fit recomputes both on train plus validation. XGBoost receives the
fixed transforms but uses native missing-value routing and does not require
standardization.

Regular-session quantities are reindexed to the official session calendar and
shifted so the feature for forecast date `t` uses the immediately preceding
completed session. Missing official sessions remain missing; lags do not jump
across them. Treasury H.15 values use a conservative
two-target-session lag because the preceding observation is commonly
published after the following morning's forecast cutoff. The Fama-French
rolling model likewise leaves the two immediately preceding factor rows out
of its estimation slice.

The simplified factor feature estimates rolling stock and sector-ETF
exposures to Mkt-RF, SMB, HML, and Mom. Common-factor covariance forms the
numerator; estimated factor variance plus idiosyncratic residual variance form
the denominator. It is an adaptation, not the exact characteristic-projection
method from Bollerslev, Li, and Tang.

The extended-hours return labels have these exact meanings:

- `stock_overnight_return` and `sector_overnight_return` measure prior
  regular-session close to the last available pre-09:00 premarket close;
- `premarket_return` and `sector_premarket_return` measure the first available
  premarket bar open to the last available premarket bar close; and
- the prior-aftermarket returns measure first available aftermarket bar open
  to last available aftermarket bar close.

The word `overnight` therefore does not mean prior close to the 09:30 opening
auction in this panel.

The realized-volatility calculation now includes the first official bar's
open-to-close log return and later close-to-close returns only across exact
15-minute gaps. A missing bar cannot be converted into a longer synthetic
return. Premarket overnight features similarly require the exact preceding
official regular-session close.

## Important limitations

- **BEA history:** the live machine-readable release JSON currently starts in
  2025. BLS and scheduled FOMC calendar coverage spans the requested period;
  BEA flags before 2025 are absent.
- **Fama-French revisions:** the downloaded research returns are the current
  files and can contain historical revisions. Treat the simplified factor
  feature as a research robustness variable, not a pristine point-in-time
  factor vintage.
- **Earnings dates:** Alpha Vantage's `EARNINGS` endpoint supplies historical
  `reportedDate` values, and the separate resumable fetcher preserves them.
  It does not supply announcement time or the schedule as it was known before
  the event. The primary pre-open panel therefore excludes this field; use it
  only as an explicitly ambiguous-timing robustness feature.
- **Sector dispersion:** it is computed from the six configured stocks in
  each sector, not the complete point-in-time index constituency.
- **Extended-hours gaps:** thinly traded symbols can have missing 15-minute
  bars. Volume and returns remain missing when a window has no usable bars;
  they are not silently imputed as zero.
- **Extended-hours cross-sectional selection:** missingness is higher for
  less-active names. The training pipeline retains all common target rows,
  uses availability indicators, and fits imputation on the training block
  only.
- **Artifact provenance:** the locked v1 rebuild verifies every declared input
  hash and records calendar/input digests, the output hash, builder parameters,
  and column-level coverage.
- **VIX licensing:** FRED identifies VIXCLS as Cboe-copyrighted. Check
  redistribution terms before publishing raw values.

## Features not produced by this pipeline

This builder itself does not produce:

- the realized-correlation target (now produced by the separate target
  builder);
- DCC-GARCH forecasts (now produced by rung 4);
- SPY-derived market return, volatility, beta, or sector-market correlation;
- point-in-time ETF weights or an exact ETF-minus-stock basket;
- options-implied features; or
- an earnings indicator safe for the primary pre-open specification.

These are tracked explicitly in
[`training_readiness.md`](training_readiness.md).

The separate target builder now produces an equal-weight five-peer LOO basket
and matching pair-history features. That basket removes target self-inclusion,
but it is not the complete point-in-time sector and should not be described as
one.

## Verification

The unit suite covers:

- Eastern/UTC conversion in standard and daylight time;
- official early-close aftermarket boundaries;
- multi-symbol narrow-window request construction;
- read-only reuse of the Alpaca calendar;
- live FRED and Ken French CSV formats;
- BLS, BEA, and both historical/current FOMC calendar parsing;
- leakage-safe lags;
- inclusion of the first regular-session open-to-close return;
- rejection of return differences across missing 15-minute bars and missing
  official sessions;
- premarket/overnight panel construction;
- exact prior-session close requirements and availability indicators;
- complete configured-sector dispersion; and
- rolling factor-implied correlation bounds.

The official-data fetcher was also smoke-tested live for January 2024 in an
isolated directory. The partial earnings fetcher was live-tested on AMD and
returned 15 in-range quarterly records. The completed extended-hours backfill
now covers all 917 matched-period market dates in both requested windows.
