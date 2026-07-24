# Additional quantitative feature pipeline

This pipeline is intentionally isolated from the main regular-session Alpaca
backfill.

## Current readiness

As audited on 2026-07-25, the full feature artifact exists at:

```text
data/features/quant/additional_quant_features.parquet
```

It contains 79,105 unique stock-date rows from 2016-01-04 through 2026-06-30.
All 27,510 rows in the matched 2022-11-01 through 2026-06-30 period join
one-to-one to the Bollerslev training panel.

The dense context block is complete over that matched period. The whole
extended-hours block is complete for 24,817 rows, so it requires explicit
missingness indicators and a training-only imputation policy rather than
unqualified complete-case deletion.

The panel is **feature-ready but not end-to-end training-ready**. Target
construction, modeling-table assembly, chronological splits, and estimators
are still absent. See
[`training_readiness.md`](training_readiness.md)
for the audited status and model ladder.

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
| Stock realized volatility | `lagged_realized_volatility`; **provisional** because the current implementation omits the first regular-session bar's open-to-close return |
| Sector realized volatility | `sector_lagged_realized_volatility`; **provisional** for the same reason |
| VIX level/change | `vix_lag1`, `vix_change_lag1` |
| 2/5/10-year yield level/change | `treasury_*_lag2`, `treasury_*_change_lag2` |
| Sector return dispersion | `lagged_sector_return_dispersion` |
| BLS/BEA/FOMC event indicators | release counts and agency/day flags |
| Simplified factor-implied correlation | `factor_implied_correlation` |
| Earnings-event indicator | Partial archive in `reported_earnings.csv`; not in primary panel |

Regular-session quantities are shifted so the feature for forecast date `t`
uses a completed session before `t`. Treasury H.15 values use a conservative
two-target-session lag because the preceding observation is commonly
published after the following morning's forecast cutoff. The Fama-French
rolling model also ends two rows before the forecast date.

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

The current realized-volatility calculation differences consecutive bar
closes within each regular session. Consequently, the first 09:30 bar has no
within-session predecessor and its open-to-close return is omitted. Correct
this before treating either lagged realized-volatility column as a final
feature. The Bollerslev core builder already handles the first bar explicitly
and can serve as the implementation reference.

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
- **Extended-hours cross-sectional selection:** in the matched period, 24,817
  of 27,510 rows are complete across all premarket and aftermarket fields.
  Missingness is much higher for some less-active names, so complete-case
  filtering would change the stock composition.
- **Artifact provenance:** the current additional-feature manifest records
  input paths, but not input/output hashes, builder parameters, or
  column-level coverage. Add those before locking the final experiment.
- **VIX licensing:** FRED identifies VIXCLS as Cboe-copyrighted. Check
  redistribution terms before publishing raw values.

## Features not produced by this pipeline

The current builder does not produce:

- the realized-correlation target;
- DCC-GARCH forecasts;
- SPY-derived market return, volatility, beta, or sector-market correlation;
- point-in-time ETF weights or leave-one-out sector baskets;
- options-implied features; or
- an earnings indicator safe for the primary pre-open specification.

These are tracked explicitly in
[`training_readiness.md`](training_readiness.md).

## Verification

The unit suite covers:

- Eastern/UTC conversion in standard and daylight time;
- official early-close aftermarket boundaries;
- multi-symbol narrow-window request construction;
- read-only reuse of the Alpaca calendar;
- live FRED and Ken French CSV formats;
- BLS, BEA, and both historical/current FOMC calendar parsing;
- leakage-safe lags;
- premarket/overnight panel construction;
- rolling factor-implied correlation bounds.

The official-data fetcher was also smoke-tested live for January 2024 in an
isolated directory. The partial earnings fetcher was live-tested on AMD and
returned 15 in-range quarterly records. The completed extended-hours backfill
now covers all 917 matched-period market dates in both requested windows.
