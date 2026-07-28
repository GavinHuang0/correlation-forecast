# V2-D3 20-session stale-D2 falsification control

Status: **complete**

Completed: `2026-07-28T06:50:33.094015Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 11190 | 0.371511 | 0.293155 | 0.209244 | 0.406059 |
| t1_loo | 11190 | 0.385162 | 0.303554 | 0.230195 | 0.384524 |
| t2_etf | 10830 | 0.242428 | 0.190073 | 0.129266 | 0.241619 |
| t2_loo | 10830 | 0.261903 | 0.203318 | 0.152949 | 0.220797 |

## Exact matched Q comparison

| Target | Q RMSE | Model RMSE | Incremental R2 | Loss delta | Better folds |
|---|---:|---:|---:|---:|---:|
| t1_etf | 0.370931 | 0.371511 | -0.003133 | 0.00043108 | 1 |
| t1_loo | 0.385767 | 0.385162 | 0.003132 | -0.00046607 | 1 |
| t2_etf | 0.244842 | 0.242428 | 0.019622 | -0.00117627 | 3 |
| t2_loo | 0.263394 | 0.261903 | 0.011292 | -0.00078339 | 3 |

## Falsification control versus contemporaneous V2-D3

| Target | Base RMSE | Model RMSE | Incremental R2 | Loss delta | Better folds |
|---|---:|---:|---:|---:|---:|
| t1_etf | 0.371505 | 0.371511 | -0.000033 | 0.00000453 | 1 |
| t1_loo | 0.385682 | 0.385162 | 0.002693 | -0.00040056 | 2 |
| t2_etf | 0.242221 | 0.242428 | -0.001712 | 0.00010042 | 1 |
| t2_loo | 0.262505 | 0.261903 | 0.004583 | -0.00031578 | 2 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`
- Outer evaluation excluded from tuning/gating: `True`

## Artifacts

- control_audit.json: `outputs/quant_deterministic_news/v2/controls/d3_stale_d2_lag20/control_audit.json` - SHA-256 `a27bedc2689fb1547fe4afc5a5f3034990449ac970278d06288cfc80a38fb548`
- fits.json: `outputs/quant_deterministic_news/v2/controls/d3_stale_d2_lag20/fits.json` - SHA-256 `f935cab698bdded901226e8e2f5da903099481de1f7bd593112daabcd8d987cc`
- fold_metrics.json: `outputs/quant_deterministic_news/v2/controls/d3_stale_d2_lag20/fold_metrics.json` - SHA-256 `0b15de5cdfc759aa30013bbf4a5ec9b23c7279ebb269a1802a0135e82e76501f`
- predictions.parquet: `outputs/quant_deterministic_news/v2/controls/d3_stale_d2_lag20/predictions.parquet` - SHA-256 `c0abec9b886570dac1a8bea33ca8fa3d2d50a920782e98e4efde8ba5784d4e98`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v2/controls/d3_stale_d2_lag20/validation_predictions.parquet` - SHA-256 `f67dbf3049585edcf89b5930f492cf6123ce89fc1a314baeb7e420d667cc2cda`
- manifest.json: `outputs/quant_deterministic_news/v2/controls/d3_stale_d2_lag20/manifest.json` - SHA-256 `ee8f749294886c8edb6265aa8dcab5f6a6d78c19e93e25227f891d623583e05d`

This is an exploratory development result from retrospective news.
It is not confirmatory point-in-time evidence.
