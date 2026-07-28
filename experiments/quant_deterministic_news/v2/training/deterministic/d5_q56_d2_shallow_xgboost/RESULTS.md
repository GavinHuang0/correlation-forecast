# V2-D5 validation-gated Q56 plus D2-Normalized shallow XGBoost

Status: **complete**

Completed: `2026-07-28T06:50:19.077160Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 11190 | 0.368766 | 0.291247 | 0.206161 | 0.414804 |

## Exact matched Q comparison

| Target | Q RMSE | Model RMSE | Incremental R2 | Loss delta | Better folds |
|---|---:|---:|---:|---:|---:|
| t1_etf | 0.370931 | 0.368766 | 0.011636 | -0.00160096 | 3 |

## Shallow XGBoost versus the D3 linear endpoint

| Target | Base RMSE | Model RMSE | Incremental R2 | Loss delta | Better folds |
|---|---:|---:|---:|---:|---:|
| t1_etf | 0.371505 | 0.368766 | 0.014690 | -0.00202752 | 3 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`
- Outer evaluation excluded from tuning/gating: `True`

## Artifacts

- fits.json: `outputs/quant_deterministic_news/v2/deterministic/d5_q56_d2_shallow_xgboost/fits.json` - SHA-256 `0ec8aafce18a8912ce9fdb9f7c387a564f03b9311f27b2ba77011e31025bdfb6`
- fold_metrics.json: `outputs/quant_deterministic_news/v2/deterministic/d5_q56_d2_shallow_xgboost/fold_metrics.json` - SHA-256 `3e03f48b5f9265129803acc8469affc3bce307c9b52afb153b4892b5d1e15c4a`
- predictions.parquet: `outputs/quant_deterministic_news/v2/deterministic/d5_q56_d2_shallow_xgboost/predictions.parquet` - SHA-256 `8e6ea8b5cc86bf70e5cc662972080841900eb484a15c37fc459d1bb84c80f451`
- validation_gate.json: `outputs/quant_deterministic_news/v2/deterministic/d5_q56_d2_shallow_xgboost/validation_gate.json` - SHA-256 `58ef71585b8f0f858d0b7ba9b6d5a6c65b1f843c94329df192902f86c81eae6f`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v2/deterministic/d5_q56_d2_shallow_xgboost/validation_predictions.parquet` - SHA-256 `45beb9a1a593006d7d705d36b8aecb8c39ded2b2f93df515ae01998d029f63b8`
- manifest.json: `outputs/quant_deterministic_news/v2/deterministic/d5_q56_d2_shallow_xgboost/manifest.json` - SHA-256 `cd8ef31d30c93f37b2803f499c79afc1412619af684f32a5df514a017d0d1553`

This is an exploratory development result from retrospective news.
It is not confirmatory point-in-time evidence.
