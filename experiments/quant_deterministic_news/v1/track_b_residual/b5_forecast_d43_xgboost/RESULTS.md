# B5 forecast+D43 residual XGBoost

Status: **complete**

Completed: `2026-07-27T10:38:42.501718Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 7530 | 0.365506 | 0.290074 | 0.211567 | 0.423297 |
| t1_loo | 7530 | 0.377889 | 0.299241 | 0.236507 | 0.402646 |
| t2_etf | 7290 | 0.219595 | 0.175230 | 0.126508 | 0.257478 |
| t2_loo | 7290 | 0.248537 | 0.194574 | 0.161878 | 0.187875 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.365758 | 0.365506 | 0.001379 | -0.00018448 |
| t1_loo | 0.375135 | 0.377889 | -0.014732 | 0.00207319 |
| t2_etf | 0.215835 | 0.219595 | -0.035140 | 0.00163702 |
| t2_loo | 0.233898 | 0.248537 | -0.129087 | 0.00706213 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- base_prediction_reference.json: `outputs/quant_deterministic_news/v1/track_b_residual/b5_forecast_d43_xgboost/base_prediction_reference.json` - SHA-256 `9d2dc7024ed542fdb4a77a719f98a3aec0244f0cc8eba11d0c662bd54542bff4`
- fits.json: `outputs/quant_deterministic_news/v1/track_b_residual/b5_forecast_d43_xgboost/fits.json` - SHA-256 `7d5f53e25e6b4d49724afeb9df6711fc59a882138827cd09cdd355e6aff818c3`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_b_residual/b5_forecast_d43_xgboost/fold_metrics.json` - SHA-256 `5ea11dbbff1b280421191b50a14a798083a961c67a296884b9293788149647ab`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_residual/b5_forecast_d43_xgboost/predictions.parquet` - SHA-256 `d6a9ce4c29f6a879f7625a92ba7fe538df6e88b27ce07a7fa7c9178bbe96f05b`
- residual_training_index.parquet: `outputs/quant_deterministic_news/v1/track_b_residual/b5_forecast_d43_xgboost/residual_training_index.parquet` - SHA-256 `d640a39c8f3f29acba4240e701125d95a689d4946acea8b85c58edba7fbe23e8`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_residual/b5_forecast_d43_xgboost/validation_predictions.parquet` - SHA-256 `c09e8388744684f3b538a531488e2ab32318c8382b3c8ce0b3221225347ed2ab`
- manifest.json: `outputs/quant_deterministic_news/v1/track_b_residual/b5_forecast_d43_xgboost/manifest.json` - SHA-256 `3b55c099166ffddd0da984b52a6eedfe380503a76b429e52b1c848325c33c588`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
