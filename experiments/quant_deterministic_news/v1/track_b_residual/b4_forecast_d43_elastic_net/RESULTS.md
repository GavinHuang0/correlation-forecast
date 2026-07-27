# B4 forecast+D43 residual Elastic Net

Status: **complete**

Completed: `2026-07-27T10:38:21.403406Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 7530 | 0.364893 | 0.289613 | 0.210353 | 0.425230 |
| t1_loo | 7530 | 0.375710 | 0.297694 | 0.234458 | 0.409515 |
| t2_etf | 7290 | 0.218654 | 0.175077 | 0.124609 | 0.263830 |
| t2_loo | 7290 | 0.240747 | 0.189144 | 0.152575 | 0.237982 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.365758 | 0.364893 | 0.004727 | -0.00063240 |
| t1_loo | 0.375135 | 0.375710 | -0.003064 | 0.00043112 |
| t2_etf | 0.215835 | 0.218654 | -0.026285 | 0.00122449 |
| t2_loo | 0.233898 | 0.240747 | -0.059423 | 0.00325093 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- base_prediction_reference.json: `outputs/quant_deterministic_news/v1/track_b_residual/b4_forecast_d43_elastic_net/base_prediction_reference.json` - SHA-256 `9d2dc7024ed542fdb4a77a719f98a3aec0244f0cc8eba11d0c662bd54542bff4`
- fits.json: `outputs/quant_deterministic_news/v1/track_b_residual/b4_forecast_d43_elastic_net/fits.json` - SHA-256 `cd028dbf83559eb772f3f3a6a7ad5c3dad8f26e2d287b437ac20614b64fcf7ee`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_b_residual/b4_forecast_d43_elastic_net/fold_metrics.json` - SHA-256 `4046f6718e590557acfd99ba44a6b989d6570134a3f919594108fca6511d557e`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_residual/b4_forecast_d43_elastic_net/predictions.parquet` - SHA-256 `7d4646f617307ce73c2306fcd5170e4def3724b79b6a001e58224cd4dacd589d`
- residual_training_index.parquet: `outputs/quant_deterministic_news/v1/track_b_residual/b4_forecast_d43_elastic_net/residual_training_index.parquet` - SHA-256 `d640a39c8f3f29acba4240e701125d95a689d4946acea8b85c58edba7fbe23e8`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_residual/b4_forecast_d43_elastic_net/validation_predictions.parquet` - SHA-256 `2dbad6a48bc8e3abd1ef86e678a4e49d1a23a5d15c04624927fa2516ba89fee9`
- manifest.json: `outputs/quant_deterministic_news/v1/track_b_residual/b4_forecast_d43_elastic_net/manifest.json` - SHA-256 `96dda3c7add68d70033b4e6c1caad9b901aabe13a1a299e246c79608408c32c3`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
