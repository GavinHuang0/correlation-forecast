# Winner-base B4 sensitivity

Status: **complete**

Completed: `2026-07-27T10:39:53.880579Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 7530 | 0.363791 | 0.288439 | 0.209800 | 0.428695 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.364862 | 0.363791 | 0.005857 | -0.00077976 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- base_prediction_reference.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b4_forecast_d43_elastic_net/base_prediction_reference.json` - SHA-256 `e8e3546ccb7b5a833341f51cf10fc2ca4b220a733fcaee82a4c8ade4451743a5`
- fits.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b4_forecast_d43_elastic_net/fits.json` - SHA-256 `d87d0aa5398600b29fb58f368f57b83aeabfd6467dd156572e27d9838c1f901c`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b4_forecast_d43_elastic_net/fold_metrics.json` - SHA-256 `06548c083145179ca8f68494672ac39eb883d9d29faf7ab1398e784b603da7ae`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b4_forecast_d43_elastic_net/predictions.parquet` - SHA-256 `c567df9f344c5bd106a4b2f38936ba16933c52b0fa045833d3899c652a94c95e`
- residual_training_index.parquet: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b4_forecast_d43_elastic_net/residual_training_index.parquet` - SHA-256 `5532f4cc21446aece37085a069b0b398b76e4f2c43e9c22551ebfdfebc99a48e`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b4_forecast_d43_elastic_net/validation_predictions.parquet` - SHA-256 `3697245a32deb00cbdcfda65d5ba8d2c108c16181613b915a031f7d9c9466768`
- manifest.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b4_forecast_d43_elastic_net/manifest.json` - SHA-256 `d4f21bd4279613d807611c87dc71eb69255d392128333f4370a8243305d0e1b3`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
