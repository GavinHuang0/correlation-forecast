# Winner-base B5 sensitivity

Status: **complete**

Completed: `2026-07-27T10:40:08.705586Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 7530 | 0.363195 | 0.287786 | 0.210181 | 0.430568 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.364862 | 0.363195 | 0.009117 | -0.00121372 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- base_prediction_reference.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b5_forecast_d43_xgboost/base_prediction_reference.json` - SHA-256 `e8e3546ccb7b5a833341f51cf10fc2ca4b220a733fcaee82a4c8ade4451743a5`
- fits.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b5_forecast_d43_xgboost/fits.json` - SHA-256 `25f36f34e3d819475db858b339af460e2f7f174e18db49d0243fbd06d716fcb1`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b5_forecast_d43_xgboost/fold_metrics.json` - SHA-256 `fca952210b3c8f26dcb8cdf19ef83c76eb3d8c333d86e8fcee5dbab99bdbae64`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b5_forecast_d43_xgboost/predictions.parquet` - SHA-256 `77c0f5239410033d156dfb0df99539f7db9638db44ec18ffbdaaebcaa233d1f6`
- residual_training_index.parquet: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b5_forecast_d43_xgboost/residual_training_index.parquet` - SHA-256 `5532f4cc21446aece37085a069b0b398b76e4f2c43e9c22551ebfdfebc99a48e`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b5_forecast_d43_xgboost/validation_predictions.parquet` - SHA-256 `b6fd655bd6937a0a63f46f0b3437975c03a20a2c84faad48577d4883ca53f81e`
- manifest.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b5_forecast_d43_xgboost/manifest.json` - SHA-256 `51cdee84c3bcead78e7d5ca2bdb6b825de8032a19912b362bfb1b28c5414cb07`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
