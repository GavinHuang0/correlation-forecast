# A0-T Q56 XGBoost

Status: **complete**

Completed: `2026-07-27T10:33:57.784649Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 11190 | 0.369951 | 0.292284 | 0.207699 | 0.411039 |
| t1_loo | 11190 | 0.381434 | 0.300462 | 0.229137 | 0.396380 |
| t2_etf | 10830 | 0.234820 | 0.183441 | 0.124759 | 0.288475 |
| t2_loo | 10830 | 0.256653 | 0.199505 | 0.148772 | 0.251721 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.369951 | 0.369951 | 0.000000 | 0.00000000 |
| t1_loo | 0.381434 | 0.381434 | 0.000000 | 0.00000000 |
| t2_etf | 0.234820 | 0.234820 | 0.000000 | 0.00000000 |
| t2_loo | 0.256653 | 0.256653 | 0.000000 | 0.00000000 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- fits.json: `outputs/quant_deterministic_news/v1/track_a_joint/a0_t_q56_xgboost/fits.json` - SHA-256 `69f9a2056563c4f69908e0605964fa29ea4b13d61448121febfb4c967d4b646d`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_a_joint/a0_t_q56_xgboost/fold_metrics.json` - SHA-256 `e4745b8810d976657e706bee7c651ca1bdf5670309bc191320a65e2e19b55b8e`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_a_joint/a0_t_q56_xgboost/predictions.parquet` - SHA-256 `396908aa10074a1c7ea0564fa4c7ffc45020105407f84a6d2df7001b97b79b5b`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_a_joint/a0_t_q56_xgboost/validation_predictions.parquet` - SHA-256 `b55d9a9a1c6df70d19c6d7c04bff3cf674e9335753f5b46df023e46bb1831f57`
- manifest.json: `outputs/quant_deterministic_news/v1/track_a_joint/a0_t_q56_xgboost/manifest.json` - SHA-256 `f81f89ca1c75cccbe9cd4f01b4fc8123e39a81473a8cf1cfefe8eda1e0427cd9`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
