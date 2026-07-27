# A6 Q56+D43 XGBoost

Status: **complete**

Completed: `2026-07-27T10:36:51.331694Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 11190 | 0.369854 | 0.291813 | 0.206845 | 0.411348 |
| t1_loo | 11190 | 0.382405 | 0.301513 | 0.229247 | 0.393304 |
| t2_etf | 10830 | 0.234152 | 0.182886 | 0.124224 | 0.292516 |
| t2_loo | 10830 | 0.256518 | 0.199145 | 0.148915 | 0.252510 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.369951 | 0.369854 | 0.000524 | -0.00007176 |
| t1_loo | 0.381434 | 0.382405 | -0.005097 | 0.00074159 |
| t2_etf | 0.234820 | 0.234152 | 0.005680 | -0.00031319 |
| t2_loo | 0.256653 | 0.256518 | 0.001054 | -0.00006943 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- fits.json: `outputs/quant_deterministic_news/v1/track_a_joint/a6_q56_d43_xgboost/fits.json` - SHA-256 `503c9e5c2f42e039e06f3fef0ed8b60dc974422210a58909d96965a251746e5d`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_a_joint/a6_q56_d43_xgboost/fold_metrics.json` - SHA-256 `ab81debdf81f757eaf3be9245230311f864c3526c4249c7bdfdf90bb1af12b4f`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_a_joint/a6_q56_d43_xgboost/predictions.parquet` - SHA-256 `9aee0d02772b03871c94420f2bf3c09bf01c336c691e09dcc8f5873b2fe18eee`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_a_joint/a6_q56_d43_xgboost/validation_predictions.parquet` - SHA-256 `bc76b0d4b634d08a7300130e319dccb03e96e52d66abeda121695ff2e6b447f3`
- manifest.json: `outputs/quant_deterministic_news/v1/track_a_joint/a6_q56_d43_xgboost/manifest.json` - SHA-256 `b14f54fd177b9d6a405890afec4c0b41d6a2059c72f123bce4cfbab39fd6adb6`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
