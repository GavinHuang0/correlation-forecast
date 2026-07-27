# A5 wrong-stock control

Status: **complete**

Completed: `2026-07-27T10:41:22.223358Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 11190 | 0.371106 | 0.293050 | 0.208906 | 0.407356 |
| t1_loo | 11190 | 0.385465 | 0.304151 | 0.230774 | 0.383556 |
| t2_etf | 10830 | 0.242850 | 0.190439 | 0.129717 | 0.238977 |
| t2_loo | 10830 | 0.263199 | 0.204574 | 0.153087 | 0.213065 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.370931 | 0.371106 | -0.000943 | 0.00012972 |
| t1_loo | 0.385767 | 0.385465 | 0.001565 | -0.00023284 |
| t2_etf | 0.244842 | 0.242850 | 0.016206 | -0.00097151 |
| t2_loo | 0.263394 | 0.263199 | 0.001481 | -0.00010275 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- fits.json: `outputs/quant_deterministic_news/v1/controls/a5_wrong_stock/fits.json` - SHA-256 `a235afede626e9a7f5e3e8a80a8daa2d995f6ca07449ddce071cbb411ac910af`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/controls/a5_wrong_stock/fold_metrics.json` - SHA-256 `c45b5a05a2326d561002f6c19a9f50d04f8209defc8569d99c9dd7cdd528e8dd`
- predictions.parquet: `outputs/quant_deterministic_news/v1/controls/a5_wrong_stock/predictions.parquet` - SHA-256 `dc8144d7bcdcc4ce6802ad77f74060db94373e09195c0889e7f18d7f42ea78ea`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/controls/a5_wrong_stock/validation_predictions.parquet` - SHA-256 `a8c5b6e3f23ff69bd288f978594546d6e69a23b96db2983dced9ec92cd96fd13`
- manifest.json: `outputs/quant_deterministic_news/v1/controls/a5_wrong_stock/manifest.json` - SHA-256 `2776ac8ceb87dcb32617e4a3477d7b18f6edd706115639fa6b0de7c089435ae0`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
