# A5 stale-news control

Status: **complete**

Completed: `2026-07-27T10:40:50.291186Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 11190 | 0.369448 | 0.291468 | 0.207873 | 0.412640 |
| t1_loo | 11190 | 0.383484 | 0.302162 | 0.229554 | 0.389876 |
| t2_etf | 10830 | 0.238953 | 0.187418 | 0.127209 | 0.263205 |
| t2_loo | 10830 | 0.260650 | 0.202576 | 0.152429 | 0.228233 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.370931 | 0.369448 | 0.007981 | -0.00109811 |
| t1_loo | 0.385767 | 0.383484 | 0.011800 | -0.00175603 |
| t2_etf | 0.244842 | 0.238953 | 0.047526 | -0.00284905 |
| t2_loo | 0.263394 | 0.260650 | 0.020727 | -0.00143794 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- fits.json: `outputs/quant_deterministic_news/v1/controls/a5_stale_news_lag20/fits.json` - SHA-256 `19eaa4b9a9a37ce1824729845fd67c6030a5032f3ba26ead1dbfe5a09687ad2e`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/controls/a5_stale_news_lag20/fold_metrics.json` - SHA-256 `672d6f129df0ab935fa91af8fbe17bc9b04d9cff7962ad5c20cc9688f63c4e13`
- predictions.parquet: `outputs/quant_deterministic_news/v1/controls/a5_stale_news_lag20/predictions.parquet` - SHA-256 `4cd109cbeab0815bae5d6ce86d0d8586778ba41d00edbc94304fc2f3ab379f0d`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/controls/a5_stale_news_lag20/validation_predictions.parquet` - SHA-256 `76d5250178df5f51868ac580511f45892d871bb971c34e4ab84f03b822867c9d`
- manifest.json: `outputs/quant_deterministic_news/v1/controls/a5_stale_news_lag20/manifest.json` - SHA-256 `7978828da27044d6e896303c61b21602a4ae9d191f27a0e0d768ff5f54e62316`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
