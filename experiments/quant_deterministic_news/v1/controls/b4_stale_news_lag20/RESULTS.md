# B4 stale-news control

Status: **complete**

Completed: `2026-07-27T10:41:39.230846Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 7530 | 0.364271 | 0.288733 | 0.211122 | 0.427189 |
| t1_loo | 7530 | 0.377560 | 0.298655 | 0.241538 | 0.403684 |
| t2_etf | 7290 | 0.219716 | 0.175319 | 0.126565 | 0.256661 |
| t2_loo | 7290 | 0.247121 | 0.193197 | 0.157406 | 0.197100 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.365758 | 0.364271 | 0.008119 | -0.00108611 |
| t1_loo | 0.375135 | 0.377560 | -0.012970 | 0.00182524 |
| t2_etf | 0.215835 | 0.219716 | -0.036279 | 0.00169007 |
| t2_loo | 0.233898 | 0.247121 | -0.116261 | 0.00636043 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- base_prediction_reference.json: `outputs/quant_deterministic_news/v1/controls/b4_stale_news_lag20/base_prediction_reference.json` - SHA-256 `9d2dc7024ed542fdb4a77a719f98a3aec0244f0cc8eba11d0c662bd54542bff4`
- fits.json: `outputs/quant_deterministic_news/v1/controls/b4_stale_news_lag20/fits.json` - SHA-256 `9fba690874e2690365658904005de940592cb0c3734f3e7d183c7fce0fb9b789`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/controls/b4_stale_news_lag20/fold_metrics.json` - SHA-256 `1ec60720ea80f6343c694821ea73d9ac66246669aef368a15f10bf3717e0cb41`
- predictions.parquet: `outputs/quant_deterministic_news/v1/controls/b4_stale_news_lag20/predictions.parquet` - SHA-256 `71e83da16419f899ebe92dbcdce6f510128a6025bd69886b55bf87ef1e6ca4c4`
- residual_training_index.parquet: `outputs/quant_deterministic_news/v1/controls/b4_stale_news_lag20/residual_training_index.parquet` - SHA-256 `d640a39c8f3f29acba4240e701125d95a689d4946acea8b85c58edba7fbe23e8`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/controls/b4_stale_news_lag20/validation_predictions.parquet` - SHA-256 `9db93dab2f75dcc1ce50a91e618e28bc099123c7de9cd65c6c1d7f334561b0c2`
- manifest.json: `outputs/quant_deterministic_news/v1/controls/b4_stale_news_lag20/manifest.json` - SHA-256 `a60799a763d46113a08675ec65c84bdfa32a140baf94bc5286f4619bbfcc3b86`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
