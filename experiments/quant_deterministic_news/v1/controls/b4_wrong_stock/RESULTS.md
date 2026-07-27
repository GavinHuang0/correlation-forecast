# B4 wrong-stock control

Status: **complete**

Completed: `2026-07-27T10:41:54.449529Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 7530 | 0.365114 | 0.289717 | 0.210332 | 0.424533 |
| t1_loo | 7530 | 0.375744 | 0.297711 | 0.234440 | 0.409408 |
| t2_etf | 7290 | 0.217945 | 0.174082 | 0.124969 | 0.268596 |
| t2_loo | 7290 | 0.240897 | 0.189250 | 0.152625 | 0.237036 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.365758 | 0.365114 | 0.003520 | -0.00047091 |
| t1_loo | 0.375135 | 0.375744 | -0.003246 | 0.00045686 |
| t2_etf | 0.215835 | 0.217945 | -0.019640 | 0.00091492 |
| t2_loo | 0.233898 | 0.240897 | -0.060739 | 0.00332291 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- base_prediction_reference.json: `outputs/quant_deterministic_news/v1/controls/b4_wrong_stock/base_prediction_reference.json` - SHA-256 `9d2dc7024ed542fdb4a77a719f98a3aec0244f0cc8eba11d0c662bd54542bff4`
- fits.json: `outputs/quant_deterministic_news/v1/controls/b4_wrong_stock/fits.json` - SHA-256 `040d0a7cc3fd26dc93a21eb31dab765c447e6baa17dccd727720ce24b45fb886`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/controls/b4_wrong_stock/fold_metrics.json` - SHA-256 `d4e5c8eeb35f5a1a3d8f93adc7308fc2b8690ea438f95e57a34494d37297c039`
- predictions.parquet: `outputs/quant_deterministic_news/v1/controls/b4_wrong_stock/predictions.parquet` - SHA-256 `8d28ac4761c1801616a9426d62059b2dfb41b29e56f992a056dc7605f423450a`
- residual_training_index.parquet: `outputs/quant_deterministic_news/v1/controls/b4_wrong_stock/residual_training_index.parquet` - SHA-256 `d640a39c8f3f29acba4240e701125d95a689d4946acea8b85c58edba7fbe23e8`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/controls/b4_wrong_stock/validation_predictions.parquet` - SHA-256 `10fa5aebea96ed5ae2ebef147dfd86ab9be59a183f1e9249a810c38adcac9975`
- manifest.json: `outputs/quant_deterministic_news/v1/controls/b4_wrong_stock/manifest.json` - SHA-256 `34672d97be7193614a8458c3ead8c9ca56379967d1e52a38e2ea172e1f11bd98`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
