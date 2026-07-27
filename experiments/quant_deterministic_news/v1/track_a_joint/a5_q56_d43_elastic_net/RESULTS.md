# A5 Q56+D43 Elastic Net

Status: **complete**

Completed: `2026-07-27T10:35:51.912779Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 11190 | 0.370576 | 0.292525 | 0.208648 | 0.409046 |
| t1_loo | 11190 | 0.385055 | 0.303625 | 0.230218 | 0.384866 |
| t2_etf | 10830 | 0.240332 | 0.188125 | 0.128153 | 0.254675 |
| t2_loo | 10830 | 0.262580 | 0.203918 | 0.152634 | 0.216762 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.370931 | 0.370576 | 0.001911 | -0.00026299 |
| t1_loo | 0.385767 | 0.385055 | 0.003686 | -0.00054846 |
| t2_etf | 0.244842 | 0.240332 | 0.036499 | -0.00218805 |
| t2_loo | 0.263394 | 0.262580 | 0.006171 | -0.00042815 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- fits.json: `outputs/quant_deterministic_news/v1/track_a_joint/a5_q56_d43_elastic_net/fits.json` - SHA-256 `55774f0fa785ab042d5ab397e7ad3ecab31ee76b10cfb92cc5b89b2808eb2a80`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_a_joint/a5_q56_d43_elastic_net/fold_metrics.json` - SHA-256 `2fb901fde2605e53be7b6da935ffdaf433691ba2c821e5f8b70e9b7e8ddfa455`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_a_joint/a5_q56_d43_elastic_net/predictions.parquet` - SHA-256 `043f6378316d225201ddbd52bd99d1e9848ba8e946bbb6e2f4960eb5bfc29a25`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_a_joint/a5_q56_d43_elastic_net/validation_predictions.parquet` - SHA-256 `0ebffafe1bffd824b4a8b3c46c697c4721a538f0da1950e16d82cd481ad2a41c`
- manifest.json: `outputs/quant_deterministic_news/v1/track_a_joint/a5_q56_d43_elastic_net/manifest.json` - SHA-256 `3086c3c5de1f595b40c8383dddd0846cfc318f2cf712d0f50ef5953ad4f6cd6b`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
