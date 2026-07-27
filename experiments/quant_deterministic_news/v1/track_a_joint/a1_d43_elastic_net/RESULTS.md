# A1 D43 Elastic Net

Status: **complete**

Completed: `2026-07-27T10:34:15.977812Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 11190 | 0.445322 | 0.355244 | 0.243578 | 0.146610 |
| t1_loo | 11190 | 0.473489 | 0.377337 | 0.275202 | 0.069870 |
| t2_etf | 10830 | 0.327122 | 0.259395 | 0.169990 | -0.380831 |
| t2_loo | 10830 | 0.364996 | 0.294502 | 0.206158 | -0.513370 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.370931 | 0.445322 | -0.441329 | 0.06072229 |
| t1_loo | 0.385767 | 0.473489 | -0.506503 | 0.07537580 |
| t2_etf | 0.244842 | 0.327122 | -0.785036 | 0.04706106 |
| t2_loo | 0.263394 | 0.364996 | -0.920272 | 0.06384532 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- fits.json: `outputs/quant_deterministic_news/v1/track_a_joint/a1_d43_elastic_net/fits.json` - SHA-256 `2bf77d2fc2a2c1247d13d7306176f3df24edb649b080a33343b416a2d8d10ddb`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_a_joint/a1_d43_elastic_net/fold_metrics.json` - SHA-256 `68dbdbcf1c5381e43454e2f8bd8f0972506016f8949b6db619c206edfb9964de`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_a_joint/a1_d43_elastic_net/predictions.parquet` - SHA-256 `dcbcb7d5b80db10a7c1ebf5afd19753a7ed29fed5223e8ea3039c4048a7a053a`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_a_joint/a1_d43_elastic_net/validation_predictions.parquet` - SHA-256 `01f7ef7607c7ddf7f2ee0e0ef4e1a6b59a71fa14e9ee1503a0c67b9094799160`
- manifest.json: `outputs/quant_deterministic_news/v1/track_a_joint/a1_d43_elastic_net/manifest.json` - SHA-256 `30eb2566ee2b672e9d5f35fee3a74bf26e6ad01274a7af22577dc70191f046a9`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
