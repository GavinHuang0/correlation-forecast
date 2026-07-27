# A4 Q56+D1+D2+D3 Elastic Net

Status: **complete**

Completed: `2026-07-27T10:35:26.835910Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 11190 | 0.371263 | 0.293058 | 0.209244 | 0.406853 |
| t1_loo | 11190 | 0.385396 | 0.304037 | 0.230770 | 0.383778 |
| t2_etf | 10830 | 0.241821 | 0.189399 | 0.129266 | 0.245412 |
| t2_loo | 10830 | 0.262691 | 0.204083 | 0.152669 | 0.216102 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.370931 | 0.371263 | -0.001793 | 0.00024676 |
| t1_loo | 0.385767 | 0.385396 | 0.001924 | -0.00028630 |
| t2_etf | 0.244842 | 0.241821 | 0.024524 | -0.00147018 |
| t2_loo | 0.263394 | 0.262691 | 0.005335 | -0.00037009 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- fits.json: `outputs/quant_deterministic_news/v1/track_a_joint/a4_q56_d1_d2_d3_elastic_net/fits.json` - SHA-256 `c5922e791b6bffccd13dc79d0aeafb72e32b33d0a5f4ac989b3c6f5b4bfbed7e`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_a_joint/a4_q56_d1_d2_d3_elastic_net/fold_metrics.json` - SHA-256 `2a243479343cee2c381f4d9c5b053c1cd76094774fff21d072e610f6e91447bf`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_a_joint/a4_q56_d1_d2_d3_elastic_net/predictions.parquet` - SHA-256 `d19be4b7bfedd40ce4d6c3c1c6876846798d11ce63bad44f0ad131a7b6f4f371`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_a_joint/a4_q56_d1_d2_d3_elastic_net/validation_predictions.parquet` - SHA-256 `64c18724956bbd0154f2c0389faf02222ecc4ef1cdfe10d191480f5357cf0049`
- manifest.json: `outputs/quant_deterministic_news/v1/track_a_joint/a4_q56_d1_d2_d3_elastic_net/manifest.json` - SHA-256 `a3480754d667a99fbf0ba2d70d9a22ae8f3872f1da87a7a64ca06bbed9da7ee5`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
