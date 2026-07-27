# A2 Q56+D1 Elastic Net

Status: **complete**

Completed: `2026-07-27T10:34:46.726320Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 11190 | 0.371053 | 0.292877 | 0.209060 | 0.407523 |
| t1_loo | 11190 | 0.385427 | 0.303746 | 0.230872 | 0.383678 |
| t2_etf | 10830 | 0.243101 | 0.190616 | 0.130008 | 0.237404 |
| t2_loo | 10830 | 0.262404 | 0.203822 | 0.153626 | 0.217814 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.370931 | 0.371053 | -0.000661 | 0.00009096 |
| t1_loo | 0.385767 | 0.385427 | 0.001762 | -0.00026228 |
| t2_etf | 0.244842 | 0.243101 | 0.014172 | -0.00084958 |
| t2_loo | 0.263394 | 0.262404 | 0.007507 | -0.00052080 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- fits.json: `outputs/quant_deterministic_news/v1/track_a_joint/a2_q56_d1_elastic_net/fits.json` - SHA-256 `a7d6770fdb55d3758bc8eaeee94fa75054b06cffcc0ff8e7ce8a369c5fbe21d3`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_a_joint/a2_q56_d1_elastic_net/fold_metrics.json` - SHA-256 `d597554e35b5dc3c940b4d5db2af035a4106c6d13df4568b6c8db467a3b09193`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_a_joint/a2_q56_d1_elastic_net/predictions.parquet` - SHA-256 `bae78d9873efd1a65d8445ce707096de0b2702b181a38c1a424ac291092a465f`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_a_joint/a2_q56_d1_elastic_net/validation_predictions.parquet` - SHA-256 `8e31ab8184a92b22ea5a48fef07dae3b0bcfbfd51f59c4e66837e0e8daac3f31`
- manifest.json: `outputs/quant_deterministic_news/v1/track_a_joint/a2_q56_d1_elastic_net/manifest.json` - SHA-256 `ed65da549f0a11fde343985b3529efa483957ba7342a10b10f1ddb12b23c0a6a`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
