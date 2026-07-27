# A3 Q56+D1+D2 Elastic Net

Status: **complete**

Completed: `2026-07-27T10:35:06.496402Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 11190 | 0.371359 | 0.293006 | 0.209252 | 0.406546 |
| t1_loo | 11190 | 0.385280 | 0.303709 | 0.230699 | 0.384147 |
| t2_etf | 10830 | 0.242409 | 0.189901 | 0.129641 | 0.241736 |
| t2_loo | 10830 | 0.261535 | 0.203292 | 0.152737 | 0.222987 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.370931 | 0.371359 | -0.002311 | 0.00031797 |
| t1_loo | 0.385767 | 0.385280 | 0.002521 | -0.00037515 |
| t2_etf | 0.244842 | 0.242409 | 0.019773 | -0.00118533 |
| t2_loo | 0.263394 | 0.261535 | 0.014070 | -0.00097612 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- fits.json: `outputs/quant_deterministic_news/v1/track_a_joint/a3_q56_d1_d2_elastic_net/fits.json` - SHA-256 `ae4b735b3841c910adff67159d5f872f3f81f40b84ae19714c5f083b0ece5c47`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_a_joint/a3_q56_d1_d2_elastic_net/fold_metrics.json` - SHA-256 `91e9cad7e616cae7b92cbd53e35959f940facfda0619cae72d9eb0c6b31a2fd3`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_a_joint/a3_q56_d1_d2_elastic_net/predictions.parquet` - SHA-256 `8607c59f2db20ac7df3069b4babd1cae125f3bbcc3b60d5b8581844bab52d313`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_a_joint/a3_q56_d1_d2_elastic_net/validation_predictions.parquet` - SHA-256 `5ab9c29d77cec27f3ae68e6d26f97cbf17ff90f78c687fbd2175c89ed15e2931`
- manifest.json: `outputs/quant_deterministic_news/v1/track_a_joint/a3_q56_d1_d2_elastic_net/manifest.json` - SHA-256 `aa7a8f23e015388c4ce24488cc56d588981387704bf063d19d6dfc3037b81964`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
