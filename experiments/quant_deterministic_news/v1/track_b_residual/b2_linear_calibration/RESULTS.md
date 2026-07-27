# B2 linear calibration

Status: **complete**

Completed: `2026-07-27T10:37:52.423170Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 7530 | 0.365332 | 0.289675 | 0.211731 | 0.423847 |
| t1_loo | 7530 | 0.376592 | 0.298136 | 0.236648 | 0.406737 |
| t2_etf | 7290 | 0.221116 | 0.176522 | 0.127682 | 0.247156 |
| t2_loo | 7290 | 0.248573 | 0.194334 | 0.157253 | 0.187637 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.365758 | 0.365332 | 0.002331 | -0.00031186 |
| t1_loo | 0.375135 | 0.376592 | -0.007783 | 0.00109526 |
| t2_etf | 0.215835 | 0.221116 | -0.049529 | 0.00230731 |
| t2_loo | 0.233898 | 0.248573 | -0.129417 | 0.00708020 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- base_prediction_reference.json: `outputs/quant_deterministic_news/v1/track_b_residual/b2_linear_calibration/base_prediction_reference.json` - SHA-256 `9d2dc7024ed542fdb4a77a719f98a3aec0244f0cc8eba11d0c662bd54542bff4`
- fits.json: `outputs/quant_deterministic_news/v1/track_b_residual/b2_linear_calibration/fits.json` - SHA-256 `4fa00b334c54e61c829693cea1b7aca6d753c01da0ad060c92e5c82d0e9970e6`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_b_residual/b2_linear_calibration/fold_metrics.json` - SHA-256 `0781a794ff05d5808b1d3f32eed9e4d19153bfdc3966a442e284cdebccca1c3d`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_residual/b2_linear_calibration/predictions.parquet` - SHA-256 `dc0d86da71e07f4246e159ddf2d529f35960061c4df5b5597fb4e5117efe9d11`
- residual_training_index.parquet: `outputs/quant_deterministic_news/v1/track_b_residual/b2_linear_calibration/residual_training_index.parquet` - SHA-256 `d640a39c8f3f29acba4240e701125d95a689d4946acea8b85c58edba7fbe23e8`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_residual/b2_linear_calibration/validation_predictions.parquet` - SHA-256 `09bef75abdae7056a0a2d0d1e03c4b3732524aecfd40c576b9af91dd466b384a`
- manifest.json: `outputs/quant_deterministic_news/v1/track_b_residual/b2_linear_calibration/manifest.json` - SHA-256 `e106439c52821a3c7cc2e027eaa42c06b917b5a931ef2c35575853e6a84781af`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
