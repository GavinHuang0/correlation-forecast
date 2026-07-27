# B1 mean-error correction

Status: **complete**

Completed: `2026-07-27T10:37:39.304347Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 7530 | 0.364260 | 0.288730 | 0.211130 | 0.427221 |
| t1_loo | 7530 | 0.377560 | 0.298655 | 0.241538 | 0.403684 |
| t2_etf | 7290 | 0.219716 | 0.175319 | 0.126565 | 0.256661 |
| t2_loo | 7290 | 0.248800 | 0.194778 | 0.162110 | 0.186150 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.365758 | 0.364260 | 0.008174 | -0.00109357 |
| t1_loo | 0.375135 | 0.377560 | -0.012970 | 0.00182524 |
| t2_etf | 0.215835 | 0.219716 | -0.036279 | 0.00169007 |
| t2_loo | 0.233898 | 0.248800 | -0.131485 | 0.00719329 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- base_prediction_reference.json: `outputs/quant_deterministic_news/v1/track_b_residual/b1_mean_error/base_prediction_reference.json` - SHA-256 `9d2dc7024ed542fdb4a77a719f98a3aec0244f0cc8eba11d0c662bd54542bff4`
- fits.json: `outputs/quant_deterministic_news/v1/track_b_residual/b1_mean_error/fits.json` - SHA-256 `7f8ca27a600ed8cb7ea6c1a2966bab1cc1c7dff14a4725c703d607e317cd73e8`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_b_residual/b1_mean_error/fold_metrics.json` - SHA-256 `2d9ffe270c6d5dc6b5be35430c47513dc117f1c217290ddd570a57c9944eb3f8`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_residual/b1_mean_error/predictions.parquet` - SHA-256 `9e9a25d14f670144b4482251cf88359f7405fc8e13861b220c14c2a727f8405f`
- residual_training_index.parquet: `outputs/quant_deterministic_news/v1/track_b_residual/b1_mean_error/residual_training_index.parquet` - SHA-256 `d640a39c8f3f29acba4240e701125d95a689d4946acea8b85c58edba7fbe23e8`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_residual/b1_mean_error/validation_predictions.parquet` - SHA-256 `186cec49a23dc1d318fba2fee64cc2b0fd7e058f6b6abfd61e51178242430e33`
- manifest.json: `outputs/quant_deterministic_news/v1/track_b_residual/b1_mean_error/manifest.json` - SHA-256 `e512891dca6dde4b96090ae058ed39bdac9ba9374363ba9ee5543061d7722e61`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
