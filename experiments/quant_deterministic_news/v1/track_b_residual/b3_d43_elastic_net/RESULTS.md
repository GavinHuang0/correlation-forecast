# B3 D43 residual Elastic Net

Status: **complete**

Completed: `2026-07-27T10:38:05.638842Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 7530 | 0.364813 | 0.289551 | 0.210264 | 0.425481 |
| t1_loo | 7530 | 0.376383 | 0.297965 | 0.238399 | 0.407395 |
| t2_etf | 7290 | 0.217511 | 0.174038 | 0.123443 | 0.271504 |
| t2_loo | 7290 | 0.241967 | 0.190719 | 0.154604 | 0.230243 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.365758 | 0.364813 | 0.005162 | -0.00069058 |
| t1_loo | 0.375135 | 0.376383 | -0.006665 | 0.00093797 |
| t2_etf | 0.215835 | 0.217511 | -0.015587 | 0.00072611 |
| t2_loo | 0.233898 | 0.241967 | -0.070184 | 0.00383964 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- base_prediction_reference.json: `outputs/quant_deterministic_news/v1/track_b_residual/b3_d43_elastic_net/base_prediction_reference.json` - SHA-256 `9d2dc7024ed542fdb4a77a719f98a3aec0244f0cc8eba11d0c662bd54542bff4`
- fits.json: `outputs/quant_deterministic_news/v1/track_b_residual/b3_d43_elastic_net/fits.json` - SHA-256 `0593332014e80c94457f99a1ebfa1567145b070bf700d4540a9283813c7f2dfd`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_b_residual/b3_d43_elastic_net/fold_metrics.json` - SHA-256 `a9b7f1251e76b9d7c4c2124631e5017bff3079aa8874d9c3593ba348a8d4dfe5`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_residual/b3_d43_elastic_net/predictions.parquet` - SHA-256 `cc5e4c8c361d98f98396b4b77ffa16ae6ac3303bf46b5cf189ae23290ceda078`
- residual_training_index.parquet: `outputs/quant_deterministic_news/v1/track_b_residual/b3_d43_elastic_net/residual_training_index.parquet` - SHA-256 `d640a39c8f3f29acba4240e701125d95a689d4946acea8b85c58edba7fbe23e8`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_residual/b3_d43_elastic_net/validation_predictions.parquet` - SHA-256 `67220a6ef05e195395383211d478e90e2902ae9712543af51212a4e2c705d228`
- manifest.json: `outputs/quant_deterministic_news/v1/track_b_residual/b3_d43_elastic_net/manifest.json` - SHA-256 `104c23c2d4822a9d453b15d089d0c60fed389ee9b59a0baa1a160f29540eaa3e`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
