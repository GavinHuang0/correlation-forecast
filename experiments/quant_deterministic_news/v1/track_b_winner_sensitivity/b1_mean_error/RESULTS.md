# Winner-base B1 sensitivity

Status: **complete**

Completed: `2026-07-27T10:39:12.375583Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 7530 | 0.363373 | 0.287847 | 0.210811 | 0.430008 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.364862 | 0.363373 | 0.008142 | -0.00108395 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- base_prediction_reference.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b1_mean_error/base_prediction_reference.json` - SHA-256 `e8e3546ccb7b5a833341f51cf10fc2ca4b220a733fcaee82a4c8ade4451743a5`
- fits.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b1_mean_error/fits.json` - SHA-256 `23bbf1646ecfe7cee0e59a9c803cf230eed274dcada947643d52c7efd37ee1d2`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b1_mean_error/fold_metrics.json` - SHA-256 `6217a5c5c820cafbe18db969df9d47543d8293c9c3f136ffcdf5702d2ecc2caa`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b1_mean_error/predictions.parquet` - SHA-256 `09e4ea48012b44f8cb392157442bfdef96a36a79c4da6cbf47c74fce8aa5c943`
- residual_training_index.parquet: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b1_mean_error/residual_training_index.parquet` - SHA-256 `5532f4cc21446aece37085a069b0b398b76e4f2c43e9c22551ebfdfebc99a48e`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b1_mean_error/validation_predictions.parquet` - SHA-256 `602c8668e0a703ba0885a5fac5c52d99d107170fef52d4255d22903099940696`
- manifest.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b1_mean_error/manifest.json` - SHA-256 `ae6d05429f2a31d151bf6b59caa53c7a9a98f9e2e1e34acf5cf99260cc0e377c`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
