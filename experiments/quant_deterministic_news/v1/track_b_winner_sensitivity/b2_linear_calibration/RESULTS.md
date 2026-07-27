# Winner-base B2 sensitivity

Status: **complete**

Completed: `2026-07-27T10:39:26.042165Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 7530 | 0.363732 | 0.288131 | 0.210601 | 0.428881 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.364862 | 0.363732 | 0.006181 | -0.00082289 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- base_prediction_reference.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b2_linear_calibration/base_prediction_reference.json` - SHA-256 `e8e3546ccb7b5a833341f51cf10fc2ca4b220a733fcaee82a4c8ade4451743a5`
- fits.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b2_linear_calibration/fits.json` - SHA-256 `0702faeca198afb4e342475fdda8f39dc63a0cd8b1c461c9b8398f5b2e7b5c86`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b2_linear_calibration/fold_metrics.json` - SHA-256 `619f2f2859efe3e841f5e00fdfa67051d0c4b785d8255de63d30c255aeec1abb`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b2_linear_calibration/predictions.parquet` - SHA-256 `30640d109b7f6500f501bcb4d27acbf99fc0a7f6d8efed48b0d08f94bab2b37b`
- residual_training_index.parquet: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b2_linear_calibration/residual_training_index.parquet` - SHA-256 `5532f4cc21446aece37085a069b0b398b76e4f2c43e9c22551ebfdfebc99a48e`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b2_linear_calibration/validation_predictions.parquet` - SHA-256 `ca16d16870f3ffcb3c19d2d4ea94af69be12011bce8e32dc795fbf801aceb140`
- manifest.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b2_linear_calibration/manifest.json` - SHA-256 `2ba39b40878318283ef0118b50ab8c3e454def8047727cd11775de63fbea6a32`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
