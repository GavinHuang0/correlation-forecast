# A7 Elastic Net/XGBoost ensemble

Status: **complete**

Completed: `2026-07-27T10:37:05.236373Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 11190 | 0.367931 | 0.290320 | 0.206958 | 0.417453 |
| t2_etf | 10830 | 0.236386 | 0.184856 | 0.125882 | 0.278950 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- ensemble_diagnostics.json: `outputs/quant_deterministic_news/v1/track_a_joint/a7_q56_d43_ensemble/ensemble_diagnostics.json` - SHA-256 `4f20e5fad02de7cd0ed432895fa517e7b2447478a0b70284d57bf56ec18e47f2`
- fits.json: `outputs/quant_deterministic_news/v1/track_a_joint/a7_q56_d43_ensemble/fits.json` - SHA-256 `37517e5f3dc66819f61f5a7bb8ace1921282415f10551d2defa5c3eb0985b570`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_a_joint/a7_q56_d43_ensemble/fold_metrics.json` - SHA-256 `ac561763d8c3673cb59418dae9c351bd86c3c904624316fe124832dd77ae9f7e`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_a_joint/a7_q56_d43_ensemble/predictions.parquet` - SHA-256 `632e8b70456c26fa42cafa269c735ca9c4f6b714941d50f583472954c66e9489`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_a_joint/a7_q56_d43_ensemble/validation_predictions.parquet` - SHA-256 `2fcd3c0afa249882f73a16b269c57e66c75e6e20ade23456a7062aed518e54dd`
- manifest.json: `outputs/quant_deterministic_news/v1/track_a_joint/a7_q56_d43_ensemble/manifest.json` - SHA-256 `aa5618345905c955e9553571793754938ec801274a77311c5238c5edb7bd3a81`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
