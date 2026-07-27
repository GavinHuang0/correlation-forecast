# Winner-base B0 sensitivity

Status: **complete**

Completed: `2026-07-27T10:38:58.921604Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 7530 | 0.364862 | 0.288790 | 0.213666 | 0.425329 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.364862 | 0.364862 | 0.000000 | 0.00000000 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- base_prediction_reference.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b0_zero_correction/base_prediction_reference.json` - SHA-256 `e8e3546ccb7b5a833341f51cf10fc2ca4b220a733fcaee82a4c8ade4451743a5`
- fits.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b0_zero_correction/fits.json` - SHA-256 `52079bd5dcb827854b6d8f32efac8f712e7e46f067335f680322346d7179fd11`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b0_zero_correction/fold_metrics.json` - SHA-256 `dead7bf1ef01469819236afbbbab37b81c15f47aa9ed74be6fe89ca703164703`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b0_zero_correction/predictions.parquet` - SHA-256 `2699d0230905ab220c555e040f9a4da8ce6f2eb402d7860dacc89dda4ae73679`
- residual_training_index.parquet: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b0_zero_correction/residual_training_index.parquet` - SHA-256 `6f1362200fb13dd7eac7a8b13e1adbb8745d8c07e06d0513990d915b28ae1095`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b0_zero_correction/validation_predictions.parquet` - SHA-256 `279da25c5586d8fd1d0b7c157326d27ff2478ba2e879cf15a278d70c262e748c`
- manifest.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b0_zero_correction/manifest.json` - SHA-256 `a5edf8907a7f648ffbf9b1c3a6fb6212a4b05a6e5639ba4845068b9d00b7d1a7`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
