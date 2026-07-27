# B0 zero correction

Status: **complete**

Completed: `2026-07-27T10:37:27.219218Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 7530 | 0.365758 | 0.289788 | 0.214013 | 0.422500 |
| t1_loo | 7530 | 0.375135 | 0.296450 | 0.239258 | 0.411319 |
| t2_etf | 7290 | 0.215835 | 0.172199 | 0.124142 | 0.282684 |
| t2_loo | 7290 | 0.233898 | 0.183493 | 0.150696 | 0.280724 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.365758 | 0.365758 | 0.000000 | 0.00000000 |
| t1_loo | 0.375135 | 0.375135 | 0.000000 | 0.00000000 |
| t2_etf | 0.215835 | 0.215835 | 0.000000 | 0.00000000 |
| t2_loo | 0.233898 | 0.233898 | 0.000000 | 0.00000000 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- base_prediction_reference.json: `outputs/quant_deterministic_news/v1/track_b_residual/b0_zero_correction/base_prediction_reference.json` - SHA-256 `9d2dc7024ed542fdb4a77a719f98a3aec0244f0cc8eba11d0c662bd54542bff4`
- fits.json: `outputs/quant_deterministic_news/v1/track_b_residual/b0_zero_correction/fits.json` - SHA-256 `65d1b41b4da85222c6d66f1dc5517b5b39f6c9c41ce5e4921ecb85940f0432c8`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_b_residual/b0_zero_correction/fold_metrics.json` - SHA-256 `ad904581972d1be715fc100b058b02da68b9aef716dd82e0a41d6cf262904e23`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_residual/b0_zero_correction/predictions.parquet` - SHA-256 `945553e15a62829260c02ea59d4f9768771f3c9aa3452381ca3d5b484dd46246`
- residual_training_index.parquet: `outputs/quant_deterministic_news/v1/track_b_residual/b0_zero_correction/residual_training_index.parquet` - SHA-256 `6f1362200fb13dd7eac7a8b13e1adbb8745d8c07e06d0513990d915b28ae1095`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_residual/b0_zero_correction/validation_predictions.parquet` - SHA-256 `5739ba46f5559e5b54cda13153f164c2c6fc9cc22ba16fb6de474c46f0450cd5`
- manifest.json: `outputs/quant_deterministic_news/v1/track_b_residual/b0_zero_correction/manifest.json` - SHA-256 `e97f48ebb070bc04546ec7c4fe962c31f12ef86ddad5f0a3791f887a2bdcda0e`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
