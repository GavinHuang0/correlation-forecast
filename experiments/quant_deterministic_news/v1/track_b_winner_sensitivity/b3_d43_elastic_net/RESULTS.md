# Winner-base B3 sensitivity

Status: **complete**

Completed: `2026-07-27T10:39:39.911693Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 7530 | 0.363714 | 0.288366 | 0.209837 | 0.428938 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.364862 | 0.363714 | 0.006281 | -0.00083617 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- base_prediction_reference.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b3_d43_elastic_net/base_prediction_reference.json` - SHA-256 `e8e3546ccb7b5a833341f51cf10fc2ca4b220a733fcaee82a4c8ade4451743a5`
- fits.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b3_d43_elastic_net/fits.json` - SHA-256 `cbae610034911310e8d9b0afc690672023197f8fb86573596514fceff49979d1`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b3_d43_elastic_net/fold_metrics.json` - SHA-256 `6399698c01a6822338c15d29aa4345aa386ff58cb58fa3bda9c7cb8791e85b2d`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b3_d43_elastic_net/predictions.parquet` - SHA-256 `b8792ad33a1b9d6ee4c89930194ba17d69ba8c831d8a2788d0f7fb1daa189d6d`
- residual_training_index.parquet: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b3_d43_elastic_net/residual_training_index.parquet` - SHA-256 `5532f4cc21446aece37085a069b0b398b76e4f2c43e9c22551ebfdfebc99a48e`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b3_d43_elastic_net/validation_predictions.parquet` - SHA-256 `62b08541158903fec75c86c9f830d94911cc00c233670309db5196dff2731cdb`
- manifest.json: `outputs/quant_deterministic_news/v1/track_b_winner_sensitivity/b3_d43_elastic_net/manifest.json` - SHA-256 `9c014739de3e8c05a5288e541299c1003f3e1d0abd84e83b2f2b3897cb611074`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
