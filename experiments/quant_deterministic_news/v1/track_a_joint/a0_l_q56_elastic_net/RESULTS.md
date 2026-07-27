# A0-L Q56 Elastic Net

Status: **complete**

Completed: `2026-07-27T10:33:07.585016Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R^2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 11190 | 0.370931 | 0.292976 | 0.208578 | 0.407915 |
| t1_loo | 11190 | 0.385767 | 0.303961 | 0.231278 | 0.382590 |
| t2_etf | 10830 | 0.244842 | 0.192202 | 0.130507 | 0.226441 |
| t2_loo | 10830 | 0.263394 | 0.204305 | 0.154095 | 0.211898 |

| Target | Base RMSE | Model RMSE | Incremental R^2 news given Q | Delta squared loss |
|---|---:|---:|---:|---:|
| t1_etf | 0.370931 | 0.370931 | 0.000000 | 0.00000000 |
| t1_loo | 0.385767 | 0.385767 | 0.000000 | 0.00000000 |
| t2_etf | 0.244842 | 0.244842 | 0.000000 | 0.00000000 |
| t2_loo | 0.263394 | 0.263394 | 0.000000 | 0.00000000 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`

## Artifacts

- fits.json: `outputs/quant_deterministic_news/v1/track_a_joint/a0_l_q56_elastic_net/fits.json` - SHA-256 `3809c6dc43037397de8cae0b051fdd8be51eefd45cb548f560257fded37e52b4`
- fold_metrics.json: `outputs/quant_deterministic_news/v1/track_a_joint/a0_l_q56_elastic_net/fold_metrics.json` - SHA-256 `3f2eb95f9af80340203e883e7c35b505d8a4fff0863fbab6c90a36c9f1456212`
- predictions.parquet: `outputs/quant_deterministic_news/v1/track_a_joint/a0_l_q56_elastic_net/predictions.parquet` - SHA-256 `1c3d43471ff93c6b4ff4c4befe7e25438c94012649daf1e23b3a8b6749deb38b`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v1/track_a_joint/a0_l_q56_elastic_net/validation_predictions.parquet` - SHA-256 `63eb9d181be3dfe8c8da369648961d885a4d8040178c7057b2de2e43733335b9`
- manifest.json: `outputs/quant_deterministic_news/v1/track_a_joint/a0_l_q56_elastic_net/manifest.json` - SHA-256 `1d2d3aee43f2660bd53985e59c7e5e89eb2ebb201abceda41afb16f27b738604`

Interpretation is exploratory only; see the experiment-level README
for the point-in-time and model-selection limitations.
