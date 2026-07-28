# V2-D1 D2-only Elastic Net

Status: **complete**

Completed: `2026-07-28T06:49:21.872345Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 11190 | 0.446496 | 0.356802 | 0.243296 | 0.142107 |
| t1_loo | 11190 | 0.474500 | 0.379210 | 0.275666 | 0.065893 |
| t2_etf | 10830 | 0.327509 | 0.260298 | 0.170531 | -0.384103 |
| t2_loo | 10830 | 0.365054 | 0.296024 | 0.206696 | -0.513851 |

## Exact matched Q comparison

| Target | Q RMSE | Model RMSE | Incremental R2 | Loss delta | Better folds |
|---|---:|---:|---:|---:|---:|
| t1_etf | 0.370931 | 0.446496 | -0.448934 | 0.06176863 | 0 |
| t1_loo | 0.385767 | 0.474500 | -0.512944 | 0.07633433 | 0 |
| t2_etf | 0.244842 | 0.327509 | -0.789266 | 0.04731465 | 0 |
| t2_loo | 0.263394 | 0.365054 | -0.920882 | 0.06388765 | 0 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`
- Outer evaluation excluded from tuning/gating: `True`

## Artifacts

- fits.json: `outputs/quant_deterministic_news/v2/deterministic/d1_d2_elastic_net/fits.json` - SHA-256 `31ef21a5fb03bbffd31e1dcf8632ac5663841200b9b2da5b241fd2d098199389`
- fold_metrics.json: `outputs/quant_deterministic_news/v2/deterministic/d1_d2_elastic_net/fold_metrics.json` - SHA-256 `03489526ab5055273212109e2a645d54ff7273120db874584cb0f22180987908`
- predictions.parquet: `outputs/quant_deterministic_news/v2/deterministic/d1_d2_elastic_net/predictions.parquet` - SHA-256 `522f427cfddf62eed34662a1692cf98829a66d16b3916330c512b2ca84739e58`
- source_preflight.json: `outputs/quant_deterministic_news/v2/deterministic/d1_d2_elastic_net/source_preflight.json` - SHA-256 `21949b1fc93d61b709d2d7fcc80c4ea0a9af07a5e893de509c22f216bbb52b26`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v2/deterministic/d1_d2_elastic_net/validation_predictions.parquet` - SHA-256 `77026f7ffcac58d6a4ba89346aa61e50d25ac3dea43fe57b439c629677ac4249`
- manifest.json: `outputs/quant_deterministic_news/v2/deterministic/d1_d2_elastic_net/manifest.json` - SHA-256 `61c6ee20b283cb526efc59b7ca6fc2dfefc9b26dda5b029d3a80bbc5e6aa79c5`

This is an exploratory development result from retrospective news.
It is not confirmatory point-in-time evidence.
