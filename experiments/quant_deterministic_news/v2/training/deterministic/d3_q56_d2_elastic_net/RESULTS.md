# V2-D3 Q56 plus D2-Normalized Elastic Net

Status: **complete**

Completed: `2026-07-28T06:49:41.856308Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 11190 | 0.371505 | 0.293429 | 0.209065 | 0.406079 |
| t1_loo | 11190 | 0.385682 | 0.304122 | 0.230799 | 0.382862 |
| t2_etf | 10830 | 0.242221 | 0.189746 | 0.129093 | 0.242915 |
| t2_loo | 10830 | 0.262505 | 0.203603 | 0.153240 | 0.217210 |

## Exact matched Q comparison

| Target | Q RMSE | Model RMSE | Incremental R2 | Loss delta | Better folds |
|---|---:|---:|---:|---:|---:|
| t1_etf | 0.370931 | 0.371505 | -0.003100 | 0.00042656 | 0 |
| t1_loo | 0.385767 | 0.385682 | 0.000440 | -0.00006551 | 1 |
| t2_etf | 0.244842 | 0.242221 | 0.021297 | -0.00127670 | 2 |
| t2_loo | 0.263394 | 0.262505 | 0.006740 | -0.00046761 | 2 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`
- Outer evaluation excluded from tuning/gating: `True`

## Artifacts

- fits.json: `outputs/quant_deterministic_news/v2/deterministic/d3_q56_d2_elastic_net/fits.json` - SHA-256 `504996e4c1ebc11c5d36d90f0ef44069a626142c5435843275d4b0e142340a36`
- fold_metrics.json: `outputs/quant_deterministic_news/v2/deterministic/d3_q56_d2_elastic_net/fold_metrics.json` - SHA-256 `5476ce9e8238a91c489031d492f3400ad2fc660297adacbba6439db9e857151a`
- predictions.parquet: `outputs/quant_deterministic_news/v2/deterministic/d3_q56_d2_elastic_net/predictions.parquet` - SHA-256 `38e3099ccfd8f265766836fcfea26d76f4fd2b705270874c856fd25083297c92`
- source_preflight.json: `outputs/quant_deterministic_news/v2/deterministic/d3_q56_d2_elastic_net/source_preflight.json` - SHA-256 `7a4bb0189129f79186f05a9f7a5f15e217b6d1e40dccbdfc60530206d8b4026d`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v2/deterministic/d3_q56_d2_elastic_net/validation_predictions.parquet` - SHA-256 `7bad521497fc8e1ddc18477ba19ea0b578dd9e89ba2626105b943892cd6cc6f7`
- manifest.json: `outputs/quant_deterministic_news/v2/deterministic/d3_q56_d2_elastic_net/manifest.json` - SHA-256 `4faf3200218ff6644da59e59eb6927e60344e4e12c53b8cc836063514c7c1125`

This is an exploratory development result from retrospective news.
It is not confirmatory point-in-time evidence.
