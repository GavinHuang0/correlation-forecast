# V2-D4 Q56 plus D2-Normalized and D2-Levels Elastic Net

Status: **complete**

Completed: `2026-07-28T06:49:56.695501Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 11190 | 0.371577 | 0.293461 | 0.209125 | 0.405848 |
| t1_loo | 11190 | 0.384592 | 0.303240 | 0.229567 | 0.386347 |
| t2_etf | 10830 | 0.242211 | 0.189875 | 0.129160 | 0.242975 |
| t2_loo | 10830 | 0.263032 | 0.204163 | 0.153638 | 0.214065 |

## Exact matched Q comparison

| Target | Q RMSE | Model RMSE | Incremental R2 | Loss delta | Better folds |
|---|---:|---:|---:|---:|---:|
| t1_etf | 0.370931 | 0.371577 | -0.003490 | 0.00048021 | 0 |
| t1_loo | 0.385767 | 0.384592 | 0.006084 | -0.00090539 | 2 |
| t2_etf | 0.244842 | 0.242211 | 0.021374 | -0.00128135 | 3 |
| t2_loo | 0.263394 | 0.263032 | 0.002749 | -0.00019071 | 2 |

## D2-Levels sensitivity versus D2-Normalized

| Target | Base RMSE | Model RMSE | Incremental R2 | Loss delta | Better folds |
|---|---:|---:|---:|---:|---:|
| t1_etf | 0.371505 | 0.371577 | -0.000389 | 0.00005365 | 1 |
| t1_loo | 0.385682 | 0.384592 | 0.005646 | -0.00083987 | 3 |
| t2_etf | 0.242221 | 0.242211 | 0.000079 | -0.00000465 | 2 |
| t2_loo | 0.262505 | 0.263032 | -0.004018 | 0.00027690 | 1 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`
- Outer evaluation excluded from tuning/gating: `True`

## Artifacts

- fits.json: `outputs/quant_deterministic_news/v2/deterministic/d4_q56_d2_levels_elastic_net/fits.json` - SHA-256 `0b20c6c637b0ad5e6ca0040bf0145502f917d5dcb36d1b9dfeff8df81c5f8be3`
- fold_metrics.json: `outputs/quant_deterministic_news/v2/deterministic/d4_q56_d2_levels_elastic_net/fold_metrics.json` - SHA-256 `4192b12aa2dab93a9235941689c7666546f1fa1f15954a081ed91cd1b29c2638`
- predictions.parquet: `outputs/quant_deterministic_news/v2/deterministic/d4_q56_d2_levels_elastic_net/predictions.parquet` - SHA-256 `7d30ffed93582daa164afa19cc489af93569724a462df0c1851fe8d2360c0bd6`
- source_preflight.json: `outputs/quant_deterministic_news/v2/deterministic/d4_q56_d2_levels_elastic_net/source_preflight.json` - SHA-256 `1de5ccfa2ab8dceea0903c6211606b225a08139c11b2fb91aae9edb0020ca629`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v2/deterministic/d4_q56_d2_levels_elastic_net/validation_predictions.parquet` - SHA-256 `b5330b914f34875c8bb8efee8da5ce41e6793afd4ff0496a34c03ff35571069f`
- manifest.json: `outputs/quant_deterministic_news/v2/deterministic/d4_q56_d2_levels_elastic_net/manifest.json` - SHA-256 `b10eff6e5ad23072a4ac192414601c7e7490dfab9d9dc0ae65c6ed8eab18c2df`

This is an exploratory development result from retrospective news.
It is not confirmatory point-in-time evidence.
