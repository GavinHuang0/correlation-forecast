# V2-D3 fixed wrong-stock-D2 falsification control

Status: **complete**

Completed: `2026-07-28T06:50:50.969837Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 11190 | 0.371743 | 0.293642 | 0.209197 | 0.405318 |
| t1_loo | 11190 | 0.385801 | 0.304186 | 0.230888 | 0.382479 |
| t2_etf | 10830 | 0.242881 | 0.190367 | 0.129340 | 0.238781 |
| t2_loo | 10830 | 0.262407 | 0.203599 | 0.153205 | 0.217797 |

## Exact matched Q comparison

| Target | Q RMSE | Model RMSE | Incremental R2 | Loss delta | Better folds |
|---|---:|---:|---:|---:|---:|
| t1_etf | 0.370931 | 0.371743 | -0.004386 | 0.00060343 | 0 |
| t1_loo | 0.385767 | 0.385801 | -0.000180 | 0.00002675 | 1 |
| t2_etf | 0.244842 | 0.242881 | 0.015953 | -0.00095636 | 2 |
| t2_loo | 0.263394 | 0.262407 | 0.007485 | -0.00051930 | 3 |

## Falsification control versus contemporaneous V2-D3

| Target | Base RMSE | Model RMSE | Incremental R2 | Loss delta | Better folds |
|---|---:|---:|---:|---:|---:|
| t1_etf | 0.371505 | 0.371743 | -0.001282 | 0.00017688 | 1 |
| t1_loo | 0.385682 | 0.385801 | -0.000620 | 0.00009226 | 0 |
| t2_etf | 0.242221 | 0.242881 | -0.005460 | 0.00032034 | 0 |
| t2_loo | 0.262505 | 0.262407 | 0.000750 | -0.00005170 | 2 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`
- Outer evaluation excluded from tuning/gating: `True`

## Artifacts

- control_audit.json: `outputs/quant_deterministic_news/v2/controls/d3_wrong_stock_d2/control_audit.json` - SHA-256 `56f3f5a540dd8ac408568b52292a6e58181f0c53f2e80b9d91cb79ea2ea398d3`
- fits.json: `outputs/quant_deterministic_news/v2/controls/d3_wrong_stock_d2/fits.json` - SHA-256 `b6a2d67cad2d66ecc8d9bbbf9c240a191b2df04dd20e7839fe096ce11c16d2ac`
- fold_metrics.json: `outputs/quant_deterministic_news/v2/controls/d3_wrong_stock_d2/fold_metrics.json` - SHA-256 `83b7ddf6b2b3bafb12f6919d6876bf771e25780dc79f47a2521898e0b0ff3d22`
- predictions.parquet: `outputs/quant_deterministic_news/v2/controls/d3_wrong_stock_d2/predictions.parquet` - SHA-256 `177b3296abfe7e7156b1ab7d90509157a50f3946b6b0488b5a592152ac28efcc`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v2/controls/d3_wrong_stock_d2/validation_predictions.parquet` - SHA-256 `382d63810a7d6492f754b69bdec5165d4fcd822d74f28db1ed3a7520531a16c1`
- manifest.json: `outputs/quant_deterministic_news/v2/controls/d3_wrong_stock_d2/manifest.json` - SHA-256 `34bb8e0f1b1faebca0a18b72e7c6aaa50bbf1a49d3f44a2f995544235a393cd4`

This is an exploratory development result from retrospective news.
It is not confirmatory point-in-time evidence.
