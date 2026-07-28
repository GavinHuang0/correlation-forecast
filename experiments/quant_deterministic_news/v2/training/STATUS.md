# Deterministic-news v2 training status

Protocol SHA-256: `2e8f8b8534dc9e29f46e10551d8a67d62673194c72135cb5217b556443718467`

Each row is an isolated model bundle. Heavy predictions and fit records
are stored under `outputs/quant_deterministic_news/v2`.

| Order | Bundle | Status | Attempts | Result |
|---:|---|---|---:|---|
| 0 | D0 - V2-D0 matched Q56 Elastic Net | **complete** | 2 | Fisher-z RMSE: t1_etf 0.3709, t1_loo 0.3858, t2_etf 0.2448, t2_loo 0.2634 [results](deterministic/d0_q56_elastic_net/RESULTS.md) |
| 1 | D1 - V2-D1 D2-only Elastic Net | **complete** | 3 | Fisher-z RMSE: t1_etf 0.4465, t1_loo 0.4745, t2_etf 0.3275, t2_loo 0.3651 [results](deterministic/d1_d2_elastic_net/RESULTS.md) |
| 2 | D2 - V2-D2 Q56 plus matched D43 Elastic Net | **skipped** | 2 | The exact D43 comparator was not materialized by the v2 builder; no 43-name contract is fabricated from the historical v1 panel. [results](deterministic/d2_q56_d43_recomputed_elastic_net/RESULTS.md) |
| 3 | D3 - V2-D3 Q56 plus D2-Normalized Elastic Net | **complete** | 2 | Fisher-z RMSE: t1_etf 0.3715, t1_loo 0.3857, t2_etf 0.2422, t2_loo 0.2625 [results](deterministic/d3_q56_d2_elastic_net/RESULTS.md) |
| 4 | D4 - V2-D4 Q56 plus D2-Normalized and D2-Levels Elastic Net | **complete** | 2 | Fisher-z RMSE: t1_etf 0.3716, t1_loo 0.3846, t2_etf 0.2422, t2_loo 0.2630 [results](deterministic/d4_q56_d2_levels_elastic_net/RESULTS.md) |
| 5 | D5 - V2-D5 validation-gated Q56 plus D2-Normalized shallow XGBoost | **complete** | 2 | Validation-gated targets: t1_etf; Fisher-z RMSE: t1_etf 0.3688 [results](deterministic/d5_q56_d2_shallow_xgboost/RESULTS.md) |
| 6 | C-D3-L20 - V2-D3 20-session stale-D2 falsification control | **complete** | 2 | Fisher-z RMSE: t1_etf 0.3715, t1_loo 0.3852, t2_etf 0.2424, t2_loo 0.2619 [results](controls/d3_stale_d2_lag20/RESULTS.md) |
| 7 | C-D3-WS - V2-D3 fixed wrong-stock-D2 falsification control | **complete** | 2 | Fisher-z RMSE: t1_etf 0.3717, t1_loo 0.3858, t2_etf 0.2429, t2_loo 0.2624 [results](controls/d3_wrong_stock_d2/RESULTS.md) |

The [final paired comparison](comparisons/final/RESULTS.md) is complete.
Neither T2 gain versus matched Q56 survived the stale-news gate, and no D3
target passed every required gate. All four semantic arms are
[construction-blocked and untrained](semantic/STATUS.md).

All retrospective-news v2 results are exploratory development
estimates, not confirmatory point-in-time evidence.
