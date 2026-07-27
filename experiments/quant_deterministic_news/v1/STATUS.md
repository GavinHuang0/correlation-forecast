# Quant plus deterministic-news v1 status

Protocol SHA-256: `3ed24d17c79b28bf88c614f2c4e8b3143ac7f6d888e6b61f50f37e5980ed772c`

This file is updated after each isolated model bundle. Heavy predictions
and fit records are under `outputs/quant_deterministic_news/v1`.

| Order | Bundle | Track | Status | Attempts | Result |
|---:|---|---|---|---:|---|
| 0 | PRE - Construction preflight | `construction` | **complete** | 3 | 27,510 rows; 917 dates; all hashes/splits passed [results](construction/preflight/RESULTS.md) |
| 1 | A0-L - A0-L Q56 Elastic Net | `track_a_joint` | **complete** | 2 | Fisher-z RMSE: t1_etf 0.3709, t1_loo 0.3858, t2_etf 0.2448, t2_loo 0.2634 [results](track_a_joint/a0_l_q56_elastic_net/RESULTS.md) |
| 2 | A0-T - A0-T Q56 XGBoost | `track_a_joint` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3700, t1_loo 0.3814, t2_etf 0.2348, t2_loo 0.2567 [results](track_a_joint/a0_t_q56_xgboost/RESULTS.md) |
| 3 | A1 - A1 D43 Elastic Net | `track_a_joint` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.4453, t1_loo 0.4735, t2_etf 0.3271, t2_loo 0.3650 [results](track_a_joint/a1_d43_elastic_net/RESULTS.md) |
| 4 | A2 - A2 Q56+D1 Elastic Net | `track_a_joint` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3711, t1_loo 0.3854, t2_etf 0.2431, t2_loo 0.2624 [results](track_a_joint/a2_q56_d1_elastic_net/RESULTS.md) |
| 5 | A3 - A3 Q56+D1+D2 Elastic Net | `track_a_joint` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3714, t1_loo 0.3853, t2_etf 0.2424, t2_loo 0.2615 [results](track_a_joint/a3_q56_d1_d2_elastic_net/RESULTS.md) |
| 6 | A4 - A4 Q56+D1+D2+D3 Elastic Net | `track_a_joint` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3713, t1_loo 0.3854, t2_etf 0.2418, t2_loo 0.2627 [results](track_a_joint/a4_q56_d1_d2_d3_elastic_net/RESULTS.md) |
| 7 | A5 - A5 Q56+D43 Elastic Net | `track_a_joint` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3706, t1_loo 0.3851, t2_etf 0.2403, t2_loo 0.2626 [results](track_a_joint/a5_q56_d43_elastic_net/RESULTS.md) |
| 8 | A6 - A6 Q56+D43 XGBoost | `track_a_joint` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3699, t1_loo 0.3824, t2_etf 0.2342, t2_loo 0.2565 [results](track_a_joint/a6_q56_d43_xgboost/RESULTS.md) |
| 9 | A7 - A7 Elastic Net/XGBoost ensemble | `track_a_joint` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3679, t2_etf 0.2364 [results](track_a_joint/a7_q56_d43_ensemble/RESULTS.md) |
| 10 | B0 - B0 zero correction | `track_b_residual` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3658, t1_loo 0.3751, t2_etf 0.2158, t2_loo 0.2339 [results](track_b_residual/b0_zero_correction/RESULTS.md) |
| 11 | B1 - B1 mean-error correction | `track_b_residual` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3643, t1_loo 0.3776, t2_etf 0.2197, t2_loo 0.2488 [results](track_b_residual/b1_mean_error/RESULTS.md) |
| 12 | B2 - B2 linear calibration | `track_b_residual` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3653, t1_loo 0.3766, t2_etf 0.2211, t2_loo 0.2486 [results](track_b_residual/b2_linear_calibration/RESULTS.md) |
| 13 | B3 - B3 D43 residual Elastic Net | `track_b_residual` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3648, t1_loo 0.3764, t2_etf 0.2175, t2_loo 0.2420 [results](track_b_residual/b3_d43_elastic_net/RESULTS.md) |
| 14 | B4 - B4 forecast+D43 residual Elastic Net | `track_b_residual` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3649, t1_loo 0.3757, t2_etf 0.2187, t2_loo 0.2407 [results](track_b_residual/b4_forecast_d43_elastic_net/RESULTS.md) |
| 15 | B5 - B5 forecast+D43 residual XGBoost | `track_b_residual` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3655, t1_loo 0.3779, t2_etf 0.2196, t2_loo 0.2485 [results](track_b_residual/b5_forecast_d43_xgboost/RESULTS.md) |
| 16 | S-B0 - Winner-base B0 sensitivity | `track_b_winner_sensitivity` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3649 [results](track_b_winner_sensitivity/b0_zero_correction/RESULTS.md) |
| 17 | S-B1 - Winner-base B1 sensitivity | `track_b_winner_sensitivity` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3634 [results](track_b_winner_sensitivity/b1_mean_error/RESULTS.md) |
| 18 | S-B2 - Winner-base B2 sensitivity | `track_b_winner_sensitivity` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3637 [results](track_b_winner_sensitivity/b2_linear_calibration/RESULTS.md) |
| 19 | S-B3 - Winner-base B3 sensitivity | `track_b_winner_sensitivity` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3637 [results](track_b_winner_sensitivity/b3_d43_elastic_net/RESULTS.md) |
| 20 | S-B4 - Winner-base B4 sensitivity | `track_b_winner_sensitivity` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3638 [results](track_b_winner_sensitivity/b4_forecast_d43_elastic_net/RESULTS.md) |
| 21 | S-B5 - Winner-base B5 sensitivity | `track_b_winner_sensitivity` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3632 [results](track_b_winner_sensitivity/b5_forecast_d43_xgboost/RESULTS.md) |
| 22 | C-A5-L20 - A5 stale-news control | `controls` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3694, t1_loo 0.3835, t2_etf 0.2390, t2_loo 0.2607 [results](controls/a5_stale_news_lag20/RESULTS.md) |
| 23 | C-A5-WS - A5 wrong-stock control | `controls` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3711, t1_loo 0.3855, t2_etf 0.2429, t2_loo 0.2632 [results](controls/a5_wrong_stock/RESULTS.md) |
| 24 | C-B4-L20 - B4 stale-news control | `controls` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3643, t1_loo 0.3776, t2_etf 0.2197, t2_loo 0.2471 [results](controls/b4_stale_news_lag20/RESULTS.md) |
| 25 | C-B4-WS - B4 wrong-stock control | `controls` | **complete** | 1 | Fisher-z RMSE: t1_etf 0.3651, t1_loo 0.3757, t2_etf 0.2179, t2_loo 0.2409 [results](controls/b4_wrong_stock/RESULTS.md) |
| 26 | FINAL - Final comparison and inference | `comparisons` | **complete** | 2 | 118 paired target comparisons; 236,000 block-bootstrap draws [results](comparisons/final/RESULTS.md) |

All results are exploratory development estimates because the news
archive is not historical-version-safe and the quant outer blocks were
already inspected in quant v1.
