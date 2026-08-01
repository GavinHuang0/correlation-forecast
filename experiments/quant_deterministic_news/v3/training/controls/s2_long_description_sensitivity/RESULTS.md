# V3-S2 description-length-at-least-150 sensitivity

Status: **complete**.

Claim scope: exploratory failed-semantic-gate development diagnostic; not primary or confirmatory evidence.

## Metrics

| Target | Fisher RMSE | Fisher MAE | OOS R2 vs persistence |
|---|---:|---:|---:|
| t1_etf | 0.371315 | 0.293232 | 0.406688 |
| t1_loo | 0.385551 | 0.303947 | 0.383280 |
| t2_etf | 0.243604 | 0.191042 | 0.234247 |
| t2_loo | 0.262567 | 0.203834 | 0.216842 |

## Integrity

- Review: `passed`.
- Choice-order semantic gate passed: `false`.
- Failed-gate exploratory override: `true`.
- Outer evaluation used for tuning: `false`.
