# V3-S3 description-length-at-least-150 sensitivity

Status: **complete**.

Claim scope: exploratory failed-semantic-gate development diagnostic; not primary or confirmatory evidence.

## Metrics

| Target | Fisher RMSE | Fisher MAE | OOS R2 vs persistence |
|---|---:|---:|---:|
| t1_etf | 0.371695 | 0.293502 | 0.405472 |
| t1_loo | 0.386000 | 0.304503 | 0.381843 |
| t2_etf | 0.242355 | 0.189776 | 0.242073 |
| t2_loo | 0.263122 | 0.204059 | 0.213525 |

## Integrity

- Review: `passed`.
- Choice-order semantic gate passed: `false`.
- Failed-gate exploratory override: `true`.
- Outer evaluation used for tuning: `false`.
