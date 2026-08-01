# V3-S3 20-session stale-event control

Status: **complete**.

Claim scope: exploratory failed-semantic-gate development diagnostic; not primary or confirmatory evidence.

## Metrics

| Target | Fisher RMSE | Fisher MAE | OOS R2 vs persistence |
|---|---:|---:|---:|
| t1_etf | 0.371606 | 0.293446 | 0.405757 |
| t1_loo | 0.385832 | 0.304264 | 0.382381 |
| t2_etf | 0.241935 | 0.189393 | 0.244699 |
| t2_loo | 0.264277 | 0.205024 | 0.206607 |

## Integrity

- Review: `passed`.
- Choice-order semantic gate passed: `false`.
- Failed-gate exploratory override: `true`.
- Outer evaluation used for tuning: `false`.
