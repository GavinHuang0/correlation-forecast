# V3-S2 20-session stale-event control

Status: **complete**.

Claim scope: exploratory failed-semantic-gate development diagnostic; not primary or confirmatory evidence.

## Metrics

| Target | Fisher RMSE | Fisher MAE | OOS R2 vs persistence |
|---|---:|---:|---:|
| t1_etf | 0.370993 | 0.293068 | 0.407717 |
| t1_loo | 0.385823 | 0.304107 | 0.382410 |
| t2_etf | 0.242728 | 0.190389 | 0.239739 |
| t2_loo | 0.264172 | 0.204931 | 0.207241 |

## Integrity

- Review: `passed`.
- Choice-order semantic gate passed: `false`.
- Failed-gate exploratory override: `true`.
- Outer evaluation used for tuning: `false`.
