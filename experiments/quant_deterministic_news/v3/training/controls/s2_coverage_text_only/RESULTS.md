# V3-S2 coverage/text-quality-only control

Status: **complete**.

Claim scope: exploratory failed-semantic-gate development diagnostic; not primary or confirmatory evidence.

## Metrics

| Target | Fisher RMSE | Fisher MAE | OOS R2 vs persistence |
|---|---:|---:|---:|
| t1_etf | 0.371255 | 0.293243 | 0.406879 |
| t1_loo | 0.385785 | 0.303979 | 0.382531 |
| t2_etf | 0.243377 | 0.190960 | 0.235671 |
| t2_loo | 0.262395 | 0.203601 | 0.217865 |

## Integrity

- Review: `passed`.
- Choice-order semantic gate passed: `false`.
- Failed-gate exploratory override: `true`.
- Outer evaluation used for tuning: `false`.
