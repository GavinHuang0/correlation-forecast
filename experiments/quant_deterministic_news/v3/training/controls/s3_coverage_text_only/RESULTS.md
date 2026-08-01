# V3-S3 coverage/text-quality-only control

Status: **complete**.

Claim scope: exploratory failed-semantic-gate development diagnostic; not primary or confirmatory evidence.

## Metrics

| Target | Fisher RMSE | Fisher MAE | OOS R2 vs persistence |
|---|---:|---:|---:|
| t1_etf | 0.371671 | 0.293552 | 0.405548 |
| t1_loo | 0.385844 | 0.304257 | 0.382342 |
| t2_etf | 0.242233 | 0.189783 | 0.242840 |
| t2_loo | 0.263120 | 0.204184 | 0.213538 |

## Integrity

- Review: `passed`.
- Choice-order semantic gate passed: `false`.
- Failed-gate exploratory override: `true`.
- Outer evaluation used for tuning: `false`.
