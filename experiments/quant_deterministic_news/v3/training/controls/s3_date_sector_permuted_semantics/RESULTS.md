# V3-S3 date-sector semantic permutation control

Status: **complete**.

Claim scope: exploratory failed-semantic-gate development diagnostic; not primary or confirmatory evidence.

## Metrics

| Target | Fisher RMSE | Fisher MAE | OOS R2 vs persistence |
|---|---:|---:|---:|
| t1_etf | 0.371453 | 0.293292 | 0.406245 |
| t1_loo | 0.385750 | 0.304171 | 0.382645 |
| t2_etf | 0.241630 | 0.189276 | 0.246604 |
| t2_loo | 0.263773 | 0.204570 | 0.209632 |

## Integrity

- Review: `passed`.
- Choice-order semantic gate passed: `false`.
- Failed-gate exploratory override: `true`.
- Outer evaluation used for tuning: `false`.
