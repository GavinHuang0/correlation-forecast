# V3-S3 fixed wrong-stock semantic control

Status: **complete**.

Claim scope: exploratory failed-semantic-gate development diagnostic; not primary or confirmatory evidence.

## Metrics

| Target | Fisher RMSE | Fisher MAE | OOS R2 vs persistence |
|---|---:|---:|---:|
| t1_etf | 0.372083 | 0.293746 | 0.404229 |
| t1_loo | 0.386347 | 0.304861 | 0.380730 |
| t2_etf | 0.241913 | 0.189482 | 0.244836 |
| t2_loo | 0.264238 | 0.204923 | 0.206844 |

## Integrity

- Review: `passed`.
- Choice-order semantic gate passed: `false`.
- Failed-gate exploratory override: `true`.
- Outer evaluation used for tuning: `false`.
