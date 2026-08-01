# V3-S2 fixed wrong-stock semantic control

Status: **complete**.

Claim scope: exploratory failed-semantic-gate development diagnostic; not primary or confirmatory evidence.

## Metrics

| Target | Fisher RMSE | Fisher MAE | OOS R2 vs persistence |
|---|---:|---:|---:|
| t1_etf | 0.371264 | 0.293254 | 0.406852 |
| t1_loo | 0.386264 | 0.304629 | 0.380997 |
| t2_etf | 0.243445 | 0.190982 | 0.235243 |
| t2_loo | 0.264694 | 0.205422 | 0.204105 |

## Integrity

- Review: `passed`.
- Choice-order semantic gate passed: `false`.
- Failed-gate exploratory override: `true`.
- Outer evaluation used for tuning: `false`.
