# V3-S2 date-sector semantic permutation control

Status: **complete**.

Claim scope: exploratory failed-semantic-gate development diagnostic; not primary or confirmatory evidence.

## Metrics

| Target | Fisher RMSE | Fisher MAE | OOS R2 vs persistence |
|---|---:|---:|---:|
| t1_etf | 0.370796 | 0.292888 | 0.408345 |
| t1_loo | 0.385722 | 0.303986 | 0.382733 |
| t2_etf | 0.243117 | 0.190735 | 0.237301 |
| t2_loo | 0.264104 | 0.204898 | 0.207645 |

## Integrity

- Review: `passed`.
- Choice-order semantic gate passed: `false`.
- Failed-gate exploratory override: `true`.
- Outer evaluation used for tuning: `false`.
