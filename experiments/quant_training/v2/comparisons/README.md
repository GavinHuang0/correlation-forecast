# Full-history quant ladder comparison

The v2 ladder is complete across 13 expanding folds. Six-month outer tests run
continuously from 2020-H1 through 2026-H1; preprocessing and tuning remain
inside each training/validation information set, and T2 labels crossing a
block boundary are purged.

## Pooled outer-test winners

| Target | Rung/model | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R² vs persistence |
|---|---|---:|---:|---:|---:|---:|
| T1 ETF | Rung 3 XGBoost | 44,058 | 0.3577 | 0.2837 | 0.1894 | 0.3921 |
| T1 LOO | Rung 3 XGBoost | 44,058 | 0.3657 | 0.2898 | 0.2118 | 0.3891 |
| T2 ETF | Rung 3 Elastic Net/XGBoost ensemble | 42,030 | 0.2330 | 0.1834 | 0.1180 | 0.2200 |
| T2 LOO | Rung 1 core-22 LASSO | 42,030 | 0.2427 | 0.1912 | 0.1382 | 0.2404 |

Rung 2 does not win a target over the full period. Rung 4 DCC-GARCH has
Fisher-z RMSE of 0.4708, 0.4522, 0.3499, and 0.3296 for T1 ETF, T1 LOO,
T2 ETF, and T2 LOO respectively. Only T1 LOO remains above persistence
(`OOS R² = 0.0657`); the other three DCC results are below persistence.

## Like-for-like recent-period effect

The final three v2 folds use the exact v1 validation and outer-test periods.
The table compares XGBoost on the common 2025-01-01 through 2026-06-30 test
sample, changing the available training history rather than the estimator.

| Target | v1 RMSE | v2 RMSE | Absolute delta | Relative delta | v1 OOS R² | v2 OOS R² |
|---|---:|---:|---:|---:|---:|---:|
| T1 ETF | 0.369951 | 0.367163 | -0.002788 | -0.75% | 0.4110 | 0.4199 |
| T1 LOO | 0.381434 | 0.376284 | -0.005150 | -1.35% | 0.3964 | 0.4126 |
| T2 ETF | 0.234820 | 0.235940 | +0.001120 | +0.48% | 0.2885 | 0.2817 |
| T2 LOO | 0.256653 | 0.249337 | -0.007317 | -2.85% | 0.2517 | 0.2938 |

Three targets benefit from the longer training history on the common recent
sample, most clearly T2 LOO. T2 ETF XGBoost is slightly worse, although the
v2 validation-qualified ensemble reaches 0.234796 on those blocks.

## Interpretation limits

- Prices begin in 2016, but the unchanged 500-session core warm-up makes
  2017-12-28 the first model-ready date.
- The complete-core filter produces an unbalanced early evaluation panel.
  DCC state estimation therefore uses the separate dense 2016–2026 daily
  return-history panel before forecasts are joined to eligible evaluation rows.
- Extended-hours values begin on 2022-11-01. They remain present as declared
  columns but have zero learned influence in training folds where every value
  is missing.
- XGBoost ran on CPU because the GPU was occupied by a pre-existing user
  workload. The model grid and statistical design were unchanged.
- This is retrospective development evidence, not a new confirmation. The v1
  results influenced the reused ladder, and its three outer periods appear
  again as v2 folds 11–13.

The complete metrics, per-rung winners, and hashes for every required artifact
are in [`summary.json`](summary.json).
