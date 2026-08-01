# Active forecasting suite

Status: **selected for continued research and prospective evaluation**.

The active suite contains four target-specific long-history quantitative
models and one narrowly scoped semantic residual overlay. The canonical
machine-readable definition is [`registry.json`](registry.json).

## Target routing

| Target | Primary quantitative model | Optional semantic overlay |
|---|---|---|
| T1 ETF | Long-Q Rung 3 XGBoost | None |
| T1 LOO | Long-Q Rung 3 XGBoost | None |
| T2 ETF | Long-Q Rung 3 Elastic Net/XGBoost ensemble | RRES-C6 |
| T2 LOO | Long-Q Rung 1 core-22 LASSO | None |

RRES-C6 is not a replacement for the T2 ETF ensemble in the full-history
ranking. It is a selected research overlay trained against the saved
long-history XGBoost out-of-sample anchor on the five semantic-era folds.
Because those samples and comparators differ, its pooled RMSE must not be
compared directly with the 13-fold long-Q ensemble RMSE.

## Long-Q selection results

The four winners were selected separately for each target from 13
non-overlapping expanding-window outer tests spanning 2020-H1 through
2026-H1.

| Target | Selected model | Rows | Fisher-z MAE | Fisher-z RMSE | Raw-correlation RMSE | OOS R² vs persistence |
|---|---|---:|---:|---:|---:|---:|
| T1 ETF | Rung 3 XGBoost | 44,058 | 0.2837 | 0.3577 | 0.1894 | 0.3921 |
| T1 LOO | Rung 3 XGBoost | 44,058 | 0.2898 | 0.3657 | 0.2118 | 0.3891 |
| T2 ETF | Rung 3 Elastic Net/XGBoost ensemble | 42,030 | 0.1834 | 0.2330 | 0.1180 | 0.2200 |
| T2 LOO | Rung 1 core-22 LASSO | 42,030 | 0.1912 | 0.2427 | 0.1382 | 0.2404 |

These are retrospective development estimates. The outer tests were used to
rank model rungs, and no untouched confirmatory period remains in the stored
sample.

## RRES-C6 promotion result

RRES-C6 adds an Elastic-Net correction to the saved long-Q XGBoost forecast.
Its six semantic predictors contrast common-news event mass with target- and
peer-idiosyncratic event mass for three event families, using both current
mass and a prior-only innovation.

For T2 ETF, against the matched long-Q XGBoost anchor:

| Measure | Result |
|---|---:|
| Evaluation rows | 18,150 |
| Evaluation dates | 605 |
| Prequential outer folds | 5 |
| Fisher-z RMSE | 0.2278 |
| OOS R² vs persistence | 0.2750 |
| Incremental Fisher-z MSE R² vs matched long-Q anchor | **+1.3844%** |
| Paired moving-block 95% interval | **[+0.3429%, +2.6228%]** |
| Folds improved | **3 of 5** |
| Bootstrap probability that the gain was nonpositive | 0.45% |

The point-loss, confidence-interval, and fold-count tests all passed for this
matched comparison. Promotion is nevertheless limited to a research overlay:
the result is retrospective, the news archive is not historical-version-safe,
the dates had been inspected in earlier experiments, and RRES-C6 was selected
without its own complete architecture-matched stale, wrong-stock,
probability-permutation, and quality-only control ladder. Prospective
confirmation must begin after 2026-06-30.

## Evidence

- Long-Q protocol and complete ladder:
  [`../../experiments/quant_training/v2/README.md`](../../experiments/quant_training/v2/README.md)
- Long-Q machine-readable ranking:
  [`../../experiments/quant_training/v2/comparisons/summary.json`](../../experiments/quant_training/v2/comparisons/summary.json)
- RRES-C6 model card:
  [`../../experiments/quant_deterministic_news/v4/training/models/rres_coupling6_en/RESULTS.md`](../../experiments/quant_deterministic_news/v4/training/models/rres_coupling6_en/RESULTS.md)
- Complete semantic comparison report:
  [`../../experiments/quant_deterministic_news/v4/training/comparisons/final/RESULTS.md`](../../experiments/quant_deterministic_news/v4/training/comparisons/final/RESULTS.md)
