# Long-Q residual + Coupling6

Status: **complete; selected as the active T2 ETF semantic research overlay**.

Selection is target-specific. The T2 ETF comparison passed its matched-base
point-loss, paired confidence-interval, and fold-count tests. The other three
target routes are archived. This remains an exploratory, retrospective v4
result rather than confirmatory point-in-time evidence.

| Target | Rows | Fisher-z RMSE | OOS R² vs persistence | Incremental MSE R² vs R0 | 95% interval | Status |
|---|---:|---:|---:|---:|---:|---|
| T1 ETF | 18,750 | 0.359157 | 0.420355 | +0.1146% | [-0.0690%, +0.3383%] | Archived target route |
| T1 LOO | 18,750 | 0.370018 | 0.411222 | -0.2545% | [-0.5134%, +0.0580%] | Archived target route |
| T2 ETF | 18,150 | 0.227810 | 0.274982 | **+1.3844%** | **[+0.3429%, +2.6228%]** | **Selected** |
| T2 LOO | 18,150 | 0.240898 | 0.280460 | -0.9851% | [-2.3553%, +0.1826%] | Archived target route |

T2 ETF improved in three of five folds. The paired moving-block bootstrap
used 2,000 fold-contained resamples of whole forecast dates with ten-session
blocks. The bootstrap probability that the incremental gain was nonpositive
was 0.45%.

The selected route adds a shrunk Elastic-Net correction to saved
out-of-sample long-Q XGBoost forecasts. The six inputs are current and
prior-only innovation contrasts between common event mass and target/peer
idiosyncratic event mass for three event families.

Claim limit: RRES-C6 was selected after retrospective v4 evaluation and did
not receive its own complete architecture-matched stale, wrong-stock,
probability-permutation, and quality-only control ladder. Confirmation
requires a prospective, version-safe period strictly after 2026-06-30.

See the [active model registry](../../../../../../models/active/registry.json)
and [complete paired comparison](../../comparisons/final/RESULTS.md).
