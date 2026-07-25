# Quant ladder comparison

Development ranking below uses aggregate outer-test Fisher-\(z\) RMSE across
the three locked chronological folds. Lower RMSE is better; OOS \(R^2\) is
measured against the exact target-specific persistence forecast. Since the
same outer blocks rank the rungs, the winning result is not an untouched
confirmatory estimate.

For \(N\) pooled outer-test rows,

\[
\operatorname{RMSE}_z
=
\sqrt{\frac1N\sum_{n=1}^{N}(z_n-\widehat z_n)^2},
\qquad
z_n=\operatorname{atanh}(\operatorname{clip}(\rho_n,-0.995,0.995)),
\]

\[
\operatorname{MAE}_z=\frac1N\sum_n|z_n-\widehat z_n|,
\qquad
\operatorname{RMSE}_{\rho}
=
\sqrt{\frac1N\sum_n(\rho_n-\tanh\widehat z_n)^2},
\]

with raw-correlation MAE defined analogously.

\[
R^2_{OOS}
=
1-
\frac{\sum_n(z_n-\widehat z_n)^2}
{\sum_n(z_n-z^{persist}_n)^2}.
\]

T1 persistence is the immediately preceding session's strict-RTH realized
correlation. T2 persistence is the component-aggregated correlation over the
five completed sessions immediately preceding the forecast. This
benchmark-relative \(R^2\) is not ordinary regression \(R^2\); it can be
negative.

## Best model within each rung

| Target | Rung 1 | Rung 2 | Rung 3 | Rung 4 |
|---|---:|---:|---:|---:|
| T1 ETF | 0.3783 / 0.3843 | 0.3709 / 0.4082 | **0.3686 / 0.4154** | 0.4668 / 0.0624 |
| T1 LOO | 0.3881 / 0.3752 | 0.3830 / 0.3915 | **0.3814 / 0.3964** | 0.4720 / 0.0756 |
| T2 ETF | 0.2442 / 0.2306 | 0.2448 / 0.2264 | **0.2348 / 0.2885** | 0.3369 / -0.4645 |
| T2 LOO | 0.2603 / 0.2301 | 0.2632 / 0.2130 | **0.2567 / 0.2517** | 0.3484 / -0.3791 |

Each cell is `Fisher-z RMSE / OOS R² versus persistence`.

The winning rung-1 models were core LASSO for both T1 targets, SHAR OLS for T2
ETF, and core OLS for T2 LOO. The full extended-hours linear block won rung 2
within each target, although rung 2 did not improve over rung 1 for T2.
Rung-3 XGBoost won overall for three targets. T1 ETF used the linear/tree
ensemble because its interior validation weight strictly beat both components
in every validation fold. T2 ETF also passed the ensemble gate, but standalone
XGBoost had lower aggregate outer-test RMSE and therefore won its development
comparison. The LOO ensembles failed the pre-locked consistency gate and were
omitted.

## Interpretation

- ETF measures tradable hedge coupling and can include the target stock.
- LOO measures coupling to five equal-weight peers and is not a full
  point-in-time sector index.
- T2 is a component-aggregated five-session target, not an average of daily
  correlations.
- Rung 4 uses one daily RTH return per asset to model conditional correlation;
  it does not directly model intraday realized covariance. Its weak T2 result
  should be read as a benchmark limitation, not a failed implementation.
- All 180 DCC systems converged and satisfied the configured
  \(a+b<0.999\) constraint. Fifty-nine DCC fits and ten marginal systems had
  persistence above 0.995, so near-unit persistence is a material diagnostic.
- These are forecast comparisons, not yet statistical-significance or
  trading-profit claims. Date-block inference and an explicit ETF hedge-error
  experiment remain follow-up work.
- Rung 3 reuses a validation block for tree stopping/configuration, linear
  tuning, and ensemble weighting. That is outside the outer test, but a
  separate or cross-fitted calibration layer would be cleaner.
- A future period must remain untouched to confirm the selected model.

The complete machine-readable table is [`summary.json`](summary.json).
