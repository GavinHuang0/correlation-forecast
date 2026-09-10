# Forecasting Stock-Sector Coupling

This repository studies whether the co-movement between an individual stock
and its sector can be forecast from lagged market data, and whether
target-relative financial-news semantics add value beyond a strong
quantitative baseline.

The project now has a selected public research suite:

- four target-specific **long-Q** quantitative models, chosen from a
  13-fold expanding-window evaluation; and
- **RRES-C6**, a compact semantic residual overlay selected only for the
  five-session stock/ETF target, where its matched-base improvement was
  statistically significant under the stored paired moving-block bootstrap.

The canonical model routing is in
[`models/active/registry.json`](models/active/registry.json). Nonselected
forecast families are cataloged in
[`models/archive/registry.json`](models/archive/registry.json), while their
original experiment records remain in place for reproducibility.

## Main result

### Selected long-Q models

The full-history quantitative experiment contains 58,500 stock-date rows for
30 stocks and 2,136 represented dates. Thirteen non-overlapping outer tests
span 2020-H1 through 2026-H1.

| Target | Selected model | Test rows | Fisher-z RMSE | Raw-correlation RMSE | OOS R² vs persistence |
|---|---|---:|---:|---:|---:|
| T1 ETF | Rung 3 XGBoost | 44,058 | **0.3577** | 0.1894 | 0.3921 |
| T1 LOO | Rung 3 XGBoost | 44,058 | **0.3657** | 0.2118 | 0.3891 |
| T2 ETF | Rung 3 Elastic Net/XGBoost ensemble | 42,030 | **0.2330** | 0.1180 | 0.2200 |
| T2 LOO | Rung 1 core-22 LASSO | 42,030 | **0.2427** | 0.1382 | 0.2404 |

The best model is target-dependent. XGBoost wins both same-day targets; the
validation-qualified linear/tree ensemble wins the five-session ETF target;
and the compact LASSO wins the five-session leave-one-out peer target.

### Selected RRES-C6 semantic overlay

RRES-C6 corrects saved out-of-sample long-Q XGBoost forecasts with six
low-dimensional semantic features. On T2 ETF it produced:

| Comparison | Incremental Fisher-z MSE R² | Paired 95% interval | Folds improved |
|---|---:|---:|---:|
| RRES-C6 vs matched long-Q XGBoost | **+1.3844%** | **[+0.3429%, +2.6228%]** | **3/5** |

The evaluation contains 18,150 rows over 605 dates from five prequential
semantic-era folds. RRES-C6 achieved Fisher-z RMSE 0.2278 and OOS R² 0.2750
against persistence on that sample. The paired bootstrap probability that
the incremental gain was nonpositive was 0.45%.

This promotion is deliberately narrow. RRES-C6 is active only as a **T2 ETF
research overlay**. It did not improve the other targets consistently, and
its pooled metrics are not directly comparable with the 13-fold T2 ETF
ensemble because the sample and matched comparator differ.

The result is statistically significant for the stored matched-base test but
is not yet confirmatory evidence: the news source is retrospective rather
than historical-version-safe, the evaluation dates had been inspected in
earlier experiment versions, and the selected C6 architecture did not receive
its own complete matched stale, wrong-stock, score-permutation, and
quality-only falsification ladder. A prospective confirmation period must
start after 2026-06-30.

## What is being forecast

For each stock $`i`$, sector reference $`s`$, and forecast date $`t`$, the response
is realized correlation transformed to Fisher-z space:

```math
z_{i,t}=\mathrm{atanh}(\rho_{i,t}).
```

The repository models four related targets:

| Target | Definition | Interpretation |
|---|---|---|
| T1 ETF | Same-day regular-session stock/sector-ETF correlation | Tradable same-day hedge coupling |
| T1 LOO | Same-day correlation with an equal-weight basket of five non-target sector peers | Research target without ETF self-inclusion |
| T2 ETF | Correlation formed from covariance and variance components accumulated over the current and next four official sessions | Five-session ETF hedge coupling |
| T2 LOO | Five-session component-aggregated correlation with the leave-one-out peer basket | Five-session peer-factor coupling |

T2 is not an average of daily correlations. Covariance and variance
components are summed over the five-session horizon and normalized once.
Rows whose forward target crosses a fold boundary are purged.

## Quantitative model ladder

The long-Q experiment preserves strict information timing: every feature for
date $`t`$ ends no later than the preceding official session, preprocessing is
fit only on the permitted training data, hyperparameters are selected only on
the immediately preceding validation block, and all stocks on a date remain
in the same fold.

The four evaluated rungs are:

1. persistence, HAR/SHAR regressions, core OLS, and core LASSO;
2. expanded LASSO and Elastic-Net variants using dense, volatility, and
   extended-hours blocks;
3. shallow XGBoost and validation-qualified Elastic Net/XGBoost ensembles;
4. causal DCC-GARCH conditional-correlation benchmarks.

The 22-feature core includes HAR/downside correlation features,
finite-window exponential stock/ETF pair states, and within-sector
exponential states. Additional quantitative inputs cover overnight and
premarket behavior, realized volatility, relative volume, sector dispersion,
calendar/macro release state, and simplified factor-implied correlation.

The raw price archive begins in 2016. A 500-session warm-up makes the first
model-ready date 2017-12-28. The modeling panel ends on 2026-06-30.

## Semantic-news program

The news work progressed through four versioned downstream experiments:

| Version | What was tested | Outcome |
|---|---|---|
| v1 | Deterministic Q+D features, ablations, residual models, and placebos | Some T2 gains, but stale-news controls performed as well or better |
| v2 | Normalized deterministic D2 plus planned weak/rich semantic contracts | D2 completed; no target passed all robustness gates; full semantic workloads were deferred |
| v3 | Cost-bounded FLAN W17-Lite Q+L and Q+D+L | T2 ETF improved 1.2015%, but the result failed coverage, stale, wrong-stock, and permutation checks |
| v4 | Cached soft score maps, target-relative routing, innovations, compact residual corrections, and controls | RRES-C6 T2 ETF passed its matched statistical test; full SoftRoute19 models failed the complete semantic-usefulness gate |

V4 reused all 50,488 cached FLAN-T5-XL article inferences rather than running
the language model again. SoftRoute19 represents three event families across
three mutually exclusive target-relative roles:

- $`I`$: target-idiosyncratic;
- $`P`$: peer-idiosyncratic; and
- $`C`$: sector- or macro-common.

For event family $`k`$, the current joint mass is

```math
M_{i,t,r,k}=
\frac{\sum_a w_{i,a,t}\,\mathbf{1}[r_{i,a,t}=r]p_{a,k}}
{\sum_a w_{i,a,t}},
```

where $`p_{a,k}`$ is the average of canonical- and reversed-order normalized
candidate scores and $`w_{i,a,t}`$ is the frozen recency/duplication weight.
Prior-only 63-session exponentially weighted baselines produce innovations
$`\Delta M_{i,t,r,k}`$.

RRES-C6 compresses those features to current and innovation contrasts for the
three event families:

```math
K_{i,t,k}=M_{i,t,C,k}-M_{i,t,I,k}-M_{i,t,P,k},
```

```math
\Delta K_{i,t,k}=\Delta M_{i,t,C,k}-\Delta M_{i,t,I,k}-\Delta M_{i,t,P,k}.
```

For each forecast, the semantic model learns a shrunk Elastic-Net correction
to an already out-of-sample quant forecast:

```math
\widehat z^{\mathrm{RRES}}_{i,t}
=\widehat z^{Q,\mathrm{long}}_{i,t}
+\lambda\,\widehat e^{\mathrm{C6}}_{i,t},
```

where $`\lambda\in\{0,0.25,0.5,0.75,1\}`$ is chosen only on the preceding
validation block. Residual training uses earlier out-of-sample Q errors; no
in-sample quant residual enters the corrector.

## What did not work

Negative results are retained because they materially constrain the claim:

- Full SoftRoute19 joint models did not pass the matched-base and four-control
  gate on any target.
- RRES-L19's T2 ETF point estimate was +0.9342%, but its paired interval
  crossed zero.
- The direct long-Q plus SoftRoute19 stack worsened every target, including a
  6.4956% T2 LOO MSE deterioration.
- The v4 J1 and J2 models significantly worsened T2 LOO.
- The semantic score maps remain diffuse: mean normalized entropy is 0.8293
  and canonical/reversed hard-label agreement is 65.12%.
- Target-specific event mass is sparse; all target-specific current masses
  are zero on 46.99% of stock-days.
- Current/innovation features are highly collinear, and seven principal
  components explain 90.20% of their standardized variance.
- The ontology lacks validated direction, surprise magnitude, materiality,
  novelty, and causal transmission labels.

These failures explain why a compact, heavily regularized residual contrast
can show a narrow T2 ETF gain while larger semantic feature blocks do not
generalize reliably.

## Repository map

| Path | Purpose |
|---|---|
| [`models/active/`](models/active/README.md) | Selected model cards and machine-readable routing |
| [`models/archive/`](models/archive/README.md) | Nonselected forecast catalog and archive policy |
| [`docs/`](docs/README.md) | Current methodology, results, reproduction, and technical references |
| [`config/`](config/) | Frozen universes, schemas, and experiment protocols |
| [`scripts/`](scripts/) | Data construction, training, comparison, and audit code |
| [`experiments/`](experiments/README.md) | Versioned protocols, result summaries, controls, and artifact pointers |
| [`annotations/`](annotations/README.md) | Public semantic reference labels without licensed article text |
| [`tests/`](tests/) | Unit, contract, leakage, manifest, and registry tests |

Generated `data/`, `outputs/`, licensed/raw news, model weights, local
environments, and secrets are intentionally Git-ignored. Tracked summaries
contain hashes and artifact pointers so a complete local run can be audited
without publishing restricted inputs.

## Reproduction

The public reproduction guide is
[`docs/reproducibility.md`](docs/reproducibility.md). The two principal frozen
protocols are:

- [`config/quant_training_protocol_v2.json`](config/quant_training_protocol_v2.json)
- [`config/quant_deterministic_news_protocol_v4.json`](config/quant_deterministic_news_protocol_v4.json)

The complete quantitative ranking is
[`experiments/quant_training/v2/comparisons/summary.json`](experiments/quant_training/v2/comparisons/summary.json).
The complete 92-row v4 comparison is
[`experiments/quant_deterministic_news/v4/training/comparisons/final/RESULTS.md`](experiments/quant_deterministic_news/v4/training/comparisons/final/RESULTS.md).

## Interpretation boundary

This repository reports forecasting research, not a trading strategy or
investment recommendation. “Active” denotes repository selection status.
Every reported winner is a development result until it is evaluated on a
new, untouched, prospectively versioned period.
