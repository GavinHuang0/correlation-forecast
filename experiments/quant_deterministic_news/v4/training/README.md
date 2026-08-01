# V4 cached-score semantic training results

Status: **complete exploratory development experiment**.

This is the human-readable result record for the separately versioned
[v4 cached-score redesign](../README.md). It does not alter the completed v1,
v2, or v3 protocols or artifacts. The machine-readable v4 protocol was locked
before fitting at SHA-256
`361c235e46bd9322525534da20df98a2f72820670a027739465f620134bc63fe`.

## What was trained

The short-history arm used the unchanged three v3 folds and matched saved v3
Q56/Q56+D2 predictions:

| Live rung | Inputs | Primary comparator |
|---|---|---|
| `J1` | Q56 + SoftRoute19 | saved v3 Q56 Elastic Net |
| `J2` | Q56 + D2 + SoftRoute19 | saved v3 Q56+D2 Elastic Net |
| `J3` | Q56 + nine current soft masses | saved v3 Q56 Elastic Net |

`J1` and `J2` each received stale-20, fixed wrong-stock, complete-score-vector
permutation, and quality-only controls. Every control used the same estimator,
folds, feature count outside the replaced block, and preprocessing scope.

The long-history arm reused the saved quant-v2 XGBoost outer-prediction
stream. Every underlying Q forecast came from an expanding model whose
training period began 2017-12-28; 2016 supplied feature warm-up. V4 did not
pretend that pre-news L features were zero. For test folds 09 through 13, the
residual models trained only on earlier news-era out-of-sample Q errors,
validated on the immediately preceding residual block, and then refit on
those two sets.

| Long rung | Forecast construction |
|---|---|
| `R0` | unchanged saved long-Q XGBoost forecast |
| `RCAL` | OLS calibration using only the base forecast |
| `RRES-C6` | base + shrunk Elastic-Net residual correction from Coupling6 |
| `RRES-L19` | base + shrunk Elastic-Net residual correction from SoftRoute19 |
| `RSTACK` | Elastic Net on base forecast + SoftRoute19 |

`RRES-L19` received the four architecture-matched controls. Correction
shrinkage was selected only on validation from `{0, 0.25, 0.5, 0.75, 1}`.

## Primary matched-base results

Values below are incremental Fisher-z MSE \(R^2\):
`1 - candidate MSE / base MSE`. Positive values favor the semantic model.
Confidence intervals use 2,000 paired ten-session moving-block resamples of
whole dates, sampled separately within each outer fold.

| Candidate vs base | T1 ETF | T1 LOO | T2 ETF | T2 LOO |
|---|---:|---:|---:|---:|
| `J1` vs Q56 | -0.2183% | +0.1740% | +0.7110% | **-1.3561%** |
| `J2` vs Q56+D2 | -0.0099% | +0.1943% | +0.3003% | **-1.2908%** |
| `J3` vs Q56 | -0.1491% | +0.2266% | **+0.6195%** | +0.1904% |
| `RRES-C6` vs long Q | +0.1146% | -0.2545% | **+1.3844%** | -0.9851% |
| `RRES-L19` vs long Q | +0.0935% | +0.0461% | +0.9342% | -0.1121% |
| `RSTACK` vs long Q | -0.5438% | **-1.4675%** | -0.3566% | **-6.4956%** |

Bold values have a paired 95% interval excluding zero in the indicated
direction. `J3` T2 ETF improved in two of three folds with interval
`[+0.0013%, +1.3428%]`. `RRES-C6` T2 ETF improved in three of five folds with
interval `[+0.3429%, +2.6228%]`. Neither diagnostic had the complete
architecture-matched four-control ladder, so neither qualifies as a passed
semantic endpoint. The C6 fold gains were also concentrated: approximately
`+0.317%, -0.152%, 0.000%, +5.182%, +2.150%`, rather than a steady effect.

The base-only `RCAL` generally hurt the long-Q anchor, especially T2 LOO
(-3.2607%). Results versus `RCAL` are therefore secondary; outperforming a
weakened calibration is not evidence of semantic value.

## Falsification results

The full SoftRoute candidates failed decisively:

- `J1` passed zero of five required comparisons for T1 ETF, T1 LOO, and T2
  ETF; it passed only the wrong-stock comparison for T2 LOO while losing to
  its base, stale, permutation, and quality controls.
- `J2` passed at most one of five required comparisons on any target.
- `RRES-L19` passed zero of six requirements for T1 ETF and T2 ETF, two for
  T1 LOO, and one for T2 LOO.
- No target for `J1`, `J2`, or `RRES-L19` passed the complete useful-semantic
  gate.
- `RSTACK` was worse than both `R0` and `RCAL` on most targets and was
  predeclared non-evaluable for the full gate because it has no matched stack
  controls.

Particularly informative failures were:

- `J1` T2 ETF was 1.3219% worse than stale-20, with the interval entirely
  below zero.
- Quality-only beat `J1` on both LOO targets and beat `J2` on T1 LOO and T2
  LOO. This shows that extraction confidence/coverage state can be at least as
  predictive as the event identity.
- The live/permuted and live/wrong-stock comparisons were mixed rather than
  consistently favoring correctly assigned current semantics.
- `RRES-L19` beat its score permutation only for T1 LOO and failed its other
  23 control requirements across the four targets.

The full comparison table and target-level gates are in
[comparisons/final/RESULTS.md](comparisons/final/RESULTS.md). The hash-bound
Parquet, JSON, and provenance records remain under
`outputs/quant_deterministic_news/v4/comparisons/final`.

## Why improvement remains small

The cached-score redesign recovered information discarded by v3 and greatly
reduced accidental cross-stock copying, but the surviving representation is
still dominated by four limitations:

1. **Weak, diffuse labels.** Mean normalized consensus entropy is 0.8293, and
   option-order hard agreement is only 65.12%. Relative score weights preserve
   uncertainty; they do not make the extractor accurate or calibrated.
2. **Routing/volume leakage into semantics.** SoftRoute uses joint event-route
   mass. Its common macro field correlates 0.8143 with D2 common-article share,
   so much of L is a smoother decomposition of already known news flow.
3. **Sparse target-specific information and collinearity.** Target-idiosyncratic
   mass is small and zero on 46.99% of stock-days. Current/innovation pairs
   correlate 0.8077-0.9612; seven components explain 90.20% of their variance.
4. **Missing economic meaning.** Event family alone does not say whether a
   shock makes the stock and sector move together. The cache has no direction,
   surprise magnitude, novelty, materiality, transmission, or causal fields.

These data defects appear in the fitted models: no SoftRoute19 coefficient
sign was stable across all five long-Q folds for any target, and residual
shrinkage ranged from zero to one. With only 917 dependent market dates and a
strong long-Q hurdle, the downstream model cannot manufacture a stable signal
from uncertain, redundant event-family scores. The long-Q **anchor** has
expanding history back to 2017, but the semantic corrector necessarily begins
with the November 2022 inference window; its first long residual fit has only
166 independent training dates and 126 validation dates.

The saved predictions, validation predictions, fits, protocol, source hashes,
and implementation hashes are sufficient for a fail-closed rerun. One
reproducibility limitation remains: `fits.json` records standardized
coefficients but not the fitted imputer medians, scaler statistics, or
intercept. Exact prediction reconstruction therefore requires rerunning the
hash-bound pipeline; a future protocol should serialize each fitted pipeline
or record all preprocessing and intercept parameters.

## Decision

Retain the long-Q model as the production/research baseline. Preserve
`RRES-C6` T2 ETF as a narrowly specified future hypothesis, not as a selected
model. Do not spend more compute rerunning the same four-way FLAN extractor:
the next semantic experiment should add validated target/sector direction,
surprise/materiality, novelty, and transmission labels, preferably on a
prospectively versioned period. Any future confirmation period must start
strictly after the already inspected 2026-06-30 endpoint.

## Verification

- Full repository suite: 457/457 tests passed.
- Focused v4 feature/training suite: 20/20 tests passed.
- The locked protocol sidecar, seven implementation hashes, and 12 unique
  source-artifact records validate.
- Exactly 20 expected bundle manifests exist. Their 160 declared artifacts
  pass SHA-256, byte-size, and Parquet-row checks; every bundle review is
  `passed`.
- Each short bundle contains 44,040 outer-test and 44,640 validation rows;
  each long bundle contains 73,800 outer-test and 74,160 validation rows, with
  unique keys and bitwise-identical actual/persistence values within family.
- `R0` is an exact 73,800-row identity projection of the locked quant-v2
  XGBoost OOS anchor. `J1` and `J2` exactly match the v3 S0 and S1 comparison
  keys and actual targets.
- The final comparison contains 92 unique target-comparison rows and 16 unique
  useful-gate rows. Its manifest and all five declared artifacts validate.
- Prior v1-v3 manifests/artifacts passed their available hash checks, and no
  prior tracked experiment file was modified.
