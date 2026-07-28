# Deterministic-news v2 final comparison

Status: **complete exploratory development inference**.

Negative paired Fisher-z squared-loss deltas favor the candidate.
Confidence intervals use 2,000 moving-block resamples, with 10 consecutive sessions per block. Blocks are sampled separately within each outer fold and retain all stocks observed on each sampled date.

## Paired comparisons

| comparison | target | incremental_r2 | mean_squared_loss_delta | bootstrap_ci_lower | bootstrap_ci_upper | folds_candidate_better |
|---|---|---|---|---|---|---|
| D1_vs_D0 | t1_etf | -0.448934 | 0.061769 | 0.051352 | 0.079830 | 0 |
| D1_vs_D0 | t1_loo | -0.512944 | 0.076334 | 0.068093 | 0.090647 | 0 |
| D1_vs_D0 | t2_etf | -0.789266 | 0.047315 | 0.038325 | 0.059644 | 0 |
| D1_vs_D0 | t2_loo | -0.920882 | 0.063888 | 0.055510 | 0.073745 | 0 |
| D3_vs_D0 | t1_etf | -0.003100 | 0.000427 | -0.000387 | 0.000946 | 0 |
| D3_vs_D0 | t1_loo | 0.000440 | -0.000066 | -0.000527 | 0.000366 | 1 |
| D3_vs_D0 | t2_etf | 0.021297 | -0.001277 | -0.002043 | -0.000781 | 2 |
| D3_vs_D0 | t2_loo | 0.006740 | -0.000468 | -0.000954 | -0.000049 | 2 |
| D4_vs_D0 | t1_etf | -0.003490 | 0.000480 | -0.000438 | 0.001034 | 0 |
| D4_vs_D0 | t1_loo | 0.006084 | -0.000905 | -0.001587 | -0.000398 | 2 |
| D4_vs_D0 | t2_etf | 0.021374 | -0.001281 | -0.002136 | -0.000672 | 3 |
| D4_vs_D0 | t2_loo | 0.002749 | -0.000191 | -0.000622 | 0.000249 | 2 |
| D4_vs_D3 | t1_etf | -0.000389 | 0.000054 | -0.000188 | 0.000201 | 1 |
| D4_vs_D3 | t1_loo | 0.005646 | -0.000840 | -0.001266 | -0.000562 | 3 |
| D4_vs_D3 | t2_etf | 0.000079 | -0.000005 | -0.000304 | 0.000212 | 2 |
| D4_vs_D3 | t2_loo | -0.004018 | 0.000277 | 0.000077 | 0.000566 | 1 |
| D3_vs_C-D3-L20 | t1_etf | 0.000033 | -0.000005 | -0.000763 | 0.000762 | 2 |
| D3_vs_C-D3-L20 | t1_loo | -0.002700 | 0.000401 | -0.000277 | 0.001122 | 1 |
| D3_vs_C-D3-L20 | t2_etf | 0.001709 | -0.000100 | -0.000636 | 0.000361 | 2 |
| D3_vs_C-D3-L20 | t2_loo | -0.004604 | 0.000316 | -0.000298 | 0.000847 | 1 |
| D3_vs_C-D3-WS | t1_etf | 0.001280 | -0.000177 | -0.000377 | 0.000041 | 2 |
| D3_vs_C-D3-WS | t1_loo | 0.000620 | -0.000092 | -0.000230 | 0.000082 | 3 |
| D3_vs_C-D3-WS | t2_etf | 0.005430 | -0.000320 | -0.000628 | -0.000035 | 3 |
| D3_vs_C-D3-WS | t2_loo | -0.000751 | 0.000052 | -0.000134 | 0.000237 | 1 |
| D5_vs_D0 | t1_etf | 0.011636 | -0.001601 | -0.003646 | 0.000395 | 3 |
| D5_vs_D3 | t1_etf | 0.014690 | -0.002028 | -0.003994 | 0.000181 | 3 |

## D3 fold consistency and falsification controls

| comparison | target | fold | incremental_r2 | mean_squared_loss_delta | candidate_better |
|---|---|---|---|---|---|
| D3_vs_D0 | t1_etf | fold_1 | -0.003713 | 0.000531 | False |
| D3_vs_D0 | t1_etf | fold_2 | -0.003912 | 0.000503 | False |
| D3_vs_D0 | t1_etf | fold_3 | -0.001720 | 0.000244 | False |
| D3_vs_D0 | t1_loo | fold_1 | -0.000015 | 0.000002 | False |
| D3_vs_D0 | t1_loo | fold_2 | -0.000564 | 0.000077 | False |
| D3_vs_D0 | t1_loo | fold_3 | 0.001887 | -0.000281 | True |
| D3_vs_D0 | t2_etf | fold_1 | -0.000489 | 0.000036 | False |
| D3_vs_D0 | t2_etf | fold_2 | 0.060422 | -0.003537 | True |
| D3_vs_D0 | t2_etf | fold_3 | 0.004748 | -0.000223 | True |
| D3_vs_D0 | t2_loo | fold_1 | -0.000829 | 0.000074 | False |
| D3_vs_D0 | t2_loo | fold_2 | 0.015104 | -0.001010 | True |
| D3_vs_D0 | t2_loo | fold_3 | 0.008357 | -0.000439 | True |
| D3_vs_C-D3-L20 | t1_etf | fold_1 | 0.001146 | -0.000165 | True |
| D3_vs_C-D3-L20 | t1_etf | fold_2 | -0.007175 | 0.000919 | False |
| D3_vs_C-D3-L20 | t1_etf | fold_3 | 0.005657 | -0.000807 | True |
| D3_vs_C-D3-L20 | t1_loo | fold_1 | -0.000000 | 0.000000 | False |
| D3_vs_C-D3-L20 | t1_loo | fold_2 | -0.011039 | 0.001496 | False |
| D3_vs_C-D3-L20 | t1_loo | fold_3 | 0.002296 | -0.000343 | True |
| D3_vs_C-D3-L20 | t2_etf | fold_1 | -0.003626 | 0.000269 | False |
| D3_vs_C-D3-L20 | t2_etf | fold_2 | 0.006101 | -0.000338 | True |
| D3_vs_C-D3-L20 | t2_etf | fold_3 | 0.004679 | -0.000220 | True |
| D3_vs_C-D3-L20 | t2_loo | fold_1 | -0.007386 | 0.000653 | False |
| D3_vs_C-D3-L20 | t2_loo | fold_2 | -0.005899 | 0.000386 | False |
| D3_vs_C-D3-L20 | t2_loo | fold_3 | 0.001769 | -0.000092 | True |
| D3_vs_C-D3-WS | t1_etf | fold_1 | -0.000548 | 0.000079 | False |
| D3_vs_C-D3-WS | t1_etf | fold_2 | 0.002574 | -0.000333 | True |
| D3_vs_C-D3-WS | t1_etf | fold_3 | 0.001884 | -0.000268 | True |
| D3_vs_C-D3-WS | t1_loo | fold_1 | 0.000000 | -0.000000 | True |
| D3_vs_C-D3-WS | t1_loo | fold_2 | 0.001427 | -0.000196 | True |
| D3_vs_C-D3-WS | t1_loo | fold_3 | 0.000510 | -0.000076 | True |
| D3_vs_C-D3-WS | t2_etf | fold_1 | 0.003182 | -0.000238 | True |
| D3_vs_C-D3-WS | t2_etf | fold_2 | 0.009050 | -0.000502 | True |
| D3_vs_C-D3-WS | t2_etf | fold_3 | 0.004521 | -0.000212 | True |
| D3_vs_C-D3-WS | t2_loo | fold_1 | -0.001634 | 0.000145 | False |
| D3_vs_C-D3-WS | t2_loo | fold_2 | 0.000019 | -0.000001 | True |
| D3_vs_C-D3-WS | t2_loo | fold_3 | -0.000270 | 0.000014 | False |

## Conservative useful-news gate

D3 must improve on D0, the 20-session-stale D2 control, and the wrong-stock D2 control with an upper 95% bootstrap bound below zero, and must beat D0 in at least two of three folds. This exact Boolean gate is a conservative final-stage operationalization, not a separately preregistered threshold.

| candidate | target | matched_quant_improvement_ci | beats_stale_d2_ci | beats_wrong_stock_d2_ci | improves_at_least_two_of_three_folds | all_required_gates_passed |
|---|---|---|---|---|---|---|
| D3 | t1_etf | False | False | False | False | False |
| D3 | t1_loo | False | False | False | False | False |
| D3 | t2_etf | True | False | True | True | False |
| D3 | t2_loo | True | False | False | True | False |

No target passes every required gate.

## Elastic Net selection stability

| bundle | target | union_selected_count | stable_all_folds_count | d2_union_selected_count | d2_stable_all_folds_count | mean_pairwise_selected_set_jaccard |
|---|---|---|---|---|---|---|
| D3 | t1_etf | 65 | 25 | 24 | 4 | 0.533013 |
| D3 | t1_loo | 76 | 13 | 26 | 0 | 0.374482 |
| D3 | t2_etf | 80 | 15 | 29 | 2 | 0.359924 |
| D3 | t2_loo | 83 | 17 | 30 | 3 | 0.374922 |
| D4 | t1_etf | 65 | 25 | 23 | 3 | 0.533551 |
| D4 | t1_loo | 69 | 13 | 19 | 0 | 0.402032 |
| D4 | t2_etf | 82 | 15 | 28 | 2 | 0.355813 |
| D4 | t2_loo | 88 | 16 | 30 | 2 | 0.352112 |

D5 was compared only for validation-gated target(s): t1_etf.

## Claim boundary

- The V2-D2 matched D43 comparator was not constructable under the v2 source contract and is not silently replaced; this does not refer to the completed D2-Normalized feature block.
- D5 is validation gated and its target subset is not a full four-target model comparison.
- No multiple-testing adjustment is applied across targets or ladder rungs.
- All dates are previously inspected development periods.
- Retrospective ordinary Massive news lacks historical article versions and local first-seen timestamps, so these results are not confirmatory point-in-time evidence.
- Protocol SHA-256: `2e8f8b8534dc9e29f46e10551d8a67d62673194c72135cb5217b556443718467`.
- Bootstrap seed: `1729`.
