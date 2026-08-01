# V3 W17-Lite final comparison

Status: **exploratory failed-semantic-gate development diagnostic**.

FLAN W17-Lite canonical/reversed choice-order agreement missed the frozen 85% semantic-quality threshold. The user-authorized fits and every comparison below remain diagnostic, even if a downstream usefulness gate passes.

## Primary matched comparisons

| comparison | target | incremental_r2_vs_base | candidate_minus_base_mse | bootstrap_ci_lower | bootstrap_ci_upper | improved_development_folds |
|---|---|---|---|---|---|---|
| S2_vs_S0 | t1_etf | -0.00167060 | 0.00022986 | -0.00052916 | 0.00086839 | 1 |
| S2_vs_S0 | t1_loo | -0.00264650 | 0.00039384 | -0.00020019 | 0.00101690 | 0 |
| S2_vs_S0 | t2_etf | 0.01201507 | -0.00072028 | -0.00119571 | -0.00033155 | 3 |
| S2_vs_S0 | t2_loo | -0.00984215 | 0.00068281 | 0.00002553 | 0.00138221 | 1 |
| S3_vs_S1 | t1_etf | -0.00273793 | 0.00037788 | -0.00031436 | 0.00107822 | 1 |
| S3_vs_S1 | t1_loo | -0.00349406 | 0.00051974 | -0.00008294 | 0.00117022 | 1 |
| S3_vs_S1 | t2_etf | 0.00236218 | -0.00013859 | -0.00054667 | 0.00028309 | 3 |
| S3_vs_S1 | t2_loo | -0.01385909 | 0.00095502 | 0.00046603 | 0.00168548 | 0 |

## All paired comparisons and controls

| comparison | target | incremental_r2_vs_base | candidate_minus_base_mse | bootstrap_ci_lower | bootstrap_ci_upper | improved_development_folds |
|---|---|---|---|---|---|---|
| S1_vs_S0 | t1_etf | -0.00310021 | 0.00042656 | -0.00032993 | 0.00093574 | 0 |
| S1_vs_S0 | t1_loo | 0.00044021 | -0.00006551 | -0.00053838 | 0.00033719 | 1 |
| S1_vs_S0 | t2_etf | 0.02129685 | -0.00127670 | -0.00203438 | -0.00073527 | 2 |
| S1_vs_S0 | t2_loo | 0.00674010 | -0.00046761 | -0.00094432 | -0.00003359 | 2 |
| S2_vs_S0 | t1_etf | -0.00167060 | 0.00022986 | -0.00052916 | 0.00086839 | 1 |
| S2_vs_S0 | t1_loo | -0.00264650 | 0.00039384 | -0.00020019 | 0.00101690 | 0 |
| S2_vs_S0 | t2_etf | 0.01201507 | -0.00072028 | -0.00119571 | -0.00033155 | 3 |
| S2_vs_S0 | t2_loo | -0.00984215 | 0.00068281 | 0.00002553 | 0.00138221 | 1 |
| S3_vs_S1 | t1_etf | -0.00273793 | 0.00037788 | -0.00031436 | 0.00107822 | 1 |
| S3_vs_S1 | t1_loo | -0.00349406 | 0.00051974 | -0.00008294 | 0.00117022 | 1 |
| S3_vs_S1 | t2_etf | 0.00236218 | -0.00013859 | -0.00054667 | 0.00028309 | 3 |
| S3_vs_S1 | t2_loo | -0.01385909 | 0.00095502 | 0.00046603 | 0.00168548 | 0 |
| S3_vs_S2 | t1_etf | -0.00416906 | 0.00057458 | -0.00013977 | 0.00109285 | 0 |
| S3_vs_S2 | t1_loo | -0.00040474 | 0.00006039 | -0.00039239 | 0.00043237 | 2 |
| S3_vs_S2 | t2_etf | 0.01173465 | -0.00069501 | -0.00131083 | -0.00021376 | 2 |
| S3_vs_S2 | t2_loo | 0.00278913 | -0.00019540 | -0.00069752 | 0.00034347 | 2 |
| S2_vs_C-S2-COV | t1_etf | 0.00007887 | -0.00001087 | -0.00074094 | 0.00059034 | 1 |
| S2_vs_C-S2-COV | t1_loo | -0.00255088 | 0.00037965 | -0.00011061 | 0.00100471 | 0 |
| S2_vs_C-S2-COV | t2_etf | 0.00008351 | -0.00000495 | -0.00024518 | 0.00024840 | 2 |
| S2_vs_C-S2-COV | t2_loo | -0.01754571 | 0.00120805 | 0.00065705 | 0.00198295 | 0 |
| S2_vs_C-S2-L20 | t1_etf | -0.00133712 | 0.00018404 | -0.00034763 | 0.00086530 | 2 |
| S2_vs_C-S2-L20 | t1_loo | -0.00235323 | 0.00035030 | -0.00024447 | 0.00112393 | 1 |
| S2_vs_C-S2-L20 | t2_etf | -0.00526606 | 0.00031026 | 0.00011998 | 0.00065573 | 0 |
| S2_vs_C-S2-L20 | t2_loo | -0.00390911 | 0.00027280 | 0.00004507 | 0.00065003 | 0 |
| S2_vs_C-S2-WS | t1_etf | 0.00012442 | -0.00001715 | -0.00010168 | 0.00007933 | 1 |
| S2_vs_C-S2-WS | t1_loo | -0.00006611 | 0.00000986 | -0.00003989 | 0.00005570 | 1 |
| S2_vs_C-S2-WS | t2_etf | 0.00064311 | -0.00003811 | -0.00009808 | 0.00000369 | 2 |
| S2_vs_C-S2-WS | t2_loo | 0.00004658 | -0.00000326 | -0.00001217 | 0.00000289 | 1 |
| S2_vs_C-S2-PERM | t1_etf | -0.00239890 | 0.00032982 | -0.00004613 | 0.00090703 | 1 |
| S2_vs_C-S2-PERM | t1_loo | -0.00287896 | 0.00042834 | -0.00002801 | 0.00110135 | 0 |
| S2_vs_C-S2-PERM | t2_etf | -0.00205288 | 0.00012134 | -0.00005444 | 0.00038766 | 0 |
| S2_vs_C-S2-PERM | t2_loo | -0.00442131 | 0.00030839 | 0.00004078 | 0.00074900 | 0 |
| C-S2-LONG_vs_S2 | t1_etf | -0.00039971 | 0.00005509 | -0.00031701 | 0.00058328 | 1 |
| C-S2-LONG_vs_S2 | t1_loo | 0.00375290 | -0.00055997 | -0.00098187 | -0.00021262 | 3 |
| C-S2-LONG_vs_S2 | t2_etf | -0.00194720 | 0.00011533 | -0.00003618 | 0.00032142 | 0 |
| C-S2-LONG_vs_S2 | t2_loo | 0.01595759 | -0.00111798 | -0.00176290 | -0.00068462 | 3 |
| S3_vs_C-S3-COV | t1_etf | -0.00184194 | 0.00025444 | -0.00038013 | 0.00090365 | 1 |
| S3_vs_C-S3-COV | t1_loo | -0.00264869 | 0.00039433 | -0.00021719 | 0.00105988 | 0 |
| S3_vs_C-S3-COV | t2_etf | 0.00246096 | -0.00014440 | -0.00048184 | 0.00021280 | 2 |
| S3_vs_C-S3-COV | t2_loo | -0.00912494 | 0.00063174 | -0.00002173 | 0.00148037 | 1 |
| S3_vs_C-S3-L20 | t1_etf | -0.00219479 | 0.00030308 | -0.00026111 | 0.00103562 | 2 |
| S3_vs_C-S3-L20 | t1_loo | -0.00271269 | 0.00040383 | -0.00022470 | 0.00120441 | 2 |
| S3_vs_C-S3-L20 | t2_etf | 0.00000547 | -0.00000032 | -0.00050127 | 0.00058793 | 1 |
| S3_vs_C-S3-L20 | t2_loo | -0.00030999 | 0.00002165 | -0.00019283 | 0.00039801 | 0 |
| S3_vs_C-S3-WS | t1_etf | 0.00037589 | -0.00005204 | -0.00078313 | 0.00038286 | 2 |
| S3_vs_C-S3-WS | t1_loo | -0.00003968 | 0.00000592 | -0.00004344 | 0.00003397 | 1 |
| S3_vs_C-S3-WS | t2_etf | -0.00017548 | 0.00001027 | -0.00006148 | 0.00005683 | 1 |
| S3_vs_C-S3-WS | t2_loo | -0.00060845 | 0.00004248 | -0.00000281 | 0.00007270 | 0 |
| S3_vs_C-S3-PERM | t1_etf | -0.00301845 | 0.00041648 | -0.00000015 | 0.00103000 | 0 |
| S3_vs_C-S3-PERM | t1_loo | -0.00314044 | 0.00046731 | 0.00004729 | 0.00110128 | 0 |
| S3_vs_C-S3-PERM | t2_etf | -0.00252301 | 0.00014731 | -0.00004466 | 0.00042219 | 0 |
| S3_vs_C-S3-PERM | t2_loo | -0.00413769 | 0.00028788 | 0.00002202 | 0.00068118 | 0 |
| C-S3-LONG_vs_S3 | t1_etf | 0.00171145 | -0.00023685 | -0.00047971 | 0.00003236 | 3 |
| C-S3-LONG_vs_S3 | t1_loo | 0.00183657 | -0.00027415 | -0.00049961 | -0.00000187 | 3 |
| C-S3-LONG_vs_S3 | t2_etf | -0.00348215 | 0.00020382 | 0.00007909 | 0.00039112 | 0 |
| C-S3-LONG_vs_S3 | t2_loo | 0.00902650 | -0.00063063 | -0.00115143 | -0.00024839 | 2 |
| S4_vs_S1 | t2_etf | 0.06937444 | -0.00407026 | -0.00699023 | -0.00187431 | 3 |
| S4_vs_S3 | t2_etf | 0.06717094 | -0.00393167 | -0.00641012 | -0.00195171 | 3 |

## Conservative useful-semantic gates

| architecture | target | matched_base_passed | controls_passed | all_required_passed |
|---|---|---|---|---|
| S2 | t1_etf | False | False | False |
| S2 | t1_loo | False | False | False |
| S2 | t2_etf | True | False | False |
| S2 | t2_loo | False | False | False |
| S3 | t1_etf | False | False | False |
| S3 | t1_loo | False | False | False |
| S3 | t2_etf | False | False | False |
| S3 | t2_loo | False | False | False |

Each S2 or S3 target must beat its matched base and all four coverage, stale-event, wrong-stock, and date-sector permutation controls, with a negative point loss delta, an upper 95% bound below zero, and improvement in at least two of three folds. Long-description is report-only.

## W17-Lite Elastic Net coefficient stability

| bundle | target | wlite_union_selected_count | wlite_selected_all_folds_count | wlite_stable_sign_all_folds_count | mean_pairwise_wlite_selected_set_jaccard |
|---|---|---|---|---|---|
| S2 | t1_etf | 14 | 3 | 3 | 0.415085 |
| S2 | t1_loo | 17 | 0 | 0 | 0.196078 |
| S2 | t2_etf | 16 | 1 | 1 | 0.194444 |
| S2 | t2_loo | 17 | 1 | 1 | 0.372549 |
| S3 | t1_etf | 13 | 1 | 1 | 0.273427 |
| S3 | t1_loo | 15 | 0 | 0 | 0.288889 |
| S3 | t2_etf | 17 | 0 | 0 | 0.058824 |
| S3 | t2_loo | 17 | 1 | 1 | 0.192810 |

## Validation-gated nonlinear subset

S4 was evaluated only for validation-gated target(s): t2_etf.

## Inference and claim boundary

- Confidence intervals use 2,000 paired, fold-contained moving-block resamples of 10 forecast sessions, retaining every stock on each sampled date.
- Loss is squared error in Fisher-z space.
- No multiple-testing adjustment is applied.
- These dates were previously inspected development periods.
- The retrospective ordinary Massive archive is not article-version safe.
- Semantic-quality gate passed: `false`.
- Protocol SHA-256: `2455e3a1bbef5ac9b358e7f0aaa92ed209f51cee2a0936de4093e23799c53def`.
