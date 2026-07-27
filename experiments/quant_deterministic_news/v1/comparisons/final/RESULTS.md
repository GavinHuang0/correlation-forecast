# Quant plus deterministic-news v1 results

Status: **complete exploratory development experiment**.

Neither of the two placebo-tested linear specifications (A5/B4) passes the conservative final-stage matched-base, calibration/placebo, and fold-consistency gate. Nonlinear A6/B5 and the winner-base sensitivity remain descriptive because matching nonlinear placebo fits were not preregistered.

Negative paired loss deltas favor the candidate. Confidence intervals
use 2,000 moving-block resamples of whole dates, 10 sessions per block, constructed separately inside each outer fold.
Pooled estimates are stock-day-row weighted; because every eligible
target-date has 30 stocks, this is equivalent to equal date weighting
with fold weights proportional to each fold's eligible date count.

## Primary and placebo comparisons

| comparison | target | incremental_r2 | mean_squared_loss_delta | bootstrap_ci_lower | bootstrap_ci_upper | folds_candidate_better | fold_count |
|---:|---:|---:|---:|---:|---:|---:|---:|
| A5_vs_A0-L | t1_etf | 0.001911 | -0.000263 | -0.001117 | 0.000342 | 2 | 3 |
| A5_vs_A0-L | t1_loo | 0.003686 | -0.000548 | -0.001466 | 0.000058 | 2 | 3 |
| A5_vs_A0-L | t2_etf | 0.036499 | -0.002188 | -0.003467 | -0.001474 | 3 | 3 |
| A5_vs_A0-L | t2_loo | 0.006171 | -0.000428 | -0.001254 | 0.000093 | 2 | 3 |
| A6_vs_A0-T | t1_etf | 0.000524 | -0.000072 | -0.001309 | 0.001344 | 2 | 3 |
| A6_vs_A0-T | t1_loo | -0.005097 | 0.000742 | -0.000404 | 0.002346 | 1 | 3 |
| A6_vs_A0-T | t2_etf | 0.005680 | -0.000313 | -0.000963 | 0.000296 | 1 | 3 |
| A6_vs_A0-T | t2_loo | 0.001054 | -0.000069 | -0.000594 | 0.000569 | 1 | 3 |
| B4_vs_B0 | t1_etf | 0.004727 | -0.000632 | -0.002382 | 0.001163 | 2 | 2 |
| B4_vs_B0 | t1_loo | -0.003064 | 0.000431 | -0.001939 | 0.002272 | 1 | 2 |
| B4_vs_B0 | t2_etf | -0.026285 | 0.001224 | -0.000876 | 0.003113 | 0 | 2 |
| B4_vs_B0 | t2_loo | -0.059423 | 0.003251 | 0.001067 | 0.005242 | 0 | 2 |
| B5_vs_B0 | t1_etf | 0.001379 | -0.000184 | -0.001662 | 0.000868 | 2 | 2 |
| B5_vs_B0 | t1_loo | -0.014732 | 0.002073 | -0.000444 | 0.004027 | 0 | 2 |
| B5_vs_B0 | t2_etf | -0.035140 | 0.001637 | 0.000500 | 0.002773 | 1 | 2 |
| B5_vs_B0 | t2_loo | -0.129087 | 0.007062 | 0.004331 | 0.009252 | 0 | 2 |
| S-B5_vs_S-B0 | t1_etf | 0.009117 | -0.001214 | -0.002512 | -0.000172 | 2 | 2 |
| A5_vs_C-A5-L20 | t1_etf | -0.006118 | 0.000835 | -0.000047 | 0.001810 | 0 | 3 |
| A5_vs_C-A5-L20 | t1_loo | -0.008211 | 0.001208 | 0.000195 | 0.002414 | 1 | 3 |
| A5_vs_C-A5-L20 | t2_etf | -0.011576 | 0.000661 | -0.000476 | 0.001503 | 1 | 3 |
| A5_vs_C-A5-L20 | t2_loo | -0.014863 | 0.001010 | -0.000180 | 0.002281 | 1 | 3 |
| A5_vs_C-A5-WS | t1_etf | 0.002852 | -0.000393 | -0.000753 | -0.000017 | 3 | 3 |
| A5_vs_C-A5-WS | t1_loo | 0.002124 | -0.000316 | -0.000703 | -0.000046 | 1 | 3 |
| A5_vs_C-A5-WS | t2_etf | 0.020628 | -0.001217 | -0.001756 | -0.000783 | 3 | 3 |
| A5_vs_C-A5-WS | t2_loo | 0.004697 | -0.000325 | -0.000511 | -0.000183 | 2 | 3 |
| B4_vs_C-B4-L20 | t1_etf | -0.003419 | 0.000454 | -0.000789 | 0.001803 | 1 | 2 |
| B4_vs_C-B4-L20 | t1_loo | 0.009780 | -0.001394 | -0.003854 | 0.000288 | 1 | 2 |
| B4_vs_C-B4-L20 | t2_etf | 0.009644 | -0.000466 | -0.002427 | 0.001497 | 1 | 2 |
| B4_vs_C-B4-L20 | t2_loo | 0.050918 | -0.003109 | -0.005220 | -0.000761 | 2 | 2 |
| B4_vs_C-B4-WS | t1_etf | 0.001211 | -0.000161 | -0.000338 | 0.000056 | 2 | 2 |
| B4_vs_C-B4-WS | t1_loo | 0.000182 | -0.000026 | -0.000111 | 0.000056 | 1 | 2 |
| B4_vs_C-B4-WS | t2_etf | -0.006517 | 0.000310 | -0.000939 | 0.001492 | 1 | 2 |
| B4_vs_C-B4-WS | t2_loo | 0.001240 | -0.000072 | -0.000252 | 0.000113 | 1 | 2 |

## Useful-news decision gate

This gate is a conservative final-stage operationalization of the
locked protocol's qualitative requirement to beat the matched quant
base, calibration/placebo controls, and more than one fold. The exact
boolean threshold was not separately preregistered.

| candidate | target | matched_quant_improvement_ci | beats_stale_placebo_ci | beats_wrong_stock_placebo_ci | improves_at_least_two_of_three_folds | all_required_gates_passed | beats_mean_error_calibration_ci | beats_linear_calibration_ci | improves_both_outer_folds |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A5 | t1_etf | False | False | True | True | False | nan | nan | nan |
| A5 | t1_loo | False | False | True | True | False | nan | nan | nan |
| A5 | t2_etf | True | False | True | True | False | nan | nan | nan |
| A5 | t2_loo | False | False | True | True | False | nan | nan | nan |
| B4 | t1_etf | False | False | False | nan | False | False | False | True |
| B4 | t1_loo | False | False | False | nan | False | False | False | False |
| B4 | t2_etf | False | False | False | nan | False | False | False | False |
| B4 | t2_loo | False | True | False | nan | False | True | True | False |

## Model metrics

| bundle | target | fisher_z_rmse | fisher_z_mae | oos_r2_vs_persistence |
|---:|---:|---:|---:|---:|
| A0-L | t1_etf | 0.370931 | 0.292976 | 0.407915 |
| A0-L | t1_loo | 0.385767 | 0.303961 | 0.382590 |
| A0-L | t2_etf | 0.244842 | 0.192202 | 0.226441 |
| A0-L | t2_loo | 0.263394 | 0.204305 | 0.211898 |
| A0-T | t1_etf | 0.369951 | 0.292284 | 0.411039 |
| A0-T | t1_loo | 0.381434 | 0.300462 | 0.396380 |
| A0-T | t2_etf | 0.234820 | 0.183441 | 0.288475 |
| A0-T | t2_loo | 0.256653 | 0.199505 | 0.251721 |
| A1 | t1_etf | 0.445322 | 0.355244 | 0.146610 |
| A1 | t1_loo | 0.473489 | 0.377337 | 0.069870 |
| A1 | t2_etf | 0.327122 | 0.259395 | -0.380831 |
| A1 | t2_loo | 0.364996 | 0.294502 | -0.513370 |
| A2 | t1_etf | 0.371053 | 0.292877 | 0.407523 |
| A2 | t1_loo | 0.385427 | 0.303746 | 0.383678 |
| A2 | t2_etf | 0.243101 | 0.190616 | 0.237404 |
| A2 | t2_loo | 0.262404 | 0.203822 | 0.217814 |
| A3 | t1_etf | 0.371359 | 0.293006 | 0.406546 |
| A3 | t1_loo | 0.385280 | 0.303709 | 0.384147 |
| A3 | t2_etf | 0.242409 | 0.189901 | 0.241736 |
| A3 | t2_loo | 0.261535 | 0.203292 | 0.222987 |
| A4 | t1_etf | 0.371263 | 0.293058 | 0.406853 |
| A4 | t1_loo | 0.385396 | 0.304037 | 0.383778 |
| A4 | t2_etf | 0.241821 | 0.189399 | 0.245412 |
| A4 | t2_loo | 0.262691 | 0.204083 | 0.216102 |
| A5 | t1_etf | 0.370576 | 0.292525 | 0.409046 |
| A5 | t1_loo | 0.385055 | 0.303625 | 0.384866 |
| A5 | t2_etf | 0.240332 | 0.188125 | 0.254675 |
| A5 | t2_loo | 0.262580 | 0.203918 | 0.216762 |
| A6 | t1_etf | 0.369854 | 0.291813 | 0.411348 |
| A6 | t1_loo | 0.382405 | 0.301513 | 0.393304 |
| A6 | t2_etf | 0.234152 | 0.182886 | 0.292516 |
| A6 | t2_loo | 0.256518 | 0.199145 | 0.252510 |
| A7 | t1_etf | 0.367931 | 0.290320 | 0.417453 |
| A7 | t2_etf | 0.236386 | 0.184856 | 0.278950 |
| B0 | t1_etf | 0.365758 | 0.289788 | 0.422500 |
| B0 | t1_loo | 0.375135 | 0.296450 | 0.411319 |
| B0 | t2_etf | 0.215835 | 0.172199 | 0.282684 |
| B0 | t2_loo | 0.233898 | 0.183493 | 0.280724 |
| B1 | t1_etf | 0.364260 | 0.288730 | 0.427221 |
| B1 | t1_loo | 0.377560 | 0.298655 | 0.403684 |
| B1 | t2_etf | 0.219716 | 0.175319 | 0.256661 |
| B1 | t2_loo | 0.248800 | 0.194778 | 0.186150 |
| B2 | t1_etf | 0.365332 | 0.289675 | 0.423847 |
| B2 | t1_loo | 0.376592 | 0.298136 | 0.406737 |
| B2 | t2_etf | 0.221116 | 0.176522 | 0.247156 |
| B2 | t2_loo | 0.248573 | 0.194334 | 0.187637 |
| B3 | t1_etf | 0.364813 | 0.289551 | 0.425481 |
| B3 | t1_loo | 0.376383 | 0.297965 | 0.407395 |
| B3 | t2_etf | 0.217511 | 0.174038 | 0.271504 |
| B3 | t2_loo | 0.241967 | 0.190719 | 0.230243 |
| B4 | t1_etf | 0.364893 | 0.289613 | 0.425230 |
| B4 | t1_loo | 0.375710 | 0.297694 | 0.409515 |
| B4 | t2_etf | 0.218654 | 0.175077 | 0.263830 |
| B4 | t2_loo | 0.240747 | 0.189144 | 0.237982 |
| B5 | t1_etf | 0.365506 | 0.290074 | 0.423297 |
| B5 | t1_loo | 0.377889 | 0.299241 | 0.402646 |
| B5 | t2_etf | 0.219595 | 0.175230 | 0.257478 |
| B5 | t2_loo | 0.248537 | 0.194574 | 0.187875 |
| C-A5-L20 | t1_etf | 0.369448 | 0.291468 | 0.412640 |
| C-A5-L20 | t1_loo | 0.383484 | 0.302162 | 0.389876 |
| C-A5-L20 | t2_etf | 0.238953 | 0.187418 | 0.263205 |
| C-A5-L20 | t2_loo | 0.260650 | 0.202576 | 0.228233 |
| C-A5-WS | t1_etf | 0.371106 | 0.293050 | 0.407356 |
| C-A5-WS | t1_loo | 0.385465 | 0.304151 | 0.383556 |
| C-A5-WS | t2_etf | 0.242850 | 0.190439 | 0.238977 |
| C-A5-WS | t2_loo | 0.263199 | 0.204574 | 0.213065 |
| C-B4-L20 | t1_etf | 0.364271 | 0.288733 | 0.427189 |
| C-B4-L20 | t1_loo | 0.377560 | 0.298655 | 0.403684 |
| C-B4-L20 | t2_etf | 0.219716 | 0.175319 | 0.256661 |
| C-B4-L20 | t2_loo | 0.247121 | 0.193197 | 0.197100 |
| C-B4-WS | t1_etf | 0.365114 | 0.289717 | 0.424533 |
| C-B4-WS | t1_loo | 0.375744 | 0.297711 | 0.409408 |
| C-B4-WS | t2_etf | 0.217945 | 0.174082 | 0.268596 |
| C-B4-WS | t2_loo | 0.240897 | 0.189250 | 0.237036 |
| S-B0 | t1_etf | 0.364862 | 0.288790 | 0.425329 |
| S-B1 | t1_etf | 0.363373 | 0.287847 | 0.430008 |
| S-B2 | t1_etf | 0.363732 | 0.288131 | 0.428881 |
| S-B3 | t1_etf | 0.363714 | 0.288366 | 0.428938 |
| S-B4 | t1_etf | 0.363791 | 0.288439 | 0.428695 |
| S-B5 | t1_etf | 0.363195 | 0.287786 | 0.430568 |

## News/no-news slices

| comparison | target | news_slice | rows | folds_present | fold_row_counts | incremental_r2 | mean_squared_loss_delta |
|---:|---:|---:|---:|---:|---:|---:|---:|
| A5_vs_A0-L | t1_etf | observed_no_relevant_news | 66 | fold_1,fold_2 | {"fold_1": 54, "fold_2": 12} | 0.003828 | -0.000420 |
| A5_vs_A0-L | t1_etf | observed_relevant_news | 11124 | fold_1,fold_2,fold_3 | {"fold_1": 3606, "fold_2": 3828, "fold_3": 3690} | 0.001902 | -0.000262 |
| A5_vs_A0-L | t1_loo | observed_no_relevant_news | 66 | fold_1,fold_2 | {"fold_1": 54, "fold_2": 12} | -0.016371 | 0.001941 |
| A5_vs_A0-L | t1_loo | observed_relevant_news | 11124 | fold_1,fold_2,fold_3 | {"fold_1": 3606, "fold_2": 3828, "fold_3": 3690} | 0.003780 | -0.000563 |
| A5_vs_A0-L | t2_etf | observed_no_relevant_news | 66 | fold_1,fold_2 | {"fold_1": 54, "fold_2": 12} | 0.006060 | -0.000241 |
| A5_vs_A0-L | t2_etf | observed_relevant_news | 10764 | fold_1,fold_2,fold_3 | {"fold_1": 3486, "fold_2": 3708, "fold_3": 3570} | 0.036623 | -0.002200 |
| A5_vs_A0-L | t2_loo | observed_no_relevant_news | 66 | fold_1,fold_2 | {"fold_1": 54, "fold_2": 12} | -0.032225 | 0.001699 |
| A5_vs_A0-L | t2_loo | observed_relevant_news | 10764 | fold_1,fold_2,fold_3 | {"fold_1": 3486, "fold_2": 3708, "fold_3": 3570} | 0.006350 | -0.000441 |
| A6_vs_A0-T | t1_etf | observed_no_relevant_news | 66 | fold_1,fold_2 | {"fold_1": 54, "fold_2": 12} | 0.039365 | -0.004144 |
| A6_vs_A0-T | t1_etf | observed_relevant_news | 11124 | fold_1,fold_2,fold_3 | {"fold_1": 3606, "fold_2": 3828, "fold_3": 3690} | 0.000347 | -0.000048 |
| A6_vs_A0-T | t1_loo | observed_no_relevant_news | 66 | fold_1,fold_2 | {"fold_1": 54, "fold_2": 12} | -0.000756 | 0.000101 |
| A6_vs_A0-T | t1_loo | observed_relevant_news | 11124 | fold_1,fold_2,fold_3 | {"fold_1": 3606, "fold_2": 3828, "fold_3": 3690} | -0.005121 | 0.000745 |
| A6_vs_A0-T | t2_etf | observed_no_relevant_news | 66 | fold_1,fold_2 | {"fold_1": 54, "fold_2": 12} | -0.026325 | 0.000850 |
| A6_vs_A0-T | t2_etf | observed_relevant_news | 10764 | fold_1,fold_2,fold_3 | {"fold_1": 3486, "fold_2": 3708, "fold_3": 3570} | 0.005794 | -0.000320 |
| A6_vs_A0-T | t2_loo | observed_no_relevant_news | 66 | fold_1,fold_2 | {"fold_1": 54, "fold_2": 12} | 0.028370 | -0.001567 |
| A6_vs_A0-T | t2_loo | observed_relevant_news | 10764 | fold_1,fold_2,fold_3 | {"fold_1": 3486, "fold_2": 3708, "fold_3": 3570} | 0.000914 | -0.000060 |
| B4_vs_B0 | t1_etf | observed_no_relevant_news | 12 | fold_2 | {"fold_2": 12} | 0.183387 | -0.023054 |
| B4_vs_B0 | t1_etf | observed_relevant_news | 7518 | fold_2,fold_3 | {"fold_2": 3828, "fold_3": 3690} | 0.004459 | -0.000597 |
| B4_vs_B0 | t1_loo | observed_no_relevant_news | 12 | fold_2 | {"fold_2": 12} | -0.182034 | 0.009812 |
| B4_vs_B0 | t1_loo | observed_relevant_news | 7518 | fold_2,fold_3 | {"fold_2": 3828, "fold_3": 3690} | -0.002954 | 0.000416 |
| B4_vs_B0 | t2_etf | observed_no_relevant_news | 12 | fold_2 | {"fold_2": 12} | -0.149990 | 0.002003 |
| B4_vs_B0 | t2_etf | observed_relevant_news | 7278 | fold_2,fold_3 | {"fold_2": 3708, "fold_3": 3570} | -0.026227 | 0.001223 |
| B4_vs_B0 | t2_loo | observed_no_relevant_news | 12 | fold_2 | {"fold_2": 12} | -0.108232 | 0.002050 |
| B4_vs_B0 | t2_loo | observed_relevant_news | 7278 | fold_2,fold_3 | {"fold_2": 3708, "fold_3": 3570} | -0.059395 | 0.003253 |
| B5_vs_B0 | t1_etf | observed_no_relevant_news | 12 | fold_2 | {"fold_2": 12} | 0.158222 | -0.019890 |
| B5_vs_B0 | t1_etf | observed_relevant_news | 7518 | fold_2,fold_3 | {"fold_2": 3828, "fold_3": 3690} | 0.001144 | -0.000153 |
| B5_vs_B0 | t1_loo | observed_no_relevant_news | 12 | fold_2 | {"fold_2": 12} | -0.089973 | 0.004850 |
| B5_vs_B0 | t1_loo | observed_relevant_news | 7518 | fold_2,fold_3 | {"fold_2": 3828, "fold_3": 3690} | -0.014686 | 0.002069 |
| B5_vs_B0 | t2_etf | observed_no_relevant_news | 12 | fold_2 | {"fold_2": 12} | -0.173142 | 0.002312 |
| B5_vs_B0 | t2_etf | observed_relevant_news | 7278 | fold_2,fold_3 | {"fold_2": 3708, "fold_3": 3570} | -0.035075 | 0.001636 |
| B5_vs_B0 | t2_loo | observed_no_relevant_news | 12 | fold_2 | {"fold_2": 12} | -0.540051 | 0.010227 |
| B5_vs_B0 | t2_loo | observed_relevant_news | 7278 | fold_2,fold_3 | {"fold_2": 3708, "fold_3": 3570} | -0.128853 | 0.007057 |
| S-B5_vs_S-B0 | t1_etf | observed_no_relevant_news | 12 | fold_2 | {"fold_2": 12} | 0.143016 | -0.019214 |
| S-B5_vs_S-B0 | t1_etf | observed_relevant_news | 7518 | fold_2,fold_3 | {"fold_2": 3828, "fold_3": 3690} | 0.008902 | -0.001185 |

The observed-no-news slice is extremely small, has no fold-3 rows,
and is descriptive only; it is not a stable across-fold sensitivity.

## Interpretation

- Track A and Track B are not a controlled architecture comparison;
  Track B can train only on earlier saved out-of-sample residuals.
- The stale-news and wrong-stock comparisons are required falsification
  checks, not alternative production models.
- No multiple-testing adjustment is applied across the ladder.
- All results are development estimates because quant-v1 outer blocks
  were previously inspected and the retrospective news archive lacks
  historical article versions and first-seen timestamps.
- Protocol SHA-256: `3ed24d17c79b28bf88c614f2c4e8b3143ac7f6d888e6b61f50f37e5980ed772c`.
- Bootstrap seed: `1729`.
