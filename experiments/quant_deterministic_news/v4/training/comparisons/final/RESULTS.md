# V4 final paired comparisons

These are exploratory retrospective results. Incremental MSE R2 is
`1 - candidate MSE / base MSE`; positive values favor the candidate.
Confidence intervals use paired moving blocks of whole forecast dates.

| Comparison | Target | Incremental MSE R2 | 95% interval | Folds improved | Gate |
|---|---|---:|---:|---:|---|
| J1_vs_v3_S0 | t1_etf | -0.002183 | [-0.005916, 0.004324] | 2/3 | fail |
| J1_vs_v3_S0 | t1_loo | 0.001740 | [-0.000303, 0.005160] | 1/3 | fail |
| J1_vs_v3_S0 | t2_etf | 0.007110 | [-0.006622, 0.023290] | 2/3 | fail |
| J1_vs_v3_S0 | t2_loo | -0.013561 | [-0.029169, -0.002450] | 1/3 | fail |
| J2_vs_v3_S1 | t1_etf | -0.000099 | [-0.003771, 0.005409] | 2/3 | fail |
| J2_vs_v3_S1 | t1_loo | 0.001943 | [-0.001073, 0.006799] | 2/3 | fail |
| J2_vs_v3_S1 | t2_etf | 0.003003 | [-0.007188, 0.015402] | 1/3 | fail |
| J2_vs_v3_S1 | t2_loo | -0.012908 | [-0.025566, -0.004229] | 0/3 | fail |
| J3_vs_v3_S0 | t1_etf | -0.001491 | [-0.003394, 0.001301] | 1/3 | fail |
| J3_vs_v3_S0 | t1_loo | 0.002266 | [0.000237, 0.005616] | 1/3 | fail |
| J3_vs_v3_S0 | t2_etf | 0.006195 | [0.000013, 0.013428] | 2/3 | pass |
| J3_vs_v3_S0 | t2_loo | 0.001904 | [-0.003730, 0.007423] | 2/3 | fail |
| J1_vs_stale20 | t1_etf | 0.001061 | [-0.002032, 0.005260] | 2/3 | fail |
| J1_vs_stale20 | t1_loo | -0.001398 | [-0.002753, 0.000576] | 0/3 | fail |
| J1_vs_stale20 | t2_etf | -0.013219 | [-0.026785, -0.002437] | 1/3 | fail |
| J1_vs_stale20 | t2_loo | -0.003617 | [-0.015405, 0.005910] | 1/3 | fail |
| J2_vs_stale20 | t1_etf | 0.004041 | [0.000643, 0.007671] | 3/3 | pass |
| J2_vs_stale20 | t1_loo | 0.003382 | [0.000066, 0.008714] | 1/3 | fail |
| J2_vs_stale20 | t2_etf | -0.001449 | [-0.012961, 0.008577] | 1/3 | fail |
| J2_vs_stale20 | t2_loo | -0.000825 | [-0.011457, 0.009062] | 1/3 | fail |
| J1_vs_wrong_stock | t1_etf | -0.000776 | [-0.002061, 0.000410] | 1/3 | fail |
| J1_vs_wrong_stock | t1_loo | -0.000211 | [-0.000685, 0.000262] | 0/3 | fail |
| J1_vs_wrong_stock | t2_etf | 0.001996 | [-0.002503, 0.007091] | 2/3 | fail |
| J1_vs_wrong_stock | t2_loo | 0.002893 | [0.000839, 0.005854] | 2/3 | pass |
| J2_vs_wrong_stock | t1_etf | -0.003015 | [-0.005908, 0.000798] | 1/3 | fail |
| J2_vs_wrong_stock | t1_loo | 0.004387 | [0.002836, 0.006802] | 2/3 | pass |
| J2_vs_wrong_stock | t2_etf | 0.012613 | [0.007014, 0.020573] | 2/3 | pass |
| J2_vs_wrong_stock | t2_loo | 0.002420 | [0.000526, 0.005292] | 1/3 | fail |
| J1_vs_permutation | t1_etf | 0.000912 | [-0.000503, 0.002212] | 2/3 | fail |
| J1_vs_permutation | t1_loo | 0.002756 | [0.000549, 0.005616] | 1/3 | fail |
| J1_vs_permutation | t2_etf | 0.000510 | [-0.006261, 0.005720] | 2/3 | fail |
| J1_vs_permutation | t2_loo | -0.005630 | [-0.013153, 0.002654] | 1/3 | fail |
| J2_vs_permutation | t1_etf | -0.003859 | [-0.007196, -0.000242] | 2/3 | fail |
| J2_vs_permutation | t1_loo | 0.003371 | [0.000659, 0.007378] | 1/3 | fail |
| J2_vs_permutation | t2_etf | 0.003150 | [-0.001329, 0.007611] | 2/3 | fail |
| J2_vs_permutation | t2_loo | -0.002924 | [-0.009406, 0.003785] | 0/3 | fail |
| J1_vs_quality | t1_etf | -0.001985 | [-0.005951, 0.004362] | 2/3 | fail |
| J1_vs_quality | t1_loo | -0.003285 | [-0.005440, -0.000901] | 0/3 | fail |
| J1_vs_quality | t2_etf | -0.002410 | [-0.020861, 0.014685] | 1/3 | fail |
| J1_vs_quality | t2_loo | -0.029290 | [-0.047773, -0.014642] | 0/3 | fail |
| J2_vs_quality | t1_etf | -0.001355 | [-0.004924, 0.003816] | 2/3 | fail |
| J2_vs_quality | t1_loo | -0.003559 | [-0.006098, -0.000666] | 0/3 | fail |
| J2_vs_quality | t2_etf | 0.000749 | [-0.011484, 0.014722] | 1/3 | fail |
| J2_vs_quality | t2_loo | -0.021040 | [-0.034100, -0.011161] | 0/3 | fail |
| RCAL_vs_R0 | t1_etf | -0.001316 | [-0.002996, 0.000781] | 3/5 | fail |
| RCAL_vs_R0 | t1_loo | -0.005729 | [-0.009391, -0.002971] | 2/5 | fail |
| RCAL_vs_R0 | t2_etf | 0.000981 | [-0.018894, 0.027770] | 2/5 | fail |
| RCAL_vs_R0 | t2_loo | -0.032607 | [-0.051826, -0.012204] | 2/5 | fail |
| RRES_C6_vs_R0 | t1_etf | 0.001146 | [-0.000690, 0.003383] | 2/5 | fail |
| RRES_C6_vs_R0 | t1_loo | -0.002545 | [-0.005134, 0.000580] | 0/5 | fail |
| RRES_C6_vs_R0 | t2_etf | 0.013844 | [0.003429, 0.026228] | 3/5 | pass |
| RRES_C6_vs_R0 | t2_loo | -0.009851 | [-0.023553, 0.001826] | 1/5 | fail |
| RRES_C6_vs_RCAL | t1_etf | 0.002459 | [0.000525, 0.004322] | 3/5 | pass |
| RRES_C6_vs_RCAL | t1_loo | 0.003165 | [-0.000706, 0.008890] | 3/5 | fail |
| RRES_C6_vs_RCAL | t2_etf | 0.012876 | [-0.008298, 0.028640] | 4/5 | fail |
| RRES_C6_vs_RCAL | t2_loo | 0.022037 | [0.000304, 0.038442] | 3/5 | pass |
| RRES_L19_vs_R0 | t1_etf | 0.000935 | [-0.000985, 0.003114] | 2/5 | fail |
| RRES_L19_vs_R0 | t1_loo | 0.000461 | [-0.001135, 0.002464] | 1/5 | fail |
| RRES_L19_vs_R0 | t2_etf | 0.009342 | [-0.001389, 0.021425] | 3/5 | fail |
| RRES_L19_vs_R0 | t2_loo | -0.001121 | [-0.009842, 0.005966] | 2/5 | fail |
| RRES_L19_vs_RCAL | t1_etf | 0.002248 | [-0.000076, 0.004181] | 3/5 | fail |
| RRES_L19_vs_RCAL | t1_loo | 0.006154 | [0.002940, 0.010869] | 4/5 | pass |
| RRES_L19_vs_RCAL | t2_etf | 0.008369 | [-0.014093, 0.024719] | 3/5 | fail |
| RRES_L19_vs_RCAL | t2_loo | 0.030491 | [0.010260, 0.047648] | 3/5 | pass |
| RRES_L19_vs_C6 | t1_etf | -0.000212 | [-0.001760, 0.001050] | 2/5 | fail |
| RRES_L19_vs_C6 | t1_loo | 0.002998 | [0.000533, 0.005091] | 4/5 | pass |
| RRES_L19_vs_C6 | t2_etf | -0.004565 | [-0.012380, 0.002591] | 2/5 | fail |
| RRES_L19_vs_C6 | t2_loo | 0.008645 | [-0.001143, 0.019321] | 3/5 | fail |
| RSTACK_vs_R0 | t1_etf | -0.005438 | [-0.009706, 0.000008] | 2/5 | fail |
| RSTACK_vs_R0 | t1_loo | -0.014675 | [-0.022594, -0.007662] | 1/5 | fail |
| RSTACK_vs_R0 | t2_etf | -0.003566 | [-0.033566, 0.028468] | 3/5 | fail |
| RSTACK_vs_R0 | t2_loo | -0.064956 | [-0.097771, -0.039936] | 1/5 | fail |
| RSTACK_vs_RCAL | t1_etf | -0.004117 | [-0.008145, 0.000611] | 2/5 | fail |
| RSTACK_vs_RCAL | t1_loo | -0.008895 | [-0.014785, -0.003366] | 1/5 | fail |
| RSTACK_vs_RCAL | t2_etf | -0.004551 | [-0.026276, 0.014178] | 2/5 | fail |
| RSTACK_vs_RCAL | t2_loo | -0.031327 | [-0.063331, -0.008906] | 1/5 | fail |
| RRES_L19_vs_stale20 | t1_etf | -0.001338 | [-0.002667, 0.000068] | 1/5 | fail |
| RRES_L19_vs_stale20 | t1_loo | 0.001186 | [-0.000087, 0.002692] | 3/5 | fail |
| RRES_L19_vs_stale20 | t2_etf | -0.001339 | [-0.008779, 0.006890] | 2/5 | fail |
| RRES_L19_vs_stale20 | t2_loo | 0.004376 | [-0.001477, 0.009404] | 2/5 | fail |
| RRES_L19_vs_wrong_stock | t1_etf | 0.000469 | [-0.000658, 0.001322] | 2/5 | fail |
| RRES_L19_vs_wrong_stock | t1_loo | -0.000187 | [-0.000820, 0.000558] | 1/5 | fail |
| RRES_L19_vs_wrong_stock | t2_etf | -0.001254 | [-0.003634, 0.001336] | 3/5 | fail |
| RRES_L19_vs_wrong_stock | t2_loo | -0.001321 | [-0.005695, 0.002296] | 2/5 | fail |
| RRES_L19_vs_permutation | t1_etf | -0.000246 | [-0.001097, 0.000492] | 1/5 | fail |
| RRES_L19_vs_permutation | t1_loo | 0.001483 | [0.000344, 0.002688] | 4/5 | pass |
| RRES_L19_vs_permutation | t2_etf | 0.003003 | [-0.001874, 0.007366] | 3/5 | fail |
| RRES_L19_vs_permutation | t2_loo | 0.000952 | [-0.003859, 0.005518] | 2/5 | fail |
| RRES_L19_vs_quality | t1_etf | -0.001665 | [-0.002953, -0.000280] | 0/5 | fail |
| RRES_L19_vs_quality | t1_loo | 0.000494 | [-0.001201, 0.002111] | 2/5 | fail |
| RRES_L19_vs_quality | t2_etf | -0.003098 | [-0.009606, 0.003727] | 1/5 | fail |
| RRES_L19_vs_quality | t2_loo | -0.000564 | [-0.006678, 0.004912] | 2/5 | fail |

## Useful-semantic gates

A live semantic candidate passes only when every required matched-base,
stale, wrong-stock, permutation, and quality comparison passes its point,
confidence-interval, and fold-count conditions.

| Candidate | Target | Comparisons passed | Useful gate |
|---|---|---:|---|
| J1 | t1_etf | 0/5 | fail |
| J1 | t1_loo | 0/5 | fail |
| J1 | t2_etf | 0/5 | fail |
| J1 | t2_loo | 1/5 | fail |
| J2 | t1_etf | 1/5 | fail |
| J2 | t1_loo | 1/5 | fail |
| J2 | t2_etf | 1/5 | fail |
| J2 | t2_loo | 0/5 | fail |
| RRES-L19 | t1_etf | 0/6 | fail |
| RRES-L19 | t1_loo | 2/6 | fail |
| RRES-L19 | t2_etf | 0/6 | fail |
| RRES-L19 | t2_loo | 1/6 | fail |
| RSTACK | t1_etf | 0/2 | fail |
| RSTACK | t1_loo | 0/2 | fail |
| RSTACK | t2_etf | 0/2 | fail |
| RSTACK | t2_loo | 0/2 | fail |
