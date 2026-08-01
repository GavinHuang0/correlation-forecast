# W17-Lite exploratory training execution

Status: **complete exploratory training run**. All four linear rungs, ten
matched controls/sensitivities, and the validation-gated nonlinear rung have
been fitted and hash-closed. The final comparison is complete.

This namespace is separate from every completed v1/v2 model and result. The
FLAN-T5-XL K-16 article inference completed all 50,488 selected articles with
zero failures, but only 32,880 canonical/reversed prompt pairs agreed:
65.124386%, below the frozen 85% semantic-stability requirement. The
downstream fit therefore proceeded only as a diagnostic experiment. It does
not make the features primary- or confirmatory-training eligible.

The protocol was locked after all feature and joined panels passed preflight:

```powershell
.venv-training\Scripts\python.exe `
  -m scripts.quant_deterministic_news_training_v3.lock_protocol
```

That command writes:

```text
config/quant_deterministic_news_protocol_v3.json
config/quant_deterministic_news_protocol_v3.sha256
```

It verifies and binds the canonical, long-description, and fixed
date-sector-permutation panels; the complete FLAN prediction ledger; the exact
17 W17-Lite columns; Q56 and D2-Normalized; all 27,510 stock-days; the three
chronological folds; T2 boundary purges; tuning grids; controls; and inference
seeds. Every training command refuses to run without the exact sidecar hash
and an explicit `--allow-failed-gate-exploratory` acknowledgement.

## Ladder

| Bundle | Inputs | Raw columns |
|---|---|---:|
| `S0` | Q56 | 56 |
| `S1` | Q56 + D2-Normalized | 86 |
| `S2` | Q56 + W17-Lite | 73 |
| `S3` | Q56 + D2-Normalized + W17-Lite | 103 |
| `S4` | Shallow XGBoost on S3 inputs | 103 |

S4 is eligible for a target only when S3 beats S1 validation MSE in at least
two of the three validation blocks. Its four ordered candidates, 2,000-tree
budget, 75-round early stopping, and seed 1729 are copied into the v3 lock
from the exact hash-bound quant-v1 protocol. The runner verifies that source
hash but consumes the copied v3 values as its only tuning authority.

If no target passes, running S4 still produces a hash-bound
`skipped_validation_gate` bundle containing the exact gate, S1/S3 validation
provenance, review, summary, locked model budget, manifest, and sidecar. A
missing S4 directory is never interpreted as an executed skip.

Both S2 and S3 receive separately fitted controls:

```text
COV   coverage/text-quality-only
L20   20-session-stale FLAN event shares, contemporaneous coverage
WS    fixed same-date within-sector wrong-stock semantic content
PERM  frozen within-date/sector accepted-event-label permutation
LONG  retained descriptions of at least 150 characters only
```

LONG is a sensitivity, not a control the canonical model must beat. The useful
semantic gate requires the live linear model to beat its matched Q or Q+D
base, COV, L20, WS, and PERM with an upper paired 95% block-bootstrap bound
below zero and improvement in at least two development folds.

PERM changes only the three accepted predictive event shares and their
derived entropy. Acceptance/disagreement and abstention/other mass, routing,
deterministic status cues, selection coverage, and text-quality fields stay
contemporaneous. Its exact builder seed is
`wlite-date-sector-event-permutation-v1`; the protocol also binds the
permutation-mapping SHA-256 from the daily-feature manifest. WS remains the
broader control that rotates all eleven event/routing/rule-content fields to
the fixed next stock within sector.

To execute the complete dependency-ordered ladder after the panels exist:

```powershell
C:\Windows\System32\WindowsPowerShell\v1.0\powershell.exe `
  -NoProfile -ExecutionPolicy Bypass `
  -File .\scripts\quant_deterministic_news_training_v3\run_wlite_ladder.ps1
```

Each bundle is written and hash-closed before the next bundle begins:

```text
outputs/quant_deterministic_news/v3/
experiments/quant_deterministic_news/v3/training/
```

The protocol also binds the v3 contract, lock, linear, XGBoost, and final
comparison implementations plus the imported v2 and quant-training helpers.
Every training entry point rejects implementation drift before reading a
modeling panel.

The final comparison uses 2,000 fold-contained moving whole-date bootstrap
resamples, ten-session blocks, and seed 1729. All dates are already-inspected
development dates, and the underlying retrospective article text is not
version-safe.

## Completed results

The canonical/reversed FLAN event-label agreement was 65.124386%, below the
frozen 85% semantic-stability gate. This was a failed-gate exploratory run;
none of the results below is primary or confirmatory evidence.

| Rung | T1 ETF RMSE | T1 LOO RMSE | T2 ETF RMSE | T2 LOO RMSE |
|---|---:|---:|---:|---:|
| `S0` Q56 | 0.370931 | 0.385767 | 0.244842 | 0.263394 |
| `S1` Q56+D2 | 0.371505 | 0.385682 | 0.242221 | 0.262505 |
| `S2` Q56+WL17 | 0.371240 | 0.386277 | 0.243367 | 0.264687 |
| `S3` Q56+D2+WL17 | 0.372014 | 0.386355 | 0.241935 | 0.264318 |

For `S2` versus matched `S0`, only T2 ETF improved: incremental MSE
$R^2=1.2015\%$, with a paired 95% loss-delta interval of
`[-0.001196, -0.000332]`. T1 ETF, T1 LOO, and T2 LOO worsened. For `S3`
versus matched `S1`, T2 ETF improved by only 0.2362%, with an interval crossing
zero; the other three targets worsened.

No S2 or S3 target passed the complete useful-semantic gate. In particular,
the apparent S2 T2 ETF gain failed the coverage/text-only, stale-event,
wrong-stock, and date-sector permutation requirements. The 20-session-stale
and permuted variants had lower point losses than the live S2 T2 ETF model.
WL17 coefficient selection was also unstable across folds.

The validation gate admitted only T2 ETF to `S4`. Its GPU XGBoost RMSE was
0.233668, a 6.9374% MSE reduction versus linear `S1` and 6.7171% versus linear
`S3`; both paired intervals were favorable. This is a nonlinear
Q56+D2+WL17 result, not isolated semantic evidence: there is no matched
Q56+D2-only XGBoost in this ladder, and the input semantic block failed both
its upstream stability gate and every downstream useful-semantic gate.

Canonical handoff:

- [Final comparison](comparisons/final/RESULTS.md)
- [Per-bundle status](STATUS.md)
- [Locked protocol](../../../../config/quant_deterministic_news_protocol_v3.json)
