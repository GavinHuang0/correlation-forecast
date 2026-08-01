# Archived forecasting models

Status: **retained as reproducibility evidence; not selected for active
forecast routing**.

Archiving is status-based. The original files stay under `experiments/` so
that paths embedded in protocols, manifests, reports, and tests remain valid.
The machine-readable archive is [`registry.json`](registry.json).

## Archived groups

| Group | Archived content | Reason |
|---|---|---|
| Quant v1 | Original three-fold quantitative ladder and its winners | Superseded by the longer 13-fold quant-v2 evaluation |
| Quant v2 nonwinners | Persistence, HAR/SHAR/OLS alternatives, nonselected regularized models, nonselected target/model pairings, and DCC-GARCH | Higher pooled Fisher-z RMSE than the selected model for the same target, or failure of the rung's validation gate |
| Deterministic-news v1-v3 | Q+D, Q+L, Q+D+L, residual, controls, and sensitivities | No forecast target passed the complete usefulness/falsification requirements |
| V4 short-history models | J1, J2, J3 and short controls | J1/J2 failed full gates; J3 was diagnostic and lacked matched controls |
| V4 long-history alternatives | R0 catalog candidate, RCAL, RRES-L19, RSTACK and long controls | R0 remains an evidence anchor rather than target winner; RCAL/RRES-L19/RSTACK were inferior or failed complete gates |
| RRES-C6 on non-T2-ETF targets | T1 ETF, T1 LOO, and T2 LOO outputs from the same bundle | The matched result was not statistically positive; two targets worsened |

The FLAN, Llama, and GPT-related directories are semantic-extractor research,
not alternative forecast endpoints. Their status remains governed by
[`../../experiments/active_extractor.json`](../../experiments/active_extractor.json)
and the extractor-specific registries.

Archived results are scientifically useful: they document negative results,
placebos, control behavior, and failure modes. “Archived” means they must not
be silently substituted into the active forecast route.
