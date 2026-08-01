# Reproducibility

The repository separates tracked protocols and summaries from generated or
licensed artifacts. A complete local run can validate hashes and regenerate
the result tables, while the public Git history does not redistribute news
text, credentials, large predictions, or binary model weights.

## Canonical protocols

| Experiment | Protocol | Result summary |
|---|---|---|
| Long-Q v2 | [`../config/quant_training_protocol_v2.json`](../config/quant_training_protocol_v2.json) | [`../experiments/quant_training/v2/comparisons/summary.json`](../experiments/quant_training/v2/comparisons/summary.json) |
| Semantic v4 | [`../config/quant_deterministic_news_protocol_v4.json`](../config/quant_deterministic_news_protocol_v4.json) | [`../experiments/quant_deterministic_news/v4/training/comparisons/final/summary.json`](../experiments/quant_deterministic_news/v4/training/comparisons/final/summary.json) |

The v4 protocol SHA-256 is
`361c235e46bd9322525534da20df98a2f72820670a027739465f620134bc63fe`.
The selected RRES-C6 output-manifest SHA-256 is
`f1be949ef9ac9d70458f1a5e6f2f699bd5cefc36ac21afcb10bc0f85f6e33b2e`.

## Long-Q reproduction

With the repository's training environment and local generated data present:

```powershell
$python = '.\.venv-training\Scripts\python.exe'
$protocol = 'config\quant_training_protocol_v2.json'

& $python -m scripts.correlation_training.build_modeling_panel `
  --core data\features\quant\training_v1\bollerslev_core_features.parquet `
  --context data\features\quant\training_v1\additional_quant_features.parquet `
  --targets data\features\quant\training_v1\correlation_targets_and_loo_features.parquet `
  --output data\features\quant\training_v2\modeling_panel.parquet `
  --start 2017-12-28 --end 2026-06-30 `
  --audit-output experiments\quant_training\v2\construction\audits\modeling_panel.json

& $python -m scripts.correlation_training.run_rung_01 --protocol $protocol
& $python -m scripts.correlation_training.run_rung_02 --protocol $protocol
& $python -m scripts.correlation_training.run_rung_03 --protocol $protocol
& $python -m scripts.correlation_training.run_rung_04 --protocol $protocol
& $python -m scripts.correlation_training.summarize_ladder --protocol $protocol
```

The expected selected rows are recorded in
[`../models/active/registry.json`](../models/active/registry.json). Rung 3
predictions contain the selected T1 XGBoost and T2 ETF ensemble streams; Rung
1 contains the selected T2 LOO core-22 LASSO stream.

## Semantic v4 reproduction

V4 depends on the completed cached FLAN inference ledger, the constructed
role-conditioned panels, saved long-Q out-of-sample forecasts, and the frozen
v4 protocol. The detailed build and training commands are kept in
[`../experiments/quant_deterministic_news/v4/README.md`](../experiments/quant_deterministic_news/v4/README.md).

The selected model card contains an artifact pointer:

```text
outputs/quant_deterministic_news/v4/models/rres_coupling6_en/manifest.json
```

That local manifest binds predictions, validation predictions, fitted
configuration, dependencies, and row counts. Generated outputs are ignored by
Git, but the pointer, manifest hash, protocol hash, selected configuration,
review, and metrics are tracked under
[`../experiments/quant_deterministic_news/v4/training/models/rres_coupling6_en/`](../experiments/quant_deterministic_news/v4/training/models/rres_coupling6_en/).

## Verification

Run the repository tests from the project root:

```powershell
.\.venv-training\Scripts\python.exe -m unittest discover -s tests -p 'test_*.py'
```

The v4 completion audit verifies:

- protocol and implementation hashes;
- declared source artifacts;
- expected bundle manifests;
- SHA-256 and byte-size checks;
- Parquet row counts and unique keys;
- fold and target alignment;
- exact identity of the R0 projection with its long-Q anchor; and
- completeness of the 92 paired comparisons and 16 useful-gate records.

The active-registry tests additionally verify that the four long-Q entries
equal the stored target-level winners and that the RRES-C6 T2 ETF statistics
equal the stored paired-comparison result.

Publication-time verification on 2026-08-01 passed **461/461 tests**.

## Public/private boundary

The following are intentionally not committed:

- licensed or raw article text and provider payloads;
- downloaded price data and generated feature panels;
- prediction Parquet files and fitted binary artifacts;
- local model checkpoints and environments; and
- API credentials and `.env` files.

Public summaries contain no credential values or redistributed news text.
Reproduction therefore requires lawful access to the underlying inputs or a
local copy whose hashes match the recorded manifests.

## Scientific boundary

Bitwise artifact reproducibility does not turn a retrospective development
comparison into confirmation. The next confirmatory evaluation must use a
protocol frozen before outcomes, a version-preserving news source, and dates
strictly after 2026-06-30.
