# FLAN-T5-XL Extractor

FLAN-T5-XL v1.1 is the repository's active research extractor. The canonical
machine-readable pointer is
[`../active_extractor.json`](../active_extractor.json), and the complete
protocol and result are under [`v1_1/`](v1_1/README.md).

It is active because it is the strongest tested extractor on the fixed
228-document comparison, not because it is production-ready:

| Extractor | Mean accuracy | Mean macro-F1 |
|---|---:|---:|
| FLAN-T5-Large v0.4 | 0.494 | 0.446 |
| FLAN-T5-XL raw v1.0 | 0.546 | 0.407 |
| **FLAN-T5-XL calibrated v1.1** | **0.612** | **0.464** |

Macro-F1 is the primary selection metric because the semantic classes are
imbalanced. v1.1 also has the highest mean accuracy, but none of its four
semantic fields passed the preregistered macro-F1 thresholds. Its permitted
uses are exploratory extraction, integration testing, and model comparison.

## Repository layout

```text
experiments/flan_t5_xl/
  README.md
  registry.json
  v1_1/                 # active implementation contract and results
  archive/v1_0/         # uncalibrated development baseline
```

Generated predictions remain under `outputs/flan_t5_xl/` and are Git-ignored.
The frozen model snapshot is reused from the Hugging Face cache and verified
against `outputs/flan_t5_xl/model_snapshot_manifest.json`.

## Active command

Run from the repository root in PowerShell:

```powershell
$env:HF_HUB_OFFLINE = "1"
.\.venv-flan-t5-xl\Scripts\python.exe scripts\run_flan_t5_xl_active.py `
  --input <POINT_IN_TIME_INPUT.jsonl> `
  --output <CALIBRATED_OUTPUT.jsonl>
```

The wrapper runs the pinned raw XL extractor and then applies the frozen v1.1
calibration. It preserves the raw predictions beside the final output using a
`.raw.jsonl` suffix. Use `--validate-only` before a large run. If inference
completed but postprocessing was interrupted, rerun with `--apply-only` and
the same `--raw-output`.

## Historical baseline

The raw v1.0 development result and its original decision are retained under
[`archive/v1_0/`](archive/v1_0/README.md). They are history, not an operational
path.
