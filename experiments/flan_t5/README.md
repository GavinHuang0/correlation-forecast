# FLAN-T5 Experiment Layout

Only FLAN-T5-Large v0.4 is operational at the top level:

- [`v0_4/README.md`](v0_4/README.md): frozen implementation and reproduction
  protocol;
- [`v0_4/evaluation_summary.json`](v0_4/evaluation_summary.json): compact
  checked-in agreement result;
- [`active_baseline.json`](active_baseline.json): machine-readable operational
  pointer; and
- [`registry.json`](registry.json): full version history.

Earlier experiments and the rejected v0.5 candidate are separated under
[`archive/`](archive/README.md). They are retained for provenance and are not
operational entry points.

## Active output layout

```text
outputs/flan_t5/
  shared/                         # fixed 300-article benchmark
  v0_4/                           # active predictions/results
    dependencies/v0_2_fine/       # exact compatibility input for the hybrid
  archive/                        # v0.1-v0.3 and rejected v0.5 outputs
```

The active v0.4 score is a fieldwise hybrid selected on the fixed development
split. `event_family` and `information_status` use the v0.4 coarse prompt,
while `shock_scope` and `directional_alignment` use mappings from the same
pinned FLAN model's v0.2 fine prompt. Consequently,
[`scripts/extract_flan_t5.py`](../../scripts/extract_flan_t5.py) remains an
internal compatibility dependency of the active v0.4 protocol; it is not a
separately usable legacy model.

The required v0.2 prediction and manifest were copied into the active
dependency folder with hashes:

```text
predictions  758b14c285f61d9d621ee5321b6114acb862ade4515268687664b8f44bbe197a
manifest     9a34fa3d993d2bf18ac51885a2a4da10b6a54c055c54a7fb554c6cfce12206ac
```

See the
[`repository reorganization record`](archive/repository_reorganization/README.md)
for the archive move map.
