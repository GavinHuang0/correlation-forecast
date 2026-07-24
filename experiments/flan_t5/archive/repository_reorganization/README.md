# Repository Reorganization Record

On 2026-07-23 the model assets were separated into operational and archival
namespaces.

Operational:

```text
experiments/flan_t5/v0_4/
outputs/flan_t5/v0_4/
outputs/flan_t5/shared/
experiments/llama_2/
outputs/llama_2/
```

Archived:

```text
experiments/flan_t5/archive/v0_1_v0_2/
experiments/flan_t5/archive/v0_3/
experiments/flan_t5/archive/v0_5/
outputs/flan_t5/archive/v0_1/
outputs/flan_t5/archive/v0_2/
outputs/flan_t5/archive/v0_3/
outputs/flan_t5/archive/v0_5/
```

The v0.5-only config, four scripts, and their tests were moved under
`archive/v0_5/implementation/`. The original one-time relocation utility and
test are retained under this directory. They are historical source snapshots,
not current commands.

The ignored output move and compatibility-copy hashes are recorded in
[`organization_manifest_v2.json`](organization_manifest_v2.json). The
original generated relocation manifest is preserved at
`outputs/flan_t5/archive/repository_reorganization/relocation_manifest_v1.json`
without rewriting its historical paths.
