# Archived FLAN-T5 Experiments

This directory contains non-operational history. Nothing here should be used
as the default extractor.

| Version | Outcome | Documentation | Local outputs |
|---|---|---|---|
| v0.1 | Grouped generation violated the output contract and yielded null parsed labels | [`v0_1_v0_2/protocol_and_null_fix.md`](v0_1_v0_2/protocol_and_null_fix.md) | `outputs/flan_t5/archive/v0_1/` |
| v0.2 | One-field constrained decoding fixed nulls, but semantic agreement remained weak | [`v0_1_v0_2/protocol_and_null_fix.md`](v0_1_v0_2/protocol_and_null_fix.md) | `outputs/flan_t5/archive/v0_2/` |
| v0.3 | Development/sanity ablations led to v0.4 but were never promoted | [`v0_3/README.md`](v0_3/README.md) | `outputs/flan_t5/archive/v0_3/` |
| v0.5 | Binary-component candidate failed its preregistered promotion rule | [`v0_5/README.md`](v0_5/README.md) | `outputs/flan_t5/archive/v0_5/` |

The usable FLAN baseline is
[`../v0_4/README.md`](../v0_4/README.md). The archived v0.5 config, scripts,
and tests are stored beside its protocol so they do not appear among current
operational code.

Completed historical manifests and `v0.5/protocol_lock.json` retain their
original paths by design. The
[`repository_reorganization/`](repository_reorganization/README.md) record
maps those historical locations to the current archive.
