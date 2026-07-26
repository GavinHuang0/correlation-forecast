# FLAN-T5-XL v1.0 Archive

v1.0 was the first full-precision XL comparison. It used the pinned
`google/flan-t5-xl` checkpoint, the v0.4 coarse prompts, and canonical/reversed
option-order averaging without supervised score calibration.

On the 72-document development split, its mean field accuracy was 0.563 and
its mean macro-F1 was 0.411. On the later locked 228-document comparison, its
mean field accuracy was 0.546 and its mean macro-F1 was 0.407.

The experiment was superseded by v1.1, which applies a field-specific
development-selected score-source and log-prior adjustment to the exact same
raw inference outputs. No evaluation labels were used to select v1.1.

The compact original development result is
[`development_summary.json`](development_summary.json). Generated v1.0
predictions remain Git-ignored under `outputs/flan_t5_xl/v1_0/` and as the
raw source artifacts under `outputs/flan_t5_xl/v1_1/`.
