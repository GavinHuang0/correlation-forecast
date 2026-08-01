# Forecast model catalog

This directory is the public entry point for model-selection status. It does
not duplicate generated predictions or binary model files.

- [`active/`](active/README.md) defines the selected forecasting suite and its
  exact target routing.
- [`archive/`](archive/README.md) records the nonselected forecasting families
  and explains why each was archived.

The experiment directories under [`../experiments/`](../experiments/README.md)
remain the source of truth for protocols, metrics, comparisons, controls, and
artifact hashes. Generated data and fitted outputs remain Git-ignored because
some inputs are licensed and the full artifacts are large.

“Active” means selected for continued research and prospective evaluation. It
does not mean production-ready, investment-ready, or confirmed on a wholly
untouched future period.
