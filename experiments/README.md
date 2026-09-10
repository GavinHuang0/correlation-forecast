# Experiment index

Experiment directories preserve protocols, result summaries, controls,
reviews, and artifact pointers. Forecast selection status is defined only by
the [`active model registry`](../models/active/registry.json); directories not
listed there are not implicitly active.

## Active forecast evidence

| Experiment | Role | Status |
|---|---|---|
| [`quant_training/v2/`](quant_training/v2/README.md) | Source experiment for the four target-specific long-Q winners | Selected quantitative family |
| [`quant_deterministic_news/v4/`](quant_deterministic_news/v4/README.md) | Source experiment for RRES-C6 and the newest Q+L controls | RRES-C6 selected only for T2 ETF; other forecast candidates archived |

## Archived forecast experiments

| Experiment | Status |
|---|---|
| [`quant_training/v1/`](quant_training/v1/README.md) | Superseded by the longer quant-v2 evaluation |
| [`quant_deterministic_news/v1/`](quant_deterministic_news/v1/README.md) | Archived deterministic-news forecast experiment |
| [`quant_deterministic_news/v2/`](quant_deterministic_news/v2/README.md) | Archived normalized D2 and semantic design experiment |
| [`quant_deterministic_news/v3/`](quant_deterministic_news/v3/README.md) | Archived W17-Lite Q+L/Q+D+L experiment |

## Extractor and data-source research

The FLAN and Llama directories evaluate semantic extraction rather than final
correlation forecasts. Their active status is governed separately by
[`active_extractor.json`](active_extractor.json). News-provider and
stock/ETF-spread directories retain data-source and downstream economic-test
research.

## Downstream economic experiments

| Experiment | Scope | Outcome |
|---|---|---|
| [`stock_etf_spread/v1/`](stock_etf_spread/v1/README.md) | Correlation-gated convergence on the shorter-history XGBoost forecasts | Initial development backtest; cost and stability improvements needed |
| [`stock_etf_spread/v2/`](stock_etf_spread/v2/README.md) | Same strategy applied to the full-history quant ensemble | Unprofitable before costs over the full sample; strategy redesign pending |

Original paths are intentionally stable. Moving old evidence into new folders
would invalidate embedded paths and make historical manifests harder to
audit; archive status is therefore expressed through the public registries
and this index.
