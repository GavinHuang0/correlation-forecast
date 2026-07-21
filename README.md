# Forecasting Stock–Sector Coupling

Research project investigating whether point-in-time financial news improves forecasts of how strongly an individual stock will move with its sector.

## Research question

Do LLM-extracted event features add predictive and economic value beyond quantitative correlation models when forecasting daily stock–sector coupling?

The main hypothesis is that the scope of a new information shock matters:

- firm-specific events may make a stock move more idiosyncratically;
- sector-wide events may strengthen stock–sector coupling; and
- macroeconomic events may strengthen broad common-factor exposure.

## Initial design

- Measure realized stock–sector correlation from intraday returns.
- Use a pooled panel of liquid stocks and their corresponding sector benchmarks.
- Establish quantitative baselines using lagged realized correlation, HAR-style features, exponentially weighted correlation, DCC-GARCH, and regularized regression.
- Use a frozen, pre-evaluation-period language model to extract structured news features such as event scope, event type, affected entities, direction, uncertainty, and expected horizon.
- Evaluate all models with chronological, point-in-time validation.
- Test economic value through hedge error, portfolio risk, and coupling-aware trading filters—not forecast error alone.

## Research principles

1. Use only information that was available before each forecast cutoff.
2. Keep every stock observed on the same date in the same temporal fold.
3. Freeze and record news-extraction models, prompts, schemas, and outputs.
4. Compare LLM features against strong quantitative and simple-text baselines.
5. Separate semantic feature extraction from the downstream forecasting model.
6. Keep licensed news and market data out of the public repository.

## Planned model ladder

1. Previous realized correlation
2. HAR / exponentially weighted correlation
3. DCC-GARCH and quant-only regularized models
4. Quant features plus news counts or conventional sentiment
5. Quant features plus structured LLM event features
6. Quant features plus LLM embeddings
7. Combined structured-feature and embedding model

## Status

The project is in the research-design phase. This initial repository intentionally contains documentation only; implementation and data pipelines will be added later.
