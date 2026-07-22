# Forecasting Stock-Sector Coupling

Research project investigating whether point-in-time financial news improves forecasts of how strongly an individual stock will move with its sector.

## Research question

Do LLM-extracted event features add predictive and economic value beyond quantitative correlation models when forecasting daily stock-sector coupling?

The main hypothesis is that the scope of a new information shock matters:

- firm-specific events may make a stock move more idiosyncratically;
- sector-wide events may strengthen stock-sector coupling; and
- macroeconomic events may strengthen broad common-factor exposure.

## Current benchmark

The repository now contains the version-0 semantic extraction protocol and a 300-article GPT-5.6 Sol silver-reference label set for evaluating frozen FLAN-T5-Large. Article text and licensed/raw provider data remain Git-ignored.

- Human-readable schema: [`docs/news_feature_schema.md`](docs/news_feature_schema.md)
- Machine-readable schema: [`config/news_feature_schema.json`](config/news_feature_schema.json)
- FLAN-T5 procedure: [`docs/flan_t5_evaluation.md`](docs/flan_t5_evaluation.md)
- Reference labels: [`annotations/chatgpt_5_6_sol_reference.jsonl`](annotations/chatgpt_5_6_sol_reference.jsonl)

The reference labels measure agreement with GPT-5.6 Sol, not objective ground-truth accuracy. Exact evidence is preserved so disagreements can be manually audited.

## Initial research design

- Measure realized stock-sector correlation from intraday returns.
- Use a pooled panel of liquid stocks and their sector benchmarks.
- Establish quantitative baselines using lagged realized correlation, HAR-style features, exponentially weighted correlation, DCC-GARCH, and regularized regression.
- Use a frozen, pre-evaluation-period language model to extract structured news features.
- Evaluate all forecasting models with chronological, point-in-time validation.
- Test economic value through hedge error, portfolio risk, and coupling-aware trading filters, not forecast error alone.

## Research principles

1. Use only information that was available before each forecast cutoff.
2. Keep every stock observed on the same date in the same temporal fold.
3. Freeze and record extraction models, prompts, schemas, runtime versions, and outputs.
4. Compare LLM features against strong quantitative and simple-text baselines.
5. Separate semantic extraction from the downstream forecasting model.
6. Keep licensed news, raw API payloads, secrets, model weights, and market data out of Git.

## Planned model ladder

1. Previous realized correlation
2. HAR and exponentially weighted correlation
3. DCC-GARCH and quant-only regularized models
4. Quant features plus news counts or conventional sentiment
5. Quant features plus structured LLM event features
6. Quant features plus LLM embeddings
7. Combined structured-feature and embedding model
