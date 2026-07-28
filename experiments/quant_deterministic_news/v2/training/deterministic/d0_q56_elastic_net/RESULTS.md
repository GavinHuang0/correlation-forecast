# V2-D0 matched Q56 Elastic Net

Status: **complete**

Completed: `2026-07-28T06:49:11.620658Z`

| Target | Rows | Fisher-z RMSE | Fisher-z MAE | Raw RMSE | OOS R2 vs persistence |
|---|---:|---:|---:|---:|---:|
| t1_etf | 11190 | 0.370931 | 0.292976 | 0.208578 | 0.407915 |
| t1_loo | 11190 | 0.385767 | 0.303961 | 0.231278 | 0.382590 |
| t2_etf | 10830 | 0.244842 | 0.192202 | 0.130507 | 0.226441 |
| t2_loo | 10830 | 0.263394 | 0.204305 | 0.154095 | 0.211898 |

## Review

- Review status: **passed**
- Prediction keys unique: `True`
- Predictions finite: `True`
- Bounds valid: `True`
- Outer evaluation excluded from tuning/gating: `True`

## Artifacts

- fits.json: `outputs/quant_deterministic_news/v2/deterministic/d0_q56_elastic_net/fits.json` - SHA-256 `13002dacb2c4327d933b66a26adab13e7787140086ca4251f79f2d5570d6db1f`
- fold_metrics.json: `outputs/quant_deterministic_news/v2/deterministic/d0_q56_elastic_net/fold_metrics.json` - SHA-256 `fc9a3f47c5f13c27f1ec49e10394c217f405454c4a6f7d9930b7463c1decf458`
- predictions.parquet: `outputs/quant_deterministic_news/v2/deterministic/d0_q56_elastic_net/predictions.parquet` - SHA-256 `6e3a284c5440a891d00cb25105ad922e74e68db3ad6586752c8aca93d5642359`
- source_preflight.json: `outputs/quant_deterministic_news/v2/deterministic/d0_q56_elastic_net/source_preflight.json` - SHA-256 `635158e589b382b7651c92bd595bcb84477388fe7114ab7f434d1b4c49b6ab5a`
- validation_predictions.parquet: `outputs/quant_deterministic_news/v2/deterministic/d0_q56_elastic_net/validation_predictions.parquet` - SHA-256 `a32739966cee7f43b090d842409135e3b10963f8ec43cf888b2608312d8cd882`
- manifest.json: `outputs/quant_deterministic_news/v2/deterministic/d0_q56_elastic_net/manifest.json` - SHA-256 `83f45fb491ae2f5868c5bf5e550ef360c2a757c4ca73f0d45e22d7d6824760da`

This is an exploratory development result from retrospective news.
It is not confirmatory point-in-time evidence.
