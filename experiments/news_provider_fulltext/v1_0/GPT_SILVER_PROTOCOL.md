# GPT-5.6 Sol full-text silver-label protocol

Protocol version: `news-fulltext-gpt-silver-v1.0.0`

This protocol creates a silver reference for the isolated 300-document
provider/full-text experiment. It does not modify or reuse the active
FLAN-T5-XL or Llama 3.1 experiments.

## Eligibility and claim

- Every parent article must be published strictly after
  `2026-03-01T00:00:00Z`.
- The date is later than the documented GPT-5.6 Sol knowledge cutoff used by
  this experiment.
- GPT sees the complete reconstructed parent article, not a Massive
  description, Alpha Vantage summary, or an individual model-input chunk.
- These annotations are a GPT-5.6 Sol **silver reference**, not human ground
  truth and not a claim of objective accuracy.
- No forecasting output, future return, or future correlation is supplied.

## Exact system instruction

Use the following text verbatim:

```text
You are creating point-in-time silver annotations for a quantitative research
dataset about stock-sector return correlation.

Use only the supplied headline, complete article text, and target-company
metadata. Do not browse the web or retrieve external content. Local filesystem
tools may be used only to read the supplied protocol and batch and to write the
required JSONL output. Do not use external facts, remembered company facts,
later events, market outcomes, stock returns, or anything not explicitly
stated in the supplied record.

This is semantic extraction only. Do not forecast returns, volatility, beta,
correlation, prices, or profitability.

Return exactly one JSON object per input document, in the same order, as JSONL.
Return no prose, Markdown, or code fences.

Copy article_id exactly. Set annotator to "gpt-5.6-sol" and protocol_version to
"news-fulltext-gpt-silver-v1.0.0".

Return labels with exactly these four keys in this order:

1. shock_scope
   - idiosyncratic: the primary shock concerns one firm, whether the target or
     one sector peer
   - common: the shock explicitly affects several sector firms, the sector, or
     the broad market
   - mixed: material firm-specific and common components coexist
   - unclear: the supplied text is irrelevant, insufficient, or does not
     establish scope

   A company belonging to a sector does not by itself make the event common.
   Use common only when breadth is explicit in the supplied text.

2. event_family
   - earnings_guidance: reported results, forecasts, outlook, or guidance
   - product_demand: products, technology, customers, contracts, or demand
   - supply_capacity: supply chains, production capacity, manufacturing
     constraints, or inventory availability
   - regulation_legal: regulation, trade policy, litigation, governance, or
     operational/legal matters
   - corporate_analyst: analyst actions or corporate actions such as M&A,
     financing, restructuring, or management changes
   - macro_market: macroeconomic or broad-market developments
   - other_or_unclear: no listed family is supported or the family is unclear

3. information_status
   - confirmed: completed, officially announced, filed, reported, or enacted
   - anticipated: scheduled, planned, forecast, expected, or upcoming
   - rumor_or_opinion: rumored, unconfirmed, analytical, or opinion-based
   - unclear: status is not established by the supplied text

4. directional_alignment
   - single_firm_only: the shock is idiosyncratic and no explicit same/opposite
     peer effect is stated
   - same_direction: the target and sector or peers are explicitly affected in
     the same direction
   - opposite_direction: the target benefits at peers' expense, or vice versa
   - common_direction_unclear: a common or mixed shock exists, but directional
     alignment is not established
   - unclear: the article is inapplicable or its shock scope is unclear

The scope/alignment combination must obey this hierarchy:
- unclear scope requires unclear alignment
- idiosyncratic scope permits single_firm_only, same_direction, or
  opposite_direction
- common or mixed scope permits same_direction, opposite_direction, or
  common_direction_unclear

Do not infer direction from general financial knowledge. Same-direction and
opposite-direction labels require explicit textual support.

Return evidence with exactly the same four keys and in the same order as
labels. Each nonempty evidence value must be a short, exact, case-sensitive
substring copied from the supplied headline or complete article text, no more
than 280 characters. Do not paraphrase evidence. Use an empty string when the
text does not support a useful excerpt.

Use abstain_reason to explain material ambiguity in one short sentence. Use
null when no abstention explanation is needed. Abstain conservatively with an
unclear label instead of guessing.

The exact output shape is:
{"article_id":"","annotator":"gpt-5.6-sol","protocol_version":"news-fulltext-gpt-silver-v1.0.0","labels":{"shock_scope":"","event_family":"","information_status":"","directional_alignment":""},"evidence":{"shock_scope":"","event_family":"","information_status":"","directional_alignment":""},"abstain_reason":null}
```

## Exact annotation instruction for each batch

Use the following text verbatim, followed by one private 20-document batch:

```text
Annotate every input object in the attached JSONL batch.

Each input contains only an article_id, headline, complete reconstructed
full_text, target metadata, and a document_number used for ordering. Treat
full_text as the complete parent article for this task. Do not label chunks
separately. Do not use article_id or document_number as semantic evidence.

Produce exactly one output JSON object for every input article_id, preserve
input order, and follow the system instruction's exact schema. Use only the
supplied batch. Return JSONL only.

INPUT JSONL:
```

## Raw part format

Each private result row must have exactly this shape:

```json
{
  "article_id": "copied-parent-id",
  "annotator": "gpt-5.6-sol",
  "protocol_version": "news-fulltext-gpt-silver-v1.0.0",
  "labels": {
    "shock_scope": "one allowed label",
    "event_family": "one allowed label",
    "information_status": "one allowed label",
    "directional_alignment": "one allowed label"
  },
  "evidence": {
    "shock_scope": "short exact excerpt or empty string",
    "event_family": "short exact excerpt or empty string",
    "information_status": "short exact excerpt or empty string",
    "directional_alignment": "short exact excerpt or empty string"
  },
  "abstain_reason": null
}
```

Raw parts remain under the ignored path:

```text
annotations/news_provider_fulltext/v1_0/parts/
```

Use one output file per input batch, for example:

```text
batch_001.jsonl
-> annotations/news_provider_fulltext/v1_0/parts/batch_001.jsonl
```

For collaboration-agent execution, request the `gpt-5.6-sol` model explicitly
and tell the agent to read this protocol completely before reading its assigned
batch. A 20-document batch can be processed one JSONL record at a time to avoid
tool-output truncation. The agent must not open web pages or search for article
context. It may use local read-only commands and `apply_patch` to create only
its assigned part file.

The collaboration surface cannot replace its platform system instructions or
pin a dated API snapshot. The merge manifest therefore records:

```text
surface: Codex collaboration agent
requested model: gpt-5.6-sol
dated API snapshot pinned: false
protocol supplied through tracked file plus task message
```

This limitation must remain in the final methodology report.

## Preparation and validation

Prepare deterministic 20-parent-document batches:

```powershell
python scripts/prepare_gpt56_silver_batches.py
```

After all 15 raw parts exist, validate and merge:

```powershell
python scripts/merge_gpt56_silver_labels.py
```

The validator requires exactly one result per parent article, validates labels
against `config/news_feature_schema_coarse.json`, and verifies every nonempty
evidence excerpt as an exact source substring. The merged annotation strips
full text, headlines, evidence strings, and abstention-reason text. It retains
only labels, identifiers, provenance, grounding flags, and SHA-256 evidence
hashes.
