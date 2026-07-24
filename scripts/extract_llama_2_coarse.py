from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import coarse_news_features as coarse
import extract_flan_t5 as base
import extract_flan_t5_coarse as flan_coarse


MODEL_DEFAULT = "meta-llama/Llama-2-7b-chat-hf"
REVISION_DEFAULT = "f5db02db724555f92da89c216ac04704f23d4590"
PROMPT_VERSION = "llama-2-stock-sector-news-v1.0.0"
MANIFEST_VERSION = "llama-2-extraction-v1"
CONSERVATIVE_DATA_CUTOFF = "2023-07-31"
CONSERVATIVE_DATA_CUTOFF_DATE = date(2023, 7, 31)
MODEL_CONTEXT_WINDOW = 4096
DECODING_METHOD = "order_averaged_letter_score"
PROMPT_PROFILE = "zero_shot"
LLAMA_2_SINGLE_TURN_CHAT_TEMPLATE = (
    "{% if messages|length != 1 or messages[0]['role'] != 'user' %}"
    "{{ raise_exception('Llama 2 extractor requires exactly one user message') }}"
    "{% endif %}"
    "{{ bos_token + '[INST] ' + messages[0]['content'].strip() + ' [/INST]' }}"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a pinned local Llama 2 7B Chat checkpoint on the versioned "
            "coarse financial-news protocol. The decoder scores one-letter choices "
            "under canonical and reversed option order; it never generates free text."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--schema",
        default=Path("config/news_feature_schema_coarse.json"),
        type=Path,
    )
    parser.add_argument("--model-id", default=MODEL_DEFAULT)
    parser.add_argument(
        "--revision",
        default=REVISION_DEFAULT,
        help="Immutable 40-character Hugging Face commit hash.",
    )
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=MODEL_CONTEXT_WINDOW)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument(
        "--quantization",
        choices=("none", "nf4"),
        default="nf4",
        help="Use nf4 only with CUDA; CPU inference requires full precision.",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help=(
            "Validate inputs, dates, deterministic routing, schema, and raw prompt "
            "construction without importing or loading model packages."
        ),
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.model_id != MODEL_DEFAULT:
        parser.error(f"--model-id must remain pinned to {MODEL_DEFAULT}")
    if not re.fullmatch(r"[0-9a-fA-F]{40}", args.revision):
        parser.error("--revision must be an immutable 40-character hexadecimal commit hash")
    if args.revision.lower() != REVISION_DEFAULT:
        parser.error(f"--revision must remain pinned to {REVISION_DEFAULT}")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not 512 <= args.max_input_tokens <= 4096:
        parser.error("--max-input-tokens must be between 512 and 4096")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.quantization == "nf4" and args.device == "cpu":
        parser.error("--quantization nf4 requires CUDA (use --device cuda or auto)")
    if args.device == "cpu" and args.precision == "float16":
        parser.error("float16 CPU inference is unsupported; use float32 or bfloat16")
    return args


def article_first_context(record: Mapping[str, Any]) -> str:
    return f"ARTICLE\nHeadline: {record['headline']}\nText: {record['article_text']}"


def target_context(record: Mapping[str, Any], *, include_peers: bool) -> str:
    target = record["target"]
    lines = [
        "TARGET METADATA",
        f"Target company: {target['company']} ({target['ticker']})",
        f"Target sector: {target['sector']}",
        f"Sector benchmark: {target['sector_benchmark']}",
    ]
    if include_peers:
        peers = ", ".join(target["known_sector_peers"]) or "none supplied"
        lines.append(f"Known sector peers: {peers}")
    return "\n".join(lines)


def field_task(
    record: Mapping[str, Any],
    field: str,
    selected_scope: str | None,
) -> str:
    source_rule = (
        "Use only the supplied article and target metadata. Do not use external "
        "knowledge, remembered events, later outcomes, market prices, or facts not "
        "stated in the input."
    )
    if field == "shock_scope":
        return (
            f"{target_context(record, include_peers=False)}\n\n"
            "TASK\nClassify the primary scope of the information shock. A company "
            "belonging to a sector does not by itself make an event common. "
            f"{source_rule}"
        )
    if field == "event_family":
        return (
            "TASK\nClassify the article's primary event family. Choose the family "
            "supported most directly by the text. "
            f"{source_rule}"
        )
    if field == "information_status":
        return (
            "TASK\nClassify the status of the primary information. Distinguish an "
            "officially reported or completed event from an expected event, rumor, "
            "or opinion. "
            f"{source_rule}"
        )
    if field == "directional_alignment":
        if selected_scope not in {"idiosyncratic", "common", "mixed"}:
            raise ValueError(
                f"directional_alignment requires a resolved scope, got {selected_scope!r}"
            )
        fallback = (
            "For a common or mixed shock whose direction is not explicitly stated, "
            "choose common_direction_unclear."
            if selected_scope in {"common", "mixed"}
            else "For an idiosyncratic shock without an explicit relative effect, "
            "choose single_firm_only."
        )
        return (
            f"{target_context(record, include_peers=True)}\n"
            f"Predicted shock scope: {selected_scope}\n\n"
            "TASK\nClassify only the direction explicitly stated for the target "
            "versus its sector or peers. Do not predict returns or invent an "
            f"unstated comparison. {fallback} {source_rule}"
        )
    raise ValueError(f"Unknown coarse field {field!r}")


def label_letter_mapping(labels_in_order: Sequence[str]) -> dict[str, str]:
    if len(labels_in_order) > 26:
        raise ValueError("Letter-choice protocol supports at most 26 labels")
    if len(labels_in_order) != len(set(labels_in_order)):
        raise ValueError("Letter-choice labels must be unique")
    return {
        label: chr(ord("A") + index) for index, label in enumerate(labels_in_order)
    }


def build_letter_prompt(
    record: Mapping[str, Any],
    schema: Mapping[str, Any],
    field: str,
    labels_in_order: Sequence[str],
    selected_scope: str | None = None,
) -> tuple[str, dict[str, str]]:
    definitions = schema["field_definitions"][field]
    allowed = set(schema["closed_label_fields"][field])
    if set(labels_in_order) - allowed:
        raise ValueError(f"Prompt contains unsupported {field} label")
    mapping = label_letter_mapping(labels_in_order)
    options = "\n".join(
        f"{mapping[label]}. {label}: {definitions[label]}"
        for label in labels_in_order
    )
    prompt = (
        f"{article_first_context(record)}\n\n"
        f"{field_task(record, field, selected_scope)}\n\n"
        f"OPTIONS\n{options}\n\n"
        "Answer with exactly one option letter and nothing else."
    )
    return prompt, mapping


def validate_schema(schema: Mapping[str, Any]) -> None:
    # The Llama comparison intentionally reuses the exact v0.4 coarse ontology.
    flan_coarse.validate_schema(dict(schema))


def allowed_labels_for(
    schema: Mapping[str, Any], field: str, selected_scope: str | None
) -> list[str]:
    return flan_coarse.allowed_labels_for(dict(schema), field, selected_scope)


def hierarchy_decision(
    field: str,
    labels: dict[str, str | None],
    deterministic: dict[str, Any],
) -> tuple[bool, str | None, str | None]:
    return flan_coarse.hierarchy_decision(field, labels, deterministic)


def prompt_variants(
    record: Mapping[str, Any],
    schema: Mapping[str, Any],
    field: str,
) -> list[tuple[str, dict[str, str], str | None]]:
    scopes: tuple[str | None, ...] = (
        ("common", "idiosyncratic")
        if field == "directional_alignment"
        else (None,)
    )
    variants: list[tuple[str, dict[str, str], str | None]] = []
    for scope in scopes:
        labels = allowed_labels_for(schema, field, scope)
        prompt, mapping = build_letter_prompt(
            record, schema, field, labels, scope
        )
        reverse_prompt, reverse_mapping = build_letter_prompt(
            record, schema, field, list(reversed(labels)), scope
        )
        variants.extend(
            (
                (prompt, mapping, scope),
                (reverse_prompt, reverse_mapping, scope),
            )
        )
    return variants


def validate_record_prompts(
    record: Mapping[str, Any], schema: Mapping[str, Any]
) -> None:
    for field in coarse.COARSE_FIELDS:
        for prompt, _, _ in prompt_variants(record, schema, field):
            if record["headline"] not in prompt or record["article_text"] not in prompt:
                raise ValueError(
                    f"Prompt omitted source content for {record['article_id']} / {field}"
                )
            if prompt.index("ARTICLE") > prompt.index("TASK"):
                raise ValueError(
                    f"Prompt is not article-first for {record['article_id']} / {field}"
                )


def validate_post_cutoff_record(record: Mapping[str, Any]) -> datetime:
    raw_timestamp = record.get("time_published_utc")
    if not isinstance(raw_timestamp, str) or not raw_timestamp.strip():
        raise ValueError("Input record has no time_published_utc timestamp")
    normalized = raw_timestamp.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"
    try:
        timestamp = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            f"Invalid time_published_utc timestamp {raw_timestamp!r}"
        ) from exc
    if timestamp.tzinfo is None:
        raise ValueError("time_published_utc must contain an explicit UTC offset")
    timestamp = timestamp.astimezone(timezone.utc)
    if timestamp.date() <= CONSERVATIVE_DATA_CUTOFF_DATE:
        raise ValueError(
            f"Article {record.get('article_id')!r} is not after the frozen "
            f"Llama 2 data cutoff {CONSERVATIVE_DATA_CUTOFF}"
        )
    return timestamp


def apply_chat_prompt_ids(
    tokenizer: Any,
    prompt: str,
) -> list[int]:
    ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}],
        chat_template=LLAMA_2_SINGLE_TURN_CHAT_TEMPLATE,
        tokenize=True,
        add_generation_prompt=True,
    )
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if ids and isinstance(ids[0], list):
        if len(ids) != 1:
            raise ValueError("Chat template unexpectedly returned a batch")
        ids = ids[0]
    if not isinstance(ids, list) or not ids or not all(isinstance(token, int) for token in ids):
        raise ValueError("Chat template did not return a nonempty token-id list")
    return ids


def candidate_letter_token_ids(
    tokenizer: Any,
    prompt: str,
    letters: Sequence[str],
) -> tuple[list[int], dict[str, int]]:
    """Resolve exact next-token IDs under the checkpoint's official chat template."""
    prompt_ids = apply_chat_prompt_ids(tokenizer, prompt)
    result: dict[str, int] = {}
    for letter in letters:
        if not re.fullmatch(r"[A-Z]", letter):
            raise ValueError(f"Invalid one-letter candidate {letter!r}")
        encoded = tokenizer(
            letter,
            add_special_tokens=False,
            truncation=False,
        )
        content_ids = encoded["input_ids"]
        if len(content_ids) != 1:
            raise ValueError(
                f"Candidate {letter!r} is not exactly one token under the pinned tokenizer"
            )
        result[letter] = content_ids[0]
    if len(set(result.values())) != len(result):
        raise ValueError("Candidate letters do not have unique next-token IDs")
    return prompt_ids, result


def score_map_from_next_token_log_probs(
    next_token_log_probs: Sequence[float],
    semantic_to_letter: Mapping[str, str],
    letter_to_token_id: Mapping[str, int],
) -> dict[str, float]:
    scores: dict[str, float] = {}
    for label, letter in semantic_to_letter.items():
        token_id = letter_to_token_id[letter]
        if token_id < 0 or token_id >= len(next_token_log_probs):
            raise ValueError(f"Candidate token id {token_id} is outside the vocabulary")
        scores[label] = float(next_token_log_probs[token_id])
    return scores


def mean_score_maps(
    first: Mapping[str, float],
    second: Mapping[str, float],
    labels: Sequence[str],
) -> dict[str, float]:
    return {label: (first[label] + second[label]) / 2.0 for label in labels}


def best_label(score_map: Mapping[str, float], labels: Sequence[str]) -> str:
    # Label/schema order is the explicit and reproducible tie-break.
    return max(labels, key=lambda label: score_map[label])


def score_margin(score_map: Mapping[str, float]) -> float:
    ordered = sorted(score_map.values(), reverse=True)
    return ordered[0] - ordered[1] if len(ordered) > 1 else 0.0


def _sha256_int_sequence(values: Sequence[int]) -> str:
    payload = json.dumps(list(values), separators=(",", ":"))
    return base.sha256_text(payload)


def encode_choice(
    *,
    tokenizer: Any,
    record: Mapping[str, Any],
    schema: Mapping[str, Any],
    field: str,
    labels_in_order: Sequence[str],
    selected_scope: str | None,
) -> dict[str, Any]:
    prompt, mapping = build_letter_prompt(
        record, schema, field, labels_in_order, selected_scope
    )
    letters = [mapping[label] for label in labels_in_order]
    input_ids, letter_token_ids = candidate_letter_token_ids(
        tokenizer, prompt, letters
    )
    return {
        "prompt": prompt,
        "prompt_sha256": base.sha256_text(prompt),
        "input_ids": input_ids,
        "chat_input_ids_sha256": _sha256_int_sequence(input_ids),
        "semantic_to_letter": mapping,
        "letter_to_token_id": letter_token_ids,
    }


def preflight_prompt_lengths(
    records: Sequence[Mapping[str, Any]],
    schema: Mapping[str, Any],
    tokenizer: Any,
    max_input_tokens: int,
) -> dict[str, int]:
    maxima: dict[str, int] = {}
    violations: list[dict[str, Any]] = []
    for field in coarse.COARSE_FIELDS:
        maximum = 0
        scopes: tuple[str | None, ...] = (
            ("common", "idiosyncratic")
            if field == "directional_alignment"
            else (None,)
        )
        for record in records:
            for scope in scopes:
                labels = allowed_labels_for(schema, field, scope)
                for ordered_labels in (labels, list(reversed(labels))):
                    encoded = encode_choice(
                        tokenizer=tokenizer,
                        record=record,
                        schema=schema,
                        field=field,
                        labels_in_order=ordered_labels,
                        selected_scope=scope,
                    )
                    count = len(encoded["input_ids"])
                    maximum = max(maximum, count)
                    if count > max_input_tokens:
                        violations.append(
                            {
                                "article_id": record["article_id"],
                                "field": field,
                                "tokens": count,
                            }
                        )
        maxima[field] = maximum
    if violations:
        raise RuntimeError(
            "Prompt preflight failed; refusing to truncate article text or the "
            "classification contract. Increase --max-input-tokens or apply a "
            "separately documented preprocessing rule: "
            + json.dumps(violations[:20], ensure_ascii=False)
        )
    return maxima


def _score_encoded_batch(
    encoded_choices: Sequence[Mapping[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: str,
    max_input_tokens: int,
) -> list[dict[str, float]]:
    if not encoded_choices:
        return []
    lengths = [len(choice["input_ids"]) for choice in encoded_choices]
    if any(length > max_input_tokens for length in lengths):
        raise RuntimeError("Refusing to truncate an encoded chat prompt")
    maximum = max(lengths)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Pinned tokenizer has neither pad_token_id nor eos_token_id")

    input_ids = torch.full(
        (len(encoded_choices), maximum),
        pad_token_id,
        dtype=torch.long,
        device=device,
    )
    attention_mask = torch.zeros(
        (len(encoded_choices), maximum),
        dtype=torch.long,
        device=device,
    )
    for row, choice in enumerate(encoded_choices):
        ids = torch.tensor(choice["input_ids"], dtype=torch.long, device=device)
        input_ids[row, : len(ids)] = ids
        attention_mask[row, : len(ids)] = 1

    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits
    results: list[dict[str, float]] = []
    for row, (choice, length) in enumerate(zip(encoded_choices, lengths)):
        next_log_probs = torch.log_softmax(logits[row, length - 1].float(), dim=-1)
        scores = {
            label: float(
                next_log_probs[choice["letter_to_token_id"][letter]].detach().cpu()
            )
            for label, letter in choice["semantic_to_letter"].items()
        }
        results.append(scores)
    return results


def classify_active_batch(
    *,
    field: str,
    active_states: list[dict[str, Any]],
    schema: Mapping[str, Any],
    tokenizer: Any,
    model: Any,
    torch: Any,
    device: str,
    max_input_tokens: int,
    _allowed_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    scopes = [state["labels"].get("shock_scope") for state in active_states]
    if _allowed_labels is None:
        groups: dict[tuple[str, ...], list[int]] = {}
        for index, scope in enumerate(scopes):
            key = tuple(allowed_labels_for(schema, field, scope))
            groups.setdefault(key, []).append(index)
        if len(groups) > 1:
            restored: list[dict[str, Any] | None] = [None] * len(active_states)
            for label_tuple, indices in groups.items():
                subset = [active_states[index] for index in indices]
                subset_results = classify_active_batch(
                    field=field,
                    active_states=subset,
                    schema=schema,
                    tokenizer=tokenizer,
                    model=model,
                    torch=torch,
                    device=device,
                    max_input_tokens=max_input_tokens,
                    _allowed_labels=list(label_tuple),
                )
                for index, result in zip(indices, subset_results):
                    restored[index] = result
            if any(result is None for result in restored):
                raise RuntimeError("Failed to restore conditional-label batch order")
            return [result for result in restored if result is not None]

    labels = _allowed_labels or allowed_labels_for(schema, field, scopes[0])
    canonical = [
        encode_choice(
            tokenizer=tokenizer,
            record=state["record"],
            schema=schema,
            field=field,
            labels_in_order=labels,
            selected_scope=scope,
        )
        for state, scope in zip(active_states, scopes)
    ]
    reversed_choices = [
        encode_choice(
            tokenizer=tokenizer,
            record=state["record"],
            schema=schema,
            field=field,
            labels_in_order=list(reversed(labels)),
            selected_scope=scope,
        )
        for state, scope in zip(active_states, scopes)
    ]
    canonical_scores = _score_encoded_batch(
        canonical,
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        device=device,
        max_input_tokens=max_input_tokens,
    )
    reversed_scores = _score_encoded_batch(
        reversed_choices,
        model=model,
        tokenizer=tokenizer,
        torch=torch,
        device=device,
        max_input_tokens=max_input_tokens,
    )

    results: list[dict[str, Any]] = []
    for index in range(len(active_states)):
        averaged = mean_score_maps(
            canonical_scores[index], reversed_scores[index], labels
        )
        value = best_label(averaged, labels)
        results.append(
            {
                "value": value,
                "raw_output": value,
                "origin": "model",
                "prompt_sha256": canonical[index]["prompt_sha256"],
                "reversed_prompt_sha256": reversed_choices[index]["prompt_sha256"],
                "chat_input_ids_sha256": canonical[index]["chat_input_ids_sha256"],
                "reversed_chat_input_ids_sha256": reversed_choices[index][
                    "chat_input_ids_sha256"
                ],
                "input_truncated": False,
                "canonical_candidate_log_probabilities": canonical_scores[index],
                "reversed_candidate_log_probabilities": reversed_scores[index],
                "order_averaged_mean_log_probabilities": averaged,
                "canonical_semantic_to_letter": canonical[index][
                    "semantic_to_letter"
                ],
                "reversed_semantic_to_letter": reversed_choices[index][
                    "semantic_to_letter"
                ],
                "canonical_letter_token_ids": canonical[index][
                    "letter_to_token_id"
                ],
                "reversed_letter_token_ids": reversed_choices[index][
                    "letter_to_token_id"
                ],
                "canonical_prediction": best_label(
                    canonical_scores[index], labels
                ),
                "reversed_prediction": best_label(
                    reversed_scores[index], labels
                ),
                "top1_top2_margin": score_margin(averaged),
            }
        )
    return results


def resolve_device_and_dtype(
    torch: Any,
    device_argument: str,
    precision: str,
    quantization: str,
) -> tuple[str, Any]:
    device = (
        "cuda"
        if device_argument == "auto" and torch.cuda.is_available()
        else "cpu"
        if device_argument == "auto"
        else device_argument
    )
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if quantization == "nf4" and device != "cuda":
        raise RuntimeError("NF4 inference requires CUDA")
    if precision == "auto":
        dtype = torch.float16 if device == "cuda" else torch.float32
    else:
        dtype = getattr(torch, precision)
    if device == "cpu" and dtype == torch.float16:
        raise RuntimeError("float16 CPU inference is unsupported")
    return device, dtype


def snapshot_file_hashes(snapshot_path: Path) -> dict[str, str]:
    files = sorted(
        path
        for path in snapshot_path.rglob("*")
        if path.is_file()
    )
    relative_names = {
        path.relative_to(snapshot_path).as_posix(): path for path in files
    }
    required = ("config.json", "tokenizer_config.json")
    missing = [name for name in required if name not in relative_names]
    if missing:
        raise FileNotFoundError(f"Pinned snapshot is missing required files: {missing}")
    if not {"tokenizer.json", "tokenizer.model"} & set(relative_names):
        raise FileNotFoundError("Pinned snapshot contains no tokenizer vocabulary")
    if not any(name.endswith(".safetensors") for name in relative_names):
        raise FileNotFoundError("Pinned snapshot contains no safetensors model weights")
    return {
        name: base.sha256_file(path)
        for name, path in sorted(relative_names.items())
    }


def validate_nf4_device_map(model: Any) -> dict[str, Any]:
    """Reject silent CPU/disk offload in the frozen 8 GB GPU configuration."""

    device_map = getattr(model, "hf_device_map", None)
    if not isinstance(device_map, dict) or not device_map:
        raise RuntimeError("NF4 model did not expose a nonempty hf_device_map")
    invalid = {
        name: value
        for name, value in device_map.items()
        if str(value).lower() not in {"0", "cuda", "cuda:0"}
    }
    if invalid:
        raise RuntimeError(
            "Frozen NF4 extraction forbids CPU or disk offload; invalid "
            f"hf_device_map entries: {invalid}"
        )
    return dict(device_map)


def _canonical_json_sha256(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    schema_text = args.schema.read_text(encoding="utf-8")
    schema = json.loads(schema_text)
    validate_schema(schema)
    all_records = base.read_jsonl(args.input)
    if not all_records:
        raise ValueError("Input contains no records")
    article_ids = [record.get("article_id") for record in all_records]
    if len(article_ids) != len(set(article_ids)):
        raise ValueError("Input contains duplicate article_id values")
    records = (
        all_records[: args.limit]
        if args.limit is not None
        else all_records
    )
    publication_times: list[datetime] = []
    for record in records:
        base.validate_input_record(record)
        coarse.validate_input_record(record)
        publication_times.append(validate_post_cutoff_record(record))
        validate_record_prompts(record, schema)

    deterministic_by_id = {
        record["article_id"]: coarse.deterministic_features(record)
        for record in records
    }
    if args.validate_only:
        routes: dict[str, int] = {}
        for features in deterministic_by_id.values():
            route = features["gate_route"]
            routes[route] = routes.get(route, 0) + 1
        print(
            json.dumps(
                {
                    "validated_input_records": len(records),
                    "prompt_version": PROMPT_VERSION,
                    "model_id": args.model_id,
                    "model_revision": args.revision,
                    "conservative_model_data_cutoff": CONSERVATIVE_DATA_CUTOFF,
                    "minimum_time_published_utc": min(publication_times).isoformat(),
                    "maximum_time_published_utc": max(publication_times).isoformat(),
                    "decoding": DECODING_METHOD,
                    "coarse_fields": list(coarse.COARSE_FIELDS),
                    "deterministic_relevance_routes": dict(sorted(routes.items())),
                    "model_packages_imported": False,
                    "model_was_loaded": False,
                },
                indent=2,
            )
        )
        return 0

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if args.local_files_only:
        os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    try:
        import torch
        import transformers
        from huggingface_hub import snapshot_download
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
            set_seed,
        )
    except ImportError as exc:
        raise RuntimeError(
            "Missing local model dependencies. Install the documented Llama "
            "environment before running extraction."
        ) from exc

    device, dtype = resolve_device_and_dtype(
        torch, args.device, args.precision, args.quantization
    )
    base.configure_determinism(torch, set_seed)
    snapshot_path = Path(
        snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            local_files_only=args.local_files_only,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model*.safetensors*",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "tokenizer.model",
            ],
        )
    )
    model_file_hashes = snapshot_file_hashes(snapshot_path)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token
    prompt_maxima = preflight_prompt_lengths(
        records, schema, tokenizer, args.max_input_tokens
    )

    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "use_safetensors": True,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
    }
    if args.quantization == "nf4":
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        )
        model_kwargs["device_map"] = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(snapshot_path, **model_kwargs)
    if args.quantization == "none":
        model.to(device)
        frozen_device_map = None
    else:
        frozen_device_map = validate_nf4_device_map(model)
    model_context = int(getattr(model.config, "max_position_embeddings", 0))
    if model_context < args.max_input_tokens:
        raise RuntimeError(
            f"Pinned model context is {model_context}, below --max-input-tokens "
            f"{args.max_input_tokens}"
        )
    model.eval()
    model.config.use_cache = False

    prompt_code = "\n".join(
        inspect.getsource(component)
        for component in (
            article_first_context,
            target_context,
            field_task,
            build_letter_prompt,
            candidate_letter_token_ids,
            hierarchy_decision,
            classify_active_batch,
        )
    )
    coarse_module_path = Path(coarse.__file__).resolve()
    hierarchy_module_path = Path(flan_coarse.__file__).resolve()
    chat_template_payload = LLAMA_2_SINGLE_TURN_CHAT_TEMPLATE
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "status": "in_progress",
        "prompt_version": PROMPT_VERSION,
        "extractor_source_sha256": base.sha256_file(Path(__file__).resolve()),
        "prompt_builder_sha256": base.sha256_text(prompt_code),
        "deterministic_rule_version": coarse.DETERMINISTIC_RULE_VERSION,
        "deterministic_rules_payload_sha256": coarse.DETERMINISTIC_RULES_SHA256,
        "deterministic_module_sha256": base.sha256_file(coarse_module_path),
        "hierarchy_module_sha256": base.sha256_file(hierarchy_module_path),
        "schema_name": schema["schema_name"],
        "schema_version": schema["schema_version"],
        "schema_sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
        "input_path": str(args.input),
        "input_sha256": base.sha256_file(args.input),
        "selected_article_ids_sha256": base.sha256_text(
            "\n".join(record["article_id"] for record in records)
        ),
        "model_id": args.model_id,
        "model_revision": args.revision,
        "tokenizer_revision": args.revision,
        "conservative_model_data_cutoff": CONSERVATIVE_DATA_CUTOFF,
        "minimum_time_published_utc": min(publication_times).isoformat(),
        "maximum_time_published_utc": max(publication_times).isoformat(),
        "model_context_window": MODEL_CONTEXT_WINDOW,
        "model_snapshot_path": str(snapshot_path),
        "model_files_sha256": model_file_hashes,
        "coarse_fields": list(coarse.COARSE_FIELDS),
        "decoding": DECODING_METHOD,
        "prompt_profile": PROMPT_PROFILE,
        "chat_template_sha256": _canonical_json_sha256(chat_template_payload),
        "chat_template_source": "frozen official Llama 2 single-turn [INST] format",
        "hierarchy": {
            "irrelevant_gate": "skip_all_semantic_fields",
            "idiosyncratic_single_firm": (
                "derive_single_firm_only_unless_target_and_peer_are_explicit"
            ),
            "unclear_scope": "derive_unclear_alignment",
            "relational_alignment": (
                "run_model_for_common_mixed_or_explicit_target_plus_peer"
            ),
        },
        "device": device,
        "precision": str(dtype).replace("torch.", ""),
        "quantization": {
            "method": args.quantization,
            "bnb_4bit_quant_type": "nf4" if args.quantization == "nf4" else None,
            "bnb_4bit_use_double_quant": (
                True if args.quantization == "nf4" else None
            ),
            "compute_dtype": str(dtype).replace("torch.", ""),
            "hf_device_map": frozen_device_map,
        },
        "attention_implementation": "eager",
        "batch_size": args.batch_size,
        "max_input_tokens": args.max_input_tokens,
        "preflight_max_prompt_tokens_by_field": prompt_maxima,
        "generation": {
            "performed": False,
            "do_sample": False,
            "seed": 0,
            "score": "next_token_log_probability_for_single_letter_choice",
            "choice_order_averaging": True,
            "option_orders": ["schema_order", "reversed_schema_order"],
            "tie_break": "schema_order",
        },
        "attention_implementation": "eager",
        "runtime": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(0) if device == "cuda" else device
            ),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms_enforced": True,
            "attention_implementation": "eager",
        },
        "total_prediction_count": len(records),
    }
    base.write_manifest(manifest_path, manifest)

    results: list[dict[str, Any]] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        states: list[dict[str, Any]] = []
        for record in batch:
            deterministic = deterministic_by_id[record["article_id"]]
            states.append(
                {
                    "record": record,
                    "deterministic": deterministic,
                    "labels": {field: None for field in coarse.COARSE_FIELDS},
                    "origins": {},
                    "passes": {},
                }
            )

        for field in coarse.COARSE_FIELDS:
            active_states: list[dict[str, Any]] = []
            for state in states:
                run_model, derived, reason = hierarchy_decision(
                    field, state["labels"], state["deterministic"]
                )
                if run_model:
                    active_states.append(state)
                    continue
                state["labels"][field] = derived
                origin = (
                    "deterministic_gate"
                    if reason == "deterministic_relevance_gate"
                    else "hierarchical_derivation"
                )
                state["origins"][field] = origin
                state["passes"][field] = {
                    "origin": origin,
                    "value": derived,
                    "reason": reason,
                    "valid": True,
                    "input_truncated": False,
                }

            if active_states:
                classified = classify_active_batch(
                    field=field,
                    active_states=active_states,
                    schema=schema,
                    tokenizer=tokenizer,
                    model=model,
                    torch=torch,
                    device=device,
                    max_input_tokens=args.max_input_tokens,
                )
                for state, classification in zip(active_states, classified):
                    value = classification.pop("value")
                    if value not in schema["closed_label_fields"][field]:
                        raise RuntimeError(
                            f"Decoder produced invalid {field} value {value!r}"
                        )
                    state["labels"][field] = value
                    state["origins"][field] = "model"
                    state["passes"][field] = {
                        **classification,
                        "value": value,
                        "valid": True,
                    }

        for state in states:
            applicable = bool(state["deterministic"]["semantic_applicable"])
            results.append(
                {
                    "row_number": state["record"]["row_number"],
                    "article_id": state["record"]["article_id"],
                    "target_ticker": state["record"]["target"]["ticker"],
                    "extractor": args.model_id,
                    "model_revision": args.revision,
                    "prompt_version": PROMPT_VERSION,
                    "protocol_version": schema["schema_version"],
                    "semantic_applicable": applicable,
                    "deterministic_features": state["deterministic"],
                    "labels": state["labels"],
                    "label_origins": state["origins"],
                    "passes": state["passes"],
                    "validity": {
                        "all_fields_resolved_when_applicable": (
                            all(
                                state["labels"][field] is not None
                                for field in coarse.COARSE_FIELDS
                            )
                            if applicable
                            else True
                        ),
                        "no_input_truncation": not any(
                            pass_record.get("input_truncated")
                            for pass_record in state["passes"].values()
                        ),
                    },
                }
            )
        print(
            f"Processed {min(start + len(batch), len(records))}/{len(records)} records",
            file=sys.stderr,
            flush=True,
        )

    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(
                json.dumps(
                    result, ensure_ascii=False, separators=(",", ":")
                )
                + "\n"
            )
    manifest.update(
        {
            "status": "complete",
            "semantic_applicable_count": sum(
                int(result["semantic_applicable"]) for result in results
            ),
            "model_field_call_count": sum(
                int(origin == "model")
                for result in results
                for origin in result["label_origins"].values()
            ),
            "output_sha256": base.sha256_file(args.output),
        }
    )
    base.write_manifest(manifest_path, manifest)
    print(f"Wrote {len(results)} predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
