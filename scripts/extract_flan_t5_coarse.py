from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import sys
from pathlib import Path
from typing import Any

import coarse_news_features as coarse
import extract_flan_t5 as base


MODEL_DEFAULT = "google/flan-t5-large"
PROMPT_VERSION = "flan-stock-sector-news-v0.4.0"
PROMPT_PROFILE_DEFAULT = "zero_shot"
PROMPT_PROFILES = ("zero_shot", "few_shot")
DECODING_METHODS = (
    "constrained",
    "label_score",
    "letter_score",
    "order_averaged_letter_score",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run frozen FLAN-T5-Large with the v0.3 coarse, hierarchical financial-news protocol."
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
        required=True,
        help="Immutable Hugging Face commit hash; branches and tags are rejected.",
    )
    parser.add_argument(
        "--decoding",
        choices=DECODING_METHODS,
        default="order_averaged_letter_score",
    )
    parser.add_argument(
        "--prompt-profile",
        choices=PROMPT_PROFILES,
        default=PROMPT_PROFILE_DEFAULT,
        help="few_shot adds one short, synthetic example per legal class.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate records, deterministic features, prompts, and schema without loading FLAN.",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", args.revision):
        parser.error("--revision must be an immutable hexadecimal commit hash")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not 128 <= args.max_input_tokens <= 512:
        parser.error("--max-input-tokens must be between 128 and 512")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def article_first_context(record: dict[str, Any]) -> str:
    return f"ARTICLE\nHeadline: {record['headline']}\nText: {record['article_text']}"


def target_context(record: dict[str, Any]) -> str:
    target = record["target"]
    return (
        f"Target company: {target['company']} ({target['ticker']})\n"
        f"Target sector: {target['sector']}"
    )


def field_definitions(schema: dict[str, Any], field: str) -> dict[str, str]:
    definitions = schema["field_definitions"][field]
    allowed = schema["closed_label_fields"][field]
    if set(definitions) != set(allowed):
        raise ValueError(f"Schema definitions and labels differ for {field}")
    return definitions


FEW_SHOT_EXAMPLES = {
    "shock_scope": {
        "idiosyncratic": "AMD launched a new accelerator.",
        "common": "A new export rule applies to AMD, Nvidia, and Intel.",
        "mixed": "AMD disclosed a firm flaw while a separate export rule covered all chipmakers.",
        "unclear": "Technology update.",
    },
    "event_family": {
        "earnings_guidance": "The company reported earnings and raised its outlook.",
        "product_demand": "The company launched a processor and won a customer order.",
        "supply_capacity": "A factory outage reduced production capacity.",
        "regulation_legal": "A regulator imposed a new export restriction.",
        "corporate_analyst": "An analyst upgraded the company.",
        "macro_market": "The Federal Reserve changed interest rates.",
        "other_or_unclear": "The company published an update without details.",
    },
    "information_status": {
        "confirmed": "The company officially announced the completed launch.",
        "anticipated": "The company is expected to launch next month.",
        "rumor_or_opinion": "An analyst reportedly believes a deal may occur.",
        "unclear": "A possible update was mentioned without further detail.",
    },
    "directional_alignment": {
        "single_firm_only": "AMD announced a company-only product update.",
        "same_direction": "One rule restricts sales by AMD, Nvidia, and Intel.",
        "opposite_direction": "A customer shifted orders from Nvidia to AMD.",
        "common_direction_unclear": "A standard applies to AMD and its peers, but no effect is stated.",
        "unclear": "The article does not establish a relative effect.",
    },
}


def few_shot_block(field: str, allowed_labels: list[str], prompt_profile: str) -> str:
    if prompt_profile == "zero_shot":
        return ""
    if prompt_profile != "few_shot":
        raise ValueError(f"Unknown prompt profile {prompt_profile!r}")
    examples = FEW_SHOT_EXAMPLES[field]
    lines = ["EXAMPLES"]
    for label in allowed_labels:
        lines.append(f"{examples[label]} => {label}")
    return "\n".join(lines) + "\n\n"


def prompt_preamble(record: dict[str, Any], field: str, selected_scope: str | None) -> str:
    article = article_first_context(record)
    if field == "shock_scope":
        question = (
            f"{target_context(record)}\n\n"
            "Classify the primary scope of the information shock. Company membership in a sector alone does "
            "not make a story common. Use only the article; do not use later outcomes or external facts."
        )
    elif field == "event_family":
        question = (
            "Classify the article's primary event family. Use only the article; do not use later outcomes or "
            "external facts."
        )
    elif field == "information_status":
        question = (
            "Classify the status of the primary information. Use only the article; do not use later outcomes "
            "or external facts."
        )
    elif field == "directional_alignment":
        peers = ", ".join(record["target"]["known_sector_peers"]) or "none supplied"
        fallback = (
            "If the article does not state whether the effects align, choose common_direction_unclear."
            if selected_scope in {"common", "mixed"}
            else "If no same/opposite relationship is explicit, choose single_firm_only."
        )
        question = (
            f"{target_context(record)}\nKnown sector peers: {peers}\n"
            f"Predicted shock scope: {selected_scope}\n\n"
            "Classify only the direction explicitly stated for the target versus its sector or peers. Do not "
            f"predict returns. {fallback}"
        )
    else:
        raise ValueError(f"Unknown coarse field {field!r}")
    return f"{article}\n\nTASK\n{question}"


def build_natural_prompt(
    record: dict[str, Any],
    schema: dict[str, Any],
    field: str,
    selected_scope: str | None = None,
    allowed_labels: list[str] | None = None,
    prompt_profile: str = PROMPT_PROFILE_DEFAULT,
) -> str:
    definitions = field_definitions(schema, field)
    labels = allowed_labels or allowed_labels_for(schema, field, selected_scope)
    options = "\n".join(f"- {label}: {definitions[label]}" for label in labels)
    return (
        f"{prompt_preamble(record, field, selected_scope)}\n\n"
        f"{few_shot_block(field, labels, prompt_profile)}"
        f"OPTIONS\n{options}\n\n"
        "Return exactly one option label and nothing else.\nAnswer:"
    )


def label_letter_mapping(labels_in_order: list[str]) -> dict[str, str]:
    if len(labels_in_order) > 26:
        raise ValueError("Letter-choice protocol supports at most 26 labels")
    return {label: chr(ord("A") + index) for index, label in enumerate(labels_in_order)}


def build_letter_prompt(
    record: dict[str, Any],
    schema: dict[str, Any],
    field: str,
    labels_in_order: list[str],
    selected_scope: str | None = None,
    prompt_profile: str = PROMPT_PROFILE_DEFAULT,
) -> tuple[str, dict[str, str]]:
    definitions = field_definitions(schema, field)
    mapping = label_letter_mapping(labels_in_order)
    options = "\n".join(
        f"{mapping[label]}. {label}: {definitions[label]}" for label in labels_in_order
    )
    prompt = (
        f"{prompt_preamble(record, field, selected_scope)}\n\n"
        f"{few_shot_block(field, labels_in_order, prompt_profile)}"
        f"OPTIONS\n{options}\n\n"
        "Answer with exactly one option letter and nothing else.\nAnswer:"
    )
    return prompt, mapping


def allowed_labels_for(
    schema: dict[str, Any], field: str, selected_scope: str | None
) -> list[str]:
    labels = list(schema["closed_label_fields"][field])
    if field != "directional_alignment":
        return labels
    if selected_scope in {"common", "mixed"}:
        return ["same_direction", "opposite_direction", "common_direction_unclear"]
    if selected_scope == "idiosyncratic":
        return ["same_direction", "opposite_direction", "single_firm_only"]
    raise ValueError(f"directional_alignment must not run for scope {selected_scope!r}")


def hierarchy_decision(
    field: str,
    labels: dict[str, str | None],
    deterministic: dict[str, Any],
) -> tuple[bool, str | None, str | None]:
    """Return (run_model, derived_value, reason)."""
    if not deterministic["semantic_applicable"]:
        return False, None, "deterministic_relevance_gate"
    if field != "directional_alignment":
        return True, None, None
    scope = labels.get("shock_scope")
    if scope == "unclear" or scope is None:
        return False, "unclear", "scope_unclear"
    if scope == "idiosyncratic":
        target_mentioned = bool(deterministic.get("text_target_mentioned"))
        peer_mentioned = int(deterministic.get("text_peer_count", 0)) > 0
        # Only ask a relational question when both sides are explicit. Otherwise the
        # coarse ontology intentionally records a one-firm information shock.
        if target_mentioned and peer_mentioned:
            return True, None, None
        return False, "single_firm_only", "single_firm_idiosyncratic"
    if scope in {"common", "mixed"}:
        return True, None, None
    raise ValueError(f"Unsupported shock_scope {scope!r}")


def validate_schema(schema: dict[str, Any]) -> None:
    if tuple(schema.get("flan_core_fields", ())) != tuple(coarse.COARSE_FIELDS):
        raise ValueError("Coarse schema flan_core_fields do not match the versioned protocol")
    for field in coarse.COARSE_FIELDS:
        allowed = schema["closed_label_fields"].get(field)
        if not isinstance(allowed, list) or len(allowed) < 2 or len(allowed) != len(set(allowed)):
            raise ValueError(f"Invalid closed-label definition for {field}")
        if not all(isinstance(value, str) and re.fullmatch(r"[a-z][a-z0-9_]*", value) for value in allowed):
            raise ValueError(f"Invalid label token in {field}")
        field_definitions(schema, field)


def prompt_variants_for_preflight(
    record: dict[str, Any], schema: dict[str, Any], field: str, prompt_profile: str
) -> list[str]:
    scopes = ("common", "idiosyncratic") if field == "directional_alignment" else (None,)
    prompts: list[str] = []
    for scope in scopes:
        labels = allowed_labels_for(schema, field, scope)
        prompts.append(
            build_natural_prompt(record, schema, field, scope, labels, prompt_profile)
        )
        canonical, _ = build_letter_prompt(
            record, schema, field, labels, scope, prompt_profile
        )
        reversed_prompt, _ = build_letter_prompt(
            record, schema, field, list(reversed(labels)), scope, prompt_profile
        )
        prompts.extend((canonical, reversed_prompt))
    return prompts


def preflight_prompt_lengths(
    records: list[dict[str, Any]],
    schema: dict[str, Any],
    tokenizer: Any,
    max_tokens: int,
    prompt_profile: str,
) -> dict[str, int]:
    maxima: dict[str, int] = {}
    violations: list[dict[str, Any]] = []
    for field in coarse.COARSE_FIELDS:
        maximum = 0
        for record in records:
            for prompt in prompt_variants_for_preflight(
                record, schema, field, prompt_profile
            ):
                count = len(tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"])
                maximum = max(maximum, count)
                if count > max_tokens:
                    violations.append(
                        {"article_id": record["article_id"], "field": field, "tokens": count}
                    )
        maxima[field] = maximum
    if violations:
        raise RuntimeError(
            "Prompt preflight failed; v0.3 never truncates article inputs: "
            + json.dumps(violations[:20], ensure_ascii=False)
        )
    return maxima


def mean_score_maps(
    first: dict[str, float], second: dict[str, float], labels: list[str]
) -> dict[str, float]:
    return {label: (first[label] + second[label]) / 2.0 for label in labels}


def best_label(score_map: dict[str, float], labels: list[str]) -> str:
    # Schema order is the explicit, reproducible tie-break.
    return max(labels, key=lambda label: score_map[label])


def score_margin(score_map: dict[str, float]) -> float:
    ordered = sorted(score_map.values(), reverse=True)
    return ordered[0] - ordered[1] if len(ordered) > 1 else 0.0


def classify_active_batch(
    *,
    field: str,
    active_states: list[dict[str, Any]],
    schema: dict[str, Any],
    decoding: str,
    tokenizer: Any,
    model: Any,
    torch: Any,
    device: str,
    max_input_tokens: int,
    prompt_profile: str,
    _allowed_labels: list[str] | None = None,
) -> list[dict[str, Any]]:
    scopes = [state["labels"].get("shock_scope") for state in active_states]
    if _allowed_labels is None:
        groups: dict[tuple[str, ...], list[int]] = {}
        for index, scope in enumerate(scopes):
            key = tuple(allowed_labels_for(schema, field, scope))
            groups.setdefault(key, []).append(index)
        if len(groups) > 1:
            ordered_results: list[dict[str, Any] | None] = [None] * len(active_states)
            for label_tuple, indices in groups.items():
                subset = [active_states[index] for index in indices]
                subset_results = classify_active_batch(
                    field=field,
                    active_states=subset,
                    schema=schema,
                    decoding=decoding,
                    tokenizer=tokenizer,
                    model=model,
                    torch=torch,
                    device=device,
                    max_input_tokens=max_input_tokens,
                    prompt_profile=prompt_profile,
                    _allowed_labels=list(label_tuple),
                )
                for index, result in zip(indices, subset_results):
                    ordered_results[index] = result
            if any(result is None for result in ordered_results):
                raise RuntimeError("Internal error while restoring conditional-label batch order")
            return [result for result in ordered_results if result is not None]
    labels = _allowed_labels or allowed_labels_for(schema, field, scopes[0])
    if decoding == "constrained":
        prompts = [
            build_natural_prompt(
                state["record"], schema, field, scope, labels, prompt_profile
            )
            for state, scope in zip(active_states, scopes)
        ]
        selected, truncated = base.generate_constrained_label_batch(
            prompts, labels, tokenizer, model, torch, device, max_input_tokens
        )
        return [
            {
                "value": value,
                "raw_output": value,
                "prompt_sha256": base.sha256_text(prompt),
                "input_truncated": was_truncated,
                "origin": "model",
            }
            for value, prompt, was_truncated in zip(selected, prompts, truncated)
        ]

    if decoding == "label_score":
        prompts = [
            build_natural_prompt(
                state["record"], schema, field, scope, labels, prompt_profile
            )
            for state, scope in zip(active_states, scopes)
        ]
        selected, truncated, scores = base.score_closed_label_batch(
            prompts,
            labels,
            labels,
            tokenizer,
            model,
            torch,
            device,
            max_input_tokens,
        )
        return [
            {
                "value": value,
                "raw_output": value,
                "prompt_sha256": base.sha256_text(prompt),
                "input_truncated": was_truncated,
                "origin": "model",
                "candidate_mean_log_probabilities": score_map,
                "top1_top2_margin": score_margin(score_map),
            }
            for value, prompt, was_truncated, score_map in zip(
                selected, prompts, truncated, scores
            )
        ]

    canonical_prompts: list[str] = []
    canonical_outputs: list[str] = []
    reversed_prompts: list[str] = []
    reversed_outputs: list[str] = []
    for state, scope in zip(active_states, scopes):
        prompt, mapping = build_letter_prompt(
            state["record"], schema, field, labels, scope, prompt_profile
        )
        reverse_prompt, reverse_mapping = build_letter_prompt(
            state["record"],
            schema,
            field,
            list(reversed(labels)),
            scope,
            prompt_profile,
        )
        canonical_prompts.append(prompt)
        canonical_outputs.append(mapping[labels[0]])  # length placeholder; values supplied below
        reversed_prompts.append(reverse_prompt)
        reversed_outputs.append(reverse_mapping[labels[0]])

    canonical_mapping = label_letter_mapping(labels)
    reversed_mapping = label_letter_mapping(list(reversed(labels)))
    canonical_candidates = [canonical_mapping[label] for label in labels]
    reversed_candidates = [reversed_mapping[label] for label in labels]
    canonical_selected, canonical_truncated, canonical_scores = base.score_closed_label_batch(
        canonical_prompts,
        labels,
        canonical_candidates,
        tokenizer,
        model,
        torch,
        device,
        max_input_tokens,
    )
    if decoding == "letter_score":
        return [
            {
                "value": value,
                "raw_output": canonical_mapping[value],
                "prompt_sha256": base.sha256_text(prompt),
                "input_truncated": was_truncated,
                "origin": "model",
                "candidate_mean_log_probabilities": score_map,
                "semantic_to_letter": canonical_mapping,
                "top1_top2_margin": score_margin(score_map),
            }
            for value, prompt, was_truncated, score_map in zip(
                canonical_selected, canonical_prompts, canonical_truncated, canonical_scores
            )
        ]

    _, reversed_truncated, reversed_scores = base.score_closed_label_batch(
        reversed_prompts,
        labels,
        reversed_candidates,
        tokenizer,
        model,
        torch,
        device,
        max_input_tokens,
    )
    results: list[dict[str, Any]] = []
    for index, state in enumerate(active_states):
        averaged = mean_score_maps(canonical_scores[index], reversed_scores[index], labels)
        value = best_label(averaged, labels)
        results.append(
            {
                "value": value,
                "raw_output": value,
                "prompt_sha256": base.sha256_text(canonical_prompts[index]),
                "reversed_prompt_sha256": base.sha256_text(reversed_prompts[index]),
                "input_truncated": canonical_truncated[index] or reversed_truncated[index],
                "origin": "model",
                "canonical_candidate_mean_log_probabilities": canonical_scores[index],
                "reversed_candidate_mean_log_probabilities": reversed_scores[index],
                "order_averaged_mean_log_probabilities": averaged,
                "canonical_semantic_to_letter": canonical_mapping,
                "reversed_semantic_to_letter": reversed_mapping,
                "canonical_prediction": canonical_selected[index],
                "reversed_prediction": best_label(reversed_scores[index], labels),
                "top1_top2_margin": score_margin(averaged),
            }
        )
    return results


def validate_record_prompts(
    record: dict[str, Any], schema: dict[str, Any], prompt_profile: str
) -> None:
    for field in coarse.COARSE_FIELDS:
        for prompt in prompt_variants_for_preflight(record, schema, field, prompt_profile):
            if record["headline"] not in prompt or record["article_text"] not in prompt:
                raise ValueError(f"Prompt omitted source content for {record['article_id']} / {field}")


def main() -> int:
    args = parse_args()
    schema_text = args.schema.read_text(encoding="utf-8")
    schema = json.loads(schema_text)
    validate_schema(schema)
    records = base.read_jsonl(args.input)
    for record in records:
        base.validate_input_record(record)
        validate_record_prompts(record, schema, args.prompt_profile)
    ids = [record["article_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Input contains duplicate article_id values")
    if args.limit is not None:
        records = records[: args.limit]

    deterministic_by_id = {
        record["article_id"]: coarse.deterministic_features(record) for record in records
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
                    "decoding": args.decoding,
                    "prompt_profile": args.prompt_profile,
                    "coarse_fields": list(coarse.COARSE_FIELDS),
                    "deterministic_relevance_routes": dict(sorted(routes.items())),
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
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, set_seed
    except ImportError as exc:
        raise RuntimeError(
            "Missing model dependencies. Install requirements-llm.txt in the project environment."
        ) from exc

    device, dtype = base.resolve_device_and_dtype(torch, args.device, args.precision)
    base.configure_determinism(torch, set_seed)
    snapshot_path = Path(
        snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            local_files_only=args.local_files_only,
            allow_patterns=[
                "config.json",
                "generation_config.json",
                "model.safetensors",
                "special_tokens_map.json",
                "spiece.model",
                "tokenizer.json",
                "tokenizer_config.json",
            ],
        )
    )
    model_file_hashes = base.snapshot_file_hashes(snapshot_path)
    tokenizer = AutoTokenizer.from_pretrained(snapshot_path, local_files_only=True)
    prompt_maxima = preflight_prompt_lengths(
        records, schema, tokenizer, args.max_input_tokens, args.prompt_profile
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        snapshot_path,
        local_files_only=True,
        dtype=dtype,
        use_safetensors=True,
    )
    model.to(device)
    model.eval()

    prompt_code = "\n".join(
        inspect.getsource(component)
        for component in (
            article_first_context,
            target_context,
            prompt_preamble,
            build_natural_prompt,
            build_letter_prompt,
            few_shot_block,
            hierarchy_decision,
            classify_active_batch,
        )
    )
    coarse_module_path = Path(coarse.__file__).resolve()
    manifest: dict[str, Any] = {
        "status": "in_progress",
        "prompt_version": PROMPT_VERSION,
        "extractor_source_sha256": base.sha256_file(Path(__file__).resolve()),
        "prompt_builder_sha256": base.sha256_text(prompt_code),
        "deterministic_rule_version": coarse.DETERMINISTIC_RULE_VERSION,
        "deterministic_rules_payload_sha256": coarse.DETERMINISTIC_RULES_SHA256,
        "deterministic_module_sha256": base.sha256_file(coarse_module_path),
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
        "model_snapshot_path": str(snapshot_path),
        "model_files_sha256": model_file_hashes,
        "coarse_fields": list(coarse.COARSE_FIELDS),
        "decoding": args.decoding,
        "prompt_profile": args.prompt_profile,
        "hierarchy": {
            "irrelevant_gate": "skip_all_semantic_fields",
            "idiosyncratic_single_firm": "derive_single_firm_only_unless_target_and_peer_are_explicit",
            "unclear_scope": "derive_unclear_alignment",
            "relational_alignment": "run_model_for_common_mixed_or_explicit_target_plus_peer",
        },
        "device": device,
        "precision": str(dtype).replace("torch.", ""),
        "batch_size": args.batch_size,
        "max_input_tokens": args.max_input_tokens,
        "preflight_max_prompt_tokens_by_field": prompt_maxima,
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "seed": 0,
            "use_cache": True,
            "score": "mean_token_log_probability_including_eos",
            "choice_order_averaging": args.decoding == "order_averaged_letter_score",
            "tie_break": "schema_order",
        },
        "runtime": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "device_name": torch.cuda.get_device_name(0) if device == "cuda" else device,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "deterministic_algorithms_enforced": True,
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
                origin = "deterministic_gate" if reason == "deterministic_relevance_gate" else "hierarchical_derivation"
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
                    decoding=args.decoding,
                    tokenizer=tokenizer,
                    model=model,
                    torch=torch,
                    device=device,
                    max_input_tokens=args.max_input_tokens,
                    prompt_profile=args.prompt_profile,
                )
                for state, classification in zip(active_states, classified):
                    value = classification.pop("value")
                    if value not in schema["closed_label_fields"][field]:
                        raise RuntimeError(f"Decoder produced invalid {field} value {value!r}")
                    state["labels"][field] = value
                    state["origins"][field] = "model"
                    state["passes"][field] = {**classification, "value": value, "valid": True}

        for state in states:
            applicable = bool(state["deterministic"]["semantic_applicable"])
            result = {
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
                        all(state["labels"][field] is not None for field in coarse.COARSE_FIELDS)
                        if applicable
                        else True
                    ),
                    "no_input_truncation": not any(
                        pass_record.get("input_truncated")
                        for pass_record in state["passes"].values()
                    ),
                },
            }
            results.append(result)
        print(
            f"Processed {min(start + len(batch), len(records))}/{len(records)} records",
            file=sys.stderr,
            flush=True,
        )

    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
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
