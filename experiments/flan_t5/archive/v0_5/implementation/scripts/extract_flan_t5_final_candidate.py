from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import coarse_news_features as coarse
import extract_flan_t5 as base
import extract_flan_t5_coarse as legacy_coarse


MODEL_ID = "google/flan-t5-large"
MODEL_REVISION = "0613663d0d48ea86ba8cb3d7a44f0f65dc596a2a"
PROTOCOL_CONFIG_DEFAULT = Path("config/flan_t5_final_candidate_v0_5.json")
SCHEMA_DEFAULT = Path("config/news_feature_schema_coarse.json")
BINARY_SEMANTICS = ("yes", "no")
COMPONENT_ORDER = (
    "scope_firm",
    "scope_common",
    "event_earnings_guidance",
    "event_product_demand",
    "event_supply_capacity",
    "event_regulation_legal",
    "event_corporate_analyst",
    "event_macro_market",
    "alignment_same",
    "alignment_opposite",
)
PROMPT_VERSION = "flan-stock-sector-news-binary-v0.5.0"
PROTOCOL_VERSION = "0.5.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Score the isolated v0.5 FLAN-T5-Large binary-component candidate. "
            "This extractor writes raw component scores, not promoted coarse labels."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--protocol-config", default=PROTOCOL_CONFIG_DEFAULT, type=Path)
    parser.add_argument("--schema", default=SCHEMA_DEFAULT, type=Path)
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
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
        help="Validate inputs, gate decisions, config, and prompt contracts without loading FLAN.",
    )
    args = parser.parse_args()
    if args.model_id != MODEL_ID:
        parser.error(f"v0.5 is hash-locked to --model-id {MODEL_ID}")
    if args.revision != MODEL_REVISION:
        parser.error(f"v0.5 is hash-locked to --revision {MODEL_REVISION}")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        parser.error("--revision must be a 40-character lowercase hexadecimal commit")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not 128 <= args.max_input_tokens <= 512:
        parser.error("--max-input-tokens must be between 128 and 512")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def load_json_object(path: Path, description: str) -> tuple[str, dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"{description} must contain one JSON object")
    return text, value


def validate_protocol_config(
    protocol: Mapping[str, Any], schema: Mapping[str, Any]
) -> None:
    if protocol.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"Protocol config must have protocol_version {PROTOCOL_VERSION}")
    if protocol.get("prompt_version") != PROMPT_VERSION:
        raise ValueError(f"Protocol config must have prompt_version {PROMPT_VERSION}")
    if tuple(protocol.get("binary_semantics", ())) != BINARY_SEMANTICS:
        raise ValueError("Protocol binary_semantics must be exactly ['yes', 'no']")
    if tuple(protocol.get("candidate_component_order", ())) != COMPONENT_ORDER:
        raise ValueError("Protocol candidate_component_order does not match v0.5")
    components = protocol.get("components")
    if not isinstance(components, Mapping) or tuple(components) != COMPONENT_ORDER:
        raise ValueError("Protocol components must exactly match the ordered v0.5 component set")
    expected_groups = {
        "scope_firm": "scope",
        "scope_common": "scope",
        "event_earnings_guidance": "event",
        "event_product_demand": "event",
        "event_supply_capacity": "event",
        "event_regulation_legal": "event",
        "event_corporate_analyst": "event",
        "event_macro_market": "event",
        "alignment_same": "alignment",
        "alignment_opposite": "alignment",
    }
    for component_name in COMPONENT_ORDER:
        component = components[component_name]
        if not isinstance(component, Mapping):
            raise ValueError(f"Protocol component {component_name} must be an object")
        if component.get("group") != expected_groups[component_name]:
            raise ValueError(f"Protocol component {component_name} has the wrong group")
        for key in ("definition", "decision_rule"):
            value = component.get(key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Protocol component {component_name}.{key} must be nonempty")
    prompt_contract = protocol.get("prompt_contract")
    if not isinstance(prompt_contract, Mapping):
        raise ValueError("Protocol prompt_contract must be an object")
    for key in ("source_rule", "positive_rule", "negative_rule", "output_rule"):
        if not isinstance(prompt_contract.get(key), str) or not prompt_contract[key].strip():
            raise ValueError(f"Protocol prompt_contract.{key} must be nonempty")

    legacy_coarse.validate_schema(dict(schema))
    event_labels = set(schema["closed_label_fields"]["event_family"])
    component_event_labels = {
        component.removeprefix("event_")
        for component in COMPONENT_ORDER
        if component.startswith("event_")
    }
    if component_event_labels != event_labels - {"other_or_unclear"}:
        raise ValueError(
            "The six event components must equal the coarse event families excluding other_or_unclear"
        )


def semantic_to_letter(semantic_order: Sequence[str]) -> dict[str, str]:
    if tuple(sorted(semantic_order)) != tuple(sorted(BINARY_SEMANTICS)):
        raise ValueError("Binary prompt order must contain yes and no exactly once")
    return {
        semantic: chr(ord("A") + index)
        for index, semantic in enumerate(semantic_order)
    }


def component_target_context(record: Mapping[str, Any], group: str) -> str:
    target = record["target"]
    base_context = legacy_coarse.target_context(dict(record))
    if group != "alignment":
        return base_context
    peers = ", ".join(target["known_sector_peers"]) or "none supplied"
    return f"{base_context}\nKnown sector peers: {peers}"


def build_binary_prompt(
    record: dict[str, Any],
    protocol: Mapping[str, Any],
    component_name: str,
    semantic_order: Sequence[str],
) -> tuple[str, dict[str, str]]:
    if component_name not in COMPONENT_ORDER:
        raise ValueError(f"Unknown v0.5 component {component_name!r}")
    mapping = semantic_to_letter(semantic_order)
    component = protocol["components"][component_name]
    contract = protocol["prompt_contract"]
    options = "\n".join(
        f"{mapping[semantic]}. {semantic}" for semantic in semantic_order
    )
    prompt = (
        f"{legacy_coarse.article_first_context(record)}\n\n"
        "TASK\n"
        f"{component_target_context(record, component['group'])}\n"
        f"Component definition: {component['definition']}\n"
        f"Decision rule: {component['decision_rule']}\n"
        f"{contract['source_rule']}\n"
        f"{contract['positive_rule']}\n"
        f"{contract['negative_rule']}\n\n"
        "Does the supplied article support this component?\n\n"
        f"OPTIONS\n{options}\n\n"
        f"{contract['output_rule']}\n"
        "Answer:"
    )
    return prompt, mapping


def prompt_pair(
    record: dict[str, Any],
    protocol: Mapping[str, Any],
    component_name: str,
) -> tuple[str, dict[str, str], str, dict[str, str]]:
    canonical_prompt, canonical_mapping = build_binary_prompt(
        record, protocol, component_name, BINARY_SEMANTICS
    )
    reversed_prompt, reversed_mapping = build_binary_prompt(
        record, protocol, component_name, tuple(reversed(BINARY_SEMANTICS))
    )
    return canonical_prompt, canonical_mapping, reversed_prompt, reversed_mapping


def validate_record_prompts(
    record: dict[str, Any], protocol: Mapping[str, Any]
) -> None:
    for component_name in COMPONENT_ORDER:
        canonical, canonical_mapping, reversed_prompt, reversed_mapping = prompt_pair(
            record, protocol, component_name
        )
        for prompt in (canonical, reversed_prompt):
            if record["headline"] not in prompt or record["article_text"] not in prompt:
                raise ValueError(
                    f"Prompt omitted source content for {record['article_id']} / {component_name}"
                )
            if prompt.index("ARTICLE") >= prompt.index("TASK"):
                raise ValueError(f"Prompt is not article-first for {component_name}")
        if canonical_mapping != {"yes": "A", "no": "B"}:
            raise ValueError("Canonical binary mapping changed")
        if reversed_mapping != {"no": "A", "yes": "B"}:
            raise ValueError("Reversed binary mapping changed")


def preflight_prompt_lengths(
    records: list[dict[str, Any]],
    protocol: Mapping[str, Any],
    tokenizer: Any,
    max_input_tokens: int,
) -> dict[str, int]:
    maxima: dict[str, int] = {}
    violations: list[dict[str, Any]] = []
    for component_name in COMPONENT_ORDER:
        maximum = 0
        for record in records:
            canonical, _, reversed_prompt, _ = prompt_pair(
                record, protocol, component_name
            )
            for order_name, prompt in (
                ("canonical", canonical),
                ("reversed", reversed_prompt),
            ):
                token_count = len(
                    tokenizer(
                        prompt,
                        add_special_tokens=True,
                        truncation=False,
                    )["input_ids"]
                )
                maximum = max(maximum, token_count)
                if token_count > max_input_tokens:
                    violations.append(
                        {
                            "article_id": record["article_id"],
                            "component": component_name,
                            "order": order_name,
                            "token_count": token_count,
                        }
                    )
        maxima[component_name] = maximum
    if violations:
        raise RuntimeError(
            "Prompt preflight failed; v0.5 never truncates article inputs: "
            + json.dumps(violations[:20], ensure_ascii=False)
        )
    return maxima


def average_binary_scores(
    canonical_scores: Mapping[str, float],
    reversed_scores: Mapping[str, float],
) -> dict[str, float]:
    if set(canonical_scores) != set(BINARY_SEMANTICS):
        raise ValueError("Canonical score map must contain exactly yes and no")
    if set(reversed_scores) != set(BINARY_SEMANTICS):
        raise ValueError("Reversed score map must contain exactly yes and no")
    averaged = {
        semantic: (float(canonical_scores[semantic]) + float(reversed_scores[semantic]))
        / 2.0
        for semantic in BINARY_SEMANTICS
    }
    if not all(math.isfinite(score) for score in averaged.values()):
        raise ValueError("Non-finite binary component score")
    return averaged


def binary_prediction(score_map: Mapping[str, float]) -> str:
    # A tie is conservatively negative; strict consensus never promotes a tie.
    return "yes" if float(score_map["yes"]) > float(score_map["no"]) else "no"


def make_component_result(
    *,
    canonical_prompt: str,
    reversed_prompt: str,
    canonical_mapping: Mapping[str, str],
    reversed_mapping: Mapping[str, str],
    canonical_scores: Mapping[str, float],
    reversed_scores: Mapping[str, float],
    canonical_truncated: bool,
    reversed_truncated: bool,
) -> dict[str, Any]:
    if canonical_truncated or reversed_truncated:
        raise RuntimeError("v0.5 refuses to record any truncated component prompt")
    canonical = {semantic: float(canonical_scores[semantic]) for semantic in BINARY_SEMANTICS}
    reversed_values = {
        semantic: float(reversed_scores[semantic]) for semantic in BINARY_SEMANTICS
    }
    averaged = average_binary_scores(canonical, reversed_values)
    canonical_choice = binary_prediction(canonical)
    reversed_choice = binary_prediction(reversed_values)
    return {
        "valid": True,
        "input_truncated": False,
        "canonical_prompt_sha256": base.sha256_text(canonical_prompt),
        "reversed_prompt_sha256": base.sha256_text(reversed_prompt),
        "canonical_semantic_to_letter": dict(canonical_mapping),
        "reversed_semantic_to_letter": dict(reversed_mapping),
        "canonical_candidate_mean_log_probabilities": canonical,
        "reversed_candidate_mean_log_probabilities": reversed_values,
        "canonical_prediction": canonical_choice,
        "reversed_prediction": reversed_choice,
        "order_averaged_mean_log_probabilities": averaged,
        "order_averaged_yes_no_log_odds": averaged["yes"] - averaged["no"],
        "order_averaged_prediction": binary_prediction(averaged),
        "strict_consensus_yes": canonical_choice == "yes" and reversed_choice == "yes",
    }


def score_component_batch(
    *,
    records: list[dict[str, Any]],
    protocol: Mapping[str, Any],
    component_name: str,
    tokenizer: Any,
    model: Any,
    torch: Any,
    device: str,
    max_input_tokens: int,
) -> list[dict[str, Any]]:
    canonical_prompts: list[str] = []
    reversed_prompts: list[str] = []
    canonical_mappings: list[dict[str, str]] = []
    reversed_mappings: list[dict[str, str]] = []
    for record in records:
        canonical, canonical_mapping, reversed_prompt, reversed_mapping = prompt_pair(
            record, protocol, component_name
        )
        canonical_prompts.append(canonical)
        canonical_mappings.append(canonical_mapping)
        reversed_prompts.append(reversed_prompt)
        reversed_mappings.append(reversed_mapping)

    _, canonical_truncated, canonical_scores = base.score_closed_label_batch(
        canonical_prompts,
        list(BINARY_SEMANTICS),
        [canonical_mappings[0][semantic] for semantic in BINARY_SEMANTICS],
        tokenizer,
        model,
        torch,
        device,
        max_input_tokens,
    )
    _, reversed_truncated, reversed_scores = base.score_closed_label_batch(
        reversed_prompts,
        list(BINARY_SEMANTICS),
        [reversed_mappings[0][semantic] for semantic in BINARY_SEMANTICS],
        tokenizer,
        model,
        torch,
        device,
        max_input_tokens,
    )
    return [
        make_component_result(
            canonical_prompt=canonical_prompts[index],
            reversed_prompt=reversed_prompts[index],
            canonical_mapping=canonical_mappings[index],
            reversed_mapping=reversed_mappings[index],
            canonical_scores=canonical_scores[index],
            reversed_scores=reversed_scores[index],
            canonical_truncated=canonical_truncated[index],
            reversed_truncated=reversed_truncated[index],
        )
        for index in range(len(records))
    ]


def source_hashes() -> dict[str, str]:
    paths = {
        "extract_flan_t5_final_candidate.py": Path(__file__).resolve(),
        "extract_flan_t5.py": Path(base.__file__).resolve(),
        "extract_flan_t5_coarse.py": Path(legacy_coarse.__file__).resolve(),
        "coarse_news_features.py": Path(coarse.__file__).resolve(),
    }
    return {name: base.sha256_file(path) for name, path in paths.items()}


def prompt_builder_hash() -> str:
    source = "\n".join(
        inspect.getsource(component)
        for component in (
            semantic_to_letter,
            component_target_context,
            build_binary_prompt,
            prompt_pair,
            average_binary_scores,
            binary_prediction,
            make_component_result,
            score_component_batch,
        )
    )
    return base.sha256_text(source)


def main() -> int:
    args = parse_args()
    protocol_text, protocol = load_json_object(args.protocol_config, "Protocol config")
    schema_text, schema = load_json_object(args.schema, "Coarse schema")
    validate_protocol_config(protocol, schema)
    records = base.read_jsonl(args.input)
    for record in records:
        base.validate_input_record(record)
        coarse.validate_input_record(record)
        validate_record_prompts(record, protocol)
    article_ids = [record["article_id"] for record in records]
    if len(article_ids) != len(set(article_ids)):
        raise ValueError("Input contains duplicate article_id values")
    if args.limit is not None:
        records = records[: args.limit]

    deterministic_by_id = {
        record["article_id"]: coarse.deterministic_features(record) for record in records
    }
    applicable_records = [
        record
        for record in records
        if deterministic_by_id[record["article_id"]]["semantic_applicable"]
    ]
    if args.validate_only:
        routes: dict[str, int] = {}
        for features in deterministic_by_id.values():
            route = features["gate_route"]
            routes[route] = routes.get(route, 0) + 1
        print(
            json.dumps(
                {
                    "validated_input_records": len(records),
                    "semantic_applicable_records": len(applicable_records),
                    "protocol_version": PROTOCOL_VERSION,
                    "prompt_version": PROMPT_VERSION,
                    "model_id": MODEL_ID,
                    "model_revision": MODEL_REVISION,
                    "candidate_component_order": list(COMPONENT_ORDER),
                    "binary_semantics": list(BINARY_SEMANTICS),
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
            repo_id=MODEL_ID,
            revision=MODEL_REVISION,
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
        applicable_records, protocol, tokenizer, args.max_input_tokens
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        snapshot_path,
        local_files_only=True,
        dtype=dtype,
        use_safetensors=True,
    )
    model.to(device)
    model.eval()

    manifest: dict[str, Any] = {
        "status": "in_progress",
        "protocol_name": protocol["protocol_name"],
        "protocol_version": PROTOCOL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "source_files_sha256": source_hashes(),
        "prompt_builder_sha256": prompt_builder_hash(),
        "deterministic_rule_version": coarse.DETERMINISTIC_RULE_VERSION,
        "deterministic_rules_payload_sha256": coarse.DETERMINISTIC_RULES_SHA256,
        "protocol_config_path": str(args.protocol_config),
        "protocol_config_sha256": hashlib.sha256(
            protocol_text.encode("utf-8")
        ).hexdigest(),
        "schema_name": schema["schema_name"],
        "schema_version": schema["schema_version"],
        "schema_path": str(args.schema),
        "schema_sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
        "input_path": str(args.input),
        "input_sha256": base.sha256_file(args.input),
        "selected_article_ids_sha256": base.sha256_text(
            "\n".join(record["article_id"] for record in records)
        ),
        "model_id": MODEL_ID,
        "model_revision": MODEL_REVISION,
        "tokenizer_revision": MODEL_REVISION,
        "model_snapshot_path": str(snapshot_path),
        "model_files_sha256": model_file_hashes,
        "candidate_component_order": list(COMPONENT_ORDER),
        "binary_semantics": list(BINARY_SEMANTICS),
        "output_contract": "raw_binary_component_scores_only_no_promoted_labels",
        "deterministic_gate": "unchanged_coarse_news_features_semantic_applicable",
        "device": device,
        "precision": str(dtype).replace("torch.", ""),
        "batch_size": args.batch_size,
        "max_input_tokens": args.max_input_tokens,
        "preflight_max_prompt_tokens_by_component": prompt_maxima,
        "scoring": {
            "candidate_representation": "A_or_B",
            "score": "mean_token_log_probability_including_eos",
            "canonical_order": ["yes", "no"],
            "reversed_order": ["no", "yes"],
            "order_averaging": "arithmetic_mean_of_semantic_log_probabilities",
            "yes_no_log_odds": "averaged_yes_minus_averaged_no",
            "strict_consensus_yes": "canonical_yes_and_reversed_yes",
            "tie_break": "no",
            "do_sample": False,
            "num_beams": 1,
            "seed": 0,
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
        "semantic_applicable_count": len(applicable_records),
        "model_component_call_count": len(applicable_records) * len(COMPONENT_ORDER),
        "scored_prompt_count": len(applicable_records) * len(COMPONENT_ORDER) * 2,
    }
    base.write_manifest(manifest_path, manifest)

    result_by_id: dict[str, dict[str, Any]] = {}
    for record in records:
        deterministic = deterministic_by_id[record["article_id"]]
        result_by_id[record["article_id"]] = {
            "row_number": record["row_number"],
            "article_id": record["article_id"],
            "target_ticker": record["target"]["ticker"],
            "extractor": MODEL_ID,
            "model_revision": MODEL_REVISION,
            "protocol_version": PROTOCOL_VERSION,
            "prompt_version": PROMPT_VERSION,
            "semantic_applicable": bool(deterministic["semantic_applicable"]),
            "deterministic_features": deterministic,
            "component_order": list(COMPONENT_ORDER),
            "components": {},
        }

    for start in range(0, len(applicable_records), args.batch_size):
        batch = applicable_records[start : start + args.batch_size]
        for component_name in COMPONENT_ORDER:
            component_results = score_component_batch(
                records=batch,
                protocol=protocol,
                component_name=component_name,
                tokenizer=tokenizer,
                model=model,
                torch=torch,
                device=device,
                max_input_tokens=args.max_input_tokens,
            )
            for record, component_result in zip(batch, component_results):
                result_by_id[record["article_id"]]["components"][
                    component_name
                ] = component_result
        print(
            f"Scored {min(start + len(batch), len(applicable_records))}/"
            f"{len(applicable_records)} applicable records",
            file=sys.stderr,
            flush=True,
        )

    results: list[dict[str, Any]] = []
    for record in records:
        result = result_by_id[record["article_id"]]
        expected_components = (
            set(COMPONENT_ORDER) if result["semantic_applicable"] else set()
        )
        actual_components = set(result["components"])
        result["validity"] = {
            "expected_component_set": actual_components == expected_components,
            "all_components_valid": all(
                component.get("valid") is True
                for component in result["components"].values()
            ),
            "no_input_truncation": not any(
                component.get("input_truncated")
                for component in result["components"].values()
            ),
        }
        if not all(result["validity"].values()):
            raise RuntimeError(
                f"Invalid v0.5 output contract for article {result['article_id']}"
            )
        results.append(result)

    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for result in results:
            handle.write(
                json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n"
            )
    manifest.update(
        {
            "status": "complete",
            "output_sha256": base.sha256_file(args.output),
            "all_output_rows_valid": all(
                all(result["validity"].values()) for result in results
            ),
        }
    )
    base.write_manifest(manifest_path, manifest)
    print(f"Wrote {len(results)} raw v0.5 candidate rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
