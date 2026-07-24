from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import coarse_news_features as coarse
import extract_flan_t5 as base
import extract_llama_2_coarse as v1


PROMPT_VERSION = "llama-2-stock-sector-news-v1.1.0-development"
MANIFEST_VERSION = "llama-2-development-experiment-v1"
DECODING_METHOD = "exact_answer_cue_all_cyclic_letter_score"
ASSISTANT_REPLY_TEMPLATE = "Answer: {letter}"
FINAL_INSTRUCTION = (
    'Reply exactly "Answer: X", replacing X with one option letter, '
    "and nothing else."
)
SNAPSHOT_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model*.safetensors*",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "tokenizer.model",
]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Llama 2 v1.1 development-only answer-boundary experiment. "
            "The pinned model, NF4 configuration, schema, hierarchy, and inputs "
            "remain unchanged. Candidate letters are scored after an exact "
            "tokenizer-derived partial assistant reply, over every cyclic option "
            "rotation."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--schema",
        default=Path("config/news_feature_schema_coarse.json"),
        type=Path,
    )
    parser.add_argument("--model-id", default=v1.MODEL_DEFAULT)
    parser.add_argument("--revision", default=v1.REVISION_DEFAULT)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=v1.MODEL_CONTEXT_WINDOW)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="float16",
    )
    parser.add_argument(
        "--quantization", choices=("none", "nf4"), default="nf4"
    )
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--validate-only", action="store_true")
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.model_id != v1.MODEL_DEFAULT:
        parser.error(f"--model-id must remain pinned to {v1.MODEL_DEFAULT}")
    if args.revision != v1.REVISION_DEFAULT:
        parser.error(f"--revision must remain pinned to {v1.REVISION_DEFAULT}")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not 512 <= args.max_input_tokens <= v1.MODEL_CONTEXT_WINDOW:
        parser.error(
            f"--max-input-tokens must be between 512 and {v1.MODEL_CONTEXT_WINDOW}"
        )
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.quantization == "nf4" and args.device == "cpu":
        parser.error("--quantization nf4 requires CUDA")
    if args.device == "cpu" and args.precision == "float16":
        parser.error("float16 CPU inference is unsupported")
    return args


def cyclic_rotations(labels: Sequence[str]) -> list[list[str]]:
    values = list(labels)
    if len(values) < 2 or len(values) != len(set(values)):
        raise ValueError("Cyclic option rotation requires at least two unique labels")
    return [values[offset:] + values[:offset] for offset in range(len(values))]


def build_cued_letter_prompt(
    record: Mapping[str, Any],
    schema: Mapping[str, Any],
    field: str,
    labels_in_order: Sequence[str],
    selected_scope: str | None,
) -> tuple[str, dict[str, str]]:
    prompt, mapping = v1.build_letter_prompt(
        record,
        schema,
        field,
        labels_in_order,
        selected_scope,
    )
    old_instruction = "Answer with exactly one option letter and nothing else."
    if not prompt.endswith(old_instruction):
        raise RuntimeError("Frozen v1 prompt contract changed unexpectedly")
    prompt = prompt[: -len(old_instruction)] + FINAL_INSTRUCTION
    return prompt, mapping


def flatten_token_ids(value: Any, *, source: str) -> list[int]:
    if hasattr(value, "tolist"):
        value = value.tolist()
    if value and isinstance(value[0], list):
        if len(value) != 1:
            raise ValueError(f"{source} unexpectedly returned a token batch")
        value = value[0]
    if (
        not isinstance(value, list)
        or not value
        or not all(isinstance(token_id, int) for token_id in value)
    ):
        raise ValueError(f"{source} did not return a nonempty token-id list")
    return list(value)


def assistant_suffix(
    prompt_ids: Sequence[int], full_chat_ids: Sequence[int]
) -> list[int]:
    if len(full_chat_ids) <= len(prompt_ids):
        raise ValueError("Full chat did not extend the user-only prompt")
    if list(full_chat_ids[: len(prompt_ids)]) != list(prompt_ids):
        raise ValueError(
            "The official full chat is not prefixed by the frozen user-only prompt"
        )
    return list(full_chat_ids[len(prompt_ids) :])


def common_prefix(sequences: Sequence[Sequence[int]]) -> list[int]:
    if not sequences:
        raise ValueError("At least one sequence is required")
    length = 0
    limit = min(len(sequence) for sequence in sequences)
    while length < limit and len({sequence[length] for sequence in sequences}) == 1:
        length += 1
    return list(sequences[0][:length])


def common_suffix(
    sequences: Sequence[Sequence[int]], *, protected_prefix_length: int
) -> list[int]:
    available = min(
        len(sequence) - protected_prefix_length for sequence in sequences
    )
    length = 0
    while length < available and len(
        {sequence[-(length + 1)] for sequence in sequences}
    ) == 1:
        length += 1
    return list(sequences[0][-length:]) if length else []


def build_answer_cue_choice(
    *,
    tokenizer: Any,
    prompt: str,
    semantic_to_letter: Mapping[str, str],
) -> dict[str, Any]:
    """Derive the exact partial-assistant cue from complete pinned chat renders."""

    prompt_ids = v1.apply_chat_prompt_ids(tokenizer, prompt)
    suffixes: dict[str, list[int]] = {}
    rendered_reply_ids: dict[str, list[int]] = {}
    for letter in semantic_to_letter.values():
        full_ids = flatten_token_ids(
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": ASSISTANT_REPLY_TEMPLATE.format(letter=letter),
                    },
                ],
                tokenize=True,
                add_generation_prompt=False,
            ),
            source=f"complete assistant reply {letter}",
        )
        rendered_reply_ids[letter] = full_ids
        suffixes[letter] = assistant_suffix(prompt_ids, full_ids)

    ordered_suffixes = list(suffixes.values())
    shared_prefix = common_prefix(ordered_suffixes)
    shared_suffix = common_suffix(
        ordered_suffixes, protected_prefix_length=len(shared_prefix)
    )
    if not shared_prefix:
        raise ValueError("Complete replies have no shared assistant Answer: prefix")

    suffix_length = len(shared_suffix)
    candidate_specific: dict[str, list[int]] = {}
    for letter, suffix in suffixes.items():
        stop = len(suffix) - suffix_length if suffix_length else len(suffix)
        candidate_specific[letter] = suffix[len(shared_prefix) : stop]
    if any(len(ids) != 1 for ids in candidate_specific.values()):
        raise ValueError(
            "Every option letter must occupy exactly one candidate-specific token"
        )
    token_ids = {
        letter: ids[0] for letter, ids in candidate_specific.items()
    }
    if len(set(token_ids.values())) != len(token_ids):
        raise ValueError("Option letters do not have unique candidate token IDs")

    cue_text = tokenizer.decode(
        shared_prefix,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if "Answer:" not in cue_text:
        raise ValueError(
            f"Derived assistant prefix does not contain Answer:: {cue_text!r}"
        )
    input_ids = prompt_ids + shared_prefix
    return {
        "prompt": prompt,
        "prompt_sha256": base.sha256_text(prompt),
        "input_ids": input_ids,
        "chat_input_ids_sha256": v1._sha256_int_sequence(input_ids),
        "user_prompt_input_ids_sha256": v1._sha256_int_sequence(prompt_ids),
        "semantic_to_letter": dict(semantic_to_letter),
        "letter_to_token_id": token_ids,
        "shared_assistant_prefix_ids": shared_prefix,
        "shared_assistant_prefix_text": cue_text,
        "shared_assistant_suffix_ids": shared_suffix,
        "complete_reply_input_ids_sha256": {
            letter: v1._sha256_int_sequence(ids)
            for letter, ids in rendered_reply_ids.items()
        },
    }


def encode_rotation(
    *,
    tokenizer: Any,
    record: Mapping[str, Any],
    schema: Mapping[str, Any],
    field: str,
    labels_in_order: Sequence[str],
    selected_scope: str | None,
) -> dict[str, Any]:
    prompt, mapping = build_cued_letter_prompt(
        record,
        schema,
        field,
        labels_in_order,
        selected_scope,
    )
    return build_answer_cue_choice(
        tokenizer=tokenizer,
        prompt=prompt,
        semantic_to_letter=mapping,
    )


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
                labels = v1.allowed_labels_for(schema, field, scope)
                for rotation in cyclic_rotations(labels):
                    encoded = encode_rotation(
                        tokenizer=tokenizer,
                        record=record,
                        schema=schema,
                        field=field,
                        labels_in_order=rotation,
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
            "Cue-augmented prompt preflight failed; refusing truncation: "
            + json.dumps(violations[:20], ensure_ascii=False)
        )
    return maxima


def mean_rotation_scores(
    score_maps: Sequence[Mapping[str, float]], labels: Sequence[str]
) -> dict[str, float]:
    if not score_maps:
        raise ValueError("At least one rotation score map is required")
    return {
        label: sum(score_map[label] for score_map in score_maps) / len(score_maps)
        for label in labels
    }


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
            key = tuple(v1.allowed_labels_for(schema, field, scope))
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

    labels = _allowed_labels or v1.allowed_labels_for(schema, field, scopes[0])
    rotations = cyclic_rotations(labels)
    scores_by_state: list[list[dict[str, float]]] = [
        [] for _ in active_states
    ]
    choices_by_state: list[list[dict[str, Any]]] = [
        [] for _ in active_states
    ]

    # One model call per rotation keeps the effective NF4 batch at the requested
    # record batch size; it does not multiply activation memory by the class count.
    for rotation in rotations:
        choices = [
            encode_rotation(
                tokenizer=tokenizer,
                record=state["record"],
                schema=schema,
                field=field,
                labels_in_order=rotation,
                selected_scope=scope,
            )
            for state, scope in zip(active_states, scopes)
        ]
        rotation_scores = v1._score_encoded_batch(
            choices,
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            device=device,
            max_input_tokens=max_input_tokens,
        )
        for index, (choice, score_map) in enumerate(
            zip(choices, rotation_scores)
        ):
            choices_by_state[index].append(choice)
            scores_by_state[index].append(score_map)

    results: list[dict[str, Any]] = []
    for choices, rotation_scores in zip(choices_by_state, scores_by_state):
        averaged = mean_rotation_scores(rotation_scores, labels)
        value = v1.best_label(averaged, labels)
        rotation_predictions = [
            v1.best_label(score_map, labels) for score_map in rotation_scores
        ]
        results.append(
            {
                "value": value,
                "raw_output": value,
                "origin": "model",
                "prompt_sha256_by_rotation": [
                    choice["prompt_sha256"] for choice in choices
                ],
                "chat_input_ids_sha256_by_rotation": [
                    choice["chat_input_ids_sha256"] for choice in choices
                ],
                "input_truncated": False,
                "rotation_label_orders": rotations,
                "rotation_semantic_to_letter": [
                    choice["semantic_to_letter"] for choice in choices
                ],
                "rotation_candidate_log_probabilities": rotation_scores,
                "all_rotation_mean_log_probabilities": averaged,
                "rotation_predictions": rotation_predictions,
                "rotation_prediction_agreement_rate": (
                    sum(prediction == value for prediction in rotation_predictions)
                    / len(rotation_predictions)
                ),
                "top1_top2_margin": v1.score_margin(averaged),
                "shared_assistant_prefix_ids": choices[0][
                    "shared_assistant_prefix_ids"
                ],
                "shared_assistant_prefix_text": choices[0][
                    "shared_assistant_prefix_text"
                ],
                "shared_assistant_suffix_ids": choices[0][
                    "shared_assistant_suffix_ids"
                ],
                "letter_token_ids": choices[0]["letter_to_token_id"],
            }
        )
    return results


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
    v1.validate_schema(schema)
    all_records = base.read_jsonl(args.input)
    if not all_records:
        raise ValueError("Input contains no records")
    records = all_records[: args.limit] if args.limit is not None else all_records
    if len({record.get("article_id") for record in records}) != len(records):
        raise ValueError("Input contains duplicate article_id values")

    publication_times: list[datetime] = []
    for record in records:
        base.validate_input_record(record)
        coarse.validate_input_record(record)
        publication_times.append(v1.validate_post_cutoff_record(record))
        for field in coarse.COARSE_FIELDS:
            scopes: tuple[str | None, ...] = (
                ("common", "idiosyncratic")
                if field == "directional_alignment"
                else (None,)
            )
            for scope in scopes:
                labels = v1.allowed_labels_for(schema, field, scope)
                prompt, _ = build_cued_letter_prompt(
                    record, schema, field, labels, scope
                )
                if record["headline"] not in prompt or record["article_text"] not in prompt:
                    raise ValueError("Cue prompt omitted source article content")

    deterministic_by_id = {
        record["article_id"]: coarse.deterministic_features(record)
        for record in records
    }
    if args.validate_only:
        print(
            json.dumps(
                {
                    "validated_input_records": len(records),
                    "prompt_version": PROMPT_VERSION,
                    "model_id": args.model_id,
                    "model_revision": args.revision,
                    "decoding": DECODING_METHOD,
                    "development_only": True,
                    "model_packages_imported": False,
                    "model_was_loaded": False,
                },
                indent=2,
            )
        )
        return 0

    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")
    manifest_path = args.output.with_suffix(args.output.suffix + ".manifest.json")
    if manifest_path.exists() and not args.overwrite:
        raise FileExistsError(f"{manifest_path} exists; pass --overwrite")
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
            "Missing the documented local Llama 2 model dependencies"
        ) from exc

    device, dtype = v1.resolve_device_and_dtype(
        torch, args.device, args.precision, args.quantization
    )
    base.configure_determinism(torch, set_seed)
    snapshot_path = Path(
        snapshot_download(
            repo_id=args.model_id,
            revision=args.revision,
            local_files_only=args.local_files_only,
            allow_patterns=SNAPSHOT_ALLOW_PATTERNS,
        )
    )
    model_file_hashes = v1.snapshot_file_hashes(snapshot_path)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path, local_files_only=True, use_fast=True
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
        frozen_device_map = v1.validate_nf4_device_map(model)
    if int(getattr(model.config, "max_position_embeddings", 0)) < args.max_input_tokens:
        raise RuntimeError("Pinned model context is below --max-input-tokens")
    model.eval()
    model.config.use_cache = False

    prompt_code = "\n".join(
        inspect.getsource(component)
        for component in (
            build_cued_letter_prompt,
            build_answer_cue_choice,
            cyclic_rotations,
            classify_active_batch,
        )
    )
    manifest: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "status": "in_progress",
        "development_only": True,
        "warning": (
            "This run may use the 72-document development reference for decoder "
            "selection. It is not a confirmatory evaluation."
        ),
        "prompt_version": PROMPT_VERSION,
        "extractor_source_sha256": base.sha256_file(Path(__file__).resolve()),
        "frozen_v1_source_sha256": base.sha256_file(Path(v1.__file__).resolve()),
        "prompt_builder_sha256": base.sha256_text(prompt_code),
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
        "conservative_model_data_cutoff": v1.CONSERVATIVE_DATA_CUTOFF,
        "minimum_time_published_utc": min(publication_times)
        .astimezone(timezone.utc)
        .isoformat(),
        "maximum_time_published_utc": max(publication_times)
        .astimezone(timezone.utc)
        .isoformat(),
        "decoding": DECODING_METHOD,
        "assistant_reply_template": ASSISTANT_REPLY_TEMPLATE,
        "assistant_reply_template_sha256": _canonical_json_sha256(
            ASSISTANT_REPLY_TEMPLATE
        ),
        "option_orders": "all_cyclic_rotations",
        "rotation_counts_by_field": {
            field: len(v1.allowed_labels_for(schema, field, None))
            if field != "directional_alignment"
            else 3
            for field in coarse.COARSE_FIELDS
        },
        "hierarchy": "unchanged from Llama 2 v1.0 / FLAN v0.4",
        "deterministic_rule_version": coarse.DETERMINISTIC_RULE_VERSION,
        "deterministic_rules_payload_sha256": coarse.DETERMINISTIC_RULES_SHA256,
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
        "batch_size": args.batch_size,
        "max_input_tokens": args.max_input_tokens,
        "preflight_max_prompt_tokens_by_field": prompt_maxima,
        "generation": {
            "performed": False,
            "score": (
                "next-token option-letter log probability after exact "
                "tokenizer-derived partial assistant Answer: reply"
            ),
            "cyclic_order_averaging": True,
            "tie_break": "schema_order",
        },
        "model_files_sha256": model_file_hashes,
        "runtime": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "device_name": (
                torch.cuda.get_device_name(0) if device == "cuda" else device
            ),
        },
        "total_prediction_count": len(records),
    }
    base.write_manifest(manifest_path, manifest)

    results: list[dict[str, Any]] = []
    for start in range(0, len(records), args.batch_size):
        batch = records[start : start + args.batch_size]
        states: list[dict[str, Any]] = []
        for record in batch:
            states.append(
                {
                    "record": record,
                    "deterministic": deterministic_by_id[record["article_id"]],
                    "labels": {field: None for field in coarse.COARSE_FIELDS},
                    "origins": {},
                    "passes": {},
                }
            )

        for field in coarse.COARSE_FIELDS:
            active_states: list[dict[str, Any]] = []
            for state in states:
                run_model, derived, reason = v1.hierarchy_decision(
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
                        "no_input_truncation": True,
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
                json.dumps(result, ensure_ascii=False, separators=(",", ":"))
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
            "model_forward_count": sum(
                len(result["passes"][field].get("rotation_label_orders", []))
                for result in results
                for field in coarse.COARSE_FIELDS
            ),
            "output_sha256": base.sha256_file(args.output),
        }
    )
    base.write_manifest(manifest_path, manifest)
    print(f"Wrote {len(results)} development predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
