from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import cache_llama_2 as cache_common
import coarse_news_features as coarse
import experiment_llama_2_v1_1 as shared_experiment
import extract_flan_t5 as base
import extract_llama_2_coarse as shared_runtime


MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
MODEL_REVISION = "0e9e39f249a16976918f6564b8830bc894c89659"
CONSERVATIVE_DATA_CUTOFF = "2023-12-31"
CONSERVATIVE_DATA_CUTOFF_DATE = date(2023, 12, 31)
OPERATIONAL_CONTEXT_LIMIT = 1024
PROMPT_VERSION = "llama-3.1-stock-sector-news-v1.0.0"
MANIFEST_VERSION = "llama-3.1-extraction-v1"
DECODING_METHOD = "official_chat_exact_answer_all_cyclic_sequence_score"
SNAPSHOT_MANIFEST = Path("outputs/llama_3_1/model_snapshot_manifest.json")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the pinned Llama 3.1 8B Instruct checkpoint on the coarse "
            "financial-news schema using deterministic candidate scoring."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("config/news_feature_schema_coarse.json"),
    )
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    return args


def flatten_ids(value: Any, *, source: str) -> list[int]:
    return shared_experiment.flatten_token_ids(value, source=source)


def official_prompt_ids(tokenizer: Any, prompt: str) -> list[int]:
    if not getattr(tokenizer, "chat_template", None):
        raise ValueError("Pinned Llama 3.1 tokenizer has no official chat template")
    return flatten_ids(
        tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        ),
        source="official user-only chat",
    )


def build_sequence_choice(
    *,
    tokenizer: Any,
    prompt: str,
    semantic_to_letter: Mapping[str, str],
) -> dict[str, Any]:
    """Derive candidate token sequences from complete official chat renders."""

    prompt_ids = official_prompt_ids(tokenizer, prompt)
    suffixes: dict[str, list[int]] = {}
    full_ids_by_letter: dict[str, list[int]] = {}
    for letter in semantic_to_letter.values():
        full_ids = flatten_ids(
            tokenizer.apply_chat_template(
                [
                    {"role": "user", "content": prompt},
                    {
                        "role": "assistant",
                        "content": shared_experiment.ASSISTANT_REPLY_TEMPLATE.format(
                            letter=letter
                        ),
                    },
                ],
                tokenize=True,
                add_generation_prompt=False,
            ),
            source=f"official complete assistant reply {letter}",
        )
        full_ids_by_letter[letter] = full_ids
        suffixes[letter] = shared_experiment.assistant_suffix(prompt_ids, full_ids)

    ordered = list(suffixes.values())
    shared_prefix = shared_experiment.common_prefix(ordered)
    shared_suffix = shared_experiment.common_suffix(
        ordered, protected_prefix_length=len(shared_prefix)
    )
    if not shared_prefix:
        raise ValueError("Official replies have no shared assistant Answer: prefix")
    suffix_length = len(shared_suffix)
    candidate_token_ids: dict[str, list[int]] = {}
    for letter, suffix in suffixes.items():
        stop = len(suffix) - suffix_length if suffix_length else len(suffix)
        candidate = suffix[len(shared_prefix) : stop]
        if not candidate:
            raise ValueError(f"Candidate {letter!r} has no distinct token sequence")
        candidate_token_ids[letter] = candidate
    if len({tuple(tokens) for tokens in candidate_token_ids.values()}) != len(
        candidate_token_ids
    ):
        raise ValueError("Candidate letters do not have unique token sequences")

    cue_text = tokenizer.decode(
        shared_prefix,
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    if "Answer:" not in cue_text:
        raise ValueError(
            f"Derived official assistant prefix does not contain Answer:: {cue_text!r}"
        )
    input_ids = prompt_ids + shared_prefix
    return {
        "prompt": prompt,
        "prompt_sha256": base.sha256_text(prompt),
        "input_ids": input_ids,
        "chat_input_ids_sha256": shared_runtime._sha256_int_sequence(input_ids),
        "user_prompt_input_ids_sha256": shared_runtime._sha256_int_sequence(
            prompt_ids
        ),
        "semantic_to_letter": dict(semantic_to_letter),
        "candidate_token_ids": candidate_token_ids,
        "letter_to_token_id": {
            letter: tokens[0]
            for letter, tokens in candidate_token_ids.items()
            if len(tokens) == 1
        },
        "shared_assistant_prefix_ids": shared_prefix,
        "shared_assistant_prefix_text": cue_text,
        "shared_assistant_suffix_ids": shared_suffix,
        "complete_reply_input_ids_sha256": {
            letter: shared_runtime._sha256_int_sequence(ids)
            for letter, ids in full_ids_by_letter.items()
        },
    }


def score_candidate_sequences(
    encoded_choices: Sequence[Mapping[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: str,
    max_input_tokens: int,
) -> list[dict[str, float]]:
    """Score one-token choices efficiently and multi-token choices exactly."""

    if not encoded_choices:
        return []
    all_single_token = all(
        all(len(tokens) == 1 for tokens in choice["candidate_token_ids"].values())
        for choice in encoded_choices
    )
    if all_single_token:
        return score_single_token_choices(
            encoded_choices,
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            device=device,
            max_input_tokens=max_input_tokens,
        )

    results: list[dict[str, float]] = []
    for choice in encoded_choices:
        prefix = list(choice["input_ids"])
        if len(prefix) > max_input_tokens:
            raise RuntimeError("Refusing to truncate an official chat prompt")
        scores: dict[str, float] = {}
        for label, letter in choice["semantic_to_letter"].items():
            candidate = list(choice["candidate_token_ids"][letter])
            combined = prefix + candidate
            if len(combined) > max_input_tokens:
                raise RuntimeError("Candidate continuation exceeds context limit")
            input_ids = torch.tensor(
                [combined], dtype=torch.long, device=device
            )
            attention_mask = torch.ones_like(input_ids)
            positions = candidate_prediction_positions(
                prefix_length=len(prefix),
                candidate_length=len(candidate),
            )
            logit_positions = torch.tensor(
                positions, dtype=torch.long, device=device
            )
            with torch.inference_mode():
                logits = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    logits_to_keep=logit_positions,
                ).logits.float()
            target = torch.tensor(candidate, dtype=torch.long, device=device)
            token_log_probs = torch.log_softmax(
                logits[0], dim=-1
            ).gather(-1, target.unsqueeze(-1)).squeeze(-1)
            scores[label] = float(token_log_probs.mean().detach().cpu())
        results.append(scores)
    return results


def candidate_prediction_positions(
    *, prefix_length: int, candidate_length: int
) -> list[int]:
    """Return the causal-logit positions that predict a candidate sequence."""

    if prefix_length < 1:
        raise ValueError("prefix_length must be positive")
    if candidate_length < 1:
        raise ValueError("candidate_length must be positive")
    return list(
        range(prefix_length - 1, prefix_length - 1 + candidate_length)
    )


def score_single_token_choices(
    encoded_choices: Sequence[Mapping[str, Any]],
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: str,
    max_input_tokens: int,
) -> list[dict[str, float]]:
    """Score one-token candidates without relying on mutable shared state."""

    if not encoded_choices:
        return []
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError(
            "Pinned tokenizer has neither pad_token_id nor eos_token_id"
        )

    results: list[dict[str, float]] = []
    for choice in encoded_choices:
        ids = list(choice["input_ids"])
        if len(ids) > max_input_tokens:
            raise RuntimeError("Refusing to truncate an official chat prompt")
        input_ids = torch.tensor([ids], dtype=torch.long, device=device)
        attention_mask = torch.ones_like(input_ids)
        with torch.inference_mode():
            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                logits_to_keep=1,
            ).logits
        next_log_probs = torch.log_softmax(
            logits[0, -1].float(), dim=-1
        )
        results.append(
            {
                label: float(
                    next_log_probs[
                        choice["candidate_token_ids"][letter][0]
                    ]
                    .detach()
                    .cpu()
                )
                for label, letter in choice["semantic_to_letter"].items()
            }
        )
    return results


@contextmanager
def bound_snapshot_download(snapshot_path: Path) -> Iterator[None]:
    """Force the shared loader to use the already hash-verified snapshot."""

    try:
        import huggingface_hub
    except ImportError as exc:
        raise RuntimeError(
            "Install requirements-llama-3-1.txt before Llama 3.1 inference"
        ) from exc

    original = huggingface_hub.snapshot_download

    def resolve_verified_snapshot(*args: Any, **kwargs: Any) -> str:
        repo_id = kwargs.get("repo_id", args[0] if args else None)
        revision = kwargs.get("revision")
        if repo_id != MODEL_ID or revision != MODEL_REVISION:
            raise ValueError(
                "Shared loader requested a model other than the verified "
                "Llama 3.1 snapshot"
            )
        if kwargs.get("local_files_only") is not True:
            raise ValueError("Llama 3.1 extraction must remain local-only")
        return str(snapshot_path)

    huggingface_hub.snapshot_download = resolve_verified_snapshot
    try:
        yield
    finally:
        huggingface_hub.snapshot_download = original


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
            key = tuple(shared_runtime.allowed_labels_for(schema, field, scope))
            groups.setdefault(key, []).append(index)
        if len(groups) > 1:
            restored: list[dict[str, Any] | None] = [None] * len(active_states)
            for label_tuple, indices in groups.items():
                subset_results = classify_active_batch(
                    field=field,
                    active_states=[active_states[index] for index in indices],
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

    labels = _allowed_labels or shared_runtime.allowed_labels_for(
        schema, field, scopes[0]
    )
    rotations = shared_experiment.cyclic_rotations(labels)
    score_sets: list[list[dict[str, float]]] = [[] for _ in active_states]
    choice_sets: list[list[dict[str, Any]]] = [[] for _ in active_states]
    for rotation in rotations:
        choices = [
            shared_experiment.encode_rotation(
                tokenizer=tokenizer,
                record=state["record"],
                schema=schema,
                field=field,
                labels_in_order=rotation,
                selected_scope=scope,
            )
            for state, scope in zip(active_states, scopes)
        ]
        rotation_scores = score_candidate_sequences(
            choices,
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            device=device,
            max_input_tokens=max_input_tokens,
        )
        for index, (choice, scores) in enumerate(zip(choices, rotation_scores)):
            choice_sets[index].append(choice)
            score_sets[index].append(scores)

    results: list[dict[str, Any]] = []
    for choices, rotation_scores in zip(choice_sets, score_sets):
        averaged = shared_experiment.mean_rotation_scores(rotation_scores, labels)
        value = shared_runtime.best_label(averaged, labels)
        rotation_predictions = [
            shared_runtime.best_label(scores, labels)
            for scores in rotation_scores
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
                "top1_top2_margin": shared_runtime.score_margin(averaged),
                "shared_assistant_prefix_ids": choices[0][
                    "shared_assistant_prefix_ids"
                ],
                "shared_assistant_prefix_text": choices[0][
                    "shared_assistant_prefix_text"
                ],
                "shared_assistant_suffix_ids": choices[0][
                    "shared_assistant_suffix_ids"
                ],
                "candidate_token_ids": choices[0]["candidate_token_ids"],
                "candidate_scoring": (
                    "single_next_token"
                    if all(
                        len(tokens) == 1
                        for tokens in choices[0]["candidate_token_ids"].values()
                    )
                    else "full_candidate_sequence_mean_log_probability"
                ),
            }
        )
    return results


@contextmanager
def configured_shared_runtime() -> Iterator[None]:
    """Temporarily adapt the proven Llama 2 pipeline, then restore it exactly."""

    runtime_overrides = {
        "MODEL_DEFAULT": MODEL_ID,
        "REVISION_DEFAULT": MODEL_REVISION,
        "CONSERVATIVE_DATA_CUTOFF": CONSERVATIVE_DATA_CUTOFF,
        "CONSERVATIVE_DATA_CUTOFF_DATE": CONSERVATIVE_DATA_CUTOFF_DATE,
        "MODEL_CONTEXT_WINDOW": OPERATIONAL_CONTEXT_LIMIT,
        "apply_chat_prompt_ids": official_prompt_ids,
        "_score_encoded_batch": score_candidate_sequences,
    }
    experiment_overrides = {
        "PROMPT_VERSION": PROMPT_VERSION,
        "MANIFEST_VERSION": MANIFEST_VERSION,
        "DECODING_METHOD": DECODING_METHOD,
        "build_answer_cue_choice": build_sequence_choice,
        "classify_active_batch": classify_active_batch,
    }
    prior_runtime = {
        name: getattr(shared_runtime, name) for name in runtime_overrides
    }
    prior_experiment = {
        name: getattr(shared_experiment, name) for name in experiment_overrides
    }
    try:
        for name, value in runtime_overrides.items():
            setattr(shared_runtime, name, value)
        for name, value in experiment_overrides.items():
            setattr(shared_experiment, name, value)
        yield
    finally:
        for name, value in prior_runtime.items():
            setattr(shared_runtime, name, value)
        for name, value in prior_experiment.items():
            setattr(shared_experiment, name, value)


def validate_without_model(args: argparse.Namespace) -> int:
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    shared_runtime.validate_schema(schema)
    all_records = base.read_jsonl(args.input)
    records = all_records[: args.limit] if args.limit is not None else all_records
    if not records:
        raise ValueError("Input contains no records")
    if len({record.get("article_id") for record in records}) != len(records):
        raise ValueError("Input contains duplicate article_id values")
    timestamps = []
    routes: dict[str, int] = {}
    for record in records:
        base.validate_input_record(record)
        coarse.validate_input_record(record)
        timestamps.append(shared_runtime.validate_post_cutoff_record(record))
        features = coarse.deterministic_features(record)
        route = features["gate_route"]
        routes[route] = routes.get(route, 0) + 1
        for field in coarse.COARSE_FIELDS:
            scopes: tuple[str | None, ...] = (
                ("common", "idiosyncratic")
                if field == "directional_alignment"
                else (None,)
            )
            for scope in scopes:
                labels = shared_runtime.allowed_labels_for(schema, field, scope)
                prompt, _ = shared_experiment.build_cued_letter_prompt(
                    record, schema, field, labels, scope
                )
                if (
                    record["headline"] not in prompt
                    or record["article_text"] not in prompt
                ):
                    raise ValueError("Prompt omitted source article content")
    print(
        json.dumps(
            {
                "validated_input_records": len(records),
                "prompt_version": PROMPT_VERSION,
                "model_id": MODEL_ID,
                "model_revision": MODEL_REVISION,
                "conservative_model_data_cutoff": CONSERVATIVE_DATA_CUTOFF,
                "minimum_time_published_utc": min(timestamps).isoformat(),
                "maximum_time_published_utc": max(timestamps).isoformat(),
                "decoding": DECODING_METHOD,
                "deterministic_relevance_routes": dict(sorted(routes.items())),
                "model_packages_imported": False,
                "model_was_loaded": False,
            },
            indent=2,
        )
    )
    return 0


def validate_snapshot_contract(path: Path = SNAPSHOT_MANIFEST) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(
            f"{path} is missing; run scripts/cache_llama_3_1.py first"
        )
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("manifest_version") != "llama-3.1-local-snapshot-v1":
        raise ValueError("Unsupported Llama 3.1 snapshot manifest")
    if manifest.get("model_id") != MODEL_ID:
        raise ValueError("Llama 3.1 snapshot manifest has the wrong model")
    if manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError("Llama 3.1 snapshot manifest has the wrong revision")
    snapshot_path_raw = manifest.get("snapshot_path")
    if not isinstance(snapshot_path_raw, str) or not snapshot_path_raw:
        raise ValueError("Llama 3.1 snapshot manifest has no snapshot_path")
    snapshot_path = Path(snapshot_path_raw).resolve()
    if not snapshot_path.is_dir():
        raise FileNotFoundError(
            f"Cached Llama 3.1 snapshot path is missing: {snapshot_path}"
        )
    expected_files = manifest.get("files")
    if not isinstance(expected_files, dict) or not expected_files:
        raise ValueError("Llama 3.1 snapshot manifest has no file inventory")
    actual_files = cache_common.snapshot_file_manifest(snapshot_path)
    if actual_files != expected_files:
        raise ValueError(
            "Cached Llama 3.1 snapshot bytes differ from the frozen manifest"
        )
    if manifest.get("file_count") != len(actual_files):
        raise ValueError("Llama 3.1 snapshot manifest file_count is inconsistent")
    if manifest.get("total_size_bytes") != sum(
        item["size_bytes"] for item in actual_files.values()
    ):
        raise ValueError(
            "Llama 3.1 snapshot manifest total_size_bytes is inconsistent"
        )
    return manifest


def forward_args(args: argparse.Namespace) -> list[str]:
    forwarded = [
        "--input",
        str(args.input),
        "--output",
        str(args.output),
        "--schema",
        str(args.schema),
        "--model-id",
        MODEL_ID,
        "--revision",
        MODEL_REVISION,
        "--batch-size",
        "1",
        "--max-input-tokens",
        str(OPERATIONAL_CONTEXT_LIMIT),
        "--device",
        "cuda",
        "--precision",
        "float16",
        "--quantization",
        "nf4",
        "--local-files-only",
    ]
    if args.limit is not None:
        forwarded.extend(["--limit", str(args.limit)])
    if args.overwrite:
        forwarded.append("--overwrite")
    return forwarded


def rewrite_manifest(output: Path, snapshot_manifest: dict[str, Any]) -> None:
    manifest_path = output.with_suffix(output.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected_model_hashes = {
        name: record["sha256"]
        for name, record in snapshot_manifest["files"].items()
    }
    if manifest.get("model_files_sha256") != expected_model_hashes:
        raise RuntimeError(
            "The model files loaded for extraction differ from the verified "
            "Llama 3.1 snapshot manifest"
        )
    input_name = Path(manifest["input_path"]).name.lower()
    split = (
        "development"
        if "development" in input_name
        else "evaluation"
        if "evaluation" in input_name
        else "unspecified"
    )
    manifest.update(
        {
            "manifest_version": MANIFEST_VERSION,
            "development_only": split == "development",
            "benchmark_split": split,
            "warning": (
                "Agreement is measured against GPT-5.6-derived silver labels. "
                "Freeze decisions on development before reading evaluation metrics."
            ),
            "extractor_source_sha256": base.sha256_file(Path(__file__).resolve()),
            "deterministic_module_sha256": base.sha256_file(
                Path(coarse.__file__).resolve()
            ),
            "shared_runtime_source_sha256": base.sha256_file(
                Path(shared_runtime.__file__).resolve()
            ),
            "shared_experiment_source_sha256": base.sha256_file(
                Path(shared_experiment.__file__).resolve()
            ),
            "hierarchy": "same frozen coarse hierarchy as FLAN v0.4",
            "official_chat_template": True,
            "generation": {
                "performed": False,
                "score": (
                    "single-token log probability after exact official assistant "
                    "Answer: cue; full candidate-sequence mean log probability "
                    "fallback when a candidate spans multiple tokens"
                ),
                "cyclic_order_averaging": True,
                "tie_break": "schema order",
            },
            "snapshot_manifest_path": str(SNAPSHOT_MANIFEST),
            "snapshot_manifest_sha256": base.sha256_file(SNAPSHOT_MANIFEST),
            "snapshot_manifest_file_count": snapshot_manifest["file_count"],
        }
    )
    base.write_manifest(manifest_path, manifest)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    with configured_shared_runtime():
        if args.validate_only:
            return validate_without_model(args)

        snapshot_manifest = validate_snapshot_contract()
        snapshot_path = Path(snapshot_manifest["snapshot_path"]).resolve()
        os.environ["HF_HUB_OFFLINE"] = "1"
        with bound_snapshot_download(snapshot_path):
            result = shared_experiment.main(forward_args(args))
        if result == 0:
            rewrite_manifest(args.output, snapshot_manifest)
            print(f"Wrote pinned Llama 3.1 predictions to {args.output}")
        return result


if __name__ == "__main__":
    raise SystemExit(main())
