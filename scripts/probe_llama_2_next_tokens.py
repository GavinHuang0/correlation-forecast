from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import extract_llama_2_coarse as extractor


SCHEMA_DEFAULT = Path("config/news_feature_schema_coarse.json")
GREEDY_NEW_TOKENS = 8
PROBE_VERSION = "llama-2-next-token-probe-v1"
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
            "Diagnose one canonical Llama 2 coarse-classification prompt using "
            "the pinned, locally cached NF4 CUDA checkpoint. The probe compares "
            "the extractor's standalone option-letter IDs with the actual "
            "assistant suffix produced by the checkpoint's full chat template."
        )
    )
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument(
        "--row",
        required=True,
        type=int,
        help="One-based nonblank JSONL record position in --input.",
    )
    parser.add_argument(
        "--field",
        required=True,
        choices=tuple(extractor.coarse.COARSE_FIELDS),
    )
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--schema", default=SCHEMA_DEFAULT, type=Path)
    parser.add_argument(
        "--selected-scope",
        choices=("idiosyncratic", "common", "mixed"),
        help=(
            "Required only for directional_alignment because its canonical "
            "options depend on the previously resolved shock scope."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing diagnostic JSON file.",
    )
    return parser


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.row < 1:
        parser.error("--row must be at least 1")
    if args.top_k < 1:
        parser.error("--top-k must be at least 1")
    if args.field == "directional_alignment" and args.selected_scope is None:
        parser.error(
            "--selected-scope is required for --field directional_alignment"
        )
    if args.field != "directional_alignment" and args.selected_scope is not None:
        parser.error(
            "--selected-scope is only valid for --field directional_alignment"
        )
    return args


def select_record(
    records: Sequence[Mapping[str, Any]], row: int
) -> Mapping[str, Any]:
    """Select a one-based nonblank JSONL record without interpreting row_number."""

    if row < 1 or row > len(records):
        raise IndexError(
            f"--row {row} is outside the input's 1..{len(records)} record range"
        )
    return records[row - 1]


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
    return value


def sha256_int_sequence(values: Sequence[int]) -> str:
    payload = json.dumps(list(values), separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assistant_suffix_from_full_chat(
    prompt_ids: Sequence[int], full_chat_ids: Sequence[int]
) -> list[int]:
    """Return the full-chat continuation and reject a mismatched user prefix."""

    if len(full_chat_ids) <= len(prompt_ids):
        raise ValueError(
            "Full user/assistant chat did not extend the frozen user prompt"
        )
    mismatch = next(
        (
            index
            for index, (prompt_id, full_id) in enumerate(
                zip(prompt_ids, full_chat_ids)
            )
            if prompt_id != full_id
        ),
        None,
    )
    if mismatch is not None or list(full_chat_ids[: len(prompt_ids)]) != list(
        prompt_ids
    ):
        detail = (
            f" at token offset {mismatch}"
            if mismatch is not None
            else " because the full chat prefix is shorter"
        )
        raise ValueError(
            "The checkpoint's full chat template does not preserve the frozen "
            f"extractor prompt{detail}"
        )
    return list(full_chat_ids[len(prompt_ids) :])


def _common_prefix(sequences: Sequence[Sequence[int]]) -> list[int]:
    if not sequences:
        raise ValueError("At least one token sequence is required")
    limit = min(len(sequence) for sequence in sequences)
    length = 0
    while length < limit and len(
        {sequence[length] for sequence in sequences}
    ) == 1:
        length += 1
    return list(sequences[0][:length])


def _common_suffix(
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
    return list(sequences[0][len(sequences[0]) - length :]) if length else []


def partition_candidate_suffixes(
    suffixes: Mapping[str, Sequence[int]]
) -> dict[str, Any]:
    """Split full assistant suffixes into shared and letter-specific segments."""

    if len(suffixes) < 2:
        raise ValueError("At least two candidate suffixes are required")
    if any(not suffix for suffix in suffixes.values()):
        raise ValueError("Candidate assistant suffixes must be nonempty")
    ordered = [list(sequence) for sequence in suffixes.values()]
    shared_prefix = _common_prefix(ordered)
    shared_suffix = _common_suffix(
        ordered, protected_prefix_length=len(shared_prefix)
    )
    specific: dict[str, list[int]] = {}
    suffix_length = len(shared_suffix)
    for candidate, sequence in suffixes.items():
        stop = len(sequence) - suffix_length if suffix_length else len(sequence)
        candidate_ids = list(sequence[len(shared_prefix) : stop])
        if not candidate_ids:
            raise ValueError(
                f"Candidate {candidate!r} has no letter-specific continuation tokens"
            )
        specific[candidate] = candidate_ids
    serialized = [tuple(token_ids) for token_ids in specific.values()]
    if len(serialized) != len(set(serialized)):
        raise ValueError(
            "Full chat template does not give every option letter a unique "
            "candidate-specific token sequence"
        )
    return {
        "shared_prefix_ids": shared_prefix,
        "candidate_specific_ids": specific,
        "shared_suffix_ids": shared_suffix,
    }


def _decode_ids(tokenizer: Any, token_ids: Sequence[int]) -> str:
    return tokenizer.decode(
        list(token_ids),
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def token_details(tokenizer: Any, token_ids: Sequence[int]) -> list[dict[str, Any]]:
    details: list[dict[str, Any]] = []
    for token_id in token_ids:
        details.append(
            {
                "token_id": int(token_id),
                "vocabulary_token": tokenizer.convert_ids_to_tokens(int(token_id)),
                "decoded_text": _decode_ids(tokenizer, [int(token_id)]),
            }
        )
    return details


def build_continuation_contract(
    *,
    tokenizer: Any,
    prompt: str,
    semantic_to_letter: Mapping[str, str],
) -> dict[str, Any]:
    """Render and verify each option through the checkpoint's full chat template."""

    messages = [{"role": "user", "content": prompt}]
    prompt_ids = extractor.apply_chat_prompt_ids(tokenizer, prompt)
    rendered_prompt = tokenizer.apply_chat_template(
        messages,
        chat_template=extractor.LLAMA_2_SINGLE_TURN_CHAT_TEMPLATE,
        tokenize=False,
        add_generation_prompt=True,
    )
    if not isinstance(rendered_prompt, str) or not rendered_prompt:
        raise ValueError("Frozen chat template did not render a prompt string")

    letters = list(semantic_to_letter.values())
    candidate_prompt_ids, standalone_letter_ids = (
        extractor.candidate_letter_token_ids(tokenizer, prompt, letters)
    )
    if candidate_prompt_ids != prompt_ids:
        raise RuntimeError("Extractor candidate helper changed the prompt token IDs")

    candidates: dict[str, dict[str, Any]] = {}
    suffixes_by_letter: dict[str, list[int]] = {}
    for label, letter in semantic_to_letter.items():
        full_messages = messages + [{"role": "assistant", "content": letter}]
        rendered_full_chat = tokenizer.apply_chat_template(
            full_messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        full_chat_ids = flatten_token_ids(
            tokenizer.apply_chat_template(
                full_messages,
                tokenize=True,
                add_generation_prompt=False,
            ),
            source=f"full chat template for option {letter}",
        )
        suffix_ids = assistant_suffix_from_full_chat(prompt_ids, full_chat_ids)
        suffixes_by_letter[letter] = suffix_ids
        candidates[label] = {
            "letter": letter,
            "extractor_standalone_letter_token_id": standalone_letter_ids[letter],
            "full_chat_sha256": extractor.base.sha256_text(rendered_full_chat),
            "full_chat_input_ids_sha256": sha256_int_sequence(full_chat_ids),
            "assistant_continuation_suffix_ids": suffix_ids,
        }

    partition = partition_candidate_suffixes(suffixes_by_letter)
    for label, candidate in candidates.items():
        letter = candidate["letter"]
        specific_ids = partition["candidate_specific_ids"][letter]
        candidate["candidate_specific_ids"] = specific_ids
        candidate["standalone_matches_candidate_specific_ids"] = (
            specific_ids
            == [candidate["extractor_standalone_letter_token_id"]]
        )

    return {
        "rendered_prompt": rendered_prompt,
        "prompt_ids": prompt_ids,
        "prompt_ids_sha256": sha256_int_sequence(prompt_ids),
        "shared_assistant_prefix_ids": partition["shared_prefix_ids"],
        "shared_assistant_suffix_ids": partition["shared_suffix_ids"],
        "candidates": candidates,
    }


def score_suffix_tokens(
    *,
    model: Any,
    torch: Any,
    device: str,
    prefix_ids: Sequence[int],
    suffix_ids: Sequence[int],
) -> list[float]:
    """Teacher-force a suffix and return one conditional log probability per token."""

    if not prefix_ids or not suffix_ids:
        raise ValueError("Both prefix_ids and suffix_ids must be nonempty")
    model_input_ids = list(prefix_ids) + list(suffix_ids[:-1])
    input_ids = torch.tensor(
        [model_input_ids], dtype=torch.long, device=device
    )
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits[0]
    positions = logits[
        len(prefix_ids) - 1 : len(prefix_ids) + len(suffix_ids) - 1
    ].float()
    log_probs = torch.log_softmax(positions, dim=-1)
    targets = torch.tensor(list(suffix_ids), dtype=torch.long, device=device)
    values = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
    return [float(value) for value in values.detach().cpu().tolist()]


def immediate_top_k(
    *,
    next_token_log_probs: Any,
    tokenizer: Any,
    torch: Any,
    top_k: int,
) -> list[dict[str, Any]]:
    values, ids = torch.topk(
        next_token_log_probs, k=min(top_k, next_token_log_probs.shape[-1])
    )
    results: list[dict[str, Any]] = []
    for token_id, log_probability in zip(
        ids.detach().cpu().tolist(), values.detach().cpu().tolist()
    ):
        results.append(
            {
                **token_details(tokenizer, [token_id])[0],
                "log_probability": float(log_probability),
            }
        )
    return results


def greedy_continuation(
    *,
    model: Any,
    tokenizer: Any,
    torch: Any,
    device: str,
    prompt_ids: Sequence[int],
) -> dict[str, Any]:
    input_ids = torch.tensor([list(prompt_ids)], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        pad_token_id = tokenizer.eos_token_id
    if pad_token_id is None:
        raise ValueError("Pinned tokenizer has neither pad_token_id nor eos_token_id")
    with torch.inference_mode():
        generated = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=GREEDY_NEW_TOKENS,
            do_sample=False,
            num_beams=1,
            pad_token_id=pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    generated_ids = generated[0, len(prompt_ids) :].detach().cpu().tolist()
    return {
        "maximum_new_tokens": GREEDY_NEW_TOKENS,
        "generated_ids": generated_ids,
        "generated_ids_sha256": sha256_int_sequence(generated_ids),
        "tokens": token_details(tokenizer, generated_ids),
        "decoded_text": _decode_ids(tokenizer, generated_ids),
    }


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_name(path.name + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    temporary_path.replace(path)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output.exists() and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --overwrite")

    schema_text = args.schema.read_text(encoding="utf-8")
    schema = json.loads(schema_text)
    extractor.validate_schema(schema)
    records = extractor.base.read_jsonl(args.input)
    if not records:
        raise ValueError("Input contains no records")
    record = select_record(records, args.row)
    extractor.base.validate_input_record(dict(record))
    extractor.coarse.validate_input_record(record)
    extractor.validate_post_cutoff_record(record)
    extractor.validate_record_prompts(record, schema)

    labels = extractor.allowed_labels_for(
        schema, args.field, args.selected_scope
    )
    prompt, semantic_to_letter = extractor.build_letter_prompt(
        record,
        schema,
        args.field,
        labels,
        args.selected_scope,
    )

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
            "environment before running this probe."
        ) from exc

    device, dtype = extractor.resolve_device_and_dtype(
        torch, "cuda", "float16", "nf4"
    )
    extractor.base.configure_determinism(torch, set_seed)
    snapshot_path = Path(
        snapshot_download(
            repo_id=extractor.MODEL_DEFAULT,
            revision=extractor.REVISION_DEFAULT,
            local_files_only=True,
            allow_patterns=SNAPSHOT_ALLOW_PATTERNS,
        )
    )
    model_file_hashes = extractor.snapshot_file_hashes(snapshot_path)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path,
        local_files_only=True,
        use_fast=True,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    continuation = build_continuation_contract(
        tokenizer=tokenizer,
        prompt=prompt,
        semantic_to_letter=semantic_to_letter,
    )
    prompt_ids = continuation["prompt_ids"]
    if len(prompt_ids) > extractor.MODEL_CONTEXT_WINDOW:
        raise RuntimeError(
            f"Prompt has {len(prompt_ids)} tokens, above the pinned "
            f"{extractor.MODEL_CONTEXT_WINDOW}-token context window"
        )

    model_kwargs: dict[str, Any] = {
        "local_files_only": True,
        "use_safetensors": True,
        "dtype": dtype,
        "low_cpu_mem_usage": True,
        "attn_implementation": "eager",
        "quantization_config": BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=dtype,
        ),
        "device_map": {"": 0},
    }
    model = AutoModelForCausalLM.from_pretrained(snapshot_path, **model_kwargs)
    frozen_device_map = extractor.validate_nf4_device_map(model)
    model.eval()
    model.config.use_cache = False

    input_ids = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(input_ids)
    with torch.inference_mode():
        next_logits = model(
            input_ids=input_ids, attention_mask=attention_mask
        ).logits[0, -1].float()
    next_token_log_probs = torch.log_softmax(next_logits, dim=-1)
    greedy_first_token_id = int(torch.argmax(next_token_log_probs).detach().cpu())
    second_input_ids = torch.tensor(
        [prompt_ids + [greedy_first_token_id]], dtype=torch.long, device=device
    )
    second_attention_mask = torch.ones_like(second_input_ids)
    with torch.inference_mode():
        second_logits = model(
            input_ids=second_input_ids, attention_mask=second_attention_mask
        ).logits[0, -1].float()
    second_token_log_probs = torch.log_softmax(second_logits, dim=-1)

    shared_prefix_length = len(continuation["shared_assistant_prefix_ids"])
    for candidate in continuation["candidates"].values():
        suffix_ids = candidate["assistant_continuation_suffix_ids"]
        token_log_probs = score_suffix_tokens(
            model=model,
            torch=torch,
            device=device,
            prefix_ids=prompt_ids,
            suffix_ids=suffix_ids,
        )
        specific_length = len(candidate["candidate_specific_ids"])
        specific_token_log_probs = token_log_probs[
            shared_prefix_length : shared_prefix_length + specific_length
        ]
        standalone_id = candidate["extractor_standalone_letter_token_id"]
        candidate.update(
            {
                "assistant_continuation_tokens": token_details(
                    tokenizer, suffix_ids
                ),
                "assistant_continuation_token_log_probabilities": (
                    token_log_probs
                ),
                "assistant_continuation_sequence_log_probability": float(
                    sum(token_log_probs)
                ),
                "candidate_specific_tokens": token_details(
                    tokenizer, candidate["candidate_specific_ids"]
                ),
                "candidate_specific_token_log_probabilities": (
                    specific_token_log_probs
                ),
                "candidate_specific_sequence_log_probability": float(
                    sum(specific_token_log_probs)
                ),
                "extractor_standalone_immediate_log_probability": float(
                    next_token_log_probs[standalone_id].detach().cpu()
                ),
            }
        )

    output: dict[str, Any] = {
        "probe_version": PROBE_VERSION,
        "model_id": extractor.MODEL_DEFAULT,
        "model_revision": extractor.REVISION_DEFAULT,
        "prompt_version": extractor.PROMPT_VERSION,
        "chat_template_sha256": extractor._canonical_json_sha256(
            extractor.LLAMA_2_SINGLE_TURN_CHAT_TEMPLATE
        ),
        "input_path": str(args.input),
        "input_sha256": extractor.base.sha256_file(args.input),
        "input_record_position": args.row,
        "source_row_number": record["row_number"],
        "article_id": record["article_id"],
        "field": args.field,
        "selected_scope": args.selected_scope,
        "canonical_labels_in_order": labels,
        "semantic_to_letter": semantic_to_letter,
        "prompt": prompt,
        "prompt_sha256": extractor.base.sha256_text(prompt),
        "rendered_chat_prompt": continuation["rendered_prompt"],
        "rendered_chat_prompt_sha256": extractor.base.sha256_text(
            continuation["rendered_prompt"]
        ),
        "prompt_token_count": len(prompt_ids),
        "prompt_input_ids": prompt_ids,
        "prompt_input_ids_sha256": continuation["prompt_ids_sha256"],
        "assistant_continuation_contract": {
            "template_source": "checkpoint tokenizer full user/assistant chat template",
            "frozen_user_prompt_is_exact_prefix": True,
            "shared_prefix_ids": continuation[
                "shared_assistant_prefix_ids"
            ],
            "shared_prefix_tokens": token_details(
                tokenizer, continuation["shared_assistant_prefix_ids"]
            ),
            "shared_suffix_ids": continuation[
                "shared_assistant_suffix_ids"
            ],
            "shared_suffix_tokens": token_details(
                tokenizer, continuation["shared_assistant_suffix_ids"]
            ),
            "candidates_by_semantic_label": continuation["candidates"],
        },
        "immediate_next_token_top_k": immediate_top_k(
            next_token_log_probs=next_token_log_probs,
            tokenizer=tokenizer,
            torch=torch,
            top_k=args.top_k,
        ),
        "after_greedy_first_token": {
            "first_token": token_details(
                tokenizer, [greedy_first_token_id]
            )[0],
            "first_token_log_probability": float(
                next_token_log_probs[greedy_first_token_id].detach().cpu()
            ),
            "next_token_top_k": immediate_top_k(
                next_token_log_probs=second_token_log_probs,
                tokenizer=tokenizer,
                torch=torch,
                top_k=args.top_k,
            ),
            "candidate_letter_log_probabilities": {
                label: float(
                    second_token_log_probs[
                        candidate["extractor_standalone_letter_token_id"]
                    ]
                    .detach()
                    .cpu()
                )
                for label, candidate in continuation["candidates"].items()
            },
        },
        "greedy_short_continuation": greedy_continuation(
            model=model,
            tokenizer=tokenizer,
            torch=torch,
            device=device,
            prompt_ids=prompt_ids,
        ),
        "runtime": {
            "python_version": sys.version,
            "torch_version": torch.__version__,
            "transformers_version": transformers.__version__,
            "cuda_device_name": torch.cuda.get_device_name(0),
            "device": device,
            "precision": "float16",
            "quantization": "nf4",
            "bnb_4bit_quant_type": "nf4",
            "bnb_4bit_use_double_quant": True,
            "attention_implementation": "eager",
            "hf_device_map": frozen_device_map,
            "local_files_only": True,
            "model_snapshot_path": str(snapshot_path),
            "model_files_sha256": model_file_hashes,
        },
    }
    _write_json_atomic(args.output, output)
    print(f"Wrote Llama 2 next-token probe to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
