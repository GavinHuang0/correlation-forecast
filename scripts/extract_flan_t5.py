from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import random
import re
import sys
from pathlib import Path
from typing import Any, Callable


MODEL_DEFAULT = "google/flan-t5-large"
PROMPT_VERSION = "flan-stock-sector-news-v0.1.0"
CORE_PASSES = ("scope", "event", "direction")
FULL_PASSES = CORE_PASSES + ("channels", "entities", "evidence")
PASS_FIELDS = {
    "scope": ("relevance", "event_scope", "affected_breadth"),
    "event": ("event_type", "information_status", "explicit_surprise"),
    "direction": ("target_direction", "sector_direction", "peer_effect"),
}
MAX_NEW_TOKENS = {
    "scope": 32,
    "event": 40,
    "direction": 40,
    "channels": 32,
    "entities": 96,
    "evidence": 160,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen FLAN-T5-Large as a deterministic, multi-pass news feature extractor."
    )
    parser.add_argument("--input", required=True, type=Path, help="Blinded benchmark JSONL")
    parser.add_argument("--output", required=True, type=Path, help="Prediction JSONL")
    parser.add_argument("--schema", default=Path("config/news_feature_schema.json"), type=Path)
    parser.add_argument("--model-id", default=MODEL_DEFAULT)
    parser.add_argument(
        "--revision",
        required=True,
        help="Immutable Hugging Face commit hash. Branch names such as main are rejected.",
    )
    parser.add_argument("--mode", choices=("core", "full"), default="full")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-input-tokens", type=int, default=512)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda", "mps"), default="auto")
    parser.add_argument(
        "--precision",
        choices=("auto", "float32", "float16", "bfloat16"),
        default="auto",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Prevent any Hugging Face network access; use after downloading the pinned snapshot.",
    )
    parser.add_argument("--resume", action="store_true", help="Append only records not already present")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int, help="Process only the first N records for a smoke test")
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate inputs and construct prompts without importing or running Transformers",
    )
    args = parser.parse_args()
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", args.revision):
        parser.error("--revision must be an immutable hexadecimal commit hash, not a branch or tag")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if not 128 <= args.max_input_tokens <= 512:
        parser.error("--max-input-tokens must be between 128 and FLAN-T5's 512-token limit")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be at least 1")
    if args.resume and args.overwrite:
        parser.error("--resume and --overwrite are mutually exclusive")
    return args


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: record must be an object")
            records.append(record)
    return records


def validate_input_record(record: dict[str, Any]) -> None:
    required = {
        "row_number",
        "article_id",
        "time_published_utc",
        "source",
        "headline",
        "article_text",
        "vendor_tickers",
        "target",
    }
    missing = required - set(record)
    if missing:
        raise ValueError(f"{record.get('article_id', '<unknown>')}: missing input keys {sorted(missing)}")
    target = record["target"]
    target_required = {"company", "ticker", "sector", "sector_benchmark", "known_sector_peers"}
    if not isinstance(target, dict) or target_required - set(target):
        raise ValueError(f"{record['article_id']}: incomplete target metadata")
    if not isinstance(record["headline"], str) or not isinstance(record["article_text"], str):
        raise ValueError(f"{record['article_id']}: headline and article_text must be strings")


def article_block(record: dict[str, Any]) -> str:
    target = record["target"]
    peers = ", ".join(target["known_sector_peers"]) or "none supplied"
    vendor_tickers = ", ".join(record["vendor_tickers"]) or "none supplied"
    return (
        f"Target company: {target['company']}\n"
        f"Target ticker: {target['ticker']}\n"
        f"Target sector: {target['sector']}\n"
        f"Sector benchmark: {target['sector_benchmark']}\n"
        f"Known sector peers: {peers}\n"
        f"Vendor-assigned tickers: {vendor_tickers}\n\n"
        f"Headline: {record['headline']}\n"
        f"Article text: {record['article_text']}"
    )


def build_prompt(
    pass_name: str,
    record: dict[str, Any],
    schema: dict[str, Any],
    extracted_labels: dict[str, Any] | None = None,
) -> str:
    context = article_block(record)
    closed = schema["closed_label_fields"]
    if pass_name == "scope":
        return (
            "Use only the supplied financial article and target metadata. Do not use external or future facts.\n"
            f"RELEVANCE: {', '.join(closed['relevance'])}\n"
            f"EVENT_SCOPE: {', '.join(closed['event_scope'])}\n"
            f"AFFECTED_BREADTH: {', '.join(closed['affected_breadth'])}\n"
            "Sector membership alone is not sector-wide. Use sector_wide only when several sector firms or the sector as a whole are explicitly affected.\n\n"
            f"{context}\n\n"
            "Return exactly three lowercase values separated by space-pipe-space:\n"
            "RELEVANCE | EVENT_SCOPE | AFFECTED_BREADTH"
        )
    if pass_name == "event":
        return (
            "Use only the supplied financial article. Do not use external or future facts.\n"
            f"EVENT_TYPE: {', '.join(closed['event_type'])}\n"
            f"INFORMATION_STATUS: {', '.join(closed['information_status'])}\n"
            f"EXPLICIT_SURPRISE: {', '.join(closed['explicit_surprise'])}\n"
            "Do not infer surprise unless the text explicitly compares with expectations, estimates, guidance, or a prior benchmark.\n\n"
            f"{context}\n\n"
            "Return exactly three lowercase values separated by space-pipe-space:\n"
            "EVENT_TYPE | INFORMATION_STATUS | EXPLICIT_SURPRISE"
        )
    if pass_name == "direction":
        return (
            "Use only the supplied financial article and target metadata. Direction is the stated effect, not a market prediction.\n"
            f"TARGET_DIRECTION: {', '.join(closed['target_direction'])}\n"
            f"SECTOR_DIRECTION: {', '.join(closed['sector_direction'])}\n"
            f"PEER_EFFECT: {', '.join(closed['peer_effect'])}\n\n"
            f"{context}\n\n"
            "Return exactly three lowercase values separated by space-pipe-space:\n"
            "TARGET_DIRECTION | SECTOR_DIRECTION | PEER_EFFECT"
        )
    if pass_name == "channels":
        allowed = schema["transmission_channels"]["allowed_values"]
        return (
            "Use only the supplied financial article. Select no more than two event transmission channels.\n"
            f"Allowed channels: {', '.join(allowed)}\n"
            "Return NONE when the text supports no transmission channel. Do not explain.\n\n"
            f"{context}\n\n"
            "Return exactly NONE, or one or two lowercase channels separated by comma-space."
        )
    if pass_name == "entities":
        return (
            "Use only the headline and article text for the output entity lists. List only companies and sectors explicitly named or explicitly covered by the event. "
            "Target metadata helps interpretation only: do not output a target, peer, benchmark, or vendor ticker unless its name also appears in the headline or article text. "
            "Do not add inferred peers, customers, subsidiaries, or sector members. Use NONE for an empty list.\n\n"
            f"{context}\n\n"
            "Return exactly:\nCOMPANIES: item; item | SECTORS: item; item"
        )
    if pass_name == "evidence":
        labels = extracted_labels or {}
        required_labels = (
            f"Extracted event scope: {labels.get('event_scope', '<EVENT_SCOPE>')}\n"
            f"Extracted target direction: {labels.get('target_direction', '<TARGET_DIRECTION>')}\n"
            f"Extracted sector direction: {labels.get('sector_direction', '<SECTOR_DIRECTION>')}\n"
            f"Extracted explicit surprise: {labels.get('explicit_surprise', '<EXPLICIT_SURPRISE>')}\n"
        )
        return (
            "Use only the supplied headline and article text. Copy short exact substrings, preserving case and punctuation. "
            "SCOPE must support the extracted event scope; DIRECTION must support any extracted positive, negative, neutral, or mixed target/sector direction; "
            "SURPRISE must support an extracted positive, negative, or mixed surprise. "
            "Use NONE when the corresponding claim is unsupported. Do not paraphrase.\n\n"
            f"{required_labels}\n"
            f"{context}\n\n"
            "Return exactly:\nSCOPE: exact text or NONE | DIRECTION: exact text or NONE | SURPRISE: exact text or NONE"
        )
    raise ValueError(f"Unknown pass {pass_name!r}")


def _clean_enum_token(piece: str, field: str) -> tuple[str, bool]:
    token = piece.strip()
    strict = True
    token = token.strip("`\"'")
    if ":" in token:
        prefix, remainder = token.split(":", 1)
        if prefix.strip().lower() == field.lower():
            token = remainder.strip()
            strict = False
    if token.endswith("."):
        token = token[:-1].rstrip()
        strict = False
    return token.lower(), strict


def parse_closed_pass(
    raw_output: str,
    fields: tuple[str, ...],
    schema: dict[str, Any],
) -> dict[str, Any]:
    pieces = raw_output.strip().split("|")
    values: dict[str, str | None] = {field: None for field in fields}
    errors: list[str] = []
    strict_tokens = True
    if len(pieces) != len(fields):
        errors.append(f"expected {len(fields)} pipe-separated values, received {len(pieces)}")
    for field, piece in zip(fields, pieces):
        value, token_was_strict = _clean_enum_token(piece, field)
        strict_tokens = strict_tokens and token_was_strict
        if value not in schema["closed_label_fields"][field]:
            errors.append(f"{field}: invalid value {value!r}")
        else:
            values[field] = value
    expected = " | ".join(values[field] or "" for field in fields)
    return {
        "values": values,
        "valid": not errors,
        "strict_format_valid": not errors and strict_tokens and raw_output.strip() == expected,
        "errors": errors,
    }


def parse_channels(raw_output: str, schema: dict[str, Any]) -> dict[str, Any]:
    text = raw_output.strip()
    strict = True
    if text.lower().startswith("channels:"):
        text = text.split(":", 1)[1].strip()
        strict = False
    channels = (
        []
        if text.upper() == "NONE"
        else [item.strip().lower().rstrip(".") for item in text.split(",") if item.strip()]
    )
    allowed = set(schema["transmission_channels"]["allowed_values"])
    max_items = schema["transmission_channels"]["max_items"]
    errors: list[str] = []
    if len(channels) > max_items:
        errors.append(f"more than {max_items} channels returned")
    if len(channels) != len(set(channels)):
        errors.append("duplicate channel returned")
    invalid = sorted(set(channels) - allowed)
    if invalid:
        errors.append(f"invalid channels: {invalid}")
    expected = ", ".join(channels) if channels else "NONE"
    return {
        "values": channels if not errors else [],
        "valid": not errors,
        "strict_format_valid": not errors and strict and raw_output.strip() == expected,
        "errors": errors,
    }


def _parse_semicolon_list(value: str) -> list[str]:
    if value.strip().upper() == "NONE":
        return []
    return [item.strip() for item in value.split(";") if item.strip()]


def parse_entities(raw_output: str, source_text: str) -> dict[str, Any]:
    pattern = re.compile(r"^COMPANIES:\s*(.*?)\s*\|\s*SECTORS:\s*(.*?)\s*$", re.IGNORECASE | re.DOTALL)
    match = pattern.match(raw_output.strip())
    if not match:
        return {
            "values": {"affected_companies": [], "affected_sectors": []},
            "valid": False,
            "strict_format_valid": False,
            "errors": ["expected COMPANIES: ... | SECTORS: ..."],
        }
    companies = _parse_semicolon_list(match.group(1))
    sectors = _parse_semicolon_list(match.group(2))
    errors: list[str] = []
    for field, values in (("company", companies), ("sector", sectors)):
        if len(values) != len({value.casefold() for value in values}):
            errors.append(f"duplicate {field} entity")
        for value in values:
            if value.casefold() not in source_text.casefold():
                errors.append(f"{field} entity is not present in the supplied source text: {value!r}")
    strict = raw_output.strip().startswith("COMPANIES:") and " | SECTORS: " in raw_output.strip()
    return {
        "values": {"affected_companies": companies, "affected_sectors": sectors},
        "valid": not errors,
        "strict_format_valid": not errors and strict,
        "errors": errors,
    }


def parse_evidence(
    raw_output: str,
    source_text: str,
    extracted_labels: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pattern = re.compile(
        r"^SCOPE:\s*(.*?)\s*\|\s*DIRECTION:\s*(.*?)\s*\|\s*SURPRISE:\s*(.*?)\s*$",
        re.IGNORECASE | re.DOTALL,
    )
    match = pattern.match(raw_output.strip())
    empty = {"scope": "", "direction": "", "surprise": ""}
    if not match:
        return {
            "values": empty,
            "valid": False,
            "strict_format_valid": False,
            "errors": ["expected SCOPE: ... | DIRECTION: ... | SURPRISE: ..."],
        }
    keys = ("scope", "direction", "surprise")
    values = {
        key: "" if value.strip().upper() == "NONE" else value.strip()
        for key, value in zip(keys, match.groups())
    }
    errors = [
        f"{key} evidence is not an exact source substring"
        for key, value in values.items()
        if value and value not in source_text
    ]
    labels = extracted_labels or {}
    if labels.get("event_scope") not in {None, "unclear"} and not values["scope"]:
        errors.append("classified event_scope requires non-empty scope evidence")
    directional_values = {labels.get("target_direction"), labels.get("sector_direction")}
    if directional_values & {"positive", "negative", "neutral", "mixed"} and not values["direction"]:
        errors.append("classified target/sector direction requires non-empty direction evidence")
    if labels.get("explicit_surprise") in {"positive", "negative", "mixed"} and not values["surprise"]:
        errors.append("classified explicit surprise requires non-empty surprise evidence")
    if labels.get("explicit_surprise") in {"none", "unknown"} and values["surprise"]:
        errors.append("surprise evidence must be empty for none/unknown surprise")
    strict = raw_output.strip().startswith("SCOPE:") and " | DIRECTION: " in raw_output.strip() and " | SURPRISE: " in raw_output.strip()
    return {
        "values": values,
        "valid": not errors,
        "strict_format_valid": not errors and strict,
        "errors": errors,
    }


def parse_pass(
    pass_name: str,
    raw_output: str,
    record: dict[str, Any],
    schema: dict[str, Any],
    extracted_labels: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if pass_name in PASS_FIELDS:
        return parse_closed_pass(raw_output, PASS_FIELDS[pass_name], schema)
    source_text = f"{record['headline']}\n{record['article_text']}"
    if pass_name == "channels":
        return parse_channels(raw_output, schema)
    if pass_name == "entities":
        return parse_entities(raw_output, source_text)
    if pass_name == "evidence":
        return parse_evidence(raw_output, source_text, extracted_labels)
    raise ValueError(f"Unknown pass {pass_name!r}")


def empty_labels() -> dict[str, Any]:
    return {
        "relevance": None,
        "event_scope": None,
        "event_type": None,
        "affected_breadth": None,
        "target_direction": None,
        "sector_direction": None,
        "peer_effect": None,
        "explicit_surprise": None,
        "information_status": None,
        "transmission_channels": [],
        "affected_companies": [],
        "affected_sectors": [],
        "evidence": {"scope": "", "direction": "", "surprise": ""},
        "abstain_reason": None,
    }


def update_labels(labels: dict[str, Any], pass_name: str, parsed_values: Any) -> None:
    if pass_name in PASS_FIELDS:
        labels.update(parsed_values)
    elif pass_name == "channels":
        labels["transmission_channels"] = parsed_values
    elif pass_name == "entities":
        labels.update(parsed_values)
    elif pass_name == "evidence":
        labels["evidence"] = parsed_values


def resolve_device_and_dtype(torch: Any, device_argument: str, precision: str) -> tuple[str, Any]:
    if device_argument == "auto":
        if torch.cuda.is_available():
            device = "cuda"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = device_argument
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device == "mps" and not (getattr(torch.backends, "mps", None) and torch.backends.mps.is_available()):
        raise RuntimeError("MPS was requested but is not available")

    if precision == "auto":
        dtype = torch.float16 if device in {"cuda", "mps"} else torch.float32
    else:
        dtype = getattr(torch, precision)
    if device == "cpu" and dtype == torch.float16:
        raise RuntimeError("float16 CPU inference is unsupported; use float32 or bfloat16")
    return device, dtype


def configure_determinism(torch: Any, set_seed: Callable[[int], None]) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    random.seed(0)
    set_seed(0)
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=False)


def generate_batch(
    prompts: list[str],
    tokenizer: Any,
    model: Any,
    torch: Any,
    device: str,
    max_input_tokens: int,
    max_new_tokens: int,
) -> tuple[list[str], list[bool]]:
    token_lists = tokenizer(prompts, add_special_tokens=True, truncation=False)["input_ids"]
    over_limit = [index for index, tokens in enumerate(token_lists) if len(tokens) > max_input_tokens]
    if over_limit:
        lengths = [len(token_lists[index]) for index in over_limit]
        raise RuntimeError(
            "Refusing to truncate article content or the output contract. "
            f"{len(over_limit)} prompt(s) exceed {max_input_tokens} tokens; lengths={lengths}. "
            "Shorten the supplied article text under a documented preprocessing rule."
        )
    encoded = tokenizer(
        prompts,
        add_special_tokens=True,
        padding=True,
        truncation=False,
        return_tensors="pt",
    )
    encoded = {key: value.to(device) for key, value in encoded.items()}
    with torch.inference_mode():
        output_ids = model.generate(
            **encoded,
            do_sample=False,
            num_beams=1,
            max_new_tokens=max_new_tokens,
            use_cache=True,
        )
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    return [output.strip() for output in outputs], [False] * len(prompts)


def preflight_prompt_lengths(
    records: list[dict[str, Any]],
    passes: tuple[str, ...],
    schema: dict[str, Any],
    tokenizer: Any,
    max_input_tokens: int,
) -> dict[str, int]:
    maxima: dict[str, int] = {}
    violations: list[dict[str, Any]] = []
    for pass_name in passes:
        maximum = 0
        for record in records:
            prompt = build_prompt(pass_name, record, schema)
            token_count = len(
                tokenizer(prompt, add_special_tokens=True, truncation=False)["input_ids"]
            )
            maximum = max(maximum, token_count)
            if token_count > max_input_tokens:
                violations.append(
                    {
                        "article_id": record["article_id"],
                        "pass": pass_name,
                        "token_count": token_count,
                    }
                )
        maxima[pass_name] = maximum
    if violations:
        raise RuntimeError(
            "Prompt preflight failed; refusing to truncate the article or output contract. "
            + json.dumps(violations[:20], ensure_ascii=False)
        )
    return maxima


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def snapshot_file_hashes(snapshot_path: Path) -> dict[str, str]:
    expected_files = (
        "config.json",
        "generation_config.json",
        "model.safetensors",
        "special_tokens_map.json",
        "spiece.model",
        "tokenizer.json",
        "tokenizer_config.json",
    )
    hashes = {
        name: sha256_file(snapshot_path / name)
        for name in expected_files
        if (snapshot_path / name).is_file()
    }
    for required in ("config.json", "model.safetensors", "spiece.model", "tokenizer_config.json"):
        if required not in hashes:
            raise FileNotFoundError(f"Pinned snapshot is missing required file {required}")
    return hashes


def load_existing_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    records = read_jsonl(path)
    ids = [record.get("article_id") for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Cannot resume: duplicate article_id values in {path}")
    return set(ids)


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def require_resume_compatibility(existing: dict[str, Any], current: dict[str, Any]) -> None:
    keys = (
        "prompt_version",
        "prompt_builder_sha256",
        "schema_name",
        "schema_version",
        "schema_sha256",
        "input_sha256",
        "selected_article_ids_sha256",
        "model_id",
        "model_revision",
        "tokenizer_revision",
        "model_files_sha256",
        "mode",
        "passes",
        "device",
        "precision",
        "batch_size",
        "max_input_tokens",
        "preflight_max_prompt_tokens_by_pass",
        "generation",
        "tokenizer",
        "base_prompt_corpus_sha256",
        "runtime",
    )
    mismatches = [key for key in keys if existing.get(key) != current.get(key)]
    if mismatches:
        detail = {key: {"existing": existing.get(key), "current": current.get(key)} for key in mismatches}
        raise RuntimeError(
            "Refusing to resume an incompatible extraction run. Use a new output path or --overwrite. "
            + json.dumps(detail, ensure_ascii=False, sort_keys=True)
        )


def main() -> int:
    args = parse_args()
    schema_text = args.schema.read_text(encoding="utf-8")
    schema = json.loads(schema_text)
    records = read_jsonl(args.input)
    for record in records:
        validate_input_record(record)
    ids = [record["article_id"] for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("Input contains duplicate article_id values")
    if args.limit is not None:
        records = records[: args.limit]
    passes = CORE_PASSES if args.mode == "core" else FULL_PASSES

    for record in records:
        for pass_name in passes:
            prompt = build_prompt(pass_name, record, schema)
            if not prompt.strip() or record["headline"] not in prompt:
                raise ValueError(f"Prompt construction failed for {record['article_id']} pass {pass_name}")

    if args.validate_only:
        print(
            json.dumps(
                {
                    "validated_input_records": len(records),
                    "passes_per_record": len(passes),
                    "total_prompts": len(records) * len(passes),
                    "mode": args.mode,
                    "model_id": args.model_id,
                    "revision": args.revision,
                    "model_was_loaded": False,
                },
                indent=2,
            )
        )
        return 0

    if args.output.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(f"{args.output} exists; pass --resume or --overwrite")
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
            "Missing model dependencies. Install torch for your hardware, then install requirements-llm.txt."
        ) from exc

    device, dtype = resolve_device_and_dtype(torch, args.device, args.precision)
    configure_determinism(torch, set_seed)
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
    model_file_hashes = snapshot_file_hashes(snapshot_path)
    tokenizer = AutoTokenizer.from_pretrained(
        snapshot_path,
        local_files_only=True,
    )
    prompt_token_maxima = preflight_prompt_lengths(
        records,
        passes,
        schema,
        tokenizer,
        args.max_input_tokens,
    )
    model = AutoModelForSeq2SeqLM.from_pretrained(
        snapshot_path,
        local_files_only=True,
        torch_dtype=dtype,
        use_safetensors=True,
    )
    model.to(device)
    model.eval()

    prompt_code = "\n".join(
        (
            inspect.getsource(article_block),
            inspect.getsource(build_prompt),
            json.dumps(MAX_NEW_TOKENS, sort_keys=True),
        )
    )
    prompt_code_hash = sha256_text(prompt_code)
    base_prompt_hashes = {
        pass_name: sha256_text(
            "\n".join(
                f"{record['article_id']}\0{build_prompt(pass_name, record, schema)}"
                for record in records
            )
        )
        for pass_name in passes
    }
    runtime = {
        "python_version": sys.version,
        "torch_version": torch.__version__,
        "transformers_version": transformers.__version__,
        "cuda_runtime_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_name": torch.cuda.get_device_name(0) if device == "cuda" else device,
        "cuda_capability": list(torch.cuda.get_device_capability(0)) if device == "cuda" else None,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "deterministic_algorithms_enforced": True,
    }
    manifest: dict[str, Any] = {
        "status": "in_progress",
        "prompt_version": PROMPT_VERSION,
        "prompt_builder_sha256": prompt_code_hash,
        "schema_name": schema["schema_name"],
        "schema_version": schema["schema_version"],
        "schema_sha256": hashlib.sha256(schema_text.encode("utf-8")).hexdigest(),
        "input_path": str(args.input),
        "input_sha256": sha256_file(args.input),
        "selected_article_ids_sha256": sha256_text("\n".join(record["article_id"] for record in records)),
        "model_id": args.model_id,
        "model_revision": args.revision,
        "tokenizer_revision": args.revision,
        "model_snapshot_path": str(snapshot_path),
        "model_files_sha256": model_file_hashes,
        "mode": args.mode,
        "passes": list(passes),
        "device": device,
        "precision": str(dtype).replace("torch.", ""),
        "batch_size": args.batch_size,
        "max_input_tokens": args.max_input_tokens,
        "preflight_max_prompt_tokens_by_pass": prompt_token_maxima,
        "generation": {
            "do_sample": False,
            "num_beams": 1,
            "seed": 0,
            "use_cache": True,
            "max_new_tokens_by_pass": {name: MAX_NEW_TOKENS[name] for name in passes},
            "skip_special_tokens": True,
            "clean_up_tokenization_spaces": False,
        },
        "tokenizer": {
            "padding": True,
            "padding_side": tokenizer.padding_side,
            "truncation": False,
            "truncation_side": tokenizer.truncation_side,
            "model_max_length": tokenizer.model_max_length,
        },
        "base_prompt_corpus_sha256": base_prompt_hashes,
        "local_files_only": args.local_files_only,
        "runtime": runtime,
    }
    existing_ids: set[str] = set()
    if args.resume:
        if not args.output.exists() or not manifest_path.exists():
            raise FileNotFoundError("--resume requires both an existing prediction file and its manifest")
        existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        require_resume_compatibility(existing_manifest, manifest)
        if existing_manifest.get("status") == "complete" and existing_manifest.get("output_sha256") != sha256_file(args.output):
            raise RuntimeError("Existing prediction file hash does not match its completed manifest")
        existing_ids = load_existing_ids(args.output)
    elif args.overwrite:
        args.output.write_text("", encoding="utf-8")
    pending = [record for record in records if record["article_id"] not in existing_ids]
    manifest["existing_prediction_count"] = len(existing_ids)
    manifest["pending_prediction_count"] = len(pending)
    write_manifest(manifest_path, manifest)

    output_mode = "a" if args.resume else "w"
    primary_valid_count = 0
    full_valid_count = 0
    strict_primary_count = 0
    processed_count = 0
    with args.output.open(output_mode, encoding="utf-8", newline="\n") as output_handle:
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start : start + args.batch_size]
            states = [
                {"record": record, "labels": empty_labels(), "passes": {}, "quality_flags": []}
                for record in batch
            ]
            for pass_name in passes:
                prompts = [
                    build_prompt(pass_name, state["record"], schema, state["labels"])
                    for state in states
                ]
                raw_outputs, truncated_flags = generate_batch(
                    prompts,
                    tokenizer,
                    model,
                    torch,
                    device,
                    args.max_input_tokens,
                    MAX_NEW_TOKENS[pass_name],
                )
                for state, prompt, raw_output, input_truncated in zip(
                    states, prompts, raw_outputs, truncated_flags
                ):
                    parsed = parse_pass(
                        pass_name,
                        raw_output,
                        state["record"],
                        schema,
                        state["labels"],
                    )
                    update_labels(state["labels"], pass_name, parsed["values"])
                    state["passes"][pass_name] = {
                        "raw_output": raw_output,
                        "prompt_sha256": sha256_text(prompt),
                        "valid": parsed["valid"],
                        "strict_format_valid": parsed["strict_format_valid"],
                        "input_truncated": input_truncated,
                        "errors": parsed["errors"],
                    }
                    if not parsed["valid"]:
                        state["quality_flags"].append(f"{pass_name}_invalid")
                    if parsed["valid"] and not parsed["strict_format_valid"]:
                        state["quality_flags"].append(f"{pass_name}_non_strict_format")
                    if input_truncated:
                        state["quality_flags"].append(f"{pass_name}_input_truncated")

            for state in states:
                labels = state["labels"]
                if labels["relevance"] == "insufficient":
                    labels["abstain_reason"] = "FLAN-T5 classified the supplied text as insufficient."
                primary_valid = all(state["passes"][name]["valid"] for name in CORE_PASSES)
                strict_primary = primary_valid and all(
                    state["passes"][name]["strict_format_valid"] for name in CORE_PASSES
                )
                full_valid = all(state["passes"][name]["valid"] for name in passes)
                result = {
                    "row_number": state["record"]["row_number"],
                    "article_id": state["record"]["article_id"],
                    "target_ticker": state["record"]["target"]["ticker"],
                    "extractor": args.model_id,
                    "model_revision": args.revision,
                    "protocol_version": schema["schema_version"],
                    "labels": labels,
                    "quality_flags": state["quality_flags"],
                    "validity": {
                        "primary_closed_labels_valid": primary_valid,
                        "primary_strict_format_valid": strict_primary,
                        "all_requested_passes_valid": full_valid,
                    },
                    "passes": state["passes"],
                }
                output_handle.write(json.dumps(result, ensure_ascii=False, separators=(",", ":")) + "\n")
                output_handle.flush()
                processed_count += 1
                primary_valid_count += int(primary_valid)
                strict_primary_count += int(strict_primary)
                full_valid_count += int(full_valid)
            print(
                f"Processed {min(start + len(batch), len(pending))}/{len(pending)} pending records",
                file=sys.stderr,
                flush=True,
            )

    manifest.update(
        {
            "status": "complete",
            "new_prediction_count": processed_count,
            "total_prediction_count": len(existing_ids) + processed_count,
            "new_primary_valid_count": primary_valid_count,
            "new_primary_strict_format_count": strict_primary_count,
            "new_all_requested_passes_valid_count": full_valid_count,
            "output_sha256": sha256_file(args.output),
        }
    )
    write_manifest(manifest_path, manifest)
    print(f"Wrote {processed_count} new predictions to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
