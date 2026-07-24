from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import extract_flan_t5 as base
import extract_flan_t5_coarse as coarse_runner


MODEL_ID = "google/flan-t5-xl"
MODEL_REVISION = "7d6315df2c2fb742f0f5b556879d730926ca9001"
EXPERIMENT_ID = "flan-t5-xl-v1.0-development"
SNAPSHOT_MANIFEST = Path("outputs/flan_t5_xl/model_snapshot_manifest.json")


def option_value(arguments: list[str], option: str) -> str | None:
    for index, argument in enumerate(arguments):
        if argument == option:
            if index + 1 >= len(arguments):
                raise ValueError(f"{option} requires a value")
            return arguments[index + 1]
        if argument.startswith(option + "="):
            return argument.split("=", 1)[1]
    return None


def inject_pinned_arguments(arguments: list[str]) -> list[str]:
    result = list(arguments)
    supplied_model = option_value(result, "--model-id")
    supplied_revision = option_value(result, "--revision")
    if supplied_model not in {None, MODEL_ID}:
        raise ValueError(f"--model-id is pinned to {MODEL_ID}")
    if supplied_revision not in {None, MODEL_REVISION}:
        raise ValueError(f"--revision is pinned to {MODEL_REVISION}")
    if supplied_model is None:
        result.extend(["--model-id", MODEL_ID])
    if supplied_revision is None:
        result.extend(["--revision", MODEL_REVISION])
    if "--validate-only" not in result and "--local-files-only" not in result:
        result.append("--local-files-only")
    return result


def low_memory_score_closed_label_batch(
    prompts: list[str],
    allowed_values: list[str],
    candidate_outputs: list[str],
    tokenizer: Any,
    model: Any,
    torch: Any,
    device: str,
    max_input_tokens: int,
) -> tuple[list[str], list[bool], list[dict[str, float]]]:
    """Score one label candidate at a time to keep XL activation memory small."""
    if len(candidate_outputs) != len(allowed_values):
        raise ValueError("candidate_outputs must align one-to-one with allowed_values")
    token_lists = tokenizer(
        prompts, add_special_tokens=True, truncation=False
    )["input_ids"]
    over_limit = [
        index for index, tokens in enumerate(token_lists)
        if len(tokens) > max_input_tokens
    ]
    if over_limit:
        lengths = [len(token_lists[index]) for index in over_limit]
        raise RuntimeError(
            "Refusing to truncate article content or the output contract. "
            f"{len(over_limit)} prompt(s) exceed {max_input_tokens} tokens; "
            f"lengths={lengths}."
        )

    selected: list[str] = []
    score_maps: list[dict[str, float]] = []
    for prompt in prompts:
        scores: dict[str, float] = {}
        encoded = tokenizer(
            prompt,
            add_special_tokens=True,
            truncation=False,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        for value, candidate in zip(allowed_values, candidate_outputs):
            targets = tokenizer(
                candidate,
                add_special_tokens=True,
                truncation=False,
                return_tensors="pt",
            )
            labels = targets["input_ids"].to(device)
            labels = labels.masked_fill(targets["attention_mask"].to(device).eq(0), -100)
            with torch.inference_mode():
                logits = model(**encoded, labels=labels).logits.float()
                log_probs = torch.log_softmax(logits, dim=-1)
                safe_labels = labels.masked_fill(labels.eq(-100), 0)
                token_log_probs = log_probs.gather(
                    -1, safe_labels.unsqueeze(-1)
                ).squeeze(-1)
                label_mask = labels.ne(-100)
                mean_log_prob = (
                    (token_log_probs * label_mask).sum(dim=1)
                    / label_mask.sum(dim=1)
                )
            scores[value] = float(mean_log_prob[0].detach().cpu())
        selected.append(max(allowed_values, key=lambda value: scores[value]))
        score_maps.append(scores)
    return selected, [False] * len(prompts), score_maps


def validate_snapshot_manifest() -> None:
    if not SNAPSHOT_MANIFEST.is_file():
        raise FileNotFoundError(
            f"{SNAPSHOT_MANIFEST} is missing. Run scripts/cache_flan_t5_xl.py first."
        )
    manifest = json.loads(SNAPSHOT_MANIFEST.read_text(encoding="utf-8"))
    if manifest.get("model_id") != MODEL_ID:
        raise ValueError("FLAN-T5-XL snapshot manifest has the wrong model_id")
    if manifest.get("model_revision") != MODEL_REVISION:
        raise ValueError("FLAN-T5-XL snapshot manifest has the wrong revision")


def output_path(arguments: list[str]) -> Path | None:
    value = option_value(arguments, "--output")
    return Path(value) if value else None


def append_launcher_provenance(prediction_path: Path) -> None:
    manifest_path = prediction_path.with_suffix(prediction_path.suffix + ".manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(
        {
            "experiment_id": EXPERIMENT_ID,
            "launcher_source_sha256": base.sha256_file(Path(__file__).resolve()),
            "candidate_scoring_microbatch_size": 1,
            "accuracy_preserving_quantization": None,
            "snapshot_manifest_path": str(SNAPSHOT_MANIFEST),
            "snapshot_manifest_sha256": base.sha256_file(SNAPSHOT_MANIFEST),
        }
    )
    base.write_manifest(manifest_path, manifest)


def main() -> int:
    arguments = inject_pinned_arguments(sys.argv[1:])
    validate_only = "--validate-only" in arguments
    if not validate_only:
        validate_snapshot_manifest()

    # Reuse the frozen v0.4 prompts and hierarchy without changing the active
    # FLAN-T5-Large source. Only candidate-score microbatching is replaced.
    base.score_closed_label_batch = low_memory_score_closed_label_batch
    sys.argv = [sys.argv[0], *arguments]
    result = coarse_runner.main()
    prediction_path = output_path(arguments)
    if result == 0 and not validate_only and prediction_path is not None:
        append_launcher_provenance(prediction_path)
    return result


if __name__ == "__main__":
    raise SystemExit(main())
