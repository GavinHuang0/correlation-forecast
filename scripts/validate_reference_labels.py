from __future__ import annotations

import argparse
import glob
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable


REQUIRED_LABEL_KEYS = {
    "relevance",
    "event_scope",
    "event_type",
    "affected_breadth",
    "target_direction",
    "sector_direction",
    "peer_effect",
    "explicit_surprise",
    "information_status",
    "transmission_channels",
    "affected_companies",
    "affected_sectors",
    "evidence",
    "abstain_reason",
}
REQUIRED_EVIDENCE_KEYS = {"scope", "direction", "surprise"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate and merge GPT-5.6 Sol silver-reference JSONL annotation parts."
    )
    parser.add_argument("--inputs", required=True, type=Path, help="Blinded all_inputs.jsonl file")
    parser.add_argument("--parts", required=True, type=Path, help="Directory containing batch JSONL files")
    parser.add_argument("--schema", default=Path("config/news_feature_schema.json"), type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each JSONL record must be an object")
            records.append(value)
    return records


def exact_substring(evidence: str, source: str) -> bool:
    return not evidence or evidence in source


def add_issue(issues: list[dict[str, Any]], severity: str, article_id: str, message: str) -> None:
    issues.append({"severity": severity, "article_id": article_id, "message": message})


def validate_record(
    record: dict[str, Any],
    input_record: dict[str, Any],
    schema: dict[str, Any],
    issues: list[dict[str, Any]],
) -> None:
    article_id = str(record.get("article_id", "<missing>"))
    if record.get("row_number") != input_record["row_number"]:
        add_issue(issues, "error", article_id, "row_number does not match the blinded input")
    if record.get("target_ticker") != input_record["target"]["ticker"]:
        add_issue(issues, "error", article_id, "target_ticker does not match the blinded input")
    if record.get("annotator") != "gpt-5.6-sol":
        add_issue(issues, "error", article_id, "annotator must be gpt-5.6-sol")
    if record.get("protocol_version") != schema["schema_version"]:
        add_issue(issues, "error", article_id, "protocol_version does not match schema_version")
    if not isinstance(record.get("quality_flags"), list) or not all(
        isinstance(item, str) for item in record.get("quality_flags", [])
    ):
        add_issue(issues, "error", article_id, "quality_flags must be an array of strings")

    labels = record.get("labels")
    if not isinstance(labels, dict):
        add_issue(issues, "error", article_id, "labels must be an object")
        return
    if set(labels) != REQUIRED_LABEL_KEYS:
        add_issue(
            issues,
            "error",
            article_id,
            f"labels keys differ from canonical schema: {sorted(set(labels) ^ REQUIRED_LABEL_KEYS)}",
        )

    for field, allowed_values in schema["closed_label_fields"].items():
        if labels.get(field) not in allowed_values:
            add_issue(
                issues,
                "error",
                article_id,
                f"{field} has invalid value {labels.get(field)!r}",
            )

    channels = labels.get("transmission_channels")
    allowed_channels = set(schema["transmission_channels"]["allowed_values"])
    max_channels = schema["transmission_channels"]["max_items"]
    if not isinstance(channels, list) or not all(isinstance(item, str) for item in channels):
        add_issue(issues, "error", article_id, "transmission_channels must be an array of strings")
    else:
        if len(channels) > max_channels:
            add_issue(issues, "error", article_id, f"more than {max_channels} transmission channels")
        if len(channels) != len(set(channels)):
            add_issue(issues, "error", article_id, "transmission_channels contains duplicates")
        invalid_channels = sorted(set(channels) - allowed_channels)
        if invalid_channels:
            add_issue(issues, "error", article_id, f"invalid transmission channels: {invalid_channels}")

    for field in ("affected_companies", "affected_sectors"):
        values = labels.get(field)
        if not isinstance(values, list) or not all(isinstance(item, str) and item.strip() for item in values):
            add_issue(issues, "error", article_id, f"{field} must be an array of non-empty strings")
        elif len(values) != len({item.casefold() for item in values}):
            add_issue(issues, "error", article_id, f"{field} contains duplicates")

    evidence = labels.get("evidence")
    source_text = f"{input_record['headline']}\n{input_record['article_text']}"
    if not isinstance(evidence, dict) or set(evidence) != REQUIRED_EVIDENCE_KEYS:
        add_issue(issues, "error", article_id, "evidence must contain scope, direction, and surprise")
    else:
        for field, snippet in evidence.items():
            if not isinstance(snippet, str):
                add_issue(issues, "error", article_id, f"evidence.{field} must be a string")
            elif not exact_substring(snippet, source_text):
                add_issue(issues, "error", article_id, f"evidence.{field} is not an exact source substring")
            elif len(snippet) > 240:
                add_issue(issues, "warning", article_id, f"evidence.{field} is longer than 240 characters")

        if labels.get("event_scope") != "unclear" and not evidence.get("scope"):
            add_issue(issues, "warning", article_id, "classified event_scope has no scope evidence")
        directional = {labels.get("target_direction"), labels.get("sector_direction")}
        if directional & {"positive", "negative", "neutral", "mixed"} and not evidence.get("direction"):
            add_issue(issues, "warning", article_id, "classified direction has no direction evidence")
        if labels.get("explicit_surprise") in {"positive", "negative", "mixed"} and not evidence.get("surprise"):
            add_issue(issues, "error", article_id, "explicit surprise has no exact surprise evidence")
        if labels.get("explicit_surprise") in {"none", "unknown"} and evidence.get("surprise"):
            add_issue(issues, "warning", article_id, "surprise evidence is populated for none/unknown")

    abstain_reason = labels.get("abstain_reason")
    if abstain_reason is not None and (not isinstance(abstain_reason, str) or not abstain_reason.strip()):
        add_issue(issues, "error", article_id, "abstain_reason must be null or a non-empty string")
    if labels.get("relevance") == "insufficient" and abstain_reason is None:
        add_issue(issues, "warning", article_id, "insufficient relevance should include abstain_reason")

    if labels.get("relevance") == "irrelevant":
        for field in ("target_direction", "sector_direction", "peer_effect"):
            if labels.get(field) != "not_applicable":
                add_issue(issues, "warning", article_id, f"irrelevant article has {field}={labels.get(field)!r}")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def label_distributions(records: Iterable[dict[str, Any]], fields: Iterable[str]) -> dict[str, dict[str, int]]:
    return {
        field: dict(sorted(Counter(record["labels"][field] for record in records).items()))
        for field in fields
    }


def main() -> int:
    args = parse_args()
    schema = json.loads(args.schema.read_text(encoding="utf-8"))
    inputs = read_jsonl(args.inputs)
    input_by_id = {record["article_id"]: record for record in inputs}

    part_paths = sorted(Path(path) for path in glob.glob(str(args.parts / "*.jsonl")))
    if not part_paths:
        raise ValueError(f"No JSONL files found in {args.parts}")
    records = [record for path in part_paths for record in read_jsonl(path)]
    issues: list[dict[str, Any]] = []

    counts = Counter(record.get("article_id") for record in records)
    for article_id, count in counts.items():
        if count != 1:
            add_issue(issues, "error", str(article_id), f"article_id appears {count} times")
    missing = sorted(set(input_by_id) - set(counts))
    extra = sorted(set(counts) - set(input_by_id))
    for article_id in missing:
        add_issue(issues, "error", article_id, "missing reference annotation")
    for article_id in extra:
        add_issue(issues, "error", article_id, "annotation does not exist in blinded inputs")

    for record in records:
        article_id = record.get("article_id")
        if article_id in input_by_id:
            validate_record(record, input_by_id[article_id], schema, issues)

    errors = [issue for issue in issues if issue["severity"] == "error"]
    warnings = [issue for issue in issues if issue["severity"] == "warning"]
    valid_records = sorted(
        (record for record in records if record.get("article_id") in input_by_id),
        key=lambda record: record.get("row_number", 10**9),
    )

    report = {
        "input_count": len(inputs),
        "annotation_count": len(records),
        "unique_annotation_count": len(counts),
        "error_count": len(errors),
        "warning_count": len(warnings),
        "part_files": [str(path) for path in part_paths],
        "schema_version": schema["schema_version"],
        "label_distributions": label_distributions(valid_records, schema["closed_label_fields"].keys())
        if valid_records and not errors
        else {},
        "issues": issues,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")

    if errors:
        print(f"Reference validation failed: {len(errors)} errors, {len(warnings)} warnings", file=sys.stderr)
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="\n") as handle:
        for record in valid_records:
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")

    report["output_sha256"] = sha256(args.output)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Validated and merged {len(valid_records)} annotations: {len(warnings)} warnings")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
