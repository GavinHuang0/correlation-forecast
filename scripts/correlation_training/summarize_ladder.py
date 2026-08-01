"""Combine rung summaries into a reproducible cross-rung comparison."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.correlation_training import training_common as common  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parser() -> argparse.ArgumentParser:
    output = argparse.ArgumentParser(description=__doc__)
    output.add_argument("--panel", type=Path)
    output.add_argument("--protocol", type=Path, default=common.PROTOCOL_PATH)
    output.add_argument("--experiment-root", type=Path)
    output.add_argument("--output-root", type=Path)
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = parser().parse_args(argv)
    protocol = common.load_protocol(args.protocol)
    paths = common.resolve_training_paths(
        protocol,
        panel=args.panel,
        experiment_root=args.experiment_root,
        output_root=args.output_root,
    )
    rows = []
    for rung in ("rung_01", "rung_02", "rung_03", "rung_04"):
        path = paths.experiment_root / rung / "summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("status") != "complete":
            raise ValueError(f"{rung} summary is not complete")
        review_path = paths.experiment_root / rung / "review.json"
        review = json.loads(review_path.read_text(encoding="utf-8"))
        if review.get("status") != "passed":
            raise ValueError(f"{rung} review did not pass")
        for record in payload["metrics"]:
            rows.append({"rung": rung, **record})
    metrics = pd.DataFrame(rows)
    best = (
        metrics.sort_values(
            ["target", "fisher_z_rmse", "raw_correlation_rmse", "rung", "model"]
        )
        .groupby("target", as_index=False)
        .first()
    )
    rung_best = (
        metrics.sort_values(["rung", "target", "fisher_z_rmse"])
        .groupby(["rung", "target"], as_index=False)
        .first()
    )
    artifact_paths = [
        paths.panel,
        paths.dcc_history_panel,
        paths.dcc_calendar,
        args.protocol,
        *(
            path
            for rung in ("rung_01", "rung_02", "rung_03", "rung_04")
            for path in (
                paths.output_root / rung / "predictions.parquet",
                paths.output_root / rung / "fold_metrics.json",
                paths.output_root / rung / "fits.json",
                paths.experiment_root / rung / "review.json",
                paths.experiment_root / rung / "summary.json",
            )
        ),
        paths.output_root / "rung_03" / "ensemble_diagnostics.json",
        paths.output_root / "rung_04" / "failures.json",
    ]
    missing_artifacts = [path for path in artifact_paths if not path.is_file()]
    if missing_artifacts:
        raise FileNotFoundError(
            "Required ladder artifacts are missing: "
            + ", ".join(str(path) for path in missing_artifacts)
        )
    failures = json.loads(
        (paths.output_root / "rung_04" / "failures.json").read_text(
            encoding="utf-8"
        )
    )
    if failures:
        raise ValueError("Rung 4 failure artifact is not empty")
    artifact_hashes = {
        path.as_posix(): sha256_file(path) for path in artifact_paths
    }
    payload = {
        "status": "complete",
        "generated_at_utc": datetime.now(UTC).isoformat().replace("+00:00", "Z"),
        "selection_metric": (
            "aggregate outer-test Fisher-z RMSE used for development ranking"
        ),
        "result_role": (
            "development model comparison; a future untouched period is "
            "required for an unbiased confirmatory estimate"
        ),
        "best_overall": best.to_dict(orient="records"),
        "best_by_rung": rung_best.to_dict(orient="records"),
        "all_metrics": metrics.to_dict(orient="records"),
        "artifact_sha256": artifact_hashes,
        "interpretation_guardrails": [
            "ETF is tradable hedge coupling; LOO is an equal-weight five-peer factor.",
            "T2 is a component-aggregated five-session target, not an average of daily correlations.",
            "The outer blocks were used to rank rungs, so the winning result is a development estimate rather than an untouched confirmatory estimate.",
            "No statistical significance claim is made yet.",
            "DCC uses daily RTH returns as an established conditional-correlation benchmark and does not directly model intraday realized covariance.",
        ],
    }
    output = paths.experiment_root / "comparisons" / "summary.json"
    common.write_json(output, payload)
    common.write_parquet(
        paths.output_root / "comparisons" / "all_metrics.parquet", metrics
    )
    print(best.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
