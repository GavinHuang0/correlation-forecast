"""Combine rung summaries into a reproducible cross-rung comparison."""

from __future__ import annotations

import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

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


def main() -> int:
    rows = []
    for rung in ("rung_01", "rung_02", "rung_03", "rung_04"):
        path = common.EXPERIMENT_ROOT / rung / "summary.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
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
        common.PANEL_PATH,
        common.PROTOCOL_PATH,
        *(
            path
            for rung in ("rung_01", "rung_02", "rung_03", "rung_04")
            for path in (
                common.OUTPUT_ROOT / rung / "predictions.parquet",
                common.OUTPUT_ROOT / rung / "fold_metrics.json",
                common.OUTPUT_ROOT / rung / "fits.json",
                common.EXPERIMENT_ROOT / rung / "review.json",
                common.EXPERIMENT_ROOT / rung / "summary.json",
            )
        ),
    ]
    artifact_hashes = {
        path.as_posix(): sha256_file(path)
        for path in artifact_paths
        if path.exists()
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
    output = common.EXPERIMENT_ROOT / "comparisons" / "summary.json"
    common.write_json(output, payload)
    common.write_parquet(
        common.OUTPUT_ROOT / "comparisons" / "all_metrics.parquet", metrics
    )
    print(best.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
