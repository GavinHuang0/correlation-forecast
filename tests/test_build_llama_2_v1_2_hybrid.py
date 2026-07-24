from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))


def load_module(filename: str, name: str):
    path = SCRIPTS / filename
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load_module(
    "build_llama_2_v1_2_hybrid.py", "llama_2_v1_2_builder_for_test"
)


def source_row(*, corrected: bool) -> dict:
    labels = {
        "shock_scope": "common" if corrected else "idiosyncratic",
        "event_family": "product_demand" if corrected else "macro_market",
        "information_status": "anticipated" if corrected else "confirmed",
        "directional_alignment": (
            "common_direction_unclear" if corrected else "single_firm_only"
        ),
    }
    return {
        "row_number": 1,
        "article_id": "example",
        "target_ticker": "AMD",
        "extractor": builder.v1_0.MODEL_DEFAULT,
        "model_revision": builder.v1_0.REVISION_DEFAULT,
        "prompt_version": (
            builder.v1_1.PROMPT_VERSION
            if corrected
            else builder.v1_0.PROMPT_VERSION
        ),
        "protocol_version": "0.2.0",
        "semantic_applicable": True,
        "deterministic_features": {"same": True},
        "labels": labels,
        "label_origins": {field: "model" for field in labels},
        "passes": {
            field: {"origin": "model", "value": value}
            for field, value in labels.items()
        },
        "validity": {"no_input_truncation": True},
    }


class LlamaV12HybridBuilderTests(unittest.TestCase):
    def test_field_source_is_exact_and_auditable(self) -> None:
        result = builder.build_hybrid_rows(
            [source_row(corrected=False)], [source_row(corrected=True)]
        )[0]
        self.assertEqual(result["labels"]["shock_scope"], "idiosyncratic")
        self.assertEqual(result["labels"]["event_family"], "product_demand")
        self.assertEqual(result["labels"]["information_status"], "anticipated")
        self.assertEqual(
            result["labels"]["directional_alignment"], "single_firm_only"
        )
        self.assertEqual(
            result["passes"]["event_family"]["hybrid_source"], "v1_1"
        )
        self.assertEqual(
            result["passes"]["shock_scope"]["hybrid_source"], "v1_0"
        )

    def test_mismatched_deterministic_features_are_rejected(self) -> None:
        old = source_row(corrected=False)
        corrected = source_row(corrected=True)
        corrected["deterministic_features"] = {"same": False}
        with self.assertRaisesRegex(ValueError, "deterministic_features"):
            builder.build_hybrid_rows([old], [corrected])


if __name__ == "__main__":
    unittest.main()
