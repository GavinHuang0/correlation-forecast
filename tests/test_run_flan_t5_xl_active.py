from __future__ import annotations

import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_flan_t5_xl_active.py"
SPEC = importlib.util.spec_from_file_location("flan_xl_active_for_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
runner = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = runner
SPEC.loader.exec_module(runner)


class FlanT5XlActiveRunnerTests(unittest.TestCase):
    def test_checked_in_active_contract_is_valid(self) -> None:
        contract, schema, calibration = runner.load_active_contract()
        self.assertEqual(contract["extractor_id"], runner.EXPECTED_EXTRACTOR_ID)
        self.assertTrue(schema.is_file())
        self.assertTrue(calibration.is_file())

    def test_default_raw_path_is_distinct(self) -> None:
        output = Path("outputs/news/features.jsonl")
        self.assertEqual(
            runner.default_raw_path(output),
            Path("outputs/news/features.raw.jsonl"),
        )

    def test_raw_command_freezes_runtime_configuration(self) -> None:
        command = runner.raw_command(
            input_path=Path("input.jsonl"),
            raw_output=Path("raw.jsonl"),
            schema_path=Path("schema.json"),
            limit=12,
            overwrite=True,
            validate_only=False,
        )
        rendered = " ".join(command)
        self.assertIn("--decoding order_averaged_letter_score", rendered)
        self.assertIn("--prompt-profile zero_shot", rendered)
        self.assertIn("--device cuda", rendered)
        self.assertIn("--precision float16", rendered)
        self.assertIn("--batch-size 1", rendered)
        self.assertIn("--limit 12", rendered)
        self.assertIn("--overwrite", command)

    def test_contract_rejects_changed_calibration(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            schema = root / "schema.json"
            calibration = root / "calibration.json"
            raw_runner = root / "raw.py"
            postprocessor = root / "post.py"
            schema.write_text("{}\n", encoding="utf-8")
            calibration.write_text("{}\n", encoding="utf-8")
            raw_runner.write_text("# raw\n", encoding="utf-8")
            postprocessor.write_text("# post\n", encoding="utf-8")
            pointer = root / "active.json"
            pointer.write_text(
                json.dumps(
                    {
                        "extractor_id": runner.EXPECTED_EXTRACTOR_ID,
                        "status": "active_research_extractor",
                        "production_ready": False,
                        "model": {
                            "id": runner.EXPECTED_MODEL_ID,
                            "revision": runner.EXPECTED_MODEL_REVISION,
                        },
                        "pipeline": {
                            "mode": "development_selected_score_calibration",
                            "raw_runner": {
                                "path": "raw.py",
                                "sha256": runner.sha256_file(raw_runner),
                            },
                            "postprocessor": {
                                "path": "post.py",
                                "sha256": runner.sha256_file(postprocessor),
                            },
                            "schema": {
                                "path": "schema.json",
                                "sha256": runner.sha256_file(schema),
                            },
                            "calibration": {
                                "path": "calibration.json",
                                "sha256": "0" * 64,
                            },
                        },
                    }
                ),
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "hash changed"):
                runner.load_active_contract(pointer, root)


if __name__ == "__main__":
    unittest.main()
