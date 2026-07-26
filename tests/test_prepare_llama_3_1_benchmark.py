from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "prepare_llama_3_1_benchmark.py"
SPEC = importlib.util.spec_from_file_location("prepare_llama31_for_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
prepare = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = prepare
SPEC.loader.exec_module(prepare)


class PrepareLlama31BenchmarkTests(unittest.TestCase):
    def test_manifest_is_rebranded_and_cutoff_is_frozen(self) -> None:
        result = prepare.model_specific_manifest(
            {
                "manifest_version": "old",
                "dataset_name": "old",
                "model_eligibility_basis": {},
            }
        )
        self.assertEqual(result["manifest_version"], "llama-3.1-benchmark-v1")
        self.assertEqual(result["knowledge_cutoff"], "2023-12-31")
        self.assertEqual(
            result["model_eligibility_basis"]["model_family"], "Llama 3.1"
        )


if __name__ == "__main__":
    unittest.main()
