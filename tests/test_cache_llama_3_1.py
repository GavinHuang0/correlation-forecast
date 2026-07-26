from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
SCRIPT = SCRIPTS / "cache_llama_3_1.py"
SPEC = importlib.util.spec_from_file_location("cache_llama31_for_test", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
cache = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = cache
SPEC.loader.exec_module(cache)


class CacheLlama31Tests(unittest.TestCase):
    def test_model_and_revision_are_immutable(self) -> None:
        self.assertEqual(cache.MODEL_ID, "meta-llama/Llama-3.1-8B-Instruct")
        self.assertRegex(cache.MODEL_REVISION, r"^[0-9a-f]{40}$")

    def test_operational_context_is_bounded(self) -> None:
        self.assertEqual(cache.CONSERVATIVE_DATA_CUTOFF, "2023-12-31")
        self.assertEqual(cache.OPERATIONAL_CONTEXT_LIMIT, 1024)
        self.assertGreater(cache.MODEL_CONTEXT_WINDOW, cache.OPERATIONAL_CONTEXT_LIMIT)


if __name__ == "__main__":
    unittest.main()
