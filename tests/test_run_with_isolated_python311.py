from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_with_isolated_python311.py"
SPEC = importlib.util.spec_from_file_location("isolated_python_launcher", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = launcher
SPEC.loader.exec_module(launcher)


class IsolatedPythonLauncherTests(unittest.TestCase):
    def test_remainder_separator_is_removed(self) -> None:
        args = launcher.parse_args(
            [
                "--site-packages",
                ".venv-example/Lib/site-packages",
                "--script",
                "scripts/example.py",
                "--",
                "--input",
                "input.jsonl",
            ]
        )
        self.assertEqual(args.script_args, ["--input", "input.jsonl"])

    def test_repository_path_resolves_relative_input(self) -> None:
        self.assertEqual(
            launcher.repository_path(Path("scripts/example.py")),
            (ROOT / "scripts" / "example.py").resolve(),
        )


if __name__ == "__main__":
    unittest.main()
