from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_massive_length_stratified_benchmark.py"
SPEC = importlib.util.spec_from_file_location("massive_length_builder", MODULE_PATH)
assert SPEC and SPEC.loader
builder = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(builder)


def test_length_stratum_boundaries() -> None:
    assert builder.length_stratum("") == "headline_only"
    assert builder.length_stratum("x") == "description_001_149"
    assert builder.length_stratum("x" * 149) == "description_001_149"
    assert builder.length_stratum("x" * 150) == "description_150_299"
    assert builder.length_stratum("x" * 299) == "description_150_299"
    assert builder.length_stratum("x" * 300) == "description_300_599"
    assert builder.length_stratum("x" * 599) == "description_300_599"
    assert builder.length_stratum("x" * 600) == "description_600_plus"


def test_model_input_keeps_headline_and_removes_only_description() -> None:
    parent = {
        "provider_article_id": "abc",
        "time_published_utc": "2024-01-02T12:00:00Z",
        "source": "Example",
        "headline": "AMD reports an update",
        "description": "A sufficiently descriptive provider summary.",
        "description_char_count": 42,
        "length_stratum": "description_001_149",
        "vendor_tickers": ["AMD"],
        "target": {
            "ticker": "AMD",
            "company": "Advanced Micro Devices",
            "sector": "Semiconductors",
            "sector_benchmark": "SOXX",
            "known_sector_peers": ["NVDA"],
        },
    }
    native = builder.model_input(parent, 1, "native_description")
    ablation = builder.model_input(parent, 2, "headline_only_ablation")
    assert native["headline"] == ablation["headline"]
    assert native["article_text"] == parent["description"]
    assert ablation["article_text"] == ""
    assert native["target"] == ablation["target"]
    assert native["article_id"] != ablation["article_id"]


def test_bounded_description_is_explicit_and_audited() -> None:
    source = ("word " * 200).strip()
    retained, audit = builder.bounded_description(source)
    assert len(retained) <= builder.DESCRIPTION_EXCERPT_MAX_CHARS
    assert audit["source_description_char_count"] == len(source)
    assert audit["retained_description_char_count"] == len(retained)
    assert audit["omitted_description_char_count"] == len(source) - len(retained)
    assert audit["description_was_bounded"] is True
