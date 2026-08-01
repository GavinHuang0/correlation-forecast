"""Frozen names and control definitions for the v3 W17-Lite ladder."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROTOCOL_PATH = Path("config/quant_deterministic_news_protocol_v3.json")
PROTOCOL_SIDECAR = Path(
    "config/quant_deterministic_news_protocol_v3.sha256"
)

BASE_PANEL_PATH = Path(
    "data/features/q_plus_d/massive_v3/"
    "modeling_panel_q_d2_wlite.parquet"
)
BASE_PANEL_MANIFEST = BASE_PANEL_PATH.parent / "manifest.json"
LONG_PANEL_PATH = Path(
    "data/features/q_plus_d/massive_v3/"
    "modeling_panel_q_d2_wlite_long_description.parquet"
)
PERMUTED_PANEL_PATH = Path(
    "data/features/q_plus_d/massive_v3/"
    "modeling_panel_q_d2_wlite_permuted.parquet"
)

DAILY_ROOT = Path(
    "data/features/news_semantic/massive_v3/flan_w17_lite_k16"
)
DAILY_PATH = DAILY_ROOT / "daily_features.parquet"
DAILY_MANIFEST = DAILY_ROOT / "daily_features.manifest.json"
LONG_DAILY_PATH = DAILY_ROOT / "daily_features_long_description.parquet"
LONG_DAILY_MANIFEST = (
    DAILY_ROOT / "daily_features_long_description.manifest.json"
)
PERMUTED_DAILY_PATH = DAILY_ROOT / "daily_features_permuted.parquet"
PERMUTED_DAILY_MANIFEST = (
    DAILY_ROOT / "daily_features_permuted.manifest.json"
)
PREDICTIONS_PATH = DAILY_ROOT / "predictions.jsonl"
PREDICTIONS_MANIFEST = DAILY_ROOT / "predictions.jsonl.manifest.json"
AGGREGATION_CONFIG = Path("config/flan_w17_lite_aggregation_v1.json")
QUANT_V1_PROTOCOL = Path("config/quant_training_protocol_v1.json")
QUANT_V1_XGBOOST_IMPLEMENTATION = Path(
    "scripts/correlation_training/run_rung_03.py"
)

BOUND_TRAINING_IMPLEMENTATION = (
    Path("scripts/quant_deterministic_news_training_v3/contract.py"),
    Path("scripts/quant_deterministic_news_training_v3/common.py"),
    Path("scripts/quant_deterministic_news_training_v3/run_linear.py"),
    Path("scripts/quant_deterministic_news_training_v3/run_xgboost.py"),
    Path("scripts/quant_deterministic_news_training_v3/lock_protocol.py"),
    Path(
        "scripts/quant_deterministic_news_training_v3/"
        "run_final_comparison.py"
    ),
    Path("scripts/quant_deterministic_news_training_v2/common.py"),
    Path("scripts/correlation_training/training_common.py"),
    Path("scripts/correlation_training/build_modeling_panel.py"),
)

EXPERIMENT_ROOT = Path("experiments/quant_deterministic_news/v3/training")
OUTPUT_ROOT = Path("outputs/quant_deterministic_news/v3")
STATUS_PATH = EXPERIMENT_ROOT / "status.json"
STATUS_MARKDOWN_PATH = EXPERIMENT_ROOT / "STATUS.md"

PANEL_KEYS = ("forecast_date", "sector", "stock", "benchmark")
PREDICTION_KEYS = ("fold", "target", *PANEL_KEYS)
TARGETS = ("t1_etf", "t1_loo", "t2_etf", "t2_loo")

WLITE_17 = (
    "wlite_event_share_firm_operating_financial",
    "wlite_event_share_policy_corporate",
    "wlite_event_share_macro_market",
    "wlite_route_share_target_idiosyncratic",
    "wlite_route_share_peer_idiosyncratic",
    "wlite_rule_status_cue_share_confirmed_action",
    "wlite_rule_status_cue_share_scheduled_expected",
    "wlite_rule_status_cue_share_rumor_unconfirmed",
    "wlite_rule_status_cue_share_analysis_opinion",
    "wlite_rule_status_cue_conflict_weight_share",
    "wlite_selection_weight_coverage",
    "wlite_selected_headline_only_weight_share",
    "wlite_selected_sub150_weight_share",
    "wlite_event_order_disagreement_weight_share",
    "wlite_observed_no_selected_article",
    "wlite_event_entropy_accepted",
    "wlite_selected_weight_hhi",
)

# The FLAN-generated event content.  The stale control shifts these three
# shares and the entropy derived from them while retaining contemporaneous
# routing, rule cues, selection coverage, and text-quality measurements.
EVENT_CONTENT_4 = (
    "wlite_event_share_firm_operating_financial",
    "wlite_event_share_policy_corporate",
    "wlite_event_share_macro_market",
    "wlite_event_entropy_accepted",
)

# These fields contain event/routing/rule content.  The wrong-stock control
# rotates this entire block while retaining the six availability, selection,
# and text-quality diagnostics below.  The date-sector permutation control is
# deliberately narrower: it changes EVENT_CONTENT_4 only.
SEMANTIC_CONTENT_11 = (
    "wlite_event_share_firm_operating_financial",
    "wlite_event_share_policy_corporate",
    "wlite_event_share_macro_market",
    "wlite_route_share_target_idiosyncratic",
    "wlite_route_share_peer_idiosyncratic",
    "wlite_rule_status_cue_share_confirmed_action",
    "wlite_rule_status_cue_share_scheduled_expected",
    "wlite_rule_status_cue_share_rumor_unconfirmed",
    "wlite_rule_status_cue_share_analysis_opinion",
    "wlite_rule_status_cue_conflict_weight_share",
    "wlite_event_entropy_accepted",
)

COVERAGE_TEXT_6 = (
    "wlite_selection_weight_coverage",
    "wlite_selected_headline_only_weight_share",
    "wlite_selected_sub150_weight_share",
    "wlite_event_order_disagreement_weight_share",
    "wlite_observed_no_selected_article",
    "wlite_selected_weight_hhi",
)

if (
    len(WLITE_17) != 17
    or len(set(WLITE_17)) != 17
    or set(EVENT_CONTENT_4) - set(WLITE_17)
    or set(SEMANTIC_CONTENT_11) - set(WLITE_17)
    or set(COVERAGE_TEXT_6) - set(WLITE_17)
    or set(SEMANTIC_CONTENT_11) & set(COVERAGE_TEXT_6)
    or set(SEMANTIC_CONTENT_11) | set(COVERAGE_TEXT_6) != set(WLITE_17)
):
    raise AssertionError("W17-Lite feature partition is inconsistent")


@dataclass(frozen=True)
class Bundle:
    order: int
    key: str
    slug: str
    label: str
    architecture: str
    control: str | None = None

    @property
    def output_path(self) -> Path:
        group = "controls" if self.control else "models"
        return OUTPUT_ROOT / group / self.slug

    @property
    def experiment_path(self) -> Path:
        group = "controls" if self.control else "models"
        return EXPERIMENT_ROOT / group / self.slug


BUNDLES = (
    Bundle(0, "S0", "s0_q56_elastic_net", "V3-S0 matched Q56", "q"),
    Bundle(
        1,
        "S1",
        "s1_q56_d2_elastic_net",
        "V3-S1 matched Q56 plus D2-Normalized",
        "qd",
    ),
    Bundle(
        2,
        "S2",
        "s2_q56_wlite_elastic_net",
        "V3-S2 Q56 plus FLAN W17-Lite",
        "ql",
    ),
    Bundle(
        3,
        "S3",
        "s3_q56_d2_wlite_elastic_net",
        "V3-S3 Q56 plus D2-Normalized plus FLAN W17-Lite",
        "qdl",
    ),
    Bundle(
        4,
        "S4",
        "s4_q56_d2_wlite_shallow_xgboost",
        "V3-S4 validation-gated shallow XGBoost",
        "qdl",
    ),
    Bundle(
        10,
        "C-S2-COV",
        "s2_coverage_text_only",
        "V3-S2 coverage/text-quality-only control",
        "ql_cov",
        "coverage_text_only",
    ),
    Bundle(
        11,
        "C-S3-COV",
        "s3_coverage_text_only",
        "V3-S3 coverage/text-quality-only control",
        "qdl_cov",
        "coverage_text_only",
    ),
    Bundle(
        12,
        "C-S2-L20",
        "s2_stale_event_lag20",
        "V3-S2 20-session stale-event control",
        "ql",
        "stale_event_lag20",
    ),
    Bundle(
        13,
        "C-S3-L20",
        "s3_stale_event_lag20",
        "V3-S3 20-session stale-event control",
        "qdl",
        "stale_event_lag20",
    ),
    Bundle(
        14,
        "C-S2-WS",
        "s2_wrong_stock_semantics",
        "V3-S2 fixed wrong-stock semantic control",
        "ql",
        "wrong_stock",
    ),
    Bundle(
        15,
        "C-S3-WS",
        "s3_wrong_stock_semantics",
        "V3-S3 fixed wrong-stock semantic control",
        "qdl",
        "wrong_stock",
    ),
    Bundle(
        16,
        "C-S2-PERM",
        "s2_date_sector_permuted_semantics",
        "V3-S2 date-sector semantic permutation control",
        "ql",
        "date_sector_permutation",
    ),
    Bundle(
        17,
        "C-S3-PERM",
        "s3_date_sector_permuted_semantics",
        "V3-S3 date-sector semantic permutation control",
        "qdl",
        "date_sector_permutation",
    ),
    Bundle(
        18,
        "C-S2-LONG",
        "s2_long_description_sensitivity",
        "V3-S2 description-length-at-least-150 sensitivity",
        "ql",
        "long_description",
    ),
    Bundle(
        19,
        "C-S3-LONG",
        "s3_long_description_sensitivity",
        "V3-S3 description-length-at-least-150 sensitivity",
        "qdl",
        "long_description",
    ),
)

BUNDLE_BY_KEY = {bundle.key: bundle for bundle in BUNDLES}
if len(BUNDLE_BY_KEY) != len(BUNDLES):
    raise AssertionError("Duplicate v3 bundle keys")

LINEAR_BUNDLES = tuple(
    bundle.key for bundle in BUNDLES if bundle.key != "S4"
)
CONTROL_BUNDLES = tuple(
    bundle.key for bundle in BUNDLES if bundle.control is not None
)
