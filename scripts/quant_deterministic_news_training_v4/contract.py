"""Names, paths, and immutable design constants for the v4 ladder.

The v4 experiment is deliberately isolated from the completed v3 experiment.
In particular, this module only reads v3 Q/Q+D predictions as matched bases;
it never writes below a v3 path.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROTOCOL_PATH = Path("config/quant_deterministic_news_protocol_v4.json")
PROTOCOL_SIDECAR = Path("config/quant_deterministic_news_protocol_v4.sha256")

PANEL_ROOT = Path("data/features/q_plus_d/massive_v4")
PANEL_PATH = PANEL_ROOT / "modeling_panel_q_d2_soft_route.parquet"
STALE20_PANEL_PATH = (
    PANEL_ROOT / "modeling_panel_q_d2_soft_route_stale20.parquet"
)
WRONG_STOCK_PANEL_PATH = (
    PANEL_ROOT / "modeling_panel_q_d2_soft_route_wrong_stock.parquet"
)
PERMUTED_PANEL_PATH = (
    PANEL_ROOT / "modeling_panel_q_d2_soft_route_permuted.parquet"
)
PANEL_MANIFEST_PATH = PANEL_ROOT / "manifest.json"

V3_PROTOCOL_PATH = Path("config/quant_deterministic_news_protocol_v3.json")
V3_PROTOCOL_SIDECAR = Path(
    "config/quant_deterministic_news_protocol_v3.sha256"
)
V3_S0_ROOT = Path(
    "outputs/quant_deterministic_news/v3/models/s0_q56_elastic_net"
)
V3_S1_ROOT = Path(
    "outputs/quant_deterministic_news/v3/models/s1_q56_d2_elastic_net"
)

QUANT_V2_PROTOCOL_PATH = Path("config/quant_training_protocol_v2.json")
QUANT_V2_PREDICTIONS_PATH = Path(
    "outputs/quant_training/v2/rung_03/predictions.parquet"
)
QUANT_V2_MANIFEST_PATH = Path("outputs/quant_training/v2/rung_03/manifest.json")

OUTPUT_ROOT = Path("outputs/quant_deterministic_news/v4")
EXPERIMENT_ROOT = Path("experiments/quant_deterministic_news/v4/training")

PANEL_KEYS = ("forecast_date", "sector", "stock", "benchmark")
PREDICTION_KEYS = ("fold", "target", *PANEL_KEYS)
TARGETS = ("t1_etf", "t1_loo", "t2_etf", "t2_loo")

EVENT_LABELS = (
    "firm_operating_financial",
    "policy_corporate",
    "macro_market",
)
ROLES = ("target_idiosyncratic", "peer_idiosyncratic", "common")

CURRENT_9 = tuple(
    f"lsoft_{role}_event_{label}_joint_mass"
    for role in ROLES
    for label in EVENT_LABELS
)
INNOVATION_9 = tuple(
    f"{name}_innovation_ewma63_hl21" for name in CURRENT_9
)
SOFT_ROUTE_19 = (*CURRENT_9, *INNOVATION_9, "lsoft_observed_no_selected_article")
COUPLING_6 = tuple(
    name
    for label in EVENT_LABELS
    for name in (
        f"lsoft_coupling_{label}_c_minus_i_minus_p",
        f"lsoft_coupling_{label}_innovation_c_minus_i_minus_p",
    )
)
QUALITY_4 = (
    "lsoft_score_order_js_divergence_normalized",
    "lsoft_score_consensus_entropy_normalized",
    "lsoft_score_other_or_unclear_mass",
    "lsoft_k16_selection_weight_coverage",
)
QUALITY_ONLY_5 = (*QUALITY_4, "lsoft_observed_no_selected_article")

if (
    len(CURRENT_9) != 9
    or len(INNOVATION_9) != 9
    or len(SOFT_ROUTE_19) != 19
    or len(COUPLING_6) != 6
    or len(QUALITY_4) != 4
    or len(QUALITY_ONLY_5) != 5
    or len(set(QUALITY_ONLY_5)) != 5
    or len(set((*SOFT_ROUTE_19, *COUPLING_6, *QUALITY_4))) != 29
):
    raise AssertionError("The v4 soft-route feature partition is inconsistent")


SHORT_FOLDS = (
    {
        "name": "fold_1",
        "train_start": "2022-11-01",
        "train_end": "2024-06-30",
        "validation_start": "2024-07-01",
        "validation_end": "2024-12-31",
        "test_start": "2025-01-01",
        "test_end": "2025-06-30",
    },
    {
        "name": "fold_2",
        "train_start": "2022-11-01",
        "train_end": "2024-12-31",
        "validation_start": "2025-01-01",
        "validation_end": "2025-06-30",
        "test_start": "2025-07-01",
        "test_end": "2025-12-31",
    },
    {
        "name": "fold_3",
        "train_start": "2022-11-01",
        "train_end": "2025-06-30",
        "validation_start": "2025-07-01",
        "validation_end": "2025-12-31",
        "test_start": "2026-01-01",
        "test_end": "2026-06-30",
    },
)

# The quant-v2 fold names are the half-year OOS blocks.  For test fold j,
# residual/correction models use folds 06..j-2 for tuning, fold j-1 for
# validation, and then refit on their union before predicting fold j.
LONG_TEST_FOLDS = tuple(f"fold_{index:02d}" for index in range(9, 14))
LONG_FIRST_TRAIN_FOLD = 6
LONG_ANCHOR_MODEL = "xgboost"

ALPHA_GRID = (0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1)
L1_RATIO_GRID = (0.1, 0.5, 0.9, 1.0)
CORRECTION_SHRINKAGE_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)


@dataclass(frozen=True)
class Bundle:
    key: str
    slug: str
    label: str
    family: str
    features: str
    control: str | None = None

    @property
    def output_path(self) -> Path:
        group = "controls" if self.control else "models"
        return OUTPUT_ROOT / group / self.slug

    @property
    def experiment_path(self) -> Path:
        group = "controls" if self.control else "models"
        return EXPERIMENT_ROOT / group / self.slug


SHORT_BUNDLES = (
    Bundle("J1", "j1_q56_soft_route19_en", "Q56 + SoftRoute19", "short", "q_l19"),
    Bundle("J2", "j2_q56_d2_soft_route19_en", "Q56 + D2 + SoftRoute19", "short", "q_d2_l19"),
    Bundle("J3", "j3_q56_current9_en", "Q56 + current soft masses", "short", "q_current9"),
    Bundle("C-J1-L20", "j1_stale20", "J1 stale-20 control", "short", "q_l19", "stale20"),
    Bundle("C-J2-L20", "j2_stale20", "J2 stale-20 control", "short", "q_d2_l19", "stale20"),
    Bundle("C-J1-WS", "j1_wrong_stock", "J1 wrong-stock control", "short", "q_l19", "wrong_stock"),
    Bundle("C-J2-WS", "j2_wrong_stock", "J2 wrong-stock control", "short", "q_d2_l19", "wrong_stock"),
    Bundle("C-J1-PERM", "j1_permuted", "J1 permuted-score control", "short", "q_l19", "permuted"),
    Bundle("C-J2-PERM", "j2_permuted", "J2 permuted-score control", "short", "q_d2_l19", "permuted"),
    Bundle("C-J1-QUALITY", "j1_quality_only", "J1 quality-only control", "short", "q_quality5", "quality"),
    Bundle("C-J2-QUALITY", "j2_quality_only", "J2 quality-only control", "short", "q_d2_quality5", "quality"),
)

LONG_BUNDLES = (
    Bundle("R0", "r0_quant_v2_xgb_anchor", "Frozen long-history quant-v2 XGBoost", "long", "base"),
    Bundle("RCAL", "rcal_base_ols", "Long-Q base-only OLS calibration", "long", "base"),
    Bundle("RRES-C6", "rres_coupling6_en", "Long-Q residual + Coupling6", "long", "c6"),
    Bundle("RRES-L19", "rres_soft_route19_en", "Long-Q residual + SoftRoute19", "long", "l19"),
    Bundle("RSTACK", "rstack_base_soft_route19_en", "Long-Q + SoftRoute19 stack", "long", "base_l19"),
    Bundle("C-RRES-L19-L20", "rres_soft_route19_stale20", "Residual SoftRoute19 stale-20 control", "long", "l19", "stale20"),
    Bundle("C-RRES-L19-WS", "rres_soft_route19_wrong_stock", "Residual SoftRoute19 wrong-stock control", "long", "l19", "wrong_stock"),
    Bundle("C-RRES-L19-PERM", "rres_soft_route19_permuted", "Residual SoftRoute19 permuted-score control", "long", "l19", "permuted"),
    Bundle("C-RRES-L19-QUALITY", "rres_soft_route19_quality_only", "Residual quality-only control", "long", "quality5", "quality"),
)

BUNDLES = (*SHORT_BUNDLES, *LONG_BUNDLES)
BUNDLE_BY_KEY = {bundle.key: bundle for bundle in BUNDLES}
if len(BUNDLE_BY_KEY) != len(BUNDLES):
    raise AssertionError("Duplicate v4 bundle keys")


BOUND_IMPLEMENTATION = (
    Path("scripts/correlation_training/training_common.py"),
    Path("scripts/quant_deterministic_news_training_v4/contract.py"),
    Path("scripts/quant_deterministic_news_training_v4/common.py"),
    Path("scripts/quant_deterministic_news_training_v4/lock_protocol.py"),
    Path("scripts/quant_deterministic_news_training_v4/run_short.py"),
    Path("scripts/quant_deterministic_news_training_v4/run_long.py"),
    Path("scripts/quant_deterministic_news_training_v4/run_final_comparison.py"),
)
