"""Project-wide constants, paths, and column conventions.

Single source of truth so we never have ENTRY_OFFSET=21 in one place and
ENTRY_OFFSET=1 in another. If you add a new constant, add it here.
"""

from __future__ import annotations
from pathlib import Path


# =============================================================================
# Paths (relative to project root). All consumers read paths via this module.
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR     = PROJECT_ROOT / "data"
RAW_DIR      = DATA_DIR / "01_raw"
CLEAN_DIR    = DATA_DIR / "02_clean"
FEATURES_DIR = DATA_DIR / "03_features"
PANELS_DIR   = DATA_DIR / "05_panels_tcnn"
OUTPUTS_DIR  = PROJECT_ROOT / "outputs"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

# Canonical files
PANEL_DAILY_PARQUET = FEATURES_DIR / "panel_daily.parquet"
DSF_RAW             = RAW_DIR / "crsp_dsf.parquet"
DSEDELIST_RAW       = RAW_DIR / "crsp_dsedelist.parquet"
FF5_RAW             = RAW_DIR / "ff5_daily.parquet"


# =============================================================================
# Time conventions
# =============================================================================

LOOKBACK_DAYS         = 252        # ~12 months of trading days; TCNN input length
HOLDING_DAYS_DEFAULT  = 25         # max forward days kept in Y_panel; actual varies
MIN_HOLDING_DAYS      = 15         # skip rebal periods with < this many trading days
EWMA_HALF_LIFE        = 20         # half-life for vol_ewma (paper convention)
EWM_MOM_HALF_LIFE     = 60         # half-life for rung-2 EWM momentum (skip-month)
SKIP_MONTH_DAYS       = 21         # 1-month skip window for momentum (~21 trading days)

# CRITICAL: T+1 entry convention to avoid look-ahead (BUG-7 fix).
# Signal at end-of-day T → execute at close[T+1] → first holding-period day = ret[T+2]
ENTRY_OFFSET_DAYS = 1


# =============================================================================
# Universe defaults
# =============================================================================

TOP_N_DEFAULT       = 2000           # paper convention
MIN_PRICE_DEFAULT   = 5.0            # exclude penny stocks
MIN_ADTV_USD        = 5_000_000      # 21-day avg dollar volume floor
SHORT_BORROW_BPS_GC = 30.0           # general-collateral assumption (annual)


# =============================================================================
# Sharpe / loss conventions
# =============================================================================

TRADING_DAYS_PER_YEAR = 252
DAILY_RET_CLIP        = 0.10         # train-time clip on daily portfolio returns
GRAD_CLIP_NORM        = 1.0


# =============================================================================
# COLUMN_REGISTRY: canonical column groupings on panel_daily.parquet
# Used by notebooks and runner to inspect / verify schema.
# =============================================================================

COLUMN_REGISTRY = {
    "id": ["date", "permno"],
    "raw": ["prc", "vol", "shrout", "ret"],
    "core_features": ["mktcap_dollars", "vol_ewma", "vol_ewma_lag", "ret_norm"],
    "factor_inputs": ["mom_12_1", "ewm_h60_skip"]
                    + [f"lag_ret_{l}" for l in range(1, LOOKBACK_DAYS + 1)],
    "universe_flags": ["mktcap_rank", "in_top_500", "in_top_1000", "in_top_2000",
                        "price_above_5", "adtv_21d_usd", "adtv_above_5m", "vol_21d"],
}


def all_required_columns() -> list[str]:
    """Flat list of columns the panel must have."""
    return [c for group in COLUMN_REGISTRY.values() for c in group]
