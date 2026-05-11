"""End-to-end panel regeneration: WRDS -> cleaned DSF -> engineered panel.

Runs in the background. Logs progress to stdout (captured by the bash tool).

Stages:
  1. Pull universe permnos (top-2000 at 8 reference dates)
  2. Pull DSF (daily stock file, 1989-2023, all permnos, chunked)
  3. Pull DSEDELIST (delisting events)
  4. Merge delisting returns into DSF (BUG-2 fix)
  5. Feature engineering (BUG-1 EWMA-lag, momentum factors, lag features,
     universe flags, T+1 forward returns from BUG-7)
  6. Write data/03_features/panel_daily.parquet
"""

import sys
import time
from pathlib import Path

# Project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import wrds
import pandas as pd
import numpy as np

from src import config, data, features

WRDS_USER = "danielyoon"
START_DATE = "1989-01-01"
END_DATE   = "2023-12-31"
TOP_N      = 2000
REFERENCE_DATES = ["1990-01-02", "1995-01-03", "2000-01-03", "2005-01-03",
                   "2010-01-04", "2015-01-05", "2020-01-02", "2023-01-03"]


def stage(label):
    print(f"\n{'='*70}\n[{time.strftime('%H:%M:%S')}] {label}\n{'='*70}")


def main():
    t0 = time.time()

    stage("Stage 1/6: Connect to WRDS")
    conn = wrds.Connection(wrds_username=WRDS_USER)

    stage(f"Stage 2/6: Pull universe permnos at {len(REFERENCE_DATES)} reference dates")
    permnos = data.pull_universe_permnos(conn, REFERENCE_DATES, top_n=TOP_N)
    print(f"  -> {len(permnos)} unique permnos")

    stage(f"Stage 3/6: Pull DSF for {len(permnos)} permnos, {START_DATE} -> {END_DATE}")
    dsf = data.load_or_pull_dsf(conn, permnos, START_DATE, END_DATE)
    print(f"  -> DSF: {len(dsf):,} rows, {dsf['permno'].nunique():,} permnos")
    print(f"  -> date range: {dsf['date'].min()} -> {dsf['date'].max()}")

    stage(f"Stage 4/6: Pull DSEDELIST")
    dsedelist = data.load_or_pull_dsedelist(conn, permnos, START_DATE, END_DATE)
    print(f"  -> DSEDELIST: {len(dsedelist):,} delisting events")
    conn.close()

    stage("Stage 5/6: Merge delisting returns (BUG-2 fix)")
    daily_clean = data.merge_delisting_returns(dsf, dsedelist)
    daily_clean.to_parquet(config.CLEAN_DIR / "daily_clean.parquet", index=False)
    print(f"  -> daily_clean: {len(daily_clean):,} rows saved to {config.CLEAN_DIR}")

    stage("Stage 6/6: Feature engineering -> panel_daily.parquet")
    panel = features.build_panel(daily_clean)
    panel.to_parquet(config.PANEL_DAILY_PARQUET, index=False)
    print(f"  -> panel_daily: {len(panel):,} rows × {len(panel.columns)} cols")
    print(f"  -> saved to: {config.PANEL_DAILY_PARQUET}")

    # Quick sanity diagnostics
    print("\n--- Feature completeness in OOS period (2010-2023) ---")
    oos = panel[pd.to_datetime(panel["date"]).dt.year.between(2010, 2023)]
    for col in ["ret", "vol_ewma_lag", "ret_norm", "mom_12_1", "ewm_h60_skip",
                "ret_fut_1m", "in_top_2000", "adtv_above_5m", "price_above_5"]:
        if col in oos.columns:
            valid_pct = oos[col].notna().mean() * 100
            print(f"  {col:<20} {valid_pct:5.1f}% non-null")

    print(f"\n--- Stocks in top-2000 over time (selected dates) ---")
    panel_dt = pd.to_datetime(panel["date"])
    for year in [1995, 2000, 2010, 2020, 2023]:
        year_data = panel[panel_dt.dt.year == year]
        if len(year_data) > 0:
            mid_year = year_data[panel_dt[panel_dt.dt.year == year].dt.month == 6].head(1)
            if len(mid_year) == 0:
                mid_year = year_data.head(1)
            sample_date = mid_year["date"].iloc[0]
            n_in_universe = panel[(panel["date"] == sample_date) & panel["in_top_2000"]].shape[0]
            print(f"  {sample_date}: {n_in_universe} stocks in top_2000 universe")

    elapsed = time.time() - t0
    print(f"\n{'='*70}\nALL STAGES COMPLETE in {elapsed/60:.1f} min\n{'='*70}")


if __name__ == "__main__":
    main()
