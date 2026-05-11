"""Strip lag_ret_1..lag_ret_252 columns from panel_daily.parquet.

These 252 columns inflate the panel from ~30 to 281 columns and turn a
~5.6 GB in-memory panel into ~50 GB. Since lag features are pure functions
of the `ret` column (just shift), we compute them on-the-fly in factors.py
when needed.

Run once:
    python scripts/strip_lag_columns.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from src import config

PANEL = config.PANEL_DAILY_PARQUET
print(f"Loading {PANEL}...")
# Use pyarrow column selection so we don't OOM on read
import pyarrow.parquet as pq
schema = pq.read_schema(PANEL)
all_cols = [c for c in schema.names]
keep_cols = [c for c in all_cols if not c.startswith("lag_ret_")]
n_dropped = len(all_cols) - len(keep_cols)
print(f"  total cols: {len(all_cols)},  keeping: {len(keep_cols)},  dropping: {n_dropped} lag_ret_* cols")

panel = pd.read_parquet(PANEL, columns=keep_cols)
print(f"  loaded {len(panel):,} rows x {len(panel.columns)} cols")

backup = PANEL.with_suffix(".parquet.with_lags")
PANEL.rename(backup)
print(f"  backed up original to {backup}")

panel.to_parquet(PANEL, index=False)
print(f"  saved slimmed panel to {PANEL}")

# Verify
import os
old_mb = os.path.getsize(backup) / 1e6
new_mb = os.path.getsize(PANEL) / 1e6
print(f"\n  size before: {old_mb:,.0f} MB")
print(f"  size after:  {new_mb:,.0f} MB  ({100*new_mb/old_mb:.1f}% of original)")
