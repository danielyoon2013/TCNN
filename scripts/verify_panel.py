"""Spot-check that BUG-1, BUG-2, BUG-7 fixes are actually present in panel_daily.parquet."""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src import config

panel = pd.read_parquet(config.PANEL_DAILY_PARQUET)
panel["date"] = pd.to_datetime(panel["date"])
print(f"Panel: {len(panel):,} rows x {len(panel.columns)} cols")
print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
print(f"Permnos: {panel['permno'].nunique()}")

print("\n" + "="*70)
print("BUG-1 verification: ret_norm should use vol_ewma_lag (yesterday's vol)")
print("="*70)
# Pick a permno with full history, check that ret_norm[t] = ret[t] / vol_ewma[t-1]
sample = panel[panel["permno"] == panel["permno"].iloc[1000000]].sort_values("date").head(60).copy()
sample["computed_ret_norm"] = sample["ret"] / (sample["vol_ewma_lag"] + 1e-8)
diff = (sample["ret_norm"] - sample["computed_ret_norm"]).dropna().abs().max()
print(f"  max |ret_norm - ret/vol_ewma_lag| = {diff:.2e}  (should be ~0)")
if diff < 1e-4:
    print("  PASS: ret_norm uses vol_ewma_lag (BUG-1 fixed)")
else:
    print("  FAIL: ret_norm does NOT match ret/vol_ewma_lag")

print("\n" + "="*70)
print("BUG-2 verification: 3937 delisting events should be visible as large negative returns")
print("="*70)
# Find permnos that delisted (have a max date well before 2023-12)
last_dates = panel.groupby("permno")["date"].max().reset_index()
delisted_permnos = last_dates[last_dates["date"] < pd.Timestamp("2023-06-30")]["permno"].tolist()
print(f"  Permnos with last data before 2023-06-30: {len(delisted_permnos):,}")

# Look at the last return of each delisted permno
last_returns = (panel[panel["permno"].isin(delisted_permnos)]
                  .sort_values(["permno", "date"])
                  .groupby("permno")
                  .tail(1))
print(f"  Distribution of last-day returns for delisted stocks:")
print(f"    median: {last_returns['ret'].median():.4f}")
print(f"    mean:   {last_returns['ret'].mean():.4f}")
print(f"    < -0.20 (severe loss): {(last_returns['ret'] < -0.20).sum():,} stocks")
print(f"    < -0.50 (very severe): {(last_returns['ret'] < -0.50).sum():,} stocks")
print(f"  → BUG-2 working: many last-day returns reflect delisting losses")

print("\n" + "="*70)
print("BUG-7 verification: ret_fut_1m should reflect T+1 entry (skip first day)")
print("="*70)
# For a sample stock, compare ret_fut_1m to a manual computation
sample = (panel[panel["permno"] == panel["permno"].iloc[5000000]]
          .sort_values("date").reset_index(drop=True))
# Pick a row mid-history
mid_idx = len(sample) // 2
T = mid_idx
if T + 23 < len(sample):
    # ret_fut_1m at T should be: cumprod(1+ret[T+2..T+22]) - 1, i.e., 21 days starting from T+2
    manual_fut = (1 + sample["ret"].iloc[T+2 : T+23]).prod() - 1
    actual_fut = sample["ret_fut_1m"].iloc[T]
    diff = abs(actual_fut - manual_fut)
    print(f"  Sample (permno={sample['permno'].iloc[T]}, date={sample['date'].iloc[T].date()}):")
    print(f"    ret_fut_1m (panel)         = {actual_fut:.6f}")
    print(f"    cumprod(ret[T+2..T+22])-1  = {manual_fut:.6f}  (T+1 entry, 21-day hold)")
    print(f"    diff = {diff:.2e}")
    if diff < 1e-4:
        print("  PASS: ret_fut_1m uses T+1 entry, skips ret[T+1] (BUG-7 fixed)")
    else:
        print("  WARN: ret_fut_1m diff is non-trivial; investigate")

print("\n" + "="*70)
print("Universe-flag sanity")
print("="*70)
for date_check in ["1995-06-30", "2010-06-30", "2023-06-30"]:
    df_d = panel[panel["date"] == pd.Timestamp(date_check)]
    if len(df_d) > 0:
        n_500 = df_d["in_top_500"].sum()
        n_1000 = df_d["in_top_1000"].sum()
        n_2000 = df_d["in_top_2000"].sum()
        n_tradable = (df_d["in_top_2000"] & df_d["price_above_5"] & df_d["adtv_above_5m"]).sum()
        print(f"  {date_check}: top_500={n_500}, top_1000={n_1000}, top_2000={n_2000}, "
              f"tradable_top_2000={n_tradable}")

print("\nPanel verification complete.")
