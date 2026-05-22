"""Pull Ken French UMD (momentum) daily factor from WRDS; merge with existing FF5 cache.

Mirrors the pattern in `03_UCHICAGO_BOOTH/Retail_Sentiment/src/data.py:pull_ff_daily()`:
read from disk cache if present, otherwise hit WRDS and cache to parquet.

Output: a combined FF5+UMD parquet at:
    03_UCHICAGO_BOOTH/Retail_Sentiment/data/processed/ff5_umd_daily.parquet

Columns: date, mktrf, smb, hml, rmw, cma, umd, rf

Run once before Phase D notebooks:
    python scripts/pull_umd_from_wrds.py

WRDS auth must be primed (~/.pgpass on linux/mac; %APPDATA%/postgresql/pgpass.conf on Windows).
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


# Path constants — keep alongside the existing FF5 file
RETAIL_SENTIMENT_PROCESSED = (
    Path(__file__).resolve().parents[3]
    / "03_UCHICAGO_BOOTH"
    / "Retail_Sentiment"
    / "data"
    / "processed"
)
FF5_CACHE = RETAIL_SENTIMENT_PROCESSED / "ff5_daily.parquet"
FF5_UMD_CACHE = RETAIL_SENTIMENT_PROCESSED / "ff5_umd_daily.parquet"


def load_ff5() -> pd.DataFrame:
    """Read the existing FF5 cache. Errors if missing — we don't re-pull FF5 here."""
    if not FF5_CACHE.exists():
        raise FileNotFoundError(
            f"FF5 cache missing at {FF5_CACHE}.\n"
            f"Pull it first via `Retail_Sentiment/src/data.py:pull_ff_daily()`."
        )
    ff5 = pd.read_parquet(FF5_CACHE)
    ff5["date"] = pd.to_datetime(ff5["date"])
    return ff5


def pull_umd_from_wrds() -> pd.DataFrame:
    """Pull UMD (momentum factor) daily from WRDS `ff.factors_daily`.

    Returns columns: date, umd (float).
    """
    import wrds
    db = wrds.Connection()
    try:
        # ff.factors_daily contains the FF 3-factor model + UMD + RF daily
        ff3umd = db.get_table("ff", "factors_daily", obs=-1)
    finally:
        db.close()

    ff3umd["date"] = pd.to_datetime(ff3umd["date"])
    if "umd" not in ff3umd.columns:
        raise ValueError(
            f"`umd` column not found in ff.factors_daily. Available columns: {list(ff3umd.columns)}"
        )
    umd = ff3umd[["date", "umd"]].copy()
    umd["umd"] = umd["umd"].astype(float)
    return umd.sort_values("date").reset_index(drop=True)


def main() -> int:
    if FF5_UMD_CACHE.exists():
        print(f"Cache exists: {FF5_UMD_CACHE}")
        df = pd.read_parquet(FF5_UMD_CACHE)
        print(f"  rows: {len(df):,}, date range: {df['date'].min().date()} -> {df['date'].max().date()}")
        print(f"  columns: {list(df.columns)}")
        return 0

    print("Loading existing FF5 cache...")
    ff5 = load_ff5()
    print(f"  ff5: {len(ff5):,} rows, {ff5['date'].min().date()} -> {ff5['date'].max().date()}")

    print("Pulling UMD daily from WRDS ff.factors_daily...")
    umd = pull_umd_from_wrds()
    print(f"  umd: {len(umd):,} rows, {umd['date'].min().date()} -> {umd['date'].max().date()}")

    print("Merging FF5 + UMD on date...")
    merged = ff5.merge(umd, on="date", how="inner")
    print(f"  merged: {len(merged):,} rows, columns: {list(merged.columns)}")

    print(f"Writing cache: {FF5_UMD_CACHE}")
    FF5_UMD_CACHE.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(FF5_UMD_CACHE, index=False)
    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
