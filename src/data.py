"""WRDS data pulls + delisting-return merge.

Three things live here, nothing else:
  1. WRDS query functions for crsp.dsf, crsp.dsedelist, ff5
  2. Universe selection (top-N permnos by mktcap at reference dates)
  3. Delisting-return merge (BUG-2 fix: combine crsp.dsedelist into the last DSF row)

No feature engineering, no panels — those live in `src.features` and `src.panels`.
"""

from __future__ import annotations
import os
import numpy as np
import pandas as pd

from . import config


# =============================================================================
# WRDS pulls
# =============================================================================

def pull_universe_permnos(conn, reference_dates, top_n: int = config.TOP_N_DEFAULT,
                           include_amex: bool = False) -> list[int]:
    """Union of top-N permnos by mktcap across multiple reference dates.

    Survivorship-bias-resistant superset; per-rebal-date universe filter is
    applied later via `in_top_N` panel columns.
    """
    exch_list  = "1,2,3" if include_amex else "1,3"
    shrcd_list = "10,11"

    all_permnos: set[int] = set()
    for asof in reference_dates:
        sql = f"""
        with eligible as (
            select permno, exchcd, shrcd
            from crsp.msenames
            where date('{asof}') between namedt and nameendt
              and exchcd in ({exch_list})
              and shrcd in ({shrcd_list})
        ),
        px as (
            select permno, date, prc, shrout
            from crsp.dsf
            where date = date('{asof}')
        )
        select e.permno,
               (abs(p.prc) * p.shrout) as mktcap_thousands
        from eligible e
        join px p on e.permno = p.permno
        where p.prc is not null and p.shrout is not null and p.shrout > 0
        order by mktcap_thousands desc
        limit {top_n}
        """
        univ = conn.raw_sql(sql)
        permnos_at_date = set(univ["permno"].astype(int).tolist())
        all_permnos.update(permnos_at_date)
        print(f"  {asof}: {len(permnos_at_date)} permnos  (cumulative {len(all_permnos)})")
    return sorted(int(p) for p in all_permnos)


def pull_dsf(conn, permnos: list[int], start_date: str, end_date: str,
              chunk_size: int = 500) -> pd.DataFrame:
    """Daily stock file. Chunked to avoid SQL length limits."""
    chunks = []
    n_chunks = (len(permnos) - 1) // chunk_size + 1
    for i in range(0, len(permnos), chunk_size):
        ids = ",".join(str(p) for p in permnos[i:i + chunk_size])
        sql = f"""
        select permno, date, prc, ret, vol, shrout, cfacpr, cfacshr,
               openprc, askhi, bidlo
        from crsp.dsf
        where permno in ({ids})
          and date between date('{start_date}') and date('{end_date}')
        order by permno, date
        """
        df = conn.raw_sql(sql, date_cols=["date"])
        chunks.append(df)
        print(f"  dsf chunk {i // chunk_size + 1}/{n_chunks}: {len(df):,} rows")
    return pd.concat(chunks, ignore_index=True).sort_values(["permno", "date"]).reset_index(drop=True)


def pull_dsedelist(conn, permnos: list[int], start_date: str, end_date: str,
                    chunk_size: int = 500) -> pd.DataFrame:
    """Delisting events (BUG-2 fix data source).

    crsp.dsf does NOT include the final delisting return for bankrupt/merged
    stocks. We pull dsedelist separately and merge in `merge_delisting_returns`.
    """
    chunks = []
    n_chunks = (len(permnos) - 1) // chunk_size + 1
    for i in range(0, len(permnos), chunk_size):
        ids = ",".join(str(p) for p in permnos[i:i + chunk_size])
        sql = f"""
        select permno, dlstdt, dlret, dlstcd
        from crsp.dsedelist
        where permno in ({ids})
          and dlstdt between date('{start_date}') and date('{end_date}')
          and dlret is not null
        order by permno, dlstdt
        """
        df = conn.raw_sql(sql, date_cols=["dlstdt"])
        chunks.append(df)
        print(f"  dsedelist chunk {i // chunk_size + 1}/{n_chunks}: {len(df):,} rows")
    if not chunks:
        return pd.DataFrame(columns=["permno", "dlstdt", "dlret", "dlstcd"])
    return pd.concat(chunks, ignore_index=True).sort_values(["permno", "dlstdt"]).reset_index(drop=True)


def pull_ff5_daily(conn, start_date: str, end_date: str) -> pd.DataFrame:
    """Daily Fama-French 5 factors + risk-free rate (for FF5 neutrality regression)."""
    sql = f"""
    select date, mktrf, smb, hml, rmw, cma, rf
    from ff.fivefactors_daily
    where date between date('{start_date}') and date('{end_date}')
    order by date
    """
    return conn.raw_sql(sql, date_cols=["date"])


# =============================================================================
# Delisting-return merge (BUG-2 fix)
# =============================================================================

def merge_delisting_returns(dsf: pd.DataFrame, dsedelist: pd.DataFrame) -> pd.DataFrame:
    """Combine delisting return into the last available daily return per permno.

    For each delisted stock:
      - find the last DSF row with date <= dlstdt
      - update that row's `ret` to (1 + ret_dsf) * (1 + dlret) - 1
      - if no DSF row exists on/before dlstdt, append a synthetic row at dlstdt

    This captures the bankrupt-stock-disappears-from-DSF survivorship bias.
    See Beaver-McNichols-Price 2007.
    """
    if len(dsedelist) == 0:
        return dsf

    out = dsf.copy()
    out["date"] = pd.to_datetime(out["date"])
    de = dsedelist.copy()
    de["dlstdt"] = pd.to_datetime(de["dlstdt"])

    out = out.sort_values(["permno", "date"]).reset_index(drop=True)
    permno_groups = {p: idx.to_numpy() for p, idx in out.groupby("permno").groups.items()}

    n_combined = 0
    n_appended = 0
    new_rows = []
    for _, row in de.iterrows():
        p = int(row["permno"])
        dlstdt = row["dlstdt"]
        dlret  = float(row["dlret"])

        idxs = permno_groups.get(p)
        if idxs is None or len(idxs) == 0:
            continue

        sub_dates = out.loc[idxs, "date"].values
        mask = sub_dates <= np.datetime64(dlstdt)
        if mask.any():
            last_idx = idxs[np.where(mask)[0][-1]]
            base_ret = float(out.at[last_idx, "ret"])
            if not np.isnan(base_ret):
                out.at[last_idx, "ret"] = (1 + base_ret) * (1 + dlret) - 1
            else:
                out.at[last_idx, "ret"] = dlret
            n_combined += 1
        else:
            template = {col: np.nan for col in out.columns}
            template.update({"permno": p, "date": dlstdt, "ret": dlret})
            new_rows.append(template)
            n_appended += 1

    if new_rows:
        out = pd.concat([out, pd.DataFrame(new_rows)], ignore_index=True)
    out = out.sort_values(["permno", "date"]).reset_index(drop=True)
    print(f"  delisting returns: {n_combined} combined into existing rows, {n_appended} appended as synthetic rows")
    return out


# =============================================================================
# Cache-aware loaders for notebooks (load if exists, else pull and save)
# =============================================================================

def load_or_pull_dsf(conn, permnos, start_date, end_date,
                     cache_path=None, force_refresh: bool = False) -> pd.DataFrame:
    cache_path = cache_path or config.DSF_RAW
    if (not force_refresh) and os.path.exists(cache_path):
        print(f"  loading DSF from cache: {cache_path}")
        return pd.read_parquet(cache_path)
    df = pull_dsf(conn, permnos, start_date, end_date)
    df.to_parquet(cache_path, index=False)
    print(f"  saved DSF to: {cache_path}")
    return df


def load_or_pull_dsedelist(conn, permnos, start_date, end_date,
                            cache_path=None, force_refresh: bool = False) -> pd.DataFrame:
    cache_path = cache_path or config.DSEDELIST_RAW
    if (not force_refresh) and os.path.exists(cache_path):
        print(f"  loading DSEDELIST from cache: {cache_path}")
        return pd.read_parquet(cache_path)
    df = pull_dsedelist(conn, permnos, start_date, end_date)
    df.to_parquet(cache_path, index=False)
    print(f"  saved DSEDELIST to: {cache_path}")
    return df


def load_or_pull_ff5(conn, start_date, end_date,
                      cache_path=None, force_refresh: bool = False) -> pd.DataFrame:
    cache_path = cache_path or config.FF5_RAW
    if (not force_refresh) and os.path.exists(cache_path):
        return pd.read_parquet(cache_path)
    df = pull_ff5_daily(conn, start_date, end_date)
    df.to_parquet(cache_path, index=False)
    return df
