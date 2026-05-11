"""Long-format panel → TCNN-ready tensors.

This is the only stage that does long → wide reshaping. Everything else stays
in long format. The output here is consumed only by TCNN-family models;
hand-crafted/OLS rungs read panel_daily.parquet directly without invoking this.

Two-step approach for speed:
  1. Pivot long panel into a dense (D, N, F) numpy array (one-time, ~10s for full data)
  2. Per rebal date, slice the dense array into (N, F, L) X plus (N, H) Y

Replaces the original per-stock-per-month groupby loop in `prep_tensor.py`
which was ~10x slower.

BUG-7 fix: Y starts at T + ENTRY_OFFSET_DAYS + 1 (skipping the entry day).
BUG-8 fix: stocks that delist mid-period are kept in the panel; their daily
            returns are NaN-padded after the last available day, then the
            portfolio-returns computer treats NaN as 0 (cash position).
"""

from __future__ import annotations
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd

from . import config


# =============================================================================
# Dense (D, N, F) array via index-based scatter (the fast pivot)
# =============================================================================

def panel_to_dense_3d(panel: pd.DataFrame, feature_cols: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pivot long panel into a dense (D, N, F) numpy array.

    Returns:
        dense:   (D, N, F) float32, NaN-filled where data is absent
        dates:   (D,) sorted unique dates (numpy datetime64[D])
        permnos: (N,) sorted unique permnos (int64)
    """
    panel = panel[["date", "permno"] + feature_cols].sort_values(["date", "permno"]).reset_index(drop=True)
    panel["date"] = pd.to_datetime(panel["date"])

    dates   = np.sort(panel["date"].unique())
    permnos = np.sort(panel["permno"].unique().astype(np.int64))
    date_idx   = {d: i for i, d in enumerate(dates)}
    permno_idx = {p: i for i, p in enumerate(permnos)}

    D, N, F = len(dates), len(permnos), len(feature_cols)
    dense = np.full((D, N, F), np.nan, dtype=np.float32)

    di = panel["date"].map(date_idx).values
    pi = panel["permno"].map(permno_idx).values
    dense[di, pi, :] = panel[feature_cols].values.astype(np.float32)

    return dense, dates, permnos


# =============================================================================
# Rebalancing schedule
# =============================================================================

def get_month_end_business_days(dates: np.ndarray | pd.Series) -> list[pd.Timestamp]:
    """Last trading day of each calendar month.

    `groupby(year_month)['date'].max()` automatically picks the last trading
    date in the month, so weekends and holidays are handled correctly.
    """
    df = pd.DataFrame({"date": pd.to_datetime(dates)})
    df["year_month"] = df["date"].dt.to_period("M")
    return df.groupby("year_month")["date"].max().sort_index().tolist()


# =============================================================================
# Slice (D, N, F) → (T, N, F, L) X-tensor + (T, N, H) Y-tensor
# =============================================================================

@dataclass
class TCNNPanels:
    X: np.ndarray              # (T, N, F, L)
    Y: np.ndarray              # (T, N, H)
    holding_days: np.ndarray   # (T,) int actual holding days per rebal
    mask: np.ndarray           # (T, N) bool — valid stocks per rebal
    rebal_dates: list          # (T,) pd.Timestamp
    permno_idx: dict           # permno -> column index into N
    permno_list: list          # (N,) permnos in column order
    feature_cols: list         # (F,) feature names in channel order
    lookback: int
    entry_offset: int
    max_holding: int


def build_tcnn_panels(
    panel: pd.DataFrame,
    feature_cols: list[str],
    universe_col: str = "in_top_2000",
    additional_filter_cols: tuple[str, ...] = ("price_above_5", "adtv_above_5m"),
    lookback: int = config.LOOKBACK_DAYS,
    max_holding_days: int = config.HOLDING_DAYS_DEFAULT,
    min_holding_days: int = config.MIN_HOLDING_DAYS,
    entry_offset: int = config.ENTRY_OFFSET_DAYS,
    min_stocks_per_rebal: int = 50,
) -> TCNNPanels:
    """Build (T, N, F, L) X-tensor and (T, N, H) Y-tensor from a long panel.

    Args:
        panel: long-format panel_daily.parquet contents
        feature_cols: which columns become channels in X (e.g. ['ret', 'vol_ewma', 'ret_norm'])
        universe_col: panel column to filter by (e.g., 'in_top_2000')
        additional_filter_cols: extra boolean columns ANDed with universe_col
        lookback: number of past days per X tensor (paper: 252)
        max_holding_days: max forward days kept in Y; actual stored in `holding_days[t]`
        min_holding_days: skip rebal periods with fewer trading days than this
        entry_offset: T+entry_offset is the entry day; first held return is at T+entry_offset+1
        min_stocks_per_rebal: skip rebal periods with fewer valid stocks

    Returns:
        TCNNPanels dataclass with X, Y, mask, metadata
    """
    panel = panel.copy()
    panel["date"] = pd.to_datetime(panel["date"])

    # Step 1: dense (D, N, F) of feature values
    dense, all_dates, all_permnos = panel_to_dense_3d(panel, feature_cols)
    permno_idx = {int(p): i for i, p in enumerate(all_permnos)}

    # Step 1b: dense (D, N) of universe-membership boolean (single source of truth)
    # NaN treated as False (not in universe) to avoid pandas nullable-bool astype error.
    panel["_univ"] = panel[universe_col].fillna(False).astype(bool)
    for col in additional_filter_cols:
        if col in panel.columns:
            panel["_univ"] = panel["_univ"] & panel[col].fillna(False).astype(bool)

    univ_dense = np.zeros((len(all_dates), len(all_permnos)), dtype=bool)
    di_u = panel["date"].map({d: i for i, d in enumerate(all_dates)}).values
    pi_u = panel["permno"].map(permno_idx).values
    univ_dense[di_u, pi_u] = panel["_univ"].values

    # Also pull daily returns for forward-returns Y tensor (BUG-8 cash-pad)
    # `ret` is assumed to be among feature_cols OR pulled separately from the panel
    if "ret" in feature_cols:
        ret_chan_idx = feature_cols.index("ret")
    else:
        # build a separate dense ret matrix
        ret_dense, _, _ = panel_to_dense_3d(panel, ["ret"])
        # broadcast to a 2D (D, N) view; use this for Y
        ret_chan_idx = None  # signals "use ret_dense_2d below"
        ret_dense_2d = ret_dense[:, :, 0]
    if ret_chan_idx is not None:
        ret_dense_2d = dense[:, :, ret_chan_idx]

    # Step 2: month-end business days from observed dates → rebal schedule
    rebal_dates = get_month_end_business_days(all_dates)

    # Step 3: per rebal date, slice the dense arrays
    T_total = len(rebal_dates) - 1   # final rebal has no future window to evaluate
    N = len(all_permnos)
    F = len(feature_cols)

    # Pre-allocate output arrays (over T_total; we'll mask out skipped months)
    X = np.full((T_total, N, F, lookback), np.nan, dtype=np.float32)
    Y = np.full((T_total, N, max_holding_days), np.nan, dtype=np.float32)
    holding_days_arr = np.zeros(T_total, dtype=np.int32)
    mask = np.zeros((T_total, N), dtype=bool)
    valid_rebal_indices = []

    date_to_idx = {d: i for i, d in enumerate(all_dates)}

    for t, rebal_date in enumerate(rebal_dates[:-1]):
        next_rebal = rebal_dates[t + 1]
        ti = date_to_idx[np.datetime64(rebal_date, "ns")] if np.datetime64(rebal_date, "ns") in date_to_idx else None
        # `all_dates` is numpy datetime64; map differently:
        ti = int(np.searchsorted(all_dates, np.datetime64(rebal_date, "ns")))
        if all_dates[ti] != np.datetime64(rebal_date, "ns"):
            continue

        # Need at least `lookback` past days
        if ti < lookback - 1:
            continue

        # Find next rebal index
        ti_next = int(np.searchsorted(all_dates, np.datetime64(next_rebal, "ns")))
        if ti_next >= len(all_dates) or all_dates[ti_next] != np.datetime64(next_rebal, "ns"):
            continue

        # Holding window: trading days strictly after T+entry_offset, up to and including next_rebal
        holding_start = ti + entry_offset + 1   # inclusive
        holding_end   = ti_next                 # inclusive
        H = holding_end - holding_start + 1
        if H < min_holding_days:
            continue
        H = min(H, max_holding_days)

        # X[t]: past `lookback` days for ALL stocks (universe filtering is via mask, not X)
        X_window = dense[ti - lookback + 1 : ti + 1, :, :]   # (L, N, F)
        X[t] = X_window.transpose(1, 2, 0)                    # (N, F, L)

        # Y[t]: forward returns over the holding window
        # BUG-8 fix: keep stocks with NaN returns post-delisting; portfolio_returns_with_drift
        #            treats NaN as 0 (cash). We do NOT skip them as the original code did.
        Y_window = ret_dense_2d[holding_start : holding_start + H, :]  # (H, N)
        Y[t, :, :H] = Y_window.T                                       # (N, H)

        # Mask: stock is valid if (a) in universe at this rebal date AND
        #       (b) has all `lookback` past feature values (no NaN in the X[t] slice)
        univ_at_t = univ_dense[ti, :]                          # (N,)
        no_nan_in_X = ~np.isnan(X[t]).any(axis=(1, 2))         # (N,)
        # Note we do NOT require no-NaN in Y — that's the BUG-8 relaxation.
        # But we DO require the stock had a non-NaN return on the entry day
        # (otherwise the position can't be opened):
        has_entry_return = ~np.isnan(ret_dense_2d[holding_start, :])

        m = univ_at_t & no_nan_in_X & has_entry_return
        if m.sum() < min_stocks_per_rebal:
            continue

        mask[t] = m
        holding_days_arr[t] = H
        valid_rebal_indices.append(t)

    if not valid_rebal_indices:
        raise ValueError("No valid rebal dates! Check universe / data coverage.")

    # Compact: keep only valid rebal indices
    X = X[valid_rebal_indices]
    Y = Y[valid_rebal_indices]
    holding_days_arr = holding_days_arr[valid_rebal_indices]
    mask = mask[valid_rebal_indices]
    valid_rebal_dates = [rebal_dates[t] for t in valid_rebal_indices]

    return TCNNPanels(
        X=X, Y=Y, holding_days=holding_days_arr, mask=mask,
        rebal_dates=valid_rebal_dates, permno_idx=permno_idx,
        permno_list=list(all_permnos), feature_cols=feature_cols,
        lookback=lookback, entry_offset=entry_offset, max_holding=max_holding_days,
    )


# =============================================================================
# Save / load (npy + pickle metadata)
# =============================================================================

def save_panels(panels: TCNNPanels, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    np.save(os.path.join(out_dir, "X_panel.npy"), panels.X)
    np.save(os.path.join(out_dir, "Y_panel.npy"), panels.Y)
    np.save(os.path.join(out_dir, "holding_days.npy"), panels.holding_days)
    np.save(os.path.join(out_dir, "mask.npy"), panels.mask)
    meta = {
        "rebal_dates":  panels.rebal_dates,
        "permno_idx":   panels.permno_idx,
        "permno_list":  panels.permno_list,
        "feature_cols": panels.feature_cols,
        "lookback":     panels.lookback,
        "entry_offset": panels.entry_offset,
        "max_holding":  panels.max_holding,
    }
    with open(os.path.join(out_dir, "metadata.pkl"), "wb") as f:
        pickle.dump(meta, f)
    print(f"  saved panels to {out_dir}")
    print(f"  X: {panels.X.shape}  Y: {panels.Y.shape}  mask: {panels.mask.shape}")


def load_panels(in_dir: str, mmap: bool = True) -> TCNNPanels:
    mode = "r" if mmap else None
    X = np.load(os.path.join(in_dir, "X_panel.npy"), mmap_mode=mode)
    Y = np.load(os.path.join(in_dir, "Y_panel.npy"), mmap_mode=mode)
    holding_days = np.load(os.path.join(in_dir, "holding_days.npy"))
    mask = np.load(os.path.join(in_dir, "mask.npy"))
    with open(os.path.join(in_dir, "metadata.pkl"), "rb") as f:
        meta = pickle.load(f)
    return TCNNPanels(X=X, Y=Y, holding_days=holding_days, mask=mask, **meta)
