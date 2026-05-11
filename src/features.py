"""Feature engineering on the long-format daily panel.

All functions:
  - take a panel DataFrame (long-format, sorted by [permno, date])
  - return the same DataFrame with new columns added
  - are strictly point-in-time (use only past data)

Critical features for the 9-rung ladder:
  - vol_ewma_lag, ret_norm  (BUG-1 fix: lagged vol)
  - mom_12_1                 (rung 1 score input)
  - ewm_h60_skip             (rung 2 score input)
  - lag_ret_1 ... lag_ret_252 (rung 3 OLS regressors; same lookback as TCNN)
  - in_top_N, price_above_5, adtv_above_5m  (universe flags)
  - ret_fut_1m_T1            (forward 1-month return with T+1 entry — BUG-7 fix)
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from . import config


# =============================================================================
# Market cap (after merge_delisting_returns has run)
# =============================================================================

def add_market_cap(panel: pd.DataFrame) -> pd.DataFrame:
    """mktcap_dollars = abs(prc) * shrout * 1000 (shrout is in thousands)."""
    out = panel.copy()
    out["mktcap_dollars"] = out["prc"].abs() * out["shrout"] * 1000.0
    return out


# =============================================================================
# EWMA volatility + lagged-vol-normalized return  (BUG-1 fix)
# =============================================================================

def add_ewma_vol(panel: pd.DataFrame, half_life: int = config.EWMA_HALF_LIFE,
                  eps: float = 1e-8) -> pd.DataFrame:
    """EWMA daily volatility (causal) + ret_norm using LAGGED vol.

    BUG-1 fix: ret_norm = ret / vol_ewma_lag (yesterday's vol).
    The original code divided by vol_ewma (today's, which includes ret_t^2) — leak.
    """
    out = panel.sort_values(["permno", "date"]).copy()
    out["ret"] = pd.to_numeric(out["ret"], errors="coerce")

    lam = 2 ** (-1.0 / float(half_life))
    r2 = out["ret"] ** 2
    out["ewma_r2"] = (
        r2.groupby(out["permno"])
          .apply(lambda s: s.ewm(alpha=(1 - lam), adjust=False).mean())
          .reset_index(level=0, drop=True)
    )
    out["vol_ewma"]     = np.sqrt(out["ewma_r2"])
    out["vol_ewma_lag"] = out.groupby("permno")["vol_ewma"].shift(1)
    out["ret_norm"]     = out["ret"] / (out["vol_ewma_lag"] + eps)  # lagged → strictly causal
    out.drop(columns=["ewma_r2"], inplace=True)
    return out


# =============================================================================
# Momentum factor inputs (rungs 1, 2)
# =============================================================================

def add_momentum_factors(panel: pd.DataFrame) -> pd.DataFrame:
    """Hand-crafted factor scores for rungs 1 and 2.

    mom_12_1: 12-1 momentum = sum(ret[t-22:t-252]). Uniform weights on lags 22-252.
    ewm_h60_skip: skip-month exponential. Sum starting at lag 22 with EWM decay.
    """
    out = panel.sort_values(["permno", "date"]).copy()
    grp = out.groupby("permno", sort=False)["ret"]

    # mom_12_1: rolling sum over [t-252, t-22], i.e., shift(22) then rolling(231).sum()
    out["mom_12_1"] = grp.transform(
        lambda s: s.shift(config.SKIP_MONTH_DAYS).rolling(
            config.LOOKBACK_DAYS - config.SKIP_MONTH_DAYS, min_periods=100
        ).sum()
    )

    # ewm_h60_skip: shift(22) then EWM mean (then * window for "sum" interpretation;
    #               we use mean which is just a different scaling — what matters
    #               cross-sectionally is the rank, not the absolute level)
    out["ewm_h60_skip"] = grp.transform(
        lambda s: s.shift(config.SKIP_MONTH_DAYS).ewm(
            halflife=config.EWM_MOM_HALF_LIFE, adjust=False, min_periods=60
        ).mean()
    )
    return out


# =============================================================================
# Lag features for rung 3 (TS regression)
# =============================================================================

def add_lag_features(panel: pd.DataFrame, n_lags: int = config.LOOKBACK_DAYS) -> pd.DataFrame:
    """Add lag_ret_1, lag_ret_2, ..., lag_ret_252 columns.

    Used as the regressor matrix for rung 3 (pooled OLS / ridge).
    Same lookback as the TCNN input — so rungs 3, 4, 5 see the same information.
    """
    out = panel.sort_values(["permno", "date"]).copy()
    grp = out.groupby("permno", sort=False)["ret"]
    for L in range(1, n_lags + 1):
        out[f"lag_ret_{L}"] = grp.shift(L)
    return out


# =============================================================================
# Universe membership flags (point-in-time, per-date)
# =============================================================================

def add_universe_flags(panel: pd.DataFrame,
                        top_n_list: tuple[int, ...] = (500, 1000, 2000),
                        min_price: float = config.MIN_PRICE_DEFAULT,
                        min_adtv: float = config.MIN_ADTV_USD,
                        adtv_window: int = 21) -> pd.DataFrame:
    """Cross-sectional ranking + tradability filters as boolean panel columns.

    - mktcap_rank (1 = largest)
    - in_top_500, in_top_1000, in_top_2000
    - price_above_5
    - adtv_21d_usd (rolling 21-day avg dollar volume)
    - adtv_above_5m
    - vol_21d (rolling 21-day annualized realized vol — used by capacity sizing)

    All strictly point-in-time at end-of-day.
    """
    out = panel.sort_values(["date", "permno"]).copy()

    # Cross-sectional rank by mktcap each date (1 = largest)
    out["mktcap_rank"] = out.groupby("date")["mktcap_dollars"].rank(ascending=False, method="min")
    for n in top_n_list:
        out[f"in_top_{n}"] = out["mktcap_rank"] <= n

    out["price_above_5"] = out["prc"].abs() >= min_price

    # ADTV: 21-day rolling avg of dollar volume (per stock)
    out["dollar_vol"] = out["prc"].abs() * out["vol"]
    out["adtv_21d_usd"] = (out.sort_values(["permno", "date"])
                              .groupby("permno", sort=False)["dollar_vol"]
                              .rolling(adtv_window, min_periods=15)
                              .mean()
                              .reset_index(level=0, drop=True))
    out["adtv_above_5m"] = out["adtv_21d_usd"] >= min_adtv

    # Realized vol (rolling 21d, annualized)
    out["vol_21d"] = (out.sort_values(["permno", "date"])
                          .groupby("permno", sort=False)["ret"]
                          .rolling(adtv_window, min_periods=15)
                          .std()
                          .reset_index(level=0, drop=True) * np.sqrt(252.0))

    out.drop(columns=["dollar_vol"], inplace=True)
    return out


# =============================================================================
# Forward returns with T+1 entry  (BUG-7 fix)
# =============================================================================

def add_forward_returns(panel: pd.DataFrame, horizons_days: tuple[int, ...] = (1, 5, 21),
                         entry_offset: int = config.ENTRY_OFFSET_DAYS) -> pd.DataFrame:
    """T+1-entry compounded forward returns at multiple horizons.

    For signal date T, ret_fut_h returns the compounded return from
    close[T+entry_offset] to close[T+entry_offset+h]. With entry_offset=1:
      - signal computed at end of day T
      - hypothetical entry at close of day T+1
      - first holding-period day's contribution is ret[T+2]

    This eliminates the soft 1-day look-ahead present in the original code.
    """
    out = panel.sort_values(["permno", "date"]).copy()
    out["_log_ret"] = np.log1p(pd.to_numeric(out["ret"], errors="coerce").fillna(0.0))

    def _add(g):
        c = g["_log_ret"].cumsum()
        for h in horizons_days:
            g[f"ret_fut_{h}d"] = np.exp(c.shift(-(entry_offset + h)) - c.shift(-entry_offset)) - 1.0
        return g

    out = out.groupby("permno", group_keys=False).apply(_add)
    out.drop(columns=["_log_ret"], inplace=True)

    # Convenience: forward 1-month return (~21 trading days, the standard horizon)
    if 21 in horizons_days:
        out["ret_fut_1m"] = out["ret_fut_21d"]
    return out


# =============================================================================
# Master orchestrator (one-stop convenience for notebook 01)
# =============================================================================

def build_panel(daily_clean: pd.DataFrame) -> pd.DataFrame:
    """Apply all feature additions in canonical order. Used by notebook 01."""
    out = add_market_cap(daily_clean)
    out = add_ewma_vol(out)
    out = add_momentum_factors(out)
    out = add_lag_features(out)
    out = add_universe_flags(out)
    out = add_forward_returns(out)
    return out
