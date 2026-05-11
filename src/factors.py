"""Factor scoring functions — rungs 1, 2, 3 of the comparison ladder.

Each function takes a per-rebal-date slice of `panel_daily.parquet`
(filtered to that date's universe) and returns a numpy array of scores
indexed in the same order as the rows.

Rung 4 (linear TCNN), 5 (1-channel TCNN), 6 (3-channel TCNN) live in
`src.train_tcnn` since they're trained models, not pure functions.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from . import config


# =============================================================================
# Rung 1: Simple 12-1 momentum
# =============================================================================

def score_simple_momentum(panel_at_date: pd.DataFrame) -> np.ndarray:
    """Rung 1. Score = `mom_12_1` column — already precomputed in features.py.

    Equation: score[i, t] = sum_{l=22}^{252} ret[i, t-l]
    Hand-crafted uniform weight on lags 22..252, zero elsewhere.
    """
    return panel_at_date["mom_12_1"].values.astype(np.float64)


# =============================================================================
# Rung 2: EWM momentum, skip-month, half-life=60d
# =============================================================================

def score_ewm_momentum(panel_at_date: pd.DataFrame) -> np.ndarray:
    """Rung 2. Score = `ewm_h60_skip` column — precomputed in features.py.

    Equation: score[i, t] = sum_{l=22}^{infty} alpha*(1-alpha)^(l-22) * ret[i, t-l]
    Hand-crafted exponentially-decaying weight starting at lag 22.
    """
    return panel_at_date["ewm_h60_skip"].values.astype(np.float64)


# =============================================================================
# Rung 3: Time-series regression on lag returns (pooled OLS, ridge-regularized)
# =============================================================================

def _build_lag_matrix(panel: pd.DataFrame, n_lags: int) -> tuple[np.ndarray, np.ndarray]:
    """Compute lag-return matrix on-the-fly from `ret` column (saves 21 GB of disk).

    For each (date, permno) row in `panel`, returns a row [ret_{t-1}, ret_{t-2}, ...,
    ret_{t-n_lags}]. Vectorized via groupby+shift.

    Args:
        panel: long-format panel with at least 'permno', 'date', 'ret' columns,
               sorted by ['permno', 'date'].
        n_lags: number of lags to compute (paper: 252 = LOOKBACK_DAYS)

    Returns:
        X: (N_rows, n_lags) numpy array; NaN rows for stocks with insufficient history
        valid_mask: (N_rows,) bool — True where ALL n_lags lags are non-NaN
    """
    panel = panel.sort_values(["permno", "date"])
    grp = panel.groupby("permno", sort=False)["ret"]
    cols = []
    for L in range(1, n_lags + 1):
        cols.append(grp.shift(L).values.astype(np.float64))
    X = np.column_stack(cols)
    valid_mask = ~np.isnan(X).any(axis=1)
    return X, valid_mask


def fit_ts_regression(
    train_panel: pd.DataFrame,
    target_col: str = "ret_fut_1m",
    n_lags: int = config.LOOKBACK_DAYS,
    ridge_alpha: float = 1.0,
) -> np.ndarray:
    """Fit pooled ridge OLS of forward return on lag returns (computed on-the-fly).

    Pooled = each (stock, rebal_date) pair is one observation. With ~96 train months
    × ~2000 stocks ≈ 192K observations and 252 lag regressors, the fit is well-
    determined. Ridge regularization protects against multicollinearity.

    Lag features are computed on-the-fly from `ret` to avoid storing 252 large
    columns in panel_daily.parquet.

    Args:
        train_panel: long-format slice covering the training period and universe.
                     Must have 'permno', 'date', 'ret', and `target_col`.
        target_col: column with next-month forward return (e.g. ret_fut_1m)
        n_lags: number of lag returns to use as regressors (paper: 252)
        ridge_alpha: L2 regularization strength

    Returns:
        beta: (n_lags + 1,) array of OLS coefficients including intercept
    """
    from sklearn.linear_model import Ridge

    panel = train_panel.sort_values(["permno", "date"]).reset_index(drop=True)
    X_full, lag_valid = _build_lag_matrix(panel, n_lags)
    y_full = panel[target_col].values.astype(np.float64)

    valid = lag_valid & ~np.isnan(y_full)
    if valid.sum() < 1000:
        raise ValueError(f"Too few training observations: {valid.sum()} (need >= 1000)")

    X = X_full[valid]
    y = y_full[valid]

    model = Ridge(alpha=ridge_alpha, fit_intercept=True)
    model.fit(X, y)

    beta = np.concatenate([[model.intercept_], model.coef_])
    return beta


def score_ts_regression(panel_at_date: pd.DataFrame, beta: np.ndarray,
                         n_lags: int = config.LOOKBACK_DAYS) -> np.ndarray:
    """Rung 3 score: lag features computed on-the-fly, then apply beta.

    For scoring at a single rebal date, we need the lag matrix only for those rows.
    But since lag values are functions of past returns from the FULL panel, we
    can't compute them from `panel_at_date` alone (no history there).

    Caller should pass `panel_at_date` that already has the necessary lags via
    `panel_with_lags` from build_lag_matrix_for_date(). Or simpler: pass the full
    panel and the rebal_date, and we filter inside.

    For simplicity here, we expect `panel_at_date` to have a precomputed `_lags`
    attribute with shape (n_rows, n_lags). If not, fall back to computing inline
    (which requires the full panel context — caller responsibility).

    Args:
        panel_at_date: rows for the current rebal date and universe. Must have
                       columns starting with '_lag_1', '_lag_2', ..., '_lag_n_lags'
                       OR a precomputed `_lag_matrix` attribute.
        beta: from fit_ts_regression(); first element is intercept
        n_lags: number of lags

    Returns:
        (N,) score vector
    """
    # Look for inline lag columns (added by the runner before calling)
    lag_cols = [f"_lag_{l}" for l in range(1, n_lags + 1)]
    if all(c in panel_at_date.columns for c in lag_cols):
        X = panel_at_date[lag_cols].values.astype(np.float64)
    else:
        raise ValueError(
            f"score_ts_regression: panel_at_date must have _lag_1.._lag_{n_lags} columns. "
            "Caller (runner._run_cell_baseline) should compute these inline before calling."
        )
    intercept, coefs = beta[0], beta[1:]
    return intercept + X @ coefs


def build_lag_features_for_dates(
    full_panel: pd.DataFrame,
    target_dates: list,
    n_lags: int = config.LOOKBACK_DAYS,
) -> pd.DataFrame:
    """Compute lag-return features only for the specified dates, then return a
    long-format DataFrame with columns _lag_1, _lag_2, ..., _lag_{n_lags}.

    Used by the runner when scoring rung 3 (ts_reg): for each rebal date, we need
    the lag vector for each stock at that date.

    Args:
        full_panel: full panel sorted by ['permno', 'date']; must have 'ret'.
        target_dates: list of dates for which to compute lag features.
        n_lags: number of lags.

    Returns:
        DataFrame with ['date', 'permno', '_lag_1', ..., '_lag_n_lags'] columns,
        rows only for (date, permno) pairs in target_dates.
    """
    panel = full_panel.sort_values(["permno", "date"]).reset_index(drop=True)
    grp = panel.groupby("permno", sort=False)["ret"]
    lag_data = {f"_lag_{L}": grp.shift(L).values for L in range(1, n_lags + 1)}
    out = pd.DataFrame({"date": panel["date"].values, "permno": panel["permno"].values, **lag_data})
    return out[out["date"].isin(set(pd.to_datetime(target_dates)))].reset_index(drop=True)


# =============================================================================
# Dispatcher
# =============================================================================

FACTOR_DISPATCHER = {
    "simple_mom": score_simple_momentum,
    "ewm_mom":    score_ewm_momentum,
    "ts_reg":     score_ts_regression,   # needs precomputed beta
}


def get_factor_fn(name: str):
    if name not in FACTOR_DISPATCHER:
        raise ValueError(f"Unknown factor: {name}. Choices: {list(FACTOR_DISPATCHER)}")
    return FACTOR_DISPATCHER[name]
