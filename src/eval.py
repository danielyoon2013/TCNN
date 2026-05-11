"""Analytics on (date, return) daily portfolio P&L series + (date, permno, score) signal panels.

Functions here are read-only on existing results — they don't train or rebalance.
Used by notebooks 04-08 to compute the diagnostics for the ladder summary,
factor-neutrality check, capacity sweep, etc.
"""

from __future__ import annotations
import numpy as np
import pandas as pd

from . import config


# =============================================================================
# Sharpe / drawdown / basic perf
# =============================================================================

def annualized_sharpe(daily_returns: np.ndarray | pd.Series, periods_per_year: int = 252) -> float:
    r = pd.Series(daily_returns).dropna().values
    if len(r) < 2:
        return float("nan")
    mu = r.mean()
    sd = r.std(ddof=0)
    return float(mu / (sd + 1e-12) * np.sqrt(periods_per_year))


def max_drawdown(daily_returns: np.ndarray | pd.Series) -> float:
    r = pd.Series(daily_returns).dropna().values
    if len(r) == 0:
        return 0.0
    cum = (1.0 + r).cumprod()
    rmax = np.maximum.accumulate(cum)
    return float(((cum - rmax) / rmax).min())


def annualized_return(daily_returns: np.ndarray | pd.Series, periods_per_year: int = 252) -> float:
    r = pd.Series(daily_returns).dropna().values
    if len(r) == 0:
        return 0.0
    return float((1.0 + r).prod() ** (periods_per_year / len(r)) - 1.0)


def perf_summary(daily_returns: pd.Series) -> dict:
    """Standard PM-ready summary: ret, vol, Sharpe, max DD, t-stat."""
    r = daily_returns.dropna().values
    if len(r) < 2:
        return {"sharpe": float("nan")}
    mu = r.mean(); sd = r.std(ddof=0)
    sharpe = mu / (sd + 1e-12) * np.sqrt(252)
    return {
        "ann_return": (1 + r).prod() ** (252 / len(r)) - 1,
        "ann_vol":    sd * np.sqrt(252),
        "sharpe":     sharpe,
        "max_dd":     max_drawdown(r),
        "t_stat":     mu / (sd / np.sqrt(len(r)) + 1e-12),
        "n_days":     len(r),
    }


# =============================================================================
# IC analysis (rank correlation between scores and forward returns)
# =============================================================================

def ic_by_month(panel: pd.DataFrame, score_col: str, ret_fut_col: str = "ret_fut_1m",
                 universe_col: str = "in_top_2000") -> pd.Series:
    """Per-rebal-date Spearman IC between score and forward 1-month return.

    Returns a date-indexed Series of monthly IC values.
    """
    df = panel[[score_col, ret_fut_col, universe_col, "date"]].copy()
    df = df[df[universe_col].astype(bool)]
    df = df.dropna(subset=[score_col, ret_fut_col])

    def _ic(g):
        if len(g) < 30:
            return np.nan
        return g[score_col].rank().corr(g[ret_fut_col].rank())

    return df.groupby("date").apply(_ic)


def ic_summary(ic_series: pd.Series) -> dict:
    """Mean IC, IC vol, IR, hit rate, FLAM-implied IR target."""
    ic = ic_series.dropna()
    if len(ic) == 0:
        return {"mean_ic": np.nan}
    return {
        "mean_ic":  float(ic.mean()),
        "ic_vol":   float(ic.std(ddof=0)),
        "ir":       float(ic.mean() / (ic.std(ddof=0) + 1e-12)),
        "hit_rate": float((ic > 0).mean()),
        "n_months": int(len(ic)),
    }


# =============================================================================
# Decile anatomy
# =============================================================================

def decile_returns(panel: pd.DataFrame, score_col: str,
                    ret_fut_col: str = "ret_fut_1m",
                    universe_col: str = "in_top_2000",
                    n_deciles: int = 10) -> pd.DataFrame:
    """Per (rebal_date, decile) avg forward return.

    Returns long-format DataFrame: columns = [date, decile, n, mean_ret, sum_ret].
    """
    df = panel[[score_col, ret_fut_col, universe_col, "date"]].copy()
    df = df[df[universe_col].astype(bool)]
    df = df.dropna(subset=[score_col, ret_fut_col])

    df["decile"] = (
        df.groupby("date", group_keys=False)[score_col]
          .transform(lambda s: pd.qcut(s.rank(method="first"), n_deciles, labels=False, duplicates="drop") + 1)
    )

    out = df.groupby(["date", "decile"]).agg(
        n=(ret_fut_col, "size"),
        mean_ret=(ret_fut_col, "mean"),
    ).reset_index()
    return out


def decile_spread_summary(decile_df: pd.DataFrame, n_deciles: int = 10) -> dict:
    """High-low spread (D{n} − D1) and monotonicity check."""
    pivot = decile_df.pivot(index="date", columns="decile", values="mean_ret")
    if 1 not in pivot.columns or n_deciles not in pivot.columns:
        return {"hi_lo_mean": np.nan}

    hi_lo = pivot[n_deciles] - pivot[1]
    means = pivot.mean()  # decile-level mean across dates
    monotonic = bool((means.diff().dropna() > 0).mean() > 0.7)  # >70% of adjacent pairs increasing

    return {
        "hi_lo_mean":  float(hi_lo.mean()),
        "hi_lo_t":     float(hi_lo.mean() / (hi_lo.std(ddof=0) / np.sqrt(len(hi_lo)) + 1e-12)),
        "monotonic":   monotonic,
        "decile_means": means.to_dict(),
    }


# =============================================================================
# Factor-neutrality regression (FF5 + UMD)
# =============================================================================

def ff5_neutrality(daily_returns: pd.Series, ff5: pd.DataFrame,
                    include_umd: bool = True) -> dict:
    """Regress daily strategy returns on FF5 + (optional) UMD; report alpha + factor exposures.

    Args:
        daily_returns: pd.Series indexed by date
        ff5: DataFrame with columns mktrf, smb, hml, rmw, cma (and optionally umd), rf
        include_umd: include momentum factor as additional regressor

    Returns dict with annualized alpha, t-stat, R², factor betas.
    """
    import statsmodels.api as sm

    r = daily_returns.copy().rename("strat")
    df = ff5.merge(r, left_on="date", right_index=True)
    df = df.dropna()
    if len(df) < 252:
        return {"alpha_ann": np.nan}

    factors = ["mktrf", "smb", "hml", "rmw", "cma"]
    if include_umd and "umd" in df.columns:
        factors.append("umd")

    X = sm.add_constant(df[factors].values)
    y = (df["strat"] - df.get("rf", 0.0)).values
    fit = sm.OLS(y, X).fit()

    alpha_daily = fit.params[0]
    return {
        "alpha_ann":   float(alpha_daily * 252),
        "alpha_t":     float(fit.tvalues[0]),
        "r_squared":   float(fit.rsquared),
        "betas":       {f: float(b) for f, b in zip(factors, fit.params[1:])},
        "n_obs":       int(len(df)),
    }


# =============================================================================
# TC sweep (locked: fixed-cost assumption, not per-stock)
# =============================================================================

def turnover_per_rebal(weights_history: pd.DataFrame) -> pd.Series:
    """One-way turnover at each rebal date.

    Args:
        weights_history: DataFrame indexed by rebal_date, columns = permno,
                          values = post-rebal weight (incl. zeros for non-held).

    Returns:
        pd.Series indexed by rebal_date (excluding the first), values = 0.5 * sum |w_new - w_old|.
        Drift between rebals is NOT counted as turnover (only forced trades at rebal).
    """
    w = weights_history.fillna(0.0).sort_index()
    diff = w.diff().abs().sum(axis=1) * 0.5
    return diff.dropna()


def net_returns_under_cost(daily_returns: pd.Series, turnover: pd.Series,
                            cost_bps_round_trip: float = 10.0,
                            rebal_dates: pd.DatetimeIndex | None = None) -> pd.Series:
    """Subtract turnover-proportional cost from gross returns at each rebal date.

    Args:
        daily_returns: gross daily portfolio returns indexed by date
        turnover: per-rebal-date one-way turnover (sum |Δw| / 2)
        cost_bps_round_trip: assumed round-trip cost in bps (e.g., 10 = 10bps)
        rebal_dates: dates on which to apply cost; if None, inferred from turnover index

    Returns:
        net_daily_returns: pd.Series, same index as daily_returns
    """
    if rebal_dates is None:
        rebal_dates = turnover.index

    cost_per_rebal = turnover * (cost_bps_round_trip / 1e4)  # convert bps to fraction
    out = daily_returns.copy()
    for rd, c in cost_per_rebal.items():
        if rd in out.index:
            out.loc[rd] = out.loc[rd] - c
    return out


def tc_sweep(daily_returns: pd.Series, turnover: pd.Series,
              cost_grid_bps: tuple[float, ...] = (0, 5, 10, 20)) -> pd.DataFrame:
    """Net Sharpe at multiple round-trip cost assumptions.

    Returns DataFrame: rows = cost_bps, cols = [sharpe, ann_return, max_dd, ann_turnover_pct].
    """
    rows = []
    n_years = (daily_returns.index.max() - daily_returns.index.min()).days / 365.25
    ann_turn_pct = float(turnover.sum() / n_years * 100) if n_years > 0 else float("nan")

    for c_bps in cost_grid_bps:
        net = net_returns_under_cost(daily_returns, turnover, cost_bps_round_trip=c_bps)
        rows.append({
            "cost_bps":           c_bps,
            "sharpe":             annualized_sharpe(net),
            "ann_return":         annualized_return(net),
            "max_dd":             max_drawdown(net),
            "ann_turnover_pct":   ann_turn_pct,
        })
    return pd.DataFrame(rows)


# =============================================================================
# Capacity (sqrt-impact-law back-of-envelope)
# =============================================================================

def capacity_curve(daily_returns: pd.Series, turnover: pd.Series,
                    aum_grid_usd: tuple[float, ...] = (50e6, 200e6, 1e9, 5e9),
                    base_aum_usd: float = 50e6,
                    base_cost_bps_round_trip: float = 10.0,
                    impact_exponent: float = 0.5) -> pd.DataFrame:
    """Estimate net Sharpe as AUM scales, using sqrt-impact-law:

        cost_bps(AUM) = base_cost * (AUM / base_AUM) ** impact_exponent

    Args:
        daily_returns: gross daily portfolio returns
        turnover: per-rebal turnover series
        aum_grid_usd: AUM levels to evaluate
        base_aum_usd: AUM at which `base_cost_bps_round_trip` applies
        base_cost_bps_round_trip: round-trip cost at base AUM
        impact_exponent: 0.5 = sqrt law; 0 = constant cost regardless of AUM

    Returns DataFrame: cols = [aum_usd, cost_bps, sharpe, ann_return, max_dd].
    """
    rows = []
    for aum in aum_grid_usd:
        c_bps = base_cost_bps_round_trip * (aum / base_aum_usd) ** impact_exponent
        net = net_returns_under_cost(daily_returns, turnover, cost_bps_round_trip=c_bps)
        rows.append({
            "aum_usd":   aum,
            "cost_bps":  c_bps,
            "sharpe":    annualized_sharpe(net),
            "ann_return": annualized_return(net),
            "max_dd":    max_drawdown(net),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Multi-experiment aggregation (read all_results.csv per rung, return one DF)
# =============================================================================

def load_master_results(experiments: list[str], outputs_root: str | None = None) -> pd.DataFrame:
    """Concatenate every experiment's all_results.csv into one master DataFrame.

    Args:
        experiments: list of experiment_id strings (each is a subdir of outputs/)
        outputs_root: defaults to config.OUTPUTS_DIR
    """
    import os
    outputs_root = outputs_root or str(config.OUTPUTS_DIR)
    dfs = []
    for exp_id in experiments:
        path = os.path.join(outputs_root, exp_id, "all_results.csv")
        if not os.path.exists(path):
            print(f"  missing: {path}")
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        if "experiment_id" not in df.columns:
            df["experiment_id"] = exp_id
        dfs.append(df)
    return pd.concat(dfs, ignore_index=True)
