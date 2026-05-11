"""Portfolio-construction functions — score vector → dollar-neutral L/S weights.

Three implementations, identical interface:
    weights = fn(scores, **kwargs)
    where scores : np.ndarray (N,) cross-section of scores
          weights: np.ndarray (N,) dollar-neutral, sum |w| = gross_leverage

`dual_softmax` is differentiable (used during TCNN training).
`decile_sort_ew` is the academic standard (used by rungs 1, 2, 3 by default).
`mvo_lw_shrinkage` is mean-variance with Ledoit-Wolf shrinkage (TCNN eval extension).
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F


# =============================================================================
# Dual-softmax: differentiable, used in TCNN training
# =============================================================================

def dual_softmax_np(scores: np.ndarray, gross_leverage: float = 1.0,
                    eps: float = 1e-6) -> np.ndarray:
    """Cross-section z-score → dual-softmax → dollar-neutral L/S weights.

    Numpy version (for non-training paths).
    """
    u = np.asarray(scores, dtype=np.float64)
    u = u - u.mean()
    u = u / (u.std(ddof=0) + eps)
    long  = _softmax_np(u)
    short = _softmax_np(-u)
    w = long - short
    w = w - w.mean()
    w = w * (gross_leverage / (np.abs(w).sum() + 1e-12))
    return w


def dual_softmax_torch(u: torch.Tensor, gross_leverage: float = 1.0,
                        eps: float = 1e-6) -> torch.Tensor:
    """Dual-softmax in torch (used inside TCNN training for gradient flow)."""
    u = u - u.mean()
    u = u / (u.std(unbiased=False) + eps)
    long  = F.softmax(u, dim=0)
    short = F.softmax(-u, dim=0)
    w = long - short
    w = w - w.mean()
    w = w * (gross_leverage / (w.abs().sum() + 1e-12))
    return w


def dual_softmax_winsor_torch(u: torch.Tensor, gross_leverage: float = 1.0,
                                z_cap: float = 2.5, eps: float = 1e-6) -> torch.Tensor:
    """Winsorized dual-softmax in torch — z-scores clipped to [-z_cap, +z_cap].

    Differentiable drop-in replacement for `dual_softmax_torch`. Use this when
    factor scores have heavy tails (kurtosis >> 3); raw dual-softmax would
    over-concentrate on the extreme stocks (~3-5 names dominating gross exposure).
    With z_cap=2.5, effective N grows from ~3-30 stocks to ~400-700.
    """
    u = u - u.mean()
    u = u / (u.std(unbiased=False) + eps)
    u = torch.clamp(u, -z_cap, z_cap)
    long  = F.softmax(u, dim=0)
    short = F.softmax(-u, dim=0)
    w = long - short
    w = w - w.mean()
    w = w * (gross_leverage / (w.abs().sum() + 1e-12))
    return w


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


# =============================================================================
# Decile-sort equal-weighted: academic standard (not differentiable)
# =============================================================================

def decile_sort_ew(scores: np.ndarray, n_deciles: int = 10,
                    gross_leverage: float = 1.0) -> np.ndarray:
    """Long top decile, short bottom decile, equal-weighted within each.

    Standard L/S definition from the asset-pricing literature (e.g., JKP).
    The middle 8 deciles get weight 0. Sum |w| = gross_leverage; sum w = 0.
    """
    scores = np.asarray(scores, dtype=np.float64)
    N = len(scores)
    if N < n_deciles:
        return np.zeros(N)

    # Use rank to handle ties stably; pd.qcut equivalent
    ranks = scores.argsort().argsort()  # 0..N-1, ties broken by original index
    decile = (ranks * n_deciles // N).clip(0, n_deciles - 1)  # 0..9

    n_long  = (decile == n_deciles - 1).sum()
    n_short = (decile == 0).sum()

    w = np.zeros(N)
    if n_long > 0:
        w[decile == n_deciles - 1] = +0.5 * gross_leverage / n_long
    if n_short > 0:
        w[decile == 0] = -0.5 * gross_leverage / n_short

    # Re-center to enforce dollar neutrality (handles tied counts on long vs short)
    w = w - w.mean()
    w = w * (gross_leverage / (np.abs(w).sum() + 1e-12))
    return w


# =============================================================================
# Mean-variance with Ledoit-Wolf shrinkage (TCNN eval extension)
# =============================================================================

def mvo_lw_shrinkage(scores: np.ndarray, returns_history: np.ndarray,
                     gross_leverage: float = 1.0,
                     target_vol: float | None = None) -> np.ndarray:
    """Mean-variance optimization with Ledoit-Wolf shrinkage of the covariance.

    Args:
        scores: (N,) per-stock expected-return proxy (e.g., TCNN scores)
        returns_history: (T, N) past daily returns matrix for covariance estimate
        gross_leverage: target sum |w|; weights rescaled to match
        target_vol: if given, additionally rescale weights to hit annualized vol target

    Returns:
        (N,) weight vector, dollar-neutral, sum |w| = gross_leverage
    """
    from sklearn.covariance import LedoitWolf

    scores = np.asarray(scores, dtype=np.float64)
    R = np.asarray(returns_history, dtype=np.float64)
    N = len(scores)

    # Drop columns (stocks) with too few observations
    valid = ~np.isnan(R).any(axis=0)
    if valid.sum() < 2:
        return np.zeros(N)

    R_valid = R[:, valid]
    Sigma   = LedoitWolf().fit(R_valid).covariance_
    Sigma_inv = np.linalg.pinv(Sigma)  # pinv defends against residual rank issues

    mu_valid = scores[valid]
    raw_w = Sigma_inv @ mu_valid

    # Re-center (dollar-neutral) and rescale to gross_leverage
    raw_w = raw_w - raw_w.mean()
    raw_w = raw_w * (gross_leverage / (np.abs(raw_w).sum() + 1e-12))

    w = np.zeros(N)
    w[valid] = raw_w

    if target_vol is not None:
        # Scale to hit target annualized vol (computed in-sample)
        port_var = w[valid] @ Sigma @ w[valid]
        port_vol = np.sqrt(port_var * 252.0)
        if port_vol > 1e-8:
            w = w * (target_vol / port_vol)
    return w


# =============================================================================
# Portfolio-return computation given a weight vector + daily returns
# =============================================================================

def portfolio_returns_with_drift(w_initial: np.ndarray,
                                  daily_returns: np.ndarray) -> np.ndarray:
    """Compute portfolio daily returns over a holding period with weight drift.

    Args:
        w_initial: (N,) initial weights at start of holding period
        daily_returns: (H, N) per-stock daily returns over the holding period.
                       NaNs are treated as 0 (position has been moved to cash;
                       see BUG-8 fix in src.panels).

    Returns:
        (H,) daily portfolio returns
    """
    H, N = daily_returns.shape
    out = np.zeros(H, dtype=np.float64)
    w = w_initial.astype(np.float64).copy()
    for t in range(H):
        r_t = daily_returns[t].copy()
        r_t = np.where(np.isnan(r_t), 0.0, r_t)  # cash for delisted stocks
        port_ret = float((w * r_t).sum())
        out[t] = port_ret
        # Weight drift: w_new = w * (1 + r) / (1 + port_ret)
        w = w * (1 + r_t) / (1 + port_ret + 1e-8)
    return out


# =============================================================================
# Dispatcher (for runner.py)
# =============================================================================

def dual_softmax_winsor(scores, gross_leverage: float = 1.0, z_cap: float = 2.5,
                         eps: float = 1e-6):
    """Dual-softmax with z-scores clipped to [-z_cap, +z_cap] before softmax.

    Fully differentiable (clip is differentiable except at endpoints), so this
    can be used as a drop-in for `dual_softmax_np` in TCNN training too.
    With z_cap=2.5: kills the fat-tail concentration that gives raw dual-softmax
    its 3-stock effective universe on heavy-tailed factor scores.
    """
    u = np.asarray(scores, dtype=np.float64)
    u = (u - u.mean()) / (u.std(ddof=0) + eps)
    u = np.clip(u, -z_cap, z_cap)
    long  = _softmax_np(u)
    short = _softmax_np(-u)
    w = long - short
    w = w - w.mean()
    w = w * (gross_leverage / (np.abs(w).sum() + 1e-12))
    return w


def dual_softmax_rank(scores, gross_leverage: float = 1.0, eps: float = 1e-6):
    """Dual-softmax on cross-sectional ranks instead of raw z-scores.

    By construction, ranks are uniform in [1, N], so after z-scoring they
    have a stable distribution regardless of the original scores' kurtosis.
    NOT differentiable (hard rank); use only as an evaluation portfolio
    or implement via torchsort for differentiable training.
    """
    s = np.asarray(scores, dtype=np.float64)
    ranks = pd.Series(s).rank(method="average").values
    u = (ranks - ranks.mean()) / (ranks.std(ddof=0) + eps)
    long  = _softmax_np(u)
    short = _softmax_np(-u)
    w = long - short
    w = w - w.mean()
    w = w * (gross_leverage / (np.abs(w).sum() + 1e-12))
    return w


PORTFOLIO_DISPATCHER = {
    "dual_softmax":         dual_softmax_np,
    "dual_softmax_winsor":  dual_softmax_winsor,
    "dual_softmax_rank":    dual_softmax_rank,
    "decile_sort_ew":       decile_sort_ew,
    "mvo_lw":               mvo_lw_shrinkage,
}


def get_portfolio_fn(name: str):
    """Look up a portfolio constructor by name (used by runner)."""
    if name not in PORTFOLIO_DISPATCHER:
        raise ValueError(f"Unknown portfolio: {name}. Choices: {list(PORTFOLIO_DISPATCHER)}")
    return PORTFOLIO_DISPATCHER[name]
