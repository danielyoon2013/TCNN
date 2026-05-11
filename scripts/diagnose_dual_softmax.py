"""Diagnose the dual-softmax concentration issue.

For multiple rebal dates, compare weight distributions of:
  - decile_sort_ew
  - dual_softmax (current implementation)
  - dual_softmax with temperature scaling (sharper or smoother)
  - rank-based dual_softmax (replace u with cross-sectional rank, then z-score)

For each: report effective N (1/sum(w^2)), top-5 weight share, top-20 weight share,
and the realized portfolio vol on the holding period after the rebal.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import scipy.special

from src import config, portfolio


def softmax_np(x):
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()


def dual_softmax_temp(scores, temperature=1.0, gross_leverage=1.0, eps=1e-6):
    """Dual-softmax with temperature: w = softmax(u/T) - softmax(-u/T)."""
    u = np.asarray(scores, dtype=np.float64)
    u = (u - u.mean()) / (u.std(ddof=0) + eps)
    u = u / temperature
    long  = softmax_np(u)
    short = softmax_np(-u)
    w = long - short
    w = w - w.mean()
    w = w * (gross_leverage / (np.abs(w).sum() + 1e-12))
    return w


def dual_softmax_rank(scores, gross_leverage=1.0, eps=1e-6):
    """Dual-softmax on ranks instead of raw z-scores. Bounded scores → less concentration."""
    ranks = pd.Series(scores).rank(method="average").values  # 1..N
    u = (ranks - ranks.mean()) / (ranks.std(ddof=0) + eps)   # cross-sectional z of ranks
    long  = softmax_np(u)
    short = softmax_np(-u)
    w = long - short
    w = w - w.mean()
    w = w * (gross_leverage / (np.abs(w).sum() + 1e-12))
    return w


def winsorize_dual_softmax(scores, z_cap=3.0, gross_leverage=1.0, eps=1e-6):
    """Dual-softmax with z-scores winsorized at +/- z_cap before softmax."""
    u = np.asarray(scores, dtype=np.float64)
    u = (u - u.mean()) / (u.std(ddof=0) + eps)
    u = np.clip(u, -z_cap, z_cap)
    long  = softmax_np(u)
    short = softmax_np(-u)
    w = long - short
    w = w - w.mean()
    w = w * (gross_leverage / (np.abs(w).sum() + 1e-12))
    return w


def weight_diagnostics(weights):
    w = np.asarray(weights)
    abs_w = np.abs(w)
    total_abs = abs_w.sum() + 1e-12
    sorted_abs = np.sort(abs_w)[::-1]   # descending
    cumshare = sorted_abs.cumsum() / total_abs
    eff_N = 1.0 / (w ** 2).sum() if (w ** 2).sum() > 0 else float("inf")
    n_active = (abs_w > 1e-6).sum()
    return {
        "eff_N":         float(eff_N),
        "n_active":      int(n_active),
        "top1_share":    float(cumshare[0]),
        "top5_share":    float(cumshare[4]),
        "top20_share":   float(cumshare[19]),
        "max_long":      float(w.max()),
        "max_short":     float(-w.min()),
        "n_long":        int((w > 1e-6).sum()),
        "n_short":       int((w < -1e-6).sum()),
        "sum_abs":       float(total_abs),
    }


# Load panel
panel = pd.read_parquet(config.PANEL_DAILY_PARQUET)
panel["date"] = pd.to_datetime(panel["date"])

SAMPLE_DATES = ["2010-01-29", "2015-06-30", "2020-02-28", "2022-12-30"]
factors = ["mom_12_1", "ewm_h60_skip"]

for date_str in SAMPLE_DATES:
    rebal_date = pd.Timestamp(date_str)
    slc = panel[(panel["date"] == rebal_date) &
                panel["in_top_2000"].fillna(False) &
                panel["price_above_5"].fillna(False) &
                panel["adtv_above_5m"].fillna(False)]

    print(f"\n{'=' * 80}")
    print(f"REBAL DATE: {date_str}   (universe = {len(slc)} stocks after filters)")
    print(f"{'=' * 80}")

    for fac in factors:
        scores = slc[fac].values
        valid = ~np.isnan(scores)
        s = scores[valid]
        N = len(s)
        if N == 0:
            continue
        z = (s - s.mean()) / s.std(ddof=0)
        print(f"\n  --- {fac} (N = {N}) ---")
        print(f"    score z-distribution: min={z.min():+.2f}, max={z.max():+.2f}, "
              f"5%-tile={np.percentile(z, 5):+.2f}, 95%-tile={np.percentile(z, 95):+.2f}, "
              f"kurt={float(pd.Series(z).kurt()):+.2f}  (gaussian kurt=0)")

        schemes = [
            ("decile_sort_ew",            portfolio.decile_sort_ew(s)),
            ("dual_softmax (T=1.0)",      dual_softmax_temp(s, temperature=1.0)),
            ("dual_softmax (T=2.0)",      dual_softmax_temp(s, temperature=2.0)),
            ("dual_softmax (T=3.0)",      dual_softmax_temp(s, temperature=3.0)),
            ("dual_softmax_rank",         dual_softmax_rank(s)),
            ("dual_softmax winsor z=2.5", winsorize_dual_softmax(s, z_cap=2.5)),
            ("dual_softmax winsor z=1.5", winsorize_dual_softmax(s, z_cap=1.5)),
        ]
        print(f"    {'scheme':<28} {'eff_N':>8}  {'top1':>6}  {'top5':>6}  {'top20':>6}  {'n_long':>6} {'n_short':>6}  {'max_l':>6}  {'max_s':>6}")
        for name, w in schemes:
            d = weight_diagnostics(w)
            print(f"    {name:<28} {d['eff_N']:8.1f}  {d['top1_share']*100:5.2f}%  "
                  f"{d['top5_share']*100:5.2f}%  {d['top20_share']*100:5.2f}%  "
                  f"{d['n_long']:>6} {d['n_short']:>6}  "
                  f"{d['max_long']*100:5.2f}%  {d['max_short']*100:5.2f}%")
