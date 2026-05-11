"""Quick CLI summary of completed ladder rungs. Loads outputs/<rung>/all_results.csv.

Usage:
    python scripts/ladder_summary.py

Shows: per-rung Sharpe / ann_ret / ann_vol / max_dd table + the F3 decomposition.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from src import config

ALL_RUNGS = [
    "rung_1_simple_momentum_decile",
    "rung_1d_simple_momentum_dual",
    "rung_2_ewm_momentum_decile",
    "rung_2d_ewm_momentum_dual",
    "rung_3_ts_regression_decile",
    "rung_3d_ts_regression_dual",
    "rung_4_linear_tcnn",
    "rung_5_tcnn_1ch",
    "rung_6_tcnn_3ch",
]


def perf_summary(daily_returns: pd.Series) -> dict:
    r = daily_returns.dropna().values
    if len(r) < 2:
        return {"sharpe": float("nan"), "n_days": len(r)}
    mu = r.mean(); sd = r.std(ddof=0)
    cum = (1 + r).cumprod()
    rmax = np.maximum.accumulate(cum)
    max_dd = float(((cum - rmax) / rmax).min())
    return {
        "ann_return": float((1 + r).prod() ** (252 / len(r)) - 1),
        "ann_vol":    float(sd * np.sqrt(252)),
        "sharpe":     float(mu / (sd + 1e-12) * np.sqrt(252)),
        "max_dd":     max_dd,
        "n_days":     int(len(r)),
    }


def main():
    rows = []
    for rung in ALL_RUNGS:
        path = config.OUTPUTS_DIR / rung / "all_results.csv"
        if not path.exists():
            rows.append({"rung": rung, "status": "NOT RUN"})
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        s = perf_summary(df["return"])
        rows.append({
            "rung": rung,
            "status": "OK",
            "n_days": s["n_days"],
            "ann_return": f"{s['ann_return']*100:+.2f}%",
            "ann_vol":    f"{s['ann_vol']*100:.2f}%",
            "sharpe":     f"{s['sharpe']:.3f}",
            "max_dd":     f"{s['max_dd']*100:+.2f}%",
        })

    summary = pd.DataFrame(rows)
    print("\n=== LADDER SUMMARY (gross of TC) ===")
    print(summary.to_string(index=False))

    # F3 decomposition: 1 -> 2 -> 3 -> 4 -> 5 (decile-sort track)
    print("\n=== F3 thesis decomposition (decile-sort track) ===")
    f3_rungs = ["rung_1_simple_momentum_decile",
                "rung_2_ewm_momentum_decile",
                "rung_3_ts_regression_decile",
                "rung_4_linear_tcnn",        # uses dual-softmax (only differentiable option)
                "rung_5_tcnn_1ch"]
    prev = None
    for r in f3_rungs:
        path = config.OUTPUTS_DIR / r / "all_results.csv"
        if not path.exists():
            print(f"  {r}: NOT RUN")
            prev = None
            continue
        df = pd.read_csv(path, parse_dates=["date"])
        s = perf_summary(df["return"])
        delta = ""
        if prev is not None:
            delta = f"  (Δ vs prev = {s['sharpe'] - prev:+.3f})"
        print(f"  {r:<35} Sharpe = {s['sharpe']:.3f}{delta}")
        prev = s["sharpe"]

    # Portfolio-step effect: a vs d at fixed factor
    print("\n=== Portfolio-step effect (decile vs dual-softmax at fixed factor) ===")
    pairs = [
        ("rung_1_simple_momentum_decile", "rung_1d_simple_momentum_dual",  "12-1 momentum"),
        ("rung_2_ewm_momentum_decile",    "rung_2d_ewm_momentum_dual",     "EWM momentum"),
        ("rung_3_ts_regression_decile",   "rung_3d_ts_regression_dual",    "TS regression"),
    ]
    for d, dual, label in pairs:
        d_path = config.OUTPUTS_DIR / d / "all_results.csv"
        dual_path = config.OUTPUTS_DIR / dual / "all_results.csv"
        if d_path.exists() and dual_path.exists():
            d_df = pd.read_csv(d_path, parse_dates=["date"])
            dual_df = pd.read_csv(dual_path, parse_dates=["date"])
            d_s = perf_summary(d_df["return"])["sharpe"]
            dual_s = perf_summary(dual_df["return"])["sharpe"]
            print(f"  {label:<20} decile={d_s:+.3f}  dual={dual_s:+.3f}  (Δ={dual_s-d_s:+.3f})")
        else:
            print(f"  {label}: not yet complete")


if __name__ == "__main__":
    main()
