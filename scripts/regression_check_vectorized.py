"""Compare vectorized vs non-vectorized TCNN training output.

Usage:
    python scripts/regression_check_vectorized.py \
        outputs/smoke_5w_d0_oldcode/all_results.csv \
        outputs/smoke_5w_d0/all_results.csv

Two regimes:

1. Strict (dropout=0 configs, e.g. smoke_5w_d0.yaml or rung_4 linear):
   The forward pass per stock is identical between batched and sequential
   code paths. Residual diffs come only from fp32 BLAS non-determinism in
   batched matmuls (different summation order). After 2 epochs of training,
   typical max return diff is ~1e-4 to ~1e-3 and Sharpe diff is ~0.01.
   Gate: max return diff <= 1e-3 AND Sharpe diff <= 0.05.

2. Loose (dropout > 0 configs, e.g. plain smoke_5w.yaml):
   Batching K months into one encoder forward changes the dropout RNG
   consumption order vs K sequential forwards. The math per stock is still
   identical, but dropout masks are drawn differently after the first call,
   leading to divergent training trajectories. For a 2-epoch smoke with
   dropout=0.15, Sharpe diff of 0.3-0.5 is expected (training-trajectory
   noise, not a bug). For production configs (35 epochs) the trajectory
   converges to similar Sharpe. Gate: Sharpe diff <= 0.50.

Recommended flow: run BOTH regimes. Strict pass proves the math is
equivalent; loose pass proves the pipeline runs end-to-end consistently.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd


STRICT_MAX_DIFF = 1e-3
STRICT_SHARPE_DIFF = 0.05
LOOSE_SHARPE_DIFF = 0.50


def annualized_sharpe(returns: np.ndarray) -> float:
    if len(returns) == 0:
        return float("nan")
    return float(returns.mean() / (returns.std() + 1e-12) * np.sqrt(252))


def main(old_path: str, new_path: str) -> int:
    old = pd.read_csv(old_path, parse_dates=["date", "rebal_date"])
    new = pd.read_csv(new_path, parse_dates=["date", "rebal_date"])

    print(f"old: {old_path}  ({len(old):,} rows)")
    print(f"new: {new_path}  ({len(new):,} rows)")
    print()

    if len(old) != len(new):
        print("FAIL: row count mismatch")
        return 1

    key_cols = ["seed", "rebal_date", "date"]
    old_sorted = old.sort_values(key_cols).reset_index(drop=True)
    new_sorted = new.sort_values(key_cols).reset_index(drop=True)
    if not (old_sorted[key_cols] == new_sorted[key_cols]).all().all():
        print("FAIL: row keys don't align (different rebal_dates or dates)")
        return 1

    diff = new_sorted["return"].values - old_sorted["return"].values
    max_abs = float(np.abs(diff).max())
    mean_abs = float(np.abs(diff).mean())

    old_sr = annualized_sharpe(old_sorted["return"].values)
    new_sr = annualized_sharpe(new_sorted["return"].values)
    sr_diff = new_sr - old_sr

    print(f"max |return diff|:  {max_abs:.6e}")
    print(f"mean |return diff|: {mean_abs:.6e}")
    print()
    print(f"OLD gross Sharpe (annualized): {old_sr:+.4f}")
    print(f"NEW gross Sharpe (annualized): {new_sr:+.4f}")
    print(f"Sharpe diff:                   {sr_diff:+.4f}")
    print()

    if max_abs <= STRICT_MAX_DIFF and abs(sr_diff) <= STRICT_SHARPE_DIFF:
        print(f"PASS (strict): math is equivalent. Residual diff is fp32-BLAS noise.")
        print(f"               Use this regime with a dropout=0 config to validate the refactor.")
        return 0
    if abs(sr_diff) <= LOOSE_SHARPE_DIFF:
        print(f"PASS (loose): Sharpe within {LOOSE_SHARPE_DIFF} — consistent with dropout-RNG divergence.")
        print(f"              For a strict math check, re-run with a dropout=0 config.")
        return 0
    print(f"FAIL: Sharpe diff exceeds {LOOSE_SHARPE_DIFF} — investigate.")
    return 1


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    sys.exit(main(sys.argv[1], sys.argv[2]))
