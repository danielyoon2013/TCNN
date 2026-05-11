"""YAML-config-driven experiment orchestrator.

Each experiment = one YAML config. The runner:
  1. Reads the config
  2. Loads the canonical panel_daily.parquet
  3. For each (year, seed) cell:
       - if `outputs/{exp_id}/year_Y/seed_S/returns.csv` exists, skip
       - else compute scores, apply portfolio mapping, write returns.csv
  4. At the end, concat all per-cell CSVs into outputs/{exp_id}/all_results.csv
  5. For TCNN experiments, also evaluate trained scores under decile-sort + MVO
     (multi-portfolio eval matrix per the locked plan)

Resumable: a partial run can be re-launched and only does missing cells.
"""

from __future__ import annotations
import os
import json
import time
import yaml
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from . import config
from . import factors
from . import portfolio


# =============================================================================
# Config loading + validation
# =============================================================================

def load_config(yaml_path: str | Path) -> dict:
    """Load and minimally validate an experiment YAML."""
    yaml_path = Path(yaml_path)
    with open(yaml_path) as f:
        cfg = yaml.safe_load(f)

    required = ["experiment_id", "factor", "portfolio", "universe", "rolling", "output_dir"]
    missing = [k for k in required if k not in cfg]
    if missing:
        raise ValueError(f"{yaml_path}: missing required keys {missing}")
    cfg["_source_path"] = str(yaml_path)
    return cfg


def load_sweep(sweep_yaml_path: str | Path) -> list[dict]:
    """A sweep manifest is a YAML with one key, `sweep`, listing experiment YAMLs."""
    sweep_yaml_path = Path(sweep_yaml_path)
    with open(sweep_yaml_path) as f:
        manifest = yaml.safe_load(f)

    base_dir = sweep_yaml_path.parent
    return [load_config(base_dir / exp_yaml) for exp_yaml in manifest["sweep"]]


# =============================================================================
# Resume: which (year, seed) cells are already done?
# =============================================================================

def list_completed_cells(output_dir: str | Path) -> list[tuple[int, int]]:
    """Inspect output_dir and return list of (year, seed) cells that have returns.csv."""
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return []
    completed = []
    for year_dir in output_dir.glob("year_*"):
        year = int(year_dir.name.split("_")[1])
        for seed_dir in year_dir.glob("seed_*"):
            seed = int(seed_dir.name.split("_")[1])
            if (seed_dir / "returns.csv").exists():
                completed.append((year, seed))
    return sorted(completed)


def expected_cells(cfg: dict) -> list[tuple[int, int]]:
    """All (year, seed) cells that should exist after a complete run."""
    rolling = cfg["rolling"]
    years = list(range(rolling["start_year"], rolling["end_year"] + 1))
    seeds = cfg.get("training", {}).get("seeds", [0])
    return [(y, s) for y in years for s in seeds]


def cells_remaining(cfg: dict) -> list[tuple[int, int]]:
    done = set(list_completed_cells(cfg["output_dir"]))
    return [c for c in expected_cells(cfg) if c not in done]


def status_table(configs: list[dict]) -> pd.DataFrame:
    rows = []
    for cfg in configs:
        all_cells = expected_cells(cfg)
        done = list_completed_cells(cfg["output_dir"])
        rows.append({
            "experiment_id": cfg["experiment_id"],
            "factor": cfg["factor"]["type"],
            "portfolio": cfg["portfolio"]["type"],
            "completed": len(done),
            "total": len(all_cells),
            "remaining": len(all_cells) - len(done),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Universe filter
# =============================================================================

def apply_universe_filter(panel: pd.DataFrame, universe_cfg: dict) -> pd.DataFrame:
    """Apply boolean column filters from the config's `universe.filters` list.

    NaN values in any filter column are treated as False (not in universe).
    This matters because some stocks may have e.g. valid `in_top_2000` but
    NaN `adtv_above_5m` if they have insufficient history for the rolling ADTV.
    """
    mask = pd.Series(True, index=panel.index)
    for col in universe_cfg["filters"]:
        if col not in panel.columns:
            raise ValueError(f"Universe filter column '{col}' not in panel.")
        mask = mask & panel[col].fillna(False).astype(bool)
    return panel[mask].copy()


# =============================================================================
# Per-cell execution dispatch
# =============================================================================

def run_cell(cfg: dict, panel: pd.DataFrame, year: int, seed: int) -> pd.DataFrame:
    """Run one (year, seed) cell. Returns daily P&L DataFrame for the test year.

    For non-trainable factors (rungs 1, 2, 3, 1d, 2d, 3d):
      - seed is ignored
      - score is computed from the panel directly
      - portfolio is constructed each rebal date
      - daily returns over the test year are computed

    For trainable factors (rungs 4, 5, 6 = linear/full TCNN):
      - delegates to src.train_tcnn (when implemented)
    """
    factor_type = cfg["factor"]["type"]
    portfolio_type = cfg["portfolio"]["type"]
    output_dir = Path(cfg["output_dir"]) / f"year_{year}" / f"seed_{seed}"
    output_dir.mkdir(parents=True, exist_ok=True)

    if factor_type in ("simple_mom", "ewm_mom", "ts_reg"):
        df = _run_cell_baseline(cfg, panel, year, factor_type, portfolio_type)
    elif factor_type in ("linear_tcnn", "tcnn"):
        from .train_tcnn import train_and_evaluate
        df = train_and_evaluate(cfg, panel, year, seed)
    else:
        raise ValueError(f"Unknown factor type: {factor_type}")

    df["experiment_id"] = cfg["experiment_id"]
    df["factor_type"]   = factor_type
    df["portfolio_type"] = portfolio_type
    df["year"]          = year
    df["seed"]          = seed
    df.to_csv(output_dir / "returns.csv", index=False)
    return df


def _run_cell_baseline(cfg: dict, panel: pd.DataFrame, year: int,
                        factor_type: str, portfolio_type: str) -> pd.DataFrame:
    """Hand-crafted / OLS baselines (rungs 1, 1d, 2, 2d, 3, 3d) — VECTORIZED.

    Optimizations vs. the original Python-loop version:
      1. Subset panel to test_year +- buffer (or +train_window for ts_reg) BEFORE
         applying the universe filter — works on ~3M rows instead of 23M.
      2. Pivot returns into a wide (D, N) matrix once per cell.
      3. Per-rebal: vectorized `holding_rets @ weights` instead of Python loop
         over (day, permno) pairs.

    Net speedup: ~10-30x. Per-cell time goes from ~80s to ~3-5s.
    Numerical equivalence preserved: same scores, same weights, same forward returns.
    """
    rolling = cfg["rolling"]

    # ----- Step 1: subset panel to the relevant date range (much smaller working set)
    test_year_start = pd.Timestamp(f"{year}-01-01")
    test_year_end   = pd.Timestamp(f"{year}-12-31")

    if factor_type == "ts_reg":
        train_start_year = year - rolling["train_years"] - rolling["val_years"]
        date_min = pd.Timestamp(f"{train_start_year}-01-01")
    else:
        date_min = test_year_start - pd.Timedelta(days=10)
    # Buffer past test-year-end for the final holding period spilling into next month
    date_max = test_year_end + pd.Timedelta(days=45)

    panel_subset = panel[(panel["date"] >= date_min) & (panel["date"] <= date_max)]
    universe_filtered = apply_universe_filter(panel_subset, cfg["universe"])

    # ----- Step 2: rebal dates (last trading day of each month within test year)
    test_year_panel = universe_filtered[universe_filtered["date"].dt.year == year]
    if len(test_year_panel) == 0:
        return pd.DataFrame(columns=["date", "return"])

    test_year_panel = test_year_panel.copy()
    test_year_panel["year_month"] = test_year_panel["date"].dt.to_period("M")
    rebal_dates = test_year_panel.groupby("year_month")["date"].max().sort_index().tolist()

    # We also need ONE more trading day past the last rebal date in test year, so
    # the final holding period has somewhere to land. Use universe_filtered which
    # already covers a buffer past test_year_end.

    # ----- Step 3: pivot returns to wide (D x N) matrix ONCE.
    # IMPORTANT: build from `panel_subset`, NOT `universe_filtered`. The universe
    # filter selects stocks AT THE REBAL DATE; once we own them, we get their actual
    # daily returns until next rebal even if they drop out of the universe mid-month
    # (e.g., price drops below $5, ADTV declines). Building from universe_filtered
    # would silently treat dropped-out positions as cash, biasing the strategy.
    ret_wide = (panel_subset
                .pivot_table(index="date", columns="permno", values="ret", aggfunc="first")
                .sort_index())
    all_dates_array = ret_wide.index.values  # numpy datetime64[ns]

    # ----- Step 4: TS regression beta fit (only for rung 3, 3d)
    # Lag features computed on-the-fly from ret_wide, ONLY at training-period
    # month-end dates. Avoids materializing a 12+ GB lag matrix for daily rows.
    beta = None
    if factor_type == "ts_reg":
        train_start_year = year - rolling["train_years"] - rolling["val_years"]
        train_end_year   = year - 1

        train_panel = universe_filtered[(universe_filtered["date"].dt.year >= train_start_year) &
                                          (universe_filtered["date"].dt.year <= train_end_year)].copy()
        if len(train_panel) == 0:
            print(f"  [ts_reg] no training data for years {train_start_year}-{train_end_year}; skipping cell")
            return pd.DataFrame(columns=["date", "return"])
        train_panel["year_month"] = train_panel["date"].dt.to_period("M")
        train_rebal_dates = train_panel.groupby("year_month")["date"].max().sort_index().tolist()

        n_lags = cfg["factor"].get("n_lags", config.LOOKBACK_DAYS)
        permno_order = ret_wide.columns.values

        X_train_blocks, y_train_blocks = [], []
        for me in train_rebal_dates:
            me_idx = int(np.searchsorted(all_dates_array, np.datetime64(me, "ns")))
            if me_idx < n_lags:
                continue
            # Lag window: ret_wide rows [me_idx - n_lags, me_idx) — chronological
            lag_window = ret_wide.iloc[me_idx - n_lags : me_idx].to_numpy(
                dtype=np.float64, na_value=np.nan)
            # X[i, l] = ret_{me - (l+1)} for stock i. Reverse rows so lag_1 (most recent) is column 0.
            X_at_me = lag_window[::-1].T   # (n_permnos, n_lags)

            # y: ret_fut_1m at me for each permno (universe filter applied for fitting)
            y_panel = universe_filtered[universe_filtered["date"] == me].set_index("permno")["ret_fut_1m"]
            y_at_me = y_panel.reindex(permno_order).to_numpy(dtype=np.float64, na_value=np.nan)

            valid = ~np.isnan(X_at_me).any(axis=1) & ~np.isnan(y_at_me)
            if valid.sum() == 0:
                continue
            X_train_blocks.append(X_at_me[valid])
            y_train_blocks.append(y_at_me[valid])

        if not X_train_blocks:
            print(f"  [ts_reg] no valid training observations; skipping cell")
            return pd.DataFrame(columns=["date", "return"])

        X_train = np.vstack(X_train_blocks)
        y_train = np.concatenate(y_train_blocks)

        from sklearn.linear_model import Ridge
        ridge_alpha = cfg["factor"].get("ridge_alpha", 1.0)
        model = Ridge(alpha=ridge_alpha, fit_intercept=True).fit(X_train, y_train)
        beta = np.concatenate([[model.intercept_], model.coef_])
        # Free memory
        del X_train_blocks, y_train_blocks, X_train, y_train

    portfolio_fn = portfolio.get_portfolio_fn(portfolio_type)

    # MVO needs return history for covariance — use the wide matrix
    needs_history = (portfolio_type == "mvo_lw")

    # ----- Step 5: per-rebal-date scoring + portfolio + vectorized daily P&L
    eo = config.ENTRY_OFFSET_DAYS  # T+1 entry: first held return is at rebal_idx + eo + 1

    daily_records = []
    for rebal_date in rebal_dates[:-1]:
        # Find next rebal date (used as holding-period end)
        next_rebal = next((d for d in rebal_dates if d > rebal_date), None)
        if next_rebal is None:
            continue

        # Score at this rebal date's universe
        slice_at_rebal = universe_filtered[universe_filtered["date"] == rebal_date]
        if len(slice_at_rebal) < 50:
            continue

        if factor_type == "simple_mom":
            scores = factors.score_simple_momentum(slice_at_rebal)
        elif factor_type == "ewm_mom":
            scores = factors.score_ewm_momentum(slice_at_rebal)
        elif factor_type == "ts_reg":
            # TS regression: compute lag matrix at this rebal date from ret_wide,
            # then apply beta directly. Avoids needing _lag_* columns on the panel.
            n_lags = cfg["factor"].get("n_lags", config.LOOKBACK_DAYS)
            rebal_idx_for_score = int(np.searchsorted(all_dates_array, np.datetime64(rebal_date, "ns")))
            if rebal_idx_for_score < n_lags:
                continue
            lag_window = ret_wide.iloc[rebal_idx_for_score - n_lags : rebal_idx_for_score].to_numpy(
                dtype=np.float64, na_value=np.nan)
            X_at_rebal_all = lag_window[::-1].T   # (n_all_permnos, n_lags)

            # Map slice_at_rebal permnos to ret_wide column indices
            permno_to_col = {p: i for i, p in enumerate(ret_wide.columns.values)}
            score_permnos = slice_at_rebal["permno"].values
            try:
                col_idxs = np.array([permno_to_col[p] for p in score_permnos])
            except KeyError:
                # Some permno not in ret_wide — should not happen since both come from panel_subset
                continue
            X_at_rebal = X_at_rebal_all[col_idxs]   # (n_universe_permnos, n_lags)

            intercept, coefs = beta[0], beta[1:]
            scores = intercept + X_at_rebal @ coefs   # (n_universe_permnos,)
            # Stocks with NaN lags (insufficient history) propagate NaN scores; dropped below.
        else:
            raise ValueError(f"Unknown factor type: {factor_type}")

        # Drop NaN scores
        valid_mask = ~np.isnan(scores)
        if valid_mask.sum() < 50:
            continue
        scores = scores[valid_mask]
        permnos = slice_at_rebal["permno"].values[valid_mask]

        # Build portfolio weights
        if needs_history:
            # MVO: covariance estimated from past 252 days of returns for these permnos.
            # Use `to_numpy(dtype=float)` to safely handle pandas nullable Float64.
            history = (ret_wide.loc[ret_wide.index < rebal_date, permnos]
                                .tail(252)
                                .to_numpy(dtype=np.float64, na_value=np.nan))
            weights = portfolio_fn(scores, history, gross_leverage=1.0)
        else:
            weights = portfolio_fn(scores, gross_leverage=1.0)

        # ----- Vectorized holding-period daily P&L (the key optimization)
        # BUG-7: T+1 entry → first held return is at index rebal_idx + eo + 1
        rebal_idx = int(np.searchsorted(all_dates_array, np.datetime64(rebal_date, "ns")))
        next_idx  = int(np.searchsorted(all_dates_array, np.datetime64(next_rebal,  "ns")))
        holding_start = rebal_idx + eo + 1   # inclusive; days strictly after entry day
        holding_end   = next_idx + 1         # exclusive; up to and including next_rebal
        if holding_start >= holding_end or holding_end > len(all_dates_array):
            continue

        holding_dates_ns = all_dates_array[holding_start:holding_end]      # (H,)
        # Slice the wide matrix: (H rows, N cols) — pandas reindexes columns to match `permnos`.
        # `.to_numpy(dtype=float, na_value=np.nan)` handles pandas nullable Float64 dtype safely.
        holding_rets = ret_wide.loc[pd.DatetimeIndex(holding_dates_ns), permnos].to_numpy(
            dtype=np.float64, na_value=np.nan)
        # BUG-8: NaN returns (delisted mid-period) treated as 0 (cash position)
        holding_rets = np.where(np.isnan(holding_rets), 0.0, holding_rets)

        # Vectorized daily portfolio returns: (H,) = (H, N) @ (N,) (also ensure weights are float)
        weights_arr = np.asarray(weights, dtype=np.float64)
        daily_port = holding_rets @ weights_arr

        for d, r in zip(holding_dates_ns, daily_port):
            daily_records.append({
                "date": pd.Timestamp(d),
                "return": float(r),
                "rebal_date": rebal_date,
            })

    return pd.DataFrame(daily_records)


# =============================================================================
# Per-experiment runner (loops over cells)
# =============================================================================

def run_experiment(cfg: dict, panel: pd.DataFrame, force: bool = False) -> dict:
    """Execute every (year, seed) cell of an experiment, writing per-cell CSVs."""
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    cells = expected_cells(cfg) if force else cells_remaining(cfg)
    print(f"\n=== {cfg['experiment_id']} ===")
    print(f"  cells to run: {len(cells)} / {len(expected_cells(cfg))}")

    if not cells:
        print("  all cells already complete")
        return {"completed": len(expected_cells(cfg)), "ran_now": 0}

    t0 = time.time()
    for i, (year, seed) in enumerate(cells):
        print(f"  [{i+1}/{len(cells)}] year={year} seed={seed}...")
        run_cell(cfg, panel, year, seed)
    elapsed = time.time() - t0
    print(f"  ran {len(cells)} cells in {elapsed:.0f}s ({elapsed/len(cells):.1f}s/cell)")

    concat_to_master(cfg)
    return {"completed": len(expected_cells(cfg)), "ran_now": len(cells), "elapsed": elapsed}


def concat_to_master(cfg: dict) -> None:
    """Aggregate per-cell returns.csv files into outputs/{exp_id}/all_results.csv."""
    output_dir = Path(cfg["output_dir"])
    dfs = []
    for year_dir in sorted(output_dir.glob("year_*")):
        for seed_dir in sorted(year_dir.glob("seed_*")):
            f = seed_dir / "returns.csv"
            if f.exists():
                dfs.append(pd.read_csv(f, parse_dates=["date"]))
    if dfs:
        master = pd.concat(dfs, ignore_index=True).sort_values(["year", "seed", "date"])
        master.to_csv(output_dir / "all_results.csv", index=False)
        print(f"  → {output_dir / 'all_results.csv'} ({len(master):,} rows)")


# =============================================================================
# Sweep runner (multi-experiment)
# =============================================================================

def run_sweep(sweep_yaml_path: str | Path, panel: pd.DataFrame, force: bool = False) -> pd.DataFrame:
    """Run every experiment in a sweep manifest sequentially. Returns status table."""
    configs = load_sweep(sweep_yaml_path)
    print(f"Sweep: {sweep_yaml_path}  ({len(configs)} experiments)")
    print(status_table(configs).to_string(index=False))

    for cfg in configs:
        run_experiment(cfg, panel, force=force)

    return status_table(configs)
