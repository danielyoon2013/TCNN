# TCNN Portfolio (ICLR 2026 Workshop on Advances in Financial AI)

Replication + extension of "Towards Representation Learning for Cross-Sectional
Portfolio Construction" — interview-prep package for quant PM interviews at
multistrats.

## Quick start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. WRDS auth-priming (one-time, in your terminal)
python -c "import wrds; wrds.Connection(wrds_username='YOUR_USERNAME').close()"
# answer 'y' to save credentials to ~/.pgpass

# 3. Build the canonical panel (once per data refresh)
jupyter notebook notebooks/01_panel_construction.ipynb
# → writes data/03_features/panel_daily.parquet

# 4. Run the CPU baselines (rungs 1, 1d, 2, 2d, 3, 3d)
jupyter notebook notebooks/02_factor_baselines.ipynb
# OR via CLI:
python -m train.train_tcnn --sweep experiments/_track_a.yaml

# 5. (On RunPod) Train the TCNN rungs
python -m train.train_tcnn --config experiments/rung_4_linear_tcnn.yaml
python -m train.train_tcnn --config experiments/rung_5_tcnn_1ch.yaml
python -m train.train_tcnn --config experiments/rung_6_tcnn_3ch.yaml

# 6. Analysis (CPU)
jupyter notebook notebooks/05_ladder_summary.ipynb
jupyter notebook notebooks/06_diagnostics.ipynb
jupyter notebook notebooks/07_capacity_tc_sweep.ipynb
```

## Layout

```
04_Projects/TCNN/
├── CLAUDE.md, PHASE_C_TEST_PLAN.md, README.md, paper/, requirements.txt
├── src/                # library code (no CLI here)
├── notebooks/          # all visualization + analysis (00-08)
├── train/              # GPU training entry point (`python -m train.train_tcnn ...`)
├── experiments/        # 9 YAML configs + _track_a.yaml sweep manifest
├── data/               # 01_raw/, 02_clean/, 03_features/, 05_panels_tcnn/
├── outputs/            # one dir per experiment_id, _master_results.csv
├── scripts/            # setup_runpod.sh, sync_results.sh, make_notebooks.py
├── code/               # _legacy_* reference files from authoring repo (read-only)
└── prep/               # interview deliverables (methodology, pitches, Q&A)
```

## The 9-rung comparison ladder

The headline test of the F3 thesis ("learnable aggregation beats hand-crafted").
All 9 rungs use the same input (past 252 daily returns) and same universe
(top-2000 by mktcap, tradability filters). What changes is the function class
and the portfolio mapping.

| Rung | Factor | Portfolio | Function class |
|---|---|---|---|
| 1   | 12-1 momentum            | decile-sort EW | hand-crafted, uniform-on-window |
| 1d  | 12-1 momentum            | dual-softmax   | hand-crafted, uniform-on-window |
| 2   | EWM momentum (H=60d)     | decile-sort EW | hand-crafted, exponential |
| 2d  | EWM momentum (H=60d)     | dual-softmax   | hand-crafted, exponential |
| 3   | TS regression on lags    | decile-sort EW | learned linear, MSE loss |
| 3d  | TS regression on lags    | dual-softmax   | learned linear, MSE loss |
| 4   | Linear TCNN              | dual-softmax   | learned linear, **Sharpe loss, end-to-end** |
| 5   | TCNN, 1-channel          | dual-softmax   | learned nonlinear, end-to-end |
| 6   | TCNN, 3-channel          | dual-softmax   | learned nonlinear, end-to-end (paper config) |

Critical decompositions:
- **3 → 4**: predict-then-optimize (MSE) → end-to-end (Sharpe). The F2 test.
- **4 → 5**: linear → nonlinear. The TCNN-architecture-specific test.
- **1 → 1d, 2 → 2d, 3 → 3d**: portfolio-step effect at fixed factor.
- **5 → 6**: 1-channel → 3-channel. Side ablation (engineered features).

## Bug fixes vs. the original paper code (8 total)

See [`CLAUDE.md`](CLAUDE.md) for the full audit. Key fixes applied:

1. **EWMA leak**: `ret_norm` now uses lagged EWMA vol (was contemporaneous)
2. **Delisting returns**: `crsp.dsedelist` joined into the last DSF row (Beaver-McNichols-Price)
3. **Multiprocessing**: `Pool.imap_unordered` restored in panel-build (~10× speedup)
4. **Temporal-rank loops**: vectorized via `groupby+rolling.rank(pct=True)`
5. **5-year reference dates**: documented (didn't fix; minor)
6. **AMEX exclusion**: documented (didn't fix; matches paper)
7. **T+1 entry-offset look-ahead**: signal at T → entry at close[T+1] → first held return ret[T+2]
8. **Mid-period delisting exclusion**: stocks delisting mid-month now kept; position goes to cash after delist day

## Convention notes

- Sharpe is reported **gross AND net** of frictions; capacity is stated explicitly. Never an unqualified Sharpe.
- **Cumulative P&L plots use `cumsum`, not `cumprod`** (additive log-equivalent comparison; doesn't compound winners visually).
- TC sweep is a **fixed-cost grid** (0/5/10/20 bps round-trip), not a per-stock spread model.
- **Single source of truth**: every rung reads `data/03_features/panel_daily.parquet`; same universe filter, same forward returns, same masks.
- T+1 entry: signal at end-of-day T → execute at close[T+1] → first holding return is ret[T+2].
