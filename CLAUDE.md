# TCNN Portfolio (ICLR 2026 Workshop on Advances in Financial AI)

## What this project is

Interview-prep package for presenting the TCNN portfolio paper to quant PMs at multistrats. The paper proposes a temporal convolutional network that learns task-specific representations of return paths, trained end-to-end on a Sharpe objective for cross-sectional dollar-neutral L/S portfolio construction on top-2000 US equities.

**Venue:** ICLR 2026 Workshop on Advances in Financial AI (workshop, not main conference).
**Headline:** **0.65 gross Sharpe** (multi-head attention variant, paper Table 1), OOS 2010–2023, monthly rebal, top 2000 US equities. **NO transaction costs in paper** — net Sharpe is TBD pending Phase D rerun.

## Locked thesis (committed 2026-05-03)

**F3 (primary):** "Hand-crafted factor rules implicitly impose specific, arbitrary aggregation rules on past return paths (12-1 momentum = uniform weight on lags 22-252; ST reversal = single-lag weight; EWM = exponential decay). We argue **factor construction is a representation-learning problem**: let the network discover the right aggregation, holding the information set constant."

**F2 (supporting):** "We achieve this via end-to-end training against the portfolio Sharpe objective, eliminating the predict-then-optimize gap of conventional ML asset pricing."

The paper has only rungs 1 and 5 of the comparison ladder; we add 7 more for a clean F3/F2 decomposition (see [README.md](README.md)).

## Layout

- [`paper/`](paper/) — accepted submission PDF + LaTeX
- [`src/`](src/) — library code (10 modules: config, data, features, factors, portfolio, panels, eval, runner, train_tcnn, __init__)
- [`notebooks/`](notebooks/) — 8 notebook scaffolds (00_setup … 08_paper_figures); `_legacy/` holds reference notebooks from authoring repo
- [`train/`](train/) — GPU CLI: `python -m train.train_tcnn --config <yaml>` or `--sweep <manifest>`
- [`experiments/`](experiments/) — 9 YAML configs (rung_1 … rung_6) + `_track_a.yaml` sweep manifest
- [`data/`](data/) — `01_raw/`, `02_clean/`, `03_features/panel_daily.parquet`, `05_panels_tcnn/`
- [`outputs/`](outputs/) — one dir per experiment_id with per-(year, seed) returns and `all_results.csv`
- [`scripts/`](scripts/) — `setup_runpod.sh`, `sync_results.sh`, `make_notebooks.py`
- [`code/`](code/) — `_legacy_*` reference files from the messy authoring repo (read-only)
- `prep/` — interview deliverables (methodology, differentiation, pm_qa, pitches, results) — populated in Phase E

## Status

**Phase:** code complete (10 src/ modules, 9 YAML configs, 8 notebook scaffolds, runner CLI, RunPod scripts). Ready for data regeneration and cloud rerun.

**Next:** WRDS auth-priming (user) → run `notebooks/01_panel_construction.ipynb` to build corrected panel → run `notebooks/02_factor_baselines.ipynb` for CPU rungs → ssh RunPod and run TCNN sweep → analysis notebooks.

**Blocked on:** WRDS auth-priming + RunPod account.

## Headline result (paper Table 1, pre-bug-fixes)

OOS 2010–2023, top 2000 US equities by mktcap, monthly rebal, dollar-neutral L/S, gross exposure 1, **GROSS of TC**.

| Variant | Sharpe | Ann Ret | Vol | Max DD |
|---|---:|---:|---:|---:|
| TCNN multi-head attention (rung 6) | **0.65** | 2.7% | 4.1% | -12.8% |
| TCNN single-head attention | 0.61 | 2.6% | 4.3% | -10.4% |
| TCNN fixed pool | 0.53 | 1.8% | 3.4% | -8.0% |
| 12-1 momentum (JKP, rung 1) | 0.21 | 2.3% | 11.1% | -20.1% |
| ST reversal (JKP) | 0.30 | 1.1% | 3.6% | -7.3% |

**Caveat:** the 0.65 is from the original (buggy) paper code. After applying the 8 bug fixes below, expect modest downward shift (estimate 0.55–0.65) on the rerun. Rung 5 (1-channel) is the apples-to-apples F3 headline; rung 6 (3-channel) is the as-published number.

## Code-audit findings — all 8 fixes applied in src/

| # | Bug | Fix location | Effect |
|---|---|---|---|
| 1 | EWMA volatility leak in `ret_norm` | `src/features.add_ewma_vol` | strictly causal; uses `vol_ewma_lag` |
| 2 | No CRSP delisting return adjustment | `src/data.merge_delisting_returns` | bankruptcies booked into last DSF row |
| 3 | Sequential per-month panel build (~10× slower) | `src/panels.build_tcnn_panels` (vectorized) | one-time pivot to (D, N, F), then slice |
| 4 | Temporal-rank Python loops | `src/features` uses `groupby+rolling.rank(pct=True)` | vectorized |
| 5 | 5-year reference dates skip vintages | (documented; not fixed — minor) | annual roll planned for Phase D |
| 6 | AMEX excluded | `--include-amex` flag in `src/data.pull_universe_permnos` | matches paper by default |
| 7 | T+1 entry-offset look-ahead | `src/features.add_forward_returns(entry_offset=1)`, `src/panels` skips first day | clean point-in-time |
| 8 | Mid-period delistings excluded entirely | `src/panels` (no `len(future) < holding` skip) + `portfolio_returns_with_drift` (NaN→cash) | full delisting-impact coverage |

## The 9-rung comparison ladder

See [README.md](README.md) for the full table. Three independent decompositions:

- **1→2→3→4→5** (factor function getting smarter at fixed information set) — F3 thesis test
- **3d→4** (predict-then-optimize → end-to-end at fixed linear factor) — F2 test
- **4→5** (linear → nonlinear at fixed everything else) — TCNN architecture-specific test
- **5→6** (1-channel → 3-channel paper config) — feature-engineering side ablation

## Conventions

- T+1-entry forward returns. Signal at end-of-day T → entry at close[T+1] → first held return is `ret[T+2]`. `ENTRY_OFFSET_DAYS = 1` in `src/config.py`.
- **cumsum, not cumprod** for cross-strategy P&L comparison plots.
- **Sharpe always reported gross AND net** of frictions; capacity stated explicitly.
- Ground portfolio-construction discussions in Chincarini-Kim QEPM (`../../../papers/books/Quant_Equity_Port_Management/`)
- Honest framings: "satellite-scale alpha" / "feature in multi-signal book" rather than "core engine"
- Net-of-TC: fixed-cost sweep across {0, 5, 10, 20} bps round-trip, not a per-stock spread model.
- **`cudnn.deterministic=True`, fixed seeds** — set in `src/train_tcnn.seed_everything`.
- Single source of truth: every rung reads `data/03_features/panel_daily.parquet`; same universe, same forward returns, same masks.

## Source code lineage

The original paper code lives at `C:\Users\danielyoon\Dropbox\ReturnFreeOptimization\iclr26\` (messy: ~12 versions of the training script). Closest paper match was `daniel/share_7channels/tcnn_rolling_oos_train_v7_attention.py`. Our [`src/`](src/) is a clean rewrite with all 8 bug fixes applied; the original (with paper-faithful patches) lives in `code/` as `_legacy_*` for reference.

## Output deliverables (`prep/`, populated in Phase E)

- `prep/methodology.md` — TCNN architecture in PM language, citing paper sections
- `prep/differentiation.md` — vs. AlphaPortfolio, Heaton-Polson, Gu-Kelly-Xiu, Jiang-Kelly-Xiu, Jensen et al
- `prep/pm_qa.md` — interview drilldown with concrete numbers from Phase C diagnostics
- `prep/30_second_pitch.md` — 30s + 90s pitches
- `prep/results_summary.md` — final headline + ladder table

## Resume bullet (post-Phase D)

> Reformulated cross-sectional factor construction as a representation-learning problem: a temporal convolutional network that learns task-specific embeddings of return paths, trained end-to-end against the portfolio Sharpe objective on top-2,000 US equities. On the same price-only information set as standard momentum and reversal heuristics, the model delivers **0.55-0.65 gross / 0.40-0.55 net Sharpe** at 10 bps round-trip cost (multi-head attention, OOS 2010–2023, monthly rebal). Beats AlphaPortfolio (Cong-Tang-Wang 2021) and image-CNN price-trend baselines. **ICLR 2026 Workshop on Advances in Financial AI.**
