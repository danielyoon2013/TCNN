"""Generate the 8 notebook scaffolds in notebooks/.

Run once: `python scripts/make_notebooks.py`

Each notebook is an .ipynb JSON file with a focused set of cells. Notebooks
are deliberately thin — most logic lives in `src/` modules. The notebook is
for orchestration, inspection, and visualization.

Re-running this script overwrites existing scaffolds. If you've added cells
to a notebook, don't re-run.
"""

import json
from pathlib import Path

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"
NB_DIR.mkdir(exist_ok=True)


def make_nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }


def md(text):
    if isinstance(text, str):
        text = [text]
    return {"cell_type": "markdown", "metadata": {}, "source": [s if s.endswith("\n") else s + "\n" for s in text]}


def code(text):
    if isinstance(text, str):
        text = [text]
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": [s if s.endswith("\n") else s + "\n" for s in text]}


# =============================================================================
# 00 — setup
# =============================================================================

nb_00 = make_nb([
    md([
        "# 00 — Setup\n",
        "\n",
        "One-time environment + WRDS sanity check. Run before any other notebook.\n",
    ]),
    code([
        "%load_ext autoreload\n",
        "%autoreload 2\n",
        "\n",
        "import sys\n",
        "from pathlib import Path\n",
        "PROJECT_ROOT = Path('..').resolve()\n",
        "if str(PROJECT_ROOT) not in sys.path:\n",
        "    sys.path.insert(0, str(PROJECT_ROOT))\n",
        "\n",
        "from src import config\n",
        "print('PROJECT_ROOT:', PROJECT_ROOT)\n",
        "print('PANEL_DAILY_PARQUET:', config.PANEL_DAILY_PARQUET)\n",
    ]),
    md(["## Path existence check"]),
    code([
        "for p in [config.RAW_DIR, config.CLEAN_DIR, config.FEATURES_DIR, config.PANELS_DIR, config.OUTPUTS_DIR]:\n",
        "    print(f'{p}: {\"EXISTS\" if p.exists() else \"MISSING\"}')\n",
    ]),
    md([
        "## WRDS auth-priming (run ONCE in your terminal, not here)\n",
        "\n",
        "```powershell\n",
        "python -c \"import wrds; conn = wrds.Connection(wrds_username='YOUR_USERNAME'); conn.close()\"\n",
        "```\n",
        "\n",
        "Type your password when prompted. Answer `y` to save credentials to `~/.pgpass`.\n",
        "After that, `data_download.py` runs without prompts.\n",
    ]),
    code([
        "# Quick check: can we connect to WRDS without re-typing the password?\n",
        "# Comment this out after first verification.\n",
        "# import wrds\n",
        "# conn = wrds.Connection(wrds_username='YOUR_USERNAME')\n",
        "# print('Connected as:', conn.username)\n",
        "# conn.close()\n",
    ]),
])


# =============================================================================
# 01 — panel construction
# =============================================================================

nb_01 = make_nb([
    md([
        "# 01 — Panel construction\n",
        "\n",
        "Pulls raw CRSP data, cleans it (delisting-return merge), engineers features,\n",
        "adds universe-membership flags, computes T+1 forward returns, and writes\n",
        "the canonical `data/03_features/panel_daily.parquet`.\n",
        "\n",
        "Run-once orchestrator. Subsequent notebooks read the panel; they never re-pull WRDS.\n",
    ]),
    code([
        "%load_ext autoreload\n",
        "%autoreload 2\n",
        "\n",
        "import sys; from pathlib import Path\n",
        "sys.path.insert(0, str(Path('..').resolve()))\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import wrds\n",
        "\n",
        "from src import config, data, features\n",
    ]),
    md(["## 1. Pull universe permnos (top-2000 at 8 reference dates)"]),
    code([
        "WRDS_USER = 'YOUR_USERNAME'  # ← edit this\n",
        "conn = wrds.Connection(wrds_username=WRDS_USER)\n",
        "\n",
        "REFERENCE_DATES = ['1990-01-02', '1995-01-03', '2000-01-03', '2005-01-03',\n",
        "                   '2010-01-04', '2015-01-05', '2020-01-02', '2023-01-03']\n",
        "permnos = data.pull_universe_permnos(conn, REFERENCE_DATES, top_n=2000)\n",
        "print(f'Total unique permnos: {len(permnos)}')\n",
    ]),
    md(["## 2. Pull raw DSF + DSEDELIST"]),
    code([
        "dsf = data.load_or_pull_dsf(conn, permnos, '1989-01-01', '2023-12-31')\n",
        "dsedelist = data.load_or_pull_dsedelist(conn, permnos, '1989-01-01', '2023-12-31')\n",
        "print(f'DSF: {len(dsf):,} rows, DSEDELIST: {len(dsedelist):,} rows')\n",
        "conn.close()\n",
    ]),
    md(["## 3. Merge delisting returns into DSF (BUG-2 fix)"]),
    code([
        "daily_clean = data.merge_delisting_returns(dsf, dsedelist)\n",
        "daily_clean.to_parquet(config.CLEAN_DIR / 'daily_clean.parquet', index=False)\n",
        "print(f'daily_clean: {len(daily_clean):,} rows saved to {config.CLEAN_DIR}')\n",
    ]),
    md(["## 4. Feature engineering — the canonical panel"]),
    code([
        "panel = features.build_panel(daily_clean)\n",
        "panel.to_parquet(config.PANEL_DAILY_PARQUET, index=False)\n",
        "print(f'panel_daily.parquet: {len(panel):,} rows × {len(panel.columns)} cols')\n",
        "print('Columns:', list(panel.columns)[:30], '...')\n",
    ]),
    md(["## 5. Quick sanity checks"]),
    code([
        "# Universe size over time\n",
        "import matplotlib.pyplot as plt\n",
        "panel.groupby('date')['in_top_2000'].sum().plot(\n",
        "    figsize=(12, 3), title='Stocks in top-2000 over time'\n",
        ")\n",
        "plt.tight_layout()\n",
    ]),
    code([
        "# Feature completeness in the OOS period (2010-2023)\n",
        "oos = panel[panel['date'].dt.year.between(2010, 2023)]\n",
        "for col in ['ret', 'vol_ewma_lag', 'mom_12_1', 'ewm_h60_skip', 'ret_fut_1m', 'in_top_2000']:\n",
        "    valid_pct = oos[col].notna().mean() * 100\n",
        "    print(f'  {col:<20} {valid_pct:5.1f}% non-null')\n",
    ]),
])


# =============================================================================
# 02 — factor baselines (rungs 1-3 + 1d-3d)
# =============================================================================

nb_02 = make_nb([
    md([
        "# 02 — Factor baselines (rungs 1-3 + 1d-3d)\n",
        "\n",
        "Runs the 6 hand-crafted / OLS rungs of the comparison ladder, all CPU.\n",
        "Each rung writes `outputs/rung_X/all_results.csv` for downstream comparison.\n",
    ]),
    code([
        "%load_ext autoreload\n",
        "%autoreload 2\n",
        "\n",
        "import sys; from pathlib import Path\n",
        "sys.path.insert(0, str(Path('..').resolve()))\n",
        "\n",
        "import pandas as pd\n",
        "import numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "from src import config, runner\n",
    ]),
    md(["## 1. Load the canonical panel"]),
    code([
        "panel = pd.read_parquet(config.PANEL_DAILY_PARQUET)\n",
        "panel['date'] = pd.to_datetime(panel['date'])\n",
        "print(f'panel: {len(panel):,} rows, {panel[\"date\"].min()} → {panel[\"date\"].max()}')\n",
    ]),
    md(["## 2. Run rungs 1, 1d, 2, 2d, 3, 3d via runner"]),
    code([
        "BASELINE_CONFIGS = [\n",
        "    'rung_1_simple_momentum_decile.yaml',\n",
        "    'rung_1d_simple_momentum_dual.yaml',\n",
        "    'rung_2_ewm_momentum_decile.yaml',\n",
        "    'rung_2d_ewm_momentum_dual.yaml',\n",
        "    'rung_3_ts_regression_decile.yaml',\n",
        "    'rung_3d_ts_regression_dual.yaml',\n",
        "]\n",
        "configs = [runner.load_config(config.EXPERIMENTS_DIR / c) for c in BASELINE_CONFIGS]\n",
        "runner.status_table(configs)\n",
    ]),
    code([
        "# Run any cells not already complete (resumable)\n",
        "for cfg in configs:\n",
        "    runner.run_experiment(cfg, panel)\n",
    ]),
    md(["## 3. Load results + compare"]),
    code([
        "from src import eval as evalmod\n",
        "rung_ids = [c['experiment_id'] for c in configs]\n",
        "master = evalmod.load_master_results(rung_ids)\n",
        "master.head()\n",
    ]),
    code([
        "# Sharpe + max-DD per rung (averaged across seeds for trainable rungs; here all are non-trainable)\n",
        "summary = (master.groupby('experiment_id')['return']\n",
        "                  .apply(lambda s: pd.Series(evalmod.perf_summary(s)))\n",
        "                  .unstack())\n",
        "summary[['ann_return', 'ann_vol', 'sharpe', 'max_dd']].round(3)\n",
    ]),
    md(["## 4. Side-by-side cumulative P&L (cumsum, not cumprod — convention)"]),
    code([
        "fig, ax = plt.subplots(figsize=(12, 5))\n",
        "for exp_id in rung_ids:\n",
        "    df = master[master['experiment_id'] == exp_id].sort_values('date')\n",
        "    ax.plot(df['date'], df['return'].cumsum(), label=exp_id, alpha=0.85)\n",
        "ax.set(title='Cumulative P&L (cumsum) — baseline rungs', ylabel='cum return')\n",
        "ax.legend(loc='upper left', fontsize=8)\n",
        "ax.grid(alpha=0.3)\n",
        "plt.tight_layout()\n",
    ]),
])


# =============================================================================
# 03 — TCNN orchestrator
# =============================================================================

nb_03 = make_nb([
    md([
        "# 03 — TCNN training orchestrator\n",
        "\n",
        "Status check + GPU-run launcher + result loader for rungs 4, 5, 6.\n",
        "\n",
        "**This notebook does NOT do GPU work.** It tells you what to run on RunPod\n",
        "and then loads the resulting CSVs after the GPU run finishes.\n",
    ]),
    code([
        "%load_ext autoreload\n",
        "%autoreload 2\n",
        "\n",
        "import sys; from pathlib import Path\n",
        "sys.path.insert(0, str(Path('..').resolve()))\n",
        "\n",
        "import pandas as pd\n",
        "import matplotlib.pyplot as plt\n",
        "\n",
        "from src import config, runner\n",
    ]),
    md(["## 1. Status of TCNN experiments"]),
    code([
        "TCNN_CONFIGS = ['rung_4_linear_tcnn.yaml', 'rung_5_tcnn_1ch.yaml', 'rung_6_tcnn_3ch.yaml']\n",
        "configs = [runner.load_config(config.EXPERIMENTS_DIR / c) for c in TCNN_CONFIGS]\n",
        "runner.status_table(configs)\n",
    ]),
    md([
        "## 2. Launch on RunPod (manual)\n",
        "\n",
        "ssh into RunPod, cd into this project root, then:\n",
        "\n",
        "```bash\n",
        "# rung 4 — linear TCNN (~3 GPU-hours for full sweep)\n",
        "python -m train.train_tcnn --config experiments/rung_4_linear_tcnn.yaml\n",
        "\n",
        "# rung 5 — full TCNN, 1-channel (~25 GPU-hours with 5 seeds)\n",
        "python -m train.train_tcnn --config experiments/rung_5_tcnn_1ch.yaml\n",
        "\n",
        "# rung 6 — full TCNN, 3-channel (~25 GPU-hours)\n",
        "python -m train.train_tcnn --config experiments/rung_6_tcnn_3ch.yaml\n",
        "\n",
        "# Or run all 3 in sequence:\n",
        "python -m train.train_tcnn --sweep experiments/_track_a.yaml\n",
        "```\n",
        "\n",
        "After the run finishes, rsync `outputs/` back from RunPod and re-run cells below.\n",
    ]),
    md(["## 3. Load results + plot training curves"]),
    code([
        "from src import eval as evalmod\n",
        "rung_ids = [c['experiment_id'] for c in configs]\n",
        "try:\n",
        "    master = evalmod.load_master_results(rung_ids)\n",
        "    print(f'Loaded {len(master):,} daily-return records across {len(rung_ids)} TCNN variants')\n",
        "except Exception as e:\n",
        "    print(f'No results yet: {e}')\n",
        "    print('Run the GPU jobs first.')\n",
    ]),
    code([
        "# Per-year Sharpe by seed (sanity check that training is stable)\n",
        "if 'master' in dir() and len(master) > 0:\n",
        "    perf_by_year_seed = (master\n",
        "        .groupby(['experiment_id', 'year', 'seed'])['return']\n",
        "        .apply(evalmod.annualized_sharpe)\n",
        "        .reset_index(name='sharpe'))\n",
        "    print(perf_by_year_seed.head(20))\n",
    ]),
    code([
        "# Cumulative P&L per TCNN variant\n",
        "if 'master' in dir() and len(master) > 0:\n",
        "    fig, ax = plt.subplots(figsize=(12, 5))\n",
        "    for exp_id in rung_ids:\n",
        "        df = master[master['experiment_id'] == exp_id].sort_values('date')\n",
        "        ax.plot(df['date'], df['return'].cumsum(), label=exp_id, alpha=0.85)\n",
        "    ax.set(title='TCNN cumulative P&L (cumsum)', ylabel='cum return')\n",
        "    ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()\n",
    ]),
])


# =============================================================================
# 04 — portfolio eval matrix (3-portfolio eval on TCNN scores)
# =============================================================================

nb_04 = make_nb([
    md([
        "# 04 — Portfolio evaluation matrix\n",
        "\n",
        "Take TCNN trained models and evaluate their scores under **three** portfolio\n",
        "constructions: dual-softmax (paper), decile-sort EW, and MVO with Ledoit-Wolf shrinkage.\n",
        "\n",
        "Tells us whether the dual-softmax portfolio mapping is the bottleneck.\n",
    ]),
    code([
        "import sys; from pathlib import Path\n",
        "sys.path.insert(0, str(Path('..').resolve()))\n",
        "\n",
        "import pandas as pd, numpy as np, torch\n",
        "from src import config, panels, portfolio, train_tcnn, eval as evalmod\n",
        "from src.train_tcnn import TCNEncoder, two_softmax_weights_batched\n",
    ]),
    md([
        "## 1. Loop over trained TCNN checkpoints, score each rebal date, evaluate under 3 portfolios\n",
        "\n",
        "Pseudo-code (concrete implementation depends on which seed/year cells exist):\n",
        "\n",
        "```python\n",
        "for variant in ['rung_5_tcnn_1ch', 'rung_6_tcnn_3ch']:\n",
        "    for year in range(2010, 2024):\n",
        "        for seed in range(5):\n",
        "            ckpt = torch.load(f'outputs/{variant}/year_{year}/seed_{seed}/model.pt')\n",
        "            scores = score_test_year(ckpt, panels, year)\n",
        "            for portfolio_name in ['dual_softmax', 'decile_sort_ew', 'mvo_lw']:\n",
        "                weights = portfolio.get_portfolio_fn(portfolio_name)(scores, ...)\n",
        "                returns = compute_returns(weights, ...)\n",
        "                save(f'outputs/{variant}_{portfolio_name}_year{year}_seed{seed}.csv', returns)\n",
        "```\n",
        "\n",
        "TODO: implement after rungs 5/6 actually run. Notebook scaffold is ready.\n",
    ]),
    md(["## 2. Compare net Sharpe across portfolio constructions"]),
    code([
        "# Skeleton — fill in once eval matrix is computed\n",
        "summary_table = pd.DataFrame({\n",
        "    'rung': ['5 (1-ch TCNN)', '5 (1-ch TCNN)', '5 (1-ch TCNN)', '6 (3-ch TCNN)', '6 (3-ch TCNN)', '6 (3-ch TCNN)'],\n",
        "    'portfolio': ['dual_softmax', 'decile_sort_ew', 'mvo_lw'] * 2,\n",
        "    'sharpe': [None] * 6,        # to be filled in\n",
        "    'max_dd': [None] * 6,\n",
        "    'turnover_ann_pct': [None] * 6,\n",
        "})\n",
        "summary_table\n",
    ]),
])


# =============================================================================
# 05 — ladder summary (all 9 rungs side-by-side)
# =============================================================================

nb_05 = make_nb([
    md([
        "# 05 — Ladder summary\n",
        "\n",
        "All 9 rungs side-by-side. The single chart that tells the F3 + F2 story.\n",
    ]),
    code([
        "import sys; from pathlib import Path\n",
        "sys.path.insert(0, str(Path('..').resolve()))\n",
        "\n",
        "import pandas as pd, numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from src import config, eval as evalmod\n",
    ]),
    code([
        "RUNG_IDS = ['rung_1_simple_momentum_decile', 'rung_1d_simple_momentum_dual',\n",
        "            'rung_2_ewm_momentum_decile',    'rung_2d_ewm_momentum_dual',\n",
        "            'rung_3_ts_regression_decile',   'rung_3d_ts_regression_dual',\n",
        "            'rung_4_linear_tcnn',            'rung_5_tcnn_1ch', 'rung_6_tcnn_3ch']\n",
        "master = evalmod.load_master_results(RUNG_IDS)\n",
        "print(f'{len(master):,} daily-return records across {master[\"experiment_id\"].nunique()} rungs')\n",
    ]),
    md(["## Sharpe / max-DD table (the headline)"]),
    code([
        "summary = (master.groupby('experiment_id')['return']\n",
        "                  .apply(lambda s: pd.Series(evalmod.perf_summary(s)))\n",
        "                  .unstack())\n",
        "summary = summary[['ann_return', 'ann_vol', 'sharpe', 'max_dd', 't_stat', 'n_days']]\n",
        "summary.loc[RUNG_IDS].round(3)  # ordered by rung\n",
    ]),
    md(["## Cumulative P&L plot (cumsum)"]),
    code([
        "fig, ax = plt.subplots(figsize=(14, 6))\n",
        "for exp_id in RUNG_IDS:\n",
        "    df = master[master['experiment_id'] == exp_id].sort_values('date')\n",
        "    ax.plot(df['date'], df['return'].cumsum(), label=exp_id, alpha=0.85)\n",
        "ax.set(title='Comparison ladder — cumulative P&L (cumsum)', ylabel='cum return')\n",
        "ax.legend(loc='upper left', fontsize=8); ax.grid(alpha=0.3); plt.tight_layout()\n",
    ]),
    md(["## Decomposition: F3 (learnable aggregation) and F2 (end-to-end) gaps"]),
    code([
        "# F3 ladder: 1 → 2 → 3 → 4 → 5 (each step adds one capability)\n",
        "f3_ladder = ['rung_1_simple_momentum_decile', 'rung_2_ewm_momentum_decile',\n",
        "             'rung_3_ts_regression_decile', 'rung_4_linear_tcnn', 'rung_5_tcnn_1ch']\n",
        "f3 = summary.loc[f3_ladder, ['sharpe']].copy()\n",
        "f3['delta_sharpe'] = f3['sharpe'].diff()\n",
        "f3.round(3)\n",
    ]),
])


# =============================================================================
# 06 — diagnostics (IC, decile, FF5 neutrality, regime)
# =============================================================================

nb_06 = make_nb([
    md([
        "# 06 — Diagnostics\n",
        "\n",
        "PM-grade defensibility checks: IC, decile anatomy, factor neutrality, regime stability.\n",
    ]),
    code([
        "import sys; from pathlib import Path\n",
        "sys.path.insert(0, str(Path('..').resolve()))\n",
        "import pandas as pd, numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from src import config, eval as evalmod\n",
    ]),
    code([
        "panel = pd.read_parquet(config.PANEL_DAILY_PARQUET)\n",
        "panel['date'] = pd.to_datetime(panel['date'])\n",
    ]),
    md(["## 1. IC by month for each baseline factor (rungs 1-3)"]),
    code([
        "for score_col in ['mom_12_1', 'ewm_h60_skip']:\n",
        "    ic_series = evalmod.ic_by_month(panel, score_col, ret_fut_col='ret_fut_1m')\n",
        "    summary = evalmod.ic_summary(ic_series)\n",
        "    print(f'{score_col:<20} mean IC = {summary[\"mean_ic\"]:.4f}  IR = {summary[\"ir\"]:.3f}  hit = {summary[\"hit_rate\"]:.2f}')\n",
    ]),
    md(["## 2. Decile anatomy (D10 - D1 spread)"]),
    code([
        "for score_col in ['mom_12_1', 'ewm_h60_skip']:\n",
        "    d = evalmod.decile_returns(panel, score_col, ret_fut_col='ret_fut_1m')\n",
        "    summary = evalmod.decile_spread_summary(d)\n",
        "    print(f'{score_col:<20} D10-D1 mean = {summary[\"hi_lo_mean\"]:.4f}  t = {summary[\"hi_lo_t\"]:.2f}  monotonic = {summary[\"monotonic\"]}')\n",
    ]),
    md(["## 3. FF5 + UMD neutrality regression"]),
    code([
        "# Requires daily strategy returns (from outputs/) and FF5 daily factors\n",
        "ff5 = pd.read_parquet(config.FF5_RAW)\n",
        "ff5['date'] = pd.to_datetime(ff5['date'])\n",
        "\n",
        "RUNG_IDS = ['rung_1_simple_momentum_decile', 'rung_5_tcnn_1ch']\n",
        "master = evalmod.load_master_results(RUNG_IDS)\n",
        "for exp_id in RUNG_IDS:\n",
        "    df = master[master['experiment_id'] == exp_id].sort_values('date').set_index('date')\n",
        "    daily = df['return']\n",
        "    result = evalmod.ff5_neutrality(daily, ff5, include_umd=False)  # umd typically pulled separately\n",
        "    print(f'{exp_id:<35} alpha = {result[\"alpha_ann\"]*100:.2f}% (t = {result[\"alpha_t\"]:.2f}), R² = {result[\"r_squared\"]:.3f}')\n",
    ]),
])


# =============================================================================
# 07 — capacity + TC sweep
# =============================================================================

nb_07 = make_nb([
    md([
        "# 07 — Capacity + transaction-cost sweep\n",
        "\n",
        "Net Sharpe at fixed cost assumptions (0/5/10/20 bps round-trip), capacity curve\n",
        "via square-root-impact-law extrapolation.\n",
    ]),
    code([
        "import sys; from pathlib import Path\n",
        "sys.path.insert(0, str(Path('..').resolve()))\n",
        "import pandas as pd, numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from src import config, eval as evalmod\n",
    ]),
    md(["## 1. TC sweep at fixed costs"]),
    code([
        "# Need: daily_returns (one rung) + per-rebal turnover. Turnover requires the weight\n",
        "# history, which we save during runs (TODO: ensure runner writes weights_history).\n",
        "# For now, illustrative skeleton:\n",
        "\n",
        "# RUNG = 'rung_5_tcnn_1ch'\n",
        "# master = evalmod.load_master_results([RUNG])\n",
        "# daily_returns = master.groupby('date')['return'].mean()  # avg across seeds\n",
        "# turnover = compute_turnover_per_rebal(weights_history)\n",
        "# tc_sweep = evalmod.tc_sweep(daily_returns, turnover, cost_grid_bps=(0, 5, 10, 20))\n",
        "# tc_sweep\n",
    ]),
    md(["## 2. Capacity curve (sqrt-impact law)"]),
    code([
        "# capacity = evalmod.capacity_curve(daily_returns, turnover,\n",
        "#                                    aum_grid_usd=(50e6, 200e6, 1e9, 5e9),\n",
        "#                                    base_aum_usd=50e6,\n",
        "#                                    base_cost_bps_round_trip=10.0,\n",
        "#                                    impact_exponent=0.5)\n",
        "# capacity\n",
    ]),
])


# =============================================================================
# 08 — paper figures
# =============================================================================

nb_08 = make_nb([
    md([
        "# 08 — Paper figures\n",
        "\n",
        "Final figures + tables for the paper / resume bullet:\n",
        "  - Figure 1: cumulative OOS returns (paper-style)\n",
        "  - Figure 2: Sharpe by realized-vol regime (paper-style)\n",
        "  - Headline summary table (Sharpe / vol / max-DD per rung)\n",
        "  - Decomposition table (F3 + F2 gaps)\n",
        "  - Net-Sharpe-vs-TC curve\n",
    ]),
    code([
        "import sys; from pathlib import Path\n",
        "sys.path.insert(0, str(Path('..').resolve()))\n",
        "import pandas as pd, numpy as np\n",
        "import matplotlib.pyplot as plt\n",
        "from src import config, eval as evalmod\n",
    ]),
    md(["## Figure 1 — cumulative P&L (vol-matched)"]),
    code([
        "# Replicate paper's Figure 1 with our 9 rungs (vs paper's 5)\n",
        "RUNG_IDS = ['rung_1_simple_momentum_decile', 'rung_2_ewm_momentum_decile',\n",
        "            'rung_3_ts_regression_decile', 'rung_4_linear_tcnn',\n",
        "            'rung_5_tcnn_1ch', 'rung_6_tcnn_3ch']\n",
        "master = evalmod.load_master_results(RUNG_IDS)\n",
        "\n",
        "# Vol-match each rung to rung 1's volatility (paper convention)\n",
        "target_vol = master[master['experiment_id'] == RUNG_IDS[0]]['return'].std() * np.sqrt(252)\n",
        "\n",
        "fig, ax = plt.subplots(figsize=(14, 6))\n",
        "for exp_id in RUNG_IDS:\n",
        "    df = master[master['experiment_id'] == exp_id].sort_values('date')\n",
        "    raw_vol = df['return'].std() * np.sqrt(252)\n",
        "    scale = target_vol / (raw_vol + 1e-12)\n",
        "    ax.plot(df['date'], (df['return'] * scale).cumsum(), label=exp_id, alpha=0.85)\n",
        "ax.set(title=f'Cumulative OOS Returns (vol-matched to rung 1, vol={target_vol*100:.1f}%)',\n",
        "       ylabel='cumulative return')\n",
        "ax.legend(); ax.grid(alpha=0.3); plt.tight_layout()\n",
    ]),
    md(["## Headline summary table"]),
    code([
        "summary = (master.groupby('experiment_id')['return']\n",
        "                  .apply(lambda s: pd.Series(evalmod.perf_summary(s)))\n",
        "                  .unstack()\n",
        "                  .loc[RUNG_IDS, ['ann_return', 'ann_vol', 'sharpe', 'max_dd', 't_stat']]\n",
        "                  .round(3))\n",
        "summary\n",
    ]),
])


# =============================================================================
# Write all
# =============================================================================

NOTEBOOKS = {
    "00_setup.ipynb":                  nb_00,
    "01_panel_construction.ipynb":     nb_01,
    "02_factor_baselines.ipynb":       nb_02,
    "03_tcnn_orchestrator.ipynb":      nb_03,
    "04_portfolio_eval_matrix.ipynb":  nb_04,
    "05_ladder_summary.ipynb":         nb_05,
    "06_diagnostics.ipynb":            nb_06,
    "07_capacity_tc_sweep.ipynb":      nb_07,
    "08_paper_figures.ipynb":          nb_08,
}

for name, nb in NOTEBOOKS.items():
    out_path = NB_DIR / name
    with open(out_path, "w") as f:
        json.dump(nb, f, indent=1)
    print(f"  wrote {out_path}")

print(f"\n{len(NOTEBOOKS)} notebooks written to {NB_DIR}")
