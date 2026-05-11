"""CLI entry point for running an experiment (or a sweep) from a YAML config.

This is the script you run on RunPod or locally:

    python -m train.train_tcnn --config experiments/rung_5_tcnn_1ch.yaml
    python -m train.train_tcnn --sweep experiments/_track_a.yaml

Loads the canonical panel from data/03_features/panel_daily.parquet and
delegates to src.runner. The actual factor / portfolio / training logic
lives in the src/ modules.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import pandas as pd

# Make `src.*` importable when running as a script from anywhere
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src import config, runner   # noqa: E402


def parse_args():
    p = argparse.ArgumentParser(description="Run a TCNN experiment or sweep from a YAML config.")
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--config", help="Path to a single experiment YAML")
    g.add_argument("--sweep",  help="Path to a sweep manifest YAML")
    p.add_argument("--panel",  default=str(config.PANEL_DAILY_PARQUET),
                   help="Path to panel_daily.parquet (default: data/03_features/panel_daily.parquet)")
    p.add_argument("--force",  action="store_true",
                   help="Re-run all cells, even if outputs exist")
    p.add_argument("--status", action="store_true",
                   help="Print status table and exit (no work)")
    return p.parse_args()


def main():
    args = parse_args()

    print(f"Loading panel from {args.panel}...")
    panel = pd.read_parquet(args.panel)
    panel["date"] = pd.to_datetime(panel["date"])
    print(f"  panel: {len(panel):,} rows, {panel['date'].min()} -> {panel['date'].max()}")

    if args.config:
        cfg = runner.load_config(args.config)
        if args.status:
            print(runner.status_table([cfg]).to_string(index=False))
            return
        runner.run_experiment(cfg, panel, force=args.force)

    elif args.sweep:
        configs = runner.load_sweep(args.sweep)
        if args.status:
            print(runner.status_table(configs).to_string(index=False))
            return
        runner.run_sweep(args.sweep, panel, force=args.force)


if __name__ == "__main__":
    main()
