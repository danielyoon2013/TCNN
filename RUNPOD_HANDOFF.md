# RunPod handoff — what to do for the TCNN training (rungs 4, 5, 6)

The CPU baselines (rungs 1, 1d, 2, 2d, 3, 3d) run locally. The TCNN rungs need a GPU.

## Cost estimate

| Run | GPU-hours | Cost (RunPod community A100, ~$0.80/hr) |
|---|---:|---:|
| Rung 4 (linear TCNN, 5 seeds × 14 years) | ~3 hr | ~$3 |
| Rung 5 (1-channel TCNN, 5 seeds × 14 years) | ~25 hr | ~$20 |
| Rung 6 (3-channel TCNN, 5 seeds × 14 years) | ~25 hr | ~$20 |
| **Track A total** | **~50 hr** | **~$45** |
| Rung 5 + 6 with seeds={0,1} only (faster check) | ~20 hr | ~$15 |

The 5-seed ensemble is recommended (smooths noise, +0.05-0.10 Sharpe lift). If budget-constrained, use seeds={0} only — gets you ~1/5 of the cost at the price of higher variance per cell.

## Setup steps

### 1. Create a RunPod pod

- Log into [runpod.io](https://runpod.io), navigate to **Community Cloud** (cheapest)
- Pick an **A100 80GB** template with PyTorch + CUDA pre-installed (e.g., `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`)
- Region: any with availability; US-CA / US-OR are usually cheapest
- Storage: ~50 GB persistent volume mounted at `/workspace` (the project + parquet + tensor cache will fit easily)
- Click "Deploy"
- Once running, note the **SSH command** RunPod gives you (looks like `ssh root@<host> -p <port> -i ~/.ssh/id_ed25519`)

### 2. rsync the project + panel from local

From your local Windows terminal (Git Bash or WSL):

```bash
# In Git Bash on Windows, from /c/Users/danielyoon/Dropbox/Y2K/Y2K_quant/interview_prep/04_Projects/TCNN/

# Set RunPod SSH details
export RUNPOD_HOST="root@<host>"
export RUNPOD_PORT="<port>"

# Push project (excludes data/, outputs/, code/_legacy_*, __pycache__)
rsync -avz -e "ssh -p $RUNPOD_PORT" \
    --exclude='data/' --exclude='outputs/' --exclude='__pycache__' \
    --exclude='code/' --exclude='paper/' --exclude='prep/' \
    --exclude='notebooks/_legacy/' \
    ./ $RUNPOD_HOST:/workspace/TCNN/

# Push the panel (1.4 GB; takes 2-5 min depending on your upload speed)
rsync -avz -e "ssh -p $RUNPOD_PORT" \
    data/03_features/panel_daily.parquet \
    $RUNPOD_HOST:/workspace/TCNN/data/03_features/

# Push the raw parquets too (so you can rebuild panel on RunPod if needed)
rsync -avz -e "ssh -p $RUNPOD_PORT" \
    data/01_raw/ \
    $RUNPOD_HOST:/workspace/TCNN/data/01_raw/

# Verify
ssh -p $RUNPOD_PORT $RUNPOD_HOST "ls -lh /workspace/TCNN/data/03_features/"
```

### 3. Set up the env on RunPod

ssh in and run:

```bash
ssh -p $RUNPOD_PORT $RUNPOD_HOST
cd /workspace/TCNN
bash scripts/setup_runpod.sh
```

This installs deps, verifies CUDA, and prints a status table.

### 4. Run the TCNN sweep

```bash
# Quick rung-4 first as a sanity check (~3 hours)
python -m train.train_tcnn --config experiments/rung_4_linear_tcnn.yaml

# Then rung 5 (~25 hours)
python -m train.train_tcnn --config experiments/rung_5_tcnn_1ch.yaml

# Then rung 6 (~25 hours)
python -m train.train_tcnn --config experiments/rung_6_tcnn_3ch.yaml

# Or run all 3 in sequence in one command:
python -m train.train_tcnn --sweep experiments/_track_a.yaml
```

Each run is **resumable**: if the pod crashes mid-run, just re-launch the same command. The runner skips cells that already have `outputs/<rung>/year_Y/seed_S/returns.csv`.

Run inside `tmux` or `nohup` so the run survives ssh disconnects:

```bash
tmux new -s tcnn_run
python -m train.train_tcnn --sweep experiments/_track_a.yaml
# Detach with Ctrl-B, then D
# Reattach later with: tmux attach -t tcnn_run
```

### 5. Sync results back to local

After the sweep completes (or partway through, if you want to inspect):

```bash
# From local Git Bash:
rsync -avz -e "ssh -p $RUNPOD_PORT" \
    $RUNPOD_HOST:/workspace/TCNN/outputs/ \
    ./outputs/

# Or use the helper script:
bash scripts/sync_results.sh "$RUNPOD_HOST:/workspace/TCNN" -p $RUNPOD_PORT
```

Outputs are small (~10 MB total — model state dicts are <1 MB each).

### 6. Stop the pod

**Important:** RunPod community-cloud charges per minute the pod is running, even if idle. After you've synced results back:

- Go to runpod.io → My Pods → click the pod → **Terminate**
- Persistent volume is destroyed unless you specifically chose "Persistent Volume" — be sure to download anything you need first.

## Troubleshooting

- **OOM on A100**: training script uses `mmap_mode='r'` for X_panel (16 GB) so VRAM should never see the full tensor. If you OOM during training, reduce `batch_size` in the YAML config.
- **Determinism warning**: `torch.use_deterministic_algorithms(True, warn_only=True)` may print warnings about non-deterministic CUDA ops; harmless.
- **Slow data loading on first cell**: the first cell of each rung builds the (T, N, F, L) tensor cache to disk (~5-10 min). Subsequent cells reuse the cache.
- **WRDS not accessible from RunPod**: that's fine — the panel is already built locally and rsynced. RunPod doesn't need WRDS.

## After all 3 TCNN rungs complete

Run the analysis notebooks locally on the synced outputs:
- `notebooks/04_portfolio_eval_matrix.ipynb` — re-evaluate trained TCNN scores under decile-sort + MVO portfolios (3-portfolio matrix)
- `notebooks/05_ladder_summary.ipynb` — all 9 rungs side-by-side
- `notebooks/06_diagnostics.ipynb` — IC, decile, FF5 neutrality
- `notebooks/07_capacity_tc_sweep.ipynb` — net Sharpe vs cost
- `notebooks/08_paper_figures.ipynb` — final figures + tables

Then the resume bullet number is whatever rung 5 (or 6) Sharpe ends up being, after the 8 bug fixes.
