#!/bin/bash
# RunPod setup — run this on the RunPod box once after creating the pod.
#
# Assumes:
#   - Project rsynced to /workspace/TCNN/
#   - panel_daily.parquet already built locally and rsynced to data/03_features/
#
# Usage:
#   bash scripts/setup_runpod.sh
#   python -m train.train_tcnn --sweep experiments/_track_a.yaml

set -e

cd /workspace/TCNN

echo "=== Installing dependencies ==="
pip install --quiet -r requirements.txt

echo "=== Verifying data ==="
ls -lh data/03_features/panel_daily.parquet || {
    echo "ERROR: panel_daily.parquet missing. rsync it first."
    exit 1
}

echo "=== Verifying torch + CUDA ==="
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available()); print('device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"

echo "=== Sanity smoke test (1 year, 2 epochs, CPU) ==="
python -m train.train_tcnn --config experiments/rung_4_linear_tcnn.yaml --status

echo ""
echo "=== Setup complete. To run Track A sweep: ==="
echo "  python -m train.train_tcnn --sweep experiments/_track_a.yaml"
echo ""
echo "=== Or one experiment at a time: ==="
echo "  python -m train.train_tcnn --config experiments/rung_4_linear_tcnn.yaml"
echo "  python -m train.train_tcnn --config experiments/rung_5_tcnn_1ch.yaml"
echo "  python -m train.train_tcnn --config experiments/rung_6_tcnn_3ch.yaml"
