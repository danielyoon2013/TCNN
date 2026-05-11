#!/bin/bash
# rsync results back from RunPod to local Dropbox project.
#
# Usage (on local machine):
#   bash scripts/sync_results.sh runpod-host:/workspace/TCNN

set -e

REMOTE="${1:-runpod:/workspace/TCNN}"
LOCAL_DIR="$(cd "$(dirname "$0")/.." && pwd)"

echo "=== Pulling outputs/ from $REMOTE ==="
rsync -avz --include='*/' --include='*.csv' --include='*.json' --include='*.pt' --exclude='*' \
    "${REMOTE}/outputs/" "${LOCAL_DIR}/outputs/"

echo "=== Done. Open notebooks/05_ladder_summary.ipynb to inspect. ==="
