#!/usr/bin/env bash
# One-time: push the evalscan kernel (full-eval decode, ~10-12h L4x4).
# Run inside the Saturday ARC quota window only.
#   bash arc-agi-2/push_evalscan.sh
set -euo pipefail
cd "$(dirname "$0")/.."
KAGGLE=/home/ubuntu/.venv-kaggle/bin/kaggle

echo "== pushing takumuhata/arc-agi2-evalscan =="
cd submit_ag2_evalscan
$KAGGLE kernels push
echo
echo "== watch: $KAGGLE kernels status takumuhata/arc-agi2-evalscan =="
echo "== when COMPLETE, download output and run: =="
echo "   mkdir -p /tmp/evalscan && cd /tmp/evalscan"
echo "   $KAGGLE kernels output takumuhata/arc-agi2-evalscan"
echo "   python3 /home/ubuntu/repos/arc-prize-2026-agent-work/arc-agi-2/dev/select_bench.py /tmp/evalscan/inference_outputs"
