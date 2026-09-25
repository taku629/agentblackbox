#!/usr/bin/env bash
# Saturday pipeline: after arc3-duck-sft-train completes, turn its adapter
# output into a dataset and push the LoRA-serving eval kernel.
# Usage: cd arc-agi-3 && ./sat_lora_pipeline.sh
set -euo pipefail
cd "$(dirname "$0")/.."

K=/home/ubuntu/.venv-kaggle/bin/kaggle
OUT=/tmp/lora_adapter_out

echo "== 1/3 fetching adapter from arc3-duck-sft-train =="
rm -rf "$OUT" && mkdir -p "$OUT"
"$K" kernels output takumuhata/arc3-duck-sft-train -p "$OUT"
ls -la "$OUT"
# adapter may land at root or under lora_adapter/
if [ -d "$OUT/lora_adapter" ]; then AD="$OUT/lora_adapter"; else AD="$OUT"; fi
test -f "$AD/adapter_config.json" || { echo "adapter_config.json missing in $AD"; find "$OUT" -name adapter_config.json; exit 1; }

echo "== 2/3 creating dataset takumuhata/taaf-duck-lora-v1 =="
cat > "$AD/dataset-metadata.json" <<'META'
{"title":"taaf-duck-lora-v1","id":"takumuhata/taaf-duck-lora-v1","licenses":[{"name":"other"}]}
META
"$K" datasets create -p "$AD"

echo "== 3/3 pushing LoRA eval kernel arc3-duck-anim-27b-lora =="
"$K" kernels push -p submit_ag3_lora
echo "done — watch: $K kernels status takumuhata/arc3-duck-anim-27b-lora"
