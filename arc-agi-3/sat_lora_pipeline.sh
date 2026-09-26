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
# adapter lands at lora_adapter/ root on success; if the kernel timed out,
# fall back to the newest Trainer checkpoint (still a usable adapter)
if [ -f "$OUT/lora_adapter/adapter_config.json" ]; then
  AD="$OUT/lora_adapter"
elif [ -f "$OUT/adapter_config.json" ]; then
  AD="$OUT"
else
  last_ckpt=$(ls -d "$OUT"/checkpoint-* "$OUT"/lora_adapter/checkpoint-* 2>/dev/null | sort -t- -k2 -n | tail -1 || true)
  if [ -n "$last_ckpt" ] && [ -f "$last_ckpt/adapter_config.json" ]; then
    AD="$last_ckpt"; echo "kernel timed out — using checkpoint adapter: $last_ckpt"
  else
    echo "no adapter found"; find "$OUT" -name adapter_config.json; exit 1
  fi
fi
test -f "$AD/adapter_model.safetensors" || { echo "adapter weights missing in $AD"; ls -la "$AD"; exit 1; }

# sanity: adapter_config parses and the safetensors header holds LoRA keys —
# catches a corrupt/partial download before we burn a GPU eval run on it
python3 - "$AD" <<'PY'
import json, struct, sys
ad = sys.argv[1]
cfg = json.load(open(f"{ad}/adapter_config.json"))
assert cfg.get("r") == 16, f"unexpected r: {cfg.get('r')}"
raw = open(f"{ad}/adapter_model.safetensors", "rb")
hlen = struct.unpack("<Q", raw.read(8))[0]
hdr = json.loads(raw.read(hlen))
lora = [k for k in hdr if "lora_" in k]
assert lora, "no lora_* tensors in adapter_model.safetensors"
print(f"adapter ok: r={cfg['r']}, {len(lora)} lora tensors, {len(hdr)} total")
PY

echo "== 2/3 upserting dataset takumuhata/taaf-duck-lora-v1 =="
cat > "$AD/dataset-metadata.json" <<'META'
{"title":"taaf-duck-lora-v1","id":"takumuhata/taaf-duck-lora-v1","licenses":[{"name":"other"}]}
META
if "$K" datasets status takumuhata/taaf-duck-lora-v1 >/dev/null 2>&1; then
  "$K" datasets version -p "$AD" -m "adapter update"
else
  "$K" datasets create -p "$AD"
fi

echo "== 3/3 pushing LoRA eval kernel arc3-duck-anim-27b-lora =="
push_out=$("$K" kernels push -p submit_ag3_lora 2>&1)
echo "$push_out"
ver=$(echo "$push_out" | grep -oE 'Kernel version [0-9]+' | grep -oE '[0-9]+' || true)
if [ -n "$ver" ]; then
  python3 - <<PY
import json
p = "arc-agi-3/kernel_versions.json"
m = json.load(open(p)) if __import__("os").path.exists(p) else {}
m["takumuhata/arc3-duck-anim-27b-lora"] = $ver
json.dump(m, open(p, "w"), indent=2)
print("kernel_versions.json updated:", m)
PY
fi
echo "done — watch: $K kernels status takumuhata/arc3-duck-anim-27b-lora"
