#!/usr/bin/env python3
"""Stream-merge a PEFT LoRA adapter into a sharded HF checkpoint.

Plan-B path for the duck-27b-lora eval kernel: if vLLM --enable-lora fails
to serve Qwen3_5ForConditionalGeneration at boot, run this inside the eval
kernel (torch+safetensors ship in the wheelhouse) to produce a merged bf16
model dir, then serve it with the plain flags (drop --enable-lora /
--lora-modules; keep SERVED_MODEL_NAME).

  BASE=/kaggle/input/datasets/rahim3/qwen3-8-27b-bf16
  ADAPT=<resolved taaf-duck-lora-v1 path>
  OUT=/kaggle/working/merged27b
  python merge_lora_stream.py "$BASE" "$ADAPT" "$OUT"

Streams shard-by-shard (peak RAM ~ one shard); copies tokenizer/config files.
Key matching is suffix-based on `layers.N.<module>.weight`, so it tolerates
both `model.language_model.layers...` (VL wrap) and `model.layers...`
(text-only repack) nesting in adapter and checkpoint.
"""
import json
import os
import re
import shutil
import sys


def _tail(key: str) -> str | None:
    """'anything.layers.7.mlp.gate_proj.weight' -> 'layers.7.mlp.gate_proj.weight'."""
    m = re.search(r"layers\.\d+\..*\.weight$", key)
    return m.group(0) if m else None


def main(base_dir: str, adapter_dir: str, out_dir: str) -> None:
    import torch
    from safetensors.torch import load_file, save_file

    cfg = json.load(open(os.path.join(adapter_dir, "adapter_config.json")))
    r, alpha = cfg["r"], cfg["lora_alpha"]
    scale = alpha / r
    ad = load_file(os.path.join(adapter_dir, "adapter_model.safetensors"))

    pairs = {}  # tail -> (A, B)
    for k in ad:
        if not k.endswith(".lora_A.weight"):
            continue
        tail = _tail(k.removesuffix(".lora_A.weight") + ".weight")
        assert tail, f"cannot locate layer tail in adapter key {k}"
        b_key = k[: -len("lora_A.weight")] + "lora_B.weight"
        assert b_key in ad, f"missing lora_B for {k}"
        pairs[tail] = (ad[k], ad[b_key])
    assert pairs, "no lora_A keys parsed — check adapter naming"
    print(f"merge: {len(pairs)} target weights, scale={scale}")

    os.makedirs(out_dir, exist_ok=True)
    for f in os.listdir(base_dir):
        if f.endswith(".safetensors"):
            continue
        src = os.path.join(base_dir, f)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(out_dir, f))

    idx_path = os.path.join(base_dir, "model.safetensors.index.json")
    if os.path.exists(idx_path):
        shards = sorted(set(json.load(open(idx_path))["weight_map"].values()))
    else:
        shards = ["model.safetensors"]
    print(f"shards: {len(shards)}")

    merged_total = 0
    for shard in shards:
        tensors = load_file(os.path.join(base_dir, shard))
        n_hit = 0
        for k in list(tensors.keys()):
            tail = _tail(k)
            if tail and tail in pairs:
                A, B = pairs[tail]
                tensors[k] = tensors[k] + (B @ A).to(tensors[k].dtype) * scale
                n_hit += 1
        save_file(tensors, os.path.join(out_dir, shard),
                  metadata={"format": "pt"})
        merged_total += n_hit
        print(f"  {shard}: +{n_hit} merged")
    print(f"done: {merged_total}/{len(pairs)} weights merged -> {out_dir}")
    assert merged_total == len(pairs), "some adapter weights matched no shard key"


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3])
