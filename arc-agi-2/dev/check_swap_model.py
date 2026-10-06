"""Is a checkpoint a DROP-IN for the perfpatch kernel? CPU only, no torch.

    python3 check_swap_model.py <candidate_dir> [<reference_dir>]
    python3 check_swap_model.py --selftest

A 12 h L4x4 run on a model that does not fit the pipeline scores 0 and burns
the weekly quota, so check the directory before building a kernel. The
pipeline hard-codes token ids (arc_solver.py ARC_VOCAB: digits 0-9, newline
10, user 11, assistant 12, pad 13, <|im_end|> 15) and restricts decoding to
them, so a model with the stock Qwen vocabulary is NOT compatible even if the
architecture matches.

Checks (FAIL = do not build, WARN = look before building):
  tokenizer   the 15 hard-coded tokens map to the expected ids
  config      vocab_size covers id 15; architecture / sizes vs the reference
  weights     safetensors headers: parameter count, dtype, embedding rows
  memory      bf16 size against a 24 GB L4 with TTT (r=256 LoRA incl. embed_tokens/lm_head)
"""
import glob
import json
import os
import struct
import sys

REQUIRED = {**{str(d): d for d in range(10)}, "user": 11, "assistant": 12, "<|im_end|>": 15}
NEWLINE_IDS = {"\u010a": 10, "\n": 10}             # byte-level BPE spells newline as U+010A
CFG_KEYS = ["model_type", "architectures", "hidden_size", "num_hidden_layers", "num_attention_heads",
            "num_key_value_heads", "intermediate_size", "vocab_size", "torch_dtype", "max_position_embeddings"]
DTYPE_BYTES = {"F64": 8, "F32": 4, "F16": 2, "BF16": 2, "I64": 8, "I32": 4, "I16": 2, "I8": 1, "U8": 1, "BOOL": 1}


def read_safetensors_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n))


def weights_summary(d):
    params, by_dtype, embed_rows, nbytes = 0, {}, None, 0
    files = sorted(glob.glob(os.path.join(d, "*.safetensors")))
    for f in files:
        for name, t in read_safetensors_header(f).items():
            if name == "__metadata__":
                continue
            n = 1
            for s in t["shape"]:
                n *= s
            params += n
            by_dtype[t["dtype"]] = by_dtype.get(t["dtype"], 0) + n
            nbytes += n * DTYPE_BYTES.get(t["dtype"], 4)
            if name.endswith("embed_tokens.weight"):
                embed_rows = t["shape"][0]
    return {"files": len(files), "params": params, "by_dtype": by_dtype, "embed_rows": embed_rows, "bytes": nbytes}


def vocab_of(d):
    p = os.path.join(d, "tokenizer.json")
    if not os.path.exists(p):
        return None, None
    tj = json.load(open(p))
    vocab = dict(tj.get("model", {}).get("vocab", {}))
    for t in tj.get("added_tokens", []):
        vocab.setdefault(t["content"], t["id"])
    return vocab, tj.get("model", {}).get("type")


def check(cand, ref=None):
    """-> (ok, [(level, message)])"""
    out = []
    add = lambda level, msg: out.append((level, msg))      # noqa: E731

    vocab, ttype = vocab_of(cand)
    if vocab is None:
        add("FAIL", "tokenizer.json missing")
    else:
        bad = {t: (vocab.get(t), i) for t, i in REQUIRED.items() if vocab.get(t) != i}
        nl = [t for t, i in NEWLINE_IDS.items() if vocab.get(t) == i]
        if bad:
            add("FAIL", f"token ids differ from arc_solver.ARC_VOCAB (token: got, expected): {bad}")
        if not nl:
            add("FAIL", "no newline token at id 10")
        if not bad and nl:
            add("ok", f"tokenizer: {len(vocab)} tokens, type {ttype}, all hard-coded ids match")
        if len(vocab) > 64:
            add("WARN", f"vocabulary has {len(vocab)} entries -- the kernel expects the shrunk grid vocabulary (16)")

    cp = os.path.join(cand, "config.json")
    cfg = json.load(open(cp)) if os.path.exists(cp) else None
    if cfg is None:
        add("FAIL", "config.json missing")
    else:
        if cfg.get("vocab_size", 0) <= 15:
            add("FAIL", f"vocab_size {cfg.get('vocab_size')} does not cover token id 15")
        if ref:
            rcfg = json.load(open(os.path.join(ref, "config.json")))
            diff = {k: (cfg.get(k), rcfg.get(k)) for k in CFG_KEYS if cfg.get(k) != rcfg.get(k)}
            if diff:
                arch = {k: v for k, v in diff.items() if k in ("model_type", "architectures", "vocab_size")}
                add("FAIL" if arch else "WARN", f"config differs from reference (candidate, reference): {diff}")
            else:
                add("ok", "config: identical to the reference on " + ", ".join(CFG_KEYS))

    w = weights_summary(cand)
    if not w["files"]:
        add("FAIL", "no *.safetensors files")
    else:
        gb = w["bytes"] / 2**30
        add("ok", f"weights: {w['params'] / 1e9:.2f} B params in {w['files']} file(s), {gb:.1f} GiB on disk, dtypes "
                  + ", ".join(f"{k}:{v / 1e9:.2f}B" for k, v in sorted(w["by_dtype"].items())))
        if cfg and w["embed_rows"] is not None and w["embed_rows"] != cfg.get("vocab_size"):
            add("FAIL", f"embed_tokens has {w['embed_rows']} rows but config.vocab_size is {cfg.get('vocab_size')}")
        bf16_gb = w["params"] * 2 / 2**30
        # measured anchor: the 4B reference (~8 GiB bf16) peaks around 10.3-10.7 GiB during TTT on a T4 with
        # gradient checkpointing; production runs WITHOUT checkpointing on a 24 GiB L4 at seq 8192
        if bf16_gb > 12:
            add("FAIL", f"{bf16_gb:.1f} GiB in bf16: TTT at seq 8192 without gradient checkpointing will not fit a 24 GiB L4")
        elif bf16_gb > 9:
            add("WARN", f"{bf16_gb:.1f} GiB in bf16: larger than the 4B reference -- check the kernel log line "
                        "'allocated ...MB for training' on a 4-task run before a full run")
        if set(w["by_dtype"]) - {"BF16"}:
            add("WARN", f"not pure bf16 ({sorted(w['by_dtype'])}); the kernel loads with load_in_4bit=False and casts "
                        "float32 params to bf16 -- expect a larger load transient")
        if ref:
            rw = weights_summary(ref)
            if rw["files"] and rw["params"] != w["params"]:
                add("WARN", f"parameter count {w['params']:,} != reference {rw['params']:,}: "
                            f"per-task time scales by ~{w['params'] / rw['params']:.2f}x")
    ok = not any(level == "FAIL" for level, _ in out)
    return ok, out


def selftest():
    import tempfile

    def make(d, vocab=None, cfg=None, shapes=None, dtype="BF16"):
        os.makedirs(d)
        v = {str(i): i for i in range(10)}
        v.update({"\u010a": 10, "user": 11, "assistant": 12, "<pad>": 13, "<|im_start|>": 14, "<|im_end|>": 15})
        json.dump({"model": {"type": "WordLevel", "vocab": vocab or v}, "added_tokens": []}, open(os.path.join(d, "tokenizer.json"), "w"))
        c = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"], "hidden_size": 2560, "num_hidden_layers": 36,
             "vocab_size": 16, "torch_dtype": "bfloat16"}
        c.update(cfg or {})
        json.dump(c, open(os.path.join(d, "config.json"), "w"))
        sh = shapes or {"model.embed_tokens.weight": [16, 2560], "model.layers.0.w": [2000000, 2000]}
        hdr = json.dumps({k: {"dtype": dtype, "shape": s, "data_offsets": [0, 0]} for k, s in sh.items()}).encode()
        with open(os.path.join(d, "model.safetensors"), "wb") as f:
            f.write(struct.pack("<Q", len(hdr)) + hdr)

    with tempfile.TemporaryDirectory() as t:
        ref = os.path.join(t, "ref")
        make(ref)
        ok, msgs = check(ref, ref)
        assert ok and not [m for lv, m in msgs if lv != "ok"], msgs
        stock = os.path.join(t, "stock")                                   # stock vocabulary: must fail
        make(stock, vocab={**{f"tok{i}": i for i in range(200)}, "0": 15, "user": 872}, cfg={"vocab_size": 151936},
             shapes={"model.embed_tokens.weight": [151936, 2560]})
        ok, msgs = check(stock, ref)
        assert not ok and any("token ids differ" in m for _, m in msgs) and any("vocab_size" in m for _, m in msgs)
        big = os.path.join(t, "big")                                       # 8B: fails the memory gate
        make(big, cfg={"hidden_size": 4096}, shapes={"model.embed_tokens.weight": [16, 4096], "w": [4000000, 2000]})
        ok, msgs = check(big, ref)
        assert not ok and any("will not fit" in m for _, m in msgs) and any("hidden_size" in m for _, m in msgs)
        f32 = os.path.join(t, "f32")
        make(f32, dtype="F32")
        ok, msgs = check(f32, ref)
        assert ok and any("not pure bf16" in m for lv, m in msgs if lv == "WARN")
        mism = os.path.join(t, "mism")
        make(mism, shapes={"model.embed_tokens.weight": [32, 2560]})
        assert not check(mism)[0]
        empty = os.path.join(t, "empty")
        os.makedirs(empty)
        assert not check(empty)[0]
    print("selftest ok: identical dir passes; stock vocabulary, oversized model, embedding/vocab mismatch and an "
          "empty dir fail; non-bf16 weights warn")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--selftest":
        selftest()
        sys.exit(0)
    ok, msgs = check(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
    for level, m in msgs:
        print(f"[{level:4s}] {m}")
    print("DROP-IN OK" if ok else "NOT a drop-in -- do not build a kernel on this checkpoint")
    sys.exit(0 if ok else 1)
