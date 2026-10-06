"""Supervised fine-tuning of the AGI-2 base checkpoint (sft139 lineage) -> a DROP-IN model directory.

    python3 sft_ag2.py --model <base dir> --out <dir> --preset t4|l4|a100 \
        --data tasks:<challenges.json>:<solutions.json> [--data jsonl:<file>] ... \
        [--max-steps N] [--max-hours H] [--stop-after N] [--resume] [--exclude exclude.json] [--full-ft]
    python3 sft_ag2.py --selftest          # CPU: tiny random model, real data path, train/resume/merge/check

What it guarantees (asserted in the selftest):
  * transcripts are produced by the production formatter (arc_loader.QwenFormatter) and encoded with
    the kernel's hard-coded ids (digits 0-9, newline 10, user 11, assistant 12, <|im_start|> 14,
    <|im_end|> 15) -- NOT with the tokenizer, whose WordLevel pre-tokenizer silently drops
    user/assistant under transformers 5.x (runbook: 'Tokenizer dead-TTT bug'). The base directory's
    tokenizer.json is checked against those ids before anything is loaded.
  * loss is completion-only on every assistant grid, exactly like the kernel's TTT collator.
  * any task sharing an (input, output) pair with the 120 evaluation tasks -- up to rotation,
    reflection and recolouring -- is dropped before training (check_sft_data.pair_hash), plus --exclude.
  * the result is base weights + merged LoRA, saved as bf16 safetensors with the base config and
    tokenizer files copied verbatim, and must pass check_swap_model.py against the base.
  * a held-out 5% of TASKS gives val loss before/after; the manifest records both. A run whose val
    loss got worse is saved but flagged (the AGI-3 QLoRA adapter failed exactly this way, unnoticed).

Data note (measured with check_sft_data.py): Nabidnur/arc-agi-2-grids sft128 is the 1,000 PUBLIC
training tasks x 128 augmentations -- the same tasks as arc-agi_training_challenges.json. Passing
tasks:<public train> generates fresh augmentations on the fly, so that 24 MB file adds nothing. The
only extra tasks in that repo are the 410 synthetic ones (jsonl:synthetic/curriculum_v0_verified.jsonl).

Presets. Memory and time are ESTIMATES from the kernel's measured anchors (TTT at r=256 without
checkpointing: 128 steps in 208-828 s on an L4; T4 4-bit run peaked at 10.3-10.7 GB). Nothing here has
been run on a GPU.
  preset  weights      max_len  LoRA r  grad-ckpt  est. peak    est. speed       sequences per 11 h
  t4      fp16+AMP      2048     16      yes        11-13 GB     5-10 s/step      ~4,000-8,000
  l4      bf16          4096     64      yes        14-18 GB     2-4 s/step       ~10,000-20,000
  a100    bf16          8192     64      no         25-35 GB     0.5-1.5 s/step   ~26,000-80,000
  The T4 has no bf16, so its preset runs the frozen base in fp16 under autocast with a GradScaler;
  fp16 can overflow, so non-finite losses are skipped and the run aborts if more than 5% of steps are.
  One free T4 session therefore covers a few thousand sequences, i.e. several augmented passes over
  the 1,000 public tasks but nowhere near 128k. --max-hours stops cleanly and --resume continues.
  --full-ft trains every weight instead of LoRA: A100-80GB class only (bf16 weights + grads + Adam state).
"""
import argparse
import json
import math
import os
import shutil
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_sft_data as csd  # noqa: E402
import check_swap_model  # noqa: E402

USER, ASSISTANT, NL, EOS, PAD = 11, 12, 10, 15, 13
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
PRESETS = {
    "t4":   dict(dtype="fp16", max_len=2048, r=16, grad_ckpt=True, accum=8, lr=2e-5),
    "l4":   dict(dtype="bf16", max_len=4096, r=64, grad_ckpt=True, accum=8, lr=2e-5),
    "a100": dict(dtype="bf16", max_len=8192, r=64, grad_ckpt=False, accum=8, lr=2e-5),
    "cpu":  dict(dtype="fp32", max_len=512, r=4, grad_ckpt=False, accum=2, lr=2e-3),
}


# ----------------------------------------------------------------------- data
def labels_of(ids):
    """Completion-only labels: every assistant grid through its <|im_end|>, as in the kernel's
    QwenDataCollatorForCompletionOnlyLM (start + 2 skips 'assistant' and the newline)."""
    lab = [-100] * len(ids)
    i = 0
    while i < len(ids):
        if ids[i] == ASSISTANT:
            j = i + 2
            while j < len(ids) and ids[j] != EOS:
                j += 1
            for k in range(i + 2, min(j + 1, len(ids))):
                lab[k] = ids[k]
            i = j
        i += 1
    return lab


def augment(pairs, rng):
    """One of the 8 joint dihedral views, a full colour permutation, shuffled demo order."""
    k, t = int(rng.integers(0, 4)), bool(rng.integers(0, 2))
    perm = rng.permutation(10)
    out = []
    for a, b in pairs:
        a, b = np.asarray(a), np.asarray(b)
        if t:
            a, b = a.T, b.T
        out.append((perm[np.rot90(a, k)].tolist(), perm[np.rot90(b, k)].tolist()))
    order = rng.permutation(len(out))
    return [out[i] for i in order]


def sequence(pairs, fmt, max_len):
    """pairs -> (ids, labels), dropping leading demos until it fits (the kernel's cut_to_len rule)."""
    while pairs:
        ids = csd.encode(csd.to_text(pairs, fmt))
        if len(ids) <= max_len:
            return ids, labels_of(ids)
        pairs = pairs[1:]
    return None, None


def load_tasks(specs, exclude=None, eval_ref=None, log=print):
    """-> (train_tasks, val_tasks): {key: (pairs, pre_augmented)}; contaminated tasks dropped."""
    ev = csd.eval_hashes(*(eval_ref or (None, None)))
    exclude = exclude or {}
    banned_all = {t for v in exclude.values() for t in v}
    groups, dropped = {}, set()
    for spec in specs:
        pre = None
        for n, (tid, pairs, text) in enumerate(csd.iter_source(spec)):
            if pairs is None or not pairs:
                continue
            if pre is None:
                pre = text is not None
            if tid in banned_all or any(csd.pair_hash(i, o) in ev for i, o in pairs):
                dropped.add(tid)
                continue
            groups.setdefault(tid, []).append((pairs, pre))
    if dropped:
        log(f"dropped {len(dropped)} task(s) that overlap the evaluation set or the exclude list: {sorted(dropped)[:6]}")
    ids = sorted(groups)
    if len(ids) < 2:
        raise SystemExit("need at least 2 clean tasks")
    val_ids = set(sorted(ids, key=lambda t: csd.hashlib.sha1(t.encode()).hexdigest())[:max(1, min(50, len(ids) // 20))])
    train = {t: groups[t] for t in ids if t not in val_ids}
    val = {t: groups[t] for t in ids if t in val_ids}
    log(f"data: {len(train)} train tasks, {len(val)} held-out tasks, "
        f"{sum(len(v) for v in train.values())} stored sequences/tasks before augmentation")
    return train, val


def train_example(train, step, seed, fmt, max_len):
    """Deterministic in (seed, step): a resumed run sees exactly the sequences it would have seen."""
    rng = np.random.default_rng([seed, step])
    keys = sorted(train)
    for _ in range(20):
        variants = train[keys[int(rng.integers(0, len(keys)))]]
        pairs, pre = variants[int(rng.integers(0, len(variants)))]
        ids, lab = sequence(list(pairs) if pre else augment(pairs, rng), fmt, max_len)
        if ids is not None:
            return ids, lab
    raise SystemExit("could not build a sequence that fits max_len")


# ---------------------------------------------------------------------- model
def check_base(model_dir):
    vocab, _ = check_swap_model.vocab_of(model_dir)
    if vocab is None:
        raise SystemExit(f"{model_dir}: tokenizer.json missing")
    want = dict(check_swap_model.REQUIRED)
    bad = {t: (vocab.get(t), i) for t, i in want.items() if vocab.get(t) != i}
    if bad or not any(vocab.get(t) == 10 for t in check_swap_model.NEWLINE_IDS):
        raise SystemExit(f"{model_dir}: tokenizer ids differ from the kernel's hard-coded vocabulary: {bad}")
    if vocab.get("<|im_start|>", 14) != 14:
        raise SystemExit(f"{model_dir}: <|im_start|> is not id 14")


def load_model(model_dir, cfg, full_ft, torch):
    from transformers import AutoModelForCausalLM
    dt = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[cfg["dtype"]]
    model = AutoModelForCausalLM.from_pretrained(model_dir, torch_dtype=dt)
    model.config.use_cache = False
    if full_ft:
        for p in model.parameters():
            p.requires_grad_(True)
    else:
        from peft import LoraConfig, get_peft_model
        model = get_peft_model(model, LoraConfig(r=cfg["r"], lora_alpha=cfg["r"], lora_dropout=0.0, bias="none",
                                                 target_modules=LORA_TARGETS, task_type="CAUSAL_LM"))
        for n, p in model.named_parameters():                      # adapters in fp32 even on an fp16/bf16 base
            if p.requires_grad:
                p.data = p.data.float()
    if cfg["grad_ckpt"]:
        model.gradient_checkpointing_enable()
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()                     # frozen embeddings + checkpointing need this
    return model


def loss_of(model, ids, lab, torch, device, amp):
    x = torch.tensor([ids], dtype=torch.long, device=device)
    y = torch.tensor([lab], dtype=torch.long, device=device)
    with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp):
        logits = model(input_ids=x).logits
    return torch.nn.functional.cross_entropy(logits[0, :-1].float(), y[0, 1:], ignore_index=-100)


def val_loss(model, val, fmt, max_len, torch, device, amp):
    model.eval()
    tot, n = 0.0, 0
    with torch.no_grad():
        for t in sorted(val):
            ids, lab = sequence(list(val[t][0][0]), fmt, max_len)
            if ids is None:
                continue
            tot += float(loss_of(model, ids, lab, torch, device, amp))
            n += 1
    model.train()
    return tot / max(n, 1)


def trainable_state(model):
    return {k: v.detach().cpu() for k, v in model.named_parameters() if v.requires_grad}


def save_merged(model, base_dir, out_dir, full_ft, torch):
    merged = model if full_ft else model.merge_and_unload()
    merged = merged.to(torch.bfloat16)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    merged.save_pretrained(out_dir, safe_serialization=True)
    for f in os.listdir(base_dir):                                 # config + tokenizer files verbatim from the base
        src = os.path.join(base_dir, f)
        if os.path.isfile(src) and not f.endswith((".safetensors", ".bin", ".pt")) and f != "model.safetensors.index.json":
            shutil.copy2(src, os.path.join(out_dir, f))
    return check_swap_model.check(out_dir, base_dir)


# ----------------------------------------------------------------------- train
def train(a, log=print):
    import torch
    cfg = dict(PRESETS[a.preset])
    for k in ("max_len", "r", "lr", "accum"):
        if getattr(a, k, None) is not None:
            cfg[k] = getattr(a, k)
    check_base(a.model)
    fmt = csd.formatter()
    exclude = json.load(open(a.exclude)) if a.exclude else None
    tr, va = load_tasks(a.data, exclude, a.eval_ref, log)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg["dtype"] == "fp16" and device.type == "cuda"
    torch.manual_seed(a.seed)
    model = load_model(a.model, cfg, a.full_ft, torch).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"preset {a.preset}: {cfg}; trainable params {sum(p.numel() for p in params):,}; device {device}")
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    total_updates = max(1, a.max_steps // cfg["accum"])
    warm = max(1, int(0.03 * total_updates))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda u: min(1.0, (u + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(u, total_updates) / total_updates)))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    os.makedirs(a.out, exist_ok=True)
    ck = os.path.join(a.out, "ckpt.pt")
    step, bad, hist, v0 = 0, 0, [], None
    if a.resume and os.path.exists(ck):
        st = torch.load(ck, map_location="cpu", weights_only=False)
        if st["cfg"] != cfg or st["max_steps"] != a.max_steps or st["full_ft"] != a.full_ft:
            raise SystemExit("checkpoint was made with different settings; use the same flags or a new --out")
        own = dict(model.named_parameters())
        for k, v in st["params"].items():
            own[k].data.copy_(v.to(own[k].dtype))
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, bad, hist, v0 = st["step"], st["bad"], st["hist"], st["v0"]
        log(f"resumed at step {step}/{a.max_steps}")
    if v0 is None:
        v0 = val_loss(model, va, fmt, cfg["max_len"], torch, device, amp)
        log(f"val loss before training: {v0:.4f}")

    def checkpoint():
        tmp = ck + ".tmp"
        torch.save({"params": trainable_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "bad": bad, "hist": hist, "v0": v0, "cfg": cfg, "max_steps": a.max_steps,
                    "full_ft": a.full_ft}, tmp)
        os.replace(tmp, ck)

    t0, stopped, step0 = time.time(), False, step
    model.train()
    opt.zero_grad(set_to_none=True)
    while step < a.max_steps:
        ids, lab = train_example(tr, step, a.seed, fmt, cfg["max_len"])
        loss = loss_of(model, ids, lab, torch, device, amp)
        if not torch.isfinite(loss):
            bad += 1
            opt.zero_grad(set_to_none=True)
            if bad > max(5, 0.05 * (step + 1)):
                raise SystemExit(f"{bad} non-finite losses in {step + 1} steps -- fp16 is overflowing; use a bf16 GPU")
        else:
            scaler.scale(loss / cfg["accum"]).backward()
            hist.append(float(loss.detach()))
        step += 1
        if step % cfg["accum"] == 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            sched.step()
        if step % a.log_every == 0:
            log(f"step {step}/{a.max_steps}  loss(last {a.log_every}) {np.mean(hist[-a.log_every:]):.4f}  "
                f"{(time.time() - t0) / max(step - step0, 1):.2f} s/step")
        boundary = step % cfg["accum"] == 0
        if boundary and (step % a.save_every == 0 or step == a.max_steps):
            checkpoint()
        out_of_time = a.max_hours and time.time() - t0 > a.max_hours * 3600
        if boundary and step < a.max_steps and (out_of_time or (a.stop_after and step - step0 >= a.stop_after)):
            checkpoint()
            stopped = True
            log(f"stopping at step {step} ({'time budget' if out_of_time else '--stop-after'}); "
                "checkpoint saved -- rerun with --resume")
            break
    if stopped:
        return {"finished": False, "step": step}

    v1 = val_loss(model, va, fmt, cfg["max_len"], torch, device, amp)
    k = max(1, len(hist) // 10)
    ok, msgs = save_merged(model, a.model, os.path.join(a.out, "merged"), a.full_ft, torch)
    manifest = {"base": os.path.abspath(a.model), "data": a.data, "preset": a.preset, "cfg": cfg, "steps": step,
                "updates": step // cfg["accum"], "non_finite_steps": bad, "full_ft": a.full_ft, "seed": a.seed,
                "train_loss_first": float(np.mean(hist[:k])), "train_loss_last": float(np.mean(hist[-k:])),
                "val_loss_before": v0, "val_loss_after": v1, "val_improved": v1 < v0,
                "train_tasks": len(tr), "val_tasks": len(va), "drop_in_ok": ok,
                "drop_in_messages": [f"[{lv}] {m}" for lv, m in msgs]}
    json.dump(manifest, open(os.path.join(a.out, "manifest.json"), "w"), indent=1)
    log(f"train loss {manifest['train_loss_first']:.4f} -> {manifest['train_loss_last']:.4f}; "
        f"val loss {v0:.4f} -> {v1:.4f} ({'improved' if v1 < v0 else 'WORSE -- do not ship without a panel run'})")
    for lv, m in msgs:
        log(f"[{lv:4s}] {m}")
    log(("DROP-IN OK: " if ok else "NOT a drop-in: ") + os.path.join(a.out, "merged"))
    return {"finished": True, "manifest": manifest}


def parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model")
    ap.add_argument("--out")
    ap.add_argument("--data", action="append")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="l4")
    ap.add_argument("--max-steps", type=int, default=4000, dest="max_steps")
    ap.add_argument("--max-hours", type=float, default=0.0, dest="max_hours")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--stop-after", type=int, default=0, dest="stop_after",
                    help="stop (with a checkpoint) after this many steps in this session")
    ap.add_argument("--exclude")
    ap.add_argument("--full-ft", action="store_true", dest="full_ft")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--save-every", type=int, default=400, dest="save_every")
    ap.add_argument("--log-every", type=int, default=50, dest="log_every")
    ap.add_argument("--max-len", type=int, dest="max_len")
    ap.add_argument("--r", type=int)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--accum", type=int)
    ap.add_argument("--selftest", action="store_true")
    ap.set_defaults(eval_ref=None)
    return ap


# -------------------------------------------------------------------- selftest
def _tiny_base(d, torch):
    """A random 2-layer model with the kernel's 16-token vocabulary and Qwen module names."""
    from transformers import Qwen2Config, Qwen2ForCausalLM
    torch.manual_seed(0)
    cfg = Qwen2Config(vocab_size=16, hidden_size=48, intermediate_size=96, num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=1024, pad_token_id=PAD, eos_token_id=EOS,
                      bos_token_id=None, tie_word_embeddings=False)
    Qwen2ForCausalLM(cfg).to(torch.bfloat16).save_pretrained(d, safe_serialization=True)
    vocab = {str(i): i for i in range(10)}
    vocab.update({"\u010a": 10, "user": 11, "assistant": 12, "<pad>": 13, "<|im_start|>": 14, "<|im_end|>": 15})
    json.dump({"model": {"type": "WordLevel", "vocab": vocab}, "added_tokens": []}, open(os.path.join(d, "tokenizer.json"), "w"))
    json.dump({"note": "copied verbatim"}, open(os.path.join(d, "tokenizer_config.json"), "w"))


def selftest():
    import tempfile
    import torch
    from safetensors.torch import load_file
    # labels: only assistant grids + their EOS
    text = "<|im_start|>user\n01<|im_end|><|im_start|>assistant\n23\n45<|im_end|><|im_start|>user\n6<|im_end|><|im_start|>assistant\n7<|im_end|>"
    ids = csd.encode(text)
    lab = labels_of(ids)
    assert [x for x in lab if x != -100] == [2, 3, 10, 4, 5, 15, 7, 15]
    assert all(l in (-100, i) for l, i in zip(lab, ids)) and lab[ids.index(ASSISTANT) + 1] == -100
    # augmentation keeps the task: same contamination hash for every pair, challenge order shuffled
    rng = np.random.default_rng(0)
    pairs = [([[1, 2, 0], [0, 2, 2]], [[2, 2], [1, 0]]), ([[3, 3], [0, 1]], [[1]]), ([[4]], [[4, 4]])]
    seen = set()
    for _ in range(40):
        aug = augment(pairs, rng)
        assert sorted(csd.pair_hash(i, o) for i, o in aug) == sorted(csd.pair_hash(i, o) for i, o in pairs)
        seen.add(csd.to_text(aug))
    assert len(seen) > 30
    fmt = csd.formatter()
    ids, lab = sequence(pairs, fmt, 10_000)
    assert csd.parse_text(csd.to_text(pairs, fmt)) == pairs and len(ids) == len(lab)
    short, _ = sequence(pairs, fmt, len(ids) - 1)                     # one token too few -> first demo dropped
    assert short == csd.encode(csd.to_text(pairs[1:], fmt)) and sequence(pairs, fmt, 3) == (None, None)

    base_j = os.path.dirname(HERE)
    tr_all = json.load(open(os.path.join(base_j, "arc-agi_training_challenges.json")))
    so_all = json.load(open(os.path.join(base_j, "arc-agi_training_solutions.json")))
    ev = json.load(open(os.path.join(base_j, "arc-agi_evaluation_challenges.json")))
    small = [k for k, t in tr_all.items() if sum(np.asarray(p["input"]).size + np.asarray(p["output"]).size
                                                 for p in t["train"]) < 220][:40]
    with tempfile.TemporaryDirectory() as d:
        base = os.path.join(d, "base")
        os.makedirs(base)
        _tiny_base(base, torch)
        leak = sorted(ev)[0]                                           # an eval task disguised by rotation
        tasks = {k: tr_all[k] for k in small}
        tasks["leak0001"] = {"train": [{"input": np.rot90(np.asarray(p["input"])).tolist(),
                                        "output": np.rot90(np.asarray(p["output"])).tolist()} for p in ev[leak]["train"]],
                             "test": []}
        cj, sj = os.path.join(d, "c.json"), os.path.join(d, "s.json")
        json.dump(tasks, open(cj, "w"))
        json.dump({k: so_all[k] for k in small}, open(sj, "w"))
        spec = f"tasks:{cj}:{sj}"
        logs = []
        tr, va = load_tasks([spec], log=logs.append)
        assert "leak0001" not in tr and "leak0001" not in va and any("leak0001" in m for m in logs)
        assert len(tr) + len(va) == len(small) and not set(tr) & set(va)
        a1, b1 = train_example(tr, 7, 1, fmt, 512), train_example(tr, 7, 1, fmt, 512)
        assert a1 == b1 and a1 != train_example(tr, 8, 1, fmt, 512)

        def run(out, steps_first=None, **kw):
            args = parser().parse_args(["--model", base, "--out", out, "--data", spec, "--preset", "cpu",
                                        "--max-steps", "120", "--save-every", "20", "--log-every", "1000"])
            for k, v in kw.items():
                setattr(args, k, v)
            if steps_first is None:
                return train(args, log=lambda *_: None)
            args.stop_after = steps_first                           # a session ending mid-run ...
            r = train(args, log=lambda *_: None)
            assert not r["finished"] and r["step"] == steps_first, r
            args.stop_after, args.resume = 0, True                  # ... and the next session
            return train(args, log=lambda *_: None)

        r1 = run(os.path.join(d, "o1"))
        m = r1["manifest"]
        assert m["drop_in_ok"] and m["updates"] == 60 and m["non_finite_steps"] == 0, m
        assert m["train_loss_last"] < m["train_loss_first"] - 0.05, m           # it learns (r=4 on a random model)
        assert m["val_loss_after"] < m["val_loss_before"] and m["val_improved"], m
        merged = os.path.join(d, "o1", "merged")
        ok, msgs = check_swap_model.check(merged, base)
        assert ok and not [x for lv, x in msgs if lv != "ok"], msgs
        for f in ("config.json", "tokenizer.json", "tokenizer_config.json"):
            assert open(os.path.join(merged, f), "rb").read() == open(os.path.join(base, f), "rb").read(), f
        w0, w1 = load_file(os.path.join(base, "model.safetensors")), load_file(os.path.join(merged, "model.safetensors"))
        assert w0.keys() == w1.keys() and all(w1[k].dtype == torch.bfloat16 for k in w1)
        changed = [k for k in w0 if not torch.equal(w0[k], w1[k])]
        assert changed and all(any(t in k for t in LORA_TARGETS) for k in changed), changed[:3]
        assert torch.equal(w0["model.embed_tokens.weight"], w1["model.embed_tokens.weight"])
        # interrupted + resumed run == uninterrupted run
        r2 = run(os.path.join(d, "o2"), steps_first=50)
        w2 = load_file(os.path.join(d, "o2", "merged", "model.safetensors"))
        assert r2["finished"] and all(torch.equal(w1[k], w2[k]) for k in w1), "resume changed the result"
        try:
            run(os.path.join(d, "o2"), resume=True, max_steps=200)
            raise SystemExit("resume with different settings accepted")
        except SystemExit as e:
            assert "different settings" in str(e)
        # full fine-tuning path
        r3 = run(os.path.join(d, "o3"), full_ft=True, lr=3e-4)
        assert r3["manifest"]["drop_in_ok"] and r3["manifest"]["train_loss_last"] < r3["manifest"]["train_loss_first"] - 0.2
        # a base with the wrong vocabulary is refused before any weight is loaded
        bad = os.path.join(d, "bad")
        shutil.copytree(base, bad)
        json.dump({"model": {"type": "BPE", "vocab": {"0": 15, "user": 872}}, "added_tokens": []},
                  open(os.path.join(bad, "tokenizer.json"), "w"))
        try:
            check_base(bad)
            raise SystemExit("wrong vocabulary accepted")
        except SystemExit as e:
            assert "hard-coded vocabulary" in str(e)
    print(f"selftest ok (torch {torch.__version__}, CPU, tiny random model): eval-overlapping task dropped; "
          f"train loss {m['train_loss_first']:.2f} -> {m['train_loss_last']:.2f}, held-out loss "
          f"{m['val_loss_before']:.2f} -> {m['val_loss_after']:.2f}; merged dir passes check_swap_model with only "
          "LoRA-target weights changed; interrupted+resumed run is bit-identical; full-FT path works")


if __name__ == "__main__":
    a = parser().parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    if not (a.model and a.out and a.data):
        sys.exit(__doc__)
    train(a)
