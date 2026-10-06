"""Supervised fine-tuning of the AGI-2 base checkpoint (sft139 lineage) -> a DROP-IN model directory.

    python3 sft_ag2.py --model <base dir> --out <dir> --preset t4|l4|a100 \
        --data tasks:<challenges.json>:<solutions.json> [--data jsonl:<file>[@share]] ... [--val-from first] \
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
  The T4 has no bf16, so its preset keeps the frozen base in fp16 under autocast, with fp32 LoRA weights, a
  GradScaler and gradient clipping at 1.0. What fp16 cannot do is hold an activation above 65504, and no
  optimiser setting fixes that, so the run starts with a probe (--fp16-guard auto, default): the first
  --fp16-probe training sequences are forwarded and every decoder layer's peak |activation| is measured. If a
  layer is non-finite or above half the fp16 range, that layer and everything after it (final norm, lm_head)
  compute in fp32 from then on -- weights stay fp16, the residual stream is promoted once and never cast
  back. A non-finite loss later in the run widens the fp32 region instead of being skipped. The cost is
  speed for the widened layers (fp32 has no tensor-core path on a T4), not memory. --fp16-guard full does it
  for all layers; the decision is stored in the checkpoint so --resume continues identically.
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
    "cpu16": dict(dtype="fp16", max_len=512, r=4, grad_ckpt=False, accum=2, lr=2e-3),      # selftest of the fp16 path
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


def split_spec(spec):
    """'jsonl:file@0.3' -> ('jsonl:file', 0.3); no '@' -> weight None (uniform over tasks)."""
    head, sep, tail = spec.rpartition("@")
    if sep:
        try:
            return head, float(tail)
        except ValueError:
            pass
    return spec, None


def load_tasks(specs, exclude=None, eval_ref=None, log=print, val_from="all", with_sources=False):
    """-> (train_tasks, val_tasks): {key: (pairs, pre_augmented)}; contaminated tasks dropped.
    val_from='first' holds out tasks of the first source only (the real tasks, not the synthetic ones).
    with_sources=True also returns {task id: index of the first spec that contains it}."""
    ev = csd.eval_hashes(*(eval_ref or (None, None)))
    exclude = exclude or {}
    banned_all = {t for v in exclude.values() for t in v}
    groups, dropped = {}, set()
    source = {}
    for si, spec in enumerate(specs):
        pre = None
        for n, (tid, pairs, text) in enumerate(csd.iter_source(split_spec(spec)[0])):
            if pairs is None or not pairs:
                continue
            if pre is None:
                pre = text is not None
            if tid in banned_all or any(csd.pair_hash(i, o) in ev for i, o in pairs):
                dropped.add(tid)
                continue
            groups.setdefault(tid, []).append((pairs, pre))
            source.setdefault(tid, si)
    if dropped:
        log(f"dropped {len(dropped)} task(s) that overlap the evaluation set or the exclude list: {sorted(dropped)[:6]}")
    ids = sorted(groups)
    if len(ids) < 2:
        raise SystemExit("need at least 2 clean tasks")
    pool = [t for t in ids if source[t] == 0] if val_from == "first" else ids
    val_ids = set(sorted(pool, key=lambda t: csd.hashlib.sha1(t.encode()).hexdigest())[:max(1, min(50, len(pool) // 20))])
    train = {t: groups[t] for t in ids if t not in val_ids}
    val = {t: groups[t] for t in ids if t in val_ids}
    log(f"data: {len(train)} train tasks, {len(val)} held-out tasks, "
        f"{sum(len(v) for v in train.values())} stored sequences/tasks before augmentation")
    if with_sources:
        return train, val, source
    return train, val


def make_mix(specs, train, source):
    """Sampling shares per source from the '@w' suffixes -> [(cumulative share, sorted task ids)] or None.
    Sources without a weight split what the weighted ones leave, in proportion to their task counts."""
    w = [split_spec(sp)[1] for sp in specs]
    if all(x is None for x in w):
        return None
    keys = [sorted(t for t in train if source[t] == i) for i in range(len(specs))]
    given = sum(x for x in w if x is not None)
    if given > 1.0 + 1e-9 or any(x is not None and x < 0 for x in w):
        raise SystemExit("source weights must be >= 0 and sum to at most 1")
    free = sum(len(k) for k, x in zip(keys, w) if x is None)
    share = [x if x is not None else (1.0 - given) * len(k) / max(free, 1) for k, x in zip(keys, w)]
    share = [sh if k else 0.0 for sh, k in zip(share, keys)]
    tot = sum(share)
    if tot <= 0:
        raise SystemExit("no source has both tasks and a positive weight")
    out, acc = [], 0.0
    for sh, k in zip(share, keys):
        if sh > 0:
            acc += sh / tot
            out.append((acc, k))
    return out


def train_example(train, step, seed, fmt, max_len, mix=None):
    """Deterministic in (seed, step): a resumed run sees exactly the sequences it would have seen."""
    rng = np.random.default_rng([seed, step])
    keys = sorted(train)
    for _ in range(20):
        if mix:
            u = float(rng.random())
            keys = next((k for acc, k in mix if u < acc), mix[-1][1])
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


# ------------------------------------------------------------------ fp16 guard
FP16_MAX = 65504.0


def decoder_layers(model, torch):
    cands = [m for n, m in model.named_modules() if n.split(".")[-1] == "layers" and isinstance(m, torch.nn.ModuleList)]
    return max(cands, key=len)


def fp16_probe(model, examples, torch, device, amp, margin=0.5):
    """Forward the examples in the training precision and watch every decoder layer.
    -> {"first_bad": layer index or None, "layers": [max |activation| per layer], "loss_finite": bool}
    A layer is bad when its output or any Linear inside it is non-finite or, while it still computes in
    fp16, exceeds margin * 65504."""
    layers = decoder_layers(model, torch)
    peak, hooks = [0.0] * len(layers), []

    def watch(i):
        def hook(_m, _inp, out):
            t = out[0] if isinstance(out, (tuple, list)) else out
            if torch.is_tensor(t) and t.is_floating_point():
                v = t.detach().float().abs().max()
                v = float("inf") if not bool(torch.isfinite(v)) else float(v)
                peak[i] = max(peak[i], v)
        return hook
    for i, layer in enumerate(layers):
        hooks.append(layer.register_forward_hook(watch(i)))
        for m in layer.modules():
            if isinstance(m, torch.nn.Linear) and m.weight.dtype == torch.float16:
                hooks.append(m.register_forward_hook(watch(i)))
    finite = True
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():
            for ids, lab in examples:
                finite = finite and bool(torch.isfinite(loss_of(model, ids, lab, torch, device, amp)))
    finally:
        for h in hooks:
            h.remove()
        model.train(was_training)
    # layers already computing in fp32 may legitimately exceed the fp16 range; only non-finite counts there
    bad = [i for i, v in enumerate(peak)
           if not (math.isfinite(v) if getattr(layers[i], "_fp32_scope", False) else v < margin * FP16_MAX)]
    return {"first_bad": bad[0] if bad else (0 if not finite else None), "layers": peak, "loss_finite": finite}


def fp32_from(model, k, torch, device):
    """Run decoder layers >= k, the final norm and lm_head in fp32 arithmetic while the weights stay fp16.
    The residual stream is promoted once, at layer k, and never cast back, so nothing can overflow after it.
    Costs speed (no fp16 tensor cores for those layers), no extra weight memory beyond one layer's transient copy."""
    F = torch.nn.functional
    layers = decoder_layers(model, torch)

    def cast(x):
        if torch.is_tensor(x):
            return x.float() if x.dtype == torch.float16 else x
        if isinstance(x, (tuple, list)):
            return type(x)(cast(v) for v in x)
        return x

    def wide_linear(m):
        if getattr(m, "_fp32_compute", False):
            return
        m._fp32_compute = True
        m.forward = lambda x, m=m: F.linear(x, m.weight.to(x.dtype), None if m.bias is None else m.bias.to(x.dtype))

    def wide_module(mod):
        if getattr(mod, "_fp32_scope", False):
            return
        mod._fp32_scope = True
        orig = mod.forward

        def forward(*a, **kw):
            with torch.autocast(device_type=device.type, enabled=False):
                return orig(*[cast(v) for v in a], **{n: cast(v) for n, v in kw.items()})
        mod.forward = forward
        for m in mod.modules():
            if isinstance(m, torch.nn.Linear):
                wide_linear(m)
    tail = [m for n, m in model.named_modules() if n.split(".")[-1] in ("norm", "lm_head") and "layers." not in n]
    for mod in list(layers[k:]) + tail:
        wide_module(mod)
    return len(layers) - k


def fp16_guard(model, examples, torch, device, amp, mode="auto", log=print):
    """Decide where fp16 arithmetic is safe for THIS checkpoint on THIS data. -> record for the manifest."""
    if mode == "off":
        return {"mode": "off", "fp32_from": None}
    if mode == "full":
        n = fp32_from(model, 0, torch, device)
        return {"mode": "full", "fp32_from": 0, "fp32_layers": n}
    pr = fp16_probe(model, examples, torch, device, amp)
    rec = {"mode": "auto", "fp32_from": None, "peak_fp16": max(pr["layers"]), "probe_examples": len(examples)}
    k = pr["first_bad"]
    tries = 0
    while k is not None:
        tries += 1
        n = fp32_from(model, k, torch, device)
        rec.update(fp32_from=k, fp32_layers=n)
        log(f"fp16 guard: layer {k} reaches {pr['layers'][k]:.3g} (fp16 max 65504) -> layers {k}.. , norm and lm_head "
            f"now compute in fp32 ({n} layers)")
        pr = fp16_probe(model, examples, torch, device, amp)
        if pr["first_bad"] is None:
            break
        if k == 0 or tries > 3:
            raise SystemExit("fp16 guard: activations are non-finite even with fp32 arithmetic everywhere -- "
                             "the checkpoint or the data is broken, not the precision")
        k = min(pr["first_bad"], k - 1) if pr["first_bad"] < k else 0
    if rec["fp32_from"] is None:
        log(f"fp16 guard: all layers stay below {0.5 * FP16_MAX:.0f} (peak {rec['peak_fp16']:.3g}) -- pure fp16")
    return rec


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
    tr, va, src = load_tasks(a.data, exclude, a.eval_ref, log, a.val_from, with_sources=True)
    mix = make_mix(a.data, tr, src)
    if mix:
        prev = 0.0
        for (acc, k), sp in zip(mix, [sp for sp in a.data]):
            log(f"mix: {acc - prev:.0%} of the sequences from {len(k)} tasks")
            prev = acc
    cfg["mix"] = [split_spec(sp)[1] for sp in a.data] if mix else None
    cfg["val_from"] = a.val_from
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = cfg["dtype"] == "fp16"
    torch.manual_seed(a.seed)
    model = load_model(a.model, cfg, a.full_ft, torch).to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    log(f"preset {a.preset}: {cfg}; trainable params {sum(p.numel() for p in params):,}; device {device}")
    opt = torch.optim.AdamW(params, lr=cfg["lr"], weight_decay=0.0)
    total_updates = max(1, a.max_steps // cfg["accum"])
    warm = max(1, int(0.03 * total_updates))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda u: min(1.0, (u + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(u, total_updates) / total_updates)))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)
    os.makedirs(a.out, exist_ok=True)
    ck = os.path.join(a.out, "ckpt.pt")
    step, bad, hist, v0 = 0, 0, [], None
    guard, skipped_updates = {"mode": "off", "fp32_from": None}, 0
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
        guard = st.get("guard") or {"mode": "off", "fp32_from": None}
        if guard.get("fp32_from") is not None:
            fp32_from(model, guard["fp32_from"], torch, device)
        skipped_updates = st.get("skipped_updates", 0)
        log(f"resumed at step {step}/{a.max_steps}")
    elif amp:
        probe = [train_example(tr, i, a.seed, fmt, cfg["max_len"], mix) for i in range(a.fp16_probe)]
        guard = fp16_guard(model, probe, torch, device, amp, a.fp16_guard, log)
    if v0 is None:
        v0 = val_loss(model, va, fmt, cfg["max_len"], torch, device, amp)
        log(f"val loss before training: {v0:.4f}")

    def checkpoint():
        tmp = ck + ".tmp"
        torch.save({"params": trainable_state(model), "opt": opt.state_dict(), "sched": sched.state_dict(),
                    "step": step, "bad": bad, "hist": hist, "v0": v0, "cfg": cfg, "max_steps": a.max_steps,
                    "full_ft": a.full_ft, "guard": guard, "skipped_updates": skipped_updates}, tmp)
        os.replace(tmp, ck)

    t0, stopped, step0 = time.time(), False, step
    model.train()
    opt.zero_grad(set_to_none=True)
    while step < a.max_steps:
        ids, lab = train_example(tr, step, a.seed, fmt, cfg["max_len"], mix)
        loss = loss_of(model, ids, lab, torch, device, amp)
        if not torch.isfinite(loss):
            bad += 1
            opt.zero_grad(set_to_none=True)
            if amp and a.fp16_guard == "auto" and guard.get("fp32_from") != 0:
                k = 0 if guard.get("fp32_from") is None else max(0, guard["fp32_from"] // 2)
                n = fp32_from(model, k, torch, device)
                guard.update(fp32_from=k, fp32_layers=n, widened_at_step=step)
                log(f"step {step}: non-finite loss -> fp16 guard widened to layers {k}.. ({n} layers in fp32)")
            elif bad > max(5, 0.05 * (step + 1)):
                raise SystemExit(f"{bad} non-finite losses in {step + 1} steps with fp32 arithmetic in the flagged "
                                 "layers -- stop: lower --lr or inspect the data, precision is not the cause")
        else:
            scaler.scale(loss / cfg["accum"]).backward()
            hist.append(float(loss.detach()))
        step += 1
        if step % cfg["accum"] == 0:
            scaler.unscale_(opt)
            gnorm = torch.nn.utils.clip_grad_norm_(params, 1.0)
            if not bool(torch.isfinite(gnorm)):
                skipped_updates += 1                               # GradScaler skips the step and lowers its scale
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
                "fp16_guard": guard, "skipped_updates": skipped_updates,
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
    ap.add_argument("--val-from", choices=["all", "first"], default="all", dest="val_from",
                    help="first = hold out tasks of the first --data source only")
    ap.add_argument("--fp16-guard", choices=["auto", "off", "full"], default="auto", dest="fp16_guard",
                    help="fp16 presets only: auto = probe, then compute overflowing layers in fp32; full = all layers")
    ap.add_argument("--fp16-probe", type=int, default=16, dest="fp16_probe", help="training examples used by the probe")
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
        # ---- fp16 path (CPU autocast): healthy base stays pure fp16 and learns like the fp32 run
        r4 = run(os.path.join(d, "o4"), preset="cpu16")
        m4 = r4["manifest"]
        assert m4["fp16_guard"]["mode"] == "auto" and m4["fp16_guard"]["fp32_from"] is None, m4["fp16_guard"]
        assert m4["non_finite_steps"] == 0 and m4["skipped_updates"] == 0 and m4["drop_in_ok"], m4
        assert m4["train_loss_last"] < m4["train_loss_first"] - 0.05 and m4["val_improved"], m4
        assert abs(m4["val_loss_before"] - m["val_loss_before"]) < 0.02, (m4["val_loss_before"], m["val_loss_before"])
        # ---- a base whose second layer overflows fp16: the probe finds it, fp32 arithmetic from there is exact
        from safetensors.torch import save_file
        hot = os.path.join(d, "hot")
        shutil.copytree(base, hot)
        w = load_file(os.path.join(hot, "model.safetensors"))
        w["model.layers.1.mlp.down_proj.weight"] = (w["model.layers.1.mlp.down_proj.weight"].float() * 3e5).to(torch.bfloat16)
        w["model.layers.1.mlp.up_proj.weight"] = (w["model.layers.1.mlp.up_proj.weight"].float() * 300).to(torch.bfloat16)
        save_file(w, os.path.join(hot, "model.safetensors"), metadata={"format": "pt"})
        dev = torch.device("cpu")
        ex = [train_example(tr, i, 1, fmt, 512) for i in range(6)]
        c16, c32 = dict(PRESETS["cpu16"]), dict(PRESETS["cpu"])
        torch.manual_seed(0)
        m16 = load_model(hot, c16, False, torch)
        pr = fp16_probe(m16, ex, torch, dev, True)
        assert pr["first_bad"] == 1 and not pr["loss_finite"] and pr["layers"][0] < 100, pr
        notes = []
        rec = fp16_guard(m16, ex, torch, dev, True, "auto", notes.append)
        assert rec["fp32_from"] == 1 and rec["fp32_layers"] == 1 and any("layer 1" in x for x in notes), rec
        torch.manual_seed(0)
        m32 = load_model(hot, c32, False, torch)
        l16 = [float(loss_of(m16, i, l, torch, dev, True)) for i, l in ex]
        l32 = [float(loss_of(m32, i, l, torch, dev, False)) for i, l in ex]
        assert all(math.isfinite(x) for x in l16) and max(abs(x - y) / max(1.0, abs(y)) for x, y in zip(l16, l32)) < 0.02, (l16, l32)
        torch.manual_seed(0)
        off = load_model(hot, c16, False, torch)
        assert fp16_guard(off, ex, torch, dev, True, "off")["fp32_from"] is None
        assert not math.isfinite(float(loss_of(off, ex[0][0], ex[0][1], torch, dev, True)))       # what the guard prevents
        torch.manual_seed(0)
        full = load_model(hot, c16, False, torch)
        assert fp16_guard(full, ex, torch, dev, True, "full")["fp32_layers"] == 2
        assert abs(float(loss_of(full, ex[0][0], ex[0][1], torch, dev, True)) - l32[0]) / max(1.0, abs(l32[0])) < 0.02
        del m16, m32, off, full
        # training on it: guarded from the first step, no skipped step, and resume re-applies the guard
        base_keep, base = base, hot
        r5 = run(os.path.join(d, "o5"), preset="cpu16", lr=2e-4)
        m5 = r5["manifest"]
        assert m5["fp16_guard"]["fp32_from"] == 1 and m5["non_finite_steps"] == 0 and m5["drop_in_ok"], m5
        assert math.isfinite(m5["train_loss_last"]) and math.isfinite(m5["val_loss_after"])
        r6 = run(os.path.join(d, "o6"), steps_first=50, preset="cpu16", lr=2e-4)
        w5 = load_file(os.path.join(d, "o5", "merged", "model.safetensors"))
        w6 = load_file(os.path.join(d, "o6", "merged", "model.safetensors"))
        assert r6["finished"] and all(torch.equal(w5[k], w6[k]) for k in w5), "guarded resume changed the result"
        base = base_keep
        # ---- source mix: '@share' fixes how often a source is sampled, whatever its size; val from the first source
        syn = os.path.join(d, "syn.jsonl")
        with open(syn, "w") as f:
            for n in range(300):
                # 1x1 grids only: every augmentation keeps the token count, so samples are recognisable by length
                f.write(json.dumps({"task_id": "syn%04d" % n, "train": [{"input": [[n % 9 + 1]], "output": [[(n + 3) % 9 + 1]]},
                                                                      {"input": [[(n + 1) % 9 + 1]], "output": [[(n + 5) % 9 + 1]]}],
                                    "test": []}) + "\n")
        specs = [spec, f"jsonl:{syn}@0.25"]
        assert split_spec(specs[1]) == (f"jsonl:{syn}", 0.25) and split_spec(spec) == (spec, None)
        tr2, va2, src2 = load_tasks(specs, log=lambda *_: None, val_from="first", with_sources=True)
        assert all(src2[t] == 0 for t in va2) and sum(src2[t] == 1 for t in tr2) == 300
        mix2 = make_mix(specs, tr2, src2)
        assert abs(mix2[0][0] - 0.75) < 1e-9 and abs(mix2[1][0] - 1.0) < 1e-9
        picks = [train_example(tr2, i, 5, fmt, 512, mix2) for i in range(600)]
        n_tok = {len(sequence(list(tr2[t][0][0]), fmt, 512)[0]) for t in tr2 if src2[t] == 1}
        shortest_real = min(len(sequence(list(tr2[t][0][0]), fmt, 512)[0]) for t in tr2 if src2[t] == 0)
        assert len(n_tok) == 1 and max(n_tok) < shortest_real / 2
        from_syn = sum(len(p[0]) in n_tok for p in picks)
        assert 0.19 < from_syn / 600 < 0.31, from_syn / 600            # 300 of ~338 tasks, yet a quarter of the samples
        assert picks[:5] == [train_example(tr2, i, 5, fmt, 512, mix2) for i in range(5)]
        assert make_mix([spec, f"jsonl:{syn}"], tr2, src2) is None
        try:
            make_mix([spec + "@0.9", f"jsonl:{syn}@0.3"], tr2, src2)
            raise SystemExit("weights above 1 accepted")
        except SystemExit as e:
            assert "at most 1" in str(e)
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
          "LoRA-target weights changed; interrupted+resumed run is bit-identical; full-FT path works; fp16 path: "
          f"healthy base stays pure fp16 (loss {m4['train_loss_first']:.2f} -> {m4['train_loss_last']:.2f}), an "
          "overflowing layer is found by the probe and computed in fp32 with losses matching fp32 to 2%, "
          "guarded resume is bit-identical")


if __name__ == "__main__":
    a = parser().parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    if not (a.model and a.out and a.data):
        sys.exit(__doc__)
    train(a)
