#!/usr/bin/env python3
"""LoRA SFT for ARC-AGI-3 on Qwen3.8-27B bf16.

Replicates the public pilot recipe (manas joshi): r=16 alpha=32 dropout=0.05,
all linear target modules, 3 epochs, lr 1e-4, cosine — but on ~240x the data
(our winner-run trajectories from the flash-next and 27B agents + public set).

Trains assistant-response tokens only (reasoning + content + tool_call markup);
system/user/tool context tokens are masked to -100.

Env vars:
  BASE_MODEL      path to bf16 Qwen3.8-27B weights (HF dir)
  SFT_JSONL       packed_sft.jsonl path
  OUT_DIR         adapter output dir
  EPOCHS          default 3
  LR              default 1e-4
  MAX_LEN         default 8192
  SUBSAMPLE       default 0 (all); N>0 takes N random lines (seed 1234) before
                  encoding — use to fit the GPU-kernel time/quota budget
  SAVE_MERGED_DIR optional — if set, merge adapter into bf16 and save full model
"""
import json, math, os, sys
from transformers import AutoTokenizer

BASE = os.environ["BASE_MODEL"]
SFT = os.environ["SFT_JSONL"]
OUT = os.environ.get("OUT_DIR", "/kaggle/working/lora_out")
EPOCHS = float(os.environ.get("EPOCHS", "3"))
LR = float(os.environ.get("LR", "1e-4"))
MAX_LEN = int(os.environ.get("MAX_LEN", "8192"))
# deterministic subsample of packed_sft lines (0 = all). ~7.3k tokens/sample,
# so N=1200 -> ~9M train tokens -> ~8-12h at 200-300 tok/s on RTX PRO 6000
SUBSAMPLE = int(os.environ.get("SUBSAMPLE", "0"))
# QLORA=1 loads the base in 4-bit NF4 (bitsandbytes) instead of bf16 —
# fits 27B on a 40GB Colab A100 (bf16 needs ~54GB)
QLORA = os.environ.get("QLORA") == "1"

tok = AutoTokenizer.from_pretrained(BASE)

def render(msgs):
    """Apply Qwen3.8 chat template; normalize OpenAI-style string arguments
    to the dict form the template iterates over (the 'flatten tool-calls'
    gotcha from the public pilot)."""
    norm = []
    for m in msgs:
        m = dict(m)
        if m.get("tool_calls"):
            fixed = []
            for tcall in m["tool_calls"]:
                tcall = dict(tcall)
                fn = dict(tcall.get("function") or {})
                args = fn.get("arguments")
                if isinstance(args, str):
                    try:
                        fn["arguments"] = json.loads(args)
                    except Exception:
                        fn["arguments"] = {"code": args}
                tcall["function"] = fn
                fixed.append(tcall)
            m["tool_calls"] = fixed
        norm.append(m)
    return tok.apply_chat_template(norm, tokenize=False)

def encode_sample(msgs):
    """input_ids + labels; labels only on assistant blocks (incl. <|im_end|>)."""
    ids, labels = [], []
    # the template rejects prefixes with no user query and injects a
    # reasoning-effort preamble — render through the first user and cut
    # there; everything before it is context (labels -100)
    u_idx = next(i for i, m in enumerate(msgs) if m.get("role") == "user")
    head = render(msgs[: u_idx + 1])
    cut = head.index("<|im_start|>user")
    prev_text = head[:cut]
    head_ids = tok(prev_text, add_special_tokens=False).input_ids
    ids += head_ids
    labels += [-100] * len(head_ids)
    header_len = len(tok("<|im_start|>assistant\n", add_special_tokens=False).input_ids)
    for i in range(u_idx, len(msgs)):
        cur = render(msgs[: i + 1])
        seg_text = cur[len(prev_text):]
        seg_ids = tok(seg_text, add_special_tokens=False).input_ids
        if msgs[i].get("role") == "assistant":
            seg_labels = list(seg_ids)
            for k in range(min(header_len, len(seg_labels))):
                seg_labels[k] = -100  # don't train the role header
        else:
            seg_labels = [-100] * len(seg_ids)
        ids += seg_ids
        labels += seg_labels
        prev_text = cur
    ids = ids[:MAX_LEN]
    labels = labels[:MAX_LEN]
    return {"input_ids": ids, "labels": labels,
            "attention_mask": [1] * len(ids)}

def load_ds():
    from datasets import Dataset
    import random
    lines = open(SFT).read().splitlines()
    if SUBSAMPLE and SUBSAMPLE < len(lines):
        # stratified pick by game: floor(N/ngames) each, remainder proportional —
        # a plain random pick would starve tail games (22:1 skew, 279 vs 13)
        rng = random.Random(1234)
        by_game = {}
        for line in lines:
            by_game.setdefault(json.loads(line).get("game", "?"), []).append(line)
        ng = len(by_game)
        floor = SUBSAMPLE // ng
        picked, leftover = [], []
        for g, gl in sorted(by_game.items()):
            rng.shuffle(gl)
            picked += gl[:floor]
            leftover += gl[floor:]
        rng.shuffle(leftover)
        picked += leftover[: SUBSAMPLE - len(picked)]
        rng.shuffle(picked)
        lines = picked
        print(f"subsampled {len(lines)} of packed lines "
              f"(stratified over {ng} games, floor {floor}/game)")
    rows = []
    skipped = 0
    total_tok = 0
    for line in lines:
        c = json.loads(line)
        msgs = c["messages"]
        try:
            enc = encode_sample(msgs)
        except Exception:
            skipped += 1
            continue
        if not any(l != -100 for l in enc["labels"]):
            skipped += 1
            continue
        rows.append(enc)
        total_tok += len(enc["input_ids"])
    print(f"dataset: {len(rows)} samples ({skipped} skipped), "
          f"~{total_tok/1e6:.1f}M train tokens")
    return Dataset.from_list(rows)

def collate(feats):
    import torch
    maxlen = max(len(f["input_ids"]) for f in feats)
    maxlen = (maxlen + 15) // 16 * 16
    pad = tok.pad_token_id or tok.eos_token_id
    input_ids, labels, attn = [], [], []
    for f in feats:
        n = maxlen - len(f["input_ids"])
        input_ids.append(f["input_ids"] + [pad] * n)
        labels.append(f["labels"] + [-100] * n)
        attn.append(f["attention_mask"] + [0] * n)
    return {"input_ids": torch.tensor(input_ids),
            "labels": torch.tensor(labels),
            "attention_mask": torch.tensor(attn)}

def load_model():
    """bf16 base is a *ForConditionalGeneration (multimodal) checkpoint —
    ImageTextToText is the canonical loader; fall back to CausalLM."""
    import torch
    kw = {"device_map": "auto", "attn_implementation": "sdpa"}
    if QLORA:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        kw["dtype"] = torch.bfloat16
    try:
        from transformers import AutoModelForImageTextToText
        return AutoModelForImageTextToText.from_pretrained(BASE, **kw)
    except Exception as e:
        print("ImageTextToText load failed:", type(e).__name__, e)
        from transformers import AutoModelForCausalLM
        return AutoModelForCausalLM.from_pretrained(BASE, **kw)


def main():
    import torch
    from transformers import Trainer, TrainingArguments
    from peft import LoraConfig, get_peft_model
    ds = load_ds()
    model = load_model()
    if QLORA:
        from peft import prepare_model_for_kbit_training
        model = prepare_model_for_kbit_training(model)
    cfg = LoraConfig(
        r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    # wrap BEFORE enabling grad ckpt so PeftModel's implementation also calls
    # enable_input_require_grads (needed for LoRA grads through a frozen base)
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    vision = [n for n, p in model.named_parameters()
              if p.requires_grad and ("visual" in n or "vision" in n)]
    if vision:
        print("WARNING: LoRA matched vision modules:", len(vision))
    for c in (model.config, getattr(model.config, "text_config", None)):
        if c is not None and hasattr(c, "use_cache"):
            c.use_cache = False
    steps_per_epoch = max(1, math.ceil(len(ds) / 8))
    save_steps = max(20, steps_per_epoch // 4)
    print(f"steps/epoch {steps_per_epoch}, adapter ckpt every {save_steps}")
    import inspect
    accepted = set(inspect.signature(TrainingArguments.__init__).parameters)
    kw = dict(
        output_dir=OUT,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=10,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=2,
        bf16=True,
        report_to=[],
        optim="adamw_torch_fused",
        max_grad_norm=1.0,
        dataloader_num_workers=2,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
    )
    dropped = sorted(k for k in kw if k not in accepted)
    if dropped:
        print("TrainingArguments unsupported kwargs dropped:", dropped)
    args = TrainingArguments(**{k: v for k, v in kw.items() if k in accepted})
    trainer = Trainer(model=model, args=args, train_dataset=ds,
                      data_collator=collate)
    trainer.train()
    model.save_pretrained(OUT)
    tok.save_pretrained(OUT)
    print("adapter saved ->", OUT)
    if os.environ.get("SAVE_MERGED_DIR"):
        merged = model.merge_and_unload()
        merged.save_pretrained(os.environ["SAVE_MERGED_DIR"],
                               safe_serialization=True)
        tok.save_pretrained(os.environ["SAVE_MERGED_DIR"])
        print("merged model saved ->", os.environ["SAVE_MERGED_DIR"])

if __name__ == "__main__":
    main()
