"""Generation-side levers for the AGI-2 perfpatch kernel, with the decision rule.

    python3 gen_boost.py --decide /tmp/evalscan/select_verdict.json [--slack 0.4]
    <venv with torch>/python gen_boost.py --selftest       # CPU, fake LM, real arc_loader / turbo_dfs / calc_scores

`decode_task` replaces the decode block of arc_solver.worker() (the part after
TTT). With every lever off it writes byte-identical pickles to production --
that is asserted in the selftest by running the ORIGINAL block, extracted from
arc_solver.py, against the same fake model.

Levers (all gated by the per-task time cap and the global end_time):
  g1  greedy fallback. The DFS keeps a beam only while its cumulative
      probability stays above 0.2; a 30x30 answer (~930 tokens) needs
      p > 0.9983 per token, so one uncertain cell empties the view and the
      input ships the [[0]] placeholder. Greedy is one forward pass per token.
        "empty_input": only for test inputs with NO candidate after the normal
                       pass -- cannot change any input that already had candidates
        "empty_view":  for every empty view -- adds 1-vote candidates to inputs
                       that have others; needs its own A/B
  g3  16 extra views (new colour permutations / example orders) for inputs
      whose best grid is backed by < min_agree views
  g2  one more DFS over the base views with the cut loosened to p > 0.05,
      keeping only beams the first pass could not have produced
  faircap  per-task cap from the time actually left instead of a fixed 1200 s
  probe_gold  measurement only, never in a competition rerun: teacher-forced
      NLL of the GOLD grid under the TTT'd model on the 8 aug-scoring views,
      appended to <probe_dir>/gold_nll_rank<r>.jsonl. Tells, per missed input,
      whether the answer was just outside the decoder's reach or nowhere near
      (read by gen_autopsy.py --gold-nll). Costs two scoring batches per input.

Measured locally and therefore NOT implemented: raising max_seq_length (3/172
eval inputs lose a train example at 8192, none extraction-type) and extra TTT
epochs (production already ends each task at training_loss ~1e-4).
"""
import bz2
import json
import math
import os
import pickle
import sys
import time
from collections import defaultdict

import numpy as np

try:
    import torch
except ImportError:                                    # --decide works without torch
    torch = None

LEVERS_OFF = {"g1": "off", "g3": False, "g2": False, "faircap": False, "probe_gold": False}
GOLD_PATH = "/kaggle/input/competitions/arc-prize-2026-arc-agi-2/arc-agi_evaluation_solutions.json"
BASE_CAP = 1200.0
LOOSE_P = 0.05
MIN_AGREE = 3


# --------------------------------------------------------------------- decode
def greedy_decode(model, prefix_tokens, max_new_tokens, end_time, arc_token_ids, eos_id, pad_id):
    """Arg-max decode for a batch of equal-length prefixes, restricted to the
    ARC vocabulary. Same return shape as inference_turbo_dfs:
    [(batch_id, [(nll, tokens)])], only rows that reached EOS."""
    with torch.no_grad():
        input_ids = torch.tensor(prefix_tokens, device=model.device, dtype=torch.long)
        out = model(input_ids=input_ids, return_dict=True, use_cache=True)
        n, pos = input_ids.size(0), input_ids.size(1)
        ids = arc_token_ids(out.logits.device)
        seqs = [[] for _ in range(n)]
        nll = [0.0] * n
        alive = [True] * n
        done = [False] * n
        logits, cache = out.logits[:, -1], out.past_key_values
        for _ in range(max_new_tokens):
            lf = logits.float()
            lp = lf.index_select(-1, ids) - torch.logsumexp(lf, dim=-1, keepdim=True)
            best = lp.argmax(dim=-1)
            best_lp = lp.gather(-1, best.view(-1, 1)).view(-1).cpu().tolist()
            toks = ids[best].cpu().tolist()
            feed = []
            for i in range(n):
                if not alive[i]:
                    feed.append(pad_id)
                    continue
                nll[i] -= best_lp[i]
                seqs[i].append(toks[i])
                if toks[i] == eos_id:
                    alive[i], done[i] = False, True
                    feed.append(pad_id)
                else:
                    feed.append(toks[i])
            if not any(alive) or time.time() > end_time:
                break
            out = model(
                input_ids=torch.tensor(feed, device=model.device, dtype=torch.long).view(-1, 1),
                position_ids=torch.full((n, 1), pos, device=model.device),
                past_key_values=cache,
                return_dict=True,
                use_cache=True,
            )
            logits, cache = out.logits[:, -1], out.past_key_values
            pos += 1
    return [(i, [(nll[i], seqs[i])]) for i in range(n) if done[i]]


def make_batches(keys):
    """Production batching: per test input, rotation-family batches for every
    test first, then transpose-family batches (equal prompt lengths per batch)."""
    by_test = defaultdict(list)
    for subkey in sorted(keys):
        by_test[subkey.split(".")[0].split("_")[1]].append(subkey)
    batches = []
    for offsets in (([0, 4], [2, 6]), ([8, 12], [10, 14])):
        for _, subkeys in by_test.items():
            for pair in offsets:
                batch = []
                for off in pair:
                    batch.extend(subkeys[off:off + 2])
                batches.append(batch)
    return batches


def agreement(decoded_by_view):
    """{view: [sample, ...]} for one test input -> views agreeing on the best-supported grid."""
    votes = defaultdict(int)
    for samples in decoded_by_view.values():
        for h in {tuple(map(tuple, s["solution"])) for s in samples}:
            votes[h] += 1
    return max(votes.values()) if votes else 0


def fair_share_seconds(end_time, tasks_left, n_workers=4, floor=600.0, ceil=2400.0, now=None):
    """Time this task may use if the remaining budget were split evenly over
    the tasks this worker still has to do (itself included)."""
    remaining = max(0.0, end_time - (time.time() if now is None else now))
    mine = math.ceil(max(tasks_left, 0) / n_workers) + 1
    return min(ceil, max(floor, remaining / mine))


def decode_task(ctx, puzzle_ds_multi, start_time, end_time, levers=None, tasks_left=0):
    """Decode every test input of one task. ctx carries what worker() already
    has: model, tokenizer, formatter, ArcDataset, inference_turbo_dfs,
    calc_scores, arc_token_ids, EOS_ID, PAD_ID, max_new_tokens, max_score,
    max_seq_length, dir_outputs, rank."""
    lv = dict(LEVERS_OFF)
    lv.update(levers or {})
    model, tokenizer, formatter = ctx["model"], ctx["tokenizer"], ctx["formatter"]
    rank, max_new, max_score = ctx["rank"], ctx["max_new_tokens"], ctx["max_score"]
    in_len = ctx["max_seq_length"] - max_new
    task_cap = fair_share_seconds(end_time, tasks_left) if lv["faircap"] else BASE_CAP
    stats = defaultdict(int)
    by_input = defaultdict(dict)                 # base key -> {file name: [sample]}
    known_scores = {}

    def out_of_time():
        return time.time() - start_time > task_cap or time.time() > end_time

    def aug_scores(bk, solution):
        """NLL of `solution` for input bk under the 8 augmented scoring views (production recipe)."""
        aug_dataset = ctx["ArcDataset"](
            keys=[bk],
            queries={bk: puzzle_ds_multi.queries.get(bk)},
            replies={bk: [np.asarray(solution).tolist()]},
        )
        aug_dataset = aug_dataset.augment(seed=hash(bk) % 1024**2)
        aug_dataset = aug_dataset.cut_to_len(formatter=formatter, name="input", max_len=in_len)
        aug_queries, aug_answers = [], []
        for augmented_sample in aug_dataset.as_list(formatter):
            aug_queries.append(augmented_sample["input"])
            aug_answers.append(augmented_sample["reply"])
        return (ctx["calc_scores"](aug_queries[:4], aug_answers[:4], tokenizer, model)
                + ctx["calc_scores"](aug_queries[4:], aug_answers[4:], tokenizer, model))

    def score_and_store(ds, subkey, scored_beams, suffix, keep=None):
        bk = subkey.split(".")[0]
        decoded_result = []
        for beam_score, tokens in scored_beams:
            if keep is not None and not keep(beam_score):
                continue
            array = formatter.convert_tokens_to_array(tokens)
            if array is None:
                continue
            solution = ds.invert_mod(array, subkey, inv_perm=True)
            grid_id = (bk, tuple(map(tuple, solution)))
            if grid_id in known_scores:
                augmented_scores = known_scores[grid_id]
            else:
                print(f"[Rank {rank}] scoring {subkey} #{len(decoded_result)}")
                augmented_scores = aug_scores(bk, solution)
                known_scores[grid_id] = augmented_scores
            decoded_result.append({"beam_score": beam_score, "score_aug": augmented_scores, "solution": solution})
        if decoded_result:
            name = subkey + suffix
            with bz2.BZ2File(os.path.join(ctx["dir_outputs"], name), "w") as f:
                pickle.dump(decoded_result, f)
            by_input[bk][name] = decoded_result
        return len(decoded_result)

    def run_pass(ds, batches, mode, suffix="", cut=None, keep=None, label=""):
        """mode: 'dfs' | 'greedy' | 'dfs+greedy' (greedy for views the DFS left empty)."""
        for subkeys in batches:
            if not subkeys:
                continue
            if out_of_time():
                print(f"[Rank {rank}] timeout after {time.time() - start_time:.1f}s for puzzle {ctx.get('key', '')}{label}")
                return False
            print(f"[Rank {rank}] decoding {subkeys}{label}")
            tokens = [tokenizer.encode(ds.get(subkey, formatter)["input"]) for subkey in subkeys]
            result = []
            if mode != "greedy":
                result = ctx["inference_turbo_dfs"](model, tokens, max_new, max_score if cut is None else cut, end_time)
            got = set()
            for subkey_id, scored_beams in result:
                if score_and_store(ds, subkeys[subkey_id], scored_beams, suffix, keep):
                    got.add(subkey_id)
            if mode != "dfs":
                empty = [i for i in range(len(subkeys)) if i not in got]
                if empty and time.time() < end_time:
                    extra = greedy_decode(model, [tokens[i] for i in empty], max_new, end_time,
                                          ctx["arc_token_ids"], ctx["EOS_ID"], ctx["PAD_ID"])
                    for j, beams in extra:
                        stats["g1_grids"] += score_and_store(ds, subkeys[empty[j]], beams, suffix + ".rung")
        return True

    eval_ds = puzzle_ds_multi.augment(n=2, seed=2)
    eval_ds = eval_ds.cut_to_len(formatter=formatter, name="input", max_len=in_len)
    base_batches = make_batches(eval_ds.keys)
    input_keys = sorted({k.split(".")[0] for k in eval_ds.keys})

    with torch.inference_mode():
        # pass 1 -- production behaviour (plus per-view greedy if asked for)
        run_pass(eval_ds, base_batches, "dfs+greedy" if lv["g1"] == "empty_view" else "dfs")

        # G1 empty_input -- inputs that would otherwise ship the placeholder
        if lv["g1"] == "empty_input":
            for bk in input_keys:
                if not by_input[bk] and not out_of_time():
                    stats["g1_inputs"] += 1
                    mine = [[s for s in b if s.split(".")[0] == bk] for b in base_batches]
                    run_pass(eval_ds, mine, "greedy", label=" [g1]")

        # G3 -- more views where there is no consensus
        if lv["g3"]:
            for bk in input_keys:
                if agreement(by_input[bk]) >= MIN_AGREE or out_of_time():
                    continue
                stats["g3_inputs"] += 1
                extra_ds = puzzle_ds_multi.change_keys([bk]).augment(n=2, seed=3)
                extra_ds = extra_ds.cut_to_len(formatter=formatter, name="input", max_len=in_len)
                run_pass(extra_ds, make_batches(extra_ds.keys), "dfs", suffix=".run2", label=" [g3]")

        # G2 -- looser cut, only beams the first pass could not have produced
        if lv["g2"]:
            loose = -math.log(LOOSE_P)
            for bk in input_keys:
                if agreement(by_input[bk]) >= MIN_AGREE or out_of_time():
                    continue
                stats["g2_inputs"] += 1
                mine = [[s for s in b if s.split(".")[0] == bk] for b in base_batches]
                run_pass(eval_ds, mine, "dfs", suffix=".run3", cut=loose,
                         keep=lambda nll: nll >= max_score, label=" [g2]")

        # probe_gold -- measurement only; impossible in a rerun (no solutions file, and guarded anyway)
        if lv["probe_gold"] and not os.getenv("KAGGLE_IS_COMPETITION_RERUN"):
            gold_path = ctx.get("gold_path", GOLD_PATH)
            if os.path.exists(gold_path):
                with open(gold_path) as f:
                    gold_all = json.load(f)
                lines = []
                for bk in input_keys:
                    task, idx = bk.split("_")
                    if task not in gold_all or time.time() > end_time:
                        continue
                    gold = np.asarray(gold_all[task][int(idx)])
                    gid = (bk, tuple(map(tuple, gold)))
                    nll = known_scores[gid] if gid in known_scores else aug_scores(bk, gold)
                    lines.append(json.dumps({"bk": bk, "gold_aug_nll": [float(x) for x in nll],
                                             "n_tok": int(gold.shape[0] * (gold.shape[1] + 1)),
                                             "generated": gid in known_scores}))
                    stats["probe_inputs"] += 1
                if lines:
                    with open(os.path.join(ctx.get("probe_dir", "/kaggle/working"), f"gold_nll_rank{rank}.jsonl"), "a") as f:
                        f.write("\n".join(lines) + "\n")

    if any(v != LEVERS_OFF[k] for k, v in lv.items()):
        print(f"[Rank {rank}] gen_boost levers={lv} cap={task_cap:.0f}s stats={dict(stats)}")
    return {"stats": dict(stats), "by_input": by_input, "task_cap": task_cap}


# ------------------------------------------------------------------- decision
def decide(diag, slack=None):
    """gen_diagnostics dict (select_verdict.json['diagnostics']) -> levers + reasons.
    slack = fraction of the kernel budget the evalscan run left unused
    (1 - sum of per-task TOTAL / (4 workers x budget)); None = unknown."""
    n = max(diag["n_inputs"], 1)
    bins = diag["gold_bins"]
    gold = sum(bins.values())
    low = bins["p .3-.5"] + bins["p .2-.3"]
    weak = diag["low_consensus_with_gold"] + diag["low_consensus_without_gold"]
    levers, why = dict(LEVERS_OFF), []

    if diag["zero_cand"] >= 3:
        levers["g1"] = "empty_input"
        why.append(f"g1=empty_input: {diag['zero_cand']}/{n} inputs have no candidate at all "
                   f"(by class {diag['zero_by_class']}); greedy can only replace their placeholder")
    else:
        why.append(f"g1 off: only {diag['zero_cand']} zero-candidate inputs")

    time_ok = slack is None or slack >= 0.2
    if gold >= 8 and low / gold >= 0.25:
        if time_ok:
            levers["g2"] = True
        why.append(f"g2 {'on' if levers['g2'] else 'blocked by time'}: {low}/{gold} generated golds sit at best-view "
                   "p < 0.5 -- the 0.2 cut is censoring a visible tail")
    else:
        why.append(f"g2 off: {low}/{gold} generated golds below p 0.5 -- no sign of censoring at the cut")

    if weak >= 5:
        if time_ok:
            levers["g3"] = True
        why.append(f"g3 {'on' if levers['g3'] else 'blocked by time'}: {weak} inputs have < {MIN_AGREE} agreeing views "
                   f"({diag['low_consensus_with_gold']} with the gold already generated)")
    else:
        why.append(f"g3 off: only {weak} low-consensus inputs")

    if slack is not None and slack >= 0.2 and (levers["g2"] or levers["g3"] or levers["g1"] != "off"):
        levers["faircap"] = True
        why.append(f"faircap on: {slack:.0%} of the budget was unused")
    elif slack is None and (levers["g2"] or levers["g3"]):
        why.append("slack unknown: g2/g3 enabled under the fixed 1200 s cap only -- pass --slack from timing.log")

    cap_share = diag["wrong_only"] / n
    why.append(f"capability: {diag['wrong_only']}/{n} inputs have candidates but never the gold "
               f"({cap_share:.0%}) -- no decode lever reaches these; that share is the model-swap target")
    return {"levers": levers, "why": why, "changed": levers != LEVERS_OFF}


# ------------------------------------------------------------------- selftest
def _extract(solver_src, names):
    """Source of selected top-level defs / assignments from arc_solver.py
    (the module itself imports unsloth and cannot be imported here)."""
    import ast
    tree = ast.parse(solver_src)
    out = []
    for node in tree.body:
        nm = getattr(node, "name", None)
        if nm is None and isinstance(node, ast.Assign):
            nm = node.targets[0].id if isinstance(node.targets[0], ast.Name) else None
        if nm in names:
            seg = ast.get_source_segment(solver_src, node)
            decos = "".join("@" + ast.get_source_segment(solver_src, d) + "\n" for d in getattr(node, "decorator_list", []))
            out.append(decos + seg)
    return "\n\n".join(out)


def selftest():
    import re
    import tempfile
    import types
    here = os.path.dirname(os.path.abspath(__file__))
    prod = os.path.join(os.path.dirname(os.path.dirname(here)), "submit_ag2_perfpatch", "out")
    if "transformers" not in sys.modules:
        try:
            import transformers  # noqa: F401
        except ImportError:
            sys.modules["transformers"] = types.SimpleNamespace(AutoTokenizer=object)
    sys.path.insert(0, prod)
    sys.path.insert(0, here)
    from arc_loader import ArcDataset, QwenFormatter
    import arc_decoder
    import select_v2

    solver_src = open(os.path.join(prod, "arc_solver.py")).read()
    ns = {"torch": torch, "time": time, "defaultdict": defaultdict, "np": np}
    exec(_extract(solver_src, {"ARC_VOCAB", "ARC_TOKENS", "USER_TOKEN_ID", "ASSISTANT_TOKEN_ID", "PAD_ID", "EOS_ID",
                               "_ARC_TOKEN_ID_CACHE", "_arc_token_ids", "turbo_dfs", "inference_turbo_dfs",
                               "calc_scores"}), ns)
    EOS, PAD, V = ns["EOS_ID"], ns["PAD_ID"], 16

    class Tok:                                           # the kernel's WordLevel tokenizer: 1 token per symbol
        pat = re.compile(r"<\|im_start\|>|<\|im_end\|>|user|assistant|\n|\d")
        ids = {"<|im_start|>": 14, "<|im_end|>": 15, "user": 11, "assistant": 12, "\n": 10}

        def encode(self, text):
            return [self.ids[m] if m in self.ids else int(m) for m in self.pat.findall(text)]

        def decode(self, toks):
            inv = {v: k for k, v in self.ids.items()}
            return "".join(inv.get(int(t), str(int(t))) for t in toks)

    tok = Tok()
    fmt = QwenFormatter(tokenizer=tok)

    class FakeLM:
        """Puts probability conf(prompt) on the next token of the registered
        answer for that prompt, the rest on one distractor; uniform off-path.
        past_key_values is the immutable token history, so DFS siblings can
        share a cache exactly as the real KV cache allows."""
        device = torch.device("cpu")

        def __init__(self):
            self.answers, self.conf, self.calls = {}, {}, 0

        def _row(self, hist):
            hist = hist.tolist()
            out = torch.full((len(hist), V), -math.log(V))
            head = max((i for i in range(len(hist) - 1) if hist[i] == 12 and hist[i + 1] == 10), default=None)
            if head is None:
                return out
            key = tuple(hist[:head + 2])
            if key not in self.answers:
                return out
            ans, c = self.answers[key], self.conf.get(key, 0.97)
            for t in range(head + 1, len(hist)):
                gen = hist[head + 2:t + 1]
                if gen == ans[:len(gen)] and len(gen) < len(ans):
                    row = torch.full((V,), -30.0)
                    nxt = ans[len(gen)]
                    row[nxt] = math.log(c)
                    row[(nxt + 1) % 10 if nxt < 10 else 0] = math.log(1 - c)
                    out[t] = row
            return out

        def __call__(self, input_ids, position_ids=None, past_key_values=None, return_dict=True, use_cache=True):
            self.calls += 1
            hist = input_ids if past_key_values is None else torch.cat([past_key_values, input_ids], dim=1)
            full = torch.stack([self._row(h) for h in hist])
            logits = full[:, -input_ids.size(1):]
            return types.SimpleNamespace(logits=logits, past_key_values=hist if use_cache else None)

    def task(seed, n_test=1):
        r = np.random.default_rng(seed)
        f = lambda g: np.flipud(g).tolist()                      # noqa: E731
        mk = lambda: r.integers(0, 10, size=(3, 3)).tolist()     # noqa: E731
        tr = [{"input": (g := mk()), "output": f(g)} for _ in range(3)]
        te = [{"input": mk()} for _ in range(n_test)]
        return {"train": tr, "test": te}, [f(t["input"]) for t in te]

    def register(lm, ds_with_replies, key, conf_of):
        """Teach the fake LM the right answer for every prompt the kernel will
        build for `key`: decode views (seed 2), G3 views (seed 3), aug-score views."""
        multi = ds_with_replies.change_keys([key]).split_multi_replies()
        # same construction (and therefore the same RNG draws) as the kernel for each kind of view
        views = [(2, multi.augment(n=2, seed=2))]                                  # pass 1: all inputs at once
        for bk in multi.keys:
            views.append((3, multi.change_keys([bk]).augment(n=2, seed=3)))        # G3: one input at a time
            sd = hash(bk) % 1024**2
            views.append((sd, multi.change_keys([bk]).augment(seed=sd)))           # aug scoring
        for sd, d in views:
            for sub in d.keys:
                item = d.get(sub, fmt)
                p = tuple(tok.encode(item["input"]))
                lm.answers[p] = tok.encode(item["reply"])
                lm.conf[p] = conf_of(sd, sub)

    def ctx_for(lm, out_dir):
        return dict(model=lm, tokenizer=tok, formatter=fmt, ArcDataset=ArcDataset,
                    inference_turbo_dfs=ns["inference_turbo_dfs"], calc_scores=ns["calc_scores"],
                    arc_token_ids=ns["_arc_token_ids"], EOS_ID=EOS, PAD_ID=PAD,
                    max_new_tokens=fmt.max_new_tokens(), max_score=-np.log(0.2), max_seq_length=8192,
                    dir_outputs=out_dir, rank=0)

    def original_block(lm, multi, out_dir):
        """The production decode block, sliced out of arc_solver.py and run as-is."""
        a = solver_src.index("        eval_ds = puzzle_ds_multi.augment(n=2, seed=2)\n")
        b = solver_src.index("        memory_allocated = torch.cuda.max_memory_allocated() // 1024**2\n        print(f\"[Rank {rank}] allocated {memory_allocated}MB for inference\")")
        block = "\n".join(l[8:] for l in solver_src[a:b].splitlines())
        env = dict(ns, puzzle_ds_multi=multi, formatter=fmt, tokenizer=tok, model=lm, max_seq_length=8192,
                   max_new_tokens=fmt.max_new_tokens(), max_score=-np.log(0.2), start_time=time.time(),
                   end_time=time.time() + 600, rank=0, key="t", dir_outputs=out_dir, ArcDataset=ArcDataset,
                   bz2=bz2, pickle=pickle, os=os)
        exec(compile(block, "original_block", "exec"), env)

    def load(d):
        out = {}
        for f in sorted(os.listdir(d)):
            with bz2.BZ2File(os.path.join(d, f)) as fh:
                out[f] = pickle.load(fh)
        return out

    def same(a, b):
        return a.keys() == b.keys() and all(
            len(a[k]) == len(b[k]) and all(
                x["beam_score"] == y["beam_score"] and x["score_aug"] == y["score_aug"]
                and np.array_equal(x["solution"], y["solution"]) for x, y in zip(a[k], b[k])) for k in a)

    quiet = open(os.devnull, "w")
    real_stdout = sys.stdout

    def run(lm, multi, levers, cap_end=600, tasks_left=0, start=None):
        d = tempfile.mkdtemp()
        sys.stdout = quiet
        try:
            r = decode_task(ctx_for(lm, d), multi, time.time() if start is None else start,
                            time.time() + cap_end, levers, tasks_left)
        finally:
            sys.stdout = real_stdout
        return load(d), r

    queries, replies = {}, {}
    for name, seed, nt in (("aaaa0001", 1, 1), ("aaaa0002", 2, 2), ("aaaa0003", 3, 1), ("aaaa0004", 4, 1), ("aaaa0005", 5, 1)):
        queries[name], replies[name] = task(seed, nt)
    ds_gold = ArcDataset(queries=queries, replies=replies, is_orig=True)
    ds_run = ArcDataset(queries=queries, is_orig=True)             # what the kernel sees: no replies

    def gold_in(files, key, i=0):
        g = np.asarray(replies[key][i])
        return sum(any(np.array_equal(s["solution"], g) for s in v) for k, v in files.items()
                   if k.startswith(f"{key}_{i}."))

    lm = FakeLM()
    L = len(tok.encode(fmt.fmt_reply([replies["aaaa0001"][0]])))                  # 12 tokens for a 3x3 grid
    register(lm, ds_gold, "aaaa0001", lambda sd, sub: 0.97)                         # confident everywhere
    register(lm, ds_gold, "aaaa0002", lambda sd, sub: 0.97)                         # 2 test inputs
    register(lm, ds_gold, "aaaa0003", lambda sd, sub: 0.80)                         # 0.8^12 = 0.07 < 0.2: DFS finds nothing
    register(lm, ds_gold, "aaaa0004", lambda sd, sub: 0.97 if sd != 2 or sub.split(".")[1] == "rot90" and "rot90.rot90" not in sub else 0.6)
    register(lm, ds_gold, "aaaa0005", lambda sd, sub: 0.85)                         # 0.85^12 = 0.14: between the two cuts

    def multi(key):
        return ds_run.change_keys([key]).split_multi_replies()

    # T1 regression: all levers off == the original production block, for 1- and 2-test tasks
    for key in ("aaaa0001", "aaaa0002", "aaaa0004"):
        d0 = tempfile.mkdtemp()
        sys.stdout = quiet
        try:
            original_block(lm, multi(key), d0)
        finally:
            sys.stdout = real_stdout
        mine, _ = run(lm, multi(key), None)
        assert same(load(d0), mine) and len(mine) > 0, key
    base1, _ = run(lm, multi("aaaa0001"), None)
    assert len(base1) == 16 and gold_in(base1, "aaaa0001") == 16
    assert len(run(lm, multi("aaaa0002"), None)[0]) == 32

    # T2 greedy NLL == DFS NLL for the same beam (the two decoders score identically)
    m1 = multi("aaaa0001").augment(n=2, seed=2)
    sub = sorted(m1.keys)[0]
    toks = [tok.encode(m1.get(sub, fmt)["input"])]
    dfs = ns["inference_turbo_dfs"](lm, toks, fmt.max_new_tokens(), -np.log(0.2), time.time() + 60)
    gre = greedy_decode(lm, toks, fmt.max_new_tokens(), time.time() + 60, ns["_arc_token_ids"], EOS, PAD)
    assert gre[0][1][0][1] == dfs[0][1][0][1] and abs(gre[0][1][0][0] - dfs[0][1][0][0]) < 1e-4
    assert abs(gre[0][1][0][0] + L * math.log(0.97)) < 1e-3

    # T3 G1: zero-candidate input gets greedy grids; inputs with candidates are untouched
    assert len(run(lm, multi("aaaa0003"), None)[0]) == 0
    g1, r = run(lm, multi("aaaa0003"), {"g1": "empty_input"})
    assert len(g1) == 16 and all(k.endswith(".rung") for k in g1) and gold_in(g1, "aaaa0003") == 16
    assert all(s["beam_score"] > -np.log(0.2) for v in g1.values() for s in v) and r["stats"]["g1_inputs"] == 1
    assert same(run(lm, multi("aaaa0001"), {"g1": "empty_input"})[0], base1)
    assert len(run(lm, multi("aaaa0004"), {"g1": "empty_view"})[0]) == 16      # per-view mode fills the 14 empty views

    # T4 G3: 2 agreeing views -> 16 extra views, loaded by the production decoder under the same input
    b4, _ = run(lm, multi("aaaa0004"), None)
    assert len(b4) == 2 and agreement({k: v for k, v in b4.items()}) == 2
    g3, r = run(lm, multi("aaaa0004"), {"g3": True})
    assert sum(k.endswith(".run2") for k in g3) == 16 and r["stats"]["g3_inputs"] == 1
    assert same(run(lm, multi("aaaa0001"), {"g3": True, "g2": True})[0], base1)   # consensus -> no second pass
    d = tempfile.mkdtemp()
    for k, v in g3.items():
        with bz2.BZ2File(os.path.join(d, k), "w") as fh:
            pickle.dump(v, fh)
    dec = arc_decoder.ArcDecoder(types.SimpleNamespace(queries={}, replies={}), n_guesses=2)
    dec.load_decoded_results(d)
    assert list(dec.decoded_results) == ["aaaa0004_0"] and len(dec.decoded_results["aaaa0004_0"]) >= 18
    cands = select_v2.group(dec.decoded_results["aaaa0004_0"])
    top = max(cands, key=lambda c: c["n_views"])
    assert np.array_equal(top["grid"], replies["aaaa0004"][0]) and top["n_views"] == 18 and len(top["geos"]) == 8
    assert np.array_equal(dec.run_selection_algo()["aaaa0004_0"][0], replies["aaaa0004"][0])

    # T5 G2: beams between the two cuts appear only with the loose pass, none duplicated from pass 1
    assert len(run(lm, multi("aaaa0005"), None)[0]) == 0
    g2, r = run(lm, multi("aaaa0005"), {"g2": True})
    assert g2 and all(k.endswith(".run3") for k in g2) and gold_in(g2, "aaaa0005") == len(g2)
    assert all(-np.log(0.2) <= s["beam_score"] < -np.log(LOOSE_P) for v in g2.values() for s in v)
    mixed, _ = run(lm, multi("aaaa0004"), {"g2": True})
    firsts = {k: v for k, v in mixed.items() if not k.endswith(".run3")}
    assert same(firsts, b4)
    assert all(s["beam_score"] >= -np.log(0.2) for k, v in mixed.items() if k.endswith(".run3") for s in v)

    # T6 time gates: an exhausted cap or end_time stops every extra pass (and pass 1, like production)
    assert len(run(lm, multi("aaaa0003"), {"g1": "empty_input", "g3": True, "g2": True}, start=time.time() - 5000)[0]) == 0
    assert len(run(lm, multi("aaaa0003"), {"g1": "empty_input"}, cap_end=-1)[0]) == 0
    now = 1000.0
    assert fair_share_seconds(now + 40000, 116, now=now) == 40000 / 30
    assert fair_share_seconds(now + 4000, 116, now=now) == 600.0 and fair_share_seconds(now + 9e9, 0, now=now) == 2400.0

    # T8 probe_gold: decode output untouched; gold NLL = L * -log(conf) on every view; skipped in a rerun
    gd = tempfile.mkdtemp()
    gp = os.path.join(gd, "sol.json")
    json.dump(replies, open(gp, "w"))

    def probe(key, env=None):
        pd_ = tempfile.mkdtemp()
        d = tempfile.mkdtemp()
        old = os.environ.pop("KAGGLE_IS_COMPETITION_RERUN", None)
        if env:
            os.environ["KAGGLE_IS_COMPETITION_RERUN"] = env
        sys.stdout = quiet
        try:
            decode_task(dict(ctx_for(lm, d), gold_path=gp, probe_dir=pd_), multi(key), time.time(),
                        time.time() + 600, {"probe_gold": True})
        finally:
            sys.stdout = real_stdout
            os.environ.pop("KAGGLE_IS_COMPETITION_RERUN", None)
            if old is not None:
                os.environ["KAGGLE_IS_COMPETITION_RERUN"] = old
        f = os.path.join(pd_, "gold_nll_rank0.jsonl")
        return load(d), ([json.loads(x) for x in open(f)] if os.path.exists(f) else [])

    files, pr = probe("aaaa0001")
    assert same(files, base1) and len(pr) == 1 and pr[0]["generated"] and pr[0]["n_tok"] == L
    assert all(abs(x + L * math.log(0.97)) < 1e-3 for x in pr[0]["gold_aug_nll"]) and len(pr[0]["gold_aug_nll"]) == 8
    files, pr = probe("aaaa0003")                                   # never generated: the probe still measures it
    assert not files and not pr[0]["generated"] and all(abs(x + L * math.log(0.80)) < 1e-3 for x in pr[0]["gold_aug_nll"])
    assert [r["bk"] for r in probe("aaaa0002")[1]] == ["aaaa0002_0", "aaaa0002_1"]
    assert probe("aaaa0001", env="1")[1] == []                      # competition rerun: no probe
    import gen_autopsy
    pd2 = tempfile.mkdtemp()
    json.dump(pr[0], open(os.path.join(pd2, "gold_nll_rank0.jsonl"), "w"))
    assert gen_autopsy.parse_gold_nll(pd2)["aaaa0003_0"]["n_tok"] == L

    # T7 decision rule
    base = {"n_inputs": 172, "zero_cand": 0, "zero_by_class": {}, "wrong_only": 90,
            "gold_bins": {"p>=.8": 30, "p .5-.8": 8, "p .3-.5": 1, "p .2-.3": 0},
            "low_consensus_with_gold": 1, "low_consensus_without_gold": 1}
    assert decide(base)["levers"] == LEVERS_OFF and not decide(base)["changed"]
    assert decide(dict(base, zero_cand=20), slack=0.4)["levers"] == dict(LEVERS_OFF, g1="empty_input", faircap=True)
    cens = dict(base, gold_bins={"p>=.8": 10, "p .5-.8": 8, "p .3-.5": 6, "p .2-.3": 4})
    assert decide(cens, slack=0.4)["levers"]["g2"] and not decide(cens, slack=0.05)["levers"]["g2"]
    weak = dict(base, low_consensus_with_gold=4, low_consensus_without_gold=6)
    assert decide(weak)["levers"]["g3"] and not decide(weak)["levers"]["faircap"]
    assert not decide(weak, slack=0.1)["levers"]["g3"]

    print(f"selftest ok (torch {torch.__version__}, fake LM, {lm.calls} forward calls): levers-off output identical to the "
          "production decode block on 3 tasks; greedy NLL == DFS NLL; G1/G3/G2 produce the expected files, "
          "leave settled inputs untouched and respect the time gates; probe_gold reports the exact gold NLL "
          "without changing any output; decide() rules hold")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        if torch is None:
            sys.exit("selftest needs torch (CPU is enough)")
        selftest()
    elif len(sys.argv) >= 3 and sys.argv[1] == "--decide":
        diag = json.load(open(sys.argv[2]))
        diag = diag.get("diagnostics", diag)
        slack = float(sys.argv[sys.argv.index("--slack") + 1]) if "--slack" in sys.argv else None
        d = decide(diag, slack)
        print(json.dumps(d["levers"]))
        for line in d["why"]:
            print(" -", line)
    else:
        sys.exit(__doc__)
