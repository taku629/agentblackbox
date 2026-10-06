"""Autopsy of generation failures on the evalscan pickles.

    python3 gen_autopsy.py <inference_outputs_dir> [--log <kernel log>] [--timing <timing.log>]
                           [--gold-nll <dir with gold_nll_rank*.jsonl>] [--csv out.csv]
    python3 gen_autopsy.py --selftest

For every eval test input it records the outcome
    generated   the gold grid is among the candidates
    wrong_only  candidates exist, none is the gold
    zero_cand   no candidate at all
and joins it with structural features of the task. Four reports:

  1. HOW it fails (wrong_only): wrong shape vs right shape, and how close the
     best candidate is cell-wise. Near misses are a decoding / view / TTT
     problem; wrong shapes and far misses are a model problem.
  2. WHAT predicts failure: per feature, generation rate by tertile and an
     AUC (0.5 = no relation), overall and per class (extract / same / larger).
  3. Neighbours: for each ungenerated input, how often its nearest eval
     neighbours (in feature space) WERE generated. High = this kind of task is
     within the model's reach, so more samples / TTT / LoRA can plausibly get
     it; ~0 = a region the model does not cover.
  4. Optional signals when the files are given: TTT loss per task (kernel
     log), per-task time and view starvation (timing.log), and the gold grid's
     NLL under the TTT'd model (gold_nll*.jsonl from the probe_gold lever) --
     the only direct measurement of "how far is the model from the answer".

Features are structural (sizes, colours, object counts, prompt length,
distance to the public training set). They are proxies: two tasks with the
same sizes can need unrelated concepts. Treat every table as evidence for or
against a lever, not as proof.
"""
import glob
import json
import math
import os
import re
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import select_v2  # noqa: E402

BASE = select_v2.BASE


# ------------------------------------------------------------------- features
def n_objects(grid):
    """4-connected same-colour components that are not the most common colour."""
    g = np.asarray(grid)
    bg = Counter(g.ravel().tolist()).most_common(1)[0][0]
    seen = np.zeros(g.shape, bool)
    n = 0
    H, W = g.shape
    for r in range(H):
        for c in range(W):
            if seen[r, c] or g[r, c] == bg:
                continue
            n += 1
            col, stack = g[r, c], [(r, c)]
            seen[r, c] = True
            while stack:
                y, x = stack.pop()
                for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == col:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
    return n


def toks(grid):
    g = np.asarray(grid)
    return g.shape[0] * (g.shape[1] + 1)


def task_features(train, test_in):
    """Features computable WITHOUT the gold output (usable on hidden tasks too)."""
    ti = np.asarray(test_in)
    ins = [np.asarray(p["input"]) for p in train]
    outs = [np.asarray(p["output"]) for p in train]
    ratio = [o.size / i.size for i, o in zip(ins, outs)]
    same_shape = np.mean([i.shape == o.shape for i, o in zip(ins, outs)])
    changed = [float(np.mean(i != o)) for i, o in zip(ins, outs) if i.shape == o.shape]
    new_cols = [len(set(np.unique(o)) - set(np.unique(i))) for i, o in zip(ins, outs)]
    return {
        "in_cells": int(ti.size),
        "in_colors": int(len(np.unique(ti))),
        "in_objects": n_objects(ti),
        "n_train": len(train),
        "train_out_cells": float(np.mean([o.size for o in outs])),
        "train_out_colors": float(np.mean([len(np.unique(o)) for o in outs])),
        "train_out_objects": float(np.mean([n_objects(o) for o in outs])),
        "out_in_ratio": float(np.mean(ratio)),
        "train_same_shape": float(same_shape),
        "train_frac_changed": float(np.mean(changed)) if changed else float("nan"),
        "train_new_colors": float(np.mean(new_cols)),
        "prompt_tokens": int(sum(toks(i) + toks(o) for i, o in zip(ins, outs)) + toks(ti)),
        "shape_spread": float(len({i.shape for i in ins} | {ti.shape})),
    }


def train_distance(feats_eval, k=5):
    """Mean distance (z-scored feature space) from each eval input to its k
    nearest PUBLIC TRAINING tasks: a proxy for 'how far from what an SFT on
    public data has seen'."""
    path = os.path.join(BASE, "arc-agi_training_challenges.json")
    if not os.path.exists(path):
        return {}
    tr = json.load(open(path))
    rows = [task_features(t["train"], t["test"][0]["input"]) for t in tr.values() if t["train"]]
    names = [n for n in rows[0] if not any(math.isnan(r[n]) for r in rows)]
    A = np.array([[r[n] for n in names] for r in rows], float)
    mu, sd = A.mean(0), A.std(0) + 1e-9
    A = (A - mu) / sd
    out = {}
    for bk, f in feats_eval.items():
        v = (np.array([f[n] if not math.isnan(f[n]) else mu[i] for i, n in enumerate(names)], float) - mu) / sd
        d = np.sort(np.sqrt(((A - v) ** 2).sum(1)))[:k]
        out[bk] = float(d.mean())
    return out


# ------------------------------------------------------------------- outcomes
def outcome_of(cands, gold):
    """-> (outcome, detail dict) for one test input."""
    if not cands:
        return "zero_cand", {}
    grids = {}
    for key, s in cands.items():
        grids.setdefault(select_v2.hashable(s["solution"]), [np.asarray(s["solution"]), set()])[1].add(select_v2.view_of(key))
    best_acc, shape_ok, votes_best = 0.0, False, 0
    for g, views in grids.values():
        if g.shape == gold.shape:
            shape_ok = True
            acc = float(np.mean(g == gold))
            if acc > best_acc:
                best_acc, votes_best = acc, len(views)
    top_views = max(len(v) for _, v in grids.values())
    d = {"n_grids": len(grids), "n_views": len({select_v2.view_of(k) for k in cands}), "top_views": top_views,
         "shape_ok": shape_ok, "best_cell_acc": best_acc, "best_views": votes_best}
    return ("generated" if best_acc == 1.0 else "wrong_only"), d


def miss_kind(d):
    if not d.get("shape_ok"):
        return "wrong_shape"
    a = d["best_cell_acc"]
    return "near (>=95% cells)" if a >= 0.95 else "mid (80-95%)" if a >= 0.80 else "far (<80%)"


# -------------------------------------------------------------- optional joins
def parse_ttt_loss(log_path):
    out = {}
    pat = re.compile(r"training stats for puzzle (\w+): .*?training_loss=([0-9.eE+-]+|nan)")
    for line in open(log_path, errors="replace"):
        m = pat.search(line)
        if m and m.group(2) != "nan":
            out[m.group(1)] = float(m.group(2))
    return out


def parse_timing(path):
    tot, ttt = {}, {}
    for line in open(path, errors="replace"):
        m = re.search(r"puzzle (\w+): TOTAL (\d+)s", line)
        if m:
            tot[m.group(1)] = int(m.group(2))
        m = re.search(r"puzzle (\w+): TTT done at (\d+)s", line)
        if m:
            ttt[m.group(1)] = int(m.group(2))
    return tot, ttt


def parse_gold_nll(d):
    out = {}
    for f in glob.glob(os.path.join(d, "gold_nll*.jsonl")):
        for line in open(f):
            if line.strip():
                r = json.loads(line)
                out[r["bk"]] = r
    return out


# ------------------------------------------------------------------ statistics
def auc(x_pos, x_neg):
    """P(feature of a generated input > feature of an ungenerated one); ties 0.5."""
    if not len(x_pos) or not len(x_neg):
        return float("nan")
    allv = np.concatenate([x_pos, x_neg])
    order = allv.argsort(kind="stable")
    ranks = np.empty(len(allv))
    ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):                      # average ranks of ties
        m = allv == v
        ranks[m] = ranks[m].mean()
    return float((ranks[:len(x_pos)].sum() - len(x_pos) * (len(x_pos) + 1) / 2) / (len(x_pos) * len(x_neg)))


def tertiles(x, y):
    """generation rate in the low / mid / high third of a feature."""
    order = np.argsort(x, kind="stable")
    parts = np.array_split(order, 3)
    return [(float(x[p].min()), float(x[p].max()), int(y[p].sum()), len(p)) for p in parts if len(p)]


def feature_table(rows, names, title, min_n=12):
    y = np.array([r["outcome"] == "generated" for r in rows], float)
    print(f"\n{title}: {int(y.sum())}/{len(rows)} generated ({y.mean() * 100:.0f}%)")
    if len(rows) < min_n or y.sum() == 0 or y.sum() == len(rows):
        print("  (too few inputs or no contrast for a feature table)")
        return []
    res = []
    for n in names:
        x = np.array([r.get(n, float("nan")) for r in rows], float)
        ok = ~np.isnan(x)
        if ok.sum() < min_n or len(np.unique(x[ok])) < 3:
            continue
        a = auc(x[ok][y[ok] == 1], x[ok][y[ok] == 0])
        res.append((abs(a - 0.5), n, a, tertiles(x[ok], y[ok])))
    res.sort(reverse=True)
    print(f"  {'feature':20s}  AUC   generation rate by tertile (low | mid | high)")
    for _, n, a, t in res:
        cells = " | ".join(f"{g}/{m} [{lo:.3g}..{hi:.3g}]" for lo, hi, g, m in t)
        flag = "  <-- " + ("higher = easier" if a > 0.5 else "higher = harder") if abs(a - 0.5) >= 0.15 else ""
        print(f"  {n:20s}  {a:.2f}  {cells}{flag}")
    return res


def neighbour_rates(rows, names, k=7):
    """Leave-one-out generation rate among each input's k nearest eval neighbours."""
    X = np.array([[r.get(n, float("nan")) for n in names] for r in rows], float)
    mu = np.nanmean(X, 0)
    X = np.where(np.isnan(X), mu, X)
    X = (X - X.mean(0)) / (X.std(0) + 1e-9)
    y = np.array([r["outcome"] == "generated" for r in rows], float)
    task = [r["bk"].split("_")[0] for r in rows]
    out = []
    for i in range(len(rows)):
        d = np.sqrt(((X - X[i]) ** 2).sum(1))
        d[[j for j in range(len(rows)) if task[j] == task[i]]] = np.inf     # not itself, not its sibling test inputs
        nn = np.argsort(d, kind="stable")[:k]
        out.append(float(y[nn].mean()))
    return np.array(out)


# ------------------------------------------------------------ SFT room verdict
DFS_CUT = -math.log(0.2)
BANDS = [("decode reach", 1.39), ("SFT-near", 6.9), ("SFT-far", 23.0), ("model", float("inf"))]


def sft_verdict(rows):
    """How much would the model have to move to GENERATE each missed gold grid?

    gain = (gold NLL in the best view) - 1.609: the nats still missing before the
    DFS (cut p > 0.2) can emit the gold in at least one view. Bands and why:
      decode reach  gain <= 1.39   p >= 0.05 already: the looser G2 cut finds it, no training needed
      SFT-near      gain <= 6.9    within x1000 in probability. A confidently wrong cell costs
                                   ~3-7 nats (p 0.05 .. 0.001), so this is 1-2 such cells.
      SFT-far       gain <= 23     several wrong cells or one hard misconception (x1e10)
      model         beyond         the answer is not a neighbour of what the model believes
    These cut-offs are reasoned defaults, not measurements. The measured version is
    `swap_eval.py compare --base-nll --cand-nll`: it reports how many nats a known
    amount of extra SFT (xcalibur: 63 updates) actually moved these same inputs."""
    un = [r for r in rows if r["outcome"] != "generated" and "gold_nll_min" in r]
    if not un:
        return None
    gain = np.array([max(0.0, r["gold_nll_min"] - DFS_CUT) for r in un])
    counts, lo = [], -1.0
    for name, hi in BANDS:
        counts.append((name, int(((gain > lo) & (gain <= hi)).sum())))
        lo = hi
    n = len(un)
    share = {k: v / n for k, v in counts}
    per_cell = []
    for r in un:                                       # nats per wrong cell of the best wrong candidate
        if r.get("shape_ok") and r.get("best_cell_acc", 1.0) < 1.0:
            wrong = max(1.0, round((1 - r["best_cell_acc"]) * r["gold_cells"]))
            per_cell.append(r["gold_nll_min"] / wrong)
    if share["decode reach"] >= 0.15:
        verdict = "decode first: a looser cut (G2) already reaches a sizeable share"
    elif share["decode reach"] + share["SFT-near"] >= 0.30:
        verdict = "SFT plausible: many missed golds are within ~x1000 of being generated"
    elif share["model"] >= 0.60:
        verdict = "model limit: most missed golds are far outside the model's beliefs; small SFT will not reach"
    else:
        verdict = "mixed: SFT can only reach the near bands; measure with swap_eval.py compare before training"
    return {"n": n, "bands": counts, "median_gain": float(np.median(gain)),
            "nats_per_wrong_cell": float(np.median(per_cell)) if per_cell else None, "verdict": verdict}


# ------------------------------------------------------------------------ main
def build_rows(store, log=None, timing=None, gold_nll=None):
    gold, queries, weight, tclass = select_v2.load_eval()
    decoded = select_v2.load_candidates(store)
    feats = {bk: task_features(q["train"], q["test"][0]["input"]) for bk, q in queries.items()}
    dist = train_distance(feats)
    ttt_loss = parse_ttt_loss(log) if log else {}
    tot, ttt_t = parse_timing(timing) if timing else ({}, {})
    gnll = parse_gold_nll(gold_nll) if gold_nll else {}
    files = Counter(f.split(".")[0] for f in os.listdir(store))
    rows = []
    for bk, g in gold.items():
        task = bk.split("_")[0]
        o, d = outcome_of(decoded.get(bk, {}), g)
        r = dict(bk=bk, cls=tclass[bk], outcome=o, weight=weight[bk], **feats[bk], **d)
        r["gold_cells"] = int(g.size)
        r["gold_tokens"] = toks(g)
        r["views_decoded"] = files.get(bk, 0)
        if bk in dist:
            r["dist_to_train"] = dist[bk]
        if task in ttt_loss:
            r["ttt_loss"] = ttt_loss[task]
        if task in tot:
            r["task_seconds"] = tot[task]
        if task in ttt_t:
            r["ttt_seconds"] = ttt_t[task]
        if bk in gnll:
            a = np.asarray(gnll[bk]["gold_aug_nll"], float)
            r["gold_nll_mean"] = float(a.mean())
            r["gold_nll_min"] = float(a.min())
            r["gold_nll_per_tok"] = float(a.mean() / max(gnll[bk].get("n_tok", toks(g)), 1))
        rows.append(r)
    return rows


FEATS = ["in_cells", "in_colors", "in_objects", "n_train", "train_out_cells", "train_out_colors",
         "train_out_objects", "out_in_ratio", "train_same_shape", "train_frac_changed", "train_new_colors",
         "prompt_tokens", "shape_spread", "gold_tokens", "dist_to_train", "ttt_loss", "task_seconds",
         "ttt_seconds", "views_decoded", "gold_nll_mean", "gold_nll_per_tok"]
NEIGH = ["in_cells", "in_colors", "in_objects", "n_train", "train_out_cells", "out_in_ratio",
         "train_same_shape", "train_frac_changed", "train_new_colors", "prompt_tokens"]


def report(rows):
    n = len(rows)
    oc = Counter(r["outcome"] for r in rows)
    print(f"{n} test inputs: generated {oc['generated']}, wrong_only {oc['wrong_only']}, zero_cand {oc['zero_cand']}")
    print("\n1. HOW it fails, by class (ungenerated inputs only)")
    kinds = ["zero_cand", "wrong_shape", "far (<80%)", "mid (80-95%)", "near (>=95% cells)"]
    tab = defaultdict(Counter)
    for r in rows:
        if r["outcome"] == "zero_cand":
            tab[r["cls"]]["zero_cand"] += 1
        elif r["outcome"] == "wrong_only":
            tab[r["cls"]][miss_kind(r)] += 1
    print(f"  {'class':8s} " + " ".join(f"{k:>19s}" for k in kinds) + "   ungenerated/total")
    for cl in sorted({r["cls"] for r in rows}):
        tot_cl = sum(1 for r in rows if r["cls"] == cl)
        print(f"  {cl:8s} " + " ".join(f"{tab[cl][k]:19d}" for k in kinds) + f"   {sum(tab[cl].values())}/{tot_cl}")
    allk = Counter()
    for c in tab.values():
        allk.update(c)
    miss = sum(allk.values())
    near, model = allk["near (>=95% cells)"], allk["wrong_shape"] + allk["far (<80%)"]
    weak = sum(1 for r in rows if r["outcome"] == "wrong_only" and r.get("top_views", 0) < 3)
    print(f"  -> near misses {near}/{miss}; wrong shape or far {model}/{miss}; zero candidates {allk['zero_cand']}/{miss}")
    print(f"     wrong_only inputs where no grid has 3 agreeing views: {weak} "
          "(the model is guessing, not confidently wrong)")

    print("\n2. WHAT predicts generation (AUC 0.5 = unrelated; flagged when |AUC-0.5| >= 0.15)")
    top = feature_table(rows, FEATS, "all classes")
    for cl in sorted({r["cls"] for r in rows}):
        feature_table([r for r in rows if r["cls"] == cl], FEATS, f"class {cl}")

    print("\n3. Neighbours: generation rate among the 7 most similar eval inputs (other tasks only)")
    nr = neighbour_rates(rows, NEIGH)
    y = np.array([r["outcome"] == "generated" for r in rows])
    for i, r in enumerate(rows):
        r["neighbour_rate"] = float(nr[i])
    a = auc(nr[y], nr[~y])
    print(f"  does the neighbour rate separate generated from ungenerated at all?  AUC {a:.2f} "
          f"({'yes - structure carries signal' if a >= 0.6 else 'barely - structural similarity says little here'})")
    fail = nr[~y]
    if len(fail):
        buckets = [("0", fail == 0), ("0-0.3", (fail > 0) & (fail < 0.3)), (">=0.3", fail >= 0.3)]
        print("  ungenerated inputs by neighbour rate: " + "  ".join(f"{k}: {int(m.sum())}" for k, m in buckets))
        print(f"  expected extra solves if each ungenerated input reached its neighbours' rate: {fail.sum():.1f} "
              f"of {len(fail)} (an upper-ish bound on what 'more of the same model' can buy)")

    has_nll = [r for r in rows if "gold_nll_mean" in r]
    if has_nll:
        print("\n4. Gold NLL under the TTT'd model (probe_gold): mean over 8 augmented views")
        for name, sel in (("generated", lambda r: r["outcome"] == "generated"),
                          ("ungenerated", lambda r: r["outcome"] != "generated")):
            v = np.array([r["gold_nll_mean"] for r in has_nll if sel(r)])
            if len(v):
                q = np.percentile(v, [25, 50, 75])
                print(f"  {name:12s} n={len(v):3d}  quartiles {q[0]:.2f} / {q[1]:.2f} / {q[2]:.2f} nats")
        un = [r for r in has_nll if r["outcome"] != "generated"]
        b = Counter("p>0.05 (decode lever reach)" if r["gold_nll_min"] < 3.0 else
                    "p 1e-2..1e-4" if r["gold_nll_min"] < 9.2 else "beyond 1e-4 (model)" for r in un)
        print("  ungenerated by best-view gold probability: " + "  ".join(f"{k}: {v}" for k, v in sorted(b.items())))
        sv = sft_verdict(rows)
        if sv:
            print(f"  nats still missing before the DFS could emit the gold (median {sv['median_gain']:.1f}): "
                  + "  ".join(f"{k}: {v}" for k, v in sv["bands"]))
            if sv["nats_per_wrong_cell"] is not None:
                print(f"  gold NLL per wrong cell of the closest candidate: median {sv['nats_per_wrong_cell']:.1f} nats "
                      "(3-7 = a few confidently wrong cells; >>7 = the miss is not local)")
            print("  SFT ROOM: " + sv["verdict"])

    print("\nReading guide")
    print("  decode / views (G1-G3)  <- many zero_cand or near misses; gold NLL best view < 3 nats; views_decoded predicts success")
    print("  TTT capacity / steps    <- ttt_loss higher on failures; near + mid misses dominate; weak view agreement")
    print("  SFT / LoRA on more data <- dist_to_train predicts failure; neighbour rate > 0 for most failures")
    print("  model swap              <- wrong shape / far misses dominate; neighbour rate ~0; gold NLL beyond 1e-4;")
    print("                             no structural feature separates success from failure")
    return top


def write_csv(rows, path):
    keys = sorted({k for r in rows for k in r})
    with open(path, "w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")


def selftest():
    import bz2
    import contextlib
    import io
    import pickle
    import tempfile
    # statistics
    assert auc(np.array([3.0, 4.0]), np.array([1.0, 2.0])) == 1.0 and auc(np.array([1.0]), np.array([1.0])) == 0.5
    assert abs(auc(np.array([1.0, 3.0]), np.array([2.0, 4.0])) - 0.25) < 1e-9
    assert n_objects([[0, 0, 1], [0, 2, 1], [3, 0, 0]]) == 3 and n_objects([[5, 5], [5, 5]]) == 0
    g = np.array([[1, 2], [3, 4]])
    mk = lambda sol: {"beam_score": 0.1, "score_aug": [1.0] * 8, "solution": np.asarray(sol)}     # noqa: E731
    assert outcome_of({}, g)[0] == "zero_cand"
    assert outcome_of({"a.out0": mk(g)}, g)[0] == "generated"
    o, d = outcome_of({"a.out0": mk([[1, 2], [3, 9]]), "b.out0": mk([[1, 2, 3]])}, g)
    assert o == "wrong_only" and d["shape_ok"] and d["best_cell_acc"] == 0.75 and miss_kind(d) == "far (<80%)"
    assert miss_kind(outcome_of({"a.out0": mk([[1, 2, 3]])}, g)[1]) == "wrong_shape"
    # end to end on synthetic pickles where success is driven by input size (small = generated)
    gold, queries, _, _ = select_v2.load_eval()
    rng = np.random.default_rng(0)
    sizes = {bk: np.asarray(q["test"][0]["input"]).size for bk, q in queries.items()}
    cut = np.median(list(sizes.values()))
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "inference_outputs")
        os.makedirs(store)
        log, timing, nll = os.path.join(d, "k.log"), os.path.join(d, "timing.log"), []
        with open(log, "w") as lf, open(timing, "w") as tf:
            for i, (bk, gd) in enumerate(sorted(gold.items())):
                task = bk.split("_")[0]
                small = sizes[bk] < cut
                lf.write(f"[Rank 0] training stats for puzzle {task}: TrainOutput(global_step=128, "
                         f"training_loss={(0.0001 if small else 0.01) * (1 + i % 3)}, metrics={{}})\n")
                tf.write(f"[Rank 0] TIMING puzzle {task}: TTT done at 300s\n[Rank 0] TIMING puzzle {task}: TOTAL 900s\n")
                nll.append({"bk": bk, "gold_aug_nll": [(0.5 if small else 30.0) + i % 4] * 8, "n_tok": toks(gd)})
                if i % 9 == 0:
                    continue                                              # zero candidates
                wrong = gd.copy()
                wrong.flat[0] = (wrong.flat[0] + 1) % 10
                sols = [gd, wrong] if small else [wrong, np.zeros((1, 1), int)]
                for v in range(5):
                    with bz2.BZ2File(os.path.join(store, f"{bk}.permute0123456789.ex{v}1"), "w") as f:
                        pickle.dump([mk(sols[int(rng.integers(0, 2))])], f)
        with open(os.path.join(d, "gold_nll_rank0.jsonl"), "w") as f:
            f.write("\n".join(json.dumps(x) for x in nll) + "\n")
        rows = build_rows(store, log, timing, d)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            top = report(rows)
        write_csv(rows, os.path.join(d, "a.csv"))
        assert len(open(os.path.join(d, "a.csv")).readlines()) == len(gold) + 1
    out = buf.getvalue()
    assert len(rows) == len(gold) == 172
    oc = Counter(r["outcome"] for r in rows)
    assert oc["zero_cand"] == 20 and oc["generated"] > 30 and oc["wrong_only"] > 30, oc
    by = {n: a for _, n, a, _ in top}
    assert by["in_cells"] < 0.25 and by["ttt_loss"] < 0.3 and by["gold_nll_mean"] < 0.25, by      # planted relations recovered
    assert all(k in out for k in ("1. HOW it fails", "2. WHAT predicts", "3. Neighbours", "4. Gold NLL", "Reading guide"))
    assert all("dist_to_train" in r and "neighbour_rate" in r for r in rows)
    assert "SFT ROOM: model limit" in out                               # planted: missed golds sit at ~30 nats
    fake = lambda nll, acc=0.98: {"outcome": "wrong_only", "gold_nll_min": nll, "shape_ok": True,     # noqa: E731
                                  "best_cell_acc": acc, "gold_cells": 100}
    v = sft_verdict([fake(2.5)] * 2 + [fake(6.0)] * 3 + [fake(20.0)] * 3 + [fake(60.0)] * 2)
    assert v["bands"] == [("decode reach", 2), ("SFT-near", 3), ("SFT-far", 3), ("model", 2)], v
    assert v["verdict"].startswith("decode first") and v["nats_per_wrong_cell"] == 6.5
    assert sft_verdict([fake(6.0)] * 4 + [fake(40.0)] * 6)["verdict"].startswith("SFT plausible")
    assert sft_verdict([fake(15.0)] * 5 + [fake(40.0)] * 5)["verdict"].startswith("mixed")
    assert sft_verdict([{"outcome": "generated", "gold_nll_min": 0.1}]) is None
    print("selftest ok: outcome classification, AUC, feature tables, neighbour rates and the optional joins "
          f"(TTT loss / timing / gold NLL) run on all {len(rows)} eval inputs; planted size->failure relation recovered "
          f"(AUC in_cells {by['in_cells']:.2f})")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--selftest":
        selftest()
        sys.exit(0)
    a = sys.argv[1:]
    opt = {k: a[a.index(k) + 1] for k in ("--log", "--timing", "--gold-nll", "--csv") if k in a}
    rows = build_rows(a[0], opt.get("--log"), opt.get("--timing"), opt.get("--gold-nll"))
    report(rows)
    if "--csv" in opt:
        write_csv(rows, opt["--csv"])
        print("\nper-input table written to", opt["--csv"])
