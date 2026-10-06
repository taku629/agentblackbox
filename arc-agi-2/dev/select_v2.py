"""Selection strategies v2 for AGI-2 decoded candidates + offline benchmark.

Supersedes select_bench.py (which has a dead diverse2 branch — see REVIEW notes
in the hand-off). Same input: the bz2 pickles an evalscan run writes to
inference_outputs/.

    python3 select_v2.py <inference_outputs_dir>      # benchmark on eval solutions
    python3 select_v2.py --priors                     # prior precision on eval gold (no pickles needed)
    python3 select_v2.py --selftest                   # synthetic mechanics check (no pickles needed)

Sample keys look like
    <task>_<test>[.transpose][.rot90]*.permute<10 digits>.ex<perm>.out<j>
and each sample is {"beam_score": NLL of the beam in that view (lower = more
likely, always < -log(0.2) = 1.609), "score_aug": [8 NLLs of the de-augmented
grid under 8 fresh views], "solution": ndarray}.

Every strategy has the production signature  f(guesses) -> [grid, ...]  (best
first), so the winner drops into arc_decoder.run_selection_algo unchanged;
strategies that need the task itself are built with a closure (see STRATEGIES).
"""
import bz2
import json
import os
import pickle
import sys
from collections import Counter, defaultdict

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAX_SCORE = -np.log(0.2)          # DFS pruning threshold in arc_solver.py
VIEWS_PER_TEST = 16               # 8 geometries x 2 colour permutations


def hashable(g):
    return tuple(map(tuple, g))


# ----------------------------------------------------------------- key parsing
def view_of(key):
    """Sample key -> the decode view (= pickle file name) it came from."""
    return key.rsplit(".out", 1)[0]


def geo_of(key):
    """Geometric family of a sample key: e.g. 'transpose.rot90.rot90' or 'id'."""
    ops = [p for p in key.split(".")[1:] if p in ("transpose", "rot90")]
    return ".".join(ops) if ops else "id"


# ------------------------------------------------------------------- grouping
def group(guesses):
    """Collapse samples to unique grids. Order follows first appearance, which
    is what production's stable sort uses to break ties."""
    cands = {}
    views_all = set()
    for key, g in guesses.items():
        v = view_of(key)
        views_all.add(v)
        h = hashable(g["solution"])
        c = cands.get(h)
        if c is None:
            sol = np.asarray(g["solution"])
            c = cands[h] = dict(grid=g["solution"], h=h, samples=[], view_p=defaultdict(float),
                                geos=set(), aug=np.asarray(g["score_aug"], dtype=float),
                                n_tok=sol.shape[0] * (sol.shape[1] + 1) + 1, shape=sol.shape)
        c["samples"].append(g)
        c["view_p"][v] += float(np.exp(-g["beam_score"]))
        c["geos"].add(geo_of(key))
    out = list(cands.values())
    for c in out:
        c["votes"] = len(c["samples"])
        c["n_views"] = len(c["view_p"])
        c["n_views_decoded"] = max(len(views_all), 1)
        c["mass"] = sum(min(p, 1.0) for p in c["view_p"].values())
        c["aug_mean"] = float(np.mean(c["aug"]))
    return out


def ranked(cands, score):
    return [c for _, c in sorted(((score(c), c) for c in cands), key=lambda x: x[0], reverse=True)]


# -------------------------------------------------------------------- scorers
def s_kgmon(c):
    """Production: vote count minus mean augmented NLL."""
    return c["votes"] - np.mean([np.mean(g["score_aug"]) for g in c["samples"]])


def s_probmul3(c, baseline=3):
    inf = np.sum([baseline - g["beam_score"] for g in c["samples"]])
    aug = np.mean([np.sum([baseline - s for s in g["score_aug"]]) for g in c["samples"]])
    return inf + aug


def s_aug(c):
    """Pure ensemble likelihood: every candidate is scored on the same 8 views,
    so this is the only term that is strictly comparable across candidates."""
    return -c["aug_mean"]


def s_aug_tok(c):
    """Same, per output token — removes the bias against larger grids when the
    candidates of one test input disagree on shape."""
    return -c["aug_mean"] / c["n_tok"]


def s_aug_trim(c):
    """Mean augmented NLL without the single worst view (one view can lose
    train examples to cut_to_len and score a correct grid badly)."""
    a = np.sort(c["aug"])
    return -float(np.mean(a[:-1])) if len(a) > 2 else -c["aug_mean"]


def s_views(c):
    """kgmon with one vote per decode view instead of per beam."""
    return c["n_views"] - c["aug_mean"]


def s_geo(c):
    """Votes counted per geometric family (8 max): agreement across rotations /
    transpose is stronger evidence than the two colour permutations of one
    geometry agreeing."""
    return len(c["geos"]) + 0.5 * (c["n_views"] - len(c["geos"])) - c["aug_mean"]


def s_poe(c, floor=0.1):
    """Product of experts over ALL views. Decode views give log p (floor for a
    decoded view that did not emit the grid: p < 0.2 there by construction);
    aug views give their mean log p. Votes weigh far less than in kgmon."""
    logp = [np.log(min(p, 1.0)) for p in c["view_p"].values()]
    logp += [np.log(floor)] * (c["n_views_decoded"] - c["n_views"])
    return float(np.mean(logp)) - c["aug_mean"]


def s_mass(c):
    """Mixture of experts for the decode term: probability mass the decode
    views put on this grid (not a 0/1 vote), plus the ensemble likelihood."""
    return c["mass"] - c["aug_mean"]


SCORERS = {
    "kgmon": s_kgmon, "probmul_3": s_probmul3, "aug": s_aug, "aug_tok": s_aug_tok,
    "aug_trim": s_aug_trim, "views": s_views, "geo": s_geo, "poe": s_poe, "mass": s_mass,
}


# --------------------------------------------------------------- task priors
# NOT a solver: these never produce a grid. They only demote candidates whose
# shape / colours contradict a rule that holds on EVERY train pair, and only
# when at least one candidate satisfies the rule.
def shape_rule(query):
    """Expected test-output shape if every train pair agrees on one rule, else None."""
    tr = query["train"]
    ti = np.asarray(query["test"][0]["input"]).shape
    ins = [np.asarray(p["input"]).shape for p in tr]
    outs = [np.asarray(p["output"]).shape for p in tr]
    if not tr:
        return None
    if all(i == o for i, o in zip(ins, outs)):
        return ti
    # (a "constant output shape" branch was measured wrong on eval 38007db0 and dropped)
    for swap in (False, True):                      # constant integer scale up / down
        rs = set()
        for i, o in zip(ins, outs):
            a, b = (o, i) if swap else (i, o)
            if b[0] % a[0] or b[1] % a[1]:
                rs = None
                break
            rs.add((b[0] // a[0], b[1] // a[1]))
        if rs and len(rs) == 1:
            r = rs.pop()
            if r == (1, 1):
                continue
            if swap:
                if ti[0] % r[0] or ti[1] % r[1]:
                    return None
                return (ti[0] // r[0], ti[1] // r[1])
            if ti[0] * r[0] <= 30 and ti[1] * r[1] <= 30:
                return (ti[0] * r[0], ti[1] * r[1])
    return None


def allowed_colors(query):
    cols = set(np.unique(np.asarray(query["test"][0]["input"])).tolist())
    for p in query["train"]:
        cols |= set(np.unique(np.asarray(p["output"])).tolist())
    return cols


def with_priors(order, query, shape=True, colors=True):
    """Stable-partition a ranked candidate list: prior-consistent ones first."""
    if query is None or not order:
        return order
    ok = [True] * len(order)
    if shape:
        want = shape_rule(query)
        if want is not None:
            ok = [o and c["shape"] == tuple(want) for o, c in zip(ok, order)]
    if colors:
        allow = allowed_colors(query)
        ok = [o and set(np.unique(c["grid"]).tolist()) <= allow for o, c in zip(ok, order)]
    if not any(ok):
        return order
    return [c for c, o in zip(order, ok) if o] + [c for c, o in zip(order, ok) if not o]


# ------------------------------------------------------------- 2-attempt logic
def hedge(order_a, order_b):
    """attempt_1 = scorer A's best; attempt_2 = scorer B's best when the two
    scorers disagree (their errors are only partly correlated), else A's #2."""
    if not order_a:
        return []
    out = [order_a[0]]
    if order_b and order_b[0]["h"] != order_a[0]["h"]:
        out.append(order_b[0])
    seen = {c["h"] for c in out}
    out += [c for c in order_a[1:] if c["h"] not in seen]
    return out


def diverse_geo(order):
    """What diverse2 intended: attempt_2 = best candidate whose supporting
    geometries are not a subset of the winner's."""
    if len(order) <= 2:
        return order
    top = order[0]
    alt = next((c for c in order[1:] if not c["geos"] <= top["geos"]), order[1])
    return [top, alt] + [c for c in order[1:] if c is not alt]


def make_strategy(scorer="kgmon", hedge_with=None, priors=False, diverse=False):
    """-> f(guesses, query=None) -> [grid, ...]"""
    def f(guesses, query=None):
        cands = group(guesses)
        order = ranked(cands, SCORERS[scorer])
        if priors:
            order = with_priors(order, query)
        if hedge_with:
            other = ranked(cands, SCORERS[hedge_with])
            if priors:
                other = with_priors(other, query)
            order = hedge(order, other)
        if diverse:
            order = diverse_geo(order)
        return [c["grid"] for c in order]
    return f


STRATEGIES = {name: make_strategy(name) for name in SCORERS}
STRATEGIES.update({
    "kgmon+diverse_geo": make_strategy("kgmon", diverse=True),
    "kgmon|aug hedge": make_strategy("kgmon", hedge_with="aug"),
    "kgmon|aug_tok hedge": make_strategy("kgmon", hedge_with="aug_tok"),
    "kgmon|poe hedge": make_strategy("kgmon", hedge_with="poe"),
    "kgmon+priors": make_strategy("kgmon", priors=True),
    "kgmon|aug hedge+priors": make_strategy("kgmon", hedge_with="aug", priors=True),
    "poe+priors": make_strategy("poe", priors=True),
})


def run_selection(decoded_results, queries, strategy, ranker=None, fallback=None):
    """Kernel entry point: decoded_results as in ArcDecoder, queries from
    data.split_multi_replies().queries, strategy a name in STRATEGIES or
    "learned" (then `ranker` = the dict written by --export-ranker).

    A hidden-test rerun must never lose an answer to a selector bug: any
    exception, or an output that is not a permutation of the candidates,
    falls back to `fallback(guesses)` (production score_kgmon) for that input."""
    if strategy == "learned":
        f = make_learned(ranker)
    else:
        f = STRATEGIES[strategy] if isinstance(strategy, str) else strategy
    fb = fallback or (lambda g: STRATEGIES["kgmon"](g))
    out, n_fb = {}, 0
    for bk, v in decoded_results.items():
        try:
            order = f(v, (queries or {}).get(bk))
            n_unique = len({hashable(g["solution"]) for g in v.values()})
            if len(order) != n_unique:
                raise ValueError("selector dropped or duplicated candidates")
        except Exception as e:                      # noqa: BLE001
            n_fb += 1
            print(f"*** select_v2: {bk}: {e.__class__.__name__}: {e} -> kgmon fallback")
            order = fb(v)
        out[bk] = order
    print(f"*** select_v2: strategy={strategy!r} inputs={len(out)} fallbacks={n_fb}")
    return out


# ------------------------------------------------------- learned ranker (CV)
FEATS = ["vote_share", "view_frac", "geo_frac", "aug_gap", "aug_gap_tok", "logmass", "prior_ok"]


def features(cands, query):
    if not cands:
        return np.zeros((0, len(FEATS)))
    tot_votes = sum(c["votes"] for c in cands)
    best_aug = min(c["aug_mean"] for c in cands)
    best_tok = min(c["aug_mean"] / c["n_tok"] for c in cands)
    want = shape_rule(query) if query else None
    allow = allowed_colors(query) if query else None
    rows = []
    for c in cands:
        p_ok = 1.0
        if want is not None and c["shape"] != tuple(want):
            p_ok = 0.0
        if allow is not None and not set(np.unique(c["grid"]).tolist()) <= allow:
            p_ok = 0.0
        rows.append([
            c["votes"] / tot_votes,
            c["n_views"] / c["n_views_decoded"],
            len(c["geos"]) / 8.0,
            -(c["aug_mean"] - best_aug),
            -(c["aug_mean"] / c["n_tok"] - best_tok) * 100.0,
            np.log(max(c["mass"], 1e-6)),
            p_ok,
        ])
    return np.asarray(rows, dtype=float)


def fit_logreg(X, y, l2=1.0, iters=500, lr=0.5):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    w, b = np.zeros(Z.shape[1]), 0.0
    pos = max(y.mean(), 1e-6)
    sw = np.where(y > 0, 0.5 / pos, 0.5 / (1 - pos))     # class-balanced
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(Z @ w + b)))
        g = (p - y) * sw
        w -= lr * (Z.T @ g / len(y) + l2 * w / len(y))
        b -= lr * g.mean()
    f = lambda Xn: ((Xn - mu) / sd) @ w + b        # noqa: E731
    f.params = {"feats": list(FEATS), "mu": mu.tolist(), "sd": sd.tolist(), "w": w.tolist(), "b": float(b)}
    return f, w


def make_learned(params):
    """Frozen ranker (from --export-ranker) as a strategy f(guesses, query)."""
    if not params or params.get("feats") != FEATS:
        raise ValueError("ranker params missing or built for a different feature set")
    mu, sd, w, b = (np.asarray(params[k], dtype=float) for k in ("mu", "sd", "w", "b"))

    def f(guesses, query=None):
        cands = group(guesses)
        if not cands:
            return []
        s = ((features(cands, query) - mu) / sd) @ w + b
        return [cands[i]["grid"] for i in np.argsort(-s, kind="stable")]
    return f


def fit_ranker(decoded, gold, queries):
    """Fit on ALL labelled inputs (ship this only if the LOTO-CV row passed the ship rule)."""
    X, y = [], []
    for bk, guesses in decoded.items():
        if bk in gold and guesses:
            cands = group(guesses)
            X.append(features(cands, queries.get(bk)))
            y.append(np.array([float(np.array_equal(c["grid"], gold[bk])) for c in cands]))
    if not X or sum(v.sum() for v in y) == 0:
        raise ValueError("no positive examples to fit on")
    f, _ = fit_logreg(np.vstack(X), np.concatenate(y))
    return f.params


def learned_loto(decoded, gold, queries):
    """Leave-one-TASK-out CV of a 7-feature logistic ranker. Returns
    {base_key: [grid, ...]} predicted by a model that never saw that task."""
    per = {}
    for bk, guesses in decoded.items():
        if bk not in gold:
            continue
        cands = group(guesses)
        X = features(cands, queries.get(bk))
        y = np.array([float(np.array_equal(c["grid"], gold[bk])) for c in cands])
        per[bk] = (cands, X, y)
    tasks = sorted({bk.split("_")[0] for bk in per})
    out, weights = {}, []
    for t in tasks:
        tr = [v for bk, v in per.items() if bk.split("_")[0] != t and len(v[0])]
        if not tr or sum(v[2].sum() for v in tr) == 0:
            continue
        f, w = fit_logreg(np.vstack([v[1] for v in tr]), np.concatenate([v[2] for v in tr]))
        weights.append(w)
        for bk, (cands, X, _) in per.items():
            if bk.split("_")[0] == t and len(cands):
                s = f(X)
                out[bk] = [cands[i]["grid"] for i in np.argsort(-s, kind="stable")]
    return out, (np.mean(weights, 0) if weights else None)


# ------------------------------------------------------------------ benchmark
def load_candidates(store):
    decoded = {}
    for fname in sorted(os.listdir(store)):
        path = os.path.join(store, fname)
        if not os.path.isfile(path):
            continue
        try:
            with bz2.BZ2File(path, "r") as f:
                outputs = pickle.load(f)
        except Exception:
            continue
        bk = fname.split(".")[0]
        decoded.setdefault(bk, {})
        for i, sample in enumerate(outputs):
            decoded[bk][f"{fname}.out{i}"] = sample
    return decoded


def load_eval():
    ch = json.load(open(f"{BASE}/arc-agi_evaluation_challenges.json"))
    so = json.load(open(f"{BASE}/arc-agi_evaluation_solutions.json"))
    gold, queries, weight, tclass = {}, {}, {}, {}
    for k, replies in so.items():
        for i, r in enumerate(replies):
            bk = f"{k}_{i}"
            gold[bk] = np.asarray(r)
            queries[bk] = {"train": ch[k]["train"], "test": [ch[k]["test"][i]]}
            weight[bk] = 1 / len(replies)
            ti = ch[k]["test"][i]["input"]
            ai, ao = len(ti) * len(ti[0]), len(r) * len(r[0])
            tclass[bk] = "same" if ao == ai else "extract" if ao < ai else "larger"
    return gold, queries, weight, tclass


def hit(order, g, n=2):
    return any(np.array_equal(g, c) for c in order[:n])


def bench(store):
    gold, queries, weight, tclass = load_eval()
    decoded = load_candidates(store)
    print(f"loaded {len(decoded)}/{len(gold)} test inputs with >=1 candidate from {store}")

    # coverage / starvation, straight from the files (16 views expected per test input)
    files = Counter(f.split(".")[0] for f in os.listdir(store))
    hist = Counter(min(files.get(bk, 0), VIEWS_PER_TEST) for bk in gold)
    print("views with >=1 parseable beam per test input (of 16):",
          " ".join(f"{k}:{hist[k]}" for k in sorted(hist)))
    geo_files = defaultdict(set)
    for f in os.listdir(store):
        geo_files[f.split(".")[0]].add(geo_of(f))
    no_tr = sum(1 for bk in gold if bk in files and not any("transpose" in g for g in geo_files[bk]))
    print(f"test inputs with rotation-family files but NO transpose-family file: {no_tr} "
          f"(transpose batches run last -> the 1200s/task cap hits them first)")

    # generation vs selection split
    tot = defaultdict(float)
    orc = defaultdict(float)
    rank_hist = Counter()
    base_order = {}
    for bk, g in gold.items():
        tot[tclass[bk]] += weight[bk]
        order = STRATEGIES["kgmon"](decoded.get(bk, {}), queries[bk])
        base_order[bk] = order
        r = next((i for i, c in enumerate(order) if np.array_equal(g, c)), None)
        if r is not None:
            orc[tclass[bk]] += weight[bk]
            rank_hist[min(r, 5)] += 1
    T, O = sum(tot.values()), sum(orc.values())
    print(f"\noracle (gold among candidates): {O:.2f}/{T:.0f}   " +
          " ".join(f"[{cl} {orc[cl]:.1f}/{tot[cl]:.1f}]" for cl in sorted(tot)))
    print("kgmon rank of gold when present (0-based, 5 = 5+):",
          " ".join(f"{k}:{rank_hist[k]}" for k in sorted(rank_hist)))
    in_top2 = rank_hist[0] + rank_hist[1]
    print(f"-> selection headroom = {sum(rank_hist.values()) - in_top2} test outputs "
          f"(gold generated but outside kgmon top-2); everything else is a generation miss")

    diag = gen_diagnostics(decoded, gold, tclass)

    base_hits = {bk for bk, g in gold.items() if hit(base_order[bk], g)}
    print(f"\n{'strategy':26s} score   vs kgmon  per class")
    results = {}

    def report(name, orders):
        hits = {bk for bk, g in gold.items() if hit(orders.get(bk, []), g)}
        sc = sum(weight[bk] for bk in hits)
        cl = defaultdict(float)
        for bk in hits:
            cl[tclass[bk]] += weight[bk]
        gained, lost = sorted(hits - base_hits), sorted(base_hits - hits)
        results[name] = (sc, gained, lost)
        print(f"{name:26s} {sc:6.2f}  +{len(gained)}/-{len(lost)}    " +
              " ".join(f"[{c} {cl[c]:.1f}]" for c in sorted(tot)) +
              (f"  gained={gained} lost={lost}" if (gained or lost) and len(gained) + len(lost) <= 8 else ""))

    for name, fn in STRATEGIES.items():
        report(name, {bk: fn(decoded.get(bk, {}), queries[bk]) for bk in gold})
    lo, w = learned_loto(decoded, gold, queries)
    if w is not None:
        report("learned (LOTO-CV)", lo)
        print("  mean standardized weights:", " ".join(f"{n}={x:+.2f}" for n, x in zip(FEATS, w)))
    print("\nship rule: a strategy must gain >=2 test outputs over kgmon with 0 lost on this run; "
          "1-output differences are inside decode noise.")
    ok = [(len(g), sc, n) for n, (sc, g, l) in results.items() if len(g) >= 2 and not l]
    # prefer the most gains; among equals a parameter-free strategy over the fitted ranker
    ok.sort(key=lambda t: (-t[0], t[2].startswith("learned"), t[2]))
    winner = ok[0][2] if ok else None
    headroom = sum(rank_hist.values()) - in_top2
    print(f"VERDICT: {'ship ' + repr(winner) if winner else 'keep kgmon (no strategy passed the ship rule)'}"
          f" | selection headroom {headroom} test outputs")
    verdict = {"winner": winner, "passing": [t[2] for t in ok], "headroom": headroom,
               "kgmon": results["kgmon"][0], "oracle": O, "diagnostics": diag,
               "scores": {n: {"score": sc, "gained": g, "lost": l} for n, (sc, g, l) in results.items()}}
    if winner == "learned (LOTO-CV)":
        verdict["ranker"] = fit_ranker(decoded, gold, queries)
    with open(os.path.join(os.path.dirname(os.path.abspath(store.rstrip("/"))), "select_verdict.json"), "w") as fh:
        json.dump(verdict, fh, indent=1)
    return verdict


def gen_diagnostics(decoded, gold, tclass):
    """Where generation fails, to choose between generation-side fixes.
    (a) zero-candidate inputs  -> DFS pruned everything: greedy fallback / looser threshold
    (b) gold beams piled up near the 1.609 cut -> correct answers are being censored by the cut
    (c) oracle by output length -> the p>0.2 cut needs ~0.997/token on a 500-token grid"""
    print("\ngeneration diagnostics")
    zero = Counter(tclass[bk] for bk in gold if not decoded.get(bk))
    wrong_only = Counter()
    bins = Counter()
    by_len = defaultdict(lambda: [0, 0, 0])            # bucket -> [gold present, zero cands, total]
    for bk, g in gold.items():
        n_tok = g.shape[0] * (g.shape[1] + 1)
        b = "<=150" if n_tok <= 150 else "151-400" if n_tok <= 400 else ">400"
        by_len[b][2] += 1
        cands = decoded.get(bk, {})
        if not cands:
            by_len[b][1] += 1
            continue
        nll = [s["beam_score"] for s in cands.values() if np.array_equal(s["solution"], g)]
        if not nll:
            wrong_only[tclass[bk]] += 1
            continue
        by_len[b][0] += 1
        m = min(nll)
        bins["p>=.8" if m < 0.223 else "p .5-.8" if m < 0.693 else "p .3-.5" if m < 1.204 else "p .2-.3"] += 1
    print("  (a) inputs with ZERO candidates, by class:", dict(zero) or 0,
          "| candidates but none correct:", dict(wrong_only) or 0)
    print("  (b) best-view probability of the gold grid when generated:",
          " ".join(f"{k}:{bins[k]}" for k in ("p>=.8", "p .5-.8", "p .3-.5", "p .2-.3")),
          "(mass in the last two bins => the 0.2 cut is hiding more of the same)")
    print("  (c) by gold output length (tokens): " +
          "  ".join(f"{b}: gold {v[0]}/{v[2]}, zero-cand {v[1]}" for b, v in
                    sorted(by_len.items(), key=lambda kv: ("<=150", "151-400", ">400").index(kv[0]))))
    # (d) consensus: how many decode views back the best-supported grid, split by whether gold was generated
    low_with_gold = low_without = 0
    for bk, g in gold.items():
        cands = decoded.get(bk, {})
        if not cands:
            continue
        top = max(c["n_views"] for c in group(cands))
        if top < 3:
            if any(np.array_equal(s["solution"], g) for s in cands.values()):
                low_with_gold += 1
            else:
                low_without += 1
    print(f"  (d) inputs whose best grid has <3 agreeing views: {low_with_gold} with gold generated, "
          f"{low_without} without")
    return {"n_inputs": len(gold), "zero_cand": sum(zero.values()), "zero_by_class": dict(zero),
            "wrong_only": sum(wrong_only.values()), "gold_bins": {k: bins[k] for k in ("p>=.8", "p .5-.8", "p .3-.5", "p .2-.3")},
            "by_len": {b: {"gold": v[0], "zero": v[1], "total": v[2]} for b, v in by_len.items()},
            "low_consensus_with_gold": low_with_gold, "low_consensus_without_gold": low_without}


# ------------------------------------------------------- prior precision check
def priors_report():
    gold, queries, weight, tclass = load_eval()
    fired = right = 0
    cfire = 0
    by = defaultdict(lambda: [0, 0])
    for bk, g in gold.items():
        want = shape_rule(queries[bk])
        if want is not None:
            fired += 1
            ok = tuple(want) == g.shape
            right += ok
            by[tclass[bk]][0] += 1
            by[tclass[bk]][1] += ok
        if set(np.unique(g).tolist()) <= allowed_colors(queries[bk]):
            cfire += 1
    print(f"shape rule fires on {fired}/{len(gold)} test outputs, correct on {right} "
          f"(precision {right / max(fired, 1) * 100:.1f}%)")
    for cl, (f, r) in sorted(by.items()):
        print(f"  class {cl:7s}: fired {f}, correct {r}")
    print(f"colour prior (gold colours within train-output + test-input colours): "
          f"holds on {cfire}/{len(gold)}")


# ------------------------------------------------------------------- selftest
def selftest():
    """Mechanics only: prod-equivalence of kgmon/probmul and key parsing."""
    sys.path.insert(0, os.path.join(os.path.dirname(BASE), "submit_ag2_perfpatch", "out"))
    import arc_decoder as prod
    rng = np.random.default_rng(0)
    geos = ["", ".rot90", ".rot90.rot90", ".rot90.rot90.rot90", ".transpose",
            ".transpose.rot90", ".transpose.rot90.rot90", ".transpose.rot90.rot90.rot90"]
    for trial in range(200):
        grids = [rng.integers(0, 10, size=(rng.integers(1, 6), rng.integers(1, 6))) for _ in range(5)]
        augs = [rng.uniform(0, 20, 8).tolist() for _ in grids]
        guesses = {}
        for gi, geo in enumerate(geos):
            for p in range(2):
                fname = f"abcd1234_0{geo}.permute{''.join(map(str, rng.permutation(10)))}.ex{p}12"
                for j in range(rng.integers(0, 4)):
                    k = int(rng.integers(0, len(grids)))
                    guesses[f"{fname}.out{j}"] = dict(beam_score=float(rng.uniform(0, MAX_SCORE)),
                                                      score_aug=augs[k], solution=grids[k])
        for mine, theirs in (("kgmon", prod.score_kgmon), ("probmul_3", prod.score_full_probmul_3)):
            a, b = STRATEGIES[mine](guesses), theirs(guesses)
            assert len(a) == len(b) and all(np.array_equal(x, y) for x, y in zip(a, b)), (trial, mine)
        for name, fn in STRATEGIES.items():
            out = fn(guesses, None)
            assert len({hashable(o) for o in out}) == len(out) == len(group(guesses)), name
        # kernel entry point: string strategy == production, broken selector falls back, learned round-trips
        dec = {"abcd1234_0": guesses}
        if guesses:
            ref = prod.score_kgmon(guesses)
            for strat in ("kgmon", lambda g, q=None: 1 / 0, lambda g, q=None: []):
                got = run_selection(dec, {}, strat, fallback=prod.score_kgmon)["abcd1234_0"]
                assert all(np.array_equal(x, y) for x, y in zip(got, ref)) and len(got) == len(ref)
            params = {"feats": list(FEATS), "mu": [0.0] * len(FEATS), "sd": [1.0] * len(FEATS),
                      "w": [1.0] + [0.0] * (len(FEATS) - 1), "b": 0.0}
            lr = run_selection(dec, {}, "learned", ranker=json.loads(json.dumps(params)))["abcd1234_0"]
            assert len(lr) == len(ref)
    assert geo_of("abcd1234_0.transpose.rot90.permute0123456789.ex012.out3") == "transpose.rot90"
    assert geo_of("abcd1234_0.permute0123456789.ex012.out0") == "id"
    assert view_of("abcd1234_0.rot90.permute0123456789.ex012.out3") == "abcd1234_0.rot90.permute0123456789.ex012"
    print("selftest ok: kgmon/probmul_3 identical to production on 200 random cases; "
          f"{len(STRATEGIES)} strategies return a full permutation of the candidates")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    if sys.argv[1] == "--selftest":
        import contextlib
        import io
        with contextlib.redirect_stdout(io.StringIO()) as buf:      # run_selection is chatty by design
            selftest()
        print(buf.getvalue().strip().splitlines()[-1])
    elif sys.argv[1] == "--priors":
        priors_report()
    else:
        bench(sys.argv[1])
