"""Local eval: run a SOLVERS battery over a challenge+solutions json pair.
Usage: python3 eval_local.py [eval|train] [limit] [battery]
"""
import json, sys, time
import numpy as np

def predict_with(solvers, task, max_cands=2):
    pairs = [(p["input"], p["output"]) for p in task["train"]]
    fns = []
    for s in solvers:
        try:
            f = s(pairs)
        except Exception:
            f = None
        if f is not None:
            fns.append((f, s.__name__))
    outs = []
    for t in task["test"]:
        ti = np.asarray(t["input"], dtype=int)
        preds = []
        for f, name in fns:
            try:
                r = np.asarray(f(ti.copy()))
                if r is not None and r.size and 0 < r.shape[0] <= 30 and 0 < r.shape[1] <= 30:
                    preds.append(r.tolist())
            except Exception:
                pass
        uniq = []
        for p in preds:
            if p not in uniq:
                uniq.append(p)
        outs.append(uniq)
    return outs

def run(battery="dsl_all", which="eval", limit=None, verbose_ids=None):
    base = "/home/ubuntu/repos/arc-prize-2026-agent-work/arc-agi-2"
    if battery == "dsl_all":
        sys.path.insert(0, f"{base}/dev")
        from dsl_all import SOLVERS
    else:
        sys.path.insert(0, f"{base}/dev")
        from solvers import SOLVERS
    split = 'evaluation' if which == 'eval' else ('training' if which == 'train' else which)
    ch = json.load(open(f"{base}/arc-agi_{split}_challenges.json"))
    so = json.load(open(f"{base}/arc-agi_{split}_solutions.json"))
    keys = list(ch.keys())[:limit] if limit else list(ch.keys())
    solved = 0
    per_test = 0
    tot_tests = 0
    t0 = time.time()
    misses = []
    for k in keys:
        task = ch[k]
        preds = predict_with(SOLVERS, task)
        gold = so.get(k, [])
        tot_tests += len(gold)
        hit = False
        for i, cand in enumerate(preds):
            if i < len(gold) and any(np.array_equal(np.asarray(c), np.asarray(gold[i])) for c in cand[:2]):
                hit = True
                per_test += 1
        if hit:
            solved += 1
            if verbose_ids:
                print(f"{k}: SOLVED")
        else:
            misses.append(k)
    n = len(keys)
    print(f"{battery}/{which}: {solved}/{n} tasks ({solved/max(n,1)*100:.1f}%), per-test {per_test}/{tot_tests}, {time.time()-t0:.0f}s")
    return misses

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "eval"
    limit = int(sys.argv[2]) if len(sys.argv) > 2 else None
    battery = sys.argv[3] if len(sys.argv) > 3 else "dsl_all"
    m = run(battery, which, limit, verbose_ids=True)
    print("misses:", len(m))
