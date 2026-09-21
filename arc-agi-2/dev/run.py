"""Run the solver battery over tasks; score on the dev test set (240 tasks
with known solutions in training_solutions.json)."""
import json, sys, time
import numpy as np

sys.path.insert(0, "/home/takumu/kaggle/arc-agi-2/dev")
from solvers import SOLVERS

def solve_task(task, verbose=False):
    """task: {'train':[...], 'test':[...]}. Returns list of (pred, solver_name)."""
    pairs = [(p["input"], p["output"]) for p in task["train"]]
    sols = []
    for i, s in enumerate(SOLVERS):
        try:
            f = s(pairs)
        except Exception:
            f = None
        if f is not None:
            sols.append((f, s.__name__))
            if verbose:
                print("   hit:", s.__name__)
    return sols

def predict(task):
    """Return list of predictions per test input: [(attempt1, attempt2), ...]."""
    sols = solve_task(task)
    outs = []
    for t in task["test"]:
        ti = np.asarray(t["input"], dtype=int)
        preds = []
        for f, name in sols:
            try:
                r = np.asarray(f(ti.copy()))
                if r is not None and r.size and r.shape[0] <= 30 and r.shape[1] <= 30:
                    preds.append(r.tolist())
            except Exception:
                pass
        # dedup
        uniq = []
        for p in preds:
            if p not in uniq: uniq.append(p)
        outs.append(uniq)
    return outs

def main():
    test = json.load(open("/home/takumu/arc-agi-2/arc-agi_test_challenges.json")) if False else \
           json.load(open("/home/takumu/kaggle/arc-agi-2/arc-agi_test_challenges.json"))
    sol = json.load(open("/home/takumu/kaggle/arc-agi-2/arc-agi_training_solutions.json"))
    n_solved = 0; total = 0; ntask = 0
    t0 = time.time()
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 40
    for k, task in list(test.items())[:limit]:
        ntask += 1
        preds = predict(task)
        gold = sol.get(k, [])
        hit = False
        for i, cand in enumerate(preds):
            if i < len(gold):
                if any(np.array_equal(np.asarray(c), np.asarray(gold[i])) for c in cand[:2]):
                    hit = True
        total += len(gold)
        if hit:
            n_solved += 1
            print(f"{k}: SOLVED")
        else:
            print(f"{k}: miss ({len(preds and preds[0] or [])} cands)")
    print(f"\n{ntask} tasks, solved {n_solved} = {n_solved/max(ntask,1)*100:.1f}%  ({time.time()-t0:.0f}s)")

if __name__ == "__main__":
    main()
