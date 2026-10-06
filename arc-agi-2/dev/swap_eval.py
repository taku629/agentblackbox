"""Evaluation protocol for a model swap: pick the task panel, compare two runs.

    python3 swap_eval.py panel   <baseline inference_outputs> [--n 24] [--timing timing.log]
    python3 swap_eval.py compare <baseline inference_outputs> <candidate inference_outputs>
                                 [--base-nll DIR] [--cand-nll DIR] [--name xcalibur]
    python3 swap_eval.py union   <model A inference_outputs> <model B inference_outputs>
    python3 swap_eval.py --selftest

Three stages, cheapest first (a 12 h full run is the last step, not the first):

  smoke   the 4 production whitelist tasks, unchanged. Purpose: the pipeline
          runs on the new checkpoint, memory and time match the baseline log
          ('allocated ...MB for training', per-task TOTAL). Not a quality test:
          4 tasks cannot separate two models.
  panel   `panel` picks N tasks stratified by what the baseline did with them
          (generated / near / mid / far-or-shape / zero), spread over prompt
          size inside each stratum. The generated tasks are the regression
          guard; the misses are where a better model has to show up.
  full    all 120, only if the panel verdict is `promote`.

`compare` restricts both runs to the tasks the candidate run decoded and
prints the per-input transition table plus the verdict.
"""
import math
import os
import sys
from collections import Counter, defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import gen_autopsy  # noqa: E402
import select_v2  # noqa: E402

SMOKE = ["0934a4d8", "36a08778", "981571dc", "aa4ec2a5"]          # production whitelist: baseline numbers exist
SEVERITY = ["generated", "near", "mid", "far_or_shape", "zero"]
# share of the panel per stratum: a quarter regression guard, the rest in proportion to where misses are
QUOTA = {"generated": 0.25, "near": 0.33, "mid": 0.21, "far_or_shape": 0.17, "zero": 0.04}


def input_kind(row):
    if row["outcome"] == "generated":
        return "generated"
    if row["outcome"] == "zero_cand":
        return "zero"
    k = gen_autopsy.miss_kind(row)
    return "near" if k.startswith("near") else "mid" if k.startswith("mid") else "far_or_shape"


def task_kinds(rows):
    """task -> its worst input kind (a task counts as generated only if every test input was)."""
    by = defaultdict(list)
    for r in rows:
        by[r["bk"].split("_")[0]].append(r)
    out = {}
    for task, rs in by.items():
        kind = max((input_kind(r) for r in rs), key=SEVERITY.index)
        out[task] = (kind, max(r["prompt_tokens"] for r in rs), rs[0].get("task_seconds"))
    return out


def pick_panel(rows, n=24):
    kinds = task_kinds(rows)
    strata = defaultdict(list)
    for task, (kind, size, _) in kinds.items():
        strata[kind].append((size, task))
    want = {k: max(1, round(QUOTA[k] * n)) if strata[k] else 0 for k in SEVERITY}
    while sum(want.values()) > n:                                   # trim the largest stratum first
        k = max(want, key=want.get)
        want[k] -= 1
    panel = []
    for k in SEVERITY:
        pool = sorted(strata[k])
        m = min(want[k], len(pool))
        if m:                                                       # evenly spaced over prompt size
            idx = sorted({round(i * (len(pool) - 1) / max(m - 1, 1)) for i in range(m)}) if m > 1 else [len(pool) // 2]
            panel += [(pool[i][1], k) for i in idx]
    spare = sorted((kinds[t][1], t) for t in kinds if t not in {p[0] for p in panel} and kinds[t][0] != "generated")
    while len(panel) < n and spare:                                 # top up from the misses if a stratum was short
        _, t = spare.pop(len(spare) // 2)
        panel.append((t, kinds[t][0]))
    return panel, kinds


def cmd_panel(store, n, timing=None):
    rows = gen_autopsy.build_rows(store, timing=timing)
    panel, kinds = pick_panel(rows, n)
    c = Counter(k for _, k in panel)
    print(f"panel of {len(panel)} tasks: " + ", ".join(f"{k} {c[k]}" for k in SEVERITY if c[k]))
    secs = [kinds[t][2] for t, _ in panel if kinds[t][2]]
    if secs:
        print(f"baseline time for these tasks: {sum(secs) / 3600:.1f} GPU-h -> ~{sum(secs) / 4 / 3600:.1f} h on 4 workers")
    else:
        print(f"no timing.log given: at the 1200 s cap this is at most {len(panel) * 1200 / 4 / 3600:.1f} h on 4 workers")
    for k in SEVERITY:
        ids = [t for t, kk in panel if kk == k]
        if ids:
            print(f"  {k:13s} {' '.join(ids)}")
    print("\n--whitelist " + ",".join(sorted(t for t, _ in panel)))
    return panel


def cmd_compare(base_store, cand_store, base_nll=None, cand_nll=None, name="candidate"):
    base = {r["bk"]: r for r in gen_autopsy.build_rows(base_store, gold_nll=base_nll)}
    cand_all = gen_autopsy.build_rows(cand_store, gold_nll=cand_nll)
    tasks = {f.split("_")[0] for f in os.listdir(cand_store)}
    nll_tasks = {bk.split("_")[0] for bk in gen_autopsy.parse_gold_nll(cand_nll)} if cand_nll else set()
    tasks |= nll_tasks                                              # a task with zero candidates still counts if probed
    cand = {r["bk"]: r for r in cand_all if r["bk"].split("_")[0] in tasks}
    if not cand:
        sys.exit("candidate run has no decoded tasks")
    if not nll_tasks:
        print("NOTE: without --cand-nll, a panel task where the candidate produced NO candidates is invisible here "
              "(no files) -- run the candidate with --probe-gold so every decoded task is counted.")
    keys = sorted(cand)
    trans = Counter((input_kind(base[k]), input_kind(cand[k])) for k in keys)
    gained = [k for k in keys if base[k]["outcome"] != "generated" and cand[k]["outcome"] == "generated"]
    lost = [k for k in keys if base[k]["outcome"] == "generated" and cand[k]["outcome"] != "generated"]
    b_gen = sum(base[k]["outcome"] == "generated" for k in keys)
    c_gen = sum(cand[k]["outcome"] == "generated" for k in keys)
    print(f"{len(keys)} test inputs from {len({k.split('_')[0] for k in keys})} tasks decoded by both")
    print(f"generated: baseline {b_gen}  ->  {name} {c_gen}   (gained {len(gained)}, lost {len(lost)})")
    print(f"\nbaseline (rows) -> {name} (columns)")
    print("  " + " " * 13 + " ".join(f"{k:>12s}" for k in SEVERITY))
    for a in SEVERITY:
        if any(trans[(a, b)] for b in SEVERITY):
            print(f"  {a:13s}" + " ".join(f"{trans[(a, b)]:12d}" for b in SEVERITY))
    if gained:
        print("  gained:", " ".join(gained))
    if lost:
        print("  lost:  ", " ".join(lost))

    still = [k for k in keys if base[k]["outcome"] == "wrong_only" and cand[k]["outcome"] == "wrong_only"
             and base[k].get("shape_ok") and cand[k].get("shape_ok")]
    if still:
        d = np.array([cand[k]["best_cell_acc"] - base[k]["best_cell_acc"] for k in still])
        print(f"\ninputs missed by both with the right shape ({len(still)}): best-candidate cell accuracy "
              f"moved {d.mean():+.3f} on average ({int((d > 0).sum())} up, {int((d < 0).sum())} down)")
    dn = None
    both = [k for k in keys if "gold_nll_min" in base[k] and "gold_nll_min" in cand[k]]
    if both:
        dn = np.array([cand[k]["gold_nll_min"] - base[k]["gold_nll_min"] for k in both])
        miss = np.array([base[k]["outcome"] != "generated" for k in both])
        print(f"\ngold NLL (best view), {name} minus baseline, {len(both)} inputs: median {np.median(dn):+.2f} nats "
              f"({int((dn < 0).sum())} closer, {int((dn > 0).sum())} further)")
        if miss.any():
            print(f"  on the {int(miss.sum())} inputs the baseline missed: median {np.median(dn[miss]):+.2f} nats")
        print("  (this is the direct measurement of how far a small amount of extra SFT moves the model)")

    net = len(gained) - len(lost)
    if len(lost) > 1 and net <= 0:
        verdict, why = "reject", f"lost {len(lost)} previously generated inputs for {len(gained)} gained"
    elif net >= 2 and len(lost) <= 1:
        verdict, why = "promote", f"net {net:+d} generated inputs with at most one regression -> run the full 120"
    elif net >= 1 or (dn is not None and np.median(dn) < -0.5):
        verdict, why = "inconclusive", ("small or indirect gain (net %+d%s) -> extend the panel before a full run"
                                        % (net, "" if dn is None else ", median gold NLL %+.2f" % np.median(dn)))
    else:
        verdict, why = "reject", f"net {net:+d}: no evidence the checkpoint generates more"
    print(f"\nVERDICT: {verdict} -- {why}")
    return {"verdict": verdict, "gained": gained, "lost": lost, "n": len(keys), "transitions": dict(trans)}


TAG2 = ".runm2"


def cmd_union(a_store, b_store, name="B"):
    """Would decoding with BOTH models beat either alone? Uses two single-model runs on the same
    tasks (e.g. the sft139 panel and the xcalibur panel) and replays selection on the merged pools.
    `fallback` is what make_union_kernel.py builds: model B only on tasks model A left unsettled."""
    gold, queries, _, _ = select_v2.load_eval()
    A, B = select_v2.load_candidates(a_store), select_v2.load_candidates(b_store)
    tasks = {k.split("_")[0] for k in A} & {k.split("_")[0] for k in B}
    if not tasks:
        tasks = {f.split("_")[0] for f in os.listdir(b_store)}
    keys = sorted(k for k in gold if k.split("_")[0] in tasks)
    if not keys:
        sys.exit("the two runs share no decoded task")
    kg = select_v2.STRATEGIES["kgmon"]

    def tag(d):
        return {k.replace(".out", TAG2 + ".out"): v for k, v in d.items()}

    def top_views(d):
        return max((c["n_views"] for c in select_v2.group(d)), default=0)

    unsettled_task = {k.split("_")[0] for k in keys if top_views(A.get(k, {})) < 3}
    rows = []
    for k in keys:
        a, b = A.get(k, {}), B.get(k, {})
        u = {**a, **tag(b)}
        f = u if k.split("_")[0] in unsettled_task else a
        g = gold[k]
        gen = lambda d: any(np.array_equal(x["solution"], g) for x in d.values())      # noqa: E731
        rows.append(dict(k=k, genA=gen(a), genB=gen(b), pA=select_v2.hit(kg(a), g), pB=select_v2.hit(kg(b), g),
                         pU=select_v2.hit(kg(u), g), pF=select_v2.hit(kg(f), g),
                         unsettled=k.split("_")[0] in unsettled_task))
    n = len(rows)
    c = lambda key: sum(r[key] for r in rows)                                          # noqa: E731
    b_only = [r for r in rows if r["genB"] and not r["genA"]]
    a_only = [r for r in rows if r["genA"] and not r["genB"]]
    print(f"{n} test inputs from {len(tasks)} tasks decoded by both models")
    print(f"gold generated: A {c('genA')}, {name} {c('genB')}, either {sum(r['genA'] or r['genB'] for r in rows)} "
          f"(only A {len(a_only)}, only {name} {len(b_only)})")
    print(f"pass@2 with kgmon: A {c('pA')}, {name} {c('pB')}, full union {c('pU')}, fallback union {c('pF')}")
    hurt = [r["k"] for r in rows if r["pA"] and not r["pF"]]
    reach = sum(r["unsettled"] for r in b_only)
    share = len(unsettled_task) / len(tasks)
    print(f"tasks model A left unsettled (<3 agreeing views on some input): {len(unsettled_task)}/{len(tasks)} "
          f"({share * 100:.0f}%) -> pass 2 costs about that share of one model's run time")
    print(f"inputs only {name} generated: {len(b_only)}, of which {reach} are in unsettled tasks (reachable by the fallback union)")
    if hurt:
        print("inputs model A got right that the fallback union loses:", " ".join(hurt))
    gain = c("pF") - c("pA")
    if gain >= 2 and len(hurt) <= 1:
        verdict = f"build the union kernel: fallback union gains {gain} input(s) over A with {len(hurt)} regression(s)"
    elif c("pB") > c("pA") and c("pB") >= c("pF"):
        verdict = f"swap, do not union: {name} alone ({c('pB')}) is at least as good as the union ({c('pF')}) for half the time"
    else:
        verdict = f"no union: fallback union {c('pF')} vs A {c('pA')} (gain {gain}, regressions {len(hurt)})"
    print("VERDICT:", verdict)
    return {"n": n, "pA": c("pA"), "pB": c("pB"), "pU": c("pU"), "pF": c("pF"), "b_only": len(b_only),
            "reachable": reach, "unsettled_share": share, "hurt": hurt, "verdict": verdict}


def selftest():
    import bz2
    import contextlib
    import io
    import json
    import pickle
    import tempfile
    gold, queries, _, _ = select_v2.load_eval()
    tasks = sorted({bk.split("_")[0] for bk in gold})
    mk = lambda sol: {"beam_score": 0.1, "score_aug": [1.0] * 8, "solution": np.asarray(sol)}     # noqa: E731

    def write(store, plan, only=None):
        """plan: task index -> kind for every input of that task."""
        os.makedirs(store)
        for i, t in enumerate(tasks):
            if only is not None and t not in only:
                continue
            for bk in [b for b in gold if b.startswith(t + "_")]:
                g = gold[bk]
                kind = plan(i, t)
                if kind == "zero":
                    continue
                if kind == "generated":
                    sol = g
                elif kind == "far_or_shape":
                    sol = np.zeros((1, 1), int)
                else:
                    sol = g.copy()
                    k = max(1, int(g.size * (0.02 if kind == "near" else 0.10)))
                    if kind == "near" and k / g.size > 0.05:
                        sol = np.zeros((1, 1), int)                      # grid too small for a <=5% miss
                    else:
                        sol.flat[:k] = (sol.flat[:k] + 1) % 10
                for v in range(3):
                    with bz2.BZ2File(os.path.join(store, f"{bk}.permute0123456789.ex{v}1"), "w") as f:
                        pickle.dump([mk(sol)], f)

    kinds5 = ["generated", "near", "mid", "far_or_shape", "zero"]
    base_plan = lambda i, t: kinds5[i % 5]                                # noqa: E731
    with tempfile.TemporaryDirectory() as d:
        b = os.path.join(d, "base")
        write(b, base_plan)
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            panel = cmd_panel(b, 24)
        ids = [t for t, _ in panel]
        c = Counter(k for _, k in panel)
        assert len(panel) == 24 and len(set(ids)) == 24, len(panel)
        assert c["generated"] == 6 and c["near"] >= 5 and c["zero"] >= 1 and c["far_or_shape"] >= 3, c
        assert "--whitelist " + ",".join(sorted(ids)) in buf.getvalue()
        rows = gen_autopsy.build_rows(b)
        kinds = task_kinds(rows)
        assert all(kinds[t][0] == k for t, k in panel)
        sizes = sorted(kinds[t][1] for t, k in panel if k == "near")
        allnear = sorted(v[1] for v in kinds.values() if v[0] == "near")
        assert sizes[0] == allnear[0] and sizes[-1] == allnear[-1]       # spread over the size range

        def run(plan, with_nll=False, shift=0.0):
            cdir = os.path.join(d, "c%d" % len(os.listdir(d)))
            write(cdir, plan, only=set(ids))
            kw = {}
            if with_nll:
                for nm, sh in (("bn", 0.0), ("cn", shift)):
                    nd = os.path.join(d, nm + str(len(os.listdir(d))))
                    os.makedirs(nd)
                    with open(os.path.join(nd, "gold_nll_rank0.jsonl"), "w") as f:
                        for bk in gold:
                            if nm == "bn" or bk.split("_")[0] in ids:      # a real candidate run probes only its tasks
                                f.write(json.dumps({"bk": bk, "gold_aug_nll": [12.0 + sh] * 8, "n_tok": 100}) + "\n")
                    kw["base_nll" if nm == "bn" else "cand_nll"] = nd
            out = io.StringIO()
            with contextlib.redirect_stdout(out):
                r = cmd_compare(b, cdir, name="cand", **kw)
            return r, out.getvalue()

        r, _ = run(base_plan)                                            # identical model
        assert r["verdict"] == "reject" and not r["gained"] and not r["lost"] and r["n"] >= 24 - 1
        near_ids = [t for t, k in panel if k == "near"]
        r, txt = run(lambda i, t: "generated" if t in near_ids[:3] else base_plan(i, t))
        assert r["verdict"] == "promote" and len(r["gained"]) >= 3 and not r["lost"], r
        gen_ids = [t for t, k in panel if k == "generated"]
        r, _ = run(lambda i, t: "far_or_shape" if t in gen_ids[:3] else base_plan(i, t))
        assert r["verdict"] == "reject" and len(r["lost"]) >= 3
        r, txt = run(base_plan, with_nll=True, shift=-2.0)               # nothing new generated, but the gold got closer
        assert r["verdict"] == "inconclusive" and "median -2.00 nats" in txt, txt
        one = [t for t in near_ids if sum(b_.startswith(t + "_") for b_ in gold) == 1][:1]
        r, txt = run(lambda i, t: "generated" if t in one else base_plan(i, t))
        assert r["verdict"] == "inconclusive" and len(r["gained"]) >= 1
        zero_ids = [t for t, k in panel if k == "zero"]                  # zero-candidate task counted once probed
        r, txt = run(base_plan, with_nll=True)
        assert r["n"] == sum(len([b_ for b_ in gold if b_.startswith(t + "_")]) for t in ids) and zero_ids
        assert r["transitions"].get(("zero", "zero"), 0) >= 1 and "NOTE: without --cand-nll" not in txt
        assert "NOTE: without --cand-nll" in run(base_plan)[1]
        # union: B generates what A only nearly got; A's settled wins stay A's
        import shutil
        ua, ub = os.path.join(d, "ua"), os.path.join(d, "ub")
        write(ua, base_plan, only=set(ids))
        write(ub, lambda i, t: "generated" if t in near_ids[:4] else ("far_or_shape" if t in gen_ids[:2] else base_plan(i, t)),
              only=set(ids))
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            u = cmd_union(ua, ub, name="xcal")
        assert u["b_only"] >= 4 and u["pU"] >= u["pA"] and u["pF"] >= u["pA"] and not u["hurt"], u
        assert u["pF"] == u["pA"], u                    # every synthetic input has 3 agreeing views -> nothing is unsettled
        assert u["verdict"].startswith(("no union", "swap")) and u["reachable"] == 0
        for f in os.listdir(ua):                         # make A's near misses unsettled: keep a single view each
            if f.split(".")[0].split("_")[0] in near_ids and not f.endswith("ex01"):
                os.remove(os.path.join(ua, f))
        with contextlib.redirect_stdout(out):
            u2 = cmd_union(ua, ub, name="xcal")
        assert u2["reachable"] >= 4 and u2["pF"] >= u2["pA"] + 4 and u2["verdict"].startswith("build the union"), u2
        shutil.rmtree(ua)
    assert pick_panel(rows, 4)[0] and len(pick_panel(rows, 200)[0]) <= len(kinds)
    print("selftest ok: 24-task panel is stratified (6 generated guard + misses by kind) and spread over prompt size; "
          "compare gives reject / promote / inconclusive on identical, better, worse and NLL-only-better candidates; "
          "union replays kgmon on the merged pools and only recommends a build when the fallback design gains")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "--selftest":
        selftest()
    elif a[0] == "panel":
        n = int(a[a.index("--n") + 1]) if "--n" in a else 24
        cmd_panel(a[1], n, a[a.index("--timing") + 1] if "--timing" in a else None)
    elif a[0] == "union":
        cmd_union(a[1], a[2], a[a.index("--name") + 1] if "--name" in a else "B")
    elif a[0] == "compare":
        opt = {k: a[a.index(k) + 1] for k in ("--base-nll", "--cand-nll", "--name") if k in a}
        cmd_compare(a[1], a[2], opt.get("--base-nll"), opt.get("--cand-nll"), opt.get("--name", "candidate"))
    else:
        sys.exit(__doc__)
