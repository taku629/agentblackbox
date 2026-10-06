"""How many hidden tasks does the kernel actually reach in the rerun? Simulation from a measured timing.log.

    python3 rerun_sim.py <timing.log> [--tasks 240] [--seeds 20]
    python3 rerun_sim.py --selftest

Why: the kernel takes tasks in sorted order, 4 workers, and stops at the global end time (12 h - 10 min).
Production gives every task up to 1200 s. The placeholder arc-agi_test_challenges.json has 240 tasks, so
if the hidden rerun is 240 tasks the average budget is ~710 s/task and the tail of the queue may never be
decoded at all (those inputs score as the [[0]] placeholder). This replays measured per-task times
(TTT and TOTAL from an evalscan run) against that budget, for three policies:

  fixed 1200      production: a task takes what it took in the measurement
  faircap 2400    gen_boost faircap (genboost kernel): cap = remaining / my remaining tasks, 600..2400 s
  faircap 1200    union pass 1: same, never above 1200 s; the time it leaves is what a second model gets

A capped task still pays its full TTT; its decode is cut to cap - TTT. The simulation cannot tell how
many correct answers a shorter decode loses -- it reports the decode time kept, not accuracy.
Tasks are drawn with replacement from the measured ones (the hidden tasks are not the measured ones),
so read the output as an estimate with the spread shown.
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
BUDGET = 12 * 3600 - 600
WORKERS = 4


def fair_share(remaining, tasks_left, ceil, floor=600.0):
    return min(ceil, max(floor, remaining / (math.ceil(max(tasks_left, 0) / WORKERS) + 1)))


def simulate(ttt, total, policy, budget=BUDGET):
    """ttt/total: arrays in queue order. -> dict of coverage figures."""
    n = len(total)
    free = [0.0] * WORKERS
    started = decoded = 0
    kept, used = [], 0.0
    for i in range(n):
        w = int(np.argmin(free))
        t = free[w]
        if t >= budget:
            break
        started += 1
        left = n - i - 1
        if policy == "fixed":
            dur = total[i]
        else:
            cap = fair_share(budget - t, left, policy)
            dur = min(total[i], max(cap, ttt[i]))
        dur = min(dur, budget - t)
        dec_have, dec_need = max(0.0, dur - ttt[i]), max(1e-9, total[i] - ttt[i])
        kept.append(min(1.0, dec_have / dec_need))
        decoded += dec_have > 0
        free[w] = t + dur
        used += dur
    return {"never_started": n - started, "no_decode": started - decoded,
            "decode_kept": float(np.mean(kept + [0.0] * (n - started))), "idle_h": (WORKERS * budget - used) / 3600}


def run(ttt_by, tot_by, n_tasks=240, seeds=20, log=print):
    keys = sorted(k for k in tot_by if k in ttt_by)
    if len(keys) < 4:
        raise SystemExit("timing.log has fewer than 4 tasks with both TTT and TOTAL lines")
    T, D = np.array([ttt_by[k] for k in keys], float), np.array([tot_by[k] for k in keys], float)
    log(f"measured {len(keys)} tasks: TOTAL mean {D.mean():.0f}s (median {np.median(D):.0f}), TTT mean {T.mean():.0f}s; "
        f"one model needs {D.mean() * n_tasks / WORKERS / 3600:.1f} h/worker for {n_tasks} tasks, budget {BUDGET / 3600:.1f} h")
    out = {}
    log(f"\n{n_tasks} tasks, {seeds} resamples (mean, [min..max]):")
    log(f"  {'policy':14s} {'never started':>16s} {'TTT but no decode':>20s} {'decode time kept':>18s} {'idle worker-h':>15s}")
    for name, pol in (("fixed 1200", "fixed"), ("faircap 2400", 2400.0), ("faircap 1200", 1200.0)):
        res = []
        for sd in range(seeds):
            idx = np.random.default_rng(sd).integers(0, len(keys), size=n_tasks)
            res.append(simulate(T[idx], D[idx], pol))
        agg = {k: (float(np.mean([r[k] for r in res])), float(min(r[k] for r in res)), float(max(r[k] for r in res)))
               for k in res[0]}
        out[name] = agg
        f = lambda k, p=0: f"{agg[k][0]:.{p}f} [{agg[k][1]:.{p}f}..{agg[k][2]:.{p}f}]"     # noqa: E731
        log(f"  {name:14s} {f('never_started'):>16s} {f('no_decode'):>20s} "
            f"{agg['decode_kept'][0] * 100:>16.0f}% {f('idle_h', 1):>15s}")
    ns = out["fixed 1200"]["never_started"][0]
    if ns >= 1:
        log(f"\n-> with the fixed cap about {ns:.0f} of {n_tasks} tasks ({ns / n_tasks * 100:.0f}%) are never decoded: "
            "coverage (faircap) comes before any lever that spends more time per task, and a second model has no room.")
    else:
        idle = out["faircap 1200"]["idle_h"][0]
        log(f"\n-> every task is reached; about {idle:.1f} worker-hours stay idle under a 1200 s cap -- that is the room "
            f"for a second model ({idle * 3600 / max(D.mean(), 1):.0f} more task-decodes).")
    return out


def selftest():
    quiet = lambda *_: None                                                             # noqa: E731
    # 120 measured tasks of 900 s -> 240 hidden tasks need 15 h/worker in an 11.83 h budget
    ttt = {f"t{i:03d}": 400 for i in range(120)}
    tot = {f"t{i:03d}": 900 for i in range(120)}
    o = run(ttt, tot, 240, 3, quiet)
    exp_started = WORKERS * math.ceil(BUDGET / 900)
    assert o["fixed 1200"]["never_started"][0] == 240 - exp_started and o["fixed 1200"]["idle_h"][0] < 0.5
    assert o["faircap 2400"]["never_started"][0] == 0 and o["faircap 2400"]["no_decode"][0] == 0
    assert 0.5 < o["faircap 2400"]["decode_kept"][0] < 0.75             # everyone decodes, each for less time
    # light tasks: everything fits, the 1200 s cap leaves time over
    o = run({k: 150 for k in ttt}, {k: 400 for k in tot}, 240, 3, quiet)
    assert all(o[p]["never_started"][0] == 0 and o[p]["decode_kept"][0] == 1.0 for p in o)
    assert abs(o["faircap 1200"]["idle_h"][0] - (WORKERS * BUDGET - 240 * 400) / 3600) < 1e-6
    # 120 tasks reproduce the plain arithmetic
    o = run(ttt, tot, 120, 3, quiet)
    assert o["fixed 1200"]["never_started"][0] == 0 and abs(o["fixed 1200"]["idle_h"][0] - (WORKERS * BUDGET - 120 * 900) / 3600) < 1e-6
    # a task whose TTT alone exceeds the cap gets no decode under faircap, and is counted
    r = simulate(np.array([1500.0] * 8), np.array([1600.0] * 8), 1200.0, budget=4000.0)
    assert r["no_decode"] >= 4 and r["decode_kept"] == 0.0
    assert fair_share(42600, 240, 2400.0) == 42600 / 61 and fair_share(100, 240, 2400.0) == 600.0
    print("selftest ok: fixed-cap tail starvation, faircap coverage, idle time under the 1200 s cap and the "
          "TTT-longer-than-cap case match hand arithmetic")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "--selftest":
        selftest()
        sys.exit(0)
    import gen_autopsy
    tot_by, ttt_by = gen_autopsy.parse_timing(a[0])
    run(ttt_by, tot_by, int(a[a.index("--tasks") + 1]) if "--tasks" in a else 240,
        int(a[a.index("--seeds") + 1]) if "--seeds" in a else 20)
