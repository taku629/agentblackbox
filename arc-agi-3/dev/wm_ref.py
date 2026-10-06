"""Reference `wm` skeleton for the persist() mechanism (AGI-3 executable world model).

WM_SOURCE is ONE fragment meant to be persisted once (by the harness as a
built-in preload, or by the model via persist()). It gives every later python
call three verbs so the model only ever rewrites its game-specific rules:

    def parse(grid): ...            # 64x64 int grid -> hashable abstract state (drop the HUD here)
    def step(state, action): ...    # predicted next state; action is 'LEFT' or ('MOUSE', row, col)
    def is_goal(state): ...         # level solved?
    def actions(state): ...         # optional: candidate actions (REQUIRED for MOUSE games)

    wm.check()               -> replay every recorded transition through step(); accuracy + first mismatches
    wm.plan()                -> BFS (or A* with h=) over step(); spends zero real actions
    wm.execute(plan)         -> real actions one at a time, stops at the first prediction miss

Written for the real sandbox, which has no `class`, no `time`, no KeyError/
object/id (measured against bundle_fast python_tool_sandbox.py): closures and
dicts only, `except Exception` only, budgets counted in nodes not seconds.

    python3 arc-agi-3/dev/wm_ref.py      # runs WM_SOURCE inside the real sandbox against a toy game
"""

WM_SOURCE = r'''
def _wm_build():
    import collections, heapq, re

    def grid_of(frame):
        g = getattr(frame, "grid", None)
        return g if g is not None else getattr(frame, "_grid", None)

    def norm(a):
        """'LEFT' | 'MOUSE(row=3, col=4)' | {'action':'MOUSE','row':3,'col':4} -> 'LEFT' | ('MOUSE',3,4)"""
        if isinstance(a, dict):
            name = str(a.get("action", "")).strip().upper()
            return (name, int(a["row"]), int(a["col"])) if "row" in a else name
        if isinstance(a, (tuple, list)):
            return (str(a[0]).upper(), int(a[1]), int(a[2])) if len(a) == 3 else str(a[0]).upper()
        s = str(a).strip()
        m = re.match(r"^(\w+)\s*\(\s*(?:row\s*=\s*)?(-?\d+)\s*,\s*(?:col\s*=\s*)?(-?\d+)\s*\)$", s)
        return (m.group(1).upper(), int(m.group(2)), int(m.group(3))) if m else s.upper()

    def to_env(a):
        a = norm(a)
        return {"action": a[0], "row": a[1], "col": a[2]} if isinstance(a, tuple) else a

    def show(a):
        a = norm(a)
        return "%s(%d,%d)" % a if isinstance(a, tuple) else a

    def diff(pred, actual, limit=4):
        """Compact description of where two abstract states differ."""
        if type(pred) != type(actual):
            return "type %s vs %s" % (type(pred).__name__, type(actual).__name__)
        if isinstance(pred, dict):
            ks = [k for k in sorted(set(pred) | set(actual), key=str) if pred.get(k) != actual.get(k)]
            return {str(k): [pred.get(k), actual.get(k)] for k in ks[:limit]}
        if isinstance(pred, (tuple, list)) and len(pred) == len(actual):
            ix = [i for i in range(len(pred)) if pred[i] != actual[i]]
            return {"idx%d" % i: [pred[i], actual[i]] for i in ix[:limit]}
        if isinstance(pred, (set, frozenset)):
            return {"only_pred": sorted(pred - actual, key=str)[:limit],
                    "only_actual": sorted(actual - pred, key=str)[:limit]}
        return [str(pred)[:80], str(actual)[:80]]

    NEED = ("define parse(grid), step(state, action) and is_goal(state), then persist them; "
            "missing: %s")

    def missing(G, names=("parse", "step", "is_goal")):
        gone = [n for n in names if not callable(G.get(n))]
        return {"error": NEED % ", ".join(gone)} if gone else None

    def hashable(s):
        try:
            hash(s)
            return True
        except Exception:
            return False

    UNHASHABLE = {"error": "parse() must return a hashable state: use tuples / frozensets, not lists / dicts / sets"}

    def check(G, last=None, level=None):
        """Replay recorded transitions through the model. `last=N` checks only
        the newest N; `level=k` only that level. Level-changing transitions
        test is_goal instead of step."""
        err = missing(G, ("parse", "step"))
        if err:
            return err
        parse, step, is_goal = G["parse"], G["step"], G.get("is_goal")
        trs = [t for t in G.get("transitions", []) if t.before_frame is not None]
        if level is not None:
            trs = [t for t in trs if t.before_frame.level == level]
        if last:
            trs = trs[-last:]
        by, miss, errs = {}, [], []
        ok = n = moved = goal_ok = goal_n = false_goal = 0
        seen = set()
        for i, t in enumerate(trs):
            a = norm(t.action)
            name = a[0] if isinstance(a, tuple) else a
            try:
                s0 = parse(grid_of(t.before_frame))
                pred = step(s0, a)
                if t.after_frame.level != t.before_frame.level:
                    if name != "RESET" and is_goal is not None:
                        goal_n += 1
                        goal_ok += bool(is_goal(pred))
                    continue
                s1 = parse(grid_of(t.after_frame))
            except Exception as e:
                if len(errs) < 2:
                    errs.append("%s: %s: %s" % (show(a), e.__class__.__name__, e))
                n += 1
                by.setdefault(name, [0, 0])[1] += 1
                continue
            n += 1
            seen.add(s0 if not isinstance(s0, (dict, list, set)) else str(s0))
            moved += s1 != s0
            hit = pred == s1
            ok += hit
            b = by.setdefault(name, [0, 0])
            b[0] += hit
            b[1] += 1
            if is_goal is not None and is_goal(s1):
                false_goal += 1                      # goal claimed but the level did not advance
            if not hit and len(miss) < 3:
                miss.append({"i": i, "action": show(a), "pred_vs_actual": diff(pred, s1)})
        out = {"n": n, "ok": ok, "acc": round(ok / n, 3) if n else None,
               "by_action": {k: "%d/%d" % (v[0], v[1]) for k, v in sorted(by.items())}}
        if miss:
            out["mismatch"] = miss
        if errs:
            out["errors"] = errs
        if goal_n:
            out["goal"] = "%d/%d level-ups predicted" % (goal_ok, goal_n)
        if false_goal:
            out["false_goal"] = false_goal
        if n >= 4 and (moved == 0 or len(seen) == 1):
            out["warning"] = "parse() maps every recorded board to the same state -- accuracy is vacuous"
        return out

    def plan(G, start=None, h=None, max_nodes=20000, max_depth=80, avoid=None):
        """Shortest action list to is_goal under step(). BFS; A* when a
        heuristic h(state)->number is given. Budget is counted in expanded
        nodes because the sandbox has no clock (keep step() cheap: ~20k nodes
        of a 64x64-cell step can exceed the 30 s tool limit)."""
        err = missing(G)
        if err:
            return err
        parse, step, is_goal = G["parse"], G["step"], G["is_goal"]
        acts_fn = G.get("actions")
        s0 = parse(grid_of(G["current_frame"])) if start is None else start
        if not hashable(s0):
            return dict(UNHASHABLE)
        base = [norm(a) for a in G.get("valid_actions", []) if str(a).upper() not in ("RESET", "MOUSE")]
        if not acts_fn and not base:
            return {"error": "no searchable actions: define actions(state) returning candidates "
                             "such as ('MOUSE', row, col) click targets"}
        if is_goal(s0):
            return {"found": True, "plan": [], "nodes": 0}
        parent = {s0: None}
        tie = 0
        frontier = [(h(s0), 0, tie, s0)] if h else collections.deque([(0, 0, tie, s0)])
        nodes = bad = 0
        best = (h(s0) if h else None, s0)
        while frontier and nodes < max_nodes:
            if h:
                _, g, _, s = heapq.heappop(frontier)
            else:
                _, g, _, s = frontier.popleft()
            nodes += 1
            if g >= max_depth:
                continue
            for a in ([norm(x) for x in acts_fn(s)] if acts_fn else base):
                try:
                    s2 = step(s, a)
                except Exception:
                    bad += 1
                    continue
                if s2 is None:
                    continue
                if not hashable(s2):
                    return dict(UNHASHABLE)
                if s2 in parent or (avoid is not None and avoid(s2)):
                    continue
                parent[s2] = (s, a)
                if is_goal(s2):
                    path = []
                    cur = s2
                    while parent[cur] is not None:
                        cur, act = parent[cur]
                        path.append(act)
                    path.reverse()
                    return {"found": True, "plan": [show(x) for x in path], "actions": path,
                            "nodes": nodes, "states": len(parent)}
                tie += 1
                if h:
                    hv = h(s2)
                    if best[0] is None or hv < best[0]:
                        best = (hv, s2)
                    heapq.heappush(frontier, (g + 1 + hv, g + 1, tie, s2))
                else:
                    frontier.append((0, g + 1, tie, s2))
        out = {"found": False, "nodes": nodes, "states": len(parent),
               "why": "budget" if frontier else "state space exhausted -- the model says the goal is unreachable"}
        if bad:
            out["step_errors"] = bad
        if h and best[1] is not s0:                       # partial plan toward the best-h state
            path, cur = [], best[1]
            while parent[cur] is not None:
                cur, act = parent[cur]
                path.append(act)
            path.reverse()
            out["partial"] = [show(x) for x in path]
            out["actions"] = path
        return out

    def execute(G, acts, max_steps=30, reserve_seconds=8):
        """Run a plan for real, one action per call, comparing each resulting
        board with the prediction. Stops at the first miss so a wrong model
        costs one action, and reports the miss as the next counterexample."""
        err = missing(G, ("parse", "step"))
        if err:
            return err
        if not acts:
            return {"executed": 0, "stopped": "empty_plan"}
        parse, step = G["parse"], G["step"]
        action = G["action"]
        s = parse(grid_of(G["current_frame"]))
        level0 = G["current_frame"].level
        done, t_first = 0, None
        for a in list(acts)[:max_steps]:
            a = norm(a)
            pred = step(s, a)
            res = action([to_env(a)])
            if not res.get("executed", True):
                return {"executed": done, "stopped": res.get("stop_reason", "not_executed"), "at": show(a)}
            done += 1
            if res.get("level_completed") or res.get("done") or res.get("game_over") or res.get("run_complete") \
                    or G["current_frame"].level != level0:
                return {"executed": done, "stopped": "terminal_or_level_change",
                        "level_completed": bool(res.get("level_completed")), "game_over": bool(res.get("game_over"))}
            s = parse(grid_of(G["current_frame"]))
            if s != pred:
                return {"executed": done, "stopped": "diverged", "at": show(a), "pred_vs_actual": diff(pred, s),
                        "remaining": len(acts) - done}
            tr = res.get("time_remaining_seconds")          # only clock available in the sandbox
            if isinstance(tr, (int, float)):
                if t_first is None:
                    t_first = tr
                elif t_first - tr > 30 - reserve_seconds:
                    return {"executed": done, "stopped": "tool_time", "remaining": len(acts) - done}
        left = len(acts) - done
        return {"executed": done, "stopped": "max_steps" if left else "plan_done", "remaining": left}

    def solve(G, h=None, max_nodes=20000, min_acc=0.9, last=40):
        """check -> plan -> execute in one call; refuses to act on a model that
        does not reproduce recent history."""
        c = missing(G) or check(G, last=last)
        if "error" in c:
            return c
        if c["n"] and c["acc"] is not None and c["acc"] < min_acc:
            return {"stage": "check", "check": c}
        p = plan(G, h=h, max_nodes=max_nodes)
        if "error" in p:
            return p
        if not p.get("found"):
            return {"stage": "plan", "check_acc": c["acc"], "plan": {k: v for k, v in p.items() if k != "actions"}}
        e = execute(G, p["actions"])
        return {"stage": "execute", "check_acc": c["acc"], "plan_len": len(p["actions"]), "execute": e}

    return {"check": check, "plan": plan, "execute": execute, "solve": solve,
            "norm": norm, "diff": diff, "grid_of": grid_of}


def _wm_bind(fns):
    # late-bound globals: parse/step/is_goal defined (or redefined) after this
    # fragment are picked up at call time, not at persist time
    def G():
        return _WM_GLOBALS
    ns = {}
    for name in ("check", "plan", "execute", "solve"):
        ns[name] = (lambda f: (lambda *a, **k: f(G(), *a, **k)))(fns[name])
    for name in ("norm", "diff", "grid_of"):
        ns[name] = fns[name]
    return type("wm", (), ns)


wm = _wm_bind(_wm_build())
'''

# The sandbox has no globals(); apply_wm_patch.py hands the namespace in as
# _WM_GLOBALS and preloads WM_SOURCE before the model's code (host patch).


# --------------------------------------------------------------------- test
def _load_patched_sandbox(tmp):
    """Copy the production agent dir, apply apply_wm_patch.py to the copy and
    import the patched python_tool_sandbox from there."""
    import os
    import shutil
    import sys
    import types
    here = os.path.dirname(os.path.abspath(__file__))
    root = os.path.join(os.path.dirname(os.path.dirname(here)), "bundle_fast", "src", "ARC3-Inference")
    dst = os.path.join(tmp, "ARC3-Inference")
    os.makedirs(os.path.join(dst, "inference"))
    shutil.copytree(os.path.join(root, "inference", "agent"), os.path.join(dst, "inference", "agent"))
    sys.path.insert(0, here)
    import apply_wm_patch
    assert apply_wm_patch.apply(dst, dry_run=True).startswith("dry run ok")
    assert apply_wm_patch.apply(dst) == "patched"
    assert apply_wm_patch.apply(dst) == "already patched"
    sys.path.insert(0, root)
    pkg = types.ModuleType("inference.agent")          # skip agent/__init__ (pulls PIL via tool_agent)
    pkg.__path__ = [os.path.join(dst, "inference", "agent")]
    import inference  # noqa: F401
    sys.modules["inference.agent"] = pkg
    from inference.agent import prompts, python_tool_sandbox as sb, wm_source
    assert wm_source.WM_SOURCE == WM_SOURCE
    assert "WORLD MODEL TOOLKIT" in prompts.PYTHON_ADDENDUM and "persist(code)" in prompts.PYTHON_ADDENDUM
    return sb, prompts


class ToyGame:
    """8x8 box-pushing level with a ticking HUD row and a hidden twist (ice:
    a pushed box slides until blocked) the first model does not know about."""
    W = 8

    def __init__(self):
        self.level, self.steps = 1, 0
        self.reset()

    def reset(self):
        self.p, self.box, self.goal = (6, 1), (4, 3), (1, 6)
        self.walls = {(r, c) for r in range(8) for c in range(8) if r in (0, 7) or c in (0, 7)}
        self.walls -= {(0, c) for c in range(8)}          # row 0 is the HUD strip

    def grid(self):
        g = [[0] * 8 for _ in range(8)]
        for r, c in self.walls:
            g[r][c] = 5
        g[self.goal[0]][self.goal[1]] = 3
        g[self.box[0]][self.box[1]] = 2
        g[self.p[0]][self.p[1]] = 1
        g[0] = [9 if c < self.steps % 8 else 0 for c in range(8)]
        return g

    def act(self, name):
        d = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}[name]
        self.steps += 1
        n = (self.p[0] + d[0], self.p[1] + d[1])
        if n in self.walls or n[0] < 1:
            return False
        if n == self.box:
            b = self.box
            while True:                                    # ice: slide until blocked
                nb = (b[0] + d[0], b[1] + d[1])
                if nb in self.walls or nb[0] < 1:
                    break
                b = nb
                if b == self.goal:
                    break
            if b == self.box:
                return False
            self.box = b
        self.p = n
        if self.box == self.goal:
            self.level += 1
            self.reset()
            return True
        return False


def _frame(game):
    g = game.grid()
    return {"ascii": "", "step": game.steps, "level": game.level, "shape": [8, 8], "grid": g}


def run_selftest():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp:
        _run_selftest(tmp)


def _run_selftest(tmp):
    sb, prompts = _load_patched_sandbox(tmp)
    game = ToyGame()
    history = [{"action": "", "frame": _frame(game)}]
    budget = [1000.0]

    def state(last=None):
        return {"current_frame": _frame(game), "history": list(history),
                "valid_actions": ["UP", "DOWN", "LEFT", "RIGHT", "RESET"], "last_action_result": last or {}}

    def handler(actions):
        res = {}
        for a in actions:
            lv = game.act(a["action"])
            budget[0] -= 0.5
            history.append({"action": a["action"], "frame": _frame(game)})
            res = {"executed": True, "level_completed": lv, "time_remaining_seconds": budget[0]}
        return {"action_result": res, "state": state(res)}

    def call(code, fragments=()):
        # `fragments` stands in for persist(): saved code re-run at the head of the call.
        # wm itself is NOT passed here -- it arrives through the host preload.
        r = sb.run_sandboxed_python(code="\n".join(fragments) + "\n" + code, timeout_seconds=30,
                                    initial_state=state(), action_handler=handler)
        assert not r["error"], r["error"]
        return r["result"]

    # guard rails before any rules exist: instructions, not tracebacks, and no real action spent
    g0 = call("result = [wm.check(), wm.plan(), wm.execute(['UP']), wm.solve()]")
    assert all("error" in x and "parse" in x["error"] for x in g0), g0
    assert game.steps == 0
    gm = call("result = [wm.norm('MOUSE(row=3, col=4)'), wm.norm({'action': 'MOUSE', 'row': 3, 'col': 4}), "
              "wm.norm(('mouse', 3, 4)), wm.norm(' left ')]")
    assert gm == [["MOUSE", 3, 4]] * 3 + ["LEFT"], gm          # the three spellings the harness produces / accepts
    g1 = call("result = wm.plan()", ["def parse(g): return [1]\ndef step(s, a): return s\ndef is_goal(s): return False"])
    assert "hashable" in g1["error"], g1

    rules_v1 = '''
def parse(g):
    p = b = t = None
    for r in range(1, 8):                       # row 0 is the HUD strip: ignored
        for c in range(8):
            v = g[r][c]
            if v == 1: p = (r, c)
            elif v == 2: b = (r, c)
            elif v == 3: t = (r, c)
    return (p, b, t)
WALL = frozenset((r, c) for r in range(8) for c in range(8) if r in (0, 7) or c in (0, 7))
D = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
def step(s, a):
    p, b, t = s
    d = D[a]
    n = (p[0] + d[0], p[1] + d[1])
    if n in WALL: return s
    if n == b:
        nb = (b[0] + d[0], b[1] + d[1])
        if nb in WALL: return s
        b = nb                                   # v1 belief: a push moves the box ONE cell
    return (n, b, t)
def is_goal(s):
    return s[1] == s[2] or s[2] is None
'''
    rules_v2 = rules_v1.replace(
        "        b = nb                                   # v1 belief: a push moves the box ONE cell\n",
        "        while True:\n            nb = (b[0] + d[0], b[1] + d[1])\n            if nb in WALL: break\n"
        "            b = nb\n            if b == t: break\n")
    assert rules_v2 != rules_v1

    # turn 1: probe moves, no box contact yet -> model looks perfect
    r1 = call("action(['RIGHT','RIGHT','UP'])\nresult = wm.check()", [rules_v1])
    assert r1["acc"] == 1.0 and r1["n"] == 3, r1
    # turn 2: plan with the wrong (one-cell) push model; execute must stop on the first miss
    r2 = call("p = wm.plan()\ne = wm.execute(p['actions'])\nresult = {'plan': p['plan'], 'exec': e, 'check': wm.check()}",
              [rules_v1])
    assert r2["exec"]["stopped"] == "diverged", r2
    assert r2["check"]["acc"] < 1.0 and r2["check"]["mismatch"], r2
    wasted = r2["exec"]["executed"]
    # turn 3: revised rule (slide) reproduces ALL history, then solve finishes the level
    r3 = call("c = wm.check()\nresult = {'check': c, 'solve': wm.solve()}", [rules_v2])
    assert r3["check"]["acc"] == 1.0, r3
    assert r3["solve"]["stage"] == "execute" and r3["solve"]["execute"].get("level_completed"), r3
    assert game.level == 2
    # vacuous-model guard
    r4 = call("result = wm.check()", ["def parse(g): return 0\ndef step(s, a): return 0\ndef is_goal(s): return False"])
    assert "warning" in r4, r4
    print("selftest ok (real sandbox subprocess):")
    print("  turn1 check:", r1)
    print("  turn2 plan %s -> execute stopped=%s after %d real action(s); mismatch=%s"
          % (r2["plan"], r2["exec"]["stopped"], wasted, r2["check"]["mismatch"][0]["pred_vs_actual"]))
    print("  turn3 check acc=%.2f over %d transitions, solve=%s" % (r3["check"]["acc"], r3["check"]["n"], r3["solve"]))
    print("  goal check:", r3["check"].get("goal"), "| vacuous guard:", r4["warning"])
    block = prompts.PYTHON_ADDENDUM[prompts.PYTHON_ADDENDUM.index("- WORLD MODEL TOOLKIT"):]
    block = block[:block.index("- Use `print(...)`")]
    print("  guard rails: no-rules ->", g0[0]["error"][:60], "| list state ->", g1["error"][:40])
    print("  WM_SOURCE: %d lines, %d chars (host-side preload: costs no model tokens)" % (WM_SOURCE.count(chr(10)), len(WM_SOURCE)))
    print("  prompt block: %d words added to the system prompt" % len(block.split()))


if __name__ == "__main__":
    run_selftest()
