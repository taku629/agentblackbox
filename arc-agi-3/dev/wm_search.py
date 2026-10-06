"""wm.search(): bounded lookahead on the model when the goal is known but too deep to plan to.

    python3 wm_search.py <bundle>/src/ARC3-Inference [--dry-run]
    python3 wm_search.py --selftest
    python3 wm_search.py --print-prompt

Applies ON TOP of apply_wm_patch.py, wm_round2.py and wm_round3.py. Anchors must match exactly once
or nothing is written; re-running is a no-op.

When it is used: wm.plan() (BFS / A*) either finds the goal or runs out of node budget. For the second
case the model supplies one more function
    def progress(state): ...   # a number, higher = closer to solved
and wm.search() runs a beam search over step():
  * predicted no-ops (step returns the same state) are never expanded -- the model-side version of
    the board_diff no-op signal;
  * a state where is_goal holds ends the search at once;
  * otherwise the sequence with the best predicted progress is played with wm.execute, which stops
    at the first prediction miss; the next call re-plans from the real board (receding horizon);
  * nothing is played unless predicted progress improves on the current state;
  * wm.health() `drop` -> stage 'fallback', no action.
Depth follows the turn budget: with many turns left it looks 4 steps ahead and re-plans often; with
15-35 turns 8 steps; under 15 turns 12 steps per call (fewer, longer commitments). Turns left are
estimated as time_remaining_seconds / 150.
wm.solve() calls it automatically when plan() fails and progress() exists.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARK_SRC = "def search(G"
MARK_PROMPT = "NO PLAN FOUND"

SEARCH = r'''
    def search(G, progress=None, depth=None, beam=8, max_nodes=4000, run=True, turns_left=None):
        """Beam search on the model toward higher progress(state); plays the best sequence."""
        err = missing(G, ("parse", "step"))
        if err:
            return err
        prog = progress if progress is not None else G.get("progress")
        if not callable(prog):
            return {"error": "define progress(state) -> number, higher = closer to solved "
                             "(e.g. minus the count of open targets), then persist it"}
        hv = health(G)
        if hv["verdict"] == "drop":
            return {"stage": "fallback", "health": hv,
                    "do": "stop editing rules for this level; play it directly; wm re-evaluates on the next level"}
        parse, step, is_goal, acts_fn = G["parse"], G["step"], G.get("is_goal"), G.get("actions")
        if turns_left is None:
            tr = (G.get("last_action_result") or {}).get("time_remaining_seconds")
            turns_left = tr / 150.0 if isinstance(tr, (int, float)) else None
        if depth is None:
            depth = 6 if turns_left is None else 4 if turns_left > 35 else 8 if turns_left >= 15 else 12
        s0 = parse(grid_of(G["current_frame"]))
        if not hashable(s0):
            return dict(UNHASHABLE)
        base = [norm(a) for a in G.get("valid_actions", []) if str(a).upper() not in ("RESET", "MOUSE")]
        if not acts_fn and not base:
            return {"error": "no searchable actions: define actions(state) returning candidates "
                             "such as ('MOUSE', row, col) click targets"}
        try:
            p0 = float(prog(s0))
        except Exception as e:
            return {"error": "progress(state) failed: %s: %s" % (e.__class__.__name__, e)}
        visited = {s0}
        frontier = [(p0, 0, s0, [])]
        best = (p0, [])
        nodes, goal, tie = 0, None, 0
        for d in range(int(depth)):
            cand = []
            for _, _, s, path in frontier:
                for a in ([norm(x) for x in acts_fn(s)] if acts_fn else base):
                    if nodes >= max_nodes:
                        break
                    nodes += 1
                    try:
                        s2 = step(s, a)
                        if s2 is None or not hashable(s2) or s2 == s or s2 in visited:
                            continue
                        visited.add(s2)
                        p2 = float(prog(s2))
                        if is_goal is not None and is_goal(s2):
                            goal = path + [a]
                            break
                    except Exception:
                        continue
                    tie += 1
                    cand.append((p2, -tie, s2, path + [a]))
                    if p2 > best[0]:
                        best = (p2, path + [a])
                if goal is not None:
                    break
            if goal is not None or not cand:
                break
            cand.sort(key=lambda c: (c[0], c[1]), reverse=True)
            frontier = cand[:int(beam)]
        out = {"stage": "search", "depth": int(depth), "nodes": nodes, "progress_now": p0}
        if goal is not None:
            path = goal
            out["reaches_goal"] = True
        elif best[1]:
            path = best[1]
            out["progress_predicted"] = best[0]
        else:
            out["improves"] = False
            out["why"] = "no sequence within %d steps raises progress() under the model" % int(depth)
            return out
        out["plan"] = [show(x) for x in path]
        if run:
            out["execute"] = execute(G, path, max_steps=len(path))
        else:
            out["actions"] = path
        return out

'''

WM_EDITS = [
    ('    def solve(G, h=None, max_nodes=20000, min_acc=0.9, last=40):\n',
     SEARCH + '    def solve(G, h=None, max_nodes=20000, min_acc=0.9, last=40):\n'),
    ('        if not p.get("found"):\n'
     '            return {"stage": "plan", "check_acc": c["acc"], "plan": {k: v for k, v in p.items() if k != "actions"}}\n',
     '        if not p.get("found"):\n'
     '            if callable(G.get("progress")) and p.get("why") == "budget":\n'
     '                s = search(G)\n'
     '                s["after"] = "plan() ran out of budget"\n'
     '                return s\n'
     '            return {"stage": "plan", "check_acc": c["acc"], "plan": {k: v for k, v in p.items() if k != "actions"}}\n'),
    ('            "health": health, "hud": hud, "clean": clean, "objects": objects, "explore": explore,\n',
     '            "health": health, "hud": hud, "clean": clean, "objects": objects, "explore": explore, "search": search,\n'),
    ('    for name in ("check", "plan", "execute", "solve", "health", "hud", "clean", "objects", "explore"):\n',
     '    for name in ("check", "plan", "execute", "solve", "health", "hud", "clean", "objects", "explore", "search"):\n'),
]

PROMPT_ANCHOR = '    "- Use `print(...)` for compact summaries, or assign a final compact object to `result`.\\n"\n'

PROMPT_LINES = [
    "- NO PLAN FOUND? If `wm.solve()` returns stage 'plan' with why 'budget', the goal is too deep for an exact "
    "search. Define `progress(state)` -> a number that is higher the closer the state is to solved (for example "
    "minus the number of open targets, or minus the distance from a piece to where it must go) and save it. "
    "`wm.solve()` then falls back to `wm.search()`: a short beam search on your model that plays the sequence with "
    "the best predicted progress and re-plans from the real board next turn. It never plays a sequence that does "
    "not improve `progress`, and it looks further ahead when few turns are left.",
]


def _py_str(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _patch(text, edits, what):
    for anchor, repl in edits:
        n = text.count(anchor)
        if n != 1:
            raise SystemExit(f"{what}: anchor matched {n} times (need exactly 1): {anchor.strip()[:70]!r}")
        text = text.replace(anchor, repl)
    return text


def patch_wm_source(src):
    """round-3 WM_SOURCE -> with wm.search. Idempotent."""
    if MARK_SRC in src:
        return src
    if "def explore(G" not in src:
        raise SystemExit("WM_SOURCE has no wm.explore -- apply wm_round3.py first")
    out = _patch(src, WM_EDITS, "WM_SOURCE")
    compile(out, "<wm>", "exec")
    return out


def patch_prompts(text):
    if MARK_PROMPT in text:
        return text
    block = "".join("    " + _py_str(line + "\n") + "\n" for line in PROMPT_LINES)
    out = _patch(text, [(PROMPT_ANCHOR, block + PROMPT_ANCHOR)], "prompts.py")
    compile(out, "prompts.py", "exec")
    return out


def apply(root, dry_run=False):
    agent = os.path.join(root, "inference", "agent")
    ws_path, pr_path = os.path.join(agent, "wm_source.py"), os.path.join(agent, "prompts.py")
    if not os.path.exists(ws_path):
        raise SystemExit("wm_source.py not found -- run apply_wm_patch.py, wm_round2.py and wm_round3.py first")
    ns = {}
    ws_text = open(ws_path).read()
    exec(compile(ws_text, ws_path, "exec"), ns)
    pr_text = open(pr_path).read()
    new_src, new_pr = patch_wm_source(ns["WM_SOURCE"]), patch_prompts(pr_text)
    if new_src == ns["WM_SOURCE"] and new_pr == pr_text:
        return "already patched"
    if dry_run:
        return "dry run ok: all anchors unique, patched sources compile"
    head = ws_text[:ws_text.index("WM_SOURCE = ")]
    open(ws_path, "w").write(head + "WM_SOURCE = " + repr(new_src) + "\n")
    open(pr_path, "w").write(new_pr)
    return "patched"


def patched_sandbox(tmp, extra=()):
    """Copy bundle_fast's agent dir, apply rounds 1-3 + search (+ extra patchers), import the sandbox."""
    import shutil
    import types
    sys.path.insert(0, HERE)
    import apply_wm_patch
    import wm_round2
    import wm_round3
    root = os.path.join(os.path.dirname(os.path.dirname(HERE)), "bundle_fast", "src", "ARC3-Inference")
    dst = os.path.join(tmp, "ARC3-Inference")
    os.makedirs(os.path.join(dst, "inference"))
    shutil.copytree(os.path.join(root, "inference", "agent"), os.path.join(dst, "inference", "agent"))
    assert apply_wm_patch.apply(dst) == "patched" and wm_round2.apply(dst) == "patched" and wm_round3.apply(dst) == "patched"
    assert apply(dst, dry_run=True).startswith("dry run ok")
    assert apply(dst) == "patched" and apply(dst) == "already patched"
    for fn in extra:
        fn(dst)
    sys.path.insert(0, root)
    for m in [k for k in sys.modules if k.startswith("inference.agent")]:
        del sys.modules[m]
    pkg = types.ModuleType("inference.agent")
    pkg.__path__ = [os.path.join(dst, "inference", "agent")]
    import inference  # noqa: F401
    sys.modules["inference.agent"] = pkg
    from inference.agent import prompts, python_tool_sandbox as sb, wm_source
    return sb, prompts, wm_source, dst


def selftest():
    import tempfile
    sys.path.insert(0, HERE)
    import wm_ref
    import wm_round2 as r2
    import wm_round3 as r3
    with tempfile.TemporaryDirectory() as tmp:
        sb, prompts, wm_source, dst = patched_sandbox(tmp)
        src = patch_wm_source(r3.patch_wm_source(r2.patch_wm_source(wm_ref.WM_SOURCE)))
        assert wm_source.WM_SOURCE == src and patch_wm_source(src) == src
        try:
            patch_wm_source(r2.patch_wm_source(wm_ref.WM_SOURCE))
            raise SystemExit("search applied without round 3")
        except SystemExit as e:
            assert "wm_round3" in str(e)
        pa = prompts.PYTHON_ADDENDUM
        assert pa.index("GOAL UNKNOWN") < pa.index(MARK_PROMPT) < pa.index("- Use `print(...)`") and pa.count(MARK_PROMPT) == 1

        prog = '''
def progress(s):
    p, boxes, targets = s
    if not targets: return 100
    b, t = boxes[0], targets[0]
    return -(abs(b[0] - t[0]) + abs(b[1] - t[1])) * 10 - (abs(p[0] - b[0]) + abs(p[1] - b[1]))
'''
        rules = r2.MOVE_RULES + prog

        # 1. no progress() -> an instruction; nothing played
        g = r2.MoveGame()
        call = r2._harness(sb, g)
        r = call("result = [wm.search(), wm.solve(max_nodes=3)]", r2.MOVE_RULES)
        assert "progress" in r[0]["error"] and r[1]["stage"] == "plan" and g.steps == 0, r

        # 2. depth follows the turn budget; a dry run plays nothing and predicts an improvement
        r = call("result = [wm.search(run=False, turns_left=t)['depth'] for t in (60, 20, 5)] + [wm.search(run=False)]", rules)
        assert r[:3] == [4, 8, 12] and r[3]["depth"] == 6 and g.steps == 0, r
        assert r[3]["progress_predicted"] > r[3]["progress_now"] and r[3]["actions"] and "execute" not in r[3]

        # 3. plan() out of budget + progress() -> solve falls back to search; repeated calls finish the level
        stages, prog_seen = [], []
        for _ in range(12):
            r = call("result = wm.solve(max_nodes=3)", rules)
            stages.append(r["stage"])
            assert r["stage"] == "search" and r.get("after") == "plan() ran out of budget", r
            assert r["execute"]["executed"] >= 1 and r["execute"]["stopped"] in ("plan_done", "terminal_or_level_change"), r
            prog_seen.append(r["progress_now"])
            if r["execute"].get("level_completed"):
                break
        assert g.level == 2 and r.get("reaches_goal"), (g.level, r)
        assert all(b > a for a, b in zip(prog_seen, prog_seen[1:])), prog_seen        # every call started closer than the last
        calls3, steps3 = len(stages), g.steps

        # 4. nothing is played when no sequence improves progress
        g = r2.MoveGame()
        call = r2._harness(sb, g)
        r = call("result = wm.search()", r2.MOVE_RULES + "def progress(s): return 0\n")
        assert r["stage"] == "search" and r["improves"] is False and g.steps == 0, r

        # 5. a wrong model costs one action; a failing progress() is reported, not raised
        g = r2.MoveGame()
        call = r2._harness(sb, g)
        r = call("result = wm.search()", r2.MOVE_RULES_WRONG + prog)
        assert r["execute"]["stopped"] == "diverged" and g.steps == 1, r
        r = call("result = wm.search()", r2.MOVE_RULES + "def progress(s): return 1 / 0\n")
        assert "progress(state) failed" in r["error"] and g.steps == 1

        # 6. hidden state -> health drop -> no action
        g = r2.MoveGame(sticky=3)
        call = r2._harness(sb, g)
        r = call("action(%r)\nresult = wm.search()" % (["RIGHT", "LEFT"] * 3), rules)
        assert r["stage"] == "fallback" and g.steps == 6, r

        # 7. click game through actions(); goal found inside the beam
        g = r2.ClickGame()
        call = r2._harness(sb, g)
        r = call("result = wm.search(turns_left=5)",
                 r2.CLICK_RULES + "def progress(s): return sum(c == 4 for _, _, c in s)\n")
        assert r["stage"] == "search" and r["plan"][0].startswith("MOUSE(") and r["execute"]["executed"] >= 1, r

    print("selftest ok (real sandbox subprocess, rounds 1-3 + search applied to a copy of bundle_fast):")
    print(f"  solve() with a 3-node plan budget fell back to search and cleared the toy level in {calls3} calls / "
          f"{steps3} actions, progress rising every call")
    print("  depth 4 / 8 / 12 for 60 / 20 / 5 turns left; no play without predicted improvement; wrong model costs "
          "1 action; drop verdict and a failing progress() spend nothing")
    print(f"  prompt: {sum(len(l.split()) for l in PROMPT_LINES)} words")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--print-prompt":
        print("\n\n".join(PROMPT_LINES))
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("root")
        ap.add_argument("--dry-run", action="store_true")
        a = ap.parse_args()
        print(apply(a.root, a.dry_run))
