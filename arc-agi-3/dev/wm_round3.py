"""Round 3 for the wm arm: wm.explore() (model-based novelty search) + optional turn-budget guidance.

    python3 wm_round3.py <bundle>/src/ARC3-Inference [--turn-planner] [--dry-run]
    python3 wm_round3.py --selftest
    python3 wm_round3.py --print-prompt [--turn-planner]

Applies ON TOP of apply_wm_patch.py and wm_round2.py (needs wm.health in WM_SOURCE). Anchors must
match exactly once or nothing is written; re-running is a no-op.

wm.explore()  For the phase where step() already reproduces history but the goal is unknown, so
              is_goal cannot be written and wm.solve() has nothing to aim at. It searches the model
              (BFS, zero game actions) for the nearest state this level has NOT been in, then plays
              the path with wm.execute (stops on the first prediction miss). Every call therefore
              either reaches a new situation with the fewest actions the model knows, or produces a
              counterexample for step(). A level-up on the way is the goal showing itself.
              No new state within the node budget -> {'found': False, 'untried_here': [...]}: the
              valid actions never tried in the current state, the cheapest real probes left.
              Respects wm.health(): on a `drop` verdict it returns stage 'fallback' and acts not at all.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARK_SRC = "def explore(G"
MARK_PROMPT = "GOAL UNKNOWN"
MARK_PLANNER = "TURN BUDGET"

EXPLORE = r'''
    def explore(G, max_nodes=5000, max_depth=40, run=True, max_steps=12):
        """Goal unknown: find the nearest state (under step) this level has not been in, and go there."""
        err = missing(G, ("parse", "step"))
        if err:
            return err
        hv = health(G)
        if hv["verdict"] == "drop":
            return {"stage": "fallback", "health": hv,
                    "do": "stop editing rules for this level; play it directly; wm re-evaluates on the next level"}
        parse, step, acts_fn = G["parse"], G["step"], G.get("actions")
        lvl = G["current_frame"].level
        s0 = parse(grid_of(G["current_frame"]))
        if not hashable(s0):
            return dict(UNHASHABLE)
        seen, tried_here = {s0}, set()
        for t in G.get("transitions", []):
            if t.before_frame is None or t.before_frame.level != lvl:
                continue
            try:
                sb = parse(grid_of(t.before_frame))
                if hashable(sb):
                    seen.add(sb)
                    if sb == s0:
                        tried_here.add(norm(t.action))
                if t.after_frame.level == lvl:
                    sa = parse(grid_of(t.after_frame))
                    if hashable(sa):
                        seen.add(sa)
            except Exception:
                pass
        base = [norm(a) for a in G.get("valid_actions", []) if str(a).upper() not in ("RESET", "MOUSE")]
        if not acts_fn and not base:
            return {"error": "no searchable actions: define actions(state) returning candidates "
                             "such as ('MOUSE', row, col) click targets"}
        parent = {s0: None}
        frontier = collections.deque([(s0, 0)])
        nodes, target = 0, None
        while frontier and nodes < max_nodes and target is None:
            s, g = frontier.popleft()
            nodes += 1
            if g >= max_depth:
                continue
            for a in ([norm(x) for x in acts_fn(s)] if acts_fn else base):
                try:
                    s2 = step(s, a)
                except Exception:
                    continue
                if s2 is None or not hashable(s2) or s2 in parent:
                    continue
                parent[s2] = (s, a)
                if s2 not in seen:
                    target = s2
                    break
                frontier.append((s2, g + 1))
        here = [norm(x) for x in acts_fn(s0)] if acts_fn else base
        untried = [show(a) for a in here if a not in tried_here]
        if target is None:
            return {"stage": "explore", "found": False, "nodes": nodes, "seen_states": len(seen),
                    "why": "the model predicts no state this level has not already been in",
                    "untried_here": untried[:12]}
        path, cur = [], target
        while parent[cur] is not None:
            cur, act = parent[cur]
            path.append(act)
        path.reverse()
        out = {"stage": "explore", "found": True, "plan": [show(x) for x in path], "actions": path,
               "nodes": nodes, "seen_states": len(seen)}
        if run:
            out["execute"] = execute(G, path, max_steps=max_steps)
            del out["actions"]
        return out

'''

WM_EDITS = [
    ('    def solve(G, h=None, max_nodes=20000, min_acc=0.9, last=40):\n',
     EXPLORE + '    def solve(G, h=None, max_nodes=20000, min_acc=0.9, last=40):\n'),
    ('            "health": health, "hud": hud, "clean": clean, "objects": objects,\n',
     '            "health": health, "hud": hud, "clean": clean, "objects": objects, "explore": explore,\n'),
    ('    for name in ("check", "plan", "execute", "solve", "health", "hud", "clean", "objects"):\n',
     '    for name in ("check", "plan", "execute", "solve", "health", "hud", "clean", "objects", "explore"):\n'),
]

PROMPT_ANCHOR = '    "- Use `print(...)` for compact summaries, or assign a final compact object to `result`.\\n"\n'

EXPLORE_LINES = [
    "- GOAL UNKNOWN? Once `wm.check()` is accurate but you cannot yet write `is_goal`, call `wm.explore()` "
    "instead of hand-picking probes: it uses your `step` to find the shortest action list to a state this level "
    "has not been in yet and plays it, stopping at the first surprise. A level-up on the way shows you the goal -- "
    "then write `is_goal` and switch to `wm.solve()`. If it returns `found: False`, your model sees nothing new to "
    "reach: try the actions in `untried_here` for real, they are the cheapest probes left.",
]

PLANNER_LINES = [
    "- TURN BUDGET: estimate the turns you have left as `last_action_result['time_remaining_seconds'] / 150`. "
    "More than 35 left: up to 3 probe turns per new level, then act. 15 to 35 left: at most one probe turn per "
    "level, otherwise execute your best plan in batches. Under 15: no probing and no rule editing -- only "
    "`wm.solve()` or the best sequence you already know. Under 5: one call that batches everything you still "
    "intend to do. A level finished with extra actions still scores; a perfect plan that is still being refined "
    "at the deadline scores nothing.",
]


def _py_str(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def _block(lines):
    return "".join("    " + _py_str(line + "\n") + "\n" for line in lines)


def _patch(text, edits, what):
    for anchor, repl in edits:
        n = text.count(anchor)
        if n != 1:
            raise SystemExit(f"{what}: anchor matched {n} times (need exactly 1): {anchor.strip()[:70]!r}")
        text = text.replace(anchor, repl)
    return text


def patch_wm_source(src):
    """round-2 WM_SOURCE -> with wm.explore. Idempotent."""
    if MARK_SRC in src:
        return src
    if "def health(G" not in src:
        raise SystemExit("WM_SOURCE has no wm.health -- apply wm_round2.py first")
    out = _patch(src, WM_EDITS, "WM_SOURCE")
    compile(out, "<wm>", "exec")
    return out


def patch_prompts(text, turn_planner=False):
    """prompts.py source -> explore bullet (and optionally the turn-budget bullet) before the
    `print(...)` bullet, i.e. after the round-1 and round-2 wm blocks. Idempotent per bullet."""
    lines = ([] if MARK_PROMPT in text else EXPLORE_LINES) + \
            (PLANNER_LINES if turn_planner and MARK_PLANNER not in text else [])
    if not lines:
        return text
    out = _patch(text, [(PROMPT_ANCHOR, _block(lines) + PROMPT_ANCHOR)], "prompts.py")
    compile(out, "prompts.py", "exec")
    return out


def apply(root, turn_planner=False, dry_run=False):
    agent = os.path.join(root, "inference", "agent")
    ws_path, pr_path = os.path.join(agent, "wm_source.py"), os.path.join(agent, "prompts.py")
    if not os.path.exists(ws_path):
        raise SystemExit("wm_source.py not found -- run apply_wm_patch.py and wm_round2.py first")
    ns = {}
    ws_text = open(ws_path).read()
    exec(compile(ws_text, ws_path, "exec"), ns)
    pr_text = open(pr_path).read()
    new_src, new_pr = patch_wm_source(ns["WM_SOURCE"]), patch_prompts(pr_text, turn_planner)
    if new_src == ns["WM_SOURCE"] and new_pr == pr_text:
        return "already patched"
    if dry_run:
        return "dry run ok: all anchors unique, patched sources compile"
    head = ws_text[:ws_text.index("WM_SOURCE = ")]
    open(ws_path, "w").write(head + "WM_SOURCE = " + repr(new_src) + "\n")
    open(pr_path, "w").write(new_pr)
    return "patched"


def selftest():
    import shutil
    import tempfile
    import types
    sys.path.insert(0, HERE)
    import apply_wm_patch
    import wm_ref
    import wm_round2 as r2
    root = os.path.join(os.path.dirname(os.path.dirname(HERE)), "bundle_fast", "src", "ARC3-Inference")
    with tempfile.TemporaryDirectory() as tmp:
        dst = os.path.join(tmp, "ARC3-Inference")
        os.makedirs(os.path.join(dst, "inference"))
        shutil.copytree(os.path.join(root, "inference", "agent"), os.path.join(dst, "inference", "agent"))
        assert apply_wm_patch.apply(dst) == "patched"
        try:
            apply(dst)
            raise SystemExit("round 3 applied without round 2")
        except SystemExit as e:
            assert "wm_round2" in str(e)
        assert r2.apply(dst) == "patched"
        assert apply(dst, dry_run=True).startswith("dry run ok")
        assert apply(dst) == "patched" and apply(dst) == "already patched"
        pr = open(os.path.join(dst, "inference", "agent", "prompts.py")).read()
        assert MARK_PROMPT in pr and MARK_PLANNER not in pr
        assert apply(dst, turn_planner=True) == "patched" and apply(dst, turn_planner=True) == "already patched"
        src3 = patch_wm_source(r2.patch_wm_source(wm_ref.WM_SOURCE))
        assert patch_wm_source(src3) == src3
        sys.path.insert(0, root)
        pkg = types.ModuleType("inference.agent")
        pkg.__path__ = [os.path.join(dst, "inference", "agent")]
        import inference  # noqa: F401
        sys.modules["inference.agent"] = pkg
        from inference.agent import prompts, python_tool_sandbox as sb, wm_source
        assert wm_source.WM_SOURCE == src3
        pa = prompts.PYTHON_ADDENDUM
        assert pa.index("NOTES vs SAVED CODE") < pa.index(MARK_PROMPT) < pa.index(MARK_PLANNER) < pa.index("- Use `print(...)`")
        assert pa.count(MARK_PROMPT) == 1 and pa.count(MARK_PLANNER) == 1

        rules = r2.MOVE_RULES[:r2.MOVE_RULES.index("def is_goal")]          # mechanics known, goal unknown
        distinct = "result = len({parse(wm.grid_of(t.after_frame)) for t in transitions} | {parse(wm.grid_of(current_frame))})"

        # 1. every explore call reaches a state the level has not been in, with the model's shortest path
        g = r2.MoveGame()
        call = r2._harness(sb, g)
        r = call("action(['RIGHT', 'UP', 'LEFT', 'DOWN'])\nresult = [wm.solve(), wm.health()['verdict']]", rules)
        assert "is_goal" in r[0]["error"] and r[1] == "use", r                # solve cannot run; the model is healthy
        seen = call(distinct, rules)
        steps_before = g.steps
        for i in range(6):
            r = call("result = wm.explore()", rules)
            assert r["stage"] == "explore" and r["found"] and r["execute"]["stopped"] == "plan_done", r
            now = call(distinct, rules)
            assert now == seen + 1 and r["seen_states"] == seen, (i, r, now, seen)   # exactly one new state per call
            assert g.steps - steps_before == len(r["plan"]) and "actions" not in r
            seen, steps_before = now, g.steps
        r = call("result = wm.explore(run=False)", rules)
        assert r["found"] and r["actions"] and "execute" not in r and g.steps == steps_before

        # 2. a model that predicts nothing new: no action spent, the untried real actions are listed
        g = r2.MoveGame()
        call = r2._harness(sb, g)
        r = call("action(['RIGHT', 'LEFT'])\nresult = wm.explore()",
                 rules[:rules.index("def step")] + "def step(s, a): return s\n")
        assert r["found"] is False and g.steps == 2 and r["untried_here"] == ["UP", "DOWN", "LEFT"], r

        # 3. a wrong model: explore turns the miss into a counterexample after one action
        g = r2.MoveGame()
        call = r2._harness(sb, g)
        wrong = r2.MOVE_RULES_WRONG[:r2.MOVE_RULES_WRONG.index("def is_goal")]
        r = call("result = wm.explore()", wrong)
        assert r["found"] and r["execute"]["stopped"] == "diverged" and r["execute"]["executed"] == 1 and g.steps == 1, r

        # 4. hidden state: health says drop -> explore refuses to act
        g = r2.MoveGame(sticky=3)
        call = r2._harness(sb, g)
        r = call("action(%r)\nresult = wm.explore()" % (["RIGHT", "LEFT"] * 3), rules)
        assert r["stage"] == "fallback" and g.steps == 6, r

        # 5. exploration finds the goal by itself: the level-up shows in the execute result
        class NearGoal(r2.MoveGame):                       # the avatar cannot stand on the target tile
            def reset(self):
                self.p, self.box, self.goal = (6, 1), (3, 5), (2, 5)

            def act(self, a):
                d = r2.DELTA[a["action"]]
                if (self.p[0] + d[0], self.p[1] + d[1]) == self.goal:
                    self.steps += 1
                    return False
                return super().act(a)
        g = NearGoal()
        call = r2._harness(sb, g)
        rules5 = rules.replace("    if n in WALL: return s\n", "    if n in WALL or n in targets: return s\n")
        assert rules5 != rules
        ups, calls5 = 0, 0
        for _ in range(150):
            r = call("result = wm.explore()", rules5)
            calls5 += 1
            assert r["stage"] == "explore", r
            if r.get("execute", {}).get("level_completed"):
                ups += 1
                break
        assert ups == 1 and g.level == 2, (g.level, r)
        found_after = g.steps

        # 6. click game: explore uses actions()
        g = r2.ClickGame()
        call = r2._harness(sb, g)
        crules = r2.CLICK_RULES[:r2.CLICK_RULES.index("def is_goal")]
        r = call("result = wm.explore()", crules)
        assert r["found"] and r["plan"][0].startswith("MOUSE(") and r["execute"]["executed"] == 1, r

    words = sum(len(l.split()) for l in EXPLORE_LINES + PLANNER_LINES)
    print("selftest ok (real sandbox subprocess, rounds 1+2+3 applied to a copy of bundle_fast):")
    print("  explore: 6 calls -> 6 new states, one per call, by the model's shortest path; nothing spent when the "
          "model predicts nothing new (untried actions listed); a wrong model costs 1 action; refuses on a drop verdict")
    print(f"  goal unknown: pure exploration completed the toy level after {calls5} calls / {found_after} actions; "
          "click games use actions()")
    print(f"  prompt: explore bullet always, turn-budget bullet only with --turn-planner ({words} words together)")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--print-prompt":
        print("\n\n".join(EXPLORE_LINES + (PLANNER_LINES if "--turn-planner" in sys.argv else [])))
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("root")
        ap.add_argument("--turn-planner", action="store_true")
        ap.add_argument("--dry-run", action="store_true")
        a = ap.parse_args()
        print(apply(a.root, a.turn_planner, a.dry_run))
