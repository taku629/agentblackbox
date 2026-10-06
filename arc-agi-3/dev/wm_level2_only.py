"""Lose branch of the wm arm: the world model is only used from level 2 on.

    python3 wm_level2_only.py <bundle>/src/ARC3-Inference [--dry-run] [--min-level 2]
    python3 wm_level2_only.py --selftest
    python3 wm_level2_only.py --print-prompt

Applies ON TOP of apply_wm_patch.py, wm_round2.py, wm_round3.py, wm_search.py and wm_modes.py --mode full
(turn_budget.py before or after, both orders work). Anchors must match exactly once; re-running is a no-op.

Why: when the wm arm loses to the notes arm, the cost is paid on level 1 -- the model spends its first
turns writing parse/step/is_goal for a game it has not understood yet. Later levels reuse the mechanics
and weigh more in the score (level index is the weight), so that is where a model pays off.

What changes
  wm.solve / explore / search / check / plan / health   on a level below --min-level return
        {'stage': 'level<k>', 'do': ...} and take no action; from --min-level on they behave as before.
  wm.changes / objects / cells / hud / report / budget   unchanged on every level (no model needed).
  prompts.py   one bullet telling the model not to write the three functions on level 1.
The level is read from current_frame.level at call time, so nothing is remembered between calls.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARK_SRC = "_WM_MIN_LEVEL"
MARK_PROMPT = "- WM FROM LEVEL"

BUILD_ANCHOR = '\n\ndef _wm_build():\n'
GATE_ANCHOR = '    if _WM_MODE == "toolbox":\n'
GATE = '''    def gated(f):
        def call(*a, **k):
            lvl = getattr(G().get("current_frame"), "level", None)
            if isinstance(lvl, int) and lvl < _WM_MIN_LEVEL:
                return {"stage": "level%d" % lvl,
                        "do": "world-model calls start at level %d: finish this level by acting directly "
                              "(wm.changes() and wm.budget() work now), then write parse/step/is_goal" % _WM_MIN_LEVEL}
            return f(*a, **k)
        return call
    for name in ("solve", "explore", "search", "check", "plan", "health"):
        if name in ns:
            ns[name] = gated(ns[name])
'''
PROMPT_ANCHOR = '    "- Use `print(...)` for compact summaries, or assign a final compact object to `result`.\\n"\n'


def prompt_lines(min_level):
    return [
        "- WM FROM LEVEL %d: on the first level%s do NOT write `parse` / `step` / `is_goal`. Learn the game by acting: "
        "send batches of actions and read `wm.changes()` after them; `wm.solve()` only answers stage 'level1' there. "
        "Once level %d starts, write the three functions from what you learned and call `out = wm.solve()` every turn."
        % (min_level, "" if min_level == 2 else "s", min_level),
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


def patch_wm_source(src, min_level=2):
    if MARK_SRC in src:
        return src
    if "def report(notes" not in src:
        raise SystemExit("WM_SOURCE has no wm.report -- apply wm_modes.py --mode full first")
    if '_WM_MODE = "toolbox"' in src:
        raise SystemExit("WM_SOURCE is in toolbox mode: there is no world model to delay")
    out = _patch(src, [(BUILD_ANCHOR, "\n_WM_MIN_LEVEL = %d" % int(min_level) + BUILD_ANCHOR),
                       (GATE_ANCHOR, GATE + GATE_ANCHOR)], "WM_SOURCE")
    compile(out, "<wm>", "exec")
    return out


def patch_prompts(text, min_level=2):
    if MARK_PROMPT in text:
        return text
    block = "".join("    " + _py_str(line + "\n") + "\n" for line in prompt_lines(min_level))
    out = _patch(text, [(PROMPT_ANCHOR, block + PROMPT_ANCHOR)], "prompts.py")
    compile(out, "prompts.py", "exec")
    return out


def apply(root, dry_run=False, min_level=2):
    agent = os.path.join(root, "inference", "agent")
    ws_path, pr_path = os.path.join(agent, "wm_source.py"), os.path.join(agent, "prompts.py")
    if not os.path.exists(ws_path):
        raise SystemExit("wm_source.py not found -- apply the wm stack first")
    ns = {}
    ws_text = open(ws_path).read()
    exec(compile(ws_text, ws_path, "exec"), ns)
    pr_text = open(pr_path).read()
    new_src, new_pr = patch_wm_source(ns["WM_SOURCE"], min_level), patch_prompts(pr_text, min_level)
    if new_src == ns["WM_SOURCE"] and new_pr == pr_text:
        return "already patched"
    if dry_run:
        return "dry run ok: all anchors unique, patched sources compile"
    head = ws_text[:ws_text.index("WM_SOURCE = ")]
    open(ws_path, "w").write(head + "WM_SOURCE = " + repr(new_src) + "\n")
    open(pr_path, "w").write(new_pr)
    return "patched"


def selftest():
    import tempfile
    sys.path.insert(0, HERE)
    import turn_budget
    import wm_modes
    import wm_round2 as r2
    import wm_search
    for order in ("budget_first", "budget_last"):
        steps = [lambda d: wm_modes.apply(d, "full")]
        mine = lambda d: [apply(d, dry_run=True), apply(d)]                                  # noqa: E731
        budget = lambda d: turn_budget.apply(d, game_seconds=1000.0)                         # noqa: E731
        steps += [budget, mine] if order == "budget_first" else [mine, budget]
        with tempfile.TemporaryDirectory() as tmp:
            sb, prompts, wm_source, dst = wm_search.patched_sandbox(tmp, extra=steps)
            assert apply(dst) == "already patched" and "_WM_MIN_LEVEL = 2" in wm_source.WM_SOURCE
            pa = prompts.PYTHON_ADDENDUM
            assert pa.count("WM FROM LEVEL 2") == 1 and pa.index("WM FROM LEVEL 2") < pa.index("- Use `print(...)`")
            g = r2.MoveGame()
            call = r2._harness(sb, g)
            # level 1: every model verb is held back and nothing is played, even with correct rules loaded
            r = call("result = [wm.solve(), wm.explore(), wm.search(), wm.check(), wm.plan(), wm.health()]", r2.MOVE_RULES)
            assert [x["stage"] for x in r] == ["level1"] * 6 and g.steps == 0 and "level 2" in r[0]["do"], r
            # helpers and the governor still work on level 1; report carries the notes
            r = call("action(['RIGHT'])\nresult = [wm.changes()['moved'][0]['to'], wm.budget()['phase'], "
                     "wm.report({'goal': 'x'}, wm.solve())]", r2.MOVE_RULES)
            assert r[0] == [24, 8] and r[1] == "probe" and r[2]["notes"] == {"goal": "x"} and r[2]["wm"]["stage"] == "level1", r
            # finish level 1 by direct play (the toy solution: push the box onto the target)
            import copy
            sim, path = copy.deepcopy(g), None
            frontier, seen = [(sim, [])], {(sim.p, sim.box)}
            while frontier and path is None:
                nxt = []
                for s, p in frontier:
                    for a in ("UP", "DOWN", "LEFT", "RIGHT"):
                        t = copy.deepcopy(s)
                        if t.act({"action": a}):
                            path = p + [a]
                            break
                        if (t.p, t.box) not in seen:
                            seen.add((t.p, t.box))
                            nxt.append((t, p + [a]))
                    if path:
                        break
                frontier = nxt
            r = call("action(%r)\nresult = current_frame.level" % path, r2.MOVE_RULES)
            assert r == 2 and g.level == 2
            # level 2: the same calls now plan and play
            before = g.steps
            r = call("out = wm.solve()\nresult = [out['stage'], out['execute'].get('level_completed'), wm.health()['verdict']]", r2.MOVE_RULES)
            assert r[0] == "execute" and r[1] is True and g.level == 3 and g.steps > before, r
    # refusals
    import wm_ref
    import wm_round3 as r3
    base = wm_search.patch_wm_source(r3.patch_wm_source(r2.patch_wm_source(wm_ref.WM_SOURCE)))
    for bad, msg in ((base, "wm_modes.py"), (wm_modes.patch_wm_source(base, "toolbox"), "toolbox mode")):
        try:
            patch_wm_source(bad)
            raise AssertionError("accepted " + msg)
        except SystemExit as e:
            assert msg in str(e), str(e)
    full = wm_modes.patch_wm_source(base, "full")
    assert "_WM_MIN_LEVEL = 3" in patch_wm_source(full, 3) and patch_wm_source(patch_wm_source(full)) == patch_wm_source(full)
    print("selftest ok (real sandbox subprocess, full wm stack + governor in both apply orders): on level 1 solve / "
          "explore / search / check / plan / health return stage 'level1' and play nothing while changes / budget / "
          "report work; after level 1 is finished by direct play the same wm.solve() plans and completes level 2; "
          "prompt: %d words" % len(" ".join(prompt_lines(2)).split()))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--print-prompt":
        print("\n\n".join(prompt_lines(2)))
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("root")
        ap.add_argument("--dry-run", action="store_true")
        ap.add_argument("--min-level", type=int, default=2)
        a = ap.parse_args()
        print(apply(a.root, a.dry_run, a.min_level))
