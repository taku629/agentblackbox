"""World-model verbs gated to level 2+: level 1 is always played by hand.

    python3 wm_level2_only.py <bundle>/src/ARC3-Inference [--dry-run]
    python3 wm_level2_only.py --selftest
    python3 wm_level2_only.py --print-prompt

Applies ON TOP of apply_wm_patch.py, wm_round2.py, wm_round3.py, wm_search.py and
wm_modes.py --mode full; turn_budget.py may come before or after it (both orders work --
the gate is the outermost wrapper either way). Anchors must match exactly once or nothing
is written; re-running is a no-op.

Why (the <=5.7 branch of medal_closeout.md 2.2): the wm arm lost clearly, but ripping the
whole toolkit out (toolbox) throws away what DID work -- the losing move was paying level-1
turns to write parse/step/is_goal on almost no evidence. This build keeps the helpers
(changes/objects/cells/hud/clean/budget/report) on every level and refuses only the acting
verbs on level 1:
    wm.solve / wm.explore / wm.search / wm.check / wm.plan / wm.health / wm.execute
        -> {"stage": "level1", "do": "play this level by hand: ..."}
The model probes level 1 directly (each untried action once, ONE call), reads wm.changes()
and drafts the rule functions in its notes; from level 2 the armed verbs work as usual,
now with a full level-1 history to check the model against.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARK_SRC = '"stage": "level1"'
MARK_PROMPT = "- LEVEL 1 IS PLAYED BY HAND"

GATED = ("check", "plan", "execute", "solve", "health", "explore", "search")

GATE = '''
    LEVEL1 = {"stage": "level1",
              "do": "the world model is armed from level 2: play this level by hand -- try each untried action "
                    "once (ONE call), read wm.changes() after it, and draft parse/step/is_goal in your notes"}
    def _lv(f):
        def call(*a, **k):
            lvl = getattr(G().get("current_frame"), "level", 1) or 1
            if lvl < 2:
                return dict(LEVEL1)
            return f(*a, **k)
        return call
    for name in ("check", "plan", "execute", "solve", "health", "explore", "search"):
        if name in ns:
            ns[name] = _lv(ns[name])
'''

WM_EDITS = [
    ('    return type("wm", (), ns)\n', GATE + '    return type("wm", (), ns)\n'),
]

PROMPT_ANCHOR = '    "- Use `print(...)` for compact summaries, or assign a final compact object to `result`.\\n"\n'

PROMPT_LINES = [
    "- LEVEL 1 IS PLAYED BY HAND: `wm.solve`/`wm.explore`/`wm.search`/`wm.check`/`wm.plan`/`wm.health`/`wm.execute` "
    "answer `stage 'level1'` and do nothing until the first level is cleared. Probe each untried action once (ONE "
    "call), read `wm.changes()`, draft parse/step/is_goal in your notes -- the verbs arm themselves from level 2 on.",
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
    """modes-level WM_SOURCE -> with the level-2 gate on the acting verbs. Idempotent."""
    if MARK_SRC in src:
        return src
    if '_WM_MODE = "toolbox"' in src:
        raise SystemExit("bundle is in toolbox mode -- wm_level2_only is for the --mode full arm")
    if "def report(notes" not in src:
        raise SystemExit("WM_SOURCE has no wm.report -- apply wm_modes.py --mode full first")
    out = _patch(src, WM_EDITS, "WM_SOURCE")
    compile(out, "<wm>", "exec")
    return out


def patch_prompts(text):
    if MARK_PROMPT in text:
        return text
    if "- TOOLBOX" in text:
        raise SystemExit("prompts.py is in toolbox mode -- wm_level2_only is for the --mode full arm")
    block = "".join("    " + _py_str(line + "\n") + "\n" for line in PROMPT_LINES)
    out = _patch(text, [(PROMPT_ANCHOR, block + PROMPT_ANCHOR)], "prompts.py")
    compile(out, "prompts.py", "exec")
    return out


def apply(root, dry_run=False):
    agent = os.path.join(root, "inference", "agent")
    ws_path, pr_path = os.path.join(agent, "wm_source.py"), os.path.join(agent, "prompts.py")
    if not os.path.exists(ws_path):
        raise SystemExit("wm_source.py not found -- apply the wm stack and wm_modes.py first")
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


def selftest():
    import tempfile
    sys.path.insert(0, HERE)
    import wm_modes
    import wm_round2 as r2
    import wm_search
    import turn_budget

    LEVEL1_SOLVE = ["UP", "UP", "RIGHT", "RIGHT", "RIGHT", "DOWN", "RIGHT", "UP", "UP"]
    for order in ("budget_first", "gate_first"):
        with tempfile.TemporaryDirectory() as tmp:
            extra = ([lambda d: turn_budget.apply(d), lambda d: apply(d)] if order == "budget_first"
                     else [lambda d: apply(d), lambda d: turn_budget.apply(d)])
            sb, prompts, wm_source, dst = wm_search.patched_sandbox(
                tmp, extra=[lambda d: wm_modes.apply(d, "full")] + extra)
            assert apply(dst) == "already patched"
            pa = prompts.PYTHON_ADDENDUM
            assert pa.count("LEVEL 1 IS PLAYED BY HAND") == 1
            assert pa.index("LEVEL 1 IS PLAYED BY HAND") < pa.index("- Use `print(...)`")

            # 1. level 1: every acting verb is off, no action is burned
            g = r2.MoveGame()
            call = r2._harness(sb, g)
            code = ("result = [wm.solve(), wm.explore(), wm.search(), wm.check(), wm.plan(), "
                    "wm.health(), wm.execute(['RIGHT'])]")
            r = call(code, r2.MOVE_RULES)
            assert [x["stage"] for x in r] == ["level1"] * 7 and g.steps == 0, r
            assert all("play this level by hand" in x["do"] for x in r)

            # 2. level 1: helpers and the governor still work
            r = call("result = [wm.budget(), wm.report({'goal': '?'}), wm.changes()]", "")
            assert r[0]["phase"] == "probe" and r[0]["untried"], r[0]
            assert "budget" in r[1] or r[1]["wm"] == {}
            assert "error" in r[2]

            # 3. level 1 by hand -> the same wm.solve() runs on level 2
            r = call("action(%r)\nresult = [wm.solve(), wm.check()]" % LEVEL1_SOLVE, r2.MOVE_RULES)
            assert g.level == 3, (g.level, r)                    # solve cleared level 2 too
            assert r[0]["stage"] == "execute" and r[0]["execute"]["level_completed"], r[0]
            assert r[1].get("stage") != "level1" and "acc" in r[1]           # check ran for real

            # 4. refusing on a toolbox-mode build
            with tempfile.TemporaryDirectory() as tmp2:
                sb2, prompts2, ws2, dst2 = wm_search.patched_sandbox(
                    tmp2, extra=[lambda d: wm_modes.apply(d, "toolbox")])
                try:
                    apply(dst2)
                    raise SystemExit("level2 gate applied on a toolbox bundle")
                except SystemExit as e:
                    assert "toolbox" in str(e)

            if order == "budget_first":
                words = sum(len(x.split()) for x in PROMPT_LINES)

    print("selftest ok (real sandbox subprocess, search + modes full + gate on bundle_fast):")
    print("  level 1: solve/explore/search/check/plan/health/execute all answer stage 'level1' and burn "
          "nothing; changes/budget/report still work")
    print("  cleared level 1 by hand -> the same wm.solve() finished level 2; gate+turn_budget apply in "
          f"either order; prompt: {words} words")


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
