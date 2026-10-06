"""Win branch of the wm arm: the ~1,100-word world-model instructions compressed to one ~110-word bullet.

    python3 wm_short_prompt.py <bundle>/src/ARC3-Inference [--persist-call 'persist(code)'] [--dry-run]
    python3 wm_short_prompt.py --selftest
    python3 wm_short_prompt.py --print-prompt

Applies ON TOP of the full stack: apply_wm_patch.py, wm_round2.py, wm_round3.py, wm_search.py,
wm_modes.py --mode full and, if used, turn_budget.py. Only prompts.py changes; wm_source.py is untouched,
so behaviour of every wm call is identical and this is a pure A/B on prompt length.

It removes every world-model bullet the earlier patches added (toolkit, the three functions, saving,
check/plan, typical game, state templates, start crude, MOVE/CLICK games, canonical states, animated
actions, is_goal, when to stop, notes vs code, goal unknown, no plan found, turn budget, one result,
budget line) and inserts one bullet that keeps the operating loop:
  write three functions -> persist -> out = wm.solve() -> result = wm.report(notes, out) -> react to out['stage'].
What is dropped is the teaching material (templates and worked examples). If the short arm scores lower,
the templates were doing work; if it scores the same, every game gets ~1,400 prompt tokens back per turn.
A bundle patched by wm_level2_only.py keeps its 'WM FROM LEVEL' bullet.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import wm_modes  # noqa: E402

MARK = "- WORLD MODEL (`wm`"
PROMPT_ANCHOR = '    "- Use `print(...)` for compact summaries, or assign a final compact object to `result`.\\n"\n'
STRIP = [b for b in wm_modes.WM_BULLETS if b not in ("- TOOLBOX", "- Reusable code:")] + ["- BUDGET LINE"]


def short_lines(persist_call="persist(code)"):
    return [
        "- WORLD MODEL (`wm`, preloaded): write `parse(grid)` -> small hashable state (sorted tuples, without "
        "`wm.hud()` cells), `step(state, action)` -> next state, `is_goal(state)`; save them once with "
        f"`{persist_call}`. Every turn: `out = wm.solve()` -- it replays your history through the model, plans, and "
        "plays until a prediction fails -- then `result = wm.report(notes, out)`. React to `out['stage']`: `check` "
        "-> fix the rule shown in `mismatch`; `plan` -> the goal is unreachable: fix `is_goal`, or define "
        "`progress(state)` (higher = closer) and call again; `fallback` -> stop modelling, play directly. Goal "
        "unknown: `wm.explore()`. `wm.changes()` tells what the last action did. If a result has a `budget` entry, "
        "do what it says first.",
    ]


def _py_str(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def strip_bullets(text):
    out, dropped = [], []
    for line in text.split("\n"):
        s = line.strip()
        hit = next((b for b in STRIP if s.startswith('"') and s[1:].startswith(b)), None)
        if hit:
            dropped.append(hit)
            continue
        out.append(line)
    return "\n".join(out), dropped


def patch_prompts(text, persist_call="persist(code)"):
    if MARK in text:
        return text
    if "- TOOLBOX" in text:
        raise SystemExit("prompts.py is in toolbox mode: there are no world-model instructions to shorten")
    if "- ONE RESULT" not in text:
        raise SystemExit("prompts.py has no ONE RESULT bullet -- apply wm_modes.py --mode full first")
    stripped, dropped = strip_bullets(text)
    if len(dropped) < 15:
        raise SystemExit(f"only {len(dropped)} world-model bullets found (expected 18+): prompts.py is not the full stack")
    n = stripped.count(PROMPT_ANCHOR)
    if n != 1:
        raise SystemExit(f"prompts.py: anchor matched {n} times (need exactly 1)")
    block = "".join("    " + _py_str(line + "\n") + "\n" for line in short_lines(persist_call))
    out = stripped.replace(PROMPT_ANCHOR, block + PROMPT_ANCHOR)
    compile(out, "prompts.py", "exec")
    return out


def apply(root, persist_call="persist(code)", dry_run=False):
    pr_path = os.path.join(root, "inference", "agent", "prompts.py")
    text = open(pr_path).read()
    new = patch_prompts(text, persist_call)
    if new == text:
        return "already patched"
    if dry_run:
        return "dry run ok: %d bullets would be replaced by one" % len(strip_bullets(text)[1])
    open(pr_path, "w").write(new)
    return "patched"


def selftest():
    import tempfile
    import turn_budget
    import wm_level2_only
    import wm_round2 as r2
    import wm_search
    words = {}
    for variant in ("full", "full+budget", "full+budget+level2"):
        steps = [lambda d: wm_modes.apply(d, "full", "persist(src)")]
        if "budget" in variant:
            steps.append(lambda d: turn_budget.apply(d, game_seconds=1000.0))
        if "level2" in variant:
            steps.append(lambda d: wm_level2_only.apply(d))
        grab = {}
        steps.append(lambda d: grab.update(before=open(os.path.join(d, "inference", "agent", "prompts.py")).read(),
                                           ws=open(os.path.join(d, "inference", "agent", "wm_source.py")).read()))
        steps.append(lambda d: [apply(d, "persist(src)", dry_run=True), apply(d, "persist(src)")])
        with tempfile.TemporaryDirectory() as tmp:
            sb, prompts, wm_source, dst = wm_search.patched_sandbox(tmp, extra=steps)
            assert apply(dst, "persist(src)") == "already patched"
            assert open(os.path.join(dst, "inference", "agent", "wm_source.py")).read() == grab["ws"]     # prompt-only change
            pa = prompts.PYTHON_ADDENDUM
            ns = {}
            exec(compile(grab["before"], "before", "exec"), ns)
            long_pa = ns["PYTHON_ADDENDUM"]
            assert pa.count("WORLD MODEL (`wm`") == 1 and "persist(src)" in pa
            gone = ["WM STATE TEMPLATES", "MOVE games", "CLICK games", "WHEN TO STOP MODELLING", "NOTES vs SAVED CODE",
                    "GOAL UNKNOWN", "NO PLAN FOUND", "TURN BUDGET", "ONE RESULT", "BUDGET LINE", "WORLD MODEL TOOLKIT"]
            assert not [w for w in gone if w in pa], [w for w in gone if w in pa]
            assert ("WM FROM LEVEL 2" in pa) == ("level2" in variant)
            # everything that is not a world-model bullet is byte-identical
            keep = lambda t: [ln for ln in t.split("\n") if "WORLD MODEL (`wm`" not in ln]                 # noqa: E731
            assert keep(pa) == keep(strip_bullets_text(long_pa)), "non-wm prompt text changed"
            for verb in ("wm.solve()", "wm.report(notes, out)", "wm.explore()", "wm.changes()", "progress(state)", "is_goal",
                         "mismatch", "fallback", "budget"):
                assert verb in pa, verb
            words[variant] = (len(long_pa.split()) - len(pa.split()), len(" ".join(short_lines()).split()))
            if variant == "full+budget":                                # the loop the bullet describes works as written
                g = r2.MoveGame()
                call = r2._harness(sb, g)
                r = call("notes = {'goal': 'box on target'}\nout = wm.solve()\nresult = wm.report(notes, out)", r2.MOVE_RULES)
                assert r["wm"]["stage"] == "execute" and r["wm"]["level_completed"] is True and r["notes"]["goal"], r
    for bad, msg in (("PYTHON_ADDENDUM = (\n" + PROMPT_ANCHOR + ")\n", "ONE RESULT"),
                     ('PYTHON_ADDENDUM = (\n    "- TOOLBOX x\\n"\n' + PROMPT_ANCHOR + ")\n", "toolbox mode")):
        try:
            patch_prompts(bad)
            raise AssertionError("accepted " + msg)
        except SystemExit as e:
            assert msg in str(e), str(e)
    saved, short = words["full+budget"]
    print("selftest ok (rounds 1-3 + search + modes full [+ governor] [+ level2] on a copy of bundle_fast): one "
          f"{short}-word bullet replaces the world-model text ({saved} words shorter with the governor line), all other "
          "prompt text and wm_source.py are byte-identical, the level-2 bullet survives, and the described loop "
          "(solve -> report) completes the toy level in the real sandbox")


def strip_bullets_text(addendum):
    """The rendered addendum with every stripped bullet's paragraph removed (for the selftest comparison)."""
    return "\n".join(ln for ln in addendum.split("\n") if not any(ln.startswith(b) for b in STRIP))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--print-prompt":
        print("\n\n".join(short_lines()))
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("root")
        ap.add_argument("--persist-call", default="persist(code)")
        ap.add_argument("--dry-run", action="store_true")
        a = ap.parse_args()
        print(apply(a.root, a.persist_call, a.dry_run))
