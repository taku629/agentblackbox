"""Compress the world-model bullets in prompts.py into one short bullet (the wm-won branch).

    python3 wm_short_prompt.py <bundle>/src/ARC3-Inference [--dry-run]
    python3 wm_short_prompt.py --selftest
    python3 wm_short_prompt.py --print-prompt

The >=8.1 branch of medal_closeout.md 2.2: the wm arm won the measurement, so the kernel
keeps it -- but the wall of wm bullets (20 lines, ~1.1k words together with the budget
governor) is the one thing worth trimming to give the model more of its context budget for
actual reasoning. The toolkit itself (wm_source.py) is untouched; this only rewrites the
prompt: every world-model bullet written by rounds 1-3 / wm_search / wm_modes / turn_budget
is replaced by a single 109-word bullet that still mentions persist/check/search/changes/
budget/report. Nothing else in prompts.py changes.

Apply AFTER wm_modes.py --mode full and turn_budget.py, in either order. Re-running is a
no-op; refuses a bundle whose prompts do not look like the full-mode arm.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARK_PROMPT = "- WORLD MODEL (`wm`, preloaded)"

# the governor line from turn_budget.py is dropped too (its advice is folded into the bullet)
EXTRA_BULLETS = ["- BUDGET LINE"]

PROMPT_ANCHOR = '    "- Use `print(...)` for compact summaries, or assign a final compact object to `result`.\\n"\n'

SHORT_LINES = [
    "- WORLD MODEL (`wm`, preloaded): write `parse(grid)` -> a small hashable state (HUD cells out), "
    "`step(state, action)` and `is_goal(state)`; click games also `actions(state)`, deep goals "
    "`progress(state)` for `wm.search()`. Save them with `persist(code)` -- they usually carry over between "
    "levels. Each turn: `out = wm.solve()`, which checks the rules against recorded transitions, plans, and "
    "executes, stopping at the first miss (`pred_vs_actual` shows the step). Fix `step` when `wm.check()` "
    "shows a mismatch -- never plan on a model that fails its own history -- then `wm.solve()` again. "
    "`wm.changes()` `wm.objects()` `wm.hud()` summarise boards. A `budget` entry in a result means do it that "
    "turn. End every call with `result = wm.report(notes, out)`.",
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


def _strip_wm_lines(text):
    """Drop every wm bullet incl. the budget governor (wm_modes' list misses '- BUDGET LINE')."""
    import wm_modes
    keys = wm_modes.WM_BULLETS + EXTRA_BULLETS + [MARK_PROMPT]
    out, dropped = [], 0
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith('"') and any(s[1:].startswith(b) for b in keys):
            dropped += 1
            continue
        out.append(line)
    return "\n".join(out), dropped


def patch_prompts(text):
    """full-mode prompts.py -> same file with every wm bullet replaced by the one SHORT_LINES bullet."""
    if MARK_PROMPT in text:
        return text
    sys.path.insert(0, HERE)
    if "- TOOLBOX" in text:
        raise SystemExit("prompts.py is in toolbox mode -- wm_short_prompt is for the --mode full arm")
    text, dropped = _strip_wm_lines(text)
    if dropped < 8:
        raise SystemExit(f"only {dropped} wm bullets found -- this does not look like the full-mode arm "
                         "(apply wm_modes.py --mode full first)")
    block = "".join("    " + _py_str(line + "\n") + "\n" for line in SHORT_LINES)
    out = _patch(text, [(PROMPT_ANCHOR, block + PROMPT_ANCHOR)], "prompts.py")
    compile(out, "prompts.py", "exec")
    return out


def apply(root, dry_run=False):
    pr_path = os.path.join(root, "inference", "agent", "prompts.py")
    if not os.path.exists(pr_path):
        raise SystemExit("prompts.py not found -- apply the wm stack first")
    text = open(pr_path).read()
    new = patch_prompts(text)
    if new == text:
        return "already patched"
    if dry_run:
        return "dry run ok: anchors unique, patched prompts.py compiles"
    open(pr_path, "w").write(new)
    return "patched"


def selftest():
    import tempfile
    sys.path.insert(0, HERE)
    import wm_modes
    import wm_round2 as r2
    import wm_search
    import turn_budget

    # canonical order: modes --mode full, then turn_budget, then this patch
    with tempfile.TemporaryDirectory() as tmp:
        sb, prompts, wm_source, dst = wm_search.patched_sandbox(
            tmp, extra=[lambda d: wm_modes.apply(d, "full"), lambda d: turn_budget.apply(d), lambda d: apply(d)])
        assert apply(dst) == "already patched"
        pa = prompts.PYTHON_ADDENDUM
        words = len(pa.split())
        assert pa.count("WORLD MODEL (`wm`, preloaded)") == 1
        assert pa.index("WORLD MODEL (`wm`, preloaded)") < pa.index("- Use `print(...)`")
        assert "BUDGET LINE" not in pa and "ONE RESULT" not in pa and "TOOLKIT" not in pa
        # a twin bundle WITHOUT this patch: same everything -> the word delta, and the check that
        # no non-wm line moved (strip every wm bullet from both -> byte-identical text)
        with tempfile.TemporaryDirectory() as tmp2:
            sb2, prompts2, _, dst2 = wm_search.patched_sandbox(
                tmp2, extra=[lambda d: wm_modes.apply(d, "full"), lambda d: turn_budget.apply(d)])
            base_pa = len(prompts2.PYTHON_ADDENDUM.split())
            own = open(os.path.join(dst, "inference", "agent", "prompts.py")).read()
            ref = open(os.path.join(dst2, "inference", "agent", "prompts.py")).read()
            assert _strip_wm_lines(own)[0] == _strip_wm_lines(ref)[0]
        assert base_pa - words >= 800, (base_pa, words)          # ~-1,043 words by design

        # the toy level still completes: every verb the bullet mentions exists and works
        g = r2.MoveGame()
        call = r2._harness(sb, g)
        r = call("result = wm.solve()", r2.MOVE_RULES)
        assert r["stage"] == "execute" and r["execute"]["level_completed"], r
        assert g.level == 2

    w = len(SHORT_LINES[0].split())
    assert w == 109, w
    print("selftest ok (real sandbox subprocess, search + modes full + turn_budget + short prompt on bundle_fast):")
    print(f"  {base_pa - words} prompt words removed; one 109-word bullet; toy level still solves")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--print-prompt":
        print("\n\n".join(SHORT_LINES))
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("root")
        ap.add_argument("--dry-run", action="store_true")
        a = ap.parse_args()
        print(apply(a.root, a.dry_run))
