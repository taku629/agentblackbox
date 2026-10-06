"""Install the `wm` toolkit into a bundle's ARC3-Inference tree (anchor-asserted
source edits, same style as the in-notebook patcher).

    python3 apply_wm_patch.py <bundle>/src/ARC3-Inference [--persist-call 'persist(code)'] [--dry-run]

Touches three files under inference/agent/:
  wm_source.py            NEW   WM_SOURCE (copied from wm_ref.py -- single source of truth)
  python_tool_sandbox.py  +4 edits: import, send `preload`, expose _WM_GLOBALS, exec preload
  prompts.py              +1 edit:  WORLD MODEL TOOLKIT bullets in PYTHON_ADDENDUM

Every anchor must match exactly once or nothing is written. Re-running on an
already patched tree is a no-op. The preload runs before the user's code and
before/independent of persisted fragments (wm looks rule functions up at call
time), inside a try/except so a broken preload can never fail a tool call.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARK = "_WM_GLOBALS"

SANDBOX_EDITS = [
    # (anchor, replacement) -- host side
    ("from inference.utils.grid_utils import ARC_COLOR_CHARS\n",
     "from inference.utils.grid_utils import ARC_COLOR_CHARS\n"
     "from inference.agent.wm_source import WM_SOURCE\n"),
    ('                "code": code,\n',
     '                "code": code,\n'
     '                "preload": WM_SOURCE,\n'),
    # sandbox side (inside _SANDBOX_BOOTSTRAP)
    ('        runtime_globals["action"] = action\n',
     '        runtime_globals["action"] = action\n'
     '        runtime_globals["_WM_GLOBALS"] = runtime_globals\n'),
    ('        _refresh_state(initial.get("state") or {})\n',
     '        _refresh_state(initial.get("state") or {})\n'
     '        try:\n'
     '            exec(compile(str(initial.get("preload", "")), "<wm>", "exec"), runtime_globals, runtime_globals)\n'
     '        except Exception:\n'
     '            pass\n'),
]

PROMPT_ANCHOR = '    "- Use `print(...)` for compact summaries, or assign a final compact object to `result`.\\n"\n'


def prompt_block(persist_call):
    lines = [
        "- WORLD MODEL TOOLKIT: `wm` is preloaded in every `python` call. You write the game's rules ONCE as three small "
        "functions and save them; after that you never re-derive mechanics or hand-write a search again.",
        "- The three functions: `parse(grid)` turns `current_frame.grid` into a small HASHABLE state (tuples / frozensets "
        "of object positions -- leave the HUD/timer out); `step(state, action)` returns the state you predict after "
        "`action`, where action is a name like 'LEFT' or a click `('MOUSE', row, col)`; `is_goal(state)` is True when "
        "the level is solved. For click games also define `actions(state)` returning the click targets worth trying.",
        f"- Save them with `{persist_call}`: saved code is re-run at the start of every later call, so the functions are "
        "simply there next turn. Save a corrected version whenever you learn a rule; do not paste the rules again otherwise.",
        "- `wm.check()` replays every recorded transition through your `step` and returns accuracy per action plus the "
        "first mismatches as `pred_vs_actual`. A mismatch is a rule you have not learned yet: fix `step` and re-check. "
        "Do not plan with a model that fails its own history.",
        "- `wm.plan()` searches your model for the shortest action list that reaches `is_goal` (pass "
        "`h=lambda s: ...` for a distance heuristic on big state spaces). It costs zero game actions. "
        "`wm.execute(p['actions'])` plays the plan and stops at the first step whose real result differs from the "
        "prediction, so a wrong rule costs one action, not a batch. `wm.solve()` runs check, plan and execute in one call.",
        "- Typical game: 1-3 short probe turns to see what each action does -> write and save parse/step/is_goal -> "
        "`result = wm.solve()` each turn. If it reports `diverged` or low accuracy, read `pred_vs_actual`, fix that one "
        "rule, save, solve again. On a new level run `wm.solve()` first: rules usually carry over.",
    ]
    return "".join("    " + _py_str(line + "\n") + "\n" for line in lines)


def _py_str(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def patch_text(text, edits, what):
    for anchor, repl in edits:
        n = text.count(anchor)
        if n != 1:
            raise SystemExit(f"{what}: anchor matched {n} times (need exactly 1): {anchor.strip()[:70]!r}")
        text = text.replace(anchor, repl)
    return text


def apply(root, persist_call="persist(code)", dry_run=False):
    agent = os.path.join(root, "inference", "agent")
    sb_path, pr_path = os.path.join(agent, "python_tool_sandbox.py"), os.path.join(agent, "prompts.py")
    sb, pr = open(sb_path).read(), open(pr_path).read()
    if MARK in sb:
        return "already patched"
    sys.path.insert(0, HERE)
    from wm_ref import WM_SOURCE
    new_sb = patch_text(sb, SANDBOX_EDITS, "python_tool_sandbox.py")
    new_pr = patch_text(pr, [(PROMPT_ANCHOR, prompt_block(persist_call) + PROMPT_ANCHOR)], "prompts.py")
    compile(new_sb, sb_path, "exec")
    compile(new_pr, pr_path, "exec")
    compile(WM_SOURCE, "<wm>", "exec")
    if dry_run:
        return "dry run ok: all anchors unique, patched sources compile"
    with open(os.path.join(agent, "wm_source.py"), "w") as f:
        f.write('"""wm toolkit preloaded into every python tool call (generated by apply_wm_patch.py)."""\n\n'
                "WM_SOURCE = " + repr(WM_SOURCE) + "\n")
    open(sb_path, "w").write(new_sb)
    open(pr_path, "w").write(new_pr)
    return "patched"


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--persist-call", default="persist(code)")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    print(apply(a.root, a.persist_call, a.dry_run))
