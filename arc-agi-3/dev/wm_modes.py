"""Two ready-made follow-ups for the wm arm result: notes+wm integration, and the toolbox fallback.

    python3 wm_modes.py <bundle>/src/ARC3-Inference --mode full    [--persist-call 'persist(code)'] [--dry-run]
    python3 wm_modes.py <bundle>/src/ARC3-Inference --mode toolbox [--persist-call 'persist(code)'] [--dry-run]
    python3 wm_modes.py --selftest
    python3 wm_modes.py --print-prompt full|toolbox

Applies ON TOP of apply_wm_patch.py, wm_round2.py, wm_round3.py and wm_search.py. Anchors must match
exactly once or nothing is written. Running --mode again switches the mode in place.

Both modes add to WM_SOURCE
  wm.changes(i=-1)   what transition i did, HUD excluded: cells changed, bounding box, colour
                     transitions, objects that moved (from -> to), appeared, disappeared. This is
                     the diff/segmentation summary a model otherwise rewrites every turn.
  wm.report(notes, out)  -> {'notes': notes, 'wm': compact outcome}: ONE value for `result`, so the
                     notes echo and the wm outcome no longer compete for it.
  health():          `drop` also when a level has 40+ actions and recent accuracy is still < 0.9
                     (the model is stuck in the 'fix' zone; before, only < 0.6 dropped).

--mode full     (wm arm won or is promising) the round-1 advice `result = wm.solve()` is replaced by
                `out = wm.solve()` + `result = wm.report(notes, out)`; one bullet explains the split.
--mode toolbox  (adoption was low / the arm lost) wm.solve / wm.explore / wm.search return
                stage 'toolbox' and never act; ALL world-model bullets are removed from prompts.py and
                replaced by two short ones (helpers + what persist is for). parse/step are not asked for.
                The helpers stay host-preloaded, so the mode costs no model tokens beyond those bullets.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARK_SRC = "def report(notes"
MODE_LINE = '_WM_MODE = "%s"\n'

HELPERS = r'''
    def changes(G, i=-1):
        """What transition i did to the board, ignoring HUD cells."""
        trs = G.get("transitions", [])
        if not trs:
            return {"error": "no action taken yet"}
        try:
            t = trs[i]
        except Exception:
            return {"error": "no transition %r (have %d)" % (i, len(trs))}
        hudc, diffs = scan(G)
        d = diffs[i]
        if t.before_frame is None or d is None:
            return {"action": show(t.action), "changed": None}
        a, b = grid_of(t.before_frame), grid_of(t.after_frame)
        real = [p for p in d if p not in hudc]
        out = {"action": show(t.action), "changed": len(real), "hud_only": bool(d) and not real}
        if t.after_frame.level != t.before_frame.level:
            out["level"] = [t.before_frame.level, t.after_frame.level]
        if not real:
            return out
        ys, xs = [p[0] for p in real], [p[1] for p in real]
        out["bbox"] = [min(ys), min(xs), max(ys), max(xs)]
        cnt = {}
        for y, x in real:
            k = "%s>%s" % (a[y][x], b[y][x])
            cnt[k] = cnt.get(k, 0) + 1
        out["colors"] = dict(sorted(cnt.items(), key=lambda kv: -kv[1])[:5])
        oa, ob = objects(G, a), objects(G, b)
        sig = lambda o: (o[0], o[3], o[4], o[5])
        ca, cb = {}, {}
        for o in oa:
            ca.setdefault(sig(o), []).append(o)
        for o in ob:
            cb.setdefault(sig(o), []).append(o)
        moved = []
        for k in ca:
            if len(ca[k]) == 1 and len(cb.get(k, [])) == 1 and ca[k][0] != cb[k][0]:
                moved.append({"color": k[0], "size": [k[1], k[2]], "from": [ca[k][0][1], ca[k][0][2]],
                              "to": [cb[k][0][1], cb[k][0][2]]})
        if moved:
            out["moved"] = moved[:6]
        gone = sum(len(v) for k, v in ca.items() if k not in cb)
        new = sum(len(v) for k, v in cb.items() if k not in ca)
        if gone or new:
            out["shapes"] = {"disappeared": gone, "appeared": new}
        return out

    def report(notes, out=None):
        """One value for `result`: your notes plus a compact wm outcome."""
        keep = ("stage", "error", "check_acc", "acc", "n", "plan_len", "found", "improves", "why", "do", "reaches_goal")
        w = {}
        if isinstance(out, dict):
            w = {k: out[k] for k in keep if k in out}
            e = out.get("execute")
            if isinstance(e, dict):
                w["executed"], w["stopped"] = e.get("executed"), e.get("stopped")
                for k in ("level_completed", "pred_vs_actual", "at"):
                    if e.get(k):
                        w[k] = e[k]
            c = out.get("check")
            if isinstance(c, dict):
                w["acc"] = c.get("acc")
                if c.get("mismatch"):
                    w["mismatch"] = c["mismatch"][:1]
            hv = out.get("health")
            if isinstance(hv, dict):
                w["verdict"], w["why"] = hv.get("verdict"), hv.get("why")
        return {"notes": notes, "wm": w}

'''

STUCK_OLD = '        elif blind >= max(2, n // 4):\n'
STUCK_NEW = ('        elif n >= 2 * drop_after and acc < 0.9:\n'
             '            out["verdict"] = "drop"\n'
             '            out["why"] = "still below 0.9 accuracy after %d actions on this level" % n\n'
             + STUCK_OLD)

GATE_OLD = '    return type("wm", (), ns)\n'
GATE_NEW = ('    if _WM_MODE == "toolbox":\n'
            '        for name in ("solve", "explore", "search"):\n'
            '            if name in ns:\n'
            '                ns[name] = lambda *a, **k: {"stage": "toolbox", "do": "world-model calls are off in this build; "\n'
            '                                           "use wm.changes() / wm.objects() and act directly"}\n'
            + GATE_OLD)

WM_EDITS = [
    ('def _wm_build():\n', MODE_LINE % "full" + '\n\ndef _wm_build():\n'),
    ('    NEED = ("define parse(grid), step(state, action) and is_goal(state), then persist them; "\n',
     HELPERS + '    NEED = ("define parse(grid), step(state, action) and is_goal(state), then persist them; "\n'),
    (STUCK_OLD, STUCK_NEW),
    ('            "health": health, "hud": hud, "clean": clean, "objects": objects, "explore": explore, "search": search,\n',
     '            "health": health, "hud": hud, "clean": clean, "objects": objects, "explore": explore, "search": search,\n'
     '            "changes": changes, "report": report,\n'),
    ('    for name in ("check", "plan", "execute", "solve", "health", "hud", "clean", "objects", "explore", "search"):\n',
     '    for name in ("check", "plan", "execute", "solve", "health", "hud", "clean", "objects", "explore", "search", "changes"):\n'),
    ('    for name in ("norm", "diff", "grid_of", "cells"):\n',
     '    for name in ("norm", "diff", "grid_of", "cells", "report"):\n'),
    (GATE_OLD, GATE_NEW),
]

PROMPT_ANCHOR = '    "- Use `print(...)` for compact summaries, or assign a final compact object to `result`.\\n"\n'
# first words of every world-model bullet written by rounds 1-3, search and this file (source form)
WM_BULLETS = ["- WORLD MODEL TOOLKIT", "- The three functions:", "- Save them with", "- `wm.check()` replays",
              "- `wm.plan()` searches", "- Typical game:", "- WM STATE TEMPLATES", "- Start crude:", "- MOVE games",
              "- CLICK games", "- Keep states canonical", "- ANIMATED actions", "- `is_goal`: describe",
              "- WHEN TO STOP MODELLING", "- NOTES vs SAVED CODE", "- GOAL UNKNOWN", "- NO PLAN FOUND", "- TURN BUDGET",
              "- ONE RESULT", "- TOOLBOX", "- Reusable code:"]
SOLVE_OLD = "`result = wm.solve()` each turn."
SOLVE_NEW = "`out = wm.solve()` each turn, ending the call with `result = wm.report(notes, out)`."

FULL_LINES = [
    "- ONE RESULT: `result` carries both your notes and the wm outcome -- end every call with "
    "`result = wm.report(notes, out)`, where `notes` is your {'goal', 'rules', 'plan', 'uncertain'} dict and `out` "
    "is what `wm.solve()` / `wm.explore()` / `wm.check()` returned (omit it if you made no wm call). Assigning the "
    "wm outcome to `result` directly would drop your notes. `wm.changes()` summarises what the last action did "
    "(cells changed, objects moved, HUD ignored) -- use it instead of writing a diff.",
]


def toolbox_lines(persist_call):
    return [
        "- TOOLBOX (`wm`, preloaded, costs you nothing): `wm.changes()` -> what the last action did with the HUD "
        "ignored: cells changed, bounding box, color transitions, which objects moved from where to where, shapes that "
        "appeared or disappeared (`wm.changes(i)` for an earlier action). `wm.objects(grid)` -> sorted tuple of "
        "`(color, top, left, height, width, cells)` per connected shape; `wm.cells(grid, color, ...)` -> set of "
        "`(row, col)`; `wm.hud()` -> the timer cells to ignore. Grids are `current_frame.grid` and "
        "`transitions[i].before_frame.grid` / `.after_frame.grid`. Call these instead of rewriting diff or "
        "segmentation code each turn.",
        f"- Reusable code: a helper you would otherwise retype (a search, a move simulator) can be saved with "
        f"`{persist_call}`; saved code is re-run at the start of every later call. Facts stay in your notes, code in "
        "saved functions -- never both. End each call with `result = wm.report(notes)`.",
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


def patch_wm_source(src, mode="full"):
    """search-level WM_SOURCE -> with changes/report/stuck rule and the mode flag. Re-running switches the mode."""
    assert mode in ("full", "toolbox")
    if MARK_SRC not in src:
        if "def search(G" not in src:
            raise SystemExit("WM_SOURCE has no wm.search -- apply wm_search.py first")
        src = _patch(src, WM_EDITS, "WM_SOURCE")
    cur = [m for m in ("full", "toolbox") if MODE_LINE % m in src]
    assert len(cur) == 1, "mode flag missing"
    src = src.replace(MODE_LINE % cur[0], MODE_LINE % mode)
    compile(src, "<wm>", "exec")
    return src


def strip_wm_bullets(text):
    """Drop every world-model bullet line from prompts.py (each bullet is one source line)."""
    out, dropped = [], 0
    for line in text.split("\n"):
        s = line.strip()
        if s.startswith('"') and any(s[1:].startswith(b) for b in WM_BULLETS):
            dropped += 1
            continue
        out.append(line)
    return "\n".join(out), dropped


def patch_prompts(text, mode="full", persist_call="persist(code)"):
    assert mode in ("full", "toolbox")
    block = lambda lines: "".join("    " + _py_str(line + "\n") + "\n" for line in lines)       # noqa: E731
    if mode == "toolbox":
        text, _ = strip_wm_bullets(text)
        out = _patch(text, [(PROMPT_ANCHOR, block(toolbox_lines(persist_call)) + PROMPT_ANCHOR)], "prompts.py")
    else:
        if "- TOOLBOX" in text:
            raise SystemExit("prompts.py is in toolbox mode (world-model bullets removed); start again from the "
                             "round-3/search bundle to build --mode full")
        out = text
        if "- ONE RESULT" not in out:
            out = _patch(out, [(PROMPT_ANCHOR, block(FULL_LINES) + PROMPT_ANCHOR)], "prompts.py")
        esc = lambda s: s                                                                   # backticks need no escaping
        if esc(SOLVE_OLD) in out:
            out = _patch(out, [(esc(SOLVE_OLD), esc(SOLVE_NEW))], "prompts.py")
    compile(out, "prompts.py", "exec")
    return out


def apply(root, mode="full", persist_call="persist(code)", dry_run=False):
    agent = os.path.join(root, "inference", "agent")
    ws_path, pr_path = os.path.join(agent, "wm_source.py"), os.path.join(agent, "prompts.py")
    if not os.path.exists(ws_path):
        raise SystemExit("wm_source.py not found -- apply rounds 1-3 and wm_search.py first")
    ns = {}
    ws_text = open(ws_path).read()
    exec(compile(ws_text, ws_path, "exec"), ns)
    pr_text = open(pr_path).read()
    new_src, new_pr = patch_wm_source(ns["WM_SOURCE"], mode), patch_prompts(pr_text, mode, persist_call)
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
    import wm_round2 as r2
    import wm_search
    notes = "{'goal': 'box on target', 'rules': {'in_code': 'v1'}, 'plan': 'wm.solve', 'uncertain': []}"

    # ---------------- full mode
    with tempfile.TemporaryDirectory() as tmp:
        sb, prompts, wm_source, dst = wm_search.patched_sandbox(
            tmp, extra=[lambda d: [apply(d, "full", dry_run=True), apply(d, "full")]])
        assert apply(dst, "full") == "already patched"
        pa = prompts.PYTHON_ADDENDUM
        assert "ONE RESULT" in pa and SOLVE_NEW in pa and SOLVE_OLD not in pa and "TOOLBOX" not in pa
        assert pa.index("NO PLAN FOUND") < pa.index("ONE RESULT") < pa.index("- Use `print(...)`")
        assert '_WM_MODE = "full"' in wm_source.WM_SOURCE
        g = r2.MoveGame()
        call = r2._harness(sb, g)
        # changes(): the avatar's move, HUD tick excluded
        r = call("action(['RIGHT'])\nresult = [wm.changes(), wm.changes(0), wm.changes(5)]", r2.MOVE_RULES)
        assert r[0] == r[1] and r[0]["action"] == "RIGHT" and r[0]["changed"] == 32 and not r[0]["hud_only"], r[0]
        assert r[0]["moved"] == [{"color": 1, "size": [4, 4], "from": [24, 4], "to": [24, 8]}] and "error" in r[2], r
        assert r[0]["colors"] == {"1>0": 16, "0>1": 16} and r[0]["bbox"] == [24, 4, 27, 11]
        r = call("action(['DOWN'])\nresult = wm.changes()", r2.MOVE_RULES)             # blocked by the wall: HUD only
        assert r["changed"] == 0 and r["hud_only"] is True, r
        assert call("result = wm.changes()", "")["action"] == "DOWN"                  # needs no rules at all
        # report(): notes survive next to a compact wm outcome
        r = call("notes = %s\nout = wm.solve()\nresult = wm.report(notes, out)" % notes, r2.MOVE_RULES)
        assert r["notes"]["rules"] == {"in_code": "v1"} and r["wm"]["stage"] == "execute", r
        assert r["wm"]["level_completed"] is True and r["wm"]["stopped"] == "terminal_or_level_change" and "plan" not in r["wm"]
        r = call("result = [wm.report({'goal': '?'}), wm.report({'g': 1}, wm.check())]", r2.MOVE_RULES)
        assert r[0] == {"notes": {"goal": "?"}, "wm": {}} and set(r[1]["wm"]) <= {"acc", "n"}, r
        r = call("result = wm.report({}, wm.solve())", r2.MOVE_RULES_WRONG)             # a failed check is reported compactly
        # stuck in the fix zone: 4 of 5 actions predicted (0.8) -> 'fix' at 20 actions, 'drop' at 40
        g = r2.MoveGame()
        call = r2._harness(sb, g)
        up_wrong = r2.MOVE_RULES.replace('D = {"UP": (-1, 0),', 'D = {"UP": (-2, 0),')
        assert up_wrong != r2.MOVE_RULES
        cyc = ["RIGHT", "LEFT", "RIGHT", "UP", "DOWN"]
        r = call("action(%r)\nresult = wm.health()" % (cyc * 4), up_wrong)
        assert r["verdict"] == "fix" and 0.6 <= r["acc_recent"] < 0.9 and r["n"] == 20, r
        r = call("action(%r)\nresult = [wm.health(), wm.solve()['stage'], wm.report({}, wm.solve())]" % (cyc * 4), up_wrong)
        assert r[0]["verdict"] == "drop" and "still below 0.9" in r[0]["why"] and r[1] == "fallback", r
        assert r[2]["wm"]["verdict"] == "drop" and g.steps == 40
        full_words = len(" ".join(FULL_LINES).split())

    # ---------------- toolbox mode
    with tempfile.TemporaryDirectory() as tmp:
        sb, prompts, wm_source, dst = wm_search.patched_sandbox(tmp, extra=[lambda d: apply(d, "toolbox", "persist(src)")])
        assert apply(dst, "toolbox", "persist(src)") == "already patched"
        pa = prompts.PYTHON_ADDENDUM
        assert "TOOLBOX" in pa and "persist(src)" in pa
        gone = ["WORLD MODEL", "parse(grid)", "is_goal", "wm.solve", "wm.explore", "wm.search", "wm.check", "wm.plan"]
        assert not [w for w in gone if w in pa], [w for w in gone if w in pa]
        assert pa.count("- TOOLBOX") == 1 and pa.index("- TOOLBOX") < pa.index("- Use `print(...)`")
        base_pa = __import__("importlib").import_module("inference.agent.prompts")       # same module, already loaded
        assert base_pa is prompts
        assert '_WM_MODE = "toolbox"' in wm_source.WM_SOURCE
        g = r2.MoveGame()
        call = r2._harness(sb, g)
        r = call("action(['RIGHT'])\nresult = [wm.solve(), wm.explore(), wm.search(), wm.changes()['moved'][0]['to'], "
                 "len(wm.objects(current_frame.grid)), wm.report({'goal': 'x'})]", r2.MOVE_RULES)
        assert [x["stage"] for x in r[:3]] == ["toolbox"] * 3 and g.steps == 1, r
        assert r[3] == [24, 8] and r[4] == 4 and r[5] == {"notes": {"goal": "x"}, "wm": {}}
        r = call("result = wm.check()['acc']", r2.MOVE_RULES)                           # still usable by a model that wants it
        assert r == 1.0
        try:
            patch_prompts(open(os.path.join(dst, "inference", "agent", "prompts.py")).read(), "full")
            raise SystemExit("full mode applied on a toolbox prompt")
        except SystemExit as e:
            assert "toolbox mode" in str(e)
        # the prompt shrinks back: only the two toolbox bullets remain of all wm text
        src_text = open(os.path.join(dst, "inference", "agent", "prompts.py")).read()
        _, n_left = strip_wm_bullets(src_text)
        assert n_left == 2
        tb_words = len(" ".join(toolbox_lines("persist(src)")).split())
        # switching the source mode back works in place
        assert '_WM_MODE = "full"' in patch_wm_source(wm_source.WM_SOURCE, "full")

    print("selftest ok (real sandbox subprocess, rounds 1-3 + search + modes on a copy of bundle_fast):")
    print("  full: wm.changes() reports the avatar's move (32 cells, from/to) and HUD-only no-ops; wm.report keeps the "
          "notes next to a compact outcome; 'fix' at 20 actions becomes 'drop' at 40 when accuracy stays at 0.8")
    print("  toolbox: solve/explore/search return stage 'toolbox' and spend nothing; helpers work; every world-model "
          f"bullet is gone from the prompt ({tb_words} words remain vs ~1,000 in full mode + {full_words} for ONE RESULT)")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) >= 3 and sys.argv[1] == "--print-prompt":
        print("\n\n".join(FULL_LINES if sys.argv[2] == "full" else toolbox_lines("persist(code)")))
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("root")
        ap.add_argument("--mode", choices=["full", "toolbox"], required=True)
        ap.add_argument("--persist-call", default="persist(code)")
        ap.add_argument("--dry-run", action="store_true")
        a = ap.parse_args()
        print(apply(a.root, a.mode, a.persist_call, a.dry_run))
