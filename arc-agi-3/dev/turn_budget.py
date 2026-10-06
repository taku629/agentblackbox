"""Turn-budget governor for the wm bundle: wm.budget() + earlier model give-up when turns run short.

    python3 turn_budget.py <bundle>/src/ARC3-Inference [--dry-run] [--game-seconds 7920]
    python3 turn_budget.py --selftest
    python3 turn_budget.py --print-prompt

Applies ON TOP of apply_wm_patch.py, wm_round2.py, wm_round3.py, wm_search.py and wm_modes.py (either
mode). Anchors must match exactly once or nothing is written; re-running is a no-op.

Why (ag3_autopsy on the notes run): 70% of all turns are spent on the level the game dies on;
7 games are `starved` (fewer actions on that level than a human needs), 10 are `no_mechanics`.
Levels are sequential, so a level cannot be skipped -- what CAN be cut is the time spent on an
approach that is not working. The governor reads only the recorded transitions of the current level
(no memory between calls, resets on a new level) and says which of these holds, first match wins:

  probe         some valid action was never tried on this level        -> try each once, in ONE call
  dead_actions  >= 2/3 of the last 12 actions changed nothing (HUD ignored) -> names the dead actions
  cycling       >= 2/3 of the last 12 actions led to boards already seen    -> change sequence / wm.explore()
  goal          the model reproduces the board (health 'use') but 30+ actions did not finish the level
                                                                        -> is_goal is the suspect; progress()+wm.search()
  no_mechanics  60+ effective, novel actions and still no level         -> state ONE goal hypothesis, test it in <= 10 actions
  ok            none of the above
plus, independent of the phase, a pace line when the game averages < 3 actions per turn:
  "send at least 6 actions per call" (turns left = time_remaining_seconds / 150, as in wm.search; turns used
  = (--game-seconds - time_remaining_seconds) / 150, so --game-seconds must be the kernel's per-game cap).

Where the model sees it: wm.report(notes, out) adds 'budget' (phase + one sentence) whenever the phase is
not ok, and so do the dict results of wm.solve / wm.explore / wm.search. wm.budget() returns the full
numbers. Nothing is shown while things are fine, so the governor costs no tokens then.

Hard effect (not advice): wm.health() gives up on a model that stays below 0.9 accuracy after
  40 actions on the level with > 35 turns left (unchanged), 30 with 20-35 turns left, 20 with < 20 turns left.
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARK_SRC = "def budget(G"
MARK_PROMPT = "- BUDGET LINE"

BUDGET = r'''
    def turns_left_of(G):
        tr = (G.get("last_action_result") or {}).get("time_remaining_seconds")
        return tr / 150.0 if isinstance(tr, (int, float)) else None

    def stuck_after(G, drop_after):
        """Actions on a level after which a model still below 0.9 accuracy is dropped."""
        tl = turns_left_of(G)
        if tl is None or tl > 35:
            return 2 * drop_after
        return int(1.5 * drop_after) if tl >= 20 else drop_after

    def budget(G, window=12, game_seconds=None):
        """Is the current approach to this level working? Phase + one instruction, from recorded history only."""
        game_seconds = _WM_GAME_SECONDS if game_seconds is None else game_seconds
        trs = G.get("transitions", [])
        lvl = G["current_frame"].level
        hudc, diffs = scan(G)
        name = lambda a: a[0] if isinstance(a, tuple) else a
        rows = []
        for t, d in zip(trs, diffs):
            if t.before_frame is None or t.before_frame.level != lvl:
                continue
            a = name(norm(t.action))
            if a == "RESET":
                continue
            real = d is None or t.after_frame.level != lvl or any(p not in hudc for p in d)
            rows.append((t, a, real))
        n = len(rows)
        valid = []
        for a in G.get("valid_actions", []):
            a = name(norm(a))
            if a != "RESET" and a not in valid:
                valid.append(a)
        uses, moved = {}, {}
        for _, a, real in rows:
            uses[a] = uses.get(a, 0) + 1
            moved[a] = moved.get(a, 0) + (1 if real else 0)
        untried = [a for a in valid if a not in uses]
        dead = sorted(a for a in uses if uses[a] >= 2 and moved[a] == 0)
        tail = rows[-window:]
        eff = round(sum(1 for r in tail if r[2]) / len(tail), 2) if tail else None
        seen, fresh = set(), []
        for i, (t, _, _) in enumerate(rows):
            if i == 0:
                seen.add(clean(G, grid_of(t.before_frame)))
            b = clean(G, grid_of(t.after_frame))
            fresh.append(b not in seen)
            seen.add(b)
        novel = round(sum(fresh[-window:]) / len(tail), 2) if tail else None
        out = {"phase": "ok", "n": n, "effective": eff, "novel": novel}
        tl = turns_left_of(G)
        if tl is not None:
            out["turns_left"] = round(tl, 1)
            used = (game_seconds - tl * 150.0) / 150.0
            if used >= 3:
                out["actions_per_turn"] = round(len([t for t in trs if t.before_frame is not None]) / used, 1)
        if untried:
            out["untried"] = untried
        if dead:
            out["dead"] = dead
        has_model = callable(G.get("parse")) and callable(G.get("step"))
        if untried and n < 40:
            out["phase"] = "probe"
            out["do"] = "untried on this level: %s -- try each once, all in ONE call, before anything else" % ", ".join(untried)
        elif len(tail) >= 6 and eff <= 0.34:
            out["phase"] = "dead_actions"
            out["do"] = ("%d of your last %d actions changed nothing%s. Stop repeating them; use the actions that did "
                         "change the board or click a different object"
                         % (sum(1 for r in tail if not r[2]), len(tail), " (dead here: %s)" % ", ".join(dead) if dead else ""))
        elif len(tail) >= 8 and novel <= 0.34:
            out["phase"] = "cycling"
            out["do"] = ("only %d of your last %d actions reached a board you had not seen on this level: you are going "
                         "in circles. Try a sequence you have not played%s"
                         % (sum(fresh[-window:]), len(tail), ", or wm.explore()" if has_model else ""))
        elif has_model and n >= 30 and health(G).get("verdict") == "use":
            out["phase"] = "goal"
            out["do"] = ("your model predicts the board, yet %d actions did not finish the level: is_goal is the suspect. "
                         "Compare the board just before earlier levels ended (wm.changes(i) of that action), fix is_goal, "
                         "or define progress(state) and call wm.search()" % n)
        elif n >= 60:
            out["phase"] = "no_mechanics"
            out["do"] = ("%d actions on this level without finishing it. Write ONE goal hypothesis in your notes, test it "
                         "in at most 10 actions, and discard it if the board does not respond as it predicts" % n)
        apt = out.get("actions_per_turn")
        if apt is not None and apt < 3 and tl is not None:
            out["pace"] = "%.1f actions per turn with ~%d turns left: send at least 6 actions per call" % (apt, int(tl))
        return out

    def budget_line(G):
        """Short form for results: None while everything is fine."""
        try:
            b = budget(G)
        except Exception:
            return None
        parts = ([b["phase"] + ": " + b["do"]] if b["phase"] != "ok" else []) + ([b["pace"]] if b.get("pace") else [])
        return " | ".join(parts) or None

'''

GAME_ANCHOR = '\n\ndef _wm_build():\n'

WM_EDITS = [
    ('    def report(notes, out=None):\n', BUDGET + '    def report(notes, out=None):\n'),
    ('        elif n >= 2 * drop_after and acc < 0.9:\n', '        elif n >= stuck_after(G, drop_after) and acc < 0.9:\n'),
    ('            "changes": changes, "report": report,\n',
     '            "changes": changes, "report": report, "budget": budget, "budget_line": budget_line,\n'),
    ('    for name in ("check", "plan", "execute", "solve", "health", "hud", "clean", "objects", "explore", "search", "changes"):\n'
     '        ns[name] = (lambda f: (lambda *a, **k: f(G(), *a, **k)))(fns[name])\n',
     '    for name in ("check", "plan", "execute", "solve", "health", "hud", "clean", "objects", "explore", "search", "changes",\n'
     '                 "budget"):\n'
     '        ns[name] = (lambda f: (lambda *a, **k: f(G(), *a, **k)))(fns[name])\n'
     '\n'
     '    def noted(f):\n'
     '        def call(*a, **k):\n'
     '            r = f(*a, **k)\n'
     '            line = fns["budget_line"](G()) if isinstance(r, dict) else None\n'
     '            if line:\n'
     '                r["budget"] = line\n'
     '            return r\n'
     '        return call\n'
     '    for name in ("solve", "explore", "search"):\n'
     '        ns[name] = noted(ns[name])\n'),
    ('    for name in ("norm", "diff", "grid_of", "cells", "report"):\n'
     '        ns[name] = fns[name]\n',
     '    for name in ("norm", "diff", "grid_of", "cells", "report"):\n'
     '        ns[name] = fns[name]\n'
     '    ns["report"] = noted(fns["report"])\n'),
]

PROMPT_ANCHOR = '    "- Use `print(...)` for compact summaries, or assign a final compact object to `result`.\\n"\n'
PROMPT_LINES = [
    "- BUDGET LINE: when a `wm` result or `wm.report(...)` contains a `budget` entry, your current approach to this "
    "level is not working -- do what it says in the SAME turn before anything else (it is computed from your own "
    "action history: untried actions, actions that change nothing, boards you keep revisiting, too few actions per "
    "turn). `wm.budget()` shows the numbers. No `budget` entry means carry on.",
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


def patch_wm_source(src, game_seconds=7920.0):
    """modes-level WM_SOURCE -> with the governor. Idempotent."""
    if MARK_SRC in src:
        return src
    if "def report(notes" not in src:
        raise SystemExit("WM_SOURCE has no wm.report -- apply wm_modes.py first")
    edits = WM_EDITS + [(GAME_ANCHOR, "\n_WM_GAME_SECONDS = %r" % float(game_seconds) + GAME_ANCHOR)]
    out = _patch(src, edits, "WM_SOURCE")
    compile(out, "<wm>", "exec")
    return out


def patch_prompts(text):
    if MARK_PROMPT in text:
        return text
    block = "".join("    " + _py_str(line + "\n") + "\n" for line in PROMPT_LINES)
    out = _patch(text, [(PROMPT_ANCHOR, block + PROMPT_ANCHOR)], "prompts.py")
    compile(out, "prompts.py", "exec")
    return out


def apply(root, dry_run=False, game_seconds=7920.0):
    agent = os.path.join(root, "inference", "agent")
    ws_path, pr_path = os.path.join(agent, "wm_source.py"), os.path.join(agent, "prompts.py")
    if not os.path.exists(ws_path):
        raise SystemExit("wm_source.py not found -- apply rounds 1-3, wm_search.py and wm_modes.py first")
    ns = {}
    ws_text = open(ws_path).read()
    exec(compile(ws_text, ws_path, "exec"), ns)
    pr_text = open(pr_path).read()
    new_src, new_pr = patch_wm_source(ns["WM_SOURCE"], game_seconds), patch_prompts(pr_text)
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

    for mode in ("full", "toolbox"):
        with tempfile.TemporaryDirectory() as tmp:
            sb, prompts, wm_source, dst = wm_search.patched_sandbox(
                tmp, extra=[lambda d: wm_modes.apply(d, mode), lambda d: [apply(d, dry_run=True), apply(d, game_seconds=1000.0)]])
            assert apply(dst) == "already patched" and "_WM_GAME_SECONDS = 1000.0" in wm_source.WM_SOURCE
            pa = prompts.PYTHON_ADDENDUM
            assert pa.count("BUDGET LINE") == 1 and pa.index("BUDGET LINE") < pa.index("- Use `print(...)`")
            if mode == "toolbox":
                # the governor needs no world model: it works on the bare action history
                g = r2.MoveGame()
                call = r2._harness(sb, g)
                r = call("result = [wm.budget(), wm.report({'goal': '?'})]", "")
                assert r[0]["phase"] == "probe" and r[0]["untried"] == ["UP", "DOWN", "LEFT", "RIGHT"] and r[0]["n"] == 0, r
                assert r[1]["budget"].startswith("probe: untried on this level: UP, DOWN, LEFT, RIGHT") and r[1]["wm"] == {}
                r = call("action(['DOWN'] * 8)\nresult = [wm.budget(), wm.report({}), wm.solve()]", "")
                assert r[0]["phase"] == "probe" and r[0]["dead"] == ["DOWN"] and r[0]["effective"] == 0.0, r
                assert r[2]["stage"] == "toolbox" and "budget" not in r[2]                # switched-off verbs stay silent
                continue

            # ---- full mode, real sandbox subprocess, toy box-pushing game
            g = r2.MoveGame()
            call = r2._harness(sb, g)
            # 1. probe: names what was never tried; disappears once every action was used
            r = call("action(['RIGHT'])\nresult = wm.budget()", "")
            assert r["phase"] == "probe" and r["untried"] == ["UP", "DOWN", "LEFT"] and r["n"] == 1, r
            r = call("action(['LEFT', 'UP', 'DOWN'])\nresult = [wm.budget(), wm.report({'goal': 'x'})]", "")
            assert r[0]["phase"] == "ok" and "untried" not in r[0] and "budget" not in r[1], r
            # 2. dead actions: pushing into the wall changes only the HUD
            r = call("action(['DOWN'] * 9)\nresult = [wm.budget(), wm.report({}, None)]", "")
            assert r[0]["phase"] == "dead_actions" and r[0]["effective"] <= 0.34 and "dead" not in r[0], r   # DOWN moved once before
            assert r[1]["budget"].startswith("dead_actions: ") and "changed nothing" in r[1]["budget"], r[1]
            # 3. cycling: every action changes the board, none reaches a new one
            g = r2.MoveGame()
            call = r2._harness(sb, g)
            r = call("action(['UP', 'DOWN', 'LEFT', 'RIGHT'] + ['RIGHT', 'LEFT'] * 6)\nresult = [wm.budget(), wm.explore()]", r2.MOVE_RULES)
            assert r[0]["phase"] == "cycling" and r[0]["effective"] >= 0.9 and r[0]["novel"] <= 0.34, r[0]
            assert "wm.explore()" in r[0]["do"] and "budget" in r[1], r
            # 4. goal: an accurate model, 30+ actions, level not finished
            g = r2.MoveGame()
            call = r2._harness(sb, g)
            import copy
            sim, walk, seen = r2.MoveGame(), ["UP", "DOWN", "LEFT", "RIGHT"], set()
            for a in walk:
                sim.act({"action": a})
            seen.add((sim.p, sim.box))
            def extend(sim, path, seen):                           # depth-first walk onto unseen (player, box) states
                if len(path) == 30:
                    return path
                for a in ("RIGHT", "UP", "LEFT", "DOWN"):
                    trial = copy.deepcopy(sim)
                    # (the toy rules do not model the player covering the goal cell, so stay off it)
                    if trial.act({"action": a}) or (trial.p, trial.box) in seen or trial.p == trial.goal:
                        continue
                    got = extend(trial, path + [a], seen | {(trial.p, trial.box)})
                    if got:
                        return got
                return None
            walk = walk + extend(sim, [], seen)
            r = call("action(%r)\nresult = [wm.budget(), wm.health()['verdict']]" % walk, r2.MOVE_RULES)
            assert r[1] == "use" and r[0]["phase"] == "goal" and "is_goal is the suspect" in r[0]["do"], r
            # 4b. no_mechanics: 60+ clicks that all change the board and keep reaching new boards, no model
            g = r2.ClickGame()
            call = r2._harness(sb, g)
            clicks = [{"action": "MOUSE", "row": p[0] * r2.S + 1, "col": p[1] * r2.S + 1}
                      for p in (r2.ClickGame.POS[(i * i + i // 9) % 9] for i in range(66))]
            r = call("action(%r)\nresult = wm.budget()" % clicks, "")
            assert g.level == 1 and r["n"] == 66 and r["phase"] == "no_mechanics" and "ONE goal hypothesis" in r["do"], r
            # 5. pace: few actions per turn -> the pace line, independent of the phase
            g = r2.MoveGame()
            call = r2._harness(sb, g)
            r = call("action(['UP', 'DOWN', 'LEFT', 'RIGHT'])\nresult = [wm.budget(game_seconds=4000.0), wm.budget()]", "")
            assert r[0]["actions_per_turn"] < 3 and "send at least 6 actions per call" in r[0]["pace"], r[0]
            assert "pace" not in r[1], r[1]
            # 6. the hard rule: a model stuck at 0.8 is dropped at 20 actions when < 20 turns are left,
            #    and only at 40 when the clock is far away (same history, different clock)
            up_wrong = r2.MOVE_RULES.replace('D = {"UP": (-1, 0),', 'D = {"UP": (-2, 0),')
            assert up_wrong != r2.MOVE_RULES
            cyc = ["RIGHT", "LEFT", "RIGHT", "UP", "DOWN"]
            g = r2.MoveGame()
            call = r2._harness(sb, g)
            code = ("action(%r)\n"
                    "last_action_result['time_remaining_seconds'] = 150.0 * 50\n"
                    "far = wm.health()\n"
                    "last_action_result['time_remaining_seconds'] = 150.0 * 25\n"
                    "mid = wm.health()\n"
                    "last_action_result['time_remaining_seconds'] = 150.0 * 10\n"
                    "near = wm.health()\n"
                    "result = [far, mid, near, wm.solve()['stage']]" % (cyc * 4))
            r = call(code, up_wrong)
            assert r[0]["n"] == 20 and 0.6 <= r[0]["acc_recent"] < 0.9, r[0]
            assert [x["verdict"] for x in r[:3]] == ["fix", "fix", "drop"] and r[3] == "fallback", [x["verdict"] for x in r[:3]]
            r = call("action(%r)\nlast_action_result['time_remaining_seconds'] = 150.0 * 25\nresult = wm.health()" % (cyc * 2), up_wrong)
            assert r["n"] == 30 and r["verdict"] == "drop" and "still below 0.9" in r["why"], r
            # 7. a level change resets everything
            g = r2.MoveGame()
            call = r2._harness(sb, g)
            r = call("out = wm.solve()\nresult = [out['execute'].get('level_completed'), wm.budget()]", r2.MOVE_RULES)
            assert r[0] is True and r[1]["n"] == 0 and r[1]["phase"] == "probe" and g.level == 2, r
            words = sum(len(x.split()) for x in PROMPT_LINES)

    print("selftest ok (real sandbox subprocess, rounds 1-3 + search + modes + governor on a copy of bundle_fast):")
    print("  phases probe / dead_actions / cycling / goal / no_mechanics fire on the toy games from the action history alone; the pace "
          "line appears below 3 actions per turn; a level change resets the history")
    print("  health(): the same 0.8-accuracy model is 'fix' at 20 actions with time to spare and 'drop' with < 20 turns "
          "left; at 30 actions it is dropped with 25 turns left")
    print(f"  toolbox mode: wm.budget() and the budget entry in wm.report work with no world model; prompt: {words} words")


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--print-prompt":
        print("\n\n".join(PROMPT_LINES))
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("root")
        ap.add_argument("--dry-run", action="store_true")
        ap.add_argument("--game-seconds", type=float, default=7920.0, help="per-game wallclock cap of the kernel")
        a = ap.parse_args()
        print(apply(a.root, a.dry_run, a.game_seconds))
