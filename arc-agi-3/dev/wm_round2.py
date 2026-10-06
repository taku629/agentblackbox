"""Round 2 for the wm arm: state helpers + templates, a host-side fallback
verdict, and the notes-vs-persist rule. Anchor patches only.

    python3 wm_round2.py <bundle>/src/ARC3-Inference [--dry-run]   # patch wm_source.py + prompts.py in place
    python3 wm_round2.py --selftest                                # real sandbox, three toy game families
    python3 wm_round2.py --print-prompt                            # the prose that gets inserted

Applies ON TOP of apply_wm_patch.py (needs inference/agent/wm_source.py).
Every anchor must match exactly once or nothing is written; re-running is a
no-op.

What it adds to WM_SOURCE (host preload -> zero model tokens):
  wm.objects(grid)   sorted tuple of (color, top, left, height, width, cells) per connected
                     shape; background and HUD cells removed
  wm.cells(grid, *colors)   frozenset of (row, col)
  wm.clean(grid)     whole board as a hashable tuple, HUD cells blanked -> a valid first parse()
  wm.hud()           cells that changed as an isolated cluster of <=4 cells (tool_agent's <=4-cell
                     volatile rule applied per cluster, same 64-cell cap)
  wm.health()        verdict from recorded history only: probe | use | fix | drop
  wm.solve()         returns {"stage": "fallback"} on a `drop` verdict instead of planning
  wm.check()         now defaults to the CURRENT level (level=None = all levels)

Why the fallback verdict lives in wm and not in the model's judgement: it is
computed from the transition history, which the host re-sends every call, so
it needs no cross-call memory, costs no model tokens, cannot be argued with,
and recomputes itself on the next level. The model keeps the decision only in
the fixable zone (`stage: check`).
"""
import argparse
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
MARK_SRC = "def health(G"
MARK_PROMPT = "WM STATE TEMPLATES"

HELPERS = r'''
    _scan_cache = {}

    def scan(G):
        """(hud cells, per-transition changed-cell lists). HUD = cells that
        changed as an isolated cluster of <=4 cells (the host's <=4-cell rule,
        applied per cluster so ticks are found even on moves), capped at ~64."""
        trs = G.get("transitions", [])
        if len(trs) in _scan_cache:
            return _scan_cache[len(trs)]
        hudc, diffs = set(), []
        for t in trs:
            if t.before_frame is None:
                diffs.append(None)
                continue
            a, b = grid_of(t.before_frame), grid_of(t.after_frame)
            if a is None or b is None or len(a) != len(b):
                diffs.append(None)
                continue
            d = []
            for r in range(len(a)):
                ra, rb = a[r], b[r]
                if ra != rb:
                    d.extend((r, c) for c in range(len(ra)) if ra[c] != rb[c])
            diffs.append(d)
            if not d or len(d) > 600 or len(hudc) > 64:
                continue
            # isolated change clusters of <=4 cells are HUD ticks even when a real move
            # (dozens of cells) happens in the same action
            left = set(d)
            while left:
                seed = left.pop()
                comp, stack = [seed], [seed]
                while stack:
                    y, x = stack.pop()
                    for q in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1),
                              (y + 1, x + 1), (y + 1, x - 1), (y - 1, x + 1), (y - 1, x - 1)):
                        if q in left:
                            left.discard(q)
                            comp.append(q)
                            stack.append(q)
                if len(comp) <= 4:
                    hudc.update(comp)
        out = (frozenset(hudc), diffs)
        _scan_cache.clear()
        _scan_cache[len(trs)] = out
        return out

    def hud(G):
        return scan(G)[0]

    def cells(grid, *colors):
        return frozenset((r, c) for r in range(len(grid)) for c in range(len(grid[r])) if grid[r][c] in colors)

    def clean(G, grid, keep_hud=False):
        H = () if keep_hud else scan(G)[0]
        return tuple(tuple(-1 if (r, c) in H else grid[r][c] for c in range(len(grid[r]))) for r in range(len(grid)))

    def objects(G, grid, bg=None, keep_hud=False):
        """4-connected same-colour shapes as a sorted, hashable tuple of
        (color, top, left, height, width, cells). bg defaults to the most
        common colour."""
        seen = set() if keep_hud else set(scan(G)[0])
        if bg is None:
            cnt = {}
            for row in grid:
                for v in row:
                    cnt[v] = cnt.get(v, 0) + 1
            bg = max(cnt, key=cnt.get) if cnt else 0
        out = []
        R = len(grid)
        for r in range(R):
            row = grid[r]
            for c in range(len(row)):
                col = row[c]
                if col == bg or (r, c) in seen:
                    continue
                seen.add((r, c))
                stack = [(r, c)]
                r0 = r1 = r
                c0 = c1 = c
                n = 0
                while stack:
                    y, x = stack.pop()
                    n += 1
                    if y < r0: r0 = y
                    if y > r1: r1 = y
                    if x < c0: c0 = x
                    if x > c1: c1 = x
                    for ny, nx in ((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)):
                        if 0 <= ny < R and 0 <= nx < len(grid[ny]) and (ny, nx) not in seen and grid[ny][nx] == col:
                            seen.add((ny, nx))
                            stack.append((ny, nx))
                out.append((col, r0, c0, r1 - r0 + 1, c1 - c0 + 1, n))
        return tuple(sorted(out))

'''

HEALTH = r'''
    def health(G, recent=12, min_n=4, drop_after=20, window=60):
        """Should this level be modelled at all? Decided from recorded history
        only, so it needs no memory between calls and resets on a new level.
          probe  not enough transitions on this level yet
          use    the model reproduces recent history
          fix    wrong but fixable -- read wm.check()['mismatch']
          drop   the board does not determine the outcome under this state
                 (or accuracy stayed low for drop_after+ actions): play directly"""
        err = missing(G, ("parse", "step"))
        if err:
            return {"verdict": "probe", "n": 0, "why": err["error"]}
        parse, step = G["parse"], G["step"]
        lvl = G["current_frame"].level
        hudc, diffs = scan(G)
        rows = []
        for t, d in zip(G.get("transitions", []), diffs):
            if t.before_frame is None or t.before_frame.level != lvl or t.after_frame.level != lvl:
                continue
            a = norm(t.action)
            if (a[0] if isinstance(a, tuple) else a) == "RESET":
                continue
            rows.append((t, a, d))
        rows = rows[-window:]
        outcomes, hits, blind, errs = {}, [], 0, 0
        for t, a, d in rows:
            try:
                s0, s1 = parse(grid_of(t.before_frame)), parse(grid_of(t.after_frame))
                hits.append(step(s0, a) == s1)
                if hashable(s0) and hashable(s1):
                    outcomes.setdefault((s0, a), set()).add(s1)
                if s0 == s1 and d is not None and len([p for p in d if p not in hudc]) > 4:
                    blind += 1
            except Exception:
                errs += 1
                hits.append(False)
        n = len(hits)
        tail = hits[-recent:]
        acc = round(sum(tail) / len(tail), 3) if tail else None
        conflicts = sum(1 for v in outcomes.values() if len(v) > 1)
        out = {"n": n, "acc_recent": acc, "conflicts": conflicts}
        if blind:
            out["blind"] = blind
        if errs:
            out["errors"] = errs
        if n < min_n:
            out["verdict"] = "probe"
            out["why"] = "only %d transitions on this level" % n
        elif conflicts >= 2:
            out["verdict"] = "drop"
            out["why"] = ("the same state+action gave different results %d times: something that decides the "
                          "outcome is not on the board as your state sees it" % conflicts)
        elif n >= drop_after and acc < 0.6:
            out["verdict"] = "drop"
            out["why"] = "accuracy %.2f after %d actions on this level" % (acc, n)
        elif blind >= max(2, n // 4):
            out["verdict"] = "fix"                       # accuracy is meaningless while the state is blind
            out["why"] = "parse() does not see %d real board changes -- add the moving thing to the state" % blind
        elif acc >= 0.9:
            out["verdict"] = "use"
        else:
            out["verdict"] = "fix"
            out["why"] = "step() mispredicts -- see wm.check()['mismatch']"
        return out

'''

WM_EDITS = [
    # helpers + health go in front of the existing guard-rail block
    ('    NEED = ("define parse(grid), step(state, action) and is_goal(state), then persist them; "\n',
     HELPERS + '    NEED = ("define parse(grid), step(state, action) and is_goal(state), then persist them; "\n'),
    # check(): current level by default (constants like WALL are per level)
    ('    def check(G, last=None, level=None):\n',
     '    def check(G, last=None, level="cur"):\n'),
    ('        if level is not None:\n            trs = [t for t in trs if t.before_frame.level == level]\n',
     '        if level == "cur":\n            level = G["current_frame"].level\n'
     '        if level is not None:\n            trs = [t for t in trs if t.before_frame.level == level]\n'),
    # solve(): consult the verdict before planning
    ('    def solve(G, h=None, max_nodes=20000, min_acc=0.9, last=40):\n',
     HEALTH + '    def solve(G, h=None, max_nodes=20000, min_acc=0.9, last=40):\n'),
    ('        c = missing(G) or check(G, last=last)\n        if "error" in c:\n            return c\n',
     '        c = missing(G) or check(G, last=last)\n        if "error" in c:\n            return c\n'
     '        hv = health(G)\n'
     '        if hv["verdict"] == "drop":\n'
     '            return {"stage": "fallback", "health": hv,\n'
     '                    "do": "stop editing rules for this level; play it directly; wm re-evaluates on the next level"}\n'),
    # exports
    ('    return {"check": check, "plan": plan, "execute": execute, "solve": solve,\n'
     '            "norm": norm, "diff": diff, "grid_of": grid_of}\n',
     '    return {"check": check, "plan": plan, "execute": execute, "solve": solve,\n'
     '            "health": health, "hud": hud, "clean": clean, "objects": objects,\n'
     '            "norm": norm, "diff": diff, "grid_of": grid_of, "cells": cells}\n'),
    ('    for name in ("check", "plan", "execute", "solve"):\n',
     '    for name in ("check", "plan", "execute", "solve", "health", "hud", "clean", "objects"):\n'),
    ('    for name in ("norm", "diff", "grid_of"):\n',
     '    for name in ("norm", "diff", "grid_of", "cells"):\n'),
]

PROMPT_ANCHOR = '    "- Use `print(...)` for compact summaries, or assign a final compact object to `result`.\\n"\n'

PROMPT_LINES = [
    # 1. templates per game family
    "- WM STATE TEMPLATES. Helpers already loaded (free): `wm.objects(grid)` -> sorted tuple of "
    "`(color, top, left, height, width, cells)` for every connected shape, background and HUD/timer cells already "
    "removed; `wm.cells(grid, color, ...)` -> frozenset of `(row, col)`; `wm.clean(grid)` -> the whole board as a "
    "hashable tuple with HUD cells blanked. Colors are the integers in `current_frame.grid`.",
    "- Start crude: `def parse(g): return wm.clean(g)` is a valid state for `wm.check()` while you are still learning "
    "what actions do. Shrink it to object positions before `wm.plan()` -- search needs small states.",
    "- MOVE games (arrow actions move an avatar or pieces): state = `(avatar_pos, tuple(sorted(movable_positions)), "
    "tuple(sorted(open_targets)))` built from `wm.objects`. Things that never move (walls) go in a constant computed "
    "from the board, e.g. `WALL = wm.cells(current_frame.grid, wall_color)`, not in the state. Measure the step size "
    "from one real move (a press usually moves a whole tile = several cells) and work in tile units. `step` = move by "
    "that delta unless the destination is blocked; pushed pieces move too.",
    "- CLICK games (MOUSE): state = tuple of every clickable shape's `(top, left, color)`; `actions(state)` = one "
    "`('MOUSE', row, col)` at each shape's centre, never one per cell; `step` = what one click changes (the shape, "
    "often its neighbours or its row/column).",
    "- Keep states canonical: build them with `tuple(sorted(...))` in BOTH `parse` and `step`. `wm.objects` sorts by "
    "color first, so a shape that changes color changes position in the tuple -- an unsorted `step` result then "
    "never equals the parsed board even when the rule is right.",
    "- ANIMATED actions: model only the final board of each action. An action that animates but leaves the board "
    "unchanged is a rejected move: `step` returns the same state.",
    "- `is_goal`: describe the solved board in state terms (no open target left, all shapes one color, avatar on the "
    "exit). After a level-up, `wm.check(level=None)['goal']` tells you whether your `is_goal` would have predicted it.",
    # 2. fallback
    "- WHEN TO STOP MODELLING: `wm.solve()` decides this from the recorded history. `stage: 'fallback'` means this "
    "level cannot be predicted from the board with your state (the same state+action gave different results, or "
    "accuracy stayed low for 20+ actions): stop editing rules for this level and play it directly, exactly as you "
    "would without `wm`; the verdict is recomputed on the next level. `stage: 'check'` means fixable: read "
    "`pred_vs_actual`, fix that one rule, save, retry. `wm.health()` shows the verdict without acting.",
    # 3. notes vs saved code
    "- NOTES vs SAVED CODE: every rule lives in exactly one place. What `parse`/`step` already encode appears in your "
    "notes only as a version tag, e.g. `'rules': {'in_code': 'v3: push + ice slide'}` -- never the logic again. "
    "`rules` otherwise holds only facts not yet in code; untested guesses go in `uncertain` with the probe that "
    "would settle them. `goal` and `plan` stay in notes; while `wm.solve()` is working, `plan` is just "
    "`'wm.solve'`. Save code only when it changed; write notes every call. Record the wm outcome as returned "
    "(`'wm': {'stage': ..., 'check_acc': ...}`), do not paraphrase it.",
]


def _py_str(text):
    return '"' + text.replace("\\", "\\\\").replace('"', '\\"').replace("\n", "\\n") + '"'


def prompt_block():
    return "".join("    " + _py_str(line + "\n") + "\n" for line in PROMPT_LINES)


def _patch(text, edits, what):
    for anchor, repl in edits:
        n = text.count(anchor)
        if n != 1:
            raise SystemExit(f"{what}: anchor matched {n} times (need exactly 1): {anchor.strip()[:70]!r}")
        text = text.replace(anchor, repl)
    return text


def patch_wm_source(src):
    """WM_SOURCE (the string) -> round-2 WM_SOURCE. Idempotent."""
    if MARK_SRC in src:
        return src
    out = _patch(src, WM_EDITS, "WM_SOURCE")
    compile(out, "<wm>", "exec")
    return out


def patch_prompts(text):
    """prompts.py source -> with the round-2 bullets right before the
    `print(...)` bullet, i.e. directly after the round-1 wm block. Idempotent."""
    if MARK_PROMPT in text:
        return text
    out = _patch(text, [(PROMPT_ANCHOR, prompt_block() + PROMPT_ANCHOR)], "prompts.py")
    compile(out, "prompts.py", "exec")
    return out


def apply(root, dry_run=False):
    agent = os.path.join(root, "inference", "agent")
    ws_path, pr_path = os.path.join(agent, "wm_source.py"), os.path.join(agent, "prompts.py")
    if not os.path.exists(ws_path):
        raise SystemExit("wm_source.py not found -- run apply_wm_patch.py first")
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


# ------------------------------------------------------------------- selftest
S = 4            # toy boards: 8x8 tiles rendered 4x4 -> 32x32, so a tile move changes 32 cells (HUD ticks change 1)


class Board:
    """Tile game rendered at scale S with a 1-cell-per-action HUD strip in raw row 0."""

    def __init__(self):
        self.level, self.steps = 1, 0
        self.reset()

    def tiles(self):
        raise NotImplementedError

    def grid(self):
        g = [[0] * (8 * S) for _ in range(8 * S)]
        for (r, c), v in self.tiles().items():
            for y in range(r * S, r * S + S):
                for x in range(c * S, c * S + S):
                    g[y][x] = v
        for c in range(self.steps % (8 * S)):
            g[0][c] = 9
        return g


RING = {(r, c) for r in range(1, 8) for c in range(8) if r in (1, 7) or c in (0, 7)}
DELTA = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}


class MoveGame(Board):
    """Box pushing. sticky=k: every k-th press is silently ignored (hidden state)."""
    valid = ["UP", "DOWN", "LEFT", "RIGHT", "RESET"]

    def __init__(self, sticky=0):
        self.sticky = sticky
        super().__init__()

    def reset(self):
        self.p, self.box, self.goal = (6, 1), (4, 3), (2, 5)

    def tiles(self):
        t = {w: 5 for w in RING}
        t[self.goal], t[self.box], t[self.p] = 3, 2, 1
        return t

    def act(self, a):
        self.steps += 1
        if self.sticky and self.steps % self.sticky == 0:
            return False
        d = DELTA[a["action"]]
        n = (self.p[0] + d[0], self.p[1] + d[1])
        if n in RING:
            return False
        if n == self.box:
            nb = (n[0] + d[0], n[1] + d[1])
            if nb in RING:
                return False
            self.box = nb
        self.p = n
        if self.box == self.goal:
            self.level += 1
            self.reset()
            return True
        return False


class ClickGame(Board):
    """3x3 lights; a click toggles the light and its orthogonal neighbours."""
    valid = ["MOUSE", "RESET"]
    POS = [(r, c) for r in (2, 4, 6) for c in (1, 3, 5)]

    def reset(self):
        self.on = {self.POS[0], self.POS[1], self.POS[3]}          # one click on the corner solves... nothing: needs search

    def tiles(self):
        return {p: (4 if p in self.on else 2) for p in self.POS}

    def act(self, a):
        self.steps += 1
        t = (a["row"] // S, a["col"] // S)
        if t not in self.POS:
            return False
        for q in [t] + [(t[0] + dr, t[1] + dc) for dr, dc in ((2, 0), (-2, 0), (0, 2), (0, -2))]:
            if q in self.POS:
                self.on ^= {q}
        if len(self.on) == len(self.POS):
            self.level += 1
            self.reset()
            return True
        return False


MOVE_RULES = '''
S = 4
def parse(g):
    o = wm.objects(g)
    t = lambda col: tuple(sorted((x[1] // S, x[2] // S) for x in o if x[0] == col))
    return (t(1)[0], t(2), t(3))
WALL = frozenset((r // S, c // S) for r, c in wm.cells(current_frame.grid, 5))
D = {"UP": (-1, 0), "DOWN": (1, 0), "LEFT": (0, -1), "RIGHT": (0, 1)}
def step(s, a):
    p, boxes, targets = s
    d = D[a]
    n = (p[0] + d[0], p[1] + d[1])
    if n in WALL: return s
    if n in boxes:
        nb = (n[0] + d[0], n[1] + d[1])
        if nb in WALL or nb in boxes: return s
        boxes = tuple(sorted(nb if b == n else b for b in boxes))
        targets = tuple(t for t in targets if t != nb)
    return (n, boxes, targets)
def is_goal(s):
    return not s[2]
'''
MOVE_RULES_WRONG = MOVE_RULES.replace("n = (p[0] + d[0], p[1] + d[1])", "n = (p[0] + 2 * d[0], p[1] + 2 * d[1])")

CLICK_RULES = '''
S = 4
def parse(g):
    return tuple(sorted((x[1], x[2], x[0]) for x in wm.objects(g)))  # (top, left, color) per light, sorted
def actions(s):
    return [("MOUSE", t + S // 2, l + S // 2) for t, l, _ in s]
def step(s, a):
    r, c = a[1] // S * S, a[2] // S * S
    hit = {(r, c), (r + 2 * S, c), (r - 2 * S, c), (r, c + 2 * S), (r, c - 2 * S)}
    return tuple(sorted((t, l, (6 - col) if (t, l) in hit else col) for t, l, col in s))
def is_goal(s):
    return all(col == 4 for _, _, col in s)
'''


def _harness(sb, game):
    history = [{"action": "", "frame": None}]
    budget = [1000.0]

    def frame():
        return {"ascii": "", "step": game.steps, "level": game.level, "shape": [8 * S, 8 * S], "grid": game.grid()}

    history[0]["frame"] = frame()

    def state(last=None):
        return {"current_frame": frame(), "history": list(history), "valid_actions": list(game.valid),
                "last_action_result": last or {}}

    def handler(actions):
        res = {}
        for a in actions:
            lv = game.act(a)
            budget[0] -= 0.5
            disp = "MOUSE(row=%d, col=%d)" % (a["row"], a["col"]) if a["action"] == "MOUSE" else a["action"]
            history.append({"action": disp, "frame": frame()})
            res = {"executed": True, "level_completed": lv, "time_remaining_seconds": budget[0]}
        return {"action_result": res, "state": state(res)}

    def call(code, rules=""):
        # bundle_fast's FrameView has no public .grid yet (Devin's bundle exposes it); same data via wm.grid_of
        src = (rules + "\n" + code).replace("current_frame.grid", "wm.grid_of(current_frame)")
        r = sb.run_sandboxed_python(code=src, timeout_seconds=30,
                                    initial_state=state(), action_handler=handler)
        assert not r["error"], r["error"]
        return r["result"]
    return call


def selftest():
    import shutil
    import tempfile
    import types
    sys.path.insert(0, HERE)
    import apply_wm_patch
    import wm_ref
    root = os.path.join(os.path.dirname(os.path.dirname(HERE)), "bundle_fast", "src", "ARC3-Inference")
    with tempfile.TemporaryDirectory() as tmp:
        dst = os.path.join(tmp, "ARC3-Inference")
        os.makedirs(os.path.join(dst, "inference"))
        shutil.copytree(os.path.join(root, "inference", "agent"), os.path.join(dst, "inference", "agent"))
        assert apply_wm_patch.apply(dst) == "patched"
        assert apply(dst, dry_run=True).startswith("dry run ok")
        assert apply(dst) == "patched" and apply(dst) == "already patched"
        assert patch_wm_source(patch_wm_source(wm_ref.WM_SOURCE)) == patch_wm_source(wm_ref.WM_SOURCE)
        sys.path.insert(0, root)
        pkg = types.ModuleType("inference.agent")
        pkg.__path__ = [os.path.join(dst, "inference", "agent")]
        import inference  # noqa: F401
        sys.modules["inference.agent"] = pkg
        from inference.agent import prompts, python_tool_sandbox as sb, wm_source
        assert wm_source.WM_SOURCE == patch_wm_source(wm_ref.WM_SOURCE)
        pa = prompts.PYTHON_ADDENDUM
        assert pa.index("WORLD MODEL TOOLKIT") < pa.index(MARK_PROMPT) < pa.index("WHEN TO STOP MODELLING") \
            < pa.index("NOTES vs SAVED CODE") < pa.index("- Use `print(...)`")

        # --- MOVE family: template rules, HUD auto-masked, crude parse first, then solve
        g = MoveGame()
        call = _harness(sb, g)
        r = call("action(['RIGHT']); action(['UP']); action(['LEFT']); action(['DOWN'])\n"
                 "result = {'hud': sorted(wm.hud()), 'crude': wm.check(), 'objs': len(wm.objects(current_frame.grid))}",
                 "def parse(g): return wm.clean(g)\ndef step(s, a): return s")
        assert r["hud"] == [[0, c] for c in range(4)], r["hud"]               # exactly the 4 ticked HUD cells
        assert r["crude"]["n"] == 4 and r["crude"]["acc"] == 0.0 and "warning" not in r["crude"]
        assert r["objs"] == 4                                                  # wall ring, player, box, target
        r = call("result = {'health': wm.health(), 'solve': wm.solve()}", MOVE_RULES)
        assert r["health"]["verdict"] == "use" and r["health"]["conflicts"] == 0, r
        assert r["solve"]["stage"] == "execute" and r["solve"]["execute"]["level_completed"], r
        move_plan = r["solve"]["plan_len"]
        r = call("result = [wm.check()['n'], wm.check(level=None)['n'], wm.check(level=None).get('goal'), wm.health()['verdict']]",
                 MOVE_RULES)
        assert r[0] == 0 and r[1] == 4 + move_plan - 1 and r[2] == "1/1 level-ups predicted" and r[3] == "probe", r

        # --- fix zone, then drop on persistently low accuracy, then self-heal when the rule is corrected
        g = MoveGame()
        call = _harness(sb, g)
        seq = ["RIGHT", "LEFT"] * 3
        r = call("action(%r)\nresult = wm.solve()" % seq, MOVE_RULES_WRONG)
        assert r["stage"] == "check" and g.steps == 6, r                       # fixable: refuses to act, no fallback yet
        r = call("action(%r)\nresult = wm.solve()" % (["RIGHT", "LEFT"] * 7), MOVE_RULES_WRONG)
        assert r["stage"] == "fallback" and "accuracy" in r["health"]["why"] and g.steps == 20, r
        r = call("result = [wm.health()['verdict'], wm.solve()['stage']]", MOVE_RULES)
        assert r == ["use", "execute"] and g.level == 2, r

        # --- hidden state: every 3rd press ignored -> same state+action, different outcomes -> drop, nothing executed
        g = MoveGame(sticky=3)
        call = _harness(sb, g)
        r = call("action(%r)\nresult = {'solve': wm.solve(), 'health': wm.health()}" % (["RIGHT", "LEFT"] * 3), MOVE_RULES)
        assert r["solve"]["stage"] == "fallback" and r["health"]["conflicts"] >= 2 and g.steps == 6, r
        hidden_why = r["health"]["why"]

        # --- blind parse: state ignores the thing that moves -> 'fix' with the blind hint, not 'use'
        g = MoveGame()
        call = _harness(sb, g)
        r = call("action(['RIGHT', 'UP', 'LEFT', 'DOWN'])\nresult = wm.health()",
                 "def parse(g): return len(wm.objects(g))\ndef step(s, a): return s\ndef is_goal(s): return False")
        assert r["verdict"] == "fix" and r["blind"] == 4 and "does not see" in r["why"], r

        # --- CLICK family: actions() from shapes, MOUSE tuples -> dicts, level solved by search
        g = ClickGame()
        call = _harness(sb, g)
        r = call("action([{'action': 'MOUSE', 'row': 10, 'col': 6}])\nresult = wm.check()", CLICK_RULES)
        assert r["acc"] == 1.0 and r["by_action"] == {"MOUSE": "1/1"}, r
        r = call("p = wm.plan()\nresult = {'plan': p['plan'], 'solve': wm.solve()}", CLICK_RULES)
        assert r["solve"]["stage"] == "execute" and r["solve"]["execute"]["level_completed"] and g.level == 2, r
        click_plan = r["plan"]

    words = sum(len(l.split()) for l in PROMPT_LINES)
    print("selftest ok (real sandbox subprocess, round-1 + round-2 patches applied to a copy of bundle_fast):")
    print("  MOVE   template: crude wm.clean parse checks, then %d-action plan clears the level; HUD cells auto-masked" % move_plan)
    print("  CLICK  template: plan %s clears the level" % click_plan)
    print("  fallback: wrong rule -> 'check' at 6 actions, 'fallback' at 20, back to 'use' once fixed;")
    print("            hidden state -> 'fallback' after 6 actions (%s...)" % hidden_why[:58])
    print("  prompt: %d words in %d bullets, inserted after the round-1 wm block" % (words, len(PROMPT_LINES)))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        selftest()
    elif len(sys.argv) >= 2 and sys.argv[1] == "--print-prompt":
        print("".join(PROMPT_LINES[i] + "\n\n" for i in range(len(PROMPT_LINES))).rstrip())
    else:
        ap = argparse.ArgumentParser()
        ap.add_argument("root")
        ap.add_argument("--dry-run", action="store_true")
        a = ap.parse_args()
        print(apply(a.root, a.dry_run))
