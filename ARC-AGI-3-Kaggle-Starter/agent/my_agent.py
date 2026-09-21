"""OCEAN: Object-Centric Exploration and Navigation agent for ARC-AGI-3.

A training-free, online world-model-induction agent. Per step it:
  * diffs consecutive frames to find which objects moved in response to the
    last action (robust in fields of identical tiles),
  * probes the four directional actions to identify the controllable object
    (the "avatar") by canonical-direction agreement, grouping co-moving sprite
    colors,
  * learns per-action motion (discrete step vs. slide-until-blocked), wall /
    hazard colors from failed or fatal moves, and the goal color from the
    completing action's destination,
  * plans with BFS over lattice positions toward hypothesized goal objects,
    falling back to systematic frontier exploration and object clicking.
  * GAME_OVER -> RESET; knowledge (walls, goal/click colors) persists across
    level-reset retries.
"""
from __future__ import annotations

import random
from collections import Counter, defaultdict, deque

import numpy as np

A_RESET = 0
SIMPLE = [1, 2, 3, 4, 5, 7]
CLICK = 6


def diff_mask(prev: np.ndarray, cur: np.ndarray) -> np.ndarray:
    return prev != cur


def color_motion(prev, cur, mask):
    """For each color: cells erased (prev==c & diff) and drawn (cur==c & diff).
    Returns {color: (dx, dy, n)} using centroid difference when both exist."""
    out = {}
    if not mask.any():
        return out
    ys, xs = np.where(mask)
    for c in np.unique(np.concatenate([prev[ys, xs], cur[ys, xs]])):
        old = mask & (prev == c)
        new = mask & (cur == c)
        no, nn = int(old.sum()), int(new.sum())
        if no and nn:
            oys, oxs = np.where(old)
            nys, nxs = np.where(new)
            out[int(c)] = (float(nxs.mean() - oxs.mean()),
                           float(nys.mean() - oys.mean()), min(no, nn))
        else:
            out[int(c)] = (0.0, 0.0, max(no, nn))
    return out


def components_of_color(grid, color, max_keep=200):
    """Connected components (4-conn) of a single color -> list of (cells_set, centroid)."""
    mask = grid == color
    h, w = grid.shape
    seen = np.zeros_like(mask, dtype=bool)
    comps = []
    ys, xs = np.where(mask & ~seen)
    for y0, x0 in zip(ys, xs):
        if seen[y0, x0] or not mask[y0, x0]:
            continue
        cells = []
        stack = [(int(x0), int(y0))]
        seen[y0, x0] = True
        while stack:
            x, y = stack.pop()
            cells.append((x, y))
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nx, ny = x + dx, y + dy
                if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((nx, ny))
        comps.append((cells, (np.mean([p[0] for p in cells]),
                              np.mean([p[1] for p in cells]))))
        if len(comps) >= max_keep:
            break
    return comps


def all_objects(grid):
    """Connected components per non-dominant color -> list of dicts."""
    h, w = grid.shape
    vals, counts = np.unique(grid, return_counts=True)
    bg = {int(v) for v, c in zip(vals, counts) if c > 0.45 * h * w}
    objs = []
    for v in vals:
        c = int(v)
        if c in bg:
            continue
        for cells, cent in components_of_color(grid, c):
            xs = [p[0] for p in cells]; ys = [p[1] for p in cells]
            objs.append({"color": c, "cells": cells, "centroid": cent,
                         "bbox": (min(xs), min(ys), max(xs), max(ys)),
                         "size": len(cells)})
    return objs


class OceanCore:
    def __init__(self, seed: int = 0):
        self.rng = random.Random(seed)
        # persistent across levels of the same game
        self.avatar_colors = set()
        self.dir = {}                  # action -> unit direction (sx,sy)
        self.mag = {}                  # action -> fixed step px (discrete moves)
        self.slide = {}                # action -> True if variable-distance move
        self.obs = defaultdict(list)   # action -> observed (dx,dy) displacements
        self.move_actions = []
        self.blocking_colors = set()   # colors that block avatar motion
        self.goal_color = None         # learned after first level completion
        self.click_color = None        # color of successfully-clicked objects
        self.hazard_count = defaultdict(int)  # color -> deaths at destination
        self._goal_locs = []           # positions where past levels completed
        self._collect_colors = set()   # colors of objects consumed pre-completion
        self._stood_colors = set()     # colors the avatar has safely stood on
        self.quantum = 1               # lattice step estimate (px)
        self.level_init()

    def level_init(self):
        self.grid = None
        self.phase = "probe"
        self.probe_queue = deque()
        self.probe_prev = None
        self.probe_action = None
        self.probe_results = []        # list of (action, color_motion dict)
        self.avatar_pos = None         # centroid (x,y) float
        self.avatar_cells = set()
        self.visited = set()           # avatar lattice positions visited
        self.blocked = set()           # lattice positions learned blocked
        self.bad_targets = set()
        self._target_obj = {}
        self.plan = deque()
        self._frozen = 0               # steps with tracked object unmoved
        self._stuck = 0                # consecutive failed moves at one cell
        self._traps = set()            # cells that immobilize the avatar
        self._per_target = None        # centroid of pending perimeter approach
        self._objs_prev = None         # (color,x,y) objects seen last frame
        self._consumed = deque(maxlen=400)  # (color, level_actions) vanished
        self._safe_colors = set()      # colors the avatar has stood beside
        self._last = None              # (action, pos_before) for verification
        self.clicked = set()
        self._eff_obj = deque()        # effective objects: {"key", "cells", "presses"}
        self._eff_obj_keys = set()
        self._eff_suppress = set()     # eff objects implicated in a death
        self._eff_presses = Counter()  # cumulative dwell presses per key
        self._eff_turn = False         # alternate eff-dwelling with sweeping
        self._press_count = defaultdict(int)  # clicks issued per cell
        self._sweep = None             # ordered click-sweep positions
        self._click_burst = 0          # forced clicks while nav is stalled
        self.stuck = 0
        self.level_actions = 0
        self.cur_level = -1
        self.no_progress_steps = 0
        self._wcells = set()
        self.noop_streak = 0         # consecutive frames with zero change
        self.unlock_left = 0         # steps spent trying selection/clicks
        self._hist = deque(maxlen=8)  # recent diff masks
        self._ticking = None          # cells that change every frame (UI)

    # ================= main =================
    def decide(self, grid, available, state, levels_completed, step):
        if state in ("NOT_PLAYED", "GAME_OVER"):
            if state == "GAME_OVER":
                self._record_death()
            return (A_RESET, {})

        grid = np.asarray(grid)
        if self.grid is None:
            self.grid = grid
            self.cur_level = levels_completed
            self._build_probe(available)
            return self._consume_probe(grid, available)

        if levels_completed != self.cur_level:
            self._on_level_change(levels_completed, grid, available)
            return self._consume_probe(grid, available) if self.phase == "probe" \
                else self._act(grid, available)

        return self._act(grid, available) if self.phase != "probe" \
            else self._consume_probe(grid, available)

    def _on_level_change(self, lvl, grid, available):
        # The completing action's predicted destination was the goal —
        # learn its color so future levels can target it directly.
        if self._last:
            a, before = self._last[0], self._last[1]
            data = self._last[2] if len(self._last) > 2 else {}
            if a in self.dir and before is not None and self.avatar_colors:
                # the goal is the cell the avatar was entering — use its
                # actual final position + one step forward (predicted dest
                # overshoots on slides and misses the exit cell entirely)
                if self.avatar_pos is not None:
                    dest = (self.avatar_pos[0] + self.dir[a][0] * self.quantum,
                            self.avatar_pos[1] + self.dir[a][1] * self.quantum)
                else:
                    dest = self._predict_dest(before, a)
                gc = self._color_at(dest)
                if gc is not None and gc not in self.avatar_colors:
                    self.goal_color = gc
            elif a == CLICK and data.get("x") is not None:
                gc = self._color_at((data["x"], data["y"]))
                if gc is not None:
                    self.click_color = gc
        # remember where this level was completed — exits tend to sit in
        # consistent regions (e.g. the same edge), which biases later levels'
        # frontier choice even when no goal color is learnable
        if self.avatar_pos is not None:
            self._goal_locs.append(self._lattice(self.avatar_pos))
        # objects consumed shortly before completion are goal-relevant
        # (collect-them-all semantics) — prioritize their colors next level
        cnt = Counter(c for c, t in self._consumed
                      if self.level_actions - t < 300)
        for c, n in cnt.items():
            if n <= 10:
                # real pickups are consumed once; animated decorations
                # flicker in and out constantly — ignore those colors
                self._collect_colors.add(c)
        keep = (dict(self.dir), dict(self.mag), dict(self.slide),
                defaultdict(list, {a: list(v) for a, v in self.obs.items()}))
        keep_avatar = set(self.avatar_colors)
        keep_blocking = set(self.blocking_colors)
        self.level_init()
        self.cur_level = lvl   # after level_init, which resets it to -1
        self.dir, self.mag, self.slide, self.obs = keep
        self.avatar_colors = keep_avatar
        self.blocking_colors = keep_blocking
        self.move_actions = sorted(self.dir)
        self.grid = grid
        self.phase = "act" if self.avatar_colors else "probe"
        if self.phase == "probe":
            self._build_probe(available)
        else:
            self._locate_avatar(grid)

    # ================= probe =================
    def _build_probe(self, available):
        self.probe_queue = deque()
        self.probe_results = []
        # probe directions twice each; never probe ACTION5 (selection
        # switching corrupts tracking) or ACTION7 (undo reverts progress)
        simple = [a for a in available if a in (1, 2, 3, 4)]
        for a in simple + simple:
            self.probe_queue.append(a)
        self._probe_action = None
        self.probe_prev = None
        self.phase = "probe"

    def _consume_probe(self, grid, available):
        if self.probe_action is not None:
            mask = diff_mask(self.probe_prev, grid)
            self.probe_results.append((self.probe_action,
                                       color_motion(self.probe_prev, grid, mask)))
            self.probe_prev = grid
        if self.probe_queue:
            a = self.probe_queue.popleft()
            self.probe_action = a
            self.probe_prev = grid
            return (a, {})
        self.probe_action = None
        self._learn_semantics()
        self.phase = "act"
        self._locate_avatar(grid)
        return self._act(grid, available, first=True)

    # canonical directions per the ARC-AGI-3 docs (A1 up, A2 down, A3 left, A4 right)
    CANON = {1: (0, -1), 2: (0, 1), 3: (-1, 0), 4: (1, 0)}

    def _learn_semantics(self):
        # per color, per action -> list of (dx,dy) observations across passes
        per = defaultdict(lambda: defaultdict(list))
        for a, motion in self.probe_results:
            for c, (dx, dy, n) in motion.items():
                if abs(dx) > 0.5 or abs(dy) > 0.5:
                    per[c][a].append((dx, dy))
        # score each color by agreement with canonical action directions
        best_c, best_score = None, 1
        for c, amap in per.items():
            score = 0
            for a, lst in amap.items():
                if a not in self.CANON:
                    continue
                mdx = sorted(d[0] for d in lst)[len(lst)//2]
                mdy = sorted(d[1] for d in lst)[len(lst)//2]
                cx, cy = self.CANON[a]
                if mdx * cx > 0.5 or mdy * cy > 0.5:
                    score += 1
            if score > best_score:
                best_c, best_score = c, score
        if best_c is not None:
            # median delta per (color, action)
            med = {c: {a: (sorted(d[0] for d in lst)[len(lst)//2],
                           sorted(d[1] for d in lst)[len(lst)//2])
                       for a, lst in amap.items() if lst}
                   for c, amap in per.items()}
            # co-moving colors: same direction (within 2px) on >=1 shared action
            co = {best_c}
            for c in per:
                if c == best_c:
                    continue
                shared = sum(
                    1 for a in med[c]
                    if a in med[best_c]
                    and abs(med[c][a][0] - med[best_c][a][0]) <= 2
                    and abs(med[c][a][1] - med[best_c][a][1]) <= 2)
                if shared >= 1:
                    co.add(c)
            # per action: displacement obs from the co-moving color with most obs
            for a in range(1, 8):
                best_l, best_n = None, -1
                for c in co:
                    if a in per[c] and len(per[c][a]) > best_n:
                        best_l, best_n = per[c][a], len(per[c][a])
                if best_l:
                    self.obs[a].extend(
                        (int(round(x)), int(round(y))) for x, y in best_l)
            self.avatar_colors = co
            self._reclassify()
        else:
            self.avatar_colors = set()
            self.dir, self.mag, self.slide = {}, {}, {}
            self.move_actions = []

    def _reclassify(self):
        """Turn raw displacement observations into dir/mag/slide per action.
        Uses the modal (most common) displacement; ACTION7 is never a move."""
        for a, lst in self.obs.items():
            if not lst or a not in (1, 2, 3, 4):
                continue
            mdx, mdy = Counter(lst).most_common(1)[0][0]
            if abs(mdx) >= abs(mdy):
                self.dir[a] = (1 if mdx > 0 else -1, 0)
            else:
                self.dir[a] = (0, 1 if mdy > 0 else -1)
            mags = [max(abs(x), abs(y)) for x, y in lst]
            mc = Counter(mags)
            mode_mag, mode_n = mc.most_common(1)[0]
            if mode_n * 3 >= len(mags) * 2:
                # one dominant step size -> discrete move
                self.slide[a] = False
                self.mag[a] = max(1, mode_mag)
            else:
                self.slide[a] = True
        allm = [max(abs(x), abs(y)) for lst in self.obs.values() for x, y in lst
                if (x, y) != (0, 0)]
        if allm:
            self.quantum = max(1, Counter(allm).most_common(1)[0][0])
        self.move_actions = sorted(a for a in self.dir if a != 7)

    # ================= perception =================
    def _locate_avatar(self, grid, anchor=None, prev_grid=None, ray=None):
        """Avatar = the same-color object that *moved* most (most cells in the
        changed region vs prev frame), nearest the anchor. This tracks the
        controllable object even if its color or identity changes (selection
        games). Falls back to nearest avatar-colored component. `ray` is a
        (origin, unit_dir) corridor for slide moves — the avatar may land
        anywhere along it."""
        if not self.avatar_colors:
            return None
        ref = anchor if anchor is not None else self.avatar_pos

        def cdist(cx, cy):
            if ray is not None:
                (bx, by), (ex, ey) = ray
                proj = (cx - bx) * ex + (cy - by) * ey
                perp = abs((cx - bx) * ey - (cy - by) * ex)
                return perp + max(0.0, -proj - self.quantum)
            if ref is None:
                return 0
            return abs(cx - ref[0]) + abs(cy - ref[1])
        changed = None
        if prev_grid is not None and prev_grid.shape == grid.shape:
            m = (prev_grid != grid)
            if self._ticking is not None:
                m = m & ~self._ticking
            changed = set(zip(*np.where(m)[::-1]))
        mask = np.isin(grid, list(self.avatar_colors))
        h, w = mask.shape
        seen = np.zeros_like(mask, dtype=bool)
        comps = []
        ys, xs = np.where(mask)
        for y0, x0 in zip(ys, xs):
            if seen[y0, x0]:
                continue
            cells = []
            stack = [(int(x0), int(y0))]
            seen[y0, x0] = True
            while stack:
                x, y = stack.pop()
                cells.append((x, y))
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx, ny = x + dx, y + dy
                    if 0 <= nx < w and 0 <= ny < h and mask[ny, nx] and not seen[ny, nx]:
                        seen[ny, nx] = True
                        stack.append((nx, ny))
            comps.append(cells)
        if not comps:
            return None
        # continuity gate: when a reference (predicted or last-known pos)
        # exists, ignore far-away components — flickering UI elements that
        # share avatar colors otherwise hijack tracking (ar25's right-edge
        # phantom). Fall back to the full pool if nothing is near.
        if ref is not None or ray is not None:
            near = [c for c in comps
                    if cdist(sum(p[0] for p in c) / len(c),
                             sum(p[1] for p in c) / len(c))
                    <= 6 * self.quantum]
            if near:
                comps = near
        # rank components: most cells in the changed region (i.e. the object
        # that moved), then nearest to the reference point
        best, bkey = None, None
        for cells in comps:
            cx = sum(p[0] for p in cells) / len(cells)
            cy = sum(p[1] for p in cells) / len(cells)
            n_new = sum(1 for c in cells if c in changed) if changed else 0
            d = cdist(cx, cy)
            key = (-n_new, d)
            if bkey is None or key < bkey:
                best, bkey = (cells, (cx, cy)), key
        self.avatar_cells, self.avatar_pos = best
        xs2 = [p[0] for p in self.avatar_cells]
        ys2 = [p[1] for p in self.avatar_cells]
        self._av_dims = (max(xs2) - min(xs2) + 1, max(ys2) - min(ys2) + 1)
        return self.avatar_pos

    # ================= act =================
    def _act(self, grid, available, first=False):
        prev_grid = self.grid
        self.grid = grid
        self.level_actions += 1

        diff = None
        if prev_grid is not None and prev_grid.shape == grid.shape:
            diff = (prev_grid != grid)
            self._hist.append(diff)
            if len(self._hist) >= 4:
                # cells that changed in EVERY recent frame are UI (step
                # counters, animations) — exclude from "real" change
                self._ticking = np.logical_and.reduce(list(self._hist))

        # object consumption: objects present last frame but gone now were
        # collected/consumed — their colors identify goal-relevant pickups
        if prev_grid is not None and prev_grid.shape == grid.shape and \
                self.avatar_colors:
            prev_sig = self._objs_prev
            cur = {(o["color"], int(round(o["centroid"][0])),
                    int(round(o["centroid"][1])))
                   for o in all_objects(grid)
                   if o["color"] not in self.avatar_colors}
            if prev_sig is not None:
                for c, x, y in prev_sig:
                    if c in self.avatar_colors:
                        continue
                    if not any(c == c2 and abs(x - x2) <= 3 and
                               abs(y - y2) <= 3 for c2, x2, y2 in cur):
                        self._consumed.append((c, self.level_actions))
                        # a consumed object was touchable, not a wall — a
                        # failed bump may have wrongly learned it blocking
                        self.blocking_colors.discard(c)
            self._objs_prev = cur
        elif self.avatar_colors:
            self._objs_prev = {(o["color"], int(round(o["centroid"][0])),
                                int(round(o["centroid"][1])))
                               for o in all_objects(grid)
                               if o["color"] not in self.avatar_colors}

        if diff is None:
            frame_changed = True
        elif self._ticking is not None:
            frame_changed = bool((diff & ~self._ticking).any())
        else:
            frame_changed = bool(diff.any())
        if frame_changed:
            self.noop_streak = 0
        else:
            self.noop_streak += 1

        # multi-press / toggle click mechanics: a click that produced a
        # *substantial* change (≥8 non-ticking cells — animations and
        # counters are masked out) marks the cell "effective". Effective
        # cells are cycled round-robin, alternating with the sweep, which
        # covers both multi-press (lp85: same cell x5) and balance/toggle
        # (s5i5: alternate two cells) mechanics.
        if self._last and self._last[0] == CLICK:
            d = self._last[2] if len(self._last) > 2 else {}
            pos = (d.get("x"), d.get("y"))
            big_change = diff is not None and \
                int((diff & ~self._ticking).sum() if self._ticking is not None
                    else diff.sum()) >= 8
            if big_change and pos[0] is not None:
                # group by containing object — presses on the same object
                # accumulate (lp85: press a button 5x), while pressing a
                # different object can reset progress
                key, cells = self._obj_key(prev_grid, pos)
                if key is None:
                    key, cells = ("cell", pos), [pos]
                if key not in self._eff_obj_keys and \
                        key not in self._eff_suppress:
                    self._eff_obj_keys.add(key)
                    self._eff_obj.append({"key": key, "cells": cells,
                                          "presses": 0})

        if self.avatar_colors:
            anchor = None
            ray = None
            if self._last and self._last[0] in self.dir \
                    and self._last[1] is not None:
                if self.slide.get(self._last[0]):
                    # a slide can land anywhere along the ray — gate on a
                    # corridor, not the predicted endpoint
                    ray = (self._last[1], self.dir[self._last[0]])
                else:
                    anchor = self._predict_dest(self._last[1],
                                                self._last[0])
            if anchor is None and ray is None:
                anchor = self.avatar_pos  # continuity: avatar rarely jumps
            new_pos = self._locate_avatar(grid, anchor, prev_grid, ray=ray)
            self._verify_move(new_pos)
            if new_pos is None:
                # avatar gone (died?) -> reset handled by state; else re-scan
                self.avatar_pos = None
                self._locate_avatar(grid)
            # frozen-tracker recovery: if the tracked object has barely moved
            # all level, it is a static decoration sharing the palette
            # (level transitions can change the real avatar's colors).
            # Re-derive the palette from whatever actually changed.
            elif self._frozen >= 40 and len(self.visited) <= 3 and self.dir:
                self._reacquire_avatar(grid, diff)
                self._frozen = 0

        # keyboard appears dead (no frame changes at all): the mover may need
        # to be selected first -> try selection actions / clicks briefly
        if self.noop_streak > 20 and self.unlock_left <= 0:
            self.unlock_left = 6
        if self.unlock_left > 0:
            self.unlock_left -= 1
            a, data = self._unlock_action(grid, available)
            self.plan.clear()
            self._last = (a, self.avatar_pos, data)
            return (a, data)

        # nav keeps failing -> the goal may need clicks, not movement
        if self._click_burst > 0:
            self._click_burst -= 1
            a, data = self._click_or_probe(grid, available)
            self._last = (a, self.avatar_pos, data)
            return (a, data)
        if self.no_progress_steps >= 15 and CLICK in available \
                and not self.dir:
            self._click_burst = 8
            self.no_progress_steps = 0

        if self.plan:
            a, data = self._pop_plan_raw()
        elif self.move_actions:
            a, data = self._navigate(grid, available)
        else:
            a, data = self._click_or_probe(grid, available)
        self._last = (a, self.avatar_pos, data)
        return (a, data)

    def _reacquire_avatar(self, grid, diff):
        """The tracked "avatar" is a static decoration — adopt the colors of
        whatever actually changed in the last frame instead."""
        if diff is None:
            return
        mask = diff & ~self._ticking if self._ticking is not None else diff
        if not mask.any() or int(mask.sum()) < 3:
            return
        vals, cnts = np.unique(grid, return_counts=True)
        bg = {int(v) for v, c in zip(vals, cnts) if c > 0.45 * grid.size}
        ys, xs = np.where(mask)
        cols = Counter(int(grid[y, x]) for y, x in zip(ys, xs)
                       if int(grid[y, x]) not in bg)
        new = {c for c, _ in cols.most_common(4)}
        if new and new != self.avatar_colors:
            self.avatar_colors = new
            self.avatar_pos = None
            self.avatar_cells = set()
            self.plan.clear()

    def _unlock_action(self, grid, available):
        """When keys do nothing: cycle selection (ACTION5), undo (ACTION7),
        then click objects to activate a mover."""
        k = self.unlock_left
        if k == 5 and 5 in available:
            return (5, {})
        if k == 4 and 7 in available:
            return (7, {})
        return self._click_or_probe(grid, available)

    def _verify_move(self, new_pos):
        """Check result of last nav action; update blocked/visited/obs."""
        la = self._last
        self._last = None
        if la is None:
            if new_pos is not None:
                self.visited.add(self._lattice(new_pos))
            return
        a, before = la[0], la[1]
        if new_pos is None:
            return
        new_pos = tuple(new_pos)
        if before is not None and a in self.dir:
            if self._lattice(new_pos) == self._lattice(before):
                self._frozen += 1
            else:
                self._frozen = 0
        if before is not None and new_pos != before:
            dx = int(round(new_pos[0] - before[0]))
            dy = int(round(new_pos[1] - before[1]))
            if (dx, dy) != (0, 0) and a in (1, 2, 3, 4):
                # only trust observations along the action's known axis —
                # perpendicular jumps mean the tracked object switched
                if a not in self.dir or \
                        dx * self.dir[a][0] + dy * self.dir[a][1] > 0:
                    self.obs[a].append((dx, dy))
                    self._reclassify()
            self.avatar_pos = new_pos
            self.visited.add(self._lattice(new_pos))
            # self-heal: a cell we actually reached cannot be blocked —
            # failed-move marks record the wall *edge*, not the node
            if self._lattice(new_pos) not in self._traps:
                self.blocked.discard(self._lattice(new_pos))
            sc = self._color_at(new_pos)
            if sc is not None:
                self._safe_colors.add(sc)
            # colors directly under the footprint are provably safe to
            # *stand on* — standing beside a lethal color is safe too,
            # but only stood-on colors may gate under-foot blame
            w2, h2 = getattr(self, "_av_dims", (1, 1))
            uc = self._color_at(new_pos, r=max(1, min(w2, h2) // 2 + 1))
            if uc is not None:
                self._stood_colors.add(uc)
            if before is not None and \
                    self._lattice(new_pos) != self._lattice(before):
                self._stuck = 0
        if a not in self.dir or before is None:
            return
        exp = self._predict_dest(before, a)
        if exp and self._lattice(new_pos) != self._lattice(exp):
            ddx = new_pos[0] - before[0]
            ddy = new_pos[1] - before[1]
            ex, ey = self.dir[a]
            dot = ddx * ex + ddy * ey          # forward component
            perp = abs(ddx * ey) + abs(ddy * ex)  # sideways component
            if perp <= self.quantum:
                # stopped short along the expected axis (or didn't move):
                # the unenterable cell is one step past where the avatar
                # actually landed — not the predicted slide destination,
                # which can lie far beyond the true wall.
                ec = self._lattice((new_pos[0] + ex * self.quantum,
                                    new_pos[1] + ey * self.quantum))
                self.blocked.add(ec)
                self._learn_blocker(exp, new_pos)
            # else: moved sideways = selection/tracking noise, ignore
        if self._lattice(new_pos) == self._lattice(before):
            self._stuck += 1
            if self._stuck > 40:
                # immobilized: this cell is a trap — exclude permanently
                self._traps.add(self._lattice(new_pos))
                self.blocked.add(self._lattice(new_pos))
        else:
            self._stuck = 0

    def _predict_dest(self, pos, a):
        """Predicted avatar position after action a from pos."""
        if pos is None or a not in self.dir:
            return pos
        dx, dy = self.dir[a]
        if not self.slide.get(a):
            m = self.mag.get(a, self.quantum)
            return (pos[0] + dx * m, pos[1] + dy * m)
        p = (int(round(pos[0])), int(round(pos[1])))
        for _ in range(64):
            np_ = (p[0] + dx * self.quantum, p[1] + dy * self.quantum)
            if not (0 <= np_[0] < 64 and 0 <= np_[1] < 64):
                break
            if self._edge_blocked(p, np_):
                break
            p = np_
        return p

    def _color_at(self, pos, r=3):
        """Most common non-avatar, non-background color near pos."""
        if self.grid is None or pos is None:
            return None
        h, w = self.grid.shape
        x, y = int(round(pos[0])), int(round(pos[1]))
        vals, counts = np.unique(self.grid, return_counts=True)
        bg = {int(v) for v, c in zip(vals, counts) if c > 0.45 * self.grid.size}
        cnt = {}
        for yy in range(max(0, y - r), min(h, y + r + 1)):
            for xx in range(max(0, x - r), min(w, x + r + 1)):
                c = int(self.grid[yy, xx])
                if c not in self.avatar_colors and c not in bg:
                    cnt[c] = cnt.get(c, 0) + 1
        return max(cnt, key=cnt.get) if cnt else None

    def _record_death(self):
        """The action that ended the game: the color at its destination is a
        hazard candidate. Repeated deaths -> treat the color as blocking and
        never click it again."""
        la = self._last
        self._last = None
        if not la:
            return
        a, before = la[0], la[1]
        data = la[2] if len(la) > 2 else {}
        if a == CLICK and data.get("x") is not None:
            c = self._color_at((data["x"], data["y"]))
            if c is not None:
                self.hazard_count[c] += 1
            # click-order puzzles: after a fatal click, allow re-clicking the
            # surviving objects in a different order
            self.clicked.clear()
            return
        if not self.avatar_colors or a not in self.dir or before is None:
            return
        # contact kill: the avatar's last tracked position is the cell it
        # died on — blame the color under it if it never stood there safely.
        # This catches hazards that were merely *adjacent* before (hence in
        # _safe_colors) but kill on entry; the predicted destination can lie
        # far past the real kill cell for slide moves.
        pos = self.avatar_pos
        if pos is not None and self._lattice(pos) != self._lattice(before):
            w2, h2 = getattr(self, "_av_dims", (1, 1))
            uc = self._color_at(pos, r=max(1, min(w2, h2) // 2 + 1))
            if uc is not None and uc not in self._stood_colors:
                self.hazard_count[uc] += 1
                if self.hazard_count[uc] >= 2:
                    self.blocking_colors.add(uc)
                return
        dest = self._predict_dest(before, a)
        c = self._color_at(dest)
        if c is None or c in self.avatar_colors or c in self._safe_colors:
            return
        self.hazard_count[c] += 1
        if self.hazard_count[c] >= 2:
            self.blocking_colors.add(c)

    def _lattice(self, pos):
        return (int(round(pos[0])), int(round(pos[1])))

    def _learn_blocker(self, exp, before):
        """Mark colors in the *gap* between the current footprint and the
        expected footprint — that's where an obstacle lies. Never marks the
        dominant background color."""
        ex, ey = int(round(exp[0])), int(round(exp[1]))
        bx, by = int(round(before[0])), int(round(before[1]))
        vals, counts = np.unique(self.grid, return_counts=True)
        bg = {int(v) for v, c in zip(vals, counts) if c > 0.45 * self.grid.size}
        w, h = getattr(self, "_av_dims", (1, 1))
        dx, dy = ex - bx, ey - by
        # gap rect strictly between the two footprints along movement axis
        cells = []
        if abs(dx) >= abs(dy):
            s = 1 if dx > 0 else -1
            x0, x1 = bx + s * (w // 2 + 1), ex - s * (w // 2 + 1)
            lo, hi = min(x0, x1), max(x0, x1)
            ylo = min(bx * 0 + by, ey) - h // 2
            yhi = max(by, ey) + h // 2
            for x in range(lo, hi + 1):
                for y in range(ylo, yhi + 1):
                    cells.append((x, y))
        else:
            s = 1 if dy > 0 else -1
            y0, y1 = by + s * (h // 2 + 1), ey - s * (h // 2 + 1)
            lo, hi = min(y0, y1), max(y0, y1)
            xlo = min(bx, ex) - w // 2
            xhi = max(bx, ex) + w // 2
            for y in range(lo, hi + 1):
                for x in range(xlo, xhi + 1):
                    cells.append((x, y))
        if not cells:
            # footprints overlap/adjacent: scan the expected footprint minus old
            cells = [(ex + ddx, ey + ddy)
                     for ddy in range(-(h // 2), h - h // 2)
                     for ddx in range(-(w // 2), w - w // 2)]
        for x2, y2 in cells:
            if 0 <= x2 < self.grid.shape[1] and 0 <= y2 < self.grid.shape[0]:
                if self._ticking is not None and self._ticking[y2, x2]:
                    continue   # animated cells (gauges, counters) aren't walls
                c = int(self.grid[y2, x2])
                if c not in self.avatar_colors and c not in bg:
                    self.blocking_colors.add(c)

    def _pop_plan(self):
        a, data = self._pop_plan_raw()
        return (a, data)

    def _pop_plan_raw(self):
        return self.plan.popleft()

    # ---------- navigation ----------
    def _navigate(self, grid, available):
        if self.avatar_pos is None:
            self._locate_avatar(grid)
            if self.avatar_pos is None:
                return self._random(available)
        start = self._lattice(self.avatar_pos)
        moves = [a for a in self.move_actions if a in available]
        if not moves:
            return self._random(available)

        # precompute wall cells from blocking colors (cheap: direct mask)
        wcells = set()
        if self.blocking_colors:
            ys, xs = np.where(np.isin(grid, list(self.blocking_colors)))
            wcells = set(zip(xs.tolist(), ys.tolist()))
        self._wcells = wcells

        # reached a solid object's perimeter: poke it once (contact may
        # consume it), then never chase it again
        pt = self._per_target
        self._per_target = None
        if pt is not None and self.avatar_pos is not None and \
                abs(self.avatar_pos[0] - pt[0]) <= self.quantum + 2 and \
                abs(self.avatar_pos[1] - pt[1]) <= self.quantum + 2:
            self.bad_targets.add(pt)
            dx = pt[0] - self.avatar_pos[0]
            dy = pt[1] - self.avatar_pos[1]
            for a in moves:
                ux, uy = self.dir.get(a, (0, 0))
                if (abs(dx) >= abs(dy) and dx * ux > 0) or \
                        (abs(dy) > abs(dx) and dy * uy > 0):
                    return (a, {})

        for _ in range(10):
            targets = self._goal_cells(grid)
            if not targets:
                break
            path = self._bfs(start, moves, targets)
            if path is None:
                # solid targets (keys, exits) cannot be *entered* — only
                # touched. Aim at cells adjacent to each target instead.
                per = {}
                q2 = self.quantum + 1
                for t in targets:
                    for dx, dy in ((q2, 0), (-q2, 0), (0, q2), (0, -q2)):
                        p2 = self._lattice((t[0] + dx, t[1] + dy))
                        if p2 not in self.blocked:
                            per[p2] = t
                if per:
                    path = self._bfs(start, moves, set(per))
                if path:
                    dest = start
                    for a2 in path:
                        dest = self._lattice(self._predict_dest(dest, a2))
                    self._per_target = per.get(dest)
                    for a in path:
                        self.plan.append((a, {}))
                    return self._pop_plan()
                break
            if path:
                for a in path:
                    self.plan.append((a, {}))
                return self._pop_plan()
            near = min(targets, key=lambda t: abs(t[0]-start[0])+abs(t[1]-start[1]))
            self.bad_targets.add(near)
        # frontier exploration — biased toward candidate-goal objects so the
        # agent explores *toward* likely exits instead of uniformly outward
        path = self._bfs_toward(start, moves, grid)
        if path:
            for a in path:
                self.plan.append((a, {}))
            return self._pop_plan()
        self.no_progress_steps += 1
        return self._random(available)

    def _goal_cells(self, grid):
        """Target cells: centroids of goal-colored objects, else all small
        non-avatar, non-wall objects. Colors with many instances are
        structural (maze lattices, decorations) — never goals."""
        out = set()
        objs = all_objects(grid)
        color_count = Counter(o["color"] for o in objs)
        for o in objs:
            if self.goal_color is not None:
                if o["color"] != self.goal_color:
                    continue
            else:
                if o["color"] in self.avatar_colors or \
                        o["color"] in self.blocking_colors:
                    continue
                if o["size"] > 0.2 * 64 * 64:
                    continue
                if color_count[o["color"]] > 8:
                    continue
                if self._collect_colors and \
                        o["color"] not in self._collect_colors:
                    continue
            c = (int(round(o["centroid"][0])), int(round(o["centroid"][1])))
            if c in self.bad_targets:
                continue
            out.add(c)
        return out

    def _bfs_toward(self, start, moves, grid):
        """Frontier BFS: among all reachable unvisited cells, return the path
        to the one closest to a candidate-goal object (or nearest overall if
        there are none)."""
        seen = {start}
        q = deque([(start, [])])
        found = []          # (pos, path) of reachable unvisited cells
        n = 0
        while q and n < 30000:
            pos, path = q.popleft()
            n += 1
            if pos not in self.visited and pos != start:
                found.append((pos, path))
            for a in moves:
                np_ = self._lattice(self._predict_dest(pos, a))
                if np_ == pos or np_ in seen or np_ in self.blocked:
                    continue
                if not (0 <= np_[0] < 64 and 0 <= np_[1] < 64):
                    continue
                if self._edge_blocked(pos, np_):
                    continue
                seen.add(np_)
                q.append((np_, path + [a]))
        if not found:
            return None
        # reference points: candidate goal objects (even blacklisted ones —
        # they are still informative landmarks)
        refs = []
        for p in self._goal_locs:
            # a past exit location only biases this level if it sits on
            # traversable ground in *this* layout
            x, y = int(p[0]), int(p[1])
            if 0 <= x < grid.shape[1] and 0 <= y < grid.shape[0] and \
                    int(grid[y, x]) not in self.blocking_colors:
                refs.append(p)
        color_count = Counter(o["color"] for o in all_objects(grid))
        for o in all_objects(grid):
            if o["color"] in self.avatar_colors:
                continue
            if o["size"] > 0.2 * 64 * 64 or color_count[o["color"]] > 8:
                continue
            # small objects of learned-blocking colors are still useful
            # landmarks — they tend to be touchable pickups/exits, not walls
            if o["color"] in self.blocking_colors and o["size"] > 64:
                continue
            refs.append((o["centroid"][0], o["centroid"][1]))
        def score(item):
            pos, path = item
            d_goal = min((abs(pos[0]-r[0])+abs(pos[1]-r[1]) for r in refs),
                         default=0)
            return (d_goal, len(path))
        return min(found, key=score)[1]

    def _bfs(self, start, moves, targets):
        seen = {start}
        q = deque([(start, [])])
        tol = max(2, self.quantum)
        n = 0
        while q and n < 30000:
            pos, path = q.popleft()
            n += 1
            if targets is not None:
                hit = None
                for t in targets:
                    if abs(t[0]-pos[0]) <= tol and abs(t[1]-pos[1]) <= tol:
                        hit = t; break
                if hit is not None:
                    return path
            else:
                if pos not in self.visited and pos != start:
                    return path
            for a in moves:
                np_ = self._lattice(self._predict_dest(pos, a))
                if np_ == pos or np_ in seen or np_ in self.blocked:
                    continue
                if not (0 <= np_[0] < 64 and 0 <= np_[1] < 64):
                    continue
                if self._edge_blocked(pos, np_):
                    continue
                seen.add(np_)
                q.append((np_, path+[a]))
        return None

    def _edge_blocked(self, pos, np_):
        """True if the gap region between avatar footprint at pos and at np_
        contains a learned wall cell (or np_ itself is a wall cell)."""
        wcells = self._wcells
        pos = (int(round(pos[0])), int(round(pos[1])))
        np_ = (int(round(np_[0])), int(round(np_[1])))
        if np_ in wcells or np_ in self.blocked:
            return True
        w, h = getattr(self, "_av_dims", (1, 1))
        dx, dy = np_[0] - pos[0], np_[1] - pos[1]
        if abs(dx) >= abs(dy):
            s = 1 if dx > 0 else -1
            x0, x1 = pos[0] + s * (w // 2 + 1), np_[0] - s * (w // 2 + 1)
            lo, hi = min(x0, x1), max(x0, x1)
            ylo = min(pos[1], np_[1]) - h // 2
            yhi = max(pos[1], np_[1]) + h // 2
            for x in range(lo, hi + 1):
                for y in range(ylo, yhi + 1):
                    if (x, y) in wcells:
                        return True
        else:
            s = 1 if dy > 0 else -1
            y0, y1 = pos[1] + s * (h // 2 + 1), np_[1] - s * (h // 2 + 1)
            lo, hi = min(y0, y1), max(y0, y1)
            xlo = min(pos[0], np_[0]) - w // 2
            xhi = max(pos[0], np_[0]) + w // 2
            for y in range(lo, hi + 1):
                for x in range(xlo, xhi + 1):
                    if (x, y) in wcells:
                        return True
        return False

    def _obj_key(self, prev_grid, pos):
        """(key, cells) of the object containing pos in the previous frame."""
        if prev_grid is None:
            return None, None
        for o in all_objects(prev_grid):
            cells = [tuple(c) for c in o["cells"]]
            if pos in cells:
                key = (o["color"], int(o["centroid"][0]) // 8,
                       int(o["centroid"][1]) // 8)
                return key, cells
        return None, None

    # ---------- click / no-avatar ----------
    def _click_or_probe(self, grid, available):
        if CLICK in available:
            # 1) dwell on "effective" objects (clicks that produced big
            #    changes), alternating with the sweep so new effective
            #    objects keep being discovered. Dwell = press the SAME
            #    object repeatedly — multi-press mechanics reset progress
            #    when a different object is pressed.
            if self._eff_obj and (self._eff_turn or not self._sweep):
                self._eff_turn = False
                for _ in range(len(self._eff_obj)):
                    eo = self._eff_obj[0]
                    if self._eff_presses[eo["key"]] >= 32:
                        # pressed dozens of times this level with no
                        # completion — dead-end object, stop dwelling
                        self._eff_suppress.add(eo["key"])
                        self._eff_obj.popleft()
                        self._eff_obj_keys.discard(eo["key"])
                        continue
                    if eo["presses"] < 8:
                        eo["presses"] += 1
                        self._eff_presses[eo["key"]] += 1
                        x, y = eo["cells"][eo["presses"] % len(eo["cells"])]
                        self._press_count[(x, y)] += 1
                        return (CLICK, {"x": int(x), "y": int(y)})
                    self._eff_obj.rotate(-1)
                self._eff_obj.clear()
                self._eff_obj_keys.clear()
            else:
                self._eff_turn = True
            # 2) a color whose click already completed a level stays hot
            if self.click_color is not None:
                pri = [o for o in all_objects(grid)
                       if o["color"] == self.click_color]
                if pri:
                    o = self.rng.choice(pri)
                    x, y = self.rng.choice(o["cells"])
                    self._press_count[(x, y)] += 1
                    return (CLICK, {"x": x, "y": y})
            # 3) systematic sweep: salient object cells first (smallest
            #    objects are the most likely targets), then a coarse lattice
            #    covering background/empty positions
            if self._sweep is None:
                self._sweep = deque()
                for o in sorted(all_objects(grid), key=lambda o: o["size"]):
                    if o["size"] > 0.2 * 64 * 64 or \
                            self.hazard_count[o["color"]] >= 2:
                        continue
                    cells = o["cells"]
                    step = max(1, len(cells) // 2)
                    for c in cells[::step][:2]:
                        self._sweep.append(tuple(c))
                for yy in range(0, 64, 4):
                    for xx in range(0, 64, 4):
                        self._sweep.append((xx, yy))
            while self._sweep:
                pos = self._sweep.popleft()
                if pos not in self.clicked:
                    self.clicked.add(pos)
                    x, y = pos
                    self._press_count[pos] += 1
                    return (CLICK, {"x": int(x), "y": int(y)})
            # 4) exhausted: hazard-aware random object clicks
            objs = [o for o in all_objects(grid)
                    if o["size"] <= 0.2 * 64 * 64
                    and self.hazard_count[o["color"]] < 2]
            if objs:
                o = self.rng.choice(objs)
                x, y = self.rng.choice(o["cells"])
                self._press_count[(x, y)] += 1
                return (CLICK, {"x": x, "y": y})
            pos = (self.rng.randint(0, 63), self.rng.randint(0, 63))
            self._press_count[pos] += 1
            return (CLICK, {"x": pos[0], "y": pos[1]})
        return self._random(available)

    def _random(self, available):
        a = self.rng.choice([x for x in available if x != A_RESET])
        data = {}
        if a == CLICK:
            data = {"x": self.rng.randint(0, 63), "y": self.rng.randint(0, 63)}
        return (a, data)


from arcengine import FrameData, GameAction, GameState

# When run inside the ARC-AGI-3-Agents framework (locally or on Kaggle)
# the `agents` package is on sys.path, so this import resolves.
from agents.agent import Agent


class MyAgent(Agent):
    """Framework wrapper around OceanCore (see module docstring)."""

    # Per-game action budget; the framework enforces global limits too.
    MAX_ACTIONS = 8000

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        seed = (hash(self.game_id) % 1_000_000)
        self.core = OceanCore(seed=seed)

    @property
    def name(self) -> str:
        return f"ocean.{self.MAX_ACTIONS}"

    def is_done(self, frames, latest_frame) -> bool:
        # Stop on WIN; on GAME_OVER we RESET and keep trying.
        return latest_frame.state is GameState.WIN

    def choose_action(self, frames, latest_frame) -> GameAction:
        if latest_frame.is_empty() or not latest_frame.frame:
            return GameAction.RESET
        grid = latest_frame.frame[-1]
        aid, data = self.core.decide(
            grid,
            list(latest_frame.available_actions),
            latest_frame.state.name,
            latest_frame.levels_completed,
            self.action_counter,
        )
        action = GameAction.from_id(aid)
        if data:
            action.set_data(data)
            action.reasoning = {"why": "ocean click", "x": data["x"], "y": data["y"]}
        else:
            action.reasoning = f"ocean action {aid}"
        return action
