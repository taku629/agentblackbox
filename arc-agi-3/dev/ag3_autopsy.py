"""AGI-3 level autopsy: where every game died, what it cost, and what 30 points would take.

    python3 ag3_autopsy.py <kernel_output_dir> [--label notes] [--target 30] [--transfer 0.48]
                           [--at par|obs|<level score>] [--json out.json] [--no-games]
    python3 ag3_autopsy.py <dir_a> --compare <dir_b>      # per-game A/B on levels, score and cause
    python3 ag3_autopsy.py --structural [--target 30]     # no logs needed: what the scoring rule demands
    python3 ag3_autopsy.py --selftest

Reads whatever the harness wrote under the directory (searched recursively), nothing else:
  benchmark.json                    game_runs[]: base_actions_per_level, actions_per_level, levels_completed,
                                    final_score, state, history[].wallclock_seconds (cumulative)
  artifacts/<game>_p<i>_viewer_data.json   levels_completed, total_levels, actions_per_level, final_score
  artifacts/<game>_p<i>_events.jsonl       one JSON per line: type initial|action|analysis|experiment with
                                    score (= levels completed), board, board_diff_cells, game_over,
                                    analysis_step and, for analysis events, the turn's transcript
  *.log / top-level *.txt           "[finished] <id> state=.. level=k/N score=.. per-level=a/b,a/b,.." lines (kernel log);
                                    enough for the loss accounting and what-ifs when nothing else was downloaded
  score.json                        {"games": {<id>: {"score": ..}}} -- only cross-checked
Baselines come from benchmark.json, else --env-dir (environment_files/*/*/metadata.json), else the
public-25 table embedded below.

Scoring rule (taaf GameRun._compute_final_score, mirrored in game_score and tested against the real source):
  level l (1-based) completed with a actions: s_l = min(115, (baseline_l / a)^2 * 100); weight l;
  game = sum(l * s_l) / sum(l), capped at sum(l for completed) / sum(l) * 100; LB = mean over games.

Every game has one FRONTIER level: the first one it did not complete. Its cause, first match wins:
  crashed        run state crashed/cancelled, or no action at all
  wm_drop        wm.health() verdict 'drop' / stage 'fallback' seen on the frontier        (wm arm only)
  wm_diverged    >= 3 wm.execute stops with 'diverged', or last wm accuracy < 0.9          (wm arm only)
  goal_unfired   wm accuracy >= 0.9 on the frontier but no plan / goal never reached       (wm arm only)
  game_over      >= 1 GAME_OVER on the frontier and >= 30% of its actions were replays after one
  noop_waste     >= 50% of frontier actions changed <= --hud-cells cells
  loop           <= 50% of the boards produced on the frontier were new
  starved        fewer frontier actions than the human baseline for that level
  no_mechanics   none of the above: enough effective, novel actions, still no completion
Buckets: explore = noop_waste + loop; model = no_mechanics + wm_diverged + wm_drop + game_over;
budget = starved; goal = goal_unfired; infra = crashed; efficiency = points lost on COMPLETED levels.

Loss accounting per game, exact: 100 - score = efficiency + frontier + beyond, where
  efficiency = 100 * sum(l for completed) / sum(l) - score
  frontier   = 100 * (k + 1) / sum(l)              (the level it died on)
  beyond     = 100 * sum(l for l > k + 1) / sum(l)  (never seen; needs the frontier first)
"""
import argparse
import glob
import hashlib
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))

# public-25 baseline_actions per level (environment_files/*/*/metadata.json, downloaded 2026-09-20)
BASELINES = {
    "ar25": [32, 50, 75, 37, 89, 159, 233, 73], "bp35": [21, 48, 44, 38, 33, 87, 86, 131, 163],
    "cd82": [55, 8, 41, 21, 23, 23], "cn04": [29, 54, 85, 300, 208, 113], "dc22": [59, 102, 67, 98, 324, 578],
    "ft09": [43, 12, 23, 28, 65, 37], "g50t": [78, 175, 179, 230, 96, 54, 67],
    "ka59": [28, 109, 51, 51, 33, 132, 326], "lf52": [32, 81, 60, 71, 205, 148, 244, 109, 164, 225],
    "lp85": [17, 38, 31, 16, 41, 60, 26, 159], "ls20": [22, 123, 73, 84, 96, 192, 186],
    "m0r0": [30, 111, 203, 26, 500, 237], "r11l": [22, 33, 51, 26, 52, 49],
    "re86": [26, 42, 86, 108, 189, 139, 424, 241], "s5i5": [20, 89, 106, 54, 162, 38, 86, 83],
    "sb26": [18, 28, 18, 19, 31, 23, 58, 18], "sc25": [36, 6, 32, 83, 143, 50],
    "sk48": [61, 177, 101, 103, 230, 181, 125, 92], "sp80": [39, 58, 25, 148, 96, 152],
    "su15": [22, 42, 26, 115, 36, 31, 8, 40, 41], "tn36": [32, 72, 26, 40, 30, 55, 62],
    "tr87": [54, 58, 40, 45, 71, 146], "tu93": [19, 16, 34, 42, 123, 80, 14, 23, 111],
    "vc33": [7, 18, 44, 61, 131, 34, 152], "wa30": [71, 119, 183, 98, 368, 68, 79, 442, 415],
}

CAUSES = ["crashed", "wm_drop", "wm_diverged", "goal_unfired", "game_over", "noop_waste", "loop", "starved",
          "no_mechanics"]
BUCKET = {"crashed": "infra", "wm_drop": "model", "wm_diverged": "model", "goal_unfired": "goal",
          "game_over": "model", "noop_waste": "explore", "loop": "explore", "starved": "budget",
          "no_mechanics": "model"}
BUCKETS = ["explore", "model", "budget", "goal", "efficiency", "infra"]
BUCKET_JA = {"explore": "exploration (no-ops / loops)", "model": "model accuracy (mechanics not understood)",
             "budget": "turn budget (ran out before trying enough)", "goal": "completion test (is_goal never fired)",
             "efficiency": "action waste on completed levels", "infra": "crash"}

RE_MARK = re.compile(r"^\[([A-Z][^\]\n]*)\]\s*$", re.M)
RE_VERDICT = re.compile(r"\bverdict['\"]?\s*[:=]\s*['\"]?(drop|fix|use|probe)\b")
RE_STAGE = re.compile(r"\bstage['\"]?\s*[:=]\s*['\"]?([a-z_]+)")
RE_STOP = re.compile(r"\bstopped['\"]?\s*[:=]\s*['\"]?([a-z_]+)")
RE_ACC = re.compile(r"\b(?:check_acc|acc_recent|acc)['\"]?\s*[:=]\s*([01](?:\.\d+)?)\b")
RE_FINISHED = re.compile(r"\[finished\] (\S+) state=(\w+) level=(\d+)/(\d+) score=([\d.]+) actions=\d+ tokens=\d+ "
                         r"per-level=([\d/?,]+)")
RE_FOUND = re.compile(r"\bfound['\"]?\s*[:=]\s*(true|false|True|False)\b")


def short(game_id):
    return str(game_id).split("-")[0]


def level_score(baseline, actions):
    return min(115.0, (baseline / actions) ** 2 * 100) if actions > 0 else 0.0


def game_score(baseline, actions, completed):
    """Mirror of taaf GameRun._compute_final_score. `actions`/`baseline` per level, `completed` = count."""
    n = len(baseline)
    if n == 0:
        return 0.0
    total, wsum, wmax = 0.0, 0, 0
    for i in range(n):
        w = i + 1
        wsum += w
        a = actions[i] if i < len(actions) else 0
        s = level_score(baseline[i], a) if i < completed and a > 0 else 0.0
        if s > 0:
            wmax += w
        total += s * w
    return min(total / wsum, wmax / wsum * 100)


# ----------------------------------------------------------------------------------------------- loading

def _tool_results(transcript):
    """Text of the TOOL RESULT sections only (the system prompt also says 'drop', 'diverged', ...)."""
    marks = list(RE_MARK.finditer(transcript))
    out = []
    for i, m in enumerate(marks):
        if m.group(1).strip().upper().startswith("TOOL RESULT"):
            out.append(transcript[m.end():marks[i + 1].start() if i + 1 < len(marks) else len(transcript)])
    return "\n".join(out)


def _wm_signals(text):
    sig = {"drop": 0, "diverged": 0, "acc": [], "noplan": 0, "wm_calls": 0}
    sig["drop"] = sum(v == "drop" for v in RE_VERDICT.findall(text)) + sum(s == "fallback" for s in RE_STAGE.findall(text))
    stages = RE_STAGE.findall(text)
    sig["wm_calls"] = len(stages) + len(RE_VERDICT.findall(text))
    sig["diverged"] = sum(s == "diverged" for s in RE_STOP.findall(text))
    sig["acc"] = [float(a) for a in RE_ACC.findall(text)]
    sig["noplan"] = sum(f.lower() == "false" for f in RE_FOUND.findall(text)) + sum(s in ("goal_unknown", "explore") for s in stages)
    return sig


def _level_stats():
    return {"actions": 0, "ineffective": 0, "game_over": 0, "after_game_over": 0, "resets": 0, "boards": 0,
            "new_boards": 0, "turns": 0, "drop": 0, "diverged": 0, "acc": [], "noplan": 0, "wm_calls": 0}


def parse_events(path, hud_cells=4):
    """events.jsonl -> per-level stats (index 0 = level 1), turns, wm signals."""
    levels, seen = {}, {}
    prev_score, turn_level, n_turns, had_game_over = 0, {}, 0, {}
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except ValueError:
                continue
            typ = e.get("type")
            if typ == "action":
                lv = prev_score
                st = levels.setdefault(lv, _level_stats())
                st["actions"] += 1
                turn_level.setdefault(e.get("analysis_step"), lv)
                d = e.get("board_diff_cells")
                if e.get("board_changed") is False or (isinstance(d, int) and d <= hud_cells):
                    st["ineffective"] += 1
                if str(e.get("action_name") or e.get("action_display") or "").upper().startswith("RESET"):
                    st["resets"] += 1
                if had_game_over.get(lv):
                    st["after_game_over"] += 1
                if e.get("game_over"):
                    st["game_over"] += 1
                    had_game_over[lv] = True
                b = e.get("board")
                if b is not None:
                    h = hashlib.md5(json.dumps(b, separators=(",", ":")).encode()).digest()
                    st["boards"] += 1
                    if h not in seen.setdefault(lv, set()):
                        seen[lv].add(h)
                        st["new_boards"] += 1
                if isinstance(e.get("score"), int):
                    prev_score = e["score"]
            elif typ == "analysis":
                n_turns += 1
                lv = turn_level.get(e.get("analysis_step"), prev_score)
                st = levels.setdefault(lv, _level_stats())
                st["turns"] += 1
                tr = e.get("transcript") or ""
                if tr:
                    s = _wm_signals(_tool_results(tr))
                    for k in ("drop", "diverged", "noplan", "wm_calls"):
                        st[k] += s[k]
                    st["acc"] += s["acc"]
            elif typ == "initial" and isinstance(e.get("score"), int):
                prev_score = e["score"]
    return levels, n_turns, prev_score


def load_run(root, env_dir=None, hud_cells=4):
    """-> list of game dicts, one per (game, pass)."""
    env_base = dict(BASELINES)
    for cand in ([env_dir] if env_dir else []) + [os.path.join(REPO, "arc-agi-3", "environment_files")]:
        for p in glob.glob(os.path.join(cand, "*", "*", "metadata.json")):
            try:
                m = json.load(open(p))
                env_base[short(m["game_id"])] = list(m["baseline_actions"])
            except Exception:
                pass
        if env_dir:
            break
    games = {}

    def slot(game_id, p):
        return games.setdefault((short(game_id), p), {"game": short(game_id), "pass": p, "sources": []})

    for bj in sorted(glob.glob(os.path.join(root, "**", "benchmark.json"), recursive=True)):
        try:
            d = json.load(open(bj))
        except Exception:
            continue
        runs = d.get("game_runs") or []
        n_pass = max(1, int(d.get("n_passes") or 1))
        per = max(1, len(runs) // n_pass)
        for i, r in enumerate(runs):
            g = slot(r.get("game_id"), i // per)
            if "benchmark" in g["sources"]:
                continue
            g["sources"].append("benchmark")
            g.update(n_levels=r.get("number_of_levels"), baseline=r.get("base_actions_per_level"),
                     actions=list(r.get("actions_per_level") or []), completed=int(r.get("levels_completed") or 0),
                     reported=r.get("final_score"), state=r.get("state"), note=r.get("solver_note"))
            wall = [h.get("wallclock_seconds") for h in r.get("history") or []]
            if wall and all(isinstance(w, (int, float)) for w in wall):
                g["wall"], g["wall_end"] = wall, r.get("final_wallclock_seconds") or wall[-1]
    for vp in sorted(glob.glob(os.path.join(root, "**", "*_viewer_data.json"), recursive=True)):
        m = re.match(r"(.+)_p(\d+)_viewer_data\.json$", os.path.basename(vp))
        if not m:
            continue
        try:
            d = json.load(open(vp))
        except Exception:
            continue
        g = slot(d.get("game_id") or m.group(1), int(m.group(2)))
        if "benchmark" not in g["sources"] and "levels_completed" in d:
            g.update(n_levels=d.get("total_levels"), actions=list(d.get("actions_per_level") or []),
                     completed=int(d.get("levels_completed") or 0), reported=d.get("final_score"), state=d.get("status"))
        g["sources"].append("viewer")
    for ep in sorted(glob.glob(os.path.join(root, "**", "*_events.jsonl"), recursive=True)):
        m = re.match(r"(.+)_p(\d+)_events\.jsonl$", os.path.basename(ep))
        if not m:
            continue
        g = slot(m.group(1), int(m.group(2)))
        g["levels"], g["turns"], g["ev_completed"] = parse_events(ep, hud_cells)
        g["sources"].append("events")
    # kernel log lines printed by taaf Game.finish_game:
    #   [finished] <id> state=gave_up level=2/6 score=14.29 actions=310 tokens=.. per-level=50/43,20/12,240/23,0/28,..
    for lp in sorted(glob.glob(os.path.join(root, "**", "*.log"), recursive=True)
                     + glob.glob(os.path.join(root, "*.txt"))):
        try:
            text = open(lp, encoding="utf-8", errors="replace").read()
        except OSError:
            continue
        for m in RE_FINISHED.finditer(text):
            g = slot(m.group(1), 0)
            if "benchmark" in g["sources"] or "log" in g["sources"]:
                continue
            pairs = [x.split("/") for x in m.group(6).split(",")]
            g["sources"].append("log")
            if "completed" in g and "viewer" in g["sources"]:
                continue
            g.update(n_levels=int(m.group(4)), completed=int(m.group(3)), reported=float(m.group(5)), state=m.group(2),
                     actions=[int(a) for a, _ in pairs])
            if all(b.isdigit() for _, b in pairs):
                g["baseline"] = [int(b) for _, b in pairs]
    score_json = {}
    for sp in glob.glob(os.path.join(root, "**", "score.json"), recursive=True):
        try:
            for gid, v in (json.load(open(sp)).get("games") or {}).items():
                score_json[short(gid)] = v.get("score") if isinstance(v, dict) else v
        except Exception:
            pass
    out = []
    for key in sorted(games):
        g = games[key]
        if not g.get("baseline"):
            g["baseline"] = env_base.get(g["game"])
        if not g.get("baseline"):
            g["skip"] = "no baseline (pass --env-dir)"
            out.append(g)
            continue
        g["n_levels"] = int(g.get("n_levels") or len(g["baseline"]))
        lv = g.get("levels") or {}
        if "completed" not in g:                              # events only
            g["completed"] = int(g.get("ev_completed") or 0)
            g["actions"] = [lv.get(i, _level_stats())["actions"] for i in range(g["n_levels"])]
            g["state"] = g.get("state") or "unknown"
        g["actions"] = (list(g.get("actions") or []) + [0] * g["n_levels"])[:g["n_levels"]]
        g["score_json"] = score_json.get(g["game"])
        out.append(g)
    return out


# -------------------------------------------------------------------------------------------- per game

def classify(g):
    """Cause of death on the frontier level (None if the game was won)."""
    k, n = g["completed"], g["n_levels"]
    if k >= n:
        return None, {}
    st = (g.get("levels") or {}).get(k) or _level_stats()
    a = g["actions"][k] if k < len(g["actions"]) else 0
    a = a or st["actions"]
    base = g["baseline"][k]
    ev = st["actions"]
    info = {"actions": a, "baseline": base, "turns": st["turns"],
            "ineff": st["ineffective"] / ev if ev else None,
            "novel": st["new_boards"] / st["boards"] if st["boards"] else None,
            "replay": st["after_game_over"] / ev if ev else None,
            "game_over": st["game_over"], "wm_calls": st["wm_calls"]}
    if g.get("state") in ("crashed", "cancelled") or sum(g["actions"]) == 0:
        return "crashed", info
    if st["wm_calls"]:
        acc = st["acc"][-1] if st["acc"] else None
        info["wm_acc"] = acc
        if st["drop"]:
            return "wm_drop", info
        if st["diverged"] >= 3 or (acc is not None and acc < 0.9):
            return "wm_diverged", info
        if acc is not None and acc >= 0.9 and st["noplan"]:
            return "goal_unfired", info
    if st["game_over"] and info["replay"] is not None and info["replay"] >= 0.3:
        return "game_over", info
    if info["ineff"] is not None and info["ineff"] >= 0.5:
        return "noop_waste", info
    if info["novel"] is not None and info["novel"] <= 0.5:
        return "loop", info
    if a < base:
        return "starved", info
    return "no_mechanics", info


def autopsy_game(g):
    base, k, n = g["baseline"], g["completed"], g["n_levels"]
    W = n * (n + 1) / 2
    g["score"] = game_score(base, g["actions"], k)
    done_w = sum(i + 1 for i in range(k) if g["actions"][i] > 0)
    g["lvl_scores"] = [level_score(base[i], g["actions"][i]) for i in range(k)]
    g["ratios"] = [g["actions"][i] / base[i] for i in range(k) if g["actions"][i] > 0]
    g["cause"], g["frontier"] = classify(g)
    g["loss_eff"] = 100.0 * done_w / W - g["score"]
    g["loss_frontier"] = 100.0 * (k + 1) / W if k < n else 0.0
    g["loss_beyond"] = 100.0 * sum(i + 1 for i in range(k + 1, n)) / W
    # completed levels that scored 0 (0 actions recorded) would break the identity; fold them into efficiency
    g["loss_eff"] += 100.0 - g["score"] - g["loss_eff"] - g["loss_frontier"] - g["loss_beyond"]
    if g.get("wall") and len(g["wall"]) == sum(g["actions"]):
        cum, t, prev = 0, [], 0.0
        for i in range(n):
            cum += g["actions"][i]
            end = g["wall"][cum - 1] if g["actions"][i] else prev
            t.append(end - prev)
            prev = end
        if k < n:
            t[k] = max(t[k], (g.get("wall_end") or prev) - (prev - t[k]))
        g["time"] = t
    return g


# ------------------------------------------------------------------------------------------- what-ifs

def what_if_levels(games, extra, s):
    """Mean score if every game completes `extra` more levels, each new level scoring s (0-115)."""
    tot = 0.0
    for g in games:
        n = g["n_levels"]
        W = n * (n + 1) / 2
        add = sum((i + 1) * min(s, 115.0) for i in range(g["completed"], min(n, g["completed"] + extra)))
        tot += min(100.0, g["score"] + add / W)
    return tot / len(games)


def levels_needed(games, target, s):
    """Fewest additional completed levels (greedy on gain per level, frontier first within a game) to reach
    `target` mean, each new level scoring s. -> (count, mean reached, {game: extra levels}) ; count None if unreachable."""
    cur = {g["game"]: g["completed"] for g in games}
    mean = sum(g["score"] for g in games) / len(games)
    plan, count = {}, 0
    while mean < target - 1e-9:
        best = None
        for g in games:
            k, n = cur[g["game"]], g["n_levels"]
            if k >= n:
                continue
            gain = (k + 1) * min(s, 115.0) / (n * (n + 1) / 2)
            if best is None or gain > best[0]:
                best = (gain, g)
        if best is None:
            return None, mean, plan
        cur[best[1]["game"]] += 1
        plan[best[1]["game"]] = plan.get(best[1]["game"], 0) + 1
        mean += best[0] / len(games)
        count += 1
    return count, mean, plan


def structural(baselines, target=30.0, ratios=(1.0, 1.25, 1.5, 1.83, 2.0, 3.0)):
    """What the scoring rule alone demands. -> rows (ratio, level score, depth fraction needed, levels needed)."""
    rows = []
    total_levels = sum(len(b) for b in baselines.values())
    for r in ratios:
        s = min(115.0, 100.0 / (r * r))
        fake = [{"game": k, "n_levels": len(b), "completed": 0, "score": 0.0} for k, b in baselines.items()]
        cnt, _, _ = levels_needed(fake, target, s)
        depth = None                                            # every game completes its first ceil(f * n) levels
        for f100 in range(1, 101):
            m = 0.0
            for b in baselines.values():
                n = len(b)
                kk = -(-n * f100 // 100)
                m += sum(i + 1 for i in range(kk)) * s / (n * (n + 1) / 2)
            if m / len(baselines) >= target:
                depth = f100 / 100.0
                break
        rows.append({"ratio": r, "level_score": s, "all_levels_mean": s if s <= 100 else 100.0,
                     "levels_needed_greedy": cnt, "of_levels": total_levels, "uniform_depth": depth})
    return rows


def analyse(games, target=30.0, transfer=None, at="obs"):
    ok = [autopsy_game(g) for g in games if not g.get("skip")]
    if not ok:
        raise SystemExit("no game found (expected benchmark.json and/or artifacts/*_events.jsonl under the directory)")
    n = len(ok)
    mean = sum(g["score"] for g in ok) / n
    rep = [g["reported"] for g in ok if isinstance(g.get("reported"), (int, float))]
    lv_scores = sorted(s for g in ok for s in g["lvl_scores"])
    ratios = sorted(r for g in ok for r in g["ratios"])
    med = lambda v: (v[len(v) // 2] if len(v) % 2 else (v[len(v) // 2 - 1] + v[len(v) // 2]) / 2) if v else None   # noqa: E731
    obs_s = med(lv_scores) if lv_scores else 100.0
    s_new = obs_s if at == "obs" else 100.0 if at == "par" else float(at)
    layers = {c: {"games": [], "frontier": 0.0, "beyond": 0.0} for c in CAUSES}
    for g in ok:
        if g["cause"]:
            L = layers[g["cause"]]
            L["games"].append(g["game"])
            L["frontier"] += g["loss_frontier"] / n
            L["beyond"] += g["loss_beyond"] / n
    buckets = {b: 0.0 for b in BUCKETS}
    buckets["efficiency"] = sum(g["loss_eff"] for g in ok) / n
    for c, L in layers.items():
        buckets[BUCKET[c]] += L["frontier"] + L["beyond"]
    fix = {}
    for c, L in layers.items():                                   # +1 level for every game that died of c
        if L["games"]:
            sub = [dict(g, completed=g["completed"] + (g["cause"] == c)) for g in ok]
            gain = sum((g["completed"] + 1) * min(s_new, 115.0) / (g["n_levels"] * (g["n_levels"] + 1) / 2)
                       for g in ok if g["cause"] == c) / n
            fix[c] = {"games": len(L["games"]), "gain": gain, "mean": mean + gain}
            del sub
    need_obs = levels_needed(ok, target, s_new)
    need_par = levels_needed(ok, target, 100.0)
    turns = [g["turns"] for g in ok if g.get("turns")]
    res = {
        "n_games": n, "mean": mean, "reported_mean": sum(rep) / len(rep) if rep else None,
        "levels_completed": sum(g["completed"] for g in ok), "levels_total": sum(g["n_levels"] for g in ok),
        "games_scoring": sum(g["score"] > 0 for g in ok),
        "median_level_score": med(lv_scores), "median_action_ratio": med(ratios),
        "new_level_score": s_new, "target": target,
        "loss": {"total": 100.0 - mean, "efficiency": buckets["efficiency"],
                 "frontier": sum(g["loss_frontier"] for g in ok) / n, "beyond": sum(g["loss_beyond"] for g in ok) / n},
        "layers": layers, "buckets": buckets, "fix_one_level": fix,
        "what_if": {"efficiency_par": sum(g["score"] + max(0.0, g["loss_eff"]) for g in ok) / n,
                    "plus_levels": {x: {"obs": what_if_levels(ok, x, s_new), "par": what_if_levels(ok, x, 100.0)}
                                    for x in (1, 2, 3)},
                    "levels_needed": {"at_new_level_score": need_obs[0], "plan": need_obs[2],
                                      "at_par": need_par[0], "plan_par": need_par[2]}},
        "turns": {"mean": sum(turns) / len(turns) if turns else None,
                  "frontier_share": (sum((g.get("levels") or {}).get(g["completed"], {}).get("turns", 0) for g in ok)
                                     / max(1, sum(turns))) if turns else None},
        "wm_signals": any(s.get("wm_calls") for g in ok for s in (g.get("levels") or {}).values()),
        "events": sum("events" in g["sources"] for g in ok),
        "games": ok, "skipped": [g["game"] for g in games if g.get("skip")],
    }
    live = {b: v for b, v in buckets.items() if b != "efficiency"}
    res["dominant"] = max(live, key=live.get) if any(live.values()) else None
    if transfer:
        res["public_target"] = target / transfer
        res["levels_needed_public"] = levels_needed(ok, target / transfer, s_new)[0]
    return res


# -------------------------------------------------------------------------------------------- printing

def _pct(v):
    return "   -" if v is None else "%3.0f%%" % (100 * v)


def report(res, label="", show_games=True):
    p = print
    p("=" * 100)
    p("AGI-3 level autopsy%s: %d games, mean %.2f%s, %d/%d levels completed, %d games score > 0"
      % (" [%s]" % label if label else "", res["n_games"], res["mean"],
         " (harness reported %.2f)" % res["reported_mean"] if res["reported_mean"] is not None else "",
         res["levels_completed"], res["levels_total"], res["games_scoring"]))
    if res["skipped"]:
        p("  skipped (no baseline): " + ", ".join(res["skipped"]))
    if res["events"] < res["n_games"]:
        p("  NOTE: events.jsonl found for %d/%d games; without it the frontier cause falls back to "
          "starved / no_mechanics from action counts alone" % (res["events"], res["n_games"]))
    if not res["wm_signals"]:
        p("  NOTE: no wm outputs (stage / verdict / acc) in any tool result -> wm_drop, wm_diverged and "
          "goal_unfired cannot occur in this run; run it on the wm arm output to fill them")
    if show_games:
        p("")
        p("%-5s %-5s %6s  %-22s | frontier: %4s %5s %5s %5s %5s %3s  %s"
          % ("game", "lv", "score", "done lv actions/base", "act", "base", "turns", "noop", "novel", "GO", "cause"))
        for g in sorted(res["games"], key=lambda g: -g["score"]):
            done = " ".join("%d/%d" % (g["actions"][i], g["baseline"][i]) for i in range(g["completed"]))[:22]
            f = g["frontier"]
            if g["cause"] is None:
                p("%-5s %d/%-3d %6.2f  %-22s | won" % (g["game"], g["completed"], g["n_levels"], g["score"], done))
                continue
            p("%-5s %d/%-3d %6.2f  %-22s |           %4d %5d %5d %5s %5s %3d  %s"
              % (g["game"], g["completed"], g["n_levels"], g["score"], done, f["actions"], f["baseline"], f["turns"],
                 _pct(f["ineff"]), _pct(f["novel"]), f["game_over"], g["cause"]))
    L = res["loss"]
    p("")
    p("WHERE THE POINTS ARE (LB points, mean over games; identity: 100 - mean = efficiency + frontier + beyond)")
    p("  100 - %.2f = %.2f   = efficiency %.2f + frontier level %.2f + levels beyond it %.2f"
      % (res["mean"], L["total"], L["efficiency"], L["frontier"], L["beyond"]))
    if res["median_action_ratio"] is not None:
        p("  completed levels: median %.2fx the baseline actions -> median level score %.1f"
          % (res["median_action_ratio"], res["median_level_score"]))
    p("")
    p("%-13s %-8s %5s %10s %9s   %s" % ("cause", "bucket", "games", "frontier", "beyond", "games"))
    for c in CAUSES:
        x = res["layers"][c]
        if x["games"]:
            p("%-13s %-8s %5d %10.2f %9.2f   %s" % (c, BUCKET[c], len(x["games"]), x["frontier"], x["beyond"],
                                                 " ".join(x["games"])))
    p("")
    p("BUCKETS (points lost = frontier + beyond of the games that died there; efficiency = completed levels)")
    for b in sorted(BUCKETS, key=lambda b: -res["buckets"][b]):
        if res["buckets"][b] > 0:
            p("  %-11s %6.2f  %s" % (b, res["buckets"][b], BUCKET_JA[b]))
    W = res["what_if"]
    p("")
    p("WHAT-IF (a new level is assumed to score %.1f; 'par' = 100)" % res["new_level_score"])
    p("  same levels, baseline-efficient:        %.2f   (+%.2f)" % (W["efficiency_par"], W["efficiency_par"] - res["mean"]))
    for x, v in W["plus_levels"].items():
        p("  every game +%d level%s:                  %.2f   (at par %.2f)" % (x, " " if x == 1 else "s", v["obs"], v["par"]))
    for c, v in sorted(res["fix_one_level"].items(), key=lambda kv: -kv[1]["gain"]):
        p("  cure %-13s (+1 level, %2d games): %.2f   (+%.2f)" % (c, v["games"], v["mean"], v["gain"]))
    need = W["levels_needed"]
    p("")
    p("TO REACH %.0f" % res["target"])
    for key, plan, name in (("at_new_level_score", "plan", "at level score %.1f" % res["new_level_score"]),
                            ("at_par", "plan_par", "at par (100)")):
        if need[key] is None:
            p("  %-22s unreachable even with every level completed" % name)
        else:
            p("  %-22s %d more levels (now %d) -- cheapest: %s"
              % (name, need[key], res["levels_completed"],
                 " ".join("%s+%d" % kv for kv in sorted(need[plan].items(), key=lambda kv: -kv[1])[:12])))
    if "public_target" in res:
        n = res["levels_needed_public"]
        p("  with public->hidden transfer: public mean %.1f needed -> %s"
          % (res["public_target"], "unreachable at this level score" if n is None else "%d more levels" % n))
    if res["turns"]["mean"]:
        p("")
        p("TURNS: %.0f per game on average, %.0f%% of them spent on the frontier level"
          % (res["turns"]["mean"], 100 * res["turns"]["frontier_share"]))
    p("")
    if res["dominant"]:
        p("VERDICT: the largest layer is '%s' (%.2f of %.2f lost points) -- %s"
          % (res["dominant"], res["buckets"][res["dominant"]], L["total"], BUCKET_JA[res["dominant"]]))
    p("=" * 100)


def report_structural(target=30.0, baselines=None):
    baselines = baselines or BASELINES
    print("What %.0f points demands from the scoring rule alone (%d games, %d levels):"
          % (target, len(baselines), sum(len(b) for b in baselines.values())))
    print("%8s %12s %16s %22s %24s" % ("actions", "level score", "all levels done", "levels needed (best)", "every game, first x%"))
    for r in structural(baselines, target):
        print("%7.2fx %12.1f %16.1f %22s %24s"
              % (r["ratio"], r["level_score"], r["all_levels_mean"],
                 "unreachable" if r["levels_needed_greedy"] is None else "%d / %d" % (r["levels_needed_greedy"], r["of_levels"]),
                 "unreachable" if r["uniform_depth"] is None else "%.0f%% of levels" % (100 * r["uniform_depth"])))
    print("('actions' = your actions / baseline on every completed level; 'best' = greedy, always the next "
          "level of the game where it is worth most; 'first x%' = the same depth in every game)")


def compare(a, b, la="A", lb="B"):
    ga, gb = {g["game"]: g for g in a["games"]}, {g["game"]: g for g in b["games"]}
    print("%-5s | %-4s %6s %-13s | %-4s %6s %-13s | %s" % ("game", "lv", la[:6], "cause", "lv", lb[:6], "cause", "delta"))
    for k in sorted(set(ga) & set(gb), key=lambda k: gb[k]["score"] - ga[k]["score"]):
        x, y = ga[k], gb[k]
        print("%-5s | %d/%-2d %6.2f %-13s | %d/%-2d %6.2f %-13s | %+6.2f"
              % (k, x["completed"], x["n_levels"], x["score"], x["cause"] or "won", y["completed"], y["n_levels"],
                 y["score"], y["cause"] or "won", y["score"] - x["score"]))
    print("mean %s %.2f  %s %.2f  (levels %d vs %d)" % (la, a["mean"], lb, b["mean"], a["levels_completed"], b["levels_completed"]))
    for bk in BUCKETS:
        if a["buckets"][bk] or b["buckets"][bk]:
            print("  %-11s %6.2f -> %6.2f  (%+.2f)" % (bk, a["buckets"][bk], b["buckets"][bk], b["buckets"][bk] - a["buckets"][bk]))


# -------------------------------------------------------------------------------------------- selftest

def _real_score_fn():
    """The harness's own scorer, exec'd from source (taaf.game needs arcengine to import)."""
    import textwrap
    path = os.path.join(REPO, "bundle_fast", "src", "tufa-arc-agi-framework", "src", "taaf", "game.py")
    if not os.path.exists(path):
        return None
    src = open(path, encoding="utf-8").read()
    i = src.index("    def _compute_final_score(self)")
    j = src.index("\n\n\n", i)
    ns = {}
    exec(textwrap.dedent(src[i:j]), ns)
    return ns["_compute_final_score"]


def _fixture(root):
    """A run in the harness's own file formats: benchmark.json + artifacts/*_events.jsonl + viewer_data + score.json."""
    import importlib.util
    import random
    from pathlib import Path
    va = os.path.join(REPO, "bundle_fast", "src", "ARC3-Inference", "inference", "utils", "viewer_artifacts.py")
    writer = None
    if os.path.exists(va):
        spec = importlib.util.spec_from_file_location("viewer_artifacts", va)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        writer = mod.append_raw_events_sidecar
    art = os.path.join(root, "out", "artifacts")
    os.makedirs(art)
    rng = random.Random(0)
    runs, expect = [], {}

    def board(tag):
        return [[tag // 1000 % 10, tag // 100 % 10, tag // 10 % 10, tag % 10]] + [[(x + y) % 10 for x in range(4)] for y in range(3)]

    def tool(body):
        return "[SYSTEM PROMPT]\nwm.health() verdict: drop means stop; stopped: diverged\n\n[TOOL RESULT: python]\n%s\n\n" % body

    def play(game, completed_actions, frontier, cause, state="gave_up", wm=None, hidden_baseline=False):
        base = BASELINES[game]
        k, evs, score, num, step, uniq = len(completed_actions), [], 0, 0, 0, 1000
        evs.append({"type": "initial", "score": 0, "level": 1, "action_num": 0, "board": board(0)})
        wall, t = [], 0.0
        for a in completed_actions:
            for i in range(a):
                num, uniq, t = num + 1, uniq + 1, t + 20.0
                if i % 4 == 0:
                    step += 1
                last = i == a - 1
                score += last
                evs.append({"type": "action", "score": score, "level": score + 1, "action_num": num,
                            "analysis_step": step, "action_name": "ACTION1", "board_changed": True,
                            "board_diff_cells": 9, "level_completed": last, "game_over": False, "board": board(uniq)})
                wall.append(t)
                if i % 4 == 3 or last:
                    evs.append({"type": "analysis", "score": score, "level": score + 1, "analysis_step": step,
                                "action_num": num, "transcript": tool("action_num: %d" % num)})
        n_act, mode = frontier
        over = False
        for i in range(n_act):
            num, t = num + 1, t + 20.0
            if i % 4 == 0:
                step += 1
            e = {"type": "action", "score": score, "level": score + 1, "action_num": num, "analysis_step": step,
                 "action_name": "ACTION1", "board_changed": True, "board_diff_cells": 9, "level_completed": False,
                 "game_over": False}
            if mode == "noop" and i % 5 < 3:
                e.update(board_changed=i % 2 == 0, board_diff_cells=2 if i % 2 == 0 else 0, board=board(uniq))
            elif mode == "loop":
                e["board"] = board(5000 + i % 3)
            else:
                uniq += 1
                e["board"] = board(uniq)
            if mode == "game_over" and i % 10 == 2:
                e["game_over"], over = True, True
            if mode == "game_over" and over and i % 10 == 3:
                e["action_name"] = "RESET"
            evs.append(e)
            wall.append(t)
            if i % 4 == 3 or i == n_act - 1:
                evs.append({"type": "analysis", "score": score, "level": score + 1, "analysis_step": step,
                            "action_num": num, "transcript": tool(wm[(i // 4) % len(wm)] if wm else "board_changed: true")})
        evs.append({"type": "experiment", "counters": {"x": 1}, "action_num": num, "analysis_step": step})
        stem = "%s-abcd1234_p0" % game
        vpath = Path(art) / (stem + "_viewer_data.json")
        if writer:
            writer(vpath, evs)
        else:
            with open(os.path.join(art, stem + "_events.jsonl"), "w") as f:
                f.writelines(json.dumps(e, separators=(",", ":")) + "\n" for e in evs)
        actions = list(completed_actions) + [n_act] + [0] * (len(base) - k - 1)
        actions = actions[:len(base)]
        sc = game_score(base, actions, k)
        json.dump({"game_id": game + "-abcd1234", "levels_completed": k, "total_levels": len(base),
                   "actions_per_level": actions, "final_score": sc, "status": state}, open(vpath, "w"))
        runs.append({"game_id": game + "-abcd1234", "number_of_levels": len(base),
                     "base_actions_per_level": None if hidden_baseline else base, "state": state,
                     "history": [{"action": {"id": "ACTION1", "data": {}}, "wallclock_seconds": w} for w in wall],
                     "actions_per_level": actions, "levels_completed": k, "final_score": sc, "solver_note": None,
                     "final_wallclock_seconds": t + 60.0})
        expect[game] = cause
        rng.random()

    play("ft09", [50, 20], (80, "ok"), "no_mechanics")
    play("vc33", [], (40, "noop"), "noop_waste")
    play("ls20", [], (60, "loop"), "loop")
    play("wa30", [90], (10, "ok"), "starved", hidden_baseline=True)
    play("bp35", [], (50, "game_over"), "game_over")
    play("sk48", [], (70, "ok"), "wm_drop", wm=["stage: check\nacc: 0.7", "stage: fallback\nhealth:\n  verdict: drop"])
    play("tn36", [], (40, "ok"), "wm_diverged",
         wm=["stage: execute\nexecute:\n  executed: 1\n  stopped: diverged", "stage: execute\nexecute:\n  stopped: diverged"])
    play("r11l", [22], (40, "ok"), "goal_unfired", wm=["stage: plan\ncheck_acc: 1.0\nplan:\n  found: false\n  why: budget"])
    play("cd82", [], (0, "ok"), "crashed", state="crashed")
    play("sb26", [18, 28, 18, 19, 31, 23, 58, 18], (0, "ok"), None, state="won")
    json.dump({"label": "fixture", "n_passes": 1, "game_runs": runs}, open(os.path.join(root, "out", "benchmark.json"), "w"))
    json.dump({"games": {r["game_id"]: {"score": r["final_score"]} for r in runs}},
              open(os.path.join(root, "out", "score.json"), "w"))
    return os.path.join(root, "out"), expect, bool(writer)


def selftest():
    import contextlib
    import io
    import tempfile
    import types
    # 1. scorer == the harness's own source
    real = _real_score_fn()
    cases = [([43, 12, 23, 28, 65, 37], [50, 20, 80, 0, 0, 0], 2), ([7, 18, 44], [3, 18, 0], 2), ([10, 10], [0, 0], 0),
             ([10, 10, 10], [10, 10, 10], 3), ([20, 30, 40, 50], [200, 30, 10, 0], 3), ([5, 5], [0, 9], 1)]
    for base, act, k in cases:
        mine = game_score(base, act, k)
        if real:
            run = types.SimpleNamespace(base_actions_per_level=base, number_of_levels=len(base), levels_completed=k,
                                        actions_per_level=act)
            assert abs(real(run) - mine) < 1e-9, (base, act, k, real(run), mine)
    assert abs(game_score([43, 12, 23, 28, 65, 37], [43, 12, 0, 0, 0, 0], 2) - 300 / 21) < 1e-9     # ft09 lv2/6 at par = 14.29
    # 2. full pipeline on a fixture in the harness's file formats
    with tempfile.TemporaryDirectory() as tmp:
        root, expect, real_writer = _fixture(tmp)
        games = load_run(root)
        res = analyse(games, target=30.0, transfer=0.48)
        got = {g["game"]: g["cause"] for g in res["games"]}
        assert got == expect, {k: (got[k], expect[k]) for k in got if got[k] != expect[k]}
        assert res["n_games"] == 10 and abs(res["mean"] - res["reported_mean"]) < 1e-9
        L = res["loss"]
        assert abs(L["total"] - (L["efficiency"] + L["frontier"] + L["beyond"])) < 1e-9
        assert abs(sum(res["buckets"].values()) - L["total"]) < 1e-9
        for g in res["games"]:
            assert abs(100 - g["score"] - g["loss_eff"] - g["loss_frontier"] - g["loss_beyond"]) < 1e-9
        ft = next(g for g in res["games"] if g["game"] == "ft09")
        assert ft["completed"] == 2 and ft["frontier"]["actions"] == 80 and ft["frontier"]["turns"] == 20
        assert abs(ft["time"][2] - (80 * 20.0 + 60.0)) < 1e-6                                 # frontier wallclock
        wa = next(g for g in res["games"] if g["game"] == "wa30")
        assert wa["baseline"] == BASELINES["wa30"] and wa["frontier"]["baseline"] == 119      # hidden baseline -> table
        W = res["what_if"]
        assert W["efficiency_par"] >= res["mean"]
        assert res["mean"] < W["plus_levels"][1]["obs"] < W["plus_levels"][2]["obs"] < W["plus_levels"][3]["obs"]
        n_par, n_obs = W["levels_needed"]["at_par"], W["levels_needed"]["at_new_level_score"]
        assert n_par is not None and (n_obs is None or n_obs >= n_par)
        sub = [dict(g) for g in res["games"]]
        cnt, reached, plan = levels_needed(sub, 30.0, 100.0)
        chk = sum(g["score"] for g in sub) / len(sub)
        for g in sub:
            n = g["n_levels"]
            chk += sum((i + 1) * 100.0 for i in range(g["completed"], g["completed"] + plan.get(g["game"], 0))) / (n * (n + 1) / 2) / len(sub)
        assert abs(chk - reached) < 1e-9 and reached >= 30.0 and sum(plan.values()) == cnt
        assert res["public_target"] == 30.0 / 0.48 and res["wm_signals"] and res["dominant"] in BUCKETS
        # events only (no benchmark.json / viewer data): same levels and causes except crashed/won bookkeeping
        os.remove(os.path.join(root, "benchmark.json"))
        for p in glob.glob(os.path.join(root, "artifacts", "*_viewer_data.json")):
            os.remove(p)
        res2 = analyse(load_run(root))
        got2 = {g["game"]: (g["completed"], g["cause"]) for g in res2["games"]}
        for g in res["games"]:
            if g["game"] not in ("cd82", "sb26"):
                assert got2[g["game"]] == (g["completed"], g["cause"]), (g["game"], got2[g["game"]])
        assert got2["cd82"][1] == "crashed" and got2["sb26"] == (8, None)
        with tempfile.TemporaryDirectory() as tmp4:                                           # kernel log only
            with open(os.path.join(tmp4, "kernel.log"), "w") as f:
                for g in res["games"]:
                    f.write("12.3s [finished] %s-abcd1234 state=%s level=%d/%d score=%.2f actions=%d tokens=0 per-level=%s\n"
                            % (g["game"], g["state"], g["completed"], g["n_levels"], g["score"], sum(g["actions"]),
                               ",".join("%d/%d" % ab for ab in zip(g["actions"], g["baseline"]))))
            res4 = analyse(load_run(tmp4))
            assert abs(res4["mean"] - res["mean"]) < 1e-9 and res4["levels_completed"] == res["levels_completed"]
            assert abs(res4["reported_mean"] - res["mean"]) < 0.01 and res4["events"] == 0
            assert {g["game"]: g["cause"] for g in res4["games"]}["wa30"] == "starved"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            report(res, "fixture")
            compare(res, res2, "full", "events")
        out = buf.getvalue()
        assert "VERDICT" in out and "goal_unfired" in out and "TO REACH 30" in out
        # a notes-only run has no wm text: the three wm causes must not appear
        with tempfile.TemporaryDirectory() as tmp2:
            root3, _, _ = _fixture(tmp2)
            for p in glob.glob(os.path.join(root3, "artifacts", "*_events.jsonl")):
                txt = open(p).read()
                for w in ("stage: ", "verdict: ", "acc: ", "found: ", "stopped: "):
                    txt = txt.replace("\\n" + w, "\\nx_" + w).replace("]\\n" + w, "]\\nx_" + w)
                open(p, "w").write(txt.replace("stage:", "st:").replace("verdict:", "vd:"))
            res3 = analyse(load_run(root3))
            assert not res3["wm_signals"] and not any(res3["layers"][c]["games"] for c in ("wm_drop", "wm_diverged", "goal_unfired"))
    # 3. structural table on the real public-25 baselines
    rows = {r["ratio"]: r for r in structural(BASELINES, 30.0)}
    assert rows[1.0]["levels_needed_greedy"] is not None and rows[2.0]["levels_needed_greedy"] is None
    assert rows[1.0]["levels_needed_greedy"] < rows[1.5]["levels_needed_greedy"]
    assert abs(rows[2.0]["level_score"] - 25.0) < 1e-9
    print("selftest ok: scorer %s; 10-game fixture (%s) classified as designed; loss identity exact; "
          "events-only and no-wm runs handled"
          % ("matches taaf GameRun._compute_final_score source on %d cases" % len(cases) if real else "formula only (bundle_fast not found)",
             "written with the harness's append_raw_events_sidecar" if real_writer else "plain jsonl"))


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "--selftest":
        selftest()
        sys.exit(0)
    ap = argparse.ArgumentParser()
    ap.add_argument("root", nargs="?")
    ap.add_argument("--label", default="")
    ap.add_argument("--target", type=float, default=30.0)
    ap.add_argument("--transfer", type=float, default=None, help="hidden LB / public-25 ratio, e.g. 0.48")
    ap.add_argument("--at", default="obs", help="score of a newly completed level: obs (median observed), par, or a number")
    ap.add_argument("--env-dir", default=None)
    ap.add_argument("--hud-cells", type=int, default=4)
    ap.add_argument("--json", default=None)
    ap.add_argument("--compare", default=None)
    ap.add_argument("--structural", action="store_true")
    ap.add_argument("--no-games", action="store_true")
    a = ap.parse_args()
    if a.structural or not a.root:
        report_structural(a.target)
        sys.exit(0)
    res = analyse(load_run(a.root, a.env_dir, a.hud_cells), a.target, a.transfer, a.at)
    report(res, a.label or os.path.basename(os.path.normpath(a.root)), not a.no_games)
    if a.compare:
        other = analyse(load_run(a.compare, a.env_dir, a.hud_cells), a.target, a.transfer, a.at)
        print("")
        compare(res, other, a.label or "A", os.path.basename(os.path.normpath(a.compare)))
    if a.json:
        slim = dict(res, games=[{k: v for k, v in g.items() if k not in ("levels", "wall")} for g in res["games"]])
        json.dump(slim, open(a.json, "w"), indent=1, default=str)
        print("wrote " + a.json)
