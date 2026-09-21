"""Fast local runner for OceanCore against LocalEnvironmentWrapper.

Usage: .venv312/bin/python run_local.py ls20 [max_actions] [seed]
       .venv312/bin/python run_local.py all 1500
"""
import json, sys, logging, importlib
from pathlib import Path
sys.path.insert(0, "/home/takumu/kaggle/arc-agi-3/ARC-AGI-3-Agents")
sys.path.insert(0, "/home/takumu/kaggle/arc-agi-3")
from arc_agi.local_wrapper import LocalEnvironmentWrapper
from arc_agi.models import EnvironmentInfo
from arcengine import GameAction
logging.disable(logging.CRITICAL)
import numpy as np

import agent_core
importlib.reload(agent_core)
from agent_core import OceanCore

BASE = Path("/home/takumu/kaggle/arc-agi-3/environment_files")


def load(game, seed=0):
    sub = next((BASE/game).iterdir())
    meta = json.loads((sub/"metadata.json").read_text())
    info = EnvironmentInfo(**meta); info.local_dir = str(sub)
    env = LocalEnvironmentWrapper(info, logging.getLogger("t"), scorecard_id="x", seed=seed)
    return env, meta


def run_game(game, max_actions=1500, seed=0, verbose=False):
    env, meta = load(game, seed)
    core = OceanCore(seed=seed)
    baselines = meta.get("baseline_actions", [])
    raw = env.step(GameAction.RESET)
    if raw is None or not raw.frame:
        return {"game": game, "error": "reset failed"}
    level_actions = 0
    level_scores = []
    per_level_actions = []
    prev_level = raw.levels_completed
    total_actions = 0
    avail = raw.available_actions
    while total_actions < max_actions:
        grid = np.asarray(raw.frame[-1])
        aid, data = core.decide(grid, raw.available_actions, raw.state.value if hasattr(raw.state,'value') else str(raw.state),
                              raw.levels_completed, total_actions)
        act = GameAction.from_id(aid)
        raw = env.step(act, data if data else None)
        total_actions += 1
        if raw is None or not raw.frame:
            # invalid action or dead-end; keep going
            raw = env.step(GameAction.RESET)
            continue
        if raw.state.name == "NOT_FINISHED":
            level_actions += 1
        if raw.levels_completed > prev_level:
            per_level_actions.append(level_actions)
            b = baselines[prev_level] if prev_level < len(baselines) else 30
            sc = min((b / max(level_actions,1))**2 * 100, 115.0)
            level_scores.append(sc)
            if verbose:
                print(f"  [{game}] level {prev_level} done in {level_actions} (base {b}) -> {sc:.1f}")
            level_actions = 0
            prev_level = raw.levels_completed
        if raw.state.name == "WIN":
            break
    return {"game": game, "levels": prev_level, "win_levels": raw.win_levels,
            "level_scores": level_scores, "per_level_actions": per_level_actions,
            "actions": total_actions, "state": raw.state.name}


def game_score(r, baselines):
    """Weighted-by-index average of level scores (levels not completed = 0)."""
    n = r["win_levels"]
    if n == 0: return 0.0
    scores = r["level_scores"] + [0.0]*(n - len(r["level_scores"]))
    num = sum((i+1)*s for i,s in enumerate(scores[:n]))
    den = sum(range(1, n+1))
    maxw = sum((i+1) for i,s in enumerate(scores[:n]) if s>0)
    return min(num/den, maxw/den*100)


if __name__ == "__main__":
    args = sys.argv[1:]
    games = args[0].split(",") if args else ["ls20"]
    max_actions = int(args[1]) if len(args)>1 else 1500
    verbose = "-v" in args
    if games == ["all"]:
        games = sorted(p.name for p in BASE.iterdir())
    total = 0.0
    for g in games:
        r = run_game(g, max_actions, verbose=verbose)
        if "error" in r:
            print(f"{g:6} ERROR {r['error']}"); continue
        _, meta = load(g)
        gs = game_score(r, meta.get("baseline_actions",[]))
        total += gs
        print(f"{g:6} levels={r['levels']}/{r['win_levels']} actions={r['actions']:5} "
              f"level_actions={r['per_level_actions']} score={gs:.2f} state={r['state']}")
    print(f"\nMEAN game score over {len(games)}: {total/len(games):.2f}%")
