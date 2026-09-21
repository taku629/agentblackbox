"""Greedy novelty-search player: pick action leading to least-seen state."""
import os, sys, time, random
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from engine import game_ids, load_game, act, reset, frame_of, valid_actions
from arcengine import ActionInput, GameAction, GameState


def play_game(prefix, action_budget=3000, seed=0, verbose=False):
    game, meta = load_game(prefix, seed=seed)
    rng = random.Random(seed)
    fr = reset(game)
    seen = {}
    solved_levels = 0
    actions_used = 0
    per_level_actions = []
    last_level_count = 0
    level_actions = 0

    while actions_used < action_budget and fr.state not in (GameState.WIN,):
        if fr.state == GameState.GAME_OVER:
            fr = reset(game); actions_used += 1
            continue
        # candidate actions
        vas = valid_actions(game)
        # keep branching manageable: all simple actions + sampled clicks
        simple = [ai for ai in vas if ai.id != GameAction.ACTION6]
        clicks = [ai for ai in vas if ai.id == GameAction.ACTION6]
        if len(clicks) > 24:
            clicks = rng.sample(clicks, 24)
        cands = simple + clicks
        # evaluate each candidate by novelty of resulting key
        best = None; best_score = 1e18; results = []
        for ai in cands:
            import copy
            g2 = copy.deepcopy(game)
            r = act(g2, ai)
            if r is None:
                continue
            k = np.asarray(r.frame[-1]).tobytes() + bytes([r.levels_completed])
            v = seen.get(k, 0)
            results.append((v, ai, r, g2))
            if v < best_score:
                best_score = v; best = (ai, r, g2)
        if not results:
            break
        # softmax-ish: prefer least seen, with some randomness among ties
        minv = min(v for v,_,_,_ in results)
        ties = [x for x in results if x[0] <= minv]
        _, ai, r, g2 = rng.choice(ties)
        k = np.asarray(r.frame[-1]).tobytes() + bytes([r.levels_completed])
        seen[k] = seen.get(k, 0) + 1
        game = g2
        fr = r
        actions_used += 1
        level_actions += 1
        if fr.levels_completed > last_level_count:
            per_level_actions.append(level_actions)
            if verbose:
                print(f"  level {fr.levels_completed} done in {level_actions} (total {actions_used})")
            level_actions = 0
            last_level_count = fr.levels_completed
            seen.clear()

    return {
        "game": prefix,
        "levels_done": fr.levels_completed,
        "win_levels": fr.win_levels,
        "actions": actions_used,
        "state": str(fr.state),
        "per_level": per_level_actions,
    }


if __name__ == "__main__":
    games = sys.argv[1:] or [g[0] for g in game_ids()]
    for prefix in games:
        t0 = time.time()
        res = play_game(prefix, action_budget=2000, verbose=True)
        print(f"{prefix}: {res['levels_done']}/{res['win_levels']} levels, "
              f"{res['actions']} acts, {time.time()-t0:.0f}s, per_level={res['per_level']}")
