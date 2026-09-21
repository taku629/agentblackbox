"""Improved solver: novelty-prioritized beam search keyed on hidden+frame state."""
import copy, os, sys, time, random, json
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from engine import load_game, reset, act, frame_of, valid_actions
from arcengine import ActionInput, GameAction, GameState


def key_of(game, fr=None):
    try:
        h = np.asarray(game._get_hidden_state()).tobytes()
    except Exception:
        h = b""
    if fr is None:
        f = frame_of(game)
    else:
        f = np.asarray(fr.frame[-1])
    return h + f.tobytes() + bytes([game._current_level_index % 256, game._score % 256])


def solve_level(game, seed=0, max_nodes=60000, beam_width=48,
                time_budget=300.0, max_depth=600, mode="bfs"):
    """Novelty-prioritized beam search. Returns (path, game) or (None, game)."""
    t0 = time.time()
    rng = random.Random(seed)
    start_score = game._score
    start_level = game._current_level_index

    root_key = key_of(game)
    visit = {root_key: 1}
    frontier = [(0.0, rng.random(), game, [])]  # (neg novelty prio, tiebreak, game, path)
    nodes = 0

    while frontier and nodes < max_nodes and time.time() - t0 < time_budget:
        frontier.sort(key=lambda t: (t[0], t[1]))
        batch = frontier[:beam_width]
        frontier = frontier[beam_width:]
        if not batch:
            break
        for _, _, ngame, path in batch:
            if len(path) >= max_depth:
                continue
            vas = valid_actions(ngame)
            simple = [a for a in vas if a.id != GameAction.ACTION6]
            clicks = [a for a in vas if a.id == GameAction.ACTION6]
            if len(clicks) > 32:
                clicks = rng.sample(clicks, 32)
            for ai in simple + clicks:
                nodes += 1
                g2 = copy.deepcopy(ngame)
                fr = act(g2, ai)
                if fr is None or g2._state == GameState.GAME_OVER:
                    continue
                newpath = path + [ai]
                if g2._score > start_score or g2._current_level_index != start_level \
                   or g2._state == GameState.WIN:
                    return newpath, g2
                k = key_of(g2, fr)
                cnt = visit.get(k, 0)
                if cnt >= 3:
                    continue
                visit[k] = cnt + 1
                # BFS-first (shortest solutions) with novelty tiebreak
                prio = len(newpath) * 10 + cnt
                frontier.append((prio, rng.random(), g2, newpath))
            # memory safety: cap live frontier
            if len(frontier) > 6000:
                frontier.sort(key=lambda t: (t[0], t[1]))
                frontier = frontier[:6000]
    return None, game


def solve_game(prefix, seed=0, max_level_nodes=60000, time_per_level=240,
               max_levels=None, out_dir="dev/solutions", verbose=True):
    game, meta = load_game(prefix, seed=seed)
    nlevels = len(game._levels)
    if max_levels:
        nlevels = min(nlevels, max_levels)
    baselines = meta.get("baseline_actions", [])
    reset(game)
    solutions = []
    for li in range(nlevels):
        if game._current_level_index != li:
            break
        path, game2 = solve_level(game, seed=seed, max_nodes=max_level_nodes,
                                  time_budget=time_per_level)
        if path is None:
            if verbose:
                print(f"  {prefix} L{li}: FAILED")
            break
        bl = baselines[li] if li < len(baselines) else "?"
        if verbose:
            print(f"  {prefix} L{li}: {len(path)} actions (baseline {bl})")
        solutions.append([(a.id.value, dict(a.data)) for a in path])
        game = game2
        if game._state == GameState.WIN:
            break
    os.makedirs(out_dir, exist_ok=True)
    _j = json
    _j.dump({"game_id": meta["game_id"], "solutions": solutions},
            open(os.path.join(out_dir, f"{prefix}.json"), "w"))
    return solutions


if __name__ == "__main__":
    prefix = sys.argv[1]
    budget = float(sys.argv[2]) if len(sys.argv) > 2 else 240
    solve_game(prefix, time_per_level=budget)
