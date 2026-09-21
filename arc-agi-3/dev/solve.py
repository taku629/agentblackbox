"""Generic beam-search solver for ARC-AGI-3 levels (offline, uses real engine)."""
import copy, heapq, os, sys, time
from collections import defaultdict
from typing import Optional

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from engine import (BASE, act, frame_of, load_game, reset, state_key,
                    valid_actions)
from arcengine import ActionInput, GameAction, GameState


def hidden_key(game) -> bytes:
    """State key from hidden_state + score + level (compact, collision-light)."""
    try:
        h = np.asarray(game._get_hidden_state()).tobytes()
    except Exception:
        h = b""
    return h + frame_of(game).tobytes() + bytes([game._current_level_index % 256])


class Node:
    __slots__ = ("game", "path", "key", "prio")

    def __init__(self, game, path, key, prio=0.0):
        self.game = game
        self.path = path
        self.key = key
        self.prio = prio


def clone(g):
    return copy.deepcopy(g)


def solve_level(game, max_nodes=30000, beam_width=64, max_depth=None,
                time_budget=120.0, verbose=False):
    """Beam search from current state until _score increments or WIN.
    Returns (action_path, solved_game) or (None, game)."""
    t0 = time.time()
    start_score = game._score
    start_level = game._current_level_index
    if max_depth is None:
        max_depth = 400

    root = Node(game, [], hidden_key(game))
    seen = {root.key}
    frontier = [root]
    nodes = 0

    while frontier and nodes < max_nodes and time.time() - t0 < time_budget:
        # pick best nodes to expand (by novelty-first, then shortest path)
        frontier.sort(key=lambda n: (n.prio, len(n.path)))
        batch = frontier[:beam_width]
        frontier = frontier[beam_width:]
        for node in batch:
            if len(node.path) >= max_depth:
                continue
            actions = valid_actions(node.game)
            for ai in actions:
                nodes += 1
                g2 = clone(node.game)
                fr = act(g2, ai)
                if fr is None:
                    continue
                if g2._state == GameState.GAME_OVER:
                    continue
                newpath = node.path + [ai]
                if g2._score > start_score or g2._state == GameState.WIN:
                    return newpath, g2
                if g2._current_level_index != start_level:
                    # level advanced without score? still a win condition
                    return newpath, g2
                k = hidden_key(g2)
                if k in seen:
                    continue
                seen.add(k)
                # novelty priority: newer distinct states first
                g2copy = g2
                frontier.append(Node(g2copy, newpath, k, prio=0))
            if nodes % 2000 == 0 and verbose:
                print(f"  nodes={nodes} seen={len(seen)} frontier={len(frontier)} depth~{len(node.path)}")
            if time.time() - t0 > time_budget:
                break
    return None, game


def solve_game(prefix, seeds=(0,), max_level_nodes=20000, time_per_level=90,
               max_levels=None, out_dir="solutions", verbose=True):
    """Solve as many levels as possible for a game. Returns list of per-level paths."""
    game, meta = load_game(prefix)
    nlevels = len(game._levels)
    if max_levels:
        nlevels = min(nlevels, max_levels)
    baselines = meta.get("baseline_actions", [])
    print(f"=== {prefix}: {nlevels} levels, baselines {baselines[:nlevels]}")

    reset(game)
    solutions = []
    for li in range(nlevels):
        if game._current_level_index != li:
            break
        path, game2 = solve_level(game, max_nodes=max_level_nodes,
                                  time_budget=time_per_level, verbose=verbose)
        if path is None:
            print(f"  L{li}: FAILED")
            break
        bl = baselines[li] if li < len(baselines) else "?"
        print(f"  L{li}: solved in {len(path)} actions (baseline {bl})")
        solutions.append([(a.id.value, a.data) for a in path])
        game = game2
        if game._state == GameState.WIN:
            break
    os.makedirs(out_dir, exist_ok=True)
    out = os.path.join(out_dir, f"{prefix}.json")
    import json as _j
    _j.dump({"game_id": meta["game_id"], "solutions": solutions}, open(out, "w"))
    return solutions


if __name__ == "__main__":
    prefix = sys.argv[1] if len(sys.argv) > 1 else "cd82"
    solve_game(prefix)
