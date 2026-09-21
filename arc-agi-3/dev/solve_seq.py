"""Sequential-level solver coverage harness.

Drives MultiSolver directly (no live agent) to measure how many levels the
solver can crack per game, using the new prefix-replay reach logic.
Usage: python dev/solve_seq.py <game_id> [<game_id> ...]
"""
import sys, os, time, json, logging

_HERE = os.path.dirname(os.path.abspath(__file__))
_STARTER = os.path.join(_HERE, '..', 'starter')
sys.path.insert(0, os.path.abspath(_STARTER))
sys.path.insert(0, os.path.abspath(os.path.join(_STARTER, 'vendor', 'ARC-AGI-3-Agents')))
os.chdir(os.path.abspath(_STARTER))  # environment_files glob is cwd-relative
os.environ.setdefault("ONLY_RESET_LEVELS", "true")

logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)

import agent.my_agent as MA


def solve_game(game_id: str, per_level_budget: float = 300.0, max_levels: int = 10):
    src, cls_name = MA.find_game_source(game_id, 'environment_files')
    if not src:
        print(f"{game_id}: NO SOURCE")
        return {}
    solver = MA.MultiSolver(src, cls_name, MA._BUDGET)
    solver.load()
    if not solver.game_cls:
        print(f"{game_id}: LOAD FAILED")
        return {}
    res = {}
    for li in range(max_levels):
        t0 = time.time()
        sol = solver.solve_level(li, total_budget=per_level_budget)
        if sol is None:
            print(f"{game_id} L{li}: FAIL ({time.time()-t0:.0f}s)")
            # can't reach deeper levels without this prefix
            break
        res[li] = len(sol)
        print(f"{game_id} L{li}: {len(sol)} steps ({time.time()-t0:.0f}s)")
        # persist the raw action path for embedding / replay validation
        out_dir = os.path.join(_HERE, 'solutions_seq')
        os.makedirs(out_dir, exist_ok=True)
        fp = os.path.join(out_dir, f'{game_id}.json')
        data = {}
        if os.path.exists(fp):
            data = json.load(open(fp))
        data[str(li)] = sol
        json.dump(data, open(fp, 'w'))
    return res


if __name__ == '__main__':
    games = sys.argv[1:]
    out = {}
    for gid in games:
        try:
            out[gid] = solve_game(gid)
        except Exception as e:
            print(f"{gid}: ERROR {e}")
            out[gid] = {}
    print("=== RESULT ===")
    print(json.dumps({k: v for k, v in out.items()}))
