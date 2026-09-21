"""Verify saved solutions by sequential replay on a fresh game instance."""
import sys, os, json, glob
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'starter'))
os.chdir(os.path.join(os.path.dirname(__file__), '..', 'starter'))
os.environ.setdefault("ONLY_RESET_LEVELS", "true")

import importlib.util
import numpy as np
from arcengine import ActionInput, GameAction

def load_game(game_id):
    hits = glob.glob(f'environment_files/{game_id}/*/{game_id}.py')
    if not hits:
        return None
    spec = importlib.util.spec_from_file_location('_g', hits[0])
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    cls_name = game_id[0].upper() + game_id[1:]
    cls = getattr(m, cls_name, None)
    if cls is None:
        for n in dir(m):
            if n.lower() == game_id:
                cls = getattr(m, n)
    return cls() if cls else None

def mk(aid, data):
    ga = GameAction.from_id(aid)
    ai = ActionInput(id=ga, data={})
    if data:
        ai.data = dict(data)
    return ai

results = {}
for f in sorted(glob.glob('../dev/solutions/*.json')):
    gid = os.path.basename(f)[:-5]
    blob = json.load(open(f))
    sols = blob.get('solutions', [])
    g = load_game(gid)
    if g is None:
        print(f"{gid}: no source"); continue
    done = 0
    ok = True
    for li, sol in enumerate(sols):
        if not sol:
            ok = False; break
        prev_lvl = g._current_level_index
        advanced = False
        for aid, data in sol:
            try:
                r = g.perform_action(mk(aid, data), raw=True)
            except Exception:
                break
            if g._current_level_index > prev_lvl or \
               getattr(r, 'levels_completed', 0) > prev_lvl:
                advanced = True
                break
            if getattr(r, 'state', None) is not None and \
               str(getattr(r, 'state')) == 'GameState.WIN':
                advanced = True
                break
        if advanced:
            done += 1
        else:
            ok = False
            print(f"{gid} L{li}: replay FAILED")
            break
    results[gid] = done
    print(f"{gid}: {done}/{len(sols)} levels verified")

print(json.dumps(results))
