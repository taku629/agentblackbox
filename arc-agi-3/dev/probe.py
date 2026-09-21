"""Probe: load a game directly, check deepcopy + valid actions + step speed."""
import copy, importlib.util, json, sys, time
from pathlib import Path

sys.path.insert(0, "/home/takumu/kaggle/arc-agi-3/.venv312/lib/python3.12/site-packages")

from arcengine import ActionInput, GameAction

BASE = Path("/home/takumu/kaggle/arc-agi-3/environment_files")

def load_game(prefix: str, seed: int = 0):
    sub = next((BASE / prefix).iterdir())
    meta = json.loads((sub / "metadata.json").read_text())
    game_id = meta["game_id"]
    class_name = game_id[:4][0].upper() + game_id[1:4]
    src = (sub / f"{prefix}.py").read_text()
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("g", None))
    exec(src, mod.__dict__)
    cls = getattr(mod, class_name)
    import inspect
    kwargs = {"seed": seed} if "seed" in inspect.signature(cls).parameters else {}
    return cls(**kwargs)

g = load_game("ls20")
print("levels:", len(g._levels), "win:", g._win_score, "avail:", g._available_actions)

# step speed
t0 = time.time()
from arcengine import ActionInput
for _ in range(20):
    g.perform_action(ActionInput(id=GameAction.ACTION1), raw=True)
print("20 actions in", time.time() - t0, "s")

# deepcopy
t0 = time.time()
g2 = copy.deepcopy(g)
print("deepcopy ok", time.time() - t0)

# valid actions
va = g._get_valid_actions()
print("valid actions:", len(va), [a.id for a in va][:10])

# state key from sprites
def state_key(game):
    lvl = game.current_level
    parts = []
    for s in lvl.get_sprites():
        parts.append((s.name, s.x, s.y))
    return (game._current_level_index, game._score, tuple(sorted(parts)))

print("key sample:", str(state_key(g))[:200])
import numpy as np
frame = g.camera.render(g.current_level.get_sprites())
print("frame shape:", frame.shape, "unique colors:", np.unique(frame))
