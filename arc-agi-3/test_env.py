import json, logging, sys
from pathlib import Path

from arc_agi.local_wrapper import LocalEnvironmentWrapper
from arc_agi.models import EnvironmentInfo
from arcengine import GameAction

logging.basicConfig(level=logging.WARNING)

BASE = Path(__file__).resolve().parent/"environment_files"

def load_env(game_prefix: str, seed: int = 0):
    gdir = BASE / game_prefix
    sub = next(gdir.iterdir())
    meta = json.loads((sub / "metadata.json").read_text())
    info = EnvironmentInfo(**meta)
    info.local_dir = str(sub)
    env = LocalEnvironmentWrapper(info, logging.getLogger("t"), scorecard_id="test", seed=seed)
    return env

env = load_env("ls20")
print("obs space:", env.observation_space)
raw = env.step(GameAction.RESET)
print("state:", raw.state, "levels:", raw.levels_completed, "/", raw.win_levels)
print("avail:", raw.available_actions)
for f in raw.frame:
    print("frame shape:", f.shape)
import numpy as np
np.set_printoptions(linewidth=200, threshold=10000)
print(np.asarray(raw.frame[0]))
