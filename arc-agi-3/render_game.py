import json, sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/"ARC-AGI-3-Agents"))
from arc_agi.local_wrapper import LocalEnvironmentWrapper
from arc_agi.models import EnvironmentInfo
from arcengine import GameAction
logging.disable(logging.CRITICAL)
import numpy as np

# ARC palette (16 colors)
PALETTE = np.array([
    [0,0,0],[255,255,255],[30,147,255],[250,61,49],[79,204,48],[255,220,0],
    [153,153,153],[229,58,163],[255,133,27],[135,212,241],[146,18,49],[255,156,182],
    [80,80,80],[166,101,41],[3,161,99],[199,168,11]
], dtype=np.uint8)

BASE = Path(__file__).resolve().parent/"environment_files"

def load(game, seed=0):
    sub = next((BASE/game).iterdir())
    meta = json.loads((sub/"metadata.json").read_text())
    info = EnvironmentInfo(**meta); info.local_dir = str(sub)
    env = LocalEnvironmentWrapper(info, logging.getLogger("t"), scorecard_id="x", seed=seed)
    return env

def save_png(arr, path):
    from PIL import Image
    img = PALETTE[np.clip(arr,0,15)]
    Image.fromarray(img).resize((256,256), Image.NEAREST).save(path)

if __name__ == "__main__":
    import os
    os.makedirs(Path(__file__).resolve().parent/"frames", exist_ok=True)
    game = sys.argv[1]
    actions = [GameAction.from_id(int(x)) for x in sys.argv[2].split(",")] if len(sys.argv)>2 else [GameAction.RESET]
    env = load(game)
    raw = env.step(GameAction.RESET)
    save_png(np.asarray(raw.frame[0]), f"{Path(__file__).resolve().parent}/frames/{game}_0.png")
    for i,a in enumerate(actions,1):
        raw = env.step(a)
        if raw and raw.frame:
            save_png(np.asarray(raw.frame[0]), f"{Path(__file__).resolve().parent}/frames/{game}_{i}.png")
            print(i, a.name, raw.state, "levels:", raw.levels_completed)
