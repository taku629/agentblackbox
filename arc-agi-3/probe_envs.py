import json, sys, logging
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent/"ARC-AGI-3-Agents"))
from arc_agi.local_wrapper import LocalEnvironmentWrapper
from arc_agi.models import EnvironmentInfo
from arcengine import GameAction
logging.disable(logging.CRITICAL)
import numpy as np

BASE = Path(__file__).resolve().parent/"environment_files"
for gdir in sorted(BASE.iterdir()):
    sub = next(gdir.iterdir())
    meta = json.loads((sub / "metadata.json").read_text())
    info = EnvironmentInfo(**meta)
    info.local_dir = str(sub)
    try:
        env = LocalEnvironmentWrapper(info, logging.getLogger("t"), scorecard_id="x", seed=0)
        raw = env.step(GameAction.RESET)
        f0 = np.asarray(raw.frame[0])
        ncolors = len(np.unique(f0))
        diffs = {}
        for aid in raw.available_actions:
            r = env.step(GameAction.from_id(aid))
            if r.frame:
                f1 = np.asarray(r.frame[0])
                diffs[aid] = int((f0 != f1).sum())
        print(f"{gdir.name:6} win={raw.win_levels:3} avail={raw.available_actions} "
              f"colors={ncolors:2} base={meta.get('baseline_actions')} diffs={diffs} tags={meta.get('tags')}")
    except Exception as e:
        print(f"{gdir.name:6} ERROR {type(e).__name__} {e}")
