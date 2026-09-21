"""Engine utilities: load games directly from source, snapshot/restore, state keys."""
import copy, importlib.util, inspect, json
from pathlib import Path
from typing import Optional

import numpy as np
from arcengine import ActionInput, GameAction, GameState

BASE = Path("/home/takumu/kaggle/arc-agi-3/environment_files")


def game_ids():
    out = []
    for gdir in sorted(BASE.iterdir()):
        sub = next(gdir.iterdir())
        meta = json.loads((sub / "metadata.json").read_text())
        out.append((gdir.name, meta["game_id"], meta))
    return out


def load_game(prefix: str, seed: int = 0):
    sub = next((BASE / prefix).iterdir())
    meta = json.loads((sub / "metadata.json").read_text())
    game_id = meta["game_id"]
    class_name = game_id[:4][0].upper() + game_id[1:4]
    src = (sub / f"{prefix}.py").read_text()
    mod = importlib.util.module_from_spec(importlib.util.spec_from_loader("g", None))
    exec(src, mod.__dict__)
    cls = getattr(mod, class_name)
    kwargs = {"seed": seed} if "seed" in inspect.signature(cls).parameters else {}
    game = cls(**kwargs)
    return game, meta


def frame_of(game) -> np.ndarray:
    return np.asarray(game.camera.render(game.current_level.get_sprites()))


def state_key(game) -> bytes:
    """Hashable state key: rendered frame + level index + score."""
    f = frame_of(game)
    return f.tobytes() + bytes([game._current_level_index % 256, game._score % 256])


def full_state_key(game) -> bytes:
    """Richer key including sprite internals (positions, states, attrs)."""
    lvl = game.current_level
    parts = [f.tobytes() for f in [frame_of(game)]]
    for s in sorted(lvl.get_sprites(), key=lambda s: s.name):
        d = {k: v for k, v in vars(s).items() if isinstance(v, (int, float, str, bool, tuple))}
        parts.append(repr(sorted(d.items())).encode())
    parts.append(f"{game._current_level_index}|{game._score}".encode())
    return b"|".join(parts)


def valid_actions(game) -> list[ActionInput]:
    try:
        return game._get_valid_actions()
    except Exception:
        return [ActionInput(id=GameAction.from_id(a)) for a in game._available_actions]


def act(game, ai: ActionInput):
    return game.perform_action(ai, raw=True)


def reset(game):
    return game.perform_action(ActionInput(id=GameAction.RESET), raw=True)
