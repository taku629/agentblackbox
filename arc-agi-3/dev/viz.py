"""Render a game's initial frames to PNG for quick visual understanding."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from engine import load_game, act, reset
from arcengine import ActionInput, GameAction

# ARC palette (16 colors)
PAL = np.array([
    [0,0,0],[0,116,217],[255,65,54],[46,204,64],[181,204,24],
    [127,219,255],[255,220,0],[255,133,27],[135,59,255],[255,45,117],
    [160,160,160],[135,216,241],[128,0,0],[0,80,80],[220,220,220],[50,50,50]
], dtype=np.uint8)

def to_img(frame, scale=6):
    f = np.asarray(frame)
    f = np.clip(f, 0, 15)
    img = PAL[f]
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)

def save_png(img, path):
    # minimal PNG via zlib
    import zlib, struct
    h, w, _ = img.shape
    def chunk(t, d):
        c = t + d
        return struct.pack(">I", len(d)) + c + struct.pack(">I", zlib.crc32(c))
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    png = (b"\x89PNG\r\n\x1a\n"
           + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(raw))
           + chunk(b"IEND", b""))
    open(path, "wb").write(png)

def main(prefix="ls20", n_actions=0, outdir="/tmp/arc3_viz"):
    os.makedirs(outdir, exist_ok=True)
    game, meta = load_game(prefix)
    fr = reset(game)
    save_png(to_img(np.asarray(fr.frame[-1])), f"{outdir}/{prefix}_L0.png")
    # take n random-ish actions and snapshot
    import random; random.seed(1)
    for i in range(n_actions):
        va = game._available_actions
        a = random.choice([x for x in va if x != 0])
        ai = ActionInput(id=GameAction.from_id(a))
        if a == 6:
            ai.data = {"x": random.randint(0,63), "y": random.randint(0,63)}
        fr = act(game, ai)
    save_png(to_img(np.asarray(fr.frame[-1])), f"{outdir}/{prefix}_after{n_actions}.png")
    print("state:", fr.state, "levels:", fr.levels_completed, "/", fr.win_levels, "avail:", fr.available_actions)

if __name__ == "__main__":
    main(*sys.argv[1:2], n_actions=int(sys.argv[2]) if len(sys.argv)>2 else 0)
