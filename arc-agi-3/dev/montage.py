"""Render initial frame of every game into a montage dir."""
import os, sys
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from engine import game_ids, load_game, reset
from viz import to_img, save_png

outdir = "/tmp/arc3_viz"
os.makedirs(outdir, exist_ok=True)
for prefix, gid, meta in game_ids():
    try:
        g, _ = load_game(prefix)
        fr = reset(g)
        img = to_img(np.asarray(fr.frame[-1]), scale=4)
        save_png(img, f"{outdir}/init_{prefix}.png")
        print(prefix, "avail:", fr.available_actions, "win:", fr.win_levels)
    except Exception as e:
        print(prefix, "ERR", e)
