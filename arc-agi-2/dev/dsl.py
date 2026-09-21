"""ARC-AGI-2 DSL: grid primitives + objects + candidate solvers.

Design: `SOLVERS` is a list of functions f(train_pairs) -> Optional[callable g].
Each tries to find parameters such that g(input)==output for ALL train pairs.
The outer runner applies found g to test inputs.
"""
import numpy as np
from collections import Counter, deque
from itertools import product, permutations

Grid = np.ndarray

# ---------------------------------------------------------------- basics
def G(a): return np.asarray(a, dtype=int)
def rot(g, k): return np.rot90(g, k)
def fliph(g): return np.fliplr(g)
def flipv(g): return np.flipud(g)
def transp(g): return g.T
def colors(g): return set(np.unique(g))
def most_common(g, bg=None):
    c = Counter(g.ravel())
    if bg is not None and bg in c: del c[bg]
    return c.most_common(1)[0][0] if c else 0
def least_common(g):
    return Counter(g.ravel()).most_common()[-1][0]
def replace(g, a, b):
    out = g.copy(); out[g == a] = b; return out
def of_color(g, c): return np.argwhere(g == c)
def bbox_of(mask_or_cells):
    cells = np.argwhere(mask_or_cells) if mask_or_cells.dtype == bool else np.asarray(mask_or_cells)
    if len(cells) == 0: return None
    (y0, x0), (y1, x1) = cells.min(0), cells.max(0)
    return y0, x0, y1 + 1, x1 + 1
def crop(g, bb):
    y0, x0, y1, x1 = bb; return g[y0:y1, x0:x1]
def trim(g, bg=None):
    bg = most_common(g) if bg is None else bg
    bb = bbox_of(g != bg)
    return crop(g, bb) if bb else g
def tile_to(g, h, w):
    th, tw = g.shape
    out = np.tile(g, (int(np.ceil(h/th)), int(np.ceil(w/tw))))
    return out[:h, :w]
def upscale(g, k): return np.repeat(np.repeat(g, k, 0), k, 1)
def downscale(g, k):
    return g[::k, ::k]
def pad_to(g, h, w, c=0):
    out = np.full((h, w), c, g.dtype)
    h2, w2 = min(h, g.shape[0]), min(w, g.shape[1])
    out[:h2, :w2] = g[:h2, :w2]
    return out

# ---------------------------------------------------------------- objects
def components(g, bg=None, conn=4):
    """Connected components of non-bg cells -> list of (cells, color)."""
    bg = 0 if bg is None else bg
    H, W = g.shape
    seen = np.zeros_like(g, bool)
    out = []
    for y in range(H):
        for x in range(W):
            if seen[y, x] or g[y, x] == bg:
                continue
            c = g[y, x]
            q = deque([(y, x)]); seen[y, x] = True; cells = []
            while q:
                cy, cx = q.popleft(); cells.append((cy, cx))
                for dy, dx in ((1,0),(-1,0),(0,1),(0,-1)) + ((1,1),(1,-1),(-1,1),(-1,-1) if conn==8 else ()):
                    ny, nx = cy+dy, cx+dx
                    if 0<=ny<H and 0<=nx<W and not seen[ny,nx] and g[ny,nx]==c:
                        seen[ny,nx]=True; q.append((ny,nx))
            out.append((cells, int(c)))
    return out

def components_multicolor(g, bg=0, conn=4):
    """Components ignoring color."""
    H, W = g.shape
    seen = np.zeros_like(g, bool)
    out = []
    nb = ((1,0),(-1,0),(0,1),(0,-1)) + (((1,1),(1,-1),(-1,1),(-1,-1)) if conn==8 else ())
    for y in range(H):
        for x in range(W):
            if seen[y, x] or g[y, x] == bg: continue
            q = deque([(y,x)]); seen[y,x]=True; cells=[]
            while q:
                cy,cx=q.popleft(); cells.append((cy,cx))
                for dy,dx in nb:
                    ny,nx=cy+dy,cx+dx
                    if 0<=ny<H and 0<=nx<W and not seen[ny,nx] and g[ny,nx]!=bg:
                        seen[ny,nx]=True; q.append((ny,nx))
            out.append(cells)
    return out

def normalize(obj_cells):
    ys=[c[0] for c in obj_cells]; xs=[c[1] for c in obj_cells]
    y0,x0=min(ys),min(xs)
    return frozenset((y-y0,x-x0) for y,x in obj_cells)

def obj_grid(cells, color):
    ys=[c[0] for c in cells]; xs=[c[1] for c in cells]
    g=np.zeros((max(ys)+1-min(ys), max(xs)+1-min(xs)), int)
    for y,x in cells: g[y-min(ys),x-min(xs)]=color
    return g

def gravity(g, bg=0, dir='down'):
    out = np.full_like(g, bg)
    H, W = g.shape
    if dir in ('down','up'):
        for x in range(W):
            col = [g[y,x] for y in range(H) if g[y,x]!=bg]
            ys = range(H-len(col),H) if dir=='down' else range(len(col))
            for y,c in zip(ys,col): out[y,x]=c
    else:
        for y in range(H):
            row=[g[y,x] for x in range(W) if g[y,x]!=bg]
            xs = range(W-len(row),W) if dir=='right' else range(len(row))
            for x,c in zip(xs,row): out[y,x]=c
    return out

def fill_enclosed(g, bg=0, color=1):
    """Flood-fill background from borders; interior bg -> color."""
    out=g.copy(); H,W=g.shape
    seen=np.zeros_like(g,bool); q=deque()
    for y in range(H):
        for x in (0,W-1):
            if g[y,x]==bg and not seen[y,x]: seen[y,x]=True; q.append((y,x))
    for x in range(W):
        for y in (0,H-1):
            if g[y,x]==bg and not seen[y,x]: seen[y,x]=True; q.append((y,x))
    while q:
        y,x=q.popleft()
        for dy,dx in DIRS:
            ny,nx=y+dy,x+dx
            if 0<=ny<H and 0<=nx<W and not seen[ny,nx] and g[ny,nx]==bg:
                seen[ny,nx]=True; q.append((ny,nx))
    out[(g==bg)&(~seen)]=color
    return out
DIRS=((-1,0),(1,0),(0,-1),(0,1))
