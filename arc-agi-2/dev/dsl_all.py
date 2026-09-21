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

"""Solver battery for ARC-AGI-2. Each solver takes list of (in,out) pairs and
returns a callable g(grid)->grid valid on all pairs, or None."""

def ok_all(fn, pairs):
    try:
        for a, b in pairs:
            r = fn(G(a))
            if r is None or not np.array_equal(np.asarray(r), np.asarray(b)):
                return False
        return True
    except Exception:
        return False

def wrap(fn):
    """Make a solver-fn into (pairs)->callable or None."""
    def s(pairs):
        f = fn(pairs)
        return f
    return s

# ---------------- individual hypothesis families ----------------

def s_geom(pairs):
    for t in [lambda g: rot(g,1), lambda g: rot(g,2), lambda g: rot(g,3),
              fliph, flipv, transp, lambda g: np.fliplr(transp(g)),
              lambda g: transp(fliph(g)), lambda g: rot(transp(g),1)]:
        if ok_all(t, pairs):
            return t
    return None

def s_identity_plus(pairs):
    cands = []
    # color permutation learned from first pair
    def learn_map(a, b):
        if a.shape != b.shape: return None
        m = {}
        for c in np.unique(a):
            tgt = b[a == c]
            if len(set(np.unique(tgt))) == 1:
                m[int(c)] = int(tgt.flat[0])
            else:
                return None
        return m
    m = learn_map(G(pairs[0][0]), G(pairs[0][1]))
    if m is not None:
        def f(g, m=m):
            out = g.copy()
            for k, v in m.items(): out[g == k] = v
            return out
        if ok_all(f, pairs): return f
    # replace each color with most-common-of-output
    return None

def s_recolor_const(pairs):
    """All cells of color a -> b (single substitution)."""
    a0, b0 = G(pairs[0][0]), G(pairs[0][1])
    if a0.shape != b0.shape: return None
    diff = a0 != b0
    if not diff.any(): return None
    cs = np.unique(a0[diff]); ts = np.unique(b0[diff])
    for c in cs:
        for t in set(np.unique(b0)) | set(np.unique(a0)):
            f = lambda g, c=c, t=t: replace(g, c, t)
            if ok_all(f, pairs): return f
    return None

def s_trim(pairs):
    for bgname, bg in [("auto", None)] + [(f"c{c}", c) for c in range(10)]:
        def f(g, bg=bg):
            bb = bg if bg is not None else most_common(g)
            return trim(g, bb)
        if ok_all(f, pairs): return f
    return None

def s_crop_color_bbox(pairs):
    for c in range(10):
        def f(g, c=c):
            m = g == c
            if not m.any(): return None
            return crop(g, bbox_of(m))
        if ok_all(f, pairs): return f
    return None

def s_upscale(pairs):
    a0, b0 = G(pairs[0][0]), G(pairs[0][1])
    if a0.size and b0.shape[0] % a0.shape[0] == 0 and b0.shape[1] % a0.shape[1] == 0:
        k = b0.shape[0] // a0.shape[0]
        if k > 1 and b0.shape[1] // a0.shape[1] == k:
            for mode in range(3):
                if mode == 0:
                    f = lambda g, k=k: upscale(g, k)
                elif mode == 1:  # pixel -> block of itself, but in pattern of input
                    def f(g, k=k):
                        h, w = g.shape
                        out = np.zeros((h*k, w*k), int)
                        for y in range(h):
                            for x in range(w):
                                out[y*k:(y+1)*k, x*k:(x+1)*k] = g[y, x]
                        return out
                else:  # each pixel becomes scaled input masked by color
                    def f(g, k=k):
                        h, w = g.shape
                        base = upscale(g, k)
                        return base
                if ok_all(f, pairs): return f
    return None

def mirror_tile(g, oh, ow, flip_v=True, flip_h=True):
    th, tw = g.shape
    out = np.zeros((oh, ow), g.dtype)
    for by in range(int(np.ceil(oh/th))):
        for bx in range(int(np.ceil(ow/tw))):
            blk = g
            if flip_v and by % 2: blk = np.flipud(blk)
            if flip_h and bx % 2: blk = np.fliplr(blk)
            y0, x0 = by*th, bx*tw
            blk = blk[:min(th, oh-y0), :min(tw, ow-x0)]
            out[y0:y0+blk.shape[0], x0:x0+blk.shape[1]] = blk
    return out

def s_tile(pairs):
    a0, b0 = G(pairs[0][0]), G(pairs[0][1])
    if b0.shape[0] % a0.shape[0] == 0 and b0.shape[1] % a0.shape[1] == 0:
        f = lambda g: tile_to(g, *G(pairs[0][1]).shape)
        if ok_all(f, pairs): return f
    # tile to double/triple
    for k in (2, 3, 4):
        f = lambda g, k=k: tile_to(g, g.shape[0]*k, g.shape[1]*k)
        if ok_all(f, pairs): return f
    # mirror tiling (alternating flips)
    for fv, fh in [(True, False), (False, True), (True, True)]:
        def f(g, fv=fv, fh=fh):
            a = G(pairs[0][0]); b = G(pairs[0][1])
            ky = b.shape[0] // a.shape[0]; kx = b.shape[1] // a.shape[1]
            return mirror_tile(g, g.shape[0]*ky, g.shape[1]*kx, fv, fh)
        if ok_all(f, pairs): return f
    return None

def s_kron_self(pairs):
    """Each nonzero input pixel -> copy of input (kron(mask, input) variants)."""
    def f1(g):
        m = (g != 0).astype(int)
        return np.kron(m, g)
    def f2(g):
        m = (g != 0).astype(int)
        return np.kron(g, np.ones_like(g)) * 0  # placeholder
    def f3(g):
        h, w = g.shape
        out = np.zeros((h*h, w*w), g.dtype)
        for y in range(h):
            for x in range(w):
                if g[y, x] != 0:
                    out[y*h:(y+1)*h, x*w:(x+1)*w] = np.where(g != 0, g[y, x], 0)
        return out
    for f in [f1, f3]:
        if ok_all(f, pairs): return f
    return None

def s_symmetry(pairs):
    """Complete symmetry along h/v/rot axes."""
    def sym_h(g):
        h, w = g.shape
        out = g.copy()
        for y in range(h):
            for x in range(w):
                if out[y, x] == 0:
                    out[y, x] = g[y, w-1-x]
        return out
    def sym_v(g):
        h, w = g.shape
        out = g.copy()
        for y in range(h):
            for x in range(w):
                if out[y, x] == 0:
                    out[y, x] = g[h-1-y, x]
        return out
    def sym_d(g):
        h, w = g.shape
        out = g.copy()
        for y in range(h):
            for x in range(w):
                if out[y, x] == 0:
                    out[y, x] = g[h-1-y, w-1-x]
        return out
    for f in [sym_h, sym_v, sym_d,
              lambda g: np.maximum(g, np.fliplr(g)),
              lambda g: np.maximum(g, np.flipud(g)),
              lambda g: np.maximum(g, np.rot90(g, 2)),
              lambda g: np.maximum.reduce([g, np.fliplr(g), np.flipud(g), np.rot90(g,2)])]:
        if ok_all(f, pairs): return f
    return None

def s_gravity(pairs):
    for d in ('down', 'up', 'left', 'right'):
        f = lambda g, d=d: gravity(g, 0, d)
        if ok_all(f, pairs): return f
    return None

def s_fill_enclosed(pairs):
    a0, b0 = G(pairs[0][0]), G(pairs[0][1])
    newc = [c for c in np.unique(b0) if c not in np.unique(a0)]
    for c in list(newc) + list(range(1, 10)):
        f = lambda g, c=c: fill_enclosed(g, 0, c)
        if ok_all(f, pairs): return f
    return None

def s_denoise(pairs):
    """Remove isolated cells / keep only large components."""
    def f(g):
        out = np.zeros_like(g)
        for cells, c in components(g, bg=0, conn=4):
            if len(cells) > 1:
                for y, x in cells: out[y, x] = c
        return out
    if ok_all(f, pairs): return f
    def f2(g):
        out = np.zeros_like(g)
        comps = components(g, bg=0, conn=8)
        if not comps: return g
        big = max(len(c) for c, _ in comps)
        for cells, c in comps:
            if len(cells) == big:
                for y, x in cells: out[y, x] = c
        return out
    if ok_all(f2, pairs): return f2
    return None

def s_extract_object(pairs):
    """Output = bbox-crop of object chosen by rank (largest/smallest/etc)."""
    for conn in (4, 8):
        for crit in ("largest", "smallest", "rarecolor"):
            def f(g, conn=conn, crit=crit):
                comps = components(g, bg=0, conn=conn)
                if not comps: return None
                if crit == "largest": cells, c = max(comps, key=lambda t: len(t[0]))
                elif crit == "smallest": cells, c = min(comps, key=lambda t: len(t[0]))
                else:
                    cnt = Counter(g[g != 0].ravel())
                    rc = cnt.most_common()[-1][0]
                    sel = [t for t in comps if t[1] == rc]
                    if not sel: return None
                    cells, c = max(sel, key=lambda t: len(t[0]))
                return obj_grid(cells, c)
            if ok_all(f, pairs): return f
    return None

def s_recolor_by_size(pairs):
    """Recolor objects based on size rank."""
    def f(g):
        out = g.copy()
        comps = components(g, bg=0, conn=4)
        sizes = sorted(set(len(c) for c, _ in comps))
        for cells, c in comps:
            pass
        return out
    return None

def s_subgrid_split(pairs):
    """Split on a separator color into subgrids; combine or select."""
    def split(g, sep):
        m = g != sep
        rows = np.where(m.any(1))[0]
        cols = np.where(m.any(0))[0]
        # find full separator rows/cols
        rowsep = [y for y in range(g.shape[0]) if (g[y] == sep).all()]
        colsep = [x for x in range(g.shape[1]) if (g[:, x] == sep).all()]
        return rowsep, colsep
    a0, b0 = G(pairs[0][0]), G(pairs[0][1])
    for sep in range(10):
        rs, cs = split(a0, sep)
        if not rs and not cs: continue
        # extract blocks
        def blocks(g, rs, cs, sep):
            rb = [-1] + rs + [g.shape[0]]; cb = [-1] + cs + [g.shape[1]]
            gs = []
            for i in range(len(rb)-1):
                row = []
                for j in range(len(cb)-1):
                    blk = g[rb[i]+1:rb[i+1], cb[j]+1:cb[j+1]]
                    row.append(blk)
                gs.append(row)
            return gs
        blks = blocks(a0, rs, cs, sep)
        flat = [b for row in blks for b in row if b.size]
        # try: combine via elementwise ops (needs equal shapes)
        if len(flat) >= 2 and all(b.shape == flat[0].shape for b in flat):
            for op in [np.maximum, np.minimum,
                       lambda x, y: np.where(x > 0, x, y),
                       lambda x, y: np.where((x > 0) & (y > 0), x, 0),
                       lambda x, y: np.where(x == y, x, 0),
                       lambda x, y: np.where(x != y, np.maximum(x, y), 0)]:
                def f(g, op=op, rs=None, cs=None, sep=sep):
                    rr, cc = split(g, sep)
                    bl = [b for row in blocks(g, rr, cc, sep) for b in row if b.size]
                    if not bl: return None
                    out = bl[0]
                    for b in bl[1:]:
                        if b.shape != out.shape: return None
                        out = op(out, b)
                    return out
                if ok_all(f, pairs): return f
    return None

def s_draw_lines(pairs):
    """Connect pairs of same-colored cells with lines."""
    def f(g):
        out = g.copy()
        for c in np.unique(g):
            if c == 0: continue
            ys, xs = np.where(g == c)
            if len(ys) == 2:
                y0, y1 = sorted(ys); x0, x1 = sorted(xs)
                if y0 == y1: out[y0, x0:x1+1] = c
                elif x0 == x1: out[y0:y1+1, x0] = c
                else:
                    for y in range(y0, y1+1): out[y, x0] = c
                    for x in range(x0, x1+1): out[y1, x] = c
        return out
    if ok_all(f, pairs): return f
    return None

def s_frame_border(pairs):
    def f(g):
        out = g.copy()
        c = most_common(g[g != 0]) if (g != 0).any() else 1
        out[0, :] = c; out[-1, :] = c; out[:, 0] = c; out[:, -1] = c
        return out
    if ok_all(f, pairs): return f
    return None

def s_complete_pattern(pairs):
    """Fill 0-cells using periodic repetition inferred from the row/col."""
    def f(g):
        h, w = g.shape
        out = g.copy()
        # find smallest period in each direction
        for axis in (0, 1):
            n = h if axis == 0 else w
            for p in range(1, n):
                ok = True
                for i in range(n - p):
                    a = g[i] if axis == 0 else g[:, i]
                    b = g[i+p] if axis == 0 else g[:, i+p]
                    m = (a != 0) & (b != 0)
                    if m.any() and not (a[m] == b[m]).all():
                        ok = False; break
                if ok:
                    for i in range(n):
                        src = i % p
                        if axis == 0:
                            row = g[src]
                            mask = out[i] == 0
                            out[i][mask] = row[mask]
                        else:
                            col = g[:, src]
                            mask = out[:, i] == 0
                            out[:, i][mask] = col[mask]
                    break
        return out
    if ok_all(f, pairs): return f
    return None

def s_extract_marked(pairs):
    """Extract subgrid marked by a distinctive colored marker/corner."""
    return None

def s_move_to_marker(pairs):
    return None

def s_fold(pairs):
    """Fold grid along mid axis combining halves."""
    def make(op):
        def f(g):
            h, w = g.shape
            for axis in (0, 1):
                n = h if axis == 0 else w
                if n % 2 == 1:
                    a = g[:n//2] if axis == 0 else g[:, :n//2]
                    b = (g[n//2+1:][::-1] if axis == 0 else g[:, n//2+1:][:, ::-1])
                    try:
                        return op(a, b)
                    except Exception:
                        return None
            return None
        return f
    for op in [np.maximum, np.minimum, lambda a,b: np.where(a!=0,a,b),
               lambda a,b: np.where((a!=0)&(b!=0), np.maximum(a,b), 0)]:
        f = make(op)
        if ok_all(f, pairs): return f
    return None

def s_count_repeat(pairs):
    """Output = input or pattern repeated/cropped per object count."""
    return None

def s_bbox_overlay(pairs):
    """Draw bounding box around each object / fill it."""
    def f(g):
        out = g.copy()
        for cells, c in components(g, bg=0, conn=8):
            y0,x0,y1,x1 = bbox_of(cells)
            out[y0:y1, x0:x1] = np.where(out[y0:y1,x0:x1]==0, c, out[y0:y1,x0:x1])
        return out
    if ok_all(f, pairs): return f
    return None

def s_swap_to_common(pairs):
    """Set bg to most common color overall."""
    def f(g):
        out = g.copy()
        c = most_common(g)
        out[g == 0] = c
        return out
    if ok_all(f, pairs): return f
    return None

def s_cellwise_binary(pairs):
    """output = binary/derived function of input mask (e.g., invert, edges)."""
    def f1(g):
        return np.where(g == 0, np.max(g) if np.max(g)>0 else 1, 0)
    for f in [f1,
              lambda g: np.where(g==0, most_common(g), 0),
              lambda g: np.where(g!=0, most_common(g), 0)]:
        if ok_all(f, pairs): return f
    return None

def s_keep_color(pairs):
    a0, b0 = G(pairs[0][0]), G(pairs[0][1])
    keep = set(np.unique(b0)) & set(np.unique(a0))
    for c in keep:
        f = lambda g, c=c: np.where(g == c, c, 0)
        if ok_all(f, pairs): return f
    return None

def s_extend_lines(pairs):
    """Extend existing line segments across the grid."""
    def f(g):
        out = g.copy(); H, W = g.shape
        for c in np.unique(g):
            if c == 0: continue
            ys, xs = np.where(g == c)
            if len(ys) < 2: continue
            if len(set(ys)) == 1: out[ys[0], :] = c
            elif len(set(xs)) == 1: out[:, xs[0]] = c
        return out
    if ok_all(f, pairs): return f
    return None

def s_project(pairs):
    """Project objects onto axis (rows/cols histogram as pattern)."""
    return None

def s_copy_per_object(pairs):
    """Each object gets its own subgrid region (object separation)."""
    return None

def s_fill_holes(pairs):
    """Fill holes inside each object with a learned/selected color."""
    a0, b0 = G(pairs[0][0]), G(pairs[0][1])
    newc = [c for c in np.unique(b0) if c not in np.unique(a0)] or list(range(10))
    for c in list(newc) + list(range(10)):
        def f(g, c=c):
            out = g.copy()
            bg = most_common(g)
            for cells, col in components(g, bg=bg, conn=4):
                ys = [x[0] for x in cells]; xs = [x[1] for x in cells]
                y0, y1, x0, x1 = min(ys), max(ys)+1, min(xs), max(xs)+1
                sub = g[y0:y1, x0:x1]
                mask = np.ones_like(sub, bool)
                for (cy, cx) in cells: mask[cy-y0, cx-x0] = False
                # flood fill from border of sub
                seen = np.zeros_like(sub, bool); q = deque()
                for yy in range(sub.shape[0]):
                    for xx in (0, sub.shape[1]-1):
                        if mask[yy, xx] and not seen[yy, xx]: seen[yy, xx]=True; q.append((yy,xx))
                for xx in range(sub.shape[1]):
                    for yy in (0, sub.shape[0]-1):
                        if mask[yy, xx] and not seen[yy, xx]: seen[yy, xx]=True; q.append((yy,xx))
                while q:
                    cy, cx = q.popleft()
                    for dy, dx in DIRS:
                        ny, nx = cy+dy, cx+dx
                        if 0<=ny<sub.shape[0] and 0<=nx<sub.shape[1] and mask[ny,nx] and not seen[ny,nx]:
                            seen[ny,nx]=True; q.append((ny,nx))
                holes = mask & ~seen
                for hy, hx in np.argwhere(holes):
                    out[y0+hy, x0+hx] = c
            return out
        if ok_all(f, pairs): return f
    return None

def s_odd_one_out(pairs):
    """Find the object whose shape differs; output it or use it as mask."""
    for conn in (4, 8):
        for mode in ('extract', 'recolor'):
            def f(g, conn=conn, mode=mode):
                comps = components(g, bg=0, conn=conn)
                if len(comps) < 2: return None
                # group by normalized shape+color
                from collections import defaultdict
                groups = defaultdict(list)
                for cells, c in comps:
                    groups[(normalize(cells), c)].append((cells, c))
                odd = [v[0] for v in groups.values() if len(v) == 1]
                if len(odd) != 1: return None
                cells, c = odd[0]
                if mode == 'extract':
                    return obj_grid(cells, c)
                out = g.copy()
                nc = least_common(g)
                for y, x in cells: out[y, x] = nc if nc else 1
                return out
            if ok_all(f, pairs): return f
    return None

def s_color_count_square(pairs):
    """Output = NxN or 1xN grid of color c where N = count of something."""
    a0, b0 = G(pairs[0][0]), G(pairs[0][1])
    oh, ow = b0.shape
    if oh == ow or ow == 1 or oh == 1:
        n = max(oh, ow)
        for what in ('nobj', 'ncolors'):
            for c in range(10):
                def f(g, n0=n, c=c, what=what):
                    if what == 'nobj':
                        nn = len(components_multicolor(g, bg=0))
                    else:
                        nn = len(colors(g) - {0})
                    if oh == ow:
                        return np.full((nn, nn), c, int)
                    return np.full((1, nn) if oh == 1 else (nn, 1), c, int)
                if ok_all(f, pairs): return f
    return None

def s_marker_line(pairs):
    """Draw line between the two marker cells of a distinctive color."""
    a0, b0 = G(pairs[0][0]), G(pairs[0][1])
    cand = [c for c in np.unique(a0) if c != 0]
    for c in cand:
        for lc in range(1, 10):
            def f(g, c=c, lc=lc):
                pts = np.argwhere(g == c)
                if len(pts) != 2: return None
                out = g.copy()
                (y0, x0), (y1, x1) = pts
                if y0 == y1:
                    out[y0, min(x0,x1):max(x0,x1)+1] = lc
                elif x0 == x1:
                    out[min(y0,y1):max(y0,y1)+1, x0] = lc
                elif abs(y1-y0) == abs(x1-x0):
                    n = abs(y1-y0)
                    for i in range(n+1):
                        out[y0 + i*(1 if y1>y0 else -1), x0 + i*(1 if x1>x0 else -1)] = lc
                else:
                    return None
                return out
            if ok_all(f, pairs): return f
    return None

def s_recolor_objects_size(pairs):
    """Recolor objects by size: largest->c1, smallest->c2 etc."""
    a0, b0 = G(pairs[0][0]), G(pairs[0][1])
    outc = [c for c in np.unique(b0) if c != 0]
    for c_big in outc:
        for c_small in outc:
            def f(g, c_big=c_big, c_small=c_small):
                comps = components(g, bg=0, conn=8)
                if len(comps) < 2: return None
                out = g.copy()
                big = max(comps, key=lambda t: len(t[0]))
                small = min(comps, key=lambda t: len(t[0]))
                for cells, c in comps:
                    nc = c_big if (cells, c) == big else (c_small if (cells, c) == small else c)
                    for y, x in cells: out[y, x] = nc
                return out
            if ok_all(f, pairs): return f
    return None

def s_repeat_dir(pairs):
    """Repeat a detected motif to fill the grid along an axis."""
    a0, b0 = G(pairs[0][0]), G(pairs[0][1])
    if a0.shape != b0.shape: return None
    def f(g):
        out = g.copy(); H, W = g.shape
        for c in np.unique(g):
            if c == 0: continue
            ys, xs = np.where(g == c)
            if len(ys) == 0: continue
            # horizontal repeat
            if len(set(ys)) == 1 and (g == c).sum() < W:
                y = ys[0]
                xs_sorted = sorted(set(xs))
                if len(xs_sorted) >= 2:
                    period = xs_sorted[1] - xs_sorted[0]
                    x = xs_sorted[0]
                    while x < W:
                        out[y, x] = c; x += period
            if len(set(xs)) == 1 and (g == c).sum() < H:
                x = xs[0]
                ys_sorted = sorted(set(ys))
                if len(ys_sorted) >= 2:
                    period = ys_sorted[1] - ys_sorted[0]
                    y = ys_sorted[0]
                    while y < H:
                        out[y, x] = c; y += period
        return out
    if ok_all(f, pairs): return f
    return None

def s_enclose_box(pairs):
    """Draw minimal rectangle border around objects of a certain color."""
    a0, b0 = G(pairs[0][0]), G(pairs[0][1])
    for c in range(10):
        for lc in range(1, 10):
            def f(g, c=c, lc=lc):
                out = g.copy()
                for cells, col in components(g, bg=0, conn=8):
                    if col != c: continue
                    y0, x0, y1, x1 = bbox_of(cells)
                    out[y0, x0:x1] = lc; out[y1-1, x0:x1] = lc
                    out[y0:y1, x0] = lc; out[y0:y1, x1-1] = lc
                return out
            if ok_all(f, pairs): return f
    return None

def s_majority_vote_quarters(pairs):
    """Split into quadrants/subgrids and take majority/union."""
    return None

def s_extract_inner(pairs):
    """Extract interior of the largest frame-like object."""
    for conn in (4, 8):
        def f(g, conn=conn):
            comps = components(g, bg=0, conn=conn)
            if not comps: return None
            cells, c = max(comps, key=lambda t: len(t[0]))
            y0, x0, y1, x1 = bbox_of(cells)
            inner = g[y0+1:y1-1, x0+1:x1-1]
            return inner if inner.size else None
        if ok_all(f, pairs): return f
    return None

def s_toeplitz(pairs):
    """Periodic Toeplitz fill: nonzero input cells agree with
    color = F[(r+c) % p], F[(r-c) % p], F[r % p] or F[c % p]; fill whole grid.
    Covers diagonal-stripe completion tasks (e.g. 05269061)."""
    idxfns = [
        lambda r, c, p: (r + c) % p,
        lambda r, c, p: (r - c) % p,
        lambda r, c, p: r % p,
        lambda r, c, p: c % p,
    ]
    for idxfn in idxfns:
        for p in range(1, 8):
            def f(g, idxfn=idxfn, p=p):
                g = np.asarray(g)
                h, w = g.shape
                m = {}
                ys, xs = np.nonzero(g)
                if len(ys) == 0:
                    return None
                for y, x in zip(ys, xs):
                    k = idxfn(y, x, p)
                    if k in m and m[k] != g[y, x]:
                        return None
                    m[k] = g[y, x]
                if len(m) < p:   # period not fully determined
                    return None
                out = np.zeros_like(g)
                for y in range(h):
                    for x in range(w):
                        k = idxfn(y, x, p)
                        if k not in m:
                            return None
                        out[y, x] = m[k]
                return out
            if ok_all(f, pairs): return f
    return None

def s_kron_negate(pairs):
    """Each input cell expands to an s×s block: nonzero cell -> complement mask
    of the input pattern painted in the cell's color; zero cell -> empty block.
    Covers 0692e18c-style tasks."""
    a0, b0 = G(pairs[0][0]), G(pairs[0][1])
    ih, iw = a0.shape; oh, ow = b0.shape
    if ih == 0 or iw == 0 or oh % ih or ow % iw:
        return None
    sh, sw = oh // ih, ow // iw
    if (sh, sw) != (ih, iw):   # block size must equal input dims
        return None
    for mode in ('neg', 'pos'):
        def f(g, mode=mode, sh=sh, sw=sw):
            g = np.asarray(g)
            h, w = g.shape
            out = np.zeros((h * sh, w * sw), dtype=g.dtype)
            for y in range(h):
                for x in range(w):
                    c = g[y, x]
                    if c == 0:
                        continue
                    mask = (g == 0) if mode == 'neg' else (g != 0)
                    out[y * sh:(y + 1) * sh, x * sw:(x + 1) * sw] = mask * c
            return out
        if ok_all(f, pairs): return f
    return None

def s_shift(pairs):
    """Constant translation of all non-background cells by (dr, dc)."""
    for dr in range(-3, 4):
        for dc in range(-3, 4):
            if dr == 0 and dc == 0:
                continue
            def f(g, dr=dr, dc=dc):
                g = np.asarray(g)
                out = np.zeros_like(g)
                ys, xs = np.nonzero(g)
                ny, nx = ys + dr, xs + dc
                keep = (ny >= 0) & (ny < g.shape[0]) & (nx >= 0) & (nx < g.shape[1])
                if not keep.all():
                    return None
                out[ny, nx] = g[ys, xs]
                return out
            if ok_all(f, pairs): return f
    return None

def s_slide_to_touch(pairs):
    """Slide one color's object in a direction until adjacent (4-conn) to another
    colored object. Learn (moved color, direction) from train."""
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for mc in range(1, 10):
        for dr, dc in dirs:
            def f(g, mc=mc, dr=dr, dc=dc):
                g = np.asarray(g).copy()
                ys, xs = np.nonzero(g == mc)
                if len(ys) == 0:
                    return None
                others = (g != 0) & (g != mc)
                for _ in range(g.shape[0] + g.shape[1]):
                    ny, nx = ys + dr, xs + dc
                    if (ny < 0).any() or (ny >= g.shape[0]).any() or \
                       (nx < 0).any() or (nx >= g.shape[1]).any():
                        return g
                    # would overlap another color -> stop before overlap
                    if others[ny, nx].any():
                        return g
                    g[ys, xs] = 0
                    g[ny, nx] = mc
                    ys, xs = ny, nx
                    # adjacency check
                    pad = np.zeros((g.shape[0] + 2, g.shape[1] + 2), dtype=bool)
                    pad[1:-1, 1:-1] = g == mc
                    adj = others & (pad[0:-2, 1:-1] | pad[2:, 1:-1] |
                                    pad[1:-1, 0:-2] | pad[1:-1, 2:])
                    if adj.any():
                        return g
                return g
            if ok_all(f, pairs): return f
    return None

def s_periodic_denoise(pairs):
    """Grid is a periodic tiling corrupted by noise.  Rebuild the majority
    tile (phase offsets included — the majority tile at offset 0 equals the
    true tile up to a shift), then zero out rows/cols that disagree with the
    reconstruction."""
    for ph in range(2, 10):
        for pw in range(2, 10):
            def f(g, ph=ph, pw=pw):
                g = np.asarray(g)
                h, w = g.shape
                if h < ph * 2 or w < pw * 2:
                    return None
                tile = np.zeros((ph, pw), dtype=g.dtype)
                for y in range(ph):
                    for x in range(pw):
                        vals = g[y::ph, x::pw].ravel()
                        c = Counter(vals.tolist()).most_common(1)[0][0]
                        tile[y, x] = c
                yi = np.arange(h)[:, None]
                xi = np.arange(w)[None, :]
                best = None
                for dy in range(ph):
                    for dx in range(pw):
                        recon = tile[(yi + dy) % ph, (xi + dx) % pw]
                        score = (recon == g).mean()
                        if best is None or score > best[0]:
                            best = (score, recon)
                if best is None or best[0] < 0.5:
                    return None
                recon = best[1]
                agree = recon == g
                out = recon.copy()
                out[agree.mean(1) < 0.5, :] = 0
                out[:, agree.mean(0) < 0.5] = 0
                return out
            if ok_all(f, pairs): return f
    return None

def s_extend_rays(pairs):
    """Cross/plus-like motif: the cell adjacent to the center in each of the
    8 directions defines a ray color; extend every ray one step beyond the
    motif's bounding box (covers 0962bcdd-style tasks)."""
    dirs = [(-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)]
    for extra in (1, 2):
        def f(g, extra=extra):
            g = np.asarray(g).copy()
            for cells, col in components_multicolor(g, bg=0):
                ys = [c[0] for c in cells]; xs = [c[1] for c in cells]
                ry = max(ys) - min(ys) + 1; rx = max(xs) - min(xs) + 1
                if ry != rx or ry % 2 == 0 or ry > 7:
                    continue
                cy, cx = (min(ys) + max(ys)) // 2, (min(xs) + max(xs)) // 2
                rad = ry // 2
                for dr, dc in dirs:
                    ay, ax = cy + dr, cx + dc
                    col_at = g[ay, ax] if 0 <= ay < g.shape[0] \
                             and 0 <= ax < g.shape[1] else 0
                    # rays with no arm cell fall back to the centre colour
                    if col_at == 0:
                        col_at = g[cy, cx]
                    for k in range(1, rad + extra + 1):
                        y, x = cy + dr * k, cx + dc * k
                        if 0 <= y < g.shape[0] and 0 <= x < g.shape[1] \
                                and g[y, x] == 0:
                            g[y, x] = col_at
            return g
        if ok_all(f, pairs): return f
    return None

def s_extrapolate_dots(pairs):
    """Single-pixel dots form an arithmetic progression; continue it with a
    new color until leaving the grid (covers 0b17323b-style tasks)."""
    for nc in range(1, 10):
        def f(g, nc=nc):
            g = np.asarray(g).copy()
            comps = components(g, bg=0, conn=8)
            if not comps:
                return None
            # all components must be single cells of one color
            src = None
            pts = []
            for cells, col in comps:
                if len(cells) != 1:
                    return None
                if src is None:
                    src = col
                elif col != src:
                    return None
                pts.append(tuple(cells[0]))
            if len(pts) < 2:
                return None
            pts.sort()
            dr = pts[-1][0] - pts[-2][0]
            dc = pts[-1][1] - pts[-2][1]
            if dr == 0 and dc == 0:
                return None
            y, x = pts[-1]
            changed = False
            while True:
                y, x = y + dr, x + dc
                if not (0 <= y < g.shape[0] and 0 <= x < g.shape[1]):
                    break
                if g[y, x] == 0:
                    g[y, x] = nc
                    changed = True
            return g if changed else None
        if ok_all(f, pairs): return f
    return None

SOLVERS = [s_geom, s_identity_plus, s_recolor_const, s_trim, s_crop_color_bbox,
           s_upscale, s_tile, s_kron_self, s_symmetry, s_gravity, s_fill_enclosed,
           s_denoise, s_extract_object, s_subgrid_split, s_draw_lines,
           s_frame_border, s_complete_pattern, s_fold, s_bbox_overlay,
           s_swap_to_common, s_cellwise_binary, s_keep_color, s_extend_lines,
           s_fill_holes, s_odd_one_out, s_color_count_square, s_marker_line,
           s_recolor_objects_size, s_repeat_dir, s_enclose_box, s_extract_inner,
           s_toeplitz, s_kron_negate, s_shift, s_slide_to_touch,
           s_periodic_denoise, s_extend_rays, s_extrapolate_dots]
