"""Procedural ARC-style tasks with a known program, for SFT of the AGI-2 base model.

    python3 gen_synth_ag2.py --n 3000 --out synth.jsonl [--seed 0] [--mix same=0.72,extract=0.23,larger=0.05]
                             [--depth2 0.35] [--exclude-train]
    python3 gen_synth_ag2.py --verify synth.jsonl
    python3 gen_synth_ag2.py --selftest

Each row is one task in the same shape as Nabidnur/arc-agi-2-grids `synthetic/` rows, so both
check_sft_data.py (jsonl:<file>) and sft_ag2.py (--data jsonl:<file>) read it unchanged; sft_ag2 adds the
8 dihedral x colour-permutation augmentation on the fly, so rows are stored un-augmented:
    {"task_id", "family", "program": [[op, params...], ...], "klass", "train": [{input, output}...],
     "test": [{input, output}...]}
`program` is the solution: solve(program, input) == output for every pair (re-checked by --verify from the
stored JSON alone). Primitives come from dsl_all.py (rot/flip/trim/crop/upscale/gravity/fill_enclosed/
...) -- used here to MAKE tasks, not to solve eval tasks.

Guards built into generation:
  * no pair may match a pair of the 120 evaluation tasks under rotation/reflection/recolouring
    (check_sft_data.pair_hash); with --exclude-train also none of the 1,000 public training tasks.
  * every pair must change the grid, outputs must differ across pairs, all grids within 1..30.
  * class mix is enforced by construction: the class of a task is the class of its test pair
    (output area vs input area), and a task is only kept if it lands in the class that was drawn.

How much to generate, and what to expect -- honest version:
  * These are 26 single-concept families (plus two-step compositions). The base checkpoint comes from
    a lineage trained on far larger synthetic corpora, and it already ends per-task TTT at loss ~1e-4;
    the eval misses were measured to be systematic (same error in every candidate). Tasks this simple
    are unlikely to teach the concepts those misses need. Expect at most a small shift.
  * So size the first run to be cheap and measurable, not large: 2,000-4,000 tasks is one L4 session
    in sft_ag2.py (each task is seen ~1-2 times with fresh augmentation; xcalibur's whole run was 63
    updates on 500 tasks). Judge it with the instruments already built -- held-out val loss in the
    sft_ag2 manifest, then the 24-task panel + gold-NLL delta (swap_runbook.md). Only if the panel
    moves is a larger corpus (and richer families) worth generating; more rows of the same 26
    families will not add information past a few thousand tasks.
  * The share solved by the independent rule-fitters in dsl_all.SOLVERS (printed by --selftest) is a
    floor on 'the rule is inferable from the demos', not a difficulty score.
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_sft_data as csd  # noqa: E402
import dsl_all as D  # noqa: E402

BASE = os.path.dirname(HERE)
EVAL_MIX = {"same": 0.72, "extract": 0.23, "larger": 0.05}       # share of the 172 eval test inputs per class


# ------------------------------------------------------------------ operations
def components(g):
    """4-connected same-colour components of non-zero cells -> [(cells, color)], scan order.
    (dsl_all.components raises for conn=4 -- an operator-precedence slip in its neighbour tuple --
    so the generator carries its own.)"""
    H, W = g.shape
    seen = np.zeros(g.shape, bool)
    out = []
    for y in range(H):
        for x in range(W):
            if seen[y, x] or g[y, x] == 0:
                continue
            c, stack, cells = g[y, x], [(y, x)], []
            seen[y, x] = True
            while stack:
                cy, cx = stack.pop()
                cells.append((cy, cx))
                for ny, nx in ((cy + 1, cx), (cy - 1, cx), (cy, cx + 1), (cy, cx - 1)):
                    if 0 <= ny < H and 0 <= nx < W and not seen[ny, nx] and g[ny, nx] == c:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            out.append((sorted(cells), int(c)))
    return out


def _rank_recolor(g, big, other):
    out = g.copy()
    comps = sorted(components(g), key=lambda c: (-len(c[0]), c[0][0]))
    for i, (cells, _) in enumerate(comps):
        for y, x in cells:
            out[y, x] = big if i == 0 else other
    return out


def _rays(g, direction, color):
    out = g.copy()
    dy, dx = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}[direction]
    for y, x in np.argwhere(g == color):
        y, x = y + dy, x + dx
        while 0 <= y < g.shape[0] and 0 <= x < g.shape[1] and g[y, x] == 0:
            out[y, x] = color
            y, x = y + dy, x + dx
    return out


def _denoise(g):
    out = g.copy()
    for cells, _ in components(g):
        if len(cells) == 1:
            out[cells[0]] = 0
    return out


def _bbox_fill(g):
    out = g.copy()
    for cells, c in components(g):
        ys, xs = [p[0] for p in cells], [p[1] for p in cells]
        out[min(ys):max(ys) + 1, min(xs):max(xs) + 1] = c
    return out


def _outline(g):
    out = g.copy()
    H, W = g.shape
    for y in range(1, H - 1):
        for x in range(1, W - 1):
            c = g[y, x]
            if c and g[y - 1, x] == c and g[y + 1, x] == c and g[y, x - 1] == c and g[y, x + 1] == c:
                out[y, x] = 0
    return out


def _crop_largest(g):
    comps = components(g)
    cells = max(comps, key=lambda c: (len(c[0]), -c[0][0][0], -c[0][0][1]))[0]
    return D.crop(g, D.bbox_of(np.array(cells)))


def _count_bar(g, color):
    return np.full((1, len(components(g))), color, int)


def _recolor(g, pairs):
    out = g.copy()
    for a, b in pairs:
        out[g == a] = b
    return out


def _frame(g, c):
    out = g.copy()
    out[0, :] = out[-1, :] = out[:, 0] = out[:, -1] = c
    return out


def _mirror(g, axis):
    f = np.fliplr(g) if axis == "lr" else np.flipud(g)
    return np.where(g != 0, g, f)


def _half(g, which):
    H, W = g.shape
    return {"top": g[:H // 2], "bottom": g[H - H // 2:], "left": g[:, :W // 2], "right": g[:, W - W // 2:]}[which]


OPS = {
    # same-size
    "rot": lambda g, k: D.rot(g, k), "flip": lambda g, a: D.fliph(g) if a == "h" else D.flipv(g),
    "transpose": lambda g: D.transp(g), "recolor": _recolor, "gravity": lambda g, d: D.gravity(g, 0, d),
    "fill_enclosed": lambda g, c: D.fill_enclosed(g, 0, c), "mirror": _mirror,
    "roll": lambda g, dy, dx: np.roll(np.roll(g, dy, 0), dx, 1), "frame": _frame, "rank_recolor": _rank_recolor,
    "rays": _rays, "keep": lambda g, c: np.where(g == c, g, 0), "denoise": _denoise, "bbox_fill": _bbox_fill,
    "outline": _outline,
    # extract
    "trim": lambda g: D.trim(g, 0), "crop_largest": _crop_largest, "downscale": lambda g, k: D.downscale(g, k),
    "half": _half, "crop_color": lambda g, c: D.crop(g, D.bbox_of(g == c)), "count_bar": _count_bar,
    # larger
    "upscale": lambda g, k: D.upscale(g, k), "tile": lambda g, ry, rx: np.tile(g, (ry, rx)),
    "concat_flip": lambda g, a: np.hstack([g, np.fliplr(g)]) if a == "h" else np.vstack([g, np.flipud(g)]),
    "pad": lambda g, c: np.pad(g, 1, constant_values=c), "kron_self": lambda g: np.kron((g != 0).astype(int), g),
}
POST = ("rot", "flip", "transpose", "recolor")                       # second step of a two-step program
FAMILIES = {
    "same": ["rot", "flip", "transpose", "recolor", "gravity", "fill_enclosed", "mirror", "roll", "frame",
             "rank_recolor", "rays", "keep", "denoise", "bbox_fill", "outline"],
    "extract": ["trim", "crop_largest", "downscale", "half", "crop_color", "count_bar"],
    "larger": ["upscale", "tile", "concat_flip", "pad", "kron_self"],
}


def solve(program, grid):
    """The solution of a generated task: apply the stored program to an input grid."""
    g = np.asarray(grid, dtype=int)
    for step in program:
        g = np.asarray(OPS[step[0]](g, *step[1:]), dtype=int)
    return g


# ---------------------------------------------------------------------- scenes
def _noise(r, lo=4, hi=14, colors=None):
    h, w = int(r.integers(lo, hi + 1)), int(r.integers(lo, hi + 1))
    cols = colors if colors is not None else r.choice(np.arange(1, 10), size=int(r.integers(2, 5)), replace=False)
    g = np.zeros((h, w), int)
    m = r.random((h, w)) < r.uniform(0.15, 0.5)
    g[m] = r.choice(cols, size=int(m.sum()))
    return g


def _rects(r, hollow=False, lo=8, hi=18, n=None, min_side=1):
    h, w = int(r.integers(lo, hi + 1)), int(r.integers(lo, hi + 1))
    g = np.zeros((h, w), int)
    cols = r.permutation(np.arange(1, 10))
    placed = 0
    for i in range(int(n if n is not None else r.integers(2, 6))):
        for _ in range(20):
            rh, rw = int(r.integers(min_side, 6)), int(r.integers(min_side, 6))
            if rh >= h - 1 or rw >= w - 1:
                continue
            y, x = int(r.integers(0, h - rh + 1)), int(r.integers(0, w - rw + 1))
            if g[max(0, y - 1):y + rh + 1, max(0, x - 1):x + rw + 1].any():
                continue
            g[y:y + rh, x:x + rw] = cols[i % 9]
            if hollow and rh >= 3 and rw >= 3:
                g[y + 1:y + rh - 1, x + 1:x + rw - 1] = 0
            placed += 1
            break
    return g if placed else None


def _dots(r, color):
    h, w = int(r.integers(7, 17)), int(r.integers(7, 17))
    g = np.zeros((h, w), int)
    for _ in range(int(r.integers(2, 6))):
        g[int(r.integers(1, h - 1)), int(r.integers(1, w - 1))] = color
    return g


def _scene(r, op, params):
    """An input grid on which `op` with `params` is well defined and non-trivial."""
    if op in ("fill_enclosed",):
        return _rects(r, hollow=True, min_side=3)
    if op in ("rank_recolor", "crop_largest", "bbox_fill", "count_bar"):
        g = _rects(r)
        if g is not None and op == "bbox_fill":                    # knock holes into the rectangles
            m = (g != 0) & (r.random(g.shape) < 0.3)
            g = np.where(m, 0, g)
        return g
    if op == "outline":
        return _rects(r, min_side=3)
    if op == "rays":
        return _dots(r, params[1])
    if op == "denoise":
        g = _rects(r, min_side=2)
        if g is None:
            return None
        m = (g == 0) & (r.random(g.shape) < 0.06)
        g[m] = r.integers(1, 10, size=int(m.sum()))
        return g
    if op == "downscale":
        return D.upscale(_noise(r, 3, 7), params[0])
    if op == "crop_color":
        g = _rects(r)
        if g is None:
            return None
        comps = components(g)
        cells = comps[int(r.integers(0, len(comps)))][0]
        g = np.where(g == params[0], 0, g)                         # the marked colour appears on exactly one object
        for y, x in cells:
            g[y, x] = params[0]
        return g
    if op in ("upscale", "tile", "kron_self"):
        return _noise(r, 2, 5)
    if op in ("concat_flip", "pad"):
        return _noise(r, 3, 10)
    if op == "trim":
        g = np.zeros((int(r.integers(8, 17)), int(r.integers(8, 17))), int)
        s = _noise(r, 2, 6)
        y, x = int(r.integers(0, g.shape[0] - s.shape[0] + 1)), int(r.integers(0, g.shape[1] - s.shape[1] + 1))
        g[y:y + s.shape[0], x:x + s.shape[1]] = s
        return g
    if op == "half":
        return _noise(r, 6, 14)
    return _noise(r)


def _params(r, op):
    c = lambda: int(r.integers(1, 10))                                                  # noqa: E731
    if op == "rot":
        return [int(r.integers(1, 4))]
    if op in ("flip", "concat_flip"):
        return [str(r.choice(["h", "v"]))]
    if op == "recolor":
        k = int(r.integers(1, 4))
        src = r.choice(np.arange(1, 10), size=k, replace=False)
        dst = r.choice(np.arange(1, 10), size=k, replace=False)
        return [[[int(a), int(b)] for a, b in zip(src, dst) if a != b] or [[1, 2]]]
    if op == "gravity":
        return [str(r.choice(["down", "up", "left", "right"]))]
    if op in ("fill_enclosed", "frame", "keep", "crop_color", "count_bar", "pad"):
        return [c()]
    if op == "mirror":
        return [str(r.choice(["lr", "ud"]))]
    if op == "roll":
        dy, dx = int(r.integers(-3, 4)), int(r.integers(-3, 4))
        return [dy, dx] if (dy or dx) else [1, 0]
    if op == "rank_recolor":
        a, b = r.choice(np.arange(1, 10), size=2, replace=False)
        return [int(a), int(b)]
    if op == "rays":
        return [str(r.choice(["up", "down", "left", "right"])), c()]
    if op in ("downscale", "upscale"):
        return [int(r.integers(2, 4))]
    if op == "half":
        return [str(r.choice(["top", "bottom", "left", "right"]))]
    if op == "tile":
        ry, rx = int(r.integers(1, 4)), int(r.integers(1, 4))
        return [ry, rx] if ry * rx > 1 else [2, 2]
    return []


def klass_of(inp, out):
    a, b = np.asarray(inp).size, np.asarray(out).size
    return "same" if a == b else "extract" if b < a else "larger"


def gen_task(r, klass, depth2=0.35):
    """One task of the requested class, or None if this draw was degenerate."""
    op = str(r.choice(FAMILIES[klass]))
    program = [[op] + _params(r, op)]
    if r.random() < depth2:
        p = str(r.choice(POST))
        program.append([p] + _params(r, p))
    n_train, n_test = int(r.choice([2, 3, 3, 3, 4, 4, 5])), int(r.choice([1, 1, 1, 2]))
    pairs = []
    for _ in range(n_train + n_test):
        for _ in range(12):
            g = _scene(r, op, program[0][1:])
            if g is None or not g.any():
                continue
            try:
                out = solve(program, g)
            except Exception:                                           # noqa: BLE001  (e.g. no object to crop)
                continue
            if not (1 <= out.shape[0] <= 30 and 1 <= out.shape[1] <= 30 and out.size and g.shape[0] <= 30 and g.shape[1] <= 30):
                continue
            if out.shape == g.shape and np.array_equal(out, g):
                continue
            pairs.append((g, out))
            break
        else:
            return None
    ins = {p[0].tobytes() + bytes(p[0].shape) for p in pairs}
    outs = {p[1].tobytes() + bytes(p[1].shape) for p in pairs}
    if len(ins) < len(pairs) or len(outs) < 2:
        return None
    if any(klass_of(i, o) != klass for i, o in pairs[n_train:]):
        return None
    to = lambda ps: [{"input": i.tolist(), "output": o.tolist()} for i, o in ps]       # noqa: E731
    return {"family": "+".join(s[0] for s in program), "program": program, "klass": klass,
            "train": to(pairs[:n_train]), "test": to(pairs[n_train:])}


def reference_hashes(with_train=False):
    ev = set(csd.eval_hashes())
    if with_train:
        tr = json.load(open(os.path.join(BASE, "arc-agi_training_challenges.json")))
        so = json.load(open(os.path.join(BASE, "arc-agi_training_solutions.json")))
        for k, t in tr.items():
            ev |= {csd.pair_hash(p["input"], p["output"]) for p in t["train"]}
            ev |= {csd.pair_hash(p["input"], so[k][i]) for i, p in enumerate(t["test"])}
    return ev


def generate(n, seed=0, mix=None, depth2=0.35, banned=None):
    """-> (rows, stats). Deterministic in (n, seed, mix, depth2)."""
    mix = mix or EVAL_MIX
    banned = set(banned) if banned is not None else reference_hashes()
    want = {k: int(round(n * v)) for k, v in mix.items()}
    want[max(want, key=want.get)] += n - sum(want.values())
    rows, seen, stats = [], set(), {"attempts": 0, "degenerate": 0, "banned": 0, "duplicate": 0}
    for klass in ("same", "extract", "larger"):
        made, i = 0, 0
        while made < want.get(klass, 0):
            r = np.random.default_rng([seed, {"same": 0, "extract": 1, "larger": 2}[klass], i])
            i += 1
            stats["attempts"] += 1
            if i > 200 * max(want[klass], 10):
                raise SystemExit(f"could not generate enough '{klass}' tasks")
            row = gen_task(r, klass, depth2)
            if row is None:
                stats["degenerate"] += 1
                continue
            hs = [csd.pair_hash(p["input"], p["output"]) for p in row["train"] + row["test"]]
            if any(h in banned for h in hs):
                stats["banned"] += 1
                continue
            if any(h in seen for h in hs):
                stats["duplicate"] += 1
                continue
            seen.update(hs)
            row = {"task_id": f"gs{seed:02d}{klass[0]}{made:06d}", **row}
            rows.append(row)
            made += 1
    return rows, stats


def verify(rows):
    """Re-execute every stored program on every stored input. -> number of failing tasks."""
    bad = 0
    for row in rows:
        ok = all(np.array_equal(solve(row["program"], p["input"]), np.asarray(p["output"])) for p in row["train"] + row["test"])
        bad += not ok
    return bad


def write(rows, path):
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def selftest():
    import collections
    import tempfile
    rows, stats = generate(400, seed=7)
    assert len(rows) == 400 and len({r["task_id"] for r in rows}) == 400
    assert all("_" not in r["task_id"] for r in rows)                       # ids must not contain the test-index separator
    kl = collections.Counter(r["klass"] for r in rows)
    assert kl == {"same": 288, "extract": 92, "larger": 20}, kl
    fams = {r["program"][0][0] for r in rows}
    missing = set(sum(FAMILIES.values(), [])) - fams
    assert not missing, f"families never generated: {missing}"
    two = sum(len(r["program"]) == 2 for r in rows)
    assert 0.2 < two / 400 < 0.5
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "synth.jsonl")
        write(rows, path)
        back = [json.loads(line) for line in open(path)]
        # (a) the stored program reproduces every output, from the JSON alone
        assert verify(back) == 0
        broken = json.loads(json.dumps(back[0]))
        broken["test"][0]["output"][0][0] = (broken["test"][0]["output"][0][0] + 1) % 10
        assert verify([broken]) == 1
        # (b) no collision with the evaluation set; a planted eval pair is rejected by the generation hook
        ev = csd.eval_hashes()
        assert not any(csd.pair_hash(p["input"], p["output"]) in ev for r in back for p in r["train"] + r["test"])
        some = csd.pair_hash(back[3]["train"][0]["input"], back[3]["train"][0]["output"])
        rows2, stats2 = generate(400, seed=7, banned=set(ev) | {some})
        assert stats2["banned"] >= 1 and back[3]["train"][0] not in [p for r in rows2 for p in r["train"]]
        # (c) the file passes the SFT data gate and feeds the trainer's data path unchanged
        res = csd.check("jsonl:" + path, ev, verbose=False)
        assert res["ok"] and res["rows"] == 400 and not res["contaminated_tasks"] and res["unparseable"] == 0, res
        assert res["classes"] == dict(kl)
        import sft_ag2
        tr, va = sft_ag2.load_tasks(["jsonl:" + path], log=lambda *_: None)
        assert len(tr) + len(va) == 400
        fmt = csd.formatter()
        for s in range(50):
            ids, lab = sft_ag2.train_example(tr, s, 1, fmt, 4096)
            assert ids and len(ids) == len(lab) <= 4096 and max(ids) <= 15
        assert generate(400, seed=7)[0] == rows and generate(40, seed=8)[0] != rows[:40]      # deterministic per seed
        only = generate(30, seed=1, mix={"extract": 1.0})[0]
        assert {r["klass"] for r in only} == {"extract"} and len(only) == 30
    # how many are solved from the demos alone by the independent rule-fitters in dsl_all
    solved = 0
    sample = rows[::4]
    for row in sample:
        prs = [(D.G(p["input"]), D.G(p["output"])) for p in row["train"]]
        for s in D.SOLVERS:
            try:
                f = s(prs)
                if f is not None and all(np.array_equal(np.asarray(f(D.G(p["input"]))), np.asarray(p["output"])) for p in row["test"]):
                    solved += 1
                    break
            except Exception:                                               # noqa: BLE001
                continue
    toks = sorted(sum(np.asarray(p["input"]).size + np.asarray(p["output"]).size for p in r["train"] + r["test"]) for r in rows)
    print(f"selftest ok: 400 tasks ({dict(kl)}), all {len(fams)} families present, {two} two-step programs; "
          f"programs reproduce every output from the stored JSON; 0 eval collisions and a planted pair is rejected; "
          f"file passes check_sft_data and feeds sft_ag2; deterministic per seed")
    print(f"  generation: {stats['attempts']} draws for 400 tasks ({stats['degenerate']} degenerate, {stats['banned']} banned, "
          f"{stats['duplicate']} duplicate); cells per task median {toks[len(toks) // 2]}, max {toks[-1]}")
    print(f"  inferable from demos by dsl_all rule-fitters: {solved}/{len(sample)} of a sample "
          "(a floor on well-posedness, not a difficulty measure)")


if __name__ == "__main__":
    a = sys.argv[1:]
    if not a:
        sys.exit(__doc__)
    if a[0] == "--selftest":
        selftest()
    elif a[0] == "--verify":
        rows = [json.loads(line) for line in open(a[1])]
        bad = verify(rows)
        print(f"{len(rows)} tasks, {bad} with a program that does not reproduce its outputs")
        sys.exit(1 if bad else 0)
    else:
        opt = {k: a[a.index(k) + 1] for k in ("--n", "--out", "--seed", "--mix", "--depth2") if k in a}
        if "--n" not in opt or "--out" not in opt:
            sys.exit(__doc__)
        mix = {k: float(v) for k, v in (x.split("=") for x in opt["--mix"].split(","))} if "--mix" in opt else None
        if mix and abs(sum(mix.values()) - 1.0) > 1e-6:
            sys.exit("--mix shares must sum to 1")
        banned = reference_hashes(with_train="--exclude-train" in a)
        rows, stats = generate(int(opt["--n"]), int(opt.get("--seed", 0)), mix, float(opt.get("--depth2", 0.35)), banned)
        write(rows, opt["--out"])
        import collections
        print(f"wrote {len(rows)} tasks to {opt['--out']}: classes {dict(collections.Counter(r['klass'] for r in rows))}; "
              f"{stats['attempts']} draws ({stats['degenerate']} degenerate, {stats['banned']} banned, {stats['duplicate']} duplicate)")
        print("next: python3 check_sft_data.py jsonl:" + opt["--out"] + "   then   sft_ag2.py --data jsonl:" + opt["--out"])
