"""Build the selection-v2 kernel dir from the production perfpatch notebook.

    python3 make_select_kernel.py --verdict /tmp/evalscan/select_verdict.json
    python3 make_select_kernel.py --strategy "kgmon+priors"          # force a strategy
    python3 make_select_kernel.py --selftest                         # GPU-free check of the built cells

Writes ../../submit_ag2_selectv2/{arc-agi2-selectv2.ipynb,kernel-metadata.json}.
It only BUILDS the directory -- pushing is a separate, manual step.

Three anchor-asserted edits to the base notebook, nothing else changes
(decode, TTT and budget cells are byte-identical to production):
  1. new cell      %%writefile select_v2.py      (this repo's select_v2.py, verbatim)
  2. arc_decoder   run_selection_algo accepts a strategy NAME and routes it to
                   select_v2.run_selection with production score_kgmon as the
                   per-input fallback
  3. final cell    decoder.run_selection_algo()  ->  run_selection_algo(SELECT_STRATEGY)
"""
import argparse
import copy
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
BASE_NB = os.path.join(REPO, "submit_ag2_perfpatch", "arc-agi2-lb33-perfpatch.ipynb")
BASE_META = os.path.join(REPO, "submit_ag2_perfpatch", "kernel-metadata.json")
OUT_DIR = os.path.join(REPO, "submit_ag2_selectv2")
TITLE, SLUG = "ARC AGI2 SelectV2", "arc-agi2-selectv2"          # id must equal the slugified title (409 otherwise)

DECODER_OLD = '''    def run_selection_algo(self, selection_algorithm=score_kgmon):
        return {bk: selection_algorithm({k: g for k, g in v.items()}) for bk, v in self.decoded_results.items()}
'''
DECODER_NEW = '''    def run_selection_algo(self, selection_algorithm=score_kgmon, ranker=None):
        if isinstance(selection_algorithm, str):
            # select_v2 strategy by name; any per-input failure falls back to score_kgmon
            try:
                import select_v2
                return select_v2.run_selection(self.decoded_results, self.dataset.queries,
                                               selection_algorithm, ranker=ranker, fallback=score_kgmon)
            except Exception as e:
                print(f"*** select_v2 unavailable ({e.__class__.__name__}: {e}) -> score_kgmon for all inputs")
                selection_algorithm = score_kgmon
        return {bk: selection_algorithm({k: g for k, g in v.items()}) for bk, v in self.decoded_results.items()}
'''
FINAL_OLD = "submission = data.get_submission(decoder.run_selection_algo())\n"
FINAL_NEW = "submission = data.get_submission(decoder.run_selection_algo(SELECT_STRATEGY, ranker=SELECT_RANKER))\n"


def src(cell):
    return "".join(cell["source"])


def set_src(cell, text):
    cell["source"] = text.splitlines(keepends=True)
    if cell["cell_type"] == "code":
        cell["outputs"], cell["execution_count"] = [], None


def find(cells, prefix):
    hits = [i for i, c in enumerate(cells) if src(c).startswith(prefix)]
    assert len(hits) == 1, f"expected exactly one cell starting with {prefix!r}, got {len(hits)}"
    return hits[0]


def build(strategy, ranker=None, base_nb=BASE_NB):
    import select_v2
    if strategy == "learned (LOTO-CV)":
        strategy = "learned"
    if strategy == "learned":
        select_v2.make_learned(ranker)                     # raises on a missing / mismatched ranker
    else:
        assert strategy in select_v2.STRATEGIES, f"unknown strategy {strategy!r}"
    nb = json.load(open(base_nb))
    cells = nb["cells"]

    i_dec = find(cells, "%%writefile arc_decoder.py")
    s = src(cells[i_dec])
    assert s.count(DECODER_OLD) == 1, "arc_decoder.run_selection_algo anchor not found"
    set_src(cells[i_dec], s.replace(DECODER_OLD, DECODER_NEW))

    sel = copy.deepcopy(cells[i_dec])
    set_src(sel, "%%writefile select_v2.py\n" + open(os.path.join(HERE, "select_v2.py")).read())
    cells.insert(i_dec + 1, sel)

    finals = [i for i, c in enumerate(cells) if FINAL_OLD in src(c)]
    assert len(finals) == 1, "final submission cell anchor not found"
    s = src(cells[finals[0]])
    header = (f"SELECT_STRATEGY = {strategy!r}\n"
              f"SELECT_RANKER = {json.dumps(ranker) if ranker else None}\n")
    set_src(cells[finals[0]], header + s.replace(FINAL_OLD, FINAL_NEW))

    meta = json.load(open(BASE_META))
    owner = meta["id"].split("/")[0]
    meta.update(id=f"{owner}/{SLUG}", title=TITLE, code_file=f"{SLUG}.ipynb")
    return nb, meta


def write(nb, meta, out_dir=OUT_DIR):
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, meta["code_file"]), "w") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    with open(os.path.join(out_dir, "kernel-metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)
    return out_dir


def selftest():
    """Materialise the built %%writefile cells and run the final-cell selection
    path on synthetic pickles with a stub dataset (arc_loader needs
    transformers, which is not required for this path)."""
    import bz2
    import pickle
    import tempfile
    import numpy as np
    import select_v2

    def materialise(nb, d):
        for c in nb["cells"]:
            s = src(c)
            if s.startswith("%%writefile "):
                name, body = s.split("\n", 1)
                open(os.path.join(d, name.split()[1]), "w").write(body)

    rng = np.random.default_rng(0)
    gold, queries, _, _ = select_v2.load_eval()
    bks = sorted(gold)[:40]
    with tempfile.TemporaryDirectory() as d:
        store = os.path.join(d, "inference_outputs")
        os.makedirs(store)
        for n, bk in enumerate(bks):
            g = gold[bk]
            grids = [g, np.zeros((2, 2), int), (g + 1) % 10, np.full(g.shape, 11)]     # 11 = illegal colour
            augs = [rng.uniform(1, 30, 8).tolist() for _ in grids]
            for v in range(6):
                res = [dict(beam_score=float(rng.uniform(0, 1.6)), score_aug=augs[k], solution=grids[k])
                       for k in rng.integers(0, len(grids), size=rng.integers(1, 4))]
                geo = ["", ".rot90", ".transpose"][v % 3]
                with bz2.BZ2File(os.path.join(store, f"{bk}{geo}.permute0123456789.ex{v}1"), "w") as f:
                    pickle.dump(res, f)

        ranker = {"feats": list(select_v2.FEATS), "mu": [0.0] * 7, "sd": [1.0] * 7,
                  "w": [0.5, 0, 0, 1, 0, 0, 2], "b": 0.0}
        outs = {}
        for strat, rk in (("kgmon", None), ("kgmon+priors", None), ("learned (LOTO-CV)", ranker)):
            nb, meta = build(strat, rk)
            assert meta["id"].endswith("/" + SLUG) and meta["code_file"] == f"{SLUG}.ipynb"
            sub = os.path.join(d, strat.replace(" ", "_").replace("|", "_"))
            os.makedirs(sub)
            materialise(nb, sub)
            for f in ("arc_decoder.py", "select_v2.py", "arc_loader.py", "arc_solver.py", "starter.py"):
                assert os.path.exists(os.path.join(sub, f)), f
                compile(open(os.path.join(sub, f)).read(), f, "exec")
            final = src(nb["cells"][-1])
            assert "run_selection_algo(SELECT_STRATEGY, ranker=SELECT_RANKER)" in final
            compile(final, "final_cell", "exec")
            ns = {}
            exec(final.split("import os")[0], ns)                       # the two SELECT_* constants
            for m in ("arc_decoder", "select_v2"):
                sys.modules.pop(m, None)
            sys.path.insert(0, sub)
            try:
                import arc_decoder
                ds = type("DS", (), {"queries": queries, "replies": {}})()
                dec = arc_decoder.ArcDecoder(ds, n_guesses=2)
                dec.load_decoded_results(store)
                outs[strat] = dec.run_selection_algo(ns["SELECT_STRATEGY"], ranker=ns["SELECT_RANKER"])
                outs[strat + "/default"] = dec.run_selection_algo()
                if strat == "kgmon":                                     # selector import failure -> production path
                    sys.modules["select_v2"] = None
                    outs["broken-import"] = dec.run_selection_algo("kgmon+priors")
            finally:
                sys.path.remove(sub)
                for m in ("arc_decoder", "select_v2"):
                    sys.modules.pop(m, None)

        def same(a, b):
            return a.keys() == b.keys() and all(
                len(a[k]) == len(b[k]) and all(np.array_equal(x, y) for x, y in zip(a[k], b[k])) for k in a)

        assert same(outs["kgmon"], outs["kgmon/default"]), "name-routed kgmon must equal production"
        assert same(outs["broken-import"], outs["kgmon/default"]), "import failure must degrade to production"
        assert not same(outs["kgmon+priors"], outs["kgmon/default"]), "priors should reorder some synthetic inputs"
        # the illegal-colour grid must never be ranked above a prior-consistent one under +priors
        for bk, order in outs["kgmon+priors"].items():
            bad = [i for i, g in enumerate(order) if (np.asarray(g) == 11).any()]
            good = [i for i, g in enumerate(order) if not (np.asarray(g) == 11).any() and np.asarray(g).shape == gold[bk].shape]
            assert not bad or not good or min(bad) > min(good), bk
        assert len(outs["learned (LOTO-CV)"]) == len(bks)
        # every other cell is untouched production
        base = json.load(open(BASE_NB))["cells"]
        nb, _ = build("kgmon")
        kept = [src(c) for c in nb["cells"] if not src(c).startswith(("%%writefile select_v2.py", "%%writefile arc_decoder.py", "SELECT_STRATEGY"))]
        orig = [src(c) for c in base if not src(c).startswith("%%writefile arc_decoder.py") and FINAL_OLD not in src(c)]
        assert kept == orig, "cells other than decoder/final must be byte-identical to production"
    print(f"selftest ok: built notebook runs the patched decoder on {len(bks)} synthetic inputs; "
          "kgmon-by-name == production, import failure degrades to production, "
          "all non-selection cells byte-identical to perfpatch")


if __name__ == "__main__":
    sys.path.insert(0, HERE)
    ap = argparse.ArgumentParser()
    ap.add_argument("--verdict")
    ap.add_argument("--strategy")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        sys.exit(0)
    ranker = None
    strategy = a.strategy
    if a.verdict:
        v = json.load(open(a.verdict))
        strategy = strategy or v.get("winner")
        ranker = v.get("ranker")
        if not strategy:
            sys.exit(f"verdict says keep kgmon (headroom {v.get('headroom')}); nothing to build. "
                     "Use --strategy to force one.")
    if not strategy:
        ap.error("give --verdict or --strategy")
    nb, meta = build(strategy, ranker)
    print("built", write(nb, meta), "strategy:", strategy, "(not pushed)")
